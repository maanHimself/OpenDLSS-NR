"""Complete upstream 71-block graph over the native Runtime operator interface.

Build once, upload half-rounded feature input, then record/capture/replay with
the same persistent buffers. Runtime owns allocations and CUDA Graph lifetime.
"""
from __future__ import annotations

from . import geometry as _geometry
from .geometry import geometry_from_valid, require_supported_geometry


class NativeGraph:
    """Fixed geometry graph with optional capture of all 75 published boundaries.

    Boundary capture uses additional VRAM. Warm `record()` once
    before CUDA Graph capture so Runtime can prepare its operator scratch space.
    Subsequent records have no host uploads or graph-owned allocations.
    """
    def __init__(self, runtime, model, width, height, mode="upstream", *,
                 full_width=None, full_height=None, capture_boundaries=False):
        if type(capture_boundaries) is not bool:
            raise ValueError("capture_boundaries must be a bool")
        self.runtime, self.model = runtime, model
        self.geometry = geometry_from_valid(width, height, mode, full_width=full_width, full_height=full_height)
        require_supported_geometry(self.geometry)
        self.weights = model.upload(runtime)
        self.ops = []
        self.operation_names = []
        self.buffers = {}
        self.boundaries = {}
        self.allocated_bytes = sum(a.nbytes for a in model.arrays.values())
        self._capture_boundaries = capture_boundaries
        self.features = self._alloc("features", (self.geometry["full_rows"], 16))
        self.head = self._alloc("head", (self.geometry["full_rows"], 4), "float32")
        self._build()

    def _alloc(self, name, shape, dtype="float16"):
        if name not in self.buffers:
            self.buffers[name] = self.runtime.empty(shape, dtype=dtype)
            count = 1
            for value in shape:
                count *= value
            self.allocated_bytes += count * (4 if dtype == "float32" else 2)
        else:
            tensor = self.buffers[name]
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"buffer shape reused incorrectly: {name}")
        return self.buffers[name]

    def _op(self, name, method, *args, **kwargs):
        self.operation_names.append(name)
        self.ops.append((method, args, kwargs))

    def record(self):
        for method, args, kwargs in self.ops:
            method(*args, **kwargs)

    def _size(self, level):
        if level == 6:
            return self.geometry["full_width"], self.geometry["full_height"]
        item = self.geometry["levels"][level]
        return item["width"], item["height"]

    def _at(self, level, name, channels):
        w, h = self._size(level)
        return self._alloc(f"level{level}.{name}", (w * h, channels))

    def _weight(self, block, name):
        return self.weights[f"block{block}.{name}"]

    def _capture(self, name, tensor):
        if not self._capture_boundaries:
            return
        target = self._alloc("capture." + name, tuple(tensor.shape))
        self.boundaries[name] = target
        self._op("capture " + name, self.runtime.copy, tensor, target)

    def _gemm(self, block, weight, source, target, *, raw=None, residual=None,
              scale=None, silu=False, broadcast=False, partition=0, quantize=True):
        w = self._weight(block, weight)
        shape = self.model.arrays[f"block{block}.{weight}"].shape
        batches = shape[0] if len(shape) == 3 else 1
        self._op(f"block{block}.{weight}", self.runtime.gemm, source, w,
                 rows=source.shape[0], k=shape[-2], n=shape[-1], output=target,
                 raw=raw, residual=residual,
                 scale=self._weight(block, scale) if scale else None,
                 silu=silu, batches=batches, broadcast=broadcast,
                 partition=partition, quantize=quantize)

    def _attention_block(self, b, state, *, residual_override=None, raw_output=None):
        block, c, level = b["block"], b["channels"], b["level"]
        w, h = self._size(level)
        output_a = self._at(level, "state_a", c)
        output_b = self._at(level, "state_b", c)
        output = output_b if state is output_a else output_a
        ffn = self._at(level, "ffn", c)
        ffn_raw = self._at(level, "ffn_raw", c) if c == 32 else None
        residual = residual_override if residual_override is not None else state
        if c == 512:
            branch = self._at(level, "branch", 512)
            expanded = self._at(level, "expanded", 2048)
            narrow = self._at(level, "narrow", 512)
            self._gemm(block, "w1", state, branch)
            self._gemm(block, "w2", branch, expanded, silu=True)
            self._gemm(block, "w3", expanded, narrow)
            self._gemm(block, "contract", narrow, ffn, residual=residual, scale="ffn_skip")
        elif c in (64, 128, 256):
            expanded = self._at(level, "expanded", (c // 32) * 128)
            narrow = self._at(level, "narrow", c)
            self._gemm(block, "w1", state, expanded, silu=True, broadcast=True)
            self._gemm(block, "w2", expanded, narrow)
            self._gemm(block, "w3", narrow, ffn, residual=residual, scale="ffn_skip")
        else:
            expanded = self._at(level, "expanded", 4096 if c == 1024 else 128)
            self._gemm(block, "w1", state, expanded, silu=True)
            self._gemm(block, "w2", expanded, ffn, raw=ffn_raw,
                       residual=residual, scale="ffn_skip", partition=1024 if c == 1024 else 0)
        qkv = self._at(level, "qkv", c * 3)
        attended = self._at(level, "attended", c)
        self._gemm(block, "qkv", ffn, qkv, quantize=False, partition=512 if c == 1024 else 0)
        if c == 1024:
            self._op(f"block{block}.attention", self.runtime.vit, qkv,
                     self._weight(block, "attention_scale"), attended, tokens=w * h, heads=c // 32)
        else:
            self._op(f"block{block}.attention", self.runtime.window, qkv, self._weight(block, "prior"),
                     self._weight(block, "attention_scale"), attended, width=w, height=h,
                     heads=c // 32, phase=b["phase"])
        self._gemm(block, "projection", attended, output, raw=raw_output,
                   residual=ffn_raw if c == 32 else ffn, scale="attn_skip", partition=256 if c == 1024 else 0)
        return output

    def _build(self):
        schedule = _geometry.block_schedule()
        self.skips = {}
        state = None
        for b in schedule:
            block, c, level = b["block"], b["channels"], b["level"]
            residual_override = None
            w, h = self._size(level)
            if block == 0:
                state = self._at(6, "state_a", 32)
                residual_override = self._at(6, "input_raw", 32)
                self._op("input adapter", self.runtime.f16gemm, self.features,
                         self._weight(0, "input_adapter"), residual_override, quantized=state)
            elif "before" in b:
                before = b["before"]
                iw, ih = self._size(before["input_level"])
                target = self._at(level, "state_a", c)
                if block == 70:
                    residual_override = self._at(level, "input_raw", 32)
                    self._op("post blend", self.runtime.post_blend, state, self.skips[0],
                             self._weight(70, "input_scale"), self._weight(70, "adapter_scale"),
                             target, residual_override, in_width=iw, in_height=ih,
                             out_width=w, out_height=h, channels=c)
                else:
                    projected = self._alloc(f"transition{block}.projection", (iw * ih, c))
                    self._gemm(block, "up_projection", state, projected, quantize=False,
                               partition=256 if block == 39 else 0)
                    residual_override = self._at(level, "input_raw", c) if c == 32 else None
                    self._op(f"upsample merge {block}", self.runtime.upsample, projected,
                             self.skips[before["skip_block"]], self._weight(block, "merge_scale"), target,
                             raw_out=residual_override, in_width=iw, in_height=ih,
                             out_width=w, out_height=h, channels=c)
                state = target
            raw = self._at(level, "output_raw", c) if "after" in b or block == 70 else None
            if b["type"] != "merge":
                state = self._attention_block(b, state, residual_override=residual_override, raw_output=raw)
            if block <= 69:
                self._capture(f"block-{block}", state)
            if "after" in b:
                after = b["after"]
                skip = self._at(level, "skip", c)
                self._op(f"retain skip {block}", self.runtime.copy, state, skip)
                self.skips[block] = skip
                ow, oh = self._size(after["level"])
                if after["operation"] == "pool":
                    pooled = self._at(after["level"], "state_a", c)
                else:
                    pooled = self._alloc(f"transition{block}.pool", (ow * oh, c))
                self._op(f"pool {block}", self.runtime.pool, raw, pooled, in_width=w, in_height=h,
                         out_width=ow, out_height=oh, channels=c)
                if after["operation"] == "pool_project":
                    state = self._at(after["level"], "state_a", after["channels"])
                    self._gemm(block, "down_projection", pooled, state)
                else:
                    state = pooled
                if block in (0, 4, 8, 14, 22):
                    self._capture(f"transition-{block}-{block + 1}", state)
            if block == 70:
                self._op("head", self.runtime.f16gemm, raw, self._weight(70, "head"), self.head)
