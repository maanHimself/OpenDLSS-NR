"""Internal ctypes binding for browser-compat-v1 on Linux SM75.

Importing this module does not load CUDA. Contexts belong to their creating host
thread and current CUDA device; explicit close or a context manager is required.
"""

from __future__ import annotations
import ctypes as C
import hashlib
import json
import math
import threading
from pathlib import Path
import numpy as np


_library_hashes = {}
_library_load_lock = threading.Lock()


def _load_library(path):
    # Context close does not unload a CDLL; glibc can retain its pathname handle.
    with _library_load_lock:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path in _library_hashes and _library_hashes[path] != digest:
            raise RuntimeError(
                "Native library changed at a loaded path; use a distinct filename or restart Python"
            )
        lib = C.CDLL(str(path))
        _library_hashes[path] = digest
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Native library changed while loading; restart Python")
        except BaseException:
            # Its loaded handle may describe either file: do not trust this path again.
            _library_hashes[path] = None
            raise
        return lib, digest


class Tensor:
    def __init__(self, owner, pointer, shape, dtype):
        self.owner = owner
        self.ptr = int(pointer)
        self.shape = tuple(int(v) for v in shape)
        self.dtype = np.dtype(dtype)
        self.size = math.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize


class Runtime:
    ABI_VERSION = 1
    POLICY = "browser-compat-v1"

    def __init__(self, library=None, policy=POLICY, *, window_impl="fused"):
        if policy != self.POLICY:
            raise ValueError("Only browser-compat-v1 is supported")
        if window_impl not in ("fused", "staged"):
            raise ValueError("window_impl must be fused or staged")
        self.window_impl = window_impl
        self._owner_thread = threading.get_ident()
        path = (
            library
            if library is not None
            else Path(__file__).resolve().parents[2] / "build" / "libnr_cuda.so"
        )
        self.library_path = Path(path).resolve(strict=True)
        if not self.library_path.is_file():
            raise ValueError("Native library must be a regular file")
        self.lib, self.library_sha256 = _load_library(self.library_path)
        try:
            self.lib.nr_abi_version.argtypes = []
            self.lib.nr_abi_version.restype = C.c_uint32
            if self.lib.nr_abi_version() != self.ABI_VERSION:
                raise RuntimeError(
                    "Native ABI version mismatch; use the matching CUDA compatibility library"
                )
            for name in ("nr_error", "nr_source_sha256", "nr_build_info"):
                fn = getattr(self.lib, name)
                fn.argtypes = []
                fn.restype = C.c_char_p
            self.source_sha256 = self.lib.nr_source_sha256().decode("ascii")
            self.build_info = json.loads(self.lib.nr_build_info().decode("utf-8"))
            if len(self.source_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.source_sha256
            ):
                raise ValueError("Invalid source hash")
            if (
                not isinstance(self.build_info, dict)
                or self.build_info.get("source_sha256") != self.source_sha256
            ):
                raise ValueError("Invalid build identity")
        except (AttributeError, UnicodeError, ValueError, TypeError) as error:
            raise RuntimeError(
                "Native library has missing or invalid ABI/build identity; use the matching CUDA compatibility library"
            ) from error
        p, i, s = C.c_void_p, C.c_int, C.c_size_t
        signatures = {
            "create": [C.POINTER(p), i, i],
            "check_context": [p],
            "allocated": [p, C.POINTER(s)],
            "destroy": [p],
            "alloc": [p, C.POINTER(p), s],
            "device_info": [p, C.POINTER(i), C.POINTER(i), C.POINTER(i)],
            "context_options": [p, C.POINTER(i), C.POINTER(i)],
            "upload": [p, p, p, s],
            "download": [p, p, p, s],
            "copy": [p, p, p, s],
            "sync": [p],
            "capture_begin": [p],
            "capture_end": [p],
            "replay": [p, C.POINTER(C.c_float)],
            "timer_start": [p],
            "timer_stop": [p, C.POINTER(C.c_float)],
            "convert": [p, p, p, s],
            "expand": [p, p, p, s],
            "gemm": [p, p, p, p, p, p, p, i, i, i, i, i, i, i, i],
            "f16gemm": [p, p, p, p, p, i, i, i],
            "pool": [p, p, p, i, i, i, i, i],
            "up": [p, p, p, p, p, p, p, i, i, i, i, i],
            "window": [p, p, p, p, p, i, i, i, i],
            "vit": [p, p, p, p, i, i],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(self.lib, "nr_" + name)
            except AttributeError as error:
                raise RuntimeError(
                    f"Native library is missing required ABI symbol nr_{name}; use the matching CUDA compatibility library"
                ) from error
            fn.argtypes = args
            fn.restype = i
        self.context = p()
        self.policy = policy
        self._scratch = {}
        self._capturing = False
        try:
            policy_id = 6
            self._check(
                self.lib.nr_create(C.byref(self.context), policy_id, int(window_impl == "fused"))
            )
            ordinal, major, minor = C.c_int(), C.c_int(), C.c_int()
            self._call("device_info", C.byref(ordinal), C.byref(major), C.byref(minor))
            if ordinal.value < 0 or (major.value, minor.value) != (7, 5):
                raise RuntimeError(
                    "Native context returned an unsupported CUDA device; require SM75"
                )
            self._device_info = {
                "ordinal": ordinal.value,
                "compute_capability": [major.value, minor.value],
            }
            actual_policy, actual_fused = C.c_int(), C.c_int()
            self._call("context_options", C.byref(actual_policy), C.byref(actual_fused))
            if actual_policy.value != policy_id or actual_fused.value != int(
                self.window_impl == "fused"
            ):
                raise RuntimeError(
                    "Native context options differ from the requested policy or window implementation"
                )
            self._context_options = {
                "policy": policy,
                "native_policy_id": policy_id,
                "abi_version": self.ABI_VERSION,
                "window_impl": "fused" if actual_fused.value else "staged",
            }
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                if hasattr(error, "add_note"):
                    error.add_note(f"Runtime cleanup also failed: {cleanup!r}")
            raise

    @property
    def device_info(self):
        """Creation-device receipt; remains readable after context close."""
        return {
            "ordinal": self._device_info["ordinal"],
            "compute_capability": list(self._device_info["compute_capability"]),
        }

    @property
    def context_options(self):
        """Resolved native options, verified against the constructor request."""
        return dict(self._context_options)

    def _ensure_open(self):
        if not self.context.value:
            raise RuntimeError("Runtime is closed")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Runtime belongs to another host thread")

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.nr_error().decode("utf-8", errors="replace"))

    def _call(self, name, *args):
        self._ensure_open()
        self._check(getattr(self.lib, "nr_" + name)(self.context, *args))

    @staticmethod
    def _ptr(t):
        return t.ptr if t is not None else None

    def _tensor(self, t, dtype=None):
        self._ensure_open()
        if not isinstance(t, Tensor) or t.owner is not self:
            raise ValueError("Tensor belongs to another runtime")
        if dtype and t.dtype != np.dtype(dtype):
            raise ValueError(f"Expected {dtype}, got {t.dtype}")
        return t.ptr

    @property
    def allocated_bytes(self):
        self._ensure_open()
        count = C.c_size_t()
        self._call("allocated", C.byref(count))
        return count.value

    def empty(self, shape, dtype="float16"):
        self._ensure_open()
        if self._capturing:
            raise RuntimeError("Allocation during graph capture")
        dtype = np.dtype(dtype)
        shape = tuple(shape)
        if (
            dtype not in (np.dtype("float16"), np.dtype("float32"))
            or not shape
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
                for v in shape
            )
        ):
            raise ValueError("Unsupported tensor shape/dtype")
        count = math.prod(int(v) for v in shape) * dtype.itemsize
        if count > 3 * 1024**3:
            raise ValueError("Tensor exceeds the 3 GiB explicit allocation limit")
        pointer = C.c_void_p()
        self._call("alloc", C.byref(pointer), count)
        return Tensor(self, pointer.value, shape, dtype)

    def upload(self, array):
        self._ensure_open()
        a = np.asarray(array)
        tensor = self.empty(a.shape, a.dtype)
        self.copy_from_host(a, tensor)
        return tensor

    def copy_from_host(self, array, destination):
        self._tensor(destination)
        if self._capturing:
            raise RuntimeError("Host copy during graph capture")
        a = np.ascontiguousarray(array, dtype=destination.dtype)
        if a.shape != destination.shape:
            raise ValueError(f"Shape {a.shape} != {destination.shape}")
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite input")
        self._call("upload", destination.ptr, a.ctypes.data, a.nbytes)

    def _copy_validated_float32(self, array, destination):
        """Internal adapter path after shape, finite and half-range validation."""
        self._tensor(destination, "float32")
        if self._capturing:
            raise RuntimeError("Host copy during graph capture")
        a = np.ascontiguousarray(array, dtype=np.float32)
        if a.shape != destination.shape:
            raise ValueError("Validated copy shape mismatch")
        self._call("upload", destination.ptr, a.ctypes.data, a.nbytes)

    def convert(self, source, destination):
        self._tensor(source, "float32")
        self._tensor(destination, "float16")
        if source.shape != destination.shape:
            raise ValueError("Conversion shape mismatch")
        self._call("convert", source.ptr, destination.ptr, source.size)

    def download(self, tensor):
        self._tensor(tensor)
        if self._capturing:
            raise RuntimeError("Host read during graph capture")
        a = np.empty(tensor.shape, tensor.dtype)
        self._call("download", a.ctypes.data, tensor.ptr, a.nbytes)
        return a

    def copy(self, source, destination):
        self._tensor(source)
        self._tensor(destination)
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("Copy shape/dtype mismatch")
        self._call("copy", destination.ptr, source.ptr, source.nbytes)

    def sync(self):
        self._call("sync")

    def capture_begin(self):
        self._ensure_open()
        if self._capturing:
            raise RuntimeError("Graph capture is already active")
        self.sync()
        self._call("capture_begin")
        self._capturing = True

    def capture_end(self):
        self._ensure_open()
        if not self._capturing:
            raise RuntimeError("No active graph capture")
        try:
            self._call("capture_end")
        finally:
            self._capturing = False

    def replay(self):
        ms = C.c_float()
        self._call("replay", C.byref(ms))
        return float(ms.value)

    def timer_start(self):
        self._call("timer_start")

    def timer_stop(self):
        ms = C.c_float()
        self._call("timer_stop", C.byref(ms))
        return float(ms.value)

    def gemm(
        self,
        a,
        w,
        rows,
        k,
        n,
        output,
        raw=None,
        residual=None,
        scale=None,
        silu=False,
        batches=1,
        broadcast=False,
        partition=0,
        quantize=True,
    ):
        for t in [a, w, output, raw, residual, scale]:
            if t is not None:
                self._tensor(t, "float16")
        if (
            a.size != rows * k * (1 if broadcast else batches)
            or w.size != batches * k * n
            or output.size != rows * n * batches
        ):
            raise ValueError("GEMM tensor sizes do not match operation")
        if raw is not None and raw.size != output.size:
            raise ValueError("Raw output size")
        if residual is not None and (
            residual.size != output.size or scale is None or scale.size != n * batches
        ):
            raise ValueError("Residual shape/scale")
        self._call(
            "gemm",
            a.ptr,
            w.ptr,
            output.ptr,
            self._ptr(raw),
            self._ptr(residual),
            self._ptr(scale),
            rows,
            k,
            n,
            batches,
            int(broadcast),
            int(silu),
            partition,
            int(quantize),
        )

    def f16gemm(self, input, weights, output, quantized=None):
        self._tensor(input, "float16")
        self._tensor(weights, "float16")
        self._tensor(output)
        k, n = weights.shape
        rows = input.size // k
        if input.size != rows * k or output.size != rows * n:
            raise ValueError("FP16 GEMM shape")
        dst = output
        if output.dtype == np.float32:
            key = ("f16-head", output.ptr)
            if key not in self._scratch:
                self._scratch[key] = self.empty(output.shape)
            dst = self._scratch[key]
        if quantized is not None:
            self._tensor(quantized, "float16")
            if quantized.size != output.size:
                raise ValueError("Quantized output shape")
        self._call("f16gemm", input.ptr, weights.ptr, dst.ptr, self._ptr(quantized), rows, k, n)
        if dst is not output:
            self._call("expand", dst.ptr, output.ptr, output.size)

    def window(self, qkv, prior, scales, out, width, height, heads, phase):
        self._tensor(qkv, "float16")
        self._tensor(prior, "float16")
        self._tensor(scales, "float32")
        self._tensor(out, "float16")
        if (
            qkv.size != width * height * heads * 96
            or out.size != width * height * heads * 32
            or prior.size != heads * 4096
            or scales.size != heads
        ):
            raise ValueError("Window shape")
        self._call(
            "window", qkv.ptr, prior.ptr, scales.ptr, out.ptr, width, height, heads, phase % 4
        )

    def vit(self, qkv, scales, out, tokens, heads):
        self._tensor(qkv, "float16")
        self._tensor(scales, "float32")
        self._tensor(out, "float16")
        if (
            qkv.size != tokens * heads * 96
            or out.size != tokens * heads * 32
            or scales.size != heads
        ):
            raise ValueError("ViT shape")
        self._call("vit", qkv.ptr, scales.ptr, out.ptr, tokens, heads)

    def pool(self, raw, out, in_width, in_height, out_width, out_height, channels):
        self._tensor(raw, "float16")
        self._tensor(out, "float16")
        if (
            raw.size != in_width * in_height * channels
            or out.size != out_width * out_height * channels
        ):
            raise ValueError("Pool shape")
        self._call("pool", raw.ptr, out.ptr, in_width, in_height, out_width, out_height, channels)

    def upsample(
        self,
        raw,
        skip,
        scale,
        out,
        raw_out=None,
        *,
        in_width,
        in_height,
        out_width,
        out_height,
        channels,
    ):
        self._up(
            raw,
            skip,
            scale,
            None,
            out,
            raw_out,
            in_width,
            in_height,
            out_width,
            out_height,
            channels,
        )

    def post_blend(
        self,
        input,
        skip,
        input_scale,
        adapter_scale,
        out,
        raw_out,
        *,
        in_width,
        in_height,
        out_width,
        out_height,
        channels,
    ):
        self._up(
            input,
            skip,
            input_scale,
            adapter_scale,
            out,
            raw_out,
            in_width,
            in_height,
            out_width,
            out_height,
            channels,
        )

    def _up(self, a, skip, s, t, out, raw, iw, ih, ow, oh, c):
        for x in [a, skip, s, t, out, raw]:
            if x is not None:
                self._tensor(x, "float16")
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
            for v in (iw, ih, ow, oh, c)
        ):
            raise ValueError("Upsample dimensions must be positive integers")
        if (
            a.size != iw * ih * c
            or skip.size != ow * oh * c
            or out.size != ow * oh * c
            or s.size != c
            or (t is not None and t.size != c)
            or (raw is not None and raw.size != out.size)
            or (ow - 1) // 2 >= iw
            or (oh - 1) // 2 >= ih
        ):
            raise ValueError("Upsample shape")
        self._call(
            "up",
            a.ptr,
            skip.ptr,
            s.ptr,
            self._ptr(t),
            out.ptr,
            self._ptr(raw),
            iw,
            ow,
            oh,
            c,
            int(t is not None),
        )

    def close(self):
        if self.context.value:
            # Reject ownership mistakes before detaching a recoverable pointer.
            self._ensure_open()
            self._check(self.lib.nr_check_context(self.context))
            # Once ownership is checked, destroy consumes the context when it
            # reports a cleanup error. Detach first so no later call or retry
            # can use a freed pointer; destroy also ends an active capture.
            context = self.context
            self.context = C.c_void_p()
            self._capturing = False
            self._scratch.clear()
            self._check(self.lib.nr_destroy(context))

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            if hasattr(error, "add_note"):
                error.add_note(f"Runtime cleanup also failed: {cleanup!r}")
        return False
