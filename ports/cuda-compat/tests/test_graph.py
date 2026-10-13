"""Graph wiring/lifetime validation without GPU allocations or model files."""
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda.export_model import array_specs
from opendlss_cuda.graph import NativeGraph


@dataclass
class Tensor:
    shape: tuple
    dtype: str = "float16"
    initialized: bool = False

    @property
    def nbytes(self):
        count = 1
        for dim in self.shape:
            count *= dim
        return count * (4 if self.dtype == "float32" else 2)


class FakeModel:
    def __init__(self):
        self.arrays = {s["name"]: Tensor(tuple(s["shape"]), "float32" if s["kind"] == "float32" else "float16", True)
                       for s in array_specs()}

    def upload(self, runtime):
        return {name: runtime.upload(a) for name, a in self.arrays.items()}


class CheckingRuntime:
    def __init__(self):
        self.allocations = 0
        self.gemms = []
        self.attention = []

    def empty(self, shape, dtype="float16"):
        self.allocations += 1
        return Tensor(tuple(shape), dtype)

    def upload(self, array):
        return Tensor(array.shape, array.dtype, True)

    def read(self, *tensors):
        for t in tensors:
            assert t.initialized, f"read before write: {t.shape}"

    def copy(self, source, destination):
        self.read(source)
        assert source.shape == destination.shape
        assert source is not destination
        destination.initialized = True

    def f16gemm(self, source, weights, output, quantized=None):
        self.read(source, weights)
        assert source.shape[1] == weights.shape[0]
        assert output.shape == (source.shape[0], weights.shape[1])
        assert source is not output
        output.initialized = True
        if quantized is not None:
            assert quantized.shape == output.shape
            quantized.initialized = True

    def gemm(self, source, weights, *, rows, k, n, output, raw=None, residual=None, scale=None,
             silu=False, batches=1, broadcast=False, partition=0, quantize=True):
        self.read(source, weights)
        assert source.shape == (rows, k if broadcast else k * batches), (source.shape, rows, k, batches, broadcast)
        assert output.shape == (rows, n * batches)
        assert output is not source
        if residual is not None:
            self.read(residual, scale)
            assert residual.shape == output.shape
            assert scale.shape == (n * batches,)
        if raw is not None:
            assert raw.shape == output.shape
            raw.initialized = True
        output.initialized = True
        self.gemms.append(dict(k=k, n=n, batches=batches, broadcast=broadcast, partition=partition, quantize=quantize, silu=silu))

    def window(self, qkv, prior, scales, out, *, width, height, heads, phase):
        self.read(qkv, prior, scales)
        assert qkv.shape == (width * height, heads * 96)
        assert prior.shape == (heads, 64, 64)
        assert scales.shape == (heads,)
        assert out.shape == (width * height, heads * 32)
        assert phase in (0, 1, 2, 3)
        out.initialized = True
        self.attention.append(("window", phase, heads))

    def vit(self, qkv, scales, out, *, tokens, heads):
        self.read(qkv, scales)
        assert qkv.shape == (tokens, heads * 96)
        assert scales.shape == (heads,)
        assert out.shape == (tokens, heads * 32)
        out.initialized = True
        self.attention.append(("global", None, heads))

    def pool(self, source, output, *, in_width, in_height, out_width, out_height, channels):
        self.read(source)
        assert source.shape == (in_width * in_height, channels)
        assert output.shape == (out_width * out_height, channels)
        output.initialized = True

    def upsample(self, source, skip, scale, output, *, raw_out=None,
                 in_width, in_height, out_width, out_height, channels):
        self.read(source, skip, scale)
        assert source.shape == (in_width * in_height, channels)
        assert output.shape == skip.shape == (out_width * out_height, channels)
        assert scale.shape == (channels,)
        output.initialized = True
        if raw_out is not None:
            assert raw_out.shape == output.shape
            raw_out.initialized = True

    def post_blend(self, source, skip, input_scale, adapter_scale, output, raw_out, **kwargs):
        self.read(adapter_scale)
        self.upsample(source, skip, input_scale, output, raw_out=raw_out, **kwargs)


class GraphTests(unittest.TestCase):
    def test_supported_720p_field_and_rejected_larger_graph(self):
        from unittest.mock import Mock
        graph = NativeGraph(CheckingRuntime(), FakeModel(), 1280, 720, capture_boundaries=True)
        self.assertEqual((graph.geometry["full_width"], graph.geometry["full_height"]), (1344, 768))
        self.assertEqual(graph.geometry["padded_vit_tokens"], 320)
        self.assertLess(graph.allocated_bytes, 4 * 1024 ** 3)
        expected = {f"block-{i}" for i in range(70)}
        expected.update(f"transition-{i}-{i + 1}" for i in (0, 4, 8, 14, 22))
        self.assertEqual(set(graph.boundaries), expected)
        model = FakeModel()
        model.upload = Mock()
        with self.assertRaisesRegex(ValueError, "Supported inference extent"):
            NativeGraph(CheckingRuntime(), model, 1920, 1080)
        model.upload.assert_not_called()

    def test_complete_graph_shapes_initialization_and_memory(self):
        for mode in ("upstream", "legacy_belvedere"):
            rt = CheckingRuntime()
            graph = NativeGraph(rt, FakeModel(), 1280, 720, mode)
            initial_allocations = rt.allocations
            graph.features.initialized = True
            graph.record()
            self.assertTrue(graph.head.initialized)
            self.assertEqual(len(rt.attention), 70)  # block 39 is merge-only
            self.assertEqual(sum(a[0] == "global" for a in rt.attention), 8)
            self.assertEqual(rt.allocations, initial_allocations)
            self.assertLess(graph.allocated_bytes, 4 * 1024 ** 3)
            self.assertEqual(set(graph.skips), {0, 4, 8, 14, 22, 30})
            self.assertEqual(sum(g["partition"] == 1024 for g in rt.gemms), 8)
            self.assertEqual(sum(g["partition"] == 512 for g in rt.gemms), 8)
            self.assertEqual(sum(g["partition"] == 256 for g in rt.gemms), 9)
            graph.record()  # buffer reuse must preserve valid shapes and ordering
            self.assertEqual(rt.allocations, initial_allocations)

    def test_optional_full_boundary_capture(self):
        for captures, expected in ((False, 0), (True, 75)):
            graph = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, capture_boundaries=captures)
            graph.features.initialized = True
            graph.record()
            self.assertEqual(len(graph.boundaries), expected)
            self.assertTrue(all(t.initialized for t in graph.boundaries.values()))
        for capture in (None, 1, [], "all"):
            with self.subTest(capture=capture), self.assertRaisesRegex(ValueError, "must be a bool"):
                NativeGraph(CheckingRuntime(), FakeModel(), 320, 320, capture_boundaries=capture)

    def test_raw_residual_paths(self):
        graph = NativeGraph(CheckingRuntime(), FakeModel(), 320, 320)
        prologues = {}
        for name, (_, _, kwargs) in zip(graph.operation_names, graph.ops):
            if name in ("block0.w2", "block66.w2", "block70.w2"):
                prologues[name] = kwargs["residual"]
        self.assertIs(prologues["block0.w2"], graph.buffers["level6.input_raw"])
        self.assertIs(prologues["block66.w2"], graph.buffers["level0.input_raw"])
        self.assertIs(prologues["block70.w2"], graph.buffers["level6.input_raw"])


if __name__ == "__main__":
    unittest.main()
