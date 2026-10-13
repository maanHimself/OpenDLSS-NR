"""Public feature contract and failure cleanup without loading CUDA."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda import backend as module


class Model:
    def __init__(self, path):
        self.closed = False

    def close(self):
        self.closed = True


class Tensor:
    def __init__(self, shape, dtype="float16"):
        self.shape = shape
        self.dtype = np.dtype(dtype)


class Graph:
    def __init__(self, runtime, model, width, height, *, mode, capture_boundaries):
        self.geometry = module.geometry_from_valid(width, height, mode)
        rows = self.geometry["full_rows"]
        self.features = Tensor((rows, 16))
        self.head = Tensor((rows, 4), "float32")
        self.boundaries = {"block-0": Tensor((rows, 32))} if capture_boundaries else {}
        self.records = 0

    def record(self):
        self.records += 1


class Runtime:
    def __init__(self, library, *, policy, window_impl):
        self.context_options = {"policy": policy, "native_policy_id": 6, "window_impl": window_impl}
        self.device_info = {"ordinal": 0, "compute_capability": [7, 5]}
        self.window_impl = window_impl
        self.owner_thread = threading.get_ident()
        self.context = SimpleNamespace(value=1)
        self.closed = False
        self.copies = self.replays = 0
        self.allocated_bytes = 1234
        self.nonfinite = False
        self.fail_capture = False

    def _call(self, name):
        assert name == "check_context"
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("another host thread")

    def empty(self, shape, dtype): return Tensor(shape, dtype)
    def _copy_validated_float32(self, array, destination):
        assert array.shape == destination.shape
        self.copies += 1
    def convert(self, *args): pass
    def sync(self): pass
    def capture_begin(self):
        if self.fail_capture:
            raise RuntimeError("capture failed")
    def capture_end(self): pass
    def replay(self):
        self.replays += 1
        return 1.25
    def download(self, tensor):
        result = np.zeros(tensor.shape, tensor.dtype)
        if self.nonfinite:
            result.flat[0] = np.nan
        return result
    def close(self):
        self._call("check_context")
        self.closed = True
        self.context.value = 0


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.object(module, "Runtime", Runtime), patch.object(module, "CanonicalModel", Model),
                        patch.object(module, "NativeGraph", Graph)]
        for item in self.patches: item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])

    def make(self, **kwargs):
        return module.Backend("model", "library", 320, 320, **kwargs)

    def test_default_policy_and_explicit_geometry(self):
        with self.make() as backend:
            self.assertEqual(backend.context_options["policy"], "browser-compat-v1")
            self.assertEqual(backend.window_impl, "fused")
            self.assertEqual(backend.geometry["mode"], "upstream")
        with self.make(geometry="legacy_belvedere", window_impl="staged") as backend:
            self.assertEqual(backend.geometry["mode"], "legacy_belvedere")
            self.assertEqual(backend.window_impl, "staged")

    def test_invalid_options_and_extent_fail_before_model_or_cuda(self):
        with patch.object(module, "Runtime") as runtime, patch.object(module, "CanonicalModel") as model:
            for kwargs in ({"policy": "f13browser"}, {"policy": "fp32browser"}, {"window_impl": "typo"},
                           {"capture_boundaries": 1}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.make(**kwargs)
            with self.assertRaises(ValueError): module.Backend("m", "l", 1920, 1080)
            runtime.assert_not_called()
            model.assert_not_called()

    def test_invalid_model_fails_before_cuda(self):
        with patch.object(module, "CanonicalModel", side_effect=ValueError("bad cache")), patch.object(module, "Runtime") as rt:
            with self.assertRaisesRegex(ValueError, "bad cache"): self.make()
            rt.assert_not_called()

    def test_features_reject_dtype_shape_nonfinite_and_half_overflow(self):
        with self.make() as backend:
            for array in (np.zeros((320, 320, 16), np.float64), np.zeros((320, 320, 15), np.float32),
                          np.full((320, 320, 16), np.inf, np.float32),
                          np.full((320, 320, 16), 65505, np.float32),
                          np.full((320, 320, 16), -65505, np.float32)):
                with self.assertRaises(ValueError): backend.run_features(array)
            self.assertEqual(backend.runtime.copies, 0)
            self.assertFalse(backend.closed)

    def test_capture_once_repeat_and_noncontiguous_inputs(self):
        with self.make(capture_boundaries=True) as backend:
            with self.assertRaisesRegex(RuntimeError, "No completed"): backend.read_boundaries()
            features = np.zeros((320, 320, 16), np.float32)[:, ::-1]
            features[0, 0, 0] = 65504
            features[0, 0, 1] = -65504
            first = backend.run_features(features)
            second = backend.run_features(features)
            self.assertEqual(first.shape, (320, 320, 4))
            self.assertEqual(first.dtype, np.dtype("float32"))
            self.assertEqual(backend.graph.records, 2)
            self.assertEqual(backend.runtime.replays, 2)
            self.assertFalse(np.shares_memory(first, second))
            self.assertEqual(backend.read_boundaries()["block-0"].shape, (102400, 32))
            self.assertEqual(backend.last_metrics["initialization_seconds"], 0)
        self.assertTrue(backend.runtime.closed)
        self.assertTrue(backend.model.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"): backend.run_features(features)
        with self.assertRaisesRegex(RuntimeError, "closed"): backend.read_boundaries()

    def test_nonfinite_head_and_capture_failure_close_context_and_model(self):
        for field in ("nonfinite", "fail_capture"):
            backend = self.make()
            setattr(backend.runtime, field, True)
            with self.assertRaises(RuntimeError): backend.run_features(np.zeros((320, 320, 16), np.float32))
            self.assertTrue(backend.closed)
            self.assertTrue(backend.runtime.closed)
            self.assertTrue(backend.model.closed)
            self.assertFalse(backend.has_result)

    def test_constructor_allocation_failure_closes_context_and_model(self):
        runtime = Runtime("lib", policy="browser-compat-v1", window_impl="fused")
        model = Model("model")
        with patch.object(module, "Runtime", return_value=runtime), patch.object(module, "CanonicalModel", return_value=model), \
                patch.object(module, "NativeGraph", side_effect=RuntimeError("allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"): self.make()
        self.assertTrue(runtime.closed)
        self.assertTrue(model.closed)

    def test_wrong_thread_does_not_consume_or_invalidate_owned_context(self):
        backend = self.make()
        failures = []
        def foreign_calls():
            for call in (backend.close, lambda: backend.run_features(np.zeros((320, 320, 16), np.float32))):
                try: call()
                except RuntimeError as error: failures.append(str(error))
        thread = threading.Thread(target=foreign_calls)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["another host thread"] * 2)
        self.assertFalse(backend.closed)
        self.assertFalse(backend.failed)
        self.assertFalse(backend.model.closed)
        self.assertEqual(backend.runtime.context.value, 1)
        backend.run_features(np.zeros((320, 320, 16), np.float32))
        backend.close()
        self.assertTrue(backend.closed)
        self.assertTrue(backend.model.closed)
        self.assertEqual(backend.runtime.context.value, 0)

    def test_consumed_native_cleanup_failure_still_releases_model(self):
        backend = self.make()
        def consume_then_fail():
            backend.runtime.context.value = 0
            raise RuntimeError("destroy cleanup failed")
        backend.runtime.close = consume_then_fail
        with self.assertRaisesRegex(RuntimeError, "destroy cleanup failed"): backend.close()
        self.assertTrue(backend.closed)
        self.assertTrue(backend.model.closed)
        backend.close()  # already consumed; no native retry


if __name__ == "__main__": unittest.main()
