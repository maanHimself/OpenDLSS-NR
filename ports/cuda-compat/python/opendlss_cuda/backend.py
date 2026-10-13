"""Fixed-geometry, feature-to-head inference with the browser compatibility policy."""
from __future__ import annotations

import time

import numpy as np

from .geometry import geometry_from_valid, require_supported_geometry
from .graph import NativeGraph
from .model import CanonicalModel
from .runtime import Runtime


POLICY = "browser-compat-v1"


class Backend:
    """Load one model and reuse one CUDA graph for finite float32 features.

    Features and returned heads cover the complete padded field, not only the
    valid image rectangle. A head contains residual RGB and a blend logit;
    composition and temporal history belong to the caller.
    """

    def __init__(self, model, library, width, height, *, geometry="upstream",
                 policy=POLICY, window_impl="fused", capture_boundaries=False):
        if policy != POLICY:
            raise ValueError(f"Only {POLICY!r} is supported")
        if window_impl not in ("fused", "staged"):
            raise ValueError("window_impl must be fused or staged")
        if type(capture_boundaries) is not bool:
            raise ValueError("capture_boundaries must be a bool")
        self.geometry = require_supported_geometry(geometry_from_valid(width, height, geometry))
        # Reject invalid geometry/model files before initializing CUDA.
        self.model = CanonicalModel(model)
        self.closed = False
        self.failed = False
        self.runtime = None
        self.captured = False
        self.has_result = False
        self.last_metrics = {}
        self.width, self.height = width, height
        try:
            self.runtime = Runtime(library, policy=policy, window_impl=window_impl)
            self.graph = NativeGraph(self.runtime, self.model, width, height,
                                     mode=geometry, capture_boundaries=capture_boundaries)
            self.input_float32 = self.runtime.empty(self.graph.features.shape, "float32")
        except BaseException as error:
            self._close_after_error(error)
            raise

    @property
    def device_info(self):
        return self.runtime.device_info

    @property
    def window_impl(self):
        return self.runtime.window_impl

    @property
    def context_options(self):
        return self.runtime.context_options

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("Backend is closed")
        if self.failed:
            raise RuntimeError("Backend invalidated by earlier failure; create a new instance")
        # Reject ownership mistakes before an inference failure could invalidate
        # an otherwise usable context. This checks both host thread and device.
        self.runtime._call("check_context")

    def _close_after_error(self, error):
        self.failed = True
        try:
            self.close()
        except BaseException as cleanup:
            if hasattr(error, "add_note"):
                error.add_note(f"Backend cleanup also failed: {cleanup!r}")

    def run_features(self, features):
        self._ensure_open()
        started = time.perf_counter()
        data = np.asarray(features)
        g = self.geometry
        shape = (g["full_height"], g["full_width"], 16)
        if data.dtype != np.dtype("float32") or data.shape != shape or not np.isfinite(data).all():
            raise ValueError("Expected finite float32 features at the exact configured field extent")
        if data.max() > 65504 or data.min() < -65504:
            raise ValueError("Feature exceeds finite FP16 range")
        try:
            self.runtime._copy_validated_float32(data.reshape(-1, 16), self.input_float32)
            uploaded = time.perf_counter()
            initialization = 0.0
            if not self.captured:
                t = time.perf_counter()
                self.runtime.convert(self.input_float32, self.graph.features)
                self.graph.record()
                self.runtime.sync()
                self.runtime.capture_begin()
                self.runtime.convert(self.input_float32, self.graph.features)
                self.graph.record()
                self.runtime.capture_end()
                self.captured = True
                initialization = time.perf_counter() - t
            gpu_ms = self.runtime.replay()
            read_start = time.perf_counter()
            head = self.runtime.download(self.graph.head).reshape(g["full_height"], g["full_width"], 4)
            if head.dtype != np.dtype("float32") or not np.isfinite(head).all():
                raise RuntimeError("Expected finite float32 head output")
            self.last_metrics = {
                "network_seconds": time.perf_counter() - started,
                "upload_seconds": uploaded - started,
                "readback_seconds": time.perf_counter() - read_start,
                "gpu_ms": gpu_ms,
                "initialization_seconds": initialization,
                "explicit_device_bytes": self.runtime.allocated_bytes,
            }
            self.has_result = True
            return head
        except BaseException as error:
            self._close_after_error(error)
            raise

    def read_boundaries(self):
        """Download the captured boundaries of the most recent completed call."""
        self._ensure_open()
        if not self.has_result:
            raise RuntimeError("No completed inference is available")
        if not self.graph.boundaries:
            raise RuntimeError("Boundary capture was not enabled")
        try:
            result = {name: self.runtime.download(tensor) for name, tensor in self.graph.boundaries.items()}
            if any(not np.isfinite(array).all() for array in result.values()):
                raise RuntimeError("Nonfinite boundary output")
            return result
        except BaseException as error:
            self._close_after_error(error)
            raise

    def close(self):
        if not self.closed:
            try:
                if self.runtime is not None:
                    self.runtime.close()
            finally:
                # Runtime retains its pointer when thread/device ownership is
                # rejected, but consumes it on a genuine destruction failure.
                # Preserve the former so its owner can still close it safely.
                if self.runtime is None or not self.runtime.context.value:
                    self.closed = True
                    self.model.close()

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
                error.add_note(f"Backend cleanup also failed: {cleanup!r}")
        return False
