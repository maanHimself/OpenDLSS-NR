"""CPU-only parity contract tests; no CUDA library is loaded."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda import fixture as f
from opendlss_cuda.geometry import geometry_from_valid


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.geometry = geometry_from_valid(320, 320)
        cls.shapes = f.required_boundary_shapes(cls.geometry)
        cls.model = cls.root / "model"
        cls.model.mkdir()
        (cls.model / "weights.bin").write_bytes(b"synthetic test model")
        blob = {"file": "weights.bin", "nbytes": 20, "sha256": f.file_sha256(cls.model / "weights.bin")}
        model = {"source": {"manifest_sha256": "1" * 64}, "blob": blob}
        (cls.model / "manifest.json").write_text(json.dumps(model), encoding="utf8")
        cls.library = cls.root / "fake-library.so"
        cls.library.write_bytes(b"CPU fake only")
        cls.base = {
            "format": f.FORMAT, "availability": "available", "checks": ["head", "all-75-boundaries"],
            "reference": {"domain": f.POLICY, "upstream_commit": "a" * 40,
                          "capture_manifest_sha256": "b" * 64, "producer": "CPU synthetic test",
                          "producer_sha256": "c" * 64, "device": "fake", "driver": "fake",
                          "browser_version": "fake", "browser_binary_sha256": "d" * 64,
                          "shader_source_sha256": "e" * 64},
            "model": {"source_manifest_sha256": "1" * 64,
                      "canonical_manifest_sha256": f.file_sha256(cls.model / "manifest.json"),
                      "canonical_blob_sha256": blob["sha256"]},
            "geometry": {key: cls.geometry[key] for key in ("mode", "valid_width", "valid_height", "full_width", "full_height")},
            "features": cls.tensor("features.npy", np.zeros((320, 320, 16), "<f4"), "f32-npy"),
            "head": cls.tensor("head.npy", np.zeros((320, 320, 4), "<f4"), "f32-npy"),
            "boundaries": {name: cls.tensor(name + ".bin", np.zeros(shape, np.uint8), "e4")
                           for name, shape in cls.shapes.items()},
        }

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def tensor(cls, name, value, fmt):
        path = cls.root / name
        if fmt == "f32-npy":
            np.save(path, value, allow_pickle=False)
        else:
            path.write_bytes(value.tobytes())
        return {"file": name, "file_sha256": f.file_sha256(path), "bytes": path.stat().st_size,
                "array_sha256": f.array_sha256(value), "format": fmt, "shape": list(value.shape)}

    def setUp(self):
        self.metadata = deepcopy(self.base)
        self.instances = []

    def manifest(self):
        path = self.root / "fixture.json"
        path.write_text(json.dumps(self.metadata), encoding="utf8")
        return path

    def factory(self, mutate=None, device=None, missing=False, fail_close=False):
        parent = self

        class Fake:
            def __init__(self, model, library, width, height, **kwargs):
                self.capture = kwargs["capture_boundaries"]
                self.context_options = {"policy": f.POLICY, "window_impl": "fused"}
                self.device_info = device or {"ordinal": 0, "compute_capability": [7, 5]}
                self.graph = SimpleNamespace(geometry=parent.geometry)
                self.model = SimpleNamespace(identity={
                    "manifest_sha256": parent.base["model"]["canonical_manifest_sha256"],
                    "blob_sha256": parent.base["model"]["canonical_blob_sha256"],
                    "source_manifest_sha256": parent.base["model"]["source_manifest_sha256"]})
                self.runtime = SimpleNamespace(library_sha256=f.file_sha256(library),
                                               source_sha256="f" * 64, build_info={"test": True})
                self.calls = 0
                self.closed = False
                parent.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True
                if fail_close:
                    raise RuntimeError("cleanup failure")

            def run_features(self, features):
                self.calls += 1
                result = np.zeros((320, 320, 4), "<f4")
                if mutate:
                    mutate(self, result)
                return result

            def read_boundaries(self):
                result = {name: np.zeros(shape, "<f2") for name, shape in parent.shapes.items()}
                if missing:
                    del result["block-0"]
                if mutate:
                    mutate(self, result)
                return result
        return Fake

    def parity(self, **kwargs):
        return f.run_parity(fixture=self.manifest(), model=self.model, library=self.library,
                            backend_factory=kwargs.pop("backend_factory", self.factory()), **kwargs)

    def test_complete_pass_three_production_one_instrumented(self):
        result = self.parity()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["original_vulkan_parity"], "unavailable")
        self.assertEqual(len(result["boundaries"]), 75)
        self.assertEqual([x.calls for x in self.instances], [3, 1])
        self.assertTrue(all(x.closed for x in self.instances))

    def symlink(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except OSError:
            self.skipTest("Symlink creation unavailable")

    def test_selected_roots_and_ancestors_can_be_symlinked(self):
        self.manifest()
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "fixture"
            self.symlink(alias, self.root)
            self.assertEqual(f.load_fixture(alias).path, self.manifest().resolve())
            self.assertEqual(f.load_fixture(alias / "fixture.json").geometry, self.geometry)
            self.assertEqual(f._model_identity(alias / "model", self.base["model"]), self.base["model"])
            model_alias = Path(directory) / "model"
            self.symlink(model_alias, self.model)
            self.assertEqual(f._model_identity(model_alias, self.base["model"]), self.base["model"])

    def test_child_file_and_directory_symlinks_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            child = Path(directory)
            self.symlink(child / "features.npy", self.root / "features.npy")
            self.symlink(child / "nested", self.root)
            for name in (f"{child.name}/features.npy", f"{child.name}/nested/features.npy"):
                self.metadata["features"]["file"] = name
                with self.assertRaises(ValueError):
                    self.parity()
            self.assertEqual(self.instances, [])

    def test_manifest_and_model_child_symlinks_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            child = Path(directory)
            self.symlink(child / "fixture.json", self.manifest())
            with self.assertRaises(ValueError):
                f.load_fixture(child)
            self.symlink(child / "manifest.json", self.model / "manifest.json")
            with self.assertRaises(ValueError):
                f._model_identity(child, self.base["model"])
            self.symlink(child / "weights.bin", self.model / "weights.bin")
            with self.assertRaises(ValueError):
                f._file(child, "weights.bin")

    def test_head_signed_zero_is_mismatch(self):
        def mutate(_, value):
            if isinstance(value, np.ndarray):
                value.flat[0] = -0.0
        result = self.parity(backend_factory=self.factory(mutate))
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["production_heads"][0]["mismatches"], 1)

    def test_production_replay_and_instrumented_are_separate_checks(self):
        def mutate(backend, value):
            if isinstance(value, np.ndarray) and (backend.capture or backend.calls == 2):
                value.flat[0] = 1
        result = self.parity(backend_factory=self.factory(mutate))
        self.assertFalse(result["deterministic"])
        self.assertFalse(result["instrumented_head_matches_production"]["exact"])

    def test_boundary_nonrepresentable_is_numerical_mismatch(self):
        def mutate(_, value):
            if isinstance(value, dict):
                value["block-0"].flat[0] = np.float16(.1234)
        result = self.parity(backend_factory=self.factory(mutate))
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["boundaries"][0]["representable_as_e4"])

    def test_original_nan_code_cannot_pass_via_zero_decode(self):
        self.metadata["reference"] = {k: v for k, v in self.metadata["reference"].items()
                                      if k not in ("browser_version", "browser_binary_sha256", "shader_source_sha256")}
        self.metadata["reference"]["domain"] = "original-vulkan"
        value = np.zeros(self.shapes["block-0"], np.uint8)
        value.flat[0] = 127
        value.flat[1] = 255
        self.metadata["boundaries"]["block-0"] = self.tensor("nan-code.bin", value, "e4")
        result = self.parity()
        self.assertEqual(result["original_vulkan_parity"], "mismatch")
        self.assertTrue(result["boundaries"][0]["decoded_half_bits"]["exact"])
        self.assertFalse(result["boundaries"][0]["codes_exact"])

    def test_invalid_subset_or_proxy_never_opens_backend(self):
        for change in ("subset", "proxy", "extra"):
            self.metadata = deepcopy(self.base)
            if change == "subset":
                self.metadata["boundaries"].pop("block-0")
            elif change == "proxy":
                self.metadata["checks"] = ["composed-rgb"]
            else:
                self.metadata["boundaries"]["block-70"] = self.metadata["boundaries"]["block-0"]
            with self.assertRaises(ValueError):
                self.parity()
        self.assertEqual(self.instances, [])

    def test_wrong_model_geometry_and_hash_rejected_before_cuda(self):
        for target in ("model", "geometry", "file"):
            self.metadata = deepcopy(self.base)
            if target == "model":
                self.metadata["model"]["source_manifest_sha256"] = "2" * 64
            elif target == "geometry":
                self.metadata["geometry"]["full_width"] = 321
            else:
                self.metadata["features"]["file_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                self.parity()
        self.assertEqual(self.instances, [])

    def test_nonfinite_head_missing_boundary_wrong_device_cleanup_are_runtime_failures(self):
        def nonfinite(_, value):
            if isinstance(value, np.ndarray):
                value.flat[0] = np.nan
        cases = [self.factory(nonfinite), self.factory(missing=True), self.factory(fail_close=True),
                 self.factory(device={"ordinal": 0, "compute_capability": [8, 6]})]
        for factory in cases:
            with self.assertRaises(RuntimeError):
                self.parity(backend_factory=factory)
        self.assertTrue(all(x.closed for x in self.instances))

    def test_unavailable_original_is_not_pass_and_needs_no_library(self):
        self.metadata = {"format": f.FORMAT, "availability": "unavailable",
                         "reference_domain": "original-vulkan", "reason": "Original shader capture not supplied"}
        result = f.run_parity(self.manifest(), model="missing", library="missing", backend_factory=self.factory())
        self.assertEqual(f.EXIT_CODES[result["status"]], 3)
        self.assertFalse(result["parity_pass"])
        self.assertEqual(self.instances, [])

    def test_input_nonfinite_or_outside_half_range_rejected(self):
        for number, value in enumerate((np.nan, np.inf, 65536)):
            array = np.zeros((320, 320, 16), "<f4")
            array.flat[0] = value
            self.metadata["features"] = self.tensor(f"bad-features-{number}.npy", array, "f32-npy")
            with self.assertRaises(ValueError):
                self.parity()
        self.assertEqual(self.instances, [])

    def test_unsafe_path_and_bad_descriptor_rejected(self):
        for name in ("../outside.npy", "C:/outside.npy", "x\\features.npy", [], None):
            self.metadata["features"]["file"] = name
            with self.assertRaises(ValueError):
                self.parity()

    def test_npy_huge_header_rejected_before_allocation(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": False,
                                                     "shape": (2**60, 16)})
        raw = stream.getvalue()
        path = self.root / "huge.npy"
        path.write_bytes(raw)
        self.metadata["features"].update(file=path.name, bytes=len(raw), file_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "exact shape"):
            self.parity()

    def test_duplicate_json_and_replays_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"format":"x","format":"y"}', encoding="utf8")
        with self.assertRaises(ValueError):
            f.load_fixture(path)
        for replays in (True, 1, 2, 101):
            with self.assertRaises(ValueError):
                self.parity(replays=replays)


class IndependentEncodingTests(unittest.TestCase):
    def test_all_finite_codes_and_zero_signs(self):
        codes = np.array([x for x in range(256) if (x & 127) != 127], np.uint8)
        # Independent scalar E4 values in units of 2^-9.
        values = []
        for code in codes.tolist():
            exp, fraction = (code & 127) // 8, code % 8
            magnitude = fraction if exp == 0 else (8 + fraction) * 2**(exp - 1)
            values.append((-1.0 if code & 128 else 1.0) * magnitude / 512)
        actual = f.encode_exact_e4(np.asarray(values, "<f2"))
        np.testing.assert_array_equal(actual, codes)
        self.assertEqual(actual[127], 128)

    def test_midpoints_reject_and_bits_keep_zero_sign(self):
        with self.assertRaises(ValueError):
            f.encode_exact_e4(np.array([1.0625], "<f2"))
        self.assertFalse(f.compare_bits(np.array([0], "<f4"), np.array([-0.0], "<f4"))["exact"])


if __name__ == "__main__":
    unittest.main()
