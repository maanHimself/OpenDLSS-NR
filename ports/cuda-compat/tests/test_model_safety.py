"""Cache/export safety using two tiny arrays, plus real fixed-layout metadata.

The two-array test schema is patched only inside these tests. Production always
derives all 640 arrays; test_model_export independently checks that full layout.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda import export_model as ex
from opendlss_cuda import model as mdl

SOURCE_SIZES = ex.expected_tensor_lengths()

SPECS = [
    {"name": "block0.w1", "tensor": "block0.layer0.layer", "kind": "fp8", "shape": [32, 16], "byte_offset": 0},
    {"name": "block70.blend_scale", "tensor": "block70.layer0.layer", "kind": "float16", "shape": [1], "byte_offset": 0},
]


def source_provenance():
    stages = [{"id": f"stage-{i:02d}", "file": f"stages/{i}.bin", "nbytes": 0, "sha256": "a" * 64} for i in range(11)]
    tensors = []
    for index, (name, size) in enumerate(SOURCE_SIZES.items()):
        stage = stages[index % 11]
        tensors.append({"name": name, "nbytes": size, "sha256": "b" * 64,
                        "stage": stage["id"], "stage_offset": stage["nbytes"]})
        stage["nbytes"] += size
    return {"manifest_sha256": "c" * 64, "stages": stages, "tensors": tensors,
            "ignored_tensors": sorted(ex.tensor_name(b, 3) for b in range(31, 39))}


class ModelSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "cache"
        self.spec_patches = [patch.object(ex, "array_specs", return_value=deepcopy(SPECS)),
                             patch.object(mdl, "array_specs", return_value=deepcopy(SPECS)),
                             patch.object(ex, "expected_tensor_lengths", return_value=SOURCE_SIZES),
                             patch.object(mdl, "expected_tensor_lengths", return_value=SOURCE_SIZES)]
        for item in self.spec_patches: item.start()
        self.addCleanup(lambda: [item.stop() for item in reversed(self.spec_patches)])
        source = {"block0.layer0.layer": bytes(512),
                  "block70.layer0.layer": np.array([0.5], dtype="<f2").tobytes()}
        with patch.object(ex, "load_verified_model", return_value=({}, source, source_provenance())):
            self.manifest = ex.export_model(self.root, self.output)

    def write(self, document=None):
        (self.output / "manifest.json").write_text(json.dumps(self.manifest if document is None else document))

    def test_export_load_roundtrip_and_readonly_lifetime(self):
        with mdl.CanonicalModel(self.output) as model:
            self.assertEqual(model.blend_scale, 0.5)
            self.assertEqual(model.identity["source_manifest_sha256"], "c" * 64)
            self.assertEqual(model.identity["blob_sha256"], self.manifest["blob"]["sha256"])
            view = model.arrays["block70.blend_scale"]
            self.assertFalse(view.flags.writeable)
        self.assertEqual(float(view[0]), 0.5)  # no forcibly unmapped dangling views
        with self.assertRaisesRegex(RuntimeError, "closed"): model.upload(None)

    def test_destination_must_be_fresh_even_if_empty(self):
        empty = self.root / "existing"
        empty.mkdir()
        with patch.object(ex, "load_verified_model") as loader:
            with self.assertRaisesRegex(ValueError, "fresh"): ex.export_model(self.root, empty)
            loader.assert_not_called()
        self.assertEqual(list(empty.iterdir()), [])

    def test_descriptor_mutations_rejected_before_mapping(self):
        changes = [lambda d: d["arrays"][0].update(name="block999.w1"),
                   lambda d: d["arrays"][0].update(shape=[16, 32]),
                   lambda d: d["arrays"][0].update(dtype="float32"),
                   lambda d: d["arrays"][0].update(offset=256),
                   lambda d: d["arrays"][0].update(nbytes=512),
                   lambda d: d["arrays"][0]["source"].update(byte_offset=1),
                   lambda d: d["arrays"].append(d["arrays"][0]),
                   lambda d: d["blocks"][1].update(phase=3),
                   lambda d: d["blob"].update(nbytes=2**63),
                   lambda d: d["source"]["tensors"][0].update(stage_offset=1),
                   lambda d: d["source"].update(manifest_sha256="not-a-hash")]
        for change in changes:
            document = deepcopy(self.manifest)
            change(document)
            self.write(document)
            with self.subTest(change=change), patch.object(mdl.np, "memmap") as memory:
                with self.assertRaises(ValueError): mdl.CanonicalModel(self.output)
                memory.assert_not_called()

    def test_blob_path_traversal_and_absolute_paths_rejected(self):
        for filename in ("../weights.bin", "/weights.bin", "C:/weights.bin", "nested/../weights.bin", "nested\\weights.bin"):
            self.manifest["blob"]["file"] = filename
            self.write()
            with self.subTest(filename=filename), self.assertRaises(ValueError): mdl.CanonicalModel(self.output)

    def test_truncated_blob_and_modified_hash_rejected(self):
        path = self.output / "weights.bin"
        payload = path.read_bytes()
        path.write_bytes(payload[:-1])
        with self.assertRaisesRegex(ValueError, "size"): mdl.CanonicalModel(self.output)
        path.write_bytes(bytes([1]) + payload[1:])
        with self.assertRaisesRegex(ValueError, "SHA256"): mdl.CanonicalModel(self.output)

    def test_nonfinite_array_rejected_even_with_consistent_hashes(self):
        path = self.output / "weights.bin"
        payload = bytearray(path.read_bytes())
        payload[0:2] = bytes([0, 124])  # +infinity half
        path.write_bytes(payload)
        self.manifest["blob"]["sha256"] = hashlib.sha256(payload).hexdigest()
        first = self.manifest["arrays"][0]
        first["sha256"] = hashlib.sha256(payload[:first["nbytes"]]).hexdigest()
        self.write()
        with self.assertRaisesRegex(ValueError, "Nonfinite"): mdl.CanonicalModel(self.output)

    def test_fp8_cache_rejects_finite_half_values_outside_e4_grid(self):
        path = self.output / "weights.bin"
        payload = bytearray(path.read_bytes())
        payload[0:2] = bytes([1, 60])  # 1 + 2^-10 is finite half but not E4
        path.write_bytes(payload)
        self.manifest["blob"]["sha256"] = hashlib.sha256(payload).hexdigest()
        first = self.manifest["arrays"][0]
        first["sha256"] = hashlib.sha256(payload[:first["nbytes"]]).hexdigest()
        self.write()
        with self.assertRaisesRegex(ValueError, "E4 representable"): mdl.CanonicalModel(self.output)

    def test_manifest_duplicate_keys_nonfinite_and_size_bound(self):
        path = self.output / "manifest.json"
        for content in ('{"format":1,"format":2}', '{"format":NaN}', '{"format":1e999}'):
            path.write_text(content)
            with self.assertRaises(ValueError): mdl.CanonicalModel(self.output)
        with path.open("wb") as stream: stream.truncate(ex.MAX_MANIFEST_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "2 MiB"): mdl.CanonicalModel(self.output)

    def test_bad_container_types_fail_as_invalid_manifests(self):
        for field in ("arrays", "source", "blocks", "blob"):
            document = deepcopy(self.manifest)
            document[field] = None
            self.write(document)
            with self.subTest(field=field), self.assertRaises(ValueError): mdl.CanonicalModel(self.output)

    def test_child_symlink_rejected(self):
        external = self.root / "outside.bin"
        external.write_bytes((self.output / "weights.bin").read_bytes())
        link = self.output / "alias.bin"
        try: link.symlink_to(external)
        except (OSError, NotImplementedError): self.skipTest("symlink creation unavailable")
        self.manifest["blob"]["file"] = "alias.bin"
        self.write()
        with self.assertRaisesRegex(ValueError, "symlink"): mdl.CanonicalModel(self.output)

    def test_source_manifest_schema_and_sizes_fail_before_stage_reads(self):
        source = self.root / "packed"
        source.mkdir()
        document = {"format": "OpenDLSS-NR-packed-layer-stages", "schemaVersion": 1,
                    "totals": {"blockCount": 71, "stageCount": 11, "tensorCount": 153,
                               "packedByteLength": sum(ex.expected_tensor_lengths().values())},
                    "stages": [{"packedByteLength": 2**63}] * 11, "tensors": [{}] * 153}
        (source / "manifest.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "stage sizes"): ex.load_verified_model(source)
        document["schemaVersion"] = 2
        (source / "manifest.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "format/schema"): ex.load_verified_model(source)

    def test_failed_export_never_publishes_manifest(self):
        target = self.root / "failed-cache"
        source = {"block0.layer0.layer": bytes(512),
                  "block70.layer0.layer": np.array([np.inf], dtype="<f2").tobytes()}
        with patch.object(ex, "load_verified_model", return_value=({}, source, source_provenance())):
            with self.assertRaisesRegex(ValueError, "Nonfinite"): ex.export_model(self.root, target)
        self.assertFalse((target / "manifest.json").exists())
        self.assertTrue((target / "weights.bin.partial").exists())


if __name__ == "__main__": unittest.main()
