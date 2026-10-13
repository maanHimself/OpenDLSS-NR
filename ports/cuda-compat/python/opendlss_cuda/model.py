"""Bounded, verified loading of the supported canonical model layout."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .export_model import (FORMAT, FP8, array_specs, expected_tensor_lengths,
                           read_manifest, regular_file, require_hash, source_nbytes,
                           tensor_name)
from .geometry import block_schedule


def expected_layout():
    """Describe the fixed cache layout without allocating any tensor storage."""
    arrays = []
    end = 0
    for spec in array_specs():
        offset = (end + 255) // 256 * 256
        dtype = "float32" if spec["kind"] == "float32" else "float16"
        size = math.prod(spec["shape"]) * (4 if dtype == "float32" else 2)
        source = {"tensor": spec["tensor"], "byte_offset": spec["byte_offset"],
                  "kind": spec["kind"], "nbytes": source_nbytes(spec)}
        if spec["kind"] == "fp8":
            source["k_permutation"] = "inverse_packed_input_index"
        row = {"name": spec["name"], "shape": spec["shape"], "dtype": dtype,
               "offset": offset, "nbytes": size, "source": source}
        if spec["kind"] == "prior":
            row["axes"] = ["head", "natural_query", "physical_key"]
        arrays.append(row)
        end = offset + size
    blocks = block_schedule()
    for block in blocks:
        block["parameters"] = [a["name"] for a in arrays if a["name"].startswith(block["parameter_prefix"])]
    return arrays, blocks, end


def _verify_source(source):
    require_hash(source["manifest_sha256"], "packed source manifest SHA256")
    sizes = expected_tensor_lengths()
    stages = source["stages"]
    tensors = source["tensors"]
    if len(stages) != 11 or len(tensors) != 153:
        raise ValueError("Invalid source provenance record counts")
    stage_sizes = {}
    for stage in stages:
        sid, size = stage["id"], stage["nbytes"]
        if not isinstance(sid, str) or not sid or sid in stage_sizes or type(size) is not int or size <= 0:
            raise ValueError("Invalid source stage provenance")
        require_hash(stage["sha256"], "source stage SHA256")
        stage_sizes[sid] = size
    if sum(stage_sizes.values()) != sum(sizes.values()):
        raise ValueError("Source stage total differs from supported layout")
    found = set()
    slices = {sid: [] for sid in stage_sizes}
    for tensor in tensors:
        name, size, sid, offset = tensor["name"], tensor["nbytes"], tensor["stage"], tensor["stage_offset"]
        if (name not in sizes or name in found or type(size) is not int or size != sizes[name]
                or sid not in stage_sizes or type(offset) is not int or offset < 0
                or offset + size > stage_sizes[sid]):
            raise ValueError("Invalid source tensor provenance")
        require_hash(tensor["sha256"], "source tensor SHA256")
        found.add(name)
        slices[sid].append((offset, offset + size))
    for sid, intervals in slices.items():
        intervals.sort()
        if (not intervals or intervals[0][0] != 0 or intervals[-1][1] != stage_sizes[sid]
                or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))):
            raise ValueError("Source provenance does not partition each stage")
    if source["ignored_tensors"] != sorted(tensor_name(b, 3) for b in range(31, 39)):
        raise ValueError("Invalid unused source record list")


class CanonicalModel:
    """Read-only model arrays. Expected names, shapes and offsets come from code.

    Hashes detect corruption and bind a cache to a receipt. They are not a
    signature authenticating an arbitrary user-supplied model.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.arrays = {}
        self.descriptors = {}
        self._blob = None
        self.closed = False
        try:
            self._load()
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            self.close()
            raise ValueError(f"Malformed canonical model manifest: {error}") from error
        except BaseException:
            self.close()
            raise

    def _load(self):
        m, raw_manifest = read_manifest(self.directory)
        self.manifest = m
        if m.get("format") != FORMAT or m.get("endianness") != "little" or m.get("layout") != "row_major":
            raise ValueError("Unsupported canonical model format")
        for key, value in (("block_count", 71), ("consumed_records", 145), ("source_records", 153)):
            if type(m.get(key)) is not int or m[key] != value:
                raise ValueError("Canonical model record counts disagree")
        _verify_source(m["source"])
        expected, blocks, blob_size = expected_layout()
        if len(m["arrays"]) != len(expected) or m["blocks"] != blocks:
            raise ValueError("Canonical arrays or graph schedule disagree with supported layout")
        blob = m["blob"]
        if type(blob["nbytes"]) is not int or blob["nbytes"] != blob_size:
            raise ValueError("Canonical blob size differs from supported layout")
        require_hash(blob["sha256"], "canonical weights SHA256")
        replaced = 0
        for actual, wanted in zip(m["arrays"], expected):
            for key in ("name", "dtype", "shape", "offset", "nbytes"):
                if actual.get(key) != wanted[key]:
                    raise ValueError(f"Canonical array {key} disagrees: {wanted['name']}")
            if (type(actual["offset"]) is not int or type(actual["nbytes"]) is not int
                    or any(type(d) is not int for d in actual["shape"])):
                raise ValueError("Canonical dimensions/offsets must be integers")
            source = actual["source"]
            if any(source.get(key) != value for key, value in wanted["source"].items()):
                raise ValueError(f"Canonical source layout disagrees: {wanted['name']}")
            if wanted["source"]["kind"] == "fp8":
                count = source["nan_codes_replaced"]
                if type(count) is not int or not 0 <= count <= source["nbytes"]:
                    raise ValueError("Invalid replaced NaN count")
                replaced += count
            if actual.get("axes") != wanted.get("axes"):
                raise ValueError("Canonical prior axes disagree")
            require_hash(actual["sha256"], "canonical array SHA256")
        if type(m["nan_weight_codes_replaced"]) is not int or m["nan_weight_codes_replaced"] != replaced:
            raise ValueError("Canonical NaN replacement count disagrees")
        blob_path = regular_file(self.directory, blob["file"])
        if blob_path.stat().st_size != blob_size:
            raise ValueError("Canonical weights blob size mismatch")
        digest = hashlib.sha256()
        with blob_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != blob["sha256"]:
            raise ValueError("Canonical weights blob SHA256 mismatch")
        self._blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
        e4_half_codes = np.zeros(65536, dtype=np.bool_)
        e4_half_codes[FP8.view("<u2")] = True
        end = 0
        for descriptor in m["arrays"]:
            offset, size = descriptor["offset"], descriptor["nbytes"]
            if np.any(self._blob[end:offset]):
                raise ValueError("Nonzero canonical alignment padding")
            dtype = "<f2" if descriptor["dtype"] == "float16" else "<f4"
            data = np.ndarray(descriptor["shape"], dtype=dtype, buffer=self._blob, offset=offset)
            if hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Canonical array SHA256 mismatch: {descriptor['name']}")
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite canonical array: {descriptor['name']}")
            if descriptor["source"]["kind"] == "fp8" and not e4_half_codes[data.view("<u2")].all():
                raise ValueError(f"Canonical FP8 array is not exactly E4 representable: {descriptor['name']}")
            self.arrays[descriptor["name"]] = data
            self.descriptors[descriptor["name"]] = descriptor
            end = offset + size
        self._identity = {"format": FORMAT,
                          "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                          "blob_sha256": blob["sha256"],
                          "source_manifest_sha256": m["source"]["manifest_sha256"]}

    @property
    def identity(self):
        return dict(self._identity)

    @property
    def blend_scale(self):
        if self.closed:
            raise RuntimeError("Model is closed")
        return float(self.arrays["block70.blend_scale"][0])

    def upload(self, runtime):
        if self.closed:
            raise RuntimeError("Model is closed")
        return {name: runtime.upload(array) for name, array in self.arrays.items()}

    def close(self):
        self.closed = True
        self.arrays.clear()
        # External read-only array views keep their mmap alive until released.
        # Closing the mmap forcibly would leave such views pointing at unmapped memory.
        self._blob = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Model is closed")
        return self

    def __exit__(self, *_):
        self.close()
