"""Strict prepared-feature parity fixtures. Importing this module uses no CUDA.

Only the complete padded head and all 75 published graph boundaries are supported.
The reference's domain is an explicit claim of provenance, not authentication.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from .geometry import block_schedule, geometry_from_valid, require_supported_geometry
from .export_model import regular_file

FORMAT = "opendlss-cuda-feature-fixture-v1"
POLICY = "browser-compat-v1"
EXIT_CODES = {"passed": 0, "mismatch": 1, "invalid": 2, "runtime_error": 2, "unavailable": 3}
MAX_FILE_BYTES = 128 * 1024**2
MAX_FIXTURE_BYTES = 512 * 1024**2
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
TRANSITIONS = (0, 4, 8, 14, 22)
REQUIRED_BOUNDARIES = tuple(
    name for block in range(70)
    for name in ((f"block-{block}", f"transition-{block}-{block+1}")
                 if block in TRANSITIONS else (f"block-{block}",)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, keys, label):
    _require(isinstance(value, dict) and set(value) == set(keys), f"{label}: unexpected or missing fields")


def _digest(value, label):
    _require(isinstance(value, str) and _HASH.fullmatch(value), f"{label}: invalid SHA256")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path):
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 << 20,
             "Missing, unsafe or oversized JSON manifest")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {x}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read manifest: {error}") from error


def _file(root, name, maximum=MAX_FILE_BYTES):
    _require(isinstance(name, str) and name, "Empty fixture filename")
    part = PurePosixPath(name)
    _require(not part.is_absolute() and part.as_posix() == name and
             not any(x in (".", "..") for x in part.parts) and "\\" not in name and ":" not in name,
             "Unsafe fixture filename")
    try:
        path = regular_file(root, name)
    except OSError as error:
        raise ValueError(f"Missing or unsafe fixture file: {name}") from error
    _require(0 < path.stat().st_size <= maximum, "Empty or oversized fixture file")
    return path


def required_boundary_shapes(geometry):
    result = {}
    for block in block_schedule():
        number, level = block["block"], block["level"]
        rows = geometry["full_rows"] if level == 6 else geometry["levels"][level]["rows"]
        if number < 70:
            result[f"block-{number}"] = (rows, block["channels"])
        if number in TRANSITIONS:
            after = block["after"]
            result[f"transition-{number}-{number+1}"] = (
                geometry["levels"][after["level"]]["rows"], after["channels"])
    _require(set(result) == set(REQUIRED_BOUNDARIES), "Internal required boundary schedule changed")
    return result


def _geometry(value):
    _keys(value, ("mode", "valid_width", "valid_height", "full_width", "full_height"), "geometry")
    for key in ("valid_width", "valid_height", "full_width", "full_height"):
        _require(type(value[key]) is int, f"geometry.{key}: expected integer")
    expected = require_supported_geometry(geometry_from_valid(value["valid_width"], value["valid_height"], value["mode"]))
    _require(all(expected[k] == v for k, v in value.items()), "Valid/padded geometry does not match the selected rule")
    _require(expected["padded_vit_tokens"] <= 1024 and expected["full_rows"] * 16 * 4 <= MAX_FILE_BYTES,
             "Fixture geometry exceeds bounded supported input")
    return expected


def _tensor(root, descriptor, shape, fmt):
    _keys(descriptor, ("file", "file_sha256", "bytes", "array_sha256", "format", "shape"), "tensor")
    _require(descriptor["format"] == fmt and descriptor["shape"] == list(shape), "Tensor format/shape metadata mismatch")
    _require(all(type(x) is int and x > 0 for x in descriptor["shape"]), "Invalid tensor dimensions")
    path = _file(root, descriptor["file"])
    _require(type(descriptor["bytes"]) is int and path.stat().st_size == descriptor["bytes"], "Tensor file size mismatch")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _digest(descriptor["file_sha256"], "tensor file"), "Tensor file SHA256 mismatch")
    if fmt == "e4":
        _require(len(raw) == int(np.prod(shape)), "E4 tensor byte count mismatch")
        value = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    else:
        try:
            # Parse and bound the header before allocating. np.load alone can
            # allocate an attacker-declared huge shape from a tiny input file.
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            _require(version in ((1, 0), (2, 0)), "Only NPY v1/v2 arrays are supported")
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            declared_shape, fortran_order, dtype = reader(stream, max_header_size=4096)
            _require(dtype.str == "<f4" and declared_shape == tuple(shape) and not fortran_order,
                     "Expected C-order little-endian float32 NPY with exact shape")
            _require(len(raw) - stream.tell() == int(np.prod(shape)) * 4, "NPY payload size mismatch")
            value = np.frombuffer(raw, dtype="<f4", offset=stream.tell()).reshape(shape)
        except (OSError, ValueError, EOFError) as error:
            raise ValueError(f"Invalid tensor NPY: {error}") from error
        _require(isinstance(value, np.ndarray) and value.dtype.str == "<f4" and value.shape == tuple(shape),
                 "Expected little-endian float32 NPY with exact shape")
        _require(value.flags.c_contiguous and np.isfinite(value).all(), "Tensor must be finite and C-contiguous")
    _require(array_sha256(value) == _digest(descriptor["array_sha256"], "tensor array"), "Tensor array SHA256 mismatch")
    value.setflags(write=False)
    return value


def _reference(value):
    _require(isinstance(value, dict), "Missing reference provenance")
    domain = value.get("domain")
    _require(domain in (POLICY, "original-vulkan"), "Unsupported reference domain")
    common = {"domain", "upstream_commit", "capture_manifest_sha256", "producer", "producer_sha256", "device", "driver"}
    _keys(value, common | ({"browser_version", "browser_binary_sha256", "shader_source_sha256"} if domain == POLICY else set()), "reference")
    _require(isinstance(value["upstream_commit"], str) and _COMMIT.fullmatch(value["upstream_commit"]), "Invalid reference source commit")
    for key in ("capture_manifest_sha256", "producer_sha256"):
        _digest(value[key], key)
    if domain == POLICY:
        for key in ("browser_binary_sha256", "shader_source_sha256"):
            _digest(value[key], key)
    for key, item in value.items():
        _require(isinstance(item, str) and 0 < len(item) <= 1024, f"Invalid reference identity: {key}")
    return domain


@dataclass(frozen=True)
class Fixture:
    path: Path
    sha256: str
    metadata: dict
    geometry: dict | None = None
    features: np.ndarray | None = None
    head: np.ndarray | None = None
    boundaries: dict | None = None


def load_fixture(path):
    """Validate and read every required reference byte before constructing CUDA."""
    path = Path(path)
    if path.is_dir():
        path = path / "fixture.json"
    path = _file(path.parent, path.name, 2 << 20)
    value = _read_json(path)
    _require(isinstance(value, dict) and value.get("format") == FORMAT, "Unsupported fixture format")
    if value.get("availability") == "unavailable":
        _keys(value, ("format", "availability", "reference_domain", "reason"), "unavailable fixture")
        _require(value["reference_domain"] == "original-vulkan" and isinstance(value["reason"], str)
                 and 0 < len(value["reason"]) <= 2048, "Only absent original/Vulkan references can be declared unavailable")
        return Fixture(path.resolve(), file_sha256(path), value)
    _keys(value, ("format", "availability", "checks", "reference", "model", "geometry", "features", "head", "boundaries"), "fixture")
    _require(value["availability"] == "available" and value["checks"] == ["head", "all-75-boundaries"],
             "Only full feature-input head/all-75-boundary checks are supported; proxy/composed/subset checks are rejected")
    _reference(value["reference"])
    _keys(value["model"], ("source_manifest_sha256", "canonical_manifest_sha256", "canonical_blob_sha256"), "model")
    for key, digest in value["model"].items():
        _digest(digest, key)
    geometry = _geometry(value["geometry"])
    _require(isinstance(value["boundaries"], dict) and set(value["boundaries"]) == set(REQUIRED_BOUNDARIES),
             "Exactly all 75 required boundaries must be present")
    descriptors = [value["features"], value["head"], *value["boundaries"].values()]
    _require(all(isinstance(x, dict) and type(x.get("bytes")) is int and x["bytes"] > 0 for x in descriptors), "Invalid tensor byte counts")
    _require(sum(x["bytes"] for x in descriptors) <= MAX_FIXTURE_BYTES, "Fixture exceeds total byte limit")
    _require(all(isinstance(x.get("file"), str) for x in descriptors), "Invalid tensor filename")
    _require(len({x["file"] for x in descriptors}) == len(descriptors), "Tensor files must be distinct")
    spatial = (geometry["full_height"], geometry["full_width"])
    features = _tensor(path.parent, value["features"], (*spatial, 16), "f32-npy")
    _require(np.max(np.abs(features)) <= 65504, "Feature exceeds finite FP16 range")
    head = _tensor(path.parent, value["head"], (*spatial, 4), "f32-npy")
    shapes = required_boundary_shapes(geometry)
    boundaries = {name: _tensor(path.parent, value["boundaries"][name], shapes[name], "e4") for name in REQUIRED_BOUNDARIES}
    return Fixture(path.resolve(), file_sha256(path), value, geometry, features, head, boundaries)


def _e4_values():
    # Independent mathematical decode, intentionally not imported from the model
    # exporter or CUDA numeric helpers. Preserve both zero signs. NaN codes are
    # decoded only for a separate value comparison; code comparison keeps them.
    result = np.empty(256, dtype="<f2")
    for code in range(256):
        exponent, mantissa = (code & 127) >> 3, code & 7
        value = mantissa / 512 if exponent == 0 else (8 + mantissa) * 2.0**(exponent - 10)
        result[code] = 0.0 if (code & 127) == 127 else (-value if code & 128 else value)
    return result


_E4_VALUES = _e4_values()
_HALF_TO_E4 = np.full(65536, -1, dtype=np.int16)
for _code in range(256):
    if (_code & 127) != 127:
        _HALF_TO_E4[_E4_VALUES[_code:_code+1].view(np.uint16)[0]] = _code


def encode_exact_e4(value):
    """Return original signed E4 codes only for exactly representable halfs."""
    value = np.asarray(value)
    _require(value.dtype.str == "<f2" and np.isfinite(value).all(), "E4 candidate must be finite little-endian float16")
    code = _HALF_TO_E4[np.ascontiguousarray(value).view(np.uint16)]
    _require(np.all(code >= 0), "Candidate boundary is not exactly E4-representable")
    return code.astype(np.uint8)


def compare_bits(expected, actual):
    """Compare bytes without numeric casts; signed zeros remain different."""
    actual = np.asarray(actual)
    _require(expected.dtype == actual.dtype and expected.shape == actual.shape, "Candidate dtype/shape mismatch")
    width = expected.dtype.itemsize
    left = np.ascontiguousarray(expected).view(np.uint8).reshape(-1, width)
    right = np.ascontiguousarray(actual).view(np.uint8).reshape(-1, width)
    mismatch = np.any(left != right, axis=1)
    where = np.flatnonzero(mismatch)
    return {"exact": not bool(where.size), "elements": int(mismatch.size), "mismatches": int(where.size),
            "first_mismatch_flat": None if not where.size else int(where[0]),
            "expected_array_sha256": array_sha256(expected), "actual_array_sha256": array_sha256(actual)}


def _model_identity(directory, expected):
    root = Path(directory)
    manifest_path = _file(root, "manifest.json", 2 << 20)
    value = _read_json(manifest_path)
    _require(isinstance(value, dict) and isinstance(value.get("source"), dict)
             and isinstance(value.get("blob"), dict), "Invalid canonical model manifest")
    _require(file_sha256(manifest_path) == expected["canonical_manifest_sha256"], "Canonical model manifest differs from fixture")
    _require(value.get("source", {}).get("manifest_sha256") == expected["source_manifest_sha256"], "Packed-source model identity differs")
    blob = value.get("blob", {})
    path = _file(root, blob.get("file"), 1 << 30)
    _require(type(blob.get("nbytes")) is int and path.stat().st_size == blob["nbytes"], "Canonical blob byte count mismatch")
    _require(file_sha256(path) == blob.get("sha256") == expected["canonical_blob_sha256"], "Canonical model blob differs from fixture")
    return dict(expected)


def run_parity(fixture, *, model, library, replays=3, backend_factory=None):
    """Return passed/mismatch/unavailable; invalid inputs raise ValueError.

    Runtime failures, including nonfinite/malformed outputs or cleanup errors,
    raise RuntimeError. This function never writes or relaxes fixture checks.
    """
    _require(type(replays) is int and 3 <= replays <= 100, "Parity requires 3..100 production replays")
    fixture = load_fixture(fixture)
    if fixture.geometry is None:
        return {"format": "opendlss-cuda-parity-v1", "status": "unavailable", "fixture_sha256": fixture.sha256,
                "reference_domain": "original-vulkan", "reason": fixture.metadata["reason"], "parity_pass": False}
    identity = _model_identity(model, fixture.metadata["model"])
    library = Path(library)
    _require(library.is_file() and not library.is_symlink(), "Native library is missing or unsafe")
    library_hash = file_sha256(library)
    if backend_factory is None:
        from .backend import Backend
        from .model import CanonicalModel
        # A malformed canonical cache is invalid input, not a CUDA failure.
        # Full schema/array validation happens before any backend is opened.
        with CanonicalModel(model):
            pass
        backend_factory = Backend
    g = fixture.geometry
    kwargs = {"geometry": g["mode"], "policy": POLICY, "window_impl": "fused"}
    production_checks, hashes, capture_checks = [], [], []
    first_head = None
    receipts = []
    try:
        def verify_backend(backend):
            options = backend.context_options
            device = backend.device_info
            if (options.get("policy") != POLICY or options.get("window_impl") != "fused"
                    or device.get("compute_capability") != [7, 5]
                    or type(device.get("ordinal")) is not int or device["ordinal"] < 0):
                raise RuntimeError("Actual native context does not match the requested compatibility/SM75 contract")
            if backend.graph.geometry != g:
                raise RuntimeError("Actual backend geometry differs from fixture")
            actual_model = backend.model.identity
            if (actual_model.get("manifest_sha256") != identity["canonical_manifest_sha256"]
                    or actual_model.get("blob_sha256") != identity["canonical_blob_sha256"]
                    or actual_model.get("source_manifest_sha256") != identity["source_manifest_sha256"]):
                raise RuntimeError("Actual backend model differs from fixture")
            if backend.runtime.library_sha256 != library_hash:
                raise RuntimeError("Actual loaded native library differs")
            receipts.append({"context_options": dict(options), "device_info": dict(device),
                             "model": dict(actual_model), "library_sha256": backend.runtime.library_sha256,
                             "native_source_sha256": backend.runtime.source_sha256,
                             "native_build_info": backend.runtime.build_info})

        def head(backend):
            value = np.asarray(backend.run_features(fixture.features))
            if value.dtype.str != "<f4" or value.shape != fixture.head.shape or not np.isfinite(value).all():
                raise RuntimeError("Malformed or nonfinite native head")
            return value.copy()

        with backend_factory(model, library, g["valid_width"], g["valid_height"], capture_boundaries=False, **kwargs) as backend:
            verify_backend(backend)
            for _ in range(replays):
                actual = head(backend)
                if first_head is None:
                    first_head = actual.copy()
                hashes.append(array_sha256(actual))
                production_checks.append(compare_bits(fixture.head, actual))
        with backend_factory(model, library, g["valid_width"], g["valid_height"], capture_boundaries=True, **kwargs) as backend:
            verify_backend(backend)
            instrumented = head(backend)
            instrumented_check = compare_bits(first_head, instrumented)
            instrumented_reference = compare_bits(fixture.head, instrumented)
            boundaries = backend.read_boundaries()
            if not isinstance(boundaries, dict) or set(boundaries) != set(REQUIRED_BOUNDARIES):
                raise RuntimeError("Native capture did not return exactly the required 75 boundaries")
            for name in REQUIRED_BOUNDARIES:
                actual = np.asarray(boundaries[name])
                expected = fixture.boundaries[name]
                if actual.dtype.str != "<f2" or actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise RuntimeError(f"Malformed or nonfinite native boundary: {name}")
                decoded = compare_bits(_E4_VALUES[expected], actual)
                try:
                    codes = encode_exact_e4(actual)
                except ValueError:
                    capture_checks.append({"name": name, "representable_as_e4": False, "codes_exact": False,
                                           "decoded_half_bits": decoded})
                else:
                    bits = compare_bits(expected, codes)
                    capture_checks.append({"name": name, "representable_as_e4": True, "codes_exact": bits["exact"],
                                           "e4_code_bits": bits, "decoded_half_bits": decoded})
    except Exception as error:
        raise RuntimeError(f"Parity execution failed: {error}") from error
    _require(file_sha256(library) == library_hash, "Native library changed during parity")
    _model_identity(model, identity)
    _require(array_sha256(fixture.features) == fixture.metadata["features"]["array_sha256"], "Backend modified the prepared features")
    deterministic = len(set(hashes)) == 1
    passed = (deterministic and all(x["exact"] for x in production_checks) and instrumented_check["exact"]
              and instrumented_reference["exact"] and all(x["codes_exact"] for x in capture_checks))
    domain = fixture.metadata["reference"]["domain"]
    return {"format": "opendlss-cuda-parity-v1", "status": "passed" if passed else "mismatch", "parity_pass": passed,
            "reference_domain": domain, "reference": fixture.metadata["reference"], "fixture_sha256": fixture.sha256,
            "model": identity, "library_sha256": library_hash, "features_array_sha256": fixture.metadata["features"]["array_sha256"],
            "geometry": fixture.metadata["geometry"], "production_replays": replays, "production_head_sha256": hashes,
            "deterministic": deterministic, "production_heads": production_checks,
            "instrumented_head_matches_production": instrumented_check, "instrumented_head_matches_reference": instrumented_reference,
            "required_boundary_names": list(REQUIRED_BOUNDARIES), "boundaries": capture_checks, "contexts": receipts,
            "original_vulkan_parity": ("passed" if passed else "mismatch") if domain == "original-vulkan" else "unavailable",
            "scope": "Full padded feature-to-head and all75 published E4 boundaries in the identified reference domain; no image, temporal or native-FP8 parity implication."}
