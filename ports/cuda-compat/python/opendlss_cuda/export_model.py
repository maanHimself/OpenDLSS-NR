"""Verify OpenDLSS-NR stages and export canonical FP16 weights for sm_75.

Public entry point: opendlss_cuda.export_model.export_model(source, destination)
Requires numpy. Source layouts follow src/nr_model.cpp and src/nr_graph.cpp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys

import numpy as np

from .geometry import align_up, block_schedule

FORMAT = "opendlss-fp16-sm75-v1"
MAX_MANIFEST_BYTES = 2 * 1024 ** 2


def require_hash(value, label="SHA256"):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {label}")
    return value


def regular_file(root: Path, name: str) -> Path:
    """Resolve a portable relative file name without following child symlinks."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\0" in name:
        raise ValueError("Invalid model file name")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Model file escapes selected directory")
    root = Path(root).resolve(strict=True)
    path = root
    for index, part in enumerate(parts):
        path = path / part
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or (index < len(parts) - 1 and not stat.S_ISDIR(mode)):
            raise ValueError("Model files cannot traverse symlinks or non-directories")
    if not stat.S_ISREG(mode) or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Model file must be regular and inside selected directory")
    return path


def read_manifest(root: Path):
    path = regular_file(root, "manifest.json")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Model manifest exceeds 2 MiB")
    raw = path.read_bytes()

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid_constant(value)
        return number

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError("Model manifest must be an object")
    return value, raw


def sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def packed_input_index(k):
    return (k & ~31) + (k & 16) + ((k & 15) >> 2) * 2 + (k & 1) + ((k & 2) != 0) * 8


def inverse_packed_input_index(k):
    return (k & ~31) + (k & 17) + ((k & 2) << 1) + ((k & 4) << 1) + ((k & 8) >> 2)


def packed_weight_index(k, n, ncols):
    lane = ((n & 7) << 2) | ((k & 15) >> 2)
    byte = (((n & 15) >> 3) << 3) | (((k & 31) >> 4) << 2) | (k & 3)
    return (k >> 5) * ncols * 32 + (n >> 7) * 4096 + ((n & 127) >> 6) * 2048 + ((n & 63) >> 4) * 512 + lane * 16 + byte


def packed_f16_weight_index(k, n, ncols):
    tile = (k >> 4) * ((ncols + 15) // 16) + (n >> 4)
    lane = ((n & 7) << 2) | ((k & 7) >> 1)
    return tile * 256 + lane * 8 + ((n >> 3) & 1) * 4 + ((k & 15) >= 8) * 2 + (k & 1)


def tiled_token(t):
    x, y = t & 7, t >> 3
    return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3)


def fp8_lookup() -> np.ndarray:
    """E4M3FN, with either NaN code replaced by positive zero like upstream."""
    code = np.arange(256, dtype=np.uint16)
    exponent, mantissa = (code & 127) >> 3, code & 7
    value = np.where(exponent == 0, mantissa.astype(np.float64) / 512,
                     (1 + mantissa.astype(np.float64) / 8) * np.exp2(exponent.astype(np.int32) - 7))
    value = np.where(code & 128, -value, value)
    value = np.where((code & 127) == 127, 0.0, value)
    return value.astype("<f2")


FP8 = fp8_lookup()


def checked_slice(raw, offset, nbytes):
    if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes < 0 or offset + nbytes > len(raw):
        raise ValueError(f"tensor slice outside bounds: offset={offset}, bytes={nbytes}, size={len(raw)}")
    return memoryview(raw)[offset:offset + nbytes]


def decode_fp8(raw, offset: int, k: int, n: int, batches: int = 1) -> tuple[np.ndarray, int]:
    if k <= 0 or k % 32 or n <= 0 or n % 16 or batches <= 0:
        raise ValueError("FP8 matrix requires positive K%32=0, N%16=0, batches")
    source = np.frombuffer(checked_slice(raw, offset, batches * k * n), dtype=np.uint8)
    result = np.empty((batches * k, n), dtype="<f2")
    columns = np.arange(n, dtype=np.int64)[None, :]
    # 32-row tiles bound temporary integer-index arrays even for the ViT.
    for first in range(0, batches * k, 32):
        rows = inverse_packed_input_index(np.arange(first, first + 32, dtype=np.int64)[:, None])
        result[first:first + 32] = FP8[source[packed_weight_index(rows, columns, n)]]
    nan_count = int(np.count_nonzero((source & 127) == 127))
    return result.reshape((batches, k, n) if batches > 1 else (k, n)), nan_count


def decode_f16(raw, offset: int, k: int, n: int) -> np.ndarray:
    if offset % 2 or k <= 0 or k % 16 or n <= 0:
        raise ValueError("F16 matrix requires even offset, positive K%16=0 and N")
    source = np.frombuffer(checked_slice(raw, offset, k * align_up(n, 16) * 2), dtype="<u2")
    rows = np.arange(k, dtype=np.int64)[:, None]
    columns = np.arange(n, dtype=np.int64)[None, :]
    # Preserve bit patterns; head output has four columns, without unused padded columns.
    return source[packed_f16_weight_index(rows, columns, n)].astype("<u2").view("<f2")


def decode_prior(raw, offset: int, heads: int) -> np.ndarray:
    source = np.frombuffer(checked_slice(raw, offset, heads * 8192), dtype="<u2")
    q = tiled_token(np.arange(64, dtype=np.int64))[:, None]
    key = np.arange(64, dtype=np.int64)[None, :]
    m, n = q & 15, key & 15
    lane = ((m & 7) << 2) | ((n & 7) >> 1)
    fragment = (m >= 8) * 2 + (n & 1)
    index = (q >> 4) * 1024 + (key >> 4) * 256 + lane * 8 + (n >> 3) * 4 + fragment
    return source[np.arange(heads)[:, None, None] * 4096 + index].astype("<u2").view("<f2")


def fused_layout(c: int, variant: str = "window") -> dict:
    if c not in (32, 64, 128, 256):
        raise ValueError("invalid fused width")
    if variant == "pre":
        return dict(w1=0, w2=4096, input_adapter=8208, ffn_skip=9232, qkv=9312,
                    prior=12384, attention_scale=20576, projection=20592, attn_skip=21616, end=21680)
    if variant == "post":
        return dict(w1=0, w2=4096, ffn_skip=8208, input_scale=8272, adapter_scale=8336,
                    qkv=8400, prior=11472, attention_scale=19664, projection=19680,
                    attn_skip=20704, head=20784, end=21808)
    experts = c // 32 if c >= 64 else 1
    expand = experts * c * 128
    weights_end = expand + experts * 128 * 32 + (c * c if c >= 64 else 0)
    layout = dict(w1=0, w2=expand)
    if c >= 64:
        layout["w3"] = expand + experts * 128 * 32
    if variant == "upsample":
        pad = 16 if c == 32 else 0
        layout["up_projection"] = weights_end
        layout["ffn_skip"] = weights_end + 2 * c * c + pad
        layout["merge_scale"] = layout["ffn_skip"] + c * 2 + pad
        layout["qkv"] = layout["merge_scale"] + c * 2
    elif variant == "window":
        layout["ffn_skip"] = weights_end + 16
        layout["qkv"] = layout["ffn_skip"] + c * 2 + 16
    else:
        raise ValueError("invalid fused variant")
    layout["prior"] = layout["qkv"] + 3 * c * c
    layout["attention_scale"] = layout["prior"] + (c // 32) * 8192
    layout["projection"] = layout["attention_scale"] + align_up((c // 32) * 4, 16)
    layout["attn_skip"] = layout["projection"] + c * c
    layout["end"] = layout["attn_skip"] + 2 * c
    return layout


def tensor_name(block: int, layer: int = 0, parameter: str = "layer") -> str:
    return f"block{block}.layer{layer}.{parameter}"


def array_specs() -> list[dict]:
    """A complete declarative extraction plan. Dimensions are mathematical K,N."""
    specs = []
    def add(block, name, kind, offset, shape, layer=0, parameter="layer"):
        specs.append(dict(name=f"block{block}.{name}", kind=kind, byte_offset=offset,
                          shape=list(shape), tensor=tensor_name(block, layer, parameter)))
    for b in block_schedule():
        block, c, kind = b["block"], b["channels"], b["type"]
        if c <= 256:
            variant = "upsample" if block in (48, 56, 62, 66) else kind
            layout = fused_layout(c, variant)
            if c == 32:
                add(block, "w1", "fp8", layout["w1"], (32, 128))
                add(block, "w2", "fp8", layout["w2"], (128, 32))
            else:
                add(block, "w1", "fp8", layout["w1"], (c // 32, c, 128))
                add(block, "w2", "fp8", layout["w2"], (c // 32, 128, 32))
                add(block, "w3", "fp8", layout["w3"], (c, c))
            for name in ("ffn_skip", "attn_skip", "merge_scale", "input_scale", "adapter_scale"):
                if name in layout:
                    add(block, name, "float16", layout[name], (c,))
            add(block, "qkv", "fp8", layout["qkv"], (c, 3 * c))
            add(block, "prior", "prior", layout["prior"], (c // 32, 64, 64))
            add(block, "attention_scale", "float32", layout["attention_scale"], (c // 32,))
            add(block, "projection", "fp8", layout["projection"], (c, c))
            if "up_projection" in layout:
                add(block, "up_projection", "fp8", layout["up_projection"], (2 * c, c))
            if block in (4, 8, 14, 22):
                add(block, "down_projection", "fp8", layout["end"], (c, 2 * c))
            if block == 0:
                add(block, "input_adapter", "f16_packed", layout["input_adapter"], (16, 32))
            if block == 70:
                add(block, "head", "f16_packed", layout["head"], (32, 4))
                add(block, "blend_scale", "float16", 0, (1,), parameter="blend_scale")
        elif kind == "split":
            add(block, "w1", "fp8", 0, (512, 512))
            add(block, "w2", "fp8", 512 * 512, (8, 64, 256))
            add(block, "w3", "fp8", 512 * 512 + 8 * 64 * 256, (8, 256, 64))
            add(block, "contract", "fp8", 0, (512, 512), layer=1)
            add(block, "ffn_skip", "float16", 512 * 512, (512,), layer=1)
            add(block, "qkv", "fp8", 0, (512, 1536), layer=2)
            add(block, "prior", "prior", 512 * 1536, (16, 64, 64), layer=2)
            add(block, "attention_scale", "float32", 512 * 1536 + 16 * 8192, (16,), layer=2)
            add(block, "projection", "fp8", 0, (512, 512), layer=3)
            add(block, "attn_skip", "float16", 512 * 512, (512,), layer=3)
            if block == 30:
                add(block, "down_projection", "fp8", 0, (512, 1024), layer=4)
        elif kind == "global":
            add(block, "w1", "fp8", 0, (1024, 4096))
            add(block, "w2", "fp8", 0, (4096, 1024), layer=1)
            add(block, "ffn_skip", "float16", 4096 * 1024, (1024,), layer=1)
            add(block, "attention_scale", "float32", 0, (32,), layer=2)
            add(block, "qkv", "fp8", 128, (1024, 3072), layer=2)
            add(block, "projection", "fp8", 0, (1024, 1024), layer=4)
            add(block, "attn_skip", "float16", 1024 * 1024, (1024,), layer=4)
        elif kind == "merge":
            add(block, "up_projection", "fp8", 0, (1024, 512))
            add(block, "merge_scale", "float16", 1024 * 512, (512,))
        else:
            raise AssertionError(kind)
    assert len({s["tensor"] for s in specs}) == 145
    return specs


def source_nbytes(spec):
    shape, kind = spec["shape"], spec["kind"]
    if kind == "f16_packed":
        return shape[0] * align_up(shape[1], 16) * 2
    return int(np.prod(shape)) * (1 if kind == "fp8" else 4 if kind == "float32" else 2)


def expected_tensor_lengths() -> dict[str, int]:
    sizes = {}
    for spec in array_specs():
        name = spec["tensor"]
        sizes[name] = max(sizes.get(name, 0), spec["byte_offset"] + source_nbytes(spec))
    for block in block_schedule():
        b, c = block["block"], block["channels"]
        if c <= 256 and b not in (4, 8, 14, 22, 70):
            sizes[tensor_name(b)] += 16
        elif block["type"] == "global":
            sizes[tensor_name(b)] += 16
            sizes[tensor_name(b, 3)] = 2
    sizes[tensor_name(30, 4)] += 16
    # The 32 -> 64 transition alone retains a final 16-byte pad.
    sizes[tensor_name(4)] += 16
    assert len(sizes) == 153
    return sizes


def _load_verified_model(directory: Path) -> tuple[dict, dict, dict]:
    directory = Path(directory)
    manifest, manifest_bytes = read_manifest(directory)
    if (manifest.get("format") != "OpenDLSS-NR-packed-layer-stages"
            or type(manifest.get("schemaVersion")) is not int or manifest["schemaVersion"] != 1):
        raise ValueError("Unsupported packed model format/schema")
    if manifest["totals"]["blockCount"] != 71 or len(manifest["stages"]) != 11 or len(manifest["tensors"]) != 153:
        raise ValueError("expected 71 blocks, 11 stages and 153 tensor records")
    if manifest["totals"]["stageCount"] != 11 or manifest["totals"]["tensorCount"] != 153:
        raise ValueError("manifest totals disagree with record counts")
    expected_total = sum(expected_tensor_lengths().values())
    if (type(manifest["totals"]["packedByteLength"]) is not int
            or manifest["totals"]["packedByteLength"] != expected_total):
        raise ValueError("Packed model total differs from supported layout")
    stage_sizes = [s["packedByteLength"] for s in manifest["stages"]]
    if (any(type(size) is not int or size <= 0 or size > expected_total for size in stage_sizes)
            or sum(stage_sizes) != expected_total):
        raise ValueError("Invalid packed stage sizes")
    stages, tensor_data, provenance = {}, {}, {"manifest_sha256": sha256(manifest_bytes), "stages": [], "tensors": []}
    for stage in manifest["stages"]:
        sid = stage["id"]
        if not isinstance(sid, str) or not sid:
            raise ValueError("Invalid stage identifier")
        if sid in stages:
            raise ValueError(f"duplicate stage: {sid}")
        if not isinstance(stage["file"], str):
            raise ValueError("Invalid stage filename")
        require_hash(stage["sha256"], "stage SHA256")
        path = regular_file(directory, "model/" + stage["file"])
        if path.stat().st_size != stage["packedByteLength"]:
            raise ValueError(f"stage size mismatch: {sid}")
        data = path.read_bytes()
        digest = sha256(data)
        if len(data) != stage["packedByteLength"] or digest != stage["sha256"]:
            raise ValueError(f"stage size/SHA256 mismatch: {sid}")
        stages[sid] = data
        provenance["stages"].append({"id": sid, "file": stage["file"], "nbytes": len(data), "sha256": digest})
    for tensor in manifest["tensors"]:
        name = tensor["name"]
        if name in tensor_data:
            raise ValueError(f"duplicate tensor: {name}")
        if name != tensor_name(tensor["block"], tensor["layer"], tensor["parameter"]):
            raise ValueError(f"tensor name disagrees with metadata: {name}")
        if not 0 <= tensor["block"] < 71 or tensor["stage"] not in stages:
            raise ValueError(f"invalid tensor block or stage: {name}")
        data = checked_slice(stages[tensor["stage"]], tensor["stageOffset"], tensor["byteLength"])
        digest = sha256(data)
        if tensor["sha256"] != digest:
            raise ValueError(f"tensor SHA256 mismatch: {name}")
        tensor_data[name] = data
        provenance["tensors"].append({"name": name, "nbytes": len(data), "sha256": digest,
                                     "stage": tensor["stage"], "stage_offset": tensor["stageOffset"]})
    expected = {s["tensor"] for s in array_specs()}
    ignored = {tensor_name(b, 3) for b in range(31, 39)}
    if set(tensor_data) != expected | ignored:
        raise ValueError(f"tensor set mismatch; missing={sorted((expected | ignored) - set(tensor_data))}; extra={sorted(set(tensor_data) - expected - ignored)}")
    for name in ignored:
        if len(tensor_data[name]) != 2:
            raise ValueError(f"unused ViT record must be two bytes: {name}")
    if sum(map(len, stages.values())) != manifest["totals"]["packedByteLength"]:
        raise ValueError("stage byte total disagrees with manifest totals")
    for name, expected_size in expected_tensor_lengths().items():
        if len(tensor_data[name]) != expected_size:
            raise ValueError(f"unexpected tensor length {name}: expected {expected_size}, got {len(tensor_data[name])}")
    for sid, data in stages.items():
        slices = sorted((t["stageOffset"], t["stageOffset"] + t["byteLength"]) for t in manifest["tensors"] if t["stage"] == sid)
        if not slices or slices[0][0] != 0 or slices[-1][1] != len(data) or any(a[1] != b[0] for a, b in zip(slices, slices[1:])):
            raise ValueError(f"tensor slices do not partition stage: {sid}")
    for spec in array_specs():
        checked_slice(tensor_data[spec["tensor"]], spec["byte_offset"], source_nbytes(spec))
    provenance["ignored_tensors"] = sorted(ignored)
    return manifest, tensor_data, provenance


def load_verified_model(directory: Path) -> tuple[dict, dict, dict]:
    try:
        return _load_verified_model(directory)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed packed model manifest: {error}") from error


def export_model(directory: Path, output: Path) -> dict:
    directory, output = Path(directory), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory must be fresh")
    _, tensors, provenance = load_verified_model(directory)
    output.mkdir(parents=True, exist_ok=False)
    arrays, replaced = [], 0
    pending = output / "weights.bin.partial"
    # Retain partial files for inspection; publish the manifest only after success.
    with pending.open("xb") as blob:
        for spec in array_specs():
            raw, shape, kind = tensors[spec["tensor"]], spec["shape"], spec["kind"]
            offset = spec["byte_offset"]
            nan_count = 0
            if kind == "fp8":
                data, nan_count = decode_fp8(raw, offset, shape[-2], shape[-1], shape[0] if len(shape) == 3 else 1)
            elif kind == "f16_packed":
                data = decode_f16(raw, offset, *shape)
            elif kind == "prior":
                data = decode_prior(raw, offset, shape[0])
            else:
                data = np.frombuffer(checked_slice(raw, offset, source_nbytes(spec)), dtype="<f4" if kind == "float32" else "<f2").reshape(shape)
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite model array: {spec['name']}")
            payload = data.tobytes(order="C")
            padding = (-blob.tell()) % 256
            blob.write(b"\0" * padding)
            descriptor = {"name": spec["name"], "dtype": "float32" if kind == "float32" else "float16",
                          "shape": shape, "offset": blob.tell(), "nbytes": len(payload), "sha256": sha256(payload),
                          "source": {"tensor": spec["tensor"], "byte_offset": offset, "kind": kind,
                                     "nbytes": source_nbytes(spec)}}
            if kind == "fp8":
                descriptor["source"].update(k_permutation="inverse_packed_input_index", nan_codes_replaced=nan_count)
            if kind == "prior":
                descriptor["axes"] = ["head", "natural_query", "physical_key"]
            blob.write(payload)
            arrays.append(descriptor)
            replaced += nan_count
    blob_digest = hashlib.sha256()
    with pending.open("rb") as blob:
        for chunk in iter(lambda: blob.read(8 << 20), b""):
            blob_digest.update(chunk)
    graph = block_schedule()
    for block in graph:
        block["parameters"] = [a["name"] for a in arrays if a["name"].startswith(block["parameter_prefix"])]
    result = {"format": FORMAT, "endianness": "little", "layout": "row_major",
              "block_count": 71, "source_records": 153, "consumed_records": 145,
              "blob": {"file": "weights.bin", "nbytes": pending.stat().st_size, "sha256": blob_digest.hexdigest()},
              "nan_weight_codes_replaced": replaced, "arrays": arrays, "blocks": graph, "source": provenance}
    (output / "manifest.json.partial").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    pending.replace(output / "weights.bin")
    (output / "manifest.json.partial").replace(output / "manifest.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    result = export_model(args.model_directory, args.output_directory)
    print(json.dumps({"format": result["format"], "arrays": len(result["arrays"]), "blob": result["blob"],
                      "nan_weight_codes_replaced": result["nan_weight_codes_replaced"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"export failed: {error}", file=sys.stderr)
        raise SystemExit(1)
