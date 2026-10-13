#!/usr/bin/env python3
"""Reproduce public synthetic F13 fixtures with an independent CPU oracle.

Only NumPy and the standard library are needed. No model, CUDA, browser, port
module or developer-local script is imported. The observed browser SiLU table
and the upstream numeric fixture are immutable reference inputs, not generated
from the CUDA implementation. --verify checks those hashes and regenerates the
seven synthetic files in memory without modifying the checked-in files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


PINNED = {
    "numerics.bin": "f82c0df26adc5a3b2222b632579314afd0c36cf7b142782a2bd5dcb746eb4b0b",
    "browser-silu-observed.bin": "9c9c1b971054a2f0bdbfe263567bf2f64ffaaab945901f4dae329ee614e57b23",
}
SHAPES = ((3, 32, 5, 0), (17, 128, 19, 64), (5, 128, 7, 32),
          (2, 1024, 3, 256), (3, 80, 5, 32), (3, 4096, 3, 1024))


def e4_table():
    values = []
    for code in range(256):
        exponent, fraction = (code & 127) // 8, code % 8
        magnitude = fraction / 512 if exponent == 0 else (8 + fraction) * 2.0**(exponent - 10)
        # This fixture records source NaN-to-positive-zero before GEMM.
        values.append(0.0 if (code & 127) == 127 else (-magnitude if code & 128 else magnitude))
    return np.asarray(values, "<f2")


E4 = e4_table()


def exponent(value, minimum):
    """Mathematical floor(log2(abs(x))), avoiding the CUDA bit manipulation."""
    _, powers = np.frexp(np.abs(np.asarray(value, np.float64)))
    return np.maximum(powers - 1, minimum)


def evaluate(a, w, seed=0, partition=0):
    """F13 by exact dyadic products and integer truncation on a shared grid.

    FP64 represents every finite E4 product and FP32 seed exactly; power-of-two
    scaling is exact. The sum is taken in integer units, independently of the
    CUDA mantissa packing/alignment implementation. Only a partition left fold
    deliberately rounds to FP32, matching the observed browser contract.
    """
    a, w = np.asarray(a, np.float64), np.asarray(w, np.float64)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[0] or a.shape[1] % 16:
        raise ValueError("Expected A[M,K], W[K,N] with K multiple of 16")
    if type(partition) is not int or partition < 0 or partition % 16:
        raise ValueError("Invalid partition")
    if not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite operands")
    acc = np.broadcast_to(np.asarray(seed, np.float32), (len(a), w.shape[1])).copy()
    if not np.isfinite(acc).all():
        raise ValueError("Nonfinite seed")
    ae, we = exponent(a, -6), exponent(w, -6)
    total = None
    for start in range(0, a.shape[1], 16):
        products = a[:, start:start+16, None] * w[None, start:start+16, :]
        powers = ae[:, start:start+16, None] + we[None, start:start+16, :]
        maximum = np.maximum(np.where(acc != 0, exponent(acc, -14), -21),
                             np.where(products != 0, powers, -1000).max(axis=1))
        incoming = np.trunc(np.ldexp(acc.astype(np.float64), 13 - maximum)).astype(np.int64)
        terms = np.trunc(np.ldexp(products, (13 - maximum)[:, None, :])).astype(np.int64)
        units = incoming + terms.sum(axis=1, dtype=np.int64)
        if np.any(np.abs(units) > 477184):
            raise ArithmeticError("F13 integer bound exceeded")
        acc = np.ldexp(units.astype(np.float32), maximum - 13)
        if partition and ((start + 16) % partition == 0 or start + 16 == a.shape[1]):
            total = acc.copy() if total is None else np.add(total, acc, dtype=np.float32)
            acc.fill(0)
    return total if partition else acc


def half_rtz(value):
    values = np.asarray(value, np.float32)
    with np.errstate(over="ignore"):
        nearest = values.astype("<f2")
    # A nearest half can only be one representable half beyond a finite value.
    bits = nearest.view("<u2").copy()
    bits -= (np.abs(nearest.astype(np.float32)) > np.abs(values)).astype(np.uint16)
    return bits.view("<f2")


def publish_e4(value):
    """Nearest finite E4 with even-code midpoint ties, without bit-round code."""
    values = np.asarray(value, np.float32)
    magnitude = np.minimum(np.abs(values), 448)
    positive = E4[:127].astype(np.float64)
    upper = np.searchsorted(positive, magnitude, side="left").clip(0, 126)
    lower = np.maximum(upper - 1, 0)
    upper_distance = positive[upper] - magnitude
    lower_distance = magnitude - positive[lower]
    use_lower = ((lower_distance < upper_distance)
                 | ((lower_distance == upper_distance) & ((lower & 1) == 0)))
    code = np.where(use_lower, lower, upper).astype(np.uint8)
    code |= np.where(values < 0, 128, 0).astype(np.uint8)
    return code


def group_cases():
    count = 4096
    rng = np.random.default_rng(206013)
    a = rng.integers(0, 256, (count, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (count, 16), dtype=np.uint8)
    seeds = np.ldexp(rng.uniform(-2, 2, count).astype(np.float32), rng.integers(-40, 32, count))
    a[:8] = 0
    b[:8] = 0
    seeds[:8] = [0, -0., 2**-48, -3*2**-27, 65504**2, -(65504**2), 2**-14, -2**-14]
    a[8] = 0x77; b[8] = 0x77; seeds[8] = 0
    a[9] = 0x77; b[9] = 0xf7; seeds[9] = 0
    a[10] = 1; b[10] = 1; seeds[10] = 0
    a[11] = 1; b[11] = 129; seeds[11] = 0
    a[12] = [0x77, 0xf7] * 8; b[12] = 0x77; seeds[12] = 2**-17
    a[13] = 0; b[13] = 126; seeds[13] = -0.
    a[14] = 127; b[14] = 255; seeds[14] = 0
    a[15] = np.arange(16); b[15] = np.arange(128, 144); seeds[15] = 0
    record = np.dtype([("a", "<f2", (16,)), ("b", "<f2", (16,)),
                       ("incoming", "<f4"), ("expected", "<f4")])
    records = np.empty(count, record)
    records["a"], records["b"], records["incoming"] = E4[a], E4[b], seeds
    for i in range(count):
        records["expected"][i] = evaluate(E4[a[i:i+1]], E4[b[i, :, None]], seeds[i])[0, 0]
    return records.tobytes()


def synthetic_files():
    result = {"f13-integer-groups.bin": group_cases()}
    rng = np.random.default_rng(206014)
    for number, (m, k, n, partition) in enumerate(SHAPES):
        ac = rng.integers(0, 256, (m, k), dtype=np.uint8)
        wc = rng.integers(0, 256, (k, n), dtype=np.uint8)
        if number & 1:
            ac = (ac & 0x80) | ((ac & 127) % 65)
            wc = (wc & 0x80) | ((wc & 127) % 65)
        a, w = E4[ac], E4[wc]
        residual = (rng.normal(size=(m, n)) * .125).astype("<f2")
        scale = rng.normal(size=n).astype("<f2")
        residual[0, 0], scale[0] = -2**-24, 2**-24
        seed = np.multiply(residual.astype(np.float32), scale.astype(np.float32), dtype=np.float32)
        expected = evaluate(a, w, seed, partition)
        result[f"f13-integer-gemm-{number:02d}.bin"] = (struct.pack("<4I", m, k, n, partition)
            + b"".join(x.tobytes() for x in (a, w, residual, scale, expected.astype("<f4"),
                                           half_rtz(expected), publish_e4(expected))))
    return result


def verify(root):
    root = Path(root)
    files = []
    for name, expected in PINNED.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned reference hash differs: {name}")
        files.append({"file": name, "sha256": actual, "verification": "immutable reference hash"})
    original = json.loads((root / "f13/report.json").read_text(encoding="utf8"))
    expected = {row["file"]: row["sha256"] for row in original["files"]}
    generated = synthetic_files()
    if set(expected) != set(generated):
        raise ValueError("Synthetic inventory differs")
    for name, raw in generated.items():
        actual = (root / "f13" / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if raw != actual or digest != expected[name]:
            raise ValueError(f"Independent regeneration differs: {name}")
        files.append({"file": f"f13/{name}", "bytes": len(raw), "sha256": digest,
                      "verification": "independent dyadic CPU regeneration"})
    return {"status": "passed", "gpu_used": False, "files": files,
            "scope": "Numeric fixture integrity and synthetic arithmetic only; no model or image-quality parity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--verify", type=Path, help="Existing tests/fixtures directory")
    choice.add_argument("--out", type=Path, help="Fresh directory for seven synthetic F13 files")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        rows = []
        for name, raw in synthetic_files().items():
            (args.out / name).write_bytes(raw)
            rows.append({"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        result = {"status": "generated", "gpu_used": False, "files": rows}
        (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
