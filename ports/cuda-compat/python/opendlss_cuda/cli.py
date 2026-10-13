"""Convert weights, run prepared features and check parity.

The output is an RGB residual plus blend logit, not a composed image.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from . import ARITHMETIC, __version__


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def package_identity():
    root = Path(__file__).resolve().parent
    files = {p.name: sha_file(p) for p in sorted(root.glob("*.py"))}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"version": __version__, "python_source_sha256": hashlib.sha256(encoded).hexdigest(),
            "python_files": files, "python": platform.python_version(), "numpy": np.__version__}


def write_report(directory, report):
    temporary = directory / "report.json.partial"
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(directory / "report.json")


def load_features(path, geometry):
    """Check the NPY header and file bounds before mapping any array payload."""
    expected = (geometry["full_height"], geometry["full_width"], 16)
    path = Path(path)
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Features require NPY version 1 or 2")
        if shape != expected or dtype != np.dtype("float32") or fortran:
            raise ValueError(f"Features require C-order float32 shape {expected}")
        size = math.prod(shape) * dtype.itemsize
        if path.stat().st_size != stream.tell() + size:
            raise ValueError("Feature file has truncated or trailing payload")
        offset = stream.tell()
    features = np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset)
    if not np.isfinite(features).all() or np.any(np.abs(features) > 65504):
        raise ValueError("Features must be finite and within finite FP16 range")
    return features


def _geometry(args):
    from .geometry import geometry_from_valid, require_supported_geometry
    return require_supported_geometry(geometry_from_valid(args.width, args.height, args.geometry))


def _receipt(args, backend, features):
    return {
        "arithmetic": ARITHMETIC,
        "device": backend.device_info,
        "context_options": backend.context_options,
        "geometry": backend.geometry,
        "library_sha256": backend.runtime.library_sha256,
        "native_source_sha256": backend.runtime.source_sha256,
        "native_build": backend.runtime.build_info,
        "model": backend.model.identity,
        "features_file_sha256": sha_file(args.features),
        "features_array_sha256": sha_array(features),
    }


def _run(args):
    from .backend import Backend
    geometry = _geometry(args)
    features = load_features(args.features, geometry)
    created = time.perf_counter()
    with Backend(args.model, args.library, args.width, args.height, geometry=args.geometry) as backend:
        setup_seconds = time.perf_counter() - created
        receipt = _receipt(args, backend, features)
        head = backend.run_features(features)
        metrics = dict(backend.last_metrics)
        partial = args.out / "head.npy.partial"
        with partial.open("xb") as stream:
            np.save(stream, head, allow_pickle=False)
        partial.replace(args.out / "head.npy")
        return dict(receipt, status="completed", setup_seconds=setup_seconds,
                    metrics=metrics, head={"file": "head.npy", "shape": list(head.shape),
                    "dtype": str(head.dtype), "file_sha256": sha_file(args.out / "head.npy"),
                    "array_sha256": sha_array(head)})


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-model", help="Convert user-supplied weights to a fresh FP16 cache")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    info = commands.add_parser("info", help="Load the matching library and report its SM75 device")
    info.add_argument("--library", type=Path, required=True)
    command = commands.add_parser("run-features", help="Run prepared features")
    command.add_argument("--model", type=Path, required=True)
    command.add_argument("--library", type=Path, required=True)
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--width", type=_positive, required=True)
    command.add_argument("--height", type=_positive, required=True)
    command.add_argument("--geometry", choices=("upstream", "legacy_belvedere"), default="upstream")
    command.add_argument("--out", type=Path, required=True)
    parity = commands.add_parser("parity", help="Require the fixture's exact head and all declared boundaries")
    parity.add_argument("--model", type=Path, required=True)
    parity.add_argument("--library", type=Path, required=True)
    parity.add_argument("--fixture", type=Path, required=True)
    parity.add_argument("--replays", type=_positive, default=3)
    parity.add_argument("--out", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    directory = None
    report = {"format": "opendlss-cuda-command-v1", "command": args.command,
              "started_utc": datetime.now(timezone.utc).isoformat(), "package": package_identity()}
    try:
        if args.command == "export-model":
            from .export_model import export_model
            exported = export_model(args.model, args.out)
            report.update(status="completed", model={"format": exported["format"],
                          "arrays": len(exported["arrays"]), "blob": exported["blob"]})
        elif args.command == "info":
            from .runtime import Runtime
            with Runtime(args.library) as runtime:
                report.update(status="completed", arithmetic=ARITHMETIC, device=runtime.device_info,
                              context_options=runtime.context_options, library_sha256=runtime.library_sha256,
                              native_source_sha256=runtime.source_sha256, native_build=runtime.build_info)
        else:
            args.out.mkdir(parents=True, exist_ok=False)
            directory = args.out
            if args.command == "parity":
                from .fixture import run_parity
                report.update(run_parity(model=args.model, library=args.library,
                                         fixture=args.fixture, replays=args.replays))
            else:
                report.update(_run(args))
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        exit_code = {"completed": 0, "passed": 0, "mismatch": 1, "unavailable": 3}[report["status"]]
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        status = "invalid" if isinstance(error, (ValueError, KeyError, FileNotFoundError, FileExistsError)) else "runtime_error"
        report.update(status=status, error_type=type(error).__name__, error=str(error))
        exit_code = 2
    if directory is not None:
        try:
            write_report(directory, report)
        except OSError as error:
            write_error = {"error_type": type(error).__name__, "error": str(error)}
            if exit_code == 2:
                report["report_write_error"] = write_error
            else:
                report.update(status="runtime_error", **write_error)
            exit_code = 2
    print(json.dumps(report, indent=2, allow_nan=False), file=sys.stderr if exit_code == 2 else sys.stdout)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
