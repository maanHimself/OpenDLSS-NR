"""Padded-token capture/replay smoke for Compute Sanitizer; needs model and SM75.

Checks replay determinism with boundary recording on/off, not eager or reference parity.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda.backend import Backend
from opendlss_cuda.geometry import geometry_from_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    for width, height in ((512, 512), (1280, 720)):
        geometry = geometry_from_valid(width, height)
        assert geometry["vit_tokens"] < geometry["padded_vit_tokens"]
        features = np.random.default_rng(0).uniform(
            -.05, .05, (geometry["full_height"], geometry["full_width"], 16)).astype("float32")
        expected = None
        for capture in (False, True):
            with Backend(args.model, args.library, width, height, capture_boundaries=capture) as backend:
                for _ in range(2):
                    head = backend.run_features(features)
                    if expected is None:
                        expected = head.copy()
                    if not np.array_equal(head.view("uint32"), expected.view("uint32")):
                        raise RuntimeError("Boundary-recording/replayed head differs")
                    if capture and len(backend.read_boundaries()) != 75:
                        raise RuntimeError("Missing graph boundaries")
        print(f"{width}x{height}: padded tokens, capture and replay completed", flush=True)


if __name__ == "__main__":
    main()
