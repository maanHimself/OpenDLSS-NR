"""Independent CPU regeneration and literal numeric publication witnesses."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_numeric_fixtures", ROOT / "tools/generate_numeric_fixtures.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class NumericFixtureTests(unittest.TestCase):
    def test_all_synthetic_fixtures_regenerate_byte_for_byte(self):
        report = oracle.verify(ROOT / "tests/fixtures")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["files"]), 9)
        self.assertFalse(report["gpu_used"])

    def test_group_and_partition_literal_witnesses(self):
        a = np.zeros((1, 32), "<f2")
        w = np.zeros((32, 1), "<f2")
        a[0, 0] = w[0, 0] = 1
        # The incoming residual is below the common F13 lattice and is lost.
        self.assertEqual(float(oracle.evaluate(a[:, :16], w[:16], 2**-20)[0, 0]), 1.0)
        # Retained group accumulation and the left fold of independent parts
        # differ: the second group's small product is lost with incoming one.
        a[0, 16] = w[16, 0] = 2**-9
        self.assertEqual(float(oracle.evaluate(a, w)[0, 0]), 1.0)
        self.assertEqual(float(oracle.evaluate(a, w, partition=16)[0, 0]), 1 + 2**-18)
        self.assertEqual(float(oracle.evaluate(a[:, :16] * 0, w[:16], -0.0)[0, 0]), 0)

    def test_independent_publication_midpoints_and_signed_underflow(self):
        values = np.array([1.0625, 1.1875, -1.0625, -2**-20, -0., 448, 1000], np.float32)
        np.testing.assert_array_equal(oracle.publish_e4(values), [56, 58, 184, 128, 0, 126, 126])
        raw = oracle.half_rtz(np.array([65536, -65536, 2**-25, -2**-25, 1 + 2**-11], np.float32))
        np.testing.assert_array_equal(raw.view(np.uint16), [0x7bff, 0xfbff, 0, 0x8000, 0x3c00])


if __name__ == "__main__":
    unittest.main()
