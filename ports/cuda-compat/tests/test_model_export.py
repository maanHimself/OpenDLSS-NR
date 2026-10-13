"""Independent fragment fixtures and structural checks; no model/GPU required."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from opendlss_cuda import export_model as ex
from opendlss_cuda.geometry import block_schedule, geometry_from_valid


def pack_fp8_fragment_fixture(k_count, n_count):
    """Pack by lane/fragment enumeration, not the decoder's address formula."""
    packed = bytearray(k_count * n_count)
    expected = np.empty((k_count, n_count), dtype=np.uint8)
    for kt in range(k_count // 32):
        for ng in range(n_count // 16):
            for lane in range(32):
                for fragment in range(16):
                    k = kt * 32 + ((fragment // 4) % 2) * 16 + (lane % 4) * 4 + fragment % 4
                    n = ng * 16 + (fragment // 8) * 8 + lane // 4
                    # The forward activation permutation independently rotates
                    # original k bits to the natural input row associated with k.
                    natural_k = (k & ~14) | ((k & 2) << 2) | ((k & 4) >> 1) | ((k & 8) >> 1)
                    code = (natural_k * 53 + n * 17 + 7) % 256
                    packed[kt * n_count * 32 + ng * 512 + lane * 16 + fragment] = code
                    expected[natural_k, n] = code
    return packed, expected


def pack_half_fragment_fixture(k_count, n_count):
    packed = np.zeros((k_count // 16) * ((n_count + 15) // 16) * 256, dtype="<u2")
    expected = np.zeros((k_count, n_count), dtype="<u2")
    for kt in range(k_count // 16):
        for nt in range((n_count + 15) // 16):
            for lane in range(32):
                for frag in range(8):
                    k = kt * 16 + (lane % 4) * 2 + ((frag % 4) // 2) * 8 + frag % 2
                    n = nt * 16 + lane // 4 + (frag // 4) * 8
                    bits = (k * 37 + n * 113) & 0xffff
                    packed[(kt * ((n_count + 15) // 16) + nt) * 256 + lane * 8 + frag] = bits
                    if n < n_count:
                        expected[k, n] = bits
    return packed.tobytes(), expected


class ModelExportTests(unittest.TestCase):
    def test_fp8_decode_lane_fixture(self):
        for batches, k, n in [(1, 32, 16), (1, 64, 96), (1, 96, 256), (3, 64, 128)]:
            packed, expected_codes = pack_fp8_fragment_fixture(batches * k, n)
            actual, count = ex.decode_fp8(packed, 0, k, n, batches)
            expected = ex.FP8[expected_codes].reshape(actual.shape)
            np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"))
            self.assertEqual(count, int(np.count_nonzero((expected_codes & 127) == 127)))

    def test_fp8_scalar_values_and_signed_zero(self):
        self.assertEqual(float(ex.FP8[1]), 2 ** -9)
        self.assertEqual(float(ex.FP8[7]), 7 * 2 ** -9)
        self.assertEqual(float(ex.FP8[8]), 2 ** -6)
        self.assertEqual(float(ex.FP8[0x38]), 1)
        self.assertEqual(float(ex.FP8[0x7e]), 448)
        self.assertEqual(float(ex.FP8[0xfe]), -448)
        self.assertEqual(int(ex.FP8.view("<u2")[128]), 0x8000)
        self.assertEqual(int(ex.FP8.view("<u2")[127]), 0)
        self.assertEqual(int(ex.FP8.view("<u2")[255]), 0)

    def test_f16_adapter_and_head_bit_patterns(self):
        for k, n in [(16, 32), (32, 4)]:
            packed, expected = pack_half_fragment_fixture(k, n)
            actual = ex.decode_f16(packed, 0, k, n)
            np.testing.assert_array_equal(actual.view("<u2"), expected)

    def test_prior_lane_fixture_and_token_axes(self):
        # A prior is an accumulator fragment: row uses the lane high bits,
        # column uses its low bits, opposite roles from the weight fixture.
        packed = np.zeros(2 * 4096, dtype="<u2")
        expected = np.empty((2, 64, 64), dtype="<u2")
        for head in range(2):
            for qt in range(4):
                for kt in range(4):
                    for lane in range(32):
                        for frag in range(8):
                            physical_q = qt * 16 + lane // 4 + ((frag % 4) // 2) * 8
                            physical_k = kt * 16 + (lane % 4) * 2 + (frag // 4) * 8 + frag % 2
                            tile, within = divmod(physical_q, 16)
                            x, y = (tile % 2) * 4 + within % 4, (tile // 2) * 4 + within // 4
                            natural_q = y * 8 + x
                            bits = head * 8192 + natural_q * 64 + physical_k
                            packed[head * 4096 + (qt * 4 + kt) * 256 + lane * 8 + frag] = bits
                            expected[head, natural_q, physical_k] = bits
        actual = ex.decode_prior(packed.tobytes(), 0, 2)
        np.testing.assert_array_equal(actual.view("<u2"), expected)

    def test_bounds_and_invalid_shapes(self):
        for call in [lambda: ex.decode_fp8(bytes(32), 0, 32, 16),
                     lambda: ex.decode_fp8(bytes(1024), 0, 16, 16),
                     lambda: ex.decode_f16(bytes(1024), 1, 16, 32),
                     lambda: ex.decode_prior(bytes(8191), 0, 1),
                     lambda: ex.checked_slice(bytes(4), -1, 2)]:
            with self.assertRaises(ValueError):
                call()

    def test_complete_record_layout_and_schedule(self):
        sizes = ex.expected_tensor_lengths()
        self.assertEqual(len(sizes), 153)
        self.assertEqual(sum(sizes.values()), 147683778)
        specs = ex.array_specs()
        self.assertEqual(len(specs), 640)
        self.assertEqual(len({s["tensor"] for s in specs}), 145)
        self.assertEqual(sizes["block23.layer2.layer"], 917568)
        self.assertEqual(sizes["block30.layer4.layer"], 524304)
        self.assertEqual(sizes["block31.layer0.layer"], 4194320)
        self.assertEqual(sizes["block39.layer0.layer"], 525312)
        blocks = block_schedule()
        self.assertEqual([b["block"] for b in blocks], list(range(71)))
        self.assertEqual(blocks[56]["phase"], 2)
        self.assertEqual(blocks[70]["phase"], 1)
        self.assertEqual(blocks[48]["before"]["skip_block"], 22)
        for spec in specs:
            self.assertLessEqual(spec["byte_offset"] + ex.source_nbytes(spec), sizes[spec["tensor"]])

    def test_upstream_native_capture_geometry(self):
        cases = [(512, 512, 576, 512, 12, 8), (768, 768, 832, 768, 16, 12),
                 (644, 768, 768, 768, 12, 12), (1920, 1080, 1920, 1152, 32, 20),
                 (3840, 2160, 3840, 2176, 60, 36)]
        for w, h, fw, fh, lw, lh in cases:
            g = geometry_from_valid(w, h)
            self.assertEqual((g["full_width"], g["full_height"]), (fw, fh))
            self.assertEqual((g["levels"][5]["width"], g["levels"][5]["height"]), (lw, lh))
            self.assertEqual(g["padded_vit_tokens"] % 64, 0)
        with self.assertRaises(ValueError):
            geometry_from_valid(0, 100)
        with self.assertRaises(ValueError):
            geometry_from_valid(8, 8)

    def test_legacy_geometry_and_override_validation(self):
        g = geometry_from_valid(512, 512, "legacy_belvedere")
        self.assertEqual((g["full_width"], g["full_height"]), (512, 512))
        g = geometry_from_valid(1280, 720, "legacy_belvedere")
        self.assertEqual((g["full_width"], g["full_height"]), (1280, 768))
        with self.assertRaises(ValueError):
            geometry_from_valid(1280, 720, "legacy_belvedere", full_width=1281)


if __name__ == "__main__":
    unittest.main()
