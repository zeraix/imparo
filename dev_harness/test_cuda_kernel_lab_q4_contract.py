from __future__ import annotations

import pathlib
import struct
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
NATIVE = ROOT / "crates/imparo-cuda/native/sm80/mmq_q4_q8_1.cuh"
DESIGN = ROOT / "docs/evidence/triton/lab-a-q4-q8-windows-sm86-design.md"


def seam_for_tile(tile: int, *, blocks: int = 320, tiles: int = 80,
                  workers: int = 30) -> int:
    worker = max((tile * workers + tiles - 1) // tiles, 1)
    boundary = (worker * (tiles * blocks)) // workers
    boundary -= (boundary % blocks) % 8
    remainder = boundary % blocks
    if worker >= workers or boundary // blocks != tile or remainder == 0:
        return 0
    return remainder


class CudaKernelLabQ4ContractTests(unittest.TestCase):
    def test_native_controls_are_the_accepted_half_k_routes(self) -> None:
        native = NATIVE.read_text(encoding="utf-8")
        self.assertIn("q4_q8_1_full_tile<false, 4>", native)
        self.assertIn(
            "q4_q8_1_full_tile<false, 4, false, true, true>", native
        )
        self.assertIn("kHalfKSharedBytes", native)
        self.assertIn("FullTileVariant::Rows128K128", native)

    def test_visible_and_hidden_argument_contract_is_explicit(self) -> None:
        design = DESIGN.read_text(encoding="utf-8")
        self.assertIn(
            "w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, "
            "numeric_stream_grid",
            design,
        )
        self.assertIn("ten total host argument addresses", design)
        self.assertIn("w_qs = weights_base + 2", design)
        self.assertIn("x_d = q8_base + 128", design)
        self.assertIn("metadata/KPARAM mismatch", design)

    def test_q4_wire_nibble_and_scale_layout(self) -> None:
        record = bytearray(18)
        record[:2] = struct.pack("<e", 0.125)
        for lane in range(16):
            record[2 + lane] = lane | ((15 - lane) << 4)
        self.assertEqual(struct.unpack("<e", record[:2])[0], 0.125)
        for index in range(32):
            packed = record[2 + (index & 15)]
            q = packed & 0xF if index < 16 else packed >> 4
            expected = index if index < 16 else 31 - index
            self.assertEqual(q, expected)

    def test_contraction_seams_and_negative_neighbors(self) -> None:
        seam_208 = {2, 10, 18, 26, 34, 42, 50, 58, 66, 74}
        seam_104 = {5, 13, 21, 29, 37, 45, 53, 61, 69, 77}
        actual_208 = {tile for tile in range(80) if seam_for_tile(tile) == 208}
        actual_104 = {tile for tile in range(80) if seam_for_tile(tile) == 104}
        self.assertEqual(actual_208, seam_208)
        self.assertEqual(actual_104, seam_104)
        self.assertEqual(seam_for_tile(2), 208)
        self.assertEqual(seam_for_tile(5), 104)
        for neighbor in (1, 3, 4, 6):
            self.assertEqual(seam_for_tile(neighbor), 0)

    def test_q8_group_major_record_and_scale_alias(self) -> None:
        n_tok = 512
        for block32 in (0, 3, 4, 79, 319):
            for token in (0, n_tok - 1):
                record = (block32 // 4) * n_tok + token
                qs = record * 144 + (block32 % 4) * 32
                scale = record * 144 + 128 + (block32 % 4) * 4
                self.assertLess(qs + 31, record * 144 + 128)
                self.assertLessEqual(scale + 4, (record + 1) * 144)


if __name__ == "__main__":
    unittest.main()
