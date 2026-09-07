from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/imparo-model/src/bin/imparo-forward.rs"


class PhaseA1PrefillWallSourceTests(unittest.TestCase):
    def test_instrument_is_exact_opt_in_and_reuses_single_elapsed_value(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"IMPARO_CUDA_PHASE_A1_PREFILL_WALL"', source)
        self.assertIn('.as_deref() == Ok("1")', source)
        self.assertEqual(source.count("let prefill_wall = t1.elapsed();"), 1)
        self.assertNotIn("IMPARO_PROF", source[source.index(
            "fn phase_a1_prefill_wall_line"
        ):source.index("fn main()")])
        self.assertIn("production_authority", source)
        self.assertIn("lab_only", source)

    def test_disabled_helper_behavior_is_covered_in_rust(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("absent_opt_in_produces_no_line", source)
        self.assertIn("assert!(line.is_none())", source)
        self.assertIn("opt_in_line_is_strict_lab_only_json", source)
        self.assertIn("assert!(!line.contains('\\n'))", source)


if __name__ == "__main__":
    unittest.main()
