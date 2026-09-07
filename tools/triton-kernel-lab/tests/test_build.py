from __future__ import annotations

import importlib.util
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUILD_PATH = ROOT / "tools" / "triton-kernel-lab" / "build.py"
SPEC = importlib.util.spec_from_file_location("imparo_triton_lab_build", BUILD_PATH)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


def minimal_cuda_elf(sm: int) -> bytes:
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    data[7] = 51
    struct.pack_into("<H", data, 18, 190)
    struct.pack_into("<I", data, 48, sm)
    return bytes(data)


class KernelLabBuildTests(unittest.TestCase):
    def test_candidate_surface_is_bounded_and_opaque(self) -> None:
        self.assertEqual(LAB.TARGET, "cuda:86:32")
        self.assertEqual((LAB.WIDTH, LAB.N_TOK), (2560, 512))
        self.assertEqual([item[2] for item in LAB.VARIANTS], [4, 8])
        source = BUILD_PATH.with_name("rms_q8.py").read_text(encoding="utf-8")
        for _, symbol, _ in LAB.VARIANTS:
            self.assertRegex(symbol, r"^ip_[0-9a-f]{64}$")
            self.assertIn(f"def {symbol}(", source)
        self.assertNotIn("torch", source)

    def test_lab_cubin_gate_requires_exact_sm86_cuda_elf(self) -> None:
        LAB.verify_lab_cubin(minimal_cuda_elf(86), 86)
        with self.assertRaisesRegex(LAB.BuildError, "real-SM 80"):
            LAB.verify_lab_cubin(minimal_cuda_elf(80), 86)
        wrong_machine = bytearray(minimal_cuda_elf(86))
        struct.pack_into("<H", wrong_machine, 18, 62)
        with self.assertRaisesRegex(LAB.BuildError, "EM_CUDA"):
            LAB.verify_lab_cubin(bytes(wrong_machine), 86)

    def test_clean_builder_exports_pinned_cuda_tool_paths(self) -> None:
        dockerfile = (ROOT / "tools" / "triton-pack" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        for entry in (
            "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
            "TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump",
            "TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm",
            "TRITON_CUDACRT_PATH=/usr/local/cuda/include",
        ):
            self.assertIn(entry, dockerfile)

    def test_zero_sized_hidden_scratch_parameters_remain_in_launch_abi(self) -> None:
        source = BUILD_PATH.read_text(encoding="utf-8")
        self.assertIn('"ordinal": 6', source)
        self.assertIn('"name": "global_scratch"', source)
        self.assertIn('"ordinal": 7', source)
        self.assertIn('"name": "profile_scratch"', source)
        self.assertEqual(source.count('"argument_required": True'), 2)


if __name__ == "__main__":
    unittest.main()
