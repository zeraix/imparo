from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
BUILD_PATH = ROOT / "tools" / "triton-kernel-lab" / "build_q4_q8_mmq_variants.py"
SPEC = importlib.util.spec_from_file_location(
    "imparo_triton_lab_q4_variant_build", BUILD_PATH
)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAB
SPEC.loader.exec_module(LAB)
from imparo_triton_pack.aot import CompileResult


def minimal_cuda_elf(sm: int) -> bytes:
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    data[7] = 51
    struct.pack_into("<H", data, 18, 190)
    struct.pack_into("<I", data, 48, sm)
    return bytes(data)


class FakeCompiler:
    global_scratch_bytes = 0
    local_memory_bytes = 0

    def __init__(self, expected_version: str) -> None:
        self.expected_version = expected_version

    def compile(self, source: Path, spec: object, target: str) -> object:
        del source, target
        return CompileResult(
            cubin=minimal_cuda_elf(86),
            symbol=spec.kernel_name,
            registers_per_thread=96 if spec.num_warps == 8 else 127,
            static_shared_bytes=16384,
            local_memory_bytes=self.local_memory_bytes,
            global_scratch_bytes=self.global_scratch_bytes,
            profile_scratch_bytes=0,
        )


class Q4Q8MmqVariantLabTests(unittest.TestCase):
    def test_matrix_is_bounded_and_opaque(self) -> None:
        self.assertEqual(LAB.TARGET, "cuda:86:32")
        self.assertEqual(len(LAB.VARIANTS), 4)
        self.assertEqual(len(LAB.SHAPES), 2)
        expected = {
            (128, 64, 8, 2),
            (64, 128, 8, 2),
            (64, 64, 8, 2),
            (64, 64, 8, 1),
        }
        self.assertEqual(
            {
                (item.block_m, item.block_n, item.num_warps, item.num_stages)
                for item in LAB.VARIANTS
            },
            expected,
        )
        LAB.verify_matrix()
        symbols = [symbol for shape in LAB.SHAPES for symbol in shape.symbols]
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertTrue(all(symbol.startswith("ip_") and len(symbol) == 67 for symbol in symbols))

    def test_sources_use_element_derived_native_tile_and_keep_fold_order(self) -> None:
        for shape in LAB.SHAPES:
            source = BUILD_PATH.with_name(shape.source).read_text(encoding="utf-8")
            self.assertIn("native_row_tile = (program_row * BLOCK_M) // 128", source)
            self.assertIn("native_token_tile = (program_token * BLOCK_N) // 128", source)
            self.assertIn("tl.where(seam != 0, suffix + prefix, prefix)", source)
            self.assertIn("for block in tl.range", source)
            self.assertNotIn("program_row // 2", source)
            self.assertNotIn("program_token // 2", source)
            self.assertNotIn("torch", source)

    def test_reqntid_is_variant_specific_but_kparam_is_frozen(self) -> None:
        entries = "\n".join(
            f"Value: Index : 0x0 Ordinal : 0x{o:x} Offset : 0x{off:x} Size : 0x{s:x}"
            for o, off, s in reversed(LAB.EXPECTED_KPARAMS)
        )
        dump = (
            ".nv.info.symbol\nAttribute: EIATTR_CBANK_PARAM_SIZE\nValue: 0x48\n"
            f"{entries}\nAttribute: EIATTR_REQNTID\nValue: 0x100 0x1 0x1\n"
            ".nv.callgraph\n"
        )
        self.assertEqual(
            LAB.parse_lab_abi_dump(dump, "symbol", 256), LAB.EXPECTED_KPARAMS
        )
        with self.assertRaisesRegex(LAB.BuildError, "256x1x1"):
            LAB.parse_lab_abi_dump(dump.replace("0x100", "0x80"), "symbol", 256)

    def test_metadata_has_eight_surviving_modules_and_runtime_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            LAB, "TritonCompiler", FakeCompiler
        ), mock.patch.object(LAB, "verify_lab_launch_abi"):
            metadata = LAB.build(Path(directory) / "run", "sha256:local")
        self.assertEqual(metadata["matrix"]["requested_modules"], 8)
        self.assertEqual(metadata["matrix"]["surviving_modules"], 8)
        self.assertEqual(metadata["matrix"]["rejected_modules"], 0)
        self.assertEqual(len(metadata["survivors"]), 8)
        self.assertEqual([item["ordinal"] for item in metadata["launch_abi"]], list(range(8)))
        self.assertEqual([item["ordinal"] for item in metadata["hidden_launch_abi"]], [8, 9])
        self.assertEqual({item["block"][0] for item in metadata["survivors"]}, {256})
        self.assertEqual(len({item["config_id"] for item in metadata["survivors"]}), 8)

    def test_local_or_stack_spills_are_marked_and_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            LAB, "TritonCompiler", FakeCompiler
        ), mock.patch.object(LAB, "verify_lab_launch_abi"), mock.patch.object(
            FakeCompiler, "local_memory_bytes", 16
        ):
            with self.assertRaisesRegex(LAB.BuildError, "all bounded"):
                LAB.build(Path(directory) / "run", "sha256:local")

    def test_nonzero_hidden_scratch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            LAB, "TritonCompiler", FakeCompiler
        ), mock.patch.object(LAB, "verify_lab_launch_abi"), mock.patch.object(
            FakeCompiler, "global_scratch_bytes", 1
        ):
            with self.assertRaisesRegex(LAB.BuildError, "non-zero hidden scratch"):
                LAB.build(Path(directory) / "run", "sha256:local")


if __name__ == "__main__":
    unittest.main()
