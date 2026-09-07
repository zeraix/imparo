from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
BUILD_PATH = ROOT / "tools" / "triton-kernel-lab" / "build_q4_q8_mmq.py"
SPEC = importlib.util.spec_from_file_location("imparo_triton_lab_q4_build", BUILD_PATH)
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


def stream_seam(tile: int, native_tiles: int, blocks: int, grid: int) -> int:
    worker = max((tile * grid + native_tiles - 1) // native_tiles, 1)
    boundary = worker * (native_tiles * blocks) // grid
    boundary -= (boundary % blocks) % 8
    if worker < grid and boundary // blocks == tile:
        return boundary % blocks
    return 0


class FakeCompiler:
    global_scratch_bytes = 0

    def __init__(self, expected_version: str) -> None:
        self.expected_version = expected_version

    def compile(self, source: Path, spec: object, target: str) -> object:
        del source, target
        return CompileResult(
            cubin=minimal_cuda_elf(86),
            symbol=spec.kernel_name,
            registers_per_thread=127,
            static_shared_bytes=16384,
            local_memory_bytes=0,
            global_scratch_bytes=self.global_scratch_bytes,
            profile_scratch_bytes=0,
        )


class Q4Q8MmqLabTests(unittest.TestCase):
    def test_candidate_surface_is_two_exact_shapes_and_one_variant(self) -> None:
        self.assertEqual(LAB.TARGET, "cuda:86:32")
        self.assertEqual(
            [(shape.n_in, shape.n_out) for shape in LAB.SHAPES],
            [(2560, 10240), (10240, 2560)],
        )
        self.assertEqual(len(LAB.SIGNATURE), 8)
        for shape in LAB.SHAPES:
            self.assertRegex(shape.symbol, r"^ip_[0-9a-f]{64}$")
            source = BUILD_PATH.with_name(shape.source).read_text(encoding="utf-8")
            self.assertIn(f"def {shape.symbol}(", source)
            self.assertIn("for block in tl.range", source)
            self.assertIn("mask=token_valid", source)
            self.assertIn("native_ntx = (n_tok + 127) // 128", source)
            self.assertNotIn("torch", source)

    def test_q4_q8_layout_and_launch_abi_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(LAB, "TritonCompiler", FakeCompiler), mock.patch.object(
                LAB, "verify_lab_launch_abi"
            ):
                metadata = LAB.build(Path(directory) / "run", "sha256:local")
        launch = metadata["launch_abi"]
        hidden = metadata["hidden_launch_abi"]
        self.assertEqual([item["ordinal"] for item in launch], list(range(8)))
        self.assertEqual([item["ordinal"] for item in hidden], [8, 9])
        self.assertEqual(launch[0]["alias"], "weights+2")
        self.assertEqual(launch[0]["alignment"], 2)
        self.assertEqual(launch[1]["alias"], "weights")
        self.assertEqual(launch[3]["alias"], "q8+128")
        self.assertEqual(launch[5]["supported_range"], [1, 512])
        self.assertEqual(metadata["weight_layout"]["record_bytes"], 18)
        self.assertEqual(metadata["activation_layout"]["record_bytes"], 144)
        self.assertTrue(
            metadata["native_control_by_shape"]["k10240-m2560-n512"][
                "numeric_seams"
            ]
        )

    def test_numeric_seam_matches_native_goldens(self) -> None:
        # Expansion uses one worker per native 128x128 tile: no split seam.
        self.assertTrue(
            all(stream_seam(tile, 320, 80, 320) == 0 for tile in range(320))
        )
        # Contraction, n_tok=512 and grid=30: the native route's two known seams.
        self.assertEqual(stream_seam(0 * 4 + 2, 80, 320, 30), 208)
        self.assertEqual(stream_seam(1 * 4 + 1, 80, 320, 30), 104)

    def test_real_cubin_parameter_dump_requires_visible8_plus_hidden2(self) -> None:
        entries = "\n".join(
            f"Value: Index : 0x0 Ordinal : 0x{o:x} Offset : 0x{off:x} Size : 0x{s:x}"
            for o, off, s in reversed(LAB.EXPECTED_KPARAMS)
        )
        dump = (
            f".nv.info.symbol\nAttribute: EIATTR_CBANK_PARAM_SIZE\nValue: 0x48\n"
            f"{entries}\nAttribute: EIATTR_REQNTID\nValue: 0x80 0x1 0x1\n"
            ".nv.callgraph\n"
        )
        self.assertEqual(LAB.parse_lab_abi_dump(dump, "symbol"), LAB.EXPECTED_KPARAMS)
        with self.assertRaisesRegex(LAB.BuildError, "parameter ABI differs"):
            LAB.parse_lab_abi_dump(dump.replace("Ordinal : 0x9", "Ordinal : 0xa"), "symbol")

    def test_nonzero_hidden_scratch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            LAB, "TritonCompiler", FakeCompiler
        ), mock.patch.object(LAB, "verify_lab_launch_abi"), mock.patch.object(
            FakeCompiler, "global_scratch_bytes", 1
        ):
            with self.assertRaisesRegex(LAB.BuildError, "non-zero hidden scratch"):
                LAB.build(Path(directory) / "run", "sha256:local")

    def test_metadata_is_lab_only(self) -> None:
        artifact = ROOT / "program-packs" / "lab" / "q4-q8-mmq-sm86" / "run-a" / "lab-metadata.json"
        if not artifact.exists():
            self.skipTest("real SM86 lab artifact not present")
        metadata = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(metadata["phase"], "A2")
        self.assertFalse(metadata["production_enabled"])
        self.assertEqual(len(metadata["variants"]), 2)


if __name__ == "__main__":
    unittest.main()
