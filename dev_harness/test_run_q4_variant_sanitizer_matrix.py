from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "dev_harness/run_q4_variant_sanitizer_matrix.py"
SPEC = importlib.util.spec_from_file_location("q4_sanitizer_v2", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Q4VariantSanitizerMatrixTests(unittest.TestCase):
    def test_current_inputs_and_exact_cartesian_identity(self) -> None:
        policy = ROOT / "config/kernel-lab-policy.toml"
        metadata = (
            ROOT
            / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json"
        )
        _, document = MODULE.load_inputs(policy, metadata)
        pairs = MODULE.build_pairs(document, metadata)
        self.assertEqual([pair["label"] for pair in pairs], list(MODULE.LABELS))
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(MODULE.exact_run_ids(pairs)), 16)
        self.assertEqual(len(set(MODULE.exact_run_ids(pairs))), 16)
        for pair in pairs:
            arguments = MODULE.harness_args(pair)
            self.assertEqual(arguments[4:7], ["1", "1", "1"])
            self.assertEqual(len(arguments), 15)
            self.assertEqual(arguments[9], "256")
            self.assertEqual(arguments[13], "256")

    def test_tool_specific_summaries_fail_closed(self) -> None:
        header = "========= COMPUTE-SANITIZER\n"
        zero = "========= ERROR SUMMARY: 0 errors\n"
        leak = "========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
        self.assertTrue(
            MODULE.sanitizer_summary("memcheck", header + leak + zero)["passed"]
        )
        self.assertFalse(
            MODULE.sanitizer_summary("memcheck", header + zero)["passed"]
        )
        self.assertFalse(
            MODULE.sanitizer_summary(
                "memcheck", header + leak + "========= ERROR SUMMARY: 1 errors\n"
            )["passed"]
        )
        race = (
            header
            + "========= RACECHECK SUMMARY: 0 hazards displayed "
            "(0 errors, 0 warnings)\n"
        )
        self.assertTrue(MODULE.sanitizer_summary("racecheck", race)["passed"])
        self.assertFalse(
            MODULE.sanitizer_summary(
                "racecheck",
                header
                + "========= RACECHECK SUMMARY: 1 hazards displayed "
                "(0 errors, 1 warnings)\n",
            )["passed"]
        )
        self.assertFalse(
            MODULE.sanitizer_summary("synccheck", header + zero + zero)["passed"]
        )
        self.assertTrue(MODULE.sanitizer_summary("initcheck", header + zero)["passed"])
        self.assertFalse(
            MODULE.sanitizer_summary(
                "initcheck", header + "========= WARNING: unexpected\n" + zero
            )["passed"]
        )

    def test_mixed_process_streams_preserve_one_json_and_diagnostics(self) -> None:
        raw = (
            "========= COMPUTE-SANITIZER\n"
            '{"schema":1,"structural_ok":true}\n'
            "========= ERROR SUMMARY: 0 errors\n"
        )
        harness, diagnostics, normalization = MODULE.split_process_streams(raw, "")
        self.assertEqual(harness, '{"schema":1,"structural_ok":true}\n')
        self.assertEqual(
            diagnostics,
            "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n",
        )
        self.assertEqual(normalization["harness_json_objects"], 1)
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.split_process_streams(raw + '{"schema":2}\n', "")

    def test_cuobjdump_parser_rejects_reqntid_and_kparam_mutations(self) -> None:
        symbol = "ip_test"
        parameters = []
        for ordinal, (offset, size) in enumerate(
            zip(MODULE.EXPECTED_KPARAM_OFFSETS, MODULE.EXPECTED_KPARAM_SIZES)
        ):
            parameters.append(
                "\tAttribute:\tEIATTR_KPARAM_INFO\n"
                f"\tValue: Ordinal : 0x{ordinal:x} Offset : 0x{offset:x} Size : 0x{size:x}\n"
            )
        fixture = (
            "64bit elf: type=EXEC, abi=7, sm=86, toolkit=129, flags = 0x560534\n"
            ".nv.info\n"
            "\tAttribute: EIATTR_REGCOUNT\n"
            f"\tValue: function: {symbol}(0xa) register count: 155\n"
            "\tAttribute: EIATTR_FRAME_SIZE\n"
            f"\tValue: function: {symbol}(0xa) frame size: 0x0\n\n"
            f".nv.info.{symbol}\n"
            "\tAttribute: EIATTR_CBANK_PARAM_SIZE\n\tValue: 0x48\n"
            + "".join(parameters)
            + "\tAttribute: EIATTR_REQNTID\n\tValue: 0x100 0x1 0x1\n"
            "\n.nv.callgraph\n<0,-1>\n\n"
            f".text.{symbol}\nbar = 0 reg = 155 lmem = 0 smem = 0\n"
        )
        parsed = MODULE.parse_cuobjdump(
            fixture, symbol=symbol, threads=256, registers=155
        )
        self.assertTrue(parsed["passed"])
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.parse_cuobjdump(
                fixture.replace("0x100 0x1", "0x80 0x1"),
                symbol=symbol,
                threads=256,
                registers=155,
            )
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.parse_cuobjdump(
                fixture.replace("Size : 0x8", "Size : 0x4", 1),
                symbol=symbol,
                threads=256,
                registers=155,
            )
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.parse_cuobjdump(
                fixture
                + "\n.nv.info.other_kernel\n"
                + "Attribute: EIATTR_CBANK_PARAM_SIZE\nValue: 0x48\n",
                symbol=symbol,
                threads=256,
                registers=155,
            )

    def test_atomic_manifest_replaces_without_temporary_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "manifest.json"
            MODULE.atomic_json(target, {"schema": 2, "all_pass": False})
            self.assertEqual(json.loads(target.read_text())["schema"], 2)
            MODULE.atomic_json(target, {"schema": 2, "all_pass": True})
            self.assertTrue(json.loads(target.read_text())["all_pass"])
            self.assertEqual(list(root.glob("tmp*")), [])

    def test_final_manifest_is_withheld_for_any_incomplete_or_failed_run(self) -> None:
        expected = [f"run-{index}" for index in range(16)]
        modules = [{"passed": True} for _ in range(8)]
        with tempfile.TemporaryDirectory() as temporary:
            final = pathlib.Path(temporary) / "sanitizer-evidence-v2.json"
            incomplete = {
                "modules": modules,
                "runs": [{"run_id": name, "pass": True} for name in expected[:-1]],
            }
            with self.assertRaises(MODULE.EvidenceError):
                MODULE.write_final(final, incomplete, expected)
            self.assertFalse(final.exists())
            failed = {
                "modules": modules,
                "runs": [
                    {"run_id": name, "pass": index != 7}
                    for index, name in enumerate(expected)
                ],
            }
            with self.assertRaises(MODULE.EvidenceError):
                MODULE.write_final(final, failed, expected)
            self.assertFalse(final.exists())
            passed = {
                "schema": 2,
                "modules": modules,
                "runs": [{"run_id": name, "pass": True} for name in expected],
            }
            MODULE.write_final(final, passed, expected)
            document = json.loads(final.read_text(encoding="utf-8"))
            self.assertTrue(document["complete"])
            self.assertTrue(document["all_pass"])

    def test_environment_tool_discovery_has_no_host_literal_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cuda = root / "cuda"
            sanitizer = cuda / "compute-sanitizer" / "compute-sanitizer.exe"
            cuobjdump = cuda / "bin" / "cuobjdump.exe"
            system_root = root / "windows"
            smi = system_root / "System32" / "nvidia-smi.exe"
            for path in (sanitizer, cuobjdump, smi):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            discovered = MODULE.discover_tool_paths(
                {
                    "CUDA_PATH": str(cuda),
                    "SystemRoot": str(system_root),
                    "PATH": "",
                }
            )
            self.assertEqual(discovered["compute_sanitizer"], sanitizer.resolve())
            self.assertEqual(discovered["cuobjdump"], cuobjdump.resolve())
            self.assertEqual(discovered["nvidia_smi"], smi.resolve())
            missing = MODULE.discover_tool_paths({"PATH": ""})
            self.assertEqual(
                missing,
                {
                    "compute_sanitizer": None,
                    "cuobjdump": None,
                    "nvidia_smi": None,
                },
            )

    def test_harness_stdout_binds_actual_geometry(self) -> None:
        pair = {
            "expansion": {
                "shape": "k2560-m10240",
                "block_m": 128,
                "block_n": 64,
                "threads": 256,
                "dynamic_shared_bytes": 32768,
                "registers_per_thread": 155,
                "module_bytes": 64032,
            },
            "contraction": {
                "shape": "k10240-m2560",
                "block_m": 128,
                "block_n": 64,
                "threads": 256,
                "dynamic_shared_bytes": 32768,
                "registers_per_thread": 155,
                "module_bytes": 64032,
            },
        }
        def difference(finite: int, changed: int = 0) -> dict[str, object]:
            return {
                "max_abs": 0.0,
                "max_rel": 0.0,
                "max_normalized_rel": 0.0,
                "rms": 0.0,
                "finite": finite,
                "non_finite": 0,
                "bitwise_different": changed,
            }

        cases = []
        for shape in ("k2560-m10240", "k10240-m2560"):
            expected = pair[
                "expansion" if shape == "k2560-m10240" else "contraction"
            ]
            n_in, n_out = (
                (2560, 10240) if shape == "k2560-m10240" else (10240, 2560)
            )
            for token in (1, 127, 128, 129, 511, 512):
                finite = token * n_out
                full = token == 512
                logical = ((n_out + 127) // 128) * ((token + 127) // 128)
                contraction = shape == "k10240-m2560"
                efficiency = 88 if full and contraction else 96
                seams = full and contraction
                cpu_oracle = None
                timing = None
                resources = None
                post = None
                if full:
                    cpu_oracle = {
                        "samples": 8 if contraction else 4,
                        "seam_104_samples": 2 if contraction else 0,
                        "seam_208_samples": 2 if contraction else 0,
                        "no_seam_samples": 4,
                        "wrong_grid_bitwise_different": 1 if contraction else 0,
                        "seam_minus_one_bitwise_different": 1 if contraction else 0,
                        "seam_plus_one_bitwise_different": 1 if contraction else 0,
                        "wrong_grid_gpu_launched": contraction,
                        "wrong_grid_gpu_canary_errors": 0,
                        "wrong_grid_gpu_input_mismatches": 0,
                        "wrong_grid_gpu_padding_errors": 0,
                        "directed_seam_tiles": [2, 5] if contraction else [],
                        "adjacent_no_seam_tiles": (
                            [1, 3, 4, 6] if contraction else []
                        ),
                        "directed_block_boundaries": (
                            [103, 104, 207, 208] if contraction else []
                        ),
                        "wrong_grid_gpu_vs_correct": difference(
                            finite if contraction else 0,
                            1 if contraction else 0,
                        ),
                    }
                    timing = {
                        "clock": "cuda_event",
                        "schedule": "ABBA_BAAB",
                        "warmup": 1,
                        "pairs": 1,
                        "launches_per_sample": 1,
                        "samples_per_route": 2,
                        "native_samples_us": [1.0, 1.0],
                        "triton_samples_us": [1.0, 1.0],
                    }
                    resources = {
                        "cubin_bytes": expected["module_bytes"],
                        "native_function": {
                            "local_bytes": 0,
                            "binary_version": 86,
                            "launch_threads": 256,
                        },
                        "triton_function": {
                            "registers_per_thread": expected[
                                "registers_per_thread"
                            ],
                            "local_bytes": 0,
                            "binary_version": 86,
                            "launch_threads": 256,
                            "launch_dynamic_shared_bytes": expected[
                                "dynamic_shared_bytes"
                            ],
                            "block_m": expected["block_m"],
                            "block_n": expected["block_n"],
                        },
                    }
                    post = difference(finite)
                cases.append(
                    {
                        "shape_id": shape,
                        "n_in": n_in,
                        "n_out": n_out,
                        "n_tok": token,
                        "out_stride": n_out if full else n_out + 17,
                        "native_route": (
                            "full-tile" if full else "physical-stream-k"
                        ),
                        "native_tiles": {"rows": 128, "tokens": 128},
                        "logical_tiles": logical,
                        "physical_tiles": logical,
                        "efficiency": efficiency,
                        "numeric_seams": seams,
                        "fused_epilogue": False,
                        "numeric_stream_grid": 30 if seams else logical,
                        "triton_launch_config": {
                            "bm": 128,
                            "bn": 64,
                            "threads": 256,
                            "dynamic_shared_bytes": 32768,
                        },
                        "workspace_bytes": 1966080,
                        "comparison": difference(finite),
                        "native_input_mismatches": 0,
                        "triton_input_mismatches": 0,
                        "native_padding_errors": 0,
                        "triton_padding_errors": 0,
                        "canary_errors": 0,
                        "cpu_oracle": cpu_oracle,
                        "timing": timing,
                        "artifact_resources": resources,
                        "post_timing_comparison": post,
                        "post_timing_native_input_mismatches": 0,
                        "post_timing_triton_input_mismatches": 0,
                        "post_timing_native_padding_errors": 0,
                        "post_timing_triton_padding_errors": 0,
                        "post_timing_canary_errors": 0,
                    }
                )
        document = {
            "schema": 1,
            "phase": "A2",
            "milestone": "C0",
            "production_enabled": False,
            "target_sm": 86,
            "device": {
                "uuid": "0123456789abcdef0123456789abcdef",
                "name": "test",
                "sm": 86,
                "sm_count": 30,
                "driver_version": 12090,
                "cuda_runtime_version": 12090,
            },
            "same_primary_context": True,
            "same_stream": True,
            "visible_arguments": 8,
            "hidden_arguments": 2,
            "tail_tokens": list(MODULE.TAIL_TOKENS),
            "measurement_request": {
                "warmup": 1,
                "pairs": 1,
                "launches_per_sample": 1,
                "samples_per_route": 2,
                "formal_contract": False,
            },
            "policy_admissible": False,
            "metadata_kparam_preflight": False,
            "debug_exact_triton_native_required": False,
            "debug_exact_triton_native_observed": True,
            "debug_exact_triton_native_pass": False,
            "direct_debug_exit_requires_exact": False,
            "formal_pass": False,
            "structural_ok": True,
            "cases": cases,
        }
        stdout = json.dumps(document, separators=(",", ":")) + "\n"
        self.assertTrue(MODULE.validate_harness_stdout(stdout, pair)["passed"])
        cases[0]["triton_launch_config"]["threads"] = 128
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.validate_harness_stdout(
                json.dumps(document, separators=(",", ":")) + "\n", pair
            )
        cases[0]["triton_launch_config"]["threads"] = 256
        cases[0]["canary_errors"] = 1
        with self.assertRaises(MODULE.EvidenceError):
            MODULE.validate_harness_stdout(
                json.dumps(document, separators=(",", ":")) + "\n", pair
            )


if __name__ == "__main__":
    unittest.main()
