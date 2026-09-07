from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = ROOT / "config/kernel-lab-policy.toml"
METADATA = ROOT / "program-packs/lab/q4-q8-mmq-sm86/run-b/lab-metadata.json"
VARIANT_METADATA = (
    ROOT / "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json"
)
CONTRACT = ROOT / "dev_harness/kernel_lab_q4_contract.ps1"
RUNNER = ROOT / "dev_harness/run_cuda_kernel_lab_q4.ps1"
RMS_RUNNER = ROOT / "dev_harness/run_cuda_kernel_lab.ps1"


def discover_cuobjdump() -> pathlib.Path | None:
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(variable)
        if not root:
            continue
        for name in ("cuobjdump.exe", "cuobjdump"):
            candidate = pathlib.Path(root) / "bin" / name
            if candidate.is_file():
                return candidate
    command = shutil.which("cuobjdump") or shutil.which("cuobjdump.exe")
    return pathlib.Path(command) if command else None


CUOBJDUMP = discover_cuobjdump()


def quote(path: pathlib.Path) -> str:
    return str(path).replace("'", "''")


def powershell(command: str) -> subprocess.CompletedProcess[str]:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        raise unittest.SkipTest("PowerShell is required")
    return subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )


class Q4KernelLabPolicyTests(unittest.TestCase):
    def test_single_policy_freezes_exact_run_b_and_oracle_envelope(self) -> None:
        q4 = tomllib.loads(POLICY.read_text(encoding="utf-8"))["q4_mmq"]
        self.assertEqual(q4["q4_numeric_policy_status"], "frozen-from-cpu-oracle")
        self.assertEqual((q4["q4_visible_arguments"], q4["q4_hidden_arguments"]), (8, 2))
        self.assertEqual(q4["q4_kparam_bytes"], 72)
        self.assertEqual(q4["q4_threads"], 128)
        self.assertEqual(q4["q4_dynamic_shared_bytes"], 16384)
        self.assertEqual(q4["q4_registers_per_thread"], 127)
        self.assertEqual(q4["q4_local_memory_bytes"], 0)
        self.assertEqual(q4["q4_static_shared_bytes"], 0)
        self.assertEqual(q4["q4_max_abs_vs_native"], 0.00006103515625)
        self.assertEqual(q4["q4_max_rms_vs_native"], 0.00001)
        self.assertEqual(q4["q4_max_normalized_rel_vs_native"], 0.00005)
        self.assertNotIn("q4_max_rel_vs_native", q4)
        self.assertEqual(q4["q4_require_tail_tokens"], [1, 127, 128, 129, 511, 512])
        self.assertEqual(q4["q4_mutation_min_bitwise_different"], 1)
        self.assertEqual((q4["q4_triton_block_m"], q4["q4_triton_block_n"]), (64, 64))
        self.assertEqual(
            q4["q4_builder_triton_commit"],
            "b252c7c4cea49216b27e5a47db061f9d0dbf49f3",
        )
        self.assertEqual(q4["q4_builder_ptxas"], "V12.9.86")
        self.assertEqual(q4["q4_builder_cuobjdump"], "V12.9.82")
        self.assertEqual(
            q4["q4_builder_recipe_sha256"],
            "5165e28709da883bd44423c6f980726ed5413b89094f3efa76d3c617cbe4efe7",
        )
        self.assertEqual(
            q4["q4_variants_metadata_sha256"],
            "f4dd69db8475ec881f5bf5062d1a7d4626605b658bd7727233af077eef3d2607",
        )
        self.assertEqual(len(q4["q4_variant_allowed_expansion"]), 4)
        self.assertEqual(len(q4["q4_variant_allowed_contraction"]), 4)
        self.assertIn("without Cartesian expansion", q4["q4_variant_allowlist_rationale"])

    def test_exact_metadata_passes_and_mutation_fails_closed(self) -> None:
        prefix = (
            f". '{quote(CONTRACT)}'; "
            f"$p=Get-Content -Raw -LiteralPath '{quote(POLICY)}'; "
        )
        valid = powershell(
            prefix + f"@(Assert-Q4MetadataContract -MetadataPath "
            f"'{quote(METADATA)}' -PolicyText $p).Count"
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(valid.stdout.strip(), "2")
        with tempfile.TemporaryDirectory() as temporary:
            mutant = pathlib.Path(temporary) / "lab-metadata.json"
            data = json.loads(METADATA.read_text(encoding="utf-8"))
            data["candidate"] = "mutated"
            mutant.write_text(json.dumps(data), encoding="utf-8")
            rejected = powershell(
                prefix + f"Assert-Q4MetadataContract -MetadataPath "
                f"'{quote(mutant)}' -PolicyText $p"
            )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("metadata SHA256 mismatch", rejected.stderr)

    def test_bounded_variant_pair_is_exact_and_shape_scoped(self) -> None:
        q4 = tomllib.loads(POLICY.read_text(encoding="utf-8"))["q4_mmq"]
        expansion = q4["q4_variant_allowed_expansion"][2]
        contraction = q4["q4_variant_allowed_contraction"][2]
        prefix = (
            f". '{quote(CONTRACT)}'; "
            f"$p=Get-Content -Raw -LiteralPath '{quote(POLICY)}'; "
        )
        valid = powershell(
            prefix
            + f"$a=@(Assert-Q4VariantPairContract -MetadataPath "
            + f"'{quote(VARIANT_METADATA)}' -PolicyText $p "
            + f"-ExpansionConfigId '{expansion}' "
            + f"-ContractionConfigId '{contraction}'); "
            + "$a|Select-Object ShapeId,ConfigId,BlockM,BlockN,Threads,"
            + "DynamicSharedBytes,Registers,CubinBytes|ConvertTo-Json"
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        artifacts = json.loads(valid.stdout)
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(
            {artifact["ShapeId"] for artifact in artifacts},
            {"k2560-m10240", "k10240-m2560"},
        )
        for artifact in artifacts:
            self.assertEqual(
                (artifact["BlockM"], artifact["BlockN"], artifact["Threads"]),
                (64, 64, 256),
            )
            self.assertEqual(artifact["DynamicSharedBytes"], 16384)
            self.assertEqual(artifact["Registers"], 101)
            self.assertGreater(artifact["CubinBytes"], 0)
        wrong_shape = powershell(
            prefix
            + f"Assert-Q4VariantPairContract -MetadataPath "
            + f"'{quote(VARIANT_METADATA)}' -PolicyText $p "
            + f"-ExpansionConfigId '{contraction}' "
            + f"-ContractionConfigId '{contraction}'"
        )
        self.assertNotEqual(wrong_shape.returncode, 0)
        self.assertIn("not policy-allowed", wrong_shape.stderr)

    def test_variant_metadata_hash_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mutant = pathlib.Path(temporary) / "lab-metadata.json"
            data = json.loads(VARIANT_METADATA.read_text(encoding="utf-8"))
            data["matrix"]["no_cartesian_expansion"] = False
            mutant.write_text(json.dumps(data), encoding="utf-8")
            q4 = tomllib.loads(POLICY.read_text(encoding="utf-8"))["q4_mmq"]
            rejected = powershell(
                f". '{quote(CONTRACT)}'; "
                f"$p=Get-Content -Raw -LiteralPath '{quote(POLICY)}'; "
                f"Assert-Q4VariantPairContract -MetadataPath '{quote(mutant)}' "
                f"-PolicyText $p -ExpansionConfigId "
                f"'{q4['q4_variant_allowed_expansion'][0]}' "
                f"-ContractionConfigId "
                f"'{q4['q4_variant_allowed_contraction'][0]}'"
            )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("variants-b metadata SHA256 mismatch", rejected.stderr)

    def test_variant_dynamic_shared_uses_distinct_policy_cap(self) -> None:
        q4 = tomllib.loads(POLICY.read_text(encoding="utf-8"))["q4_mmq"]
        policy_text = POLICY.read_text(encoding="utf-8")
        self.assertIn("max_static_shared_bytes = 101376", policy_text)
        self.assertIn("max_dynamic_shared_bytes = 101376", policy_text)
        capped = policy_text.replace(
            "max_dynamic_shared_bytes = 101376",
            "max_dynamic_shared_bytes = 16000",
            1,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".toml", delete=False, encoding="utf-8"
        ) as stream:
            stream.write(capped)
            fixture = pathlib.Path(stream.name)
        try:
            rejected = powershell(
                f". '{quote(CONTRACT)}'; "
                f"$p=Get-Content -Raw -LiteralPath '{quote(fixture)}'; "
                f"Assert-Q4VariantPairContract -MetadataPath "
                f"'{quote(VARIANT_METADATA)}' -PolicyText $p "
                f"-ExpansionConfigId "
                f"'{q4['q4_variant_allowed_expansion'][0]}' "
                f"-ContractionConfigId "
                f"'{q4['q4_variant_allowed_contraction'][0]}'"
            )
        finally:
            fixture.unlink(missing_ok=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(
            "selected variant resource/geometry contract mismatch",
            rejected.stderr,
        )

    def test_runner_rejects_partial_variant_selection_before_preflight(self) -> None:
        q4 = tomllib.loads(POLICY.read_text(encoding="utf-8"))["q4_mmq"]
        expansion = q4["q4_variant_allowed_expansion"][0]
        commands = (
            (
                f"& '{quote(RUNNER)}' -Mode preflight "
                f"-ExpansionConfigId '{expansion}'"
            ),
            (
                f"& '{quote(RUNNER)}' -Mode preflight "
                f"-VariantMetadata '{quote(VARIANT_METADATA)}' "
                f"-ExpansionConfigId '{expansion}'"
            ),
        )
        for invocation in commands:
            rejected = powershell(
                "$env:IMPARO_CUDA_KERNEL_LAB='1'; " + invocation
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "variant-pair mode requires metadata plus both "
                "expansion/contraction config IDs",
                rejected.stderr,
            )

    def test_cuobjdump_discovers_cuda_environment_without_literal_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            tool = root / "bin" / "cuobjdump.exe"
            tool.parent.mkdir()
            tool.write_bytes(b"fixture")
            with (
                mock.patch.dict(os.environ, {"CUDA_PATH": str(root)}, clear=False),
                mock.patch.object(shutil, "which", return_value=None),
            ):
                self.assertEqual(discover_cuobjdump(), tool)

    def test_runner_tool_discovery_has_no_machine_absolute_path(self) -> None:
        for runner_path in (RMS_RUNNER, RUNNER):
            runner = runner_path.read_text(encoding="utf-8")
            self.assertNotRegex(runner, r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\/]")
            self.assertIn('Get-Command "vswhere.exe"', runner)
            self.assertIn('"ProgramFiles(x86)", "ProgramFiles"', runner)
            self.assertIn("[Environment]::GetEnvironmentVariable", runner)
            self.assertIn(
                "Visual Studio Installer discovery tool vswhere.exe not found",
                runner,
            )

    def test_kparam_and_reqntid_mutations_fail_closed(self) -> None:
        if CUOBJDUMP is None:
            raise unittest.SkipTest("cuobjdump is unavailable via CUDA_PATH/PATH")
        variant = json.loads(METADATA.read_text(encoding="utf-8"))["variants"][0]
        cubin = METADATA.parent / variant["module"]
        inspected = subprocess.run(
            [str(CUOBJDUMP), "-elf", cubin],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        mutations = (
            ("Value:\t0x48", "Value:\t0x40", "KPARAM"),
            ("Value:\t0x80 0x1 0x1", "Value:\t0x40 0x1 0x1", "REQNTID"),
        )
        for needle, replacement, expected in mutations:
            self.assertIn(needle, inspected)
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            ) as stream:
                stream.write(inspected.replace(needle, replacement, 1))
                mutation = pathlib.Path(stream.name)
            try:
                command = (
                    f". '{quote(CONTRACT)}'; "
                    f"$t=Get-Content -Raw -LiteralPath '{quote(mutation)}'; "
                    f"Assert-Q4CubinInspection -Text $t "
                    f"-Symbol '{variant['symbol']}' -Registers 127"
                )
                rejected = powershell(command)
            finally:
                mutation.unlink(missing_ok=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(expected, rejected.stderr)

    def test_corrupt_result_fields_are_named_violations(self) -> None:
        base_case = {
            "shape_id": "k2560-m10240", "n_in": 2560, "n_out": 10240,
            "n_tok": 1, "out_stride": 10257,
            "native_route": "physical-stream-k",
            "native_tiles": {"rows": 128, "tokens": 128},
            "logical_tiles": 80, "physical_tiles": 30, "efficiency": 88,
            "numeric_seams": True, "fused_epilogue": False,
            "numeric_stream_grid": 30, "workspace_bytes": 1966080,
            "comparison": {
                "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
                "rms": 0, "finite": 10240, "non_finite": 0,
                "bitwise_different": 0,
            },
            "native_input_mismatches": 0, "triton_input_mismatches": 0,
            "native_padding_errors": 0, "triton_padding_errors": 0,
            "canary_errors": 0, "cpu_oracle": None,
        }
        mutations = (
            ("comparison", "max_abs", 1.0, "max_abs mismatch"),
            ("comparison", "non_finite", 1, "non-finite output"),
            ("case", "canary_errors", 1, "canary error"),
            ("case", "triton_input_mismatches", 1, "Triton input mutation"),
            ("case", "triton_padding_errors", 1, "Triton padding error"),
        )
        for section, field, value, expected in mutations:
            with self.subTest(expected=expected):
                case = json.loads(json.dumps(base_case))
                target = case["comparison"] if section == "comparison" else case
                target[field] = value
                document = {
                    "schema": 1, "phase": "A2", "production_enabled": False,
                    "target_sm": 86, "same_primary_context": True,
                    "same_stream": True, "visible_arguments": 8,
                    "hidden_arguments": 2,
                    "tail_tokens": [1, 127, 128, 129, 511, 512],
                    "policy_admissible": False, "formal_pass": False,
                    "metadata_kparam_preflight": False,
                    "cases": [case], "structural_ok": True,
                }
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".json", delete=False, encoding="utf-8"
                ) as stream:
                    json.dump(document, stream)
                    fixture = pathlib.Path(stream.name)
                try:
                    command = (
                        f". '{quote(CONTRACT)}'; "
                        f"$p=Get-Content -Raw '{quote(POLICY)}'; "
                        f"$d=Get-Content -Raw '{quote(fixture)}'|ConvertFrom-Json; "
                        "(Get-Q4ResultViolations -Data $d -PolicyText $p)"
                    )
                    result = powershell(command)
                finally:
                    fixture.unlink(missing_ok=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)

    def test_runner_refuses_missing_formal_evidence(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('$runExit -notin @(0, 3)', runner)
        self.assertIn("formal CUDA-event ABBA/BAAB timing/noise evidence missing", runner)
        self.assertIn("four-tool sanitizer evidence missing", runner)
        self.assertIn(
            "cold module/first-launch/VRAM/function-resource evidence missing",
            runner,
        )
        self.assertIn("final_gate_a_decision = $false", runner)
        self.assertIn('if ($case.n_tok -ne 512)', runner)
        self.assertIn('$timingEvidenceCases -ne 2', runner)
        self.assertIn('$timing.clock -eq "cuda_event"', runner)
        self.assertIn('$timing.schedule -eq "ABBA_BAAB"', runner)
        self.assertIn("median_speedup_native_over_triton", runner)
        self.assertIn("$measurementAdmissible", runner)
        self.assertIn('$Mode -eq "measure"', runner)
        self.assertIn("post_timing_comparison", runner)
        self.assertIn("$kernelGeometryArguments", runner)

    def test_runner_binds_complete_provenance_and_wrapper_preflight(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        for required in (
            "metadata_kparam_preflight = $true",
            "policy_sha256 = $policySha",
            "metadata_sha256 = $metadataSha",
            "harness_source_sha256 = $sourceSha",
            "harness_exe_sha256 = $exeSha",
            "result_json_sha256 = $resultSha",
            "device = $device",
            "nvcc = $nvccVersion",
            "cuobjdump = $cuobjdumpVersion",
            "semantic_recipe_sha256",
            "linux_amd64_manifest",
            "oci_config",
            "bytes = (Get-Item -LiteralPath $_.cubin).Length",
            "variant-pair mode requires metadata plus both expansion/contraction config IDs",
            "Assert-Q4VariantPairContract",
            "selected_pair = @($artifacts",
            "gates_kernel_candidate = $false",
        ):
            self.assertIn(required, runner)

    def test_post_timing_numeric_and_safety_mutations_fail_closed(self) -> None:
        zero_difference = {
            "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
            "rms": 0, "finite": 4, "non_finite": 0,
            "bitwise_different": 0,
        }
        zero_f64 = {
            "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
            "rms": 0, "count": 4, "non_finite": 0,
        }
        oracle = {
            "samples": 4, "seam_104_samples": 0, "seam_208_samples": 0,
            "no_seam_samples": 4, "wrong_grid_gpu_launched": False,
            "native_vs_strict_float": zero_difference,
            "triton_vs_strict_float": zero_difference,
            "native_vs_f64": zero_f64, "triton_vs_f64": zero_f64,
        }
        case = {
            "shape_id": "k2560-m10240", "n_in": 2560, "n_out": 10240,
            "n_tok": 512, "out_stride": 10240, "native_route": "full-tile",
            "native_tiles": {"rows": 128, "tokens": 128},
            "logical_tiles": 320, "physical_tiles": 320, "efficiency": 96,
            "numeric_seams": False, "fused_epilogue": False,
            "numeric_stream_grid": 320, "workspace_bytes": 1966080,
            "comparison": {
                "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
                "rms": 0, "finite": 5242880, "non_finite": 0,
                "bitwise_different": 0,
            },
            "native_input_mismatches": 0, "triton_input_mismatches": 0,
            "native_padding_errors": 0, "triton_padding_errors": 0,
            "canary_errors": 0, "cpu_oracle": oracle,
            "post_timing_comparison": {
                "max_abs": 1, "max_rel": 1, "max_normalized_rel": 1,
                "rms": 1, "finite": 5242879, "non_finite": 1,
                "bitwise_different": 1,
            },
            "post_timing_native_input_mismatches": 1,
            "post_timing_triton_input_mismatches": 1,
            "post_timing_native_padding_errors": 1,
            "post_timing_triton_padding_errors": 1,
            "post_timing_canary_errors": 1,
        }
        data = {
            "schema": 1, "phase": "A2", "production_enabled": False,
            "target_sm": 86, "same_primary_context": True,
            "same_stream": True, "visible_arguments": 8,
            "hidden_arguments": 2,
            "tail_tokens": [1, 127, 128, 129, 511, 512],
            "policy_admissible": False, "formal_pass": False,
            "metadata_kparam_preflight": False,
            "debug_exact_triton_native_required": False,
            "debug_exact_triton_native_pass": False,
            "cases": [case], "structural_ok": True,
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as stream:
            json.dump(data, stream)
            fixture = pathlib.Path(stream.name)
        try:
            command = (
                f". '{quote(CONTRACT)}'; $p=Get-Content -Raw '{quote(POLICY)}'; "
                f"$d=Get-Content -Raw '{quote(fixture)}'|ConvertFrom-Json; "
                "(Get-Q4ResultViolations -Data $d -PolicyText $p)"
            )
            result = powershell(command)
        finally:
            fixture.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "post-timing finite count mismatch",
            "post-timing non-finite output",
            "post-timing max_abs mismatch",
            "post-timing RMS mismatch",
            "post-timing normalized relative mismatch",
            "post-timing native input mutation",
            "post-timing Triton input mutation",
            "post-timing native padding error",
            "post-timing Triton padding error",
            "post-timing canary error",
        ):
            self.assertIn(expected, result.stdout)

    def test_expansion_n511_rejects_false_positive_numeric_seam(self) -> None:
        case = {
            "shape_id": "k2560-m10240", "n_in": 2560, "n_out": 10240,
            "n_tok": 511, "out_stride": 10257,
            "native_route": "physical-stream-k",
            "native_tiles": {"rows": 128, "tokens": 128},
            "logical_tiles": 320, "physical_tiles": 320, "efficiency": 96,
            "numeric_seams": True, "fused_epilogue": False,
            "numeric_stream_grid": 320, "workspace_bytes": 1966080,
            "comparison": {
                "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
                "rms": 0, "finite": 5232640, "non_finite": 0,
                "bitwise_different": 0,
            },
            "native_input_mismatches": 0, "triton_input_mismatches": 0,
            "native_padding_errors": 0, "triton_padding_errors": 0,
            "canary_errors": 0, "cpu_oracle": None,
        }
        data = {
            "schema": 1, "phase": "A2", "production_enabled": False,
            "target_sm": 86, "same_primary_context": True, "same_stream": True,
            "visible_arguments": 8, "hidden_arguments": 2,
            "tail_tokens": [1, 127, 128, 129, 511, 512],
            "policy_admissible": False, "formal_pass": False,
            "metadata_kparam_preflight": False, "cases": [case],
            "structural_ok": True,
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as stream:
            json.dump(data, stream)
            fixture = pathlib.Path(stream.name)
        try:
            command = (
                f". '{quote(CONTRACT)}'; $p=Get-Content -Raw '{quote(POLICY)}'; "
                f"$d=Get-Content -Raw '{quote(fixture)}'|ConvertFrom-Json; "
                "(Get-Q4ResultViolations -Data $d -PolicyText $p)"
            )
            result = powershell(command)
        finally:
            fixture.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("numeric seam flag mismatch in k2560-m10240/511", result.stdout)

    def test_bitwise_debug_cannot_claim_required_status(self) -> None:
        data = {
            "schema": 1, "phase": "A2", "production_enabled": False,
            "target_sm": 86, "same_primary_context": True, "same_stream": True,
            "visible_arguments": 8, "hidden_arguments": 2,
            "tail_tokens": [1, 127, 128, 129, 511, 512],
            "policy_admissible": False, "formal_pass": False,
            "metadata_kparam_preflight": False,
            "debug_exact_triton_native_required": True,
            "debug_exact_triton_native_pass": False,
            "cases": [], "structural_ok": True,
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as stream:
            json.dump(data, stream)
            fixture = pathlib.Path(stream.name)
        try:
            command = (
                f". '{quote(CONTRACT)}'; $p=Get-Content -Raw '{quote(POLICY)}'; "
                f"$d=Get-Content -Raw '{quote(fixture)}'|ConvertFrom-Json; "
                "(Get-Q4ResultViolations -Data $d -PolicyText $p)"
            )
            result = powershell(command)
        finally:
            fixture.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bitwise comparison must be diagnostic-only", result.stdout)

    def test_missing_mutation_evidence_and_safety_errors_fail_closed(self) -> None:
        zero_diff = {
            "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
            "rms": 0, "finite": 8, "non_finite": 0,
            "bitwise_different": 0,
        }
        oracle = {
            "samples": 8, "seam_104_samples": 2, "seam_208_samples": 2,
            "no_seam_samples": 4, "wrong_grid_bitwise_different": 4,
            "seam_minus_one_bitwise_different": 0,
            "seam_plus_one_bitwise_different": 0,
            "wrong_grid_gpu_launched": False,
            "wrong_grid_gpu_canary_errors": 1,
            "wrong_grid_gpu_input_mismatches": 1,
            "wrong_grid_gpu_padding_errors": 1,
            "directed_seam_tiles": [2, 5],
            "adjacent_no_seam_tiles": [1, 3, 4, 6],
            "directed_block_boundaries": [103, 104, 207, 208],
            "native_vs_strict_float": zero_diff,
            "triton_vs_strict_float": zero_diff,
            "wrong_grid_gpu_vs_correct": zero_diff,
            "wrong_grid_gpu_vs_native": zero_diff,
            "native_vs_f64": {
                "max_abs": 0, "non_finite": 0, "count": 8
            },
            "triton_vs_f64": {
                "max_abs": 0, "non_finite": 0, "count": 8
            },
        }
        case = {
            "shape_id": "k10240-m2560", "n_in": 10240, "n_out": 2560,
            "n_tok": 512, "out_stride": 2560, "native_route": "full-tile",
            "native_tiles": {"rows": 128, "tokens": 128},
            "logical_tiles": 80, "physical_tiles": 80, "efficiency": 88,
            "numeric_seams": True, "fused_epilogue": False,
            "numeric_stream_grid": 30, "workspace_bytes": 1966080,
            "comparison": {
                "max_abs": 0, "max_rel": 0, "max_normalized_rel": 0,
                "rms": 0, "finite": 1310720, "non_finite": 0,
                "bitwise_different": 0,
            },
            "native_input_mismatches": 0, "triton_input_mismatches": 0,
            "native_padding_errors": 0, "triton_padding_errors": 0,
            "canary_errors": 0, "cpu_oracle": oracle,
        }
        data = {
            "schema": 1, "phase": "A2", "production_enabled": False,
            "target_sm": 86, "same_primary_context": True, "same_stream": True,
            "visible_arguments": 8, "hidden_arguments": 2,
            "tail_tokens": [1, 127, 128, 129, 511, 512],
            "policy_admissible": False, "formal_pass": False,
            "metadata_kparam_preflight": False,
            "debug_exact_triton_native_required": False,
            "debug_exact_triton_native_pass": False,
            "cases": [case], "structural_ok": True,
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as stream:
            json.dump(data, stream)
            fixture = pathlib.Path(stream.name)
        try:
            command = (
                f". '{quote(CONTRACT)}'; $p=Get-Content -Raw '{quote(POLICY)}'; "
                f"$d=Get-Content -Raw '{quote(fixture)}'|ConvertFrom-Json; "
                "(Get-Q4ResultViolations -Data $d -PolicyText $p)"
            )
            result = powershell(command)
        finally:
            fixture.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "seam-1 CPU mutation was not detected",
            "seam+1 CPU mutation was not detected",
            "wrong-grid GPU mutation was not launched",
            "wrong-grid GPU mutation did not differ from correct Triton",
            "wrong-grid GPU mutation did not differ from native",
            "wrong-grid GPU mutation corrupted canary",
            "wrong-grid GPU mutation changed input",
            "wrong-grid GPU mutation corrupted padding",
        ):
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
