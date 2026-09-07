from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/imparo-cuda/native/tests/kernel_lab_rms_q8.cu"
RUNNER = ROOT / "dev_harness/run_cuda_kernel_lab.ps1"
TOOLS = ROOT / "dev_harness/kernel_lab_tools.ps1"
POLICY = ROOT / "config/kernel-lab-policy.toml"
DUPLICATE_POLICY = ROOT / "tools/triton-kernel-lab/kernel-lab-policy.toml"


class CudaKernelLabContractTests(unittest.TestCase):
    def test_policy_is_single_non_production_gate_a_authority(self) -> None:
        policy = tomllib.loads(POLICY.read_text(encoding="utf-8"))
        self.assertFalse(DUPLICATE_POLICY.exists())
        self.assertEqual(policy["schema"], 1)
        self.assertEqual(policy["phase"], "A2")
        self.assertEqual(policy["decision"], "gate-a")
        self.assertFalse(policy["production_authority"])
        self.assertEqual(policy["measurement"]["launches_per_sample"], 16)
        self.assertEqual(policy["scope"]["allowed_sms"], [86])
        for key in (
            "allow_signing",
            "allow_installer",
            "allow_receipt_update",
            "allow_graph_integration",
        ):
            self.assertFalse(policy["scope"][key], key)

    def test_harness_reuses_native_kernel_and_exact_engine_stream(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("#error \"kernel_lab_rms_q8.cu is test-only", source)
        self.assertIn("constexpr uint32_t kNativeThreads = 1024;", source)
        self.assertIn("k_rms_norm_q8_1_mmq<kNativeThreads>", source)
        self.assertIn("ensure_program_context(program_catalog)", source)
        self.assertIn("program_catalog.driver.module_load_data_ex", source)
        self.assertIn("reinterpret_cast<CUstream>(g.stream)", source)
        self.assertIn("CUdeviceptr global_scratch = 0;", source)
        self.assertIn("CUdeviceptr profile_scratch = 0;", source)
        self.assertIn("&global_scratch, &profile_scratch", source)
        self.assertIn("IMPARO_CUDA_KERNEL_LAB_TRACE", source)
        self.assertIn("cuModuleUnload(w4)", source)
        self.assertIn("cuModuleUnload(w8)", source)
        self.assertNotIn("Installer", source)
        self.assertNotIn("TrustStore", source)
        self.assertNotIn("receipt", source.lower())

    def test_result_and_runner_cover_gate_a_evidence(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        for field in (
            "q8_byte_agreement",
            "native_cv",
            "triton_cv",
            "paired_mad_fraction",
            "module_load_wall_us",
            "module_device_bytes_delta",
            "cold_first_us",
            "all_canary_errors",
            "structural_ok",
            "launches_per_sample",
        ):
            self.assertIn(field, source)
        self.assertIn("config/kernel-lab-policy.toml", runner)
        self.assertIn("Resolve-ComputeSanitizer", runner)
        for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
            self.assertIn(f'"{tool}"', runner)
        self.assertIn('$kernelFloorMet -or $projectedFloorMet', runner)
        self.assertIn("$data.launches_per_sample -eq $launchesPerSample", runner)
        self.assertIn("$eps 0 1 1", runner)
        self.assertIn("Gate A candidate is inadmissible", runner)
        self.assertIn("LAB-A harness target failed with exit code", runner)
        self.assertLess(
            runner.index("LAB-A harness target failed with exit code"),
            runner.index("LAB-A harness must emit exactly one JSON line"),
        )
        self.assertIn("final_gate_a_decision = $false", runner)
        self.assertIn("ProjectedE2EImprovement", runner)
        self.assertNotIn("python", runner.lower())
        self.assertNotIn("import triton", runner.lower())
        self.assertNotIn("triton.exe", runner.lower())

    def test_compute_sanitizer_resolution_is_portable_and_fail_closed(self) -> None:
        shell = shutil.which("pwsh") or shutil.which("powershell")
        self.assertIsNotNone(shell, "PowerShell is required for the Windows LAB runner")
        tool_script = str(TOOLS).replace("'", "''")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "cuda"
            nvcc = root / "bin" / "nvcc.exe"
            sanitizer = root / "compute-sanitizer" / "compute-sanitizer.exe"
            nvcc.parent.mkdir(parents=True)
            sanitizer.parent.mkdir(parents=True)
            nvcc.touch()
            sanitizer.touch()
            nvcc_arg = str(nvcc).replace("'", "''")
            command = (
                f". '{tool_script}'; $env:Path=''; "
                f"Resolve-ComputeSanitizer -NvccPath '{nvcc_arg}'"
            )
            resolved = subprocess.run(
                [shell, "-NoProfile", "-Command", command],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(resolved.returncode, 0, resolved.stderr)
            self.assertEqual(pathlib.Path(resolved.stdout.strip()), sanitizer)

            missing_nvcc = pathlib.Path(temporary) / "missing" / "bin" / "nvcc.exe"
            missing_nvcc.parent.mkdir(parents=True)
            missing_nvcc.touch()
            missing_arg = str(missing_nvcc).replace("'", "''")
            missing_command = (
                f". '{tool_script}'; $env:Path=''; "
                f"Resolve-ComputeSanitizer -NvccPath '{missing_arg}'"
            )
            missing = subprocess.run(
                [shell, "-NoProfile", "-Command", missing_command],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("was not found", missing.stderr)

    def test_release_build_does_not_compile_or_export_lab(self) -> None:
        build = (ROOT / "crates/imparo-cuda/build.rs").read_text(encoding="utf-8")
        exports = (ROOT / "crates/imparo-cuda/native/imparo_cuda.def").read_text(
            encoding="utf-8"
        )
        self.assertIn('name == "tests"', build)
        self.assertNotIn("IMPARO_CUDA_KERNEL_LAB", build)
        self.assertNotIn("kernel_lab", exports.lower())


if __name__ == "__main__":
    unittest.main()
