import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from dev_harness import verify_program_pack_release as release_boundary


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "program-packs" / "community" / "cuda" / "smoke.py"
CONTRACT = SOURCE.with_name("smoke-contract-v1.json")
FIXTURE = SOURCE.with_name("smoke-fixture-v1.json")
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RUST_SMOKE = ROOT / "crates" / "imparo-cuda" / "src" / "program_pack" / "sm86_smoke.rs"
EXPECTED_KERNEL = "ip_ab8511ce441e25e37a94fd947a072a657077a099a3ce4b944cbcc7e002a34a2c"


def _aot_metadata():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "IMPARO_AOT"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("smoke source must define a literal IMPARO_AOT descriptor")


class TritonPackSmokeSourceTests(unittest.TestCase):
    def test_source_imports_only_triton_and_never_torch(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertEqual(imports, ["triton", "triton.language"])
        self.assertNotIn("torch", imports)

    def test_source_descriptor_is_stable_and_smoke_only(self):
        spec = _aot_metadata()
        self.assertEqual(spec["kernel"], EXPECTED_KERNEL)
        self.assertEqual(spec["signature"], ["*fp32:16", "*fp32:16", "i32"])
        self.assertEqual((spec["num_warps"], spec["num_stages"]), (1, 1))
        manifest = spec["manifest"]
        self.assertEqual(manifest["distribution"], "community")
        self.assertEqual(manifest["choice_group"]["contract"]["id"], "imparo.cuda.infrastructure.copy")
        self.assertTrue(manifest["choice_group"]["screened"])
        self.assertFalse(manifest["variant"]["bit_affecting"])
        self.assertEqual(manifest["variant"]["numerical_class"], {"kind": "diagnostic_only"})
        self.assertEqual(manifest["variant"]["scratch"]["max_bytes"], 0)
        self.assertNotIn("target", manifest, "the CLI owns exact-SM targeting")
        for generated in ("variant_id", "config_id", "module_id", "symbol", "resources"):
            self.assertNotIn(generated, manifest["variant"])

    def test_canonical_contract_and_fixture_hashes_match_descriptor(self):
        spec = _aot_metadata()["manifest"]
        contract_hash = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        fixture_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        self.assertEqual(spec["choice_group"]["contract"]["sha256"], contract_hash)
        self.assertEqual(spec["variant"]["contract"]["sha256"], contract_hash)
        self.assertEqual(spec["choice_group"]["workload"]["fixture_sha256"], fixture_hash)
        self.assertEqual(json.loads(CONTRACT.read_text(encoding="utf-8"))["revision"], 1)
        self.assertEqual(json.loads(FIXTURE.read_text(encoding="utf-8"))["schema"], 1)

    def test_formal_sm86_cross_language_lane_is_fail_closed(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        rust_smoke = RUST_SMOKE.read_text(encoding="utf-8")
        for required in (
            "if: matrix.sm == 86",
            "name: triton-aot-sm86",
            "triton-sm86-windows-runtime:",
            "imparo-cuda-sm86",
            "imparo-no-python",
            "IMPARO_REQUIRE_TRITON_SMOKE: '1'",
            "IMPARO_TRITON_SMOKE_PACK",
            "IMPARO_TRITON_SMOKE_PUBLIC_KEY_HEX",
            "IMPARO_TRITON_SMOKE_KEY_ID",
            "cargo test --release --locked -p imparo-cuda --features cuda-static triton_pack_",
        ):
            self.assertIn(required, workflow)
        self.assertIn('std::env::var("IMPARO_REQUIRE_TRITON_SMOKE")', rust_smoke)
        self.assertIn(
            "formal Triton smoke requires IMPARO_TRITON_SMOKE_PACK", rust_smoke
        )


class TritonPackArtifactTests(unittest.TestCase):
    def test_external_artifact_is_data_only_and_reproducible_when_requested(self):
        first = os.environ.get("IMPARO_TRITON_SMOKE_PACK_A")
        second = os.environ.get("IMPARO_TRITON_SMOKE_PACK_B")
        if not first or not second:
            self.skipTest("set IMPARO_TRITON_SMOKE_PACK_A/B to compare clean AOT builds")
        expected_feasibility = os.environ.get("IMPARO_TRITON_EXPECT_FEASIBILITY")
        if expected_feasibility not in {"0", "1"}:
            self.fail(
                "external pack smoke requires IMPARO_TRITON_EXPECT_FEASIBILITY=0|1"
            )
        first_root, second_root = Path(first), Path(second)
        self.assertEqual(_tree_identity(first_root), _tree_identity(second_root))
        feasibility_only = expected_feasibility == "1"
        for root in (first_root, second_root):
            report = release_boundary.scan_pack(
                root,
                allow_feasibility_smoke=feasibility_only,
            )
            self.assertIs(report["feasibility_only"], feasibility_only)
            _assert_data_only(self, root)

    def test_builder_help_is_available_without_importing_torch(self):
        builder = ROOT / "tools" / "triton-pack" / "build.py"
        if not builder.exists():
            self.fail("Step 4 builder is missing")
        script = (
            "import runpy,sys;"
            f"sys.path.insert(0,{str(builder.parent)!r});"
            "sys.argv=['build.py','--help'];"
            "code=0;"
            "\ntry:\n"
            f" runpy.run_path({str(builder)!r},run_name='__main__')\n"
            "except SystemExit as error:\n"
            " code=error.code or 0\n"
            "assert 'torch' not in sys.modules;"
            "raise SystemExit(code)"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_builder_accepts_descriptor_but_rejects_hidden_scratch(self):
        tools = ROOT / "tools" / "triton-pack"
        lock = tools / "toolchain.lock"
        script = f'''\
import sys, tempfile, types
from pathlib import Path
sys.path.insert(0, {str(tools)!r})
triton = types.ModuleType("triton")
triton.jit = lambda function: function
language = types.ModuleType("triton.language")
triton.language = language
sys.modules["triton"] = triton
sys.modules["triton.language"] = language
from imparo_triton_pack import BuildError, build_pack, load_toolchain_lock
from imparo_triton_pack.aot import CompileResult, load_aot_spec
spec = load_aot_spec(Path({str(SOURCE)!r}))
assert spec.kernel_name == {EXPECTED_KERNEL!r}
assert spec.signature == ("*fp32:16", "*fp32:16", "i32")
class HiddenScratchCompiler:
    def compile(self, source, spec, target):
        return CompileResult(cubin=b"not-reached", symbol=spec.kernel_name,
                             registers_per_thread=1, static_shared_bytes=0,
                             local_memory_bytes=0, global_scratch_bytes=1,
                             profile_scratch_bytes=0)
with tempfile.TemporaryDirectory() as temporary:
    try:
        build_pack(lock=load_toolchain_lock(Path({str(lock)!r})),
                   target="cuda:86:32", source=Path({str(SOURCE)!r}),
                   output=Path(temporary) / "pack",
                   compiler=HiddenScratchCompiler(), enforce_environment=False)
    except BuildError as error:
        assert "global_scratch/profile_scratch" in str(error), error
    else:
        raise AssertionError("builder accepted non-zero hidden scratch")
assert "torch" not in sys.modules
'''
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


def _tree_identity(root: Path):
    self_manifest = root / "manifest.json"
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.sig":
            relative = path.relative_to(root).as_posix()
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not self_manifest.exists():
        raise AssertionError(f"missing manifest.json in {root}")
    return files


def _assert_data_only(case: unittest.TestCase, root: Path):
    forbidden_suffixes = {".py", ".ptx", ".ttir", ".ttgir", ".ll", ".bc", ".dll", ".so"}
    forbidden_tokens = (b".debug_", b".lineinfo", b".py\x00", b"/home/", b"\\Users\\")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        case.assertNotIn(path.suffix.lower(), forbidden_suffixes, str(path))
        data = path.read_bytes()
        for token in forbidden_tokens:
            case.assertNotIn(token, data, f"{token!r} leaked into {path}")


if __name__ == "__main__":
    unittest.main()
