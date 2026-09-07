from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imparo_triton_pack.aot import (
    CompileResult,
    TritonCompiler,
    _resource_usage,
    audit_source_imports,
    load_aot_spec,
)
from imparo_triton_pack.artifacts import RECIPE_FILES, build_pack
from imparo_triton_pack.common import canonical_json
from imparo_triton_pack.elf import verify_release_cubin
from imparo_triton_pack.errors import BuildError
from imparo_triton_pack.lock import load_toolchain_lock
from imparo_triton_pack.signing import sign_manifest, signature_message
from imparo_triton_pack.verify import verify_pack

SYMBOL = "ip_" + "a" * 64


def minimal_elf(section: str = ".text." + SYMBOL, *, sm: int = 86) -> bytes:
    names = b"\0.shstrtab\0" + section.encode("ascii") + b"\0"
    section_offset = 64
    section_count = 3
    names_offset = section_offset + 64 * section_count
    payload_offset = names_offset + len(names)
    data = bytearray(payload_offset + 1)
    data[:16] = b"\x7fELF\x02\x01\x01\x33\x07" + b"\0" * 7
    flags = 0x560500 | sm
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 2, 190, 1, 0, 0, section_offset, flags, 64, 0, 0, 64, section_count, 1)
    shstr = section_offset + 64
    struct.pack_into("<IIQQQQIIQQ", data, shstr, 1, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    text = section_offset + 128
    struct.pack_into("<IIQQQQIIQQ", data, text, 11, 1, 0, 0, payload_offset, 1, 0, 0, 1, 0)
    data[names_offset : names_offset + len(names)] = names
    return bytes(data)


def manifest_template() -> dict[str, object]:
    contract = {
        "id": "imparo.cuda.infrastructure.copy",
        "revision": 1,
        "sha256": "1" * 64,
    }
    workload = {
        "workload_id": "imparo.workload.attention_decode",
        "revision": 1,
        "parameters": {},
        "parameters_sha256": "2" * 64,
        "fixture_sha256": "3" * 64,
    }
    group = {
        "choice_group_id": "imparo.cuda.infrastructure.copy.v1",
        "contract": contract,
        "workload": workload,
        "screened": True,
        "bit_affecting": False,
        "joint_with": [],
    }
    variant = {
        "choice_group_id": group["choice_group_id"],
        "contract": contract,
        "constraints": {
            "shapes": [],
            "dtypes": [
                {"slot": "output", "allowed": ["f32"]},
                {"slot": "input", "allowed": ["f32"]},
            ],
            "quantizations": [
                {"slot": "output", "allowed": ["none"]},
                {"slot": "input", "allowed": ["none"]},
            ],
            "layouts": [
                {"slot": "output", "allowed": ["contiguous"]},
                {"slot": "input", "allowed": ["contiguous"]},
            ],
            "alignments": [
                {"slot": "output", "bytes": 16},
                {"slot": "input", "bytes": 16},
            ],
        },
        "effects": [
            {"slot": "output", "access": "write", "aliasing": []},
            {"slot": "input", "access": "read", "aliasing": []},
        ],
        "scratch": {"max_bytes": 0, "alignment": 16, "zero_initialized": False},
        "launch": {
            "arguments": [
                {"kind": "slot", "slot": "output", "wire_type": "tensor_ptr"},
                {"kind": "slot", "slot": "input", "wire_type": "tensor_ptr"},
                {"kind": "slot", "slot": "n", "wire_type": "scalar_i32"},
            ],
            "grid": {
                "x": {"kind": "const", "value": 1},
                "y": {"kind": "const", "value": 1},
                "z": {"kind": "const", "value": 1},
            },
            "block": {"x": 32, "y": 1, "z": 1},
            "dynamic_shared_bytes": {"kind": "const", "value": 0},
        },
        "graph_capture": "capture_only",
        "graph_update_slots": [],
        "numerical_class": {"kind": "diagnostic_only"},
        "determinism": "required",
        "bit_affecting": False,
        "required_entitlement_features": [],
        "requires": [],
        "conflicts": [],
        "provides": [],
        "joint_with": [],
    }
    return {
        "pack_id": "com.zeraix.imparo.community.cuda.smoke",
        "pack_version": "1.0.0",
        "distribution": "community",
        "release_channel": "imparo.community.stable",
        "required_entitlement_features": [],
        "engine_api": {"min": 1, "max_exclusive": 2},
        "backend_abi": {"min": 26, "max_exclusive": 27},
        "driver_min": 12000,
        "math_mode": "strict",
        "choice_group": group,
        "variant": variant,
    }


def write_source(path: Path, *, forbidden_import: str | None = None) -> None:
    imported = f"import {forbidden_import}\n" if forbidden_import else "import triton\n"
    path.write_text(
        imported
        + f"@triton.jit\ndef {SYMBOL}(output, input, n):\n    return\n"
        + "IMPARO_AOT = "
        + repr(
            {
                "kernel": SYMBOL,
                "signature": ["*fp32:16", "*fp32:16", "i32"],
                "num_warps": 1,
                "num_stages": 1,
                "manifest": manifest_template(),
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_recipe(path: Path) -> None:
    for relative in RECIPE_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"recipe:{relative}\n", encoding="utf-8")


class FakeCompiler:
    def __init__(self, *, global_scratch: int = 0, profile_scratch: int = 0):
        self.global_scratch = global_scratch
        self.profile_scratch = profile_scratch

    def compile(self, source: Path, spec: object, target: str) -> CompileResult:
        return CompileResult(
            cubin=minimal_elf(),
            symbol=SYMBOL,
            registers_per_thread=8,
            static_shared_bytes=0,
            local_memory_bytes=0,
            global_scratch_bytes=self.global_scratch,
            profile_scratch_bytes=self.profile_scratch,
        )


class BuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_triton = sys.modules.get("triton")
        fake = types.ModuleType("triton")
        fake.__path__ = []
        fake.__version__ = "3.8.0"
        fake.jit = lambda function: function
        sys.modules["triton"] = fake

    def tearDown(self) -> None:
        if self.old_triton is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = self.old_triton
        self.temp.cleanup()

    def lock(self):
        value = json.loads((ROOT / "toolchain.lock").read_text(encoding="utf-8"))
        value["python"]["version"] = sys.version.split()[0]
        value["status"] = "release_ready"
        value["builder"]["image_sha256"] = "9" * 64
        value["builder"]["image_identity_kind"] = "oci_manifest_digest"
        path = self.root / "test.lock"
        path.write_bytes(canonical_json(value))
        return load_toolchain_lock(path)

    def test_default_lock_is_strict_and_uses_real_source_pin(self) -> None:
        lock = load_toolchain_lock(ROOT / "toolchain.lock")
        self.assertEqual(
            lock.triton_revision,
            "b252c7c4cea49216b27e5a47db061f9d0dbf49f3",
        )
        self.assertEqual(
            lock.raw["triton"]["source_sha256"],
            "1e25ae6d0dd4a1172002c53bb516b5a3c0ceb33c72900e7ad70394c58e1bd6a5",
        )
        forged = {
            "IMPARO_BUILDER_IMAGE_SHA256": "9" * 64,
            "IMPARO_TRITON_SOURCE_REVISION": lock.triton_revision,
            "IMPARO_TRITON_LLVM_REVISION": lock.llvm_revision,
            "IMPARO_PTXAS_VERSION": lock.ptxas_version,
            "IMPARO_CUOBJDUMP_VERSION": lock.cuobjdump_version,
        }
        with mock.patch.dict("os.environ", forged, clear=True), self.assertRaisesRegex(
            BuildError, "not release-ready"
        ):
            lock.require_build_environment()

    def test_source_pin_feasibility_requires_explicit_flag_and_exact_env(self) -> None:
        value = json.loads((ROOT / "toolchain.lock").read_text(encoding="utf-8"))
        value["python"]["version"] = sys.version.split()[0]
        value["builder"]["image_sha256"] = "9" * 64
        value["builder"]["image_identity_kind"] = (
            "oci_config_digest_local_feasibility"
        )
        path = self.root / "feasibility.lock"
        path.write_bytes(canonical_json(value))
        lock = load_toolchain_lock(path)
        exact = {
            "IMPARO_BUILDER_IMAGE_SHA256": "9" * 64,
            "IMPARO_TRITON_SOURCE_REVISION": lock.triton_revision,
            "IMPARO_TRITON_LLVM_REVISION": lock.llvm_revision,
            "IMPARO_PTXAS_VERSION": lock.ptxas_version,
            "IMPARO_CUOBJDUMP_VERSION": lock.cuobjdump_version,
        }
        with mock.patch.dict("os.environ", exact, clear=True):
            with self.assertRaisesRegex(BuildError, "not release-ready"):
                lock.require_build_environment()
            lock.require_build_environment(feasibility_source_pin=True)

    def test_real_builder_output_marks_and_gates_feasibility(self) -> None:
        value = json.loads((ROOT / "toolchain.lock").read_text(encoding="utf-8"))
        value["python"]["version"] = sys.version.split()[0]
        value["builder"]["image_sha256"] = "9" * 64
        value["builder"]["image_identity_kind"] = (
            "oci_config_digest_local_feasibility"
        )
        lock_path = self.root / "source-pin.lock"
        lock_path.write_bytes(canonical_json(value))
        source = self.root / "feasibility.py"
        write_source(source)
        recipe = self.root / "feasibility-recipe"
        write_recipe(recipe)
        output = self.root / "feasibility-pack"
        build_pack(
            lock=load_toolchain_lock(lock_path),
            target="cuda:86:32",
            source=source,
            output=output,
            compiler=FakeCompiler(),
            enforce_environment=False,
            feasibility_source_pin=True,
            recipe_root=recipe,
        )
        provenance = json.loads((output / "provenance.json").read_bytes())
        self.assertEqual(provenance["toolchain_status"], "source_pin_pre_release")
        self.assertIs(provenance["feasibility_only"], True)
        with self.assertRaisesRegex(BuildError, "not admissible for release"):
            verify_pack(output / "manifest.json")
        verified = verify_pack(
            output / "manifest.json", allow_feasibility_smoke=True
        )
        self.assertIs(verified["feasibility_only"], True)

    def test_final_builder_identity_cannot_alias_base_image(self) -> None:
        value = json.loads((ROOT / "toolchain.lock").read_text(encoding="utf-8"))
        value["status"] = "release_ready"
        value["builder"]["image_sha256"] = value["builder"]["base_image_sha256"]
        value["builder"]["image_identity_kind"] = "oci_manifest_digest"
        path = self.root / "aliased-image.lock"
        path.write_bytes(canonical_json(value))
        with self.assertRaisesRegex(BuildError, "cannot equal its base"):
            load_toolchain_lock(path)

    def test_release_lock_rejects_local_config_digest_kind(self) -> None:
        value = json.loads((ROOT / "toolchain.lock").read_text(encoding="utf-8"))
        value["status"] = "release_ready"
        value["builder"]["image_sha256"] = "9" * 64
        value["builder"]["image_identity_kind"] = (
            "oci_config_digest_local_feasibility"
        )
        path = self.root / "local-config-release.lock"
        path.write_bytes(canonical_json(value))
        with self.assertRaisesRegex(BuildError, "pullable OCI manifest"):
            load_toolchain_lock(path)

    def test_duplicate_lock_key_fails_closed(self) -> None:
        path = self.root / "duplicate.lock"
        path.write_text('{"schema":1,"schema":1}', encoding="utf-8")
        with self.assertRaisesRegex(BuildError, "duplicate JSON key"):
            load_toolchain_lock(path)

    def test_only_triton_imports_are_allowed(self) -> None:
        source = self.root / "bad.py"
        write_source(source, forbidden_import="torch")
        with self.assertRaisesRegex(BuildError, "forbidden module 'torch'"):
            audit_source_imports(source)
        collision = self.root / "prefix-collision.py"
        write_source(collision, forbidden_import="triton.language_evil")
        with self.assertRaisesRegex(BuildError, "triton.language_evil"):
            audit_source_imports(collision)

    def test_nonzero_hidden_scratch_is_rejected(self) -> None:
        source = self.root / "smoke.py"
        write_source(source)
        recipe = self.root / "recipe"
        write_recipe(recipe)
        for name, compiler in [
            ("global", FakeCompiler(global_scratch=1)),
            ("profile", FakeCompiler(profile_scratch=1)),
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(
                BuildError, "non-zero Triton"
            ):
                build_pack(
                    lock=self.lock(),
                    target="cuda:86:32",
                    source=source,
                    output=self.root / name,
                    compiler=compiler,
                    enforce_environment=False,
                    recipe_root=recipe,
                )

    def test_two_clean_builds_are_byte_identical_and_verify(self) -> None:
        source = self.root / "smoke.py"
        write_source(source)
        recipe = self.root / "recipe"
        write_recipe(recipe)
        outputs = [self.root / "one", self.root / "two"]
        first = build_pack(
            lock=self.lock(),
            target="cuda:86:32",
            source=source,
            output=outputs[0],
            compiler=FakeCompiler(),
            enforce_environment=False,
            recipe_root=recipe,
        )
        (recipe / "README.md").write_text(
            "documentation does not alter recipe identity\n", encoding="utf-8"
        )
        second = build_pack(
            lock=self.lock(),
            target="cuda:86:32",
            source=source,
            output=outputs[1],
            compiler=FakeCompiler(),
            enforce_environment=False,
            recipe_root=recipe,
        )
        results = [first, second]
        self.assertEqual(results[0].manifest_sha256, results[1].manifest_sha256)
        for relative in [
            "manifest.json",
            "SBOM.spdx.json",
            "THIRD_PARTY_NOTICES",
            "provenance.json",
            f"modules/{results[0].module_sha256}.cubin",
        ]:
            self.assertEqual(
                (outputs[0] / relative).read_bytes(),
                (outputs[1] / relative).read_bytes(),
            )
        verified = verify_pack(outputs[0] / "manifest.json")
        self.assertFalse(verified["signed"])
        manifest = json.loads((outputs[0] / "manifest.json").read_bytes())
        resources = manifest["variants"][0]["resources"]
        self.assertEqual(resources["registers_per_thread_max"], 8)
        self.assertEqual(resources["local_memory_bytes_max"], 0)
        sbom = json.loads((outputs[0] / "SBOM.spdx.json").read_bytes())
        self.assertEqual(
            {package["name"] for package in sbom["packages"]},
            {
                "triton",
                "llvm-project-toolchain",
                "NVIDIA CUDA Toolkit",
                "nvidia-cuda-nvcc-cu12-ptxas",
            },
        )

    def test_missing_semantic_recipe_input_is_rejected(self) -> None:
        source = self.root / "missing-recipe.py"
        write_source(source)
        recipe = self.root / "incomplete-recipe"
        write_recipe(recipe)
        (recipe / RECIPE_FILES[0]).unlink()
        with self.assertRaisesRegex(BuildError, "builder recipe is incomplete"):
            build_pack(
                lock=self.lock(),
                target="cuda:86:32",
                source=source,
                output=self.root / "missing-recipe-pack",
                compiler=FakeCompiler(),
                enforce_environment=False,
                recipe_root=recipe,
            )

    def test_pack_extra_python_and_debug_section_are_rejected(self) -> None:
        with self.assertRaisesRegex(BuildError, "forbidden section"):
            verify_release_cubin(minimal_elf(".debug_info"), 86)
        with self.assertRaisesRegex(BuildError, "real-SM 80"):
            verify_release_cubin(minimal_elf(sm=80), 86)

    def test_cuobjdump_parser_matches_exact_colon_terminated_symbol(self) -> None:
        output = f"""Resource usage:\n Function {SYMBOL}suffix:\n  REG:99 STACK:7 LOCAL:11 SHARED:0\n Function {SYMBOL}:\n  REG:17 STACK:3 LOCAL:5 SHARED:0\n"""
        completed = subprocess.CompletedProcess(
            args=["cuobjdump"], returncode=0, stdout=output, stderr=""
        )
        with mock.patch(
            "imparo_triton_pack.aot.subprocess.run", return_value=completed
        ):
            self.assertEqual(_resource_usage(minimal_elf(), SYMBOL), (17, 8))

    def test_aot_compiler_does_not_bind_active_driver(self) -> None:
        source = self.root / "driverless.py"
        write_source(source)
        spec = load_aot_spec(source)

        class Kernel:
            arg_names = ["output", "input", "n"]

            def create_binder(self):
                raise AssertionError("host driver binding is forbidden")

        kernel = Kernel()
        fake_triton = types.ModuleType("triton")
        fake_triton.__path__ = []
        fake_triton.__version__ = "3.8.0"
        fake_triton.jit = lambda function: kernel
        fake_backends = types.ModuleType("triton.backends")
        fake_backends.__path__ = []
        fake_backend_compiler = types.ModuleType("triton.backends.compiler")
        fake_compiler = types.ModuleType("triton.compiler")

        class Target:
            def __init__(self, backend, arch, warp_size):
                self.backend = backend
                self.arch = arch
                self.warp_size = warp_size

        class Source:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Backend:
            binary_ext = "cubin"

            def parse_options(self, options):
                return types.SimpleNamespace(**options)

        fake_backend_compiler.GPUTarget = Target
        fake_compiler.ASTSource = Source
        fake_compiler.make_backend = lambda target: Backend()
        fake_compiler.compile = lambda *args, **kwargs: None
        fake_triton.compiler = types.SimpleNamespace(
            make_backend=lambda target: Backend()
        )
        cache_paths: list[str] = []

        def compile_without_driver(*args, **kwargs):
            cache_paths.append(os.environ["TRITON_CACHE_DIR"])
            return types.SimpleNamespace(
                asm={"cubin": minimal_elf()},
                metadata=types.SimpleNamespace(
                    shared=0,
                    global_scratch_size=0,
                    profile_scratch_size=0,
                ),
                name=SYMBOL,
            )

        fake_triton.compile = compile_without_driver
        modules = {
            "triton": fake_triton,
            "triton.backends": fake_backends,
            "triton.backends.compiler": fake_backend_compiler,
            "triton.compiler": fake_compiler,
        }
        with mock.patch.dict(sys.modules, modules), mock.patch(
            "imparo_triton_pack.aot._resource_usage", return_value=(8, 0)
        ):
            compiler = TritonCompiler("3.8.0")
            result = compiler.compile(
                source, spec, "cuda:80:32"
            )
            compiler.compile(source, spec, "cuda:80:32")
        self.assertEqual(result.symbol, SYMBOL)
        self.assertEqual(len(cache_paths), 2)
        self.assertNotEqual(cache_paths[0], cache_paths[1])
        self.assertFalse(any(Path(path).exists() for path in cache_paths))

    def test_unknown_target_and_lock_environment_drift_fail_closed(self) -> None:
        source = self.root / "smoke.py"
        write_source(source)
        with self.assertRaisesRegex(BuildError, "target must be exactly"):
            build_pack(
                lock=self.lock(),
                target="cuda:89:32",
                source=source,
                output=self.root / "wrong-sm",
                compiler=FakeCompiler(),
                enforce_environment=False,
            )
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(
            BuildError, "builder image identity"
        ):
            self.lock().require_build_environment()

    def test_tampered_inventory_fails_closed(self) -> None:
        source = self.root / "smoke.py"
        write_source(source)
        recipe = self.root / "recipe-tamper"
        write_recipe(recipe)
        output = self.root / "pack"
        build_pack(
            lock=self.lock(),
            target="cuda:86:32",
            source=source,
            output=output,
            compiler=FakeCompiler(),
            enforce_environment=False,
            recipe_root=recipe,
        )
        (output / "payload.py").write_text("raise SystemExit\n", encoding="utf-8")
        with self.assertRaisesRegex(BuildError, "inventory differs"):
            verify_pack(output / "manifest.json")

    def test_signature_message_and_envelope_match_rust_order(self) -> None:
        manifest = self.root / "manifest.json"
        manifest.write_bytes(b"{}")
        expected = bytes.fromhex(
            "696d7061726f2d70726f6772616d2d7061636b2d763100"
            "0200000000000000"
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            "7b7d"
        )
        self.assertEqual(signature_message(b"{}"), expected)
        key = self.root / "key.pem"
        key.write_text("test-only mock key", encoding="utf-8")
        output = self.root / "manifest.sig"
        completed = subprocess.CompletedProcess(
            args=["openssl"], returncode=0, stdout=b"\x07" * 64, stderr=b""
        )
        with mock.patch(
            "imparo_triton_pack.signing.subprocess.run", return_value=completed
        ):
            sign_manifest(manifest, key, "imparo.test.community", output)
        raw = output.read_bytes()
        expected_prefix = b'{"schema":1,"domain":"imparo-program-pack-v1","algorithm":"ed25519","key_id":'
        self.assertTrue(raw.startswith(expected_prefix))
        with self.assertRaisesRegex(BuildError, "namespaced id"):
            sign_manifest(manifest, key, "Invalid..Key", self.root / "bad.sig")

    def test_signature_verification_requires_key_and_rejects_wrong_key(self) -> None:
        source = self.root / "signed-smoke.py"
        write_source(source)
        recipe = self.root / "signed-recipe"
        write_recipe(recipe)
        output = self.root / "signed-pack"
        build_pack(
            lock=self.lock(),
            target="cuda:86:32",
            source=source,
            output=output,
            compiler=FakeCompiler(),
            enforce_environment=False,
            recipe_root=recipe,
        )
        key = self.root / "mock-private.pem"
        key.write_text("mock private", encoding="utf-8")
        signed = subprocess.CompletedProcess(
            args=["openssl"], returncode=0, stdout=b"\x05" * 64, stderr=b""
        )
        with mock.patch(
            "imparo_triton_pack.signing.subprocess.run", return_value=signed
        ):
            sign_manifest(
                output / "manifest.json",
                key,
                "imparo.test.community",
                output / "manifest.sig",
            )
        with self.assertRaisesRegex(BuildError, "requires an Ed25519"):
            verify_pack(output / "manifest.json", require_signature=True)
        public_key = self.root / "public.pem"
        public_key.write_text("mock public", encoding="utf-8")
        accepted = subprocess.CompletedProcess(
            args=["openssl"], returncode=0, stdout=b"Signature Verified", stderr=b""
        )
        with mock.patch(
            "imparo_triton_pack.verify.subprocess.run", return_value=accepted
        ):
            self.assertTrue(
                verify_pack(
                    output / "manifest.json",
                    require_signature=True,
                    public_key=public_key,
                )["signed"]
            )
        rejected = subprocess.CompletedProcess(
            args=["openssl"], returncode=1, stdout=b"", stderr=b"bad signature"
        )
        with mock.patch(
            "imparo_triton_pack.verify.subprocess.run", return_value=rejected
        ), self.assertRaisesRegex(BuildError, "signature verification failed"):
            verify_pack(
                output / "manifest.json",
                require_signature=True,
                public_key=public_key,
            )
        envelope = json.loads((output / "manifest.sig").read_bytes())
        envelope["signature"] = "AA" + envelope["signature"][2:]
        (output / "manifest.sig").write_text(
            json.dumps(envelope, separators=(",", ":")), encoding="ascii"
        )
        with mock.patch(
            "imparo_triton_pack.verify.subprocess.run", return_value=rejected
        ), self.assertRaisesRegex(BuildError, "signature verification failed"):
            verify_pack(
                output / "manifest.json",
                require_signature=True,
                public_key=public_key,
            )


if __name__ == "__main__":
    unittest.main()
