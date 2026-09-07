from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

if __package__:
    from dev_harness import verify_program_pack_release as release
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dev_harness import verify_program_pack_release as release


def elf64(section_names: tuple[str, ...] = (".text.ip_smoke",), sm: int = 86) -> bytes:
    names = b"\0.shstrtab\0" + b"".join(name.encode("ascii") + b"\0" for name in section_names)
    name_offsets = {name: names.index(name.encode("ascii")) for name in section_names}
    shstr = names.index(b".shstrtab")
    header_size = 64
    section_data = b"\x90" * 64
    names_offset = header_size + len(section_data)
    table_offset = (names_offset + len(names) + 7) & ~7
    section_count = 2 + len(section_names)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01\x33\x07" + b"\0" * 7,
        2, 190, 1, 0, 0, table_offset, sm, 64, 0, 0, 64, section_count, 1,
    )
    pad = b"\0" * (table_offset - names_offset - len(names))
    sections = [b"\0" * 64]
    sections.append(struct.pack("<IIQQQQIIQQ", shstr, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0))
    for name in section_names:
        sections.append(struct.pack("<IIQQQQIIQQ", name_offsets[name], 1, 0, 0, header_size, len(section_data), 0, 0, 16, 0))
    return header + section_data + names + pad + b"".join(sections)


class PackFixture:
    def __init__(
        self,
        root: Path,
        sm: int = 86,
        sections: tuple[str, ...] = (".text.ip_smoke",),
        cubin_sm: int | None = None,
        feasibility_only: bool = False,
    ) -> None:
        self.root = root
        module = elf64(sections, sm if cubin_sm is None else cubin_sm)
        module_hash = hashlib.sha256(module).hexdigest()
        module_path = f"modules/{module_hash}.cubin"
        sbom = {
            "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT", "name": "smoke",
            "documentNamespace": "https://imparo.dev/spdx/smoke",
            "creationInfo": {"created": "2026-08-28T00:00:00Z", "creators": ["Tool: imparo-triton-pack"]},
            "packages": [{"name": "triton", "SPDXID": "SPDXRef-Package-triton", "licenseDeclared": "MIT"}],
        }
        sbom_raw = json.dumps(sbom, sort_keys=True, separators=(",", ":")).encode()
        notice = b"Third-party license notice: Triton is licensed under MIT.\n"
        toolchain_status = "source_pin_pre_release" if feasibility_only else "release_ready"
        provenance = json.dumps(
            {
                "schema": 1,
                "builder": "imparo-triton-pack",
                "toolchain_status": toolchain_status,
                "feasibility_only": feasibility_only,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        manifest = {
            "schema": 1, "target": {"sm": sm, "warp_size": 32},
            "modules": [{"file": module_path, "bytes": len(module), "sha256": module_hash}],
            "sbom_sha256": hashlib.sha256(sbom_raw).hexdigest(),
            "notices_sha256": hashlib.sha256(notice).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance).hexdigest(),
        }
        manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        signature = {
            "schema": 1, "domain": "imparo-program-pack-v1", "algorithm": "ed25519",
            "key_id": "imparo.release.test", "signature": "A" * 86,
            "manifest_bytes": len(manifest_raw),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
        files = {
            module_path: module, "SBOM.spdx.json": sbom_raw,
            "THIRD_PARTY_NOTICES": notice, "provenance.json": provenance,
            "manifest.json": manifest_raw,
            "manifest.sig": (
                f'{{"schema":1,"domain":"imparo-program-pack-v1","algorithm":"ed25519",'
                f'"key_id":"imparo.release.test","manifest_bytes":{len(manifest_raw)},'
                f'"manifest_sha256":"{hashlib.sha256(manifest_raw).hexdigest()}",'
                f'"signature":"{"A" * 86}"}}'
            ).encode(),
        }
        for path, raw in files.items():
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        self.module = root / module_path


class ProgramPackReleaseTests(unittest.TestCase):
    def test_complete_signed_data_only_pack_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary))
            self.assertEqual(release.scan_pack(fixture.root, 86)["sm"], 86)

    def test_extra_source_ir_host_code_and_unsigned_pack_fail(self) -> None:
        for name in ("kernel.py", "kernel.ptx", "host.dll", "signing-message.bin"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = PackFixture(Path(temporary))
                (fixture.root / name).write_bytes(b"forbidden")
                with self.assertRaises(release.ReleaseBoundaryError):
                    release.scan_pack(fixture.root, 86)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary))
            (fixture.root / "manifest.sig").unlink()
            with self.assertRaises(release.ReleaseBoundaryError):
                release.scan_pack(fixture.root, 86)

    def test_wrong_sm_hash_and_sidecar_binding_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary))
            with self.assertRaises(release.ReleaseBoundaryError):
                release.scan_pack(fixture.root, 80)
            raw = bytearray(fixture.module.read_bytes())
            raw[64] ^= 1
            fixture.module.write_bytes(raw)
            with self.assertRaises(release.ReleaseBoundaryError):
                release.scan_pack(fixture.root, 86)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary), sm=86, cubin_sm=80)
            with self.assertRaisesRegex(release.ReleaseBoundaryError, "ELF target"):
                release.scan_pack(fixture.root, 86)

    def test_signature_requires_rust_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary))
            signature_path = fixture.root / "manifest.sig"
            value = json.loads(signature_path.read_text())
            signature_path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
            with self.assertRaisesRegex(release.ReleaseBoundaryError, "canonical"):
                release.scan_pack(fixture.root, 86)

    def test_feasibility_pack_is_rejected_by_default_and_smoke_only_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary), feasibility_only=True)
            with self.assertRaisesRegex(release.ReleaseBoundaryError, "release boundary"):
                release.scan_pack(fixture.root, 86)
            report = release.scan_pack(
                fixture.root, 86, allow_feasibility_smoke=True
            )
            self.assertTrue(report["feasibility_only"])

    def test_real_builder_output_obeys_release_scanner_feasibility_contract(self) -> None:
        tools = Path(__file__).resolve().parents[1] / "tools" / "triton-pack"
        sys.path.insert(0, str(tools))
        try:
            from imparo_triton_pack import build_pack, load_toolchain_lock
            from imparo_triton_pack.aot import CompileResult
            from imparo_triton_pack.common import canonical_json
        finally:
            sys.path.pop(0)

        class FakeCompiler:
            def compile(self, source, spec, target):
                del source, target
                return CompileResult(
                    cubin=elf64((f".text.{spec.kernel_name}",), sm=86),
                    symbol=spec.kernel_name,
                    registers_per_thread=8,
                    static_shared_bytes=0,
                    local_memory_bytes=0,
                    global_scratch_bytes=0,
                    profile_scratch_bytes=0,
                )

        fake_triton = types.ModuleType("triton")
        fake_triton.__path__ = []
        fake_triton.__version__ = "3.8.0"
        fake_triton.jit = lambda function: function
        fake_language = types.ModuleType("triton.language")
        fake_triton.language = fake_language
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            lock_value = json.loads((tools / "toolchain.lock").read_text(encoding="utf-8"))
            lock_value["python"]["version"] = sys.version.split()[0]
            lock_value["builder"]["image_sha256"] = "9" * 64
            lock_value["builder"]["image_identity_kind"] = (
                "oci_config_digest_local_feasibility"
            )
            lock_path = temporary_root / "source-pin.lock"
            lock_path.write_bytes(canonical_json(lock_value))
            output = temporary_root / "pack"
            with mock.patch.dict(
                sys.modules,
                {"triton": fake_triton, "triton.language": fake_language},
            ):
                build_pack(
                    lock=load_toolchain_lock(lock_path),
                    target="cuda:86:32",
                    source=tools.parents[1] / "program-packs" / "community" / "cuda" / "smoke.py",
                    output=output,
                    compiler=FakeCompiler(),
                    enforce_environment=False,
                    feasibility_source_pin=True,
                    recipe_root=tools,
                )
            manifest_raw = (output / "manifest.json").read_bytes()
            manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
            (output / "manifest.sig").write_bytes(
                (
                    '{"schema":1,"domain":"imparo-program-pack-v1",'
                    '"algorithm":"ed25519","key_id":"imparo.integration.test",'
                    f'"manifest_bytes":{len(manifest_raw)},'
                    f'"manifest_sha256":"{manifest_hash}",'
                    f'"signature":"{"A" * 86}"}}'
                ).encode("ascii")
            )
            with self.assertRaisesRegex(release.ReleaseBoundaryError, "release boundary"):
                release.scan_pack(output, 86)
            report = release.scan_pack(
                output, 86, allow_feasibility_smoke=True
            )
            self.assertIs(report["feasibility_only"], True)

    def test_debug_section_and_absolute_source_path_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary), sections=(".debug_line",))
            manifest = json.loads((fixture.root / "manifest.json").read_text())
            raw = fixture.module.read_bytes()
            manifest["modules"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
            (fixture.root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
            signature = json.loads((fixture.root / "manifest.sig").read_text())
            manifest_raw = (fixture.root / "manifest.json").read_bytes()
            signature["manifest_bytes"] = len(manifest_raw)
            signature["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
            (fixture.root / "manifest.sig").write_text(
                f'{{"schema":1,"domain":"imparo-program-pack-v1","algorithm":"ed25519",'
                f'"key_id":"imparo.release.test","manifest_bytes":{len(manifest_raw)},'
                f'"manifest_sha256":"{hashlib.sha256(manifest_raw).hexdigest()}",'
                f'"signature":"{"A" * 86}"}}'
            )
            with self.assertRaisesRegex(release.ReleaseBoundaryError, "debug"):
                release.scan_pack(fixture.root, 86)

        # Assemble the synthetic leak in memory so the reviewed public test source
        # does not itself contain a real-looking developer absolute path.
        raw = elf64() + b"/" + b"home/fixture-user/private/kernel.py\0"
        with self.assertRaisesRegex(release.ReleaseBoundaryError, "source path"):
            release._scan_cubin(raw, "modules/x.cubin", 86)

    def test_sbom_license_and_notice_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PackFixture(Path(temporary))
            sbom = json.loads((fixture.root / "SBOM.spdx.json").read_text())
            sbom["packages"][0]["licenseDeclared"] = "NOASSERTION"
            with self.assertRaises(release.ReleaseBoundaryError):
                release._verify_sbom(sbom)
            (fixture.root / "THIRD_PARTY_NOTICES").write_bytes(b"placeholder")
            manifest = json.loads((fixture.root / "manifest.json").read_text())
            manifest["notices_sha256"] = hashlib.sha256(b"placeholder").hexdigest()
            manifest_raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            (fixture.root / "manifest.json").write_bytes(manifest_raw)
            signature = json.loads((fixture.root / "manifest.sig").read_text())
            signature["manifest_bytes"] = len(manifest_raw)
            signature["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
            (fixture.root / "manifest.sig").write_text(
                f'{{"schema":1,"domain":"imparo-program-pack-v1","algorithm":"ed25519",'
                f'"key_id":"imparo.release.test","manifest_bytes":{len(manifest_raw)},'
                f'"manifest_sha256":"{hashlib.sha256(manifest_raw).hexdigest()}",'
                f'"signature":"{"A" * 86}"}}'
            )
            with self.assertRaises(release.ReleaseBoundaryError):
                release.scan_pack(fixture.root, 86)

    def test_cargo_and_binary_dependency_scans(self) -> None:
        clean = "imparo-server v0.1.0\nserde v1.0.0\n"
        dirty = clean + "├── pyo3 v0.23.0\n"
        self.assertNotIn("pyo3", release.parse_cargo_tree_packages(clean))
        self.assertIn("pyo3", release.parse_cargo_tree_packages(dirty))
        completed = type("Completed", (), {"returncode": 0, "stdout": dirty, "stderr": ""})()
        with mock.patch.object(release.subprocess, "run", return_value=completed):
            with self.assertRaises(release.ReleaseBoundaryError):
                release.scan_cargo_runtime(Path("."))
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "engine.bin"
            binary.write_bytes(b"MZ\0kernel32.dll\0")
            release.scan_runtime_binary(binary)
            binary.write_bytes(b"MZ\0python311.dll\0")
            with self.assertRaises(release.ReleaseBoundaryError):
                release.scan_runtime_binary(binary)


if __name__ == "__main__":
    unittest.main()
