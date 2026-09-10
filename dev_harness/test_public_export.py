from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

if __package__:
    from dev_harness import public_export, verify_public_export
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dev_harness import public_export, verify_public_export


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "dev_harness" / "public-export.allowlist"
EXPECTED_ALLOWLIST_SHA256 = "b9efb03338169034a63246b88451681e49b70a8ffbc4ba7073156303626b41c7"
REQUIRED_COMMUNITY_FILES = {
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/performance_report.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
}
REQUIRED_CONTROL_PLANE = {
    ".gitattributes",
    ".github/workflows/ci.yml",
    "dev_harness/public-export.allowlist",
    "dev_harness/public_export.py",
    "dev_harness/sync_public.sh",
    "dev_harness/test_program_pack_release.py",
    "dev_harness/test_public_export.py",
    "dev_harness/verify_program_pack_release.py",
    "dev_harness/verify_public_export.py",
}
REQUIRED_PROGRAM_PACK_PUBLIC = {
    "crates/imparo-program-pack/Cargo.toml",
    "crates/imparo-program-pack/fuzz/Cargo.lock",
    "crates/imparo-program-pack/fuzz/Cargo.toml",
    "crates/imparo-program-pack/fuzz/fuzz_targets/launch_expression.rs",
    "crates/imparo-program-pack/fuzz/fuzz_targets/manifest.rs",
    "crates/imparo-program-pack/fuzz/fuzz_targets/signature.rs",
    "crates/imparo-program-pack/src/extensions.rs",
    "crates/imparo-program-pack/src/identity.rs",
    "crates/imparo-program-pack/src/install.rs",
    "crates/imparo-program-pack/src/lib.rs",
    "crates/imparo-program-pack/src/manifest.rs",
    "crates/imparo-program-pack/src/policy.rs",
    "crates/imparo-program-pack/src/trust.rs",
    "program-pack-policy.toml",
    "schemas/__init__.py",
    "schemas/examples/program-contract-v1.minimal.json",
    "schemas/examples/program-pack-signature-v1.minimal.json",
    "schemas/examples/program-pack-v1.minimal.json",
    "schemas/program-contract-v1.schema.json",
    "schemas/program-pack-signature-v1.schema.json",
    "schemas/program-pack-v1.schema.json",
    "schemas/requirements-test.txt",
    "schemas/test_program_pack_schemas.py",
    "schemas/validate_with_jsonschema.py",
}
REQUIRED_PROGRAM_BRIDGE_PUBLIC = {
    "crates/imparo-cuda/native/program_pack.cu",
    "crates/imparo-cuda/native/program_pack.h",
    "crates/imparo-cuda/native/program_pack_test.cu",
    "crates/imparo-cuda/src/program_pack/constraints.rs",
    "crates/imparo-cuda/src/program_pack/launch.rs",
    "crates/imparo-cuda/src/program_pack/loader.rs",
    "crates/imparo-cuda/src/program_pack/mod.rs",
    "crates/imparo-cuda/src/program_pack/profile_identity.rs",
    "crates/imparo-cuda/src/program_pack/registry.rs",
    "crates/imparo-cuda/src/program_pack/sm86_smoke.rs",
    "crates/imparo-cuda/tests/program_pack_contract.rs",
    "docs/evidence/triton/pr-e.md",
}
REQUIRED_TRITON_BUILDER_PUBLIC = {
    "dev_harness/test_program_pack_release.py",
    "dev_harness/test_triton_pack_smoke.py",
    "dev_harness/verify_program_pack_release.py",
    "docs/evidence/triton/pr-f.md",
    "program-packs/community/cuda/NOTICE",
    "program-packs/community/cuda/smoke-contract-v1.json",
    "program-packs/community/cuda/smoke-fixture-v1.json",
    "program-packs/community/cuda/smoke.py",
    "tools/triton-pack/Dockerfile",
    "tools/triton-pack/README.md",
    "tools/triton-pack/build.py",
    "tools/triton-pack/imparo_triton_pack/__init__.py",
    "tools/triton-pack/imparo_triton_pack/aot.py",
    "tools/triton-pack/imparo_triton_pack/artifacts.py",
    "tools/triton-pack/imparo_triton_pack/common.py",
    "tools/triton-pack/imparo_triton_pack/elf.py",
    "tools/triton-pack/imparo_triton_pack/errors.py",
    "tools/triton-pack/imparo_triton_pack/lock.py",
    "tools/triton-pack/imparo_triton_pack/signing.py",
    "tools/triton-pack/imparo_triton_pack/verify.py",
    "tools/triton-pack/lock_verify.py",
    "tools/triton-pack/requirements-build.lock",
    "tools/triton-pack/sign.py",
    "tools/triton-pack/tests/test_builder.py",
    "tools/triton-pack/toolchain.lock",
    "tools/triton-pack/verify.py",
}

REQUIRED_R2_FEASIBILITY_PUBLIC = {
    "config/kernel-lab-policy.toml",
    "crates/imparo-cuda/native/tests/kernel_lab_q4_q8.cu",
    "crates/imparo-cuda/native/tests/kernel_lab_rms_q8.cu",
    "dev_harness/aggregate_triton_gate_a.py",
    "dev_harness/audit_q4_gate_a_evidence.py",
    "dev_harness/finalize_q4_gate_a_evidence.py",
    "dev_harness/finalize_q4_sanitizer_evidence.py",
    "dev_harness/finalize_rms_gate_a_evidence.py",
    "dev_harness/finalize_triton_gate_a_decision.py",
    "dev_harness/kernel_lab_q4_contract.ps1",
    "dev_harness/kernel_lab_tools.ps1",
    "dev_harness/run_cuda_kernel_lab.ps1",
    "dev_harness/run_cuda_kernel_lab_q4.ps1",
    "dev_harness/run_q4_variant_sanitizer_matrix.py",
    "dev_harness/test_aggregate_triton_gate_a.py",
    "dev_harness/test_audit_q4_gate_a_evidence.py",
    "dev_harness/test_cuda_kernel_lab.py",
    "dev_harness/test_cuda_kernel_lab_q4_contract.py",
    "dev_harness/test_cuda_kernel_lab_q4_policy.py",
    "dev_harness/test_finalize_q4_gate_a_evidence.py",
    "dev_harness/test_finalize_rms_gate_a_evidence.py",
    "dev_harness/test_finalize_triton_gate_a_decision.py",
    "dev_harness/test_phase_a1_prefill_wall_source.py",
    "dev_harness/test_program_pack_release.py",
    "dev_harness/test_run_q4_variant_sanitizer_matrix.py",
    "dev_harness/test_triton_pack_smoke.py",
    "dev_harness/test_validate_q4_sanitizer_schema2.py",
    "dev_harness/test_verify_nsys_prefill_denominator.py",
    "dev_harness/validate_q4_sanitizer_schema2.py",
    "dev_harness/verify_nsys_prefill_denominator.py",
    "dev_harness/verify_program_pack_release.py",
    "docs/STATUS.md",
    "docs/evidence/triton/decision-a.md",
    "docs/evidence/triton/lab-a-phase-a1.md",
    "docs/evidence/triton/lab-a-q4-q8-builder.md",
    "docs/evidence/triton/lab-a-q4-q8-variants-sm86.md",
    "docs/evidence/triton/lab-a-q4-q8-windows-sm86-design.md",
    "docs/evidence/triton/lab-a-rms-q8-builder.md",
    "docs/evidence/triton/lab-a-rms-q8-windows-sm86.md",
    "docs/evidence/triton/pr-f.md",
    "program-packs/lab/q4-q8-mmq-sm86/run-a/lab-metadata.json",
    "program-packs/lab/q4-q8-mmq-sm86/run-b/lab-metadata.json",
    "program-packs/lab/q4-q8-mmq-sm86/variants-a/lab-metadata.json",
    "program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json",
    "program-packs/lab/rms-q8-sm86/run-e/lab-metadata.json",
    "program-packs/lab/rms-q8-sm86/run-f/lab-metadata.json",
    "program-packs/lab/rms-q8-sm86/run-g/lab-metadata.json",
    "tools/triton-kernel-lab/build.py",
    "tools/triton-kernel-lab/build_q4_q8_mmq.py",
    "tools/triton-kernel-lab/build_q4_q8_mmq_variants.py",
    "tools/triton-kernel-lab/q4_q8_mmq_10240x2560.py",
    "tools/triton-kernel-lab/q4_q8_mmq_2560x10240.py",
    "tools/triton-kernel-lab/q4_q8_mmq_variants_10240x2560.py",
    "tools/triton-kernel-lab/q4_q8_mmq_variants_2560x10240.py",
    "tools/triton-kernel-lab/rms_q8.py",
    "tools/triton-kernel-lab/tests/test_build.py",
    "tools/triton-kernel-lab/tests/test_q4_mmq.py",
    "tools/triton-kernel-lab/tests/test_q4_mmq_variants.py",
}

class TemporaryRepository:
    def __init__(self, parent: Path, name: str = "source") -> None:
        self.path = parent / name
        self.path.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "public-export@example.invalid")
        self.git("config", "user.name", "Public Export Test")
        self.git("config", "core.autocrlf", "false")
        self.allowlist_path = "public.allowlist"

    def git(self, *args: str, input_bytes: bytes | None = None) -> bytes:
        return subprocess.run(
            ["git", "-C", str(self.path), *args],
            input=input_bytes,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    def write(self, path: str, data: bytes) -> None:
        destination = self.path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def commit(self, files: dict[str, bytes], paths: list[str] | None = None) -> str:
        for path, data in files.items():
            self.write(path, data)
        selected = paths or sorted(
            {self.allowlist_path, *files}, key=lambda item: item.encode("utf-8")
        )
        self.write(self.allowlist_path, ("\n".join(selected) + "\n").encode("utf-8"))
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD").decode("ascii").strip()

    def plan(self, tree: str = "HEAD") -> public_export.ExportPlan:
        return public_export.build_plan(self.path, tree, self.allowlist_path)


class PublicAllowlistTests(unittest.TestCase):
    def test_production_allowlist_is_exact_ordinal_and_auditable(self) -> None:
        raw = ALLOWLIST.read_bytes()
        paths = public_export._parse_allowlist(
            raw, "dev_harness/public-export.allowlist"
        )
        self.assertEqual(len(paths), 301)
        self.assertEqual(paths, sorted(paths, key=lambda item: item.encode("utf-8")))
        self.assertEqual(set(paths) & REQUIRED_COMMUNITY_FILES, REQUIRED_COMMUNITY_FILES)
        self.assertEqual(set(paths) & REQUIRED_CONTROL_PLANE, REQUIRED_CONTROL_PLANE)
        self.assertEqual(
            set(paths) & REQUIRED_PROGRAM_PACK_PUBLIC, REQUIRED_PROGRAM_PACK_PUBLIC
        )
        self.assertEqual(
            set(paths) & REQUIRED_PROGRAM_BRIDGE_PUBLIC,
            REQUIRED_PROGRAM_BRIDGE_PUBLIC,
        )
        self.assertEqual(
            set(paths) & REQUIRED_TRITON_BUILDER_PUBLIC,
            REQUIRED_TRITON_BUILDER_PUBLIC,
        )
        self.assertEqual(
            set(paths) & REQUIRED_R2_FEASIBILITY_PUBLIC,
            REQUIRED_R2_FEASIBILITY_PUBLIC,
        )
        self.assertNotIn("handoff/triton-program-packs.md", paths)
        self.assertNotIn("docs/evidence/triton/pr-a.md", paths)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_ALLOWLIST_SHA256)

        wrapper = (ROOT / "dev_harness" / "sync_public.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("exec", wrapper)
        self.assertIn("public_export.py", wrapper)
        for retired in ("DROP=", "rm -rf", "find ", "tar -"):
            self.assertNotIn(retired, wrapper)

    def test_allowlist_rejects_format_order_duplicate_and_portable_collision(self) -> None:
        cases = (
            (b"a\r\n", "allowlist-format"),
            (b"a", "allowlist-format"),
            (b"b\na\n", "allowlist-order"),
            (b"a\na\n", "allowlist-duplicate"),
            (b"README\nreadme\n", "allowlist-collision"),
            (b"../escape\n", "path"),
            (("dir" + chr(92) + "file\n").encode("ascii"), "path"),
            (b"con.txt\n", "path-portability"),
            (b"dir/-option.txt\n", "path-portability"),
            (("x" * 256 + "\n").encode("ascii"), "path-portability"),
            (b"model.gguf\n", "private-path"),
            (b"handoff/triton-program-packs.md\n", "private-path"),
            (b"program-packs/lab/example/modules/a.cubin\n", "private-path"),
        )
        for raw, code in cases:
            with self.subTest(code=code, raw=raw):
                with self.assertRaises(public_export.PublicExportError) as raised:
                    public_export._parse_allowlist(raw, "fixture.allowlist")
                self.assertEqual(raised.exception.code, code)

        oversized = "".join(f"file-{index:04d}.txt\n" for index in range(public_export.MAX_FILES + 1))
        with self.assertRaises(public_export.PublicExportError) as raised:
            public_export._parse_allowlist(oversized.encode("ascii"), "fixture.allowlist")
        self.assertEqual(raised.exception.code, "allowlist-size")


class PublicExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def repo(self, name: str = "source") -> TemporaryRepository:
        return TemporaryRepository(self.root, name)

    def test_fixed_tree_reads_blobs_not_dirty_worktree(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"committed safe\n"})
        repository.write(
            "safe.txt", ("github" + "_" + "pat_" + "A" * 30).encode("ascii")
        )
        plan = repository.plan()
        exported = {item.path: item.data for item in plan.files}
        self.assertEqual(exported["safe.txt"], b"committed safe\n")

    def test_committed_secret_is_rejected_even_when_worktree_is_safe(self) -> None:
        repository = self.repo()
        secret = ("github" + "_" + "pat_" + "A" * 30).encode("ascii")
        repository.commit({"safe.txt": secret})
        repository.write("safe.txt", b"safe dirty replacement\n")
        with self.assertRaises(public_export.PublicExportError) as raised:
            repository.plan()
        self.assertEqual(raised.exception.code, "github-token")

    def test_secret_private_reference_and_unapproved_binary_scans(self) -> None:
        cases = (
            (
                "secret.txt",
                ("github" + "_" + "pat_" + "A" * 30).encode("ascii"),
                "github-token",
            ),
            (
                "private.txt",
                (
                    "https://github.com/"
                    + "zeraix/"
                    + "imparo"
                    + "-internal"
                ).encode("ascii"),
                "private-repository-url",
            ),
            ("binary.dat", b"text\x00binary", "binary-file"),
            (
                "lfs.txt",
                ("version https://git-" + "lfs.github.com/spec/v1\n").encode("ascii"),
                "git-lfs-pointer",
            ),
            (
                "url.txt",
                ("https://" + "user:password@host.example/path\n").encode("ascii"),
                "url-embedded-credentials",
            ),
            (
                "host.txt",
                ("service.build." + "corp\n").encode("ascii"),
                "private-hostname",
            ),
            (
                "drive.txt",
                ("Z:" + chr(92) + "private" + chr(92) + "file\n").encode("ascii"),
                "drive-absolute-path",
            ),
            (
                "encrypted-key.txt",
                ("-----BEGIN ENCRYPTED " + "PRIVATE KEY-----\n").encode("ascii"),
                "pem-private-key",
            ),
            (
                "dsa-key.txt",
                ("-----BEGIN DSA " + "PRIVATE KEY-----\n").encode("ascii"),
                "pem-private-key",
            ),
            (
                "pgp-key.txt",
                ("-----BEGIN PGP " + "PRIVATE KEY BLOCK-----\n").encode("ascii"),
                "pgp-private-key",
            ),
            (
                "putty-key.txt",
                ("PuTTY-" + "User-Key-File-3: ssh-ed25519\n").encode("ascii"),
                "putty-private-key",
            ),
            (
                "putty-lines.txt",
                ("Private-" + "Lines: 4\n").encode("ascii"),
                "putty-private-key",
            ),
        )
        for path, data, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(public_export.PublicExportError) as raised:
                    public_export._scan_blob(path, data)
                self.assertEqual(raised.exception.code, code)

    def test_approved_binary_requires_exact_reviewed_digest(self) -> None:
        path = "assets/imparo-wordmark-black.png"
        prefix, suffix, expected_bytes, expected_sha256 = public_export.BINARY_ALLOWLIST[path]
        reviewed = (ROOT / path).read_bytes()
        self.assertEqual(len(reviewed), expected_bytes)
        self.assertEqual(hashlib.sha256(reviewed).hexdigest(), expected_sha256)
        public_export._scan_blob(path, reviewed)
        with self.assertRaises(public_export.PublicExportError) as raised:
            public_export._scan_blob(path, b"not a png")
        self.assertEqual(raised.exception.code, "binary-signature")
        token = "github" + "_" + "pat_" + "A" * 30
        for encoded in (
            token.encode("ascii"),
            token.encode("utf-16le"),
            zlib.compress(token.encode("ascii")),
            b"payload",
        ):
            with self.subTest(encoding=len(encoded)):
                with self.assertRaises(public_export.PublicExportError) as raised:
                    public_export._scan_blob(path, prefix + encoded + suffix)
                self.assertEqual(raised.exception.code, "binary-digest")

    def test_allowlisted_symlink_mode_is_rejected(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"}, ["link", "public.allowlist"])
        oid = repository.git("hash-object", "-w", "--stdin", input_bytes=b"safe.txt").decode("ascii").strip()
        repository.git("update-index", "--add", "--cacheinfo", f"120000,{oid},link")
        repository.git("commit", "-qm", "symlink fixture")
        with self.assertRaises(public_export.PublicExportError) as raised:
            repository.plan()
        self.assertEqual(raised.exception.code, "git-mode")

    def test_allowlisted_submodule_mode_is_rejected(self) -> None:
        repository = self.repo()
        parent = repository.commit(
            {"safe.txt": b"safe\n"}, ["module", "public.allowlist"]
        )
        repository.git(
            "update-index", "--add", "--cacheinfo", f"160000,{parent},module"
        )
        repository.git("commit", "-qm", "gitlink fixture")
        with self.assertRaises(public_export.PublicExportError) as raised:
            repository.plan()
        self.assertEqual(raised.exception.code, "git-mode")

    def test_missing_allowlisted_path_is_rejected(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"}, ["missing.txt", "public.allowlist"])
        with self.assertRaises(public_export.PublicExportError) as raised:
            repository.plan()
        self.assertEqual(raised.exception.code, "missing-path")

    def test_dry_run_json_is_stable_and_does_not_write(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = public_export.main(
                [
                    "--source", str(repository.path),
                    "--tree", "HEAD",
                    "--allowlist", repository.allowlist_path,
                    "--dry-run",
                    "--json",
                ]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["ok"])
        self.assertFalse(report["applied"])
        self.assertEqual(report["operation"], "dry-run")
        self.assertEqual(report["file_count"], 2)
        self.assertFalse(report["destination_supplied"])
        self.assertNotIn("source", report)
        self.assertNotIn("destination", report)
        self.assertEqual(list(self.root.iterdir()), [repository.path])

    def test_invalid_source_json_returns_error_without_traceback(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = public_export.main(
                ["--source", str(self.root / "missing"), "--dry-run", "--json"]
            )
        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"][0]["code"], "source-path")
        self.assertNotIn("source", report)

    def test_apply_uses_verified_stage_and_refuses_dirty_managed_destination(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe v1\n"})
        destination = self.root / "public-mirror"
        public_export.apply_plan(repository.plan(), destination)
        self.assertEqual((destination / "safe.txt").read_bytes(), b"safe v1\n")
        self.assertTrue((destination.parent / ".public-mirror.imparo-public-export.json").is_file())
        (destination / "safe.txt").write_bytes(b"user edit\n")
        with self.assertRaises(public_export.PublicExportError) as raised:
            public_export.apply_plan(repository.plan(), destination)
        self.assertEqual(raised.exception.code, "destination-dirty")
        self.assertEqual((destination / "safe.txt").read_bytes(), b"user edit\n")

    @unittest.skipIf(os.name == "nt", "POSIX executable bits are not stable on Windows")
    def test_apply_rejects_executable_mode_tamper(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "mode-mirror"
        public_export.apply_plan(repository.plan(), destination)
        (destination / "safe.txt").chmod(0o755)
        with self.assertRaises(public_export.PublicExportError) as raised:
            public_export.apply_plan(repository.plan(), destination)
        self.assertEqual(raised.exception.code, "destination-dirty")

    def test_apply_refuses_unmanaged_nonempty_destination(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "unmanaged"
        destination.mkdir()
        (destination / "personal.txt").write_text("keep\n", encoding="utf-8")
        with self.assertRaises(public_export.PublicExportError) as raised:
            public_export.apply_plan(repository.plan(), destination)
        self.assertEqual(raised.exception.code, "destination-unmanaged")
        self.assertEqual((destination / "personal.txt").read_text(encoding="utf-8"), "keep\n")

    def test_marker_rejects_oversize_malformed_and_unbounded_inventory(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        plan = repository.plan()
        cases: list[tuple[str, bytes]] = []

        malformed = public_export._marker_payload(plan)
        malformed["files"][0]["bytes"] = True
        cases.append(("malformed", json.dumps(malformed).encode("utf-8")))

        too_many = public_export._marker_payload(plan)
        too_many["files"] = [too_many["files"][0]] * (public_export.MAX_FILES + 1)
        cases.append(("too-many", json.dumps(too_many).encode("utf-8")))

        cases.append(("oversize", b" " * (public_export.MAX_MARKER_BYTES + 1)))
        cases.append(("duplicate-key", b'{"schema":1,"schema":1}\n'))
        for name, raw in cases:
            with self.subTest(name=name):
                destination = self.root / f"marker-{name}"
                destination.mkdir()
                public_export._marker_path(destination).write_bytes(raw)
                with self.assertRaises(public_export.PublicExportError) as raised:
                    public_export._load_marker(destination)
                self.assertEqual(raised.exception.code, "destination-marker")

    def test_apply_rolls_back_all_known_files_on_install_failure(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe v1\n"})
        destination = self.root / "mirror"
        public_export.apply_plan(repository.plan(), destination)
        old_marker = (self.root / ".mirror.imparo-public-export.json").read_bytes()
        repository.commit({"safe.txt": b"safe v2\n"})
        updated = repository.plan()
        real_replace = public_export._durable_replace

        def fail_new_safe(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
            source_path = Path(source)
            if ".staging-" in str(source_path.parent) and source_path.name == "safe.txt":
                raise OSError("injected install failure")
            real_replace(source, target)

        with mock.patch.object(public_export, "_durable_replace", side_effect=fail_new_safe):
            with self.assertRaises(public_export.PublicExportError) as raised:
                public_export.apply_plan(updated, destination)
        self.assertEqual(raised.exception.code, "filesystem-error")
        self.assertEqual((destination / "safe.txt").read_bytes(), b"safe v1\n")
        self.assertEqual((self.root / ".mirror.imparo-public-export.json").read_bytes(), old_marker)
        self.assertFalse(any("staging-" in item.name or "rollback-" in item.name for item in self.root.iterdir()))

    def test_restore_failure_retains_recovery_and_next_apply_repairs(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe v1\n"})
        destination = self.root / "recoverable"
        public_export.apply_plan(repository.plan(), destination)
        old_marker = public_export._marker_path(destination).read_bytes()
        repository.commit({"safe.txt": b"safe v2\n"})
        updated = repository.plan()
        real_replace = public_export._durable_replace

        def fail_install_and_restore(
            source: os.PathLike[str] | str, target: os.PathLike[str] | str
        ) -> None:
            source_path = Path(source)
            if source_path.name == "safe.txt" and (
                ".staging-" in str(source_path.parent)
                or ".rollback-" in str(source_path.parent)
            ):
                raise OSError("private absolute sentinel restore failure")
            real_replace(source, target)

        with mock.patch.object(
            public_export, "_durable_replace", side_effect=fail_install_and_restore
        ):
            with self.assertRaises(public_export.PublicExportError) as raised:
                public_export.apply_plan(updated, destination)
        self.assertEqual(raised.exception.code, "recovery-required")
        self.assertEqual(public_export._marker_path(destination).read_bytes(), old_marker)
        self.assertTrue(public_export._recovery_path(destination).is_file())
        self.assertTrue(any("rollback-" in item.name for item in self.root.iterdir()))

        public_export.apply_plan(updated, destination)
        self.assertEqual((destination / "safe.txt").read_bytes(), b"safe v2\n")
        self.assertFalse(public_export._recovery_path(destination).exists())
        self.assertFalse(any("staging-" in item.name or "rollback-" in item.name for item in self.root.iterdir()))

    def test_committed_stale_transaction_is_cleaned_before_new_plan(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe v1\n"})
        destination = self.root / "committed-recovery"
        public_export.apply_plan(repository.plan(), destination)
        repository.commit({"safe.txt": b"safe v2\n"})

        with mock.patch.object(
            public_export,
            "_cleanup_transaction",
            side_effect=OSError("private cleanup sentinel"),
        ):
            with self.assertRaises(public_export.PublicExportError) as raised:
                public_export.apply_plan(repository.plan(), destination)
        self.assertEqual(raised.exception.code, "recovery-required")
        self.assertEqual((destination / "safe.txt").read_bytes(), b"safe v2\n")
        self.assertTrue(public_export._recovery_path(destination).is_file())

        repository.commit({"safe.txt": b"safe v3\n"})
        public_export.apply_plan(repository.plan(), destination)
        self.assertEqual((destination / "safe.txt").read_bytes(), b"safe v3\n")
        self.assertFalse(public_export._recovery_path(destination).exists())

    def test_apply_lock_is_exclusive_and_process_scoped(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "locked"
        with public_export._destination_lock(destination):
            with self.assertRaises(public_export.PublicExportError) as raised:
                public_export.apply_plan(repository.plan(), destination)
        self.assertEqual(raised.exception.code, "destination-locked")
        public_export.apply_plan(repository.plan(), destination)

    def test_process_exit_releases_stale_advisory_lock(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "stale-lock"
        script = (
            "import os,sys\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from dev_harness import public_export\n"
            "with public_export._destination_lock(Path(sys.argv[1])):\n"
            "    os._exit(0)\n"
        )
        subprocess.run([sys.executable, "-c", script, str(destination)], check=True)
        public_export.apply_plan(repository.plan(), destination)
        self.assertEqual((destination / "safe.txt").read_bytes(), b"safe\n")

    def test_lock_rejects_hardlink_and_symlink_without_touching_target(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        for kind in ("hardlink", "symlink"):
            with self.subTest(kind=kind):
                destination = self.root / f"{kind}-destination"
                victim = self.root / f"{kind}-victim.txt"
                victim.write_bytes(b"")
                lock = public_export._lock_path(destination)
                if kind == "hardlink":
                    os.link(victim, lock)
                else:
                    try:
                        os.symlink(victim, lock)
                    except OSError:
                        continue
                with self.assertRaises(public_export.PublicExportError) as raised:
                    public_export.apply_plan(repository.plan(), destination)
                self.assertEqual(raised.exception.code, "destination-lock")
                self.assertEqual(victim.read_bytes(), b"")

    def test_failure_reports_hash_untrusted_paths_and_hide_os_details(self) -> None:
        private_path = "customers/acme-secret/kernel.cu"
        record = public_export.PublicExportError("private-path", "blocked", private_path).record()
        self.assertNotIn("path", record)
        self.assertNotIn(private_path, json.dumps(record))
        self.assertEqual(record["path_sha256"], hashlib.sha256(private_path.encode()).hexdigest())

        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "json-target"
        output = io.StringIO()
        errors = io.StringIO()
        sentinel = "C:" + chr(92) + "Users" + chr(92) + "private-sentinel"
        with mock.patch.object(
            public_export, "_write_stage", side_effect=PermissionError(sentinel)
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            exit_code = public_export.main(
                [
                    "--source", str(repository.path),
                    "--tree", "HEAD",
                    "--allowlist", repository.allowlist_path,
                    "--apply", "--dest", str(destination),
                    "--json",
                ]
            )
        report = json.loads(output.getvalue())
        combined = output.getvalue() + errors.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["errors"][0]["code"], "filesystem-error")
        self.assertNotIn(sentinel, combined)
        self.assertNotIn("Traceback", combined)

    def test_detached_artifact_verifier_rechecks_exact_tree(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "artifact"
        plan = repository.plan()
        public_export.apply_plan(plan, destination)
        report_path = self.root / "report.json"
        report_path.write_text(
            json.dumps(plan.report("apply", destination, True)), encoding="ascii"
        )
        result = verify_public_export.verify(destination, report_path)
        self.assertTrue(result["ok"])
        (destination / "safe.txt").write_bytes(b"tampered\n")
        with self.assertRaises(public_export.PublicExportError) as raised:
            verify_public_export.verify(destination, report_path)
        self.assertEqual(raised.exception.code, "destination-dirty")

    def test_detached_verifier_rejects_git_metadata_and_allowlist_drift(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        destination = self.root / "exact-artifact"
        plan = repository.plan()
        public_export.apply_plan(plan, destination)
        report_path = self.root / "exact-report.json"
        report = plan.report("apply", destination, True)
        report_path.write_text(json.dumps(report), encoding="ascii")

        (destination / ".git").mkdir()
        (destination / ".git" / "config").write_text("private\n", encoding="utf-8")
        with self.assertRaises(public_export.PublicExportError) as raised:
            verify_public_export.verify(destination, report_path)
        self.assertIn(raised.exception.code, {"destination-dirty", "private-path"})
        shutil.rmtree(destination / ".git")

        changed = ("extra.txt\n" + repository.allowlist_path + "\nsafe.txt\n").encode("ascii")
        (destination / repository.allowlist_path).write_bytes(changed)
        (destination / "extra.txt").write_bytes(b"extra\n")
        allowlist_entry = next(
            item for item in report["files"] if item["path"] == repository.allowlist_path
        )
        allowlist_entry["bytes"] = len(changed)
        allowlist_entry["sha256"] = hashlib.sha256(changed).hexdigest()
        framed = f"blob {len(changed)}\0".encode("ascii") + changed
        allowlist_entry["blob_oid"] = hashlib.sha1(framed).hexdigest()
        report["allowlist_sha256"] = allowlist_entry["sha256"]
        report["allowlist_blob_oid"] = allowlist_entry["blob_oid"]
        report["total_bytes"] = sum(item["bytes"] for item in report["files"])
        report_path.write_text(json.dumps(report), encoding="ascii")
        with self.assertRaises(public_export.PublicExportError) as raised:
            verify_public_export.verify(destination, report_path)
        self.assertIn(raised.exception.code, {"artifact-report", "destination-dirty"})

    def test_apply_can_adopt_only_a_clean_exact_git_worktree(self) -> None:
        source = self.repo("source")
        source.commit({"safe.txt": b"new public\n"})
        destination_repository = self.repo("destination")
        destination_repository.commit({"old.txt": b"old public\n"})
        public_export.apply_plan(source.plan(), destination_repository.path)
        self.assertTrue((destination_repository.path / ".git").is_dir())
        self.assertFalse((destination_repository.path / "old.txt").exists())
        self.assertEqual((destination_repository.path / "safe.txt").read_bytes(), b"new public\n")

    def test_apply_rejects_source_containment(self) -> None:
        repository = self.repo()
        repository.commit({"safe.txt": b"safe\n"})
        with self.assertRaises(public_export.PublicExportError) as raised:
            public_export.apply_plan(repository.plan(), repository.path / "public")
        self.assertEqual(raised.exception.code, "destination-scope")


if __name__ == "__main__":
    unittest.main()
