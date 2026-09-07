# PR-F / Step 4 evidence — no-PyTorch Triton AOT builder

Historical status (**pre-R2**): local implementation and host-side release-boundary
gates were present; the release-ready AOT compile/hardware gates remained open. No
push, merge, tag, release, or performance claim. Branch: `triton`. Parent boundary:
local PR-E commit `e53b18a`.

This file preserves builder/productization facts from the superseded pre-R2 plan. It
is not current execution authority. In particular, the `215`-file allowlist/export
snapshot and the open production AOT/CI release lanes below are historical evidence,
not current-tree validation or an authorized work queue. The R2 Gate A No-Go
intentionally stops production AOT and CI release work.

## Public builder boundary

- `tools/triton-pack` is development/CI-only Python code. Cargo, `build.rs`, the
  product binaries, runtime, and tuner do not invoke it.
- The adapter imports only `triton` and `triton.language`; PyTorch is neither required
  nor permitted. It uses the compiler API rather than `triton.tools.compile`.
- Inputs are a reviewed exact-SM target (`cuda:80:32` or `cuda:86:32`), a literal AOT
  descriptor, and a strict toolchain lock. Pack source cannot supply target/toolchain
  facts or generated identities.
- The builder rejects nonzero Triton global/profile scratch, unknown targets/imports,
  environment drift, debug/IR-bearing cubin, non-content-addressed inventory, and
  output replacement.
- Final output is data-only: `manifest.json`, `manifest.sig`, declared
  `modules/<sha256>.cubin`, `SBOM.spdx.json`, `THIRD_PARTY_NOTICES`, and
  `provenance.json`. No host stub, Python, PTX, TTIR, TTGIR, LLVM IR, or signing-message
  file is admitted.
- Ed25519 signing is a separate command. Its canonical envelope order is frozen to the
  Rust trust boundary: `schema`, `domain`, `algorithm`, `key_id`, `manifest_bytes`,
  `manifest_sha256`, `signature`.

## Toolchain availability and fail-closed CI

The checked lock names Triton `release/3.8.x` source commit
`b252c7c4cea49216b27e5a47db061f9d0dbf49f3` and its archive SHA-256. At the audit date,
an official `v3.8.0` tag/wheel was not available. The lock therefore truthfully says
`source_pin_pre_release`, `official_tag` is null, and the builder-image digest is null.
No nonexistent wheel or invented image digest is used.

CI always runs builder golden, smoke-source, release-inventory, public-export, Cargo
dependency, and product-binary scans. Separate SM80 and SM86 AOT compile jobs are
defined, but run only after a separately reviewed lock changes to `release_ready` and
pins the builder image by digest. On an ordinary PR the green contract job is named and
summarized as contract-only; skipped compile jobs are explicitly reported as OPEN and
cannot be represented as Step 4 compile evidence. A manual workflow dispatch with
`require_triton_release=true` is the formal fail-closed gate and fails while the lock is
not release-ready. Each enabled lane performs two clean builds, signs both with a CI-only
ephemeral key, verifies/scans exact bytes, and compares complete trees.
The SM86 lane then uploads only the signed data-only pack, raw public key, and key id
(never the private key) to a dedicated self-hosted Windows worker labelled
`imparo-cuda-sm86` and `imparo-no-python`. That formal worker rejects any visible Python
command, requires the artifact through `IMPARO_REQUIRE_TRITON_SMOKE=1`, admits the exact
Python/OpenSSL bytes through Rust `Installer`, checks the live device is exactly SM86,
and launches the Linux-built cubin. Missing artifact or metadata, a noncanonical key,
wrong SM, missing CUDA/Rust tooling, or an unavailable reviewed worker cannot become a
successful formal runtime result. Ordinary contract-only CI still skips this
release-only lane while the lock is pre-release.
The external-artifact smoke additionally requires
`IMPARO_TRITON_EXPECT_FEASIBILITY=0|1`: the formal lane sets `0`, while a separately
invoked source-pin infrastructure probe must set `1`. Missing or inconsistent state
fails; feasibility is never inferred from the artifact or silently accepted.
Hosted compile results remain compile-only and never count as hardware correctness or
performance evidence.

## Local host-side evidence

```text
python -B -m unittest discover -s tools/triton-pack/tests -v
exit 0; 17 passed, 0 failed

python -B -m unittest dev_harness.test_program_pack_release \
  dev_harness.test_triton_pack_smoke -v
exit 0; 15 passed, 0 failed, 1 skipped
skip: external clean-build artifacts were not supplied

IMPARO_CUDA_PROGRAM_SMOKE=1 IMPARO_CUDA_ARCHS=86 \
  IMPARO_CUDA_SKIP_NVCC=1 cargo check --locked --tests \
  -p imparo-cuda --features cuda-static
exit 0; forced external-artifact gate compiled on Rust 1.85 without a CUDA build

python -B -m unittest \
  dev_harness.test_public_export.PublicAllowlistTests.\
test_production_allowlist_is_exact_ordinal_and_auditable -v
exit 0; 1 passed; reviewed allowlist = 215 LF paths,
sha256=d39825f86ab34bdf737ea8f369fafaca6fa07ce1939423cdd1f9e8f0347086fc

python -B dev_harness/public_export.py --source . --tree <temporary-tree> \
  --allowlist dev_harness/public-export.allowlist --apply --dest <temporary-root> --json
python -B dev_harness/verify_public_export.py \
  --root <temporary-root> --report <temporary-report>
exit 0; 215-file detached temporary tree verified;
private-reference/secret/path/binary scans passed

python -B dev_harness/verify_program_pack_release.py cargo --repo .
exit 0; 55 unique normal/build Cargo packages scanned; no Python/Torch/Triton runtime dependency

python -B dev_harness/verify_program_pack_release.py binary \
  target/release/imparo-server.exe target/release/imparo-tune.exe
exit 0; both existing Windows release binaries passed the conservative dependency/process-marker scan
```

The `215`-file detached tree, allowlist hash, and existing-binary scan above describe
that pre-R2 snapshot only. They do not claim a current fixed-tree export or current
release-binary authority.

A later independent audit shell did not expose `cargo` on `PATH`; repeating the Cargo
scan there returned exit 1 / `WinError 2`. That environment-limited rerun is not counted
as another dependency pass and does not weaken the CI requirement to run the same scan
in its provisioned Rust environment.

The release scanner's nine tests include missing signature, extra source/IR/host code,
wrong manifest/lane or binary-ELF SM, module/sidecar hash drift,
malformed/noncanonical signature, debug section,
absolute source path, weak SBOM license metadata, non-substantive NOTICE, forbidden
Cargo package, and forbidden runtime import negatives. One cross-module test feeds
actual `build_pack(FakeCompiler)` output into the independent scanner and proves that
the builder's `source_pin_pre_release` provenance is rejected by default and accepted
only with the explicit feasibility-smoke opt-in. The scanner implementation remains
independent of the builder verifier so a shared implementation error cannot make both
gates pass.

## Historical open evidence (not claimed; stopped by R2)

The following items were open under the pre-R2 productization plan. They remain useful
historical gaps, but the R2 Gate A No-Go means they must not now be executed as PR-F
production work. Reopening production AOT/CI release work requires a future reviewed
Gate A Go and the subsequent R2 gates; none is implied here.

- The pinned builder image must be constructed, independently reviewed, published, and
  written into `toolchain.lock` by immutable digest before either AOT compile lane runs.
- Two real clean Linux AOT builds for SM80 and SM86 have not run under that final image;
  no reproducible cubin identity is claimed yet.
- No SM80 hardware run exists. SM80 remains compile-only even after its CI lane runs.
- The formal cross-platform CI dependency and fail-closed Windows/SM86 worker contract
  are implemented, but that worker has not yet loaded a Linux-CI-generated final signed
  SM86 pack on RTX 3060. Existing PR-E native SM86 bridge evidence does not substitute
  for an actual green PR-F runtime result.
- A machine having Python installed is not proof of the required clean Windows
  no-Python runtime. That test remains a separately provisioned environment gate.

The smoke source proves infrastructure only. It is not a Community production kernel
and carries no performance result.

## Rollback

Revert PR-F as one unit: `tools/triton-pack`, `program-packs/community/cuda` smoke
source/contracts, the PR-F CI lanes/scans, and this evidence. PR-E's signed data-only
loader and built-in CUDA fallback remain valid without any AOT pack. Do not retain a
profile or receipt produced from a reverted pack catalog.
