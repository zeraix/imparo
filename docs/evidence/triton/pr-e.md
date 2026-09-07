# PR-E / Step 3 evidence — CUDA Program ABI bridge

Status: local implementation and Step-3 gates complete; no push, merge, tag, release, or performance
claim. Branch: `triton`. Parent boundary: local PR-D commit `ceae579`.

## Scope frozen by this PR

- CUDA backend ABI 26 and Program ABI v1.
- Exactly six generic DLL/SO symbols: install, catalog identity, bind, freeze, launch,
  and reset. There are no pack-specific or kernel-specific exports.
- Signed/admitted data-only cubin bytes flow through the existing CUDA backend's
  selected device, retained primary context, stream, error state, and Graph owner.
- Eligibility fails closed on exact SM, driver minimum, backend ABI interval, math mode,
  compiled contract/adapter/workload fixture, slots/effects/aliasing, bounded scratch,
  actual Driver resource queries, and Graph policy.
- Every admitted choice group exposes the existing built-in native implementation as
  its safe fallback candidate.
- PR-G identity encodings remain deliberately inactive: `identity_ready=false` and the
  pack-set/catalog/eligible hashes are zero. No ABI-25 receipt authorizes ABI-26 code.

## Static and host-side evidence

The checked public contract test is
`crates/imparo-cuda/tests/program_pack_contract.rs`. It freezes the ABI number, six
exports, Rust/C wire sizes and key offsets, unity-build ownership, Driver-only module
path, shared primary-context/stream markers, reset boundary, inactive PR-G identity,
and absence of Python/Triton/torch host dependencies in the native bridge.

The reviewed export also includes `native/program_pack_test.cu` and
`src/program_pack/sm86_smoke.rs`. Their results are hardware evidence only when the
SM86 smoke command actually runs; merely exporting or compiling them is not a pass.

The detached-public CI materializes only the reviewed allowlist, verifies its exact
inventory, then checks the complete workspace plus both CUDA paths independently:

```text
cargo check --locked --all-targets -p imparo-cuda --features cuda-dynamic
IMPARO_CUDA_SKIP_NVCC=1 cargo check --locked --all-targets -p imparo-cuda --features cuda-static
```

Local host-only verification snapshots:

```text
cargo test --locked -p imparo-cuda --test program_pack_contract --no-default-features -- --nocapture
exit 0; 5 passed, 0 failed

Rust 1.85 validation:
cargo test --locked -p imparo-program-pack
exit 0; 25 passed, 0 failed

cargo test --locked -p imparo-host correctness -- --nocapture
exit 0; 11 passed, 0 failed

cargo test --locked -p imparo-cuda --lib --features cuda-dynamic
exit 0; 44 passed, 0 failed

cargo check --locked --all-targets -p imparo-cuda --features cuda-dynamic
exit 0

IMPARO_CUDA_SKIP_NVCC=1 cargo check --locked --all-targets -p imparo-cuda --features cuda-static
exit 0

cargo test --workspace --locked
exit 0

cargo clippy --workspace --release --locked --all-targets -- -D warnings
exit 0

```

The first Rust 1.85 workspace attempt rejected one unstable let-chain left by the
earlier PR-C tuner guard. The equivalent nested form was applied and the full workspace
then passed. The first strict clippy pass also rejected only pre-existing ambiguous
local names and implicit bitwise precedence in CPU/tokenizer/tuner/server code; those
were changed without altering behavior, the affected package tests passed, and the
strict workspace clippy command above then exited 0.

The final pre-commit detached-public snapshot passed 26 tests with one Windows
executable-bit skip and scanned 189 files. Its reviewed allowlist SHA-256 is
`53654d94b487861d8358661062da3461e1d1aceb16d9fc23a031d88c7d798cc8`; binary,
path, private-reference, and secret scans all passed. Hardware-only results must not
be inferred from public-export static checks.

One intermediate detached-style dynamic check failed while the concurrent ABI edit was
incomplete:

```text
cargo check --locked --all-targets -p imparo-cuda --features cuda-dynamic
exit 101; E0063 at src/program_pack/loader.rs:107: newly added argument_schema missing
```

The later Rust 1.85 dynamic all-target check above passed after the loader/native wire
was synchronized. The intermediate failure is retained as audit history and is not
counted as a pass.

## Exact-SM86 hardware evidence

The final hardware command was run from a VS 2022 `vcvars64` environment:

```text
cmd.exe /d /v:on /c 'call "%VS2022_VCVARS64%" >nul && set "IMPARO_CUDA_ARCHS=86" && set "IMPARO_CUDA_PROGRAM_SMOKE=1" && set "PATH=%RUST_TOOLCHAIN_BIN%;!PATH!" && cargo test --locked -p imparo-cuda --features cuda-static program_pack_sm86_signed_lifecycle_and_graph_smoke -- --nocapture'
exit 0; 1 passed, 0 failed, 42 filtered out; test body 13.46 s; build 1m02s
```

Verification host: RTX 3060 Laptop GPU, exact SM 8.6, driver 596.08, CUDA compiler
12.9.86, MSVC 19.44.35228, and rustc/cargo 1.97.1. The test compiled a real SM86 cubin,
admitted a real Ed25519-signed data-only pack, and passed native query/freeze/bind,
launch/readback, and checked shutdown. It also passed:

- signed wrong-SM, wrong-ABI-interval, newer-driver, and resource-underclaim rejection,
  with built-in fallback still available after each rejection;
- corrupted cached-object rejection with the already frozen built-in fallback intact;
- 1000 complete install/freeze/bind/launch/readback/shutdown lifecycles (Graph off);
- capture-only Graph capture/replay, then route-change invalidation;
- replay-update-safe capture/replay with engine-owned source 1
  (`decode_start_pos_u32`) changing the observed value from `0x1111` to `0x2222`;
- shutdown while that Graph remained live, with checked Graph destruction before module
  unload and the final Graph-alive probe returning zero.

This is infrastructure correctness evidence for SM86 only. It is not a Program-kernel
performance result, and it does not claim validation on another SM.

A second real-NVCC SM86 run under the repository's Rust 1.85 toolchain also exited 0
with 1 passed, 0 failed, and 42 filtered out (test body 13.25 s; build 1m05s).
`VS2022_VCVARS64` and `RUST_TOOLCHAIN_BIN` denote the selected toolchain locations;
host-specific absolute paths are intentionally excluded from the public evidence.

## Rollback

Revert the PR-E commit as one unit. ABI-25 native CUDA remains the parent fallback
boundary; pack absence or rejection already selects built-in native code. Do not copy
ABI-26 Program selections into the ABI-25 receipt/config domain.
