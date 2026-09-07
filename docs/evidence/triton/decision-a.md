# Triton R2 Decision-A — SM86

Status: **Gate A No-Go**. This record is LAB evidence only and grants no
production authority.

The finalized RMS candidate is admissible, but R2 does not permit a single
RMS micro win to authorize Gate A. The mandatory high-share Q4 family passed
correctness, four-tool sanitizer (16/16), formal-measurement noise, resource,
and provenance checks. Every tested result failed the `1.10x` kernel floor;
V4 was the best pair, but its required pair score is the slower-shape minimum,
`min(0.341972x, 0.390734x) = 0.341972x` native/Triton.

Evidence bindings:

- RMS finalized decision: `7b7c5e0dd4538f66f135b9ac708ebc838aa49eec813f43ec43ebc974b1decca4`.
- Q4 finalized candidate: `1abe7f4bbb87f77b4563c4bad9b424cf3b00840f81d27cff527f16189ec2b4ef`.
  The Q4 finalizer's native process exit was `3`, its designed fail-closed
  non-admissible-candidate outcome; the JSON above was still emitted and its
  SHA-256 was verified independently.
- Q4 formal matrix: `76b5d7a6b485d53c762ebface6996b39cbad04678a4ee2b7b5e292891c3a54af`.
- Q4 sanitizer schema-2 manifest: `39b036b8e2b3b267d613d5c3ae44bdfa3d7e42f39a27faf5c7db568500443451`.
- Q4 current environment witness: `ca1bc89778ed87bd82356a9065e99471c8472a25fc3a366c3a91ec43e27753f4`.
- Policy: `b6572d7f48f695022eaf7b2bcba3881df938ce1f0457273e8b4800743f9f22ca`.
- Derived Q4 JSON: `artifacts/kernel-lab/q4-variants-perf/variants-b/gate-a-q4-decision.json`.
- Final Decision-A JSON: `artifacts/kernel-lab/decision-a/gate-a-decision.json`,
  SHA-256 `961c33e3737162db89b9d87bc17ea943746dd88900ccd478c86d3efd13ae9cf7`.

These are hash-bound evidence records. The hardened validator also replayed
the three bound tool version commands and all eight bound `cuobjdump` commands
on the local verification host, then compared exit/stdout/stderr byte-for-byte
with the manifest. This is replayed local validation, not a claim that hashes
alone cryptographically prove the historical process that produced the files.

## Final current-tree validation

The final current-tree validation pass recorded all of the following without
changing the Decision-A inputs or conclusions:

- From the VS 2022 `vcvars64` environment with `IMPARO_CUDA_ARCHS=86`,
  `python dev_harness/build.py --locked -p imparo-model -p imparo-server
  -p imparo-tune --bins --features cuda` exited `0` and
  reported `FRESH: all engine binaries at/after newest source`.
- Full `dev_harness` discovery exited `0`: `219` passed with `2` conditional
  skips. The Triton kernel-lab suite exited `0`: `16/16` passed. The
  `tools/triton-pack` suite exited `0`: `17/17` passed.
- `cargo test --workspace --locked`, `cargo fmt --all -- --check`, and
  `cargo clippy --workspace --release --locked --all-targets -- -D warnings`
  each exited `0`.
- The Cargo runtime-dependency scan exited `0` after checking `55` unique
  normal/build packages and found no Python, PyTorch, or Triton runtime
  dependency.
- An independent read-only audit of the schema-2 sanitizer manifest, finalized
  Q4 record, and Decision-A bindings was clean. Repository LF and
  `git diff --check` checks were also clean.

This is current-tree validation only. The detached fixed-tree public export is
the final sealing step outside this self-referential record; its exact tree OID
and result must be reported with the local commit/PR.

Phase A1 percentages divide measured logical-matmat time, not whole-prefill
time. The 3% whole-prefill projection is therefore not evaluable and is not
claimed. This missing denominator cannot override the mandatory Q4 floor
failure. R2 requires stopping PR-G and Phase B/C/D; the dormant
`identity_ready=false` parser/trust/ABI26/bridge surface stays non-production.
