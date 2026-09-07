# Phase A2 LAB-A bounded Q4_0 x Q8_1 MMQ variants — Windows SM86

Status: **Phase A2 complete; mandatory Gate A performance floor failed**.
All four configurations compile without hidden scratch or local/stack spills,
pass C0 and the full 16/16 four-tool sanitizer matrix, and have formal
20/50/16 interleaved timing evidence on the same RTX 3060.  Correctness,
sanitizer, noise, resource, and provenance gates pass, but all eight shape
results are slower than native.  Under R2 this is a fail-closed Gate A No-Go,
not a production Program Pack, winner, receipt, ABI authority, or installer.

## Scope and bounded rationale

The R2 validation-first handoff was read at SHA-256
`80691e9e3f044ed5173dd1ff329bd6afdb0da78842c0025363e243d1b3625b55`.
Its Phase A2 rule permits only a small, evidence-based matrix before Gate A.
The existing `BM64 x BN64`, four-warp, two-stage baseline and its `run-a` and
`run-b` artifacts were not modified.

The formal baseline was stable but slower than native: expansion was
`998.7520142 us` native versus `2482.816162 us` Triton (`0.4022657937x`), and
contraction was `1184.384033 us` versus `2499.387939 us` (`0.473869628x`).
The baseline used 127 registers/thread, 16 KiB dynamic shared memory, and no
local/hidden scratch.  The bounded matrix tests two concrete explanations:

- `BM128 x BN64`, w8s2 halves row CTA count and repeated Q8 reads;
- `BM64 x BN128`, w8s2 halves token CTA count and repeated Q4 reads;
- `BM64 x BN64`, w8s2 isolates warp/register pressure at fixed geometry;
- `BM64 x BN64`, w8s1 isolates stage/shared/register cost from the w8 tile.

No `BM128 x BN128` or Cartesian tile/warp/stage sweep was added.  The wider
tile was excluded because the kernel retains independent prefix and suffix
accumulators and therefore has a materially greater spill/occupancy risk.

All four variants use the same implementation body.  The native numerical
tile is derived from element coordinates:

```text
native_row_tile   = (program_row   * BLOCK_M) / 128
native_token_tile = (program_token * BLOCK_N) / 128
native_tile       = native_row_tile * native_ntx + native_token_tile
```

Every selected BM/BN divides 128 and cannot cross a native numerical tile.
K-block traversal, separate prefix/suffix accumulation, runtime
`numeric_stream_grid`, and the final `suffix + prefix` fold are unchanged.

## Source and builder identity

- Branch baseline: `triton@e53b18a9093f189ea67994b8d00ca7d7e463c7c2`.
- Builder: `tools/triton-kernel-lab/build_q4_q8_mmq_variants.py`, SHA-256
  `8e13cc2824b918e1f30d8a89fa2e9772bcf7dc33bc057bd92dba1bbf9acf2611`.
- Expansion source: `q4_q8_mmq_variants_2560x10240.py`, SHA-256
  `d698639499e98a40b5e14be6bca8a4e37a5b82a0b2b65e0f7914983e106c8d10`.
- Contraction source: `q4_q8_mmq_variants_10240x2560.py`, SHA-256
  `abc66c8182c4f0a25d8100f95984b669a7212b7036d31a139d080b8545591f56`.
- Builder image identity:
  `sha256:357ea84b8e099d8b1884bde98172feef40a1acd9dc91f6b256dd16245573a1cf`.
- Python `3.12.3`; Triton `3.8.0-source-pin`; torch absent.
- Target `cuda:86:32`; no other SM was compiled or claimed.

The two clean builds used the following command, changing only `<run>` to
`variants-a` and `variants-b`; both exited `0`:

```powershell
wsl.exe -e docker run --rm `
  -v /mnt/d/imparo:/workspace -w /workspace `
  --entrypoint python3 imparo-triton-builder:step4 -I `
  tools/triton-kernel-lab/build_q4_q8_mmq_variants.py `
  --out program-packs/lab/q4-q8-mmq-sm86/<run> `
  --builder-image-id sha256:357ea84b8e099d8b1884bde98172feef40a1acd9dc91f6b256dd16245573a1cf
```

Metadata SHA-256 values are
`d7de94670e4a36bcaaa1c4c4cb8b10c5505022be33e6ae6d5fac10380a5d270d`
for `variants-a` and
`f4dd69db8475ec881f5bf5062d1a7d4626605b658bd7727233af077eef3d2607`
for `variants-b`.  Metadata bytes intentionally differ because
`compile_seconds` is diagnostic.  The module hashes, byte sizes, symbols,
launch geometry, and resources are identical across the two clean builds.

## Modules and resource screen

The builder verified the real cubin ELF target as SM86, the exact 72-byte
KPARAM table as eight visible plus two mandatory zero-sized hidden pointer
arguments, and `REQNTID=(256,1,1)` from each opaque symbol's `.nv.info`
section.  It hard-fails non-zero hidden scratch and excludes any module with
non-zero `LOCAL+STACK`.  All eight modules survived.

| Shape | Config | Opaque symbol suffix | Cubin SHA-256 | Bytes | Registers | Dynamic shared | Local/hidden scratch |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2560 -> 10240 | BM128/BN64/w8/s2 | `cd1dd7ee...c7d5ed` | `bbf26d71916173b353d642dc09b8cf4a905f6d51d02dec5040bcf6611a116509` | 64,288 | 155 | 32,768 B | 0 / 0 |
| 2560 -> 10240 | BM64/BN128/w8/s2 | `040845b8...741bc0` | `8c566758cc9603a6899f3848494b20e353f9df5bb436da7ba4e69901d74649b9` | 60,320 | 178 | 32,768 B | 0 / 0 |
| 2560 -> 10240 | BM64/BN64/w8/s2 | `58bab621...d857c` | `715b630ae6a9c743df57f63ad733ea77a5d9498301b9f1c6224a0bf7dbf0b6d4` | 42,528 | 101 | 16,384 B | 0 / 0 |
| 2560 -> 10240 | BM64/BN64/w8/s1 | `2f16dc07...7198b` | `7006694d5fb04c95a701b251994b7dcf8aeee14a1b73702907f4ec1c70fd4cab` | 41,888 | 96 | 16,384 B | 0 / 0 |
| 10240 -> 2560 | BM128/BN64/w8/s2 | `5ee29234...c9b32` | `c439040b3b338a869bee453a5f855b8aa08b1b2edba4951e38daf6fa2782bb3f` | 64,032 | 155 | 32,768 B | 0 / 0 |
| 10240 -> 2560 | BM64/BN128/w8/s2 | `05e1dac7...280d4` | `05cf736717e36bdc2b7fcf8f6de82fe637aed7c29a9236646b665fa771867c1c` | 60,320 | 178 | 32,768 B | 0 / 0 |
| 10240 -> 2560 | BM64/BN64/w8/s2 | `99be389d...ed6eb` | `aec71430c575a4eada8f18f567644f695a34fdc036ca3746bb852092bc74c87a` | 42,400 | 101 | 16,384 B | 0 / 0 |
| 10240 -> 2560 | BM64/BN64/w8/s1 | `6b4e87dd...c264d` | `87ab7b48453dc0eb0ebd105d3461b57b1937b34ed131384bfd86730dce688225` | 41,760 | 96 | 16,384 B | 0 / 0 |

The exact full symbols, opaque `config_id`, module paths, grid formulas, and
launch fields are in
`program-packs/lab/q4-q8-mmq-sm86/variants-b/lab-metadata.json`.

## Software gates

```powershell
python -m unittest discover -s tools/triton-kernel-lab/tests -v
```

Exited `0`: 16 tests passed.  This includes the original RMS/Q4 builder
contracts and six new matrix tests for bounded identities, source seam rules,
variant-specific REQNTID, eight-module metadata, spill rejection, and hidden
scratch fail-closed behavior.

```powershell
python -m unittest discover -s tools/triton-pack/tests -v
```

Exited `0`: 17 tests passed.  `git diff --check` also exited `0`.

## Windows C0 harness contract

The hardened standalone harness source SHA-256 is
`ed7860950f657d7e0affc37ed46729742fd5a34a4d572a0670d2cafaa56d0137`.
It was compiled for only SM86 with CUDA 12.9 `nvcc`, optimization `-O3`,
C++17, fast math, line info, shared CUDA runtime, Backend ABI 26, and an
all-zero non-production build hash.  Compilation exited `0`; the executable
SHA-256 is
`44c5e0ad8686352e52e37003740ff11572c402d9808e14cd1da5663c3ce56d7b`.

Each paired configuration was invoked with the argc16 contract:

```text
kernel_lab_q4_q8_sm86.exe
  <expansion.cubin> <expansion.symbol>
  <contraction.cubin> <contraction.symbol>
  1 1 1
  <expansion BM> <expansion BN> 256 <expansion dynamic shared>
  <contraction BM> <contraction BN> 256 <contraction dynamic shared>
```

The `1/1/1` values request only a minimal warmup/pair/launch structural pass.
They are deliberately not the formal 20/50/16 performance contract.  For
each V1 through V4 invocation the process exited `0`, emitted one JSON line,
reported `formal_contract=false`, and passed the versioned policy's complete
`Get-Q4ResultViolations` correctness check with zero violations.

## C0 hardware and results

- Device: NVIDIA GeForce RTX 3060 Laptop GPU, SM86, 30 SMs.
- UUID: `a69e87d6c66ade6aea365d1e861aba1d`.
- Driver API version: `13020`; CUDA runtime: `12090`.
- Same primary context, stream, inputs, shapes, and native controls were used.
- Required token cases per configuration: `1,127,128,129,511,512` for both
  expansion and contraction, 12 cases total.

| Config | Result SHA-256 | Exit | Cases | Structural | Policy violations |
| --- | --- | ---: | ---: | --- | ---: |
| V1 BM128/BN64/w8/s2 | `c41ef93f0ed362f8895589c8bbcc9e6ceb30b149bdca2f4f24d0f2039719e48f` | 0 | 12 | pass | 0 |
| V2 BM64/BN128/w8/s2 | `b70bd67cdf4dbf492cf1807bcac6867049eae81977d8ac188e31ad2dedf484c1` | 0 | 12 | pass | 0 |
| V3 BM64/BN64/w8/s2 | `3734a4a84419617a38a9f962288682147d50f0046a58bbc50a5acd809b1e77ca` | 0 | 12 | pass | 0 |
| V4 BM64/BN64/w8/s1 | `bddc330eebd9bc7f685fd9a2ebc299f8a08c95c48904d34dda38cdcf87aa69da` | 0 | 12 | pass | 0 |

Across every configuration and all 12 cases, the worst observed values were:

- max absolute error versus native: `4.196166992e-05`;
- max RMS error versus native: `6.720676658e-06`;
- max normalized relative error: `2.39442401e-05`;
- non-finite, input mismatch, output-padding error, canary error, and
  post-timing canary error counts: all zero.

At `n_tok=512`, expansion had max absolute error `1.525878906e-05` and
contraction `2.956390381e-05`; post-timing comparison reproduced the same
maxima.  Native matched the strict float oracle bit-for-bit.  Triton strict
oracle max absolute error was `3.814697266e-06` for expansion and
`7.033348083e-06` for contraction.  Triton-versus-f64 max absolute error was
`1.205934677e-05` and `1.630699262e-05`, respectively.

For every configuration the contraction oracle covered two seam-104 samples,
two seam-208 samples, four no-seam samples, directed seam tiles `2,5`, adjacent
no-seam tiles `1,3,4,6`, and boundaries `103,104,207,208`.  The CPU wrong-grid
mutation changed four directed values; seam-minus-one changed two and
seam-plus-one changed one.  The GPU wrong-grid mutation changed 290,144 output
elements versus the correct Triton output, remained finite, and caused zero
input, padding, or canary errors.  Expansion correctly used four no-seam
oracle samples and did not launch the wrong-grid mutation.

Raw C0 results and the executable are under
`artifacts/kernel-lab/q4-q8-mmq-sm86/variants-c0/`.

## Completed sanitizer, formal timing, and Decision-A boundary

The four configurations each passed memcheck, initcheck, racecheck, and
synccheck: 16/16 runs plus 8 preflights. The final schema-2 sanitizer manifest
is `artifacts/kernel-lab/q4-variants-sanitizer/variants-c/sanitizer-evidence-v2.json`,
SHA-256 `39b036b8e2b3b267d613d5c3ae44bdfa3d7e42f39a27faf5c7db568500443451`.
It binds runner SHA-256
`03897281cc62829b8dac32f57906ad3b94c6dc604a79fc721fa744d54ea7c670`,
exact tool identities, exact argv, raw and normalized streams, exit files,
module preflight, and GPU snapshots. The hardened validator replayed the three
bound version commands and all eight bound `cuobjdump` commands locally and
compared their exit/stdout/stderr exactly. This is hash-bound plus replayed
local evidence, not a claim that hashes cryptographically prove execution
origin.
The formal matrix used 20 warmups, 50 ABBA/BAAB pairs, and 16 launches/sample;
its SHA-256 is
`76b5d7a6b485d53c762ebface6996b39cbad04678a4ee2b7b5e292891c3a54af`.

| Variant | Expansion native/Triton us | Expansion speedup | Contraction native/Triton us | Contraction speedup |
| --- | ---: | ---: | ---: | ---: |
| V1 BM128/BN64/w8/s2 | 798.400 / 2851.639 | 0.279979x | 966.432 / 2987.520 | 0.323490x |
| V2 BM64/BN128/w8/s2 | 840.954 / 2608.096 | 0.322440x | 1001.952 / 2685.470 | 0.373101x |
| V3 BM64/BN64/w8/s2 | 955.711 / 2843.232 | 0.336135x | 1112.256 / 2913.788 | 0.381722x |
| V4 BM64/BN64/w8/s1 | 936.576 / 2738.750 | 0.341972x | 1125.277 / 2879.904 | **0.390734x** |

V4 is the best pair. Its single-shape maximum is `0.390734x`, but R2 scores a
pair as `min(expansion, contraction)`, so the controlling best-pair score is
`min(0.341972x, 0.390734x) = 0.341972x`, below the mandatory `1.10x` kernel
floor. The finalized Q4 candidate SHA-256 is
`1abe7f4bbb87f77b4563c4bad9b424cf3b00840f81d27cff527f16189ec2b4ef`;
the final Decision-A No-Go SHA-256 is
`961c33e3737162db89b9d87bc17ea943746dd88900ccd478c86d3efd13ae9cf7`.
The whole-prefill denominator remains absent, so the 3% projected
whole-prefill floor is not evaluable; this cannot override a mandatory Q4
kernel-floor failure.  R2 therefore requires Gate A No-Go and stops Phase B,
Phase C, Phase D, PR-G, production identity/receipt, installer, Commercial,
Graph integration, and cross-SM expansion.  All variants remain LAB-only.
