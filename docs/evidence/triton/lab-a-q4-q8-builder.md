# Phase A2 LAB-A Q4_0 x Q8_1 MMQ builder evidence

Status: **builder/codegen evidence only**.  RTX 3060 native-control
correctness, canary, sanitizer and timing gates are pending.  This is not a
production Program Pack, ABI, receipt, installer, trust decision, winner, or
performance claim.

## Bounded candidate and native control

- Starting branch/HEAD: `triton@e53b18a9093f189ea67994b8d00ca7d7e463c7c2`.
- Candidate: plain Q4_0 x Q8_1 MMQ, epilogue 0, initially one justified
  `BM64 x BN64`, four-warp, two-stage variant.
- Exact shapes: expansion `K=2560, M=10240`; contraction
  `K=10240, M=2560`; evidence workload `N=512`.
- Target: exact `cuda:86:32`; no other SM was compiled locally.
- Expansion native control:
  `imparo_sm80_mmq::q4_q8_1_full_tile<false,4>`, without NumericSeams.
- Contraction native control:
  `imparo_sm80_mmq::q4_q8_1_full_tile<false,4,false,true,true>`, with
  NumericSeams.  The lab source uses the runtime `numeric_stream_grid` and
  the native Stream-K boundary formula; it does not hard-code the RTX 3060's
  30-SM seam table.  At `N=512`, the two audited native seam examples are
  block 208 for row-tile-even/token-tile 2 and block 104 for
  row-tile-odd/token-tile 1.  Prefix and suffix are accumulated separately
  and folded `suffix + prefix`.

Native-launch audit caught a fail-closed route trap before GPU testing.  With
contraction efficiency 88, `full_tile_min_efficiency=90` does **not** select
that NumericSeams kernel; it selects PhysicalStreamK.  The exact control must
leave the tuned threshold at zero so SM86 architecture policy selects its
default 50 (or explicitly use a threshold no greater than 88).  GPU evidence
must assert the returned route and must not relabel PhysicalStreamK as the
requested control.  Expansion efficiency is 96 and legitimately selects the
non-seam full-tile route at threshold 90.

The implementation uses `tl.range`, not Python-unrolled loops.  `n_tok` is a
runtime value in `[1,512]`; Q8 loads, scale loads and output stores are token
masked, and native token-tile count is `ceil_div(n_tok,128)`.

## Frozen data and launch contract

Q4_0 weights use an 18-byte row-major record: fp16 scale at byte 0 and 16
packed bytes at byte 2.  The low nibble represents `k=0..15`, the high nibble
`k=16..31`, and dequantization is `(nibble-8)*d4`.

The activation is the existing 144-byte `BlockQ8_1Mmq`: `int8 qs[128]` at
byte 0 and four fp32 scales at byte 128, with record
`(block32/4)*n_tok+token`.  This candidate consumes that allocation; it does
not implement or replace the Q8 producer.  Output indexing is
`dst[token*out_stride+row]`.

Both shape-specialized modules have visible argument ordinals 0 through 7:

```text
w_qs *i8        = weights + 2 bytes (guaranteed alignment 2)
w_d *fp16       = weights base
x_qs *i8        = Q8 allocation base
x_d *fp32       = Q8 allocation base + 128 bytes
y *fp32
n_tok i32
out_stride i32
numeric_stream_grid i32
```

Triton appends zero-sized `global_scratch` and `profile_scratch` device
pointer arguments at ordinals 8 and 9.  Both arguments remain mandatory at
Driver launch even though their allocation size is zero.  The builder parses
the real cuobjdump `.nv.info.<symbol>` section and rejects anything other
than the frozen ten-parameter layout, 0x48-byte parameter bank, and required
block `(128,1,1)`.

At `N=512`, grids are `(160,8,1)` for expansion and `(40,8,1)` for
contraction.  In general the grid is
`(n_out/64,ceil_div(n_tok,64),1)`.  Both modules require 16,384 bytes dynamic
shared memory.

## Toolchain identity

- Triton source commit:
  `b252c7c4cea49216b27e5a47db061f9d0dbf49f3` (`3.8.0` source tree;
  source-pin/pre-release feasibility only).
- Python `3.12.3`; Triton `3.8.0`; torch absent.
- ptxas `V12.9.86`; cuobjdump `V12.9.82`.
- Builder semantic recipe SHA-256:
  `5165e28709da883bd44423c6f980726ed5413b89094f3efa76d3c617cbe4efe7`.
- Local Docker image-store ID:
  `sha256:357ea84b8e099d8b1884bde98172feef40a1acd9dc91f6b256dd16245573a1cf`.
- Linux/amd64 manifest:
  `sha256:cc77b73970499241bf4e5c9f70a1f4f9571994a7a992937a3dc6827241f0ab76`.
- OCI config digest:
  `sha256:91239d1dae39c4006010a7ca5f2d0c90f342a6c9327053553ee3b1b36c6a4e54`.

The local image-store/config identities are not pullable release-manifest
identities and were not written to the checked-in release lock.

## Clean SM86 builds

The following command was run twice into fresh `run-a` and `run-b`
directories.  Both invocations exited `0`; the second also enforced the real
cubin parameter-table gate:

```powershell
wsl.exe -e docker run --rm `
  -v /mnt/d/imparo:/workspace -w /workspace `
  --entrypoint python3 imparo-triton-builder:step4 -I `
  tools/triton-kernel-lab/build_q4_q8_mmq.py `
  --out program-packs/lab/q4-q8-mmq-sm86/<run> `
  --builder-image-id sha256:357ea84b8e099d8b1884bde98172feef40a1acd9dc91f6b256dd16245573a1cf
```

| Shape | Opaque symbol suffix | Registers | Dynamic shared | Local/hidden scratch | Bytes | SHA-256 | Compile seconds run-a / run-b |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2560 -> 10240 | `a1da19c6...09fda6` | 127 | 16,384 B | 0 / 0 | 63,840 | `3c727c0b7de714334b67e0e09ef4d6ac044f84de1d5a02d34e2e8d6478813d75` | 1.583 / 1.579 |
| 10240 -> 2560 | `951d2021...039fc6` | 127 | 16,384 B | 0 / 0 | 63,840 | `ddf82b19bc20d7f54f0f4da1e1b43c456d47c6fd22afaf7293d17c5549e286be` | 0.255 / 0.266 |

Source identities used by both builds:

- expansion source:
  `83ec7ea4fcbba817567c4ddae9b9e65a6d99fa23e2e9b59b28aa0398d9a16f8f`
- contraction source:
  `f795a3dbd9dcc61996c2da1f1621f9feed9fc6a68c01a9d0df1dd36485965a1d`
- lab builder:
  `70622cb4a28d69270582f8db052c6ca345afb947ddd2776df5d798924990ada3`

Artifacts and complete metadata:

- `program-packs/lab/q4-q8-mmq-sm86/run-a/lab-metadata.json`
- `program-packs/lab/q4-q8-mmq-sm86/run-b/lab-metadata.json`

## Software gates and pending work

`python -m unittest discover -s tools/triton-kernel-lab/tests -v` exited `0`
with 10 tests passed.  The suite freezes exact shapes and the single initial
variant, aliases, runtime tail masking, native seam goldens, exact-SM ELF,
visible-eight-plus-hidden-two ABI, and fail-closed non-zero scratch handling.

Still pending before this candidate can be called admissible:

- Run both modules on RTX 3060 through the existing CUDA Driver bridge in the
  same primary context and exact engine stream.
- Compare every output against the exact native control for both shapes,
  including numerical-seam and tail workloads; reject non-finite values,
  canary corruption, or tolerance failure.
- Run the required sanitizer set and interleaved native/Triton timing policy.
- Do not add the proposed 64/128/128 variants until this initial C0 candidate
  passes correctness.  Do not connect it to production runtime or packaging.
