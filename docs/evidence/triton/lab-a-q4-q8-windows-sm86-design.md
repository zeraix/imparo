# Phase A2 LAB-A Q4_0 x Q8_1 MMQ Windows SM86 design boundary

Status: **design only; launch ABI and artifacts are not frozen**. No Q4
candidate result or Gate A conclusion exists yet. This file must not be used
as production ABI, Program Pack, receipt, installer, or graph authority.

## Frozen workload facts

- Exact expansion native control:
  `imparo_sm80_mmq::q4_q8_1_full_tile<false, 4>`.
- Exact contraction native control:
  `imparo_sm80_mmq::q4_q8_1_full_tile<false, 4, false, true, true>`.
- Native block: `dim3(32, 8, 1)`; 256 threads.
- Native dynamic shared memory: `kHalfKSharedBytes`.
- Native epilogue pointer: null. Expansion is a no-seam route; its Triton
  runtime argument uses `native_tiles=320`. Contraction passes the runtime
  device SM count (30 for the RTX 3060 evidence).
- Shape A: `K=2560, M=10240, N=512`; native grid x is 320.
- Shape B: `K=10240, M=2560, N=512`; native grid x is 80.
- Q4_0 wire layout: one 18-byte record per 32 K values, containing an f16
  scale followed by 16 packed bytes. Low and high nibbles represent values
  `i` and `i+16`; dequantization is `(q-8)*d`.
- Q8_1 MMQ wire layout remains the 144-byte K-group-major record used by the
  first lab: 128 signed values and four fp32 scales, indexed by
  `(block32/4)*N+token`.
- Output is token-major:
  `dst[token*out_stride+row]`, with `row_base=0`.
- Initial Triton tile under evaluation: `BM=64`, `BN=64`, four warps,
  two stages; grid is
  `(ceil(n_out/64), ceil(n_tok/64), 1)`, block size is 128, and each shape
  has a separate exact-SM86 cubin.

## Required independent harness behavior

The Windows executable will remain a separate test-only translation unit
behind `IMPARO_CUDA_KERNEL_LAB=1`. It will include the existing native
translation unit so the control is the exact template above on the exact
primary context and `g.stream`.

For each shape it must:

1. Generate deterministic Q4 and Q8 wire inputs with explicit first/last
   boundary records and valid alignment.
2. Allocate separate guarded native and Triton outputs and verify both
   canaries.
3. Compare every Triton output element with native. For directed oracle
   points, compute the exact i32 dot per K block, multiply in the order
   `float(dot) * weight_scale * activation_scale`, and accumulate in f32.
   Also retain a blockwise f64 reference without changing the production
   kernel's own reduction order.
4. Record non-finite counts, max absolute/relative error, cold module load,
   cold first launch, resources, and device-memory deltas.
5. Use the common policy's CUDA-event ABBA/BAAB method, 100 normalized samples,
   and 16 launches per warm sample; sanitizer runs stay at one launch.
6. Run memcheck, initcheck, racecheck, and synccheck separately and fail
   closed.

## ABI freeze requirement

No host argument array is encoded yet. The target source surface is now eight
visible parameters in this order:
`w_qs, w_d, x_qs, x_d, y, n_tok, out_stride, numeric_stream_grid`.
Triton is expected to append the required global/profile scratch pointers,
giving ten total host argument addresses. Runtime `n_tok` preserves boundary
coverage, and runtime `numeric_stream_grid` prevents embedding the local
device's SM count. Earlier contradictory shape-specialized argument counts are
superseded by this surface decision.

The expected aliases are:

- `w_qs = weights_base + 2`, guaranteed two-byte rather than sixteen-byte
  alignment;
- `w_d = weights_base`, with scale half index
  `(row*nblocks+block)*9`;
- `x_qs = q8_base`;
- `x_d = q8_base + 128`;
- `y` is a separately guarded fp32 output;
- ordinals 5 through 7 are runtime i32 values;
- ordinals 8 and 9 are addresses of independent zero-valued global/profile
  scratch `CUdeviceptr` variables.

Executable implementation may start only after final per-shape metadata
corroborates that expected surface against the cubin KPARAM table. That
metadata must state, in ordinal order:

- every visible parameter's type and alias;
- `numeric_stream_grid` as an explicit runtime parameter;
- required global/profile scratch pointers even when allocation size is zero;
- total KPARAM size and argument count;
- opaque symbol, grid, block, dynamic shared memory, registers, local memory,
  and hidden scratch sizes.

The runner must reject a metadata/KPARAM mismatch before `cuLaunchKernel`.
This is a hard requirement derived from the first candidate's six-versus-eight
argument host access violation.

## Contraction seam oracle

For shape B, `B=K/32=320`, `ntiles=80`, and the runtime numerical stream
grid is 30 on the evidence device. For canonical tile
`tile=(row/128)*4+(token/128)`:

```text
worker = max(ceil(tile * 30 / 80), 1)
boundary = floor(worker * (80 * 320) / 30)
boundary -= (boundary % 320) % 8
seam = boundary % 320
```

The seam is used only when `worker<30`, `boundary/320==tile`, and the
remainder is nonzero. Twenty tiles have a seam:

- seam 208 (K=6656): tile IDs 2, 10, 18, 26, 34, 42, 50, 58, 66, 74;
- seam 104 (K=3328): tile IDs 5, 13, 21, 29, 37, 45, 53, 61, 69, 77.

The structured oracle performs a prefix fold over `[0,seam)`, a suffix fold
over `[seam,B)`, then combines `suffix+prefix`. A no-seam tile uses one
continuous prefix fold. Directed tests must cover block boundaries 103/104
and 207/208, representative seam tiles 5 and 2, and adjacent no-seam tiles.
The checked metadata records the general formula and vectors; the executable
must not specialize correctness to the RTX 3060 tile list.

## Remaining policy inputs

The sole `config/kernel-lab-policy.toml` will receive a dedicated Q4 section
only after the final ABI and cubins exist. The following values remain
deliberately unspecified rather than hard-coded:

- native-versus-Triton output tolerance;
- f64 reference tolerance and reference sampling contract;
- final per-shape Triton dynamic shared memory and resource ceilings;
- candidate-specific projected end-to-end share.

Production binding, Backend ABI changes, selector/tuner changes, receipts,
signing, installer, trust, graph integration, and non-SM86 claims remain out
of scope.
