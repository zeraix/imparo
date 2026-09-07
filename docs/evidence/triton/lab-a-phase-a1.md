# LAB-A Phase A1 hotspot selection

Date: 2026-08-28

This is feasibility evidence under `handoff/triton-program-packs.md` R2. It is
not a production receipt, Program winner, release result, or cross-SM claim.

## Bound execution identity

- Branch and HEAD: `triton@e53b18a9093f189ea67994b8d00ca7d7e463c7c2`.
- Backend ABI: 26; exact target: SM86 only; math mode: fast.
- Native semantic build id:
  `1c7e79c7cb62a7d28ca9a329cdfd751f769557f93868df4ef5a00ef20c278025`.
- Locally rebuilt CI-recipe DLL SHA-256:
  `7ac241b74090f184767a2f89bea2dd8fd9f79e94df35dc26aa61281fa59ed39d`.
- Device: NVIDIA GeForce RTX 3060 Laptop GPU, SM 8.6,
  UUID `GPU-a69e87d6-c66a-de6a-ea36-5d1e861aba1d`, 6144 MiB.
- Driver: 596.08 (CUDA driver identity 13020 in existing receipts).
- Profile-start state observed by `nvidia-smi`: P0, 53 C, 18.36 W.
- Model: `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`, 4215695776 bytes,
  SHA-256 `df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3`.
- Workload: one 512-token prefill, f16 KV, default native route, Graph off.

The DLL was built with the Windows release workflow's SM-specific recipe:
CUDA 12.9 `nvcc`, `-O3 -std=c++17 -lineinfo --use_fast_math -shared`, static
cudart, `/MT`, `compute_86,sm_86`, and the repository export definition. The
common runtime was separately built with locked Rust 1.85 and `cuda-dynamic`.

## Current-HEAD logical matmul profile

`IMPARO_CUDA_PROFILE_MATMUL=1` uses CUDA events around each complete native
`imparo_cuda_matmat` call. The 512-token run completed with exit code 0,
344 measured calls, 180.983 ms total measured matmul time, and finite logits.

| Q4_0 logical shape | epilogue | calls | total ms | matmul share | median us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10240 -> 2560 x 512 | 0 | 42 | 46.53 | 25.71% | 1105.92 |
| 2560 -> 10240 x 512 | 1 | 42 | 46.47 | 25.68% | 1040.38 |
| 2560 -> 10240 x 512 | 0 | 42 | 36.98 | 20.43% | 886.78 |

The minimal second LAB-A contract excludes the fused epilogue. Its two real FFN
directions therefore account for 46.14% of this current run's measured logical
matmul time. The epilogue-1 route is retained as future evidence, not silently
included in the candidate's claimed coverage.

## Independent Nsight selection cross-check

The earlier structured capture
`target/cuda-plugin-test/nsys-e4b-fast-baseline.sqlite` is a 512-token E4B
prefill on the same RTX 3060. A read-only aggregation over
`CUPTI_ACTIVITY_KIND_KERNEL` gives 353.782942 ms total kernel time, all launches
on stream 13. The strict plain Q4_0 x Q8_1 contract contains:

| Native kernel phase | calls | total ms | all-kernel share |
| --- | ---: | ---: | ---: |
| `q4_q8_1_stream<false,true>` | 258 | 108.929935 | 30.7900% |
| `q4_q8_1_full_tile<false>` | 43 | 71.003837 | 20.0699% |
| `q4_q8_1_stream_fixup` | 258 | 12.275637 | 3.4698% |
| **plain contract total** | | | **54.3298%** |

The fused `full_tile<true>` phase is another 23.2548%, but is outside the first
plain contract. Historical and current-head measurements therefore agree on the
choice of the second candidate without broadening its semantics.

## Decision and evidence boundary

LAB-A candidate order is now fixed:

1. mandated fused `RMSNorm -> Q8_1Mmq`, exact shape width 2560, rows 512;
2. plain Q4_0 x Q8_1 MMQ, epilogue 0, rows 512, FFN shapes
   2560 -> 10240 and 10240 -> 2560.

The Nsight capture selects a hotspot only. Neither it nor the current logical-op
profile compares Triton, establishes a noise floor, or passes Gate A. Gate A still
requires per-op correctness, boundary/alignment/canary and sanitizer checks,
same-device/same-shape/same-stream interleaved native/Triton CUDA-event data,
resources, cold/warm behavior, and the versioned policy decision.
