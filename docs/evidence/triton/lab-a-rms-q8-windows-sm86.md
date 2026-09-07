# Phase A2 LAB-A RMSNorm -> Q8_1 MMQ Windows SM86 evidence

Status: **first Gate A candidate admissible; final Gate A not decided**.
This is isolated RTX 3060 / SM86 kernel-lab evidence. It does not authorize a
production binding, Program Pack, release ABI, receipt, signing, installer,
trust root, graph integration, or cross-SM claim.

## Identity and scope

- Branch/starting HEAD: `triton@e53b18a9093f189ea67994b8d00ca7d7e463c7c2`.
- Workload: `rms_norm_q8_1_mmq`, `width=2560`, `n_tok=512`.
- Native control: exact existing `k_rms_norm_q8_1_mmq<1024>`.
- Candidate modules and build identity are recorded in
  `program-packs/lab/rms-q8-sm86/run-g/lab-metadata.json`.
- Hardware evidence is exact SM86 only. No SM80 hardware result is claimed.
- Sole threshold authority: `config/kernel-lab-policy.toml`.
- Compile-time isolation: `IMPARO_CUDA_KERNEL_LAB=1`; the lab translation
  unit is excluded from release compilation and exports.

## Launch ABI finding

The first Driver launch ended with Windows access violation `0xC0000005`
inside `cuLaunchKernel`. Context creation, module loading, function lookup,
resource queries, allocation, upload, and the exact native control had all
completed.

Cubin `.nv.info` and the pinned Triton 3.8 launch template established a
64-byte, eight-parameter KPARAM ABI. The six visible parameters occupy
ordinals 0 through 5. Triton appends `global_scratch` and
`profile_scratch` device pointers at ordinals 6 and 7 even when both
allocation sizes are zero. Passing only six host argument addresses allowed
the Driver to read beyond the argument array. The corrected harness passes
the addresses of two independent zero-valued `CUdeviceptr` variables.

The complete visible and hidden ABI is frozen in the run-g metadata. The
initial six-argument crash is retained as a negative bring-up result and was
not counted as kernel evidence.

## Measurement method

The original method timed one launch per CUDA-event sample. Two complete
fail-closed runs demonstrated Windows scheduling/quantization noise:

1. `artifacts/kernel-lab/rms-q8-sm86/run-g`: correctness, resources,
   canaries, all four sanitizers, and the kernel floor passed, but one w8
   sample reached 285.600 us. Raw candidate CV was 0.41955, so the candidate
   was rejected.
2. `artifacts/kernel-lab/rms-q8-sm86/run-g-repeat`: correctness,
   resources, canaries, all four sanitizers, and the kernel floor passed, but
   native CV was 0.05227 for w4 and 0.06410 for w8, above the unchanged 0.05
   ceiling. This run was also rejected.

No sample was deleted and the 5% CV threshold was not relaxed. The policy now
fixes `launches_per_sample=16`. Each warm CUDA-event sample contains sixteen
consecutive launches of one route and records elapsed time divided by sixteen.
ABBA/BAAB remains interleaved at the sample-batch level, with 50 pairs and 100
normalized samples per route. Shape, primary context, exact engine stream,
warmup, and cold single-launch measurements are unchanged. Sanitizer runs use
one launch per sample because they validate memory and synchronization rather
than timing noise.

## Admissible candidate result

The complete command through `dev_harness/run_cuda_kernel_lab.ps1` exited
`0`. Evidence is under
`artifacts/kernel-lab/rms-q8-sm86/run-g-batch16`.

| Metric | w4-s1 | w8-s1 |
| --- | ---: | ---: |
| Native median | 73.278 us | 67.392 us |
| Triton median | 50.880 us | 49.024 us |
| Kernel speedup | 1.4402x | 1.3747x |
| Native / candidate CV | 0.01812 / 0.01143 | 0.01362 / 0.01246 |
| Paired MAD fraction | 0.00500 | 0.00572 |
| Normalized max abs / rel vs native | 2.384e-7 / 1.192e-7 | 2.384e-7 / 1.192e-7 |
| Q8 scale max abs vs native | 0 | 0 |
| Q8 byte agreement | 0.999818 | 0.999818 |
| Q8 value max delta | 1 | 1 |
| Q8 dequant max abs | 0.020590 | 0.020590 |
| Registers / dynamic shared / local | 72 / 2,048 B / 0 | 48 / 4,096 B / 0 |
| Cold first launch | 71.520 us | 72.704 us |
| Module load wall | 452.5 us | 78.6 us |
| Module device-memory delta | 2,097,152 B | 0 B |

The native cold first launch was 18,739.201 us. Guarded buffer allocation
changed reported free device memory by 23,068,672 bytes. Both variants had
zero canary errors and the top-level structural result was true.

The decision file records:

- `candidate_admissible=true`
- `final_gate_a_decision=false`
- `correct_variants=2`
- `fast_variants=2`
- `kernel_floor_met=true`
- no violations

## Sanitizer and software gates

The same formal run executed each sanitizer separately and retained one log
per tool:

- memcheck: exit `0`, 0 errors
- initcheck: exit `0`, 0 errors
- racecheck: exit `0`, 0 hazards, 0 errors, 0 warnings
- synccheck: exit `0`, 0 errors

The following software checks also exited `0`:

```text
python -B -m unittest dev_harness.test_cuda_kernel_lab -v
git diff --check -- config/kernel-lab-policy.toml \
  crates/imparo-cuda/native/tests/kernel_lab_rms_q8.cu \
  dev_harness/kernel_lab_tools.ps1 \
  dev_harness/run_cuda_kernel_lab.ps1 \
  dev_harness/test_cuda_kernel_lab.py
```

Five contract tests passed. All five new lab source/config/test files contain
LF line endings and zero CRLF sequences. Diagnostic phase tracing is disabled
by default and can be enabled only with
`IMPARO_CUDA_KERNEL_LAB_TRACE=1`.

Key SHA-256 values:

- harness executable:
  `ea671441b5d90c005f86e3c9d15c83e64287e77fc02b538eb5da183edc352543`
- result:
  `e465b49d5f05087b104a9872cfcc41a7f038112472f4761bb93e30bdb2fb94f9`
- candidate decision:
  `6a770f8f4176af4d4abe342b9656b6ce3337c668b0da9dac7de791abd8db6072`

## Remaining boundary

This evidence closes only the first candidate. Final Gate A still requires
the separately frozen second candidate and high-share workload coverage.
Second-candidate Windows code and policy are intentionally not guessed before
its exact native control, fixed shape, layouts, visible and hidden argument
ordinals, launch geometry, resources, and correctness contract are supplied.
