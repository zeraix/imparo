# SM86 CUDA benchmark protocol

This protocol separates hypothesis screens from cross-engine promotion claims.
It applies to single-GPU Windows development and is intentionally independent
of one laptop, model path, or experimental selector.

## 1. Serialize GPU work

Only one correctness, sanitizer, profiler, or performance process may use the
GPU at a time. Parallel agents may perform source review, CPU tests, compilation,
and documentation, but must wait for the benchmark owner before launching GPU
work.

Before every performance leg, require five continuous seconds of:

- P8;
- zero GPU utilization; and
- at most 64 MiB device memory in use.

Failure to reach idle invalidates the session. It is not a candidate No-Go.

## 2. Start from an environment allowlist

Remove inherited `IMPARO_*`, `GGML_*`, `LLAMA_*`, `OMP_*`, and `KMP_*`
variables plus CUDA debug/tuning variables. Add back only the declared control
or candidate route. In particular, keep profiler, trace, failure injection,
forced OOM, capture, probe, and unrelated `*_LAB` variables unset.

Record both the variables that are set and the rejected/unset families. A
benchmark process must not silently inherit an interactive shell's experiment.

CUDA-event operator profiling is mechanism evidence. Its per-operation
synchronization changes the execution schedule, so never compare a profiled
whole run with an unprofiled whole run.

## 3. Warmup and balanced ordering

For a same-binary candidate screen, use balanced interleaving such as
control/candidate/candidate/control. For a cross-engine claim, use two balanced
blocks:

```text
A B B A   B A A B
```

Use at least 15 requests per leg. Discard requests 0, 1, and 2; analyze requests
3 through 14. Give every server leg a unique port and preserve its stdout,
server log, and telemetry before the next leg can overwrite temporary files.

Every measured request must prove the same evaluated-token count, no prompt or
KV reuse, the same KV type, the same FA policy, and the same output work.

## 4. Record load-time GPU telemetry

Sample at 100 ms throughout each leg:

- timestamp and P-state;
- temperature and power;
- SM and memory clocks;
- GPU utilization and memory use; and
- software/thermal/power-brake throttle reasons.

Reject a session when another GPU workload overlaps, thermal or hardware power
brake throttling activates, same-engine load-clock medians differ by more than
2%, starting temperatures span more than 3 C, or same-engine load-power medians
drift more than 5%. A laptop's software power-cap state is recorded but is not
alone a rejection.

## 5. Per-leg stability gate

For the 12 measured samples in each leg, require:

- `MAD / median <= 1%`;
- `(P90 - P10) / median <= 3%`; and
- the first-six and last-six medians differ by at most 1.5%.

All commands must exit zero. A telemetry or stability failure means invalid
evidence and must be rerun; it must not be converted into a pass or a No-Go.

## 6. Ratio and promotion rule

Let `L1..L8` be the steady-state leg medians for `ABBA BAAB`:

```text
R1 = sqrt(L2 * L3 / (L1 * L4))
R2 = sqrt(L5 * L8 / (L6 * L7))
screen_ratio       = sqrt(R1 * R2)
conservative_ratio = min(R1, R2)
```

The target ratio is an input to the session and is SHA-bound before any leg
starts. Do not hard-code the commercial 1.20x objective into an open-source
parity decision, and never select a target after seeing the samples. For any
declared target `T`:

- `R1`, `R2`, and `conservative_ratio` must all be at least `T`;
- the 95% lower confidence bound must be at least `T`;
- a result below `T`, or a confidence interval crossing `T`, is not achieved;
- an invalid leg is rerun rather than counted as a ratio; and
- a first passing eight-leg session is repeated after a fresh idle period before
  release promotion.

The confidence calculation is fixed rather than chosen after measurement. Use
seed `20260831` and 20,000 bootstrap replicates. In each replicate, independently
resample with replacement the 12 measured target-metric samples inside every
leg, take each leg median, and recompute `R1`, `R2`, and `screen_ratio`. Sort the
20,000 `log(screen_ratio)` values; exponentiating the element at
`floor(0.05 * 20000)` gives the 95% lower bound. The aggregate receipt binds the
implementation hash, seed, replicate count, target metric and target ratio.

Cross-check each engine's reported Prefill timing with external wall time for
the same max-output request. Their ratio must agree within one percentage point.

## 7. Final E4B/SM86 matrix

Run each regime as an independent `ABBA BAAB` session. Do not average a weak
short-Prefill result into a stronger Decode or long-Prefill result:

| regime | raw input tokens | max output tokens | target metric |
| --- | ---: | ---: | --- |
| exact short Prefill | 128 | 1 | Prefill tok/s |
| common short Prefill | 449 | 1 | Prefill tok/s |
| long Prefill | 5651 | 1 | Prefill tok/s |
| short Decode | 449 | 128 | Decode tok/s |
| long Decode | 5651 | 128 | Decode tok/s |

Before the cross-engine matrix, run a same-binary shape-regression matrix for
slot 39 safe-off versus the production receipted route at 128/449/5651 Prefill
and 449/5651 Decode. Separately prove, without performance timing, that the
complete exact-route reachability mask appears at 128 only; 127/129/449/512 and
5651 must retain their established routes. Trace/profiler processes are never
used as performance legs.

Every request records a canonical raw-token-vector SHA-256, endpoint request and
response SHA-256, evaluated-token count, cached-token count and completion-token
count. Across matching legs, the canonical workload hash must agree, cached
tokens must be zero, evaluated tokens must equal the raw length, and completion
tokens must equal the declared maximum. Missing counters make the leg invalid;
they are not interpreted as zero. Llama runs use explicit FA on, context
checkpoints off and q4_0 K/V. Imparo performance legs load only the production
host config plus its adjacent correctness receipt; laboratory, profiling, trace,
capture and failure-injection variables are absent.

The strict single-leg runner requires 15 requests, exactly 3 warmups, 12 measured
samples, a receipt directory, an externally produced telemetry artifact and a
passing telemetry verdict. The session aggregator accepts only eight unique
ports in exact `ABBA BAAB` order and rejects any identity, workload, stability,
telemetry or target drift before computing a promotion result.

## 8. Identity receipt

Bind every claim to:

- source commit plus dirty diff or manifest;
- executable and loaded backend/DLL hashes;
- reference source role: correctness oracle or performance target;
- model, tokenizer, and benchmark-script hashes;
- device, SM, driver, toolkit, compiler, and build flags;
- complete commands and environment allowlist;
- raw request samples, exit codes, server logs, and token/no-reuse proof;
- telemetry and stability calculations; and
- numerical, Decode, determinism, sanitizer, and rollback gates appropriate to
  the changed route.

An observed peak remains an observation until this receipt is complete. A
confidence interval crossing the target means "not proved", not success.
