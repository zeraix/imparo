# SM86 CUDA experiment ledger

This file is the durable decision record for SM86 CUDA performance work. It is
not a list of every kernel revision. It records enough context to reproduce a
result, explains why a candidate was accepted or rejected, and keeps failed
experiments available as design evidence.

The entries below describe the 2026-08-30 RTX 3060 Laptop GPU session. The
worktree was detached at `9424517daf234e1ae71a4b8435be0c7f1e2ed0ee` and had
uncommitted CUDA experiments, so the commit alone is not a complete source
identity. Before any result is promoted to a release policy, attach the exact
diff or commit, binary hash, command line, environment, driver, and raw output.

## Decision vocabulary

- **Admitted**: complete correctness, resource, sanitizer, and end-to-end
  evidence supports enabling the route inside a narrow selector.
- **Validated lab**: the numerical and measured performance hypothesis is
  positive, but one or more promotion receipts are still missing. It remains
  opt-in.
- **Promising**: an early interleaved performance screen is positive, but the
  effect is still within the local noise band or required correctness evidence
  is incomplete. It is not a winner or an admission candidate yet.
- **No-Go**: the candidate was measured and rejected. Keep the reason so the
  same idea is not rediscovered under a new name.
- **Observation**: useful local data that is not portable policy.

## Session summary

The table below reflects the latest result in this session. Earlier records are
retained as chronology, but a later **No-Go** supersedes an earlier positive lab
observation for admission purposes.

| Experiment | Decision | Main result | Remaining boundary |
|---|---|---|---|
| Q8-only gate/up output | Contract failure found and repaired | Dense `G` must remain authoritative while the fused MMA-ready Q8 representation is optional | Re-run dense fallback, Q8 byte identity, sanitizers, and whole A/B before promotion |
| R64xT256 pair-CTA | **No-Go** | 2526.11 tok/s versus R64xT128 2536.40 tok/s, about -0.41% | Do not enlarge the token tile without a different latency-hiding mechanism |
| PLE fused route | **No-Go** | 2634.37 tok/s | Whole-run result did not justify replacing the established PLE boundary; keep opt-in disabled |
| Pair projection route | **No-Go** | 2657.61 tok/s | No demonstrated whole-run win over the current control; retain independent projections |
| RMS cache reuse | **No-Go** | 0 cache hits and 2601.38 tok/s | The intended reuse did not occur, so the route paid machinery without removing work |
| RMS float-only route | **No-Go** | 2658.14 tok/s | Removing the fused Q8-ready handoff was not beneficial on this pipeline |
| Packed-Q4 sidecar coverage | **Observation** | 1800-MiB budget covered the complete admitted set; 1786.64 MiB used | Capacity above full coverage cannot improve hit coverage; preserve runtime headroom policy |
| R64 down Stream-K grid sweep | **No-Go** | grids 60/80/90/100/120/160: 2461.79/2458.51/2482.61/2343.09/2375.92/2505.50 tok/s | No tested forced grid beat the current route; do not promote a device-count-derived grid |
| Virtual-substitution (`vsub`) route | **No-Go** | 2667.81 tok/s and worse SASS | Source-level simplification did not survive lowering as an instruction improvement |
| Q8 + prefetch route | **No-Go** | 2687.88 tok/s | Prefetch overhead/pressure did not produce a sufficient whole-run gain |
| Exact REDUX / vector-scale screen | **No-Go** | redux repeat-12 median 2685.22 vs same-binary subsequent baseline 2684.69 tok/s, about +0.02% | SASS and registers improved, but whole-run difference was within noise; vec-scale compiled identically to redux |
| Direct-register Q8 handoff | **Validated lab; logit gate passed** | initial 2686.96; token-pair `uint32` version 2721.82 tok/s; q4_0 logit agreement passed at 128/449/512/2000 | Decode, determinism, and all four Compute Sanitizer classes remain pending |
| Sidecar-only complete FFN transaction | **Promising; numerical gates passed** | four interleaved 449-token legs give cross-leg medians B=2698.505 and I=2725.800 tok/s, about +1.011%; logit/decode/determinism gates exit 0 | Within local machine noise; failure injection, binary identity, and sanitizers remain pending |
| Strict sidecar/llama.cpp bracket | Provisional comparison | llama.cpp 2253.53 -> Imparo sidecar 2717.32 -> llama.cpp 2231.89 tok/s; 1.20580x versus the faster llama.cpp run and 1.21162x versus the two-run llama.cpp median of 2242.71 | Current machine/reference binaries only; binary and model hashes remain pending |
| R32 two-resident-CTA direct-Q8 producer | **No-Go** | R32 cross-leg median 2381.545 versus same-binary R64 2666.810 tok/s, `0.89304x` (-10.70%) despite 86 registers and two theoretical resident CTAs | Keep exact opt-in laboratory-only; occupancy alone did not repay the extra tile, barrier, and shared-memory traffic |
| Exact-449 full-K down, R128/R64 | **No-Go** | physical control cross-leg median 2732.27 tok/s; R128 2676.77 (0.97969x), R64 2561.74 (0.93758x). CUDA-event down averages 0.7340/0.8368/0.9802 ms | Removing fixup did not repay coarser full-K ownership; do not retry these exact grids/schedules |
| Fresh clock-unbound cross-engine bracket | **Observation; 1.2x not yet stable** | llama 2288.53 -> Imparo 2647.91 -> llama 2265.61 tok/s: 1.15704x conservative; immediate Imparo repeat 2695.94 is 1.17802x | Clock/power samples were not captured under load; the earlier 1.20580x observation is not a reproducible admission claim |
| Exact-449 AOT Stream-K schedule | **No-Go** | same-binary generic/AOT/AOT/generic medians 2682.36/2679.68/2679.73/2659.49 tok/s; cross-leg gain only `1.00329x` (+0.33%) | Correct and spill-free, but below the predeclared 1% whole-Prefill continuation threshold; retain exact selector default-off |
| Exact W4A8 digit-sliced SM86 MMA | **No-Go** | bit-exact CPU/GPU Gate, but consumer `0.75075x` and cached raw-input operator `0.93767x` versus one S8xS8 MMA control | Two INT4 MMA instructions plus integer combine cost more than the removed Q4/Q8 digit transformation in this m16n8k32 design; keep standalone only |
| Exact-key full-logits Prefill Graph | **No-Go as primary route** | strict same-binary whole-Prefill A/B measured `+0.52%` | Capture/replay is valid, but the present boundary removes too little host work to clear the `3%` continuation floor; retain as a later additive candidate |
| Exact-128 Direct-K + ready-Q8 + full-K FFN | **Validated lab; auto-fit closed** | without a reserve override, strict 128-token Imparo `2035.55 tok/s` versus two-leg llama `2065.08 tok/s` (`0.98570x`); latest reclaim build quick-checks at `1.01325x` llama | Four sanitizers and formal model gates pass; remaining work is a receipted atomic tuner selector and runtime OOM-injection evidence for the new priority-reclaim path |

## Experiment records

### E-20260830-01: Q8-only output and the dense-buffer contract

**Hypothesis.** The accepted R64 shared-gate pair-CTA can quantize its final
`GELU(gate) * up` tile directly into the MMA-ready split Q8 layout, allowing the
immediately following 10240 -> 2560 down projection to reuse it.

**Invariants.** The dense `G` buffer is the semantic backend output. The Q8
sidecar is a cache of that exact buffer epoch, not a replacement for it. A
cache miss, disabled consumer, missing packed sidecar, probe, or conservative
fallback must still read correct dense values. Input and output Q8 scratch must
be distinct while the pair kernel is running. Cache publication order is
`mark_buf_written(G)`, scratch swap, then `own_q8_cache(G, ..., MMA_READY)`.

**Outcome.** The first Q8-only version wrote the final value only to shared
memory and the Q8 sidecar, then advanced the dense-buffer epoch. This violated
the invariant: the direct Q8 consumer could appear correct while dense
fallbacks read stale `G`. The implementation was repaired to compute `result`
once, always write it to dense `G`, and additionally write the same f32 value
to shared memory when fused quantization is requested.

**Correctness status.** The design-level failure is closed in source, but this
entry is not a performance admission. Required evidence is byte-for-byte Q8
comparison against the standalone authority producer, exact dense comparison,
449 -> 512 tail zero-fill, forced no-consume fallback, missing-sidecar fallback,
logit agreement, and all four Compute Sanitizer classes.

**Decision and rollback.** Keep the fused Q8 path laboratory-only. Disable the
fused-output selector to return to the accepted dense R64 shared-gate path and
standalone down quantizer.

**Migration lesson.** A derived representation may be promoted only after the
authoritative value is materialized and its epoch is advanced. A fast immediate
consumer does not weaken the public buffer contract.

### E-20260830-02: R64xT256 pair-CTA weight reuse

**Hypothesis.** One 16-warp R64xT256 CTA could reuse each gate/up Q4 weight tile
across two 128-token halves and keep the same 16 resident warps as two R64xT128
CTAs.

**Environment and resources.** SM86, 30 SMs, 2560 -> 10240 gate/up shape. The
compiled candidate used 128 registers, zero stack and local memory, and 65,536
bytes of dynamic shared memory. It therefore had no register spill, but only
one large CTA residency and a wider barrier cohort.

**End-to-end result.** R64xT256 measured 2526.11 tok/s; the R64xT128 control
measured 2536.40 tok/s. The candidate was about 0.41% slower.

**Decision and rollback.** **No-Go.** Leave R64xT128 as the control and keep the
T256 selector disabled.

**Migration lesson.** Weight reuse alone is not sufficient. Doubling a CTA's
token ownership also enlarges shared state and synchronization scope, removes
an independent CTA that could hide latency, and can neutralize saved loads.
Future wide-token work needs a pipeline that demonstrates overlap rather than
assuming equal resident warp count means equal latency hiding.

### E-20260830-03: 10240 -> 2560 physical Stream-K grid sweep

**Hypothesis.** The default physical grid underfills the 30-SM device for the
canonical down projection. A grid of 60, exactly two work units per SM, may
improve scheduling while preserving the existing Stream-K ownership and fixup
order.

**Environment.** SM86 RTX 3060 Laptop GPU, 30 SMs, canonical short-Prefill down
projection, Q8-ready activation and packed Q4 weights. Grid forcing uses the
laboratory stream-grid override; it is bounded between one-SM grid and the
logical tile count.

**Result.** Grid 60 produced a robust 2601.16 tok/s. The configured numerical
gate passed 10/10, with maximum distribution delta 0.12315.

**Decision and rollback.** **Validated lab**, not admitted. Removing the grid
override restores automatic grid selection. Promotion requires all four
sanitizers, repeated A/B/B/A evidence, 449 and 512 tails, additional token
shapes, workspace-OOM fallback, and a receipt binding device, driver, model,
numeric route, and binary identity.

**Migration lesson.** Grid size is architecture and shape policy, not a kernel
constant. Express candidates relative to SM count and logical work, then tune
under a receipt. `60 == 2 * 30 SM` is the observed geometry, not a portable
magic number.

### E-20260830-04: dual RMS + MMA-ready Q8 producer

**Hypothesis.** The adjacent post-attention residual normalization and FFN
normalization can share one read/reduction boundary while directly producing
the MMA-ready Q8 input used by gate/up.

**Invariants.** Both dense normalized outputs remain valid; the second output's
Q8 cache must match its dense epoch and exact split layout. Metal and other
backends retain the default trait fallback. Probes, graph capture, unsupported
widths, and unsupported token counts fall back to the established sequence.

**Resources.** 36 registers per thread, zero stack, zero local memory, and
10,368 bytes shared memory on SM86.

**Result.** On the validated grid-60 base, whole short-Prefill moved from
2601.16 to 2649.24 tok/s, a 1.85% gain. The logit gate passed.

**Decision and rollback.** **Validated lab.** It remains behind the dual-RMS
Q8-ready opt-in. Removing that opt-in restores the two established normalization
calls. Promotion still needs dense-output comparison for both outputs, raw Q8
identity, four sanitizers, graph/probe fallback checks, and A/B across token
shapes and sidecar policies.

**Migration lesson.** Producer-consumer fusion can give a useful whole-engine
gain even when no single downstream kernel changes. Its receipt must bind the
entire boundary, including dense outputs, derived cache, and fallback behavior.

### E-20260830-05: packed-Q4 sidecar memory budget sweep

**Hypothesis.** A larger packed-Q4 cache may keep more hot projections on the
fast path, but excessive reservation can reduce runtime headroom and degrade
the rest of the engine.

**Observed short-Prefill throughput.** These are local observations from one
RTX 3060 Laptop GPU session:

| Sidecar budget | Throughput |
|---:|---:|
| 1800 MiB | 2677.14 tok/s |
| 2000 MiB | 2666.89 tok/s |
| 2200 MiB | 2680.27 tok/s |
| 2400 MiB | 2661.74 tok/s |
| 2600 MiB | 2608.12 tok/s |

The 2200-MiB setting was repeated for 12 samples and produced a median of
2673.05 tok/s.

**Decision and rollback.** **Observation only.** Keep the existing configurable
budget and conservative fit fallback. Do not make 2200 MiB a compiled default.
The rollback is to remove the budget override and use the runtime-derived
headroom policy.

**Migration lesson.** Cache capacity is a discontinuous placement decision, not
a monotonic performance knob. A portable tuner should choose a set of tensors
under a measured free-memory guard, record which tensors were resident, and
fall back safely when allocation fails.

### E-20260830-06: same-session pinned llama.cpp comparison

**Scope.** The final local configuration used the 2200-MiB sidecar observation,
the grid-60 laboratory candidate, and the dual-RMS Q8-ready candidate. In the
same session, Imparo measured 2673.05 tok/s and the pinned local llama.cpp CUDA
reference measured 2265.99 tok/s.

`2673.05 / 2265.99 = 1.1796x`.

**Decision.** This is positive same-session evidence, but it is not rounded up
to 1.2x and is not yet a release claim. The formal receipt must preserve exact
commands, prompt/token counts, warmup and repetition schedule, Flash Attention
state, KV types, model hash, binary hashes, driver, clocks, and raw per-leg
values. Any token-count mismatch invalidates the formal cross-engine ratio even
when both routes execute a nominal 512-token tile.

### E-20260830-07: follow-up cross-operator fusion and reuse probes

**Scope.** Four laboratory routes were measured on the current 449-token SM86
Prefill configuration. The measurements are whole-run throughput; they are not
isolated-kernel timings and must not be compared across different sessions.

| Candidate | Throughput | Decision | Measured reason |
|---|---:|---|---|
| PLE fused route | 2634.37 tok/s | **No-Go** | The removed boundary did not translate into a competitive whole-run result |
| Pair projection | 2657.61 tok/s | **No-Go** | Sharing projection work/launch structure did not recover enough end-to-end cost |
| RMS cache reuse | 2601.38 tok/s | **No-Go** | Instrumentation recorded 0 cache hits; the intended reuse never removed an authority RMS/quantization operation |
| RMS float-only | 2658.14 tok/s | **No-Go** | Keeping only the float result did not beat the Q8-ready producer/consumer pipeline |

**Interpretation limits.** The zero-hit RMS result proves that this route did
not exercise reuse in the measured workflow. It does not establish whether a
correctly hit cache would be fast, and no unobserved root cause is asserted.
Likewise, the PLE and pair results reject these exact selectors and schedules,
not all possible PLE or projection fusion.

**Applicable boundary.** These decisions apply to the measured single-request,
449-token/padded-tile SM86 workflow and the tested Q4/Q8 layouts. They do not
admit or reject training, multi-request batching, other widths, or other SMs.

**Rollback.** Keep every route strictly opt-in. For the PLE persistent
prototype, leave `IMPARO_CUDA_PLE_PERSISTENT_V1` unset or `0`. Disable the
pair-projection, RMS-cache-reuse, and RMS-float-only laboratory selectors to
fall back to the established independent projection and normalization calls.
The raw receipt supplied for this update did not bind the exact environment
variable spelling for those three selectors, so this ledger intentionally does
not invent names; the next reproducibility receipt must add them.

**Migration lesson.** A fused launch is useful only if instrumentation proves
that the intended intermediate operation was actually removed. Record cache
attempts and hits beside throughput, and treat a zero-hit run as a mechanism
failure rather than evidence about the hypothetical hit path.

### E-20260830-08: packed-Q4 sidecar reaches full useful coverage

**Observation.** With a configured 1800-MiB sidecar budget, the current admitted
packed-Q4 set was fully covered and reported 1786.64 MiB used.

**Decision.** This supersedes the earlier interpretation that 2200 MiB was
needed for coverage. The earlier 2200-MiB throughput remains a valid local
sample, but capacity above 1786.64 MiB cannot improve hit coverage for this
exact admitted set. Any difference above that point must be attributed to
measurement variance or secondary memory effects unless counters prove
otherwise.

**Boundary and rollback.** The byte count is specific to the current model,
tensor admission set, packing layout, and process lifetime. Keep the
runtime-derived free-memory guard and allocation-failure fallback; removing the
laboratory budget override restores that policy. Do not compile 1800 MiB or
1786.64 MiB as a device-independent constant.

**Migration lesson.** Report both budget and actual used bytes. Tune admission
by hot-tensor benefit under a live headroom guard, and stop increasing capacity
once the admitted working set is resident.

### E-20260830-09: R64 down-projection grid follow-up

**Hypothesis.** A different forced physical grid might improve the R64
10240 -> 2560 down projection while preserving the existing Stream-K ownership,
rounded seam, K-phase, MMA/FMA, and fixup order.

| Forced grid | Throughput |
|---:|---:|
| 60 | 2461.79 tok/s |
| 80 | 2458.51 tok/s |
| 90 | 2482.61 tok/s |
| 100 | 2343.09 tok/s |
| 120 | 2375.92 tok/s |
| 160 | 2505.50 tok/s |

**Decision.** **No-Go** for all tested forced grids. This follow-up supersedes
E-20260830-03's earlier grid-60 **Validated lab** status. Grid 160 was the best
of this sweep, but a best member of a uniformly rejected sweep is not an
admission candidate.

**Failure reason.** Increasing or redistributing work units did not compensate
for the additional scheduling/fixup cost on the current route. The data does
not justify a more specific microarchitectural cause without event-level and
occupancy receipts.

**Boundary and rollback.** This rejects the exact R64 down shape and tested
449-token SM86 selector. It does not generalize the literal grid numbers to
GPUs with another SM count. Remove the laboratory stream-grid override to
restore automatic selection; do not leave a forced grid active by default.

### E-20260830-10: instruction substitution and prefetch probes

**Virtual substitution.** The `vsub` candidate measured 2667.81 tok/s and its
inspected SASS was worse than the control. **No-Go.** The source rewrite did
not lower to a cheaper instruction sequence, so there is no basis to accept it
on source-level operation count alone.

**Q8 plus prefetch.** The Q8+prefetch candidate measured 2687.88 tok/s.
**No-Go.** The tested prefetch distance and resource trade did not deliver a
sufficient whole-run improvement. This rejects the tested schedule only; it is
not proof that every asynchronous pipeline is ineffective.

**Numerical boundary.** Both probes were intended to retain the established
quantization and GEMM ordering. Since neither is being promoted, this record
does not upgrade that intent to a correctness claim.

**Rollback.** Disable the corresponding `vsub` and Q8-prefetch laboratory
selectors to return to the established SM86 kernel. Exact environment-variable
names were not present in the supplied measurement receipt and are deliberately
not guessed here.

**Migration lesson.** Treat generated SASS, registers, local memory, and whole
throughput as a single gate. A syntactically smaller expression or an added
prefetch is not an optimization if lowering or register pressure becomes worse.

### E-20260830-11: direct-register Q8 handoff

**Hypothesis.** Produce the downstream Q8 representation directly from the
producer's registers, avoiding a redundant dense/shared reload. The second
revision packs a token pair into a `uint32` transaction to reduce handoff
instruction and store overhead.

**Performance.** The initial direct-register version measured 2686.96 tok/s.
The token-pair `uint32` version measured 2721.82 tok/s and is the fastest
current result in this session.

**Numerical order.** The candidate is a producer-layout/store change,
not a new GEMM reduction schedule: it is intended to preserve the existing Q8
scale computation, rounding/saturation, lane-to-byte mapping, dense authority,
and downstream MMA order. Packing two tokens into one transaction must not
change per-token scales, signed-byte order, padded-tail zeroing, epoch
publication, or fallback data.

**Correctness evidence.** `logit_agree` with q4_0 and tolerance 0.75 exited
successfully for every tested token count:

| Tokens | Top-1 | Overlap | Maximum delta | Exit |
|---:|---|---:|---:|---:|
| 128 | identical | 9/10 | 0.31426 | 0 |
| 449 | identical | 10/10 | 0.12315 | 0 |
| 512 | identical | 8/10 | 0.41576 | 0 |
| 2000 | identical | 9/10 | 0.41741 | 0 |

All maximum deltas are below the configured 0.75 tolerance and top-1 agrees in
all four cases. This closes the stated logit-agreement gate, including the
449-token tail, the 512-token full tile, a smaller case, and a longer Prefill
case. It does not substitute for decode, determinism, or memory/concurrency
evidence.

**Decision.** **Validated lab; logit gate passed.** Do not default-enable or use
this result in a release claim. Recurrent decode agreement, determinism, and
Compute Sanitizer memcheck, initcheck, racecheck, and synccheck remain pending.
Byte-level Q8/layout evidence and fallback receipts remain required wherever
the promotion policy calls for them.

**Applicable boundary.** The numerical receipt now covers q4_0 token counts
128, 449, 512, and 2000 at tolerance 0.75 on the current SM86 route. Other
strides, alignment, quant formats, SMs, graph capture, probes, and OOM paths
must still take or validate against the established route.

**Rollback.** Keep the direct-register and token-pair routes behind their
laboratory opt-in selector. Unset it or force its predicate false to return to
the dense authoritative output plus established Q8 producer. The exact flag
name must be attached to the pending correctness receipt rather than inferred
in this document.

### E-20260830-12: updated same-session llama.cpp bracket

The current candidate measured 2721.82 tok/s and the same-session pinned
llama.cpp reference measured 2274.98 tok/s:

`2721.82 / 2274.98 = 1.19642x`.

This replaces the older session-summary bracket but does not erase its raw
historical sample. The current ratio remains below 1.2x. It must not be rounded
to 1.2x, described as meeting the target, or extrapolated beyond the exact
commands and token parity. The direct-register route has passed the stated
logit gate, but its pending decode, determinism, and sanitizer evidence still
prevents this from becoming a release comparison.

### E-20260830-13: exact REDUX and vector-scale screen

**Hypothesis.** Replace exact warp reduction/min-max instruction sequences with
the SM86 REDUX lowering, and separately test a vector-scale spelling, while
preserving the established values and numerical order visible at the kernel
boundary.

**Build and resources.** The build passed. The baseline compiled with 126
registers; both redux and vector-scale variants compiled with 124 registers.
All three had no spill, and shared-memory use was unchanged.

**Generated instructions.**

| Variant | SHFL | FMNMX | REDUX | LOP | IADD |
|---|---:|---:|---:|---:|---:|
| Baseline | 112 | 112 | 0 | 3427 | 3236 |
| Redux | 64 | 64 | 26 | 3370 | 3260 |
| Vector-scale | 64 | 64 | 26 | 3370 | 3260 |

The vector-scale source produced SASS identical to redux, so it did not create
an independent machine-code candidate or independent performance mechanism.
Redux reduced SHFL, FMNMX, LOP, and register counts, while IADD increased.

**Performance.** A continuous repeat-12 redux run had median 2685.22 tok/s. A
same-binary baseline run immediately afterward had median 2684.69 tok/s. The
difference is about +0.02%, which is inside session noise and not a useful
whole-run win.

The older 2721.82 tok/s baseline belongs to another time window. It must not be
subtracted from or used to reject this candidate; the decision uses only the
adjacent same-binary comparison.

**Decision.** **No-Go.** Do not retain a winner. Better-looking register and
instruction counts did not produce a measurable end-to-end improvement, and
the vector-scale spelling did not produce distinct code.

**Boundary.** This rejects the exact tested REDUX replacement and vector-scale
spelling on the current SM86 shape/binary. It does not establish that every
future REDUX use is ineffective; a different hot reduction or dependency chain
requires its own SASS and interleaved whole-run receipt.

**Rollback.** Leave the new laboratory flags disabled. Remove the experimental
flags/template to return the source to the established route if the candidate
is not retained. The supplied receipt did not include the exact flag names, so
none are invented here.

**Migration lesson.** Resource and SASS improvement is a necessary static
screen, not an admission result. When a candidate and control differ by less
than the run-to-run noise, keep the same-binary adjacent comparison and record
No-Go even if the generated instruction mix appears cleaner.

### E-20260830-14: sidecar-only complete FFN transaction

**Hypothesis.** Own the complete gated FFN boundary
`gate/up -> GELU multiply -> Q8 handoff -> down` as one backend transaction.
The gate/up kernel quantizes its exact register result directly into a private
MMA-ready sidecar. The down kernel consumes that sidecar and materializes only
the final public destination. This removes the dense gated-intermediate write
without changing the established down-projection accumulation schedule.

**Backend and ABI contract.** The optional backend trait method returns
`true` only when the final `dst` transaction has committed. A successful
call does not promise meaningful contents in `gated_tmp`. Returning `false`
promises that no public buffer was written and asks the workflow to execute the
established gate/up/down sequence.

The native entry point is a static-build laboratory surface. Static CUDA links
`imparo_cuda_ffn_gated_down_lab` directly, while the dynamic-plugin build
deliberately exposes only a local stub that returns `0`. The experiment
therefore does not add a symbol to the dynamic release table, change the stable
backend ABI version, or alter Metal/CPU behavior.

**Dense and sidecar ownership.** The selected gate/up instantiation uses
`WriteDense=false`. `G` is passed only to preserve the launch signature; the
compiler removes its address and stores. Gate/up results remain private in the
MMA-ready Q8 sidecar until down writes `dst`.

**Generated-code evidence.**

| Metric | Dense-writing | Sidecar-only | Delta |
|---|---:|---:|---:|
| Instructions | 4096 | 4072 | -24 |
| STG | 112 | 48 | -64 |
| IADD3 | 236 | 109 | -127 |
| LOP3 | 427 | 428 | +1 |
| SHFL | 112 | 112 | 0 |
| FMNMX | 112 | 112 | 0 |
| Registers | 126 | 124 | -2 |
| Stack | 0 | 0 | 0 |
| Local | 0 | 0 | 0 |

The primary static change is the removal of dense-output store and address
work: STG and IADD3 fall materially, while the SHFL/FMNMX reduction geometry is
unchanged. The sidecar-only route has zero stack and local memory in this
receipt. No shared-memory comparison was supplied, so none is inferred here.

**Eligibility and bounds.** The route is fail-closed to:

- explicit request, Q4_0 gate/up/down kinds, resident weights, SM86, and
  `8 < n_tok <= 512`;
- no active epilogue, graph capture, Prefill capture, or GPU probe;
- valid batch geometry whose token count equals `n_tok`;
- three distinct, allocated buffer identifiers within `B_COUNT`;
- non-zero dimensions, `n_in == n_out`, and
  `uint64_t(n_mid) == 4 * uint64_t(n_out)`;
- 128-aligned input, middle, and output widths;
- checked `uint64_t(n_tok) * width` fits in the source/destination buffer byte
  sizes;
- checked Q4 row-byte multiplication before every weight slice; and
- a logical-tile calculation performed in `uint64_t`, rejected on zero,
  `UINT32_MAX` overflow, or insufficient physical work.

All three packed-Q4 sidecars, both Q8 scratch regions, fixup workspace, and
dynamic shared-memory launch attributes are preflighted before a public output
can be enqueued.

**Commit-point bug and repair.** The original transaction treated a failed
down launcher as pre-commit even when the main down kernel had already been
enqueued and only the later fixup launch failed. Returning `false` in that
state could execute the byte-authoritative fallback after a partial public
`dst` write, violating the trait's all-or-fallback promise.

The down launcher now reports `public_output_committed`. It remains false
through argument/configuration checks and a failed main launch, and becomes
true immediately after the first public-output kernel passes launch checking.
If a later fixup launch fails after that point, the caller marks `dst`
written, records a fatal pending CUDA error, increments the committed trace
counter, and returns success so the workflow cannot enqueue the fallback.
Failures before the commit point still return false with public buffers
untouched.

**Route observability.** With
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_TRACE=1`, each non-Decode forward prints
`attempts`, `admitted`, and `committed`. A promotion receipt must have
`attempts >= admitted >= committed`, explain every difference, and show that
the expected eligible layers committed. The packed-Q4 and Q8 cache traces are
separate diagnostics and must not be enabled in timed performance legs.

**Flag matrix.**

| Purpose | Baseline B | Candidate I |
|---|---|---|
| Transaction selector | `IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB` unset | `IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB=1` |
| Sidecar capacity used by the current coverage policy | same setting in both legs; record exact value | same setting in both legs; `IMPARO_CUDA_Q4_PACKED_SIDECAR_MIB=1800` is the known full-coverage configuration |
| Transaction trace audit | off in timed legs | `IMPARO_CUDA_PREFILL_FFN_SIDECAR_TRACE=1` only in a separate route audit |
| Packed-sidecar trace audit | off in timed legs | `IMPARO_CUDA_Q4_PACKED_SIDECAR_TRACE=1` only in a separate memory audit |
| Q8 cache trace audit | off in timed legs | `IMPARO_CUDA_PROFILE_Q8=1` only in a separate ownership audit |
| Q8 cache opt-out | `IMPARO_CUDA_NO_Q8_CACHE` unset | `IMPARO_CUDA_NO_Q8_CACHE` unset unless an explicit miss-control leg is being run |
| Down grid override | `IMPARO_CUDA_PREFILL_DOWN_Q8_READY_STREAM_GRID_LAB` unset | unset unless a separately named grid experiment is being run |
| Profilers | off unless identical in both legs | `IMPARO_CUDA_PROFILE_FORWARD`, `IMPARO_CUDA_PROFILE_MATMUL`, and `IMPARO_CUDA_PROFILE_OPS` are audit-only unless explicitly shared |

The raw performance receipt must list every environment variable, including
variables explicitly unset, rather than relying on this expected isolation
matrix.

**Binary identity.**

- Source commit and exact dirty-diff hash: **pending main-task receipt**
- Candidate/control executable SHA-256: **pending main-task receipt**
- CUDA backend artifact SHA-256: **pending main-task receipt**
- Build profile, NVCC flags, CUDA toolkit, driver, device clocks/power:
  **pending main-task receipt**

The four legs may be interpreted as same-binary evidence only after those
fields prove that B and I differed solely by the transaction selector.

**449-token interleaved whole-run screen.**

| Leg | First | Second |
|---:|---:|---:|
| 1 | B 2729.02 tok/s | I 2722.91 tok/s |
| 2 | I 2728.69 tok/s | B 2667.42 tok/s |
| 3 | B 2697.22 tok/s | I 2717.47 tok/s |
| 4 | I 2738.96 tok/s | B 2699.79 tok/s |

Across all legs, the baseline median is 2698.505 tok/s and the candidate
median is 2725.800 tok/s:

`2725.800 / 2698.505 = 1.01011x`, approximately `+1.011%`.

The direction is promising, but the magnitude remains inside the observed
local-machine noise. It is not a final winner, is not default-enabled, and must
not be combined with the older cross-time 2721.82 result to manufacture a
larger claim.

**Numerical correctness evidence.** `logit_agree` passed for q4_0 at every
tested shape:

| Tokens | Top-1 | Overlap | Maximum delta | Exit |
|---:|---|---:|---:|---:|
| 128 | identical | 9/10 | 0.31426 | 0 |
| 449 | identical | 10/10 | 0.12315 | 0 |
| 512 | identical | 8/10 | 0.41576 | 0 |
| 2000 | identical | 9/10 | 0.41741 | 0 |

`decode_agree` at the 449-token start passed all eight recurrent steps. Every
step had identical top-1, the largest maximum delta across the eight steps was
0.64433, and the command exited 0.

`det_gate` with q4 reported `1/4 distinct` at 128, 449, 512, and 2000
tokens. Every shape reported `ALL PASS`, and the command exited 0.

These receipts close the listed logit, recurrent Decode, and determinism gates
for the tested q4 shapes. They do not close the transaction's failure-path or
memory/concurrency gates.

**Remaining evidence.** Compute Sanitizer memcheck, initcheck, racecheck, and
synccheck remain pending. Forced sidecar/scratch/configuration failures before
commit and an injected fixup failure after commit remain pending, as do the
source dirty-diff, executable, and CUDA backend artifact hashes. Byte-level
private-Q8/final-output evidence should be attached if it is required by the
promotion policy for this representation-exact route.

**Decision and rollback.** **Promising, non-final.** Keep
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB` unset by default. Unsetting it
restores the established workflow sequence. Dynamic plugins already return
false and follow that fallback.

**Migration lesson.** A multi-kernel fusion hook is a transaction, not merely a
shorter call sequence. Preflight every recoverable resource before the first
public write, expose the exact commit point from nested launchers, and make
post-commit failure forward-fatal. Compiler register savings and a positive
median are useful screens, but neither replaces failure injection, binary
identity, correctness, sanitizer, and noise-separated whole-run evidence.

### E-20260830-15: strict sidecar/llama.cpp cross-engine bracket

**Scope and controls.** This bracket used the same model, raw 449-token input,
`max1`, and Q4 KV for both engines. The llama.cpp reference was run with
`-fa on -ctxcp 0`. Each reported value is a repeat-12 median. Running the
reference engine on both sides of the Imparo leg provides a local drift
bracket; it does not establish cross-machine portability or binary identity.

| Order | Engine | Repeat-12 median |
|---:|---|---:|
| 1 | llama.cpp | 2253.53 tok/s |
| 2 | Imparo sidecar | 2717.32 tok/s |
| 3 | llama.cpp | 2231.89 tok/s |

The conservative comparison uses the faster reference run:

`2717.32 / 2253.53 = 1.20580x`.

The median of the two llama.cpp bracket runs is 2242.71 tok/s, giving:

`2717.32 / 2242.71 = 1.21162x`.

**Decision.** **Provisional current-machine/reference-binary bracket.** The
result shows greater than 1.2x against both the faster reference run and the
two-sided reference median under the listed runtime controls. It is not a
release or portable performance claim. Exact Imparo and llama.cpp executable
hashes, model hash, full command receipts, raw per-repeat values, and token
parity receipts remain pending. This cross-engine observation also does not
promote the sidecar route: its same-engine four-leg gain remains within local
noise, and failure injection plus all four Compute Sanitizer classes are still
pending.

### E-20260831-16: R32 two-resident-CTA direct-Q8 producer

**Identity and routes.** The control and candidate were selected from the same
SM86 release binary. Both used the complete sidecar-only FFN transaction and
the same dependency stack. The control set
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_R32_LAB=0` and therefore selected the R64
producer; the candidate set that variable to exactly `1`. All other route
variables were identical. Source and binary hashes remain unresolved because
the worktree is dirty; no identity value is inferred or reconstructed.

**Hypothesis.** Reducing row ownership from R64 to R32 while retaining 256
threads would reduce register and shared-memory use enough to improve residency
and hide Q4 load/MMA latency. An older draft incorrectly described the current
R64/T128 launcher as 512 threads; its production call is
`dim3(32, kWarps)` with `kWarps=8`, hence 256. The falsifier was a same-binary
interleaved whole-run median below the established R64 route.

**Invariants and rollback.** The R32 producer preserves the K32 accumulation
order and writes the same private MMA-ready Q8 layout. It is selected only on
the exact supported SM86 FFN shape. Unsupported configuration or a launch
failure before the public-output commit falls through to R64. The flag is
default-off, so unsetting it restores R64 without changing the public ABI,
dynamic DLL surface, Metal path, or workflow contract.

**Static resources.** The measured R32 instantiation uses 86 registers/thread,
zero stack, zero local memory, 27,648 bytes dynamic shared memory, and 256
threads/CTA. Its register footprint is approximately 22,016 registers/CTA, so
two CTAs per SM are theoretically possible on SM86. The R64 control uses 124
registers/thread, zero stack/local memory, 36,864 bytes dynamic shared memory,
and 256 threads/CTA, or approximately 31,744 logical registers/CTA before
allocation granularity. The old one-CTA inference was based on the wrong block
width and is withdrawn; exact occupancy must be re-established from the
measured binary and occupancy API or profiler. The 10.70% R32 No-Go result does
not depend on that correction.

**Correctness screen.** With Q4 KV and tolerance 0.75, `logit_agree.py` passed
at 128, 449, 512, and 2000 input tokens. The observed overlap/max-delta pairs
were 9/10 and 0.31426, 10/10 and 0.12315, 8/10 and 0.41576, and 9/10 and
0.41741. Since the performance hypothesis failed decisively, Decode,
determinism, sanitizer, and failure-injection promotion gates were not spent on
this candidate.

**Same-binary interleaved performance.** Raw 449-token, repeat-12, max1 legs
were run in R64/R32/R32/R64 order:

| Order | Route | Repeat-12 median (tok/s) |
|---:|---|---:|
| 1 | R64 control | 2629.94 |
| 2 | R32 candidate | 2372.68 |
| 3 | R32 candidate | 2390.41 |
| 4 | R64 control | 2703.68 |

The cross-leg medians are R64 `2666.810` and R32 `2381.545` tok/s. Therefore
R32/R64 is `0.89304x`, a 10.70% regression. The separation is much larger than
the local noise band and is consistent on both candidate legs.

**Decision and code state.** **No-Go.** Retain the candidate behind its exact,
default-off laboratory selector while the generated-code mechanism is still
useful for research; do not promote it or retry the identical R32 schedule.
Higher theoretical CTA occupancy was not the binding whole-run limit. The
extra tile phases, barriers, and shared-memory traffic outweighed its lower
register footprint.

**Next hypothesis / stop rule.** The next experiment must remove work or a
producer-consumer boundary rather than only increase occupancy. Prefer either
an exact-449 specialization that measurably removes tail waste, or a persistent
full-K down path that removes Stream-K seam/fixup work. Require an explicit
mechanism counter/event reduction and at least a noise-separated same-binary
whole-run win; otherwise mark that exact schedule No-Go rather than tuning its
occupancy blindly.

### E-20260831-17: exact-449 full-K down ownership screen

**Identity and controls.** One SM86 release binary selected all three routes.
The sidecar FFN dependency stack and every other environment value were held
constant. With `IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_449_LAB` unset, the
control used the established physical Stream-K grid 60. Exact values `r128`
and `r64` selected full-K logical grids 80 and 160 respectively. The selector
is further fail-closed to 449 tokens and the validated 2560 -> 10240 -> 2560
E4B FFN shape. Source/binary hashes remain unresolved because the worktree is
dirty.

**Hypothesis.** Assigning a complete K range to each logical tile would remove
Stream-K seam ownership, global partials, and the fixup launch. The prediction
required at least an 8% down-event reduction or a noise-separated 1% whole-run
gain. A slower down event falsified the mechanism directly.

**Resources and transaction.** The pre-existing R128 full-K instantiation uses
128 registers/thread, 36,864 bytes dynamic shared memory, zero stack/local
memory, 256 threads, and grid 80. The R64 version uses 72 registers/thread,
27,648 bytes dynamic shared memory, zero stack/local memory, 256 threads, and
grid 160. The explicit internal row-mode parameter prevents an inherited
`IMPARO_CUDA_PREFILL_Q8_READY_R64_LAB` value from silently changing the
measured route. A successful full-K launch is the public-output commit point;
preflight or launch rejection before that point returns to the conservative
workflow, while later asynchronous failure is forward-fatal.

**Correctness screen.** Both R128 and R64 passed the 449-token Q4-KV
`logit_agree.py` gate with 9/10 overlap and maximum distribution delta
0.34410 at tolerance 0.75. They produced the same observed top distribution.
Because the performance hypothesis failed, broader Decode, determinism, and
sanitizer promotion gates were not run.

**Whole-run evidence.** Repeat-12, raw 449-token, max1 measurements:

| Order | Route | Median (tok/s) |
|---:|---|---:|
| 1 | physical grid60 control | 2739.71 |
| 2 | full-K R128 grid80 | 2676.77 |
| 3 | full-K R64 grid160 | 2561.74 |
| 4 | physical grid60 control | 2724.83 |

The physical cross-leg median is 2732.27 tok/s. R128 is `0.97969x`
(-2.03%); R64 is `0.93758x` (-6.24%).

**Mechanism evidence.** With diagnostic CUDA-event synchronization enabled,
82 down events (41 layers x two requests) measured:

| Route | Sum (ms) | Mean (ms) | Min..max (ms) |
|---|---:|---:|---:|
| physical grid60 | 60.1880 | 0.7340 | 0.6441..0.7967 |
| full-K R128 | 68.6196 | 0.8368 | 0.6881..0.8885 |
| full-K R64 | 80.3769 | 0.9802 | 0.8817..1.0291 |

The candidate did remove the separate fixup launch, but the down operation
itself became 14.0% slower for R128 and 33.5% slower for R64. Complete-K tile
ownership reduced schedulable K parallelism more than it saved in seam/fixup
work.

**Decision and rollback.** **No-Go.** Keep the two routes behind the exact,
default-off selector for reproducibility, but do not promote or sweep the same
80/160 schedules. Unsetting the selector restores physical grid60.

**Next hypothesis / stop rule.** Preserve physical K parallelism while removing
its repeated runtime planning and tail waste: precompute the exact-449 worker,
segment, seam, and fixup plan, then separate full384 from tail65 only if SASS
and event counters prove less 64-bit division, inactive work, or fixup scanning.
Reject the candidate at compile/profile time if the generated schedule is
identical or its FFN event improves less than 5% and whole Prefill less than 1%.

### E-20260831-18: fresh cross-engine variability bracket

**Scope.** After rebuilding the same dirty SM86 tree with all new experiments
default-off, a raw 449-token, Q4-KV, max1, repeat-12 bracket used llama.cpp
`-fa on -ctxcp 0`. The route environment otherwise matched the sidecar
dependency stack. GPU state was queried only after the runs and was idle P8,
50 C, 210-MHz SM, driver 596.08; those values are not load-time clock/power
evidence.

| Order | Engine | Median (tok/s) |
|---:|---|---:|
| 1 | llama.cpp | 2288.53 |
| 2 | Imparo | 2647.91 |
| 3 | llama.cpp | 2265.61 |
| 4 | immediate Imparo repeat | 2695.94 |

The first Imparo leg is `1.15704x` against the faster llama leg and
`1.16286x` against the two-sided llama median 2277.07. The immediate Imparo
repeat is `1.17802x` and `1.18395x` respectively.

**Decision.** **Observation.** The previous E-15 result above 1.2x remains a
real receipt for that session, but it is not stable under this fresh bracket.
Do not state that the current route reproducibly exceeds 1.2x. Future
cross-engine promotion requires load-time clocks/power/temperature, binary and
model hashes, token parity, and multiple interleaved legs. Optimization
experiments should continue using same-binary A/B so clock drift cannot create
a false winner.

A later protocol audit found that the old runner discarded only one warmup,
inherited the complete parent environment, and did not capture load-time GPU
telemetry. One inspected llama process reached 2130.96, 2153.15, then 2287.35
tok/s for its first three 449-token requests, so a single discarded request is
not a sufficient steady-state definition. The performance-target fork and the
upstream-base correctness oracle must also be named separately. See
`docs/cuda-sm86-benchmark-protocol.md` for the replacement allowlist, idle,
telemetry, eight-leg, dispersion, and confidence-bound rules.

**Next hypothesis / stop rule.** Prefer changes with a mechanism-level CUDA
event win large enough to survive the observed 2-4% whole-run drift. Re-run the
strict reference bracket only after a candidate wins same-binary interleaving;
otherwise do not spend reference runs on it.

### E-20260831-19: exact-449 AOT Stream-K cycle plan

**Identity and selector.** This candidate is based on E-20260831-17's physical
R128/grid60 down route and is compiled into the same dirty release binary as the
generic control. It is admitted only for the exact N=449, K=10240, M=2560,
R128/grid60 shape by
`IMPARO_CUDA_PREFILL_FFN_DOWN_STREAMK_AOT_449_LAB=1`. Unsetting the variable
restores the generic planner. It does not change the public ABI, workflow, or
Metal path.

Measurement-time SHA-256 identities were:

- `imparo-server.exe`: `296fe8a2f2595c9f75686bed3e63e382e6ab379a40b218124766a1a44f2a114e`;
- `imparo-forward.exe`: `b595009368777aca84becfc6dea92c181c11a594b9fd767403d8322ef1cd8430`;
- measured `imparo_cuda.obj`: `19210f99b7fd0874fd14ebb69c6b07ceccd2d587cd1454593fa3119f16420b13`;
- exact header: `40494e0fc9e77f34f653ad3303748ede82949d891a1bfe6de0d551987445e1a9`;
- CUDA translation unit: `4ec597a522730ccd2fbf82c89663e6354152ef0d3dfa21116fc3523da1b81129`;
- exact static contract: `11ca0549611202f927a348afc541780e29633c7ab4c21bd057b39766b699f81f`;
- measurement-time `raw_bracket.py`: `2e23d1c41ab59c1571cf9c86f27ff36abc0ca0c5d365fcca57a5b7ce3a8ce8a7`.

The worktree was dirty and contains untracked experiment files; these hashes
bind the measured artifacts but do not replace a complete dirty-tree manifest.

**Hypothesis.** Preserve the established physical Stream-K K parallelism while
removing per-launch boundary planning, 64-bit division/modulo, empty compact
fixup work, and the previous-prefix scan. The exact three-worker cycle owns the
same K32 sequence and 40 seams; its fixup launch is 40x4=160 CTAs instead of
60x4=240.

**Compile and resource evidence.** An initial helper/template implementation
compiled with 128 bytes of stack per thread. Reusing reference arrays did not
remove it. Rewriting the exact path as one kernel with a fixed two-pass,
three-phase cycle plan produced:

| Kernel | Registers | Stack/local | Dynamic shared | Generated-code note |
|---|---:|---:|---:|---|
| generic physical main | 128 | 0/0 | 36,864 B | 2238 static instructions, CALL7, MUFU8 |
| exact-449 main | 128 | 0/0 | 36,864 B | 3712 static instructions, CALL2, MUFU0, LDL0/STL0 |
| exact-449 compact fixup | 16 | 0/0 | 0 B | no previous/empty scan |

The final SM86 release build exited 0 in 1m40s. The exact and FFN transaction
static contract tests both passed 3/3. Static instruction counts describe code
size and cannot be interpreted as a performance win.

**Correctness evidence.** The q4_0 llama-reference gate at 449 tokens exited 0:
top-10 overlap was 10/10 and the maximum relative-logit delta difference was
0.12315 at tolerance 0.75. Broader Decode, determinism, sanitizer, and failure
injection were intentionally not run after the performance stop rule fired.

**Whole-run evidence.** A repeat-12, raw 449-token, max1 same-binary screen used
generic/AOT/AOT/generic order:

| Order | Route | Median (tok/s) |
|---:|---|---:|
| 1 | generic physical grid60 | 2682.36 |
| 2 | exact-449 AOT | 2679.68 |
| 3 | exact-449 AOT | 2679.73 |
| 4 | generic physical grid60 | 2659.49 |

The generic cross-leg median is 2670.925 tok/s and the AOT cross-leg median is
2679.705 tok/s, a ratio of `1.00329x` (+0.33%). This is below the predeclared
1% whole-Prefill continuation threshold and inside the observed local drift.

**Decision and rollback.** **No-Go.** The candidate is numerically correct,
spill-free, and useful as a reproducible exact schedule, but does not remove
enough work to justify promotion or the remaining sanitizer cost. Keep it
laboratory-only and default-off. Unset the exact selector to roll back.

**Migration experience.** A compile-time schedule can remove division and
empty fixup CTAs without materially improving the whole transaction. Template
abstraction can also create hidden per-thread stack even when local memory is
reported as zero; resource inspection must precede GPU benchmarking. Do not
repeat schedule-only exact specialization unless a mechanism counter predicts
more than the measured noise floor.

**Next hypothesis / stop rule.** Move from schedule reshaping to removed data
transformation: test exact W4A8 digit slicing using native SM86 INT4 MMA so Q4
weights need not expand to s8. First require bit-exact K32 fragment oracles,
zero stack/local/spill, and at least 1.10x consumer-only or 1.05x honest
producer-plus-consumer improvement before any engine selector is added.

### E-20260831-20: exact W4A8 digit-sliced SM86 MMA

**Identity and boundary.** This Gate-A laboratory candidate exists only in
`native/sm86/mmq_q4_q8_digit_sliced_lab.cuh` and
`native/tests/sm86_w4a8_digit_lab.cu`. It has no production launcher, selector,
ABI export, workflow dependency, or Metal impact. The executable is compiled
standalone for SM86.

Final SHA-256 identities are header
`569591a277eb753b3d26a3b54bf01eefe9fa76cb7af99129e87abc1141dbe058`,
test translation unit
`9095ce23cbf49cbeb45f41c18a01f9ca5da17ef9f3633d6eb31d4716607d70eb`,
and standalone executable
`9dcbfd863c9d1a8427a74b89a87ab160507cb24e611a69618f2823362caaa574`.

**Hypothesis.** Preserve the exact Q4_0 and Q8 integer dot using:

```text
w_s4 = q4_storage_nibble xor 8
q8   = low_u4 + 16 * high_s4
dot  = MMA.s4.u4(w, low) + 16 * MMA.s4.s4(w, high)
```

The two integer results are combined before the existing per-K32 `d4*d8`
scale and FP32 accumulation. The potential benefit was not "two times faster
INT4": two INT4 MMA instructions replace one INT8 MMA. The candidate instead
needed removed Q4-to-s8 expansion, half-size weight fragments, and lower shared
traffic to repay that instruction cost.

**Correctness and generated code.** CPU exhaustive testing covered all 16 Q4
storage nibbles by all 256 signed Q8 values: 4096/4096 passed. On the RTX 3060
SM86, three 16x8x32 cases produced 384 outputs with zero control/candidate
integer mismatch, zero scaled float bit mismatch, and maximum absolute error
zero. SASS contained the required `IMMA.16832.S4.U4` and
`IMMA.16832.S4.S4`; the control contained only `S8.S8`.

| Kernel | Registers | Stack/local/spill | MMA instructions |
|---|---:|---:|---|
| consumer S8 control | 28 | 0/0/0 | one S8xS8 |
| consumer W4A8 | 30 | 0/0/0 | one S4xU4 plus one S4xS4 |
| cached raw-input S8 control | 53 | 0/0/0 | one S8xS8 |
| cached raw-input W4A8 | 62 | 0/0/0 | one S4xU4 plus one S4xS4 |

All four kernels had zero LDL, STL, and CALL. The cached raw-input loads and
MMA instructions remained inside the loop and were not hoisted.

**Harness failure retained.** The first honest-input harness used volatile
byte loads to prevent hoisting. It generated 24 strong-system U8 loads per
iteration and measured about 152.5 ms versus about 0.3 ms for the consumer
core. Its `0.999997x` ratio was a load-serialization result, not an algorithm
result, and is classified as a harness No-Go.

The corrected harness used ordinary cached loads and a 256-tile, 128-KiB
rotating raw-input pool. The tile index depended on global warp and iteration;
control and candidate read the same packed Q4 and Q8 bytes each iteration.
Both routes were bit-exact. CUDA events used the balanced
`ABBA BAAB` schedule for nine cycles:

| Gate | S8 control median/MAD (us) | W4A8 median/MAD (us) | Control / W4A8 |
|---|---:|---:|---:|
| prepared-fragment consumer | 268.287994 / 1.023987 | 357.359985 / 1.024002 | `0.750750x` |
| cached raw-input operator | 2611.200195 / 5.119629 | 2784.768066 / 3.584229 | `0.937672x` |

**Decision and rollback.** **No-Go.** Numerical equivalence and clean resources
are established, but neither the mechanism upper bound nor the corrected
raw-input operator passes its predeclared 1.10x/1.05x continuation threshold.
Do not add an engine selector or compile the lab into production. Rollback is
structural: production does not include or call these standalone files.

**Migration experience.** Instruction support is not a speed result. On this
SM86 shape, two sub-byte MMA instructions plus integer combination were 33.2%
slower than one INT8 MMA in the prepared-fragment core, and the compact
representation recovered only part of that deficit. Also, volatile loads are
not a safe anti-hoist technique for a performance oracle; use changing legal
addresses and verify the loop in SASS.

**Next hypothesis / stop rule.** Preserve the single S8xS8 MMA while testing
packed-Q4 shared staging: expand one A fragment in registers and reuse it across
all applicable token fragments. Continue only if source analysis proves that
the saved shared traffic exceeds repeated expansion, then require a bit-exact
standalone operator improvement of at least 1.05x before engine integration.

### E-20260831-21: packed-shared single-S8 SM86 MMA

**Identity and boundary.** This Gate-A experiment is isolated to the two
untracked laboratory files:

- `crates/imparo-cuda/native/sm86/mmq_q4_q8_packed_shared_s8_lab.cuh`
- `crates/imparo-cuda/native/tests/sm86_packed_shared_s8_lab.cu`

There is no production launcher, selector, ABI export, workflow dependency,
build-script entry, or Metal change. The exact final identities are:

| Artifact | SHA-256 |
|---|---|
| laboratory header | `59949e5b0990e0587f3b599ec47141c5cd2d0bbfe360174351c336eface06e39` |
| test translation unit | `f432ac6e1751a795356447b0f124a8ed64544b5f2f5fedaa6967713709822c37` |
| standalone executable | `3b8680eadb7e1238b1e0dd73ca1c0c1090ff727b7a8b88b87601ef507afa774c` |

The measured executable was built only for SM86:

```powershell
cmd.exe /d /s /c '"%IMPARO_VCVARS64%" >nul && "%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 -arch=sm_86 -lineinfo --ptxas-options=-v crates\imparo-cuda\native\tests\sm86_packed_shared_s8_lab.cu -o target\sm86_packed_shared_s8_lab.exe'
```

It exited 0. The two routes are in the same binary and use the same stream,
raw input bytes, scale arrays, output layout, grid, K32 scale/FMA order, tail
policy, and final write.

**Hypothesis and layout.** The control expands packed Q4 weights into shared
signed bytes before issuing one `m16n8k32 S8xS8` MMA per token fragment. The
candidate keeps qblock-major packed words in shared, expands an A fragment once
into registers, and reuses it across eight token fragments while preserving the
same single S8 MMA. Raw weights use
`[projection][qblock][row][4 words]`; f32 weight and activation scales remain in
shared memory. This tested whether removed expanded-weight shared traffic was
large enough to pay for register unpack and candidate bookkeeping without
changing the core arithmetic.

**Correctness and occupancy.** The CPU-only command was:

```powershell
& 'target\sm86_packed_shared_s8_lab.exe' --cpu-only
```

It exited 0 with 128 Q4 layout cases and zero mismatches. After the final
four-copy timing harness rebuild, the GPU command was:

```powershell
& 'target\sm86_packed_shared_s8_lab.exe' --gpu-oracle
```

It exited 0 on the RTX 3060 Laptop GPU at SM86. The shape was two projections,
three qblocks, 32 rows, and 65 tokens. All 4160 outputs were bit-exact:
control-versus-CPU, candidate-versus-CPU, and control-versus-candidate each had
zero mismatch. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reported 16
active CTAs/SM for both routes, above the predeclared two-CTA floor.

**Resources and generated code.** Static resources from the final executable
were collected with:

```powershell
& (Join-Path $env:CUDA_PATH 'bin\cuobjdump.exe') --dump-resource-usage 'target\sm86_packed_shared_s8_lab.exe'
```

The command exited 0:

| Route | Registers | Shared bytes | Stack | Local | Spill stores/loads |
|---|---:|---:|---:|---:|---:|
| expanded-shared control | 107 | 2880 | 0 | 0 | 0/0 |
| packed-shared candidate | 115 | 2624 | 0 | 0 | 0/0 |

`cuobjdump --dump-sass` showed eight `IMMA.16832.S8.S8` instructions in each
route and no LDL or STL. Thus the candidate reduced static shared allocation by
256 bytes but did not reduce the number of core MMA instructions.

**Memory and synchronization checks.** Before the host-only four-copy timing
harness change, the isolated GPU oracle was run serially under:

```powershell
& (Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe') --tool memcheck  --error-exitcode 99 'target\sm86_packed_shared_s8_lab.exe' --gpu-oracle
& (Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe') --tool initcheck --error-exitcode 99 'target\sm86_packed_shared_s8_lab.exe' --gpu-oracle
& (Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe') --tool racecheck --error-exitcode 99 'target\sm86_packed_shared_s8_lab.exe' --gpu-oracle
& (Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe') --tool synccheck --error-exitcode 99 'target\sm86_packed_shared_s8_lab.exe' --gpu-oracle
```

All four exited 0. Memcheck, initcheck, and synccheck each reported
`ERROR SUMMARY: 0 errors`; racecheck reported
`0 hazards displayed (0 errors, 0 warnings)`. Because the executable identity
changed when four independent input allocations were added, retain these as
pre-rebuild kernel evidence rather than claiming they are a sanitizer receipt
for the final executable hash. The final executable's ordinary GPU oracle did
pass as recorded above; the No-Go stop rule made another sanitizer sweep
unnecessary.

**Profiler limitation.** The first and only requested bank-conflict capture
used one target-kernel launch and the metrics
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum`, matching shared
wavefront metrics, shared load/store request metrics, and LDSM count. The exact
command began:

```powershell
& $env:IMPARO_NCU --target-processes all --kernel-name 'regex:.*expanded_shared_control.*' --launch-count 1 --metrics 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__inst_executed_op_ldsm.sum' --export 'target\ncu_packed_s8_control_n449' --force-overwrite 'target\sm86_packed_shared_s8_lab.exe' --profile control --tokens 449
```

It exited 1 with `ERR_NVGPUCTRPERM`: the account lacked permission to access
NVIDIA GPU performance counters. No bank metric or report was produced, and
the remaining three requested captures were correctly not run. Bank replay is
therefore **pending**, not zero.

**Four-copy performance harness.** Four address-independent, bit-identical
device input copies were allocated before timing; there was no timed-leg copy.
Within each eight-leg `ABBABAAB` cycle, adjacent route pairs were
`AB, BA, BA, AB`. Pair offsets were `0,1,2,3`, and the copy was selected by
`(cycle + floor(leg/2)) % 4`, so both routes in every adjacent pair used the
same bytes and address set. Warmups used `warmup % 4`, A then B. CUDA events
measured nine cycles on one nonblocking stream:

```powershell
& 'target\sm86_packed_shared_s8_lab.exe' --benchmark --warmup 4 --cycles 9
```

The command exited 0 after repeating the zero-mismatch GPU oracle and then
validating the two performance shapes bit-for-bit. Both shapes used 256 CTAs
with grid `[16,8,2]`.

| Tokens | Bytes/copy; four-copy total | Control median/MAD (us) | Candidate median/MAD (us) | Control/candidate |
|---:|---:|---:|---:|---:|
| 449 | 2,112,320; 8,449,280 | 701.439941 / 97.376007 | 707.536011 / 102.447998 | `0.991384x` |
| 512 | 2,293,760; 9,175,040 | 804.864014 / 24.000000 | 828.351990 / 26.623993 | `0.971645x` |

For N=449, the first four-cycle median ratio was `0.991384x` and the last
four-cycle ratio was `0.999293x`. For N=512 they were `0.970948x` and
`0.968477x`. A paired analysis aggregated the four control and four candidate
legs within each cycle, then applied a t(8) interval to the nine log ratios.
The geometric control/candidate ratio and paired 95% interval were
`0.989901x [0.980784, 0.999103]` at N=449 and
`0.987180x [0.974218, 1.000314]` at N=512.

The raw legs exposed a strong position/cache effect. At N=449, first-in-pair
legs were 794.624--809.984 us while second-in-pair legs were
535.488--611.232 us. At N=512 those ranges were 822.272--855.040 us and
686.080--805.760 us. Balanced AB/BA ordering gave each route equal exposure,
but the resulting bimodal samples make an unstratified median fragile. The
paired cycle analysis agrees with the headline result: there is no hidden
candidate win.

**Independent statistical correction; no rerun.** A read-only recomputation
from the retained raw legs verified that the implemented route sequence is
`A B B A B A A B`, the adjacent pairs are `AB, BA, BA, AB`, and pair pool
offsets are `0, 1, 2, 3`. Both routes therefore visit every pool exactly once
and occupy two first-access plus two second-access positions in every cycle.
No route-assignment or paired-pool bug was found.

The primary calculation above is exactly reproducible by forming, for each
cycle, `r = sum(four control legs) / sum(four candidate legs)`, then applying a
two-sided t(8) interval to the nine `log(r)` values. It gives
`0.989901120x [0.980783895, 0.999103098]` at N=449 and
`0.987179715x [0.974217993, 1.000313889]` at N=512. A second calculation that
first takes the geometric mean of the four matched-pair ratios in each cycle
gives `0.990462744x [0.979458302, 1.001590823]` and
`0.989155411x [0.975666319, 1.002830996]`, respectively. The small estimator
difference cannot reverse the 1.05x stop decision.

The previously reported front/back values flatten two cache modes before
taking route medians. Retain them as reproducible receipt fields, but do not use
them as the stability definition. With front defined as the geometric mean of
the balanced cycle-sum ratios for cycles 0--3 and back as the same statistic
for cycles 5--8, excluding the center cycle, the corrected values are
`0.994531212x` front versus `0.987880589x` back at N=449 (-0.6687%), and
`0.986484883x` versus `0.986720784x` at N=512 (+0.0239%). Nine cycles leave a
5/4 pool-by-access-order imbalance; restricting this receipt to its first eight
fully balanced cycles still gives only `0.991584154x` at N=449 and
`0.986613370x` at N=512.

Do not modify or selectively rerun the hashed No-Go laboratory artifact to
rewrite this result. Future rotating-pool standalone brackets must use a cycle
count divisible by four (12 by default), emit the per-cycle sums, log ratios,
and confidence interval directly, and report first-access/cold and
second-access/hot strata separately. Front/back must be computed from complete
balanced cycles, not from a flattened bimodal sample.

**Decision and rollback.** **No-Go.** The candidate passed exact numerical,
resource, occupancy, and preliminary sanitizer gates, but it missed the
predeclared 1.05x continuation threshold at both tail and full-token shapes.
Do not add a selector, build entry, production ABI, workflow path, or broader
SM specialization. Rollback is deletion of the two untracked laboratory files
listed above; production already excludes them. The standalone target binary
is not a production artifact.

**Migration experience and next hypothesis.** Reducing shared allocation and
moving unpack from shared bytes to a reusable register fragment did not remove
the dominant instruction sequence: both routes still issued eight S8 MMA
instructions and the candidate used eight more registers. Do not continue
enumerating shared-layout variants without a bank-counter mechanism and a
larger arithmetic upper bound. The next isolated route should remove core
arithmetic or a complete producer-consumer transaction--for example fewer MMA
or scale/FMA operations, fewer K32 passes, or a provably eliminated conversion
boundary--rather than merely relocating the same work between shared memory
and registers. It must again clear a predeclared standalone threshold before
any engine integration.

### E-20260831-22: W4A4 granularity CPU Gate0

**Identity and boundary.** This is a pure-CPU mathematical screen, isolated to
two untracked laboratory files:

- `crates/imparo-cuda/native/sm86/mmq_q4_a4_granularity_lab.cuh`
- `crates/imparo-cuda/native/tests/sm86_w4a4_granularity_lab.cu`

It defines no GPU kernel, production launcher, selector, build-script entry,
ABI, workflow, or Metal change. No GPU or server was run. The exact final
identities are:

| Artifact | SHA-256 |
|---|---|
| CPU math/reader header | `64060265772a94f203cbeb34f1b47e531e1614713845fb244c165aa8a76b4481` |
| CPU test translation unit | `7c842aaa30ddc02cf86dfc441d36dfe00a9f75c25bd16e801453e139c700ce91` |
| host-only executable | `54a14a498e589869a7be87d9872ce8b659a7e7a805d927bb86290dfa5d451ee6` |

The host-only build used MSVC, not nvcc:

```powershell
cmd.exe /d /s /c '"%IMPARO_VCVARS64%" >nul && cl.exe /nologo /std:c++17 /O2 /EHsc /W4 /permissive- /TP crates\imparo-cuda\native\tests\sm86_w4a4_granularity_lab.cu /Fe:target\sm86_w4a4_granularity_lab.exe'
```

It exited 0 without a compiler warning.

**Fixed mathematics.** Q4_0 remains the authority at every K32 boundary:
signed code `q=storage-8`, real value `d_j*q`, low nibbles for indices 0--15,
and high nibbles for 16--31. For G64 or G128, `r_w` is that unchanged signed
authority code; it is never trained, re-quantized, or made activation-dependent.
The common weight scale is the closed-form least-squares solution:

```text
S2_j = sum_i q_ji^2
dG   = sum_j(d_j * S2_j) / sum_j S2_j
```

The analytic test used two K32 blocks with `q=1,d=2` and `q=2,d=4`, giving
`dG=3.6` and `sum_q=96`. The implementation returned `3.5999999`; changing
every activation byte and scale left `dG`, `r_w`, and `sum_q` unchanged.

Each group's A4 representation is derived independently from dequantized Q8
authority values using the fixed deployable unsigned-affine rule
`d_a=(max-min)/15`, `z=clamp(round(-min/d_a),0,15)`, and
`r_a=clamp(round(x/d_a)+z,0,15)`. Centered rounding is nearest with ties away
from zero; double-domain bounds are checked before any conversion to `int`.
The all-zero group is the sole zero-scale special case. There is no iterative
fitter, seed search, or runtime coordinate descent. The integer dot is
corrected before scaling:

This exact fixed estimator is intentionally conservative rather than globally
optimal: a one-sided nonconstant group can clip an endpoint after `z` is
clamped, and a constant nonzero group falls back to exact. That can create a
false No-Go or reduce approximate coverage; it cannot create an unsafe
admission. Any future estimator change requires a new version and fresh screen.

```text
sum(r_w * (r_a-z)) = sum(r_w*r_a) - z*sum_q
```

The ladder tries G128, then two G64 groups, then assigns the exact K32-ordered
Q4-by-Q8 authority result. Per-group synthetic selection thresholds are weight
NRMSE at most 0.04 and activation NRMSE at most 0.12. These thresholds select a
screening representation; they are not a real-capture success claim.

**Invariant self-test.** The canonical invariant command is:

```powershell
& 'target\sm86_w4a4_granularity_lab.exe' --self-test
```

It exited 0. Evidence was Q4 layout 512/512, 4096 exhaustive same-byte
other-nibble preservation cases, scalar zero-point correction 4096/4096, 768
random G64/G128 segment identities, forced ladder order G128-to-G64-to-exact,
and bit-exact exact fallback. The fixed min/max path and finite/range suite
passed 14 cases, including all-zero and constant-nonzero degeneracies,
ties-away rounding, finite values far outside `int`, signed `begin/count`
overflow, unrepresentable A4 scale, `dG*dA` overflow, exact scale/FMA
overflow, NaN rejection, safe exact fallback, and a normal finite chain.
Statistics passed 6 positive/negative API cases: length mismatch, empty input,
NaN, Inf, and accumulation overflow are rejected before indexing or sorting.

The versioned capture envelope passed a known SHA-256 vector, one positive
envelope, the original 17 negative cases, and six added negative cases (23
negative total). The added cases cover null input, `TooLarge`, inactive
nonzero dimensions, record-count-by-record-bytes overflow, a still-valid Q4/Q8
section-length swap, and a still-valid `2x3` to `1x6` reshape.

The capture v1 envelope is explicitly opaque. Its SHA-256 identity covers the
canonical 128-byte header with digest bytes 96--127 zeroed, followed by the
payload. Therefore the external expected identity and embedded identity bind
version, schema, rank/shape, record and section lengths, and payload together.
The parser rejects null input. The reader also validates little-endian encoding,
checked arithmetic, exact file size, and a 64 MiB provisional buffered-reader
cap; allocation failure is caught. The real schema reader must stream records
rather than raise this cap. A valid envelope still returns a non-success
analysis status until a real record producer and schema are audited. No real
capture was invented or evaluated.

**First synthetic failure retained.** Before the modes were separated and
before the deployability review, the fixed 384-record screen was run as:

```powershell
& 'target\sm86_w4a4_granularity_lab.exe' --synthetic-only
```

That run used the removed 16-zero-point, multi-seed iterative activation
fitter. It is an **un-deployable fitting upper bound**, not an estimate of the
fixed min/max runtime candidate, and it still exited 1. All invariants passed,
but an extra predeclared normalized-p99
sanity ceiling of 2.0 failed at `2.08351139`; the G64 subset was `2.76244113`.
That statistic divides by `max(abs(reference), 0.01*reference_rms)` and is
strongly amplified by near-zero dot products. The failure is retained rather
than loosened or rewritten as a pass.

The same first run produced exactly 128 G128, 128 G64, and 128 exact fallbacks:
approximate coverage `0.666666667`, relative L2/NRMSE `0.0380160556`, cosine
`0.999278740`, p99 absolute error `0.210865617`, zero correction mismatches 0,
and exact-fallback bit mismatches 0. G128 NRMSE was `0.0629799179`; G64 NRMSE
was `0.0616789613`.

The CLI is split into three explicit modes:

1. `--self-test` proves only invariants and must exit 0.
2. `--synthetic-screen` uses the fixed seed and current fixed min/max estimator,
   but can only report `continue_to_real_capture`; it can never output a Gate
   `go` decision or exit 0. It was not run after the min/max replacement.
3. `--capture PATH --expect-sha256 HEX` validates a real envelope but refuses
   analysis while the record schema remains unbound.

For a future real capture, the provisional hard Gate0 thresholds are cosine at
least 0.999, relative L2 at most 3%, and approximate G128-plus-G64 coverage at
least 70%. Before analysis is enabled, the schema must stratify by layer,
shape, and ladder level, aggregate by work, and add a worst-layer bound. The
retained old-fitter synthetic observation meets cosine but misses relative L2
(`3.80161%`) and coverage (`66.6667%`), so it is a **screen No-Go** independently
of the original p99 failure. The near-zero-sensitive statistic is now named
`p99_floor_relative_v2`, includes its denominator in output, and is descriptive
only; this versioning does not retroactively turn the first run into a pass.

**Decision and rollback.** **Synthetic screen No-Go; real Gate0 pending.** The
CPU invariants establish a reproducible mathematical and reader skeleton, not
W4A4 admission or performance. The fixed min/max synthetic screen has not been
run and the real schema remains unbound. Do not write a GPU kernel or connect
production until an audited, stratified real capture meets all registered
thresholds. Rollback is deletion of the two untracked laboratory files;
production never compiles them.

### E-20260831-23: real E4B activation pre-screen for W4A4

**Scope and identity.** This experiment closes the activation half of the first
real Gate0 question without pretending that activations alone prove W4A4. The
non-default `cuda-gate0-capture` binary captured complete f32
`ffn_norm_input` and `ffn_down_input` tensors for layers 0, 21, and 41 from a
cold 449-token E4B/SM86 forward. The sealed payload is 68,966,400 bytes. The
capture remains timing- and promotion-inadmissible and no CUDA W4A4 kernel,
selector, ABI, workflow hook, or Metal path was added.

| Artifact | SHA-256 |
|---|---|
| capture-only `imparo-forward.exe` | `b2679621255ade00ce42773ee575c3cc23885dfea702ae45a29c4f54f54adea7` |
| `run.manifest.json` | `5470cdd77fb4cbb72373e8825d42acb90d233c4e11ea54e050c481e6299ada71` |
| `run.complete.json` | `7b19f6c2d896e3f477f14f55c7321e2822c28afb30a7f48c03df22f808d0cba6` |
| activation pre-screen | `7854c4f4493e758a74a853b2610a811cde5d61d88bbf5fe09411094f1f236d93` |
| pre-screen tests | `a74cd0ee8951c943f031bc773b3be9d96f45329ba163992b97a8c9a92f3ff5aa` |
| raw JSON report | `4e1ef23a3d87f31d8e94bc04f70481bfd3ddb832fd942bf9d6599d497b10f214` |

The capture-on and capture-off runs used the same binary, request, Q4 KV route,
model, and production CUDA defaults. Their complete 262,144-value logits dumps
were byte-identical at SHA-256
`87174d2a2566fc02fe33d3c314b27adafbc7994fe5406731acc55830ba21d728`.
The capture module's 15 focused unit tests passed before the run. The new
pre-screen's six tests also pass; they cover ties-away rounding, K32-local Q8
authority, zero and constant groups, ladder accounting, and duplicate JSON
keys.

**Audited activation mathematics.** `dev_harness/w4a4_activation_gate0.py`
validates the completion seal, canonical run identity, ordered record set,
manifest/tensor hashes, safe member names, tensor header, exact lengths, shapes,
and finite values before mapping a payload. It reproduces the CUDA MMQ Q8 order
`d_inv=127/amax`, `q=roundf(x*d_inv)`, `d=f16(1/d_inv)`, then applies the fixed
unsigned-affine A4 estimator and the registered G128-to-G64-to-exact ladder.
The output is mechanically marked `is_gate=false`: no Q4 weight scales or
projection outputs are interpreted.

**Result.** The fixed ladder fails before weight-correlated analysis. Across
134,700 real activation groups, approximate coverage is only `18.4202%`, far
below the registered 70% floor. Worst-stratum relative L2 is `8.9171%` against
3%, and worst cosine is `0.996048959` against 0.999. Per-stratum coverage is:

| Layer | Operation | G128/G64 approximate coverage | relative L2 | cosine |
|---:|---|---:|---:|---:|
| 0 | ffn_norm_input | 75.7795% | 8.9171% | 0.996048959 |
| 0 | ffn_down_input | 12.1158% | 5.6005% | 0.998439915 |
| 21 | ffn_norm_input | 51.9265% | 7.2754% | 0.997364278 |
| 21 | ffn_down_input | 12.0657% | 3.4897% | 0.999391750 |
| 41 | ffn_norm_input | 21.3697% | 4.1109% | 0.999156263 |
| 41 | ffn_down_input | 7.6253% | 4.2645% | 0.999091048 |

Because adding real Q4 weight pooling can only reject more groups, the final
fixed-rule coverage is upper-bounded by `18.4202%`; it cannot recover to 70%.
The tool therefore exits 1 with
`decision=stop_before_weight_correlated_gate0`. This is an intentional No-Go,
not a harness failure.

**Decision and migration experience.** **Fixed G128/G64 W4A4 ladder No-Go for
real 449-token E4B.** Do not implement its SM86 MMA kernel or tune its grids.
The synthetic fitter's 66.7% coverage substantially overestimated deployable
coverage, especially for post-GELU/multiply Down inputs. The next admissible
hypothesis must change the representation rather than loosen thresholds: for
example, a bounded sparse-outlier or mixed-precision residual whose real-data
coverage, correction density, arithmetic upper bound, and final distribution
are measured before GPU implementation. The exact Q4-by-Q8 route remains the
safe fallback and production default.

### E-20260831-24: packed-Q4 DP4A gate/up screen

**Hypothesis and scope.** The existing default-off SM86 laboratory kernel in
`native/sm86/mmq_q4_q8_dp4a.cuh` keeps Q4 nibbles packed, uses DP4A with the
exact `-8*sum(q8)` correction, and targets only the 449-token
`2560 -> 10240` Gate/Up projections. This is a llama-inspired implementation
comparison, not W4A4: the Q4-by-Q8 numerical contract is unchanged. No selector,
ABI, tuner, Metal path, or production default was changed.

**Mechanism and correctness.** `IMPARO_CUDA_MMQ_PACKED_DP4A_LAB=1` produced
84 `packed-dp4a-lab` route hits (42 Gate and 42 Up). The canonical 449-token,
q4_0 `logit_agree.py` gate passed with top-1 equal, 9/10 top-id overlap, and
maximum distribution delta `0.16664` at the registered `0.75` limit.

**Performance screen.** A diagnostic-only CUDA-event pass used the same
CUDA-enabled binary, E4B model, token file, shape, and numerical route. Each
projection supplied 42 samples. The event profiler synchronizes after every
operation and is not end-to-end timing evidence, but the regressions are far
beyond the local noise floor:

| Projection | established median | packed-DP4A median | candidate/control |
|---|---:|---:|---:|
| Gate, epilogue 0 | `0.947200 ms` | `2.261504 ms` | `0.418836x` |
| Up+GELU, epilogue 1 | `1.088512 ms` | `2.518416 ms` | `0.432221x` |

The established-profile log SHA-256 is
`8fc734cf249c895edcf16e4cd0d447ba60fe47aeaae88161f1e691ecd962344`;
the candidate-profile log SHA-256 is
`76b5fd2be3fd571b08d995e0e6773ae44c263be1352276d29442a351eb56a80c`.
Two earlier diagnostic launches that omitted the explicit CUDA backend
environment produced zero CUDA events and are excluded rather than treated as
performance evidence.

**Decision and migration experience.** **No-Go; stop before whole-engine and
sanitizer expansion.** Preserving packed Q4 bytes does not offset replacing the
SM86 S8 tensor-core path with scalar DP4A in this tile: Gate is `2.387568x`
slower and Up+GELU is `2.313632x` slower. Keep the candidate laboratory-only
and unreachable without its explicit environment selector. Do not retry this
same packed-DP4A accumulator geometry by changing grids or launch bounds. A
future exact Q4-by-Q8 candidate must retain tensor-core arithmetic or remove a
larger producer/consumer transaction.

### E-20260831-25: K128 tensor-core gate/up pipeline screen

**Hypothesis and scope.** The existing default-off SM86 laboratory route in
`native/sm86/mmq_q4_q8_gate_up_pipe.cuh` keeps exact Q4-by-Q8 semantics but
processes K128 per pipeline stage, with packed Q4 register expansion and
`cp.async`, instead of the established K32-oriented Gate/Up path. The selector
was restricted to the real 449-token E4B `2560 -> 10240` projection. No ABI,
tuner, Metal path, or production default was changed.

**Mechanism and correctness.** `IMPARO_CUDA_MMQ_GATE_UP_PACKED_K128=1` passed
the canonical 449-token q4_0 `logit_agree.py` gate: top-1 was equal, top-id
overlap was 9/10, and maximum distribution delta was `0.16664` at the registered
`0.75` limit. Route tracing proved that the candidate reached 34/42 Gate and
34/42 Up projections as `full-tile`; the other eight of each safely used the
established `virtual-no-seam-r64` route. The partial-hit count is part of the
result and is not summarized as full-model admission.

**Performance screen.** A diagnostic CUDA-event pass used the same measured
binary, model, 449-token file, backend environment, shape, and numerical route
as `E-20260831-24`. Each projection supplied 42 samples, including the eight
safe-fallback samples; this makes the result favorable to the candidate rather
than exaggerating its regression.

| Projection | established median | K128 mixed-route median | candidate/control |
|---|---:|---:|---:|
| Gate, epilogue 0 | `0.947200 ms` | `2.299392 ms` | `0.411935x` |
| Up+GELU, epilogue 1 | `1.088512 ms` | `2.561792 ms` | `0.424903x` |

The candidate-profile log SHA-256 is
`f6ccfd3930aaebe994624b9770b5360de67ef2bba8ee6253543168460a86c767`.
The event profiler synchronizes after every operation and is not end-to-end
evidence, but Gate and Up+GELU are respectively `2.427568x` and `2.353481x`
slower, far outside the local noise floor.

**Decision and migration experience.** **No-Go; stop before whole-engine and
sanitizer expansion.** Merely grouping four K32-equivalent chunks into one K128
software pipeline did not remove MMA arithmetic or an intermediate tensor.
Packed-Q4 register expansion, the 27,648-byte shared working set, and lower
effective parallelism dominate any reduction in stage/control overhead. Keep
the candidate default-off and require a future retry to demonstrate a changed
mechanism--for example, eliminated dequant/quant work or a transactional
producer-consumer fusion--rather than another larger-K grouping alone.

### E-20260831-26: current-v2 sidecar FFN recheck and race attribution

**Purpose and identity.** This rechecks `E-20260830-14` on the current dirty
`9424517` v2 worktree after the fresh default route measured below llama.cpp.
The performance binary was `target/release/imparo-server.exe`, SHA-256
`2fbccdd35cb3f4c7c09d50d9c762063157377eab6d576b7dd494e3999b7fbced`.
The route/correctness/sanitizer binary was the capture-capable CUDA forward
artifact, SHA-256
`b2679621255ade00ce42773ee575c3cc23885dfea702ae45a29c4f54f54adea7`.
These are separate artifacts and their evidence is not represented as a
single-binary promotion receipt.

**Same-binary B/I/I/B result.** Both routes used the same 1,800-MiB packed-Q4
sidecar capacity, 449 raw tokens, q4_0 KV, repeat 15, warmup 3, and strict
environment policy. The only candidate difference was
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB=1`.

| Leg | Route | Prefill median | Decode median | Receipt SHA-256 |
|---|---|---:|---:|---|
| B1 | established workflow | `1983.6748 tok/s` | `62.1900 tok/s` | `bcb552724f09f4b9a5f5cb114752b6f643be4f6e31ac8614ada2b85ac56d3864` |
| I1 | sidecar FFN | `2427.5561 tok/s` | `60.9271 tok/s` | `fefabf2bb7319c6953d2e3826e25ece13226ca65a8f059026db239a368f8a9f9` |
| I2 | sidecar FFN | `2451.7502 tok/s` | `61.0513 tok/s` | `8799d6463f70468084516268986edc89d548b9dc16a261441e4c73b31d9618b1` |
| B2 | established workflow | `1996.1630 tok/s` | `61.2900 tok/s` | `87967bafe047ca1b8a5d56666cc297c9dac309e4f5e7d4713b38fb7f850821de` |

Cross-leg Prefill medians are `1989.9189` and `2439.6532 tok/s`, so the
candidate is `1.226006x` (`+22.6006%`) versus the current default. This is
noise-separated and materially larger than the old session's approximately
one-percent result.

**Fresh llama.cpp bracket.** The pinned upstream-base CUDA server ran with
explicit FA on and context checkpoints off. Its two surrounding 449-token legs
were `2227.9025` and `2211.9920 tok/s`; the intervening sidecar leg was
`2423.6050 tok/s`. The candidate is `1.087842x` versus the faster llama leg and
`1.091740x` versus their two-leg median. Its Decode median was `61.3921 tok/s`
versus llama's `57.7247` and `57.6604 tok/s`. Receipt SHA-256 values are,
respectively, `68ace5c4b62e2d77d5847dfca081828555dc184af6218c2f5a863efb48079870`,
`87a82b000379b99acdeaa522e77038df8d9746da9e840218d59f4d17243604a3`,
and `215896f4c7503395e602f890e5d3a8be49bdbc513a2e67e51c02ed5855b0bf4d`.

**Route and numerical evidence.** The current 449-token q4_0 logit gate passed
with equal top-1, 9/10 overlap, and maximum distribution delta `0.16664` at
the registered `0.75` limit. A non-timed trace recorded `attempts=41`,
`admitted=41`, and `committed=41`; the route-audit log SHA-256 is
`92eca8cf6019ca291d5a130ba40e3b4e5aaccde427833890db832e2e11d9802d`.
The missing forty-second workflow layer is not counted as a rejected attempt;
all attempted eligible transactions committed.

**Sanitizer evidence and attribution.** Full-path memcheck, initcheck, and
synccheck each exited zero with `ERROR SUMMARY: 0 errors`; the identical
summary-only log SHA-256 is
`adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832`.
The unfiltered full-model racecheck exited 86 and reported 87 errors, all in
the pre-existing `k_head_norm_rope_hadamard`/`block_sum` reduction rather than
the sidecar kernels; its log SHA-256 is
`f34ae56aee694b995bb8831e467099041076f10a9fe94c5ca3c656aeccf812fb`.
A second 449-token racecheck filtered to `q8_ready_paircta_r64`,
`q8_ready_physical_stream`, and `q8_ready_stream_fixup` exited zero with
`0 hazards displayed (0 errors, 0 warnings)`; its log SHA-256 is
`c69d09b31a66173faf98f267db57c7eb988035ce364fa0f0f184ffc933c541a7`.
The candidate kernels therefore had component-level sanitizer clearance, while
the first complete E4B CUDA run correctly failed its global race gate.
Instrumented full-path logits changed, consistent with scheduling exposing the
existing head-norm hazard; that failed run is not rewritten as candidate
correctness evidence.

**Barrier repair and global race closure.** `block_sum` uses `row[0..Warps)`
as reduction scratch. `k_head_norm_rope_hadamard` previously allowed a faster
warp to start overwriting `row` with normalized values while another warp
could still read that scratch. A block barrier was added immediately after the
head-norm `block_sum` call, without changing the reusable reduction helper or
the sidecar kernels. The repaired SM86 forward executable has SHA-256
`bc7955f177894c59fe0f46c4f2ba6da934bbdf79aeb271cc837c3f9560368972`.
The fixed 449-token q4_0 logit gate retained equal top-1, 9/10 overlap, and
maximum distribution delta `0.16664`. A 128-token filtered head-norm
racecheck, a 128-token full unfiltered racecheck, and finally the 449-token
full unfiltered racecheck each exited zero. The final full run reported
`RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`; its persisted
summary log SHA-256 is
`c69d09b31a66173faf98f267db57c7eb988035ce364fa0f0f184ffc933c541a7`.
This closes the previously identified global race blocker rather than hiding
it behind a kernel filter.

**Post-fix same-binary performance.** The repaired SM86 release server has
SHA-256
`4455069d86a0de302e1926c529beae19114a91c78f5eeb32798e4a22765cfc8d`.
Every leg began after six consecutive `P8, 0% GPU, <=64 MiB` samples and used
449 raw tokens, q4_0 KV, 15 requests, three excluded warmups, and the same
binary. The only candidate difference was
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB=1`; both Imparo routes used the same
1800-MiB experimental packed-weight sidecar budget.

| Leg | Route | Prefill median | Decode median | Receipt SHA-256 |
|---|---|---:|---:|---|
| B1 | established workflow | `2002.3221 tok/s` | `61.5445 tok/s` | `1ffcb7cad0b7c99561a1ea160c7e869bbc6b821853bf40e3869020c40532c784` |
| I1 | sidecar FFN | `2404.3675 tok/s` | `61.6696 tok/s` | `35b0fde0a91dc06f0b2fde6a302dd88d5ebf1715cbdb8222953209b12c4d3169` |
| I2 | sidecar FFN | `2459.2722 tok/s` | `62.1646 tok/s` | `3379599b4721918b8f932304cdb60774fe903be9a5fb5511b01ffaf4e096bfd9` |
| B2 | established workflow | `1997.8928 tok/s` | `62.7995 tok/s` | `7144fdbe6188c2f2655e7d0d35a8a0798176fda41bfbf44b1a61344d8f67d7f4` |

The two-route cross-leg medians are `2000.1074` and `2431.8199 tok/s`, so the
post-fix candidate is `1.215845x` (`+21.5845%`) versus the current workflow.
A fresh llama/Imparo/llama bracket used explicit FA on and context checkpoints
off. The llama legs were `2242.6010` and `2220.4118 tok/s`; the intervening
candidate was `2457.8431 tok/s`. This is `1.095979x` versus the faster llama
leg and `1.101428x` versus their two-leg median. Candidate Decode was
`61.6921 tok/s` versus the llama two-leg median `57.1911 tok/s`, or
`1.078700x`. Receipt SHA-256 values in order are
`f46e1584b4601f6dee4c5cb9b3890b8d42e37d120311fddaf91dc16b7d34aecb`,
`591575d467140eabca6791a5f095fa83475f1e558f8b56900d6bf9b3d5ac8e1e`,
and `81b524cbeafb81f8cf71220f4da0775b6428b591c34bf9b48f6f10c31463e464`.

**Decision.** **Validated high-value lab; performance and global sanitizer
targets cleared, but not yet an open-source default.** Keep the LAB selector
explicit until pre-commit refusal and post-commit fatal failure injection,
the exact public/private buffer table, rollback exercise, non-449 shape
regression, and one production-identity receipt close. The remaining work is
transaction/promotion hardening, not another claim that the sidecar mechanism
or the global race gate is unproven.

### E-20260831-27: exact-key full-logits Prefill Graph boundary

**Hypothesis.** Replaying an exact-shape, exact-route Prefill graph could remove
enough host dispatch and launch overhead to close the remaining short-Prefill
gap without changing kernel arithmetic.

**Evidence.** The implementation captured and replayed the full-logits boundary
under an exact identity key and retained fail-closed fallback. Strict same-binary
whole-Prefill A/B measured only `+0.52%`. Raw evidence is retained under
`dev_harness/results/raw-e4b-sm86-prefill-graph-lab/`; the original execution
boundary was restored after the screen.

**Decision.** **No-Go as the primary breakthrough route.** This is not evidence
that CUDA Graph is invalid or exhausted. It proves that this capture boundary
removes too little of the measured workload to clear the predeclared `3%`
whole-Prefill continuation floor. Reopen only after the operator/data-flow route
is stable, and then test a wider capture that removes measurable host/device
synchronization or argument traffic. Graph and fusion both remove launch work,
so their gains must be remeasured together and must not be added arithmetically.

### E-20260831-28: exact-128 Direct-K and full-K sidecar closure

**Identity and scope.** The worktree remained detached at
`9424517daf234e1ae71a4b8435be0c7f1e2ed0ee` with preserved uncommitted CUDA
experiments. The final-source-aligned `imparo-forward.exe` SHA-256 is
`c9d6df15d4bc4f382758fb19763d41c8c16f4b6f88077b2960088ea87f7c5554`;
the final `imparo-server.exe` SHA-256 is
`e6140ea414fde632573b44903164eeb5cb9f58f530efc849fbd01ffc14e104b4`.
The model SHA-256 is
`df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3`.
The pinned llama launcher and CUDA backend SHA-256 values are respectively
`e403689aaca3540917ba11209b70c15845c65630a1ad7fca3812d73225f04fbe`
and `986ec81da4a63a0fe30e6339a6ec5b4af533f6e446ecdef0013faff3b0e35b4c`.
Hardware was an RTX 3060 Laptop GPU, SM86, 6,144 MiB, driver 596.08, CUDA
toolkit 12.9. The formal comparison used q4_0 K/V, explicit llama FA, context
checkpoints off, raw-token parity, max1, repeat12 and four discarded warmups.

**Mechanism.** Exact 128-token PLE uses 16 independent complete-K CTAs instead
of the long-Prefill Stream-K seam and can publish the consumer-native ready-Q8
layout. The FFN transaction retains its complete Gate/Up/private-Q8/Down
ownership and uses the exact-128 R128 complete-K Down kernel. The 449--512
schedule remains separate. Same-binary 128-token screening moved the established
control from `1443.2086` to `1571.8290 tok/s` for Direct-K plus ready-Q8
(`1.0891x`). Adding the admitted FFN sidecar produced the final route.

**Fit boundary.** The full-model packed-Q4 sidecar requires 1,857,945,600 bytes
(about 1,771.9 MiB). The default post-allocation budget was 1,749,090,304 bytes,
about 103.9 MiB short, so all 42 attempts correctly fell back. The experiment-only
`IMPARO_CUDA_RESERVE_MIB=640` produced a 1,883,242,496-byte budget and all 42
layers were admitted and committed. This is fit evidence, not permission to
hard-code a 640-MiB global reserve. Production promotion must express the route
through the versioned tuner and admit the sidecar from actual model/device/KV
headroom, with conservative fallback when the complete hot set does not fit.

**Resources.** `cuobjdump --dump-resource-usage` on the measured forward binary
reports 128 registers/thread and zero stack/local memory for all four PLE
`fused_gate<ReadyOutput,DirectK>` instances. The launch uses 32,512 bytes dynamic
shared memory. The exact full-K Down instance uses 126 registers/thread, zero
stack/local memory and 36,864 bytes dynamic shared memory. No neighboring kernel
resource record is used as a substitute.

**Correctness and safety.** The fixed formal gates all passed without relaxing
the 0.75 threshold: n=512 logit agreement was 9/10 with maximum delta 0.11079;
n=512 recurrent Decode passed 8/8 with maximum delta 0.47566; n=2000 recurrent
Decode passed 8/8 with maximum delta 0.73078. The additional 128-token Decode
canary kept 8/8 top-1 identity, but its first distribution row reached 0.94353;
this extra stress result remains recorded rather than being rewritten as a pass.
Determinism passed at 128/449/512/2000 (`1/4 distinct` each) and recurrent
n=521 (`4/4` samples, one distinct).

The transaction boundary was also exercised with explicit failure injection.
With `IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_PRECOMMIT_LAB=1`, all 42 layer
attempts were admitted but none committed (`attempts=42`, `admitted=42`,
`committed=0`); the established workflow completed with finite output and the
same top-10 result as the candidate run. This proves that refusal before the
first public-output launch remains a legal conservative fallback. With
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_POSTCOMMIT_LAB=1`, the first committed
layer made the forward fail closed (`attempts=42`, `admitted=1`, `committed=1`,
process exit 1) with `sidecar-only FFN injected failure after commit` and
`model admission was not all-or-none`; no legacy Gate/Up/Down path was enqueued
after the committed public write. The default insufficient-headroom run above
provides the third transaction case: zero admission and conservative fallback
when the complete model hot set does not fit.

Selector isolation was checked in five fresh forward processes with
`IMPARO_CUDA_PROFILE_OPS=1` and both route traces enabled. The exact labels
`ffn_sidecar_down_full_k_r128`, `ple_gate_ready_sm86`, and
`route=ready tokens=128 direct_k=1` appeared for every one of the 42 model
layers at n=128 and were absent at n=127, 129, 449, and 512. All five processes
exited 0; the non-128 processes retained their established sidecar schedule.
The event scopes synchronize and therefore this matrix is selector evidence
only, never admissible timing evidence. Static SM86 unit tests were run with
`cargo test -p imparo-cuda --no-default-features --features cuda-static --lib
knobs::tests` (7 passed) and the corresponding `correctness::tests` filter
(8 passed); both commands exited 0.

The final full-model sanitizer matrix used the same 128-token route and binary:

| Tool | Exit/result | Log SHA-256 |
|---|---|---|
| memcheck | exit 0, `ERROR SUMMARY: 0 errors` | `adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832` |
| initcheck | exit 0, `ERROR SUMMARY: 0 errors` | `adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832` |
| racecheck | exit 0, `0 hazards` (`0 errors`, `0 warnings`) | `c69d09b31a66173faf98f267db57c7eb988035ce364fa0f0f184ffc933c541a7` |
| synccheck | exit 0, `ERROR SUMMARY: 0 errors` | `adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832` |

Logs are under
`dev_harness/results/e4b-sm86-direct128-final-sanitizer/`.

**Final cross-engine bracket.** Receipt files contain every raw sample and the
strict environment:

| Tokens | Leg | Prefill median | Receipt SHA-256 |
|---:|---|---:|---|
| 128 | llama A | `2105.1630 tok/s` | `d2a493891591eaa252608b1fe20620bac3a4db977ac7f2434fc6ee88c7104fd3` |
| 128 | Imparo | `2054.0736 tok/s` | `d381c8c6b8411d14f973549488f15afc8a3801eb80209a33fddd955eb72bb3d9` |
| 128 | llama B | `2031.3761 tok/s` | `7a0d13a2348826d2d1711c67fdc2dc3468e936e2c79172fb980a1230d5cb8f3e` |
| 449 | llama A | `2275.4376 tok/s` | `a508644835d8a6bac56d42ba0cd6aee06b901f0d4448156d5ed662443d3ea1c9` |
| 449 | Imparo | `2487.0453 tok/s` | `d33f60d4441828f599a265594048e54c95c84f321578a4c898bd51f1018f87ff` |
| 449 | llama B | `2253.6149 tok/s` | `8a92857ff27d09b96eafb1cc8a4254cae2238f3576d24111e4532fbd88821a2e` |

At 128 tokens, the two llama medians average `2068.2696 tok/s`; Imparo is
`0.993136x` against that two-sided reference and `0.975731x` against the faster
leg. At 449 tokens, the two llama medians average `2264.5263 tok/s`; Imparo is
`1.098263x` against the two-sided reference and `1.092996x` against the faster
leg. Exact-128 specialization therefore did not contaminate the 449 schedule.

**Decision and rollback.** **Validated lab; not a release default.** The route
has crossed the formal model, determinism, sanitizer, resource and final
cross-engine performance gates, and it brings the short case to the requested
same-performance class on this evidence domain. It remains selected by LAB
environment variables and depends on an experimental reserve override, which
the correctness policy deliberately refuses to treat as a production numerical
selector. Unset the five experiment variables to roll back to the safe native
route. Before admission, replace them with a versioned, receipt-bound opaque
candidate in the existing CUDA tuner/config path, prove complete-hot-set fit
without a device-specific reserve constant, and exercise precommit fallback,
postcommit fatal handling and insufficient-headroom fallback on that identity.

### E-20260831-29: complete-model warmup admission without reserve override

**Hypothesis.** The startup residency reserve was being subtracted twice: once
when resident weights were selected and again when the optional packed-Q4 cache
was budgeted after activations and current KV had already been placed. Replacing
the second device-specific guard with current free-memory admission is safe only
if no public FFN result may commit before every layer's hot set has been built.

**Implementation and invariant.** Model execution now has a two-stage
transaction. The first eligible forward packs Gate, Up, and Down for every FFN
layer but returns to the established workflow before any sidecar public output.
`end()` marks the model ready only when all 42 calls packed successfully, the
stream synchronized, and no pending error exists. Later forwards may commit the
sidecar transaction. An explicit packed-cache MiB cap remains a conservative
upper bound and is clamped to current free memory; it cannot mint fictitious
capacity. Kernel-lab microbench behavior is unchanged.

**Identity.** The detached source base remains
`9424517daf234e1ae71a4b8435be0c7f1e2ed0ee`. The rebuilt SM86-only
`imparo-forward.exe` SHA-256 is
`89c0568c1f1954c73cae7ad5a479fd421cd1aeb37e9b1f932cacebfb3cf3397f`;
the matching `imparo-server.exe` SHA-256 is
`fa8b637e20a8633cb0c550afb94910f99cf0b1507458f07c3bd0a0878950fef9`.

**Fit and transaction evidence.** With both
`IMPARO_CUDA_RESERVE_MIB` and `IMPARO_CUDA_Q4_PACKED_SIDECAR_MIB` unset, CUDA
reported 2,634,022,912 free bytes and the same bounded cache budget. A
three-repeat 128-token process recorded:

- repeat 1: `pack_calls=42`, `admitted=0`, `committed=0`, then complete hot-set
  promotion;
- repeats 2 and 3: `admitted=42`, `committed=42`;
- all three top-10 outputs were identical.

With a deliberate 64-MiB cache cap, two repeats remained
`prepared=0`, `committed=0`, exited 0, and produced the same finite top-10
output. After one preparation repeat, pre-commit injection admitted all 42
layers but committed none and fell back with identical output. Post-commit
injection failed the second repeat after the first public write, reported one
admission/commit and exited 1; it did not enqueue a legacy double write.

**Strict cross-engine bracket.** The same raw-token, q4_0 K/V, llama FA-on,
context-checkpoint-off protocol used 12 requests per leg with four discarded
warmups:

| Leg | Prefill median | Receipt SHA-256 |
|---|---:|---|
| llama A | `2078.8489 tok/s` | `021ae4fd49ff54a30f7dcd9ac9e0de4d535e1ca192c8fafb39d081bdd7195a27` |
| Imparo auto-fit | `2035.5510 tok/s` | `4c23b2e8dc50bc8bab33c49228dd1c7809b9d3956ab5dd7276346d974c834f80` |
| llama B | `2051.3195 tok/s` | `c1b56b8a0b396310ada3d67466514a48e84f7c8aba992ce3e9a4480b748d6fbb` |

The two llama legs average `2065.0842 tok/s`; Imparo is `0.985699x` against
that reference and `0.979172x` against the faster leg. Raw receipts are under
`dev_harness/results/raw-e4b-sm86-direct128-auto-fit-final/`.

**Tests and decision.** `ffn_gated_down_contract` passed 10/10, static SM86
`knobs::tests` passed 7/7, and static SM86 `correctness::tests` passed 8/8; all
commands exited 0. **Validated lab; auto-fit blocker closed.** The route now
meets the requested same-performance class without a device-specific reserve
constant. It remains laboratory-selected until one receipt-bound atomic route
knob replaces the four interacting environment flags. The optional cache must
also gain priority reclaim/retry before a later activation or KV grow can make
the model fail solely because a Prefill optimization is resident.

### E-20260831-30: optional sidecar priority reclaim

**Mechanism.** Packed Q4 weights are now explicitly a lower-priority cache.
Activation buffer/arena growth, transactional KV arena/page-table growth, and
Decode control or Decode scratch allocation first attempt their normal
allocation. On CUDA OOM they synchronize the owning stream, invalidate both
Graph executables, release packed sidecars, clear model admission and budget
state, then retry once. The next eligible Prefill rebuilds the cache through
the complete-model warmup; no stale packed pointer or ready identity survives
reclaim. Prefill scratch allocation does not invoke this helper while a packed
pointer may be active.

**Build and tests.** The latest SM86 release server SHA-256 is
`c238bd46791c7f53e81428a36c4179cb1caf43d044f0bb5f8a11f5388640c7aa`.
`ffn_gated_down_contract` passed 11/11 after adding the priority/reclaim
invariants. A short strict-environment sanity bracket used six requests per
engine with two warmups: llama measured `2016.2403 tok/s` and Imparo measured
`2042.9637 tok/s`, or `1.013254x`. Receipt hashes are respectively
`a57cd4fa92dd0f674e55b91ab5aff88e4fb9ebec521483845d1a9f3f07617dd4`
and `07a7f40d8ea63ef8bbd55ee709202869f1bf353f722c1f13a481f656b6ca56d3`;
raw files are under
`dev_harness/results/raw-e4b-sm86-direct128-reclaim-quick/`.

**Decision.** The quick screen is a regression check, not a replacement for
E-20260831-29's two-sided 12-repeat result. Source structure, compilation,
contract semantics and performance are positive. Promotion still requires a
runtime OOM/fault-injection witness that observes actual release, retry,
fallback/rebuild and unchanged outputs; until then the reclaim branch is not a
fully closed release gate.

### E-20260831-31: receipt-bound exact-128 atomic tuner slice

**Scope and identity.** This entry productionizes the already validated
E-20260831-28/29 mechanism; it does not add or claim a new performance result.
The worktree remains detached at
`9424517daf234e1ae71a4b8435be0c7f1e2ed0ee`, with all pre-existing dirty work
preserved. CUDA search space is version 26, selector version 4, and route
implementation version 4. The new `prefill_exact128_sm86_route` is slot 39,
legal values `{0,1}`, and compiled default `0` on every architecture. General
FFN threshold slot 38 begins at 129 tokens, so exact 128 cannot be selected as
an accidental threshold side effect.

**Atomic workload and fail-closed proof.** The Gemma4 bench extension names the
real `inp_gate`/`proj` PLE pair in addition to the real FFN Gate/Up/Down triple.
The tuner validates independent offsets, Q4 kinds, dimensions and checked
per-layer stride, allocates the same private Gate/Back/PerLayer layout as the
workflow, and measures one production-order FFN-then-PLE transaction. Native
code resets a tuner-only reachability mask for every submission and sets bits
only after these concrete success points:

1. PLE ready-Q8 gate launch;
2. PLE exact-128 Direct-K public projection commit;
3. FFN transaction admission after all recoverable precommit checks;
4. FFN exact-128 Rows128 Down launch;
5. FFN public-output transaction commit.

The required mask is `0x1f`. Candidate warmup and every timed submission must
equal it; control must equal zero. A partial mask is a silent fallback and the
tuner rejects value 1. Dynamic CUDA plugins and every non-CUDA backend return
zero, so this static laboratory evidence surface does not expand ABI 25 or
affect Metal. Config and correctness hashes bind the exact slot value, complete
registry order, space/selector/implementation versions, model, device, driver,
ABI and numerical route before a stored value can load.

The Rust backend snapshots the complete architecture-selected knob table
immediately after native SM discovery and before applying any config. Every
later config lookup restores that snapshot before its first possible early
return. A same-process reload with a missing, torn, stale or rejected receipt
therefore cannot retain a numerical route admitted by an older snapshot.

**Non-timed SM86 functional evidence.** The gated
`IMPARO_CUDA_EXACT128_SMOKE=1` static test initializes 42.9 MiB of deterministic
non-degenerate Q4 weights on the RTX 3060 Laptop GPU, runs the safe selector and
the atomic candidate at 128 tokens, and compares both public FFN and PLE
outputs. The final run exited 0 with five of five candidate bits, zero control
bits, FFN cosine `1.000000000` / relative-L2 `0.000000000`, and PLE cosine
`1.000000000` / relative-L2 `0.000000000`. A start-position-1 negative case
kept FFN eligible but forced PLE fallback; its missing PLE bits made the mask
incomplete as required.

The first smoke run is retained as a harness failure: the initial 2^-10 scale
and periodic low-amplitude activation stimulus quantized the complete FFN
output to zero, so the test failed with “FFN was not exercised” instead of
accepting a meaningless all-zero comparison. The corrected LCG activation and
2^-6 Q4 scale moved the stimulus above the quantizer floor. This is a reusable
rule: correctness stimuli for quantized fused paths must prove nonzero norm and
finite values before cosine or equality can count as evidence.

**Build and tests.** All commands used `IMPARO_CUDA_ARCHS=86`; none measured
throughput or compared engines.

- `cargo build -p imparo-tune --features cuda`: exit 0; real NVCC/static link
  completed.
- `cargo test -p imparo-cuda --no-default-features --features cuda-static --lib
  knobs::tests`: exit 0, 8/8.
- corresponding `correctness::tests`: exit 0, 9/9.
- `cargo test -p imparo-cuda --test ffn_gated_down_contract`: exit 0, 14/14.
- `cargo test -p imparo-tune`: exit 0, 18/18.
- the gated exact-128 GPU smoke above: exit 0, 1/1.

**Decision and rollback.** **Implementation and functional Gate-A closure;
safe-off.** The former four-flag laboratory mixture is now representable as
one versioned, receipt-bound tuner candidate and cannot be persisted after a
partial fallback. It is not yet promoted: no new tuner performance sweep,
model logit/decode suite, sanitizer matrix or signed correctness receipt was
run for this new selector identity. That work was deliberately not fabricated
from E-28/29 and remains pending until performance measurement is explicitly
requested. Rollback is slot 39 value 0; missing/invalid receipt, identity drift,
failed numerical gates or incomplete route evidence all retain that value.

### E-20260831-32: exact-128 production admission and D256 batch32 repair

**Scope.** This entry closes the numerical, loader and memory/concurrency work
that E-20260831-31 deliberately left open. It does not claim a new performance
result. The source baseline remains detached
`9424517daf234e1ae71a4b8435be0c7f1e2ed0ee`; all pre-existing dirty work was
preserved. The route is exact to SM86, 128 tokens and slot 39 value 1. Other
shapes, other SM versions and slot 39 value 0 retain their prior selectors.

The model gate isolated D256 attention as the remaining recurrent-distribution
instability. The exact route now excludes the tiled D256 implementation and
selects the existing batch32 implementation with f32 value accumulation and no
extra llama-style reduction. Ambient `IMPARO_CUDA_ATTN_BATCH32_HALF` and
`IMPARO_CUDA_ATTN_LLAMA_REDUCE` laboratory variables cannot change this
receipt-bound arithmetic. The trace identity is `batch32-exact128`.

Several narrower hypotheses were rejected rather than hidden: physical and
virtual PLE K seams, four multi-seam patterns and single seams from K=8 through
K=72 all collapsed to the less-stable control result. Disabling D256 virtual
Stream-K failed immediately; disabling fused D256 or using the tiled route
improved only part of the distribution and later failed overlap. Half-value
accumulation also failed. The plain batch32 route was the first candidate to
pass all eight recurrent steps, so the fix is an atomic numerical-route choice,
not a claim that batch32 is universally faster.

**Artifact identity.** The final SM86 release build used
`IMPARO_CUDA_ARCHS=86` and completed successfully. The 21,820,928-byte
`target/release/imparo-forward.exe` has SHA-256
`2d7b6ad4c1b92f4d71e677eaad3ec5dc553d3395b66b0d8ca7d57f2c9061533a`.
The exact config SHA-256 is
`d8d61793d16b24cb84fed62896652d47c86e7817b6c5d959d344800e0e08c0f7`;
its adjacent receipt SHA-256 is
`becf0039d70e0185bc49b29fcbf215fd5e0bdde612a52316f5b627d69c6021ee`.
The receipt binds model SHA-256
`df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3`
and the authenticated llama oracle manifest/revision already recorded in the
receipt. A separate dirty-diff hash remains unresolved because this worktree
contains a large pre-existing, user-owned change set; it must be produced at a
clean branch/commit boundary rather than guessed from this session.

**Fixed correctness and production-loader evidence.** Receipt producer v5 ran
the fixed suite and sealed only after all six gates exited zero:
`logit_agree` at 128/449/512 and eight-step `decode_agree` at 128/512/2000,
all with q4_0 KV and the authenticated llama/FA oracle. The final direct
128-token recurrent run had all eight top-1 values equal, top-10 overlap 9 at
every step and maximum deltas from 0.10700 through 0.46505 under the 0.75 gate.
A production smoke then explicitly removed `IMPARO_CORRECTNESS_GATE`, loaded
the adjacent receipt through the normal fail-closed loader, executed the
128-token model forward, exited zero and produced 262,144/262,144 finite logits
and a 1,048,576-byte raw f32 dump. This proves the receipt is usable by the
production path rather than only by the isolated sealer.

The final source/binary test matrix was rerun after the D256 repair:

- `cargo test -p imparo-host --lib receipted_config`: 9/9 pass;
- `python dev_harness/test_seal_receipt.py`: 10/10 pass;
- SM86 static-feature `cargo test -p imparo-cuda --lib --features cuda`:
  29/29 pass, including the exact-route GPU smoke when enabled;
- `cargo test -p imparo-cuda --features cuda --test
  ffn_gated_down_contract`: 15/15 pass.

An initially repeated featureless CUDA test command ran zero filtered tests and
is not counted as evidence. Loading `vcvars64` and enabling the static CUDA
feature produced the real 29-test matrix above.

**Final sanitizer evidence.** Both the gated 42.9-MiB kernel transaction and
the complete E4B 128-token production forward were rerun against the final
binary under memcheck, initcheck, racecheck and synccheck. Every application
exited zero. All six non-race logs report `ERROR SUMMARY: 0 errors`; both race
logs report `0 hazards displayed (0 errors, 0 warnings)`. The eight immutable
session logs are named
`exact128-v26-batch32-final-{kernel,full-model}-{memcheck,initcheck,racecheck,synccheck}.log`
under `docs/evidence/cuda-sm86-e4b-open-source/`.

**Priority-reclaim witness.** A temporary dev-tool hook installed the existing
native reclaim request only after gate-mode config admission, so the production
fail-closed environment policy remained unchanged. A separate SM86 target ran
three same-process 128-token forwards: complete-model warmup; forced allocation
OOM/reclaim followed by fallback and complete rebuild; then the rebuilt atomic
route. Native evidence reported `reclaim=1873428480 retained=0`, the rebuild
again reported 42 pack calls and promoted the complete hot set, and the third
forward reported `attempts=42 admitted=42 committed=42 prepared=1`. Control and
forced final dumps were both 1,048,576 bytes and had identical SHA-256
`01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261`.
All three top-10 rows were also identical and every final logit was finite. The
instrument-only source was then removed in reverse-patch order; the final
production executable remained byte-identical at SHA-256
`2d7b6ad4c1b92f4d71e677eaad3ec5dc553d3395b66b0d8ca7d57f2c9061533a`,
and the final 15-test structural contract suite passed again.

**Decision and rollback.** Numerical admission, production receipt loading and
the final sanitizer gate are **Go** for this exact SM86/128 route. Rollback is
slot 39 value 0; a missing/torn/mismatched receipt, model/device/driver/ABI/KV/
space drift or failed gate restores architecture-safe defaults before applying
any candidate knob. Open-source performance admission is deliberately still
pending: the prior 128-token bracket was only approximately 0.986--1.013x
llama.cpp and predates this D256 arithmetic change. Per the explicit instruction
not to run unsolicited benchmarks, no new whole-engine timing is inferred from
these correctness runs. A final interleaved same-binary/llama bracket and
shape-regression check must be requested and recorded before claiming the
short-Prefill performance requirement is closed.

### E-20260831-33: exact-128 compile-time attention specialization

**Hypothesis and boundary.** Static inspection of the E-20260831-32 production
binary found that the exact-128 route passed `half_v_accum=0` and
`llama_reduce=0` as runtime kernel arguments. NVCC therefore retained the
dormant half-value WMMA accumulator, alternate reduction, shared tiles and
control flow in the same kernel used by the receipted route. This experiment
does not change the numerical algorithm or widen the selector: SM86, D256,
exactly 128 tokens and slot 39 value 1 remain the only production users. The
environment-controlled batch32 laboratory route keeps the original generic
specialization as an immediate rollback.

The implementation uses one `ExactF32=true` compile-time specialization and
one `ExactF32=false` generic specialization. It deliberately does not emit four
boolean combinations: an initial local build proved that design needlessly
inflated the executable, so it was replaced before admission. The final build
used the repository-owned `IMPARO_CUDA_ARCHS=86` selector; `cuobjdump
--list-elf` reports only `imparo_cuda.sm_86.cubin`.

**Binary and resource evidence.** The current 21,857,792-byte release binary is
`target/release/imparo-forward.exe`, SHA-256
`6dabc3d516e4adce1d82d129b3fb62130683cbb061a4780a878052302f588c82`.
CUDA 12.9 `cuobjdump` reports:

| batch32 specialization | registers | stack/thread | static shared | SASS |
| --- | ---: | ---: | ---: | ---: |
| generic `ExactF32=false` | 56 | 64 B | 10,928 B | 2,447 |
| receipted `ExactF32=true` | 56 | 0 B | 2,192 B | 1,390 |

The exact specialization removes all 63 static LDL and 87 STL sites, reduces
CTA barriers from 10 to 6, shared stores from 93 to 33 and branch-family sites
from 96 to 34. Dynamic shared remains the unchanged `head_dim * 4` bytes
(`1,024` for D256). These are static structural/resource results, not a timing
claim. The next deeper attention opportunity remains the serial 32-key score
loop, where only warp 0 computes each score and all warps synchronize per key;
that algorithmic change is not mixed into this mechanical specialization.

**Admission evidence.** The SM86-only release build exited zero. The targeted
source contract passed 15/15 and the static CUDA library suite passed 29/29.
Receipt producer v5 then reran all six fixed llama/FA q4_0 gates and atomically
sealed the updated receipt: logit agreement at 128/449/512 and eight-step decode
agreement at 128/512/2000 all passed. Its new receipt SHA-256 is
`eeb15f2e3cfca8610fa90a1d6183f87139c09214fb9e94332853624211d65ce4`;
the backend fingerprint is
`6748709974fa00a070c75888a0839af35f047cd9fea13c23da51c90549f42bb9`.
With gate mode removed, the normal production loader accepted that receipt and
one 128-token forward exited zero with 262,144/262,144 finite logits.

The complete 128-token production forward also exited zero under all four
Compute Sanitizer tools. Evidence files are
`exact128-v26-static-specialized-full-model-{memcheck,initcheck,racecheck,synccheck}.log`.
Memcheck, initcheck and synccheck report `ERROR SUMMARY: 0 errors`; racecheck
reports `0 hazards displayed (0 errors, 0 warnings)`. Their SHA-256 values are
`adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832`
for each non-race summary and
`c69d09b31a66173faf98f267db57c7eb988035ce364fa0f0f184ffc933c541a7`
for racecheck.

**Decision and rollback.** Static resource, numerical, loader and sanitizer
gates are **Go** for this specialization. Rollback is either slot 39 value 0 or
the retained `ExactF32=false` generic launch; neither requires an ABI, tuning
space or persisted-config change. Open-source performance admission remains
pending by explicit policy: no new performance benchmark was run, so the large
resource reduction is not reported as a Prefill throughput gain. The final
interleaved candidate/control and candidate/llama shape matrix is still required
before claiming parity.

### E-20260901-34: exact-128 four-warp D256 score scheduling

**Hypothesis and boundary.** E-33 removed dormant arithmetic from the receipted
D256 batch32 kernel, but its 32-key score loop still assigned every score to
warp 0 and placed one CTA barrier after every key. The other three warps were
idle during all sixteen K16 WMMA operations for each score. This candidate
changes only independent work scheduling for the `ExactF32=true` specialization:
four warps own four private Q/K/C shared tiles and advance four scores together.
Every score retains the established K16 WMMA accumulation order. The generic
laboratory specialization, every non-128 shape, every non-SM86 architecture and
Metal remain unchanged. Exact production launch width is fixed to 128 threads;
slot 39 value zero restores the architecture-safe route.

**Static mechanism and resources.** The score batch now has eight four-score
rounds instead of 32 single-score rounds. Per-warp tiles increase static shared
memory from 2,192 to 8,336 bytes; the unchanged D256 numerator adds 1,024 bytes
of dynamic shared memory. CUDA 12.9 `cuobjdump` reports 56 registers/thread,
zero stack and zero local memory. The exact specialization has 782 static SASS
instruction sites, five barrier sites and zero LDL/STL sites, versus E-33's
1,390 sites, six barrier sites and zero LDL/STL. These are structural results,
not a throughput claim. The SM86-only release executable is 21,855,744 bytes,
SHA-256 `280e0eb87db0b421e4f0919716c0b9f27e38a9778e7b1d5f6cc32b0308530844`.

**Build, numerical and loader evidence.** The following all exited zero:

- `cargo test -p imparo-cuda --no-default-features --features cuda-static
  --lib`: 29/29;
- the updated `ffn_gated_down_contract`: 16/16, including private four-warp
  tiles and fixed 128-thread exact launch;
- `python dev_harness/test_harness.py`: 47/47;
- authenticated q4_0 `decode_agree.py 128 8 0.75`: 8/8, with exact serial
  control and four-warp candidate producing the same per-step gate results;
- receipt producer v5: logit agreement at 128/449/512 and eight-step decode
  agreement at 128/512/2000 all passed against the manifest-bound
  `4695f001` llama/FA bundle.

The new receipt SHA-256 is
`939bb31c2aa8543d2106219cb94dcd61ee23b4b739621d969a6208b983a4fa5f`;
its backend fingerprint is
`19f477cefb55d575204c09a3571617c81f19bdb6f0ff0b8de9112f767b0fbc8e`.
With gate mode removed, the production loader accepted that receipt and one
128-token forward produced 262,144/262,144 finite logits.

**Concurrency and memory evidence.** The complete 128-token E4B forward exited
zero under all four CUDA 12.9 Compute Sanitizer tools. Memcheck, initcheck and
synccheck report `ERROR SUMMARY: 0 errors`; racecheck reports
`0 hazards displayed (0 errors, 0 warnings)`. Evidence files are
`exact128-v26-parallel-score-full-model-{memcheck,initcheck,racecheck,synccheck}.log`.
The three non-race logs each hash to
`adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832`;
racecheck hashes to
`c69d09b31a66173faf98f267db57c7eb988035ce364fa0f0f184ffc933c541a7`.

**Invalid-run lesson and harness repair.** Two initial manual diagnostics were
invalid: one used the non-runtime name `IMPARO_TUNE_CONFIG`, so the engine
quietly exercised safe defaults; another pointed at a nearby but unauthenticated
llama directory. Route trace and the repository bundle authenticator exposed
both mistakes. `harness.py`, `logit_agree.py` and `decode_agree.py` now fail
closed in correctness-gate mode unless `IMPARO_HOST_CONFIG` exists, the tracked
manifest hash is bound and the complete reference inventory authenticates.
The flexibility of ordinary diagnostic runs is unchanged.

**Decision and rollback.** Numerical, production-loader, resource and four-tool
sanitizer gates are **Go**. The candidate remains performance-pending because
the user explicitly reserved benchmark runs for a separate request. No Prefill
speedup or open-source parity claim is inferred from SASS reduction. Rollback is
the E-33 serial score loop or slot 39 value zero; neither changes ABI, tuning
space, other shapes, other SM families or Metal.

### E-20260901-35: exact-128 immutable Q and physical-row reuse

**Hypothesis and boundary.** E-34 schedules four independent scores at once,
but each score still converted and scaled the same 256-element FP32 Q vector
and rebuilt the same sparse WMMA A tiles. It also translated the same 32
logical KV positions once in the score phase and again in the value phase.
This candidate caches the sixteen immutable scaled Q tiles once per CTA and
caches the 32 physical rows once per score batch. A shared, runtime-visible
K-round count prevents the worst barrier/code cloning observed in the first
cache implementation. The K16 WMMA accumulation order, f32 value accumulator,
selector, output layout and fallback are unchanged. The change remains confined
to `ExactF32=true`: SM86, D256, exactly 128 tokens and slot 39 value 1. The
generic laboratory route, other shapes/SM families and Metal are unchanged.

**Rejected refinements.** A scaled-Q-vector variant reduced static shared
memory to 8,976 bytes, but CUDA 12.9 emitted 1,576 SASS instruction sites,
13 barrier sites and 16 HMMA sites (52 registers, zero stack/local). It was
reverted. Combining the runtime-visible bound with `#pragma unroll 1` likewise
regressed to 1,568 SASS sites, 13 barriers and 16 HMMA sites (50 registers,
zero stack/local), so that one-line experiment was also reverted. These are
compiler-generated-code failures, not numerical failures; retaining them would
make the final candidate harder to maintain without evidence of a mechanism win.

**Final binary and resources.** Repository freshness build with
`IMPARO_CUDA_ARCHS=86` exited zero, and `cuobjdump --list-elf` lists only
`imparo_cuda.sm_86.cubin`. The 21,852,160-byte
`target/release/imparo-forward.exe` hashes to
`0ddbcbd181b705579eeac5ea7266adeaccd5b339e37756d53bce21eb300c825d`.
For `k_attention_batch32_f16<true>`, CUDA 12.9 reports 48 registers/thread,
14,608 bytes static shared, zero stack and zero local memory. Targeted SASS
inspection finds 1,184 instruction sites, six CTA barrier sites, 14 HMMA sites
and zero LDL/STL sites. The unchanged D256 accumulator contributes 1,024 bytes
dynamic shared. These are generated-code/resource facts, not timing evidence.

**Correctness, loader and concurrency evidence.** All of the following exited
zero on the RTX 3060 / SM86 host:

- CUDA static library tests: 29/29;
- exact route/source transaction contracts: 16/16;
- correctness-gate preflight tests: 47/47;
- receipt producer v5: all six authenticated llama/FA q4_0 gates, namely
  logit agreement at 128/449/512 and eight-step Decode agreement at
  128/512/2000;
- ordinary production loading with gate mode unset: 262,144/262,144 finite
  logits for a 128-token forward;
- complete-model memcheck, initcheck and synccheck: `ERROR SUMMARY: 0 errors`;
- complete-model racecheck: `0 hazards displayed (0 errors, 0 warnings)`.

The atomically replaced receipt hashes to
`18d8a7aeecb2986f5fbfc84ef32cfd2d1de733a349b838166b76a54f474f7691`;
its backend fingerprint is
`87a00d167ae069a67f1fa02cf7c9096e9d06167c4ecf394109774019f5c65b0a`.
Sanitizer logs are
`exact128-v26-qtile-runtime-rounds-full-model-{memcheck,initcheck,racecheck,synccheck}.log`.
The three non-race logs each hash to
`adba10db853bd67fe9009b776eefe7b61210d2cbe3eee029ba6fa188978b5832`;
racecheck hashes to
`c69d09b31a66173faf98f267db57c7eb988035ce364fa0f0f184ffc933c541a7`.

**Decision and rollback.** Structural, numerical, loader and concurrency gates
are **Go** for continued exact-128 A/B work. Performance and open-source
admission remain **Pending**: by explicit policy no new performance benchmark
was run, and the cache candidate has a real shared-memory/instruction-cache
tradeoff that static inspection alone cannot resolve. Slot 39 value zero is the
immediate conservative rollback; reverting the Q/row cache restores E-34 while
preserving the same ABI, tuning space and fixed correctness suite.

### E-20260901-36: exact-128 atomic tuner coverage repair

**Failure found.** Slot 39 is an atomic production selector: at exactly 128
tokens on SM86 it controls the D256 batch32 Attention route, the PLE gate/direct
projection and the ready-Q8 gated FFN/Down transaction. The tuner workload named
`PrefillFfnExact128`, however, timed only FFN followed by PLE. Its reachability
mask was `0x1f`, containing the two PLE and three FFN commit bits but no Attention
bit. A candidate could therefore win the micro sweep on FFN/PLE while regressing
Attention by more than both gains, then persist one value that changed all three
stages. That is a selector-admission bug, not a kernel-speed observation.

**Repair.** The exact-128 micro transaction now selects a real D256 attention
layer and its model window, then executes every controlled stage in production
order inside one begin/end CUDA-event interval: Attention, FFN, then PLE. The
native exact D256 batch32 launch publishes `EXACT128_ATTN` only in tuner-lab mode,
and the complete evidence mask is now `0x3f`. A missing stage, fallback, launch
error or asynchronous failure leaves a partial mask and rejects the candidate.
The public runtime selector, numerical class, kernel arithmetic, buffers, ABI,
tuning-space version, conservative slot-39 value zero and all non-SM86/non-128
routes are unchanged.

**Build and tests.** The following commands exited zero on the RTX 3060 / SM86
host after the repair:

- `cargo test -p imparo-cuda --no-default-features --features cuda-static
  --test ffn_gated_down_contract`: 16/16;
- `cargo test -p imparo-cuda --no-default-features --features cuda-static
  --lib`: 29/29, including the native exact-route smoke with all six hit bits;
- `cargo test -p imparo-tune`: 18/18;
- `cargo test -p imparo-backend --lib`: 22/22;
- `IMPARO_CUDA_ARCHS=86 python dev_harness/build.py --features cuda`: release
  build fresh.

The attempted `cargo test -p imparo-tune --lib` exited one because this package
has no library target; it is not counted as a test result and was replaced by
the actual package test command above. `git diff --check` passed. The fixed
Rust 1.85 toolchain lacks the `cargo-fmt` proxy; direct `rustfmt --check` exposed
pre-existing formatting drift outside this change, so no repository-wide
mechanical rewrite was performed.

**Correctness and loader binding.** The fresh release executable SHA-256 is
`975808d9e0f14e98f28c059f8670138c2c86f83b5857edb99ed149767bdd3e92`.
Receipt producer v5 reran all six authenticated llama/FA q4_0 gates and every
gate passed: logit agreement at 128/449/512 and eight-step Decode agreement at
128/512/2000. The adjacent receipt SHA-256 is
`29f8a71e27e2711fe68a5f4cf0c788956f05d5efd5e04ba48a2741ac55550c3b`;
its backend fingerprint is
`254d75cd3c17c45bf655654c594e0a8d401434cdb78407aabd49f87f2648d202`.
With all inherited `IMPARO_*` variables cleared, gate mode absent and only the
production config/GPU/q4_0 route installed, the normal loader accepted the new
receipt. One 128-token forward exited zero and produced 262,144/262,144 finite
logits plus a 1,048,576-byte raw f32 witness.

**Decision.** The selector is now fail-closed on the cost and reachability of
every stage it controls. This strengthens future tuning decisions but does not
claim a speedup: no throughput benchmark was run. E4B/SM86 Decode retains its
existing parity evidence; 449-token Prefill retains its existing lead; stable
128-token and representative long-Prefill parity remain the open-source
performance gate. Slot 39 value zero remains the immediate rollback.

### E-20260901-37: exact-128 model-weighted tuner transaction

**Second selection defect.** E-36 made every slot-39-controlled stage visible,
but timed one D256 Attention, one FFN and one PLE. That 1:1:1 ratio is not the
E4B transaction: the model facts contain 42 layers, 35 with D256 Attention,
while every layer executes FFN and PLE. Equal stage weighting could still select
a route whose micro result disagreed with whole Prefill, even though all six
reachability bits were present.

**Repair.** The transaction now requires a complete per-layer attention table
whose length equals the model layer count and contains at least one D256 layer.
It walks that table in production order. A D256 layer contributes its real
Attention dispatch, including the real KV layer and window; every layer then
contributes FFN and PLE. PLE uses the real layer-derived vector offset rather
than row zero. The resulting E4B ratio is therefore 35 Attention to 42 FFN to
42 PLE without hard-coding 35 or 42. Missing/incomplete model facts refuse the
candidate before timing. Projection, residual and normalization work common to
both selector values remains outside the interval.

This is a tuner-only selection-policy correction. It does not change the legal
slot-39 values, production selector semantics, numerical route, CUDA kernel,
ABI or tuning-space contents. The v26 runtime config and its correctness receipt
therefore remain valid, but a pre-E-37 micro result is not accepted as performance
provenance for an open-source decision; the candidate must be reranked by the new
transaction and still pass the whole-engine bracket.

**Build-system failure and fix.** The first release validation compiled
`imparo-tune` successfully but `build.py` exited one: its freshness walker counted
`crates/imparo-cuda/tests/ffn_gated_down_contract.rs` as an input to every
production binary. Cargo correctly does not relink production binaries after an
integration-test-only edit, so the forced rebuild could never make their mtimes
newer. The walker now prunes crate-root `tests`, `benches` and `examples` while
continuing to track `src`, build scripts, manifests and CUDA native/catalog inputs.
A regression fixture makes newer files in all three excluded trees lose to the
actual production source.

**Evidence.** All of the following exited zero:

- exact route/source contract: 16/16;
- `imparo-tune` package tests: 18/18;
- static CUDA library tests: 29/29;
- complete Python harness after the freshness regression: 48/48;
- SM86-only release freshness build after the fix.

The release `imparo-tune.exe` SHA-256 is
`afe8befbe25cfda415552439854bcda195f947e49dca56d77611255da69a6f23`.
The production `imparo-forward.exe` remains byte-identical at
`975808d9e0f14e98f28c059f8670138c2c86f83b5857edb99ed149767bdd3e92`,
confirming this slice did not alter the runtime artifact. Consequently the E-36
receipt and backend fingerprint remain the current correctness authority and
were not pointlessly regenerated.

**Decision.** The exact-128 micro selector is now both stage-complete and
model-weighted. This removes another false-selection path but is not a throughput
claim. No performance benchmark was run; stable short-Prefill parity remains the
open-source gate, with slot 39 value zero as conservative rollback.

### E-20260901-38: exact-128 per-layer route evidence

**Third selection defect.** E-37 replayed the correct 35:42:42 E4B stage mix,
but the native receipt remained an OR-only six-bit mask. One successful D256
Attention, FFN or PLE layer could therefore hide a later layer's fallback: the
mask proved that every controlled stage ran somewhere, not that every expected
layer reached its final commit point.

**Repair.** The static tuner-lab evidence remains a 32-bit private surface, but
now packs the six stage bits plus saturated six-bit commit counts for D256
Attention, FFN and PLE. The tuner derives the expected counts from the model
facts; E4B expects 35, 42 and 42 respectively. Zero D256 layers, incomplete
per-layer facts or any count above the representable 63-layer bound fail closed
before timing. Attention is counted only after its exact batch32 launch, FFN
after the public Down output commits, and PLE after the direct public projection
commits. `time_us_checked` already forces one complete evidence-bearing
transaction per CUDA submission, so counts cannot be merged across repetitions.
The dynamic/plugin ABI stub remains zero and no public ABI, numerical route,
kernel arithmetic, config value or tuning-space version changed.

**Build and test evidence.** The following completed successfully on the RTX
3060 / SM86 host:

- exact-route source contract: 16/16;
- `imparo-tune` package tests: 19/19, including direct packing, zero-layer and
  overflow rejection tests;
- SM86 static CUDA library suite with `IMPARO_CUDA_EXACT128_SMOKE=1`: 29/29;
- shared backend tests: 22/22;
- Python build harness: 48/48;
- receipt sealer tests: 10/10;
- `IMPARO_CUDA_ARCHS=86 python dev_harness/build.py --features cuda`: fresh
  release after entering the Visual Studio x64 developer environment;
- `git diff --check`: pass.

Two setup failures are retained rather than rewritten as passes. The first
static-CUDA attempt exited 101 because NVCC could not find `cl.exe`; the same
command passed after entering `VsDevCmd.bat`. The first release attempt also
lost the VS PATH because cmd expanded `%PATH%` before the batch file ran; delayed
expansion preserved it and the standard freshness build then passed. Direct
`rustfmt --check` still reports pre-existing drift elsewhere in the already
dirty files, so no repository-wide mechanical rewrite was made.

**Correctness and identity.** The private counters change the static engine
bytes, so the old E-36 receipt was not reused. The release identities are:

- `imparo-forward.exe` SHA-256:
  `6249dfc79c09bdf74bcaabbce5cec9013b884a12744900353b38689d7dc23114`;
- `imparo-tune.exe` SHA-256:
  `e1329a9c3da8eb8813e3131ba07b755643720f82edcaaf9eea81035f9aa0a6f6`;
- receipt SHA-256:
  `f9742059cb36971333f605c2cf87ac4a6f29e3da3985da05947c6fc8af88f44f`;
- backend fingerprint:
  `533f425127c0bb92695c44a80996374c71ceae3783ccbd0a1fb96d6fddaa5632`.

The fixed authenticated llama/FA q4_0 suite passed logit agreement at
128/449/512 and eight-step recurrent Decode agreement at 128/512/2000. A first
sealer invocation pointed at a reduced reference directory and failed before
any model gate because its DLL inventory did not match the manifest; the
existing receipt remained untouched. The successful invocation used the exact
manifest-matching `4695f001` SM86 oracle bundle. With all inherited `IMPARO_*`
variables removed, correctness gate mode absent and only the production config,
GPU and q4_0 KV route installed, the normal loader accepted the new receipt.
One 128-token forward exited zero and produced 262,144/262,144 finite logits and
a 1,048,576-byte raw f32 witness.

**Decision.** The exact-128 selector can no longer rank a partially falling-back
model transaction as a complete candidate. This is selection/correctness
infrastructure, not a throughput claim. No performance benchmark was run.
Stable 128-token and representative long-Prefill parity remain the E4B/SM86
open-source admission gate; slot 39 value zero remains the immediate rollback.

### E-20260901-39: exact-128 full-logits Prefill Graph over the receipted sidecar DAG

**Reachability defect.** The existing Prefill-Graph laboratory path required
device argmax, while every real multi-token server/CLI Prefill requests full
logits. It was therefore unreachable for the workload it purported to test. In
addition, the Rust bridge rejected every FFN-sidecar forward before native Graph
admission. The old exact-key Graph result in E-27 did not include the now-stable
Q8 sidecar transaction and remains No-Go for its earlier mechanism.

**Repair and invariants.** Full-logits Prefill may now enter the exact-key
laboratory Graph path. The receipted exact-128 route requires two complete
ordinary forwards before capture: one builds the all-layer packed-Q4 hot set and
falls back, and the second proves every sidecar, scratch and launch-attribute
preflight can commit without allocation. Capture starts on the third matching
forward. During capture FFN and PLE only look up existing packed spans; a missing
span, scratch region, producer or direct projection marks the capture
incompatible, records a pending error and discards the Graph. Diagnostic CUDA
event scopes become no-ops while capture is active. Token count, absolute start
and output mode remain an exact Graph key, and input tokens are uploaded before
capture/replay rather than becoming a stale host-memory node.

The first live capture exposed two older Decode-only assumptions: reaching the
multi-token Attention family and the resident PLE norm/gather kernel
unconditionally invalidated any Graph. Those guards now distinguish Decode from
Prefill capture; the exact D256 batch32 Attention and resident PLE kernels are
normal capturable nodes. The explicit `IMPARO_CUDA_PREFILL_GRAPH_LAB` scheduling
wrapper and its trace-only companion are permitted by the correctness environment
validator, while every kernel/arithmetic override remains rejected. Production
env-off behavior is unchanged and the path remains laboratory-only.

**Negative evidence retained.** The first smoke correctly rejected the receipt
because the Graph laboratory environment was not receiptable; no bypass was used.
After classifying Graph as an exact-DAG scheduling wrapper, the next third forward
still exited one before capture because of the two unconditional invalidations
above. Both failures changed the implementation and are not reported as passes.
The final capture path emits an explicit discard record with CUDA end status,
Graph presence, compatibility and pending error for future failures.

**Build and correctness evidence.** On the RTX 3060 / SM86 host:

- exact-route/transaction source contract: 16/16;
- SM86 static CUDA library suite with the native exact-128 smoke: 29/29;
- SM86-only release freshness build: pass;
- authenticated llama/FA q4_0 receipt gates: logit 128/449/512 and recurrent
  eight-step Decode 128/512/2000 all pass;
- ordinary exact-128 full-logits forward: exit zero, 262,144/262,144 finite;
- one-process `--repeat 4` Graph smoke: forward one builds the hot set, forward
  two commits the warmed sidecar route, forward three emits `action=capture`,
  and forward four emits `action=replay` with `argmax=0` and
  `exact128_sidecar=1`;
- baseline and replay raw logits are byte-identical, both SHA-256
  `01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261`;
- reusable gate unit tests: 5/5;
- `dev_harness/prefill_graph_agree.py` independently reran the complete smoke
  and wrote the structured PASS receipt to
  `docs/evidence/cuda-sm86-e4b-open-source/`
  `exact128-v26-prefill-graph-agree.json`;
- `git diff --check`: pass.

Final release identities are:

- `imparo-forward.exe` SHA-256:
  `566c6533135d2eb39302dd1544bfb45b51c5de59b2cc4928b6215eb03d185853`;
- `imparo-tune.exe` SHA-256:
  `91d0fa17a2036b9f12d1eee780c5fade347defe9dc5a1aaadf6c65634ebfa9f`;
- receipt SHA-256:
  `46a08590b9beeb6990dd44636869f25f9eca25dedf9ce929e114a98afd8d5182`;
- backend fingerprint:
  `53d5cc497a147cbd2859b27add9d9c48454fc68016328c5f95d1e2bf7ab78c72`.

**Decision and rollback.** The combined exact-128 sidecar/Graph mechanism is now
a correctness-qualified laboratory candidate. It is not performance-admitted:
no throughput benchmark was run, so this result does not close the short-Prefill
open-source gate. Unset `IMPARO_CUDA_PREFILL_GRAPH_LAB` to return immediately to
the receipted ordinary path; set slot 39 to zero to remove the whole exact-128
bundle. The next permitted step is one final same-binary interleaved performance
and shape-regression bracket when performance measurement is explicitly requested.

### E-20260901-40: receipt-bound exact-128 production Graph policy (v27)

**Hypothesis and boundary.** E-39 proved byte-identical exact-key Graph replay but
left activation in `IMPARO_CUDA_PREFILL_GRAPH_LAB`. This change promotes only the
control mechanism, not its performance: slot 40 (`prefill_exact128_graph`) is a
safe-off scheduling knob declared after slot 39, and native production admission
requires both slot 40 = 1 and the already receipted exact-128 route. The explicit
laboratory override remains available for proving a safe-off config, but cannot
silently widen production to another shape. Generic sidecars remain outside Graph.

**Versioned contract.** CUDA tuning space advances from v26 to v27. Gate-suite v6
adds `prefill_graph_replay_agree_n128` after the six authenticated llama/FA q4_0
numerical gates, and sealer producer v6 records it in the adjacent receipt. The
Graph gate reads the config: an enabled production candidate must capture without
the lab override; a safe-off candidate may exercise the same wrapper only through
the explicit lab override. In both modes, one ordinary full-logits result and the
final repeat-four replay must be byte-identical and the trace must contain exactly
one capture and one replay. Every unrelated `IMPARO_*` numerical override is removed;
only the fixed gate/device identity is preserved.

**Verification evidence (no performance timing).** Exact commands and exits:

- Python Graph/sealer unit tests: 19/19, exit 0;
- exact route/Graph static transaction contract: 16/16, exit 0;
- CUDA knob tests: 9/9, exit 0;
- CUDA correctness policy tests: 10/10, exit 0;
- full `cuda-static` SM86 suite with `IMPARO_CUDA_EXACT128_SMOKE=1`:
  31 library + 2 diagnostic + 3 exact-449 + 16 FFN/Graph + 20 LFM2 tests,
  all pass, exit 0;
- SM86 release build: fresh, exit 0;
- fixed receipt suite: all seven gates pass, exit 0;
- production-on standalone Graph gate: `activation=config`, one capture, one replay,
  262,144 logits / 1,048,576 bytes, byte-identical SHA-256
  `01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261`.

The sealed candidate and standalone evidence are
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-exact128-v27.txt`, its adjacent
receipt, and `exact128-v27-prefill-graph-agree.json`. Final identities:

- `imparo-forward.exe`: `031e57970ea817a204b29e95a735a370b373bb5e233cc4c91b87582a687d2af8`;
- `imparo-tune.exe`: `71861d4fa0c2fef5b5b7034588b10a5108161b4d0509d9c1a762af3cb66e841c`;
- receipt: `19042debd5f655f0ab4d6b752ba09b35e0d2df90c34ae58675761b5f99ca09a4`;
- Graph evidence: `0a32c5bea41cd5cc903ac4e5cd4d32ae930ab924de799855ceab18cda5b068c6`;
- backend fingerprint: `48efde6a92a9f99300fe6971872e8f29f86124c4d19b1795da6a231b87848e9a`.

**Retained negative evidence.** A root-directory Python invocation failed because
`dev_harness` was not on `sys.path`; rerunning from the documented harness directory
passed. One Cargo invocation supplied two test filters and was rejected before tests;
the filters were then run separately and passed. A template call without
`IMPARO_CORRECTNESS_GATE=1` was rejected as designed; the sealer-equivalent call
passed. `cargo fmt` is unavailable as a Cargo subcommand in the pinned toolchain;
standalone `rustfmt --check` reports pre-existing formatting differences in the dirty
contract file, so no whole-file rewrite was performed. None of these failures is
reported as a successful gate.

**Decision and rollback.** The v27 production activation and receipt mechanism is
correctness-qualified. It does not prove short-Prefill parity because no throughput
benchmark was run. Set slot 40 to zero to disable only production Prefill Graph while
retaining the exact-128 numerical route; set slot 39 to zero to remove the complete
exact-128 bundle. Both defaults remain zero when no accepted receipt supplies values.

### E-20260901-41: end-to-end Graph admission semantics and final v27 reseal

**Selection defect and framework repair.** E-40 declared slot 40 as a benched value
on `PrefillFfnExact128`, but that micro transaction invokes Attention, FFN and PLE
directly and never crosses `prefill_prepare`. Values zero and one therefore timed the
same execution path, so a micro tuner could preserve or reject Graph for a reason that
was unrelated to Graph. The shared taxonomy now has `KnobCategory::EndToEnd` and
`SweepKind::External`: workflow/Graph policies remain registry-visible and value-bound,
but the micro screen, tuple pass and registry driver preserve the incumbent without
calling the timing sweep. Promotion requires an external whole-engine bracket plus the
normal correctness receipt. Slot 40 is the first policy using that reusable contract;
operator knobs remain `Benched` and other backends retain their existing behavior.

**Version and identity decision.** The config wire, legal values, runtime numerical
route and Graph execution path remain the v27 contract; only the tuner authority that
may choose slot 40 changed. v27 has not been released, so no second space bump was
manufactured. The release binaries and receipt were rebuilt/reissued after the source
stabilized. E-40's executable and receipt hashes are historical pre-framework
identities and are superseded by this entry; the unchanged config and Graph-evidence
hashes are expected because neither artifact's bytes changed.

**Verification evidence (no performance timing).** Exact results after the framework
repair:

- `imparo-backend`: 22/22 tests pass, exit 0;
- `imparo-tune --features cuda`: 19/19 tests pass, exit 0;
- complete `imparo-cuda --features cuda-static` SM86 suite with
  `IMPARO_CUDA_EXACT128_SMOKE=1`: 31 library + 2 diagnostic + 3 exact-449 +
  16 FFN/Graph + 20 LFM2 tests pass, exit 0;
- Graph/sealer Python tests from `dev_harness`: 19/19 pass, exit 0;
- SM86-only release build is fresh, exit 0;
- fixed correctness sealer: all seven gates pass and a new adjacent receipt is
  issued, exit 0;
- production-on standalone Graph gate: `activation=config`, exactly one capture and
  one replay, 262,144 logits / 1,048,576 bytes, byte-identical SHA-256
  `01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261`, exit 0.

Final identities:

- `imparo-forward.exe`: `c79689e4af896acd28dee58f5455a3fb76ba7c2025766471f5afdfdbd8bb7687`;
- `imparo-tune.exe`: `ca7d019e9b909dd9ef97fa944d02b2f9e2f75416e9e434dad08fb3b7054cec58`;
- v27 config: `b020a5851a6e114afff48684baca3def24fea8849ea797f07b3e1e01126a290b`;
- reissued receipt: `39902b838bf5f27ad1dc9deeaea3c2b501ad412ba2627c29b5d2e053ac806b1d`;
- standalone Graph evidence:
  `0a32c5bea41cd5cc903ac4e5cd4d32ae930ab924de799855ceab18cda5b068c6`;
- backend fingerprint:
  `48efde6a92a9f99300fe6971872e8f29f86124c4d19b1795da6a231b87848e9a`.

**Retained negative evidence.** The first shell did not expose `cargo` in PATH; the
fixed Rust 1.85 path was then used. The first CUDA test outside `VsDevCmd` failed because
NVCC could not find `cl.exe`; the same test passed in the documented VS environment.
That first test then exposed a test-parser defect: the old helper stopped at the
`applies` closure instead of reading the containing `KnobDecl`. The parser was repaired
to balance the whole declaration and the test passed. A root-directory Python run had
11 passing tests plus one import error because the Graph test imports its same-directory
module; the documented `dev_harness` run passed all 19. None of these setup/parser
failures is reported as a production or correctness pass.

**Decision and rollback.** Tuner authority and Graph replay correctness are now
qualified, but short-Prefill performance admission is still pending: no throughput
benchmark was run. Set slot 40 to zero to disable only production Prefill Graph while
retaining the exact-128 route; set slot 39 to zero to remove the entire exact-128
bundle. A final same-binary interleaved short/representative-long Prefill bracket is
still required before calling E4B/SM86 open-source-ready.
## Correctness and promotion gates

Performance data is considered only after the candidate passes the applicable
gates in this order:

1. **Structural/resource gate**: SM86-only selector, exact supported shapes,
   stable ABI/layout version, no stack/local/spill, bounded shared memory, and a
   reversible fallback.
2. **Dense contract gate**: every public destination buffer is materialized;
   epochs advance only after its producer launch; derived Q8/cache data matches
   that exact epoch.
3. **Per-op numerical gate**: compare the candidate with the established route,
   including 449 tails and 512 full tiles. Q8-producing paths also compare quant
   bytes and scale bits with the authority producer.
4. **Model gate**: fixed logit agreement first, then recurrent decode agreement,
   determinism, KV resume/grid/scramble, and graph identity/paged replay where
   the changed path can affect them.
5. **Memory/concurrency gate**: Compute Sanitizer memcheck, initcheck, racecheck,
   and synccheck. Instrumentation-induced application failure must be reported,
   never rewritten as a pass.
6. **Performance gate**: same binary A/B/B/A or interleaved CUDA-event samples,
   followed by whole-engine Prefill/Decode. Record every raw leg and medians.
7. **Receipt gate**: bind source diff/commit, build identity, model, device, SM,
   driver, numerical route, shape, environment, and all preceding evidence.

For this session, the later R64 down sweep supersedes the earlier positive
grid-60 observation and leaves every tested forced grid at **No-Go**. The
direct-register token-pair Q8 route has passed q4_0 `logit_agree` at
128/449/512/2000 tokens with tolerance 0.75. The complete sidecar-only FFN
transaction has a newer positive four-leg screen and has passed the listed
q4_0 logit, eight-step recurrent Decode, and determinism gates. Its
approximately +1.011% difference remains within local noise, however, and
failure injection, binary identity, and all four sanitizer classes remain
pending; it does not establish a default-admitted winner.

## Unresolved receipt index

Never guess or reconstruct a missing identity from a neighboring experiment.
Close these items only from the original command output or a new complete run:

| Experiment | Unresolved evidence |
|---|---|
| E-20260830-07 | Exact selector names/values for the supplied follow-up measurements |
| E-20260830-10 | Exact selector names/values for the rejected combined route |
| E-20260830-11 | Exact command/environment receipt for the initial direct-Q8 observation |
| E-20260830-14 | Dirty-diff, executable/backend hashes, build profile/toolchain receipt, failure injection, four sanitizer classes |
| E-20260830-15 | Both engine hashes, model hash, complete commands, raw per-repeat samples, token-parity receipt |
| E-20260831-16 | Dirty-diff and binary/backend hashes; broader gates intentionally not run after No-Go |
| E-20260831-17 | Dirty-diff and binary/backend hashes; broader gates intentionally not run after No-Go |
| E-20260831-19 | Complete dirty-tree manifest, raw per-repeat receipt and load-time telemetry; broader gates intentionally not run after No-Go |
| E-20260831-20 | Final lab source/binary hashes and persisted raw JSON output; engine/model/sanitizer gates intentionally not applicable after standalone No-Go |
| E-20260831-21 | Final-binary four-sanitizer binding and shared bank counters; NCU failed with `ERR_NVGPUCTRPERM`; raw timing JSON was emitted to the terminal but not persisted as a tracked receipt |
| E-20260831-22 | Q4-weight-correlated record/schema remains pending; E-20260831-23 proves the fixed activation ladder is already No-Go and therefore stops that implementation before weight analysis |
| E-20260831-31/32 | Dirty-diff hash and final exact-route interleaved performance/shape-regression bracket remain pending. E-32 closes full model gates, final binary hash, production receipt loading, priority-reclaim/rebuild byte identity and both kernel/full-model four-class sanitizer matrices. |

## Reusable experiment template

Copy this section for every new candidate.

```markdown
### E-YYYYMMDD-NN: short candidate name

**Identity and lineage**
- Base experiment ID and enabled-route dependency stack:
- Supersedes / superseded by:
- Source commit, dirty-diff hash, executable hash, backend artifact hash:
- Control experiment/route/selector, candidate selector, and same-binary status:
- Raw receipt location, owner, and public/internal/commercial classification:

**Hypothesis**
- What bottleneck is being removed?
- What measurable result would falsify the hypothesis?

**Invariants**
- Numerical class and exact accumulation/rounding order:
- Buffer ownership, cache epoch, layout/ABI version:
- Tail, seam, workspace, OOM, graph, probe, and fallback behavior:
- Platforms and paths that must remain unaffected:
- Public/private buffer table and commit point, if transactional:
- Pre-commit refusal and post-commit failure behavior:

**Environment**
- Model/weights hash and quantization:
- GPU, SM count, driver, CUDA toolkit, clocks/power state:
- Prompt shape, token counts, KV types, FA state:
- Complete environment matrix, including explicitly unset variables:
- Exact commands, warmup, ordering, and profiler state:

**Resources**
- Registers/thread:
- Static/dynamic shared memory:
- Stack/local/spill:
- Threads/CTA and resident CTA/warps per SM:
- Persistent/transient device memory:
- Sidecar budget/used/headroom, admission list, pack time, hits/misses/rejects:
- Selector arithmetic, buffer bounds, and integer-overflow cases:

**Single-op evidence**
- Control raw samples and median:
- Candidate raw samples and median:
- CUDA-event schedule, warmup, repeats, dispersion, and declared noise floor:
- Event share, predicted whole gain, and generated-code/resource receipt:

**End-to-end evidence**
- Prefill raw legs and median:
- Decode raw legs and median:
- Interleaving order, measured whole gain, and prediction gap:
- Cross-engine bracket, if applicable:

**Correctness evidence**
- Dense output (PASS / FAIL / PENDING / N/A + receipt URI):
- Derived Q8/layout bytes and scales (PASS / FAIL / PENDING / N/A + receipt URI):
- Logit/decode/determinism/KV/Graph gates (PASS / FAIL / PENDING / N/A + receipt URI):
- Unsupported/OOM/probe/graph fallback and injected pre/post-commit failures
  (PASS / FAIL / PENDING / N/A + receipt URI):
- memcheck/initcheck/racecheck/synccheck commands, exits, status, and receipt URI:

**Decision**
- Admitted / Validated lab / Promising / No-Go / Observation:
- Reason and remaining evidence:
- Code state: retained active / laboratory-only / unreachable / deleted:

**Rollback**
- Selector/feature flag and conservative route:
- Evidence that rollback/fallback was exercised:

**Migration experience**
- What generalized to other shapes/SMs?
- What failed, and what should not be tried unchanged again?
- Which policy belongs in tuning rather than hard-coded source?

**Next hypothesis / stop rule**
- Next causal hypothesis and the mechanism counter/event that must move:
- Smallest experiment that distinguishes it from the rejected explanation:
- Expected gain, rejection threshold, and conditions that permit a retry:
```

## Cross-experiment lessons

1. Preserve dense semantics even when a derived Q8 consumer is adjacent.
2. Count CTA independence and barrier cohorts, not only resident warps.
3. Express Stream-K grids relative to SM count; validate every ownership/fixup
   schedule numerically before interpreting speed.
4. Measure fusion across the producer-consumer boundary. A fast isolated kernel
   can lose after conversion, cache publication, or fallback costs are included.
5. Treat memory budgets as runtime policy with safe fallback, never as a GPU
   model constant.
6. Keep No-Go results. They narrow the search space and are part of the engine's
   maintainability story.
7. A same-session ratio is evidence, not a release claim, until the complete
   identity and correctness receipt exists.
8. A fused backend hook is a transaction. Returning false is legal only before
   any public output launch commits; a post-commit failure must be forward-fatal
   and must never enqueue the fallback.
9. Exact scheduling is not removed arithmetic. The exact-449 AOT route removed
   runtime boundary division and one third of fixup CTAs, yet moved whole
   Prefill only +0.33%; require a mechanism with a larger Amdahl upper bound.
10. Shared-layout reduction is not core-work reduction. The packed-shared S8
    candidate saved 256 static shared bytes but retained eight S8 MMA operations,
    used eight more registers, and measured no faster at either tested shape.
    Require a causal counter and balanced cache-position evidence before
    iterating more layouts with unchanged arithmetic.
11. Synthetic quantization coverage is not deployable coverage. The fixed real
    E4B activation ladder covered only 18.42%, with Down strata at 7.63--12.12%,
    so a synthetic 66.7% result could not justify even writing the W4A4 kernel.
    Capture real intermediates and stop before code generation when an upstream
    coverage bound already makes the end-to-end target impossible.
### E-20260901-42: exact-128 R96 Full-K Down occupancy laboratory

**Hypothesis and isolation.** The retained exact-128 Down kernel owns 128 output
rows per CTA, so E4B's 2,560 output rows launch only 20 CTAs on the 30-SM RTX
3060 verification device. The new `r96` laboratory schedule owns 96 rows per
CTA and launches 27 CTAs: 26 full tiles plus one explicitly guarded 64-row tail.
It retains one full-K owner and the established K32 MMA and f32 scale order for
each output row. Only independent rows are repartitioned. The selector is
`IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_128_LAB=r96` under the existing sidecar
laboratory parent. The production exact-128 selector still chooses `Rows128`;
no ABI, tuning-space value, receipt class, Metal path or non-SM86 route changed.

**Implementation contract.** `compute_segment` now accepts an explicit kernel
warp count and an optional active-row guard while preserving the compiled R64
and R128 defaults. R96 uses six warps, 96 staged rows and two row fragments.
The final CTA zero-stages rows 64..95 and the epilogue refuses to publish those
rows. Grid construction is ceil-divided only because R96 does not divide 2,560;
the existing R64/R128 grids are unchanged by that arithmetic. A static contract
proves the selector is default-off, the tail has both load and store guards, and
the receipt-backed production selector remains R128.

**Generated resources.** CUDA 12.9 `cuobjdump --dump-resource-usage` reports
150 registers/thread, zero stack and zero local memory for both R96 epilogues.
R96 launches 192 threads with 32,256 bytes dynamic shared memory. The R128
control reports 128 registers/thread, zero stack/local, 256 threads and 36,864
bytes dynamic shared. R96 therefore buys a denser first wave and less work per
CTA at a real register-cost tradeoff; this structural evidence is not a speed
claim.

**Verification without performance timing.** The first static CUDA invocation
failed before compilation because PATH was reassigned after `VsDevCmd` and
removed `cl.exe`; adding Rust/CUDA before entering the VS x64 environment fixed
the setup. The corrected SM86-only verification then passed:

- `ffn_gated_down_contract`: 17/17, exit 0;
- complete `cuda-static --tests` suite with the exact-128 smoke enabled: 31
  library + 2 diagnostic + 3 exact-449 + 17 FFN/Graph + 20 LFM2, exit 0;
- `IMPARO_CUDA_ARCHS=86 python dev_harness/build.py --features cuda`: fresh
  release, exit 0. The 21,994,496-byte engine SHA-256 is
  `c9628298822e2efabf435a8cbbfd53d770d58b709c488f4e11123dbbb7ef03c8`.

The decisive model test used the same binary, E4B file, q4_0/q4_0 KV, exact
128-token input and two repeats for each leg. Repeat one prepared the complete
packed-Q4 model hot set. On repeat two, both R128 and R96 attempted, admitted
and committed all 42 layers and produced 262,144 finite logits. Their complete
1,048,576-byte f32 dumps were byte-identical, both SHA-256
`db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651`.
Structured evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-r96-down-correctness.json`.

**Decision and rollback.** **Correctness-qualified, safe-off laboratory
candidate.** No throughput benchmark was run, so R96 is not promoted and does
not change E4B open-source admission. Unset the R96 environment selector or use
`r128` to restore the control. An explicitly requested same-session,
interleaved whole-Prefill bracket is still required before any performance
decision.
### E-20260901-43: R96 register-pressure and residency screen

**Question.** E-42's R96 Down candidate uses 150 registers/thread. Its 32,256
bytes of dynamic shared memory can theoretically fit three CTAs on SM86, so
this screen asked whether compiler occupancy directives could make three
192-thread CTAs register-resident without changing arithmetic. No throughput
timing was run and none of the temporary variants was enabled for model
execution.

**Rejected launch-bounds constraint.** `__launch_bounds__(192, 3)` compiled to
96 registers/thread but introduced a 40-byte per-thread stack frame. That
violates the candidate's zero-stack/zero-local structural gate, so it was
reverted before any kernel launch. Forcing more occupancy by spilling the 64
cross-K-stage f32 accumulators is not an admissible optimization hypothesis.

**Tail-guard attribution.** The R96 load-side tail guard was temporarily removed
for a compile-only resource screen. The kernel remained at 150 registers,
zero stack and zero local memory. The final 32 inactive rows are therefore not
the source of register pressure; padding the packed sidecar or weakening the
guard would add memory and correctness complexity without a supporting resource
mechanism. The guard was restored before any runtime test.

**Rejected register cap.** `__maxnreg__(112)` produced exactly 112 registers,
zero stack and zero local memory. Direct CUDA 12.9 SASS parsing showed 1,840
instructions and three barrier operations for the Down epilogue, versus 1,784
instructions and three barriers for the 150-register control: 56 additional
instructions, approximately 3.1%. More importantly, exact-128 R96 launches 27
CTAs on the 30-SM verification GPU. There is no second or third CTA per SM in
that first wave, so the extra residency made possible by the cap cannot supply
the claimed mechanism for this shape. The cap was rejected and reverted rather
than treating a smaller register number as a speed result.

Three `cuobjdump --extract-text` attempts (short name, complete name, and `all`)
returned exit 1 on this Windows host. They are retained as tooling failures.
The successful fallback parsed only the requested function interval from
`cuobjdump --dump-sass`; no performance counter or wall-clock output was used.

**Final-state proof.** The retained source returned to E-42's SHA-256
`97299576bfc0cbc341abbb53abc2019e64c02e8f183b398321a3394d1f010cd4`.
The complete SM86 static suite passed again: 31 library + 2 diagnostic + 3
exact-449 + 17 FFN/Graph + 20 LFM2 tests. The standard SM86 release build is
fresh; the 21,994,496-byte engine SHA-256 is
`6baaac8fb35779b79d4f6ffbe4777fd98d74574904c6cad9eeb6ae16d8b2362b`.
On that exact final binary, R128 and R96 again attempted/admitted/committed all
42 model layers and emitted byte-identical 1,048,576-byte full-logit dumps,
both SHA-256
`db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651`.

**Decision.** The original zero-stack R96 candidate remains safe-off and
production remains R128. Compiler occupancy directives and tail padding are
now closed for this exact-128 hypothesis. A future structural candidate must
reduce the lifetime or count of the 64 f32 accumulators, or change the CTA/grid
decomposition with a mechanism that exact-128 can actually exercise. Structured
evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-r96-register-screen.json`.
### E-20260901-44: R96W24 token-group-partitioned Down laboratory

**Causal hypothesis.** E-42's R96 schedule improved the exact-128 first wave
from 20 to 27 CTAs, but each of its six warps retained all four 32-token groups
across every K stage. That required 64 live f32 accumulators per thread and
compiled to 150 registers/thread. A rejected T64 split could divide the token
groups without duplicating Q8 activation bytes, but it would launch two CTAs per
row tile and therefore read, unpack and stage every Q4 Down weight twice. It was
rejected before implementation. `r96w24` instead keeps one R96 CTA and assigns
one warp pair to each `(32-row group, 32-token group)`: three row groups times
four token groups times two warps equals 24 warps. Q4 weights and Q8 activations
are still staged once per CTA, total MMA work is unchanged, and each output keeps
the established K32 accumulation order. The per-thread accumulator falls from
64 to 16 floats.

**Isolation and public boundary.** The candidate is available only through
`IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_128_LAB=r96w24` under the existing
`IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB=1` parent. The receipt-backed exact-128
selector still maps to `Rows128`; no ABI, tuning-space value, receipt class,
Metal path, non-SM86 path or production default changed. `DirectRows::Rows96Warp24`
is a schedule sentinel rather than a public shape. Unset the selector or choose
`r128` for immediate rollback.

**Generated-code gate.** A real `--features cuda` CUDA 12.9 SM86 build reports
78 registers/thread, zero stack and zero local memory for both R96W24 epilogues,
versus 150 registers/thread for the retained six-warp R96 control. Targeted SASS
inspection finds no `LDL` or `STL`. The candidate launches 768 threads (24 warps)
with 32,256 bytes dynamic shared memory. Thus its 59,904-register CTA fits the
65,536-register SM86 file without spill and exposes 24 active warps on each of
the 27 first-wave SMs. These are structural facts, not a throughput claim.

**Verification without performance timing.** The complete SM86 CUDA suite
passed with `cargo test -p imparo-cuda --features cuda --lib --tests`: 31 library
+ 2 diagnostic + 3 exact-449 + 18 FFN/Graph + 20 LFM2 tests, exit 0. The standard
`IMPARO_CUDA_ARCHS=86 python dev_harness/build.py --features cuda` release build
is fresh, exit 0. `imparo-forward.exe` is 22,062,080 bytes with SHA-256
`f1b6ba61719889cdf3b66a15f593d7c9dba6bd1cf010dca50b934b3e059915ec`.

The model check used that same binary, the 4,215,695,776-byte E4B Q4_K_XL file,
q4_0/q4_0 KV, the established exact-128 token vector and two repeats per leg.
Repeat one prepared the complete packed-Q4 model set. On repeat two, R128 and
R96W24 each attempted, admitted and committed all 42 layers. Both produced
262,144 finite logits, and their complete 1,048,576-byte f32 dumps were
byte-identical with SHA-256
`db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651`.

**Retained setup failures.** The first contract invocation omitted
`--features cuda`; it passed Rust static checks but generated no CUDA object.
The missing `w24` symbol in `cuobjdump` prevented that cached object from being
misclassified as evidence. A package-scoped clean followed by the correct CUDA
feature produced the resource result above. The first PowerShell model verifier
also exited during parsing before launching the engine; the corrected Python
verifier then completed both legs. Neither setup failure is a kernel failure.

**Decision and next gate.** **Correctness-qualified, structurally promising,
safe-off laboratory candidate.** It closes E-43's stated requirement to reduce
the lifetime/count of the 64 f32 accumulators without spilling, but no performance
benchmark was run. It therefore does not change E4B open-source admission or
replace production R128. Only a user-requested same-session, interleaved whole-
Prefill bracket may decide whether the mechanism moves the current 128-token
gap. Structured evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-r96w24-down-correctness.json`.
### E-20260901-45: exact-divisor R80 Down laboratory

**Causal hypothesis.** E4B Down has 2,560 output rows. R128 launches 20 CTAs
and leaves ten of the RTX 3060's 30 SMs without a first-wave block; R96 launches
27 CTAs but has a guarded 64-row tail. Eighty rows divide 2,560 exactly, yielding
32 independent CTAs with no inactive row work. `r80` assigns one warp pair to
each 16-row group, so a CTA uses ten warps and retains all four token groups per
owner. It keeps the same full-K owner, K32 MMA/f32 scaling order and one Q4/Q8
global staging pass as the established route, while reducing the per-thread
accumulator from R96's 64 floats to 32.

**Isolation.** The candidate is enabled only by
`IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_128_LAB=r80` under the existing sidecar
laboratory parent. Production exact-128 remains `Rows128`; no ABI, tuning space,
receipt class, Metal workflow, non-SM86 path or default selector changed. The
launcher retains guarded reads and writes even though the E4B shape divides
exactly, so accidental non-divisor use cannot publish an out-of-range row.

**Generated resources.** A real CUDA 12.9 SM86 source build reports 92
registers/thread, zero stack and zero local memory for both R80 epilogues;
targeted SASS contains no `LDL` or `STL`. R80 launches 320 threads (ten warps)
with 29,952 bytes dynamic shared memory. One CTA therefore consumes 29,440
registers; two use 58,880 registers, 59,904 shared bytes, 640 threads and 20
warps, all within SM86's per-SM limits. Resource arithmetic therefore permits
two-CTA residency, while the 32-block grid supplies work to every SM in the
first wave. This is a structural mechanism, not a throughput result.

**Verification without performance timing.** The complete
`cargo test -p imparo-cuda --features cuda --lib --tests` suite passed: 31
library + 2 diagnostic + 3 exact-449 + 19 FFN/Graph + 20 LFM2 tests, exit 0.
The standard SM86 release build is fresh, exit 0. The resulting 22,134,272-byte
`imparo-forward.exe` has SHA-256
`8972c0f339a9ed325c8800747f9ae209a175582651d7580af7f418bf7b10edec`.

The same-binary model comparison used E4B Q4_K_XL, q4_0/q4_0 KV, the established
128-token vector and two repeats. On the second repeat both R128 and R80
attempted, admitted and committed all 42 layers and produced 262,144 finite
logits. Their complete 1,048,576-byte f32 outputs were byte-identical with
SHA-256
`db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651`.

**Decision and rollback.** **Correctness-qualified, structurally promising,
safe-off laboratory candidate.** R80 and E-44's R96W24 express different
tradeoffs and are deliberately not ranked without whole-Prefill evidence. No
performance benchmark was run, so production stays R128 and E4B open-source
admission does not change. Unset the selector or choose `r128` to roll back.
Structured evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-r80-down-correctness.json`.
### E-20260901-46: single-authority Direct Down schedule framework

**Maintenance defect and boundary.** E-42, E-44 and E-45 added independently
qualified exact-128 Down schedules, but selector spelling, profiler identity,
launch geometry and launch-attribute configuration were repeated across the
environment parser, preflight and launcher. That duplication made a future
model/SM candidate able to drift without changing the tuner or receipt. E-46
introduces `DirectScheduleDescriptor` as the single metadata authority for the
existing R128, R80, R96, R96W24 and R64 schedules. The descriptor alone is not a
tuning-space value and cannot promote a candidate. The receipt-backed exact-128
route still resolves to R128, the legacy `Environment` sentinel retains the R64
laboratory compatibility switch, and Metal, non-SM86, ABI and numerical-route
contracts are unchanged.

**Fail-closed selection.** `parse_direct_schedule` accepts only a descriptor's
exact selector. An invalid laboratory selector stays at `Environment`; an invalid
ambient selector therefore cannot launch a candidate, while an invalid override
on the receipt-backed path preserves its already selected R128 incumbent. The
preflight and launch path both consume the same descriptor and centralized
configuration dispatcher. Static contracts prove the production R128 mapping,
the descriptor-only laboratory boundary and absence of tuner promotion.

**Verification without performance timing.** `git diff --check` and the focused
20-test FFN/Graph contract passed. A real CUDA 12.9 SM86 `--features cuda`
compile of that contract completed in 1m36s, exit 0. The complete CUDA suite
then passed 31 library + 2 diagnostic + 3 exact-449 + 20 FFN/Graph + 20 LFM2
tests, 76 total, exit 0. The standard SM86 release build completed in 1m39s,
reported every engine binary fresh and produced a 22,134,272-byte
`imparo-forward.exe`, SHA-256
`5646f0f64a6236f6a271d396a2d55b0d2bb33489a488132df4bf125f48be5bba`.

The same-binary E4B non-timing check compared descriptor-selected R128 and R80
with q4_0/q4_0 KV, the established 128-token vector and two repeats. On the
second repeat both legs attempted, admitted and committed all 42 layers and
emitted 262,144 finite logits. Their complete 1,048,576-byte f32 outputs were
byte-identical, SHA-256
`db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651`.
An initial evidence command looked for the nonexistent `imparo_engine.dll` and
failed before hashing; it was corrected to the actual `imparo-forward.exe`.
This was an evidence-path failure, not a build or runtime failure.

Final source SHA-256 values are
`398054bba8cd9000384877aa05709df0aae0b665a43ce1579294375ecd18b551`
for `q8_ready_batched_r2.cuh`,
`3c10d4f2724ab62b7cd00c37692875757589600e4a87bec3cdb18a9dbc01435d`
for `imparo_cuda.cu`, and
`8e1695d96f517c595977868a395cfc28b229d3deb0a764cbde66902830bcfbda`
for `ffn_gated_down_contract.rs`.

**Decision and rollback.** The framework refactor is correctness-qualified and
retained. It does not rank R80/R96/R96W24, change production R128 or close E4B
open-source admission; no performance benchmark was run. Reverting the
descriptor consumers to the prior explicit parser/configuration dispatch is the
code rollback. Unsetting the laboratory selector remains the immediate runtime
rollback. Structured evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-direct-schedule-framework.json`.
### E-20260901-47: Dual-RMS/Q8 direct contract and shared-reduction race repair

**Why this candidate.** E-04 measured a historical 1.85% whole-Prefill gain
from combining the adjacent post-attention residual normalization, FFN
normalization and MMA-ready Q8 producer. That positive result remained a lab
signal because it proved only final logits. E-47 adds a dedicated non-timing
SM86 harness that compares the fused kernel against two separate control
kernels preserving the established per-thread ownership, reduction order,
f16-rounded Q8 scales and padded ready-layout bytes. It covers 9, 128, 449 and
512 tokens plus zero-token and wrong-SM fallback. The route remains behind
`IMPARO_CUDA_PREFILL_DUAL_RMS_Q8_READY_LAB`; no tuner value, receipt, ABI,
Metal path, non-SM86 path or production default changed.

**Failure discovered before admission.** The first ordinary harness run was
byte-exact for both dense outputs, Q8 codes and Q8 scales at every shape. Its
first memcheck and initcheck also exited zero. The first racecheck, however,
made every shape diverge and returned one. The racecheck log was decisive: it
reported a read at `block_sum` line 62 racing a write at line 60, with four
displayed error groups and `RACECHECK SUMMARY: 4 hazards displayed (4 errors,
0 warnings)`. The fused kernel calls the same block reduction twice and reused
the same 32-float shared array. The first reduction synchronized publication but
did not prevent a faster warp from writing the second reduction while warp zero
was still reading the first reduction's per-warp values.

**Repair.** `block_sum` now completes the final warp XOR, executes one block
barrier and only then returns. This barrier is specifically a shared-slot reuse
boundary; it does not alter arithmetic ownership or introduce a global seam.
The native route also gains a default-off, post-commit, non-timing trace
(`IMPARO_CUDA_PREFILL_DUAL_RMS_TRACE`) so whole-model evidence can prove that a
successful output actually used this path. A three-test Rust source contract
pins the post-read barrier, graph/probe/safe-off boundary, post-launch commit
order, trace position and harness shape/output coverage.

**Verification after repair.** The rebuilt SM86 harness again produced
byte-identical `mid`, `norm`, Q8-code and Q8-scale arrays at all four token
counts, with zero maximum dense delta. All four final Compute Sanitizer logs are
retained: memcheck zero errors, initcheck zero errors, racecheck zero hazards/
errors/warnings and synccheck zero errors. The actual SM86 cubin reports 34
registers/thread, zero stack and zero local memory for the fused kernel; dynamic
shared memory remains the launch-time 10,368-byte contract. The complete CUDA
suite passed 31 library + 2 diagnostic + 3 Dual-RMS + 3 exact-449 + 20
FFN/Graph + 20 LFM2 tests, 79 total, exit zero. The standard SM86 release build
completed in 1m41s, reported all binaries fresh and produced a 22,134,272-byte
`imparo-forward.exe`, SHA-256
`05fcaee4cd7e3f317d6556023c9a5bae2e391c691a6faf8e1d51eda22c8205fa`.

The non-timing E4B check used q4_0/q4_0 KV, the established 128-token vector and
two repeats per leg. With the route disabled, the trace reported zero hits. With
it enabled, the two 42-layer forwards reported exactly 84 committed Dual-RMS
hits, and the warmed FFN transaction consumed the ready-Q8 input on 42/42
layers. Both routes emitted 262,144 finite logits and their complete
1,048,576-byte f32 outputs were byte-identical, SHA-256
`db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651`.

Final source SHA-256 values are
`ecc7d8a2c0bd8f0c9d851388b6b7a7b61fe97a4eea41f1a08e25dd25ac1aef6e`
for `rms_norm_add_dual_q8_ready.cuh`,
`9ca91eb405f360870e5bebda882281eb7f3d9bbcce60fc0e8da5bfe1a7262635`
for `imparo_cuda.cu`,
`f19696dfc700b2fe12f226799522b610f8cb8abe232f80adfad1e32c30d3bfd5`
for the native harness, and
`7217cc4f3e36a96852c5c575da127b28aee1bde7ff97ff751a1ba67b736cbc1c`
for the Rust contract.

**Retained setup and evidence failures.** The first sanitizer command assumed
the executable lived under CUDA's `bin`; the actual toolkit path is
`CUDA\v12.9\compute-sanitizer\compute-sanitizer.exe`, so the first command
failed before launching the target. The first focused Cargo command also lacked
the Rust toolchain in PATH and failed before compilation; the fixed absolute
toolchain invocation passed 3/3. The failing pre-fix race log was overwritten by
the required final log on rerun; its exact summary and source lines are retained
here and in the structured evidence, but it is not misrepresented as a retained
raw log.

**Decision and rollback.** **Correctness-qualified, sanitizer-clean,
safe-off laboratory candidate.** This closes E-04's direct dense/Q8 and kernel
sanitizer gaps and repairs a real race, but it does not promote the route: no new
performance A/B was run, full-model sanitizer coverage and a versioned
tuner/receipt value remain future admission work. Unset
`IMPARO_CUDA_PREFILL_DUAL_RMS_Q8_READY_LAB` for immediate rollback. Removing the
post-read barrier is not a valid rollback because racecheck proves the old code
unsafe. Structured evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-dual-rms-q8-race-repair.json`.
### E-20260901-48: short-Prefill fixed cost and split-Graph input ownership

**Attribution.** Formal 128/449 Sidecar+Dual-RMS receipts imply a two-point
diagnostic intercept of 25.27 ms for Imparo versus 6.69 ms for llama, while Imparo
has the better per-token slope. Admission-time Q4 prepack is persistent and does not
explain steady 128-token requests. Nsight attributes roughly 1043 host kernel
submissions to each ordinary 128-token forward, and the common one-token-tile MMQ
grid leaves some E4B projections at about 67% first-wave SM utilization.

**Rejected implementation.** Capturing the entire forward was fast in an internal
screen and byte-exact for one repeated prompt, but failed the mandatory varied-input
check. Five of six prompts happened to share the expected top-1; the sixth should
return `t` and instead replayed the capture prompt's `,`. The Graph-off route returned
`t` for that prompt 10/10 times, ruling out top-1 instability. Exit code 17 was
retained as a real correctness failure, not described as success.

**Root cause and repair.** On the 6 GiB RTX 3060, E4B keeps part of its large token
and PLE tables in the streamed tier. Whole-forward capture therefore recorded
request-dependent host-staged rows. The shared Backend contract now exposes a
post-embedding Prefill-body boundary with a conservative no-op default; Metal and
other backends retain their workflow. CUDA reuses the existing FFI/ABI, runs token
embedding and PLE per request, synchronizes the prefix once at capture, and records
only the Transformer body. Replay remains ordered after the fresh prefix on the same
stream.

**Verification.** The CUDA-feature library suite exits zero with 32/32 tests, the
generic model suite exits zero with 42/42 tests, and the release server build exits
zero after a real SM86 NVCC compile. Six varied 128-token requests match Graph-off
choice hashes 6/6; the previously failing token-2005 request matches SHA-256
`3eb11bb919f1adbb12bae26c74a053394be45f39e66b4363340f6a764fd6e12f`.
Trace reports one capture followed by three `boundary=post-embedding` replays. A
dev-only `--vary-last`/`IMPARO_LOGITS_DUMP_DIR` forward witness then compared all
262,144 f32 logits for each prompt: all six 1,048,576-byte Graph/control pairs are
byte-identical. Structured hashes are in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-split-prefill-graph-varied-logits.json`.

**Full-model sanitizer.** The q4_0/q4_0 Sidecar + Dual-RMS split-Graph target ran
four varied-input forwards under each Compute Sanitizer tool: two ordinary warm
forwards, one capture/launch and one `boundary=post-embedding` replay. Memcheck,
initcheck and synccheck each exited zero with zero errors. Racecheck completed in
about 60 minutes, exited zero and reported `0 hazards displayed (0 errors, 0
warnings)`. Raw logs are retained under
`dev_harness/results/sidecar-graph-lab-20260901/split-graph-dual-rms-*.log` (the full-combination memcheck retains the earlier `split-graph-memcheck.log` name). An earlier 47-minute racecheck omitted the Dual-RMS selector and is retained only as a Sidecar-Graph baseline, not complete-candidate evidence.

**Decision.** Retain the split-Graph framework as a default-off laboratory slice.
The unsafe whole-forward timing is mechanism evidence only and cannot support
promotion. Full-logit varied-input proof and full-model Compute Sanitizer coverage
are closed; fixed-distribution gates, corrected-path timing and versioned receipt
identity remain open.
### E-20260901-49: exact-128 Token64 tuner slice and end-to-end authority

**Candidate and invariant.** The default-off Token64 candidate splits each exact
J128 MMQ token tile into two J64 CTAs while retaining the parent J128 logical
tile and its canonical Stream-K seam. It is restricted to SM86 exact-128,
aligned, non-epilogue local-Q 2560 -> 2048 and FFN-down 10240 -> 2560 shapes.
The generated J64 instance reports 248 registers/thread, zero stack, zero local
memory and 31,744 bytes static shared; the launch supplies 48,128 bytes dynamic
shared. The original J128 instance remains the safe default.

**Initial laboratory signal.** A same-binary four-control/four-candidate
full-logit check produced eight 1,048,576-byte outputs with SHA-256
db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651.
The route trace reported 112 Token64 launches: 70 majority local-Q and 42
warm-path FFN-down launches. Wall time minus each process's reported load time
suggested about 1.047x directionally. This was explicitly nonformal,
noninterleaved evidence and was not treated as admission.

**Versioned tuner integration and setup failure.** CUDA space v28 adds atomic
slot-39 value 2, keeps zero safe-off and keeps value 1 as the established
exact-128 Attention/FFN/PLE route. The tuner transaction now includes the real
named attn_q offset once for each of the 35 D256 layers and packs a 6-bit
Token64 commit count at evidence bits 24..29. The first run failed with
misaligned address: the synthetic blob spread tensors at arbitrary byte offsets
although the MMQ contract assumes a CUDA-aligned tensor base. Aligning each
synthetic tensor base down to 256 bytes fixed the actual defect; ordinary
real-model inference was unaffected and passed before the fix.

The corrected, non-writing exact-only tuner completed with exit zero. Its
candidate-emission run measured safe-off 63,266.2 us, value 1 56,885.2 us and
value 2 58,590.1 us; it selected value 1. This rejects Token64 value 2 against
the existing route on this RTX 3060, despite both beating the synthetic safe
proxy.

**Real-model contradiction and No-Go.** An isolated unreceipted value-1 config
was then applied only under IMPARO_CORRECTNESS_GATE=1. Four safe-off dumps were
internally deterministic at SHA-256
db0fae18fd154fa7e7ca012f4837e6b95ef7231049d30a65b97aa5b265411651;
four value-1 dumps were internally deterministic at
01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261.
All outputs were finite and top-1 remained token 497. The legs are not
byte-equivalent because value 1 is the already known alternate exact numerical
route; a v28 receipt would still be required if it ever passed performance.

The final no-dump eight-repeat bracket reported control wall/load
7616.321/6107.0 ms and candidate wall/load 10179.578/8580.3 ms. Subtracting the
process-reported load terms gives 1509.321 ms versus 1599.278 ms; value 1 is
about 5.96% slower, not faster. This is a directional same-host bracket, not a
llama comparison, but the sign and margin are sufficient to reject promotion.

**Framework decision.** The micro transaction ranked value 1 ahead of zero while
the real workflow ranked it behind zero. Slot 39 is therefore now
EndToEnd/External, the same authority class already used for CUDA Graph policy.
The micro transaction remains diagnostic and can compare kernels, but cannot
persist a nonzero workflow route. The final non-writing tuner run exits zero and
prints prefill_exact128_sm86_route = 0 with micro tuner preserving incumbent.
This prevents a fast proxy from misconfiguring the engine and preserves
per-host/model/space knob identity for a future whole-engine tuner.

**Verification and retained failures.** The SM86-only release rebuild completed
in 1m50s after the native signature change and subsequent incremental builds
completed in 6.17s and 6.63s, all exit zero and FRESH. The two directly affected
FFN/tuner source-contract tests pass, as does the focused tuner evidence unit
test. A whole 20-test source-contract run passed 17/20: one assertion was
updated for slot value 2, while two unrelated pre-existing Graph/attention
string assertions still disagree with the current implementation and are not
reported as green. Two earlier build invocations failed before compilation
because of Windows command-shell PATH expansion; the delayed-expansion build
script fixed the command. A rejected trace attempt exited one because the tuner
correctly refuses diagnostic CUDA route environments.

**Rollback.** Slot 39 zero is immediate runtime rollback. Removing value 2, the
Token64 counter and J64 launch instance reverts the candidate; reverting space
v28 is valid only together with those removals. The 256-byte synthetic tensor
alignment and EndToEnd admission classification are correctness/authority
repairs and should not be rolled back merely to recover the old micro-tuner
behavior.

### E-20260901-50: exact-128 grouped K/V physical Stream-K

**Question.** Gemma4 already calls the semantic `matmat_pair` hook for K/V. CUDA's
multi-token laboratory implementation shared one activation-to-Q8 conversion but
still launched two independent physical Stream-K MMQs and two fixups. For E4B at
128 tokens, each 2560 -> 512 projection owns only four logical R128 tiles and the
selector expands it to 30 physical workers. This experiment asked whether one
interleaved 60-CTA submission plus one paired fixup could remove short-request
launch cost without changing per-projection ownership.

**Candidate and numerical invariant.** Even/odd CTAs selected K/V while sharing the
same worker number. Each projection retained the established 30-worker
`stream_boundary`, K segment, suffix write and nearest-prefix fixup order. The
SM86 release compiled in 2m16s with exit zero and reported FRESH. Real CUDA
full-logit runs proved the grouped route and the old two-launch Q8-ready route
produced the same 1,048,576-byte output, SHA-256
`63344a4083b475dec8149a6c39ac6c4ac1d6dcf8c41701c6a84f2d8cd85aef55`.
The safe default has a different known numerical route and produced SHA-256
`4bd15469cbde592761d072e4a4fcd3b5c3a5058847b13245c0738231dbdecf7e`;
therefore grouped versus safe-off is not a receipt-equivalent comparison.

**Rejected early signal.** Four separate real-CUDA processes gave post-load
directional means 581.689 ms for safe-off and 571.097 ms for grouped, superficially
about +1.85%. Load completion is asynchronous and one-shot work includes lazy
preparation and output tooling, so this was not admitted. An earlier four-process
screen accidentally omitted `IMPARO_GPU=1`, ran the CPU workflow and was explicitly
discarded; its common hash is not CUDA evidence.

**Warm-state decision.** Six forwards in one GPU process removed model loading and
first-use preparation. Old Q8-ready pair wall times were
`141, 90, 87, 87, 87, 87 ms`; grouped wall times were
`141, 92, 87, 87, 87, 88 ms`. Excluding first-use work, both medians are 87 ms:
no measurable steady-state improvement. The candidate is **No-Go**, was removed
from native source, never entered the versioned tuner/receipt space and never changed
the production default. Structured evidence is in
`docs/evidence/cuda-sm86-e4b-open-source/e4b-sm86-exact128-grouped-kv-no-go.json`.

**Reusable lesson.** A semantic pair hook and fewer CUDA submissions are not enough
when the paired MMQs retain identical arithmetic and physical-worker count. Future K/V
work must remove material data movement or computation (for example a proven
producer/consumer fusion), and must compare warm same-numerical-route legs before
spending receipt effort. A one-shot wall-minus-load delta is not admission evidence.

### E-20260901-51: route-5 split-Graph admission and low-grid schedule screen

**Admission defect and repair.** The Rust split-Prefill policy already admitted the
exact-128 fast transaction (slot 39 values 4/5), but native Graph eligibility, warmup
and capture-ready checks still recognized only values 1/2. A graph=1 timing screen
therefore observed no capture or replay and was discarded. Native eligibility now
recognizes the same fast transaction at all three boundaries. The rebuilt SM86 binary
reported one capture and one post-embedding replay; Graph-off and final replay each
produced 1,048,576 bytes with identical SHA-256
`b92756089812d9d7115bb6c13fb13f1572cba9bfad202bd35e5f7ea67b2c9a9d`.

**Internal timing signal.** A same-build, correctness-gate-only 128-token screen
reported Graph-off median `2148.44 tok/s` and retained split-Graph median
`2242.59 tok/s`, a directional `1.0438x`. This is not a formal llama comparison or
promotion result. The candidate remains unreceipted and therefore fails closed in an
ordinary process.

**Low-grid No-Go screen.** One temporary same-binary mask compared virtual MMQ for
local-Q, K/V, Attention-O and full-Q against the fast transaction's physical schedule.
Local-Q regressed; K/V was noise; full-Q was below one percent. Attention-O showed
about `1.027x` with Graph off but only about `1.005x` when combined with split-Graph,
so the mask and its laboratory selector were removed rather than entering the tuning
space. This rejects scheduler substitution as the next additive breakthrough.

**Candidate reproducibility.** `imparo-tune` now accepts repeated, duplicate-checked
`--external-candidate NAME=VALUE` arguments and emits them as one whole-model tuple.
It can therefore materialize route5+graph1 without hand-editing while retaining the
existing no-receipt/no-authority fallback. Fixed distribution gates and a new receipt
remain required before default or release use.

### E-20260901-52: route-5 Graph additive-candidate timebox

**Question and boundary.** The retained route-5 plus split-Graph screen was already
`1.085x` the same-host llama reference. Reaching `1.10x` required only about another
`1.36%` from Imparo, so this timebox priced existing correctness-qualified components
before authoring another large kernel. All legs used the same SM86 release binary,
E4B, q4_0/q4_0 KV, exact 128-token raw input, start position zero, disabled KV reuse,
and post-warm Graph replay. They were directional process-separated screens, not the
formal ABBA promotion bracket.

**Profile attribution.** A corrected Nsight Systems run (the first invocation omitted
`IMPARO_GPU=1` and captured CPU initialization only) traced five route-5 Graph forwards
with CUDA Graph node tracing. Excluding the one-time 210-weight admission prepack, the
steady GPU body was about `57.6 ms/forward`. The four dominant groups were:

- Q8-ready Gate/Up pairCTA: about `15.97 ms/forward`;
- Stream-K ordered fixup: about `11.77 ms/forward`;
- Q8-ready physical Stream-K main: about `9.17 ms/forward`;
- Q4 Stream-K main: about `7.37 ms/forward`.

The report is `target/route5-graph1-gpu-current.nsys-rep` with its adjacent SQLite
export. Admission prepack must not be divided by the five forwards: its 210 calls are
one model-load event, not request work.

**Additive screens.** The retained control's seven Graph replays had a median
`2212.6 tok/s`. None of the following exceeded it:

| candidate | stable median | candidate/control | decision |
|---|---:|---:|---|
| Dual-RMS -> Q8-ready | `2179.5 tok/s` | `0.9850x` | No-Go; old non-Graph gain does not compose |
| Q8 persistent-L2 window | `2206.3 tok/s` | `0.9972x` | No-Go / noise |
| virtual full-Q | `2175.6 tok/s` | `0.9833x` | No-Go |
| virtual Attention-O | `2160.1 tok/s` | `0.9763x` | No-Go |
| virtual full-Q + Attention-O | `2145.4 tok/s` | `0.9696x` | No-Go; negative interaction |
| Full-K Down R128 | `2154.4 tok/s` | `0.9737x` | No-Go |
| Full-K Down R96 | `2186.7 tok/s` | `0.9883x` | No-Go |
| Full-K Down R96W24 | `2181.3 tok/s` | `0.9859x` | No-Go |
| Full-K Down R80 | `2091.7 tok/s` | `0.9454x` | No-Go |

The first Dual-RMS attempt deliberately failed closed because its laboratory
environment was not named by the route-5 receipt; it therefore did not load the host
config and is excluded. A temporary default-off fast-transaction alias was then used
only to make both isolated legs identical. The alias and the temporary virtual-shape
mask were removed after the screens; production and the tuning space were never
widened.

**Decision.** No additive candidate cleared even the `1.0136x` local threshold needed
for `1.10x llama`. Retain route-5 plus split-Graph and its latest external comparison
of `2246.32 tok/s` versus llama `2069.80 tok/s` (`1.0853x`). Do not promote Dual-RMS,
persistent-L2, virtual-Q/O, or a Full-K Down schedule. The next structural hypothesis,
if reopened, is true heterogeneous Q/K/V projection fusion that shares an activation
stage inside a CTA; grouped submission without reducing staging was already rejected
by E-50. It is intentionally outside this short timebox and requires its own single-op
resource/correctness gate before any workflow or ABI expansion.

### E-20260901-53: route-5 plus split-Graph fixed-suite receipt

**Admitted candidate.** After the additive-candidate timebox found no winner, the
retained exact-128 tuple was regenerated with the final SM86 release rather than
reusing the earlier pre-rebuild file. The versioned v30 candidate is
`target/exact128-v30-route5-graph1-final.txt`, SHA-256
`f05a39385879c2baa7d9fc1b7deef54eb2da7a198058dc8e55e4ac28cc43346b`.
It selects `prefill_exact128_sm86_route=5` and
`prefill_exact128_graph=1`; all other shapes retain their established routes.

**Fixed correctness suite.** `dev_harness/seal_receipt.py` authenticated the pinned
llama upstream-base CUDA/SM86 oracle bundle and ran gate-suite 7. All seven gates
exited zero: 128/449/512 logit agreement, 128/512/2000 eight-step decode agreement,
and the exact-128 full-logit Graph replay byte-equality gate. The schema-v3 receipt
was written atomically at
`target/exact128-v30-route5-graph1-final.txt.receipt.json`, SHA-256
`cc76781af3515ef71488d362d84bf03ef75bae49bf7b85a15c572a84f6921bb7`.
Its producer is `dev_harness/seal_receipt.py` version 7 and every gate records a
passed result. The engine was
`target/release/imparo-forward.exe`, SHA-256
`a8ebd4cf1a7d0bb3016b8366422b6a249e5dc5c990fcf8c9c4e782d96ec09c4e`.

**Production-load proof.** With `IMPARO_CORRECTNESS_GATE` absent, the ordinary
receipt-backed Graph agreement command exited zero, reported activation `config`,
one capture and one replay, and compared 262,144 f32 logits / 1,048,576 bytes
exactly. The replay SHA-256 was
`e50c4aa81239f6bb46f57cd6867ec80b841c020ac1bd2d55ced07315e84454bf`.
This proves the result is not dependent on the laboratory receipt bypass.

**Focused regressions.** The 11 sealer unit tests passed. A first combined unittest
invocation exited one because `test_prefill_graph_agree.py` imports its sibling as
a top-level module and was launched from the repository root; this is an invocation
error, not a green test. Re-running from `dev_harness` passed all nine Graph tests.
The failed command is retained rather than rewritten as success.

The repository-level review initially found three stale string-contract assertions:
the tests still treated the old Route-1/2 predicate as the entire atomic family and
did not include the separate Route-4/5 fast transaction in PLE, while one Attention
assertion predated its explicit F32 laboratory term. They were updated to assert the
current narrower predicates, including `n_tok == 128`, SM86, slot-39 value 4/5 and
sidecar readiness. The FFN/Graph/Metal contract then passed 20/20. Host receipt tests
passed 9/9, including missing/tampered receipts and model/device/driver/ABI/space/KV/
numeric-route/gate mismatch fallback.

The first CUDA-feature correctness filter could not start NVCC because `cl.exe` was
not on PATH; the first VS wrapper then failed to locate Cargo. The corrected absolute-
Cargo VS invocation compiled SM86. Its first valid run exposed another stale test that
treated now-declared slot-39 value 3 as illegal. The strengthened test binds distinct
parameter hashes for every declared value 0--5 and rejects value 6. One intermediate
edit failed Rust ownership checks before `.clone()` was added. The next exact-128 run
passed 5/5, and the complete CUDA library suite exposed and then repaired the final
stale `space=27` assertion to the current v30 contract. The final CUDA library result
was 33/33.

`cargo fmt --all -- --check` was also run and exited one. It reports broad pre-existing
format drift across unrelated CPU, model, server, tokenizer, tuner and CUDA files in
the already-dirty worktree, so this result is not reported as green and no workspace-
wide mechanical rewrite was applied. The scoped `git diff --check` for this admission
slice exits zero.

Because Rust changed the executable SHA even though the repairs were test-only, the
SM86 release was rebuilt (`FRESH`, exit zero), the candidate was regenerated byte-
identically, and all seven fixed gates were rerun against the new binary. The review
copy is under
`docs/evidence/cuda-sm86-e4b-open-source/exact128-route5-graph-v30/`; the runtime
copy is under `artifacts/`. Both pairs are byte-identical. An ordinary loader run
against the final `artifacts/` pair again passed capture/replay byte equality. A
separate temporary candidate without its receipt was rejected as expected
(`capture=0`, `replay=0`) and then removed.

**Performance decision and scope.** The code and runtime route are unchanged from
E-52, whose latest same-host external screen measured `2246.32 tok/s` against llama
`2069.80 tok/s`, or `1.0853x`. The user explicitly accepted this fallback after the
timeboxed search did not reach `1.10x`. This is the retained exact-128 candidate, but
it is not relabelled as a formal `ABBA BAAB` matrix receipt: the five-regime release
performance protocol remains a separate packaging/PR admission step.

**Rollback.** Setting slot 40 (`prefill_exact128_graph`) to zero disables Graph while
retaining route 5; setting slot 39 (`prefill_exact128_sm86_route`) to zero restores
the established exact-128 path. Removing, changing or mismatching the adjacent
receipt causes the ordinary loader to reject the candidate and fall back safely.

### E-20260901-54: fail-closed public-export boundary

**Gap found.** The repository's public mirror uses `git ls-files` and drops private
top-level directories, but it had no cross-platform read-only verifier. The first
manual pending-file inventory also found that `.imparo/` KV/session state and alternate
`target-dynamic/` / `target-gate0-capture/` Cargo trees were not ignored. They were not
currently exported because they were untracked, but 2,472 public-scope untracked paths
could have been accidentally staged and then mirrored. `.gitignore` now excludes
`.imparo/` and every `target-*` tree, and `sync_public.sh` explicitly drops local state,
models, artifacts, evidence, experiments, reference bundles and every `target-*`
top-level directory even if a file is force-tracked.

**Verifier.** `dev_harness/verify_public_export.py` is read-only and uses the same
drop policy as the sync script. By default it audits tracked plus non-ignored untracked
files so its result models the pending PR; `--tracked-only` reproduces the current index.
It rejects unsafe Git paths, symlinks, missing/non-regular files, model/key/state
filenames, private-key or GitHub-token material and Windows/macOS developer paths. Six
focused unit tests bind the shell/Python policy and negative content rules.

**Evidence.** Unit tests passed 6/6. The pending-PR audit passed with 165 files,
4,685,808 bytes and manifest SHA-256
`d454c1e925e73de581268a1efb062ee1028de196c4fa5af31dfe80a68f2aabe0`.
The tracked-only audit passed with 141 files and manifest
`2f79e4beab97e980942c9e22a634025824606e842bb4023f3518b665c685c7dd`.
`git check-ignore -v` proved the new rules own `.imparo/`, `target-dynamic/` and
`target-gate0-capture/`. The structured pending-PR result is tracked privately as
`docs/evidence/cuda-sm86-e4b-open-source/exact128-route5-graph-v30/public-export.json`,
SHA-256 `bbaa703c3e77ae3fb20538f5a41ecc55a00687c1302b86f3da4d7fe50c113ee`.

**Boundary decision.** Default-off laboratory source remains part of the public
engineering basis; it is not treated as secrecy because receipt loading rejects
diagnostic route environments and contributors need the safe experimentation surface.
Raw model/candidate evidence, private thresholds and Handoff analysis remain in dropped
directories. Future commercial kernels or candidate data must be placed outside the
public file set; an environment selector is not a source-confidentiality boundary.
No runtime source, candidate bytes, receipt or engine binary changed in this slice.

### E-20260903-58: LFM2 capture-local Q8 producer reuse

**Hypothesis.** LFM2 Decode repeatedly normalized one activation and then launched a
standalone Q8 quantizer before its projection kernels. The existing CUDA projection RMS
kernel could publish the identical private Q8 representation, but graph capture could
not safely trust ordinary transient-cache ownership across later replays.

**Implementation.** CUDA space v43 adds default-off slot 52 as a bit-affecting,
EndToEnd/External selector. LFM2 routes its operator, FFN and final one-token projection
norms through the backend preparation hook only when the selector is active. The shared
Backend default remains false, preserving Metal and Prefill. Native ownership is bound
to a monotonically increasing Decode capture generation and is invalidated before
capture, after capture, before replay, on scratch replacement and on policy changes.
The existing graph trace now reports total nodes without expanding the public ABI.

**Graph and performance evidence.** The dependency-complete v42 control captured 491
nodes; v43 captured 414 while both retained 40/40 dynamic nodes. The 77-node difference
matches the removed per-token quantizers. Four process-separated A/B/A/B legs at a
128-token prompt, each using 62 post-capture replays, measured control medians of
12178.950 and 12181.050 us and candidate medians of 12132.950 and 12114.950 us. The
paired improvements are approximately 0.38% and 0.54%. This is accepted as a small
whole-Decode gain, not extrapolated into a llama.cpp claim.

**Correctness and failure evidence.** Backend tests passed 22/22, LFM2 tests 16/16 and
the CUDA transaction contracts 28/28. The SM86 product build completed and reported
fresh binaries. The first seal attempt rejected a mutable build directory whose file
inventory did not match the pinned oracle manifest and wrote no receipt. The corrected
immutable oracle bundle passed all six fixed Q8 gates: logits 128/449/512 and recurrent
Decode 128/512/2000. An ordinary, non-gate process then accepted the tracked v43 receipt
and reproduced the retained 414-node graph.

**Decision.** Keep the versioned candidate and receipt. Slot 52 remains compiled off for
all unreceipted identities; value zero restores the v42 graph. The gain validates
capture-local producer/consumer preparation but also shows that launch-node removal is
only a secondary Decode lever behind weight traffic and projection arithmetic.

### E-20260903-59: LFM2 Q8 Prefill high-leverage screens

**Cached F16 Down screen.** A default-off provider dequantized each resident Q8_0_TM
Down matrix once into an F16 sidecar, converted the existing Gate/Up Q8_1Mmq sidecar
directly to F16, and issued FP32-accumulating Tensor Core GEMM. At 512 tokens, the v43
warm control was 116.9/116.4 ms and the candidate was 115.4/114.1 ms (about 1.017x by
warm medians). The candidate also changed logits and required about 1.3 GiB of cached
F16 Down weights. This does not justify its memory or numerical-route cost; the source
experiment was removed.

**Post-embedding Q8 Prefill Graph screen.** A default-off generic body split captured
after token gather, preserving request-specific embeddings. Capture succeeded at 512
tokens and replay was bit-identical, but two replays measured 129.8/129.3 ms versus the
same 116.9/116.4 ms ordinary control. Large full-body Graph replay is therefore a
negative scheduling result on this SM86 workload, not a route to the remaining Prefill
gap. The experimental LFM2 lifecycle and Q8 Graph widening were removed.

**Decision.** Keep neither implementation. The evidence redirects work toward the
ordinary Q8 projection transaction and smaller operator/launch families rather than
F16 weight duplication or a monolithic Prefill graph.

**64-token Down tile screen.** Reusing the established arithmetic with a 64-token
aligned tile reduced accumulator pressure but doubled weight-tile traversal. At 512
tokens the bit-identical candidate measured 116.1/116.0 ms warm versus the adjacent
116.9/116.4 ms control, only about 1.005x by medians. This is below a useful promotion
margin and the laboratory launcher was removed; smaller token tiles should not be
repeated without a design that also achieves two resident CTAs per SM.

**Residual Add -> RMSNorm -> D4 Q8 screen.** A model-agnostic backend transaction
combined the mixer residual update with the private FFN projection producer and retained
the post-add 2048-wide row in shared memory. The bit-identical 512-token candidate had a
111.8 ms warm median versus 112.0 ms for the adjacent RMS-D4-only control (about 1.002x),
which is indistinguishable from run noise and far below the cost of a new ABI operation.
The fused operation was removed; the independently useful RMS-D4 producer remains.

**T64/S4 two-CTA Down screen.** The aligned tile was parameterized to combine a
64-token tile with four K32 blocks per stage, reducing dynamic shared memory to about
38.9 KiB so two CTAs can reside on an SM86 SM. It remained bit-identical, but the
512-token warm median regressed from 111.0 ms to 112.4 ms (about 0.988x). Occupancy did
not repay the doubled token-tile weight traversal and K-stage overhead; the variant and
its template expansion were removed. The next Down experiment must preserve T128 data
reuse rather than shrinking the token tile.

**R64 short-Prefill Down screen.** A 64-row tile doubled the 128-token Down grid
from 16 to 32 CTAs and halved each thread''s accumulator set while preserving T128
weight reuse. The extra activation staging and reduced row reuse dominated: the
bit-identical 128-token warm median regressed from 35.3 ms to 37.4 ms (about 0.944x).
The row-tile generalization was removed; short-Prefill work should not trade away R128
reuse solely to fill the first SM wave.

**Register-direct single-token Down screen.** Mapping each TileMajor 8-row file
unit directly to one warp removed the shared-memory weight round trip and K-stage
barriers, while preserving bit-identical graph Decode results. It nevertheless
regressed the 128-prompt, 96-step graph-Decode median from 12038.8 us/token to
12549.0 us/token (about 0.959x). The implementation was removed. The existing
shared tile provides beneficial weight/activation reuse and scheduling; future
Decode work must not assume that fewer barriers alone outweighs that locality.

**Stage64 single-token Down screen.** The incumbent TileMajor algorithm was
extended from 32 to 64 K32 blocks per shared stage, with each thread consuming
both block cohorts in the original order. The corrected candidate preserved the
complete 96-step Decode trail and final logits, but regressed the median from
12038.8 to 12237.35 us/token (about 0.984x). Doubling shared memory reduced
resident parallelism enough to dominate the saved barriers. The variant was
removed; further shared-stage enlargement should not be repeated on SM86.

**R4 single-token Down screen.** A four-row CTA split each physical eight-row
TileMajor unit into two CTAs, halving per-thread accumulators and doubling the
grid while preserving the incumbent K32 ownership and reduction order. The
SM86-only release build completed successfully. In an adjacent 128-prompt,
96-step Graph Decode A/B, the complete token trail and printed logits were
identical, but the candidate regressed the steady median from 11999.700 to
12672.150 us/token (about 0.947x). The lost row-level weight and activation reuse
dominates any occupancy benefit, so the laboratory kernel and dispatch were
removed. Future exact Decode experiments must preserve the eight-row storage
unit as the minimum reuse granularity.

**R16 single-token Down screen.** One CTA consumed two adjacent physical
eight-row TileMajor units so activation reads and stage barriers were amortized
across sixteen output rows. The complete 96-step trail and printed logits were
identical to the adjacent control, but the steady median regressed from
12152.400 to 12271.050 us/token (about 0.990x). Register pressure and the
non-contiguous two-unit staging outweighed the saved fixed work. The candidate
was removed; changing row granularity in either direction is now closed for
this SM86 kernel unless a future layout changes the underlying storage unit.

**Quartet-cooperative scale screen.** The incumbent R8/Stage32 mapping was kept
intact while each four-lane K32 group loaded and converted one activation scale
and one copy of each row scale, broadcasting the products to its peers. Aligned
shared weights were read as an int2 without changing either DP4A order. The
96-step trail and printed logits remained identical, but the steady median
regressed from 11968.750 to 12354.550 us/token (about 0.969x). Quartet shuffle
dependencies cost more than the eliminated half conversions and multiplies, so
the candidate was removed. Scale instruction count is not the limiting Decode
lever while the kernel remains dominated by weight traffic.

**Canonical LM-head R8 screen.** A very-wide-output single-token path grouped
eight canonical Q8 rows into one CTA, sharing activation reads and reduction
setup without changing any row's K32 order. The 96-step trail and printed
logits were identical, but the steady median moved from 12048.900 to
12086.500 us/token (about 0.997x). Reducing CTA count does not address the
canonical head's weight-traffic limit, so the candidate was removed. Any future
exact head experiment needs a different physical layout or a transaction-level
consumer, not another row-grouping variant.

**Gate/Up W16 Prefill screen.** The paired Q8 TileMajor Gate/Up transaction was
expanded from eight to sixteen output rows per CTA while retaining the same
input-side Q8 producer and sidecar contract. At 512 tokens, warm control and
candidate medians were 143.616 ms and 143.547 ms respectively, a roughly 0.05%
movement inside run noise. The candidate was removed; doubling row ownership
does not provide a useful Prefill lever for this transaction on SM86.

**MSE-refined Q4 admission scale screen.** Two least-squares scale refinement
iterations were applied while constructing the Decode-only Q4 shadow. The fixed
128-token Decode gate kept all top-1 tokens but did not reduce the material
logit errors: maximum deltas remained as high as 0.642. The refinement was
removed; future lower-bit work must improve the representation rather than tune
one scalar against already-rounded values.

### E-20260903-60: LFM2 mixed-precision Decode Down admission candidate

**Search result.** A model-owned per-layer policy keeps exact Q8 for layers
15--24, 26 and 29, uses Q5 for layers 0, 2, 5, 9, 13, 27 and 28, and uses Q4 for
layers 1, 3, 4, 6--8, 10--12, 14 and 25. CUDA receives only explicit requested
formats with resolved tensor offsets and shapes; it does not contain model names
or layer-number policy.

**Fixed correctness evidence.** Against the authenticated 4695f001 Windows SM86
oracle, recurrent Decode passed 8/8 steps at prompt lengths 128, 512 and 2000
with a 0.35 maximum-logit-delta gate. The observed per-suite maxima were 0.28203,
0.23633 and 0.31412 respectively; all top-1 tokens matched.

**Performance evidence.** The first passing laboratory mask improved adjacent
128-prompt Graph Decode from 12067.000 to 11368.650 us/token, approximately
1.061x internally. The final policy additionally restores layer 29 to exact Q8
to pass the 2000-token gate; an adjacent final-mask control/candidate pair
measured 12097.200 and 11504.000 us/token, approximately 1.052x internally.
This evidence is internal A/B only and is not a llama.cpp claim.

**Q6 extension screen.** The twelve layers left exact-Q8 by the final mask were
converted to a 26-byte K32 Q6 shadow and consumed by the same R8/Stage64
TileMajor schedule. The 128-token fixed gate passed 8/8 with a maximum delta of
0.32471, but Graph Decode measured 11500.900 us/token versus 11504.000 for the
Q4/Q5/exact-Q8 policy. Extra high-bit unpacking consumed the saved bandwidth;
the Q6 implementation was removed without running redundant longer gates.

### E-20260903-61: LFM2 final-Prefill tail compaction from v2

**Applicability review.** The latest v2 E4B shared-KV layer skipping is not
applicable because every LFM2 layer owns recurrent or KV state. The reusable
LFM2 optimization is the final-chunk boundary: after the last layer commits its
state, only the aligned tail containing the requested final token must continue
through that layer's remaining Attention/ShortConv projection and FFN work.

**Implementation.** The common backend now exposes non-overlapping float-range
copying with CPU, CUDA and Metal implementations. LFM2 compacts the final 64--79
aligned rows of a Prefill chunk in place after the last state-writing operator.
Non-final chunks retain the earlier state-write cut, and Decode does not enter
the tail path. This composes with the existing CUDA TileMajor, sidecar and
projection-preparation routes instead of replacing them.

**Focused evidence.** The SM86 release build completed successfully. A same-binary
455-token A/B used identical token rows and safe defaults because the historical
v43 receipt correctly rejected the rebuilt binary identity. The tail candidate's
warm median was 127.483 ms; the full-width control median was 134.188 ms, or
approximately 1.053x internally. All four candidate and control runs printed the
same top-10 token ids and values. This is a useful workflow result, not a new
external llama.cpp claim or a replacement for the final receipt-bound gates.

### E-20260903-62: LFM2 Decode Gate/Up Q5 shadow screens

**Fused representation screen.** Decode-only Gate and Up Q8 TileMajor weights
were independently admitted as Q5 shadows, then consumed by one four-warp
transaction that keeps both projections private through the SiLU product.
Compressing all 30 layers established a useful speed ceiling: the 128-prompt
Graph Decode median moved from 11464.200 to 10526.350 us/token, about 1.089x
internally, but recurrent output diverged and the route is not admissible.

**Layer-policy search.** The best 128-token mask, `0x3AAAFFBF`, selected 23 of
30 layers and passed all eight authenticated recurrent steps with maximum
logit delta 0.33789. Its Graph Decode median was 10777.250 us/token, about
1.064x over the adjacent control. The same mask failed closed at 512 tokens
(maximum delta 0.36269) and 2000 tokens (0.42216), despite matching every
top-1 token. It is therefore a short-context tuner candidate only, not a global
default. Replacing three low layers with Q4 caused top-1 or distribution
failures and was rejected.

**Rejected execution variants.** A mixed Q5/exact fused kernel reached internal
ceilings of about 1.063x for Gate-Q5 and 1.050x for Up-Q5, but neither produced
a useful three-context numerical policy; the implementation was removed.
Running Gate and Up concurrently as two four-warp cohorts preserved the
23-layer candidate's exact output but regressed its median from 10777.250 to
11774.950 us/token because the 256-thread, double-stage CTA reduced effective
residency. That implementation was also removed. Future work should keep the
serial four-warp staging schedule and improve representation or transaction
coverage rather than duplicating the shared-memory stage.

### E-20260903-63: exact-Q8 Gate/Up serial-staging screen

**Hypothesis.** The selected exact Q8 Gate/Up/SiLU transaction uses two
independent four-warp cohorts and two shared-memory stages. A laboratory
candidate instead used one four-warp cohort, reused a single stage for Gate and
Up, and retained both results until the same SiLU product. Each projection kept
the established K32 ownership, accumulation order and reduction tree.

**Evidence and decision.** The 128-token recurrent gate passed 8/8 with a
maximum logit delta of 0.12550. A same-binary Graph Decode pair measured
12080.500 us/token for the selected parallel control and 12111.850 us/token for
the serial candidate, approximately 0.997x. Reducing threads and shared memory
does not repay lost projection-level concurrency for exact Q8 on SM86. The
candidate and its environment selector were removed; future exact-Q8 Decode
work should target a larger transaction boundary or measured instruction/data
movement rather than cohort count.

### E-20260903-64: LFM2 SM86 Q8 open-source v48 candidate

**Selected changes.** CUDA search space v48 promotes three shape- and
receipt-bound choices without model-name or exact-request routing:

- prefill_projection_q8_d4=2 folds residual add, RMSNorm, dense output and the
  MMQ D4 activation into one width-2048 producer;
- mmq_q8_tm_async_weight_stage=2 selects the 16-warp schedule for eligible
  contracting Down projections while preserving the value-1 schedule elsewhere;
- prefill_head_post_threads=64 replaces the 256-thread default only for
  multi-token, head-dimension-64 postprocessing. Decode remains on its established
  schedule.

The receipted host configuration also selects batch=1024. The config loader
already owns this per-model/per-device decision and installs it before activation
allocation, so E4B and other models keep their own independently validated batch.

**Rejected screens.** A bit-preserving Q8 row-major-to-TileMajor output-head
shadow was slower and changed the reduction route; it was removed. Eight
single-token MMVQ warp/row schedules were no faster than the incumbent and were
removed. A generic 512-thread add+RMS kernel was slower; only the width-2048
contiguous-float4 specialization survived. These failures argue for keeping
output-head MMVQ unchanged and tuning pre-normalization and chunk transactions
instead.

**Correctness.** The Q8 llama agreement gate passed at 128 tokens (top-1 equal,
9/10 overlap, maximum relative-logit delta 0.10930 at tolerance 0.75) and 5963
tokens (top-1 equal, 10/10 overlap, maximum delta 0.19682). The focused CUDA
registry suite passed 26/26 tests. All measurements below used the final SM86
release binary with no laboratory selector.

**Same-machine performance.** Reference: llama.cpp b10545/a30273376, CUDA,
Flash Attention on, Q8 K/V, batch/ubatch 512. Imparo used the v48 config and its
selected 1024-token Prefill chunk.

| Prompt | Imparo Prefill | llama.cpp Prefill | Ratio |
|---:|---:|---:|---:|
| 128 | 4134.6 tok/s | 3719.1 tok/s | 1.112x |
| 455 | 4513.0 tok/s | 3893.2 tok/s | 1.159x |
| 5963 | 4496.3 tok/s | 4060.3 tok/s | 1.107x |

Steady CUDA-Graph Decode medians were 10.802, 10.834 and 11.536 ms/token
after 128-, 455- and 5963-token prompts. The corresponding llama.cpp Decode
rates, derived from its paired prompt+48-generation timings after subtracting
the matched prompt-only timing, were approximately 81.55, 81.17 and 76.90
tok/s. Imparo therefore reached approximately 1.135x, 1.137x and 1.127x.
Every representative Prefill and Decode leg is above the 1.1x open-source
target; the 5963-token Prefill leg has the narrowest margin and should remain
in the release bracket.

### E-20260904-65: rebased LFM2 SM86 Q8 v49 release candidate

**Numerical repair.** After rebasing onto `origin/v2@efd66df`, the original
three-map Decode tuple failed the pinned Q8 receipt suite at recurrent step 1:
top-1 remained equal, but the maximum relative-logit delta was 0.89544 against
the 0.75 limit. Lowering the limit was rejected. Disabling both Down maps
passed all six gates but reduced representative Graph Decode to about
1.082--1.088x llama.cpp. The final v49 candidate restores layer 1 to exact Q8
by adding `0x02005dd8` as a versioned `decode_down_q4_layer_mask` choice while
retaining the selected Q5 maps. This is a tuner candidate, not a model-name or
literal-token branch.

**Receipt.** `lfm2-sm86-q8-v49-open-source.txt` and its adjacent schema-3
receipt bind CUDA ABI 26, numerical space 49, selector 4, SM86, the exact
device/driver/runtime, model plan, Q8 K/V layout, configuration bytes and
pinned `zeraix/llama-cpp@4695f001`. All six fixed gates passed: logits at
128/449/512 and eight-step Decode at 128/512/2000. The focused CUDA registry
suite passed 26/26, the SM86 release binary built successfully, and rustfmt
plus `git diff --check` passed.

**Post-rebase performance.** Normal loader execution used the sealed v49
configuration. Phase-A1 Prefill medians discard rep 0 and take the median of
five hot repetitions. The immediately adjacent latest-llama pp128 baseline was
3632.54 tok/s; longer reference values use the same b10545/a30273376 binary and
the established fixed options.

| Prompt | Imparo Prefill | llama.cpp Prefill | Ratio |
|---:|---:|---:|---:|
| 128 | 3998.2 tok/s | 3632.5 tok/s | 1.101x |
| 455 | 4542.8 tok/s | 3893.2 tok/s | 1.167x |
| 5963 | 4502.3 tok/s | 4060.3 tok/s | 1.109x |

Steady CUDA-Graph Decode medians after two warmups were 10.859, 10.922 and
11.554 ms/token after 128-, 455- and 5963-token prompts. Against the matched
llama.cpp Decode denominators, these are approximately 1.129x, 1.128x and
1.125x. Every representative Prefill and Decode leg therefore remains at or
above the 1.1x open-source acceptance target after the v2 rebase and receipt
repair.
