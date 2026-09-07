# SM86 CUDA optimization playbook

This playbook turns the SM86 experiment ledger into a repeatable optimization
process. It is a selection and evidence guide, not a list of universally fast
kernel constants. The durable raw results and superseding decisions remain in
`cuda-sm86-experiment-ledger.md`.

## Scope and non-goals

The primary scope is single-request CUDA inference on SM86 with the engine's
existing Q4_0 weights, Q8_1 activations, dense-buffer contracts, and exact
fallback paths. The current evidence is strongest for the observed 449-token
Prefill workflow, its padded 512-token tile, and the canonical gate/up and down
projection shapes.

This document does not:

- claim that an RTX 3060 Laptop result transfers to another GPU or power state;
- turn one measured grid, tile, cache budget, or register count into a default;
- weaken dense-output, cache-epoch, quantization-layout, tail, or seam rules;
- claim decode speed from Prefill measurements;
- treat a logit gate as a substitute for determinism or Compute Sanitizer; or
- publish commercial candidate data as if it were an upstream open-source
  benchmark.

## The optimization loop

### 1. Freeze the control before choosing a kernel

Record the source diff and binary identity, model and quantization, device and
SM count, driver/toolkit, clocks or power state, prompt and token counts, KV and
Flash Attention state, environment variables, warmup, and repeat schedule.

Profile the complete forward pass first. Attribute both absolute time and
whole-run share to the candidate boundary. A kernel with a large local speedup
cannot produce a large end-to-end gain when its boundary is a small fraction of
the run. Conversely, launch and conversion costs may make a multi-operation
boundary more valuable than its largest isolated kernel.

Do not compare numbers from different binaries or sessions as an interleaved
A/B result. For a same-session cross-engine comparison, bracket the candidate
between reference runs, report the conservative ratio against the faster
reference as well as the ratio against the reference median, and retain token
parity, exact command, binary hash, model hash, and raw-sample receipts.

### 2. Write a falsifiable candidate card

Before implementation, state:

- the exact operation or data movement being removed;
- the eligible SM, shapes, quant formats, strides, alignment, and token tails;
- the CTA ownership, seam, K-phase, MMA/FMA, reduction, and epilogue order that
  must remain unchanged;
- the expected resource envelope: threads, warps, registers, shared memory,
  stack/local/spill, and resident CTAs;
- persistent and transient memory cost, including sidecars and workspace;
- the authoritative dense destination and any derived cache epoch/layout;
- the unsupported, OOM, graph, probe, and cache-miss fallback; and
- a numerical, resource, and whole-run threshold that will reject the route.

If the candidate cannot be described with a narrow selector and a conservative
fallback, it is not ready for an implementation experiment.

### 3. Choose by shape, not by model name

Selectors belong to an architecture-and-shape layer. Express them in terms of
SM version, matrix dimensions, token count or padded tile, quant format, memory
layout, alignment, stride, epilogue, and scheduler geometry. Do not key a CUDA
route on a marketing/model name.

Start with the exact measured shape. Expand only after separate correctness and
performance receipts. In particular:

- distinguish a 449-token tail from a full 512-token tile;
- keep 2560 -> 10240 gate/up separate from 10240 -> 2560 down;
- distinguish virtual/no-seam ownership from physical Stream-K fixup;
- require the exact Q4_0 and Q8_1 layout version expected by the kernel; and
- route every mismatch to the established implementation without partial
  output publication.

### 4. Treat occupancy as a resource conjunction

Resident warps alone do not determine latency hiding. Evaluate registers,
dynamic shared memory, CTA width, barrier cohort, independent CTA count, and
the scheduler/fixup work together.

The session's R64xT256 pair-CTA used 128 registers and 65,536 bytes of dynamic
shared memory. Although it reused weights across a wider token tile, it measured
2526.11 tok/s versus 2536.40 tok/s for the R64xT128 control. The reusable lesson
is that a larger CTA can lose independence and enlarge synchronization even
when the resident-warp story initially looks reasonable.

The inverse can fail too. An R32 direct-Q8 producer reduced resources from
124 to 86 registers/thread and was initially believed to change the theoretical
limit from one to two CTAs/SM, yet measured only 0.89304x of R64. The old R64
occupancy premise was later found to use the wrong block width, but the broader
lesson still holds: lower resource counts are not removed work. Extra tiles,
barriers, and shared traffic can dominate. Require measured occupancy and a
mechanism counter beyond the resource calculation.

The current R64/T128 pair-CTA launcher uses 256 threads (`32 x 8`), not 512 as
an older experiment draft stated. The accompanying one-CTA inference is
therefore withdrawn until occupancy is rebound to the exact binary. Never infer
block width from a candidate name or a neighboring T256 screen.

For every compiled candidate:

1. inspect compiler resource output;
2. reject stack, local-memory, or spill regressions unless measured evidence
   decisively compensates for them;
3. calculate actual resident CTA and warp limits from all active resources;
4. inspect barriers and fixup work, not just the main MMA loop; and
5. bind the resource receipt to the exact binary used for A/B.

### 5. Admit sidecars by fit and benefit

A packed sidecar is useful only when the consumer selector can hit it and the
process retains enough headroom for runtime work. Record:

- configured budget;
- actual allocated/used bytes;
- admitted and rejected tensors;
- attempts, hits, misses, and fallback reasons;
- allocation failures and free-memory headroom; and
- cold packing cost separately from warm execution.

In the current session, a configured 1800-MiB budget covered the complete
admitted packed-Q4 set and used 1786.64 MiB. This is a model- and admission-set
observation, not a portable constant. Once the useful set is resident, more
budget cannot improve coverage without changing admission. Preserve the
runtime-derived headroom guard and allocation-failure fallback.

Prefer benefit-per-byte admission for hot tensors. Paired consumers should be
admitted atomically when one half without the other cannot take the fast path.
Do not convert a transient OOM or budget miss into a permanent semantic
rejection.

## Numerical-change classes

Classify the candidate before choosing its tests.

### Class E: representation-exact

Examples include aligned vector loads, byte-neutral Q4 repacking, or combining
two already-computed Q8 bytes into one wider store while retaining each token's
scale, rounding, saturation, and byte position.

Required evidence:

- byte-for-byte derived layout and scale comparison;
- exact dense output where the dense buffer is authoritative;
- boundary-row, qblock, alignment, tail, and seam tests;
- forced miss and unsupported-shape fallback; and
- confirmation that cache publication follows the authoritative buffer epoch.

Do not label a route Class E merely because its mathematical formula is the
same. Changed reduction association, rounding points, or producer values make
it bit-affecting.

### Class O: order-preserving arithmetic

These candidates retain CTA ownership, K-phase order, MMA/FMA sequence,
reduction tree, and epilogue but alter staging, scheduling, or launch
boundaries. Test exact intermediate outputs where available, then fixed logits,
tails, full tiles, longer shapes, fallback paths, and sanitizers.

### Class B: bit-affecting arithmetic

Any changed reduction tree, accumulation precision, activation approximation,
quantization scale, rounding point, or atomic/fixup order is Class B. Define
the tolerance before running performance tests. Require distribution/top-k
evidence, recurrent decode, determinism, KV resume/replay where relevant, and
explicit approval of the numerical policy. Do not silently promote Class B
under a selector originally described as exact.

## Default-off and rollback rules

Every experimental path starts default-off behind one route-specific opt-in.
Its selector must check every eligibility condition before launching. An
unsupported or failed candidate must return control without publishing output,
epochs, or cache ownership, and the caller must execute the established path.

Rollback must be one of:

- unset the laboratory environment variable;
- force the shape/resource predicate false; or
- remove the selector branch while leaving the established kernel unchanged.

Metal, CPU, other SM versions, public stable ABI versions, and default workflow
behavior remain unchanged unless their own evidence explicitly authorizes a
change. Record the exact switch spelling in the experiment receipt; do not
reconstruct it from memory later.

## Multi-kernel fusion is a transaction

A backend hook that replaces several workflow operations has a stronger
contract than an ordinary kernel selector. It must define:

- every public input, authoritative output, and private scratch/sidecar;
- whether intermediate workflow buffers are valid, undefined, or private after
  success;
- the last point at which returning false is guaranteed to leave every public
  buffer untouched;
- the first enqueued operation that commits public output;
- how a launch or runtime error is surfaced before and after that point; and
- the exact fallback that runs only for a pre-commit refusal.

The sidecar-only FFN experiment is the concrete example. Its gate/up stage uses
`WriteDense=false`, so the gated intermediate is not a public result. It
publishes a private MMA-ready Q8 sidecar, and the first down kernel that can
write the final destination is the transaction commit point.

An early implementation allowed the down main kernel to enqueue, then returned
false when the subsequent fixup launch failed. That could run the established
fallback after a partial public write. The repaired launcher reports a
`public_output_committed` bit: preflight and a failed main launch remain safe
to reject; failure after the main output launch marks the destination epoch,
sets a forward-fatal pending error, and returns success solely to prevent
fallback execution.

Use this failure matrix for every fused transaction:

| State | Public output | Hook result | Required action |
|---|---|---|---|
| Unsupported or preflight failure | untouched | false | caller executes established fallback |
| First public launch rejected | untouched | false | clear recoverable launch state and fall back |
| Public launch accepted, later launch fails | possibly partial | true with pending fatal error | never enqueue fallback; report failure at stream/end boundary |
| All launches accepted | complete after stream execution | true | publish output epoch/cache ownership exactly once |

Preflight before commit must include weight slices, integer-overflow checks,
buffer byte bounds, sidecar fit, scratch/workspace allocation, pointer
alignment, launch attributes, grid bounds, graph/probe eligibility, and every
condition that can fail recoverably.

### Transaction route trace

Record three counters:

- `attempts`: the workflow requested the route;
- `admitted`: all shape/ABI/preflight eligibility before producer launches
  passed; and
- `committed`: the first public output launch crossed the commit point.

Require `attempts >= admitted >= committed` and explain every gap. A zero-hit
or zero-commit performance run does not test the intended optimization.
Diagnostic traces must be collected separately from timed legs unless the same
instrumentation is deliberately enabled in both control and candidate.

### Static versus dynamic ABI

Laboratory hooks should prefer a default-false trait method and a static-build
entry point when the contract is still changing. Dynamic plugins should use a
local false-returning stub until a deliberate ABI-version decision promotes
the symbol. Document both surfaces: a working static experiment is not evidence
that the release plugin ABI changed, and a dynamic fallback is not a failed
static route.

## Static and performance gates

### Generated code and resources

Source-level simplification is not evidence. Inspect generated SASS/PTX and
compiler resource reports from the measured binary.

The session's `vsub` probe measured 2667.81 tok/s and produced worse SASS, so
it was rejected. The Q8+prefetch probe measured 2687.88 tok/s and was also
rejected: the tested latency-hiding schedule did not justify its cost. These
results reject the exact rewrites, not every possible instruction substitution
or asynchronous pipeline.

Static review should account for:

- global transaction width and alignment;
- address-generation and unpack instructions;
- barrier count and scope;
- register reuse and live ranges;
- stack, local memory, and spills;
- shared-memory bank/layout risks;
- branch and predicate structure;
- MMA issue pattern and dependent instructions; and
- scheduler/fixup work outside the inner loop.

### Interleaved A/B

After correctness and static resource gates:

1. use the same binary with only the route switch changed;
2. warm both paths;
3. run A/B/B/A or a longer interleaved schedule;
4. retain every raw leg, median, variance, and event timing;
5. separate cold sidecar creation from warm steady state;
6. test the full workflow, not only the isolated kernel; and
7. reject results smaller than the predeclared noise and whole-run gates.

For cross-engine screens, run reference/candidate/reference under matched
model, prompt, output-count, KV, and attention controls. The faster reference
is the conservative denominator; the two-sided reference median describes the
center of the bracket. Neither ratio is final until both engine binaries and
the model are hash-bound to the raw receipt.

Do not choose the best member of a losing sweep. The R64 down forced-grid sweep
measured 2461.79, 2458.51, 2482.61, 2343.09, 2375.92, and 2505.50 tok/s for
grids 60, 80, 90, 100, 120, and 160. Grid 160 was best within that sweep, but
every tested forced grid remained No-Go.

## Proven success and failure patterns

### Patterns with positive local evidence

- Producer-consumer handoff can matter. The initial direct-register Q8 route
  measured 2686.96 tok/s; packing a token pair into a `uint32` transaction
  measured 2721.82 tok/s.
- That direct-Q8 route passed q4_0 `logit_agree` with tolerance 0.75 at
  128/449/512/2000 tokens. Top-1 was identical; overlaps were
  9/10, 10/10, 8/10, and 9/10; maximum deltas were 0.31426, 0.12315, 0.41576,
  and 0.41741; every run exited 0.
- The logit gate is not full admission. Decode, determinism, and memcheck,
  initcheck, racecheck, and synccheck remain pending.
- A dual RMS + MMA-ready Q8 producer previously moved one measured base from
  2601.16 to 2649.24 tok/s and passed its logit gate. The lesson is to measure
  the complete producer-consumer boundary; later experiments and superseding
  routes still require their own receipts.
- Packed-Q4 sidecars can help only when the admitted hot set fits and consumers
  actually hit the expected layout.
- The complete sidecar-only FFN transaction produced four interleaved
  449-token cross-leg medians of 2698.505 tok/s for the baseline and
  2725.800 tok/s for the candidate, approximately +1.011%. This is
  **promising, non-final** because it remains within the local noise band.
- Compiling the sidecar-only gate/up producer with `WriteDense=false` changed
  dense-to-sidecar static counts from 4096 to 4072 instructions, 112 to 48 STG,
  236 to 109 IADD3, 427 to 428 LOP3, and 126 to 124 registers. SHFL and FMNMX
  remained 112 each; stack and local memory remained zero. The useful result is
  the removed dense intermediate contract, not the two-register difference by
  itself.
- The transaction passed q4_0 `logit_agree` at 128/449/512/2000 tokens with
  identical top-1, overlap 9/10, 10/10, 8/10, and 9/10, maximum deltas
  0.31426, 0.12315, 0.41576, and 0.41741, and exit 0 for every run.
- Its 449-token `decode_agree` passed eight recurrent steps with identical
  top-1, maximum observed delta 0.64433, and exit 0. Its q4 `det_gate` reported
  `1/4 distinct` and `ALL PASS` at 128/449/512/2000, with exit 0.
- Numerical model gates do not make the transaction final. Failure injection
  on both sides of commit, binary/source/backend hashes, and memcheck,
  initcheck, racecheck, and synccheck remain pending.
- A strict current-machine cross-engine bracket measured llama.cpp 2253.53,
  Imparo sidecar 2717.32, then llama.cpp 2231.89 tok/s as repeat-12 medians.
  This is 1.20580x against the faster reference and 1.21162x against the
  two-sided llama.cpp median of 2242.71. The controls were the same model,
  raw 449-token input, `max1`, Q4 KV, and llama.cpp `-fa on -ctxcp 0`.
  Treat this as provisional: executable and model hashes, full command/raw
  receipts, and token-parity evidence remain pending.

### External mature-implementation migration rules

- Triton's [persistent matmul tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py)
  bounds persistent CTA count by `min(SM count, tile count)`. Reuse that
  scheduling structure only after preserving Imparo's existing tile/seam
  ownership. Its TMA and warp-specialization paths primarily target SM90+;
  their shared-memory/resource assumptions are not a direct SM86 recipe.
- vLLM's [modular MoE kernel feature design](https://github.com/vllm-project/vllm/blob/main/docs/design/moe_kernel_features.md)
  separates kernels that quantize activations internally from kernels that
  consume a pre-quantized activation format, while treating GELU/SILU and
  interface compatibility as explicit features. For Imparo, define the Q8_1
  producer/consumer contract and fallback-compatible interface before swapping
  a fused implementation.
- SGLang's [Triton fused-MoE kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py)
  make shape, configuration, and quant-format dispatch explicit and cache the
  matching compiled choice. Borrow that keyed-dispatch discipline for narrow
  SM86 experiments; do not copy a multi-expert schedule into the single-expert
  E4B path without a separate occupancy and whole-run case.
- The public [SGLang configuration-key mismatch issue](https://github.com/sgl-project/sglang/issues/35252)
  is the failure pattern: the runtime lookup key and tuner-emitted key must be
  structurally identical. Every dimension, dtype/quant format, layout, group
  count, architecture, and epilogue field used to select a kernel must be
  present and normalized the same way on both sides, with a tested cache-miss
  fallback.
- NVIDIA CUTLASS's [SM80 MMA definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp)
  are the architecture authority for the W4A8 digit-slice laboratory path:
  Ampere exposes both `m16n8k32 S32=S4*U4+S32` and `S4*S4`. This supports an
  exact `q8=low_u4+16*high_s4` decomposition, but it does not predict a 2x
  kernel win: two INT4 MMA instructions replace one INT8 MMA. The measurable
  opportunity is removed Q4-to-s8 expansion, lower shared-weight traffic, and
  smaller operand fragments.
- vLLM's [W4A8 scheme](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8.py)
  validates the complete quantization/shape contract before asking a backend
  registry to choose an implementation; its current public path is restricted
  to group size 128 and Hopper-class FP8 activation. Borrow the explicit
  capability/configuration admission pattern, not those different numerical or
  architecture assumptions.
- Unsloth's [FP8 backend probing and fallback](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fp8.py)
  records that a portable Triton fallback can be materially slower than a
  working specialized backend and executes a numerical probe before admitting
  FBGEMM. The reusable rule is probe, bind the chosen backend to device/shape,
  and fall back conservatively; neither Triton nor native CUDA is globally the
  winner by name alone.

These sources provide design structure and falsifiable dispatch rules. They do
not justify directly copying SM90/TMA code or publishing their service/MoE
performance as evidence for Imparo's single-request SM86 workflow.

### Packed-shared S8 Gate-A lesson

Experiment `E-20260831-21` tested the narrow successor to failed W4A8 digit
slicing without changing the numerical route. The control expanded packed Q4
into shared s8; the candidate retained packed qblock-major weights in shared,
expanded an A fragment once in registers, and reused it across eight token
fragments. Both routes retained the same eight `m16n8k32 S8xS8` MMA
instructions, K32 scale/FMA order, tail behavior, and final write.

This is a useful negative design result:

- The candidate reduced static shared allocation from 2880 to 2624 bytes, but
  registers rose from 107 to 115. Both routes remained stack/local/spill-free
  and the occupancy API reported 16 CTAs/SM.
- CPU layout testing, a 4160-output GPU oracle, and the preliminary four-class
  Compute Sanitizer sweep were clean. The final host-only four-copy harness
  rebuild has a different executable hash, so the earlier sanitizer sweep is
  kernel evidence rather than a final-binary receipt.
- Nsight Compute bank metrics are unknown, not zero. The one permitted capture
  failed with `ERR_NVGPUCTRPERM`; preflight counter permission before spending a
  future idle bracket on profiler-dependent hypotheses.
- A four-copy, address-independent input pool and paired `AB,BA,BA,AB` schedule
  revealed a large first-versus-second access-position effect. Rotate legal
  cached inputs, give each route equal cold/hot positions, retain every raw
  leg, and analyze paired cycle ratios. A single median of a 50/50 bimodal
  distribution can land between modes and should not stand alone.
- The candidate produced only `0.991384x` control/candidate at N=449 and
  `0.971645x` at N=512. Paired cycle-log 95% intervals were
  `[0.980784,0.999103]` and `[0.974218,1.000314]`. This is a standalone No-Go,
  not permission to try more shared layouts in production.

An independent read-only recomputation found no route or pool-pairing error.
The cycle-sum log estimator reproduces `0.989901120x
[0.980783895,0.999103098]` at N=449 and `0.987179715x
[0.974217993,1.000313889]` at N=512; the first eight cycles, which remove the
9-cycle pool-by-order imbalance, remain `0.991584154x` and `0.986613370x`.
This confirms the No-Go without changing or rerunning the hashed lab artifact.

For future rotating-pool standalone brackets, use 12 cycles so the four pool
offsets and both route orders close exactly. Emit per-cycle control/candidate
sums, log ratios, and their confidence interval rather than relying on the
median/MAD of a 50/50 cold/hot mixture. Report cold and hot strata separately,
and define front/back only over complete balanced cycles (with any center cycle
excluded), so cache phase cannot masquerade as algorithm gain or drift.

The reusable stop rule is causal: if a candidate preserves the number of MMA,
K32 scale/FMA operations, and producer-consumer transactions, a smaller shared
footprint alone is not a sufficient optimization hypothesis. The next Gate-A
candidate should remove core arithmetic or an entire conversion/publication
boundary and declare the expected Amdahl upper bound first. Do not integrate,
specialize across SMs, or enumerate neighboring layouts until that isolated
mechanism clears its threshold.

### W4A4 CPU Gate0: separate invariants, screens, and real evidence

Experiment `E-20260831-22` establishes a reusable CPU-only screening boundary,
not a W4A4 kernel result. Keep three outcomes mechanically distinct:

1. A self-test proves the Q4_0 nibble/K32 authority, the analytic pooled-scale
   formula, affine-A4 zero-point correction, ladder ordering, exact fallback,
   SHA-256, and fail-closed capture parsing. Its exit 0 says only that those
   invariants hold.
2. A fixed synthetic screen measures coverage and numerical error. It may fail
   while every invariant passes and must never be renamed a successful Gate0.
3. Real Gate0 requires an audited capture schema, an externally bound canonical
   envelope identity, and capture-derived statistics. The identity must hash all
   interpreted header metadata with its digest field zeroed plus the payload;
   payload-only hashes permit semantic header replay. Synthetic distributions
   cannot stand in for model activations or weight-scale correlations.

The CUDA workflow producer is a **provisional, laboratory-only bridge** behind
the non-default `cuda-gate0-capture` feature. It records complete f32
`ffn_norm_input` and materialized-fallback `ffn_down_input` tensors only when an
explicit directory and layer/op allowlist are present. Its tensor format is not
the opaque W4A4 capture envelope and must not be fed to that parser without an
audited converter that creates a new canonical identity. The current infallible
backend read surface is guarded by two distinct NaN-sentinel reads and exact bit
agreement, but this does not replace a production fallible read API.

A run is consumable only after the final device completion and logits read have
atomically published `run.complete.json`. The seal binds the executable hash and
size, run-manifest hash, ordered record manifests/tensor hashes, and the exact
expected/actual set. Missing, incomplete, mismatched, or non-finite runs fail
closed. Even a valid seal remains timing- and promotion-inadmissible: the first
real capture must be repeated capture-off with identical request/route identity
and pass final-logits equivalence before its data can support a Gate0 decision.

The pooled weight scale has no tuner, seed, activation input, or iterative
re-quantization. Preserve Q4 signed codes `r_w=q` and compute exactly
`dG=sum(d_j*S2_j)/sum(S2_j)`, with `S2_j=sum(q_ji^2)`. Generate affine A4
`d_a/z/r_a` separately from each group's Q8-authority real values with the
deployable fixed rule `d_a=(max-min)/15`,
`z=clamp(round(-min/d_a),0,15)`, and
`r_a=clamp(round(x/d_a)+z,0,15)`. Round nearest with ties away from zero
and saturate in double before converting to `int`; the all-zero group is the
only zero-scale special case. This avoids training a representation to the same
synthetic data used to judge it and avoids an undeployable runtime search.
The rule is not claimed globally optimal: one-sided groups may clip after
zero-point clamping and constant nonzero groups conservatively fall back. Treat
that as a possible false No-Go/coverage loss, never as evidence that W4A4 itself
is impossible.

The first fixed-seed screen is deliberately retained as No-Go, but it used the
now-removed 16-zero-point, multi-seed iterative activation fitter. Treat it only
as an undeployable fitting upper bound, not as evidence for the current fixed
min/max estimator. Even that upper bound only reached cosine
`0.999278740` (above the future 0.999 floor), while relative L2 was `3.80161%`
against a 3% ceiling and approximate coverage was `66.6667%` against a 70%
floor. Its original normalized-p99 check also failed (`2.083511`, with G64 at
`2.762441`). Because near-zero references dominate that ratio, the statistic is
now versioned as descriptive `p99_floor_relative_v2` and prints its denominator
rule. Reclassifying the statistic does not reclassify the original run.

A synthetic command is never a Gate command: it must print
`decision=continue_to_real_capture`, mark `is_gate=false`, and return
nonzero even if reference thresholds happen to be met. Do not rerun a changed
estimator and compare it with the archived fitter numbers as though only one
variable changed.

Fail closed at every numerical boundary. Check fixed-scale conversion, pooled
energy/error, `dG*dA`, exact and approximate FMA outputs, selected outputs,
and all statistics for finite/range validity. Statistics check vector lengths
and reject NaN/Inf before indexing or sorting. Parser identity tests must include
semantically legal metadata mutations, not only malformed headers.

Until a streaming record reader exists, keep the opaque buffered reader capped
at 64 MiB and catch allocation failure. Before enabling real analysis, define a
schema stratified by layer, shape, and ladder level, aggregate by actual work,
and enforce a worst-layer bound in addition to overall cosine/L2/coverage.

Do not implement W4A4 MMA, packing, selectors, or multi-SM variants from this
synthetic skeleton. First audit a real capture producer and record format; then
require cosine at least 0.999, relative L2 at most 3%, and G128-plus-G64 coverage
at least 70%, together with the registered per-stratum bound. A miss at Gate0
stops GPU work before kernel optimization begins.

The first real 449-token E4B activation pre-screen (`E-20260831-23`) makes that
stop rule concrete. Capture-on/off full logits were byte-identical, but the
fixed activation ladder covered only `18.4202%` overall; the three Down-input
strata covered `12.1158%`, `12.0657%`, and `7.6253%`. Worst-stratum relative L2
was `8.9171%` and cosine `0.996048959`. Weight pooling can only reduce eligible
coverage, so this fixed G128/G64 representation is No-Go without a GPU kernel.
Do not retry it by loosening the numerical thresholds. A future INT4 hypothesis
must first demonstrate a bounded mixed-precision/outlier correction on the same
real tensors and price that correction against the one-K64-MMA arithmetic gain.

The current-v2 sidecar recheck (`E-20260831-26`) demonstrates the higher-value
alternative: remove a complete producer/consumer transaction rather than only
changing MMA granularity. Owning Gate/Up, the private Q8 handoff, and Down moved
the repaired same-binary short Prefill cross-leg median by `1.215845x`, and
measured `1.095979x` against the faster surrounding llama.cpp leg
(`1.101428x` against the two-leg llama median). Component-filtered racecheck
reported zero sidecar hazards. Full-model racecheck initially exposed 87 real
head-norm reduction hazards; adding the missing post-reduction block barrier
then made the complete 128- and 449-token unfiltered racechecks exit zero with
no hazards. Preserve both the failure and the repair. This is a validated
performance mechanism with the global sanitizer blocker closed, but not a
release default until the transactional failure matrix, buffer ownership,
rollback, shape-regression, and production-identity receipts close.

The exact-128 closure (`E-20260831-28`) adds two reusable limits. First, a
complete-K mapping can win at short Prefill when the grid still represents the
entire output and removes a real Stream-K seam; the same idea lost at 449 because
it surrendered too much K parallelism. Specialize ownership from measured shape
arithmetic, never by merely shrinking a long-shape grid. Second, equal-size packed
weights are not free just because their kernel is faster. On the 6-GiB evidence
device the full hot set missed the conservative default budget by about 103.9 MiB
and only the experimental reserve override admitted all 42 layers. A production
candidate must participate in global model/KV/activation fit and fall back as one
complete transaction; do not encode a laptop-specific reserve or accept partial
layer coverage as the same numerical route.

The final exact-128 admission (`E-20260831-32`) adds a numerical-route lesson:
a locally plausible Stream-K seam can be less stable than an established complete
accumulation order, and attention arithmetic can decide recurrent agreement even
when all first-step top-1 values match. Factor a failing fixed gate by route family,
then bind the winning attention, PLE and FFN choices as one versioned selector.
Do not let ambient laboratory flags modify a receipted route. A batch32 attention
fallback that passes the complete recurrent distribution is a valid correctness
repair for one exact shape; it is not a performance claim until the final binary is
remeasured with interleaved whole-engine legs.

CUDA Graph remains an additive scheduling tool, not a substitute for removed
operator work. The exact-key full-logits experiment (`E-20260831-27`) proved
capture/replay and fail-closed identity but moved whole Prefill only `+0.52%`,
below the declared `3%` continuation floor. Retain the evidence and reopen after
the operator route is stable if a wider capture can remove measured synchronization
or parameter traffic. Because fusion and Graph both eliminate launch work, always
remeasure the combination; never sum their isolated percentages.

### Rejected patterns in the current boundary

- Larger token ownership without a demonstrated overlap pipeline:
  R64xT256 was slower than R64xT128.
- Fusion justified only by fewer launches: PLE fused measured 2634.37 tok/s and
  pair projection measured 2657.61 tok/s; both were No-Go.
- Cache reuse without observed hits: RMS cache reuse recorded 0 hits and
  2601.38 tok/s. A zero-hit run tests machinery, not the hypothetical hit path.
- Removing the Q8-ready handoff: RMS float-only measured 2658.14 tok/s and was
  No-Go.
- Forced grid tuning without a winning whole-run member: all tested R64 down
  grids were No-Go.
- Replacing physical Stream-K with complete-K logical-tile ownership merely to
  remove fixup: exact-449 R128 and R64 full-K routes were 2.03% and 6.24%
  slower end to end. CUDA events showed the down itself became 14.0% and 33.5%
  slower, so the lost K parallelism—not an unrelated boundary—falsified those
  exact schedules.
- AOT-specializing the winning physical ownership without changing its data
  path: the exact-449 cycle plan removed runtime boundary division, reduced
  fixup from 240 to 160 CTAs, and remained REG128/STACK0/LOCAL0, but improved
  the same-binary cross-leg whole result only 0.33%. Keep compile-time schedules
  as a tool, not a performance claim; require a larger removed-work mechanism.
- Increasing the Gate/Up software-pipeline K span without removing arithmetic
  or a producer/consumer boundary: the exact-Q4-by-Q8 K128 candidate passed the
  logit gate and hit 34/42 layers, but its mixed-route medians were only
  `0.411935x` (Gate) and `0.424903x` (Up+GELU) of the established route. Packed
  nibble expansion plus a 27,648-byte shared working set overwhelmed reduced
  stage/control work. Do not retry larger K grouping alone; first identify
  material work or traffic that the candidate actually eliminates.
- Template-shaped compile-time plans can hide per-thread stack. The first
  exact-449 helper versions used 128 bytes/thread even though the algorithm had
  no intended local array. Flattening the fixed two-pass cycle removed it.
  Always bind stack/LDL/STL inspection to the exact measured instantiation.
- Replacing one S8xS8 `m16n8k32` with exact W4A8 digit slicing: the candidate
  was bit-exact and resource-clean, but two INT4 MMA instructions plus combine
  measured only 0.75075x of the prepared-fragment control and 0.93767x from the
  same cached packed inputs. Do not infer speed from sub-byte instruction
  support; count the full instruction sequence and data transformation.
- Using volatile global bytes to prevent benchmark hoisting: this produced 24
  strong-system byte loads per iteration and hid the arithmetic by roughly
  500x. Rotate through legal cached input tiles instead, then confirm both the
  loads and MMA remain inside the SASS loop.
- Moving packed-Q4 expansion from shared bytes into a reusable register A
  fragment while retaining the same S8 MMA count: the candidate was bit-exact,
  spill-free, and smaller in shared memory, but measured `0.991384x` of the
  expanded-shared control at N=449 and `0.971645x` at N=512. Its balanced
  four-copy harness also exposed strong first/second cache-position modes.
  Reduce core arithmetic or a complete transaction next; do not keep tuning
  the shared layout without bank-counter evidence.
- Keeping Q4 packed but replacing the S8 tensor-core path with scalar DP4A for
  the 449-token Gate/Up pair: correctness passed and the selector recorded 84
  real hits, but Gate measured `0.418836x` and Up+GELU `0.432221x` of the
  established route. Packed storage is not a performance result; on SM86 the
  lost tensor-core throughput dominates the avoided shared-memory expansion.
  Do not retune this DP4A geometry unchanged.
- Source substitutions with worse lowering, and prefetch without sufficient
  end-to-end gain: `vsub` and Q8+prefetch were No-Go.
- A fused transaction may be numerically correct in its private producer and
  still be contract-invalid if a post-commit failure returns false. Commit-point
  failure injection belongs in the correctness gate, not only code review.

These are selector- and schedule-specific results. Keep them as negative design
evidence, but do not claim they disprove every future implementation of the
same broad idea.

## Triton or hand-written CUDA

Use Triton for a rapid hypothesis test when:

- the operation can be expressed with supported dense/strided block tensors;
- the quant format can be loaded and unpacked without relying on an engine-only
  packed ABI;
- the purpose is to estimate whether fusion, tile shape, or launch removal has
  enough upper bound to justify native work;
- autotuning can be bounded to explicit shapes and cached outside timed runs;
- exact CTA ownership and Stream-K seam behavior are not part of the public
  numerical contract; and
- generated code and resource reports can still be inspected.

Use hand-written CUDA when:

- Q4_0/Q8_1 packed layout and byte identity are part of the ABI;
- inline MMA mapping, lane ownership, fixed K-phase order, or exact FMA/fixup
  order must match the established kernel;
- SM86-specific shared-memory, vector-load, or occupancy policy is the point of
  the experiment;
- virtual/direct seam ownership or workspace publication must remain exact;
- a producer must publish an engine cache epoch/layout atomically with dense
  output; or
- Triton lowering cannot express or verify the needed transaction and SASS
  geometry.

Triton results are feasibility evidence, not automatic permission to replace a
native route. A native implementation must repeat correctness and whole-run
measurement in its actual ABI.

## Open-source basis and candidate-data isolation

Keep three layers distinct:

1. **Open-source basis.** Record the upstream repository, revision, file/symbol,
   license, and the general technique being borrowed, such as consumer-native
   weight packing, fused epilogues, persistent scheduling, or shape
   specialization.
2. **Engine adaptation.** Record the Imparo ABI, selector, invariants, resource
   budget, fallback, and source diff. This is an implementation claim, not an
   upstream performance claim.
3. **Candidate evidence.** Store model identity, unpublished shapes, raw
   throughput, logits, commercial acceptance thresholds, and binary receipts in
   the access-controlled experiment record appropriate to that data.

Do not put private weights, prompts, model identifiers, raw commercial traces,
or customer thresholds into an upstream citation or public benchmark. Do not
attribute an Imparo speedup to vLLM, SGLang, Unsloth, Triton, CUTLASS, or Marlin
unless the measured implementation actually uses the cited code under its
license. It is valid to say a public pattern motivated a candidate; it is not
valid to transfer upstream benchmark numbers to this engine.

Public documentation may contain sanitized architecture, shape class, selector
logic, and reproducible open-model evidence. Commercial candidate data should
use opaque receipt identifiers when a public decision record needs to point to
private evidence.

## Coupled-route tuner evidence

A multi-op selector must be measured as one transaction and prove every
component reached the intended implementation. Do not infer reachability from
equal timings, one successful half, an enabled knob, or an environment flag.
Reset a backend-owned evidence mask per submission, set bits only after real
launch/commit points, and require the complete mask for warmup and every timed
sample. Pack one repetition per checked submission; otherwise OR-ing several
repetitions can let one success hide a later fallback. Control must prove a zero
mask. Dynamic/unsupported backends must also return zero.

Synthetic correctness data must exercise the numerical route. Before treating
cosine, relative-L2 or equality as evidence, require finite output and nonzero
norm from both authority and candidate. Quantized paths can turn an apparently
reasonable low-amplitude periodic stimulus into an all-zero tensor; that is a
harness failure, not perfect agreement. Preserve the failed stimulus and its
replacement in the experiment ledger so later model/SM ports do not repeat it.

## Promotion checklist

A route may move from default-off laboratory status only when all applicable
items are recorded:

- [ ] Exact source diff/commit and measured binary hash
- [ ] Control/candidate executable and backend artifact SHA-256 values
- [ ] Base experiment ID, enabled-route dependency stack, and supersession links
- [ ] Device, SM count, driver/toolkit, clocks/power state
- [ ] Model/quantization identity and exact shape/layout selector
- [ ] Complete environment matrix, including variables explicitly unset
- [ ] Threads, registers, shared memory, local/stack/spill, resident CTA/warps
- [ ] Persistent/transient memory, sidecar used bytes, attempts/hits/misses
- [ ] Dense authority, cache epoch, tail, seam, OOM, graph, and probe fallback
- [ ] Public/private buffer table and pre-commit/post-commit failure matrix
- [ ] Route attempts/admitted/committed trace with every gap explained
- [ ] Buffer byte bounds and `uint64_t` overflow tests at selector boundaries
- [ ] Numerical class declared before performance interpretation
- [ ] Per-op or byte-level evidence appropriate to that class
- [ ] Fixed logits across tail, full tile, smaller, and longer cases
- [ ] Decode, determinism, KV/replay tests where the route can affect them
- [ ] memcheck, initcheck, racecheck, and synccheck
- [ ] SASS and resource inspection from the measured binary
- [ ] Interleaved single-op and whole-run A/B with raw legs, dispersion, and a
      predeclared local noise threshold
- [ ] No material regression on other admitted Prefill/Decode shapes
- [ ] Exact opt-in and rollback switch documented
- [ ] Rollback and injected pre/post-commit failures actually exercised
- [ ] No-Go code cleanup state: deleted, retained unreachable, or laboratory-only
- [ ] Upstream repository/revision/file/license and raw receipt location/owner
- [ ] Public-source provenance separated from commercial candidate evidence

Missing evidence keeps the route laboratory-only. A positive throughput result
never changes that default by itself.

## Template audit contract

Audit every reusable ledger entry for the following explicit fields. Missing
items stay in the unresolved-receipt index rather than being guessed from a
neighboring run:

- experiment ID, base experiment ID, dependency stack, `supersedes`, and
  `superseded_by`;
- source commit, dirty-diff hash, executable SHA-256, backend artifact SHA-256,
  build profile, compiler flags, and generated-code receipt;
- full environment matrix with set and unset values, plus route-conflict rules;
- public/private buffer table, authoritative epochs/layouts, commit point, and
  injected failure results on both sides of commit;
- selector arithmetic and buffer bounds, including overflow test cases;
- resource table for the exact measured instantiation, not a neighboring
  template;
- raw A/B legs, ordering, warmup, dispersion, thermal/power state, declared
  noise floor, event share, predicted whole gain, measured whole gain, and the
  explanation for any gap;
- sidecar admission list, budget and used bytes, free-memory headroom, cold pack
  time, attempts/hits/misses/reject reasons, and allocation order;
- correctness matrix with explicit pass, fail, pending, or not-applicable for
  every shape, Decode, determinism, KV/Graph/probe/OOM, and sanitizer gate;
- exact rollback flag plus evidence that fallback was executed;
- upstream source/revision/file/license, evidence classification
  (public/internal/commercial), raw receipt URI, and receipt owner; and
- final code state for failed candidates so an unavailable flag or dead
  template is not mistaken for an active route.

For success records, preserve both the winning raw evidence and the boundaries
that prevent generalization. For failure records, preserve the exact mechanism
that was exercised, whether it actually hit/committed, the same-binary control,
and the code-cleanup decision. A zero-hit cache run, identical generated code,
or post-commit contract violation is a mechanism result and must not be
summarized only as a throughput number.
