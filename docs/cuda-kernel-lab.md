# CUDA kernel lab: admission rules and reusable lessons

This file records successful and rejected kernel experiments.  A microbenchmark is
evidence about one boundary, not evidence that the engine is faster.  A candidate is
admitted only after the same binary, model, cache type and request shapes show a whole
Prefill or Decode improvement with its kill switch toggled.

## Admission sequence

1. Preserve a backend semantic fallback.  Metal and other CUDA SM families must not
   select an SM86 experiment accidentally.
2. AOT-compile the candidate before timing.  JIT/compile latency is never inference
   latency.
3. Compare on one device, stream and shape with CUDA events and interleaved A/B/B/A.
4. Gate the operation''s intermediate representation, then `logit_agree.py`,
   `decode_agree.py`, KV resume/grid/Graph and determinism.
5. Run Compute Sanitizer `memcheck`, `initcheck`, `racecheck` and `synccheck` on a
   trace-proven launch.  For `initcheck`, instrument every producer and consumer in a
   filtered chain; instrumenting only the consumer makes writes by an uninstrumented
   producer appear uninitialized.
6. Run `bracket.py` against the same pinned llama.cpp build.  Record prompt tokens,
   cache hits, every raw sample, medians, dispersion and VRAM.  No isolated kernel win
   can substitute for this gate.

## Patterns borrowed from other Triton users

The useful pattern is not "rewrite every CUDA kernel in Triton".  It is to fuse a
strict producer/consumer boundary so a quantized intermediate never becomes a full
precision global-memory tensor.

- vLLM documents `RMSNorm -> quant` and `activation -> quant` fusions specifically to
  remove the intermediate activation read/write.  That maps directly to Imparo''s
  `RMSNorm -> Q8_1` and `GELU x per-layer -> Q8_1` boundaries:
  <https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md>
- SGLang keeps the model-facing operation stable while dispatching to interchangeable
  Triton, DeepGEMM, CUTLASS and other runners.  Imparo follows the same separation:
  semantic backend hook first, architecture/shape implementation second, safe fallback
  last:
  <https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx>
- Unsloth''s RMSNorm assigns one Triton program to one row, promotes the reduction to
  f32 and makes block size/warp count shape decisions outside the kernel.  The reusable
  lesson is explicit numerical and launch policy, not the particular PyTorch wrapper:
  <https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rms_layernorm.py>

## Current SM86 results

### Admitted native Decode candidate

The D512 full-attention Decode route now expands Q4 K/V pairs directly into the
existing shared-memory MMA tiles.  It preserves the staged-F16 MMA, softmax and reverse
combine order while eliminating the global F16 K/V mirror.

- Same-binary long Decode: `46.9 -> 57.4 tok/s`, about `+22.4%`.
- `logit_agree` and 8-step Decode gates pass at 512 and 2000 tokens.
- Graph identity and paged replay, KV resume/grid and determinism pass.
- `memcheck`, producer+consumer `initcheck`, `racecheck` and `synccheck` report zero
  errors/hazards with exit code zero.

This is a native CUDA result, not a Triton result.

### Admitted SM86 virtual-no-seam Prefill route

Short Prefill profiling showed that the two 2560 -> 10240 FFN projections each
launched 320 virtual logical tiles on 320 physical CTAs. There is no cross-CTA K
boundary in that case, but the established NumericSplit=true, SingleSeam=true
kernel still made lane zero scan every possible physical boundary. The admitted
route proves physical_blocks == logical_tiles in the host selector, then calls the
same AOT kernel with a runtime no_seam_scan flag. Its K-stage, MMA, suffix and
write-back arithmetic remain unchanged. The 10240 -> 2560 down projection is
80/30 and therefore remains on the original virtual Stream-K path.

Interleaved same-binary A/B/B/A, q4 K/V:

| Regime | Default medians | Candidate medians | Mean change |
|---|---|---|---:|
| long Prefill, 5651 tokens | 1937.3, 1950.7 | 1973.2, 1955.8 | +1.05% |
| short Prefill, 449 tokens | 1075.2, 1102.8 | 1390.0, 1412.9 | +28.69% |

Admission evidence:

- trace shows virtual-no-seam only for physical == logical, including both
  10240-wide gate/up projections; the down projection remains virtual-stream-k;
- 512-token logit, 512/2000-token eight-step Decode, KV resume at
  256/512/1024/1920, grid, 449/512/1117/2000 determinism and identity/paged Graph
  gates pass;
- a 449-token non-standard Decode probe fails in exactly the same way with the
  route both enabled and disabled, so it is recorded as a pre-existing baseline
  limitation rather than hidden or attributed to this change;
- trace-proven memcheck and initcheck report zero errors; kernel-filtered
  racecheck reports zero hazards/errors/warnings and filtered synccheck reports
  zero errors. Windows test-server termination makes each wrapper report a target
  application error, so these are zero-error kernel results, not strict wrapper
  exit-zero claims. An unfiltered whole-engine racecheck was rejected after its
  HTTP request exceeded 1200 seconds;
- the SM86 cubin reports the reused <true,true> instance at 255 registers,
  zero stack and zero local memory. Dynamic shared memory remains the existing
  launch allocation.

The route is default-on only for SM86 and retains
IMPARO_CUDA_NO_MMQ_VIRTUAL_NO_SEAM=1 as an immediate rollback. Other SMs, Metal,
physical Stream-K and every shape with a real seam retain their prior paths.

### Admitted SM86 virtual-direct-seam Prefill route

After removing the empty seam scan from gate/up, a fresh 449-token profile still
placed about 72.0 ms in the 10240 -> 2560 down projection. Sparse virtual grids
were using the older synchronous staging kernel even though the existing
full-tile primitive already provided cp.async activation staging, double buffering
and a direct one-seam locator. The admitted route is limited to aligned compact
batches whose real token grid is identical to the 512-token virtual grid, complete
128-row tiles, plain projections, and a schedule proven to contain at most one
reference seam per tile. Unaligned tails retain the established compact-token
mapper.

The new dispatch does not invent a new Stream-K schedule. It passes the same
logical tile count and physical worker count to the full-tile primitive, replays
the same rounded reference seam, computes its suffix and prefix independently,
and performs the same suffix + prefix addition. It therefore changes staging and
resource use, not ownership or reduction boundaries.

Same-binary short-Prefill A/B/B/A, q4 K/V, 449 prompt tokens:

| Route | Median samples | Mean |
|---|---|---:|
| rollback/default-before | 1405.5, 1393.3 tok/s | 1399.4 tok/s |
| virtual-direct-seam | 1551.2, 1553.8 tok/s | 1552.5 tok/s |

The mean whole-request improvement is 10.9%. A long-Prefill spot bracket was
neutral within noise (1959.4 -> 1954.0 tok/s), and Decode was unchanged at
57.7 tok/s. Trace proves 42 down-projection launches use
virtual-direct-seam with logical=80 and physical=30; gate/up remain on
virtual-no-seam. Other eligible full-row, aligned, single-seam projections use
the same shape-derived rule rather than a model-name or dimension allow-list.

Admission evidence and limitations:

- 449- and 2000-token logit agreement pass; 2000-token eight-step Decode passes;
  candidate and rollback produce the same 449-token eight-step trail, including
  the pre-existing step-4 fork disagreement;
- 449 determinism passes at 1/4 distinct and an expanded 512 run passes at 1/8;
  an earlier 512 sample reported 2/4 before the expanded rerun, and is retained as
  an anomalous failed observation rather than discarded;
- KV grid reports no off-grid batches, and identity/paged Graph capture and replay
  pass;
- filtered memcheck and initcheck finish with exit code zero and zero errors;
  filtered racecheck reports zero hazards and filtered synccheck reports zero
  errors. Both candidate and rollback produce unrelated final logits when run
  under racecheck instrumentation, so sanitizer-time whole-output agreement is a
  pre-existing/tool perturbation and is not claimed as an acceptance result;
- the actual SM86 cubin instance uses 142 registers, zero stack and zero local
  memory. Dynamic shared memory remains the existing 75,776-byte allocation.

The route is default-on only for SM86. Set
IMPARO_CUDA_NO_MMQ_VIRTUAL_DIRECT_SEAM=1 to restore the prior virtual Stream-K
kernel immediately. Metal, other SMs, epilogue projections, unaligned compact
batches and multi-seam shapes are unchanged.

### Admitted SM86 fused D256 Prefill attention

The post-MMQ profile made the next whole-Prefill bottleneck explicit. At 449
prompt tokens the staged D256 family spent 8.663 ms in Values, 4.107 ms in
Scores, 3.089 ms in Softmax and 0.026 ms in Q-cache work (15.885 ms summed,
about 16.89 ms under the outer attention event). The fused candidate keeps Q
resident, stages each 32-key K/V batch with cp.async, keeps scores and
probabilities inside one 128-thread CTA, and writes only a real Stream-K seam.
This removes the staged score/probability global-memory round trip and replaces
three launches with one without changing the public backend ABI.
The post-admission 449-token profile totals 5.66 ms across the 35 fused D256
layers, down 64.4% from the staged 15.885 ms phase sum. The same profile keeps
whole-request GPU time as the admission ruler rather than promoting this stage
number by itself.

The first fast candidate passed one-shot logits but failed the recurrent
2000-token Decode gate at steps 3 and 5. It used physical Stream-K bounds while
the admitted staged route used the stable virtual 512-token scheduling cell.
The fix was not a wider tolerance: fused now calls
stable_virtual_stream_segment_bounds with the same absolute query position,
KV head, ring capacity, virtual worker count, reverse segment order and f32
fixup as staged attention. Schedules with more than two real segments fail
closed to the staged N-way combiner.

Same-binary A/B/B/A, q4 K/V:

| Workload | staged medians | fused medians | mean change |
|---|---:|---:|---:|
| 449-token Prefill | 1540.4, 1538.1 tok/s | 1615.0, 1587.4 tok/s | +4.0% |
| 5651-token Prefill | 1954.6, 1946.5 tok/s | 2059.7, 2065.3 tok/s | +5.7% |
| 449-token Decode | 66.7, 66.2 tok/s | 66.6, 65.8 tok/s | -0.4% |
| 5651-token Decode | 57.6, 57.4 tok/s | 57.6, 57.4 tok/s | neutral |

A fresh default-on interleaved bracket measured Imparo/llama.cpp at
2076.9/2243.4 tok/s long Prefill and 57.8/58.3 tok/s long Decode, plus
1586.8/2295.7 tok/s short Prefill and 66.1/64.0 tok/s short Decode. The chat
templates reported one more prompt token for Imparo (5651/5650 and 449/448), so
these are a close whole-engine bracket, not an exact same-token microbenchmark.

Admission evidence:

- fork logit agreement passes at 128, 449, 512 and 2000 tokens with worst
  delta-to-top1 differences 0.18378, 0.16664, 0.11079 and 0.71337
  respectively (q4 K/V, tolerance 0.75);
- 2000-token eight-step Decode agreement passes 8/8 after the virtual-seam
  repair; the previously failing steps 3 and 5 are now 0.64242 and 0.42338;
- 744- and 2000-token resume points are byte-equal, the 64-token batch grid has
  no off-grid result, and identity/paged CUDA Graph each capture and replay once;
- 128/449/512/2000 Prefill determinism is 1/4 distinct and repeated 521-token
  Decode is 1/4 distinct;
- filtered memcheck is exit zero with zero errors. Initcheck must instrument
  the actual producer k_head_norm_rope_hadamard (and fallback cache_scaled_q)
  together with the fused consumer; consumer-only filtering reports a false
  uninitialised-Q read. Producer+consumer initcheck exits zero with zero errors.
  Racecheck reports zero hazards/errors/warnings and synccheck zero errors;
- as with the admitted MMQ routes, racecheck/synccheck instrumentation changes
  the final model distribution, so sanitizer-time whole-output agreement is
  not claimed. Their evidence is limited to the trace-proven kernel hazards;
- both ringed and identity SM86 cubin instances use 238 registers, 39,936 bytes
  static shared memory, zero stack and zero local memory.

The versioned attn_d256_fused knob is default-on only for SM86. Set
IMPARO_CUDA_NO_ATTN_D256_FUSED=1 for immediate rollback. Other SMs, Metal,
non-D256 shapes, non-wide query tiles and schedules needing more than two
segments retain the staged implementation.

### Rejected Prefill scheduling variants

- The experimental SM86 gate/up paired 512-token kernel passes 512- and
  2000-token logit agreement, but same-binary long-Prefill A/B/B/A measures
  2020.9 and 2013.9 tok/s against staged/default 2088.1 and 2063.5 tok/s:
  about -2.8% by the mean. Computing both MMQs serially in one CTA does not
  reuse their independent weights, and the saved intermediate traffic does not
  repay the occupancy/register pressure. IMPARO_CUDA_PREFILL_PAIR_V1 remains
  diagnostic opt-in and is not an admitted route.
- Virtual K128 halved the shared K stage and passed 2000-token KV resume plus
  Decode, but whole short Prefill fell from 1075.2 to 1035.8 tok/s (-3.7%);
  it was removed.
- Same-CTA gate/up pairing serialized two large accumulators. Long Prefill fell
  from 1937.3 to 1909.4 tok/s, so the opt-in prototype is not admitted.
- Assigning one full CTA to every short-tail tile reached roughly +90% short
  Prefill, but changed Stream-K reduction ownership. KV resume and long Decode
  gates failed in the affected variants, so none entered default dispatch.
- A NumericSplit=false no-seam instance showed the right performance direction,
  but it changed the compiled numerical route unnecessarily. The admitted version
  instead reuses the established numerical instance and skips only the proven-empty
  scan at runtime.
- A virtual-no-seam fused-Q8 epilogue attempted to remove the 42 standalone
  10240-wide activation quantizers. Its first short-Prefill sample was only about
  +0.6% and the generated Q8 layout changed the next projection enough to fail
  449-token top-1 agreement. Replacing the standalone float4 subgroup reduction
  with the full-tile accumulator lane pattern did not restore correctness. The
  candidate and its runtime flag were removed; future fusion must prove the Q8
  scratch bytes against k_quantize_q8_1_mmq before any whole-engine timing claim.

- The existing IMPARO_CUDA_COOP_Q4 route passed the 449-token distribution
  check, but it did not improve whole-Prefill throughput. A same-binary
  5651-token A/B/B/A bracket measured default medians of 2085.4 and
  2079.3 tok/s versus cooperative medians of 2025.0 and 2026.8 tok/s.
  The two-leg means are 2082.35 versus 2025.90 tok/s, a 2.71% regression.
  The earlier short-Prefill bracket was neutral to slightly negative
  (about -0.17%). This route remains diagnostic-only; no default selector
  change is justified.

### Experimental SM86 packed-Q4 down pipeline

IMPARO_CUDA_MMQ_DOWN_PACKED_PIPE_V1=1 selects a strict SM86-only laboratory
route for the 449-token 10240 -> 2560 down projection. It preserves the
admitted virtual Stream-K worker and direct-seam ownership, K traversal order,
suffix-before-prefix order and rounded seam. Two 18,432-byte raw packed-Q4
stages feed one 18,432-byte expanded-weight stage and one 18,432-byte Q8
activation stage, for 73,728 bytes of dynamic shared memory. Unsupported shapes,
non-resident weights and other architectures fail closed to the generic MMQ.

The v1 cubin uses 154 registers, zero stack and zero local memory. Fork
agreement passed at 128, 449, 512 and 2000 prompt tokens with worst
delta-to-top1 values 0.18378, 0.16664, 0.11079 and 0.71337. The 2000-token
eight-step recurrent Decode gate passed 8/8. CUDA-event profiling of all 42
down projections improved from 51.968448 ms to 45.848576 ms, or 11.78%.

That kernel-local improvement did not translate into a sufficiently stable
whole-engine win. Two independent short-Prefill A/B/B/A brackets produced
1614.05 versus 1626.40 tok/s (+0.77%) and 1621.40 versus 1653.55 tok/s
(+1.98%); pooling all eight legs gives +1.38%. A v2 experiment double-buffered
the activation stage as well, raised dynamic shared memory to 92,160 bytes and
reduced the event improvement to 9.33%, so source was restored to v1.

The v1 route therefore remains explicit opt-in and is not an admitted/default
dispatch. The rejection happened at the whole-engine value gate, before the
sanitizer admission matrix, so no sanitizer claim is made. The reusable lesson
is that packed-weight prefetch can reduce the down-projection event, but the
remaining Prefill bottleneck is distributed broadly enough that this isolated
gain is not the requested global breakthrough.

### Rejected D256 persistent F16 KV shadow

An opt-in SM86 laboratory slice tested whether a persistent F16 shadow could
remove repeated Q4 KV dequantization from the 512-token D256 sliding Decode
path. The slice reused the high-address tail of the existing Kdq/Vdq buffers,
materialized one exact 1024-slot layer from the rounded Q4 representation and
updated later rows in the Q4 store kernel. It changed no public ABI and left the
default path untouched.

The route was trace-proven at a 2000-token prompt: layer 0 registered a
1,048,576-byte K and V tail slice, materialized both shadows and entered the
F16 small-attention consumer. It then failed the fixed recurrent distribution
gate before performance admission. In the 2000-token eight-step q4/q4 Decode
oracle, steps 4 and 5 reached max delta differences 1.01618 and 0.77203 against
the 0.75 limit. Top-1 remained equal, but that is not sufficient for the
distribution contract.

The failure identifies a numerical-class boundary rather than a tolerance to
widen: persistent F16 storage adds a half writeback after Q4 dequantization,
whereas the admitted direct-Q4 vector route consumes the dequantized float
inside its established reduction. Expanding the shadow to all 35 D256 layers
would amplify rather than repair that seam. The implementation was therefore
removed, no performance or sanitizer claim is made, and future cache work must
preserve the direct-Q4 vector arithmetic or earn a new full numerical receipt.

### Prefill piecewise Graph Go/No-Go

A default-off IMPARO_CUDA_NSYS_CAPTURE boundary pairs cudaProfilerStart/Stop
with each non-Decode forward so Nsight Systems can measure a complete tile
without enabling the synchronization-heavy per-op profiler. A 2000-token run
captured four consecutive Prefill batches to
target/nsys/prefill2000.1..4.nsys-rep. The exported CUDA GPU timelines report:

| batch | GPU events | kernel + copy | complete span | idle gaps | gap share |
|---|---:|---:|---:|---:|---:|
| first/warm | 1086 | 196.04 ms | 226.99 ms | 30.95 ms | 13.63% |
| canonical 2 | 1106 | 224.57 ms | 227.19 ms | 2.62 ms | 1.16% |
| canonical 3 | 1127 | 443.73 ms | 447.25 ms | 3.53 ms | 0.79% |
| tail | 1079 | 552.63 ms | 556.16 ms | 3.53 ms | 0.63% |

The first batch's 23.02 ms maximum gap is allocation and cold initialization,
not replayable launch overhead. Every warmed batch has less than 1.2% total
GPU idle time between consecutive operations. Even a perfect piecewise Graph
cannot remove kernel execution time, so its measured whole-Prefill upper bound
is below the laboratory 3% Go threshold. Building and maintaining 42 per-layer
post-attention graphs is therefore a No-Go on this SM86 workload. The next
global Prefill work must reduce execution time in the dominant gate/up/down
MMQs rather than relabeling cold setup or launch count as a throughput gain.

### Rejected Triton Q4 x Q8 candidate

The first Triton Q4 x Q8 MMQ candidate reached only about `0.3-0.4x` the existing native
SM86 kernel.  It is not eligible for dispatch.  Useful failure evidence:

- the packed-nibble decode and scale handling did not map efficiently to the generated
  instruction sequence;
- the candidate did not reproduce the native tile/reduction ownership;
- AOT removes compile latency but cannot repair excess instructions, poor occupancy or
  redundant global traffic.

PTX/SASS and resource evidence from rejected candidates remains useful for designing a
native CUDA replacement, but a rejected binary must not enter the runtime selector.

### Rejected compact-tail asynchronous activation staging

The SM86 virtual-no-seam gate/up laboratory route used predicated 16-byte
cp.async copies to overlap phase-0 Q8 activation staging with the independent
packed-Q4 transform. Invalid compact tokens used cp.async zero-fill from a legal
allocation rather than a null source. The experiment was limited to
2560 -> 10240 gate/up tiles below 512 tokens and remained explicit opt-in through
IMPARO_CUDA_MMQ_VIRTUAL_ASYNC_ACTIVATION.

At 449 tokens, CUDA-event A/B/B/A reduced combined gate/up time from
145.197696 ms to 134.492448 ms, about 1.0796x throughput. The canonical
512-token shape improved only about 1.012x and was excluded. More importantly,
production-server short-Prefill A/B/B/A medians were:

| Route | Prefill medians | Combined |
|---|---:|---:|
| established virtual-no-seam | 1629.1, 1598.4 tok/s | 1613.75 tok/s |
| async activation staging | 1631.0, 1627.7 tok/s | 1629.35 tok/s |

The whole-engine improvement is only 0.97%, below the 3% admission threshold.
The route therefore remains laboratory-only and is not selected by default.
This is a useful reminder that an 8% improvement in one dominant-looking MMQ
slice can still be diluted by the complete model workflow; operation events
never substitute for the production bracket.

### Triton RMSNorm -> Q8_1 candidate

For measured small work counts, the AOT Triton boundary was `1.375-1.440x` the previous
standalone native boundary.  This remains a shape-scoped candidate: it is not a claim
about whole-engine Prefill or Decode until it is connected through a receipt-backed
selector and passes the bracket gate.

### PLE tail experiment

Passing `output_scale` through the existing fused RMSNorm+add kernel removes one scale
launch and one full-width pass per PLE layer.  The first same-binary A/B measured about
`+0.8%` long Prefill, noise-level short Prefill and no material Decode change.  Keep the
semantic seam and kill switch, but do not call this the requested global breakthrough.

The next candidate is the fixed SM86 `2560 -> 256 -> 2560` PLE bottleneck:

`gate MMQ -> GELU x per-layer -> Q8_1 -> project MMQ`

The two Q4 matrices total about 720 KiB and the quantized intermediate for eight tokens
is about 2.25 KiB.  Admission requires the whole long-Prefill bracket to improve at
least 5%, no short-Prefill regression beyond noise, no Decode regression, no spills,
and all correctness/sanitizer gates above.

### Experimental SM86 fused PLE gate

The fixed-shape laboratory route behind IMPARO_CUDA_PLE_FUSED_SM86 fuses the
canonical 512-token PLE 2560 -> 256 gate, GELU-times-per-layer semantic
epilogue and Q8_1 publication consumed by the existing 256 -> 2560 back
projection. A 32-row x 64-token CTA replays the established 30-worker
Stream-K seams and nearest-to-farthest prefix fixup order inside each original
128x128 logical tile. Other token widths, SMs, paged weights and Metal fail
closed to the existing path.

The first resource-clean implementation uses 128 registers, 32,512 bytes of
dynamic shared memory, and zero stack/local memory. The 512-token q4/q4 fork
gate reproduces the admitted result: overlap 9/10 and max delta 0.11079 at the
0.75 distribution tolerance. Same-binary 5651-token A/B/B/A measured default
medians 2058.1 and 2073.3 tok/s against fused medians 2100.9 and 2105.9 tok/s.
The two-leg means are 2065.7 versus 2103.4 tok/s, a 1.83% whole-Prefill gain.

Two follow-ups did not add value:

- selecting the existing 64-row full-tile kernel for the narrow back
  projection changed the fused-gate mean from 2097.45 to 2100.65 tok/s,
  only +0.15%; the selector was removed;
- overlapping the next packed-weight stage with the current MMA lowered
  register use to 124 but warmed fused-gate events remained around 0.061 ms
  versus roughly 0.058-0.060 ms for the simpler schedule. The added pipeline
  was reverted.

The fused gate remains explicit opt-in while recurrent, byte-level Q8 and
sanitizer evidence is incomplete. Its positive whole-engine result is retained
as a reusable candidate, but 1.83% is below the 3% admission target and is not
the requested global breakthrough by itself.

### Admitted SM86 D256 GQA4 Decode KV sharing

The long-Decode profile showed that the 35 D256 sliding-attention layers still
loaded and decoded the same Q4 K/V row four times for the four query heads that
share one KV head. The admitted SM86 route remaps each CTA from four source
stripes of one query head to four query heads over one source stripe. A 32-row
Q4 K/V tile is staged once and consumed by all four warps. This follows the
multi-query grouping used by mature paged-attention implementations, while
retaining Imparo's existing vector-FA numerical class and fixed ring schedule.

The first implementation added a third kernel to combine source-warp child
records. It reduced full D256 attention from 0.063488 ms to 0.046976 ms
(1.351x throughput), but whole Decode improved only about 1.9%. The admitted
version fuses source-warp and schedule-partial combination into one epilogue,
restoring the old two-launch structure and removing the 33 KiB partial
write/read round trip per layer.

Resource evidence from the SM86 release artifact:

- shared partial kernel: 105 registers, 0 stack, 0 local, 6,400 bytes shared;
- fused final epilogue: 52 registers, 0 stack, 0 local, 128 bytes shared.

The numerical replay initially differed from the old partials by one or two ULP
because nvcc contracted scale*x+accumulator into FFMA. Explicit round-to-nearest
multiply and add at the source-warp boundary restored exact behavior. The
diagnostic dual-run then reported 0/8,256 mismatches for every D256 layer.

Correctness and state evidence:

- 512-token and 2,000-token eight-step q4_0 Decode agreement: 8/8 PASS;
- q4_0 determinism at 512 and 2,000 tokens: 1/4 distinct; 521-token
  48-step Decode: 1/4 distinct;
- the complete engine KV battery passed resume, grid, scramble, identity and
  paged Graph, and spill/restore for every prescribed KV pairing;
- Graph capture retained all 179/179 dynamic nodes and replayed;
- filtered memcheck and initcheck: zero errors; racecheck: zero hazards,
  errors, or warnings; synccheck: zero errors.

Production server A/B/B/A, q4_0, 5,651 prompt tokens and 128 generated tokens:

| Route | Decode medians | Combined |
|---|---:|---:|
| established per-head | 57.6, 57.4 tok/s | 57.50 tok/s |
| GQA4 shared K/V | 59.6, 59.3 tok/s | 59.45 tok/s |

The long-Decode throughput gain is 3.39%. Short Decode is neutral within noise:
66.25 versus 66.40 tok/s. Prefill does not select this route and remained
neutral within noise. The fixed SM86/P4/D256/GQA4/Q4 route is default-on;
IMPARO_CUDA_NO_ATTN_D256_GQA4=1 restores the established per-head kernel.
Other SMs, cache types, head shapes, early schedule buckets and Metal retain
their previous paths.

### Admitted SM86 D512 StableCell fused Prefill attention

Phase-A profiling showed that the staged D512 path
(scores_d512 + softmax_d512_stream + values_partitioned_stream_d512)
occupied 4.74-5.35% of warmed long-Prefill GPU time. The admitted fused route
keeps Q/K/V half inputs, scores, probabilities and the half numerator inside
one CTA and removes the score/probability global round trip and two launches.
It reuses the exact staged selector in this order:

1. stable virtual Stream-K bounds when virtual blocks are present;
2. StableCell canonical bounds when canonical_parts is non-zero;
3. the established physical Stream-K bounds otherwise.

The reverse segment traversal, upper-then-lower D512 QK order, half probability
rounding, half numerator rescale, explicit rn partition sum and segment combine
order are unchanged. The default selector is deliberately narrow: SM86,
D512, GQA4, wide Prefill, ring zero, resident F16 staging, StableCell geometry,
no virtual blocks and no more than the two canonical segments. The physical
Stream-K fused experiment remains opt-in. IMPARO_CUDA_NO_ATTN_D512_FUSED=1
restores the staged scores/softmax/values path immediately.

Correctness and state evidence:

- q4_0 logit agreement at 128, 449, 512 and 2,000 tokens passed; the 2,000-token
  maximum distribution delta was 0.71337 at the fixed 0.75 limit;
- 2,000-token, eight-step recurrent Decode agreement passed 8/8;
- the complete CUDA KV battery passed byte-exact resume at every prescribed
  split, the grid gate, five scramble layouts, identity and paged Graph capture
  and replay, and all five spill/restore layouts;
- default-on route tracing reports stage=flash-d512 without an opt-in variable;
- diagnostic and LFM2 static contracts pass 2/2 and 19/19.

Resource evidence from the SM86 release cubin is 248 registers, zero stack and
zero local memory. Dynamic shared memory is the existing 90,112-byte opt-in
allocation, so the kernel intentionally owns one resident CTA per SM.
Filtered memcheck and producer+consumer initcheck both exit zero with zero
errors. Filtered racecheck reports zero hazards/errors/warnings and synccheck
reports zero errors; as with the other high-register attention kernels, those
two instrumentations make final logits non-finite and the fail-closed
application exits one, so no sanitizer-time output-correctness claim is made.

Production server A/B/B/A, q4_0, 5,651 prompt tokens and 128 generated tokens:

| Route | Prefill medians | Combined | Decode |
|---|---:|---:|---:|
| staged D512 | 2082.1, 2055.7 tok/s | 2068.90 tok/s | 59.3-59.5 tok/s |
| fused StableCell D512 | 2191.8, 2178.9 tok/s | 2185.35 tok/s | 59.3-59.4 tok/s |

This is a 5.63% whole long-Prefill throughput gain with neutral Decode, not a
local-kernel projection. It is default-on for the receipt-backed SM86 route.

### Admitted SM86 64-row virtual-no-seam gate/up MMQ

The short-Prefill profile showed that the two 2560 -> 10240 gate/up MMQs
dominate the remaining 449-token gap. Their established 128-row kernel carried
nearly 255 registers per thread and one resident CTA per SM. The admitted
variant reuses the existing 64-row full-K primitive: it changes only ownership
of independent output rows, while preserving every per-output K block, integer
dot, scale multiply and f32 accumulation in the same order. The smaller result
set uses 90 registers, zero stack/local memory and 37,888 bytes of dynamic
shared memory, allowing two resident CTAs on SM86.

The first broad experiment proved the occupancy hypothesis but also exposed an
important selector lesson: 64 rows made the wide gate/up pair much faster while
duplicated activation staging slowed smaller projections. The admitted selector
is therefore fail-closed to SM86, aligned no-seam virtual schedules,
2560 -> 10240, and canonical 512-aligned batch starts. Other shapes, canonical
512 batches, other SMs and Metal retain their established routes.

CUDA-event evidence at 449 tokens:

| Projection | 128-row total | 64-row total | Throughput |
|---|---:|---:|---:|
| gate/up, epilogue 0, 42 calls | 62.92 ms | 43.90 ms | 1.433x |
| gate/up, epilogue 1, 42 calls | 75.68 ms | 51.97 ms | 1.456x |
| combined gate/up | 138.60 ms | 95.87 ms | 1.446x |

Production-server same-binary A/B/B/A, q4_0:

| Regime | established medians | 64-row medians | Combined gain |
|---|---:|---:|---:|
| 449-token Prefill | 1646.3, 1631.4 tok/s | 1915.2, 1943.8 tok/s | +17.73% |
| 5651-token Prefill | 2172.4, 2184.0 tok/s | 2207.7, 2206.5 tok/s | +1.33% |
| Decode | 62.9, 63.3 tok/s | 61.9, 62.5 tok/s | noise-level |

Correctness passed the fixed q4_0 oracle at 128, 449, 512 and 2,000 prompt
tokens; 2,000-token recurrent Decode passed 8/8. Compute Sanitizer memcheck and
initcheck report zero errors. Kernel-filtered racecheck covers both direct and
gated template instances with zero hazards/errors/warnings, and synccheck
reports zero errors. CUDA diagnostic plus LFM2 contracts pass 2/2 and 20/20.

The route is default-on only inside the strict selector above.
IMPARO_CUDA_NO_MMQ_VIRTUAL_R64=1 restores the established 128-row route
immediately.

## Latest same-ruler baseline

Pinned llama.cpp revision `4695f001fece1660d8bb1b3748f50726ddcc100b`, q4 K/V,
CUDA SM86, explicit Flash Attention, q4 K/V, two interleaved clean pairs per
regime:

| Regime | Imparo | llama.cpp | Imparo / llama.cpp |
|---|---:|---:|---:|
| long Prefill | 2188.55 tok/s | 2233.75 tok/s | 0.9798x |
| long Decode | 59.40 tok/s | 58.20 tok/s | 1.0206x |
| short Prefill | 1591.80 tok/s | 2303.60 tok/s | 0.6910x |
| short Decode | 66.05 tok/s | 64.10 tok/s | 1.0304x |

The zeraix llama fork at e0949576a was also attempted as the performance bar.
Its first long request measured 2146.0 Prefill / 57.7 Decode, but the process
terminated abruptly during the second request with a connection reset and no
diagnostic in its server log. That incomplete leg is retained as failure
evidence and is not used for a ratio claim. The stable upstream-base bracket
above is therefore the current reproducible Windows/CUDA comparison.

A post-r64 same-text I/L/L/I rerun measured Imparo short-Prefill medians
1916.8 and 1923.1 tok/s versus upstream medians 2375.4 and 2372.2 tok/s;
Decode medians were 62.2/62.2 versus 61.6/61.8 tok/s. The servers reported
449 and 448 prompt tokens respectively. Both execute the same 512-token MMQ
tile, but the repository requires equal reported token counts for a formal
cross-engine ratio, so the approximate 0.809x Prefill ratio is directional
evidence only. The accepted claim from this round is the same-binary
17.73% improvement over the established Imparo route.

Long Prefill is now within about 2.0% of the pinned upstream CUDA/FA reference,
and Decode is ahead in both regimes. Short Prefill remains the largest global
gap; the next optimization must target its small/compact MMQ and attention
mixture rather than extrapolating the long-context D512 result.
