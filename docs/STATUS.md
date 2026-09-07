# Status and lessons

This file holds the current verified state, the lessons that were expensive to learn, and
the open list. The full day-by-day ledger (every superseded head-to-head, every dead
experiment with its numbers) lives in this file's git history.

## LFM2 CUDA SM86 shared-K/V D64 attention v37 (2026-09-02)

CUDA space v37 adds a second physical staging graph to the existing D64/GQA4 whole-K
MMA Prefill kernel. The original graph lets each of four query warps load the same K
and V tile independently. The new graph has all 128 CTA threads stage one complete
64-key by D64 K tile once in shared memory, lets all four query warps reuse it, then
reuses the same allocation for V. MMA order, online-softmax order, causal masking and
the public output remain unchanged. The old graph remains compiled as the fallback.

The route is controlled by the default-off, receipt-bound
`attn_d64_mma_shared_kv_min_tokens` tuner boundary. It applies from model facts
(D64 and GQA4), not a model name. The selected LFM2 value is 129: 128-token requests
retain the original graph, while ordinary 512-token Prefill chunks and the 331-token
tail of the measured long request use shared K/V. Setting the knob to zero immediately
restores v36; `IMPARO_CUDA_NO_ATTN_D64_SHARED_KV` remains a laboratory kill switch.
Metal and non-D64 routes are unchanged.

A same-binary control/candidate interleave at 5963 tokens produced control medians
3595.68/3589.40 tok/s and candidate medians 3839.59/3859.66 tok/s. Averaged centers
improve from 3592.54 to 3849.62 tok/s, or 7.16%. An additional clean run of the final
receipted threshold config measured 3866.42 tok/s. Decode remained statistically flat.
At 455 tokens the first crossover pair measured 4074.42 versus 4102.72 tok/s (+0.69%).
Before the threshold was introduced, forcing shared K/V at 128 measured 3370.65 versus
3316.41 tok/s (-1.61%); this is why the candidate is a per-request boundary rather than
a global boolean.

Both threshold=129 and threshold=0 configs passed the fixed six-gate Q8 suite: logits
agreement at 128/449/512 and eight-step Decode agreement at 128/512/2000. The focused
SM86 native build/registry contract and the SM86 release product build passed. These
are internal route-selection measurements, not a fresh external llama.cpp bracket.
The reusable lesson is that GQA query-warps should share physically identical K/V
loads only after the request is large enough to amortize CTA barriers; the crossover
belongs in the tuner space.

## LFM2 CUDA SM86 private SwiGLU-to-Down transaction v36 (2026-09-02)

CUDA space v36 adds the default-off, receipt-bound
`mmq_q8_tm_silu_private_down_min_tokens` decision. On an eligible resident Q8_0_TM
SwiGLU FFN, Gate still writes the backend scratch buffer, but the Up epilogue writes
only the established D4 Q8 sidecar. Down consumes that sidecar immediately. Compared
with v35, the transaction therefore also removes the dense gated-intermediate write;
the model workflow observes only the completed Down output. Probes and every ineligible
call retain the materialized fallback. Metal inherits the default `false` capability
and is unchanged.

The selected per-request boundary is 129 tokens. It is expressed as a tuner knob with
the ladder `off, 9, 129, 257, 513`, not as a model name or a fixed request-length branch.
Setting it to zero restores v35. Two independently interleaved same-binary pairs, each
with one warmup and two measured requests, produced:

- 455 tokens: control medians 3766.99/3844.16 tok/s; candidate medians
  4013.14/4013.67 tok/s; averaged centers improve by 5.46%.
- 5963 tokens: control medians 3399.01/3395.55 tok/s; candidate medians
  3579.88/3573.33 tok/s; averaged centers improve by 5.28%.

These are internal single-knob route-selection results and are not a fresh llama.cpp
comparison. Both candidate and control passed all six fixed Q8 correctness gates:
logits at 128/449/512 and eight-step Decode at 128/512/2000. The SM86 release product
build passed, as did 22 complete-transaction contracts, 24 LFM2 contracts and the
native default-off registry test. Receipted configs and full commands are recorded in
`docs/evidence/cuda-sm86-lfm2-q8/`.

The positive result establishes that the complete backend transaction has materially
more leverage than v35's quantizer-only removal. The next Prefill work should profile
the remaining long-prompt MMQ/attention time rather than adding another dense-sidecar
variant.

## LFM2 CUDA SM86 fused SwiGLU D4 sidecar v35 (2026-09-02)

CUDA space v35 adds the default-off, receipt-bound
`mmq_q8_tm_silu_q8_sidecar_min_tokens` choice. The Q8_0_TM Gate/Up kernel now has an
epilogue that keeps each completed 128-by-128 SwiGLU tile on chip long enough to emit
the exact existing D4 `BlockQ8_1Mmq` layout. The following Down projection can consume
that cache and omit the standalone dense-to-Q8 quantizer and its global reread. Dense
output remains available for probes and fallback. Allocation failure falls back to the
v34 dense epilogue without publishing a sidecar.

The knob is a workload boundary rather than a model-name or exact-length special case.
Its candidate ladder is `off, 9, 129, 257, 513`; the selected LFM2 value is 129. An
initial unbounded screen at 128 tokens was inconclusive and slightly negative
(3302.03 vs 3280.09 tok/s, -0.66%, with 3.28% within-leg drift), so the final candidate
leaves 128 on the v34 route. Stable same-binary, single-knob screens measured:

- 455 tokens: 3734.76 to 3782.82 tok/s, +1.29%;
- 5963 tokens: 3375.08 to 3404.18 tok/s, +0.86%.

Each leg used Q8 K/V, context 8192, one warmup and two measured requests. These are
internal route-selection results, not a new llama.cpp bracket. Both the sidecar=129
candidate and sidecar=off control passed all six fixed Q8 correctness gates. The SM86
native build, the new knob registry test, and all 24 LFM2 route contracts pass. The
tracked configs and receipts are under `docs/evidence/cuda-sm86-lfm2-q8/`.

The result proves the transaction boundary is reusable and positive, but also shows
that saving only the standalone quantizer is not a large Prefill breakthrough. A future
candidate should test whether Down can consume a private sidecar without publishing the
dense intermediate; it must remain separately tuned and receipt-gated.

## LFM2 CUDA SM86 Q8_0_TM SiLU pair v34 (2026-09-02)

CUDA space v34 introduces the default-off, receipt-bound
`mmq_q8_tm_silu_pair` choice. For a dense aligned Q8_0_TM SwiGLU pair, the Up MMQ
reads the completed Gate element and publishes `silu(gate) * up` in the same output
store. This removes one full Up activation write and the standalone SiLU-multiply
launch without changing the Gate arithmetic, MMQ reduction order, Decode route, Metal,
or any non-Q8_0_TM model. The backend exposes the pair only when the versioned knob is
selected; ineligible or non-resident calls retain the established materialized path.

The v34 config extends the sealed v33 LFM2 route by one knob and passed the same six
fixed Q8 llama correctness gates. The SM86 native build succeeded; 12 tuner registry
tests, three Step-1 contracts, and all 24 LFM2 CUDA route contracts pass.

The first same-binary on/off screen is a material improvement rather than launch noise:

- 455 tokens: 3481.99 to 3810.52 tok/s, +9.44%;
- 5963 tokens: 3116.76 to 3391.35 tok/s, +8.81%.

These numbers isolate the transaction with `IMPARO_CUDA_NO_MATMAT_GATED=1`; they are
not a new external llama.cpp bracket. The candidate config and fixed-gate receipt are
`docs/evidence/cuda-sm86-lfm2-q8/lfm2-sm86-q8-tm-silu-v34.txt` and its adjacent
`.receipt.json`.

## LFM2 CUDA SM86 Q8_0_TM readers v33 (2026-09-02)

CUDA space v33 implements the private `Q8_0_TM` projection layout on CUDA instead of
merely accepting its wire kind. The common dispatch now routes kind 3 through separate
tile-major readers for MMQ, MMVQ and the conservative F32 fallback. Paged weights retain
complete 8-row file units; a cache that cannot hold one complete unit fails explicitly.
The existing `mmq_q8_aligned_whole_k` knob applies to either Q8 layout, while the model
fingerprint keeps their configurations and receipts independent.

The first correct TM MMQ reused row-major thread ownership and exposed 256-byte-stride
loads. Its replacement assigns consecutive lanes to the eight rows of an on-file unit
and stages each row with two aligned 16-byte loads. On the RTX 3060 Laptop GPU, the
real-shape PrefillGemm tuner screen selected aligned whole-K: 920.1 to 802.8 us, or
+14.6% throughput. A 455-token whole-model laboratory screen, excluding each process's
first warm-up forward, measured 155.938 ms for the original Q8_0 file and 116.676 ms for
the converted TM file, or 1.337x. This is an internal same-engine file-layout A/B, not a
new llama.cpp comparison.

The first address-correct TM MMVQ cost 31.153 ms per Decode forward against 25.271 ms
for row-major Q8_0. A one-token TM kernel now owns one complete 8-row file tile, stages
successive units contiguously, reuses each activation block across all eight rows, and
retains the original 32-way block partition and reduction tree. The same 16-step screen
then measured 25.709 ms, reducing the gap from 23.3% to 1.7%. Every step hash and the
complete generated-token trail matched the original file.

Correctness evidence is byte-exact between original and converted model logits for
128-token MMQ, 3-token generic MMVQ and 16-token conservative F32 fallback inputs. The
specialized one-token path also matched all 16 Decode-step hashes. The SM86 product
build, 24 LFM2 route contracts and the v33 release identity contract pass. A forced-fit
run reserved 4 GiB of device budget and limited the page cache to 4 MiB, making the
2.7 GiB model non-resident and splitting its large projections. Both 3-token MMVQ and
128-token MMQ completed through the dual-plane page builder and remained byte-identical
to the resident original-file outputs.

The converted file is `LFM2.5-2.6B-Q8_0.tm.gguf`; the original remains unchanged. The
combined v33 config enables the independently selected v31 vector-Decode route and the
v33 aligned whole-K TM Prefill route. It passed all six Q8 llama gates and was sealed.
The direct interleaved bracket used the `e0949576a` performance fork, llama FA enabled,
context checkpoints 32, Q8 K/V, ubatch 512 and 32 generated tokens:

- 455 tokens: Prefill 3848.9 vs 3009.0 tok/s (1.279x); Decode 79.8 vs 78.7 (1.014x).
- 5963 tokens: Prefill 3310.8 vs 3851.5 tok/s (0.860x); Decode 73.4 vs 74.9 (0.980x).

Short Prefill is now above the open-source parity bar. Long Prefill improved by 30.9%
over v31 inside Imparo but remains 14.0% below the reference and is the next optimization
target. Decode remains at parity in both measured regimes.


## LFM2 CUDA SM86 aligned whole-K Q8 MMQ v32 (2026-09-02)

CUDA space v32 adds the default-off `mmq_q8_aligned_whole_k` Prefill knob. The
specialization keeps the established 128x128 Q8_0 x Q8_1 arithmetic loop but removes
runtime replay planning, K-tail predicates, row-tail predicates, and replay writeback
from resident projections whose K and output rows are fully aligned. Eligibility is
checked per dispatch; paged or unaligned projections continue through the existing
general MMQ route. The knob is scoped by Q8 weight kind rather than one model name or
one fixed model dimension.

On the RTX 3060 Laptop GPU, the v32 tuner selected value 1 on LFM2.5-2.6B Q8_0:
the real-shape PrefillGemm workload changed from 1154.7 to 1112.3 us (+3.81%). Two
interleaved 455-token engine pairs independently favored the candidate. Their stable
CUDA-event forward medians changed from 158.99 to 154.73 ms (-2.68%), while the matching
server Prefill medians changed from 2768.8 to 2847.3 tok/s (+2.84%).

The candidate and control produced byte-identical 512,000-byte dumps covering all
128,000 final logits at 455 tokens. The final binary then passed all six fixed
`cuda-llama-fa-q8_0` v1 gates (logits at 128/449/512 and eight-step decode at
128/512/2000), and the ordinary runtime loaded the sealed v32 config without any
laboratory route variable. This phase did not rerun the external llama performance
bracket, so the v31 table below remains the latest direct llama comparison rather than
being extrapolated from the internal gain.

Evidence and exact commands are recorded in
`docs/evidence/cuda-sm86-lfm2-q8/README.md`. The candidate config and receipt are
`lfm2-sm86-q8-v32-candidate.txt` and its adjacent `.receipt.json`.

## LFM2 CUDA SM86 Q8 vector Decode v31 (2026-09-02)

LFM2's long-Decode bottleneck was the scalar D64 full-attention path, accounting for
approximately 84% of one 5963-token Decode step. CUDA space v31 now exposes a default-off,
bit-affecting `attn_d64_q8_vec` knob scoped to D64/GQA4 models. The SM86 implementation
reads Q8 K/V directly, uses Q8_0 x Q8_1 DP4A for QK, retains FP32 V accumulation, and
combines partitions deterministically. The tuner selected value 1 for this exact model,
device, driver, and Q8 cache fingerprint.

The route passed the complete `cuda-llama-fa-q8_0` v1 suite and ordinary receipt-backed
loading. Decode CUDA Graph capture now includes the route: 40/40 dynamic nodes were
classified, the graph was retained, and the following token replayed it. The Graph scan
also now treats a completely empty dormant Program Pack catalog as no Driver nodes rather
than a fatal catalog state; populated-but-unfrozen catalogs remain rejected.

Interleaved Q8-KV results against the `e0949576a` performance fork, with llama FA enabled,
context checkpoints 32, ubatch 512, and 32 generated tokens:

- 455 tokens: Prefill 2693.2 vs 2968.2 tok/s (0.907x); Decode 80.6 vs 77.6 (1.039x).
- 5963 tokens: Prefill 2528.9 vs 3859.2 tok/s (0.655x); Decode 73.4 vs 74.9 (0.980x).

The tested Decode regimes meet the current open-source parity objective. Prefill remains
open, especially the long-prompt route; it is the next optimization phase. Evidence is in
`docs/evidence/cuda-sm86-lfm2-q8/lfm2-sm86-q8-v31.txt`, its adjacent receipt, and the
directory README.

## LFM2 CUDA SM86 Q8 correctness (2026-09-02)

The first LFM2 CUDA phase is closed on an RTX 3060 Laptop GPU (SM86) without changing
the existing E4B Q4 contract. CUDA tuning space v30 now excludes D256/D512 attention
and Q4 MMVQ controls when the LFM2/Q8 model cannot dispatch them. Attention-decode
candidates are screened through an attention-decode probe instead of a generic matmul.

A separate `cuda-llama-fa-q8_0` correctness suite is sealed against the pinned
`zeraix/llama-cpp` revision `4695f001fece1660d8bb1b3748f50726ddcc100b` with FA on,
context-copy off, and Q8 K/V. All six gates pass: logits at 128/449/512 tokens and
eight-step decode at 128/512/2000 tokens. The sealed config was then loaded by a real
LFM2 CUDA forward pass; the runtime reported the selected host config and produced
128,000 finite logits.

Evidence:

- `docs/evidence/cuda-sm86-lfm2-q8/lfm2-sm86-q8-v30.txt`
- `docs/evidence/cuda-sm86-lfm2-q8/lfm2-sm86-q8-v30.txt.receipt.json`
- `docs/evidence/cuda-sm86-lfm2-q8/README.md`

Performance tuning remains a separate next phase; this record makes no llama throughput claim.


## State (2026-08-20)

Engine: gemma4 E4B on M3 Pro, measured against the llama.cpp fork with interleaved cold
prompts (`bracket.py`), byte-pinned logits, and determinism gates.

```
correctness   pins EXACT -- re-pinned 2026-08-25 at 521/522/5642 only, the lengths
              whose final chunk is under 64 tokens, for the half-staging floor below;
              determinism 1/N distinct at every gate length, the 517..527 band and
              decode, q4/q8 KV included; fork agreement at n=521 (9-token tail chunk)
              f16 0.006 (tol 2e-2), q4 0.649 (tol 1.0, 8/10 ids), q8 0.035 (tol 0.3,
              9/10), and f16 n=2000 0.009
f16   LONG    prefill tie (566/564 vs 562/565)   decode tie
      SHORT   prefill +9.7%                      decode tie      resting -45 MiB
q4/q4 LONG    prefill +3.9%   decode +10.3%      resting lighter    <- app configuration
      SHORT   prefill +10.7%  decode +6.5%       resting lighter
q8/q8 LONG    prefill +4.9%   decode +8.4%       resting heavier (fork's disk-tier edge)
      SHORT   prefill +10.2%  decode +5.6%
```

The "ambient n=128 wobble" that ran through every earlier verification is root-caused and
fixed (arena aliasing, below); the reproducer is 1/60-distinct after the fix and the full
det gate is green. The v2 refactor changed no engine behavior — verified byte-identical.

Startup 6.6 s -> 1.6 s; decode went 2113 ms/token -> ~32 over the campaign. Windows
dependency graph of the portable crates is verified Metal-free (cargo tree per-target);
CUDA backend is written but has never run on a CUDA host.

## What a co-batched decode is worth (2026-08-25)

Priced before building it (`dev_harness/batch_cost.py`, `imparo-forward --dbatch B`):
what does a B-row forward cost against B one-row forwards, at the same context?

```
ctx 2000        LFM2.5-2.6B Q8_0            gemma-4-E4B Q4_K_XL
rows  ms/fwd    per row   ceiling           ms/fwd   per row   ceiling
   1   23.4      23.4        --              27.1     27.1       --
   2   50.0      25.0      -6.7%             65.9     33.0     -21.4%
   4   50.7      12.7      45.9%             67.8     16.9      37.6%
   8   90.1      11.3      51.9%             70.1      8.8      67.7%
  16   61.0       3.8      83.7%             84.4      5.3      80.6%
  32   70.1       2.2      90.6%            122.6      3.8      85.9%
  64  110.8       1.7      92.6%            173.7      2.7      90.0%
```

Two conversations in one forward is a LOSS: a 2-row batch leaves the decode GEMV for
the token-tile GEMM and that costs more than two GEMVs. Four is where it starts paying.

A ceiling, not a promise: these rows sit at consecutive positions in ONE conversation,
so they share KV reads separate conversations would not. The weight-GEMM half is priced
correctly; attention is optimistic.

LFM2's row 8 is not noise. Three repeats, spread under 0.4 ms: 8 rows takes 90.5 ms
where 12 takes 60.8 -- strictly less work, more time, which is a code path and not a
machine. `use_gemm = n_tok > g_q8_gemv_max_tok` with the default 8 keeps a 8-row batch
on the token tile. The knob is declared `Sw::Crossing`, so the tuner owns the fix once
its crossing sweep is ported (task #5).

## Unified KV pool (2026-08-22)

docs/unified-kv-pool.md is implemented on Metal end to end; the "Shipped state"
appendix there records the design-level decisions and cuts. The verified numbers:

```
sharing      three agents, ~700-token shared preamble: reused 0/512/512, the
             preamble's blocks exist once, outputs equal a no-pool server
             (kv_gates.py share)
reprocess    tail-only on every path: cold 0, append 512, branch 512,
             switch-back 512, restart 512 of ~600-token prompts (kv_gates.py accept)
byte-safety  scattered placement, split resume grids, spill/restore under
             f16/q4_0/q8_0 all BYTE-EQUAL (kv_gates.py engine); det pins EXACT
fork         a second agent adopts a first agent's finished turn from disk, in a
             fresh process (kv_gates.py fork)
seed         a preamble prefilled with no user turn: E4B 1408 of 1420 reused,
             LFM2 1280 of 1322, cold (kv_gates.py seed)
continuation two-turn: turn 2 prefills only the tail (reused=640 of 721)
durability   restart restores from disk through the tables; erase sweeps units
             and survives a cold re-ask byte-identically (kv_e2e)
keyless      no conversation id: one conversation leaves ONE manifest across
             restarts, and a declared fork leaves the trunk's resume untouched
             (kv_gates.py keyless)
```

Checkpoints are chained deltas (min(gap, window) rows), RECORDED at branch
points and turn ends but captured only at switch-out/release/shutdown — a
superseded record dies uncaptured, so hot single-conversation traffic moves
zero bytes per turn.

Disk holds the SAME chain (2026-08-25). A manifest names every boundary it still
keeps, each as the link `[from, boundary)` its turn added, and a restore loads
the target link plus the ancestors that carry it back a whole window. So rewind
and branch survive a restart, and a turn's checkpoint costs a turn's slice:

```
gemma4 E4B    anchor [0, 1280)   20.0 MiB    link [1280, 1344)   3.5 MiB
LFM2.5-2.6B   anchor [0, 1216)    3.3 MiB    link [1216, 1280)   0.3 MiB
```

Blobs are content addressed, so re-committing a boundary writes nothing, and a
second agent can stand on a first agent's finished turn in a fresh process
(kv_gates.py fork: 1472 of 1498 reused on E4B against a 1408 preamble-only
baseline; 1408 of 1462 on LFM2).

Branching from an old turn is ONE operation with two meanings, and the client
picks: REWIND replaces this conversation (the default), FORK starts a new one.
With a conversation id the id decides; without one, `x-new-conversation: true`
does. A thinking model's own next prompt diverges inside its last turn by
construction — its generation prompt ends with an opener the finished-turn
render omits — and that divergence is a rewind, not a stranger, so identity
follows the longest agreement rather than an exact prefix. A rewind REPLACES:
boundaries it abandoned are dropped, which also stops dead boundaries crowding
out live rewind targets under a cap that keeps by position.

```
LFM2, one keyless conversation over three processes    before   after
manifests left on disk                                    3        1
labels minted                                       per turn   per process
```

Speed and memory vs the fork after the identity-placement kernels and the
resident-bytes budget (interleaved brackets, 2026-08-22):

```
q4 (app)  prefill +2.4-3.7%   decode 37.2-37.4 vs 34.2-34.4 (+8.5%)
f16       prefill +1.2%/tie   decode tie (37.7-37.8 vs 37.8-38.0)
resting   q4 126-137 MiB (fork 111-116)   f16 206 (fork ~150; pre-kv 177)
          the remainder is the retained conversation serving continuation;
          budget = IMPARO_KV_RESIDENT_MB (64)
wired     ~0 at rest against the fork's held +4 GB (it pins weights+state;
          we re-wire from page cache on the next request, unmeasurable in
          run-0-vs-run-1 prefill)
A/B/B/A   current vs pre-kv c8b38d2: prefill within 0.4%, decode within
          0.3 tok/s
```

## Long context (2026-08-22)

The decode span limit moved from ~73k to 2M positions (scores-aware slice
slicing; 96k validated in production shape on E4B, whose GGUF declares
131072 trained context -- the server clamps -c to the model file's value).
n=16384 is a pinned det_gate length. Where we stand against the fork, all
measured interleaved with the same prompts (tok/s, imparo / fork):

```
prompt   kv     prefill tok/s              decode tok/s             resting MiB   wired at rest MiB
                imparo   fork              imparo   fork            imparo fork   imparo    fork
  449   f16     592.0   546.0   +8.4%       40.7   39.9    +2.0%      391   136     -978   +4123
  449   q8_0    580.8   534.7   +8.6%       40.0   37.7    +6.1%      293   116     -816   +4174
  449   q4_0    585.5   542.3   +8.0%       39.5   37.1    +6.5%      213   109    -2716   +3947
 5651   f16     570.7   560.9   +1.7%       38.0   37.9    +0.3%      360   146       +2   +4316
 5651   q8_0    560.6   528.8   +6.0%       37.7   34.5    +9.3%      285   116      +14   +4215
 5651   q4_0    561.1   542.2   +3.5%       37.5   34.3    +9.3%      212   119      +61   +3515
16310   f16     530.9   513.8   +3.3%       35.1   34.5    +1.7%      359   150       -7   +4320
16310   q8_0    517.5   459.6  +12.6%       35.8   29.6   +20.9%      278   129      +10   +4115
16310   q4_0    517.5   471.9   +9.7%       35.9   29.0   +23.8%      208   104     +610   +4093
```

**THE FORK COLUMN ABOVE IS NOT COMPARABLE TO ANYTHING MEASURED AFTER 2026-08-27.**
Re-measured 2026-08-31: the harness did not pass `-fa on -ctxcp 0` to the reference until
2026-08-27 (commit 7f15ce7), four days after this table was recorded, and those flags are
worth ~20% to llama at the 449 cell:

```
E4B 446-token prefill, tok/s, measured 2026-08-31
  fork    default flags      495.7  498.4      <- what this table's column resembles
  fork    -fa on -ctxcp 0    594.5  599.6      <- what the harness measures today
  upstream 9723942ad         607.3  607.3
```

The 546.0 recorded here reproduces under NEITHER setting, so it is not merely a flag
difference and no explanation is offered for it. Our own number is stable across the same
span (592.0 here, 598.8 on 2026-08-31), so whatever moved, it was not this engine. Treat
the fork column as a record of one day's configuration, not as a baseline.

UPSTREAM DID NOT CHANGE. Interleaved on 2026-08-31, upstream at this table's fork base
(4695f001f) against upstream master (9723942ad, +254 commits) measured 604.6/607.8 against
607.3/607.3 -- 0% across a fortnight of commits. There is no upstream improvement behind
the closed gap.

Every cell is a win on both speed axes. Read the shape, not just the signs:

PREFILL is strongest at the ends and weakest in the middle. Short prompts win ~8% on
dispatch efficiency; 16k wins 3-13% on the attention work that dominates there. The 5651
f16 cell at +1.7% is the narrowest, and it is the honest one to quote when a single
number is wanted for a mid-length f16 turn.

DECODE splits by cache type. f16 is a tie to +2%; quantized is +6% short and +21-24% at
16k, because the fork does not handle a quantized cache efficiently at depth while our
dequant path does.

MEMORY goes both ways and both belong in the same sentence. Our RESTING footprint is
heavier in every row -- 208-391 MiB against 104-150 -- which is the standing gap and is
not going to be argued away. WIRED memory at rest is the opposite and much larger: the
fork leaves +3.5 to +4.3 GB wired after every turn, we leave between -2716 and +610 MiB.
On a machine that has to stay usable while a model is resident, that is the number a user
feels; on a machine that just wants the process small, theirs is.

Measured with bracket.py -- interleaved legs, cold unique prompts, three runs per leg --
after the tuner's own config (attn_threads 256, attn_stream_min_pos 6144) and with
det_gate pins EXACT.

f16 prefill moved from +1.7% to +3.2% by dropping every transposing K load from
the prefill attention score MMA (ac64e9a): computing St[p][q] = K[p][d] . Qt[d][q]
instead of S[q][p] = Q[q][d] . Kt[d][p] lets K load the way it is STORED and pays
one transposing threadgroup store per score fragment instead of 1920 transposing
device loads per position tile. Bit-identical -- det_gate pins EXACT at every
length, and q4/q8 agreement unchanged to five decimals.

The gain is smaller at shorter prompts because attention scales with N^2: it is
15.6% of a 16k prefill and about 5.8% at 5.6k, so the same 5.3% dispatch win is
worth 0.8% and 0.3% respectively.

MEMORY, from the same brackets: our resting footprint is heavier (358 MiB against
137-146), which is the standing gap. But WIRED memory at rest is the other way
round and by a wide margin -- the fork leaves +4317 and +4334 MiB wired after the
turn, we leave -249 and +12. On a machine that has to stay usable that is the
number a user feels.

Every cell is now a win or a tie. f16 prefill was the last one behind and
turned on K sharing (below); q4 was re-measured after it and is unaffected
except by the shared unroll change, which helped it slightly. q8 has not been
re-bracketed since; it uses the same kernels as q4.

16k decode began the campaign at -15%. Three kernel generations closed it:
the reference flash-decode shape, then per-head-size templates (NE=1, 32-row
blocks, register accumulators), then GQA ROW SHARING -- two query heads per
threadgroup, since at E4B's 8-over-2 geometry a per-head threadgroup pulls
every K and V row through the cache four times. That last step is 15% per
dispatch and bit-identical (same arithmetic, different owner).

Quantized KV is where the lead is, and it is not a trick of ours: quantizing
costs us nothing (f16 35.1 -> q8 35.8 -> q4 35.5) and costs the fork 14-17%
(34.7 -> 29.9 -> 28.9). Our own q4 was the slowest of the three until the
dequant path was fixed -- the block scale had been rebuilt from two byte loads
on every float4 of a 32-value block, and the nibble select was a
simdgroup-divergent ternary. Both cost ALU, not bytes; q4 gained 8% and became
the fastest, which is what its byte count always said it should be.

DECODE ROUTING. Below ~3k spans the score-tile kernel wins because flash's
per-block bookkeeping (accumulator rescale, barrier, cross-simdgroup merge)
does not amortize; above it the streaming kernel wins and keeps winning. The
boundary is micro-tuned as a span crossing (see imparo-tune) with 3072 as the
compiled fallback. It was 8192 for most of the campaign because it had been
derived END TO END, where it cannot be seen: only 7 of 42 layers are
full-attention, so at 4k the choice moves a decode step under 1% against 7%
run-to-run spread.

PREFILL IS NOW AHEAD TOO: two independent 16k f16 brackets read 531.6 vs
521.3 (+2.0%) and 528.6 vs 521.2 (+1.4%), from -1.7% when the campaign
started; the per-run spreads do not overlap (ours 526.1-528.6, theirs
519.5-521.2). Decode is a tie at 34.9 -- the +1.7% the first bracket
showed did not reproduce. The fix was sharing K across query row
groups -- the score phase's work unit was (row group, position group), so
two simdgroups each re-loaded the same K, which is why widening the query
tile never helped in any earlier attempt. See docs/metal_kernel.md for the
configuration and why every term of it is forced. What follows is the
analysis that located it. Fitting prefill time as base*N + attn*N^2/2
separates it cleanly and needs no cross-harness comparison at all: our BASE is
1.1% FASTER than the fork's (1664 vs 1682 us/token) and our attention
coefficient is ~21-25% WORSE (36.4 vs 29.2 ns/token^2). Attention is 15% of a
16k prefill, so a tie needs ~11% off that kernel and a win ~15%.

The ceilings say where it can come from: matrix ops reach 16.76 TFLOPS with
operands in registers, 9.71 from threadgroup memory, 6.03 from device memory
at the KV row stride. The score phase reads K from device and sustains ~2.9 --
half of its own ceiling, because it issues four multiplies per five loads
where the ceiling probe issues eight per six. Raising that ratio is what the
register budget blocks, and the way through is coupled (see #37).

MEASURED AND FALSIFIED, so nobody re-tries them: half-Q score operands; a
transposed-K device mirror; QT-16 x half-Q at hd 512 (5 s slower, the row
groups serialize); stream-kernel C=64 and nsg=8; a blocked lane mapping for
quantized rows (worse for every KV type); and the "in-engine drag", which was
an ablation artifact -- per-CB hardware timestamps show the same 39.7-48.7 ms
per deep dispatch inside the live chunk loop as in isolation.

METHOD TRAPS THAT EACH COST AN IMPLEMENTATION:
  - probe binaries call imparo_metal::init, which does NOT parse the IMPARO_*
    knob env vars (init_tuned does), so an env-driven sweep there measures one
    configuration repeatedly;
  - an isolated kernel bench must ROTATE LAYERS, or reps are SLC-served and
    read above this machine's DRAM peak;
  - never run a probe while a bracket holds the GPU -- it corrupts both;
  - end-to-end decode cannot resolve a per-dispatch question (see routing).

The Metal kernels themselves -- what each is for, which one the host picks, and
the rules for changing them -- are in docs/metal_kernel.md.

## Matmul routing

One comparison chain in `matmat`; both boundaries are tuner-owned (hostconfig v11):

```
n_tok == 1                  -> GEMV                  (decode fast path; lanes/nr0 tuned)
n_tok 2  .. gemv_max_tok    -> GEMV, multi-token     (measured 1 on M3 Pro: band empty)
n_tok    .. nb8_max         -> nb8 narrow GEMM 64x8  (measured 47 on M3 Pro)
n_tok  > nb8_max            -> wide GEMM, 64-token tiles
                               + tail: floor-to-64 main pass, remainder 1..nb8_max
                                 through nb8 in the same encode (same k-order as the
                                 wide tile, so the split is bit-identical)
```

Which TILE runs is width-keyed, and bit-identical. Which PRECISION runs is not width-keyed
and is not tuner-owned: `HALF_A_MIN` is 2, so every prefill width stages its activations as
half. See "A cached prefix only helps if the resumed pass takes the SAME path" below for
what a width-keyed precision cost.

`n_tok` is the whole batch handed to `matmat`: with ubatch 512, long prompts always take
the wide+tail path, so the GEMV/nb8 bands only see a prompt's short final chunk or a
future MTP verify batch. The tail remainder never routes to GEMV whatever the knob says —
it must stay bit-identical with the wide pass, and GEMV's k-order differs.

## Tuner gates (2026-08-28)

The registry is a protocol, and one test checks it holds. Every tuned value reaches the
engine as `(decl.apply)(v)` and is read back as `(decl.current)()`; nothing else enforces
that the two agree, and a tuned `rt_shape` was once dropped exactly there.

```
cargo test --release                       3 pure checks, no GPU, in every run
  probe covers every registry coordinate     a new knob with no probe entry FAILS
  a settable vector round trips              apply(v) then current() == v, pre-init
  a report-only coordinate is not moved      pt_512x / pt_256x, whose apply is |_| {}

cargo test -p imparo-metal --test preinit_registry -- --ignored     LOCAL GPU gate
  an accepted tuple survives successful init exactly
```

The gated one is the property the `rt_shape` defect broke, and only a real init can see
it: apply a tuple, run `imparo_metal::init`, assert every knob still reads what was
accepted. Run it beside det_gate / kv_gates; it is not in CI because it compiles real
pipelines.

## Speed is gated too (2026-08-29)

det_gate proves the numbers did not move and kv_gates proves the reuse did not shrink.
Neither notices a kernel change that is correct and 20% slower, which is most of what the
tuner decides.

```
python3 dev_harness/speed_gate.py --model M.gguf --kv f16 --record    write the baseline
python3 dev_harness/speed_gate.py --model M.gguf --kv f16             compare against it
   rc 0  within tolerance        rc 1  measured and worse        rc 2  no measurement
```

Baselines live in `docs/evidence/speed/`, keyed by model, cache type and rep count; a key
mismatch REFUSES rather than comparing against whatever is on disk. Median of three with
the warm-up dropped. Recorded 2026-08-29 on the M3 Pro, every sample verified cold:

```
                       prefill_ms   decode_tok_s
LFM2.5-2.6B Q8_0  f16      3524        42.59
                  q8_0     3529        42.14
                  q4_0     3528        42.54
gemma-4-E4B q4    f16      4956        39.04
                  q8_0     5042        39.30
                  q4_0     5047        39.31
```

Tolerance 2%, set from two measurements rather than chosen: same-binary drift is <=1.1%
(E4B f16 prefill read 4902 then 4956 hours apart), and the smallest regression worth
catching is +2.4% (PT 440 vs 128, 3610 against a 3524 baseline). Verified both ways --
the unchanged binary passes at +0.1%, `IMPARO_QCOMB_PT=440` fails at +2.4%. The limit is
stated rather than hidden: a regression under ~2% will not fail here, because the
tolerance has to clear the drift between baselines. Measure a suspected sub-2% change
against a same-session baseline instead.

Every sample must be a COLD prefill, and the harness verifies that from the server's own
`reused=` count rather than assuming it. It did assume it, and was wrong: the four prompts
differ from each other but repeat on every run, and the pool is on disk keyed by port, so
only a port's first run ever prefilled.

```
first run of a port    reused=0      prefill_ms=3503
every run after        reused=2880   prefill_ms=62     "-98.2%   SPEED PASS"
```

The pool directory is now removed before each run AND any sample that restored anything
exits 2. `IMPARO_ATTN_WHICH` is the matching tool one level down: it prints the `hd/blk/qt/pt`
a dispatch actually used, which is the only thing that separates "this knob is inert" from
"this knob never reached the kernel".

## Pins cover the cache types, and a missing pin says so (2026-08-29)

`refs/pins.json` is keyed by model, backend, CACHE TYPE and length. The cache type is part
of the key because a quantized cache takes different kernels -- the QT-8 prefill attention
path exists only for it -- so an f16 pin says nothing about whether they are correct.
Quantized modes used to run determinism only, which means a q4_0 kernel could have returned
stable wrong numbers and the gate would have printed a pass.

A missing pin now prints `pin ABSENT` instead of nothing at all. It printed nothing, so
`determinism 1/6 distinct` for a model with no pins read exactly like a model whose numbers
had been checked -- and LFM2 on metal had been running that way, stable and unverified.

```
                     before                after
E4B   metal      f16:8                 f16:8  q4_0:8  q8_0:8
LFM2  metal      (none at all)         f16:8  q4_0:8  q8_0:8
```

All six combinations verified EXACT after recording. What these pins do is FREEZE today's
numbers against future drift; they are not evidence that the numbers are right. Correctness
against another engine stays where it was -- logit_agree.py and decode_agree.py -- and
cross-backend equality is still checked as drift, never as a byte pin, because float
reduction order differs across backends.

## The device profile is measured once per machine (2026-08-29)

```
~/.imparo/device-<host>-dev-<backend>.txt    MEASURED: spill cliff, cache knee, DRAM
                                             rate, threadgroup budget, thread limit
~/.imparo/tune-<host>-...-space-vN-kv-X.txt  TUNED: knob values
```

Two files because they answer different questions. Kernel shapes are DERIVED from the
measured numbers at library-compile time, and the measurements used to live in the tune
file — which is keyed by the knob space, the cache type and the model as well as the host.
None of those changes what a GPU's register file does, so adding a knob (space v12 -> v13)
discarded a cliff that was still true and every derived shape silently reverted to its
compiled literal: the engine ran a "0-accumulator cliff" while 24 sat on disk.

The device profile is applied unconditionally, including while tuning — a measurement is
not a tuning, and a tuner that skips it sweeps knobs against a library the engine will not
compile. `imparo-tune MODEL.gguf --discover-only` measures a machine in seconds; a tuner
run on a machine with no profile measures it, writes it, and stops, because its own kernel
library was compiled before those numbers existed.

**A boundary can only be measured on the kernel it routes.** `NarrowMix` read the model's
own weight kind; `PrefillGemm` and the crossing scan hardcoded Q4_0, so every Q8 boundary
was ranked against a kernel it could not move. On LFM2.5-2.6B Q8_0:

```
before   n_tok=2 0.96x  4 1.00x  8 1.00x  16 1.00x  32 1.00x   scans [32,32,1] -> guard held 8
after    n_tok=2 0.96x  4 0.96x  8 0.46x  16 0.25x  32 0.14x   scans [7,7,7]   -> 7
```

1.00x at every rung is the signature: forcing the knob changed nothing because the kernel
it routes was never dispatched. The three-scan agreement guard behaved correctly the whole
time -- it refused `[32,32,1]` and kept the compiled default rather than storing a rung
picked out of noise.

## Lessons worth the price paid

**A knob that cannot move the kernel it names measures noise and reports a winner.** Six
in one session, each announced by a number that was too CLEAN rather than by a failure:

```
q8_gemv_max_tok   swept against a Q4 kernel        1.00x at every rung
qcomb_blk         no dispatch read it              dead: global, setter, getter, no reader
attn_live_mask    one pipeline set compiled        unmeasurable after init
attn_qcomb_pt     both A/B arms took QT-16 rows    pin EXACT, ALL PASS
IMPARO_QCOMB_PT   env set a static local, not the
                  global the other family reads    three identical runs
attn_qcomb_mask   comment described a 2^slots
                  derivation nobody wrote          empty ladder, never swept
```

The tuner ranked candidates in every case and every correctness gate passed, because the
build was correct — it was the same build under each label. Only the dispatch's own report
separates "inert" from "never reached": `IMPARO_ATTN_WHICH` prints the `hd/blk/qt/pt` a
kernel actually ran with. Read that before recording any result that looks too tidy.

The same defect at the library level is worse and quieter: `imparo-tune` never called
`set_attention_head_dims`, so its Metal library compiled ZERO qcomb slots. Its prefill
attention workload dispatched qtile while the engine dispatched qcomb, and both qcomb
knobs read "N/A to this model" — a whole kernel family the tuner could not see, let alone
rank.

**Never wrap the same bytes in two MTLBuffer objects.** Wrapping overlapping ranges of one allocation in
distinct MTLBuffer objects breaks Metal's hazard tracking, which is per-RESOURCE: the FFN
gate's write raced attention's read of the same bytes, ~25%/rep wobble at warm clocks,
first forward always clean (cold clocks leave slack). Serial-encoder memoryBarrier does
NOT fix it — that is concurrent-encoder machinery. Fix: one buffer spans the arena, placed
buffers are offsets. The hunt method that cracked it: a cheap in-process reproducer
(`imparo-forward --repeat`), then discriminators flipped one at a time
(fork warm-process 20/20 stable -> our bug; alternating prompts -> not stale reads;
NO_ARENA_OVERLAP=1 stable -> aliasing). Full story: `dev_harness/metal_visibility_repro/`.

**An eviction that forgets the index leaves ghosts that serve empty state.** Evicted
pool units kept their content-hash entries; a later re-seal merged onto the
placement-less ghost and served an empty block table — a crash two requests later,
nowhere near the bug. Eviction must remove every index entry, and merge-on-seal must
adopt a fresh placement when the survivor lost its backing. The E2E that caught it:
erase, then immediately re-ask cold.

**A whole-range guard on a paged layout is a race you have not met yet.** The device-spill
tail check asked "does this range wrap?" — under scattered block tables every tile
answered yes, all tiles took a fallback path whose indexing is collision-free only when
tails are rare, and five threadgroups raced. The guard must ask the RING question
(wrapping is a ring property), not the range question. Probe-checksum diffing between a
forced-slow run and the fast path localized it; an identity-mapped control (same
machinery, same bytes) proved the machinery clean.

**Read `red[0]` into a register, THEN let the next phase reuse the slot — with a barrier
between.** The decode nondeterminism (task #26, RESOLVED): both attention kernels read
`mx = red[0]` and the sum phase wrote `red[sgid] = local_sum` with no barrier between, so
a fast simdgroup 0 overwrote the max while a slow simdgroup was still reading it —
`exp(score - sum)` instead of `exp(score - max)`, decided by warp scheduling, firing at
warm clocks (more at 512 threads than 256). The GQA variant had already dodged the same
hazard with a separate `red[nsg]` slot: a hazard dodged once and not fixed in the
siblings is a bug filed for later. One `threadgroup_barrier` fixed it; decode is now
byte-stable across reps (12x at n=255, 8x48 tokens at n=521, server reps under pool and
legacy). Regression row: det_gate.py's decode determinism check; engine-level
`imparo-forward --repeat N --decode K` with per-step logits hashes cracked it.

**"Serialization kills it" does not prove an ordering race.** Per-dispatch probe syncs
also IDLE the GPU and drop clocks, which masks warm-clock intra-kernel races just as
well. The discriminator: a flush+wait barrier per layer preserves full ordering without
the idling — it did NOT fix this bug, which killed the race-between-dispatches theory
and pointed back inside the kernel.

**A misaligned vector load on Metal is silent corruption, not a fault.** Spec 2.5 lets the
compiler assume alignment: `uchar4` claims 4, a Q4_0 payload starts at `18k + 2`. Use
`packed_uchar4`. Three shipping kernels had this; llama.cpp 30de65202 documents the hazard.

**Self-diff proves nothing; the fork is the oracle.** Two correctness bugs were perfectly
consistent run-to-run. Softmax cancels in a difference, so `logprob[i] - logprob[top1]`
from the fork's `n_probs` compares against raw logit differences — that identity is what
makes cross-engine byte-level checking possible at all.

**Attribute wall time, not GPU time.** A 1.06 ms/token host-side scalar argmax (~4% of
decode) hid for a whole session because every attribution scheme partitioned GPU time
only. Every millisecond of wall must land in a bucket, host included; "GPU busy = N%" is
a claim about the other (100-N)% too.

**A cached prefix only helps if the resumed pass takes the SAME path.** Reuse changes the
WIDTH of the chunk that computes the tail — 40 tokens where the cold pass used 232 — so any
decision keyed on width follows the reuse into the answer. Which TILE runs is fine (nb8
keeps the wide tile's k-order, bit-identical). Which PRECISION runs is not: the wo
projection staged half at `n_tok >= 64`, and that alone made a q4_0 conversation answer
differently from a cold one whenever it branched at 704. A path may depend on what a
position IS; never on how wide the batch that carried it was.

**Prefill tok/s is 64-token-tile-sensitive; compare engines only at equal ptok.** A
449-token prompt pads to 8 tiles where 447 fits 7 — 14% more work, zero code difference.
That same padding was a real perf bug at chunk tails: fixed by splitting the last tile
through the narrow-N (nb8) kernel in the same encode, bit-identical because both paths
share k-order and mirror bytes (449 step 108 -> 67 ms). The fork has the same defect on a
32 grid and no tail handling.

**The Q4 GEMM was memory-bound, never compute-bound**: it read 18 useful bytes of every
128-byte cache line. Layout and staging beat any amount of tile tuning — the fork's GEMM
shape, ported exactly, was NOT the advantage.

**Half-staged activations must be produced half, not converted.** A separate f32->f16 cvt
pass cost 252 ms at long prefill and hid inside the matmat category; producers now write
the half mirror directly. Intermediary layers hide cost — attribute per stage before
believing a category number.

**Quantized KV needs the Hadamard rotation.** The fork tolerates q4 KV because attn_rot
spreads massive-activation channels before quantization (QuaRot family, static Sylvester
form); without it q4 drift is a quality bug, not noise.

**Freeing is not returning.** The arena was never given back to the OS; resting footprint
is governed by purgeability/decommit, not by free() calls. Measure phys_footprint at rest,
not just peak.

**A stale binary reads exactly like a result.** Bare `cargo build` built only the CLI
(workspace default-members); same-second mtimes defeat freshness checks; `ls` sorts
alphabetically, not by argument order. `dev_harness/build.py` now asserts freshness and
checks the selected binaries explicitly. A stale selection is repaired by cleaning and
rebuilding only its owning package/target; touching source files or forcing a global
rebuild is not a freshness mechanism. Every measurement goes through the selected-bin
gate.

**Determinism is proven at the spec level, not by sampling.** A 10-run-clean "fix" (flush
cadence) collapsed at 60 runs. If the argument for a fix is a sample statistic instead of
a mechanism, it is not a fix and will not transfer to another machine.

**Tuner lessons.** A benchmark that freezes the machine never gets run — screen at small
sizes and duty-cycle sweeps. A search that only compares candidates never asks whether it
beat the compiled default (the defaults were the 6.7x; the tuner declines to store when
they win). Every knob declares its category — shape-derived and arithmetic knobs are
computed, device knobs come from the once-per-machine profile, only genuinely uncertain
knobs are benched. Search spaces are versioned per backend; stored configs are rejected on
mismatch.

**Attention shape fact:** P·V costs 1.9x the score phase for the same MACs (K is read
coalesced, V is not) — the tail guard on P·V, not more score tuning, was the win.

## CUDA E4B/SM86 exact-128 admission (2026-08-31)

The v26 exact-128 route is now correctness-admissible and production-loadable on
the RTX 3060 / SM86 evidence host. The six-gate receipt covers llama/FA q4_0
`logit_agree` at 128/449/512 and eight-step recurrent `decode_agree` at
128/512/2000. The normal loader accepts the receipt without gate mode and the
128-token forward produces 262,144/262,144 finite logits. Current Rust/Python
tests pass 29/29 CUDA library, 15/15 exact-route integration, 9/9 host receipt and
10/10 sealer. Kernel-level and complete-model memcheck, initcheck, racecheck and
synccheck all report zero errors/hazards against the final binary.
A post-admission failure witness also forced release of 1,873,428,480 bytes of
optional packed-Q4 cache, rebuilt all 42 layers, recommitted 42/42 transactions
and produced a byte-identical final logit dump. Its temporary dev-tool hook was
removed afterwards; the sanitized production binary hash did not change.

The current candidate subsequently received a compile-time-only exact-128
attention specialization. The receipted route no longer carries the dormant
half-value and alternate-reduction kernel footprint: static shared memory fell
from 10,928 to 2,192 bytes, the 64-byte/thread stack and all LDL/STL sites were
eliminated, and static SASS fell from 2,447 to 1,390 instructions. The generic
batch32 specialization remains available for laboratory fallback.

The current SM86-only release binary SHA-256 is
`6dabc3d516e4adce1d82d129b3fb62130683cbb061a4780a878052302f588c82`.
The updated six-gate receipt, normal production loader, 15/15 source contract,
29/29 CUDA library tests and complete-model memcheck/initcheck/racecheck/
synccheck all pass. This supersedes the E-32 binary/receipt identity while
preserving its numerical route, failure witness and rollback contract; exact
commands and hashes are recorded in E-20260831-33.

This closes correctness, receipt loading and memory/concurrency admission, not
the open-source performance claim. Existing evidence places the older 128-token
route at approximately 0.986--1.013x llama.cpp, while 449 and Decode were ahead;
the final D256 batch32 arithmetic repair has not been rebenchmarked because new
performance runs were explicitly deferred until requested. The short-Prefill
release decision therefore remains pending one final interleaved performance and
shape-regression bracket. Exact evidence and rollback are recorded in
`docs/cuda-sm86-experiment-ledger.md` E-20260831-32/33.

The next exact-128 D256 candidate now uses all four warps to compute four
independent attention scores at once, while retaining each score's K16 WMMA
accumulation order. It is isolated to the existing SM86/128/slot-39 numerical
route. The release kernel reports 56 registers, zero stack/local and 8,336 bytes
static shared memory; static SASS sites fell from 1,390 to 782. The increased
shared footprint remains shape-bounded and does not affect other routes.

The candidate passed the 29-test CUDA library suite, 16 source contracts, 47
harness tests, authenticated eight-step 128-token Decode agreement, the full
six-gate llama/FA q4_0 receipt suite and complete-model memcheck/initcheck/
racecheck/synccheck. Production loading without gate mode accepts receipt
SHA-256 `939bb31c2aa8543d2106219cb94dcd61ee23b4b739621d969a6208b983a4fa5f`.
No performance benchmark was run, so the short-Prefill open-source performance
claim is still pending rather than inferred from generated-code improvements.

Correctness-gate harnesses now also reject the non-runtime
`IMPARO_TUNE_CONFIG` alias, missing `IMPARO_HOST_CONFIG`, manifest drift and an
unauthenticated llama DLL inventory. This closes a concrete false-evidence path
found while validating the candidate; ordinary diagnostic mode remains flexible.

The current exact-128 follow-up caches the immutable scaled Q WMMA tiles once
per CTA and reuses one physical-row translation table in both score and value
phases. Two generated-code regressions were rejected locally: a compact
scaled-Q-vector cache and a forced `unroll 1` variant both expanded to 16 HMMA
and 13 barrier sites. The retained runtime-bound Q-tile candidate is SM86-only,
uses 48 registers, 14,608 bytes static shared, zero stack/local, 1,184 SASS
sites and six CTA barrier sites. The fresh 21,852,160-byte release executable
hashes to `0ddbcbd181b705579eeac5ea7266adeaccd5b339e37756d53bce21eb300c825d`.

The final candidate passes 29/29 CUDA library tests, 16/16 exact-route source
contracts, 47/47 harness tests, all six authenticated llama/FA q4_0 numerical
gates, ordinary receipt-backed production loading and all four complete-model
Compute Sanitizer tools. Its receipt SHA-256 is
`18d8a7aeecb2986f5fbfc84ef32cfd2d1de733a349b838166b76a54f474f7691`
and the backend fingerprint is
`87a00d167ae069a67f1fa02cf7c9096e9d06167c4ecf394109774019f5c65b0a`.
No throughput benchmark was run, so 128-token Prefill parity remains pending;
the result is a correctness-qualified candidate, not an open-source admission
claim. Full evidence and rejected alternatives are recorded in E-20260901-35.

The exact-128 tuner has since been repaired to match the selector it persists.
Slot 39 controls D256 Attention, ready-Q8 FFN/Down and PLE, but its old micro
transaction measured only FFN plus PLE and accepted a five-bit `0x1f` route
receipt. The tuner now executes Attention -> FFN -> PLE inside one device-timed
transaction and requires the complete six-bit `0x3f` evidence mask. Any partial
route or fallback is rejected rather than ranked.

The repair passes 16/16 exact-route contracts, 29/29 CUDA library tests, 18/18
tuner tests and 22/22 shared-backend tests. The SM86-only release build is fresh;
`target/release/imparo-forward.exe` hashes to
`975808d9e0f14e98f28c059f8670138c2c86f83b5857edb99ed149767bdd3e92`.
All six authenticated llama/FA q4_0 correctness gates passed again. The new
receipt SHA-256 is
`29f8a71e27e2711fe68a5f4cf0c788956f05d5efd5e04ba48a2741ac55550c3b`
and its backend fingerprint is
`254d75cd3c17c45bf655654c594e0a8d401434cdb78407aabd49f87f2648d202`.
Normal production loading with gate mode unset accepts it and a 128-token
forward produces 262,144/262,144 finite logits. No performance benchmark was
run, so short-Prefill open-source admission remains pending rather than inferred
from the safer tuner decision. Exact evidence is in E-20260901-36.

The exact-128 tuner transaction is now also weighted by the real model layer
mix. It refuses incomplete layer facts, walks all 42 E4B layers, dispatches the
controlled D256 Attention on the 35 matching layers and dispatches FFN plus PLE
on every layer. PLE uses each layer's actual vector offset. These counts are
derived from model facts rather than encoded as GPU/model constants. A pre-E-37
1:1:1 micro result is no longer considered performance provenance.

The change passes 16/16 exact-route contracts, 18/18 tuner tests, 29/29 CUDA
library tests and 48/48 harness tests. Release freshness initially failed for a
real tooling reason: `build.py` counted crate-root integration tests as production
binary inputs even though Cargo cannot relink a production target for a test-only
edit. The walker now excludes crate-root tests/benches/examples and has a
regression test; the subsequent SM86 release build is fresh.

The tuner executable hashes to
`afe8befbe25cfda415552439854bcda195f947e49dca56d77611255da69a6f23`.
The production engine remains byte-identical at
`975808d9e0f14e98f28c059f8670138c2c86f83b5857edb99ed149767bdd3e92`,
so the E-36 correctness receipt remains valid. No performance benchmark was run;
the 128-token open-source performance decision remains pending. Full evidence is
in E-20260901-37.

The exact-128 tuner now proves full per-layer route coverage rather than only
OR-ing six stage bits. Its private packed evidence binds all 35 controlled D256
Attention commits and all 42 FFN and 42 PLE commits derived from E4B model facts;
a missing layer, fallback, incomplete fact table or unrepresentable count fails
closed before a candidate can be ranked. Evidence-bearing timings retain one
complete model-weighted transaction per CUDA submission, so repetitions cannot
hide a partial route.

The change passes 16/16 exact-route contracts, 19/19 tuner tests, 29/29 SM86
static CUDA tests including the native GPU smoke, 22/22 shared-backend tests,
48/48 Python harness tests and 10/10 sealer tests. The fresh SM86 release engine
SHA-256 is
`6249dfc79c09bdf74bcaabbce5cec9013b884a12744900353b38689d7dc23114`.
All six authenticated llama/FA q4_0 correctness gates passed; receipt SHA-256 is
`f9742059cb36971333f605c2cf87ac4a6f29e3da3985da05947c6fc8af88f44f`
and backend fingerprint is
`533f425127c0bb92695c44a80996374c71ceae3783ccbd0a1fb96d6fddaa5632`.
Ordinary production loading with gate mode unset accepts it, and a 128-token
forward produces 262,144/262,144 finite logits. No throughput benchmark was run;
short-Prefill performance admission remains pending. Full evidence and retained
setup failures are recorded in E-20260901-38.

The exact-128 full-logits path now has a correctness-qualified CUDA Graph
candidate over the complete receipted sidecar DAG. The first forward builds the
all-layer packed-Q4 hot set, the second proves allocation-free sidecar commits,
the third captures and the fourth replays. Capture is fail-closed: FFN/PLE may
only look up warmed packed spans, and any missing span, scratch, producer or
direct projection discards the Graph. Attention and resident PLE norm/gather now
distinguish multi-token Prefill capture from Decode's dynamic-node contract.
Production remains env-off and unchanged.

The final one-process exact-128 smoke emitted both
`[cuda-prefill-graph] action=capture ... argmax=0 exact128_sidecar=1` and
`action=replay`. Baseline and replay produced 262,144/262,144 finite logits and
byte-identical raw output SHA-256
`01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261`.
The reusable non-timing gate and its 5/5 unit tests pass; its structured
evidence is
`docs/evidence/cuda-sm86-e4b-open-source/exact128-v26-prefill-graph-agree.json`.
The 16/16 exact-route contracts, 29/29 SM86 static CUDA tests and all six
authenticated llama/FA q4_0 gates pass. Final identities are engine
`566c6533135d2eb39302dd1544bfb45b51c5de59b2cc4928b6215eb03d185853`,
receipt
`46a08590b9beeb6990dd44636869f25f9eca25dedf9ce929e114a98afd8d5182`
and backend fingerprint
`53d5cc497a147cbd2859b27add9d9c48454fc68016328c5f95d1e2bf7ab78c72`.

This closes Graph reachability and replay correctness, not short-Prefill
performance admission. No throughput benchmark was run. The laboratory Graph
is disabled by unsetting `IMPARO_CUDA_PREFILL_GRAPH_LAB`; slot 39 zero remains
the full exact-route rollback. Full evidence and the two retained failed capture
attempts are recorded in `docs/cuda-sm86-experiment-ledger.md` E-20260901-39.

The exact-128 Graph wrapper is now a versioned production policy rather than an
environment-only experiment. CUDA space v27 adds safe-off slot 40 after the slot-39
numerical route; production can enable it only for the receipted exact-128 sidecar
DAG. Gate-suite v6 and sealer producer v6 add a seventh fixed gate that proves one
capture and one replay are byte-identical to ordinary full logits. A safe-off config
can still test the wrapper via the explicit lab override without widening production.

The v27 candidate passes all seven fixed gates, 19/19 Graph/sealer unit tests and the
complete SM86 static suite (31+2+3+16+20). Production-on evidence records
`activation=config`, 262,144 byte-identical logits and raw SHA-256
`01cdabca9607c687b1782ed3676e01efe7fa05b4f1bb74558a9b7d7614936261`.
Engine SHA-256 is
`031e57970ea817a204b29e95a735a370b373bb5e233cc4c91b87582a687d2af8`,
receipt SHA-256 is
`19042debd5f655f0ab4d6b752ba09b35e0d2df90c34ae58675761b5f99ca09a4`,
and backend fingerprint is
`48efde6a92a9f99300fe6971872e8f29f86124c4d19b1795da6a231b87848e9a`.
Slot 40 zero rolls back only Graph; slot 39 zero rolls back the full exact route.
No throughput benchmark was run, so E4B/SM86 short-Prefill open-source performance
admission remains pending. Full evidence is in E-20260901-40.

The Graph policy is now classified correctly as an end-to-end decision. Slot 40 is
`EndToEnd`/`External`: the micro tuner never ranks its zero and one values on the
Attention/FFN/PLE proxy that cannot execute `prefill_prepare`; it preserves the
incumbent until a whole-engine admission bracket exists. This is a reusable framework
rule for future workflow and Graph policies, not a CUDA-shape special case.

After that source change, the complete SM86 static suite, 22 backend tests, 19 tuner
tests and 19 Graph/sealer Python tests pass; the release build is fresh, all seven fixed
correctness gates resealed, and production-on Graph replay remains raw-byte identical.
Current engine SHA-256 is
`c79689e4af896acd28dee58f5455a3fb76ba7c2025766471f5afdfdbd8bb7687`,
tuner SHA-256 is
`ca7d019e9b909dd9ef97fa944d02b2f9e2f75416e9e434dad08fb3b7054cec58`,
and reissued receipt SHA-256 is
`39902b838bf5f27ad1dc9deeaea3c2b501ad412ba2627c29b5d2e053ac806b1d`.
These supersede E-40's pre-framework executable/receipt identities. No throughput
benchmark was run, so Decode remains previously qualified while stable 128-token and
representative long-Prefill parity remain the open-source admission gate. Full evidence
is in E-20260901-41.
The next exact-128 Down occupancy candidate is now available only through the
safe-off `r96` laboratory selector. It launches 27 R96 CTAs instead of 20 R128
CTAs for E4B's 2,560 output rows and guards the final 64-row tail. The fresh
SM86 build passes the complete static suite, and a same-binary 42-layer model
check produced byte-identical R128/R96 full logits. Generated R96 resources are
150 registers/thread, zero stack/local and 32,256 bytes dynamic shared memory.
No performance timing was run, production remains R128, and short-Prefill
open-source admission remains pending. Full evidence is in E-20260901-42.
R96 compiler-pressure screening did not produce a justified replacement for the
E-42 control. Three-CTA launch bounds spilled a 40-byte stack frame; removing
the tail load guard did not change resources; and a spill-free 112-register cap
added 56 SASS instructions while exact-128's 27-CTA grid cannot use multi-CTA
residency on 30 SMs. All variants were reverted, the final source and model
logits returned to the E-42 identity, and the final SM86 release is fresh. Full
evidence is in E-20260901-43.
The E-44 `r96w24` exact-128 Down laboratory is now correctness-qualified and
safe-off. It partitions R96 work by token group inside one 24-warp CTA, reducing
the accumulator from 64 to 16 f32 values/thread and generated registers from 150
to 78 without stack, local memory, `LDL` or `STL`; Q4/Q8 staging remains once per
CTA. The complete SM86 suite passes, the release is fresh, and same-binary R128/
R96W24 E4B runs committed 42/42 layers with byte-identical full logits. Production
remains R128 and no performance timing was run, so short-Prefill open-source
admission remains pending. Full evidence is in E-20260901-44.
The E-45 safe-off `r80` exact-128 Down candidate is also correctness-qualified.
Its 80-row tile divides E4B's 2,560 output rows into 32 CTAs, removes the R96
tail, covers all 30 SMs in the first wave and compiles to 92 registers/thread
with zero stack/local/`LDL`/`STL`. Resource arithmetic permits two CTAs/SM. The
complete SM86 suite passes and R128/R80 full E4B logits are byte-identical across
42 committed layers. R80 and R96W24 remain unranked laboratory choices;
production stays R128 and short-Prefill admission is pending because no
performance timing was run. Full evidence is in E-20260901-45.
The E-46 Direct Down scheduling refactor is correctness-qualified. Selector
spelling, profiler identity, geometry and launch-attribute configuration for
R128/R80/R96/R96W24/R64 now have one descriptor authority; descriptors remain
laboratory metadata and do not enter the tuner or receipt. A real SM86 CUDA
compile, all 76 CUDA tests, the release build and a 42-layer R128/R80 byte-exact
E4B check pass. Production remains R128, no performance timing was run, and
short-Prefill open-source admission remains pending. Full evidence is in
E-20260901-46.
E-47 closes the direct Dual-RMS/Q8 numerical and kernel-sanitizer gap and repairs
a real shared-reduction reuse race. The fused kernel now has a post-read block
barrier; 9/128/449/512-token dense outputs and ready-Q8 bytes match the split
control exactly, all four sanitizers are green, all 79 CUDA tests pass, and an
E4B run proves 84/84 route hits with byte-identical final logits. The route is
still safe-off and unpromoted because no new performance A/B or versioned
tuner/receipt admission was run. Full evidence is in E-20260901-47.
E-48 separates the 128-token fixed-cost problem from the conservative exact-
attention penalty. Existing formal 128/449 receipts imply an approximately
18.58 ms larger Imparo fixed intercept despite a better per-token slope; persistent
Q4 admission prepack is not the steady-request cause. The first whole-forward
Sidecar Graph lab removed roughly 1043 host submissions but failed a varied-token
screen by replaying stale streamed embedding/PLE rows, so its earlier 11% timing is
rejected for promotion. Capture now starts after request-specific embedding and PLE,
without changing the dynamic CUDA ABI or Metal default workflow. CUDA tests pass
32/32, model tests 42/42, release build exits zero, and six varied 128-token server
requests match Graph-off choices and all 262,144 logits byte-for-byte on 6/6 varied
prompts with traced post-embedding replay. Full-model memcheck, initcheck, racecheck
and synccheck then covered two warm forwards, capture and a varied-input replay with
zero errors (racecheck: zero hazards/warnings). The route remains default-off:
fixed-distribution, decode, formal timing and versioned receipt gates remain.
Full evidence is in
E-20260901-48.
## Open

- R2 Phase A2 is complete on SM86. RMSNorm-to-Q8 remains a valid micro-candidate,
  but the mandatory high-share Q4 matrix passed correctness, sanitizer, noise,
  resource, and provenance gates. V4's best single shape reached `0.390734x`,
  but the required pair-min score was only `0.341972x`, versus the `1.10x`
  kernel floor. Decision-A is therefore **Gate A No-Go**. The absent
  whole-prefill denominator does not block this fail-closed result. PR-G and
  Phase B/C/D production work must not proceed.
- The existing parser/trust surface, Program ABI 26 declarations, and six-symbol
  driver bridge remain frozen dormant infrastructure with `identity_ready=false`.
  They grant no winner, catalog identity, receipt, signing, installer, Commercial,
  Graph, or cross-SM authority and must not be activated after this No-Go.
- CUDA onto-v2 is at the Step 10 PR boundary. The authoritative runtime contract is
  ABI 26, tuning space 30, selector 4 and correctness gate-suite 7. ABI 23-25 and
  space-v21-v29 receipts below remain historical milestones, not loadable identities
  for the current DLL.
- Step 9 is complete on the RTX 3060 / SM86 verification host. CUDA implements the LFM2
  Q8 row/matmul routes, SiLU path, short convolution, snapshot/commit ownership and
  recurrent restore without changing the Metal workflow. LFM2's cache-basis policy is
  canonical by default and admits CUDA rotation only through an explicit, independently
  gated backend override; Metal and CPU therefore retain their established bytes. The
  Step 9 LFM2 Q4-KV receipt remains byte-exact historical ABI-25 review evidence under
  `docs/evidence/cuda-onto-v2/sm86-step9/`; it is not a loadable identity for ABI 26.
- The current loadable local E4B/SM86 evidence is the v30 Route-5 + Graph pair under
  `docs/evidence/cuda-sm86-e4b-open-source/exact128-route5-graph-v30/`. Its seven fixed
  llama/FA correctness gates and ordinary receipt-backed Graph replay pass. The latest
  exact-128 same-host performance screen is 1.0853x llama; the formal five-regime
  `ABBA BAAB` performance receipt remains intentionally separate and has not been run
  without an explicit benchmark request.
- The full workspace tests and release clippy are clean. SM86 static CUDA is 24+2+17
  tests PASS; dynamic CUDA is 27+2+17 PASS. The E4B engine KV matrix, server accept,
  share, lifecycle, seed, isolation, fork and keyless gates all pass. Lifecycle restart
  reuses 1344 tokens after preserving the resident conversation's durable cut history.
- E4B and LFM2 CUDA determinism pins are exact through 16384-token prefill, and both
  f16 and Q4-KV decode trails are single-valued. The existing Metal pin object is
  unchanged; generated repository JSON is LF-only.
- Hardware correctness/performance evidence is SM86-only. The release catalog can
  compile separate SM DLLs, but that compile coverage is not a claim of validation on
  unavailable GPUs.
- The historical at-least-2x llama.cpp target remains unproven and is not claimed. It is
  a non-blocking optimization target after PR, not a correctness, testing or PR gate.
  Performance and memory behavior beyond the measured host still require later work.
E-49 tested an exact-128 J64 token-tile candidate and repaired the admission
authority exposed by that experiment. The J64 CUDA kernel builds for SM86 at
248 registers/thread, zero stack/local and 48,128 bytes launch shared memory;
eight full-model laboratory repeats were byte-identical to their own control
route and an early wall-minus-load screen suggested about 4.7% directionally.
After versioned integration, however, the exact transaction measured the
established value-1 route at 56.89 ms, J64 value 2 at 58.59 ms and safe-off at
63.27 ms. More importantly, a real E4B eight-repeat bracket measured value 1
about 6% slower than safe-off after subtracting each process's reported load
time. Therefore neither nonzero value is promoted on the RTX 3060.

The disagreement proves the Attention/FFN/PLE micro transaction is not authority
for this workflow policy. Slot 39 is now EndToEnd/External, remains safe-off,
and the micro tuner preserves its incumbent until a whole-engine bracket plus
numerical receipt admits a value. Synthetic tuner tensor offsets are now
256-byte aligned; this fixes the misaligned-address failure exposed when a
nonzero-offset prefill MMQ was first included. CUDA space v28 binds the new
Token64 value and invalidates older receipts. The focused tuner/evidence
contracts pass, the SM86 release is fresh, and the final non-writing tuner check
exits zero with prefill_exact128_sm86_route=0. Short-Prefill admission remains
open. Full evidence is in E-20260901-49.

E-50 rejected grouped K/V physical Stream-K for exact-128. The candidate
interleaved K/V into one 60-CTA MMQ submission and one paired fixup while retaining
each projection's 30-worker boundaries. It compiled for SM86 and was byte-identical
to the old Q8-ready pair route, but warm six-forward medians were 87 ms versus
87 ms. The misleading one-shot +1.85% signal was rejected after warm-state testing;
an earlier CPU run missing `IMPARO_GPU=1` is also explicitly excluded. Candidate
source was removed, no tuner/receipt identity changed, and 128-token admission
remains open. Full evidence is in E-20260901-50.

E-53 closes the fixed numerical admission gate for the retained exact-128 route-5
plus split-Graph tuple. The candidate was regenerated against the final SM86 release,
then the authenticated fixed CUDA/llama suite passed all seven gates: 128/449/512
logits, 128/512/2000 eight-step decode and exact-128 full-logit Graph replay. The
schema-v3 receipt is adjacent to the v30 candidate, and a second run with the
correctness override absent proved ordinary receipt-backed activation, one capture,
one replay and byte-identical 262,144-f32 output. The latest retained timing remains
the E-52 same-host screen: 2246.32 versus llama 2069.80 tok/s, or 1.0853x. The user
accepted this result after the timeboxed additive search did not reach 1.10x. It is
not misrepresented as a complete formal five-regime `ABBA BAAB` release matrix;
that performance receipt and final PR packaging remain open. Exact commands, hashes,
the failed root-directory unittest invocation, its corrected 9/9 rerun and rollback
are recorded in E-20260901-53.

The E-53 repository packaging/audit is now also complete locally. Byte-identical
runtime copies live under `artifacts/`, and tracked review copies plus the ordinary-
loader Graph witness live under
`docs/evidence/cuda-sm86-e4b-open-source/exact128-route5-graph-v30/`. Review found
and repaired stale test assumptions for split Route predicates, legal slot-39 values
and numerical space v30; it did not widen a selector or change Metal. The final scoped
results are FFN/Graph/Metal contracts 20/20, host receipt fallback 9/9 and CUDA-feature
library tests 33/33. The release rebuild exited zero and reported `FRESH`; because its
executable SHA changed, the candidate was regenerated and the seven-gate receipt was
sealed again rather than reusing stale evidence. Final candidate/receipt/engine SHA256
values are `f05a3938...`, `cc76781a...` and `a8ebd4cf...`. The final runtime pair loads
without the correctness override; an exact temporary missing-receipt pair was rejected
and removed. No commit, push, merge or release has occurred.

E-54 closes the public-export safety gap for this pending PR. A new read-only verifier
shares the sync script's explicit private-directory policy, includes non-ignored
untracked files by default and rejects local paths, model/key/state files, secret
material, symlinks and missing paths. Its six unit tests pass. The pending-PR export
passes at 165 files / 4,685,808 bytes with manifest `d454c1e9...aabe0`; tracked-only
passes at 141 files with manifest `2f79e4be...5c7dd`. `.imparo/` KV/session state and
all `target-*` Cargo trees are now ignored and explicitly excluded from sync, closing
the 2,472-file accidental-stage risk found during audit. Structured evidence is beside
the v30 candidate as `public-export.json`. Laboratory source remains default-off public
engineering code; private evidence and commercial data stay in dropped directories,
and future commercial kernels must live outside the public path set. Runtime binaries,
candidate bytes and receipt identity were not changed.

E-55 supersedes E-54's provisional denylist-style packaging audit with the
`triton` branch's fixed-tree exporter. The reviewed public surface is now one
UTF-8 byte-sorted allowlist of 292 exact Git paths with allowlist SHA256
`ce71d0cc549692ed2b9a4a32ccdabee87c2c049915547be9e84d3cfc6cd89a28`.
The exporter reads committed blobs rather than the worktree, and the detached
verifier independently rechecked all exported bytes. Secret, private-reference,
portable-path and binary-policy scans all passed. Machine-specific paths found
in the contributor experiment ledger were replaced by explicit environment
variables before the export was accepted. Raw sanitizer logs, internal receipts,
handoff documents and rejected private laboratory variants remain outside the
public set; the production E4B SM86 sources, contract tests and contributor
optimization playbook are included.

The resolved integration was validated both internally and as an exported tree.
The exact public tree passes locked all-target checks for `imparo-tune`,
`imparo-cuda` and `imparo-server`. A fresh SM86-only CUDA build then passed
107/107 library and contract tests; host receipt/fallback tests passed 22/22 and
tuner tests passed 25/25. The fixed-tree exporter suite passed 26 tests with one
Windows-only executable-mode test skipped because POSIX mode bits are not stable
on Windows. The stale Step-1 assertion exposed by this run was corrected to bind
the current E4B tuning space 30 while independently requiring Program Pack
`identity_ready=false`.

The retained open-source candidate remains exact-128 Route 5 plus split CUDA
Graph: 2246.32 versus llama 2069.80 tok/s on the recorded RTX 3060/SM86 screen,
or 1.0853x, with all seven fixed correctness gates passing. This is the
user-accepted result after the timeboxed search failed to reach 1.10x; it is not
a cross-SM claim or a substitute for the intentionally deferred five-regime
`ABBA BAAB` performance receipt. Rollback is the entire detached integration
commit: built-in safe-off native routes remain available, Metal is unchanged,
and the dormant Program Pack surface gains no production authority. No push,
merge, tag or release was performed during this audit.

E-56 revalidates that retained candidate on v2 base
`5bd89c814a6f2617a2f3982ca154495943186168`. The rebuilt SM86 engine has SHA256
`1def2dc7...ce2d3`; the regenerated candidate and schema-v3 receipt are
`066c7c83...a6320e` and `569a62e6...12505`. All seven fixed numerical gates
passed again, and ordinary receipt-backed loading recorded one Graph capture,
one replay and byte-identical output SHA256 `e50c4aa8...454bf`. The two-hour
additive optimization window produced no admissible improvement, so the user
accepted the existing 1.0853x directional screen as the convergence point.
No new performance claim, push, merge, tag or release is implied.

### E-57: final latest-v2 integration and PR gate (2026-09-02)

The retained E4B/SM86 candidate is now applied to
`origin/v2@b9a11709c8fd9929c1aed7049fa20ee36aada497` without changing the existing
Metal workflow. Two latest-v2 contract drifts found by the release suite were
repaired: the CUDA correctness fixture now initializes the host config `path`,
and the FFN source contract follows the renamed `lookup_tensor` helper. The full
workspace tests and release clippy with `-D warnings` exit zero.

The SM86-only CUDA product build is fresh. Complete release-contract suites pass
for both `cuda-static` and `cuda-dynamic`; the static suite includes live RTX 3060
event/profile, KV arena and exact-route smokes. The regenerated candidate, receipt
and engine SHA-256 values are respectively `066c7c83...6320e`,
`54db5491...393f7` and `841ad53d...d1f44`. All seven fixed llama/FA gates pass.
With `IMPARO_CORRECTNESS_GATE` absent, the ordinary loader activates the config,
captures once, replays once and matches all 262,144 logits byte-for-byte.

The first PR CI pass exposed four portability defects rather than runtime gate
failures: the Metal probe omitted the new `attn_fa_nsg` registry coordinate;
minimal CUDA 13 installations omit `cuda_profiler_api.h`; Linux CPU parsing retained
the tab before `/proc/cpuinfo`'s `model name` colon and therefore could not reload its
own strict fingerprint; and detached public-tree tests embedded repository-only audit
fixtures at compile time. Each path now has a narrow fail-closed repair. The Linux
receipt suite passes 21/21 under Rust 1.98, and the detached public tree builds without
the private fixtures.

The detached public export of pre-documentation staged tree
`19ae7b5abc3799f975c9604f07725be94fb7c83e` passes with 292 files / 6,781,626
bytes; all path, binary, private-reference and secret scans pass. Final-commit CI
regenerates this report after the evidence update. The inherited
full-repository rustfmt drift is already present on the clean v2 base and is not
mixed into this CUDA PR; every changed file passes `git diff --check`. Rollback
remains receipt removal or slot 40/39 safe-off, followed by process restart. No
auto-merge, tag or release is authorized.