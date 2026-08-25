// Metal backend. Model-agnostic: kernels are named for the operation and quantisation,
// never for a model.
//
// WHY OBJECTIVE-C++ (user decision, 2026-08-20, settled -- do not relitigate): this file
// is the ONE sanctioned non-Rust host file in the engine. The alternative -- driving
// Metal from Rust -- requires a bindings dependency (metal-rs / objc2-metal), and the
// user prefers the current direct approach over taking that dependency. So the language
// rule reads: Rust everywhere else, MSL kernels, and this file as the thin, direct
// Metal-API glue. Keep it thin: orchestration and model logic belong on the Rust side.
//
#include <algorithm>
//
// Two properties matter and both are Apple-Silicon specific:
//   * the weight mapping is wrapped with newBufferWithBytesNoCopy, so the GPU reads the
//     same pages the CPU mapped -- no upload, no copy, and weights never enter process RSS;
//   * ACTIVATIONS STAY RESIDENT. The first version issued one command buffer per matmul
//     with waitUntilCompleted and two memcpys, ~420 round trips per token at ~192 us each.
//     Now the caller opens one command buffer, encodes a whole layer, and syncs once.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>

namespace {

extern "C" void * objc_autoreleasePoolPush(void);
extern "C" void   objc_autoreleasePoolPop(void *);

// The MSL lives in native/imparo.metal (real Metal syntax highlighting, its
// own file history); build.rs wraps it back into this raw-string constant at
// compile time. Runtime source compilation is unchanged -- no metallib, no
// extra toolchain dependency.
#include "imparo_msl.inc"

enum Buf : uint32_t {
    B_X = 0, B_CUR, B_Q, B_K, B_V, B_ATTN, B_O, B_G, B_U,
    B_MODEL0, B_MODEL1, B_MODEL2, B_LOGITS, B_TOKENS, B_TMP, B_ATTN_PART, B_XH,
    // Half scratch a quantized KV cache is dequantised into, once per layer per prefill
    // batch, so the prefill attention kernels read exactly what they read under f16.
    B_KDQ, B_VDQ,
    // Second half mirror, aliasing the free half of U's arena region: the fused up
    // epilogue writes G's half copy here while B_XH still holds CUR's (the same
    // dispatch READS B_XH, so the two mirrors cannot share bytes).
    B_XH2,
    // Model-private slots, named by the model rather than here. Appended so every index
    // above keeps its wire value.
    B_MODEL3, B_MODEL4, B_MODEL5, B_MODEL6, B_MODEL7,
    B_DEFINED
};
// THE CEILING IS THE HAZARD MASK, not a taste in array sizes. hb(id) is `1ull << id` and
// bits 62 and 63 are HZ_KVK and HZ_KVV, so an id there would silently alias a KV cache's
// hazard bit and lose a real dependency. Sixty-two is what a 64-bit mask can express, and
// raising it further means replacing the bitmask with per-buffer epochs, not editing this
// line. imparo_metal_buf_count reports it and the Rust side checks BufId::COUNT at init.
constexpr uint32_t B_COUNT = 62;
static_assert(B_COUNT <= 62, "hb() needs id < 62: bits 62 and 63 are the KV hazard bits");
static_assert(B_DEFINED <= B_COUNT, "buffer table smaller than the ids it must hold");

uint32_t g_sgs = 8;
uint32_t g_rows = 1;
// LANES_PER_ROW, selected via a function constant. 16, measured, is the decode default:
// the stored v6 tuning picked 16 (38.53 tok/s) and an A/B on the corrected engine reads
// 38.3 / 38.1 / 37.4 for 16 / 32 / 8 -- the compiled default is what every host runs
// until the tuner does, so it carries the measured value, not a placeholder. Logits are
// bit-identical across 8/16/32 (verified at n=2000): the per-lane partials are reduced in
// a fixed order regardless of the lane count.
uint32_t g_lanes = 16;
uint32_t g_lanes_log2 = 4;

// Attention threadgroup width, PER PHASE. Prefill and decode disagree: measured
// 418.6 prefill / 36.5 decode at 256, against 414.2 / 36.8 at 512. One tuned scalar has to
// give up one of them, so the search space carries both.
uint32_t g_attn_threads   = 512; // decode: many positions, one token
uint32_t g_attn_threads_pf = 256; // prefill: many tokens already in flight
// Below this many attention threadgroups the GPU is not filled, so the KV range is split.
// 72 is four per core on an 18-core M3 Pro; it is a tunable, not a constant of the design.
uint32_t g_attn_min_tgs   = 72;
// Force the streaming decode kernel on regardless of depth (debug A/B only;
// the depth router below is what decides in production).
uint32_t g_attn_stream    = 0;
// The span where the streaming decode kernel takes over from the score-tile one.
//
// Measured per-dispatch on the reference M3 Pro with paged_attn_check's
// crossover sweep (cold layer-alternating, median of three, both kernels
// interleaved), us/dispatch at hd 512:
//
//   span   2048  3072  4096  5120  6144  8192  12288  16384
//   stream  344   192   217   250   271   347    478    603
//   scoret  188   195   250   317   410   458    662    951
//
// so 3072 is where the streaming kernel takes the lead and never gives it
// back. The earlier 8192 came from END-TO-END decode runs, which cannot
// resolve this: only 7 of 42 layers are full-attention, so at 4k the choice
// is worth ~0.9% of a decode step while repeat runs of one configuration
// spread 7%. Kernel choice is a per-dispatch question and is measured as one.
//
// A COMPILED default with an env override (IMPARO_ATTN_STREAM_MIN_POS). The tuner
// declares it and MEASURES it, but its estimator declines to move it -- the samples do
// not locate a crossing tightly enough to act on -- so this value is what ships, and it
// has to be the one with end-to-end evidence behind it.
//
// 32768, NOT 3072, and the evidence is the 9-cell bracket table:
//
//   5651 tokens    3072 streams, 32768 takes the score tile   decode +1.4 to +1.8%,
//                                                             on f16, q8_0 AND q4_0
//   17041 tokens   3072 and 32768 differ in path              indistinguishable:
//                                                             35.2/34.9 against 35.0/35.3
//   449 tokens     both below either boundary                 unchanged, as expected
//
// So the score-tile path is worth having up to at least ~17k, and the gain is specific to
// the band where the two settings differ -- which is what makes it a boundary result and
// not drift. Raising the default is how that measured gain survives an estimator that
// (correctly) refuses to guess.
// 8192, AND THE TUNER DERIVED IT. Its span_crossing fit reads
//     hi = 18428 us + 0.2333/pos   lo = 17338 us + 0.3717/pos   -> meet at 7873 -> 8192
// on ratios that are monotone across 4096..32768 (1.00 1.01 1.02 1.03 1.06 1.16) with a
// 2.5% noise floor. That became trustworthy only when AttentionDecodeDeep started
// streaming the weight mix the engine runs attention against; before that the workload had
// the SIGN wrong and swung 10% between runs on the deciding rung.
//
// Checked against the bracket at every depth, 2 legs each:
//     f16   10003  8192 -> 36.9  32768 -> 37.0      -0.3%
//           11992  8192 -> 36.3  32768 -> 36.4      -0.3%
//           15987  8192 -> 35.1  32768 -> 35.0      +0.3%
//     q4_0  15987  8192 -> 35.6  32768 -> 35.0      +1.7%
// Neutral on f16, a real gain on a quantized cache -- which is where it was unreachable
// until the depth boundary was made to outrank grouping (see stream_first).
// UINT32_MAX means DERIVE FROM THE CACHE TYPE, resolved per dispatch at stream_first --
// f16 8192, quantized 512. It is not a stored number because the cache type can differ per
// layer (kvq_mask_on), which one stored value cannot express. An explicit override (env or
// tuner) is used as given; to force score-tile use 1 << 30, which no span reaches.
uint32_t g_attn_stream_min_pos = 0xFFFFFFFFu;
// Stream slice-count cap (IMPARO_ATTN_STREAM_SLICES overrides; see the sweep
// note at the dispatch site).
uint32_t g_attn_stream_slices = 32;
// GQA row sharing in the stream kernel: query heads per threadgroup.
// DEFAULT 2 -- at E4B's 8-over-2 geometry it halves the unique KV bytes a
// decode step pulls through the cache and measured 595 us/dispatch against
// HQ=1's 701-729 (cold 6-layer alternating probe, 16k, reference M3 Pro).
// Output is bit-identical: the per-head arithmetic and block order do not
// change, only which threadgroup performs them. IMPARO_ATTN_HQ=1 reverts.
uint32_t g_attn_stream_hq = 2;
extern "C" void imparo_metal_set_attn_stream_hq(uint32_t v) {
    g_attn_stream_hq = v >= 2u ? 2u : 1u;
}
extern "C" uint32_t imparo_metal_attn_stream_hq_current(void) { return g_attn_stream_hq; }
extern "C" uint32_t imparo_metal_attn_stream_slices_current(void) {
    return g_attn_stream_slices;
}
extern "C" void imparo_metal_set_attn_stream_slices(uint32_t v) {
    g_attn_stream_slices = v < 1u ? 1u : (v > 256u ? 256u : v);
}
// 64, not 16: the GQA-grouped path has a quarter of the threadgroups for the same
// work, so it needs four times the slices to fill the machine. The partial buffer
// this sizes is n_heads x slices x (head_dim + 2) floats -- 1 MiB at 64.
constexpr uint32_t ATTN_MAX_SLICES = 256;   // x 8192-score slices = 2M positions
uint32_t g_skip_mma       = 0;   // diagnostic: stage but do not multiply
// Diagnostic (task #5): when nonzero, multi-token batches use the GEMV ONLY for matmats
// whose n_out equals this value; everything else takes the GEMM. Sweeping the model's
// n_out values isolates which op's GEMV routing carries the run-to-run wobble.
// Prefill GEMM activations staged as HALF (converted into XH, which aliases U's arena
// pages -- U is dead during prefill since the epilogue fuses gelu). The technique is
// llama.cpp's own mul_mm staging; approved 2026-08-20, replacing the old exact-to-self
// logit gate with the fork-agreement gate (dev_harness/logit_agree.py). DEFAULT ON:
// -70 ms on a 5642-token prefill, twice replicated; IMPARO_HALF_A=0 restores float.
// The in-kernel threadgroup variant of the same idea measured 73% SLOWER (occupancy),
// which is why the conversion rides through device memory.
uint32_t g_half_a         = 1;
// Narrowest batch that may take the half-A path. It is 2 -- every prefill width -- so the
// precision of a position's activations does NOT depend on how wide the batch that
// carried it was. That is what lets a resumed prefill reproduce the cold pass BYTE-EXACTLY
// at any resume point: a 40-token tail chunk now stages the same way the 232-token chunk
// that originally computed those positions did.
//
//   floor 64: cold 744 = 22.156891   split@704 = 22.085224   q4_0, differ
//   floor 2 : cold 744 = 22.156891   split@704 = 22.156891   byte-equal
//
// It was 64 for a run-to-run nondeterminism seen on small remainder batches (task #5's
// wobble). That does not reproduce: 2026-08-25, floor 1, lengths 517..527, 6 runs each,
// 1/6 distinct at every length. Decode is left alone (n_tok == 1 takes the GEMV, not this
// GEMM) -- hence 2 rather than 1.
//
// A constant, not a knob: a tuned per-machine boundary is the defect, whatever value it
// lands on. Re-measure by editing it here, so the build that was verified is the build
// that ships.
constexpr uint32_t HALF_A_MIN = 2;
// KV cache storage types, GGML ids (1 = f16, 2 = q4_0, 8 = q8_0). User config
// (--cache-type-k/-v via IMPARO_CTK/IMPARO_CTV), NOT a tuned knob. f16 keeps the
// pre-feature path bit for bit: same kernels, same pipelines, no branches taken.
uint32_t g_kv_type_k      = 1;
uint32_t g_kv_type_v      = 1;
// Diagnostic (IMPARO_KVQ_MASK_K/_V hex layer bitmasks + IMPARO_KVQ_TYPE): quantize
// ONLY the masked layers' K/V while the global types stay f16. Requires f16-sized
// caches (no --cache-type / IMPARO_CTK/CTV) and prefill-only use -- the decode
// kernels are specialised on the GLOBAL types and would misread masked layers.
uint32_t g_kvq_mask_k     = 0;
uint32_t g_kvq_mask_v     = 0;
uint32_t g_kvq_type       = 2;   // 2 = q4_0, 8 = q8_0
// Roundtrip mode (IMPARO_KVQ_RT): the masked layers' stores quantize+dequantize but the
// cache keeps the f16 LAYOUT, so every reader treats it as f16.
uint32_t g_kvq_rt         = 0;   // 0 off, else 2/8
uint32_t g_kvq_rt_lo      = 0;
uint32_t g_kvq_rt_hi      = 0xffffffffu;
static inline uint32_t kv_eff_type(uint32_t layer, uint32_t is_v) {
    const uint32_t base = is_v ? g_kv_type_v : g_kv_type_k;
    if (base != 1u) { return base; }
    if (g_kvq_rt != 0u) { return 1u; }   // roundtrip: f16 layout everywhere
    const uint32_t msk = is_v ? g_kvq_mask_v : g_kvq_mask_k;
    return (layer < 32u && ((msk >> layer) & 1u)) ? g_kvq_type : 1u;
}
// Task #5 probe (IMPARO_BUFSUM=1): checksum every matmat output of small multi-token
// batches into TMP, printed at imparo_metal_end. Diagnostic, default off.
typedef struct { uint32_t x, y; } uint2p_t;
// Conversion cache: which buffer XH currently mirrors, and how many elements. Invalidated
// in haz() when anything writes the cached source, and at every begin().
uint32_t g_xh_src         = 0xffffffffu;
uint64_t g_xh_elems       = 0;
uint32_t g_xh_buf         = B_XH;   // which scratch holds the mirror (B_XH or B_XH2)

uint32_t g_skip_attn      = 0;   // diagnostic: encode no attention at all
// EXPERIMENT (IMPARO_CONCURRENT=1, default off): encode with MTLDispatchTypeConcurrent and
// insert a barrier only where a dispatch touches a buffer the unbarriered window already
// wrote (or writes one it read). Every dispatch site declares its read and write sets, so
// the independence is DERIVED, not hand-marked -- the earlier attempt hand-marked four
// pairs and measured neutral-to-worse. Values are bit-identical either way: no kernel's
// arithmetic changes, only whether independent dispatches may overlap.
// Concurrent dispatch with per-site hazard tracking (haz()), DEFAULT ON: the
// reference encoder is concurrent-with-barriers too, and serial encoding
// measured +1.4 ms/token of drain bubbles at 16k decode. Bit-identical output
// either way -- it changes overlap, not arithmetic. IMPARO_CONCURRENT=0 reverts.
uint32_t g_concurrent     = 1;
uint64_t g_haz_reads      = 0;   // buffer bits read since the last barrier
// ARENA ALIASING was task #5: place() used to wrap overlapping byte ranges of ONE
// allocation in DISTINCT MTLBuffer objects (the ffn group's G spans the attn group's
// bytes; XH rides U's pages). Metal's hazard tracker is per-RESOURCE, so a serial
// encoder happily overlapped the FFN gate's write to G with attention still reading
// the same bytes as Q -- a ~25%/rep logit wobble at warm clocks, absent from llama.cpp
// (no aliasing) and from the standalone repro (honest separate buffers). Explicit
// memoryBarrierWithScope did NOT cure it (concurrent-encoder machinery).
// THE FIX IS RESOURCE-LEVEL: one MTLBuffer spans the whole arena and every placed
// logical buffer is an OFFSET into it (g.buf_off) -- one resource per byte means the
// tracker sees every hazard itself, on any Mac, per spec. Proven by discriminator:
// IMPARO_NO_ARENA_OVERLAP=1 (separate resources) was 1/60-stable where overlap wobbled.
uint64_t g_haz_writes     = 0;   // buffer bits written since the last barrier
// Bit i (i < B_COUNT) is activation buffer i. The KV caches get one bit per side, not one
// per layer: the only same-layer store->attention dependency is real anyway, and a
// cross-layer false conflict never binds because the ops between them barrier regardless.
//
// SIXTY-FOUR BITS, not thirty-two, and the width is the only thing that caps how many
// buffers an engine can have. At uint32_t the reserved KV bits sat at 30 and 31, so a
// model was allowed thirty activation buffers -- a limit with no reason behind it beyond
// the integer someone picked. Nothing else about the mechanism cares: it is a
// set-intersection test that emits a barrier when a dispatch conflicts with the
// unbarriered window. A bitmask of any width still has SOME ceiling; if a model ever needs
// more than sixty-two buffers the answer is per-buffer epochs (last_write[id] against a
// counter), which trades one AND for iterating a small read set.
constexpr uint64_t HZ_KVK = 1ull << 62;
constexpr uint64_t HZ_KVV = 1ull << 63;
// Diagnostic: skip one kernel category entirely, so the wall-time difference is its share.
// Mirrors GGML_METAL_SKIP_OP added to the fork, so the two breakdowns are comparable.
// Output is wrong on purpose. 0 = skip nothing.
uint32_t g_skip_cat       = 0xffffffffu;  // NOT 0: PC_MATMAT_PREFILL is 0, so a 0
                                          // sentinel made category 0 unskippable and
                                          // silently measured the baseline instead.
// Set for the one matmul whose write-back folds in the gated activation; cleared after.
uint32_t g_epilogue       = 0;
// EPI_GELU. See imparo_metal_set_epilogue_act.
uint32_t g_epi_act        = 1;
// rms_norm threadgroup width, per phase. A one-row norm at decode is pure latency and
// wants every thread it can get; a 440-row prefill norm already has parallelism from the
// rows. Measured decode 36.7 at 256 against 37.1 at 1024, prefill 482.7 against 481.2.
uint32_t g_qtile          = 1;   // query-tiled prefill attention with matrix scores
// The combined prefill-attention rewrite: staged float Q, register accumulator,
// spill-only tails. DEFAULT ON: bit-identical to the tiled kernel at n=128/900/2000/5642
// and 4.7% faster end to end on a 5642-token prefill (11538 -> 10975 ms). IMPARO_QCOMB=0
// restores the tiled kernel; 2/3 scope it to full-attention/windowed layers for bisection.
// While it is on, attn_blk and attn_threads_prefill are INERT at prefill (they configure
// the tiled kernel); they still matter under IMPARO_QCOMB=0.
uint32_t g_qcomb          = 1;
// Narrow-N GEMM routing (task #11): batches of 2..g_nb8_max tokens take the 64x8
// tile; 0 disables it (IMPARO_NB8=0, the A/B switch). Bit-identical either way,
// and with it on no multi-token batch can reach the GEMV in shipping (task #5).
// The boundary and the tile variant are TUNER-OWNED (hostconfig v10): these are
// the compiled defaults an untuned host runs. 47 is the measured crossing on the
// reference M3 Pro -- the tuner's scan found [47,47,47] and the end-to-end MTP
// bench confirms it (16-token forwards: 19.05s at boundary 15 -> 10.25s at 47).
uint32_t g_nb8_max        = 47;
uint32_t g_nb8_shape      = 0;   // 0: two simdgroups / 64 threads; 1: one / 32
// GEMV boundary: batches of 2..g_gemv_max_tok tokens take the GEMV (TOKEN_TILE 4);
// above it, the GEMM family (nb8 then wide). TUNER-OWNED (hostconfig v11). The
// boundary was pinned at 1 while the multi-token GEMV wobbled (task #5); the wobble
// is root-caused (arena aliasing, fixed at the resource level), so the choice is a
// performance question again. NOTE the GEMV's k-order differs from the GEMM family's,
// so moving this boundary legitimately shifts logits for whole chunks of 2..N tokens
// -- pinned lengths chunk wide (128, 464) and are unaffected.
uint32_t g_gemv_max_tok   = 1;
// Force the QT 8 path, which halves threadgroup memory (12 KB against 24) and doubles the
// KV re-reads. A diagnostic for whether occupancy or traffic binds this kernel.
uint32_t g_attn_short     = 0;
// Register-blocking depth in the two attention matrix phases: 2, 4 or 8, one pipeline each.
// Selects among the qtile16h prefill attention kernels, which DO NOT RUN: g_qcomb
// defaults to 1 and the qcomb branch catches every prefill attention dispatch and returns
// before the qtile branch is reached (verified with IMPARO_ATTN_WHICH: 84 of 84). Kept as
// an env override for anyone forcing IMPARO_QCOMB=0, but it is no longer a tuner knob --
// it was a stage-2 knob, so the tuner ran a full prefill and decode per candidate, three
// values, to measure a kernel that never executes.
//
// The shape knob this kernel path SHOULD have is BLK on the live qcomb kernels, which is
// a template literal there (BLK 2 on the QT-16 variants, 4 on the QT-8 ones). See #44.
uint32_t g_attn_blk       = 4;
// BLK for the QT-8 qcomb prefill attention kernels, the ones q4/q8 prefill runs on.
// 4 was the shipped literal and leaves half the simdgroups idle (4 work units over NSG 8);
// 2 fills them. Tuner-owned so the DEVICE decides which trade wins.
uint32_t g_qcomb_blk      = 4;
// Skip the prefill mask loop on a block that is provably live for every query in the
// tile. A function constant in the kernel, so this is a BUILD choice: the ON pipelines
// carry no flag and the OFF pipelines carry no fast-path test. IMPARO_ATTN_LIVE_MASK=0
// builds the OFF set, which is the whole A/B.
uint32_t g_attn_live_mask = 1;
// SIMDGROUPS DERIVED from the measured accumulator cliff. The kernel holds, per
// simdgroup: the output accumulator o[QROWS][NDB] with NDB = hd/8/nsg, the score
// accumulators sacc[QROWS][BLK], and the Q and K operand pairs. Fewer simdgroups means a
// bigger NDB and more live matrices, so the cliff sets a FLOOR on nsg.
//
// It reproduces the shipped 16 and explains the note beside it. At nsg 8 the count is 28
// against a measured cliff of 24 -- which is why "at 8 it spills, 14% slower" was
// recorded by hand two sessions ago. Deriving it turns that note into arithmetic.
//
// nsg 32 also fits, at 16 matrices, and is rejected for a different reason: 1024 threads
// with a 32 KB threadgroup leaves ONE threadgroup per core, so its idle simdgroups have
// no neighbour to cover them. Intra-threadgroup occupancy usually does not matter --
// measured -- but that is because other threadgroups are resident, and at 1024 threads
// they are not.
static uint32_t qcomb_derive_nsg(uint32_t qt, uint32_t hd, uint32_t blk, uint32_t max_acc,
                                 uint32_t fallback) {
    if (max_acc == 0u) { return fallback; }
    const uint32_t qrows = qt / 8u;
    for (uint32_t nsg = 8u; nsg <= 32u; nsg *= 2u) {
        if ((hd / 8u) % nsg != 0u) { continue; }          // NDB must be whole
        if (nsg * 32u > 512u) { continue; }               // one threadgroup per core past this
        const uint32_t ndb = hd / 8u / nsg;
        const uint32_t live = qrows * ndb + qrows * blk + 2u * qrows + 2u * blk;
        if (live <= max_acc) { return nsg; }
    }
    return fallback;
}

// PT DERIVED, not chosen. This inverts qcomb_tg_floats: given the DEVICE's real
// threadgroup limit and the kernel's shape, the widest position tile that fits is
// arithmetic, and rounding down to a whole number of work units (8*BLK) is the only
// judgement in it.
//
// It reproduces the two values that were hand-picked and shipped -- 240 at head_dim 512,
// 112 at 256 -- which is the point: a human computed this formula once and froze the
// answer, so the answer was correct here and silently wrong anywhere with a different
// limit. Derived, it follows the device.
static uint32_t qcomb_derive_pt(uint32_t qt, uint32_t hd, bool half_q, bool device_spill,
                                uint32_t blk, uint64_t budget_bytes) {
    const uint64_t floats = budget_bytes / 4u;
    const uint64_t sq = (half_q && qt == 16u) ? (uint64_t)qt * hd / 2u : (uint64_t)qt * hd;
    const uint64_t spill = device_spill ? 0u : (uint64_t)qt * hd;
    const uint64_t fixed = sq + 2u * qt + (uint64_t)(qt / 8u) * 64u + spill;
    if (fixed >= floats) { return 0u; }
    const uint64_t pt = (floats - fixed) / qt;
    const uint32_t unit = 8u * blk;
    return (uint32_t)(pt / unit) * unit;
}

// Derived at init from the queried threadgroup budget; see qcomb_derive_pt.
// The MEASURED accumulator cliff, from the cached host profile. 0 = never measured on
// this host, in which case every derivation below keeps its compiled fallback rather
// than computing against a number nobody established.
uint32_t g_measured_max_acc = 0;
uint32_t g_pt_512x        = 240;
uint32_t g_pt_256x        = 112;
uint32_t g_nsg_512x       = 16;
uint32_t g_blk_x          = 2;   // position blocks per work unit, QT-16 kernels
uint32_t g_nsg_256x       = 8;
extern "C" void imparo_metal_set_measured_max_acc(uint32_t v) { g_measured_max_acc = v; }
extern "C" uint32_t imparo_metal_pt_512x(void) { return g_pt_512x; }
extern "C" uint32_t imparo_metal_pt_256x(void) { return g_pt_256x; }
// IMPARO_ATTN_STAGE: how far through the attention kernel to run. 1 scores only, 2 adds the
// softmax, 3 (default) is the whole thing. Output is WRONG below 3; this exists to split the
// kernel's time between its phases, which the category profiler cannot do because timestamp
// sampling returns nothing on this device.
uint32_t g_attn_stage     = 3;
// Group the query heads that share a KV head into one threadgroup at decode.
//
// OFF BECAUSE IT WAS MEASURED AND IT STILL LOSES ON THIS DEVICE -- but by 2%, not the 15%
// it lost as first written. At 12009 positions, pairs on a verified-idle machine:
//
//     grouped, as found                    30.8 / 30.9    -15% vs ungrouped
//       + Q address hoisted out of the innermost loop
//                                          31.2 / 31.1
//       + softmax barriers batched, 16 -> 4
//                                          31.4 / 31.5
//       + templated on group size, so the accumulator array is HQ not MAXHQ
//                                          36.0 / 35.6     -2%
//     ungrouped (each KV byte read 4x)     36.5 / 36.5
//
// The traffic arithmetic said grouping should WIN by 9%, because it reads each KV byte once
// instead of once per query head in the group. It was wrong, and the reason is that these
// kernels are not bandwidth-bound: the kernel note below measures 69 GB/s against 147 for a
// plain read. At 47% of achievable bandwidth the redundant reads are absorbed for free, so
// the traffic term buys nothing here -- while grouping's cost was real and mostly self-
// inflicted. Two terms decide it:
//
//     traffic    (n_heads / share) * head_dim * 2 * bytes_per_elem / bandwidth
//                grouping divides this by `share`. Binds only when the path is near peak.
//     registers  accumulators per thread = share * head_dim / threads
//                grouping multiplies this by `share`. Bound here: sizing the array for the
//                largest supported group instead of the actual one cost 8% by itself.
//
// So the choice is genuinely per-device, which is why it is a knob and not a constant: a
// machine with less bandwidth per FLOP makes the first term bind and grouping win. The 2%
// that remains on THIS device is the V pass, where scores[j * ns + sq] is `share`
// threadgroup reads per V element strided by ns.
// UINT32_MAX = DERIVE from the cache type at dispatch (see the note at the selection
// site). An explicit IMPARO_ATTN_GQA, INCLUDING 0, overrides -- which needs a sentinel
// distinct from 0, or "off" and "unset" would be the same value.
uint32_t g_attn_gqa       = 0xFFFFFFFFu;
// The register-tiled GEMM is the default: it is 1.5x the old prefill kernel on every
// shape measured and produces bit-identical logits. Shape 2 is the fastest on the host
// this was developed on; the tuner is what should choose it per device, and IMPARO_RT=0
// keeps the old kernel reachable as an A/B reference.
uint32_t g_rt             = 1;   // use the register-tiled prefill GEMM
// Shape 1 = (NA 4, NB 4, SGX 2, SGY 2): 64 rows x 64 tokens, 128 threads, SIXTEEN
// accumulators. It ties shape 2 on ffn_down and wins end to end, 498.1 against 493.4 tok/s.
// It was 4.10 against 4.21 before the loop was pipelined -- 16 accumulators have a better
// multiply-to-load ratio (8 loads per 16) and only pay off once the loads are overlapped.
uint32_t g_rt_shape       = 1;   // which register-tile candidate
uint32_t g_rt_all         = 0;   // build every candidate, for sweeping only
uint32_t g_shrink         = 1;   // return buffer memory when the batch narrows

// Which activation range the half copy currently holds, and the dispatch count it was
// valid at. Reuse requires that NOTHING has been dispatched since -- listing the kernels
// that write buffers would be a dozen call sites to keep in step, and missing one would
// corrupt results silently. Counting every dispatch cannot miss one.
uint64_t g_disp_seq  = 0;
uint32_t g_cvt_valid = 0, g_cvt_src = 0, g_cvt_off = 0, g_cvt_n = 0;
uint64_t g_cvt_seq   = 0;

// (NA, NB, SGX, SGY): tokens-per-simdgroup, rows-per-simdgroup, and how many simdgroups
// span each axis. Threads = SGX*SGY*32; accumulators per simdgroup = NA*NB.
constexpr uint32_t RT_CANDIDATES = 8;
// RT_TOKENS decides how many times a batch re-reads the weight matrix -- ceil(n_tok /
// RT_TOKENS) passes -- so it is the lever on the staging third of this kernel's time.
//
// Widening it by adding SIMDGROUPS does not work: (2,4,2,8) and (2,4,4,4) both hold 8
// accumulators like the best shape and both collapse to ~110 and ~97 tok/s at 512 threads,
// while (2,2,4,4) survives at 512 threads with 4. What binds is threads x registers per
// thread against the register file, not either alone. So the wide tiles below buy tokens
// with NA -- more tokens per simdgroup -- and keep the threadgroup at 256.
constexpr uint32_t RT_SHAPES[RT_CANDIDATES][4] = {
    {2, 4, 2, 2},   //  64 rows x  32 tok,  128 thr, 8 acc
    {4, 4, 2, 2},   //  64 rows x  64 tok,  128 thr, 16 acc: 8 loads per 16 multiplies
    {2, 4, 2, 4},   //  64 rows x  64 tok,  256 thr, 8 acc
    {2, 4, 4, 2},   // 128 rows x  32 tok,  256 thr, 8 acc: halves activation re-reads
    {4, 2, 4, 2},   //  64 rows x  64 tok,  256 thr, 8 acc
    {2, 2, 4, 4},   //  64 rows x  64 tok,  512 thr, 4 acc
    {4, 4, 2, 4},   //  64 rows x 128 tok,  256 thr, 16 acc: 4.17, more threads lose again
    {8, 2, 2, 2},   //  32 rows x 128 tok,  128 thr, 16 acc
    // Two more were built and MEASURED at a 5642-token prompt, then removed: the tuner
    // should not spend sweeps on known losses. rows x toks is the output tile and
    // threads x accumulators is the register budget, which the working shapes all hold
    // near 2048 (128x16, 256x8, 512x4) -- so the tile cannot exceed 4096.
    //
    //   {2,4,2,8}   64 x 128, 512 thr, 8 acc   12503 ms   tile 8192, over budget:
    //                                                     dequant -799 ms, multiply +1543
    //   {4,1,4,4}   32 x 128, 512 thr, 4 acc   16118 ms   in budget, but half the rows
    //                                                     doubles activation re-reads
    //
    // A 128-token tile really does halve the dequantisation passes (each weight tile is
    // staged once per TOKEN tile: 512/128 = 4 against 512/64 = 8), worth 799 ms with
    // multiplies skipped. There is no way to spend it: at 4096 outputs the three slices
    // are 64x64 (shape 1, 11759 ms), 32x128 (16118) and 128x32 (shape 3, 13239). Shape 1
    // is the optimum of its family, and beating it needs fewer registers per output --
    // a different staging scheme, not another entry in this table.
};

// ---- Q8_0 WEIGHTS ------------------------------------------------------------------
//
// Three routes, none sharing launch geometry with Q4_0 (see the note above the kernels in
// imparo.metal). The knobs mirror the Q4_0 family one for one:
//
//     Q4_0                        Q8_0
//     lanes / nr0                 q8_decode_sgs / q8_decode_rows      decode
//     nb8_shape / nb8_max         q8_token_tile / q8_batch_sgs        narrow batch
//     rt_shape                    q8_gemm_shape                       wide prefill
//     gemv_max_tok                q8_gemv_max_tok                     the route boundary
constexpr uint32_t Q8_GEMM_CANDIDATES = 10;
constexpr uint32_t Q8_BLOCK_ELEMENTS = 32;    // Q8_0 wire-format block width
// {ROWS, TOKENS, NSG}. MUST match the IMPARO_Q8_GEMM list in imparo.metal: the host sizes
// the grid and the threadgroup from this table, so a disagreement is a wrong-size launch,
// not a slow one. SGR lives only in the kernel template.
constexpr uint32_t Q8_GEMM_SHAPES[Q8_GEMM_CANDIDATES][3] = {
    {32, 16,  4}, {32, 32,  4}, {64, 16,  4},
    {64, 32,  4}, {64, 32,  8}, {64, 32, 16},
    {64, 64,  8}, {64, 64,  4},
    {32,128,  4}, {128,32,  4},
};
uint32_t g_q8_decode_sgs       = 4;
uint32_t g_q8_decode_rows_log2 = 1;   // two output rows per cooperating threadgroup
uint32_t g_q8_batch_sgs        = 8;
uint32_t g_q8_token_tile_log2  = 2;   // four tokens per narrow-batch tile
uint32_t g_q8_gemm_shape       = 3;   // 64 rows x 32 tokens x 4 SG, the fork's shape
// A second prefill geometry may take over at a tuned token boundary. Zero disables the
// route, so an untuned build is exactly the single-shape path. The large shape starts
// EQUAL to the small one: neither one Apple GPU nor one model's long-prompt winner is a
// portable default.
uint32_t g_q8_gemm_large_shape   = 3;
uint32_t g_q8_gemm_large_min_tok = 0;
uint32_t g_q8_full_tiles         = 1;
// The fork routes a Q8 matrix multiply above eight columns; below it the token-tile
// kernel reuses each weight byte across the tile.
uint32_t g_q8_gemv_max_tok     = 8;
uint32_t g_q8_all              = 0;   // build every candidate, for sweeping only

// NOT KNOBS, and the reason is pipeline count. imparo.metal declares two more function
// constants for this family -- Q8_GRID_TOKEN_X (threadgroup enumeration order) and
// Q8_TYPED_SCALE (a typed two-byte scale load) -- each guarded by
// is_function_constant_defined with a false default, so leaving them unset costs nothing
// and compiles ONE pipeline per (shape, full, half). Turning either into a knob doubles
// the built pipelines, and the LFM branch that introduced them tuned both end to end,
// which this registry does not do. Adding them later is a host-side change only.

// Submission profile. Wall time minus GPU-busy time is the part of a forward pass the GPU
// spends idle -- waiting on a commit, a completion, or the host between encoders. Counting
// it separately from kernel time says whether the next win is a faster kernel or a fuller
// pipe, which guessing cannot.
uint32_t g_prof = 0;
double   g_prof_gpu_s = 0.0;    // summed GPUEndTime - GPUStartTime
double   g_prof_wall_s = 0.0;   // summed commit -> waitUntilCompleted returns
uint64_t g_prof_cbs = 0;        // command buffers
uint32_t g_nr0_log2 = 0;        // decode output rows per thread, as log2
// Build every nr0 variant without selecting one, for sweeping. Separate from `g_nr0_log2`
// because IMPARO_NR0 used to mean both: the tuner set it to 8 to get the pipelines built
// and thereby ran every one of its own measurements at nr0=8, the value this model measures
// WORST (nr0=1 39.4 tok/s, nr0=2 39.3, nr0=4 38.2). Its "compiled defaults" baseline was a
// configuration the engine would never choose.
uint32_t g_nr0_all  = 0;
uint64_t g_prof_disp = 0;       // dispatchThreadgroups calls

// Per-dispatch GPU timestamps, tagged by kernel category.
//
// GPU-busy time is already 100% of prefill wall time, so the remaining question is not
// where the pipe stalls but which kernels own the busy time. Diffing "build a variant with
// category X removed" answers that for one category per build and perturbs the cache; the
// hardware timestamp counter answers it for all categories in one run.
enum ProfCat {
    PC_MATMAT_PREFILL = 0, PC_MATMAT_DECODE, PC_ROW, PC_RMSNORM, PC_ROPE,
    PC_KV_STORE, PC_ATTENTION, PC_ELEMENTWISE, PC_MUL_STRIDED, PC_PLE, PC_N
};
static const char * PROF_CAT_NAME[PC_N] = {
    "matmat_prefill", "matmat_decode", "row", "rms_norm", "rope",
    "kv_store", "attention", "elementwise", "mul_strided", "ple_combine"
};
constexpr uint32_t PROF_MAX_SAMPLES = 16384;    // 8192 dispatches
id<MTLCounterSampleBuffer> g_prof_sbuf = nil;
uint8_t  g_prof_cat[PROF_MAX_SAMPLES / 2];
uint32_t g_prof_pairs = 0;                       // pairs written this command buffer
double   g_prof_cat_ticks[PC_N];                 // resolved, accumulated across buffers
uint64_t g_prof_cat_calls[PC_N];
uint8_t  g_prof_cur = 0;



// Grouped score-tile kernels, in the order the [3] pipeline arrays index them:
// [0]=hq2 [1]=hq4 [2]=hq8. One table, four flavours built from it (plain, quant,
// identity-placement, quant+identity-placement).
static NSString * const kGqaScoretileNames[3] = {
    @"imparo_attention_decode_scoretile_gqa_hq2",
    @"imparo_attention_decode_scoretile_gqa_hq4",
    @"imparo_attention_decode_scoretile_gqa_hq8",
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> weights = nil;
    id<MTLBuffer> bufs[B_COUNT] = {nil};
    uint8_t in_arena[B_COUNT] = {0};   // placed in the shared arena, not owned
    NSUInteger buf_off[B_COUNT] = {0}; // byte offset into bufs[id] (nonzero only for arena ids)
    id<MTLBuffer> arena_buf = nil;     // THE one resource spanning the whole arena
    size_t sizes[B_COUNT] = {0};
    std::vector<id<MTLBuffer>> kv_k, kv_v;
    // Byte offset of the LIVE REGION inside each layer's cache. A windowed layer holds
    // one ring per resident conversation; binding at the region's base keeps every
    // kernel addressing 0..ring-1, so the ring rule, the wrap test and the dequant
    // scratch are all unchanged by regions. Zero for pooled layers, which place through
    // the block table instead.
    std::vector<uint64_t> kv_reg_k, kv_reg_v;
    std::vector<id<MTLBuffer>> kv_pt;
    std::vector<uint8_t>       kv_pt_ident;  // 1 = table is the identity mapping
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    // [lane-count log2 (2..5)][NR0 log2 (0..3), so NR0 in 1,2,4,8]
    id<MTLComputePipelineState> p_q4mm_lanes[6][4];
    id<MTLComputePipelineState> p_ple_gather, p_attn_dec_scoretile, p_attn_dec_combine, p_cvt_f16;
    id<MTLComputePipelineState> p_attn_pre_qtile16;
    // Indexed by group size: [0]=hq2 [1]=hq4 [2]=hq8. The kernel is a template on HQ so
    // its accumulators are sized exactly; see the note there.
    id<MTLComputePipelineState> p_attn_dec_scoretile_gqa[3];
    // Identity-placement flavours of the same three. Their absence was a silent handicap:
    // the ungrouped path picks p_attn_dec_scoretile_id whenever the KV layout is
    // contiguous, so every grouped-vs-ungrouped A/B raced ungrouped WITH that
    // optimization against grouped WITHOUT it.
    id<MTLComputePipelineState> p_attn_dec_scoretile_gqa_id[3];
    id<MTLComputePipelineState> p_attn_pre_qtile16h;
    id<MTLComputePipelineState> p_attn_pre_qtile16h2, p_attn_pre_qtile16h8;
    id<MTLComputePipelineState> p_attn_pre_qcomb256, p_attn_pre_qcomb256x;
    id<MTLComputePipelineState> p_attn_pre_qcomb512, p_attn_pre_qcomb512x;
    id<MTLComputePipelineState> p_attn_pre_qcomb512b2, p_attn_pre_qcomb256b2;
    id<MTLComputePipelineState> p_kvstore_q4, p_kvstore_q8, p_kvstore_rt, p_kv_dq;
    id<MTLComputePipelineState> p_hadamard;
    // Decode attention variants specialised (via function constants) for quantized KV.
    id<MTLComputePipelineState> p_attn_dec_direct_q, p_attn_dec_scoretile_q;
    id<MTLComputePipelineState> p_attn_dec_scoretile_gqa_q[3];
    id<MTLComputePipelineState> p_attn_dec_scoretile_gqa_q_id[3];
    // Identity-placement variants (KV_PAGED = false): the page-table load is
    // compiled out. Selected per dispatch when the layer's table is identity.
    id<MTLComputePipelineState> p_attn_dec_direct_id, p_attn_dec_direct_q_id, p_attn_dec_scoretile_id,
        p_attn_dec_scoretile_q_id, p_kvstore_id, p_kvstore_q4_id, p_kvstore_q8_id,
        p_kvstore_rt_id;
    // Fused one-pass streaming decode attention v7, templated per head size
    // (index: dk128 -> 0, dk256 -> 1, dk512 -> 2), x quant x identity-placement.
    id<MTLComputePipelineState> p_attn_dec_stream[3], p_attn_dec_stream_q[3],
        p_attn_dec_stream_id[3], p_attn_dec_stream_q_id[3];
    // GQA row sharing: two query heads per threadgroup (hd-512), x quant x
    // identity-placement, same four flavors as the HQ=1 stream pipelines.
    id<MTLComputePipelineState> p_attn_dec_stream_g2, p_attn_dec_stream_g2_id,
        p_attn_dec_stream_g2_q, p_attn_dec_stream_g2_q_id;
    id<MTLComputePipelineState> p_attn_pre_qtile, p_mma_peak, p_mma_loaded, p_mma_dev_a;
    id<MTLComputePipelineState> p_scoremix;
    id<MTLComputePipelineState> p_spill[8];   // the NACC ladder
    id<MTLComputePipelineState> p_bw_read;
    id<MTLComputePipelineState> p_shortconv, p_shortconv_state;
    std::vector<id<MTLCommandBuffer>> pending;   // flushed, awaiting accounting
    void * pool = nullptr;                      // autorelease pool for one forward pass
    id<MTLComputePipelineState> p_gemv_probe;
    id<MTLComputePipelineState> p_add4, p_copy4, p_actmul4, p_act4, p_scale4;
    id<MTLComputePipelineState> p_addscale4, p_addscale;
    id<MTLComputePipelineState> p_rt[10];  // must be >= RT_CANDIDATES
    id<MTLComputePipelineState> p_rt_nb8, p_rt_nb8_h;     // narrow-N tile, variant a
    id<MTLComputePipelineState> p_rt_nb8b, p_rt_nb8b_h;   // variant b, one simdgroup
    id<MTLComputePipelineState> p_rt_h[10];// half-activation variants (IMPARO_HALF_A)
    // Q8_0 weights: decode rows {1,2,4}, narrow token tile {1,2,4,8}, prefill shapes.
    // The _h (half activation) entry points exist in imparo.metal but no pipeline is
    // built for them: reading a half activation needs the B_XH mirror bookkeeping that
    // the Q4_0 rt path owns (g_xh_src / g_xh_elems), and hooking a second GEMM family
    // into it is a separate change with its own measurement.
    id<MTLComputePipelineState> p_q8mv_rows[3];
    id<MTLComputePipelineState> p_q8mm_tile[4];
    id<MTLComputePipelineState> p_q8gemm[Q8_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_q8gemm_full[Q8_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_q8row;
    // Batched embedding gather, one per weight kind (index = the wire value).
    id<MTLComputePipelineState> p_gather[3];
    id<MTLComputePipelineState> p_q4mm, p_q4mm_pre, p_q4mm_pre_nomma, p_f32mm, p_q4row, p_rms, p_rope, p_kvstore,
        p_attn_dec_direct, p_actmul, p_act, p_add, p_mul, p_scale, p_copy, p_softcap, p_argmax,
        p_ple;
};

Context g;

static void prof_begin(uint8_t cat) {
    // Count first: dispatch counts are exact and need no hardware support, while the
    // timestamps below need a counter-sampling capability this device does not report.
    // Counting only when sampling works would have made the counts silently unavailable.
    g_prof_cat_calls[cat] += 1;
    if (!g_prof_sbuf || g_prof_pairs * 2 + 1 >= PROF_MAX_SAMPLES) { return; }
    g_prof_cur = cat;
    [g.enc sampleCountersInBuffer:g_prof_sbuf atSampleIndex:g_prof_pairs * 2 withBarrier:YES];
}

static void prof_end(void) {
    if (!g_prof_sbuf || g_prof_pairs * 2 + 1 >= PROF_MAX_SAMPLES) { return; }
    [g.enc sampleCountersInBuffer:g_prof_sbuf atSampleIndex:g_prof_pairs * 2 + 1 withBarrier:YES];
    g_prof_cat[g_prof_pairs] = g_prof_cur;
    g_prof_pairs += 1;
}

// Resolve after the command buffer completes; the samples are only valid then.
static void prof_resolve(void) {
    if (!g_prof_sbuf || g_prof_pairs == 0) { g_prof_pairs = 0; return; }
    NSData * d = [g_prof_sbuf resolveCounterRange:NSMakeRange(0, g_prof_pairs * 2)];
    if (d != nil) {
        const MTLCounterResultTimestamp * t = (const MTLCounterResultTimestamp *)[d bytes];
        const NSUInteger have = [d length] / sizeof(MTLCounterResultTimestamp);
        for (uint32_t i = 0; i < g_prof_pairs && (i * 2 + 1) < have; ++i) {
            const uint64_t a = t[i * 2].timestamp, b = t[i * 2 + 1].timestamp;
            // A sample the hardware could not take comes back as the "invalid" pattern;
            // counting it would silently inflate a category.
            if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) { continue; }
            g_prof_cat_ticks[g_prof_cat[i]] += (double)(b - a);
        }
    }
    g_prof_pairs = 0;
}

// Every pipeline is specialised with the process's epilogue activation (constant 11).
// Stamped in the BUILDERS rather than at each call site: a kernel that uses the constant
// and was built without it silently gets GELU, and the wrong activation produces
// plausible numbers, not a crash.
static void stamp_epi_act(MTLFunctionConstantValues * cv) {
    [cv setConstantValue:&g_epi_act type:MTLDataTypeUInt atIndex:11];
}

id<MTLComputePipelineState> make(id<MTLLibrary> lib, NSString * name) {
    NSError * e = nil;
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    stamp_epi_act(cv);
    id<MTLFunction> f = [lib newFunctionWithName:name constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: function %@ not found: %@", name, e); return nil; }
    id<MTLComputePipelineState> p =
        [g.device newComputePipelineStateWithFunction:f error:&e];
    if (p == nil) { NSLog(@"imparo metal: pipeline %@ failed: %@", name, e); }
    return p;
}

// Declare one dispatch's buffer accesses. Under IMPARO_CONCURRENT, emits a barrier before
// the dispatch when it conflicts with the unbarriered window (RAW, WAW or WAR), then adds
// it to the window. A no-op when concurrency is off. Must run BEFORE the dispatch call of
// the op it describes; encoder state setting on either side is fine.
inline void haz(uint64_t reads, uint64_t writes) {
    // The XH conversion cache dies when anything writes its source buffer; haz() runs at
    // every encode site with exact masks, which is exactly the tracking the cache needs.
    if (g_xh_src < 62u && (writes & (1ull << g_xh_src))) { g_xh_src = 0xffffffffu; }
    if (!g_concurrent) { return; }
    if ((reads & g_haz_writes) || (writes & g_haz_writes) || (writes & g_haz_reads)) {
        [g.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        g_haz_reads = 0; g_haz_writes = 0;
    }
    g_haz_reads |= reads; g_haz_writes |= writes;
}

inline uint64_t hb(uint32_t buf_id) { return 1ull << buf_id; }

void dispatch1(id<MTLComputePipelineState> p, NSUInteger n) {
    [g.enc setComputePipelineState:p];
    NSUInteger tw = p.maxTotalThreadsPerThreadgroup;
    if (tw > 256) { tw = 256; }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
    [g.enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
}

}  // namespace

extern "C" void imparo_metal_tune(uint32_t sgs, uint32_t rows) {
    if (sgs >= 1 && sgs <= 32) { g_sgs = sgs; }
    if (rows >= 1 && rows <= 8) { g_rows = rows; }
}


extern "C" void imparo_metal_set_skip_mma(uint32_t on) { g_skip_mma = on; }
extern "C" void imparo_metal_set_half_a(uint32_t on) { g_half_a = on; }
// Which region of a windowed layer's cache is live: the resident conversation's own ring.
// Bytes, not slots, because the row stride is the host's to know.
extern "C" void imparo_metal_set_kv_region(uint32_t layer, uint64_t k_off, uint64_t v_off) {
    if (layer < g.kv_reg_k.size()) { g.kv_reg_k[layer] = k_off; }
    if (layer < g.kv_reg_v.size()) { g.kv_reg_v[layer] = v_off; }
}
extern "C" void imparo_metal_kv_types(uint32_t * k, uint32_t * v) {
    if (k) { *k = g_kv_type_k; }
    if (v) { *v = g_kv_type_v; }
}

extern "C" void imparo_metal_set_kv_types(uint32_t k, uint32_t v) {
    g_kv_type_k = k; g_kv_type_v = v;
}

extern "C" void imparo_metal_set_concurrent(uint32_t on) { g_concurrent = on; }

extern "C" void imparo_metal_set_skip_attn(uint32_t on) { g_skip_attn = on; }

extern "C" void imparo_metal_set_skip_cat(uint32_t cat) { g_skip_cat = cat; }

extern "C" void imparo_metal_set_epilogue(uint32_t on) { g_epilogue = on; }

// WHICH activation the fused epilogue applies, for this process. Read at pipeline BUILD
// (function constant 11), so it must be set before imparo_metal_init; setting it after
// is a no-op and would silently leave GELU. Default 1 = EPI_GELU, which is the pipeline
// set that existed before this constant did.
extern "C" void imparo_metal_set_epilogue_act(uint32_t kind) { g_epi_act = kind; }

// Returns TFLOPS achieved by back-to-back simdgroup multiply-accumulates.
// Same loop, but the threadgroup allocation is padded to `smem` bytes. The operands still
// come from the first 2 KB; the rest exists only to occupy the allocation, so the only
// thing that changes is how many threadgroups stay resident per core.
extern "C" double imparo_metal_mma_loaded_smem(uint32_t tgs, uint32_t sgs, uint32_t iters,
                                               uint32_t smem) {
    if (g.p_mma_loaded == nil) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                              options:MTLResourceStorageModeShared];
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_mma_loaded];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e setThreadgroupMemoryLength:smem atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double s = CACurrentMediaTime() - t0;
        const double flops = (double)tgs * sgs * iters * 8.0 * 1024.0;
        if (pass == 1) { best = flops / s / 1e12; }
    }
    return best;
}

extern "C" double imparo_metal_mma_device_a(uint32_t tgs, uint32_t sgs, uint32_t iters,
                                            uint32_t stride) {
    if (g.p_mma_dev_a == nil) { return 0.0; }
    // 32 MB, comfortably past any cache, so the walk above actually misses
    const NSUInteger bytes = (NSUInteger)32 << 20;
    id<MTLBuffer> src = [g.device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                              options:MTLResourceStorageModeShared];
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_mma_dev_a];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e setBuffer:src offset:0 atIndex:2];
        [e setBytes:&stride length:4 atIndex:3];
        [e setThreadgroupMemoryLength:64 * 8 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double s = CACurrentMediaTime() - t0;
        const double flops = (double)tgs * sgs * iters * 8.0 * 1024.0;
        if (pass == 1) { best = flops / s / 1e12; }
    }
    return best;
}

// COMMAND-BUFFER COST, host side, and the reason it is measured rather than guessed:
// `flush_layers` decides how many layers are encoded before a command buffer is committed,
// and that is a pure trade between two costs neither of which any API reports.
//
//   commit too often   pay the fixed per-command-buffer cost on every batch
//   commit too rarely  the GPU sits idle while the CPU is still encoding
//
// Returns microseconds per SUBMISSION of an empty command buffer -- created and committed,
// NOT waited on.
//
// Waiting would measure the wrong thing, and which one is right follows from what the
// engine does: `imparo_metal_flush` commits and returns, so a mid-graph flush costs the
// host a submission and nothing more. A commit+wait measurement is dominated by the GPU
// round trip (28.9 us here against a submission's fraction of that), and using it would
// price a flush at what a synchronisation costs.
//
// The last buffer is waited on so the probe leaves nothing in flight; that one wait is
// outside the timed region.
extern "C" double imparo_metal_commit_overhead(uint32_t n) {
    if (g.queue == nil || n == 0) { return 0.0; }
    id<MTLCommandBuffer> last = nil;
    for (uint32_t i = 0; i < 4; ++i) {              // warm the queue, discarded
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        [cb commit];
        last = cb;
    }
    [last waitUntilCompleted];
    const double t0 = CACurrentMediaTime();
    for (uint32_t i = 0; i < n; ++i) {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        [cb commit];
        last = cb;
    }
    const double us = (CACurrentMediaTime() - t0) * 1e6 / (double)n;
    [last waitUntilCompleted];                      // outside the clock
    return us;
}

// The OTHER cost, and a different quantity: a full commit AND wait. This is what the end
// of a graph pays, where the host has to see the result before it can go on.
extern "C" double imparo_metal_sync_overhead(uint32_t n) {
    if (g.queue == nil || n == 0) { return 0.0; }
    for (uint32_t i = 0; i < 4; ++i) {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }
    const double t0 = CACurrentMediaTime();
    for (uint32_t i = 0; i < n; ++i) {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return (CACurrentMediaTime() - t0) * 1e6 / (double)n;
}

// The OTHER half of that trade: what the CPU pays to encode one dispatch. Encodes n
// dispatches into a single command buffer and never commits it, so the number is encode
// cost with no execution and no commit in it.
extern "C" double imparo_metal_encode_cost(uint32_t n) {
    if (g.queue == nil || g.p_mma_peak == nil || n == 0) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    const uint32_t iters = 1;
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {          // pass 0 warms, pass 1 counts
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const double t0 = CACurrentMediaTime();
        for (uint32_t i = 0; i < n; ++i) {
            [e setComputePipelineState:g.p_mma_peak];
            [e setBuffer:out offset:0 atIndex:0];
            [e setBytes:&iters length:4 atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        const double us = (CACurrentMediaTime() - t0) * 1e6 / (double)n;
        [e endEncoding];
        // Committed and waited so the buffer is not left live; the clock stopped above.
        [cb commit];
        [cb waitUntilCompleted];
        if (pass == 1) { best = us; }
    }
    return best;
}

// The score phase's OWN ceiling: same operand mix as the prefill attention score loop
// (staged Q re-read from threadgroup, K streamed from device, 4 MACs per 4 loads).
// Compare the score phase against THIS, not against imparo_metal_mma_peak -- a peak
// measured on a different access pattern is a target that kernel cannot reach.
extern "C" double imparo_metal_scoremix_rate(uint32_t tgs, uint32_t sgs, uint32_t iters,
                                             uint32_t stride, uint32_t kspan) {
    if (g.p_scoremix == nil) { return 0.0; }
    // 32 MB of K, comfortably past the cache knee, so the device-side stream really misses.
    const NSUInteger bytes = (NSUInteger)32 << 20;
    id<MTLBuffer> kbuf = [g.device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                              options:MTLResourceStorageModeShared];
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_scoremix];
        [e setBuffer:kbuf offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e setBytes:&iters length:4 atIndex:2];
        [e setBytes:&stride length:4 atIndex:3];
        [e setBytes:&kspan length:4 atIndex:4];
        [e setThreadgroupMemoryLength:16 * 64 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double s = CACurrentMediaTime() - t0;
        // 4 MACs per iteration, each an 8x8x8 multiply-accumulate = 1024 flops.
        const double flops = (double)tgs * sgs * iters * 4.0 * 1024.0;
        if (pass == 1) { best = flops / s / 1e12; }
    }
    return best;
}

extern "C" double imparo_metal_mma_loaded(uint32_t tgs, uint32_t sgs, uint32_t iters) {
    if (g.p_mma_loaded == nil) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                              options:MTLResourceStorageModeShared];
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_mma_loaded];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e setThreadgroupMemoryLength:64 * 8 * sizeof(float) atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double s = CACurrentMediaTime() - t0;
        const double flops = (double)tgs * sgs * iters * 8.0 * 1024.0;
        if (pass == 1) { best = flops / s / 1e12; }
    }
    return best;
}

extern "C" double imparo_metal_gemv_probe(uint32_t n_in, uint32_t n_out, uint32_t lanes,
                                          uint32_t sgs, uint32_t mode, uint32_t iters,
                                          uint32_t split) {
    if (g.p_gemv_probe == nil || g.weights == nil) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:n_out * 4
                                             options:MTLResourceStorageModeShared];
    if (out == nil) { return 0.0; }
    const NSUInteger rows_per_tg = sgs * (32u / lanes);
    const NSUInteger tgs = (n_out + rows_per_tg - 1) / rows_per_tg;
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (uint32_t it = 0; it < iters; ++it) {
            [e setComputePipelineState:g.p_gemv_probe];
            [e setBuffer:g.weights offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&n_in length:4 atIndex:2];
            [e setBytes:&n_out length:4 atIndex:3];
            [e setBytes:&lanes length:4 atIndex:4];
            [e setBytes:&mode length:4 atIndex:5];
            [e setBytes:&split length:4 atIndex:6];
            [e dispatchThreadgroups:MTLSizeMake(tgs, split, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
        }
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double s = CACurrentMediaTime() - t0;
        const double per_block = (mode == 0u) ? 18.0 : 16.0;
        const double bytes = (double)n_out * (n_in / 32.0) * per_block * iters;
        if (pass == 1) { best = bytes / s / 1e9; }
    }
    return best;
}

// Read bandwidth at a given working set. This is a DISCOVERY probe -- the number it
// returns feeds the re-read arithmetic that decides whether a duplicated operand is
// cache-absorbed or DRAM-priced -- so it has to be right, and the first version was not.
// It reported 33-41 GB/s where this machine does 121, moved its own cache knee between
// runs minutes apart, and returned rates that did not order with grid size. Four causes,
// all fixed here:
//
//   1. It timed commandBuffer creation, encoding, commit and waitUntilCompleted with a
//      wall clock. That is submission latency, which is variable and comparable to the
//      whole measurement at small working sets. Now timed with the command buffer's own
//      GPUEndTime - GPUStartTime, which covers exactly the dispatch.
//   2. It allocated a fresh private buffer PER CALL, so a sweep allocated and freed
//      hundreds of MB and paid first-touch faulting inside the measurement. The buffer
//      is now allocated once at the high-water mark and reused; a smaller working set
//      reads a prefix of it.
//   3. Two passes cannot show variance and it returned a number regardless. Now five
//      timed passes after a warm-up, returning the MEDIAN.
//   4. No way for a caller to know the reading was unstable. `spread_out`, when not
//      null, receives (max-min)/median so the caller can refuse a wide one.
static id<MTLBuffer> g_bw_buf = nil;
static uint64_t      g_bw_cap = 0;

extern "C" double imparo_metal_bw_read_v(uint64_t bytes, uint32_t reps,
                                         uint32_t tgs, uint32_t tpg,
                                         double * spread_out) {
    if (spread_out) { *spread_out = 0.0; }
    if (g.p_bw_read == nil || bytes < 16) { return 0.0; }
    if (g_bw_buf == nil || g_bw_cap < bytes) {
        g_bw_buf = [g.device newBufferWithLength:bytes
                                         options:MTLResourceStorageModePrivate];
        if (g_bw_buf == nil) { g_bw_cap = 0; return 0.0; }
        g_bw_cap = bytes;
    }
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                             options:MTLResourceStorageModeShared];
    if (out == nil) { return 0.0; }
    const uint32_t n4 = (uint32_t)(bytes / 16ull);
    double r[6];
    for (int pass = 0; pass < 6; ++pass) {           // pass 0 is warm-up, discarded
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_bw_read];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBuffer:g_bw_buf offset:0 atIndex:1];
        [e setBytes:&n4 length:4 atIndex:2];
        [e setBytes:&reps length:4 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double gpu = [cb GPUEndTime] - [cb GPUStartTime];
        r[pass] = gpu > 0.0 ? (double)bytes * reps / gpu / 1e9 : 0.0;
    }
    double v[5];
    for (int i = 0; i < 5; ++i) { v[i] = r[i + 1]; }
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) { if (v[j] < v[i]) { double t = v[i]; v[i] = v[j]; v[j] = t; } }
    }
    if (spread_out && v[2] > 0.0) { *spread_out = (v[4] - v[0]) / v[2]; }
    return v[2];
}

extern "C" double imparo_metal_bw_read(uint64_t bytes, uint32_t reps,
                                       uint32_t tgs, uint32_t tpg) {
    return imparo_metal_bw_read_v(bytes, reps, tgs, tpg, nullptr);
}

// TFLOPS at a given live-accumulator count. Sweeping this finds the SPILL CLIFF: the
// rate holds while the accumulators fit in registers and collapses once the compiler
// starts spilling them to device memory. That cliff is the register budget, which no
// Metal API reports.
extern "C" double imparo_metal_spill_rate(uint32_t idx, uint32_t tgs, uint32_t tpg,
                                          uint32_t iters) {
    static const uint32_t NACC[8] = {4, 8, 12, 16, 24, 32, 48, 64};
    if (idx >= 8 || g.p_spill[idx] == nil) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                              options:MTLResourceStorageModeShared];
    if (out == nil) { return 0.0; }
    double best = 0.0;
    for (int pass = 0; pass < 3; ++pass) {   // two warm-ups, then the measured run
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_spill[idx]];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double gpu = [cb GPUEndTime] - [cb GPUStartTime];
        const double sgs = (double)tpg / 32.0;
        const double flops = (double)tgs * sgs * iters * (double)NACC[idx] * 1024.0;
        if (pass == 2 && gpu > 0.0) { best = flops / gpu / 1e12; }
    }
    return best;
}

extern "C" double imparo_metal_mma_peak(uint32_t tgs, uint32_t sgs, uint32_t iters) {
    if (g.p_mma_peak == nil) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:tgs * 4
                                              options:MTLResourceStorageModeShared];
    // one untimed warm-up, then the measured run
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        const double t0 = CACurrentMediaTime();
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_mma_peak];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(sgs * 32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const double s = CACurrentMediaTime() - t0;
        // 8 multiply-accumulates of 8x8x8 per iteration per simdgroup = 8192 FLOP
        const double flops = (double)tgs * sgs * iters * 8.0 * 1024.0;
        if (pass == 1) { best = flops / s / 1e12; }
    }
    return best;
}

extern "C" void imparo_metal_set_nr0(uint32_t nr0) {
    uint32_t lg = 0;
    while ((1u << lg) < nr0 && lg < 3u) { lg += 1u; }
    g_nr0_log2 = lg;
}

extern "C" void imparo_metal_set_qtile(uint32_t on) { g_qtile = on; }
extern "C" void imparo_metal_set_qcomb(uint32_t on) { g_qcomb = on; }
extern "C" void imparo_metal_set_kvq_mask(uint32_t mk, uint32_t mv, uint32_t ty) {
    g_kvq_mask_k = mk; g_kvq_mask_v = mv; g_kvq_type = ty;
}
extern "C" void imparo_metal_set_kvq_rt(uint32_t ty, uint32_t lo, uint32_t hi) {
    g_kvq_rt = ty; g_kvq_rt_lo = lo; g_kvq_rt_hi = hi;
}
extern "C" void imparo_metal_set_nb8_max(uint32_t n) { g_nb8_max = n; }
extern "C" void imparo_metal_set_gemv_max_tok(uint32_t n) { g_gemv_max_tok = n < 1u ? 1u : n; }
extern "C" uint32_t imparo_metal_gemv_max_tok_current(void) { return g_gemv_max_tok; }
extern "C" void imparo_metal_set_nb8_shape(uint32_t v) { g_nb8_shape = v & 1u; }
extern "C" uint32_t imparo_metal_nb8_max_current(void) { return g_nb8_max; }
extern "C" uint32_t imparo_metal_nb8_shape_current(void) { return g_nb8_shape; }
extern "C" void imparo_metal_set_attn_short(uint32_t on) { g_attn_short = on; }
extern "C" void imparo_metal_set_attn_blk(uint32_t v) { g_attn_blk = (v == 2u || v == 8u) ? v : 4u; }
extern "C" void imparo_metal_set_qcomb_blk(uint32_t v) { g_qcomb_blk = (v == 2u) ? 2u : 4u; }
extern "C" uint32_t imparo_metal_qcomb_blk_current(void) { return g_qcomb_blk; }
extern "C" uint32_t imparo_metal_attn_blk_current(void) { return g_attn_blk; }
extern "C" void imparo_metal_set_attn_stage(uint32_t v) { g_attn_stage = v < 1u ? 1u : (v > 7u ? 7u : v); }
extern "C" void imparo_metal_set_attn_gqa(uint32_t on) { g_attn_gqa = on; }

extern "C" void imparo_metal_set_rt(uint32_t on) { g_rt = on; }

extern "C" void imparo_metal_set_shrink(uint32_t on) { g_shrink = on; }

// Must be set before init: pipelines are built there.
extern "C" void imparo_metal_set_rt_all(uint32_t on) { g_rt_all = on; }

// ---- Q8_0 knob setters -------------------------------------------------------------
// Every one clamps to a value a pipeline exists for. The registry applies knobs BEFORE
// init as well as after, so a setter may run with no pipelines built: these only touch
// globals, and the pipeline selection reads them at dispatch.
extern "C" void imparo_metal_set_q8_decode_sgs(uint32_t sgs) {
    if (sgs >= 1u && sgs <= 32u) { g_q8_decode_sgs = sgs; }
}
extern "C" uint32_t imparo_metal_q8_decode_sgs(void) { return g_q8_decode_sgs; }
// ROWS, not log2, on the wire: the knob's values are 1/2/4 so the config reads as the
// thing the kernel does. Anything else is ignored, since only three pipelines exist.
extern "C" void imparo_metal_set_q8_decode_rows(uint32_t rows) {
    if (rows == 1u) { g_q8_decode_rows_log2 = 0u; }
    else if (rows == 2u) { g_q8_decode_rows_log2 = 1u; }
    else if (rows == 4u) { g_q8_decode_rows_log2 = 2u; }
}
extern "C" uint32_t imparo_metal_q8_decode_rows(void) {
    return 1u << g_q8_decode_rows_log2;
}
extern "C" void imparo_metal_set_q8_batch_sgs(uint32_t sgs) {
    if (sgs >= 1u && sgs <= 32u) { g_q8_batch_sgs = sgs; }
}
extern "C" uint32_t imparo_metal_q8_batch_sgs(void) { return g_q8_batch_sgs; }
extern "C" void imparo_metal_set_q8_token_tile(uint32_t tile) {
    if (tile == 1u) { g_q8_token_tile_log2 = 0u; }
    else if (tile == 2u) { g_q8_token_tile_log2 = 1u; }
    else if (tile == 4u) { g_q8_token_tile_log2 = 2u; }
    else if (tile == 8u) { g_q8_token_tile_log2 = 3u; }
}
extern "C" uint32_t imparo_metal_q8_token_tile(void) {
    return 1u << g_q8_token_tile_log2;
}
extern "C" void imparo_metal_set_q8_gemm_shape(uint32_t shape) {
    if (shape < Q8_GEMM_CANDIDATES) { g_q8_gemm_shape = shape; }
}
extern "C" uint32_t imparo_metal_q8_gemm_shape(void) { return g_q8_gemm_shape; }
extern "C" void imparo_metal_set_q8_gemm_large_shape(uint32_t shape) {
    if (shape < Q8_GEMM_CANDIDATES) { g_q8_gemm_large_shape = shape; }
}
extern "C" uint32_t imparo_metal_q8_gemm_large_shape(void) {
    return g_q8_gemm_large_shape;
}
extern "C" void imparo_metal_set_q8_gemm_large_min_tok(uint32_t n) {
    g_q8_gemm_large_min_tok = n;
}
extern "C" uint32_t imparo_metal_q8_gemm_large_min_tok(void) {
    return g_q8_gemm_large_min_tok;
}
extern "C" void imparo_metal_set_q8_full_tiles(uint32_t on) { g_q8_full_tiles = on; }
extern "C" uint32_t imparo_metal_q8_full_tiles(void) { return g_q8_full_tiles; }
extern "C" void imparo_metal_set_q8_gemv_max_tok(uint32_t n) {
    g_q8_gemv_max_tok = n < 1u ? 1u : n;
}
extern "C" uint32_t imparo_metal_q8_gemv_max_tok(void) { return g_q8_gemv_max_tok; }
extern "C" void imparo_metal_set_q8_all(uint32_t on) { g_q8_all = on; }
// Must be set BEFORE init: it decides which prefill pipelines are compiled.
extern "C" void imparo_metal_set_attn_live_mask(uint32_t on) { g_attn_live_mask = on; }
extern "C" uint32_t imparo_metal_attn_live_mask(void) { return g_attn_live_mask; }
extern "C" uint32_t imparo_metal_q8_gemm_shapes(void) { return Q8_GEMM_CANDIDATES; }

extern "C" void imparo_metal_set_rt_shape_pre(uint32_t i) {
    if (i < RT_CANDIDATES) { g_rt_shape = i; }
}

extern "C" uint32_t imparo_metal_rt_shapes(void) { return RT_CANDIDATES; }

extern "C" uint32_t imparo_metal_buf_count(void) { return B_COUNT; }

extern "C" uint32_t imparo_metal_rt_shape_current(void) { return g_rt_shape; }

extern "C" uint32_t imparo_metal_lanes_current(void) { return g_lanes; }
extern "C" uint32_t imparo_metal_nr0_current(void) { return 1u << g_nr0_log2; }
extern "C" void imparo_metal_set_nr0_all(uint32_t on) { g_nr0_all = on; }
extern "C" uint32_t imparo_metal_sgs_current(void) { return g_sgs; }
extern "C" uint32_t imparo_metal_attn_threads_current(void) { return g_attn_threads; }
extern "C" uint32_t imparo_metal_attn_threads_pf_current(void) { return g_attn_threads_pf; }

// MEASURED, by sweeping live accumulators and finding where the rate collapses. The sweep
// lives in the TUNER (imparo-tune's discovery pass, `spill_cliff`), which is where probes
// belong -- it used to be an example binary a human had to remember to run, and that folder
// is gone. On this device:
//
//   threads   24 acc        32 acc
//     128     70.01 TFLOPS   0.78     <- a 90x collapse: that is a spill
//     256    127.33          0.76
//     512    130.53          0.76
//
// The cliff is PER THREAD and flat: 24 accumulators work at 128, 256 and 512 threads
// alike. It does not scale with thread count.
//
// The previous bound said otherwise. It was "threads x accumulators <= 2048", read off
// the shapes that already worked -- which implies the budget divides among threads, and
// therefore wrongly called shape 6 illegal (256 threads x 16 accumulators). Shape 6 is
// legal: 16 is well under 24, and it measures 1.3% slower end to end, which is a slower
// shape and not a spilling one. Guessing a formula from three working points produced a
// rule that excluded a legal candidate; measuring it took one probe kernel.
//
// Shape 7 still spills (292x on the screen) at 16 accumulators, so accumulators are not
// the whole story -- its 128-token tile costs operand registers this does not count. The
// screen catches what this cannot, which is the correct division of labour.
static constexpr uint32_t RT_MAX_ACC_PER_THREAD = 24;

// Accumulator fragments each THREAD holds live, which is what the cliff measures.
static uint32_t rt_shape_acc(uint32_t i) {
    const uint32_t toks = RT_SHAPES[i][0] * 8u * RT_SHAPES[i][3];
    const uint32_t rows = RT_SHAPES[i][1] * 8u * RT_SHAPES[i][2];
    const uint32_t thr  = RT_SHAPES[i][2] * RT_SHAPES[i][3] * 32u;
    return (rows / 8u) * (toks / 8u) / (thr / 32u);
}

// The legality rule, asked as a QUESTION rather than only enforced as a setter's
// refusal. The tuner consults this before measuring, so a shape that provably spills is
// never dispatched; set_rt_shape keeps refusing it as the last line of defence for
// anything that sets the knob directly.
extern "C" uint32_t imparo_metal_rt_shape_legal(uint32_t i) {
    if (i >= RT_CANDIDATES) { return 0u; }
    return rt_shape_acc(i) <= RT_MAX_ACC_PER_THREAD ? 1u : 0u;
}

// THE PIPELINE GUARD ONLY APPLIES AFTER INIT, and getting that wrong silently discarded
// every tuned value this knob ever had. MEASURED 2026-08-24: a host config carrying
// rt_shape=0 left the engine reporting rt_shape=1, while IMPARO_RT_SHAPE=0 (which goes
// through the _pre setter) reported 0.
//
//   Wrong: apply_host_config -> set_rt_shape -> p_rt[i] is nil pre-init -> REFUSED
//                            -> imparo_metal_init builds the DEFAULT shape
//   Right: apply_host_config -> set_rt_shape -> pre-init, so record it
//                            -> imparo_metal_init builds the shape that was recorded
//
// Post-init the guard stays, and it is not decoration: with g_rt_all off only the selected
// shape is built, so selecting another one there would hand the dispatch a nil pipeline.
// `g.device == nil` is this file's established pre-init test (see threadgroup_bytes).
extern "C" int imparo_metal_set_rt_shape(uint32_t i) {
    if (i >= RT_CANDIDATES) { return 1; }
    if (rt_shape_acc(i) > RT_MAX_ACC_PER_THREAD) { return 1; }
    if (g.device != nil && g.p_rt[i] == nil) { return 1; }
    g_rt_shape = i;
    return 0;
}

extern "C" uint64_t imparo_metal_threadgroup_bytes(void) {
    return g.device == nil ? 0 : (uint64_t)[g.device maxThreadgroupMemoryLength];
}
extern "C" uint32_t imparo_metal_max_threads_tg(void) {
    return g.device == nil ? 0 : (uint32_t)[g.device maxThreadsPerThreadgroup].width;
}

extern "C" void imparo_metal_prof_enable(uint32_t on) {
    g_prof = on;
    if (!on || g_prof_sbuf != nil) { return; }
    if (![g.device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        NSLog(@"imparo metal: no dispatch-boundary counter sampling; per-kernel profile off");
        return;
    }
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> cs in [g.device counterSets]) {
        if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) { ts = cs; }
    }
    if (ts == nil) { NSLog(@"imparo metal: no timestamp counter set"); return; }
    MTLCounterSampleBufferDescriptor * d = [MTLCounterSampleBufferDescriptor new];
    [d setCounterSet:ts];
    [d setStorageMode:MTLStorageModeShared];
    [d setSampleCount:PROF_MAX_SAMPLES];
    NSError * e = nil;
    g_prof_sbuf = [g.device newCounterSampleBufferWithDescriptor:d error:&e];
    if (g_prof_sbuf == nil) { NSLog(@"imparo metal: counter sample buffer: %@", e); }
}

extern "C" void imparo_metal_prof_cats(double * ticks, uint64_t * calls, uint32_t * n) {
    *n = PC_N;
    for (uint32_t i = 0; i < PC_N; ++i) {
        ticks[i] = g_prof_cat_ticks[i]; calls[i] = g_prof_cat_calls[i];
        g_prof_cat_ticks[i] = 0.0; g_prof_cat_calls[i] = 0;
    }
}

extern "C" const char * imparo_metal_prof_cat_name(uint32_t i) {
    return i < PC_N ? PROF_CAT_NAME[i] : "";
}

extern "C" void imparo_metal_prof_read(double * gpu_s, double * wall_s,
                                       uint64_t * cbs, uint64_t * disp) {
    *gpu_s = g_prof_gpu_s; *wall_s = g_prof_wall_s;
    *cbs = g_prof_cbs; *disp = g_prof_disp;
    g_prof_gpu_s = 0.0; g_prof_wall_s = 0.0; g_prof_cbs = 0; g_prof_disp = 0;
}

extern "C" void imparo_metal_set_lanes(uint32_t lanes) {
    if (lanes == 4 || lanes == 8 || lanes == 16 || lanes == 32) {
        g_lanes = lanes;
        g_lanes_log2 = (lanes == 4) ? 2 : (lanes == 8) ? 3 : (lanes == 16) ? 4 : 5;
    }
}

extern "C" void imparo_metal_set_attn_min_tgs(uint32_t n) { g_attn_min_tgs = n; }
extern "C" void imparo_metal_set_attn_stream(uint32_t on) { g_attn_stream = on ? 1u : 0u; }
extern "C" uint32_t imparo_metal_attn_stream_current(void) { return g_attn_stream; }
extern "C" void imparo_metal_set_attn_stream_min_pos(uint32_t v) { g_attn_stream_min_pos = v; }
extern "C" uint32_t imparo_metal_attn_stream_min_pos_current(void) { return g_attn_stream_min_pos; }
extern "C" uint32_t imparo_metal_attn_min_tgs_current(void) { return g_attn_min_tgs; }

extern "C" void imparo_metal_set_attn_threads(uint32_t n) {
    if (n >= 32 && n <= 1024 && (n % 32) == 0) { g_attn_threads = n; }
}

extern "C" void imparo_metal_set_attn_threads_prefill(uint32_t n) {
    if (n >= 32 && n <= 1024 && (n % 32) == 0) { g_attn_threads_pf = n; }
}

extern "C" int imparo_metal_init(const void * base, uint64_t len) {
    @autoreleasepool {
        // ONE MODEL PER PROCESS, and this is where that is enforced.
        //
        // `Context g` holds one device, one queue, one weight buffer, one activation
        // buffer array and one pipeline set. A second init would rebuild the pipelines
        // over the first model's live state -- and because the epilogue activation is a
        // FUNCTION CONSTANT baked in at build, the first model would then run with the
        // second model's activation. That produces plausible numbers, not a crash, which
        // is the failure mode worth refusing outright.
        //
        // Running two models at once (a speculative drafter, several served models) means
        // making Context a per-model object handed to the workflow, which backend.rs
        // already names as the shape that change takes. Until then: rc=4.
        if (g.device != nil) {
            NSLog(@"imparo metal: init called twice -- this build runs ONE model per "
                   "process (one pipeline set, specialised for one activation)");
            return 4;
        }
        g.device = MTLCreateSystemDefaultDevice();
        if (g.device == nil) { return 1; }
        g.queue = [g.device newCommandQueue];
        NSError * error = nil;
        // IMPARO_METAL_TIMING=1: where init time goes. The library is compiled from SOURCE
        // on every process start, and each pipeline specialises that source again, so this
        // is the one place a process can spend seconds before doing any work.
        const bool t_on = getenv("IMPARO_METAL_TIMING") != NULL;
        const double t_start = CACurrentMediaTime();
        // Device-derived shape values reach the kernel as preprocessor defines. The
        // library is compiled from SOURCE at init, which is what makes this possible at
        // all -- the alternative is a hand-picked matrix of instantiations frozen at
        // whatever budget the author's Mac had.
        const uint64_t tg_budget = (uint64_t)[g.device maxThreadgroupMemoryLength];
        // BLK feeds BOTH derivations -- PT rounds down to a whole work unit of 8*BLK, and
        // NSG is the work-unit count PT/(8*BLK) capped by the spill cliff -- so it has to
        // be read before either.
        //
        // MEASURED on the deep prefill attention workload, us per pass, two runs each:
        //
        //     BLK 1   36388 / 34909     too many work units for the tile
        //     BLK 2   31409 / 31687     <- shipped, and the minimum
        //     BLK 4   32148 / 32153
        //     BLK 8   52694 / 55297     +70%: PT/(8*BLK) leaves most simdgroups idle
        //
        // So it MATTERS -- a clean interior minimum with both sides turning over -- and the
        // compiled 2 is right on this device. It stays a constant rather than a knob for
        // the same reason the unroll does: it is a preprocessor define consumed when the
        // library is compiled at init, so sweeping it needs a PROCESS PER CANDIDATE, which
        // is the shape this tuner exists to remove. The env override is what makes
        // re-checking it on another machine a re-run rather than an edit.
        uint32_t blk_x = 2u;
        if (const char * bx = getenv("IMPARO_QCOMB_BLK_X")) {
            const uint32_t v = (uint32_t)atoi(bx);
            if (v == 1u || v == 2u || v == 4u || v == 8u) { blk_x = v; }
        }
        g_blk_x = blk_x;
        g_pt_512x = qcomb_derive_pt(16u, 512u, true, true,  blk_x, tg_budget);
        g_pt_256x = qcomb_derive_pt(16u, 256u, true, false, blk_x, tg_budget);
        g_nsg_512x = qcomb_derive_nsg(16u, 512u, blk_x, g_measured_max_acc, 16u);
        g_nsg_256x = qcomb_derive_nsg(16u, 256u, blk_x, g_measured_max_acc, 8u);
        // IMPARO_QCOMB_UNROLL: a shape change can invert the best unroll, and it has
        // twice. An env override makes re-checking it a re-run.
        uint32_t unroll = 64u;
        if (const char * u = getenv("IMPARO_QCOMB_UNROLL")) {
            const uint32_t v = (uint32_t)atoi(u);
            if (v == 8u || v == 16u || v == 32u || v == 64u) { unroll = v; }
        }
        MTLCompileOptions * copts = [MTLCompileOptions new];
        [copts setPreprocessorMacros:@{
            @"IMPARO_PT_512X" : [NSNumber numberWithUnsignedInt:g_pt_512x],
            @"IMPARO_PT_256X" : [NSNumber numberWithUnsignedInt:g_pt_256x],
            @"IMPARO_NSG_512X" : [NSNumber numberWithUnsignedInt:g_nsg_512x],
            @"IMPARO_NSG_256X" : [NSNumber numberWithUnsignedInt:g_nsg_256x],
            @"IMPARO_QCOMB_UNROLL" : [NSNumber numberWithUnsignedInt:unroll],
            @"IMPARO_BLK_512X" : [NSNumber numberWithUnsignedInt:blk_x],
            @"IMPARO_BLK_256X" : [NSNumber numberWithUnsignedInt:blk_x],
        }];
        if (t_on) {
            NSLog(@"imparo metal: derived from a %llu B budget and a %u-accumulator "
                  @"cliff at BLK %u -- PT 512x=%u 256x=%u, NSG 512x=%u 256x=%u",
                  tg_budget, g_measured_max_acc, g_blk_x, g_pt_512x, g_pt_256x,
                  g_nsg_512x, g_nsg_256x);
        }
        id<MTLLibrary> lib = [g.device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                                    options:copts error:&error];
        if (lib == nil) {
            NSLog(@"imparo metal library: %@", error);
            return 2;
        }
        const double t_lib = CACurrentMediaTime();
        if (t_on) { NSLog(@"imparo metal timing: library source compile %.2f s",
                          t_lib - t_start); }
        // One pipeline per (LANES_PER_ROW, NR0) candidate.
        //
        // NR0 > 1 is built ONLY when asked for. It measured worse at every size on this
        // model -- lanes=32: nr0=1 39.4 tok/s, nr0=2 39.3, nr0=4 38.2 -- so the extra
        // pipelines would be startup cost for a setting nothing selects. llama.cpp needs
        // NR0=4 because it puts just 2 lanes on each block and has to find independent
        // work somewhere; 32 lanes per row already have it.
        const uint32_t n_nr = (g_nr0_all || g_nr0_log2 > 0u) ? 4u : 1u;
        for (uint32_t lg = 2; lg <= 5; ++lg) {
            for (uint32_t ng = 0; ng < n_nr; ++ng) {
                const uint32_t lanes = 1u << lg;
                const uint32_t nr0   = 1u << ng;
                MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
                [cv setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
                [cv setConstantValue:&nr0   type:MTLDataTypeUInt atIndex:2];
                NSError * e = nil;
                id<MTLFunction> f = [lib newFunctionWithName:@"imparo_q4_0_matmat"
                                              constantValues:cv error:&e];
                if (f == nil) {
                    NSLog(@"imparo metal: q4 lanes=%u nr0=%u: %@", lanes, nr0, e); continue;
                }
                g.p_q4mm_lanes[lg][ng] = [g.device newComputePipelineStateWithFunction:f
                                                                                 error:&e];
                if (g.p_q4mm_lanes[lg][ng] == nil) {
                    NSLog(@"imparo metal: pipeline lanes=%u nr0=%u: %@", lanes, nr0, e);
                }
            }
        }
        g.p_q4mm = g.p_q4mm_lanes[3][0];   // default 8 lanes, one row per thread
        {
            MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
            const uint32_t lanes = 8; const bool skip = false;
            [cv setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
            [cv setConstantValue:&skip type:MTLDataTypeBool atIndex:1];
            NSError * e = nil;
            id<MTLFunction> f = [lib newFunctionWithName:@"imparo_q4_0_matmat_prefill"
                                          constantValues:cv error:&e];
            g.p_q4mm_pre = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
            if (g.p_q4mm_pre == nil) { NSLog(@"imparo metal: prefill pipeline: %@", e); }

            // Register-tile candidates. Names must match the IMPARO_RT_KERNEL list.
            //
            // Only the selected shape is built unless a sweep asks for all of them. A
            // compute pipeline is not free: it is GPU-resident code, and these kernels are
            // fully unrolled, so building all eight to use one is pure resident memory.
            const double t_rt0 = CACurrentMediaTime();
            uint32_t built = 0;
            for (uint32_t i = 0; i < RT_CANDIDATES; ++i) {
                if (!g_rt_all && i != g_rt_shape) { continue; }
                NSString * nm = [NSString stringWithFormat:@"imparo_rt_%u", i];
                g.p_rt[i] = make(lib, nm);
                NSString * nmh = [NSString stringWithFormat:@"imparo_rt_%u_h", i];
                g.p_rt_h[i] = make(lib, nmh);
                ++built;
            }
            g.p_rt_nb8    = make(lib, @"imparo_rt_nb8");
            g.p_rt_nb8_h  = make(lib, @"imparo_rt_nb8_h");
            g.p_rt_nb8b   = make(lib, @"imparo_rt_nb8b");
            g.p_rt_nb8b_h = make(lib, @"imparo_rt_nb8b_h");
            if (t_on) { NSLog(@"imparo metal timing: %u register-tile pipelines %.2f s",
                              built, CACurrentMediaTime() - t_rt0); }


            const bool skip2 = true;
            [cv setConstantValue:&skip2 type:MTLDataTypeBool atIndex:1];
            id<MTLFunction> f2 = [lib newFunctionWithName:@"imparo_q4_0_matmat_prefill"
                                           constantValues:cv error:&e];
            g.p_q4mm_pre_nomma = f2 ? [g.device newComputePipelineStateWithFunction:f2 error:&e] : nil;
        }
        // No prefill pipeline means no correct batched matmul: the only fallback would
        // be the multi-token GEMV, which this engine no longer exposes (task #5). Fail
        // the init loudly instead of degrading silently.
        if (g.p_q4mm_pre == nil && g.p_rt[g_rt_shape] == nil) {
            NSLog(@"imparo metal: no prefill pipeline built on this device; refusing to "
                  @"run batched matmul on the decode kernel");
            return 2;
        }
        g.p_f32mm  = make(lib, @"imparo_f32_matmat");
        g.p_q4row  = make(lib, @"imparo_q4_0_row");
        g.p_rms    = make(lib, @"imparo_rms_norm");
        g.p_rope   = make(lib, @"imparo_rope_neox");
        // The decode-attention kernels reference function constants (KV types), so even
        // their DEFAULT form must be built through constantValues -- an empty set leaves
        // every constant undefined and the in-source defaults (f16) apply, which is the
        // pre-feature code exactly.
        auto make_cv = [&](NSString * nm, MTLFunctionConstantValues * cv)
            -> id<MTLComputePipelineState> {
            NSError * e = nil;
            stamp_epi_act(cv);
            id<MTLFunction> f = [lib newFunctionWithName:nm constantValues:cv error:&e];
            if (f == nil) { NSLog(@"imparo metal: %@ specialize: %@", nm, e); return nil; }
            id<MTLComputePipelineState> ps =
                [g.device newComputePipelineStateWithFunction:f error:&e];
            if (ps == nil) { NSLog(@"imparo metal: %@ pipeline: %@", nm, e); }
            return ps;
        };
        MTLFunctionConstantValues * cv_empty = [MTLFunctionConstantValues new];
        // Q8_0 weight pipelines. Q8_DECODE_ROWS and Q8_TOKEN_TILE have no in-source
        // default, so every one of these must go through constantValues -- an unset
        // constant fails newFunctionWithName outright rather than picking a value.
        {
            const double t_q8 = CACurrentMediaTime();
            for (uint32_t rg = 0; rg < 3u; ++rg) {
                const uint32_t rows = 1u << rg;          // 1, 2, 4
                MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
                [cv setConstantValue:&rows type:MTLDataTypeUInt atIndex:6];
                g.p_q8mv_rows[rg] = make_cv(@"imparo_q8_0_gemv", cv);
            }
            for (uint32_t tg = 0; tg < 4u; ++tg) {
                const uint32_t tile = 1u << tg;          // 1, 2, 4, 8
                MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
                [cv setConstantValue:&tile type:MTLDataTypeUInt atIndex:7];
                g.p_q8mm_tile[tg] = make_cv(@"imparo_q8_0_matmat", cv);
            }
            // Only the selected shapes are built unless a sweep asks for all of them,
            // for the same reason as the register-tile candidates: a compute pipeline is
            // GPU-resident code and these kernels are fully unrolled.
            uint32_t q8_built = 0;
            for (uint32_t i = 0; i < Q8_GEMM_CANDIDATES; ++i) {
                if (!g_q8_all && i != g_q8_gemm_shape && i != g_q8_gemm_large_shape) {
                    continue;
                }
                NSString * n  = [NSString stringWithFormat:@"imparo_q8_gemm_%u", i];
                NSString * nf = [NSString stringWithFormat:@"imparo_q8_gemm_%u_full", i];
                g.p_q8gemm[i]      = make_cv(n,  cv_empty);
                g.p_q8gemm_full[i] = make_cv(nf, cv_empty);
                ++q8_built;
            }
            g.p_q8row = make_cv(@"imparo_q8_0_row", cv_empty);
            g.p_gather[0] = make_cv(@"imparo_f32_gather_rows", cv_empty);
            g.p_gather[1] = make_cv(@"imparo_q4_0_gather_rows", cv_empty);
            g.p_gather[2] = make_cv(@"imparo_q8_0_gather_rows", cv_empty);
            if (t_on) { NSLog(@"imparo metal timing: %u Q8 GEMM shapes %.2f s",
                              q8_built, CACurrentMediaTime() - t_q8); }
        }
        g.p_attn_dec_direct = make_cv(@"imparo_attention_decode_direct", cv_empty);
        // WHICH TYPE THESE PIPELINES ARE SPECIALIZED FOR, and why it is not simply the
        // global. There are two ways a layer's cache ends up quantized, and this build
        // used to see only one of them:
        //
        //   global    g_kv_type_k/v != f16   the server's --cache-type-k/v; every layer
        //   per-layer g_kvq_mask_k/v bit set the IMPARO_KVQ_MASK_* diagnostic, WHICH ONLY
        //                                    APPLIES WHEN THE GLOBAL IS f16 (kv_eff_type
        //                                    consults the mask only for base == 1)
        //
        // So the mask's intended use -- quantize these layers, leave the rest f16 -- took
        // the global branch to f16 and never built the _q pipelines that the masked layers
        // then ask for. Call sites check for nil, so it degraded instead of crashing,
        // which is worse: the layers silently ran a path nobody chose.
        const uint32_t kq_ty = g_kv_type_k != 1u ? g_kv_type_k
                             : (g_kvq_mask_k != 0u ? g_kvq_type : 1u);
        const uint32_t vq_ty = g_kv_type_v != 1u ? g_kv_type_v
                             : (g_kvq_mask_v != 0u ? g_kvq_type : 1u);
        if (kq_ty != 1u || vq_ty != 1u) {
            MTLFunctionConstantValues * cv_q = [MTLFunctionConstantValues new];
            [cv_q setConstantValue:&kq_ty type:MTLDataTypeUInt atIndex:3];
            [cv_q setConstantValue:&vq_ty type:MTLDataTypeUInt atIndex:4];
            g.p_attn_dec_direct_q           = make_cv(@"imparo_attention_decode_direct", cv_q);
            g.p_attn_dec_scoretile_q     = make_cv(@"imparo_attention_decode_scoretile", cv_q);
            {
                NSString * sn[3] = { @"imparo_attention_decode_stream_dk128", @"imparo_attention_decode_stream_dk256", @"imparo_attention_decode_stream_dk512" };
                for (int i = 0; i < 3; ++i) { g.p_attn_dec_stream_q[i] = make_cv(sn[i], cv_q); }
                g.p_attn_dec_stream_g2_q =
                    make_cv(@"imparo_attention_decode_stream_dk512_g2", cv_q);
            }
            for (int gi = 0; gi < 3; ++gi) {
                g.p_attn_dec_scoretile_gqa_q[gi] = make_cv(kGqaScoretileNames[gi], cv_q);
            }
        }
        g.p_actmul = make(lib, @"imparo_act_mul");
        g.p_add4    = make(lib, @"imparo_add4");
        g.p_copy4   = make(lib, @"imparo_copy4");
        g.p_actmul4= make(lib, @"imparo_act_mul4");
        g.p_act4    = make(lib, @"imparo_act4");
        g.p_scale4  = make(lib, @"imparo_scale4");
        g.p_addscale4 = make(lib, @"imparo_add_scale4");
        g.p_act    = make(lib, @"imparo_act");
        g.p_shortconv       = make(lib, @"imparo_shortconv");
        g.p_shortconv_state = make(lib, @"imparo_shortconv_state");
        g.p_add    = make(lib, @"imparo_add");
        g.p_mul    = make(lib, @"imparo_mul");
        g.p_scale  = make(lib, @"imparo_scale");
        g.p_addscale  = make(lib, @"imparo_add_scale");
        g.p_copy   = make(lib, @"imparo_copy");
        g.p_softcap= make(lib, @"imparo_softcap");
        g.p_argmax = make(lib, @"imparo_argmax");
        g.p_ple    = make(lib, @"imparo_ple_combine");
        g.p_ple_gather = make(lib, @"imparo_ple_gather_combine");
        g.p_cvt_f16    = make(lib, @"imparo_cvt_f32_f16");
        g.p_mma_peak   = make(lib, @"imparo_mma_peak");
        g.p_bw_read    = make(lib, @"imparo_bw_read");
        {
            // The spill-cliff ladder. A probe kernel exists ONLY to isolate one variable
            // -- here, live accumulator count -- which is why it can answer a question
            // the production kernels only hint at.
            const char * nm[8] = {"imparo_spill_4","imparo_spill_8","imparo_spill_12",
                                  "imparo_spill_16","imparo_spill_24","imparo_spill_32",
                                  "imparo_spill_48","imparo_spill_64"};
            for (int i = 0; i < 8; ++i) {
                g.p_spill[i] = make(lib, [NSString stringWithUTF8String:nm[i]]);
            }
        }
        g.p_gemv_probe = make(lib, @"imparo_gemv_probe");
        g.p_mma_loaded = make(lib, @"imparo_mma_loaded");
        g.p_mma_dev_a  = make(lib, @"imparo_mma_device_a");
        g.p_scoremix   = make(lib, @"imparo_mma_scoremix");
        for (int gi = 0; gi < 3; ++gi) {
            g.p_attn_dec_scoretile_gqa[gi] = make_cv(kGqaScoretileNames[gi], cv_empty);
        }
        g.p_hadamard   = make(lib, @"imparo_hadamard64");
        g.p_attn_dec_scoretile = make_cv(@"imparo_attention_decode_scoretile", cv_empty);
        if (getenv("IMPARO_ATTN_WHICH")) {
            // REGISTER PRESSURE, per group size. maxTotalThreadsPerThreadgroup is the
            // compiler's own verdict: it falls when a kernel needs more registers per
            // thread. Printed because HQ=2 measures 3.6% slower than BOTH HQ=1 and HQ=4
            // at equal threadgroups, equal work per threadgroup and equal shared memory --
            // non-monotone in every dispatch parameter, which leaves codegen.
            for (int gi = 0; gi < 3; ++gi) {
                id<MTLComputePipelineState> p = g.p_attn_dec_scoretile_gqa[gi];
                fprintf(stderr, "attn pipeline %s maxthreads=%lu simdwidth=%lu\n",
                        [kGqaScoretileNames[gi] UTF8String],
                        (unsigned long)(p ? [p maxTotalThreadsPerThreadgroup] : 0),
                        (unsigned long)(p ? [p threadExecutionWidth] : 0));
            }
            fprintf(stderr, "attn pipeline ungrouped maxthreads=%lu\n",
                    (unsigned long)(g.p_attn_dec_scoretile
                                    ? [g.p_attn_dec_scoretile maxTotalThreadsPerThreadgroup]
                                    : 0));
        }
        {
            NSString * sn[3] = { @"imparo_attention_decode_stream_dk128", @"imparo_attention_decode_stream_dk256", @"imparo_attention_decode_stream_dk512" };
            for (int i = 0; i < 3; ++i) { g.p_attn_dec_stream[i] = make_cv(sn[i], cv_empty); }
            g.p_attn_dec_stream_g2 = make_cv(@"imparo_attention_decode_stream_dk512_g2", cv_empty);
        }
        g.p_attn_dec_combine  = make(lib, @"imparo_attention_decode_combine");
        // Every kernel that inlines kv_slot references KV_PAGED_FC now, so the
        // DEFAULT (paged) forms must also be created through constantValues --
        // a plain newFunctionWithName fails on functions with constant refs.
        g.p_kvstore      = make_cv(@"imparo_kv_store", cv_empty);
        g.p_kvstore_q4   = make_cv(@"imparo_kv_store_q4", cv_empty);
        g.p_kvstore_q8   = make_cv(@"imparo_kv_store_q8", cv_empty);
        g.p_kvstore_rt   = make_cv(@"imparo_kv_store_rt", cv_empty);
        g.p_kv_dq        = make_cv(@"imparo_kv_dq", cv_empty);
        g.p_attn_pre_qtile   = make_cv(@"imparo_attention_prefill_qtile", cv_empty);
        g.p_attn_pre_qtile16 = make_cv(@"imparo_attention_prefill_qtile16", cv_empty);
        g.p_attn_pre_qtile16h  = make_cv(@"imparo_attention_prefill_qtile16h", cv_empty);
        g.p_attn_pre_qtile16h2 = make_cv(@"imparo_attention_prefill_qtile16h2", cv_empty);
        g.p_attn_pre_qtile16h8 = make_cv(@"imparo_attention_prefill_qtile16h8", cv_empty);
        // The qcomb family reads ATTN_LIVE_MASK. cv_empty leaves it at the in-source
        // default (true); with the knob off they are built with it explicitly false, so
        // exactly one pipeline set exists either way.
        MTLFunctionConstantValues * cv_pre = cv_empty;
        if (!g_attn_live_mask) {
            cv_pre = [MTLFunctionConstantValues new];
            const bool lm_off = false;
            [cv_pre setConstantValue:&lm_off type:MTLDataTypeBool atIndex:10];
        }
        g.p_attn_pre_qcomb256  = make_cv(@"imparo_attention_prefill_qcomb256", cv_pre);
        g.p_attn_pre_qcomb256x = make_cv(@"imparo_attention_prefill_qcomb256x", cv_pre);
        g.p_attn_pre_qcomb512  = make_cv(@"imparo_attention_prefill_qcomb512", cv_pre);
        g.p_attn_pre_qcomb512b2 = make_cv(@"imparo_attention_prefill_qcomb512b2", cv_pre);
        g.p_attn_pre_qcomb256b2 = make_cv(@"imparo_attention_prefill_qcomb256b2", cv_pre);
        g.p_attn_pre_qcomb512x = make_cv(@"imparo_attention_prefill_qcomb512x", cv_pre);
        {
            const bool paged_off = false;
            MTLFunctionConstantValues * cv_id = [MTLFunctionConstantValues new];
            [cv_id setConstantValue:&paged_off type:MTLDataTypeBool atIndex:5];
            g.p_attn_dec_direct_id       = make_cv(@"imparo_attention_decode_direct", cv_id);
            g.p_attn_dec_scoretile_id = make_cv(@"imparo_attention_decode_scoretile", cv_id);
            for (int gi = 0; gi < 3; ++gi) {
                g.p_attn_dec_scoretile_gqa_id[gi] = make_cv(kGqaScoretileNames[gi], cv_id);
            }
            {
                NSString * sn[3] = { @"imparo_attention_decode_stream_dk128", @"imparo_attention_decode_stream_dk256", @"imparo_attention_decode_stream_dk512" };
                for (int i = 0; i < 3; ++i) { g.p_attn_dec_stream_id[i] = make_cv(sn[i], cv_id); }
                g.p_attn_dec_stream_g2_id =
                    make_cv(@"imparo_attention_decode_stream_dk512_g2", cv_id);
            }
            g.p_kvstore_id    = make_cv(@"imparo_kv_store", cv_id);
            g.p_kvstore_q4_id = make_cv(@"imparo_kv_store_q4", cv_id);
            g.p_kvstore_q8_id = make_cv(@"imparo_kv_store_q8", cv_id);
            g.p_kvstore_rt_id = make_cv(@"imparo_kv_store_rt", cv_id);
            if (g_kv_type_k != 1u || g_kv_type_v != 1u) {
                MTLFunctionConstantValues * cv_qid = [MTLFunctionConstantValues new];
                [cv_qid setConstantValue:&g_kv_type_k type:MTLDataTypeUInt atIndex:3];
                [cv_qid setConstantValue:&g_kv_type_v type:MTLDataTypeUInt atIndex:4];
                [cv_qid setConstantValue:&paged_off type:MTLDataTypeBool atIndex:5];
                g.p_attn_dec_direct_q_id       = make_cv(@"imparo_attention_decode_direct", cv_qid);
                g.p_attn_dec_scoretile_q_id = make_cv(@"imparo_attention_decode_scoretile", cv_qid);
                for (int gi = 0; gi < 3; ++gi) {
                    g.p_attn_dec_scoretile_gqa_q_id[gi] = make_cv(kGqaScoretileNames[gi], cv_qid);
                }
                {
                    NSString * sn[3] = { @"imparo_attention_decode_stream_dk128", @"imparo_attention_decode_stream_dk256", @"imparo_attention_decode_stream_dk512" };
                    for (int i = 0; i < 3; ++i) { g.p_attn_dec_stream_q_id[i] = make_cv(sn[i], cv_qid); }
                    g.p_attn_dec_stream_g2_q_id =
                        make_cv(@"imparo_attention_decode_stream_dk512_g2", cv_qid);
                }
            }
        }
        struct { id<MTLComputePipelineState> p; const char * name; } required[] = {
            {g.p_q4mm, "q4mm"}, {g.p_rms, "rms_norm"}, {g.p_attn_dec_direct, "attention_decode_direct"},
            {g.p_rope, "rope"}, {g.p_ple, "ple_combine"},
            {g.p_ple_gather, "ple_gather_combine"}, {g.p_cvt_f16, "cvt_f32_f16"},
            {g.p_attn_dec_scoretile, "attention_decode_scoretile"},
            {g.p_attn_dec_combine, "attention_decode_combine"},
            {g.p_attn_pre_qtile, "attention_prefill_qtile"},
            {g.p_q4mm_pre, "matmat_prefill"}, {g.p_rt[g_rt_shape], "rt_gemm"},
            {g.p_f32mm, "f32_matmat"}, {g.p_q4row, "q4_row"},
            {g.p_kvstore, "kv_store"}, {g.p_actmul, "act_mul"},
            // A nil pipeline makes its op a SILENT no-op: the shortconv check read back
            // an output of exactly zeros and a state exactly as written, which is what
            // "the dispatch never happened" looks like from outside.
            {g.p_shortconv, "shortconv"}, {g.p_shortconv_state, "shortconv_state"},
        };
        bool missing = false;
        for (const auto & r : required) {
            if (r.p == nil) { NSLog(@"imparo metal: pipeline '%s' missing", r.name); missing = true; }
        }
        if (missing) { return 3; }
        NSLog(@"imparo metal: pipelines ok rt_shape=%u rt=%p cvt=%p pre=%p",
              g_rt_shape, (__bridge void *)g.p_rt[g_rt_shape], (__bridge void *)g.p_cvt_f16,
              (__bridge void *)g.p_q4mm_pre);

        const size_t page = 16384;
        const size_t mapped = ((size_t)len + page - 1) / page * page;
        g.weights = [g.device newBufferWithBytesNoCopy:(void *)base length:mapped
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        return g.weights == nil ? 4 : 0;
    }
}

// What one qcomb threadgroup needs, in floats. This MIRRORS the layout in
// attention_prefill_qcomb_body -- staged Q, score tile, running max/sum, the
// per-row-group diagonals, and the threadgroup tail spill when the tail does not
// live in device memory. Kept as arithmetic on the shape rather than a hand-added
// constant per instantiation, because a hand-added constant is how the head_dim 256
// QT-16 shape came to ask for 33408 bytes against a 32768 limit: 640 over, never
// validated, and latent only because no layer of the model in front of us reaches
// that branch. Any model with head_dim 256 layers and an f16 cache would have hit it.
static NSUInteger qcomb_tg_floats(uint32_t qt, uint32_t hd, uint32_t pt, bool half_q,
                                  bool device_spill) {
    const uint32_t qrows = qt / 8u;
    const NSUInteger sq = (half_q && qt == 16u) ? (NSUInteger)qt * hd / 2u
                                               : (NSUInteger)qt * hd;
    return sq + (NSUInteger)qt * pt + 2u * qt + (NSUInteger)qrows * 64u
         + (device_spill ? 0u : (NSUInteger)qt * hd);
}

extern "C" int imparo_metal_alloc(uint32_t id, uint64_t bytes) {
    if (id >= B_COUNT) { return 1; }
    // Reuse the existing buffer when it fits, but ONLY while it is not much too large.
    //
    // Growing-only reuse is what kept a prefill-width buffer alive through every decode
    // step: activations are reserved for the widest batch, decode runs one token wide, and
    // the wide allocation was never given back. Shrinking at half is hysteresis, so a batch
    // that wobbles by a few tokens does not reallocate on every call.
    if (g.bufs[id] != nil && !g.in_arena[id] && g.sizes[id] >= bytes
        && (g_shrink == 0 || bytes * 2 >= g.sizes[id])) { return 0; }
    if (g.bufs[id] != nil) {
        // Releasing the reference is not enough: the driver keeps the pages in its own
        // cache, so the footprint does not move. Marking it empty first discards them.
        [g.bufs[id] setPurgeableState:MTLPurgeableStateEmpty];
        g.bufs[id] = nil;
    }
    g.bufs[id] = [g.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    g.sizes[id] = bytes;
    g.in_arena[id] = 0;
    g.buf_off[id] = 0;
    return g.bufs[id] == nil ? 2 : 0;
}

extern "C" uint64_t imparo_metal_allocated_bytes(void) {
    if (g.device == nil) { return 0ull; }
    return (uint64_t) [g.device currentAllocatedSize];
}

// Grow a KV layer's buffers, keeping what is already in them.
//
// The KV cache was sized to the CONTEXT at startup, so a 440-token conversation at ctx 8192
// held the same 148 MiB as a full one -- and on this model 128 MiB of that is four
// full-attention layers whose head_dim is 512. The window caps the other twenty at 512 slots
// already, which is why they are only 20 MiB.
//
// Growing costs a copy of what is already stored, so it doubles rather than creeping: the
// copies are then a constant fraction of the bytes ever written. A Metal buffer cannot be
// resized in place, and this runs BETWEEN command buffers, never inside one.
extern "C" int imparo_metal_grow_kv(uint32_t n_layers, const uint64_t * bytes) {
    if (g.device == nil) { return 1; }
    id<MTLCommandBuffer> cb = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    // The buffers being replaced, held until the blit that READS them has completed and
    // then discarded. Dropping the last reference is not enough in general -- the driver
    // keeps the pages in its own cache -- which is why `imparo_metal_alloc` marks a buffer
    // empty before releasing it; this makes the two paths agree.
    //
    // MEASURED NEUTRAL on the cold_compare long prompt (241 MiB against 239-240 without
    // it), because that request grows the cache once and those pages were being returned
    // anyway. Kept for the case the benchmark does not cover: at KV_BLOCK 64 a long
    // generation grows every 64 decode tokens, and each growth retires a larger buffer.
    std::vector<id<MTLBuffer>> retired;
    for (uint32_t i = 0; i < n_layers && i < g.kv_k.size(); ++i) {
        if (bytes[i] == 0) { continue; }
        for (int kv = 0; kv < 2; ++kv) {
            id<MTLBuffer> old = kv == 0 ? g.kv_k[i] : g.kv_v[i];
            if (old != nil && [old length] >= bytes[i]) { continue; }
            id<MTLBuffer> fresh = [g.device newBufferWithLength:bytes[i]
                                                       options:MTLResourceStorageModeShared];
            if (fresh == nil) { return 2; }
            if (old != nil && [old length] > 0) {
                if (cb == nil) { cb = [g.queue commandBuffer]; blit = [cb blitCommandEncoder]; }
                [blit copyFromBuffer:old sourceOffset:0
                            toBuffer:fresh destinationOffset:0 size:[old length]];
            }
            if (old != nil) { retired.push_back(old); }
            if (kv == 0) { g.kv_k[i] = fresh; } else { g.kv_v[i] = fresh; }
        }
    }
    if (cb != nil) {
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    // Only now: the blit read these, so they could not be discarded any earlier.
    for (id<MTLBuffer> b : retired) { [b setPurgeableState:MTLPurgeableStateEmpty]; }
    retired.clear();
    return 0;
}

// An arena that several activation buffers share, because their lifetimes do not overlap.
//
// A layer runs attention and then the feed-forward. Q, K, V and ATTN are live only in the
// first; G and U only in the second. They were nonetheless all permanently resident, so the
// footprint was their SUM rather than the peak live set -- which is why it grew about
// 160 MiB per doubling of ubatch against llama.cpp's 25-50, its graph allocator reusing
// memory by liveness.
//
//   attention set   Q + K + V + ATTN   40960 bytes a token
//   feed-forward    G + U              81920
//   separate        122880 a token     shared   81920, saving 40960
//
// These are StorageModeShared, so distinct MTLBuffer objects can be created over OVERLAPPING
// ranges of one page-aligned host allocation with newBufferWithBytesNoCopy -- the same call
// the weights mmap already uses. Every existing bind site keeps working: each id still has
// its own buffer object.
static void * g_arena = nullptr;
static uint64_t g_arena_size = 0;

static uint64_t page_round(uint64_t n) {
    const uint64_t pg = (uint64_t) sysconf(_SC_PAGESIZE);
    return (n + pg - 1u) / pg * pg;
}

extern "C" uint64_t imparo_metal_page_round(uint64_t n) { return page_round(n); }

extern "C" int imparo_metal_arena(uint64_t bytes) {
    // Keep the arena when it fits AND is not much too large. Growing-only reuse meant a
    // prefill-width arena outlived prefill: the batch narrows to 1 for decode, `bytes`
    // collapses, and this returned early holding the full width for the rest of the
    // conversation. Metal-owned buffers already shrink on the same hysteresis; the arena
    // was the one allocation that never gave anything back.
    if (g_arena != nullptr && bytes <= g_arena_size
        && (g_shrink == 0 || bytes * 2 >= g_arena_size)) { return 0; }
    // Every buffer placed in the old arena must go before it does.
    for (uint32_t i = 0; i < B_COUNT; ++i) {
        if (g.in_arena[i]) {
            g.bufs[i] = nil;
            g.sizes[i] = 0;
            g.in_arena[i] = 0;
        }
    }
    if (g_arena != nullptr) {
        munmap(g_arena, (size_t) g_arena_size);
        g_arena = nullptr; g_arena_size = 0;
    }
    const uint64_t want = page_round(bytes);
    // mmap, not posix_memalign + memset, for two reasons at once.
    //
    // Zeroed: `newBufferWithLength:` hands back zeroed memory and
    // `newBufferWithBytesNoCopy` does not, so moving activations into the arena silently
    // dropped that guarantee. A GEMM reads whole 8-row tiles and cannot mask, so the rows
    // past a batch's token count ARE read -- out of heap garbage that differed every run.
    // Identical runs disagreed: batch 512 with a 9-token remainder gave 19.1566, 19.1567
    // and 19.1548 on the same input.
    //
    // LAZY: an anonymous mapping is zero-filled by the kernel a page at a time, so pages
    // the batch never touches are never resident. `memset` over the whole arena made every
    // page resident at allocation and charged all of it to phys_footprint.
    void * m = mmap(nullptr, (size_t) want, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) { g_arena = nullptr; return 1; }
    g_arena = m;
    g_arena_size = want;
    g.arena_buf = [g.device newBufferWithBytesNoCopy:g_arena
                                              length:(NSUInteger) want
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
    if (g.arena_buf == nil) { return 3; }
    return 0;
}

/// Place one buffer at `offset` in the arena. The host lays the groups out so that two
/// groups with disjoint lifetimes both start at 0 and therefore overlap.
extern "C" int imparo_metal_place(uint32_t id, uint64_t offset, uint64_t bytes) {
    if (id >= B_COUNT || g_arena == nullptr || g.arena_buf == nil) { return 1; }
    const uint64_t len = page_round(bytes);
    if (offset + len > g_arena_size) { return 2; }
    if (g.bufs[id] != nil && !g.in_arena[id]) {
        [g.bufs[id] setPurgeableState:MTLPurgeableStateEmpty];
    }
    // One RESOURCE, many offsets: every placed id binds the same arena_buf at its
    // offset, so Metal's per-resource hazard tracking orders every aliased access.
    g.bufs[id] = g.arena_buf;
    g.buf_off[id] = (NSUInteger) offset;
    g.sizes[id] = len;
    g.in_arena[id] = 1;
    return 0;
}


// Per-layer 64-position block tables (KV pool paging). Identity until the pool
// assigns real blocks; grown at dispatch time so stage one needs no host plumbing.
static void kv_pt_audit(const char * who, uint32_t layer, uint32_t needed) {
    if (!getenv("IMPARO_KV_PT_AUDIT")) { return; }
    id<MTLBuffer> b = layer < g.kv_pt.size() ? g.kv_pt[layer] : nil;
    if (b == nil) { NSLog(@"pt-audit %s layer=%u NIL (needed=%u)", who, layer, needed); return; }
    const uint32_t n = (uint32_t)([b length] / 4u);
    const uint32_t * e = (const uint32_t *)[b contents];
    uint32_t bad = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; ++i) { if (e[i] > 65536u) { bad = i; break; } }
    NSLog(@"pt-audit %s layer=%u entries=%u needed=%u e0=%u e1=%u bad_at=%d",
          who, layer, n, needed, n > 0 ? e[0] : 0, n > 1 ? e[1] : 0, (int)bad);
}

static void ensure_kv_pt(uint32_t layer, uint32_t entries_needed) {
    if (layer >= g.kv_pt.size()) { g.kv_pt.resize(layer + 1, nil); }
    id<MTLBuffer> cur = g.kv_pt[layer];
    const uint32_t have = cur == nil ? 0u : (uint32_t)([cur length] / 4u);
    if (have >= entries_needed) { return; }
    const uint32_t want = entries_needed < 64u ? 64u : entries_needed * 2u;
    id<MTLBuffer> fresh = [g.device newBufferWithLength:(NSUInteger)want * 4u
                                                options:MTLResourceStorageModeShared];
    uint32_t * e = (uint32_t *)[fresh contents];
    // Growth PRESERVES pool-assigned entries and identity-extends the tail, so a
    // table set by the pool survives a mid-forward grow.
    for (uint32_t i = 0; i < have; ++i) { e[i] = ((uint32_t *)[cur contents])[i]; }
    for (uint32_t i = have; i < want; ++i) { e[i] = i; }
    g.kv_pt[layer] = fresh;
}

// The pool assigns physical blocks: replace a layer's table entries wholesale.
extern "C" void imparo_metal_set_kv_pages(uint32_t layer, const uint32_t * e, uint32_t n) {
    ensure_kv_pt(layer, n);
    memcpy((uint32_t *)[g.kv_pt[layer] contents], e, (size_t)n * 4u);
    bool ident = true;
    for (uint32_t i = 0; i < n; ++i) {
        if (e[i] != i) { ident = false; break; }
    }
    if (g.kv_pt_ident.size() <= layer) { g.kv_pt_ident.resize(layer + 1, 1); }
    g.kv_pt_ident[layer] = ident ? 1 : 0;
}

static inline bool kv_pt_is_ident(uint32_t layer) {
    return layer >= g.kv_pt_ident.size() || g.kv_pt_ident[layer] != 0;
}

static inline uint64_t kv_reg(uint32_t layer, bool is_v) {
    const std::vector<uint64_t> & v = is_v ? g.kv_reg_v : g.kv_reg_k;
    return layer < v.size() ? v[layer] : 0ull;
}

extern "C" int imparo_metal_alloc_kv(uint32_t n_layers, const uint64_t * bytes) {
    g.kv_k.assign(n_layers, nil);
    g.kv_v.assign(n_layers, nil);
    g.kv_pt.assign(n_layers, nil);
    g.kv_reg_k.assign(n_layers, 0ull);
    g.kv_reg_v.assign(n_layers, 0ull);
    for (uint32_t i = 0; i < n_layers; ++i) {
        if (bytes[i] == 0) { continue; }
        g.kv_k[i] = [g.device newBufferWithLength:bytes[i] options:MTLResourceStorageModeShared];
        g.kv_v[i] = [g.device newBufferWithLength:bytes[i] options:MTLResourceStorageModeShared];
        if (g.kv_k[i] == nil || g.kv_v[i] == nil) { return 1; }
    }
    return 0;
}

// `[queue commandBuffer]` returns an AUTORELEASED object. Splitting a decode token into
// several command buffers therefore piles them up for as long as the enclosing pool lives,
// which on a server thread is the whole request: 6 per token x 64 tokens sat unreleased,
// and the footprint grew with the split (1 cb/token 214.2 MiB, 21 cb/token 220.6 MiB).
// One pool per forward pass releases them at the end of the token instead.
extern "C" void imparo_metal_begin(void) {
    if (g.pool == nullptr) { g.pool = objc_autoreleasePoolPush(); }
    g.cb = [g.queue commandBuffer];
    // SERIAL, deliberately. llama.cpp encodes with MTLDispatchTypeConcurrent and barriers
    // only where a dependency needs one; that was implemented here and MEASURED, marking
    // the Q/K/V projections, the K/V norms and the two KV stores as overlappable:
    //
    //   prefill  11516 / 11581 / 11551 ms  against a serial 11517-11537
    //   decode   38.6 tok/s  against 38.9-39.0, and footprint 165 MiB against 157-159
    //
    // Neutral to slightly worse on both, so it was reverted. At a 512-token prefill each
    // projection is ~0.67 GFLOP and already fills the machine, and overlapping saturated
    // kernels buys nothing; the extra encoder state costs a few MiB. Concurrency is worth
    // revisiting only if the dispatches get small enough not to saturate.
    //
    // Decode is exactly that case (~1100 dispatches for ~25 ms of GPU work), which is what
    // IMPARO_CONCURRENT revisits -- with the read/write sets declared per dispatch site
    // rather than four hand-marked pairs. See haz().
    g.enc = g_concurrent
        ? [g.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
        : [g.cb computeCommandEncoder];
    g_haz_reads = 0; g_haz_writes = 0;
    g_xh_src = 0xffffffffu;
}

// Commit what has been encoded so far and start a new command buffer, WITHOUT waiting.
//
// Encoding is CPU work that is otherwise serial with the GPU: a decode token encodes ~1100
// dispatches, commits, then waits, and the GPU sits idle for the whole encode. Committing
// part way lets the GPU start on the first half while the CPU encodes the second.
//
//   Before:  encode 1100 dispatches -> commit -> GPU runs -> read
//   After:   encode 550 -> commit -> GPU runs || CPU encodes 550 -> commit -> read
//
// Command buffers on one queue execute in the order they are committed, so the split does
// not reorder any work. Hazards WITHIN a command buffer are tracked by Metal; across the
// split the ordering guarantee is what keeps the result identical, which is checked by
// comparing logits against the unsplit path.
extern "C" void imparo_metal_flush(void) {
    if (g.enc == nil) { return; }
    [g.enc endEncoding];
    [g.cb commit];
    // ALWAYS retained until the region ends, not only under g_prof. `end` needs every
    // buffer of the region to report the region's GPU time, and the tuner runs with
    // profiling off -- gated on g_prof, a region that flushed mid-way silently lost the
    // flushed buffer's time and under-reported.
    g.pending.push_back(g.cb);
    g.cb = [g.queue commandBuffer];
    // Same dispatch type as at the other encoder. A new encoder starts a fresh hazard
    // window: cross-encoder ordering comes from commit order plus Metal's tracking, which
    // still applies between encoders.
    g.enc = g_concurrent
        ? [g.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
        : [g.cb computeCommandEncoder];
    g_haz_reads = 0; g_haz_writes = 0;
}

// GPU seconds of the LAST begin/end region, always recorded. The tuner times its
// candidates with this instead of a wall clock: a wall clock around commit and
// waitUntilCompleted also charges submission latency, which is variable and, for a region
// holding one or two short buffers, comparable to the work itself. That is what put the
// rt_shape micro-bench's noise floor at 26% -- wider than the 1.3x that separates the
// viable tile shapes, so no candidate could ever beat its switching margin, and the knob
// was moved end-to-end instead of the measurement being fixed. Two property reads on an
// already-completed buffer, so it costs nothing to leave on.
double g_last_gpu_s = 0.0;

extern "C" double imparo_metal_last_gpu_us(void) { return g_last_gpu_s * 1e6; }

extern "C" int imparo_metal_end(void) {
    [g.enc endEncoding];
    const double t0 = g_prof ? CACurrentMediaTime() : 0.0;
    [g.cb commit];
    [g.cb waitUntilCompleted];
    // Buffers on one queue complete in commit order, so the wait above means every
    // flushed buffer of this region is done and its timestamps are valid.
    // One sum for both consumers. Buffers on a queue complete in commit order, so the
    // wait above means every flushed buffer of this region is done and its timestamps
    // are valid; summing only the last would undercount by however many flushes ran.
    g_last_gpu_s = [g.cb GPUEndTime] - [g.cb GPUStartTime];
    for (id<MTLCommandBuffer> pcb : g.pending) {
        g_last_gpu_s += [pcb GPUEndTime] - [pcb GPUStartTime];
    }
    if (g_prof) {
        prof_resolve();
        g_prof_wall_s += CACurrentMediaTime() - t0;
        g_prof_gpu_s  += g_last_gpu_s;
        g_prof_cbs    += (uint64_t)g.pending.size() + 1;
    }
    g.pending.clear();
    const bool bad = [g.cb error] != nil;
    g.enc = nil; g.cb = nil;
    // Drained only after the wait, so every buffer in the pool has already completed.
    if (g.pool != nullptr) { objc_autoreleasePoolPop(g.pool); g.pool = nullptr; }
    return bad ? 1 : 0;
}

extern "C" void imparo_metal_write(uint32_t id, uint64_t off, const float * src, uint64_t n) {
    g_cvt_valid = 0;   // host wrote a buffer directly; the half copy may be stale
    std::memcpy((char *)[g.bufs[id] contents] + g.buf_off[id] + off * 4, src, n * 4);
}
extern "C" void imparo_metal_read(uint32_t id, uint64_t off, float * dst, uint64_t n) {
    std::memcpy(dst, (char *)[g.bufs[id] contents] + g.buf_off[id] + off * 4, n * 4);
}

// Q8_0 has its own dispatch: three routes, its own grid and threadgroup sizing, and a
// 64-bit weight offset. Returns false when nothing was encoded, so the caller logs the
// reason once instead of faulting inside the driver on a nil pipeline.
static bool q8_matmat(uint64_t w_off, uint32_t n_in, uint32_t n_out, uint32_t src,
                      uint32_t dst, uint32_t n_tok, uint32_t src_row) {
    if (n_in == 0u || n_out == 0u || n_tok == 0u) { return false; }
    if ((n_in % Q8_BLOCK_ELEMENTS) != 0u) {
        NSLog(@"imparo metal: Q8 matmat n_in=%u is not a multiple of %u; every route "
              @"walks whole blocks, so refusing rather than reading past the row",
              n_in, Q8_BLOCK_ELEMENTS);
        return false;
    }
    const bool use_gemm = n_tok > g_q8_gemv_max_tok;

    // The wide prefill shape may hand over to a second geometry at a tuned boundary.
    uint32_t shape = g_q8_gemm_shape;
    if (use_gemm && g_q8_gemm_large_min_tok != 0u && n_tok >= g_q8_gemm_large_min_tok
        && g.p_q8gemm[g_q8_gemm_large_shape] != nil) {
        shape = g_q8_gemm_large_shape;
    }
    const uint32_t rows = Q8_GEMM_SHAPES[shape][0];
    const uint32_t toks = Q8_GEMM_SHAPES[shape][1];
    const uint32_t nsg  = Q8_GEMM_SHAPES[shape][2];
    // FULL removes every edge predicate, so it is legal only on an exact grid with no
    // fused epilogue -- the kernel would otherwise write whole tiles past n_out/n_tok.
    const bool full = use_gemm && g_q8_full_tiles != 0u && g_epilogue == 0u
                   && (n_out % rows) == 0u && (n_tok % toks) == 0u
                   && g.p_q8gemm_full[shape] != nil;

    id<MTLComputePipelineState> sel = nil;
    if (use_gemm) {
        sel = full ? g.p_q8gemm_full[shape] : g.p_q8gemm[shape];
    } else if (n_tok == 1u) {
        sel = g.p_q8mv_rows[g_q8_decode_rows_log2];
    } else {
        sel = g.p_q8mm_tile[g_q8_token_tile_log2];
    }
    if (sel == nil) {
        NSLog(@"imparo metal: nil Q8 pipeline (n_tok=%u gemm=%d shape=%u full=%d "
              @"rows_log2=%u tile_log2=%u)", n_tok, (int)use_gemm, shape, (int)full,
              g_q8_decode_rows_log2, g_q8_token_tile_log2);
        return false;
    }
    haz(hb(src), hb(dst));
    [g.enc setComputePipelineState:sel];
    [g.enc setBuffer:g.weights offset:0 atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    // EIGHT bytes, not four: these kernels take `constant ulong & w_offset`. The Q4_0
    // path narrows the same value to 32 bits, which is why matmat refuses a non-Q8
    // offset above UINT32_MAX rather than truncating it.
    [g.enc setBytes:&w_off length:8 atIndex:3];
    [g.enc setBytes:&n_in length:4 atIndex:4];
    [g.enc setBytes:&n_out length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBytes:&src_row length:4 atIndex:7];
    [g.enc setBytes:&g_epilogue length:4 atIndex:13];

    // IMPARO_Q8_LOG=1 prints what each route actually selected. Gated, not commented
    // out: the whole block is behind the flag, so a measured run pays one getenv-cached
    // branch. Identical OUTPUT across shapes is expected -- the template keeps the same K
    // traversal and MMA order per output -- so results alone cannot show the shape knob
    // taking effect, and this is what does.
    static int q8_log = -1;
    if (q8_log < 0) { q8_log = getenv("IMPARO_Q8_LOG") != NULL ? 1 : 0; }
    if (q8_log) {
        if (use_gemm) {
            NSLog(@"imparo metal q8: gemm shape=%u rows=%u toks=%u nsg=%u full=%d "
                  @"grid=%ux%u n_out=%u n_tok=%u", shape, rows, toks, nsg, (int)full,
                  (n_out + rows - 1) / rows, (n_tok + toks - 1) / toks, n_out, n_tok);
        } else if (n_tok == 1u) {
            NSLog(@"imparo metal q8: gemv rows=%u sgs=%u n_out=%u",
                  1u << g_q8_decode_rows_log2, g_q8_decode_sgs, n_out);
        } else {
            NSLog(@"imparo metal q8: tile tile=%u sgs=%u n_out=%u n_tok=%u",
                  1u << g_q8_token_tile_log2, g_q8_batch_sgs, n_out, n_tok);
        }
    }
    if (use_gemm) {
        // Staged operands are halves: K x ROWS weights plus TOKENS x K activations. The
        // masked write-back reuses the same block as a TOKENS x ROWS float tile, so size
        // for whichever is larger -- they overlap in time, never in use.
        const NSUInteger stage_bytes =
            (NSUInteger)(Q8_BLOCK_ELEMENTS * rows + toks * Q8_BLOCK_ELEMENTS)
            * sizeof(uint16_t);
        const NSUInteger out_bytes = (NSUInteger)rows * toks * sizeof(float);
        [g.enc setThreadgroupMemoryLength:std::max(stage_bytes, out_bytes) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
        [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows - 1) / rows,
                                               (n_tok + toks - 1) / toks, 1)
              threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        return true;
    }
    if (n_tok == 1u) {
        const uint32_t dec_rows = 1u << g_q8_decode_rows_log2;
        // One float per (simdgroup, row) for the cross-simdgroup reduction.
        [g.enc setThreadgroupMemoryLength:(NSUInteger)dec_rows * g_q8_decode_sgs
                                         * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_DECODE); }
        [g.enc dispatchThreadgroups:MTLSizeMake((n_out + dec_rows - 1) / dec_rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(g_q8_decode_sgs * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        return true;
    }
    // Narrow batch: one simdgroup per output row, one threadgroup column per token tile.
    const uint32_t tile = 1u << g_q8_token_tile_log2;
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_DECODE); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + g_q8_batch_sgs - 1) / g_q8_batch_sgs,
                                           (n_tok + tile - 1) / tile, 1)
          threadsPerThreadgroup:MTLSizeMake(g_q8_batch_sgs * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// `wkind` is the weight-type -> kernel table index (0 = F32, 1 = Q4_0, 2 = Q8_0).
// Load-time validation on the Rust side guarantees no other value arrives; the guard
// below is defense in depth, not a path.
extern "C" void imparo_metal_matmat(uint32_t wkind, uint64_t w_off, uint32_t n_in,
                                    uint32_t n_out, uint32_t src, uint32_t dst,
                                    uint32_t n_tok, uint32_t src_row) {
    // one lane per token only pays off with a batch; decode keeps the split-row kernel
    // Batches up to g_gemv_max_tok take the GEMV; above it, the GEMM family (nb8
    // then wide). A tuned boundary again since the task-#5 wobble was root-caused
    // and fixed -- see the g_gemv_max_tok declaration.
    if (wkind > 2u) {
        NSLog(@"imparo metal: matmat got unknown weight kind %u (n_out=%u) -- load "
              @"validation should have rejected this model; refusing the dispatch", 
              wkind, n_out);
        return;
    }
    if (wkind == 2u) {
        // Q8_0 shares the buffer indices and the epilogue convention, nothing else: its
        // three routes have their own grid, threadgroup and boundary knobs.
        if (g_skip_cat == (n_tok > g_q8_gemv_max_tok ? PC_MATMAT_PREFILL
                                                    : PC_MATMAT_DECODE)) { return; }
        q8_matmat(w_off, n_in, n_out, src, dst, n_tok, src_row);
        return;
    }
    const bool is_q4 = wkind == 1u;
    bool use_prefill = is_q4 && n_tok > g_gemv_max_tok;
    // Skips the WHOLE matmul, staging and write-back included, so it is comparable with
    // llama.cpp's GGML_METAL_SKIP_OP=MUL_MAT. IMPARO_SKIP_MMA only removes the multiplies
    // and the dequantisation from inside the kernel, which is a different quantity.
    if (use_prefill  && g_skip_cat == PC_MATMAT_PREFILL) { return; }
    if (!use_prefill && g_skip_cat == PC_MATMAT_DECODE)  { return; }
    haz(hb(src), hb(dst));
    // Lanes per row is a single tuned value, NOT chosen per shape.
    //
    // Cold, ffn_down (10240 in) reaches 125 GB/s while ffn_gate (2560 in) manages 82.9, so
    // giving short rows fewer lanes -- more blocks each -- looked obviously right. Measured
    // it is slower (32.7 -> 32.0 tok/s), because lanes-per-row also sets how many
    // SIMDGROUPS are in flight: 32 lanes over a 10240-row matmul is 10240 simdgroups,
    // 8 lanes is a quarter of that. Parallelism beats per-lane run length here, which is
    // why the host sweep picked 32.
    // NR0 applies to the DECODE fast path only. The batched path keeps one row per lane
    // group, so prefill is always dispatched with the NR0 == 1 pipeline.
    const uint32_t nr_log2 = (n_tok > 1) ? 0u : g_nr0_log2;
    const uint32_t nr0     = 1u << nr_log2;
    id<MTLComputePipelineState> chosen = g.p_q4mm_lanes[g_lanes_log2][nr_log2];
    if (chosen == nil) { chosen = g.p_q4mm; }
    id<MTLComputePipelineState> pre = g.p_q4mm_pre;
    // The register-tiled kernel takes precedence and implements `skip` ITSELF via its
    // uniform. Selecting p_q4mm_pre_nomma on any nonzero skip meant every skip measurement
    // silently ran a DIFFERENT kernel -- which is why bits 1, 2 and 4 all produced the same
    // number, and why that was misread as the compiler eliminating dead code.
    if (g_rt && g.p_rt[g_rt_shape] != nil)                 { pre = g.p_rt[g_rt_shape]; }
    else if (g_skip_mma && g.p_q4mm_pre_nomma != nil)      { pre = g.p_q4mm_pre_nomma; }
    id<MTLComputePipelineState> sel = use_prefill ? pre : (is_q4 ? chosen : g.p_f32mm);
    if (sel == nil) {
        // setComputePipelineState: with nil faults inside the driver, and the backtrace
        // names neither the kernel nor the reason. Say which selection was empty.
        NSLog(@"imparo metal: nil pipeline in matmat (use_prefill=%d is_q4=%d g_rt=%d "
              @"shape=%u lanes_log2=%u)", (int)use_prefill, (int)is_q4, (int)g_rt,
              g_rt_shape, g_lanes_log2);
        return;
    }
    [g.enc setComputePipelineState:sel];
    [g.enc setBuffer:g.weights offset:0 atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    // EIGHT BYTES. Every one of these kernels takes `constant ulong & w_offset` now;
    // the value used to be narrowed to 32 bits, and this file's own gemma4 mapping is
    // 4,215,695,776 bytes -- 79 MB under the ceiling.
    [g.enc setBytes:&w_off length:8 atIndex:3];
    [g.enc setBytes:&n_in length:4 atIndex:4];
    [g.enc setBytes:&n_out length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBytes:&src_row length:4 atIndex:7];
    const uint32_t rows = g_rows;
    [g.enc setBytes:&rows length:4 atIndex:8];
    [g.enc setBytes:&g_epilogue length:4 atIndex:13];
    const NSUInteger sgs = g_sgs;                    // simdgroups per threadgroup
    if (use_prefill) {
        // Only the weight tile is staged; activations are read from device memory and
        // the write-back reuses the same block. Keep this well under 32 KB or occupancy
        // collapses. The register-tiled variant stages RT_K x RT_WS, half as much.
        // Narrow-N routing (task #11): batches of 2..g_nb8_max tokens take a 64x8
        // tile. Same k-order into every output as the wide tile, so the swap is
        // BIT-IDENTICAL; the boundary and the tile variant are tuner-owned
        // (hostconfig v10), swept under shipping routing.
        const bool nb8 = n_tok >= 2 && n_tok <= g_nb8_max
                      && pre == g.p_rt[g_rt_shape]
                      && g.p_rt_nb8 != nil && g.p_rt_nb8b != nil;
        if (nb8) {
            pre = g_nb8_shape ? g.p_rt_nb8b : g.p_rt_nb8;
            static bool nb8_logged = false;
            if (!nb8_logged && getenv("IMPARO_NB8_LOG")) {
                NSLog(@"imparo metal: nb8 tile engaged n_tok=%u n_out=%u shape=%u",
                      n_tok, n_out, g_nb8_shape);
                nb8_logged = true;
            }
        }
        const uint32_t rt_rows = nb8 ? 64 : RT_SHAPES[g_rt_shape][1] * 8 * RT_SHAPES[g_rt_shape][2];
        const uint32_t rt_toks = nb8 ?  8 : RT_SHAPES[g_rt_shape][0] * 8 * RT_SHAPES[g_rt_shape][3];
        const uint32_t rt_thr  = nb8 ? (g_nb8_shape ? 32u : 64u)
                                     : RT_SHAPES[g_rt_shape][2] * RT_SHAPES[g_rt_shape][3] * 32;
        // Two uses share the block: the staged weight tile (RT_K x RT_WS halves) and the
        // float write-back (tokens x RT_WS). The float use is always the larger, so size
        // for it. RT_WS is rows+2 in the kernel; keep the two in step.
        // Staged weights (RT_K x rows) plus staged activations (tokens x RT_K), or the
        // write-back tile (tokens x rows+2) -- whichever is larger, since they overlap.
        // two staged chunks (double buffered), each RT_K x (rows + 2)
        // tile-major weights (RT_K x rows) + tile-major activations (tokens x RT_K)
        // Only the staged weight tile now; the write-back goes straight to device.
        // staged tile is RT_K x (rows + 2) HALVES; express it in floats for the API
        const NSUInteger rt_k = 64;   // must match the kernel
        // Staged weight tile (RT_K x rows+2 halves), or the epilogue's output tile
        // (tokens x rows floats) when that path runs -- whichever is larger.
        const NSUInteger stage_f = (NSUInteger)rt_k * (rt_rows + 2) / 2;
        const NSUInteger epi_f   = g_epilogue ? (NSUInteger)rt_toks * rt_rows : 0;
        const NSUInteger shared_floats = g_rt ? std::max(stage_f, epi_f)
                                              : (NSUInteger)(64 * 72);
        [g.enc setThreadgroupMemoryLength:shared_floats * sizeof(float) atIndex:0];
        if (g_rt) {
            static bool rt_logged = false;
            static uint32_t rt_last_skip = 0xffffffffu;
            // Log the first dispatch, and again whenever the skip uniform CHANGES: a
            // diagnostic that silently reverts mid-run reads as a real measurement.
            if (!rt_logged || g_skip_mma != rt_last_skip) {
                rt_logged = true;
                rt_last_skip = g_skip_mma;
                NSLog(@"imparo metal: rt shape=%u pipeline=%p cvt=%p thr=%u max_thr=%lu "
                      @"shared=%lu rows=%u toks=%u skip=%u epi=%u",
                      g_rt_shape, (__bridge void *)pre, (__bridge void *)g.p_cvt_f16, rt_thr,
                      (unsigned long)(pre ? [pre maxTotalThreadsPerThreadgroup] : 0),
                      (unsigned long)(shared_floats * sizeof(float)), rt_rows, rt_toks,
                      g_skip_mma, g_epilogue);
            }
            id<MTLComputePipelineState> rt_sel = pre;
            uint32_t src_bind = src;
            // TAIL-SPLIT (task #21): the wide tile pads the last token-tile to 64, so a
            // 449-token batch computes 512 -- full staging AND full MACs for 1/64 useful
            // output, a measured 1.14x step between 447 and 449 tokens. llama.cpp has the
            // same defect on a 32 grid (no tail handling; ceil(ne11/32) padded tiles).
            // Fix: floor-to-64 through the wide tile, remainder 1..nb8_max through the
            // narrow tile IN THE SAME ENCODE. nb8's k-order is byte-equal to the wide
            // tile's (verified at task #11), and under half-A both read the SAME mirror
            // bytes the padded tile reads today, so the split is BIT-IDENTICAL.
            // Remainders above nb8_max keep the padded tile: the tuner's own crossover
            // says multi-pass nb8 loses there, and the waste is at most 16 tokens.
            const uint32_t wide_toks = RT_SHAPES[g_rt_shape][0] * 8 * RT_SHAPES[g_rt_shape][3];
            static bool ts_init = false; static bool ts_on = true;
            if (!ts_init) { const char * e = getenv("IMPARO_TAIL_SPLIT");
                            ts_on = !(e && e[0] == '0'); ts_init = true; }
            const uint32_t tail_r = n_tok % wide_toks;
            const bool tail_split = ts_on && !nb8 && n_tok > wide_toks
                                 && tail_r >= 1u && tail_r <= g_nb8_max
                                 && pre == g.p_rt[g_rt_shape]
                                 && g.p_rt_nb8 != nil && g.p_rt_nb8_h != nil;
            const uint32_t main_tok = tail_split ? n_tok - tail_r : n_tok;
            // Every prefill width takes this path (HALF_A_MIN, declared above with the
            // measurement): the batch's width must not choose the precision, or a
            // resumed pass stops reproducing the cold one.
            if (g_half_a && n_tok >= HALF_A_MIN
                && g.p_rt_h[g_rt_shape] != nil && g.bufs[B_XH] != nil) {
                // Convert the activation slice to half once per (buffer, content): the
                // cache lives until haz() sees a write to the source or the cb turns
                // over. Padded to whole token tiles because the GEMM reads whole tiles.
                const uint32_t padded = (n_tok + rt_toks - 1) / rt_toks * rt_toks;
                const uint64_t need = (uint64_t)padded * n_in;
                // Diagnostic (default off): IMPARO_SKIP_CVT=1 skips the conversion pass
                // and lets the GEMM read stale XH -- output wrong on purpose; the wall
                // difference is the cvt passes' share.
                static bool skip_cvt_init = false; static bool skip_cvt = false;
                if (!skip_cvt_init) { skip_cvt = getenv("IMPARO_SKIP_CVT") != nullptr;
                                      skip_cvt_init = true; }
                // A producer that already wrote the mirror (rms_norm, add, the up
                // epilogue) covers n_tok rows exactly; the GEMM's padding rows then read
                // stale mirror bytes, which is the float path's own padding story: row j
                // reaches only output row j, masked at write-back. The cvt fallback keeps
                // converting the padded range as before.
                const bool mirrored = g_xh_src == src
                                   && g_xh_elems >= (uint64_t)n_tok * n_in;
                if (!mirrored && !skip_cvt) {
                    static bool cvtlog_i = false; static bool cvtlog = false;
                    if (!cvtlog_i) { cvtlog = getenv("IMPARO_CVT_LOG") != nullptr; cvtlog_i = true; }
                    if (cvtlog) { NSLog(@"CVT src=%u n_in=%u n_tok=%u n_out=%u", src, n_in, n_tok, n_out); }
                    haz(hb(src), hb(B_XH));
                    [g.enc setComputePipelineState:g.p_cvt_f16];
                    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
                    [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:1];
                    const uint32_t cn = (uint32_t)need, coff = 0;
                    [g.enc setBytes:&cn length:4 atIndex:2];
                    [g.enc setBytes:&coff length:4 atIndex:3];
                    g_disp_seq += 1;
                    if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
                    [g.enc dispatchThreads:MTLSizeMake(cn, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    if (g_prof) { prof_end(); }
                    g_xh_src = src; g_xh_elems = need; g_xh_buf = B_XH;
                }
                haz(hb(g_xh_buf), hb(dst));   // order the GEMM behind the mirror's writer
                rt_sel = nb8 ? (g_nb8_shape ? g.p_rt_nb8b_h : g.p_rt_nb8_h)
                             : g.p_rt_h[g_rt_shape];
                src_bind = g_xh_buf;
            }
            [g.enc setComputePipelineState:rt_sel];
            [g.enc setBuffer:g.weights offset:0 atIndex:0];
            [g.enc setBuffer:g.bufs[src_bind] offset:g.buf_off[src_bind] atIndex:1];
            [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
            [g.enc setBytes:&w_off length:8 atIndex:3];
            [g.enc setBytes:&n_in length:4 atIndex:4];
            [g.enc setBytes:&n_out length:4 atIndex:5];
            [g.enc setBytes:&main_tok length:4 atIndex:6];
            [g.enc setBytes:&src_row length:4 atIndex:7];
            [g.enc setBytes:&g_skip_mma length:4 atIndex:12];
            [g.enc setBytes:&g_epilogue length:4 atIndex:13];
            // Fused-epilogue half mirror of G: written only when the down projection
            // will read half (same batch, so the read side's n_tok >= 64 gate applies).
            // epi_half is set EVERY dispatch -- encoder state persists, so a stale 1
            // from the previous up projection would leak into the next matmul.
            uint32_t epi_half = 0;
            if (g_epilogue && g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH2] != nil) {
                epi_half = 1;
                [g.enc setBuffer:g.bufs[B_XH2] offset:g.buf_off[B_XH2] atIndex:8];
                haz(0u, hb(B_XH2));
            }
            [g.enc setBytes:&epi_half length:4 atIndex:9];
            g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
            [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rt_rows - 1) / rt_rows,
                                                    (main_tok + rt_toks - 1) / rt_toks, 1)
                  threadsPerThreadgroup:MTLSizeMake(rt_thr, 1, 1)];
            if (g_prof) { prof_end(); }
            if (tail_split) {
                // The remainder rides the narrow tile. Reads use src_row + main (the
                // kernel adds src_row); WRITES are dispatch-relative, so the output and
                // the epilogue mirror bind with a row offset instead.
                const bool half_sel = (src_bind == g_xh_buf) && g_half_a;
                id<MTLComputePipelineState> tp = half_sel
                    ? (g_nb8_shape ? g.p_rt_nb8b_h : g.p_rt_nb8_h)
                    : (g_nb8_shape ? g.p_rt_nb8b   : g.p_rt_nb8);
                const uint32_t t_thr  = g_nb8_shape ? 32u : 64u;
                const uint32_t t_src_row = src_row + main_tok;
                [g.enc setComputePipelineState:tp];
                [g.enc setBuffer:g.weights offset:0 atIndex:0];
                [g.enc setBuffer:g.bufs[src_bind] offset:g.buf_off[src_bind] atIndex:1];
                [g.enc setBuffer:g.bufs[dst] offset:(g.buf_off[dst] + (NSUInteger)main_tok * n_out * 4) atIndex:2];
                [g.enc setBytes:&w_off length:8 atIndex:3];
                [g.enc setBytes:&n_in length:4 atIndex:4];
                [g.enc setBytes:&n_out length:4 atIndex:5];
                [g.enc setBytes:&tail_r length:4 atIndex:6];
                [g.enc setBytes:&t_src_row length:4 atIndex:7];
                [g.enc setBytes:&g_skip_mma length:4 atIndex:12];
                [g.enc setBytes:&g_epilogue length:4 atIndex:13];
                if (epi_half != 0u) {
                    [g.enc setBuffer:g.bufs[B_XH2] offset:(g.buf_off[B_XH2] + (NSUInteger)main_tok * n_out * 2) atIndex:8];
                }
                [g.enc setBytes:&epi_half length:4 atIndex:9];
                // Narrow tile's threadgroup budget: staged weights, or the epilogue tile.
                const NSUInteger t_stage = (NSUInteger)rt_k * (64 + 2) / 2;
                const NSUInteger t_epi   = g_epilogue ? (NSUInteger)8 * 64 : 0;
                [g.enc setThreadgroupMemoryLength:std::max(t_stage, t_epi) * sizeof(float) atIndex:0];
                g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
                [g.enc dispatchThreadgroups:MTLSizeMake((n_out + 63) / 64,
                                                        (tail_r + 7) / 8, 1)
                      threadsPerThreadgroup:MTLSizeMake(t_thr, 1, 1)];
                if (g_prof) { prof_end(); }
            }
            if (epi_half != 0u) {
                // G's mirror is complete in B_XH2 (both parts): the down projection can
                // read it. Elems counts the FULL batch, not the last dispatch's part.
                g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
            }
        } else {
            g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
            [g.enc dispatchThreadgroups:MTLSizeMake((n_out + 63) / 64, (n_tok + 31) / 32, 1)
                  threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];   // 16 simdgroups
            if (g_prof) { prof_end(); }
        }
    } else {
        const NSUInteger rows_per_tg = sgs * (is_q4 ? (32 / g_lanes) * nr0 : 1);
        const NSUInteger tile = is_q4 ? 4 : 1;       // TOKEN_TILE in the q4 kernel
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_DECODE); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1) / rows_per_tg,
                                                (n_tok + tile - 1) / tile, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
    if (g_prof) { prof_end(); }
    }
}

// The embedding lookup used to be Q4_0 by assumption -- it took no `wkind` at all, so a
// table with any other layout would have been dequantised by the Q4_0 unpacker. It takes
// the table index now, like matmat.
extern "C" void imparo_metal_row(uint32_t wkind, uint64_t w_off, uint32_t width,
                                 uint32_t index, float scale, uint32_t dst,
                                 uint32_t dst_off) {
    if (wkind != 1u && wkind != 2u) {
        NSLog(@"imparo metal: row got weight kind %u; only Q4_0 and Q8_0 embedding "
              @"tables have a kernel -- refusing the dispatch", wkind);
        return;
    }
    const bool is_q8 = wkind == 2u;
    const uint32_t block = is_q8 ? Q8_BLOCK_ELEMENTS : 32u;
    if (width == 0u || (width % block) != 0u) {
        NSLog(@"imparo metal: row width %u is not a multiple of the %u-value block",
              width, block);
        return;
    }
    id<MTLComputePipelineState> sel = is_q8 ? g.p_q8row : g.p_q4row;
    if (sel == nil) { NSLog(@"imparo metal: nil row pipeline (kind %u)", wkind); return; }
    haz(0, hb(dst));
    [g.enc setComputePipelineState:sel];
    [g.enc setBuffer:g.weights offset:0 atIndex:0];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&w_off length:8 atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&index length:4 atIndex:4];
    [g.enc setBytes:&scale length:4 atIndex:5];
    [g.enc setBytes:&dst_off length:4 atIndex:6];
    dispatch1(sel, width / block);
}

extern "C" void imparo_metal_rms_norm_add(uint32_t buf, uint64_t w_off, uint32_t width,
                                          float eps, uint32_t n_row, uint32_t row_stride,
                                          uint32_t base_off, uint32_t add_buf);

extern "C" void imparo_metal_rms_norm_from(uint32_t buf, uint32_t src_buf, uint64_t w_off,
                                           uint32_t width, float eps, uint32_t n_row,
                                           uint32_t row_stride, uint32_t base_off) {
    if (g_skip_cat == PC_RMSNORM) { return; }
    // Dual-write the half mirror when this norm feeds the prefill GEMM: CUR's contiguous
    // rows are exactly the GEMM's activation slice, so the separate cvt dispatch
    // disappears. Contiguous full-width rows only -- the mirror indexes must match the
    // cvt's flat layout.
    const bool xh_on = g_half_a != 0u && buf == B_CUR && n_row >= HALF_A_MIN && base_off == 0u
                    && row_stride == width && g.bufs[B_XH] != nil
                    && (uint64_t)n_row * width * 2ull <= g.sizes[B_XH]
                    // The consumers must actually take the half path: without the half
                    // pipelines the GEMM reads FLOAT activations.
                    && g.p_rt_h[g_rt_shape] != nil;
    haz(hb(src_buf), hb(buf) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_rms];
    [g.enc setBuffer:g.weights offset:0 atIndex:0];
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:1];
    [g.enc setBytes:&w_off length:8 atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&eps length:4 atIndex:4];
    [g.enc setBytes:&n_row length:4 atIndex:5];
    [g.enc setBytes:&row_stride length:4 atIndex:6];
    [g.enc setBytes:&base_off length:4 atIndex:7];
    // no addend on this path
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:8];
    const uint32_t no_add = 0;
    [g.enc setBytes:&no_add length:4 atIndex:9];
    [g.enc setBuffer:g.bufs[src_buf] offset:g.buf_off[src_buf] atIndex:10];
    if (xh_on) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:11]; }
    // 1 = write BOTH the float row and the half mirror. 2 (skip the float store) is
    // REJECTED -- it wakes the task #5 wobble at n=128; see the kernel's note.
    const uint32_t xh_flag = xh_on ? 1u : 0u;
    [g.enc setBytes:&xh_flag length:4 atIndex:12];
    // one threadgroup per row; cap threads at the row width so narrow rows do not
    // launch threads that immediately fall out of the strided loop
    // Thread count derived from the row width in FLOAT4 units, so each thread handles one
    // vector and the strided loop disappears. This is what llama.cpp does
    // (`nth = 32; while (nth < ne00_t ...) nth *= 2;` then capped to ne00_t rounded up),
    // and its norm costs 1.7 us against this engine's 7.3 for the same row width -- a fixed
    // 1024 threads makes most of them idle on a narrow row and loop on a wide one.
    const NSUInteger vec_units = (width + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    [g.enc setThreadgroupMemoryLength:(threads / 32) * sizeof(float) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (g_prof) { prof_end(); }
    if (xh_on) { g_xh_src = buf; g_xh_elems = (uint64_t)n_row * width; g_xh_buf = B_XH; }
}

extern "C" void imparo_metal_rms_norm(uint32_t buf, uint64_t w_off, uint32_t width,
                                      float eps, uint32_t n_row, uint32_t row_stride,
                                      uint32_t base_off) {
    imparo_metal_rms_norm_from(buf, buf, w_off, width, eps, n_row, row_stride, base_off);
}

// ONE DISPATCH FOR THE WHOLE BATCH, against `row`'s one per token. Returns 0 on success,
// nonzero on refusal, so a caller can keep the per-token loop rather than silently
// producing nothing.
extern "C" int imparo_metal_gather_rows(uint32_t wkind, uint64_t w_off, uint32_t width,
                                        uint32_t table_rows, float scale, uint32_t dst,
                                        uint32_t dst_off, uint32_t idx_buf,
                                        uint32_t n_rows) {
    // IMPARO_GATHER=0 refuses, which sends the caller down the per-token `row` loop. The
    // A/B lives in one binary on purpose: two builds cannot be compared without also
    // trusting that nothing else moved between them.
    static int gather_on = -1;
    if (gather_on < 0) {
        const char * v = getenv("IMPARO_GATHER");
        gather_on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    if (!gather_on) { return 1; }
    if (wkind > 2u || g.p_gather[wkind] == nil) {
        NSLog(@"imparo metal: gather_rows has no kernel for weight kind %u", wkind);
        return 1;
    }
    // Four values per thread, so the row and its destination base must both be float4
    // aligned. The quantised kinds already need width % 32 == 0, which implies it.
    const uint32_t block = wkind == 0u ? 4u : (wkind == 1u ? 32u : Q8_BLOCK_ELEMENTS);
    if (width == 0u || (width % block) != 0u || (width % 4u) != 0u
        || (dst_off % 4u) != 0u || n_rows == 0u || table_rows == 0u) {
        NSLog(@"imparo metal: gather_rows refused (kind=%u width=%u dst_off=%u n_rows=%u "
              @"table_rows=%u): needs width a multiple of %u and of 4, and a 4-aligned "
              @"destination", wkind, width, dst_off, n_rows, table_rows, block);
        return 1;
    }
    haz(hb(idx_buf), hb(dst));
    [g.enc setComputePipelineState:g.p_gather[wkind]];
    [g.enc setBuffer:g.weights offset:0 atIndex:0];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBuffer:g.bufs[idx_buf] offset:g.buf_off[idx_buf] atIndex:2];
    [g.enc setBytes:&w_off length:8 atIndex:3];
    [g.enc setBytes:&width length:4 atIndex:4];
    [g.enc setBytes:&table_rows length:4 atIndex:5];
    [g.enc setBytes:&scale length:4 atIndex:6];
    [g.enc setBytes:&dst_off length:4 atIndex:7];
    [g.enc setBytes:&n_rows length:4 atIndex:8];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
    [g.enc dispatchThreads:MTLSizeMake(width / 4u, n_rows, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

extern "C" void imparo_metal_rms_norm_add(uint32_t buf, uint64_t w_off, uint32_t width,
                                          float eps, uint32_t n_row, uint32_t row_stride,
                                          uint32_t base_off, uint32_t add_buf) {
    if (g_skip_cat == PC_RMSNORM) { return; }
    haz(hb(buf) | hb(add_buf), hb(buf));
    [g.enc setComputePipelineState:g.p_rms];
    [g.enc setBuffer:g.weights offset:0 atIndex:0];
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:1];
    [g.enc setBytes:&w_off length:8 atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&eps length:4 atIndex:4];
    [g.enc setBytes:&n_row length:4 atIndex:5];
    [g.enc setBytes:&row_stride length:4 atIndex:6];
    [g.enc setBytes:&base_off length:4 atIndex:7];
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:10];
    [g.enc setBuffer:g.bufs[add_buf] offset:g.buf_off[add_buf] atIndex:8];
    const uint32_t yes_add = 1;
    [g.enc setBytes:&yes_add length:4 atIndex:9];
    // No half mirror on this path -- and the flag must be SET, not merely left alone:
    // encoder state persists, so a stale 1 from a mirrored norm would leak in here.
    const uint32_t xh_flag = 0;
    [g.enc setBytes:&xh_flag length:4 atIndex:12];
    // Thread count derived from the row width in FLOAT4 units, so each thread handles one
    // vector and the strided loop disappears. This is what llama.cpp does
    // (`nth = 32; while (nth < ne00_t ...) nth *= 2;` then capped to ne00_t rounded up),
    // and its norm costs 1.7 us against this engine's 7.3 for the same row width -- a fixed
    // 1024 threads makes most of them idle on a narrow row and loop on a wide one.
    const NSUInteger vec_units = (width + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    [g.enc setThreadgroupMemoryLength:(threads / 32) * sizeof(float) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (g_prof) { prof_end(); }
}

extern "C" void imparo_metal_rope(uint32_t buf, uint32_t n_rot, float base,
                                  uint32_t head_dim, uint32_t n_heads,
                                  uint32_t start_pos, uint32_t n_tok,
                                  const float * freqs, uint32_t n_freqs) {
    if (g_skip_cat == PC_ROPE) { return; }
    haz(hb(buf), hb(buf));
    [g.enc setComputePipelineState:g.p_rope];
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:0];
    [g.enc setBytes:&n_rot length:4 atIndex:1];
    [g.enc setBytes:&base length:4 atIndex:2];
    [g.enc setBytes:&head_dim length:4 atIndex:3];
    [g.enc setBytes:&n_heads length:4 atIndex:4];
    [g.enc setBytes:&start_pos length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    // Small enough for setBytes: rope_dim/2 floats, 512 bytes at rope_dim 256.
    static const float kNoFreqs[1] = {1.0f};
    [g.enc setBytes:(n_freqs > 0 ? freqs : kNoFreqs)
             length:(n_freqs > 0 ? n_freqs * 4 : 4) atIndex:7];
    [g.enc setBytes:&n_freqs length:4 atIndex:8];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ROPE); }
    [g.enc dispatchThreads:MTLSizeMake(n_rot / 2, n_heads, n_tok)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    if (g_prof) { prof_end(); }
}

extern "C" void imparo_metal_kv_store(uint32_t src, uint32_t layer, uint32_t width,
                                      uint32_t start_pos, uint32_t n_tok, uint32_t is_v,
                                      uint32_t ring) {
    if (g_skip_cat == PC_KV_STORE) { return; }
    haz(hb(src), is_v ? HZ_KVV : HZ_KVK);
    // Roundtrip diagnostic: masked layers quantize+dequantize in the store; the cache
    // keeps the f16 layout and every reader stays on the plain f16 path.
    const uint32_t rt_msk = is_v ? g_kvq_mask_v : g_kvq_mask_k;
    if (g_kvq_rt != 0u && (is_v ? g_kv_type_v : g_kv_type_k) == 1u
        && layer < 32u && ((rt_msk >> layer) & 1u) && g.p_kvstore_rt != nil) {
        [g.enc setComputePipelineState:
            (kv_pt_is_ident(layer) && g.p_kvstore_rt_id != nil ? g.p_kvstore_rt_id
                                                               : g.p_kvstore_rt)];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
        [g.enc setBuffer:(is_v ? g.kv_v[layer] : g.kv_k[layer]) offset:kv_reg(layer, is_v) atIndex:1];
        [g.enc setBytes:&width length:4 atIndex:2];
        [g.enc setBytes:&start_pos length:4 atIndex:3];
        [g.enc setBytes:&n_tok length:4 atIndex:4];
        [g.enc setBytes:&ring length:4 atIndex:5];
        [g.enc setBytes:&g_kvq_rt length:4 atIndex:6];
        [g.enc setBytes:&g_kvq_rt_lo length:4 atIndex:7];
        [g.enc setBytes:&g_kvq_rt_hi length:4 atIndex:8];
        ensure_kv_pt(layer, (start_pos + n_tok + 63u) / 64u);
        [g.enc setBuffer:g.kv_pt[layer] offset:0 atIndex:9];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_KV_STORE); }
        [g.enc dispatchThreads:MTLSizeMake(width / 32u, n_tok, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (g_prof) { prof_end(); }
        return;
    }
    const uint32_t kt = kv_eff_type(layer, is_v);
    if (kt != 1u) {
        id<MTLComputePipelineState> qp = kt == 2u
            ? (kv_pt_is_ident(layer) && g.p_kvstore_q4_id != nil ? g.p_kvstore_q4_id
                                                                 : g.p_kvstore_q4)
            : (kv_pt_is_ident(layer) && g.p_kvstore_q8_id != nil ? g.p_kvstore_q8_id
                                                                 : g.p_kvstore_q8);
        [g.enc setComputePipelineState:qp];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
        [g.enc setBuffer:(is_v ? g.kv_v[layer] : g.kv_k[layer]) offset:kv_reg(layer, is_v) atIndex:1];
        [g.enc setBytes:&width length:4 atIndex:2];
        [g.enc setBytes:&start_pos length:4 atIndex:3];
        [g.enc setBytes:&n_tok length:4 atIndex:4];
        [g.enc setBytes:&ring length:4 atIndex:5];
        ensure_kv_pt(layer, (start_pos + n_tok + 63u) / 64u);
        [g.enc setBuffer:g.kv_pt[layer] offset:0 atIndex:6];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_KV_STORE); }
        [g.enc dispatchThreads:MTLSizeMake(width / 32u, n_tok, 1)
         threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (g_prof) { prof_end(); }
        return;
    }
    [g.enc setComputePipelineState:
        (kv_pt_is_ident(layer) && g.p_kvstore_id != nil ? g.p_kvstore_id
                                                        : g.p_kvstore)];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:(is_v ? g.kv_v[layer] : g.kv_k[layer]) offset:kv_reg(layer, is_v) atIndex:1];
    [g.enc setBytes:&width length:4 atIndex:2];
    [g.enc setBytes:&start_pos length:4 atIndex:3];
    [g.enc setBytes:&n_tok length:4 atIndex:4];
    [g.enc setBytes:&ring length:4 atIndex:5];
    ensure_kv_pt(layer, (start_pos + n_tok + 63u) / 64u);
    kv_pt_audit("store-f16", layer, (start_pos + n_tok + 63u) / 64u);
    [g.enc setBuffer:g.kv_pt[layer] offset:0 atIndex:6];
    // One thread per FOUR values, plus up to four for a non-divisible tail.
    const NSUInteger kvx = width / 4 + ((width % 4) ? 4 : 0);
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_KV_STORE); }
    [g.enc dispatchThreads:MTLSizeMake(kvx, n_tok, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

static void kv_dequant_now(uint32_t layer, uint32_t width, uint32_t slots,
                           uint32_t is_v, uint32_t ring) {
    const uint32_t kt = kv_eff_type(layer, is_v);
    if (getenv("IMPARO_KVDQ_LOG")) {
        NSLog(@"kv_dq layer=%u w=%u slots=%u is_v=%u kt=%u p=%p buf=%p kv=%p",
              layer, width, slots, is_v, kt, (__bridge void *)g.p_kv_dq,
              (__bridge void *)g.bufs[is_v ? B_VDQ : B_KDQ],
              (__bridge void *)(is_v ? g.kv_v[layer] : g.kv_k[layer]));
    }
    if (kt == 1u || g.p_kv_dq == nil) { return; }
    haz(is_v ? HZ_KVV : HZ_KVK, hb(is_v ? B_VDQ : B_KDQ));
    [g.enc setComputePipelineState:g.p_kv_dq];
    [g.enc setBuffer:(is_v ? g.kv_v[layer] : g.kv_k[layer]) offset:kv_reg(layer, is_v) atIndex:0];
    [g.enc setBuffer:g.bufs[is_v ? B_VDQ : B_KDQ] offset:0 atIndex:1];
    [g.enc setBytes:&width length:4 atIndex:2];
    [g.enc setBytes:&slots length:4 atIndex:3];
    [g.enc setBytes:&kt length:4 atIndex:4];
    [g.enc setBytes:&ring length:4 atIndex:5];
    // The SAME table the attention will read through, so the rows written are the
    // rows read. `slots` is a logical count, so the table must span it.
    ensure_kv_pt(layer, (slots + 63u) / 64u);
    [g.enc setBuffer:g.kv_pt[layer] offset:0 atIndex:6];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_KV_STORE); }
    [g.enc dispatchThreads:MTLSizeMake(width / 32u, slots, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

extern "C" void imparo_metal_attention(uint32_t kv_layer, uint32_t head_dim, uint32_t n_heads,
                                       uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
                                       uint32_t window, uint32_t n_tok, uint32_t max_scores,
                                       uint32_t ring) {
    // Diagnostic only; output is wrong on purpose. 1 = skip all attention,
    // 2 = skip only the hd-512 (full-attention) layers, 3 = skip only hd-256.
    if (g_skip_attn == 1u || (g_skip_attn == 2u && head_dim == 512u)
                          || (g_skip_attn == 3u && head_dim == 256u)) { return; }

    // Prefill, combined rewrite (IMPARO_QCOMB=1): staged float Q, register accumulator.
    // Scope for bisection: 1 = all layers, 2 = full-attention only, 3 = windowed only.
    // A QUANTIZED cache forces this path: the runtime dequantised the layer into the
    // half scratch (B_KDQ/B_VDQ) and only this branch binds it.
    const bool kq = kv_eff_type(kv_layer, 0) != 1u;
    const bool vq = kv_eff_type(kv_layer, 1) != 1u;
    const bool kv_quant = kq || vq;
    // DEQUANTISE HERE, not in the caller. A quantized cache makes the prefill read a
    // half scratch instead of the cache, and filling that scratch is part of reading
    // the cache -- not a step each model's graph is asked to remember. gemma4
    // remembered and LFM2 did not, so LFM2 with q4_0 or q8_0 attended over a scratch
    // nothing had ever written. Covers exactly the slots this call will read.
    if (n_tok > 1 && kv_quant) {
        const uint32_t slots = ring > 0u ? ring + 1u : start_pos + n_tok;
        if (kq) { kv_dequant_now(kv_layer, kv_width, slots, 0u, ring); }
        if (vq) { kv_dequant_now(kv_layer, kv_width, slots, 1u, ring); }
    }
    if (n_tok > 1 && (g_qcomb || kv_quant) && (head_dim == 256 || head_dim == 512)
        && g.p_attn_pre_qcomb256 != nil && g.p_attn_pre_qcomb512 != nil
        && (kv_quant || g_qcomb == 1u || (g_qcomb == 2u && window == 0u)
                          || (g_qcomb == 3u && window != 0u))) {
        // Half mirror of the attention output for the wo projection (IMPARO_HALF_A):
        // written by the kernel's store paths, so the wo GEMM's cvt pass disappears.
        const bool axh = g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                      && g.p_rt_h[g_rt_shape] != nil
                      && (uint64_t)n_tok * n_heads * head_dim * 2ull <= g.sizes[B_XH];
        // B_TMP is the DEVICE tail scratch this kernel spills fragments to, and
        // it was missing from this declaration. Under concurrent encoding two
        // layers' attention dispatches could then use it at once -- a latent
        // race that only ever fired on the rare tail, until the K-sharing
        // kernels put the window layers on the same path and it went
        // nondeterministic at n=5642 and n=16384.
        haz(hb(B_Q) | (kq ? hb(B_KDQ) : HZ_KVK)
                    | (vq ? hb(B_VDQ) : HZ_KVV),
            hb(B_ATTN) | hb(B_TMP) | (axh ? hb(B_XH) : 0u));
        // K-SHARING prefill attention: DEFAULT ON. One simdgroup carries both
        // query row groups of its position group, so each K fragment is loaded
        // once and multiplied twice -- halving the K traffic a prefill chunk
        // pulls, which is what the phase was actually short of. Measured on a
        // 16k prefill: 32687 ms -> 32365 with the hd-512 layers converted,
        // -> 32052 with the window layers too, 1.94% of the WHOLE prefill.
        // Q stages as half here (the tile does not fit otherwise), so the
        // logits move slightly and the pins are regenerated with it.
        // IMPARO_QCOMB_X=0 reverts to the QT-8 kernels.
        static const bool qx_env = [] {
            const char * e = getenv("IMPARO_QCOMB_X");
            return e == nullptr || e[0] != '0';
        }();
        // f16 KV ONLY. These kernels stage Q as half (the tile does not fit
        // otherwise), and that rounding stacks on top of a quantized cache's
        // own: the q4 agreement gate went from 0.301 to 1.025 against a 1.0
        // tolerance, with two of the top ten reordering. A quantized cache has
        // no numeric headroom to spend on a 1% prefill gain, and q4/q8 prefill
        // already leads the reference by ~6%.
        // The QT-16 shape is used only if it actually FITS this device. The limit is
        // queried, never assumed: 32768 on the M3 Pro this was tuned on, but the Metal
        // source ships to every Mac and the shape is what has to bend, not the device.
        const NSUInteger qx_floats = head_dim == 512
            ? qcomb_tg_floats(16u, 512u, g_pt_512x, true, true)
            : qcomb_tg_floats(16u, 256u, g_pt_256x, true, false);
        const bool qx_fits = qx_floats * sizeof(float) <= [g.device maxThreadgroupMemoryLength];
        const bool qx = qx_env && !kv_quant && qx_fits
                     && (head_dim == 512 ? g.p_attn_pre_qcomb512x != nil
                                         : g.p_attn_pre_qcomb256x != nil);
        if (!qx_fits && getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn qcomb QT16 hd=%u needs %lu B > device limit %lu B, using QT8\n",
                    head_dim, (unsigned long)(qx_floats * sizeof(float)),
                    (unsigned long)[g.device maxThreadgroupMemoryLength]);
        }
        [g.enc setComputePipelineState:
            head_dim == 512 ? (qx ? g.p_attn_pre_qcomb512x : g.p_attn_pre_qcomb512)
                            : (qx ? g.p_attn_pre_qcomb256x : g.p_attn_pre_qcomb256)];
        const uint32_t qt_n = qx ? 16u : 8u;
        [g.enc setBuffer:g.bufs[B_Q] offset:g.buf_off[B_Q] atIndex:0];
        if (kq) { [g.enc setBuffer:g.bufs[B_KDQ] offset:g.buf_off[B_KDQ] atIndex:1]; }
        else    { [g.enc setBuffer:g.kv_k[kv_layer] offset:kv_reg(kv_layer, false) atIndex:1]; }
        if (vq) { [g.enc setBuffer:g.bufs[B_VDQ] offset:g.buf_off[B_VDQ] atIndex:2]; }
        else    { [g.enc setBuffer:g.kv_v[kv_layer] offset:kv_reg(kv_layer, true) atIndex:2]; }
        [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:3];
        [g.enc setBytes:&head_dim length:4 atIndex:4];
        [g.enc setBytes:&n_heads length:4 atIndex:5];
        [g.enc setBytes:&n_kv length:4 atIndex:6];
        [g.enc setBytes:&kv_width length:4 atIndex:7];
        [g.enc setBytes:&start_pos length:4 atIndex:8];
        [g.enc setBytes:&window length:4 atIndex:9];
        [g.enc setBytes:&ring length:4 atIndex:10];
        [g.enc setBytes:&n_tok length:4 atIndex:11];
        [g.enc setBytes:&g_attn_stage length:4 atIndex:12];
        if (getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn branch qcomb layer=%u n_tok=%u\n", kv_layer, n_tok);
        }
        ensure_kv_pt(kv_layer, (start_pos + n_tok + 63u) / 64u);
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        // The device tail scratch. TMP is a standalone allocation, never in the arena
        // overlap, so nothing live aliases it during attention.
        [g.enc setBuffer:g.bufs[B_TMP] offset:g.buf_off[B_TMP] atIndex:13];
        if (axh) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:16]; }
        const uint32_t axh_flag = axh ? 1u : 0u;
        [g.enc setBytes:&axh_flag length:4 atIndex:17];
        // Staged Q + scores + max/sum/diagonal, plus the threadgroup tail spill at 256
        // (at 512 the spill is in device memory instead).
        const NSUInteger sfloats = head_dim == 512
            ? (qx ? qcomb_tg_floats(16u, 512u, g_pt_512x, true,  true)
                  : qcomb_tg_floats( 8u, 512u, 128u, false, true))
            : (qx ? qcomb_tg_floats(16u, 256u, g_pt_256x, true,  false)
                  : qcomb_tg_floats( 8u, 256u, 128u, false, false));
        [g.enc setThreadgroupMemoryLength:sfloats * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        // The thread count must equal the instantiation's NSG * 32: the kernel pins a
        // dim slice to each simdgroup.
        [g.enc dispatchThreadgroups:
            MTLSizeMake(n_heads, (n_tok + qt_n - 1) / qt_n, 1)
              threadsPerThreadgroup:MTLSizeMake(
                  qx ? (head_dim == 512 ? g_nsg_512x : g_nsg_256x) * 32u : 256u, 1, 1)];
        if (g_prof) { prof_end(); }
        if (axh) {
            g_xh_src = B_ATTN;
            g_xh_elems = (uint64_t)n_tok * n_heads * head_dim;
            g_xh_buf = B_XH;
        }
        return;
    }
    // Prefill: one threadgroup per QT query tokens, so a K row serves QT queries instead of
    // one. Decode keeps the per-token path, where there is only one query anyway.
    if (n_tok > 1 && g_qtile && g.p_attn_pre_qtile != nil) {
        // Taller query tile when the running output fits: acc is QT x head_dim floats, so
        // 16 queries fit at head_dim 256 (16 KB) and not at 512 (32 KB). 35 of this
        // model's 42 layers are the narrow kind, and a taller tile halves their KV reads.
        // Registers only where the fragments fit: head_dim 512 needs 64 of them, 128 floats
        // a lane, and spills. Those layers keep the tiled kernel.
        const bool tall = head_dim <= 256 && !g_attn_short && g.p_attn_pre_qtile16 != nil;
        // PT_H must match the PT the selected kernel was instantiated with.
        const uint32_t QT_H = tall ? 16 : 8, PT_H = 128;
        haz(hb(B_Q) | HZ_KVK | HZ_KVV, hb(B_ATTN));
        [g.enc setComputePipelineState:
            (tall && head_dim == 256 && g.p_attn_pre_qtile16h != nil)
              ? (g_attn_blk == 2 && g.p_attn_pre_qtile16h2 != nil ? g.p_attn_pre_qtile16h2
               : g_attn_blk == 8 && g.p_attn_pre_qtile16h8 != nil ? g.p_attn_pre_qtile16h8
                                                              : g.p_attn_pre_qtile16h)
                  : (tall ? g.p_attn_pre_qtile16 : g.p_attn_pre_qtile)];
        [g.enc setBuffer:g.bufs[B_Q] offset:g.buf_off[B_Q] atIndex:0];
        [g.enc setBuffer:g.kv_k[kv_layer] offset:kv_reg(kv_layer, false) atIndex:1];
        [g.enc setBuffer:g.kv_v[kv_layer] offset:kv_reg(kv_layer, true) atIndex:2];
        [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:3];
        [g.enc setBytes:&head_dim length:4 atIndex:4];
        [g.enc setBytes:&n_heads length:4 atIndex:5];
        [g.enc setBytes:&n_kv length:4 atIndex:6];
        [g.enc setBytes:&kv_width length:4 atIndex:7];
        [g.enc setBytes:&start_pos length:4 atIndex:8];
        [g.enc setBytes:&window length:4 atIndex:9];
        [g.enc setBytes:&ring length:4 atIndex:10];
        [g.enc setBytes:&n_tok length:4 atIndex:11];
        [g.enc setBytes:&g_attn_stage length:4 atIndex:12];
        if (getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn branch qtile layer=%u n_tok=%u\n", kv_layer, n_tok);
        }
        ensure_kv_pt(kv_layer, (start_pos + n_tok + 63u) / 64u);
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        const NSUInteger qthreads = 256;
        const uint32_t QT_R = QT_H;
        const NSUInteger sfloats =
            QT_H * PT_H + QT_H * head_dim + 2 * QT_H + (qthreads / 32);
        [g.enc setThreadgroupMemoryLength:sfloats * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, (n_tok + QT_R - 1) / QT_R, 1)
              threadsPerThreadgroup:MTLSizeMake(qthreads, 1, 1)];
        if (g_prof) { prof_end(); }
        return;
    }

    // Split the KV range when there is not enough work to fill the GPU otherwise.
    //
    // One threadgroup per (head, token) is plenty during prefill, where n_tok is large,
    // and far too few at decode, where it is 1. The split path costs an extra combine
    // dispatch, so it is used only when the direct grid would leave cores idle.
    const uint32_t n_pos = (window > 0 && start_pos + 1 > window) ? window : start_pos + 1;
    // THE DEPTH BOUNDARY IS DECIDED FIRST, and grouping yields to it. It used to be the
    // other way round by accident: want_stream carried a `!gqa` term with no stated
    // reason, written when grouping was off by default so it never mattered. Grouping now
    // defaults ON for a quantized cache, which silently disabled streaming on exactly the
    // configuration long context runs -- and the measurement says streaming is what should
    // run there:
    //
    //     15987 positions, 2 legs each      streaming   grouped score-tile
    //       q4_0                            35.85       35.15        +2.0%
    //       q8_0                            36.00       35.15        +2.4%
    //
    // `!gqa` was never a capability limit either: streaming has its own row sharing
    // (hq_grp = 2, the _g2 pipelines). It was precedence, and the precedence was wrong.
    const uint32_t hdi_e = head_dim == 128u ? 0u
                         : head_dim == 256u ? 1u
                         : head_dim == 512u ? 2u : 3u;
    // DERIVED FROM THE CACHE TYPE, and it has to be per dispatch because kvq_mask_on lets
    // the type differ per layer -- the same argument as gqa_env below. UINT32_MAX means
    // "derive"; any other value is an explicit override and is used as given.
    //
    // Score-tile's per-position advantage is CONSTANT (it computes each score once and
    // reuses it; streaming re-scales its accumulator every block). Streaming's advantage
    // GROWS with span, because score-tile holds the scores -- its threadgroup allocation
    // is min(max_scores, ceil(n_pos/slices)) floats, so residency falls as context grows,
    // while streaming's footprint is a running max, sum and O accumulator, fixed in span.
    // Constant against growing must cross. Where it crosses is what differs by cache type,
    // and it is MEASURED end to end, at --repeat 5 or better, not fitted:
    //
    //     decode tok/s          streaming   score-tile
    //       f16     143            41.2        41.1      tie
    //       f16     551            40.4        40.7      score-tile +0.7%
    //       f16    2081            39.8        40.2      score-tile +1.0%
    //       f16    5651            38.4        38.7      score-tile +0.8%
    //       f16    9969            37.3        37.1      streaming  +0.5%   <- crosses here
    //       q4_0    143            39.9        40.1      score-tile +0.5%
    //       q4_0    551            39.3        39.3      tie                <- crosses here
    //       q4_0   1061            39.3        39.2      streaming  +0.3%
    //       q4_0   2081            38.9        38.7      streaming  +0.5%
    //       q4_0   5651            38.1        37.7      streaming  +1.05%
    //       q4_0   6518            37.7        37.4      streaming  +0.8%
    //       q4_0  16384            35.85       34.9      streaming  +2.7%
    //       q8_0    143            40.1        40.1      tie
    //       q8_0   2081            39.4        39.1      streaming  +0.8%
    //       q8_0   5651            38.3        38.1      streaming  +0.5%
    //
    // The q4_0 margin grows monotonically with depth (-0.5 / 0 / +0.3 / +0.5 / +1.05 /
    // +2.7), which is the signature the constant-vs-growing argument predicts.
    //
    // QUANTIZED IS 512, NOT 0. An earlier version of this derive used 0 -- "streaming wins
    // at every depth measured" -- when the shallowest depth measured was 1061. At 143
    // tokens score-tile is 0.5% ahead, so 0 would have regressed exactly the short chat
    // turn that is the most common request there is. A crossing does not stop existing
    // below the range you sampled.
    //
    // f16 crosses between 5651 and 9969. Quantized caches cross at or below 1061 -- below
    // any depth worth a boundary -- because quantizing shifts attention from memory-bound
    // to unpack-ALU-bound, and streaming's g2 row sharing halves that term for free.
    //
    // WHY THE OLD 32768 WAS WRONG, and it was not wrong when written: the comment it came
    // from measured "score-tile beats streaming by 1.4-1.8% at 5651 on all three cache
    // types" against a streaming kernel that ran 512 threads. Dropping to 128 (see the
    // sthreads note at the dispatch) bought +10.4% at 11941 and +20.6% at 24003, which
    // turns that deficit into a surplus. The boundary was reasoned from a kernel that no
    // longer exists.
    const uint32_t stream_min_pos = g_attn_stream_min_pos != 0xFFFFFFFFu
                                  ? g_attn_stream_min_pos
                                  : (kv_quant ? 512u : 8192u);
    const bool stream_first = (g_attn_stream || n_pos >= stream_min_pos)
                           && hdi_e < 3u && window == 0u && ring == 0u && n_tok == 1u;
    const uint32_t hq_all = n_kv > 0 ? n_heads / n_kv : 1;
    // Only the group sizes that were instantiated: 2, 4 and 8. Anything else -- including the
    // n_kv == 0 fallback above, and any group that does not divide evenly -- takes the
    // ungrouped kernel rather than silently dropping heads.
    // THE GROUP SIZE IS NOT FORCED TO THE WHOLE SHARE. g_attn_gqa carries it: 0 off,
    // 1 "use the whole share", and 2/4/8 an explicit group. Grouping divides traffic by
    // the group and multiplies accumulators per thread by it, and this kernel is bound by
    // the second, so the largest group that fits is not automatically the best one.
    // DEFAULT ON FOR A QUANTIZED CACHE, OFF FOR f16, and that is not a preference -- the
    // two cases are bound by different resources. Measured at 11941 positions, each kernel
    // on its own tuned config, A/B/B/A:
    //
    //                grouped        ungrouped      grouping
    //     f16        36.4 / 36.7    36.8 / 37.0     -0.9%
    //     q8_0       36.6 / 36.2    34.9 / 34.8     +4.4%
    //     q4_0       36.5 / 36.0    34.1 / 34.6     +5.5%
    //
    // Grouping divides two terms by the group size: KV traffic, and the DEQUANT work.
    // Traffic never pays -- the path runs at 69 of 147 GB/s and the four threadgroups on
    // one KV head read it concurrently, so the cache already collapses the duplication.
    // Dequant is compute, nothing absorbs it, and the ungrouped kernel unpacks the same
    // element once per query head. f16 has no unpack step, so its term is zero and only
    // grouping's register cost remains; a quantized cache is ALU-bound on unpack and
    // grouping divides exactly what binds.
    //
    // Read the grouped column: 36.55 / 36.4 / 36.25, nearly flat. Grouping makes a
    // quantized cache almost free, while ungrouped pays 5-7% for it.
    //
    // Like the window test below, this is a RUNTIME PREDICATE, not a knob: the cache type
    // is known per dispatch and can differ per layer (kvq_mask_on), so one stored value
    // could not express it. IMPARO_ATTN_GQA overrides -- 0 off, 1 whole share, 2/4/8 an
    // explicit group.
    //
    // WHAT IS AND IS NOT ESTABLISHED ACROSS DEVICES. The WINNING term is device
    // independent: dequant is divided by the group everywhere, and f16 has no dequant
    // anywhere. The LOSING term is not -- register pressure depends on the register file
    // and achievable occupancy, and it is exactly what the f16 -0.9% measures. So the net
    // could flip on a device with much tighter registers; it would have to swing about
    // five points to do it. Measured on one machine (M3 Pro). If a device is ever found
    // where it flips, this predicate has to become a knob, because that is what a knob is
    // for.
    const uint32_t gqa_env = g_attn_gqa != 0xFFFFFFFFu ? g_attn_gqa
                           : (kv_quant ? 1u : 0u);
    const uint32_t gqa_group = gqa_env == 1u ? hq_all : gqa_env;
    const bool gqa_group_ok = gqa_group >= 2u && gqa_group <= hq_all
                           && hq_all % gqa_group == 0u;
    const int gqa_idx = !gqa_group_ok ? -1
                      : (gqa_group == 2u ? 0 : (gqa_group == 4u ? 1
                                             : (gqa_group == 8u ? 2 : -1)));
    // FULL-ATTENTION LAYERS ONLY (window == 0). Grouping trades traffic for accumulators
    // per thread, and a windowed layer has no traffic to trade: its span is capped at the
    // sliding window, so what grouping divides is a constant while the register cost is
    // paid in full. They are also the MAJORITY -- at ~450 positions the dispatch counts
    // are 245 windowed against 49 full -- so leaving them grouped meant 83% of dispatches
    // paid grouping's cost for none of its benefit, and dominated every measurement.
    const bool want_gqa = !stream_first && gqa_env != 0u && gqa_idx >= 0 && n_kv > 0
                       && n_heads % n_kv == 0
                       && window == 0u
                       && g.p_attn_dec_scoretile_gqa[gqa_idx] != nil;
    // Splits must be sized from the grid that WILL run. Grouping replaces n_heads
    // threadgroups with n_kv, so computing slices from n_heads left the grouped path with a
    // quarter of the machine busy -- which cost more than the traffic it saved.
    const uint32_t direct_tgs = (want_gqa ? n_heads / gqa_group : n_heads) * n_tok;
    uint32_t slices = 1;
    // Two reasons to split: occupancy (few threadgroups at decode) and SCORES
    // CAPACITY -- a slice's scores live in threadgroup memory (max_scores
    // floats, 32 KiB), so past that span the range MUST be sliced regardless of
    // occupancy. 256 slices x 8192 scores = 2M positions of headroom; the
    // per-model limit is the GGUF's trained context, enforced by the server.

    // The slice's scores live in threadgroup memory; the GQA-grouped kernel
    // holds HQ score rows per slice, so its per-slice budget is max_scores/HQ.
    const uint32_t score_tile = want_gqa ? std::max(1u, max_scores / gqa_group)
                                         : max_scores;
    if (g.p_attn_dec_scoretile != nil && n_tok == 1
        && (direct_tgs < g_attn_min_tgs || n_pos > score_tile)) {
        slices = (g_attn_min_tgs + direct_tgs - 1) / direct_tgs;
        const uint32_t cap_slices = (n_pos + score_tile - 1) / score_tile;
        if (cap_slices > ATTN_MAX_SLICES) {
            NSLog(@"imparo metal: attention span %u exceeds %u slices x %u scores; "
                  @"output would be wrong -- refusing the dispatch",
                  n_pos, ATTN_MAX_SLICES, max_scores);
            return;
        }
        slices = std::max(slices, cap_slices);
        slices = std::min(slices, ATTN_MAX_SLICES);
        // Never leave a slice with fewer positions than a simdgroup can score at once.
        slices = std::max(1u, std::min(slices, n_pos / 32u));
    }
    if (slices > 1) {
        // Grouped only when the heads divide evenly and the kernel exists; the grid is then
        // one threadgroup per KV head instead of per query head.
        const uint32_t hq = gqa_group;
        const bool gqa = want_gqa;
        haz(hb(B_Q) | HZ_KVK | HZ_KVV, hb(B_ATTN_PART));
        const bool ident = kv_pt_is_ident(kv_layer);
        // stream when forced on (attn_stream=1) OR past the routed crossover.
        // v7 is templated per head size and walks 32-row blocks from a page
        // base, so: only the instantiated head sizes, and FULL attention only
        // (window == 0, no ring) -- a windowed span's first row is not
        // 32-aligned and its blocks would straddle KV pages.
        const uint32_t hdi = head_dim == 128u ? 0u
                           : head_dim == 256u ? 1u
                           : head_dim == 512u ? 2u : 3u;
        const bool want_stream = (g_attn_stream || n_pos >= stream_min_pos)
                              && !gqa && hdi < 3u && window == 0u && ring == 0u;
        // GQA row sharing needs the grouped heads to share a KV head, so the
        // group size must divide the GQA ratio, and it is f16/hd-512 only.
        const bool g2_built = kv_quant ? g.p_attn_dec_stream_g2_q != nil
                                       : g.p_attn_dec_stream_g2 != nil;
        const uint32_t hq_grp = (g_attn_stream_hq == 2u && head_dim == 512u
                                 && n_kv > 0u && n_heads % 2u == 0u
                                 && (n_heads / n_kv) % 2u == 0u && g2_built)
                              ? 2u : 1u;
        id<MTLComputePipelineState> stream_sel = hdi >= 3u ? nil
            : hq_grp == 2u
            ? (kv_quant
               ? (ident && g.p_attn_dec_stream_g2_q_id != nil ? g.p_attn_dec_stream_g2_q_id
                                                          : g.p_attn_dec_stream_g2_q)
               : (ident && g.p_attn_dec_stream_g2_id != nil ? g.p_attn_dec_stream_g2_id
                                                        : g.p_attn_dec_stream_g2))
            : kv_quant
            ? (ident && g.p_attn_dec_stream_q_id[hdi] != nil ? g.p_attn_dec_stream_q_id[hdi]
                                                         : g.p_attn_dec_stream_q[hdi])
            : (ident && g.p_attn_dec_stream_id[hdi] != nil ? g.p_attn_dec_stream_id[hdi]
                                                       : g.p_attn_dec_stream[hdi]);
        id<MTLComputePipelineState> sp_sel = kv_quant
            ? (gqa ? (ident && g.p_attn_dec_scoretile_gqa_q_id[gqa_idx] != nil
                        ? g.p_attn_dec_scoretile_gqa_q_id[gqa_idx]
                        : g.p_attn_dec_scoretile_gqa_q[gqa_idx])
                   : (ident && g.p_attn_dec_scoretile_q_id != nil ? g.p_attn_dec_scoretile_q_id
                                                          : g.p_attn_dec_scoretile_q))
            : (gqa ? (ident && g.p_attn_dec_scoretile_gqa_id[gqa_idx] != nil
                        ? g.p_attn_dec_scoretile_gqa_id[gqa_idx]
                        : g.p_attn_dec_scoretile_gqa[gqa_idx])
                   : (ident && g.p_attn_dec_scoretile_id != nil ? g.p_attn_dec_scoretile_id
                                                        : g.p_attn_dec_scoretile));
        const bool streaming = want_stream && stream_sel != nil;
        if (streaming) {
            sp_sel = stream_sel;
            // Slice count for the stream kernel. The cap is a measured constant
            // (no derivation): swept on the reference M3 Pro at 16k via
            // IMPARO_ATTN_STREAM_SLICES (cold 6-layer probe, us/dispatch):
            // 16 -> 735..745, 32 -> 696..713, 64 -> 695..696, 96 -> 746,
            // 128 -> 701. Flat inside noise from 32 to 64; 32 keeps the combine
            // merging half the partials. The /4 targets ~4 blocks per simdgroup
            // stream (the reference's geometry).
            const uint32_t nblk = (n_pos + 31u) / 32u;
            slices = std::clamp(nblk / 4u, 1u, g_attn_stream_slices);
        }
        [g.enc setComputePipelineState:sp_sel];
        [g.enc setBuffer:g.bufs[B_Q] offset:g.buf_off[B_Q] atIndex:0];
        [g.enc setBuffer:g.kv_k[kv_layer] offset:kv_reg(kv_layer, false) atIndex:1];
        [g.enc setBuffer:g.kv_v[kv_layer] offset:kv_reg(kv_layer, true) atIndex:2];
        [g.enc setBuffer:g.bufs[B_ATTN_PART] offset:g.buf_off[B_ATTN_PART] atIndex:3];
        [g.enc setBytes:&head_dim length:4 atIndex:4];
        [g.enc setBytes:&n_heads length:4 atIndex:5];
        [g.enc setBytes:&n_kv length:4 atIndex:6];
        [g.enc setBytes:&kv_width length:4 atIndex:7];
        [g.enc setBytes:&start_pos length:4 atIndex:8];
        [g.enc setBytes:&window length:4 atIndex:9];
        [g.enc setBytes:&ring length:4 atIndex:10];
        [g.enc setBytes:&slices length:4 atIndex:11];
        [g.enc setBytes:&g_attn_stage length:4 atIndex:12];
        if (getenv("IMPARO_ATTN_WHICH")) {
            // The group size the DISPATCHED kernel uses, which is not one variable:
            // streaming groups by hq_grp, the score-tile kernel by hq when grouping is
            // on and not at all otherwise. This printed hq_grp unconditionally, so a
            // split dispatch reported the streaming path's group -- and IMPARO_ATTN_GQA
            // on read byte-identical to off, which is how a knob that does nothing
            // looks and how a knob that works also looks.
            // `ident` too, because it selects a DIFFERENT PIPELINE (the KV_PAGED=false
            // build, where kv_slot collapses to `return gp`). Without it a null result
            // from adding an identity variant cannot be told apart from that variant
            // never being selected.
            // kvq too: grouping now DEFAULTS from the cache type, so a group=1 reading
            // is ambiguous without it -- the default may be off, or kv_quant may just be
            // false when the flag was expected to make it true.
            fprintf(stderr,
                    "attn branch %s layer=%u hd=%u group=%u ident=%u kvq=%u slices=%u n_tok=%u\n",
                    streaming ? "stream" : "split", kv_layer, head_dim,
                    streaming ? hq_grp : (gqa ? hq : 1u), ident ? 1u : 0u,
                    kv_quant ? 1u : 0u, slices, n_tok);
        }
        ensure_kv_pt(kv_layer, (start_pos + n_tok + 63u) / 64u);
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        NSUInteger sthreads = (n_tok > 1) ? g_attn_threads_pf : g_attn_threads;
        // 128, AND IT IS MEASURED, not inherited from the reference. Racing it against
        // the score-tile width (which needed the accumulator in registers to fit at all):
        //     11941   512 threads 32.6 / 32.6    128 threads 35.9 / 36.1
        //     24003   512 threads 27.0           128 threads 32.6 / 32.5
        // More simdgroups means fewer positions each, while the per-block barriers and the
        // merge both scale with nsg.
        if (streaming) { sthreads = 128; }
        // Grouped: HQ score rows, and `red` carries per-head maxima and sums past the
        // per-simdgroup slots.
        const NSUInteger sc_mul = gqa ? hq : 1;
        // `red` also backs the V pass partials: tcount float4s past the reduction slots.
        // The GQA kernel keeps a per-simdgroup partial PER HEAD now (red[sgid*hq + j]),
        // so its reduction area is nsg*MAXHQ rather than nsg, plus 2*MAXHQ combined
        // outputs -- that is what lets its softmax use 4 barriers instead of 4*hq. The
        // `sthreads * 4` term below already covers it several times over; the floor added
        // after the `streaming` branch is a guard, so a future change to `sthreads` cannot
        // silently shrink the area under what the indices need.
        NSUInteger red_n = (((sthreads / 32) + 2 + 3) & ~3ul)
                         + (gqa ? (2 * 8) : 0) + sthreads * 4;
        if (streaming) {
            // v7 layout: Q + per-sg O + headers + per-sg score strips (32 each),
            // every region scaled by the heads this threadgroup owns.
            const NSUInteger nsg = sthreads / 32;
            red_n = hq_grp * (head_dim + nsg * head_dim + 2 * nsg + nsg * 32);
        }
        if (gqa) {
            constexpr NSUInteger MAXHQ = 8;
            const NSUInteger nsg_g = sthreads / 32;
            red_n = std::max<NSUInteger>(red_n, nsg_g * MAXHQ + 2 * MAXHQ + 4);
        }
        // Per-SLICE scores: the kernel indexes s < ns = ceil(n_pos / slices).
        // Allocating the whole span here put the GQA path (x HQ rows) at 4x the
        // 32 KiB threadgroup limit past ~2k positions -- failed dispatches,
        // garbage output. That is why grouping never worked at long context.
        const NSUInteger slice_scores =
            std::min<NSUInteger>(max_scores, (n_pos + slices - 1) / slices);
        [g.enc setThreadgroupMemoryLength:slice_scores * 4 * sc_mul atIndex:0];
        [g.enc setThreadgroupMemoryLength:red_n * sizeof(float) atIndex:1];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        const uint32_t grid_x = streaming ? n_heads / hq_grp
                              : (gqa ? n_heads / gqa_group : n_heads);
        [g.enc dispatchThreadgroups:MTLSizeMake(grid_x, n_tok, slices)
              threadsPerThreadgroup:MTLSizeMake(sthreads, 1, 1)];
        if (g_prof) { prof_end(); }

        haz(hb(B_ATTN_PART), hb(B_ATTN));
        [g.enc setComputePipelineState:g.p_attn_dec_combine];
        [g.enc setBuffer:g.bufs[B_ATTN_PART] offset:g.buf_off[B_ATTN_PART] atIndex:0];
        [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:1];
        [g.enc setBytes:&head_dim length:4 atIndex:2];
        [g.enc setBytes:&n_heads length:4 atIndex:3];
        [g.enc setBytes:&slices length:4 atIndex:4];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n_tok, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (g_prof) { prof_end(); }
        return;
    }

    haz(hb(B_Q) | HZ_KVK | HZ_KVV, hb(B_ATTN));
    [g.enc setComputePipelineState:
        kv_quant ? (kv_pt_is_ident(kv_layer) && g.p_attn_dec_direct_q_id != nil ? g.p_attn_dec_direct_q_id
                                                                     : g.p_attn_dec_direct_q)
                 : (kv_pt_is_ident(kv_layer) && g.p_attn_dec_direct_id != nil ? g.p_attn_dec_direct_id
                                                                   : g.p_attn_dec_direct)];
    [g.enc setBuffer:g.bufs[B_Q] offset:g.buf_off[B_Q] atIndex:0];
    [g.enc setBuffer:g.kv_k[kv_layer] offset:kv_reg(kv_layer, false) atIndex:1];
    [g.enc setBuffer:g.kv_v[kv_layer] offset:kv_reg(kv_layer, true) atIndex:2];
    [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:3];
    [g.enc setBytes:&head_dim length:4 atIndex:4];
    [g.enc setBytes:&n_heads length:4 atIndex:5];
    [g.enc setBytes:&n_kv length:4 atIndex:6];
    [g.enc setBytes:&kv_width length:4 atIndex:7];
    [g.enc setBytes:&start_pos length:4 atIndex:8];
    [g.enc setBytes:&window length:4 atIndex:9];
    [g.enc setBytes:&ring length:4 atIndex:10];
        if (getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn branch plain layer=%u n_tok=%u\n", kv_layer, n_tok);
        }
    ensure_kv_pt(kv_layer, (start_pos + n_tok + 63u) / 64u);
    [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
    [g.enc setThreadgroupMemoryLength:max_scores * 4 atIndex:0];
    const NSUInteger athreads = (n_tok > 1) ? g_attn_threads_pf : g_attn_threads;
    [g.enc setThreadgroupMemoryLength:(athreads / 32) * sizeof(float) atIndex:1];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n_tok, 1)
          threadsPerThreadgroup:MTLSizeMake(athreads, 1, 1)];
    if (g_prof) { prof_end(); }
}

// `kind` is EPI_GELU or EPI_SILU -- the same wire values the fused epilogue takes, so
// one enum on the Rust side describes both paths.
// LFM2's gated short convolution: outputs, then the state advance. Two dispatches with
// the encoder's hazard barrier between them -- see the kernel note for why one will not do.
extern "C" void imparo_metal_shortconv(uint32_t bcx, uint64_t w_off, uint32_t state,
                                       uint32_t state_off, uint32_t out, uint32_t width,
                                       uint32_t kern, uint32_t n_tok) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    if (g.p_shortconv == nil || g.p_shortconv_state == nil) { return; }
    haz(hb(bcx) | hb(state), hb(out));
    [g.enc setComputePipelineState:g.p_shortconv];
    [g.enc setBuffer:g.bufs[bcx] offset:g.buf_off[bcx] atIndex:0];
    [g.enc setBuffer:g.weights offset:w_off atIndex:1];
    [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
             atIndex:2];
    [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] atIndex:3];
    [g.enc setBytes:&width length:4 atIndex:4];
    [g.enc setBytes:&kern length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    dispatch1(g.p_shortconv, n_tok * width);

    // The state advance READS what the pass above read and WRITES over it, so it must not
    // start until that one has finished. Declared to the hazard tracker rather than
    // assumed: under IMPARO_CONCURRENT the encoder is unordered.
    haz(hb(bcx) | hb(state), hb(state));
    [g.enc setComputePipelineState:g.p_shortconv_state];
    const NSUInteger st_off = g.buf_off[state] + (NSUInteger)state_off * 4;
    [g.enc setBuffer:g.bufs[bcx] offset:g.buf_off[bcx] atIndex:0];
    [g.enc setBuffer:g.bufs[state] offset:st_off atIndex:1];
    [g.enc setBuffer:g.bufs[state] offset:st_off atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&kern length:4 atIndex:4];
    [g.enc setBytes:&n_tok length:4 atIndex:5];
    dispatch1(g.p_shortconv_state, width);
}

// The state as of a BOUNDARY inside this chunk, written to `snap` -- the live state is
// read and left alone. `n_tok` is how many of the chunk's tokens precede the boundary.
//
// One small dispatch instead of cutting the batch to stand on the boundary: the cut
// measured +29 ms on an 800-token prefill, against `width` threads here.
extern "C" void imparo_metal_shortconv_snapshot(uint32_t bcx, uint32_t state,
                                                uint32_t state_off, uint32_t snap,
                                                uint32_t snap_off, uint32_t width,
                                                uint32_t kern, uint32_t n_tok) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    if (g.p_shortconv_state == nil) { return; }
    haz(hb(bcx) | hb(state), hb(snap));
    [g.enc setComputePipelineState:g.p_shortconv_state];
    [g.enc setBuffer:g.bufs[bcx] offset:g.buf_off[bcx] atIndex:0];
    [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
             atIndex:1];
    [g.enc setBuffer:g.bufs[snap] offset:g.buf_off[snap] + (NSUInteger)snap_off * 4
             atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&kern length:4 atIndex:4];
    [g.enc setBytes:&n_tok length:4 atIndex:5];
    dispatch1(g.p_shortconv_state, width);
}

extern "C" void imparo_metal_act_mul(uint32_t a, uint32_t b, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a) | hb(b), hb(a));
    // Four elements per thread when the range divides by four. Buffers come
    // from newBufferWithLength, which is page aligned, and every offset used
    // here is zero, so alignment holds.
    if ((n % 4u) == 0u && g.p_actmul4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_actmul4];
        [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
        [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
        [g.enc setBytes:&n4 length:4 atIndex:2];
        dispatch1(g.p_actmul4, n4);
        return;
    }
    [g.enc setComputePipelineState:g.p_actmul];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_actmul, n);
}
extern "C" void imparo_metal_act(uint32_t a, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a), hb(a));
    if ((n % 4u) == 0u && g.p_act4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_act4];
        [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
        [g.enc setBytes:&n4 length:4 atIndex:1];
        dispatch1(g.p_act4, n4);
        return;
    }
    [g.enc setComputePipelineState:g.p_act];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBytes:&n length:4 atIndex:1];
    dispatch1(g.p_act, n);
}
// In-place blockwise Hadamard rotation of the first n floats of a buffer (n % 64 == 0).
// The quantized-KV rotation; see the kernel note.
extern "C" void imparo_metal_hadamard(uint32_t buf, uint32_t n, uint32_t nrot) {
    // IMPARO_HAD: 0 disables the rotation (A/B without rebuild), 1 normal, 2 applies it
    // twice at every site -- H*H = I, so a correct kernel makes 2 read like OFF up to
    // rounding; a broken one does not. IMPARO_HAD_LOG=1 prints each call.
    static int had_reps = -1; static bool had_log = false;
    if (had_reps < 0) {
        const char * e = getenv("IMPARO_HAD");
        had_reps = e ? atoi(e) : 1;
        had_log = getenv("IMPARO_HAD_LOG") != nullptr;
    }
    if (g.p_hadamard == nil || nrot == 0u || (nrot & (nrot - 1u)) != 0u
        || (n % nrot) != 0u || had_reps == 0) {
        if (had_log) { NSLog(@"HAD skip buf=%u n=%u nrot=%u reps=%d", buf, n, nrot, had_reps); }
        return;
    }
    if (had_log) { NSLog(@"HAD buf=%u n=%u nrot=%u reps=%d", buf, n, nrot, had_reps); }
    [g.enc setComputePipelineState:g.p_hadamard];
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:0];
    [g.enc setBytes:&n length:4 atIndex:1];
    [g.enc setBytes:&nrot length:4 atIndex:2];
    const float scale = 1.0f / sqrtf((float)nrot);
    [g.enc setBytes:&scale length:4 atIndex:3];
    [g.enc setThreadgroupMemoryLength:nrot * sizeof(float) atIndex:0];
    const NSUInteger thr = nrot < 256u ? nrot : 256u;
    for (int r = 0; r < had_reps; ++r) {
        haz(hb(buf), hb(buf));
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n / nrot, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(thr, 1, 1)];
        if (g_prof) { prof_end(); }
    }
}

extern "C" void imparo_metal_add(uint32_t a, uint32_t b, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    // Dual-write the half mirror when this add produces X for a prefill GEMM (the PLE
    // gate projection reads X right after `add(X, O)`). Prefill-scale ranges only; the
    // 64-token half-A gate on the read side makes a smaller mirror simply unused.
    const bool xh_on = g_half_a != 0u && a == B_X && (n % 4u) == 0u && n >= 65536u
                    && g.bufs[B_XH] != nil && (uint64_t)n * 2ull <= g.sizes[B_XH]
                    && g.p_rt_h[g_rt_shape] != nil;
    haz(hb(a) | hb(b), hb(a) | (xh_on ? hb(B_XH) : 0u));
    // Four elements per thread when the range divides by four. Buffers come
    // from newBufferWithLength, which is page aligned, and every offset used
    // here is zero, so alignment holds.
    if ((n % 4u) == 0u && g.p_add4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_add4];
        [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
        [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
        [g.enc setBytes:&n4 length:4 atIndex:2];
        if (xh_on) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:3]; }
        const uint32_t xh_flag = xh_on ? 1u : 0u;
        [g.enc setBytes:&xh_flag length:4 atIndex:4];
        dispatch1(g.p_add4, n4);
        if (xh_on) { g_xh_src = a; g_xh_elems = n; g_xh_buf = B_XH; }
        return;
    }
    [g.enc setComputePipelineState:g.p_add];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_add, n);
}
extern "C" void imparo_metal_mul_strided(uint32_t a, uint32_t b, uint32_t n,
                                         uint32_t b_off, uint32_t b_stride,
                                         uint32_t a_stride, uint32_t n_tok) {
    if (g_skip_cat == PC_MUL_STRIDED) { return; }
    haz(hb(a) | hb(b), hb(a));
    [g.enc setComputePipelineState:g.p_mul];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    [g.enc setBytes:&b_off length:4 atIndex:3];
    [g.enc setBytes:&b_stride length:4 atIndex:4];
    [g.enc setBytes:&a_stride length:4 atIndex:5];
    // A quarter of the threads when the vector path applies; the kernel decides by the
    // same test, so the two must agree.
    const bool vec4 = ((n | b_stride | a_stride | b_off) & 3u) == 0u;
    const NSUInteger lanes = vec4 ? (n / 4) : n;
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MUL_STRIDED); }
    [g.enc dispatchThreads:MTLSizeMake(lanes, n_tok, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}
extern "C" void imparo_metal_scale(uint32_t a, float k, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a), hb(a));
    if ((n % 4u) == 0u && g.p_scale4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_scale4];
        [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
        [g.enc setBytes:&k length:4 atIndex:1];
        [g.enc setBytes:&n4 length:4 atIndex:2];
        dispatch1(g.p_scale4, n4);
        return;
    }
    [g.enc setComputePipelineState:g.p_scale];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBytes:&k length:4 atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_scale, n);
}
extern "C" void imparo_metal_add_scale(uint32_t a, uint32_t b, float k, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a) | hb(b), hb(a));
    if ((n % 4u) == 0u && g.p_addscale4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_addscale4];
        [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
        [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
        [g.enc setBytes:&k length:4 atIndex:2];
        [g.enc setBytes:&n4 length:4 atIndex:3];
        dispatch1(g.p_addscale4, n4);
        return;
    }
    [g.enc setComputePipelineState:g.p_addscale];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&k length:4 atIndex:2];
    [g.enc setBytes:&n length:4 atIndex:3];
    dispatch1(g.p_addscale, n);
}
extern "C" void imparo_metal_copy(uint32_t dst, uint32_t src, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(src), hb(dst));
    // Four elements per thread when the range divides by four. Buffers come
    // from newBufferWithLength, which is page aligned, and every offset used
    // here is zero, so alignment holds.
    if ((n % 4u) == 0u && g.p_copy4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_copy4];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:0];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
        [g.enc setBytes:&n4 length:4 atIndex:2];
        dispatch1(g.p_copy4, n4);
        return;
    }
    [g.enc setComputePipelineState:g.p_copy];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_copy, n);
}
extern "C" void imparo_metal_softcap(uint32_t a, float cap, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a), hb(a));
    [g.enc setComputePipelineState:g.p_softcap];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBytes:&cap length:4 atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_softcap, n);
}

// Debug: copy raw bytes out of a KV cache buffer (valid after a completed cb).
extern "C" void imparo_metal_read_kv(uint32_t layer, uint32_t is_v, uint64_t off,
                                     uint8_t * dst, uint64_t n) {
    id<MTLBuffer> b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    if (b == nil) { memset(dst, 0xEE, n); return; }
    // Offsets are region-relative, the same coordinates the kernels use.
    memcpy(dst, (const uint8_t *)[b contents] + kv_reg(layer, is_v) + off, n);
}

// The Darwin reclaim protocol for freed KV block ranges (the pair Apple's own
// malloc uses): REUSABLE on release -- the pages leave phys_footprint at once
// and their content is forfeit -- and REUSE when the block is allocated again,
// BEFORE any GPU kernel writes it, so the GPU only ever touches committed
// pages. Content correctness is structural: a freed block leaves the pool
// index, and a re-allocated block's rows are written before `filled` lets any
// attention read them. Rounded INWARD to page boundaries; a sub-page remainder
// just stays resident. Call with the GPU idle.
static void kv_advise(uint32_t layer, uint32_t is_v, uint64_t off, uint64_t len,
                      int advice) {
    id<MTLBuffer> b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    if (b == nil || len == 0) { return; }
    const uint64_t page = (uint64_t)getpagesize();
    uint8_t * base = (uint8_t *)[b contents];
    const uint64_t reg = kv_reg(layer, is_v != 0u);
    uint64_t lo = (reg + off + page - 1) & ~(page - 1);
    uint64_t hi = (reg + off + len) & ~(page - 1);
    if (hi <= lo || hi > (uint64_t)[b length]) { return; }
    madvise(base + lo, (size_t)(hi - lo), advice);
}
extern "C" void imparo_metal_kv_advise_free(uint32_t layer, uint32_t is_v,
                                            uint64_t off, uint64_t len) {
    kv_advise(layer, is_v, off, len, MADV_FREE_REUSABLE);
}
extern "C" void imparo_metal_kv_advise_reuse(uint32_t layer, uint32_t is_v,
                                             uint64_t off, uint64_t len) {
    kv_advise(layer, is_v, off, len, MADV_FREE_REUSE);
}

// Restore raw bytes into a KV cache buffer (call after the GPU is idle): the KV
// pool's restore path. Shared address space, so this is the whole transfer.
extern "C" void imparo_metal_write_kv(uint32_t layer, uint32_t is_v, uint64_t off,
                                      const uint8_t * src, uint64_t n) {
    id<MTLBuffer> b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    const uint64_t reg = kv_reg(layer, is_v);
    if (b == nil || reg + off + n > [b length]) {
        NSLog(@"imparo metal: write_kv refused layer=%u is_v=%u off=%llu n=%llu len=%llu",
              layer, is_v, off, n, b == nil ? 0 : (uint64_t)[b length]);
        return;
    }
    memcpy((uint8_t *)[b contents] + reg + off, src, n);
}

// Dequantise one layer's K or V cache into the half scratch (B_KDQ / B_VDQ) covering
// `slots` slots, so the prefill attention kernels read half exactly as under f16.

// Greedy pick on the GPU; see the kernel comment. `dst` receives one uint at offset 0.
extern "C" void imparo_metal_argmax(uint32_t src, uint32_t dst, uint32_t n) {
    haz(hb(src), hb(dst));
    [g.enc setComputePipelineState:g.p_argmax];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    NSUInteger tw = std::min<NSUInteger>(1024, g.p_argmax.maxTotalThreadsPerThreadgroup);
    while (tw & (tw - 1)) { tw &= tw - 1; }   // the tree reduce needs a power of two
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
    [g.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
}
extern "C" void imparo_metal_ple_gather_combine(uint32_t proj, uint32_t tokens_buf,
                                                uint64_t w_offset, uint32_t width,
                                                float emb_scale, float comb_scale,
                                                uint32_t n_tok) {
    if (g_skip_cat == PC_PLE) { return; }
    haz(hb(proj) | hb(tokens_buf), hb(proj));
    [g.enc setComputePipelineState:g.p_ple_gather];
    [g.enc setBuffer:g.bufs[proj] offset:g.buf_off[proj] atIndex:0];
    [g.enc setBuffer:g.weights offset:0 atIndex:1];
    [g.enc setBuffer:g.bufs[tokens_buf] offset:g.buf_off[tokens_buf] atIndex:2];
    [g.enc setBytes:&w_offset length:8 atIndex:3];
    [g.enc setBytes:&width length:4 atIndex:4];
    [g.enc setBytes:&emb_scale length:4 atIndex:5];
    [g.enc setBytes:&comb_scale length:4 atIndex:6];
    [g.enc setBytes:&n_tok length:4 atIndex:7];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_PLE); }
    [g.enc dispatchThreads:MTLSizeMake(width / 4, n_tok, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

extern "C" void imparo_metal_ple_combine(uint32_t proj, uint32_t emb, uint32_t width,
                                         float emb_scale, float comb_scale, uint32_t n_tok) {
    haz(hb(proj) | hb(emb), hb(proj));
    [g.enc setComputePipelineState:g.p_ple];
    [g.enc setBuffer:g.bufs[proj] offset:g.buf_off[proj] atIndex:0];
    [g.enc setBuffer:g.bufs[emb] offset:g.buf_off[emb] atIndex:1];
    [g.enc setBytes:&width length:4 atIndex:2];
    [g.enc setBytes:&emb_scale length:4 atIndex:3];
    [g.enc setBytes:&comb_scale length:4 atIndex:4];
    [g.enc setBytes:&n_tok length:4 atIndex:5];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_PLE); }
    [g.enc dispatchThreads:MTLSizeMake(width, n_tok, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}
