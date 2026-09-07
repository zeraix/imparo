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
#import <IOKit/IOKitLib.h>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <algorithm>
#include <execinfo.h>

namespace {

extern "C" void * objc_autoreleasePoolPush(void);
extern "C" void   objc_autoreleasePoolPop(void *);

// The MSL lives in native/imparo.metal (real Metal syntax highlighting, its
// own file history); build.rs wraps it back into this raw-string constant at
// compile time. Runtime source compilation is unchanged -- no metallib, no
// extra toolchain dependency.
#include "imparo_msl.inc"
#include "mega_slots.h"   // the mega entry's slot enums, emitted by build.rs from mega_slots.rs (task #158)

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
// ONE MAP FROM HEAD DIM TO ITS QCOMB INSTANTIATION -- pipeline, row group, threadgroup
// floats and thread count together, because they must agree and were four parallel
// ternaries that could disagree. `want_x` asks for the QT-16 K-sharing shape; dims that
// have no such variant simply return the QT-8 one, so callers need no per-dim test.
struct QcombPick {
    id<MTLComputePipelineState> pipe;
    uint32_t                    qt;
    uint32_t                    threads;
    uint32_t                    pt;       // position tile, now passed TO the kernel
    NSUInteger                  sfloats;
};

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

// PAGED ATTENTION'S PAGE, in KV cells. `kv_slot` maps a logical position through the
// block table a page at a time: `(pt[gp / KV_PAGE_CELLS] * KV_PAGE_CELLS) + gp %
// KV_PAGE_CELLS`. Windowed layers do not page -- they take the ring -- but the pool
// allocates every layer on this quantum, so an extent must be a whole number of pages.
//
// DECLARED AND USED FROM ONE PLACE. This was a literal 64 in `pool_caps` and a `>> 6`
// in the shader, which cannot disagree loudly: change one and the block table maps to
// somebody else's rows with nothing failing. The value reaches MSL as a preprocessor
// macro at library-compile time, so the shader still folds it to a shift and pays
// nothing at run time -- a uniform would cost speed AND move the FP answer.
//
// NOT the byte-identity resume grid, which comes from the attention kernel's query
// group and is a different quantity (docs/kv-identity-grid.md).
//
// MUST NOT BECOME A KNOB. `imparo_kv`'s identity grid takes its value from this, so the
// disk layout is cut on it -- prefixes are hashed at multiples of it. A swept page would
// re-cut that grid on every retune and orphan every stored conversation. If a page ever
// needs to be tuned, the grid has to go back to a format constant first, with placement
// entries holding several blocks to bridge the two (docs/unified-kv-pool.md).
constexpr uint32_t KV_PAGE_CELLS = 64;
// Tokens per lane in the Q4 matmat, injected into the shader as Q4_TOKEN_TILE. THE
// DISPATCH GRID DIVIDES BY IT, so the host and the kernel must hold one number; they held
// two, written four lines apart in different languages. See imparo.metal for the
// measurement that fixes the value at 4.
constexpr uint32_t Q4_TOKEN_TILE = 4;
// The prefill GEMM's tile, injected into the shader as SG_ROWS/SG_TOKENS. THE DISPATCH
// GRID AND THREAD COUNT ARE COMPUTED FROM THESE -- they were written here as 63/64, 31/32
// and a literal 512 while the shader held its own copies. See imparo.metal for the
// sixteen-variant measurement that fixes them: every change that WIDENED a tile lost.
constexpr uint32_t SG_ROWS   = 64;
constexpr uint32_t SG_TOKENS = 32;
// One simdgroup per two 8x8 output tiles, which is what the kernel's accumulator layout
// assumes: 64x32 outputs over 16 simdgroups is two tiles each.
constexpr uint32_t SG_GEMM_THREADS = SG_ROWS * SG_TOKENS / (8 * 8 * 2) * 32;

// The NARROW tile, a DIFFERENT shape that happens to share the row count. It is
// rt_gemm<NA, NB, SGX, SGY>, whose tile is RT_ROWS = NB*8*SGX by RT_TOKENS = NA*8*SGY, so
// these are the template arguments rather than the answer -- injected into the shader,
// which instantiates imparo_rt_nb8 from them. Written as 64 and 8 on both sides, the two
// were one careless edit away from disagreeing, and substituting SG_ROWS here (same value,
// different meaning) would have hidden that rather than fixed it.
constexpr uint32_t NB8_NA = 1, NB8_NB = 4, NB8_SGX = 2, NB8_SGY = 1;
constexpr uint32_t NB8_ROWS   = NB8_NB * 8 * NB8_SGX;   // 64
constexpr uint32_t NB8_TOKENS = NB8_NA * 8 * NB8_SGY;   //  8
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
// Concurrent dispatch with per-site hazard tracking (haz()), DEFAULT ON: encode with
// MTLDispatchTypeConcurrent and insert a barrier only where a dispatch touches a buffer the
// unbarriered window already wrote (or writes one it read). Every dispatch site declares its
// read and write sets, so the independence is DERIVED, not hand-marked -- the earlier attempt
// hand-marked four pairs and measured neutral-to-worse. The reference encoder is
// concurrent-with-barriers too, and serial encoding measured +1.4 ms/token of drain bubbles
// at 16k decode. Bit-identical either way: no kernel's arithmetic changes, only whether
// independent dispatches may overlap. IMPARO_CONCURRENT=0 reverts.
//
// CONSEQUENCE FOR ATTRIBUTION, and it is easy to forget: with overlap, a skip-and-diff
// measures a stage's share of the CRITICAL PATH, not its cost. Two stages that overlap will
// not sum. Test additivity before reading any skip number as a cost.
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
// Selects among the qtile16h prefill attention kernels, which DO NOT RUN on either model
// here -- but not for one reason, and the earlier note gave only half of it. Measured with
// IMPARO_ATTN_WHICH, counting dispatch lines:
//
//   E4B  hd 256/512   84 qcomb, 0 qtile   qcomb serves it, qtile is never reached
//   LFM2 hd 64        16 qcomb, 0 qtile   qcomb serves it too: qcomb_mask_default sets
//                                         the bit of every dim the library has a kernel
//                                         for, and hd 64 has one (41b412d). Before that
//                                         the default left dims below 256 off and LFM2
//                                         took the generic qtile kernels -- never
//                                         qtile16h, which is gated on head_dim == 256.
//
// So qtile16h needs a model at head_dim 256 whose qcomb bit is CLEAR: today that means
// forcing IMPARO_QCOMB=0. Kept as an env override for exactly that, but it is no longer a
// tuner knob -- it was a stage-2 knob, so the tuner ran a full prefill and decode per
// candidate, three values, to measure a kernel that never executes.
//
// HD_C=256 in those three instantiations is therefore a FALLBACK specialisation, not the
// hardcoded dim that #67 removed elsewhere: the routing condition matches the
// instantiation exactly, and generalising it would compile more variants of a path that
// does not execute.
//
// The shape knob this kernel path SHOULD have is BLK on the live qcomb kernels, which is
// a template literal there (BLK 2 on the QT-16 variants, 4 on the QT-8 ones). See #44.
uint32_t g_attn_blk       = 4;
// BLK for the QT-8 qcomb prefill attention kernels, the ones q4/q8 prefill runs on.
// 4 was the shipped literal and leaves half the simdgroups idle (4 work units over NSG 8);
// 2 fills them. Tuner-owned so the DEVICE decides which trade wins.
uint32_t g_qcomb_blk      = 4;
// The qcomb position tile. TUNED, not derived: `qcomb_derive_pt` bounds it to what fits
// this device, and this picks inside that bound. 128 measured best on E4B q4_0 prefill
// (the derived maximum of 480 was 1.3% slower -- wider tiles cost occupancy).
uint32_t g_qcomb_pt       = 128;
// THE HEAD DIMS THIS MODEL USES, set from the plan before init and consumed when the
// library is compiled. Four slots covers any architecture in the tree (gemma4 uses two,
// LFM2 one); a model needing more compiles for the first four and the rest fall through
// to qtile, which carries a dynamic head dim and serves anything.
#define QCOMB_HD_SLOTS 4u
uint32_t g_qcomb_hds[QCOMB_HD_SLOTS] = { 0u, 0u, 0u, 0u };
// K/V row width (kv heads x head dim) per slot, 0 = not given: the kernel reads the
// stride from its uniform instead of compiling it in.
uint32_t g_attn_kvws[QCOMB_HD_SLOTS] = { 0u, 0u, 0u, 0u };
static uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
// WHICH OF THIS MODEL'S HEAD DIMS TAKE QCOMB: one bit per compiled slot, bit i for the
// model's i-th head dim. Clear means that dim falls through to qtile, whose head dim is
// dynamic and serves anything.
//
// A BITMASK AND NOT A FLOOR, because a floor assumes qcomb-worthiness is monotone in dim
// size and nothing measured that. A mask asks the question the engine actually faces --
// for each dim this model uses, which kernel? -- and the candidate set is small: one model
// here uses one dim, the other two.
//
// The candidates are DERIVED from the slot count (see the registry's `candidates` hook,
// whose own doc gives attn_min_tgs as the precedent), so no ladder of numbers is written
// anywhere: a model with three dims gets eight candidates, one with one dim gets two.
//
// EVERY COMPILED SLOT, which is what the measurements say and what the old comment here
// promised someone would eventually do.
//
// This used to set the bit only for dims >= 256 -- "a pin-preservation constant, not a
// performance rule" -- because switching LFM2 to qcomb moves its logits and needs a
// deliberate re-pin. Two things had to happen before that trade could even be priced:
//
//   1. THE MASK HAD TO REACH THE DISPATCH. It did not. This default was assigned at
//      library-compile time, which runs AFTER the stored config and the env overrides,
//      and it overwrote both -- so the knob, the config and IMPARO_QCOMB_MASK were all
//      discarded and the route never moved. The dispatch printed the proof:
//      "qcomb REFUSED hd=64 ... mask=0x0 ... pipe=yes" -- kernel compiled and ready, mask
//      erased. Fixed by g_qcomb_mask_set.
//
//   2. THE PRICE HAD TO BE MEASURED WHERE IT MATTERS. The 3.7% once recorded was on a
//      prompt where attention is ~1.5% of prefill. At 17k it is 33%, and the same routing
//      choice is worth far more (LFM2, 17121 tokens, two passes each):
//
//          qtile   28443 / 28470 ms          qcomb   24433 / 24432 ms      -14.1%
//          453 tok   523 ms                            519 ms              -0.8%
//
// E4B is untouched by this change: its dims are 256 and 512, which the old rule already
// set, so all-bits is the same mask it always had.
static uint32_t qcomb_mask_default(void) {
    uint32_t m = 0u;
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
        if (g_qcomb_hds[i] != 0u) { m |= (1u << i); }
    }
    return m;
}
uint32_t g_qcomb_mask = 0u;   // set from the default once the dims are known
// WAS THE MASK CHOSEN, OR IS IT STILL THE DEFAULT? The default can only be computed once
// the model's head dims are known, which happens at library compile -- AFTER the stored
// config and the env overrides have been applied. Assigning it unconditionally there
// overwrote both, so `attn_qcomb_mask` could be set, stored, swept and A/B'd while the
// route never moved: the dispatch always saw the default. Measured with
// IMPARO_QCOMB_MASK=1 on LFM2: "qcomb REFUSED ... mask=0x0 ... pipe=yes" -- the kernel
// was compiled and ready, and the mask asking for it had been erased.
bool g_qcomb_mask_set = false;
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
// nsg 32 also fits, at 16 matrices, and is rejected for a different reason: a
// threadgroup that takes the device's whole thread budget leaves ONE of them per core, so
// its idle simdgroups have no neighbour to cover them. Intra-threadgroup occupancy
// usually does not matter -- measured -- but that is because other threadgroups are
// resident, and at the full budget they are not. So the rule is "a second threadgroup
// must still fit", which is arithmetic against the DEVICE's limit rather than the 512
// that was written here: 512 is what max_threads/2 comes to on the machine this was
// written on, and it would have been silently wrong on any machine with a different one.
//
// Both inputs are required. A derivation missing an input does not guess -- it returns
// the compiled fallback, and says so at the one place that reports the derivation.
// IMPARO_QCOMB_NSG: measure a simdgroup count the derivation's floor excludes.
//
// The loop below starts at 8, and only its UPPER bound was ever argued ("a second
// threadgroup must still fit"). At head_dim 64, hd/8 is 8, so 1, 2 and 4 all satisfy the
// NDB-must-be-whole rule and none of them is ever tried -- a written floor standing where
// a derived candidate list belongs. Worse, it was not measurable: the only env lever,
// attn_threads_prefill, is inert at prefill under qcomb, so a sweep of it reported three
// runs of one configuration.
//
// This override goes through the SAME legality checks rather than around them; an illegal
// request is refused and named, not silently honoured, because forcing an nsg that does
// not divide hd/8 breaks the kernel's dim-slice invariant instead of tuning it. 0 derives.
uint32_t g_qcomb_nsg = 0;
// FA route: 0 = qcomb (default), 1 = the kernel_flash_attn_ext port. A/B only until it wins.
// The FA attention op is the DEFAULT prefill attention where its pipeline exists (head dim
// <= 128; LFM2). Measured end to end 2026-09-01 against qcomb: +0.2 / +0.3 / +1.8% at
// 455 / 5963 / 17123 tokens, and >= qcomb at every depth in the bench probe, so it is a
// compiled default and not a knob. IMPARO_ATTN_FA=0 is the route switch for A/Bs.
uint32_t g_attn_fa = 1;
// THE FA OP'S SIMDGROUP COUNT (knob attn_fa_nsg). A template parameter of the kernel (it
// sizes register arrays), so it is injected as a macro when the library compiles -- exactly
// as IMPARO_NSG<slot> is for qcomb -- and the tuner sets it before init through the knob's
// hook. 0 = the compiled default. The LIMIT is derived (imparo_metal_fa_nsg_mask: the
// kernel's divisibility rules and the device's thread cap); the VALUE is measured by the
// tuner (on the reference M3 Pro at hd 64, 4 beat 8 by 29%). The query tile (8 rows, one
// MMA tile) and the block (64 keys, one page) are structural, not knobs -- see the kernel.
uint32_t g_fa_nsg = 0;
uint32_t g_fa_nsg_built = 0;                       // what the library was compiled with
static const uint32_t FA_QB = 8u;                  // one 8-row query tile per threadgroup
static const uint32_t FA_CB = 64u;                 // one 64-cell page; the one-pass softmax's 2 x 32 lanes
static uint32_t fa_nsg_value(void) { return g_fa_nsg ? g_fa_nsg : 4u; }
// The nsg the last qcomb prefill dispatch used, observed rather than assumed.
uint32_t g_qcomb_nsg_live = 0;

// THE LEGALITY TEST, WRITTEN ONCE. The derivation below and the candidate mask that the
// knob registry reads both call this, so a rule change cannot reach one and not the other.
// It answers "does this nsg FIT" -- a limit. It does not answer "which nsg is FASTEST",
// and the derivation returning the first fit was the whole defect: at head_dim 64 the
// loop's floor of 8 hid nsg 4, which is legal, bit-identical and 2.9% faster at 17k.
static bool qcomb_nsg_fits(uint32_t qt, uint32_t hd, uint32_t blk, uint32_t max_acc,
                           uint32_t max_threads, uint32_t nsg) {
    if (nsg == 0u || (hd / 8u) % nsg != 0u) { return false; }   // NDB must be whole
    if (nsg * 32u * 2u > max_threads) { return false; }         // room for a second tg
    const uint32_t qrows = qt / 8u;
    const uint32_t ndb   = hd / 8u / nsg;
    const uint32_t live  = qrows * ndb + qrows * blk + 2u * qrows + 2u * blk;
    return live <= max_acc;
}

static uint32_t qcomb_derive_nsg(uint32_t qt, uint32_t hd, uint32_t blk, uint32_t max_acc,
                                 uint32_t max_threads, uint32_t fallback) {
    if (max_acc == 0u || max_threads == 0u) { return fallback; }
    const uint32_t qrows = qt / 8u;
    if (g_qcomb_nsg != 0u) {
        const uint32_t want = g_qcomb_nsg;
        const bool ok = qcomb_nsg_fits(qt, hd, blk, max_acc, max_threads, want);
        // ONE LINE PER (hd, qt), NOT ONE LINE EVER. This function is called for head
        // dims the loaded model does not use -- g_nsg_512x and g_nsg_256x are derived
        // whatever the model is -- so a single global flag reports the FIRST call's
        // verdict and then goes quiet for the dim that actually dispatches. The first
        // version of this probe did exactly that: it printed REFUSED for hd 512 while
        // silently accepting hd 64, which is the instrument lying about its own subject.
        static uint32_t seen[8][2] = {};
        static uint32_t n_seen = 0;
        bool fresh = true;
        for (uint32_t i = 0; i < n_seen; ++i) {
            if (seen[i][0] == hd && seen[i][1] == qt) { fresh = false; break; }
        }
        if (fresh && n_seen < 8u) { seen[n_seen][0] = hd; seen[n_seen][1] = qt; ++n_seen; }
        if (fresh) {
            fprintf(stderr, ok
                    ? "imparo metal: IMPARO_QCOMB_NSG=%u accepted for hd=%u qt=%u\n"
                    : "imparo metal: IMPARO_QCOMB_NSG=%u REFUSED for hd=%u qt=%u -- it "
                      "must divide hd/8, leave room for a second threadgroup, and fit the "
                      "accumulator budget; deriving instead\n",
                    want, hd, qt);
        }
        if (ok) { return want; }
    }
    // UNTUNED FALLBACK, deliberately unchanged. Still first-fit from 8, so a host with no
    // tune file runs exactly what it ran before this knob existed. The knob is what picks
    // inside the legal set; this is only what happens when nobody has.
    for (uint32_t nsg = 8u; nsg <= 32u; nsg *= 2u) {
        if (qcomb_nsg_fits(qt, hd, blk, max_acc, max_threads, nsg)) { return nsg; }
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
// Flash-decoding route for hd <= 128 (see imparo_attention_decode_fd_t): on by default;
// IMPARO_ATTN_FD=0 refuses it for A/Bs.
uint32_t g_attn_fd        = 1u;
// Keys per flash-decoding slice (knob attn_fd_chunk): each threadgroup takes one chunk, so
// the chunk sets the grid (n_kv x n_pos/chunk) and the scores it stages (HQ x chunk floats).
// The LIMIT is derived (the scores must fit the device's threadgroup memory; a chunk is at
// least one position per thread); the VALUE is the tuner's: on the reference M3 Pro its
// decode-step workload read 128 / 256 / 512 / 1024 at 2532 / 2366 / 2510 / 3546 us, so 256
// is the compiled default (8 KV heads x 64 slices at 16k keys; upstream's nwg is 32).
uint32_t g_attn_fd_chunk  = 256u;
// The vector decode kernel's regime: a single query over a span of at most this many keys
// (knob `attn_vec_max_keys`). It reads K/V once per QUERY head, so it wins while the span is
// cache-resident and loses to the K-once-per-KV-head routes past that, which split a long
// span across dozens of threadgroups where this kernel has one per query head. The tuner
// ranks the limit per model on its deep decode step: 1024 on both models of the reference
// M3 Pro (routing E4B's 512-dim layers here at 16k keys lost 11% of the step -- the
// isolated probe had read the opposite; a lone dispatch cannot stream 66 MB with eight
// threadgroups). The ladder runs past the deep span for the day a sliced form exists. Its
// threadgroup is 16 simdgroups (512 threads). In the isolated probe 8 and 16 read the same;
// INSIDE the engine, where the dispatch runs alone between the KV store and the o-projection,
// 8 cost every windowed layer +4..12 us over the previous route and 16 saves 2..12 us on
// every layer (per-command-buffer GPU timestamps, E4B at 449 tokens, 48 steps): a lone
// dispatch needs the threads in flight that overlapped probe dispatches supply for free.
// IMPARO_ATTN_VEC=0 keeps the previous route in the same binary, the A/B for this kernel
// the way IMPARO_DECODE_PIPE=0 is for the pipelined loop.
uint32_t g_attn_vec_max_keys = 1024u;
// MEGA BLOCKS (task #141). IMPARO_MEGA_FFN=1 runs the FFN block as one persistent dispatch,
// =2 the FFN + PLE block with both norms (measured ahead of the dispatch path by 3% and of
// oMLX on every leg, evidence section 26), =3 also the o_proj rows and the sandwich norm in
// front of it (section 28), =4 also the decode attention over the cache as the first phase
// -- the vec body split across the grid's threadgroups per head, for spans within the vec
// limit (section 29), =5 (the default when unset) also the input norm, the q/k/v rows, head
// norm + rope and the KV store: a decode layer is one dispatch (section 30); =0 disables
// all. Every
// threadgroup of the grid must be resident at
// once (they wait on each other), so the grid is bounded by a DERIVED limit:
//   threads_limit = gpu_cores * pipeline.maxTotalThreadsPerThreadgroup
// -- the compiler's own statement of how many of this kernel's threads one core holds, times
// the cores. Measured on an 18-core M3 Pro: 36x16 (18432 threads, exactly the limit) runs,
// 48x16 / 32x32 / 64x16 time out; the trivial-kernel all-arrive capacity is 2.7x higher, so
// the bound is conservative for lighter kernels, which is the safe side. The grid SHAPE
// inside the limit is a tuned value (knobs mega_tgs / mega_nsg): 32x16 measured 3% faster
// than 256x2 (barrier cost grows with the arriver count) and than 16x32 (1024-thread groups
// leave cores idle). The compiled fallback is nsg = half the largest legal threadgroup and
// two threadgroups per core with two cores' worth of slots left free for the other in-flight
// dispatches. IMPARO_MEGA_TGS / IMPARO_MEGA_NSG override, clamped to the limit; a barrier
// timeout sets the error word and the host disables the route.
static int mega_level(void) {
    static int lvl = -1;
    if (lvl < 0) { const char * e = getenv("IMPARO_MEGA_FFN"); lvl = (e == nullptr) ? 5 : (int)strtol(e, nullptr, 10); if (lvl < 0) { lvl = 0; } }
    return lvl;
}
static bool mega_ffn_wanted(void) { return mega_level() >= 1; }
static uint32_t g_gpu_cores = 0;            // IORegistry gpu-core-count; 0 = unreadable
static uint32_t g_mega_threads_limit = 0;   // derived at init (see above); 0 = no mega pipeline
static uint32_t g_mega_tgs = 32;
static uint32_t g_mega_nsg = 16;
static bool     g_mega_failed = false;   // a region's barrier timed out: the route is closed until the engine recovers
// FAIL-SAFE 2 (task #149): after a failure the engine rolls the step back and re-runs it on
// the dispatch path; the route then stays closed for a backoff of regions (8, 32, 128, 512
// for consecutive failures) and reopens -- a transient stall (a cold start paging weights
// under a spinning barrier) costs one re-run token, a lasting one degrades to the dispatch
// path with a bounded number of 1.2 s timeouts. A good region with the route open resets
// the streak. IMPARO_MEGA_FAIL_AT=<region ordinal> injects a failure (the test's lever).
static uint32_t g_mega_hold = 0;         // regions the route stays closed after a recovery
static uint32_t g_mega_streak = 0;       // consecutive failed regions
static uint64_t g_mega_region_ordinal = 0;
static std::vector<uint64_t> g_mega_fail_at;   // IMPARO_MEGA_FAIL_AT: region ordinals to fail (a comma list); read once
static bool     g_mega_fail_at_read = false;
static bool     g_mega_inject_pending = false;  // this region is listed: its first mega dispatch sets the sticky error
// The route is open when no region failed, no hold is running, and everything the kernel
// reads is resident by construction: with a residency set the set must be attached to the
// queue (Metal then keeps it resident during execution); a set detached for being over its
// budget means the OS pages the weights, and a persistent kernel must not spin on a barrier
// while a threadgroup waits on paging (task #154).
static bool g_rset_attached_now(void);
static inline bool mega_route_open(void) { return !g_mega_failed && g_mega_hold == 0u && g_rset_attached_now(); }
// A region retired without a failure: the hold counts down and, when it reaches zero, the
// route reopens (logged once per recovery); with the route open a good region ends the streak.
static inline void mega_good_retire(void) {
    if (g_mega_hold > 0u) {
        g_mega_hold -= 1u;
        if (g_mega_hold == 0u) { NSLog(@"imparo metal: mega route reopened after the hold at region %llu (failure streak %u)", (unsigned long long)g_mega_region_ordinal, g_mega_streak); }
    } else {
        g_mega_streak = 0u;
    }
}
// IMPARO_MEGA_DEBUG=1: the mega block records its attention phase per (dispatch, head, part)
// in the sync buffer (kernel side: MEGA_DBG) and the host scans each region after it retires.
static bool mega_dbg(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_MEGA_DEBUG"); on = (e != nullptr && e[0] == '1') ? 1 : 0; }
    return on == 1;
}
static uint32_t g_mega_region_slot = 0;   // rotates at every region end; the debug records key on it (< 4)
static uint32_t g_mega_dbg_seq = 0;       // this region's block dispatch index (debug records, < 64)
constexpr uint32_t MEGA_DBG_ITEMS = 32u, MEGA_DBG_REC = 8u, MEGA_DBG_SEQS = 64u, MEGA_DBG_SLOTS = 4u;
constexpr uint32_t Q8_TM_UNIT_ROWS_HOST = 8u;   // the shader's Q8_TM_UNIT_ROWS (rows per tile-major unit)
constexpr size_t MEGA_DBG_REC_WORDS = (size_t)MEGA_DBG_SLOTS * MEGA_DBG_SEQS * MEGA_DBG_ITEMS * MEGA_DBG_REC;
constexpr size_t MEGA_DBG_CAP_WORDS = 8u + 512u;
constexpr size_t MEGA_DBG_WORDS = MEGA_DBG_REC_WORDS + (size_t)MEGA_DBG_SLOTS * MEGA_DBG_CAP_WORDS;   // after the partial scratch
// The mega sync buffer: words [0,16) counters + error, [MEGA_SYNC_HDR_WORDS, + scratch) the
// block's own attention-partial scratch (floats), then the debug records when MEGA_DBG is on. The scratch
// is the block's memory because the activation arena overlaps its groups (G is Q's memory) and
// the attention phase writes partials while Q is still being read (task #148).
constexpr uint32_t MEGA_SYNC_HDR_WORDS = 1024u;   // mirrors MEGA_SYNC_HDR in the kernel: counters alone on their cache lines
static uint32_t g_mega_scratch_words = 0;   // capacity of the partial scratch, in floats
// IMPARO_MEGA_PROGRAM (task #153): the program form -- a token's layers recorded into runs, each
// run one persistent dispatch at one threadgroup per core (the entry indexed at runtime, the deep
// body compiled in). 1 forces it on, 0 forces it off; unset, each architecture's entry has its
// own default (measured 2026-09-07: LFM2's 30 layers are one pipeline and one run per token, a
// win; E4B's windowed and global layers are two pipelines, 14 runs per token, and the one-per-core
// grid costs more than the boundaries save at short context).
static int mega_program_env(void) {
    static int v = -2;
    if (v == -2) { const char * e = getenv("IMPARO_MEGA_PROGRAM"); v = (e == nullptr || e[0] == 0) ? -1 : (e[0] == '1' ? 1 : 0); }
    return v;
}
static bool mega_program_for(bool arch_default) {
    const int v = mega_program_env();
    return v < 0 ? arch_default : v == 1;
}
// The program pipelines are built unless the form is forced off (either entry may default to it).
static bool mega_program_build(void) { return mega_program_env() != 0; }
static const bool MEGA_PROGRAM_DEFAULT_E4B = false;
static const bool MEGA_PROGRAM_DEFAULT_LFM2 = true;
// THE PROGRAM RING (task #153): entries recorded per layer into a device ring -- 4 region slots
// (the debug slots' rotation, so a pipelined region's entries are not overwritten for four
// regions) x MEGA_PROG_CAP entries, one ring per architecture (its entry size) -- flushed as ONE
// dispatch when the run changes (pipeline, grid, threadgroup memory, rope table, position), the
// slot fills, a layer is refused, the model ends its layer loop (mega_program_end) or the region
// ends. A foreign encode while entries are pending (haz() from any other site) would reorder the
// work: the pending run is dropped and the region is failed on purpose (the failsafe re-runs it).
constexpr uint32_t MEGA_PROG_CAP = 64u;
static id<MTLBuffer> g_mega_prog = nil;   // the entry ring: one entry type for every architecture (task #158)
static id<MTLBuffer> g_prog_buf = nil;                  // the pending run's ring
static uint32_t g_prog_stride = 0u, g_prog_entry_bytes = 0u;
static uint32_t g_prog_n = 0u, g_prog_base = 0u;        // pending entries; the slot's entries flushed before them
static id<MTLComputePipelineState> g_prog_pipe = nil;
static id<MTLBuffer> g_prog_freqs = nil;
static uint32_t g_prog_tgs = 0u, g_prog_tgmem = 0u, g_prog_slot = 0u, g_prog_dbg_seq = 0u, g_prog_start_pos = 0u;
static bool g_prog_pos_set = false;   // the run's position is fixed once an entry that reads it (attention) joined
static uint64_t g_prog_rd = 0ull, g_prog_wr = 0ull;
static bool g_prog_in_flush = false, g_prog_foreign = false;
static uint32_t g_mega_kq_ty = 1u, g_mega_vq_ty = 1u;   // the K / V storage types the mega _q pipelines were built with (1 = none built)
// THE GROUPED DEEP ATTENTION BODY (task #151). Past the vec regime a layer's attention runs
// inside the block as (KV head, head sub-group, key slice) items, K/V read once per item.
// mega_attn_hq mirrors the kernel's attn_group_hq(HD) (heads per item from the register
// budget: 2 at hd 512, 4 at hd 256, 8 at hd <= 128); the kernel checks the value it is handed.
// The scratch reserved at load covers MEGA_ATTN_MAX_SLICES slices per (KV head, sub-group).
// IMPARO_MEGA_ATTN_SLICED=0 keeps the old refusal (the dispatch path's stream kernel): the A/B.
static uint32_t mega_attn_hq(uint32_t hd) { const uint32_t v = 1024u / std::max(1u, hd); return v < 8u ? v : 8u; }
constexpr uint32_t MEGA_ATTN_MAX_SLICES = 64u;
static bool mega_attn_sliced(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_MEGA_ATTN_SLICED"); on = (e != nullptr && e[0] == '0') ? 0 : 1; }
    return on == 1;
}
// A layer on the deep body is dispatched at ONE threadgroup per core: its register footprint
// (two heads' Q and accumulators per lane) is past the two-per-core admission the other
// layers run at -- measured 2026-09-07 at 32x16 on E4B's hd-512 layers: every threadgroup
// entered and finished the body, 25 of 32 had reached the barrier when the cap hit (the rest
// were waiting for a core); at 18x16 the same run is clean (design doc, constraint 8). The
// grid is per dispatch (the barrier counter is reset by the last threadgroup out), so only
// the deep layers pay it.
static uint32_t mega_deep_tgs(void) {
    const uint32_t cores = g_gpu_cores > 0u ? g_gpu_cores : 7u;
    return std::max(1u, std::min(g_mega_tgs, cores));
}
// The deep body's geometry for a layer: slices per (KV head, sub-group) so the items fill the
// deep grid, refused (0) when the items or the partials do not fit.
static uint32_t mega_attn_deep_slices(uint32_t n_heads, uint32_t n_kv, uint32_t hd) {
    const uint32_t hq = mega_attn_hq(hd);
    const uint32_t n_sub = (n_heads / n_kv + hq - 1u) / hq;
    const uint32_t tgs = mega_deep_tgs();
    if (n_kv * n_sub > tgs) { return 0u; }
    const uint32_t slices = std::max(1u, std::min(MEGA_ATTN_MAX_SLICES, tgs / (n_kv * n_sub)));
    if ((uint64_t)n_heads * slices * (hd + 2u) > g_mega_scratch_words) { return 0u; }
    return slices;
}
// The GPU core count from the IORegistry (the AGXAccelerator service's gpu-core-count).
static uint32_t gpu_core_count(void) {
    uint32_t cores = 0;
    io_iterator_t it = IO_OBJECT_NULL;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &it) == KERN_SUCCESS) {
        io_object_t svc;
        while (cores == 0 && (svc = IOIteratorNext(it)) != IO_OBJECT_NULL) {
            CFTypeRef v = IORegistryEntryCreateCFProperty(svc, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
            if (v != nullptr) {
                if (CFGetTypeID(v) == CFNumberGetTypeID()) {
                    int n = 0;
                    if (CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &n) && n > 0) { cores = (uint32_t)n; }
                }
                CFRelease(v);
            }
            IOObjectRelease(svc);
        }
        IOObjectRelease(it);
    }
    return cores;
}
// Keeps tgs * nsg * 32 within the derived limit: nsg is bounded by the pipeline, tgs by what
// is left. Returns true when a requested value had to move.
static bool mega_clamp(void) {
    bool moved = false;
    if (g_mega_threads_limit == 0u) { return false; }
    const uint32_t nsg_max = std::max(1u, std::min(32u, g_mega_threads_limit / 32u));
    if (g_mega_nsg < 1u) { g_mega_nsg = 1u; moved = true; }
    if (g_mega_nsg > nsg_max) { g_mega_nsg = nsg_max; moved = true; }
    const uint32_t tgs_max = std::max(1u, g_mega_threads_limit / (g_mega_nsg * 32u));
    if (g_mega_tgs < 1u) { g_mega_tgs = 1u; moved = true; }
    if (g_mega_tgs > tgs_max) { g_mega_tgs = tgs_max; moved = true; }
    return moved;
}
extern "C" void imparo_metal_set_mega_tgs(uint32_t v) { g_mega_tgs = v; if (mega_clamp()) { NSLog(@"imparo metal: mega_tgs %u exceeds the co-residency limit; running %u x %u", v, g_mega_tgs, g_mega_nsg); } }
extern "C" void imparo_metal_set_mega_nsg(uint32_t v) { g_mega_nsg = v; if (mega_clamp()) { NSLog(@"imparo metal: mega_nsg %u exceeds the co-residency limit; running %u x %u", v, g_mega_tgs, g_mega_nsg); } }
extern "C" uint32_t imparo_metal_mega_tgs_current(void) { return g_mega_tgs; }
extern "C" uint32_t imparo_metal_mega_nsg_current(void) { return g_mega_nsg; }
extern "C" uint32_t imparo_metal_mega_threads_limit(void) { return g_mega_threads_limit; }
extern "C" uint32_t imparo_metal_gpu_cores(void) { return g_gpu_cores; }
extern "C" uint32_t imparo_metal_mega_level(void) { return (uint32_t)mega_level(); }
constexpr uint32_t ATTN_VEC_NSG = 16u;
static bool attn_vec_enabled(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_ATTN_VEC"); on = (e != nullptr && e[0] == '0') ? 0 : 1; }
    return on != 0;
}
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
//     rt_shape                    st_gemm_shape                       wide prefill
//     gemv_max_tok                q8_gemv_max_tok                     the route boundary
constexpr uint32_t ST_GEMM_CANDIDATES = 12;
constexpr uint32_t Q8_BLOCK_ELEMENTS = 32;    // Q8_0 wire-format block width
// {ROWS, TOKENS, NSG}. MUST match the IMPARO_ST_GEMM list in imparo.metal: the host sizes
// the grid and the threadgroup from this table, so a disagreement is a wrong-size launch,
// not a slow one. SGR lives only in the kernel template.
// {ROWS, TOKENS, NSG, K_CHUNK}. K_CHUNK is the depth of one staging round in elements,
// a multiple of the 32-element Q8_0 block: 32 is llama.cpp's NK, which every ported shape
// uses; 64 halves the chunk barriers and doubles the staged tile.
constexpr uint32_t ST_GEMM_SHAPES[ST_GEMM_CANDIDATES][4] = {
    {32, 16,  4, 32}, {32, 32,  4, 32}, {64, 16,  4, 32},
    {64, 32,  4, 32}, {64, 32,  8, 32}, {64, 32, 16, 32},
    {64, 64,  8, 32}, {64, 64,  4, 32},
    {32,128,  4, 32}, {128,32,  4, 32},
    {64, 32,  4, 64}, {64, 64,  4, 64},
};
uint32_t g_q8_decode_sgs       = 4;
// Tile-major decode GEMV: the 256-byte unit fixes 8 rows per threadgroup, so only the
// simdgroup count is tunable. 8 measured decode +1.8/+2.5% at 17123 over 4 on the
// converted file (the interleaving with attention at depth is what the isolated probe
// could not see); the row-major knob keeps its own value.
uint32_t g_q8_tm_decode_sgs    = 8;
uint32_t g_q8_decode_rows_log2 = 1;   // two output rows per cooperating threadgroup
uint32_t g_q8_batch_sgs        = 8;
uint32_t g_q8_token_tile_log2  = 2;   // four tokens per narrow-batch tile
// Whether ANY prefill GEMM will read the half mirror: the Q4 register-tiled half pipeline or
// the Q8 staged half pipeline of the selected shapes. Every mirror PRODUCER (norm, add,
// attention, shortconv) gates on this; until 2026-09-02 they all gated on the Q4 pipeline
// alone, so a build without it would silently stop mirroring and every Q8 GEMM would fall
// back to its own conversion pass (review #116, D14).
static inline bool half_consumers_exist(void);
uint32_t g_st_gemm_shape       = 3;   // 64 rows x 32 tokens x 4 SG, the fork's shape
// WHICH GEMM DESIGN SERVES Q8_0 PREFILL. 0 = st_gemm (both operands staged, the tiles
// above); k >= 1 = rt_gemm<Q8> at RT_SHAPES[k - 1] (weights staged, activations through
// the register prefetch pipeline, K chunk 64). One knob, because the rt shape only
// exists when rt is the design: a separate shape knob would sweep flat under st.
// TUNER-OWNED; 0 is the compiled default an untuned host runs.
uint32_t g_q8_design           = 0;
int32_t  g_q8_design_pin       = -1;
// The SECOND prefill geometry. The pair is what the tuner picks; which of the two a given
// dispatch uses is decided from its own token count by the padding rule in `q8_matmat`,
// so there is no stored boundary to get wrong. Equal to the first shape means one shape,
// which is the untuned path exactly: neither one Apple GPU nor one model's long-prompt
// winner is a portable default.
uint32_t g_st_gemm_large_shape   = 3;
uint32_t g_q8_full_tiles         = 1;
// The fork routes a Q8 matrix multiply above eight columns; below it the token-tile
// kernel reuses each weight byte across the tile.
uint32_t g_q8_gemv_max_tok     = 8;
uint32_t g_q8_all              = 0;   // build every candidate, for sweeping only
// WHICH THREADGROUP AXIS CARRIES THE TOKEN GROUPS. Threadgroups are enumerated x
// fastest, so the x axis decides what is reused between neighbouring threadgroups:
//
//   x = output-row groups  (this engine's order)   a token group sweeps the WHOLE weight
//                                                  matrix before the next one starts, so
//                                                  the weights are streamed once per
//                                                  token group
//   x = token groups       (llama.cpp's order)     the token groups sharing one 64-row
//                                                  weight block run next to each other,
//                                                  so that block is read once
//
// The kernel already reads the axes either way (Q8_GRID_TOKEN_X); the host has to swap
// the grid EXTENTS to match, which is what this flag does. Setting the shader constant
// alone would index rows by the token-group count and write a fraction of the output.
uint32_t g_q8_grid_token_x     = 0;
// TYPED TWO-BYTE SCALE LOAD (function constant 9). llama.cpp reads a Q8_0 block's scale
// as one `half` field; this kernel rebuilds it from two byte loads, a shift and an or,
// once per block per thread in the staging loop. The constant to do it llama's way has
// been in the shader since the port with nothing to set it, because a `ushort` load needs
// two-byte alignment and nothing had checked. A block is 34 bytes and a row is a whole
// number of blocks, so the alignment question reduces to whether the binding offset is
// even -- which the dispatch can simply test.
uint32_t g_q8_typed_scale      = 0;
// Read the activation operand straight from the f16 mirror instead of copying it into
// threadgroup memory first. The point is the ALLOCATION, not the copy: it takes 64x32
// from 6144 bytes to 4096, and this kernel's own probe says it overlaps staging with
// multiplying across RESIDENT threadgroups. Only shapes 3 and 7 are instantiated.
// 0 off, 1 unstaged with no prefetch, 2 with depth 2, 3 with depth 5.
uint32_t g_q8_dev_a            = 0;
// Clamp the staging loop's edge index instead of branching on it, llama.cpp's arrangement.
uint32_t g_q8_clamp_edge       = 0;
// Scheduling fences in the MMA loop, llama.cpp's arrangement. ON by default -- this is
// the shader's own default too, so an unset build compiles them in. Setting it to 0
// through IMPARO_Q8_MMA_FENCE compiles the variant WITHOUT them, which is how the +5.1%
// short / +3.1% deep was measured and how it can be reproduced.
uint32_t g_q8_mma_fence        = 1;
// The same fence question for the Q4 rt_gemm, which has none today. Default OFF: that
// kernel already beats upstream, so it does not move without a measurement.
// The same for prefill attention, the deep leg's remaining growing term. Default off.
// Phase probe for attention: 1 no score MMA, 2 no softmax, 4 no P x V.
uint32_t g_attn_skip           = 0;
// ATTRIBUTION PROBE for the prefill GEMM (function constant 10 in the shader):
// bit 0 no multiply, bit 1 no staging, bit 2 no device weight read. Zero compiles the
// shipping pipeline, so a measured run with this unset is byte-identical to one from a
// build without the probe.
uint32_t g_q8_skip             = 0;

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
// Encoder-mode kernel time: resolved sample ticks summed per region, converted with the
// device's CPU/GPU timestamp pair taken at the region's begin and end.
double   g_prof_region_ticks = 0.0;
double   g_prof_kernel_s = 0.0;   // summed kernel durations (encoder mode only)
MTLTimestamp g_ts_cpu0 = 0, g_ts_gpu0 = 0;
// Barriers emitted by haz(). One per dispatch means the concurrent encoder buys nothing:
// every kernel's tail waits for the next one's head. Counted because the alternative is
// to reason about which projections are independent, and reasoning has been wrong here
// before -- the number says it directly.
uint64_t g_prof_barriers = 0;

// Per-dispatch GPU timestamps, tagged by kernel category.
//
// GPU-busy time is already 100% of prefill wall time, so the remaining question is not
// where the pipe stalls but which kernels own the busy time. Diffing "build a variant with
// category X removed" answers that for one category per build and perturbs the cache; the
// hardware timestamp counter answers it for all categories in one run.
enum ProfCat {
    PC_MATMAT_PREFILL = 0, PC_MATMAT_DECODE, PC_ROW, PC_RMSNORM, PC_ROPE,
    PC_KV_STORE, PC_ATTENTION, PC_ELEMENTWISE, PC_MUL_STRIDED, PC_PLE,
    // LFM2's gated short convolution is the MIXER on 22 of its 30 blocks -- the
    // counterpart of attention, not an elementwise op. It was priced as elementwise
    // because it reaches the GPU through dispatch1(), which used to stamp that
    // category on everything, so the model's dominant block kind never appeared in
    // an attribution.
    PC_SHORTCONV, PC_MEGA, PC_N
};
static const char * PROF_CAT_NAME[PC_N] = {
    "matmat_prefill", "matmat_decode", "row", "rms_norm", "rope",
    "kv_store", "attention", "elementwise", "mul_strided", "ple_combine",
    "shortconv", "mega_ffn"
};
constexpr uint32_t PROF_MAX_SAMPLES = 4096;     // 2048 dispatches per resolve: the device caps a sample buffer at 32768 B (8 B per sample); a decode token is ~784 dispatches, a 512-token prefill 1255
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
    id<MTLComputePipelineState> p_repack_q8_tm;
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
    // One pair per injected head-dim slot (blk 4 and blk 2). Nil where the slot is unused.
    id<MTLComputePipelineState> p_qcomb[4], p_qcomb_b2[4];
    // FA-shaped prefill attention: the port of llama's kernel_flash_attn_ext (IMPARO_ATTN_FA=1).
    // One pipeline per legal simdgroup count (index log2 nsg): the seated one in the engine,
    // every legal one under the tuner (IMPARO_FA_ALL). nil where not built.
    id<MTLComputePipelineState> p_fa_n[4];
    // K-sharing (QT-16) instantiation of the same slots.
    id<MTLComputePipelineState> p_qcomb_x[4];
    // The QT-16 K-sharing variants stay named for the dims they were tuned at: their NSG
    // and PT are derived per dim (g_nsg_256x, g_pt_512x, ...) and generalising THAT
    // derivation is its own change. The QT-8 rows above carry no dim.
    id<MTLComputePipelineState> p_attn_pre_qcomb256x, p_attn_pre_qcomb512x;
    id<MTLComputePipelineState> p_kvstore_q4, p_kvstore_q8, p_kvstore_rt, p_kv_dq;
    id<MTLComputePipelineState> p_hadamard;
    // Decode attention variants specialised (via function constants) for quantized KV.
    id<MTLComputePipelineState> p_attn_dec_direct_q, p_attn_dec_scoretile_q;
    id<MTLComputePipelineState> p_attn_dec_scoretile_gqa_q[3];
    // Flash-decoding for hd <= 128 (slot 0), grouped by KV head: HQ 2 / 4 / 8. nil where not built.
    id<MTLComputePipelineState> p_attn_dec_fd[3];
    id<MTLComputePipelineState> p_attn_dec_vec[2];   // vector decode kernel, one per head-dim slot
    id<MTLComputePipelineState> p_mega_ffn = nil;    // mega FFN block (IMPARO_MEGA_FFN=1)
    id<MTLComputePipelineState> p_mega_ffn_ple = nil; // the mega layer block (IMPARO_MEGA_FFN>=2), slot 0's instantiation
    id<MTLComputePipelineState> p_mega_layer[2] = { nil, nil }; // per head-dim slot (the attention phase is compile-time in HD)
    id<MTLComputePipelineState> p_mega_lfm2[2] = { nil, nil };  // the LFM2 layer block per head-dim slot (tile-major Q8, the model's activation)
    id<MTLComputePipelineState> p_mega_layer_deep[2] = { nil, nil }; // the same kernels with the deep attention body compiled in (MEGA_DEEP), one threadgroup per core
    id<MTLComputePipelineState> p_mega_lfm2_deep[2] = { nil, nil };
    // The four again for a quantized cache (KVT_K / KVT_V stamped, task #156), built only when one is configured.
    id<MTLComputePipelineState> p_mega_layer_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_layer_deep_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_lfm2_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_lfm2_deep_q[2] = { nil, nil };
    // The program form (task #153): fc 20 with the deep body compiled in; f16 and typed pairs.
    id<MTLComputePipelineState> p_mega_layer_prog[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_layer_prog_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_lfm2_prog[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_lfm2_prog_q[2] = { nil, nil };
    id<MTLBuffer> mega_sync = nil;                   // its 4 counters, shared storage, zeroed once
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
    id<MTLComputePipelineState> p_shortconv_step = nil;   // one token: conv + state shift
    std::vector<id<MTLCommandBuffer>> pending;   // flushed, awaiting accounting
    void * pool = nullptr;                      // autorelease pool for one forward pass
    // PIPELINED DECODE (docs/decode-turnaround.md): regions committed by `end_async` and
    // not yet waited on, oldest first. The host encodes decode step N+1 while step N
    // runs; `wait_outstanding` retires the oldest.
    struct Region { std::vector<id<MTLCommandBuffer>> cbs; uint32_t dbg_slot = 0; };
    std::vector<Region> outstanding;
    id<MTLComputePipelineState> p_gemv_probe;
    id<MTLComputePipelineState> p_add4, p_copy4, p_actmul4, p_act4, p_scale4;
    id<MTLComputePipelineState> p_addscale4, p_addscale;
    id<MTLComputePipelineState> p_rt[10];  // must be >= RT_CANDIDATES
    // rt_gemm<Q8>: the same shapes on Q8_0 weights, and their half-mirror twins.
    id<MTLComputePipelineState> p_rt8[10], p_rt8_h[10];
    // THE GATED PAIR (_gh): gate and up as one virtual matrix on the half-activation
    // route, one pipeline per design and shape, built beside the _h it stands in for.
    id<MTLComputePipelineState> p_rt_gh[10], p_rt8_gh[10];
    id<MTLComputePipelineState> p_rt_nb8_gh, p_rt_nb8b_gh;
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
    id<MTLComputePipelineState> p_stgemm[ST_GEMM_CANDIDATES];
    // Q8_0_TM twins (function constant 15 = true): the same kernels reading the tile-major
    // layout imparo-repack writes. Selected by the weight kind of the dispatch, never by a
    // knob, so a file's layout picks its pipeline and nothing else changes.
    id<MTLComputePipelineState> p_q8mv_tm;   // tile-major: 8 rows per threadgroup, fixed by the unit
    id<MTLComputePipelineState> p_q8mm_tile_tm[4];
    id<MTLComputePipelineState> p_stgemm_tm[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_h_tm[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_gh_tm[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_full_tm[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_full_h_tm[ST_GEMM_CANDIDATES];
    // Half-activation twins: same shapes, HALF_A=true.
    id<MTLComputePipelineState> p_stgemm_h[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_gh[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_full[ST_GEMM_CANDIDATES];
    // FULL *and* the half mirror. Without this pair, selecting FULL dropped the mirror.
    id<MTLComputePipelineState> p_stgemm_full_h[ST_GEMM_CANDIDATES];
    // Unstaged-activation twins: only shapes 3 and 7 have them, and only the generic
    // half-activation route. Built when g_q8_dev_a is on.
    id<MTLComputePipelineState> p_stgemm_da[ST_GEMM_CANDIDATES];
    // TOKEN-MAJOR GRID twins (Q8_GRID_TOKEN_X=true): the same kernels reading the two
    // threadgroup axes the other way round, so the host can put TOKEN groups on x.
    // Built only when the grid order is switched on, because the order is a whole-
    // dispatch property and nothing needs both at once.
    id<MTLComputePipelineState> p_stgemm_tx[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_stgemm_tx_h[ST_GEMM_CANDIDATES];
    id<MTLComputePipelineState> p_q8row;
    // Batched embedding gather, one per weight kind (index = the wire value).
    id<MTLComputePipelineState> p_gather[3];
    id<MTLComputePipelineState> p_q4mm, p_q4mm_pre, p_q4mm_pre_nomma, p_f32mm, p_q4row, p_rms, p_rope, p_head_norm_rope, p_kvstore,
        p_attn_dec_direct, p_actmul, p_act, p_add, p_mul, p_scale, p_copy, p_softcap, p_argmax,
        p_argmax_feed,
        p_ple;
    id<MTLComputePipelineState> p_rms_add_row = nil;   // norm + residual (+ next norm), one row per threadgroup
};

Context g;

// ============================================================================================
// WHERE THE WEIGHTS LIVE (docs/memory-tiers-and-fit.md)
//
// The common runtime places every byte range of the model in a tier -- fast (wired unified
// memory), slow (the pageable mapping) or table (row-gathered, never wired) -- and hands the
// segments over before the mapping is wrapped. This backend keeps one MTLBuffer per segment,
// each a page-rounded window onto the same mapping (zero-copy; neighbouring segments may
// share a boundary page). A dispatch binds the segment its weight offset falls in and passes
// the offset LOCAL to that buffer; a kernel never sees a file offset.
//
// Wire layout shared with `WSegWire` in imparo-metal/src/lib.rs.
struct WSegWire { uint64_t off; uint64_t bytes; uint32_t tier; uint32_t layer; };
static_assert(sizeof(WSegWire) == 24, "weight segment wire drift");
enum : uint32_t { WT_FAST = 0, WT_SLOW = 1, WT_HOST_STAGED = 2 };

struct WSeg {
    uint64_t off, bytes;      // the segment in file coordinates
    uint64_t base, len;       // the buffer's page-rounded window
    uint32_t tier, layer;
    id<MTLBuffer> buf;
};
static std::vector<WSegWire> g_placement;       // received before init; empty = one fast segment
static uint64_t g_placement_budget = 0;         // the fast tier the placement was computed against
static std::vector<WSeg> g_wsegs;               // sorted by off, disjoint in file coordinates
// Host-staged tier: row-gathered tensors get no buffer at all. The host copies a batch's rows
// out of the mapping into a small staging buffer (imparo_metal_stage_rows); the GPU
// never touches the table, so nothing of it is wired.
struct WStaged { uint64_t off, bytes; };
static std::vector<WStaged> g_wstaged;
static const uint8_t * g_map_base = nullptr;    // the weight mapping, for the host gather
static uint64_t g_map_len = 0;

extern "C" void imparo_metal_set_placement(const WSegWire * segs, uint32_t n, uint64_t budget) {
    g_placement.assign(segs, segs + n);
    g_placement_budget = budget;
}

static bool wsegs_build(const uint8_t * base, uint64_t len) {
    const uint64_t page = 16384;
    std::vector<WSegWire> plan = g_placement;
    if (plan.empty()) { plan.push_back({0, len, WT_FAST, 0}); }
    std::sort(plan.begin(), plan.end(),
              [](const WSegWire & a, const WSegWire & b) { return a.off < b.off; });
    g_wsegs.clear();
    g_wstaged.clear();
    g_map_base = base;
    g_map_len = len;
    for (const WSegWire & w : plan) {
        if (!w.bytes || w.off > len || w.bytes > len - w.off) {
            NSLog(@"imparo metal: weight segment off=%llu bytes=%llu outside the mapping (%llu)",
                  (unsigned long long)w.off, (unsigned long long)w.bytes, (unsigned long long)len);
            return false;
        }
        if (w.tier == WT_HOST_STAGED) { g_wstaged.push_back({w.off, w.bytes}); continue; }
        const uint64_t b0 = w.off & ~(page - 1);
        const uint64_t b1 = (w.off + w.bytes + page - 1) & ~(page - 1);
        id<MTLBuffer> buf = [g.device newBufferWithBytesNoCopy:(void *)(base + b0)
                                                        length:(NSUInteger)(b1 - b0)
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (buf == nil) {
            NSLog(@"imparo metal: could not wrap weight segment off=%llu bytes=%llu",
                  (unsigned long long)w.off, (unsigned long long)w.bytes);
            return false;
        }
        g_wsegs.push_back({w.off, w.bytes, b0, b1 - b0, w.tier, w.layer, buf});
    }
    return true;
}

// The segment a file offset falls in. A miss is a programming error (an offset the
// placement never covered), and it aborts rather than reading from the wrong buffer.
static const WSeg & wseg_at(uint64_t off) {
    size_t lo = 0, hi = g_wsegs.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (g_wsegs[mid].off + g_wsegs[mid].bytes <= off) { lo = mid + 1; } else { hi = mid; }
    }
    if (lo < g_wsegs.size() && g_wsegs[lo].off <= off) { return g_wsegs[lo]; }
    for (const WStaged & t : g_wstaged) {
        if (off >= t.off && off - t.off < t.bytes) {
            NSLog(@"imparo metal: weight offset %llu is in a row-gathered table; tables are "
                  "staged by the host (stage_rows), never bound", (unsigned long long)off);
            abort();
        }
    }
    NSLog(@"imparo metal: weight offset %llu is in no segment (%zu segments)",
          (unsigned long long)off, g_wsegs.size());
    void * frames[24];
    const int n = backtrace(frames, 24);
    backtrace_symbols_fd(frames, n, 2);
    abort();
}
// "No weight here": the sentinels the call sites already pass for an absent second tensor
// (UINT64_MAX for an ungated matmat, 0 for a norm without a dual). Both go through
// unchanged; 0 can never name a weight, it is the GGUF magic at the head of the file.
static constexpr uint64_t W_NONE = UINT64_MAX;
static inline bool w_absent(uint64_t off) { return off == W_NONE || off == 0; }

// Is the weight at `off` in the fast tier (wired, in the residency set)? The mega route
// refuses a layer with a slow-tier (pageable) weight: a persistent kernel must not spin on
// a barrier while a threadgroup waits on paging. Absent weights pass (never read); an
// offset in no segment refuses (the route never reaches the aborting lookup wseg_at).
static bool w_fast(uint64_t off) {
    if (w_absent(off)) { return true; }
    size_t lo = 0, hi = g_wsegs.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (g_wsegs[mid].off + g_wsegs[mid].bytes <= off) { lo = mid + 1; } else { hi = mid; }
    }
    return lo < g_wsegs.size() && g_wsegs[lo].off <= off && g_wsegs[lo].tier != WT_SLOW;
}

// Bind the segment holding `off` at argument `idx`; return the offset local to it. The
// sentinel binds a valid buffer (the kernel never reads through it) and stays a sentinel.
static uint64_t wbind(id<MTLComputeCommandEncoder> enc, uint64_t off, NSUInteger idx) {
    if (w_absent(off)) {
        [enc setBuffer:g_wsegs.front().buf offset:0 atIndex:idx];
        return off;
    }
    const WSeg & s = wseg_at(off);
    [enc setBuffer:s.buf offset:0 atIndex:idx];
    return off - s.base;
}
// A second offset a kernel reads through the SAME bound buffer: it must fall in the
// sibling's segment (gate and up of one layer do), and its local offset follows.
static uint64_t wlocal(uint64_t off, uint64_t sibling) {
    if (w_absent(off)) { return off; }
    const WSeg & s = wseg_at(sibling);
    if (off < s.off || off - s.off >= s.bytes) {
        NSLog(@"imparo metal: weight offsets %llu and %llu are in different segments",
              (unsigned long long)off, (unsigned long long)sibling);
        abort();
    }
    return off - s.base;
}

// ---- GPU residency ----------------------------------------------------------------------
// After 1-2 s without a submission the OS drops this process's GPU residency and the first
// command buffer after that waits ~80 ms while every buffer it references is made resident
// again (measured, docs/evidence/bracket/2026-09-03-idle-wake-residency.md). The fast tier
// lives in one residency set: fast-tier weight segments, the activation arena and buffers,
// the KV pool and its page tables. Membership changes as buffers come and go; the set is
// committed at the next region begin. Residency is HELD for a window past the last region
// and released after it (a resident set keeps ~4.5 GB wired for E4B): requests inside the
// window skip the wake, a server left idle gives the memory back, and the request after a
// long pause pays the wake once. A set attached to the queue is made resident by the
// queue's command buffers whether or not residency was requested (measured), so the set
// is attached only while the placement's fast tier fits the budget it was computed for.
// IMPARO_METAL_RESIDENCY_IDLE_S: the window (default 180 s, 0 = hold for the process's
// life). IMPARO_METAL_NO_RESIDENCY=1: no set at all (the A/B).
static id     g_rset = nil;
static bool   g_rset_dirty = false;
static bool   g_rset_resident = false;
static bool   g_rset_attached = false;
static bool   g_rset_in_region = false;
static double g_rset_last_use = 0.0;
static int    g_rset_idle_s = 180;
static uint64_t g_rset_bytes = 0;
static uint64_t g_rset_budget = 0;
static bool   g_rset_over_logged = false;
static std::mutex g_rset_mu;
// The idle watcher is a joinable thread stopped from an atexit handler. A detached thread
// that locked g_rset_mu every second raced static destruction at process exit: the mutex
// was gone, the lock failed with EINVAL, and the process aborted after printing its
// answer (seen once in ~200 gate runs). atexit handlers run before the destructors of
// statics constructed earlier, so the join below always precedes the mutex's teardown.
static std::condition_variable g_rset_cv;
static bool        g_rset_stop = false;
static std::thread g_rset_thread;
static void rset_stop_thread(void) {
    { std::lock_guard<std::mutex> lk(g_rset_mu); g_rset_stop = true; }
    g_rset_cv.notify_all();
    if (g_rset_thread.joinable()) { g_rset_thread.join(); }
}

static void rset_init(void) {
    if (getenv("IMPARO_METAL_NO_RESIDENCY") != NULL) { return; }
    if (const char * e = getenv("IMPARO_METAL_RESIDENCY_IDLE_S")) { g_rset_idle_s = atoi(e); }
    if (g_rset_idle_s < 0) { g_rset_idle_s = 0; }
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor * d = [[MTLResidencySetDescriptor alloc] init];
        d.label = @"imparo fast tier";
        d.initialCapacity = 512;
        NSError * err = nil;
        id<MTLResidencySet> r = [g.device newResidencySetWithDescriptor:d error:&err];
        if (r == nil) {
            NSLog(@"imparo metal: residency set unavailable (%@); per-command-buffer residency", err);
            return;
        }
        g_rset = r;
        g_rset_budget = g_placement_budget != 0 ? g_placement_budget
                                                : (uint64_t)[g.device recommendedMaxWorkingSetSize];
        NSLog(@"imparo metal: residency set on, held %d s past the last region (0 = always), "
              "fast-tier budget %llu MiB", g_rset_idle_s, (unsigned long long)(g_rset_budget >> 20));
        if (g_rset_idle_s > 0) {
            g_rset_thread = std::thread([] {
                std::unique_lock<std::mutex> lk(g_rset_mu);
                while (!g_rset_stop) {
                    g_rset_cv.wait_for(lk, std::chrono::seconds(1));
                    if (g_rset_stop) { break; }
                    if (!g_rset_resident || g_rset_in_region) { continue; }
                    if (CACurrentMediaTime() - g_rset_last_use < (double)g_rset_idle_s) { continue; }
                    if (@available(macOS 15.0, *)) { [(id<MTLResidencySet>)g_rset endResidency]; }
                    g_rset_resident = false;
                }
            });
            atexit(rset_stop_thread);
        }
    }
}
static void rset_add(id<MTLBuffer> b) {
    if (g_rset == nil || b == nil) { return; }
    if (@available(macOS 15.0, *)) {
        [(id<MTLResidencySet>)g_rset addAllocation:b];
        g_rset_bytes += (uint64_t)[b length];
        g_rset_dirty = true;
    }
}
static void rset_remove(id<MTLBuffer> b) {
    if (g_rset == nil || b == nil) { return; }
    if (@available(macOS 15.0, *)) {
        [(id<MTLResidencySet>)g_rset removeAllocation:b];
        const uint64_t n = (uint64_t)[b length];
        g_rset_bytes = n <= g_rset_bytes ? g_rset_bytes - n : 0;
        g_rset_dirty = true;
    }
}
// Region begin: commit membership changes, attach and request while under budget.
static bool g_rset_attached_now(void) { return g_rset == nil || (g_rset_attached && g_rset_resident); }
static void rset_begin(void) {
    if (g_rset == nil) { return; }
    std::lock_guard<std::mutex> lk(g_rset_mu);
    g_rset_in_region = true;
    if (@available(macOS 15.0, *)) {
        if (g_rset_dirty) {
            [(id<MTLResidencySet>)g_rset commit];
            g_rset_dirty = false;
            g_rset_resident = false;
        }
        if (g_rset_bytes > g_rset_budget) {
            // The runtime sized the fast tier; landing here means a buffer grew past it
            // (a KV pool beyond the reserve). Fall back to per-command-buffer residency.
            if (g_rset_attached) {
                [g.queue removeResidencySet:(id<MTLResidencySet>)g_rset];
                g_rset_attached = false;
            }
            if (g_rset_resident) {
                [(id<MTLResidencySet>)g_rset endResidency];
                g_rset_resident = false;
            }
            if (!g_rset_over_logged) {
                g_rset_over_logged = true;
                NSLog(@"imparo metal: fast tier over its budget (%llu of %llu MiB); "
                      "per-command-buffer residency, the OS pages the weights",
                      (unsigned long long)(g_rset_bytes >> 20),
                      (unsigned long long)(g_rset_budget >> 20));
            }
            return;
        }
        if (!g_rset_attached) {
            [g.queue addResidencySet:(id<MTLResidencySet>)g_rset];
            g_rset_attached = true;
        }
        if (!g_rset_resident) {
            [(id<MTLResidencySet>)g_rset requestResidency];
            g_rset_resident = true;
        }
    }
}
// Region end (after the wait): the idle window starts now.
static void rset_end(void) {
    if (g_rset == nil) { return; }
    std::lock_guard<std::mutex> lk(g_rset_mu);
    g_rset_in_region = false;
    g_rset_last_use = CACurrentMediaTime();
}

static inline bool half_consumers_exist(void) {
    return g.p_rt_h[g_rt_shape] != nil || g.p_stgemm_h[g_st_gemm_shape] != nil;
}

// PER-ENCODER TIMING MODE (IMPARO_PROF_ENC=1 with IMPARO_PROF=1). This device samples the
// timestamp counter only at STAGE (encoder) boundaries -- supportsCounterSampling reports
// atDispatchBoundary false on the M3 Pro -- so the per-dispatch sample pairs below never
// fire here. In this mode every dispatch gets its OWN encoder whose start/end-of-encoder
// samples bracket exactly that kernel. It serialises the dispatches (no concurrent window
// spans two encoders), so it measures kernel DURATIONS, not overlap: the sum of durations
// against the normal-mode GPU time per token is the dispatch-boundary pool.
// After a region completes: a barrier that timed out has set the error word. Disable the
// route (the dispatch path takes over from the next call) and report the region bad.
// The injected failure lands on the region's first mega dispatch (a real timeout can only
// come from one), so a region without a mega dispatch injects nothing.
static inline void mega_inject_if_pending(void) {
    if (!g_mega_inject_pending) { return; }
    g_mega_inject_pending = false;
    ((uint32_t *)[g.mega_sync contents])[15] = 0xd0000001u;
    NSLog(@"imparo metal: mega failure INJECTED at region %llu (IMPARO_MEGA_FAIL_AT)", (unsigned long long)g_mega_region_ordinal);
}
// The kernel's scratch (attention partials) is sized ONCE, at load, from the model's widest
// FFN (imparo_metal_mega_reserve); an entry that needs more refuses instead of regrowing.
// A regrow inside a region swapped the sync buffer under dispatches already encoded in it,
// and a timeout they wrote into the old buffer was invisible to the retire's check (found
// by the injection sweep, 2026-09-07, task #155): the sticky error must live in the one
// buffer every dispatch of the region runs with.
static bool g_mega_scratch_reserved = false;
static bool mega_scratch_ensure(uint32_t words) {
    if (g.mega_sync != nil && g_mega_scratch_words >= words) { return true; }
    if (g_mega_scratch_reserved) {
        static bool said = false;
        if (!said) { said = true; NSLog(@"imparo metal: mega scratch needs %u floats, %u reserved at load; the layer takes the dispatch path", words, g_mega_scratch_words); }
        return false;
    }
    const size_t total_words = (size_t)MEGA_SYNC_HDR_WORDS + words + (mega_dbg() ? MEGA_DBG_WORDS : 0u);
    id<MTLBuffer> fresh = [g.device newBufferWithLength:(NSUInteger)(total_words * 4u) options:MTLResourceStorageModeShared];
    if (fresh == nil) { return false; }
    memset([fresh contents], 0, total_words * 4u);
    rset_add(fresh);       // the kernel spins on it: resident with the rest of what it reads
    g.mega_sync = fresh;   // an in-flight region keeps its own reference to the old buffer
    g_mega_scratch_words = words;
    NSLog(@"imparo metal: mega scratch %u floats (sync buffer %zu bytes)", words, total_words * 4u);
    return true;
}

static bool mega_check_error(void) {
    if (g.mega_sync == nil) { return false; }
    const uint32_t * w = (const uint32_t *)[g.mega_sync contents];
    if (w[3] == 0u && w[15] == 0u) { return false; }
    if (!g_mega_failed) {
        if ((w[15] >> 28) == 0xeu) {
            NSLog(@"imparo metal: mega block ENTRY MISMATCH -- a threadgroup read an entry with the wrong n_tg or a zero width (err=%08x: entry %u, threadgroup %u); route disabled", w[15], (w[15] >> 16) & 0xfffu, (w[15] >> 4) & 0xfffu);
        } else {
            NSLog(@"imparo metal: mega block barrier TIMED OUT -- route disabled for this process; the region's output is invalid (err=%08x: phase %u, arrivals in it %u of %u, by threadgroup %u; phase 4095 = entry layout drift; w3=%08x; deep body entered=%08x finished=%08x [debug builds])",
                  w[15], (w[15] >> 4) & 0xfffu, (w[15] >> 16) & 0xffu, g_mega_tgs, w[15] >> 24, w[3], w[11], w[12]);
        }
    }
    g_mega_failed = true;
    return true;
}
// The engine calls this after every outstanding region is retired and the failed step's
// state is rolled back: nothing of the kernel's is in flight, so its sync words (the
// barrier counter, the exit counter, the sticky error) can be zeroed and the route held.
// Called once at load with the model's widest FFN and its attention geometry (the most query
// heads, the widest head): the kernel's scratch is allocated here, before any region, and never
// regrown (see mega_scratch_ensure). It holds the FFN staging row and the deep attention body's
// partials, n_heads x MEGA_ATTN_MAX_SLICES x (hd + 2) floats.
extern "C" int imparo_metal_mega_reserve(uint32_t n_mid, uint32_t attn_heads, uint32_t attn_hd) {
    if (!mega_ffn_wanted()) { return 0; }
    const uint64_t attn_words = (uint64_t)attn_heads * MEGA_ATTN_MAX_SLICES * (attn_hd + 2u);
    const uint32_t words = (uint32_t)std::max<uint64_t>(n_mid, attn_words);
    const bool ok = mega_scratch_ensure(words);
    g_mega_scratch_reserved = true;
    return ok ? 0 : 1;
}
extern "C" int imparo_metal_mega_recover(void) {
    if (!g.outstanding.empty()) { return 1; }
    if (g.mega_sync != nil) { memset([g.mega_sync contents], 0, 16u * sizeof(uint32_t)); }
    // One failure per recovery, whatever number of drained regions read the sticky word.
    g_mega_streak += 1u;
    const uint32_t streak = g_mega_streak;
    g_mega_hold = std::min<uint32_t>(512u, 8u << (2u * std::min<uint32_t>(streak - 1u, 3u)));
    g_mega_failed = false;
    NSLog(@"imparo metal: mega route recovered (failure %u): the dispatch path takes the next %u regions", streak, g_mega_hold);
    return 0;
}

// Scan one region's debug records (MEGA_DBG): a record is live when its PA word 3 is set.
// Reports a combine that read something other than what the attention phase wrote, a
// non-finite input, or a non-finite / empty output; then clears the slot for its next use.
static uint32_t g_mega_dbg_cap_layer = 0u;   // the layer of the region's first mega dispatch (the KV capture's)
// One cache value dequantised on the host, for the MEGA_DBG KV capture's comparison.
static float kv_host_val(uint32_t layer, bool is_v, uint32_t kvt, uint32_t kvw, uint32_t ps, uint32_t head_off, uint32_t i) {
    id<MTLBuffer> b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    if (b == nil) { return NAN; }
    const std::vector<uint64_t> & regs = is_v ? g.kv_reg_v : g.kv_reg_k;
    const uint8_t * base = (const uint8_t *)[b contents] + (layer < regs.size() ? regs[layer] : 0ull);
    if (kvt == 1u) {
        _Float16 h; memcpy(&h, base + ((size_t)ps * kvw + head_off + i) * 2u, 2); return (float)h;
    }
    const size_t bs = kvt == 2u ? 18u : 34u;
    const uint8_t * blk = base + ((size_t)ps * (kvw / 32u) + (head_off + i) / 32u) * bs;
    _Float16 dh; memcpy(&dh, blk, 2); const float d = (float)dh;
    const uint32_t j = i % 32u;
    if (kvt == 2u) { const int q = (int)((blk[2 + (j & 15u)] >> ((j >> 4) * 4u)) & 0xFu) - 8; return (float)q * d; }
    return (float)(int8_t)blk[2 + j] * d;
}
static void mega_dbg_scan(uint32_t slot) {
    if (!mega_dbg() || g.mega_sync == nil) { return; }
    uint32_t * w = (uint32_t *)[g.mega_sync contents];
    uint32_t * base = w + MEGA_SYNC_HDR_WORDS + g_mega_scratch_words + (size_t)slot * MEGA_DBG_SEQS * MEGA_DBG_ITEMS * MEGA_DBG_REC;
    uint32_t seen = 0, reported = 0;
    for (uint32_t seq = 0; seq < MEGA_DBG_SEQS; ++seq) {
        for (uint32_t item = 0; item < MEGA_DBG_ITEMS; ++item) {
            const uint32_t * r = base + ((size_t)seq * MEGA_DBG_ITEMS + item) * MEGA_DBG_REC;
            if (r[3] == 0u) { continue; }
            seen += 1;
            const uint32_t split = std::max(1u, (r[3] >> 16) & 0xffu), heads = r[3] >> 24, n = r[3] & 0xffffu;
            const bool mismatch = (r[0] != r[4]) || (r[1] != r[5]);
            if (mismatch || r[2] != 0u || r[7] != 0u) {
                float pm, pl, cm, cl, L;
                memcpy(&pm, &r[0], 4); memcpy(&pl, &r[1], 4); memcpy(&cm, &r[4], 4); memcpy(&cl, &r[5], 4); memcpy(&L, &r[6], 4);
                NSLog(@"imparo metal: mega DEBUG slot=%u seq=%u head=%u part=%u n=%u split=%u heads=%u | PA wrote M=%g L=%g (bits %08x %08x) in_bad=%u | combine read M=%g L=%g (bits %08x %08x) L_total=%g flags=%u%s",
                      slot, seq, item / split, item % split, n, split, heads,
                      pm, pl, r[0], r[1], r[2], cm, cl, r[4], r[5], L, r[7], mismatch ? " MISMATCH" : "");
                reported += 1;
            }
        }
    }
    static bool live_said = false;
    if (!live_said && seen != 0u) { live_said = true; NSLog(@"imparo metal: mega DEBUG scan live: slot=%u records=%u (first region with records)", slot, seen); }
    if (reported != 0u) { NSLog(@"imparo metal: mega DEBUG slot=%u records=%u reported=%u", slot, seen, reported); }
    memset(base, 0, (size_t)MEGA_DBG_SEQS * MEGA_DBG_ITEMS * MEGA_DBG_REC * 4u);
    uint32_t * cap = w + MEGA_SYNC_HDR_WORDS + g_mega_scratch_words + MEGA_DBG_REC_WORDS + (size_t)slot * MEGA_DBG_CAP_WORDS;
    if (cap[0] == 2u) {
        // The KV capture (task #156 probe): the kernel's typed-loader values for the span's
        // first position next to the host's own dequantisation of the same cache bytes.
        const uint32_t hd = std::min(cap[6], 256u), ps = cap[2], kvw = cap[3], kvh = cap[7];
        const uint32_t layer = g_mega_dbg_cap_layer;
        NSMutableString * ks = [NSMutableString new]; NSMutableString * vs = [NSMutableString new];
        NSMutableString * hk = [NSMutableString new]; NSMutableString * hv = [NSMutableString new];
        for (uint32_t i = 0; i < 8u && i < hd; ++i) {
            float kf, vf; memcpy(&kf, &cap[8 + i], 4); memcpy(&vf, &cap[8 + hd + i], 4);
            [ks appendFormat:@" %g", kf]; [vs appendFormat:@" %g", vf];
            [hk appendFormat:@" %g", kv_host_val(layer, false, kv_eff_type(layer, 0), kvw, ps, kvh * hd, i)];
            [hv appendFormat:@" %g", kv_host_val(layer, true, kv_eff_type(layer, 1), kvw, ps, kvh * hd, i)];
        }
        NSLog(@"imparo metal: mega DEBUG KV CAPTURE slot=%u seq=%u layer=%u ps=%u kvw=%u kvh=%u hd=%u n=%u tgid=%u kt=%u vt=%u | kernel K:%@ | host K:%@ | kernel V:%@ | host V:%@",
              slot, cap[1], layer, ps, kvw, kvh, hd, cap[4], cap[5], kv_eff_type(layer, 0), kv_eff_type(layer, 1), ks, hk, vs, hv);
        // Neighbouring rows, for the comparison across cache types: slots ps..ps+3, dims 0..16.
        for (uint32_t q = 0; q < 4u; ++q) {
            NSMutableString * rk = [NSMutableString new]; NSMutableString * rv = [NSMutableString new];
            for (uint32_t i = 0; i < 16u; ++i) {
                [rk appendFormat:@" %.4g", kv_host_val(layer, false, kv_eff_type(layer, 0), kvw, ps + q, kvh * hd, i)];
                [rv appendFormat:@" %.4g", kv_host_val(layer, true, kv_eff_type(layer, 1), kvw, ps + q, kvh * hd, i)];
            }
            NSLog(@"imparo metal: mega DEBUG KV ROW layer=%u slot=%u K:%@ | V:%@", layer, ps + q, rk, rv);
        }
        memset(cap, 0, MEGA_DBG_CAP_WORDS * 4u);
    }
    if (cap[0] != 0u) {
        const uint32_t hd = std::min(cap[6], 512u);
        uint32_t nbad = 0; NSMutableString * bad = [NSMutableString new]; NSMutableString * good = [NSMutableString new];
        uint32_t ngood = 0;
        for (uint32_t i = 0; i < hd; ++i) {
            const uint32_t bits = cap[8 + i];
            const bool nf = (bits & 0x7f800000u) == 0x7f800000u;
            float v; memcpy(&v, &bits, 4);
            if (nf) { if (nbad < 24) { [bad appendFormat:@" %u:%08x", i, bits]; } nbad += 1; }
            else if (ngood < 8) { [good appendFormat:@" %u:%g", i, v]; ngood += 1; }
        }
        NSLog(@"imparo metal: mega DEBUG Q CAPTURE slot=%u seq=%u head=%u part=%u n=%u tgid=%u hd=%u non_finite=%u of %u | bad(idx:bits)%@ | first finite(idx:val)%@",
              slot, cap[1], cap[2], cap[3], cap[4], cap[5], hd, nbad, hd, bad, good);
        // Runs of non-finite entries: where the bad region starts and ends.
        uint32_t run_start = 0xffffffffu; NSMutableString * runs = [NSMutableString new];
        for (uint32_t i = 0; i <= hd; ++i) {
            const bool nf = i < hd && ((cap[8 + i] & 0x7f800000u) == 0x7f800000u);
            if (nf && run_start == 0xffffffffu) { run_start = i; }
            if (!nf && run_start != 0xffffffffu) { [runs appendFormat:@" [%u,%u)", run_start, i]; run_start = 0xffffffffu; }
        }
        NSLog(@"imparo metal: mega DEBUG Q CAPTURE runs:%@", runs);
        memset(cap, 0, MEGA_DBG_CAP_WORDS * 4u);
    }
}

static bool prof_enc_mode(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_PROF_ENC"); on = (e != nullptr && e[0] == '1') ? 1 : 0; }
    return on == 1;
}
static id<MTLComputeCommandEncoder> new_encoder(void) {
    if (g_prof && prof_enc_mode() && g_prof_sbuf != nil && g_prof_pairs * 2 + 1 < PROF_MAX_SAMPLES) {
        MTLComputePassDescriptor * pd = [MTLComputePassDescriptor computePassDescriptor];
        pd.dispatchType = g_concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
        MTLComputePassSampleBufferAttachmentDescriptor * a = pd.sampleBufferAttachments[0];
        a.sampleBuffer = g_prof_sbuf;
        a.startOfEncoderSampleIndex = g_prof_pairs * 2;
        a.endOfEncoderSampleIndex = g_prof_pairs * 2 + 1;
        return [g.cb computeCommandEncoderWithDescriptor:pd];
    }
    return g_concurrent ? [g.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                        : [g.cb computeCommandEncoder];
}

static void prof_begin(uint8_t cat) {
    // Count first: dispatch counts are exact and need no hardware support, while the
    // timestamps below need a counter-sampling capability this device does not report.
    // Counting only when sampling works would have made the counts silently unavailable.
    g_prof_cat_calls[cat] += 1;
    if (!g_prof_sbuf || g_prof_pairs * 2 + 1 >= PROF_MAX_SAMPLES) { return; }
    g_prof_cur = cat;
    if (prof_enc_mode()) { return; }   // the encoder's own start sample stands in
    [g.enc sampleCountersInBuffer:g_prof_sbuf atSampleIndex:g_prof_pairs * 2 withBarrier:YES];
}

static void prof_end(void) {
    if (!g_prof_sbuf || g_prof_pairs * 2 + 1 >= PROF_MAX_SAMPLES) { return; }
    if (prof_enc_mode()) {
        // Close this dispatch's encoder (its end sample fires) and open the next one.
        g_prof_cat[g_prof_pairs] = g_prof_cur;
        g_prof_pairs += 1;
        [g.enc endEncoding];
        g.enc = new_encoder();
        g_haz_reads = 0; g_haz_writes = 0;
        return;
    }
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
            g_prof_region_ticks += (double)(b - a);
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
    // Stamped on EVERY pipeline, so the Q4 rt_gemm's fences follow the same route the
    // epilogue activation already does rather than needing their own build path. The
    // shader defaults it false, so an unset engine compiles exactly what it compiled
    // before.
    [cv setConstantValue:&g_attn_skip type:MTLDataTypeUInt atIndex:17];
    // PROBE ONLY (bench, identity pages): IMPARO_ATTN_KV_PAGED=0 compiles EVERY pipeline with
    // the page table off (KV_PAGED false), so the cost of the paged indirection itself can be
    // read as a difference. Wrong answers under the pool -- never set it in the engine.
    static const bool kv_paged_off = getenv("IMPARO_ATTN_KV_PAGED") != nullptr
                                  && strcmp(getenv("IMPARO_ATTN_KV_PAGED"), "0") == 0;
    if (kv_paged_off) {
        static bool said = false;
        if (!said) { said = true; fprintf(stderr, "imparo metal: PROBE -- KV_PAGED compiled OFF on every pipeline (IMPARO_ATTN_KV_PAGED=0): identity placement assumed\n"); }
        const bool paged_off = false;
        [cv setConstantValue:&paged_off type:MTLDataTypeBool atIndex:5];
    }
    // Diagnostic: flips only the transpose flag on the score store, to price the transpose
    // itself. Wrong answers on purpose; see the shader note at ATTN_STRAIGHT_STORE_FC.
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
// RULE: haz() never encodes a dispatch. Several sites bind their pipeline and buffers before
// they call haz() (matmat_impl binds at its top and calls haz() just before dispatching), so
// anything encoded from here would run under the caller's pipeline binding -- a lazy mega
// flush placed here once launched the persistent kernel with the lm-head GEMV's ~32k
// threadgroups and rebooted this Mac (2026-09-06; evidence section 32).
inline void haz(uint64_t reads, uint64_t writes) {
    // A program run is pending and something else is about to encode (task #153): the run
    // would land after this dispatch. Recorded here, acted on at the flush (never encode here:
    // the caller has already bound its pipeline -- the phantom-dispatch rule).
    if (g_prog_n != 0u && !g_prog_in_flush) { g_prog_foreign = true; }
    // The XH conversion cache dies when anything writes its source buffer; haz() runs at
    // every encode site with exact masks, which is exactly the tracking the cache needs.
    if (g_xh_src < 62u && (writes & (1ull << g_xh_src))) { g_xh_src = 0xffffffffu; }
    if (!g_concurrent) { return; }
    if ((reads & g_haz_writes) || (writes & g_haz_writes) || (writes & g_haz_reads)) {
        if (g_prof) { g_prof_barriers += 1; }
        [g.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        g_haz_reads = 0; g_haz_writes = 0;
    }
    g_haz_reads |= reads; g_haz_writes |= writes;
}

inline uint64_t hb(uint32_t buf_id) { return 1ull << buf_id; }

// Metal requires a threadgroup memory length that is a multiple of 16 bytes (the validation
// layer asserts on 8: a norm's per-simdgroup scratch at 64 threads). Kernels use the first
// N floats; the padding is never read.
inline NSUInteger tg_bytes16(NSUInteger bytes) { return (bytes + 15u) & ~(NSUInteger)15u; }

// `cat` is REQUIRED, with no default. This helper used to stamp PC_ELEMENTWISE on
// everything it dispatched, so the embedding row gather and LFM2's short convolution
// both reported as elementwise -- a category each of them has its own name for. A
// default would have let the next op inherit the same wrong label silently.
void dispatch1(id<MTLComputePipelineState> p, NSUInteger n, uint8_t cat) {
    [g.enc setComputePipelineState:p];
    NSUInteger tw = p.maxTotalThreadsPerThreadgroup;
    if (tw > 256) { tw = 256; }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(cat); }
    [g.enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
}

}  // namespace
// The mega program's flush (task #153): defined with the entry functions (needs their structs),
// called from the region ends above them.
static void mega_prog_flush(void);

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
// WHAT A MID-GRAPH FLUSH COSTS THE STEP, measured 2026-09-04 (docs/decode-turnaround.md):
// the host has slack at decode (it encodes a token in ~0.4 ms against ~22 ms of GPU
// work), so its submission time is never on the critical path. What the step pays for one
// more command buffer is the GPU idling between the end of one buffer and the start of
// the next -- read off the buffers' own timestamps. Committed N buffers of one tiny
// dispatch each, back to back, host far ahead; the median of the N-1 gaps.
extern "C" double imparo_metal_commit_overhead(uint32_t n) {
    if (g.queue == nil || g.p_mma_peak == nil || n < 2) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    auto submit = [&](uint32_t iters) -> id<MTLCommandBuffer> {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.p_mma_peak];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        return cb;
    };
    // THE ENGINE'S REGIME, not an empty queue's: a decode token's buffers hold ~4 ms of
    // work and the next is committed long before the current ends, so the boundary costs
    // the scheduler's hand-off and nothing more. Buffers of a few microseconds drain the
    // queue between them and read the driver's submission latency instead (10-14 us
    // measured, against <1 us per boundary inside a live token). Calibrate the peak
    // kernel to ~300 us per buffer first.
    id<MTLCommandBuffer> cal = submit(4096u);
    [cal waitUntilCompleted];
    const double cal_s = [cal GPUEndTime] - [cal GPUStartTime];
    uint32_t iters = 4096u;
    if (cal_s > 0.0) {
        const double want = 4096.0 * (300e-6 / cal_s);
        iters = (uint32_t)std::max(64.0, std::min(want, 1.0e7));
    }
    std::vector<id<MTLCommandBuffer>> cbs;
    cbs.reserve(n + 4);
    for (uint32_t i = 0; i < n + 4; ++i) { cbs.push_back(submit(iters)); }   // first 4 warm
    [cbs.back() waitUntilCompleted];
    std::vector<double> gaps;
    for (size_t i = 5; i < cbs.size(); ++i) {
        gaps.push_back(([cbs[i] GPUStartTime] - [cbs[i - 1] GPUEndTime]) * 1e6);
    }
    std::sort(gaps.begin(), gaps.end());
    const double med = gaps[gaps.size() / 2];
    return med < 0.0 ? 0.0 : med;
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
// A REAL dispatch's encode, not a trivial one's: the decode GEMV's binding sequence
// (three buffers, eight constants, threadgroup memory, the grid) is what the engine pays
// per dispatch. The trivial kernel read 0.08-0.18 us; the engine's own token encodes at
// ~1.1 us per dispatch (measured 2026-09-04), and a flush derivation fed the trivial
// number sat the seat at 7 where the measured optimum is 2-3.
extern "C" double imparo_metal_encode_cost(uint32_t n) {
    if (g.queue == nil || g.p_mma_peak == nil || n == 0) { return 0.0; }
    id<MTLBuffer> out = [g.device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> pipe = g.p_q8mv_rows[0] != nil ? g.p_q8mv_rows[0] : g.p_mma_peak;
    const uint32_t iters = 1;
    const uint64_t off64 = 0; const uint32_t u32 = 1;
    double best = 0.0;
    for (int pass = 0; pass < 2; ++pass) {          // pass 0 warms, pass 1 counts
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const double t0 = CACurrentMediaTime();
        for (uint32_t i = 0; i < n; ++i) {
            [e setComputePipelineState:pipe];
            if (pipe == g.p_mma_peak) {
                [e setBuffer:out offset:0 atIndex:0];
                [e setBytes:&iters length:4 atIndex:1];
            } else {
                [e setBuffer:out offset:0 atIndex:0];
                [e setBuffer:out offset:0 atIndex:1];
                [e setBuffer:out offset:0 atIndex:2];
                [e setBytes:&off64 length:8 atIndex:3];
                [e setBytes:&off64 length:8 atIndex:10];
                [e setBytes:&u32 length:4 atIndex:4];
                [e setBytes:&u32 length:4 atIndex:5];
                [e setBytes:&u32 length:4 atIndex:6];
                [e setBytes:&u32 length:4 atIndex:7];
                [e setBytes:&u32 length:4 atIndex:13];
                [e setBuffer:out offset:0 atIndex:8];
                [e setBytes:&u32 length:4 atIndex:9];
                [e setThreadgroupMemoryLength:64 atIndex:0];
            }
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
    if (g.p_gemv_probe == nil || g_wsegs.empty()) { return 0.0; }
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
            wbind(e, 0, 0);   // the probe reads from the file's first bytes
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
/// Paged attention's page, in KV cells -- the one the shader was compiled with.
extern "C" uint32_t imparo_metal_kv_page_cells(void) { return KV_PAGE_CELLS; }
extern "C" uint32_t imparo_metal_nb8_shape_current(void) { return g_nb8_shape; }
extern "C" void imparo_metal_set_attn_short(uint32_t on) { g_attn_short = on; }
extern "C" void imparo_metal_set_qcomb_nsg(uint32_t v) { g_qcomb_nsg = v; }
extern "C" void imparo_metal_set_attn_fa(uint32_t v) { g_attn_fa = v; }
extern "C" void imparo_metal_set_fa_nsg(uint32_t v) { g_fa_nsg = v; }
extern "C" uint32_t imparo_metal_fa_nsg(void) { return fa_nsg_value(); }
// The FA op is built for slot 0 at head dims up to 128 (its register budget; see the kernel).
extern "C" uint32_t imparo_metal_fa_has_hd(uint32_t hd) {
    return (hd != 0u && g_qcomb_hds[0] == hd && hd <= 128u) ? 1u : 0u;
}
extern "C" uint32_t imparo_metal_max_threads_tg(void);   // defined with the device profile below
// LEGAL simdgroup counts at this head dim, as a bit mask over 1 << i: the kernel splits the
// queries (QB % NSG), the block's column tiles ((CB/8) % NSG) and the output tiles ((HD/8) %
// NSG) over the simdgroups, and the threadgroup must fit the device's thread cap. Derived,
// not written: a different QB or head dim changes the answer.
extern "C" uint32_t imparo_metal_fa_nsg_mask(uint32_t hd) {
    if (!imparo_metal_fa_has_hd(hd)) { return 0u; }
    const uint32_t max_threads = imparo_metal_max_threads_tg();
    uint32_t mask = 0u;
    for (uint32_t i = 0; i < 6u; ++i) {
        const uint32_t nsg = 1u << i;
        if (FA_QB % nsg == 0u && (FA_CB / 8u) % nsg == 0u && (hd / 8u) % nsg == 0u
            && nsg * 32u <= max_threads) { mask |= 1u << i; }
    }
    return mask;
}
// What the engine WOULD use, so the registry's `current` reports the live value rather
// than a remembered one: the override when set, else what the derivation returns for the
// slot that actually dispatches at prefill.
extern "C" uint32_t imparo_metal_qcomb_nsg(void) {
    if (g_qcomb_nsg != 0u) { return g_qcomb_nsg; }
    // NOT A REMEMBERED DEFAULT. There is no global holding "the nsg in use" -- the slot's
    // NSG is a compile-time macro and the dispatch reads pick.threads -- so this reports
    // what the last prefill dispatch actually used. 0 before the first one, which is
    // honest: nothing has been picked yet.
    return g_qcomb_nsg_live;
}

extern "C" void imparo_metal_set_attn_blk(uint32_t v) {
    g_attn_blk = (v == 2u || v == 8u) ? v : 4u;
    // Inert at prefill under qcomb for the same reason as attn_threads_prefill: the
    // qcomb row carries its own blk. See that setter's note.
    static bool said = false;
    if (g_qcomb && !said) {
        said = true;
        fprintf(stderr,
                "imparo metal: attn_blk is INERT at prefill while qcomb is on -- the "
                "qcomb row carries its own blk. Use IMPARO_QCOMB_BLK.\n");
    }
}
extern "C" void imparo_metal_set_qcomb_blk(uint32_t v) { g_qcomb_blk = (v == 2u) ? 2u : 4u; }
extern "C" uint32_t imparo_metal_qcomb_blk_current(void) { return g_qcomb_blk; }
extern "C" uint32_t imparo_metal_attn_blk_current(void) { return g_attn_blk; }
extern "C" void imparo_metal_set_attn_stage(uint32_t v) { g_attn_stage = v < 1u ? 1u : (v > 7u ? 7u : v); }
extern "C" void imparo_metal_set_attn_gqa(uint32_t on) { g_attn_gqa = on; }
extern "C" void imparo_metal_set_attn_fd(uint32_t on) { g_attn_fd = on; }
extern "C" void imparo_metal_set_attn_fd_chunk(uint32_t v) { g_attn_fd_chunk = v < 128u ? 128u : v; }
extern "C" void imparo_metal_set_attn_vec_max_keys(uint32_t v) { g_attn_vec_max_keys = v; }
extern "C" uint32_t imparo_metal_attn_vec_max_keys(void) { return g_attn_vec_max_keys; }
extern "C" uint32_t imparo_metal_attn_fd_chunk(void) { return g_attn_fd_chunk; }
// LEGAL chunks at this head dim, as a bit mask over 128 << i: the slice's HQ x chunk scores
// must fit the device's threadgroup memory beside the Q staging and the reductions.
extern "C" uint32_t imparo_metal_attn_fd_chunk_mask(uint32_t hd, uint32_t share) {
    if (!(hd != 0u && g_qcomb_hds[0] == hd && hd <= 128u)) { return 0u; }
    if (!(share == 2u || share == 4u || share == 8u)) { return 0u; }
    const uint64_t budget = g.device ? (uint64_t)[g.device maxThreadgroupMemoryLength] : 32768ull;
    const uint64_t fixed = ((uint64_t)share * hd + 2u * 4u * share + 4u * share * hd) * 4ull;
    uint32_t mask = 0u;
    for (uint32_t i = 0; i < 5u; ++i) {
        const uint32_t chunk = 128u << i;
        if ((uint64_t)share * chunk * 4ull + fixed <= budget) { mask |= 1u << i; }
    }
    return mask;
}

extern "C" void imparo_metal_set_rt(uint32_t on) { g_rt = on; }

extern "C" void imparo_metal_set_shrink(uint32_t on) { g_shrink = on; }

// Must be set before init: pipelines are built there.
extern "C" void imparo_metal_set_rt_all(uint32_t on) { g_rt_all = on; }

// ---- Q8_0 knob setters -------------------------------------------------------------
// Every one clamps to a value a pipeline exists for. The registry applies knobs BEFORE
// init as well as after, so a setter may run with no pipelines built: these only touch
// globals, and the pipeline selection reads them at dispatch.
extern "C" void imparo_metal_set_q8_tm_decode_sgs(uint32_t sgs) {
    if (sgs >= 1u && sgs <= 32u) { g_q8_tm_decode_sgs = sgs; }
}
extern "C" uint32_t imparo_metal_q8_tm_decode_sgs(void) { return g_q8_tm_decode_sgs; }
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
// IMPARO_ST_GEMM_SHAPE pins the prefill shape end to end, over the host config.
// -1 means nothing is pinned. A pin has to survive `apply_host_config`, which runs
// after the environment is read and would otherwise put the tuned index straight back;
// a lever that a later write silently undoes is the failure this file has hit before.
static int32_t g_st_gemm_shape_pin = -1;
extern "C" void imparo_metal_set_st_gemm_shape_pin(uint32_t shape) {
    if (shape >= ST_GEMM_CANDIDATES) {
        NSLog(@"imparo metal: IMPARO_ST_GEMM_SHAPE=%u REFUSED -- only %u shapes exist; "
              @"leaving the tuned shape in place", shape, ST_GEMM_CANDIDATES);
        return;
    }
    g_st_gemm_shape_pin = (int32_t)shape;
    g_st_gemm_shape     = shape;
    NSLog(@"imparo metal: IMPARO_ST_GEMM_SHAPE=%u pinned -- %u rows x %u tokens x %u SG, "
          @"K chunk %u", shape, ST_GEMM_SHAPES[shape][0], ST_GEMM_SHAPES[shape][1],
          ST_GEMM_SHAPES[shape][2], ST_GEMM_SHAPES[shape][3]);
}
extern "C" void imparo_metal_set_st_gemm_shape(uint32_t shape) {
    if (g_st_gemm_shape_pin >= 0) {
        if (shape != (uint32_t)g_st_gemm_shape_pin) {
            NSLog(@"imparo metal: host config st_gemm_shape=%u ignored -- "
                  @"IMPARO_ST_GEMM_SHAPE=%d is pinned", shape, g_st_gemm_shape_pin);
        }
        return;
    }
    if (shape < ST_GEMM_CANDIDATES) {
        // THE FIRST TILE DEFINES THE SINGLE-SHAPE ENGINE, and the pair's second tile is
        // set after it (its knob is declared `after: st_gemm_shape`, and the stored config
        // lists it after), so setting the first resets the pair to one shape. That is
        // also what makes the first tile RANKABLE: with the second tile left at some
        // other value, `q8_geometry` selected THAT tile for every candidate the tuner
        // tried (its rule prefers the second tile on a padding tie, and the ranking widths
        // 256 and 224 tie for every 32-token-tile candidate), and the sweep read twelve
        // candidates within 2.8% -- INERT -- while the second tile's own sweep, at the
        // same widths, spread its candidates 1.9x.
        g_st_gemm_shape = shape;
        g_st_gemm_large_shape = shape;
    }
}
extern "C" uint32_t imparo_metal_st_gemm_shape(void) { return g_st_gemm_shape; }
// q8_design: 0 = st_gemm, k = rt_gemm<Q8> at RT_SHAPES[k - 1]. Legal when the rt shape
// passes the same register model rt_shape does; the pipeline must also have been built.
extern "C" uint32_t imparo_metal_rt_shape_legal(uint32_t i);
extern "C" uint32_t imparo_metal_q8_design_legal(uint32_t v) {
    if (v == 0u) { return 1u; }
    return imparo_metal_rt_shape_legal(v - 1u);
}
extern "C" uint32_t imparo_metal_q8_designs(void) { return 1u + RT_CANDIDATES; }
extern "C" void imparo_metal_set_q8_design(uint32_t v) {
    if (g_q8_design_pin >= 0) {
        if (v != (uint32_t)g_q8_design_pin) {
            NSLog(@"imparo metal: host config q8_design=%u ignored -- IMPARO_Q8_DESIGN=%d "
                  @"is pinned", v, g_q8_design_pin);
        }
        return;
    }
    if (imparo_metal_q8_design_legal(v)) { g_q8_design = v; }
}
extern "C" uint32_t imparo_metal_q8_design(void) { return g_q8_design; }
extern "C" void imparo_metal_set_st_gemm_large_shape(uint32_t shape) {
    if (shape < ST_GEMM_CANDIDATES) { g_st_gemm_large_shape = shape; }
}
extern "C" uint32_t imparo_metal_st_gemm_large_shape(void) {
    return g_st_gemm_large_shape;
}
extern "C" void imparo_metal_set_q8_full_tiles(uint32_t on) { g_q8_full_tiles = on; }
extern "C" uint32_t imparo_metal_q8_full_tiles(void) { return g_q8_full_tiles; }
extern "C" void imparo_metal_set_q8_gemv_max_tok(uint32_t n) {
    g_q8_gemv_max_tok = n < 1u ? 1u : n;
}
extern "C" uint32_t imparo_metal_q8_gemv_max_tok(void) { return g_q8_gemv_max_tok; }
extern "C" void imparo_metal_set_q8_all(uint32_t on) { g_q8_all = on; }
extern "C" void imparo_metal_set_q8_grid_token_x(uint32_t on) { g_q8_grid_token_x = on; }
extern "C" void imparo_metal_set_q8_skip(uint32_t bits) { g_q8_skip = bits; }
extern "C" void imparo_metal_set_q8_typed_scale(uint32_t on) { g_q8_typed_scale = on; }
extern "C" void imparo_metal_set_q8_dev_a(uint32_t on) { g_q8_dev_a = on; }
extern "C" void imparo_metal_set_q8_clamp_edge(uint32_t on) { g_q8_clamp_edge = on; }
extern "C" void imparo_metal_set_q8_mma_fence(uint32_t on) { g_q8_mma_fence = on; }
extern "C" void imparo_metal_set_attn_skip(uint32_t bits) { g_attn_skip = bits; }
extern "C" uint32_t imparo_metal_q8_grid_token_x(void) { return g_q8_grid_token_x; }
// Must be set BEFORE init: it decides which prefill pipelines are compiled.
extern "C" void imparo_metal_set_attention_head_dims(const uint32_t * hds, uint32_t n) {
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
        g_qcomb_hds[i] = (hds != nullptr && i < n) ? hds[i] : 0u;
    }
}
// Same order as the head dims, same timing (before init).
extern "C" void imparo_metal_set_attention_kv_widths(const uint32_t * ws, uint32_t n) {
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
        g_attn_kvws[i] = (ws != nullptr && i < n) ? ws[i] : 0u;
    }
}

extern "C" void imparo_metal_set_qcomb_pt(uint32_t v) { g_qcomb_pt = v < 8u ? 8u : v; }
extern "C" uint32_t imparo_metal_qcomb_pt(void) { return g_qcomb_pt; }

extern "C" void imparo_metal_set_qcomb_mask(uint32_t m) {
    g_qcomb_mask = m;
    g_qcomb_mask_set = true;   // an explicit choice outranks the default below
}
extern "C" uint32_t imparo_metal_qcomb_mask(void) { return g_qcomb_mask; }

// How many head dims this model uses -- the registry derives its candidate set from it.
extern "C" uint32_t imparo_metal_qcomb_slot_count(void) {
    uint32_t n = 0u;
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) { if (g_qcomb_hds[i] != 0u) { ++n; } }
    return n;
}
extern "C" void imparo_metal_set_attn_live_mask(uint32_t on) { g_attn_live_mask = on; }
extern "C" uint32_t imparo_metal_attn_live_mask(void) { return g_attn_live_mask; }
extern "C" uint32_t imparo_metal_st_gemm_shapes(void) { return ST_GEMM_CANDIDATES; }
extern "C" uint32_t imparo_metal_q8_shape_tokens(uint32_t shape) {
    return shape < ST_GEMM_CANDIDATES ? ST_GEMM_SHAPES[shape][1] : 0u;
}

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
// THE WIDEST TILE THIS DEVICE CAN HOLD for one head dim, as arithmetic against the
// queried threadgroup budget. The tuner asks for this so its PT ladder stops where the
// hardware stops instead of at a number someone wrote: derive the LIMIT, tune the VALUE.
// 0 when the device is not up yet, or when the fixed cost alone exceeds the budget.

extern "C" uint32_t imparo_metal_qcomb_pt_limit(uint32_t hd) {
    if (g.device == nil) { return 0u; }
    const uint64_t budget = (uint64_t)[g.device maxThreadgroupMemoryLength];
    const bool ds = qcomb_derive_pt(8u, hd, false, false, g_blk_x, budget) == 0u;
    return qcomb_derive_pt(8u, hd, false, ds, g_blk_x, budget);
}

// THE LANES THE Q4 MATMAT PIPELINE TABLE IS BUILT FOR, as 2^lg over this range.
//
// `lanes` is how many threads cooperate on one output row, so the ceiling is the
// simdgroup width: 32 lanes is one row per simdgroup and nothing wider means anything.
// That is 32 on every Apple GPU, which is why it is written rather than queried -- unlike
// the threadgroup thread limit, which really does differ (512 vs 1024) and IS queried.
//
// Exported because the registry used to write `[4, 8, 16, 32]` a second time. Extend the
// loop and the tuner would simply never sweep the new rung, silently -- the same defect
// `rt_shape_count` exists to prevent.
#define IMPARO_LANES_LG_MIN 2u
#define IMPARO_LANES_LG_MAX 5u
extern "C" uint32_t imparo_metal_lanes_min(void) { return 1u << IMPARO_LANES_LG_MIN; }
extern "C" uint32_t imparo_metal_lanes_max(void) { return 1u << IMPARO_LANES_LG_MAX; }

extern "C" uint32_t imparo_metal_max_threads_tg(void) {
    return g.device == nil ? 0 : (uint32_t)[g.device maxThreadsPerThreadgroup].width;
}

// THE LEGAL SET, for the registry's derived candidates. A bit per nsg = 1<<i, set only
// when the value fits EVERY compiled qt variant at this head dim -- the knob is one value
// applied to both slots, so a candidate that is illegal for one of them is not a candidate.
// Intersecting here rather than at the call site keeps "what is legal" in one place.
extern "C" uint32_t imparo_metal_qcomb_nsg_mask(uint32_t hd) {
    if (g_measured_max_acc == 0u) { return 0u; }
    const uint32_t max_threads = imparo_metal_max_threads_tg();
    const uint32_t blk_x = g_blk_x == 0u ? 2u : g_blk_x;
    uint32_t mask = 0u;
    for (uint32_t i = 0; i < 6u; ++i) {
        const uint32_t nsg = 1u << i;
        const bool q8  = qcomb_nsg_fits(8u,  hd, 4u,   g_measured_max_acc, max_threads, nsg);
        const bool q16 = qcomb_nsg_fits(16u, hd, blk_x, g_measured_max_acc, max_threads, nsg);
        if (q8 && q16) { mask |= 1u << i; }
    }
    return mask;
}

extern "C" uint64_t imparo_metal_prof_barriers(void) { return g_prof_barriers; }
extern "C" void imparo_metal_prof_enable(uint32_t on) {
    g_prof = on;
    if (!on || g_prof_sbuf != nil) { return; }
    const bool disp_ok  = [g.device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    const bool stage_ok = [g.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    if (!disp_ok && !(prof_enc_mode() && stage_ok)) {
        NSLog(@"imparo metal: no dispatch-boundary counter sampling; per-kernel profile off (IMPARO_PROF_ENC=1 uses one encoder per dispatch instead)");
        return;
    }
    if (!disp_ok) { NSLog(@"imparo metal: per-kernel profile in ENCODER mode (one encoder per dispatch, serialised)"); }
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
    if (prof_enc_mode()) {
        NSLog(@"[prof-enc] kernel_sum=%.1f ms  gpu=%.1f ms  wall=%.1f ms  dispatches=%llu  (one encoder per dispatch; kernel_sum is pure kernel time, gpu/wall are serialised)",
              g_prof_kernel_s * 1e3, g_prof_gpu_s * 1e3, g_prof_wall_s * 1e3, (unsigned long long)g_prof_disp);
        g_prof_kernel_s = 0.0;
    }
    g_prof_gpu_s = 0.0; g_prof_wall_s = 0.0; g_prof_cbs = 0; g_prof_disp = 0;
    // The barrier counter is read separately (imparo_metal_prof_barriers) but belongs to
    // the same window: until 2026-09-02 it was never reset, so every "barriers per
    // dispatch" on record divided a process-cumulative numerator by a window-local count.
    g_prof_barriers = 0;
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
    // SAY SO WHEN IT CANNOT REACH THE KERNEL. With qcomb on -- the default -- prefill
    // dispatches `pick.threads` from the compiled qcomb row, whose NSG is DERIVED from
    // the head dim (qcomb_derive_nsg: NDB must be whole), so this value configures only
    // the tiled kernel. Setting it and measuring produced three "different" prefill
    // configurations that were the same one, reading 768.1 / 768.1 / 769.4 -- the exact
    // signature this codebase has been caught by before. Silence is what made that cost
    // a measurement; refuse to be silent instead.
    static bool said = false;
    if (g_qcomb && !said) {
        said = true;
        // NO VALUE IN THE MESSAGE. The tuned config calls this setter before any env
        // override does, so printing `n` names a number the reader never typed.
        fprintf(stderr,
                "imparo metal: attn_threads_prefill is INERT at prefill while qcomb is "
                "on -- the qcomb row's NSG is derived from the head dim. It still "
                "applies to decode and to IMPARO_QCOMB=0.\n");
    }
}

// The fast tier's size: Metal's recommended working set (about three quarters of RAM on
// Apple silicon). Asked before the weights are mapped, so it may create the device.
extern "C" uint64_t imparo_metal_working_set_budget(void) {
    id<MTLDevice> d = g.device != nil ? g.device : MTLCreateSystemDefaultDevice();
    return d == nil ? 0ull : (uint64_t)[d recommendedMaxWorkingSetSize];
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
        // The device's own thread limit, queried: the occupancy cap below is derived
        // from it rather than written down.
        const uint32_t max_threads = imparo_metal_max_threads_tg();
        g_pt_512x = qcomb_derive_pt(16u, 512u, true, true,  blk_x, tg_budget);
        g_pt_256x = qcomb_derive_pt(16u, 256u, true, false, blk_x, tg_budget);
        g_nsg_512x = qcomb_derive_nsg(16u, 512u, blk_x, g_measured_max_acc, max_threads, 16u);
        g_nsg_256x = qcomb_derive_nsg(16u, 256u, blk_x, g_measured_max_acc, max_threads, 8u);
        // IMPARO_QCOMB_UNROLL: a shape change can invert the best unroll, and it has
        // twice. An env override makes re-checking it a re-run.
        uint32_t unroll = 64u;
        if (const char * u = getenv("IMPARO_QCOMB_UNROLL")) {
            const uint32_t v = (uint32_t)atoi(u);
            if (v == 8u || v == 16u || v == 32u || v == 64u) { unroll = v; }
        }
        // ONE SLOT PER HEAD DIM THIS MODEL USES. Everything the instantiation needs is
        // derived here from the device, so the shader states no dim and no shape: the
        // widest tile that fits (PT), the simdgroup count the accumulator cliff allows
        // (NSG), and whether the tail scratch still fits in threadgroup memory (DS).
        // A slot left 0 compiles nothing.
        NSMutableDictionary * macros = [NSMutableDictionary dictionary];
        for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
            const uint32_t hd = g_qcomb_hds[i];
            NSString * kh = [NSString stringWithFormat:@"IMPARO_HD%u", i];
            macros[kh] = [NSNumber numberWithUnsignedInt:hd];
            if (i == 0u) {
                g_fa_nsg_built = fa_nsg_value();
                macros[@"IMPARO_FA_NSG0"] = [NSNumber numberWithUnsignedInt:g_fa_nsg_built];
                const bool all = getenv("IMPARO_FA_ALL") && strcmp(getenv("IMPARO_FA_ALL"), "1") == 0;
                macros[@"IMPARO_FA_ALL"] = [NSNumber numberWithUnsignedInt:all ? 1u : 0u];
            }
            macros[[NSString stringWithFormat:@"IMPARO_KVW%u", i]] =
                [NSNumber numberWithUnsignedInt:g_attn_kvws[i]];
            if (hd == 0u) { continue; }
            // DS ASKS "DID THE SPILL STILL FIT?", AND FITTING IS NOT THE SAME QUESTION AS
            // "should it live here?". `qcomb_derive_pt` returns 0 when the fixed cost alone
            // exceeds the budget, so the spill is pushed to device memory ONLY when it does
            // not fit -- and at head_dim 64 it fits, so it sits in threadgroup memory and
            // costs QT*HD floats of it (4 KB at QT 16). Threadgroup memory is what caps how
            // many threadgroups a core can hold, and this kernel is latency-bound, so that
            // 4 KB is paid in occupancy on every dispatch to buy a scratch the TAIL path
            // uses. Derive the LIMIT, tune the VALUE: the limit is "does it fit", the value
            // is a measurement nobody has taken. IMPARO_QCOMB_DSPILL=1 forces device spill
            // where it fits, =0 forces threadgroup; unset keeps the derivation.
            const bool ds = qcomb_derive_pt(8u, hd, false, false, 4u, tg_budget) == 0u;
            const uint32_t nsg = qcomb_derive_nsg(8u, hd, 4u, g_measured_max_acc, imparo_metal_max_threads_tg(), 8u);
            // The K-sharing shape for the SAME dim, answered at QT 16: it doubles QROWS,
            // so the accumulator cliff and the tail-scratch question both get different
            // answers from the QT-8 slot above. Q stages as half here, which is what lets
            // the wider tile fit at all.
            const bool dsx = qcomb_derive_pt(16u, hd, true, false, blk_x, tg_budget) == 0u;
            const uint32_t nsgx = qcomb_derive_nsg(16u, hd, blk_x, g_measured_max_acc,
                                                   imparo_metal_max_threads_tg(), 8u);
            macros[[NSString stringWithFormat:@"IMPARO_BLKX%u", i]] =
                [NSNumber numberWithUnsignedInt:blk_x];
            macros[[NSString stringWithFormat:@"IMPARO_NSGX%u", i]] =
                [NSNumber numberWithUnsignedInt:nsgx];
            macros[[NSString stringWithFormat:@"IMPARO_DSX%u", i]] =
                [NSNumber numberWithUnsignedInt:dsx ? 1u : 0u];
            macros[[NSString stringWithFormat:@"IMPARO_NSG%u", i]] =
                [NSNumber numberWithUnsignedInt:nsg];
            macros[[NSString stringWithFormat:@"IMPARO_DS%u", i]] =
                [NSNumber numberWithUnsignedInt:ds ? 1u : 0u];
            if (t_on) {
                // The SAME expression qcomb_tg_floats evaluates, written out because that
                // function is defined further down beside the encoder. Reported because
                // threadgroup bytes is what caps threadgroups per core, and it is the one
                // number that separates this kernel from upstream's.
                const unsigned long tgb = 4ul * (unsigned long)(
                    (unsigned long)16u * hd / 2u          // staged Q, half at QT 16
                  + (unsigned long)16u * g_qcomb_pt       // the score tile
                  + 2ul * 16u                             // rmax, rsum
                  + (16u / 8u) * 64ul                     // the rescale diagonals
                  + (dsx ? 0ul : (unsigned long)16u * hd));
                NSLog(@"imparo metal: qcomb slot %u -> head_dim %u, NSG %u, DSPILL %u, "
                      @"NSGX %u, DSPILLX %u, tg_bytes qt16 %lu",
                      i, hd, nsg, ds ? 1u : 0u, nsgx, dsx ? 1u : 0u, tgb);
            }
        }
        // ONLY IF NOBODY CHOSE. The stored config and the env overrides run before the
        // library is compiled; this line used to discard whatever they set.
        if (!g_qcomb_mask_set) { g_qcomb_mask = qcomb_mask_default(); }
        MTLCompileOptions * copts = [MTLCompileOptions new];
        [macros addEntriesFromDictionary:@{
            @"IMPARO_PT_512X" : [NSNumber numberWithUnsignedInt:g_pt_512x],
            @"IMPARO_PT_256X" : [NSNumber numberWithUnsignedInt:g_pt_256x],
            @"IMPARO_NSG_512X" : [NSNumber numberWithUnsignedInt:g_nsg_512x],
            @"IMPARO_NSG_256X" : [NSNumber numberWithUnsignedInt:g_nsg_256x],
            @"IMPARO_QCOMB_UNROLL" : [NSNumber numberWithUnsignedInt:unroll],
            @"IMPARO_BLK_512X" : [NSNumber numberWithUnsignedInt:blk_x],
            @"IMPARO_BLK_256X" : [NSNumber numberWithUnsignedInt:blk_x],
            // The host's page size, so the shader cannot hold a different one.
            @"KV_PAGE_CELLS" : [NSNumber numberWithUnsignedInt:KV_PAGE_CELLS],
            // Same contract for the Q4 matmat's token tile: the dispatch grid divides
            // n_tok by this, so the shader must be compiled with the host's value.
            @"Q4_TOKEN_TILE" : [NSNumber numberWithUnsignedInt:Q4_TOKEN_TILE],
            @"IMPARO_SG_ROWS" : [NSNumber numberWithUnsignedInt:SG_ROWS],
            @"IMPARO_SG_TOKENS" : [NSNumber numberWithUnsignedInt:SG_TOKENS],
            @"NB8_NA" : [NSNumber numberWithUnsignedInt:NB8_NA],
            @"NB8_NB" : [NSNumber numberWithUnsignedInt:NB8_NB],
            @"NB8_SGX" : [NSNumber numberWithUnsignedInt:NB8_SGX],
            @"NB8_SGY" : [NSNumber numberWithUnsignedInt:NB8_SGY],
        }];
        [copts setPreprocessorMacros:macros];
        if (t_on) {
            NSLog(@"imparo metal: derived from a %llu B budget and a %u-accumulator "
                  @"cliff at BLK %u -- PT 512x=%u 256x=%u, NSG 512x=%u 256x=%u",
                  tg_budget, g_measured_max_acc, g_blk_x, g_pt_512x, g_pt_256x,
                  g_nsg_512x, g_nsg_256x);
        }
        // The shader compiles from source at init, so the text the GPU actually gets -- the
        // emitted mega kernels included -- can be read out. IMPARO_MEGA_DUMP names the file.
        if (const char * dump = getenv("IMPARO_MEGA_DUMP")) {
            if (dump[0]) {
                FILE * f = fopen(dump, "w");
                if (f) {
                    fwrite(kSource, 1, strlen(kSource), f);
                    fclose(f);
                    NSLog(@"imparo metal: shader source written to %s", dump);
                } else {
                    NSLog(@"imparo metal: cannot write IMPARO_MEGA_DUMP file %s", dump);
                }
            }
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
        for (uint32_t lg = IMPARO_LANES_LG_MIN; lg <= IMPARO_LANES_LG_MAX; ++lg) {
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
                g.p_rt_gh[i] = make(lib, [NSString stringWithFormat:@"imparo_rt_%u_gh", i]);
                ++built;
            }
            // rt_gemm<Q8>. IMPARO_Q8_DESIGN pins the design for an A/B, the way
            // IMPARO_ST_GEMM_SHAPE pins the st tile; the tuner sets it otherwise.
            if (const char * e = getenv("IMPARO_Q8_DESIGN")) {
                const uint32_t v = (uint32_t)atoi(e);
                if (imparo_metal_q8_design_legal(v)) {
                    g_q8_design_pin = (int32_t)v; g_q8_design = v;
                    NSLog(@"imparo metal: IMPARO_Q8_DESIGN=%u pinned -- %s", v,
                          v == 0u ? "st_gemm" : "rt_gemm<Q8>");
                } else {
                    NSLog(@"imparo metal: IMPARO_Q8_DESIGN=%u REFUSED -- 0 = st_gemm, "
                          @"1..%u = rt_gemm<Q8> shape", v, RT_CANDIDATES);
                }
            }
            for (uint32_t i = 0; i < RT_CANDIDATES; ++i) {
                if (!g_rt_all && !g_q8_all && i + 1u != g_q8_design) { continue; }
                g.p_rt8[i]   = make(lib, [NSString stringWithFormat:@"imparo_rt8_%u", i]);
                g.p_rt8_h[i] = make(lib, [NSString stringWithFormat:@"imparo_rt8_%u_h", i]);
                g.p_rt8_gh[i] = make(lib, [NSString stringWithFormat:@"imparo_rt8_%u_gh", i]);
            }
            g.p_rt_nb8    = make(lib, @"imparo_rt_nb8");
            g.p_rt_nb8_h  = make(lib, @"imparo_rt_nb8_h");
            g.p_rt_nb8b   = make(lib, @"imparo_rt_nb8b");
            g.p_rt_nb8b_h = make(lib, @"imparo_rt_nb8b_h");
            g.p_rt_nb8_gh  = make(lib, @"imparo_rt_nb8_gh");
            g.p_rt_nb8b_gh = make(lib, @"imparo_rt_nb8b_gh");
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
        g.p_rms_add_row = make(lib, @"imparo_rms_norm_add_row");
        g.p_rope   = make(lib, @"imparo_rope_neox");
        g.p_head_norm_rope = make(lib, @"imparo_head_norm_rope");
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
            // IMPARO_PIPE_LOG prints what the COMPILER made of each kernel.
            // maxTotalThreadsPerThreadgroup falls when a kernel needs more registers per
            // lane, and registers are the other thing -- besides threadgroup memory --
            // that decides how many threadgroups stay resident. This exists because a Q8
            // GEMM variant that does STRICTLY LESS WORK measured slower (the
            // edge-predicate-free `_full` entry point, 822.8 against 827.6), which the
            // dispatch-side numbers cannot explain and register pressure can.
            static int pipe_log = -1;
            if (pipe_log < 0) { pipe_log = getenv("IMPARO_PIPE_LOG") != NULL ? 1 : 0; }
            if (pipe_log && ps != nil) {
                NSLog(@"imparo metal pipe: %@ max_threads=%lu simd_width=%lu "
                      @"static_tg_bytes=%lu", nm,
                      (unsigned long)[ps maxTotalThreadsPerThreadgroup],
                      (unsigned long)[ps threadExecutionWidth],
                      (unsigned long)[ps staticThreadgroupMemoryLength]);
            }
            return ps;
        };
        MTLFunctionConstantValues * cv_empty = [MTLFunctionConstantValues new];
        // The same constant values plus Q8_TM=true: one pipeline per layout per variant.
        auto with_tm = [](MTLFunctionConstantValues * cv) -> MTLFunctionConstantValues * {
            MTLFunctionConstantValues * c = [cv copy];
            const bool tm = true;
            [c setConstantValue:&tm type:MTLDataTypeBool atIndex:15];
            return c;
        };
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
                if (rg == 0u) { g.p_q8mv_tm = make_cv(@"imparo_q8_0_gemv", with_tm(cv)); }
            }
            for (uint32_t tg = 0; tg < 4u; ++tg) {
                const uint32_t tile = 1u << tg;          // 1, 2, 4, 8
                MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
                [cv setConstantValue:&tile type:MTLDataTypeUInt atIndex:7];
                g.p_q8mm_tile[tg] = make_cv(@"imparo_q8_0_matmat", cv);
                g.p_q8mm_tile_tm[tg] = make_cv(@"imparo_q8_0_matmat", with_tm(cv));
            }
            // Only the selected shapes are built unless a sweep asks for all of them,
            // for the same reason as the register-tile candidates: a compute pipeline is
            // GPU-resident code and these kernels are fully unrolled.
            uint32_t q8_built = 0;
            // Shape 3 is also the fallback when a selected shape's K chunk does not
            // divide a projection's n_in, so it must exist whenever a deeper shape is
            // selected -- otherwise that projection would find a nil pipeline.
            const bool q8_deep = ST_GEMM_SHAPES[g_st_gemm_shape][3] > Q8_BLOCK_ELEMENTS
                || ST_GEMM_SHAPES[g_st_gemm_large_shape][3] > Q8_BLOCK_ELEMENTS;
            for (uint32_t i = 0; i < ST_GEMM_CANDIDATES; ++i) {
                if (!g_q8_all && i != g_st_gemm_shape && i != g_st_gemm_large_shape
                    && !(q8_deep && i == 3u)) {
                    continue;
                }
                NSString * n  = [NSString stringWithFormat:@"imparo_st_gemm_%u", i];
                NSString * nf = [NSString stringWithFormat:@"imparo_st_gemm_%u_full", i];
                // cv_empty unless the probe is armed, so the shipping pipelines are the
                // ones this file has always built.
                MTLFunctionConstantValues * cvg = cv_empty;
                if (g_q8_typed_scale != 0u) {
                    cvg = [MTLFunctionConstantValues new];
                    const bool ts = true;
                    [cvg setConstantValue:&ts type:MTLDataTypeBool atIndex:9];
                }
                if (g_q8_clamp_edge != 0u) {
                    if (cvg == cv_empty) { cvg = [MTLFunctionConstantValues new]; }
                    const bool ce = true;
                    [cvg setConstantValue:&ce type:MTLDataTypeBool atIndex:13];
                }
                if (g_q8_mma_fence == 0u) {
                    // Only the OFF case needs a constant: the shader defaults to on.
                    if (cvg == cv_empty) { cvg = [MTLFunctionConstantValues new]; }
                    const bool mf = false;
                    [cvg setConstantValue:&mf type:MTLDataTypeBool atIndex:14];
                }
                if (g_q8_skip != 0u) {
                    if (cvg == cv_empty) { cvg = [MTLFunctionConstantValues new]; }
                    [cvg setConstantValue:&g_q8_skip type:MTLDataTypeUInt atIndex:12];
                    NSLog(@"imparo metal: Q8 GEMM probe armed, skip=%u "
                          @"(1 no-mma, 2 no-stage, 4 no-weight-read, 8 no-conversion) -- NUMBERS ARE WRONG "
                          @"BY DESIGN", g_q8_skip);
                }
                g.p_stgemm[i]      = make_cv(n,  cvg);
                g.p_stgemm_full[i] = make_cv(nf, cvg);
                g.p_stgemm_h[i]    = make_cv(
                    [NSString stringWithFormat:@"imparo_st_gemm_%u_h", i], cvg);
                g.p_stgemm_gh[i]   = make_cv(
                    [NSString stringWithFormat:@"imparo_st_gemm_%u_gh", i], cvg);
                g.p_stgemm_full_h[i] = make_cv(
                    [NSString stringWithFormat:@"imparo_st_gemm_%u_full_h", i], cvg);
                MTLFunctionConstantValues * cvt = with_tm(cvg);
                g.p_stgemm_tm[i]      = make_cv(n,  cvt);
                g.p_stgemm_full_tm[i] = make_cv(nf, cvt);
                g.p_stgemm_h_tm[i]    = make_cv(
                    [NSString stringWithFormat:@"imparo_st_gemm_%u_h", i], cvt);
                g.p_stgemm_gh_tm[i]   = make_cv(
                    [NSString stringWithFormat:@"imparo_st_gemm_%u_gh", i], cvt);
                g.p_stgemm_full_h_tm[i] = make_cv(
                    [NSString stringWithFormat:@"imparo_st_gemm_%u_full_h", i], cvt);
                if (g_q8_dev_a && (i == 3u || i == 7u)) {
                    // g_q8_dev_a is the PREFETCH DEPTH, not a flag: 1 selects the
                    // no-prefetch variant that measured -3.6%/-4.1%, 2 and 5 the
                    // pipelined ones. Anything else falls back to no prefetch.
                    NSString * suffix = g_q8_dev_a == 3u ? @"_da5"
                                      : (g_q8_dev_a == 2u ? @"_da2" : @"_da");
                    g.p_stgemm_da[i] = make_cv(
                        [NSString stringWithFormat:@"imparo_st_gemm_%u%@", i, suffix],
                        cvg);
                }
                if (g_q8_grid_token_x) {
                    MTLFunctionConstantValues * cvx = [MTLFunctionConstantValues new];
                    const bool tx = true;
                    [cvx setConstantValue:&tx type:MTLDataTypeBool atIndex:8];
                    g.p_stgemm_tx[i]   = make_cv(n, cvx);
                    g.p_stgemm_tx_h[i] = make_cv(
                        [NSString stringWithFormat:@"imparo_st_gemm_%u_h", i], cvx);
                }
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
        g_mega_kq_ty = kq_ty; g_mega_vq_ty = vq_ty;   // the mega kernels' quantized variants carry the same pair
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
        g.p_shortconv_step = make(lib, @"imparo_shortconv_step");
        g.p_shortconv_state = make(lib, @"imparo_shortconv_state");
        g.p_add    = make(lib, @"imparo_add");
        g.p_mul    = make(lib, @"imparo_mul");
        g.p_scale  = make(lib, @"imparo_scale");
        g.p_addscale  = make(lib, @"imparo_add_scale");
        g.p_copy   = make(lib, @"imparo_copy");
        g.p_softcap= make(lib, @"imparo_softcap");
        g.p_argmax = make(lib, @"imparo_argmax");
        g.p_argmax_feed = make(lib, @"imparo_argmax_feed");
        g.p_ple    = make(lib, @"imparo_ple_combine");
        g.p_ple_gather = make(lib, @"imparo_ple_gather_combine");
        g.p_repack_q8_tm = make(lib, @"imparo_repack_q8_0_tm");
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
        if (mega_ffn_wanted()) {
            // Same lane geometry as the decode GEMV it replaces (bit-identical rows).
            MTLFunctionConstantValues * cvm = [MTLFunctionConstantValues new];
            const uint32_t lanes = g_lanes, nr0 = 1u;
            [cvm setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
            [cvm setConstantValue:&nr0   type:MTLDataTypeUInt atIndex:2];
            const bool dbg = mega_dbg();
            [cvm setConstantValue:&dbg type:MTLDataTypeBool atIndex:18];
            g.p_mega_ffn = make_cv(@"imparo_mega_ffn", cvm);
            for (uint32_t slot = 0; slot < 2u; ++slot) {
                const uint32_t hd = g_qcomb_hds[slot];
                g.p_mega_layer[slot] = (hd != 0u && hd % 32u == 0u)
                    ? make_cv([NSString stringWithFormat:@"imparo_mega_layer_s%u", slot], cvm) : nil;
            }
            g.p_mega_ffn_ple = g.p_mega_layer[0] != nil ? g.p_mega_layer[0] : g.p_mega_layer[1];
            // The deep variants (fc 19): the grouped attention body compiled in, dispatched at one
            // threadgroup per core; the plain variants above keep their footprint.
            {
                const bool deep = true, plain = false;
                [cvm setConstantValue:&deep type:MTLDataTypeBool atIndex:19];
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    g.p_mega_layer_deep[slot] = (hd != 0u && hd % 32u == 0u)
                        ? make_cv([NSString stringWithFormat:@"imparo_mega_layer_s%u", slot], cvm) : nil;
                }
                [cvm setConstantValue:&plain type:MTLDataTypeBool atIndex:19];
            }
            if (mega_program_build()) {
                // The program form (task #153): deep body in, entry indexed at runtime; one per core.
                MTLFunctionConstantValues * cvp = [cvm copy];
                const bool on = true;
                [cvp setConstantValue:&on type:MTLDataTypeBool atIndex:19];
                [cvp setConstantValue:&on type:MTLDataTypeBool atIndex:20];
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    if (hd == 0u || hd % 32u != 0u) { continue; }
                    g.p_mega_layer_prog[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_layer_s%u", slot], cvp);
                }
                if (g_mega_kq_ty != 1u || g_mega_vq_ty != 1u) {
                    [cvp setConstantValue:&g_mega_kq_ty type:MTLDataTypeUInt atIndex:3];
                    [cvp setConstantValue:&g_mega_vq_ty type:MTLDataTypeUInt atIndex:4];
                    for (uint32_t slot = 0; slot < 2u; ++slot) {
                        const uint32_t hd = g_qcomb_hds[slot];
                        if (hd == 0u || hd % 32u != 0u) { continue; }
                        g.p_mega_layer_prog_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_layer_s%u", slot], cvp);
                    }
                }
            }
            // A quantized cache (task #156): the same two variants with the K / V storage types
            // stamped (fc 3 / 4, the decode-attention loaders' constants), on copies of the
            // constant set so the plain pipelines keep theirs.
            if (g_mega_kq_ty != 1u || g_mega_vq_ty != 1u) {
                MTLFunctionConstantValues * cvq = [cvm copy];
                [cvq setConstantValue:&g_mega_kq_ty type:MTLDataTypeUInt atIndex:3];
                [cvq setConstantValue:&g_mega_vq_ty type:MTLDataTypeUInt atIndex:4];
                const bool deep = true;
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    if (hd == 0u || hd % 32u != 0u) { continue; }
                    g.p_mega_layer_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_layer_s%u", slot], cvq);
                }
                [cvq setConstantValue:&deep type:MTLDataTypeBool atIndex:19];
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    if (hd == 0u || hd % 32u != 0u) { continue; }
                    g.p_mega_layer_deep_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_layer_s%u", slot], cvq);
                }
            }
            // The LFM2 layer block: tile-major Q8 rows (fc 15) and the model's activation (fc 11;
            // set before init, like every epilogue pipeline); one pipeline per head-dim slot.
            {
                const bool tm = true;
                [cvm setConstantValue:&tm type:MTLDataTypeBool atIndex:15];
                stamp_epi_act(cvm);
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    g.p_mega_lfm2[slot] = (hd != 0u && hd % 32u == 0u)
                        ? make_cv([NSString stringWithFormat:@"imparo_mega_lfm2_layer_s%u", slot], cvm) : nil;
                }
                const bool deep = true, plain = false;
                [cvm setConstantValue:&deep type:MTLDataTypeBool atIndex:19];
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    g.p_mega_lfm2_deep[slot] = (hd != 0u && hd % 32u == 0u)
                        ? make_cv([NSString stringWithFormat:@"imparo_mega_lfm2_layer_s%u", slot], cvm) : nil;
                }
                [cvm setConstantValue:&plain type:MTLDataTypeBool atIndex:19];
                if (mega_program_build()) {
                    MTLFunctionConstantValues * cvp = [cvm copy];
                    const bool on = true;
                    [cvp setConstantValue:&on type:MTLDataTypeBool atIndex:19];
                    [cvp setConstantValue:&on type:MTLDataTypeBool atIndex:20];
                    for (uint32_t slot = 0; slot < 2u; ++slot) {
                        const uint32_t hd = g_qcomb_hds[slot];
                        if (hd == 0u || hd % 32u != 0u) { continue; }
                        g.p_mega_lfm2_prog[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_lfm2_layer_s%u", slot], cvp);
                    }
                    if (g_mega_kq_ty != 1u || g_mega_vq_ty != 1u) {
                        [cvp setConstantValue:&g_mega_kq_ty type:MTLDataTypeUInt atIndex:3];
                        [cvp setConstantValue:&g_mega_vq_ty type:MTLDataTypeUInt atIndex:4];
                        for (uint32_t slot = 0; slot < 2u; ++slot) {
                            const uint32_t hd = g_qcomb_hds[slot];
                            if (hd == 0u || hd % 32u != 0u) { continue; }
                            g.p_mega_lfm2_prog_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_lfm2_layer_s%u", slot], cvp);
                        }
                    }
                }
                if (g_mega_kq_ty != 1u || g_mega_vq_ty != 1u) {
                    MTLFunctionConstantValues * cvq = [cvm copy];
                    [cvq setConstantValue:&g_mega_kq_ty type:MTLDataTypeUInt atIndex:3];
                    [cvq setConstantValue:&g_mega_vq_ty type:MTLDataTypeUInt atIndex:4];
                    for (uint32_t slot = 0; slot < 2u; ++slot) {
                        const uint32_t hd = g_qcomb_hds[slot];
                        if (hd == 0u || hd % 32u != 0u) { continue; }
                        g.p_mega_lfm2_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_lfm2_layer_s%u", slot], cvq);
                    }
                    [cvq setConstantValue:&deep type:MTLDataTypeBool atIndex:19];
                    for (uint32_t slot = 0; slot < 2u; ++slot) {
                        const uint32_t hd = g_qcomb_hds[slot];
                        if (hd == 0u || hd % 32u != 0u) { continue; }
                        g.p_mega_lfm2_deep_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_lfm2_layer_s%u", slot], cvq);
                    }
                }
            }
            // The grid must never exceed the co-residency limit: a spinning grid above it wedged
            // the GPU firmware and panicked this Mac on 2026-09-05. The limit is derived (see the
            // comment at g_mega_threads_limit); the env can move the shape inside it only.
            g_gpu_cores = gpu_core_count();
            // The limit must cover EVERY pipeline the block can dispatch: the stage-1 kernel and
            // both head-dim slots of the layer kernel (E4B's global layers run slot 1). A slot's
            // maxTotalThreadsPerThreadgroup is the compiler's register-pressure verdict; if one
            // slot drops to 512, a 512-thread threadgroup fills a core and only gpu_cores of
            // them are co-resident -- fewer than the 32 the grid asks for, and the persistent
            // barrier stalls on threadgroups that never got a slot (seen 2026-09-06).
            uint32_t tmax = 0;
            {
                id<MTLComputePipelineState> ps[5] = { g.p_mega_ffn, g.p_mega_layer[0], g.p_mega_layer[1], g.p_mega_lfm2[0], g.p_mega_lfm2[1] };
                const char * nm[5] = { "ffn(stage1)", "layer_s0", "layer_s1", "lfm2_s0", "lfm2_s1" };
                NSMutableString * rep = [NSMutableString new];
                for (uint32_t i = 0; i < 5u; ++i) {
                    if (ps[i] == nil) { [rep appendFormat:@" %s=nil", nm[i]]; continue; }
                    const uint32_t t = (uint32_t)[ps[i] maxTotalThreadsPerThreadgroup];
                    tmax = tmax ? std::min(tmax, t) : t;
                    [rep appendFormat:@" %s=%u(tgmem %lu)", nm[i], t, (unsigned long)[ps[i] staticThreadgroupMemoryLength]];
                }
                NSLog(@"imparo metal: mega pipelines max_threads/tg:%@ -> limit uses %u", rep, tmax);
            }
            // The deep variants run one threadgroup per core: each must admit the block's
            // threadgroup at all (its own max-threads verdict), else that slot's deep body is
            // off and the layer takes the dispatch path past the vec regime.
            {
                __strong id<MTLComputePipelineState> * dp[8] = { &g.p_mega_layer_deep[0], &g.p_mega_layer_deep[1], &g.p_mega_lfm2_deep[0], &g.p_mega_lfm2_deep[1],
                                                                 &g.p_mega_layer_deep_q[0], &g.p_mega_layer_deep_q[1], &g.p_mega_lfm2_deep_q[0], &g.p_mega_lfm2_deep_q[1] };
                const char * dn[8] = { "layer_deep_s0", "layer_deep_s1", "lfm2_deep_s0", "lfm2_deep_s1", "layer_deep_q_s0", "layer_deep_q_s1", "lfm2_deep_q_s0", "lfm2_deep_q_s1" };
                NSMutableString * rep = [NSMutableString new];
                for (uint32_t i = 0; i < 8u; ++i) {
                    if (*dp[i] == nil) { [rep appendFormat:@" %s=nil", dn[i]]; continue; }
                    const uint32_t t = (uint32_t)[*dp[i] maxTotalThreadsPerThreadgroup];
                    [rep appendFormat:@" %s=%u", dn[i], t];
                    if (t < tmax) { [rep appendFormat:@"(below the plain verdict %u: deep body OFF for this slot)", tmax]; *dp[i] = nil; }
                }
                NSLog(@"imparo metal: mega deep pipelines max_threads/tg:%@", rep);
            }
            // Unreadable core count: assume the smallest Apple GPU shipped (7 cores), which can
            // only under-fill a bigger one.
            const uint32_t cores_for_limit = g_gpu_cores > 0u ? g_gpu_cores : 7u;
            g_mega_threads_limit = cores_for_limit * tmax;
            g_mega_nsg = std::max(1u, std::min(16u, tmax / 64u));            // half the largest legal group
            // Two threadgroups per core: measured to fit for this kernel (2026-09-06), not
            // guaranteed by any API -- the compiler's max-threads verdict covers ONE threadgroup
            // per core, and a footprint change halved admission once (design doc, constraint 8).
            g_mega_tgs = cores_for_limit > 2u ? 2u * (cores_for_limit - 2u) : 2u * cores_for_limit;
            if (const char * e = getenv("IMPARO_MEGA_NSG")) { g_mega_nsg = (uint32_t)strtoul(e, nullptr, 10); }
            if (const char * e = getenv("IMPARO_MEGA_TGS")) { g_mega_tgs = (uint32_t)strtoul(e, nullptr, 10); }
            if (mega_clamp()) { NSLog(@"imparo metal: mega grid request exceeds the co-residency limit (%u threads); clamped", g_mega_threads_limit); }
            // Counters only until the first block dispatch sizes the partial scratch.
            g_mega_scratch_words = 0;
            mega_scratch_ensure(0u);
            if (mega_dbg()) { NSLog(@"imparo metal: mega DEBUG records on"); }
            NSLog(@"imparo metal: mega blocks %s (gpu_cores=%u max_threads/tg=%u threads_limit=%u grid=%ux%u lanes=%u)",
                  (g.p_mega_ffn != nil && g.p_mega_ffn_ple != nil && g.mega_sync != nil) ? "built" : "UNAVAILABLE (needs MSL 3.2)",
                  g_gpu_cores, tmax, g_mega_threads_limit, g_mega_tgs, g_mega_nsg, g_lanes);
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
        // The slots the library was compiled for. A slot with no dim built nothing, so
        // the lookup below simply finds nil and the dispatch falls through to qtile.
        for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
            if (g_qcomb_hds[i] == 0u) {
                g.p_qcomb[i] = nil; g.p_qcomb_b2[i] = nil; g.p_qcomb_x[i] = nil; continue;
            }
            g.p_qcomb[i] = make_cv(
                [NSString stringWithFormat:@"imparo_attention_prefill_qcomb_s%u", i], cv_pre);
            g.p_qcomb_b2[i] = make_cv(
                [NSString stringWithFormat:@"imparo_attention_prefill_qcomb_s%ub2", i], cv_pre);
            g.p_qcomb_x[i] = make_cv(
                [NSString stringWithFormat:@"imparo_attention_prefill_qcomb_s%ux", i], cv_pre);
            // Built for slot 0 at head dims up to 128 only (see the kernel's note); nil otherwise.
            if (i == 0) {
                const bool all = getenv("IMPARO_FA_ALL") && strcmp(getenv("IMPARO_FA_ALL"), "1") == 0;
                for (uint32_t n = 0; n < 4u; ++n) {
                    const uint32_t nsg = 1u << n;
                    const bool built = g_qcomb_hds[0] <= 128u && (all || nsg == g_fa_nsg_built);
                    g.p_fa_n[n] = built
                        ? make_cv([NSString stringWithFormat:@"imparo_attention_prefill_fa_s0_n%u", nsg], cv_pre)
                        : nil;
                    if (g.p_fa_n[n] != nil && getenv("IMPARO_ATTN_WHICH")) {
                        fprintf(stderr, "fa pipeline hd=%u nsg=%u maxthreads=%lu\n", g_qcomb_hds[0], nsg,
                                (unsigned long)g.p_fa_n[n].maxTotalThreadsPerThreadgroup);
                    }
                }
            }
        }
        g.p_attn_pre_qcomb256x = make_cv(@"imparo_attention_prefill_qcomb256x", cv_pre);
        g.p_attn_pre_qcomb512x = make_cv(@"imparo_attention_prefill_qcomb512x", cv_pre);
        if (getenv("IMPARO_ATTN_WHICH")) {
            // REGISTER PRESSURE for the PREFILL kernels, the same verdict the decode
            // scoretile block above prints and for the same reason. Our score loop holds
            // 16 simdgroup matrices live (4 score accumulators, 2 output accumulators
            // carried across the whole kernel, and 8 double-buffered K fragments) where
            // upstream's holds 5. maxTotalThreadsPerThreadgroup is the compiler saying how
            // many threads it can give registers to; a fall below 1024 is the direct
            // evidence, not an inference from counting declarations.
            for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
                if (g_qcomb_hds[i] == 0u) { continue; }
                fprintf(stderr,
                        "qcomb pipeline hd=%u blk4 maxthreads=%lu blk2 maxthreads=%lu "
                        "x16 maxthreads=%lu\n",
                        g_qcomb_hds[i],
                        (unsigned long)(g.p_qcomb[i]
                            ? [g.p_qcomb[i] maxTotalThreadsPerThreadgroup] : 0),
                        (unsigned long)(g.p_qcomb_b2[i]
                            ? [g.p_qcomb_b2[i] maxTotalThreadsPerThreadgroup] : 0),
                        (unsigned long)(g.p_qcomb_x[i]
                            ? [g.p_qcomb_x[i] maxTotalThreadsPerThreadgroup] : 0));
            }
        }
        {
            const bool paged_off = false;
            MTLFunctionConstantValues * cv_id = [MTLFunctionConstantValues new];
            [cv_id setConstantValue:&paged_off type:MTLDataTypeBool atIndex:5];
            for (int gi = 0; gi < 3; ++gi) {
                g.p_attn_dec_fd[gi] = (g_qcomb_hds[0] != 0u && g_qcomb_hds[0] <= 128u)
                    ? make_cv([NSString stringWithFormat:@"imparo_attention_decode_fd_s0_hq%u", 2u << gi], cv_pre)
                    : nil;
            }
            // The vector decode kernel exists for every head-dim slot the model declared
            // (any multiple of 32); the route below picks it by regime, not by dim.
            for (uint32_t slot = 0; slot < 2u; ++slot) {
                const uint32_t hd = g_qcomb_hds[slot];
                g.p_attn_dec_vec[slot] = (hd != 0u && hd % 32u == 0u)
                    ? make_cv([NSString stringWithFormat:@"imparo_attention_decode_vec_s%u", slot], cv_pre)
                    : nil;
            }
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
            {g.p_rope, "rope"}, {g.p_head_norm_rope, "head_norm_rope"}, {g.p_ple, "ple_combine"},
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

        if (!wsegs_build((const uint8_t *)base, len)) { return 4; }
        rset_init();
        // The fast tier is wired; slow-tier segments stay pageable and take
        // per-command-buffer residency; host-staged rows have no buffer.
        uint64_t fast = 0, slow = 0, table = 0;
        for (const WSeg & s : g_wsegs) {
            if (s.tier == WT_SLOW) { slow += s.bytes; continue; }
            rset_add(s.buf);
            fast += s.bytes;
        }
        for (const WStaged & t : g_wstaged) { table += t.bytes; }
        NSLog(@"imparo metal: weights in %zu segments: fast %llu MiB, slow %llu MiB; host-staged "
              "rows %llu MiB, not wired",
              g_wsegs.size(), (unsigned long long)(fast >> 20), (unsigned long long)(slow >> 20),
              (unsigned long long)(table >> 20));
        return 0;
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
        rset_remove(g.bufs[id]);
        [g.bufs[id] setPurgeableState:MTLPurgeableStateEmpty];
        g.bufs[id] = nil;
    }
    g.bufs[id] = [g.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    rset_add(g.bufs[id]);
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
            rset_add(fresh);
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
    for (id<MTLBuffer> b : retired) {
        rset_remove(b);
        [b setPurgeableState:MTLPurgeableStateEmpty];
    }
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
    rset_add(g.arena_buf);
    return 0;
}

/// Place one buffer at `offset` in the arena. The host lays the groups out so that two
/// groups with disjoint lifetimes both start at 0 and therefore overlap.
extern "C" int imparo_metal_place(uint32_t id, uint64_t offset, uint64_t bytes) {
    if (id >= B_COUNT || g_arena == nullptr || g.arena_buf == nil) { return 1; }
    const uint64_t len = page_round(bytes);
    if (offset + len > g_arena_size) { return 2; }
    if (g.bufs[id] != nil && !g.in_arena[id]) {
        rset_remove(g.bufs[id]);
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
    rset_remove(cur);
    rset_add(fresh);
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
        rset_add(g.kv_k[i]);
        rset_add(g.kv_v[i]);
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
    g_mega_region_ordinal += 1u;
    if (!g_mega_fail_at_read) {
        g_mega_fail_at_read = true;
        if (const char * e = getenv("IMPARO_MEGA_FAIL_AT")) {
            for (const char * q = e; *q != 0; ) {
                char * end = nullptr; const long long v = strtoll(q, &end, 10);
                if (end == q) { break; }
                if (v >= 0) { g_mega_fail_at.push_back((uint64_t)v); }
                q = (*end == ',') ? end + 1 : end;
            }
        }
    }
    // The injected failure: the region's first mega dispatch sets the sticky error before it
    // runs, so it and every later mega dispatch of the region leave at entry, as after a real
    // timeout; the retire reports the region bad and the engine re-runs the step.
    g_mega_inject_pending = !g_mega_fail_at.empty() && g.mega_sync != nil && mega_route_open()
        && std::find(g_mega_fail_at.begin(), g_mega_fail_at.end(), g_mega_region_ordinal) != g_mega_fail_at.end();
    rset_begin();
    if (g_prof && prof_enc_mode()) { [g.device sampleTimestamps:&g_ts_cpu0 gpuTimestamp:&g_ts_gpu0]; }
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
    g.enc = new_encoder();
    g_haz_reads = 0; g_haz_writes = 0;
    // The half-activation mirror SURVIVES a command-buffer turnover: its bytes live in a
    // dedicated buffer, buffers on one queue complete in commit order, and every GPU
    // write to its source goes through haz(), which invalidates it. Host writes
    // invalidate it in imparo_metal_write. Resetting it here made a probe's end/begin
    // change the computation -- the next GEMM re-converted from the FLOAT source, which
    // under the fused epilogue is not the activation at all -- so a probed run and an
    // unprobed run computed different numbers.
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
    g.enc = new_encoder();
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
    mega_prog_flush(); g_prog_base = 0u;   // a pending program run ends with its region (task #153)
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
        g_prof_region_ticks = 0.0;
        prof_resolve();
        if (prof_enc_mode()) {
            MTLTimestamp c1 = 0, g1 = 0;
            [g.device sampleTimestamps:&c1 gpuTimestamp:&g1];
            if (g1 > g_ts_gpu0 && c1 > g_ts_cpu0) {
                const double ns_per_tick = (double)(c1 - g_ts_cpu0) / (double)(g1 - g_ts_gpu0);
                g_prof_kernel_s += g_prof_region_ticks * ns_per_tick * 1e-9;
            }
        }
        g_prof_wall_s += CACurrentMediaTime() - t0;
        g_prof_gpu_s  += g_last_gpu_s;
        g_prof_cbs    += (uint64_t)g.pending.size() + 1;
    }
    g.pending.clear();
    const bool bad = ([g.cb error] != nil) || mega_check_error();
    if (!bad) { mega_good_retire(); }
    if (mega_dbg()) { mega_dbg_scan(g_mega_region_slot); }
    g_mega_region_slot = (g_mega_region_slot + 1u) % MEGA_DBG_SLOTS; g_mega_dbg_seq = 0u;
    g.enc = nil; g.cb = nil;
    rset_end();
    // Drained only after the wait, so every buffer in the pool has already completed.
    if (g.pool != nullptr) { objc_autoreleasePoolPop(g.pool); g.pool = nullptr; }
    return bad ? 1 : 0;
}

// PIPELINED DECODE. `end_async` commits the region and returns without waiting; the
// region's buffers stay outstanding so the next region can be encoded and committed behind
// them (buffers on one queue run in commit order, and Metal's tracking orders their
// accesses). `wait_outstanding` retires the OLDEST outstanding region: waits for its last
// buffer, accounts its GPU time, reports its error. The autorelease pool is drained at
// end_async -- everything the region still needs (its command buffers) is held by the
// Region strongly, and pools must nest, so it cannot wait for the retire.
extern "C" int imparo_metal_end_async(void) {
    mega_prog_flush(); g_prog_base = 0u;   // a pending program run ends with its region (task #153)
    [g.enc endEncoding];
    [g.cb commit];
    Context::Region r;
    r.cbs = g.pending;
    r.cbs.push_back(g.cb);
    r.dbg_slot = g_mega_region_slot;
    g_mega_region_slot = (g_mega_region_slot + 1u) % MEGA_DBG_SLOTS; g_mega_dbg_seq = 0u;
    g.pending.clear();
    g.outstanding.push_back(std::move(r));
    g.enc = nil; g.cb = nil;
    rset_end();
    if (g.pool != nullptr) { objc_autoreleasePoolPop(g.pool); g.pool = nullptr; }
    return 0;
}
extern "C" int imparo_metal_wait_outstanding(void) {
    if (g.outstanding.empty()) { return 0; }
    Context::Region r = std::move(g.outstanding.front());
    g.outstanding.erase(g.outstanding.begin());
    [r.cbs.back() waitUntilCompleted];
    double s = 0.0;
    bool bad = false;
    for (id<MTLCommandBuffer> cb : r.cbs) {
        s += [cb GPUEndTime] - [cb GPUStartTime];
        if ([cb error] != nil) { bad = true; }
    }
    g_last_gpu_s = s;
    if (g_prof) { g_prof_gpu_s += s; g_prof_cbs += (uint64_t)r.cbs.size(); }
    if (mega_check_error()) { bad = true; }
    if (!bad) { mega_good_retire(); }
    if (mega_dbg()) { mega_dbg_scan(r.dbg_slot); }
    return bad ? 1 : 0;
}
extern "C" uint32_t imparo_metal_decode_pipelining(void) { return 1u; }
extern "C" uint32_t imparo_metal_stages_rows(void) { return g_wstaged.empty() ? 0u : 1u; }

extern "C" void imparo_metal_write(uint32_t id, uint64_t off, const float * src, uint64_t n) {
    g_cvt_valid = 0;   // host wrote a buffer directly; the half copy may be stale
    if (id == g_xh_src) { g_xh_src = 0xffffffffu; }   // and so may the mirror
    std::memcpy((char *)[g.bufs[id] contents] + g.buf_off[id] + off * 4, src, n * 4);
}
extern "C" void imparo_metal_read(uint32_t id, uint64_t off, float * dst, uint64_t n) {
    std::memcpy(dst, (char *)[g.bufs[id] contents] + g.buf_off[id] + off * 4, n * 4);
}

// WHICH GEOMETRY SERVES ONE Q8 PREFILL GEMM. Separated from the encode path for the
// same reason llama.cpp keeps `ggml_metal_library_get_pipeline_mul_mm` out of its
// encoder: the decision is arithmetic over the dispatch's own shape, and interleaving it
// with setBuffer calls means the only way to check it is to run the engine and read a
// log. `imparo_metal_q8_pick_shape` exposes it so a test can assert the table below
// without a GPU.
struct Q8Geometry {
    uint32_t shape;
    uint32_t rows;
    uint32_t toks;
    uint32_t nsg;
    uint32_t kch;
    bool     fell_back;   // the pair's choice was illegal here and shape 3 took over
};

// The pair's choice, then legality. Both are pure functions of (n_tok, n_in) and the two
// tuned shape indices; nothing here touches Metal.
static Q8Geometry q8_geometry(uint32_t n_tok, uint32_t n_in, bool large_available) {
    uint32_t shape = g_st_gemm_shape;
    // TWO TILES, AND THE DISPATCH PICKS. The tuner names which pair is in play; which of
    // the pair suits THIS dispatch is arithmetic, so no threshold is stored.
    //
    // A wider token tile walks the weight matrix half as many times and pays for it in
    // padding, so compare the two padded token counts and take the wider tile when it
    // pads no worse. Measured on LFM2's deep prefill at four ubatch widths, and the sign
    // flips exactly where the arithmetic says it does:
    //
    //   n_tok   32-wide pads to   64-wide pads to   rule     measured
    //     448        448               448          64-wide  819.2 vs 788.2  +3.9% wide
    //     455        480               512          32-wide  870.5 vs 861.1  +1.1% narrow
    //     480        480               512          32-wide  790.4 vs 785.5  +0.6% narrow
    //     512        512               512          64-wide  827.6 vs 789.9  +4.8% wide
    //
    // This replaces a `n_tok >= g_st_gemm_large_min_tok` threshold, which no workload in
    // the registry could ever have measured -- PrefillGemm times ONE token count -- and
    // which would have got 448 wrong for any value that got 455 right.
    if (large_available && g_st_gemm_large_shape != g_st_gemm_shape) {
        const uint32_t tw = ST_GEMM_SHAPES[g_st_gemm_shape][1];
        const uint32_t tl = ST_GEMM_SHAPES[g_st_gemm_large_shape][1];
        const uint32_t pad_w = (n_tok + tw - 1u) / tw * tw;
        const uint32_t pad_l = (n_tok + tl - 1u) / tl * tl;
        if (pad_l <= pad_w) { shape = g_st_gemm_large_shape; }
    }
    // A deeper K chunk walks whole KCH steps, so a shape whose depth does not divide n_in
    // would read past the row. Fall back to the ported 32-deep shape rather than refuse:
    // the shapes differ only in staging depth, so the fallback computes the same numbers.
    bool fell_back = false;
    if ((n_in % ST_GEMM_SHAPES[shape][3]) != 0u) {
        shape = 3u;
        fell_back = true;
    }
    return Q8Geometry{shape, ST_GEMM_SHAPES[shape][0], ST_GEMM_SHAPES[shape][1],
                      ST_GEMM_SHAPES[shape][2], ST_GEMM_SHAPES[shape][3], fell_back};
}

// Test hook: the shape this dispatch would use, with both pipelines assumed present.
extern "C" uint32_t imparo_metal_q8_pick_shape(uint32_t n_tok, uint32_t n_in) {
    return q8_geometry(n_tok, n_in, true).shape;
}

// `w_off2` of a plain projection. THE GATED PAIR passes the up tensor's offset instead,
// and the encode paths below then select the _gh pipeline, walk 2 * n_out virtual rows
// and bind the second offset at buffer 10 (imparo.metal, "THE GATED PAIR").
constexpr uint64_t NO_PAIR = ~0ull;

// Q8_0 has its own dispatch: three routes, its own grid and threadgroup sizing, and a
// 64-bit weight offset. Returns false when nothing was encoded, so the caller logs the
// reason once instead of faulting inside the driver on a nil pipeline.
static bool q8_matmat(bool tm, uint64_t w_off, uint64_t w_off2, uint32_t n_in, uint32_t n_out,
                      uint32_t src, uint32_t dst, uint32_t n_tok, uint32_t src_row) {
    if (n_in == 0u || n_out == 0u || n_tok == 0u) { return false; }
    const bool gated = w_off2 != NO_PAIR;
    const uint32_t vrows = gated ? 2u * n_out : n_out;   // rows the GEMM grid walks
    if ((n_in % Q8_BLOCK_ELEMENTS) != 0u) {
        NSLog(@"imparo metal: Q8 matmat n_in=%u is not a multiple of %u; every route "
              @"walks whole blocks, so refusing rather than reading past the row",
              n_in, Q8_BLOCK_ELEMENTS);
        return false;
    }
    const bool use_gemm = n_tok > g_q8_gemv_max_tok;

    // One call, no decisions inline: `q8_geometry` owns the pair's choice AND the K-chunk
    // legality fallback, and `imparo_metal_q8_pick_shape` lets a test assert the rule with
    // no GPU. Everything below only consumes the geometry.
    const Q8Geometry geo =
        q8_geometry(n_tok, n_in, use_gemm && g.p_stgemm[g_st_gemm_large_shape] != nil);
    const uint32_t shape = geo.shape;
    const uint32_t rows = geo.rows, toks = geo.toks, nsg = geo.nsg;
    if (geo.fell_back) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            NSLog(@"imparo metal: a Q8 GEMM shape's K chunk does not divide n_in=%u, so "
                  @"this projection uses shape 3", n_in);
        }
    }
    // FULL removes every edge predicate, so it is legal only on an exact grid with no
    // fused epilogue -- the kernel would otherwise write whole tiles past n_out/n_tok.
    // THE DESIGN. q8_design k >= 1 sends the GEMM regime to rt_gemm<Q8> at RT_SHAPES[k-1]
    // (built at init, so a nil here means the knob asked for a shape the register model
    // refused, and the staged design serves instead).
    const uint32_t rt_i  = g_q8_design > 0u ? g_q8_design - 1u : 0u;
    // rt_gemm<Q8> reads the row-major layout only; a tile-major tensor keeps st_gemm.
    const bool rt_q8 = use_gemm && !tm && g_q8_design > 0u && g.p_rt8[rt_i] != nil;
    const uint32_t rt_rows = RT_SHAPES[rt_i][1] * 8u * RT_SHAPES[rt_i][2];
    const uint32_t rt_toks = RT_SHAPES[rt_i][0] * 8u * RT_SHAPES[rt_i][3];
    const bool full = use_gemm && !rt_q8 && g_q8_full_tiles != 0u && g_epilogue == 0u
                   && (n_out % rows) == 0u && (n_tok % toks) == 0u
                   && g.p_stgemm_full[shape] != nil;

    // THE f16 ACTIVATION MIRROR, the same one the Q4 rt path uses. This kernel's own
    // traffic note says the activations dominate, and they were being read as f32 here
    // while the Q4 family read half: on LFM2's FFN up-projection that is 705 MB of
    // activations against 374 MB of weights, so halving them removes about a third of
    // the bytes. `st_gemm` has carried HALF_A since it was ported; nothing instantiated
    // it.
    //
    // Generic variant only: `full` deletes the edge predicates and is a separate
    // interaction to get wrong. Mirror reuse is keyed by (source buffer, element count)
    // exactly as the rt path keys it, so a chain of GEMMs off one activation converts
    // once.
    uint32_t q8_src_bind = src;
    bool q8_half = false;
    // `full` no longer excludes the mirror: `_full_h` exists now. It used to, and that
    // made `q8_full_tiles` a two-variable A/B -- selecting the predicate-free kernel also
    // took the activations back to f32, and the knob measured the fast path slower for
    // that reason rather than on its own merits.
    // THE GATED PAIR rides the half route on its own pipeline. Decided BEFORE the mirror
    // is encoded, so a refusal has encoded nothing and the caller's two dispatches run.
    id<MTLComputePipelineState> gated_pipe = rt_q8 ? g.p_rt8_gh[rt_i]
                                           : (tm ? g.p_stgemm_gh_tm[shape] : g.p_stgemm_gh[shape]);
    if (gated && !(use_gemm && g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                   && g_epilogue != 0u && gated_pipe != nil)) {
        return false;
    }
    id<MTLComputePipelineState> half_pipe = gated ? gated_pipe
        : (rt_q8 ? g.p_rt8_h[rt_i]
                 : (full ? (tm ? g.p_stgemm_full_h_tm[shape] : g.p_stgemm_full_h[shape])
                         : (tm ? g.p_stgemm_h_tm[shape] : g.p_stgemm_h[shape])));
    if (use_gemm && g_half_a && n_tok >= HALF_A_MIN
        && half_pipe != nil && g.bufs[B_XH] != nil) {
        // The mirror covers whole token tiles of WHICHEVER kernel reads it.
        const uint32_t toks_q8 = rt_q8 ? rt_toks : ST_GEMM_SHAPES[shape][1];
        const uint32_t padded  = (n_tok + toks_q8 - 1u) / toks_q8 * toks_q8;
        const uint64_t need    = (uint64_t)padded * n_in;
        const bool mirrored = g_xh_src == src
                           && g_xh_elems >= (uint64_t)n_tok * n_in;
        if (!mirrored) {
            // Diagnostic (default off), the Q4 route's switch honoured here too:
            // IMPARO_SKIP_CVT=1 skips the conversion pass and lets the GEMM read stale XH
            // -- wrong on purpose; the wall difference is the cvt passes' share. Until
            // 2026-09-02 only the Q4 route honoured it, so on a Q8 model it silently
            // measured the baseline (review #116, D14).
            static bool skip_cvt_init = false; static bool skip_cvt = false;
            if (!skip_cvt_init) { skip_cvt = getenv("IMPARO_SKIP_CVT") != nullptr;
                                  skip_cvt_init = true; }
            if (!skip_cvt) {
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
            }
            g_xh_src = src; g_xh_elems = need; g_xh_buf = B_XH;
            // Which activations no producer mirrored. Each of these is a full pass over
            // the activation that upstream never runs, because it stages inside the
            // kernel instead. The fix for one is to make its PRODUCER dual-write the
            // mirror, which is what gemma4's four producers already do.
            static int cvt_log = -1;
            if (cvt_log < 0) { cvt_log = getenv("IMPARO_Q8_LOG") != NULL ? 1 : 0; }
            if (cvt_log) {
                NSLog(@"imparo metal q8: CVT src=%u n_in=%u n_tok=%u elems=%llu",
                      src, n_in, n_tok, (unsigned long long)need);
            }
        }
        // The GEMM's own haz() below (reads q8_src_bind = the mirror, writes dst) is what
        // makes it wait on whoever wrote the mirror. A second, identical haz() used to sit
        // here: the later one then always found `dst` in the window it had just written and
        // emitted a spurious memoryBarrier on every Q8 GEMM (136 per LFM2 chunk; review
        // #116, D3). Declared once now, at the dispatch.
        q8_src_bind = g_xh_buf;
        q8_half = true;
    }
    if (rt_q8) {
        // rt_gemm<Q8>: the register-tiled dispatch, bound exactly as the Q4 rt path binds
        // it (same entry signature), on this route's operand and mirror decisions.
        id<MTLComputePipelineState> rt_sel =
            gated ? g.p_rt8_gh[rt_i] : (q8_half ? g.p_rt8_h[rt_i] : g.p_rt8[rt_i]);
        const uint32_t rt_thr = RT_SHAPES[rt_i][2] * RT_SHAPES[rt_i][3] * 32u;
        const NSUInteger rt_k = 64;   // must match the kernel
        const NSUInteger stage_f = (NSUInteger)rt_k * (rt_rows + 2u) / 2u;
        // The gated pair closes inside each simdgroup on 128 floats of scratch; the
        // ungated epilogue spills the whole token x row tile.
        const NSUInteger epi_f   = gated ? (NSUInteger)(rt_thr / 32u) * 128u
                                 : (g_epilogue ? (NSUInteger)rt_toks * rt_rows : 0);
        haz(hb(q8_src_bind), hb(dst));
        [g.enc setComputePipelineState:rt_sel];
        const uint64_t w_off2_l = wlocal(w_off2, w_off);   // gate and up: one layer, one segment
        const uint64_t w_off_l = wbind(g.enc, w_off, 0);
        [g.enc setBuffer:g.bufs[q8_src_bind] offset:g.buf_off[q8_src_bind] atIndex:1];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
        [g.enc setBytes:&w_off_l length:8 atIndex:3];
        [g.enc setBytes:&w_off2_l length:8 atIndex:10];
        [g.enc setBytes:&n_in length:4 atIndex:4];
        [g.enc setBytes:&n_out length:4 atIndex:5];
        [g.enc setBytes:&n_tok length:4 atIndex:6];
        [g.enc setBytes:&src_row length:4 atIndex:7];
        [g.enc setBytes:&g_skip_mma length:4 atIndex:12];
        [g.enc setBytes:&g_epilogue length:4 atIndex:13];
        uint32_t rt_epi_half = 0;
        if (g_epilogue && g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH2] != nil) {
            rt_epi_half = 1;
            [g.enc setBuffer:g.bufs[B_XH2] offset:g.buf_off[B_XH2] atIndex:8];
            haz(0u, hb(B_XH2));
        } else {
            [g.enc setBuffer:g.bufs[B_XH2] != nil ? g.bufs[B_XH2] : g.bufs[dst]
                      offset:g.bufs[B_XH2] != nil ? g.buf_off[B_XH2] : g.buf_off[dst]
                     atIndex:8];
        }
        [g.enc setBytes:&rt_epi_half length:4 atIndex:9];
        [g.enc setThreadgroupMemoryLength:std::max(stage_f, epi_f) * sizeof(float)
                                  atIndex:0];
        static int rt8_log = -1;
        if (rt8_log < 0) { rt8_log = getenv("IMPARO_Q8_LOG") != NULL ? 1 : 0; }
        if (rt8_log) {
            NSLog(@"imparo metal q8: rt shape=%u rows=%u toks=%u thr=%u half=%d epi=%u "
                  @"tg=%lu grid=%ux%u n_out=%u n_tok=%u gated=%d", rt_i, rt_rows, rt_toks, rt_thr,
                  (int)q8_half, g_epilogue,
                  (unsigned long)(std::max(stage_f, epi_f) * sizeof(float)),
                  (vrows + rt_rows - 1u) / rt_rows, (n_tok + rt_toks - 1u) / rt_toks,
                  n_out, n_tok, (int)gated);
        }
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
        [g.enc dispatchThreadgroups:MTLSizeMake((vrows + rt_rows - 1u) / rt_rows,
                                                (n_tok + rt_toks - 1u) / rt_toks, 1)
              threadsPerThreadgroup:MTLSizeMake(rt_thr, 1, 1)];
        if (g_prof) { prof_end(); }
        if (rt_epi_half != 0u) {
            g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
        }
        return true;
    }
    id<MTLComputePipelineState> sel = nil;
    // The token-major twins exist only for the two generic entry points, so the FULL
    // route keeps the row-major order and its own grid. Nothing selects both.
    const bool grid_tx = use_gemm && !tm && g_q8_grid_token_x && !full && !gated
                      && (q8_half ? g.p_stgemm_tx_h[shape] : g.p_stgemm_tx[shape]) != nil;
    if (use_gemm) {
        if (gated) {
            sel = gated_pipe;                      // the pair, row-major grid only
        } else if (grid_tx) {
            sel = q8_half ? g.p_stgemm_tx_h[shape] : g.p_stgemm_tx[shape];
        } else if (q8_half && !tm && g.p_stgemm_da[shape] != nil) {
            // Full-chunk dispatches too (2026-09-02): with `!full` here the drop-A variant
            // could never serve the deep leg's 512-token dispatches, so every earlier
            // "measured" q8_dev_a number was the short leg only; its edge predicates simply
            // never fire on a full tile (and `_full` itself is not a measured win).
            sel = g.p_stgemm_da[shape];            // operand straight from the mirror
        } else if (q8_half) {
            sel = half_pipe;                       // _full_h when full, _h otherwise
        } else {
            sel = full ? (tm ? g.p_stgemm_full_tm[shape] : g.p_stgemm_full[shape])
                       : (tm ? g.p_stgemm_tm[shape] : g.p_stgemm[shape]);
        }
    } else if (n_tok == 1u) {
        sel = tm ? g.p_q8mv_tm : g.p_q8mv_rows[g_q8_decode_rows_log2];
    } else {
        sel = tm ? g.p_q8mm_tile_tm[g_q8_token_tile_log2] : g.p_q8mm_tile[g_q8_token_tile_log2];
    }
    if (sel == nil) {
        NSLog(@"imparo metal: nil Q8 pipeline (n_tok=%u gemm=%d shape=%u full=%d "
              @"rows_log2=%u tile_log2=%u)", n_tok, (int)use_gemm, shape, (int)full,
              g_q8_decode_rows_log2, g_q8_token_tile_log2);
        return false;
    }
    haz(hb(q8_src_bind), hb(dst));
    [g.enc setComputePipelineState:sel];
    const uint64_t w_off2_l = wlocal(w_off2, w_off);   // gate and up: one layer, one segment
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    // The mirror when one was made, the f32 source otherwise. Binding `src` here while
    // `sel` is the _h pipeline would hand f32 bytes to a kernel reading halves -- wrong
    // numbers, not a crash.
    [g.enc setBuffer:g.bufs[q8_src_bind] offset:g.buf_off[q8_src_bind] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    // EIGHT bytes, not four: these kernels take `constant ulong & w_offset`. The Q4_0
    // path narrows the same value to 32 bits, which is why matmat refuses a non-Q8
    // offset above UINT32_MAX rather than truncating it.
    [g.enc setBytes:&w_off_l length:8 atIndex:3];
    [g.enc setBytes:&w_off2_l length:8 atIndex:10];
    [g.enc setBytes:&n_in length:4 atIndex:4];
    [g.enc setBytes:&n_out length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBytes:&src_row length:4 atIndex:7];
    [g.enc setBytes:&g_epilogue length:4 atIndex:13];
    // The fused epilogue's half mirror of the output, the same one the Q4 rt path writes.
    // Without it LFM2's down projection converted G -- 10752 wide -- on every one of the
    // 30 FFN blocks, because no producer had mirrored it. Set EVERY dispatch: encoder
    // state persists, so a stale 1 from the previous epilogue would leak into the next.
    uint32_t q8_epi_half = 0;
    if (use_gemm && g_epilogue && g_half_a && n_tok >= HALF_A_MIN
        && g.bufs[B_XH2] != nil) {
        q8_epi_half = 1;
        [g.enc setBuffer:g.bufs[B_XH2] offset:g.buf_off[B_XH2] atIndex:8];
        haz(0u, hb(B_XH2));
    } else {
        // A kernel argument must be bound even when the flag is off.
        [g.enc setBuffer:g.bufs[B_XH2] != nil ? g.bufs[B_XH2] : g.bufs[dst]
                  offset:g.bufs[B_XH2] != nil ? g.buf_off[B_XH2] : g.buf_off[dst]
                 atIndex:8];
    }
    [g.enc setBytes:&q8_epi_half length:4 atIndex:9];

    // IMPARO_Q8_LOG=1 prints what each route actually selected. Gated, not commented
    // out: the whole block is behind the flag, so a measured run pays one getenv-cached
    // branch. Identical OUTPUT across shapes is expected -- the template keeps the same K
    // traversal and MMA order per output -- so results alone cannot show the shape knob
    // taking effect, and this is what does.
    static int q8_log = -1;
    if (q8_log < 0) { q8_log = getenv("IMPARO_Q8_LOG") != NULL ? 1 : 0; }
    if (q8_log) {
        if (use_gemm) {
            NSLog(@"imparo metal q8: st shape=%u rows=%u toks=%u nsg=%u kc=%u full=%d "
                  @"half=%d tokx=%d deva=%d tg=%lu grid=%ux%u n_out=%u n_tok=%u gated=%d", shape,
                  rows, toks, nsg, ST_GEMM_SHAPES[shape][3], (int)full, (int)q8_half,
                  (int)grid_tx,
                  (int)(sel != nil && sel == g.p_stgemm_da[shape]),
                  (unsigned long)std::max(
                      (NSUInteger)(ST_GEMM_SHAPES[shape][3] * rows
                          + ((sel != nil && sel == g.p_stgemm_da[shape])
                                 ? 0u : toks * ST_GEMM_SHAPES[shape][3]))
                          * sizeof(uint16_t),
                      (NSUInteger)rows * (toks / 2u) * sizeof(float)),
                  grid_tx ? (n_tok + toks - 1) / toks : (vrows + rows - 1) / rows,
                  grid_tx ? (vrows + rows - 1) / rows : (n_tok + toks - 1) / toks,
                  n_out, n_tok, (int)gated);
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
        // The staged tile is K_CHUNK deep, not one block: sizing it from
        // Q8_BLOCK_ELEMENTS was right only while every shape used llama's NK of 32.
        const NSUInteger kch = ST_GEMM_SHAPES[shape][3];
        // The activation stage is gone on the unstaged route, and the allocation is the
        // whole point of it -- leaving this at the staged size would keep the occupancy
        // exactly where it was and measure the change as nothing.
        const bool dev_a = (sel == g.p_stgemm_da[shape]) && sel != nil;
        const NSUInteger stage_bytes =
            (NSUInteger)(kch * rows + (dev_a ? 0u : toks * kch)) * sizeof(uint16_t);
        // HALF the tile: the masked write-back spills one token half at a time, so the
        // staged operands are what sets the allocation again. Keep this in step with
        // WB_PASSES in st_gemm -- a smaller allocation than the kernel writes is
        // out-of-bounds threadgroup memory, not a slow kernel.
        const NSUInteger out_bytes = (NSUInteger)rows * (toks / 2u) * sizeof(float);
        [g.enc setThreadgroupMemoryLength:std::max(stage_bytes, out_bytes) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
        const NSUInteger row_groups = (vrows + rows - 1) / rows;
        const NSUInteger tok_groups = (n_tok + toks - 1) / toks;
        [g.enc dispatchThreadgroups:(grid_tx ? MTLSizeMake(tok_groups, row_groups, 1)
                                             : MTLSizeMake(row_groups, tok_groups, 1))
              threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        if (q8_epi_half != 0u) {
            // The output's mirror is complete, so the next GEMM off this buffer reads it
            // instead of converting. Elems counts the FULL batch.
            g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
        }
        return true;
    }
    if (n_tok == 1u) {
        // Tile-major: the unit fixes 8 rows per threadgroup (two per lane, the row-major
        // kernel's reuse of each activation load); only the simdgroup count is a knob,
        // and it is the TM kernel's own (q8_tm_decode_sgs), not the row-major one's --
        // the two layouts measured different winners (see the knob's comment).
        const uint32_t rm_rows  = 1u << g_q8_decode_rows_log2;
        const uint32_t dec_rows = tm ? 8u : rm_rows;
        const uint32_t dec_sgs  = tm ? g_q8_tm_decode_sgs : g_q8_decode_sgs;
        // One float per (simdgroup, row) for the cross-simdgroup reduction.
        [g.enc setThreadgroupMemoryLength:(NSUInteger)dec_rows * dec_sgs
                                         * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_DECODE); }
        [g.enc dispatchThreadgroups:MTLSizeMake((n_out + dec_rows - 1) / dec_rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(dec_sgs * 32u, 1, 1)];
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
// One encode path for the plain projection and THE GATED PAIR (`w_off2 != NO_PAIR`).
// Returns whether the request was handled -- encoded, or deliberately skipped by a
// diagnostic -- and false when the caller must issue the two-dispatch form itself.
static bool matmat_impl(uint32_t wkind, uint64_t w_off, uint64_t w_off2, uint32_t n_in,
                        uint32_t n_out, uint32_t src, uint32_t dst,
                        uint32_t n_tok, uint32_t src_row) {
    const bool gated = w_off2 != NO_PAIR;
    const uint32_t vrows = gated ? 2u * n_out : n_out;   // rows the rt grid walks
    // one lane per token only pays off with a batch; decode keeps the split-row kernel
    // Batches up to g_gemv_max_tok take the GEMV; above it, the GEMM family (nb8
    // then wide). A tuned boundary again since the task-#5 wobble was root-caused
    // and fixed -- see the g_gemv_max_tok declaration.
    if (wkind > 3u) {
        NSLog(@"imparo metal: matmat got unknown weight kind %u (n_out=%u) -- load "
              @"validation should have rejected this model; refusing the dispatch", 
              wkind, n_out);
        return false;
    }
    // A prefill batch with the half mirror ENABLED and no buffer behind it means the
    // model's workflow never declared BufId::Xh. Every gate on that path tests the
    // buffer, so the whole family switches off and the engine converts f32 to half
    // inline in every threadgroup: correct, slower, and otherwise invisible. LFM2 ran
    // that way from its first commit. Warn once -- not every dispatch.
    if (g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] == nil) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "imparo metal: half-activation mirror is on but BufId::Xh is not "
                    "allocated -- this model's buffer_requirements is missing "
                    "half_activation_mirror_requirements(); prefill converts inline\n");
        }
    }
    if (wkind == 2u || wkind == 3u) {
        // Q8_0 (row-major) and Q8_0_TM (tile-major, kind 3) share every kernel; the layout
        // selects the pipeline twin. They share the buffer indices and the epilogue
        // convention with Q4, nothing else: its
        // three routes have their own grid, threadgroup and boundary knobs.
        if (g_skip_cat == (n_tok > g_q8_gemv_max_tok ? PC_MATMAT_PREFILL
                                                    : PC_MATMAT_DECODE)) { return true; }
        return q8_matmat(wkind == 3u, w_off, w_off2, n_in, n_out, src, dst, n_tok, src_row);
    }
    const bool is_q4 = wkind == 1u;
    bool use_prefill = is_q4 && n_tok > g_gemv_max_tok;
    if (gated) {
        // The pair rides the register-tiled half-activation route only; its pipelines are
        // that route's _gh twins. Decided before anything is encoded.
        const bool rt_half = use_prefill && g_rt && g.p_rt[g_rt_shape] != nil
                          && g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                          && g.p_rt_h[g_rt_shape] != nil && g.p_rt_gh[g_rt_shape] != nil
                          && g.p_rt_nb8_gh != nil && g.p_rt_nb8b_gh != nil
                          && g_epilogue != 0u;
        if (!rt_half) { return false; }
    }
    // Skips the WHOLE matmul, staging and write-back included, so it is comparable with
    // llama.cpp's GGML_METAL_SKIP_OP=MUL_MAT. IMPARO_SKIP_MMA only removes the multiplies
    // and the dequantisation from inside the kernel, which is a different quantity.
    if (use_prefill  && g_skip_cat == PC_MATMAT_PREFILL) { return true; }
    if (!use_prefill && g_skip_cat == PC_MATMAT_DECODE)  { return true; }
    // Hazards are declared ONCE, at each dispatch, with the operand that dispatch actually
    // reads (the float source or its half mirror). A route-wide haz(src, dst) used to sit
    // here as well, so the mirror route's own declaration always found `dst` in the window
    // and emitted a second barrier on every GEMM (review #116, D3).
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
        return false;
    }
    [g.enc setComputePipelineState:sel];
    const uint64_t w_off2_l = wlocal(w_off2, w_off);   // gate and up: one layer, one segment
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    // EIGHT BYTES. Every one of these kernels takes `constant ulong & w_offset` now;
    // the value used to be narrowed to 32 bits, and this file's own gemma4 mapping is
    // 4,215,695,776 bytes -- 79 MB under the ceiling.
    [g.enc setBytes:&w_off_l length:8 atIndex:3];
    [g.enc setBytes:&w_off2_l length:8 atIndex:10];
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
        // The gated pair closes inside each simdgroup on 128 floats of scratch; the
        // ungated epilogue spills the whole token x row tile.
        // The gated pair closes inside each simdgroup on 128 floats of scratch; the
        // ungated epilogue spills the whole token x row tile.
        const NSUInteger epi_f   = gated ? (NSUInteger)(rt_thr / 32u) * 128u
                                 : (g_epilogue ? (NSUInteger)rt_toks * rt_rows : 0);
        const NSUInteger shared_floats = g_rt ? std::max(stage_f, epi_f)
                                              : (NSUInteger)(64 * 72);
        [g.enc setThreadgroupMemoryLength:shared_floats * sizeof(float) atIndex:0];
        if (g_rt) {
            static bool rt_logged = false;
            static uint32_t rt_last_skip = 0xffffffffu;
            static bool rt_logged_gated = false;
            // Log the first dispatch, and again whenever the skip uniform CHANGES: a
            // diagnostic that silently reverts mid-run reads as a real measurement.
            // THE GATED PAIR is logged ONCE, the first time it runs -- not on every change
            // of route, which on a gated-FFN model is every block: 1470 lines in one deep
            // prefill, ~0.13% of its time, and a measured arm should not pay for its log.
            static int rt_log_all = -1;
            if (rt_log_all < 0) { rt_log_all = getenv("IMPARO_RT_LOG_ALL") != NULL ? 1 : 0; }
            if (rt_log_all) {
                NSLog(@"imparo metal: rt dispatch n_in=%u n_out=%u n_tok=%u src=%u dst=%u "
                      @"w_off=%llu w_off2=%llu epi=%u half=%d gated=%d shape=%u",
                      n_in, n_out, n_tok, src, dst, (unsigned long long)w_off,
                      (unsigned long long)w_off2, g_epilogue,
                      (int)(g_half_a && n_tok >= HALF_A_MIN), (int)gated, g_rt_shape);
            }
            if (!rt_logged || g_skip_mma != rt_last_skip || (gated && !rt_logged_gated)) {
                rt_logged = true;
                rt_last_skip = g_skip_mma;
                if (gated) { rt_logged_gated = true; }
                NSLog(@"imparo metal: rt shape=%u pipeline=%p cvt=%p thr=%u max_thr=%lu "
                      @"shared=%lu rows=%u toks=%u skip=%u epi=%u gated=%d",
                      g_rt_shape, (__bridge void *)pre, (__bridge void *)g.p_cvt_f16, rt_thr,
                      (unsigned long)(pre ? [pre maxTotalThreadsPerThreadgroup] : 0),
                      (unsigned long)(shared_floats * sizeof(float)), rt_rows, rt_toks,
                      g_skip_mma, g_epilogue, (int)gated);
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
                rt_sel = gated
                    ? (nb8 ? (g_nb8_shape ? g.p_rt_nb8b_gh : g.p_rt_nb8_gh)
                           : g.p_rt_gh[g_rt_shape])
                    : (nb8 ? (g_nb8_shape ? g.p_rt_nb8b_h : g.p_rt_nb8_h)
                           : g.p_rt_h[g_rt_shape]);
                src_bind = g_xh_buf;
            }
            haz(hb(src_bind), hb(dst));   // behind the mirror's writer, or the float producer
            [g.enc setComputePipelineState:rt_sel];
            const uint64_t w_off2_l = wlocal(w_off2, w_off);   // gate and up: one layer, one segment
            const uint64_t w_off_l = wbind(g.enc, w_off, 0);
            [g.enc setBuffer:g.bufs[src_bind] offset:g.buf_off[src_bind] atIndex:1];
            [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
            [g.enc setBytes:&w_off_l length:8 atIndex:3];
            [g.enc setBytes:&w_off2_l length:8 atIndex:10];
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
            if (g.bufs[B_XH2] != nil) { [g.enc setBuffer:g.bufs[B_XH2] offset:g.buf_off[B_XH2] atIndex:8]; }   // written only when epi_half
            else { [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:8]; }                                // a stand-in, never written
            if (g_epilogue && g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH2] != nil) {
                epi_half = 1;
                haz(0u, hb(B_XH2));
            }
            [g.enc setBytes:&epi_half length:4 atIndex:9];
            // Encoder state is bound HERE, next to the dispatch, not only at the top of the
            // route: the half-mirror conversion above is its own dispatch, and in the
            // per-encoder profile mode (IMPARO_PROF_ENC=1) every dispatch closes its
            // encoder, so a length set before it was lost and this GEMM ran with its
            // shared array unbound -- the wrong numerics of task #152.
            [g.enc setThreadgroupMemoryLength:shared_floats * sizeof(float) atIndex:0];
            g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
            [g.enc dispatchThreadgroups:MTLSizeMake((vrows + rt_rows - 1) / rt_rows,
                                                    (main_tok + rt_toks - 1) / rt_toks, 1)
                  threadsPerThreadgroup:MTLSizeMake(rt_thr, 1, 1)];
            if (g_prof) { prof_end(); }
            if (tail_split) {
                // The remainder rides the narrow tile. Reads use src_row + main (the
                // kernel adds src_row); WRITES are dispatch-relative, so the output and
                // the epilogue mirror bind with a row offset instead.
                const bool half_sel = (src_bind == g_xh_buf) && g_half_a;
                id<MTLComputePipelineState> tp = gated
                    ? (g_nb8_shape ? g.p_rt_nb8b_gh : g.p_rt_nb8_gh)
                    : (half_sel ? (g_nb8_shape ? g.p_rt_nb8b_h : g.p_rt_nb8_h)
                                : (g_nb8_shape ? g.p_rt_nb8b   : g.p_rt_nb8));
                const uint32_t t_thr  = g_nb8_shape ? 32u : 64u;
                const uint32_t t_src_row = src_row + main_tok;
                [g.enc setComputePipelineState:tp];
                const uint64_t w_off2_l = wlocal(w_off2, w_off);   // gate and up: one layer, one segment
                const uint64_t w_off_l = wbind(g.enc, w_off, 0);
                [g.enc setBuffer:g.bufs[src_bind] offset:g.buf_off[src_bind] atIndex:1];
                [g.enc setBuffer:g.bufs[dst] offset:(g.buf_off[dst] + (NSUInteger)main_tok * n_out * 4) atIndex:2];
                [g.enc setBytes:&w_off_l length:8 atIndex:3];
                [g.enc setBytes:&w_off2_l length:8 atIndex:10];
                [g.enc setBytes:&n_in length:4 atIndex:4];
                [g.enc setBytes:&n_out length:4 atIndex:5];
                [g.enc setBytes:&tail_r length:4 atIndex:6];
                [g.enc setBytes:&t_src_row length:4 atIndex:7];
                [g.enc setBytes:&g_skip_mma length:4 atIndex:12];
                [g.enc setBytes:&g_epilogue length:4 atIndex:13];
                if (g.bufs[B_XH2] != nil) {   // written only when epi_half
                    [g.enc setBuffer:g.bufs[B_XH2] offset:(g.buf_off[B_XH2] + (NSUInteger)main_tok * n_out * 2) atIndex:8];
                } else { [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:8]; }   // a stand-in, never written
                [g.enc setBytes:&epi_half length:4 atIndex:9];
                // Narrow tile's threadgroup budget: staged weights, or the epilogue tile.
                const NSUInteger t_stage = (NSUInteger)rt_k * (64 + 2) / 2;
                const NSUInteger t_epi   = gated ? (NSUInteger)(t_thr / 32u) * 128u
                                         : (g_epilogue ? (NSUInteger)8 * 64 : 0);
                [g.enc setThreadgroupMemoryLength:std::max(t_stage, t_epi) * sizeof(float) atIndex:0];
                g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
                [g.enc dispatchThreadgroups:MTLSizeMake(
                                            (vrows + NB8_ROWS - 1) / NB8_ROWS,
                                            (tail_r + NB8_TOKENS - 1) / NB8_TOKENS, 1)
                      threadsPerThreadgroup:MTLSizeMake(t_thr, 1, 1)];
                if (g_prof) { prof_end(); }
            }
            if (epi_half != 0u) {
                // G's mirror is complete in B_XH2 (both parts): the down projection can
                // read it. Elems counts the FULL batch, not the last dispatch's part.
                g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
            }
        } else {
            haz(hb(src), hb(dst));
            [g.enc setThreadgroupMemoryLength:shared_floats * sizeof(float) atIndex:0];
            g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_PREFILL); }
            [g.enc dispatchThreadgroups:MTLSizeMake((n_out + SG_ROWS - 1) / SG_ROWS,
                                                    (n_tok + SG_TOKENS - 1) / SG_TOKENS, 1)
                  threadsPerThreadgroup:MTLSizeMake(SG_GEMM_THREADS, 1, 1)];
            if (g_prof) { prof_end(); }
        }
    } else {
        const NSUInteger rows_per_tg = sgs * (is_q4 ? (32 / g_lanes) * nr0 : 1);
        // The kernel's TOKEN_TILE, from the same constant the shader is compiled with.
        const NSUInteger tile = is_q4 ? Q4_TOKEN_TILE : 1;
    haz(hb(src), hb(dst));
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MATMAT_DECODE); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1) / rows_per_tg,
                                                (n_tok + tile - 1) / tile, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
    if (g_prof) { prof_end(); }
    }
    return true;
}

extern "C" void imparo_metal_matmat(uint32_t wkind, uint64_t w_off, uint32_t n_in,
                                    uint32_t n_out, uint32_t src, uint32_t dst,
                                    uint32_t n_tok, uint32_t src_row) {
    (void)matmat_impl(wkind, w_off, NO_PAIR, n_in, n_out, src, dst, n_tok, src_row);
}

// THE GATED PAIR: dst = act(gate @ src) * (up @ src), n_out wide, one dispatch. The
// activation is the one the library was compiled for (EPI_ACT), which is what the
// workflow's set_epilogue would have selected; `epilogue` is raised for the dispatch and
// put back. Prefill only -- at one token the split GEMV path is the measured better form.
// Returns 0 having encoded nothing when the route cannot serve the pair.
extern "C" uint32_t imparo_metal_matmat_gated(uint32_t gate_kind, uint64_t gate_off,
                                              uint32_t up_kind, uint64_t up_off,
                                              uint32_t n_in, uint32_t n_out, uint32_t src,
                                              uint32_t dst, uint32_t n_tok) {
    if (gate_kind != up_kind || (gate_kind != 1u && gate_kind != 2u && gate_kind != 3u)) { return 0u; }
    if (n_tok < 2u || g_epi_act == 0u) { return 0u; }
    // IMPARO_GATED_PAIR=0 refuses the pair, so the two-dispatch form can be timed and
    // pinned against it in one binary. A config, not a knob: it selects a route.
    static int pair_on = -1;
    if (pair_on < 0) { const char * e = getenv("IMPARO_GATED_PAIR"); pair_on = !(e && e[0] == '0'); }
    if (!pair_on) { return 0u; }
    const uint32_t saved = g_epilogue;
    g_epilogue = g_epi_act;
    const bool ok = matmat_impl(gate_kind, gate_off, up_off, n_in, n_out, src, dst, n_tok, 0u);
    g_epilogue = saved;
    return ok ? 1u : 0u;
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
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&w_off_l length:8 atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&index length:4 atIndex:4];
    [g.enc setBytes:&scale length:4 atIndex:5];
    [g.enc setBytes:&dst_off length:4 atIndex:6];
    dispatch1(sel, width / block, PC_ROW);
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
                    && half_consumers_exist();
    haz(hb(src_buf), hb(buf) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_rms];
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:1];
    [g.enc setBytes:&w_off_l length:8 atIndex:2];
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
    // Read only when xh_on (which requires the mirror to exist); bound always, the input as
    // a stand-in otherwise, so no dispatch leans on a stale binding and validation is clean.
    if (g.bufs[B_XH] != nil) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:11]; }
    else { [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:11]; }
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
    [g.enc setThreadgroupMemoryLength:tg_bytes16((threads / 32) * sizeof(float)) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (g_prof) { prof_end(); }
    if (xh_on) { g_xh_src = buf; g_xh_elems = (uint64_t)n_row * width; g_xh_buf = B_XH; }
}

// dst = rms_norm(resid + other) * w AND resid += other, one dispatch: the pre-norm residual
// order (LFM2), where every residual add is followed by the next norm reading the sum. The
// separate add kernel is gone from that path -- on LFM2 it was 60 dispatches per chunk and
// per decoded token, each re-reading X, writing X and a half mirror of X that nothing read,
// and taking a barrier: 1.4 % of a 2048-token prefill by skip-and-diff (review #116). Same
// float adds on the same operands, the same reduction over the same values, so the bits
// match the two-dispatch form (det_gate pins EXACT is the check).
// post-norm + residual add (+ the next pre-norm): the gemma4 sandwich boundary as one
// dispatch. dst = (add + rms(src) * w1) * out_scale; dual: out = rms(dst) * w2. Returns
// false (nothing encoded) when the pipeline is missing.
extern "C" bool imparo_metal_rms_norm_add_row(uint32_t dst, uint32_t src, uint32_t add,
                                              uint64_t w1_off, uint32_t width, float eps,
                                              uint32_t n_row, float out_scale, uint32_t dual,
                                              uint64_t w2_off, uint32_t out) {
    if (g.p_rms_add_row == nil || (uint64_t)width * 4ull + 256ull > 32768ull) { return false; }
    if (g_skip_cat == PC_RMSNORM) { return true; }
    haz(hb(src) | hb(add), hb(dst) | (dual ? hb(out) : 0ull));
    [g.enc setComputePipelineState:g.p_rms_add_row];
    const uint64_t w2_off_l = wbind(g.enc, w2_off, 12);   // the next layer's norm: its own segment
    const uint64_t w1_off_l = wbind(g.enc, w1_off, 0);
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[add] offset:g.buf_off[add] atIndex:2];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:3];
    [g.enc setBytes:&w1_off_l length:8 atIndex:4];
    [g.enc setBytes:&width length:4 atIndex:5];
    [g.enc setBytes:&eps length:4 atIndex:6];
    [g.enc setBytes:&n_row length:4 atIndex:7];
    [g.enc setBytes:&out_scale length:4 atIndex:8];
    [g.enc setBytes:&dual length:4 atIndex:9];
    [g.enc setBytes:&w2_off_l length:8 atIndex:10];
    [g.enc setBuffer:g.bufs[dual ? out : dst] offset:g.buf_off[dual ? out : dst] atIndex:11];
    // THE SAME THREAD COUNT imparo_rms_norm gets for this width: the reduction order, and
    // therefore the bits, depend on it.
    const NSUInteger vec_units = (width + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    // partials (rounded up to a float4 boundary) + the staged row
    const NSUInteger nsg = threads / 32;
    [g.enc setThreadgroupMemoryLength:tg_bytes16((((nsg + 3) & ~3u) + width) * sizeof(float)) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}
extern "C" void imparo_metal_add_rms_norm(uint32_t dst, uint32_t resid, uint32_t other,
                                          uint64_t w_off, uint32_t width, float eps,
                                          uint32_t n_row, uint32_t row_stride,
                                          uint32_t base_off) {
    if (g_skip_cat == PC_RMSNORM) { return; }
    // Same mirror rule as rms_norm_from: CUR's contiguous rows are the GEMM's activation.
    const bool xh_on = g_half_a != 0u && dst == B_CUR && n_row >= HALF_A_MIN && base_off == 0u
                    && row_stride == width && g.bufs[B_XH] != nil
                    && (uint64_t)n_row * width * 2ull <= g.sizes[B_XH]
                    && half_consumers_exist();
    haz(hb(resid) | hb(other), hb(resid) | hb(dst) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_rms];
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&w_off_l length:8 atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&eps length:4 atIndex:4];
    [g.enc setBytes:&n_row length:4 atIndex:5];
    [g.enc setBytes:&row_stride length:4 atIndex:6];
    [g.enc setBytes:&base_off length:4 atIndex:7];
    [g.enc setBuffer:g.bufs[resid] offset:g.buf_off[resid] atIndex:8];   // the residual, read + written
    const uint32_t pre_add = 2;
    [g.enc setBytes:&pre_add length:4 atIndex:9];
    [g.enc setBuffer:g.bufs[other] offset:g.buf_off[other] atIndex:10];  // the mixer / FFN output
    // Read only when xh_on (which requires the mirror to exist); bound always, the output as
    // a stand-in otherwise, so no dispatch leans on a stale binding and validation is clean.
    if (g.bufs[B_XH] != nil) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:11]; }
    else { [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:11]; }
    const uint32_t xh_flag = xh_on ? 1u : 0u;
    [g.enc setBytes:&xh_flag length:4 atIndex:12];
    const NSUInteger vec_units = (width + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    [g.enc setThreadgroupMemoryLength:tg_bytes16((threads / 32) * sizeof(float)) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (g_prof) { prof_end(); }
    if (xh_on) { g_xh_src = dst; g_xh_elems = (uint64_t)n_row * width; g_xh_buf = B_XH; }
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
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBuffer:g.bufs[idx_buf] offset:g.buf_off[idx_buf] atIndex:2];
    [g.enc setBytes:&w_off_l length:8 atIndex:3];
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
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:1];
    [g.enc setBytes:&w_off_l length:8 atIndex:2];
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
    [g.enc setThreadgroupMemoryLength:tg_bytes16((threads / 32) * sizeof(float)) atIndex:0];
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

// Per-head rms_norm then NEOX rope, one dispatch (see the kernel). Counted under rms_norm.
extern "C" void imparo_metal_head_norm_rope(uint32_t buf, uint64_t w_off, uint32_t head_dim,
                                            float eps, uint32_t n_heads, uint32_t start_pos,
                                            uint32_t n_tok, uint32_t n_rot, float base,
                                            const float * freqs, uint32_t n_freqs) {
    if (g_skip_cat == PC_RMSNORM) { return; }
    haz(hb(buf), hb(buf));
    const uint32_t n_row = n_tok * n_heads;
    [g.enc setComputePipelineState:g.p_head_norm_rope];
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] atIndex:1];
    [g.enc setBytes:&w_off_l length:8 atIndex:2];
    [g.enc setBytes:&head_dim length:4 atIndex:3];
    [g.enc setBytes:&eps length:4 atIndex:4];
    [g.enc setBytes:&n_row length:4 atIndex:5];
    [g.enc setBytes:&n_rot length:4 atIndex:6];
    [g.enc setBytes:&base length:4 atIndex:7];
    [g.enc setBytes:&n_heads length:4 atIndex:8];
    [g.enc setBytes:&start_pos length:4 atIndex:9];
    static const float kNoFreqs[1] = {1.0f};
    [g.enc setBytes:(n_freqs > 0 ? freqs : kNoFreqs)
             length:(n_freqs > 0 ? n_freqs * 4 : 4) atIndex:10];
    [g.enc setBytes:&n_freqs length:4 atIndex:11];
    // Thread count exactly as imparo_metal_rms_norm derives it: the reduction order (and
    // so the bits) depends on it.
    const NSUInteger vec_units = (head_dim + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    [g.enc setThreadgroupMemoryLength:tg_bytes16((threads / 32) * sizeof(float)) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
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
        ensure_kv_pt(layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
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
        ensure_kv_pt(layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
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
    ensure_kv_pt(layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
    kv_pt_audit("store-f16", layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
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
    ensure_kv_pt(layer, (slots + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
    [g.enc setBuffer:g.kv_pt[layer] offset:0 atIndex:6];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_KV_STORE); }
    [g.enc dispatchThreads:MTLSizeMake(width / 32u, slots, 1)
     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

// The map itself. Every row states its instantiation's own parameters, so a dim's
// threadgroup size cannot drift from the kernel it was compiled with -- the four values
// travel together instead of being recomputed at three call sites.
// WHICH DECODE-ATTENTION SPECIALISATION SERVES A HEAD DIM. Index 3 is the catch-all, so
// a dim with no specialisation still has a path -- which is why an unlisted model runs at
// all rather than needing a case added for it.
//
// Stated ONCE. It was written out twice, 150 lines apart and byte-identical, which is two
// places to update when a specialisation is added and one of them to forget.
static uint32_t decode_hd_index(uint32_t head_dim) {
    switch (head_dim) {
        case 128u: return 0u;
        case 256u: return 1u;
        case 512u: return 2u;
        default:   return 3u;   // the dynamic-head-dim kernels
    }
}

static QcombPick qcomb_for(uint32_t head_dim, uint32_t blk, bool want_x) {
    // KEYED ON THE SHAPE PARAMETERS THAT ACTUALLY EXIST: head dim, BLK, and whether the
    // QT-16 K-sharing variant is wanted. It was keyed on (head_dim, x) alone, which is why
    // `qcomb_blk` could be set, stored and swept while never reaching a dispatch -- the
    // b2 pipelines were compiled every run and never selected.
    //
    // Threadgroup floats travel in the row with the pipeline they belong to, because they
    // must agree with the shape that was compiled. BLK does not touch threadgroup memory
    // (only QT and PT do), so the two blk rows of a dim share a size.
    struct Row {
        uint32_t                    hd;
        uint32_t                    blk;    // 0 = the QT-16 K-sharing row, blk compiled in
        id<MTLComputePipelineState> pipe;
        uint32_t                    pt;
        NSUInteger                  floats;
        uint32_t                    qt;
        uint32_t                    threads;
    };
    // PT 128 on the QT-8 rows is a REMAINING HARDCODE (task #66): `qcomb_derive_pt` exists
    // and the QT-16 rows use it, so those follow the device's real threadgroup budget and
    // these do not. Deriving it changes a shipping shape and so moves the logits, which
    // needs its own measurement and re-pin rather than a quiet edit here.
    // THE TILE FOLLOWS THE DEVICE. `qcomb_derive_pt` inverts qcomb_tg_floats: given the
    // real threadgroup budget it returns the widest tile that fits, rounded to a whole
    // number of work units. It reproduces both shipped QT-16 values (112 at hd 256, 240 at
    // 512), which is why it is trusted here. Frozen at 128 these rows used a fraction of
    // the budget -- this device fits 480 at hd 256 and 864 at 64.
    // DERIVE THE LIMIT, TUNE THE VALUE. `qcomb_derive_pt` returns the widest tile that
    // FITS, which is not the fastest tile: measured on E4B q4_0 prefill, the derived 480
    // is 1.3% SLOWER than 128 (5098 ms against 5032, three cold prefills each). A wider
    // tile means more threadgroup memory per threadgroup and so fewer of them resident,
    // and that costs more than the saved passes over the KV.
    //
    // So the derivation bounds the legal range and the knob picks inside it. 128 is the
    // compiled default because it is what measured best here, not because it was written
    // down first -- and an untuned host runs it.
    // READ THE GLOBAL, not a local override. An env read lived here and set only this
    // helper's copy, so IMPARO_QCOMB_PT moved the qcomb tile while qtile kept its default
    // -- a 128/256/440 sweep then measured three identical runs and would have been
    // recorded as "the tile does not matter". The env sets the global at init now.
    const uint32_t pt_want = g_qcomb_pt;
    const uint64_t budget = (uint64_t)[g.device maxThreadgroupMemoryLength];
    // DSPILL at 512: the staged Q leaves no room for the tail scratch.
    // BUILT FROM THE SLOTS THE LIBRARY WAS COMPILED FOR, so this states no dim either.
    Row rows[3u * QCOMB_HD_SLOTS + 2u];
    uint32_t nrows = 0u;
    // THE DEDICATED ROWS COME FIRST, and the order is the whole point. A dim that has a
    // hand-built K-sharing kernel keeps it: those two shapes are what the determinism
    // pins were recorded against, and the per-slot row below derives its own PT and NSG,
    // so it is a DIFFERENT kernel at the same (hd, qt). Placed first, the per-slot row
    // shadowed them -- E4B's 256 and 512 layers silently moved onto it and det_gate went
    // to 8 MISMATCH with kv_gates failing, while the probe still read `qt=16` at both
    // dims because the branch and the tile size had not changed. Unifying them is a
    // deliberate re-pin of E4B, not a side effect of adding a dim.
    rows[nrows++] = { 256u, 0u, g.p_attn_pre_qcomb256x, g_pt_256x,
        qcomb_tg_floats(16u, 256u, g_pt_256x, true, false), 16u, g_nsg_256x * 32u };
    rows[nrows++] = { 512u, 0u, g.p_attn_pre_qcomb512x, g_pt_512x,
        qcomb_tg_floats(16u, 512u, g_pt_512x, true, true),  16u, g_nsg_512x * 32u };
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
        const uint32_t hd = g_qcomb_hds[i];
        if (hd == 0u || g.p_qcomb[i] == nil) { continue; }
        const bool ds = qcomb_derive_pt(8u, hd, false, false, 4u, budget) == 0u;
        const uint32_t pt = min_u32(pt_want,
            qcomb_derive_pt(8u, hd, false, ds, blk == 0u ? 4u : blk, budget));
        const NSUInteger fl = qcomb_tg_floats(8u, hd, pt, false, ds);
        const uint32_t nsg = qcomb_derive_nsg(8u, hd, 4u, g_measured_max_acc, imparo_metal_max_threads_tg(), 8u);
        rows[nrows++] = { hd, 4u, g.p_qcomb[i],    pt, fl, 8u, nsg * 32u };
        rows[nrows++] = { hd, 2u, g.p_qcomb_b2[i], pt, fl, 8u, nsg * 32u };
        // The K-sharing row for the SAME dim. blk 0 is how a caller asks for it, and its
        // shape is answered at QT 16 -- different tile, different threadgroup size, and Q
        // staged as half, so none of the QT-8 row's numbers carry over.
        const bool dsx = qcomb_derive_pt(16u, hd, true, false, blk == 0u ? 2u : blk,
                                         budget) == 0u;
        const uint32_t ptx = min_u32(pt_want,
            qcomb_derive_pt(16u, hd, true, dsx, blk == 0u ? 2u : blk, budget));
        const NSUInteger flx = qcomb_tg_floats(16u, hd, ptx, true, dsx);
        const uint32_t nsgx = qcomb_derive_nsg(16u, hd, blk == 0u ? 2u : blk,
                                               g_measured_max_acc,
                                               imparo_metal_max_threads_tg(), 8u);
        rows[nrows++] = { hd, 0u, g.p_qcomb_x[i], ptx, flx, 16u, nsgx * 32u };
    }
    QcombPick p = { nil, 8u, 256u, 0u, 0 };
    // The K-sharing row first when it is wanted; a dim without one falls through to its
    // QT-8 rows, so no caller has to know which dims have K-sharing.
    for (uint32_t ri = 0; ri < nrows; ++ri) {
        const Row & r = rows[ri];
        const bool match = want_x ? (r.hd == head_dim && r.blk == 0u)
                                  : (r.hd == head_dim && r.blk == blk);
        if (!match || r.pipe == nil) { continue; }
        p.pipe = r.pipe; p.qt = r.qt; p.threads = r.threads;
        p.pt = r.pt; p.sfloats = r.floats;
        return p;
    }
    // Wanted K-sharing and this dim has none, or an unbuilt blk: take the dim's blk-4 row.
    if (want_x || blk != 4u) { return qcomb_for(head_dim, 4u, false); }
    return p;   // pipe stays nil for a dim with no kernel: the caller uses qtile
}

// Does a qcomb kernel exist for this head dim? Answered from the SAME table the dispatch
// reads, so "the knob applies here" and "the route can actually change here" cannot
// disagree. A dim with no kernel always takes qtile whatever the floor says, which makes
// the knob inert -- and a knob swept where it cannot move produces a number, which is
// worse than producing nothing.
extern "C" uint32_t imparo_metal_qcomb_has_hd(uint32_t hd) {
    return qcomb_for(hd, g_qcomb_blk, false).pipe != nil ? 1u : 0u;
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
    // WHICH HEAD DIMS THE QCOMB FAMILY SERVES: whichever have a kernel, at or above the
    // floor. `qcomb_for` is the one place that maps a dim to its instantiation, so adding
    // a dim is adding a kernel, not editing a condition here.
    // Its slot index, not its size: dims below the floor fall through to qtile.
    uint32_t slot_of = QCOMB_HD_SLOTS;
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
        if (g_qcomb_hds[i] == head_dim) { slot_of = i; break; }
    }
    const bool qcomb_hd = slot_of < QCOMB_HD_SLOTS
                       && (g_qcomb_mask & (1u << slot_of)) != 0u
                       && qcomb_for(head_dim, g_qcomb_blk, false).pipe != nil;

    // THE FA ROUTE, the default where its pipeline exists (IMPARO_ATTN_FA=0 refuses it for
    // A/Bs). Scoped to what the op can serve: slot 0's head dim at <= 128, no ring (its 8-row
    // K/V tiles need contiguous slots), and a real prefill batch. Everything outside that
    // falls through to qcomb unchanged, so the route cannot silently take a layer the kernel
    // was never built for (E4B's slots are 256 and 512: qcomb serves them).
    // The knob's LIVE value selects the pipeline; a value whose kernel was not built (the
    // engine builds the seated one only) finds nil and falls through to qcomb, loudly.
    const uint32_t fa_nsg_live = fa_nsg_value();
    const uint32_t fa_idx = fa_nsg_live == 1u ? 0u : fa_nsg_live == 2u ? 1u : fa_nsg_live == 4u ? 2u : fa_nsg_live == 8u ? 3u : 4u;
    id<MTLComputePipelineState> p_fa = fa_idx < 4u ? g.p_fa_n[fa_idx] : nil;
    if (g_attn_fa && slot_of == 0u && g_qcomb_hds[0] <= 128u && ring == 0u && n_tok > 1u && p_fa == nil) {
        static bool said = false;
        if (!said) { said = true; fprintf(stderr, "imparo metal: attn_fa_nsg=%u has no built pipeline (seated %u; IMPARO_FA_ALL=1 builds every legal one) -- qcomb serves this layer\n", fa_nsg_live, g_fa_nsg_built); }
    }
    if (g_attn_fa && slot_of == 0u && p_fa != nil && ring == 0u && n_tok > 1u) {
        const uint32_t QB = FA_QB, NSG_FA = fa_nsg_live, THREADS = NSG_FA * 32u;
        // sq (QB x hd half) + ss (QB x CB float) + so (QB x hd float).
        const uint32_t CB_FA = FA_CB;
        const uint32_t sfloats = (QB * head_dim) / 2u + QB * CB_FA + QB * head_dim;
        if (getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn branch FA layer=%u n_tok=%u hd=%u qb=%u cb=64 threads=%u "
                            "tg_bytes=%lu\n",
                    kv_layer, n_tok, head_dim, QB, THREADS,
                    (unsigned long)(sfloats * sizeof(float)));
        }
        // Same half-mirror precondition the qcomb path applies further down: the mirror is
        // only live when the model DECLARED Xh and it is big enough for this batch.
        const bool axh = g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                      && half_consumers_exist()
                      && (uint64_t)n_tok * n_heads * head_dim * 2ull <= g.sizes[B_XH];
        ensure_kv_pt(kv_layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
        haz(hb(B_Q) | (kq ? hb(B_KDQ) : HZ_KVK) | (vq ? hb(B_VDQ) : HZ_KVV),
            hb(B_ATTN) | (axh ? hb(B_XH) : 0u));
        [g.enc setComputePipelineState:p_fa];
        [g.enc setBuffer:g.bufs[B_Q]    offset:g.buf_off[B_Q]    atIndex:0];
        // Same K/V binding qcomb uses: a quantized cache was dequantised into the half
        // scratch above, so the kernel reads that instead of the cache itself.
        if (kq) { [g.enc setBuffer:g.bufs[B_KDQ] offset:g.buf_off[B_KDQ] atIndex:1]; }
        else    { [g.enc setBuffer:g.kv_k[kv_layer] offset:kv_reg(kv_layer, false) atIndex:1]; }
        if (vq) { [g.enc setBuffer:g.bufs[B_VDQ] offset:g.buf_off[B_VDQ] atIndex:2]; }
        else    { [g.enc setBuffer:g.kv_v[kv_layer] offset:kv_reg(kv_layer, true) atIndex:2]; }
        [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:3];
        [g.enc setBytes:&n_heads   length:4 atIndex:4];
        [g.enc setBytes:&n_kv      length:4 atIndex:5];
        [g.enc setBytes:&kv_width  length:4 atIndex:6];
        [g.enc setBytes:&start_pos length:4 atIndex:7];
        [g.enc setBytes:&window    length:4 atIndex:8];
        [g.enc setBytes:&ring      length:4 atIndex:9];
        [g.enc setBytes:&n_tok     length:4 atIndex:10];
        if (axh) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:11]; }
        else     { [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:11]; }   // a stand-in, never written
        const uint32_t axh_flag_fa = axh ? 1u : 0u;
        [g.enc setBytes:&axh_flag_fa length:4 atIndex:12];
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:13];
        [g.enc setThreadgroupMemoryLength:sfloats * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, (n_tok + QB - 1u) / QB, 1)
                threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
        if (g_prof) { prof_end(); }
        if (axh) {
            g_xh_src = B_ATTN;
            g_xh_elems = (uint64_t)n_tok * n_heads * head_dim;
            g_xh_buf = B_XH;   // which scratch holds it -- was left to whoever set it last
        }
        return;
    }
    // WHY, not just which. Three independent terms gate this route and a flat A/B cannot
    // tell which one refused: setting the mask and seeing no change reads identically to
    // "qcomb is not faster here". Print the terms so the next person prices the kernel
    // instead of the routing.
    // READ ONCE, NOT PER DISPATCH. getenv walks environ, and this sits on the attention
    // path that the tuner's micro loops hammer -- as a plain call it turned a 33 s tune
    // into minutes. The file already uses this idiom for IMPARO_QCOMB_X; probes must cost
    // a load and a branch, never a lookup.
    static const bool which_env = getenv("IMPARO_ATTN_WHICH") != nullptr;
    if (which_env) {
        static uint32_t said = 0u;
        if (!qcomb_hd && said < 4u) {
            ++said;
            fprintf(stderr,
                    "attn qcomb REFUSED hd=%u slot=%u/%u mask=0x%x bit=%u pipe=%s "
                    "blk=%u qcomb=%u kv_quant=%u\n",
                    head_dim, slot_of, QCOMB_HD_SLOTS, g_qcomb_mask,
                    slot_of < QCOMB_HD_SLOTS
                        ? ((g_qcomb_mask >> slot_of) & 1u) : 0u,
                    qcomb_for(head_dim, g_qcomb_blk, false).pipe ? "yes" : "NIL",
                    g_qcomb_blk, g_qcomb, (uint32_t)kv_quant);
        }
    }
    if (n_tok > 1 && (g_qcomb || kv_quant) && qcomb_hd
        && (kv_quant || g_qcomb == 1u || (g_qcomb == 2u && window == 0u)
                          || (g_qcomb == 3u && window != 0u))) {
        // Half mirror of the attention output for the wo projection (IMPARO_HALF_A):
        // written by the kernel's store paths, so the wo GEMM's cvt pass disappears.
        const bool axh = g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                      && half_consumers_exist()
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
        // ASK THE ROW, DO NOT RE-DERIVE IT. This was a written-down ladder -- 512 -> one
        // formula, 256 -> another, anything else -> 0 -- and the 0 made `qx_fits`
        // trivially TRUE at every other dim, because 0 bytes fit any budget. It happened
        // to be harmless while no other dim had a QT-16 row; the moment one did, the fit
        // check was passing on a number it had never computed.
        //
        // `qcomb_for(.., true)` already returns the row's own threadgroup floats, and
        // falls back to the QT-8 row when a dim has no K-sharing shape -- so `qt == 16`
        // is how "this dim really has one" is asked, and sfloats is the size that row
        // actually needs. One lookup, one source.
        const QcombPick xpick = qcomb_for(head_dim, g_qcomb_blk, true);
        const bool qx_fits = xpick.pipe != nil && xpick.qt == 16u
            && xpick.sfloats * sizeof(float) <= [g.device maxThreadgroupMemoryLength];
        const bool qx = qx_env && !kv_quant && qx_fits;
        // ONE LOOKUP. Pipeline, row group, thread count and threadgroup floats come from
        // the same row, and a dim with no QT-16 variant quietly yields the QT-8 one.
        const QcombPick pick = qcomb_for(head_dim, g_qcomb_blk, qx);
        // Say WHICH of the two reasons declined the K-sharing shape: this dim has no such
        // row at all, or it has one that will not fit this device. They call for
        // different work and used to print the same line.
        static const bool which_x = getenv("IMPARO_ATTN_WHICH") != nullptr;
        if (!qx_fits && which_x) {
            if (xpick.pipe == nil || xpick.qt != 16u) {
                fprintf(stderr, "attn qcomb hd=%u has no QT16 row; using QT8\n", head_dim);
            } else {
                fprintf(stderr,
                        "attn qcomb QT16 hd=%u needs %lu B > device limit %lu B, using QT8\n",
                        head_dim, (unsigned long)(xpick.sfloats * sizeof(float)),
                        (unsigned long)[g.device maxThreadgroupMemoryLength]);
            }
        }
        [g.enc setComputePipelineState:pick.pipe];
        const uint32_t qt_n = pick.qt;
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
            // WHICH SHAPE, not just which branch. The branch name alone cannot show that a
            // route or a blk actually changed, and "the timing moved" is not proof that it
            // did -- qcomb_blk was set, stored and swept for a long time while reaching no
            // dispatch at all.
            fprintf(stderr,
                    "attn branch qcomb layer=%u n_tok=%u hd=%u blk=%u qt=%u pt=%u "
                    "threads=%u\n",
                    kv_layer, n_tok, head_dim, g_qcomb_blk, pick.qt, pick.pt, pick.threads);
        }
        ensure_kv_pt(kv_layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        // The device tail scratch. TMP is a standalone allocation, never in the arena
        // overlap, so nothing live aliases it during attention.
        [g.enc setBuffer:g.bufs[B_TMP] offset:g.buf_off[B_TMP] atIndex:13];
        if (axh) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:16]; }
        else     { [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:16]; }   // a stand-in, never written
        const uint32_t axh_flag = axh ? 1u : 0u;
        [g.enc setBytes:&axh_flag length:4 atIndex:17];
        // Staged Q + scores + max/sum/diagonal, plus the threadgroup tail spill at 256
        // (at 512 the spill is in device memory instead).
        // THE SAME TILE the threadgroup size was computed from. One value, one row: the
        // kernel's stride and the memory it is given cannot disagree.
        [g.enc setBytes:&pick.pt length:4 atIndex:19];
        [g.enc setThreadgroupMemoryLength:pick.sfloats * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        // The thread count must equal the instantiation's NSG * 32: the kernel pins a
        // dim slice to each simdgroup.
        g_qcomb_nsg_live = pick.threads / 32u;
        [g.enc dispatchThreadgroups:
            MTLSizeMake(n_heads, (n_tok + qt_n - 1) / qt_n, 1)
              threadsPerThreadgroup:MTLSizeMake(pick.threads, 1, 1)];
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
        // THE TILE FOLLOWS THE DEVICE HERE TOO. It used to be a literal 128 that "must
        // match the PT the selected kernel was instantiated with" -- a coupling that only
        // existed because the kernel had it compiled in. It is passed now, so the only
        // rule left is that the threadgroup size below is computed from the SAME value.
        //
        // qtile's layout is its own: scores QT*PT plus the running output QT*head_dim,
        // where qcomb also holds a staged Q and a spill. Inverting THIS layout for the
        // widest tile that fits is the same arithmetic qcomb_derive_pt does, so it is
        // done here rather than borrowed.
        const uint32_t QT_H = tall ? 16 : 8;
        const NSUInteger tg_floats =
            (NSUInteger)[g.device maxThreadgroupMemoryLength] / sizeof(float);
        const NSUInteger qthreads = 256;
        const NSUInteger fixed_h =
            (NSUInteger)QT_H * head_dim + 2u * QT_H + (qthreads / 32);
        const uint32_t pt_fits = fixed_h >= tg_floats
            ? 8u
            : (uint32_t)(((tg_floats - fixed_h) / QT_H) / 8u) * 8u;
        // Tuned inside that bound, exactly as the qcomb tile is: the widest that FITS is
        // not the fastest -- a wider tile costs threadgroup occupancy.
        const uint32_t PT_H = pt_fits < 8u ? 8u : min_u32(g_qcomb_pt, pt_fits);
        // READ THE SCRATCH THE DEQUANT JUST FILLED, exactly as the qcomb branch does.
        // This kernel reads halves. Binding the cache itself when it holds q4_0 or q8_0
        // makes it read packed nibbles and block scales as halves, and the logits come
        // out non-finite. The dequant above ran for every prefill with a quantized
        // cache, so the scratch was written and then never read: the only branch that
        // bound it was the one gated on head_dim 256/512, and LFM2 is head_dim 64.
        haz(hb(B_Q) | (kq ? hb(B_KDQ) : HZ_KVK) | (vq ? hb(B_VDQ) : HZ_KVV),
            hb(B_ATTN));
        [g.enc setComputePipelineState:
            (tall && head_dim == 256 && g.p_attn_pre_qtile16h != nil)
              ? (g_attn_blk == 2 && g.p_attn_pre_qtile16h2 != nil ? g.p_attn_pre_qtile16h2
               : g_attn_blk == 8 && g.p_attn_pre_qtile16h8 != nil ? g.p_attn_pre_qtile16h8
                                                              : g.p_attn_pre_qtile16h)
                  : (tall ? g.p_attn_pre_qtile16 : g.p_attn_pre_qtile)];
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
            fprintf(stderr, "attn branch qtile layer=%u n_tok=%u\n", kv_layer, n_tok);
        }
        ensure_kv_pt(kv_layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        const uint32_t QT_R = QT_H;
        if (getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn branch qtile layer=%u n_tok=%u hd=%u qt=%u pt=%u\n",
                    kv_layer, n_tok, head_dim, QT_H, PT_H);
        }
        // The SAME PT_H the kernel is given, so the memory and the stride agree.
        [g.enc setBytes:&PT_H length:4 atIndex:19];
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
    const uint32_t hdi_e = decode_hd_index(head_dim);
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
    // THE FLASH-DECODING ROUTE ENGAGES ON ITS OWN TERMS. The sliced path below (which is
    // where the fd and streaming kernels live) used to be reachable only through the
    // score-tile gate above, i.e. only when direct_tgs < attn_min_tgs or the span passed
    // 8192 keys. attn_min_tgs is the score-tile kernel's slice count -- the tuner ranks it
    // as that -- but through this gate it also decided whether the fd kernel ran at all
    // below 8192 keys: ranked at 8 (LFM2's 32 head-threadgroups no longer under it) the
    // direct kernel ran instead and decode at 455 keys lost 6 % (review #116, #120). The
    // fd conditions are re-evaluated verbatim inside the block; this only opens the door.
    {
        const uint32_t share_all = n_kv > 0u ? n_heads / n_kv : 0u;
        const int fd_idx = share_all == 2u ? 0 : share_all == 4u ? 1 : share_all == 8u ? 2 : -1;
        const bool fd_ok = n_tok == 1 && !kv_quant && slot_of == 0u && head_dim <= 128u
                        && window == 0u && fd_idx >= 0 && n_heads % n_kv == 0u
                        && n_pos > g_attn_fd_chunk && g.p_attn_dec_fd[fd_idx] != nil
                        && g_attn_fd != 0u;
        if (fd_ok) { slices = std::max(slices, 2u); }
    }
    // THE VECTOR DECODE KERNEL: one query, f16 cache, a span within its regime (at most
    // attn_vec_max_keys keys, cache-resident). One query head per threadgroup, a simdgroup
    // per position, the output written directly -- no partials, no combine.
    if (n_tok == 1 && !kv_quant && slot_of < 2u && g.p_attn_dec_vec[slot_of] != nil
        && n_kv > 0u && n_heads % n_kv == 0u && n_pos <= g_attn_vec_max_keys
        && attn_vec_enabled()) {
        if (getenv("IMPARO_ATTN_WHICH")) {
            fprintf(stderr, "attn branch vec layer=%u hd=%u slot=%u nsg=%u n_pos=%u window=%u\n",
                    kv_layer, head_dim, slot_of, ATTN_VEC_NSG, n_pos, window);
        }
        haz(hb(B_Q) | HZ_KVK | HZ_KVV, hb(B_ATTN));
        [g.enc setComputePipelineState:g.p_attn_dec_vec[slot_of]];
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
        // The page table: a full-attention layer resolves positions through it (a windowed
        // layer's ring mask never dereferences it). Left unbound, the kernel read whichever
        // table the encoder last had at this index -- the decode_agree gate caught that at
        // exactly the depth where the full layers first fell under the span limit.
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        // The merge runs in two rounds over half the simdgroups' slots (see the kernel), so
        // the buffer is NSG/2 x (hd + 2) floats: 16.4 KB at hd 512 inside the 32 KB limit.
        [g.enc setThreadgroupMemoryLength:(NSUInteger)(ATTN_VEC_NSG / 2u) * (head_dim + 2u) * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n_tok, 1)
              threadsPerThreadgroup:MTLSizeMake(ATTN_VEC_NSG * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        return;
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
        const uint32_t hdi = decode_hd_index(head_dim);
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
        // FLASH-DECODING FOR SMALL HEAD DIMS: f16 cache, no window, the whole GQA share in one
        // threadgroup (HQ = share, one threadgroup per KV head), the context in ~512-key
        // slices. Selected ahead of the stream/scoretile kernels where it is built; see the
        // kernel for why the scoretile pair is 2-5x off at hd 64.
        const uint32_t share_all = n_kv > 0u ? n_heads / n_kv : 0u;
        const int fd_idx = share_all == 2u ? 0 : share_all == 4u ? 1 : share_all == 8u ? 2 : -1;
        const bool want_fd = !kv_quant && slot_of == 0u && head_dim <= 128u && window == 0u
                          && fd_idx >= 0 && n_heads % n_kv == 0u && n_pos > g_attn_fd_chunk
                          && g.p_attn_dec_fd[fd_idx] != nil && g_attn_fd != 0u;
        if (want_fd) {
            // one chunk (the knob) per threadgroup, as many slices as that takes, within the
            // slice cap; below two slices the route above serves the short context
            slices = std::min(ATTN_MAX_SLICES, (n_pos + g_attn_fd_chunk - 1u) / g_attn_fd_chunk);
            slices = std::max(2u, slices);
        }
        const bool streaming = !want_fd && want_stream && stream_sel != nil;
        if (want_fd) {
            sp_sel = g.p_attn_dec_fd[fd_idx];
        } else if (streaming) {
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
                    want_fd ? "fd" : streaming ? "stream" : "split", kv_layer, head_dim,
                    want_fd ? share_all : streaming ? hq_grp : (gqa ? hq : 1u), ident ? 1u : 0u,
                    kv_quant ? 1u : 0u, slices, n_tok);
        }
        ensure_kv_pt(kv_layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
        [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:18];
        NSUInteger sthreads = (n_tok > 1) ? g_attn_threads_pf : g_attn_threads;
        // 128, AND IT IS MEASURED, not inherited from the reference. Racing it against
        // the score-tile width (which needed the accumulator in registers to fit at all):
        //     11941   512 threads 32.6 / 32.6    128 threads 35.9 / 36.1
        //     24003   512 threads 27.0           128 threads 32.6 / 32.5
        // More simdgroups means fewer positions each, while the per-block barriers and the
        // merge both scale with nsg.
        if (streaming) { sthreads = 128; }
        if (want_fd) { sthreads = 128; }
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
        if (want_fd) {
            // HQ x chunk scores; Q staging + per-simdgroup maxima, sums and P x V partials
            const uint32_t chunk = (n_pos + slices - 1u) / slices;
            const uint32_t hq_fd = share_all, nsg_fd = 4u;
            [g.enc setThreadgroupMemoryLength:(NSUInteger)hq_fd * chunk * sizeof(float) atIndex:0];
            [g.enc setThreadgroupMemoryLength:(NSUInteger)(hq_fd * head_dim + 2u * nsg_fd * hq_fd
                                                            + nsg_fd * hq_fd * head_dim) * sizeof(float) atIndex:1];
        } else {
            [g.enc setThreadgroupMemoryLength:slice_scores * 4 * sc_mul atIndex:0];
            [g.enc setThreadgroupMemoryLength:red_n * sizeof(float) atIndex:1];
        }
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        const uint32_t grid_x = want_fd ? n_kv
                              : streaming ? n_heads / hq_grp
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
    ensure_kv_pt(kv_layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
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
    if (g_skip_cat == PC_SHORTCONV) { return; }
    if (g.p_shortconv == nil || g.p_shortconv_state == nil) { return; }
    // ONE TOKEN: one dispatch for the conv output and the state shift (IMPARO_SHORTCONV_STEP=0
    // keeps the two dispatches, the A/B arm). No half mirror at one token.
    static int step_on = -1;
    if (step_on < 0) { const char * e = getenv("IMPARO_SHORTCONV_STEP"); step_on = !(e && e[0] == '0'); }
    if (n_tok == 1u && step_on && g.p_shortconv_step != nil) {
        haz(hb(bcx) | hb(state), hb(out) | hb(state));
        [g.enc setBuffer:g.bufs[bcx] offset:g.buf_off[bcx] atIndex:0];
        { const WSeg & ws = wseg_at(w_off);
          [g.enc setBuffer:ws.buf offset:(NSUInteger)(w_off - ws.base) atIndex:1]; }
        [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
                 atIndex:2];
        [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] atIndex:3];
        [g.enc setBytes:&width length:4 atIndex:4];
        [g.enc setBytes:&kern length:4 atIndex:5];
        dispatch1(g.p_shortconv_step, width, PC_SHORTCONV);
        return;
    }
    // Dual-write the half mirror when the out_proj GEMM will read it (the same rule the
    // norm and add producers use): contiguous [token][channel] rows, prefill scale only.
    const bool xh_on = g_half_a != 0u && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                    && (uint64_t)n_tok * width * 2ull <= g.sizes[B_XH]
                    && half_consumers_exist();
    haz(hb(bcx) | hb(state), hb(out) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_shortconv];
    [g.enc setBuffer:g.bufs[bcx] offset:g.buf_off[bcx] atIndex:0];
    { const WSeg & ws = wseg_at(w_off);
      [g.enc setBuffer:ws.buf offset:(NSUInteger)(w_off - ws.base) atIndex:1]; }
    [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
             atIndex:2];
    [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] atIndex:3];
    [g.enc setBytes:&width length:4 atIndex:4];
    [g.enc setBytes:&kern length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBuffer:xh_on ? g.bufs[B_XH] : g.bufs[out]
             offset:xh_on ? g.buf_off[B_XH] : g.buf_off[out] atIndex:7];
    const uint32_t xh_flag = xh_on ? 1u : 0u;
    [g.enc setBytes:&xh_flag length:4 atIndex:8];
    dispatch1(g.p_shortconv, n_tok * width, PC_SHORTCONV);
    if (xh_on) { g_xh_src = out; g_xh_elems = (uint64_t)n_tok * width; g_xh_buf = B_XH; }

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
    dispatch1(g.p_shortconv_state, width, PC_SHORTCONV);
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
    if (g_skip_cat == PC_SHORTCONV) { return; }
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
    dispatch1(g.p_shortconv_state, width, PC_SHORTCONV);
}

// The mega FFN block: gate|up -> act*mul -> down as one persistent dispatch (kernel comment in
// imparo.metal). `gtmp` and `utmp` are the two n_mid scratch rows the dispatch path writes
// too. Returns false when the route is off or unavailable; the caller then dispatches.
extern "C" bool imparo_metal_ffn_persistent(uint64_t gate_off, uint64_t up_off, uint64_t down_off,
                                            uint32_t n_in, uint32_t n_mid, uint32_t n_out,
                                            uint32_t src, uint32_t gtmp, uint32_t utmp, uint32_t dst) {
    if (!mega_ffn_wanted() || !mega_route_open() || g.p_mega_ffn == nil || g.mega_sync == nil) { return false; }
    mega_inject_if_pending();
    if (n_in % 32u != 0u || n_mid % 32u != 0u || w_absent(gate_off) || w_absent(up_off) || w_absent(down_off)) { return false; }
    haz(hb(src), hb(gtmp) | hb(utmp) | hb(dst));
    [g.enc setComputePipelineState:g.p_mega_ffn];
    const uint64_t go = wbind(g.enc, gate_off, 0);
    const uint64_t uo = wbind(g.enc, up_off, 1);
    const uint64_t dn = wbind(g.enc, down_off, 2);
    [g.enc setBuffer:g.bufs[src]  offset:g.buf_off[src]  atIndex:3];
    [g.enc setBuffer:g.bufs[gtmp] offset:g.buf_off[gtmp] atIndex:4];
    [g.enc setBuffer:g.bufs[utmp] offset:g.buf_off[utmp] atIndex:5];
    [g.enc setBuffer:g.bufs[dst]  offset:g.buf_off[dst]  atIndex:6];
    [g.enc setBuffer:g.mega_sync offset:0 atIndex:7];
    [g.enc setBytes:&go length:8 atIndex:8];
    [g.enc setBytes:&uo length:8 atIndex:9];
    [g.enc setBytes:&dn length:8 atIndex:10];
    [g.enc setBytes:&n_in  length:4 atIndex:11];
    [g.enc setBytes:&n_mid length:4 atIndex:12];
    [g.enc setBytes:&n_out length:4 atIndex:13];
    [g.enc setBytes:&g_mega_tgs length:4 atIndex:14];
    static bool said = false;
    if (!said) { NSLog(@"imparo metal: mega FFN block engaged (n_in=%u n_mid=%u n_out=%u tgs=%u threads=%u)", n_in, n_mid, n_out, g_mega_tgs, g_mega_nsg * 32u); said = true; }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MEGA); }
    [g.enc dispatchThreadgroups:MTLSizeMake(g_mega_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(g_mega_nsg * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// THE ENTRY (task #158 step 2): one shape for every architecture. MegaEntryGHost mirrors the
// kernel's MegaEntryG (GPU addresses = MTLBuffer.gpuAddress + offset, the argument-buffer
// encoding of pointers; weight offsets local to the address's segment; words; floats); the slot
// enums come from build.rs (mega_slots.h), the same table the shader's constants come from.
struct MegaEntryGHost { uint64_t ptr[MEGA_NP]; uint64_t off[MEGA_NO]; uint32_t u[MEGA_NU]; float f[MEGA_NF]; };
static_assert(sizeof(MegaEntryGHost) == MEGA_NP * 8u + MEGA_NO * 8u + MEGA_NU * 4u + MEGA_NF * 4u, "MegaEntryG drift");
// THE ENTRY AS THE HOST SEES IT (imparo-metal/src/lib.rs MegaEntryFfi): the Rust side fills the
// program's words and floats, names every pointer slot's ROLE and source, and states what the
// bridge decides from: the pipeline family (arch), the level the entry needs, the head dim (the
// pipeline slot), whether the attention phase runs, whether the entry writes the cache row, the
// cache layer and the position. The bridge resolves addresses, checks presence and the fast
// tier, applies the grid and geometry rules, derives the split / body / grid words into the
// header, forms the hazard masks from the roles, and records or dispatches. Architecture rules
// (weight kinds, width divisibility, kernel taps) are the Rust side's, next to the program.
enum : uint32_t {
    MEGA_SLOT_NONE = 0u,        // unused in this entry: a valid address the kernel never reads
    MEGA_SLOT_WEIGHT = 1u,      // a weight the kernel reads: off = global offset, id = the offset-table slot for the segment-local offset (MEGA_NO = none); absent or slow-tier refuses
    MEGA_SLOT_WEIGHT_OPT = 2u,  // a weight read only under a flag the kernel also gets: absent = the dummy address, local offset 0
    MEGA_SLOT_BUF_R = 3u,       // an activation buffer: id = buffer id, off = element offset
    MEGA_SLOT_BUF_W = 4u,
    MEGA_SLOT_BUF_RW = 5u,
    MEGA_SLOT_KV_K_R = 6u,      // this layer's K cache, read view (attention over the cache)
    MEGA_SLOT_KV_K_W = 7u,      // the write view (this token's cache row)
    MEGA_SLOT_KV_V_R = 8u,
    MEGA_SLOT_KV_V_W = 9u,
    MEGA_SLOT_KV_PT = 10u,      // the layer's page table
};
struct MegaSlotFfi { uint32_t role, id; uint64_t off; };
struct MegaEntryFfi {
    uint32_t arch, min_level, head_dim, attn_on;
    uint32_t kv_write, kv_layer, start_pos, layer;
    uint32_t scratch_rows, n_freqs, pad0, pad1;
    const float * freqs;
    MegaSlotFfi slots[MEGA_NP];
    uint32_t u[MEGA_NU];
    float f[MEGA_NF];
};
static_assert(sizeof(MegaEntryFfi) == 12u * 4u + 8u + MEGA_NP * 16u + MEGA_NU * 4u + MEGA_NF * 4u, "MegaEntryFfi drift");
constexpr uint32_t MEGA_ARCH_GEMMA4 = 0u, MEGA_ARCH_LFM2 = 1u;
constexpr uint32_t SHORTCONV_MAX_HISTORY_HOST = 8u;   // the shader's SHORTCONV_MAX_HISTORY
struct MegaTokenHost { uint32_t start_pos, dbg_slot, n_tg, entry_bytes, dbg_seq, entry_index, n_entries, pad3; };   // mirrors MegaToken
// A weight's segment base address, and its offset local to that segment (the block adds
// them). Absent weights get a valid address they never read and keep their sentinel offset.
// Every referenced buffer is declared to the encoder: it is reached by address, not binding.
static uint64_t mega_waddr(uint64_t off, uint64_t & local) {
    if (w_absent(off)) { local = off; return (uint64_t)[g_wsegs.front().buf gpuAddress]; }
    const WSeg & sg = wseg_at(off);
    local = off - sg.base;
    [g.enc useResource:sg.buf usage:MTLResourceUsageRead];
    return (uint64_t)[sg.buf gpuAddress];
}
// The model's rope factor table as a constant-space buffer for the layer kernel (buffer 3): one
// per model, whatever its length, re-made when the table the workflow hands over changes; a
// 1-float dummy (never read: n_freqs is 0) when the layer has none.
static id<MTLBuffer> mega_freqs_buffer(const float * freqs, uint32_t n_freqs) {
    static id<MTLBuffer> buf = nil;
    static const float * src = nullptr;
    static uint32_t n = 0;
    const bool none = (freqs == nullptr || n_freqs == 0u);
    if (buf == nil || (none ? (n != 0u) : (src != freqs || n != n_freqs))) {
        const size_t bytes = none ? 4u : (size_t)n_freqs * 4u;
        buf = [g.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        rset_add(buf);   // read by the layer kernel's head phase: resident with the rest
        if (none) { *(float *)[buf contents] = 1.0f; } else { memcpy([buf contents], freqs, bytes); }
        src = none ? nullptr : freqs; n = none ? 0u : n_freqs;
    }
    return buf;
}
static uint64_t mega_baddr(uint32_t id) {
    [g.enc useResource:g.bufs[id] usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    return (uint64_t)[g.bufs[id] gpuAddress] + (uint64_t)g.buf_off[id];
}
static uint64_t mega_addr_of(id<MTLBuffer> b, uint64_t off, MTLResourceUsage usage) {
    [g.enc useResource:b usage:usage];
    return (uint64_t)[b gpuAddress] + off;
}
// Is `off` inside the segment `sibling` is bound from? (wlocal aborts on a miss; the mega
// route refuses instead, so a model whose norms straddle segments takes the dispatch path.)
static uint32_t mega_norm_threads(uint32_t width) {
    // imparo_metal_rms_norm_add_row's rule, so the norm phase's virtual geometry matches.
    const uint32_t vec_units = (width + 3) / 4;
    uint32_t threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    return std::max<uint32_t>(threads, 32);
}
// The program rings (task #153): 4 slots x MEGA_PROG_CAP entries per architecture, allocated at
// the first record (bound directly at the dispatch, so the encoder makes them resident; joined to
// the residency set for the regions after), never swapped.
static void mega_prog_alloc(void) {
    if (g_mega_prog != nil) { return; }
    g_mega_prog = [g.device newBufferWithLength:(NSUInteger)4u * MEGA_PROG_CAP * sizeof(MegaEntryGHost) options:MTLResourceStorageModeShared];
    if (g_mega_prog != nil) { rset_add(g_mega_prog); }
}
// Flush the pending program run as one dispatch (task #153). Nothing pending: a no-op.
static void mega_prog_flush(void) {
    if (g_prog_n == 0u) { return; }
    if (g_prog_foreign) {
        // Another dispatch was encoded while these entries waited: their order is lost. Drop
        // them and fail the region (the sticky word; the retire rolls the region back and the
        // dispatch path re-runs it). Loud: this is a code path bug, never a runtime condition.
        NSLog(@"imparo metal: mega program: a foreign dispatch was encoded with %u entries pending -- region failed on purpose", g_prog_n);
        if (g.mega_sync != nil) { ((uint32_t *)[g.mega_sync contents])[15] = 1u | (0xffdu << 4); }
        g_prog_n = 0u; g_prog_rd = 0ull; g_prog_wr = 0ull; g_prog_foreign = false;
        return;
    }
    g_prog_in_flush = true;
    haz(g_prog_rd, g_prog_wr);
    [g.enc setComputePipelineState:g_prog_pipe];
    [g.enc setBuffer:g_prog_buf offset:(NSUInteger)(g_prog_slot * MEGA_PROG_CAP) * g_prog_stride atIndex:0];
    [g.enc setBuffer:g.mega_sync offset:0 atIndex:1];
    MegaTokenHost tok; memset(&tok, 0, sizeof(tok));
    tok.start_pos = g_prog_start_pos; tok.dbg_slot = g_prog_slot; tok.n_tg = g_prog_tgs;
    tok.entry_bytes = g_prog_entry_bytes; tok.dbg_seq = g_prog_dbg_seq;
    tok.entry_index = g_prog_base; tok.n_entries = g_prog_n;
    [g.enc setBytes:&tok length:sizeof(tok) atIndex:2];
    if (g_prog_freqs != nil) { [g.enc setBuffer:g_prog_freqs offset:0 atIndex:3]; }
    [g.enc setThreadgroupMemoryLength:(NSUInteger)g_prog_tgmem atIndex:0];
    static bool said = false;
    if (!said) { said = true; NSLog(@"imparo metal: mega program engaged: %u entries in one dispatch (tgs=%u threads=%u)", g_prog_n, g_prog_tgs, g_mega_nsg * 32u); }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MEGA); }
    [g.enc dispatchThreadgroups:MTLSizeMake(g_prog_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(g_mega_nsg * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    g_prog_base += g_prog_n; g_prog_n = 0u; g_prog_rd = 0ull; g_prog_wr = 0ull;
    g_prog_in_flush = false;
}
// A refused layer runs on the dispatch path right after this call returns: the pending run must
// be encoded first. Every early return of the entry functions goes through this guard.
struct MegaProgRefuseGuard { bool ok = false; ~MegaProgRefuseGuard() { if (!ok) { mega_prog_flush(); } } };
// Record one entry into the pending run (starting a run, or flushing the previous one when this
// entry cannot join it); false when the slot's ring is full (the caller refuses the layer).
// `pos_matters`: the entry reads tok.start_pos (an attention layer); a layer that does not (LFM2's
// short convolution) joins any run and never fixes its position.
static bool mega_prog_record(id<MTLBuffer> ring, uint32_t entry_bytes, const void * entry,
                             id<MTLComputePipelineState> pipe, uint32_t tgs, uint32_t tgmem, id<MTLBuffer> freqs,
                             uint32_t start_pos, bool pos_matters, uint32_t dbg_seq, uint64_t rd, uint64_t wr) {
    if (ring == nil) { return false; }
    const bool joins = g_prog_n != 0u && g_prog_buf == ring && g_prog_pipe == pipe && g_prog_tgs == tgs
        && g_prog_tgmem == tgmem && g_prog_freqs == freqs && g_prog_slot == g_mega_region_slot
        && (!pos_matters || !g_prog_pos_set || g_prog_start_pos == start_pos) && g_prog_base + g_prog_n < MEGA_PROG_CAP;
    if (g_prog_n != 0u && !joins) { mega_prog_flush(); }
    if (g_prog_slot != g_mega_region_slot) { g_prog_base = 0u; }   // a new region: the slot's ring starts empty
    if (g_prog_base + g_prog_n >= MEGA_PROG_CAP) { return false; }
    if (g_prog_n == 0u) {
        g_prog_buf = ring; g_prog_stride = entry_bytes; g_prog_entry_bytes = entry_bytes;
        g_prog_pipe = pipe; g_prog_tgs = tgs; g_prog_tgmem = tgmem; g_prog_freqs = freqs;
        g_prog_slot = g_mega_region_slot; g_prog_start_pos = start_pos; g_prog_pos_set = pos_matters; g_prog_dbg_seq = dbg_seq;
    } else if (pos_matters && !g_prog_pos_set) { g_prog_start_pos = start_pos; g_prog_pos_set = true; }
    memcpy((uint8_t *)[ring contents] + ((size_t)(g_prog_slot * MEGA_PROG_CAP) + g_prog_base + g_prog_n) * entry_bytes, entry, entry_bytes);
    g_prog_rd |= rd; g_prog_wr |= wr; g_prog_n += 1u;
    return true;
}
// The model calls this after its layer loop; the region end calls it too.
extern "C" void imparo_metal_mega_program_end(void) { mega_prog_flush(); }

// The cache basis (task #156): a Hadamard width the kernel's bricks can apply to a head row of
// `hd` -- a power of two dividing hd, at least the per-lane width hd/32 (the combine's lane
// form), 0 = the plain basis.
static bool mega_had_ok(uint32_t had, uint32_t hd) {
    if (had == 0u) { return true; }
    if ((had & (had - 1u)) != 0u || hd % had != 0u) { return false; }
    return had >= std::max(1u, hd / 32u);
}
// ONE LAYER OF ANY ARCHITECTURE AS ONE PERSISTENT DISPATCH (task #158 step 2): see MegaEntryFfi.
// Refuses (false, nothing encoded) when the route is off or below the entry's level, a weight is
// absent or in the slow tier, a buffer or cache is missing, or a grid / geometry rule breaks; the
// caller then runs the dispatch path. A refusal flushes the pending program run first.
extern "C" bool imparo_metal_mega_layer(const MegaEntryFfi * e) {
    MegaProgRefuseGuard guard;   // a refusal flushes the pending program run first (task #153)
    if (e == nullptr || e->arch > MEGA_ARCH_LFM2 || mega_level() < (int)e->min_level || !mega_route_open() || g.mega_sync == nil) { return false; }
    mega_inject_if_pending();
    const bool g4 = e->arch == MEGA_ARCH_GEMMA4;
    const bool attn_on = e->attn_on != 0u;
    const uint32_t hd = e->head_dim;
    const uint32_t n_embd = e->u[MEGA_U_N_EMBD], n_heads = e->u[MEGA_U_N_HEADS], n_kv = e->u[MEGA_U_N_KV];
    const uint32_t window = e->u[MEGA_U_WINDOW], had_k = e->u[MEGA_U_HAD_K], had_v = e->u[MEGA_U_HAD_V];
    if (n_embd == 0u || n_embd % 4u != 0u) { return false; }
    // The pipeline family and the head-dim slot (only the attention body depends on the head dim).
    uint32_t slot_of = 2u;
    for (uint32_t i = 0; i < 2u; ++i) { if (hd != 0u && g_qcomb_hds[i] == hd) { slot_of = i; break; } }
    // A quantized cache (task #156) selects the _q variants, built with the process-wide K / V
    // types; a layer whose types differ from that pair is refused (the dispatch path's rule).
    const uint32_t kv_kt = attn_on ? kv_eff_type(e->kv_layer, 0) : 1u, kv_vt = attn_on ? kv_eff_type(e->kv_layer, 1) : 1u;
    const bool kvq = kv_kt != 1u || kv_vt != 1u;
    if (kvq && (kv_kt != g_mega_kq_ty || kv_vt != g_mega_vq_ty)) { return false; }
    const bool prog = mega_program_for(g4 ? MEGA_PROGRAM_DEFAULT_E4B : MEGA_PROGRAM_DEFAULT_LFM2);   // the program form (task #153)
    __strong id<MTLComputePipelineState> * plain = g4 ? g.p_mega_layer : g.p_mega_lfm2;
    __strong id<MTLComputePipelineState> * qp    = g4 ? g.p_mega_layer_q : g.p_mega_lfm2_q;
    __strong id<MTLComputePipelineState> * deepp = g4 ? g.p_mega_layer_deep : g.p_mega_lfm2_deep;
    __strong id<MTLComputePipelineState> * deepq = g4 ? g.p_mega_layer_deep_q : g.p_mega_lfm2_deep_q;
    __strong id<MTLComputePipelineState> * progp = g4 ? g.p_mega_layer_prog : g.p_mega_lfm2_prog;
    __strong id<MTLComputePipelineState> * progq = g4 ? g.p_mega_layer_prog_q : g.p_mega_lfm2_prog_q;
    id<MTLComputePipelineState> pipe = nil;
    if (attn_on) {
        if (slot_of >= 2u) { return false; }
        pipe = prog ? (kvq ? progq[slot_of] : progp[slot_of]) : (kvq ? qp[slot_of] : plain[slot_of]);
    } else {
        // Without the attention phase the head dim plays no part: any built slot serves the layer.
        __strong id<MTLComputePipelineState> * set = prog ? progp : plain;
        pipe = set[0] != nil ? set[0] : set[1];
    }
    if (pipe == nil) { return false; }
    // The quantized and program variants run at one threadgroup per core (tasks #156, #153).
    uint32_t tgs = (kvq || prog) ? mega_deep_tgs() : g_mega_tgs;
    bool deep = false;
    uint32_t attn_split = 1u, n_pos = 0u;
    if (attn_on) {
        // The attention phase: one token, a span within the vec limit or the grouped deep body,
        // the fold geometry (nsg a multiple of the slot count, at least twice it), the block's
        // threadgroup row holding the fold scratch, this layer's cache present.
        if (n_kv == 0u || n_heads == 0u || n_heads % n_kv != 0u) { return false; }
        if (!kvq && !prog && n_heads > tgs) { return false; }   // the f16 plain pipelines compute one vec item per threadgroup (constraint 8)
        if (!mega_had_ok(had_k, hd) || !mega_had_ok(had_v, hd)) { return false; }   // the cache basis (task #156)
        if (g_mega_nsg % 4u != 0u || g_mega_nsg < 8u) { return false; }
        if (!g4 && g_mega_nsg % std::max(1u, hd / Q8_TM_UNIT_ROWS_HOST) != 0u) { return false; }   // LFM2: a head row's Q8 units inside one threadgroup
        if ((uint64_t)n_embd < (uint64_t)4u * (hd + 2u)) { return false; }
        if (e->kv_layer >= g.kv_k.size() || g.kv_k[e->kv_layer] == nil || g.kv_v[e->kv_layer] == nil) { return false; }
        n_pos = (window > 0 && e->start_pos + 1 > window) ? window : e->start_pos + 1;
        // Threadgroups per head: the idle ones take a share of the span; the partials live in the
        // block's scratch (scratch_rows floats), which bounds the split. The vec regime cap
        // (attn_vec_max_keys) is a per-threadgroup span; past it the grouped deep body (task
        // #151) takes the layer, or the refusal stands with IMPARO_MEGA_ATTN_SLICED=0 (the A/B).
        attn_split = std::max(1u, std::min(tgs / n_heads, e->scratch_rows / std::max(1u, n_heads * (hd + 2u))));
        if ((n_pos + attn_split - 1u) / attn_split > g_attn_vec_max_keys) {
            id<MTLComputePipelineState> dpipe = prog ? pipe : (kvq ? deepq[slot_of] : deepp[slot_of]);
            if (!mega_attn_sliced() || dpipe == nil) { return false; }
            attn_split = mega_attn_deep_slices(n_heads, n_kv, hd);
            if (attn_split == 0u) { return false; }
            deep = true; pipe = dpipe; tgs = mega_deep_tgs();
        }
    }
    // Pass 1 over the slots: presence, the fast tier (task #154), buffers exist, the hazard masks
    // from the roles. No encoder state is touched here (haz() comes first).
    uint64_t rd = 0ull, wr = 0ull;
    for (uint32_t i = 0; i < MEGA_NP; ++i) {
        const MegaSlotFfi & sl = e->slots[i];
        switch (sl.role) {
        case MEGA_SLOT_NONE: break;
        case MEGA_SLOT_WEIGHT:
            if (w_absent(sl.off) || !w_fast(sl.off)) { return false; }
            break;
        case MEGA_SLOT_WEIGHT_OPT:
            if (!w_fast(sl.off)) { return false; }
            break;
        case MEGA_SLOT_BUF_R: case MEGA_SLOT_BUF_W: case MEGA_SLOT_BUF_RW:
            if (sl.id >= B_COUNT || g.bufs[sl.id] == nil) { return false; }
            if (sl.role != MEGA_SLOT_BUF_W) { rd |= hb(sl.id); }
            if (sl.role != MEGA_SLOT_BUF_R) { wr |= hb(sl.id); }
            break;
        case MEGA_SLOT_KV_K_R: case MEGA_SLOT_KV_K_W: case MEGA_SLOT_KV_V_R: case MEGA_SLOT_KV_V_W: {
            if (!attn_on) { return false; }
            const uint64_t hz = (sl.role == MEGA_SLOT_KV_V_R || sl.role == MEGA_SLOT_KV_V_W) ? HZ_KVV : HZ_KVK;
            rd |= hz;
            if (sl.role == MEGA_SLOT_KV_K_W || sl.role == MEGA_SLOT_KV_V_W) { wr |= hz; }
            break;
        }
        case MEGA_SLOT_KV_PT:
            if (!attn_on || e->kv_layer >= g.kv_pt.size()) { return false; }
            break;
        default: return false;
        }
    }
    if (!mega_scratch_ensure(e->scratch_rows)) { return false; }
    if (attn_on) { ensure_kv_pt(e->kv_layer, (e->start_pos + 1u + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS); }
    if (attn_on && getenv("IMPARO_ATTN_WHICH")) {
        fprintf(stderr, "mega attn body=%s layer=%u hd=%u n_pos=%u split=%u hq=%u tgs=%u kvq=%u\n",
                deep ? "grouped" : "vec", e->kv_layer, hd, n_pos, attn_split, deep ? mega_attn_hq(hd) : 0u, tgs, kvq ? 1u : 0u);
    }
    // The hazard check comes BEFORE any encoder state is bound (see haz()); the program form
    // defers it to the run's flush (nothing is bound at record time).
    if (!prog) { haz(rd, wr); }
    [g.enc setComputePipelineState:pipe];
    // Pass 2: the addresses (each declares its buffer to the encoder), then the derived header words.
    MegaEntryGHost ent; memset(&ent, 0, sizeof(ent));
    memcpy(ent.u, e->u, sizeof(ent.u)); memcpy(ent.f, e->f, sizeof(ent.f));
    const uint64_t none = (uint64_t)[g_wsegs.front().buf gpuAddress];   // a valid address for an operand the entry never touches
    for (uint32_t i = 0; i < MEGA_NP; ++i) {
        const MegaSlotFfi & sl = e->slots[i];
        switch (sl.role) {
        case MEGA_SLOT_WEIGHT: case MEGA_SLOT_WEIGHT_OPT: {
            if (w_absent(sl.off)) { ent.ptr[i] = none; break; }   // an absent optional weight: the dummy address, local offset 0
            uint64_t local = 0ull;
            ent.ptr[i] = mega_waddr(sl.off, local);
            if (sl.id < MEGA_NO) { ent.off[sl.id] = local; }
            break;
        }
        case MEGA_SLOT_BUF_R: case MEGA_SLOT_BUF_W: case MEGA_SLOT_BUF_RW:
            ent.ptr[i] = mega_baddr(sl.id) + sl.off * 4ull;
            break;
        case MEGA_SLOT_KV_K_R: case MEGA_SLOT_KV_K_W: case MEGA_SLOT_KV_V_R: case MEGA_SLOT_KV_V_W: {
            const bool is_v = sl.role == MEGA_SLOT_KV_V_R || sl.role == MEGA_SLOT_KV_V_W;
            const bool is_w = sl.role == MEGA_SLOT_KV_K_W || sl.role == MEGA_SLOT_KV_V_W;
            ent.ptr[i] = mega_addr_of(is_v ? g.kv_v[e->kv_layer] : g.kv_k[e->kv_layer], kv_reg(e->kv_layer, is_v),
                                      is_w ? (MTLResourceUsageRead | MTLResourceUsageWrite) : MTLResourceUsageRead);
            break;
        }
        case MEGA_SLOT_KV_PT:
            ent.ptr[i] = mega_addr_of(g.kv_pt[e->kv_layer], 0, MTLResourceUsageRead);
            break;
        default: ent.ptr[i] = none; break;
        }
    }
    ent.u[MEGA_U_N_TG] = tgs;   // the deep body and the quantized / program variants run at one threadgroup per core
    ent.u[MEGA_U_SCRATCH] = g_mega_scratch_words;
    ent.u[MEGA_U_NORM_T] = mega_norm_threads(n_embd);
    ent.u[MEGA_U_NORM_T_HEAD] = attn_on ? mega_norm_threads(hd) : 0u;
    ent.u[MEGA_U_ATTN_SPLIT] = attn_split;
    ent.u[MEGA_U_ATTN_BODY] = deep ? 1u : 0u;
    ent.u[MEGA_U_ATTN_HQ] = deep ? mega_attn_hq(hd) : 0u;   // the host's heads per item; the kernel checks it against its own rule
    MegaTokenHost tok; memset(&tok, 0, sizeof(tok));
    tok.start_pos = e->start_pos; tok.dbg_slot = g_mega_region_slot; tok.n_tg = tgs;
    tok.entry_bytes = (uint32_t)sizeof(MegaEntryGHost); tok.dbg_seq = std::min(g_mega_dbg_seq, MEGA_DBG_SEQS - 1u);
    if (g_mega_dbg_seq == 0u && attn_on) { g_mega_dbg_cap_layer = e->kv_layer; }   // the KV capture's layer (MEGA_DBG)
    g_mega_dbg_seq += 1u;
    id<MTLBuffer> fb = mega_freqs_buffer(e->n_freqs != 0u ? e->freqs : nullptr, e->n_freqs);   // the rope factor table, or the 1-float dummy
    if (prog) {
        // The program form (task #153): the entry joins the pending run; the run is one dispatch.
        mega_prog_alloc();
        if (!mega_prog_record(g_mega_prog, (uint32_t)sizeof(MegaEntryGHost), &ent, pipe, tgs, n_embd * 4u, fb,
                              e->start_pos, attn_on, tok.dbg_seq, rd, wr)) { return false; }
        guard.ok = true;
        return true;
    }
    [g.enc setBytes:&ent length:sizeof(ent) atIndex:0];
    [g.enc setBuffer:g.mega_sync offset:0 atIndex:1];
    [g.enc setBytes:&tok length:sizeof(tok) atIndex:2];
    [g.enc setBuffer:fb offset:0 atIndex:3];
    // The row every threadgroup forms for itself (n_embd floats of threadgroup memory).
    [g.enc setThreadgroupMemoryLength:(NSUInteger)n_embd * 4u atIndex:0];
    static bool said[2] = { false, false };
    if (!said[e->arch]) {
        said[e->arch] = true;
        NSLog(@"imparo metal: mega %s layer block engaged (level=%d attn=%u kv_write=%u n_embd=%u scratch_rows=%u tgs=%u threads=%u)",
              g4 ? "gemma4" : "LFM2", mega_level(), attn_on ? 1u : 0u, e->kv_write, n_embd, e->scratch_rows, tgs, g_mega_nsg * 32u);
    }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MEGA); }
    [g.enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(g_mega_nsg * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    guard.ok = true;
    return true;
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
        dispatch1(g.p_actmul4, n4, PC_ELEMENTWISE);
        return;
    }
    [g.enc setComputePipelineState:g.p_actmul];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_actmul, n, PC_ELEMENTWISE);
}
extern "C" void imparo_metal_act(uint32_t a, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a), hb(a));
    if ((n % 4u) == 0u && g.p_act4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_act4];
        [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
        [g.enc setBytes:&n4 length:4 atIndex:1];
        dispatch1(g.p_act4, n4, PC_ELEMENTWISE);
        return;
    }
    [g.enc setComputePipelineState:g.p_act];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBytes:&n length:4 atIndex:1];
    dispatch1(g.p_act, n, PC_ELEMENTWISE);
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
                    && half_consumers_exist();
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
        if (g.bufs[B_XH] != nil) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:3]; }   // read only when xh_on
        else { [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:3]; }                                // a stand-in, never read
        const uint32_t xh_flag = xh_on ? 1u : 0u;
        [g.enc setBytes:&xh_flag length:4 atIndex:4];
        dispatch1(g.p_add4, n4, PC_ELEMENTWISE);
        if (xh_on) { g_xh_src = a; g_xh_elems = n; g_xh_buf = B_XH; }
        return;
    }
    [g.enc setComputePipelineState:g.p_add];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_add, n, PC_ELEMENTWISE);
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
        dispatch1(g.p_scale4, n4, PC_ELEMENTWISE);
        return;
    }
    [g.enc setComputePipelineState:g.p_scale];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBytes:&k length:4 atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_scale, n, PC_ELEMENTWISE);
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
        dispatch1(g.p_addscale4, n4, PC_ELEMENTWISE);
        return;
    }
    [g.enc setComputePipelineState:g.p_addscale];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&k length:4 atIndex:2];
    [g.enc setBytes:&n length:4 atIndex:3];
    dispatch1(g.p_addscale, n, PC_ELEMENTWISE);
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
        dispatch1(g.p_copy4, n4, PC_ELEMENTWISE);
        return;
    }
    [g.enc setComputePipelineState:g.p_copy];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_copy, n, PC_ELEMENTWISE);
}
// n floats from src[src_off..] to dst[dst_off..], element offsets. The caller keeps the
// ranges disjoint when dst == src (the tail-row move in gemma4's final prefill chunk).
extern "C" void imparo_metal_copy_range(uint32_t dst, uint32_t dst_off, uint32_t src,
                                        uint32_t src_off, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(src), hb(dst));
    const NSUInteger doff = g.buf_off[dst] + (NSUInteger)dst_off * 4u;
    const NSUInteger soff = g.buf_off[src] + (NSUInteger)src_off * 4u;
    // copy4 reads float4: every offset must stay 16-byte aligned.
    if ((n % 4u) == 0u && (dst_off % 4u) == 0u && (src_off % 4u) == 0u && g.p_copy4 != nil) {
        const uint32_t n4 = n / 4u;
        [g.enc setComputePipelineState:g.p_copy4];
        [g.enc setBuffer:g.bufs[dst] offset:doff atIndex:0];
        [g.enc setBuffer:g.bufs[src] offset:soff atIndex:1];
        [g.enc setBytes:&n4 length:4 atIndex:2];
        dispatch1(g.p_copy4, n4, PC_ELEMENTWISE);
        return;
    }
    [g.enc setComputePipelineState:g.p_copy];
    [g.enc setBuffer:g.bufs[dst] offset:doff atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:soff atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_copy, n, PC_ELEMENTWISE);
}
extern "C" void imparo_metal_softcap(uint32_t a, float cap, uint32_t n) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    haz(hb(a), hb(a));
    [g.enc setComputePipelineState:g.p_softcap];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBytes:&cap length:4 atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    dispatch1(g.p_softcap, n, PC_ELEMENTWISE);
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
extern "C" void imparo_metal_argmax_feed(uint32_t src, uint32_t tokens, uint32_t pick,
                                         uint32_t pick_slot, uint32_t n) {
    haz(hb(src), hb(tokens) | hb(pick));
    [g.enc setComputePipelineState:g.p_argmax_feed];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[tokens] offset:g.buf_off[tokens] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    [g.enc setBuffer:g.bufs[pick] offset:g.buf_off[pick] atIndex:3];
    [g.enc setBytes:&pick_slot length:4 atIndex:4];
    NSUInteger tw = std::min<NSUInteger>(1024, g.p_argmax_feed.maxTotalThreadsPerThreadgroup);
    while (tw & (tw - 1)) { tw &= tw - 1; }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
    [g.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
}
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
    const uint64_t w_offset_l = wbind(g.enc, w_offset, 1);
    [g.enc setBuffer:g.bufs[tokens_buf] offset:g.buf_off[tokens_buf] atIndex:2];
    [g.enc setBytes:&w_offset_l length:8 atIndex:3];
    [g.enc setBytes:&width length:4 atIndex:4];
    [g.enc setBytes:&emb_scale length:4 atIndex:5];
    [g.enc setBytes:&comb_scale length:4 atIndex:6];
    [g.enc setBytes:&n_tok length:4 atIndex:7];
    const uint32_t staged = 0;
    [g.enc setBytes:&staged length:4 atIndex:8];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_PLE); }
    [g.enc dispatchThreads:MTLSizeMake(width / 4, n_tok, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

// Host-staged tier, host side: copy rows ids[0..n) of the tensor at file offset `off` into
// `dst` in order. Plain memcpy like imparo_metal_write: the caller writes between
// regions (next to the token write), before any dispatch of the region that reads it.
// Returns 0 when `off` is not inside a host-staged segment of the placement (the GPU reads
// the table itself then), so the caller keeps one contract on every backend.
extern "C" int32_t imparo_metal_stage_rows(uint64_t off, uint32_t row_bytes,
                                           const uint32_t * ids, uint32_t n, uint32_t dst) {
    if (g_map_base == nullptr || row_bytes == 0) { return 0; }
    const WStaged * table = nullptr;
    for (const WStaged & t : g_wstaged) {
        if (off >= t.off && off - t.off < t.bytes) { table = &t; break; }
    }
    if (table == nullptr) { return 0; }
    const uint64_t rows = (table->off + table->bytes - off) / row_bytes;
    if ((uint64_t)n * row_bytes > g.sizes[dst]) {
        NSLog(@"imparo metal: stage_rows: %u rows of %u bytes exceed buffer %u (%zu bytes)",
              n, row_bytes, dst, g.sizes[dst]);
        abort();
    }
    uint8_t * out = (uint8_t *)[g.bufs[dst] contents] + g.buf_off[dst];
    const uint8_t * src = g_map_base + off;
    for (uint32_t i = 0; i < n; ++i) {
        if (ids[i] >= rows) {
            NSLog(@"imparo metal: stage_rows: row %u of %llu", ids[i], (unsigned long long)rows);
            abort();
        }
        std::memcpy(out + (uint64_t)i * row_bytes, src + (uint64_t)ids[i] * row_bytes, row_bytes);
    }
    return 1;
}

// ---- Load-time repack (docs/memory-tiers-and-fit.md section 7) -------------------------
// A fast-tier segment that holds a convertible tensor gets a PRIVATE twin: the mapped
// window is blit-copied into it once, then each convertible tensor is rewritten by the
// repack kernel from the mapping into the twin at the same local offset. The twin then
// replaces the mapping buffer in the segment table and the residency set, so every
// dispatch keeps binding the segment at the same local offsets and reads tile-major bytes;
// the mapped pages of that window are no longer referenced and the OS may drop them.
// Only Q8_0 -> Q8_0_TM has readers today; other kinds are left as they are (applied = 0).
struct WXformWire { uint64_t off, bytes; uint32_t from_type, to_type, n_in, n_out; };
static_assert(sizeof(WXformWire) == 32, "transform wire drift");
enum : uint32_t { GT_Q8_0 = 8, GT_Q8_0_TM = 1000 };

static WSeg * wseg_mut_at(uint64_t off) {
    size_t lo = 0, hi = g_wsegs.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (g_wsegs[mid].off + g_wsegs[mid].bytes <= off) { lo = mid + 1; } else { hi = mid; }
    }
    if (lo < g_wsegs.size() && g_wsegs[lo].off <= off) { return &g_wsegs[lo]; }
    return nullptr;
}

// A segment's twin: a Metal-allocated buffer of the segment's window length, filled from
// the mapped buffer by `blit`. SHARED storage, the mode every other weight buffer uses.
// Measured 2026-09-04 (docs/evidence/bracket/2026-09-04-memory-tiers-step-6-load-time-repack.md):
// shared, private, anonymous-mmap and the mapped file all stream at the same ~136 GB/s, so
// the storage mode is not a speed choice; shared reads back without a blit.
static id<MTLBuffer> make_twin(const WSeg * s, id<MTLBlitCommandEncoder> blit) {
    id<MTLBuffer> t = [g.device newBufferWithLength:(NSUInteger)s->len
                                            options:MTLResourceStorageModeShared];
    if (t == nil) {
        NSLog(@"imparo metal: no Metal-allocated twin of %llu bytes", (unsigned long long)s->len);
        return nil;
    }
    [blit copyFromBuffer:s->buf sourceOffset:0 toBuffer:t destinationOffset:0 size:(NSUInteger)s->len];
    return t;
}

// The twins take their segments' places at the same windows, in the residency set instead
// of the mapped buffers. A mapped buffer is a no-copy wrapper: dropping it releases the
// wrapper, and the file pages fall out of the page cache on their own. Returns the bytes.
static uint64_t swap_twins(const std::vector<id<MTLBuffer>> & twin) {
    uint64_t swapped = 0;
    for (size_t si = 0; si < g_wsegs.size(); ++si) {
        if (twin[si] == nil) { continue; }
        rset_remove(g_wsegs[si].buf);
        g_wsegs[si].buf = twin[si];
        rset_add(twin[si]);
        swapped += g_wsegs[si].len;
    }
    return swapped;
}

extern "C" int32_t imparo_metal_transform_weights(const WXformWire * jobs, uint32_t n,
                                                  uint8_t * applied) {
    if (g.device == nil || g_wsegs.empty()) { return 1; }
    for (uint32_t i = 0; i < n; ++i) { applied[i] = 0; }
    if (g.p_repack_q8_tm == nil) { return 0; }
    // Twins per segment, created on first use; the segment index keys them.
    std::vector<id<MTLBuffer>> twin(g_wsegs.size(), nil);
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    uint32_t planned = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const WXformWire & j = jobs[i];
        if (j.from_type != GT_Q8_0 || j.to_type != GT_Q8_0_TM) { continue; }
        WSeg * s = wseg_mut_at(j.off);
        if (s == nullptr || s->tier != WT_FAST) { continue; }
        if (j.off + j.bytes > s->off + s->bytes) { continue; }   // a tensor never straddles
        const size_t si = (size_t)(s - g_wsegs.data());
        if (twin[si] == nil) {
            twin[si] = make_twin(s, blit);
            if (twin[si] == nil) { [blit endEncoding]; return 2; }
        }
        applied[i] = 1;
        planned += 1;
    }
    [blit endEncoding];
    if (planned == 0) { [cb commit]; [cb waitUntilCompleted]; return 0; }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g.p_repack_q8_tm];
    for (uint32_t i = 0; i < n; ++i) {
        if (!applied[i]) { continue; }
        const WXformWire & j = jobs[i];
        WSeg * s = wseg_mut_at(j.off);
        const size_t si = (size_t)(s - g_wsegs.data());
        const NSUInteger local = (NSUInteger)(j.off - s->base);
        [enc setBuffer:s->buf offset:local atIndex:0];
        [enc setBuffer:twin[si] offset:local atIndex:1];
        [enc setBytes:&j.n_in length:4 atIndex:2];
        [enc setBytes:&j.n_out length:4 atIndex:3];
        const uint32_t blocks = j.n_in / 32u;
        [enc dispatchThreads:MTLSizeMake(blocks, j.n_out, 1)
       threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(blocks, 64u), 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] != MTLCommandBufferStatusCompleted) {
        NSLog(@"imparo metal: repack command buffer failed: %@", [cb error]);
        return 3;
    }
    const uint64_t swapped = swap_twins(twin);
    NSLog(@"imparo metal: repack at load: %u tensors -> Q8_0_TM in %zu Metal-allocated segments (%llu MiB)",
          planned, (size_t)std::count_if(twin.begin(), twin.end(), [](id<MTLBuffer> b) { return b != nil; }),
          (unsigned long long)(swapped >> 20));
    return 0;
}


// Weight bytes as the GPU serves them (verification): a shared segment is read directly, a
// private one through a blit into a scratch buffer.
extern "C" int32_t imparo_metal_read_weight_bytes(uint64_t off, uint64_t bytes, uint8_t * out) {
    WSeg * s = wseg_mut_at(off);
    if (s == nullptr || off + bytes > s->off + s->bytes) { return 1; }
    const NSUInteger local = (NSUInteger)(off - s->base);
    if ([s->buf storageMode] != MTLStorageModePrivate) {
        std::memcpy(out, (const uint8_t *)[s->buf contents] + local, (size_t)bytes);
        return 0;
    }
    id<MTLBuffer> tmp = [g.device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
    if (tmp == nil) { return 2; }
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:s->buf sourceOffset:local toBuffer:tmp destinationOffset:0 size:(NSUInteger)bytes];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    std::memcpy(out, [tmp contents], (size_t)bytes);
    return 0;
}

// The gather-combine over staged rows: row t is token t, so the kernel reads `rows`
// directly and never sees the table or the token ids.
extern "C" void imparo_metal_ple_gather_combine_staged(uint32_t proj, uint32_t rows,
                                                       uint32_t width, float emb_scale,
                                                       float comb_scale, uint32_t n_tok) {
    if (g_skip_cat == PC_PLE) { return; }
    haz(hb(proj) | hb(rows), hb(proj));
    [g.enc setComputePipelineState:g.p_ple_gather];
    [g.enc setBuffer:g.bufs[proj] offset:g.buf_off[proj] atIndex:0];
    [g.enc setBuffer:g.bufs[rows] offset:g.buf_off[rows] atIndex:1];
    [g.enc setBuffer:g.bufs[rows] offset:g.buf_off[rows] atIndex:2];   // unread when staged
    const uint64_t w_offset = 0;
    const uint32_t staged = 1;
    [g.enc setBytes:&w_offset length:8 atIndex:3];
    [g.enc setBytes:&width length:4 atIndex:4];
    [g.enc setBytes:&emb_scale length:4 atIndex:5];
    [g.enc setBytes:&comb_scale length:4 atIndex:6];
    [g.enc setBytes:&n_tok length:4 atIndex:7];
    [g.enc setBytes:&staged length:4 atIndex:8];
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
