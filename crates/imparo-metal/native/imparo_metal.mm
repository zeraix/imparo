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
#include <sys/stat.h>
#include <libproc.h>
#include <mach/mach.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include <execinfo.h>

// The KV tier's own residency set (defined with the KV storage below): committed at a region
// begin, held at a region end, released with the weights' set when idle.
static void kvset_sync_locked(void);
static void kvset_hold(void);
static void kvset_release_hold(void);

namespace {

extern "C" void * objc_autoreleasePoolPush(void);
extern "C" void   objc_autoreleasePoolPop(void *);

// The MSL lives in native/imparo.metal (real Metal syntax highlighting, its
// own file history); build.rs wraps it back into this raw-string constant at
// compile time. Runtime source compilation is unchanged -- no metallib, no
// extra toolchain dependency.
#include "imparo_msl.inc"
#include "mega_slots.h"   // the mega entry's slot enums, emitted by build.rs from mega_slots.rs (task #158)
#include "mega_grid.h"    // what each program's grid-striding phases run over, emitted from mega_program.rs

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
    g_attn_stream_hq = v >= 3u ? 3u : (v >= 2u ? 2u : 1u);
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
// Buffers whose FLOAT contents are stale, one bit per id: a GEMM epilogue in mirror mode writes
// only the half mirror of its output (nothing after it was meant to read the floats). Set there,
// cleared by any later write to the buffer (haz) or from the host. A reader that takes floats
// -- the rows matmul -- reads the mirror instead while the bit is set.
uint64_t g_float_stale    = 0;

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
// IMPARO_HAZ_SKIP=N: drop every Nth hazard barrier (wrong answers; timing only).
uint32_t g_haz_skip       = 0;
// Name the conflicting buffers in the barrier instead of taking the whole buffer scope.
// Correct either way -- the masks here are exact and the step hash is identical -- so this is
// a SEAT, not a fix, and it is OFF until it has a measurement. IMPARO_HAZ_NAMED=1 selects it.
// The single-run smoke test read it SLOWER (11.940 vs 11.583), which is why it does not
// default on: a53ca476 shipped it on by accident, through a `git add -A` that swept an
// unmeasured kernel change into a commit about something else.
bool     g_haz_named      = false;
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
// WHERE THE GEMV ENDS, for every weight family: a dispatch of ONE row takes the decode
// GEMV; two rows or more take the GEMM family (Q4_0: nb8 then the wide tile; Q8_0: the
// staged GEMM; block quants: the register-tiled GEMM). Fixed at 1 and not tuned, because
// of the KV identity contract (docs/kv-identity-grid.md): two chunkings of one prompt must
// write the same K/V bytes, and the GEMV's k-order is not the GEMM's -- a row summed by
// the GEMV (simd_sum over sub-blocks) is not bit-equal to the same row summed by the MMA
// k-loop. So the kernel a row takes may depend on whether it is alone in its dispatch (a
// decode step) but never on how many other rows share its chunk. Measured on Qwen3.8-27B
// at n=2000: with the Q8 crossing at 32, a batch of 64 (16-row tail) differed from a
// batch of 512 (464-row tail) from layer 0's alpha/beta projections on; at 1 the two are
// byte-equal. The Q4_0 crossing has the same property (E4B at IMPARO_GEMV_MAX=32 differs).
// Evidence: docs/evidence/bracket/2026-09-11-gemv-crossing-vs-chunk-identity.md.
//
// What the rule costs is a few narrow prefill tails. The block-quant GEMV reads the whole
// weight stream once PER TOKEN (its weight loop sits inside its token loop) and the GEMM
// reads it once per padded 64-token tile, so below one tile the GEMV is linear and the
// GEMM flat. Qwen3.8-27B UD-Q4_K_S, one chunk, ms (2026-09-09):
//     n_tok      4      5      6  |     7      8     10
//     gemv   407.2  509.0  614.0  | 715.6  820.5 1024.2     linear, 102 ms per token
//     gemm   689.7  691.6  693.2  | 692.4  691.2  694.4     flat, one padded tile
// A prompt whose last chunk is 2..6 rows pays up to 485 ms once; a chunk policy that
// balances the tail instead of leaving it ragged removes such tails. The Q8 token tile and
// the multi-token Q4 GEMV stay compiled and are reached only through the diagnostic.
//
// IMPARO_GEMV_MAX=N re-runs the A/B: every family's GEMV up to N rows. Nothing persists it.
uint32_t g_gemv_max_tok   = 1;
// DECODE ROWS. While set, a projection of 2..8 rows is a set of INDEPENDENT decode rows -- one
// co-batched step over several conversations. The model sets a route around the step
// (imparo_metal_set_decode_rows):
//
//   1 EXACT  every row takes the decode GEMV's arithmetic (the multi-row GEMV, function
//            constant 28), so its bits equal what it gets decoding alone on the dispatch path.
//            The gate that proves the per-row plumbing compares exactly that.
//   2 FAST   the rows take the GEMV up to the family's measured row count (q8_rows_gemv_max,
//            q4_rows_gemv_max), then on tile-major Q8 the rows matmul up to q8_tm_rows_mma_max,
//            and the GEMM above that: at each row count the kernel the tuner timed faster.
//
// The one-row crossing rule above is about chunks of ONE prompt, whose K/V bytes the pool
// shares by identity. A decode row's K/V bytes already depend on its route -- the mega route's
// differ from the dispatch path's -- and a restored reply is held to the logit rule, not to
// bytes, so the fast route breaks no contract a lone decode keeps. IMPARO_DECODE_ROWS=1 forces
// the exact route for every dispatch, a measurement switch for imparo-forward --dbatch.
uint32_t g_decode_rows    = 0;
// Rows up to which the fast route keeps the decode GEMV, per weight family. 8, the multi-row
// GEMV's widest, keeps it at every row count: the exact route's choice, and what a config with
// no measured value runs.
uint32_t g_q8_rows_gemv_max = 8;
uint32_t g_q4_rows_gemv_max = 8;
// The rows matmul's widest step: RM_MAX_FRAGS token columns of 8 in imparo.metal.
constexpr uint32_t RM_MAX_FRAGS_HOST = 3;
constexpr uint32_t RM_MAX_ROWS_HOST = 8u * RM_MAX_FRAGS_HOST;
// Rows up to which the fast route takes the rows matmul (imparo_q8_tm_rows_mma) once the GEMV's
// rows are passed, tile-major Q8 only; above it the GEMM. 0 keeps the GEMM there.
uint32_t g_q8_tm_rows_mma_max = RM_MAX_ROWS_HOST;
// The same for Q4_0, on the same kernel reading each row's blocks in place: rows past
// q4_rows_gemv_max and up to this many take it; above it the GEMM. 0 keeps the GEMM there.
uint32_t g_q4_rows_mma_max = RM_MAX_ROWS_HOST;
// Rows-matmul dispatches encoded by this process: a test or harness proves the route ran with
// this, not with the numbers the route produced.
uint64_t g_rows_mma_dispatches = 0;
// The block formats' decode-rows GEMV (imparo_blk_gemv_rows): rows up to which the fast route
// takes it; above it the matrix-unit rows kernel, then the GEMM. Its forms hold up to 8 tokens.
// The GEMV's cost grows about 23 ms with every row and the matrix unit's hardly at all, so the
// crossing is low; it is 1 for most formats -- the matrix unit from 2 rows.
//
// THE CROSSING IS PER FORMAT. The matrix unit's products cost the same per 8x8 weight tile
// whatever number of token rows is live, so whether they are free depends on how many weight
// bytes that tile carries. A format at the bandwidth wall hides them; a format whose unpack
// leaves it short pays them on top of an already ALU-limited kernel, and there the GEMV -- which
// computes only the live columns -- is cheaper. Qwen3.8-27B (M3 Pro) at 2 rows, the shipping
// kernel against itself with the products dropped, then the two kernels against each other
// (docs/evidence/cobatch/2026-09-20-27b-rows-no-tile.md section 24, two rounds agreeing to
// 0.04 ms):
//
//   format        GB/s with products -> without    GEMV against the matrix unit at 2 rows
//   Q3_K_TM        80.9 -> 108.7  (27 short)       -0.91, -0.89   <- the GEMV wins
//   IQ3_S_TM       81.3 -> 104.8  (31 short)       -0.98, -0.94
//   IQ3_XXS_TM     71.8 ->  95.0  (41 short)       -0.61, -0.60
//   IQ2_XS / IQ2_S 56-64 -> 76-91                  -0.12, -0.10
//   Q4_K_TM       128.1 -> 136.3  (at the wall)    +0.41, +0.51   <- the matrix unit wins
//   IQ4_XS_TM     122.0 -> 137.0  (at the wall)    +7.56, +7.76
//   Q6_K_TM       142.6 -> 142.4  (products free)  +0.15, +0.14
//
// At 3 rows no format wants the GEMV (Q3_K +0.41, IQ3_S +0.70, IQ4_XS_TM +17.4), so 2 is the only
// crossing that differs. Both layouts of a format pick the same winner (IQ4_XS / IQ4_XS_TM, Q4_K /
// Q4_K_TM, Q5_K / Q5_K_TM, Q6_K / Q6_K_TM), so the key is the ggml source type `wfmt_for` returns.
constexpr uint32_t BLK_ROWS_MAX_HOST = 8u;
// The crossing for each block format, by the ggml source type `wfmt_for` returns. 2 for the
// formats measured short of the wall -- Q3_K 11, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22 -- and
// 1 for the rest, the matrix unit from 2 rows. Adding a format here needs the two measurements
// above for it, not a guess from its bit width.
//
// THIS TABLE IS THE FALLBACK, not the answer. It is right for the formats whose two kernels are
// far apart and wrong for the ones where they nearly tie -- see `blk_rows_gemv_max_for` below,
// which prefers a MEASURED seat per tensor when the tune file carries one.
//
// `imparo_metal_set_blk_rows_gemv_max` sets EVERY format: tests/blk_decode_rows.rs uses it to put
// every format on one kernel, and imparo-metalbench's IMPARO_BENCH_BLK_ROWS_GEMV_MAX to time the
// two against each other.
constexpr uint32_t BLK_WFMT_COUNT = 32u;
static uint32_t blk_rows_gemv_max_compiled(uint32_t wf) {
    switch (wf) {
        case 11u: case 17u: case 18u: case 21u: case 22u: return 2u;
        default: return 1u;
    }
}
static uint32_t g_blk_rows_gemv_max_fmt[BLK_WFMT_COUNT];
static bool g_blk_rows_gemv_max_set = false;
static uint32_t blk_rows_gemv_max_fmt(uint32_t wf) {
    if (!g_blk_rows_gemv_max_set) { return blk_rows_gemv_max_compiled(wf); }
    return wf < BLK_WFMT_COUNT ? g_blk_rows_gemv_max_fmt[wf] : 1u;
}

// A SEAT PER TENSOR: (wfmt, n_in, n_out) -> the rows up to which the scalar decode-rows GEMV
// beats the matrix unit. Absent, the format's value above answers.
//
// WHY PER TENSOR. On Qwen3.8-27B UD-Q4_K_S at 2 rows the GEMV beats the matrix unit on six of
// Q4_K_TM's seven shapes (attn_qkv 5120->10240, 222.8 against 230.8 us) and loses on the seventh
// (ffn_down 17408->5120, 458.4 against 388.8). Q4_K TILE-MAJOR wants the GEMV where Q4_K
// ROW-MAJOR wants it not, and 17408->5120 wants the matrix unit for Q4_K and the GEMV for Q3_K.
// Format, layout and shape interact and none of them predicts the winner alone.
//
// WHY THIS CAN BE A SEAT WHEN THE PER-FORMAT ONE COULD NOT. The objection to seating the crossing
// was that a sweep reads ONE step total, and that total is 35% a single tensor group, so it picks
// the matrix unit for everything and flattens the table. This seat is not swept: the tuner TIMES
// BOTH KERNELS ON EACH TRIPLE (imparo-metalbench's GEMV probe already reports per tensor) and
// records the winner, so there is no aggregate to hide in.
//
// The lookup runs on every block matmul dispatch -- about 60 a layer, 64 layers, so ~3800 times a
// step. A linear scan of 67 triples there would cost more than the table saves, so the entries go
// in an open-addressed table probed by hash.
struct BlkRowsSeat {
    uint32_t wf, n_in, n_out, gemv_max;
    bool used;
};
constexpr uint32_t BLK_ROWS_SEAT_SLOTS = 512u;          // a power of two; load factor <= 0.5
constexpr uint32_t BLK_ROWS_SEAT_MAX = BLK_ROWS_SEAT_SLOTS / 2u;
static BlkRowsSeat g_blk_rows_seats[BLK_ROWS_SEAT_SLOTS];
static uint32_t g_blk_rows_seat_count = 0u;

static inline uint32_t blk_rows_seat_slot(uint32_t wf, uint32_t n_in, uint32_t n_out) {
    uint64_t h = (uint64_t)wf * 0x9E3779B97F4A7C15ull;
    h ^= (uint64_t)n_in * 0xC2B2AE3D27D4EB4Full;
    h ^= (uint64_t)n_out * 0x165667B19E3779F9ull;
    h ^= h >> 29;
    return (uint32_t)h & (BLK_ROWS_SEAT_SLOTS - 1u);
}

static uint32_t blk_rows_gemv_max_for(uint32_t wf, uint32_t n_in, uint32_t n_out) {
    // THE EXPLICIT OVERRIDE OUTRANKS A SEAT, and it has to. Its whole purpose is to put EVERY
    // format on one kernel so a test or the bench can time that kernel alone; a seat that
    // outranked it would leave some tensors on the other one and label the mixture as one. That
    // is not hypothetical: with 37 seats loaded, IMPARO_BENCH_BLK_ROWS_GEMV_MAX=2 ran 28 tensors
    // on the GEMV and 38 on the matrix unit while claiming to force the GEMV.
    if (g_blk_rows_gemv_max_set) { return blk_rows_gemv_max_fmt(wf); }
    if (g_blk_rows_seat_count != 0u) {
        uint32_t s = blk_rows_seat_slot(wf, n_in, n_out);
        for (uint32_t probe = 0u; probe < BLK_ROWS_SEAT_SLOTS; ++probe) {
            const BlkRowsSeat & e = g_blk_rows_seats[s];
            if (!e.used) { break; }                      // an empty slot ends the chain
            if (e.wf == wf && e.n_in == n_in && e.n_out == n_out) { return e.gemv_max; }
            s = (s + 1u) & (BLK_ROWS_SEAT_SLOTS - 1u);
        }
    }
    return blk_rows_gemv_max_fmt(wf);
}
// Its dispatches, for the same proof.
uint64_t g_blk_rows_dispatches = 0;
// The block formats' rows on the simdgroup matrix unit (imparo_blk_rows_mma): rows past the
// format's own GEMV crossing and up to this many take it -- one 8-token column, so 8 at most;
// above it the GEMM. 0 keeps the GEMM there. Qwen3.8-27B, projections a step: 128.4 / 130.0 ms at
// 4 / 8 rows against the GEMM's 199.5 / 200.4.
constexpr uint32_t BLK_MMA_MAX_HOST = 8u;
uint32_t g_blk_rows_mma_max = BLK_MMA_MAX_HOST;
static uint32_t decode_rows_route(void) {
    static int forced = -1;
    if (forced < 0) { const char * e = getenv("IMPARO_DECODE_ROWS"); forced = (e != nullptr && e[0] == '1') ? 1 : 0; }
    return forced == 1 ? 1u : g_decode_rows;
}
static bool decode_rows_gemv(uint32_t n_tok, uint32_t fast_max) {
    const uint32_t route = decode_rows_route();
    if (route == 0u || n_tok < 2u || n_tok > 8u) { return false; }
    return route == 1u || n_tok <= fast_max;
}
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
// THE RECURRENT HEAD WIDTHS, injected at library compile like the attention head dims and
// for the same reason: the delta kernel holds one state row per lane in registers, and a
// register array's size must be a constant expression. Zero means the model has no
// recurrent mixer, and then no delta pipeline is built.
// ROWS PER SIMDGROUP in the tile-major decode GEMV. One is the shape the kernel had when
// the lm head was its only dispatch; more give the kernel that many independent
// decode-then-multiply chains and read the x values once for all of them. Measured here,
// not assumed: IMPARO_BLK_GEMV_NR is the A/B arm.
// THE SCALAR ROUTED KERNEL'S SKIP BITS (IMPARO_MOE_SKIP). Read in one place because TWO
// kernels serve this path and only one of them carries the constant -- see the refusal in
// imparo_metal_moe_grouped.
static uint32_t moe_skip_bits() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_MOE_SKIP"); v = e ? atoi(e) : 0; }
    return (uint32_t)(v > 0 ? v : 0);
}

static uint32_t blk_gemv_nr() {
    static uint32_t v = 0u;
    if (v == 0u) {
        const char * e = getenv("IMPARO_BLK_GEMV_NR");
        // TWO, measured on Qwen3.8-27B at 512 keys, three rotated rounds, warm:
        // 1 -> 136.0 / 137.5 / 137.0 ms per token, 2 -> 134.7 / 134.7 / 135.0 (-1.7%,
        // same sign every round). 4 is a wash (136.8) and 8 loses 6% to register
        // pressure. A knob's worth of value; a knob when a second model reaches here.
        const uint32_t want = e ? (uint32_t)atoi(e) : 2u;
        v = (want == 1u || want == 2u || want == 4u || want == 8u) ? want : 2u;
    }
    return v;
}
uint32_t g_delta_kd = 0u;   // key coordinate = Q/K head width
uint32_t g_delta_vd = 0u;   // value coordinate = V head width
// The depths the tree delta kernel keeps a branching row's state for: every node must be
// shallower. Injected into the library as IMPARO_DELTA_TREE_DEPTH and read back by the Rust
// layout check.
static const uint32_t g_delta_tree_depth = 16u;
// Must equal DELTA_SGS * 32 in imparo.metal: the kernel divides the state's rows among
// its simdgroups by that count, so a smaller launch would leave rows unowned.
// Threads per delta-rule threadgroup: DELTA_SGS simdgroups, the same macro the kernel is
// compiled with (IMPARO_DELTA_SGS, read once where the library's macros are set).
static uint32_t g_delta_sgs = 32u;
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
// A verify's row-layout attention (imparo_metal_attention_rows): threadgroups the key split aims
// for (0 = no split) and whether the four query heads of a KV head share a threadgroup. Measured
// on Qwen3-4B Q4_K_M, 8-row verify at 2512 keys, packed heads, verify ms by target: 32 27.6,
// 48 27.5, 64 27.4, 96 27.9, 128 28.0, 256 28.8 (no split 35.9, neither 38.5).
uint32_t g_verify_attn_tgs   = 64u;
uint32_t g_verify_attn_heads = 1u;
// Cleared by a caller that needs the causal forward's arithmetic from a row layout (the tree
// gate's chain check); the split is the only part of the verify grid that changes it.
uint32_t g_verify_split_on   = 1u;
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
// all. The software barriers need grid-wide progress, which Metal does not guarantee.
// maxTotalThreadsPerThreadgroup bounds ONE threadgroup; multiplying it by the core count
// is only a grid-sizing heuristic, not a co-residency proof. The old two-per-core default
// worked on M3 Pro but on M4 Pro 36x16 measured 19.9 tok/s against 60.0 at 20x16
// (adjacent A/B, E4B, 2026-09-08). Default to one threadgroup per detected core.
// Stored tuning values obey that conservative ceiling: the old micro tuner never ran the
// persistent kernel when ranking these knobs. Only explicit IMPARO_MEGA_TGS experiments
// can request a wider grid, still bounded by the historical heuristic and pipeline limits.
// This policy reduces admission pressure; it is NOT an API progress guarantee. Residency
// checks, bounded spins and state rollback remain necessary even at one group per core.
static int mega_level(void) {
    static int lvl = -1;
    if (lvl < 0) { const char * e = getenv("IMPARO_MEGA_FFN"); lvl = (e == nullptr) ? 5 : (int)strtol(e, nullptr, 10); if (lvl < 0) { lvl = 0; } }
    return lvl;
}
// The mega pipelines are built at any level >= 1; an entry needs >= 2 (its min_level). Level 1
// once dispatched the stage-1 FFN block alone; that kernel is gone (task #203: it read the
// layer seat without the layer route's fast-tier check, and nothing tuned reached it).
static bool mega_blocks_wanted(void) { return mega_level() >= 1; }
static uint32_t g_gpu_cores = 0;            // IORegistry gpu-core-count; 0 = unreadable
// THE PIPELINE FAMILIES, and THE GRID IS PER FAMILY. One shape for every architecture would
// mean a new architecture can shrink an existing model's grid: the pipeline limit is a
// MIN over the pipelines it covers, so one heavier kernel anywhere lowers the shape
// everywhere. Each family derives its shape from ITS OWN pipelines'
// max-threads verdict, out of the same requested seat. (The file already applies this rule
// one level down: a deep variant below the plain verdict is switched OFF for that slot
// rather than lowering the grid.)
constexpr uint32_t MEGA_ARCH_GEMMA4 = 0u, MEGA_ARCH_LFM2 = 1u, MEGA_ARCH_QWEN35 = 2u, MEGA_ARCH_LFM2MOE = 3u;
constexpr uint32_t MEGA_ARCH_COUNT = 4u;
static uint32_t g_mega_threads_limit[MEGA_ARCH_COUNT] = { 0u, 0u, 0u, 0u };   // derived at init; 0 = no pipeline for this family
static uint32_t g_mega_nsg_limit[MEGA_ARCH_COUNT] = { 0u, 0u, 0u, 0u };   // single-threadgroup pipeline limit, not the grid's
// THE GRID SEAT IS PER PIPELINE: per family AND per head-dim slot (task #203). E4B's hd-256
// and hd-512 layers are two compiled pipelines with different footprints, and admission is a
// pipeline's register footprint, so one seat for both would hold the 35 windowed layers at
// whatever the 7 deep ones admit. Slot 0 is the smaller head dim (the mega_tgs knob), slot 1
// the larger (mega_tgs_large). One per core by default; a stored tune value or IMPARO_MEGA_TGS
// moves it.
static uint32_t g_mega_tgs[MEGA_ARCH_COUNT][2] = { { 1u, 1u }, { 1u, 1u }, { 1u, 1u }, { 1u, 1u } };
// What the admission probe found a pipeline will hold resident, as a threadgroup count,
// PER WIDTH (task #203): indexed by family, head-dim slot and simdgroups per threadgroup,
// because admission is a register footprint and the footprint is per thread -- a limit taken
// at 16 simdgroups says nothing about 32 (18 x 32 timed out under a limit measured at 16).
// 0 = not measured in this process. MEASURED BY THE TUNER, never by the engine:
// imparo_metal_mega_admission() runs it during discovery at every width the sweep may set,
// and the registry ranks the grid knobs under it. The engine applies the stored values as
// written and never reads this.
constexpr uint32_t MEGA_NSG_MAX = 32u;
static uint32_t g_mega_tgs_probed[MEGA_ARCH_COUNT][2][MEGA_NSG_MAX + 1u] = {};
// The family the tuner asked about last (MEGA_ARCH_COUNT = none): what the registry reads back.
static uint32_t g_mega_probe_family = MEGA_ARCH_COUNT;
static uint32_t g_mega_nsg[MEGA_ARCH_COUNT] = { 16u, 16u, 16u, 16u };
// The seats the tuner asks for -- the grid per slot, the width per family -- applied to every
// family and clamped per pipeline (0 = derive).
static uint32_t g_mega_tgs_req[2] = { 0u, 0u }, g_mega_nsg_req = 0u;
static bool g_mega_tgs_explicit = false;   // env-only wider-grid experiment; a stored knob is not one
static uint32_t mega_core_seat(void) { return g_gpu_cores > 0u ? g_gpu_cores : 1u; }
// READ ONLY: what the probe measured for `arch`, or 0 when it has not run in this process or
// ran at another threadgroup width (the one-per-core floor is then the only known-safe grid).
static uint32_t mega_ceiling(uint32_t arch, uint32_t slot) {
    if (arch >= MEGA_ARCH_COUNT || slot >= 2u) { return 0u; }
    const uint32_t nsg = g_mega_nsg[arch];
    if (nsg == 0u || nsg > MEGA_NSG_MAX) { return 0u; }
    return g_mega_tgs_probed[arch][slot][nsg];
}
// What the last entry of each family DERIVED, so the knob's readback reports what is running
// rather than the compiled seat (#74's requested-vs-running rule). 0 = never derived.
static uint32_t g_mega_derived_nsg[MEGA_ARCH_COUNT] = { 0u, 0u, 0u, 0u };
static uint32_t g_mega_derived_tgs[MEGA_ARCH_COUNT] = { 0u, 0u, 0u, 0u };
// IMPARO_MEGA_NSG_EXACT=1 turns the derivation off and runs the seat verbatim: the A/B arm
// that prices the balance rule, and the way to reproduce a grid from an evidence table.
static bool g_mega_nsg_exact = false;
static uint32_t g_mega_last_tgs = 0u;   // the grid of the last mega dispatch: what a timeout's arrival count is out of
static uint32_t g_tgmem_limit = 0u;     // MTLDevice.maxThreadgroupMemoryLength; 0 = not read yet
// What the kernel declares statically beside the row: partial[32] and tg_err.
constexpr uint32_t MEGA_TG_STATIC_BYTES = 32u * 4u + 4u;
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
//
// Attachment is that guarantee only once the tier has been wired at least once: before the
// first hold the pages are still arriving, and "resident during execution" is what the
// command buffer WAITS for, which is the paging a spinning barrier must not sit behind. So
// the route also waits for the first hold -- one region on the dispatch path, at process
// start, in exchange for never spinning on a page fault.
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
// qwen35 starts on the PER-LAYER form: the per-token program is worth what its admission
// measures (task #167), and nothing has measured it for this entry size yet.
static const bool MEGA_PROGRAM_DEFAULT_Q35 = false;
// lfm2moe's entry is the routed feed-forward alone and its mixer runs on the dispatch path
// between two entries, so a run could never hold more than one entry: the per-layer form.
static const bool MEGA_PROGRAM_DEFAULT_L2M = false;
// By family index (MEGA_ARCH_GEMMA4, MEGA_ARCH_LFM2, MEGA_ARCH_QWEN35, MEGA_ARCH_LFM2MOE): the
// entry dispatch and the tuner's applicability question (imparo_metal_mega_seat_form) read the
// same table.
static const bool MEGA_PROGRAM_DEFAULTS[MEGA_ARCH_COUNT] = { MEGA_PROGRAM_DEFAULT_E4B, MEGA_PROGRAM_DEFAULT_LFM2, MEGA_PROGRAM_DEFAULT_Q35, MEGA_PROGRAM_DEFAULT_L2M };
// THE PROGRAM RING (task #153): entries recorded per layer into a device ring -- 4 region slots
// (the debug slots' rotation, so a pipelined region's entries are not overwritten for four
// regions) x MEGA_PROG_CAP entries, ONE ring for every architecture (task #158 gave them one
// entry type, so one entry size) -- flushed as ONE
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
static uint32_t g_prog_nsg = 0u;   // the run's threadgroup width: its family's, fixed when the run opened
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
static uint32_t mega_deep_tgs(uint32_t arch, uint32_t slot) {
    const uint32_t cores = g_gpu_cores > 0u ? g_gpu_cores : 7u;
    return std::max(1u, std::min(g_mega_tgs[arch][slot], cores));
}
// THE OTHER HALF OF ADMISSION: THREADGROUP MEMORY. The max-threads verdict is the
// compiler's REGISTER answer and says nothing about the row every threadgroup forms for
// itself -- n_embd floats of threadgroup memory, which a core has 32 KB of in total. Two
// threadgroups per core is only reachable while two rows fit:
//
//   n_embd 2048 (E4B, LFM2)   8192 B   ->  4 per core; the measured 32 x 16 grid stands
//   n_embd 5120 (qwen35)     20480 B   ->  1 per core; a 32-threadgroup grid then waits at
//                                          the first barrier for threadgroups that cannot
//                                          start, hits the spin cap, and the failsafe runs
//                                          the step on the dispatch path (measured
//                                          2026-09-08: 157 ms vs 126 ms per token, and a
//                                          third forward's probes in a two-forward run)
//
// So the grid is capped by what the row costs. Every architecture gets this; the two whose
// rows are 8 KB are unaffected, which is the point.
static uint32_t mega_tgs_for_row(uint32_t arch, uint32_t slot, uint32_t tgmem_bytes) {
    const uint32_t cores = g_gpu_cores > 0u ? g_gpu_cores : 7u;
    const uint32_t limit = g_tgmem_limit > 0u ? g_tgmem_limit : 32768u;
    const uint32_t per_core = std::max(1u, tgmem_bytes > 0u ? limit / tgmem_bytes : limit);
    return std::max(1u, std::min(g_mega_tgs[arch][slot], cores * per_core));
}
// SIMDGROUPS PER THREADGROUP, DERIVED FROM WHAT THE PHASES RUN OVER.
// A phase strides tile-major units over the grid's simdgroups, so it runs ceil(items / G)
// waves and the grid barrier that ends it waits for the fullest simdgroup: items that do not
// divide G are paid for as a WHOLE extra wave. qwen35's down projection writes n_embd = 5120
// rows = 640 units; at G = 18 x 16 = 288 that is three waves for 2.22 units of work, and the
// phase cost +33% over the same GEMV on the dispatch path while the gated pair (2176 units,
// 7.56 each) stayed within 2%. Measured 2026-09-08, UD-Q4_K_S, two rounds, ms per token:
//
//   nsg   G     down waves / ideal   worst waste   measured
//    9   162        4 / 3.95            1.04        155.4   balanced and STARVED
//   16   288        3 / 2.22            1.35        142.4
//   18   324        2 / 1.98            1.04        131.4   <- what this derives
//   19   342        2 / 1.87            1.10        137.7
//   20   360        2 / 1.78            1.16        135.7
//   24   432        2 / 1.48            1.35        140.4
//   32   576        2 / 1.11            1.80        153.1
//
// THE TWO HALVES BELONG TO DIFFERENT OWNERS, which is why only one of them is computed here.
// Balance is arithmetic: ceil(items / G) follows from the model's shapes and the grid, it
// changes with every model, and no measurement can produce it -- mega_nsg's tuner ladder is
// 8 / 16 / 32 and the answer for this model is 18. Occupancy is a DEVICE property: how many
// simdgroups a core needs in flight to cover memory latency, one number per device, the same
// for every model. So the search below computes the balance and starts from the SEAT, and the
// seat (knob mega_nsg, IMPARO_MEGA_NSG) is the floor the tuner ranks. Fitting all seven grids
// above, that floor sits at 16 on this M3 Pro: flat within 0.97 +- 0.03 of the balance model
// from 16 up, and 26% worse at 9.
constexpr uint32_t MEGA_TM_UNIT_ROWS_HOST = 8u;   // the shader's TM_UNIT_ROWS (rows per unit)
// The grid rules an entry's OWN phases impose, in one place: the attention fold reads its
// scratch in quarters and needs at least eight simdgroups, and LFM2 keeps a head row's Q8
// units inside one threadgroup. The refusal in the entry and the search below ask this same
// question, so a derived value can never be one the entry would then refuse.
static bool mega_nsg_legal(uint32_t nsg, uint32_t arch, bool attn_on, uint32_t hd) {
    if (!attn_on) { return true; }
    if (nsg % 4u != 0u || nsg < 8u) { return false; }
    if (arch == MEGA_ARCH_LFM2 && nsg % std::max(1u, hd / Q8_TM_UNIT_ROWS_HOST) != 0u) { return false; }
    return true;
}
// THE GRID IS tgs x nsg AND BOTH MATTER, so search both. Only nsg was searched at first, and
// that is enough only while something else pins tgs -- which is exactly what the staged row
// does (its 20 KB admits one threadgroup per core, so tgs IS the core count). Take the staging
// away and tgs jumps to its seat, 32, where the co-residency bound caps nsg at 18 and NO grid
// in reach is balanced: qwen35's down projection wastes 1.60 at every legal nsg. A joint
// search finds 20 x 16 = 320 simdgroups, where down is exact and the gated pair wastes 1.03.
// So a one-variable search would have priced "do not stage" at the grid's expense and blamed
// the memory -- which is what the first device-row A/B did (157.7 against 140.5).
//
// tgs ranges from the core count (fewer leaves cores idle) up to whatever the caller's cap
// allows -- the tgmem row and the seat. With the row staged, lo == hi and this is exactly the
// nsg search. Ties go to the SMALLER grid: barrier cost grows with the number of arrivers.
struct MegaGrid { uint32_t tgs; uint32_t nsg; };   // {0, 0} = nothing declared, the seat stands
static MegaGrid mega_grid_derive(uint32_t arch, const uint32_t * u, uint32_t tgs_lo, uint32_t tgs_hi,
                                 uint32_t floor, uint32_t nsg_pipe, bool attn_on, uint32_t hd) {
    uint32_t rows[MEGA_GRID_ROWS_MAX];
    const uint32_t n = mega_phase_rows(arch, u, rows);
    MegaGrid best = { 0u, 0u };
    if (n == 0u || tgs_lo == 0u || tgs_hi < tgs_lo || floor == 0u) { return best; }
    // THREADGROUPS COME IN WHOLE CORES. A grid that is not a multiple of the core count leaves
    // some cores hosting one threadgroup and some hosting two, and every barrier waits for the
    // doubled ones: at tgs 20 on 18 cores that bound is ceil(20/18)/(20/18) = 1.8, and the
    // device-row form measured 148.7 / 154.9 there against 129.7 / 130.6 at tgs 18 -- the same
    // phases, the same grid size (320 vs 324 simdgroups), a BETTER item balance. E4B looked
    // like a tie at tgs 18 / 32 / 36 only because 32 on 18 cores is 1.78 per core, a bound of
    // 1.125; the term is real, it was mild there. So the candidates are the multiples of the
    // core count, and the item balance is chosen inside each.
    const uint32_t cores = g_gpu_cores > 0u ? g_gpu_cores : 7u;
    double best_waste = 0.0;
    for (uint32_t tgs = tgs_lo; tgs <= tgs_hi; ++tgs) {
        if (tgs % cores != 0u && tgs != tgs_hi) { continue; }
        if (tgs % cores != 0u && tgs_hi >= cores) { continue; }   // a whole-core grid exists; take it
        // Co-residency caps the GRID, not one threadgroup: every threadgroup waits on every
        // other, so tgs x nsg x 32 threads must all be resident (the 2026-09-05 panic).
        const uint32_t nsg_res = g_mega_threads_limit[arch] != 0u
                               ? std::max(1u, g_mega_threads_limit[arch] / std::max(1u, tgs * 32u))
                               : nsg_pipe;
        const uint32_t nsg_cap = std::min(nsg_pipe, nsg_res);
        for (uint32_t nsg = floor; nsg <= nsg_cap; ++nsg) {
            if (!mega_nsg_legal(nsg, arch, attn_on, hd)) { continue; }
            const uint64_t grid = (uint64_t)tgs * nsg;
            double worst = 1.0;
            for (uint32_t i = 0; i < n; ++i) {
                const uint64_t items = rows[i] / MEGA_TM_UNIT_ROWS_HOST;
                if (items == 0u) { continue; }
                const uint64_t waves = (items + grid - 1u) / grid;
                worst = std::max(worst, (double)(waves * grid) / (double)items);
            }
            if (best.nsg == 0u || worst < best_waste - 1e-9) { best.tgs = tgs; best.nsg = nsg; best_waste = worst; }
        }
    }
    return best;
}
// The deep body's geometry for a layer: slices per (KV head, sub-group) so the items fill the
// deep grid, refused (0) when the items or the partials do not fit.
static uint32_t mega_attn_deep_slices(uint32_t arch, uint32_t slot, uint32_t n_heads, uint32_t n_kv, uint32_t hd) {
    const uint32_t hq = mega_attn_hq(hd);
    const uint32_t n_sub = (n_heads / n_kv + hq - 1u) / hq;
    const uint32_t tgs = mega_deep_tgs(arch, slot);
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
// Separate the legal single-group width from the grid policy. The grid's thread budget
// cannot bound one group's width (a 512-thread pipeline still rejects a 1024-thread group).
// Before init the requests are just stored; once pipelines exist all setters use this path.
static bool mega_clamp_arch(uint32_t arch) {
    bool moved = false;
    if (g_mega_threads_limit[arch] == 0u) { return false; }
    const uint32_t nsg_max = std::max(1u, std::min(32u, g_mega_nsg_limit[arch]));
    if (g_mega_nsg[arch] < 1u) { g_mega_nsg[arch] = 1u; moved = true; }
    if (g_mega_nsg[arch] > nsg_max) { g_mega_nsg[arch] = nsg_max; moved = true; }
    const uint32_t tgs_max = std::max(1u, g_mega_threads_limit[arch] / (g_mega_nsg[arch] * 32u));
    // THE GRID IS NOT CAPPED HERE (task #203). One per core is the DEFAULT, a stored tune value
    // is applied as written, IMPARO_MEGA_TGS wins; the only bound is the pipeline's own width
    // budget above. The LIMIT a value must respect -- how many threadgroups this pipeline holds
    // resident -- is the tuner's business: imparo_metal_mega_admission() measures it during
    // discovery and the tuner only writes a value under it (docs/tuner-design.md: derive the
    // limit, tune the value). A value above it that reaches the engine anyway (a kernel that
    // grew, an env experiment) costs one spin cap per region and the failsafe's re-run on the
    // dispatch path -- the M4 Pro run at 36 x 16 (handoff/mac-m4-mega-admission.md) -- never a hang.
    for (uint32_t s = 0; s < 2u; ++s) {
        if (g_mega_tgs[arch][s] < 1u) { g_mega_tgs[arch][s] = 1u; moved = true; }
        if (g_mega_tgs[arch][s] > tgs_max) { g_mega_tgs[arch][s] = tgs_max; moved = true; }
    }
    return moved;
}
// Applies the requested seat to every family; each one clamps to its own limit.
static bool mega_clamp(void) {
    bool moved = false;
    for (uint32_t a = 0; a < MEGA_ARCH_COUNT; ++a) {
        for (uint32_t s = 0; s < 2u; ++s) { if (g_mega_tgs_req[s] != 0u) { g_mega_tgs[a][s] = g_mega_tgs_req[s]; } }
        if (g_mega_nsg_req != 0u) { g_mega_nsg[a] = g_mega_nsg_req; }
        moved |= mega_clamp_arch(a);
    }
    return moved;
}
// The knob's readback is the SMALLEST running value over the families with pipelines: if a
// family had to clamp, the tuner sees it (#74's requested-vs-running rule) while every other
// family still runs the seat.
static uint32_t mega_shape_min(const uint32_t * v) {
    uint32_t m = 0u;
    for (uint32_t a = 0; a < MEGA_ARCH_COUNT; ++a) {
        if (g_mega_threads_limit[a] == 0u) { continue; }
        m = (m == 0u) ? v[a] : std::min(m, v[a]);
    }
    return m != 0u ? m : v[MEGA_ARCH_GEMMA4];
}
// The grid seat of one slot, min over the families with pipelines (the readback rule above).
static uint32_t mega_seat_min(uint32_t slot) {
    uint32_t v[MEGA_ARCH_COUNT];
    for (uint32_t a = 0; a < MEGA_ARCH_COUNT; ++a) { v[a] = g_mega_tgs[a][slot]; }
    return mega_shape_min(v);
}
static void mega_set_tgs_slot(uint32_t slot, uint32_t v, const char * knob) {
    g_mega_tgs_req[slot] = v;
    if (mega_clamp()) { NSLog(@"imparo metal: %s %u limited by the pipeline's width budget; running %u x %u", knob, v, mega_seat_min(slot), mega_shape_min(g_mega_nsg)); }
}
extern "C" void imparo_metal_set_mega_tgs(uint32_t v) { mega_set_tgs_slot(0u, v, "mega_tgs"); }
extern "C" void imparo_metal_set_mega_tgs_large(uint32_t v) { mega_set_tgs_slot(1u, v, "mega_tgs_large"); }
extern "C" void imparo_metal_set_mega_nsg(uint32_t v) { g_mega_nsg_req = v; if (mega_clamp()) { NSLog(@"imparo metal: mega_nsg %u limited by the pipeline's width budget; running %u/%u x %u", v, mega_seat_min(0u), mega_seat_min(1u), mega_shape_min(g_mega_nsg)); } }
extern "C" uint32_t imparo_metal_mega_tgs_current(void) { return mega_seat_min(0u); }
extern "C" uint32_t imparo_metal_mega_tgs_large_current(void) { return mega_seat_min(1u); }
// The seat is a floor, so the running value is what the last entry derived from it; reporting
// the floor as if it were the grid would be the lie #74 forbids.
extern "C" uint32_t imparo_metal_mega_nsg_current(void) {
    if (!g_mega_nsg_exact) {
        uint32_t m = 0u;
        for (uint32_t a = 0; a < MEGA_ARCH_COUNT; ++a) {
            if (g_mega_derived_nsg[a] == 0u) { continue; }
            m = (m == 0u) ? g_mega_derived_nsg[a] : std::min(m, g_mega_derived_nsg[a]);
        }
        if (m != 0u) { return m; }
    }
    return mega_shape_min(g_mega_nsg);
}
extern "C" uint32_t imparo_metal_mega_threads_limit(void) { return mega_shape_min(g_mega_threads_limit); }
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
constexpr uint32_t RT_CANDIDATES = 7;
// THE TILE a register-tiled pipeline is compiled for, as one index: a wide shape (its
// RT_SHAPES row) or one of the two narrow 64x8 tiles. The per-format pipeline cache is
// keyed by it, so a shape change (the tuner's rt_shape sweep builds every shape in one
// process) reaches a pipeline compiled for THAT shape, never the first one built.
enum { RT_TILE_NB8 = RT_CANDIDATES, RT_TILE_NB8B = RT_CANDIDATES + 1, RT_TILES = RT_CANDIDATES + 2 };
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
    // THREE were built and MEASURED, then removed: the tuner should not spend sweeps on
    // known losses. rows x toks is the output tile and threads x accumulators is the
    // register budget, which the working shapes all hold near 2048 (128x16, 256x8, 512x4)
    // -- so the tile cannot exceed 4096.
    //
    //   {2,4,2,8}   64 x 128, 512 thr, 8 acc   12503 ms   tile 8192, over budget:
    //                                                     dequant -799 ms, multiply +1543
    //   {4,1,4,4}   32 x 128, 512 thr, 4 acc   16118 ms   in budget, but half the rows
    //                                                     doubles activation re-reads
    //   {8,2,2,2}   32 x 128, 128 thr, 16 acc   >= 600x   REMOVED 2026-09-09, and it is
    //                                                     the only one here INSIDE every
    //                                                     budget above: 16 accumulators,
    //                                                     128 threads, register product
    //                                                     2048, tile 4096. It still runs
    //                                                     two to three orders of magnitude
    //                                                     slow at the engine's widths -- a
    //                                                     128-token prefill that takes
    //                                                     ~1.3 s at shape 1 had not
    //                                                     finished after 15 minutes in ONE
    //                                                     command buffer, long enough to
    //                                                     starve the display server. The
    //                                                     mechanism is NOT understood, and
    //                                                     that is why it is out: a shape
    //                                                     the budgets above call legal and
    //                                                     the machine calls unusable means
    //                                                     the budgets are missing a term.
    //                                                     Task #111 measured the same order
    //                                                     ("~1000x") for this index in the
    //                                                     rt_gemm<Q8> instantiation, so it
    //                                                     is the SHAPE, not one kernel's
    //                                                     use of it.
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
//     g_gemv_max_tok, one row for both families: the route boundary (not a knob)
constexpr uint32_t ST_GEMM_CANDIDATES = 13;
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
    {64,  8,  4, 32},
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
// THE NARROW SEAT, AND WHY IT IS A THIRD ONE AND NOT A THIRD MEMBER OF THE PAIR.
//
// A dispatch's cost has two terms, in `walks = ceil(n_tok / TOKENS)` and
// `padded = walks * TOKENS`:
//
//   weight traffic  =  walks * (the whole projection)     set by WALKS
//   multiply work   =  padded * (K * N per row)           set by PADDED ROWS
//
// Which one is scarce depends on the token count, and the two regimes want opposite
// tiles. Measured on LFM2 (Q8, M3 Pro):
//
//   n_tok=8..32, ONE walk either way -- the weight read is fixed and the kernel waits on
//     it, so what is paid for is padded rows: 28.81 / 31.27 / 44.6 ms per verify step at
//     8 / 16 / 32 padded rows. Narrowest tile that holds the batch wins.
//   n_tok=464, 29 walks against 15 -- weight bytes dominate and padding is at most TOKENS
//     of 464: 9422.9 vs 8676.3 ms of prefill. Widest tile wins.
//
// The PAIR's rule (`q8_geometry`) compares padded counts only, which is the first regime's
// answer. It is right for the 32/64 pair because at the widths where those two compete
// (448..512) padding and walks move together; it is wrong for a 16- or 8-token tile, which
// is why putting one in `g_st_gemm_shape` cost prefill 8.6%. So the narrow tile gets its
// own seat with a token bound, and the pair keeps its measured values untouched.
//
// ONE KNOB, NOT TWO. Which tile the seat should use is not a free choice: below the bound
// every candidate walks the weights once, so the answer is always the NARROWEST tile that
// holds this dispatch's rows, and the engine derives that per dispatch. A separate shape
// knob was built first and its sweep came back inert -- 763.2 to 766.4 us across all
// thirteen candidates against a 764.6 control -- because the shape does nothing until the
// bound is non-zero and the bound is swept after it. Circular, and one knob removes it.
//
// max = 0 is the seat OFF, which is the engine exactly as it was: a stored config from an
// earlier space carries no line for this key and therefore keeps that.
uint32_t g_st_gemm_narrow_max    = 0;

static int32_t g_st_gemm_shape_pin = -1;   // IMPARO_ST_GEMM_SHAPE; -1 = nothing pinned

// WHICH TOKEN TILE A DISPATCH TAKES. Two regimes and one bound between them, with no
// tile index stored anywhere:
//
//   n <= narrow_max   the NARROWEST tile that holds n_tok in one token group
//   n >  narrow_max   the SEAT
//
// The lower regime follows from the two-term cost. A dispatch of n rows through a tile of
// width T
//
//   WALKS        ceil(n/T)      the whole projection re-read once per walk
//   PADDED ROWS  ceil(n/T)*T    K x N of multiply per row, live or not
//
// A verify batch is ONE walk whatever the tile, so only padded rows are left and the
// narrowest tile wins: measured on LFM2, 64x8 against the seat's 64x32 is -19.5% at 8 rows,
// 64x16 against it -9.9% at 10 and -12.4% at 16, and 32 rows (which pad to 32 either way)
// tie. A prefill chunk walks many times, so walks decide instead, and going narrower there
// is what the bound prevents: at 464 rows a 16-token tile walks 29 times against 32's 15,
// and prefill reads 9422.9 against 8676.3 ms.
//
// WHY THE UPPER REGIME IS THE SEAT AND NOT THE WIDEST TILE THAT PADS NO WORSE. That rule
// was built and measured, because it would have retired the second tuned tile outright, and
// it LOSES: one binary, four prefill samples interleaved with the arms reversed between
// rounds, 8444 tokens at the 512 chunk --
//
//   derived widest-no-worse   8334.4  8334.6
//   the seat (64x32)          8287.4  8295.8      +0.5% for the wider tile
//
// It contradicts a +3.1% / +3.0% that an earlier comment claimed for a 64x64 second tile at
// 5963 / 17123 tokens by env pin; that measurement has no surviving log and this one does,
// so the rule follows this one. A second tile above the bound is an open question with a
// measurement against it, not a knob.
static uint32_t q8_tile(uint32_t n_tok) {
    // A PIN IS A PIN: with IMPARO_ST_GEMM_SHAPE set, every dispatch runs that exact tile
    // and the bound does not move it. Deriving under a pin would answer a different shape
    // at a different width, which is the opposite of what an A/B pin -- or the correctness
    // bench's per-shape sweep -- is asking for.
    if (g_st_gemm_shape_pin >= 0) { return g_st_gemm_shape; }
    if (g_st_gemm_narrow_max == 0u || n_tok > g_st_gemm_narrow_max) {
        return g_st_gemm_shape;
    }
    // THE TOKEN WIDTH IS THE ONLY AXIS THIS MOVES. Candidates match the seat in output
    // rows, simdgroup count and K chunk depth: those three are what the tuner ranked, and a
    // shape differing in any of them is a different kernel that nothing here measured. With
    // them fixed the widths are distinct, so there is no tie to break -- an earlier form
    // matched on rows alone and answered shape 6 (64x64 at EIGHT simdgroups) where the seat
    // was 64x32 at four.
    const uint32_t rows = ST_GEMM_SHAPES[g_st_gemm_shape][0];
    const uint32_t nsg  = ST_GEMM_SHAPES[g_st_gemm_shape][2];
    const uint32_t kch  = ST_GEMM_SHAPES[g_st_gemm_shape][3];
    uint32_t best = ST_GEMM_CANDIDATES, best_toks = 0u;
    for (uint32_t i = 0; i < ST_GEMM_CANDIDATES; ++i) {
        const uint32_t t = ST_GEMM_SHAPES[i][1];
        if (ST_GEMM_SHAPES[i][0] != rows || ST_GEMM_SHAPES[i][2] != nsg
            || ST_GEMM_SHAPES[i][3] != kch || t < n_tok) { continue; }
        if (best == ST_GEMM_CANDIDATES || t < best_toks) { best = i; best_toks = t; }
    }
    // A batch wider than every tile in the family has no candidate and keeps the seat.
    return best == ST_GEMM_CANDIDATES ? g_st_gemm_shape : best;
}

// WHICH SHAPES THE RULE CAN NAME. Asked once per candidate at init, because init builds
// only the shapes it is told to and a shape the rule names but init did not build is a
// REFUSED projection, not a slow one: the dispatch is skipped, the previous contents of
// the output stand, and the run continues with wrong numbers. That cost a whole A/B --
// 100079 "nil Q8 pipeline" lines in one verify run, every FFN projection skipped. The
// tuner never saw it because it builds every candidate (`g_q8_all`).
//
// Ask the rule rather than restate it. Above the bound it answers the seat, so one pass
// over the widths under the bound is every other answer there is.
static bool q8_tile_reaches(uint32_t shape) {
    if (shape == g_st_gemm_shape) { return true; }
    for (uint32_t n = 1u; n <= g_st_gemm_narrow_max; ++n) {
        if (q8_tile(n) == shape) { return true; }
    }
    return false;
}
uint32_t g_q8_full_tiles         = 1;
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
// ON with the pad skip OFF (below): the two together leave the generic kernel no runtime bound
// in its staging loops, so it compiles like the full-tile kernel, and dead rows only read
// duplicates whose results are never stored. Either one alone left a runtime condition and
// measured flat, which is how the clamp was first recorded as a negative. Together, at 24 / 32
// rows of LFM2.5's co-batched step on the GEMM: 48.85 -> 44.88 / 49.85 -> 45.34 ms (the full
// kernel reads 45.77 at 32); one-chunk prefill of 455 / 192 tokens -1.8 / -1.7%; live rows'
// bits unchanged. IMPARO_Q8_CLAMP_EDGE=0 builds the branching form.
uint32_t g_q8_clamp_edge       = 1;
// Scheduling fences in the MMA loop, llama.cpp's arrangement. ON by default -- this is
// the shader's own default too, so an unset build compiles them in. Setting it to 0
// through IMPARO_Q8_MMA_FENCE compiles the variant WITHOUT them, which is how the +5.1%
// short / +3.1% deep was measured and how it can be reproduced.
uint32_t g_q8_mma_fence        = 1;
// Skip the activation staging and the multiplies of token tiles past the last live token
// row (shader constant 27). OFF by default, with the edge clamp ON (above): the tile rule
// already picks the narrowest tile that holds the rows, so the skip rarely had whole dead
// simdgroup shares to skip, and its runtime bound cost the unrolled staging loop.
// IMPARO_Q8_PAD_SKIP=1 builds the skipping form.
uint32_t g_q8_pad_skip         = 0;
// The same fence question for the Q4 rt_gemm. Default OFF: that kernel already beats
// upstream, so it does not move without a measurement. Constant 16 in the shader; set it
// through IMPARO_RT_MMA_FENCE=1 to compile the fenced variant.
uint32_t g_rt_mma_fence        = 0;
// The same for prefill attention, the deep leg's remaining growing term. Default off.
// Phase probe for attention: 1 no score MMA, 2 no softmax, 4 no P x V.
uint32_t g_attn_skip           = 0;
// IMPARO_SKIP_MM_NOUT: the one decode-matmul output width to drop (0 = none). See the
// guard beside PC_MATMAT_DECODE's.
uint32_t g_skip_mm_nout        = 0;
// Threads a dim in the attention combine (function constant 35); the dispatch width
// must match what the kernel was compiled for.
uint32_t g_attn_comb_spd       = 1;
// IMPARO_DUP_MM_NOUT: the one decode-matmul output width to encode TWICE (0 = none).
uint32_t g_dup_mm_nout         = 0;
// IMPARO_DUP_ATTN_COMBINE: encode the flash-decoding combine twice (diagnostic).
bool     g_dup_attn_combine    = false;
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
    // The RECURRENT MIXER: LFM2's gated short convolution on 22 of its 30 blocks,
    // Qwen3.8's delta-net convolution and delta rule on 48 of its 64. It is the
    // counterpart of attention, not an elementwise op. It was priced as elementwise
    // because it reaches the GPU through dispatch1(), which used to stamp that
    // category on everything, so those models' dominant block kind never appeared in
    // an attribution.
    // ...and PC_DELTA, the delta RULE, separately from it. The two share a block and
    // nothing else: the convolution is a short elementwise stencil over the projection,
    // the rule is a matrix recurrence whose serial depth is the token count. Priced
    // together they read as one 3.5% class and neither one's shape is visible.
    PC_RECUR, PC_DELTA, PC_MEGA,
    // The drafter's candidates and its confidence head, apart from the elementwise class so a
    // profile shows them on their own.
    PC_TOP_K_CHUNKS, PC_TOP_K_MERGE, PC_LOGISTIC,
    // A one-row projection out of a small rank space: at most 512 inputs and at least 64 outputs
    // per input, the Markov head's W2 (256 to 128000 on LFM2.5's drafter). Every model's output
    // head has more than 512 inputs and stays in matmat_decode.
    PC_MATMAT_RANK,
    // A ROUTED FEED-FORWARD, in two classes. The route (gating, the pick, the counting sort
    // and the combine) is small and fixed; the expert matmuls are the work. Priced together
    // they would read as one class and neither shape would be visible -- the same mistake
    // splitting PC_RECUR from PC_DELTA undid.
    // PC_MOE_ROUTE HAS NO SKIP. Every other class here can be dropped by IMPARO_SKIP_CAT
    // so a skip-and-diff can price it, but the route produces the plan the expert matmuls
    // index, and a skipped route leaves them reading a garbage segment table. Asking to
    // skip it is silently a no-op, so its skip-and-diff figure is noise, not a price --
    // read the route's cost off the profile's own timing instead.
    PC_MOE_ROUTE, PC_MOE_EXPERT,
    // The COMBINE apart from the route. Both run one threadgroup and both are small, which
    // is exactly why they hid each other: priced together they read as one 6% class with no
    // shape, and the combine's is a grid of 1 over 2048 independent outputs.
    PC_MOE_COMBINE,
    // THE OUTPUT HEAD ON ITS OWN. On LFM2.5-8B-A1B it is 215 MB of Q6_K read for ONE row --
    // 45% of every dense byte a decode token moves -- so priced inside matmat_decode it
    // hides whether the dense class is slow or merely large. 16384 outputs separates it
    // with room to spare: the head has 128000, and the next widest decode projection is
    // in_proj's 6144.
    PC_MATMAT_HEAD,
    // A NARROW PROJECTION, at most 64 outputs. On a routed model that is the router's own
    // gate (ffn_gate_inp, 2048 -> 32): 22 of them a token, and a GEMV two rows to a
    // simdgroup gives 32 outputs sixteen simdgroups to cover eighteen cores. Priced inside
    // matmat_decode it is 2% of the class's bytes and invisible.
    PC_MATMAT_NARROW,
    // THE SAME DECODE PROJECTIONS, SPLIT BY SOURCE TYPE. The output head is the one big
    // Q6_K tensor in this file and the only class near the device ceiling; everything else
    // is Q4_K. Priced together the two cannot be told apart, and "is the gap the format?"
    // is the question the rest of the class turns on.
    PC_MATMAT_DEC_Q6,
    // THE ROUTED GATE|UP PAIR, apart from the down projection that shares moe_expert: the
    // two have different K and different epilogues, and one class could not say which of
    // them a routed-GEMM change moved. IMPARO_SKIP_CAT=moe_expert still skips both.
    PC_MOE_PAIR, PC_N
};
static const char * PROF_CAT_NAME[PC_N] = {
    "matmat_prefill", "matmat_decode", "row", "rms_norm", "rope",
    "kv_store", "attention", "elementwise", "mul_strided", "ple_combine",
    "recurrent", "delta_rule", "mega_ffn", "top_k_chunks", "top_k_merge", "logistic", "matmat_rank",
    // IN THE ENUM'S ORDER. The names are indexed by ProfCat, so a value inserted in the
    // middle of the enum has to be inserted here too -- appending it swaps two classes'
    // labels and the profile reads as though one kernel did the other's work.
    "moe_route", "moe_expert", "moe_combine", "matmat_head", "matmat_narrow",
    "matmat_dec_q6", "moe_pair"
};

// The output head's class, by the width that separates it -- see PC_MATMAT_HEAD.
// Defined beside w_bytes, which reads the same wire table.
static bool w_is_q6k(uint32_t wkind);
static inline ProfCat mm_decode_cat(uint32_t n_out, uint32_t wkind = 64u) {
    if (n_out >= 16384u) { return PC_MATMAT_HEAD; }
    if (n_out <= 64u) { return PC_MATMAT_NARROW; }
    return w_is_q6k(wkind) ? PC_MATMAT_DEC_Q6 : PC_MATMAT_DECODE;
}
constexpr uint32_t PROF_MAX_SAMPLES = 4096;     // 2048 dispatches per resolve: the device caps a sample buffer at 32768 B (8 B per sample); a decode token is ~784 dispatches, a 512-token prefill 1255
id<MTLCounterSampleBuffer> g_prof_sbuf = nil;
uint8_t  g_prof_cat[PROF_MAX_SAMPLES / 2];
uint32_t g_prof_pairs = 0;                       // pairs written this command buffer
// TICKS THIS REGION, then SECONDS FOR EVER. The counter buffer reports ticks in the GPU's
// own clock, and the tick period is only known once the region samples the two clocks
// together -- so a category's ticks are converted at the end of the region that produced
// them and only the seconds are kept. Keeping ticks and converting later would use one
// region's period on another's ticks.
double   g_prof_cat_ticks[PC_N];                 // this region, cleared when converted
double   g_prof_cat_s[PC_N];                     // converted, accumulated across regions
uint64_t g_prof_cat_calls[PC_N];
// WEIGHT BYTES per category, so a class reads as a BANDWIDTH and not only a duration. A
// duration says a class is big; bytes over duration says whether it is slow. On this model
// the output head and the dense per-layer projections differ by a third in GB/s while both
// read as "matmat_decode" -- which is the whole reason for counting them.
uint64_t g_prof_cat_bytes[PC_N];
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
    std::vector<uint32_t>      kv_pt_set;    // entries the pool set last (0 = never set)
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    // [lane-count log2 (2..5)][NR0 log2 (0..3), so NR0 in 1,2,4,8]
    id<MTLComputePipelineState> p_q4mm_lanes[6][4];
    id<MTLComputePipelineState> p_ple_gather, p_attn_dec_scoretile, p_attn_dec_combine, p_cvt_f16;
    // Retained so a tile-major FORMAT's pipeline can be built on first use: which formats
    // a model carries is not known until it loads, and building all of them at init would
    // be GPU-resident code for kernels nothing dispatches.
    id<MTLLibrary> lib;
    // [variant][tile][wfmt + 32*rowmajor]. The variant is the rt entry point the route
    // would have taken anyway: plain, _h (half-activation mirror), _gh (gated pair); the
    // tile is a wide shape's index or one of the two narrow tiles (RT_TILE_*).
    id<MTLComputePipelineState> p_rt_fmt[4][RT_TILES][64];
    // The plain Q4_0 family's residual-add twins (RT_HALF_RESID), by tile; built on first use.
    id<MTLComputePipelineState> p_rt_plain_resid[RT_TILES];
    id<MTLComputePipelineState> p_gather_fmt[32];
    id<MTLComputePipelineState> p_gemv_fmt[64];
    // Its decode-rows form: [wfmt + 32*rowmajor][form], the forms holding 2, 4 and 8 tokens.
    id<MTLComputePipelineState> p_blk_rows_fmt[64][3];
    // The rows on the matrix unit (imparo_blk_rows_mma): [wfmt + 32*rowmajor].
    // The matrix-unit rows kernel, per format x layout x units a threadgroup (1, 2, 4) x whether
    // it reads the activations from the half mirror.
    id<MTLComputePipelineState> p_blk_rows_mma_fmt[64][3][2];
    // The same over two 8-row token fragments (imparo_blk_rows_mma_t2), built on first use.
    id<MTLComputePipelineState> p_blk_rows_mma2_fmt[64][3];
    id<MTLComputePipelineState> p_moe_grouped_mma_fmt[64][3];
    // The gate|up pair of the same kernel: one more entry per (format, layout, tiles).
    id<MTLComputePipelineState> p_moe_grouped_mma_pair_fmt[64][3];
    // [wfmt + 32*rowmajor][shape], shape 0 = 32 tokens, 1 = 64 tokens
    // [fmt+layout][moe_st_shape index][bit 0 half source, bit 1 half result]
    id<MTLComputePipelineState> p_moe_st_gemm_fmt[64][4][4];
    id<MTLComputePipelineState> p_repack_q8_tm;
    id<MTLComputePipelineState> p_repack_tm;
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
    // The register-softmax form of the FA op (default; IMPARO_ATTN_FARS=0 keeps the 8-query op).
    id<MTLComputePipelineState> p_fars;
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
    id<MTLComputePipelineState> p_mega_ffn_ple = nil; // the mega layer block (IMPARO_MEGA_FFN>=2), slot 0's instantiation
    id<MTLComputePipelineState> p_mega_layer[2] = { nil, nil }; // per head-dim slot (the attention phase is compile-time in HD)
    id<MTLComputePipelineState> p_mega_q35[2] = { nil, nil };   // the qwen35 layer block per head-dim slot (the weight format comes from the entry, not the pipeline)
    id<MTLComputePipelineState> p_mega_q35_deep[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_q35_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_q35_deep_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_q35_prog[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_q35_prog_q[2] = { nil, nil };
    // The lfm2moe routed feed-forward block per head-dim slot (formats from the entry, like
    // qwen35's). It has no attention phase, so its deep and quantized-cache variants stay nil:
    // the tables that index every family by its variants need the slots to exist.
    id<MTLComputePipelineState> p_mega_l2m[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_l2m_deep[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_l2m_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_l2m_deep_q[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_l2m_prog[2] = { nil, nil };
    id<MTLComputePipelineState> p_mega_l2m_prog_q[2] = { nil, nil };
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
    // The same sharing at hd 256, two and three query heads per threadgroup ([0] = 2,
    // [1] = 3): what the threadgroup budget allows at that head dim.
    id<MTLComputePipelineState> p_attn_dec_stream_g256[2], p_attn_dec_stream_g256_id[2],
        p_attn_dec_stream_g256_q[2], p_attn_dec_stream_g256_q_id[2];
    id<MTLComputePipelineState> p_attn_pre_qtile, p_mma_peak, p_mma_loaded, p_mma_dev_a;
    id<MTLComputePipelineState> p_scoremix;
    id<MTLComputePipelineState> p_spill[8];   // the NACC ladder
    id<MTLComputePipelineState> p_bw_read;
    // Indexed by ConvForm: 0 = gated (LFM2), 1 = plain + SiLU (Qwen3.8's delta net).
    id<MTLComputePipelineState> p_conv[2] = { nil, nil };
    id<MTLComputePipelineState> p_conv_state[2] = { nil, nil };
    // ROW LAYOUT (a tree verify), built on first use by row_layout_ready.
    id<MTLComputePipelineState> p_fa_rows_n[4] = { nil, nil, nil, nil };
    // The register-softmax op over a row layout, built on first use: a verify takes the kernel
    // prefill takes, so a chain's rows equal a causal forward's.
    id<MTLComputePipelineState> p_fars_rows = nil;
    // Its verify variants, indexed bit 0 = ROW_SPLIT (constant 63), bit 1 = ROW_HEADS (64); entry
    // 0 unused (p_fars_rows). And ROW_SPLIT's merge.
    id<MTLComputePipelineState> p_fars_rows_v[4] = { nil, nil, nil, nil };
    id<MTLComputePipelineState> p_rows_combine = nil;
    id<MTLComputePipelineState> p_fa_rows_fq_n[4] = { nil, nil, nil, nil };   // float Q, built on first use
    id<MTLComputePipelineState> p_head_norm_rope_rows = nil;
    id<MTLComputePipelineState> p_conv_rows[2] = { nil, nil };
    id<MTLComputePipelineState> p_conv_row_inputs[2] = { nil, nil };
    id<MTLComputePipelineState> p_shortconv_step = nil;   // one token, gated: conv + state shift
    id<MTLComputePipelineState> p_shortconv_step_plain = nil;  // the same for the plain+SiLU form
    // CO-BATCHED ROWS (function constant 32), built on a process's first co-batched step: the
    // f16 KV store, head norm + rope, the vector decode attention per head-dim slot, the
    // one-token convolution step per form. `tried` keeps a build that failed from being
    // retried every step.
    struct CobPipe { id<MTLComputePipelineState> p = nil; bool tried = false; };
    CobPipe cob_kvstore, cob_head_norm_rope, cob_attn_vec[2], cob_conv_step[2], cob_delta_net;
    id<MTLComputePipelineState> p_delta_net = nil;        // the gated delta rule
    id<MTLComputePipelineState> p_delta_net_tree = nil;   // the same rule over a draft tree
    id<MTLComputePipelineState> p_mul_sigmoid = nil, p_copy_strided = nil, p_scatter_strided = nil;
    std::vector<id<MTLCommandBuffer>> pending;   // flushed, awaiting accounting
    void * pool = nullptr;                      // autorelease pool for one forward pass
    // PIPELINED DECODE (docs/decode-turnaround.md): regions committed by `end_async` and
    // not yet waited on, oldest first. The host encodes decode step N+1 while step N
    // runs; `wait_outstanding` retires the oldest.
    struct Region { std::vector<id<MTLCommandBuffer>> cbs; uint32_t dbg_slot = 0; uint64_t seq = 0; };
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
    // DECODE ROWS (function constant 28 = 2, 4 or 8): the one-row GEMVs over that many independent
    // rows, built on first use (q8mv_tok_pipe). Indexed [rows log2 - 1]; row-major also by the
    // rows per threadgroup.
    id<MTLComputePipelineState> p_q8mv_tok_tm[3];
    id<MTLComputePipelineState> p_q8mv_tok_rows[3][3];
    // CO-BATCHED ROWS through the simdgroup matrix unit (imparo_q8_tm_rows_mma), tile-major only,
    // built on first use: [token columns / 8 - 1][activations from the half mirror].
    id<MTLComputePipelineState> p_q8_rows_mma[RM_MAX_FRAGS_HOST][2];
    // ...and its row-major Q4_0 form (function constant 37), one or two unit tiles a
    // threadgroup (function constant 38).
    id<MTLComputePipelineState> p_q4_rows_mma[RM_MAX_FRAGS_HOST][2][3];
    // The Q4_0 decode-rows twins of p_q4mm_lanes, built on first use (q4mv_tok_pipe): [lanes log2]
    // [rows per lane log2][lane groups splitting the tokens, log2][rows log2 - 1].
    id<MTLComputePipelineState> p_q4mv_tok[6][4][4][3];
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
    id<MTLComputePipelineState> p_q4mm, p_q4mm_pre, p_q4mm_pre_nomma, p_f32mm, p_f32nmm, p_f32gemv, p_rms_staged, p_q4row, p_rms, p_rope, p_head_norm_rope, p_kvstore,
        p_attn_dec_direct, p_actmul, p_act, p_add, p_mul, p_scale, p_copy, p_softcap, p_argmax,
        p_argmax_feed, p_top_k_chunks, p_top_k_merge, p_top_k8_chunks, p_top_k8_merge,
        p_top_k8_one,
        p_logistic_blocks, p_logistic_finish,
        p_moe_gate, p_moe_plan, p_moe_route, p_moe_combine, p_moe_combine_norm,
        p_ple;
    // The routed matmul, one pipeline per weight format (index = the wire value).
    id<MTLComputePipelineState> p_moe_grouped_fmt[64];
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
enum : uint32_t { WT_FAST = 0, WT_SLOW = 1, WT_HOST_STAGED = 2, WT_UNREAD = 3 };

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
// The model file's path: the repack reads converted weights from it
// (imparo_metal_set_weight_path) and the slow tier's readahead advises it (wpf_post).
static std::string g_weight_path;

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
        // A block the forward never runs: no buffer, so nothing of it is ever wired.
        if (w.tier == WT_UNREAD) { continue; }
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

// How many weight segments are in the slow tier (0 = the whole model is wired).
extern "C" uint32_t imparo_metal_slow_segments(void) {
    uint32_t n = 0;
    for (const WSeg & s : g_wsegs) { n += (s.tier == WT_SLOW) ? 1u : 0u; }
    return n;
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
// ---- Slow-tier readahead ----------------------------------------------------------------
// A slow segment is the model file's own pages (the mapping is shared), wired by the driver
// for each command buffer that binds it. Pages the system has dropped are read back by that
// wire before the buffer can start, at about 2.4 GB/s (measured). This thread reads them
// first, with F_RDADVISE at about 4.8 GB/s, in the order the encoder binds them:
//
//   encoder: binds a slow segment -> posts it, tagged with its buffer's slow ordinal
//   thread:  resident -> skip, else F_RDADVISE; never past two slow buffers ahead of the GPU
//   GPU:     that buffer completes -> done(ordinal)
//
// Two buffers ahead is double buffering: the one the GPU runs and the one the SSD fills. A
// deeper window reads no faster than the SSD, and with a model larger than RAM it could
// evict pages before the GPU reached them (reasoned, not measured). The model file only: a
// paired drafter is always in the fast tier (placement). IMPARO_READAHEAD_LOG=1 prints each
// decision. Evidence: docs/evidence/bracket/2026-09-18-slow-tier-streaming.md.
struct WpfItem { uint64_t base, len, cb; uint32_t layer; };
// Never destroyed. The completion handler in wpf_commit runs on Metal's own thread with no
// order against the process's exit, and it locks `mu`; a mutex torn down by the destructors of
// statics fails that lock with EINVAL and aborts the process after its answer (the race the
// idle watcher's comment records). The detached thread parked on `cv` is safe for the same
// reason.
struct WpfState {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<WpfItem> queue;        // posted, in bind order
    uint64_t done_cb = 0;             // highest slow buffer ordinal the GPU has completed
    // The encoder thread's own:
    uint64_t cb = 0;                  // ordinal of the buffer being encoded, 0 = no slow bind yet
    uint64_t ordinals = 0;            // ordinals handed out
    uint64_t last_base = UINT64_MAX;  // the segment this buffer posted last
    int fd = -1;                      // the model file; -1 not opened yet, -2 unavailable
    uint64_t file_len = 0;
};
static WpfState & g_wpf = *new WpfState();
static bool wpf_log(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_READAHEAD_LOG"); on = (e != nullptr && e[0] == '1') ? 1 : 0; }
    return on == 1;
}
static void wpf_loop(void);
// Opens the model file and starts the thread, once. False when there is no file to read.
static bool wpf_ready(void) {
    if (g_wpf.fd >= 0) { return true; }
    if (g_wpf.fd == -2) { return false; }
    const int fd = g_weight_path.empty() ? -1 : open(g_weight_path.c_str(), O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        NSLog(@"imparo metal: no slow-tier readahead, the model file is not readable (%s)",
              g_weight_path.c_str());
        if (fd >= 0) { close(fd); }
        g_wpf.fd = -2;
        return false;
    }
    g_wpf.file_len = (uint64_t)st.st_size;
    g_wpf.fd = fd;
    std::thread(wpf_loop).detach();
    return true;
}
// The encoder binds `s`, a slow segment, into the command buffer it is encoding.
static void wpf_post(const WSeg & s) {
    if (g.cb == nil || !wpf_ready()) { return; }
    if (g_wpf.cb == 0) { g_wpf.cb = ++g_wpf.ordinals; g_wpf.last_base = UINT64_MAX; }
    if (s.base == g_wpf.last_base) { return; }
    g_wpf.last_base = s.base;
    { std::lock_guard<std::mutex> l(g_wpf.mu); g_wpf.queue.push_back({s.base, s.len, g_wpf.cb, s.layer}); }
    g_wpf.cv.notify_one();
}
// Just before `cb` is committed: if it bound a slow segment, its completion moves the window.
static void wpf_commit(id<MTLCommandBuffer> cb) {
    if (g_wpf.cb == 0) { return; }
    const uint64_t k = g_wpf.cb;
    g_wpf.cb = 0;
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        { std::lock_guard<std::mutex> l(g_wpf.mu); if (k > g_wpf.done_cb) { g_wpf.done_cb = k; } }
        g_wpf.cv.notify_one();
    }];
}
static void wpf_loop(void) {
    std::unique_lock<std::mutex> l(g_wpf.mu);
    for (;;) {
        g_wpf.cv.wait(l, [] {
            return !g_wpf.queue.empty() && g_wpf.queue.front().cb <= g_wpf.done_cb + 2;
        });
        const WpfItem it = g_wpf.queue.front();
        g_wpf.queue.pop_front();
        const uint64_t done = g_wpf.done_cb;
        if (it.cb <= done) {   // its buffer already ran: the driver read what it needed
            if (wpf_log()) {
                fprintf(stderr, "[imparo] readahead layer %u buffer %llu: already ran\n", it.layer,
                        (unsigned long long)it.cb);
            }
            continue;
        }
        l.unlock();
        // Resident already? A full mincore of an 84 MB segment costs ~2 ms (measured), more than
        // a decode step's share, so three pages stand for it: a segment is wired and dropped as a
        // unit, so its pages age together. A miss the sample does not see costs only the driver's
        // own read, and F_RDADVISE itself skips whatever part is cached.
        const uint64_t pg = (uint64_t)vm_page_size;
        const uint64_t probe[3] = { it.base, it.base + (it.len / 2 / pg) * pg, it.base + it.len - pg };
        size_t missing = 0;
        const double tc = CACurrentMediaTime();
        for (uint64_t at : probe) {
            char v = 0;
            if (mincore((const void *)(g_map_base + at), (size_t)pg, &v) != 0 || !(v & MINCORE_INCORE)) {
                missing += 1;
            }
        }
        const size_t n = 3;
        const double check_ms = 1e3 * (CACurrentMediaTime() - tc);
        double ms = 0.0;
        if (missing > 0) {
            const double t0 = CACurrentMediaTime();
            const uint64_t end = std::min(it.base + it.len, g_wpf.file_len);
            for (uint64_t at = it.base; at < end; ) {
                const uint64_t take = std::min<uint64_t>(end - at, 1ull << 30);   // ra_count is an int
                struct radvisory ra = { (off_t)at, (int)take };
                if (fcntl(g_wpf.fd, F_RDADVISE, &ra) == -1) { break; }
                at += take;
            }
            ms = 1e3 * (CACurrentMediaTime() - t0);
        }
        if (wpf_log()) {
            fprintf(stderr, "[imparo] readahead layer %u buffer %llu (gpu done %llu): %zu of %zu sampled pages "
                    "missing, check %.3f ms, advise %.1f ms\n", it.layer, (unsigned long long)it.cb,
                    (unsigned long long)done, missing, n, check_ms, ms);
        }
        l.lock();
    }
}
// The segment a dispatch being encoded binds: `wseg_at`, and a slow segment is posted to the
// readahead. Every binding site goes through here; lookups that bind nothing use wseg_at.
static const WSeg & wseg_use(uint64_t off) {
    const WSeg & s = wseg_at(off);
    if (s.tier == WT_SLOW) { wpf_post(s); }
    return s;
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
    const WSeg & s = wseg_use(off);
    [enc setBuffer:s.buf offset:0 atIndex:idx];
    return off - s.base;
}
// The segment an offset falls in, or nullptr. `wseg_at` aborts on a miss because reaching it
// with an unplaced offset is a programming error; a ROUTING question is not, so it asks here.
static const WSeg * wseg_find(uint64_t off) {
    size_t lo = 0, hi = g_wsegs.size();
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (g_wsegs[mid].off + g_wsegs[mid].bytes <= off) { lo = mid + 1; } else { hi = mid; }
    }
    return (lo < g_wsegs.size() && g_wsegs[lo].off <= off) ? &g_wsegs[lo] : nullptr;
}

// Can one bound buffer serve both halves of a pair? A kernel that reads two weights through
// the buffer at index 0 addresses the second by an offset LOCAL TO THE FIRST'S SEGMENT, so
// the answer is no the moment they are in different segments.
static bool w_pair_in_one_segment(uint64_t a, uint64_t b) {
    if (w_absent(a) || w_absent(b)) { return false; }
    const WSeg * sa = wseg_find(a);
    return sa != nullptr && sa == wseg_find(b);
}

// A second offset a kernel reads through the SAME bound buffer: it must fall in the
// sibling's segment, which the caller has already established (imparo_metal_matmat_gated
// refuses a pair that straddles). Still aborts: it is the invariant, not the routing test.
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
// committed at the next region begin. A set attached to the queue is made resident by the
// queue's command buffers whether or not residency was requested (measured), so ATTACHMENT
// is what correctness needs -- the mega route also waits for the first hold, see
// mega_route_open -- and the set is attached only while the placement's fast tier fits the
// budget it was computed for.
//
// NEVER WIRE ON A PATH SOMETHING IS WAITING ON. `requestResidency` returns only once the
// whole set is resident, and on a cold process that is seconds: measured 1478 ms for LFM2's
// 2801 MiB and 6044-13187 ms for Qwen3.8-27B's 15513 MiB. Whatever thread asks for it, the
// machine spends those seconds making room and the display stops with it, so it cannot run
// at load and it cannot run at a region begin:
//
//   Wrong: load -> wire 15.5 GB (7.3 s, display frozen) -> first request
//   Right: load -> first request, command buffers wire what they touch as they run
//                              -> region end -> hold what is already in (ms)
//
// A region that starts unwired is correct (attachment is enough) and paced (the paging
// interleaves with the GPU work that needs it). Residency is then HELD from the region's
// END, where nothing is waiting on an answer and every page has already been touched, for
// a window past the last region (a held set keeps ~4.5 GB wired for E4B): requests inside
// the window skip the OS's own ~80 ms wake, and a server left idle gives the memory back.
//
// Only the FIRST hold is expensive. Measured back to back on the 15513 MiB tier, ending
// residency and asking for it again costs 240 ms, not the 6-13 s the first one did
// -- and 240 ms for 15.5 GB is the same rate as the
// 78-86 ms #129 measured re-wiring E4B's 4.5 GB. So a region begin may pay a RE-hold, and
// that is the wake #129 removed; only the first one has to wait for a region end. That is
// what `g_rset_ever_held` distinguishes.
// IMPARO_METAL_RESIDENCY_IDLE_S: the window (default 180 s, 0 = hold for the process's
// life). IMPARO_METAL_NO_RESIDENCY=1: no set at all (the A/B).
static id<MTLResidencySet> g_rset = nil;
static bool   g_rset_dirty = false;
static bool   g_rset_held = false;      // requestResidency called and not yet ended
static bool   g_rset_ever_held = false; // ... at least once: the disk read is behind us
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

// The system's memory pressure, reported by a dispatch source and taken by the KV tier, which
// gives idle conversations back when there is any. Set from the source's queue, read and
// cleared by whoever asks.
static std::atomic<bool> g_mem_pressure{false};
static dispatch_source_t g_mem_source = nil;
static void mem_pressure_watch(void) {
    if (g_mem_source != nil) { return; }
    g_mem_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    if (g_mem_source == nil) { return; }
    dispatch_source_set_event_handler(g_mem_source, ^{
        const unsigned long level = dispatch_source_get_data(g_mem_source);
        if ((level & (DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL)) != 0) {
            g_mem_pressure.store(true);
        }
    });
    dispatch_resume(g_mem_source);
}
extern "C" uint32_t imparo_metal_take_memory_pressure(void) {
    return g_mem_pressure.exchange(false) ? 1u : 0u;
}
// Seconds the residency is held past the last region before it is released; 0 when it is held
// for good or there is no residency set.
extern "C" uint32_t imparo_metal_idle_release_s(void) {
    return g_rset == nil ? 0u : (uint32_t)g_rset_idle_s;
}

static void rset_init(void) {
    if (getenv("IMPARO_METAL_NO_RESIDENCY") != NULL) { return; }
    if (const char * e = getenv("IMPARO_METAL_RESIDENCY_IDLE_S")) { g_rset_idle_s = atoi(e); }
    if (g_rset_idle_s < 0) { g_rset_idle_s = 0; }
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
                if (!g_rset_held || g_rset_in_region) { continue; }
                if (CACurrentMediaTime() - g_rset_last_use < (double)g_rset_idle_s) { continue; }
                [g_rset endResidency];
                g_rset_held = false;
                kvset_release_hold();
            }
        });
        atexit(rset_stop_thread);
    }
}
// `commit` applies pending membership changes, so what it should scale with is the NUMBER of
// allocations and how many of them changed -- not the set's bytes. Both are on the line.
static uint32_t g_rset_n = 0;         // allocations in the set
static uint32_t g_rset_changed = 0;   // add/remove calls since the last commit
// THE TIER DID NOT FIT, so stop asking the OS to wire it. Set when one piece of the wire at
// load runs past the host-stall budget, and never cleared: re-testing a bound by doing the
// thing it exists to prevent is the mistake #182's ratchet comment names, and here the thing
// costs the user their machine for two minutes. Attachment is untouched, so correctness is
// unaffected and the pages stay pageable, which is what they were before #129. The mega
// route is not: it opens only after the first hold (g_rset_attached_now), and a refused hold
// never makes one, so decode stays on the dispatch path for the rest of the process.
static bool g_rset_hold_refused = false;

static void rset_add(id<MTLBuffer> b) {
    if (g_rset == nil || b == nil) { return; }
    [g_rset addAllocation:b];
    g_rset_bytes += (uint64_t)[b length];
    g_rset_n += 1;
    g_rset_changed += 1;
    g_rset_dirty = true;
}
static void rset_remove(id<MTLBuffer> b) {
    if (g_rset == nil || b == nil) { return; }
    [g_rset removeAllocation:b];
    const uint64_t n = (uint64_t)[b length];
    g_rset_bytes = n <= g_rset_bytes ? g_rset_bytes - n : 0;
    if (g_rset_n) { g_rset_n -= 1; }
    g_rset_changed += 1;
    g_rset_dirty = true;
}
// Region begin: commit membership changes, attach and request while under budget.
static bool g_rset_attached_now(void) { return g_rset == nil || (g_rset_attached && g_rset_ever_held); }
// Commit membership and attach; with `hold`, also ask the OS to keep the pages wired.
// THE CALLER HOLDS g_rset_mu. Only the region END passes hold=true -- see the rule above.
static void rset_sync(bool hold) {
    const double t_commit0 = CACurrentMediaTime();
    const uint32_t changed = g_rset_changed;
    if (g_rset_dirty) {
        [g_rset commit];
        g_rset_dirty = false;
        g_rset_changed = 0;
        // A committed set is no longer held: the membership it was held over is gone.
        g_rset_held = false;
    }
    const double t_commit1 = CACurrentMediaTime();
    if (g_rset_bytes > g_rset_budget) {
        // The runtime sized the fast tier; landing here means a buffer grew past it
        // (a KV pool beyond the reserve). Fall back to per-command-buffer residency.
        if (g_rset_attached) {
            [g.queue removeResidencySet:g_rset];
            g_rset_attached = false;
        }
        if (g_rset_held) {
            [g_rset endResidency];
            g_rset_held = false;
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
    // ATTACHMENT IS THE GUARANTEE, and it happens once: a set attached to the queue is
    // made resident by that queue's command buffers whether or not residency was
    // requested. `requestResidency` only decides WHEN the pages are wired, and the tier
    // is wired at load, so after load neither call does anything.
    if (!g_rset_attached) {
        [g.queue addResidencySet:g_rset];
        g_rset_attached = true;
    }
    if (hold && !g_rset_held && !g_rset_hold_refused) {
        [g_rset requestResidency];
        g_rset_held = true;
        g_rset_ever_held = true;
    }
    // TIMED APART, because they scale with different things: `commit` walks the
    // MEMBERSHIP list, `requestResidency` walks the PAGES -- and every commit forces a
    // wire, by clearing the held flag above. Measured on one unchanged 15509 MiB set:
    //
    //   quiet machine                     commit 0.0 ms, wire     26.8 ms
    //   one other 15 GB process alive                    wire   4895.9 ms
    //   memory over-subscribed                           wire 122177.6 ms
    //
    // So the bill is the pages, never the walk, and it is unbounded in memory pressure
    // (that is the defect; this line is only how you see it). It runs at a region BEGIN,
    // ahead of the region's GPU work, so report a slow one with both halves named.
    const double t_end = CACurrentMediaTime();
    if (t_end - t_commit0 > 0.001) {
        NSLog(@"imparo metal: residency %s %llu MiB in %.1f ms "
              "(commit %.1f ms, wire %.1f ms, %u allocations, %u changed)",
              hold ? "held" : "attached", (unsigned long long)(g_rset_bytes >> 20),
              1e3 * (t_end - t_commit0), 1e3 * (t_commit1 - t_commit0),
              1e3 * (t_end - t_commit1), g_rset_n, changed);
    }
}
// Region begin: re-hold if an idle release let it go. That is cheap -- the pages are unwired
// but still in RAM (measured 235-251 ms for 15513 MiB, against 4275 ms cold) -- and it is the
// wake #129 removed. The FIRST hold already happened at load, in pieces.
static void rset_begin(void) {
    if (g_rset == nil) { return; }
    std::lock_guard<std::mutex> lk(g_rset_mu);
    g_rset_in_region = true;
    rset_sync(true);
    kvset_sync_locked();
}
// THE WEIGHT WINDOW: the largest piece of a fast segment that one step of the repack or of
// the wire at load handles at once (the repack's twin, further down, is why there is a window).
//
// The cap is DERIVED from the headroom this placement actually has -- budget minus what the
// fast tier already holds -- and half of that is left for everything else the process needs
// (the KV pool, activations, the residency set's own accounting). Clamped so a tiny headroom
// still makes progress and a huge one does not allocate more than a window needs to amortise
// its command buffer. The wire at load (imparo_metal_wire_weights) pieces every fast segment
// by the same windows, so its stall bound never meets a piece larger than this.
static uint64_t weight_window_cap() {
    uint64_t fast = 0;
    for (const WSeg & s : g_wsegs) { if (s.tier == WT_FAST) { fast += s.bytes; } }
    const uint64_t head = g_placement_budget > fast ? g_placement_budget - fast : 0;
    uint64_t cap = head / 2;
    const uint64_t lo = 64ull << 20, hi = 1024ull << 20;
    if (cap < lo) { cap = lo; }
    if (cap > hi) { cap = hi; }
    return cap;
}
// Called once every load-time allocation has joined the set. Attaches it, which is all
// correctness needs, and holds it, which the mega route waits for (g_rset_attached_now); a
// later `addAllocation` dirties the set and the next region re-commits, which is the
// pre-existing behaviour.
extern "C" void imparo_metal_wire_weights(double stall_budget_s) {
    if (g_rset == nil) { return; }
    std::lock_guard<std::mutex> lk(g_rset_mu);
    // WIRE IN SEGMENT-SIZED PIECES, THEN HOLD. `requestResidency` over a cold 15.5 GB tier is
    // one call the host cannot interrupt -- 4275 ms even with the memory pressure of the
    // whole-segment bind removed (cf4421b), and 6044-13187 ms before it. A residency set is
    // requested as a unit, so the call itself cannot be split.
    //
    // What CAN be split is the paging. Metal makes a referenced resource resident WHOLE for
    // the duration of the command buffer that references it -- the same rule that made the
    // repack wire 14.6 GB per window -- so a residency set over a wrapper of one window of a
    // segment pages in that window and nothing else. Every piece is at most
    // weight_window_cap(): short waits instead of one long one, and the host gets the GPU back
    // between each.
    //
    // The hold afterwards is then over pages that are already in, which is the 1.2 ms case,
    // not the 4275 ms one. That is the whole reason this can run at load: nothing is deferred
    // to the first request, so no warm is needed in the server or in a harness.
    const double tw0 = CACurrentMediaTime();
    uint32_t pieces = 0;
    double worst = 0.0;
    std::vector<id<MTLResidencySet>> per_seg;
    std::vector<id<MTLBuffer>> wraps;   // a set is not relied on to keep its allocations alive
    // THE PIECE IS A WINDOW, NOT A SEGMENT. A segment the repack never tiles used to be wired
    // as one piece of its whole size: LFM2's one 2733 MiB segment took 1220-1504 ms cold and
    // E4B's one 4005 MiB segment 1200-2314 ms, on an idle machine, past the 1000 ms budget.
    // So a cold start refused the hold, and the mega route, which waits for the first hold,
    // stayed closed for the process. The bound is for the OS making room, not for reading the
    // file, and a window keeps a cold read under it.
    const uint64_t page = 16384;
    const uint64_t window = std::max(page, weight_window_cap() & ~(page - 1));
    uint64_t planned = 0;
    for (const WSeg & sg : g_wsegs) {
        if (sg.tier == WT_FAST && sg.buf != nil) { planned += (sg.len + window - 1) / window; }
    }
    for (const WSeg & sg : g_wsegs) {
        if (sg.tier != WT_FAST || sg.buf == nil) { continue; }
        for (uint64_t w0 = 0; w0 < sg.len && !g_rset_hold_refused;) {
            uint64_t span = std::min(window, sg.len - w0);
            id<MTLBuffer> piece = sg.buf;
            if (span != sg.len) {
                // A wrapper binds its window only, as the repack's source windows do.
                id<MTLBuffer> wrap = [g.device
                    newBufferWithBytesNoCopy:(void *)((uint8_t *)[sg.buf contents] + w0)
                                      length:(NSUInteger)span
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
                if (wrap != nil) {
                    piece = wrap;
                    wraps.push_back(wrap);
                } else {
                    span = sg.len - w0;   // no wrapper: wire the rest of the segment whole
                }
            }
            w0 += span;
            MTLResidencySetDescriptor * d = [[MTLResidencySetDescriptor alloc] init];
            d.label = @"imparo weight window";
            d.initialCapacity = 1;
            NSError * err = nil;
            id<MTLResidencySet> one = [g.device newResidencySetWithDescriptor:d error:&err];
            if (one == nil) { continue; }   // fall through: the tier set below still wires it
            [one addAllocation:piece];
            [one commit];
            [g.queue addResidencySet:one];
            const double p0 = CACurrentMediaTime();
            [one requestResidency];
            const double took = CACurrentMediaTime() - p0;
            worst = std::max(worst, took);
            per_seg.push_back(one);
            pieces += 1;
            // ONE PIECE OVER THE BUDGET MEANS THE TIER DOES NOT FIT. A piece whose pages are
            // already in memory is tens of milliseconds (26.0 ms worst, measured over 15
            // pieces of at most 1024 MiB); a piece that takes seconds is the OS evicting to
            // make room, and the host cannot preempt it. Finishing the loop would pay that
            // for every remaining piece and then again, whole, at the tier hold -- 122 s
            // measured, with the Mac unusable throughout. Stop, and leave the rest pageable.
            if (stall_budget_s > 0.0 && took > stall_budget_s) {
                g_rset_hold_refused = true;
                NSLog(@"imparo metal: wiring the fast tier stalled %.0f ms on one %llu MiB "
                      "piece, past the %.0f ms budget; %u of %llu pieces wired and the rest "
                      "stay pageable (the OS faults them in). The mega route stays closed "
                      "for this process: it opens after the first hold, and none is made. "
                      "IMPARO_CB_STALL_MS sets the budget, IMPARO_FAST_TIER_MB shrinks the "
                      "tier.",
                      1e3 * took, (unsigned long long)([piece length] >> 20),
                      1e3 * stall_budget_s, pieces, (unsigned long long)planned);
            }
        }
    }
    const double tw1 = CACurrentMediaTime();
    // The tier set's own hold now finds every page already wired, so it is the microsecond
    // case rather than the 4275 ms one. Once it holds them, the per-segment sets have done
    // their job and must let go -- otherwise they would keep the tier wired past the idle
    // release and #129's "a server left idle gives the memory back" would silently stop.
    rset_sync(true);
    for (id<MTLResidencySet> one : per_seg) {
        [one endResidency];
        [g.queue removeResidencySet:one];
    }
    // LETTING GO IS NOT FREE, and load pays for it. Ending the window sets leaves the driver
    // work it does at the next submission that carries a resource. Measured on LFM2.5 (a
    // 2733 MiB tier in 3 windows): one blit here waited 17.2 ms before the GPU started it. With
    // no submission here the first request paid instead -- 18-47 ms inside its first buffer
    // release (the driver's resource finalize), or 27-53 ms inside the first KV preparation's
    // requestResidency, which then held up the prefill beside it. Keeping the window sets
    // alive made that release 0.0 ms, so it is their end, not the window buffers. An EMPTY
    // command buffer does not do it (0.28 ms, and the release after it still took 47 ms).
    double settle_ms = 0.0;
    if (!per_seg.empty()) {
        per_seg.clear();
        const double s0 = CACurrentMediaTime();
        id<MTLBuffer> scratch = [g.device newBufferWithLength:16384 options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        [be fillBuffer:scratch range:NSMakeRange(0, 16384) value:0];
        [be endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        settle_ms = 1e3 * (CACurrentMediaTime() - s0);
    }
    if (pieces != 0) {
        NSLog(@"imparo metal: weights wired in %u pieces in %.1f ms (worst piece %.1f ms); "
              "the driver settled their release in %.1f ms",
              pieces, 1e3 * (tw1 - tw0), 1e3 * worst, settle_ms);
    }
    // Not in a region: the idle timer starts from here, exactly as after a real one.
    g_rset_last_use = CACurrentMediaTime();
}
// Region end (after the wait): the answer is out, so this is where the pages are held --
// every one of them was touched by the region that just ran. The idle window starts now.
static void rset_end(void) {
    if (g_rset == nil) { return; }
    std::lock_guard<std::mutex> lk(g_rset_mu);
    g_rset_in_region = false;
    // Only when it would DO something: once held and clean, this ran four clock reads and
    // three false branches at every region end, which for decode is every token.
    if (!g_rset_held || g_rset_dirty) { rset_sync(true); }
    kvset_sync_locked();
    kvset_hold();
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
                  w[15], (w[15] >> 4) & 0xfffu, (w[15] >> 16) & 0xffu, g_mega_last_tgs, w[15] >> 24, w[3], w[11], w[12]);
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
    if (!mega_blocks_wanted()) { return 0; }
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

static void prof_begin(uint8_t cat, uint64_t wbytes = 0) {
    // Count first: dispatch counts are exact and need no hardware support, while the
    // timestamps below need a counter-sampling capability this device does not report.
    // Counting only when sampling works would have made the counts silently unavailable.
    g_prof_cat_calls[cat] += 1;
    g_prof_cat_bytes[cat] += wbytes;
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

// One command buffer's GPU duration, or 0 when the stamps do not describe one. See the
// call site: an unguarded subtraction of these is the profile's own silent-garbage path.
static double cb_gpu_seconds(id<MTLCommandBuffer> cb) {
    const double a = [cb GPUStartTime], b = [cb GPUEndTime];
    return (a > 0.0 && b > a) ? b - a : 0.0;
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

// WHICH GGML TYPE EACH WIRE KIND IS -- handed over once at load by the common runtime
// (Backend::set_weight_kind_types), never derived here. A tile-major id is 1000 + its
// source's, and the shader's WFMT codes ARE the source ids, so the decode format is the
// low part; a row-major kind has no WFMT and stays on the kernels it always used.
static uint32_t g_wire_ggml[64];
static bool g_wire_ggml_set = false;

// DISPATCHES THIS PROCESS REFUSED. Every refusal below already NSLogs, and that was not
// enough: a tuner run refused 38005 k-quant matmats, printed 38005 lines, and still wrote
// a config in which every prefill-tile candidate had measured the same empty dispatch.
// A log line is not a result a harness can act on, so the count is readable and a harness
// that must have dispatched refuses to record anything when it is nonzero.
static uint64_t g_refused = 0;
extern "C" uint64_t imparo_metal_refused_dispatches(void) { return g_refused; }

// WHICH KERNEL EACH MATMUL RAN, counted per call by route. A harness prints the counts that grew
// while it measured, so a number names the kernel behind it. Without them a route that could not
// run and handed its dispatch to another reads exactly like the route itself: the GEMM without
// room for its padded tile gave 2..4 rows to the one-row block GEMV, which re-reads every weight
// per row, and a benchmark recorded that as the GEMM. A route named *_fallback ran because the
// chosen kernel could not; it is also logged, once per shape.
enum MatmatRoute : uint32_t {
    MR_Q4_GEMV, MR_Q4_GEMV_ROWS, MR_Q4_ROWS_MMA, MR_Q4_GEMM,
    MR_Q8_GEMV, MR_Q8_GEMV_ROWS, MR_Q8_ROWS_MMA, MR_Q8_TILE, MR_Q8_GEMM,
    MR_BLK_GEMV, MR_BLK_GEMV_ROWS, MR_BLK_ROWS_MMA, MR_BLK_ROWS_MMA_H, MR_BLK_GEMM,
    MR_BLK_GEMV_TOKENS_FALLBACK,
    MR_COUNT
};
static const char * const MR_NAMES[MR_COUNT] = {
    "q4.gemv", "q4.gemv_rows", "q4.rows_mma", "q4.gemm",
    "q8.gemv", "q8.gemv_rows", "q8.rows_mma", "q8.tile", "q8.gemm",
    "blk.gemv", "blk.gemv_rows", "blk.rows_mma", "blk.rows_mma_h", "blk.gemm",
    "blk.gemv_tokens_fallback",
};
static uint64_t g_route_counts[MR_COUNT] = {};
static inline void route_count(MatmatRoute r) { g_route_counts[r] += 1u; }
extern "C" uint32_t imparo_metal_route_count_n(void) { return MR_COUNT; }
extern "C" const char * imparo_metal_route_name(uint32_t i) {
    return i < MR_COUNT ? MR_NAMES[i] : "";
}
extern "C" uint64_t imparo_metal_route_count(uint32_t i) {
    return i < MR_COUNT ? g_route_counts[i] : 0u;
}

extern "C" void imparo_metal_set_weight_kind_types(const uint32_t * pairs, uint32_t n) {
    for (uint32_t i = 0; i < 64u; ++i) { g_wire_ggml[i] = 0u; }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t wire = pairs[2 * i], ggml = pairs[2 * i + 1];
        if (wire < 64u) { g_wire_ggml[wire] = ggml; }
    }
    g_wire_ggml_set = true;
}

// The shader's WFMT for a wire kind: 0 (row-major) unless the kind is tile-major AND its
// source has a decode brick.
static bool wfmt_has_brick(uint32_t src) {
    switch (src) {
        // The ggml types tm_sub32 decodes: Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
        // IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M.
        case 3: case 6: case 7: case 10: case 11: case 12: case 13: case 14: case 16:
        case 17: case 18: case 19: case 20: case 21: case 22: case 23: case 29: return true;
        default: return false;                       // a format with no brick arm
    }
}
// THE BYTES A WEIGHT OF THIS KIND OCCUPIES, from the ggml type the wire kind carries. Used
// to report a profile category as a bandwidth; a type this table does not know returns 0,
// which the printer shows as no bandwidth rather than as a wrong one.
// Q6_K by the ggml type the wire kind carries; the tile-major layout moves bytes, it never
// changes which type they came from.
static bool w_is_q6k(uint32_t wkind) {
    if (!g_wire_ggml_set || wkind >= 64u) { return false; }
    const uint32_t t = g_wire_ggml[wkind];
    return (t >= 1000u ? t - 1000u : t) == 14u;
}

static uint64_t w_bytes(uint32_t wkind, uint64_t n_in, uint64_t n_out) {
    if (!g_wire_ggml_set || wkind >= 64u) { return 0ull; }
    const uint32_t t = g_wire_ggml[wkind];
    const uint32_t src = t >= 1000u ? t - 1000u : t;     // the tile-major layout moves bytes,
    uint32_t blk = 0, bytes = 0;                         // it never adds or drops one
    switch (src) {
        case 0:  blk = 1;   bytes = 4;   break;          // F32
        case 1:  blk = 1;   bytes = 2;   break;          // F16
        case 30: blk = 1;   bytes = 2;   break;          // BF16
        case 2:  blk = 32;  bytes = 18;  break;          // Q4_0
        case 3:  blk = 32;  bytes = 20;  break;          // Q4_1
        case 6:  blk = 32;  bytes = 22;  break;          // Q5_0
        case 7:  blk = 32;  bytes = 24;  break;          // Q5_1
        case 8:  blk = 32;  bytes = 34;  break;          // Q8_0
        case 20: blk = 32;  bytes = 18;  break;          // IQ4_NL
        case 10: blk = 256; bytes = 84;  break;          // Q2_K
        case 11: blk = 256; bytes = 110; break;          // Q3_K
        case 12: blk = 256; bytes = 144; break;          // Q4_K
        case 13: blk = 256; bytes = 176; break;          // Q5_K
        case 14: blk = 256; bytes = 210; break;          // Q6_K
        case 16: blk = 256; bytes = 66;  break;          // IQ2_XXS
        case 17: blk = 256; bytes = 74;  break;          // IQ2_XS
        case 18: blk = 256; bytes = 98;  break;          // IQ3_XXS
        case 19: blk = 256; bytes = 50;  break;          // IQ1_S
        case 21: blk = 256; bytes = 110; break;          // IQ3_S
        case 22: blk = 256; bytes = 82;  break;          // IQ2_S
        case 23: blk = 256; bytes = 136; break;          // IQ4_XS
        case 29: blk = 256; bytes = 56;  break;          // IQ1_M
        default: return 0ull;
    }
    return n_in * n_out / (uint64_t)blk * (uint64_t)bytes;
}

static uint32_t wfmt_for(uint32_t wkind) {
    if (!g_wire_ggml_set || wkind >= 64u) { return 0u; }
    const uint32_t t = g_wire_ggml[wkind];
    const bool tile_major = t >= 1000u;
    const uint32_t src = tile_major ? (t - 1000u) : t;
    // THE MULTI-SPAN FORMATS: Q2_K, IQ3_S, IQ2_XS, IQ2_S and IQ3_XXS each carry scales in
    // two or three separate runs of the source block, so a row-major block has no single
    // scale pointer for the brick to take. Tile-major it does -- the layout rule gathers
    // every run into the unit's scale region -- so the brick serves only that layout, and
    // `serves_weight_type` refuses the row-major kind at load for the same reason.
    if (!tile_major && (src == 10u || src == 17u || src == 18u
                        || src == 21u || src == 22u)) { return 0u; }
    // TILE-MAJOR Q4_0 is the brick's (WF_Q4_0). Only expert stacks are converted to it --
    // a 2-D Q4_0 tensor stays row-major and keeps the Q4_0 kernels, which is why row-major
    // Q4_0 is not a brick format here (see moe_wfmt_for for the routed kernels' row-major arm).
    // Its id is 1001 (GGML_Q4_0_TM), NOT 1000 + 2: the first two tile-major ids, Q8_0_TM
    // 1000 and Q4_0_TM 1001, were assigned before the "1000 + source type" numbering the
    // rest follow, so the subtraction above reads it as type 1.
    if (t == 1001u) { return 2u; }
    return wfmt_has_brick(src) ? src : 0u;
}
// THE ROUTED MATMULS' FORMAT: wfmt_for's, plus Q4_0. Q4_0 has its own dense kernels (E4B's),
// so wfmt_for keeps it out of the block-format family -- adding it there would reroute every
// dense Q4_0 matmul. An expert stack has no other kernel, and without this a Q4_0 MoE file
// (LFM2.5-8B-A1B-Q4_0) was refused at its first routed layer. Row-major only: the load-time
// transform leaves Q4_0 row-major (its tile-major rule exists; nothing reads it).
static uint32_t moe_wfmt_for(uint32_t wkind) {
    const uint32_t w = wfmt_for(wkind);
    if (w != 0u || !g_wire_ggml_set || wkind >= 64u) { return w; }
    return g_wire_ggml[wkind] == 2u ? 2u : 0u;
}
// WHICH WIRE KINDS THE REGISTER-TILED GEMM SERVES, one bit per kind. `matmat_impl` says
// it in one line -- `(is_q4 && n_tok > gemv_max_tok) || wfmt != 0` -- and this answers the
// same question for the tuner from the same `wfmt_for`, so the two cannot disagree.
//
// It exists because they did. rt_shape selects the rt tile, and its `applies` predicate
// read "Q4_0 present", which was the whole rt route when it was written. The tile-major
// family joined that route and the predicate did not, so on a k-quant model the tuner
// skipped the knob that picks the tile of the kernel doing 94% of its prefill.
extern "C" uint64_t imparo_metal_rt_route_kinds(void) {
    uint64_t mask = 1ull << 1;                     // Q4_0, above the GEMV crossing
    for (uint32_t k = 0; k < 64u; ++k) {
        if (wfmt_for(k) != 0u) { mask |= 1ull << k; }
    }
    return mask;
}
// Whether that format kept the ROW-MAJOR layout: a row-gathered tensor by rule, and the
// lm head when it is tied to one.
static bool wfmt_is_rowmajor(uint32_t wkind) {
    if (!g_wire_ggml_set || wkind >= 64u) { return false; }
    return g_wire_ggml[wkind] < 1000u;
}
// THE MEGA ROW BRICK'S FORMAT for a wire kind, or 0 when it cannot read the weight. The
// mega phases read TILE-MAJOR only (the route needs the fast tier, which is where the
// load-time transform puts these tensors), so a row-major kind is 0 = refuse -- and 0 is
// also `tm_sub32`'s row-major arm, which decodes nothing, so a caller must treat it as a
// refusal rather than pass it on. Kept separate from `wfmt_for` on purpose: that one is
// the PIPELINE's stamped format and changing it would restamp existing kernels.
extern "C" uint32_t imparo_metal_mega_wfmt(uint32_t wkind) {
    if (wfmt_is_rowmajor(wkind)) { return 0u; }
    return wfmt_for(wkind);
}

id<MTLComputePipelineState> make_with(id<MTLLibrary> lib, NSString * name,
                                      MTLFunctionConstantValues * cv) {
    NSError * e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:name constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: function %@ not found: %@", name, e); return nil; }
    id<MTLComputePipelineState> p =
        [g.device newComputePipelineStateWithFunction:f error:&e];
    if (p == nil) { NSLog(@"imparo metal: pipeline %@ failed: %@", name, e); }
    // IMPARO_PIPE_LOG prints what the compiler made of it, as the load-time builder does.
    if (p != nil && getenv("IMPARO_PIPE_LOG") != NULL) {
        NSLog(@"imparo metal pipe: %@ max_threads=%lu simd_width=%lu static_tg_bytes=%lu", name,
              (unsigned long)[p maxTotalThreadsPerThreadgroup],
              (unsigned long)[p threadExecutionWidth],
              (unsigned long)[p staticThreadgroupMemoryLength]);
    }
    return p;
}

id<MTLComputePipelineState> make(id<MTLLibrary> lib, NSString * name) {
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    stamp_epi_act(cv);
    return make_with(lib, name, cv);
}

// THE STRADDLE VARIANT of an rt_gemm pipeline: the same function and constants plus RT_EDGE8
// (constant 30), for an n_out that is not a multiple of 8, where the straight store's last
// 8x8 block straddles n_out (see the shader). Built on first use; compiled into every
// pipeline, that store cost E4B's prefill 16% without ever running there.
struct RtSpec { NSString * name; MTLFunctionConstantValues * cv; };
static std::unordered_map<const void *, RtSpec> g_rt_spec;   // rt pipeline -> how it was built
static std::unordered_map<const void *, id<MTLComputePipelineState>> g_rt_edge;

// Records how an rt_gemm pipeline was built, so rt_for_rows can build its variant.
static id<MTLComputePipelineState> rt_register(id<MTLComputePipelineState> p, NSString * name,
                                               MTLFunctionConstantValues * cv) {
    if (p != nil) { g_rt_spec[(__bridge const void *)p] = RtSpec{name, cv}; }
    return p;
}

// The pipeline a dispatch of n_out output rows runs: `p`, or for an rt_gemm pipeline and an
// n_out that is not a multiple of 8, its RT_EDGE8 variant. nil when that variant cannot be
// built, and the caller refuses the dispatch.
static id<MTLComputePipelineState> rt_for_rows(id<MTLComputePipelineState> p, uint32_t n_out) {
    if (p == nil || (n_out % 8u) == 0u) { return p; }
    const void * key = (__bridge const void *)p;
    const auto spec = g_rt_spec.find(key);
    if (spec == g_rt_spec.end()) { return p; }
    const auto made = g_rt_edge.find(key);
    if (made != g_rt_edge.end()) { return made->second; }
    MTLFunctionConstantValues * cv = [spec->second.cv copy];
    const bool edge = true;
    [cv setConstantValue:&edge type:MTLDataTypeBool atIndex:30];
    id<MTLComputePipelineState> v = make_with(g.lib, spec->second.name, cv);
    if (v == nil) {
        NSLog(@"imparo metal: no n_out %% 8 variant of %@; its dispatch is refused",
              spec->second.name);
    } else {
        NSLog(@"imparo metal: built the n_out %% 8 variant of %@ (n_out %u)",
              spec->second.name, n_out);
    }
    g_rt_edge[key] = v;
    return v;
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
    g_float_stale &= ~writes;
    if (!g_concurrent) { return; }
    if ((reads & g_haz_writes) || (writes & g_haz_writes) || (writes & g_haz_reads)) {
        if (g_prof) { g_prof_barriers += 1; }
        // DIAGNOSTIC (IMPARO_HAZ_SKIP=1): do not emit the barrier this conflict needs. The
        // dispatches still run, in the same order, with the same grids -- they just race, so
        // the ANSWER IS WRONG and only the time may be read. This is the one instrument that
        // separates the two readings of the decode chain's ~2.2 ms: whether it is the
        // serialisation those barriers impose (removing them gives it back) or slack the rest
        // of the token already absorbs (removing them gives nothing). Never set in the engine.
        // IMPARO_HAZ_SKIP=N skips every Nth barrier (1 = all of them). A COUNT rather than a
        // flag because the total is not the marginal: skipping all 256 a token saves 1.9 ms,
        // but the elementwise class skip -- which drops 32 dispatches AND their barriers --
        // saved 0.009 ms. So 1.9/256 = 7.4 us is an AVERAGE and the cost is concentrated in
        // the barriers that follow BIG dispatches, where the drain is long.
        static uint64_t haz_n = 0;
        const bool drop = g_haz_skip != 0u && (++haz_n % (uint64_t)g_haz_skip) == 0u;
        if (!drop) {
            // NAME THE RESOURCES rather than the whole scope where we can. The masks here are
            // already exact, and `memoryBarrierWithScope:MTLBarrierScopeBuffers` waits for
            // EVERY outstanding buffer write; naming only the buffers this conflict is about
            // asks the driver to wait for less. Same ordering guarantee, same bits.
            // The KV caches live outside g.bufs (their hazard bits are 62 and 63), so a
            // conflict touching them takes the scope form.
            const uint64_t conflict =
                (reads & g_haz_writes) | (writes & g_haz_writes) | (writes & g_haz_reads);
            id<MTLResource> __unsafe_unretained rs[B_COUNT];
            NSUInteger nr = 0;
            const bool kv = (conflict & (HZ_KVK | HZ_KVV)) != 0ull;
            if (!kv && g_haz_named) {
                for (uint32_t i = 0; i < B_COUNT; ++i) {
                    if ((conflict & (1ull << i)) && g.bufs[i] != nil) { rs[nr++] = g.bufs[i]; }
                }
            }
            if (nr != 0) {
                [g.enc memoryBarrierWithResources:rs count:nr];
            } else {
                [g.enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        }
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
extern "C" void imparo_metal_set_haz_skip(uint32_t n) { g_haz_skip = n; }
extern "C" void imparo_metal_set_haz_named(uint32_t on) { g_haz_named = on != 0u; }

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
extern "C" void imparo_metal_set_decode_rows(uint32_t route) { g_decode_rows = route <= 2u ? route : 0u; }
extern "C" void imparo_metal_set_q8_rows_gemv_max(uint32_t n) { g_q8_rows_gemv_max = n; }
extern "C" uint32_t imparo_metal_q8_rows_gemv_max(void) { return g_q8_rows_gemv_max; }
extern "C" void imparo_metal_set_q8_tm_rows_mma_max(uint32_t n) {
    g_q8_tm_rows_mma_max = std::min(n, RM_MAX_ROWS_HOST);
}
extern "C" uint32_t imparo_metal_q8_tm_rows_mma_max(void) { return g_q8_tm_rows_mma_max; }
extern "C" void imparo_metal_set_q4_rows_mma_max(uint32_t n) {
    g_q4_rows_mma_max = std::min(n, RM_MAX_ROWS_HOST);
}
extern "C" uint32_t imparo_metal_q4_rows_mma_max(void) { return g_q4_rows_mma_max; }
extern "C" uint64_t imparo_metal_rows_mma_dispatches(void) { return g_rows_mma_dispatches; }
// EVERY format, so a test or the bench can put them all on one kernel. 0 drops the override and
// each format goes back to the crossing measured for it, which is what a served run uses.
extern "C" void imparo_metal_set_blk_rows_gemv_max(uint32_t n) {
    if (n == 0u) { g_blk_rows_gemv_max_set = false; return; }
    const uint32_t v = std::min(n, BLK_ROWS_MAX_HOST);
    for (uint32_t i = 0; i < BLK_WFMT_COUNT; ++i) { g_blk_rows_gemv_max_fmt[i] = v; }
    g_blk_rows_gemv_max_set = true;
}
// The override, or 0 when there is none and the per-format table applies. A caller saving and
// restoring this round-trips either state.
extern "C" uint32_t imparo_metal_blk_rows_gemv_max(void) {
    return g_blk_rows_gemv_max_set ? g_blk_rows_gemv_max_fmt[0] : 0u;
}
// The crossing one weight kind takes WITHOUT a per-tensor seat. 0 is a kind with no block rows
// kernel. `imparo_metal_blk_rows_gemv_max_at` answers for a tensor, seat included.
extern "C" uint32_t imparo_metal_blk_rows_gemv_max_for(uint32_t wkind) {
    const uint32_t wf = wfmt_for(wkind);
    return wf == 0u ? 0u : blk_rows_gemv_max_fmt(wf);
}
extern "C" uint32_t imparo_metal_blk_rows_gemv_max_at(uint32_t wkind, uint32_t n_in,
                                                      uint32_t n_out) {
    const uint32_t wf = wfmt_for(wkind);
    return wf == 0u ? 0u : blk_rows_gemv_max_for(wf, n_in, n_out);
}
// ONE MEASURED TENSOR, from the tune file. `gemv_max` is the rows up to which the scalar rows
// GEMV won when the tuner timed both kernels on this triple. Re-seating the same triple replaces
// it, so a reload does not grow the table. Returns false when the table is full (the model has
// more distinct (format, shape) triples than BLK_ROWS_SEAT_MAX -- 67 on the widest file seen,
// a UD mixed quant of the 27B) or the value is out of range; the caller logs it and that tensor
// keeps its format's crossing.
extern "C" bool imparo_metal_set_blk_rows_seat(uint32_t wkind, uint32_t n_in, uint32_t n_out,
                                               uint32_t gemv_max) {
    const uint32_t wf = wfmt_for(wkind);
    if (wf == 0u || n_in == 0u || n_out == 0u || gemv_max > BLK_ROWS_MAX_HOST) { return false; }
    uint32_t s = blk_rows_seat_slot(wf, n_in, n_out);
    for (uint32_t probe = 0u; probe < BLK_ROWS_SEAT_SLOTS; ++probe) {
        BlkRowsSeat & e = g_blk_rows_seats[s];
        if (e.used && e.wf == wf && e.n_in == n_in && e.n_out == n_out) {
            e.gemv_max = gemv_max;                       // replace, do not add
            return true;
        }
        if (!e.used) {
            if (g_blk_rows_seat_count >= BLK_ROWS_SEAT_MAX) { return false; }
            e = BlkRowsSeat{wf, n_in, n_out, gemv_max, true};
            g_blk_rows_seat_count += 1u;
            return true;
        }
        s = (s + 1u) & (BLK_ROWS_SEAT_SLOTS - 1u);
    }
    return false;
}
extern "C" void imparo_metal_clear_blk_rows_seats(void) {
    for (uint32_t i = 0; i < BLK_ROWS_SEAT_SLOTS; ++i) { g_blk_rows_seats[i].used = false; }
    g_blk_rows_seat_count = 0u;
}
extern "C" uint32_t imparo_metal_blk_rows_seat_count(void) { return g_blk_rows_seat_count; }
extern "C" void imparo_metal_set_blk_rows_mma_max(uint32_t n) {
    g_blk_rows_mma_max = std::min(n, BLK_MMA_MAX_HOST);
}
extern "C" uint32_t imparo_metal_blk_rows_mma_max(void) { return g_blk_rows_mma_max; }
extern "C" uint64_t imparo_metal_blk_rows_dispatches(void) { return g_blk_rows_dispatches; }
extern "C" void imparo_metal_set_q4_rows_gemv_max(uint32_t n) { g_q4_rows_gemv_max = n; }
extern "C" uint32_t imparo_metal_q4_rows_gemv_max(void) { return g_q4_rows_gemv_max; }
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
// THE REGISTER-SOFTMAX FORM SERVES the FA route wherever it is built (slot 0, head dims up
// to 128). At 2048 queries against 16384 keys: LFM2.5 (hd 64) 62.9 -> 51.7 ms, Qwen3-8B
// (hd 128) 241.7 -> 101.8 ms (M3 Pro, imparo-metalbench IMPARO_BENCH_ATTN, 2026-09-24).
// IMPARO_ATTN_FARS=0 keeps the 8-query op for A/Bs.
static bool fars_route_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_ATTN_FARS"); v = (e && e[0] == '0') ? 0 : 1; }
    return v != 0;
}
// The 8-query FA op serves slot 0 at head dims up to 128 (its register budget; see the kernel)
// when the register-softmax form does not -- the condition attn_fa_nsg applies under.
extern "C" uint32_t imparo_metal_fa_has_hd(uint32_t hd) {
    return (hd != 0u && g_qcomb_hds[0] == hd && hd <= 128u && !fars_route_on()) ? 1u : 0u;
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
extern "C" void imparo_metal_set_verify_attention(uint32_t tgs, uint32_t heads) {
    g_verify_attn_tgs = tgs;
    g_verify_attn_heads = heads;
}
extern "C" void imparo_metal_set_verify_split_on(uint32_t on) { g_verify_split_on = on; }
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
    // THE SEAT, and the only stored tile. Every other tile a dispatch can take is derived
    // from this one by `q8_tile`, so this is rankable on its own: what a candidate changes
    // is the row tile, the simdgroup count and the K chunk, and the token width it starts
    // from.
    if (shape < ST_GEMM_CANDIDATES) { g_st_gemm_shape = shape; }
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
extern "C" void imparo_metal_set_st_gemm_narrow_max(uint32_t n) { g_st_gemm_narrow_max = n; }
extern "C" uint32_t imparo_metal_st_gemm_narrow_max(void) { return g_st_gemm_narrow_max; }
extern "C" void imparo_metal_set_q8_full_tiles(uint32_t on) { g_q8_full_tiles = on; }
extern "C" uint32_t imparo_metal_q8_full_tiles(void) { return g_q8_full_tiles; }
extern "C" void imparo_metal_set_q8_all(uint32_t on) { g_q8_all = on; }
extern "C" void imparo_metal_set_q8_grid_token_x(uint32_t on) { g_q8_grid_token_x = on; }
extern "C" void imparo_metal_set_q8_skip(uint32_t bits) { g_q8_skip = bits; }
extern "C" void imparo_metal_set_q8_typed_scale(uint32_t on) { g_q8_typed_scale = on; }
extern "C" void imparo_metal_set_q8_dev_a(uint32_t on) { g_q8_dev_a = on; }
extern "C" void imparo_metal_set_q8_clamp_edge(uint32_t on) { g_q8_clamp_edge = on; }
extern "C" void imparo_metal_set_q8_mma_fence(uint32_t on) { g_q8_mma_fence = on; }
extern "C" void imparo_metal_set_q8_pad_skip(uint32_t on) { g_q8_pad_skip = on; }
extern "C" void imparo_metal_set_rt_mma_fence(uint32_t on) { g_rt_mma_fence = on; }
extern "C" void imparo_metal_set_attn_skip(uint32_t bits) { g_attn_skip = bits; }
extern "C" void imparo_metal_set_skip_mm_nout(uint32_t n) { g_skip_mm_nout = n; }
extern "C" void imparo_metal_set_dup_mm_nout(uint32_t n) { g_dup_mm_nout = n; }
extern "C" void imparo_metal_set_dup_attn_combine(uint32_t on) { g_dup_attn_combine = on != 0u; }
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

// Same timing as the head dims: before init, because the value is baked into the library.
extern "C" void imparo_metal_set_recurrent_dims(uint32_t key_dim, uint32_t value_dim) {
    g_delta_kd = key_dim;
    g_delta_vd = value_dim;
}
extern "C" uint32_t imparo_metal_supports_gated_delta(void) {
    return g.p_delta_net != nil ? 1u : 0u;
}
// Whether the delta pipeline applies the gated-RMS epilogue ITSELF.
//
// THE WIDTH RULE, and it is a bit-identity rule, not a capacity one. The fused epilogue
// reproduces imparo_rms_norm's ONE-SIMDGROUP branch: the host derives that kernel's
// thread count as ceil(width/4) rounded up to 32 and floored at 32, so a row of 128 or
// fewer runs exactly 32 threads and reduces with a single `simd_sum`. Above 128 it runs
// two or more simdgroups and finishes through `rms_finish`, a DIFFERENT summation order
// and therefore different bits, which the fused form does not reproduce. A width that is
// not a multiple of 4 takes the norm's scalar path, also a different order.
//
// Answered from the dims the backend already holds (set_recurrent_dims runs before init),
// so the caller's "did it fuse" and the dispatch's "will I fuse" read the same source.
extern "C" uint32_t imparo_metal_delta_net_fuses_epilogue(void) {
    return (g.p_delta_net != nil && g_delta_vd >= 4u && g_delta_vd <= 128u
            && (g_delta_vd % 4u) == 0u) ? 1u : 0u;
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

// SECONDS per category, and the counters reset on read. A category's figure is the summed
// duration of ITS OWN dispatches, so categories that ran concurrently each report their
// full time and the column adds up to more than the wall it ran in -- that is the machine
// reporting overlap, not an error.
// The window's per-category seconds and calls, with the WEIGHT BYTES each one read, and the
// counters reset on read. A category that declares no bytes reports 0, which the printer
// shows as no bandwidth rather than as zero bandwidth.
extern "C" void imparo_metal_prof_cats_b(double * secs, uint64_t * calls, uint64_t * bytes,
                                         uint32_t * n) {
    *n = PC_N;
    for (uint32_t i = 0; i < PC_N; ++i) {
        secs[i] = g_prof_cat_s[i]; calls[i] = g_prof_cat_calls[i];
        if (bytes != nullptr) { bytes[i] = g_prof_cat_bytes[i]; }
        g_prof_cat_s[i] = 0.0; g_prof_cat_calls[i] = 0; g_prof_cat_bytes[i] = 0;
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
        // The recurrent head widths, or 32 when the model has no recurrent mixer: the
        // kernel must still compile, and its pipeline is simply not built.
        macros[@"IMPARO_BLK_GEMV_NR"] =
            [NSNumber numberWithUnsignedInt:blk_gemv_nr()];
        // PROBE: compile the decode GEMV with a grid-stride loop so the host can dispatch
        // a threadgroup count of its choosing (IMPARO_BLK_GEMV_TGS). Off by default -- a
        // loop costs registers even when it runs once.
        macros[@"IMPARO_BLK_GEMV_STRIDE"] =
            [NSNumber numberWithUnsignedInt:getenv("IMPARO_BLK_GEMV_STRIDE") != nullptr ? 1u : 0u];
        macros[@"IMPARO_DELTA_KD"] =
            [NSNumber numberWithUnsignedInt:g_delta_kd != 0u ? g_delta_kd : 32u];
        macros[@"IMPARO_DELTA_VD"] =
            [NSNumber numberWithUnsignedInt:g_delta_vd != 0u ? g_delta_vd : 32u];
        macros[@"IMPARO_DELTA_TREE_DEPTH"] =
            [NSNumber numberWithUnsignedInt:g_delta_tree_depth];
        // TOKENS THE DELTA RULE STAGES PER GROUP. The kernel's comment carries the
        // reason; the value is a preprocessor define consumed when the library is
        // compiled at init, so an env override is what makes re-checking it on another
        // machine a re-run rather than an edit. 1 is the per-token form it replaced.
        // Bounded by the threadgroup memory it costs: DSTAGE * (2*DKD + DVD + 2) floats.
        uint32_t delta_stage = 8u;
        if (const char * ds = getenv("IMPARO_DELTA_STAGE")) {
            const uint32_t v = (uint32_t)atoi(ds);
            const uint32_t kd = g_delta_kd != 0u ? g_delta_kd : 32u;
            const uint32_t vd = g_delta_vd != 0u ? g_delta_vd : 32u;
            if (v >= 1u && v <= 32u && v * (2u * kd + vd + 2u) * 4u <= tg_budget) {
                delta_stage = v;
            }
        }
        macros[@"IMPARO_DELTA_STAGE"] =
            [NSNumber numberWithUnsignedInt:delta_stage];
        // Simdgroups per delta-rule threadgroup: 4, 8, 16 or 32 (the staging assigns four
        // roles per token slot, so at least four). Both sides read this one value.
        if (const char * dsg = getenv("IMPARO_DELTA_SGS")) {
            const uint32_t v = (uint32_t)atoi(dsg);
            if (v == 4u || v == 8u || v == 16u || v == 32u) { g_delta_sgs = v; }
        }
        macros[@"IMPARO_DELTA_SGS"] = [NSNumber numberWithUnsignedInt:g_delta_sgs];
        // The row pipeline in the delta rule; 0 is the row-at-a-time form, for the A/B.
        uint32_t delta_pipe = 1u;
        if (const char * dp = getenv("IMPARO_DELTA_PIPE")) {
            delta_pipe = atoi(dp) != 0 ? 1u : 0u;
        }
        macros[@"IMPARO_DELTA_PIPE"] =
            [NSNumber numberWithUnsignedInt:delta_pipe];
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
        id<MTLLibrary> lib = g.lib = [g.device newLibraryWithSource:[NSString stringWithUTF8String:kSource]
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
            // Only the ON case needs a constant: the shader defaults to no fences, so an
            // unset build specializes exactly what `make` has always specialized.
            if (g_rt_mma_fence != 0u) {
                NSLog(@"imparo metal: rt_gemm MMA scheduling fences ON");
            }
            auto make_rt = [&](NSString * nm) -> id<MTLComputePipelineState> {
                MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
                stamp_epi_act(cv);
                if (g_rt_mma_fence != 0u) {
                    const bool rf = true;
                    [cv setConstantValue:&rf type:MTLDataTypeBool atIndex:16];
                }
                return rt_register(make_with(lib, nm, cv), nm, cv);
            };
            // rt_gemm<Q8> and the narrow tiles: the constants `make` stamps, recorded too.
            auto make_rt_plain = [&](NSString * nm) -> id<MTLComputePipelineState> {
                MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
                stamp_epi_act(cv);
                return rt_register(make_with(lib, nm, cv), nm, cv);
            };
            for (uint32_t i = 0; i < RT_CANDIDATES; ++i) {
                if (!g_rt_all && i != g_rt_shape) { continue; }
                NSString * nm = [NSString stringWithFormat:@"imparo_rt_%u", i];
                g.p_rt[i] = make_rt(nm);
                NSString * nmh = [NSString stringWithFormat:@"imparo_rt_%u_h", i];
                g.p_rt_h[i] = make_rt(nmh);
                g.p_rt_gh[i] = make_rt([NSString stringWithFormat:@"imparo_rt_%u_gh", i]);
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
                g.p_rt8[i]   = make_rt_plain([NSString stringWithFormat:@"imparo_rt8_%u", i]);
                g.p_rt8_h[i] = make_rt_plain([NSString stringWithFormat:@"imparo_rt8_%u_h", i]);
                g.p_rt8_gh[i] = make_rt_plain([NSString stringWithFormat:@"imparo_rt8_%u_gh", i]);
            }
            g.p_rt_nb8    = make_rt_plain(@"imparo_rt_nb8");
            g.p_rt_nb8_h  = make_rt_plain(@"imparo_rt_nb8_h");
            g.p_rt_nb8b   = make_rt_plain(@"imparo_rt_nb8b");
            g.p_rt_nb8b_h = make_rt_plain(@"imparo_rt_nb8b_h");
            g.p_rt_nb8_gh  = make_rt_plain(@"imparo_rt_nb8_gh");
            g.p_rt_nb8b_gh = make_rt_plain(@"imparo_rt_nb8b_gh");
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
        g.p_f32nmm = make(lib, @"imparo_f32_narrow_mm");
        g.p_f32gemv = make(lib, @"imparo_f32_gemv_ksplit");
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
            const bool q8_deep = ST_GEMM_SHAPES[g_st_gemm_shape][3] > Q8_BLOCK_ELEMENTS;
            for (uint32_t i = 0; i < ST_GEMM_CANDIDATES; ++i) {
                if (!g_q8_all && !(q8_deep && i == 3u) && !q8_tile_reaches(i)) {
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
                if (g_q8_clamp_edge == 0u) {
                    // Only the OFF case needs a constant: the shader defaults to on.
                    if (cvg == cv_empty) { cvg = [MTLFunctionConstantValues new]; }
                    const bool ce = false;
                    [cvg setConstantValue:&ce type:MTLDataTypeBool atIndex:13];
                }
                if (g_q8_mma_fence == 0u) {
                    // Only the OFF case needs a constant: the shader defaults to on.
                    if (cvg == cv_empty) { cvg = [MTLFunctionConstantValues new]; }
                    const bool mf = false;
                    [cvg setConstantValue:&mf type:MTLDataTypeBool atIndex:14];
                }
                if (g_q8_pad_skip != 0u) {
                    // Only the ON case needs a constant: the shader defaults to off.
                    if (cvg == cv_empty) { cvg = [MTLFunctionConstantValues new]; }
                    const bool ps = true;
                    [cvg setConstantValue:&ps type:MTLDataTypeBool atIndex:27];
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
                g.p_attn_dec_stream_g256_q[0] = make_cv(@"imparo_attention_decode_stream_dk256_g2", cv_q);
                g.p_attn_dec_stream_g256_q[1] = make_cv(@"imparo_attention_decode_stream_dk256_g3", cv_q);
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
        g.p_conv[0]         = make(lib, @"imparo_causal_conv_gated");
        g.p_conv[1]         = make(lib, @"imparo_causal_conv_plain");
        g.p_conv_state[0]   = make(lib, @"imparo_causal_conv_state_gated");
        g.p_conv_state[1]   = make(lib, @"imparo_causal_conv_state_plain");
        g.p_shortconv_step = make(lib, @"imparo_shortconv_step");
        g.p_shortconv_step_plain = make(lib, @"imparo_shortconv_step_plain");
        // Built only when the model declared recurrent dims: the kernel's register array
        // is sized by them, so a zero build would be a pipeline nothing can dispatch.
        if (g_delta_kd != 0u && g_delta_vd != 0u) {
            g.p_delta_net   = make(lib, @"imparo_delta_net");
            g.p_delta_net_tree = make(lib, @"imparo_delta_net_tree");
        }
        g.p_mul_sigmoid     = make(lib, @"imparo_mul_sigmoid");
        g.p_copy_strided    = make(lib, @"imparo_copy_strided");
        g.p_scatter_strided = make(lib, @"imparo_scatter_strided");
        g.p_add    = make(lib, @"imparo_add");
        g.p_mul    = make(lib, @"imparo_mul");
        g.p_scale  = make(lib, @"imparo_scale");
        g.p_addscale  = make(lib, @"imparo_add_scale");
        g.p_copy   = make(lib, @"imparo_copy");
        g.p_softcap= make(lib, @"imparo_softcap");
        g.p_argmax = make(lib, @"imparo_argmax");
        g.p_argmax_feed = make(lib, @"imparo_argmax_feed");
        g.p_top_k_chunks = make(lib, @"imparo_top_k_chunks");
        g.p_top_k8_one = make(lib, @"imparo_top_k8_one");
        g.p_top_k_merge = make(lib, @"imparo_top_k_merge");
        g.p_top_k8_chunks = make(lib, @"imparo_top_k8_chunks");
        g.p_top_k8_merge = make(lib, @"imparo_top_k8_merge");
        g.p_moe_gate = make(lib, @"imparo_moe_gate");
        g.p_moe_plan = make(lib, @"imparo_moe_plan");
        g.p_moe_route = make(lib, @"imparo_moe_route");
        g.p_moe_combine = make(lib, @"imparo_moe_combine");
        g.p_moe_combine_norm = make(lib, @"imparo_moe_combine_add_rms_norm");
        g.p_logistic_blocks = make(lib, @"imparo_logistic_blocks");
        g.p_logistic_finish = make(lib, @"imparo_logistic_finish");
        g.p_ple    = make(lib, @"imparo_ple_combine");
        g.p_ple_gather = make(lib, @"imparo_ple_gather_combine");
        g.p_repack_q8_tm = make(lib, @"imparo_repack_q8_0_tm");
        g.p_repack_tm    = make(lib, @"imparo_repack_tm");
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
            g.p_attn_dec_stream_g256[0] = make_cv(@"imparo_attention_decode_stream_dk256_g2", cv_empty);
            g.p_attn_dec_stream_g256[1] = make_cv(@"imparo_attention_decode_stream_dk256_g3", cv_empty);
        }
        // THE CORE COUNT IS A DEVICE FACT, read whatever the mega level. It was read only inside
        // the block below, so IMPARO_MEGA_FFN=0 also left it 0 -- and the dispatch path reads it
        // too (the router GEMV's K-split and the staged norm size their grids from it): the
        // "mega off" arm silently ran the one-simdgroup-per-row router, and every A/B that took
        // that arm as the dispatch path measured the router along with the mega entry.
        g_gpu_cores = gpu_core_count();
        if (mega_blocks_wanted()) {
            // Same lane geometry as the decode GEMV it replaces (bit-identical rows).
            MTLFunctionConstantValues * cvm = [MTLFunctionConstantValues new];
            const uint32_t lanes = g_lanes, nr0 = 1u;
            [cvm setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
            [cvm setConstantValue:&nr0   type:MTLDataTypeUInt atIndex:2];
            const bool dbg = mega_dbg();
            [cvm setConstantValue:&dbg type:MTLDataTypeBool atIndex:18];
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
            // The qwen35 layer block. NOTHING about the weight format is stamped here: this
            // architecture's file assigns a quant per TENSOR, so the format is an entry word
            // and the row brick switches on it (task #165). The activation (fc 11) still is,
            // like every epilogue pipeline.
            {
                stamp_epi_act(cvm);
                const bool deep = true, plain = false;
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    g.p_mega_q35[slot] = (hd != 0u && hd % 32u == 0u)
                        ? make_cv([NSString stringWithFormat:@"imparo_mega_qwen35_layer_s%u", slot], cvm) : nil;
                }
                [cvm setConstantValue:&deep type:MTLDataTypeBool atIndex:19];
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    g.p_mega_q35_deep[slot] = (hd != 0u && hd % 32u == 0u)
                        ? make_cv([NSString stringWithFormat:@"imparo_mega_qwen35_layer_s%u", slot], cvm) : nil;
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
                        g.p_mega_q35_prog[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_qwen35_layer_s%u", slot], cvp);
                    }
                    if (g_mega_kq_ty != 1u || g_mega_vq_ty != 1u) {
                        [cvp setConstantValue:&g_mega_kq_ty type:MTLDataTypeUInt atIndex:3];
                        [cvp setConstantValue:&g_mega_vq_ty type:MTLDataTypeUInt atIndex:4];
                        for (uint32_t slot = 0; slot < 2u; ++slot) {
                            const uint32_t hd = g_qcomb_hds[slot];
                            if (hd == 0u || hd % 32u != 0u) { continue; }
                            g.p_mega_q35_prog_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_qwen35_layer_s%u", slot], cvp);
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
                        g.p_mega_q35_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_qwen35_layer_s%u", slot], cvq);
                    }
                    [cvq setConstantValue:&deep type:MTLDataTypeBool atIndex:19];
                    for (uint32_t slot = 0; slot < 2u; ++slot) {
                        const uint32_t hd = g_qcomb_hds[slot];
                        if (hd == 0u || hd % 32u != 0u) { continue; }
                        g.p_mega_q35_deep_q[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_qwen35_layer_s%u", slot], cvq);
                    }
                }
            }
            // The lfm2moe routed feed-forward block: the format per tensor is an entry word, the
            // activation (fc 11) is stamped like every epilogue pipeline.
            {
                stamp_epi_act(cvm);
                MTLFunctionConstantValues * cvl = [cvm copy];
                for (uint32_t slot = 0; slot < 2u; ++slot) {
                    const uint32_t hd = g_qcomb_hds[slot];
                    g.p_mega_l2m[slot] = (hd != 0u && hd % 32u == 0u)
                        ? make_cv([NSString stringWithFormat:@"imparo_mega_lfm2moe_layer_s%u", slot], cvl) : nil;
                }
                if (mega_program_build()) {
                    MTLFunctionConstantValues * cvp = [cvl copy];
                    const bool on = true;
                    [cvp setConstantValue:&on type:MTLDataTypeBool atIndex:19];
                    [cvp setConstantValue:&on type:MTLDataTypeBool atIndex:20];
                    for (uint32_t slot = 0; slot < 2u; ++slot) {
                        const uint32_t hd = g_qcomb_hds[slot];
                        if (hd == 0u || hd % 32u != 0u) { continue; }
                        g.p_mega_l2m_prog[slot] = make_cv([NSString stringWithFormat:@"imparo_mega_lfm2moe_layer_s%u", slot], cvp);
                    }
                }
            }
            // Keep the historical grid heuristic as an outer bound for explicit experiments;
            // ordinary runs additionally use the conservative one-group-per-core ceiling.
            g_tgmem_limit = (uint32_t)[g.device maxThreadgroupMemoryLength];
            // The limit must cover EVERY pipeline the block can dispatch: both head-dim slots of
            // the layer kernel (E4B's global layers run slot 1). A slot's
            // maxTotalThreadsPerThreadgroup bounds a legal group, not groups per core. A
            // lower value also tightens the historical grid heuristic; the conservative
            // grid policy below does not infer two-group admission from a 1024-thread value.
            // ONE VERDICT PER FAMILY, never one over all of them: see the comment at
            // g_mega_threads_limit. `tmax[arch]` is the min over that family's own pipelines.
            uint32_t tmax[MEGA_ARCH_COUNT] = { 0u, 0u, 0u, 0u };
            {
                struct { uint32_t arch; id<MTLComputePipelineState> pipe; const char * name; } ps[8] = {
                    { MEGA_ARCH_GEMMA4, g.p_mega_layer[0],  "layer_s0" },
                    { MEGA_ARCH_GEMMA4, g.p_mega_layer[1],  "layer_s1" },
                    { MEGA_ARCH_LFM2,   g.p_mega_lfm2[0],   "lfm2_s0" },
                    { MEGA_ARCH_LFM2,   g.p_mega_lfm2[1],   "lfm2_s1" },
                    { MEGA_ARCH_QWEN35, g.p_mega_q35[0],    "q35_s0" },
                    { MEGA_ARCH_QWEN35, g.p_mega_q35[1],    "q35_s1" },
                    { MEGA_ARCH_LFM2MOE, g.p_mega_l2m[0],   "l2m_s0" },
                    { MEGA_ARCH_LFM2MOE, g.p_mega_l2m[1],   "l2m_s1" },
                };
                NSMutableString * rep = [NSMutableString new];
                for (uint32_t i = 0; i < 8u; ++i) {
                    if (ps[i].pipe == nil) { [rep appendFormat:@" %s=nil", ps[i].name]; continue; }
                    const uint32_t t = (uint32_t)[ps[i].pipe maxTotalThreadsPerThreadgroup];
                    uint32_t & fam = tmax[ps[i].arch];
                    fam = fam ? std::min(fam, t) : t;
                    [rep appendFormat:@" %s=%u(tgmem %lu)", ps[i].name, t, (unsigned long)[ps[i].pipe staticThreadgroupMemoryLength]];
                }
                NSLog(@"imparo metal: mega pipelines max_threads/tg:%@ -> per-family limit gemma4=%u lfm2=%u qwen35=%u lfm2moe=%u",
                      rep, tmax[MEGA_ARCH_GEMMA4], tmax[MEGA_ARCH_LFM2], tmax[MEGA_ARCH_QWEN35], tmax[MEGA_ARCH_LFM2MOE]);
            }
            // The deep variants run one threadgroup per core: each must admit the block's
            // threadgroup at all (its own max-threads verdict), else that slot's deep body is
            // off and the layer takes the dispatch path past the vec regime.
            {
                struct { uint32_t arch; __strong id<MTLComputePipelineState> * pipe; const char * name; } dp[12] = {
                    { MEGA_ARCH_GEMMA4, &g.p_mega_layer_deep[0],   "layer_deep_s0" },
                    { MEGA_ARCH_GEMMA4, &g.p_mega_layer_deep[1],   "layer_deep_s1" },
                    { MEGA_ARCH_LFM2,   &g.p_mega_lfm2_deep[0],    "lfm2_deep_s0" },
                    { MEGA_ARCH_LFM2,   &g.p_mega_lfm2_deep[1],    "lfm2_deep_s1" },
                    { MEGA_ARCH_GEMMA4, &g.p_mega_layer_deep_q[0], "layer_deep_q_s0" },
                    { MEGA_ARCH_GEMMA4, &g.p_mega_layer_deep_q[1], "layer_deep_q_s1" },
                    { MEGA_ARCH_LFM2,   &g.p_mega_lfm2_deep_q[0],  "lfm2_deep_q_s0" },
                    { MEGA_ARCH_LFM2,   &g.p_mega_lfm2_deep_q[1],  "lfm2_deep_q_s1" },
                    { MEGA_ARCH_QWEN35, &g.p_mega_q35_deep[0],     "q35_deep_s0" },
                    { MEGA_ARCH_QWEN35, &g.p_mega_q35_deep[1],     "q35_deep_s1" },
                    { MEGA_ARCH_QWEN35, &g.p_mega_q35_deep_q[0],   "q35_deep_q_s0" },
                    { MEGA_ARCH_QWEN35, &g.p_mega_q35_deep_q[1],   "q35_deep_q_s1" },
                };
                NSMutableString * rep = [NSMutableString new];
                for (uint32_t i = 0; i < 12u; ++i) {
                    if (*dp[i].pipe == nil) { [rep appendFormat:@" %s=nil", dp[i].name]; continue; }
                    const uint32_t t = (uint32_t)[*dp[i].pipe maxTotalThreadsPerThreadgroup];
                    const uint32_t fam = tmax[dp[i].arch];
                    [rep appendFormat:@" %s=%u", dp[i].name, t];
                    if (t < fam) { [rep appendFormat:@"(below its family's plain verdict %u: deep body OFF for this slot)", fam]; *dp[i].pipe = nil; }
                }
                NSLog(@"imparo metal: mega deep pipelines max_threads/tg:%@", rep);
            }
            // Unknown device: one group, or the ordinary dispatch path if the layer needs more.
            const uint32_t cores_for_limit = mega_core_seat();
            for (uint32_t a = 0; a < MEGA_ARCH_COUNT; ++a) {
                g_mega_threads_limit[a] = cores_for_limit * tmax[a];
                g_mega_nsg_limit[a] = tmax[a] / 32u;
                g_mega_nsg[a] = std::max(1u, std::min(16u, tmax[a] / 64u));   // half the largest legal group
                // THE DEFAULT VALUE: one per core, what an untuned host runs. A stored tune
                // value or IMPARO_MEGA_TGS moves it; the LIMIT it must respect is measured by
                // the tuner, not here (task #203, docs/tuner-design.md).
                g_mega_tgs[a][0] = g_mega_tgs[a][1] = cores_for_limit;
            }
            if (const char * e = getenv("IMPARO_MEGA_NSG")) { g_mega_nsg_req = (uint32_t)strtoul(e, nullptr, 10); }
            if (const char * e = getenv("IMPARO_MEGA_NSG_EXACT")) { g_mega_nsg_exact = strtoul(e, nullptr, 10) != 0ul; }
            if (const char * e = getenv("IMPARO_MEGA_TGS")) {   // both slots: an experiment, not a tuned value
                g_mega_tgs_req[0] = g_mega_tgs_req[1] = (uint32_t)strtoul(e, nullptr, 10);
                g_mega_tgs_explicit = g_mega_tgs_req[0] != 0u;
            }
            if (mega_clamp()) {
                NSLog(@"imparo metal: mega requested seat %u/%ux%u clamped by the pipeline's width budget; actual per-family grids follow", g_mega_tgs_req[0], g_mega_tgs_req[1], g_mega_nsg_req);
            }
            if (std::max(g_mega_tgs_req[0], g_mega_tgs_req[1]) > cores_for_limit) {
                NSLog(@"imparo metal: mega grid %u/%u groups over %u cores from %s; admission is what the tuner measured for it, or an experiment",
                      g_mega_tgs_req[0], g_mega_tgs_req[1], cores_for_limit, g_mega_tgs_explicit ? "IMPARO_MEGA_TGS" : "the stored tune values");
            }
            // Counters only until the first block dispatch sizes the partial scratch.
            g_mega_scratch_words = 0;
            mega_scratch_ensure(0u);
            if (mega_dbg()) { NSLog(@"imparo metal: mega DEBUG records on"); }
            NSLog(@"imparo metal: mega blocks %s (gpu_cores=%u lanes=%u; per family (slot0/slot1 x nsg) gemma4=%u/%ux%u lfm2=%u/%ux%u qwen35=%u/%ux%u)",
                  (g.p_mega_ffn_ple != nil && g.mega_sync != nil) ? "built" : "UNAVAILABLE (IMPARO_MEGA_FFN=0, no head dim divisible by 32, or a pipeline or sync buffer was not created)",
                  g_gpu_cores, g_lanes,
                  g_mega_tgs[MEGA_ARCH_GEMMA4][0], g_mega_tgs[MEGA_ARCH_GEMMA4][1], g_mega_nsg[MEGA_ARCH_GEMMA4],
                  g_mega_tgs[MEGA_ARCH_LFM2][0], g_mega_tgs[MEGA_ARCH_LFM2][1], g_mega_nsg[MEGA_ARCH_LFM2],
                  g_mega_tgs[MEGA_ARCH_QWEN35][0], g_mega_tgs[MEGA_ARCH_QWEN35][1], g_mega_nsg[MEGA_ARCH_QWEN35]);
        }
        {
            // IMPARO_ATTN_COMB_STAGE=0 builds the combine's pre-staging form for an A/B.
            MTLFunctionConstantValues * cv_cb = cv_empty;
            const char * cs = getenv("IMPARO_ATTN_COMB_STAGE");
            if (cs != nullptr) {
                const uint32_t arm = (uint32_t)strtoul(cs, NULL, 10);
                if (arm <= 2u) {
                    cv_cb = [MTLFunctionConstantValues new];
                    [cv_cb setConstantValue:&arm type:MTLDataTypeUInt atIndex:33];
                }
            }
            if (const char * sp = getenv("IMPARO_ATTN_COMB_SPD")) {
                const uint32_t spd = (uint32_t)strtoul(sp, NULL, 10);
                if (spd == 2u || spd == 4u || spd == 8u) {
                    if (cv_cb == cv_empty) { cv_cb = [MTLFunctionConstantValues new]; }
                    [cv_cb setConstantValue:&spd type:MTLDataTypeUInt atIndex:35];
                    g_attn_comb_spd = spd;
                }
            }
            g.p_attn_dec_combine = make_cv(@"imparo_attention_decode_combine", cv_cb);
        }
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
                MTLFunctionConstantValues * cv_fars = [MTLFunctionConstantValues new];
                if (!g_attn_live_mask) {
                    const bool lm_off = false;
                    [cv_fars setConstantValue:&lm_off type:MTLDataTypeBool atIndex:10];
                }
                g.p_fars = g_qcomb_hds[0] <= 128u
                    ? make_cv(@"imparo_attention_prefill_fars_s0", cv_fars) : nil;

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
                g.p_attn_dec_stream_g256_id[0] = make_cv(@"imparo_attention_decode_stream_dk256_g2", cv_id);
                g.p_attn_dec_stream_g256_id[1] = make_cv(@"imparo_attention_decode_stream_dk256_g3", cv_id);
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
                    g.p_attn_dec_stream_g256_q_id[0] = make_cv(@"imparo_attention_decode_stream_dk256_g2", cv_qid);
                    g.p_attn_dec_stream_g256_q_id[1] = make_cv(@"imparo_attention_decode_stream_dk256_g3", cv_qid);
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
            {g.p_f32mm, "f32_matmat"}, {g.p_f32gemv, "f32_gemv_ksplit"},
            {g.p_f32nmm, "f32_narrow_mm"},
            {g.p_q4row, "q4_row"},
            {g.p_kvstore, "kv_store"}, {g.p_actmul, "act_mul"},
            // A nil pipeline makes its op a SILENT no-op: the conv check read back an
            // output of exactly zeros and a state exactly as written, which is what
            // "the dispatch never happened" looks like from outside.
            {g.p_conv[0], "causal_conv_gated"}, {g.p_conv_state[0], "causal_conv_state_gated"},
            {g.p_conv[1], "causal_conv_plain"}, {g.p_conv_state[1], "causal_conv_state_plain"},
            {g.p_mul_sigmoid, "mul_sigmoid"}, {g.p_copy_strided, "copy_strided"},
            {g.p_scatter_strided, "scatter_strided"},
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
        mem_pressure_watch();
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

// BUFFERS THAT ARE STILL AS ALLOCATED. `newBufferWithLength:` hands back zeroed memory,
// so zeroing one that nothing has written since is a no-op that costs a first touch of
// every page -- 2749 ms for Qwen3.8-27B's ~700 MiB of recurrent state, on the first
// request. The bit is set when the buffer is created and cleared the moment anything
// COULD have written it: a host write, or any command buffer at all (a kernel's output
// binding is not visible from here, so every submission clears the whole mask).
static uint64_t g_buf_fresh = 0ull;
static inline void buf_fresh_set(uint32_t id)   { if (id < 64u) { g_buf_fresh |= (1ull << id); } }
static inline void buf_fresh_clear(uint32_t id) { if (id < 64u) { g_buf_fresh &= ~(1ull << id); } }
static inline bool buf_is_fresh(uint32_t id)    { return id < 64u && (g_buf_fresh >> id & 1ull) != 0ull; }

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
    buf_fresh_set(id);
    g.sizes[id] = bytes;
    g.in_arena[id] = 0;
    g.buf_off[id] = 0;
    return g.bufs[id] == nil ? 2 : 0;
}

extern "C" uint64_t imparo_metal_allocated_bytes(void) {
    if (g.device == nil) { return 0ull; }
    return (uint64_t) [g.device currentAllocatedSize];
}

// ============================================================================================
// KV STORAGE: one address-space reservation per layer side, and the kernels bind a no-copy
// VIEW over its used part. Design: docs/memory-tiers-and-fit.md section 12.
//
// A buffer the GPU can reach is committed WHOLE while it is resident (measured on macOS 15:
// newBufferWithLength commits all of it at its first use, a no-copy mapping of anonymous memory
// is wired whole), so the only way to hold just the blocks in use is to make the resident
// object only as big as they are:
//
//   grow     a bigger view over the SAME memory -- every row already written stays where it
//            is, nothing is copied, and only the new tail is wired
//   release  a smaller view; the tail's pages go back once no view covers them
//
// Making a view resident costs ~20-25 ms per GiB of view, so the view the next steps will need
// is prepared on a background thread (`imparo_metal_kv_prefetch`) and adopted between two
// steps. Adoption never happens inside a region: every caller runs at a forward's entry.
//
// The views live in THEIR OWN residency set. A commit of the weights' set clears its hold and
// the next region re-wires every weight page (4.9 s measured with one other 15 GB process
// alive), which KV growth must never cause.
// ============================================================================================
static uint64_t page_round(uint64_t n);
struct KvRes {
    uint8_t * va = nullptr;   // anonymous mapping: a page exists once something writes it
    uint64_t reserved = 0;    // bytes of address space
    uint64_t committed = 0;   // bytes the current view covers
};
static std::vector<KvRes> g_kvres_k, g_kvres_v;
// Bumped whenever the reservations are replaced; a preparation made for older ones is dropped.
static uint64_t g_kv_gen = 0;
static id<MTLResidencySet> g_kv_set = nil;
static bool g_kv_set_dirty = false, g_kv_set_attached = false, g_kv_set_held = false;
// Regions begun, and regions known complete. Buffers on one queue complete in commit order, so
// region N complete means every region before it is complete too.
static uint64_t g_region_begun = 0, g_region_done = 0;
// Views that a command buffer committed before a swap may still address, with the staging set
// that made their replacements resident: kept until region `after` has completed. A release's
// tail is advised free at the same moment, when no view in use covers it.
struct KvRetired {
    uint64_t after = 0;
    std::vector<id<MTLBuffer>> views;
    id<MTLResidencySet> stage = nil;
    std::vector<std::pair<uint8_t *, uint64_t>> tails;   // [base, base + len) to give back
};
static std::vector<KvRetired> g_kv_retired;
// Views prepared off the step's path: made resident on a background thread, adopted by the
// next forward entry that wants them.
struct KvPrep {
    uint64_t gen = 0;
    std::vector<uint64_t> k, v;                 // bytes each side covers after adoption
    std::vector<id<MTLBuffer>> kb, vb;          // nil where the side keeps its current view
    id<MTLResidencySet> stage = nil;
    double ms = 0.0;
};
static std::mutex g_kvprep_mu;
static std::thread g_kvprep_th;
static bool g_kvprep_busy = false;              // under g_kvprep_mu
static bool g_kvprep_ready = false;             // under g_kvprep_mu
static KvPrep g_kvprep;                         // under g_kvprep_mu, valid while ready

static bool kv_log(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_LOG"); on = (e != nullptr && e[0] == '1') ? 1 : 0; }
    return on == 1;
}
static void kvset_add(id<MTLBuffer> b) {
    if (g_kv_set == nil || b == nil) { return; }
    [g_kv_set addAllocation:b];
    g_kv_set_dirty = true;
}
static void kvset_remove(id<MTLBuffer> b) {
    if (g_kv_set == nil || b == nil) { return; }
    [g_kv_set removeAllocation:b];
    g_kv_set_dirty = true;
}
// Commit membership and keep the set attached, which is what correctness needs: a set attached
// to the queue is resident for its command buffers whether or not residency was requested.
// Holding it is left to the region end (`kvset_hold`), off the step's path.
// THE CALLER HOLDS g_rset_mu.
static void kvset_sync_locked(void) {
    if (g_kv_set == nil) { return; }
    if (g_kv_set_dirty) {
        [g_kv_set commit];
        g_kv_set_dirty = false;
        g_kv_set_held = false;
    }
    if (!g_kv_set_attached) {
        [g.queue addResidencySet:g_kv_set];
        g_kv_set_attached = true;
    }
}
static void kvset_sync(void) {
    std::lock_guard<std::mutex> lk(g_rset_mu);
    kvset_sync_locked();
}
// THE CALLER HOLDS g_rset_mu. Every page in the set was touched by a region or made resident
// by a preparation, so this only renews the hold the last commit cleared.
static void kvset_hold(void) {
    if (g_kv_set == nil || !g_kv_set_attached || g_kv_set_held || g_kv_set_dirty) { return; }
    [g_kv_set requestResidency];
    g_kv_set_held = true;
}
// THE CALLER HOLDS g_rset_mu. The idle release: the weights' set lets go, so does this one.
static void kvset_release_hold(void) {
    if (g_kv_set == nil || !g_kv_set_held) { return; }
    [g_kv_set endResidency];
    g_kv_set_held = false;
}
static void kvprep_join(void) {
    if (g_kvprep_th.joinable()) { g_kvprep_th.join(); }
}
// Drop a finished preparation nobody adopted. Its staging set lets go of its views; the views
// were never bound, so nothing else holds them.
static void kvprep_discard(void) {
    kvprep_join();
    std::lock_guard<std::mutex> lk(g_kvprep_mu);
    if (g_kvprep_ready && g_kvprep.stage != nil) { [g_kvprep.stage endResidency]; }
    g_kvprep = KvPrep();
    g_kvprep_ready = false;
}
static void kv_retire_ready(void) {
    if (g_kv_retired.empty()) { return; }
    std::vector<KvRetired> keep;
    std::vector<std::pair<uint8_t *, uint64_t>> tails;
    for (KvRetired & r : g_kv_retired) {
        if (r.after > g_region_done) { keep.push_back(std::move(r)); continue; }
        for (id<MTLBuffer> b : r.views) { kvset_remove(b); }
        r.views.clear();
        if (r.stage != nil) { [r.stage endResidency]; r.stage = nil; }
        tails.insert(tails.end(), r.tails.begin(), r.tails.end());
    }
    g_kv_retired.swap(keep);
    // The set lets go of the retired views here, and only then are their pages unwired.
    // REUSABLE on a page that is still wired frees nothing: the advice comes after.
    //
    //   Wrong: advise REUSABLE -> commit the set (view leaves, pages unwired, still counted)
    //   Right: commit the set (view leaves, pages unwired) -> advise REUSABLE -> pages go back
    //
    // `kv_build` says REUSE before any view covers the range again.
    kvset_sync();
    for (const auto & t : tails) { madvise(t.first, (size_t)t.second, MADV_FREE_REUSABLE); }
}
// Adopt prepared views. The old ones stay resident until every region begun so far is done.
static void kv_adopt(KvPrep & p) {
    KvRetired r;
    r.after = g_region_begun;
    r.stage = p.stage;
    auto side = [&](std::vector<id<MTLBuffer>> & bound, std::vector<KvRes> & res,
                    const std::vector<uint64_t> & to, std::vector<id<MTLBuffer>> & made) {
        for (size_t i = 0; i < made.size() && i < bound.size() && i < res.size(); ++i) {
            if (made[i] == nil) { continue; }
            if (to[i] < res[i].committed) {
                r.tails.push_back({res[i].va + to[i], res[i].committed - to[i]});
            }
            kvset_add(made[i]);
            if (bound[i] != nil) { r.views.push_back(bound[i]); }
            bound[i] = made[i];
            res[i].committed = to[i];
        }
    };
    side(g.kv_k, g_kvres_k, p.k, p.kb);
    side(g.kv_v, g_kvres_v, p.v, p.vb);
    kvset_sync();   // the new views are members before any command buffer binds them
    g_kv_retired.push_back(std::move(r));
}
// Build the views a preparation names. `stage` true: in a staging set, made resident here --
// the background thread's job. False: the first command buffer that binds them wires them.
static void kv_build(KvPrep & p, const std::vector<uint8_t *> & vak,
                     const std::vector<uint8_t *> & vav, const std::vector<uint64_t> & ck,
                     const std::vector<uint64_t> & cv, bool stage) {
    const size_t n = p.k.size();
    p.kb.assign(n, nil);
    p.vb.assign(n, nil);
    const double t_set0 = CACurrentMediaTime();
    if (stage) {
        MTLResidencySetDescriptor * d = [[MTLResidencySetDescriptor alloc] init];
        d.label = @"imparo kv stage";
        d.initialCapacity = (NSUInteger)(2 * n + 1);
        p.stage = [g.device newResidencySetWithDescriptor:d error:nil];
    }
    const double t_views0 = CACurrentMediaTime();
    for (size_t i = 0; i < n; ++i) {
        for (int s = 0; s < 2; ++s) {
            uint8_t * va = s == 0 ? vak[i] : vav[i];
            const uint64_t to = s == 0 ? p.k[i] : p.v[i];
            const uint64_t now = s == 0 ? ck[i] : cv[i];
            if (va == nullptr || to == 0 || to == now) { continue; }
            if (to > now) { madvise(va + now, (size_t)(to - now), MADV_FREE_REUSE); }
            id<MTLBuffer> b = [g.device newBufferWithBytesNoCopy:va length:(NSUInteger)to
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];
            if (s == 0) { p.kb[i] = b; } else { p.vb[i] = b; }
            if (p.stage != nil && b != nil) { [p.stage addAllocation:b]; }
        }
    }
    const double t_commit0 = CACurrentMediaTime();
    if (p.stage != nil) { [p.stage commit]; }
    const double t_wire0 = CACurrentMediaTime();
    if (p.stage != nil) { [p.stage requestResidency]; }
    if (kv_log()) {
        const double t1 = CACurrentMediaTime();
        fprintf(stderr, "[imparo] kv build%s: set %.2f ms, views %.2f ms, commit %.2f ms, wire %.2f ms\n",
                stage ? " (prepared)" : "", 1e3 * (t_views0 - t_set0), 1e3 * (t_commit0 - t_views0),
                1e3 * (t_wire0 - t_commit0), 1e3 * (t1 - t_wire0));
    }
}
// What `bytes` asks each side to cover: 0 keeps the side as it is; otherwise page-rounded and
// clamped to the reservation. `grow_only`: never below what the side covers now.
static bool slot_ring_layer(uint32_t layer);
static void kv_targets(uint32_t n_layers, const uint64_t * bytes, bool grow_only,
                       std::vector<uint64_t> & tk, std::vector<uint64_t> & tv, bool & change,
                       bool & over) {
    const size_t n = std::min<size_t>(n_layers, g_kvres_k.size());
    tk.assign(g_kvres_k.size(), 0);
    tv.assign(g_kvres_v.size(), 0);
    change = false;
    over = false;
    for (size_t i = 0; i < g_kvres_k.size(); ++i) {
        tk[i] = g_kvres_k[i].committed;
        tv[i] = g_kvres_v[i].committed;
        if (i >= n || bytes[i] == 0 || g_kvres_k[i].va == nullptr) { continue; }
        // A ring each slot holds its own of keeps the size it was loaded with (its bytes do not
        // follow a position count), and g.kv_k holds the selected slot's, not this reservation's.
        if (slot_ring_layer((uint32_t)i)) { continue; }
        const uint64_t want = page_round(bytes[i]);
        if (want > g_kvres_k[i].reserved) { over = true; continue; }
        const uint64_t k = grow_only ? std::max(want, g_kvres_k[i].committed) : want;
        const uint64_t v = grow_only ? std::max(want, g_kvres_v[i].committed) : want;
        if (k != tk[i] || v != tv[i]) { change = true; }
        tk[i] = k;
        tv[i] = v;
    }
}
// Adopt a finished preparation when it covers `need` (bytes per layer, 0 = no need); a stale or
// short one is discarded. True when one was adopted.
static bool kv_adopt_prepared(uint32_t n_layers, const uint64_t * need) {
    {
        std::lock_guard<std::mutex> lk(g_kvprep_mu);
        if (g_kvprep_busy || !g_kvprep_ready) { return false; }
    }
    kvprep_join();
    KvPrep p;
    {
        std::lock_guard<std::mutex> lk(g_kvprep_mu);
        p = std::move(g_kvprep);
        g_kvprep = KvPrep();
        g_kvprep_ready = false;
    }
    bool ok = p.gen == g_kv_gen && p.k.size() == g_kvres_k.size();
    for (size_t i = 0; ok && i < p.k.size() && i < n_layers; ++i) {
        if (need != nullptr && need[i] != 0 && (page_round(need[i]) > p.k[i] || page_round(need[i]) > p.v[i])) {
            ok = false;
        }
    }
    if (!ok) {
        if (p.stage != nil) { [p.stage endResidency]; }
        return false;
    }
    kv_adopt(p);
    if (kv_log()) {
        uint64_t k = 0;
        for (size_t i = 0; i < g_kvres_k.size(); ++i) { k += g_kvres_k[i].committed + g_kvres_v[i].committed; }
        fprintf(stderr, "[imparo] kv views adopted: %.1f MiB committed, prepared in %.1f ms off the step\n",
                (double)k / 1048576.0, p.ms);
    }
    return true;
}

// Grow each layer's view to cover `bytes` (0 = leave the layer alone). Never copies: the bigger
// view maps the same memory. A preparation that covers it is adopted with no wait; otherwise the
// views are built here and the first command buffer that binds them wires the new pages -- the
// wait a preparation exists to hide.
extern "C" int imparo_metal_grow_kv(uint32_t n_layers, const uint64_t * bytes) {
    if (g.device == nil) { return 1; }
    kv_retire_ready();
    std::vector<uint64_t> tk, tv;
    bool change = false, over = false;
    kv_targets(n_layers, bytes, true, tk, tv, change, over);
    if (over) {
        NSLog(@"imparo metal: kv grow past the reservation refused (the fit sized it; a caller asked for more)");
        return 5;
    }
    if (!change) { return 0; }
    // Wait for a preparation in flight: it is the same work, already under way.
    bool busy;
    { std::lock_guard<std::mutex> lk(g_kvprep_mu); busy = g_kvprep_busy; }
    if (busy) {
        const double t0 = CACurrentMediaTime();
        kvprep_join();
        if (kv_log()) {
            fprintf(stderr, "[imparo] kv grow waited %.1f ms for the preparation in flight\n",
                    1e3 * (CACurrentMediaTime() - t0));
        }
    }
    if (kv_adopt_prepared(n_layers, bytes)) {
        kv_targets(n_layers, bytes, true, tk, tv, change, over);
        if (!change) { return 0; }
    }
    const double t0 = CACurrentMediaTime();
    KvPrep p;
    p.gen = g_kv_gen;
    p.k = tk;
    p.v = tv;
    std::vector<uint8_t *> vak(g_kvres_k.size()), vav(g_kvres_v.size());
    std::vector<uint64_t> ck(g_kvres_k.size()), cv(g_kvres_v.size());
    for (size_t i = 0; i < g_kvres_k.size(); ++i) {
        vak[i] = g_kvres_k[i].va; ck[i] = g_kvres_k[i].committed;
        vav[i] = g_kvres_v[i].va; cv[i] = g_kvres_v[i].committed;
    }
    kv_build(p, vak, vav, ck, cv, false);
    for (size_t i = 0; i < p.kb.size(); ++i) {
        if ((vak[i] != nullptr && tk[i] != ck[i] && p.kb[i] == nil)
            || (vav[i] != nullptr && tv[i] != cv[i] && p.vb[i] == nil)) { return 2; }
    }
    kv_adopt(p);
    if (kv_log()) {
        uint64_t k = 0;
        for (size_t i = 0; i < g_kvres_k.size(); ++i) { k += g_kvres_k[i].committed + g_kvres_v[i].committed; }
        fprintf(stderr, "[imparo] kv views grown on the step: %.1f MiB committed (%.2f ms; no preparation covered it)\n",
                (double)k / 1048576.0, 1e3 * (CACurrentMediaTime() - t0));
    }
    return 0;
}

// Prepare views covering `bytes` on a background thread (0 = leave the layer alone), so the
// forward that needs them adopts them with no wait. Bigger than now: the growth the next steps
// will need. Smaller: a release, adopted when it still covers the next forward's need. One
// preparation at a time; a finished one waits for its adoption.
extern "C" void imparo_metal_kv_prefetch(uint32_t n_layers, const uint64_t * bytes) {
    if (g.device == nil) { return; }
    static int off = -1;
    if (off < 0) { const char * e = getenv("IMPARO_KV_PREFETCH"); off = (e != nullptr && e[0] == '0') ? 1 : 0; }
    if (off == 1) { return; }   // diagnostic: every growth builds its views on the step
    {
        std::lock_guard<std::mutex> lk(g_kvprep_mu);
        if (g_kvprep_busy || g_kvprep_ready) { return; }
    }
    std::vector<uint64_t> tk, tv;
    bool change = false, over = false;
    kv_targets(n_layers, bytes, false, tk, tv, change, over);
    if (over || !change) { return; }
    kvprep_join();
    std::vector<uint8_t *> vak(g_kvres_k.size()), vav(g_kvres_v.size());
    std::vector<uint64_t> ck(g_kvres_k.size()), cv(g_kvres_v.size());
    for (size_t i = 0; i < g_kvres_k.size(); ++i) {
        vak[i] = g_kvres_k[i].va; ck[i] = g_kvres_k[i].committed;
        vav[i] = g_kvres_v[i].va; cv[i] = g_kvres_v[i].committed;
    }
    const uint64_t gen = g_kv_gen;
    { std::lock_guard<std::mutex> lk(g_kvprep_mu); g_kvprep_busy = true; }
    // Joined at exit, as the idle watcher is. The last preparation's thread stays joinable
    // until something joins it, and a joinable std::thread destroyed with the statics calls
    // std::terminate: every process that grew its KV views aborted after its answer (rc 134).
    // atexit runs this before the destructors of statics built earlier.
    static const bool joined_at_exit = (atexit(kvprep_join), true);
    (void)joined_at_exit;
    g_kvprep_th = std::thread([gen, tk, tv, vak, vav, ck, cv]() {
        @autoreleasepool {
            const double t0 = CACurrentMediaTime();
            KvPrep p;
            p.gen = gen;
            p.k = tk;
            p.v = tv;
            kv_build(p, vak, vav, ck, cv, true);
            p.ms = 1e3 * (CACurrentMediaTime() - t0);
            std::lock_guard<std::mutex> lk(g_kvprep_mu);
            g_kvprep = std::move(p);
            g_kvprep_ready = true;
            g_kvprep_busy = false;
        }
    });
}

// Adopt a finished preparation if it covers `need` (0 = no need for that layer): the forward
// entry's chance to take a prepared release, or a growth that finished early.
extern "C" void imparo_metal_kv_adopt(uint32_t n_layers, const uint64_t * need) {
    if (g.device == nil) { return; }
    kv_retire_ready();
    (void)kv_adopt_prepared(n_layers, need);
}

// Shrink each layer's view to `bytes` (0 leaves a layer alone; a target above the view leaves
// it too): the release, once the pool holds nothing above. The smaller view maps the same
// memory, so nothing moves. Called between requests with no command buffer in flight, so the
// old views retire here and the pages above the new ones go back to the system at once.
extern "C" int imparo_metal_kv_release(uint32_t n_layers, const uint64_t * bytes) {
    if (g.device == nil) { return 1; }
    const double t0 = CACurrentMediaTime();
    // A preparation was built for the old size; it covers what is being given up.
    kvprep_discard();
    kv_retire_ready();
    std::vector<uint64_t> tk, tv;
    bool change = false, over = false;
    kv_targets(n_layers, bytes, false, tk, tv, change, over);
    if (over) { return 5; }
    uint64_t before = 0, after = 0;
    for (size_t i = 0; i < g_kvres_k.size(); ++i) {
        before += g_kvres_k[i].committed + g_kvres_v[i].committed;
        tk[i] = std::min(tk[i], g_kvres_k[i].committed);
        tv[i] = std::min(tv[i], g_kvres_v[i].committed);
        after += tk[i] + tv[i];
    }
    if (after == before) { return 0; }
    KvPrep p;
    p.gen = g_kv_gen;
    p.k = tk;
    p.v = tv;
    std::vector<uint8_t *> vak(g_kvres_k.size()), vav(g_kvres_v.size());
    std::vector<uint64_t> ck(g_kvres_k.size()), cv(g_kvres_v.size());
    for (size_t i = 0; i < g_kvres_k.size(); ++i) {
        vak[i] = g_kvres_k[i].va; ck[i] = g_kvres_k[i].committed;
        vav[i] = g_kvres_v[i].va; cv[i] = g_kvres_v[i].committed;
    }
    kv_build(p, vak, vav, ck, cv, false);
    for (size_t i = 0; i < p.kb.size(); ++i) {
        if ((vak[i] != nullptr && tk[i] != ck[i] && p.kb[i] == nil)
            || (vav[i] != nullptr && tv[i] != cv[i] && p.vb[i] == nil)) { return 2; }
    }
    kv_adopt(p);
    kv_retire_ready();
    if (kv_log()) {
        fprintf(stderr, "[imparo] kv views released: %.1f -> %.1f MiB committed (%.2f ms)\n",
                (double)before / 1048576.0, (double)after / 1048576.0,
                1e3 * (CACurrentMediaTime() - t0));
    }
    return 0;
}

// The device's largest buffer: no KV view can be longer, so the fit keeps a reservation under it.
extern "C" uint64_t imparo_metal_max_buffer_bytes(void) {
    return g.device == nil ? 0ull : (uint64_t)[g.device maxBufferLength];
}

// Bytes the KV views cover now, K and V together: what the KV tier holds resident.
extern "C" uint64_t imparo_metal_kv_committed_bytes(void) {
    uint64_t b = 0;
    for (size_t i = 0; i < g_kvres_k.size(); ++i) { b += g_kvres_k[i].committed + g_kvres_v[i].committed; }
    return b;
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
    // Replacement is a retired-region operation. A queued step may still address
    // this no-copy mapping; refuse a resize instead of invalidating it in flight.
    if (g.enc != nil || g.cb != nil || !g.pending.empty() || !g.outstanding.empty()) { return 4; }
    // Every buffer placed in the old arena must go before it does.
    for (uint32_t i = 0; i < B_COUNT; ++i) {
        if (g.in_arena[i]) {
            g.bufs[i] = nil;
            g.sizes[i] = 0;
            g.in_arena[i] = 0;
            buf_fresh_clear(i);
        }
    }
    if (g.arena_buf != nil) {
        // removeAllocation is deferred until commit. Commit the removal while the
        // old no-copy backing still exists, then release the resource before unmap.
        // Without this pair, each wide-prefill / tail / decode resize left another
        // arena in the residency set and inflated its fast-tier budget accounting.
        std::lock_guard<std::mutex> lk(g_rset_mu);
        rset_remove(g.arena_buf);
        if (g_rset != nil) {
            [g_rset commit];
            g_rset_dirty = false;
            // A COMMIT CLEARS THE HOLD. Membership changed, so whatever
            // `requestResidency` pinned is no longer pinned -- the next region has to
            // ask again. Saying otherwise here would skip that ask and leave the tier
            // pageable while the engine believed it was wired.
            g_rset_held = false;
        }
        g.arena_buf = nil;
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
    if (g.arena_buf == nil) {
        munmap(g_arena, (size_t)g_arena_size);
        g_arena = nullptr; g_arena_size = 0;
        return 3;
    }
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
    buf_fresh_clear(id);   // arena memory is newBufferWithBytesNoCopy: nothing zeroed it
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

// IMPARO_KV_PT_LOG=1: log every page-table use past the entries the pool set for the layer.
// Such a page maps to whatever the buffer held -- the identity value from creation, or an older
// conversation's entry -- so the write lands on a block the pool did not assign to it.
static void kv_pt_overrun_probe(uint32_t layer, uint32_t entries_needed) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_KV_PT_LOG"); on = (e != nullptr && e[0] == '1') ? 1 : 0; }
    if (on == 0 || layer >= g.kv_pt_set.size()) { return; }
    const uint32_t set = g.kv_pt_set[layer];
    if (set == 0 || entries_needed <= set || g.kv_pt[layer] == nil) { return; }
    const uint32_t have = (uint32_t)([g.kv_pt[layer] length] / 4u);
    const uint32_t * e = (const uint32_t *)[g.kv_pt[layer] contents];
    for (uint32_t page = set; page < entries_needed && page < have; ++page) {
        int32_t owner = -1;
        for (uint32_t i = 0; i < set; ++i) { if (e[i] == e[page]) { owner = (int32_t)i; break; } }
        fprintf(stderr, "[imparo] kv page past the pool's table: layer %u page %u (pool set %u) -> block %u%s%d\n",
                layer, page, set, e[page], owner >= 0 ? ", the block of live page " : ", no live page", owner);
    }
}

static void ensure_kv_pt(uint32_t layer, uint32_t entries_needed) {
    kv_pt_overrun_probe(layer, entries_needed);
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
    // KV-tier objects: in the KV set, so a table that grows never commits the weights' set.
    kvset_remove(cur);
    kvset_add(fresh);
    g.kv_pt[layer] = fresh;
}

// The pool assigns physical blocks: replace a layer's table entries wholesale.
extern "C" void imparo_metal_set_kv_pages(uint32_t layer, const uint32_t * e, uint32_t n) {
    if (g.kv_pt_set.size() <= layer) { g.kv_pt_set.resize(layer + 1, 0u); }
    g.kv_pt_set[layer] = 0u;
    ensure_kv_pt(layer, n);
    g.kv_pt_set[layer] = n;
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

// CO-BATCHED DECODE SLOTS (docs/continuous-batching.md). A slot is one conversation's device
// state: its page table per layer, its per-conversation buffers (the recurrent state and its
// snapshot twin, named by the caller) and the rings of its windowed layers (named by the
// caller too). The SELECTED slot's state lives in the fields every one-row path already reads
// -- g.kv_pt, g.kv_pt_ident, g.kv_pt_set, the named g.bufs and the ringed layers' g.kv_k /
// g.kv_v -- so prefill, a lone decode, capture and restore work on it unchanged; the other
// slots' state waits here. Selecting swaps the two (std::swap, no copies). A co-batched row
// selects its slot around its own dispatches, which bind their buffers at encode time. A slot's
// buffers are made the first time it is selected, so a server that never runs two
// conversations at once holds one slot's worth.
struct SlotState {
    std::vector<id<MTLBuffer>> kv_pt;
    std::vector<uint8_t> kv_pt_ident;
    std::vector<uint32_t> kv_pt_set;
    std::vector<id<MTLBuffer>> bufs;   // one per g_slot_buf_ids entry
    std::vector<size_t> sizes;
    // The rings of the windowed layers, K and V, one per g_slot_ring_layers entry: made on the
    // slot's first select and given back at its release, like `bufs`. Slot 0's are the caches
    // the load allocated, copied out of the KV view machinery when the slots are made.
    std::vector<id<MTLBuffer>> ring_k, ring_v;
    std::vector<size_t> ring_bytes;
    bool made = true;                  // false: `bufs` not made yet, `sizes` are theirs to be
};
static std::vector<SlotState> g_slots(1);
static std::vector<uint32_t> g_slot_buf_ids;
static std::vector<uint32_t> g_slot_ring_layers;
static uint32_t g_slot_cur = 0;

// Whether `layer`'s cache is a ring each slot holds its own of. The KV view machinery (grow,
// prepare, release) never resizes such a layer: the buffer in g.kv_k / g.kv_v is whichever
// slot's is selected, not a view over the layer's reservation.
static bool slot_ring_layer(uint32_t layer) {
    return std::find(g_slot_ring_layers.begin(), g_slot_ring_layers.end(), layer)
        != g_slot_ring_layers.end();
}

static void slot_swap(uint32_t s) {
    SlotState & st = g_slots[s];
    std::swap(g.kv_pt, st.kv_pt);
    std::swap(g.kv_pt_ident, st.kv_pt_ident);
    std::swap(g.kv_pt_set, st.kv_pt_set);
    st.bufs.resize(g_slot_buf_ids.size(), nil);
    st.sizes.resize(g_slot_buf_ids.size(), 0);
    for (size_t i = 0; i < g_slot_buf_ids.size(); ++i) {
        const uint32_t bid = g_slot_buf_ids[i];
        std::swap(g.bufs[bid], st.bufs[i]);
        std::swap(g.sizes[bid], st.sizes[i]);
    }
    st.ring_k.resize(g_slot_ring_layers.size(), nil);
    st.ring_v.resize(g_slot_ring_layers.size(), nil);
    for (size_t i = 0; i < g_slot_ring_layers.size(); ++i) {
        const uint32_t l = g_slot_ring_layers[i];
        std::swap(g.kv_k[l], st.ring_k[i]);
        std::swap(g.kv_v[l], st.ring_v[i]);
    }
}

// Makes slot s's buffers, zeroed, at the sizes set_slots recorded for it, and its rings where it
// has none. False: a buffer could not be made, and the slot stays unmade.
static bool slot_make(uint32_t s) {
    SlotState & st = g_slots[s];
    std::vector<id<MTLBuffer>> bufs;
    for (size_t bytes : st.sizes) {
        id<MTLBuffer> b = nil;
        if (bytes > 0) {
            b = [g.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (b == nil) { return false; }
            memset([b contents], 0, bytes);
        }
        bufs.push_back(b);
    }
    // Zeroed like the rest: a prefill attention tile may read ring cells past its last key and
    // weigh them by zero, and a zero weight times a NaN left in the cell is a NaN.
    std::vector<id<MTLBuffer>> rk, rv;
    const bool need_rings = !g_slot_ring_layers.empty()
        && (st.ring_k.size() < g_slot_ring_layers.size() || st.ring_k[0] == nil);
    if (need_rings) {
        for (size_t bytes : st.ring_bytes) {
            for (std::vector<id<MTLBuffer>> * side : {&rk, &rv}) {
                id<MTLBuffer> b = [g.device newBufferWithLength:bytes
                                                        options:MTLResourceStorageModeShared];
                if (b == nil) { return false; }
                memset([b contents], 0, bytes);
                side->push_back(b);
            }
        }
    }
    for (id<MTLBuffer> b : bufs) {
        if (b != nil) { rset_add(b); }
    }
    st.bufs = std::move(bufs);
    if (need_rings) {
        for (id<MTLBuffer> b : rk) { rset_add(b); }
        for (id<MTLBuffer> b : rv) { rset_add(b); }
        st.ring_k = std::move(rk);
        st.ring_v = std::move(rv);
    }
    st.made = true;
    if (kv_log()) {
        size_t rings = 0;
        for (size_t b : st.ring_bytes) { rings += 2 * b; }
        fprintf(stderr, "[imparo] slot %u made: rings %s (%.1f MiB)\n", s,
                need_rings ? "made" : "kept", (double)(need_rings ? rings : 0) / 1048576.0);
    }
    return true;
}

// Returns 0 with slot s selected; 1 for a slot that does not exist, 4 when its buffers could
// not be made on this first select -- the selection is unchanged on either.
extern "C" int imparo_metal_select_slot(uint32_t s) {
    if (s >= g_slots.size()) { return 1; }
    if (s == g_slot_cur) { return 0; }
    if (!g_slots[s].made) {
        // Its own pool, as at set_slots: a reference autoreleased on a thread that never drains
        // would keep this slot's buffers alive past their release.
        bool made = false;
        @autoreleasepool { made = slot_make(s); }
        if (!made) { return 4; }
    }
    slot_swap(g_slot_cur);   // the live state goes back to its slot
    slot_swap(s);            // and the chosen slot's comes out
    g_slot_cur = s;
    return 0;
}

// Gives back slot s's per-conversation buffers: its next select makes them again, zeroed, at the
// sizes they had. For a slot other than the selected one, with nothing in flight. Returns 0 when
// buffers went back, 1 for the selected slot or one that does not exist, 2 for a slot holding
// none (never selected, or given back already), 3 with work encoded or in flight (a region may
// still read them, and emptying a buffer discards its contents at once).
extern "C" int imparo_metal_release_slot(uint32_t s) {
    if (s >= g_slots.size() || s == g_slot_cur) { return 1; }
    SlotState & st = g_slots[s];
    if (!st.made) { return 2; }
    if (g.enc != nil || g.cb != nil || !g.pending.empty() || !g.outstanding.empty()) { return 3; }
    {
        // A removal waits for the set's commit, and the set keeps the buffer until then:
        // commit now, or the pages stay until the next region begins.
        std::lock_guard<std::mutex> lk(g_rset_mu);
        for (id<MTLBuffer> b : st.bufs) {
            if (b != nil) { rset_remove(b); }
        }
        for (id<MTLBuffer> b : st.ring_k) { if (b != nil) { rset_remove(b); } }
        for (id<MTLBuffer> b : st.ring_v) { if (b != nil) { rset_remove(b); } }
        if (g_rset != nil) {
            [g_rset commit];
            g_rset_dirty = false;
        }
    }
    // Dropping the reference alone leaves the pages in the driver's cache.
    for (size_t i = 0; i < st.bufs.size(); ++i) {
        if (st.bufs[i] != nil) {
            [st.bufs[i] setPurgeableState:MTLPurgeableStateEmpty];
            st.bufs[i] = nil;
        }
    }
    uint64_t ring_bytes = 0;
    for (std::vector<id<MTLBuffer>> * side : {&st.ring_k, &st.ring_v}) {
        for (size_t i = 0; i < side->size(); ++i) {
            if ((*side)[i] != nil) {
                ring_bytes += [(*side)[i] length];
                [(*side)[i] setPurgeableState:MTLPurgeableStateEmpty];
                (*side)[i] = nil;
            }
        }
    }
    if (kv_log()) {
        fprintf(stderr, "[imparo] slot %u released: rings %.1f MiB, device now %.1f MiB\n", s,
                (double)ring_bytes / 1048576.0,
                (double)[g.device currentAllocatedSize] / 1048576.0);
    }
    st.made = false;
    return 0;
}

// Moves the load's rings out of the KV view machinery into buffers of their own, contents
// kept, so that slot 0's rings are made and given back like every other slot's. The views are
// retired as a shrink retires them: they leave the set, then their pages go back.
static bool slot_rings_detach(void) {
    KvRetired r;
    r.after = g_region_begun;
    std::vector<std::pair<uint32_t, std::pair<id<MTLBuffer>, id<MTLBuffer>>>> made;
    for (uint32_t l : g_slot_ring_layers) {
        id<MTLBuffer> own[2] = {nil, nil};
        for (int s = 0; s < 2; ++s) {
            id<MTLBuffer> view = s == 0 ? g.kv_k[l] : g.kv_v[l];
            own[s] = [g.device newBufferWithLength:[view length]
                                           options:MTLResourceStorageModeShared];
            if (own[s] == nil) { return false; }
            memcpy([own[s] contents], [view contents], [view length]);
        }
        made.push_back({l, {own[0], own[1]}});
    }
    for (const auto & m : made) {
        const uint32_t l = m.first;
        for (int s = 0; s < 2; ++s) {
            KvRes & res = s == 0 ? g_kvres_k[l] : g_kvres_v[l];
            std::vector<id<MTLBuffer>> & side = s == 0 ? g.kv_k : g.kv_v;
            r.views.push_back(side[l]);
            if (res.va != nullptr && res.committed > 0) {
                r.tails.push_back({res.va, res.committed});
            }
            res.committed = 0;
            side[l] = s == 0 ? m.second.first : m.second.second;
            rset_add(side[l]);
        }
    }
    if (kv_log()) {
        uint64_t bytes = 0;
        for (const auto & t : r.tails) { bytes += t.second; }
        fprintf(stderr, "[imparo] slot rings: %zu layers copied out of the KV views, %.1f MiB of "
                "views retired\n", made.size(), (double)bytes / 1048576.0);
    }
    g_kv_retired.push_back(std::move(r));
    return true;
}

// Slots 0..n. A new slot will own a buffer for each id the selected slot holds and a ring for
// each windowed layer named, sized like the selected slot's are now, made on the slot's first
// select; and no page table: its conversation sets one before use. The ids and layers are read
// on the first call only, which needs nothing encoded or in flight (it copies the rings). Returns
// 0, or nonzero for an id or a layer this cannot hold per slot, or work in flight.
extern "C" int imparo_metal_set_slots(uint32_t n, const uint32_t * buf_ids, uint32_t n_ids,
                                      const uint32_t * ring_layers, uint32_t n_rings) {
    if (n == 0u) { return 1; }
    if (g_slots.size() == 1u && g_slot_buf_ids.empty() && g_slot_ring_layers.empty()) {
        for (uint32_t i = 0; i < n_ids; ++i) {
            if (buf_ids[i] >= B_COUNT) { return 2; }
            if (g.bufs[buf_ids[i]] != nil && g.in_arena[buf_ids[i]]) { return 3; }
        }
        for (uint32_t i = 0; i < n_rings; ++i) {
            const uint32_t l = ring_layers[i];
            if (l >= g.kv_k.size() || g.kv_k[l] == nil || g.kv_v[l] == nil) { return 4; }
            // One size for both sides: a slot's ring is made at the K side's length.
            if ([g.kv_k[l] length] != [g.kv_v[l] length]) { return 5; }
        }
        if (n_rings > 0u
            && (g.enc != nil || g.cb != nil || !g.pending.empty() || !g.outstanding.empty())) {
            return 6;
        }
        g_slot_buf_ids.assign(buf_ids, buf_ids + n_ids);
        g_slot_ring_layers.assign(ring_layers, ring_layers + n_rings);
        // Its own pool: the caller may be a thread with none, and an autoreleased reference
        // left there keeps slot 0's rings alive past their release, for the process's life.
        bool detached = false;
        @autoreleasepool { detached = slot_rings_detach(); }
        if (!detached) {
            g_slot_ring_layers.clear();
            return 7;
        }
    }
    while (g_slots.size() < n) {
        SlotState st;
        for (uint32_t bid : g_slot_buf_ids) {
            st.sizes.push_back(g.bufs[bid] != nil ? g.sizes[bid] : 0);
        }
        for (uint32_t l : g_slot_ring_layers) {
            st.ring_bytes.push_back((size_t)[g.kv_k[l] length]);
        }
        st.made = false;
        g_slots.push_back(std::move(st));
    }
    return 0;
}

static inline uint64_t kv_reg(uint32_t layer, bool is_v) {
    const std::vector<uint64_t> & v = is_v ? g.kv_reg_v : g.kv_reg_k;
    return layer < v.size() ? v[layer] : 0ull;
}

// Reserve each layer side's address space and commit its first view. `reserve[i]` bytes of
// address space, `bytes[i]` of it covered now (the view), for K and V alike; a layer with both
// zero owns no cache. Replaces whatever was there, so it runs with nothing in flight: at load,
// and when the pool takes over the cache before its first request.
extern "C" int imparo_metal_alloc_kv_reserved(uint32_t n_layers, const uint64_t * bytes,
                                              const uint64_t * reserve) {
    if (g.device == nil) { return 1; }
    if (g.cb != nil || !g.outstanding.empty()) { return 4; }
    // Slots hold rings sized from this allocation, and the selected slot's are in g.kv_k:
    // replacing the cache under them would leave every slot's rings sized for the old one.
    if (!g_slot_ring_layers.empty() && g_slots.size() > 1u) { return 6; }
    kvprep_discard();
    // Everything the old reservations backed goes: the views leave the set before the mappings
    // they cover are unmapped.
    for (id<MTLBuffer> b : g.kv_k) { kvset_remove(b); }
    for (id<MTLBuffer> b : g.kv_v) { kvset_remove(b); }
    for (KvRetired & r : g_kv_retired) {
        for (id<MTLBuffer> b : r.views) { kvset_remove(b); }
        if (r.stage != nil) { [r.stage endResidency]; }
    }
    g_kv_retired.clear();
    g.kv_k.assign(n_layers, nil);
    g.kv_v.assign(n_layers, nil);
    kvset_sync();
    for (std::vector<KvRes> * side : {&g_kvres_k, &g_kvres_v}) {
        for (KvRes & r : *side) {
            if (r.va != nullptr) { munmap(r.va, (size_t)r.reserved); }
        }
        side->assign(n_layers, KvRes());
    }
    g_kv_gen += 1;
    if (g_kv_set == nil && g_rset != nil) {
        MTLResidencySetDescriptor * d = [[MTLResidencySetDescriptor alloc] init];
        d.label = @"imparo kv tier";
        d.initialCapacity = (NSUInteger)(4 * n_layers + 64);
        g_kv_set = [g.device newResidencySetWithDescriptor:d error:nil];
    }
    g.kv_pt.assign(n_layers, nil);
    g.kv_reg_k.assign(n_layers, 0ull);
    g.kv_reg_v.assign(n_layers, 0ull);
    for (uint32_t i = 0; i < n_layers; ++i) {
        const uint64_t commit = page_round(bytes[i]);
        const uint64_t res = std::max(page_round(reserve[i]), commit);
        if (res == 0) { continue; }
        for (int s = 0; s < 2; ++s) {
            KvRes & r = s == 0 ? g_kvres_k[i] : g_kvres_v[i];
            void * m = mmap(nullptr, (size_t)res, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
            if (m == MAP_FAILED) { return 1; }
            r.va = (uint8_t *)m;
            r.reserved = res;
            r.committed = commit;
            if (commit == 0) { continue; }
            id<MTLBuffer> b = [g.device newBufferWithBytesNoCopy:m length:(NSUInteger)commit
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
            if (b == nil) { return 1; }
            if (s == 0) { g.kv_k[i] = b; } else { g.kv_v[i] = b; }
            kvset_add(b);
        }
    }
    kvset_sync();
    { std::lock_guard<std::mutex> lk(g_rset_mu); kvset_hold(); }
    return 0;
}

// The cache with no room to grow in place: reserve exactly what is committed. Kept for callers
// that size the cache once.
extern "C" int imparo_metal_alloc_kv(uint32_t n_layers, const uint64_t * bytes) {
    return imparo_metal_alloc_kv_reserved(n_layers, bytes, bytes);
}

// `[queue commandBuffer]` returns an AUTORELEASED object. Splitting a decode token into
// several command buffers therefore piles them up for as long as the enclosing pool lives,
// which on a server thread is the whole request: 6 per token x 64 tokens sat unreleased,
// and the footprint grew with the split (1 cb/token 214.2 MiB, 21 cb/token 220.6 MiB).
// One pool per forward pass releases them at the end of the token instead.
// Diagnostic (IMPARO_CB_PROBE=1): at a synchronous region end, print when each of its command
// buffers started on the GPU relative to the region's begin, and how long each ran -- a wait
// before the GPU starts (residency) and a slower GPU are different defects.
static bool cb_probe(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_CB_PROBE"); on = (e != nullptr && e[0] == '1') ? 1 : 0; }
    return on == 1;
}
static double g_cb_probe_t0 = 0.0;
extern "C" void imparo_metal_begin(void) {
    if (g.pool == nullptr) { g.pool = objc_autoreleasePoolPush(); }
    if (cb_probe()) { g_cb_probe_t0 = CACurrentMediaTime(); }
    kv_retire_ready();
    g_region_begun += 1;
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
    // A command buffer may write any bound buffer, and which ones is not knowable here,
    // so no buffer is "as allocated" past this point.
    g_buf_fresh = 0ull;
    // Both modes: the tick period is what turns a counter reading into a duration, and the
    // dispatch-boundary mode needs it exactly as much as the encoder mode does.
    if (g_prof) { [g.device sampleTimestamps:&g_ts_cpu0 gpuTimestamp:&g_ts_gpu0]; }
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
    wpf_commit(g.cb);
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
// The LONGEST single command buffer of the last region, not the sum. What starves the
// rest of the machine is one buffer holding the GPU, not a region made of many short
// ones -- ten 100 ms buffers leave nine gaps for the compositor, one 5 s buffer leaves
// none. See `imparo_metal_longest_cb_us`.
static double g_longest_cb_s = 0.0;

extern "C" double imparo_metal_last_gpu_us(void) { return g_last_gpu_s * 1e6; }
extern "C" double imparo_metal_longest_cb_us(void) { return g_longest_cb_s * 1e6; }

extern "C" int imparo_metal_end(void) {
    mega_prog_flush(); g_prog_base = 0u;   // a pending program run ends with its region (task #153)
    [g.enc endEncoding];
    const double t0 = g_prof ? CACurrentMediaTime() : 0.0;
    const double t_probe_commit = cb_probe() ? CACurrentMediaTime() : 0.0;
    wpf_commit(g.cb);
    [g.cb commit];
    [g.cb waitUntilCompleted];
    if (cb_probe()) {
        const double b = g_cb_probe_t0, now = CACurrentMediaTime();
        fprintf(stderr, "[imparo] cb probe: region at %.4f s, wall %.2f ms, encode-to-last-commit %.2f ms, cbs %zu:",
                b, 1e3 * (now - b), 1e3 * (t_probe_commit - b), g.pending.size() + 1);
        for (id<MTLCommandBuffer> pcb : g.pending) {
            fprintf(stderr, " [start +%.2f gpu %.2f]", 1e3 * ([pcb GPUStartTime] - b), 1e3 * cb_gpu_seconds(pcb));
        }
        fprintf(stderr, " [start +%.2f gpu %.2f]\n", 1e3 * ([g.cb GPUStartTime] - b), 1e3 * cb_gpu_seconds(g.cb));
    }
    // Buffers on one queue complete in commit order, so the wait above means every
    // flushed buffer of this region is done and its timestamps are valid.
    // One sum for both consumers. Buffers on a queue complete in commit order, so the
    // wait above means every flushed buffer of this region is done and its timestamps
    // are valid; summing only the last would undercount by however many flushes ran.
    // A buffer that never reached the GPU reports zeroes for both stamps, and one whose
    // end stamp was not recorded reports an end BEFORE its start -- so the subtraction is
    // not always a duration. Unguarded it is a huge negative that poisons every consumer:
    // the per-category profile divides by the total, so all-negative ticks still produced
    // shares that summed to 1 and PRINTED PLAUSIBLE MILLISECONDS (2026-09-09, found when
    // the profile started printing raw sums instead of wall-rescaled shares).
    g_last_gpu_s = cb_gpu_seconds(g.cb);
    g_longest_cb_s = g_last_gpu_s;
    for (id<MTLCommandBuffer> pcb : g.pending) {
        const double one = cb_gpu_seconds(pcb);
        g_last_gpu_s += one;
        if (one > g_longest_cb_s) { g_longest_cb_s = one; }
    }
    if (g_prof) {
        g_prof_region_ticks = 0.0;
        for (uint32_t i = 0; i < PC_N; ++i) { g_prof_cat_ticks[i] = 0.0; }
        prof_resolve();
        MTLTimestamp c1 = 0, g1 = 0;
        [g.device sampleTimestamps:&c1 gpuTimestamp:&g1];
        if (g1 > g_ts_gpu0 && c1 > g_ts_cpu0) {
            const double ns_per_tick = (double)(c1 - g_ts_cpu0) / (double)(g1 - g_ts_gpu0);
            g_prof_kernel_s += g_prof_region_ticks * ns_per_tick * 1e-9;
            for (uint32_t i = 0; i < PC_N; ++i) {
                g_prof_cat_s[i] += g_prof_cat_ticks[i] * ns_per_tick * 1e-9;
            }
        }
        g_prof_wall_s += CACurrentMediaTime() - t0;
        g_prof_gpu_s  += g_last_gpu_s;
        g_prof_cbs    += (uint64_t)g.pending.size() + 1;
    }
    g.pending.clear();
    g_region_done = g_region_begun;
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
    wpf_commit(g.cb);
    [g.cb commit];
    Context::Region r;
    r.cbs = g.pending;
    r.cbs.push_back(g.cb);
    r.dbg_slot = g_mega_region_slot;
    r.seq = g_region_begun;
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
    if (r.seq > g_region_done) { g_region_done = r.seq; }
    double s = 0.0;
    bool bad = false;
    for (id<MTLCommandBuffer> cb : r.cbs) {
        s += cb_gpu_seconds(cb);   // guarded: see the helper -- these stamps are not always a duration
        if ([cb error] != nil) {
            bad = true;
            // SAY WHAT FAILED. This branch set a flag and printed nothing, so a GPU fault
            // in a forward reached the caller as "wrong numbers" with no cause named --
            // the same class of silent failure the mega failsafe was built for.
            static uint32_t said = 0u;
            if (said < 4u) {
                said += 1u;
                NSLog(@"imparo metal: FORWARD command buffer failed: %@", [cb error]);
            }
        }
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
    if (id < 64u) { g_float_stale &= ~(1ull << id); }
    buf_fresh_clear(id);
    std::memcpy((char *)[g.bufs[id] contents] + g.buf_off[id] + off * 4, src, n * 4);
}
extern "C" void imparo_metal_zero(uint32_t id, uint64_t off, uint64_t n) {
    // Already zero everywhere, and proving it costs nothing: skip the write, keep the bit.
    if (buf_is_fresh(id)) { return; }
    g_cvt_valid = 0;   // same invalidation as a host write: the buffer changed
    if (id == g_xh_src) { g_xh_src = 0xffffffffu; }
    std::memset((char *)[g.bufs[id] contents] + g.buf_off[id] + off * 4, 0, n * 4);
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
    bool     fell_back;   // the rule's choice was illegal here and shape 3 took over
};

// The rule's choice, then legality. Both are pure functions of (n_tok, n_in) and the one
// seated shape index; nothing here touches Metal.
static Q8Geometry q8_geometry(uint32_t n_tok, uint32_t n_in) {
    uint32_t shape = q8_tile(n_tok);
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

// Test hook: the shape this dispatch would use.
extern "C" uint32_t imparo_metal_q8_pick_shape(uint32_t n_tok, uint32_t n_in) {
    return q8_geometry(n_tok, n_in).shape;
}

// `w_off2` of a plain projection. THE GATED PAIR passes the up tensor's offset instead,
// and the encode paths below then select the _gh pipeline, walk 2 * n_out virtual rows
// and bind the second offset at buffer 10 (imparo.metal, "THE GATED PAIR").
constexpr uint64_t NO_PAIR = ~0ull;

// THE ONE-ROW GEOMETRY of the Q8 decode GEMV. A lane's first K block is sgid * 8 + lane / 4 in the
// row-major kernel and sgid * 2 + lane / 16 in the tile-major one, so simdgroup ceil(blocks / 8)
// (row-major) or ceil(blocks / 2) (tile-major) and every one after it read nothing: each adds 32
// threads, a barrier partner and a +0.0 partial. A DSpark drafter's Markov projection has 256
// inputs, 8 K blocks: simdgroup 0 holds them all, and the row-major seat of 4 ran three idle
// simdgroups per threadgroup, 0.78 ms per projection against 0.26 ms with one. A row takes the
// simdgroups its blocks need, at most the tuned seat. No row's bits move: the simdgroups this
// drops added only +0.0 to its sum (docs/evidence/dspark/2026-09-15-step4-one-row-geometry-lfm25.md).
// IMPARO_Q8_ONE_ROW=seat is a route probe, like IMPARO_GATED_PAIR: the seat for every row.
static uint32_t q8_one_row_sgs(bool tm, uint32_t n_in) {
    static int seat_only = -1;
    if (seat_only < 0) {
        const char * e = getenv("IMPARO_Q8_ONE_ROW");
        seat_only = (e != nullptr && strcmp(e, "seat") == 0) ? 1 : 0;
    }
    const uint32_t seat = tm ? g_q8_tm_decode_sgs : g_q8_decode_sgs;
    if (seat_only == 1) { return seat; }
    const uint32_t starts = tm ? 2u : 8u;
    return std::min(seat, (n_in / Q8_BLOCK_ELEMENTS + starts - 1u) / starts);
}

// Simdgroups of a rows-matmul threadgroup: the tile-major GEMV's seat, and no more than the row
// has K blocks, since each simdgroup takes one block per step.
static uint32_t q8_rows_mma_sgs(uint32_t n_in) {
    return std::max(1u, std::min(g_q8_tm_decode_sgs, n_in / Q8_BLOCK_ELEMENTS));
}
// The Q4_0 form's: IMPARO_Q4_RM_SGS (a probe, 1..32) or 8, capped the same way.
static uint32_t q4_rows_mma_sgs(uint32_t n_in) {
    static uint32_t seat = 0u;
    if (seat == 0u) {
        const char * e = getenv("IMPARO_Q4_RM_SGS");
        const int v = e != nullptr ? atoi(e) : 0;
        seat = (v >= 1 && v <= 32) ? (uint32_t)v : 8u;
    }
    return std::max(1u, std::min(seat, n_in / 32u));
}
// Its unit tiles per threadgroup: 2 where the rows divide into pairs of tiles, else 1.
// IMPARO_Q4_RM_TILES (1, 2 or 4) overrides it for a measurement. Measured on E4B (M3 Pro,
// co-batched step ms, one run each): 4 rows 30.1 at one tile, 28.0 at two, 28.6 at four; 8 rows
// 33.0, 30.6, 31.2 (docs/evidence/cobatch/2026-09-19-q4-rows-matmul.md).
static uint32_t q4_rows_mma_tiles(uint32_t n_out) {
    static uint32_t want = 0u;
    if (want == 0u) {
        const char * e = getenv("IMPARO_Q4_RM_TILES");
        const int v = e != nullptr ? atoi(e) : 2;
        want = (v == 1 || v == 2 || v == 4) ? (uint32_t)v : 2u;
    }
    for (uint32_t t = want; t > 1u; t /= 2u) {
        if (n_out % (t * Q8_TM_UNIT_ROWS_HOST) == 0u) { return t; }
    }
    return 1u;
}

// The decode-rows GEMV for 1 << tok_log2 rows (1, 2 or 3), built on first use: a process that
// never co-batches builds none. The same constants as the one-row pipeline it widens --
// the epilogue stamp, rows per threadgroup (1 for the tile-major unit, as at init), the layout --
// plus the row count at index 28.
static id<MTLComputePipelineState> q8mv_tok_pipe(bool tm, uint32_t rows_log2, uint32_t tok_log2) {
    if (tok_log2 < 1u || tok_log2 > 3u || rows_log2 > 2u) { return nil; }
    id<MTLComputePipelineState> have = tm ? g.p_q8mv_tok_tm[tok_log2 - 1u]
                                          : g.p_q8mv_tok_rows[rows_log2][tok_log2 - 1u];
    if (have != nil || g.lib == nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    stamp_epi_act(cv);
    const uint32_t rows = tm ? 1u : 1u << rows_log2;
    [cv setConstantValue:&rows type:MTLDataTypeUInt atIndex:6];
    const uint32_t toks = 1u << tok_log2;
    [cv setConstantValue:&toks type:MTLDataTypeUInt atIndex:28];
    if (tm) {
        const bool on = true;
        [cv setConstantValue:&on type:MTLDataTypeBool atIndex:15];
    }
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_q8_0_gemv" constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: decode-rows GEMV function: %@", e); return nil; }
    id<MTLComputePipelineState> ps = [g.device newComputePipelineStateWithFunction:f error:&e];
    if (ps == nil) { NSLog(@"imparo metal: decode-rows GEMV pipeline: %@", e); return nil; }
    if (getenv("IMPARO_PIPE_LOG") != NULL) {
        NSLog(@"imparo metal pipe: imparo_q8_0_gemv tm=%d rows=%u toks=%u max_threads=%lu",
              (int)tm, rows, toks, (unsigned long)[ps maxTotalThreadsPerThreadgroup]);
    }
    if (tm) { g.p_q8mv_tok_tm[tok_log2 - 1u] = ps; }
    else    { g.p_q8mv_tok_rows[rows_log2][tok_log2 - 1u] = ps; }
    return ps;
}

// The rows matmul for `frags` token columns of 8 (1..RM_MAX_FRAGS_HOST), reading float
// activations or their half mirror, built on first use: a process that never co-batches builds
// none.
static id<MTLComputePipelineState> q8_rows_mma_pipe(uint32_t frags, bool half_x,
                                                    bool q4 = false, uint32_t tiles = 1u) {
    if (frags < 1u || frags > RM_MAX_FRAGS_HOST || (tiles != 1u && tiles != 2u && tiles != 4u)) {
        return nil;
    }
    if (!q4 && tiles != 1u) { return nil; }
    id<MTLComputePipelineState> __strong & slot =
        q4 ? g.p_q4_rows_mma[frags - 1u][half_x ? 1 : 0][tiles == 4u ? 2u : tiles - 1u]
           : g.p_q8_rows_mma[frags - 1u][half_x ? 1 : 0];
    if (slot != nil || g.lib == nil) { return slot; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    stamp_epi_act(cv);
    [cv setConstantValue:&frags type:MTLDataTypeUInt atIndex:31];
    [cv setConstantValue:&half_x type:MTLDataTypeBool atIndex:36];
    if (q4) { [cv setConstantValue:&q4 type:MTLDataTypeBool atIndex:37]; }
    if (tiles > 1u) { [cv setConstantValue:&tiles type:MTLDataTypeUInt atIndex:38]; }
    // Diagnostic only (the kernel's RM_SKIP): read once, so a process builds one kind.
    static int skip = -1;
    if (skip < 0) { const char * e = getenv("IMPARO_RM_SKIP"); skip = e != nullptr ? atoi(e) : 0; }
    if (skip != 0) {
        const uint32_t sk = (uint32_t)skip;
        [cv setConstantValue:&sk type:MTLDataTypeUInt atIndex:34];
    }
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_q8_tm_rows_mma" constantValues:cv
                                             error:&e];
    if (f == nil) { NSLog(@"imparo metal: rows matmul function: %@", e); return nil; }
    id<MTLComputePipelineState> ps = [g.device newComputePipelineStateWithFunction:f error:&e];
    if (ps == nil) { NSLog(@"imparo metal: rows matmul pipeline: %@", e); return nil; }
    if (getenv("IMPARO_PIPE_LOG") != NULL) {
        NSLog(@"imparo metal pipe: imparo_q8_tm_rows_mma frags=%u half_x=%d q4=%d tiles=%u "
              @"max_threads=%lu", frags, (int)half_x, (int)q4, tiles,
              (unsigned long)[ps maxTotalThreadsPerThreadgroup]);
    }
    slot = ps;
    return ps;
}

// The Q4_0 decode-rows GEMV: the one-row pipeline's constants as init builds them (lanes per row
// at 0, rows per lane at 2), the row count at 28 and the token split at 29. Built on first use.
static id<MTLComputePipelineState> q4mv_tok_pipe(uint32_t lanes_log2, uint32_t nr_log2,
                                                 uint32_t split_log2, uint32_t tok_log2) {
    if (lanes_log2 > 5u || nr_log2 > 3u || split_log2 > 3u || tok_log2 < 1u || tok_log2 > 3u) {
        return nil;
    }
    id<MTLComputePipelineState> have = g.p_q4mv_tok[lanes_log2][nr_log2][split_log2][tok_log2 - 1u];
    if (have != nil || g.lib == nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    const uint32_t lanes = 1u << lanes_log2, nr0 = 1u << nr_log2;
    const uint32_t split = 1u << split_log2, toks = 1u << tok_log2;
    [cv setConstantValue:&lanes type:MTLDataTypeUInt atIndex:0];
    [cv setConstantValue:&nr0 type:MTLDataTypeUInt atIndex:2];
    [cv setConstantValue:&toks type:MTLDataTypeUInt atIndex:28];
    [cv setConstantValue:&split type:MTLDataTypeUInt atIndex:29];
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_q4_0_matmat" constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: Q4_0 decode-rows function: %@", e); return nil; }
    id<MTLComputePipelineState> ps = [g.device newComputePipelineStateWithFunction:f error:&e];
    if (ps == nil) { NSLog(@"imparo metal: Q4_0 decode-rows pipeline: %@", e); return nil; }
    if (getenv("IMPARO_PIPE_LOG") != NULL) {
        NSLog(@"imparo metal pipe: imparo_q4_0_matmat lanes=%u nr0=%u split=%u toks=%u "
              @"max_threads=%lu", lanes, nr0, split, toks,
              (unsigned long)[ps maxTotalThreadsPerThreadgroup]);
    }
    g.p_q4mv_tok[lanes_log2][nr_log2][split_log2][tok_log2 - 1u] = ps;
    return ps;
}

// Whether the half-activation mirror holds `rows` rows of `n_in` halves. The GEMMs' half routes
// convert (or read) whole token tiles of it, so a route that checked only that the mirror exists
// wrote past the end of a smaller one, into whatever buffer followed it. A mirror too small for
// the padded rows turns the half route down (tests/decode_rows.rs, check_xh_guard).
static inline bool xh_holds(uint64_t rows, uint32_t n_in) {
    return g.bufs[B_XH] != nil && rows * n_in * 2ull <= g.sizes[B_XH];
}

// Q8_0 has its own dispatch: its routes (the GEMM, the rows matmul, the one-row and decode-rows
// GEMVs, the token tile), its own grid and threadgroup sizing, and a 64-bit weight offset.
// Returns false when nothing was encoded, so the caller logs the reason once instead of
// faulting inside the driver on a nil pipeline.
static bool q8_matmat(bool tm, uint64_t w_off, uint64_t w_off2, uint32_t n_in, uint32_t n_out,
                      uint32_t src, uint32_t dst, uint32_t n_tok, uint32_t src_row) {
    if (n_in == 0u || n_out == 0u || n_tok == 0u) { return false; }
    const bool gated = w_off2 != NO_PAIR;
    const uint32_t vrows = gated ? 2u * n_out : n_out;   // rows the GEMM grid walks
    if ((n_in % Q8_BLOCK_ELEMENTS) != 0u) {
        NSLog(@"imparo metal: Q8 matmat n_in=%u is not a multiple of %u; every route "
              @"walks whole blocks, so refusing rather than reading past the row",
              n_in, Q8_BLOCK_ELEMENTS);
        g_refused += 1;
        return false;
    }
    // Independent decode rows on the GEMV (g_decode_rows: always on the exact route, up to the
    // measured row count on the fast one). A gated pair falls back below to its two dispatches,
    // as a one-row decode does.
    const bool rows_gemv = decode_rows_gemv(n_tok, g_q8_rows_gemv_max);
    const uint32_t tok_log2 = n_tok <= 2u ? 1u : (n_tok <= 4u ? 2u : 3u);
    // The fast route's rows past the GEMV's, up to the rows matmul's own count (tile-major only).
    // A gated pair is refused below on this route, so the caller issues the gate and up
    // projections here and multiplies them: the FFN stays in float, where the pair's GEMM
    // would hand the down projection a half mirror of G and nothing else.
    const bool rows_mma = tm && decode_rows_route() == 2u && !rows_gemv
                       && n_tok >= 2u && n_tok <= g_q8_tm_rows_mma_max
                       && (n_out % Q8_TM_UNIT_ROWS_HOST) == 0u;
    const uint32_t mma_frags = (n_tok + 7u) / 8u;
    // Floats first; the half mirror only when a mirror-mode producer left the floats stale.
    const bool mma_half = rows_mma && src < 64u && ((g_float_stale >> src) & 1ull) != 0ull;
    if (mma_half && !(g_xh_src == src && g_xh_elems >= (uint64_t)n_tok * n_in)) {
        // Stale floats and no mirror covering the rows: nothing holds the activations.
        g_refused += 1;
        NSLog(@"imparo metal: rows matmul source %u has stale floats and no current mirror", src);
        return false;
    }
    const bool use_gemm = n_tok > g_gemv_max_tok && !rows_gemv && !rows_mma;   // one row is the decode GEMV

    // One call, no decisions inline: `q8_geometry` owns the tile rule AND the K-chunk
    // legality fallback, and `imparo_metal_q8_pick_shape` lets a test assert the rule with
    // no GPU. Everything below only consumes the geometry.
    const Q8Geometry geo = q8_geometry(n_tok, n_in);
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
    // The mirror covers whole token tiles of WHICHEVER kernel reads it.
    const uint32_t toks_q8 = rt_q8 ? rt_toks : ST_GEMM_SHAPES[shape][1];
    const uint32_t padded  = (n_tok + toks_q8 - 1u) / toks_q8 * toks_q8;
    const bool xh_fits = xh_holds(padded, n_in);
    if (gated && !(use_gemm && g_half_a && n_tok >= HALF_A_MIN && xh_fits
                   && g_epilogue != 0u && gated_pipe != nil)) {
        return false;
    }
    id<MTLComputePipelineState> half_pipe = gated ? gated_pipe
        : (rt_q8 ? g.p_rt8_h[rt_i]
                 : (full ? (tm ? g.p_stgemm_full_h_tm[shape] : g.p_stgemm_full_h[shape])
                         : (tm ? g.p_stgemm_h_tm[shape] : g.p_stgemm_h[shape])));
    if (use_gemm && g_half_a && n_tok >= HALF_A_MIN && half_pipe != nil && xh_fits) {
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
        id<MTLComputePipelineState> rt_sel = rt_for_rows(
            gated ? g.p_rt8_gh[rt_i] : (q8_half ? g.p_rt8_h[rt_i] : g.p_rt8[rt_i]), n_out);
        if (rt_sel == nil) { g_refused += 1; return false; }
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
            // rt_gemm's epilogue writes the mirror INSTEAD of the floats, gated or not.
            g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
            g_float_stale |= hb(dst);
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
    } else if (rows_mma) {
        sel = q8_rows_mma_pipe(mma_frags, mma_half);
        if (mma_half) { q8_src_bind = g_xh_buf; }
    } else if (n_tok == 1u) {
        sel = tm ? g.p_q8mv_tm : g.p_q8mv_rows[g_q8_decode_rows_log2];
    } else if (rows_gemv) {
        sel = q8mv_tok_pipe(tm, g_q8_decode_rows_log2, tok_log2);
    } else {
        sel = tm ? g.p_q8mm_tile_tm[g_q8_token_tile_log2] : g.p_q8mm_tile[g_q8_token_tile_log2];
    }
    if (sel == nil) {
        // COUNTED, not only logged. A projection that never ran leaves the previous
        // contents of its output in place: the run continues and its numbers are wrong,
        // which is exactly how a seat naming a shape init had not built survived a whole
        // A/B. The log line names the shape; this makes the engine's refusal count say it.
        g_refused += 1;
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
                      (NSUInteger)rows
                          * (toks % 16u == 0u ? toks / 2u : toks) * sizeof(float)),
                  grid_tx ? (n_tok + toks - 1) / toks : (vrows + rows - 1) / rows,
                  grid_tx ? (vrows + rows - 1) / rows : (n_tok + toks - 1) / toks,
                  n_out, n_tok, (int)gated);
        } else if (rows_mma) {
            NSLog(@"imparo metal q8: rows mma frags=%u half_x=%d sgs=%u n_in=%u n_out=%u n_tok=%u",
                  mma_frags, (int)mma_half, q8_rows_mma_sgs(n_in), n_in, n_out, n_tok);
        } else if (n_tok == 1u || rows_gemv) {
            NSLog(@"imparo metal q8: gemv rows=%u sgs=%u n_in=%u n_out=%u n_tok=%u",
                  tm ? 8u : 1u << g_q8_decode_rows_log2, q8_one_row_sgs(tm, n_in), n_in, n_out,
                  n_tok);
        } else {
            NSLog(@"imparo metal q8: tile tile=%u sgs=%u n_out=%u n_tok=%u",
                  1u << g_q8_token_tile_log2, g_q8_batch_sgs, n_out, n_tok);
        }
    }
    if (use_gemm) {
        route_count(MR_Q8_GEMM);
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
        const NSUInteger out_bytes =
            (NSUInteger)rows * (toks % 16u == 0u ? toks / 2u : toks) * sizeof(float);
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
            // instead of converting. Elems counts the FULL batch. The gated pair writes ONLY
            // the mirror; the plain epilogue writes the floats as well.
            g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
            if (gated) { g_float_stale |= hb(dst); }
        }
        return true;
    }
    if (rows_mma) {
        // One threadgroup per unit tile; its simdgroups split the K blocks and add their sums in
        // order through threadgroup memory (64 floats per 8 token columns).
        const uint32_t sgs = q8_rows_mma_sgs(n_in);
        [g.enc setThreadgroupMemoryLength:(NSUInteger)sgs * mma_frags * 64u * sizeof(float)
                                  atIndex:0];
        g_disp_seq += 1; g_rows_mma_dispatches += 1; route_count(MR_Q8_ROWS_MMA);
        if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, tm ? 3u : 2u), w_bytes(tm ? 3u : 2u, n_in, n_out)); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_out / Q8_TM_UNIT_ROWS_HOST, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(sgs * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        return true;
    }
    if (n_tok == 1u || rows_gemv) {
        // Tile-major: the unit fixes 8 rows per threadgroup (two per lane, the row-major
        // kernel's reuse of each activation load); only the simdgroup count is a knob,
        // and it is the TM kernel's own (q8_tm_decode_sgs), not the row-major one's --
        // the two layouts measured different winners (see the knob's comment). Either
        // count is capped by the simdgroups the row's width can use (q8_one_row_sgs).
        // Decode rows take the SAME simdgroups: they decide a row's K split, so a different
        // count would move its bits.
        const uint32_t rm_rows  = 1u << g_q8_decode_rows_log2;
        const uint32_t dec_rows = tm ? 8u : rm_rows;
        const uint32_t dec_sgs  = q8_one_row_sgs(tm, n_in);
        const uint32_t dec_toks = rows_gemv ? 1u << tok_log2 : 1u;
        route_count(rows_gemv ? MR_Q8_GEMV_ROWS : MR_Q8_GEMV);
        // One float per (simdgroup, row, token) for the cross-simdgroup reduction.
        [g.enc setThreadgroupMemoryLength:(NSUInteger)dec_rows * dec_sgs * dec_toks
                                         * sizeof(float) atIndex:0];
        g_disp_seq += 1;
        if (g_prof) {
            g_prof_disp += 1;
            prof_begin(n_in <= 512u && n_out >= 64u * n_in ? PC_MATMAT_RANK : PC_MATMAT_DECODE);
        }
        [g.enc dispatchThreadgroups:MTLSizeMake((n_out + dec_rows - 1) / dec_rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(dec_sgs * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        return true;
    }
    // Narrow batch: one simdgroup per output row, one threadgroup column per token tile.
    route_count(MR_Q8_TILE);
    const uint32_t tile = 1u << g_q8_token_tile_log2;
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, tm ? 3u : 2u), w_bytes(tm ? 3u : 2u, n_in, n_out)); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + g_q8_batch_sgs - 1) / g_q8_batch_sgs,
                                           (n_tok + tile - 1) / tile, 1)
          threadsPerThreadgroup:MTLSizeMake(g_q8_batch_sgs * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// Defined with the weight-kind table further down (the load-time repack section); declared
// here because matmat is their first user and C++ reads in order.
static id<MTLComputePipelineState> gather_pipeline_for_fmt(uint32_t wfmt);
enum { RT_PLAIN = 0, RT_HALF = 1, RT_GATED_HALF = 2, RT_HALF_RESID = 3 };
// Set only inside imparo_metal_matmat_resid: the dispatch adds into dst (RT_HALF_RESID). Every
// matmat_impl route that is not the tile-major register-tiled GEMM on the half mirror refuses
// while it is set, so no other kernel ever sees the request.
static bool g_mm_resid = false;
// The plain family's half-mirror entry point for `tile` (a wide RT_SHAPES index or a narrow
// tile) compiled with the residual-add store (constant 60) -- the constants its init build
// stamps, plus that one. Built on first use and kept.
static id<MTLComputePipelineState> rt_plain_resid(uint32_t tile);
static id<MTLComputePipelineState> rt_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                       uint32_t variant, uint32_t tile);
// The narrow tile the nb8_shape knob selects, as a pipeline-cache index.
static inline uint32_t rt_narrow_tile(void) { return g_nb8_shape ? RT_TILE_NB8B : RT_TILE_NB8; }
static id<MTLComputePipelineState> gemv_pipeline_for_fmt(uint32_t wfmt, bool rowmajor);

// `wkind` is the weight-type -> kernel table index (0 = F32, 1 = Q4_0, 2 = Q8_0).
// Load-time validation on the Rust side guarantees no other value arrives; the guard
// below is defense in depth, not a path.
// The block GEMV's SIMDGROUPS PER THREADGROUP, for the one-row kernel and its rows form. 8 was
// inherited from imparo_q8_0_gemv's shape, and the first ranking in this kernel's own regime kept
// it on one model's numbers. FOUR IS THE SEAT, ranked on all three, 128-160 decode steps, every
// arm warmed, the arm order rotated each round, minimum of three rounds, ms/token:
//
//   LFM2.5-8B-A1B Q4_K_M   10.352 -> 10.309   -0.56%   (a second sweep: 10.333 -> 10.321)
//   gemma-4-E4B  Q4_K_XL   23.107 -> 23.029   -0.34%
//   Qwen3.8-27B  Q4_K_S   115.675 -> 115.613  -0.05%   (a tie, and not the loss 8 was kept for)
//
// The step hash is unchanged in every arm: `sgs` only repacks the same simdgroups into more
// threadgroups, and each row's dot product stays inside one simdgroup either way.
//
// IT IS THE THREADGROUP COUNT, NOT THE PARALLELISM. Every arm of the ladder runs the same total
// simdgroups, so what moves is how many threadgroups the scheduler has to spread over eighteen
// cores: on LFM2.5 a 2048-row projection is 128 threadgroups at 8 and 256 at 4. The other axis
// says the same thing from the other side -- blk_gemv_nr 1, which DOUBLES the simdgroups and
// halves each one's work, loses 3.3% at both packings.
//
//   sgs   threadgroups (2048 rows)   LFM2.5 delta
//    16          64                   +0.42%
//     8         128                    0.00%   (was the seat)
//     4         256                   -0.56%
//     2         512                   -0.45%
//
// IMPARO_BLK_GEMV_SGS re-runs that A/B; a value outside the ladder falls back to the seat rather
// than dispatching a shape nothing built.
static uint32_t blk_gemv_sgs() {
    static uint32_t sgs_v = 0u;
    if (sgs_v == 0u) {
        const char * e = getenv("IMPARO_BLK_GEMV_SGS");
        const uint32_t w = e ? (uint32_t)atoi(e) : 4u;
        sgs_v = (w == 2u || w == 4u || w == 8u || w == 16u || w == 32u) ? w : 4u;
    }
    return sgs_v;
}
static id<MTLComputePipelineState> blk_rows_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                             uint32_t form_log2);
static id<MTLComputePipelineState> blk_rows_mma_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                                uint32_t tiles, bool xhalf,
                                                                uint32_t frags = 1u);

// The matrix-unit rows kernel's simdgroups (each takes every nsg-th sub-block) and 8-row units a
// threadgroup. Measured on Qwen3.8-27B (M3 Pro, projections a step at 4 rows): 8 simdgroups 128.3
// ms against 133.1 at 4 and 132.5 at 16; two units 143.4 against 153.8 at one and 158.1 at four
// (both before the fetch's word loads). IMPARO_BLK_MMA_SGS (1..32) and IMPARO_BLK_MMA_TILES (1, 2
// or 4) override them for a measurement. The simdgroups are capped by the row's sub-blocks.
static uint32_t blk_mma_sgs(uint32_t n_in) {
    static uint32_t seat = 0u;
    if (seat == 0u) {
        const char * e = getenv("IMPARO_BLK_MMA_SGS");
        const int v = e != nullptr ? atoi(e) : 0;
        seat = (v >= 1 && v <= 32) ? (uint32_t)v : 8u;
    }
    return std::max(1u, std::min(seat, n_in / 32u));
}
static uint32_t blk_mma_tiles() {
    static uint32_t t = 0u;
    if (t == 0u) {
        const char * e = getenv("IMPARO_BLK_MMA_TILES");
        const int v = e != nullptr ? atoi(e) : 2;
        t = (v == 1 || v == 2 || v == 4) ? (uint32_t)v : 2u;
    }
    return t;
}

// The block formats' rows on the matrix unit for n_tok (2..8) rows starting at activation row
// src_row, its outputs dst_off bytes into dst: blk_mma_tiles() 8-row units a threadgroup, their
// simdgroups splitting the sub-blocks. False when its pipeline did not build.
static bool blk_rows_mma_encode(uint32_t wkind, uint32_t wf, uint64_t w_off, uint32_t n_in,
                                uint32_t n_out, uint32_t src, uint32_t dst, uint32_t n_tok,
                                uint32_t src_row, uint64_t dst_off) {
    const uint32_t tiles = blk_mma_tiles();
    // 9..16 rows: two 8-row token fragments in one dispatch, each weight fragment decoded once
    // for both (imparo_blk_rows_mma_t2), float activations.
    const uint32_t frags = n_tok > 8u ? 2u : 1u;
    // THE ACTIVATIONS FROM THE HALF MIRROR when a producer left one covering this step's rows:
    // the norms dual-write it for the GEMM, and reading it here costs 1.2% less a step on the 27B
    // (2 / 4 / 8 rows 131.7 / 140.1 / 156.6 -> 130.1 / 138.4 / 154.9 ms in the forward).
    // OFF unless asked (IMPARO_BLK_MMA_XHALF=1): the mirror ROUNDS the activations, and these
    // kernels hold float accuracy by the user's decision. With it on, a co-batched request's top
    // ten logits move up to 0.66 from the same request decoded alone, against 0.01 in float (32
    // steps, 4 rows; the top pick did not change). The exact route reads floats either way.
    static int xh_want = -1;
    if (xh_want < 0) { const char * xe = getenv("IMPARO_BLK_MMA_XHALF"); xh_want = (xe && xe[0] == '1') ? 1 : 0; }
    // The rows this dispatch reads start at src_row, not at row 0 as every other mirror consumer
    // does, so the mirror must cover src_row + n_tok rows -- not n_tok.
    const bool xhalf = frags == 1u && xh_want == 1 && g_half_a != 0u && g_xh_buf < B_COUNT
                    && g.bufs[g_xh_buf] != nil && g_xh_src == src
                    && g_xh_elems >= (uint64_t)(src_row + n_tok) * n_in;
    id<MTLComputePipelineState> ps =
        blk_rows_mma_pipeline_for_fmt(wf, wfmt_is_rowmajor(wkind), tiles, xhalf, frags);
    if (ps == nil) { return false; }
    const uint32_t sgs = blk_mma_sgs(n_in);
    haz(hb(src) | (xhalf ? hb(g_xh_buf) : 0u), hb(dst));
    [g.enc setComputePipelineState:ps];
    const uint64_t wl = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    // The mirror at its own index, and only in the half kernel: the float entry point has no
    // argument 14.
    if (xhalf) { [g.enc setBuffer:g.bufs[g_xh_buf] offset:g.buf_off[g_xh_buf] atIndex:14]; }
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] + dst_off atIndex:2];
    [g.enc setBytes:&wl length:8 atIndex:3];
    [g.enc setBytes:&n_in length:4 atIndex:4];
    [g.enc setBytes:&n_out length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBytes:&src_row length:4 atIndex:7];
    [g.enc setBytes:&g_epilogue length:4 atIndex:13];
    // The simdgroups' sums, 64 floats a unit and token fragment, added in simdgroup order.
    [g.enc setThreadgroupMemoryLength:(NSUInteger)sgs * tiles * frags * 64u * sizeof(float)
                              atIndex:0];
    // Counted apart so a run says which activations the rows took: the kernel line of the
    // co-batch gate and the bench print it, and a route that never ran cannot be measured.
    g_disp_seq += 1; route_count(xhalf ? MR_BLK_ROWS_MMA_H : MR_BLK_ROWS_MMA);
    if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out)); }
    const uint32_t rows_per_tg = 8u * tiles;
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1u) / rows_per_tg, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32u * sgs, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// The block formats' decode-rows GEMV for n_tok (2..8) rows starting at activation row src_row,
// its outputs dst_off bytes into dst. False when its pipeline did not build.
static bool blk_rows_gemv_encode(uint32_t wkind, uint32_t wf, uint64_t w_off, uint32_t n_in,
                                 uint32_t n_out, uint32_t src, uint32_t dst, uint32_t n_tok,
                                 uint32_t src_row, uint64_t dst_off) {
    const uint32_t form_log2 = n_tok <= 2u ? 1u : (n_tok <= 4u ? 2u : 3u);
    id<MTLComputePipelineState> ps = blk_rows_pipeline_for_fmt(wf, wfmt_is_rowmajor(wkind), form_log2);
    if (ps == nil) { return false; }
    haz(hb(src), hb(dst));
    [g.enc setComputePipelineState:ps];
    const uint64_t wl = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] + dst_off atIndex:2];
    [g.enc setBytes:&wl length:8 atIndex:3];
    [g.enc setBytes:&n_in length:4 atIndex:4];
    [g.enc setBytes:&n_out length:4 atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBytes:&src_row length:4 atIndex:7];
    [g.enc setBytes:&g_epilogue length:4 atIndex:13];
    // A threadgroup covers 32 weight rows, one a lane; its simdgroups each sum a slice of the
    // row, and the slices' sums meet in threadgroup memory: one per (simdgroup, token of the
    // form, lane).
    const NSUInteger sgs = blk_gemv_sgs();
    [g.enc setThreadgroupMemoryLength:sgs * ((NSUInteger)1u << form_log2) * 32u * sizeof(float)
                              atIndex:0];
    g_disp_seq += 1; g_blk_rows_dispatches += 1; route_count(MR_BLK_GEMV_ROWS);
    if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out)); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + 31u) / 32u, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// THE BLOCK FORMATS' ROWS PAST ONE RUN OF THE ROWS KERNELS: n_tok rows in the fewest runs of at
// most BLK_ROWS_MAX_HOST rows, the runs as even as they can be (9 rows are 5 + 4, never 8 + 1), so
// every run takes a rows kernel and the weights are read once a run. Past 8 rows the GEMM took
// these before, and its padded tile is slow at a few rows: the 9-row DSpark drafter's layer
// matmuls (LFM2.5-8B-A1B) took 5.75 ms a round on it and 3.37 on the runs. Each run takes the
// decode-rows GEMV up to the format's crossing and the matrix-unit rows kernel above it. False
// when a pipeline did not build.
static bool blk_rows_runs_encode(uint32_t wkind, uint32_t wf, uint64_t w_off, uint32_t n_in,
                                 uint32_t n_out, uint32_t src, uint32_t dst, uint32_t n_tok,
                                 uint32_t src_row) {
    // 9..16 rows whose run of 8 is the matrix unit's: ONE dispatch of two 8-row token fragments,
    // each weight fragment decoded once for both (imparo_blk_rows_mma_t2).
    if (n_tok > BLK_ROWS_MAX_HOST && n_tok <= 2u * BLK_ROWS_MAX_HOST
        && BLK_ROWS_MAX_HOST > blk_rows_gemv_max_for(wf, n_in, n_out)
        && BLK_ROWS_MAX_HOST <= g_blk_rows_mma_max) {
        return blk_rows_mma_encode(wkind, wf, w_off, n_in, n_out, src, dst, n_tok, src_row, 0u);
    }
    const uint32_t runs = (n_tok + BLK_ROWS_MAX_HOST - 1u) / BLK_ROWS_MAX_HOST;
    uint32_t c = 0u;
    for (uint32_t r = 0u; r < runs; ++r) {
        // The first n_tok % runs runs take one row more.
        const uint32_t n = n_tok / runs + (r < n_tok % runs ? 1u : 0u);
        const bool ok = n > blk_rows_gemv_max_for(wf, n_in, n_out) && n <= g_blk_rows_mma_max
            ? blk_rows_mma_encode(wkind, wf, w_off, n_in, n_out, src, dst, n, src_row + c,
                                  (uint64_t)c * n_out * 4u)
            : blk_rows_gemv_encode(wkind, wf, w_off, n_in, n_out, src, dst, n, src_row + c,
                                   (uint64_t)c * n_out * 4u);
        if (!ok) { return false; }
        c += n;
    }
    return true;
}

// Rows up to this many take blk_rows_runs_encode on the fast route where the GEMM has room;
// above it the GEMM. Two runs: the widths measured (a DSpark drafter's 9-row block, verify trees
// up to 16 rows); past them the GEMM's narrow tile is unmeasured against three or more runs.
constexpr uint32_t BLK_ROWS_RUNS_MAX_HOST = 2u * BLK_ROWS_MAX_HOST;

// One encode path for the plain projection and THE GATED PAIR (`w_off2 != NO_PAIR`).
// Returns whether the request was handled -- encoded, or deliberately skipped by a
// diagnostic -- and false when the caller must issue the two-dispatch form itself.
static bool matmat_impl(uint32_t wkind, uint64_t w_off, uint64_t w_off2, uint32_t n_in,
                        uint32_t n_out, uint32_t src, uint32_t dst,
                        uint32_t n_tok, uint32_t src_row) {
    // THE BLOCK FORMATS' ROWS ON THE FAST ROUTE: up to this format's crossing the decode-rows
    // GEMV (imparo_blk_gemv_rows), up to g_blk_rows_mma_max the matrix-unit rows kernel
    // (imparo_blk_rows_mma), above both the GEMM; each weight sub-block decoded once for all the
    // rows. The first crossing is per TENSOR when the tune file seats one and per format
    // otherwise (blk_rows_gemv_max_for) -- the matrix unit's products
    // cost the same per weight tile whatever rows are live, so a format short of the bandwidth
    // wall pays them and a format at it does not. Not the exact route's: their rows differ from
    // their lone decodes in the last bits, so the exact route keeps the one-row GEMV per row
    // (below). Neither has a gated twin: a gated pair is refused, and the caller issues gate and
    // up on it, as a one-row decode does.
    const uint32_t wf = wfmt_for(wkind);
    const bool blk_gemv_rows = wf != 0u && n_tok <= blk_rows_gemv_max_for(wf, n_in, n_out);
    const bool blk_mma_rows = wf != 0u && !blk_gemv_rows && n_tok <= g_blk_rows_mma_max;
    if (decode_rows_route() == 2u && n_tok >= 2u && (blk_gemv_rows || blk_mma_rows)
        && src < B_COUNT && dst < B_COUNT) {
        // Nor a residual matmul (`imparo_metal_matmat_resid`): these kernels store W . x, not
        // resid + W . x, so taking one overwrote the residual. A co-batched step never folds
        // the residual; a speculative round's verify (a prefill-shaped forward) does.
        if (w_off2 != NO_PAIR || g_mm_resid) { return false; }
        if (g_skip_cat == PC_MATMAT_DECODE) { return true; }
        if (blk_gemv_rows
                ? blk_rows_gemv_encode(wkind, wf, w_off, n_in, n_out, src, dst, n_tok,
                                       src_row, 0u)
                : blk_rows_mma_encode(wkind, wf, w_off, n_in, n_out, src, dst, n_tok,
                                      src_row, 0u)) {
            return true;
        }
        // No pipeline: the GEMM.
    }
    // A ONE-ROW PROJECTION WHOSE ROW IS NARROWER THAN A SIMDGROUP'S LANES: the one-row GEMV's
    // lanes each take a 32-value sub-block of the row, so a row of fewer than 32 of them leaves
    // lanes idle, and the decode-rows GEMV (one weight row a lane) takes it. A DSpark drafter's
    // Markov table, 256 -> 128000 nine times a round on LFM2.5-8B-A1B: the drafter 9.9 -> 8.4 ms.
    // Measured at that output width only; a narrow output stays on the one-row GEMV.
    if (n_tok == 1u && wf != 0u && n_in < 32u * 32u && n_out >= 16384u
        && w_off2 == NO_PAIR && !g_mm_resid && src < B_COUNT && dst < B_COUNT) {
        if (blk_rows_gemv_encode(wkind, wf, w_off, n_in, n_out, src, dst, 1u, src_row, 0u)) {
            return true;
        }
    }
    // Past one run of rows, and only where a full run is the matrix unit's (its crossing at 8):
    // a lower crossing says the GEMM beats the rows kernel above it, and a run would be worse.
    if (decode_rows_route() == 2u && wf != 0u && n_tok > BLK_ROWS_MAX_HOST
        && g_blk_rows_mma_max >= BLK_ROWS_MAX_HOST && n_tok <= BLK_ROWS_RUNS_MAX_HOST
        && src < B_COUNT && dst < B_COUNT) {
        // A gated pair is refused, as above, so the caller issues gate and up on the runs. Left
        // to the GEMM, the pair writes only the half mirror of its output, and the down
        // projection on the runs reads the floats, which the pair never wrote.
        if (w_off2 != NO_PAIR || g_mm_resid) { return false; }
        if (g_skip_cat == PC_MATMAT_DECODE) { return true; }
        if (blk_rows_runs_encode(wkind, wf, w_off, n_in, n_out, src, dst, n_tok, src_row)) {
            return true;
        }
        // No pipeline: the GEMM.
    }
    // THE EXACT ROUTE'S ROWS FOR A FORMAT WITHOUT A DECODE-ROWS GEMV (every block quant): the
    // one-row decode GEMV once per row, so each row gets its lone decode's bits by
    // construction. The exact route is the gate that proves the per-row plumbing, not a speed
    // route (docs/continuous-batching.md, section 6); Q4_0 and Q8_0 have a decode-rows GEMV.
    if (decode_rows_route() == 1u && n_tok >= 2u && w_off2 == NO_PAIR && src < B_COUNT
        && dst < B_COUNT && !(wkind >= 1u && wkind <= 3u)) {
        const uint64_t src0 = g.buf_off[src], dst0 = g.buf_off[dst];
        bool ok = true;
        for (uint32_t r = 0; r < n_tok && ok; ++r) {
            g.buf_off[src] = src0 + (uint64_t)(src_row + r) * n_in * 4u;
            g.buf_off[dst] = dst0 + (uint64_t)r * n_out * 4u;
            ok = matmat_impl(wkind, w_off, NO_PAIR, n_in, n_out, src, dst, 1u, 0u);
        }
        g.buf_off[src] = src0;
        g.buf_off[dst] = dst0;
        return ok;
    }
    const bool gated = w_off2 != NO_PAIR;
    const uint32_t vrows = gated ? 2u * n_out : n_out;   // rows the rt grid walks
    // one lane per token only pays off with a batch; decode keeps the split-row kernel
    // One row takes the GEMV; above it, the GEMM family (nb8 then wide). Not a tuned
    // boundary: see the g_gemv_max_tok declaration for the identity rule that fixes it.
    if (wkind > 3u && wfmt_for(wkind) == 0u) {
        NSLog(@"imparo metal: matmat got unknown weight kind %u (n_out=%u) -- load "
              @"validation should have rejected this model; refusing the dispatch", 
              wkind, n_out);
        g_refused += 1;
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
        if (g_mm_resid) { return false; }
        if (g_skip_cat == (n_tok > g_gemv_max_tok ? PC_MATMAT_PREFILL
                                                    : PC_MATMAT_DECODE)) { return true; }
        return q8_matmat(wkind == 3u, w_off, w_off2, n_in, n_out, src, dst, n_tok, src_row);
    }
    const uint32_t wfmt = wfmt_for(wkind);
    const bool is_q4 = wkind == 1u;
    // Independent decode rows on the GEMV, as in q8_matmat. A gated pair needs the GEMM, so the
    // check below refuses it and the caller issues the two projections a one-row decode issues.
    const bool rows_gemv = is_q4 && decode_rows_gemv(n_tok, g_q4_rows_gemv_max);
    // THE FAST ROUTE'S ROWS PAST THE GEMV'S, up to the rows matmul's own count: tile-major
    // Q8's kernel (imparo_q8_tm_rows_mma) reading each row's Q4_0 blocks where they lie, each
    // weight block read once for every row. A gated pair is refused here, as on Q8's route, so
    // the caller issues gate and up on it and multiplies them in float.
    if (is_q4 && decode_rows_route() == 2u && !rows_gemv && n_tok >= 2u
        && n_tok <= g_q4_rows_mma_max && (n_out % Q8_TM_UNIT_ROWS_HOST) == 0u
        && (n_in % 32u) == 0u) {
        if (gated || g_mm_resid) { return false; }
        if (g_skip_cat == PC_MATMAT_DECODE) { return true; }
        const uint32_t frags = (n_tok + 7u) / 8u;
        // Floats first; the half mirror only when a mirror-mode producer left them stale.
        const bool half_x = src < 64u && ((g_float_stale >> src) & 1ull) != 0ull;
        if (half_x && !(g_xh_src == src && g_xh_elems >= (uint64_t)n_tok * n_in)) {
            g_refused += 1;
            NSLog(@"imparo metal: rows matmul source %u has stale floats and no current mirror",
                  src);
            return false;
        }
        const uint32_t src_bind = half_x ? g_xh_buf : src;
        const uint32_t tiles = q4_rows_mma_tiles(n_out);
        id<MTLComputePipelineState> ps = q8_rows_mma_pipe(frags, half_x, true, tiles);
        if (ps == nil) { g_refused += 1; return false; }
        haz(hb(src_bind), hb(dst));
        [g.enc setComputePipelineState:ps];
        const uint64_t wl = wbind(g.enc, w_off, 0);
        [g.enc setBuffer:g.bufs[src_bind] offset:g.buf_off[src_bind] atIndex:1];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
        [g.enc setBytes:&wl length:8 atIndex:3];
        [g.enc setBytes:&n_in length:4 atIndex:4];
        [g.enc setBytes:&n_out length:4 atIndex:5];
        [g.enc setBytes:&n_tok length:4 atIndex:6];
        [g.enc setBytes:&src_row length:4 atIndex:7];
        [g.enc setBytes:&g_epilogue length:4 atIndex:13];
        const uint32_t sgs = q4_rows_mma_sgs(n_in);
        [g.enc setThreadgroupMemoryLength:(NSUInteger)sgs * tiles * frags * 64u * sizeof(float)
                                  atIndex:0];
        g_disp_seq += 1; g_rows_mma_dispatches += 1; route_count(MR_Q4_ROWS_MMA);
        if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out)); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_out / (Q8_TM_UNIT_ROWS_HOST * tiles), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(sgs * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
        return true;
    }
    // THE TILE-MAJOR FAMILY takes the register-tiled GEMM at EVERY batch width, decode
    // included: its stage arm lives there and nowhere else yet. Correct first; whether a
    // decode GEMV is worth its own arm is a measurement, the same order Q8_0_TM followed.
    bool use_prefill = (is_q4 && n_tok > g_gemv_max_tok && !rows_gemv) || wfmt != 0u;
    if (g_mm_resid && (!use_prefill || gated || !g_rt)) { return false; }
    if (gated) {
        // The pair rides the register-tiled half-activation route only; its pipelines are
        // that route's _gh twins. Decided before anything is encoded.
        const uint32_t wide = RT_SHAPES[g_rt_shape][0] * 8u * RT_SHAPES[g_rt_shape][3];
        const bool rt_half = use_prefill && g_rt && g.p_rt[g_rt_shape] != nil
                          && g_half_a && n_tok >= HALF_A_MIN
                          && xh_holds((n_tok + wide - 1u) / wide * wide, n_in)
                          && g.p_rt_h[g_rt_shape] != nil && g.p_rt_gh[g_rt_shape] != nil
                          && g.p_rt_nb8_gh != nil && g.p_rt_nb8b_gh != nil
                          && g_epilogue != 0u;
        if (!rt_half) { return false; }
    }
    // Skips the WHOLE matmul, staging and write-back included, so it is comparable with
    // llama.cpp's GGML_METAL_SKIP_OP=MUL_MAT. IMPARO_SKIP_MMA only removes the multiplies
    // and the dequantisation from inside the kernel, which is a different quantity.
    // The class a block-quant dispatch belongs to is decided by its width, not by the
    // kernel family: one row is the decode GEMV, more is the prefill GEMM. Keyed on
    // `use_prefill` alone, IMPARO_SKIP_CAT=matmat_decode left every block-quant GEMV
    // running (the 27B's step read unchanged with the class "skipped") and the prefill
    // skip would have dropped its decode GEMVs.
    const bool decode_class = wfmt != 0u ? n_tok <= g_gemv_max_tok : !use_prefill;
    if (!decode_class && g_skip_cat == PC_MATMAT_PREFILL) { return true; }
    if (decode_class  && g_skip_cat == PC_MATMAT_DECODE)  { return true; }
    // DIAGNOSTIC: skip only the decode matmuls of ONE output width, so the class can be
    // decomposed by shape in the stream rather than by an encoder-per-dispatch profile
    // (which gives a tiny dispatch its own encoder and reads it far above its real cost).
    // Wrong answers; only the difference in time is read.
    if (decode_class && g_skip_mm_nout != 0u && n_out == g_skip_mm_nout) { return true; }
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
    // DECODE ROWS: how a lane's work is shaped (the bits do not depend on it; tests/decode_rows.rs
    // runs every shape). Per block a lane loads a scale and four payload words for each of its
    // rows and eight activation float4s for each of its tokens, so its weight loads are shared by
    // its tokens and its activation loads by its rows; it holds rows x tokens accumulators twice
    // over (acc and part). Measured on E4B (16 lanes per row), ms per forward: 2 rows 29.4 at
    // 2 rows x 2 tokens per lane against 34.0 at 2 x 1; 4 rows 35.8 at 2 x 2, 37.5 at 4 x 2, 37.8
    // at 4 x 4; 8 rows 70.1 at 4 x 4, 74.4 at 2 x 4, 84.3 at 8 x 4, 89.5 at 4 x 8. So a lane takes
    // two tokens and four accumulators up to 4 rows, four tokens and sixteen at 8; the lane groups
    // of a simdgroup split the tokens to get there.
    uint32_t mv_rows_per_sg = 0;   // rows one simdgroup covers on the decode-rows route
    if (rows_gemv) {
        const uint32_t tok_log2 = n_tok <= 2u ? 1u : (n_tok <= 4u ? 2u : 3u);
        const uint32_t groups_log2 = 5u - g_lanes_log2;      // lane groups per simdgroup
        const uint32_t lane_tok_log2 = tok_log2 <= 2u ? 1u : 2u;
        const uint32_t split_log2 = std::min(groups_log2, tok_log2 - std::min(tok_log2, lane_tok_log2));
        const uint32_t acc_log2 = tok_log2 <= 2u ? 2u : 4u;
        const uint32_t tpl_log2 = tok_log2 - split_log2;
        const uint32_t rows_log2 = std::min(3u, acc_log2 - std::min(acc_log2, tpl_log2));
        chosen = q4mv_tok_pipe(g_lanes_log2, rows_log2, split_log2, tok_log2);
        if (chosen == nil) { return false; }
        mv_rows_per_sg = ((32u / g_lanes) >> split_log2) << rows_log2;
    }
    id<MTLComputePipelineState> pre = g.p_q4mm_pre;
    // The register-tiled kernel takes precedence and implements `skip` ITSELF via its
    // uniform. Selecting p_q4mm_pre_nomma on any nonzero skip meant every skip measurement
    // silently ran a DIFFERENT kernel -- which is why bits 1, 2 and 4 all produced the same
    // number, and why that was misread as the compiler eliminating dead code.
    // THE TILE-MAJOR FAMILY LIVES IN THE REGISTER-TILED GEMM AND NOWHERE ELSE YET, so the
    // route is only legal when that GEMM is the one being dispatched. Binding its pipeline
    // while the host dispatches the OTHER prefill kernel's grid and uniforms is a silently
    // wrong answer -- which is exactly what happened: the dump inside `if (g_rt)` never
    // printed while the pipeline log did.
    if (wfmt != 0u && !g_rt) {
        static bool said = false;
        if (!said) {
            said = true;
            NSLog(@"imparo metal: weight format %u needs the register-tiled GEMM, which is "
                  @"off (g_rt=0) -- refusing rather than dispatching it on another kernel's "
                  @"grid. Set IMPARO_RT=1 or tune rt on.", wfmt);
        }
        g_refused += 1;
        return false;
    }
    // A TILE-MAJOR WEIGHT TAKES THE GEMV WHEN THE GEMM HAS NO ROOM, or when there are too
    // few tokens for a padded tile to pay. The register-tiled write-back assumes WHOLE
    // token tiles (rt_toks = 64): sent a 1-token dispatch it writes 64 tokens' worth of
    // rows into a destination sized for one and lands in the NEXT allocation. Measured, it
    // overwrote the mega-kernel's sync buffer with float data, which the failsafe reported
    // as
    //     "barrier TIMED OUT ... err=7fc00000 ... arrivals in it 192 of 32"
    // (0x7fc00000 is NaN's bit pattern). The symptom was a NONDETERMINISTIC count of finite
    // logits; GPU shader validation saw nothing, because the write is in bounds of SOME
    // buffer. The lm head is n_tok = 1 by construction, so this arm is not an optimisation.
    //
    // UNTIL 2026-09-09 THIS ARM WAS TAKEN AT `n_tok < 64`, which named a token count where
    // the requirement is a destination big enough. That cost Qwen3.8-27B 6965 ms on a
    // 63-token prefill against 713 on a 64-token one, because the GEMV re-reads the whole
    // weight stream per token. See `tile_fits` below and g_gemv_max_tok's declaration.
    //
    // DIAGNOSTIC: IMPARO_BLK_GEMV_ALWAYS=1 sends EVERY tile-major matmul through the GEMV.
    // The GEMV and the GEMM read the same bytes with the same brick, so they must agree;
    // running a whole forward both ways is the equivalence test for this arm, and it is
    // the reference the sub-tile crossing was checked against.
    static int gemv_always = -1;
    if (gemv_always < 0) { gemv_always = getenv("IMPARO_BLK_GEMV_ALWAYS") != nullptr; }
    // WHAT THE GEMM ACTUALLY NEEDS IS ROOM, NOT A TOKEN COUNT. It reads and writes whole
    // 64-token tiles, so a sub-tile dispatch is legal exactly when both operands hold a
    // padded tile -- which a prefill chunk does (activations are allocated at max_batch)
    // and the lm head does not (one row of vocab). `g.sizes[id]` is the placed byte size
    // of that id, so the question is answered per dispatch instead of guessed from n_tok.
    const uint32_t rt_pad = RT_SHAPES[g_rt_shape][0] * 8u * RT_SHAPES[g_rt_shape][3];
    const uint32_t pad_tok = ((n_tok + rt_pad - 1u) / rt_pad) * rt_pad;
    const uint64_t need_dst = (uint64_t)pad_tok * n_out * 4ull;
    const uint64_t need_src = (uint64_t)(src_row + pad_tok) * n_in * 4ull;
    const bool tile_fits = src < B_COUNT && dst < B_COUNT
                        && (uint64_t)g.sizes[dst] >= need_dst
                        && (uint64_t)g.sizes[src] >= need_src;
    const bool rowmajor = wfmt_is_rowmajor(wkind);
    if (wfmt != 0u && (n_tok <= g_gemv_max_tok || !tile_fits || gemv_always)) {
        if (g_mm_resid) { return false; }
        id<MTLComputePipelineState> gv = gemv_pipeline_for_fmt(wfmt, rowmajor);
        if (gv == nil) { return false; }
        if (gated) { return false; }   // the pair has no GEMV twin; two dispatches instead
        // A CO-BATCHED STEP'S ROWS WHERE THE GEMM HAS NO ROOM (the lm head: its logits hold the
        // step's rows, not a padded tile). The decode-rows GEMV or the matrix-unit rows kernel
        // reads the weights once for up to 8 rows, by the same crossings as above; the one-row
        // kernel below would read them once per row -- on Qwen3.8-27B the 1 GB lm head four times
        // a 4-row step.
        if (n_tok > 1u && decode_rows_route() == 2u && !gemv_always) {
            const bool ok = blk_rows_runs_encode(wkind, wfmt, w_off, n_in, n_out, src, dst,
                                                 n_tok, src_row);
            // A pipeline that did not build: the token loop below redoes every row.
            if (ok) { return true; }
        }
        if (n_tok > 1u) {
            // The one-row kernel loops the tokens outermost: every weight is read again per
            // token. It runs here only because the GEMM cannot (no room for its padded tile, or
            // IMPARO_BLK_GEMV_ALWAYS), so it is counted as a fallback and said once per shape.
            route_count(MR_BLK_GEMV_TOKENS_FALLBACK);
            static uint64_t said[256]; static uint32_t n_said = 0u;
            const uint64_t key = ((uint64_t)wkind << 48) | ((uint64_t)n_out << 24) | n_in;
            bool fresh = true;
            for (uint32_t i = 0; i < n_said; ++i) { if (said[i] == key) { fresh = false; break; } }
            if (fresh && n_said < 256u) {
                said[n_said++] = key;
                NSLog(@"imparo metal: FALLBACK kind=%u %u->%u n_tok=%u: the one-row block GEMV "
                      @"loops the tokens (%s)", wkind, n_in, n_out, n_tok,
                      gemv_always ? "IMPARO_BLK_GEMV_ALWAYS" : "no room for the GEMM's padded tile");
            }
        } else {
            route_count(MR_BLK_GEMV);
        }
        haz(hb(src), hb(dst));
        [g.enc setComputePipelineState:gv];
        const uint64_t wl = wbind(g.enc, w_off, 0);
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
        [g.enc setBytes:&wl length:8 atIndex:3];
        [g.enc setBytes:&n_in length:4 atIndex:4];
        [g.enc setBytes:&n_out length:4 atIndex:5];
        [g.enc setBytes:&n_tok length:4 atIndex:6];
        [g.enc setBytes:&src_row length:4 atIndex:7];
        [g.enc setBytes:&g_epilogue length:4 atIndex:13];
        const NSUInteger sgs = blk_gemv_sgs();         // simdgroups per threadgroup
        const NSUInteger rows = sgs * blk_gemv_nr();   // rows one threadgroup covers
        NSUInteger tgs = (n_out + rows - 1) / rows;
        // PROBE: force the threadgroup count, which only covers n_out when the kernel was
        // compiled with the stride loop (IMPARO_BLK_GEMV_STRIDE=1). Setting one without
        // the other would silently drop rows, so it refuses.
        static NSUInteger forced = 0;
        static bool forced_ok = false;
        if (forced == 0) {
            const char * e = getenv("IMPARO_BLK_GEMV_TGS");
            forced = (e != nullptr && atoi(e) > 0) ? (NSUInteger)atoi(e) : (NSUInteger)-1;
            forced_ok = getenv("IMPARO_BLK_GEMV_STRIDE") != nullptr;
            if (forced != (NSUInteger)-1 && !forced_ok) {
                NSLog(@"imparo metal: IMPARO_BLK_GEMV_TGS needs IMPARO_BLK_GEMV_STRIDE=1; ignored");
            }
        }
        if (forced != (NSUInteger)-1 && forced_ok && forced < tgs) { tgs = forced; }
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out)); }
        [g.enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
        if (g_prof) { prof_end(); }
        // DIAGNOSTIC: encode the SAME dispatch a second time. The matmul's epilogue is an
        // activation, not an accumulate, so a second copy writes the same values and the
        // answer is unchanged -- the step hash is the check. This prices one output width
        // ADDITIVELY, which IMPARO_SKIP_MM_NOUT does not: skipping four widths there summed
        // to 2.33 ms against the class's own 4.23, so each skip is a lower bound. A
        // duplicate adds exactly its own cost and nothing else.
        if (g_dup_mm_nout != 0u && n_out == g_dup_mm_nout) {
            haz(hb(src), hb(dst));
            [g.enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32 * sgs, 1, 1)];
        }
        return true;
    }
    if (wfmt != 0u) {
        pre = rt_pipeline_for_fmt(wfmt, rowmajor, RT_PLAIN, g_rt_shape);
        // Dedup by the SHAPE too, not just the format: keying on (fmt, layout) alone hid
        // every projection after the first of a format and made a running route look
        // absent. An absent line has to mean absent.
        if (getenv("IMPARO_WFMT_LOG")) {
            static uint64_t seen_mm[64]; static uint32_t n_seen = 0u;
            const uint64_t key = ((uint64_t)wkind << 48) | ((uint64_t)n_out << 24)
                               | (uint64_t)n_in | ((uint64_t)gated << 63);
            bool fresh = true;
            for (uint32_t i = 0; i < n_seen; ++i) { if (seen_mm[i] == key) { fresh = false; break; } }
            if (fresh && n_seen < 64u) {
                seen_mm[n_seen++] = key;
                NSLog(@"imparo metal: matmat kind=%u fmt=%u %s n_in=%u n_out=%u n_tok=%u%s",
                      wkind, wfmt, rowmajor ? "row-major" : "tile-major",
                      n_in, n_out, n_tok, gated ? " GATED" : "");
            }
        }
    }
    else if (g_rt && g.p_rt[g_rt_shape] != nil)            { pre = g.p_rt[g_rt_shape]; }
    else if (g_skip_mma && g.p_q4mm_pre_nomma != nil)      { pre = g.p_q4mm_pre_nomma; }
    id<MTLComputePipelineState> sel =
        use_prefill ? rt_for_rows(pre, n_out) : (is_q4 ? chosen : g.p_f32mm);
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
    route_count(use_prefill ? (wfmt != 0u ? MR_BLK_GEMM : MR_Q4_GEMM)
                            : (rows_gemv ? MR_Q4_GEMV_ROWS : MR_Q4_GEMV));
    if (use_prefill) {
        // Only the weight tile is staged; activations are read from device memory and
        // the write-back reuses the same block. Keep this well under 32 KB or occupancy
        // collapses. The register-tiled variant stages RT_K x RT_WS, half as much.
        // Narrow-N routing (task #11): batches of 2..g_nb8_max tokens take a 64x8
        // tile. Same k-order into every output as the wide tile, so the swap is
        // BIT-IDENTICAL; the boundary and the tile variant are tuner-owned
        // (hostconfig v10), swept under shipping routing.
        //
        // THE NARROW TILE SERVES EVERY FAMILY THE WIDE ONE DOES. `rt_wide` says the
        // dispatch is on the register-tiled route of ITS family -- the plain family's
        // pipeline built at init, or the format's, compiled on first use -- and
        // `rt_narrow` hands back that family's narrow twin of a variant. Until
        // 2026-09-11 both this test and the tail split's asked for the plain family's
        // wide pipeline by identity, so a block-quant batch of 2..nb8_max rows never
        // reached the narrow tile: the 27B's 14-row final chunk ran the 64-token tile
        // padded (docs/evidence/bracket/2026-09-11-27b-narrow-tile-every-family.md).
        const bool rt_wide = pre != nil
            && pre == (wfmt != 0u ? rt_pipeline_for_fmt(wfmt, rowmajor, RT_PLAIN, g_rt_shape)
                                  : g.p_rt[g_rt_shape]);
        auto rt_narrow = [&](uint32_t variant) -> id<MTLComputePipelineState> {
            if (wfmt != 0u) { return rt_pipeline_for_fmt(wfmt, rowmajor, variant, rt_narrow_tile()); }
            switch (variant) {
                case RT_HALF:       return g_nb8_shape ? g.p_rt_nb8b_h  : g.p_rt_nb8_h;
                case RT_HALF_RESID: return rt_plain_resid(rt_narrow_tile());
                case RT_GATED_HALF: return g_nb8_shape ? g.p_rt_nb8b_gh : g.p_rt_nb8_gh;
                default:            return g_nb8_shape ? g.p_rt_nb8b    : g.p_rt_nb8;
            }
        };
        const bool nb8 = n_tok >= 2 && n_tok <= g_nb8_max && rt_wide
                      && rt_narrow(RT_PLAIN) != nil;
        if (nb8) {
            pre = rt_narrow(RT_PLAIN);
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
                                 && tail_r >= 1u && tail_r <= g_nb8_max && rt_wide
                                 && rt_narrow(RT_PLAIN) != nil
                                 && rt_narrow(gated ? RT_GATED_HALF : RT_HALF) != nil;
            const uint32_t main_tok = tail_split ? n_tok - tail_r : n_tok;
            // Every prefill width takes this path (HALF_A_MIN, declared above with the
            // measurement): the batch's width must not choose the precision, or a
            // resumed pass stops reproducing the cold one.
            // The widest tile's padding, as the gated pair's precondition checks it: the two
            // must agree, or a pair admitted there would run here without its half operand.
            bool half_route = false;
            if (g_half_a && n_tok >= HALF_A_MIN && g.p_rt_h[g_rt_shape] != nil
                && xh_holds((n_tok + wide_toks - 1u) / wide_toks * wide_toks, n_in)) {
                half_route = true;
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
                const uint32_t hv = gated ? RT_GATED_HALF : (g_mm_resid ? RT_HALF_RESID : RT_HALF);
                if (wfmt != 0u) {
                    rt_sel = nb8 ? rt_narrow(hv)
                                 : rt_pipeline_for_fmt(wfmt, rowmajor, hv, g_rt_shape);
                    if (rt_sel == nil) { return false; }   // refuse; never the wrong kernel
                } else {
                    rt_sel = nb8 ? rt_narrow(hv)
                                 : (gated ? g.p_rt_gh[g_rt_shape]
                                          : (g_mm_resid ? rt_plain_resid(g_rt_shape)
                                                        : g.p_rt_h[g_rt_shape]));
                    if (rt_sel == nil) { return false; }   // refuse; never the wrong kernel
                }
                src_bind = g_xh_buf;
            }
            // The residual add has only the half-mirror twin; the float route refuses it.
            if (g_mm_resid && !half_route) { return false; }
            rt_sel = rt_for_rows(rt_sel, n_out);
            if (rt_sel == nil) { g_refused += 1; return false; }
            // RT_HALF_RESID reads dst as well as writing it.
            haz(hb(src_bind) | (g_mm_resid ? hb(dst) : 0u), hb(dst));   // behind the mirror's writer, or the float producer
            [g.enc setComputePipelineState:rt_sel];
            const uint64_t w_off2_l = wlocal(w_off2, w_off);   // gate and up: one layer, one segment
            const uint64_t w_off_l = wbind(g.enc, w_off, 0);
            [g.enc setBuffer:g.bufs[src_bind] offset:g.buf_off[src_bind] atIndex:1];
            [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
            [g.enc setBytes:&w_off_l length:8 atIndex:3];
            [g.enc setBytes:&w_off2_l length:8 atIndex:10];
            [g.enc setBytes:&n_in length:4 atIndex:4];
            [g.enc setBytes:&n_out length:4 atIndex:5];
            // WFMT_DUMP: what the TM GEMM was ASKED to do. A dispatch that logs its
            // shape and then leaves its destination untouched is the difference between
            // "decoded wrong" and "never ran", and only this tells them apart.
            if (wfmt != 0u && getenv("IMPARO_WFMT_DUMP")) {
                static uint32_t n = 0u;
                if (n < 6u) {
                    n += 1u;
                    NSLog(@"imparo metal: TM dispatch fmt=%u n_in=%u n_out=%u vrows=%u "
                          @"rt_rows=%u main_tok=%u rt_toks=%u src=%u dst=%u epi=%u",
                          wfmt, n_in, n_out, vrows, rt_rows, main_tok, rt_toks, src, dst,
                          g_epilogue);
                }
            }
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
                id<MTLComputePipelineState> tp = rt_for_rows(
                    rt_narrow(gated ? RT_GATED_HALF
                                    : (half_sel ? (g_mm_resid ? RT_HALF_RESID : RT_HALF) : RT_PLAIN)),
                    n_out);
                if (tp == nil) { g_refused += 1; return false; }
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
                // read it. Elems counts the FULL batch, not the last dispatch's part. The
                // floats were not written (rt_gemm's epilogue writes the mirror instead).
                g_xh_src = dst; g_xh_elems = (uint64_t)n_tok * n_out; g_xh_buf = B_XH2;
                g_float_stale |= hb(dst);
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
        // A NARROW PROJECTION LEAVES MOST OF THE DEVICE EMPTY. `sgs` output rows share a
        // threadgroup, so the router's own gate -- 2048 -> 32, once a routed layer -- is
        // FOUR threadgroups on an eighteen-core GPU, and reads 13.1 GB/s against the
        // output head's 132. Shrinking the group puts the SAME rows on more cores:
        // imparo_f32_matmat gives each output row one simdgroup and takes its row from
        // `tgid.x * nsg + sgid`, so the row's lane partition and its simd_sum are
        // untouched and NO BIT MOVES -- only which threadgroup hosts the row changes.
        //
        // The F32 GEMV only. The Q4 decode kernels tie threadgroup memory and their row
        // tiles to sgs, so the same reshape there is a different change with its own A/B.
        // IMPARO_NARROW_SPREAD=0 keeps the old grid so the two can be timed in one binary.
        // THE K-SPLIT GEMV when the rows alone cannot fill the device. One threadgroup an
        // output row, its simdgroups splitting the dot product -- see
        // imparo_f32_gemv_ksplit. Bit-affecting, so it is a route with an A/B:
        // IMPARO_NARROW_SPREAD=0 keeps the one-simdgroup-per-row kernel.
        static int spread = -1;
        if (spread < 0) {
            const char * e = getenv("IMPARO_NARROW_SPREAD");
            spread = !(e != nullptr && e[0] == '0');
        }
        const NSUInteger cores = g_gpu_cores > 0u ? g_gpu_cores : 1u;
        // ONE ROW ONLY (decode). The split IS the row's arithmetic (its K partition and the
        // order its partials add), and its count followed the batch: a router row got different
        // bits at 1, 2, 3-4 and 5+ rows, so a 3-token prefill tail disagreed with the same
        // tokens inside a 512-token chunk, and a 4-row verify with an 8-row one, where a near
        // tie picks another expert. Every count from 2 now takes the one-simdgroup-per-row
        // kernel prefill takes (and its many-token twin, bit-identical to it). Deriving the
        // split from the output rows alone for every count instead cost an 8444-token
        // LFM2.5-8B-A1B prefill 5.6% (5347 -> 5645 ms).
        if (spread && !is_q4 && !rows_gemv && g.p_f32gemv != nil && n_out > 0u && n_tok == 1u
            && (NSUInteger)n_out < cores * 8u && n_in >= 256u) {
            // Enough simdgroups to give every core a few, bounded by the K it has to
            // split and by the threadgroup width the device allows.
            // HOW MANY SIMDGROUPS A CORE WANTS, measured on LFM2.5-8B-A1B's router gate
            // (2048 -> 32, M3 Pro, 18 cores), reading the matmat_narrow class alone:
            //
            //     per core     2       8      32     128
            //     GB/s      21.8    34.0    47.0    40.9
            //
            // It climbs while the extra simdgroups hide load latency and turns over when
            // the threadgroup outgrows 512 threads: 128 asks for 64 simdgroups, the
            // device clamps to 32, and one core's registers are split too many ways. This
            // is a measured seat, not a derived one -- it belongs in the tuner, and until
            // it is there IMPARO_NARROW_SG_PER_CORE moves it.
            static int sg_per_core = -1;
            if (sg_per_core < 0) {
                const char * e = getenv("IMPARO_NARROW_SG_PER_CORE");
                sg_per_core = e != nullptr ? atoi(e) : 32;
                if (sg_per_core < 1) { sg_per_core = 1; }
            }
            NSUInteger want = (cores * (NSUInteger)sg_per_core + (NSUInteger)n_out - 1)
                            / (NSUInteger)n_out;
            NSUInteger nsg = 1;
            while (nsg * 2u <= want && nsg * 2u * 32u <= (NSUInteger)n_in
                   && nsg * 2u * 32u <= g.p_f32gemv.maxTotalThreadsPerThreadgroup) {
                nsg *= 2u;
            }
            [g.enc setComputePipelineState:g.p_f32gemv];
            const uint64_t kw = wbind(g.enc, w_off, 0);
            [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
            [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
            [g.enc setBytes:&kw length:8 atIndex:3];
            [g.enc setBytes:&n_in length:4 atIndex:4];
            [g.enc setBytes:&n_out length:4 atIndex:5];
            [g.enc setBytes:&n_tok length:4 atIndex:6];
            [g.enc setBytes:&src_row length:4 atIndex:7];
            [g.enc setThreadgroupMemoryLength:nsg * sizeof(float) atIndex:0];
            haz(hb(src), hb(dst));
            g_disp_seq += 1;
            if (g_prof) {
                g_prof_disp += 1;
                prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out));
            }
            [g.enc dispatchThreadgroups:MTLSizeMake(n_out, n_tok, 1)
                  threadsPerThreadgroup:MTLSizeMake(32u * nsg, 1, 1)];
            if (g_prof) { prof_end(); }
            return true;
        }
        // MANY TOKENS INTO AT MOST 32 OUTPUTS: one simdgroup holds every output of its
        // tokens and the threadgroup stages the weight once for all of them -- see
        // imparo_f32_narrow_mm. BIT-IDENTICAL to imparo_f32_matmat, so the boundary only
        // decides speed: it needs a threadgroup for every core, or the old grid, which
        // spreads the same work over n_out times as many simdgroups, wins.
        // Eight simdgroups of two tokens each (imparo_f32_narrow_mm).
        const NSUInteger f32n_sgs = 8u;
        const NSUInteger f32n_tg_tok = f32n_sgs * 2u;
        id<MTLComputePipelineState> f32n_ps = g.p_f32nmm;
        if (!is_q4 && !rows_gemv && sel == g.p_f32mm && f32n_ps != nil
            && n_out <= 32u && (n_in % 128u) == 0u && (w_off_l % 16u) == 0u
            && (n_tok + f32n_tg_tok - 1u) / f32n_tg_tok >= cores) {
            [g.enc setComputePipelineState:f32n_ps];
            haz(hb(src), hb(dst));
            g_disp_seq += 1;
            if (g_prof) {
                g_prof_disp += 1;
                prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out));
            }
            [g.enc dispatchThreadgroups:MTLSizeMake((n_tok + f32n_tg_tok - 1u) / f32n_tg_tok, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(32u * f32n_sgs, 1, 1)];
            if (g_prof) { prof_end(); }
            return true;
        }
        const NSUInteger dsgs = sgs;
        NSUInteger rows_per_tg = dsgs * (is_q4 ? (32 / g_lanes) * nr0 : 1);
        // The kernel's TOKEN_TILE, from the same constant the shader is compiled with.
        NSUInteger tile = is_q4 ? Q4_TOKEN_TILE : 1;
        if (rows_gemv) {
            // One threadgroup covers its rows for every token.
            rows_per_tg = dsgs * mv_rows_per_sg;
            tile = n_tok;
        }
    haz(hb(src), hb(dst));
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(mm_decode_cat(n_out, wkind), w_bytes(wkind, n_in, n_out)); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1) / rows_per_tg,
                                                (n_tok + tile - 1) / tile, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * dsgs, 1, 1)];
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
    // ONE FORMAT FOR BOTH TENSORS: the pair kernel stages gate and up through one WFMT.
    // The plain families (Q4_0, the Q8_0 pair) and every tile-major format qualify; a
    // layer whose gate and up were quantised differently keeps the two-projection path.
    // Until 2026-09-11 only the plain kinds passed here, so a block-quant model ran gate,
    // up with the activation epilogue reading G back, then down -- three passes over the
    // activations where the pair makes one (docs/evidence/bracket/2026-09-11-27b-gated-pair.md).
    if (gate_kind != up_kind) { return 0u; }
    // A one-row decode never runs the pair, so the exact route's rows do not either.
    if (decode_rows_route() == 1u) { return 0u; }
    const bool plain = gate_kind == 1u || gate_kind == 2u || gate_kind == 3u;
    if (!plain && wfmt_for(gate_kind) == 0u) { return 0u; }
    if (n_tok < 2u || g_epi_act == 0u) { return 0u; }
    // ONE BOUND BUFFER FOR BOTH WEIGHTS, so the pair has to live in one segment. With the
    // whole model in one segment every pair does, which is why this held until a tiered
    // placement existed: `fit` merges a layer's spans only where the FILE is contiguous, so a
    // slow tier can split a layer -- and then gate and up land in different segments. The
    // two-projection path computes the same thing, so refuse rather than abort in `wlocal`.
    if (!w_pair_in_one_segment(gate_off, up_off)) {
        static bool said = false;
        if (!said) {
            said = true;
            NSLog(@"imparo metal: the gated pair straddles a weight segment (gate %llu, up "
                  @"%llu); every such pair takes the two-projection path",
                  (unsigned long long)gate_off, (unsigned long long)up_off);
        }
        return 0u;
    }
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
        g_refused += 1;
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
// THE NORM'S STAGED TWIN. Same kernel, compiled with the threadgroup stage on: the
// pre-add order reads the residual and the source twice each, and staging their sum makes
// it once. Built on first use, because it is a second pipeline of a kernel that already
// exists and most models never take this path.
//
// STAGE_OFF is 32 floats -- the most reduction partials any threadgroup width here can
// need (1024 threads is 32 simdgroups) -- so the offset is a compile-time constant and the
// stage always starts in the same place.
static const uint32_t RMS_STAGE_OFF = 32u;
static id<MTLComputePipelineState> rms_staged_pipeline() {
    static bool tried = false;
    if (!tried) {
        tried = true;
        if (g.lib != nil) {
            MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
            const bool on = true;
            [cv setConstantValue:&on type:MTLDataTypeBool atIndex:56];
            [cv setConstantValue:&RMS_STAGE_OFF type:MTLDataTypeUInt atIndex:57];
            g.p_rms_staged = make_with(g.lib, @"imparo_rms_norm", cv);
        }
    }
    return g.p_rms_staged;
}

// Whether this norm should take it: the pre-add order (the only one that reads two rows
// twice), a width the stage fits under the 32 KB threadgroup limit, and few enough rows
// that a core hosts ONE threadgroup -- the occupancy the stage costs is occupancy nothing
// else was going to use. IMPARO_RMS_STAGE=0 keeps the unstaged kernel for the A/B.
static bool rms_stage_on(uint32_t width, uint32_t n_row) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("IMPARO_RMS_STAGE"); on = !(e != nullptr && e[0] == '0'); }
    if (!on || (width % 4u) != 0u) { return false; }
    if ((uint64_t)(RMS_STAGE_OFF + width) * sizeof(float) > 32768ull) { return false; }
    const uint32_t cores = g_gpu_cores > 0u ? g_gpu_cores : 1u;
    return (uint64_t)n_row <= cores;
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
    const bool staged = rms_stage_on(width, n_row) && rms_staged_pipeline() != nil;
    [g.enc setComputePipelineState:staged ? g.p_rms_staged : g.p_rms];
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
    // The reduction partials, plus the staged row when the staged twin is bound.
    [g.enc setThreadgroupMemoryLength:
        tg_bytes16((staged ? (RMS_STAGE_OFF + width) : (threads / 32)) * sizeof(float))
                              atIndex:0];
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
    // A row-gathered tensor keeps the row-major layout, so its FORMAT is the file's type
    // and `wfmt_for` (which answers for tile-major kinds) does not apply: ask the table.
    const uint32_t src_t = (g_wire_ggml_set && wkind < 64u) ? g_wire_ggml[wkind] : 0u;
    uint32_t gfmt = 0u;
    // Every format whose row-major block is ONE scale run and one payload run. The
    // multi-span formats (Q2_K, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) have no row-major scale
    // pointer, which is why `serves_weight_type` refuses them row-major.
    switch (src_t) {
        case 3: case 6: case 7: case 11: case 12: case 13: case 14: case 16: case 19:
        case 20: case 23: case 29: gfmt = src_t; break;
        default: break;
    }
    if (gfmt == 0u && (wkind > 2u || g.p_gather[wkind] == nil)) {
        NSLog(@"imparo metal: gather_rows has no kernel for weight kind %u", wkind);
        return 1;
    }
    // Four values per thread, so the row and its destination base must both be float4
    // aligned. The quantised kinds already need width % 32 == 0, which implies it.
    const uint32_t block = gfmt != 0u ? 32u
                        : (wkind == 0u ? 4u : (wkind == 1u ? 32u : Q8_BLOCK_ELEMENTS));
    if (width == 0u || (width % block) != 0u || (width % 4u) != 0u
        || (dst_off % 4u) != 0u || n_rows == 0u || table_rows == 0u) {
        NSLog(@"imparo metal: gather_rows refused (kind=%u width=%u dst_off=%u n_rows=%u "
              @"table_rows=%u): needs width a multiple of %u and of 4, and a 4-aligned "
              @"destination", wkind, width, dst_off, n_rows, table_rows, block);
        return 1;
    }
    haz(hb(idx_buf), hb(dst));
    id<MTLComputePipelineState> gp = gfmt != 0u ? gather_pipeline_for_fmt(gfmt)
                                                : g.p_gather[wkind];
    // WHAT ACTUALLY ENGAGED. An empty log is not proof a route ran; say it once.
    if (getenv("IMPARO_WFMT_LOG")) {
        static uint32_t seen = 0u;
        if ((seen & (1u << (gfmt & 31u))) == 0u) {
            seen |= 1u << (gfmt & 31u);
            NSLog(@"imparo metal: gather_rows kind=%u fmt=%u width=%u rows=%u", wkind,
                  gfmt, width, n_rows);
        }
    }
    if (gp == nil) { return 1; }
    [g.enc setComputePipelineState:gp];
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
    // The brick kernel walks 32-value SUB-BLOCKS, one per thread; the legacy kinds walk
    // float4s. Different grid, same dispatch.
    [g.enc dispatchThreads:MTLSizeMake(gfmt != 0u ? width / 32u : width / 4u, n_rows, 1)
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

extern "C" void imparo_metal_scale(uint32_t a, float k, uint32_t n);

// The qcomb head-dim slot of `head_dim`: its index in g_qcomb_hds, QCOMB_HD_SLOTS when no slot
// holds it. The decode kernels are built per slot too.
static uint32_t attn_hd_slot(uint32_t head_dim) {
    for (uint32_t i = 0; i < QCOMB_HD_SLOTS; ++i) {
        if (g_qcomb_hds[i] == head_dim) { return i; }
    }
    return QCOMB_HD_SLOTS;
}
// The positions one decode query at `start_pos` attends to.
static inline uint32_t attn_decode_span(uint32_t start_pos, uint32_t window) {
    return (window > 0 && start_pos + 1 > window) ? window : start_pos + 1;
}
// Diagnostic (imparo_metal_set_skip_attn): whether attention at `head_dim` is left out.
static inline bool attn_skipped(uint32_t head_dim) {
    return g_skip_attn == 1u || (g_skip_attn == 2u && head_dim == 512u)
        || (g_skip_attn == 3u && head_dim == 256u);
}
// THE VECTOR DECODE ROUTE'S TERMS: one query over an f16 cache, a span within the kernel's
// regime. imparo_metal_attention takes the route on exactly these terms; a co-batched step
// reads them to know each row's attention is one dispatch with no scratch.
static bool attn_vec_serves(uint32_t head_dim, uint32_t n_heads, uint32_t n_kv, uint32_t n_pos,
                            bool kv_quant) {
    const uint32_t slot = attn_hd_slot(head_dim);
    return !kv_quant && slot < 2u && g.p_attn_dec_vec[slot] != nil && n_kv > 0u
        && n_heads % n_kv == 0u && n_pos <= g_attn_vec_max_keys && attn_vec_enabled();
}

extern "C" void imparo_metal_attention(uint32_t kv_layer, uint32_t head_dim, uint32_t n_heads,
                                       uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
                                       uint32_t window, uint32_t n_tok, uint32_t max_scores,
                                       uint32_t ring, float scale) {
    // Diagnostic only; output is wrong on purpose. 1 = skip all attention,
    // 2 = skip only the hd-512 (full-attention) layers, 3 = skip only hd-256.
    if (attn_skipped(head_dim)) { return; }

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
    const uint32_t slot_of = attn_hd_slot(head_dim);
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
    // The register-softmax form where it is built (see fars_route_on).
    const bool fars = fars_route_on() && g.p_fars != nil;
    if (g_attn_fa && slot_of == 0u && (p_fa != nil || fars) && ring == 0u && n_tok > 1u) {
        const uint32_t QB = fars ? 32u : FA_QB;
        const uint32_t NSG_FA = fars ? 4u : fa_nsg_live, THREADS = NSG_FA * 32u;
        const uint32_t CB_FA = FA_CB;
        // 8-query op: sq (QB x hd half) + ss (QB x CB float) + so (QB x hd float).
        // Register-softmax op: two 32-key blocks of (hd + 8)-half rows, in floats.
        const uint32_t sfloats = fars ? (2u * 32u * (head_dim + 8u)) / 2u
                                      : (QB * head_dim) / 2u + QB * CB_FA + QB * head_dim;
        if (fars) { p_fa = g.p_fars; }
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
        [g.enc setBytes:&scale length:sizeof(scale) atIndex:14];
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
    // Keep the existing arithmetic of non-FA routes. FA instead rounds unscaled Q
    // to half and scales its float scores; prescaling loses half subnormal bits.
    if (scale != 1.0f) {
        imparo_metal_scale(B_Q, scale, n_tok * n_heads * head_dim);
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
    const uint32_t n_pos = attn_decode_span(start_pos, window);
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
            g_refused += 1;
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
    if (n_tok == 1 && attn_vec_serves(head_dim, n_heads, n_kv, n_pos, kv_quant)) {
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
        // GQA row sharing needs the grouped heads to share a KV head, so the group must
        // divide the share. The variants built are what the threadgroup budget allows
        // (HQ x (DK + nsg DK + 2 nsg + nsg C) floats): hd 512 x 2, hd 256 x 2 and x 3.
        const uint32_t hq_want = g_attn_stream_hq;
        id<MTLComputePipelineState> grp_sel = nil;
        if (hq_want >= 2u && n_kv > 0u && n_heads % hq_want == 0u
            && (n_heads / n_kv) % hq_want == 0u) {
            if (head_dim == 512u && hq_want == 2u) {
                grp_sel = kv_quant
                    ? (ident && g.p_attn_dec_stream_g2_q_id != nil ? g.p_attn_dec_stream_g2_q_id
                                                               : g.p_attn_dec_stream_g2_q)
                    : (ident && g.p_attn_dec_stream_g2_id != nil ? g.p_attn_dec_stream_g2_id
                                                             : g.p_attn_dec_stream_g2);
            } else if (head_dim == 256u && hq_want <= 3u) {
                const uint32_t gi = hq_want - 2u;
                grp_sel = kv_quant
                    ? (ident && g.p_attn_dec_stream_g256_q_id[gi] != nil
                           ? g.p_attn_dec_stream_g256_q_id[gi] : g.p_attn_dec_stream_g256_q[gi])
                    : (ident && g.p_attn_dec_stream_g256_id[gi] != nil
                           ? g.p_attn_dec_stream_g256_id[gi] : g.p_attn_dec_stream_g256[gi]);
            }
        }
        const uint32_t hq_grp = grp_sel != nil ? hq_want : 1u;
        id<MTLComputePipelineState> stream_sel = hdi >= 3u ? nil
            : hq_grp >= 2u
            ? grp_sel
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
        // The staged per-slice headers: `slices` x (weight, sum) floats. ATTN_MAX_SLICES
        // bounds it at 2 KiB, well inside the 32 KiB threadgroup budget, so this costs no
        // occupancy at the 64-thread shape the combine runs.
        [g.enc setThreadgroupMemoryLength:slices * 2u * sizeof(float) atIndex:0];
        const NSUInteger comb_thr = 64u * g_attn_comb_spd;
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n_tok, 1)
              threadsPerThreadgroup:MTLSizeMake(comb_thr, 1, 1)];
        if (g_prof) { prof_end(); }
        // DIAGNOSTIC (IMPARO_DUP_ATTN_COMBINE=1): encode the flash-decoding combine a SECOND
        // time. It reads the slice partials and writes B_ATTN, so a second copy writes the
        // same values -- the step hash is the check. Additive by construction, which a skip
        // of one class inside a chain is not. The combine's own work grows with `slices`
        // (24 at 5962 keys, 67 at 17122) while its grid stays n_heads x n_tok = 32
        // threadgroups, so this is how to tell whether that shape costs anything.
        if (g_dup_attn_combine) {
            haz(hb(B_ATTN_PART), hb(B_ATTN));
            [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n_tok, 1)
                  threadsPerThreadgroup:MTLSizeMake(comb_thr, 1, 1)];
        }
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

// The one-token convolution step's pipeline for `form` (the output and the state shift in one
// dispatch), nil when the step route is off (IMPARO_SHORTCONV_STEP=0, the A/B arm) or not
// built. imparo_metal_causal_conv takes the route on exactly this; a co-batched step reads it
// to know each row's convolution is one dispatch.
static id<MTLComputePipelineState> conv_step_pipe(uint32_t form) {
    static int step_on = -1;
    if (step_on < 0) { const char * e = getenv("IMPARO_SHORTCONV_STEP"); step_on = !(e && e[0] == '0'); }
    if (!step_on || form > 1u) { return nil; }
    return form == 0u ? g.p_shortconv_step : g.p_shortconv_step_plain;
}

// A CAUSAL DEPTHWISE CONVOLUTION over per-conversation history: outputs, then the state
// advance. Two dispatches with the encoder's hazard barrier between them -- see the
// kernel note for why one will not do. `form` selects the pipeline (0 gated, 1 plain +
// SiLU); it is a compile-time choice on the device, so each form runs its own code.
extern "C" void imparo_metal_causal_conv(uint32_t form, uint32_t src, uint64_t w_off,
                                         uint32_t state, uint32_t state_off,
                                         uint32_t state_out_off, uint32_t out,
                                         uint32_t width, uint32_t kern, uint32_t n_tok) {
    if (g_skip_cat == PC_RECUR) { return; }
    if (form > 1u) { return; }
    if (g.p_conv[form] == nil || g.p_conv_state[form] == nil) { return; }
    const uint32_t stride = form == 0u ? 3u * width : width;
    // ONE TOKEN: one dispatch for the conv output and the state shift
    // (IMPARO_SHORTCONV_STEP=0 keeps the two dispatches, the A/B arm). No half mirror at
    // one token.
    //
    // BOTH FORMS, since the step brick is templated on the form like every other conv body.
    // It was the gated form's only, so qwen35's plain+SiLU conv ran the output and the shift
    // as two dependent dispatches -- 48 boundaries a token that LFM2's identical pair has
    // not paid since task #117.
    id<MTLComputePipelineState> step_p = conv_step_pipe(form);
    if (n_tok == 1u && step_p != nil) {
        haz(hb(src) | hb(state), hb(out) | hb(state));
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
        { const WSeg & ws = wseg_use(w_off);
          [g.enc setBuffer:ws.buf offset:(NSUInteger)(w_off - ws.base) atIndex:1]; }
        [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
                 atIndex:2];
        [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] atIndex:3];
        [g.enc setBytes:&width length:4 atIndex:4];
        [g.enc setBytes:&kern length:4 atIndex:5];
        [g.enc setBuffer:g.bufs[state]
                  offset:g.buf_off[state] + (NSUInteger)state_out_off * 4 atIndex:6];
        dispatch1(step_p, width, PC_RECUR);
        return;
    }
    // Dual-write the half mirror when the out_proj GEMM will read it (the same rule the
    // norm and add producers use): contiguous [token][channel] rows, prefill scale only.
    const bool xh_on = g_half_a != 0u && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                    && (uint64_t)n_tok * width * 2ull <= g.sizes[B_XH]
                    && half_consumers_exist();
    haz(hb(src) | hb(state), hb(out) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_conv[form]];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    { const WSeg & ws = wseg_use(w_off);
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
    dispatch1(g.p_conv[form], n_tok * width, PC_RECUR);
    if (xh_on) { g_xh_src = out; g_xh_elems = (uint64_t)n_tok * width; g_xh_buf = B_XH; }
    (void)stride;

    // The state advance READS what the pass above read and WRITES over it, so it must not
    // start until that one has finished. Declared to the hazard tracker rather than
    // assumed: under IMPARO_CONCURRENT the encoder is unordered.
    haz(hb(src) | hb(state), hb(state));
    [g.enc setComputePipelineState:g.p_conv_state[form]];
    const NSUInteger st_off = g.buf_off[state] + (NSUInteger)state_off * 4;
    const NSUInteger st_out = g.buf_off[state] + (NSUInteger)state_out_off * 4;
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[state] offset:st_off atIndex:1];
    [g.enc setBuffer:g.bufs[state] offset:st_out atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&kern length:4 atIndex:4];
    [g.enc setBytes:&n_tok length:4 atIndex:5];
    dispatch1(g.p_conv_state[form], width, PC_RECUR);
}

// The state as of a BOUNDARY inside this chunk, written to `snap` -- the live state is
// read and left alone. `n_tok` is how many of the chunk's tokens precede the boundary.
//
// One small dispatch instead of cutting the batch to stand on the boundary: the cut
// measured +29 ms on an 800-token prefill, against `width` threads here.
extern "C" void imparo_metal_causal_conv_snapshot(uint32_t form, uint32_t src, uint32_t state,
                                                  uint32_t state_off, uint32_t snap,
                                                  uint32_t snap_off, uint32_t width,
                                                  uint32_t kern, uint32_t n_tok) {
    if (g_skip_cat == PC_RECUR) { return; }
    if (form > 1u || g.p_conv_state[form] == nil) { return; }
    haz(hb(src) | hb(state), hb(snap));
    [g.enc setComputePipelineState:g.p_conv_state[form]];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
             atIndex:1];
    [g.enc setBuffer:g.bufs[snap] offset:g.buf_off[snap] + (NSUInteger)snap_off * 4
             atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&kern length:4 atIndex:4];
    [g.enc setBytes:&n_tok length:4 atIndex:5];
    dispatch1(g.p_conv_state[form], width, PC_RECUR);
}

// ---- ROW LAYOUT: a batch whose rows are not one causal chain (a tree verify) -------------------
// The FA prefill kernel and head norm + rope compiled with ROW_LAYOUT (function constant 25) read
// each row's visibility and position from the layout buffer; the convolution has its own
// kernels. Built when a model first asks, so a process that never verifies a tree builds none.
static id<MTLComputePipelineState> row_layout_pipeline(NSString * name, bool row_layout,
                                                       bool float_q = false,
                                                       uint32_t verify_variant = 0u) {
    if (g.lib == nil) { return nil; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    stamp_epi_act(cv);
    if (!g_attn_live_mask) {
        const bool lm_off = false;
        [cv setConstantValue:&lm_off type:MTLDataTypeBool atIndex:10];
    }
    if (row_layout) {
        const bool on = true;
        [cv setConstantValue:&on type:MTLDataTypeBool atIndex:25];
    }
    if (float_q) {
        const bool on = true;
        [cv setConstantValue:&on type:MTLDataTypeBool atIndex:26];
    }
    if ((verify_variant & 1u) != 0u) {
        const bool on = true;
        [cv setConstantValue:&on type:MTLDataTypeBool atIndex:63];
    }
    if ((verify_variant & 2u) != 0u) {
        const bool on = true;
        [cv setConstantValue:&on type:MTLDataTypeBool atIndex:64];
    }
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:name constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: row-layout function %@: %@", name, e); return nil; }
    id<MTLComputePipelineState> ps = [g.device newComputePipelineStateWithFunction:f error:&e];
    if (ps == nil) { NSLog(@"imparo metal: row-layout pipeline %@: %@", name, e); }
    // IMPARO_PIPE_LOG prints what the compiler made of it, as the load-time builder does.
    if (ps != nil && getenv("IMPARO_PIPE_LOG") != NULL) {
        NSLog(@"imparo metal pipe: %@ row_layout=%d float_q=%d max_threads=%lu simd_width=%lu "
              @"static_tg_bytes=%lu", name, row_layout ? 1 : 0, float_q ? 1 : 0,
              (unsigned long)[ps maxTotalThreadsPerThreadgroup],
              (unsigned long)[ps threadExecutionWidth],
              (unsigned long)[ps staticThreadgroupMemoryLength]);
    }
    return ps;
}

static uint32_t fa_nsg_index(uint32_t nsg) {
    return nsg == 1u ? 0u : nsg == 2u ? 1u : nsg == 4u ? 2u : nsg == 8u ? 3u : 4u;
}

// What the FA route serves (slot 0's head dim, at most 128) is what a row layout serves.
static bool row_layout_ready(uint32_t head_dim) {
    if (g.lib == nil || !g_attn_fa || head_dim == 0u || head_dim != g_qcomb_hds[0]
        || head_dim > 128u) { return false; }
    const uint32_t nsg = fa_nsg_value();
    const uint32_t idx = fa_nsg_index(nsg);
    if (idx >= 4u) { return false; }
    static bool tried[4] = { false, false, false, false };
    if (!tried[idx]) {
        tried[idx] = true;
        g.p_fa_rows_n[idx] = row_layout_pipeline(
            [NSString stringWithFormat:@"imparo_attention_prefill_fa_s0_n%u", nsg], true);
        if (g.p_head_norm_rope_rows == nil) {
            g.p_head_norm_rope_rows = row_layout_pipeline(@"imparo_head_norm_rope", true);
        }
        for (uint32_t form = 0; form < 2u; ++form) {
            if (g.p_conv_rows[form] == nil) {
                g.p_conv_rows[form] = row_layout_pipeline(
                    form == 0u ? @"imparo_causal_conv_rows_gated" : @"imparo_causal_conv_rows_plain", false);
            }
            if (g.p_conv_row_inputs[form] == nil) {
                g.p_conv_row_inputs[form] = row_layout_pipeline(
                    form == 0u ? @"imparo_conv_row_inputs_gated"
                               : @"imparo_conv_row_inputs_plain", false);
            }
        }
        NSLog(@"imparo metal: row-layout pipelines for head dim %u nsg %u: fa=%d head_norm_rope=%d "
              @"conv=%d/%d inputs=%d/%d", head_dim, nsg, g.p_fa_rows_n[idx] != nil,
              g.p_head_norm_rope_rows != nil, g.p_conv_rows[0] != nil, g.p_conv_rows[1] != nil,
              g.p_conv_row_inputs[0] != nil, g.p_conv_row_inputs[1] != nil);
    }
    return g.p_fa_rows_n[idx] != nil && g.p_head_norm_rope_rows != nil
        && g.p_conv_rows[0] != nil && g.p_conv_row_inputs[0] != nil;
}

extern "C" uint32_t imparo_metal_supports_row_layout(uint32_t head_dim) {
    return row_layout_ready(head_dim) ? 1u : 0u;
}

// Words per row of a row layout: imparo_backend::ROW_LAYOUT_WORDS (imparo.metal holds it too).
static constexpr uint32_t ROW_LAYOUT_WORDS = 12u;

// The FA dispatch of imparo_metal_attention with the row-layout pipeline, the layout bound at
// 15, the furthest any row sees past itself at 16 and the key floor at 17 (no row sees a key
// below it; 0 for every caller but a drafter rebuilt from a restore point). No window and no
// ring: the batch's rows sit in consecutive cache slots. float_q: the pipeline that keeps Q in
// float (function constant 26), built on its first use.
extern "C" uint32_t imparo_metal_attention_rows(uint32_t kv_layer, uint32_t head_dim,
                                                uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
                                                uint32_t start_pos, uint32_t key_lo, uint32_t n_tok,
                                                float scale, uint32_t layout, uint32_t float_q) {
    if (n_tok < 2u || n_tok > 64u || layout >= B_COUNT || g.bufs[layout] == nil
        || key_lo > start_pos || !row_layout_ready(head_dim)) { return 0u; }
    const uint32_t nsg = fa_nsg_value();
    const bool fq = float_q != 0u;
    if (fq && g.p_fa_rows_fq_n[fa_nsg_index(nsg)] == nil) {
        g.p_fa_rows_fq_n[fa_nsg_index(nsg)] = row_layout_pipeline(
            [NSString stringWithFormat:@"imparo_attention_prefill_fa_s0_n%u", nsg], true, true);
    }
    id<MTLComputePipelineState> p = fq ? g.p_fa_rows_fq_n[fa_nsg_index(nsg)]
                                       : g.p_fa_rows_n[fa_nsg_index(nsg)];
    // WHERE PREFILL TAKES THE REGISTER-SOFTMAX OP, SO DOES A ROW LAYOUT: with the key split
    // off, a chain's rows are prefill's rows bit for bit. Q in float (a drafter's option) has no
    // such twin, and keeps the 8-query op.
    // The kernel is compiled for slot 0's head dim (g.p_fars is built only at <= 128).
    const bool fars = !fq && fars_route_on() && g.p_fars != nil && head_dim == g_qcomb_hds[0];
    if (fars && g.p_fars_rows == nil) {
        g.p_fars_rows = row_layout_pipeline(@"imparo_attention_prefill_fars_s0", true, false);
    }
    // The verify's grid. One query tile a head is n_heads threadgroups each walking every key:
    // ROW_HEADS makes it n_kv threadgroups doing four heads' useful rows each, ROW_SPLIT cuts the
    // keys into `slices` threadgroups each (partials, then a merge). Both change the arithmetic
    // against the causal forward (the split reassociates the softmax), so a verify row is held to
    // the causal row by tolerance, not bits, where either is on.
    const uint32_t grp = n_kv > 0u && n_heads % n_kv == 0u ? n_heads / n_kv : 0u;
    const bool vheads = fars && g_verify_attn_heads != 0u && grp == 4u;
    const uint32_t QT = vheads ? 8u : 32u;
    const uint32_t gx = vheads ? n_kv : n_heads, gy = (n_tok + QT - 1u) / QT;
    // THE SLICES FOLLOW THE CACHE BELOW THE BATCH, NOT THE BATCH: the chunk is sized from the
    // keys under start_pos and the query tile count is left out, so a tree's row and the same
    // row of a chain cut the keys at the same places -- the two stay bit-identical. A slice is
    // at least two 32-key blocks. Only past the partials buffer (rows x slices beyond what it
    // holds) does the chunk grow with the rows.
    const uint32_t scan_lo = key_lo & ~31u;
    const uint32_t span = start_pos + n_tok - scan_lo;
    uint32_t slices = 1u, chunk = 0u;
    if (fars && g_verify_split_on != 0u && g_verify_attn_tgs > gx && g.bufs[B_ATTN_PART] != nil) {
        const uint32_t want = (g_verify_attn_tgs + gx - 1u) / gx;
        const uint32_t cache_blk = (start_pos - scan_lo + 31u) / 32u;
        chunk = std::max(2u, (cache_blk + want - 1u) / want) * 32u;
        const uint64_t per_slice = (uint64_t)n_tok * n_heads * (head_dim + 2u) * 4u;
        const uint64_t fit = g.sizes[B_ATTN_PART] / per_slice;
        if (fit >= 2u && (span + chunk - 1u) / chunk > fit) {
            chunk = ((span + (uint32_t)fit * 32u - 1u) / ((uint32_t)fit * 32u)) * 32u;
        }
        slices = fit >= 2u ? (span + chunk - 1u) / chunk : 1u;
    }
    const bool vsplit = slices >= 2u;
    if (!vsplit) { chunk = 0u; }
    const uint32_t variant = (vsplit ? 1u : 0u) | (vheads ? 2u : 0u);
    if (variant != 0u && g.p_fars_rows_v[variant] == nil) {
        g.p_fars_rows_v[variant] = row_layout_pipeline(@"imparo_attention_prefill_fars_s0", true,
                                                       false, variant);
    }
    if (vsplit && g.p_rows_combine == nil) {
        g.p_rows_combine = row_layout_pipeline(@"imparo_attention_rows_combine", false);
    }
    if (variant != 0u && (g.p_fars_rows_v[variant] == nil
                          || (vsplit && g.p_rows_combine == nil))) { return 0u; }
    if (fars) { p = variant != 0u ? g.p_fars_rows_v[variant] : g.p_fars_rows; }
    if (p == nil) { return 0u; }
    if (getenv("IMPARO_ATTN_WHICH")) {
        fprintf(stderr, "attn rows layer=%u n_tok=%u start=%u key_lo=%u fars=%u heads=%u slices=%u "
                        "chunk=%u grid=%ux%ux%u\n", kv_layer, n_tok, start_pos, key_lo, fars ? 1u : 0u,
                vheads ? 1u : 0u, slices, chunk, gx, gy, slices);
    }
    // How many rows past its own the furthest-seeing row attends to: 0 for a chain or a tree
    // (a row sees only earlier rows), the rest of the block for a drafted block. The kernel
    // ends each query tile's key scan that far past the tile's last row. A layout naming a
    // row outside the batch is refused: those keys are not the batch's.
    if ((uint64_t)n_tok * ROW_LAYOUT_WORDS * 4u > g.sizes[layout]) { return 0u; }
    const uint32_t * words = (const uint32_t *)((char *)[g.bufs[layout] contents] + g.buf_off[layout]);
    uint32_t reach = 0u;
    for (uint32_t t = 0; t < n_tok; ++t) {
        const uint64_t seen = (uint64_t)words[t * ROW_LAYOUT_WORDS + 2u]
                            | ((uint64_t)words[t * ROW_LAYOUT_WORDS + 3u] << 32);
        if (seen == 0u) { continue; }
        const uint32_t last = 63u - (uint32_t)__builtin_clzll(seen);
        if (last >= n_tok) { return 0u; }
        if (last > t && last - t > reach) { reach = last - t; }
    }
    const bool kq = kv_eff_type(kv_layer, 0) != 1u;
    const bool vq = kv_eff_type(kv_layer, 1) != 1u;
    if (kq) { kv_dequant_now(kv_layer, kv_width, start_pos + n_tok, 0u, 0u); }
    if (vq) { kv_dequant_now(kv_layer, kv_width, start_pos + n_tok, 1u, 0u); }
    // The register-softmax op: 32 queries and four simdgroups a threadgroup, two 32-key blocks
    // of (hd + 8)-half rows staged (the prefill dispatch's sizes).
    const uint32_t QB = fars ? 32u : FA_QB;
    const uint32_t THREADS = fars ? 128u : nsg * 32u;
    // Staged Q is QB x hd halves, or QB x hd floats with Q kept in float.
    const uint32_t q_floats = fq ? FA_QB * head_dim : (FA_QB * head_dim) / 2u;
    const uint32_t sfloats = fars ? (2u * 32u * (head_dim + 8u)) / 2u
                                  : q_floats + FA_QB * FA_CB + FA_QB * head_dim;
    const bool axh = g_half_a && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                  && half_consumers_exist()
                  && (uint64_t)n_tok * n_heads * head_dim * 2ull <= g.sizes[B_XH];
    ensure_kv_pt(kv_layer, (start_pos + n_tok + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS);
    haz(hb(B_Q) | hb(layout) | (kq ? hb(B_KDQ) : HZ_KVK) | (vq ? hb(B_VDQ) : HZ_KVV),
        vsplit ? hb(B_ATTN_PART) : (hb(B_ATTN) | (axh ? hb(B_XH) : 0u)));
    const uint32_t window = 0u, ring = 0u;
    [g.enc setComputePipelineState:p];
    [g.enc setBuffer:g.bufs[B_Q]    offset:g.buf_off[B_Q]    atIndex:0];
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
    else     { [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:11]; }
    const uint32_t axh_flag = axh ? 1u : 0u;
    [g.enc setBytes:&axh_flag length:4 atIndex:12];
    [g.enc setBuffer:g.kv_pt[kv_layer] offset:0 atIndex:13];
    [g.enc setBytes:&scale length:sizeof(scale) atIndex:14];
    [g.enc setBuffer:g.bufs[layout] offset:g.buf_off[layout] atIndex:15];
    [g.enc setBytes:&reach length:4 atIndex:16];
    [g.enc setBytes:&key_lo length:4 atIndex:17];
    if (vsplit) {
        [g.enc setBuffer:g.bufs[B_ATTN_PART] offset:g.buf_off[B_ATTN_PART] atIndex:18];
        [g.enc setBytes:&chunk length:4 atIndex:19];
    }
    [g.enc setThreadgroupMemoryLength:sfloats * sizeof(float) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
    if (fars) {
        [g.enc dispatchThreadgroups:MTLSizeMake(gx, gy, slices)
              threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    } else {
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, (n_tok + QB - 1u) / QB, 1)
              threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    }
    if (g_prof) { prof_end(); }
    if (vsplit) {
        haz(hb(B_ATTN_PART), hb(B_ATTN) | (axh ? hb(B_XH) : 0u));
        [g.enc setComputePipelineState:g.p_rows_combine];
        [g.enc setBuffer:g.bufs[B_ATTN_PART] offset:g.buf_off[B_ATTN_PART] atIndex:0];
        [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:1];
        [g.enc setBytes:&head_dim length:4 atIndex:2];
        [g.enc setBytes:&n_heads length:4 atIndex:3];
        [g.enc setBytes:&slices length:4 atIndex:4];
        if (axh) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:5]; }
        else     { [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] atIndex:5]; }
        const uint32_t axh_c = axh ? 1u : 0u;
        [g.enc setBytes:&axh_c length:4 atIndex:6];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n_tok, 1)
              threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
        if (g_prof) { prof_end(); }
    }
    if (axh) {
        g_xh_src = B_ATTN;
        g_xh_elems = (uint64_t)n_tok * n_heads * head_dim;
        g_xh_buf = B_XH;
    }
    return 1u;
}

// imparo_metal_head_norm_rope with each row roped at its layout position (bound at 12).
extern "C" uint32_t imparo_metal_head_norm_rope_rows(uint32_t buf, uint64_t w_off,
                                                     uint32_t head_dim, float eps, uint32_t n_heads,
                                                     uint32_t n_tok, uint32_t n_rot, float base,
                                                     const float * freqs, uint32_t n_freqs,
                                                     uint32_t layout) {
    if (layout >= B_COUNT || g.bufs[layout] == nil || g.p_head_norm_rope_rows == nil) {
        return 0u;
    }
    haz(hb(buf) | hb(layout), hb(buf));
    const uint32_t n_row = n_tok * n_heads;
    const uint32_t start_pos = 0u;   // unread under ROW_LAYOUT: positions come from the layout
    [g.enc setComputePipelineState:g.p_head_norm_rope_rows];
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
    [g.enc setBuffer:g.bufs[layout] offset:g.buf_off[layout] atIndex:12];
    // Thread count exactly as imparo_metal_head_norm_rope derives it: the bits depend on it.
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
    return 1u;
}

// CO-BATCHED ROWS: row r of a step runs the one-row kernel with its slot selected and its
// operands moved to row r. The one-row paths read row 0 of each operand and bind a buffer's
// offset when the dispatch is encoded, so moving it for one call moves nothing else -- and a
// row's arithmetic is the one-row decode's by construction.
struct RowRebase {
    uint32_t id;
    uint64_t saved;
    RowRebase(uint32_t id_, uint64_t bytes) : id(id_), saved(g.buf_off[id_]) { g.buf_off[id_] += bytes; }
    ~RowRebase() { g.buf_off[id] = saved; }
    RowRebase(const RowRebase &) = delete;
    RowRebase & operator=(const RowRebase &) = delete;
};

// Whether no slot appears twice among the rows.
static bool slots_distinct(const uint32_t * slots, uint32_t n_rows) {
    for (uint32_t a = 0; a < n_rows; ++a) {
        for (uint32_t b = a + 1u; b < n_rows; ++b) {
            if (slots[a] == slots[b]) { return false; }
        }
    }
    return true;
}

// ---- CO-BATCHED ROWS AS ONE DISPATCH -----------------------------------------------------------
// A per-row operation of a co-batched step runs its one-row kernel's code once over all the rows
// (function constant 32): grid row t is row t of the activations, and row t's conversation -- its
// position, its page table for the layer, its recurrent state -- is entry t of a CobRow table in
// constant memory. A row's arithmetic is its one-row dispatch's; only where it finds its
// conversation changes. The table holds addresses, so every buffer it names is declared to the
// encoder, as the mega route's operands are.
//
// Run as the one-row kernel per row with the row's slot selected, an operation cost B dispatches,
// and a barrier between each two: rows of different slots touch different buffers under one
// buffer id, and the hazard tracker keys on ids. That loop stays for what the rows kernels do not
// serve (a ring, the store's roundtrip diagnostic, attention off the vector route, two rows of one
// conversation).
struct CobRowHost { uint64_t pt, state, state_out; uint32_t pos, pad; };   // imparo.metal CobRow
static_assert(sizeof(CobRowHost) == 32u, "CobRow drift");
// Rows one table carries: setBytes takes at most 4 KB, so a wider step is several dispatches.
static constexpr uint32_t COB_TABLE_ROWS = 4096u / (uint32_t)sizeof(CobRowHost);

// The rows variant of a one-row pipeline: the one-row pipeline's constants (`live_mask`: the
// live-mask setting the decode attention kernels are built with) plus constant 32. Built on first
// use, so a process that never co-batches builds none.
static id<MTLComputePipelineState> cob_pipeline(Context::CobPipe & cp, NSString * name,
                                                bool live_mask) {
    if (cp.p != nil || cp.tried || g.lib == nil) { return cp.p; }
    cp.tried = true;
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    if (live_mask && !g_attn_live_mask) {
        const bool lm_off = false;
        [cv setConstantValue:&lm_off type:MTLDataTypeBool atIndex:10];
    }
    const bool on = true;
    [cv setConstantValue:&on type:MTLDataTypeBool atIndex:32];
    stamp_epi_act(cv);
    const double t0 = CACurrentMediaTime();
    cp.p = make_with(g.lib, name, cv);
    NSLog(@"imparo metal: co-batch rows pipeline %@ %s in %.1f ms", name,
          cp.p != nil ? "built" : "FAILED", (CACurrentMediaTime() - t0) * 1e3);
    return cp.p;
}

// Each row's conversation, for a rows kernel's table: the rows' slots are selected in turn, `fill`
// writes row r's entry from the selected slot's state, and the selection is restored. Every
// buffer `fill` names in `used` is declared to the encoder once. False, with nothing encoded, when
// a slot cannot be selected or `fill` refuses a row.
template <typename F>
static bool cob_gather(const uint32_t * slots, uint32_t n_rows, std::vector<CobRowHost> & tab,
                       MTLResourceUsage usage, F && fill) {
    static std::vector<id<MTLResource>> used;
    tab.assign(n_rows, CobRowHost{});
    used.clear();
    const uint32_t cur = g_slot_cur;
    bool ok = true;
    for (uint32_t r = 0; r < n_rows && ok; ++r) {
        ok = imparo_metal_select_slot(slots[r]) == 0 && fill(r, tab[r], used);
    }
    imparo_metal_select_slot(cur);
    if (ok && !used.empty()) {
        [g.enc useResources:used.data() count:used.size() usage:usage];
    }
    used.clear();
    return ok;
}

// imparo_metal_kv_store for co-batched rows: row t of `src` stored at pos[t] through its slot's
// page table, the one-row f16 store's kernel. False, with nothing encoded, for a quantized cache
// (co-batched decode reads f16 only: Workflow::set_slots refuses the others), a ring, the
// roundtrip diagnostic, or two rows of one slot.
static bool cob_kv_store(uint32_t src, uint32_t layer, uint32_t width, const uint32_t * slots,
                         const uint32_t * pos, uint32_t n_rows, uint32_t is_v, uint32_t ring) {
    if (g.enc == nil || n_rows == 0u || ring != 0u || kv_eff_type(layer, is_v) != 1u
        || !slots_distinct(slots, n_rows)) {
        return false;
    }
    const uint32_t rt_msk = is_v ? g_kvq_mask_v : g_kvq_mask_k;
    if (g_kvq_rt != 0u && (is_v ? g_kv_type_v : g_kv_type_k) == 1u && layer < 32u
        && ((rt_msk >> layer) & 1u) && g.p_kvstore_rt != nil) { return false; }
    id<MTLComputePipelineState> p = cob_pipeline(g.cob_kvstore, @"imparo_kv_store", false);
    if (p == nil) { return false; }
    static std::vector<CobRowHost> tab;
    id<MTLBuffer> any_pt = nil;
    const bool ok = cob_gather(slots, n_rows, tab, MTLResourceUsageRead,
        [&](uint32_t r, CobRowHost & e, std::vector<id<MTLResource>> & used) {
            // The page table sized as the row's one-row store sizes it.
            const uint32_t need = (pos[r] + 1u + KV_PAGE_CELLS - 1u) / KV_PAGE_CELLS;
            ensure_kv_pt(layer, need);
            kv_pt_audit("store-f16", layer, need);
            id<MTLBuffer> pt = g.kv_pt[layer];
            used.push_back(pt);
            e.pt = (uint64_t)[pt gpuAddress];
            e.pos = pos[r];
            if (any_pt == nil) { any_pt = pt; }
            return true;
        });
    if (!ok) { return false; }
    haz(hb(src), is_v ? HZ_KVV : HZ_KVK);
    const uint32_t start_pos = 0u;   // unread under the constant: positions come from the table
    // One thread per four values, plus up to four for a tail (the one-row store's grid).
    const NSUInteger gx = width / 4u + ((width % 4u) ? 4u : 0u);
    for (uint32_t r0 = 0; r0 < n_rows; r0 += COB_TABLE_ROWS) {
        const uint32_t n = std::min(COB_TABLE_ROWS, n_rows - r0);
        [g.enc setComputePipelineState:p];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] + (NSUInteger)r0 * width * 4u atIndex:0];
        [g.enc setBuffer:(is_v ? g.kv_v[layer] : g.kv_k[layer]) offset:kv_reg(layer, is_v)
                 atIndex:1];
        [g.enc setBytes:&width length:4 atIndex:2];
        [g.enc setBytes:&start_pos length:4 atIndex:3];
        [g.enc setBytes:&n length:4 atIndex:4];
        [g.enc setBytes:&ring length:4 atIndex:5];
        [g.enc setBuffer:any_pt offset:0 atIndex:6];   // unread under the constant
        [g.enc setBytes:tab.data() + r0 length:n * sizeof(CobRowHost) atIndex:7];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_KV_STORE); }
        [g.enc dispatchThreads:MTLSizeMake(gx, n, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (g_prof) { prof_end(); }
    }
    return true;
}

// The vector decode route of imparo_metal_attention for co-batched rows: row t of B_Q (already
// scaled) attends over its slot's cache through pos[t] into row t of B_ATTN. False, with nothing
// encoded, unless every row takes that route (one query, an f16 cache, a span within
// attn_vec_max_keys) with no ring and no two rows of one slot.
static bool cob_attention(uint32_t kv_layer, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv,
                          uint32_t kv_width, uint32_t window, const uint32_t * slots,
                          const uint32_t * pos, uint32_t n_rows, uint32_t ring) {
    if (g.enc == nil || n_rows == 0u || ring != 0u || !slots_distinct(slots, n_rows)) {
        return false;
    }
    const bool kv_quant = kv_eff_type(kv_layer, 0) != 1u || kv_eff_type(kv_layer, 1) != 1u;
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (!attn_vec_serves(head_dim, n_heads, n_kv, attn_decode_span(pos[r], window),
                             kv_quant)) { return false; }
    }
    const uint32_t slot_of = attn_hd_slot(head_dim);   // < 2: attn_vec_serves held
    static NSString * const names[2] = {
        @"imparo_attention_decode_vec_s0", @"imparo_attention_decode_vec_s1" };
    id<MTLComputePipelineState> p = cob_pipeline(g.cob_attn_vec[slot_of], names[slot_of], true);
    if (p == nil) { return false; }
    static std::vector<CobRowHost> tab;
    id<MTLBuffer> any_pt = nil;
    const bool ok = cob_gather(slots, n_rows, tab, MTLResourceUsageRead,
        [&](uint32_t r, CobRowHost & e, std::vector<id<MTLResource>> & used) {
            // The table as the step's store left it, which is what the one-row route binds.
            if (kv_layer >= g.kv_pt.size() || g.kv_pt[kv_layer] == nil) { return false; }
            id<MTLBuffer> pt = g.kv_pt[kv_layer];
            used.push_back(pt);
            e.pt = (uint64_t)[pt gpuAddress];
            e.pos = pos[r];
            if (any_pt == nil) { any_pt = pt; }
            return true;
        });
    if (!ok) { return false; }
    if (getenv("IMPARO_ATTN_WHICH")) {
        fprintf(stderr, "attn branch vec rows=%u layer=%u hd=%u slot=%u nsg=%u window=%u\n",
                n_rows, kv_layer, head_dim, slot_of, ATTN_VEC_NSG, window);
    }
    haz(hb(B_Q) | HZ_KVK | HZ_KVV, hb(B_ATTN));
    const uint64_t row_bytes = (uint64_t)n_heads * head_dim * 4u;
    const uint32_t start_pos = 0u;   // unread under the constant
    for (uint32_t r0 = 0; r0 < n_rows; r0 += COB_TABLE_ROWS) {
        const uint32_t n = std::min(COB_TABLE_ROWS, n_rows - r0);
        [g.enc setComputePipelineState:p];
        [g.enc setBuffer:g.bufs[B_Q] offset:g.buf_off[B_Q] + r0 * row_bytes atIndex:0];
        [g.enc setBuffer:g.kv_k[kv_layer] offset:kv_reg(kv_layer, false) atIndex:1];
        [g.enc setBuffer:g.kv_v[kv_layer] offset:kv_reg(kv_layer, true) atIndex:2];
        [g.enc setBuffer:g.bufs[B_ATTN] offset:g.buf_off[B_ATTN] + r0 * row_bytes atIndex:3];
        [g.enc setBytes:&head_dim length:4 atIndex:4];
        [g.enc setBytes:&n_heads length:4 atIndex:5];
        [g.enc setBytes:&n_kv length:4 atIndex:6];
        [g.enc setBytes:&kv_width length:4 atIndex:7];
        [g.enc setBytes:&start_pos length:4 atIndex:8];
        [g.enc setBytes:&window length:4 atIndex:9];
        [g.enc setBytes:&ring length:4 atIndex:10];
        [g.enc setBuffer:any_pt offset:0 atIndex:18];   // unread under the constant
        [g.enc setBytes:tab.data() + r0 length:n * sizeof(CobRowHost) atIndex:19];
        [g.enc setThreadgroupMemoryLength:(NSUInteger)(ATTN_VEC_NSG / 2u) * (head_dim + 2u) * sizeof(float) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ATTENTION); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_heads, n, 1)
              threadsPerThreadgroup:MTLSizeMake(ATTN_VEC_NSG * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
    }
    return true;
}

// The one-token step of imparo_metal_causal_conv for co-batched rows: row t reads row t of `src`
// and its slot's history, writes row t of `out` and its advanced history. False, with nothing
// encoded, when the step route is off or two rows share a slot.
static bool cob_causal_conv(uint32_t form, uint32_t src, uint64_t w_off, uint32_t state,
                            const uint32_t * slots, const uint32_t * state_off,
                            const uint32_t * state_out_off, uint32_t n_rows, uint32_t out,
                            uint32_t width, uint32_t kern) {
    if (g.enc == nil || n_rows == 0u || form > 1u || conv_step_pipe(form) == nil
        || !slots_distinct(slots, n_rows)) { return false; }
    id<MTLComputePipelineState> p = cob_pipeline(
        g.cob_conv_step[form], form == 0u ? @"imparo_shortconv_step" : @"imparo_shortconv_step_plain",
        false);
    if (p == nil) { return false; }
    static std::vector<CobRowHost> tab;
    id<MTLBuffer> any_state = nil;
    const bool ok = cob_gather(slots, n_rows, tab, MTLResourceUsageRead | MTLResourceUsageWrite,
        [&](uint32_t r, CobRowHost & e, std::vector<id<MTLResource>> & used) {
            id<MTLBuffer> b = g.bufs[state];   // the selected slot's
            if (b == nil) { return false; }
            used.push_back(b);
            const uint64_t base = (uint64_t)[b gpuAddress] + g.buf_off[state];
            e.state = base + (uint64_t)state_off[r] * 4u;
            e.state_out = base + (uint64_t)state_out_off[r] * 4u;
            if (any_state == nil) { any_state = b; }
            return true;
        });
    if (!ok) { return false; }
    haz(hb(src) | hb(state), hb(out) | hb(state));
    // The gated form's source row is [b | c | x], three widths; the plain form's is one.
    const uint64_t src_row = (form == 0u ? 3ull : 1ull) * width * 4u;
    NSUInteger tw = p.maxTotalThreadsPerThreadgroup;
    if (tw > 256) { tw = 256; }
    for (uint32_t r0 = 0; r0 < n_rows; r0 += COB_TABLE_ROWS) {
        const uint32_t n = std::min(COB_TABLE_ROWS, n_rows - r0);
        [g.enc setComputePipelineState:p];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] + r0 * src_row atIndex:0];
        { const WSeg & ws = wseg_use(w_off);
          [g.enc setBuffer:ws.buf offset:(NSUInteger)(w_off - ws.base) atIndex:1]; }
        [g.enc setBuffer:any_state offset:0 atIndex:2];   // unread under the constant
        [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] + (NSUInteger)r0 * width * 4u atIndex:3];
        [g.enc setBytes:&width length:4 atIndex:4];
        [g.enc setBytes:&kern length:4 atIndex:5];
        [g.enc setBuffer:any_state offset:0 atIndex:6];   // unread under the constant
        [g.enc setBytes:tab.data() + r0 length:n * sizeof(CobRowHost) atIndex:7];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RECUR); }
        [g.enc dispatchThreads:MTLSizeMake(width, n, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        if (g_prof) { prof_end(); }
    }
    return true;
}

// imparo_metal_head_norm_rope for co-batched rows: row t's heads normed and roped at pos[t].
// False, with nothing encoded, when the rows pipeline could not be built.
static bool cob_head_norm_rope(uint32_t buf, uint64_t w_off, uint32_t head_dim, float eps,
                               uint32_t n_heads, const uint32_t * pos, uint32_t n_rows,
                               uint32_t n_rot, float base, const float * freqs,
                               uint32_t n_freqs) {
    if (g.enc == nil || n_rows == 0u) { return false; }
    id<MTLComputePipelineState> p =
        cob_pipeline(g.cob_head_norm_rope, @"imparo_head_norm_rope", false);
    if (p == nil) { return false; }
    static std::vector<CobRowHost> tab;
    tab.assign(n_rows, CobRowHost{});
    for (uint32_t r = 0; r < n_rows; ++r) { tab[r].pos = pos[r]; }
    haz(hb(buf), hb(buf));
    const uint64_t row = (uint64_t)n_heads * head_dim * 4u;
    const uint32_t start_pos = 0u;   // unread under the constant
    static const float kNoFreqs[1] = {1.0f};
    // Thread count exactly as imparo_metal_head_norm_rope derives it: the bits depend on it.
    const NSUInteger vec_units = (head_dim + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    for (uint32_t r0 = 0; r0 < n_rows; r0 += COB_TABLE_ROWS) {
        const uint32_t n = std::min(COB_TABLE_ROWS, n_rows - r0);
        const uint32_t n_row = n * n_heads;
        [g.enc setComputePipelineState:p];
        const uint64_t w_off_l = wbind(g.enc, w_off, 0);
        [g.enc setBuffer:g.bufs[buf] offset:g.buf_off[buf] + r0 * row atIndex:1];
        [g.enc setBytes:&w_off_l length:8 atIndex:2];
        [g.enc setBytes:&head_dim length:4 atIndex:3];
        [g.enc setBytes:&eps length:4 atIndex:4];
        [g.enc setBytes:&n_row length:4 atIndex:5];
        [g.enc setBytes:&n_rot length:4 atIndex:6];
        [g.enc setBytes:&base length:4 atIndex:7];
        [g.enc setBytes:&n_heads length:4 atIndex:8];
        [g.enc setBytes:&start_pos length:4 atIndex:9];
        [g.enc setBytes:(n_freqs > 0 ? freqs : kNoFreqs)
                 length:(n_freqs > 0 ? n_freqs * 4 : 4) atIndex:10];
        [g.enc setBytes:&n_freqs length:4 atIndex:11];
        [g.enc setBytes:tab.data() + r0 length:n * sizeof(CobRowHost) atIndex:13];
        [g.enc setThreadgroupMemoryLength:tg_bytes16((threads / 32) * sizeof(float)) atIndex:0];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_RMSNORM); }
        [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        if (g_prof) { prof_end(); }
    }
    return true;
}

extern "C" void imparo_metal_kv_store_slot_rows(uint32_t src, uint32_t layer, uint32_t width,
                                                const uint32_t * slots, const uint32_t * pos,
                                                uint32_t n_rows, uint32_t is_v, uint32_t ring) {
    if (g_skip_cat == PC_KV_STORE) { return; }
    if (cob_kv_store(src, layer, width, slots, pos, n_rows, is_v, ring)) { return; }
    const uint32_t cur = g_slot_cur;
    for (uint32_t r = 0; r < n_rows; ++r) {
        imparo_metal_select_slot(slots[r]);
        RowRebase row(src, (uint64_t)r * width * 4u);
        imparo_metal_kv_store(src, layer, width, pos[r], 1u, is_v, ring);
    }
    imparo_metal_select_slot(cur);
}

extern "C" void imparo_metal_attention_slot_rows(uint32_t kv_layer, uint32_t head_dim,
                                                 uint32_t n_heads, uint32_t n_kv,
                                                 uint32_t kv_width, uint32_t window, float scale,
                                                 const uint32_t * slots, const uint32_t * pos,
                                                 const uint32_t * max_scores, uint32_t n_rows,
                                                 uint32_t ring) {
    if (attn_skipped(head_dim)) { return; }
    // Every decode route scales Q in place before it attends (see imparo_metal_attention): one
    // elementwise dispatch over all the rows gives each row the bits its own would, and the
    // rows then attend unscaled.
    if (scale != 1.0f) { imparo_metal_scale(B_Q, scale, n_rows * n_heads * head_dim); }
    if (cob_attention(kv_layer, head_dim, n_heads, n_kv, kv_width, window, slots, pos, n_rows,
                      ring)) { return; }
    const uint64_t row_bytes = (uint64_t)n_heads * head_dim * 4u;
    const uint32_t cur = g_slot_cur;
    for (uint32_t r = 0; r < n_rows; ++r) {
        imparo_metal_select_slot(slots[r]);
        RowRebase q(B_Q, (uint64_t)r * row_bytes);
        RowRebase o(B_ATTN, (uint64_t)r * row_bytes);
        imparo_metal_attention(kv_layer, head_dim, n_heads, n_kv, kv_width, pos[r], window, 1u,
                               max_scores[r], ring, 1.0f);
    }
    imparo_metal_select_slot(cur);
}

extern "C" void imparo_metal_causal_conv_slot_rows(uint32_t form, uint32_t src, uint64_t w_off,
                                                   uint32_t state, const uint32_t * slots,
                                                   const uint32_t * state_off,
                                                   const uint32_t * state_out_off,
                                                   uint32_t n_rows, uint32_t out, uint32_t width,
                                                   uint32_t kern) {
    if (g_skip_cat == PC_RECUR) { return; }
    if (cob_causal_conv(form, src, w_off, state, slots, state_off, state_out_off, n_rows, out,
                        width, kern)) { return; }
    // The gated form's source row is [b | c | x], three widths; the plain form's is one.
    const uint64_t src_row = (form == 0u ? 3ull : 1ull) * width * 4u;
    const uint32_t cur = g_slot_cur;
    for (uint32_t r = 0; r < n_rows; ++r) {
        imparo_metal_select_slot(slots[r]);
        RowRebase in(src, (uint64_t)r * src_row);
        RowRebase o(out, (uint64_t)r * width * 4u);
        imparo_metal_causal_conv(form, src, w_off, state, state_off[r], state_out_off[r], out,
                                 width, kern, 1u);
    }
    imparo_metal_select_slot(cur);
}

// Per-head norm and rope of rows that sit at unrelated positions: row r at pos[r], each with
// the one-row kernel's arithmetic, so a row's bits are its lone decode's.
extern "C" void imparo_metal_head_norm_rope_at(uint32_t buf, uint64_t w_off, uint32_t head_dim,
                                               float eps, uint32_t n_heads, const uint32_t * pos,
                                               uint32_t n_rows, uint32_t n_rot, float base,
                                               const float * freqs, uint32_t n_freqs) {
    if (g_skip_cat == PC_RMSNORM) { return; }
    if (cob_head_norm_rope(buf, w_off, head_dim, eps, n_heads, pos, n_rows, n_rot, base, freqs,
                           n_freqs)) { return; }
    const uint64_t row = (uint64_t)n_heads * head_dim * 4u;
    for (uint32_t r = 0; r < n_rows; ++r) {
        RowRebase at(buf, (uint64_t)r * row);
        imparo_metal_head_norm_rope(buf, w_off, head_dim, eps, n_heads, pos[r], 1u, n_rot, base,
                                    freqs, n_freqs);
    }
}

// The output pass of imparo_metal_causal_conv for a layout batch; the state is not advanced.
extern "C" uint32_t imparo_metal_causal_conv_rows(uint32_t form, uint32_t src, uint64_t w_off,
                                                  uint32_t state, uint32_t state_off, uint32_t out,
                                                  uint32_t width, uint32_t kern, uint32_t n_tok,
                                                  uint32_t layout) {
    if (form > 1u || n_tok == 0u || layout >= B_COUNT || g.bufs[layout] == nil
        || g.p_conv_rows[form] == nil) { return 0u; }
    const bool xh_on = g_half_a != 0u && n_tok >= HALF_A_MIN && g.bufs[B_XH] != nil
                    && (uint64_t)n_tok * width * 2ull <= g.sizes[B_XH]
                    && half_consumers_exist();
    haz(hb(src) | hb(state) | hb(layout), hb(out) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_conv_rows[form]];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    { const WSeg & ws = wseg_use(w_off);
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
    [g.enc setBuffer:g.bufs[layout] offset:g.buf_off[layout] atIndex:9];
    dispatch1(g.p_conv_rows[form], n_tok * width, PC_RECUR);
    if (xh_on) { g_xh_src = out; g_xh_elems = (uint64_t)n_tok * width; g_xh_buf = B_XH; }
    return 1u;
}

// Each batch row's input to a convolution window: `width` values at inputs_off + row * row_elems.
extern "C" uint32_t imparo_metal_conv_row_inputs(uint32_t form, uint32_t src, uint32_t inputs,
                                                 uint32_t inputs_off, uint32_t row_elems,
                                                 uint32_t width, uint32_t n_tok) {
    if (form > 1u || n_tok == 0u || width == 0u || width > row_elems || src >= B_COUNT
        || inputs >= B_COUNT || g.bufs[src] == nil || g.bufs[inputs] == nil
        || g.p_conv_row_inputs[form] == nil) {
        return 0u;
    }
    const uint64_t end = (uint64_t)inputs_off + (uint64_t)(n_tok - 1u) * row_elems + width;
    if (end * 4ull > g.sizes[inputs]) { return 0u; }
    haz(hb(src), hb(inputs));
    [g.enc setComputePipelineState:g.p_conv_row_inputs[form]];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[inputs] offset:g.buf_off[inputs] + (NSUInteger)inputs_off * 4
             atIndex:1];
    [g.enc setBytes:&width length:4 atIndex:2];
    [g.enc setBytes:&row_elems length:4 atIndex:3];
    [g.enc setBytes:&n_tok length:4 atIndex:4];
    dispatch1(g.p_conv_row_inputs[form], n_tok * width, PC_RECUR);
    return 1u;
}

// A tree commit's row move: cache rows of one full-attention layer copied between logical
// positions, row from[i] -> row to[i] in order, K then V, through the layer's block table (the
// page rule kv_slot applies for the store and the attention). The cache is shared storage, so a
// host copy is the whole transfer; call with the device idle. Every row is checked against both
// buffers before any byte moves, so a refusal leaves the cache as it was.
static inline uint64_t kv_host_slot(uint32_t layer, uint32_t pos) {
    id<MTLBuffer> pt = layer < g.kv_pt.size() ? g.kv_pt[layer] : nil;
    const uint32_t page = pos / KV_PAGE_CELLS;
    // A table that does not reach this page is the identity there: ensure_kv_pt extends a table
    // with identity entries before any kernel reads it.
    if (pt == nil || page >= (uint32_t)([pt length] / 4u)) { return pos; }
    const uint32_t * e = (const uint32_t *)[pt contents];
    return (uint64_t)e[page] * KV_PAGE_CELLS + pos % KV_PAGE_CELLS;
}
extern "C" uint32_t imparo_metal_kv_move_rows(uint32_t layer, uint64_t k_stride,
                                              uint64_t v_stride, const uint32_t * from,
                                              const uint32_t * to, uint32_t n) {
    if (layer >= g.kv_k.size() || layer >= g.kv_v.size() || k_stride == 0 || v_stride == 0) {
        return 0u;
    }
    id<MTLBuffer> kb = g.kv_k[layer];
    id<MTLBuffer> vb = g.kv_v[layer];
    if (kb == nil || vb == nil) { return 0u; }
    const uint64_t k_reg = kv_reg(layer, false);
    const uint64_t v_reg = kv_reg(layer, true);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t top = std::max(kv_host_slot(layer, from[i]), kv_host_slot(layer, to[i])) + 1;
        if (k_reg + top * k_stride > [kb length] || v_reg + top * v_stride > [vb length]) {
            return 0u;
        }
    }
    uint8_t * k = (uint8_t *)[kb contents] + k_reg;
    uint8_t * v = (uint8_t *)[vb contents] + v_reg;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t src = kv_host_slot(layer, from[i]);
        const uint64_t dst = kv_host_slot(layer, to[i]);
        if (src == dst) { continue; }
        memcpy(k + dst * k_stride, k + src * k_stride, (size_t)k_stride);
        memcpy(v + dst * v_stride, v + src * v_stride, (size_t)v_stride);
    }
    return 1u;
}

// THE GATED DELTA RULE, one threadgroup per value head, the batch's tokens looped inside
// the kernel so the head's state matrix is loaded once and stored once. Returns false
// when the pipeline does not exist -- the model then refuses, because a no-op here leaves
// the output buffer holding the previous layer's values, which is a plausible answer.
extern "C" bool imparo_metal_delta_net(uint32_t qkv, uint32_t alpha, uint32_t beta,
                                       uint64_t a_off, uint64_t dt_off,
                                       uint32_t state, uint32_t state_off,
                                       uint32_t state_out_off, uint32_t out,
                                       uint32_t k_heads, uint32_t v_heads,
                                       uint32_t key_dim, uint32_t value_dim,
                                       uint32_t n_tok, float eps,
                                       uint64_t norm_w_off, uint32_t gate,
                                       uint32_t snap, uint32_t snap_off,
                                       uint32_t snap_row) {
    if (g.p_delta_net == nil) { return false; }
    // The dims are compiled into the kernel's register array; a mismatch would read the
    // state with the wrong stride and still produce numbers.
    if (key_dim != g_delta_kd || value_dim != g_delta_vd) { return false; }
    if (g_skip_cat == PC_DELTA) { return true; }
    // THE FUSED EPILOGUE writes the GATE, not `out`. Both are declared to the hazard
    // tracker either way: a slot the kernel does not touch this call still has to be
    // bound (a flag-guarded binding is bound ALWAYS -- task #152), and naming a buffer
    // that is only read as written costs one barrier, never a wrong answer.
    const bool fuse = norm_w_off != UINT64_MAX;
    // A caller that asked for the epilogue and a kernel that cannot reproduce its bits is
    // a wrong answer waiting to be believed: refuse the whole rule instead, which the
    // model turns into an error. Same predicate as the capability, read once.
    if (fuse && imparo_metal_delta_net_fuses_epilogue() == 0u) { return false; }
    // THE BOUNDARY SNAPSHOT: a checkpoint boundary `snap_row` tokens into the batch, and
    // the plane that receives the matrix as of that row. None = UINT32_MAX, and then
    // the slot is bound to the state buffer itself (bound always, task #152) with
    // snap_row 0, which the kernel never reaches.
    const bool snapping = snap != UINT32_MAX;
    haz(hb(qkv) | hb(alpha) | hb(beta) | hb(state) | (fuse ? hb(gate) : 0u),
        (fuse ? hb(gate) : hb(out)) | hb(state) | (snapping ? hb(snap) : 0u));
    [g.enc setComputePipelineState:g.p_delta_net];
    [g.enc setBuffer:g.bufs[qkv] offset:g.buf_off[qkv] atIndex:0];
    [g.enc setBuffer:g.bufs[alpha] offset:g.buf_off[alpha] atIndex:1];
    [g.enc setBuffer:g.bufs[beta] offset:g.buf_off[beta] atIndex:2];
    { const WSeg & ws = wseg_use(a_off);
      [g.enc setBuffer:ws.buf offset:(NSUInteger)(a_off - ws.base) atIndex:3]; }
    { const WSeg & ws = wseg_use(dt_off);
      [g.enc setBuffer:ws.buf offset:(NSUInteger)(dt_off - ws.base) atIndex:4]; }
    [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
             atIndex:5];
    [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] atIndex:6];
    [g.enc setBytes:&k_heads length:4 atIndex:7];
    [g.enc setBytes:&v_heads length:4 atIndex:8];
    [g.enc setBytes:&n_tok length:4 atIndex:9];
    [g.enc setBytes:&eps length:4 atIndex:10];
    // The plane the updated matrix lands in; the same offset is the in-place form.
    [g.enc setBuffer:g.bufs[state]
              offset:g.buf_off[state] + (NSUInteger)state_out_off * 4 atIndex:11];
    // The gated-RMS epilogue. Both slots are bound on EVERY call -- the weight segment
    // and the gate stand in for themselves when the epilogue is off -- so no dispatch
    // leans on a stale binding and the validation layer stays clean (#152).
    // The stand-in is `a_off`, not 0: offset 0 is in NO weight segment and wseg_at
    // aborts on it. A flag-guarded slot is bound on every call (#152), so the stand-in
    // has to be an offset that actually resolves -- any real one will do, because
    // fuse_epi == 0 means the kernel never reads it.
    { const uint64_t w = fuse ? norm_w_off : a_off;
      const WSeg & ws = wseg_use(w);
      [g.enc setBuffer:ws.buf offset:(NSUInteger)(w - ws.base) atIndex:12]; }
    { const uint32_t gb = fuse ? gate : out;
      [g.enc setBuffer:g.bufs[gb] offset:g.buf_off[gb] atIndex:13]; }
    const uint32_t fuse_flag = fuse ? 1u : 0u;
    [g.enc setBytes:&fuse_flag length:4 atIndex:14];
    { const uint32_t sb = snapping ? snap : state;
      const uint32_t so = snapping ? snap_off : state_off;
      [g.enc setBuffer:g.bufs[sb] offset:g.buf_off[sb] + (NSUInteger)so * 4 atIndex:15]; }
    const uint32_t snap_row_v = snapping ? snap_row : 0u;
    [g.enc setBytes:&snap_row_v length:4 atIndex:16];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_DELTA); }
    [g.enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(g_delta_sgs * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// THE ROWS FORM OF THE DELTA RULE, one dispatch for every row. Each row's matrix lives in its own
// slot -- a different Metal buffer, which a single binding cannot reach -- so the addresses go in
// the row table and the buffers are declared with `useResources`, the shape plan step 7b gave the
// other per-row ops. The rule's own cost barely changes with the row count (38.5 us a call at one
// row, 44.7 at two, measured per encoder), so what this removes is the second dispatch, not work.
// False, with nothing encoded, when the pipeline cannot be built or two rows share a slot; the
// caller then runs the per-row loop.
// A/B: IMPARO_DELTA_ROWS=0 keeps the per-row loop, so one binary times both arms.
static bool delta_rows_on() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_DELTA_ROWS"); v = (e && e[0] == '0') ? 0 : 1; }
    return v == 1;
}
static bool cob_delta_net(uint32_t qkv, uint32_t alpha, uint32_t beta, uint64_t a_off,
                          uint64_t dt_off, uint32_t state, const uint32_t * slots,
                          const uint32_t * state_off, const uint32_t * state_out_off,
                          uint32_t n_rows, uint32_t out, uint32_t k_heads, uint32_t v_heads,
                          uint32_t key_dim, uint32_t value_dim, float eps, uint64_t norm_w_off,
                          uint32_t gate) {
    if (!delta_rows_on() || g.enc == nil || n_rows == 0u || g.p_delta_net == nil
        || key_dim != g_delta_kd || value_dim != g_delta_vd
        || !slots_distinct(slots, n_rows)) { return false; }
    const bool fuse = norm_w_off != UINT64_MAX;
    if (fuse && imparo_metal_delta_net_fuses_epilogue() == 0u) { return false; }
    id<MTLComputePipelineState> p = cob_pipeline(g.cob_delta_net, @"imparo_delta_net", false);
    if (p == nil) { return false; }
    static std::vector<CobRowHost> tab;
    id<MTLBuffer> any_state = nil;
    const bool ok = cob_gather(slots, n_rows, tab, MTLResourceUsageRead | MTLResourceUsageWrite,
        [&](uint32_t r, CobRowHost & e, std::vector<id<MTLResource>> & used) {
            id<MTLBuffer> b = g.bufs[state];   // the selected slot's
            if (b == nil) { return false; }
            used.push_back(b);
            const uint64_t base = (uint64_t)[b gpuAddress] + g.buf_off[state];
            e.state = base + (uint64_t)state_off[r] * 4u;
            e.state_out = base + (uint64_t)state_out_off[r] * 4u;
            if (any_state == nil) { any_state = b; }
            return true;
        });
    if (!ok || any_state == nil) { return false; }
    if (g_skip_cat == PC_DELTA) { return true; }
    haz(hb(qkv) | hb(alpha) | hb(beta) | hb(state) | (fuse ? hb(gate) : 0u),
        (fuse ? hb(gate) : hb(out)) | hb(state));
    const uint32_t one = 1u, fuse_flag = fuse ? 1u : 0u, snap_row_v = 0u;
    for (uint32_t r0 = 0; r0 < n_rows; r0 += COB_TABLE_ROWS) {
        const uint32_t n = std::min(COB_TABLE_ROWS, n_rows - r0);
        const uint64_t qkv_row = ((uint64_t)2 * k_heads * key_dim + (uint64_t)v_heads * value_dim) * 4u;
        const uint64_t head_row = (uint64_t)v_heads * 4u;
        const uint64_t out_row = (uint64_t)v_heads * value_dim * 4u;
        [g.enc setComputePipelineState:p];
        [g.enc setBuffer:g.bufs[qkv]   offset:g.buf_off[qkv]   + (NSUInteger)(r0 * qkv_row)  atIndex:0];
        [g.enc setBuffer:g.bufs[alpha] offset:g.buf_off[alpha] + (NSUInteger)(r0 * head_row) atIndex:1];
        [g.enc setBuffer:g.bufs[beta]  offset:g.buf_off[beta]  + (NSUInteger)(r0 * head_row) atIndex:2];
        { const WSeg & ws = wseg_use(a_off);
          [g.enc setBuffer:ws.buf offset:(NSUInteger)(a_off - ws.base) atIndex:3]; }
        { const WSeg & ws = wseg_use(dt_off);
          [g.enc setBuffer:ws.buf offset:(NSUInteger)(dt_off - ws.base) atIndex:4]; }
        // Slots 5, 11 and 15 stand in: under COB_ROWS the matrix comes from the table and the
        // snapshot is not taken, but a flag-guarded binding is bound on every call (#152).
        [g.enc setBuffer:any_state offset:0 atIndex:5];
        [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] + (NSUInteger)(r0 * out_row) atIndex:6];
        [g.enc setBytes:&k_heads length:4 atIndex:7];
        [g.enc setBytes:&v_heads length:4 atIndex:8];
        [g.enc setBytes:&one     length:4 atIndex:9];
        [g.enc setBytes:&eps     length:4 atIndex:10];
        [g.enc setBuffer:any_state offset:0 atIndex:11];
        { const uint64_t w = fuse ? norm_w_off : a_off;
          const WSeg & ws = wseg_use(w);
          [g.enc setBuffer:ws.buf offset:(NSUInteger)(w - ws.base) atIndex:12]; }
        { const uint32_t gb = fuse ? gate : out;
          [g.enc setBuffer:g.bufs[gb] offset:g.buf_off[gb] + (NSUInteger)(r0 * out_row) atIndex:13]; }
        [g.enc setBytes:&fuse_flag length:4 atIndex:14];
        [g.enc setBuffer:any_state offset:0 atIndex:15];
        [g.enc setBytes:&snap_row_v length:4 atIndex:16];
        [g.enc setBytes:tab.data() + r0 length:n * sizeof(CobRowHost) atIndex:17];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_DELTA); }
        [g.enc dispatchThreadgroups:MTLSizeMake(v_heads, n, 1)
              threadsPerThreadgroup:MTLSizeMake(g_delta_sgs * 32u, 1, 1)];
        if (g_prof) { prof_end(); }
    }
    return true;
}

// The one-token delta rule for co-batched rows: row t reads row t of `qkv`, `alpha` and `beta`, its
// matrix at state_off[t] in its slot's `state`, writes the advanced matrix at state_out_off[t]
// there and row t of `out` (of `gate` with the fused epilogue), by the one-token step's kernel:
// the rows select their slots in turn. False when a slot cannot be selected or the rule refuses
// a row; rows before it were encoded, so the caller fails the step.
extern "C" bool imparo_metal_delta_net_slot_rows(uint32_t qkv, uint32_t alpha, uint32_t beta,
                                                 uint64_t a_off, uint64_t dt_off, uint32_t state,
                                                 const uint32_t * slots,
                                                 const uint32_t * state_off,
                                                 const uint32_t * state_out_off, uint32_t n_rows,
                                                 uint32_t out, uint32_t k_heads,
                                                 uint32_t v_heads, uint32_t key_dim,
                                                 uint32_t value_dim, float eps,
                                                 uint64_t norm_w_off, uint32_t gate) {
    const bool fuse = norm_w_off != UINT64_MAX;
    const uint64_t qkv_row = ((uint64_t)2 * k_heads * key_dim + (uint64_t)v_heads * value_dim) * 4u;
    const uint64_t head_row = (uint64_t)v_heads * 4u;
    const uint64_t out_row = (uint64_t)v_heads * value_dim * 4u;
    if (cob_delta_net(qkv, alpha, beta, a_off, dt_off, state, slots, state_off, state_out_off,
                      n_rows, out, k_heads, v_heads, key_dim, value_dim, eps, norm_w_off, gate)) {
        return true;
    }
    const uint32_t cur = g_slot_cur;
    bool ok = true;
    for (uint32_t r = 0; r < n_rows && ok; ++r) {
        if (imparo_metal_select_slot(slots[r]) != 0) { ok = false; break; }
        RowRebase q(qkv, r * qkv_row);
        RowRebase a(alpha, r * head_row);
        RowRebase bt(beta, r * head_row);
        RowRebase o(out, r * out_row);
        // The gate is only rebased when it is its own buffer; unfused it stands in as `out`.
        const bool gate_row = fuse && gate != out && gate < B_COUNT;
        const uint64_t gate0 = gate_row ? g.buf_off[gate] : 0ull;
        if (gate_row) { g.buf_off[gate] += r * out_row; }
        ok = imparo_metal_delta_net(qkv, alpha, beta, a_off, dt_off, state, state_off[r],
                                    state_out_off[r], out, k_heads, v_heads, key_dim, value_dim,
                                    1u, eps, norm_w_off, gate, UINT32_MAX, 0u, 0u);
        if (gate_row) { g.buf_off[gate] = gate0; }
    }
    imparo_metal_select_slot(cur);
    return ok;
}

// THE GATED DELTA RULE OVER A DRAFT TREE (imparo_delta_net_tree): every node's output from the
// state and its own ancestors, the state only read; the inputs are the serial rule's, plus the
// row layout and the rows with more than one child as a mask (`keep_lo` rows 0..31, `keep_hi`
// 32..63). The rows must be depth-first and shallower than imparo_metal_delta_tree_depth(): the
// Rust entry checks the words it was handed and derives the mask from them, and the kernel
// clamps a depth to its slots, so a layout that slipped past gives wrong numbers but never
// reaches outside them. `layout_words` is the row width of the caller's words, checked here
// against the layout buffer's size.
extern "C" bool imparo_metal_delta_net_tree(uint32_t qkv, uint32_t alpha, uint32_t beta,
                                            uint64_t a_off, uint64_t dt_off,
                                            uint32_t state, uint32_t state_off, uint32_t out,
                                            uint32_t layout, uint32_t layout_words,
                                            uint32_t keep_lo, uint32_t keep_hi,
                                            uint32_t k_heads, uint32_t v_heads,
                                            uint32_t key_dim, uint32_t value_dim,
                                            uint32_t n_tok, float eps) {
    if (g.p_delta_net_tree == nil) { return false; }
    if (key_dim != g_delta_kd || value_dim != g_delta_vd) { return false; }
    if (n_tok == 0u || n_tok > 64u || layout >= B_COUNT || g.bufs[layout] == nil
        || (uint64_t)n_tok * layout_words * 4u > g.sizes[layout]) { return false; }
    if (g_skip_cat == PC_DELTA) { return true; }
    haz(hb(qkv) | hb(alpha) | hb(beta) | hb(state) | hb(layout), hb(out));
    [g.enc setComputePipelineState:g.p_delta_net_tree];
    [g.enc setBuffer:g.bufs[qkv] offset:g.buf_off[qkv] atIndex:0];
    [g.enc setBuffer:g.bufs[alpha] offset:g.buf_off[alpha] atIndex:1];
    [g.enc setBuffer:g.bufs[beta] offset:g.buf_off[beta] atIndex:2];
    { const WSeg & ws = wseg_use(a_off);
      [g.enc setBuffer:ws.buf offset:(NSUInteger)(a_off - ws.base) atIndex:3]; }
    { const WSeg & ws = wseg_use(dt_off);
      [g.enc setBuffer:ws.buf offset:(NSUInteger)(dt_off - ws.base) atIndex:4]; }
    [g.enc setBuffer:g.bufs[state] offset:g.buf_off[state] + (NSUInteger)state_off * 4
             atIndex:5];
    [g.enc setBuffer:g.bufs[out] offset:g.buf_off[out] atIndex:6];
    [g.enc setBytes:&k_heads length:4 atIndex:7];
    [g.enc setBytes:&v_heads length:4 atIndex:8];
    [g.enc setBytes:&n_tok length:4 atIndex:9];
    [g.enc setBytes:&eps length:4 atIndex:10];
    [g.enc setBuffer:g.bufs[layout] offset:g.buf_off[layout] atIndex:11];
    [g.enc setBytes:&keep_lo length:4 atIndex:12];
    [g.enc setBytes:&keep_hi length:4 atIndex:13];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_DELTA); }
    [g.enc dispatchThreadgroups:MTLSizeMake(v_heads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(g_delta_sgs * 32u, 1, 1)];
    if (g_prof) { prof_end(); }
    return true;
}

// The tree delta kernel's depth limit; every node must be shallower.
extern "C" uint32_t imparo_metal_delta_tree_depth(void) { return g_delta_tree_depth; }

extern "C" void imparo_metal_mul_sigmoid(uint32_t a, uint32_t b, uint32_t n,
                                         uint32_t b_off, uint32_t b_stride,
                                         uint32_t a_stride, uint32_t n_row) {
    if (g_skip_cat == PC_MUL_STRIDED) { return; }
    if (g.p_mul_sigmoid == nil) { return; }
    haz(hb(a) | hb(b), hb(a));
    [g.enc setComputePipelineState:g.p_mul_sigmoid];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBytes:&n length:4 atIndex:2];
    [g.enc setBytes:&b_off length:4 atIndex:3];
    [g.enc setBytes:&b_stride length:4 atIndex:4];
    [g.enc setBytes:&a_stride length:4 atIndex:5];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MUL_STRIDED); }
    [g.enc dispatchThreads:MTLSizeMake(n, n_row, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

extern "C" void imparo_metal_copy_strided(uint32_t dst, uint32_t src, uint32_t width,
                                          uint32_t src_off, uint32_t src_stride,
                                          uint32_t n_row) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    if (g.p_copy_strided == nil) { return; }
    haz(hb(src), hb(dst));
    [g.enc setComputePipelineState:g.p_copy_strided];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBytes:&width length:4 atIndex:2];
    [g.enc setBytes:&src_off length:4 atIndex:3];
    [g.enc setBytes:&src_stride length:4 atIndex:4];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
    [g.enc dispatchThreads:MTLSizeMake(width, n_row, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
}

extern "C" void imparo_metal_scatter_strided(uint32_t dst, uint32_t src, uint32_t width,
                                             uint32_t dst_off, uint32_t dst_stride,
                                             uint32_t n_row) {
    if (g_skip_cat == PC_ELEMENTWISE) { return; }
    if (g.p_scatter_strided == nil || width == 0u || n_row == 0u) { return; }
    haz(hb(src), hb(dst));
    [g.enc setComputePipelineState:g.p_scatter_strided];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:0];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBytes:&width length:4 atIndex:2];
    [g.enc setBytes:&dst_off length:4 atIndex:3];
    [g.enc setBytes:&dst_stride length:4 atIndex:4];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
    [g.enc dispatchThreads:MTLSizeMake(width, n_row, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    if (g_prof) { prof_end(); }
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
constexpr uint32_t SHORTCONV_MAX_HISTORY_HOST = 8u;   // the shader's SHORTCONV_MAX_HISTORY
struct MegaTokenHost { uint32_t start_pos, dbg_slot, n_tg, entry_bytes, dbg_seq, entry_index, n_entries, probe; };   // mirrors MegaToken
// A weight's segment base address, and its offset local to that segment (the block adds
// them). Absent weights get a valid address they never read and keep their sentinel offset.
// Every referenced buffer is declared to the encoder: it is reached by address, not binding.
static uint64_t mega_waddr(uint64_t off, uint64_t & local) {
    if (w_absent(off)) { local = off; return (uint64_t)[g_wsegs.front().buf gpuAddress]; }
    const WSeg & sg = wseg_use(off);
    local = off - sg.base;
    [g.enc useResource:sg.buf usage:MTLResourceUsageRead];
    return (uint64_t)[sg.buf gpuAddress];
}
// Immutable rope tables for the layer kernel (buffer 3), including the no-factors dummy.
// A single "last table" slot churns on E4B: windowed and full-attention layers alternate
// between the dummy and the model's factors. Each replacement allocated again and left
// the old buffer in the residency set. Keep each distinct table once instead. Content
// equality also covers equal tables at different host addresses. Never overwrite one:
// an earlier encoded layer or queued decode region can still be reading its bytes.
static id<MTLBuffer> mega_freqs_buffer(const float * freqs, uint32_t n_freqs) {
    struct Table { uint32_t n; id<MTLBuffer> buf; };
    static std::vector<Table> tables;
    const bool none = (freqs == nullptr || n_freqs == 0u);
    const uint32_t n = none ? 0u : n_freqs;
    const size_t bytes = none ? sizeof(float) : (size_t)n * sizeof(float);
    for (const Table & table : tables) {
        if (table.n == n && (none || memcmp([table.buf contents], freqs, bytes) == 0)) {
            return table.buf;
        }
    }
    id<MTLBuffer> buf = [g.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf == nil) { return nil; }
    if (none) { *(float *)[buf contents] = 1.0f; } else { memcpy([buf contents], freqs, bytes); }
    rset_add(buf);   // exactly once per immutable table, resident with the model
    tables.push_back({n, buf});
    return buf;
}
// THE ADMISSION PROBE (task #203).
//
// How many threadgroups of a given pipeline this GPU will hold resident at once is the one
// input the grid seat cannot derive: `maxTotalThreadsPerThreadgroup` bounds ONE threadgroup,
// and multiplying it by the core count is a grid-sizing heuristic, not a co-residency proof.
// Both constants ever written into that heuristic were wrong on some machine:
//
//   2 * (cores - 2)   hung an M4 Pro -- every region timed out and ran on the dispatch path
//                     (E4B decode 19.9 tok/s at 36x16 against 60.0 at 20x16, 2026-09-08)
//   cores             its replacement, and 2.6% of this M3 Pro's E4B decode at depth
//
// A constant cannot be right for both, so the seat is MEASURED here, on the machine that is
// running, with the kernel that will run. The probe dispatches the real pipeline in probe mode
// (tok.probe): every threadgroup arrives at one grid barrier and leaves. It reads no entry and
// writes nothing but the sync words, so a grid above admission costs one short spin cap and no
// state -- it can never become the hang the seat itself could produce.
//
// WHY THE REAL PIPELINE. Admission is this kernel's register footprint. A trivial probe kernel
// admits about 2.7x more (handoff/mac-m4-mega-admission.md), so a purpose-built probe would
// measure a number no real dispatch can use.
//
// WHAT IT DOES NOT ANSWER: the threadgroup-memory half. The probe runs at the 16-byte minimum,
// so it measures the register/co-residency ceiling alone; the row's threadgroup-memory cost is
// the separate cap in mega_tgs_for_row, and the two multiply as they did before.
static const uint32_t MEGA_PROBE_SPIN_K = 16u;   // 16 x 1024 spins, ~20 ms: an admitted grid arrives in microseconds, and a grid that does not is a display stall for exactly this long
// IMPARO_MEGA_PROBE_CAP=<n> (test instrument): report "did not admit" for any grid above n,
// WITHOUT dispatching it. The M4 Pro case -- a GPU that holds fewer threadgroups than this one
// -- cannot be produced on an M3 Pro, and a fix for a hang must be exercised, not argued
// (#193's rule). Set it to a number below the core count and the halve-down path runs too.
static uint32_t mega_probe_cap(void) {
    static uint32_t cap = 0u;
    static bool read = false;
    if (!read) {
        read = true;
        if (const char * e = getenv("IMPARO_MEGA_PROBE_CAP")) { cap = (uint32_t)strtoul(e, nullptr, 10); }
    }
    return cap;
}

// A DISCOVERY PROBE, with the same standing as imparo_metal_spill_rate: the TUNER calls it
// (Backend::mega_admission) between regions, it dispatches on its own command buffer and
// waits, and it clears the barrier counters -- so it refuses to run while a region is open.
// The engine never calls it: an untuned host runs one threadgroup per core, a tuned one
// applies the stored value as written, and the tuner only writes a value it measured under
// this limit (docs/tuner-design.md: derive the limit, tune the value).
//
// Threadgroups that reached the barrier. == n_tg means the grid is admitted; 0 means the probe
// could not run at all.
static uint32_t mega_probe_arrivals(id<MTLComputePipelineState> pipe, uint32_t n_tg, uint32_t nsg) {
    if (pipe == nil || g.queue == nil || g.mega_sync == nil || n_tg == 0u || nsg == 0u) { return 0u; }
    if ((uint64_t)nsg * 32u > (uint64_t)[pipe maxTotalThreadsPerThreadgroup]) { return 0u; }
    // The injected verdict comes BEFORE the dispatch: the point is to test the host's ladder
    // and the engine underneath the answer, not to provoke a real timeout we already know how
    // to survive. Report one threadgroup short, which is what a real shortfall looks like.
    if (mega_probe_cap() != 0u && n_tg > mega_probe_cap()) { return n_tg - 1u; }
    id<MTLBuffer> fb = mega_freqs_buffer(nullptr, 0u);
    if (fb == nil) { return 0u; }
    uint32_t * w = (uint32_t *)[g.mega_sync contents];
    // A grid that does not admit never reaches mega_exit's reset, so the counters are cleared
    // on BOTH sides of the dispatch: the next probe must start from zero, and the engine must
    // find the buffer as it left it (mega_check_error reads words 3 and 15).
    memset(w, 0, 16u * sizeof(uint32_t));
    MegaEntryGHost ent; memset(&ent, 0, sizeof(ent));      // never read: the probe leaves before the entry loop
    MegaTokenHost tok; memset(&tok, 0, sizeof(tok));
    tok.n_tg = n_tg;
    tok.entry_bytes = (uint32_t)sizeof(MegaEntryGHost);
    tok.probe = MEGA_PROBE_SPIN_K;
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipe];
    [e setBytes:&ent length:sizeof(ent) atIndex:0];
    [e setBuffer:g.mega_sync offset:0 atIndex:1];
    [e setBytes:&tok length:sizeof(tok) atIndex:2];
    [e setBuffer:fb offset:0 atIndex:3];
    [e setThreadgroupMemoryLength:16 atIndex:0];           // the probe forms no row; 16 = the binding's minimum (task #152)
    [e dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint32_t err = w[15];
    // mega_step's timeout stores the arrivals it saw in bits 16..23; at phase 1 that count IS
    // the number of threadgroups that started.
    const uint32_t arrived = (err == 0u) ? n_tg : ((err >> 16) & 0xffu);
    memset(w, 0, 16u * sizeof(uint32_t));
    return arrived;
}

// The ladder is MULTIPLES OF THE CORE COUNT. A grid that is not a whole number of cores leaves
// some cores hosting one threadgroup and some two, and every barrier waits for the doubled ones
// -- the whole-core rule mega_grid_derive already searches under. Ascending, stopping at the
// first failure, so the only over-capacity dispatch in the run is the one that answers the
// question. If not even one per core admits, halve down; one threadgroup always admits.
static uint32_t mega_probe_seat(id<MTLComputePipelineState> pipe, uint32_t cores, uint32_t nsg,
                                uint32_t tgs_cap, const char * name) {
    uint32_t best = 0u;
    for (uint32_t k = 1u; k * cores <= tgs_cap; ++k) {
        const uint32_t cand = k * cores;
        uint32_t got = mega_probe_arrivals(pipe, cand, nsg);
        // COULD NOT RUN is not DID NOT ADMIT. Answering 0 for a probe that never dispatched
        // and treating it as a verdict would collapse the grid to one threadgroup on any host
        // where the probe is unavailable -- worse than the policy it replaces. Leave the
        // one-per-core policy standing and say which happened.
        if (got == 0u) {
            NSLog(@"imparo metal: mega admission probe %s could not run; the one-per-core policy stands (%u)", name, cores);
            return cores;
        }
        // ONE RETRY. A threadgroup the GPU descheduled for another client (a browser's GPU
        // process took the GPU in bursts here, task #123) reads exactly like a grid that does
        // not admit, and a false negative would cost the grid for the whole process.
        if (got != cand) { got = mega_probe_arrivals(pipe, cand, nsg); }
        if (got != cand) {
            // Never print a seat here when there is not one yet: at k = 1 the ladder has
            // measured nothing and the halve-down below decides. Say which happened.
            if (best != 0u) {
                NSLog(@"imparo metal: mega admission probe %s: %u x %u threads -> %u of %u arrived; the seat is %u",
                      name, cand, nsg * 32u, got, cand, best);
            } else {
                NSLog(@"imparo metal: mega admission probe %s: %u x %u threads -> %u of %u arrived; halving down",
                      name, cand, nsg * 32u, got, cand);
            }
            break;
        }
        best = cand;
    }
    if (best == 0u) {
        // Not even one threadgroup per core. Halve down; Metal guarantees one threadgroup of
        // the pipeline's own maximum width, so 1 is the floor and it always admits.
        for (uint32_t cand = cores / 2u; cand >= 1u; cand /= 2u) {
            if (mega_probe_arrivals(pipe, cand, nsg) == cand) { best = cand; break; }
        }
        if (best == 0u) { best = 1u; }
        NSLog(@"imparo metal: mega admission probe %s: one per core (%u) did NOT admit; seat %u", name, cores, best);
    }
    return best;
}


// MEASURE ONE PIPELINE'S CEILING at its family's seated width: the plain pipeline of (family,
// head-dim slot). The seat is per pipeline (see g_mega_tgs), so the limit is too.
static void mega_measure_ceiling(uint32_t family, uint32_t slot, uint32_t cores) {
    __strong id<MTLComputePipelineState> * const plain[MEGA_ARCH_COUNT] =
        { g.p_mega_layer, g.p_mega_lfm2, g.p_mega_q35, g.p_mega_l2m };
    static const char * const nm[MEGA_ARCH_COUNT] = { "gemma4", "LFM2", "qwen35", "lfm2moe" };
    if (family >= MEGA_ARCH_COUNT || slot >= 2u || g_mega_threads_limit[family] == 0u) { return; }
    id<MTLComputePipelineState> pipe = plain[family][slot];
    if (pipe == nil) { return; }   // no pipeline compiled for this slot (one attention geometry)
    const uint32_t nsg = g_mega_nsg[family];
    if (nsg == 0u || nsg > MEGA_NSG_MAX) { return; }
    const uint32_t cap = std::max(1u, g_mega_threads_limit[family] / std::max(1u, nsg * 32u));
    const uint32_t seat = mega_probe_seat(pipe, cores, nsg, cap, nm[family]);
    g_mega_tgs_probed[family][slot][nsg] = seat;
    NSLog(@"imparo metal: mega admission MEASURED %s slot %u (hd %u): %u threadgroups x %u threads (%.2f per core, cap %u)",
          nm[family], slot, g_qcomb_hds[slot], seat, nsg * 32u, (double)seat / (double)std::max(1u, cores), cap);
}

// Backend::mega_admission (task #203): measure one pipeline's ceiling once per width and
// report it; 0 = no pipeline at that slot, or a region is open (a probe inside a region would
// clear the counters that region spins on -- the phantom-dispatch class of failure, never risked).
extern "C" uint32_t imparo_metal_mega_admission(uint32_t family, uint32_t slot) {
    if (family >= MEGA_ARCH_COUNT || slot >= 2u || g_mega_threads_limit[family] == 0u) { return 0u; }
    if (g.enc != nil) { NSLog(@"imparo metal: mega admission probe refused: a region is open"); return 0u; }
    if (g.mega_sync == nil) { mega_scratch_ensure(0u); }
    if (mega_ceiling(family, slot) == 0u) { mega_measure_ceiling(family, slot, mega_core_seat()); }
    g_mega_probe_family = family;
    return mega_ceiling(family, slot);
}
// What the tuner's probe measured for the family it asked about, per slot (0 = nothing measured
// in this process, or the width has moved since): the LIMIT the registry ranks a grid knob under.
extern "C" uint32_t imparo_metal_mega_admission_current(uint32_t slot) {
    return g_mega_probe_family < MEGA_ARCH_COUNT ? mega_ceiling(g_mega_probe_family, slot) : 0u;
}
// Whether any family compiled a plain pipeline at `slot` -- i.e. whether the model this process
// compiled for has a second attention geometry. The registry's applicability question for the
// second grid knob.
extern "C" uint32_t imparo_metal_mega_slot_present(uint32_t slot) {
    if (slot >= 2u) { return 0u; }
    return (g.p_mega_layer[slot] != nil || g.p_mega_lfm2[slot] != nil || g.p_mega_q35[slot] != nil
            || g.p_mega_l2m[slot] != nil) ? 1u : 0u;
}
// Backend::mega_seat_form: 0 = no pipeline family, 1 = per-layer dispatches at the grid seat,
// 2 = the per-token program at one threadgroup per core (task #153's default for the family,
// or IMPARO_MEGA_PROGRAM). The tuner's applicability question; the same table the entry
// dispatch reads, so the answer cannot drift from what runs.
extern "C" uint32_t imparo_metal_mega_seat_form(uint32_t family) {
    if (family >= MEGA_ARCH_COUNT || g_mega_threads_limit[family] == 0u) { return 0u; }
    // A quantized cache takes the typed pipelines, which run one threadgroup per core
    // (task #156), so with one configured the seat governs no grid either.
    if (g_mega_kq_ty != 1u || g_mega_vq_ty != 1u) { return 2u; }
    return mega_program_for(MEGA_PROGRAM_DEFAULTS[family]) ? 2u : 1u;
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
    if (!said) { said = true; NSLog(@"imparo metal: mega program engaged: %u entries in one dispatch (tgs=%u threads=%u)", g_prog_n, g_prog_tgs, g_prog_nsg * 32u); }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MEGA); }
    g_mega_last_tgs = g_prog_tgs;
    [g.enc dispatchThreadgroups:MTLSizeMake(g_prog_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(g_prog_nsg * 32u, 1, 1)];
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
                             id<MTLComputePipelineState> pipe, uint32_t tgs, uint32_t nsg, uint32_t tgmem, id<MTLBuffer> freqs,
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
        g_prog_pipe = pipe; g_prog_tgs = tgs; g_prog_nsg = nsg; g_prog_tgmem = tgmem; g_prog_freqs = freqs;
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
// WHY AN ENTRY WAS REFUSED, on request (IMPARO_MEGA_REFUSE_LOG=1): the line of the rule that
// refused it. A refusal is silent by design -- the layer takes the dispatch path -- which is
// right for the engine and blind for the tuner, whose mega workload read 42 of 42 layers
// refused with nothing to say which rule (task #203). Off, it costs one static read.
static bool mega_refuse_log(void) {
    static int on = -1;
    if (on < 0) { on = getenv("IMPARO_MEGA_REFUSE_LOG") != nullptr ? 1 : 0; }
    return on != 0;
}
#define MEGA_REFUSE() do { if (mega_refuse_log()) { NSLog(@"imparo metal: mega entry refused (imparo_metal.mm:%d)", __LINE__); } return false; } while (0)
extern "C" bool imparo_metal_mega_layer(const MegaEntryFfi * e) {
    MegaProgRefuseGuard guard;   // a refusal flushes the pending program run first (task #153)
    if (e == nullptr || e->arch >= MEGA_ARCH_COUNT || mega_level() < (int)e->min_level || !mega_route_open() || g.mega_sync == nil) { MEGA_REFUSE(); }
    mega_inject_if_pending();
    const bool g4 = e->arch == MEGA_ARCH_GEMMA4;
    const bool attn_on = e->attn_on != 0u;
    const uint32_t hd = e->head_dim;
    const uint32_t n_embd = e->u[MEGA_U_N_EMBD], n_heads = e->u[MEGA_U_N_HEADS], n_kv = e->u[MEGA_U_N_KV];
    const uint32_t window = e->u[MEGA_U_WINDOW], had_k = e->u[MEGA_U_HAD_K], had_v = e->u[MEGA_U_HAD_V];
    if (n_embd == 0u || n_embd % 4u != 0u) { MEGA_REFUSE(); }
    // THE ROW MUST FIT. Every threadgroup forms the layer's row in threadgroup memory, and a
    // core's budget is finite (32 KB here): at n_embd 8192 the row alone exceeds it and the
    // dispatch would be illegal, not slow. Refuse instead -- the layer takes the dispatch
    // path, which has no such bound. Task #175 replaces the whole-row staging with a k-slice
    // and removes the bound; until then this is what keeps a wide model running at all.
    // What this entry's phases actually need in threadgroup memory. ZERO MEANS ZERO: every
    // program sets this word, and a phase that reads its activation from device memory needs
    // no row at all -- reading 0 as "assume the row" would silently stage what nothing reads
    // and spend the core's admission on it.
    const uint32_t tgmem_f = e->u[MEGA_U_TGMEM_F];
    const uint32_t tgmem_b = ((tgmem_f * 4u) + 15u) & ~15u;   // 16-byte multiples (task #152)
    if (g_tgmem_limit != 0u && (uint64_t)tgmem_b + MEGA_TG_STATIC_BYTES > g_tgmem_limit) {
        static bool said = false;
        if (!said) {
            said = true;
            NSLog(@"imparo metal: mega route refused -- the layer's row needs %llu B of threadgroup memory and a core has %u (n_embd=%u); the dispatch path takes it",
                  (uint64_t)tgmem_b + MEGA_TG_STATIC_BYTES, g_tgmem_limit, n_embd);
        }
        MEGA_REFUSE();
    }
    // The pipeline family and the head-dim slot (only the attention body depends on the head dim).
    uint32_t slot_of = 2u;
    for (uint32_t i = 0; i < 2u; ++i) { if (hd != 0u && g_qcomb_hds[i] == hd) { slot_of = i; break; } }
    // A quantized cache (task #156) selects the _q variants, built with the process-wide K / V
    // types; a layer whose types differ from that pair is refused (the dispatch path's rule).
    const uint32_t kv_kt = attn_on ? kv_eff_type(e->kv_layer, 0) : 1u, kv_vt = attn_on ? kv_eff_type(e->kv_layer, 1) : 1u;
    const bool kvq = kv_kt != 1u || kv_vt != 1u;
    if (kvq && (kv_kt != g_mega_kq_ty || kv_vt != g_mega_vq_ty)) { MEGA_REFUSE(); }
    // The program form's default is per architecture (task #153): a table, so a new
    // architecture adds a row rather than another ternary.
    const bool * const prog_default = MEGA_PROGRAM_DEFAULTS;
    const bool prog = mega_program_for(prog_default[e->arch]);
    __strong id<MTLComputePipelineState> * const plain_of[MEGA_ARCH_COUNT] = { g.p_mega_layer, g.p_mega_lfm2, g.p_mega_q35, g.p_mega_l2m };
    __strong id<MTLComputePipelineState> * const qp_of[MEGA_ARCH_COUNT]    = { g.p_mega_layer_q, g.p_mega_lfm2_q, g.p_mega_q35_q, g.p_mega_l2m_q };
    __strong id<MTLComputePipelineState> * const deepp_of[MEGA_ARCH_COUNT] = { g.p_mega_layer_deep, g.p_mega_lfm2_deep, g.p_mega_q35_deep, g.p_mega_l2m_deep };
    __strong id<MTLComputePipelineState> * const deepq_of[MEGA_ARCH_COUNT] = { g.p_mega_layer_deep_q, g.p_mega_lfm2_deep_q, g.p_mega_q35_deep_q, g.p_mega_l2m_deep_q };
    __strong id<MTLComputePipelineState> * const progp_of[MEGA_ARCH_COUNT] = { g.p_mega_layer_prog, g.p_mega_lfm2_prog, g.p_mega_q35_prog, g.p_mega_l2m_prog };
    __strong id<MTLComputePipelineState> * const progq_of[MEGA_ARCH_COUNT] = { g.p_mega_layer_prog_q, g.p_mega_lfm2_prog_q, g.p_mega_q35_prog_q, g.p_mega_l2m_prog_q };
    __strong id<MTLComputePipelineState> * plain = plain_of[e->arch];
    __strong id<MTLComputePipelineState> * qp    = qp_of[e->arch];
    __strong id<MTLComputePipelineState> * deepp = deepp_of[e->arch];
    __strong id<MTLComputePipelineState> * deepq = deepq_of[e->arch];
    __strong id<MTLComputePipelineState> * progp = progp_of[e->arch];
    __strong id<MTLComputePipelineState> * progq = progq_of[e->arch];
    id<MTLComputePipelineState> pipe = nil;
    uint32_t seat_slot = 0u;   // the pipeline slot whose grid seat this dispatch takes (task #203)
    if (attn_on) {
        if (slot_of >= 2u) { MEGA_REFUSE(); }
        pipe = prog ? (kvq ? progq[slot_of] : progp[slot_of]) : (kvq ? qp[slot_of] : plain[slot_of]);
        seat_slot = slot_of;
    } else {
        // Without the attention phase the head dim plays no part: any built slot serves the layer.
        __strong id<MTLComputePipelineState> * set = prog ? progp : plain;
        seat_slot = set[0] != nil ? 0u : 1u;
        pipe = set[seat_slot];
    }
    if (pipe == nil) { MEGA_REFUSE(); }
    // The quantized and program variants run at one threadgroup per core (tasks #156, #153).
    // The grid is this FAMILY's (see g_mega_threads_limit): another architecture's kernel
    // never moves it.
    // The threadgroup count comes first -- it is capped by what this threadgroup's own row
    // costs, and a grid whose threadgroups cannot be co-resident waits at its first barrier
    // for threadgroups that never start.
    uint32_t tgs = (kvq || prog) ? mega_deep_tgs(e->arch, seat_slot) : mega_tgs_for_row(e->arch, seat_slot, tgmem_b);
    // Then the simdgroups. The seat is the FLOOR (the device's occupancy term, which the tuner
    // ranks); above it the count comes from what this entry's phases run over, so no phase pays
    // for a mostly-idle last wave. IMPARO_MEGA_NSG_EXACT=1 pins the seat instead, for the A/B.
    // TWO bounds, and they coincide only at tgs == gpu_cores. The pipeline's own verdict caps
    // ONE threadgroup; the family's co-residency limit caps the GRID, because every threadgroup
    // waits on every other one -- a grid above it spins on threadgroups that never start, which
    // panicked this Mac on 2026-09-05. Today tgs IS the core count (the row admits one per
    // core) so the two agree; at any other tgs they do not, so take the smaller.
    const uint32_t nsg_pipe = std::min(32u, (uint32_t)([pipe maxTotalThreadsPerThreadgroup] / 32u));
    const uint32_t nsg_res  = g_mega_threads_limit[e->arch] != 0u
                            ? std::max(1u, g_mega_threads_limit[e->arch] / std::max(1u, tgs * 32u))
                            : nsg_pipe;
    uint32_t nsg = std::min(g_mega_nsg[e->arch], std::max(1u, std::min(nsg_pipe, nsg_res)));
    if (!g_mega_nsg_exact) {
        // `tgs` above is the CAP (the seat, capped by what the row costs a core). The search
        // may take fewer threadgroups when that divides the phases better, but never fewer
        // than the cores -- below that a core sits idle for the whole dispatch.
        const uint32_t cores = g_gpu_cores > 0u ? g_gpu_cores : 7u;
        const MegaGrid gd = mega_grid_derive(e->arch, e->u, std::min(cores, tgs), tgs,
                                             nsg, nsg_pipe, attn_on, hd);
        if (gd.nsg != 0u) {
            tgs = gd.tgs; nsg = gd.nsg;
            g_mega_derived_tgs[e->arch] = gd.tgs; g_mega_derived_nsg[e->arch] = gd.nsg;
        }
    }
    bool deep = false;
    uint32_t attn_split = 1u, n_pos = 0u;
    if (attn_on) {
        // The attention phase: one token, a span within the vec limit or the grouped deep body,
        // the fold geometry (nsg a multiple of the slot count, at least twice it), the block's
        // threadgroup row holding the fold scratch, this layer's cache present.
        if (n_kv == 0u || n_heads == 0u || n_heads % n_kv != 0u) { MEGA_REFUSE(); }
        if (!kvq && !prog && n_heads > tgs) { MEGA_REFUSE(); }   // the f16 plain pipelines compute one vec item per threadgroup (constraint 8)
        if (!mega_had_ok(had_k, hd) || !mega_had_ok(had_v, hd)) { MEGA_REFUSE(); }   // the cache basis (task #156)
        if (!mega_nsg_legal(nsg, e->arch, true, hd)) { MEGA_REFUSE(); }   // the fold geometry, stated once above
        if ((uint64_t)n_embd < (uint64_t)4u * (hd + 2u)) { MEGA_REFUSE(); }
        if (e->kv_layer >= g.kv_k.size() || g.kv_k[e->kv_layer] == nil || g.kv_v[e->kv_layer] == nil) { MEGA_REFUSE(); }
        n_pos = (window > 0 && e->start_pos + 1 > window) ? window : e->start_pos + 1;
        // Threadgroups per head: the idle ones take a share of the span; the partials live in the
        // block's scratch (scratch_rows floats), which bounds the split. The vec regime cap
        // (attn_vec_max_keys) is a per-threadgroup span; past it the grouped deep body (task
        // #151) takes the layer, or the refusal stands with IMPARO_MEGA_ATTN_SLICED=0 (the A/B).
        attn_split = std::max(1u, std::min(tgs / n_heads, e->scratch_rows / std::max(1u, n_heads * (hd + 2u))));
        if ((n_pos + attn_split - 1u) / attn_split > g_attn_vec_max_keys) {
            id<MTLComputePipelineState> dpipe = prog ? pipe : (kvq ? deepq[slot_of] : deepp[slot_of]);
            if (!mega_attn_sliced() || dpipe == nil) { MEGA_REFUSE(); }
            attn_split = mega_attn_deep_slices(e->arch, seat_slot, n_heads, n_kv, hd);
            if (attn_split == 0u) { MEGA_REFUSE(); }
            deep = true; pipe = dpipe; tgs = mega_deep_tgs(e->arch, seat_slot);
            // The grid just changed, so the balance the simdgroup count was chosen for is
            // stale; choose it again from the same floor against the deep grid.
            if (!g_mega_nsg_exact) {
                // The deep body fixes its own threadgroup count, so only nsg is free here.
                const uint32_t deep_cap = std::max(1u, std::min(nsg_pipe,
                    g_mega_threads_limit[e->arch] != 0u
                        ? std::max(1u, g_mega_threads_limit[e->arch] / std::max(1u, tgs * 32u)) : nsg_pipe));
                const MegaGrid gd = mega_grid_derive(e->arch, e->u, tgs, tgs,
                                                     std::min(g_mega_nsg[e->arch], deep_cap),
                                                     nsg_pipe, attn_on, hd);
                if (gd.nsg != 0u) { nsg = gd.nsg; g_mega_derived_nsg[e->arch] = gd.nsg; }
            }
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
            if (w_absent(sl.off) || !w_fast(sl.off)) { MEGA_REFUSE(); }
            break;
        case MEGA_SLOT_WEIGHT_OPT:
            if (!w_fast(sl.off)) { MEGA_REFUSE(); }
            break;
        case MEGA_SLOT_BUF_R: case MEGA_SLOT_BUF_W: case MEGA_SLOT_BUF_RW:
            if (sl.id >= B_COUNT || g.bufs[sl.id] == nil) { MEGA_REFUSE(); }
            if (sl.role != MEGA_SLOT_BUF_W) { rd |= hb(sl.id); }
            if (sl.role != MEGA_SLOT_BUF_R) { wr |= hb(sl.id); }
            break;
        case MEGA_SLOT_KV_K_R: case MEGA_SLOT_KV_K_W: case MEGA_SLOT_KV_V_R: case MEGA_SLOT_KV_V_W: {
            if (!attn_on) { MEGA_REFUSE(); }
            const uint64_t hz = (sl.role == MEGA_SLOT_KV_V_R || sl.role == MEGA_SLOT_KV_V_W) ? HZ_KVV : HZ_KVK;
            rd |= hz;
            if (sl.role == MEGA_SLOT_KV_K_W || sl.role == MEGA_SLOT_KV_V_W) { wr |= hz; }
            break;
        }
        case MEGA_SLOT_KV_PT:
            if (!attn_on || e->kv_layer >= g.kv_pt.size()) { MEGA_REFUSE(); }
            break;
        default: MEGA_REFUSE();
        }
    }
    // EVERY EXPERT THROUGH ONE BASE: the lfm2moe phases address expert e at e * stride past the
    // stack's segment base, so the whole stack must lie in that segment -- the dispatch path's
    // moe_grouped rule. The slot pass above checked only where each stack starts.
    if (e->arch == MEGA_ARCH_LFM2MOE) {
        const uint64_t ne = e->u[L2M_U_N_EXPERT];
        const uint32_t stacks[3][2] = { { L2M_W_GATE, L2M_U_STRIDE_GATE }, { L2M_W_UP, L2M_U_STRIDE_UP },
                                        { L2M_W_DOWN, L2M_U_STRIDE_DOWN } };
        for (uint32_t i = 0; i < 3u; ++i) {
            const uint64_t off = e->slots[stacks[i][0]].off, span = ne * (uint64_t)e->u[stacks[i][1]];
            if (span == 0ull || !w_pair_in_one_segment(off, off + span - 1ull)) { MEGA_REFUSE(); }
        }
    }
    if (!mega_scratch_ensure(e->scratch_rows)) { MEGA_REFUSE(); }
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
    if (fb == nil) { MEGA_REFUSE(); }
    if (prog) {
        // The program form (task #153): the entry joins the pending run; the run is one dispatch.
        mega_prog_alloc();
        if (!mega_prog_record(g_mega_prog, (uint32_t)sizeof(MegaEntryGHost), &ent, pipe, tgs, nsg, tgmem_b, fb,
                              e->start_pos, attn_on, tok.dbg_seq, rd, wr)) { MEGA_REFUSE(); }
        guard.ok = true;
        return true;
    }
    [g.enc setBytes:&ent length:sizeof(ent) atIndex:0];
    [g.enc setBuffer:g.mega_sync offset:0 atIndex:1];
    [g.enc setBytes:&tok length:sizeof(tok) atIndex:2];
    [g.enc setBuffer:fb offset:0 atIndex:3];
    // What this entry's phases need in threadgroup memory: the row when a phase reuses it
    // across its simdgroups, a few bytes when every phase works on device memory (task #175).
    [g.enc setThreadgroupMemoryLength:(NSUInteger)tgmem_b atIndex:0];
    static const char * const arch_name[MEGA_ARCH_COUNT] = { "gemma4", "LFM2", "qwen35", "lfm2moe" };
    static bool said[MEGA_ARCH_COUNT] = { false, false, false, false };
    if (!said[e->arch]) {
        said[e->arch] = true;
        NSLog(@"imparo metal: mega %s layer block engaged (level=%d attn=%u kv_write=%u n_embd=%u scratch_rows=%u tgs=%u threads=%u)",
              arch_name[e->arch], mega_level(), attn_on ? 1u : 0u, e->kv_write, n_embd, e->scratch_rows, tgs, nsg * 32u);
    }
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MEGA); }
    g_mega_last_tgs = tgs;
    [g.enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
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

// The pool's per-block advice. Nothing to do on this backend: a block lives in a resident view,
// and the pages of a resident buffer are wired, so MADV_FREE_REUSABLE on them frees nothing
// (measured: footprint unchanged). Pages go back when a smaller view replaces the one covering
// them (`imparo_metal_kv_prefetch` with a smaller size, then the retirement).
extern "C" void imparo_metal_kv_advise_free(uint32_t layer, uint32_t is_v,
                                            uint64_t off, uint64_t len) {
    (void)layer; (void)is_v; (void)off; (void)len;
}
extern "C" void imparo_metal_kv_advise_reuse(uint32_t layer, uint32_t is_v,
                                             uint64_t off, uint64_t len) {
    (void)layer; (void)is_v; (void)off; (void)len;
}

// The host address of `n` bytes of a layer's KV cache at `off` (the coordinates
// imparo_metal_write_kv takes), or NULL when the range is not inside the side's current
// view. The KV pool's restore reads a file straight into it; the storage is shared with
// the GPU, so that read is the whole transfer. Use it with the GPU idle, as the write.
extern "C" uint8_t * imparo_metal_kv_host_span(uint32_t layer, uint32_t is_v, uint64_t off,
                                               uint64_t n) {
    const std::vector<id<MTLBuffer>> & side = is_v ? g.kv_v : g.kv_k;
    if (layer >= side.size() || side[layer] == nil) { return nullptr; }
    id<MTLBuffer> b = side[layer];
    const uint64_t at = kv_reg(layer, is_v) + off;
    if (at < off || at + n < at || at + n > [b length]) { return nullptr; }
    return (uint8_t *)[b contents] + at;
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
// One greedy pick per row: the single-row kernel dispatched once per row at that row's
// offsets, so row r's index is exactly imparo_metal_argmax over row r. Each row declares its
// own hazard, so the rows run one after another, as that many single-row calls would.
extern "C" void imparo_metal_argmax_rows(uint32_t src, uint32_t dst, uint32_t width,
                                         uint32_t rows) {
    NSUInteger tw = std::min<NSUInteger>(1024, g.p_argmax.maxTotalThreadsPerThreadgroup);
    while (tw & (tw - 1)) { tw &= tw - 1; }   // the tree reduce needs a power of two
    for (uint32_t r = 0; r < rows; r++) {
        haz(hb(src), hb(dst));
        [g.enc setComputePipelineState:g.p_argmax];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] + (NSUInteger)r * width * 4
                 atIndex:0];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] + (NSUInteger)r * 4 atIndex:1];
        [g.enc setBytes:&width length:4 atIndex:2];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_ELEMENTWISE); }
        [g.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        if (g_prof) { prof_end(); }
    }
}
// THE K LARGEST ENTRIES OF EACH ROW: see imparo_top_k_chunks. A chunk thread scans `chunk`
// floats and a row's merge visits `chunks * k` list slots; chunks near sqrt(width / k) keeps
// the two about equal.
// The list capacity: imparo.metal holds it too, as the size of its lists.
static constexpr uint32_t TOP_K_ROWS_MAX = 64u;
// The slots of imparo_top_k8_chunks and imparo_top_k8_merge.
static constexpr uint32_t TOP_K_NARROW = 8u;
// The widest row the one-pass kernel takes. A thread scans the whole row, so this trades a
// longer scan against a dispatch and a barrier; 64 covers every router this engine serves
// (LFM2.5-8B-A1B has 32 experts) and leaves a wide vocabulary argmax on the two-pass form.
static constexpr uint32_t TOP_K_ONE_MAX = 64u;
// IMPARO_TOPK_ONE=0 is the A/B arm: the two-pass form on the same binary.
static bool top_k_one_on() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_TOPK_ONE"); v = (e && e[0] == '0') ? 0 : 1; }
    return v != 0;
}
struct TopKArgs { uint32_t width, rows, k, chunk, chunks; };
static TopKArgs top_k_args(uint32_t width, uint32_t rows, uint32_t k) {
    const uint64_t kk = std::max(k, 1u);
    uint32_t chunks = 1u;
    while ((uint64_t)chunks * chunks * kk < width) { chunks += 1u; }
    const uint32_t chunk = std::max(1u, (width + chunks - 1u) / chunks);
    chunks = std::max(1u, (width + chunk - 1u) / chunk);   // no empty chunk
    return TopKArgs{width, rows, k, chunk, chunks};
}
extern "C" uint32_t imparo_metal_top_k_rows_max(void) { return TOP_K_ROWS_MAX; }
extern "C" uint64_t imparo_metal_top_k_rows_len(uint32_t width, uint32_t rows, uint32_t k) {
    const TopKArgs a = top_k_args(width, rows, k);
    return 2ull * (uint64_t)rows * k * (1ull + a.chunks);
}
extern "C" int imparo_metal_top_k_rows(uint32_t src, uint32_t dst, uint32_t width,
                                       uint32_t rows, uint32_t k) {
    if (k == 0u || k > TOP_K_ROWS_MAX || k > width || rows == 0u || src == dst
        || src >= B_COUNT || dst >= B_COUNT || g.bufs[src] == nil || g.bufs[dst] == nil
        || (uint64_t)rows * width * 4ull > g.sizes[src]
        || imparo_metal_top_k_rows_len(width, rows, k) * 4ull > g.sizes[dst]) {
        NSLog(@"imparo metal: top_k_rows refused (src=%u dst=%u width=%u rows=%u k=%u)", src,
              dst, width, rows, k);
        return 1;
    }
    const TopKArgs a = top_k_args(width, rows, k);
    // Eight entries or fewer: the scalar-list kernels.
    const bool narrow = k <= TOP_K_NARROW;
    // ONE PASS when the row is narrow. The split into chunk lists plus a merge pays for
    // itself only when a row is wide enough that one thread scanning it is the bottleneck.
    // A 32-expert router picking 4 gives chunks=3: a three-thread dispatch and a one-thread
    // dispatch, both of them launch and nothing else, once per routed layer per token.
    if (narrow && width <= TOP_K_ONE_MAX && g.p_top_k8_one != nil && top_k_one_on()) {
        haz(hb(src), hb(dst));
        [g.enc setComputePipelineState:g.p_top_k8_one];
        [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
        [g.enc setBytes:&a length:sizeof(a) atIndex:3];
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_TOP_K_MERGE); }
        [g.enc dispatchThreads:MTLSizeMake(rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
              std::min<NSUInteger>(rows, g.p_top_k8_one.maxTotalThreadsPerThreadgroup), 1, 1)];
        if (g_prof) { prof_end(); }
        return 0;
    }
    id<MTLComputePipelineState> chunks_ps = narrow ? g.p_top_k8_chunks : g.p_top_k_chunks;
    id<MTLComputePipelineState> merge_ps = narrow ? g.p_top_k8_merge : g.p_top_k_merge;
    haz(hb(src), hb(dst));
    [g.enc setComputePipelineState:chunks_ps];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    [g.enc setBytes:&a length:sizeof(a) atIndex:3];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_TOP_K_CHUNKS); }
    [g.enc dispatchThreads:MTLSizeMake(a.chunks, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64, chunks_ps.maxTotalThreadsPerThreadgroup), 1, 1)];
    if (g_prof) { prof_end(); }
    // The merge reads the lists pass one wrote.
    haz(hb(dst), hb(dst));
    [g.enc setComputePipelineState:merge_ps];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:0];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&a length:sizeof(a) atIndex:2];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_TOP_K_MERGE); }
    [g.enc dispatchThreads:MTLSizeMake(rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(rows, merge_ps.maxTotalThreadsPerThreadgroup), 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}
// ============================================================================================
// A ROUTED FEED-FORWARD (imparo.metal, "MIXTURE OF EXPERTS"). Four entries, in the order a
// layer calls them: gate, plan, grouped (three times -- gate, up, down), combine.
//
// The model owns the buffers and the order; this file owns only the dispatches. Every entry
// checks its ranges and returns non-zero rather than writing outside them, because a routed
// layer's indices come from a kernel rather than from the host and a wrong one would read a
// neighbouring expert's weights and still produce plausible text.
static constexpr uint32_t MOE_MAX_EXPERTS = 256u;   // the shader's threadgroup arrays
static constexpr uint32_t MOE_MAX_K = 8u;
static constexpr uint32_t MOE_GROUPED_SGS = 8u;     // simdgroups a threadgroup, one row each
// The work rows the grouped matmul stages at once, and the bytes that costs. Must match
// MOE_STAGE_ROWS in the kernel: the kernel indexes `xstage + i * 1024` and the host is
// what says how much of it exists.
static constexpr uint32_t MOE_STAGE_ROWS = 4u;
static constexpr NSUInteger MOE_STAGE_BYTES = MOE_STAGE_ROWS * 1024u * sizeof(float);

// The activation staging: OFF unless IMPARO_MOE_STAGE=1. A measured negative -- the kernel
// comment says why, and the short version is that the cost is load COUNT, which staging
// does not change. Kept as an arm so the next attempt starts from a measurement.
static uint32_t moe_stage() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_MOE_STAGE"); v = e ? atoi(e) : 0; }
    return (uint32_t)(v != 0);
}

extern "C" int imparo_metal_moe_gate(uint32_t scores, uint32_t probs, uint32_t sel,
                                     uint64_t bias_off, uint32_t n_tok, uint32_t n_expert,
                                     uint32_t gating) {
    const uint64_t words = (uint64_t)n_tok * n_expert;
    if (n_tok == 0u || n_expert == 0u || n_expert > MOE_MAX_EXPERTS || gating > 1u
        || scores >= B_COUNT || probs >= B_COUNT || sel >= B_COUNT
        || g.bufs[scores] == nil || g.bufs[probs] == nil || g.bufs[sel] == nil
        || probs == sel
        || words * 4ull > g.sizes[scores] || words * 4ull > g.sizes[probs]
        || words * 4ull > g.sizes[sel]) {
        NSLog(@"imparo metal: moe_gate refused (n_tok=%u n_expert=%u gating=%u)", n_tok,
              n_expert, gating);
        return 1;
    }
    haz(hb(scores), hb(probs) | hb(sel));
    [g.enc setComputePipelineState:g.p_moe_gate];
    const uint64_t bias_l = (bias_off == W_NONE) ? W_NONE : wbind(g.enc, bias_off, 0);
    if (bias_off == W_NONE) { [g.enc setBuffer:g_wsegs.front().buf offset:0 atIndex:0]; }
    [g.enc setBuffer:g.bufs[scores] offset:g.buf_off[scores] atIndex:1];
    [g.enc setBuffer:g.bufs[probs] offset:g.buf_off[probs] atIndex:2];
    [g.enc setBuffer:g.bufs[sel] offset:g.buf_off[sel] atIndex:3];
    [g.enc setBytes:&n_expert length:4 atIndex:4];
    [g.enc setBytes:&gating length:4 atIndex:5];
    [g.enc setBytes:&bias_l length:8 atIndex:6];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_ROUTE); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

extern "C" int imparo_metal_moe_plan(uint32_t topk, uint32_t probs, uint32_t perm,
                                     uint32_t wgt, uint32_t seg, uint32_t inv, uint32_t n_tok,
                                     uint32_t n_expert, uint32_t k, uint32_t normalise,
                                     float scale) {
    const uint64_t rows = (uint64_t)n_tok * k;
    if (n_tok == 0u || k == 0u || k > MOE_MAX_K || k > n_expert || n_expert > MOE_MAX_EXPERTS
        || topk >= B_COUNT || probs >= B_COUNT || perm >= B_COUNT || wgt >= B_COUNT
        || seg >= B_COUNT || inv >= B_COUNT || g.bufs[topk] == nil || g.bufs[probs] == nil
        || g.bufs[perm] == nil || g.bufs[wgt] == nil || g.bufs[seg] == nil
        || g.bufs[inv] == nil
        || rows * 4ull > g.sizes[topk] || (uint64_t)n_tok * n_expert * 4ull > g.sizes[probs]
        || rows * 4ull > g.sizes[perm] || rows * 4ull > g.sizes[wgt]
        || rows * 4ull > g.sizes[inv]
        // seg holds n_expert+1 offsets, then the active count, then up to n_expert ids.
        || (2ull * (uint64_t)n_expert + 2ull) * 4ull > g.sizes[seg]) {
        NSLog(@"imparo metal: moe_plan refused (n_tok=%u n_expert=%u k=%u)", n_tok, n_expert, k);
        return 1;
    }
    haz(hb(topk) | hb(probs), hb(perm) | hb(wgt) | hb(seg) | hb(inv));
    [g.enc setComputePipelineState:g.p_moe_plan];
    [g.enc setBuffer:g.bufs[topk] offset:g.buf_off[topk] atIndex:0];
    [g.enc setBuffer:g.bufs[probs] offset:g.buf_off[probs] atIndex:1];
    [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:2];
    [g.enc setBuffer:g.bufs[wgt] offset:g.buf_off[wgt] atIndex:3];
    [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:4];
    [g.enc setBuffer:g.bufs[inv] offset:g.buf_off[inv] atIndex:5];
    [g.enc setBytes:&n_tok length:4 atIndex:6];
    [g.enc setBytes:&n_expert length:4 atIndex:7];
    [g.enc setBytes:&k length:4 atIndex:8];
    [g.enc setBytes:&normalise length:4 atIndex:9];
    [g.enc setBytes:&scale length:4 atIndex:10];
    // ONE THREADGROUP. The counting sort's positions come from threadgroup atomics and its
    // bases from one prefix sum over them, so a second threadgroup would need a device-wide
    // barrier this encoder does not have -- and the sort is tens of microseconds of work.
    const NSUInteger tw = std::min<NSUInteger>(256, g.p_moe_plan.maxTotalThreadsPerThreadgroup);
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_ROUTE); }
    [g.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

// THE GATE, THE PICK AND THE PLAN AS ONE DISPATCH. Refused -- with nothing encoded, so the
// caller runs the three -- whenever a precondition does not hold; the three compute the same
// thing, which is why every refusal here is silent.
//
// The threshold is in TOKENS and it is what keeps this honest: the plan is one threadgroup
// at every width, but the gate is n_tok threadgroups, and folding it in puts 512 softmaxes
// on one core at a full prefill chunk. Below the threshold the whole route is a few hundred
// operations and the two saved dispatches are the entire cost of it.
static uint32_t moe_route_fuse_max() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_MOE_ROUTE_FUSE_MAX"); v = e ? atoi(e) : 32; }
    return (uint32_t)(v < 0 ? 0 : v);
}

extern "C" int imparo_metal_moe_route(uint32_t scores, uint32_t probs, uint32_t sel,
                                      uint32_t topk, uint32_t perm, uint32_t wgt,
                                      uint32_t seg, uint32_t inv, uint64_t bias_off,
                                      uint32_t n_tok, uint32_t n_expert, uint32_t k,
                                      uint32_t gating, uint32_t normalise, float scale) {
    const uint64_t rows = (uint64_t)n_tok * k;
    const uint64_t words = (uint64_t)n_tok * n_expert;
    // THE PICK IS THE ONE-PASS TOP-8 LIST, so this serves only what that serves: eight
    // entries or fewer over a row narrow enough for one thread to scan. Wider rows keep the
    // chunked top-k, which is two dispatches of its own and not foldable into one group.
    if (g.p_moe_route == nil || n_tok == 0u || n_tok > moe_route_fuse_max()
        || k == 0u || k > TOP_K_NARROW || k > MOE_MAX_K || k > n_expert
        || n_expert == 0u || n_expert > MOE_MAX_EXPERTS || n_expert > TOP_K_ONE_MAX
        || !top_k_one_on() || gating > 1u
        || scores >= B_COUNT || probs >= B_COUNT || sel >= B_COUNT || topk >= B_COUNT
        || perm >= B_COUNT || wgt >= B_COUNT || seg >= B_COUNT || inv >= B_COUNT
        || g.bufs[scores] == nil || g.bufs[probs] == nil || g.bufs[sel] == nil
        || g.bufs[topk] == nil || g.bufs[perm] == nil || g.bufs[wgt] == nil
        || g.bufs[seg] == nil || g.bufs[inv] == nil || probs == sel
        || words * 4ull > g.sizes[scores] || words * 4ull > g.sizes[probs]
        || words * 4ull > g.sizes[sel]
        || imparo_metal_top_k_rows_len(n_expert, n_tok, k) * 4ull > g.sizes[topk]
        || rows * 4ull > g.sizes[perm] || rows * 4ull > g.sizes[wgt]
        || rows * 4ull > g.sizes[inv]
        || (2ull * (uint64_t)n_expert + 2ull) * 4ull > g.sizes[seg]) {
        return 1;
    }
    haz(hb(scores), hb(probs) | hb(sel) | hb(topk) | hb(perm) | hb(wgt) | hb(seg) | hb(inv));
    [g.enc setComputePipelineState:g.p_moe_route];
    const uint64_t bias_l = (bias_off == W_NONE) ? W_NONE : wbind(g.enc, bias_off, 0);
    if (bias_off == W_NONE) { [g.enc setBuffer:g_wsegs.front().buf offset:0 atIndex:0]; }
    [g.enc setBuffer:g.bufs[scores] offset:g.buf_off[scores] atIndex:1];
    [g.enc setBuffer:g.bufs[probs] offset:g.buf_off[probs] atIndex:2];
    [g.enc setBuffer:g.bufs[sel] offset:g.buf_off[sel] atIndex:3];
    // The top-k destination carries ids then values in ONE buffer, bound twice, exactly as
    // imparo_metal_top_k_rows binds it.
    [g.enc setBuffer:g.bufs[topk] offset:g.buf_off[topk] atIndex:4];
    [g.enc setBuffer:g.bufs[topk] offset:g.buf_off[topk] atIndex:5];
    [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:6];
    [g.enc setBuffer:g.bufs[wgt] offset:g.buf_off[wgt] atIndex:7];
    [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:8];
    [g.enc setBuffer:g.bufs[inv] offset:g.buf_off[inv] atIndex:9];
    [g.enc setBytes:&n_tok length:4 atIndex:10];
    [g.enc setBytes:&n_expert length:4 atIndex:11];
    [g.enc setBytes:&k length:4 atIndex:12];
    [g.enc setBytes:&gating length:4 atIndex:13];
    [g.enc setBytes:&bias_l length:8 atIndex:14];
    [g.enc setBytes:&normalise length:4 atIndex:15];
    [g.enc setBytes:&scale length:4 atIndex:16];
    const NSUInteger tw = std::min<NSUInteger>(256, g.p_moe_route.maxTotalThreadsPerThreadgroup);
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_ROUTE); }
    [g.enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

// The routed matmul's pipeline, built on first use: the weight format alone, since it always
// addresses row-major blocks.
static id<MTLComputePipelineState> moe_grouped_pipeline_for_fmt(uint32_t wfmt,
                                                                bool rowmajor) {
    if (wfmt == 0u || wfmt >= 32u || g.lib == nil) { return nil; }
    // One slot per (format, layout): the kernel now addresses both, so a stack that was
    // repacked must not be served by a pipeline compiled for the row-major addresses.
    id<MTLComputePipelineState> __strong & have =
        g.p_moe_grouped_fmt[wfmt + (rowmajor ? 0u : 32u)];
    if (have != nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    // DIAGNOSTIC (IMPARO_MOE_SKIP, the kernel's skip bits); unset in every served run.
    if (moe_skip_bits() > 0) {
        const uint32_t sk = moe_skip_bits();
        [cv setConstantValue:&sk type:MTLDataTypeUInt atIndex:45];
    }
    const uint32_t stage = moe_stage();
    [cv setConstantValue:&stage type:MTLDataTypeUInt atIndex:46];
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_moe_grouped" constantValues:cv
                                             error:&e];
    have = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (have == nil) {
        NSLog(@"imparo metal: routed matmul for weight format %u (%s): %@", wfmt,
              rowmajor ? "row-major" : "tile-major", e);
    } else if (getenv("IMPARO_WFMT_LOG")) {
        NSLog(@"imparo metal: built the routed matmul for weight format %u", wfmt);
    }
    return have;
}

// The routed matmul on the matrix unit. Same constants as the rows matmul it shares a body
// with; no half arm, because the routed path holds float activations like every other
// non-fast route.
static id<MTLComputePipelineState> moe_grouped_mma_pipeline_for_fmt(uint32_t wfmt,
                                                                   uint32_t tiles,
                                                                   bool rowmajor,
                                                                   bool pair = false) {
    if (wfmt == 0u || wfmt >= 32u || g.lib == nil) { return nil; }
    if (tiles != 1u && tiles != 2u && tiles != 4u) { return nil; }
    id<MTLComputePipelineState> __strong & have =
        pair ? g.p_moe_grouped_mma_pair_fmt[wfmt + (rowmajor ? 0u : 32u)]
                                           [tiles == 4u ? 2u : tiles - 1u]
             : g.p_moe_grouped_mma_fmt[wfmt + (rowmajor ? 0u : 32u)]
                                      [tiles == 4u ? 2u : tiles - 1u];
    if (have != nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    stamp_epi_act(cv);
    [cv setConstantValue:&tiles type:MTLDataTypeUInt atIndex:43];
    // The shared body's skip bits reach the routed kernel too (IMPARO_BLK_MMA_SKIP): the
    // routed matmul is priced by the same three differences as the rows matmul it is.
    static int mskip = -1;
    if (mskip < 0) { const char * se = getenv("IMPARO_BLK_MMA_SKIP"); mskip = se ? atoi(se) : 0; }
    if (mskip > 0) {
        const uint32_t sk = (uint32_t)mskip;
        [cv setConstantValue:&sk type:MTLDataTypeUInt atIndex:44];
    }
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:pair ? @"imparo_moe_grouped_mma_pair"
                                                        : @"imparo_moe_grouped_mma"
                                    constantValues:cv error:&e];
    have = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (have == nil) {
        NSLog(@"imparo metal: routed matrix-unit matmul%s for weight format %u: %@",
              pair ? " (pair)" : "", wfmt, e);
    } else if (getenv("IMPARO_WFMT_LOG")) {
        NSLog(@"imparo metal: built the routed matrix-unit matmul%s for weight format %u "
              @"(tiles=%u, max_threads=%lu)", pair ? " (pair)" : "", wfmt, tiles,
              (unsigned long)[have maxTotalThreadsPerThreadgroup]);
    }
    return have;
}

// THE STAGED ROUTED GEMM (IMPARO_MOE_ST=1). Both operands staged as half in threadgroup
// memory and multiplied on the matrix unit -- the design st_gemm uses for Q8_0, which a
// k-quant has never had. Measured on Q8_0, where IMPARO_Q8_DESIGN switches the two designs
// on one shape: staged beats register-tiled 1.63x at 64 rows, 1.25x at 128, 1.12x at 512.
// A routed layer is nothing but thin shapes, so this is where that 1.63x lives.
static constexpr uint32_t MOE_ST_ROWS = 64u, MOE_ST_TOKENS = 32u, MOE_ST_K = 64u;
// ONE buffer the kernel partitions (imparo.metal, moe_st_gemm_body), so no binding is trusted
// to be distinct from another: the weight tile (K / 8 * ROWS / 8 tiles of 64 halves), the
// activation tile (K / 8 * TOKENS / 8 of them), then the resolved row bases. The paired and
// two-block shapes restage one weight tile, so they need no more than the single shape.
static constexpr NSUInteger moe_st_tg_bytes(uint32_t rows, uint32_t tokens, uint32_t k) {
    const NSUInteger stage =
        (NSUInteger)(k / 8u) * ((rows / 8u) + (tokens / 8u)) * 64u * sizeof(uint16_t);
    return stage + (NSUInteger)tokens * sizeof(uint64_t);
}

// A WORK-ROW GROUP THAT COVERS A WHOLE EXPERT stages the weight tile ONCE. The grid is
// (row groups, token groups, experts) and every token group of an expert stages the SAME
// weight tile, so at a 512-token chunk -- 2048 work rows over 32 experts, 64 each -- a
// 32-token group dequantises every weight twice. 64 covers the segment in one.
// THE ROUTED GEMM'S SHAPES: one kernel each (imparo.metal, IMPARO_MOE_ST_KERNEL). `tokens` is
// the token tile (the grid's token axis and the n_tok floor), `nsg` the simdgroups a
// threadgroup, `paired` whether it is the gate|up form.
// `rblk` row blocks of `rows` a threadgroup (imparo.metal, ROW2): the grid's row axis covers
// rows * rblk, the threadgroup buffer holds one block.
struct MoeStShape { NSString * fn; uint32_t rows, tokens, k, nsg; bool paired; uint32_t rblk = 1u; };
static MoeStShape moe_st_shape(uint32_t i) {
    switch (i) {
    case 0: return {@"imparo_moe_st_gemm_pair",    64u, 32u, 64u, 4u, true};
    case 1: return {@"imparo_moe_st_gemm_n8",      64u, 8u, 64u, 2u, false};
    case 2: return {@"imparo_moe_st_gemm_pair_n8", 64u, 8u, 64u, 2u, true};
    case 3: return {@"imparo_moe_st_gemm_r2",      64u, 32u, 64u, 4u, false, 2u};
    default: return {nil, 0u, 0u, 0u, 0u, false};
    }
}
// The single-projection shape (the down projection): two 64-row blocks a threadgroup (3), which
// stages each activation tile once for 128 rows as the pair does for gate and up -- 868 -> 839
// ms of down projection at a 5955-token prefill against one 64-row block, bit-identical; a
// 64-token tile read 937. The 8-token shape serves a chunk under 32 tokens (16 tokens: 42.9 ms
// a forward against 72.3 on the 32-token shape; the rows kernel's 34.7 is not the same
// arithmetic).
static uint32_t moe_st_single_shape(uint32_t n_tok = 0xffffffffu) { return n_tok < 32u ? 1u : 3u; }
// The gate|up shape: the paired kernel, its 8-token form under 32 tokens.
static uint32_t moe_st_pair_shape(uint32_t n_tok = 0xffffffffu) { return n_tok < 32u ? 2u : 0u; }
// THE HALF HANDOFF (imparo.metal, MOE_ST_SRC_HALF / MOE_ST_DST_HALF): the staged GEMM reads a
// current half mirror of its source instead of the floats, and the gate|up pair writes its
// result as half when the down projection will stage it. Bit-identical: both round to half at
// the same point the tile staging did.
// Where a routed matmul's activation rows live. A current half mirror of `src` covering
// `elems` is read in place of the floats (`*half` true, the mirror's buffer returned); floats
// a mirror-mode producer left stale with no such mirror mean nothing holds the activations,
// and the caller refuses (returns B_COUNT).
static uint32_t moe_st_source(uint32_t src, uint64_t elems, bool * half) {
    const bool stale = src < 64u && ((g_float_stale >> src) & 1ull) != 0ull;
    const bool mirrored = g_xh_src == src && g_xh_elems >= elems
                       && g_xh_buf < B_COUNT && g.bufs[g_xh_buf] != nil;
    *half = mirrored;
    if (mirrored) { return g_xh_buf; }
    return stale ? (uint32_t)B_COUNT : src;
}
static uint32_t moe_st_tokens() {
    const uint32_t a = moe_st_shape(moe_st_single_shape()).tokens;
    const uint32_t b = moe_st_shape(moe_st_pair_shape()).tokens;
    return a > b ? a : b;
}

// ON. Prefill 926.6 -> 1348.4 tok/s against llama's 1374.3 on the long leg and 1279.8
// against 1247.8 on the short one (bracket, LFM2.5-8B-A1B Q4_K_M, M3 Pro, 2026-09-21).
// IMPARO_MOE_ST=0 is the off arm.
//
// DECODE IS UNTOUCHED: a decode step (1 row) and co-batched decode rows keep the rows kernel
// (moe_st_wanted). Every multi-token chunk of one sequence -- a prefill chunk or a speculative
// verify batch -- stages both operands as half, as the dense GEMMs do from two tokens; on this
// model that lands CLOSER to llama than the rows kernel does (max |delta-to-top1| 0.698 vs
// 0.793).
static uint32_t moe_st_on() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_MOE_ST"); v = (e && e[0] == '0') ? 0 : 1; }
    return (uint32_t)v;
}
// WHICH FORWARDS TAKE THE STAGED GEMM.
//
//   a prompt chunk of one sequence, 2+ tokens   always (narrow shape under 32 tokens)
//   a verify batch, 32+ tokens                  yes
//   a verify batch under 32, a decode step,
//   co-batched decode rows                      no: the rows kernel, in float
//
// A PROMPT CHUNK'S POSITIONS MUST COMPUTE THE SAME AT EVERY WIDTH: a resumed prompt is chunked
// differently from a cold one and the KV pool reuses one for the other byte for byte. With the
// floor at the tile (32) a 9-token resumed tail ran the rows kernel where the cold prefill had
// staged half, and kv_gates' split and grid phases stopped being byte-equal. The narrow shape
// gives the same per-element arithmetic, so the width no longer matters.
//
// A VERIFY BATCH KEEPS THE ROWS KERNEL under 32: speculation on LFM2.5-8B-A1B (agentic --tiny,
// second turn, 2026-09-24) ran 75.7 / 76.0 tok/s that way and 47.5 / 47.4 with the staged GEMM
// (the rows kernel streams a few rows an expert far better). Verify positions are decode-time
// work, like a decode step, which is float everywhere.
static bool moe_st_wanted(uint32_t n_tok, bool prefill_chunk) {
    if (!moe_st_on() || decode_rows_route() != 0u) { return false; }
    return n_tok >= 32u || (prefill_chunk && n_tok >= HALF_A_MIN);
}
// The routed GEMM's token tile, or 0 when the staged GEMM is off: the prefill chunk knob
// derives from how many whole tiles a chunk gives each expert.
extern "C" uint32_t imparo_metal_moe_token_tile(void) { return moe_st_on() ? moe_st_tokens() : 0u; }

static id<MTLComputePipelineState> moe_st_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                           uint32_t shape,
                                                           bool src_half = false,
                                                           bool dst_half = false) {
    if (wfmt == 0u || wfmt >= 32u || g.lib == nil || moe_st_shape(shape).fn == nil) { return nil; }
    const uint32_t io = (src_half ? 1u : 0u) | (dst_half ? 2u : 0u);
    id<MTLComputePipelineState> __strong & have =
        g.p_moe_st_gemm_fmt[wfmt + (rowmajor ? 0u : 32u)][shape][io];
    if (have != nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    // THE MODEL'S ACTIVATION. imparo_act_f reads function constant 11 and DEFAULTS TO GELU
    // when it is not set, so a paired pipeline built without it silently computes gelu
    // where the model wants silu -- a ~1% difference in the SwiGLU, which is what this
    // kernel's first build produced. The single-projection shapes never noticed because
    // they do not call imparo_act_f at all.
    [cv setConstantValue:&g_epi_act type:MTLDataTypeUInt atIndex:11];
    [cv setConstantValue:&src_half type:MTLDataTypeBool atIndex:61];
    [cv setConstantValue:&dst_half type:MTLDataTypeBool atIndex:62];
    NSError * e = nil;
    id<MTLFunction> f =
        [g.lib newFunctionWithName:moe_st_shape(shape).fn constantValues:cv error:&e];
    have = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (have == nil) {
        NSLog(@"imparo metal: staged routed GEMM for weight format %u: %@", wfmt, e);
    } else if (getenv("IMPARO_WFMT_LOG")) {
        NSLog(@"imparo metal: built the staged routed GEMM for weight format %u (%s, shape %u, "
              @"%s source, %s result, max_threads=%lu)", wfmt,
              rowmajor ? "row-major" : "tile-major", shape, src_half ? "half" : "float",
              dst_half ? "half" : "float", (unsigned long)[have maxTotalThreadsPerThreadgroup]);
    }
    return have;
}

// THERE IS NO CROSSING: the matrix-unit routed matmul wins at every width, so it runs at
// every width. Measured on LFM2.5-8B-A1B (M3 Pro, 2026-09-21) by setting the prefill chunk
// to the width with IMPARO_BATCH and taking the median chunk of 512/width, first dropped:
//
//     width      1      2      4      8     16     32     64
//     scalar  10.1   17.6   28.4   52.9  110.8  241.4  514.5  ms
//     mma     10.0   15.6   18.5   21.8   31.1   46.8   73.1
//     ratio  0.990  0.886  0.651  0.412  0.281  0.194  0.142
//
// The scalar kernel is linear in width, the matrix unit sub-linear -- 29x against 4.7x over
// the same 32x in rows. Width 1 is a tie, and the decode step agrees (12.678 vs 12.486 ms,
// inside this harness's noise), so nothing is given up at the bottom.
//
// I seated this at 16 first, by analogy with llama.cpp crossing its own two routed kernels
// at 32 tokens (ggml-metal-common.cpp:19, `ne21 >= 32`). That was wrong in the direction
// that mattered most: a speculative verify batch is about nine rows, so every verify was
// taking the kernel that loses 2.4x at that width.
//
// IMPARO_MOE_MMA_MIN is kept as the A/B arm -- a large value forces the scalar kernel back.
// THE ROUTED MATMUL'S OWN SHAPE, separate from blk_rows_mma's.
//
// The two kernels share a body and nothing else about their shape. blk_rows_mma runs N rows
// of ONE tensor; the routed matmul at decode runs ONE work row in each of k experts, so its
// grid is k times shorter and its threadgroups want to be smaller and more numerous. Sharing
// blk_mma_tiles/blk_mma_sgs made the routed path inherit a seat chosen for the other shape,
// and nobody had searched this one -- neither knob is declared to the tuner.
//
// Searched on LFM2.5-8B-A1B at decode, every arm warmed, 128 steps a point:
//
//             sgs=1     sgs=2     sgs=4     sgs=8
//   tiles=1  11.190    10.999    11.165    11.385
//   tiles=2  11.460    11.218    11.123    11.246   <- what it inherited
//   tiles=4     --     13.199    13.018    13.443
//
// ABAB: inherited 11.283 / 11.231 against (1,2)'s 10.985 / 10.979 -- -2.44%. (1,2) is a real
// optimum and not the end of a trend: sgs=1 is worse than sgs=2.
//
// SEPARATE SEATS ARE THE POINT. Moving the shared ones would move E4B, LFM2 and the 27B
// co-batch, whose bits and gates have nothing to do with this shape. These move only the
// routed matmul, so only a routed model's gate is at stake.
static uint32_t moe_mma_tiles() {
    static uint32_t t = 0u;
    if (t == 0u) {
        const char * e = getenv("IMPARO_MOE_MMA_TILES");
        const int v = e != nullptr ? atoi(e) : 1;
        t = (v == 1 || v == 2 || v == 4) ? (uint32_t)v : 1u;
    }
    return t;
}
static uint32_t moe_mma_sgs(uint32_t n_in) {
    static uint32_t seat = 0u;
    if (seat == 0u) {
        const char * e = getenv("IMPARO_MOE_MMA_SGS");
        const int v = e != nullptr ? atoi(e) : 0;
        seat = (v >= 1 && v <= 32) ? (uint32_t)v : 2u;
    }
    return std::max(1u, std::min(seat, n_in / 32u));
}

// THE PAIR'S K-SPLIT, WHICH FOLLOWS THE SINGLE PROJECTION'S BY DEFAULT. Matching it is what
// makes the pair BIT-IDENTICAL to the three dispatches it replaces: the split decides the
// order the partial sums are added, so a pair on a different split computes the same
// arithmetic in a different order and every logit moves. Measured both ways -- matched, a
// 16-token chunk agrees to the last bit; at 4 against the single's 2 it does not.
//
// The override exists because the pair carries two weight stacks in one threadgroup and may
// well want its own shape. It is a SEPARATE seat so that searching it cannot move the down
// projection too (that confounded the first routed search), and it is BIT-AFFECTING, so
// seating it away from the single's needs the logit gate first.
static uint32_t moe_pair_sgs(uint32_t n_in) {
    static int seat = -1;
    if (seat < 0) {
        const char * e = getenv("IMPARO_MOE_PAIR_SGS");
        const int v = e != nullptr ? atoi(e) : 0;
        seat = (v >= 1 && v <= 32) ? v : 0;
    }
    if (seat == 0) { return moe_mma_sgs(n_in); }
    return std::max(1u, std::min((uint32_t)seat, n_in / 32u));
}

// THE PAIR'S ADMISSION, in tokens. Below it the pair's halved grid costs more than its
// saved dispatch and activation loads; above it the win grows to -9.6% by 24 tokens. The
// measurement is in imparo_metal_moe_grouped_pair, where the route is chosen.
static uint32_t moe_pair_min_tok() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_MOE_PAIR_MIN"); v = e ? atoi(e) : 8; }
    return (uint32_t)(v < 1 ? 1 : v);
}

static uint32_t moe_mma_min_tok() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("IMPARO_MOE_MMA_MIN"); v = e ? atoi(e) : 1; }
    return (uint32_t)(v < 1 ? 1 : v);
}

extern "C" int imparo_metal_moe_grouped(uint32_t wkind, uint64_t w_off, uint64_t expert_stride,
                                        uint32_t src, uint32_t dst, uint32_t perm, uint32_t seg,
                                        uint32_t n_in, uint32_t n_out, uint32_t n_expert,
                                        uint32_t n_tok, uint32_t rows,
                                        uint32_t src_work_rows, uint32_t prefill_chunk) {
    // ROW-MAJOR ONLY, and said here rather than assumed: `wfmt_for` maps a tile-major kind
    // to its SOURCE format, so a tile-major stack would build a pipeline that reads the
    // right format at the wrong addresses. No stack can be tile-major today -- the
    // transform refuses a 3-D tensor by name (`tm_applies`, "not 2-D") -- and this is what
    // makes that a checked fact rather than a remembered one.
    // A STACK MAY NOW BE TILE-MAJOR. tm_applies used to refuse a 3-D tensor, so 91% of an
    // MoE file's bytes stayed row-major and the matrix unit read eight streams a unit
    // instead of one. It converts slice by slice, which for a stack whose slice rows divide
    // the unit is the same as converting it whole. `wfmt_for` maps the tile-major kind back
    // to its SOURCE format -- the decode is the same, only the addresses differ -- and the
    // layout rides on constant 22.
    const bool tile_major =
        g_wire_ggml_set && wkind < 64u && g_wire_ggml[wkind] >= 1000u;
    const uint32_t wfmt = moe_wfmt_for(wkind);
    id<MTLComputePipelineState> ps = moe_grouped_pipeline_for_fmt(wfmt, !tile_major);
    const uint64_t span = (uint64_t)n_expert * expert_stride;
    if (ps == nil || rows == 0u || n_expert == 0u || n_expert > MOE_MAX_EXPERTS || n_in == 0u
        || n_out == 0u || (n_in % 32u) != 0u
        || src >= B_COUNT || dst >= B_COUNT || perm >= B_COUNT || seg >= B_COUNT
        || g.bufs[src] == nil || g.bufs[dst] == nil || g.bufs[perm] == nil
        || g.bufs[seg] == nil || src == dst
        // THE SOURCE HOLDS ONE ROW PER TOKEN unless it is the down projection's input,
        // which the gate and up projections wrote one row per WORK ROW. Checking a
        // token-indexed source against the work rows refused every routed layer of a real
        // model, where there are k times as many.
        || (uint64_t)(src_work_rows ? rows : n_tok) * n_in * 4ull > g.sizes[src]
        || (uint64_t)rows * n_out * 4ull > g.sizes[dst]
        || (uint64_t)rows * 4ull > g.sizes[perm]
        // seg holds n_expert+1 offsets, then the active count, then up to n_expert ids.
        || (2ull * (uint64_t)n_expert + 2ull) * 4ull > g.sizes[seg]
        // EVERY EXPERT THROUGH ONE BOUND BUFFER: the kernel offsets from the stack's base,
        // so the whole stack has to be in the segment that base falls in.
        || !w_pair_in_one_segment(w_off, w_off + span - 1ull)) {
        NSLog(@"imparo metal: moe_grouped refused (kind=%u %u->%u experts=%u tokens=%u "
              @"rows=%u%s)", wkind, n_in, n_out, n_expert, n_tok, rows,
              ps == nil ? ", no pipeline" : "");
        return 1;
    }
    // DIAGNOSTIC: IMPARO_SKIP_CAT=moe_expert drops the routed matmuls so a skip-and-diff
    // can price them in situ, the way matmat_decode is priced. The category existed in the
    // profiler's table with no dispatch honouring it, which is the failure the skip-cat
    // parser warns about for "attention": asking to skip it silently skipped nothing and
    // would have priced the routed matmul at ZERO.
    if (g_skip_cat == PC_MOE_EXPERT) { return 0; }
    // THE STAGED GEMM when it is on and the batch is worth a 64x32 tile. At one token the
    // tile is 1/32 live, so decode keeps the rows kernel.
    // HOW MANY EXPERT GROUPS TO LAUNCH. A token picks k DISTINCT experts, so the work
    // rows can reach at most `rows` experts, and never more than n_expert. The kernel
    // maps group -> the g-th ACTIVE expert through moe_plan's compacted tail, so this
    // bound needs no read-back. Decode: 4 work rows over 32 experts, 4 groups not 32.
    const NSUInteger zg = (NSUInteger)((rows < n_expert) ? rows : n_expert);
    const MoeStShape st_sh = moe_st_shape(moe_st_single_shape(n_tok));
    const uint32_t st_tok = st_sh.tokens;
    // The activations: a current half mirror when one exists (the pair's half result), the
    // floats otherwise. Only the staged GEMM reads a mirror, so floats left stale by a
    // mirror-mode producer are refused on every other route -- nothing else holds them.
    bool st_src_half = false;
    const uint32_t st_src =
        moe_st_source(src, (uint64_t)(src_work_rows ? rows : n_tok) * n_in, &st_src_half);
    id<MTLComputePipelineState> st = (moe_st_wanted(n_tok, prefill_chunk != 0u) && st_src < B_COUNT)
        ? moe_st_pipeline_for_fmt(wfmt, !tile_major, moe_st_single_shape(n_tok), st_src_half)
        : nil;
    const bool src_stale = src < 64u && ((g_float_stale >> src) & 1ull) != 0ull;
    if (st == nil && src_stale) {
        NSLog(@"imparo metal: moe_grouped source %u holds a half result only this route's "
              @"staged GEMM reads, and it cannot run (tokens=%u)", src, n_tok);
        return 1;
    }
    haz(hb(st != nil ? st_src : src) | hb(perm) | hb(seg), hb(dst));
    if (st != nil) {
        {
            [g.enc setComputePipelineState:st];
            const uint64_t wls = wbind(g.enc, w_off, 0);
            [g.enc setBuffer:g.bufs[st_src] offset:g.buf_off[st_src] atIndex:1];
            [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
            [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:3];
            [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:4];
            [g.enc setBytes:&wls length:8 atIndex:5];
            [g.enc setBytes:&expert_stride length:8 atIndex:6];
            [g.enc setBytes:&n_in length:4 atIndex:7];
            [g.enc setBytes:&n_out length:4 atIndex:8];
            [g.enc setBytes:&src_work_rows length:4 atIndex:9];
            [g.enc setBytes:&n_expert length:4 atIndex:14];
            const uint64_t no_pair_off = 0ull;      // unused by the single-projection form
            [g.enc setBytes:&no_pair_off length:8 atIndex:15];
            (void)wbind(g.enc, w_off, 16);           // likewise; bound so no argument dangles
            // The staged tiles and the resolved row bases (moe_st_tg_bytes); the result
            // leaves from the fragments, so there is no spill to size.
            [g.enc setThreadgroupMemoryLength:moe_st_tg_bytes(st_sh.rows, st_tok, st_sh.k)
                                      atIndex:0];
            // A token never picks one expert twice, so an expert holds at most n_tok work
            // rows: this covers ANY routing without reading one, and a group past its
            // expert's segment returns on seg.
            const NSUInteger rg = (n_out + st_sh.rows * st_sh.rblk - 1u) / (st_sh.rows * st_sh.rblk);
            const NSUInteger tg = (n_tok + st_tok - 1u) / st_tok;
            g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_EXPERT, (uint64_t)zg * expert_stride); }
            // The shape's own simdgroup count: NSG * RT_SG * TT_SG covers its output tile.
            [g.enc dispatchThreadgroups:MTLSizeMake(rg, tg, zg)
                  threadsPerThreadgroup:MTLSizeMake(32u * st_sh.nsg, 1, 1)];
            if (g_prof) { prof_end(); }
            return 0;
        }
    }
    // THE MATRIX UNIT above the crossing. Same buffers in the same slots -- the routed mma
    // kernel takes the scalar one's arguments plus an epilogue it is always handed 0 for,
    // because a routed layer's activation runs as its own dispatch (act_mul) between the up
    // projection and the down one.
    if (n_tok >= moe_mma_min_tok()) {
        // THE PROBE DOES NOT REACH THIS KERNEL, and saying so is the whole point. The skip
        // bits are a function constant on imparo_moe_grouped, the SCALAR kernel; the matrix
        // unit serves every width from moe_mma_min_tok() up, which defaults to 1 -- so at
        // DECODE the scalar kernel does not run and IMPARO_MOE_SKIP changed nothing while
        // reporting a clean measurement. It cost this chapter two built-and-measured "ties"
        // that were edits to a kernel the decode step never dispatches. A probe that is
        // silently a no-op is worse than no probe: it answers.
        static bool warned = false;
        if (moe_skip_bits() != 0u && !warned) {
            warned = true;
            NSLog(@"imparo metal: IMPARO_MOE_SKIP=%u IGNORED -- the matrix-unit routed kernel "
                  @"serves n_tok >= %u and carries no skip bits. Set IMPARO_MOE_MMA_MIN above "
                  @"n_tok to reach the scalar kernel the bits belong to.",
                  moe_skip_bits(), moe_mma_min_tok());
        }
        const uint32_t tiles = moe_mma_tiles();
        id<MTLComputePipelineState> mm =
            moe_grouped_mma_pipeline_for_fmt(wfmt, tiles, !tile_major);
        if (mm != nil) {
            const uint32_t sgs = moe_mma_sgs(n_in);
            [g.enc setComputePipelineState:mm];
            const uint64_t wlm = wbind(g.enc, w_off, 0);
            [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
            [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
            [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:3];
            [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:4];
            [g.enc setBytes:&wlm length:8 atIndex:5];
            [g.enc setBytes:&expert_stride length:8 atIndex:6];
            [g.enc setBytes:&n_in length:4 atIndex:7];
            [g.enc setBytes:&n_out length:4 atIndex:8];
            [g.enc setBytes:&src_work_rows length:4 atIndex:9];
            const uint32_t no_epi = 0u;
            [g.enc setBytes:&no_epi length:4 atIndex:13];
            [g.enc setBytes:&n_expert length:4 atIndex:14];
            [g.enc setThreadgroupMemoryLength:(NSUInteger)sgs * tiles * 64u * sizeof(float)
                                      atIndex:0];
            const uint32_t rows_per_tg = 8u * tiles;
            g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_EXPERT, (uint64_t)zg * expert_stride); }
            [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1u) / rows_per_tg,
                                                    zg, 1)
                  threadsPerThreadgroup:MTLSizeMake(32u * sgs, 1, 1)];
            if (g_prof) { prof_end(); }
            return 0;
        }
    }
    // THE SCALAR KERNEL NOW READS EITHER LAYOUT. It used to refuse a tile-major stack --
    // which, since the load-time repack converts 91% of an MoE file, meant it could not be
    // reached at all on a served model, and forcing it produced a run with the routed
    // matmuls silently MISSING rather than a slow one.
    [g.enc setComputePipelineState:ps];
    const uint64_t wl = wbind(g.enc, w_off, 0);
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:3];
    [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:4];
    [g.enc setBytes:&wl length:8 atIndex:5];
    [g.enc setBytes:&expert_stride length:8 atIndex:6];
    [g.enc setBytes:&n_in length:4 atIndex:7];
    [g.enc setBytes:&n_out length:4 atIndex:8];
    [g.enc setBytes:&src_work_rows length:4 atIndex:9];
    [g.enc setBytes:&n_expert length:4 atIndex:14];
    // The staged activation slice. 16 bytes when the staging is off: the argument is
    // declared either way, and a length the kernel never reads must not cost occupancy.
    [g.enc setThreadgroupMemoryLength:moe_stage() ? MOE_STAGE_BYTES : 16u atIndex:0];
    // The grid covers every (output row, ACTIVE expert) pair -- see the bound above.
    const NSUInteger tgx = (n_out + MOE_GROUPED_SGS - 1u) / MOE_GROUPED_SGS;
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_EXPERT, (uint64_t)zg * expert_stride); }
    [g.enc dispatchThreadgroups:MTLSizeMake(tgx, zg, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * MOE_GROUPED_SGS, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

// GATE, UP AND THE ACTIVATION IN ONE ROUTED DISPATCH. Refused -- with nothing encoded, so
// the caller runs the three -- whenever the pair's preconditions do not hold; the three
// compute the same thing, which is why every refusal here is silent.
//
// ONE BOUND BUFFER FOR BOTH WEIGHTS, so the two stacks have to live in one segment. A tiered
// placement can split a layer, and then they do not.
extern "C" int imparo_metal_moe_grouped_pair(uint32_t wkind, uint64_t gate_off, uint64_t up_off,
                                             uint64_t expert_stride, uint32_t src, uint32_t dst,
                                             uint32_t perm, uint32_t seg, uint32_t n_in,
                                             uint32_t n_out, uint32_t n_expert, uint32_t n_tok,
                                             uint32_t rows, uint32_t prefill_chunk) {
    // ABOVE A TOKEN THRESHOLD, AND THE THRESHOLD IS THE WHOLE FINDING. Measured on
    // LFM2.5-8B-A1B (M3 Pro, gpu_per_cb over one prefill chunk, both arms warmed), with the
    // weight layout UNCHANGED across the sweep:
    //
    //     n_tok        2        4        8       16       24
    //     delta    +1.01%   -0.04%   -2.10%   -8.25%   -9.63%
    //
    // So the 63.3 MiB between `ffn_gate_exps` and `ffn_up_exps` is NOT what decides it --
    // that gap is the same at every column. THE GRID SIZE IS. The pair halves the
    // threadgroup count, and at one token only `rows` experts are active (4 of 32), so the
    // grid is already the smallest it ever gets: 224 x 4 groups against two dispatches'
    // 224 x 4 twice. By 8 tokens every expert holds a work row, both grids are far past
    // saturation, and the pair keeps its halved activation loads and its saved dispatch.
    //
    // This is the same rule the DENSE pair reached first: imparo_metal_matmat_gated is
    // "prefill only -- at one token the split GEMV path is the measured better form".
    //
    // The threshold is 8, not 4: 4 is the measured tie and 8 is the first clear win.
    // IMPARO_MOE_PAIR=0 refuses the pair outright, so the two forms can be timed in one
    // binary. A config, not a knob: it selects a route.
    static int pair_on = -1;
    if (pair_on < 0) { const char * e = getenv("IMPARO_MOE_PAIR"); pair_on = !(e && e[0] == '0'); }
    if (!pair_on || n_tok < moe_pair_min_tok()) { return 1; }
    const bool tile_major =
        g_wire_ggml_set && wkind < 64u && g_wire_ggml[wkind] >= 1000u;
    const uint32_t wfmt = moe_wfmt_for(wkind);
    // THE STAGED GEMM OWNS THE BATCHED SHAPE, AND NOW HAS A PAIRED FORM. It used to have
    // none, so this returned and the caller ran gate, up and act_mul as three dispatches --
    // which writes gate (14.7 MB a routed layer-chunk on LFM2.5-8B-A1B), writes up, then
    // reads both back and writes the hidden. 11.63 GB over a 5962-token prefill at
    // 138 GB/s, the device ceiling: not slow, just work that need not happen.
    // IMPARO_MOE_ST_PAIR=0 restores the three dispatches for the A/B.
    const MoeStShape sp_sh = moe_st_shape(moe_st_pair_shape(n_tok));
    if (moe_st_wanted(n_tok, prefill_chunk != 0u)) {
        static int st_pair = -1;
        if (st_pair < 0) {
            const char * e = getenv("IMPARO_MOE_ST_PAIR");
            st_pair = !(e != nullptr && e[0] == '0');
        }
        const uint32_t st_tok = sp_sh.tokens;
        // The FFN input's half mirror when the norm left one; its result as half when the
        // down projection will stage it (the same token floor), published below as G's mirror.
        bool sp_src_half = false;
        const uint32_t sp_src = moe_st_source(src, (uint64_t)n_tok * n_in, &sp_src_half);
        const bool sp_dst_half = moe_st_wanted(n_tok, prefill_chunk != 0u);
        id<MTLComputePipelineState> sp = (st_pair && sp_src < B_COUNT)
            ? moe_st_pipeline_for_fmt(wfmt, !tile_major, moe_st_pair_shape(n_tok), sp_src_half,
                                      sp_dst_half)
            : nil;
        const uint64_t span2 = (uint64_t)n_expert * expert_stride;
        if (sp == nil || rows == 0u || n_expert == 0u || n_expert > MOE_MAX_EXPERTS
            || n_in == 0u || n_out == 0u || (n_in % 32u) != 0u
            || src >= B_COUNT || dst >= B_COUNT || perm >= B_COUNT || seg >= B_COUNT
            || g.bufs[src] == nil || g.bufs[dst] == nil || g.bufs[perm] == nil
            || g.bufs[seg] == nil || src == dst
            || (uint64_t)n_tok * n_in * 4ull > g.sizes[src]
            || (uint64_t)rows * n_out * 4ull > g.sizes[dst]
            || (uint64_t)rows * 4ull > g.sizes[perm]
            || (2ull * (uint64_t)n_expert + 2ull) * 4ull > g.sizes[seg]
            // EACH STACK IN ONE SEGMENT; the two stacks may sit in different ones. Gate is
            // bound at 0 and up at 16, so a file whose repack windows split a layer's gate
            // and up stacks (Q4_0: 4 of LFM2.5-8B-A1B's 22 layers) still takes the pair.
            // It used to require both in one bound buffer and fell back to three
            // dispatches and an act_mul for those layers.
            || !w_pair_in_one_segment(gate_off, gate_off + span2 - 1ull)
            || !w_pair_in_one_segment(up_off, up_off + span2 - 1ull)) {
            return 1;
        }
        if (g_skip_cat == PC_MOE_EXPERT) { return 0; }
        haz(hb(sp_src) | hb(perm) | hb(seg), hb(dst));
        const NSUInteger zg2 = (NSUInteger)((rows < n_expert) ? rows : n_expert);
        [g.enc setComputePipelineState:sp];
        const uint64_t wls = wbind(g.enc, gate_off, 0);
        const uint64_t wls2 = wbind(g.enc, up_off, 16);
        [g.enc setBuffer:g.bufs[sp_src] offset:g.buf_off[sp_src] atIndex:1];
        [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
        [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:3];
        [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:4];
        [g.enc setBytes:&wls length:8 atIndex:5];
        [g.enc setBytes:&expert_stride length:8 atIndex:6];
        [g.enc setBytes:&n_in length:4 atIndex:7];
        [g.enc setBytes:&n_out length:4 atIndex:8];
        const uint32_t swr = 0u;
        [g.enc setBytes:&swr length:4 atIndex:9];
        [g.enc setBytes:&n_expert length:4 atIndex:14];
        [g.enc setBytes:&wls2 length:8 atIndex:15];
        [g.enc setThreadgroupMemoryLength:
            moe_st_tg_bytes(sp_sh.rows, st_tok, sp_sh.k) atIndex:0];
        const NSUInteger rg2 = (n_out + sp_sh.rows * sp_sh.rblk - 1u) / (sp_sh.rows * sp_sh.rblk);
        const NSUInteger tg2 = (n_tok + st_tok - 1u) / st_tok;
        g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_PAIR); }
        [g.enc dispatchThreadgroups:MTLSizeMake(rg2, tg2, zg2)
              threadsPerThreadgroup:MTLSizeMake(32u * sp_sh.nsg, 1, 1)];
        if (g_prof) { prof_end(); }
        if (sp_dst_half) {
            // G holds its rows as HALF, in its own first half: the down projection's staged
            // GEMM reads them as its mirror, and every float reader refuses G until it is
            // written again. Not B_XH2: that aliases U, which the down projection writes whole.
            g_xh_src = dst; g_xh_elems = (uint64_t)rows * n_out; g_xh_buf = dst;
            g_float_stale |= hb(dst);
        }
        return 0;
    }
    if (n_tok < moe_mma_min_tok()) { return 1; }
    if (src < 64u && ((g_float_stale >> src) & 1ull) != 0ull) { return 1; }
    const uint32_t tiles = moe_mma_tiles();
    id<MTLComputePipelineState> mm =
        moe_grouped_mma_pipeline_for_fmt(wfmt, tiles, !tile_major, true);
    const uint64_t span = (uint64_t)n_expert * expert_stride;
    if (mm == nil || rows == 0u || n_expert == 0u || n_expert > MOE_MAX_EXPERTS || n_in == 0u
        || n_out == 0u || (n_in % 32u) != 0u
        || src >= B_COUNT || dst >= B_COUNT || perm >= B_COUNT || seg >= B_COUNT
        || g.bufs[src] == nil || g.bufs[dst] == nil || g.bufs[perm] == nil
        || g.bufs[seg] == nil || src == dst
        // The pair is the gate and up projections, whose source holds one row per TOKEN.
        || (uint64_t)n_tok * n_in * 4ull > g.sizes[src]
        || (uint64_t)rows * n_out * 4ull > g.sizes[dst]
        || (uint64_t)rows * 4ull > g.sizes[perm]
        || (2ull * (uint64_t)n_expert + 2ull) * 4ull > g.sizes[seg]
        || !w_pair_in_one_segment(gate_off, gate_off + span - 1ull)
        || !w_pair_in_one_segment(up_off, up_off + span - 1ull)
        || !w_pair_in_one_segment(gate_off, up_off)) {
        return 1;
    }
    if (g_skip_cat == PC_MOE_EXPERT) { return 0; }
    haz(hb(src) | hb(perm) | hb(seg), hb(dst));
    const uint32_t sgs = moe_pair_sgs(n_in);
    [g.enc setComputePipelineState:mm];
    const uint64_t wlg = wbind(g.enc, gate_off, 0);
    const uint64_t wlu = wlocal(up_off, gate_off);
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:1];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:2];
    [g.enc setBuffer:g.bufs[perm] offset:g.buf_off[perm] atIndex:3];
    [g.enc setBuffer:g.bufs[seg] offset:g.buf_off[seg] atIndex:4];
    [g.enc setBytes:&wlg length:8 atIndex:5];
    [g.enc setBytes:&expert_stride length:8 atIndex:6];
    [g.enc setBytes:&n_in length:4 atIndex:7];
    [g.enc setBytes:&n_out length:4 atIndex:8];
    // The gate and up projections read the TOKEN's activations, always.
    const uint32_t src_work_rows = 0u;
    [g.enc setBytes:&src_work_rows length:4 atIndex:9];
    const uint32_t epi = 1u;
    [g.enc setBytes:&epi length:4 atIndex:13];
    [g.enc setBytes:&n_expert length:4 atIndex:14];
    [g.enc setBytes:&wlu length:8 atIndex:15];
    // TWICE the single projection's: each simdgroup reduces two stacks, the second stack's
    // sums laid after every simdgroup's first.
    [g.enc setThreadgroupMemoryLength:(NSUInteger)2u * sgs * tiles * 64u * sizeof(float)
                              atIndex:0];
    const uint32_t rows_per_tg = 8u * tiles;
    const NSUInteger zg = (NSUInteger)((rows < n_expert) ? rows : n_expert);
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_EXPERT, (uint64_t)zg * expert_stride); }
    [g.enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1u) / rows_per_tg, zg, 1)
          threadsPerThreadgroup:MTLSizeMake(32u * sgs, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

// THE COMBINE FOLDED INTO THE NEXT NORM (imparo.metal, imparo_moe_combine_add_rms_norm):
// resid += sum over the k slots of wgt * ydown, then dst = rms_norm(resid) * w, with the half
// mirror under the same rule as imparo_metal_add_rms_norm -- whose thread count this copies,
// because the reduction's partition is part of the bits. Returns 0 having encoded it, nonzero
// having encoded nothing (the caller then runs moe_combine and add_rms_norm).
// k == 0: THE RESIDUAL ALONE (imparo_metal_rms_norm_resid) -- resid already holds resid + O;
// ydown / wgt / inv are then stand-ins the kernel never reads.
static int combine_norm_impl(uint32_t ydown, uint32_t wgt, uint32_t inv, uint32_t k,
                             uint32_t dst, uint32_t resid, uint64_t w_off, uint32_t width,
                             float eps, uint32_t n_row) {
    const uint64_t rows = (uint64_t)n_row * k;
    if (g.p_moe_combine_norm == nil || n_row == 0u || k > MOE_MAX_K
        || width == 0u || (width % 4u) != 0u || w_off == ~0ull
        || ydown >= B_COUNT || wgt >= B_COUNT || inv >= B_COUNT || dst >= B_COUNT
        || resid >= B_COUNT || g.bufs[ydown] == nil || g.bufs[wgt] == nil
        || g.bufs[inv] == nil || g.bufs[dst] == nil || g.bufs[resid] == nil
        || dst == resid || (k != 0u && (ydown == dst || ydown == resid))
        || rows * width * 4ull > g.sizes[ydown] || rows * 4ull > g.sizes[wgt]
        || rows * 4ull > g.sizes[inv] || (uint64_t)n_row * width * 4ull > g.sizes[dst]
        || (uint64_t)n_row * width * 4ull > g.sizes[resid]
        || (g.buf_off[resid] % 16u) != 0u || (g.buf_off[dst] % 16u) != 0u
        || (g.buf_off[ydown] % 16u) != 0u) {
        return 1;
    }
    if (g_skip_cat == (k != 0u ? PC_MOE_COMBINE : PC_RMSNORM)) { return 0; }
    const bool xh_on = g_half_a != 0u && dst == B_CUR && n_row >= HALF_A_MIN
                    && g.bufs[B_XH] != nil
                    && (uint64_t)n_row * width * 2ull <= g.sizes[B_XH]
                    && half_consumers_exist();
    // The kernel reads the norm weight as float4: a misaligned tensor is refused before
    // haz() records anything (binding a buffer writes no memory).
    const uint64_t w_off_l = wbind(g.enc, w_off, 0);
    if ((w_off_l % 16u) != 0u) { return 1; }
    haz((k != 0u ? hb(ydown) | hb(wgt) | hb(inv) : 0u) | hb(resid),
        (k != 0u ? hb(resid) : 0u) | hb(dst) | (xh_on ? hb(B_XH) : 0u));
    [g.enc setComputePipelineState:g.p_moe_combine_norm];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&w_off_l length:8 atIndex:2];
    [g.enc setBytes:&width length:4 atIndex:3];
    [g.enc setBytes:&eps length:4 atIndex:4];
    [g.enc setBytes:&n_row length:4 atIndex:5];
    [g.enc setBuffer:g.bufs[resid] offset:g.buf_off[resid] atIndex:8];
    [g.enc setBuffer:g.bufs[ydown] offset:g.buf_off[ydown] atIndex:10];
    if (g.bufs[B_XH] != nil) { [g.enc setBuffer:g.bufs[B_XH] offset:g.buf_off[B_XH] atIndex:11]; }
    else { [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:11]; }
    const uint32_t xh_flag = xh_on ? 1u : 0u;
    [g.enc setBytes:&xh_flag length:4 atIndex:12];
    [g.enc setBuffer:g.bufs[wgt] offset:g.buf_off[wgt] atIndex:13];
    [g.enc setBuffer:g.bufs[inv] offset:g.buf_off[inv] atIndex:14];
    [g.enc setBytes:&k length:4 atIndex:15];
    // imparo_metal_add_rms_norm's thread count, line for line.
    const NSUInteger vec_units = (width + 3) / 4;
    NSUInteger threads = 32;
    while (threads < vec_units && threads < 1024) { threads *= 2; }
    threads = std::min(threads, (vec_units + 31) / 32 * 32);
    threads = std::max<NSUInteger>(threads, 32);
    [g.enc setThreadgroupMemoryLength:tg_bytes16((threads / 32) * sizeof(float)) atIndex:0];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(k != 0u ? PC_MOE_COMBINE : PC_RMSNORM); }
    [g.enc dispatchThreadgroups:MTLSizeMake(n_row, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (g_prof) { prof_end(); }
    if (xh_on) { g_xh_src = dst; g_xh_elems = (uint64_t)n_row * width; g_xh_buf = B_XH; }
    return 0;
}
extern "C" int imparo_metal_moe_combine_add_rms_norm(uint32_t ydown, uint32_t wgt, uint32_t inv,
                                                     uint32_t k, uint32_t dst, uint32_t resid,
                                                     uint64_t w_off, uint32_t width, float eps,
                                                     uint32_t n_row) {
    if (k == 0u) { return 1; }
    return combine_norm_impl(ydown, wgt, inv, k, dst, resid, w_off, width, eps, n_row);
}

// THE MIXER'S OUTPUT PROJECTION ADDED INTO THE RESIDUAL (RT_RESID): resid += W . src, on the
// tile-major register-tiled GEMM over the half mirror only (g_mm_resid makes every other route
// refuse). Then imparo_metal_rms_norm_resid normalises resid with the pre-add norm's floats.
// Returns 0 having encoded it, nonzero having encoded nothing that changes a result.
extern "C" int imparo_metal_matmat_resid(uint32_t wkind, uint64_t w_off, uint32_t n_in,
                                         uint32_t n_out, uint32_t src, uint32_t resid,
                                         uint32_t n_tok) {
    if (g_epilogue != 0u || resid == src) { return 1; }
    g_mm_resid = true;
    const bool ok = matmat_impl(wkind, w_off, NO_PAIR, n_in, n_out, src, resid, n_tok, 0u);
    g_mm_resid = false;
    return ok ? 0 : 1;
}
extern "C" int imparo_metal_rms_norm_resid(uint32_t dst, uint32_t resid, uint64_t w_off,
                                           uint32_t width, float eps, uint32_t n_row) {
    return combine_norm_impl(resid, resid, resid, 0u, dst, resid, w_off, width, eps, n_row);
}

extern "C" int imparo_metal_moe_combine(uint32_t src, uint32_t wgt, uint32_t inv, uint32_t dst,
                                        uint32_t n_embd, uint32_t k, uint32_t n_tok) {
    const uint64_t rows = (uint64_t)n_tok * k;
    if (n_tok == 0u || k == 0u || k > MOE_MAX_K || n_embd == 0u
        || src >= B_COUNT || wgt >= B_COUNT || inv >= B_COUNT || dst >= B_COUNT
        || g.bufs[src] == nil || g.bufs[wgt] == nil || g.bufs[inv] == nil
        || g.bufs[dst] == nil || dst == src
        || rows * n_embd * 4ull > g.sizes[src] || rows * 4ull > g.sizes[wgt]
        || rows * 4ull > g.sizes[inv] || (uint64_t)n_tok * n_embd * 4ull > g.sizes[dst]) {
        NSLog(@"imparo metal: moe_combine refused (n_tok=%u k=%u n_embd=%u)", n_tok, k, n_embd);
        return 1;
    }
    haz(hb(src) | hb(wgt) | hb(inv), hb(dst));
    [g.enc setComputePipelineState:g.p_moe_combine];
    [g.enc setBuffer:g.bufs[src] offset:g.buf_off[src] atIndex:0];
    [g.enc setBuffer:g.bufs[wgt] offset:g.buf_off[wgt] atIndex:1];
    [g.enc setBuffer:g.bufs[inv] offset:g.buf_off[inv] atIndex:2];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:3];
    [g.enc setBytes:&n_embd length:4 atIndex:4];
    [g.enc setBytes:&k length:4 atIndex:5];
    const NSUInteger tw =
        std::min<NSUInteger>(256, g.p_moe_combine.maxTotalThreadsPerThreadgroup);
    // ONE THREADGROUP PER (TOKEN, EMBEDDING CHUNK). A token's n_embd outputs are
    // independent sums, so a grid of n_tok alone put all of them on one core -- 1x1x1 at
    // decode, 12.5 us to move 40 KB. The kernel strides by tcount * grid.y, so any chunk
    // count is correct and this one gives each threadgroup a single pass.
    // IMPARO_MOE_COMBINE_CHUNKS=1 puts the whole embedding back on one threadgroup, so the
    // split can be timed against what it replaced in ONE binary. A config, not a knob: any
    // value is correct, because the kernel strides by tcount * chunks.
    static int chunk_cap = -1;
    if (chunk_cap < 0) {
        const char * e = getenv("IMPARO_MOE_COMBINE_CHUNKS");
        chunk_cap = e != nullptr ? atoi(e) : 0;
    }
    uint32_t chunks = (uint32_t)std::max<NSUInteger>(1, ((NSUInteger)n_embd + tw - 1) / tw);
    if (chunk_cap > 0 && (uint32_t)chunk_cap < chunks) { chunks = (uint32_t)chunk_cap; }
    [g.enc setBytes:&chunks length:4 atIndex:6];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_MOE_COMBINE); }
    [g.enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tok * chunks, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
}

// ONE LOGISTIC OUTPUT PER ROW: see imparo_logistic_blocks. dst holds the outputs, then one partial
// sum per row and block; LOGISTIC_BLOCK matches the shader's.
static constexpr uint32_t LOGISTIC_BLOCK = 64u;
struct LogisticArgs { uint32_t a_width, b_width, w_off, dst_off, rows, a_blocks, blocks; };
static LogisticArgs logistic_args(uint32_t a_width, uint32_t b_width, uint32_t w_off,
                                  uint32_t dst_off, uint32_t rows) {
    const uint32_t a_blocks = (uint32_t)(((uint64_t)a_width + LOGISTIC_BLOCK - 1u) / LOGISTIC_BLOCK);
    const uint32_t b_blocks = (uint32_t)(((uint64_t)b_width + LOGISTIC_BLOCK - 1u) / LOGISTIC_BLOCK);
    return LogisticArgs{a_width, b_width, w_off, dst_off, rows, a_blocks, a_blocks + b_blocks};
}
extern "C" uint64_t imparo_metal_logistic_rows_len(uint32_t a_width, uint32_t b_width,
                                                   uint32_t rows) {
    return (uint64_t)rows * (1ull + logistic_args(a_width, b_width, 0u, 0u, rows).blocks);
}
extern "C" int imparo_metal_logistic_rows(uint32_t a, uint32_t a_width, uint32_t b,
                                          uint32_t b_width, uint32_t w, uint32_t w_off,
                                          uint32_t dst, uint32_t dst_off, uint32_t rows) {
    const LogisticArgs p = logistic_args(a_width, b_width, w_off, dst_off, rows);
    const uint64_t w_end = (uint64_t)w_off + a_width + b_width + 1ull;
    const uint64_t dst_end =
        (uint64_t)dst_off + imparo_metal_logistic_rows_len(a_width, b_width, rows);
    if (rows == 0u || p.blocks == 0u || a >= B_COUNT || b >= B_COUNT || w >= B_COUNT
        || dst >= B_COUNT || g.bufs[a] == nil || g.bufs[b] == nil || g.bufs[w] == nil
        || g.bufs[dst] == nil || dst == a || dst == b
        || (uint64_t)rows * a_width * 4ull > g.sizes[a]
        || (uint64_t)rows * b_width * 4ull > g.sizes[b]
        || w_end * 4ull > g.sizes[w] || dst_end * 4ull > g.sizes[dst]
        || (w == dst && w_off < dst_end && dst_off < w_end)) {
        NSLog(@"imparo metal: logistic_rows refused (a=%u b=%u w=%u dst=%u rows=%u)", a, b, w,
              dst, rows);
        return 1;
    }
    haz(hb(a) | hb(b) | hb(w), hb(dst));
    [g.enc setComputePipelineState:g.p_logistic_blocks];
    [g.enc setBuffer:g.bufs[a] offset:g.buf_off[a] atIndex:0];
    [g.enc setBuffer:g.bufs[b] offset:g.buf_off[b] atIndex:1];
    [g.enc setBuffer:g.bufs[w] offset:g.buf_off[w] atIndex:2];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:3];
    [g.enc setBytes:&p length:sizeof(p) atIndex:4];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_LOGISTIC); }
    // One SIMD group per threadgroup.
    const NSUInteger tw = std::max<NSUInteger>(1, std::min<NSUInteger>(
        g.p_logistic_blocks.threadExecutionWidth, p.blocks));
    [g.enc dispatchThreads:MTLSizeMake(p.blocks, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    if (g_prof) { prof_end(); }
    // The finish reads the block sums the blocks pass wrote.
    haz(hb(dst) | hb(w), hb(dst));
    [g.enc setComputePipelineState:g.p_logistic_finish];
    [g.enc setBuffer:g.bufs[w] offset:g.buf_off[w] atIndex:0];
    [g.enc setBuffer:g.bufs[dst] offset:g.buf_off[dst] atIndex:1];
    [g.enc setBytes:&p length:sizeof(p) atIndex:2];
    g_disp_seq += 1; if (g_prof) { g_prof_disp += 1; prof_begin(PC_LOGISTIC); }
    [g.enc dispatchThreads:MTLSizeMake(rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(rows, g.p_logistic_finish.maxTotalThreadsPerThreadgroup), 1, 1)];
    if (g_prof) { prof_end(); }
    return 0;
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
    buf_fresh_clear(dst);
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

// THE FILE, FOR CONVERTED WEIGHTS ONLY. A converted tensor is read once and its source dies
// the moment the twin exists, so reading it through the mapping leaves the file's bytes in
// the page cache next to the twin that replaced them: measured +9588 MiB of file-backed pages
// across a Qwen3.8-27B load, a second copy of the whole model. Reading it with F_NOCACHE
// instead leaves nothing behind, and is also ~4.7x faster cold (848 vs 3962 MB/s measured,
// mmap fault-in against pread), because page faults are the slow part of a bulk read.
//
// Weights used DIRECTLY keep the mapping: there the mapping IS the resident copy, one copy,
// and pread would only trade clean file pages for anonymous ones.
extern "C" void imparo_metal_set_weight_path(const char * path) {
    g_weight_path = (path != nullptr) ? path : "";
}

// ---- Load-time repack (docs/memory-tiers-and-fit.md section 7) -------------------------
// A fast-tier segment that holds a convertible tensor gets a PRIVATE twin: the mapped
// window is blit-copied into it once, then each convertible tensor is rewritten by the
// repack kernel from the mapping into the twin at the same local offset. The twin then
// replaces the mapping buffer in the segment table and the residency set, so every
// dispatch keeps binding the segment at the same local offsets and reads tile-major bytes;
// the mapped pages of that window are no longer referenced and the OS may drop them.
// Only Q8_0 -> Q8_0_TM has readers today; other kinds are left as they are (applied = 0).
// The job carries THE LAYOUT (imparo_backend::WeightBlockLayout), so this file states no
// format's block structure: the repack moves the spans it is handed.
struct RepackSpans {
    uint32_t block_elems, block_bytes, unit_rows, n_spans, n_scale_spans;
    uint32_t span_off[5], span_len[5];
};
struct WXformWire {
    uint64_t off, bytes;
    uint32_t from_type, to_type, n_in, n_out;
    RepackSpans layout;
};
static_assert(sizeof(RepackSpans) == 60, "repack span table drift");
static_assert(sizeof(WXformWire) == 96, "transform wire drift");

// The register-tiled GEMM compiled for one tile-major FORMAT (WFMT, constant 21). Built on
// first use and kept: a pipeline is GPU-resident code, and a model touches at most a
// handful of formats.
static id<MTLComputePipelineState> rt_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                       uint32_t variant, uint32_t tile) {
    if (wfmt == 0u || wfmt >= 32u || variant > RT_HALF_RESID || tile >= RT_TILES
        || g.lib == nil) { return nil; }
    const uint32_t slot = wfmt + (rowmajor ? 32u : 0u);
    if (g.p_rt_fmt[variant][tile][slot] != nil) { return g.p_rt_fmt[variant][tile][slot]; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    stamp_epi_act(cv);   // constant 11: SiLU vs GELU, see the builder
    static int probe = -1;
    if (probe < 0) { probe = getenv("IMPARO_WFMT_PROBE") != nullptr; }
    const bool pb = probe != 0;
    [cv setConstantValue:&pb type:MTLDataTypeBool atIndex:23];
    if (pb) { NSLog(@"imparo metal: WFMT PROBE ON -- weights replaced by a constant"); }
    if (g_rt_mma_fence != 0u) {
        const bool rf = true;
        [cv setConstantValue:&rf type:MTLDataTypeBool atIndex:16];
    }
    // RT_HALF_RESID: the half-mirror entry point with the residual-add store (constant 60).
    if (variant == RT_HALF_RESID) {
        const bool ra = true;
        [cv setConstantValue:&ra type:MTLDataTypeBool atIndex:60];
    }
    NSError * e = nil;
    // THE VARIANT IS PART OF THE PIPELINE, NOT A LATER SUBSTITUTION. The route picks an
    // rt entry point from (half-activation mirror, gated pair) before it dispatches; the
    // tile-major family has to be compiled into THAT entry point, because every entry
    // point stages its own weights. Building only the plain one and letting the half-A
    // branch overwrite the selection with `g.p_rt_h[shape]` dispatched a Q4_K tensor on
    // the Q4_0 kernel -- payload bytes read as half scales, Inf and NaN by construction,
    // with the "built the register-tiled GEMM for weight format 12" line in the log.
    // The same entry points the plain family dispatches, compiled for this format: a
    // wide shape by its index, or the narrow tile the route takes for 2..nb8_max rows.
    static const char * const SUF[4] = { "", "_h", "_gh", "_h" };
    NSString * nm = tile < RT_CANDIDATES
        ? [NSString stringWithFormat:@"imparo_rt_%u%s", tile, SUF[variant]]
        : [NSString stringWithFormat:@"imparo_rt_nb8%s%s", tile == RT_TILE_NB8B ? "b" : "",
                                     SUF[variant]];
    id<MTLFunction> f = [g.lib newFunctionWithName:nm constantValues:cv error:&e];
    if (f == nil) {
        NSLog(@"imparo metal: no rt function %@ for WFMT %u: %@", nm, wfmt, e);
        return nil;
    }
    g.p_rt_fmt[variant][tile][slot] =
        rt_register([g.device newComputePipelineStateWithFunction:f error:&e], nm, cv);
    if (g.p_rt_fmt[variant][tile][slot] == nil) {
        NSLog(@"imparo metal: rt pipeline %@ for WFMT %u: %@", nm, wfmt, e);
    } else {
        NSLog(@"imparo metal: built the register-tiled GEMM %@ for weight format %u "
              @"(%s, max threads %lu, mma_fence=%u)", nm, wfmt,
              rowmajor ? "row-major" : "tile-major",
              (unsigned long)[g.p_rt_fmt[variant][tile][slot] maxTotalThreadsPerThreadgroup],
              g_rt_mma_fence);
    }
    return g.p_rt_fmt[variant][tile][slot];
}

static id<MTLComputePipelineState> rt_plain_resid(uint32_t tile) {
    if (tile >= RT_TILES || g.lib == nil) { return nil; }
    if (g.p_rt_plain_resid[tile] != nil) { return g.p_rt_plain_resid[tile]; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    stamp_epi_act(cv);
    // The wide tiles are built by make_rt, which stamps the fence when it is on; the narrow
    // tiles by make_rt_plain, which does not.
    if (tile < RT_CANDIDATES && g_rt_mma_fence != 0u) {
        const bool rf = true;
        [cv setConstantValue:&rf type:MTLDataTypeBool atIndex:16];
    }
    const bool ra = true;
    [cv setConstantValue:&ra type:MTLDataTypeBool atIndex:60];
    NSString * nm = tile < RT_CANDIDATES
        ? [NSString stringWithFormat:@"imparo_rt_%u_h", tile]
        : (tile == RT_TILE_NB8B ? @"imparo_rt_nb8b_h" : @"imparo_rt_nb8_h");
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:nm constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: no rt function %@ (residual add): %@", nm, e); return nil; }
    g.p_rt_plain_resid[tile] =
        rt_register([g.device newComputePipelineStateWithFunction:f error:&e], nm, cv);
    if (g.p_rt_plain_resid[tile] == nil) {
        NSLog(@"imparo metal: rt pipeline %@ (residual add): %@", nm, e);
    }
    return g.p_rt_plain_resid[tile];
}

// THE BRICK'S GATE (test-only entry). Decodes `n_bytes` of ROW-MAJOR blocks of format
// `wfmt` through `tm_sub32` and returns `n_elems` floats, so a Rust test can diff the MSL
// transcription against the CPU row codec -- which is itself pinned bit-exact against
// llama.cpp's own dequantiser. Nothing on a serving path calls this.
extern "C" int imparo_metal_decode_probe(uint32_t wfmt,
                                         const void * scales, uint64_t n_scale_bytes,
                                         const void * payload, uint64_t n_pay_bytes,
                                         uint32_t n_elems, uint32_t mode, float * out) {
    if (g.device == nil || g.lib == nil) { return 1; }
    if (wfmt == 0u || wfmt >= 32u || scales == nullptr || payload == nullptr
        || out == nullptr) { return 2; }
    if (n_elems == 0u || (n_elems % 32u) != 0u) { return 3; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    // mode bit 0: the half output; bit 1: the run fetch (tm_run8); bit 2: the aligned flag;
    // bit 3: with bit 1, the pair fetch (tm_run8_pair) where the format has one; bit 4: with bit
    // 1, the half-block fetch (tm_quad_scales) where the format has one.
    const bool probe_half = (mode & 1u) != 0u;
    const bool probe_run = (mode & 2u) != 0u;
    const bool probe_a16 = (mode & 4u) != 0u;
    const bool probe_pair = (mode & 8u) != 0u;
    const bool probe_quad = (mode & 16u) != 0u;
    [cv setConstantValue:&probe_half type:MTLDataTypeBool atIndex:24];
    [cv setConstantValue:&probe_run type:MTLDataTypeBool atIndex:41];
    [cv setConstantValue:&probe_a16 type:MTLDataTypeBool atIndex:42];
    [cv setConstantValue:&probe_pair type:MTLDataTypeBool atIndex:48];
    [cv setConstantValue:&probe_quad type:MTLDataTypeBool atIndex:52];
    stamp_epi_act(cv);
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_blk_decode_probe"
                                    constantValues:cv error:&e];
    if (f == nil) { NSLog(@"imparo metal: decode probe fn for %u: %@", wfmt, e); return 4; }
    id<MTLComputePipelineState> p = [g.device newComputePipelineStateWithFunction:f error:&e];
    if (p == nil) { NSLog(@"imparo metal: decode probe pipeline %u: %@", wfmt, e); return 5; }
    id<MTLBuffer> sb = [g.device newBufferWithBytes:scales length:(NSUInteger)n_scale_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> pb = [g.device newBufferWithBytes:payload length:(NSUInteger)n_pay_bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> db = [g.device newBufferWithLength:(NSUInteger)n_elems * 4
                                             options:MTLResourceStorageModeShared];
    if (sb == nil || pb == nil || db == nil) { return 6; }
    const uint32_t n_subs = n_elems / 32u;
    id<MTLCommandBuffer> cb = [g.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:sb offset:0 atIndex:0];
    [enc setBuffer:pb offset:0 atIndex:1];
    [enc setBuffer:db offset:0 atIndex:2];
    [enc setBytes:&n_subs length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(n_subs, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(n_subs < 32u ? n_subs : 32u, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) { NSLog(@"imparo metal: decode probe cb: %@", cb.error); return 7; }
    memcpy(out, db.contents, (size_t)n_elems * 4);
    return 0;
}

// The row gather compiled for one block format. Same lazy rule as the GEMM's.
static id<MTLComputePipelineState> gather_pipeline_for_fmt(uint32_t wfmt) {
    if (wfmt == 0u || wfmt >= 32u || g.lib == nil) { return nil; }
    if (g.p_gather_fmt[wfmt] != nil) { return g.p_gather_fmt[wfmt]; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    stamp_epi_act(cv);   // constant 11: SiLU vs GELU, see the builder
    static int gprobe = -1;
    if (gprobe < 0) { gprobe = getenv("IMPARO_WFMT_PROBE") != nullptr; }
    const bool gpb = gprobe != 0;
    [cv setConstantValue:&gpb type:MTLDataTypeBool atIndex:23];
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_blk_gather_rows"
                                    constantValues:cv error:&e];
    g.p_gather_fmt[wfmt] = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (g.p_gather_fmt[wfmt] == nil) {
        NSLog(@"imparo metal: gather pipeline for weight format %u: %@", wfmt, e);
    }
    return g.p_gather_fmt[wfmt];
}

// The small-batch GEMV compiled for one (format, layout). Same lazy rule as the GEMM's.
static id<MTLComputePipelineState> gemv_pipeline_for_fmt(uint32_t wfmt, bool rowmajor) {
    if (wfmt == 0u || wfmt >= 32u || g.lib == nil) { return nil; }
    const uint32_t slot = wfmt + (rowmajor ? 32u : 0u);
    if (g.p_gemv_fmt[slot] != nil) { return g.p_gemv_fmt[slot]; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    stamp_epi_act(cv);   // constant 11: SiLU vs GELU, see the builder
    const bool pb = getenv("IMPARO_WFMT_PROBE") != nullptr;
    [cv setConstantValue:&pb type:MTLDataTypeBool atIndex:23];
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_blk_gemv" constantValues:cv
                                             error:&e];
    g.p_gemv_fmt[slot] = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (g.p_gemv_fmt[slot] == nil) {
        NSLog(@"imparo metal: gemv pipeline for weight format %u: %@", wfmt, e);
    } else {
        NSLog(@"imparo metal: built the small-batch GEMV for weight format %u (%s, %u rows "
              @"per simdgroup)", wfmt, rowmajor ? "row-major" : "tile-major", blk_gemv_nr());
    }
    return g.p_gemv_fmt[slot];
}

// Its decode-rows form for 1 << form_log2 tokens (2, 4 or 8), built on first use: a process
// that never co-batches builds none. The one-row pipeline's constants plus the token count at 39.
static id<MTLComputePipelineState> blk_rows_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                             uint32_t form_log2) {
    if (wfmt == 0u || wfmt >= 32u || form_log2 < 1u || form_log2 > 3u || g.lib == nil) {
        return nil;
    }
    const uint32_t slot = wfmt + (rowmajor ? 32u : 0u);
    id<MTLComputePipelineState> __strong & have = g.p_blk_rows_fmt[slot][form_log2 - 1u];
    if (have != nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    stamp_epi_act(cv);   // constant 11: SiLU vs GELU, see the builder
    const bool pb = getenv("IMPARO_WFMT_PROBE") != nullptr;
    [cv setConstantValue:&pb type:MTLDataTypeBool atIndex:23];
    const uint32_t tok = 1u << form_log2;
    [cv setConstantValue:&tok type:MTLDataTypeUInt atIndex:39];
    // DIAGNOSTIC (IMPARO_BLK_ROWS_SKIP, the kernel's skip bits); unset in every served run.
    static int skip = -1;
    if (skip < 0) { const char * e = getenv("IMPARO_BLK_ROWS_SKIP"); skip = e ? atoi(e) : 0; }
    if (skip > 0) {
        const uint32_t sk = (uint32_t)skip;
        [cv setConstantValue:&sk type:MTLDataTypeUInt atIndex:40];
    }
    NSError * e = nil;
    id<MTLFunction> f = [g.lib newFunctionWithName:@"imparo_blk_gemv_rows" constantValues:cv
                                             error:&e];
    have = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (have == nil) {
        NSLog(@"imparo metal: decode-rows GEMV for weight format %u, %u tokens: %@", wfmt, tok, e);
    } else if (getenv("IMPARO_WFMT_LOG")) {
        NSLog(@"imparo metal: built the decode-rows GEMV for weight format %u (%s, %u tokens, "
              @"max_threads=%lu)", wfmt, rowmajor ? "row-major" : "tile-major", tok,
              (unsigned long)[have maxTotalThreadsPerThreadgroup]);
    }
    return have;
}

// Its matrix-unit form (imparo_blk_rows_mma), built on first use: the one-row pipeline's
// constants; one 8-token column serves every row count it takes.
static id<MTLComputePipelineState> blk_rows_mma_pipeline_for_fmt(uint32_t wfmt, bool rowmajor,
                                                                uint32_t tiles, bool xhalf,
                                                                uint32_t frags) {
    if (wfmt == 0u || wfmt >= 32u || g.lib == nil) { return nil; }
    if (tiles != 1u && tiles != 2u && tiles != 4u) { return nil; }
    if (frags != 1u && (frags != 2u || xhalf)) { return nil; }
    const uint32_t slot = wfmt + (rowmajor ? 32u : 0u);
    id<MTLComputePipelineState> __strong & have = frags == 2u
        ? g.p_blk_rows_mma2_fmt[slot][tiles == 4u ? 2u : tiles - 1u]
        : g.p_blk_rows_mma_fmt[slot][tiles == 4u ? 2u : tiles - 1u][xhalf ? 1u : 0u];
    if (have != nil) { return have; }
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    [cv setConstantValue:&wfmt type:MTLDataTypeUInt atIndex:21];
    [cv setConstantValue:&rowmajor type:MTLDataTypeBool atIndex:22];
    stamp_epi_act(cv);   // constant 11: SiLU vs GELU, see the builder
    [cv setConstantValue:&tiles type:MTLDataTypeUInt atIndex:43];
    // DIAGNOSTIC (IMPARO_BLK_MMA_SKIP, the kernel's skip bits); unset in every served run.
    static int skip = -1;
    if (skip < 0) { const char * se = getenv("IMPARO_BLK_MMA_SKIP"); skip = se ? atoi(se) : 0; }
    if (skip > 0) {
        const uint32_t sk = (uint32_t)skip;
        [cv setConstantValue:&sk type:MTLDataTypeUInt atIndex:44];
    }
    // DIAGNOSTIC (IMPARO_BLK_RUN_PROBE, the run fetch's probe bits); unset in every served run.
    static int runp = -1;
    if (runp < 0) { const char * re = getenv("IMPARO_BLK_RUN_PROBE"); runp = re ? atoi(re) : 0; }
    if (runp > 0) {
        const uint32_t rp = (uint32_t)runp;
        [cv setConstantValue:&rp type:MTLDataTypeUInt atIndex:50];
    }
    NSError * e = nil;
    // A SEPARATE ENTRY POINT for the half mirror: the two kernels share one templated body in the
    // .metal, and the float one carries no half code and no extra buffer argument.
    id<MTLFunction> f = [g.lib newFunctionWithName:(frags == 2u ? @"imparo_blk_rows_mma_t2"
                                                    : xhalf ? @"imparo_blk_rows_mma_xh"
                                                            : @"imparo_blk_rows_mma")
                                    constantValues:cv error:&e];
    have = f ? [g.device newComputePipelineStateWithFunction:f error:&e] : nil;
    if (have == nil) {
        NSLog(@"imparo metal: matrix-unit rows for weight format %u: %@", wfmt, e);
    } else if (getenv("IMPARO_WFMT_LOG")) {
        NSLog(@"imparo metal: built the matrix-unit rows for weight format %u (%s, tiles=%u, "
              @"activations %s, max_threads=%lu)", wfmt, rowmajor ? "row-major" : "tile-major",
              tiles, xhalf ? "half" : "float", (unsigned long)[have maxTotalThreadsPerThreadgroup]);
    }
    return have;
}

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
// THE TRANSFORM'S WORKING SET IS ONE WINDOW, NOT THE WHOLE MODEL.
//
// A fast segment's buffer is a NO-COPY WRAPPER over the weight mapping, so a twin of it is
// the only private allocation the repack makes -- and a twin of the whole segment costs the
// model's size a second time. That is affordable at 2.7 GiB and not at 15.7:
//
//   Qwen3.8-27B, 36 GiB Mac, 27648 MiB working set
//     whole-segment twin   15691 resident + 15691 twin = 31382 MiB   OOM
//     windowed twin        15691 resident +  1024 twin = 16715 MiB   fits
//
// The OOM is a command-buffer failure ("Insufficient Memory
// (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"), not an allocation returning nil,
// so it cannot be caught by checking the buffer.
//
// So a segment is TILED into windows, and each window becomes a segment of its own backed by
// its own twin. The windows tile the segment exactly, so the union of the new segments is the
// old one and every lookup still resolves; boundaries fall between transformed tensors,
// because a tensor that straddled two twins would be half repacked. The byte count is
// unchanged by the layout rule, so a window's twin is the same length as its source.
//
// The window cap is weight_window_cap(), defined beside the wire at load, which pieces a
// segment by the same windows.

extern "C" int32_t imparo_metal_transform_weights(const WXformWire * jobs, uint32_t n,
                                                  uint8_t * applied) {
    if (g.device == nil || g_wsegs.empty()) { return 1; }
    for (uint32_t i = 0; i < n; ++i) { applied[i] = 0; }
    if (g.p_repack_tm == nil) { return 0; }

    // The jobs this backend will do, in file order. Order is what lets a window end
    // between two tensors rather than inside one.
    std::vector<uint32_t> todo;
    for (uint32_t i = 0; i < n; ++i) {
        const WXformWire & j = jobs[i];
        // No type test: the caller only sends jobs whose rule this backend serves
        // (Backend::serves_weight_type gates it), and the layout says how to move them.
        if (j.layout.block_elems == 0 || j.layout.n_spans == 0) { continue; }
        const WSeg * s = wseg_mut_at(j.off);
        if (s == nullptr || s->tier != WT_FAST) { continue; }
        if (j.off + j.bytes > s->off + s->bytes) { continue; }   // a tensor never straddles
        todo.push_back(i);
    }
    if (todo.empty()) { return 0; }
    std::sort(todo.begin(), todo.end(),
              [&](uint32_t x, uint32_t y) { return jobs[x].off < jobs[y].off; });

    const uint64_t page = 16384, cap = weight_window_cap();
    // The mapping starts with the model's file at offset 0 (imparo-gguf `Weights`), so a segment
    // offset below the file's end IS a file offset and no translation is needed. A paired
    // drafter's file follows past that end; it is never read from this file (`in_file`).
    int src_fd = -1;
    uint64_t src_len = 0;
    id<MTLBuffer> staging = nil;
    if (!g_weight_path.empty()) {
        src_fd = open(g_weight_path.c_str(), O_RDONLY);
        struct stat st;
        if (src_fd >= 0 && fstat(src_fd, &st) == 0) { src_len = (uint64_t)st.st_size; }
        if (src_fd >= 0 && fcntl(src_fd, F_NOCACHE, 1) == -1) {
            NSLog(@"imparo metal: F_NOCACHE unavailable (%s); the repack reads the mapping",
                  strerror(errno));
            close(src_fd);
            src_fd = -1;
        }
        if (src_fd >= 0) {
            // TWO PAGES WIDER THAN THE CAP, because that is the widest read this loop can
            // ask for: the planner bounds the job span by the cap, and the span is then
            // rounded OUT to page boundaries at both ends. At exactly `cap` a span that grew
            // to the cap would not fit its own staging buffer.
            staging = [g.device newBufferWithLength:(NSUInteger)(cap + 2 * page)
                                            options:MTLResourceStorageModeShared];
            if (staging == nil) { close(src_fd); src_fd = -1; }
        }
    }
    std::vector<WSeg> rebuilt;
    rebuilt.reserve(g_wsegs.size() + 8);
    uint64_t swapped = 0, dropped = 0;
    uint64_t read_bytes = 0, map_bytes = 0;   // pread route vs mapping route
    uint32_t windows = 0, planned = 0;
    size_t at = 0;                       // next job in `todo`

    for (const WSeg & seg : g_wsegs) {
        const size_t first = at;
        while (at < todo.size() && jobs[todo[at]].off < seg.off + seg.bytes) { at += 1; }
        if (seg.tier != WT_FAST || first == at) { rebuilt.push_back(seg); continue; }

        uint64_t cur = seg.off;
        size_t k = first;
        while (k < at) {
          // ONE POOL PER WINDOW. `commandBuffer`, `blitCommandEncoder` and
          // `computeCommandEncoder` return autoreleased objects, and this function had no
          // pool, so all 15 command buffers stayed alive to the end of the repack -- and a
          // live command buffer retains what it referenced. That is why setting the staging
          // buffer to nil did not free it: measured gpu=15662 MiB after load against 14636
          // MiB of twins, exactly one 1024 MiB window still held. Draining per window
          // releases each command buffer, and with it that window's references.
          //
          // The twins are NOT affected: `rebuilt` holds each one strongly (an `id` field in
          // a C++ struct is a strong reference under ARC), so they outlive the pool.
          @autoreleasepool {
            // Grow the window job by job while it fits. A single job wider than the cap
            // takes a window of its own: it cannot be split, and refusing it would leave
            // the model unable to load at all.
            size_t e = k;
            uint64_t end = jobs[todo[e]].off + jobs[todo[e]].bytes;
            while (e + 1 < at) {
                const uint64_t next = jobs[todo[e + 1]].off + jobs[todo[e + 1]].bytes;
                if (next - cur > cap) { break; }
                e += 1;
                end = next;
            }
            // The last window of the segment runs to the segment's end, so the windows
            // tile it exactly and no byte loses its buffer.
            const bool last = (e + 1 >= at);
            const uint64_t win_end = last ? seg.off + seg.bytes : jobs[todo[e + 1]].off;
            const uint64_t base = cur & ~(page - 1);
            const uint64_t top = (win_end + page - 1) & ~(page - 1);
            id<MTLBuffer> twin = [g.device newBufferWithLength:(NSUInteger)(top - base)
                                                       options:MTLResourceStorageModeShared];
            if (twin == nil) {
                NSLog(@"imparo metal: no twin of %llu bytes for the repack window at %llu",
                      (unsigned long long)(top - base), (unsigned long long)cur);
                return 2;
            }
            // BIND THE WINDOW, NOT THE SEGMENT. Metal makes a referenced resource resident
            // WHOLE, never by sub-range, so binding `seg.buf` -- the no-copy wrapper over the
            // entire 14.6 GB segment -- made every window's command buffer wire the whole
            // mapping next to the twins: measured 29528 MiB of system wired memory for a
            // 14636 MiB model, on a 36 GB Mac. The window cap sized the twin and nothing
            // sized the source. A wrapper over just [base, top) fixes that, and it is why
            // advising the source pages away did nothing -- the next window re-wired them.
            // THE WINDOW SPLITS IN TWO, AND ONLY THE FIRST HALF NEEDS STAGING.
            //
            //   [base .............. job_top) [job_top ....... top)
            //    the JOB span: the kernel      the GAP to the next job: tensors with no
            //    reads it while writing the    tile-major form. Nothing reads them; they
            //    twin, so it needs a copy      only have to REACH the twin, so they are
            //    that is not the twin          read straight into it
            //
            // The planner bounds the job span by the cap, so it always fits the staging
            // buffer; the gap has no such bound. Sizing the read by the WHOLE window sent
            // any window with a large gap to the mapping instead -- silently, since the
            // fallback had no message. Measured on Qwen3.8-27B: one window of 1516 MiB (a
            // 1001 MiB job span plus a 515 MiB gap) against a 1024 MiB staging buffer.
            uint64_t job_top = (jobs[todo[e]].off + jobs[todo[e]].bytes + page - 1) & ~(page - 1);
            if (job_top > top) { job_top = top; }
            // A window's last page runs past the end of the file, so a read of the full page
            // returns short. Those trailing bytes are padding no dispatch reads: stop at the
            // file's end and leave them as the buffer's own zeros.
            auto read_into = [&](uint8_t * dst, uint64_t off, uint64_t len) {
                const uint64_t want = off >= src_len ? 0 : std::min(len, src_len - off);
                uint64_t got = 0;
                while (got < want) {
                    ssize_t n = pread(src_fd, dst + got, (size_t)(want - got), (off_t)(off + got));
                    if (n <= 0) { break; }
                    got += (uint64_t)n;
                }
                return got == want;
            };
            id<MTLBuffer> src = nil;
            uint64_t src_org = base;
            // Past the file's last page the mapping holds another file (a paired drafter): a
            // window reaching there takes the mapping route.
            const bool in_file = top <= ((src_len + page - 1) & ~(page - 1));
            if (src_fd >= 0 && in_file && (job_top - base) <= (uint64_t)[staging length]) {
                if (read_into((uint8_t *)[staging contents], base, job_top - base)
                    && read_into((uint8_t *)[twin contents] + (job_top - base), job_top,
                                 top - job_top)) {
                    src = staging;
                    read_bytes += top - base;
                } else {
                    NSLog(@"imparo metal: short read of the weight window at %llu: %s; "
                          @"the repack reads the mapping",
                          (unsigned long long)base, strerror(errno));
                }
            }
            if (src == nil) {
                map_bytes += top - base;
                src = [g.device newBufferWithBytesNoCopy:(void *)((uint8_t *)[seg.buf contents] + (base - seg.base))
                                                  length:(NSUInteger)(top - base)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
                if (src == nil) { src = seg.buf; src_org = seg.base; }
            }
            id<MTLCommandBuffer> cb = [g.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            // The staged route has already put the gap in the twin; the mapping route has
            // not, so it still copies the whole window.
            const uint64_t blit_bytes = (src == staging) ? job_top - base : top - base;
            [blit copyFromBuffer:src sourceOffset:(NSUInteger)(base - src_org)
                        toBuffer:twin destinationOffset:0 size:(NSUInteger)blit_bytes];
            [blit endEncoding];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g.p_repack_tm];
            for (size_t q = k; q <= e; ++q) {
                const WXformWire & j = jobs[todo[q]];
                [enc setBuffer:src offset:(NSUInteger)(j.off - src_org) atIndex:0];
                [enc setBuffer:twin offset:(NSUInteger)(j.off - base) atIndex:1];
                [enc setBytes:&j.n_in length:4 atIndex:2];
                [enc setBytes:&j.n_out length:4 atIndex:3];
                [enc setBytes:&j.layout length:sizeof(RepackSpans) atIndex:4];
                const uint32_t blocks = j.n_in / j.layout.block_elems;
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
            // THIS WINDOW'S SOURCE IS DEAD: its bytes now live in the twin, and no later
            // command buffer references it -- each binds only its own window since cf4421b.
            // Release it so the page cache does not keep a second copy of the whole model
            // beside the twins (measured +9588 MiB of file-backed pages across a load).
            //
            // Measured as a NULL before cf4421b and reverted (52058ef); that measurement was
            // taken when every command buffer bound the WHOLE segment, so the next window
            // faulted the mapping straight back in. The premise changed with the bind.
            //
            // Clean, read-only, file-backed: a stray read re-faults from the file.
            uint8_t * dead = (uint8_t *)[seg.buf contents] + (base - seg.base);
            if (madvise(dead, (size_t)(top - base), MADV_DONTNEED) != 0) {
                NSLog(@"imparo metal: could not release a repacked source window (%llu MiB): "
                      @"%s", (unsigned long long)((top - base) >> 20), strerror(errno));
            } else {
                dropped += top - base;
            }
            for (size_t q = k; q <= e; ++q) { applied[todo[q]] = 1; }
            planned += (uint32_t)(e + 1 - k);
            rebuilt.push_back({cur, win_end - cur, base, top - base, seg.tier, seg.layer, twin});
            rset_add(twin);
            swapped += win_end - cur;
            windows += 1;
            cur = win_end;
            k = e + 1;
          }
        }
        // The source wrapper's last reference goes here: every byte of it now lives in a
        // twin, so the file pages fall out of the page cache on their own.
        rset_remove(seg.buf);
    }
    if (src_fd >= 0) { close(src_fd); }
    staging = nil;   // a transfer buffer, released as soon as its job is done
    g_wsegs.swap(rebuilt);
    NSLog(@"imparo metal: repack at load: %u tensors -> tile-major in %u windows of at most "
          @"%llu MiB (%llu MiB moved, %llu MiB pread, %llu MiB mapped, %llu MiB of source "
          @"released)",
          planned, windows, (unsigned long long)(cap >> 20),
          (unsigned long long)(swapped >> 20), (unsigned long long)(read_bytes >> 20),
          (unsigned long long)(map_bytes >> 20), (unsigned long long)(dropped >> 20));
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
