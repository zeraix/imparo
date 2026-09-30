#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint QK4_0 = 32u;

// Defined here rather than beside its own kernel: the prefill GEMM's epilogue uses it,
// and Metal has no forward declarations across the single translation unit.
inline float imparo_gelu_f(float t) {
    const float a = 0.79788456f * (t + 0.044715f * t * t * t);
    return 0.5f * t * (1.0f + tanh(clamp(a, -15.0f, 15.0f)));
}

inline float imparo_silu_f(float t) { return t / (1.0f + exp(-t)); }

// Which gated activation a fused epilogue applies. The WIRE VALUES are what the host
// binds at buffer 13, so they are fixed; they must match `Epilogue` in imparo-backend.
//
// It used to be a bool -- "epilogue on" meaning GELU, because gemma4 was the only model.
// A SwiGLU model that turned it on would have got GELU and plausible wrong numbers, and
// the knob table carried a comment warning not to.
constant uint EPI_NONE = 0u;
constant uint EPI_GELU = 1u;
constant uint EPI_SILU = 2u;

// WHICH activation, as a FUNCTION CONSTANT rather than a kernel argument.
//
// It was a runtime `uint kind` for one build. That alone -- with the value still GELU --
// moved gemma4's n=16 logits from 25.582184 to 25.582018 and held there across four
// fresh processes, and putting the branch back at `return imparo_gelu_f(t)` restored the
// old value exactly. A branch in the epilogue changes what the compiler does to the
// surrounding code; this is the same observer effect a phase probe once cost 2% of the
// kernel it was measuring.
//
// Specialised at pipeline build, so the ternary folds away and each pipeline carries
// exactly the arithmetic it had before this existed. The default is GELU, so a pipeline
// built without setting constant 11 is byte-for-byte the pre-feature kernel.
//
// One value per PROCESS, which is what a model is: gemma4 is GELU in every layer and
// LFM2 is SiLU in every layer. A file that mixed them per layer would need pipelines of
// both kinds and a dispatch-time choice.

// SCHEDULING FENCES INSIDE THE MMA LOOP, which llama.cpp's mul_mm has and this kernel
// does not. Between the weight loads, the activation loads and the multiplies it places
// `simdgroup_barrier(mem_flags::mem_none)` -- no memory ordering at all, purely a fence
// the compiler may not schedule across.
//
// Worth testing precisely because it runs the OTHER WAY from everything else tried here.
// Seven attempts to give this kernel less to do have measured slower; if what limits it
// is how the compiler orders the loads against the multiplies rather than how many there
// are, then constraining that order is the lever and removing work is not.
// DEFAULT TRUE, and measured: +5.1% at the short leg, +3.1% at depth. The constant stays
// so the negative can be reproduced, but the shipping pipeline has the fences.
// The same question asked of the Q4 rt_gemm, which has NO simdgroup barrier today and
// gets its ordering from an explicit prefetch pipeline instead. Default FALSE: that
// kernel is the one that already beats upstream, so it does not change until measured.
// And the same question for PREFILL ATTENTION, which is what the deep leg's remaining
// deficit scales with. Its score loop already prefetches one d-step ahead -- the same
// shape as the Q4 rt_gemm, which has no fence -- so the prior is that it does not want
// one. Default false; the lever exists so that is measured rather than assumed.
// PHASE PROBE for prefill attention, the shape that worked on the Q8 GEMM:
//   bit 0  no score MMA      bit 1  no softmax      bit 2  no P x V
// A function constant, so an unset build compiles the shipping kernel byte for byte.
// Attention is 15.4% of the deep prefill and runs 20.9% behind
// `kernel_flash_attn_ext`; neither KV bandwidth nor the query tile explains that, so the
// next question is which PHASE of the inner loop the time is in. Answers are wrong on
// purpose; only the time is read.
constant uint ATTN_SKIP_FC [[function_constant(17)]];
constant uint ATTN_SKIP = is_function_constant_defined(ATTN_SKIP_FC) ? ATTN_SKIP_FC : 0u;

// ROW LAYOUT (constant 25): the batch's rows are not one causal chain -- a tree verify, or a
// drafted block whose rows all see each other. Each row's position, the batch rows it sees
// and its ancestors come from a buffer of
// ROW_LAYOUT_WORDS u32 per row (imparo_backend::ROW_LAYOUT_WORDS):
//   [0] position  [1] depth  [2] sees batch rows 0..31  [3] rows 32..63
//   [4..11] the batch row 1..8 steps up the row's path (read only for steps <= depth)
// A kernel compiled without the constant keeps its causal code.
constant bool ROW_LAYOUT_FC [[function_constant(25)]];
constant bool ROW_LAYOUT = is_function_constant_defined(ROW_LAYOUT_FC) ? ROW_LAYOUT_FC : false;
// FA_FLOAT_Q (constant 26): the FA entry keeps Q in float for the scores instead of rounding it to
// half. Only the row-layout entry builds it, for a caller that asks for float Q.
constant bool FA_FLOAT_Q_FC [[function_constant(26)]];
constant bool FA_FLOAT_Q = is_function_constant_defined(FA_FLOAT_Q_FC) ? FA_FLOAT_Q_FC : false;
// ROW_SPLIT (constant 63), row-layout register-softmax op only: the key range is cut into
// consecutive slices, one a threadgroup (grid z), each writing its rows' unnormalised output, max
// and sum to a partials buffer that imparo_attention_rows_combine merges. A verify has one query
// tile, so without it the grid is one threadgroup a head walking every key.
constant bool ROW_SPLIT_FC [[function_constant(63)]];
constant bool ROW_SPLIT = is_function_constant_defined(ROW_SPLIT_FC) ? ROW_SPLIT_FC : false;
// ROW_HEADS (constant 64), same op: a threadgroup serves one KV head, its four simdgroups the four
// query heads that share it, 8 rows each -- each staged K/V block serves four heads, and no
// simdgroup computes rows past the batch at 8 rows. Only where a KV head has exactly four query
// heads.
constant bool ROW_HEADS_FC [[function_constant(64)]];
constant bool ROW_HEADS = is_function_constant_defined(ROW_HEADS_FC) ? ROW_HEADS_FC : false;
constant uint ROW_LAYOUT_WORDS = 12u;
// CO-BATCHED ROWS (constant 32): the dispatch's rows are different conversations
// (docs/continuous-batching.md), each at its own position with its own page table and recurrent
// state, so row t reads them from rows[t] (constant memory, one entry per row). A kernel
// compiled without the constant keeps its one-conversation code.
constant bool COB_ROWS_FC [[function_constant(32)]];
constant bool COB_ROWS = is_function_constant_defined(COB_ROWS_FC) ? COB_ROWS_FC : false;
struct CobRow {
    device const uint * pt;     // the conversation's page table for the dispatch's layer
    device float * state;       // its recurrent state, at the row's live history
    device float * state_out;   // where its advanced history lands
    uint pos;                   // the row's position
    uint pad;
};
// Every row sees the whole cache below the batch; a batch key is seen when its bit is set.
inline bool row_layout_sees(device const uint * layout, uint t, uint gp, uint start_pos) {
    if (gp < start_pos) { return true; }
    const uint j = gp - start_pos;
    if (j >= 64u) { return false; }
    const uint w = layout[t * ROW_LAYOUT_WORDS + 2u + (j >> 5)];
    return ((w >> (j & 31u)) & 1u) != 0u;
}

constant uint EPI_ACT_FC [[function_constant(11)]];
constant uint EPI_ACT = is_function_constant_defined(EPI_ACT_FC) ? EPI_ACT_FC : EPI_GELU;

inline float imparo_act_f(float t) {
    return EPI_ACT == EPI_SILU ? imparo_silu_f(t) : imparo_gelu_f(t);
}

constant uint Q4_0_BYTES = 18u;

// Q8_0: 32 values per block, an f16 scale then 32 SIGNED bytes. Element i is byte 2+i.
constant uint QK8_0 = 32u;
constant uint Q8_0_BYTES = 34u;

// DEQUANT FORM of the prefill GEMMs, st_gemm and rt_gemm (task #105): half(q) * d in HALF
// arithmetic -- one convert and one half multiply per element. It is bit-identical to the
// ported form half(float(q) * float(d)): q (or nibble - 8) is exact in half, the product of
// an int8 by a half is exact in float (19 significant bits), so both round the same exact
// product once. Measured 2026-09-02 on the original files: LFM2 +6.8% (455), E4B +2.3% (449)
// / +2.0% (16191) -- the conversion arithmetic had been 7% of the short leg.

// "this norm has no weight tensor". It used to be 0xFFFFFFFF, which collided with any
// real offset congruent to it mod 2^32 while the offset was 32 bits, and with exactly
// 4294967295 once it was widened. All-ones in 64 bits cannot be a real offset.
constant ulong IMPARO_NO_WEIGHT = ~0ul;
// The norm's threadgroup stage: on when the host sized the memory for it, and the offset
// (in floats) past the reduction partials where the staged row starts.
constant bool STAGED_FC [[function_constant(56)]];
constant bool STAGED = is_function_constant_defined(STAGED_FC) && STAGED_FC;
constant uint STAGE_OFF_FC [[function_constant(57)]];
constant uint STAGE_OFF = is_function_constant_defined(STAGE_OFF_FC) ? STAGE_OFF_FC : 0u;

// `x` is a float buffer and every n_in here is a multiple of 4, so it can be read as
// float4 -- one load instruction instead of four.
//
// Weights are the real cost. A Q4_0 block is 18 bytes, so its payload lands at 2 mod 4 and
// cannot be read as an aligned word: the byte-at-a-time version issued 18 load
// instructions for 18 bytes, the worst possible ratio. TWO blocks are 36 bytes, which IS
// 4-aligned and is exactly 9 uints, so a pair costs 9 loads instead of 36. The second
// block's payload is word-aligned within the pair; the first block's is offset by two
// bytes and is recovered with a shift-and-merge.
inline float4 unpack_lo(uchar4 packed) { return float4(packed & 0x0F) - 8.0f; }
inline float4 unpack_hi(uchar4 packed) { return float4(packed >> 4)   - 8.0f; }

inline float q4_0_dot_words(thread const uint * w, device const float4 * x4, uint base4) {
    float part = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        const uchar4 packed = as_type<uchar4>(w[g]);
        part += dot(unpack_lo(packed), x4[base4 + g])
              + dot(unpack_hi(packed), x4[base4 + g + 4u]);
    }
    return part;
}

// byte-at-a-time fallback for rows whose block count is odd
inline float q4_0_block_dot(device const uchar * blk, device const float4 * x4, uint base4) {
    float part = 0.0f;
    for (uint g = 0; g < 4; ++g) {
        const uint o = 2u + g * 4u;
        const uchar4 packed = uchar4(blk[o], blk[o+1], blk[o+2], blk[o+3]);
        part += dot(unpack_lo(packed), x4[base4 + g])
              + dot(unpack_hi(packed), x4[base4 + g + 4u]);
    }
    return part;
}

// LANES_PER_ROW lanes cooperate on one output row, and each lane carries TOKEN_TILE
// tokens at once.
//
// The token tile is the point. Dispatching one thread per (row, token) made a batch of 32
// read every weight row 32 times, so batched prefill amortised nothing on the GPU even
// though the CPU path already did this correctly. Reading a block once and using it for
// every token in the tile is what makes prefill scale.
//
// Five earlier tuning attempts were flat -- uchar4/float4 unpack, float4 activations,
// aligned uint pairs (slower), rows per simdgroup, and a 2..32 sweep of simdgroups per
// threadgroup -- so the kernel is neither ALU, issue, nor occupancy bound.
// Lanes cooperating on one output row. A FUNCTION CONSTANT, not a literal: it is a value
// the hardware decides, so it belongs in the measured search space rather than baked in.
// Pipelines are built for each candidate and selected at run time.
constant uint LANES_PER_ROW [[function_constant(0)]];
// Set to skip the matrix multiply-accumulates while keeping all staging. Timing a variant
// with the mma removed attributes the GEMM's cost between staging and arithmetic, instead
// of inferring it from fifteen geometry experiments.
constant bool SKIP_MMA [[function_constant(1)]];
// Output rows per thread at DECODE. One row per thread leaves a single chain of dependent
// weight loads in flight, which is why prefetching this kernel measured worse: it added
// registers without adding independent work. NR0 rows give NR0 chains for free, and share
// one activation read. llama.cpp's mul_vec_q_n_f32_impl uses 4.
constant uint NR0 [[function_constant(2)]];
// Tokens per lane in the Q4 matmat. Injected as a preprocessor macro by the host
// (imparo_metal.mm: Q4_TOKEN_TILE) because the DISPATCH GRID divides n_tok by the same
// number -- they were two independent 4s, and a change to either alone would have sized
// the grid for a tile the kernel does not use. Compile-time, so the unrolled body and the
// `tiles == 1` decode branch fold exactly as before.
//
// MEASURED, on the Q8 kernel that does the same thing with a tunable tile (LFM2.5-2.6B
// Q8_0, narrow batch, us per pass): 1 -> 1286, 2 -> 833, 4 -> 799, 8 -> 1534. A clean
// interior minimum at 4 with both sides turning over, and +92% at 8 where the registers
// spill. That is why this one is a constant rather than a knob: the analogous kernel
// answers the question, and four times the pipelines would rediscover it.
#ifndef Q4_TOKEN_TILE
#define Q4_TOKEN_TILE 4u
#endif
constant uint TOKEN_TILE    = Q4_TOKEN_TILE;

// DECODE ROWS: the decode GEMV over MV_TOKENS independent rows at once, the rows of different
// conversations in one co-batched step (imparo_q4_0_matmat, imparo_q8_0_gemv). Every row keeps
// the one-row kernel's K split, its accumulation and its reduction order, so a row's bits equal
// what the one-row kernel gives it; only the weight loads are shared. A row's K/V bytes must not
// depend on who else is in its step (docs/kv-identity-grid.md), and this is what keeps them the
// same. A compile-time count: undefined is 1, the one-row kernel. Rows past n_tok read the last
// live row and store nothing. The arrays are sized by the largest count (a function constant
// cannot size an array); the loops run to the specialized count, so a one-row pipeline's unused
// slots are dead code.
constant uint MV_TOKENS_FC [[function_constant(28)]];
constant uint MV_TOKENS = is_function_constant_defined(MV_TOKENS_FC) ? MV_TOKENS_FC : 1u;
#define MV_MAX 8u
// Lane groups of a simdgroup that share one row set and split its tokens (imparo_q4_0_matmat's
// decode rows): 1 gives every lane group its own rows and all the tokens.
constant uint MV_SPLIT_FC [[function_constant(29)]];
constant uint MV_SPLIT = is_function_constant_defined(MV_SPLIT_FC) ? MV_SPLIT_FC : 1u;

// BRICK: Q4_0 ROW GROUP PARTIAL. This lane's blocks (sub, sub + LANES_PER_ROW, ...) of NR0 rows
// (row_off[i] = byte offset of row i from `w`), one activation read shared by all NR0 rows;
// acc[i] accumulates block-by-block in the decode GEMV's order: part = sum over the 4 nibble
// groups of dot(lo, xa) + dot(hi, xb), then acc += part * d. The decode GEMV's fast path and
// every mega row phase are this one body (NR0 is the pipeline's function constant: the GEMV's
// tuned row count, 1 in the mega block), so a row's bits are the same wherever it is computed.
// XS is the activation row's address space: device const float4 * or threadgroup const float4 *.
// BRICK: LANE-GROUP SUM. The LANES_PER_ROW lanes that share a row fold their block partials
// into the group's first lane (sub == 0) with the decode GEMV's shuffle tree; the other lanes
// hold garbage afterwards, as they always did.
inline float lane_group_sum(float acc) {
    #pragma unroll
    for (uint off = LANES_PER_ROW / 2u; off > 0u; off >>= 1u) { acc += simd_shuffle_down(acc, off); }
    return acc;
}

template <typename XS>
inline void q4_rows_partial(device const uchar * w, thread const ulong * row_off, XS xs,
                            uint blocks, uint sub, thread float * acc) {
    for (uint b = sub; b < blocks; b += LANES_PER_ROW) {
        const uint base4 = b * (QK4_0 / 4u);
        float d[8], part[8];
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            device const uchar * blk = w + row_off[i] + b * Q4_0_BYTES;
            d[i] = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
            part[i] = 0.0f;
        }
        #pragma unroll
        for (uint g = 0; g < 4; ++g) {
            const float4 xa = xs[base4 + g];
            const float4 xb = xs[base4 + g + 4u];
            #pragma unroll
            for (uint i = 0; i < NR0; ++i) {
                // One 32-bit load per four values; the payload is 2-byte aligned inside the
                // block, so the address is cast rather than indexed through a uchar4 pointer.
                const uchar4 pk = *(device const uchar4 *)(w + row_off[i] + b * Q4_0_BYTES + 2u + g * 4u);
                part[i] += dot(unpack_lo(pk), xa) + dot(unpack_hi(pk), xb);
            }
        }
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) { acc[i] += part[i] * d[i]; }
    }
}

kernel void imparo_q4_0_matmat(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in    [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok   [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]], constant uint & rows_per_sg [[buffer(8)]],
    constant uint & epilogue [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    if (MV_TOKENS > 1u) {
        // DECODE ROWS. Each row keeps what fixes its bits in the one-row path below: its blocks
        // go to the LANES_PER_ROW lanes of one lane group by b % LANES_PER_ROW, each lane sums
        // them in order as q4_rows_partial does (part over the four nibble groups, then
        // acc += part * d), and lane_group_sum folds the group with the same shuffle tree.
        // Which lane group holds a row does not enter, so the MV_SPLIT lane groups of a row set
        // share its weight loads and split the tokens; every lane serves NR0 rows, so one
        // activation load feeds NR0 multiplies. Rows past n_out and tokens past n_tok read
        // clamped operands and store nothing. The grid is n_out / (nsg * groups * NR0) wide.
        const uint split = MV_SPLIT;
        const uint tpl   = MV_TOKENS / split;                  // tokens per lane
        const uint sub   = lane % LANES_PER_ROW;
        const uint slot  = lane / LANES_PER_ROW;
        const uint sets  = (32u / LANES_PER_ROW) / split;      // row sets per simdgroup
        const uint r     = ((tgid.x * nsg + sgid) * sets + slot / split) * NR0;
        const uint tok0  = slot % split;
        if (r >= n_out) { return; }
        const uint blocks = n_in / QK4_0;
        const uint last = n_tok - 1u;
        ulong row_off[8];
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            row_off[i] = w_offset + (ulong)min(r + i, n_out - 1u) * blocks * Q4_0_BYTES;
        }
        device const float4 * xs[MV_MAX];
        #pragma unroll
        for (uint tt = 0; tt < tpl; ++tt) {
            const uint t = min(tok0 + tt * split, last);
            xs[tt] = (device const float4 *)(x + (ulong)(src_row + t) * n_in);
        }
        float acc[8][MV_MAX];
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            #pragma unroll
            for (uint tt = 0; tt < tpl; ++tt) { acc[i][tt] = 0.0f; }
        }
        for (uint b = sub; b < blocks; b += LANES_PER_ROW) {
            const uint base4 = b * (QK4_0 / 4u);
            float d[8], part[8][MV_MAX];
            #pragma unroll
            for (uint i = 0; i < NR0; ++i) {
                device const uchar * blk = weights + row_off[i] + b * Q4_0_BYTES;
                d[i] = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
                #pragma unroll
                for (uint tt = 0; tt < tpl; ++tt) { part[i][tt] = 0.0f; }
            }
            #pragma unroll
            for (uint g = 0; g < 4; ++g) {
                uchar4 pk[8];
                #pragma unroll
                for (uint i = 0; i < NR0; ++i) {
                    pk[i] = *(device const uchar4 *)(weights + row_off[i] + b * Q4_0_BYTES + 2u + g * 4u);
                }
                #pragma unroll
                for (uint tt = 0; tt < tpl; ++tt) {
                    const float4 xa = xs[tt][base4 + g];
                    const float4 xb = xs[tt][base4 + g + 4u];
                    #pragma unroll
                    for (uint i = 0; i < NR0; ++i) {
                        part[i][tt] += dot(unpack_lo(pk[i]), xa) + dot(unpack_hi(pk[i]), xb);
                    }
                }
            }
            {
                // The one-row brick rounds part * d before the add; this body, with more rows
                // in flight, was contracted to one fma, which rounds once and moves the bits of
                // every lane that sums two or more blocks.
#pragma clang fp contract(off)
                #pragma unroll
                for (uint i = 0; i < NR0; ++i) {
                    #pragma unroll
                    for (uint tt = 0; tt < tpl; ++tt) { acc[i][tt] += part[i][tt] * d[i]; }
                }
            }
        }
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            #pragma unroll
            for (uint tt = 0; tt < tpl; ++tt) { acc[i][tt] = lane_group_sum(acc[i][tt]); }
        }
        if (sub == 0u) {
            #pragma unroll
            for (uint i = 0; i < NR0; ++i) {
                #pragma unroll
                for (uint tt = 0; tt < tpl; ++tt) {
                    const uint t = tok0 + tt * split;
                    if (r + i >= n_out || t >= n_tok) { continue; }
                    device float * slot_p = y + (ulong)t * n_out + r + i;
                    *slot_p = epilogue ? (imparo_act_f(*slot_p) * acc[i][tt]) : acc[i][tt];
                }
            }
        }
        return;
    }
    const uint rows_per_simd = 32u / LANES_PER_ROW;
    const uint sub  = lane % LANES_PER_ROW;
    const uint slot = lane / LANES_PER_ROW;
    // Each lane group owns NR0 CONSECUTIVE rows. The batched path below is always
    // dispatched with the NR0 == 1 pipeline, so its row mapping is unchanged.
    const uint r  = ((tgid.x * nsg + sgid) * rows_per_simd + slot) * NR0;
    const uint t0 = tgid.y * TOKEN_TILE;
    if (r >= n_out || t0 >= n_tok) { return; }
    const uint tiles = min(TOKEN_TILE, n_tok - t0);

    const uint blocks = n_in / QK4_0;
    device const uchar * row = weights + w_offset + (ulong)r * blocks * Q4_0_BYTES;

    // DECODE FAST PATH. With a tile of 4 and n_tok == 1 the unrolled body still evaluates
    // a1..a3 against a clamped token 0 -- four times the work for one token. That cost
    // decode 37.6 -> 15.8 tok/s when the tile was introduced, and again when a line-based
    // edit dropped this branch. If decode regresses by ~2.4x, look here first.
    if (tiles == 1u) {
        device const float4 * xs =
            (device const float4 *)(x + (ulong)(src_row + t0) * n_in);
        // NO PREFETCHING HERE, deliberately -- and the reason is NR0, below.
        //
        // Prefetching is what took the prefill GEMM from 4.41 to 4.71 TFLOPS. Tried here it
        // measured decode 38.2 -> 34.7 tok/s, because with one output row per thread there
        // is a single chain of dependent loads: a prefetch adds registers without adding
        // independent work. NR0 rows per thread give NR0 chains instead, which is how
        // llama.cpp covers the same latency (mul_vec_q_n_f32_impl, NR0 = 4).
        //
        // Measured, so neither of these is guesswork:
        //   read-only probe, this exact addressing   121 GB/s (ffn_down) / 137 (lm_head)
        //   this kernel with the arithmetic          109 GB/s            / 127
        //   pure streaming read, no structure        147 GB/s
        // The 18-byte block stride is NOT the cost: the same bytes read as contiguous
        // uint4 measured no faster, so repacking the weights would buy nothing.
        //
        // One 32-bit load per four values, not four byte loads. The payload is only 2-byte
        // aligned inside the block, so the address is cast directly rather than indexed
        // through a uchar4 pointer -- indexed access assumes a vector-aligned base.
        // NR0 accumulators, NR0 rows, ONE activation read shared by all of them: the
        // q4_rows_partial brick.
        float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        ulong row_off[8];
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            // Clamp rather than branch: a tail thread reads a valid row and throws the
            // result away at the store, which keeps the inner loop uniform.
            const uint rr = min(r + i, n_out - 1u);
            row_off[i] = w_offset + (ulong)rr * blocks * Q4_0_BYTES;
        }
        q4_rows_partial(weights, row_off, xs, blocks, sub, acc);
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) { acc[i] = lane_group_sum(acc[i]); }
        if (sub == 0u) {
            #pragma unroll
            for (uint i = 0; i < NR0; ++i) {
                if (r + i >= n_out) { continue; }
                // Same gated-activation epilogue as the batched kernel, so decode does not
                // need the separate gelu_mul pass either.
                device float * slot_p = y + (ulong)t0 * n_out + r + i;
                *slot_p = epilogue ? (imparo_act_f(*slot_p) * acc[i]) : acc[i];
            }
        }
        return;
    }

    // Scalars, not an array, and ONE base pointer with index arithmetic.
    //
    // Weight traffic per token is exactly 1/TOKEN_TILE, so the tile is the only lever on
    // prefill. It stalled at 4 because eight accumulators PLUS eight activation pointers
    // spilled to thread-local memory (24.8 ms/token against 19.3 at four). Holding a single
    // base pointer and stepping it by n_in halves the register demand -- and a tile of 8
    // STILL spilled (24.0 ms/token), so four is the register budget on this part.
    const uint last = tiles - 1u;
    const uint stride4 = n_in / 4u;
    device const float4 * xb = (device const float4 *)(x + (ulong)(src_row + t0) * n_in);
    const uint o1 = min(1u, last) * stride4, o2 = min(2u, last) * stride4;
    const uint o3 = min(3u, last) * stride4;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = sub; b < blocks; b += LANES_PER_ROW) {
        device const uchar * blk = row + b * Q4_0_BYTES;
        const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
        const uint base4 = b * (QK4_0 / 4u);
        #pragma unroll
        for (uint g = 0; g < 4; ++g) {
            const uint o = 2u + g * 4u;
            const uchar4 packed = uchar4(blk[o], blk[o+1], blk[o+2], blk[o+3]);
            const float4 lo = unpack_lo(packed) * d;
            const float4 hi = unpack_hi(packed) * d;
            const uint gl = base4 + g, gh = base4 + g + 4u;
            a0 += dot(lo, xb[gl])      + dot(hi, xb[gh]);
            a1 += dot(lo, xb[o1 + gl]) + dot(hi, xb[o1 + gh]);
            a2 += dot(lo, xb[o2 + gl]) + dot(hi, xb[o2 + gh]);
            a3 += dot(lo, xb[o3 + gl]) + dot(hi, xb[o3 + gh]);
        }
    }
    #pragma unroll
    for (uint off = LANES_PER_ROW / 2u; off > 0u; off >>= 1u) {
        a0 += simd_shuffle_down(a0, off); a1 += simd_shuffle_down(a1, off);
        a2 += simd_shuffle_down(a2, off); a3 += simd_shuffle_down(a3, off);
    }
    if (sub == 0u) {
        device float * yo = y + (ulong)t0 * n_out + r;
        if (epilogue) {
            yo[0] = imparo_act_f(yo[0]) * a0;
            if (tiles > 1u) { yo[(ulong)1 * n_out] = imparo_act_f(yo[(ulong)1 * n_out]) * a1; }
            if (tiles > 2u) { yo[(ulong)2 * n_out] = imparo_act_f(yo[(ulong)2 * n_out]) * a2; }
            if (tiles > 3u) { yo[(ulong)3 * n_out] = imparo_act_f(yo[(ulong)3 * n_out]) * a3; }
        } else {
            yo[0] = a0;
            if (tiles > 1u) { yo[(ulong)1 * n_out] = a1; }
            if (tiles > 2u) { yo[(ulong)2 * n_out] = a2; }
            if (tiles > 3u) { yo[(ulong)3 * n_out] = a3; }
        }
    }
}



// PREFILL matmul on SIMDGROUP MATRIX hardware.
//
// Seven scalar restructurings lost to the simple token tile, and the arithmetic says why:
// weight traffic per token is 1/TOKEN_TILE, and the tile cannot pass 4 before registers
// spill. An 8x8 accumulator has to live in registers, which is exactly what
// simdgroup_float8x8 provides.
//
// Both operand tiles are staged once per K chunk, so each weight row is read ONCE for the
// whole batch. The weight tile is staged TRANSPOSED so the product comes out token-major,
// matching the activation and output layouts without a separate transpose.
//
//   C[t][r] = sum_k X[t][k] * Wt[k][r]
//
// The two traffics, exactly:
//
//   activations = n_in * n_tok * 4 * (n_out / SG_ROWS)      only SG_ROWS reduces this
//   weights     = n_out * n_in * 0.5625 * (n_tok / SG_TOKENS)
//
// For one ffn_gate at batch 64 that is 210 MB of activations against 29 MB of weights --
// activations dominate 7:1, so SG_ROWS is the lever that matters. Two earlier attempts to
// widen it failed for reasons that were NOT the row count: SG_ROWS 64 with SG_K 64 needed
// 27.6 KB of threadgroup memory (74.5 tok/s), and carrying four accumulators per simdgroup
// instead of two cost 47% (83.5 tok/s).
//
// The way to widen rows without paying either price is MORE SIMDGROUPS, not more
// accumulators each: 64 rows x 32 tokens is 2048 outputs, which over 16 simdgroups is still
// two 8x8 tiles apiece, and SG_K stays 32 so threadgroup memory is 14.3 KB.
// SG_ROWS and SG_TOKENS are INJECTED by the host (imparo_metal.mm: SG_ROWS/SG_TOKENS),
// because the DISPATCH GRID is n_out/SG_ROWS by n_tok/SG_TOKENS and the thread count is
// SG_ROWS*SG_TOKENS/128 simdgroups. Those were written on the host as 63/64, 31/32 and
// 512, three independent copies of numbers that live here: changing this tile alone would
// have sized the grid for a shape the kernel does not use, and the kernel would have read
// past its tile rather than failed.
//
// The injected macro carries the VALUE; the names stay `constant uint` as before. Defining
// SG_ROWS as a bare macro instead made `min(SG_ROWS, n_out - r0)` ambiguous between min(int,
// int) and min(uint, uint), because the host injects `64` and not `64u` -- the library
// stopped compiling, every forward failed in a second, and det_gate caught it.
#ifndef IMPARO_SG_ROWS
#define IMPARO_SG_ROWS 64
#endif
#ifndef IMPARO_SG_TOKENS
#define IMPARO_SG_TOKENS 32
#endif
constant uint SG_ROWS   = IMPARO_SG_ROWS;
constant uint SG_TOKENS = IMPARO_SG_TOKENS;
// SG_TOKENS sets how many times each weight is DEQUANTISED: n_tok / SG_TOKENS. At 64 a
// batch of 64 dequantises every weight ONCE instead of twice. MEASURED SLOWER TWICE:
// 232-241 -> 196-200 when the activation tile still sat in threadgroup memory, and again
// 241-247 -> 213-219 after that tile was removed and the widening cost no threadgroup
// memory at all. The occupancy loss from 1024 threads is what beats it, not the memory.
//
// That is the whole pattern of this kernel, across sixteen variants: the changes that won
// removed REDUNDANT WORK at a fixed tile shape; every change that WIDENED a tile lost, no
// matter which cost it removed.
constant uint SG_K      = 32u;
//
// SG_K = 64 halves the barrier count and was tried twice, both times slower. Paired with
// SG_ROWS 64 it needs 27.6 KB of threadgroup memory (74.5 tok/s). Paired with SG_ROWS 32 it
// fits in 19.5 KB and needs only ONE accumulator per simdgroup -- both binding constraints
// slack -- and still measured 208-212 against 232-241, because halving SG_ROWS doubles the
// activation re-reads. Fewer barriers is not worth a narrower row tile here.
constant uint SG_XS     = SG_K + 8u;
// The weight tile is stored PRE-TRANSPOSED as [k][row]. Storing it row-major and letting
// simdgroup_load transpose in hardware makes the staging writes contiguous and shrinks
// threadgroup memory, so it looked strictly better -- but MEASURED it is slower
// (240 -> 220 tok/s): the transposing load costs more than the scattered writes save.
constant uint SG_WS     = SG_ROWS + 8u;
constant uint SG_OUT    = SG_ROWS + 8u;   // write-back stride, [token][row]
constant uint SG_TILES  = (SG_ROWS / 8u) * (SG_TOKENS / 8u);

// Convert an activation range to half so the GEMM can feed the matrix units half operands.
//
// The accumulator stays float, so accumulation precision over thousands of terms is
// unchanged; only the operands round, which is what ggml's mul_mm does. Converting costs
// one pass over b x n_in elements against a matmul of b x n_in x n_out, so it is roughly
// three orders of magnitude cheaper than the work it speeds up.
kernel void imparo_cvt_f32_f16(
    device const float * src [[buffer(0)]], device half * dst [[buffer(1)]],
    constant uint & n [[buffer(2)]], constant uint & off [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    dst[off + gid] = half(src[off + gid]);
}

// ---------------------------------------------------------------------------------------
// THE GATED PAIR: one GEMM over gate AND up, closed by act(G) * U in the write-back.
//
// A gated FFN runs two projections over the SAME activation tile and then multiplies
// them elementwise. As two dispatches that costs every row-block threadgroup a second
// read of the activation tile, a float G buffer written by the first dispatch and read
// back by the second, and the dispatch itself. Here the two weight tensors are read as
// ONE virtual matrix of 2 * n_out rows whose 8-row tiles alternate gate (even tile) and
// up (odd tile); nothing is copied or re-laid out in memory -- the interleave is an
// address mapping, so the weights stay zero-copy from the file. Any row tile of 16 or
// more rows then holds matched G/U pairs at tile positions (2m, 2m+1), whatever shape the
// tuner picks, and the epilogue reads both from the tile it already staged for the
// write-back. Per output the K walk is the one the split GEMM did, and act(G) * U is
// computed on the same float bits, so the answer is bit-identical to the two-dispatch
// form.
//
// Tile row v of the virtual matrix comes from tensor (v / 8) % 2, row (v / 16) * 8 + v % 8.
template<bool GATED> inline bool gated_is_up(uint v) { return GATED && (((v >> 3) & 1u) != 0u); }
template<bool GATED> inline uint gated_src_row(uint v) { return GATED ? (((v >> 4) << 3) | (v & 7u)) : v; }
// Where output column c (0 <= c < rows / 2) of a gated tile finds its gate slot; the up
// slot is 8 further on.
inline uint gated_gate_slot(uint c) { return ((c >> 3) << 4) | (c & 7u); }

// ---------------------------------------------------------------------------------------
// Register-tiled prefill GEMM.
//
// The other prefill kernel computes two INDEPENDENT 8x8 output tiles per simdgroup, so
// each multiply needs a fresh load of both operands: 4 simdgroup_loads feed 2 multiplies.
// Here each simdgroup holds an NA x NB grid of accumulators, so NA + NB loads feed NA * NB
// multiplies -- 6 loads for 8 multiplies at 2x4. Operand loads per multiply fall from 2.0
// to 0.75.
//
// An earlier note in this file claimed the live-accumulator count was "a hard performance
// cliff". That was wrong, and it closed off this direction for fifteen experiments. What
// that measurement actually changed was a RUNTIME BRANCH that blocked unrolling; the
// accumulator count was along for the ride. Every loop below is bounded by a template
// parameter, so the whole tile unrolls into registers.
//
// The shape is a TEMPLATE parameter rather than a function constant because MSL will not
// accept a function constant as an array size, and `mc` must be exactly as large as the
// shape uses -- a padded worst-case array would hand every small shape the register
// pressure of the largest one. Each shape is instantiated as its own named kernel and the
// host picks between them at run time, so measuring a shape still costs no rebuild.
// ---- THE TILE-MAJOR DECODE BRICKS (task Qwen3.8-27B-1b) --------------------------------
//
// Nine formats in one file (an unsloth UD mix) is not nine readers. Every tile-major rule
// makes the ADDRESS uniform -- a row's scale bytes at the head of its 8-row unit, its
// payload after them, both one aligned load -- so all that differs is the ARITHMETIC, and
// that is one of three:
//
//     affine      value = q * scale + min          Q4_1 Q5_1 Q2_K Q4_K Q5_K
//     symmetric   value = (q - bias) * scale       Q4_0 Q5_0 Q8_0 Q3_K Q6_K
//     codebook    value = table[q] * scale         IQ4_NL IQ4_XS IQ3_S
//
// WFMT is a FUNCTION CONSTANT, so the format is resolved when the pipeline is built and the
// arms a model does not use are dead-stripped: no runtime branch, and the row-major kernels
// stay byte for byte what they were. Same mechanism as Q8_TM (constant 15), one pipeline
// per layout per variant.
//
// Offsets below MIRROR imparo_gguf::weights::TM_RULES. The scale span of each format is
// what the rule moves to the unit head, in span order, and the payload is the complement in
// SOURCE order -- so Q5_K's payload opens with qh (source byte 16) and Q6_K's with ql
// (source byte 0), even though Q6_K's scales live at source byte 192.
constant uint TM_UNIT_ROWS = 8u;   // rows per tile-major unit (TmRule::unit_rows)
constant uint WFMT_FC [[function_constant(21)]];
// A weight of a brick-readable format that kept the ROW-MAJOR layout: a row-gathered
// tensor by rule (token_embd), and the lm head when it is tied to it. Same decode, the
// addresses differ -- which is the whole point of the family.
constant bool WFMT_ROW_FC [[function_constant(22)]];
constant bool WFMT_PROBE_FC [[function_constant(23)]];
constant uint WFMT = is_function_constant_defined(WFMT_FC) ? WFMT_FC : 0u;
constant bool WFMT_ROW = is_function_constant_defined(WFMT_ROW_FC) ? WFMT_ROW_FC : false;
constant bool WFMT_PROBE = is_function_constant_defined(WFMT_PROBE_FC) ? WFMT_PROBE_FC : false;

// SCHEDULING FENCES IN THE rt_gemm MMA LOOP. `simdgroup_barrier(mem_flags::mem_none)`
// orders nothing in memory; it is a point the compiler may not schedule instructions
// across. The Q8 GEMM gained +5.1% short / +3.1% deep from exactly this arrangement
// (Q8_MMA_FENCE, constant 14), and it had NO pipeline of its own at the time.
//
// This kernel does: it rotates A and B prefetch registers by hand, so the data
// dependency through the rotation already imposes an order. Whether a second constraint
// helps or fights that pipeline is a measurement, and this constant is how it is made.
//
// MEASURED 2026-09-09 AND IT IS A TIE, so the default stays FALSE. Qwen3.8-27B
// UD-Q4_K_S, 512-token prefill, both arms warmed first, rotated OFF/ON/ON/OFF, ms per
// chunk (second repeat of each run):
//     off  4988.1  4992.8      median 4990.5
//     on   4989.6  4990.1      median 4989.9      -0.01%
// The logits are identical in all four runs, which a scheduling fence cannot change --
// it orders instructions, not arithmetic. The pipelines were confirmed to differ: the
// engine logs mma_fence=1 on each tile-major rt pipeline it builds, and the ON arm's
// pipeline builds took ~9 ms against ~1 ms.
//
// So the difference from the Q8 GEMM's +5.1% is the pipeline, not the device: that
// kernel had no ordering of its own and this one already has one.
constant bool RT_MMA_FENCE_FC [[function_constant(16)]];
constant bool RT_MMA_FENCE = is_function_constant_defined(RT_MMA_FENCE_FC)
                           ? RT_MMA_FENCE_FC : false;
// rt_gemm's straight store for an n_out that is not a multiple of 8, where the last 8x8 block
// straddles n_out (see the store). The host builds this variant only for such an n_out.
constant bool RT_EDGE8_FC [[function_constant(30)]];
constant bool RT_EDGE8 = is_function_constant_defined(RT_EDGE8_FC) ? RT_EDGE8_FC : false;
// THE RESIDUAL ADD IN THE STORE (a pipeline variant, never a runtime branch: a runtime epilogue
// value once moved gemma4's logits without running). The plain store writes y = acc; this one
// writes y = y + acc, so a mixer's output projection adds into the residual stream X instead of
// writing O for the next norm to add. Same float sum as that norm's `resid + O`.
constant bool RT_RESID_FC [[function_constant(60)]];
constant bool RT_RESID = is_function_constant_defined(RT_RESID_FC) && RT_RESID_FC;
inline void rt_resid_add(thread simdgroup_float8x8 & m, device const float * yp, uint ld) {
    simdgroup_float8x8 old;
    simdgroup_load(old, yp, ld);
    const float2 a = reinterpret_cast<thread float2 &>(old.thread_elements());
    const float2 b = reinterpret_cast<thread float2 &>(m.thread_elements());
    reinterpret_cast<thread float2 &>(m.thread_elements()) = float2(a.x + b.x, a.y + b.y);
}
inline bool getenv_probe_const() { return WFMT_PROBE; }
// Codes follow the ggml type ids so a reader can recover the source at a glance.
constant uint WF_ROWMAJOR = 0u;
constant uint WF_Q4_K = 12u, WF_Q5_K = 13u, WF_Q6_K = 14u;
constant uint WF_Q3_K = 11u, WF_IQ4_NL = 20u, WF_IQ4_XS = 23u;
constant uint WF_IQ3_S = 21u;
constant uint WF_IQ2_XS = 17u, WF_IQ3_XXS = 18u, WF_IQ2_S = 22u, WF_IQ2_XXS = 16u;
constant uint WF_Q2_K = 10u;
constant uint WF_Q4_1 = 3u, WF_Q5_0 = 6u, WF_Q5_1 = 7u, WF_IQ1_S = 19u, WF_IQ1_M = 29u;
// Q4_0 has its own dense kernels (E4B's); this arm serves only the ROUTED matmuls, which
// the host stamps with it through moe_wfmt_for -- wfmt_for never returns it, so no dense
// pipeline changes format by its presence here.
constant uint WF_Q4_0 = 2u;

// The sign table IQ2_XS and IQ3_XXS share: a 7-bit index whose EIGHTH sign is the parity
// of the other seven. llama.cpp writes it out as ksigns_iq2xs[128]; it is one expression,
// and a copied table is 128 chances to mistype a number that reads as a plausible weight.
inline uchar ksign_iq(uchar i) { return i | (uchar)((popcount((uint)i) & 1u) << 7); }

// IQ4_NL / IQ4_XS quantise to sixteen NON-LINEAR levels: a 4-bit field is a position in
// this table, so no arithmetic reproduces the value -- the table is part of the format.
constant char KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
// The same levels as floats, for the rows matmul's fetch (tm_run8): float((int)level) is exact,
// so a value decoded through either table has the same bits (tests/brick_rows.rs).
constant float KVALUES_IQ4NL_F[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f
};
// Both levels of one IQ4 byte, low nibble then high nibble, for the rows matmul's fetch: one
// lookup per byte instead of one per nibble. Generated from the table above.
constant float2 KVALUES_IQ4NL_PAIR[256] = {
    float2(-127.0f, -127.0f), float2(-104.0f, -127.0f), float2(-83.0f, -127.0f), float2(-65.0f, -127.0f),
    float2(-49.0f, -127.0f), float2(-35.0f, -127.0f), float2(-22.0f, -127.0f), float2(-10.0f, -127.0f),
    float2(1.0f, -127.0f), float2(13.0f, -127.0f), float2(25.0f, -127.0f), float2(38.0f, -127.0f),
    float2(53.0f, -127.0f), float2(69.0f, -127.0f), float2(89.0f, -127.0f), float2(113.0f, -127.0f),
    float2(-127.0f, -104.0f), float2(-104.0f, -104.0f), float2(-83.0f, -104.0f), float2(-65.0f, -104.0f),
    float2(-49.0f, -104.0f), float2(-35.0f, -104.0f), float2(-22.0f, -104.0f), float2(-10.0f, -104.0f),
    float2(1.0f, -104.0f), float2(13.0f, -104.0f), float2(25.0f, -104.0f), float2(38.0f, -104.0f),
    float2(53.0f, -104.0f), float2(69.0f, -104.0f), float2(89.0f, -104.0f), float2(113.0f, -104.0f),
    float2(-127.0f, -83.0f), float2(-104.0f, -83.0f), float2(-83.0f, -83.0f), float2(-65.0f, -83.0f),
    float2(-49.0f, -83.0f), float2(-35.0f, -83.0f), float2(-22.0f, -83.0f), float2(-10.0f, -83.0f),
    float2(1.0f, -83.0f), float2(13.0f, -83.0f), float2(25.0f, -83.0f), float2(38.0f, -83.0f),
    float2(53.0f, -83.0f), float2(69.0f, -83.0f), float2(89.0f, -83.0f), float2(113.0f, -83.0f),
    float2(-127.0f, -65.0f), float2(-104.0f, -65.0f), float2(-83.0f, -65.0f), float2(-65.0f, -65.0f),
    float2(-49.0f, -65.0f), float2(-35.0f, -65.0f), float2(-22.0f, -65.0f), float2(-10.0f, -65.0f),
    float2(1.0f, -65.0f), float2(13.0f, -65.0f), float2(25.0f, -65.0f), float2(38.0f, -65.0f),
    float2(53.0f, -65.0f), float2(69.0f, -65.0f), float2(89.0f, -65.0f), float2(113.0f, -65.0f),
    float2(-127.0f, -49.0f), float2(-104.0f, -49.0f), float2(-83.0f, -49.0f), float2(-65.0f, -49.0f),
    float2(-49.0f, -49.0f), float2(-35.0f, -49.0f), float2(-22.0f, -49.0f), float2(-10.0f, -49.0f),
    float2(1.0f, -49.0f), float2(13.0f, -49.0f), float2(25.0f, -49.0f), float2(38.0f, -49.0f),
    float2(53.0f, -49.0f), float2(69.0f, -49.0f), float2(89.0f, -49.0f), float2(113.0f, -49.0f),
    float2(-127.0f, -35.0f), float2(-104.0f, -35.0f), float2(-83.0f, -35.0f), float2(-65.0f, -35.0f),
    float2(-49.0f, -35.0f), float2(-35.0f, -35.0f), float2(-22.0f, -35.0f), float2(-10.0f, -35.0f),
    float2(1.0f, -35.0f), float2(13.0f, -35.0f), float2(25.0f, -35.0f), float2(38.0f, -35.0f),
    float2(53.0f, -35.0f), float2(69.0f, -35.0f), float2(89.0f, -35.0f), float2(113.0f, -35.0f),
    float2(-127.0f, -22.0f), float2(-104.0f, -22.0f), float2(-83.0f, -22.0f), float2(-65.0f, -22.0f),
    float2(-49.0f, -22.0f), float2(-35.0f, -22.0f), float2(-22.0f, -22.0f), float2(-10.0f, -22.0f),
    float2(1.0f, -22.0f), float2(13.0f, -22.0f), float2(25.0f, -22.0f), float2(38.0f, -22.0f),
    float2(53.0f, -22.0f), float2(69.0f, -22.0f), float2(89.0f, -22.0f), float2(113.0f, -22.0f),
    float2(-127.0f, -10.0f), float2(-104.0f, -10.0f), float2(-83.0f, -10.0f), float2(-65.0f, -10.0f),
    float2(-49.0f, -10.0f), float2(-35.0f, -10.0f), float2(-22.0f, -10.0f), float2(-10.0f, -10.0f),
    float2(1.0f, -10.0f), float2(13.0f, -10.0f), float2(25.0f, -10.0f), float2(38.0f, -10.0f),
    float2(53.0f, -10.0f), float2(69.0f, -10.0f), float2(89.0f, -10.0f), float2(113.0f, -10.0f),
    float2(-127.0f, 1.0f), float2(-104.0f, 1.0f), float2(-83.0f, 1.0f), float2(-65.0f, 1.0f),
    float2(-49.0f, 1.0f), float2(-35.0f, 1.0f), float2(-22.0f, 1.0f), float2(-10.0f, 1.0f),
    float2(1.0f, 1.0f), float2(13.0f, 1.0f), float2(25.0f, 1.0f), float2(38.0f, 1.0f),
    float2(53.0f, 1.0f), float2(69.0f, 1.0f), float2(89.0f, 1.0f), float2(113.0f, 1.0f),
    float2(-127.0f, 13.0f), float2(-104.0f, 13.0f), float2(-83.0f, 13.0f), float2(-65.0f, 13.0f),
    float2(-49.0f, 13.0f), float2(-35.0f, 13.0f), float2(-22.0f, 13.0f), float2(-10.0f, 13.0f),
    float2(1.0f, 13.0f), float2(13.0f, 13.0f), float2(25.0f, 13.0f), float2(38.0f, 13.0f),
    float2(53.0f, 13.0f), float2(69.0f, 13.0f), float2(89.0f, 13.0f), float2(113.0f, 13.0f),
    float2(-127.0f, 25.0f), float2(-104.0f, 25.0f), float2(-83.0f, 25.0f), float2(-65.0f, 25.0f),
    float2(-49.0f, 25.0f), float2(-35.0f, 25.0f), float2(-22.0f, 25.0f), float2(-10.0f, 25.0f),
    float2(1.0f, 25.0f), float2(13.0f, 25.0f), float2(25.0f, 25.0f), float2(38.0f, 25.0f),
    float2(53.0f, 25.0f), float2(69.0f, 25.0f), float2(89.0f, 25.0f), float2(113.0f, 25.0f),
    float2(-127.0f, 38.0f), float2(-104.0f, 38.0f), float2(-83.0f, 38.0f), float2(-65.0f, 38.0f),
    float2(-49.0f, 38.0f), float2(-35.0f, 38.0f), float2(-22.0f, 38.0f), float2(-10.0f, 38.0f),
    float2(1.0f, 38.0f), float2(13.0f, 38.0f), float2(25.0f, 38.0f), float2(38.0f, 38.0f),
    float2(53.0f, 38.0f), float2(69.0f, 38.0f), float2(89.0f, 38.0f), float2(113.0f, 38.0f),
    float2(-127.0f, 53.0f), float2(-104.0f, 53.0f), float2(-83.0f, 53.0f), float2(-65.0f, 53.0f),
    float2(-49.0f, 53.0f), float2(-35.0f, 53.0f), float2(-22.0f, 53.0f), float2(-10.0f, 53.0f),
    float2(1.0f, 53.0f), float2(13.0f, 53.0f), float2(25.0f, 53.0f), float2(38.0f, 53.0f),
    float2(53.0f, 53.0f), float2(69.0f, 53.0f), float2(89.0f, 53.0f), float2(113.0f, 53.0f),
    float2(-127.0f, 69.0f), float2(-104.0f, 69.0f), float2(-83.0f, 69.0f), float2(-65.0f, 69.0f),
    float2(-49.0f, 69.0f), float2(-35.0f, 69.0f), float2(-22.0f, 69.0f), float2(-10.0f, 69.0f),
    float2(1.0f, 69.0f), float2(13.0f, 69.0f), float2(25.0f, 69.0f), float2(38.0f, 69.0f),
    float2(53.0f, 69.0f), float2(69.0f, 69.0f), float2(89.0f, 69.0f), float2(113.0f, 69.0f),
    float2(-127.0f, 89.0f), float2(-104.0f, 89.0f), float2(-83.0f, 89.0f), float2(-65.0f, 89.0f),
    float2(-49.0f, 89.0f), float2(-35.0f, 89.0f), float2(-22.0f, 89.0f), float2(-10.0f, 89.0f),
    float2(1.0f, 89.0f), float2(13.0f, 89.0f), float2(25.0f, 89.0f), float2(38.0f, 89.0f),
    float2(53.0f, 89.0f), float2(69.0f, 89.0f), float2(89.0f, 89.0f), float2(113.0f, 89.0f),
    float2(-127.0f, 113.0f), float2(-104.0f, 113.0f), float2(-83.0f, 113.0f), float2(-65.0f, 113.0f),
    float2(-49.0f, 113.0f), float2(-35.0f, 113.0f), float2(-22.0f, 113.0f), float2(-10.0f, 113.0f),
    float2(1.0f, 113.0f), float2(13.0f, 113.0f), float2(25.0f, 113.0f), float2(38.0f, 113.0f),
    float2(53.0f, 113.0f), float2(69.0f, 113.0f), float2(89.0f, 113.0f), float2(113.0f, 113.0f),
};

// The 6-bit sub-block scale and min for sub-block j of a Q4_K / Q5_K super-block, unpacked
// from the 12 packed bytes (llama.cpp's get_scale_min_k4): the first four pairs are plain
// 6-bit fields, the last four steal their top two bits from the first four's spare bits.
// The k-quant 6-bit scale and min of sub-block j from the header's three scale words
// (bytes 4..15 of a Q4_K / Q5_K block as little-endian uints). Fields 0..3 are the low
// six bits of bytes 0..3 (scales) and 4..7 (mins); fields 4..7 take their low nibble from
// bytes 8..11 and their top two bits from the first eight bytes. Branch-free: both forms
// are computed and `j < 4` picks one, so the eight lanes of a block, whose j differ, do
// not split into two paths.
inline void k_scale_min(uint j, uint y, uint z, uint w, thread uint & sc, thread uint & m) {
    const uint sh = (j & 3u) * 8u;
    const uint by = (y >> sh) & 0xFFu, bz = (z >> sh) & 0xFFu, bw = (w >> sh) & 0xFFu;
    const uint s_lo = by & 63u,                          m_lo = bz & 63u;
    const uint s_hi = (bw & 0xFu) | ((by >> 6) << 4),    m_hi = (bw >> 4) | ((bz >> 6) << 4);
    sc = j < 4u ? s_lo : s_hi;
    m  = j < 4u ? m_lo : m_hi;
}

// IQ3_S's codebook, 512 entries of four packed byte levels. Transcribed from the
// gated Rust codec (crates/imparo-cpu/src/quants.rs). Dead-stripped unless a
// pipeline compiles with WFMT == WF_IQ3_S, so no existing kernel carries it.
constant uint IQ3S_GRID[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

// IQ2_XXS's codebook: 256 entries of eight packed byte levels (ggml-common.h
// iq2xxs_grid). Dead-stripped unless a pipeline compiles with WFMT == WF_IQ2_XXS.
constant ulong IQ2XXS_GRID[256] = {
    0x0808080808080808ul, 0x080808080808082bul, 0x0808080808081919ul, 0x0808080808082b08ul,
    0x0808080808082b2bul, 0x0808080808190819ul, 0x0808080808191908ul, 0x08080808082b0808ul,
    0x08080808082b082bul, 0x08080808082b2b08ul, 0x08080808082b2b2bul, 0x0808080819080819ul,
    0x0808080819081908ul, 0x0808080819190808ul, 0x0808080819192b08ul, 0x08080808192b0819ul,
    0x08080808192b1908ul, 0x080808082b080808ul, 0x080808082b08082bul, 0x080808082b082b2bul,
    0x080808082b2b082bul, 0x0808081908080819ul, 0x0808081908081908ul, 0x0808081908190808ul,
    0x0808081908191919ul, 0x0808081919080808ul, 0x080808192b081908ul, 0x080808192b192b08ul,
    0x0808082b08080808ul, 0x0808082b0808082bul, 0x0808082b082b082bul, 0x0808082b2b08082bul,
    0x0808190808080819ul, 0x0808190808081908ul, 0x0808190808190808ul, 0x08081908082b0819ul,
    0x08081908082b1908ul, 0x0808190819080808ul, 0x080819081908082bul, 0x0808190819082b08ul,
    0x08081908192b0808ul, 0x080819082b080819ul, 0x080819082b081908ul, 0x080819082b190808ul,
    0x080819082b2b1908ul, 0x0808191908080808ul, 0x080819190808082bul, 0x0808191908082b08ul,
    0x08081919082b0808ul, 0x080819191908192bul, 0x08081919192b2b19ul, 0x080819192b080808ul,
    0x080819192b190819ul, 0x0808192b08082b19ul, 0x0808192b08190808ul, 0x0808192b19080808ul,
    0x0808192b2b081908ul, 0x0808192b2b2b1908ul, 0x08082b0808080808ul, 0x08082b0808081919ul,
    0x08082b0808082b08ul, 0x08082b0808191908ul, 0x08082b08082b2b08ul, 0x08082b0819080819ul,
    0x08082b0819081908ul, 0x08082b0819190808ul, 0x08082b081919082bul, 0x08082b082b082b08ul,
    0x08082b1908081908ul, 0x08082b1919080808ul, 0x08082b2b0808082bul, 0x08082b2b08191908ul,
    0x0819080808080819ul, 0x0819080808081908ul, 0x0819080808190808ul, 0x08190808082b0819ul,
    0x0819080819080808ul, 0x08190808192b0808ul, 0x081908082b081908ul, 0x081908082b190808ul,
    0x081908082b191919ul, 0x0819081908080808ul, 0x0819081908082b08ul, 0x08190819082b0808ul,
    0x0819081919190808ul, 0x0819081919192b2bul, 0x081908192b080808ul, 0x0819082b082b1908ul,
    0x0819082b19081919ul, 0x0819190808080808ul, 0x0819190808082b08ul, 0x08191908082b0808ul,
    0x08191908082b1919ul, 0x0819190819082b19ul, 0x081919082b080808ul, 0x0819191908192b08ul,
    0x08191919192b082bul, 0x0819192b08080808ul, 0x0819192b0819192bul, 0x08192b0808080819ul,
    0x08192b0808081908ul, 0x08192b0808190808ul, 0x08192b0819080808ul, 0x08192b082b080819ul,
    0x08192b1908080808ul, 0x08192b1908081919ul, 0x08192b192b2b0808ul, 0x08192b2b19190819ul,
    0x082b080808080808ul, 0x082b08080808082bul, 0x082b080808082b2bul, 0x082b080819081908ul,
    0x082b0808192b0819ul, 0x082b08082b080808ul, 0x082b08082b08082bul, 0x082b0819082b2b19ul,
    0x082b081919082b08ul, 0x082b082b08080808ul, 0x082b082b0808082bul, 0x082b190808080819ul,
    0x082b190808081908ul, 0x082b190808190808ul, 0x082b190819080808ul, 0x082b19081919192bul,
    0x082b191908080808ul, 0x082b191919080819ul, 0x082b1919192b1908ul, 0x082b192b2b190808ul,
    0x082b2b0808082b08ul, 0x082b2b08082b0808ul, 0x082b2b082b191908ul, 0x082b2b2b19081908ul,
    0x1908080808080819ul, 0x1908080808081908ul, 0x1908080808190808ul, 0x1908080808192b08ul,
    0x19080808082b0819ul, 0x19080808082b1908ul, 0x1908080819080808ul, 0x1908080819082b08ul,
    0x190808081919192bul, 0x19080808192b0808ul, 0x190808082b080819ul, 0x190808082b081908ul,
    0x190808082b190808ul, 0x1908081908080808ul, 0x19080819082b0808ul, 0x19080819192b0819ul,
    0x190808192b080808ul, 0x190808192b081919ul, 0x1908082b08080819ul, 0x1908082b08190808ul,
    0x1908082b19082b08ul, 0x1908082b1919192bul, 0x1908082b192b2b08ul, 0x1908190808080808ul,
    0x1908190808082b08ul, 0x19081908082b0808ul, 0x190819082b080808ul, 0x190819082b192b19ul,
    0x190819190819082bul, 0x19081919082b1908ul, 0x1908192b08080808ul, 0x19082b0808080819ul,
    0x19082b0808081908ul, 0x19082b0808190808ul, 0x19082b0819080808ul, 0x19082b0819081919ul,
    0x19082b1908080808ul, 0x19082b1919192b08ul, 0x19082b19192b0819ul, 0x19082b192b08082bul,
    0x19082b2b19081919ul, 0x19082b2b2b190808ul, 0x1919080808080808ul, 0x1919080808082b08ul,
    0x1919080808190819ul, 0x1919080808192b19ul, 0x19190808082b0808ul, 0x191908082b080808ul,
    0x191908082b082b08ul, 0x1919081908081908ul, 0x191908191908082bul, 0x191908192b2b1908ul,
    0x1919082b2b190819ul, 0x191919082b190808ul, 0x191919082b19082bul, 0x1919191908082b2bul,
    0x1919192b08080819ul, 0x1919192b19191908ul, 0x19192b0808080808ul, 0x19192b0808190819ul,
    0x19192b0808192b19ul, 0x19192b08192b1908ul, 0x19192b1919080808ul, 0x19192b2b08082b08ul,
    0x192b080808081908ul, 0x192b080808190808ul, 0x192b080819080808ul, 0x192b0808192b2b08ul,
    0x192b081908080808ul, 0x192b081919191919ul, 0x192b082b08192b08ul, 0x192b082b192b0808ul,
    0x192b190808080808ul, 0x192b190808081919ul, 0x192b191908190808ul, 0x192b19190819082bul,
    0x192b19192b081908ul, 0x192b2b081908082bul, 0x2b08080808080808ul, 0x2b0808080808082bul,
    0x2b08080808082b2bul, 0x2b08080819080819ul, 0x2b0808082b08082bul, 0x2b08081908081908ul,
    0x2b08081908192b08ul, 0x2b08081919080808ul, 0x2b08082b08190819ul, 0x2b08190808080819ul,
    0x2b08190808081908ul, 0x2b08190808190808ul, 0x2b08190808191919ul, 0x2b08190819080808ul,
    0x2b081908192b0808ul, 0x2b08191908080808ul, 0x2b0819191908192bul, 0x2b0819192b191908ul,
    0x2b08192b08082b19ul, 0x2b08192b19080808ul, 0x2b08192b192b0808ul, 0x2b082b080808082bul,
    0x2b082b1908081908ul, 0x2b082b2b08190819ul, 0x2b19080808081908ul, 0x2b19080808190808ul,
    0x2b190808082b1908ul, 0x2b19080819080808ul, 0x2b1908082b2b0819ul, 0x2b1908190819192bul,
    0x2b1908192b080808ul, 0x2b19082b19081919ul, 0x2b19190808080808ul, 0x2b191908082b082bul,
    0x2b19190819081908ul, 0x2b19191919190819ul, 0x2b192b082b080819ul, 0x2b192b19082b0808ul,
    0x2b2b08080808082bul, 0x2b2b080819190808ul, 0x2b2b08082b081919ul, 0x2b2b081908082b19ul,
    0x2b2b082b08080808ul, 0x2b2b190808192b08ul, 0x2b2b2b0819190808ul, 0x2b2b2b1908081908ul,
};

// IQ1_S / IQ1_M's codebook: 2048 entries of eight levels in {-1, 0, 1}, two bits each
// (level + 1), from ggml-common.h's iq1s_grid; the same numbers as gguf-py's grid_hex and as
// ggml's GPU table, which packs them four bits apart.
constant ushort IQ1S_GRID[2048] = {
    0x0000, 0x0002, 0x0005, 0x0008, 0x000a, 0x0011, 0x0015, 0x0020, 0x0022, 0x0028, 0x002a, 0x0045,
    0x0051, 0x0054, 0x0056, 0x0065, 0x0080, 0x0082, 0x0088, 0x008a, 0x0095, 0x00a0, 0x00a2, 0x00a8,
    0x00aa, 0x0104, 0x0105, 0x0111, 0x0114, 0x0116, 0x0119, 0x011a, 0x0125, 0x0141, 0x0146, 0x0149,
    0x0152, 0x0155, 0x015a, 0x0161, 0x0164, 0x0166, 0x0168, 0x0185, 0x0191, 0x0194, 0x0196, 0x01a5,
    0x0200, 0x0202, 0x0208, 0x020a, 0x0215, 0x0220, 0x0222, 0x0228, 0x022a, 0x0245, 0x0251, 0x0259,
    0x0264, 0x0269, 0x0280, 0x0282, 0x0288, 0x028a, 0x0291, 0x0295, 0x0299, 0x02a0, 0x02a2, 0x02a8,
    0x02aa, 0x0411, 0x0414, 0x0416, 0x0425, 0x0441, 0x0449, 0x0455, 0x045a, 0x0464, 0x0465, 0x0491,
    0x0499, 0x04a5, 0x0501, 0x0504, 0x0505, 0x0506, 0x0515, 0x0518, 0x051a, 0x0529, 0x0540, 0x0545,
    0x054a, 0x0550, 0x0551, 0x0554, 0x0555, 0x0556, 0x0559, 0x0560, 0x0562, 0x0565, 0x0568, 0x056a,
    0x0581, 0x0591, 0x0595, 0x0598, 0x059a, 0x05a1, 0x05a4, 0x05a5, 0x05a6, 0x05a9, 0x0614, 0x0619,
    0x0641, 0x0644, 0x0650, 0x0652, 0x0655, 0x0658, 0x0660, 0x0661, 0x0666, 0x0669, 0x0685, 0x0691,
    0x0694, 0x0699, 0x0800, 0x0802, 0x0808, 0x080a, 0x0815, 0x0820, 0x0822, 0x0828, 0x082a, 0x0845,
    0x0851, 0x0856, 0x0865, 0x0880, 0x0882, 0x0888, 0x088a, 0x0895, 0x08a0, 0x08a2, 0x08a8, 0x08aa,
    0x0905, 0x0911, 0x0914, 0x0919, 0x0924, 0x0925, 0x0941, 0x0950, 0x0951, 0x0955, 0x0961, 0x0964,
    0x0969, 0x0991, 0x0994, 0x0996, 0x0999, 0x09a5, 0x0a00, 0x0a02, 0x0a08, 0x0a0a, 0x0a15, 0x0a20,
    0x0a22, 0x0a28, 0x0a2a, 0x0a45, 0x0a51, 0x0a59, 0x0a61, 0x0a65, 0x0a80, 0x0a82, 0x0a85, 0x0a88,
    0x0a8a, 0x0a95, 0x0aa0, 0x0aa2, 0x0aa8, 0x0aaa, 0x1010, 0x1011, 0x1014, 0x1019, 0x1024, 0x1025,
    0x1041, 0x1044, 0x1050, 0x1055, 0x1058, 0x1061, 0x1064, 0x1065, 0x1069, 0x1091, 0x1094, 0x1096,
    0x10a1, 0x10a5, 0x1101, 0x1104, 0x1106, 0x1109, 0x1110, 0x1112, 0x1115, 0x1118, 0x1121, 0x1124,
    0x1129, 0x1145, 0x114a, 0x1150, 0x1151, 0x1152, 0x1154, 0x1155, 0x1156, 0x1159, 0x1160, 0x1165,
    0x1184, 0x1192, 0x1195, 0x11a1, 0x11a4, 0x1211, 0x1214, 0x1216, 0x1225, 0x1240, 0x1246, 0x1249,
    0x1252, 0x1255, 0x1258, 0x125a, 0x1264, 0x1266, 0x1285, 0x1291, 0x1294, 0x1296, 0x12a5, 0x1401,
    0x1406, 0x1409, 0x1414, 0x1415, 0x1418, 0x1419, 0x1421, 0x1426, 0x1441, 0x1445, 0x1446, 0x1448,
    0x144a, 0x1451, 0x1454, 0x1455, 0x1456, 0x1459, 0x1462, 0x1465, 0x1468, 0x1484, 0x1489, 0x1490,
    0x1494, 0x1495, 0x1498, 0x1499, 0x149a, 0x14a1, 0x14a4, 0x14a5, 0x14a9, 0x1502, 0x1505, 0x150a,
    0x1511, 0x1514, 0x1515, 0x1516, 0x1519, 0x1520, 0x1522, 0x1525, 0x1528, 0x152a, 0x1541, 0x1544,
    0x1545, 0x1546, 0x1551, 0x1552, 0x1554, 0x1555, 0x1556, 0x1559, 0x155a, 0x1561, 0x1564, 0x1565,
    0x1566, 0x1569, 0x1580, 0x1582, 0x1584, 0x1585, 0x1588, 0x158a, 0x1590, 0x1591, 0x1594, 0x1595,
    0x1596, 0x1599, 0x159a, 0x15a0, 0x15a2, 0x15a5, 0x1601, 0x1604, 0x1605, 0x1606, 0x1615, 0x1616,
    0x1618, 0x161a, 0x1621, 0x1626, 0x1640, 0x1642, 0x1644, 0x1645, 0x1648, 0x164a, 0x1651, 0x1655,
    0x1656, 0x1658, 0x1659, 0x1661, 0x1664, 0x1665, 0x1668, 0x1669, 0x166a, 0x1686, 0x168a, 0x1692,
    0x1695, 0x16a4, 0x16a9, 0x1811, 0x1816, 0x1825, 0x1841, 0x1844, 0x1846, 0x1849, 0x1850, 0x1855,
    0x1858, 0x185a, 0x1860, 0x1861, 0x1864, 0x1866, 0x1869, 0x1885, 0x1891, 0x1894, 0x18a5, 0x1910,
    0x1912, 0x1915, 0x191a, 0x1921, 0x1925, 0x1942, 0x1944, 0x1945, 0x1948, 0x1951, 0x1954, 0x1955,
    0x1956, 0x1959, 0x195a, 0x1960, 0x1965, 0x196a, 0x1989, 0x1991, 0x1992, 0x1995, 0x1998, 0x19a1,
    0x19a6, 0x19a9, 0x1a09, 0x1a16, 0x1a24, 0x1a26, 0x1a44, 0x1a46, 0x1a49, 0x1a50, 0x1a52, 0x1a55,
    0x1a58, 0x1a61, 0x1a66, 0x1a69, 0x1a85, 0x1a91, 0x1a96, 0x1a9a, 0x2000, 0x2002, 0x2008, 0x200a,
    0x2015, 0x2020, 0x2022, 0x2025, 0x2028, 0x202a, 0x2045, 0x2051, 0x2059, 0x2061, 0x2065, 0x2080,
    0x2082, 0x2088, 0x208a, 0x2095, 0x20a0, 0x20a2, 0x20a5, 0x20a8, 0x20aa, 0x2105, 0x2111, 0x2114,
    0x2119, 0x2125, 0x2142, 0x2144, 0x2149, 0x2155, 0x2158, 0x215a, 0x2161, 0x2164, 0x2165, 0x2166,
    0x2185, 0x2190, 0x2196, 0x2199, 0x21a5, 0x2201, 0x2208, 0x220a, 0x2211, 0x2215, 0x2220, 0x2222,
    0x2228, 0x222a, 0x2245, 0x2251, 0x2256, 0x2259, 0x2265, 0x2281, 0x2288, 0x228a, 0x2291, 0x2295,
    0x22a0, 0x22a2, 0x22a8, 0x22aa, 0x2405, 0x2414, 0x2416, 0x2419, 0x2425, 0x2444, 0x2445, 0x2446,
    0x2449, 0x2452, 0x2455, 0x2458, 0x245a, 0x2466, 0x2485, 0x2491, 0x2494, 0x2499, 0x24a1, 0x24a5,
    0x2509, 0x2515, 0x2521, 0x2529, 0x2540, 0x2545, 0x2548, 0x2551, 0x2554, 0x2555, 0x2559, 0x2562,
    0x2565, 0x2568, 0x2589, 0x2590, 0x2594, 0x2595, 0x2598, 0x259a, 0x25a1, 0x25a4, 0x25a6, 0x25a9,
    0x2605, 0x2610, 0x2612, 0x2619, 0x2625, 0x2641, 0x2649, 0x2655, 0x2660, 0x2661, 0x2669, 0x2684,
    0x2686, 0x2690, 0x269a, 0x2800, 0x2802, 0x2808, 0x280a, 0x2815, 0x2820, 0x2822, 0x2828, 0x282a,
    0x2845, 0x2851, 0x2854, 0x2865, 0x2880, 0x2882, 0x2888, 0x288a, 0x28a0, 0x28a2, 0x28a8, 0x28aa,
    0x2909, 0x2911, 0x2914, 0x2919, 0x2925, 0x2946, 0x2949, 0x2952, 0x2955, 0x2961, 0x2964, 0x2966,
    0x2969, 0x2985, 0x2990, 0x2996, 0x2999, 0x29a4, 0x29a5, 0x2a00, 0x2a02, 0x2a08, 0x2a0a, 0x2a20,
    0x2a22, 0x2a28, 0x2a2a, 0x2a45, 0x2a51, 0x2a56, 0x2a59, 0x2a65, 0x2a80, 0x2a82, 0x2a88, 0x2a8a,
    0x2a95, 0x2aa0, 0x2aa2, 0x2aa8, 0x2aaa, 0x4005, 0x4011, 0x4016, 0x4025, 0x4049, 0x4052, 0x4055,
    0x4058, 0x405a, 0x4061, 0x4064, 0x4066, 0x4094, 0x4099, 0x40a1, 0x40a6, 0x4100, 0x4101, 0x4104,
    0x4106, 0x4109, 0x4112, 0x4115, 0x4116, 0x4118, 0x411a, 0x4121, 0x4126, 0x4129, 0x4145, 0x4148,
    0x414a, 0x4151, 0x4154, 0x4155, 0x4156, 0x4159, 0x415a, 0x4165, 0x4168, 0x416a, 0x4181, 0x4184,
    0x4186, 0x4190, 0x4192, 0x4195, 0x41a0, 0x41a1, 0x41a2, 0x4205, 0x4211, 0x4214, 0x4216, 0x4225,
    0x4241, 0x4252, 0x4255, 0x425a, 0x4264, 0x4269, 0x4289, 0x4294, 0x42a5, 0x4401, 0x4415, 0x4419,
    0x4429, 0x4445, 0x4448, 0x444a, 0x4451, 0x4454, 0x4455, 0x4456, 0x4461, 0x4462, 0x4465, 0x4468,
    0x446a, 0x4481, 0x4486, 0x4489, 0x4490, 0x4492, 0x4495, 0x44a0, 0x44a1, 0x44a9, 0x4501, 0x4502,
    0x4505, 0x450a, 0x4511, 0x4514, 0x4515, 0x4516, 0x4519, 0x4520, 0x4525, 0x452a, 0x4541, 0x4544,
    0x4545, 0x4546, 0x4549, 0x4550, 0x4551, 0x4554, 0x4555, 0x4556, 0x4558, 0x4559, 0x4561, 0x4564,
    0x4565, 0x4566, 0x4569, 0x4582, 0x4584, 0x4585, 0x4588, 0x4591, 0x4594, 0x4595, 0x4596, 0x4599,
    0x459a, 0x45a5, 0x45a8, 0x45aa, 0x4601, 0x4605, 0x4609, 0x4614, 0x4615, 0x4618, 0x461a, 0x4621,
    0x4624, 0x4629, 0x4640, 0x4642, 0x4645, 0x4648, 0x4650, 0x4651, 0x4652, 0x4655, 0x4656, 0x4659,
    0x4662, 0x4665, 0x4668, 0x4681, 0x4685, 0x468a, 0x4694, 0x4695, 0x46a1, 0x46a4, 0x46a6, 0x4805,
    0x4811, 0x4815, 0x481a, 0x4825, 0x4842, 0x4849, 0x4850, 0x4855, 0x4858, 0x4861, 0x4864, 0x4866,
    0x4869, 0x4885, 0x4891, 0x4894, 0x4896, 0x4899, 0x48a5, 0x4901, 0x4905, 0x4906, 0x490a, 0x4910,
    0x4914, 0x4915, 0x4918, 0x4921, 0x4924, 0x4926, 0x4940, 0x4945, 0x494a, 0x4951, 0x4952, 0x4954,
    0x4955, 0x4956, 0x4959, 0x4960, 0x4962, 0x4965, 0x4966, 0x496a, 0x4986, 0x4989, 0x4992, 0x4995,
    0x4996, 0x4998, 0x49a1, 0x49a4, 0x49a6, 0x49a9, 0x4a16, 0x4a44, 0x4a46, 0x4a49, 0x4a55, 0x4a58,
    0x4a5a, 0x4a64, 0x4a69, 0x4a94, 0x4aa5, 0x5001, 0x5004, 0x5005, 0x5006, 0x5009, 0x5012, 0x5015,
    0x501a, 0x5021, 0x5024, 0x5029, 0x5040, 0x5045, 0x5048, 0x5051, 0x5054, 0x5055, 0x5056, 0x5059,
    0x5065, 0x5068, 0x5086, 0x5089, 0x5095, 0x5098, 0x50a0, 0x50a1, 0x50a6, 0x50a9, 0x5105, 0x5108,
    0x5109, 0x510a, 0x5111, 0x5114, 0x5115, 0x5116, 0x5118, 0x5119, 0x5120, 0x5125, 0x5126, 0x5128,
    0x512a, 0x5141, 0x5144, 0x5145, 0x5146, 0x5149, 0x5150, 0x5151, 0x5152, 0x5154, 0x5155, 0x5156,
    0x5158, 0x5159, 0x515a, 0x5161, 0x5164, 0x5165, 0x5166, 0x5169, 0x5182, 0x5185, 0x5191, 0x5194,
    0x5195, 0x5196, 0x5199, 0x51a0, 0x51a5, 0x51aa, 0x5201, 0x5206, 0x5212, 0x5215, 0x521a, 0x5221,
    0x5224, 0x5242, 0x5245, 0x524a, 0x5251, 0x5254, 0x5255, 0x5256, 0x5259, 0x5262, 0x5265, 0x5285,
    0x5290, 0x5292, 0x5295, 0x5299, 0x529a, 0x52a4, 0x5404, 0x5405, 0x5411, 0x5414, 0x5415, 0x5416,
    0x5418, 0x5419, 0x5421, 0x5425, 0x5428, 0x542a, 0x5441, 0x5444, 0x5445, 0x5446, 0x5449, 0x544a,
    0x5450, 0x5451, 0x5454, 0x5455, 0x5456, 0x5458, 0x5459, 0x545a, 0x5461, 0x5462, 0x5464, 0x5465,
    0x5466, 0x5469, 0x5480, 0x5488, 0x548a, 0x5491, 0x5494, 0x5495, 0x5496, 0x5499, 0x54a1, 0x54a4,
    0x54a5, 0x54aa, 0x5501, 0x5502, 0x5504, 0x5505, 0x5506, 0x5509, 0x5510, 0x5511, 0x5512, 0x5514,
    0x5515, 0x5516, 0x5519, 0x551a, 0x5521, 0x5524, 0x5525, 0x5526, 0x5529, 0x5540, 0x5541, 0x5542,
    0x5544, 0x5545, 0x5546, 0x5548, 0x5549, 0x5550, 0x5551, 0x5552, 0x5554, 0x5555, 0x5556, 0x5558,
    0x5559, 0x555a, 0x5560, 0x5561, 0x5564, 0x5565, 0x5566, 0x5568, 0x5569, 0x556a, 0x5581, 0x5584,
    0x5585, 0x5589, 0x558a, 0x5590, 0x5591, 0x5594, 0x5595, 0x5596, 0x5598, 0x5599, 0x55a1, 0x55a4,
    0x55a5, 0x55a6, 0x55a9, 0x5600, 0x5601, 0x5602, 0x5604, 0x5606, 0x5608, 0x5609, 0x5611, 0x5614,
    0x5615, 0x5618, 0x5619, 0x5620, 0x5621, 0x5622, 0x5624, 0x5625, 0x5626, 0x5628, 0x5629, 0x5641,
    0x5645, 0x5646, 0x5648, 0x5649, 0x564a, 0x5650, 0x5651, 0x5652, 0x5654, 0x5655, 0x5656, 0x5658,
    0x5659, 0x565a, 0x5661, 0x5664, 0x5665, 0x5669, 0x5682, 0x5685, 0x5686, 0x5688, 0x5689, 0x568a,
    0x5691, 0x5695, 0x569a, 0x56a2, 0x56a5, 0x56a6, 0x56a8, 0x56a9, 0x5804, 0x5805, 0x5806, 0x5809,
    0x5810, 0x5815, 0x5818, 0x5821, 0x582a, 0x5845, 0x5848, 0x584a, 0x5851, 0x5854, 0x5855, 0x5856,
    0x5858, 0x5859, 0x5860, 0x5862, 0x5864, 0x5865, 0x5882, 0x5889, 0x5890, 0x5892, 0x5895, 0x5898,
    0x58a1, 0x58a9, 0x5901, 0x5902, 0x5905, 0x590a, 0x5911, 0x5914, 0x5915, 0x5916, 0x5919, 0x5925,
    0x5941, 0x5944, 0x5945, 0x5946, 0x5949, 0x5950, 0x5951, 0x5952, 0x5954, 0x5955, 0x5956, 0x5958,
    0x5959, 0x595a, 0x5961, 0x5964, 0x5965, 0x5966, 0x5969, 0x5981, 0x5985, 0x5989, 0x5991, 0x5994,
    0x5995, 0x5996, 0x5998, 0x5999, 0x59a5, 0x5a04, 0x5a08, 0x5a15, 0x5a1a, 0x5a20, 0x5a25, 0x5a26,
    0x5a29, 0x5a45, 0x5a48, 0x5a49, 0x5a51, 0x5a55, 0x5a56, 0x5a58, 0x5a59, 0x5a62, 0x5a65, 0x5a68,
    0x5a6a, 0x5a81, 0x5a8a, 0x5a92, 0x5a95, 0x5a96, 0x5a98, 0x5a9a, 0x5aa1, 0x6005, 0x6014, 0x6016,
    0x6019, 0x6025, 0x6044, 0x6050, 0x6055, 0x6056, 0x6058, 0x605a, 0x6061, 0x6064, 0x6066, 0x6069,
    0x6081, 0x6096, 0x60a5, 0x6101, 0x6104, 0x6106, 0x6109, 0x6112, 0x6115, 0x6121, 0x6122, 0x6126,
    0x6129, 0x6145, 0x6149, 0x6151, 0x6155, 0x6156, 0x6159, 0x6165, 0x6166, 0x616a, 0x6184, 0x618a,
    0x6192, 0x6195, 0x61a1, 0x61a6, 0x61a9, 0x6211, 0x6216, 0x6219, 0x6240, 0x6241, 0x6246, 0x6255,
    0x6256, 0x6258, 0x6260, 0x6285, 0x6291, 0x6296, 0x62a5, 0x6411, 0x6412, 0x6415, 0x6416, 0x641a,
    0x6421, 0x6426, 0x6429, 0x6440, 0x6442, 0x6445, 0x6448, 0x644a, 0x6451, 0x6454, 0x6455, 0x6456,
    0x6459, 0x645a, 0x6460, 0x6462, 0x6465, 0x6484, 0x6485, 0x6489, 0x6490, 0x6492, 0x6494, 0x6495,
    0x6496, 0x6498, 0x649a, 0x64a1, 0x64a4, 0x64a9, 0x6505, 0x6508, 0x650a, 0x6511, 0x6515, 0x6516,
    0x6519, 0x6544, 0x6545, 0x6546, 0x6549, 0x6550, 0x6551, 0x6554, 0x6555, 0x6556, 0x6559, 0x6561,
    0x6564, 0x6565, 0x6566, 0x6569, 0x6586, 0x6589, 0x658a, 0x6591, 0x6595, 0x6596, 0x6599, 0x659a,
    0x65a2, 0x65a5, 0x65a6, 0x65a8, 0x6602, 0x6609, 0x6615, 0x6620, 0x6626, 0x6628, 0x6629, 0x6640,
    0x6645, 0x6648, 0x664a, 0x6651, 0x6654, 0x6655, 0x6656, 0x6658, 0x665a, 0x6660, 0x6665, 0x6668,
    0x6680, 0x6682, 0x6685, 0x668a, 0x6694, 0x6696, 0x6698, 0x6699, 0x66a0, 0x66a4, 0x66a6, 0x66aa,
    0x6816, 0x6819, 0x6825, 0x6841, 0x6852, 0x6855, 0x685a, 0x6861, 0x6869, 0x6885, 0x6891, 0x6898,
    0x68a6, 0x6901, 0x6904, 0x6910, 0x6915, 0x6921, 0x6924, 0x6926, 0x6929, 0x6940, 0x6941, 0x6945,
    0x6946, 0x6948, 0x6951, 0x6954, 0x6955, 0x6956, 0x6959, 0x6960, 0x6965, 0x696a, 0x6982, 0x6984,
    0x698a, 0x6995, 0x69a1, 0x69a4, 0x69a5, 0x69a9, 0x6a11, 0x6a16, 0x6a18, 0x6a41, 0x6a44, 0x6a49,
    0x6a50, 0x6a55, 0x6a58, 0x6a5a, 0x6a64, 0x6a65, 0x6a69, 0x6a86, 0x6a94, 0x6a98, 0x6a9a, 0x6aa6,
    0x8000, 0x8002, 0x8008, 0x800a, 0x8020, 0x8022, 0x8028, 0x802a, 0x8045, 0x8050, 0x8051, 0x8054,
    0x8056, 0x8059, 0x8065, 0x8080, 0x8082, 0x8088, 0x808a, 0x8095, 0x80a0, 0x80a2, 0x80a8, 0x80aa,
    0x8105, 0x8111, 0x8114, 0x8116, 0x8119, 0x8125, 0x8141, 0x8144, 0x8149, 0x8150, 0x8152, 0x8155,
    0x8156, 0x8158, 0x8159, 0x8164, 0x8166, 0x8169, 0x8185, 0x8189, 0x8194, 0x8196, 0x8199, 0x81a5,
    0x8200, 0x8202, 0x8208, 0x820a, 0x8215, 0x8220, 0x8222, 0x8228, 0x822a, 0x8251, 0x8254, 0x8259,
    0x8265, 0x8280, 0x8282, 0x8288, 0x828a, 0x8295, 0x82a0, 0x82a2, 0x82a8, 0x82aa, 0x8414, 0x8419,
    0x8441, 0x8444, 0x8451, 0x8455, 0x845a, 0x8461, 0x8464, 0x8469, 0x8494, 0x8499, 0x8501, 0x8509,
    0x8512, 0x8515, 0x851a, 0x8526, 0x8529, 0x8540, 0x8541, 0x8545, 0x8548, 0x8551, 0x8554, 0x8555,
    0x8556, 0x8559, 0x855a, 0x8565, 0x8566, 0x8568, 0x856a, 0x8581, 0x8584, 0x8586, 0x8589, 0x8590,
    0x8592, 0x8595, 0x8598, 0x85a6, 0x8611, 0x8616, 0x8619, 0x8625, 0x8641, 0x8644, 0x8649, 0x864a,
    0x8650, 0x8655, 0x8659, 0x865a, 0x8661, 0x8666, 0x866a, 0x8685, 0x8691, 0x869a, 0x86a4, 0x8800,
    0x8802, 0x8808, 0x880a, 0x8815, 0x8820, 0x8822, 0x8828, 0x882a, 0x8841, 0x8845, 0x8851, 0x8854,
    0x8859, 0x8865, 0x8869, 0x8880, 0x8882, 0x8888, 0x888a, 0x8895, 0x88a0, 0x88a2, 0x88a8, 0x88aa,
    0x8905, 0x8906, 0x8911, 0x8914, 0x8916, 0x8925, 0x8941, 0x8944, 0x8946, 0x8949, 0x8950, 0x8952,
    0x8955, 0x895a, 0x8961, 0x8964, 0x8985, 0x8996, 0x8999, 0x89a5, 0x8a00, 0x8a02, 0x8a08, 0x8a0a,
    0x8a15, 0x8a20, 0x8a22, 0x8a28, 0x8a2a, 0x8a45, 0x8a51, 0x8a54, 0x8a56, 0x8a80, 0x8a82, 0x8a88,
    0x8a8a, 0x8a95, 0x8aa0, 0x8aa2, 0x8aa8, 0x8aaa, 0x9005, 0x9011, 0x9016, 0x9018, 0x9019, 0x9025,
    0x9041, 0x9046, 0x9049, 0x9055, 0x9058, 0x905a, 0x9069, 0x906a, 0x9085, 0x9091, 0x9094, 0x9096,
    0x9099, 0x90a5, 0x9101, 0x9104, 0x9106, 0x9109, 0x9110, 0x9115, 0x9118, 0x911a, 0x9121, 0x9124,
    0x9126, 0x9129, 0x9140, 0x9145, 0x9150, 0x9151, 0x9154, 0x9155, 0x9156, 0x9159, 0x9162, 0x9165,
    0x9184, 0x9186, 0x9192, 0x9195, 0x9198, 0x91a1, 0x91a4, 0x91a6, 0x91a9, 0x9205, 0x9211, 0x9214,
    0x9219, 0x9225, 0x9244, 0x9246, 0x9249, 0x9250, 0x9252, 0x9255, 0x9258, 0x9266, 0x9269, 0x9285,
    0x9294, 0x9296, 0x92a9, 0x9401, 0x9404, 0x9406, 0x9410, 0x9415, 0x9418, 0x9426, 0x9440, 0x944a,
    0x9451, 0x9454, 0x9455, 0x9456, 0x9458, 0x9459, 0x9460, 0x9461, 0x9462, 0x9465, 0x9484, 0x9486,
    0x9492, 0x9494, 0x9495, 0x9498, 0x94a1, 0x94a9, 0x9500, 0x9505, 0x9508, 0x950a, 0x9510, 0x9511,
    0x9514, 0x9515, 0x9516, 0x9519, 0x9521, 0x9525, 0x9529, 0x952a, 0x9541, 0x9544, 0x9545, 0x9546,
    0x9549, 0x9550, 0x9551, 0x9552, 0x9554, 0x9555, 0x9556, 0x9558, 0x9559, 0x955a, 0x9561, 0x9564,
    0x9565, 0x9566, 0x9569, 0x9581, 0x9585, 0x9588, 0x9591, 0x9592, 0x9594, 0x9595, 0x9596, 0x9599,
    0x959a, 0x95a0, 0x95a2, 0x95a5, 0x95a8, 0x95aa, 0x9601, 0x9604, 0x9610, 0x9615, 0x9619, 0x9620,
    0x9626, 0x9629, 0x9645, 0x9648, 0x9649, 0x9651, 0x9652, 0x9655, 0x9656, 0x9659, 0x9665, 0x9668,
    0x9682, 0x9684, 0x9689, 0x968a, 0x9692, 0x9694, 0x9695, 0x96a4, 0x96a6, 0x96a9, 0x9805, 0x9816,
    0x9819, 0x9825, 0x9841, 0x9846, 0x9850, 0x9852, 0x9855, 0x9856, 0x985a, 0x9864, 0x9865, 0x9885,
    0x9891, 0x9896, 0x9899, 0x98a5, 0x9904, 0x9906, 0x9909, 0x9910, 0x9912, 0x9915, 0x9918, 0x991a,
    0x9920, 0x9921, 0x9924, 0x9926, 0x9940, 0x9942, 0x9945, 0x9948, 0x994a, 0x9951, 0x9954, 0x9955,
    0x9956, 0x9959, 0x9962, 0x9965, 0x9966, 0x996a, 0x9981, 0x9984, 0x9990, 0x9992, 0x9995, 0x999a,
    0x99a1, 0x99a6, 0x9a05, 0x9a15, 0x9a25, 0x9a44, 0x9a46, 0x9a49, 0x9a50, 0x9a55, 0x9a58, 0x9a61,
    0x9a85, 0x9a91, 0x9a94, 0x9a95, 0x9a96, 0xa000, 0xa002, 0xa008, 0xa00a, 0xa015, 0xa020, 0xa022,
    0xa028, 0xa02a, 0xa045, 0xa051, 0xa054, 0xa056, 0xa059, 0xa080, 0xa082, 0xa088, 0xa08a, 0xa095,
    0xa0a0, 0xa0a2, 0xa0a8, 0xa0aa, 0xa105, 0xa109, 0xa111, 0xa114, 0xa116, 0xa119, 0xa11a, 0xa146,
    0xa149, 0xa151, 0xa155, 0xa158, 0xa15a, 0xa161, 0xa164, 0xa185, 0xa190, 0xa192, 0xa196, 0xa199,
    0xa202, 0xa208, 0xa20a, 0xa210, 0xa219, 0xa222, 0xa228, 0xa22a, 0xa245, 0xa251, 0xa256, 0xa259,
    0xa265, 0xa280, 0xa282, 0xa288, 0xa28a, 0xa295, 0xa2a0, 0xa2a2, 0xa2a8, 0xa2aa, 0xa419, 0xa425,
    0xa441, 0xa444, 0xa450, 0xa454, 0xa455, 0xa458, 0xa45a, 0xa461, 0xa465, 0xa466, 0xa468, 0xa469,
    0xa485, 0xa506, 0xa509, 0xa510, 0xa512, 0xa515, 0xa518, 0xa526, 0xa529, 0xa542, 0xa545, 0xa551,
    0xa554, 0xa555, 0xa556, 0xa559, 0xa565, 0xa56a, 0xa581, 0xa584, 0xa585, 0xa586, 0xa589, 0xa592,
    0xa595, 0xa598, 0xa605, 0xa611, 0xa616, 0xa61a, 0xa621, 0xa625, 0xa644, 0xa646, 0xa64a, 0xa652,
    0xa655, 0xa656, 0xa658, 0xa660, 0xa662, 0xa686, 0xa690, 0xa695, 0xa696, 0xa699, 0xa6a1, 0xa6a4,
    0xa6a6, 0xa800, 0xa802, 0xa808, 0xa80a, 0xa820, 0xa822, 0xa828, 0xa82a, 0xa851, 0xa854, 0xa856,
    0xa859, 0xa880, 0xa882, 0xa888, 0xa88a, 0xa895, 0xa8a0, 0xa8a2, 0xa8a8, 0xa8aa, 0xa905, 0xa914,
    0xa919, 0xa921, 0xa925, 0xa941, 0xa950, 0xa955, 0xa95a, 0xa961, 0xa966, 0xa969, 0xa990, 0xa996,
    0xaa00, 0xaa02, 0xaa08, 0xaa0a, 0xaa20, 0xaa22, 0xaa28, 0xaa2a, 0xaa51, 0xaa54, 0xaa56, 0xaa80,
    0xaa82, 0xaa88, 0xaa8a, 0xaa95, 0xaaa0, 0xaaa2, 0xaaa8, 0xaaaa,
};

// IQ2_XS's codebook: 512 entries of eight packed byte levels. Dead-stripped unless a
// pipeline compiles with WFMT == WF_IQ2_XS.
constant ulong IQ2XS_GRID[512] = {
    0x0808080808080808ul, 0x080808080808082bul, 0x0808080808081919ul, 0x0808080808082b08ul,
    0x0808080808082b2bul, 0x0808080808190819ul, 0x0808080808191908ul, 0x080808080819192bul,
    0x0808080808192b19ul, 0x08080808082b0808ul, 0x08080808082b082bul, 0x08080808082b1919ul,
    0x08080808082b2b08ul, 0x0808080819080819ul, 0x0808080819081908ul, 0x080808081908192bul,
    0x0808080819082b19ul, 0x0808080819190808ul, 0x080808081919082bul, 0x0808080819191919ul,
    0x0808080819192b08ul, 0x08080808192b0819ul, 0x08080808192b1908ul, 0x080808082b080808ul,
    0x080808082b08082bul, 0x080808082b081919ul, 0x080808082b082b08ul, 0x080808082b190819ul,
    0x080808082b191908ul, 0x080808082b192b19ul, 0x080808082b2b0808ul, 0x0808081908080819ul,
    0x0808081908081908ul, 0x080808190808192bul, 0x0808081908082b19ul, 0x0808081908190808ul,
    0x080808190819082bul, 0x0808081908191919ul, 0x0808081908192b08ul, 0x0808081908192b2bul,
    0x08080819082b0819ul, 0x08080819082b1908ul, 0x0808081919080808ul, 0x080808191908082bul,
    0x0808081919081919ul, 0x0808081919082b08ul, 0x0808081919190819ul, 0x0808081919191908ul,
    0x08080819192b0808ul, 0x08080819192b2b08ul, 0x080808192b080819ul, 0x080808192b081908ul,
    0x080808192b190808ul, 0x0808082b08080808ul, 0x0808082b0808082bul, 0x0808082b08081919ul,
    0x0808082b08082b08ul, 0x0808082b08190819ul, 0x0808082b08191908ul, 0x0808082b082b0808ul,
    0x0808082b19080819ul, 0x0808082b19081908ul, 0x0808082b19190808ul, 0x0808082b19191919ul,
    0x0808082b2b080808ul, 0x0808082b2b082b2bul, 0x0808190808080819ul, 0x0808190808081908ul,
    0x080819080808192bul, 0x0808190808082b19ul, 0x0808190808190808ul, 0x080819080819082bul,
    0x0808190808191919ul, 0x0808190808192b08ul, 0x08081908082b0819ul, 0x08081908082b1908ul,
    0x0808190819080808ul, 0x080819081908082bul, 0x0808190819081919ul, 0x0808190819082b08ul,
    0x0808190819190819ul, 0x0808190819191908ul, 0x080819081919192bul, 0x08081908192b0808ul,
    0x080819082b080819ul, 0x080819082b081908ul, 0x080819082b190808ul, 0x0808191908080808ul,
    0x080819190808082bul, 0x0808191908081919ul, 0x0808191908082b08ul, 0x0808191908190819ul,
    0x0808191908191908ul, 0x08081919082b0808ul, 0x0808191919080819ul, 0x0808191919081908ul,
    0x0808191919190808ul, 0x08081919192b0819ul, 0x080819192b080808ul, 0x0808192b08080819ul,
    0x0808192b08081908ul, 0x0808192b08190808ul, 0x0808192b082b192bul, 0x0808192b19080808ul,
    0x0808192b1908082bul, 0x0808192b2b081908ul, 0x08082b0808080808ul, 0x08082b080808082bul,
    0x08082b0808081919ul, 0x08082b0808082b08ul, 0x08082b0808082b2bul, 0x08082b0808190819ul,
    0x08082b0808191908ul, 0x08082b08082b0808ul, 0x08082b08082b1919ul, 0x08082b0819080819ul,
    0x08082b0819081908ul, 0x08082b0819190808ul, 0x08082b0819192b08ul, 0x08082b082b080808ul,
    0x08082b082b2b0808ul, 0x08082b082b2b2b2bul, 0x08082b1908080819ul, 0x08082b1908081908ul,
    0x08082b1908190808ul, 0x08082b1919080808ul, 0x08082b192b080819ul, 0x08082b192b082b19ul,
    0x08082b2b08080808ul, 0x08082b2b082b0808ul, 0x08082b2b082b2b08ul, 0x08082b2b2b19192bul,
    0x08082b2b2b2b0808ul, 0x0819080808080819ul, 0x0819080808081908ul, 0x081908080808192bul,
    0x0819080808082b19ul, 0x0819080808190808ul, 0x081908080819082bul, 0x0819080808191919ul,
    0x0819080808192b08ul, 0x08190808082b0819ul, 0x08190808082b1908ul, 0x0819080819080808ul,
    0x081908081908082bul, 0x0819080819081919ul, 0x0819080819082b08ul, 0x0819080819190819ul,
    0x0819080819191908ul, 0x08190808192b0808ul, 0x08190808192b2b2bul, 0x081908082b080819ul,
    0x081908082b081908ul, 0x081908082b190808ul, 0x0819081908080808ul, 0x081908190808082bul,
    0x0819081908081919ul, 0x0819081908082b08ul, 0x0819081908190819ul, 0x0819081908191908ul,
    0x08190819082b0808ul, 0x0819081919080819ul, 0x0819081919081908ul, 0x0819081919190808ul,
    0x081908192b080808ul, 0x081908192b191908ul, 0x081908192b19192bul, 0x0819082b08080819ul,
    0x0819082b08081908ul, 0x0819082b0808192bul, 0x0819082b08190808ul, 0x0819082b19080808ul,
    0x0819082b192b0808ul, 0x0819190808080808ul, 0x081919080808082bul, 0x0819190808081919ul,
    0x0819190808082b08ul, 0x0819190808190819ul, 0x0819190808191908ul, 0x08191908082b0808ul,
    0x0819190819080819ul, 0x0819190819081908ul, 0x0819190819082b19ul, 0x0819190819190808ul,
    0x08191908192b1908ul, 0x081919082b080808ul, 0x0819191908080819ul, 0x0819191908081908ul,
    0x0819191908190808ul, 0x0819191919080808ul, 0x0819192b08080808ul, 0x0819192b08191908ul,
    0x0819192b19082b19ul, 0x08192b0808080819ul, 0x08192b0808081908ul, 0x08192b0808190808ul,
    0x08192b080819082bul, 0x08192b0819080808ul, 0x08192b0819191908ul, 0x08192b082b08192bul,
    0x08192b1908080808ul, 0x08192b1908081919ul, 0x08192b19192b192bul, 0x08192b2b19190819ul,
    0x08192b2b2b2b2b19ul, 0x082b080808080808ul, 0x082b08080808082bul, 0x082b080808081919ul,
    0x082b080808082b08ul, 0x082b080808082b2bul, 0x082b080808190819ul, 0x082b080808191908ul,
    0x082b0808082b0808ul, 0x082b080819080819ul, 0x082b080819081908ul, 0x082b080819190808ul,
    0x082b08082b080808ul, 0x082b08082b2b0808ul, 0x082b081908080819ul, 0x082b081908081908ul,
    0x082b081908190808ul, 0x082b081919080808ul, 0x082b081919082b08ul, 0x082b0819192b1919ul,
    0x082b082b08080808ul, 0x082b082b082b082bul, 0x082b082b2b080808ul, 0x082b082b2b2b2b08ul,
    0x082b190808080819ul, 0x082b190808081908ul, 0x082b190808190808ul, 0x082b1908082b2b19ul,
    0x082b190819080808ul, 0x082b191908080808ul, 0x082b191919080819ul, 0x082b19191919082bul,
    0x082b19192b192b19ul, 0x082b192b08080819ul, 0x082b192b08192b2bul, 0x082b192b2b2b192bul,
    0x082b2b0808080808ul, 0x082b2b0808082b08ul, 0x082b2b0808082b2bul, 0x082b2b08082b0808ul,
    0x082b2b0819191919ul, 0x082b2b082b082b08ul, 0x082b2b082b2b082bul, 0x082b2b19192b2b08ul,
    0x082b2b192b190808ul, 0x082b2b2b08082b08ul, 0x082b2b2b082b0808ul, 0x082b2b2b2b08082bul,
    0x082b2b2b2b082b08ul, 0x082b2b2b2b082b2bul, 0x1908080808080819ul, 0x1908080808081908ul,
    0x190808080808192bul, 0x1908080808082b19ul, 0x1908080808190808ul, 0x190808080819082bul,
    0x1908080808191919ul, 0x1908080808192b08ul, 0x19080808082b0819ul, 0x19080808082b1908ul,
    0x1908080819080808ul, 0x190808081908082bul, 0x1908080819081919ul, 0x1908080819082b08ul,
    0x1908080819082b2bul, 0x1908080819190819ul, 0x1908080819191908ul, 0x19080808192b0808ul,
    0x19080808192b1919ul, 0x190808082b080819ul, 0x190808082b081908ul, 0x190808082b190808ul,
    0x1908081908080808ul, 0x190808190808082bul, 0x1908081908081919ul, 0x1908081908082b08ul,
    0x1908081908190819ul, 0x1908081908191908ul, 0x19080819082b0808ul, 0x1908081919080819ul,
    0x1908081919081908ul, 0x1908081919190808ul, 0x190808192b080808ul, 0x190808192b081919ul,
    0x190808192b2b082bul, 0x1908082b08080819ul, 0x1908082b08081908ul, 0x1908082b08190808ul,
    0x1908082b0819082bul, 0x1908082b082b2b19ul, 0x1908082b19080808ul, 0x1908190808080808ul,
    0x190819080808082bul, 0x1908190808081919ul, 0x1908190808082b08ul, 0x1908190808190819ul,
    0x1908190808191908ul, 0x1908190808192b19ul, 0x19081908082b0808ul, 0x1908190819080819ul,
    0x1908190819081908ul, 0x1908190819190808ul, 0x190819082b080808ul, 0x190819082b191908ul,
    0x1908191908080819ul, 0x1908191908081908ul, 0x1908191908190808ul, 0x19081919082b1908ul,
    0x1908191919080808ul, 0x190819192b192b2bul, 0x1908192b08080808ul, 0x1908192b08082b2bul,
    0x1908192b19081908ul, 0x1908192b19190808ul, 0x19082b0808080819ul, 0x19082b0808081908ul,
    0x19082b0808190808ul, 0x19082b0819080808ul, 0x19082b0819081919ul, 0x19082b0819191908ul,
    0x19082b08192b082bul, 0x19082b1908080808ul, 0x19082b1908190819ul, 0x19082b1919081908ul,
    0x19082b1919190808ul, 0x19082b19192b2b19ul, 0x19082b2b08081908ul, 0x1919080808080808ul,
    0x191908080808082bul, 0x1919080808081919ul, 0x1919080808082b08ul, 0x1919080808190819ul,
    0x1919080808191908ul, 0x19190808082b0808ul, 0x19190808082b2b08ul, 0x1919080819080819ul,
    0x1919080819081908ul, 0x1919080819190808ul, 0x191908082b080808ul, 0x1919081908080819ul,
    0x1919081908081908ul, 0x1919081908190808ul, 0x1919081908191919ul, 0x1919081919080808ul,
    0x191908191908082bul, 0x1919082b08080808ul, 0x1919082b19081908ul, 0x1919082b2b2b2b2bul,
    0x1919190808080819ul, 0x1919190808081908ul, 0x1919190808190808ul, 0x19191908082b0819ul,
    0x1919190819080808ul, 0x19191908192b0808ul, 0x191919082b080819ul, 0x191919082b2b0819ul,
    0x1919191908080808ul, 0x1919191908082b08ul, 0x191919192b080808ul, 0x191919192b082b08ul,
    0x1919192b082b0819ul, 0x1919192b192b2b08ul, 0x1919192b2b2b0819ul, 0x19192b0808080808ul,
    0x19192b0808191908ul, 0x19192b0819080819ul, 0x19192b0819190808ul, 0x19192b082b192b19ul,
    0x19192b1908192b2bul, 0x19192b1919080808ul, 0x19192b191908082bul, 0x19192b2b2b081919ul,
    0x192b080808080819ul, 0x192b080808081908ul, 0x192b080808190808ul, 0x192b080819080808ul,
    0x192b080819191908ul, 0x192b0808192b082bul, 0x192b08082b08192bul, 0x192b08082b2b2b19ul,
    0x192b081908080808ul, 0x192b082b082b1908ul, 0x192b082b19082b2bul, 0x192b082b2b19082bul,
    0x192b190808080808ul, 0x192b19080819192bul, 0x192b191908190808ul, 0x192b191919080808ul,
    0x192b191919081919ul, 0x192b19192b2b1908ul, 0x192b2b0808080819ul, 0x192b2b08192b2b2bul,
    0x192b2b19082b1919ul, 0x192b2b2b0808192bul, 0x192b2b2b19191908ul, 0x192b2b2b192b082bul,
    0x2b08080808080808ul, 0x2b0808080808082bul, 0x2b08080808081919ul, 0x2b08080808082b08ul,
    0x2b08080808190819ul, 0x2b08080808191908ul, 0x2b080808082b0808ul, 0x2b080808082b2b2bul,
    0x2b08080819080819ul, 0x2b08080819081908ul, 0x2b08080819190808ul, 0x2b0808082b080808ul,
    0x2b0808082b08082bul, 0x2b0808082b2b2b08ul, 0x2b0808082b2b2b2bul, 0x2b08081908080819ul,
    0x2b08081908081908ul, 0x2b0808190808192bul, 0x2b08081908190808ul, 0x2b08081919080808ul,
    0x2b08081919190819ul, 0x2b08081919192b19ul, 0x2b08082b08080808ul, 0x2b08082b082b0808ul,
    0x2b08082b2b080808ul, 0x2b08082b2b08082bul, 0x2b08082b2b2b0808ul, 0x2b08082b2b2b2b08ul,
    0x2b08190808080819ul, 0x2b08190808081908ul, 0x2b08190808190808ul, 0x2b0819080819082bul,
    0x2b08190808191919ul, 0x2b08190819080808ul, 0x2b081908192b0808ul, 0x2b0819082b082b19ul,
    0x2b08191908080808ul, 0x2b08191919081908ul, 0x2b0819192b2b1919ul, 0x2b08192b08192b08ul,
    0x2b08192b192b2b2bul, 0x2b082b0808080808ul, 0x2b082b0808082b08ul, 0x2b082b08082b1919ul,
    0x2b082b0819192b2bul, 0x2b082b082b080808ul, 0x2b082b082b08082bul, 0x2b082b082b2b2b08ul,
    0x2b082b190808192bul, 0x2b082b2b082b082bul, 0x2b082b2b2b080808ul, 0x2b082b2b2b082b08ul,
    0x2b082b2b2b19192bul, 0x2b082b2b2b2b2b08ul, 0x2b19080808080819ul, 0x2b19080808081908ul,
    0x2b19080808190808ul, 0x2b19080819080808ul, 0x2b1908081919192bul, 0x2b1908082b081908ul,
    0x2b19081908080808ul, 0x2b190819082b082bul, 0x2b190819192b1908ul, 0x2b19082b1919192bul,
    0x2b19082b2b082b19ul, 0x2b19190808080808ul, 0x2b19190808081919ul, 0x2b19190819081908ul,
    0x2b19190819190808ul, 0x2b19190819192b08ul, 0x2b191919082b2b19ul, 0x2b1919192b190808ul,
    0x2b1919192b19082bul, 0x2b19192b19080819ul, 0x2b192b0819190819ul, 0x2b192b082b2b192bul,
    0x2b192b1919082b19ul, 0x2b192b2b08191919ul, 0x2b192b2b192b0808ul, 0x2b2b080808080808ul,
    0x2b2b08080808082bul, 0x2b2b080808082b08ul, 0x2b2b080808082b2bul, 0x2b2b0808082b0808ul,
    0x2b2b0808082b2b2bul, 0x2b2b08082b2b0808ul, 0x2b2b081919190819ul, 0x2b2b081919192b19ul,
    0x2b2b08192b2b192bul, 0x2b2b082b08080808ul, 0x2b2b082b0808082bul, 0x2b2b082b08082b08ul,
    0x2b2b082b082b2b2bul, 0x2b2b082b2b080808ul, 0x2b2b082b2b2b0808ul, 0x2b2b190819080808ul,
    0x2b2b19082b191919ul, 0x2b2b192b192b1919ul, 0x2b2b192b2b192b08ul, 0x2b2b2b0808082b2bul,
    0x2b2b2b08082b0808ul, 0x2b2b2b08082b082bul, 0x2b2b2b08082b2b08ul, 0x2b2b2b082b2b0808ul,
    0x2b2b2b082b2b2b08ul, 0x2b2b2b1908081908ul, 0x2b2b2b192b081908ul, 0x2b2b2b192b08192bul,
    0x2b2b2b2b082b2b08ul, 0x2b2b2b2b082b2b2bul, 0x2b2b2b2b2b190819ul, 0x2b2b2b2b2b2b2b2bul,
};

// IQ2_S's codebook: 1024 entries of eight packed byte levels (a 10-bit index).
constant ulong IQ2S_GRID[1024] = {
    0x0808080808080808ul, 0x080808080808082bul, 0x0808080808081919ul, 0x0808080808082b08ul,
    0x0808080808082b2bul, 0x0808080808190819ul, 0x0808080808191908ul, 0x080808080819192bul,
    0x0808080808192b19ul, 0x08080808082b0808ul, 0x08080808082b082bul, 0x08080808082b1919ul,
    0x08080808082b2b08ul, 0x0808080819080819ul, 0x0808080819081908ul, 0x080808081908192bul,
    0x0808080819082b19ul, 0x0808080819190808ul, 0x080808081919082bul, 0x0808080819191919ul,
    0x0808080819192b08ul, 0x08080808192b0819ul, 0x08080808192b1908ul, 0x08080808192b192bul,
    0x08080808192b2b19ul, 0x080808082b080808ul, 0x080808082b08082bul, 0x080808082b081919ul,
    0x080808082b082b08ul, 0x080808082b190819ul, 0x080808082b191908ul, 0x080808082b2b0808ul,
    0x080808082b2b1919ul, 0x080808082b2b2b2bul, 0x0808081908080819ul, 0x0808081908081908ul,
    0x080808190808192bul, 0x0808081908082b19ul, 0x0808081908190808ul, 0x080808190819082bul,
    0x0808081908191919ul, 0x0808081908192b08ul, 0x08080819082b0819ul, 0x08080819082b1908ul,
    0x0808081919080808ul, 0x080808191908082bul, 0x0808081919081919ul, 0x0808081919082b08ul,
    0x0808081919190819ul, 0x0808081919191908ul, 0x080808191919192bul, 0x0808081919192b19ul,
    0x08080819192b0808ul, 0x08080819192b1919ul, 0x08080819192b2b08ul, 0x080808192b080819ul,
    0x080808192b081908ul, 0x080808192b190808ul, 0x080808192b19082bul, 0x080808192b191919ul,
    0x080808192b2b0819ul, 0x080808192b2b1908ul, 0x0808082b08080808ul, 0x0808082b0808082bul,
    0x0808082b08081919ul, 0x0808082b08082b08ul, 0x0808082b08190819ul, 0x0808082b08191908ul,
    0x0808082b082b0808ul, 0x0808082b082b2b2bul, 0x0808082b19080819ul, 0x0808082b19081908ul,
    0x0808082b1908192bul, 0x0808082b19082b19ul, 0x0808082b19190808ul, 0x0808082b19191919ul,
    0x0808082b2b080808ul, 0x0808082b2b081919ul, 0x0808082b2b082b2bul, 0x0808082b2b191908ul,
    0x0808082b2b2b082bul, 0x0808190808080819ul, 0x0808190808081908ul, 0x080819080808192bul,
    0x0808190808082b19ul, 0x0808190808190808ul, 0x080819080819082bul, 0x0808190808191919ul,
    0x0808190808192b08ul, 0x08081908082b0819ul, 0x08081908082b1908ul, 0x08081908082b192bul,
    0x08081908082b2b19ul, 0x0808190819080808ul, 0x080819081908082bul, 0x0808190819081919ul,
    0x0808190819082b08ul, 0x0808190819082b2bul, 0x0808190819190819ul, 0x0808190819191908ul,
    0x080819081919192bul, 0x0808190819192b19ul, 0x08081908192b0808ul, 0x08081908192b082bul,
    0x08081908192b1919ul, 0x080819082b080819ul, 0x080819082b081908ul, 0x080819082b08192bul,
    0x080819082b082b19ul, 0x080819082b190808ul, 0x080819082b191919ul, 0x080819082b192b08ul,
    0x080819082b2b0819ul, 0x080819082b2b1908ul, 0x0808191908080808ul, 0x080819190808082bul,
    0x0808191908081919ul, 0x0808191908082b08ul, 0x0808191908082b2bul, 0x0808191908190819ul,
    0x0808191908191908ul, 0x080819190819192bul, 0x0808191908192b19ul, 0x08081919082b0808ul,
    0x08081919082b1919ul, 0x08081919082b2b08ul, 0x0808191919080819ul, 0x0808191919081908ul,
    0x080819191908192bul, 0x0808191919082b19ul, 0x0808191919190808ul, 0x080819191919082bul,
    0x0808191919191919ul, 0x0808191919192b08ul, 0x08081919192b0819ul, 0x08081919192b1908ul,
    0x080819192b080808ul, 0x080819192b08082bul, 0x080819192b081919ul, 0x080819192b082b08ul,
    0x080819192b190819ul, 0x080819192b191908ul, 0x080819192b2b0808ul, 0x0808192b08080819ul,
    0x0808192b08081908ul, 0x0808192b0808192bul, 0x0808192b08082b19ul, 0x0808192b08190808ul,
    0x0808192b08191919ul, 0x0808192b19080808ul, 0x0808192b19081919ul, 0x0808192b19082b08ul,
    0x0808192b19190819ul, 0x0808192b19191908ul, 0x0808192b192b0808ul, 0x0808192b2b080819ul,
    0x0808192b2b081908ul, 0x0808192b2b190808ul, 0x08082b0808080808ul, 0x08082b080808082bul,
    0x08082b0808081919ul, 0x08082b0808082b08ul, 0x08082b0808190819ul, 0x08082b0808191908ul,
    0x08082b080819192bul, 0x08082b0808192b19ul, 0x08082b08082b0808ul, 0x08082b08082b1919ul,
    0x08082b08082b2b2bul, 0x08082b0819080819ul, 0x08082b0819081908ul, 0x08082b081908192bul,
    0x08082b0819082b19ul, 0x08082b0819190808ul, 0x08082b081919082bul, 0x08082b0819191919ul,
    0x08082b0819192b08ul, 0x08082b08192b0819ul, 0x08082b08192b1908ul, 0x08082b082b080808ul,
    0x08082b082b081919ul, 0x08082b082b191908ul, 0x08082b082b2b2b2bul, 0x08082b1908080819ul,
    0x08082b1908081908ul, 0x08082b1908190808ul, 0x08082b190819082bul, 0x08082b1908191919ul,
    0x08082b1908192b08ul, 0x08082b19082b0819ul, 0x08082b1919080808ul, 0x08082b1919081919ul,
    0x08082b1919082b08ul, 0x08082b1919190819ul, 0x08082b1919191908ul, 0x08082b19192b0808ul,
    0x08082b192b080819ul, 0x08082b192b190808ul, 0x08082b2b08080808ul, 0x08082b2b08190819ul,
    0x08082b2b08191908ul, 0x08082b2b082b082bul, 0x08082b2b082b2b08ul, 0x08082b2b082b2b2bul,
    0x08082b2b19190808ul, 0x08082b2b2b192b19ul, 0x0819080808080819ul, 0x0819080808081908ul,
    0x081908080808192bul, 0x0819080808082b19ul, 0x0819080808190808ul, 0x081908080819082bul,
    0x0819080808191919ul, 0x0819080808192b08ul, 0x08190808082b0819ul, 0x08190808082b1908ul,
    0x08190808082b192bul, 0x0819080819080808ul, 0x081908081908082bul, 0x0819080819081919ul,
    0x0819080819082b08ul, 0x0819080819190819ul, 0x0819080819191908ul, 0x081908081919192bul,
    0x0819080819192b19ul, 0x08190808192b0808ul, 0x08190808192b082bul, 0x08190808192b1919ul,
    0x08190808192b2b08ul, 0x081908082b080819ul, 0x081908082b081908ul, 0x081908082b08192bul,
    0x081908082b190808ul, 0x081908082b191919ul, 0x081908082b192b08ul, 0x081908082b2b0819ul,
    0x081908082b2b1908ul, 0x0819081908080808ul, 0x081908190808082bul, 0x0819081908081919ul,
    0x0819081908082b08ul, 0x0819081908082b2bul, 0x0819081908190819ul, 0x0819081908191908ul,
    0x081908190819192bul, 0x0819081908192b19ul, 0x08190819082b0808ul, 0x08190819082b082bul,
    0x08190819082b1919ul, 0x08190819082b2b08ul, 0x0819081919080819ul, 0x0819081919081908ul,
    0x081908191908192bul, 0x0819081919082b19ul, 0x0819081919190808ul, 0x081908191919082bul,
    0x0819081919191919ul, 0x0819081919192b08ul, 0x08190819192b0819ul, 0x08190819192b1908ul,
    0x081908192b080808ul, 0x081908192b08082bul, 0x081908192b081919ul, 0x081908192b082b08ul,
    0x081908192b190819ul, 0x081908192b191908ul, 0x0819082b08080819ul, 0x0819082b08081908ul,
    0x0819082b08082b19ul, 0x0819082b08190808ul, 0x0819082b08191919ul, 0x0819082b082b0819ul,
    0x0819082b082b1908ul, 0x0819082b19080808ul, 0x0819082b19081919ul, 0x0819082b19190819ul,
    0x0819082b19191908ul, 0x0819082b2b080819ul, 0x0819082b2b081908ul, 0x0819082b2b190808ul,
    0x0819190808080808ul, 0x081919080808082bul, 0x0819190808081919ul, 0x0819190808082b08ul,
    0x0819190808190819ul, 0x0819190808191908ul, 0x081919080819192bul, 0x0819190808192b19ul,
    0x08191908082b0808ul, 0x08191908082b1919ul, 0x08191908082b2b08ul, 0x0819190819080819ul,
    0x0819190819081908ul, 0x081919081908192bul, 0x0819190819082b19ul, 0x0819190819190808ul,
    0x081919081919082bul, 0x0819190819191919ul, 0x0819190819192b08ul, 0x08191908192b0819ul,
    0x08191908192b1908ul, 0x081919082b080808ul, 0x081919082b08082bul, 0x081919082b081919ul,
    0x081919082b082b08ul, 0x081919082b190819ul, 0x081919082b191908ul, 0x081919082b2b0808ul,
    0x0819191908080819ul, 0x0819191908081908ul, 0x081919190808192bul, 0x0819191908082b19ul,
    0x0819191908190808ul, 0x081919190819082bul, 0x0819191908191919ul, 0x0819191908192b08ul,
    0x08191919082b0819ul, 0x08191919082b1908ul, 0x0819191919080808ul, 0x081919191908082bul,
    0x0819191919081919ul, 0x0819191919082b08ul, 0x0819191919190819ul, 0x0819191919191908ul,
    0x08191919192b0808ul, 0x081919192b080819ul, 0x081919192b081908ul, 0x081919192b190808ul,
    0x0819192b08080808ul, 0x0819192b08081919ul, 0x0819192b08082b08ul, 0x0819192b08190819ul,
    0x0819192b08191908ul, 0x0819192b082b0808ul, 0x0819192b19080819ul, 0x0819192b19081908ul,
    0x0819192b19190808ul, 0x0819192b2b080808ul, 0x0819192b2b2b2b2bul, 0x08192b0808080819ul,
    0x08192b0808081908ul, 0x08192b080808192bul, 0x08192b0808082b19ul, 0x08192b0808190808ul,
    0x08192b0808191919ul, 0x08192b0808192b08ul, 0x08192b08082b0819ul, 0x08192b0819080808ul,
    0x08192b081908082bul, 0x08192b0819081919ul, 0x08192b0819082b08ul, 0x08192b0819190819ul,
    0x08192b0819191908ul, 0x08192b08192b0808ul, 0x08192b082b080819ul, 0x08192b082b081908ul,
    0x08192b1908080808ul, 0x08192b190808082bul, 0x08192b1908081919ul, 0x08192b1908082b08ul,
    0x08192b1908190819ul, 0x08192b1908191908ul, 0x08192b19082b0808ul, 0x08192b1919080819ul,
    0x08192b1919081908ul, 0x08192b1919190808ul, 0x08192b19192b2b19ul, 0x08192b192b2b082bul,
    0x08192b2b08081908ul, 0x08192b2b08190808ul, 0x08192b2b19080808ul, 0x08192b2b1919192bul,
    0x082b080808080808ul, 0x082b08080808082bul, 0x082b080808081919ul, 0x082b080808082b08ul,
    0x082b080808190819ul, 0x082b080808191908ul, 0x082b08080819192bul, 0x082b080808192b19ul,
    0x082b0808082b0808ul, 0x082b0808082b1919ul, 0x082b0808082b2b2bul, 0x082b080819080819ul,
    0x082b080819081908ul, 0x082b080819190808ul, 0x082b08081919082bul, 0x082b080819191919ul,
    0x082b0808192b1908ul, 0x082b08082b080808ul, 0x082b08082b082b2bul, 0x082b08082b191908ul,
    0x082b08082b2b2b2bul, 0x082b081908080819ul, 0x082b081908081908ul, 0x082b081908190808ul,
    0x082b08190819082bul, 0x082b081908191919ul, 0x082b0819082b0819ul, 0x082b081919080808ul,
    0x082b08191908082bul, 0x082b081919081919ul, 0x082b081919190819ul, 0x082b081919191908ul,
    0x082b0819192b0808ul, 0x082b08192b080819ul, 0x082b08192b081908ul, 0x082b08192b190808ul,
    0x082b082b08080808ul, 0x082b082b08082b2bul, 0x082b082b082b082bul, 0x082b082b082b2b08ul,
    0x082b082b082b2b2bul, 0x082b082b19081908ul, 0x082b082b19190808ul, 0x082b082b2b082b08ul,
    0x082b082b2b082b2bul, 0x082b082b2b2b2b08ul, 0x082b190808080819ul, 0x082b190808081908ul,
    0x082b19080808192bul, 0x082b190808082b19ul, 0x082b190808190808ul, 0x082b190808191919ul,
    0x082b190808192b08ul, 0x082b1908082b0819ul, 0x082b1908082b1908ul, 0x082b190819080808ul,
    0x082b19081908082bul, 0x082b190819081919ul, 0x082b190819082b08ul, 0x082b190819190819ul,
    0x082b190819191908ul, 0x082b1908192b0808ul, 0x082b19082b080819ul, 0x082b19082b081908ul,
    0x082b19082b190808ul, 0x082b191908080808ul, 0x082b191908081919ul, 0x082b191908082b08ul,
    0x082b191908190819ul, 0x082b191908191908ul, 0x082b1919082b0808ul, 0x082b191919080819ul,
    0x082b191919081908ul, 0x082b191919190808ul, 0x082b1919192b192bul, 0x082b19192b080808ul,
    0x082b192b08080819ul, 0x082b192b08081908ul, 0x082b192b08190808ul, 0x082b192b19080808ul,
    0x082b192b19192b19ul, 0x082b2b0808080808ul, 0x082b2b0808081919ul, 0x082b2b0808190819ul,
    0x082b2b0808191908ul, 0x082b2b0819080819ul, 0x082b2b0819081908ul, 0x082b2b0819190808ul,
    0x082b2b082b082b2bul, 0x082b2b082b2b2b2bul, 0x082b2b1908080819ul, 0x082b2b1908081908ul,
    0x082b2b1908190808ul, 0x082b2b192b191919ul, 0x082b2b2b08082b2bul, 0x082b2b2b082b082bul,
    0x082b2b2b192b1908ul, 0x082b2b2b2b082b08ul, 0x082b2b2b2b082b2bul, 0x1908080808080819ul,
    0x1908080808081908ul, 0x190808080808192bul, 0x1908080808082b19ul, 0x1908080808190808ul,
    0x190808080819082bul, 0x1908080808191919ul, 0x1908080808192b08ul, 0x1908080808192b2bul,
    0x19080808082b0819ul, 0x19080808082b1908ul, 0x19080808082b192bul, 0x1908080819080808ul,
    0x190808081908082bul, 0x1908080819081919ul, 0x1908080819082b08ul, 0x1908080819082b2bul,
    0x1908080819190819ul, 0x1908080819191908ul, 0x190808081919192bul, 0x1908080819192b19ul,
    0x19080808192b0808ul, 0x19080808192b082bul, 0x19080808192b1919ul, 0x190808082b080819ul,
    0x190808082b081908ul, 0x190808082b190808ul, 0x190808082b191919ul, 0x190808082b192b08ul,
    0x190808082b2b0819ul, 0x190808082b2b1908ul, 0x1908081908080808ul, 0x190808190808082bul,
    0x1908081908081919ul, 0x1908081908082b08ul, 0x1908081908190819ul, 0x1908081908191908ul,
    0x190808190819192bul, 0x1908081908192b19ul, 0x19080819082b0808ul, 0x19080819082b082bul,
    0x19080819082b1919ul, 0x1908081919080819ul, 0x1908081919081908ul, 0x190808191908192bul,
    0x1908081919082b19ul, 0x1908081919190808ul, 0x190808191919082bul, 0x1908081919191919ul,
    0x1908081919192b08ul, 0x19080819192b0819ul, 0x19080819192b1908ul, 0x190808192b080808ul,
    0x190808192b08082bul, 0x190808192b081919ul, 0x190808192b082b08ul, 0x190808192b190819ul,
    0x190808192b191908ul, 0x190808192b2b0808ul, 0x1908082b08080819ul, 0x1908082b08081908ul,
    0x1908082b08190808ul, 0x1908082b0819082bul, 0x1908082b08191919ul, 0x1908082b08192b08ul,
    0x1908082b082b1908ul, 0x1908082b19080808ul, 0x1908082b19081919ul, 0x1908082b19082b08ul,
    0x1908082b19190819ul, 0x1908082b19191908ul, 0x1908082b192b0808ul, 0x1908082b2b080819ul,
    0x1908082b2b081908ul, 0x1908190808080808ul, 0x190819080808082bul, 0x1908190808081919ul,
    0x1908190808082b08ul, 0x1908190808082b2bul, 0x1908190808190819ul, 0x1908190808191908ul,
    0x190819080819192bul, 0x1908190808192b19ul, 0x19081908082b0808ul, 0x19081908082b082bul,
    0x19081908082b1919ul, 0x19081908082b2b08ul, 0x1908190819080819ul, 0x1908190819081908ul,
    0x190819081908192bul, 0x1908190819082b19ul, 0x1908190819190808ul, 0x190819081919082bul,
    0x1908190819191919ul, 0x1908190819192b08ul, 0x19081908192b0819ul, 0x19081908192b1908ul,
    0x190819082b080808ul, 0x190819082b08082bul, 0x190819082b081919ul, 0x190819082b082b08ul,
    0x190819082b190819ul, 0x190819082b191908ul, 0x190819082b2b0808ul, 0x1908191908080819ul,
    0x1908191908081908ul, 0x190819190808192bul, 0x1908191908082b19ul, 0x1908191908190808ul,
    0x190819190819082bul, 0x1908191908191919ul, 0x1908191908192b08ul, 0x19081919082b0819ul,
    0x19081919082b1908ul, 0x1908191919080808ul, 0x190819191908082bul, 0x1908191919081919ul,
    0x1908191919082b08ul, 0x1908191919190819ul, 0x1908191919191908ul, 0x19081919192b0808ul,
    0x19081919192b2b2bul, 0x190819192b080819ul, 0x190819192b081908ul, 0x190819192b190808ul,
    0x1908192b08080808ul, 0x1908192b0808082bul, 0x1908192b08081919ul, 0x1908192b08082b08ul,
    0x1908192b08190819ul, 0x1908192b08191908ul, 0x1908192b082b0808ul, 0x1908192b19080819ul,
    0x1908192b19081908ul, 0x1908192b19190808ul, 0x1908192b2b080808ul, 0x1908192b2b2b1919ul,
    0x19082b0808080819ul, 0x19082b0808081908ul, 0x19082b0808082b19ul, 0x19082b0808190808ul,
    0x19082b080819082bul, 0x19082b0808191919ul, 0x19082b0808192b08ul, 0x19082b08082b0819ul,
    0x19082b08082b1908ul, 0x19082b0819080808ul, 0x19082b081908082bul, 0x19082b0819081919ul,
    0x19082b0819082b08ul, 0x19082b0819190819ul, 0x19082b0819191908ul, 0x19082b08192b0808ul,
    0x19082b082b081908ul, 0x19082b082b190808ul, 0x19082b1908080808ul, 0x19082b190808082bul,
    0x19082b1908081919ul, 0x19082b1908082b08ul, 0x19082b1908190819ul, 0x19082b1908191908ul,
    0x19082b19082b0808ul, 0x19082b1919080819ul, 0x19082b1919081908ul, 0x19082b1919190808ul,
    0x19082b192b080808ul, 0x19082b192b19192bul, 0x19082b2b08080819ul, 0x19082b2b08081908ul,
    0x19082b2b08190808ul, 0x19082b2b19080808ul, 0x1919080808080808ul, 0x191908080808082bul,
    0x1919080808081919ul, 0x1919080808082b08ul, 0x1919080808190819ul, 0x1919080808191908ul,
    0x191908080819192bul, 0x1919080808192b19ul, 0x19190808082b0808ul, 0x19190808082b082bul,
    0x19190808082b1919ul, 0x19190808082b2b08ul, 0x1919080819080819ul, 0x1919080819081908ul,
    0x191908081908192bul, 0x1919080819082b19ul, 0x1919080819190808ul, 0x191908081919082bul,
    0x1919080819191919ul, 0x1919080819192b08ul, 0x19190808192b0819ul, 0x19190808192b1908ul,
    0x191908082b080808ul, 0x191908082b08082bul, 0x191908082b081919ul, 0x191908082b082b08ul,
    0x191908082b190819ul, 0x191908082b191908ul, 0x1919081908080819ul, 0x1919081908081908ul,
    0x191908190808192bul, 0x1919081908082b19ul, 0x1919081908190808ul, 0x191908190819082bul,
    0x1919081908191919ul, 0x1919081908192b08ul, 0x19190819082b0819ul, 0x19190819082b1908ul,
    0x1919081919080808ul, 0x191908191908082bul, 0x1919081919081919ul, 0x1919081919082b08ul,
    0x1919081919190819ul, 0x1919081919191908ul, 0x19190819192b0808ul, 0x191908192b080819ul,
    0x191908192b081908ul, 0x191908192b190808ul, 0x1919082b08080808ul, 0x1919082b08081919ul,
    0x1919082b08082b08ul, 0x1919082b08190819ul, 0x1919082b08191908ul, 0x1919082b082b0808ul,
    0x1919082b19080819ul, 0x1919082b19081908ul, 0x1919082b19190808ul, 0x1919082b192b2b19ul,
    0x1919082b2b080808ul, 0x1919190808080819ul, 0x1919190808081908ul, 0x191919080808192bul,
    0x1919190808082b19ul, 0x1919190808190808ul, 0x191919080819082bul, 0x1919190808191919ul,
    0x1919190808192b08ul, 0x19191908082b0819ul, 0x19191908082b1908ul, 0x1919190819080808ul,
    0x191919081908082bul, 0x1919190819081919ul, 0x1919190819082b08ul, 0x1919190819190819ul,
    0x1919190819191908ul, 0x19191908192b0808ul, 0x191919082b080819ul, 0x191919082b081908ul,
    0x191919082b190808ul, 0x1919191908080808ul, 0x191919190808082bul, 0x1919191908081919ul,
    0x1919191908082b08ul, 0x1919191908190819ul, 0x1919191908191908ul, 0x19191919082b0808ul,
    0x1919191919080819ul, 0x1919191919081908ul, 0x1919191919190808ul, 0x191919192b080808ul,
    0x1919192b08080819ul, 0x1919192b08081908ul, 0x1919192b08190808ul, 0x1919192b082b192bul,
    0x1919192b19080808ul, 0x19192b0808080808ul, 0x19192b080808082bul, 0x19192b0808081919ul,
    0x19192b0808082b08ul, 0x19192b0808190819ul, 0x19192b0808191908ul, 0x19192b08082b0808ul,
    0x19192b0819080819ul, 0x19192b0819081908ul, 0x19192b0819190808ul, 0x19192b0819192b2bul,
    0x19192b082b080808ul, 0x19192b1908080819ul, 0x19192b1908081908ul, 0x19192b1908190808ul,
    0x19192b1919080808ul, 0x19192b2b08080808ul, 0x19192b2b08192b19ul, 0x19192b2b2b081919ul,
    0x19192b2b2b2b2b08ul, 0x192b080808080819ul, 0x192b080808081908ul, 0x192b08080808192bul,
    0x192b080808190808ul, 0x192b08080819082bul, 0x192b080808191919ul, 0x192b080808192b08ul,
    0x192b0808082b0819ul, 0x192b0808082b1908ul, 0x192b080819080808ul, 0x192b080819081919ul,
    0x192b080819082b08ul, 0x192b080819190819ul, 0x192b080819191908ul, 0x192b0808192b0808ul,
    0x192b08082b081908ul, 0x192b08082b190808ul, 0x192b081908080808ul, 0x192b08190808082bul,
    0x192b081908081919ul, 0x192b081908082b08ul, 0x192b081908190819ul, 0x192b081908191908ul,
    0x192b0819082b0808ul, 0x192b081919080819ul, 0x192b081919081908ul, 0x192b081919190808ul,
    0x192b08192b080808ul, 0x192b08192b192b19ul, 0x192b082b08081908ul, 0x192b082b08190808ul,
    0x192b082b19080808ul, 0x192b082b1919192bul, 0x192b082b2b2b0819ul, 0x192b190808080808ul,
    0x192b190808081919ul, 0x192b190808082b08ul, 0x192b190808190819ul, 0x192b190808191908ul,
    0x192b1908082b0808ul, 0x192b190819080819ul, 0x192b190819081908ul, 0x192b190819190808ul,
    0x192b19082b080808ul, 0x192b191908080819ul, 0x192b191908081908ul, 0x192b191908190808ul,
    0x192b191919080808ul, 0x192b191919082b2bul, 0x192b1919192b2b08ul, 0x192b19192b19082bul,
    0x192b192b08080808ul, 0x192b192b2b191908ul, 0x192b2b0808080819ul, 0x192b2b0808081908ul,
    0x192b2b0808190808ul, 0x192b2b08192b1919ul, 0x192b2b082b192b08ul, 0x192b2b1908080808ul,
    0x192b2b19082b2b2bul, 0x192b2b2b1908082bul, 0x192b2b2b2b2b0819ul, 0x2b08080808080808ul,
    0x2b0808080808082bul, 0x2b08080808081919ul, 0x2b08080808082b08ul, 0x2b08080808190819ul,
    0x2b08080808191908ul, 0x2b08080808192b19ul, 0x2b080808082b0808ul, 0x2b080808082b1919ul,
    0x2b08080819080819ul, 0x2b08080819081908ul, 0x2b08080819190808ul, 0x2b0808081919082bul,
    0x2b08080819191919ul, 0x2b08080819192b08ul, 0x2b080808192b0819ul, 0x2b0808082b080808ul,
    0x2b0808082b081919ul, 0x2b0808082b190819ul, 0x2b0808082b191908ul, 0x2b08081908080819ul,
    0x2b08081908081908ul, 0x2b08081908082b19ul, 0x2b08081908190808ul, 0x2b0808190819082bul,
    0x2b08081908191919ul, 0x2b08081908192b08ul, 0x2b080819082b0819ul, 0x2b080819082b1908ul,
    0x2b08081919080808ul, 0x2b0808191908082bul, 0x2b08081919081919ul, 0x2b08081919082b08ul,
    0x2b08081919190819ul, 0x2b08081919191908ul, 0x2b0808192b080819ul, 0x2b0808192b081908ul,
    0x2b0808192b190808ul, 0x2b0808192b2b2b19ul, 0x2b08082b08080808ul, 0x2b08082b08081919ul,
    0x2b08082b08082b2bul, 0x2b08082b08190819ul, 0x2b08082b08191908ul, 0x2b08082b19080819ul,
    0x2b08082b19081908ul, 0x2b08082b19190808ul, 0x2b08190808080819ul, 0x2b08190808081908ul,
    0x2b0819080808192bul, 0x2b08190808082b19ul, 0x2b08190808190808ul, 0x2b0819080819082bul,
    0x2b08190808191919ul, 0x2b08190808192b08ul, 0x2b081908082b0819ul, 0x2b08190819080808ul,
    0x2b0819081908082bul, 0x2b08190819081919ul, 0x2b08190819082b08ul, 0x2b08190819190819ul,
    0x2b08190819191908ul, 0x2b081908192b0808ul, 0x2b0819082b080819ul, 0x2b0819082b081908ul,
    0x2b0819082b190808ul, 0x2b08191908080808ul, 0x2b0819190808082bul, 0x2b08191908081919ul,
    0x2b08191908082b08ul, 0x2b08191908190819ul, 0x2b08191908191908ul, 0x2b081919082b0808ul,
    0x2b08191919080819ul, 0x2b08191919081908ul, 0x2b08191919190808ul, 0x2b0819192b080808ul,
    0x2b0819192b082b2bul, 0x2b08192b08080819ul, 0x2b08192b08081908ul, 0x2b08192b08190808ul,
    0x2b08192b082b2b19ul, 0x2b08192b19080808ul, 0x2b082b0808080808ul, 0x2b082b0808081919ul,
    0x2b082b0808190819ul, 0x2b082b0808191908ul, 0x2b082b0819080819ul, 0x2b082b0819081908ul,
    0x2b082b0819190808ul, 0x2b082b082b2b082bul, 0x2b082b1908080819ul, 0x2b082b1908081908ul,
    0x2b082b1919080808ul, 0x2b082b19192b1919ul, 0x2b082b2b082b082bul, 0x2b082b2b19192b08ul,
    0x2b082b2b19192b2bul, 0x2b082b2b2b08082bul, 0x2b082b2b2b2b082bul, 0x2b19080808080819ul,
    0x2b19080808081908ul, 0x2b19080808082b19ul, 0x2b19080808190808ul, 0x2b1908080819082bul,
    0x2b19080808191919ul, 0x2b19080808192b08ul, 0x2b190808082b1908ul, 0x2b19080819080808ul,
    0x2b1908081908082bul, 0x2b19080819081919ul, 0x2b19080819082b08ul, 0x2b19080819190819ul,
    0x2b19080819191908ul, 0x2b190808192b0808ul, 0x2b1908082b080819ul, 0x2b1908082b081908ul,
    0x2b1908082b190808ul, 0x2b19081908080808ul, 0x2b19081908081919ul, 0x2b19081908190819ul,
    0x2b19081908191908ul, 0x2b19081919080819ul, 0x2b19081919081908ul, 0x2b19081919190808ul,
    0x2b19081919192b2bul, 0x2b19082b08080819ul, 0x2b19082b08081908ul, 0x2b19082b08190808ul,
    0x2b19082b19080808ul, 0x2b19082b2b2b192bul, 0x2b19190808080808ul, 0x2b1919080808082bul,
    0x2b19190808081919ul, 0x2b19190808082b08ul, 0x2b19190808190819ul, 0x2b19190808191908ul,
    0x2b191908082b0808ul, 0x2b19190819080819ul, 0x2b19190819081908ul, 0x2b19190819190808ul,
    0x2b1919082b080808ul, 0x2b1919082b19192bul, 0x2b19191908080819ul, 0x2b19191908081908ul,
    0x2b19191908190808ul, 0x2b19191919080808ul, 0x2b1919192b192b08ul, 0x2b1919192b2b0819ul,
    0x2b19192b08080808ul, 0x2b19192b1908192bul, 0x2b19192b192b1908ul, 0x2b192b0808080819ul,
    0x2b192b0808081908ul, 0x2b192b0808190808ul, 0x2b192b08082b192bul, 0x2b192b0819080808ul,
    0x2b192b082b2b2b19ul, 0x2b192b1908080808ul, 0x2b192b1919082b19ul, 0x2b192b191919082bul,
    0x2b192b2b2b190808ul, 0x2b2b080808080808ul, 0x2b2b080808081919ul, 0x2b2b080808082b2bul,
    0x2b2b080808191908ul, 0x2b2b0808082b082bul, 0x2b2b0808082b2b2bul, 0x2b2b080819080819ul,
    0x2b2b080819081908ul, 0x2b2b080819190808ul, 0x2b2b08082b2b082bul, 0x2b2b08082b2b2b2bul,
    0x2b2b081919080808ul, 0x2b2b0819192b1919ul, 0x2b2b082b0808082bul, 0x2b2b082b08082b2bul,
    0x2b2b082b082b082bul, 0x2b2b082b082b2b08ul, 0x2b2b082b082b2b2bul, 0x2b2b082b2b08082bul,
    0x2b2b082b2b082b08ul, 0x2b2b082b2b082b2bul, 0x2b2b082b2b2b2b08ul, 0x2b2b190808080819ul,
    0x2b2b190808081908ul, 0x2b2b190808190808ul, 0x2b2b190819080808ul, 0x2b2b19082b082b19ul,
    0x2b2b19082b2b1908ul, 0x2b2b191908080808ul, 0x2b2b191908192b19ul, 0x2b2b192b19190819ul,
    0x2b2b2b0808082b2bul, 0x2b2b2b08082b2b08ul, 0x2b2b2b082b2b082bul, 0x2b2b2b1919191908ul,
    0x2b2b2b192b08192bul, 0x2b2b2b2b08082b08ul, 0x2b2b2b2b08082b2bul, 0x2b2b2b2b082b0808ul,
    0x2b2b2b2b082b082bul, 0x2b2b2b2b082b2b08ul, 0x2b2b2b2b2b082b08ul, 0x2b2b2b2b2b2b2b2bul,
};

// IQ3_XXS's codebook: 256 entries of four packed byte levels.
constant uint IQ3XXS_GRID[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

// The 32 values of SUB-BLOCK `sub` of one tile-major block, as T.
//
// T is only the final conversion: every arm computes the value in float, and the consumer
// picks T. The prefill GEMM stages half, because its matrix multiply works in half and one
// rounding is the price of that. The decode GEMV, the row gather and the mega-kernel unit
// stage float, because what they do with the value is float; a half there would round the
// weight and buy nothing (docs/dequant-precision.md).
//
// `sc` is the row's scale bytes at the unit head, `pay` its payload bytes. For a 32-element
// format `sub` is always 0; for a 256-element super-block it selects one of eight.
template <uint F, typename T>
inline void tm_sub32_t(device const uchar * sc, device const uchar * pay, uint sub,
                       thread T out[32], bool a16)
{
    // PROBE (diagnostic only, IMPARO_WFMT_PROBE): remove the UNPACK ARITHMETIC and keep
    // the TRAFFIC. It reads the same bytes the real path reads -- the scale header and the
    // sub-block's 32 payload bytes -- and does one add per value instead of the format's
    // shift/mask/affine work.
    //
    // IT USED TO WRITE A CONSTANT AND TOUCH NEITHER POINTER, and that measured something
    // else entirely: with nothing consuming `sc` or `pay` the compiler deletes the loads,
    // so the arm ran with NO WEIGHT READ AT ALL. On Qwen3.8-27B that read 122 ms against
    // 48 ms (2.6x) and the difference is the model's 15 GB weight stream, not its decode
    // arithmetic -- a number that would have been recorded as "the dequant costs 60% of
    // decode" on a kernel already measured at 90% of the streaming wall. A probe that
    // deletes the thing it claims to hold fixed is worse than no probe.
    //
    // The output is deliberately WRONG (the gates are what check correctness); this arm
    // exists to be timed, never to be shipped.
    //
    // STILL NOT A VALID ISOLATION, 2026-09-09, and the check that says so is the DRAM
    // FLOOR. Qwen3.8-27B reads 15.35 GB of weights per decode step and the measured
    // ceiling is 136 GB/s, so ~113 ms is the least any arm that reads every weight can
    // take. This arm measures 48.7-49.8 ms against the real path's 121.7-122.2. Reading
    // the payload back (above) did not move it -- 48.9/47.6 before, 49.8/48.7 after -- so
    // the traffic is still missing and the difference still is NOT "the decode
    // arithmetic". Do not quote 73 ms as the dequant cost.
    //
    // WHAT TO CHECK NEXT, in order: whether qwen35 decode's weight reads actually come
    // through THIS function (the mega per-layer route has its own bricks, so a probe here
    // may cover only part of the step); then whether the compiler still sinks the loads
    // because `out[]` feeds a dot product whose other operand it can see is unused.
    if (getenv_probe_const()) {
        device const uchar * pp = pay + (sub >> 1) * 32u;
        const half hs = half(sc[0]) + half(sc[1]);
        for (uint l = 0; l < 32u; ++l) { out[l] = hs + half(pp[l]); }
        return;
    }
    if (F == WF_Q4_K) {
        // d, dmin, scales[12]. Sub-blocks pair into the low and high nibbles of one
        // 32-byte group, which is why RT_K = 64 maps onto exactly one pair.
        //
        // READ AS 16-BYTE WORDS, NOT BYTES. The header is 16 bytes and the pair's payload
        // 32, and both sit on 16-byte boundaries in either layout (a row-major block is
        // 144 bytes; the tile-major unit puts headers at slot*16 and payloads at
        // 128 + slot*128). Byte loads cost one instruction each, so a lane spent ~38 load
        // instructions on 48 bytes; three loads do the same. The nibble is then a shift
        // by 0 or 4 instead of a lane-divergent select, and the scale unpack takes its
        // bytes from the words without the j < 4 branch. Same arithmetic, same bits.
        const uint4 h = *((device const uint4 *)sc);
        const half d    = as_type<half>((ushort)(h.x & 0xFFFFu));
        const half dmin = as_type<half>((ushort)(h.x >> 16));
        uint s, m; k_scale_min(sub, h.y, h.z, h.w, s, m);
        // ONE ROUNDING, NOT THREE. The result must be half -- it feeds the MMA operands --
        // but the ARITHMETIC does not have to be. An affine format is `ds * v - off`, and
        // `off = dmin * m` is the SAME for all 32 values of a sub-block, so rounding it to
        // half puts a systematic bias on the whole sub-block; unlike random rounding, a
        // bias shared by 32 weights does not cancel in a dot product, it multiplies the sum
        // of 32 activations. The symmetric formats (Q8_0, Q4_0 -- `q * d`, no min) have no
        // such term, which is why the half dequant of task #105 was safe for them and is
        // not safe here. Computing in float and converting once costs nothing: the value
        // was going to be converted anyway.
        const float ds = float(d) * float(s), off = float(dmin) * float(m);
        device const uint4 * q4 = (device const uint4 *)(pay + (sub >> 1) * 32u);
        const uint4 p0 = q4[0], p1 = q4[1];
        const uint shift = (sub & 1u) * 4u;
        #pragma unroll
        for (uint l = 0; l < 32u; ++l) {
            const uint word = l < 16u ? p0[l >> 2] : p1[(l >> 2) & 3u];
            const uint v = (word >> ((l & 3u) * 8u + shift)) & 0xFu;
            out[l] = T(ds * float(v) - off);
        }
        return;
    }
    if (F == WF_Q5_K) {
        // Same header; the payload opens with the 32-byte high-bit plane, then qs. Read as
        // 16-byte words like Q4_K (a block is 176 bytes, the unit's payloads sit at
        // 128 + slot * 160), the fifth bit taken from the plane by a shift.
        const uint4 h = *((device const uint4 *)sc);
        const half d    = as_type<half>((ushort)(h.x & 0xFFFFu));
        const half dmin = as_type<half>((ushort)(h.x >> 16));
        uint s, m; k_scale_min(sub, h.y, h.z, h.w, s, m);
        const float ds = float(d) * float(s), off = float(dmin) * float(m);   // see Q4_K
        device const uint4 * qh4 = (device const uint4 *)pay;
        device const uint4 * q4  = (device const uint4 *)(pay + 32u + (sub >> 1) * 32u);
        const uint4 h0 = qh4[0], h1 = qh4[1], p0 = q4[0], p1 = q4[1];
        const uint shift = (sub & 1u) * 4u;
        #pragma unroll
        for (uint l = 0; l < 32u; ++l) {
            const uint bsh  = (l & 3u) * 8u;
            const uint word = l < 16u ? p0[l >> 2] : p1[(l >> 2) & 3u];
            const uint hw   = l < 16u ? h0[l >> 2] : h1[(l >> 2) & 3u];
            const uint v = ((word >> (bsh + shift)) & 0xFu) + (((hw >> (bsh + sub)) & 1u) << 4);
            out[l] = T(ds * float(v) - off);
        }
        return;
    }
    if (F == WF_Q6_K) {
        // SYMMETRIC around 32 with SIGNED 8-bit sub-scales and no min at all. The 256
        // values are two groups of 128, each four 32-value chunks; chunk c takes ql at
        // (c&1)*32, its nibble low for c < 2, and qh shifted 2*c.
        const uint n = sub >> 2, c = sub & 3u;
        const half d = as_type<half>((ushort)(sc[16] | (sc[17] << 8)));
        device const char * s8 = (device const char *)sc + n * 8u;
        const uint shift = 2u * c, sidx = 2u * c;
        // The two sub-scales once: values 0..15 take s8[sidx], 16..31 s8[sidx + 1], and
        // d * s is the same product the per-value form computed before multiplying q.
        const float ds0 = float(d) * float((int)s8[sidx]);
        const float ds1 = float(d) * float((int)s8[sidx + 1u]);
        // ds * (q - 32) as fma(ds, q, -32 ds): one rounding, -32 ds exact, same float.
        const float c0 = -32.0f * ds0, c1 = -32.0f * ds1;
        const uint shift4 = (c >= 2u) ? 4u : 0u;
        if (a16) {
            // ql's 32 bytes and qh's 32 as 16-byte words (tile-major payloads at
            // unit + 144 + slot * 192); a row-major block is 210 bytes, byte path below.
            device const uint4 * ql4 = (device const uint4 *)(pay + n * 64u + ((c & 1u) * 32u));
            device const uint4 * qh4 = (device const uint4 *)(pay + 128u + n * 32u);
            const uint4 l0 = ql4[0], l1 = ql4[1], h0 = qh4[0], h1 = qh4[1];
            #pragma unroll
            for (uint k = 0; k < 8u; ++k) {
                const uint lw = k < 4u ? l0[k] : l1[k & 3u];
                const uint hw = k < 4u ? h0[k] : h1[k & 3u];
                const float ds = k < 4u ? ds0 : ds1, cc = k < 4u ? c0 : c1;
                const uchar4 v6 = as_type<uchar4>(((lw >> shift4) & 0x0F0F0F0Fu)
                                                | (((hw >> shift) & 0x03030303u) << 4));
                out[4u * k]      = T(fma(ds, float(v6.x), cc));
                out[4u * k + 1u] = T(fma(ds, float(v6.y), cc));
                out[4u * k + 2u] = T(fma(ds, float(v6.z), cc));
                out[4u * k + 3u] = T(fma(ds, float(v6.w), cc));
            }
        } else {
            device const uchar * ql = pay + n * 64u + ((c & 1u) * 32u);
            device const uchar * qh = pay + 128u + n * 32u;
            #pragma unroll
            for (uint l = 0; l < 32u; ++l) {
                const uint v6 = ((ql[l] >> shift4) & 0xFu) | (((qh[l] >> shift) & 3u) << 4);
                out[l] = T(fma(l < 16u ? ds0 : ds1, float(v6), l < 16u ? c0 : c1));
            }
        }
        return;
    }
    if (F == WF_Q3_K) {
        // scales[12] then d at the unit head; hmask[32] then qs[64] in the payload. The
        // third bit is INVERTED: a CLEAR hmask bit subtracts 4.
        const uint n = sub >> 2, j = sub & 3u;
        // The 14 header bytes as seven 16-bit words: the scale run starts on an even byte
        // in both layouts (tile-major slot * 14; row-major block 110, scales at 96).
        device const ushort * sh = (device const ushort *)sc;
        const half d = as_type<half>(sh[6]);
        // The 12 bytes hold 16 six-bit fields: four low nibbles per word, top two bits of
        // every field in the last word.
        uint aux[4];
        aux[0] = (uint)sh[0] | ((uint)sh[1] << 16);
        aux[1] = (uint)sh[2] | ((uint)sh[3] << 16);
        aux[2] = (uint)sh[4] | ((uint)sh[5] << 16);
        const uint km1 = 0x03030303u, km2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        aux[0] = (aux[0] & km2)        | (((tmp     ) & km1) << 4);
        aux[1] = (aux[1] & km2)        | (((tmp >> 2) & km1) << 4);
        // THE TWO SUB-SCALES ONCE, NOT PER VALUE. Values 0..15 take field is0 and 16..31
        // the next one; is0 is even, so both fields sit in one word, picked by selects
        // and split by a shift. Indexing aux[] with a lane-varying subscript per value
        // made the compiler keep the array in memory and read it back 32 times per
        // sub-block: 84 GB/s against a 136 GB/s wall.
        const uint is0 = n * 8u + j * 2u;
        const uint wi = is0 >> 2;
        const uint w = wi == 0u ? aux[0] : wi == 1u ? aux[1] : wi == 2u ? aux[2] : aux[3];
        const uint fs = (is0 & 3u) * 8u;
        const int sv0 = (int)(char)((w >> fs) & 0xFFu) - 32;
        const int sv1 = (int)(char)((w >> (fs + 8u)) & 0xFFu) - 32;
        const float dl0 = float(d) * float(sv0), dl1 = float(d) * float(sv1);
        // A value is its two qs bits plus 4 when its hmask bit is SET, minus 4: the
        // clear-bit-subtracts-4 rule as one integer v3 = q2 | hbit << 2, and dl * (v3 - 4)
        // as fma(dl, v3, -4 dl) -- one rounding either way, and -4 dl is exact, so the
        // float is the one the select form produced.
        const uint mb = n * 4u + j;
        const uint shift = 2u * j;
        const float c0 = -4.0f * dl0, c1 = -4.0f * dl1;
        if (a16) {
            // hmask[32] and this chunk's qs[32] as 16-byte words (tile-major payloads sit
            // at unit + 112 + slot * 96); the row-major block is 110 bytes and takes the
            // byte path below. ONE SHIFT PER WORD: the four values of a word are masked
            // together and read out as bytes by reinterpretation, not shifted out one
            // by one -- the per-value shift-and-select form ran at 96 GB/s.
            device const uint4 * hm4 = (device const uint4 *)pay;
            device const uint4 * q4  = (device const uint4 *)(pay + 32u + n * 32u);
            const uint4 m0 = hm4[0], m1 = hm4[1], p0 = q4[0], p1 = q4[1];
            #pragma unroll
            for (uint k = 0; k < 8u; ++k) {
                const uint qw = k < 4u ? p0[k] : p1[k & 3u];
                const uint mw = k < 4u ? m0[k] : m1[k & 3u];
                const float dl = k < 4u ? dl0 : dl1, c = k < 4u ? c0 : c1;
                // The four values' bits merged at the word level: two qs bits per byte,
                // the hmask bit moved into bit 2 of its byte.
                const uchar4 v3 = as_type<uchar4>(((qw >> shift) & 0x03030303u)
                                                | (((mw >> mb) & 0x01010101u) << 2));
                out[4u * k]      = T(fma(dl, float(v3.x), c));
                out[4u * k + 1u] = T(fma(dl, float(v3.y), c));
                out[4u * k + 2u] = T(fma(dl, float(v3.z), c));
                out[4u * k + 3u] = T(fma(dl, float(v3.w), c));
            }
        } else {
            device const uchar * hm = pay;
            device const uchar * q  = pay + 32u + n * 32u;
            #pragma unroll
            for (uint l = 0; l < 32u; ++l) {
                const float dl = l < 16u ? dl0 : dl1, c = l < 16u ? c0 : c1;
                const uint v3 = ((q[l] >> shift) & 3u) | (((hm[l] >> mb) & 1u) << 2);
                out[l] = T(fma(dl, float(v3), c));
            }
        }
        return;
    }
    if (F == WF_IQ4_XS) {
        // IQ4_NL's codebook with a k-quant's super-block: eight 6-bit sub-scales split
        // across a nibble in scales_l and two bits in scales_h, biased by 32.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        const uint sh = (uint)sc[2] | ((uint)sc[3] << 8);
        const uint ls = (uint)((sc[4u + (sub >> 1)] >> (4u * (sub & 1u))) & 0xFu)
                      | (((sh >> (2u * sub)) & 3u) << 4);
        const float dl = float(d) * float((int)ls - 32);
        device const uchar * q = pay + sub * 16u;
        #pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            out[j]      = T(dl * float((int)KVALUES_IQ4NL[q[j] & 0xFu]));
            out[j + 16] = T(dl * float((int)KVALUES_IQ4NL[q[j] >> 4]));
        }
        return;
    }
    if (F == WF_IQ3_S) {
        // Three packings in one block. Every 8 values are ONE grid entry: a 9-bit index
        // (8 bits in qs, the 9th in qh) selecting four packed byte levels, twice; a sign
        // bit per value in `signs`; and an ODD scale (1 + 2*s) per 32 values, the two
        // nibbles of one scale byte covering sub-blocks 2i and 2i+1.
        //
        // IQ3_S is the one format whose scales are TWO spans -- d at the front of the
        // block, the 4 sub-scale bytes at the back -- so `sc` here is the tile-major
        // unit's 6 concatenated scale bytes (d, then scales), which is exactly what the
        // TM layout produces. A row-major IQ3_S row has no single scale pointer and no
        // tensor needs one: all four are FFN projections, which are tile-major.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        const uint scb = (uint)sc[2u + (sub >> 1)];
        const uint s6 = (sub & 1u) != 0u ? (scb >> 4) : (scb & 0xFu);
        const float db = float(d) * (1.0f + 2.0f * float(s6));
        device const uchar * qs = pay + sub * 8u;            // qs[64] then qh[8] then signs[32]
        device const uchar * qh = pay + 64u;
        device const uchar * sg = pay + 72u + sub * 4u;
        const uint qhb = (uint)qh[sub];
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const uint g1 = IQ3S_GRID[(uint)qs[2u * l]      | ((qhb << (8u - 2u * l)) & 256u)];
            const uint g2 = IQ3S_GRID[(uint)qs[2u * l + 1u] | ((qhb << (7u - 2u * l)) & 256u)];
            const uint sgn = (uint)sg[l];
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float v1 = float((g1 >> (8u * j)) & 0xFFu);
                const float v2 = float((g2 >> (8u * j)) & 0xFFu);
                out[l * 8u + j]      = T((sgn & (1u << j))        ? -db * v1 : db * v1);
                out[l * 8u + j + 4u] = T((sgn & (1u << (j + 4u))) ? -db * v2 : db * v2);
            }
        }
        return;
    }
    if (F == WF_Q2_K) {
        // 2-bit quants with a 4-bit scale AND a 4-bit min per 16 values, both packed in
        // one byte. `sc` is the 16 scale bytes then d and dmin -- two spans in the source
        // block (the scales open it, d/dmin close it), one run at the unit head.
        // Sub-block `sub` takes the 2-bit field at shift 2*(sub & 3) of the 32 payload
        // bytes at (sub >> 2) * 32, and the two scale bytes at 2*sub.
        const half d    = as_type<half>((ushort)(sc[16] | (sc[17] << 8)));
        const half dmin = as_type<half>((ushort)(sc[18] | (sc[19] << 8)));
        const uint shift = 2u * (sub & 3u);
        device const uchar * q = pay + (sub >> 2) * 32u;
        #pragma unroll
        for (uint half_i = 0; half_i < 2u; ++half_i) {
            const uint scb = (uint)sc[2u * sub + half_i];
            const float dl = float(d)    * float(scb & 0xFu);
            const float ml = float(dmin) * float(scb >> 4);
            #pragma unroll
            for (uint l = 0; l < 16u; ++l) {
                const uint v = ((uint)q[half_i * 16u + l] >> shift) & 3u;
                out[half_i * 16u + l] = T(dl * float(v) - ml);
            }
        }
        return;
    }
    if (F == WF_IQ2_XXS) {
        // The leanest codebook. A sub-block's 32 values are two 32-bit words of the payload:
        // four 8-bit grid indices (eight levels each), then four 7-bit sign indices with the
        // 4-bit scale in the top nibble, read as (0.5 + s) * 0.25 like IQ2_XS. `sc` is d
        // alone -- one span, so a row-major block is served by the same two pointers.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        device const uchar * q = pay + sub * 8u;
        const uint w0 = (uint)q[0] | ((uint)q[1] << 8) | ((uint)q[2] << 16) | ((uint)q[3] << 24);
        const uint w1 = (uint)q[4] | ((uint)q[5] << 8) | ((uint)q[6] << 16) | ((uint)q[7] << 24);
        const float db = float(d) * (0.5f + float(w1 >> 28)) * 0.25f;
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const ulong gg = IQ2XXS_GRID[(w0 >> (8u * l)) & 0xFFu];
            const uint g0 = (uint)gg, g1 = (uint)(gg >> 32);
            const uint sgn = (uint)ksign_iq((uchar)((w1 >> (7u * l)) & 127u));
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float v0 = db * float((g0 >> (8u * j)) & 0xFFu);
                const float v1 = db * float((g1 >> (8u * j)) & 0xFFu);
                out[l * 8u + j]      = T((sgn & (1u << j))        ? -v0 : v0);
                out[l * 8u + j + 4u] = T((sgn & (1u << (j + 4u))) ? -v1 : v1);
            }
        }
        return;
    }
    if (F == WF_IQ2_XS) {
        // Eight values per codebook entry. Each u16 of `pay` holds a 9-bit grid index in
        // its low bits and a 7-bit SIGN index in its top bits; one 4-bit sub-scale per 16
        // values, read as (0.5 + s) * 0.25 -- the half-step bias is part of the format.
        // `sc` is d (2) then the 8 sub-scale bytes: two spans in the source block, one run
        // at the tile-major unit head.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        const uint scb = (uint)sc[2u + sub];
        const float db[2] = { float(d) * (0.5f + float(scb & 0xFu))  * 0.25f,
                              float(d) * (0.5f + float(scb >> 4))    * 0.25f };
        device const uchar * q = pay + sub * 8u;
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const uint w = (uint)q[2u * l] | ((uint)q[2u * l + 1u] << 8);
            const ulong gg = IQ2XS_GRID[w & 511u];
            const uint g0 = (uint)gg, g1 = (uint)(gg >> 32);
            const uint sgn = (uint)ksign_iq((uchar)(w >> 9));
            const float dl = db[l >> 1];
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float v0 = dl * float((g0 >> (8u * j)) & 0xFFu);
                const float v1 = dl * float((g1 >> (8u * j)) & 0xFFu);
                out[l * 8u + j]      = T((sgn & (1u << j))        ? -v0 : v0);
                out[l * 8u + j + 4u] = T((sgn & (1u << (j + 4u))) ? -v1 : v1);
            }
        }
        return;
    }
    if (F == WF_IQ2_S) {
        // IQ2_XS's scale rule over a 10-bit index -- eight bits in the payload, the top
        // two from qh -- with the signs stored OUTRIGHT (the payload's second half) rather
        // than as a parity-coded index. `sc` is d (2), qh (8), scales (8).
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        const uint qh  = (uint)sc[2u + sub];
        const uint scb = (uint)sc[10u + sub];
        const float db[2] = { float(d) * (0.5f + float(scb & 0xFu)) * 0.25f,
                              float(d) * (0.5f + float(scb >> 4))   * 0.25f };
        device const uchar * q  = pay + sub * 4u;
        device const uchar * sg = pay + 32u + sub * 4u;
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const uint idx = (uint)q[l] | ((qh << (8u - 2u * l)) & 0x300u);
            const ulong gg = IQ2S_GRID[idx];
            const uint g0 = (uint)gg, g1 = (uint)(gg >> 32);
            const uint sgn = (uint)sg[l];
            const float dl = db[l >> 1];
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float v0 = dl * float((g0 >> (8u * j)) & 0xFFu);
                const float v1 = dl * float((g1 >> (8u * j)) & 0xFFu);
                out[l * 8u + j]      = T((sgn & (1u << j))        ? -v0 : v0);
                out[l * 8u + j + 4u] = T((sgn & (1u << (j + 4u))) ? -v1 : v1);
            }
        }
        return;
    }
    if (F == WF_IQ3_XXS) {
        // FOUR values per codebook entry, two entries per eight values, and one 32-bit
        // word per 32 values holding four 7-bit sign indices with the sub-block scale in
        // its top nibble. Scale and signs share a word, so both live in `sc`: d (2) then
        // the eight words.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        device const uchar * w = sc + 2u + sub * 4u;
        const uint aux = (uint)w[0] | ((uint)w[1] << 8) | ((uint)w[2] << 16) | ((uint)w[3] << 24);
        const float db = float(d) * (0.5f + float(aux >> 28)) * 0.5f;
        device const uchar * qs = pay + sub * 8u;
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const uint sgn = (uint)ksign_iq((uchar)((aux >> (7u * l)) & 127u));
            const uint g1 = IQ3XXS_GRID[(uint)qs[2u * l]];
            const uint g2 = IQ3XXS_GRID[(uint)qs[2u * l + 1u]];
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const float v1 = db * float((g1 >> (8u * j)) & 0xFFu);
                const float v2 = db * float((g2 >> (8u * j)) & 0xFFu);
                out[l * 8u + j]      = T((sgn & (1u << j))        ? -v1 : v1);
                out[l * 8u + j + 4u] = T((sgn & (1u << (j + 4u))) ? -v2 : v2);
            }
        }
        return;
    }
    if (F == WF_Q4_1 || F == WF_Q5_1) {
        // THE AFFINE LEGACY PAIR, one 32-value block: q * d + m with `sc` = d then m. Q5_1's
        // payload opens with a 32-bit word whose bit j is element j's fifth bit, then the
        // sixteen nibble bytes; byte j's low nibble is element j and its high nibble element
        // j + 16, as in every legacy format. Byte loads only: a Q5_1 unit's payload rows sit
        // 20 bytes apart, so `a16` does not hold for it.
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        const float m = float(as_type<half>((ushort)(sc[2] | (sc[3] << 8))));
        const bool q5 = F == WF_Q5_1;
        const uint qh = q5 ? ((uint)pay[0] | ((uint)pay[1] << 8) | ((uint)pay[2] << 16)
                              | ((uint)pay[3] << 24)) : 0u;
        device const uchar * q = pay + (q5 ? 4u : 0u);
        #pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            const uint x0 = ((uint)q[j] & 0xFu) | (((qh >> j) & 1u) << 4);
            const uint x1 = ((uint)q[j] >> 4)   | (((qh >> (j + 16u)) & 1u) << 4);
            out[j]      = T(fma(d, float(x0), m));
            out[j + 16] = T(fma(d, float(x1), m));
        }
        return;
    }
    if (F == WF_Q4_0) {
        // THE SYMMETRIC 4-BIT LEGACY BLOCK: (q - 8) * d with `sc` = d; byte j's low nibble is
        // element j and its high nibble element j + 16. Byte loads: a row-major block is 18
        // bytes, so its payload is not word aligned.
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        if (a16) {
            // Tile-major: a row's 16 payload bytes sit 16-aligned after the unit's scales.
            const uint4 w = *((device const uint4 *)pay);
            const uint words[4] = { w.x, w.y, w.z, w.w };
            #pragma unroll
            for (uint q = 0; q < 4u; ++q) {
                const uchar4 by = as_type<uchar4>(words[q]);
                #pragma unroll
                for (uint i = 0; i < 4u; ++i) {
                    const uint j = 4u * q + i;
                    out[j]      = T(d * float((int)((uint)by[i] & 0xFu) - 8));
                    out[j + 16] = T(d * float((int)((uint)by[i] >> 4) - 8));
                }
            }
            return;
        }
        #pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            const int x0 = (int)((uint)pay[j] & 0xFu) - 8;
            const int x1 = (int)((uint)pay[j] >> 4) - 8;
            out[j]      = T(d * float(x0));
            out[j + 16] = T(d * float(x1));
        }
        return;
    }
    if (F == WF_Q5_0) {
        // THE SYMMETRIC 5-BIT LEGACY BLOCK: ((q | fifth bit) - 16) * d with `sc` = d; the
        // payload is laid out as Q5_1's (the fifth-bit word, then the nibbles).
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        const uint qh = (uint)pay[0] | ((uint)pay[1] << 8) | ((uint)pay[2] << 16)
                      | ((uint)pay[3] << 24);
        device const uchar * q = pay + 4u;
        #pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            const int x0 = (int)(((uint)q[j] & 0xFu) | (((qh >> j) & 1u) << 4)) - 16;
            const int x1 = (int)(((uint)q[j] >> 4)   | (((qh >> (j + 16u)) & 1u) << 4)) - 16;
            out[j]      = T(d * float(x0));
            out[j + 16] = T(d * float(x1));
        }
        return;
    }
    if (F == WF_IQ1_S) {
        // THE 1.56-BIT CODEBOOK. A sub-block's 32 values are four 11-bit grid indices -- eight
        // bits in qs, three in the sub-block's qh word -- each selecting eight levels in
        // {-1, 0, 1}; the same word holds an odd 3-bit scale (2s + 1) and the sign of a
        // +-0.125 delta added to every level. `sc` is d alone; the payload is qs[32] then the
        // eight qh words.
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        device const uchar * qs = pay + sub * 4u;
        const uint qh = (uint)pay[32u + 2u * sub] | ((uint)pay[33u + 2u * sub] << 8);
        const float dl = d * float(2u * ((qh >> 12) & 7u) + 1u);
        const float delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const uint g = (uint)IQ1S_GRID[(uint)qs[l] | (((qh >> (3u * l)) & 7u) << 8)];
            #pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                out[l * 8u + j] = T(dl * (float((g >> (2u * j)) & 3u) - 1.0f + delta));
            }
        }
        return;
    }
    if (F == WF_IQ1_M) {
        // IQ1_S's codebook with TWO scales per 32 values (one per half) and a delta sign per 8.
        // The f16 super-block scale has no field: its sixteen bits are the top nibbles of the
        // four scale words that close the block, which is `sc` here; the other twelve bits of
        // word k hold the 3-bit scales of sub-blocks 2k and 2k + 1. The payload is qs[32] then
        // qh[16], each qh byte carrying two indices' top three bits and delta signs.
        const uint s0 = (uint)sc[0] | ((uint)sc[1] << 8), s1 = (uint)sc[2] | ((uint)sc[3] << 8);
        const uint s2 = (uint)sc[4] | ((uint)sc[5] << 8), s3 = (uint)sc[6] | ((uint)sc[7] << 8);
        const float d = float(as_type<half>((ushort)((s0 >> 12) | ((s1 >> 8) & 0x00F0u)
                                                    | ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u))));
        const uint sw = (uint)sc[2u * (sub >> 1)] | ((uint)sc[2u * (sub >> 1) + 1u] << 8);
        const uint sh = 6u * (sub & 1u);
        const float dl1 = d * float(2u * ((sw >> sh) & 7u) + 1u);
        const float dl2 = d * float(2u * ((sw >> (sh + 3u)) & 7u) + 1u);
        device const uchar * qs = pay + sub * 4u;
        device const uchar * qh = pay + 32u + sub * 2u;
        #pragma unroll
        for (uint l = 0; l < 4u; ++l) {
            const uint hb = (uint)qh[l >> 1] >> (4u * (l & 1u));
            const uint g = (uint)IQ1S_GRID[(uint)qs[l] | ((hb & 7u) << 8)];
            const float delta = (hb & 8u) != 0u ? -0.125f : 0.125f;
            const float dl = l < 2u ? dl1 : dl2;
            #pragma unroll
            for (uint j = 0; j < 8u; ++j) {
                out[l * 8u + j] = T(dl * (float((g >> (2u * j)) & 3u) - 1.0f + delta));
            }
        }
        return;
    }
    if (F == WF_IQ4_NL) {
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        #pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            out[j]      = T(float(d) * float((int)KVALUES_IQ4NL[pay[j] & 0xFu]));
            out[j + 16] = T(float(d) * float((int)KVALUES_IQ4NL[pay[j] >> 4]));
        }
        return;
    }
    #pragma unroll
    for (uint l = 0; l < 32u; ++l) { out[l] = T(0.0h); }
}

// DIAGNOSTIC (IMPARO_BLK_RUN_PROBE, 0 in every served pipeline): what a repacked layout could
// remove from the run fetch, priced by removing it and keeping every load. Bit 0 replaces the
// packed sub-scale unpack with a byte of a word the fetch already holds (Q4_K, Q5_K, Q3_K, Q6_K,
// IQ4_XS); bit 1 replaces a codebook or grid lookup with the index itself (IQ4_XS, IQ4_NL, IQ3_S,
// IQ3_XXS, IQ2_*). Wrong values on purpose.
constant uint BLK_RUN_PROBE_FC [[function_constant(50)]];
constant uint BLK_RUN_PROBE = is_function_constant_defined(BLK_RUN_PROBE_FC) ? BLK_RUN_PROBE_FC : 0u;
// Bit 4: DROP THE OFFSET TERM from the affine formats (Q4_K, Q5_K, Q6_K, Q3_K, Q2_K), keeping the
// scale. It prices the ceiling of deferring the affine -- `s*Sum(q*x) - off*Sum(x)` with Sum(x)
// precomputed outside the matmul -- WITHOUT building Sum(x), because what the deferral removes
// from this loop is exactly this subtract. Wrong values on purpose.

// THE RUN FETCH: the eight values one lane holds in a rows matmul's 8x8 weight fragments, for
// SUB-BLOCK `sub` and the lane's pair of fragment columns `q` (0..3). Each value is the float
// `tm_sub32_t<F, float>` gives it -- the same expression per value, so the same bits. Which eight
// is the format's K ORDER (`tm_run_order_t`), picked so they sit next to each other in the block:
//
//   order 0   lo = k 8q .. 8q+3,   hi = k 8q+4 .. 8q+7          the k-quants and the grid formats
//   order 1   lo = k 4q .. 4q+3,   hi = k 16+4q .. 16+4q+3      the nibble formats, whose byte j
//                                                             holds value j low, value j+16 high
//
// `a16` as for `tm_sub32`: the payload sits where word loads are aligned (every tile-major unit);
// without it the plane formats read bytes.
template <uint F>
inline uint tm_run_order_t() {
    return (F == WF_IQ4_XS || F == WF_IQ4_NL || F == WF_Q4_0 || F == WF_Q4_1 || F == WF_Q5_0
            || F == WF_Q5_1) ? 1u : 0u;
}
inline uint2 tm_bytes8(device const uchar * p) {
    return uint2((uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24),
                 (uint)p[4] | ((uint)p[5] << 8) | ((uint)p[6] << 16) | ((uint)p[7] << 24));
}
template <uint F>
inline void tm_run8_t(device const uchar * sc, device const uchar * pay, uint sub, uint q,
                      bool a16, thread float4 & lo, thread float4 & hi)
{
    if (F == WF_Q4_K || F == WF_Q5_K) {
        // Bytes 8q .. 8q + 7 of the sub-block pair's 32: this sub-block's nibble of each (Q5_K:
        // and bit `sub` of the same bytes of the high-bit plane). Both layouts keep these words
        // aligned (see tm_sub32_t).
        const uint4 h = *((device const uint4 *)sc);
        const half d    = as_type<half>((ushort)(h.x & 0xFFFFu));
        const half dmin = as_type<half>((ushort)(h.x >> 16));
        uint s, m;
        if (BLK_RUN_PROBE & 1u) { s = h.y & 63u; m = (h.z >> 8) & 63u; }
        else { k_scale_min(sub, h.y, h.z, h.w, s, m); }
        const float ds = float(d) * float(s), off = float(dmin) * float(m);
        // Four values a word: one shift and one mask, then each byte read as a uchar (the
        // fifth bit or'd into bit 4 of its byte), where a per-value shift cost two instructions.
        const uint shift = (sub & 1u) * 4u;
        const uint qoff = (F == WF_Q5_K ? 32u : 0u) + (sub >> 1) * 32u + 8u * q;
        const uint2 p = *((device const uint2 *)(pay + qoff));
        uint2 v = (p >> shift) & 0x0F0F0F0Fu;
        if (F == WF_Q5_K) {
            const uint2 hb = *((device const uint2 *)(pay + 8u * q));
            v |= ((hb >> sub) & 0x01010101u) << 4;
        }
        const uchar4 a = as_type<uchar4>(v.x), b = as_type<uchar4>(v.y);
        lo = float4(ds * float(a.x) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off), ds * float(a.y) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off),
                    ds * float(a.z) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off), ds * float(a.w) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off));
        hi = float4(ds * float(b.x) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off), ds * float(b.y) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off),
                    ds * float(b.z) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off), ds * float(b.w) - ((BLK_RUN_PROBE & 4u) ? 0.0f : off));
        return;
    }
    if (F == WF_Q6_K) {
        // The lane's eight values sit in one half of the sub-block, so they take one sub-scale.
        const uint n = sub >> 2, c = sub & 3u;
        const half d = as_type<half>((ushort)(sc[16] | (sc[17] << 8)));
        device const char * s8 = (device const char *)sc + n * 8u;
        const float ds = float(d) * float((int)((BLK_RUN_PROBE & 1u) ? s8[0] : s8[2u * c + (q >> 1)]));
        const float cc = -32.0f * ds;
        const uint shift = 2u * c, shift4 = (c >= 2u) ? 4u : 0u;
        device const uchar * ql = pay + n * 64u + ((c & 1u) * 32u) + 8u * q;
        device const uchar * qh = pay + 128u + n * 32u + 8u * q;
        const uint2 lw = a16 ? *((device const uint2 *)ql) : tm_bytes8(ql);
        const uint2 hw = a16 ? *((device const uint2 *)qh) : tm_bytes8(qh);
        const uchar4 a = as_type<uchar4>(((lw.x >> shift4) & 0x0F0F0F0Fu)
                                       | (((hw.x >> shift) & 0x03030303u) << 4));
        const uchar4 b = as_type<uchar4>(((lw.y >> shift4) & 0x0F0F0F0Fu)
                                       | (((hw.y >> shift) & 0x03030303u) << 4));
        lo = float4(fma(ds, float(a.x), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)), fma(ds, float(a.y), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)),
                    fma(ds, float(a.z), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)), fma(ds, float(a.w), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)));
        hi = float4(fma(ds, float(b.x), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)), fma(ds, float(b.y), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)),
                    fma(ds, float(b.z), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)), fma(ds, float(b.w), ((BLK_RUN_PROBE & 4u) ? 0.0f : cc)));
        return;
    }
    if (F == WF_Q3_K) {
        // The lane's one 6-bit field, is = n*8 + j*2 + (q >> 1), straight from the 12 packed bytes
        // (ggml's layout): its low four bits are the low nibble of byte is (is < 8) or the high
        // nibble of byte is - 8, its top two bits sit in byte 8 + (is & 3) at 2 * (is >> 2). The
        // same integer tm_sub32_t gets by unpacking all sixteen fields.
        const uint n = sub >> 2, j = sub & 3u;
        device const ushort * sh = (device const ushort *)sc;
        const half d = as_type<half>(sh[6]);
        const uint is = n * 8u + j * 2u + (q >> 1);
        int sv;
        if (BLK_RUN_PROBE & 1u) {
            sv = (int)((uint)sc[0] & 63u) - 32;
        } else {
            const uint lo4 = is < 8u ? ((uint)sc[is] & 0xFu) : ((uint)sc[is - 8u] >> 4);
            const uint hi2 = ((uint)sc[8u + (is & 3u)] >> (2u * (is >> 2))) & 3u;
            sv = (int)(lo4 | (hi2 << 4)) - 32;
        }
        const float dl = float(d) * float(sv);
        const float c = -4.0f * dl;
        const uint mb = n * 4u + j, shift = 2u * j;
        device const uchar * hm = pay + 8u * q;
        device const uchar * qs = pay + 32u + n * 32u + 8u * q;
        const uint2 qw = a16 ? *((device const uint2 *)qs) : tm_bytes8(qs);
        const uint2 mw = a16 ? *((device const uint2 *)hm) : tm_bytes8(hm);
        const uchar4 a = as_type<uchar4>(((qw.x >> shift) & 0x03030303u)
                                       | (((mw.x >> mb) & 0x01010101u) << 2));
        const uchar4 b = as_type<uchar4>(((qw.y >> shift) & 0x03030303u)
                                       | (((mw.y >> mb) & 0x01010101u) << 2));
        lo = float4(fma(dl, float(a.x), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)), fma(dl, float(a.y), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)),
                    fma(dl, float(a.z), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)), fma(dl, float(a.w), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)));
        hi = float4(fma(dl, float(b.x), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)), fma(dl, float(b.y), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)),
                    fma(dl, float(b.z), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)), fma(dl, float(b.w), ((BLK_RUN_PROBE & 4u) ? 0.0f : c)));
        return;
    }
    if (F == WF_Q2_K) {
        const half d    = as_type<half>((ushort)(sc[16] | (sc[17] << 8)));
        const half dmin = as_type<half>((ushort)(sc[18] | (sc[19] << 8)));
        const uint shift = 2u * (sub & 3u);
        const uint scb = (uint)sc[2u * sub + (q >> 1)];
        const float dl = float(d)    * float(scb & 0xFu);
        const float ml = float(dmin) * float(scb >> 4);
        device const uchar * qb = pay + (sub >> 2) * 32u + 8u * q;
        const uint2 w = a16 ? *((device const uint2 *)qb) : tm_bytes8(qb);
        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            lo[i] = dl * float((w.x >> (8u * i + shift)) & 3u) - ((BLK_RUN_PROBE & 4u) ? 0.0f : ml);
            hi[i] = dl * float((w.y >> (8u * i + shift)) & 3u) - ((BLK_RUN_PROBE & 4u) ? 0.0f : ml);
        }
        return;
    }
    if (F == WF_IQ4_XS || F == WF_IQ4_NL) {
        // Bytes 4q .. 4q + 3: values 4q + i in the low nibbles, 16 + 4q + i in the high ones.
        float dl;
        device const uchar * qb;
        if (F == WF_IQ4_XS) {
            // The 8-byte header as one word (d, scales_h, then scales_l, whose nibble 'sub' is
            // this sub-block's low four bits), and the payload as one word below, in either
            // layout: a row-major block is 136 bytes, so both stay aligned there too.
            const uint2 h = *((device const uint2 *)sc);
            const uint hx = h.x, hy = h.y;
            const half d = as_type<half>((ushort)(hx & 0xFFFFu));
            const uint sh = hx >> 16;
            const uint ls = ((hy >> (4u * sub)) & 0xFu) | (((sh >> (2u * sub)) & 3u) << 4);
            dl = float(d) * float((int)ls - 32);
            qb = pay + sub * 16u + 4u * q;
        } else {
            dl = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
            qb = pay + 4u * q;
        }
        const uint w = (a16 || F == WF_IQ4_XS)
            ? *((device const uint *)qb)
            : ((uint)qb[0] | ((uint)qb[1] << 8) | ((uint)qb[2] << 16) | ((uint)qb[3] << 24));
        // One lookup per byte: its low nibble's level (value 4q + i) and its high nibble's (value
        // 16 + 4q + i), as floats, so no convert per value either.
        const uchar4 by = as_type<uchar4>(w);
        const bool nolut = (BLK_RUN_PROBE & 2u) != 0u;
        const float2 p0 = nolut ? float2(float(by.x & 15u), float(by.x >> 4)) : KVALUES_IQ4NL_PAIR[by.x];
        const float2 p1 = nolut ? float2(float(by.y & 15u), float(by.y >> 4)) : KVALUES_IQ4NL_PAIR[by.y];
        const float2 p2 = nolut ? float2(float(by.z & 15u), float(by.z >> 4)) : KVALUES_IQ4NL_PAIR[by.z];
        const float2 p3 = nolut ? float2(float(by.w & 15u), float(by.w >> 4)) : KVALUES_IQ4NL_PAIR[by.w];
        lo = float4(dl * p0.x, dl * p1.x, dl * p2.x, dl * p3.x);
        hi = float4(dl * p0.y, dl * p1.y, dl * p2.y, dl * p3.y);
        return;
    }
    if (F == WF_Q4_1 || F == WF_Q5_1) {
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        const float m = float(as_type<half>((ushort)(sc[2] | (sc[3] << 8))));
        const bool q5 = F == WF_Q5_1;
        const uint qh = q5 ? ((uint)pay[0] | ((uint)pay[1] << 8) | ((uint)pay[2] << 16)
                              | ((uint)pay[3] << 24)) : 0u;
        device const uchar * qb = pay + (q5 ? 4u : 0u) + 4u * q;
        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            const uint j = 4u * q + i;
            const uint x0 = ((uint)qb[i] & 0xFu) | (((qh >> j) & 1u) << 4);
            const uint x1 = ((uint)qb[i] >> 4)   | (((qh >> (j + 16u)) & 1u) << 4);
            lo[i] = fma(d, float(x0), m);
            hi[i] = fma(d, float(x1), m);
        }
        return;
    }
    if (F == WF_Q4_0) {
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        device const uchar * qb = pay + 4u * q;
        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            const int x0 = (int)((uint)qb[i] & 0xFu) - 8;
            const int x1 = (int)((uint)qb[i] >> 4) - 8;
            lo[i] = d * float(x0);
            hi[i] = d * float(x1);
        }
        return;
    }
    if (F == WF_Q5_0) {
        const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
        const uint qh = (uint)pay[0] | ((uint)pay[1] << 8) | ((uint)pay[2] << 16)
                      | ((uint)pay[3] << 24);
        device const uchar * qb = pay + 4u + 4u * q;
        #pragma unroll
        for (uint i = 0; i < 4u; ++i) {
            const uint j = 4u * q + i;
            const int x0 = (int)(((uint)qb[i] & 0xFu) | (((qh >> j) & 1u) << 4)) - 16;
            const int x1 = (int)(((uint)qb[i] >> 4)   | (((qh >> (j + 16u)) & 1u) << 4)) - 16;
            lo[i] = d * float(x0);
            hi[i] = d * float(x1);
        }
        return;
    }
    if (F == WF_IQ3_S) {
        // Values 8q .. 8q + 7 are grid entries 2q and 2q + 1: tm_sub32_t's iteration l = q.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        const uint scb = (uint)sc[2u + (sub >> 1)];
        const uint s6 = (sub & 1u) != 0u ? (scb >> 4) : (scb & 0xFu);
        const float db = float(d) * (1.0f + 2.0f * float(s6));
        device const uchar * qs = pay + sub * 8u;
        const uint qhb = (uint)pay[64u + sub];
        const uint i1 = (uint)qs[2u * q]      | ((qhb << (8u - 2u * q)) & 256u);
        const uint i2 = (uint)qs[2u * q + 1u] | ((qhb << (7u - 2u * q)) & 256u);
        const uint g1 = (BLK_RUN_PROBE & 2u) ? (i1 * 0x01010101u) : IQ3S_GRID[i1];
        const uint g2 = (BLK_RUN_PROBE & 2u) ? (i2 * 0x01010101u) : IQ3S_GRID[i2];
        const uint sgn = (uint)pay[72u + sub * 4u + q];
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            const float v1 = float((g1 >> (8u * j)) & 0xFFu);
            const float v2 = float((g2 >> (8u * j)) & 0xFFu);
            lo[j] = (sgn & (1u << j))        ? -db * v1 : db * v1;
            hi[j] = (sgn & (1u << (j + 4u))) ? -db * v2 : db * v2;
        }
        return;
    }
    if (F == WF_IQ2_XXS || F == WF_IQ2_XS || F == WF_IQ2_S) {
        // Values 8q .. 8q + 7 are one grid entry: tm_sub32_t's iteration l = q.
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        float dl;
        ulong gg;
        uint sgn;
        if (F == WF_IQ2_XXS) {
            device const uchar * qb = pay + sub * 8u;
            const uint w1 = (uint)qb[4] | ((uint)qb[5] << 8) | ((uint)qb[6] << 16)
                          | ((uint)qb[7] << 24);
            dl = float(d) * (0.5f + float(w1 >> 28)) * 0.25f;
            gg = (BLK_RUN_PROBE & 2u) ? ((ulong)qb[q] * 0x0101010101010101ul) : IQ2XXS_GRID[(uint)qb[q]];
            sgn = (uint)ksign_iq((uchar)((w1 >> (7u * q)) & 127u));
        } else if (F == WF_IQ2_XS) {
            const uint scb = (uint)sc[2u + sub];
            const uint nib = (q >> 1) != 0u ? (scb >> 4) : (scb & 0xFu);
            dl = float(d) * (0.5f + float(nib)) * 0.25f;
            device const uchar * qb = pay + sub * 8u;
            const uint w = (uint)qb[2u * q] | ((uint)qb[2u * q + 1u] << 8);
            gg = (BLK_RUN_PROBE & 2u) ? ((ulong)(w & 511u) * 0x0101010101010101ul) : IQ2XS_GRID[w & 511u];
            sgn = (uint)ksign_iq((uchar)(w >> 9));
        } else {
            const uint qh  = (uint)sc[2u + sub];
            const uint scb = (uint)sc[10u + sub];
            const uint nib = (q >> 1) != 0u ? (scb >> 4) : (scb & 0xFu);
            dl = float(d) * (0.5f + float(nib)) * 0.25f;
            const uint gi = (uint)pay[sub * 4u + q] | ((qh << (8u - 2u * q)) & 0x300u);
            gg = (BLK_RUN_PROBE & 2u) ? ((ulong)gi * 0x0101010101010101ul) : IQ2S_GRID[gi];
            sgn = (uint)pay[32u + sub * 4u + q];
        }
        const uint g0 = (uint)gg, g1 = (uint)(gg >> 32);
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            const float v0 = dl * float((g0 >> (8u * j)) & 0xFFu);
            const float v1 = dl * float((g1 >> (8u * j)) & 0xFFu);
            lo[j] = (sgn & (1u << j))        ? -v0 : v0;
            hi[j] = (sgn & (1u << (j + 4u))) ? -v1 : v1;
        }
        return;
    }
    if (F == WF_IQ3_XXS) {
        const half d = as_type<half>((ushort)(sc[0] | (sc[1] << 8)));
        device const uchar * w = sc + 2u + sub * 4u;
        const uint aux = (uint)w[0] | ((uint)w[1] << 8) | ((uint)w[2] << 16) | ((uint)w[3] << 24);
        const float db = float(d) * (0.5f + float(aux >> 28)) * 0.5f;
        const uint sgn = (uint)ksign_iq((uchar)((aux >> (7u * q)) & 127u));
        device const uchar * qs = pay + sub * 8u;
        const uint j1 = (uint)qs[2u * q], j2 = (uint)qs[2u * q + 1u];
        const uint g1 = (BLK_RUN_PROBE & 2u) ? (j1 * 0x01010101u) : IQ3XXS_GRID[j1];
        const uint g2 = (BLK_RUN_PROBE & 2u) ? (j2 * 0x01010101u) : IQ3XXS_GRID[j2];
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            const float v1 = db * float((g1 >> (8u * j)) & 0xFFu);
            const float v2 = db * float((g2 >> (8u * j)) & 0xFFu);
            lo[j] = (sgn & (1u << j))        ? -v1 : v1;
            hi[j] = (sgn & (1u << (j + 4u))) ? -v2 : v2;
        }
        return;
    }
    if (F == WF_IQ1_S || F == WF_IQ1_M) {
        float dl, delta;
        uint g;
        if (F == WF_IQ1_S) {
            const float d = float(as_type<half>((ushort)(sc[0] | (sc[1] << 8))));
            const uint qh = (uint)pay[32u + 2u * sub] | ((uint)pay[33u + 2u * sub] << 8);
            dl = d * float(2u * ((qh >> 12) & 7u) + 1u);
            delta = (qh & 0x8000u) != 0u ? -0.125f : 0.125f;
            g = (uint)IQ1S_GRID[(uint)pay[sub * 4u + q] | (((qh >> (3u * q)) & 7u) << 8)];
        } else {
            const uint s0 = (uint)sc[0] | ((uint)sc[1] << 8), s1 = (uint)sc[2] | ((uint)sc[3] << 8);
            const uint s2 = (uint)sc[4] | ((uint)sc[5] << 8), s3 = (uint)sc[6] | ((uint)sc[7] << 8);
            const float d = float(as_type<half>((ushort)((s0 >> 12) | ((s1 >> 8) & 0x00F0u)
                                                        | ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u))));
            const uint sw = (uint)sc[2u * (sub >> 1)] | ((uint)sc[2u * (sub >> 1) + 1u] << 8);
            const uint sh = 6u * (sub & 1u);
            dl = q < 2u ? d * float(2u * ((sw >> sh) & 7u) + 1u)
                        : d * float(2u * ((sw >> (sh + 3u)) & 7u) + 1u);
            const uint hb = (uint)pay[32u + sub * 2u + (q >> 1)] >> (4u * (q & 1u));
            g = (uint)IQ1S_GRID[(uint)pay[sub * 4u + q] | ((hb & 7u) << 8)];
            delta = (hb & 8u) != 0u ? -0.125f : 0.125f;
        }
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            lo[j] = dl * (float((g >> (2u * j)) & 3u) - 1.0f + delta);
            hi[j] = dl * (float((g >> (2u * (j + 4u))) & 3u) - 1.0f + delta);
        }
        return;
    }
    lo = float4(0.0f);
    hi = float4(0.0f);
}

// THE PAIR FETCH (Q4_K, Q5_K): sub-blocks 2i and 2i+1 keep their codes in the low and the high
// nibble of the same bytes (and Q5_K their fifth bits in the same plane bytes) under one header,
// so one load of each serves both sub-blocks. `sub0` is the even sub-block; each value is the
// float tm_run8_t gives it (tests/brick_rows.rs).
template <uint F>
inline bool tm_run_pairs_t() { return F == WF_Q4_K || F == WF_Q5_K; }
template <uint F>
inline void tm_run8_pair_t(device const uchar * sc, device const uchar * pay, uint sub0, uint q,
                           thread float4 & lo0, thread float4 & hi0,
                           thread float4 & lo1, thread float4 & hi1)
{
    if (F == WF_Q4_K || F == WF_Q5_K) {
        const uint4 h = *((device const uint4 *)sc);
        const half d    = as_type<half>((ushort)(h.x & 0xFFFFu));
        const half dmin = as_type<half>((ushort)(h.x >> 16));
        uint s0, m0, s1, m1;
        if (BLK_RUN_PROBE & 1u) {
            s0 = h.y & 63u; m0 = (h.z >> 8) & 63u; s1 = (h.y >> 8) & 63u; m1 = h.z & 63u;
        } else {
            k_scale_min(sub0, h.y, h.z, h.w, s0, m0);
            k_scale_min(sub0 + 1u, h.y, h.z, h.w, s1, m1);
        }
        const float ds0 = float(d) * float(s0), off0 = float(dmin) * float(m0);
        const float ds1 = float(d) * float(s1), off1 = float(dmin) * float(m1);
        const uint qoff = (F == WF_Q5_K ? 32u : 0u) + (sub0 >> 1) * 32u + 8u * q;
        const uint2 p = *((device const uint2 *)(pay + qoff));
        uint2 v0 = p & 0x0F0F0F0Fu, v1 = (p >> 4u) & 0x0F0F0F0Fu;
        if (F == WF_Q5_K) {
            const uint2 hb = *((device const uint2 *)(pay + 8u * q));
            v0 |= ((hb >> sub0) & 0x01010101u) << 4;
            v1 |= ((hb >> (sub0 + 1u)) & 0x01010101u) << 4;
        }
        const uchar4 a0 = as_type<uchar4>(v0.x), b0 = as_type<uchar4>(v0.y);
        const uchar4 a1 = as_type<uchar4>(v1.x), b1 = as_type<uchar4>(v1.y);
        lo0 = float4(ds0 * float(a0.x) - off0, ds0 * float(a0.y) - off0,
                     ds0 * float(a0.z) - off0, ds0 * float(a0.w) - off0);
        hi0 = float4(ds0 * float(b0.x) - off0, ds0 * float(b0.y) - off0,
                     ds0 * float(b0.z) - off0, ds0 * float(b0.w) - off0);
        lo1 = float4(ds1 * float(a1.x) - off1, ds1 * float(a1.y) - off1,
                     ds1 * float(a1.z) - off1, ds1 * float(a1.w) - off1);
        hi1 = float4(ds1 * float(b1.x) - off1, ds1 * float(b1.y) - off1,
                     ds1 * float(b1.z) - off1, ds1 * float(b1.w) - off1);
        return;
    }
    lo0 = hi0 = lo1 = hi1 = float4(0.0f);
}

// THE QUAD SCALES (Q4_K, Q5_K, Q3_K): the four sub-block scales of one HALF-BLOCK, unpacked
// once with word-wide masks instead of once per run. Sub-blocks 4i .. 4i + 3 are always the low
// or the high half of a k-quant block, and each half's six-bit fields lie in one byte position of
// the header words, so four scales (and four mins) come out of two masks. Every value is the
// float the per-run unpack gives it: the same integer, the same multiply.
template <uint F>
inline bool tm_run_quads_t() { return F == WF_Q4_K || F == WF_Q5_K || F == WF_Q3_K; }
template <uint F>
inline void tm_quad_scales_t(device const uchar * sc, uint sub0, uint q,
                             thread float4 & a, thread float4 & b)
{
    if (F == WF_Q4_K || F == WF_Q5_K) {
        // a = d * scale, b = dmin * min, for sub-blocks sub0 .. sub0 + 3.
        const uint4 h = *((device const uint4 *)sc);
        const half d    = as_type<half>((ushort)(h.x & 0xFFFFu));
        const half dmin = as_type<half>((ushort)(h.x >> 16));
        uint sw, mw;
        if (sub0 < 4u) {
            sw = h.y & 0x3F3F3F3Fu;
            mw = h.z & 0x3F3F3F3Fu;
        } else {
            sw = (h.w & 0x0F0F0F0Fu) | (((h.y >> 6) & 0x03030303u) << 4);
            mw = ((h.w >> 4) & 0x0F0F0F0Fu) | (((h.z >> 6) & 0x03030303u) << 4);
        }
        const uchar4 sb = as_type<uchar4>(sw), mb = as_type<uchar4>(mw);
        a = float4(float(d) * float(sb.x), float(d) * float(sb.y),
                   float(d) * float(sb.z), float(d) * float(sb.w));
        b = float4(float(dmin) * float(mb.x), float(dmin) * float(mb.y),
                   float(dmin) * float(mb.z), float(dmin) * float(mb.w));
        return;
    }
    if (F == WF_Q3_K) {
        // a = d * (field - 32) for the lane's own six-bit field of each sub-block, b its -4x
        // companion (the 2-bit codes are biased by 4). The two bytes holding the top bits are
        // the same for j and j + 2, so the loads fold.
        device const ushort * sh = (device const ushort *)sc;
        const float d = float(as_type<half>(sh[6]));
        const uint n = sub0 >> 2, hq = q >> 1;
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            const uint is = n * 8u + 2u * j + hq;
            const uint lo4 = is < 8u ? ((uint)sc[is] & 0xFu) : ((uint)sc[is - 8u] >> 4);
            const uint hi2 = ((uint)sc[8u + (is & 3u)] >> (2u * (is >> 2))) & 3u;
            a[j] = d * float((int)(lo4 | (hi2 << 4)) - 32);
        }
        b = -4.0f * a;
        return;
    }
    a = float4(0.0f);
    b = float4(0.0f);
}

// The pair fetch with the quad's scales (Q4_K, Q5_K): `i` is the pair's first sub-block inside the
// half-block, so the scales are a[i], b[i] and a[i + 1], b[i + 1].
template <uint F>
inline void tm_run8_pair_q_t(float4 a, float4 b, uint i, device const uchar * pay, uint sub0,
                             uint q, thread float4 & lo0, thread float4 & hi0,
                             thread float4 & lo1, thread float4 & hi1)
{
    if (F == WF_Q4_K || F == WF_Q5_K) {
        const uint qoff = (F == WF_Q5_K ? 32u : 0u) + (sub0 >> 1) * 32u + 8u * q;
        const uint2 p = *((device const uint2 *)(pay + qoff));
        uint2 v0 = p & 0x0F0F0F0Fu, v1 = (p >> 4u) & 0x0F0F0F0Fu;
        if (F == WF_Q5_K) {
            const uint2 hb = *((device const uint2 *)(pay + 8u * q));
            v0 |= ((hb >> sub0) & 0x01010101u) << 4;
            v1 |= ((hb >> (sub0 + 1u)) & 0x01010101u) << 4;
        }
        const uchar4 a0 = as_type<uchar4>(v0.x), b0 = as_type<uchar4>(v0.y);
        const uchar4 a1 = as_type<uchar4>(v1.x), b1 = as_type<uchar4>(v1.y);
        const float ds0 = a[i], off0 = b[i], ds1 = a[i + 1u], off1 = b[i + 1u];
        lo0 = float4(ds0 * float(a0.x) - off0, ds0 * float(a0.y) - off0,
                     ds0 * float(a0.z) - off0, ds0 * float(a0.w) - off0);
        hi0 = float4(ds0 * float(b0.x) - off0, ds0 * float(b0.y) - off0,
                     ds0 * float(b0.z) - off0, ds0 * float(b0.w) - off0);
        lo1 = float4(ds1 * float(a1.x) - off1, ds1 * float(a1.y) - off1,
                     ds1 * float(a1.z) - off1, ds1 * float(a1.w) - off1);
        hi1 = float4(ds1 * float(b1.x) - off1, ds1 * float(b1.y) - off1,
                     ds1 * float(b1.z) - off1, ds1 * float(b1.w) - off1);
        return;
    }
    lo0 = hi0 = lo1 = hi1 = float4(0.0f);
}

// The run fetch with the quad's scales (Q3_K): `i` is the sub-block inside the half-block.
template <uint F>
inline void tm_run8_q_t(float4 a, float4 b, uint i, device const uchar * pay, uint sub, uint q,
                        bool a16, thread float4 & lo, thread float4 & hi)
{
    if (F == WF_Q3_K) {
        const uint n = sub >> 2, j = sub & 3u;
        const float dl = a[i], c = b[i];
        const uint mb = n * 4u + j, shift = 2u * j;
        device const uchar * hm = pay + 8u * q;
        device const uchar * qs = pay + 32u + n * 32u + 8u * q;
        const uint2 qw = a16 ? *((device const uint2 *)qs) : tm_bytes8(qs);
        const uint2 mw = a16 ? *((device const uint2 *)hm) : tm_bytes8(hm);
        const uchar4 av = as_type<uchar4>(((qw.x >> shift) & 0x03030303u)
                                        | (((mw.x >> mb) & 0x01010101u) << 2));
        const uchar4 bv = as_type<uchar4>(((qw.y >> shift) & 0x03030303u)
                                        | (((mw.y >> mb) & 0x01010101u) << 2));
        lo = float4(fma(dl, float(av.x), c), fma(dl, float(av.y), c),
                    fma(dl, float(av.z), c), fma(dl, float(av.w), c));
        hi = float4(fma(dl, float(bv.x), c), fma(dl, float(bv.y), c),
                    fma(dl, float(bv.z), c), fma(dl, float(bv.w), c));
        return;
    }
    lo = float4(0.0f);
    hi = float4(0.0f);
}

// Geometry of the tile-major unit for format F, mirroring TmRule. Sub-blocks per block is
// block_elems / 32, so a legacy format is one and a super-block is eight.
template <uint F>
inline uint tm_block_elems_t() {
    return (F == WF_IQ4_NL || F == WF_Q4_0 || F == WF_Q4_1 || F == WF_Q5_0 || F == WF_Q5_1)
        ? 32u : 256u;
}
template <uint F>
inline uint tm_block_bytes_t() {
    if (F == WF_IQ3_S)  { return 110u; }
    if (F == WF_Q4_K)   { return 144u; }
    if (F == WF_Q5_K)   { return 176u; }
    if (F == WF_Q6_K)   { return 210u; }
    if (F == WF_Q3_K)   { return 110u; }
    if (F == WF_IQ4_XS) { return 136u; }
    if (F == WF_IQ2_XS)  { return 74u; }
    if (F == WF_IQ2_XXS) { return 66u; }
    if (F == WF_IQ2_S)   { return 82u; }
    if (F == WF_IQ3_XXS) { return 98u; }
    if (F == WF_Q2_K)    { return 84u; }
    if (F == WF_IQ1_S)   { return 50u; }
    if (F == WF_IQ1_M)   { return 56u; }
    if (F == WF_Q4_0)    { return 18u; }
    if (F == WF_Q4_1)    { return 20u; }
    if (F == WF_Q5_0)    { return 22u; }
    if (F == WF_Q5_1)    { return 24u; }
    return 18u;                                    // IQ4_NL
}
// Byte offset of the scale run inside a ROW-MAJOR block. Every format this brick reads
// has exactly ONE contiguous scale run and one contiguous payload run, so a row-major
// block is also just "a scale pointer and a payload pointer" -- which is why `tm_sub32`
// serves both layouts and only the ADDRESSES differ.
template <uint F>
inline uint tm_scale_src_off_t() {
    if (F == WF_Q6_K) { return 192u; }
    if (F == WF_Q3_K) { return 96u; }
    if (F == WF_IQ1_M) { return 48u; }
    return 0u;                                     // every other format
}
template <uint F>
inline uint tm_scale_bytes_t() {
    if (F == WF_IQ3_S) { return 6u; }            // d (2) + the 4 sub-scale bytes
    if (F == WF_Q4_K || F == WF_Q5_K) { return 16u; }
    if (F == WF_Q6_K)   { return 18u; }
    if (F == WF_Q3_K)   { return 14u; }
    if (F == WF_IQ4_XS) { return 8u; }
    if (F == WF_IQ2_XS)  { return 10u; }        // d (2) + the 8 sub-scale bytes
    if (F == WF_IQ2_XXS) { return 2u; }         // d alone; scales ride in the payload words
    if (F == WF_IQ2_S)   { return 18u; }        // d (2) + qh (8) + scales (8)
    if (F == WF_IQ3_XXS) { return 34u; }        // d (2) + the 8 scale-and-sign words
    if (F == WF_Q2_K)    { return 20u; }        // scales[16] + d + dmin
    if (F == WF_IQ1_M)   { return 8u; }         // the four scale words; d in their top nibbles
    if (F == WF_Q4_1 || F == WF_Q5_1) { return 4u; }   // d, m
    return 2u;                                     // d alone: IQ4_NL, Q4_0, Q5_0, IQ1_S
}

// THE FORMAT LIST, written once. Every brick above is a template over the format, so its
// body compiles with the format as a constant either way; what differs is WHERE the format
// comes from:
//
//   the pipeline's   (E4B Q4_0, LFM2 Q8_0 tile-major, every blk_* kernel)  a function
//                    constant -- the switch below folds to one arm when the pipeline is
//                    specialised, exactly as the `if (WFMT == ...)` chain folded before
//   the entry's      (the qwen35 mega-kernel, whose file assigns a format per TENSOR: one
//                    UD file gave 53 distinct per-layer signatures, so a pipeline per
//                    signature is not a design) a runtime word -- the switch is a real
//                    branch, taken once per phase, uniform across the grid, and the row
//                    loop inside the arm is still compiled for one format
//
// A function constant cannot BE a template argument (it is resolved after the AST), which
// is why this is a switch over literals rather than `tm_sub32_t<WFMT>`. The body is
// variadic because a template argument list carries a comma the preprocessor would
// otherwise read as an argument separator; every arm's body must end in `return` or
// `break`, since the cases do not fall through to one.
#define TM_BY_FORMAT(fmtval, ...)                                              \
    switch (fmtval) {                                                          \
    case WF_Q4_K:    { constexpr uint FMT = WF_Q4_K;    __VA_ARGS__ }                 \
    case WF_Q5_K:    { constexpr uint FMT = WF_Q5_K;    __VA_ARGS__ }                 \
    case WF_Q6_K:    { constexpr uint FMT = WF_Q6_K;    __VA_ARGS__ }                 \
    case WF_Q3_K:    { constexpr uint FMT = WF_Q3_K;    __VA_ARGS__ }                 \
    case WF_Q2_K:    { constexpr uint FMT = WF_Q2_K;    __VA_ARGS__ }                 \
    case WF_IQ4_XS:  { constexpr uint FMT = WF_IQ4_XS;  __VA_ARGS__ }                 \
    case WF_IQ4_NL:  { constexpr uint FMT = WF_IQ4_NL;  __VA_ARGS__ }                 \
    case WF_IQ3_S:   { constexpr uint FMT = WF_IQ3_S;   __VA_ARGS__ }                 \
    case WF_IQ3_XXS: { constexpr uint FMT = WF_IQ3_XXS; __VA_ARGS__ }                 \
    case WF_IQ2_XS:  { constexpr uint FMT = WF_IQ2_XS;  __VA_ARGS__ }                 \
    case WF_IQ2_XXS: { constexpr uint FMT = WF_IQ2_XXS; __VA_ARGS__ }                 \
    case WF_IQ2_S:   { constexpr uint FMT = WF_IQ2_S;   __VA_ARGS__ }                 \
    case WF_Q4_0:    { constexpr uint FMT = WF_Q4_0;    __VA_ARGS__ }                 \
    case WF_Q4_1:    { constexpr uint FMT = WF_Q4_1;    __VA_ARGS__ }                 \
    case WF_Q5_0:    { constexpr uint FMT = WF_Q5_0;    __VA_ARGS__ }                 \
    case WF_Q5_1:    { constexpr uint FMT = WF_Q5_1;    __VA_ARGS__ }                 \
    case WF_IQ1_S:   { constexpr uint FMT = WF_IQ1_S;   __VA_ARGS__ }                 \
    case WF_IQ1_M:   { constexpr uint FMT = WF_IQ1_M;   __VA_ARGS__ }                 \
    default:         { constexpr uint FMT = WF_ROWMAJOR; __VA_ARGS__ }                \
    }

template <typename T>
inline void tm_sub32_fmt(uint f, device const uchar * sc, device const uchar * pay, uint sub,
                         thread T out[32], bool a16) {
    TM_BY_FORMAT(f, tm_sub32_t<FMT, T>(sc, pay, sub, out, a16); return;)
}
inline uint tm_block_elems_fmt(uint f)   { TM_BY_FORMAT(f, return tm_block_elems_t<FMT>();) }
inline uint tm_block_bytes_fmt(uint f)   { TM_BY_FORMAT(f, return tm_block_bytes_t<FMT>();) }
inline uint tm_scale_src_off_fmt(uint f) { TM_BY_FORMAT(f, return tm_scale_src_off_t<FMT>();) }
inline uint tm_scale_bytes_fmt(uint f)   { TM_BY_FORMAT(f, return tm_scale_bytes_t<FMT>();) }
inline uint tm_run_order_fmt(uint f)     { TM_BY_FORMAT(f, return tm_run_order_t<FMT>();) }
inline bool tm_run_pairs_fmt(uint f)     { TM_BY_FORMAT(f, return tm_run_pairs_t<FMT>();) }
inline bool tm_run_quads_fmt(uint f)     { TM_BY_FORMAT(f, return tm_run_quads_t<FMT>();) }
inline void tm_quad_scales_fmt(uint f, device const uchar * sc, uint sub0, uint q,
                               thread float4 & a, thread float4 & b) {
    TM_BY_FORMAT(f, tm_quad_scales_t<FMT>(sc, sub0, q, a, b); return;)
}
inline void tm_run8_pair_q_fmt(uint f, float4 a, float4 b, uint i, device const uchar * pay,
                               uint sub0, uint q, thread float4 & lo0, thread float4 & hi0,
                               thread float4 & lo1, thread float4 & hi1) {
    TM_BY_FORMAT(f, tm_run8_pair_q_t<FMT>(a, b, i, pay, sub0, q, lo0, hi0, lo1, hi1); return;)
}
inline void tm_run8_q_fmt(uint f, float4 a, float4 b, uint i, device const uchar * pay, uint sub,
                          uint q, bool a16, thread float4 & lo, thread float4 & hi) {
    TM_BY_FORMAT(f, tm_run8_q_t<FMT>(a, b, i, pay, sub, q, a16, lo, hi); return;)
}
inline void tm_run8_pair_fmt(uint f, device const uchar * sc, device const uchar * pay, uint sub0,
                             uint q, thread float4 & lo0, thread float4 & hi0,
                             thread float4 & lo1, thread float4 & hi1) {
    TM_BY_FORMAT(f, tm_run8_pair_t<FMT>(sc, pay, sub0, q, lo0, hi0, lo1, hi1); return;)
}
inline void tm_run8_fmt(uint f, device const uchar * sc, device const uchar * pay, uint sub, uint q,
                        bool a16, thread float4 & lo, thread float4 & hi) {
    TM_BY_FORMAT(f, tm_run8_t<FMT>(sc, pay, sub, q, a16, lo, hi); return;)
}

// The pipeline's own format: what every existing caller compiles to.

// `a16`: the caller's payload pointers sit on 16-byte boundaries. A TILE-MAJOR unit
// guarantees it when the format's payload per block is a multiple of 16 -- every format
// but Q5_0 and Q5_1, whose 20-byte payload rows the tile-major callers flag all the same,
// so their arms ignore the flag -- and a row-major block only when its byte size is one
// (Q4_K 144, Q5_K 176; not Q3_K 110, Q6_K 210, IQ4_XS 136). The arms that need it take
// 16-byte loads under the flag and byte loads without; the always-aligned arms ignore it.
template <typename T>
inline void tm_sub32(device const uchar * sc, device const uchar * pay, uint sub,
                     thread T out[32], bool a16) { tm_sub32_fmt<T>(WFMT, sc, pay, sub, out, a16); }
inline uint tm_block_elems()   { return tm_block_elems_fmt(WFMT); }
inline uint tm_block_bytes()   { return tm_block_bytes_fmt(WFMT); }
inline uint tm_scale_src_off() { return tm_scale_src_off_fmt(WFMT); }
inline uint tm_scale_bytes()   { return tm_scale_bytes_fmt(WFMT); }
inline uint tm_run_order()     { return tm_run_order_fmt(WFMT); }
inline bool tm_run_pairs()     { return tm_run_pairs_fmt(WFMT); }
inline bool tm_run_quads()     { return tm_run_quads_fmt(WFMT); }
inline void tm_quad_scales(device const uchar * sc, uint sub0, uint q,
                           thread float4 & a, thread float4 & b) {
    tm_quad_scales_fmt(WFMT, sc, sub0, q, a, b);
}
inline void tm_run8_pair_q(float4 a, float4 b, uint i, device const uchar * pay, uint sub0,
                           uint q, thread float4 & lo0, thread float4 & hi0,
                           thread float4 & lo1, thread float4 & hi1) {
    tm_run8_pair_q_fmt(WFMT, a, b, i, pay, sub0, q, lo0, hi0, lo1, hi1);
}
inline void tm_run8_q(float4 a, float4 b, uint i, device const uchar * pay, uint sub, uint q,
                      bool a16, thread float4 & lo, thread float4 & hi) {
    tm_run8_q_fmt(WFMT, a, b, i, pay, sub, q, a16, lo, hi);
}
inline void tm_run8_pair(device const uchar * sc, device const uchar * pay, uint sub0, uint q,
                         thread float4 & lo0, thread float4 & hi0,
                         thread float4 & lo1, thread float4 & hi1) {
    tm_run8_pair_fmt(WFMT, sc, pay, sub0, q, lo0, hi0, lo1, hi1);
}
inline void tm_run8(device const uchar * sc, device const uchar * pay, uint sub, uint q, bool a16,
                    thread float4 & lo, thread float4 & hi) {
    tm_run8_fmt(WFMT, sc, pay, sub, q, a16, lo, hi);
}

constant bool BLK_PROBE_HALF [[function_constant(24)]];
// The probe's other outputs: BLK_PROBE_RUN assembles each sub-block from the run fetch
// (`tm_run8`, all four column pairs, each value put back at its k), so a test can require it to
// equal `tm_sub32`'s floats bit for bit; BLK_PROBE_A16 hands both bricks the aligned flag the
// tile-major kernels pass (the probe's runs keep every payload on the alignment a unit gives it).
constant bool BLK_PROBE_RUN_FC [[function_constant(41)]];
constant bool BLK_PROBE_RUN = is_function_constant_defined(BLK_PROBE_RUN_FC) && BLK_PROBE_RUN_FC;
constant bool BLK_PROBE_A16_FC [[function_constant(42)]];
constant bool BLK_PROBE_A16 = is_function_constant_defined(BLK_PROBE_A16_FC) && BLK_PROBE_A16_FC;
// With BLK_PROBE_RUN, the pair fetch (tm_run8_pair) where the format has one: each sub-block's
// values come from the pair call on its even sub-block.
constant bool BLK_PROBE_PAIR_FC [[function_constant(48)]];
constant bool BLK_PROBE_PAIR = is_function_constant_defined(BLK_PROBE_PAIR_FC) && BLK_PROBE_PAIR_FC;
// With BLK_PROBE_RUN, the half-block fetch (tm_quad_scales with tm_run8_pair_q / tm_run8_q) where
// the format has one: each sub-block's values come from its half-block's scales.
constant bool BLK_PROBE_QUAD_FC [[function_constant(52)]];
constant bool BLK_PROBE_QUAD = is_function_constant_defined(BLK_PROBE_QUAD_FC) && BLK_PROBE_QUAD_FC;

// THE BRICK'S GATE. Decodes a ROW-MAJOR block row through `tm_sub32` -- the same brick the
// prefill GEMM, the decode GEMV and the row gather call -- so a test can diff it against
// the CPU row codec, which is itself pinned bit-exact against llama.cpp's own dequantiser
// (crates/imparo-cpu/tests/quant_rows.rs, worst relative error 0e0). BLK_PROBE_HALF picks
// the output under test: half, as the prefill GEMM stages it, or float, as the decode GEMV,
// the row gather and the mega-kernel unit stage it.
//
// A transcription is a CLAIM. This is where the claim is checked, and it is checked before
// a forward is built on top of it: a wrong nibble pairing or a flipped sign reads as
// plausible logits, never as a crash, and then costs a day of bisecting a whole model.
kernel void imparo_blk_decode_probe(
    device const uchar * scales [[buffer(0)]], device const uchar * payload [[buffer(1)]],
    device float * out [[buffer(2)]], constant uint & n_subs [[buffer(3)]],
    uint sb [[thread_position_in_grid]])
{
    if (sb >= n_subs) { return; }
    // The SCALE RUN and the PAYLOAD RUN, not a block. That is exactly `tm_sub32`'s
    // contract, so this gates the decode with no addressing mixed in -- and it is the only
    // shape that can express IQ3_S, whose scales are TWO spans of the source block (d at
    // the front, the sub-scale bytes at the back). The caller splits by the rule's spans,
    // which is the same table the repack moves bytes with.
    const uint sub_per_block = tm_block_elems() / 32u;
    const uint blk = sb / sub_per_block, sub = sb % sub_per_block;
    const uint sc_bytes = tm_scale_bytes();
    device const uchar * sc = scales + blk * sc_bytes;
    device const uchar * pay = payload + blk * (tm_block_bytes() - sc_bytes);
    if (BLK_PROBE_RUN) {
        const bool o1 = tm_run_order() == 1u;
        for (uint q = 0; q < 4u; ++q) {
            float4 lo, hi;
            if (BLK_PROBE_QUAD && tm_run_quads()) {
                // The sub-block's half-block: scales for sub0 .. sub0 + 3, then its own values.
                const uint sub0 = sub & ~3u, i = sub & 3u;
                float4 a, b;
                tm_quad_scales(sc, sub0, q, a, b);
                if (tm_run_pairs()) {
                    float4 lo0, hi0, lo1, hi1;
                    tm_run8_pair_q(a, b, i & ~1u, pay, sub & ~1u, q, lo0, hi0, lo1, hi1);
                    lo = (sub & 1u) != 0u ? lo1 : lo0;
                    hi = (sub & 1u) != 0u ? hi1 : hi0;
                } else {
                    tm_run8_q(a, b, i, pay, sub, q, BLK_PROBE_A16, lo, hi);
                }
            } else if (BLK_PROBE_PAIR && tm_run_pairs()) {
                float4 lo0, hi0, lo1, hi1;
                tm_run8_pair(sc, pay, sub & ~1u, q, lo0, hi0, lo1, hi1);
                lo = (sub & 1u) != 0u ? lo1 : lo0;
                hi = (sub & 1u) != 0u ? hi1 : hi0;
            } else {
                tm_run8(sc, pay, sub, q, BLK_PROBE_A16, lo, hi);
            }
            for (uint f = 0; f < 4u; ++f) {
                out[sb * 32u + (o1 ? 4u * q + f : 8u * q + f)] = lo[f];
                out[sb * 32u + (o1 ? 16u + 4u * q + f : 8u * q + 4u + f)] = hi[f];
            }
        }
    } else if (BLK_PROBE_HALF) {
        half v[32];
        tm_sub32(sc, pay, sub, v, BLK_PROBE_A16);
        for (uint l = 0; l < 32u; ++l) { out[sb * 32u + l] = float(v[l]); }
    } else {
        float v[32];
        tm_sub32(sc, pay, sub, v, BLK_PROBE_A16);
        for (uint l = 0; l < 32u; ++l) { out[sb * 32u + l] = v[l]; }
    }
}

// WQ is the WEIGHT FORMAT this instantiation stages: 4 for Q4_0 (18-byte blocks, nibble
// pairs), 8 for Q8_0 (34-byte blocks, signed bytes). Both are 32 elements wide, so the
// chunking, the tile and the whole multiply loop are shared; only `stage` reads bytes
// differently. The register-tiled design against the staged one (st_gemm) is therefore
// the same comparison at either format: rt_gemm<Q8> and st_gemm<Q8> multiply identical
// half operands in the same k order.
template<uint NA, uint NB, uint SGX, uint SGY, bool HALF_A = false, uint WQ = 4u,
         bool GATED = false>
static void rt_gemm(
    device const uchar * weights, device const float * x, device float * y,
    ulong w_offset, ulong w_offset2, uint n_in, uint n_out, uint n_tok, uint src_row,
    threadgroup float * shared, uint3 tgid, uint tid, uint tcount, uint sgid, uint lane,
    uint skip, uint epilogue, device half * xh2, uint epi_half)
{
    static_assert(WQ == 4u || WQ == 8u, "rt_gemm stages Q4_0 or Q8_0 weights");
    // GATED (see THE GATED PAIR above): `w_offset` is the gate tensor, `w_offset2` the up
    // tensor, `n_out` the OUTPUT width, and the grid walks 2 * n_out virtual rows. The
    // pair lives inside one row tile only if the tile holds whole 16-row groups.
    static_assert(!GATED || (NB % 2u) == 0u,
                  "a gated rt_gemm simdgroup must hold whole gate/up pairs");
    static_assert(QK4_0 == QK8_0, "one chunking for both formats needs one block width");
    constexpr uint RT_ROWS   = NB * 8u * SGX;           // output rows per threadgroup
    constexpr uint RT_TOKENS = NA * 8u * SGY;           // tokens per threadgroup
    // Two Q4_0 blocks per row per chunk. At RT_K == QK4_0 a 2560-wide matmul runs 80
    // chunks, each with a threadgroup barrier and a loop trip; doubling the chunk halves
    // both. The staged tile doubles to RT_K x RT_WS, which is why the double buffering is
    // gone -- it measured worth nothing (416.3 against 418.4) and two 64-deep tiles would
    // not fit in 32 KB anyway.
    // k-chunk of 128 = FOUR Q4_0 blocks per row.
    //
    // This is the whole GEMM. At RT_K = 32 a row contributes 18 bytes per chunk while the
    // cache line is 128, so 86% of the kernel's time went to device weight reads running at
    // an effective 20 GB/s on a 150 GB/s machine -- a 7x fetch amplification that matches
    // the shortfall exactly. Removing the read (skip bit2) takes ffn_down from 250 ms to
    // 36 ms, which is what everything else in the kernel costs put together.
    //
    // Four blocks is 72 contiguous bytes per row per chunk. The staged tile grows with
    // RT_K, which is why it is HALF here: 128 x (rows+2) halves is 16.9 KB, where float
    // would be 33.8 and would not fit. Half costs Q4_0 one rounding: (x-8)*d is exact in
    // float (a 4-bit integer times d's 16 bits) and the tile holds it rounded once to
    // half. It is not exact in half: a factor of 3, 5, 6 or 7 needs more mantissa bits
    // than half has.
    // k-chunk depth follows the row tile, so the staged tile stays around 16 KB whatever
    // the shape. Both terms of the traffic matter and they pull opposite ways:
    //   activations = (n_out / RT_ROWS) x RT_TOKENS x n_in x 4
    //   weights     = (n_tok / RT_TOKENS) x n_out x n_in x 0.5625
    // a wider row tile cuts the activation term, a deeper chunk cuts the per-line waste on
    // the weight term, and threadgroup memory has to hold RT_K x RT_ROWS halves.
    // 64 was tried for the extra occupancy an 8.4 KB tile buys: 4.11 TFLOPS against 4.27.
    // Contiguity of the weight read beats residency.
    // k-chunk of 64 = two Q4_0 blocks per row.
    //
    // Swept AFTER the loop was software pipelined, which changed the answer: 128 won before
    // (4.27 against 4.11 at 64) and loses now (4.32 against 4.41), because a shallower chunk
    // halves the staged tile and lets more threadgroups stay resident -- which only pays
    // once the loop has latency left to hide. 32 gives back more in per-line waste on the
    // weight read than it gains: 4.24.
    constexpr uint RT_K      = 64u;
    // Weight tile as [k][row]. Three layouts were implemented and measured, and they land
    // within 1.5% of each other, so the staging LAYOUT is not where the time goes:
    //
    //   [k][row], strided writes                 418.4 tok/s   <- this one
    //   [row][k], contiguous writes + transpose  412.0
    //   tile-major, 8x8 blocks contiguous        400.7
    //
    // SINGLE staged tile, and the reason is at the chunk loop below with its numbers:
    // double buffering was implemented and measured WORSE. This comment used to claim
    // the kernel was double buffered, which contradicted the loop it describes.
    //
    // What the deeper chunk does buy: llama.cpp dequantises a 440-token batch 14 times
    // where this kernel does it 7, because RT_K is 64 against its NK of 32.
    // Weight tile as [k][row]; activations read straight from device.
    //
    // llama.cpp's mul_mm stages BOTH operands tile-major, so every simdgroup_load reads one
    // contiguous 64-float run at elements_per_row = 8, and it reaches ~4.85 TFLOPS on this
    // GPU. That exact arrangement was implemented here and measured 3.13 against 3.93 for
    // this one. Each half had also been measured alone: tile-major weights 3.83, staged
    // activations 3.87. All three lose. On this GPU the activation read wants to come from
    // device, and the staged weight tile wants the cheap scattered write, not the layout
    // that makes the matrix load contiguous.
    constexpr uint RT_WS     = RT_ROWS + 2u;   // staged weight tile row stride, [k][row]
    // Activations are NOT staged. Doing so was implemented twice -- once tangled with a
    // tile-major weight layout, once cleanly with float4 writes on the best config -- and
    // measured slower both times (3.87 against 3.98 TFLOPS). llama.cpp stages both
    // operands; on this GPU reading activations from device wins.

    // Everything is FLOAT: activations, staged weight tile, and accumulator.
    //
    // Half operands were tried -- the tile staged as half and activations converted by a
    // pre-pass, both feeding a float accumulator, which this compiler accepts. MEASURED
    // WORTH NOTHING on M3 Pro (the matrix units evidently do not run fp16 at twice the
    // rate), while the conversion shifted logits in the third decimal against llama.cpp.
    // No speed for lost exactness is not a trade, so the float tile stands.
    // Half tile: float measured 4.07 against 4.11 at matched memory, and half is what
    // lets RT_K reach 128 at 64 rows.
    threadgroup half * ws = (threadgroup half *)shared;   // weights, [k][row]
    // Activations are read from DEVICE, not staged, and that is measured rather than
    // assumed. llama.cpp stages BOTH operands and multiplies from threadgroup memory at
    // stride 8, which looks like the obvious fix for the strided device reads the APF=5
    // prefetch below exists to hide. Staging them here, as float, [token][k]:
    //
    //   26219 / 26263 / 26371 ms against 11517-11537     2.3x SLOWER, logits identical
    //
    // Threadgroup memory goes 8448 -> 24832 bytes and occupancy collapses. The fork gets
    // away with it because its tile is different in three ways at once: NK 32 rather than
    // RT_K 64, operands stored as HALF, and a 64x32 output tile -- sa 4096 B plus sb
    // 2048 B, six KB total, so many threadgroups stay resident. Every row-block
    // threadgroup stages the SAME activation tile, 40 of them for n_out 2560, which is
    // cheap at 6 KB and ruinous at 24.8.
    //
    // So this is not a patch to bolt on: matching it means adopting the fork's tile shape,
    // k-slice and half staging together.

    // EXPERIMENT: which operand do CONCURRENT threadgroups share? Metal walks x fastest,
    // so with x = row blocks the neighbours all read the same activation tile (655 KB at
    // rt_toks 64) and stream different weights; with x = token tiles they share the weight
    // tile (92 KB) and stream different activations. The kernel is latency-bound on the
    // activation reads, so this is the one lever that changes their cache behaviour
    // without changing the tile shape.
    // Metal walks x fastest, so x decides which operand CONCURRENT threadgroups share:
    // x = row blocks means neighbours read the SAME activation tile and stream different
    // weights. Swapping the grid so they share the weight tile instead was measured --
    // 11520 and 11528 ms against 11517-11537 -- exactly nothing, logits unchanged. The
    // kernel is latency-bound on the activation reads, but not because of this ordering.
    const uint vrows = GATED ? 2u * n_out : n_out;   // rows the grid walks
    const uint r0 = tgid.x * RT_ROWS;
    const uint t0 = tgid.y * RT_TOKENS;
    if (r0 >= vrows || t0 >= n_tok) { return; }
    const uint nrow = min(RT_ROWS, vrows - r0);
    const uint ntok = min(RT_TOKENS, n_tok - t0);
    const uint blocks = n_in / QK4_0;

    // Where this simdgroup's accumulator grid sits inside the threadgroup tile.
    const uint rx = (sgid % SGX) * NB * 8u;             // row offset
    const uint ty = (sgid / SGX) * NA * 8u;             // token offset

    simdgroup_float8x8 mc[NA * NB];
    #pragma unroll
    for (uint i = 0; i < NA * NB; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Dequantise chunk `c00` into `dst`. RT_K is 64 = TWO Q4_0 blocks per row per chunk --
    // a stale version of this comment said one, and that assumption produced the
    // wrong-logits failures of both whole-block attempts.
    auto stage = [&](uint c00, threadgroup half * dst) {
        const uint bi = c00 / QK4_0;
        if (WFMT != WF_ROWMAJOR) {
            // THE TILE-MAJOR FAMILY. One thread per (row, 32-value sub-block), the same
            // shape as the two arms below -- what changes is where the bytes are, and
            // `tm_sub32` is the only thing that knows the format. A super-block's eight
            // sub-blocks span four chunks at RT_K = 64, so the chunk index picks the block
            // and the sub-block inside it; no chunking changed.
            const uint sub_per_block = tm_block_elems() / QK4_0;
            const uint bb_bytes = tm_block_bytes();
            const uint sc_bytes = tm_scale_bytes();
            const uint pay_bytes = bb_bytes - sc_bytes;
            const uint blocks_tm = n_in / tm_block_elems();
            constexpr uint BPCH = RT_K / QK4_0;                 // sub-blocks per row per chunk
            for (uint e = tid; e < RT_ROWS * BPCH; e += tcount) {
                const uint rr = e / BPCH, bb = e % BPCH;
                const bool read = rr < nrow && !(skip & 4u);
                half v[32];
                if (read) {
                    const uint gs  = bi + bb;                   // global sub-block index
                    const uint blk = gs / sub_per_block;
                    const uint sub = gs % sub_per_block;
                    const uint row = gated_src_row<GATED>(r0 + rr);
                    device const uchar * base = weights
                        + (gated_is_up<GATED>(r0 + rr) ? w_offset2 : w_offset);
                    if (WFMT_ROW) {
                        // Row-major: the block's bytes are contiguous, and each format has
                        // ONE scale run and one payload run, so it is still two pointers.
                        device const uchar * b = base
                            + ((ulong)row * blocks_tm + blk) * bb_bytes;
                        const uint so = tm_scale_src_off();
                        tm_sub32(b + so, b + (so == 0u ? sc_bytes : 0u), sub, v, false);
                    } else {
                        const ulong unit = ((ulong)(row / TM_UNIT_ROWS) * blocks_tm + blk)
                                         * (ulong)(TM_UNIT_ROWS * bb_bytes);
                        const uint slot = row % TM_UNIT_ROWS;
                        tm_sub32(base + unit + (ulong)slot * sc_bytes,
                                 base + unit + (ulong)TM_UNIT_ROWS * sc_bytes
                                      + (ulong)slot * pay_bytes,
                                 sub, v, true);
                    }
                } else {
                    #pragma unroll
                    for (uint l = 0; l < 32u; ++l) { v[l] = half(0.0h); }
                }
                const uint k0 = bb * QK4_0;
                #pragma unroll
                for (uint l = 0; l < 32u; ++l) { dst[(k0 + l) * RT_WS + rr] = v[l]; }
            }
            return;
        }
        if (WQ == 8u) {
            // ONE THREAD PER 34-BYTE Q8_0 BLOCK: the half scale, then 32 signed bytes into
            // 32 consecutive k slots of the [k][row] tile. `half(q * d)` is exactly the
            // rounding st_gemm applies when it stages, so the two Q8 designs feed the
            // matrix units identical operands. Dead rows stage zeros, as the Q4 arm does.
            // The device read of the NEXT chunk's bytes into registers, to overlap this
            // chunk's multiplies, was built and measured: -2.0% on the 17123-token prefill
            // (787 against 803 tok/s, three interleaved rounds). Its ceiling was the read's
            // whole cost, +3.1% with the reads removed, and nine more live registers on
            // top of the activation pipeline cost more than the latency they hid.
            constexpr uint BPCH = RT_K / QK8_0;   // blocks per row per chunk: TWO
            for (uint e = tid; e < RT_ROWS * BPCH; e += tcount) {
                const uint rr = e / BPCH, bb = e % BPCH;
                const bool read = rr < nrow && !(skip & 4u);
                float d = 0.0f;
                device const uchar * blk = weights;
                if (read) {
                    blk = weights + (gated_is_up<GATED>(r0 + rr) ? w_offset2 : w_offset)
                        + (ulong)gated_src_row<GATED>(r0 + rr) * blocks * Q8_0_BYTES
                        + (bi + bb) * Q8_0_BYTES;
                    d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
                }
                const uint k0 = bb * QK8_0;
                #pragma unroll
                for (uint g = 0; g < 8u; ++g) {
                    char4 q = char4(0);
                    if (read) {
                        q = char4(*(device const packed_char4 *)(blk + 2u + g * 4u));
                    }
                    // Same exact product rounded once (see the dequant-form note above).
                    const half4 v4h = half4(q) * half(d);
                    #pragma unroll
                    for (uint j = 0; j < 4u; ++j) {
                        dst[(k0 + g * 4u + j) * RT_WS + rr] = v4h[j];
                    }
                }
            }
            return;
        }
        // One thread dequantises one WHOLE 18-byte block: one scale read and one base
        // address instead of four of each, and a quarter of the loop iterations. Same
        // values to the same slots in the same k-mapping as the 4-byte-per-thread form it
        // replaced, so bit-identical -- verified (n=2000 exact) -- and measured 11038 ->
        // 10729 ms on a 5642-token prefill. The 2025 attempts at this failed with wrong
        // logits because a stale comment said a chunk was ONE block per row; RT_K is 64,
        // so it is two, and the old code's kk range only covered half the staged tile.
        //
        // packed_uchar4, not uchar4: a Q4_0 block is 18 bytes, so `blk + 2` is 4-byte
        // aligned only every other block; uchar4 carries alignment 4 and Metal 2.5 lets
        // the compiler assume it. See llama.cpp 30de65202.
        {
            constexpr uint BPCH = RT_K / QK4_0;   // blocks per row per chunk: TWO
            for (uint e = tid; e < RT_ROWS * BPCH; e += tcount) {
                const uint rr = e / BPCH, bb = e % BPCH;
                const bool live = rr < nrow;
                const bool read = live && !(skip & 4u);
                float d = live ? 1.0f : 0.0f;
                device const uchar * blk = weights;
                if (read) {
                    blk = weights + (gated_is_up<GATED>(r0 + rr) ? w_offset2 : w_offset)
                        + (ulong)gated_src_row<GATED>(r0 + rr) * blocks * Q4_0_BYTES
                        + (bi + bb) * Q4_0_BYTES;
                    d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
                }
                const uint k0 = bb * QK4_0;
                #pragma unroll
                for (uint g = 0; g < 4u; ++g) {
                    uchar4 quad = uchar4(0x88);
                    if (read) {
                        quad = uchar4(*(device const packed_uchar4 *)(blk + 2u + g * 4u));
                    }
                    // nibble - 8 is exact in half and its product with the half scale is
                    // exact in float, so the half-arithmetic form rounds the same product
                    // once: bit-identical, one convert and one multiply fewer per element.
                    const half dh = half(d);
                    const half4 lo4h = (half4(quad & uchar4(0x0F)) - half(8.0h)) * dh;
                    const half4 hi4h = (half4(quad >> 4)           - half(8.0h)) * dh;
                    #pragma unroll
                    for (uint j = 0; j < 4u; ++j) {
                        const uint kk = k0 + g * 4u + j;
                        dst[kk * RT_WS + rr]         = lo4h[j];
                        dst[(kk + 16u) * RT_WS + rr] = hi4h[j];
                    }
                }
            }
            return;
        }
    };

    // Single staged tile.
    //
    // Staging and multiplying are serialised by the chunk barrier -- the kernel costs
    // 218 ms where the multiplies are 185 and the staging 33, and 185 + 33 is exactly 218.
    // Double buffering to overlap them measured WORSE, 230 against 218: two tiles are
    // 16.9 KB against 8.4, and the occupancy that costs is worth more than the overlap.
    for (uint c00 = 0; c00 < n_in; c00 += RT_K) {
        threadgroup half * cur = ws;
        if (!(skip & 2u)) { stage(c00, ws); }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Diagnostic bitmask: bit0 skips the multiplies, bit1 skips the dequantisation.
        // Running each separately attributes this kernel between staging and arithmetic;
        // the branch is uniform across the threadgroup and costs nothing when off.
        // SOFTWARE PIPELINED: the next k-step's operands load before the current step's
        // multiplies, so their latency overlaps the arithmetic instead of preceding it.
        //
        // Everything measured says this kernel is latency-bound on operand loads, not
        // bandwidth- or occupancy-bound: the identical loop reaches 12.8 TFLOPS when its
        // device reads hit a cached window against 4.2 here, while halving the bytes,
        // varying threadgroup memory from 2 to 32 KB, and swapping half for float each
        // changed nothing. Two operand sets cost NA + NB more matrices in registers.
        //
        // Activations come straight from device memory; x is [token][k] with stride n_in.
        // Tokens past ntok read live buffer data, which is harmless because token row j
        // only reaches output row j, masked at write-back.
        constexpr uint KSTEPS = RT_K;
        // A is prefetched TWO steps ahead, B one.
        //
        // They come from different places and have different latencies: B is threadgroup
        // memory, A is device memory that all n_out/RT_ROWS row-block threadgroups read in
        // common, so it lands in L2 rather than L1. One step of overlap covers the
        // threadgroup load; the device load needs more.
        // A prefetched APF steps ahead, B one.
        //
        // A comes from device memory that every row-block threadgroup reads in common, so
        // it lands in L2; B is threadgroup memory. Different latencies want different
        // depths, and deeper keeps paying until the registers run out:
        //   1 step 4.41   2: 4.48   3: 4.56   4: 4.69   5: 4.71   6: 4.70
        // Each step costs NA more matrices held live, and the chunk is only RT_K/8 steps
        // deep, so the depth cannot usefully exceed it.
        //
        // THE CHUNK BOUND WAS TESTED AND IT IS NOT THE THING TO REMOVE (2026-09-09).
        // A is one sequential stream over the whole reduction and the barrier below
        // belongs to the WEIGHT tile, so the fetch CAN run past c00 + RT_K: hoist these
        // registers and their fill out of the chunk loop and index the fetch by global k.
        // Built, bit-identical (top-10 logits equal to the digit), and SLOWER -- rotated
        // OLD/NEW/NEW/OLD, Qwen3.8-27B 512-token prefill, ms per chunk:
        //
        //   old  5017.8  4989.6  4998.2  4983.7      median 4994
        //   new  8118.7  5323.9  5354.9  5321.9      median 5323      +6.6%
        //
        // What it costs is register RESIDENCY, not loads: the counts are identical (eight
        // fetch sets per chunk either way, five in the prologue plus three in the loop
        // against eight in the loop). Declared inside the loop, A's APF+1 matrices are
        // dead across `stage`, and the dequantiser can have those registers; declared
        // outside, they are live through it and the two halves of the kernel no longer
        // share. The drain at a chunk edge is real and it is cheaper than the sharing.
        constexpr uint APF = 5u;
        simdgroup_float8x8 a[APF + 1u][NA];
        // B prefetched BPF steps ahead. It comes from threadgroup memory, which is far
        // lower latency than A's device reads, so one step is enough: 2 measured 4.66
        // against 4.70. A needs five (4.41 at one), and that asymmetry is the point --
        // operands from different memories want different depths.
        constexpr uint BPF = 1u;
        simdgroup_half8x8  b[BPF + 1u][NB];
        // HALF_A: the activations were converted to half (into the XH scratch bound as x)
        // and load as half fragments -- llama.cpp's mul_mm stages the same way. Only the
        // OPERAND rounds; the accumulator stays float. The half pipeline mirrors the float
        // one below; the dead branch of the compile-time constant costs nothing.
        // HALF_A reads the XH scratch (device half, converted by a cvt pass with a
        // host-side cache). Staging the tile to THREADGROUP instead was measured 73%
        // slower here: the extra 9 KB collapses occupancy, the same failure float
        // staging had. The fork affords in-kernel staging only because its whole tile
        // is ~6 KB; ours is not, so the conversion rides through device memory.
        device const half * xh = (device const half *)x;
        simdgroup_half8x8 ah[HALF_A ? APF + 1u : 1u][NA];
        if (!(skip & 1u)) {
            #pragma unroll
            for (uint i = 0; i < NA; ++i) {
                const ulong base = (ulong)(src_row + t0 + ty + i * 8u) * n_in + c00;
                #pragma unroll
                for (uint d = 0; d < APF; ++d) {
                    if (HALF_A) {
                        simdgroup_load(ah[d][i], xh + base + min(d * 8u, KSTEPS - 1u), n_in);
                    } else {
                        simdgroup_load(a[d][i], x + base + min(d * 8u, KSTEPS - 1u), n_in);
                    }
                }
            }
            #pragma unroll
            for (uint d = 0; d < BPF; ++d) {
                #pragma unroll
                for (uint j = 0; j < NB; ++j) {
                    simdgroup_load(b[d][j], cur + min(d * 8u, KSTEPS - 8u) * RT_WS
                                                + rx + j * 8u, RT_WS);
                }
            }

            #pragma unroll
            for (uint k = 0; k < KSTEPS; k += 8u) {
                // THE SAME SCHEDULING FENCES the Q8 GEMM gained, asked of a kernel that
                // already has an explicit pipeline. This one rotates prefetch registers
                // by hand, so the data dependency through the rotation ALREADY imposes an
                // order the compiler must respect -- the Q8 kernel had neither and gained
                // 5.1%. RT_MMA_FENCE (constant 16) is the lever; see its declaration.
                // Placed the way llama.cpp's mul_mm places them and the way the Q8 kernel
                // does: one before each operand group and one before the multiplies.
                if (RT_MMA_FENCE) { simdgroup_barrier(mem_flags::mem_none); }
                if (k + APF * 8u < KSTEPS) {
                    #pragma unroll
                    for (uint i = 0; i < NA; ++i) {
                        if (HALF_A) {
                            simdgroup_load(ah[APF][i],
                                           xh + (ulong)(src_row + t0 + ty + i * 8u) * n_in
                                             + c00 + k + APF * 8u, n_in);
                        } else {
                            simdgroup_load(a[APF][i],
                                           x + (ulong)(src_row + t0 + ty + i * 8u) * n_in
                                             + c00 + k + APF * 8u, n_in);
                        }
                    }
                }
                if (RT_MMA_FENCE) { simdgroup_barrier(mem_flags::mem_none); }
                if (k + BPF * 8u < KSTEPS) {
                    #pragma unroll
                    for (uint j = 0; j < NB; ++j) {
                        simdgroup_load(b[BPF][j],
                                       cur + (k + BPF * 8u) * RT_WS + rx + j * 8u, RT_WS);
                    }
                }
                if (RT_MMA_FENCE) { simdgroup_barrier(mem_flags::mem_none); }
                #pragma unroll
                for (uint i = 0; i < NA; ++i) {
                    #pragma unroll
                    for (uint j = 0; j < NB; ++j) {
                        if (HALF_A) {
                            simdgroup_multiply_accumulate(mc[i * NB + j], ah[0][i], b[0][j],
                                                          mc[i * NB + j]);
                        } else {
                            simdgroup_multiply_accumulate(mc[i * NB + j], a[0][i], b[0][j],
                                                          mc[i * NB + j]);
                        }
                    }
                }
                #pragma unroll
                for (uint d = 0; d < APF; ++d) {
                    #pragma unroll
                    for (uint i = 0; i < NA; ++i) {
                        if (HALF_A) { ah[d][i] = ah[d + 1u][i]; }
                        else        { a[d][i]  = a[d + 1u][i]; }
                    }
                }
                #pragma unroll
                for (uint d = 0; d < BPF; ++d) {
                    #pragma unroll
                    for (uint j = 0; j < NB; ++j) { b[d][j] = b[d + 1u][j]; }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back STRAIGHT TO DEVICE, as llama.cpp's mul_mm does.
    //
    // Going through threadgroup memory caps RT_ROWS: the write-back tile is
    // RT_TOKENS x (RT_ROWS + 2) floats, which at 128 rows x 64 tokens is 33 KB and does not
    // fit. RT_ROWS is what decides how many times a batch re-reads its ACTIVATIONS --
    // n_out / RT_ROWS threadgroups each pull the whole tile from device, 40 times over for
    // ffn_down -- so the cap was costing far more than the masking was worth.
    //
    // Rows CAN straddle the end. Every tile-major matmul of the first files had n_out a
    // multiple of 64, and this store wrote its whole 64-row tile on that assumption; a
    // file whose 48-wide gated-delta projections (ssm_alpha, ssm_beta) are block quants
    // (byteshape's IQ4_XS-4.40bpw: IQ4_XS, Q5_K, Q6_K, IQ3_XXS on them) sends n_out = 48
    // here, and the unguarded tile wrote columns 48..63 of every token into the next
    // token's row -- one prompt token agreed with the CPU backend, two or more did not
    // (docs/evidence/bracket/2026-09-11-27b-dequant-tables.md). So the absent 8x8 blocks are
    // skipped. The full-tile case keeps the unconditional loop (the same code as before).
    //
    // AN 8x8 BLOCK CAN STRADDLE n_out TOO, when n_out is not a multiple of 8 (a tile-major
    // n_out always is; a Q4_0 or Q8_0 row-major one need not be). Stored whole, its columns
    // past n_out are the next token's first outputs, written with zeros (dead rows stage zero
    // weights) in a race with the row tile that owns them, and the last token's run past the
    // buffer: Q4_0 at n_out 4097 wrote 7 floats past an 8-row output and changed the first
    // outputs of a random row in 2 of 5 identical runs. Each lane of that block stores its
    // own two elements, stopping at n_out -- in the RT_EDGE8 variant only, which the host
    // builds for such an n_out. Compiled into every pipeline, the straddle store never ran on
    // E4B (its n_out are multiples of 64) and still cost its prefill 16%: 1042 -> 876 tok/s
    // at 5643 tokens, in either form (per lane, or through the staging tile).
    //
    // Tokens can straddle, so the caller rounds the activation buffers up to a whole token
    // tile; rows past the token count receive values nothing reads.
    // A GATED dispatch always has an activation (the host refuses it otherwise), so it
    // never takes the straight store: its rows are gate and up halves, not outputs.
    if (epilogue == 0u && !GATED) {
        if (nrow == RT_ROWS) {
            #pragma unroll
            for (uint i = 0; i < NA; ++i) {
                #pragma unroll
                for (uint j = 0; j < NB; ++j) {
                    device float * yp = y + (ulong)(t0 + ty + i * 8u) * n_out + r0 + rx + j * 8u;
                    if (RT_RESID) { rt_resid_add(mc[i * NB + j], yp, n_out); }
                    simdgroup_store(mc[i * NB + j], yp, n_out);
                }
            }
            return;
        }
        #pragma unroll
        for (uint i = 0; i < NA; ++i) {
            #pragma unroll
            for (uint j = 0; j < NB; ++j) {
                const uint c0 = rx + j * 8u;
                if (!RT_EDGE8 || c0 + 8u <= nrow) {
                    if (c0 < nrow) {
                        device float * yp = y + (ulong)(t0 + ty + i * 8u) * n_out + r0 + c0;
                        if (RT_RESID) { rt_resid_add(mc[i * NB + j], yp, n_out); }
                        simdgroup_store(mc[i * NB + j], yp, n_out);
                    }
                } else if (c0 < nrow) {
                    // Lane l holds row (l/4 & 4) + (l/2 % 4) of the block and the two columns
                    // from (l/4 & 2)*2 + (l % 2)*2 -- the 8x8 layout MLX's steel GEMM stores
                    // by (BaseMMAFrag::get_coord) -- and stores them itself, up to n_out.
                    const uint q = lane / 4u;
                    const uint fr = (q & 4u) + ((lane / 2u) % 4u);
                    const uint fc = (q & 2u) * 2u + (lane % 2u) * 2u;
                    const thread auto & e = mc[i * NB + j].thread_elements();
                    device float * o = y + (ulong)(t0 + ty + i * 8u + fr) * n_out + r0 + c0 + fc;
                    if (c0 + fc < nrow) { o[0] = RT_RESID ? o[0] + e[0] : e[0]; }
                    if (c0 + fc + 1u < nrow) { o[1] = RT_RESID ? o[1] + e[1] : e[1]; }
                }
            }
        }
        return;
    }

    // GATED-ACTIVATION EPILOGUE: y = act(y) * result, act = GELU or SiLU.
    //
    // The FFN otherwise runs gate -> G, up -> U, then a separate gelu_mul(G, U) -> G pass
    // that moves 54 MB per layer. Folding it into the up projection's write-back means the
    // up matmul reads G and writes G, and the separate pass and the U buffer both go away.
    // The accumulators go through threadgroup memory here because a simdgroup_store cannot
    // read-modify-write; that costs one tile round trip against a whole kernel pass.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * ob = shared;
    constexpr uint OB_S = RT_ROWS;
    // UNGATED ONLY. The gated pair below never reads this tile, and the host sizes a gated
    // dispatch at max(staged weight tile, 128 floats per simdgroup) = 8448 B for the
    // 64x64 shape -- this spill is RT_TOKENS x RT_ROWS floats = 16384 B. Until 2026-09-02
    // the loop ran for both variants: 16 dead simdgroup_stores per simdgroup on every
    // gated FFN dispatch, written past the end of the threadgroup allocation.
    if (!GATED) {
        #pragma unroll
        for (uint i = 0; i < NA; ++i) {
            #pragma unroll
            for (uint j = 0; j < NB; ++j) {
                simdgroup_store(mc[i * NB + j], ob + (ty + i * 8u) * OB_S + rx + j * 8u, OB_S);
            }
        }
    }
    if (GATED) {
        // THE PAIR CLOSED INSIDE ITS SIMDGROUP, with no threadgroup round trip. The ungated
        // epilogue below spills the whole RT_TOKENS x RT_ROWS tile and walks it with every
        // thread: two threadgroup barriers and a 16 KB allocation against the plain
        // store's 8.4 KB. Routed that way the pair LOST 0.6-2.3% on E4B (Q4, 64x64) while
        // it gained ~1% on LFM2, whose st_gemm write-back spills either way -- the gate
        // half had been trading a straight register store for the big spill. A simdgroup
        // owns both tiles of each pair, so it stores just those two (512 B of the dead
        // staging tile), multiplies, writes, and moves on; the threadgroup allocation
        // stays at the staged weight tile.
        //
        // ONE threadgroup barrier stands between the last chunk's MMA loop and the first
        // accumulator store here: the one above the ungated spill. A second one used to sit
        // below, kept as "REQUIRED" because without it a 64-token E4B forward gave three
        // different logit vectors in three runs. The mechanism was found on 2026-09-02: the
        // ungated spill (RT_TOKENS x RT_ROWS floats = 16 KB) also ran for this variant, into
        // an 8448 B allocation, so the over-run raced whatever the extra barrier happened to
        // order. With the spill scoped to the ungated path, three runs hash identical without
        // the second barrier, det_gate pins EXACT, and the A/B reads +0.4 / +1.3 %.
        //
        // act(G) * U on the same float bits the two-dispatch form multiplied (a
        // simdgroup_store of the same accumulator, read back), so the same product.
        // Output column of pair m: virtual tile row (r0 + rx + 16 m) maps to
        // (r0 + rx) / 2 + 8 m. Tokens past ntok are masked; rows never straddle n_out.
        //
        threadgroup float * sc = shared + sgid * 128u;
        #pragma unroll
        for (uint i = 0; i < NA; ++i) {
            #pragma unroll
            for (uint m = 0; m < NB / 2u; ++m) {
                simdgroup_store(mc[i * NB + 2u * m],      sc,      16u);
                simdgroup_store(mc[i * NB + 2u * m + 1u], sc + 8u, 16u);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) {
                    const uint tt = e >> 3, cc = e & 7u;
                    const uint tok = ty + i * 8u + tt;
                    if (tok < ntok) {
                        const float v = imparo_act_f(sc[tt * 16u + cc]) * sc[tt * 16u + 8u + cc];
                        const ulong idx = (ulong)(t0 + tok) * n_out
                                        + ((r0 + rx) >> 1) + m * 8u + cc;
                        if (epi_half != 0u) { xh2[idx] = half(v); }
                        else                { y[idx] = v; }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < ntok * nrow; e += tcount) {
        const uint tt = e / nrow, rr = e % nrow;
        device float * slot = y + (ulong)(t0 + tt) * n_out + r0 + rr;
        const float v = imparo_act_f(*slot) * ob[tt * OB_S + rr];
        // Half mirror of G for the down projection (IMPARO_HALF_A): the exact value the
        // cvt pass would produce, so that dispatch -- the largest of the conversions --
        // disappears. The FLOAT store is skipped in mirror mode: the down GEMM reads the
        // mirror, and nothing else reads float G at prefill (G probes under HALF_A read
        // the pre-gelu gate values; IMPARO_HALF_A=0 restores them).
        if (epi_half != 0u) { xh2[(ulong)(t0 + tt) * n_out + r0 + rr] = half(v); }
        else                { *slot = v; }
    }
}

// One named kernel per shape. The host builds all of them and selects by index; the names
// must stay in step with RT_SHAPES on the host side.
#define IMPARO_RT_ENTRY(NAME, NA, NB, SGX, SGY, HALF_A, WQ, GATED)                        \
kernel void NAME(                                                                          \
    device const uchar * weights [[buffer(0)]],                                            \
    device const float * x       [[buffer(1)]],                                            \
    device float       * y       [[buffer(2)]],                                            \
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in  [[buffer(4)]],           \
    constant ulong & w_offset2 [[buffer(10)]],                                             \
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],           \
    constant uint & src_row  [[buffer(7)]], constant uint & skip [[buffer(12)]],           \
    constant uint & epilogue [[buffer(13)]],                                               \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],               \
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  sgid  [[simdgroup_index_in_threadgroup]],                                        \
    uint  lane  [[thread_index_in_simdgroup]])                                             \
{                                                                                          \
    rt_gemm<NA, NB, SGX, SGY, HALF_A, WQ, GATED>(weights, x, y, w_offset, w_offset2,      \
                              n_in, n_out, n_tok,                                          \
                              src_row, shared, tgid, tid3.x, tcnt3.x, sgid, lane, skip,   \
                              epilogue, xh2, epi_half);                                   \
}
// Q4_0 weights, f32 activations (IMPARO_RT_KERNEL) or the half mirror (_H); the Q8_0
// twins follow the _H list. One body, four names, so a signature change lands once.
#define IMPARO_RT_KERNEL(NAME, NA, NB, SGX, SGY)    IMPARO_RT_ENTRY(NAME, NA, NB, SGX, SGY, false, 4u, false)
#define IMPARO_RT_KERNEL_H(NAME, NA, NB, SGX, SGY)  IMPARO_RT_ENTRY(NAME, NA, NB, SGX, SGY, true, 4u, false)
// The gated pair (_gh): half activations, Q4_0 gate and up as one virtual matrix.
#define IMPARO_RT_KERNEL_GH(NAME, NA, NB, SGX, SGY) IMPARO_RT_ENTRY(NAME, NA, NB, SGX, SGY, true, 4u, true)

IMPARO_RT_KERNEL(imparo_rt_0, 2, 4, 2, 2)
IMPARO_RT_KERNEL(imparo_rt_1, 4, 4, 2, 2)
IMPARO_RT_KERNEL(imparo_rt_2, 2, 4, 2, 4)
IMPARO_RT_KERNEL(imparo_rt_3, 2, 4, 4, 2)
IMPARO_RT_KERNEL(imparo_rt_4, 4, 2, 4, 2)
IMPARO_RT_KERNEL(imparo_rt_5, 2, 2, 4, 4)
IMPARO_RT_KERNEL(imparo_rt_6, 4, 4, 2, 4)
IMPARO_RT_KERNEL(imparo_rt_7, 8, 2, 2, 2)
// TWO MORE 128-TOKEN SHAPES WERE BUILT AND MEASURED, then removed. RT_TOKENS decides how
// often a 512-token chunk re-reads the weight matrix -- ceil(512/toks) -- and the measured
// cache knee (~8 MB) puts a 14.1 MB FFN matrix outside cache, so halving the passes from 8
// to 4 should have removed real DRAM traffic. It does the opposite, per-dispatch at the
// prefill shape, median of three interleaved rounds:
//
//   rt_shape 1  (4,4,2,2)   64 tok,  8 passes   3077.9 us   incumbent
//   rt_shape 0  (2,4,2,2)   32 tok, 16 passes   3042.5 us   TWICE the re-reads, no worse
//   rt_shape 8  (4,2,2,4)  128 tok,  4 passes   3295.5 us   half the re-reads, 7% slower
//   rt_shape 9  (4,1,4,4)  128 tok,  4 passes   4524.5 us   47% slower
//
// Both survived the register screen, so this is not spilling -- shape 7's 128-token tile
// does spill (292x) and is screened out. Weight re-reads simply are not what limits this
// kernel: ablation puts the GEMM at 66.6% of prefill in multiplies against 12.0% in
// staging, and doubling the staging traffic costs nothing measurable. Do not re-derive
// the traffic argument; it is right about the bytes and wrong about the bottleneck.

// Half-activation variants (IMPARO_HALF_A): x is the XH scratch, converted by
// imparo_cvt_f32_f16. One per shape, so the sweep stays possible.
IMPARO_RT_KERNEL_H(imparo_rt_0_h, 2, 4, 2, 2)
IMPARO_RT_KERNEL_H(imparo_rt_1_h, 4, 4, 2, 2)
IMPARO_RT_KERNEL_H(imparo_rt_2_h, 2, 4, 2, 4)
IMPARO_RT_KERNEL_H(imparo_rt_3_h, 2, 4, 4, 2)
IMPARO_RT_KERNEL_H(imparo_rt_4_h, 4, 2, 4, 2)
IMPARO_RT_KERNEL_H(imparo_rt_5_h, 2, 2, 4, 4)
IMPARO_RT_KERNEL_H(imparo_rt_6_h, 4, 4, 2, 4)
IMPARO_RT_KERNEL_H(imparo_rt_7_h, 8, 2, 2, 2)
IMPARO_RT_KERNEL_GH(imparo_rt_0_gh, 2, 4, 2, 2)
IMPARO_RT_KERNEL_GH(imparo_rt_1_gh, 4, 4, 2, 2)
IMPARO_RT_KERNEL_GH(imparo_rt_2_gh, 2, 4, 2, 4)
IMPARO_RT_KERNEL_GH(imparo_rt_3_gh, 2, 4, 4, 2)
IMPARO_RT_KERNEL_GH(imparo_rt_4_gh, 4, 2, 4, 2)
IMPARO_RT_KERNEL_GH(imparo_rt_5_gh, 2, 2, 4, 4)
IMPARO_RT_KERNEL_GH(imparo_rt_6_gh, 4, 4, 2, 4)
IMPARO_RT_KERNEL_GH(imparo_rt_7_gh, 8, 2, 2, 2)

// rt_gemm<Q8>: the register-tiled design on Q8_0 weights, one entry per RT_SHAPES row so
// the tuner can rank the tile for this format (Q8 weights are 1.9x the bytes of Q4 per
// element, so the winning shape need not be Q4's). The half-mirror twin of each, as above.
IMPARO_RT_ENTRY(imparo_rt8_0, 2, 4, 2, 2, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_1, 4, 4, 2, 2, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_2, 2, 4, 2, 4, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_3, 2, 4, 4, 2, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_4, 4, 2, 4, 2, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_5, 2, 2, 4, 4, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_6, 4, 4, 2, 4, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_7, 8, 2, 2, 2, false, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_0_h, 2, 4, 2, 2, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_1_h, 4, 4, 2, 2, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_2_h, 2, 4, 2, 4, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_3_h, 2, 4, 4, 2, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_4_h, 4, 2, 4, 2, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_5_h, 2, 2, 4, 4, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_6_h, 4, 4, 2, 4, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_7_h, 8, 2, 2, 2, true, 8u, false)
IMPARO_RT_ENTRY(imparo_rt8_0_gh, 2, 4, 2, 2, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_1_gh, 4, 4, 2, 2, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_2_gh, 2, 4, 2, 4, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_3_gh, 2, 4, 4, 2, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_4_gh, 4, 2, 4, 2, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_5_gh, 2, 2, 4, 4, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_6_gh, 4, 4, 2, 4, true, 8u, true)
IMPARO_RT_ENTRY(imparo_rt8_7_gh, 8, 2, 2, 2, true, 8u, true)

// Narrow-N tile for 2..15-token batches (task #11, modeled on the fork's
// kernel_mul_mm_nb8, author liuliquan): 64 rows x 8 tokens, 2 simdgroups, 64
// threads, 4 accumulators each. The weight stage (RT_K x 66 halves) is the same
// 8.4 KB whatever the token width, so cost is FLAT in n_tok while a 64-token
// tile wastes up to 56/64 of its arithmetic on a narrow batch. Same k-order into
// every output as any other shape, so routing through it is BIT-IDENTICAL.
// This is also the #5 story's end state: with 2..15 covered here, no multi-token
// batch can reach the GEMV in shipping (the wobble's only reachable expression).
// The template arguments come from the host (imparo_metal.mm: NB8_NA..NB8_SGY) because the
// DISPATCH GRID is n_out/RT_ROWS by n_tok/RT_TOKENS, and RT_ROWS = NB*8*SGX is computed
// from them. Both sides said 64 and 8 independently.
#ifndef NB8_NA
#define NB8_NA 1
#define NB8_NB 4
#define NB8_SGX 2
#define NB8_SGY 1
#endif
IMPARO_RT_KERNEL(imparo_rt_nb8, NB8_NA, NB8_NB, NB8_SGX, NB8_SGY)
IMPARO_RT_KERNEL_H(imparo_rt_nb8_h, NB8_NA, NB8_NB, NB8_SGX, NB8_SGY)
IMPARO_RT_KERNEL_GH(imparo_rt_nb8_gh, NB8_NA, NB8_NB, NB8_SGX, NB8_SGY)
// Variant b: the same 64x8 tile on ONE simdgroup (32 threads, 8 accumulators).
// Which occupancy shape wins is a device property, so it is a tuned knob.
IMPARO_RT_KERNEL(imparo_rt_nb8b, 1, 8, 1, 1)
IMPARO_RT_KERNEL_H(imparo_rt_nb8b_h, 1, 8, 1, 1)
IMPARO_RT_KERNEL_GH(imparo_rt_nb8b_gh, 1, 8, 1, 1)


kernel void imparo_q4_0_matmat_prefill(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in    [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok   [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    threadgroup float * shared [[threadgroup(0)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]])
{
    // f32 tiles. Half tiles halve threadgroup memory, and occupancy is the binding
    // constraint here, so they looked like the obvious lever -- but MEASURED they are both
    // slower (157.9 -> 148.6 tok/s) and less accurate: the accumulator is half too, and
    // 2560 accumulation steps shifted logits ~0.15 and reshuffled ranks 7-10. Losing on
    // both axes means there is no trade to weigh.
    const uint tid = tid3.x, tcount = tcnt3.x;
    threadgroup float * ws = shared;                        // SG_K x SG_WS (transposed)

    const uint r0 = tgid.x * SG_ROWS;
    const uint t0 = tgid.y * SG_TOKENS;
    if (r0 >= n_out || t0 >= n_tok) { return; }
    const uint nrow = min(SG_ROWS, n_out - r0);
    const uint ntok = min(SG_TOKENS, n_tok - t0);
    const uint blocks = n_in / QK4_0;

    // EXACTLY two accumulators here. The note this comment used to carry -- that the live
    // accumulator count is "a hard performance cliff" -- drew the wrong conclusion from the
    // 157.6 -> 83.5 tok/s measurement behind it: that variant also added a RUNTIME BRANCH
    // which blocked unrolling, and the branch is what cost the time.
    //
    // Sweeping shapes in rt_gemm settled it. A cliff exists, but between 8 and 16:
    //   4 acc 335 tok/s | 8 acc 421 tok/s | 16 acc 85-106 tok/s
    // Two accumulators is not a constraint, it is simply a small tile -- and rt_gemm at
    // eight is 1.5x faster than this kernel. This one is kept only as the A/B reference.
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint c00 = 0; c00 < n_in; c00 += SG_K) {
        // Dequantise PER BLOCK, not per value.
        //
        // The previous loop gave one output value to each thread, so for every one of the
        // 32 values in a Q4_0 block it recomputed the row pointer and re-loaded and
        // re-converted that block's single f16 scale -- 32x redundant on both. A block is
        // 16 bytes under one scale, so here a thread takes FOUR bytes: it reads the scale
        // once and emits eight values, since byte i carries value i in its low nibble and
        // value i+16 in its high one.
        //
        // SG_K == QK4_0, so a chunk is exactly one block per row and `bi` is constant
        // across the staging loop rather than being recomputed per value.
        const uint bi = c00 / QK4_0;
        const uint groups_per_row = 4u;                  // 16 payload bytes / 4
        for (uint e = tid; e < SG_ROWS * groups_per_row; e += tcount) {
            const uint rr = e / groups_per_row;
            const uint g  = e % groups_per_row;
            const bool live = rr < nrow && c00 < n_in;
            float d = 0.0f;
            device const uchar * blk = weights;
            if (live) {
                blk = weights + w_offset
                    + (ulong)(r0 + rr) * blocks * Q4_0_BYTES + bi * Q4_0_BYTES;
                d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
            }
            #pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                const uint i = g * 4u + j;
                const uchar packed = live ? blk[2u + i] : uchar(0x88);
                ws[i * SG_WS + rr]         = (float(packed & 0x0F) - 8.0f) * d;
                ws[(i + 16u) * SG_WS + rr] = (float(packed >> 4)   - 8.0f) * d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile = SKIP_MMA ? SG_TILES : sgid; tile < SG_TILES; tile += nsg) {
            const uint tt = (tile / (SG_ROWS / 8u)) * 8u;   // token offset of this tile
            const uint rr = (tile % (SG_ROWS / 8u)) * 8u;   // row offset
            simdgroup_float8x8 a, w;
            simdgroup_float8x8 c = (tile == sgid) ? acc0 : acc1;
            #pragma unroll
            for (uint k = 0; k < SG_K; k += 8u) {
                // Activations are read STRAIGHT FROM DEVICE MEMORY.
                //
                // Staging them cost 1024 device loads plus 1024 threadgroup writes per
                // chunk, repeated by all n_out/SG_ROWS row-block threadgroups, and bought
                // nothing: simdgroup_load takes a device pointer, and x is already laid out
                // [token][k] with stride n_in, which is exactly the tile wanted. Removing
                // it also drops threadgroup memory from 14.3 KB to 9.2 KB.
                //
                // Rows past ntok read live buffer data rather than zeros; that is harmless
                // because row j of A only reaches row j of C, and those rows are masked at
                // write-back.
                simdgroup_load(a, x + (ulong)(src_row + t0 + tt) * n_in + c00 + k, n_in);
                simdgroup_load(w, ws + k * SG_WS + rr, SG_WS);
                simdgroup_multiply_accumulate(c, a, w, c);
            }
            if (tile == sgid) { acc0 = c; } else { acc1 = c; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // write back through the activation tile, which is free by now, so partial edges mask
    for (uint tile = sgid; tile < SG_TILES; tile += nsg) {
        const uint tt = (tile / (SG_ROWS / 8u)) * 8u, rr = (tile % (SG_ROWS / 8u)) * 8u;
        simdgroup_store((tile == sgid) ? acc0 : acc1, shared + tt * SG_OUT + rr, SG_OUT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < ntok * nrow; e += tcount) {
        const uint tt = e / nrow, rr = e % nrow;
        y[(ulong)(t0 + tt) * n_out + r0 + rr] = shared[tt * SG_OUT + rr];
    }
}

kernel void imparo_f32_matmat(
    device const float * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in  [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x * nsg + sgid;
    const uint t = tgid.y;
    if (r >= n_out || t >= n_tok) { return; }
    device const float * row = weights + (w_offset / 4u) + (ulong)r * n_in;
    device const float * xt  = x + (ulong)(src_row + t) * n_in;
    float acc = 0.0f;
    for (uint i = lane; i < n_in; i += 32u) { acc += row[i] * xt[i]; }
    acc = simd_sum(acc);
    if (lane == 0) { y[(ulong)t * n_out + r] = acc; }
}

// AN F32 PROJECTION TO AT MOST 32 OUTPUTS OVER MANY TOKENS: LFM2.5-8B-A1B's router gate,
// 2048 -> 32, on every routed layer of a prefill chunk. imparo_f32_matmat above gives each
// (output, token) pair its own simdgroup, so a token's activation row is read once per
// output and the weight once per token: 0.74 ms for a 2048-token chunk, ~5x the bandwidth
// floor.
//
// Here a simdgroup owns TPS tokens and holds all 32 outputs of each in registers, and the
// threadgroup stages the weight 128 columns at a time (32 x 128 floats, 16 KB) once for
// all its tokens. Each activation is read once.
//
// BIT-IDENTICAL to imparo_f32_matmat: lane l still adds i = l, l+32, l+64, ... in ascending
// order into each output, and the same simd_sum reduces the lanes. Rows past n_out stage as
// zeros and are never written. The host routes here only when n_in is a multiple of 128.
constant constexpr uint F32N_K = 128u;   // weight columns staged per step
constant constexpr uint F32N_R = 32u;    // outputs held per lane

template <uint TPS>
inline void f32_narrow_mm_body(device const float * weights, device const float * x,
                               device float * y, ulong w_offset, uint n_in, uint n_out,
                               uint n_tok, uint src_row, threadgroup float * wt, uint tg,
                               uint tid, uint lane, uint sgid, uint nsg)
{
    const uint t0 = (tg * nsg + sgid) * TPS;
    device const float * w = weights + (w_offset / 4u);
    float acc[TPS][F32N_R];
    for (uint u = 0; u < TPS; ++u) {
        for (uint r = 0; r < F32N_R; ++r) { acc[u][r] = 0.0f; }
    }
    // A token past n_tok reads token 0's row and is never written, so the loop below has no
    // per-token test.
    device const float * xt[TPS];
    for (uint u = 0; u < TPS; ++u) {
        const uint t = t0 + u < n_tok ? t0 + u : 0u;
        xt[u] = x + (ulong)(src_row + t) * n_in;
    }
    const uint nthr = nsg * 32u;
    for (uint k0 = 0; k0 < n_in; k0 += F32N_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < F32N_R * F32N_K / 4u; e += nthr) {
            const uint r = e / (F32N_K / 4u);
            const uint c = (e % (F32N_K / 4u)) * 4u;
            const float4 v = r < n_out
                ? *(device const float4 *)(w + (ulong)r * n_in + k0 + c) : float4(0.0f);
            *(threadgroup float4 *)(wt + r * F32N_K + c) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint q = 0; q < F32N_K / 32u; ++q) {
            const uint j = q * 32u + lane;
            float xv[TPS];
            for (uint u = 0; u < TPS; ++u) { xv[u] = xt[u][k0 + j]; }
            for (uint r = 0; r < F32N_R; ++r) {
                const float wv = wt[r * F32N_K + j];
                for (uint u = 0; u < TPS; ++u) { acc[u][r] += wv * xv[u]; }
            }
        }
    }
    // Output r's sum is the same on every lane; lane r keeps it and writes it, so the
    // store is one coalesced row.
    for (uint u = 0; u < TPS; ++u) {
        const uint t = t0 + u;
        if (t >= n_tok) { break; }
        float mine = 0.0f;
        for (uint r = 0; r < F32N_R; ++r) {
            const float s = simd_sum(acc[u][r]);
            if (lane == r) { mine = s; }
        }
        if (lane < n_out) { y[(ulong)t * n_out + lane] = mine; }
    }
}

#define IMPARO_F32_NARROW_MM(NAME, TPS)                                                      \
kernel void NAME(                                                                           \
    device const float * weights [[buffer(0)]], device const float * x [[buffer(1)]],       \
    device float * y [[buffer(2)]],                                                         \
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],            \
    constant uint & n_out [[buffer(5)]], constant uint & n_tok [[buffer(6)]],               \
    constant uint & src_row [[buffer(7)]],                                                  \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],  \
    uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]],  \
    uint nsg [[simdgroups_per_threadgroup]])                                                \
{                                                                                           \
    threadgroup float wt[F32N_R * F32N_K];                                                  \
    f32_narrow_mm_body<TPS>(weights, x, y, w_offset, n_in, n_out, n_tok, src_row, wt,       \
                            tgid.x, tid, lane, sgid, nsg);                                  \
}
IMPARO_F32_NARROW_MM(imparo_f32_narrow_mm, 2u)

// A NARROW PROJECTION SPLIT ALONG K, not along the rows. `imparo_f32_matmat` gives each
// output row ONE simdgroup, so a 2048 -> 32 projection -- the router's gate on every routed
// layer -- has 32 simdgroups for the whole device and reads 13.1 GB/s against the output
// head's 132. There are no more rows to hand out; the parallelism has to come from the dot
// product itself.
//
// Here a threadgroup owns one output row and its `nsg` simdgroups split K between them.
// Lane strides run across the WHOLE threadgroup (`sgid * 32 + lane`, stepping `nsg * 32`),
// so consecutive lanes still read consecutive floats and the loads stay coalesced.
//
// BIT-AFFECTING, and deliberately so: the row's sum is now nsg partial simd_sums added
// together instead of one. The partials are summed in SIMDGROUP INDEX ORDER by a single
// thread, so the result is the same on every run -- reproducible, but not equal to what
// the one-simdgroup kernel produced.
kernel void imparo_f32_gemv_ksplit(
    device const float * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in  [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    threadgroup float * part [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x, t = tgid.y;
    if (r >= n_out || t >= n_tok) { return; }
    device const float * row = weights + (w_offset / 4u) + (ulong)r * n_in;
    device const float * xt  = x + (ulong)(src_row + t) * n_in;
    float acc = 0.0f;
    for (uint i = sgid * 32u + lane; i < n_in; i += nsg * 32u) { acc += row[i] * xt[i]; }
    acc = simd_sum(acc);
    if (lane == 0u) { part[sgid] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u && lane == 0u) {
        float total = 0.0f;
        for (uint j = 0u; j < nsg; ++j) { total += part[j]; }   // index order, every run
        y[(ulong)t * n_out + r] = total;
    }
}

// ---- Q8_0 WEIGHTS -----------------------------------------------------------------
//
// Ported from the LFM2.5 bring-up branch, where a Q8_0 model was the target quant. Three
// deliberately separate routes, none of which shares launch geometry with Q4_0:
//
//   decode   n_tok == 1: llama.cpp's q8_0 mul_mv decomposition -- every simdgroup in the
//            threadgroup cooperates on a small group of output rows and each lane
//            consumes eight contiguous values.
//   narrow   a tuner-selected token tile reuses one weight byte across 1..8 tokens.
//   prefill  both operands staged as half, accumulated in float with simdgroup matrices,
//            which is what llama.cpp's q8_0 mul_mm does.
//
// A Q8_0 block is an f16 scale then 32 SIGNED bytes: element i is byte 2+i, with no
// nibble pairing and no -8 bias. Reading it with the Q4_0 unpacker returns 16 plausible
// values instead of 32 correct ones, which is the whole reason WeightKind exists.

// Function-constant indices 0..5 are taken (LANES_PER_ROW, SKIP_MMA, NR0, KVT_K, KVT_V,
// KV_PAGED). The Q8 family continues from 6.
constant uint Q8_DECODE_ROWS [[function_constant(6)]];
constant uint Q8_TOKEN_TILE  [[function_constant(7)]];
// Grid order is a compile-time policy: row-x is the compatibility default, token-x is a
// separately compiled pipeline the tuner may select. Only threadgroup coordinates are
// exchanged; tile ownership, K traversal, MMA order and the stores are byte-for-byte the
// same inside each logical output tile.
constant bool Q8_GRID_TOKEN_X_FC [[function_constant(8)]];
constant bool Q8_GRID_TOKEN_X = is_function_constant_defined(Q8_GRID_TOKEN_X_FC)
                              ? Q8_GRID_TOKEN_X_FC : false;
// The byte reconstruction stays a separately compiled pipeline. A typed two-byte scale
// load is selected only after the host has validated the live tensor's absolute binding
// offset; neither the shader nor its default assumes GGUF alignment.
constant bool Q8_TYPED_SCALE_FC [[function_constant(9)]];
constant bool Q8_TYPED_SCALE = is_function_constant_defined(Q8_TYPED_SCALE_FC)
                             ? Q8_TYPED_SCALE_FC : false;
// ATTRIBUTION PROBE for the prefill GEMM, the same three bits the Q4 rt_gemm carries:
//
//   bit 0   do not multiply          -> what the MMA costs
//   bit 1   do not stage             -> what the threadgroup round trip costs
//   bit 2   do not read the weights  -> what the device weight fetch costs
//
// Index 12: 10 is the attention live mask and 11 the epilogue activation.
// A FUNCTION CONSTANT, not a kernel argument: the shipping pipeline is compiled with it
// undefined, so every arm below folds away and the measured build is byte-identical to
// the one without this probe. A runtime uniform would cost a branch in the hot loop and
// could move the answer. The host compiles the probe variants only when asked.
constant uint Q8_SKIP_FC [[function_constant(12)]];
constant uint Q8_SKIP = is_function_constant_defined(Q8_SKIP_FC) ? Q8_SKIP_FC : 0u;

// CLAMP THE EDGE INSTEAD OF PREDICATING IT, which is what llama.cpp's mul_mm does:
//
//   llama   lr0 = (tiitg/NL0) < nr0 ? (tiitg/NL0) : nr0 - 1;   ONCE, outside the loop
//           then every load in the K loop is unconditional
//   here    if (live) { ...read... } else { ...zero... }        every K chunk
//
// A clamped index loads a DUPLICATE row for the out-of-range lanes instead of zeroing
// them. Those lanes' outputs are masked at write-back either way, so the answer cannot
// change -- what changes is that the staging loop has no branch and no zero-fill arm.
// This is the one difference between the two staging loops that had not been tested;
// unlike the two instruction reductions already measured slower, it removes BRANCHES
// rather than arithmetic. ON by default, with the pad skip above OFF: alone it measured flat,
// together they remove every runtime bound from the staging loops.
constant bool Q8_CLAMP_EDGE_FC [[function_constant(13)]];
constant bool Q8_CLAMP_EDGE = is_function_constant_defined(Q8_CLAMP_EDGE_FC)
                            ? Q8_CLAMP_EDGE_FC : true;

constant bool Q8_MMA_FENCE_FC [[function_constant(14)]];
constant bool Q8_MMA_FENCE = is_function_constant_defined(Q8_MMA_FENCE_FC)
                           ? Q8_MMA_FENCE_FC : true;

// A SIMDGROUP WHOSE TOKEN ROWS ARE ALL PAST THE LAST LIVE TOKEN DOES NOTHING.
//
// The grid covers ceil(n_tok / TOKENS) token groups, so the last group holds
// n_tok mod TOKENS live token rows and TOKENS - that many dead ones. Those dead rows
// were staged (from a clamped duplicate row) and multiplied like any other, and only the
// write-back masked them: a 33-row batch on the 64x32 tile paid for 64 token rows of MMA.
//
// THE GRAIN IS ONE SIMDGROUP'S TOKEN ROWS, not one 8x8 tile, and that is a measurement,
// not a simplification. The finer and more obvious form -- a per-tile predicate inside
// the K loop -- was built and measured: it won 3.4-3.8% on a 10- or 16-row batch and LOST
// 2.3% on a full 32-row one and 2.9% at prefill, because the loop is fully unrolled with
// the accumulators in registers and a runtime test between the loads and the multiplies
// breaks the schedule that carries them. A simdgroup's token tiles are contiguous, so
// rounding the live rows up to TOKEN_TILES_PER_SG * 8 makes every simdgroup's set wholly
// live or wholly dead, and the test becomes ONE loop-invariant branch outside the K loop.
// A full tile -- every prefill chunk -- takes the instructions it always did.
//
// The write-back already skips dead rows (`tn` clamps to ntok, and the unmasked fast path
// runs only at ntok == TOKENS), so the output is unchanged bit for bit.
//
// The weight stage is per ROW and does not move: a token group still reads the whole
// weight tile. What this removes is the activation staging and the multiplies of the
// token rows no simdgroup writes back.
//
// OFF by default, with the edge clamp below ON: the tile rule picks the narrowest tile that
// holds the rows, so whole dead shares are rare, and the skip's runtime bound kept the staging
// loop from compiling like the full tile's (the host's g_q8_pad_skip says what it cost).
constant bool Q8_PAD_SKIP_FC [[function_constant(27)]];
constant bool Q8_PAD_SKIP = is_function_constant_defined(Q8_PAD_SKIP_FC)
                          ? Q8_PAD_SKIP_FC : false;

// Q8_0_TM: the weight bytes are in the tile-major order imparo-repack writes
// (docs/q8-tile-major-weights.md): per (8-row tile, 32-wide K block) a 256-byte unit
// [row][k] of int8, units row-tile-major with K blocks adjacent, and every scale after the
// payload as half[unit][8]. A pipeline is compiled for one layout or the other -- the
// constant is a function constant, not a uniform, so the row-major kernels are unchanged
// byte for byte and the TM kernels carry no runtime branch.

constant bool Q8_TM_FC [[function_constant(15)]];
constant bool Q8_TM = is_function_constant_defined(Q8_TM_FC) ? Q8_TM_FC : false;
// One unit = the eight rows' half scales (16 bytes) then their eight 32-byte payload rows:
// 272 bytes, units row-tile-major with K blocks adjacent. MIRRORS imparo_gguf::weights::
// TM_RULES (pinned bit-identical by the gates). The scales moved into the unit on
// 2026-09-04: kept in one array after the payload, the decode GEMV streamed two regions
// per tensor and read 0.7..1.2% slower than row-major.
constant uint Q8_TM_UNIT_ROWS  = 8u;
constant uint Q8_TM_UNIT_BYTES = Q8_TM_UNIT_ROWS * (2u + QK8_0);   // 272
inline ulong q8_tm_unit(uint row, uint block, uint blocks) {
    return ((ulong)(row / Q8_TM_UNIT_ROWS) * blocks + block) * Q8_TM_UNIT_BYTES;
}
// Byte offset of `row`'s 32 values for K block `block`; `blocks` = K blocks per row.
inline ulong q8_tm_payload(uint row, uint block, uint blocks) {
    return q8_tm_unit(row, block, blocks) + Q8_TM_UNIT_ROWS * 2u
         + (ulong)(row % Q8_TM_UNIT_ROWS) * QK8_0;
}
// Byte offset of `row`'s half scale for K block `block` (`n_rows` unused by the layout now).
inline ulong q8_tm_scale(uint row, uint block, uint blocks, uint n_rows) {
    (void)n_rows;
    return q8_tm_unit(row, block, blocks) + (ulong)(row % Q8_TM_UNIT_ROWS) * 2u;
}

// SKIP THE MASK LOOP when a position block is provably live for every query in the tile.
// Defaults TRUE: the test is a handful of uniform integer ops per block against a loop of
// QROWS*8*8*PB/32 iterations per lane, and it is uniform across the simdgroup so there is
// no divergence when it fails. The constant exists so the two can be compared in one
// binary -- a runtime flag would put a branch in the kernel it is measuring, which is the
// mistake a phase probe once made at a cost of 2% of the kernel.
constant bool ATTN_LIVE_MASK_FC [[function_constant(10)]];
constant bool ATTN_LIVE_MASK = is_function_constant_defined(ATTN_LIVE_MASK_FC)
                             ? ATTN_LIVE_MASK_FC : true;

// DECODE. w_offset is 64-bit here and 32-bit on the Q4_0 path: a weight blob past 4 GiB
// wraps a uint offset silently, and this engine's own gemma4 file is 4.22 GB -- 79 MB
// under the wrap. New code gets the wide offset; the Q4_0 path is a separate fix.
kernel void imparo_q8_0_gemv(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    if (n_tok == 0u || n_tok > MV_TOKENS) { return; }
    const uint blocks = n_in / QK8_0;
    device const float * xb = x + (ulong)src_row * n_in;
    const uint last = n_tok - 1u;
    if (Q8_TM) {
        // TILE-MAJOR: a threadgroup owns one 8-row unit tile and every simdgroup step reads
        // ONE WHOLE 256-BYTE UNIT -- lane = (row in tile, k quarter), 8 bytes each, one
        // contiguous span. The row-major mapping below (32 lanes over 8 K blocks of one row)
        // would read eight 32-byte pieces 256 bytes apart here, and the other seven rows of
        // each unit's cache lines from other threadgroups: measured -1.5..-2.9% decode on
        // the first A/B. The host dispatches n_out/8 threadgroups for this layout.
        const uint r0 = tgid.x * Q8_TM_UNIT_ROWS;
        if (r0 >= n_out) { return; }
        // TWO ROWS PER LANE, like the row-major kernel: each activation load feeds two
        // rows' multiplies. One row per lane doubled the x loads per MAC (2 float4 per 8
        // MACs against 2 per 16) and the probe read the mean 2-6% over row-major with the
        // min tied. Lane = (unit half u, row pair rp, k quarter il): 16 lanes cover one
        // unit's 256 payload bytes, the simdgroup covers two ADJACENT units (K blocks ib,
        // ib+1) per step -- 544 contiguous bytes including the two 16-byte scale heads.
        const uint u = lane / 16u, rp = (lane / 4u) % 4u, il = lane % 4u;
        const uint rowa = min(r0 + rp * 2u, n_out - 1u);
        const uint rowb = min(r0 + rp * 2u + 1u, n_out - 1u);
        device const uchar * w = weights + w_offset;
        if (MV_TOKENS > 1u) {
            // DECODE ROWS. The one-row mapping gives each lane two rows and every token, so four
            // lanes load the same activations and each serves two rows: activation reads per
            // tile grow as tokens/2 times the weight bytes, and at 8 rows they measured
            // 73 ms/forward against 23. Here the four row-pair lanes (rp) take TOKENS instead,
            // each over all eight rows of the unit, so one activation load feeds eight rows.
            // What fixes a row's bits is untouched: its blocks still stream by (sgid, u), its
            // 8-value quarter by il, and it reduces over the same partners (xor 1 and 2 over il,
            // xor 16 over u, then the simdgroups in order) -- which lane holds it does not enter.
            //   2 rows: lane group rp = (row half, token)   4 rows x 1 token
            //   4 rows: rp = token                          8 rows x 1 token
            //   8 rows: rp = tokens rp and rp + 4           8 rows x 2 tokens
            const uint RPL = MV_TOKENS == 2u ? 4u : 8u;
            const uint TPL = MV_TOKENS == 8u ? 2u : 1u;
            const uint row0 = MV_TOKENS == 2u ? (rp / 2u) * 4u : 0u;
            const uint tok0 = MV_TOKENS == 2u ? rp % 2u : rp;
            uint rows[8];
            #pragma unroll
            for (uint rr = 0; rr < RPL; ++rr) { rows[rr] = min(r0 + row0 + rr, n_out - 1u); }
            float acc[8][2];
            #pragma unroll
            for (uint rr = 0; rr < RPL; ++rr) {
                #pragma unroll
                for (uint tt = 0; tt < TPL; ++tt) { acc[rr][tt] = 0.0f; }
            }
            for (uint ib = sgid * 2u + u; ib < blocks; ib += nsg * 2u) {
                const uint i = ib * QK8_0 + il * 8u;
                char4 q0[8], q1[8];
                float d[8];
                #pragma unroll
                for (uint rr = 0; rr < RPL; ++rr) {
                    device const uchar * pr = w + q8_tm_payload(rows[rr], ib, blocks) + il * 8u;
                    d[rr]  = float(*(device const half *)(w + q8_tm_scale(rows[rr], ib, blocks, n_out)));
                    q0[rr] = as_type<char4>(*(device const uint *)pr);
                    q1[rr] = as_type<char4>(*(device const uint *)(pr + 4u));
                }
                #pragma unroll
                for (uint tt = 0; tt < TPL; ++tt) {
                    device const float * xt = xb + (ulong)min(tok0 + tt * 4u, last) * n_in;
                    const float4 xa  = *(device const float4 *)(xt + i);
                    const float4 xb4 = *(device const float4 *)(xt + i + 4u);
                    {
                        // The one-row body rounds the block's product before the add; this
                        // body, with more rows in flight, was contracted to one fma (see the
                        // Q4_0 twin).
                    #pragma clang fp contract(off)
                        #pragma unroll
                        for (uint rr = 0; rr < RPL; ++rr) {
                            acc[rr][tt] += (dot(float4(q0[rr]), xa) + dot(float4(q1[rr]), xb4)) * d[rr];
                        }
                    }
                }
            }
            #pragma unroll
            for (uint rr = 0; rr < RPL; ++rr) {
                #pragma unroll
                for (uint tt = 0; tt < TPL; ++tt) {
                    acc[rr][tt] += simd_shuffle_xor(acc[rr][tt], 1u);
                    acc[rr][tt] += simd_shuffle_xor(acc[rr][tt], 2u);
                    acc[rr][tt] += simd_shuffle_xor(acc[rr][tt], 16u);
                }
            }
            if (lane < 16u && il == 0u) {
                #pragma unroll
                for (uint rr = 0; rr < RPL; ++rr) {
                    #pragma unroll
                    for (uint tt = 0; tt < TPL; ++tt) {
                        const uint t = tok0 + tt * 4u;
                        partial[(sgid * MV_TOKENS + t) * Q8_TM_UNIT_ROWS + row0 + rr] = acc[rr][tt];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u && lane < Q8_TM_UNIT_ROWS && r0 + lane < n_out) {
                for (uint t = 0; t < n_tok; ++t) {
                    float total = 0.0f;
                    for (uint s = 0; s < nsg; ++s) {
                        total += partial[(s * MV_TOKENS + t) * Q8_TM_UNIT_ROWS + lane];
                    }
                    device float * slot = y + (ulong)t * n_out + r0 + lane;
                    *slot = epilogue ? (imparo_act_f(*slot) * total) : total;
                }
            }
            return;
        }
        float acca = 0.0f, accb = 0.0f;
        for (uint ib = sgid * 2u + u; ib < blocks; ib += nsg * 2u) {
            const uint i = ib * QK8_0 + il * 8u;
            const float4 xa  = *(device const float4 *)(xb + i);
            const float4 xb4 = *(device const float4 *)(xb + i + 4u);
            device const uchar * pa = w + q8_tm_payload(rowa, ib, blocks) + il * 8u;
            device const uchar * pb = w + q8_tm_payload(rowb, ib, blocks) + il * 8u;
            const float da = float(*(device const half *)(w + q8_tm_scale(rowa, ib, blocks, n_out)));
            const float db = float(*(device const half *)(w + q8_tm_scale(rowb, ib, blocks, n_out)));
            const char4 qa0 = as_type<char4>(*(device const uint *)pa);
            const char4 qa1 = as_type<char4>(*(device const uint *)(pa + 4u));
            const char4 qb0 = as_type<char4>(*(device const uint *)pb);
            const char4 qb1 = as_type<char4>(*(device const uint *)(pb + 4u));
            acca += (dot(float4(qa0), xa) + dot(float4(qa1), xb4)) * da;
            accb += (dot(float4(qb0), xa) + dot(float4(qb1), xb4)) * db;
        }
        // The four k-quarter lanes of a row pair sum first, then the two unit halves,
        // then the simdgroups through threadgroup memory.
        acca += simd_shuffle_xor(acca, 1u);  accb += simd_shuffle_xor(accb, 1u);
        acca += simd_shuffle_xor(acca, 2u);  accb += simd_shuffle_xor(accb, 2u);
        acca += simd_shuffle_xor(acca, 16u); accb += simd_shuffle_xor(accb, 16u);
        if (lane < 16u && il == 0u) {
            partial[sgid * Q8_TM_UNIT_ROWS + rp * 2u]      = acca;
            partial[sgid * Q8_TM_UNIT_ROWS + rp * 2u + 1u] = accb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u && lane < Q8_TM_UNIT_ROWS && r0 + lane < n_out) {
            float total = 0.0f;
            for (uint s = 0; s < nsg; ++s) { total += partial[s * Q8_TM_UNIT_ROWS + lane]; }
            device float * slot = y + r0 + lane;
            *slot = epilogue ? (imparo_act_f(*slot) * total) : total;
        }
        return;
    }
    const uint r0 = tgid.x * Q8_DECODE_ROWS;
    if (r0 >= n_out) { return; }
    const uint ix = lane / 4u;
    const uint il = lane % 4u;
    const uint ib0 = sgid * 8u + ix;

    device const uchar * rows[4];
    #pragma unroll
    for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
        const uint row = min(r0 + rr, n_out - 1u);
        rows[rr] = weights + w_offset + (ulong)row * blocks * Q8_0_BYTES;
    }
    if (MV_TOKENS > 1u) {
        // DECODE ROWS in the row-major layout: every lane already serves all of the
        // threadgroup's rows from one activation load, so the tokens are simply added to it --
        // each row's blocks, quarters and simd_sum are the one-row kernel's.
        float acc[4][MV_MAX];
        for (uint rr = 0; rr < 4u; ++rr) {
            #pragma unroll
            for (uint t = 0; t < MV_TOKENS; ++t) { acc[rr][t] = 0.0f; }
        }
        for (uint ib = ib0; ib < blocks; ib += nsg * 8u) {
            const uint i = ib * QK8_0 + il * 8u;
            float d[4];
            char4 qa[4], qb[4];
            #pragma unroll
            for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
                device const uchar * blk = rows[rr] + (ulong)ib * Q8_0_BYTES;
                d[rr]  = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
                qa[rr] = char4(*(device const packed_char4 *)(blk + 2u + il * 8u));
                qb[rr] = char4(*(device const packed_char4 *)(blk + 6u + il * 8u));
            }
            #pragma unroll
            for (uint t = 0; t < MV_TOKENS; ++t) {
                device const float * xt = xb + (ulong)min(t, last) * n_in;
                const float4 xa = *(device const float4 *)(xt + i);
                const float4 xb4 = *(device const float4 *)(xt + i + 4u);
                #pragma unroll
                for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
                    acc[rr][t] += (dot(float4(qa[rr]), xa) + dot(float4(qb[rr]), xb4)) * d[rr];
                }
            }
        }
        #pragma unroll
        for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
            #pragma unroll
            for (uint t = 0; t < MV_TOKENS; ++t) { acc[rr][t] = simd_sum(acc[rr][t]); }
        }
        if (lane == 0u) {
            #pragma unroll
            for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
                #pragma unroll
                for (uint t = 0; t < MV_TOKENS; ++t) {
                    partial[(sgid * MV_TOKENS + t) * Q8_DECODE_ROWS + rr] = acc[rr][t];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0u && lane < Q8_DECODE_ROWS && r0 + lane < n_out) {
            for (uint t = 0; t < n_tok; ++t) {
                float total = 0.0f;
                for (uint s = 0; s < nsg; ++s) {
                    total += partial[(s * MV_TOKENS + t) * Q8_DECODE_ROWS + lane];
                }
                device float * slot = y + (ulong)t * n_out + r0 + lane;
                *slot = epilogue ? (imparo_act_f(*slot) * total) : total;
            }
        }
        return;
    }
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint ib = ib0; ib < blocks; ib += nsg * 8u) {
        const uint i = ib * QK8_0 + il * 8u;
        const float4 xa = *(device const float4 *)(xb + i);
        const float4 xb4 = *(device const float4 *)(xb + i + 4u);
        #pragma unroll
        for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
            device const uchar * blk = rows[rr] + (ulong)ib * Q8_0_BYTES;
            const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
            const char4 qa = char4(*(device const packed_char4 *)(blk + 2u + il * 8u));
            const char4 qb = char4(*(device const packed_char4 *)(blk + 6u + il * 8u));
            acc[rr] += (dot(float4(qa), xa) + dot(float4(qb), xb4)) * d;
        }
    }
    #pragma unroll
    for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) { acc[rr] = simd_sum(acc[rr]); }
    if (lane == 0u) {
        #pragma unroll
        for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
            partial[sgid * Q8_DECODE_ROWS + rr] = acc[rr];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u && lane < Q8_DECODE_ROWS && r0 + lane < n_out) {
        float total = 0.0f;
        for (uint s = 0; s < nsg; ++s) {
            total += partial[s * Q8_DECODE_ROWS + lane];
        }
        device float * slot = y + r0 + lane;
        *slot = epilogue ? (imparo_act_f(*slot) * total) : total;
    }
}

// CO-BATCHED ROWS ON THE FAST ROUTE, TILE-MAJOR Q8_0: 2..24 independent rows through the
// simdgroup matrix unit, each weight block read once for all of them, on the one-row GEMV's
// grid. A threadgroup owns one 8-row unit tile and its simdgroups split the K blocks, so a
// projection keeps the one-row GEMV's parallelism. The GEMM's 64-row tiles give a 2048-wide
// projection 32 threadgroups, each walking all of K: at 8 rows LFM2.5's down projection
// (K = 10752) took 398 us on the GEMM against 173 us for one row on the GEMV.
//
// Per block (one 272-byte unit: 8 row scales, then 8 rows x 32 values) a simdgroup forms
// P[8 rows x 8 tokens] = W[8 x 32] . X[8 x 32]^T as four 8x8x8 products and adds d_row * P to
// its sums. THE K ORDER inside a block is permuted so every load is one word: fragment f's
// slot j stands for k = 4j + f. Lane (fr, fc) of the 8x8 layout holds W row fr at slots fc and
// fc + 1 -- k in [4fc, 4fc + 8), one 8-byte load -- and X^T slot fr for tokens fc and fc + 1 --
// k in [4fr, 4fr + 4) of each, one float4 per token. A sum does not care in which order its
// terms arrive, so the permutation moves rounding only, which the fast route allows. Weights
// are exact in float; activations are read as float unless a producer left only their half
// mirror (RM_HALF_X). From float, a row lands within about 1e-6 of its rms from the one-row
// GEMV's value (tests/decode_rows.rs).
// Token columns past n_tok read the last live row and are not stored.
constant uint RM_TOKEN_FRAGS_FC [[function_constant(31)]];
constant uint RM_TOKEN_FRAGS = is_function_constant_defined(RM_TOKEN_FRAGS_FC)
                             ? RM_TOKEN_FRAGS_FC : 1u;   // token columns / 8: 1..3
// Three columns at most: every threadgroup reads all its rows' activations, and a fourth column
// (25..32 rows) made those reads cost more than the GEMM's whole step (evidence in
// docs/evidence/cobatch/2026-09-19-per-row-operations.md, section 6).
#define RM_MAX_FRAGS 3u
// THE ACTIVATIONS' FORM: float from the source buffer, or half from its mirror when a producer
// in mirror mode left the floats stale (a GEMM epilogue that writes only the half mirror).
constant bool RM_HALF_X_FC [[function_constant(36)]];
constant bool RM_HALF_X = is_function_constant_defined(RM_HALF_X_FC) && RM_HALF_X_FC;
// THE WEIGHTS' FORM: tile-major Q8_0 (an 8-row unit's scales, then its rows' 32 values), or
// row-major Q4_0 (each row's 18-byte blocks where the file has them: a half scale, then 16
// bytes holding value k in the low nibble of byte k and value 16 + k in the high one). Either
// gives a lane eight values of its row per block, in a K order of its own that the activation
// offset follows (a sum does not care in which order its terms arrive), so everything after
// the fetch -- the products, the scale, the sums -- is one body.
constant bool RM_Q4_FC [[function_constant(37)]];
constant bool RM_Q4 = is_function_constant_defined(RM_Q4_FC) && RM_Q4_FC;
// UNIT TILES PER THREADGROUP, 1, 2 or 4: more tiles cut the threadgroups -- each pays its
// set-up and its cross-simdgroup sum once -- and each activation fetch feeds both tiles' products.
constant uint RM_TILES_FC [[function_constant(38)]];
constant uint RM_TILES = is_function_constant_defined(RM_TILES_FC) ? RM_TILES_FC : 1u;
#define RM_MAX_TILES 4u
// DIAGNOSTIC, 0 in every served pipeline (IMPARO_RM_SKIP): bit 0 drops the matrix products (the
// sums take one fragment element instead), bit 1 the weight loads, bit 2 the activation loads,
// bit 3 every simdgroup matrix (plain multiply-adds on the loaded values), bit 4 the whole block
// loop -- each skipped part's time is the difference. Wrong numbers on purpose.
constant uint RM_SKIP_FC [[function_constant(34)]];
constant uint RM_SKIP = is_function_constant_defined(RM_SKIP_FC) ? RM_SKIP_FC : 0u;

// A lane's two elements of an 8x8 fragment, written or read as one float2 -- MLX steel's form
// (mma.h). Two element writes through a bound reference cost this kernel 61.6 us against 41.0
// on a 2048x2048 projection at 8 rows, with the same bits.
inline void rm_frag_set(thread simdgroup_float8x8 & m, float a, float b) {
    reinterpret_cast<thread float2 &>(m.thread_elements()) = float2(a, b);
}
inline float2 rm_frag_get(thread simdgroup_float8x8 & m) {
    return reinterpret_cast<thread float2 &>(m.thread_elements());
}

kernel void imparo_q8_tm_rows_mma(
    device const uchar * weights [[buffer(0)]],
    device const void  * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    const uint blocks = n_in / QK8_0;
    // The threadgroup's RM_TILES unit tiles: rows 8 * RM_TILES * tgid.x on, 8 per tile.
    const uint g0 = tgid.x * RM_TILES;
    if (g0 * Q8_TM_UNIT_ROWS >= n_out || n_tok == 0u) { return; }
    // The 8x8 layout MLX's steel GEMM stores by (BaseMMAFrag::get_coord): lane holds row fr
    // and the two columns fc, fc + 1.
    const uint qd = lane / 4u;
    const uint fr = (qd & 4u) + ((lane / 2u) % 4u);
    const uint fc = (qd & 2u) * 2u + (lane % 2u) * 2u;
    // Tile-major Q8_0: tile u's unit, block 0. Q4_0: this lane's row of tile u, block 0.
    device const uchar * base0[RM_MAX_TILES];
    #pragma unroll
    for (uint u = 0; u < RM_TILES; ++u) {
        base0[u] = weights + w_offset
                 + (RM_Q4 ? (ulong)((g0 + u) * Q8_TM_UNIT_ROWS + fr) * blocks * Q4_0_BYTES
                          : (ulong)(g0 + u) * blocks * Q8_TM_UNIT_BYTES);
    }
    const uint last = n_tok - 1u;
    // This lane's element offset in each of its token rows: the first k of its slot fr in block 0
    // -- 4fr for Q8_0; for Q4_0, whose order puts the low nibbles in even slots and the high ones
    // in odd slots, 2fr, or 16 + 2(fr - 1) (see the fetch below).
    const uint k0 = RM_Q4 ? ((fr & 1u) != 0u ? 14u + 2u * fr : 2u * fr) : 4u * fr;
    ulong xo[RM_MAX_FRAGS][2];
    #pragma unroll
    for (uint t = 0; t < RM_TOKEN_FRAGS; ++t) {
        #pragma unroll
        for (uint c = 0; c < 2u; ++c) {
            xo[t][c] = (ulong)(src_row + min(t * 8u + fc + c, last)) * n_in + k0;
        }
    }
    float acc[RM_MAX_TILES][RM_MAX_FRAGS][2];
    #pragma unroll
    for (uint u = 0; u < RM_TILES; ++u) {
        #pragma unroll
        for (uint t = 0; t < RM_TOKEN_FRAGS; ++t) { acc[u][t][0] = 0.0f; acc[u][t][1] = 0.0f; }
    }
    for (uint b = sgid; b < ((RM_SKIP & 16u) ? 0u : blocks); b += nsg) {
        // This lane's row fr of each tile at slots fc and fc + 1 of fragment f, f = 0..3. Q8_0:
        // W[fr][4fc + f] and W[fr][4fc + 4 + f]. Q4_0: W[fr][2fc + f] and W[fr][16 + 2fc + f].
        float d[RM_MAX_TILES];
        float4 wlo[RM_MAX_TILES], whi[RM_MAX_TILES];
        #pragma unroll
        for (uint u = 0; u < RM_TILES; ++u) {
            if (RM_SKIP & 2u) {
                d[u] = float(b);
                wlo[u] = float4(as_type<char4>(lane * 0x01010101u + b + u));
                whi[u] = float4(as_type<char4>(fr));
            } else if (RM_Q4) {
                // Bytes 2fc .. 2fc + 3 of the row's block hold k = 2fc .. 2fc + 3 in their low
                // nibbles and 16 + 2fc .. in their high ones: four bytes a lane, and the row's four
                // lanes read its 16 bytes once. The block is 2-byte aligned, so they are read as
                // packed halves.
                device const uchar * blk = base0[u] + (ulong)b * Q4_0_BYTES;
                d[u] = float(*(device const half *)blk);
                const packed_ushort2 pw = *(device const packed_ushort2 *)(blk + 2u + 2u * fc);
                wlo[u] = float4(float(pw[0] & 0xFu), float((pw[0] >> 8u) & 0xFu),
                                float(pw[1] & 0xFu), float((pw[1] >> 8u) & 0xFu)) - 8.0f;
                whi[u] = float4(float((pw[0] >> 4u) & 0xFu), float((pw[0] >> 12u) & 0xFu),
                                float((pw[1] >> 4u) & 0xFu), float((pw[1] >> 12u) & 0xFu)) - 8.0f;
            } else {
                device const uchar * unit = base0[u] + (ulong)b * Q8_TM_UNIT_BYTES;
                d[u] = float(*(device const half *)(unit + fr * 2u));
                const uint2 wq =
                    *(device const uint2 *)(unit + Q8_TM_UNIT_ROWS * 2u + fr * QK8_0 + 4u * fc);
                wlo[u] = float4(as_type<char4>(wq.x));
                whi[u] = float4(as_type<char4>(wq.y));
            }
        }
        // One activation fetch per block, for every tile.
        float4 xa[RM_MAX_FRAGS][2];
        #pragma unroll
        for (uint t = 0; t < RM_TOKEN_FRAGS; ++t) {
            if (RM_SKIP & 4u) {
                xa[t][0] = float4(float(b + t));
                xa[t][1] = float4(float(lane + b));
            } else if (RM_HALF_X) {
                device const half * xh = (device const half *)x;
                xa[t][0] = float4(*(device const half4 *)(xh + xo[t][0] + b * QK8_0));
                xa[t][1] = float4(*(device const half4 *)(xh + xo[t][1] + b * QK8_0));
            } else {
                device const float * xf = (device const float *)x;
                xa[t][0] = *(device const float4 *)(xf + xo[t][0] + b * QK8_0);
                xa[t][1] = *(device const float4 *)(xf + xo[t][1] + b * QK8_0);
            }
        }
        if (RM_SKIP & 8u) {
            #pragma unroll
            for (uint u = 0; u < RM_TILES; ++u) {
                #pragma unroll
                for (uint t = 0; t < RM_TOKEN_FRAGS; ++t) {
                    acc[u][t][0] += d[u] * dot(wlo[u], xa[t][0]);
                    acc[u][t][1] += d[u] * dot(whi[u], xa[t][1]);
                }
            }
            continue;
        }
        simdgroup_float8x8 wm[RM_MAX_TILES][4];
        #pragma unroll
        for (uint u = 0; u < RM_TILES; ++u) {
            #pragma unroll
            for (uint f = 0; f < 4u; ++f) { rm_frag_set(wm[u][f], wlo[u][f], whi[u][f]); }
        }
        #pragma unroll
        for (uint t = 0; t < RM_TOKEN_FRAGS; ++t) {
            simdgroup_float8x8 xm[4];
            #pragma unroll
            for (uint f = 0; f < 4u; ++f) { rm_frag_set(xm[f], xa[t][0][f], xa[t][1][f]); }
            #pragma unroll
            for (uint u = 0; u < RM_TILES; ++u) {
                if (RM_SKIP & 1u) {
                    #pragma unroll
                    for (uint f = 0; f < 4u; ++f) {
                        const float2 a = rm_frag_get(wm[u][f]) * rm_frag_get(xm[f]);
                        acc[u][t][0] += a.x;
                        acc[u][t][1] += a.y;
                    }
                    continue;
                }
                simdgroup_float8x8 p;
                simdgroup_multiply(p, wm[u][0], xm[0]);
                simdgroup_multiply_accumulate(p, wm[u][1], xm[1], p);
                simdgroup_multiply_accumulate(p, wm[u][2], xm[2], p);
                simdgroup_multiply_accumulate(p, wm[u][3], xm[3], p);
                const float2 e = rm_frag_get(p);
                acc[u][t][0] += d[u] * e.x;
                acc[u][t][1] += d[u] * e.y;
            }
        }
    }
    // The simdgroups' sums through threadgroup memory, then added in simdgroup order: 64 floats
    // per (tile, 8-token column).
    const uint per_sg = RM_TILES * RM_TOKEN_FRAGS * 64u;
    #pragma unroll
    for (uint u = 0; u < RM_TILES; ++u) {
        #pragma unroll
        for (uint t = 0; t < RM_TOKEN_FRAGS; ++t) {
            threadgroup float * o =
                partial + sgid * per_sg + (u * RM_TOKEN_FRAGS + t) * 64u + fr * 8u + fc;
            o[0] = acc[u][t][0];
            o[1] = acc[u][t][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = sgid * 32u + lane; i < per_sg; i += nsg * 32u) {
        const uint col = i / 64u;                 // u * RM_TOKEN_FRAGS + t
        const uint u = col / RM_TOKEN_FRAGS;
        const uint row = (i / 8u) % 8u;
        const uint tok = (col % RM_TOKEN_FRAGS) * 8u + i % 8u;
        if (tok >= n_tok) { continue; }
        float total = 0.0f;
        for (uint s = 0; s < nsg; ++s) { total += partial[s * per_sg + i]; }
        device float * slot = y + (ulong)tok * n_out + (g0 + u) * Q8_TM_UNIT_ROWS + row;
        *slot = epilogue ? (imparo_act_f(*slot) * total) : total;
    }
}

// NARROW BATCH. One weight byte is read once and reused across Q8_TOKEN_TILE tokens.
// The offsets are clamped with min(k, last) so a partial tile reads a live row instead
// of past the end; only `tiles` results are written.
kernel void imparo_q8_0_matmat(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x * nsg + sgid;
    const uint t0 = tgid.y * Q8_TOKEN_TILE;
    if (r >= n_out || t0 >= n_tok) { return; }
    const uint tiles = min(Q8_TOKEN_TILE, n_tok - t0);
    const uint last = tiles - 1u;
    const uint blocks = n_in / QK8_0;
    device const uchar * row = weights + w_offset + (Q8_TM ? 0ul : (ulong)r * blocks * Q8_0_BYTES);
    device const float * xb = x + (ulong)(src_row + t0) * n_in;
    const ulong o1 = (ulong)min(1u, last) * n_in;
    const ulong o2 = (ulong)min(2u, last) * n_in;
    const ulong o3 = (ulong)min(3u, last) * n_in;
    const ulong o4 = (ulong)min(4u, last) * n_in;
    const ulong o5 = (ulong)min(5u, last) * n_in;
    const ulong o6 = (ulong)min(6u, last) * n_in;
    const ulong o7 = (ulong)min(7u, last) * n_in;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        float d, wv;
        if (Q8_TM) {
            d  = float(*(device const half *)(row + q8_tm_scale(r, b, blocks, n_out)));
            wv = float(as_type<char>(row[q8_tm_payload(r, b, blocks) + lane])) * d;
        } else {
            device const uchar * blk = row + (ulong)b * Q8_0_BYTES;
            d  = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
            wv = float(as_type<char>(blk[2u + lane])) * d;
        }
        const uint i = b * QK8_0 + lane;
        a0 += wv * xb[i];
        if (Q8_TOKEN_TILE > 1u) { a1 += wv * xb[o1 + i]; }
        if (Q8_TOKEN_TILE > 2u) { a2 += wv * xb[o2 + i]; }
        if (Q8_TOKEN_TILE > 3u) { a3 += wv * xb[o3 + i]; }
        if (Q8_TOKEN_TILE > 4u) { a4 += wv * xb[o4 + i]; }
        if (Q8_TOKEN_TILE > 5u) { a5 += wv * xb[o5 + i]; }
        if (Q8_TOKEN_TILE > 6u) { a6 += wv * xb[o6 + i]; }
        if (Q8_TOKEN_TILE > 7u) { a7 += wv * xb[o7 + i]; }
    }
    a0 = simd_sum(a0);
    if (Q8_TOKEN_TILE > 1u) { a1 = simd_sum(a1); }
    if (Q8_TOKEN_TILE > 2u) { a2 = simd_sum(a2); }
    if (Q8_TOKEN_TILE > 3u) { a3 = simd_sum(a3); }
    if (Q8_TOKEN_TILE > 4u) { a4 = simd_sum(a4); }
    if (Q8_TOKEN_TILE > 5u) { a5 = simd_sum(a5); }
    if (Q8_TOKEN_TILE > 6u) { a6 = simd_sum(a6); }
    if (Q8_TOKEN_TILE > 7u) { a7 = simd_sum(a7); }
    if (lane == 0u) {
        device float * slot = y + (ulong)t0 * n_out + r;
        slot[0] = epilogue ? (imparo_act_f(slot[0]) * a0) : a0;
        if (tiles > 1u) { device float * p = slot + (ulong)n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a1) : a1; }
        if (tiles > 2u) { device float * p = slot + (ulong)2u * n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a2) : a2; }
        if (tiles > 3u) { device float * p = slot + (ulong)3u * n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a3) : a3; }
        if (tiles > 4u) { device float * p = slot + (ulong)4u * n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a4) : a4; }
        if (tiles > 5u) { device float * p = slot + (ulong)5u * n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a5) : a5; }
        if (tiles > 6u) { device float * p = slot + (ulong)6u * n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a6) : a6; }
        if (tiles > 7u) { device float * p = slot + (ulong)7u * n_out;
                          *p = epilogue ? (imparo_act_f(*p) * a7) : a7; }
    }
}

// PREFILL. Both operands staged as half in threadgroup memory, accumulated in float by
// simdgroup matrices. FULL is a separately compiled entry point the host selects only
// for an exact grid with no fused epilogue, so the compiler can delete every edge
// predicate and zero-fill arm from the hot tile; the generic instantiation is the
// fallback. HALF_A reads an already-converted half activation instead of converting f32
// here.
template<uint ROWS, uint TOKENS, uint NSG, uint SGR, uint KCH, bool FULL = false,
         bool HALF_A = false, bool STAGE_A = true, uint APF = 0u, bool GATED = false>
static void st_gemm(
    device const uchar * weights, device const uchar * weights2,
    device const uchar * x_bytes, device float * y,
    uint n_in, uint n_out, uint n_tok, uint src_row, uint epilogue,
    device half * xh2, uint epi_half,
    threadgroup half * shared, uint3 tgid, uint tid, uint sgid)
{
    static_assert(ROWS > 0u && (ROWS % 8u) == 0u,
                  "Q8 GEMM rows must be a non-zero multiple of one MMA tile");
    // GATED (see THE GATED PAIR above rt_gemm): `weights` is the gate tensor, `weights2`
    // the up tensor, `n_out` the OUTPUT width, and the grid walks 2 * n_out virtual rows.
    static_assert(!GATED || (ROWS % 16u) == 0u,
                  "a gated st_gemm tile must hold whole gate/up pairs");
    static_assert(!GATED || !FULL, "the gated write-back is the masked one");
    static_assert(TOKENS > 0u && (TOKENS % 8u) == 0u,
                  "Q8 GEMM tokens must be a non-zero multiple of one MMA tile");
    static_assert(NSG > 0u && SGR > 0u && (NSG % SGR) == 0u,
                  "Q8 GEMM simdgroups must split evenly across rows and tokens");
    static_assert(((ROWS / 8u) % SGR) == 0u,
                  "Q8 GEMM row tiles must split evenly across row simdgroups");
    static_assert(((TOKENS / 8u) % (NSG / SGR)) == 0u,
                  "Q8 GEMM token tiles must split evenly across token simdgroups");
    // K CHUNK DEPTH, in elements, a multiple of the 32-element Q8_0 block.
    //
    // At KCH == 32 -- llama.cpp's NK, which this kernel was ported from -- the loop pays
    // TWO threadgroup barriers for every four 8-deep MMA steps. Measured on LFM2's short
    // prefill, the MMA block runs at 5.6-6.8 TFLOP/s where this device sustains 20.33
    // TFLOPS on the same six-loads-per-eight-multiplies ratio with no barriers in the
    // way, and removing the multiplies takes the whole prefill from 521.6 ms to 194.0.
    // Doubling the chunk halves the barrier count per unit of MMA work; it also doubles
    // the contiguous weight bytes per row per chunk, from 34 to 68 against a 128-byte
    // line. What it costs is threadgroup memory, and therefore resident threadgroups --
    // the two pull opposite ways, so the depth is a shape the host picks and the tuner
    // ranks, not a constant argued for here.
    //
    // The host must not select a depth that does not divide n_in: the chunk loop walks
    // whole KCH steps and would read past the row.
    static_assert(KCH >= QK8_0 && (KCH % QK8_0) == 0u,
                  "Q8 GEMM K chunk must be a non-zero multiple of one Q8_0 block");
    constexpr uint K = KCH;
    constexpr uint BLOCKS_PER_CHUNK = KCH / QK8_0;
    constexpr uint ROW_TILES = ROWS / 8u;
    constexpr uint TOKEN_TILES = TOKENS / 8u;
    constexpr uint K_TILES = K / 8u;
    constexpr uint SGT = NSG / SGR;
    constexpr uint ROW_TILES_PER_SG = ROW_TILES / SGR;
    constexpr uint TOKEN_TILES_PER_SG = TOKEN_TILES / SGT;
    constexpr uint ACC = ROW_TILES_PER_SG * TOKEN_TILES_PER_SG;
    // Every entry point fixes NSG in its template and the host dispatches exactly that
    // many 32-wide simdgroups. Keeping the staging stride compile-time means the hot
    // shapes carry no dynamic loop increment and back-edge. The wrapper checks this
    // launch contract before any barrier.
    constexpr uint THREADS_PER_TG = NSG * 32u;
    const uint row_group = Q8_GRID_TOKEN_X ? tgid.y : tgid.x;
    const uint token_group = Q8_GRID_TOKEN_X ? tgid.x : tgid.y;
    const uint vrows = GATED ? 2u * n_out : n_out;   // rows the grid walks
    const uint r0 = row_group * ROWS, t0 = token_group * TOKENS;
    if (!FULL && (r0 >= vrows || t0 >= n_tok)) { return; }
    const uint nrow = FULL ? ROWS : min(ROWS, vrows - r0);
    const uint ntok = FULL ? TOKENS : min(TOKENS, n_tok - t0);
    const uint row_tile0 = (sgid % SGR) * ROW_TILES_PER_SG;
    const uint token_tile0 = (sgid / SGR) * TOKEN_TILES_PER_SG;
    // Live token rows rounded up to one simdgroup's share, so a simdgroup's tiles are
    // either all live or all dead. Equals TOKENS when the skip is off or the tile is full.
    constexpr uint TOKEN_GRAIN = TOKEN_TILES_PER_SG * 8u;
    const uint atok = (FULL || !Q8_PAD_SKIP)
        ? TOKENS
        : min(TOKENS, ((ntok + TOKEN_GRAIN - 1u) / TOKEN_GRAIN) * TOKEN_GRAIN);
    const bool sg_live = (token_tile0 * 8u) < atok;
    const uint blocks = n_in / QK8_0;
    // STAGE_A=false reads the activation operand straight from device instead of copying
    // it into threadgroup memory first, which is what the Q4 rt_gemm does and what its
    // notes measured 2.3x better THERE. The reason to want it here is RESIDENCY: the
    // activation stage is TOKENS*KCH halves, and dropping it takes the 64x32 shape's
    // allocation from 6144 bytes to 4096. Two instruction reductions in this kernel have
    // already measured slower, and the probe says it overlaps staging with multiplying
    // across resident threadgroups -- so more resident threadgroups is the lever the
    // evidence points at, and fewer instructions is not.
    //
    // Every row-block threadgroup of one dispatch reads the SAME activation rows, so the
    // device read lands in L2 rather than going to DRAM; that is why the copy can be
    // removed rather than merely moved. Requires HALF_A: the operand has to already be
    // half in memory for `simdgroup_load` to take it as one.
    static_assert(STAGE_A || HALF_A,
                  "reading A from device needs the half mirror; f32 has no half fragment");
    threadgroup half * ws = shared;
    threadgroup half * as = shared + K * ROWS;

    simdgroup_float8x8 mc[ACC];
    #pragma unroll
    for (uint i = 0; i < ACC; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // THE A PREFETCH PIPELINE, and the reason it can exist at this chunk depth.
    //
    // When A is read straight from the mirror its address is a function of the GLOBAL k
    // position, not of anything the chunk barrier orders -- so unlike the weight tile,
    // its loads can be issued arbitrarily far ahead and the barrier does not break the
    // pipeline. A 32-deep chunk is only four MMA steps, but the prefetch walks all
    // n_in/8 of them.
    //
    // That matters because the device read is what this route trades for its threadgroup
    // memory: an 8x8 half fragment at stride n_in is eight separate cache lines. Measured
    // WITHOUT prefetch it lost 3.6% at the short leg and 4.1% at depth. The Q4 rt_gemm's
    // own ladder for the same operand runs 4.41 -> 4.71 TFLOPS from depth one to five.
    device const half * xh_a = (device const half *)x_bytes;
    const uint a_row0 = src_row + t0;
    simdgroup_half8x8 apre[STAGE_A ? 1u : APF + 1u][TOKEN_TILES_PER_SG];
    if (!STAGE_A) {
        #pragma unroll
        for (uint d = 0; d < APF; ++d) {
            #pragma unroll
            for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
                const uint tt = (token_tile0 + ti) * 8u;
                const ulong base = (ulong)(a_row0 + min(tt, ntok - 1u)) * n_in
                                 + min(d * 8u, n_in - 8u);
                simdgroup_load(apre[d][ti], xh_a + base, n_in);
            }
        }
    }
    uint a_k = APF * 8u;   // the global k position the next issued load reads

    for (uint c00 = 0; c00 < n_in; c00 += K) {
        const uint bi = c00 / QK8_0;
        if (!(Q8_SKIP & 2u)) {
        // Two threads dequantise one BLOCK: each handles sixteen contiguous signed bytes.
        // A chunk is BLOCKS_PER_CHUNK blocks per row, so the work item carries which block
        // it is as well as which half of it.
        constexpr uint HALVES_PER_ROW = BLOCKS_PER_CHUNK * 2u;
        #pragma unroll
        for (uint e = tid; e < ROWS * HALVES_PER_ROW; e += THREADS_PER_TG) {
            // Row-major: consecutive threads walk one row's blocks. Tile-major: sixteen
            // consecutive threads own one 256-byte unit (8 rows x 2 halves), so their
            // sixteen aligned 16-byte loads are one contiguous span instead of sixteen
            // misaligned 4-byte pieces of eight different 34-byte blocks.
            uint rr0, bb, h;
            if (Q8_TM) {
                const uint unit = e / 16u, within = e % 16u;
                rr0 = (unit / BLOCKS_PER_CHUNK) * Q8_TM_UNIT_ROWS + within / 2u;
                bb  = unit % BLOCKS_PER_CHUNK;
                h   = within % 2u;
            } else {
                rr0 = e / HALVES_PER_ROW;
                const uint hh = e % HALVES_PER_ROW;
                bb = hh / 2u; h = hh % 2u;
            }
            // Clamped: read a duplicate row rather than branch. `rr` still addresses the
            // staging slot, so the tile's shape is unchanged; only WHICH row those
            // out-of-range slots hold differs, and the write-back masks them.
            const uint rr = rr0;
            const uint rr_src = (Q8_CLAMP_EDGE && !FULL) ? min(rr0, nrow - 1u) : rr0;
            const bool live = (FULL || Q8_CLAMP_EDGE || rr0 < nrow) && !(Q8_SKIP & 4u);
            device const uchar * blk = weights;
            float d = 0.0f;
            // TM: the sixteen bytes this thread stages, loaded once, aligned.
            uint4 tm16 = uint4(0u);
            if (live) {
                device const uchar * base = gated_is_up<GATED>(r0 + rr_src) ? weights2 : weights;
                const uint srow = gated_src_row<GATED>(r0 + rr_src);
                if (Q8_TM) {
                    blk = base + q8_tm_payload(srow, bi + bb, blocks) + h * 16u;
                    tm16 = *(device const uint4 *)blk;
                    d = float(*(device const half *)(base + q8_tm_scale(srow, bi + bb, blocks, n_out)));
                } else {
                    blk = base + (ulong)srow * blocks * Q8_0_BYTES + (ulong)(bi + bb) * Q8_0_BYTES;
                    const ushort scale_bits = Q8_TYPED_SCALE
                        ? *(device const ushort *)blk
                        : (ushort)(blk[0] | (blk[1] << 8));
                    d = float(as_type<half>(scale_bits));
                }
            }
            #pragma unroll
            for (uint g4 = 0; g4 < 4u; ++g4) {
                char4 q = char4(0);
                if (live) {
                    q = Q8_TM ? as_type<char4>(tm16[g4])
                              : char4(*(device const packed_char4 *)(blk + 2u + h * 16u + g4 * 4u));
                }
                // Q8_SKIP bit 8 (probe): keep the loads and the stores, drop the
                // int8 -> half conversion -- a bit-cast of the same bytes -- so the
                // conversion's ALU share separates from the tile-store share.
                half4 v;
                if (Q8_SKIP & 8u) {
                    // Four bytes reinterpreted, masked to the low nibble so the halves are
                    // tiny (garbage that cannot overflow into NaN logits), times the scale
                    // in half so `d` stays live and the scale READ is still paid: what this
                    // probe removes is the conversion arithmetic and nothing else.
                    const half2 bits = as_type<half2>(q & char4(0x0f));
                    v = half4(bits, bits) * half(d);
                } else {
                    v = half4(q) * half(d);
                }
                #pragma unroll
                for (uint j = 0; j < 4u; ++j) {
                    const uint k = bb * QK8_0 + h * 16u + g4 * 4u + j;
                    const uint tile = (k / 8u) * ROW_TILES + rr / 8u;
                    ws[tile * 64u + (k & 7u) * 8u + (rr & 7u)] = v[j];
                }
            }
        }
        // llama.cpp's Q8_0 mul_mm stages f32 activations as half before MMA. One work
        // item owns an aligned run of eight K values, so both the device reads and the
        // tile-major threadgroup writes are vector operations rather than eight scalar
        // values from different token rows.
        #pragma unroll
        for (uint e = tid; STAGE_A && e < atok * K_TILES; e += THREADS_PER_TG) {
            const uint tt = e / K_TILES, kt = e % K_TILES;
            // K-major activation tiles keep the two token operands consumed by one
            // simdgroup adjacent for each K step. This is only a threadgroup-memory
            // transpose: every output keeps the same K/MMA accumulation order.
            const uint tile = kt * TOKEN_TILES + tt / 8u;
            threadgroup half4 * dst4 =
                (threadgroup half4 *)(as + tile * 64u + (tt & 7u) * 8u);
            if (FULL || Q8_CLAMP_EDGE || tt < ntok) {
                const uint tt_src =
                    (Q8_CLAMP_EDGE && !FULL) ? min(tt, ntok - 1u) : tt;
                const ulong element = (ulong)(src_row + t0 + tt_src) * n_in
                                    + c00 + kt * 8u;
                if (HALF_A) {
                    device const half4 * src4 =
                        (device const half4 *)(x_bytes + element * sizeof(half));
                    dst4[0] = src4[0];
                    dst4[1] = src4[1];
                } else {
                    device const float4 * src4 =
                        (device const float4 *)(x_bytes + element * sizeof(float));
                    dst4[0] = half4(src4[0]);
                    dst4[1] = half4(src4[1]);
                }
            } else {
                dst4[0] = half4(0.0h);
                dst4[1] = half4(0.0h);
            }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!(Q8_SKIP & 1u) && sg_live) {
        #pragma unroll
        for (uint kk = 0; kk < K; kk += 8u) {
            // A simdgroup owns a rectangular set of output tiles. Load each of its row
            // and token operands once, then form their outer product. A flat tile loop
            // loads two matrices for every output tile -- 16 loads for the 64x32/4-SG
            // shape -- where this is the same per-output K order with four weight and
            // two activation loads.
            simdgroup_half8x8 wm[ROW_TILES_PER_SG];
            simdgroup_half8x8 am[TOKEN_TILES_PER_SG];
            if (Q8_MMA_FENCE) { simdgroup_barrier(mem_flags::mem_none); }
            #pragma unroll
            for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
                const uint tile = (kk / 8u) * ROW_TILES + row_tile0 + ri;
                simdgroup_load(wm[ri], ws + tile * 64u, 8u);
            }
            if (Q8_MMA_FENCE) { simdgroup_barrier(mem_flags::mem_none); }
            #pragma unroll
            for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
                if (STAGE_A) {
                    const uint tile = (kk / 8u) * TOKEN_TILES + token_tile0 + ti;
                    simdgroup_load(am[ti], as + tile * 64u, 8u);
                } else {
                    // Issue the load APF steps ahead FIRST, so its latency overlaps this
                    // step's multiplies rather than preceding them, then take the one
                    // that was issued APF steps ago. `min` clamps the tail; those loads
                    // are consumed by no multiply.
                    const uint tt = (token_tile0 + ti) * 8u;
                    const ulong base = (ulong)(a_row0 + min(tt, ntok - 1u)) * n_in
                                     + min(a_k, n_in - 8u);
                    simdgroup_load(apre[APF][ti], xh_a + base, n_in);
                    am[ti] = apre[0][ti];
                }
            }
            if (!STAGE_A) {
                a_k += 8u;
                #pragma unroll
                for (uint d = 0; d < APF; ++d) {
                    #pragma unroll
                    for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
                        apre[d][ti] = apre[d + 1u][ti];
                    }
                }
            }
            if (Q8_MMA_FENCE) { simdgroup_barrier(mem_flags::mem_none); }
            #pragma unroll
            for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
                #pragma unroll
                for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
                    const uint ai = ti * ROW_TILES_PER_SG + ri;
                    simdgroup_multiply_accumulate(mc[ai], am[ti], wm[ri], mc[ai]);
                }
            }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Full tiles need no masked epilogue: write each cooperative result matrix straight
    // to device memory. An unconditional shared-memory spill would add another barrier
    // and a scalar copy to every Q8 projection. Partial rows/tokens and the fused gated
    // epilogue take the masked path below.
    if (!GATED && (FULL || (epilogue == 0u && nrow == ROWS && ntok == TOKENS))) {
        #pragma unroll
        for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
            #pragma unroll
            for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
                const uint ai = ti * ROW_TILES_PER_SG + ri;
                const uint tt = (token_tile0 + ti) * 8u;
                const uint rr = (row_tile0 + ri) * 8u;
                simdgroup_store(mc[ai], y + (ulong)(t0 + tt) * n_out + r0 + rr, n_out);
            }
        }
        return;
    }

    // The operand stage is dead now, so its bytes become a masked float output tile --
    // IN TOKEN HALVES, and that is what decides how many threadgroups stay resident.
    //
    // Spilling the whole tile needs ROWS * TOKENS floats: 8192 bytes at 64x32, against
    // 6144 for the staged operands. The host allocates the larger of the two, so the
    // SPILL was setting the threadgroup budget and the operands were riding along under
    // it. Measured on this device, residency is what this kernel lives on -- the same
    // 64x32 tile at a 64-deep K chunk allocates 12288, drops from four resident
    // threadgroups to two, and loses 10.4% on LFM2's short prefill, while at a MATCHED
    // allocation the deeper chunk wins 1.1%. Halving the spill puts the allocation back
    // on the operands at 6144, which is five resident instead of four.
    //
    // What it costs: one extra barrier pair. A simdgroup's token tiles are contiguous, so
    // with SGT == 2 each pass is exactly one simdgroup group's share and the other group
    // stores nothing; with SGT == 1 every simdgroup stores half its tiles per pass.
    // ONE PASS below 16 token rows: the spill is TOKENS/WB_PASSES token rows wide and a
    // simdgroup_store writes 8 of them, so a tile of 8 cannot be halved. At 8 tokens the
    // whole spill is ROWS * 8 floats, which is smaller than the staged operands anyway,
    // so the allocation does not grow. `out_bytes` in the bridge computes the same thing.
    constexpr uint WB_PASSES = (TOKENS % 16u) == 0u ? 2u : 1u;
    constexpr uint WB_TOKENS = TOKENS / WB_PASSES;
    threadgroup float * out = (threadgroup float *)shared;
    for (uint p = 0; p < WB_PASSES; ++p) {
        const uint t_lo = p * WB_TOKENS;
        #pragma unroll
        for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
            #pragma unroll
            for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
                const uint ai = ti * ROW_TILES_PER_SG + ri;
                const uint tt = (token_tile0 + ti) * 8u;
                const uint rr = (row_tile0 + ri) * 8u;
                if (tt >= t_lo && tt < t_lo + WB_TOKENS) {
                    simdgroup_store(mc[ai], out + (tt - t_lo) * ROWS + rr, ROWS);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint tn = ntok > t_lo ? min(WB_TOKENS, ntok - t_lo) : 0u;
        if (GATED) {
            // act(G) * U from the two accumulators already in `out` (gate at the pair's
            // even slot, up 8 on); same float bits the two-dispatch form multiplied, so
            // the same product. Output column of tile column c is r0 / 2 + c.
            // In mirror mode ONLY the mirror is written: the down projection reads the
            // mirror and nothing else reads float G at prefill (the rt design's epilogue
            // has skipped the float store all along). Until 2026-09-02 both were written --
            // n_ff x tokens x 4 B of dead stores per FFN, 22 MB per 512-token chunk on LFM2.
            const uint ncol = nrow / 2u;
            for (uint e = tid; e < tn * ncol; e += THREADS_PER_TG) {
                const uint tt = e / ncol, c = e % ncol;
                const uint gs = gated_gate_slot(c);
                const float o = imparo_act_f(out[tt * ROWS + gs]) * out[tt * ROWS + gs + 8u];
                const ulong idx = (ulong)(t0 + t_lo + tt) * n_out + (r0 >> 1) + c;
                if (epi_half != 0u) { xh2[idx] = half(o); }
                else                { y[idx] = o; }
            }
        } else
        for (uint e = tid; e < tn * nrow; e += THREADS_PER_TG) {
            const uint tt = e / nrow, rr = e % nrow;
            const ulong idx = (ulong)(t0 + t_lo + tt) * n_out + r0 + rr;
            device float * slot = y + idx;
            const float v = out[tt * ROWS + rr];
            const float o = epilogue ? (imparo_act_f(*slot) * v) : v;
            *slot = o;
            // THE FUSED EPILOGUE'S HALF MIRROR, which the Q4 rt path has had all along
            // and this one never did. G is 10752 wide on LFM2 and the down projection
            // reads it next, so without this every one of the 30 down projections ran a
            // full conversion pass over 5.16M elements -- 929 MB of traffic per forward
            // that the Q4 family does not pay. Same value, written once more as half,
            // so the mirror cannot disagree with what the next GEMM would have converted.
            if (epi_half != 0u) { xh2[idx] = half(o); }
        }
        // The next pass overwrites the same bytes, so every reader must be done first.
        if (p + 1u < WB_PASSES) { threadgroup_barrier(mem_flags::mem_threadgroup); }
    }
}

// The launch contract is checked in the wrapper, before any barrier: a threadgroup that
// is not exactly NSG*32 wide would desynchronise the staging loops.
#define IMPARO_ST_GEMM(NAME, ROWS, TOKENS, NSG, SGR, KCH, FULL)                          \
kernel void NAME(                                                                         \
    device const uchar * weights [[buffer(0)]], device const float * x [[buffer(1)]],     \
    device float * y [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],              \
    constant uint & n_in [[buffer(4)]], constant uint & n_out [[buffer(5)]],              \
    constant uint & n_tok [[buffer(6)]], constant uint & src_row [[buffer(7)]],           \
    constant uint & epilogue [[buffer(13)]],                                              \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],              \
    threadgroup half * shared [[threadgroup(0)]],                                         \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint3 tcnt3 [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) \
{                                                                                         \
    if (tcnt3.x != NSG * 32u || tcnt3.y != 1u || tcnt3.z != 1u) { return; }                \
    st_gemm<ROWS, TOKENS, NSG, SGR, KCH, FULL, false>(                                       \
        weights + w_offset, weights + w_offset, (device const uchar *)x, y, n_in, n_out, n_tok, src_row,       \
        epilogue, xh2, epi_half, shared, tgid, tid, sgid);                                                \
}


// The shape table. Keep it in step with Q8_SHAPES in the bridge -- the host sizes the
// grid and the threadgroup from that table, and a disagreement is a wrong-size launch,
// not a slow one. Shape 9 (128x32) is retained because the LFM branch measured it exact
// and slower, and a rejected candidate that is still reachable is evidence; the host
// builds only the selected shapes.
// HALF-ACTIVATION ENTRY POINTS. The template has carried HALF_A since it was ported --
// "reads an already-converted half activation instead of converting f32 here" -- and
// nothing instantiated it, so the Q8 family read f32 activations while the Q4 rt family
// read the f16 mirror.
//
// WORTH +0.66% ON LFM2, NOT THE THIRD THE BYTE COUNT PREDICTS. On the FFN up-projection
// (n_in 2048, n_out 10752, 512-token chunk) the activation is REQUESTED 705 MB against
// 374 MB of weights, so halving it should remove about a third of the GEMM's bytes. The
// clean A/B at 17123 tokens measured 768.3 against 763.3 tok/s. Requested bytes are not
// DRAM traffic: the 168 row groups re-read one 4 MB activation tile, and the cache
// already absorbs it, so what the mirror actually removes is the repeated f32 -> f16
// conversion, not memory traffic. Same lesson as the QT-16 attention tiles.
//
// It changes no arithmetic: both paths stage the operand as simdgroup_half8x8 into an
// f32 accumulator, so the mirror only moves WHERE the single rounding happens. Logits
// measured bit-identical with it on and off.
//
// Generic variant only. `_full` deletes the edge predicates and is legal solely on an
// exact grid with no epilogue; keeping the mirror out of that interaction is one less
// thing to be wrong on the first landing, and q8_full_tiles currently tunes to 0 anyway.
#define IMPARO_ST_GEMM_H(NAME, ROWS, TOKENS, NSG, SGR, KCH, FULL)                        \
kernel void NAME(                                                                         \
    device const uchar * weights [[buffer(0)]], device const half * x [[buffer(1)]],      \
    device float * y [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],              \
    constant uint & n_in [[buffer(4)]], constant uint & n_out [[buffer(5)]],              \
    constant uint & n_tok [[buffer(6)]], constant uint & src_row [[buffer(7)]],           \
    constant uint & epilogue [[buffer(13)]],                                              \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],              \
    threadgroup half * shared [[threadgroup(0)]],                                         \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint3 tcnt3 [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) \
{                                                                                         \
    if (tcnt3.x != NSG * 32u || tcnt3.y != 1u || tcnt3.z != 1u) { return; }                \
    st_gemm<ROWS, TOKENS, NSG, SGR, KCH, FULL, true>(                                       \
        weights + w_offset, weights + w_offset, (device const uchar *)x, y, n_in, n_out, n_tok, src_row,       \
        epilogue, xh2, epi_half, shared, tgid, tid, sgid);                                                \
}

IMPARO_ST_GEMM(imparo_st_gemm_0,      32, 16,  4, 2, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_0_full, 32, 16,  4, 2, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_1,      32, 32,  4, 2, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_1_full, 32, 32,  4, 2, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_2,      64, 16,  4, 2, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_2_full, 64, 16,  4, 2, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_3,      64, 32,  4, 2, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_3_full, 64, 32,  4, 2, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_4,      64, 32,  8, 4, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_4_full, 64, 32,  8, 4, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_5,      64, 32, 16, 4, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_5_full, 64, 32, 16, 4, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_6,      64, 64,  8, 4, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_6_full, 64, 64,  8, 4, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_7,      64, 64,  4, 2, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_7_full, 64, 64,  4, 2, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_8,      32,128,  4, 2, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_8_full, 32,128,  4, 2, 32, true)
IMPARO_ST_GEMM(imparo_st_gemm_9,     128, 32,  4, 4, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_9_full,128, 32,  4, 4, 32, true)
// DEEPER K CHUNK, 64 elements = two Q8_0 blocks per row per chunk. Same tiles, same MMA
// order, same output: only the number of chunk barriers changes, and with it the
// threadgroup memory the staged operands need. 10 and 11 are 3 and 7 at twice the depth,
// so each pair is a one-variable comparison.
IMPARO_ST_GEMM(imparo_st_gemm_10,     64, 32,  4, 2, 64, false)
IMPARO_ST_GEMM(imparo_st_gemm_10_full,64, 32,  4, 2, 64, true)
IMPARO_ST_GEMM(imparo_st_gemm_11,     64, 64,  4, 2, 64, false)
IMPARO_ST_GEMM(imparo_st_gemm_11_full,64, 64,  4, 2, 64, true)
// EIGHT TOKEN ROWS, one 8x8 token tile: the narrowest tile the MMA can serve. Every
// simdgroup owns that one tile (SGR == NSG), so the row tiles split four ways.
IMPARO_ST_GEMM(imparo_st_gemm_12,     64,  8,  4, 4, 32, false)
IMPARO_ST_GEMM(imparo_st_gemm_12_full,64,  8,  4, 4, 32, true)

// The half-activation twin of each shape above, GENERATED FROM THOSE LINES: the
// tuples were transcribed by hand once and four of ten had the wrong SGR, which
// compiles a differently shaped kernel under the same index.
// THE GATED PAIR on the staged design: gate at `w_offset`, up at `w_offset2` (buffer
// 10), half activations, masked write-back. See THE GATED PAIR above rt_gemm.
#define IMPARO_ST_GEMM_GH(NAME, ROWS, TOKENS, NSG, SGR, KCH)                              \
kernel void NAME(                                                                         \
    device const uchar * weights [[buffer(0)]], device const half * x [[buffer(1)]],      \
    device float * y [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],              \
    constant uint & n_in [[buffer(4)]], constant uint & n_out [[buffer(5)]],              \
    constant uint & n_tok [[buffer(6)]], constant uint & src_row [[buffer(7)]],           \
    constant ulong & w_offset2 [[buffer(10)]],                                            \
    constant uint & epilogue [[buffer(13)]],                                              \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],              \
    threadgroup half * shared [[threadgroup(0)]],                                         \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint3 tcnt3 [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) \
{                                                                                         \
    if (tcnt3.x != NSG * 32u || tcnt3.y != 1u || tcnt3.z != 1u) { return; }                \
    st_gemm<ROWS, TOKENS, NSG, SGR, KCH, false, true, true, 0u, true>(                     \
        weights + w_offset, weights + w_offset2, (device const uchar *)x, y,              \
        n_in, n_out, n_tok, src_row, epilogue, xh2, epi_half, shared, tgid, tid, sgid);   \
}

IMPARO_ST_GEMM_H(imparo_st_gemm_0_h, 32, 16, 4, 2, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_1_h, 32, 32, 4, 2, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_2_h, 64, 16, 4, 2, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_3_h, 64, 32, 4, 2, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_4_h, 64, 32, 8, 4, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_5_h, 64, 32, 16, 4, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_6_h, 64, 64, 8, 4, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_7_h, 64, 64, 4, 2, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_8_h, 32, 128, 4, 2, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_9_h, 128, 32, 4, 4, 32, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_10_h, 64, 32, 4, 2, 64, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_11_h, 64, 64, 4, 2, 64, false)
IMPARO_ST_GEMM_H(imparo_st_gemm_12_h, 64, 8, 4, 4, 32, false)
IMPARO_ST_GEMM_GH(imparo_st_gemm_0_gh, 32, 16, 4, 2, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_1_gh, 32, 32, 4, 2, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_2_gh, 64, 16, 4, 2, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_3_gh, 64, 32, 4, 2, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_4_gh, 64, 32, 8, 4, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_5_gh, 64, 32, 16, 4, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_6_gh, 64, 64, 8, 4, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_7_gh, 64, 64, 4, 2, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_8_gh, 32, 128, 4, 2, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_9_gh, 128, 32, 4, 4, 32)
IMPARO_ST_GEMM_GH(imparo_st_gemm_10_gh, 64, 32, 4, 2, 64)
IMPARO_ST_GEMM_GH(imparo_st_gemm_11_gh, 64, 64, 4, 2, 64)
IMPARO_ST_GEMM_GH(imparo_st_gemm_12_gh, 64, 8, 4, 4, 32)
// FULL *and* the f16 activation mirror. Until these existed, selecting the
// edge-predicate-free entry point silently dropped the mirror -- the host's mirror
// condition carried `!full` because there was nothing to bind it to -- so the one knob
// that chose between them was really choosing between two things at once, and it
// measured the fast path 2.6% SLOWER at 17123 tokens for that reason.
IMPARO_ST_GEMM_H(imparo_st_gemm_0_full_h,  32, 16, 4, 2, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_1_full_h,  32, 32, 4, 2, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_2_full_h,  64, 16, 4, 2, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_3_full_h,  64, 32, 4, 2, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_4_full_h,  64, 32, 8, 4, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_5_full_h,  64, 32, 16, 4, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_6_full_h,  64, 64, 8, 4, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_7_full_h,  64, 64, 4, 2, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_8_full_h,  32, 128, 4, 2, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_9_full_h, 128, 32, 4, 4, 32, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_10_full_h, 64, 32, 4, 2, 64, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_11_full_h, 64, 64, 4, 2, 64, true)
IMPARO_ST_GEMM_H(imparo_st_gemm_12_full_h, 64, 8, 4, 4, 32, true)

// A READ STRAIGHT FROM THE MIRROR, no activation stage. Same tiles, same K order, same
// answers -- only the operand's route into the MMA changes, and with it the threadgroup
// allocation: 64x32 goes from 6144 bytes to 4096, so more threadgroups stay resident.
// That is the lever the probe points at; two attempts to make this kernel do FEWER
// INSTRUCTIONS both measured slower, and residency is what it overlaps staging against.
#define IMPARO_ST_GEMM_DA(NAME, ROWS, TOKENS, NSG, SGR, KCH, APF)                        \
kernel void NAME(                                                                         \
    device const uchar * weights [[buffer(0)]], device const half * x [[buffer(1)]],      \
    device float * y [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],              \
    constant uint & n_in [[buffer(4)]], constant uint & n_out [[buffer(5)]],              \
    constant uint & n_tok [[buffer(6)]], constant uint & src_row [[buffer(7)]],           \
    constant uint & epilogue [[buffer(13)]],                                              \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],              \
    threadgroup half * shared [[threadgroup(0)]],                                         \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint3 tcnt3 [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) \
{                                                                                         \
    if (tcnt3.x != NSG * 32u || tcnt3.y != 1u || tcnt3.z != 1u) { return; }                \
    st_gemm<ROWS, TOKENS, NSG, SGR, KCH, false, true, false, APF>(                            \
        weights + w_offset, weights + w_offset, (device const uchar *)x, y, n_in, n_out, n_tok, src_row,       \
        epilogue, xh2, epi_half, shared, tgid, tid, sgid);                                                \
}

// The prefetch depth is part of the entry point because it sizes a register array. Depth
// 0 is the variant measured WITHOUT prefetch (-3.6% short, -4.1% deep); the Q4 kernel's
// ladder for the same operand says the useful range is 1..5, so the sweep is over the
// shapes the host can select rather than over a knob.
IMPARO_ST_GEMM_DA(imparo_st_gemm_3_da,  64, 32, 4, 2, 32, 0)
IMPARO_ST_GEMM_DA(imparo_st_gemm_7_da,  64, 64, 4, 2, 32, 0)
IMPARO_ST_GEMM_DA(imparo_st_gemm_3_da2, 64, 32, 4, 2, 32, 2)
IMPARO_ST_GEMM_DA(imparo_st_gemm_7_da2, 64, 64, 4, 2, 32, 2)
IMPARO_ST_GEMM_DA(imparo_st_gemm_3_da5, 64, 32, 4, 2, 32, 5)
IMPARO_ST_GEMM_DA(imparo_st_gemm_7_da5, 64, 64, 4, 2, 32, 5)



// dequantise one Q4_0 row (an embedding lookup) and scale
kernel void imparo_q4_0_row(
    device const uchar * weights [[buffer(0)]],
    device float       * y       [[buffer(1)]],
    constant ulong & w_offset [[buffer(2)]], constant uint & width [[buffer(3)]],
    constant uint & index    [[buffer(4)]], constant float & scale [[buffer(5)]],
    constant uint & dst_off  [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint blocks = width / QK4_0;
    if (gid >= blocks) { return; }
    device const uchar * blk = weights + w_offset + ((ulong)index * blocks + gid) * Q4_0_BYTES;
    const half d = as_type<half>((ushort)(blk[0] | (blk[1] << 8)));
    const uint base = dst_off + gid * QK4_0;
    for (uint i = 0; i < QK4_0 / 2; ++i) {
        const uchar packed = blk[2 + i];
        y[base + i]              = (float(packed & 0x0F) - 8.0f) * float(d) * scale;
        y[base + i + QK4_0 / 2]  = (float(packed >> 4)   - 8.0f) * float(d) * scale;
    }
}

// Dequantise one Q8_0 row (an embedding lookup) and scale. One thread per BLOCK, as in
// the Q4_0 version, so the grid is width/32 either way.
kernel void imparo_q8_0_row(
    device const uchar * weights [[buffer(0)]],
    device float       * y       [[buffer(1)]],
    constant ulong & w_offset [[buffer(2)]], constant uint & width [[buffer(3)]],
    constant uint & index    [[buffer(4)]], constant float & scale [[buffer(5)]],
    constant uint & dst_off  [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint blocks = width / QK8_0;
    if (gid >= blocks) { return; }
    device const uchar * blk = weights + w_offset
                             + ((ulong)index * blocks + gid) * Q8_0_BYTES;
    const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
    const uint base = dst_off + gid * QK8_0;
    for (uint i = 0; i < QK8_0; ++i) {
        y[base + i] = float(as_type<char>(blk[2u + i])) * d * scale;
    }
}

// ONE THREADGROUP per row, reduced through threadgroup memory.
//
// The previous version gave one simdgroup (32 lanes) to a whole 2560-element row, so a
// single simdgroup did 2560 reads, a reduction and 2560 writes while the rest of the GPU
// sat idle: measured at 19.95 us for 10 KB, against 1.79 us for a trivial dispatch of the
// same size. At 210 norms per decode token that was 4.19 ms, the largest non-matmul cost
// in the whole forward.
// ONE ROW PER THREADGROUP: post-norm, residual add, and (dual) the next pre-norm, as one
// dispatch. mid = (add + rms(src) * w1) * out_scale  -> dst;  dual: out = rms(mid) * w2.
// Bit-identical to the three separate dispatches it replaces: the sum of squares runs in
// imparo_rms_norm's order (the host launches the same thread count for the width), every
// product is a rounded float before the add (contraction off), the add is the elementwise
// kernel's `a + b`, and the second norm re-reads the row it just wrote, as the separate
// kernel would. Rows are float4-aligned by contract (width % 4 == 0, base 0).
kernel void imparo_rms_norm_add_row(
    device const uchar * weights [[buffer(0)]],
    device const float * src     [[buffer(1)]],
    device const float * addend  [[buffer(2)]],
    device float       * dst     [[buffer(3)]],
    constant ulong & w1_off [[buffer(4)]], constant uint & width [[buffer(5)]],
    constant float & eps    [[buffer(6)]], constant uint & n_row [[buffer(7)]],
    constant float & out_scale [[buffer(8)]],
    constant uint  & dual   [[buffer(9)]], constant ulong & w2_off [[buffer(10)]],
    device float       * out     [[buffer(11)]],
    // The second norm weight belongs to the NEXT layer, which may live in another weight
    // segment (docs/memory-tiers-and-fit.md): its own buffer, its own local offset.
    device const uchar * weights2 [[buffer(12)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x;
    if (r >= n_row) { return; }
    const uint tid = tid3.x, tcount = tcnt3.x;
    const uint w4 = width / 4u;
    device const float4 * s4 = (device const float4 *)(src + (ulong)r * width);
    device const float4 * a4 = (device const float4 *)(addend + (ulong)r * width);
    device float4       * d4 = (device float4 *)(dst + (ulong)r * width);
    float sq = 0.0f;
    for (uint i = tid; i < w4; i += tcount) { sq += dot(s4[i], s4[i]); }
    sq = simd_sum(sq);
    float inv;
    if (nsg == 1u) {
        inv = rsqrt(simd_broadcast_first(sq) / float(width) + eps);
    } else {
        if (lane == 0) { partial[sgid] = sq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            const float v = (lane < nsg) ? partial[lane] : 0.0f;
            const float total = simd_sum(v);
            if (lane == 0) { partial[0] = rsqrt(total / float(width) + eps); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv = partial[0];
    }
    // THE ROUNDED PRODUCT GOES THROUGH THREADGROUP MEMORY. The separate kernels round the
    // normalised value into the row and the add kernel reads it back; staging it here keeps
    // it a rounded float before the add whatever the compiler would contract (a pragma
    // moved the sum of squares instead, 2026-09-02), and threadgroup memory rather than
    // the row because `dst` may alias `addend` (the scaled per-layer site: X = (X + t) k).
    device const float * w1 = (device const float *)(weights + w1_off);
    device const float * sr = src + (ulong)r * width;
    threadgroup float * stage = partial + ((nsg + 3u) & ~3u);
    threadgroup float4 * st4 = (threadgroup float4 *)stage;
    if (((w1_off / 4u) % 4u) == 0u) {
        device const float4 * w14 = (device const float4 *)w1;
        for (uint i = tid; i < w4; i += tcount) { st4[i] = s4[i] * inv * w14[i]; }
    } else {
        for (uint i = tid; i < width; i += tcount) { stage[i] = sr[i] * inv * w1[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (out_scale == 1.0f) {
        for (uint i = tid; i < w4; i += tcount) { d4[i] = a4[i] + st4[i]; }
    } else {
        // add_scale's `(a + b) * k`, the sum rounded before the scale.
        for (uint i = tid; i < w4; i += tcount) { st4[i] = a4[i] + st4[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < w4; i += tcount) { d4[i] = st4[i] * out_scale; }
    }
    if (!dual) { return; }
    // The second norm reads the row this threadgroup just wrote, as a separate dispatch
    // would; the barrier makes every thread's stores visible to every other's loads.
    threadgroup_barrier(mem_flags::mem_device);
    float sq2 = 0.0f;
    for (uint i = tid; i < w4; i += tcount) { sq2 += dot(d4[i], d4[i]); }
    sq2 = simd_sum(sq2);
    float inv2;
    if (nsg == 1u) {
        inv2 = rsqrt(simd_broadcast_first(sq2) / float(width) + eps);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // partial[] is reused
        if (lane == 0) { partial[sgid] = sq2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            const float v = (lane < nsg) ? partial[lane] : 0.0f;
            const float total = simd_sum(v);
            if (lane == 0) { partial[0] = rsqrt(total / float(width) + eps); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv2 = partial[0];
    }
    device const float * w2 = (device const float *)(weights2 + w2_off);
    device float4 * o4 = (device float4 *)(out + (ulong)r * width);
    if (((w2_off / 4u) % 4u) == 0u) {
        device const float4 * w24 = (device const float4 *)w2;
        for (uint i = tid; i < w4; i += tcount) { o4[i] = d4[i] * inv2 * w24[i]; }
    } else {
        device const float * dr = dst + (ulong)r * width;
        device float * orow = out + (ulong)r * width;
        for (uint i = tid; i < width; i += tcount) { orow[i] = dr[i] * inv2 * w2[i]; }
    }
}

// BRICK: RMS NORM SCALE, in two stages, at a VIRTUAL geometry of norm_t threads (vt = norm_t / 32
// simdgroups): virtual thread (vg, lane) owns the float4 vectors vg*32 + lane, + norm_t, ...;
// each virtual simdgroup reduces its sum of squares with simd_sum; simdgroup 0 sums the vt
// partials with one simd_sum and forms rsqrt(total / width + eps). With norm_t equal to the
// real thread count this IS imparo_rms_norm's reduction; the mega block runs the same virtual
// geometry with fewer real simdgroups (each covers the virtual groups vg = sgid, sgid + nsg, ...),
// so a row's scale has the kernel's bits whichever threadgroup shape computes it.
// XS is the row's address space: device const float4 * or threadgroup const float4 *.
template <typename XS>
inline float rms_sumsq(XS s4, uint w4, uint norm_t, uint vg, uint lane) {
    float sq = 0.0f;
    for (uint i = vg * 32u + lane; i < w4; i += norm_t) { sq += dot(s4[i], s4[i]); }
    return simd_sum(sq);
}
// partial[0..vt) holds the virtual groups' sums (written by their lane 0). Returns the scale to
// every thread. `guard_reuse` adds the barrier that lets the caller overwrite `partial` at once
// (the mega block norms several rows in a row); a kernel that norms one row skips it.
inline float rms_finish(uint vt, uint width, float eps, threadgroup float * partial,
                        uint lane, uint sgid, bool guard_reuse) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float total;
        if (vt == 1u) { total = partial[0]; }
        else { const float v = (lane < vt) ? partial[lane] : 0.0f; total = simd_sum(v); }
        if (lane == 0u) { partial[0] = rsqrt(total / float(width) + eps); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    if (guard_reuse) { threadgroup_barrier(mem_flags::mem_threadgroup); }
    return inv;
}
// Both stages over a whole row at the virtual geometry (the mega block's form).
template <typename XS>
inline float rms_inv(XS s4, uint w4, uint width, float eps, uint norm_t,
                     threadgroup float * partial, uint lane, uint sgid, uint nsg) {
    const uint vt = norm_t / 32u;
    for (uint vg = sgid; vg < vt; vg += nsg) {
        const float sq = rms_sumsq(s4, w4, norm_t, vg, lane);
        if (lane == 0u) { partial[vg] = sq; }
    }
    return rms_finish(vt, width, eps, partial, lane, sgid, true);
}

kernel void imparo_rms_norm(
    device const uchar * weights [[buffer(0)]],
    device float       * x       [[buffer(1)]],
    constant ulong & w_offset [[buffer(2)]], constant uint & width [[buffer(3)]],
    constant float & eps     [[buffer(4)]], constant uint & n_row [[buffer(5)]],
    constant uint & row_stride [[buffer(6)]], constant uint & base_off [[buffer(7)]],
    // has_add: 0 none; 1 POST-add (row = norm(src) + addend, gemma4's order); 2 PRE-add
    // (t = addend + src; addend = t; row = norm(t) -- LFM2's residual order, where the
    // addend is the residual stream and is written back). Non-const for mode 2.
    device float * addend [[buffer(8)]], constant uint & has_add [[buffer(9)]],
    // Separate SOURCE row. Bound to the same buffer as `x` for an in-place norm; bound to
    // a different one where the caller previously had to copy the row first. Two of those
    // copies ran per layer -- 84 dispatches per decoded token to move 10 KB each, in a
    // kernel whose cost is almost all dispatch latency.
    device const float * src [[buffer(10)]],
    // Optional half mirror of the written rows (IMPARO_HALF_A): the prefill GEMM's
    // activation copy, written here so the separate cvt dispatch disappears.
    device half * xh [[buffer(11)]], constant uint & xh_on [[buffer(12)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x;
    if (r >= n_row) { return; }
    const uint tid = tid3.x, tcount = tcnt3.x;
    device float * row = x + base_off + (ulong)r * row_stride;
    device const float * srow = src + base_off + (ulong)r * row_stride;

    // Two passes over the row, re-reading it to scale. Caching it in registers across the
    // passes was measured SLOWER (decode 36.7 -> 35.6): ten floats per thread costs more
    // occupancy than the second read costs latency.
    //
    // Both passes go through float4. This kernel is latency-bound -- 40 KB of traffic and
    // ~0.4 us of work against 5.8 us measured, 294 of them per decoded token -- so what
    // matters is the NUMBER of memory instructions, not the bytes. llama.cpp's rms_norm
    // reduces with `dot(x[i], x[i])` over a vector type for the same reason.
    // PRE-ADD (has_add == 2): the row normalised is t = resid + src, resid being `addend`
    // (the residual stream X, read here and written back below) and src the mixer / FFN
    // output. The separate add kernel computed exactly resid[i] + src[i] in float4 and
    // the norm then read the sum back, so the same operands here give the same bits. The
    // sum is recomputed in the scale pass instead of kept (the register note above).
    const bool pre_add = (has_add == 2u);
    device float * rrow = addend + base_off + (ulong)r * row_stride;
    const bool vec4 = (width % 4u) == 0u && ((base_off + r * row_stride) % 4u) == 0u
                   && (!pre_add || (((ulong)rrow & 15ul) == 0ul));
    // THE ROW STAGED IN THREADGROUP MEMORY, when the host has allowed the space for it.
    // The pre-add order reads BOTH rows twice -- resid and src in the sum pass, again in
    // the scale pass -- which is 32 KB of the ~56 KB this kernel moves. Staging t once
    // makes it 8 KB read plus 8 KB of threadgroup traffic, and the kernel is bandwidth
    // bound on ONE CORE (48 KB in 6.1 us is 7.9 GB/s, one core's share of 136), so the
    // bytes are the cost.
    //
    // The register form of this was measured SLOWER once (decode 36.7 -> 35.6): ten floats
    // a thread costs occupancy. Threadgroup memory does too -- but at decode there is ONE
    // row and so ONE threadgroup a core, and occupancy it cannot use is free. The host
    // only sizes the stage when that holds.
    threadgroup float4 * stage = (threadgroup float4 *)(partial + STAGE_OFF);
    if (STAGED && pre_add && vec4) {
        device const float4 * s4 = (device const float4 *)srow;
        device const float4 * r4 = (device const float4 *)rrow;
        const uint w4s = width / 4u;
        float sqs = 0.0f;
        for (uint i = tid; i < w4s; i += tcount) {
            const float4 t = r4[i] + s4[i];
            stage[i] = t;
            sqs += dot(t, t);
        }
        sqs = simd_sum(sqs);
        float invs;
        if (nsg == 1u) {
            invs = rsqrt(simd_broadcast_first(sqs) / float(width) + eps);
            threadgroup_barrier(mem_flags::mem_threadgroup);   // the stage, not the sum
        } else {
            if (lane == 0) { partial[sgid] = sqs; }
            invs = rms_finish(nsg, width, eps, partial, lane, sgid, false);
        }
        device float4 * row4s = (device float4 *)row;
        device float4 * r4w = (device float4 *)rrow;
        device const float4 * w4p = (device const float4 *)(weights + w_offset);
        device half4 * xh4 = (device half4 *)(xh + base_off + (ulong)r * row_stride);
        const bool wt = w_offset != IMPARO_NO_WEIGHT;
        for (uint i = tid; i < w4s; i += tcount) {
            const float4 t = stage[i];
            r4w[i] = t;                                   // the residual stream
            const float4 v = wt ? (t * invs * w4p[i]) : (t * invs);
            row4s[i] = v;
            if (xh_on != 0u) { xh4[i] = half4(v); }
        }
        return;
    }
    float sq = 0.0f;
    if (vec4) {
        device const float4 * s4 = (device const float4 *)srow;
        device const float4 * r4 = (device const float4 *)rrow;
        const uint w4 = width / 4u;
        if (pre_add) {
            for (uint i = tid; i < w4; i += tcount) { const float4 t = r4[i] + s4[i]; sq += dot(t, t); }
            sq = simd_sum(sq);
        } else {
            sq = rms_sumsq(s4, w4, tcount, sgid, lane);   // the brick: this simdgroup's virtual group is itself
        }
    } else if (pre_add) {
        for (uint i = tid; i < width; i += tcount) { const float t = rrow[i] + srow[i]; sq += t * t; }
        sq = simd_sum(sq);
    } else {
        for (uint i = tid; i < width; i += tcount) { sq += srow[i] * srow[i]; }
        sq = simd_sum(sq);
    }

    float inv;
    if (nsg == 1u) {
        // One simdgroup needs NO threadgroup barrier: simd_sum has already reduced across
        // every thread, and the result is broadcast in a register. At decode this kernel is
        // pure latency -- 40 KB of traffic, ~0.4 us of work, 5.8 us measured -- and the two
        // barriers are most of the difference. 294 of these run per token.
        inv = rsqrt(simd_broadcast_first(sq) / float(width) + eps);
    } else {
        // Second stage by simd_sum, not a serial loop on thread 0 (rms_finish): the thread
        // count is derived from the row width, so a 2560-wide row runs 640 threads = 20
        // simdgroups, and summing 20 partials one at a time on one thread put a 20-iteration
        // serial loop on the critical path with 639 threads idle behind it.
        if (lane == 0) { partial[sgid] = sq; }
        inv = rms_finish(nsg, width, eps, partial, lane, sgid, false);
    }

    // The residual add is folded into the scale pass.
    //
    // Every one of these is followed by `add(row, other)` -- three times per layer, 126
    // dispatches per decoded token -- and the norm has already loaded the row. A separate
    // add costs another dispatch, another read of the row and another write of it, in a
    // kernel whose cost is almost entirely dispatch latency.
    device const float * add_row = addend + base_off + (ulong)r * row_stride;
    const uint w4 = width / 4u;
    device float4 * row4 = (device float4 *)row;
    device const float4 * srow4 = (device const float4 *)srow;
    if (pre_add) {
        // The sum goes back to the residual stream, the normalised row to `x`. Each
        // thread rewrites only the elements it read in the first pass (same ownership).
        device float4 * r4w = (device float4 *)rrow;
        if (w_offset == IMPARO_NO_WEIGHT) {
            if (vec4) {
                for (uint i = tid; i < w4; i += tcount) {
                    const float4 t = r4w[i] + srow4[i]; r4w[i] = t; row4[i] = t * inv;
                }
            } else {
                for (uint i = tid; i < width; i += tcount) {
                    const float t = rrow[i] + srow[i]; rrow[i] = t; row[i] = t * inv;
                }
            }
        } else {
            device const float * w = (device const float *)(weights + w_offset);
            if (vec4 && ((w_offset / 4u) % 4u) == 0u) {
                device const float4 * w4p = (device const float4 *)w;
                for (uint i = tid; i < w4; i += tcount) {
                    const float4 t = r4w[i] + srow4[i]; r4w[i] = t; row4[i] = t * inv * w4p[i];
                }
            } else {
                for (uint i = tid; i < width; i += tcount) {
                    const float t = rrow[i] + srow[i]; rrow[i] = t; row[i] = t * inv * w[i];
                }
            }
        }
    } else if (has_add) {
        device const float4 * add4 = (device const float4 *)add_row;
        const bool add_vec4 = vec4 && (((ulong)add_row & 15ul) == 0ul);
        if (w_offset == IMPARO_NO_WEIGHT) {
            if (add_vec4) {
                for (uint i = tid; i < w4; i += tcount) { row4[i] = srow4[i] * inv + add4[i]; }
            } else {
                for (uint i = tid; i < width; i += tcount) {
                    row[i] = srow[i] * inv + add_row[i];
                }
            }
        } else {
            device const float * w = (device const float *)(weights + w_offset);
            if (add_vec4 && ((w_offset / 4u) % 4u) == 0u) {
                device const float4 * w4p = (device const float4 *)w;
                for (uint i = tid; i < w4; i += tcount) {
                    row4[i] = srow4[i] * inv * w4p[i] + add4[i];
                }
            } else {
                for (uint i = tid; i < width; i += tcount) {
                    row[i] = srow[i] * inv * w[i] + add_row[i];
                }
            }
        }
    } else if (w_offset == IMPARO_NO_WEIGHT) {
        if (vec4) {
            for (uint i = tid; i < w4; i += tcount) { row4[i] = srow4[i] * inv; }
        } else {
            for (uint i = tid; i < width; i += tcount) { row[i] = srow[i] * inv; }
        }
    } else {
        device const float * w = (device const float *)(weights + w_offset);
        // The norm weight is a plain tensor at a 4-aligned offset when the row is.
        if (vec4 && ((w_offset / 4u) % 4u) == 0u) {
            device const float4 * w4p = (device const float4 *)w;
            // REJECTED, kept as the record: xh_on == 2 skips the float row store (every
            // prefill consumer reads the mirror, so the row looked unread). The mirror
            // bytes are identical, but removing the store WAKES the task #5 visibility
            // wobble at n=128 -- up to 6 distinct outcomes in 8 runs, ~3e-3 spread, on a
            // single power-of-two batch, which EXTENDS #5 beyond small remainders. The
            // host sends 1 (write both), never 2.
            if (xh_on == 2u) {
                device half4 * x4 = (device half4 *)(xh + base_off + (ulong)r * row_stride);
                for (uint i = tid; i < w4; i += tcount) {
                    x4[i] = half4(srow4[i] * inv * w4p[i]);
                }
                return;
            }
            for (uint i = tid; i < w4; i += tcount) { row4[i] = srow4[i] * inv * w4p[i]; }
        } else {
            for (uint i = tid; i < width; i += tcount) { row[i] = srow[i] * inv * w[i]; }
        }
    }

    // Half mirror: convert exactly the values written above -- same rounding, same slots
    // as the cvt pass. Each thread re-reads only elements it wrote itself (same strided
    // ownership as every loop above), so no barrier is needed.
    if (xh_on != 0u) {
        device half * xrow = xh + base_off + (ulong)r * row_stride;
        if (vec4) {
            device half4 * x4 = (device half4 *)xrow;
            for (uint i = tid; i < w4; i += tcount) { x4[i] = half4(row4[i]); }
        } else {
            for (uint i = tid; i < width; i += tcount) { xrow[i] = half(row[i]); }
        }
    }
}

// BRICK: NEOX ROPE, one pair. Rotates (row[i], row[i + half_rot]) at position pos by the
// inverse frequency base^(-2i / n_rot) / freqs[i] (freqs: the model's rope factor table, 1 when
// absent). imparo_rope_neox's expression; imparo_head_norm_rope and the mega block's head phase
// call it, so a rotated pair has the same bits on every route.
inline void rope_neox_pair(device float * row, uint i, uint half_rot, uint n_rot, float base,
                           constant float * freqs, uint n_freqs, uint pos) {
    const float ff = (n_freqs > 0u && i < n_freqs) ? freqs[i] : 1.0f;
    const float inv = pow(base, -2.0f * float(i) / float(n_rot)) / ff;
    const float theta = float(pos) * inv;
    const float c = cos(theta), s = sin(theta);
    const float x0 = row[i], x1 = row[i + half_rot];
    row[i]            = x0 * c - x1 * s;
    row[i + half_rot] = x0 * s + x1 * c;
}

// HEAD NORM + ROPE in one dispatch (gemma4's Q and K post-projection: per-head rms_norm,
// then NEOX rope). One threadgroup per (token, head) row exactly as imparo_rms_norm runs it
// (same thread count, same two-stage simd_sum reduction, same scale expression), then a
// device barrier and imparo_rope_neox's rotation on the written row. Two dispatches and a
// barrier become one; the bits are the two-dispatch form's (review #116, P10).
kernel void imparo_head_norm_rope(
    device const uchar * weights [[buffer(0)]],
    device float       * x       [[buffer(1)]],
    constant ulong & w_offset [[buffer(2)]], constant uint & width [[buffer(3)]],
    constant float & eps     [[buffer(4)]], constant uint & n_row [[buffer(5)]],
    constant uint & n_rot    [[buffer(6)]], constant float & base [[buffer(7)]],
    constant uint & n_heads  [[buffer(8)]], constant uint & start_pos [[buffer(9)]],
    constant float * freqs   [[buffer(10)]], constant uint & n_freqs [[buffer(11)]],
    device const uint * layout [[buffer(12), function_constant(ROW_LAYOUT)]],
    constant CobRow * rows [[buffer(13), function_constant(COB_ROWS)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x;
    if (r >= n_row) { return; }
    const uint tid = tid3.x, tcount = tcnt3.x;
    device float * row = x + (ulong)r * width;                 // base_off 0, stride == width
    // ---- imparo_rms_norm, src == x ----
    const bool vec4 = (width % 4u) == 0u && ((r * width) % 4u) == 0u;
    float sq = 0.0f;
    if (vec4) {
        sq = rms_sumsq((device const float4 *)row, width / 4u, tcount, sgid, lane);   // the brick
    } else {
        for (uint i = tid; i < width; i += tcount) { sq += row[i] * row[i]; }
        sq = simd_sum(sq);
    }
    float inv;
    if (nsg == 1u) {
        inv = rsqrt(simd_broadcast_first(sq) / float(width) + eps);
    } else {
        if (lane == 0) { partial[sgid] = sq; }
        inv = rms_finish(nsg, width, eps, partial, lane, sgid, false);
    }
    const uint w4 = width / 4u;
    device float4 * row4 = (device float4 *)row;
    if (w_offset == IMPARO_NO_WEIGHT) {
        if (vec4) {
            for (uint i = tid; i < w4; i += tcount) { row4[i] = row4[i] * inv; }
        } else {
            for (uint i = tid; i < width; i += tcount) { row[i] = row[i] * inv; }
        }
    } else {
        device const float * w = (device const float *)(weights + w_offset);
        if (vec4 && ((w_offset / 4u) % 4u) == 0u) {
            device const float4 * w4p = (device const float4 *)w;
            for (uint i = tid; i < w4; i += tcount) { row4[i] = row4[i] * inv * w4p[i]; }
        } else {
            for (uint i = tid; i < width; i += tcount) { row[i] = row[i] * inv * w[i]; }
        }
    }
    // ---- imparo_rope_neox on the written row: a pair may span two threads' writes ----
    threadgroup_barrier(mem_flags::mem_device);
    const uint half_rot = n_rot / 2u;
    const uint t = r / n_heads;
    uint pos = start_pos + t;
    if (ROW_LAYOUT) { pos = layout[t * ROW_LAYOUT_WORDS]; }   // a tree row ropes at its depth
    if (COB_ROWS) { pos = rows[t].pos; }                        // a co-batched row at its own position
    for (uint i = tid; i < half_rot; i += tcount) {
        rope_neox_pair(row, i, half_rot, n_rot, base, freqs, n_freqs, pos);
    }
}

kernel void imparo_rope_neox(
    device float * x [[buffer(0)]],
    constant uint & n_rot [[buffer(1)]], constant float & base [[buffer(2)]],
    constant uint & head_dim [[buffer(3)]], constant uint & n_heads [[buffer(4)]],
    constant uint & start_pos [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant float * freqs [[buffer(7)]], constant uint & n_freqs [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint i = gid.x;              // pair index within the head
    const uint h = gid.y;              // head
    const uint t = gid.z;              // token
    const uint half_rot = n_rot / 2u;
    if (i >= half_rot || h >= n_heads || t >= n_tok) { return; }
    device float * head = x + ((ulong)t * n_heads + h) * head_dim;
    // `rope_freqs.weight` divides the inverse frequency, and this model carries it on the
    // FULL-attention layers only. The kernel had no buffer for it, so those layers -- 5, 11,
    // 17, 23 here -- got unscaled rope while the CPU scaled it. Position 0 is unaffected,
    // rope being the identity there, which is why token 0 always agreed and later tokens did
    // not.
    rope_neox_pair(head, i, half_rot, n_rot, base, freqs, n_freqs, start_pos + t);
}

// copy this batch's K/V rows into the cache at their positions
// `ring_mask` > 0 maps a position to slot (pos & ring_mask). A sliding-window layer never
// attends beyond `window` positions back, so it only needs `window` slots -- allocating
// full context for those layers wasted 4x on 20 of this model's 24 KV-owning layers.
//
// A MASK, not a modulus. The first version used `pos % ring`; the compiler cannot know
// `ring` is a power of two at compile time, so it emitted an integer division inside the
// attention position loop and decode fell from 33 to 15.9 tok/s. Window sizes are powers
// of two, and a layer whose window is not gets full-context allocation instead.
// The KV cache is stored as HALF. It is the largest single allocation the engine owns
// (~109 MiB at ctx 2048 in f32), and attention re-reads all of it every decode step, so the
// precision choice sets both footprint and decode bandwidth. f16 KV is what the reference
// engines default to.
// KV pool paging (docs/unified-kv-pool.md). FULL-ATTENTION layers address their
// cache through a 64-position block table: physical slot = pt[pos/64]*64 + pos%64.
// Windowed layers keep their ring (bounded per-conversation state is not pooled).
// One helper, one rule; the table is identity until the pool assigns real blocks,
// so stage one of the rollout is byte-identical by construction.
// KV_PAGED = false compiles the identity-placement variant: full-attention
// addressing collapses to gp and the page-table load disappears. Pipelines
// built WITHOUT the constant default to true (the paged form), so every
// existing build keeps its behavior; the host picks the identity variant per
// dispatch when the layer's table is the identity mapping.
// Paged attention's page, in KV cells. Injected as a preprocessor macro by the host
// (imparo_metal.mm: KV_PAGE_CELLS) so the two cannot hold different numbers -- a
// compile-time constant, so `/` and `%` still fold to a shift and a mask here.
#ifndef KV_PAGE_CELLS
#define KV_PAGE_CELLS 64u
#endif

constant bool KV_PAGED_FC [[function_constant(5)]];
constant bool KV_PAGED = is_function_constant_defined(KV_PAGED_FC) ? KV_PAGED_FC : true;

inline uint kv_slot(uint gp, uint ring_mask, device const uint * pt) {
    if (ring_mask > 0u) { return gp & ring_mask; }
    return KV_PAGED ? (pt[gp / KV_PAGE_CELLS] * KV_PAGE_CELLS + gp % KV_PAGE_CELLS)
                    : gp;
}
// An n-run starting at gp0 reads contiguous cache bytes unless it straddles the
// ring wrap (windowed) or a non-adjacent page boundary (paged full layers). With
// an identity table adjacent pages ARE adjacent, so stage one keeps every fast
// path it had.
// WHOLE-RANGE wrap test: does [gp0, gp0+n) straddle the ring seam? Ring semantics
// only. On paged full layers this must stay FALSE: their 8-runs are 8-aligned and a
// page (KV_PAGE_CELLS) cannot be straddled by an aligned 8-run, so nothing is ever skipped
// to the tail -- and routing whole tiles to the device-spill tail makes threadgroups
// race on its scratch (the collision its indexing comment warns about; found the
// hard way by the pair-swap placement test).
inline bool kv_range_wraps(uint gp0, uint n, uint ring_mask) {
    return ring_mask > 0u && (gp0 & ring_mask) + n > ring_mask + 1u;
}
inline bool kv_run_breaks(uint gp0, uint n, uint ring_mask, device const uint * pt) {
    if (ring_mask > 0u) { return (gp0 & ring_mask) + n > ring_mask + 1u; }
    if (!KV_PAGED) { return false; } // identity: full-layer runs are contiguous
    if (gp0 % KV_PAGE_CELLS + n <= KV_PAGE_CELLS) { return false; }
    return pt[gp0 / KV_PAGE_CELLS] + 1u
         != pt[(gp0 + n - 1u) / KV_PAGE_CELLS];
}

kernel void imparo_kv_store(
    device const float * src [[buffer(0)]],
    device half        * cache [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & start_pos [[buffer(3)]],
    constant uint & n_tok [[buffer(4)]], constant uint & ring_mask [[buffer(5)]],
    device const uint * pt [[buffer(6)]],
    constant CobRow * rows [[buffer(7), function_constant(COB_ROWS)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Four values per thread. `width` is n_kv_heads * head_dim, a multiple of four for
    // every layer in this model, and both buffers start page aligned. The scalar tail
    // covers anything that does not divide.
    const uint w4 = width / 4u;
    const uint i = gid.x, t = gid.y;
    if (t >= n_tok) { return; }
    uint pos = start_pos + t;
    device const uint * table = pt;
    if (COB_ROWS) { pos = rows[t].pos; table = rows[t].pt; }
    const uint slot = kv_slot(pos, ring_mask, table);
    if (i < w4) {
        device const float4 * s4 = (device const float4 *)(src + (ulong)t * width);
        device half4 * c4 = (device half4 *)(cache + (ulong)slot * width);
        c4[i] = half4(s4[i]);
    }
    // remainder, when width is not a multiple of four
    const uint tail = w4 * 4u + i;
    if (tail < width && i < 4u) {
        cache[(ulong)slot * width + tail] = half(src[(ulong)t * width + tail]);
    }
}

// one threadgroup per (token, head). gemma4 attention scale is 1.0.
// Matrix-unit ceiling. Operands stay in registers, so nothing but the multiply is timed.
//
// The GEMM sits at 3.93 TFLOPS and the assumed ceiling was ~6.45, derived from ALU counts
// rather than measured. If the matrix units top out near 4 then that kernel is already at
// the hardware limit and the fork must be doing something arithmetically different; if they
// reach 6+ then the GEMM is leaving throughput on the table. Guessing which has cost
// several experiments.
// Quantizing KV store: one thread per 32-value block of one token's row. The quantizer
// is llama.cpp's `quantize_q4_0` verbatim (signed max / -8 scale, x/d + 8.5 rounding,
// low nibbles = values 0..15, high = 16..31), so a cache our engine writes holds the
// same bytes the fork's would for the same inputs.
kernel void imparo_kv_store_q4(
    device const float * src [[buffer(0)]],
    device uchar * dst       [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & start_pos [[buffer(3)]],
    constant uint & n_tok [[buffer(4)]], constant uint & ring [[buffer(5)]],
    device const uint * pt [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
#pragma METAL fp math_mode(safe)
    const uint b = gid.x, t = gid.y;
    const uint blocks = width / 32u;
    if (b >= blocks || t >= n_tok) { return; }
    device const float * x = src + (ulong)t * width + b * 32u;
    const uint gp = start_pos + t;
    const uint ps = kv_slot(gp, ring, pt);
    device uchar * blk = dst + ((ulong)ps * blocks + b) * 18u;

    float amax = 0.0f, vmax = 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        const float v = x[j];
        if (amax < fabs(v)) { amax = fabs(v); vmax = v; }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const half dh = half(d);
    blk[0] = as_type<ushort>(dh) & 0xFFu;
    blk[1] = (as_type<ushort>(dh) >> 8) & 0xFFu;
    for (uint j = 0; j < 16u; ++j) {
        const float x0 = x[j] * id;
        const float x1 = x[16u + j] * id;
        const uchar xi0 = (uchar)min(15, (int)(char)(x0 + 8.5f));
        const uchar xi1 = (uchar)min(15, (int)(char)(x1 + 8.5f));
        blk[2u + j] = xi0 | (xi1 << 4);
    }
}

// Q8_0 (34-byte block: half scale + 32 int8), llama.cpp's quantize_q8_0.
kernel void imparo_kv_store_q8(
    device const float * src [[buffer(0)]],
    device uchar * dst       [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & start_pos [[buffer(3)]],
    constant uint & n_tok [[buffer(4)]], constant uint & ring [[buffer(5)]],
    device const uint * pt [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
#pragma METAL fp math_mode(safe)
    const uint b = gid.x, t = gid.y;
    const uint blocks = width / 32u;
    if (b >= blocks || t >= n_tok) { return; }
    device const float * x = src + (ulong)t * width + b * 32u;
    const uint gp = start_pos + t;
    const uint ps = kv_slot(gp, ring, pt);
    device uchar * blk = dst + ((ulong)ps * blocks + b) * 34u;

    float amax = 0.0f;
    for (uint j = 0; j < 32u; ++j) { amax = max(amax, fabs(x[j])); }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const half dh = half(d);
    blk[0] = as_type<ushort>(dh) & 0xFFu;
    blk[1] = (as_type<ushort>(dh) >> 8) & 0xFFu;
    for (uint j = 0; j < 32u; ++j) {
        blk[2u + j] = (uchar)(char)rint(x[j] * id);
    }
}

// Blockwise Hadamard rotation, in place, over consecutive 64-value blocks.
//
// llama.cpp's own quantized-KV defense (llama-kv-cache.cpp attn_rot_k/_v): rotate Q, K
// and V by an orthonormal Hadamard before the cache quantization, and rotate the
// attention output back. Scores are invariant -- (Hq)-dot-(Hk) = q-dot-k -- and the
// rotation spreads each 32-value block's energy evenly, which is what a shared 4-bit
// scale needs. Sylvester order: H[i][j] = (-1)^popcount(i AND j) / sqrt(64), and
// 1/sqrt(64) = 0.125 exactly, so the scale costs no rounding. The sum runs j = 0..63
// in a fixed order: deterministic.
kernel void imparo_hadamard64(
    device float * x [[buffer(0)]],
    constant uint & n [[buffer(1)]],
    constant uint & nrot [[buffer(2)]],
    constant float & scale [[buffer(3)]],
    threadgroup float * blk [[threadgroup(0)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    const uint b0 = gid * nrot;
    if (b0 + nrot > n) { return; }
    for (uint i = tid; i < nrot; i += tcount) { blk[i] = x[b0 + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Fast Walsh-Hadamard butterfly in natural (Sylvester) order -- the same matrix as
    // the popcount form, n log n adds instead of n^2 MACs (the matrix form measured
    // -7% prefill / -3 tok/s decode at q4). Fixed pair order per stage: deterministic.
    for (uint stride = 1u; stride < nrot; stride <<= 1u) {
        for (uint p = tid; p < nrot / 2u; p += tcount) {
            const uint i = ((p & ~(stride - 1u)) << 1u) | (p & (stride - 1u));
            const float a = blk[i], b = blk[i | stride];
            blk[i] = a + b;
            blk[i | stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < nrot; i += tcount) { x[b0 + i] = blk[i] * scale; }
}

// Diagnostic roundtrip store (IMPARO_KVQ_RT): quantize with EXACTLY the sibling store
// kernels' arithmetic, then write the DEQUANTIZED halves into an f16-layout cache --
// the same values the real quantized path's scratch holds, with the quantized-format
// read side taken out of the loop entirely. Position bounds [rt_lo, rt_hi) scope the
// roundtrip to part of the context; rows outside store plain f16.
kernel void imparo_kv_store_rt(
    device const float * src [[buffer(0)]],
    device half * dst        [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & start_pos [[buffer(3)]],
    constant uint & n_tok [[buffer(4)]], constant uint & ring [[buffer(5)]],
    constant uint & rt_type [[buffer(6)]],
    constant uint & rt_lo [[buffer(7)]], constant uint & rt_hi [[buffer(8)]],
    device const uint * pt [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
#pragma METAL fp math_mode(safe)
    const uint b = gid.x, t = gid.y;
    const uint blocks = width / 32u;
    if (b >= blocks || t >= n_tok) { return; }
    device const float * x = src + (ulong)t * width + b * 32u;
    const uint gp = start_pos + t;
    const uint ps = kv_slot(gp, ring, pt);
    device half * out = dst + (ulong)ps * width + b * 32u;
    if (gp < rt_lo || gp >= rt_hi) {
        for (uint j = 0; j < 32u; ++j) { out[j] = half(x[j]); }
        return;
    }
    if (rt_type == 2u) {
        float amax = 0.0f, vmax = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            const float v = x[j];
            if (amax < fabs(v)) { amax = fabs(v); vmax = v; }
        }
        const float d = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const float dd = float(half(d));     // what the dequant kernel reads back
        for (uint j = 0; j < 32u; ++j) {
            const int qi = min(15, (int)(char)(x[j] * id + 8.5f));
            out[j] = half((float(qi) - 8.0f) * dd);
        }
    } else {
        float amax = 0.0f;
        for (uint j = 0; j < 32u; ++j) { amax = max(amax, fabs(x[j])); }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const float dd = float(half(d));
        for (uint j = 0; j < 32u; ++j) {
            const int qi = (int)(char)rint(x[j] * id);
            out[j] = half(float(qi) * dd);
        }
    }
}

// Dequantise the slots the attention is ABOUT TO READ, into the half scratch it
// reads them from.
//
// `slots` counts LOGICAL positions and gid.y walks them; the physical row comes from
// the same `kv_slot` the attention applies. It used to walk PHYSICAL rows 0..slots
// and the attention looked them up through the page table, which agrees only while
// the table is the identity. Under the KV pool it is not: a conversation gets the
// lowest FREE blocks, so a neighbour of four blocks moves every row this scratch has
// to hold, and the attention read rows nothing had written. That is why a quantized
// cache made an answer depend on what else was resident (docs/kv-pool-review.md).
kernel void imparo_kv_dq(
    device const uchar * src [[buffer(0)]],
    device half * dst        [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & slots [[buffer(3)]],
    constant uint & ktype [[buffer(4)]],
    constant uint & ring_mask [[buffer(5)]],
    device const uint * pt [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint b = gid.x, gp = gid.y;
    const uint blocks = width / 32u;
    if (b >= blocks || gp >= slots) { return; }
    const uint sslot = kv_slot(gp, ring_mask, pt);
    device half * out = dst + (ulong)sslot * width + b * 32u;
    if (ktype == 2u) {
        device const uchar * blk = src + ((ulong)sslot * blocks + b) * 18u;
        const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
        for (uint g = 0; g < 4u; ++g) {
            const uchar4 q = uchar4(*(device const packed_uchar4 *)(blk + 2u + g * 4u));
            const float4 lo = (float4(q & uchar4(0x0F)) - 8.0f) * d;
            const float4 hi = (float4(q >> 4)           - 8.0f) * d;
            for (uint j = 0; j < 4u; ++j) {
                out[g * 4u + j]        = half(lo[j]);
                out[16u + g * 4u + j]  = half(hi[j]);
            }
        }
    } else {   // q8_0
        device const uchar * blk = src + ((ulong)sslot * blocks + b) * 34u;
        const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
        for (uint j = 0; j < 32u; ++j) {
            out[j] = half(float(char(blk[2u + j])) * d);
        }
    }
}


// THE SCORE PHASE'S OWN CEILING. Not a generic matrix-op peak -- the exact operand mix
// the prefill attention score loop uses, so the number it returns is a target that kernel
// could actually reach.
//
// Why this exists: the score phase measures 2.68 TFLOPS and was compared against 7.30
// from a probe with a DIFFERENT access pattern, giving "37% of ceiling" -- a gap that
// drove a session of work. A ceiling measured on the wrong access pattern belongs to a
// different kernel, and the honest comparison is against a probe shaped like the caller.
// Shaped like the caller it reads ~4.1 TFLOPS, so the score phase is at ~65%, and it
// reads the same whether K is cached or streamed: the operand mix is the bound.
//
// The caller's per-d-step shape, from attention_prefill_qcomb_body: QROWS query fragments
// read from THREADGROUP memory, PB key fragments read from DEVICE memory, and QROWS*PB
// multiply-accumulates. At the shipping shape that is 2 + 2 loads for 4 multiplies.
kernel void imparo_mma_scoremix(
    device const half * kbuf [[buffer(0)]], device float * out [[buffer(1)]],
    constant uint & iters [[buffer(2)]], constant uint & kstride [[buffer(3)]],
    constant uint & kspan [[buffer(4)]],
    threadgroup float * sq [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    // Staged Q, exactly as the kernel stages it: written once, then re-read every step.
    for (uint e = tid; e < 16u * 64u; e += tcount) { sq[e] = 0.01f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[2][2];
    #pragma unroll
    for (uint g = 0; g < 2; ++g) {
        #pragma unroll
        for (uint b = 0; b < 2; ++b) { acc[g][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
    }
    // Walk K the way the score loop walks it: eight dim steps across a head slice, then
    // on to the next position tile. The address MUST advance with the iteration -- a probe
    // that re-reads one tile stays in cache and reports a ceiling for a kernel whose K
    // fits in cache, which at 16k it does not.
    // kspan is a MASK over position tiles, not a byte count, and it must be a mask: a
    // modulo here is a 64-bit integer divide per iteration, and it cost more than the
    // memory traffic the probe exists to measure -- the cached case came out 3.5x SLOWER
    // than the streaming one, which is the shape of a probe measuring itself.
    const uint tile_mask = kspan;
    for (uint i = 0; i < iters; ++i) {
        const uint d = (i & 7u) * 8u;             // 8 dim steps = a 64-wide slice
        const uint tile = (i >> 3) & tile_mask;   // next 2 position blocks per slice
        const ulong koff = (ulong)tile * 16u * kstride + d;
        simdgroup_float8x8 q[2];
        simdgroup_half8x8  k[2];
        #pragma unroll
        for (uint g = 0; g < 2; ++g) { simdgroup_load(q[g], sq + g * 8u * 64u + d, 64u); }
        #pragma unroll
        for (uint b = 0; b < 2; ++b) {
            simdgroup_load(k[b], kbuf + koff + (ulong)b * 8u * kstride, kstride, 0, false);
        }
        #pragma unroll
        for (uint b = 0; b < 2; ++b) {
            #pragma unroll
            for (uint g = 0; g < 2; ++g) {
                simdgroup_multiply_accumulate(acc[g][b], k[b], q[g], acc[g][b]);
            }
        }
    }
    threadgroup float sink[64];
    #pragma unroll
    for (uint g = 0; g < 2; ++g) {
        #pragma unroll
        for (uint b = 0; b < 2; ++b) { simdgroup_store(acc[g][b], sink, 8); }
    }
    if (tid == 0) { out[tgid] = sink[0]; }
}

// SPILL CLIFF. The register budget is the one device limit nothing exposes: Metal will
// not say how many registers a thread has, and the compiler will not say when it gave up
// and spilled. So measure it -- hold NACC accumulators live across a loop and time the
// multiplies. Below the cliff the rate is flat, because more accumulators is more
// independent work. At the cliff the compiler starts spilling them to device memory and
// the rate collapses.
//
// This replaces RT_REG_BUDGET, which was 2048 read off the shapes that already worked --
// a literal wearing a formula's clothes, and known wrong: one shape satisfies it and
// spills anyway, because a wide tile costs operand registers an accumulator count does
// not see. Measuring the cliff at the real thread count answers what the arithmetic
// cannot.
template <uint NACC>
static void spill_cliff_body(device float * out, uint iters, uint tgid, uint tid,
                             threadgroup float * sink)
{
    simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
    simdgroup_float8x8 b = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
    simdgroup_float8x8 c[NACC];
    #pragma unroll
    for (uint k = 0; k < NACC; ++k) { c[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
    for (uint i = 0; i < iters; ++i) {
        #pragma unroll
        for (uint k = 0; k < NACC; ++k) {
            simdgroup_multiply_accumulate(c[k], a, b, c[k]);
        }
    }
    // Every accumulator must be consumed or the compiler drops the ones it can prove
    // dead -- which would measure a smaller NACC than the one asked for.
    #pragma unroll
    for (uint k = 0; k < NACC; ++k) { simdgroup_store(c[k], sink, 8); }
    if (tid == 0) { out[tgid] = sink[0]; }
}

#define IMPARO_SPILL_KERNEL(NAME, NACC)                                                    \
kernel void NAME(device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]],     \
                 uint tgid [[threadgroup_position_in_grid]],                                \
                 uint tid [[thread_position_in_threadgroup]])                               \
{                                                                                           \
    threadgroup float sink[64];                                                             \
    spill_cliff_body<NACC>(out, iters, tgid, tid, sink);                                    \
}

IMPARO_SPILL_KERNEL(imparo_spill_4,   4)
IMPARO_SPILL_KERNEL(imparo_spill_8,   8)
IMPARO_SPILL_KERNEL(imparo_spill_12, 12)
IMPARO_SPILL_KERNEL(imparo_spill_16, 16)
IMPARO_SPILL_KERNEL(imparo_spill_24, 24)
IMPARO_SPILL_KERNEL(imparo_spill_32, 32)
IMPARO_SPILL_KERNEL(imparo_spill_48, 48)
IMPARO_SPILL_KERNEL(imparo_spill_64, 64)

kernel void imparo_mma_peak(
    device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    simdgroup_float8x8 a = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
    simdgroup_float8x8 b = make_filled_simdgroup_matrix<float, 8, 8>(1.0f);
    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(c4, a, b, c4);
        simdgroup_multiply_accumulate(c5, a, b, c5);
        simdgroup_multiply_accumulate(c6, a, b, c6);
        simdgroup_multiply_accumulate(c7, a, b, c7);
    }
    // Consume the accumulators so none of it is optimised away.
    threadgroup float sink[64];
    simdgroup_store(c0, sink, 8);
    simdgroup_store(c7, sink, 8);
    if (tid == 0) { out[tgid] = sink[0]; }
}

// Same ceiling test, but the operands are RELOADED from threadgroup memory each step, in
// the exact 6-loads-per-8-multiplies ratio the GEMM uses. The difference between this and
// imparo_mma_peak is what operand movement costs, which decides whether the lever is a
// higher multiply-to-load ratio or something else entirely.
kernel void imparo_mma_loaded(
    device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]],
    threadgroup float * sh [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    for (uint i = tid; i < 64 * 8; i += 256u) { sh[i] = 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 c[8];
    #pragma unroll
    for (uint i = 0; i < 8; ++i) { c[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
    for (uint it = 0; it < iters; ++it) {
        simdgroup_float8x8 a[2], b[4];
        #pragma unroll
        for (uint i = 0; i < 2; ++i) { simdgroup_load(a[i], sh + i * 64, 8); }
        #pragma unroll
        for (uint j = 0; j < 4; ++j) { simdgroup_load(b[j], sh + (2 + j) * 64, 8); }
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(c[i * 4 + j], a[i], b[j], c[i * 4 + j]);
            }
        }
    }
    threadgroup float sink[64];
    simdgroup_store(c[0], sink, 8);
    simdgroup_store(c[7], sink, 8);
    if (tid == 0) { out[tgid] = sink[0]; }
}

// Why the decode GEMV sustains less than a plain streaming read. Same rows, same lanes,
// same launch geometry -- only the addressing and the arithmetic change:
//   mode 0: the real kernel's addressing (2-byte scale + four uchar4 at an 18-byte block
//           stride) with the dot products removed. Gap to the real kernel = arithmetic.
//   mode 1: the same bytes read as contiguous uint4, which is only possible if scales and
//           payloads live in separate arrays. Gap to mode 0 = what the 18-byte stride costs.
// Q4_0_BYTES is 18, so consecutive lanes never start on consecutive 16-byte addresses.
kernel void imparo_gemv_probe(
    device const uchar * weights [[buffer(0)]],
    device float * y [[buffer(1)]],
    constant uint & n_in  [[buffer(2)]],
    constant uint & n_out [[buffer(3)]],
    constant uint & lanes [[buffer(4)]],
    constant uint & mode  [[buffer(5)]],
    constant uint & split [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    const uint rows_per_simd = 32u / lanes;
    const uint sub  = lane % lanes;
    const uint slot = lane / lanes;
    const uint r = (tgid.x * nsg + sgid) * rows_per_simd + slot;
    if (r >= n_out) { return; }
    const uint all_blocks = n_in / QK4_0;
    // Split the dot product over `split` threadgroups, each taking a slice of K. This is
    // the only way to add threads to a GEMV whose output is short: n_out * lanes threads
    // is all the row decomposition can give.
    const uint per = (all_blocks + split - 1u) / split;
    const uint b0 = tgid.y * per;
    const uint b1 = min(b0 + per, all_blocks);
    const uint blocks = all_blocks;
    uint acc = 0u;
    if (mode == 0u) {
        device const uchar * row = weights + (ulong)r * blocks * Q4_0_BYTES;
        for (uint b = b0 + sub; b < b1; b += lanes) {
            device const uchar * blk = row + b * Q4_0_BYTES;
            acc += (uint)blk[0] + (uint)blk[1];
            #pragma unroll
            for (uint g = 0; g < 4; ++g) {
                const uchar4 pk = uchar4(*(device const packed_uchar4 *)(blk + 2u + g * 4u));
                acc += (uint)pk.x + (uint)pk.y + (uint)pk.z + (uint)pk.w;
            }
        }
    } else {
        // 16 payload bytes per block, 16-byte aligned rows: the layout a repack would give.
        device const uint4 * rv = (device const uint4 *)(weights + (ulong)r * blocks * 16ull);
        for (uint i = b0 + sub; i < b1; i += lanes) {
            const uint4 v = rv[i];
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == 0xFFFFFFFFu) { y[r] = (float)acc; }
}

// Pure streaming read: no math, no dequantise, no matrix ops -- just uint4 loads off
// device memory, summed so the compiler cannot delete them. This is the number the decode
// GEMV has to be judged against. Calling 106 GB/s "the ceiling" was an assertion; this
// measures where the ceiling actually is.
kernel void imparo_bw_read(
    device uint * out [[buffer(0)]],
    device const uint4 * src [[buffer(1)]],
    constant uint & n4 [[buffer(2)]],
    constant uint & reps [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint gsz [[threads_per_grid]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint4 acc = uint4(0u);
    // EVERY REP READS A DIFFERENT ADDRESS SET. A rep that re-read the thread's own set
    // measured the cache, not the bus: once a threadgroup's set fit the core's cache (a
    // grid of 2.4M threads over 256 MB is 7 loads per thread) the probe reported 380 to
    // 560 GB/s on a 153.6 GB/s bus, and the "153 at the fattest grid" it reported before
    // was the same reuse at 1.1x. Rotating the set by a rep-sized stride keeps the
    // whole buffer between one address and its re-read, which no cache holds.
    const uint off = n4 / (reps > 0u ? reps : 1u);
    for (uint r = 0u; r < reps; ++r) {
        const uint shift = r * off;
        for (uint i = gid; i < n4; i += gsz) {
            uint j = i + shift;
            if (j >= n4) { j -= n4; }
            acc += src[j];
        }
    }
    const uint sum = acc.x + acc.y + acc.z + acc.w;
    // Never taken for real data, but the compiler cannot prove it, so the loads stay.
    if (sum == 0xFFFFFFFFu) { out[tgid] = sum; }
}

// Same 6-loads-per-8-multiplies loop, but the A operands come from DEVICE memory at a
// configurable row stride -- exactly how the GEMM reads activations. The B operands stay in
// threadgroup memory. The gap against imparo_mma_loaded is what device-sourced operands
// cost, which threadgroup-memory size and register pressure have both been ruled out for.
kernel void imparo_mma_device_a(
    device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]],
    device const float * src [[buffer(2)]], constant uint & stride [[buffer(3)]],
    threadgroup float * sh [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]])
{
    for (uint i = tid; i < 64 * 8; i += 256u) { sh[i] = 1.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 c[8];
    #pragma unroll
    for (uint i = 0; i < 8; ++i) { c[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
    for (uint it = 0; it < iters; ++it) {
        simdgroup_float8x8 a[2], b[4];
        // Walk a large region so the reads STREAM instead of sitting in L1. The earlier
        // version stepped through 512 floats, which stayed cached and made every row stride
        // look identical -- it could not distinguish a tiled layout from a strided one.
        const ulong off = ((ulong)it * 4096u) % (1ul << 22);
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            simdgroup_load(a[i], src + off + (ulong)i * 8u * stride, stride);
        }
        #pragma unroll
        for (uint j = 0; j < 4; ++j) { simdgroup_load(b[j], sh + j * 64, 8); }
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(c[i * 4 + j], a[i], b[j], c[i * 4 + j]);
            }
        }
    }
    threadgroup float sink[64];
    simdgroup_store(c[0], sink, 8);
    simdgroup_store(c[7], sink, 8);
    if (tid == 0) { out[tgid] = sink[0]; }
}

// Query-tiled prefill attention with online softmax.
//
// The per-token kernel gives one threadgroup to each (head, token), so every query token
// re-reads the WHOLE K and V. Over a 440-token prefill that is ~16 GB of KV traffic, which
// at this machine's 150 GB/s is ~110 ms -- and measured attention was 122 ms of a 1052 ms
// prefill, against 11 ms for llama.cpp's flash-attention path.
//
// Here one threadgroup owns QT query tokens, so a K row loaded once serves QT queries and
// the traffic falls by QT. Positions are walked in tiles with a running max and sum per
// query, so nothing scales with context length: threadgroup memory holds QT x PT scores and
// the QT x head_dim output, never QT x n_positions.
// WHAT SETS EACH ATTENTION CONSTANT, and therefore how it is decided.
//
// Two mechanisms, and mixing them up is how a value ends up hardcoded that should not be.
//
//   DERIVED from the model shape, by arithmetic, every run. No measurement can improve an
//   answer that follows from the numbers, and no stored file should carry it.
//
//     dblocks   = head_dim / 8
//     kv slots  = window for a windowed layer, else the positions the conversation reaches
//     kv bytes  = n_kv * head_dim * slots * 2
//     slices    = attn_min_tgs / (heads * tokens), capped by positions available
//     tpd       = threadgroup size / (head_dim/4), threads sharing a dim in the V pass
//
//   BENCHED from the PAIR (model shape, device). Neither alone decides it, which is why the
//   stored file is keyed by host fingerprint AND model -- the same device wants different
//   answers for a different model, and the same model wants different answers on a different
//   Mac. These belong in the search space and nowhere else.
//
//     swept today   lanes, sgs, attn_threads, attn_threads_prefill, rt_shape, batch,
//                   flush_layers, flush_layers_prefill, nr0, prefill_kernel,
//                   prefill_min_tok, attn_min_tgs
//     NOT YET SWEPT, and each is a real gap:
//       PB, DBB   register-blocking depth in the two matrix phases. The ceiling is the
//                 register file, a device property; how much is needed depends on QT and
//                 head_dim, model properties. 4 here, 8 spills and measures 434 tok/s
//                 against 496. Making it sweepable needs one pipeline per depth, the way
//                 rt_shape already has one per shape.
//       PT        positions per tile. Bounded by threadgroup memory against QT x head_dim.
//                 64 measures 481 against 496 at 128 here.
//
// `tpd` is the cautionary one. It was written as one thread per dim, which is correct on any
// model whose head_dim/4 reaches the threadgroup size and leaves three quarters of the
// threads idle on this one. A derived value written as a constant is wrong on every shape
// but the one it was written for.
//
// Positions per tile is now a TEMPLATE parameter, not one global for every kernel.
//
// The P x V accumulator is loaded from and stored back to threadgroup memory once per
// POSITION BLOCK, so the traffic is (n / PT) round-trips of QT x hd floats. Doubling PT
// halves it. What stops PT being 256 everywhere is the 32 KB threadgroup budget:
// QT*PT + QT*head_dim floats, which at QT 16 and head_dim 256 is already 32.9 KB.
// QT 8 with head_dim 512 needs 24.7 KB at PT 256 and has the room.
//
// MEASURED AND REVERTED. PT 256 on the QT 8 kernel halves those round-trips and buys
// nothing: 11719 and 11741 ms against 11759 at PT 128, with attention 1653 ms against
// 1685 -- 0.3%, inside the noise, for 4 KB more threadgroup memory and logits that move
// ~1e-3 from the reassociated summation. So the accumulator traffic is NOT why P x V
// costs 1.9x the score phase for the same multiply count, and a register-resident
// accumulator would not buy that back either. Whatever the asymmetry is, it is not this.
//
// The removed experiment bodies also took this global; inside
// `attention_prefill_qtile_body` the template parameter of the same name shadows it.
constant uint PT = 128u;    // positions per tile, for the kernels that are not templated on it

// QT is a template parameter, not a constant.
//
// It sets how many queries share each K and V read: with QT = 8 a 440-token prefill gives
// 55 threadgroups per head all re-reading the same KV. Doubling it halves that traffic. The
// ceiling is threadgroup memory, which holds QT x head_dim for the running output -- 16 KB
// at QT 8 and head_dim 512, so 16 does not fit there. It does at head_dim 256, which is 35
// of this model's 42 layers, so the host picks per layer.
// PT IS A RUNTIME ARGUMENT here for the same reason as in the qcomb body: it appears only
// as a stride, an offset and a loop bound, and no array is declared with it. It matters
// MORE here -- qtile carries a dynamic head dim (HD_C = 0), so it is the kernel a model
// with no specialisation actually runs on, and freezing its tile froze the fallback path
// for every such model.
template<uint QT, uint HD_C, uint BLK>
static void attention_prefill_qtile_body(
    device const float * q, device const half * kc, device const half * vc,
    device float * out, uint PT, uint head_dim, uint n_heads, uint n_kv, uint kv_width,
    uint start_pos, uint window, uint ring_mask, device const uint * pt,
    uint n_tok, uint stage,
    threadgroup float * shared, uint3 tgid, uint tid, uint tcount,
    uint lane, uint sgid, uint nsg)
{
    // head_dim as a COMPILE-TIME constant where the host can supply one.
    //
    // It arrives as `constant uint`, so `for (d = 0; d < head_dim; d += 8)` has a runtime
    // bound and cannot unroll: 32 iterations each paying a compare, a branch and address
    // arithmetic around a single matrix multiply and two loads. The prefill GEMM's
    // equivalent loop is over compile-time tiles, and it retires matrix instructions about
    // six times faster than this kernel does.
    const uint hd = HD_C ? HD_C : head_dim;
    const uint h = tgid.x, q0 = tgid.y * QT;
    const uint kvh = h / (n_heads / n_kv);

    threadgroup float * sc = shared;                       // QT x PT scores
    threadgroup float * acc = shared + QT * PT;            // QT x hd output
    threadgroup float * rmax = acc + QT * hd;        // QT running maxima
    threadgroup float * rsum = rmax + QT;                  // QT running sums
    threadgroup float * red  = rsum + QT;                  // per-simdgroup reduction

    for (uint e = tid; e < QT * hd; e += tcount) { acc[e] = 0.0f; }
    if (tid < QT) { rmax[tid] = -INFINITY; rsum[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The tile spans QT queries whose causal limits differ; scan the union and mask per
    // query. The upper bound comes from the LAST query, the lower bound from the FIRST.
    const uint last = min(q0 + QT, n_tok) - 1u;
    const uint pos_last = start_pos + last;
    // A WINDOWED layer's lower bound RISES with position, so the widest bound in the
    // tile belongs to its FIRST query, not its last. Taking it from `pos_last` dropped
    // every position in [first_lo, last_lo) for the earlier queries, and the per-query
    // mask below can only remove positions from the scan, never add them back.
    const uint pos_first = start_pos + q0;
    const uint lo = (window > 0u && pos_first + 1u > window) ? (pos_first + 1u - window) : 0u;
    const uint n = pos_last + 1u - lo;

    for (uint p0 = 0; p0 < n; p0 += PT) {
        const uint np = min(PT, n - p0);

        // Scores via SIMDGROUP MATRICES.
        //
        // QT queries against 8 positions is an 8x8 block, which is one
        // simdgroup_multiply_accumulate -- no reduction at all. The scalar version needed a
        // simd_sum per query per position (8 shuffle-reductions per position), and with KV
        // traffic already cut by the query tile those reductions were what the kernel spent
        // its time on: 19.4 GMACs in 97 ms is 0.4 TFLOPS.
        //
        // K is loaded transposed straight from device. That flag is expensive in the GEMM,
        // where a non-transposed alternative exists; here there is none short of staging a
        // hd x 8 tile per step, and the reduction it removes is worth far more.
        // Scores via SIMDGROUP MATRICES, REGISTER-BLOCKED over PB position blocks.
        //
        // One position block at a time re-reads Q for every block: per block the loop does
        // 32 K loads and 32*QB Q loads for 8*QB MACs, which at QB 2 is 96 loads for 64
        // MACs. The prefill GEMM gets 16 MACs from 8 loads, and it is 10x this kernel's
        // throughput -- 5 TFLOP/s against 0.5 -- for exactly this reason.
        //
        // Holding PB blocks at once makes Q a per-dim load shared by all of them:
        //
        //   per dim step   loads = QB (Q) + PB (K)      MACs = QB * PB
        //   PB 1, QB 2     3 loads for 2 MACs
        //   PB 4, QB 2     6 loads for 8 MACs           4x the ratio
        //
        // Registers hold QB*PB accumulators; at QB 2 and PB 4 that is 8 fragments, 16
        // floats per lane. The group falls back to one block at a time when it is partial
        // or straddles the ring wrap, which is the same code as before.
        constexpr uint PB = BLK;
        constexpr uint QBS = QT / 8u;
        const uint pgroups = (np + 8u * PB - 1u) / (8u * PB);
        for (uint gidx = sgid; gidx < pgroups; gidx += nsg) {
            const uint sp8b = gidx * 8u * PB;
            bool blocked = (sp8b + 8u * PB <= np);
            if (blocked) {
                for (uint pb = 0; pb < PB; ++pb) {
                    const uint gp = lo + p0 + sp8b + pb * 8u;
                    if (kv_run_breaks(gp, 8u, ring_mask, pt)) { blocked = false; }
                }
            }
            if (blocked) {
                simdgroup_float8x8 sacc[QBS][PB];
                #pragma unroll
                for (uint qb = 0; qb < QBS; ++qb) {
                    #pragma unroll
                    for (uint pb = 0; pb < PB; ++pb) {
                        sacc[qb][pb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                }
                // Operands fetched ONE DIM STEP AHEAD.
                //
                // Loading them immediately before the multiplies that consume them leaves
                // the fetch latency fully exposed: this kernel retires matrix instructions
                // at 567M/s while the prefill GEMM, which keeps a prefetch pipeline, manages
                // 4.86G/s on the same hardware. Nothing else explains that gap -- the reads
                // are not DRAM-bound (making them cache-resident changes nothing), the
                // threadgroup memory does not cap occupancy, and unrolling the loop does not
                // help.
                device const half * krows[PB];
                #pragma unroll
                for (uint pb = 0; pb < PB; ++pb) {
                    const uint gp = lo + p0 + sp8b + pb * 8u;
                    uint ps = kv_slot(gp, ring_mask, pt);
                    if (stage >= 5u) { ps = 0u; }        // K reads become cache-resident
                    krows[pb] = kc + (ulong)ps * kv_width + kvh * hd;
                }
                const ulong qstride = (ulong)n_heads * hd;
                device const float * qrow = q + ((ulong)q0 * n_heads + h) * hd;

                // stage >= 7: the SAME loads, but with the row stride replaced by 8, so
                // each fragment comes from one contiguous 128-byte region instead of eight
                // regions 1-8 KB apart. Results are wrong; the difference is the cost of the
                // gather, which stages 4 and 5 could not separate from traffic.
                // Prefetch only where there are registers for it.
                //
                // The pipeline holds qn[QBS] and kn[PB] on top of QBS x PB accumulators. At
                // BLK 4 that is 20 fragments and it gains 1.1%; at BLK 8 it is 36 and it
                // spills, which is why BLK 8 measured 442 tok/s against 496. But 8 doubles
                // the independent accumulator chains from eight to sixteen, and sixteen is
                // what the prefill GEMM carries -- so the two are worth separating rather
                // than rejecting BLK 8 on a measurement that had the pipeline forced on.
                constexpr bool PREFETCH = (BLK <= 4u);
                const ulong qs = (stage >= 7u) ? 8ul : qstride;
                const ulong ks = (stage >= 7u) ? 8ul : (ulong)kv_width;
                simdgroup_float8x8 qa[QBS], qn[QBS];
                simdgroup_half8x8 kb[PB], kn[PB];
                #pragma unroll
                for (uint qb = 0; qb < QBS; ++qb) {
                    simdgroup_load(qa[qb], qrow + qb * 8u * qstride, qs);
                }
                #pragma unroll
                for (uint pb = 0; pb < PB; ++pb) {
                    simdgroup_load(kb[pb], krows[pb], ks, 0, true);
                }
                if (PREFETCH) {
                    for (uint d = 0; d < hd; d += 8u) {
                        const uint dn = d + 8u;
                        if (dn < hd) {
                            #pragma unroll
                            for (uint qb = 0; qb < QBS; ++qb) {
                                simdgroup_load(qn[qb], qrow + qb * 8u * qstride + dn, qs);
                            }
                            #pragma unroll
                            for (uint pb = 0; pb < PB; ++pb) {
                                simdgroup_load(kn[pb], krows[pb] + dn, ks, 0, true);
                            }
                        }
                        #pragma unroll
                        for (uint pb = 0; pb < PB; ++pb) {
                            #pragma unroll
                            for (uint qb = 0; qb < QBS; ++qb) {
                                simdgroup_multiply_accumulate(sacc[qb][pb], qa[qb], kb[pb],
                                                              sacc[qb][pb]);
                            }
                        }
                        #pragma unroll
                        for (uint qb = 0; qb < QBS; ++qb) { qa[qb] = qn[qb]; }
                        #pragma unroll
                        for (uint pb = 0; pb < PB; ++pb) { kb[pb] = kn[pb]; }
                    }
                } else {
                    for (uint d = 8u; d < hd; d += 8u) {
                        #pragma unroll
                        for (uint pb = 0; pb < PB; ++pb) {
                            #pragma unroll
                            for (uint qb = 0; qb < QBS; ++qb) {
                                simdgroup_multiply_accumulate(sacc[qb][pb], qa[qb], kb[pb],
                                                              sacc[qb][pb]);
                            }
                        }
                        #pragma unroll
                        for (uint qb = 0; qb < QBS; ++qb) {
                            simdgroup_load(qa[qb], qrow + qb * 8u * qstride + d, qs);
                        }
                        #pragma unroll
                        for (uint pb = 0; pb < PB; ++pb) {
                            simdgroup_load(kb[pb], krows[pb] + d, ks, 0, true);
                        }
                    }
                    #pragma unroll
                    for (uint pb = 0; pb < PB; ++pb) {
                        #pragma unroll
                        for (uint qb = 0; qb < QBS; ++qb) {
                            simdgroup_multiply_accumulate(sacc[qb][pb], qa[qb], kb[pb],
                                                          sacc[qb][pb]);
                        }
                    }
                }
                #pragma unroll
                for (uint pb = 0; pb < PB; ++pb) {
                    #pragma unroll
                    for (uint qb = 0; qb < QBS; ++qb) {
                        simdgroup_store(sacc[qb][pb], sc + qb * 8u * PT + sp8b + pb * 8u,
                                        PT);
                    }
                }
                // Mask after the fact: causal, window, and past-the-batch.
                // stage >= 6 skips it: wrong results, but it says what the mask costs.
                for (uint e = (stage >= 6u ? QT * 8u * PB : lane); e < QT * 8u * PB; e += 32u) {
                    const uint pb = e / (QT * 8u);
                    const uint r = e % (QT * 8u);
                    const uint qi = r / 8u, sj = r % 8u;
                    const uint t = q0 + qi;
                    const uint gp = lo + p0 + sp8b + pb * 8u + sj;
                    const uint t_pos = start_pos + t;
                    const uint t_lo = (window > 0u && t_pos + 1u > window)
                                    ? (t_pos + 1u - window) : 0u;
                    if (!(t < n_tok && gp <= t_pos && gp >= t_lo)) {
                        sc[qi * PT + sp8b + pb * 8u + sj] = -INFINITY;
                    }
                }
                continue;
            }
            for (uint pb = 0; pb < PB; ++pb) {
                const uint sp8 = sp8b + pb * 8u;
                if (sp8 >= np) { break; }
                const uint gp0 = lo + p0 + sp8;
                // An 8-position block must not straddle the ring wrap, or its rows would come
                // from two different places. Rare, and the scalar path below covers it.
                const bool wraps = kv_run_breaks(gp0, 8u, ring_mask, pt);
                if (!wraps && sp8 + 8u <= np) {
                    // One 8x8 score block PER QUERY-BLOCK: QT queries is QT/8 blocks, and a
                    // single simdgroup_store writes only eight query rows.
                    constexpr uint QB = QT / 8u;
                    simdgroup_float8x8 sacc[QB];
                    #pragma unroll
                    for (uint qb = 0; qb < QB; ++qb) {
                        sacc[qb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                    const uint ps0 = kv_slot(gp0, ring_mask, pt);
                    // K prefetched KPF steps ahead, for the same reason the GEMM prefetches
                    // its A operands: this load is from device memory and its latency would
                    // otherwise sit fully exposed in front of the multiplies. The matrix type
                    // must match the SOURCE element type -- a float8x8 cannot be loaded from a
                    // half buffer, and getting that wrong fails the whole library compile.
                    constexpr uint KPF = 4u;
                    device const half * krow = kc + (ulong)ps0 * kv_width + kvh * hd;
                    simdgroup_half8x8 kb[KPF + 1u];
                    #pragma unroll
                    for (uint pf = 0; pf < KPF; ++pf) {
                        simdgroup_load(kb[pf], krow + min(pf * 8u, hd - 8u), kv_width,
                                       0, true);
                    }
                    for (uint d = 0; d < hd; d += 8u) {
                        if (d + KPF * 8u < hd) {
                            simdgroup_load(kb[KPF], krow + d + KPF * 8u, kv_width, 0, true);
                        }
                        #pragma unroll
                        for (uint qb = 0; qb < QB; ++qb) {
                            simdgroup_float8x8 qa;
                            simdgroup_load(qa, q + ((ulong)(q0 + qb * 8u) * n_heads + h)
                                                    * hd + d,
                                           (ulong)n_heads * hd);
                            simdgroup_multiply_accumulate(sacc[qb], qa, kb[0], sacc[qb]);
                        }
                        #pragma unroll
                        for (uint pf = 0; pf < KPF; ++pf) { kb[pf] = kb[pf + 1u]; }
                    }
                    #pragma unroll
                    for (uint qb = 0; qb < QB; ++qb) {
                        simdgroup_store(sacc[qb], sc + qb * 8u * PT + sp8, PT);
                    }
                    // Mask after the fact: causal, window, and past-the-batch.
                    for (uint e = lane; e < QT * 8u; e += 32u) {
                        const uint qi = e / 8u, sj = e % 8u;
                        const uint t = q0 + qi, gp = gp0 + sj;
                        const uint t_pos = start_pos + t;
                        const uint t_lo = (window > 0u && t_pos + 1u > window)
                                        ? (t_pos + 1u - window) : 0u;
                        if (!(t < n_tok && gp <= t_pos && gp >= t_lo)) {
                            sc[qi * PT + sp8 + sj] = -INFINITY;
                        }
                    }
                    continue;
                }
                // Edge block: partial or wrapped. Scalar, one position at a time.
                for (uint sj = 0; sj < 8u && sp8 + sj < np; ++sj) {
                    const uint gp = lo + p0 + sp8 + sj;
                    const uint ps = kv_slot(gp, ring_mask, pt);
                    device const half * k = kc + (ulong)ps * kv_width + kvh * hd;
                    device const float * qb = q + ((ulong)q0 * n_heads + h) * hd;
                    const ulong qstride = (ulong)n_heads * hd;
                    float dot[QT];
                    #pragma unroll
                    for (uint qi = 0; qi < QT; ++qi) { dot[qi] = 0.0f; }
                    for (uint i = lane; i < hd; i += 32u) {
                        const float kv = float(k[i]);
                        #pragma unroll
                        for (uint qi = 0; qi < QT; ++qi) { dot[qi] += qb[qi * qstride + i] * kv; }
                    }
                    for (uint qi = 0; qi < QT; ++qi) {
                        const float dsum = simd_sum(dot[qi]);
                        if (lane == 0) {
                            const uint t = q0 + qi, t_pos = start_pos + t;
                            const uint t_lo = (window > 0u && t_pos + 1u > window)
                                            ? (t_pos + 1u - window) : 0u;
                            const bool live = t < n_tok && gp <= t_pos && gp >= t_lo;
                            sc[qi * PT + sp8 + sj] = live ? dsum : -INFINITY;
                        }
                    }
                }
        
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stage < 2u) { continue; }

        // Online-softmax update, ONE SIMDGROUP PER QUERY.
        //
        // Looping the queries with threadgroup reductions cost four barriers per query per
        // tile -- 128 barriers per threadgroup at this context -- and measured slower than
        // no tiling at all despite cutting KV traffic eightfold. A simdgroup owns one query
        // row of `sc`, so simd_max and simd_sum suffice and the whole update needs no
        // threadgroup barrier.
        // One simdgroup per query, STRIDED: a threadgroup has nsg simdgroups and QT may
        // exceed it. Writing `if (sgid < QT)` silently skipped every query past the
        // simdgroup count, which at QT 16 with 256 threads is half of them.
        for (uint qi = sgid; !(ATTN_SKIP & 2u) && qi < QT; qi += nsg) {
            const float prev_max = rmax[qi];
            float m = -INFINITY;
            for (uint sp = lane; sp < np; sp += 32u) { m = max(m, sc[qi * PT + sp]); }
            m = simd_max(m);
            const float new_max = max(prev_max, m);
            const float scale = (prev_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);

            float ssum = 0.0f;
            for (uint sp = lane; sp < np; sp += 32u) {
                const float e = exp(sc[qi * PT + sp] - new_max);
                sc[qi * PT + sp] = e;
                ssum += e;
            }
            ssum = simd_sum(ssum);
            if (lane == 0) {
                rsum[qi] = rsum[qi] * scale + ssum;
                rmax[qi] = new_max;
            }
            // Rescale this query's running output to the new maximum.
            //
            // Guarding this on `scale != 1.0f` was measured and reverted: skipping it is
            // bit-identical (multiplying by 1.0f is exact) but worth 10-15 ms of 11.5 s,
            // inside the noise. The running max moves more often than "it settles early"
            // suggests, so most blocks do not skip.
            for (uint d = lane; d < hd; d += 32u) { acc[qi * hd + d] *= scale; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stage < 3u) { continue; }

        // Accumulate P x V, REGISTER-BLOCKED over DBB dim blocks.
        //
        // Same ratio problem the scores had. One dim block at a time loads one V fragment
        // and QB score fragments per position block, so three fetches feed two block
        // multiply-accumulates. Holding DBB dim blocks makes the score fragments shared:
        //
        //   per position block   fetches = QB (scores) + DBB (V)   ops = QB * DBB
        //   DBB 1, QB 2          3 fetches for 2 ops
        //   DBB 4, QB 2          6 fetches for 8 ops
        //
        // The running output is loaded from threadgroup memory into the accumulators and
        // stored back once per group rather than once per dim block, so the rescale the
        // online softmax applies to `acc` still works -- it operates on the same memory
        // between groups.
        const uint dblocks = hd / 8u;
        const uint full_np = (np / 8u) * 8u;
        constexpr uint DBB = BLK;
        constexpr uint QBV = QT / 8u;
        const uint dgroups = (dblocks + DBB - 1u) / DBB;
        for (uint dg = sgid; dg < dgroups; dg += nsg) {
            const uint db0 = dg * DBB;
            const uint ndb = min(DBB, dblocks - db0);
            simdgroup_float8x8 o[QBV][DBB];
            #pragma unroll
            for (uint qb = 0; qb < QBV; ++qb) {
                for (uint k = 0; k < ndb; ++k) {
                    simdgroup_load(o[qb][k],
                                   acc + qb * 8u * hd + (db0 + k) * 8u, hd);
                }
            }
            auto wraps = [&](uint sp) {
                const uint gp = lo + p0 + sp;
                return kv_run_breaks(gp, 8u, ring_mask, pt);
            };
            // V is NOT prefetched here, deliberately.
            //
            // The same one-block-ahead pipeline that gains 1.1% in the scores phase costs
            // 3.5x here: vb[DBB] and vn[DBB] on top of QBV x DBB accumulators is 16 live
            // fragments, and it spills -- 144.6 tok/s against 501.3 at a 5642-token prompt.
            // The scores phase has room because its accumulators are QBS x PB of the same
            // count but its operands are two, not eight.
            for (uint sp8 = 0; sp8 < full_np; sp8 += 8u) {
                if (wraps(sp8)) { continue; }
                const uint gp = lo + p0 + sp8;
                uint ps = kv_slot(gp, ring_mask, pt);
                // stage >= 4: same loads, same count, but always from position 0, so they
                // hit cache. The difference against stage 3 is DRAM traffic alone.
                if (stage >= 4u) { ps = 0u; }
                device const half * vrow = vc + (ulong)ps * kv_width + kvh * hd;
                simdgroup_float8x8 pa[QBV];
                #pragma unroll
                for (uint qb = 0; qb < QBV; ++qb) {
                    simdgroup_load(pa[qb], sc + qb * 8u * PT + sp8, PT);
                }
                const ulong vs = (stage >= 7u) ? 8ul : (ulong)kv_width;
                // A single-fragment V prefetch was measured here too, not just the
                // vb[DBB]/vn[DBB] pipeline the note above rejects: carrying ONE fragment
                // costs one register, not eight. 11532.9 and 11535.8 ms against a
                // 11517-11537 baseline -- noise. So P x V's cost is not the V load latency
                // either, and the absence of prefetch is not the asymmetry with the scores.
                for (uint k = 0; k < ndb; ++k) {
                    simdgroup_half8x8 vb;
                    simdgroup_load(vb, vrow + (db0 + k) * 8u, vs);
                    #pragma unroll
                    for (uint qb = 0; qb < QBV; ++qb) {
                        simdgroup_multiply_accumulate(o[qb][k], pa[qb], vb, o[qb][k]);
                    }
                }
            }
            #pragma unroll
            for (uint qb = 0; qb < QBV; ++qb) {
                for (uint k = 0; k < ndb; ++k) {
                    simdgroup_store(o[qb][k],
                                    acc + qb * 8u * hd + (db0 + k) * 8u, hd);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Positions the 8-wide blocks could not cover: the tail, and any block straddling
        // the ring wrap.
        //
        // "Normally empty" describes what it WRITES, not what it runs. Without the guard
        // every thread still walks hd/tcount x np iterations of the predicate for every
        // position block -- and on a FULL-ATTENTION layer ring_mask is 0 and np is a
        // multiple of 8, so the tail is empty every single time. Those are the 22 layers
        // that scan the whole context, so it is the worst place to spend the iterations.
        //
        // Both conditions are uniform across the threadgroup, so the barrier after this
        // block is still reached by every thread.
        // ZEROING full_np DOES NOT SKIP THE WORK, it moves every position to the SCALAR
        // tail below -- which measured 346.3 against a baseline of 864.3, 2.5x SLOWER with
        // work supposedly removed. An impossible number, and the reason the probe skips
        // the MMA inside the loop and the tail separately instead.
        const uint tail_gp0 = lo + p0;
        const bool tail_wrap = kv_range_wraps(tail_gp0, np, ring_mask);
        if (!(ATTN_SKIP & 4u) && (full_np < np || tail_wrap)) {
            for (uint d = tid; d < hd; d += tcount) {
                for (uint sp = 0; sp < np; ++sp) {
                    const uint gp = lo + p0 + sp;
                    const bool wrapped = kv_run_breaks(gp - (sp % 8u), 8u, ring_mask, pt);
                    if (sp < full_np && !wrapped) { continue; }
                    const uint ps = kv_slot(gp, ring_mask, pt);
                    const float v = float(vc[(ulong)ps * kv_width + kvh * hd + d]);
                    for (uint qi = 0; qi < QT; ++qi) {
                        acc[qi * hd + d] += sc[qi * PT + sp] * v;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint qi = 0; qi < QT; ++qi) {
        const uint t = q0 + qi;
        if (t >= n_tok) { continue; }
        const float inv = 1.0f / rsum[qi];
        device float * o = out + ((ulong)t * n_heads + h) * hd;
        for (uint d = tid; d < hd; d += tcount) { o[d] = acc[qi * hd + d] * inv; }
    }
}

#define IMPARO_QTILE_KERNEL(NAME, QT_N, HD_C, BLK_N)                                                    \
kernel void NAME(                                                                          \
    device const float * q     [[buffer(0)]],                                              \
    device const half  * kc    [[buffer(1)]],                                              \
    device const half  * vc    [[buffer(2)]],                                              \
    device float       * out   [[buffer(3)]],                                              \
    constant uint & hd [[buffer(4)]], constant uint & n_heads [[buffer(5)]],         \
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],        \
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],         \
    constant uint & ring_mask [[buffer(10)]], constant uint & n_tok [[buffer(11)]],        \
    constant uint & stage    [[buffer(12)]],                                               \
    device const uint * pt   [[buffer(18)]],                                               \
    constant uint & ptile    [[buffer(19)]],                                               \
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  lane  [[thread_index_in_simdgroup]],                                             \
    uint  sgid  [[simdgroup_index_in_threadgroup]],                                        \
    uint  nsg   [[simdgroups_per_threadgroup]])                                            \
{                                                                                          \
    attention_prefill_qtile_body<QT_N, HD_C, BLK_N>(q, kc, vc, out, ptile, hd, n_heads, n_kv, kv_width,    \
                               start_pos, window, ring_mask, pt, n_tok, stage, shared, tgid,   \
                               tid3.x, tcnt3.x, lane, sgid, nsg);                          \
}

IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile, 8, 0, 4)
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16, 16, 0, 4)
// One pipeline per blocking depth, the way rt_shape has one per shape. The ceiling is the
// register file -- a DEVICE property -- and how much is needed depends on QT and head_dim,
// which are MODEL properties, so neither alone decides it and it has to be benched on the
// pair. 4 measures 496 tok/s here, 8 measures 434; 2 has never been tried.
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16h2, 16, 256, 2)
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16h, 16, 256, 4)
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16h8, 16, 256, 8)

// COMBINED rewrite of the query-tiled kernel, per "The prefill attention rewrite,
// specified" in STATUS.md: the P x V accumulator moves from threadgroup memory into
// per-simdgroup registers, and the freed budget stages Q -- as FLOAT, because staging it
// as half rounds the inputs and moves the logits. K and V stay in device memory, which is
// what llama.cpp itself does for a half KV cache (kernel_flash_attn_ext stages K/V in
// threadgroup only for QUANTIZED caches).
//
// Bit-identity with the shipping kernel is a design input, not an aspiration:
//   - QT is 8 (the float Q tile plus the spill scratch does not fit at 16), but a windowed
//     layer's scan LOWER bound is taken from the containing SIXTEEN-query tile, so every
//     query keeps exactly the position-block partition the QT-16 kernel gave it and the
//     online softmax reassociates nothing. The positions this over-scans are masked, and
//     a masked slot contributes exp(-INF) = 0 to the sum and 0 * V to the accumulator,
//     both of which are exact.
//   - The accumulator rescale and the final 1/sum are simdgroup_multiply by a DIAGONAL
//     matrix: row qi of diag*O is scale[qi]*O[qi][*] plus seven products by 0.0f, exact
//     for the finite values this kernel produces.
//   - Tail positions (a partial 8-block, or a block straddling the ring wrap) must add to
//     the accumulator one position at a time in the shipping kernel's order, so the owning
//     simdgroups spill their fragments to threadgroup scratch, the scalar tail runs there
//     unchanged, and the fragments reload. Conditional and uniform, and on the
//     full-attention layers that dominate the scan it never fires.
//
// Threadgroup budget at QT 8, head_dim 256, PT 128:
//   staged Q     8 x 256 floats    8 KB
//   scores       8 x 128 floats    4 KB
//   spill        8 x 256 floats    8 KB
//   diag + running max/sum         ~0.3 KB
//                                  20.3 KB of 32
// PT IS A RUNTIME ARGUMENT, not a template parameter. It appears only as a stride and a
// loop bound -- `sc[qi * PT + sp]`, `simdgroup_load(..., PT)`, `p0 += PT` -- and no array
// is declared with it, so nothing forces it to be compiled in. Frozen at 128 it ignored
// most of the device: `qcomb_derive_pt` says this budget fits 480 at head_dim 256 and 864
// at 64, and PT is the position tile a scan advances by, so a bigger one is fewer passes
// over the KV.
//
// HD_C and NSG stay compile-time because they size a REGISTER array (o[QROWS][NDB],
// NDB = HD_C/8/NSG), which is the only thing here the hardware actually forces.
template<uint QT, uint HD_C, uint BLK, uint NSG, bool DSPILL, uint KVW, bool HALF_Q = false,
         bool KT = false>
static void attention_prefill_qcomb_body(
    device const float * q, device const half * kc, device const half * vc,
    device float * out, device float * dspill,
    device const half * kt, uint kt_stride,
    uint PT,
    uint head_dim, uint n_heads, uint n_kv, uint kv_width_u,
    uint start_pos, uint window, uint ring_mask, device const uint * pt,
    uint n_tok, uint stage,
    device half * xh, uint xh_on,
    threadgroup float * shared, uint3 tgid, uint tid, uint tcount,
    uint lane, uint sgid, uint nsg)
{
    // The K/V row stride as a COMPILE-TIME constant when the host gave it (IMPARO_KVW<slot>,
    // set with the head dims before init): every K and V tile load below carries this
    // stride, and as an immediate it costs no uniform read and no multiply per tile. Measured
    // on the FA op first: -10% at 16896 keys, -13% at 8704, -15% at 1024. 0 = the uniform.
    const uint kv_width = (KVW != 0u) ? KVW : kv_width_u;
    // The body is written in 8-row GROUPS: QT 8 is one group and keeps the original
    // structure (and its bit-exact arithmetic) unchanged; QT 16 runs two groups that
    // SHARE every K and V read, which is what halves the KV traffic per scan.
    static_assert(QT == 8u || QT == 16u, "the row-group structure covers QT 8 and 16");
    constexpr uint QROWS = QT / 8u;
    (void)head_dim;
    constexpr uint hd = HD_C;
    const uint h = tgid.x, q0 = tgid.y * QT;
    const uint kvh = h / (n_heads / n_kv);

    // At QT 16 the float Q region would be 32 KB on the hd-512 layers, so the QT-16
    // variants stage Q as HALF and the layout packs the score tile right after the half
    // region. QT 8 keeps the original float-sized region whatever HALF_Q says, so the
    // shipping layouts are byte-identical to what they were.
    constexpr uint SQ_FLOATS = (HALF_Q && QT == 16u) ? QT * hd / 2u : QT * hd;
    threadgroup float * sq    = shared;                    // QT x hd staged queries
    threadgroup float * sc    = sq + SQ_FLOATS;            // QT x PT scores
    threadgroup float * rmax  = sc + QT * PT;              // QT running maxima
    threadgroup float * rsum  = rmax + QT;                 // QT running sums
    threadgroup float * diag  = rsum + QT;                 // QROWS x 8x8, scale on diagonal
    // Tail scratch. At head_dim 256 it fits in threadgroup memory; at 512 the staged Q
    // leaves no room, so it lives in device memory instead. Tails fire only for the two
    // tiles of the final partial 16-group (full-attention layers have no ring and their
    // scan length is n_tok mod 8 away from a whole chunk only there), and those two tiles
    // have distinct tgid.y parity, so head*2 + parity indexes disjoint regions. (At QT 16
    // the final group is ONE tile, so the same indexing is disjoint a fortiori.)
    threadgroup float * spill = diag + QROWS * 64u;        // QT x hd, HD 256 only
    device float * dsp = DSPILL
        ? dspill + ((ulong)tgid.x * 2u + (tgid.y & 1u)) * QT * HD_C : dspill;

    // Stage Q once for the whole scan. Rows past the batch are zero; their scores are
    // masked to -INFINITY below, exactly as the shipping kernel masks them. HALF_Q stages
    // as half -- llama.cpp's own flash-attention staging; the scalar edge path below
    // rounds identically, so both score paths see the same operand values.
    threadgroup half * sqh = (threadgroup half *)sq;
    // Staged TRANSPOSED: sq[dim][query], not sq[query][dim]. Same bytes, same loop, one
    // different index -- and it is what lets the score MMA take K the way K is stored.
    // S[q][p] = Q[q][d] . Kt[d][p] forces a TRANSPOSING load of K from device, 1920 of
    // them per position tile. St[p][q] = K[p][d] . Qt[d][q] loads K non-transposed,
    // matching its [slot][dim] layout, and pays one transposing THREADGROUP store per
    // score fragment instead -- 60 per tile. Everything downstream still sees
    // sc[query][position], so softmax, masking and P x V are untouched.
    for (uint e = tid; e < QT * hd; e += tcount) {
        const uint row = e / hd, dim = e % hd;
        const uint t = q0 + row;
        const float qv = t < n_tok ? q[((ulong)t * n_heads + h) * hd + dim] : 0.0f;
        if (HALF_Q) { sqh[dim * QT + row] = half(qv); } else { sq[dim * QT + row] = qv; }
    }
    // Only the diagonals are ever written after this; the off-diagonal zeros are permanent.
    for (uint e = tid; e < QROWS * 64u; e += tcount) { diag[e] = 0.0f; }
    if (tid < QT) { rmax[tid] = -INFINITY; rsum[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fixed dim-slice ownership: simdgroup sgid holds dims [db0*8, db0*8 + NDB*8) in
    // registers for the whole kernel. Requires exactly hd / (8 * NDB) simdgroups, which
    // the host guarantees by dispatching NSG * 32 threads for this kernel.
    constexpr uint NDB = HD_C / 8u / NSG;
    const uint db0 = sgid * NDB;
    simdgroup_float8x8 o[QROWS][NDB];
    #pragma unroll
    for (uint g = 0; g < QROWS; ++g) {
        #pragma unroll
        for (uint k = 0; k < NDB; ++k) {
            o[g][k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // BOTH scan bounds come from the containing 16-query tile, not from this tile's own
    // queries. The lower bound was argued in the header note; the upper bound matters for
    // the same reason at the other end: it decides where the final position block ends,
    // and with it which chunk is partial and falls to the scalar tail. Scanning to this
    // tile's own last query moved that boundary 8 positions down, so positions the QT-16
    // kernel summed through the full-chunk MMA went through the tail's sequential adds
    // instead -- same values, different association, logits off in the 4th decimal. The
    // extra positions this scans are masked for every query here and contribute exact
    // zeros through both phases.
    // At QT 8: 256 matches the QT-16 tiled kernel, so both bounds come from the containing
    // 16-tile; 512's reference is the QT-8 kernel, whose bounds are the tile's own. A
    // QT-16 instantiation's own tile IS 16 queries, so it takes its own bounds directly.
    const uint qb0   = (HD_C == 256u && QT == 8u) ? (q0 & ~15u) : q0;
    const uint qspan = (HD_C == 256u && QT == 8u) ? 16u : QT;
    const uint last16 = min(qb0 + qspan, n_tok) - 1u;
    const uint pos_last = start_pos + last16;
    const uint pos_first16 = start_pos + qb0;
    const uint lo = (window > 0u && pos_first16 + 1u > window)
                  ? (pos_first16 + 1u - window) : 0u;
    const uint n = pos_last + 1u - lo;

    for (uint p0 = 0; p0 < n; p0 += PT) {
        const uint np = min(PT, n - p0);

        // THE TILE'S PAGES, LOADED ONCE. On a paged layer kv_slot reads pt[gp / 64] for
        // every 8-slot block, and the block's K or V address depends on that load -- so
        // each block paid a device round trip before its first operand could even be
        // requested: 4 per tile per simdgroup in the score phase and 16 in P x V at hd 64,
        // PT 128. A tile spans at most PT/64 + 1 pages, so lane l fetches the l-th one
        // here, in ONE round trip, and every run below reads its page with a register
        // shuffle instead of a load. Same slots in the same order: bit-identical.
        // Ring layers never consult the table; a KV_PAGED = false build compiles it out.
        const uint pg0 = (lo + p0) / KV_PAGE_CELLS;
        uint tile_pg = 0u;
        if (KV_PAGED && ring_mask == 0u) {
            const uint npg = (lo + p0 + np - 1u) / KV_PAGE_CELLS - pg0 + 1u;
            if (lane < npg) { tile_pg = pt[pg0 + lane]; }
        }
        // UNIFORM call sites only: simd_shuffle needs every lane present. The scalar
        // edge and tail paths, which run per thread, keep kv_slot.
        auto tile_slot = [&](uint gp) -> uint {
            if (ring_mask > 0u) { return gp & ring_mask; }
            if (!KV_PAGED) { return gp; }
            return simd_shuffle(tile_pg, (ushort)(gp / KV_PAGE_CELLS - pg0)) * KV_PAGE_CELLS
                 + gp % KV_PAGE_CELLS;
        };

        // Scores: the shipping kernel's register-blocked simdgroup MMA, with Q read from
        // threadgroup instead of device. Same d order into the same accumulators, so every
        // score is the same sum in the same order. The work unit is (row group, position
        // group): at QT 8 that is exactly the original position-group assignment.
        constexpr uint PB = BLK;
        const uint pgroups = (np + 8u * PB - 1u) / (8u * PB);
        // The work unit is a POSITION GROUP, and one simdgroup carries every row
        // group in it, so each K fragment is loaded once and multiplied QROWS
        // times. It used to be (row group, position group), which meant two
        // simdgroups each re-loaded the same K -- so QT 16 halved the number of
        // query tiles and doubled the row groups per tile, leaving total K
        // traffic exactly unchanged. That is why every QT-16 configuration
        // measured flat or worse. At QROWS 1 this loop is what it always was.
        for (uint gidx = sgid; gidx < pgroups; gidx += nsg) {
            const uint sp8b = gidx * 8u * PB;
            bool blocked = (sp8b + 8u * PB <= np);
            if (blocked) {
                for (uint pb = 0; pb < PB; ++pb) {
                    const uint gp = lo + p0 + sp8b + pb * 8u;
                    if (kv_run_breaks(gp, 8u, ring_mask, pt)) { blocked = false; }
                }
            }
            if (blocked) {
                simdgroup_float8x8 sacc[QROWS][PB];
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    #pragma unroll
                    for (uint pb = 0; pb < PB; ++pb) {
                        sacc[g][pb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                }
                device const half * krows[PB];
                // One slot lookup for the whole group when its PB blocks share a run
                // (no page boundary, no wrap between them); the blocks' slots are then
                // consecutive and the addresses affine. Otherwise one lookup per block.
                const uint gpg = lo + p0 + sp8b;
                const bool one_run = (ring_mask > 0u)
                    ? ((gpg & ring_mask) + 8u * PB <= ring_mask + 1u)
                    : (!KV_PAGED || gpg % KV_PAGE_CELLS + 8u * PB <= KV_PAGE_CELLS);
                const uint ps_g = tile_slot(gpg);
                #pragma unroll
                for (uint pb = 0; pb < PB; ++pb) {
                    const uint gp = gpg + pb * 8u;
                    const uint ps = one_run ? ps_g + pb * 8u : tile_slot(gp);
                    // KT: base of the transposed tile [dim][slot] for this chunk's 8
                    // slots; the per-step offset advances by 8 dims * kt_stride.
                    krows[pb] = KT ? kt + (ulong)(kvh * hd) * kt_stride + ps
                                   : kc + (ulong)ps * kv_width + kvh * hd;
                }
                simdgroup_float8x8 qa[QROWS], qn[QROWS];
                simdgroup_half8x8 qah[QROWS], qnh[QROWS];
                simdgroup_half8x8 kb[PB], kn[PB];
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    if (HALF_Q) {
                        simdgroup_load(qah[g], sqh + g * 8u, QT);
                    } else {
                        simdgroup_load(qa[g], sq + g * 8u, QT);
                    }
                }
                #pragma unroll
                for (uint pb = 0; pb < PB; ++pb) {
                    if (KT) { simdgroup_load(kb[pb], krows[pb], kt_stride, 0, true); }
                    else    { simdgroup_load(kb[pb], krows[pb], kv_width, 0, false); }
                }
                // Unroll FULL (64 = the loop's own trip count at head size 512).
                // Re-swept after the row-group loop moved inside this body,
                // which changed what one iteration carries. 16k prefill, ms,
                // median of three: 8 -> 32440, 16 -> 32302, 32 -> 32179,
                // 64 -> 31591; and on the QT-8 path the quantized caches still
                // use, 64 -> 32868 against 16 -> 33263. Before the restructure
                // the answer was the opposite (16 won, uncapped lost by 3.7%),
                // which is the whole reason this is swept per shape rather than
                // reasoned about. Bit-identical either way: unrolling changes
                // neither the operations nor their order.
// The unroll factor is a PREPROCESSOR value, injected at library compile, because
// Metal's pragma is processed before template instantiation -- making it a template
// parameter fails with "use of undeclared identifier". That is why it cannot be swept:
// each candidate needs its own library compile, and this tuner deliberately measures
// within one process.
//
// It is still not frozen in source. IMPARO_QCOMB_UNROLL overrides it, so re-checking it
// after a kernel shape changes is a re-run rather than an edit-and-rebuild -- and it has
// needed re-checking twice, because the answer INVERTED: 16 won before the row-group
// restructure, 64 after.
#pragma unroll(IMPARO_QCOMB_UNROLL)
                for (uint d = 0; d < hd; d += 8u) {
                    const uint dn = d + 8u;
                    if (dn < hd) {
                        #pragma unroll
                        for (uint g = 0; g < QROWS; ++g) {
                            if (HALF_Q) {
                                simdgroup_load(qnh[g], sqh + (ulong)dn * QT + g * 8u, QT);
                            } else {
                                simdgroup_load(qn[g], sq + (ulong)dn * QT + g * 8u, QT);
                            }
                        }
                        // ATTN_SKIP bit 3 (8): drop the per-step K RELOAD but keep every
                        // MMA. kb was validly filled before this loop, so the multiplies run
                        // on real data with the memory traffic removed -- which separates
                        // "this loop is waiting on K" from "this loop is at the matrix
                        // unit's limit". Wrong answers, only the time is read.
                        if (!(ATTN_SKIP & 8u))
                        #pragma unroll
                        for (uint pb = 0; pb < PB; ++pb) {
                            if (KT) {
                                simdgroup_load(kn[pb], krows[pb] + (ulong)dn * kt_stride,
                                               kt_stride, 0, true);
                            } else {
                                simdgroup_load(kn[pb], krows[pb] + dn, kv_width, 0, false);
                            }
                        }
                    }
                    if (!(ATTN_SKIP & 1u))
                    #pragma unroll
                    for (uint pb = 0; pb < PB; ++pb) {
                        #pragma unroll
                        for (uint g = 0; g < QROWS; ++g) {
                            // K first: the accumulator is St[position][query].
                            if (HALF_Q) {
                                simdgroup_multiply_accumulate(sacc[g][pb], kb[pb], qah[g],
                                                              sacc[g][pb]);
                            } else {
                                simdgroup_multiply_accumulate(sacc[g][pb], kb[pb], qa[g],
                                                              sacc[g][pb]);
                            }
                        }
                    }
                    #pragma unroll
                    for (uint g = 0; g < QROWS; ++g) {
                        if (HALF_Q) { qah[g] = qnh[g]; } else { qa[g] = qn[g]; }
                    }
                    // The rotation goes with the reload: with bit 3 set, kn is never
                    // written, so propagating it would multiply garbage registers.
                    if (!(ATTN_SKIP & 8u))
                    #pragma unroll
                    for (uint pb = 0; pb < PB; ++pb) { kb[pb] = kn[pb]; }
                }
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    #pragma unroll
                    for (uint pb = 0; pb < PB; ++pb) {
                        // transpose=true: St[position][query] lands as sc[query][position],
                        // so every phase after this sees the layout it always saw.
                        simdgroup_store(sacc[g][pb],
                                        sc + (ulong)(g * 8u) * PT + sp8b + pb * 8u, PT,
                                        0, true);
                    }
                }
                // NOTHING TO MASK when every slot in this block is live for every
                // query in the tile. `blocked` has already proved the 8*PB positions are
                // complete and do not straddle the ring; what is left is
                //
                //   the whole QUERY tile exists   or its nonexistent rows stay live
                //   gp_max <= EARLIEST query pos  causal holds for all QT of them
                //   gp_min >= LATEST query's lo   the window holds for all QT of them
                //
                // Taking the earliest position for the causal bound and the latest for the
                // window bound is what makes one test cover the whole tile. In a deep
                // prefill almost every block is far below the diagonal and passes.
                const uint gp_min = lo + p0 + sp8b;
                const uint gp_max = gp_min + 8u * PB - 1u;
                const uint pos_first = start_pos + q0;
                const uint pos_last = pos_first + QT - 1u;
                const uint latest_lo = (window > 0u && pos_last + 1u > window)
                                     ? (pos_last + 1u - window) : 0u;
                const bool all_live = ATTN_LIVE_MASK && q0 + QT <= n_tok
                                   && gp_max <= pos_first && gp_min >= latest_lo;
                if (!all_live) {
                    for (uint e = lane; e < QROWS * 8u * 8u * PB; e += 32u) {
                        const uint g = e / (64u * PB);
                        const uint rem = e % (64u * PB);
                        const uint pb = rem / 64u;
                        const uint r = rem % 64u;
                        const uint qi = g * 8u + r / 8u, sj = r % 8u;
                        const uint t = q0 + qi;
                        const uint gp = lo + p0 + sp8b + pb * 8u + sj;
                        const uint t_pos = start_pos + t;
                        const uint t_lo = (window > 0u && t_pos + 1u > window)
                                        ? (t_pos + 1u - window) : 0u;
                        if (!(t < n_tok && gp <= t_pos && gp >= t_lo)) {
                            sc[qi * PT + sp8b + pb * 8u + sj] = -INFINITY;
                        }
                    }
                }
                continue;
            }
            // Edge group: a partial 8-block or one straddling the ring wrap.
            // Row groups run one at a time here -- the path is scalar anyway,
            // and on the full-attention layers that dominate it never fires.
            for (uint g = 0; g < QROWS; ++g)
            for (uint pb = 0; pb < PB; ++pb) {
                const uint sp8 = sp8b + pb * 8u;
                if (sp8 >= np) { break; }
                const uint gp0 = lo + p0 + sp8;
                const bool wraps8 = kv_run_breaks(gp0, 8u, ring_mask, pt);
                if (!wraps8 && sp8 + 8u <= np) {
                    simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    const uint ps0 = kv_slot(gp0, ring_mask, pt);
                    constexpr uint KPF = 4u;
                    device const half * krow = KT
                        ? kt + (ulong)(kvh * hd) * kt_stride + ps0
                        : kc + (ulong)ps0 * kv_width + kvh * hd;
                    simdgroup_half8x8 kb[KPF + 1u];
                    #pragma unroll
                    for (uint pf = 0; pf < KPF; ++pf) {
                        if (KT) {
                            simdgroup_load(kb[pf],
                                           krow + (ulong)min(pf * 8u, hd - 8u) * kt_stride,
                                           kt_stride, 0, true);
                        } else {
                            simdgroup_load(kb[pf], krow + min(pf * 8u, hd - 8u), kv_width,
                                           0, false);
                        }
                    }
                    for (uint d = 0; d < hd; d += 8u) {
                        if (d + KPF * 8u < hd) {
                            if (KT) {
                                simdgroup_load(kb[KPF],
                                               krow + (ulong)(d + KPF * 8u) * kt_stride,
                                               kt_stride, 0, true);
                            } else {
                                simdgroup_load(kb[KPF], krow + d + KPF * 8u, kv_width,
                                               0, false);
                            }
                        }
                        // Same swap as the blocked path: Qt is staged [dim][query], K
                        // loads the way it is stored, and the accumulator is
                        // St[position][query].
                        if (HALF_Q) {
                            simdgroup_half8x8 qa;
                            simdgroup_load(qa, sqh + (ulong)d * QT + g * 8u, QT);
                            simdgroup_multiply_accumulate(sacc, kb[0], qa, sacc);
                        } else {
                            simdgroup_float8x8 qa;
                            simdgroup_load(qa, sq + (ulong)d * QT + g * 8u, QT);
                            simdgroup_multiply_accumulate(sacc, kb[0], qa, sacc);
                        }
                        #pragma unroll
                        for (uint pf = 0; pf < KPF; ++pf) { kb[pf] = kb[pf + 1u]; }
                    }
                    simdgroup_store(sacc, sc + (ulong)(g * 8u) * PT + sp8, PT, 0, true);
                    for (uint e = lane; e < 8u * 8u; e += 32u) {
                        const uint qi = g * 8u + e / 8u, sj = e % 8u;
                        const uint t = q0 + qi, gp = gp0 + sj;
                        const uint t_pos = start_pos + t;
                        const uint t_lo = (window > 0u && t_pos + 1u > window)
                                        ? (t_pos + 1u - window) : 0u;
                        if (!(t < n_tok && gp <= t_pos && gp >= t_lo)) {
                            sc[qi * PT + sp8 + sj] = -INFINITY;
                        }
                    }
                    continue;
                }
                for (uint sj = 0; sj < 8u && sp8 + sj < np; ++sj) {
                    const uint gp = lo + p0 + sp8 + sj;
                    const uint ps = kv_slot(gp, ring_mask, pt);
                    device const half * krow = kc + (ulong)ps * kv_width + kvh * hd;
                    float dot[8u];
                    #pragma unroll
                    for (uint ql = 0; ql < 8u; ++ql) { dot[ql] = 0.0f; }
                    for (uint i = lane; i < hd; i += 32u) {
                        const float kv = float(krow[i]);
                        #pragma unroll
                        for (uint ql = 0; ql < 8u; ++ql) {
                            const float qv = HALF_Q
                                ? float(sqh[i * QT + g * 8u + ql])
                                : sq[i * QT + g * 8u + ql];
                            dot[ql] += qv * kv;
                        }
                    }
                    for (uint ql = 0; ql < 8u; ++ql) {
                        const float dsum = simd_sum(dot[ql]);
                        if (lane == 0) {
                            const uint qi = g * 8u + ql;
                            const uint t = q0 + qi, t_pos = start_pos + t;
                            const uint t_lo = (window > 0u && t_pos + 1u > window)
                                            ? (t_pos + 1u - window) : 0u;
                            const bool live = t < n_tok && gp <= t_pos && gp >= t_lo;
                            sc[qi * PT + sp8 + sj] = live ? dsum : -INFINITY;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax, one simdgroup per query: identical arithmetic to the shipping
        // kernel. The accumulator rescale becomes a diagonal write, applied by MMA below.
        // NO phase staging here, deliberately. Two `if (stage < n) continue;`
        // guards were added to attribute the phases and cost 2.0% of the whole
        // prefill -- `stage` is a runtime uniform, so the branches survive
        // compilation and break scheduling across the loop body. The
        // measurement they bought, at a 16k-deep 512-token chunk: scores
        // 25.2 ms, softmax 0.2, P x V 21.2, so the score phase is the slower
        // half at 2.68 TFLOPS against P x V's 3.18. Recorded here rather than
        // re-measurable, because the instrument was slower than the thing it
        // measured. Attribute in the query-tiled kernel (which is staged) or
        // behind a function constant, never with a runtime branch.
        for (uint qi = sgid; qi < QT; qi += nsg) {
            const float prev_max = rmax[qi];
            float m = -INFINITY;
            for (uint sp = lane; sp < np; sp += 32u) { m = max(m, sc[qi * PT + sp]); }
            m = simd_max(m);
            const float new_max = max(prev_max, m);
            const float scale = (prev_max == -INFINITY) ? 0.0f : exp(prev_max - new_max);
            // A row past the batch is all -INFINITY forever: exp(-INF - -INF) is NaN, and
            // unlike the shipping kernel's elementwise ops, the diagonal MMA below MIXES
            // rows -- 0.0f * NaN is NaN, so one dead row would poison the live ones. Dead
            // rows get exact zeros instead; they never reach the output either way, and a
            // LIVE row always has a live position in its first block, so its arithmetic is
            // untouched.
            const bool dead = new_max == -INFINITY;
            float ssum = 0.0f;
            for (uint sp = lane; sp < np; sp += 32u) {
                const float e = dead ? 0.0f : exp(sc[qi * PT + sp] - new_max);
                sc[qi * PT + sp] = e;
                ssum += e;
            }
            ssum = simd_sum(ssum);
            if (lane == 0) {
                rsum[qi] = rsum[qi] * scale + ssum;
                rmax[qi] = new_max;
                diag[(qi / 8u) * 64u + (qi % 8u) * 9u] = scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P x V into the register accumulator. diag(scale) * O scales row qi by scale[qi]
        // exactly -- what the shipping kernel's elementwise rescale did.
        {
            #pragma unroll
            for (uint g = 0; g < QROWS; ++g) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, diag + g * 64u, 8u);
                #pragma unroll
                for (uint k = 0; k < NDB; ++k) {
                    simdgroup_float8x8 t;
                    simdgroup_multiply(t, dm, o[g][k]);
                    o[g][k] = t;
                }
            }
        }
        const uint full_np = (np / 8u) * 8u;
        // Filled once from a real V row so the ATTN_SKIP bit-4 probe multiplies defined
        // data rather than uninitialised registers.
        simdgroup_half8x8 vfix[NDB];
        if (ATTN_SKIP & 16u) {
            const uint ps0 = kv_slot(lo + p0, ring_mask, pt);
            device const half * v0 = vc + (ulong)ps0 * kv_width + kvh * hd;
            #pragma unroll
            for (uint k = 0; k < NDB; ++k) {
                simdgroup_load(vfix[k], v0 + (db0 + k) * 8u, kv_width);
            }
        }
        // P x V in RUNS of whole 8-slot blocks that share one slot base: a page on a
        // paged layer, the stretch up to the wrap on a ring. One slot lookup per run and
        // pointer increments inside it -- the loop shape a KV_PAGED = false build gets
        // for free, and the shape is what mattered. Measured on the 17123-token prefill,
        // pool on, three interleaved rounds against a page-table load per block:
        //   this loop            +1.8%  of the WHOLE prefill (~12% of attention)
        //   shuffle per block    +0.1%  (the load gone, the dependency kept)
        //   KV_PAGED=false build +2.0%  (pool off, identity table)
        // An address that depends on anything but the loop counter stops the compiler
        // issuing a run's V loads ahead of its MMAs. A block straddling the wrap is
        // skipped here and summed by the tail, as before; a block cannot straddle a page
        // (8 divides KV_PAGE_CELLS). Same blocks, same order: bit-identical.
        for (uint sp8 = 0; sp8 < full_np; ) {
            const uint gp = lo + p0 + sp8;
            const uint left = (full_np - sp8) / 8u;
            uint nblk = left;
            bool straddle = false;
            if (ring_mask > 0u) {
                const uint lim = ring_mask + 1u - (gp & ring_mask);
                nblk = min(left, lim / 8u);
                straddle = (lim % 8u != 0u) && (nblk < left);
            } else if (KV_PAGED) {
                nblk = min(left, (KV_PAGE_CELLS - gp % KV_PAGE_CELLS) / 8u);
            }
            device const half * vrow = vc + (ulong)tile_slot(gp) * kv_width + kvh * hd;
            // EXPLICIT ONE-BLOCK LOOK-AHEAD: block b+1's P and V fragments are requested
            // before block b's multiplies, the rotation the score loop already uses for K.
            simdgroup_float8x8 pa[QROWS], pn[QROWS];
            simdgroup_half8x8  vb[NDB],   vn[NDB];
            if (nblk > 0u) {
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    simdgroup_load(pa[g], sc + (ulong)(g * 8u) * PT + sp8, PT);
                }
                #pragma unroll
                for (uint k = 0; k < NDB; ++k) {
                    if (ATTN_SKIP & 16u) { vb[k] = vfix[k]; }
                    else { simdgroup_load(vb[k], vrow + (db0 + k) * 8u, kv_width); }
                }
            }
            for (uint b = 0; b < nblk; ++b, sp8 += 8u, vrow += 8u * kv_width) {
                const bool more = b + 1u < nblk;
                if (more) {
                    #pragma unroll
                    for (uint g = 0; g < QROWS; ++g) {
                        simdgroup_load(pn[g], sc + (ulong)(g * 8u) * PT + sp8 + 8u, PT);
                    }
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) {
                        if (ATTN_SKIP & 16u) { vn[k] = vfix[k]; }
                        else { simdgroup_load(vn[k], vrow + 8u * kv_width + (db0 + k) * 8u,
                                              kv_width); }
                    }
                }
                // ATTN_SKIP bit 4 (16): drop the per-position V LOAD but keep every MMA.
                // vfix is filled once from the first row this threadgroup would read, so
                // the multiplies run on real data with the traffic removed -- the P x V
                // half of the load-bound vs MMA-bound question. Wrong answers, only the
                // time is read.
                if (!(ATTN_SKIP & 4u)) {
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) {
                        #pragma unroll
                        for (uint g = 0; g < QROWS; ++g) {
                            simdgroup_multiply_accumulate(o[g][k], pa[g], vb[k], o[g][k]);
                        }
                    }
                }
                if (more) {
                    #pragma unroll
                    for (uint g = 0; g < QROWS; ++g) { pa[g] = pn[g]; }
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) { vb[k] = vn[k]; }
                }
            }
            if (straddle) { sp8 += 8u; }
        }
        // Tail: partial 8-block, or a block straddling the ring wrap. Position-at-a-time
        // adds in the shipping kernel's order, on spilled fragments. Conditional, uniform.
        const uint tail_gp0 = lo + p0;
        const bool tail_wrap = kv_range_wraps(tail_gp0, np, ring_mask);
        if (full_np < np || tail_wrap) {
            if (DSPILL) {
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) {
                        simdgroup_store(o[g][k], dsp + (ulong)(g * 8u) * hd + (db0 + k) * 8u,
                                        hd);
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                for (uint d = tid; d < hd; d += tcount) {
                    for (uint sp = 0; sp < np; ++sp) {
                        const uint gp = lo + p0 + sp;
                        const bool wrapped = kv_run_breaks(gp - (sp % 8u), 8u, ring_mask, pt);
                        if (sp < full_np && !wrapped) { continue; }
                        const uint ps = kv_slot(gp, ring_mask, pt);
                        const float v = float(vc[(ulong)ps * kv_width + kvh * hd + d]);
                        for (uint qi = 0; qi < QT; ++qi) {
                            dsp[qi * hd + d] += sc[qi * PT + sp] * v;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) {
                        simdgroup_load(o[g][k], dsp + (ulong)(g * 8u) * hd + (db0 + k) * 8u,
                                       hd);
                    }
                }
            } else {
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) {
                        simdgroup_store(o[g][k], spill + (ulong)(g * 8u) * hd + (db0 + k) * 8u,
                                        hd);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint d = tid; d < hd; d += tcount) {
                    for (uint sp = 0; sp < np; ++sp) {
                        const uint gp = lo + p0 + sp;
                        const bool wrapped = kv_run_breaks(gp - (sp % 8u), 8u, ring_mask, pt);
                        if (sp < full_np && !wrapped) { continue; }
                        const uint ps = kv_slot(gp, ring_mask, pt);
                        const float v = float(vc[(ulong)ps * kv_width + kvh * hd + d]);
                        for (uint qi = 0; qi < QT; ++qi) {
                            spill[qi * hd + d] += sc[qi * PT + sp] * v;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    #pragma unroll
                    for (uint k = 0; k < NDB; ++k) {
                        simdgroup_load(o[g][k], spill + (ulong)(g * 8u) * hd + (db0 + k) * 8u,
                                       hd);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalise and write out: diag(1/sum) by MMA -- multiplying by the same reciprocal
    // the shipping kernel used -- then a guarded copy through the spill scratch.
    for (uint qi = sgid; qi < QT; qi += nsg) {
        // 0, not 1/0: a dead row's INF entry would meet its zero accumulator in the MMA
        // and make NaN. Live rows have rsum >= 1.
        if (lane == 0) {
            diag[(qi / 8u) * 64u + (qi % 8u) * 9u] = rsum[qi] > 0.0f ? 1.0f / rsum[qi] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (DSPILL) {
        // A FULL tile stores its normalised fragments straight to the output: all rows
        // are valid, and the device scratch's head*2+parity indexing is only
        // collision-free for the <= 2 partial tiles of the final 16-group -- routing every
        // tile through it made 16 threadgroups race on one region.
        if (q0 + QT <= n_tok) {
            #pragma unroll
            for (uint g = 0; g < QROWS; ++g) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, diag + g * 64u, 8u);
                device float * ob = out + ((ulong)(q0 + g * 8u) * n_heads + h) * hd;
                #pragma unroll
                for (uint k = 0; k < NDB; ++k) {
                    simdgroup_float8x8 t;
                    simdgroup_multiply(t, dm, o[g][k]);
                    simdgroup_store(t, ob + (db0 + k) * 8u, (ulong)n_heads * hd);
                }
            }
            // Half mirror for the wo projection (IMPARO_HALF_A): converts the rows this
            // threadgroup just stored -- same rounding, same slots as the cvt pass. The
            // device barrier makes the simdgroup stores visible to the elementwise loop;
            // the re-read is L1-hot.
            if (xh_on != 0u) {
                threadgroup_barrier(mem_flags::mem_device);
                for (uint e = tid; e < QT * hd; e += tcount) {
                    const ulong idx = ((ulong)(q0 + e / hd) * n_heads + h) * hd + e % hd;
                    xh[idx] = half(out[idx]);
                }
            }
        } else {
            #pragma unroll
            for (uint g = 0; g < QROWS; ++g) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, diag + g * 64u, 8u);
                #pragma unroll
                for (uint k = 0; k < NDB; ++k) {
                    simdgroup_float8x8 t;
                    simdgroup_multiply(t, dm, o[g][k]);
                    simdgroup_store(t, dsp + (ulong)(g * 8u) * hd + (db0 + k) * 8u, hd);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
            for (uint qi = 0; qi < QT; ++qi) {
                const uint t = q0 + qi;
                if (t >= n_tok) { continue; }
                device float * op = out + ((ulong)t * n_heads + h) * hd;
                device half * xp = xh + ((ulong)t * n_heads + h) * hd;
                for (uint d = tid; d < hd; d += tcount) {
                    const float v = dsp[qi * hd + d];
                    op[d] = v;
                    if (xh_on != 0u) { xp[d] = half(v); }
                }
            }
        }
    } else {
        {
            #pragma unroll
            for (uint g = 0; g < QROWS; ++g) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, diag + g * 64u, 8u);
                #pragma unroll
                for (uint k = 0; k < NDB; ++k) {
                    simdgroup_float8x8 t;
                    simdgroup_multiply(t, dm, o[g][k]);
                    simdgroup_store(t, spill + (ulong)(g * 8u) * hd + (db0 + k) * 8u, hd);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint qi = 0; qi < QT; ++qi) {
            const uint t = q0 + qi;
            if (t >= n_tok) { continue; }
            device float * op = out + ((ulong)t * n_heads + h) * hd;
            device half * xp = xh + ((ulong)t * n_heads + h) * hd;
            for (uint d = tid; d < hd; d += tcount) {
                const float v = spill[qi * hd + d];
                op[d] = v;
                if (xh_on != 0u) { xp[d] = half(v); }
            }
        }
    }
}

#define IMPARO_QCOMB_KERNEL(NAME, QT_N, HD_N, BLK_N, NSG_N, DS_N, KVW_N, ...)  \
    IMPARO_QCOMB_KERNEL_KT(NAME, QT_N, HD_N, BLK_N, NSG_N, DS_N, KVW_N, ##__VA_ARGS__)
#define IMPARO_QCOMB_KERNEL_KT(NAME, QT_N, HD_N, BLK_N, NSG_N, DS_N, KVW_N, ...)                       \
kernel void NAME(                                                                          \
    device const float * q     [[buffer(0)]],                                              \
    device const half  * kc    [[buffer(1)]],                                              \
    device const half  * vc    [[buffer(2)]],                                              \
    device float       * out   [[buffer(3)]],                                              \
    constant uint & hd [[buffer(4)]], constant uint & n_heads [[buffer(5)]],               \
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],        \
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],         \
    constant uint & ring_mask [[buffer(10)]], constant uint & n_tok [[buffer(11)]],        \
    constant uint & stage    [[buffer(12)]],                                               \
    device float * dspill      [[buffer(13)]],                                             \
    device const half * ktb    [[buffer(14)]],                                             \
    constant uint & kt_stride  [[buffer(15)]],                                             \
    device half * xh           [[buffer(16)]],                                             \
    constant uint & xh_on      [[buffer(17)]],                                             \
    device const uint * pt     [[buffer(18)]],                                             \
    constant uint & ptile      [[buffer(19)]],                                             \
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  lane  [[thread_index_in_simdgroup]],                                             \
    uint  sgid  [[simdgroup_index_in_threadgroup]],                                        \
    uint  nsg   [[simdgroups_per_threadgroup]])                                            \
{                                                                                          \
    attention_prefill_qcomb_body<QT_N, HD_N, BLK_N, NSG_N, DS_N, KVW_N, ##__VA_ARGS__>(            \
                               q, kc, vc, out, dspill, ktb, kt_stride, ptile,              \
                               hd, n_heads, n_kv,                                          \
                               kv_width, start_pos, window, ring_mask, pt, n_tok, stage,   \
                               xh, xh_on,                                                  \
                               shared, tgid, tid3.x, tcnt3.x, lane, sgid, nsg);            \
}

// ONE SLOT PER HEAD DIM THE MODEL USES, injected at library-compile time. No dim is named
// here: the host reads them off the plan's layers and derives each one's NSG and DSPILL
// from THIS device, exactly as it already does for the QT-16 rows' NSG and PT.
//
// The dim has to be a compile-time constant -- NDB = HD/8/NSG sizes `o[QROWS][NDB]`, a
// register array, and the Metal compiler rejects a runtime bound ("array size is not a
// constant expression"). Since the library is compiled from source on the device at every
// start, specialising for the model's own dims costs nothing and compiles nothing it
// cannot dispatch. A dim with no slot falls through to qtile, whose head dim is dynamic.
#ifndef IMPARO_HD0
#define IMPARO_HD0 0
#define IMPARO_NSG0 8
#define IMPARO_DS0 0
#endif
#ifndef IMPARO_HD1
#define IMPARO_HD1 0
#define IMPARO_NSG1 8
#define IMPARO_DS1 0
#endif
#ifndef IMPARO_HD2
#define IMPARO_HD2 0
#define IMPARO_NSG2 8
#define IMPARO_DS2 0
#endif
#ifndef IMPARO_HD3
#define IMPARO_HD3 0
#define IMPARO_NSG3 8
#define IMPARO_DS3 0
#endif

// THE K-SHARING VARIANT BELONGS TO EVERY SLOT, not to two written-down dims.
//
// QT-16 is the shape where one simdgroup carries BOTH query row groups of its position
// group, so each K fragment is loaded once and multiplied twice -- half the K traffic a
// prefill scan pulls. It existed only as `qcomb256x` and `qcomb512x`, hand-named at two
// dims, so every dim the host injects got QT-8 only and re-read K every 8 queries.
//
// That is the same defect as the dims themselves once were: a family that specialises per
// model, with one member still naming sizes. It costs most exactly where it was never
// built -- LFM2 at head_dim 64 is FULL attention (no sliding window in the file), so its
// scan is quadratic and K traffic dominates; attention is 33% of its prefill against 15%
// for E4B, where the 1.94% this shape was worth was measured.
//
// NSGX/DSX are separate from the QT-8 slot's NSG/DS because QT 16 doubles QROWS, and both
// the accumulator budget and the tail-scratch question are answered against that.
#ifndef IMPARO_BLKX0
#define IMPARO_BLKX0 2
#endif
#ifndef IMPARO_NSGX0
#define IMPARO_NSGX0 8
#endif
#ifndef IMPARO_DSX0
#define IMPARO_DSX0 0
#endif
#ifndef IMPARO_BLKX1
#define IMPARO_BLKX1 2
#endif
#ifndef IMPARO_NSGX1
#define IMPARO_NSGX1 8
#endif
#ifndef IMPARO_DSX1
#define IMPARO_DSX1 0
#endif
#ifndef IMPARO_BLKX2
#define IMPARO_BLKX2 2
#endif
#ifndef IMPARO_NSGX2
#define IMPARO_NSGX2 8
#endif
#ifndef IMPARO_DSX2
#define IMPARO_DSX2 0
#endif
#ifndef IMPARO_BLKX3
#define IMPARO_BLKX3 2
#endif
#ifndef IMPARO_NSGX3
#define IMPARO_NSGX3 8
#endif
#ifndef IMPARO_DSX3
#define IMPARO_DSX3 0
#endif

// ===========================================================================
// FA-SHAPED PREFILL ATTENTION -- llama.cpp's `kernel_flash_attn_ext` block
// structure, a NEW op beside qcomb (IMPARO_ATTN_FA=1 until it earns the default).
//
// THE BLOCK STRUCTURE IS UPSTREAM'S: QB queries per threadgroup staged once as
// half, CB keys per block, Q*K^T split over the simdgroups by column tile, scores
// through threadgroup memory (row stride 2*CB, upstream's SH), online softmax
// split by QUERY row across simdgroups in one float2 pass per lane, O in
// threadgroup memory rescaled elementwise by `ms`, P x V split by output column
// tile, three threadgroup barriers per block. 7168 B of threadgroup memory at
// HD 64 / QB 8 / CB 64 -- the same bytes as upstream's.
//
// WHAT MADE IT FAST, measured one change per build against upstream's kernel on
// identical geometry (docs/evidence/bracket/2026-09-01-fa-attention-bisection.md;
// the revived port ran 45% behind with the SAME structure):
//   - the K/V row stride is a COMPILE-TIME constant (KVW, injected per slot as
//     IMPARO_KVW<slot> beside IMPARO_HD<slot>; upstream's NS10/NS20 are function
//     constants): -10..15%. A runtime stride is a uniform read and a multiply in
//     every transposed tile load.
//   - every K tile and every V tile of a block is loaded into registers BEFORE
//     the block's MMAs, and the Q tile lives in registers for the whole key loop:
//     -15% and -6%. The compiler does not reorder simdgroup loads across MMAs on
//     its own, unrolled or not; the phase costs summed to the total before this,
//     which is what a latency-bound kernel looks like.
//   - no branch inside the block: a tile past the last key clamps its ADDRESS to
//     the last valid tile and the mask zeroes its P (-3%).
// Register budget for the tiles at HD 64, NSG 4, CB 64: Q 16 + K 32 + V 16 a lane.
// The op is therefore built for head dims up to 128 (see the instantiation).
//
// WHAT IS DIFFERENT FROM UPSTREAM, only what our interface forces:
//   - no mask TENSOR: causality and the window are synthesised from positions,
//     as qcomb does, so no `blk` precompute pass and no pad kernel either; the
//     block that crosses a query's position takes the per-position test, every
//     block below it takes the fully-live fast path
//   - no ALiBi/sinks/softcap: LFM2 and E4B use none of them
//   - Q is rounded to half before the float QK scores receive their scale
//   - K/V come from the pool through `kv_slot`; the host routes only non-ringed
//     layers here (a block's 8-row tiles must be contiguous)
//
// ATTN_SKIP probe bits (function constant 17, compiled out at 0), for attribution
// in the bench: 1 no QK MMA, 2 no softmax, 4 no V load, 8 no K load, 32 no PV MMA.
template<uint HD, uint QB, uint CB, uint NSG, uint KVW>
static void attention_prefill_fa_body(
    device const float * q, device const half * kc, device const half * vc,
    device float * out,
    uint n_heads, uint n_kv, uint kv_width,
    uint start_pos, uint window, uint ring_mask, device const uint * pt,
    uint n_tok, float scale, device const uint * layout, uint reach, uint key_lo,
    device half * xh, uint xh_on,
    threadgroup float * shared, uint3 tgid, uint tid, uint tcount,
    uint lane, uint sgid, uint nsg)
{
    (void)nsg;
    // The K/V row stride as a COMPILE-TIME constant when the host gave it (upstream's
    // NS10/NS20 are function constants for the same reason): every transposed K tile
    // load and every V tile load carries this stride, and as an immediate it costs no
    // uniform read and no multiply per tile. 0 falls back to the uniform.
    const uint kvw = (KVW != 0u) ? KVW : kv_width;
    constexpr uint HD8 = HD / 8u;
    constexpr uint NQ  = QB / NSG;         // queries each simdgroup owns in softmax
    constexpr uint NC  = (CB / 8u) / NSG;  // score column tiles per simdgroup
    constexpr uint NO  = HD8 / NSG;        // output column tiles per simdgroup
    static_assert(QB % NSG == 0u, "queries must divide over the simdgroups");
    static_assert((CB / 8u) % NSG == 0u, "score columns must divide over the simdgroups");
    static_assert(HD8 % NSG == 0u, "output columns must divide over the simdgroups");
    static_assert(CB == 64u, "the one-pass softmax owns two columns per lane: CB is 2 x 32");
    static_assert(QB == 8u, "one 8-row MMA query tile per threadgroup: mq, mqk and lo are sized for it");

    const uint h = tgid.x, q0 = tgid.y * QB;
    const uint kvh = h / (n_heads / n_kv);

    // Scores row stride CB. Upstream's is 2*CB with the mask tile in the second half; this
    // kernel synthesises its mask, so that half was 2 KB of threadgroup memory nothing read
    // -- and threadgroup memory sets how many threadgroups share a core, which is what lets
    // one threadgroup's loads overlap another's MMAs: 7168 -> 5120 B measured -2.4% at
    // depth (register-resident prefetch of the next block measured -5..-27% instead, and
    // 4096 B by aliasing the output tile onto the dead Q staging measured no further gain).
    constexpr uint SS = CB;
    // Staged Q: QB x HD halves, or QB x HD floats under FA_FLOAT_Q.
    threadgroup half  * sq  = (threadgroup half *)shared;
    threadgroup float * sqf = shared;
    threadgroup float * ss = shared + (FA_FLOAT_Q ? QB * HD : (QB * HD) / 2u); // QB x SS, scores then P
    threadgroup float * so = ss + QB * SS;                        // QB x HD, output

    for (uint e = tid; e < QB * HD; e += tcount) {
        const uint row = e / HD, dim = e % HD;
        const uint t = q0 + row;
        const float qv = t < n_tok ? q[((ulong)t * n_heads + h) * HD + dim] : 0.0f;
        if (FA_FLOAT_Q) { sqf[row * HD + dim] = qv; } else { sq[row * HD + dim] = half(qv); }
    }
    for (uint e = tid; e < QB * HD; e += tcount) { so[e] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ], M[NQ];
    #pragma unroll
    for (uint jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -INFINITY; }

    // THE QUERY TILE LIVES IN REGISTERS FOR THE WHOLE KEY LOOP. It never changes, and the
    // QB x HD half tile is HD8 8x8 fragments -- 16 registers a lane at HD 64 -- so every
    // Q*K^T step reads it from registers instead of re-loading it from threadgroup memory
    // per K tile (upstream re-loads; the port did too).
    simdgroup_half8x8 mq[HD8];
    simdgroup_float8x8 mqf[HD8];
    #pragma unroll
    for (uint i = 0; i < HD8; ++i) {
        if (FA_FLOAT_Q) { simdgroup_load(mqf[i], sqf + 8u * i, HD); }
        else { simdgroup_load(mq[i], sq + 8u * i, HD); }
    }

    // Retain the output fragments across KV blocks. The shared output area carries
    // an 8x8 broadcast of the row rescale factors during the loop; loading that as
    // a fragment aligns the factors without assuming a lane-to-matrix layout.
    simdgroup_float8x8 lo[NO];
    #pragma unroll
    for (uint ii = 0; ii < NO; ++ii) {
        lo[ii] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Both bounds come from the whole query tile: the earliest query sets where the
    // scan starts under a window, the latest sets where it ends. A ROW LAYOUT row may see
    // batch rows after its own (every row of a drafted block sees the whole block), so there
    // the scan ends `reach` rows past the tile's last row: the furthest any row of the batch
    // sees past itself, 0 for a chain or a tree. Ending at the tile's last row dropped the
    // later rows' keys for every row of an earlier tile.
    const uint last_q    = min(q0 + QB, n_tok) - 1u;
    const uint pos_last  = start_pos + (ROW_LAYOUT ? min(last_q + reach, n_tok - 1u) : last_q);
    const uint pos_first = start_pos + q0;
    // A ROW LAYOUT may carry a key floor: no row sees a key below `key_lo` (a drafter whose
    // cache holds rows only from a restore point up). The scan starts at the floor's block,
    // so blocks wholly below it are never loaded; keys of that block below the floor are
    // masked. Without a layout the floor is 0.
    const uint scan_lo   = max((window > 0u && pos_first + 1u > window)
                                   ? (pos_first + 1u - window) : 0u,
                               key_lo & ~(CB - 1u));

    // ONE PAGE PER BLOCK, FETCHED A BLOCK AHEAD. kv_slot reads pt[gp / 64] for every 8-key
    // tile, and with the tile loads hoisted below that dependent load stands in front of all
    // of them: compiled out (KV_PAGED false, identity placement) the kernel is 8% faster at
    // every depth, while a scattered table costs nothing over an identity one -- the cost is
    // the per-tile lookup, not the placement. A 64-key block is exactly one 64-cell page
    // when its start is page-aligned (the FA route has no ring, and scan_lo is 0 without a
    // window), so the block's page is one uniform load, issued during the PREVIOUS block so
    // no tile load ever waits on it. An unaligned start (a window) keeps kv_slot per tile.
    static_assert(CB == KV_PAGE_CELLS, "a block is one page: the page fetch below assumes it");
    const bool paged_blocks = KV_PAGED && ring_mask == 0u;
    uint pg_next = paged_blocks ? pt[scan_lo / KV_PAGE_CELLS] : 0u;

    for (uint ic = scan_lo; ic <= pos_last; ic += CB) {
        const uint pg_cur  = pg_next;
        const uint ic_next = ic + CB;
        if (paged_blocks && ic_next <= pos_last) { pg_next = pt[ic_next / KV_PAGE_CELLS]; }
        const bool one_page = paged_blocks && (ic % KV_PAGE_CELLS) == 0u;
        auto tile_slot = [&](uint gp) -> uint {   // gp >= ic, inside this block
            return one_page ? pg_cur * KV_PAGE_CELLS + (gp - ic) : kv_slot(gp, ring_mask, pt);
        };

        // NO BOUNDS BRANCH INSIDE THE BLOCK. A tile past the last key CLAMPS its address to
        // the last valid tile instead of skipping: the load hits a line already fetched,
        // and the mask pass below sets every column past the query's position to -INF, so
        // P is 0 there and the duplicate contributes nothing -- the same numbers as the
        // skip. What it buys is the loop shape upstream has by padding K/V: every load in
        // the block can be issued ahead of every MMA, with no data-dependent branch or
        // early exit between them.
        const uint last_tile = pos_last & ~7u;

        // ---- Q * K^T, NC 8-position column tiles per simdgroup per block ----
        // EVERY K TILE OF THE BLOCK IS LOADED BEFORE THE FIRST MMA (as the V tiles below):
        // NC x HD8 transposed fragments, 32 registers a lane at HD 64 with NC 2.
        // ATTN_SKIP probes (function constant, compiled out at 0): bit 8 drops the K load
        // and keeps the MMA; bit 1 drops the MMA and with it the load.
        simdgroup_half8x8 mkb[NC][HD8];
        #pragma unroll
        for (uint cc = 0; cc < NC; ++cc) {
            const uint gp0 = min(ic + (cc * NSG + sgid) * 8u, last_tile);
            const uint ps0 = tile_slot(gp0);
            device const half * pk = kc + (ulong)ps0 * kvw + kvh * HD;
            #pragma unroll
            for (uint i = 0; i < HD8; ++i) {
                if (ATTN_SKIP & 8u) { mkb[cc][i] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.01f)); }
                else { simdgroup_load(mkb[cc][i], pk + 8u * i, kvw, 0, true); }
            }
        }
        #pragma unroll
        for (uint cc = 0; cc < NC; ++cc) {
            const uint col = (cc * NSG + sgid) * 8u;
            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint i = 0; i < HD8; ++i) {
                if (ATTN_SKIP & 1u) { continue; }
                if (FA_FLOAT_Q) { simdgroup_multiply_accumulate(mqk, mqf[i], mkb[cc][i], mqk); }
                else { simdgroup_multiply_accumulate(mqk, mq[i], mkb[cc][i], mqk); }
            }
            simdgroup_store(mqk, ss + col, SS, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- online softmax: each simdgroup owns NQ whole query rows ----
        // The masking is applied before the max so a masked position can never set it.
        #pragma unroll
        for (uint jj = 0; jj < NQ; ++jj) {
            if (ATTN_SKIP & 2u) { break; }   // probe: no softmax, P = raw scores
            const uint j = jj * NSG + sgid;
            const uint t = q0 + j;
            const uint t_pos = start_pos + t;
            const uint t_lo = (window > 0u && t_pos + 1u > window)
                            ? (t_pos + 1u - window) : 0u;
            const bool row_live = t < n_tok;
            const float m_prev = M[jj];

            // FULLY-LIVE FAST PATH, the same idea as qcomb's ATTN_LIVE_MASK and upstream's
            // blk_cur == 2: when the whole block is live for this query, the per-position
            // causal test is 2 comparisons per position that can never fire. At depth
            // almost every block is far below the diagonal and takes this path.
            // A ROW LAYOUT sees the whole cache below the batch and the batch rows its mask
            // names, so only a block wholly below the batch is live for every row.
            const bool blk_live = ROW_LAYOUT
                ? (row_live && (ic + CB) <= start_pos && ic >= key_lo)
                : (row_live && (ic + CB - 1u) <= t_pos && ic >= t_lo);
            // ONE PASS, upstream's form: a lane owns columns 2*lane and 2*lane+1 as a
            // float2, so the block max is one simd_max and every score is read once and
            // written once. CB is 64 by construction (static_assert above).
            threadgroup float2 * ss2 = (threadgroup float2 *)(ss + j * SS);
            float2 s2 = ss2[lane] * scale;
            if (!blk_live) {
                const uint gp = ic + 2u * lane;
                if (ROW_LAYOUT) {
                    s2.x = (row_live && gp >= key_lo
                            && row_layout_sees(layout, t, gp, start_pos))      ? s2.x : -INFINITY;
                    s2.y = (row_live && gp + 1u >= key_lo
                            && row_layout_sees(layout, t, gp + 1u, start_pos)) ? s2.y : -INFINITY;
                } else {
                    s2.x = (row_live && gp      <= t_pos && gp      >= t_lo) ? s2.x : -INFINITY;
                    s2.y = (row_live && gp + 1u <= t_pos && gp + 1u >= t_lo) ? s2.y : -INFINITY;
                }
            }
            M[jj] = simd_max(max(M[jj], max(s2.x, s2.y)));
            // A row with nothing live yet would make exp(-INF - -INF) a NaN; zero it
            // instead, exactly as qcomb does for rows past the batch.
            const bool dead = M[jj] == -INFINITY;
            const float  ms  = dead ? 0.0f : exp(m_prev - M[jj]);
            const float2 vs2 = dead ? float2(0.0f) : exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2.x + vs2.y);
            ss2[lane] = vs2;
            if (lane < 8u) { so[j * 8u + lane] = ms; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O += P * V, one 8-dim output tile per simdgroup per step ----
        {
            // The skip-softmax probe leaves scores raw and has no row rescale.
            // Keep its accumulated output instead of multiplying it by unwritten data.
            if (!(ATTN_SKIP & 2u)) {
                simdgroup_float8x8 rescale;
                simdgroup_load(rescale, so, 8u);
                #pragma unroll
                for (uint ii = 0; ii < NO; ++ii) {
                    lo[ii].thread_elements() *= rescale.thread_elements();
                }
            }
            // EVERY V TILE OF THE BLOCK IS LOADED BEFORE THE FIRST MMA. The loads do not
            // depend on P, so issuing all CB/8 x NO of them first puts the block's whole V
            // latency in flight at once instead of one tile's worth per MMA pair; the
            // attribution read the V load at 3x the K load for the same bytes.
            simdgroup_half8x8 mvb[CB / 8u][NO];
            #pragma unroll
            for (uint cc = 0; cc < CB / 8u; ++cc) {
                const uint gp0 = min(ic + cc * 8u, last_tile);   // clamp, see above
                const uint ps0 = tile_slot(gp0);
                device const half * pv = vc + (ulong)ps0 * kvw + kvh * HD;
                #pragma unroll
                for (uint ii = 0; ii < NO; ++ii) {
                    // ATTN_SKIP bit 4 drops the V load (MMA kept).
                    if (ATTN_SKIP & 4u) { mvb[cc][ii] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.01f)); }
                    else { simdgroup_load(mvb[cc][ii], pv + 8u * (sgid + ii * NSG), kvw); }
                }
            }
            #pragma unroll
            for (uint cc = 0; cc < CB / 8u; ++cc) {
                simdgroup_float8x8 vs;
                simdgroup_load(vs, ss + 8u * cc, SS);
                #pragma unroll
                for (uint ii = 0; ii < NO; ++ii) {
                    // ATTN_SKIP bit 32 drops the MMA.
                    if (!(ATTN_SKIP & 32u)) { simdgroup_multiply_accumulate(lo[ii], vs, mvb[cc][ii], lo[ii]); }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma unroll
    for (uint ii = 0; ii < NO; ++ii) {
        simdgroup_store(lo[ii], so + 8u * (sgid + ii * NSG), HD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each query row is owned by exactly one simdgroup (j % NSG == sgid), and
    // simd_max/simd_sum already broadcast S across its lanes.
    #pragma unroll
    for (uint jj = 0; jj < NQ; ++jj) {
        const uint j = jj * NSG + sgid;
        const uint t = q0 + j;
        if (t >= n_tok) { continue; }
        const float inv = S[jj] > 0.0f ? 1.0f / S[jj] : 0.0f;
        for (uint d = lane; d < HD; d += 32u) {
            const ulong idx = ((ulong)t * n_heads + h) * HD + d;
            const float v = so[j * HD + d] * inv;
            out[idx] = v;
            if (xh_on != 0u) { xh[idx] = half(v); }
        }
    }
}

#define IMPARO_FA_KERNEL(NAME, HD_N, QB_N, CB_N, NSG_N, KVW_N)                                    \
kernel void NAME(                                                                          \
    device const float * q      [[buffer(0)]],                                             \
    device const half  * kc     [[buffer(1)]],                                             \
    device const half  * vc     [[buffer(2)]],                                             \
    device float       * out    [[buffer(3)]],                                             \
    constant uint & n_heads     [[buffer(4)]],                                             \
    constant uint & n_kv        [[buffer(5)]],                                             \
    constant uint & kv_width    [[buffer(6)]],                                             \
    constant uint & start_pos   [[buffer(7)]],                                             \
    constant uint & window      [[buffer(8)]],                                             \
    constant uint & ring_mask   [[buffer(9)]],                                             \
    constant uint & n_tok       [[buffer(10)]],                                            \
    device half        * xh     [[buffer(11)]],                                            \
    constant uint & xh_on       [[buffer(12)]],                                            \
    device const uint  * pt     [[buffer(13)]],                                            \
    constant float & scale      [[buffer(14)]],                                            \
    device const uint  * layout [[buffer(15), function_constant(ROW_LAYOUT)]],             \
    constant uint & row_reach   [[buffer(16), function_constant(ROW_LAYOUT)]],             \
    constant uint & row_key_lo  [[buffer(17), function_constant(ROW_LAYOUT)]],             \
    threadgroup float * shared  [[threadgroup(0)]],                                        \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  lane  [[thread_index_in_simdgroup]],                                             \
    uint  sgid  [[simdgroup_index_in_threadgroup]],                                        \
    uint  nsg   [[simdgroups_per_threadgroup]])                                            \
{                                                                                          \
    device const uint * lay = nullptr;                                                     \
    uint reach = 0u, key_lo = 0u;                                                          \
    if (ROW_LAYOUT) { lay = layout; reach = row_reach; key_lo = row_key_lo; }              \
    attention_prefill_fa_body<HD_N, QB_N, CB_N, NSG_N, KVW_N>(                                    \
        q, kc, vc, out, n_heads, n_kv, kv_width, start_pos, window, ring_mask, pt,         \
        n_tok, scale, lay, reach, key_lo, xh, xh_on, shared, tgid, tid3.x, tcnt3.x, lane,  \
        sgid, nsg);                                                                        \
}

#ifndef IMPARO_KVW0
#define IMPARO_KVW0 0u
#endif
#ifndef IMPARO_KVW1
#define IMPARO_KVW1 0u
#endif
#ifndef IMPARO_KVW2
#define IMPARO_KVW2 0u
#endif
#ifndef IMPARO_KVW3
#define IMPARO_KVW3 0u
#endif
// The FA op holds a block's K and V tiles in registers ahead of the MMAs (NC x HD/8 and
// CB/8 x HD/8/NSG fragments) plus the Q tile: 64 registers a lane at HD 64. At HD 256 that
// is four times over the file, so the kernel is built for head dims up to 128 only; the
// host finds no pipeline for a wider slot and qcomb serves it, as before.
#ifndef IMPARO_FA_NSG0
#define IMPARO_FA_NSG0 4
#endif
#ifndef IMPARO_FA_ALL
#define IMPARO_FA_ALL 0
#endif
// NSG is the op's knob (attn_fa_nsg). It sizes register arrays, so each value is its own
// kernel: the engine builds the seated one (IMPARO_FA_NSG0), the tuner builds every legal
// one (IMPARO_FA_ALL, its prepare step) so a sweep can select them at dispatch -- a knob
// that only exists at library compile is inert to a sweep that runs after init. QB 8 is
// structural: one 8-row MMA query tile per threadgroup (a two-tile body measured 70%
// slower at 16). CB 64 is one page and the one-pass softmax's 2 x 32 lanes.

// PREFILL ATTENTION WITH THE SOFTMAX IN REGISTERS, the default at head dims up to 128
// (IMPARO_ATTN_FARS=0 keeps the 8-query op above): MLX's steel-attention structure. 32 queries a threadgroup, 8 rows a
// simdgroup; each 32-key block of K and V staged once in threadgroup memory (rows padded by 8
// halves, so a fragment load's 8 rows fall in different banks) for all four simdgroups; the
// scores and probabilities stay in the simdgroup's fragments, each lane's row reduced with two
// shuffles (a row's elements live on lanes l, l^1, l^8, l^9); two barriers a block. The 8-query
// op stages S, P and a rescale broadcast through threadgroup memory with three barriers a
// block and re-reads every K and V tile once per 8 queries.
//
// NOT the 8-query op's arithmetic: blocks of 32 keys, exp2 on prescaled scores, the row sum
// by shuffles -- a numerics change, gated as one.
// ROW LAYOUT (constant 25) as the 8-query op reads it: a tree verify's rows are masked by the
// layout, not by position. For a chain the layout selects exactly the keys the causal test does,
// so a chain as a row layout equals the causal forward over the same rows, bit for bit
// (`imparo-forward --verify-tree`), unless ROW_SPLIT cuts the keys (it reassociates the softmax).
// ROW_HEADS keeps every row's arithmetic.
// Execution-only simdgroup fences around each fragment load (MLX steel attention's scheduling
// hint; no arithmetic changes): attention 1428 / 1425 -> 1417 / 1413 ms at a 17115-token
// prefill.
// The P.V loop takes a full unroll, not the "#pragma unroll" hint: attention 1406 -> 1368 ms at
// a 17115-token prefill (LFM2.5-8B-A1B), same bits. Full unrolls on the kernel's other loops
// tie, and so do 16 query rows a simdgroup (1394 ms, same bits; 2.05x slower under the hint).
#define FARS_UNROLL _Pragma("clang loop unroll(full)")
template<uint HD, uint KVW>
static void attention_prefill_fars_body(
    device const float * q, device const half * kc, device const half * vc,
    device float * out,
    uint n_heads, uint n_kv, uint kv_width,
    uint start_pos, uint window, uint ring_mask, device const uint * pt,
    uint n_tok, float scale, device half * xh, uint xh_on,
    device const uint * layout, uint reach, uint key_lo,
    device float * part, uint slices, uint chunk,
    threadgroup half * shared, uint3 tgid, uint tid, uint lane, uint sgid)
{
    constexpr uint QB = 32u, KB = 32u, THREADS = 128u;
    constexpr uint HD8 = HD / 8u, KS = HD + 8u;
    // Query rows a threadgroup: 32, or 8 under ROW_HEADS (four heads' 8 rows fill the 32).
    const uint QT = ROW_HEADS ? 8u : QB;
    const uint kvw = (KVW != 0u) ? KVW : kv_width;
    const uint grp = n_heads / n_kv;
    const uint h = ROW_HEADS ? tgid.x * grp + sgid : tgid.x;
    const uint q0 = tgid.y * QT;
    const uint kvh = ROW_HEADS ? tgid.x : h / grp;
    threadgroup half * tk = shared;
    threadgroup half * tv = shared + KB * KS;
    // Staged 8 halves (16 bytes) a load: NF of K and NF of V a thread.
    constexpr uint NF = (KB * (HD / 8u)) / THREADS;
    static_assert((KB * (HD / 8u)) % THREADS == 0u, "the staging divides over the threads");

    // Q through the K/V staging: QB x KS halves fit in the two blocks' 2 x KB x KS.
    for (uint e = tid; e < QB * HD; e += THREADS) {
        const uint row = e / HD, dim = e % HD;
        const uint t = ROW_HEADS ? q0 + row % 8u : q0 + row;
        const uint hq = ROW_HEADS ? tgid.x * grp + row / 8u : h;
        shared[row * KS + dim] = half(t < n_tok ? q[((ulong)t * n_heads + hq) * HD + dim] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 mq[HD8];
    #pragma unroll
    for (uint i = 0; i < HD8; ++i) { simdgroup_load(mq[i], shared + (8u * sgid) * KS + 8u * i, KS); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This lane's place in every 8x8 fragment: row fm, columns fn and fn + 1 (MLX steel mma.h).
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint t = ROW_HEADS ? q0 + fm : q0 + 8u * sgid + fm;
    const bool row_live = t < n_tok;
    const uint t_pos = start_pos + t;
    const uint t_lo = (window > 0u && t_pos + 1u > window) ? (t_pos + 1u - window) : 0u;
    const float sl2 = scale * M_LOG2E_F;

    simdgroup_float8x8 lo[HD8];
    #pragma unroll
    for (uint ii = 0; ii < HD8; ++ii) { lo[ii] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
    float m = -INFINITY, l = 0.0f;

    const uint last_q    = min(q0 + QT, n_tok) - 1u;
    // A layout row may see batch rows past its own (a drafted block): the scan ends `reach` rows
    // past the tile's last row, and starts at the key floor's block.
    const uint pos_last  = start_pos + (ROW_LAYOUT ? min(last_q + reach, n_tok - 1u) : last_q);
    const uint pos_first = start_pos + q0;
    const uint scan_lo = max((window > 0u && pos_first + 1u > window) ? (pos_first + 1u - window) : 0u,
                             ROW_LAYOUT ? key_lo : 0u)
                       & ~(KB - 1u);
    const bool paged = KV_PAGED && ring_mask == 0u;

    // A 32-aligned block lies inside one 64-cell page. Keys past the last position are staged
    // as zeros (their probabilities are 0; garbage V could be NaN).
    float4 pk[NF], pv[NF];   // 8 halves each, moved as raw 16-byte words
    auto fetch = [&](uint kbn) {
        const uint pgbase = paged ? pt[kbn / KV_PAGE_CELLS] * KV_PAGE_CELLS + (kbn % KV_PAGE_CELLS) : 0u;
        // A block wholly at or below the last position needs no per-row test.
        const bool whole = kbn + KB - 1u <= pos_last;
        #pragma unroll
        for (uint f = 0; f < NF; ++f) {
            const uint e = tid + f * THREADS;
            const uint key = e / (HD / 8u), d8 = (e % (HD / 8u)) * 8u;
            const uint gp = kbn + key;
            pk[f] = float4(0.0f); pv[f] = float4(0.0f);
            if (whole || gp <= pos_last) {
                const uint slot = paged ? pgbase + key : kv_slot(gp, ring_mask, pt);
                const ulong off = (ulong)slot * kvw + kvh * HD + d8;
                pk[f] = *(device const float4 *)(kc + off);
                pv[f] = *(device const float4 *)(vc + off);
            }
        }
    };
    auto store = [&](threadgroup half * dk, threadgroup half * dv) {
        #pragma unroll
        for (uint f = 0; f < NF; ++f) {
            const uint e = tid + f * THREADS;
            const uint key = e / (HD / 8u), d8 = (e % (HD / 8u)) * 8u;
            *(threadgroup float4 *)(dk + key * KS + d8) = pk[f];
            *(threadgroup float4 *)(dv + key * KS + d8) = pv[f];
        }
    };
    // ROW_SPLIT: this threadgroup's slice of the scan, `chunk` keys (a multiple of KB) from the
    // floor's block.
    const uint kb_lo = ROW_SPLIT ? scan_lo + tgid.z * chunk : scan_lo;
    const uint kb_hi = ROW_SPLIT ? min(pos_last, kb_lo + chunk - 1u) : pos_last;
    for (uint kb = kb_lo; kb <= kb_hi; kb += KB) {
        // Double-buffering this staging (the next block's rows in registers across this
        // block's multiplies) measured 52.6 against 51.6 ms at 16384 keys; 64-key blocks 56.5;
        // 64 queries a threadgroup 52.1; unpadded rows a tie.
        // V fetched after the scores instead (MLX's order) measured 1444 against 1428 ms
        // of attention at a 17115-token prefill, same bits.
        fetch(kb);
        store(tk, tv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half * bk = tk;
        threadgroup half * bv = tv;

        simdgroup_float8x8 s[KB / 8u];
        // Loading the block's K fragments ahead of the multiplies (or its V fragments ahead of
        // the softmax) measured 2-5x SLOWER: 32 more fragments a lane spill.
        #pragma unroll
        for (uint c = 0; c < KB / 8u; ++c) {
            s[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint i = 0; i < HD8; ++i) {
                simdgroup_half8x8 kf;
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(kf, bk + (8u * c) * KS + 8u * i, KS, 0, true);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(s[c], mq[i], kf, s[c]);
            }
        }

        // A layout row sees the whole cache below the batch, so only a block wholly below it
        // (and above the floor) is live for every row.
        const bool blk_live = ROW_LAYOUT
            ? (row_live && (kb + KB) <= start_pos && kb >= key_lo)
            : (row_live && (kb + KB - 1u) <= t_pos && kb >= t_lo);
        float2 sv[KB / 8u];
        float mx = -INFINITY;
        #pragma unroll
        for (uint c = 0; c < KB / 8u; ++c) {
            float2 e2 = rm_frag_get(s[c]) * sl2;
            if (!blk_live) {
                const uint gp = kb + 8u * c + fn;
                if (ROW_LAYOUT) {
                    e2.x = (row_live && gp >= key_lo
                            && row_layout_sees(layout, t, gp, start_pos))      ? e2.x : -INFINITY;
                    e2.y = (row_live && gp + 1u >= key_lo
                            && row_layout_sees(layout, t, gp + 1u, start_pos)) ? e2.y : -INFINITY;
                } else {
                    e2.x = (row_live && gp      <= t_pos && gp      >= t_lo) ? e2.x : -INFINITY;
                    e2.y = (row_live && gp + 1u <= t_pos && gp + 1u >= t_lo) ? e2.y : -INFINITY;
                }
            }
            sv[c] = e2;
            mx = max(mx, max(e2.x, e2.y));
        }
        mx = max(mx, simd_shuffle_xor(mx, 1u));
        mx = max(mx, simd_shuffle_xor(mx, 8u));
        const float m_new = max(m, mx);
        const bool dead = m_new == -INFINITY;
        const float fac = dead ? 0.0f : exp2(m - m_new);
        float rs = 0.0f;
        #pragma unroll
        for (uint c = 0; c < KB / 8u; ++c) {
            const float2 p = dead ? float2(0.0f) : exp2(sv[c] - m_new);
            rm_frag_set(s[c], p.x, p.y);
            rs += p.x + p.y;
        }
        rs += simd_shuffle_xor(rs, 1u);
        rs += simd_shuffle_xor(rs, 8u);
        l = l * fac + rs;
        m = m_new;
        #pragma unroll
        for (uint ii = 0; ii < HD8; ++ii) {
            const float2 o = rm_frag_get(lo[ii]) * fac;
            rm_frag_set(lo[ii], o.x, o.y);
        }
        FARS_UNROLL
        for (uint c = 0; c < KB / 8u; ++c) {
            FARS_UNROLL
            for (uint ii = 0; ii < HD8; ++ii) {
                simdgroup_half8x8 vf;
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(vf, bv + (8u * c) * KS + 8u * ii, KS);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(lo[ii], s[c], vf, lo[ii]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before the next block's staging
    }

    if (ROW_SPLIT) {
        // Unnormalised: the combine rescales each slice by its max (log2 units of the prescaled
        // score) against the row's overall max. A slice with no key the row sees writes m = -inf,
        // l = 0 and a zero output, which the combine weighs by 0.
        if (row_live) {
            device float * pp = part + (((ulong)t * n_heads + h) * slices + tgid.z) * (HD + 2u);
            #pragma unroll
            for (uint ii = 0; ii < HD8; ++ii) {
                *(device float2 *)(pp + 8u * ii + fn) = rm_frag_get(lo[ii]);
            }
            if (fn == 0u) { pp[HD] = m; pp[HD + 1u] = l; }
        }
        return;
    }
    if (row_live) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        #pragma unroll
        for (uint ii = 0; ii < HD8; ++ii) {
            const float2 o = rm_frag_get(lo[ii]) * inv;
            const ulong idx = ((ulong)t * n_heads + h) * HD + 8u * ii + fn;
            *(device float2 *)(out + idx) = o;
            if (xh_on != 0u) { *(device half2 *)(xh + idx) = half2(o); }
        }
    }
}

// ROW_SPLIT's merge: one threadgroup a (head, row), a thread a dim. out = sum_s o_s w_s / sum_s l_s w_s
// with w_s = exp2(m_s - max_s m_s); the half mirror too when the op would have written it.
kernel void imparo_attention_rows_combine(
    device const float * part [[buffer(0)]], device float * out [[buffer(1)]],
    constant uint & head_dim  [[buffer(2)]], constant uint & n_heads [[buffer(3)]],
    constant uint & slices    [[buffer(4)]], device half * xh [[buffer(5)]],
    constant uint & xh_on     [[buffer(6)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]])
{
    const uint h = tgid.x, t = tgid.y, stride = head_dim + 2u;
    device const float * pb = part + ((ulong)t * n_heads + h) * slices * stride;
    float mx = -INFINITY;
    for (uint s = 0; s < slices; ++s) { mx = max(mx, pb[s * stride + head_dim]); }
    float den = 0.0f;
    if (mx != -INFINITY) {
        for (uint s = 0; s < slices; ++s) {
            den += exp2(pb[s * stride + head_dim] - mx) * pb[s * stride + head_dim + 1u];
        }
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    const ulong o0 = ((ulong)t * n_heads + h) * head_dim;
    for (uint i = tid3.x; i < head_dim; i += tcnt3.x) {
        float acc = 0.0f;
        if (mx != -INFINITY) {
            for (uint s = 0; s < slices; ++s) {
                acc += exp2(pb[s * stride + head_dim] - mx) * pb[s * stride + i];
            }
        }
        const float o = acc * inv;
        out[o0 + i] = o;
        if (xh_on != 0u) { xh[o0 + i] = half(o); }
    }
}

#define IMPARO_FARS_KERNEL(NAME, HD_N, KVW_N)                                                 \
kernel void NAME(                                                                          \
    device const float * q      [[buffer(0)]],                                             \
    device const half  * kc     [[buffer(1)]],                                             \
    device const half  * vc     [[buffer(2)]],                                             \
    device float       * out    [[buffer(3)]],                                             \
    constant uint & n_heads     [[buffer(4)]],                                             \
    constant uint & n_kv        [[buffer(5)]],                                             \
    constant uint & kv_width    [[buffer(6)]],                                             \
    constant uint & start_pos   [[buffer(7)]],                                             \
    constant uint & window      [[buffer(8)]],                                             \
    constant uint & ring_mask   [[buffer(9)]],                                             \
    constant uint & n_tok       [[buffer(10)]],                                            \
    device half        * xh     [[buffer(11)]],                                            \
    constant uint & xh_on       [[buffer(12)]],                                            \
    device const uint  * pt     [[buffer(13)]],                                            \
    constant float & scale      [[buffer(14)]],                                            \
    device const uint  * layout [[buffer(15), function_constant(ROW_LAYOUT)]],             \
    constant uint & row_reach   [[buffer(16), function_constant(ROW_LAYOUT)]],             \
    constant uint & row_key_lo  [[buffer(17), function_constant(ROW_LAYOUT)]],             \
    device float       * part   [[buffer(18), function_constant(ROW_SPLIT)]],              \
    constant uint & row_chunk   [[buffer(19), function_constant(ROW_SPLIT)]],              \
    threadgroup half * shared   [[threadgroup(0)]],                                        \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tpg   [[threadgroups_per_grid]],                                                 \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint  lane  [[thread_index_in_simdgroup]],                                             \
    uint  sgid  [[simdgroup_index_in_threadgroup]])                                        \
{                                                                                          \
    device const uint * lay = nullptr;                                                     \
    uint reach = 0u, key_lo = 0u;                                                          \
    if (ROW_LAYOUT) { lay = layout; reach = row_reach; key_lo = row_key_lo; }              \
    device float * pp = nullptr;                                                           \
    uint chunk = 0u;                                                                       \
    if (ROW_SPLIT) { pp = part; chunk = row_chunk; }                                       \
    attention_prefill_fars_body<HD_N, KVW_N>(q, kc, vc, out, n_heads, n_kv, kv_width,       \
        start_pos, window, ring_mask, pt, n_tok, scale, xh, xh_on, lay, reach, key_lo,     \
        pp, tpg.z, chunk, shared, tgid, tid3.x, lane, sgid);                               \
}

#if IMPARO_HD0 && (IMPARO_HD0 <= 128)
IMPARO_FARS_KERNEL(imparo_attention_prefill_fars_s0, IMPARO_HD0, IMPARO_KVW0)
#endif
#if IMPARO_HD0 && (IMPARO_HD0 <= 128)
#if IMPARO_FA_ALL || (IMPARO_FA_NSG0 == 1)
IMPARO_FA_KERNEL(imparo_attention_prefill_fa_s0_n1, IMPARO_HD0, 8, 64, 1, IMPARO_KVW0)
#endif
#if IMPARO_FA_ALL || (IMPARO_FA_NSG0 == 2)
IMPARO_FA_KERNEL(imparo_attention_prefill_fa_s0_n2, IMPARO_HD0, 8, 64, 2, IMPARO_KVW0)
#endif
#if IMPARO_FA_ALL || (IMPARO_FA_NSG0 == 4)
IMPARO_FA_KERNEL(imparo_attention_prefill_fa_s0_n4, IMPARO_HD0, 8, 64, 4, IMPARO_KVW0)
#endif
#if IMPARO_FA_ALL || (IMPARO_FA_NSG0 == 8)
IMPARO_FA_KERNEL(imparo_attention_prefill_fa_s0_n8, IMPARO_HD0, 8, 64, 8, IMPARO_KVW0)
#endif
#endif

#define IMPARO_QCOMB_SLOT(I, HD, NSG, DS, BLKX, NSGX, DSX)                                 \
    IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb_s##I,   8, HD, 4, NSG, DS, IMPARO_KVW##I)   \
    IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb_s##I##b2, 8, HD, 2, NSG, DS, IMPARO_KVW##I) \
    IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb_s##I##x, 16, HD, BLKX, NSGX, DSX, IMPARO_KVW##I, true)

#if IMPARO_HD0
IMPARO_QCOMB_SLOT(0, IMPARO_HD0, IMPARO_NSG0, IMPARO_DS0, IMPARO_BLKX0, IMPARO_NSGX0, IMPARO_DSX0)
#endif
#if IMPARO_HD1
IMPARO_QCOMB_SLOT(1, IMPARO_HD1, IMPARO_NSG1, IMPARO_DS1, IMPARO_BLKX1, IMPARO_NSGX1, IMPARO_DSX1)
#endif
#if IMPARO_HD2
IMPARO_QCOMB_SLOT(2, IMPARO_HD2, IMPARO_NSG2, IMPARO_DS2, IMPARO_BLKX2, IMPARO_NSGX2, IMPARO_DSX2)
#endif
#if IMPARO_HD3
IMPARO_QCOMB_SLOT(3, IMPARO_HD3, IMPARO_NSG3, IMPARO_DS3, IMPARO_BLKX3, IMPARO_NSGX3, IMPARO_DSX3)
#endif

// BLK 2 variants of the QT-8 kernels. Their score phase splits a PT-128 position tile into
// ceil(PT / (8*BLK)) work units and hands one to each simdgroup, so at BLK 4 that is 4
// units over NSG 8 -- HALF THE SIMDGROUPS IDLE. At BLK 2 it is 8 units over 8. The trade is
// arithmetic intensity: (QROWS + BLK) / (QROWS * BLK) with QROWS 1 goes from 1.25 loads per
// MAC to 1.50, against double the parallelism. BLK also does not touch threadgroup memory
// (only QT and PT do) and needs FEWER accumulator registers, so the wider occupancy costs
// nothing structural. These are the kernels q4 and q8 prefill run on.
// The BLK-2 shape at hd 64, so the blk axis exists at every dim the family serves rather
// than only where someone happened to write it.

// The same K-sharing shape for the WINDOW layers (35 of 42). Head size 256
// makes NDB = 256/8/8 = 4, so two row groups fit at NSG 8 without the extra
// simdgroups the hd-512 variant needs.
//
// PT 128 and THREADGROUP spill, not device. A windowed scan is window + QT - 1
// = 527 positions, which is not a multiple of 8, so the tail path fires for
// every query tile here -- unlike the full-attention layers, whose scan length
// is always a multiple of 16 and which therefore never reach it. The device
// spill buffer is indexed by (head, tile parity), which holds two tiles per
// head; sending 32 tiles through it raced and went nondeterministic at n=2000
// and beyond. Threadgroup spill is per-threadgroup and cannot collide, and it
// fits here: half Q 8 KB + scores 16*128*4 = 8 KB + spill 16*256*4 = 16 KB.
// PT 112, not 128. The layout is half Q 8192 + scores 16*PT*4 + max/sum 128 +
// diagonals 512 + the threadgroup tail spill 16*256*4 = 16384, and at PT 128 that
// totals 33408 bytes against this device's 32768 limit -- 640 over. It ran anyway,
// because Apple cores carry 64 KB of threadgroup memory and tolerated a request past
// the documented per-threadgroup limit, so the overflow never announced itself. A
// device that enforces the limit would have failed the dispatch. PT 112 is the widest
// tile that fits (118 rounded down to a multiple of 8*BLK), costing one work unit of
// occupancy: 7 over 8 simdgroups instead of 8.
// These layers keep THREADGROUP spill rather than the device scratch the hd-512 layers
// use: their scan is window + QT - 1 = 527 positions, not a multiple of 8, so their
// tail fires on every query tile, and the device scratch holds only two slots per head.
#ifndef IMPARO_PT_256X
#define IMPARO_PT_256X 112
#endif
#ifndef IMPARO_NSG_256X
#define IMPARO_NSG_256X 8
#endif
// BLK -- position blocks per work unit. A literal 2 here while the QT-8 kernels take it
// from a swept knob, so injected like PT and NSG, which are both DERIVED FROM IT.
#ifndef IMPARO_BLK_256X
#define IMPARO_BLK_256X 2
#endif
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb256x, 16, 256, IMPARO_BLK_256X, IMPARO_NSG_256X, false, 0u, true)
// QT 16 was tried at head size 256 (the WINDOW layers, 35 of 42, whose
// accumulator is half the hd-512 one so the wider tile fits): 32472 ms against
// 32672 on a 16k prefill, 0.6%. It stages Q as half to fit the budget, so it
// moves the logits -- not worth regenerating pins and giving up numeric ground
// for 0.6%.
// head_dim 512 (the full-attention layers): the tail scratch moves to device memory.
// K-SHARING configuration (IMPARO_QCOMB_X=1). Now that one simdgroup carries
// every row group in its position group, QT 16 finally halves K traffic
// instead of merely re-loading it twice. Every term is forced:
//   NSG 16   the output accumulator is o[QROWS][NDB], NDB = HD/8/NSG, so 16
//            simdgroups make two row groups cost what one costs at 8 (it
//            measured 14% SLOWER at NSG 8, where 16 fragments spill)
//   BLK 2    unit count is PT/(8*BLK) and must reach NSG for occupancy
//   PT 240   the largest tile that fits: half-staged Q 16384 B + scores
//            16*240*4 = 15360 + rmax/rsum 128 + diag 512 = 32384 of 32768.
//            PT 256 overflows by 640 bytes.
// PT comes from the HOST, derived from this device's queried threadgroup limit and
// injected as a preprocessor define at library-compile time. The fallback is only for a
// source-level compile outside the engine.
#ifndef IMPARO_QCOMB_UNROLL
#define IMPARO_QCOMB_UNROLL 64
#endif
#ifndef IMPARO_PT_512X
#define IMPARO_PT_512X 240
#endif
#ifndef IMPARO_NSG_512X
#define IMPARO_NSG_512X 16
#endif
// BLK -- position blocks per work unit. A literal 2 here while the QT-8 kernels take it
// from a swept knob, so injected like PT and NSG, which are both DERIVED FROM IT.
#ifndef IMPARO_BLK_512X
#define IMPARO_BLK_512X 2
#endif
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb512x, 16, 512, IMPARO_BLK_512X, IMPARO_NSG_512X, true, 0u, true)
// The coupled configuration of task #37 was built and measured here (QT 16,
// NSG 16, PT 224, half-staged Q): 32860 ms against 32658 on a 16k prefill,
// 0.6% SLOWER. The reason is above, at the work-unit definition: a unit is
// (row group, position group), so different simdgroups take different row
// groups and EACH RE-LOADS K. QT 16 has never improved the load-to-multiply
// ratio -- it only adds parallelism. Sharing K across row groups needs one
// simdgroup to own both, and that has no solution in this budget: sharing
// makes the unit count pgroups rather than QROWS*pgroups, so NSG 16 wants
// PT >= 512 to stay busy, and scores alone are then 16*512*4 = 32 KB, the
// entire threadgroup allowance. NSG 8 fits the tile but spills the output
// accumulator (o is QROWS*NDB = 16 fragments there). The fork reaches the
// ratio with a different data flow, not different parameters.
// PT 256 was tried (the score phase splits a tile into ceil(PT/(8*BLK))
// position groups and hands them round-robin to 8 simdgroups, so at PT 128
// only four ever run it). Worth 4.8% against the UNCAPPED build and nothing
// at all against the capped one -- both were reaching the same
// instruction-scheduling slack -- so it is not worth reassociating the online
// softmax and regenerating the pins for.

// Sliced decode attention (score tile in threadgroup memory).
//
// The kernel above uses one threadgroup per (head, token). At decode that is n_heads
// threadgroups -- 8 on this model -- so on an 18-core GPU more than half the machine idles,
// and each of those 8 threadgroups walks the entire KV alone. The cost therefore grows with
// context while the parallelism stays fixed, which is exactly the shape of the measured
// decode loss: 38.3 tok/s at a 32-token context against 29.7 at 440.
//
// Here the positions are SLICED across threadgroups per (head, token). Each computes
// a softmax over its own slice and leaves the result UNNORMALISED, carrying its running max
// and its exponent sum; the combine pass below rescales them to a common max. This is the
// standard flash-attention decomposition, and it is exact -- not an approximation of the
// single-pass version.
// Decode-attention KV loaders, specialised by FUNCTION CONSTANT so the f16 pipelines
// (built with no constants, defaults = 1) compile to exactly the pre-feature code, and
// the quantized variants are separate pipelines built only when configured.
constant uint KVT_K_FC [[function_constant(3)]];
constant uint KVT_V_FC [[function_constant(4)]];
constant uint KVT_K = is_function_constant_defined(KVT_K_FC) ? KVT_K_FC : 1u;
constant uint KVT_V = is_function_constant_defined(KVT_V_FC) ? KVT_V_FC : 1u;
// A quantized cache on either side (task #156): the mega-kernel's typed loader, quantizer and
// cache-basis rotations are compiled only into pipelines built with a typed K or V, so the
// f16 pipelines keep the code (and the footprint) they were measured with.
constant bool MEGA_KVQ = (KVT_K != 1u) || (KVT_V != 1u);

// Byte offset of value `head_off` in cache row `slot` for storage type `kvt`.
inline ulong imparo_kv_row_off(uint kvt, uint width, uint slot, uint head_off) {
    if (kvt == 2u) {
        return ((ulong)slot * (width / 32u) + head_off / 32u) * 18u;
    }
    if (kvt == 8u) {
        return ((ulong)slot * (width / 32u) + head_off / 32u) * 34u;
    }
    return ((ulong)slot * width + head_off) * 2u;
}

// Four values at 4-aligned index i of a row, dequantised.
//
// Two things here are deliberate, and both were defects until 2026-08-22, when
// a KV-type sweep at 16k showed q4_0 decoding SLOWER than f16 (31.91 against
// 30.48 ms/token) despite moving a QUARTER of the bytes -- the unpack, not the
// traffic, was the cost:
//
//  - The block scale is ONE aligned 2-byte load. Block strides (18 and 34) are
//    even and row offsets are multiples of them, so `half` is always 2-byte
//    aligned; the old form assembled it from two byte loads plus a shift, an
//    or and an as_type, on every float4 of a 32-value block.
//  - The q4 nibble select is BRANCHLESS. `j` is (4*lane) & 31, so within one
//    simdgroup lanes 0-3 want low nibbles and lanes 4-7 want high ones: a
//    `j < 16 ? and : shift` ternary is divergent and both sides execute. A
//    per-lane shift of 0 or 4 followed by one mask is uniform and does the
//    same work once. Values are unchanged either way.
inline float4 imparo_kv_load4(uint kvt, device const uchar * row, uint i) {
    if (kvt == 2u) {
        device const uchar * blk = row + (i >> 5) * 18u;
        const float d = float(*(device const half *)blk);
        const uint j = i & 31u;
        const uchar4 qv = uchar4(*(device const packed_uchar4 *)(blk + 2u + (j & 15u)));
        const uchar sh = (uchar)((j >> 2) & 4u);   // 0 for j < 16, else 4
        const float4 v = float4((qv >> uchar4(sh)) & uchar4(0x0F)) - 8.0f;
        return v * d;
    }
    if (kvt == 8u) {
        device const uchar * blk = row + (i >> 5) * 34u;
        const float d = float(*(device const half *)blk);
        const char4 qv = as_type<char4>(
            uchar4(*(device const packed_uchar4 *)(blk + 2u + (i & 31u))));
        return float4(qv) * d;
    }
    return float4(*(device const half4 *)(row + (ulong)i * 2u));
}

// One value at index i of a row, dequantised.
inline float imparo_kv_load1(uint kvt, device const uchar * row, uint i) {
    if (kvt == 2u) {
        device const uchar * blk = row + (i >> 5) * 18u;
        const float d = float(*(device const half *)blk);
        const uint j = i & 31u;
        const uchar b = blk[2u + (j & 15u)];
        const float v = float((b >> ((j >> 2) & 4u)) & 0x0Fu) - 8.0f;
        return v * d;
    }
    if (kvt == 8u) {
        device const uchar * blk = row + (i >> 5) * 34u;
        const float d = float(*(device const half *)blk);
        return float(char(blk[2u + (i & 31u)])) * d;
    }
    return float(*(device const half *)(row + (ulong)i * 2u));
}

kernel void imparo_attention_decode_scoretile(
    device const float * q     [[buffer(0)]],
    device const half  * kc    [[buffer(1)]],
    device const half  * vc    [[buffer(2)]],
    device float       * part  [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]], constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],
    constant uint & ring_mask [[buffer(10)]], constant uint & slices [[buffer(11)]],
    constant uint & stage    [[buffer(12)]],
    device const uint * pt [[buffer(18)]],
    threadgroup float * scores [[threadgroup(0)]],
    threadgroup float * red    [[threadgroup(1)]],
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tcnt3  [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  nsg    [[simdgroups_per_threadgroup]])
{
    const uint tid = tid3.x, tcount = tcnt3.x;
    const uint h = tgid.x, t = tgid.y, sp = tgid.z;
    const uint pos = start_pos + t;
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;
    const uint kvh = h / (n_heads / n_kv);
    device const float * qh = q + ((ulong)t * n_heads + h) * head_dim;

    // Slice of positions this threadgroup owns.
    const uint chunk = (n + slices - 1u) / slices;
    const uint s0 = sp * chunk;
    const uint s1 = min(n, s0 + chunk);
    device float * pw = part + ((ulong)(t * n_heads + h) * slices + sp) * (head_dim + 2u);

    if (s0 >= s1) {
        // Empty slice: contribute nothing. -INFINITY as the max makes its weight exactly
        // zero in the combine, with no special case there.
        for (uint i = tid; i < head_dim; i += tcount) { pw[i] = 0.0f; }
        if (tid == 0) { pw[head_dim] = -INFINITY; pw[head_dim + 1u] = 0.0f; }
        return;
    }
    const uint ns = s1 - s0;

    for (uint s = sgid; s < ns; s += nsg) {
        const uint gp = lo + s0 + s;
        const uint ps = kv_slot(gp, ring_mask, pt);
        float acc = 0.0f;
        if (KVT_K == 1u) {
            device const half * k = kc + (ulong)ps * kv_width + kvh * head_dim;
            // float4 / half4, with K prefetched one lane-stride ahead. Four times fewer
            // memory instructions for the same bytes, and the device load's latency
            // overlaps the arithmetic instead of preceding it.
            device const float4 * q4 = (device const float4 *)qh;
            device const half4  * k4 = (device const half4 *)k;
            const uint hd4 = head_dim / 4u;
            if (lane < hd4) {
                half4 knext = k4[lane];
                for (uint j = lane; j < hd4; j += 32u) {
                    const half4 kcur = knext;
                    if (j + 32u < hd4) { knext = k4[j + 32u]; }
                    acc += dot(q4[j], float4(kcur));
                }
            }
        } else {
            // Quantized K: dequantise four values per step in-register. The row's bytes
            // are 2-4x fewer than f16, which is the point of the mode.
            device const uchar * kb = (device const uchar *)kc
                + imparo_kv_row_off(KVT_K, kv_width, ps, kvh * head_dim);
            device const float4 * q4 = (device const float4 *)qh;
            const uint hd4 = head_dim / 4u;
            for (uint j = lane; j < hd4; j += 32u) {
                acc += dot(q4[j], imparo_kv_load4(KVT_K, kb, j * 4u));
            }
        }
        acc = simd_sum(acc);
        if (lane == 0) { scores[s] = acc; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stage < 2u) {
        if (tid == 0) { pw[head_dim] = -INFINITY; pw[head_dim + 1u] = 0.0f; }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < ns; s += tcount) { local_max = max(local_max, scores[s]); }
    local_max = simd_max(local_max);
    if (lane == 0) { red[sgid] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint i = 0; i < nsg; ++i) { mx = max(mx, red[i]); }
        red[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mx = red[0];
    // Every thread has mx in a register before the sum phase reuses red[0]:
    // without this barrier a fast simdgroup 0 writes red[sgid=0] = local_sum
    // while a slow simdgroup is still reading red[0] as the max -- exp(s - sum)
    // instead of exp(s - max), decided by warp scheduling. That was the decode
    // nondeterminism (task #26). The GQA variant dodges it with red[nsg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < ns; s += tcount) {
        const float e = exp(scores[s] - mx);
        scores[s] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) { red[sgid] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < nsg; ++i) { total += red[i]; }
        red[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (stage < 3u) {
        if (tid == 0) { pw[head_dim] = -INFINITY; pw[head_dim + 1u] = 0.0f; }
        return;
    }
    // Deliberately NOT divided by the sum: the combine pass needs the unnormalised
    // accumulator and the sum separately to merge slices with different maxima.
    // P x V four dims at a time: V read as half4, the running sum kept in float4.
    {
        // Parallel over POSITIONS as well as dims.
        //
        // `for (i = tid; i < hd4; i += tcount)` leaves tcount - hd4 threads with nothing to
        // do, and hd4 is head_dim/4 -- a MODEL shape -- while tcount is `attn_threads`, a
        // tuned MACHINE value. On this model at head_dim 512 that is 128 of 512 threads
        // working while each walks every position serially, and the V pass measured 3.34 ms
        // a token against 2.13 for the K pass over the same bytes.
        //
        // `tpd` is DERIVED from both rather than picked: it is whatever the threadgroup has
        // spare once each dim has an owner. A model with head_dim >= 4 * attn_threads gets
        // tpd 1 and the original loop back, with no special case.
        const uint hd4 = head_dim / 4u;
        device float4 * pw4 = (device float4 *)pw;
        auto vrow = [&](uint sidx) {
            const uint gp = lo + s0 + sidx;
            const uint ps = kv_slot(gp, ring_mask, pt);
            return (device const half4 *)(vc + (ulong)ps * kv_width + kvh * head_dim);
        };
        auto vrowq = [&](uint sidx) {
            const uint gp = lo + s0 + sidx;
            const uint ps = kv_slot(gp, ring_mask, pt);
            return (device const uchar *)vc
                 + imparo_kv_row_off(KVT_V, kv_width, ps, kvh * head_dim);
        };
        const uint tpd = max(1u, tcount / max(1u, hd4));     // threads sharing one dim
        const uint di = tid % hd4;                           // the dim this thread owns
        const uint slice = tid / hd4;                        // which position slice
        // float4 needs a 16-byte base; nsg + 2 floats is 8-byte aligned at
        // nsg 16, and a misaligned vector access is UB (the q4 payload lesson).
        threadgroup float4 * vred =
            (threadgroup float4 *)(red + ((nsg + 2u + 3u) & ~3u));
        if (tid < hd4 * tpd) {
            float4 acc = float4(0.0f);
            if (KVT_V == 1u) {
            half4 vnext = (slice < ns) ? vrow(slice)[di] : half4(0.0h);
            for (uint sq = slice; sq < ns; sq += tpd) {
                const half4 vcur = vnext;
                if (sq + tpd < ns) { vnext = vrow(sq + tpd)[di]; }
                acc += scores[sq] * float4(vcur);
            }
            } else {
            for (uint sq = slice; sq < ns; sq += tpd) {
                acc += scores[sq] * imparo_kv_load4(KVT_V, vrowq(sq), di * 4u);
            }
            }
            vred[tid] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < hd4; i += tcount) {
            float4 acc = vred[i];
            for (uint k = 1u; k < tpd; ++k) { acc += vred[k * hd4 + i]; }
            pw4[i] = acc;
        }
    }
    if (tid == 0) { pw[head_dim] = mx; pw[head_dim + 1u] = red[0]; }
}


// STREAMING decode attention v7 (n_tok == 1): the reference vec-kernel shape,
// TEMPLATED on the head size (like the reference) so every inner loop unrolls
// at compile time; KV quant and paging stay function-constant variants.
//
//  - NE=1: all 32 lanes cooperate on ONE cache row (the reference's choice for
//    every head size that is a multiple of 128).
//  - Each simdgroup walks 32-row blocks (C), interleaved across slices:
//    block ib goes to (slice, sg) = (ib / nsg % slices, ib % nsg).
//  - Scores use NO register array for the STRIP -- a runtime-indexed array
//    spills to thread memory (measured 2x slower here) -- each unrolled cc
//    scores a scalar and lane cc stores it to the per-sg score strip; the
//    online-softmax step is then one simd_max and one simd_sum per lane.
//  - V accumulates in ROUND-LOCAL registers (dead after the block, so no
//    occupancy cost), folded into the per-sg tg accumulator once per block.
//  - FULL-ATTENTION only (host gates window == 0 && ring == 0): C divides the
//    64-row KV page, so paging costs ONE table lookup per 32 rows.
//
// HQ = QUERY HEADS PER THREADGROUP (GQA row sharing). Under GQA each K and V
// row serves n_heads/n_kv query heads, but at HQ=1 every one of them re-reads
// that row from its own threadgroup: at E4B's 8-over-2 the kernel moves 4x the
// cache's unique bytes. HQ=2 loads the row ONCE into registers and scores and
// weights it for two heads -- half the traffic, paid for with one extra dot
// and one extra simd_sum per row. What caps HQ is the threadgroup budget:
// state is HQ*(DK + nsg*DK + 2*nsg + nsg*C) floats, which at DK 512 and nsg 4
// is 10.8 KB per head against the 32 KB limit, so 2 fits and 4 does not.
//
// Slice/combine protocol unchanged: one (acc, m, l) partial per (head, slice).
template <uint DK, uint HQ = 1u>
kernel void imparo_attention_decode_stream_t(
    device const float * q     [[buffer(0)]],
    device const half  * kc    [[buffer(1)]],
    device const half  * vc    [[buffer(2)]],
    device float       * part  [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]], constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],
    constant uint & ring_mask [[buffer(10)]], constant uint & slices [[buffer(11)]],
    constant uint & stage    [[buffer(12)]],
    device const uint * pt [[buffer(18)]],
    threadgroup float * red    [[threadgroup(1)]],
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  nsg    [[simdgroups_per_threadgroup]])
{
    const uint tid = tid3.x;
    const uint h0 = tgid.x * HQ, t = tgid.y, sp = tgid.z;
    const uint n = start_pos + t + 1u;   // full span: the host gates window == 0
    const uint kvh = h0 / (n_heads / n_kv);
    const uint tcount = nsg * 32u;

    constexpr uint DK4 = DK / 4u;
    constexpr uint C = 32u;         // cache rows per block
    constexpr uint PL = DK4 / 32u;  // float4 loads per lane per row

    // red layout, every region HQ-strided:
    //   [ Q: HQ*DK | per-sg O: nsg*HQ*DK | headers: 2*nsg*HQ | scores: nsg*HQ*C ]
    //
    // THE O REGION LOOKS LIKE REGISTERS IN DISGUISE -- so4[j*DK4 + ii*32 + lane] is
    // lane-disjoint -- AND MOVING IT TO REGISTERS WAS MEASURED AND LOSES. It was tried to
    // lift the 128-thread cap this layout imposes (nsg*HQ*DK grows with thread count).
    // Both halves of that idea failed:
    //
    //   f16, streaming forced, A/B/B/A
    //     11941   512 threads 32.6 / 32.6     128 threads 35.9 / 36.1
    //     24003   512 threads 27.0            128 threads 32.6 / 32.5
    //     24003   register O @128 32.6 / 32.5     this layout @128 33.3
    //
    // 128 threads beats 512 by 10% -- more simdgroups means fewer positions each while the
    // per-block barriers and the merge both scale with nsg -- so the cap was never a
    // limitation. And with registers the merge needs PL barrier-separated rounds through a
    // small buffer with one simdgroup doing each fold, against the flat all-threads loop
    // below, which cost 2%. The cap is desirable and this layout is the cheaper one.
    threadgroup float * sq = red;
    threadgroup float * so = red + HQ * DK + sgid * HQ * DK;
    threadgroup float * hdr = red + HQ * DK + nsg * HQ * DK;
    threadgroup float * ss = hdr + 2u * nsg * HQ + sgid * HQ * C;
    threadgroup const float4 * sq4 = (threadgroup const float4 *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;

    if (stage < 3u) {
        for (uint j = 0; j < HQ; ++j) {
            device float * pwj = part
                + ((ulong)(t * n_heads + h0 + j) * slices + sp) * (DK + 2u);
            for (uint i = tid; i < DK; i += tcount) { pwj[i] = 0.0f; }
            if (tid == 0) { pwj[DK] = -INFINITY; pwj[DK + 1u] = 0.0f; }
        }
        return;
    }

    for (uint i = tid; i < HQ * DK; i += tcount) {
        const uint j = i / DK;
        sq[i] = q[((ulong)t * n_heads + h0 + j) * DK + (i - j * DK)];
    }
#pragma unroll
    for (uint j = 0; j < HQ; ++j) {
#pragma unroll
        for (uint ii = 0; ii < PL; ++ii) {
            so4[j * DK4 + ii * 32u + lane] = float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M[HQ], S[HQ];
#pragma unroll
    for (uint j = 0; j < HQ; ++j) { M[j] = -INFINITY; S[j] = 0.0f; }
    const uint nblk = (n + C - 1u) / C;
    const uint hs = kv_width / 4u;  // position-to-position stride, in half4s

    for (uint ib = sp * nsg + sgid; ib < nblk; ib += slices * nsg) {
        const uint gp = ib * C;
        const uint nc = min(C, n - gp);
        const ulong prow = KV_PAGED
            ? (ulong)(pt[gp / KV_PAGE_CELLS] * KV_PAGE_CELLS + gp % KV_PAGE_CELLS)
            : (ulong)gp;

        // scores: the row is read ONCE into registers, then scored for every
        // head this threadgroup owns; lane cc keeps row cc's score.
        if (KVT_K == 1u) {
            device const half4 * pk4 =
                (device const half4 *)kc + prow * hs + (kvh * DK) / 4u + lane;
#pragma unroll
            for (uint cc = 0; cc < C; ++cc) {
                float4 kr[PL];
#pragma unroll
                for (uint ii = 0; ii < PL; ++ii) {
                    kr[ii] = float4(pk4[cc * hs + ii * 32u]);
                }
#pragma unroll
                for (uint j = 0; j < HQ; ++j) {
                    float d = 0.0f;
#pragma unroll
                    for (uint ii = 0; ii < PL; ++ii) {
                        d += dot(sq4[j * DK4 + ii * 32u + lane], kr[ii]);
                    }
                    d = simd_sum(d);
                    if (lane == cc) { ss[j * C + cc] = cc < nc ? d : -INFINITY; }
                }
            }
        } else {
#pragma unroll
            for (uint cc = 0; cc < C; ++cc) {
                device const uchar * kb = (device const uchar *)kc
                    + imparo_kv_row_off(KVT_K, kv_width, (uint)prow + cc, kvh * DK);
                float4 kr[PL];
#pragma unroll
                for (uint ii = 0; ii < PL; ++ii) {
                    kr[ii] = imparo_kv_load4(KVT_K, kb, (ii * 32u + lane) * 4u);
                }
#pragma unroll
                for (uint j = 0; j < HQ; ++j) {
                    float d = 0.0f;
#pragma unroll
                    for (uint ii = 0; ii < PL; ++ii) {
                        d += dot(sq4[j * DK4 + ii * 32u + lane], kr[ii]);
                    }
                    d = simd_sum(d);
                    if (lane == cc) { ss[j * C + cc] = cc < nc ? d : -INFINITY; }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax: one score per lane, per head
#pragma unroll
        for (uint j = 0; j < HQ; ++j) {
            const float s = ss[j * C + lane];
            const float m0 = M[j];
            M[j] = simd_max(max(M[j], s));
            const float ms = exp(m0 - M[j]);
            const float vs = s == -INFINITY ? 0.0f : exp(s - M[j]);
            S[j] = S[j] * ms + simd_sum(vs);
            ss[j * C + lane] = vs;
#pragma unroll
            for (uint ii = 0; ii < PL; ++ii) {
                so4[j * DK4 + ii * 32u + lane] *= ms;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // V: the row is read ONCE and weighted for every head; round-local
        // register accumulators, one tg fold per block.
        float4 acc[HQ][PL];
#pragma unroll
        for (uint j = 0; j < HQ; ++j) {
#pragma unroll
            for (uint ii = 0; ii < PL; ++ii) { acc[j][ii] = float4(0.0f); }
        }
        if (KVT_V == 1u) {
            device const half4 * pv4 =
                (device const half4 *)vc + prow * hs + (kvh * DK) / 4u + lane;
#pragma unroll
            for (uint cc = 0; cc < C; ++cc) {
                float4 vr[PL];
#pragma unroll
                for (uint ii = 0; ii < PL; ++ii) {
                    vr[ii] = float4(pv4[cc * hs + ii * 32u]);
                }
#pragma unroll
                for (uint j = 0; j < HQ; ++j) {
                    const float w = ss[j * C + cc];
#pragma unroll
                    for (uint ii = 0; ii < PL; ++ii) { acc[j][ii] += w * vr[ii]; }
                }
            }
        } else {
#pragma unroll
            for (uint cc = 0; cc < C; ++cc) {
                device const uchar * vb = (device const uchar *)vc
                    + imparo_kv_row_off(KVT_V, kv_width, (uint)prow + cc, kvh * DK);
                float4 vr[PL];
#pragma unroll
                for (uint ii = 0; ii < PL; ++ii) {
                    vr[ii] = imparo_kv_load4(KVT_V, vb, (ii * 32u + lane) * 4u);
                }
#pragma unroll
                for (uint j = 0; j < HQ; ++j) {
                    const float w = ss[j * C + cc];
#pragma unroll
                    for (uint ii = 0; ii < PL; ++ii) { acc[j][ii] += w * vr[ii]; }
                }
            }
        }
#pragma unroll
        for (uint j = 0; j < HQ; ++j) {
#pragma unroll
            for (uint ii = 0; ii < PL; ++ii) {
                so4[j * DK4 + ii * 32u + lane] += acc[j][ii];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // merge simdgroups per head (headers first, then rescaled accumulator fold)
    if (lane == 0u) {
#pragma unroll
        for (uint j = 0; j < HQ; ++j) {
            hdr[2u * (sgid * HQ + j)] = M[j];
            hdr[2u * (sgid * HQ + j) + 1u] = S[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < HQ; ++j) {
        device float * pwj = part
            + ((ulong)(t * n_heads + h0 + j) * slices + sp) * (DK + 2u);
        float mx = -INFINITY;
        for (uint g = 0; g < nsg; ++g) { mx = max(mx, hdr[2u * (g * HQ + j)]); }
        if (mx == -INFINITY) {
            // slice past the span (over-sliced shallow force-stream): empty partial
            for (uint i = tid; i < DK; i += tcount) { pwj[i] = 0.0f; }
            if (tid == 0) { pwj[DK] = -INFINITY; pwj[DK + 1u] = 0.0f; }
            continue;
        }
        for (uint i = tid; i < DK; i += tcount) {
            float a = 0.0f;
            for (uint g = 0; g < nsg; ++g) {
                a += red[HQ * DK + (g * HQ + j) * DK + i]
                   * exp(hdr[2u * (g * HQ + j)] - mx);
            }
            pwj[i] = a;
        }
        if (tid == 0u) {
            float lsum = 0.0f;
            for (uint g = 0; g < nsg; ++g) {
                lsum += hdr[2u * (g * HQ + j) + 1u] * exp(hdr[2u * (g * HQ + j)] - mx);
            }
            pwj[DK] = mx;
            pwj[DK + 1u] = lsum;
        }
    }
}

typedef decltype(imparo_attention_decode_stream_t<512u>) imparo_attn_decode_stream_kt;
template [[host_name("imparo_attention_decode_stream_dk128")]] kernel imparo_attn_decode_stream_kt imparo_attention_decode_stream_t<128u>;
template [[host_name("imparo_attention_decode_stream_dk256")]] kernel imparo_attn_decode_stream_kt imparo_attention_decode_stream_t<256u>;
template [[host_name("imparo_attention_decode_stream_dk512")]] kernel imparo_attn_decode_stream_kt imparo_attention_decode_stream_t<512u>;
// GQA row sharing, two query heads per threadgroup (full-attention geometry).
template [[host_name("imparo_attention_decode_stream_dk512_g2")]] kernel imparo_attn_decode_stream_kt imparo_attention_decode_stream_t<512u, 2u>;
// hd 256 (Qwen3.8-27B's full-attention layers: 24 query heads over 4 KV heads, six per
// KV head): 2 and 3 fit the threadgroup budget (HQ x 1416 floats at nsg 4), 6 does not.
template [[host_name("imparo_attention_decode_stream_dk256_g2")]] kernel imparo_attn_decode_stream_kt imparo_attention_decode_stream_t<256u, 2u>;
template [[host_name("imparo_attention_decode_stream_dk256_g3")]] kernel imparo_attn_decode_stream_kt imparo_attention_decode_stream_t<256u, 3u>;

// Sliced decode attention, GROUPED BY KV HEAD.
//
// `imparo_attention_decode_scoretile` gives a threadgroup one QUERY head, so with 8 query heads over
// 2 KV heads every K and V row is read FOUR times. At a 5642-position context that is
// 454 MB of KV per decoded token, of which 341 MB is re-reads, and it measures 69 GB/s
// against 147 for a plain read.
//
// An earlier test concluded grouping "would buy nothing" -- time was flat across 2/4/8
// query heads. That was measured at pos=512, where one KV head's slice is 0.5 MB and the
// cache absorbs the repeats. At 5642 positions it is 5.8 MB and it does not.
//
// Here a threadgroup owns a KV HEAD and serves every query head that shares it: one K row
// scores HQ queries, one V row weights HQ outputs. The partial buffer is still written per
// query head, so the combine pass is unchanged.
template <uint HQ>
kernel void imparo_attention_decode_scoretile_gqa_t(
    device const float * q     [[buffer(0)]],
    device const half  * kc    [[buffer(1)]],
    device const half  * vc    [[buffer(2)]],
    device float       * part  [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]], constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],
    constant uint & ring_mask [[buffer(10)]], constant uint & slices [[buffer(11)]],
    device const uint * pt [[buffer(18)]],
    threadgroup float * scores [[threadgroup(0)]],
    threadgroup float * red    [[threadgroup(1)]],
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tcnt3  [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  nsg    [[simdgroups_per_threadgroup]])
{
    // TEMPLATED ON THE GROUP SIZE, because sizing the accumulators for a worst case costs
    // real time. With a runtime `hq` the arrays below had to be declared `[MAXHQ]` = 8, so
    // a model needing 4 still paid 8 float4s a thread in the V pass -- 32 registers against
    // the ungrouped kernel's 4, at 512 threads a threadgroup. Measured, at 12009 tokens:
    //
    //     arrays sized 8 (runtime hq)   decode 31.4 / 31.5
    //     arrays sized 4 (this model)          32.8 / 33.3     +5.4%
    //
    // A compile-time HQ sizes them exactly and unrolls the head loops, and unlike hacking
    // the constant down it stays correct for every model: `hq = min(8, n_heads/n_kv)` would
    // silently drop heads 4..7 on a model with eight per KV head. The streaming kernel is
    // already a template on its group size; this one now matches.
    constexpr uint MAXHQ = HQ;
    const uint tid = tid3.x, tcount = tcnt3.x;
    const uint t = tgid.y, sp = tgid.z;
    const uint hq = HQ;
    // HQ NEED NOT BE THE WHOLE GQA SHARE. Grouping divides KV traffic by HQ and
    // multiplies accumulators per thread by it, so the best HQ is not automatically the
    // largest one that fits: this kernel is register-bound, not traffic-bound. Splitting
    // a KV head's query heads across several threadgroups is what lets HQ be 2 on a
    // model whose share is 4.
    //
    //   share 4, HQ 4    grid.x = n_kv        tgid.x IS the kv head
    //   share 4, HQ 2    grid.x = n_heads/2   two threadgroups per kv head
    const uint share = n_heads / n_kv;
    const uint groups_per_kvh = share / HQ;     // 1 when HQ is the whole share
    const uint kvh = tgid.x / groups_per_kvh;
    const uint h0 = kvh * share + (tgid.x % groups_per_kvh) * HQ;
    const uint pos = start_pos + t;
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;

    const uint chunk = (n + slices - 1u) / slices;
    const uint s0 = sp * chunk;
    const uint s1 = min(n, s0 + chunk);

    if (s0 >= s1) {
        for (uint j = 0; j < hq; ++j) {
            device float * pw = part
                + ((ulong)(t * n_heads + h0 + j) * slices + sp) * (head_dim + 2u);
            for (uint i = tid; i < head_dim; i += tcount) { pw[i] = 0.0f; }
            if (tid == 0) { pw[head_dim] = -INFINITY; pw[head_dim + 1u] = 0.0f; }
        }
        return;
    }
    const uint ns = s1 - s0;

    // THE Q BASE, HOISTED. Head j's vector sits at a fixed stride of head_dim from head
    // h0, so the address is loop-invariant -- but it used to be rebuilt inside the
    // INNERMOST loop, once per K element per query head:
    //
    //     for i in dims:                       <- innermost
    //         for j in heads:
    //             q4 = (float4*)(q + ((ulong)t*n_heads + h0 + j)*head_dim);   64-bit
    //             acc[j] += dot(q4[i], kf);                                   rebuild
    //
    // That is a 64-bit multiply-add per MAC, and hq device loads of Q per K element, paid
    // to save reading K hq times. The saving was free -- this path measures 69 GB/s against
    // 147 for a plain read, so its K re-reads were absorbed and the memory system was never
    // the limit. The cost was not: grouping measured 15% SLOWER end to end than the
    // ungrouped kernel, which hoists its own q4 out of the position loop.
    //
    // Walking `qp` by hd4 per head keeps the same addresses with one add.
    const uint hd4_q = head_dim / 4u;
    device const float4 * const qbase =
        (device const float4 *)(q + ((ulong)t * n_heads + h0) * head_dim);

    // One K row, HQ dot products. The row is read once instead of HQ times.
    for (uint s = sgid; s < ns; s += nsg) {
        const uint gp = lo + s0 + s;
        const uint ps = kv_slot(gp, ring_mask, pt);
        const uint hd4 = head_dim / 4u;
        float acc[MAXHQ];
        for (uint j = 0; j < hq; ++j) { acc[j] = 0.0f; }
        if (KVT_K == 1u) {
            device const half4 * k4 = (device const half4 *)(kc + (ulong)ps * kv_width
                                                                + kvh * head_dim);
            half4 knext = (lane < hd4) ? k4[lane] : half4(0.0h);
            for (uint i = lane; i < hd4; i += 32u) {
                const half4 kcur = knext;
                if (i + 32u < hd4) { knext = k4[i + 32u]; }
                const float4 kf = float4(kcur);
                device const float4 * qp = qbase + i;
                for (uint j = 0; j < hq; ++j) {
                    acc[j] += dot(*qp, kf);
                    qp += hd4_q;
                }
            }
        } else {
            device const uchar * kb = (device const uchar *)kc
                + imparo_kv_row_off(KVT_K, kv_width, ps, kvh * head_dim);
            for (uint i = lane; i < hd4; i += 32u) {
                const float4 kf = imparo_kv_load4(KVT_K, kb, i * 4u);
                device const float4 * qp = qbase + i;
                for (uint j = 0; j < hq; ++j) {
                    acc[j] += dot(*qp, kf);
                    qp += hd4_q;
                }
            }
        }
        for (uint j = 0; j < hq; ++j) {
            const float dsum = simd_sum(acc[j]);
            if (lane == 0) { scores[j * ns + s] = dsum; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax for ALL query heads, ONE set of barriers.
    //
    // These four barriers used to sit INSIDE a `for j` loop, so a threadgroup carrying hq
    // query heads paid 4*hq of them -- 16 at hq=4 against the ungrouped kernel's 4. And
    // grouping gives each threadgroup 1/hq of the positions, so the barrier cost per
    // position rose by hq^2: sixteenfold here. That is what made grouping lose 15% end to
    // end while reading 4x FEWER bytes, and it is why holding the slice count down changed
    // nothing -- the barrier count per slice is 4*hq whatever the slicing.
    //
    // Batching the heads inside each phase makes the count independent of hq. The
    // per-simdgroup partials are indexed [sgid * hq + j] instead of [sgid], which is what
    // the widened `red` allocation on the host is for.
    const uint MXOFF  = nsg * MAXHQ;          // combined maxima, one per head
    const uint SUMOFF = MXOFF + MAXHQ;        // combined sums, one per head

    for (uint j = 0; j < hq; ++j) {
        threadgroup float * sc = scores + j * ns;
        float local_max = -INFINITY;
        for (uint s = tid; s < ns; s += tcount) { local_max = max(local_max, sc[s]); }
        local_max = simd_max(local_max);
        if (lane == 0) { red[sgid * hq + j] = local_max; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hq) {
        float mx = -INFINITY;
        for (uint i = 0; i < nsg; ++i) { mx = max(mx, red[i * hq + tid]); }
        red[MXOFF + tid] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < hq; ++j) {
        threadgroup float * sc = scores + j * ns;
        const float mx = red[MXOFF + j];
        float local_sum = 0.0f;
        for (uint s = tid; s < ns; s += tcount) {
            const float e = exp(sc[s] - mx);
            sc[s] = e;
            local_sum += e;
        }
        local_sum = simd_sum(local_sum);
        if (lane == 0) { red[sgid * hq + j] = local_sum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < hq) {
        float total = 0.0f;
        for (uint i = 0; i < nsg; ++i) { total += red[i * hq + tid]; }
        red[SUMOFF + tid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One V row, HQ weighted accumulations. NOT divided by the sum: the combine pass needs
    // the unnormalised accumulator and the sum separately to merge slices with different
    // maxima.
    {
        const uint hd4 = head_dim / 4u;
        auto vrow = [&](uint sidx) {
            const uint gp = lo + s0 + sidx;
            const uint ps = kv_slot(gp, ring_mask, pt);
            return (device const half4 *)(vc + (ulong)ps * kv_width + kvh * head_dim);
        };
        auto vrowq = [&](uint sidx) {
            const uint gp = lo + s0 + sidx;
            const uint ps = kv_slot(gp, ring_mask, pt);
            return (device const uchar *)vc
                 + imparo_kv_row_off(KVT_V, kv_width, ps, kvh * head_dim);
        };
        for (uint i = tid; i < hd4; i += tcount) {
            float4 acc[MAXHQ];
            for (uint j = 0; j < hq; ++j) { acc[j] = float4(0.0f); }
            if (KVT_V == 1u) {
            half4 vnext = vrow(0u)[i];
            for (uint sq = 0; sq < ns; ++sq) {
                const half4 vcur = vnext;
                if (sq + 1u < ns) { vnext = vrow(sq + 1u)[i]; }
                const float4 vf = float4(vcur);
                for (uint j = 0; j < hq; ++j) { acc[j] += scores[j * ns + sq] * vf; }
            }
            } else {
            for (uint sq = 0; sq < ns; ++sq) {
                const float4 vf = imparo_kv_load4(KVT_V, vrowq(sq), i * 4u);
                for (uint j = 0; j < hq; ++j) { acc[j] += scores[j * ns + sq] * vf; }
            }
            }
            for (uint j = 0; j < hq; ++j) {
                device float4 * pw4 = (device float4 *)(part
                    + ((ulong)(t * n_heads + h0 + j) * slices + sp) * (head_dim + 2u));
                pw4[i] = acc[j];
            }
        }
    }
    if (tid == 0) {
        for (uint j = 0; j < hq; ++j) {
            device float * pw = part
                + ((ulong)(t * n_heads + h0 + j) * slices + sp) * (head_dim + 2u);
            pw[head_dim] = red[nsg * MAXHQ + j];
            pw[head_dim + 1u] = red[nsg * MAXHQ + MAXHQ + j];
        }
    }
}

// The group sizes the host may select. Not every value in 2..8: each instantiation is a
// pipeline compiled at start-up, so this covers the shapes real GQA models use and the host
// falls back to the ungrouped kernel for anything else.
typedef decltype(imparo_attention_decode_scoretile_gqa_t<4u>) imparo_attn_gqa_kt;
template [[host_name("imparo_attention_decode_scoretile_gqa_hq2")]] kernel imparo_attn_gqa_kt imparo_attention_decode_scoretile_gqa_t<2u>;
template [[host_name("imparo_attention_decode_scoretile_gqa_hq4")]] kernel imparo_attn_gqa_kt imparo_attention_decode_scoretile_gqa_t<4u>;
template [[host_name("imparo_attention_decode_scoretile_gqa_hq8")]] kernel imparo_attn_gqa_kt imparo_attention_decode_scoretile_gqa_t<8u>;


// FLASH-DECODING FOR SMALL HEAD DIMS (hd <= 128), GROUPED BY KV HEAD.
//
// Decode attention at depth on LFM2 (hd 64, GQA 4) read 134 MB per layer per token at
// 209 GB/s: the ungrouped scoretile kernel gives every QUERY head its own threadgroup, so
// each K/V row of a KV head is read four times, and at 16k keys the cache does not
// collapse the repeats (skip-and-diff, 17123 tokens: 6.9 ms/token of attention against
// upstream's 3.1). The grouped scoretile kernel reads each row once but gives one
// SIMDGROUP one position -- 16 of 32 lanes load a half4 and then FOUR simd_sums per
// position -- which at hd 64 is the kernel (2-5x slower than ungrouped). So this kernel:
//
//   scores  one POSITION per thread: the thread reads its K row (HD halves) once and
//           forms HQ dot products against Q staged in threadgroup memory -- no cross-
//           lane reduction per position; HQ x chunk scores go to threadgroup memory
//   softmax per head over the slice's chunk: one max and one sum reduction per head
//           per SLICE (a threadgroup), not per position
//   P x V   lanes own DIMS (HD/32 each), simdgroups split the positions: a V row is
//           read once, coalesced, and weights HQ outputs; HQ x HD/32 accumulators a
//           thread; the simdgroups' partials meet once in threadgroup memory
//   slices  the context is split into ~512-key chunks by the host (upstream's decode
//           kernel: nwg 32 at 16k keys), so 8 KV heads x 32 slices keep the memory
//           system busy; the existing combine merges the slices.
//
// Same bindings as scoretile_gqa (a drop-in on the split route); the partial layout
// (HD accumulators, max, sum per query head per slice) is the combine's. Positions
// resolve through kv_slot: f16 cache only (a quantized cache keeps the scoretile route).
template <uint HD, uint HQ, uint KVW>
kernel void imparo_attention_decode_fd_t(
    device const float * q     [[buffer(0)]],
    device const half  * kc    [[buffer(1)]],
    device const half  * vc    [[buffer(2)]],
    device float       * part  [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]], constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],
    constant uint & ring_mask [[buffer(10)]], constant uint & slices [[buffer(11)]],
    device const uint * pt [[buffer(18)]],
    threadgroup float * scores [[threadgroup(0)]],   // HQ x chunk
    threadgroup float * red    [[threadgroup(1)]],   // HQ*HD (Q) + 2*nsg*HQ + nsg*HQ*HD
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tcnt3  [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  nsg    [[simdgroups_per_threadgroup]])
{
    static_assert(HD % 32u == 0u, "P x V gives each lane HD/32 dims");
    constexpr uint HD4 = HD / 4u;
    constexpr uint DPL = HD / 32u;                 // dims per lane in P x V
    const uint kvw = (KVW != 0u) ? KVW : kv_width;
    const uint tid = tid3.x, tcount = tcnt3.x;
    (void)head_dim;
    const uint share = n_heads / n_kv;
    const uint groups_per_kvh = share / HQ;
    const uint kvh = tgid.x / groups_per_kvh;
    const uint h0 = kvh * share + (tgid.x % groups_per_kvh) * HQ;
    const uint t = tgid.y, sp = tgid.z;
    const uint pos = start_pos + t;
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;
    const uint chunk = (n + slices - 1u) / slices;
    const uint s0 = min(n, sp * chunk), s1 = min(n, s0 + chunk);

    threadgroup float * qs = red;                       // HQ x HD, this group's queries
    threadgroup float * rmax = red + HQ * HD;           // nsg x HQ
    threadgroup float * rsum = rmax + nsg * HQ;         // nsg x HQ
    threadgroup float * racc = rsum + nsg * HQ;         // nsg x HQ x HD
    for (uint e = tid; e < HQ * HD; e += tcount) {
        qs[e] = q[((ulong)t * n_heads + h0 + e / HD) * HD + e % HD];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- scores: one position per thread per pass, HQ dot products from one K row ----
    float lmax[HQ];
    for (uint j = 0; j < HQ; ++j) { lmax[j] = -INFINITY; }
    for (uint s = s0 + tid; s < s1; s += tcount) {
        const uint ps = kv_slot(lo + s, ring_mask, pt);
        // PROBE (IMPARO_ATTN_SKIP bit 32): read K at a CONTIGUOUS per-thread offset instead
        // of at the position stride. The cache is position-major, so adjacent threads take
        // adjacent positions and stride by kvw -- 1024 B on this model. The answer is WRONG
        // under the probe (the wrong keys are scored) and only the time is read, which is
        // what separates "the stride costs" from "the kernel is latency-bound anyway".
        device const half4 * k4 = (ATTN_SKIP & 32u) != 0u
            ? (device const half4 *)(kc + (ulong)(s - s0) * HD + kvh * HD)
            : (device const half4 *)(kc + (ulong)ps * kvw + kvh * HD);
        float acc[HQ];
        for (uint j = 0; j < HQ; ++j) { acc[j] = 0.0f; }
        for (uint i = 0; i < HD4; ++i) {
            const float4 kf = float4(k4[i]);
            for (uint j = 0; j < HQ; ++j) {
                const float4 qf = *((threadgroup const float4 *)(qs + j * HD) + i);
                acc[j] += dot(qf, kf);
            }
        }
        for (uint j = 0; j < HQ; ++j) {
            scores[j * chunk + (s - s0)] = acc[j];
            lmax[j] = max(lmax[j], acc[j]);
        }
    }
    for (uint j = 0; j < HQ; ++j) {
        const float m = simd_max(lmax[j]);
        if (lane == 0u) { rmax[sgid * HQ + j] = m; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx[HQ];
    for (uint j = 0; j < HQ; ++j) {
        float m = -INFINITY;
        for (uint g = 0; g < nsg; ++g) { m = max(m, rmax[g * HQ + j]); }
        mx[j] = m;
    }
    // ---- softmax over the slice: exp and sum, in place ----
    float lsum[HQ];
    for (uint j = 0; j < HQ; ++j) { lsum[j] = 0.0f; }
    for (uint s = s0 + tid; s < s1; s += tcount) {
        for (uint j = 0; j < HQ; ++j) {
            const float p = exp(scores[j * chunk + (s - s0)] - mx[j]);
            scores[j * chunk + (s - s0)] = p;
            lsum[j] += p;
        }
    }
    for (uint j = 0; j < HQ; ++j) {
        const float sm = simd_sum(lsum[j]);
        if (lane == 0u) { rsum[sgid * HQ + j] = sm; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- P x V: lanes own dims, simdgroups split positions ----
    float vacc[HQ][DPL];
    for (uint j = 0; j < HQ; ++j) { for (uint d = 0; d < DPL; ++d) { vacc[j][d] = 0.0f; } }
    for (uint s = s0 + sgid; s < s1; s += nsg) {
        const uint ps = kv_slot(lo + s, ring_mask, pt);
        device const half * vr = vc + (ulong)ps * kvw + kvh * HD + lane * DPL;
        float vf[DPL];
        for (uint d = 0; d < DPL; ++d) { vf[d] = float(vr[d]); }
        for (uint j = 0; j < HQ; ++j) {
            const float p = scores[j * chunk + (s - s0)];
            for (uint d = 0; d < DPL; ++d) { vacc[j][d] += p * vf[d]; }
        }
    }
    for (uint j = 0; j < HQ; ++j) {
        for (uint d = 0; d < DPL; ++d) { racc[(sgid * HQ + j) * HD + lane * DPL + d] = vacc[j][d]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- the slice's partial: HD accumulators, max, sum per query head ----
    for (uint e = tid; e < HQ * HD; e += tcount) {
        const uint j = e / HD, d = e % HD;
        float a = 0.0f;
        for (uint g = 0; g < nsg; ++g) { a += racc[(g * HQ + j) * HD + d]; }
        part[((ulong)(t * n_heads + h0 + j) * slices + sp) * (HD + 2u) + d] = a;
    }
    if (tid < HQ) {
        const uint j = tid;
        float sm = 0.0f;
        for (uint g = 0; g < nsg; ++g) { sm += rsum[g * HQ + j]; }
        device float * pj = part + ((ulong)(t * n_heads + h0 + j) * slices + sp) * (HD + 2u);
        pj[HD] = mx[j];
        pj[HD + 1u] = sm;
    }
}
#if IMPARO_HD0 && (IMPARO_HD0 <= 128)
template [[host_name("imparo_attention_decode_fd_s0_hq2")]] kernel void imparo_attention_decode_fd_t<IMPARO_HD0, 2u, IMPARO_KVW0>(
    device const float *, device const half *, device const half *, device float *, constant uint &, constant uint &,
    constant uint &, constant uint &, constant uint &, constant uint &, constant uint &, constant uint &,
    device const uint *, threadgroup float *, threadgroup float *, uint3, uint3, uint3, uint, uint, uint);
template [[host_name("imparo_attention_decode_fd_s0_hq4")]] kernel void imparo_attention_decode_fd_t<IMPARO_HD0, 4u, IMPARO_KVW0>(
    device const float *, device const half *, device const half *, device float *, constant uint &, constant uint &,
    constant uint &, constant uint &, constant uint &, constant uint &, constant uint &, constant uint &,
    device const uint *, threadgroup float *, threadgroup float *, uint3, uint3, uint3, uint, uint, uint);
template [[host_name("imparo_attention_decode_fd_s0_hq8")]] kernel void imparo_attention_decode_fd_t<IMPARO_HD0, 8u, IMPARO_KVW0>(
    device const float *, device const half *, device const half *, device float *, constant uint &, constant uint &,
    constant uint &, constant uint &, constant uint &, constant uint &, constant uint &, constant uint &,
    device const uint *, threadgroup float *, threadgroup float *, uint3, uint3, uint3, uint, uint, uint);
#endif

// BRICKS: DECODE ATTENTION, ONE SIMDGROUP PER POSITION, LANES OWN HD/32 DIMS.
// attn_span_online: positions s0, s0 + step, ... below n over the span starting at lo -- this
// lane's dims of the K and V rows (half4 loads when they divide, half2 otherwise), ONE simd_sum
// for the score, the online softmax kept per simdgroup in (m, l, o[DPL]). The vec kernel
// strides it by its simdgroup count; the mega block by (part, simdgroup) over its split.
// attn_store_slot / attn_merge_slot / attn_fold_slots: the (o, m, l) hand-off through
// threadgroup memory, in the vec kernel's form. A state that saw no position carries
// m = -inf and weight 0; two empty states stay empty (exp(-inf - -inf) is NaN; task #142).
// attn_load_row: this lane's HD/32 dims of one K row and one V row (half4 loads when they
// divide, half2 otherwise), as floats.
template <uint HD>
inline void attn_load_row(device const half * kc, device const half * vc, ulong row,
                          thread float * kf, thread float * vf) {
    constexpr uint DPL = HD / 32u;
    constexpr bool V4 = (DPL % 4u) == 0u;
    constexpr uint NV = V4 ? DPL / 4u : DPL / 2u;
    if (V4) {
        device const half4 * k4 = (device const half4 *)(kc + row);
        device const half4 * v4 = (device const half4 *)(vc + row);
        for (uint i = 0; i < NV; ++i) {
            const float4 k = float4(k4[i]), v = float4(v4[i]);
            kf[4u * i] = k.x; kf[4u * i + 1u] = k.y; kf[4u * i + 2u] = k.z; kf[4u * i + 3u] = k.w;
            vf[4u * i] = v.x; vf[4u * i + 1u] = v.y; vf[4u * i + 2u] = v.z; vf[4u * i + 3u] = v.w;
        }
    } else {
        device const half2 * k2 = (device const half2 *)(kc + row);
        device const half2 * v2 = (device const half2 *)(vc + row);
        for (uint i = 0; i < NV; ++i) {
            const float2 k = float2(k2[i]), v = float2(v2[i]);
            kf[2u * i] = k.x; kf[2u * i + 1u] = k.y;
            vf[2u * i] = v.x; vf[2u * i + 1u] = v.y;
        }
    }
}
// One side of a cache row for this lane, whatever the storage type (KVT_K / KVT_V, the
// decode-attention function constants): block-scaled rows (q4_0 18, q8_0 34 bytes per 32
// values) dequantised four values at a time through imparo_kv_load4; at two dims per lane
// (hd 64) a lane pair shares one four-value group. Task #156.
template <uint HD>
inline void attn_load_side(uint kvt, device const uchar * cache, uint kvw, uint ps, uint kvh, uint lane,
                           thread float * f) {
    constexpr uint DPL = HD / 32u;
    device const uchar * row = cache + imparo_kv_row_off(kvt, kvw, ps, kvh * HD);
    if (DPL >= 4u) {
        for (uint i = 0; i < DPL / 4u; ++i) {
            const float4 v = imparo_kv_load4(kvt, row, lane * DPL + 4u * i);
            f[4u * i] = v.x; f[4u * i + 1u] = v.y; f[4u * i + 2u] = v.z; f[4u * i + 3u] = v.w;
        }
    } else {
        const float4 v = imparo_kv_load4(kvt, row, (lane * DPL) & ~3u);
        f[0] = (lane & 1u) ? v.z : v.x;
        f[1] = (lane & 1u) ? v.w : v.y;
    }
}
// This lane's dims of K and V at cache slot `ps`: the f16 form when both sides are f16 (the
// pipelines built without the constants compile exactly the code they had), the typed
// loader otherwise.
template <uint HD>
inline void attn_load_kv(device const half * kc, device const half * vc, uint kvw, uint ps, uint kvh, uint lane,
                         thread float * kf, thread float * vf) {
    constexpr uint DPL = HD / 32u;
    if (!MEGA_KVQ) {
        attn_load_row<HD>(kc, vc, (ulong)ps * kvw + kvh * HD + lane * DPL, kf, vf);
        return;
    }
    attn_load_side<HD>(KVT_K, (device const uchar *)kc, kvw, ps, kvh, lane, kf);
    attn_load_side<HD>(KVT_V, (device const uchar *)vc, kvw, ps, kvh, lane, vf);
}
template <uint HD, uint KVW>
inline void attn_span_online(device const half * kc, device const half * vc, uint kvw, uint kvh,
                             uint lo, uint n, uint s0, uint step, uint ring_mask,
                             device const uint * pt, uint lane, thread const float * qr,
                             thread float & m, thread float & l, thread float * o) {
    constexpr uint DPL = HD / 32u;
    for (uint s = s0; s < n; s += step) {
        const uint ps = kv_slot(lo + s, ring_mask, pt);
        float kf[DPL], vf[DPL];
        attn_load_kv<HD>(kc, vc, kvw, ps, kvh, lane, kf, vf);
        float sc = 0.0f;
        for (uint i = 0; i < DPL; ++i) { sc += qr[i] * kf[i]; }
        sc = simd_sum(sc);
        const float mn = max(m, sc);
        const float f = exp(m - mn);
        const float p = exp(sc - mn);
        l = l * f + p;
        for (uint i = 0; i < DPL; ++i) { o[i] = o[i] * f + p * vf[i]; }
        m = mn;
    }
}
template <uint HD>
inline void attn_store_slot(threadgroup float * slot, uint lane, float m, float l, thread const float * o) {
    constexpr uint DPL = HD / 32u;
    for (uint i = 0; i < DPL; ++i) { slot[lane * DPL + i] = o[i]; }
    if (lane == 0u) { slot[HD] = m; slot[HD + 1u] = l; }
}
template <uint HD>
inline void attn_merge_slot(threadgroup const float * slot, uint lane, thread float & m, thread float & l, thread float * o) {
    constexpr uint DPL = HD / 32u;
    const float m2 = slot[HD], l2 = slot[HD + 1u];
    const float M = max(m, m2);
    const float f1 = (m == -INFINITY) ? 0.0f : exp(m - M);
    const float f2 = (m2 == -INFINITY) ? 0.0f : exp(m2 - M);
    l = l * f1 + l2 * f2;
    for (uint i = 0; i < DPL; ++i) { o[i] = o[i] * f1 + slot[lane * DPL + i] * f2; }
    m = M;
}
// Simdgroup 0 folds its own state with slots 1..G-1 onto a common maximum: (M, L, acc) unnormalised.
template <uint HD>
inline void attn_fold_slots(threadgroup const float * red, uint G, uint lane, float m, float l,
                            thread const float * o, thread float & M, thread float & L, thread float * acc) {
    constexpr uint DPL = HD / 32u;
    M = m;
    for (uint g = 1; g < G; ++g) { M = max(M, red[g * (HD + 2u) + HD]); }
    const float f0 = (m == -INFINITY) ? 0.0f : exp(m - M);
    L = l * f0;
    for (uint i = 0; i < DPL; ++i) { acc[i] = o[i] * f0; }
    for (uint g = 1; g < G; ++g) {
        threadgroup const float * rg = red + g * (HD + 2u);
        const float f = (rg[HD] == -INFINITY) ? 0.0f : exp(rg[HD] - M);
        L += f * rg[HD + 1u];
        for (uint i = 0; i < DPL; ++i) { acc[i] += f * rg[lane * DPL + i]; }
    }
}

// BRICK: GROUPED DECODE ATTENTION OVER A CONTIGUOUS RUN, K/V ONCE PER KV HEAD (task #151).
// The mega-kernel's deep body: one simdgroup walks positions [s0, s1) for `hq` query heads
// of one KV head. Each K/V row is loaded once (this lane's HD/32 dims) and scored for every
// head -- one simd_sum per (position, head) -- and the online softmax state of every head
// stays in registers. A paged full layer looks its page up once per KV page (the run is
// contiguous); a ring layer masks. Heads the item does not own (j >= hq) keep the empty
// state (m = -inf, weight 0). ATTN_GROUP_HQ(HD) is the heads per item, from the register
// budget of two HD/32-float arrays per head within 64 floats: 2 at hd 512, 4 at hd 256,
// 8 at hd 128 and below; a wider query group takes several items (sub-groups).
constexpr uint attn_group_hq(uint hd) { return (1024u / hd) < 8u ? (1024u / hd) : 8u; }
// Positions in flight per simdgroup: the run is a dependent chain (row load, dot, simd_sum,
// exp, update) and at a small head dim the rows are too short to hide it -- LFM2's hd-64
// layers at 16k keys read 2-4% behind the flash-decoding dispatch one position at a time.
// So a batch of ATTN_GROUP_U rows is loaded at once and folded with one rescale: 8 at
// hd <= 128, 4 at hd 256, 1 at hd 512 (the two-head state already fills the registers there).
constexpr uint attn_group_u(uint hd) { return hd >= 512u ? 1u : ((64u / (hd / 16u)) < 8u ? (64u / (hd / 16u)) : 8u); }
template <uint HD, uint KVW>
inline void attn_group_run_online(device const half * kc, device const half * vc, uint kvw, uint kvh,
                                  uint lo, uint s0, uint s1, uint ring_mask, device const uint * pt,
                                  uint lane, uint hq, thread const float * qr,
                                  thread float * m, thread float * l, thread float * o) {
    constexpr uint HQ = attn_group_hq(HD);
    constexpr uint DPL = HD / 32u;
    constexpr uint U = attn_group_u(HD);
    uint page = 0u;
    uint s = s0;
    if (U > 1u) {
        for (; s + U <= s1; s += U) {
            float kf[U * DPL], vf[U * DPL];
#pragma unroll
            for (uint u = 0; u < U; ++u) {
                const uint gp = lo + s + u;
                uint ps;
                if (ring_mask > 0u) { ps = gp & ring_mask; }
                else if (KV_PAGED) {
                    if ((u == 0u && s == s0) || (gp % KV_PAGE_CELLS) == 0u) { page = pt[gp / KV_PAGE_CELLS]; }
                    ps = page * KV_PAGE_CELLS + gp % KV_PAGE_CELLS;
                } else { ps = gp; }
                attn_load_kv<HD>(kc, vc, kvw, ps, kvh, lane, kf + u * DPL, vf + u * DPL);
            }
#pragma unroll
            for (uint j = 0; j < HQ; ++j) {
                if (j < hq) {
                    float sc[U];
                    float smax = -INFINITY;
#pragma unroll
                    for (uint u = 0; u < U; ++u) {
                        float d = 0.0f;
                        for (uint i = 0; i < DPL; ++i) { d += qr[j * DPL + i] * kf[u * DPL + i]; }
                        sc[u] = simd_sum(d);
                        smax = max(smax, sc[u]);
                    }
                    const float mn = max(m[j], smax);
                    const float f = exp(m[j] - mn);
                    l[j] *= f;
                    for (uint i = 0; i < DPL; ++i) { o[j * DPL + i] *= f; }
#pragma unroll
                    for (uint u = 0; u < U; ++u) {
                        const float pw = exp(sc[u] - mn);
                        l[j] += pw;
                        for (uint i = 0; i < DPL; ++i) { o[j * DPL + i] += pw * vf[u * DPL + i]; }
                    }
                    m[j] = mn;
                }
            }
        }
    }
    // The tail (and the whole run at U == 1): one position at a time.
    for (; s < s1; ++s) {
        const uint gp = lo + s;
        uint ps;
        if (ring_mask > 0u) { ps = gp & ring_mask; }
        else if (KV_PAGED) {
            if (s == s0 || (gp % KV_PAGE_CELLS) == 0u) { page = pt[gp / KV_PAGE_CELLS]; }
            ps = page * KV_PAGE_CELLS + gp % KV_PAGE_CELLS;
        } else { ps = gp; }
        float kf[DPL], vf[DPL];
        attn_load_kv<HD>(kc, vc, kvw, ps, kvh, lane, kf, vf);
#pragma unroll
        for (uint j = 0; j < HQ; ++j) {
            if (j < hq) {
                float sc = 0.0f;
                for (uint i = 0; i < DPL; ++i) { sc += qr[j * DPL + i] * kf[i]; }
                sc = simd_sum(sc);
                const float mn = max(m[j], sc);
                const float f = exp(m[j] - mn);
                const float pw = exp(sc - mn);
                l[j] = l[j] * f + pw;
                for (uint i = 0; i < DPL; ++i) { o[j * DPL + i] = o[j * DPL + i] * f + pw * vf[i]; }
                m[j] = mn;
            }
        }
    }
}

// DECODE ATTENTION AS A VECTOR KERNEL: ONE SIMDGROUP PER POSITION, LANES OWN DIMS.
//
// For a single query over a span whose K/V stays cache-resident (a windowed layer, a short
// context), the time is not DRAM traffic but the per-position loop and how many simdgroups
// keep it going: measured on a 256-dim head over 512 keys, the prefill-shaped route took
// 20-27 us and the grouped fd kernel above 25-44 us against an 8 us bandwidth floor. So: a
// query head per threadgroup, NSG simdgroups striding over the positions, each lane holding
// HD/32 dims of Q in registers and loading its own bytes of the K row (one coalesced row per
// simdgroup), ONE simd_sum per position, an online softmax kept per simdgroup, the V row
// accumulated the same way, and the simdgroups merged once through threadgroup memory. The
// normalised output is written directly: no partials, no combine. K is read once per QUERY
// head, which on a cache-resident span costs nothing and keeps the loop at one reduction
// per position; past the span limit the host's other routes (K once per KV head, sliced)
// take over -- `attn_vec_max_keys` is that limit and the tuner ranks it.
// Instantiated for every head-dim slot the model declares; any head dim that is a multiple
// of 32 fits (each lane owns HD/32 dims, loaded as half4 when that is a multiple of 4).
template <uint HD, uint KVW, uint NSG>
kernel void imparo_attention_decode_vec_t(
    device const float * q     [[buffer(0)]],
    device const half  * kc    [[buffer(1)]],
    device const half  * vc    [[buffer(2)]],
    device float       * out   [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]], constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],
    constant uint & ring_mask [[buffer(10)]],
    device const uint * pt [[buffer(18)]],
    constant CobRow * rows [[buffer(19), function_constant(COB_ROWS)]],
    threadgroup float * red [[threadgroup(0)]],   // (NSG / 2) x (HD + 2)
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]])
{
    static_assert(HD % 32u == 0u, "lanes own HD/32 dims");
    constexpr uint DPL = HD / 32u;
    const uint kvw = (KVW != 0u) ? KVW : kv_width;
    (void)head_dim;
    const uint h = tgid.x, t = tgid.y;
    const uint kvh = h / (n_heads / n_kv);
    uint pos = start_pos + t;
    device const uint * table = pt;
    if (COB_ROWS) { pos = rows[t].pos; table = rows[t].pt; }
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;

    float qr[DPL];
    {
        device const float * qp = q + ((ulong)t * n_heads + h) * HD + lane * DPL;
        for (uint i = 0; i < DPL; ++i) { qr[i] = qp[i]; }
    }
    float m = -INFINITY, l = 0.0f;
    float o[DPL];
    for (uint i = 0; i < DPL; ++i) { o[i] = 0.0f; }
    attn_span_online<HD, KVW>(kc, vc, kvw, kvh, lo, n, sgid, NSG, ring_mask, table, lane, qr, m, l, o);
    // ---- merge the simdgroups in two rounds over NSG/2 slots of (o, m, l): the upper half
    // hands its state to the lower half, then simdgroups 1.. hand theirs to simdgroup 0.
    // Half the slots keep the buffer inside the 32 KB threadgroup limit at 16 simdgroups
    // and hd 512 (16 x 514 floats would be 32.9 KB).
    constexpr uint HALF = NSG / 2u;
    static_assert(NSG % 2u == 0u && NSG >= 2u, "two-round merge");
    if (sgid >= HALF) { attn_store_slot<HD>(red + (sgid - HALF) * (HD + 2u), lane, m, l, o); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid < HALF) { attn_merge_slot<HD>(red + sgid * (HD + 2u), lane, m, l, o); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid > 0u && sgid < HALF) { attn_store_slot<HD>(red + sgid * (HD + 2u), lane, m, l, o); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float M, L, acc[DPL];
        attn_fold_slots<HD>(red, HALF, lane, m, l, o, M, L, acc);
        device float * op = out + ((ulong)t * n_heads + h) * HD + lane * DPL;
        const float inv = 1.0f / L;
        for (uint i = 0; i < DPL; ++i) { op[i] = acc[i] * inv; }
    }
}
#define IMPARO_VEC_INST(SLOT, HDV, KVWV) \
template [[host_name("imparo_attention_decode_vec_s" #SLOT)]] kernel void imparo_attention_decode_vec_t<HDV, KVWV, 16u>( \
    device const float *, device const half *, device const half *, device float *, constant uint &, constant uint &, \
    constant uint &, constant uint &, constant uint &, constant uint &, constant uint &, \
    device const uint *, constant CobRow *, threadgroup float *, uint3, uint, uint);
#if IMPARO_HD0 && (IMPARO_HD0 % 32 == 0)
IMPARO_VEC_INST(0, IMPARO_HD0, IMPARO_KVW0)
#endif
#if IMPARO_HD1 && (IMPARO_HD1 % 32 == 0)
IMPARO_VEC_INST(1, IMPARO_HD1, IMPARO_KVW1)
#endif
#undef IMPARO_VEC_INST

// THE COMBINE HAS TWO INDEPENDENT REDUNDANCIES and IMPARO_ATTN_COMB_STAGE prices them
// SEPARATELY, because a change with two effects measured as one bundle hides whichever of
// them is worthless:
//
//   0   the original: every thread rescans the per-slice headers from DEVICE memory, and
//       calls exp(m - mx) once per slice in BOTH the denom loop and the accumulate loop
//   1   the headers are staged in threadgroup memory once; the exp() calls stay per thread
//   2   the weights are also computed once into that staging, so exp() runs `slices` times
//       instead of 2 x slices x tcount
//
// All three produce the same bits: exp(m - mx) is deterministic, and no sum is reassociated.
// THREADS PER OUTPUT DIM in the combine (IMPARO_ATTN_COMB_SPD). NEGATIVE, kept as the arm
// that says so: SPD threads share a dim, each taking every SPD'th slice, with the partials
// meeting in simd shuffles -- SPD times the threads and an SPD times shorter chain, against
// a grid of n_heads x n_tok = 32 threadgroups (3.6 simdgroups a core on 18 cores) and a
// `slices`-deep serial dependent accumulate. It reads -0.57% at 64 decode steps and +0.35%
// at 160, which is the 64-step number being noise: at 17122 keys the run-to-run spread is
// 3.6%, so a sub-1% effect needs >= 128 steps to have a sign at all.
// Default 1. Bit-affecting when on (the accumulate splits into SPD partials).
constant uint ATTN_COMB_SPD_FC [[function_constant(35)]];
constant uint ATTN_COMB_SPD =
    is_function_constant_defined(ATTN_COMB_SPD_FC) ? ATTN_COMB_SPD_FC : 1u;
constant uint ATTN_COMB_STAGE_FC [[function_constant(33)]];
constant uint ATTN_COMB_STAGE =
    is_function_constant_defined(ATTN_COMB_STAGE_FC) ? ATTN_COMB_STAGE_FC : 2u;

// Merge the split-KV slices' unnormalised partials onto a common maximum and normalise.
//
// THE PER-SLICE HEADER IS STAGED, and that is the whole design. Every thread needs the same
// `slices` (max, sum) pairs and the same weight exp(m - mx) for every slice, and the loop that
// produced them used to be written out per thread against DEVICE memory:
//
//   before   2 prologue scans x slices x tcount device loads, and one exp() per thread per
//            slice in the main loop -- at 67 slices and 64 threads that is 8576 header loads
//            of which 134 are distinct, and 4288 exponentials of which 67 are distinct
//   after    the headers are staged once, cooperatively; the weights are computed once into
//            the same threadgroup array; the main loop reads a weight instead of recomputing
//            it and no longer re-reads m from device
//
// Bit-identical: exp(m - mx) is a deterministic function of values that do not change, so a
// staged weight equals the one the inner loop used to compute, and no sum is reassociated --
// mx, denom and acc keep their original order.
kernel void imparo_attention_decode_combine(
    device const float * part [[buffer(0)]], device float * out [[buffer(1)]],
    constant uint & head_dim [[buffer(2)]], constant uint & n_heads [[buffer(3)]],
    constant uint & slices   [[buffer(4)]],
    threadgroup float * hdr [[threadgroup(0)]],     // slices x (weight, sum)
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]])
{
    const uint tid = tid3.x, tcount = tcnt3.x;
    const uint h = tgid.x, t = tgid.y;
    const uint stride = head_dim + 2u;
    device const float * pb = part + (ulong)(t * n_heads + h) * slices * stride;

    device float * o = out + ((ulong)t * n_heads + h) * head_dim;

    if (ATTN_COMB_STAGE == 0u) {
        float mx = -INFINITY;
        for (uint sp = 0; sp < slices; ++sp) { mx = max(mx, pb[sp * stride + head_dim]); }
        float denom = 0.0f;
        for (uint sp = 0; sp < slices; ++sp) {
            const float m = pb[sp * stride + head_dim];
            denom += exp(m - mx) * pb[sp * stride + head_dim + 1u];
        }
        const float inv = 1.0f / denom;
        for (uint i = tid; i < head_dim; i += tcount) {
            float acc = 0.0f;
            for (uint sp = 0; sp < slices; ++sp) {
                const float m = pb[sp * stride + head_dim];
                acc += exp(m - mx) * pb[sp * stride + i];
            }
            o[i] = acc * inv;
        }
        return;
    }

    for (uint sp = tid; sp < slices; sp += tcount) {
        hdr[sp * 2u]      = pb[sp * stride + head_dim];
        hdr[sp * 2u + 1u] = pb[sp * stride + head_dim + 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -INFINITY;
    for (uint sp = 0; sp < slices; ++sp) { mx = max(mx, hdr[sp * 2u]); }

    // Arm 2 only: the weight replaces the max in place, ONCE, and everything downstream reads
    // it. Ordering the conversion before `denom` is what removes the second bank of exp()
    // calls as well as the accumulate loop's.
    if (ATTN_COMB_STAGE >= 2u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sp = tid; sp < slices; sp += tcount) { hdr[sp * 2u] = exp(hdr[sp * 2u] - mx); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float denom = 0.0f;
    for (uint sp = 0; sp < slices; ++sp) {
        const float w = (ATTN_COMB_STAGE >= 2u) ? hdr[sp * 2u] : exp(hdr[sp * 2u] - mx);
        denom += w * hdr[sp * 2u + 1u];
    }
    const float inv = 1.0f / denom;

    if (ATTN_COMB_SPD <= 1u) {
        for (uint i = tid; i < head_dim; i += tcount) {
            float acc = 0.0f;
            for (uint sp = 0; sp < slices; ++sp) {
                const float w = (ATTN_COMB_STAGE >= 2u) ? hdr[sp * 2u] : exp(hdr[sp * 2u] - mx);
                acc += w * pb[sp * stride + i];
            }
            o[i] = acc * inv;
        }
        return;
    }
    // SPD threads a dim. They are CONSECUTIVE lanes, so the join is a simd shuffle and needs
    // no threadgroup memory and no barrier; SPD is a power of two no greater than 32, so the
    // group never straddles two simdgroups.
    const uint pslot = tid % ATTN_COMB_SPD, d0 = tid / ATTN_COMB_SPD;
    const uint dstep = tcount / ATTN_COMB_SPD;
    for (uint i = d0; i < head_dim; i += dstep) {
        float acc = 0.0f;
        for (uint sp = pslot; sp < slices; sp += ATTN_COMB_SPD) {
            const float w = (ATTN_COMB_STAGE >= 2u) ? hdr[sp * 2u] : exp(hdr[sp * 2u] - mx);
            acc += w * pb[sp * stride + i];
        }
        for (uint m2 = 1u; m2 < ATTN_COMB_SPD; m2 <<= 1) { acc += simd_shuffle_xor(acc, m2); }
        if (pslot == 0u) { o[i] = acc * inv; }
    }
}

kernel void imparo_attention_decode_direct(
    device const float * q     [[buffer(0)]],
    device const half  * kc    [[buffer(1)]],
    device const half  * vc    [[buffer(2)]],
    device float       * out   [[buffer(3)]],
    constant uint & head_dim [[buffer(4)]], constant uint & n_heads [[buffer(5)]],
    constant uint & n_kv     [[buffer(6)]], constant uint & kv_width [[buffer(7)]],
    constant uint & start_pos [[buffer(8)]], constant uint & window [[buffer(9)]],
    constant uint & ring_mask [[buffer(10)]],
    device const uint * pt [[buffer(18)]],
    threadgroup float * scores [[threadgroup(0)]],
    threadgroup float * red    [[threadgroup(1)]],
    // all position attributes must share dimensionality; the simdgroup ones are
    // scalar-only and exempt, which is why uint3 + uint compiles here
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint3 tid3   [[thread_position_in_threadgroup]],
    uint3 tcnt3  [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  sgid   [[simdgroup_index_in_threadgroup]],
    uint  nsg    [[simdgroups_per_threadgroup]])
{
    const uint tid = tid3.x;
    const uint tcount = tcnt3.x;
    const uint h = tgid.x, t = tgid.y;
    const uint pos = start_pos + t;
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;
    const uint kvh = h / (n_heads / n_kv);
    device const float * qh = q + ((ulong)t * n_heads + h) * head_dim;

    // ONE SIMDGROUP PER POSITION for the Q.K scores.
    //
    // Previously one thread walked an entire K row while its neighbours walked rows
    // kv_width floats away, so every lane in a simdgroup touched a different 4 KB region
    // and nothing coalesced. Now the 32 lanes of a simdgroup sweep one K row together.
    for (uint s = sgid; s < n; s += nsg) {
        const uint ps = kv_slot(lo + s, ring_mask, pt);
        float acc = 0.0f;
        if (KVT_K == 1u) {
            device const half * k = kc + (ulong)ps * kv_width + kvh * head_dim;
            for (uint i = lane; i < head_dim; i += 32u) { acc += qh[i] * float(k[i]); }
        } else {
            device const uchar * kb = (device const uchar *)kc
                + imparo_kv_row_off(KVT_K, kv_width, ps, kvh * head_dim);
            for (uint i = lane; i < head_dim; i += 32u) {
                acc += qh[i] * imparo_kv_load1(KVT_K, kb, i);
            }
        }
        acc = simd_sum(acc);
        if (lane == 0) { scores[s] = acc; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax in parallel. It used to run three serial passes over every position on
    // thread 0 while the other 127 threads waited.
    float local_max = -INFINITY;
    for (uint s = tid; s < n; s += tcount) { local_max = max(local_max, scores[s]); }
    local_max = simd_max(local_max);
    if (lane == 0) { red[sgid] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint i = 0; i < nsg; ++i) { mx = max(mx, red[i]); }
        red[0] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mx = red[0];
    // Every thread has mx in a register before the sum phase reuses red[0]:
    // without this barrier a fast simdgroup 0 writes red[sgid=0] = local_sum
    // while a slow simdgroup is still reading red[0] as the max -- exp(s - sum)
    // instead of exp(s - max), decided by warp scheduling. That was the decode
    // nondeterminism (task #26). The GQA variant dodges it with red[nsg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < n; s += tcount) {
        const float e = exp(scores[s] - mx);
        scores[s] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) { red[sgid] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < nsg; ++i) { total += red[i]; }
        red[0] = 1.0f / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float insum = red[0];
    for (uint s = tid; s < n; s += tcount) { scores[s] *= insum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * o = out + ((ulong)t * n_heads + h) * head_dim;
    for (uint i = tid; i < head_dim; i += tcount) {
        float acc = 0.0f;
        for (uint s = 0; s < n; ++s) {
            const uint ps = kv_slot(lo + s, ring_mask, pt);
            if (KVT_V == 1u) {
                acc += scores[s] * float(vc[(ulong)ps * kv_width + kvh * head_dim + i]);
            } else {
                acc += scores[s] * imparo_kv_load1(KVT_V, (device const uchar *)vc
                        + imparo_kv_row_off(KVT_V, kv_width, ps, kvh * head_dim), i);
            }
        }
        o[i] = acc;
    }
}


// tanh-approximation GELU, guarded.
//
// Metal's tanh overflows for large arguments: it evaluates (exp(2x)-1)/(exp(2x)+1), and
// exp(2*313) is Inf in f32, so Inf/Inf = NaN. A gate value of 20.27 in layer 0 reaches an
// argument of ~313 and produced exactly one NaN, which then poisoned a whole row of the
// following matmul. tanh saturates to 1 well before 15, so clamping is exact, not an
// approximation.

// The DECODE form of the gated epilogue. Prefill folds this into the up projection's
// write-back; at one token that measured worse -- a read-modify-write per output row
// inside the GEMV against a wide vectorised pass here -- so decode keeps the two steps.
// `kind` is EPI_GELU or EPI_SILU, the same wire values the fused epilogue takes.
// A CAUSAL DEPTHWISE CONVOLUTION over per-conversation history. `imparo_cpu::ops::
// causal_conv` is the oracle these are checked against, and carries the derivation:
//
//   value  = the form's value            per token per channel
//   seq    = state ++ value              causal: the state is PREPENDED
//   sum[t] = sum_k conv_w[ch][k] * seq[t + k]
//   out[t] = the form's epilogue of sum[t]
//   state' = the last (kernel - 1) values of seq
//
// TWO FORMS, and nothing else differs between LFM2's gated short convolution and
// Qwen3.8's delta-net convolution:
//
//   CONV_GATED       src row = [b | c | x]   value = b * x   out = c * sum
//   CONV_PLAIN_SILU  src row = [x]           value = x       out = silu(sum)
//
// The form is a TEMPLATE parameter, not a kernel argument: a runtime branch in a Metal
// kernel costs speed and moves the floating-point answer (see EPI_ACT_FC above). Each
// instantiation compiles exactly the arithmetic it had as its own kernel.
//
// TWO kernels per form, because the new state is the TAIL of the same sequence the
// outputs read. One dispatch would have threads writing state slots other threads still
// need, and a kernel cannot barrier its own grid -- the dispatch boundary is the barrier.
//
// `conv_w` is channel-major with the tap fastest: element (k, ch) is at ch * kernel + k,
// which is what the GGUF dims (kernel, width) mean. Tap 0 multiplies the OLDEST value.
constant uint CONV_GATED = 0u;
constant uint CONV_PLAIN_SILU = 1u;

template <uint FORM> inline uint conv_src_stride(uint width) {
    return FORM == CONV_GATED ? 3u * width : width;
}
// The value at source row `row` (an element offset already scaled by the form's stride).
template <uint FORM>
inline float conv_value(device const float * src, uint width, ulong row, uint ch) {
    return FORM == CONV_GATED ? src[row + ch] * src[row + 2u * width + ch] : src[row + ch];
}
template <uint FORM>
inline float conv_epilogue(device const float * src, uint width, ulong row, uint ch, float acc) {
    return FORM == CONV_GATED ? src[row + width + ch] * acc : imparo_silu_f(acc);
}

template <uint FORM>
inline void causal_conv_body(
    device const float * src, device const float * cw, device const float * state,
    device float * out, uint width, uint kern, uint n_tok,
    device half * outh, uint xh_on, uint gid)
{
    if (gid >= n_tok * width) { return; }
    const uint t = gid / width, ch = gid % width;
    const uint history = kern - 1u;
    const uint stride = conv_src_stride<FORM>(width);
    // Accumulated in tap order, oldest first, the same order the CPU reference uses.
    float acc = 0.0f;
    for (uint k = 0; k < kern; ++k) {
        const uint e = t + k;
        float v;
        if (e < history) {
            v = state[e * width + ch];
        } else {
            v = conv_value<FORM>(src, width, (ulong)(e - history) * stride, ch);
        }
        acc += cw[ch * kern + k] * v;
    }
    const float v = conv_epilogue<FORM>(src, width, (ulong)t * stride, ch, acc);
    out[(ulong)t * width + ch] = v;
    if (xh_on != 0u) { outh[(ulong)t * width + ch] = half(v); }
}

kernel void imparo_causal_conv_gated(
    device const float * src   [[buffer(0)]],
    device const float * cw    [[buffer(1)]],
    device const float * state [[buffer(2)]],
    device float * out         [[buffer(3)]],
    constant uint & width      [[buffer(4)]],
    constant uint & kern       [[buffer(5)]],
    constant uint & n_tok      [[buffer(6)]],
    // Optional half mirror of `out` (IMPARO_HALF_A): the out_proj GEMM's activation copy,
    // written here -- same rounding, same flat [token][channel] slots as the cvt pass --
    // so the separate conversion dispatch (one per conv block per chunk, 22 on LFM2,
    // 138 MB per 512-token chunk) disappears, as it did for the norm and add producers.
    device half * outh         [[buffer(7)]],
    constant uint & xh_on      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    causal_conv_body<CONV_GATED>(src, cw, state, out, width, kern, n_tok, outh, xh_on, gid);
}

kernel void imparo_causal_conv_plain(
    device const float * src   [[buffer(0)]],
    device const float * cw    [[buffer(1)]],
    device const float * state [[buffer(2)]],
    device float * out         [[buffer(3)]],
    constant uint & width      [[buffer(4)]],
    constant uint & kern       [[buffer(5)]],
    constant uint & n_tok      [[buffer(6)]],
    device half * outh         [[buffer(7)]],
    constant uint & xh_on      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    causal_conv_body<CONV_PLAIN_SILU>(src, cw, state, out, width, kern, n_tok, outh, xh_on, gid);
}

// The tail of `seq` becomes the new state. ONE THREAD PER CHANNEL, reading every value it
// needs before writing any: for a batch SHORTER than the history the new state is a shift
// of the old one, and a thread that wrote first would clobber a slot it still has to read.
// One thread per channel makes that a register question instead of a cross-thread one.
constant uint SHORTCONV_MAX_HISTORY = 8u;
// ONE CHANNEL of the conv step, shared by the standalone step kernels and the mega conv
// phase (task #158). TEMPLATED ON THE FORM for the same reason `causal_conv_body` is: the
// two forms differ only in how a source row yields a value and how the sum is finished, and
// writing either one twice is how the two copies drift.
//
// This is `causal_conv_body<FORM>` at t = 0, expressed through the same three helpers so
// that equivalence holds by construction rather than by inspection.
template <uint FORM>
inline float shortconv_channel_out(device const float * src, device const float * cw, device const float * state,
                                   uint width, uint kern, uint ch) {
    const uint history = kern - 1u;
    const uint stride = conv_src_stride<FORM>(width);
    float acc = 0.0f;
    for (uint k = 0u; k < kern; ++k) {
        float v;
        if (k < history) {
            v = state[k * width + ch];
        } else {
            v = conv_value<FORM>(src, width, (ulong)(k - history) * stride, ch);
        }
        acc += cw[ch * kern + k] * v;
    }
    return conv_epilogue<FORM>(src, width, 0u, ch, acc);
}
// The channel's history advanced by one token. IN AND OUT ARE SEPARATE: `state_in` is the
// live plane it carries values forward from, `state_out` is where the advanced row lands
// (the same pointer for the in-place form). Every value is read before any is written, so
// in == out is a safe shift and in != out is safe by construction. `snap` takes the same
// advanced row when a checkpoint is armed at this token.
//
// The in/out split is what lets a decode step write a DIFFERENT plane, so a failed step is
// rolled back by not advancing an index rather than by copying the state first (task #165);
// passing the out plane as the source instead would read a stale row, which is why these
// are two parameters and not one.
template <uint FORM>
inline void shortconv_channel_shift(device const float * src, device const float * state_in,
                                    device float * state_out, device float * snap, bool has_snap,
                                    uint width, uint history, uint ch) {
    const uint stride = conv_src_stride<FORM>(width);
    float next[SHORTCONV_MAX_HISTORY];
    for (uint sI = 0u; sI < history; ++sI) {
        const uint e = 1u + sI;
        if (e < history) {
            next[sI] = state_in[e * width + ch];
        } else {
            next[sI] = conv_value<FORM>(src, width, (ulong)(e - history) * stride, ch);
        }
    }
    for (uint sI = 0u; sI < history; ++sI) { state_out[sI * width + ch] = next[sI]; }
    if (has_snap) { for (uint sI = 0u; sI < history; ++sI) { snap[sI * width + ch] = next[sI]; } }
}
// ONE TOKEN: the conv output and the state shift in one dispatch, one thread per channel.
// The two kernels below are dependent through `state` (the conv reads it, the shift writes
// it), so nothing overlapped before and nothing is lost; the thread reads every state value
// it needs before it writes any. Same expressions in the same order as the two kernels.
// `state_out` is the PLANE the advanced history lands in; the host binds the same buffer at
// the same offset for the in-place form. The shift already read every value it needs before
// writing any, so a separate destination is safe by construction and costs an address --
// which is what makes a failed decode step recoverable by not advancing an index instead of
// by copying the whole state before every step (task #165).
template <uint FORM>
inline void shortconv_step_body(
    device const float * src, device const float * cw, device float * state,
    device float * out, uint width, uint kern, device float * state_out, uint ch)
{
    if (ch >= width) { return; }
    out[ch] = shortconv_channel_out<FORM>(src, cw, state, width, kern, ch);
    shortconv_channel_shift<FORM>(src, state, state_out, nullptr, false, width,
                                  min(kern - 1u, SHORTCONV_MAX_HISTORY), ch);
}

// Co-batched (COB_ROWS): grid row t is a conversation's token, its source and output at row t,
// its history in its own state (rows[t]).
template <uint FORM>
inline void shortconv_step_rows(
    device const float * src, device const float * cw, device float * out, uint width,
    uint kern, constant CobRow * rows, uint2 gid)
{
    const uint t = gid.y;
    shortconv_step_body<FORM>(src + (ulong)t * conv_src_stride<FORM>(width), cw, rows[t].state,
                              out + (ulong)t * width, width, kern, rows[t].state_out, gid.x);
}

kernel void imparo_shortconv_step(
    device const float * bcx   [[buffer(0)]],
    device const float * cw    [[buffer(1)]],
    device float       * state [[buffer(2)]],
    device float       * out   [[buffer(3)]],
    constant uint & width      [[buffer(4)]],
    constant uint & kern       [[buffer(5)]],
    device float       * state_out [[buffer(6)]],
    constant CobRow * rows [[buffer(7), function_constant(COB_ROWS)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (COB_ROWS) {
        shortconv_step_rows<CONV_GATED>(bcx, cw, out, width, kern, rows, gid);
    } else {
        shortconv_step_body<CONV_GATED>(bcx, cw, state, out, width, kern, state_out, gid.x);
    }
}

// The PLAIN form's twin. qwen35's gated delta-net conv is `CONV_PLAIN_SILU`, and without
// this kernel its decode step ran the conv output and the state shift as TWO dispatches --
// 48 dependent boundaries a token, on a route where LFM2's gated form has folded them into
// one since task #117. Same two ops, same order, one dispatch.
kernel void imparo_shortconv_step_plain(
    device const float * src   [[buffer(0)]],
    device const float * cw    [[buffer(1)]],
    device float       * state [[buffer(2)]],
    device float       * out   [[buffer(3)]],
    constant uint & width      [[buffer(4)]],
    constant uint & kern       [[buffer(5)]],
    device float       * state_out [[buffer(6)]],
    constant CobRow * rows [[buffer(7), function_constant(COB_ROWS)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (COB_ROWS) {
        shortconv_step_rows<CONV_PLAIN_SILU>(src, cw, out, width, kern, rows, gid);
    } else {
        shortconv_step_body<CONV_PLAIN_SILU>(src, cw, state, out, width, kern, state_out, gid.x);
    }
}

// SEPARATE in and out, which is what lets the same kernel serve two jobs:
//
//   advance    state_in == state_out, n_tok = the whole chunk  -> the live state moves on
//   snapshot   state_out is a scratch buffer, n_tok = tokens up to a BOUNDARY inside the
//              chunk -> the state as of that boundary, without stopping the batch there
//
// The second is why the batch does not have to be cut at a unit boundary. The state at
// any position is just the last (kernel - 1) values of the form's value sequence ending
// there, and `src` already holds them -- so it is computed, not stood on. The reference
// reaches the same place from the other side: its scan kernel writes rollback planes as it
// goes rather than having the caller stop.
template <uint FORM>
inline void causal_conv_state_body(
    device const float * src, device const float * state_in, device float * state_out,
    uint width, uint kern, uint n_tok, uint ch)
{
    if (ch >= width) { return; }
    const uint history = min(kern - 1u, SHORTCONV_MAX_HISTORY);
    const uint stride = conv_src_stride<FORM>(width);
    float next[SHORTCONV_MAX_HISTORY];
    for (uint sI = 0; sI < history; ++sI) {
        const uint e = n_tok + sI;
        if (e < history) {
            next[sI] = state_in[e * width + ch];
        } else {
            next[sI] = conv_value<FORM>(src, width, (ulong)(e - history) * stride, ch);
        }
    }
    // Still read-all-then-write-all: with in == out and a batch shorter than the history
    // this is a shift, and a thread that wrote first would clobber a slot it still needs.
    for (uint sI = 0; sI < history; ++sI) { state_out[sI * width + ch] = next[sI]; }
}

kernel void imparo_causal_conv_state_gated(
    device const float * src       [[buffer(0)]],
    device const float * state_in  [[buffer(1)]],
    device float * state_out       [[buffer(2)]],
    constant uint & width          [[buffer(3)]],
    constant uint & kern           [[buffer(4)]],
    constant uint & n_tok          [[buffer(5)]],
    uint ch [[thread_position_in_grid]])
{
    causal_conv_state_body<CONV_GATED>(src, state_in, state_out, width, kern, n_tok, ch);
}

kernel void imparo_causal_conv_state_plain(
    device const float * src       [[buffer(0)]],
    device const float * state_in  [[buffer(1)]],
    device float * state_out       [[buffer(2)]],
    constant uint & width          [[buffer(3)]],
    constant uint & kern           [[buffer(4)]],
    constant uint & n_tok          [[buffer(5)]],
    uint ch [[thread_position_in_grid]])
{
    causal_conv_state_body<CONV_PLAIN_SILU>(src, state_in, state_out, width, kern, n_tok, ch);
}

// ROW-LAYOUT CONVOLUTION (a tree verify). The value `back` steps up row t's own path is a batch
// row while the path stays in the batch and the committed state once it leaves (state slot
// history + depth - back, the slot the chain body reads), so a chain layout reads exactly what
// causal_conv_body reads, in the same tap order.
template <uint FORM>
inline float conv_rows_value(device const float * src, device const float * state,
                             device const uint * layout, uint width, uint stride,
                             uint history, uint t, uint back, uint ch) {
    const uint depth = layout[t * ROW_LAYOUT_WORDS + 1u];
    if (back > depth) {
        return state[(history + depth - back) * width + ch];
    }
    const uint row = back == 0u ? t : layout[t * ROW_LAYOUT_WORDS + 3u + back];
    return conv_value<FORM>(src, width, (ulong)row * stride, ch);
}

// Outputs only: tap k of row t reads the value history - k steps up its path. The state is
// read, never advanced -- conv_row_inputs keeps each row's input for the commit.
template <uint FORM>
inline void causal_conv_rows_body(
    device const float * src, device const float * cw, device const float * state,
    device float * out, uint width, uint kern, uint n_tok,
    device half * outh, uint xh_on, device const uint * layout, uint gid)
{
    if (gid >= n_tok * width) { return; }
    const uint t = gid / width, ch = gid % width;
    const uint history = kern - 1u;
    const uint stride = conv_src_stride<FORM>(width);
    float acc = 0.0f;
    for (uint k = 0; k < kern; ++k) {
        const float v = conv_rows_value<FORM>(src, state, layout, width, stride, history, t,
                                              history - k, ch);
        acc += cw[ch * kern + k] * v;
    }
    const float v = conv_epilogue<FORM>(src, width, (ulong)t * stride, ch, acc);
    out[(ulong)t * width + ch] = v;
    if (xh_on != 0u) { outh[(ulong)t * width + ch] = half(v); }
}

kernel void imparo_causal_conv_rows_gated(
    device const float * src    [[buffer(0)]],
    device const float * cw     [[buffer(1)]],
    device const float * state  [[buffer(2)]],
    device float * out          [[buffer(3)]],
    constant uint & width       [[buffer(4)]],
    constant uint & kern        [[buffer(5)]],
    constant uint & n_tok       [[buffer(6)]],
    device half * outh          [[buffer(7)]],
    constant uint & xh_on       [[buffer(8)]],
    device const uint * layout  [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    causal_conv_rows_body<CONV_GATED>(src, cw, state, out, width, kern, n_tok, outh, xh_on,
                                      layout, gid);
}

kernel void imparo_causal_conv_rows_plain(
    device const float * src    [[buffer(0)]],
    device const float * cw     [[buffer(1)]],
    device const float * state  [[buffer(2)]],
    device float * out          [[buffer(3)]],
    constant uint & width       [[buffer(4)]],
    constant uint & kern        [[buffer(5)]],
    constant uint & n_tok       [[buffer(6)]],
    device half * outh          [[buffer(7)]],
    constant uint & xh_on       [[buffer(8)]],
    device const uint * layout  [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    causal_conv_rows_body<CONV_PLAIN_SILU>(src, cw, state, out, width, kern, n_tok, outh, xh_on,
                                           layout, gid);
}

// Each batch row's input to the window: the value causal_conv_body shifts into the state for
// that row, written at row * row_elems. A tree commit rebuilds the accepted path's window from
// these rows, so no row keeps a whole window.
template <uint FORM>
inline void conv_row_inputs_body(
    device const float * src, device float * inputs, uint width, uint row_elems, uint n_tok,
    uint gid)
{
    if (gid >= n_tok * width) { return; }
    const uint t = gid / width, ch = gid % width;
    inputs[(ulong)t * row_elems + ch] =
        conv_value<FORM>(src, width, (ulong)t * conv_src_stride<FORM>(width), ch);
}

kernel void imparo_conv_row_inputs_gated(
    device const float * src    [[buffer(0)]],
    device float * inputs       [[buffer(1)]],
    constant uint & width       [[buffer(2)]],
    constant uint & row_elems   [[buffer(3)]],
    constant uint & n_tok       [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    conv_row_inputs_body<CONV_GATED>(src, inputs, width, row_elems, n_tok, gid);
}

kernel void imparo_conv_row_inputs_plain(
    device const float * src    [[buffer(0)]],
    device float * inputs       [[buffer(1)]],
    constant uint & width       [[buffer(2)]],
    constant uint & row_elems   [[buffer(3)]],
    constant uint & n_tok       [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    conv_row_inputs_body<CONV_PLAIN_SILU>(src, inputs, width, row_elems, n_tok, gid);
}

// THE GATED DELTA RULE. `imparo_cpu::ops::delta_net` is the oracle this is checked
// against, and carries the derivation. Per value head h, reading key head h % k_heads:
//
//   S      *= exp(g[h])                   g = a[h] * softplus(alpha[h] + dt_bias[h])
//   sk[j]   = SUM_i S[j][i] * khat[i]     what the state already remembers of k
//   d[j]    = (v[j] - sk[j]) * beta[h]    beta = sigmoid(the beta projection)
//   S[j][i]+= khat[i] * d[j]              rank-one update
//   o[j]    = SUM_i S[j][i] * qhat[i]     read it back with the query
//
// ONE THREADGROUP PER VALUE HEAD, holding that head's whole state matrix in REGISTERS and
// looping the batch's tokens inside the kernel. The state is value_dim x key_dim floats
// per head -- 64 KB at 128x128, four times what a threadgroup may allocate -- and it is
// read and written once per token. Loading it from device memory per token would move
// hundreds of megabytes per forward; held in registers it is loaded once and stored once
// per dispatch, whatever the token count.
//
// The division of labour: simdgroup s owns state ROWS s, s+32, s+64, ...; lane l owns
// COLUMNS l*DCOLS .. l*DCOLS+DCOLS-1 of each of them. Both reductions over the key
// coordinate are therefore one `simd_sum` per row, and every lane of a simdgroup runs
// them, so control flow stays uniform.
//
// DKD / DVD are compiled in, because a register array's size must be a constant
// expression -- the same reason the attention head dims are injected at library compile.
#ifndef IMPARO_DELTA_KD
#define IMPARO_DELTA_KD 32
#endif
#ifndef IMPARO_DELTA_VD
#define IMPARO_DELTA_VD 32
#endif
// TOKENS STAGED PER GROUP. The recurrence is serial in the token, but the READ of a
// token's q/k/v is not: nothing about token t+1's row depends on token t's answer. Staged
// one at a time the kernel pays 512 dependent device reads per head, each one exposed --
// four simdgroups fetch while the other 28 wait at the barrier, and the next fetch cannot
// start until the current token's rank-one update has finished. Staged in groups of
// DSTAGE the reads of a group are issued together (four simdgroups per staged token, all
// 32 busy at DSTAGE = 8) and the group's recurrence then runs out of threadgroup memory.
// Cost is DSTAGE * (2*DKD + DVD + 2) floats of threadgroup memory.
// IMPARO_DELTA_STAGE=1 is exactly the per-token form this replaced.
#ifndef IMPARO_DELTA_STAGE
#define IMPARO_DELTA_STAGE 8
#endif
// Whether a token's DROWS state rows have their cross-lane reductions issued together.
// See the loop for the reason; 0 is the row-at-a-time form.
#ifndef IMPARO_DELTA_PIPE
#define IMPARO_DELTA_PIPE 1
#endif
constant uint DKD = IMPARO_DELTA_KD;              // key coordinate = Q/K head width
constant uint DVD = IMPARO_DELTA_VD;              // value coordinate = V head width
// SIMDGROUPS PER THREADGROUP. 32, and it is not a tuning question: the state is
// DVD x DKD floats however this is set and there is ONE threadgroup per value head, so
// this IS the thread count. Halving it halves the threads and doubles what each carries --
// less parallelism for the same work, which is what it measures. BOTH REGIMES AGREE, so
// there is no crossover to tune and no knob:
//
//   prefill, 512 tokens, rule cost, two rounds    98.6 ms at 32, 165.2 at 16, 212.4 at 8
//   decode, rule by skip-and-diff, three rounds    3.11 ms/token at 32, 3.55 at 16, 4.20 at 8
//
// Decode was re-tested on purpose (2026-09-09): the prefill number is earned by the
// register-resident state amortising across the token loop and decode has no token loop,
// so the REASON for 32 does not carry even though the answer does. More rows per simdgroup
// at unchanged parallelism means more value HEADS per threadgroup, a different change.
// The simdgroup count is a build-time macro (IMPARO_DELTA_SGS, injected by the host with
// the staging depth) so the threadgroup's width can be re-measured beside the projections
// it is now issued next to: a 1024-thread threadgroup fills a core, and a kernel that
// fills 14 cores twice over leaves the GEMV beside it nowhere to run.
#ifndef IMPARO_DELTA_SGS
#define IMPARO_DELTA_SGS 32
#endif
constant uint DELTA_SGS = IMPARO_DELTA_SGS;
constant uint DELTA_TPT = DELTA_SGS * 32u;
constant uint DSTAGE = IMPARO_DELTA_STAGE;        // tokens staged per group
constant uint DROWS = (DVD + DELTA_SGS - 1u) / DELTA_SGS;   // state rows per simdgroup
constant uint DCOLS = (DKD + 31u) / 32u;                    // state columns per lane

// `ln(1 + exp(x))`, saturating to x where the two are equal in f32 -- the same cutoff the
// CPU reference uses, and for the same reason: above ~20 they differ by less than an ulp
// while exp(x) is still far from overflowing.
inline float imparo_softplus_f(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}
// Logistic sigmoid, per-sign so exp never overflows for large negative x.
inline float imparo_sigmoid_f(float x) {
    if (x >= 0.0f) { return 1.0f / (1.0f + exp(-x)); }
    const float e = exp(x);
    return e / (1.0f + e);
}

kernel void imparo_delta_net(
    device const float * qkv    [[buffer(0)]],
    device const float * alpha  [[buffer(1)]],
    device const float * beta   [[buffer(2)]],
    device const float * wa     [[buffer(3)]],
    device const float * wdt    [[buffer(4)]],
    device const float * state  [[buffer(5)]],
    device float * out          [[buffer(6)]],
    constant uint & k_heads     [[buffer(7)]],
    constant uint & v_heads     [[buffer(8)]],
    constant uint & n_tok       [[buffer(9)]],
    constant float & eps        [[buffer(10)]],
    // THE PLANE THE UPDATED MATRIX LANDS IN. The host binds the same buffer at the same
    // offset for the in-place form. Every element the kernel loads is stored back, so
    // aiming the store at another plane costs an address and no traffic -- which is what
    // makes a failed decode step recoverable by not advancing an index (task #165).
    device float * state_out    [[buffer(11)]],
    // THE GATED-RMS EPILOGUE, folded in. `nrm` is one head's norm weight (DVD floats,
    // shared by every head) and `gate` is the SiLU gate, read and written in place; with
    // `fuse_epi` off both are bound to stand-ins and the rule writes `out` as before.
    device const float * nrm    [[buffer(12)]],
    device float * gate         [[buffer(13)]],
    constant uint & fuse_epi    [[buffer(14)]],
    // THE BOUNDARY SNAPSHOT. A checkpoint boundary `snap_row` tokens into this batch:
    // the matrix as it stands after that token's update is written to `snap` (the
    // snapshot plane's copy of this layer's region) and the rule carries on. 0 = none.
    // The matrix lives in registers for the whole batch and is stored once at the end,
    // so without this a checkpoint inside a chunk carried the convolution history and
    // NO matrix -- a conversation adopting another's prefix ran from zeros.
    device float * snap         [[buffer(15)]],
    constant uint & snap_row    [[buffer(16)]],
    // CO-BATCHED ROWS (COB_ROWS): grid row tg.y is a conversation's single token. Its q/k/v,
    // alpha, beta, output and gate sit at that row of their buffers, and its matrix lives in its
    // OWN slot's state, a different Metal buffer per row -- so those two addresses come from the
    // table rather than from a binding. `snap` and `snap_row` are not used in this form.
    constant CobRow * rows [[buffer(17), function_constant(COB_ROWS)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint h = tg.x;
    if (h >= v_heads) { return; }
    const uint kw = k_heads * DKD;                       // Q width, and K width
    const uint qkv_width = 2u * kw + v_heads * DVD;
    // This row's view of the step. Without COB_ROWS the row is 0 and every pointer is its binding.
    const uint crow = COB_ROWS ? tg.y : 0u;
    device const float * qkv_r   = qkv   + (ulong)crow * qkv_width;
    device const float * alpha_r = alpha + (ulong)crow * v_heads;
    device const float * beta_r  = beta  + (ulong)crow * v_heads;
    device float       * out_r   = out   + (ulong)crow * v_heads * DVD;
    device float       * gate_r  = gate  + (ulong)crow * v_heads * DVD;
    device const float * st_in   = COB_ROWS ? (device const float *)rows[crow].state : state;
    device float       * st_out  = COB_ROWS ? rows[crow].state_out : state_out;
    // KEY HEAD MAPPING: the reference widens Q and K with a repeat, and a repeat TILES,
    // so value head h reads key head h % k_heads. A grouped-query attention in the same
    // model uses h / group; the two genuinely differ.
    const uint kb = (h % k_heads) * DKD;
    const uint vb = h * DVD;
    device const float * S = st_in + (ulong)h * DVD * DKD;
    device float * S_out = st_out + (ulong)h * DVD * DKD;

    threadgroup float tq[DSTAGE * DKD];
    threadgroup float tk[DSTAGE * DKD];
    threadgroup float tv[DSTAGE * DVD];
    threadgroup float tsc[DSTAGE * 2];                   // decay, beta
    // The group's OUTPUT rows, when the epilogue is fused. The rule leaves element j of
    // token `slot` in lane 0 of simdgroup `j % DELTA_SGS`, and the norm reduces over all
    // DVD of them, so they have to meet somewhere: 4 KB of threadgroup memory at
    // DSTAGE 8, against the 32 KB a core has.
    threadgroup float tout[DSTAGE * DVD];
    // WRITTEN THROUGH A VOLATILE POINTER, and that is a correctness requirement.
    //
    // The rule leaves `acc[r]` here and the epilogue reads it back. For the elements where
    // the writing thread is also the reading one -- lane 0 of simdgroup 0 writes j = 0,
    // which the epilogue's lane 0 reads -- the compiler may forward the value without ever
    // materialising it as a rounded f32, and those elements alone then differ from what the
    // separate rms_norm dispatch computes after a real store.
    //
    // MEASURED, and this is how it was localised: with a diagnostic device store added
    // alongside (so the value had to round), the fused arm was bit-identical to the plain
    // one -- delta_core 0136b50fdbb5a74e and delta_gated 6181c9d2dec3b21b both ways. Take
    // the store away and only the gated checksum moved. The store was doing nothing except
    // forcing the rounding, which is what `volatile` asks for directly.
    threadgroup volatile float * tout_w = (threadgroup volatile float *)tout;

    float s[DROWS][DCOLS];
    for (uint r = 0u; r < DROWS; ++r) {
        const uint j = sg + r * DELTA_SGS;
        for (uint c = 0u; c < DCOLS; ++c) {
            const uint i = lane * DCOLS + c;
            s[r][c] = (j < DVD && i < DKD) ? S[(ulong)j * DKD + i] : 0.0f;
        }
    }

    const float qscale = 1.0f / sqrt(float(DKD));
    // FOUR SIMDGROUPS PER STAGED TOKEN: `slot` says which of the group's tokens this
    // simdgroup fetches, `role` says which quarter of it (query, key, value, scalars).
    // The slot STRIDES, so DSTAGE is bounded by threadgroup memory and not by the
    // simdgroup count: at DSTAGE 8 every simdgroup takes one slot, above it some take two.
    const uint slot0 = sg / 4u, role = sg % 4u, slot_stride = DELTA_SGS / 4u;
    for (uint t0 = 0u; t0 < n_tok; t0 += DSTAGE) {
        const uint tn = min(DSTAGE, n_tok - t0);
        // STAGE the group. Every simdgroup whose slot lands inside the group issues its
        // read now, so the group's rows are in flight together instead of one at a time.
        // Every `simd_sum` below is reached by all 32 lanes of its simdgroup.
        for (uint slot = slot0; slot < tn; slot += slot_stride) {
            const uint t = t0 + slot;
            device const float * qrow = qkv_r + (ulong)t * qkv_width;
            if (role == 0u || role == 1u) {
                device const float * src = qrow + (role == 0u ? kb : kw + kb);
                float sq = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    const uint i = lane * DCOLS + c;
                    const float x = i < DKD ? src[i] : 0.0f;
                    sq += x * x;
                }
                // `max(norm, eps)`, not `norm + eps`: the second shrinks every vector
                // slightly, which is a different function.
                const float inv = 1.0f / max(sqrt(simd_sum(sq)), eps);
                for (uint c = 0u; c < DCOLS; ++c) {
                    const uint i = lane * DCOLS + c;
                    if (i < DKD) {
                        // The query is scaled by 1/sqrt(key_dim) HERE, before the state
                        // product, which is where the reference scales and rounds.
                        const float v = src[i] * inv;
                        if (role == 0u) { tq[slot * DKD + i] = v * qscale; }
                        else            { tk[slot * DKD + i] = v; }
                    }
                }
            } else if (role == 2u) {
                device const float * src = qrow + 2u * kw + vb;
                for (uint c = 0u; c < DCOLS; ++c) {
                    const uint i = lane * DCOLS + c;
                    if (i < DVD) { tv[slot * DVD + i] = src[i]; }
                }
            } else if (lane == 0u) {
                // `wa` is already -exp(A_log), so this is the NEGATIVE log decay.
                const float g = wa[h] * imparo_softplus_f(alpha_r[(ulong)t * v_heads + h] + wdt[h]);
                tsc[slot * 2u + 0u] = exp(g);
                tsc[slot * 2u + 1u] = imparo_sigmoid_f(beta_r[(ulong)t * v_heads + h]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The group's recurrence, out of threadgroup memory: no device read in this loop.
        //
        // THE ROWS ARE ISSUED TOGETHER. A simdgroup owns DROWS state rows and they are
        // independent within a token -- nothing in row r's update reads row r'. Written
        // as one loop over r, each `simd_sum` is CONSUMED before the next is issued, so
        // the row's whole cross-lane latency sits on the critical path DROWS times per
        // token. Split into phases, all DROWS reductions of a phase are in flight at
        // once and the token costs one latency instead of DROWS. Same operations in the
        // same order on every element, so the answer does not move.
        // IMPARO_DELTA_PIPE=0 is the row-at-a-time form this replaced.
        for (uint tt = 0u; tt < tn; ++tt) {
            const float decay = tsc[tt * 2u + 0u], bt = tsc[tt * 2u + 1u];
            float kv[DCOLS], qv[DCOLS];
            for (uint c = 0u; c < DCOLS; ++c) {
                const uint i = lane * DCOLS + c;
                kv[c] = i < DKD ? tk[tt * DKD + i] : 0.0f;
                qv[c] = i < DKD ? tq[tt * DKD + i] : 0.0f;
            }
#if IMPARO_DELTA_PIPE
            float acc[DROWS];
            for (uint r = 0u; r < DROWS; ++r) {
                acc[r] = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    s[r][c] *= decay;
                    acc[r] += s[r][c] * kv[c];
                }
            }
            for (uint r = 0u; r < DROWS; ++r) { acc[r] = simd_sum(acc[r]); }
            for (uint r = 0u; r < DROWS; ++r) {
                const uint j = sg + r * DELTA_SGS;
                const float d = ((j < DVD ? tv[tt * DVD + j] : 0.0f) - acc[r]) * bt;
                acc[r] = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    s[r][c] += kv[c] * d;
                    acc[r] += s[r][c] * qv[c];
                }
            }
            for (uint r = 0u; r < DROWS; ++r) { acc[r] = simd_sum(acc[r]); }
            if (lane == 0u) {
                for (uint r = 0u; r < DROWS; ++r) {
                    const uint j = sg + r * DELTA_SGS;
                    if (j < DVD) {
                        if (fuse_epi) { tout_w[tt * DVD + j] = acc[r]; }
                        else { out_r[(ulong)(t0 + tt) * v_heads * DVD + vb + j] = acc[r]; }
                    }
                }
            }
#else
            for (uint r = 0u; r < DROWS; ++r) {
                const uint j = sg + r * DELTA_SGS;
                float part = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    s[r][c] *= decay;
                    part += s[r][c] * kv[c];
                }
                const float sk = simd_sum(part);
                const float d = ((j < DVD ? tv[tt * DVD + j] : 0.0f) - sk) * bt;
                float proj = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    s[r][c] += kv[c] * d;
                    proj += s[r][c] * qv[c];
                }
                const float o = simd_sum(proj);
                if (lane == 0u && j < DVD) {
                    if (fuse_epi) { tout_w[tt * DVD + j] = o; }
                    else { out_r[(ulong)(t0 + tt) * v_heads * DVD + vb + j] = o; }
                }
            }
#endif
            // The state after this token IS the checkpoint's, when the boundary sits
            // here: every thread stores the elements it owns, the same rows and columns
            // the final store below covers. Uniform across the threadgroup; one 64 KB
            // store per head on the one token of a batch that reaches a boundary.
            if (t0 + tt + 1u == snap_row) {
                device float * S_snap = snap + (ulong)h * DVD * DKD;
                for (uint r = 0u; r < DROWS; ++r) {
                    const uint j = sg + r * DELTA_SGS;
                    for (uint c = 0u; c < DCOLS; ++c) {
                        const uint i = lane * DCOLS + c;
                        if (j < DVD && i < DKD) { S_snap[(ulong)j * DKD + i] = s[r][c]; }
                    }
                }
            }
        }
        // The staging arrays are rewritten by the next group, so no simdgroup may run
        // ahead into them while another is still reading. It also PUBLISHES tout, which
        // the epilogue below reads across simdgroups.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // THE EPILOGUE: gated RMS norm per (token, value head), then the SiLU gate.
        //
        //   rms_norm(Attn, ssm_norm, DVD, eps, b*v_heads, DVD, 0)   <- one dispatch
        //   act_mul(Z, Attn, b * v_heads * DVD)                     <- another
        //
        // BIT-IDENTICAL, and that is a constraint on the SHAPE, not a hope. At width DVD
        // the host gives imparo_rms_norm ceil(DVD/4) threads floored to 32, so a 128-wide
        // row runs ONE simdgroup: lane L owns float4 L, `dot(v,v)`, one `simd_sum`, then
        // `rsqrt(sq/DVD + eps)` and `core * inv * w`. One simdgroup per token here
        // reproduces that lane mapping exactly, so the reduction tree and every rounding
        // are the same. The gate then repeats act_mul's `act(g) * u` per component.
        //
        // ONE SIMDGROUP PER TOKEN, not four: the reduction is over the whole DVD row, and
        // splitting it across simdgroups would need a second threadgroup reduction with a
        // different summation order -- different bits for no work saved on 8 rows.
        if (fuse_epi) {
            // CONTRACTION OFF for the epilogue. The two dispatches this replaces are
            // small kernels; inlined into this large one the compiler is free to fuse a
            // multiply and an add that it left separate there, which rounds once instead
            // of twice and moves the last bit on a few elements per row (measured: rms,
            // sum and the leading elements match, the checksum does not). Same rule as
            // the gemma4 norm+add fold that had to be reverted for the same reason.
#pragma clang fp contract(off)
            for (uint slot = sg; slot < tn; slot += DELTA_SGS) {
                threadgroup const float * crow = tout + slot * DVD;
                const uint i0 = lane * 4u;
                float4 core = 0.0f;
                if (i0 < DVD) {
                    core = float4(crow[i0], crow[i0 + 1u], crow[i0 + 2u], crow[i0 + 3u]);
                }
                const float inv = rsqrt(simd_sum(dot(core, core)) / float(DVD) + eps);
                if (i0 < DVD) {
                    device float * grow = gate_r + (ulong)(t0 + slot) * v_heads * DVD + vb;
                    const float4 w = float4(nrm[i0], nrm[i0 + 1u],
                                            nrm[i0 + 2u], nrm[i0 + 3u]);
                    const float4 normed = core * inv * w;
                    const float4 g = float4(grow[i0], grow[i0 + 1u],
                                            grow[i0 + 2u], grow[i0 + 3u]);
                    const float4 out4 = float4(imparo_act_f(g.x), imparo_act_f(g.y),
                                               imparo_act_f(g.z), imparo_act_f(g.w))
                                      * normed;
                    grow[i0] = out4.x; grow[i0 + 1u] = out4.y;
                    grow[i0 + 2u] = out4.z; grow[i0 + 3u] = out4.w;
                }
            }
        }
    }

    for (uint r = 0u; r < DROWS; ++r) {
        const uint j = sg + r * DELTA_SGS;
        for (uint c = 0u; c < DCOLS; ++c) {
            const uint i = lane * DCOLS + c;
            if (j < DVD && i < DKD) { S_out[(ulong)j * DKD + i] = s[r][c]; }
        }
    }
}

// THE GATED DELTA RULE OVER A DRAFT TREE: a verify's read-only form (design 7.2). The rows are
// the tree's nodes depth-first, so every node follows its parent and the rows in between are
// its parent's earlier subtrees. S0, the state before the verify, is read and never written.
// Each node's output is what `imparo_delta_net` returns for it run along the node's own path
// from S0, in the same arithmetic.
//
// A THREAD WALKS THE TREE WITH ITS STATE BLOCK IN REGISTERS, as the serial kernel walks tokens:
// a node whose parent is the row before it continues from the block as it stands, so a chain
// runs exactly the serial kernel's loop. A later child -- a node whose parent is further back
// -- starts from the block its parent left, which the thread copied into `kept`, by depth, when
// that parent finished; the host marks the rows with more than one child (`keep_lo` / `keep_hi`,
// bit t for row t), and only those are copied. The host refuses a layout that is not depth-first
// or reaches DTREE_DEPTH.
//
// NOT THE MASKED SOLVE (each u from S0 k, S0 q and the key products with every ancestor), and
// NOT EVERY DEPTH'S BLOCK IN THREAD MEMORY: built first, they measured 1122 and 309 us per
// dispatch for a 16-node tree, where the serial kernel takes 98 over 16 rows
// (docs/evidence/dspark/2026-09-15-step6-gdn-tree-kernel.md).
#ifndef IMPARO_DELTA_TREE_DEPTH
#define IMPARO_DELTA_TREE_DEPTH 16
#endif
constant uint DTREE_DEPTH = IMPARO_DELTA_TREE_DEPTH;

kernel void imparo_delta_net_tree(
    device const float * qkv    [[buffer(0)]],
    device const float * alpha  [[buffer(1)]],
    device const float * beta   [[buffer(2)]],
    device const float * wa     [[buffer(3)]],
    device const float * wdt    [[buffer(4)]],
    device const float * state  [[buffer(5)]],
    device float * out          [[buffer(6)]],
    constant uint & k_heads     [[buffer(7)]],
    constant uint & v_heads     [[buffer(8)]],
    constant uint & n_tok       [[buffer(9)]],
    constant float & eps        [[buffer(10)]],
    device const uint * layout  [[buffer(11)]],
    constant uint & keep_lo     [[buffer(12)]],
    constant uint & keep_hi     [[buffer(13)]],
    uint h    [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (h >= v_heads) { return; }
    const uint kw = k_heads * DKD;                       // Q width, and K width
    const uint qkv_width = 2u * kw + v_heads * DVD;
    const uint kb = (h % k_heads) * DKD;                 // the serial kernel's key head mapping
    const uint vb = h * DVD;
    device const float * S = state + (ulong)h * DVD * DKD;

    threadgroup float tq[DSTAGE * DKD];
    threadgroup float tk[DSTAGE * DKD];
    threadgroup float tv[DSTAGE * DVD];
    threadgroup float tsc[DSTAGE * 2];                   // decay, beta
    threadgroup uint  tdep[DSTAGE];                      // depth

    float s[DROWS][DCOLS];
    for (uint r = 0u; r < DROWS; ++r) {
        const uint j = sg + r * DELTA_SGS;
        for (uint c = 0u; c < DCOLS; ++c) {
            const uint i = lane * DCOLS + c;
            s[r][c] = (j < DVD && i < DKD) ? S[(ulong)j * DKD + i] : 0.0f;
        }
    }
    float kept[DTREE_DEPTH + 1][DROWS][DCOLS];
    uint prev_d = 0u;

    const float qscale = 1.0f / sqrt(float(DKD));
    const uint slot0 = sg / 4u, role = sg % 4u, slot_stride = DELTA_SGS / 4u;
    for (uint t0 = 0u; t0 < n_tok; t0 += DSTAGE) {
        const uint tn = min(DSTAGE, n_tok - t0);
        // STAGE the group as the serial kernel does, each node's depth beside its gates.
        for (uint slot = slot0; slot < tn; slot += slot_stride) {
            const uint t = t0 + slot;
            device const float * row = qkv + (ulong)t * qkv_width;
            if (role == 0u || role == 1u) {
                device const float * src = row + (role == 0u ? kb : kw + kb);
                float sq = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    const uint i = lane * DCOLS + c;
                    const float x = i < DKD ? src[i] : 0.0f;
                    sq += x * x;
                }
                const float inv = 1.0f / max(sqrt(simd_sum(sq)), eps);
                for (uint c = 0u; c < DCOLS; ++c) {
                    const uint i = lane * DCOLS + c;
                    if (i < DKD) {
                        const float v = src[i] * inv;
                        if (role == 0u) { tq[slot * DKD + i] = v * qscale; }
                        else            { tk[slot * DKD + i] = v; }
                    }
                }
            } else if (role == 2u) {
                device const float * src = row + 2u * kw + vb;
                for (uint c = 0u; c < DCOLS; ++c) {
                    const uint i = lane * DCOLS + c;
                    if (i < DVD) { tv[slot * DVD + i] = src[i]; }
                }
            } else if (lane == 0u) {
                const float g = wa[h] * imparo_softplus_f(alpha[(ulong)t * v_heads + h] + wdt[h]);
                tsc[slot * 2u + 0u] = exp(g);
                tsc[slot * 2u + 1u] = imparo_sigmoid_f(beta[(ulong)t * v_heads + h]);
                tdep[slot] = layout[(ulong)t * ROW_LAYOUT_WORDS + 1u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tt = 0u; tt < tn; ++tt) {
            const uint t = t0 + tt;
            const float decay = tsc[tt * 2u + 0u], bt = tsc[tt * 2u + 1u];
            const uint d = min(tdep[tt], DTREE_DEPTH - 1u);   // inside `kept` whatever the layout
            // A later child starts from its parent's kept block; the root and a first child
            // continue from the block as it stands.
            if (t > 0u && d != prev_d + 1u) {
                for (uint r = 0u; r < DROWS; ++r) {
                    for (uint c = 0u; c < DCOLS; ++c) { s[r][c] = kept[d][r][c]; }
                }
            }
            float kv[DCOLS], qv[DCOLS];
            for (uint c = 0u; c < DCOLS; ++c) {
                const uint i = lane * DCOLS + c;
                kv[c] = i < DKD ? tk[tt * DKD + i] : 0.0f;
                qv[c] = i < DKD ? tq[tt * DKD + i] : 0.0f;
            }
            // The serial kernel's recurrence (its IMPARO_DELTA_PIPE form), unchanged.
            float acc[DROWS];
            for (uint r = 0u; r < DROWS; ++r) {
                acc[r] = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    s[r][c] *= decay;
                    acc[r] += s[r][c] * kv[c];
                }
            }
            for (uint r = 0u; r < DROWS; ++r) { acc[r] = simd_sum(acc[r]); }
            for (uint r = 0u; r < DROWS; ++r) {
                const uint j = sg + r * DELTA_SGS;
                const float dl = ((j < DVD ? tv[tt * DVD + j] : 0.0f) - acc[r]) * bt;
                acc[r] = 0.0f;
                for (uint c = 0u; c < DCOLS; ++c) {
                    s[r][c] += kv[c] * dl;
                    acc[r] += s[r][c] * qv[c];
                }
            }
            for (uint r = 0u; r < DROWS; ++r) { acc[r] = simd_sum(acc[r]); }
            if (lane == 0u) {
                for (uint r = 0u; r < DROWS; ++r) {
                    const uint j = sg + r * DELTA_SGS;
                    if (j < DVD) { out[(ulong)t * v_heads * DVD + vb + j] = acc[r]; }
                }
            }
            const uint keep = t < 32u ? (keep_lo >> t) & 1u : (keep_hi >> (t - 32u)) & 1u;
            if (keep != 0u) {
                for (uint r = 0u; r < DROWS; ++r) {
                    for (uint c = 0u; c < DCOLS; ++c) { kept[d + 1u][r][c] = s[r][c]; }
                }
            }
            prev_d = d;
        }
        // The staging arrays are rewritten by the next group.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// `a[r * a_stride + i] *= sigmoid(b[r * b_stride + b_off + i])` -- the strided sigmoid
// gate. Same shape as imparo_mul, which is the same op without the sigmoid; Qwen3.8's
// attention output gate reads the second half of each packed [query | gate] head, so the
// row stride and the sub-block offset are what make it one dispatch instead of a split.
kernel void imparo_mul_sigmoid(
    device float * a [[buffer(0)]], device const float * b [[buffer(1)]],
    constant uint & n [[buffer(2)]], constant uint & b_off [[buffer(3)]],
    constant uint & b_stride [[buffer(4)]], constant uint & a_stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= n) { return; }
    a[(ulong)t * a_stride + i] *= imparo_sigmoid_f(b[(ulong)t * b_stride + b_off + i]);
}

// `dst[r * width + i] = src[r * src_stride + src_off + i]` -- ONE sub-block out of every
// row. A projection that packs two tensors per head (Qwen3.8's Q weight is 24 heads of
// [query(256) | gate(256)]) is read apart with this; the default in the Backend trait
// walks the rows with copy_range, which is one dispatch per row.
kernel void imparo_copy_strided(
    device float * dst [[buffer(0)]], device const float * src [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & src_off [[buffer(3)]],
    constant uint & src_stride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, r = gid.y;
    if (i >= width) { return; }
    dst[(ulong)r * width + i] = src[(ulong)r * src_stride + src_off + i];
}

// `dst[r * dst_stride + dst_off + i] = src[r * width + i]` -- the inverse of
// imparo_copy_strided: contiguous rows written into ONE sub-block of every row. A drafter's
// feature taps put each tapped layer's residual beside the other taps' with it.
kernel void imparo_scatter_strided(
    device float * dst [[buffer(0)]], device const float * src [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & dst_off [[buffer(3)]],
    constant uint & dst_stride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, r = gid.y;
    if (i >= width) { return; }
    dst[(ulong)r * dst_stride + dst_off + i] = src[(ulong)r * width + i];
}

kernel void imparo_act_mul(
    device float * g [[buffer(0)]], device const float * u [[buffer(1)]],
    constant uint & n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    g[gid] = imparo_act_f(g[gid]) * u[gid];
}

// Ungated: the activation alone. gemma4's per-layer-embedding gate uses it.
kernel void imparo_act(
    device float * g [[buffer(0)]], constant uint & n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) { return; }
    g[gid] = imparo_act_f(g[gid]);
}

kernel void imparo_add(
    device float * a [[buffer(0)]], device const float * b [[buffer(1)]],
    constant uint & n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if (gid < n) { a[gid] += b[gid]; } }

// float4 forms of the elementwise kernels.
//
// These are dispatch-latency-bound at decode -- 339 of them per token -- but a quarter of
// the threads still costs a quarter of the execution, and the norm kernel showed the same
// change was worth two thirds of its time. Used only when the range divides by four and
// both pointers are 16-byte aligned, which the caller checks.
kernel void imparo_add4(
    device float4 * a [[buffer(0)]], device const float4 * b [[buffer(1)]],
    constant uint & n4 [[buffer(2)]], device half4 * xh [[buffer(3)]],
    constant uint & xh_on [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n4) { return; }
    const float4 v = a[gid] + b[gid];
    a[gid] = v;
    // Half mirror for the prefill GEMM (IMPARO_HALF_A): the same value the cvt pass
    // would produce, written here instead of by a separate dispatch.
    if (xh_on != 0u) { xh[gid] = half4(v); }
}

kernel void imparo_copy4(
    device float4 * dst [[buffer(0)]], device const float4 * src [[buffer(1)]],
    constant uint & n4 [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if (gid < n4) { dst[gid] = src[gid]; } }

kernel void imparo_act4(
    device float4 * a [[buffer(0)]], constant uint & n4 [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n4) { return; }
    const float4 v = a[gid];
    a[gid] = float4(imparo_act_f(v.x), imparo_act_f(v.y),
                    imparo_act_f(v.z), imparo_act_f(v.w));
}

kernel void imparo_scale4(
    device float4 * a [[buffer(0)]], constant float & k [[buffer(1)]],
    constant uint & n4 [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if (gid < n4) { a[gid] *= k; } }

// Fused residual-add + layer-out scale: a = (a + b) * k. Same two rounded operations,
// same order, as the imparo_add / imparo_scale pair it replaces -- bit-identical -- in
// one pass over `a` instead of two.
kernel void imparo_add_scale4(
    device float4 * a [[buffer(0)]], device const float4 * b [[buffer(1)]],
    constant float & k [[buffer(2)]], constant uint & n4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{ if (gid < n4) { a[gid] = (a[gid] + b[gid]) * k; } }

kernel void imparo_add_scale(
    device float * a [[buffer(0)]], device const float * b [[buffer(1)]],
    constant float & k [[buffer(2)]], constant uint & n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{ if (gid < n) { a[gid] = (a[gid] + b[gid]) * k; } }

kernel void imparo_act_mul4(
    device float4 * g [[buffer(0)]], device const float4 * u [[buffer(1)]],
    constant uint & n4 [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n4) { return; }
    const float4 v = g[gid];
    g[gid] = float4(imparo_act_f(v.x), imparo_act_f(v.y),
                    imparo_act_f(v.z), imparo_act_f(v.w)) * u[gid];
}

kernel void imparo_mul(
    device float * a [[buffer(0)]], device const float * b [[buffer(1)]],
    constant uint & n [[buffer(2)]], constant uint & b_off [[buffer(3)]],
    constant uint & b_stride [[buffer(4)]], constant uint & a_stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Four per thread when the width and every stride and offset divide by four, which
    // they do for the per-layer-embedding gate: ple is 256 and the strides are multiples
    // of it. `n` arrives pre-divided in that case; see the host side.
    const uint i = gid.x, t = gid.y;
    if (((n | a_stride | b_stride | b_off) & 3u) == 0u) {
        device float4 * a4 = (device float4 *)(a + (ulong)t * a_stride);
        device const float4 * b4 =
            (device const float4 *)(b + (ulong)t * b_stride + b_off);
        const uint n4 = n / 4u;
        if (i < n4) { a4[i] *= b4[i]; }
        return;
    }
    a[(ulong)t * a_stride + i] *= b[(ulong)t * b_stride + b_off + i];
}

kernel void imparo_scale(
    device float * a [[buffer(0)]], constant float & k [[buffer(1)]],
    constant uint & n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if (gid < n) { a[gid] *= k; } }

kernel void imparo_copy(
    device float * dst [[buffer(0)]], device const float * src [[buffer(1)]],
    constant uint & n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if (gid < n) { dst[gid] = src[gid]; } }

kernel void imparo_softcap(
    device float * a [[buffer(0)]], constant float & cap [[buffer(1)]],
    constant uint & n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{ if (gid < n) { a[gid] = cap * tanh(a[gid] / cap); } }

kernel void imparo_argmax(
    device const float * x [[buffer(0)]],
    device uint * out      [[buffer(1)]],
    constant uint & n      [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tw  [[threads_per_threadgroup]])
{
    threadgroup float bv[1024];
    threadgroup uint  bi[1024];
    float best = -INFINITY;
    uint  besti = 0u;
    // float4 loads over the aligned run (both vocabularies are multiples of 4): a quarter of
    // the load and compare instructions of the scalar scan. Components are compared in
    // index order with a strict `>`, so each thread's local best is its smallest-index
    // maximum and the tree below keeps the global smallest-index rule (2026-09-02, #119 D8).
    const uint n4 = n / 4u;
    device const float4 * x4 = (device const float4 *)x;
    for (uint i4 = tid; i4 < n4; i4 += tw) {
        const float4 v = x4[i4];
        const uint i = i4 * 4u;
        if (v.x > best) { best = v.x; besti = i; }
        if (v.y > best) { best = v.y; besti = i + 1u; }
        if (v.z > best) { best = v.z; besti = i + 2u; }
        if (v.w > best) { best = v.w; besti = i + 3u; }
    }
    for (uint i = n4 * 4u + tid; i < n; i += tw) {
        const float v = x[i];
        if (v > best) { best = v; besti = i; }
    }
    bv[tid] = best; bi[tid] = besti;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tw / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float ov = bv[tid + s];
            const uint  oi = bi[tid + s];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                bv[tid] = ov; bi[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { out[0] = bi[0]; }
}

// THE PICK THAT FEEDS THE NEXT STEP (docs/decode-turnaround.md): the same argmax, stored
// twice -- into `tokens[0]`, which the next decode step's embedding gather reads, and into
// `pick[pick_off]`, which the host reads one step later. Same reduction, same tie rule,
// so the token is the plain argmax's.
kernel void imparo_argmax_feed(
    device const float * x  [[buffer(0)]],
    device uint * tokens    [[buffer(1)]],
    constant uint & n       [[buffer(2)]],
    device uint * pick      [[buffer(3)]],
    constant uint & pick_off [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tw  [[threads_per_threadgroup]])
{
    threadgroup float bv[1024];
    threadgroup uint  bi[1024];
    float best = -INFINITY;
    uint  besti = 0u;
    const uint n4 = n / 4u;
    device const float4 * x4 = (device const float4 *)x;
    for (uint i4 = tid; i4 < n4; i4 += tw) {
        const float4 v = x4[i4];
        const uint i = i4 * 4u;
        if (v.x > best) { best = v.x; besti = i; }
        if (v.y > best) { best = v.y; besti = i + 1u; }
        if (v.z > best) { best = v.z; besti = i + 2u; }
        if (v.w > best) { best = v.w; besti = i + 3u; }
    }
    for (uint i = n4 * 4u + tid; i < n; i += tw) {
        const float v = x[i];
        if (v > best) { best = v; besti = i; }
    }
    bv[tid] = best; bi[tid] = besti;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tw / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float ov = bv[tid + s];
            const uint  oi = bi[tid + s];
            if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                bv[tid] = ov; bi[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { tokens[0] = bi[0]; pick[pick_off] = bi[0]; }
}

// THE K LARGEST ENTRIES OF EACH ROW (the drafter's candidates), in two dispatches with no
// threadgroup memory and no barriers. Pass one: every chunk of every row keeps its best k in a
// working list past the output. Pass two: each row merges its chunks' lists in chunk order.
// Larger values first; among equal values the smaller index first, the rule of imparo_argmax.
// Both passes hold it the same way: a value enters a full list only when it beats the list's
// last entry, and it lands behind every entry it does not beat, so an equal value that came
// later -- a larger index -- stays behind.
// The list capacity. imparo_metal.mm holds it too and refuses a larger k.
#define TOP_K_ROWS_MAX 64u

struct imparo_top_k_args {
    uint width;    // floats per row
    uint rows;
    uint k;
    uint chunk;    // floats per chunk
    uint chunks;   // chunks per row
};

// Admits (v, i) to a list of n entries kept larger first; returns the new count.
static inline uint top_k_admit(thread float * lv, thread uint * li, uint n, uint k,
                               float v, uint i) {
    if (n == k && !(v > lv[k - 1u])) { return n; }
    uint p = n < k ? n : k - 1u;
    while (p > 0u && lv[p - 1u] < v) {
        lv[p] = lv[p - 1u];
        li[p] = li[p - 1u];
        p -= 1u;
    }
    lv[p] = v;
    li[p] = i;
    return n < k ? n + 1u : n;
}

// The destination, in elements: ids [rows * k], values [rows * k], then the lists' ids
// [rows * chunks * k] and values [rows * chunks * k]. An empty list slot holds id 0xffffffff.
kernel void imparo_top_k_chunks(
    device const float * src [[buffer(0)]],
    device uint * ids        [[buffer(1)]],
    device float * vals      [[buffer(2)]],
    constant imparo_top_k_args & a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint r = gid.y;
    const uint lo = c * a.chunk;
    const uint hi = min(lo + a.chunk, a.width);
    float lv[TOP_K_ROWS_MAX];
    uint  li[TOP_K_ROWS_MAX];
    uint n = 0u;
    const ulong row = (ulong)r * a.width;
    for (uint i = lo; i < hi; i++) {
        n = top_k_admit(lv, li, n, a.k, src[row + i], i);
    }
    const ulong out = (ulong)a.rows * a.k;
    const ulong lists = out * a.chunks;
    const ulong at = ((ulong)r * a.chunks + c) * a.k;
    for (uint j = 0u; j < a.k; j++) {
        ids[2ul * out + at + j]          = j < n ? li[j] : 0xffffffffu;
        vals[2ul * out + lists + at + j] = j < n ? lv[j] : -INFINITY;
    }
}

kernel void imparo_top_k_merge(
    device uint * ids   [[buffer(0)]],
    device float * vals [[buffer(1)]],
    constant imparo_top_k_args & a [[buffer(2)]],
    uint r [[thread_position_in_grid]])
{
    float lv[TOP_K_ROWS_MAX];
    uint  li[TOP_K_ROWS_MAX];
    uint n = 0u;
    const ulong out = (ulong)a.rows * a.k;
    const ulong lists = out * a.chunks;
    for (uint c = 0u; c < a.chunks; c++) {
        const ulong at = ((ulong)r * a.chunks + c) * a.k;
        for (uint j = 0u; j < a.k; j++) {
            const uint i = ids[2ul * out + at + j];
            const float v = vals[2ul * out + lists + at + j];
            // A list is sorted: past an empty slot or a refused value nothing enters.
            if (i == 0xffffffffu || (n == a.k && !(v > lv[a.k - 1u]))) { break; }
            n = top_k_admit(lv, li, n, a.k, v, i);
        }
    }
    for (uint j = 0u; j < a.k; j++) {
        ids[(ulong)r * a.k + j]        = j < n ? li[j] : 0xffffffffu;
        vals[out + (ulong)r * a.k + j] = j < n ? lv[j] : -INFINITY;
    }
}

// THE K LARGEST ENTRIES OF EACH ROW AT K <= 8: the passes, layout and order of
// imparo_top_k_chunks and imparo_top_k_merge, with the working list in eight named slots instead
// of 64-entry thread arrays. A value that does not enter costs its own load and one compare with
// the eighth slot, and the merge loads an id only for a value that enters. The array-list passes
// took 0.47 ms and 0.09 ms for 9 x 128000 at K = 8
// (docs/evidence/dspark/2026-09-15-step3f-candidates-lfm25.md).
#define TOP_K_EMPTY 0xffffffffu

struct top8_list {
    float v0, v1, v2, v3, v4, v5, v6, v7;
    uint  i0, i1, i2, i3, i4, i5, i6, i7;
};

inline __attribute__((always_inline)) top8_list top8_empty() {
    top8_list t;
    t.v0 = t.v1 = t.v2 = t.v3 = t.v4 = t.v5 = t.v6 = t.v7 = -INFINITY;
    t.i0 = t.i1 = t.i2 = t.i3 = t.i4 = t.i5 = t.i6 = t.i7 = TOP_K_EMPTY;
    return t;
}

// One slot of an insertion. The travelling entry (x, xi) takes the slot when the slot is empty or
// x beats it, and once it has taken one, every later slot moves down by one. An equal value does
// not take the slot, so equal values must arrive smaller id first.
inline __attribute__((always_inline)) void top8_slot(thread float & v, thread uint & i,
                                                     thread float & x, thread uint & xi,
                                                     thread bool & moved) {
    const bool take = moved || i == TOP_K_EMPTY || x > v;
    const float tv = v;
    const uint ti = i;
    v = take ? x : v;
    i = take ? xi : i;
    x = take ? tv : x;
    xi = take ? ti : xi;
    moved = take;
}

inline __attribute__((always_inline)) void top8_insert(thread top8_list & t, float x, uint xi) {
    bool moved = false;
    top8_slot(t.v0, t.i0, x, xi, moved);
    top8_slot(t.v1, t.i1, x, xi, moved);
    top8_slot(t.v2, t.i2, x, xi, moved);
    top8_slot(t.v3, t.i3, x, xi, moved);
    top8_slot(t.v4, t.i4, x, xi, moved);
    top8_slot(t.v5, t.i5, x, xi, moved);
    top8_slot(t.v6, t.i6, x, xi, moved);
    top8_slot(t.v7, t.i7, x, xi, moved);
}

// The list's first k entries: ids from ids[at], values from vals[vat].
inline __attribute__((always_inline)) void top8_write(thread const top8_list & t, uint k,
                                                      device uint * ids, ulong at,
                                                      device float * vals, ulong vat) {
    const float v[8] = {t.v0, t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7};
    const uint i[8] = {t.i0, t.i1, t.i2, t.i3, t.i4, t.i5, t.i6, t.i7};
    for (uint j = 0u; j < k; j++) {
        ids[at + j] = i[j];
        vals[vat + j] = v[j];
    }
}

kernel void imparo_top_k8_chunks(
    device const float * src [[buffer(0)]],
    device uint * ids        [[buffer(1)]],
    device float * vals      [[buffer(2)]],
    constant imparo_top_k_args & a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint r = gid.y;
    const uint lo = c * a.chunk;
    const uint hi = min(lo + a.chunk, a.width);
    const ulong row = (ulong)r * a.width;
    top8_list t = top8_empty();
    for (uint i = lo; i < hi; i++) {
        const float x = src[row + i];
        if (t.i7 != TOP_K_EMPTY && !(x > t.v7)) { continue; }
        top8_insert(t, x, i);
    }
    const ulong out = (ulong)a.rows * a.k;
    const ulong lists = out * a.chunks;
    const ulong at = ((ulong)r * a.chunks + c) * a.k;
    top8_write(t, a.k, ids, 2ul * out + at, vals, 2ul * out + lists + at);
}

// ONE PASS, for a row narrow enough that splitting it costs more than it saves. The two-pass
// form exists because a wide row wants many threads on it: chunk threads each scan a slice,
// then one thread merges their lists. A 32-expert router picking 4 gives chunks=3 -- a
// three-thread dispatch and a one-thread dispatch, both pure launch -- and a routed layer
// runs this every token. This is the chunk scan over the WHOLE row, writing straight to the
// merge's output slots, so the result is the same list by the same insertion order.
kernel void imparo_top_k8_one(
    device const float * src [[buffer(0)]],
    device uint * ids        [[buffer(1)]],
    device float * vals      [[buffer(2)]],
    constant imparo_top_k_args & a [[buffer(3)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= a.rows) { return; }
    const ulong row = (ulong)r * a.width;
    top8_list t = top8_empty();
    for (uint i = 0u; i < a.width; i++) {
        const float x = src[row + i];
        if (t.i7 != TOP_K_EMPTY && !(x > t.v7)) { continue; }
        top8_insert(t, x, i);
    }
    top8_write(t, a.k, ids, (ulong)r * a.k, vals, (ulong)a.rows * a.k + (ulong)r * a.k);
}

kernel void imparo_top_k8_merge(
    device uint * ids   [[buffer(0)]],
    device float * vals [[buffer(1)]],
    constant imparo_top_k_args & a [[buffer(2)]],
    uint r [[thread_position_in_grid]])
{
    const ulong out = (ulong)a.rows * a.k;
    const ulong lists = out * a.chunks;
    top8_list t = top8_empty();
    for (uint c = 0u; c < a.chunks; c++) {
        const ulong at = ((ulong)r * a.chunks + c) * a.k;
        for (uint j = 0u; j < a.k; j++) {
            // A list is sorted: past a value the full list refuses nothing enters, and an empty
            // slot ends the list.
            const float v = vals[2ul * out + lists + at + j];
            if (t.i7 != TOP_K_EMPTY && !(v > t.v7)) { break; }
            const uint i = ids[2ul * out + at + j];
            if (i == TOP_K_EMPTY) { break; }
            top8_insert(t, v, i);
        }
    }
    top8_write(t, a.k, ids, (ulong)r * a.k, vals, out + (ulong)r * a.k);
}

// ONE LOGISTIC OUTPUT PER ROW (the drafter's confidence head), in two passes over dst:
//   blocks  dst[dst_off + rows + r * blocks + c] = block c's sum of weight x input in index order,
//           a block being LOGISTIC_BLOCK entries of a, then of b
//   finish  dst[dst_off + r] = sigmoid(the row's block sums in block order, then the bias)
// One thread per row over all a_width + b_width products took 0.2 ms at 9 x 2304
// (docs/evidence/dspark/2026-09-15-step3f-candidates-lfm25.md); the blocks spread those products
// over many threads. imparo_metal.mm holds LOGISTIC_BLOCK too.
#define LOGISTIC_BLOCK 64u

struct imparo_logistic_args {
    uint a_width;
    uint b_width;
    uint w_off;
    uint dst_off;
    uint rows;
    uint a_blocks;   // blocks over a; the rest are over b
    uint blocks;
};

kernel void imparo_logistic_blocks(
    device const float * a [[buffer(0)]],
    device const float * b [[buffer(1)]],
    device const float * w [[buffer(2)]],
    device float * dst     [[buffer(3)]],
    constant imparo_logistic_args & p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint r = gid.y;
    const bool in_a = c < p.a_blocks;
    const uint width = in_a ? p.a_width : p.b_width;
    const uint lo = (in_a ? c : c - p.a_blocks) * LOGISTIC_BLOCK;
    const uint hi = min(lo + LOGISTIC_BLOCK, width);
    const ulong src = (ulong)r * width;
    const ulong wb = (ulong)p.w_off + (in_a ? 0u : p.a_width);
    float z = 0.0f;
    if (in_a) {
        for (uint j = lo; j < hi; j++) { z += w[wb + j] * a[src + j]; }
    } else {
        for (uint j = lo; j < hi; j++) { z += w[wb + j] * b[src + j]; }
    }
    dst[(ulong)p.dst_off + p.rows + (ulong)r * p.blocks + c] = z;
}

kernel void imparo_logistic_finish(
    device const float * w [[buffer(0)]],
    device float * dst     [[buffer(1)]],
    constant imparo_logistic_args & p [[buffer(2)]],
    uint r [[thread_position_in_grid]])
{
    const ulong part = (ulong)p.dst_off + p.rows + (ulong)r * p.blocks;
    float z = 0.0f;
    for (uint c = 0u; c < p.blocks; c++) { z += dst[part + c]; }
    z += w[(ulong)p.w_off + p.a_width + p.b_width];
    dst[(ulong)p.dst_off + r] = 1.0f / (1.0f + exp(-z));
}

// per-layer path: y[t*ple + i] = (proj[t*W + l*ple + i] + emb[i]*embscale) * combscale
// Gather each token's per-layer embedding row and fold it into the projection in ONE pass.
//
// It used to take one imparo_metal_row dispatch per token to materialise a b x width table
// -- 512 dispatches and 21 MiB at batch 512 -- which imparo_ple_combine then read exactly
// once and never touched again. The row index is the only thing those dispatches carried,
// so passing the token ids to the combine kernel removes both the dispatches and the
// buffer. Nothing here changes the arithmetic: the same value is dequantised, scaled and
// added, just without a round trip through device memory.
kernel void imparo_ple_gather_combine(
    device float * proj          [[buffer(0)]],
    device const uchar * weights [[buffer(1)]],
    device const uint  * tokens  [[buffer(2)]],
    constant ulong & w_offset  [[buffer(3)]], constant uint & width [[buffer(4)]],
    constant float & emb_scale [[buffer(5)]], constant float & comb_scale [[buffer(6)]],
    constant uint & n_tok     [[buffer(7)]],
    // staged = 1: `weights` holds this batch's rows in token order (the host gathered
    // them; Table tier), so row t is token t and `tokens` is not read.
    constant uint & staged    [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    // FOUR values per thread. Values j..j+3 are the low nibbles of four adjacent payload
    // bytes (or the high nibbles of the same four), so one uchar4 load covers them, and
    // the four projections are one float4 read-modify-write.
    const uint i4 = gid.x, t = gid.y;
    const uint width4 = width / 4u;
    if (i4 >= width4 || t >= n_tok) { return; }
    const uint i = i4 * 4u;
    const uint blocks = width / QK4_0;
    const uint bi = i / QK4_0, j = i % QK4_0;
    const uint row = staged ? t : tokens[t];
    device const uchar * blk = weights + w_offset
                             + ((ulong)row * blocks + bi) * Q4_0_BYTES;
    const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
    // Alignment here is arbitrary, not merely every-other-block: the offset carries a
    // runtime `j % 16`.
    const uchar4 quad = uchar4(*(device const packed_uchar4 *)(blk + 2 + (j % (QK4_0 / 2))));
    const float4 e = ((j < QK4_0 / 2 ? float4(quad & uchar4(0x0F))
                                     : float4(quad >> 4)) - 8.0f) * d;
    device float4 * p4 = (device float4 *)(proj + (ulong)t * width);
    p4[i4] = (p4[i4] + e * emb_scale) * comb_scale;
}

// ---- BATCHED EMBEDDING GATHER ------------------------------------------------------
//
// ONE DISPATCH FOR THE WHOLE BATCH. `row` materialises a single table row, so the prefill
// path issued one dispatch per token -- 512 of them at the tuned batch. These take the
// index vector and cover every row at once.
//
// FOUR VALUES PER THREAD, not one block per thread, which is where this differs from the
// branch it came from. A block-per-thread gather gives width/32 threads per token and
// writes 32 scalars; this gives width/4 and writes one float4, the same shape
// imparo_ple_gather_combine already uses for exactly this access pattern.
//
// width must be a multiple of 4 (and of 32 for the quantised kinds, which implies it).
// The host refuses anything else rather than writing a partial float4.
kernel void imparo_f32_gather_rows(
    device const uchar * weights [[buffer(0)]], device float * y [[buffer(1)]],
    device const uint * indices [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],
    constant uint & width [[buffer(4)]], constant uint & table_rows [[buffer(5)]],
    constant float & scale [[buffer(6)]], constant uint & dst_off [[buffer(7)]],
    constant uint & n_rows [[buffer(8)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i4 = gid.x, t = gid.y;
    if (i4 >= width / 4u || t >= n_rows) { return; }
    const uint index = indices[t];
    // A token id past the table is a bug upstream, but reading there would fault or
    // return another tensor's bytes; leaving the slot alone is the recoverable choice.
    if (index >= table_rows) { return; }
    device const float4 * table =
        (device const float4 *)(weights + w_offset) + (ulong)index * (width / 4u);
    device float4 * y4 = (device float4 *)(y + (ulong)dst_off + (ulong)t * width);
    y4[i4] = table[i4] * scale;
}

kernel void imparo_q4_0_gather_rows(
    device const uchar * weights [[buffer(0)]], device float * y [[buffer(1)]],
    device const uint * indices [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],
    constant uint & width [[buffer(4)]], constant uint & table_rows [[buffer(5)]],
    constant float & scale [[buffer(6)]], constant uint & dst_off [[buffer(7)]],
    constant uint & n_rows [[buffer(8)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i4 = gid.x, t = gid.y;
    if (i4 >= width / 4u || t >= n_rows) { return; }
    const uint index = indices[t];
    if (index >= table_rows) { return; }
    // Values j..j+3 are the low nibbles of four adjacent payload bytes, or the high
    // nibbles of the same four, so one uchar4 load covers them. Same expression as
    // imparo_ple_gather_combine -- element i and i+16 share a byte, and getting that
    // pairing wrong produces plausible garbage.
    const uint i = i4 * 4u;
    const uint blocks = width / QK4_0;
    const uint bi = i / QK4_0, j = i % QK4_0;
    device const uchar * blk = weights + w_offset
                             + ((ulong)index * blocks + bi) * Q4_0_BYTES;
    const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
    const uchar4 quad = uchar4(*(device const packed_uchar4 *)(blk + 2 + (j % (QK4_0 / 2))));
    const float4 e = ((j < QK4_0 / 2 ? float4(quad & uchar4(0x0F))
                                     : float4(quad >> 4)) - 8.0f) * d;
    device float4 * y4 = (device float4 *)(y + (ulong)dst_off + (ulong)t * width);
    y4[i4] = e * scale;
}

kernel void imparo_q8_0_gather_rows(
    device const uchar * weights [[buffer(0)]], device float * y [[buffer(1)]],
    device const uint * indices [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],
    constant uint & width [[buffer(4)]], constant uint & table_rows [[buffer(5)]],
    constant float & scale [[buffer(6)]], constant uint & dst_off [[buffer(7)]],
    constant uint & n_rows [[buffer(8)]], uint2 gid [[thread_position_in_grid]])
{
    const uint i4 = gid.x, t = gid.y;
    if (i4 >= width / 4u || t >= n_rows) { return; }
    const uint index = indices[t];
    if (index >= table_rows) { return; }
    // Four CONSECUTIVE signed bytes: no pairing, so one packed_char4 is the four values.
    const uint i = i4 * 4u;
    const uint blocks = width / QK8_0;
    const uint bi = i / QK8_0, j = i % QK8_0;
    device const uchar * blk = weights + w_offset
                             + ((ulong)index * blocks + bi) * Q8_0_BYTES;
    const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
    const char4 q = char4(*(device const packed_char4 *)(blk + 2u + j));
    device float4 * y4 = (device float4 *)(y + (ulong)dst_off + (ulong)t * width);
    y4[i4] = float4(q) * d * scale;
}

// DECODE / SMALL-BATCH GEMV for the tile-major family, and for a row-major block format
// that kept its layout (a row-gathered tensor, and the lm head when it is tied to one).
//
// Why this exists: the register-tiled GEMM's write-back assumes WHOLE token tiles, so a
// 1-token dispatch there writes 64 tokens' worth of rows into a destination sized for one
// and lands in the next allocation -- measured, it overwrote the mega-kernel's sync buffer
// (task #168). The lm head is n_tok = 1 by construction, so this arm is not an
// optimisation, it is what lets the family serve a whole model.
//
// Shape: one output row per simdgroup, and EACH LANE OWNS A WHOLE SUB-BLOCK -- lane l
// takes sub-blocks l, l+32, l+64, ... and multiplies every one of the 32 values it
// decodes. One simd_sum per row. Tokens are a loop because this path only runs below one
// token tile.
//
// IT USED TO SPLIT THE OTHER WAY: the 32 lanes covered ONE sub-block, so every lane
// decoded all 32 values and kept the one at its own index -- 32x the decode ALU, written
// as a deliberate "correctness first" cost when the lm head was the only dispatch that
// reached this kernel. Qwen3.8 is tile-major in every projection, so decode sends 497
// dispatches a token through here, and the profile read matmat_decode 99.2% of the step
// at 2.19 s per token (the lm head alone, one dispatch, took 897 ms -- about 1 GB/s).
// Turning the split around costs nothing: each lane still decodes one sub-block, and now
// it uses all of it.
//
// The reads stay coalesced. Within a block the 32 values of a sub-block are contiguous,
// so consecutive lanes take consecutive 16-byte payload chunks; `x` likewise, 32
// consecutive floats per lane over a contiguous row segment.
// ROWS PER SIMDGROUP. One row per simdgroup makes the whole kernel one dependent chain of
// decode-then-multiply; several give it that many independent chains and read the x values
// once for all of them.
//
// A PREPROCESSOR MACRO, not a function constant: it sizes `acc[]` and Metal rejects a
// function constant there ("array size is not a constant expression"), which is the same
// wall the attention head dims hit. Injected at library compile from the host's value.
#ifndef IMPARO_BLK_GEMV_NR
#define IMPARO_BLK_GEMV_NR 2
#endif
constant uint BLK_GEMV_NR = IMPARO_BLK_GEMV_NR;

kernel void imparo_blk_gemv(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
    uint3 tgpg [[threadgroups_per_grid]])
{
    // PROBE ONLY (IMPARO_BLK_GEMV_STRIDE=1 at library build). The shipped kernel gives one
    // row group to one simdgroup and the host sizes the grid to cover n_out, so the loop
    // below would run exactly once -- but a loop is not free even when it runs once (the
    // mega item loop alone cost a pipeline half its admission), so it is compiled out.
    // With it in, the host may dispatch ANY threadgroup count and the grid strides,
    // which is what lets the same kernel be timed at the mega-kernel's 18-threadgroup
    // grid to separate "the fold's decomposition costs" from "a small grid costs".
    // Each row's dot product stays inside one simdgroup either way, so the answer does
    // not move -- the step hash is the check.
#if IMPARO_BLK_GEMV_STRIDE
    const uint r_stride = tgpg.x * nsg * BLK_GEMV_NR;
    for (uint r = (tgid.x * nsg + sgid) * BLK_GEMV_NR; r < n_out; r += r_stride) {
#else
    const uint r = (tgid.x * nsg + sgid) * BLK_GEMV_NR;
    if (r >= n_out) { return; }
#endif

    const uint sub_per_block = tm_block_elems() / 32u;
    const uint bb = tm_block_bytes();
    const uint sc_bytes = tm_scale_bytes();
    const uint pay_bytes = bb - sc_bytes;
    const uint blocks = n_in / tm_block_elems();
    const uint subs = n_in / 32u;
    const uint so = tm_scale_src_off();

    for (uint t = 0; t < n_tok; ++t) {
        device const float * xb = x + (ulong)(src_row + t) * n_in;
        float acc[BLK_GEMV_NR];
        for (uint i = 0u; i < BLK_GEMV_NR; ++i) { acc[i] = 0.0f; }
        for (uint sb = lane; sb < subs; sb += 32u) {
            const uint blk = sb / sub_per_block, sub = sb % sub_per_block;
            // The x values ONCE for all BLK_GEMV_NR rows: they are the same 32 floats,
            // and holding them in registers is what makes the extra rows nearly free.
            // Eight 16-byte loads: a row is n_in floats (a multiple of 32) from the
            // buffer's start, so every sub-block's 32 floats begin 128-byte aligned.
            device const float4 * xs4 = (device const float4 *)(xb + sb * 32u);
            float xv[32];
            for (uint j = 0u; j < 8u; ++j) {
                const float4 t = xs4[j];
                xv[4u * j] = t.x; xv[4u * j + 1u] = t.y; xv[4u * j + 2u] = t.z; xv[4u * j + 3u] = t.w;
            }
            for (uint i = 0u; i < BLK_GEMV_NR; ++i) {
                const uint rr = r + i;
                if (rr >= n_out) { break; }
                device const uchar * sc;
                device const uchar * pay;
                if (WFMT_ROW) {
                    device const uchar * b = weights + w_offset
                        + ((ulong)rr * blocks + blk) * bb;
                    sc  = b + so;
                    pay = b + (so == 0u ? sc_bytes : 0u);
                } else {
                    const ulong unit = ((ulong)(rr / TM_UNIT_ROWS) * blocks + blk)
                                     * (ulong)(TM_UNIT_ROWS * bb);
                    const uint slot = rr % TM_UNIT_ROWS;
                    sc  = weights + w_offset + unit + (ulong)slot * sc_bytes;
                    pay = weights + w_offset + unit
                        + (ulong)TM_UNIT_ROWS * sc_bytes + (ulong)slot * pay_bytes;
                }
                float v[32];
                tm_sub32(sc, pay, sub, v, !WFMT_ROW);
                float part = 0.0f;
                for (uint j = 0; j < 32u; ++j) { part += v[j] * xv[j]; }
                acc[i] += part;
            }
        }
        // Reached by every lane: the loop above leaves the lanes with no sub-block of
        // their own at acc = 0, it does not skip them.
        for (uint i = 0u; i < BLK_GEMV_NR; ++i) {
            const float total = simd_sum(acc[i]);
            const uint rr = r + i;
            if (lane == 0u && rr < n_out) {
                device float * slot = y + (ulong)t * n_out + rr;
                *slot = epilogue ? (imparo_act_f(*slot) * total) : total;
            }
        }
    }
#if IMPARO_BLK_GEMV_STRIDE
    }
#endif
}

// THE DECODE-ROWS FORM OF imparo_blk_gemv, for a co-batched step's rows on the fast route. The
// kernel above loops the tokens outermost, so each extra row decodes every weight again and reads
// it again: two rows of Qwen3.8-27B cost two lone decodes. Here each sub-block is decoded once
// and multiplied into every token's activations.
//
// ONE WEIGHT ROW PER LANE. A threadgroup covers 32 weight rows, lane l owning row 32 * tgid + l;
// simdgroup s takes the row's sub-blocks s, s + nsg, ..., so the threadgroup's simdgroups work
// through the same blocks together, and their sums meet once, in threadgroup memory, in
// simdgroup order. All 32 lanes of a simdgroup multiply the same activations at the same time,
// so an activation read is one read for the simdgroup, straight from the buffer, and the loop has
// no barrier. (Contiguous slices, one per simdgroup, took 146.0 ms at 2 rows with the
// multiply-adds skipped, these 116.6: the rows' weight lines are shared while the simdgroups
// are near each other. The whole kernel moved under 1% either way: 132.6 -> 131.5 ms at 2 rows,
// 156.0 -> 156.5 at 3. The matrix-unit rows kernel below takes its sub-blocks the same way.)
//
// It replaced a split that gave each lane one sub-block of two rows (Qwen3.8-27B, M3 Pro,
// projections a step). There every lane needed its own activations of every token -- 2 bytes
// of threadgroup memory per weight per row, about 205 GB a step at 4 rows -- and each
// 1024-value round met at two barriers, so the weight reads and the multiply-adds ran one after
// the other: 287.6 ms at 4 rows, 186.9 with no weight reads, 110.6 with no multiply-adds. This
// split: 132.6 / 156.0 / 178.8 / 282.1 ms at 2 / 3 / 4 / 8 rows, where the old one took
// 154.4 / - / 287.6 / 402.7 and the GEMM takes 199.1 / 199.3 / 199.5 / 200.4
// (docs/evidence/cobatch/2026-09-19-27b-rows-lane.md).
//
// Every loop over tokens runs to a compile-time count and is unrolled, with a guard where fewer
// tokens are live: a loop that ends on the runtime n_tok indexes the accumulators at run time
// (2 rows cost 179.1 ms instead of 154.5 in the old split).
//
// Each (row, token) sums its simdgroup's sub-blocks in order and the simdgroups in order -- the
// same every run, but not the one-row kernel's order, so a row differs from its lone decode in
// the last bits (tests/blk_decode_rows.rs). The exact route keeps the one-row kernel per row;
// this one serves the fast route, whose rows must pick what their lone decodes pick.
//
// BLK_ROWS_TOK (function constant 39): the tokens a pipeline holds, 2, 4 or 8. A step with
// fewer (3 rows on the 4-token form) passes n_tok, and the tokens past it are neither read nor
// stored. The accumulators are sized by the widest form; a narrower form's loops end at its
// constant, so the entries past it are never touched.
constant uint BLK_ROWS_TOK_FC [[function_constant(39)]];
constant uint BLK_ROWS_TOK = is_function_constant_defined(BLK_ROWS_TOK_FC) ? BLK_ROWS_TOK_FC : 2u;
#define BLK_ROWS_MAX_TOK 8u
// DIAGNOSTIC, 0 in every served pipeline (IMPARO_BLK_ROWS_SKIP): bit 0 drops the weight decode
// and its loads (the values are constants), bit 1 the activation loads (1), bit 2 the 32
// multiply-adds (one product) -- each skipped part's time is the difference. Wrong numbers on
// purpose.
constant uint BLK_ROWS_SKIP_FC [[function_constant(40)]];
constant uint BLK_ROWS_SKIP = is_function_constant_defined(BLK_ROWS_SKIP_FC) ? BLK_ROWS_SKIP_FC : 0u;

kernel void imparo_blk_gemv_rows(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    threadgroup float * red [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    // No early return: every simdgroup meets the barrier. A lane past n_out decodes row
    // n_out - 1 again and is never stored.
    const uint row = tgid.x * 32u + lane;
    const uint rr = min(row, n_out - 1u);

    const uint sub_per_block = tm_block_elems() / 32u;
    const uint bb = tm_block_bytes();
    const uint sc_bytes = tm_scale_bytes();
    const uint pay_bytes = bb - sc_bytes;
    const uint blocks = n_in / tm_block_elems();
    const uint subs = n_in / 32u;
    const uint so = tm_scale_src_off();
    // The row's first block; a tile-major unit holds one block of TM_UNIT_ROWS rows.
    const uint unit_bytes = TM_UNIT_ROWS * bb;
    const uint slot = rr % TM_UNIT_ROWS;
    device const uchar * rbase = WFMT_ROW
        ? weights + w_offset + (ulong)rr * blocks * bb
        : weights + w_offset + (ulong)(rr / TM_UNIT_ROWS) * blocks * unit_bytes;
    float acc[BLK_ROWS_MAX_TOK];
    #pragma unroll
    for (uint t = 0u; t < BLK_ROWS_TOK; ++t) { acc[t] = 0.0f; }
    for (uint sb = sgid; sb < subs; sb += nsg) {
        const uint blk = sb / sub_per_block, sub = sb % sub_per_block;
        device const uchar * sc;
        device const uchar * pay;
        if (WFMT_ROW) {
            device const uchar * b = rbase + blk * bb;
            sc  = b + so;
            pay = b + (so == 0u ? sc_bytes : 0u);
        } else {
            device const uchar * u = rbase + blk * unit_bytes;
            sc  = u + slot * sc_bytes;
            pay = u + TM_UNIT_ROWS * sc_bytes + slot * pay_bytes;
        }
        float v[32];
        if (BLK_ROWS_SKIP & 1u) {
            for (uint j = 0u; j < 32u; ++j) { v[j] = 1.0f + float(sub + j + lane); }
        } else {
            tm_sub32(sc, pay, sub, v, !WFMT_ROW);
        }
        #pragma unroll
        for (uint t = 0u; t < BLK_ROWS_TOK; ++t) {
            if (t >= n_tok) { continue; }
            // The same address in every lane: one read for the simdgroup.
            device const float4 * xt =
                (device const float4 *)(x + (ulong)(src_row + t) * n_in + sb * 32u);
            float part = 0.0f;
            #pragma unroll
            for (uint m = 0u; m < 8u; ++m) {
                const float4 q = (BLK_ROWS_SKIP & 2u) ? float4(1.0f) : xt[m];
                if (BLK_ROWS_SKIP & 4u) {
                    if (m == 0u) { part = v[t] * q.x; }
                    continue;
                }
                part += v[4u * m] * q.x;
                part += v[4u * m + 1u] * q.y;
                part += v[4u * m + 2u] * q.z;
                part += v[4u * m + 3u] * q.w;
            }
            acc[t] += part;
        }
    }
    // The simdgroups meet: simdgroup s's sums at red[s][t][lane], added in s order.
    #pragma unroll
    for (uint t = 0u; t < BLK_ROWS_TOK; ++t) {
        red[(sgid * BLK_ROWS_TOK + t) * 32u + lane] = acc[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = sgid; t < n_tok; t += nsg) {
        float total = 0.0f;
        for (uint s = 0u; s < nsg; ++s) { total += red[(s * BLK_ROWS_TOK + t) * 32u + lane]; }
        if (row < n_out) {
            device float * dst = y + (ulong)t * n_out + row;
            *dst = epilogue ? (imparo_act_f(*dst) * total) : total;
        }
    }
}

// THE BLOCK FORMATS' ROWS ON THE SIMDGROUP MATRIX UNIT, for a co-batched step's rows on the fast
// route (up to blk_rows_mma_max, 8 at most: one 8-token column). The Q8_0 / Q4_0 rows matmul's form:
// a threadgroup takes BLK_MMA_TILES 8-row weight units and its simdgroups split the 32-value
// sub-blocks. Per sub-block a lane decodes only the eight values it holds in the four 8x8 weight
// fragments (`tm_run8`, in the format's K order) and reads the activations in the same order, so
// each unit's W[8 x 32] . X[8 x 32]^T is four products straight from registers -- no tile, no
// barrier in the loop. The values are the brick's floats, fully decoded, so the products add up
// across sub-blocks with no scale step and a row keeps float accuracy (tests/brick_rows.rs pins
// every fetched value to tm_sub32's bits). Token columns past n_tok read the last live row and are
// not stored.
//
// Qwen3.8-27B (M3 Pro), projections a step at 2 / 4 / 8 rows: 123.8 / 125.1 / 126.8 ms (126.9 /
// 128.4 / 130.0 before Q4_K and Q5_K took sub-block pairs), where the form that staged 32 rows x
// 32 decoded floats in threadgroup memory took 170-171 from 4 to 8 and the row-per-lane GEMV 131.4
// at 2 rows (one row: 108.4). In the co-batched step, before the pairs, the 2-row case tied with
// that GEMV (138.7 against 138.5 ms) and 3 to 8 rows ran 13-20% faster than before; the pairs
// took 2 / 4 rows to 135.2 / 143.2 (docs/evidence/cobatch/2026-09-20-27b-rows-no-tile.md). Two units a threadgroup share each
// activation fetch (153.8 -> 143.4 ms at 4 rows; four units 158.1); 8 simdgroups (4: 133.1, 16:
// 132.5). A scalar form with K split over the lanes and 8 rows a lane lost at every row count
// (134.7 / 175.8 / 257.5 at 2 / 3 / 4 rows).
constant uint BLK_MMA_TILES_FC [[function_constant(43)]];
constant uint BLK_MMA_TILES = is_function_constant_defined(BLK_MMA_TILES_FC) ? BLK_MMA_TILES_FC : 2u;
// The token-fragment loops of the rows matmul: a constant trip count (the template's TF), fully
// unrolled so the per-fragment arrays stay in registers.
#define BLK_TF_UNROLL _Pragma("clang loop unroll(full)")
#define BLK_MMA_MAX_TILES 4u
// DIAGNOSTIC, 0 in every served pipeline (IMPARO_BLK_MMA_SKIP): bit 0 drops the matrix products
// (the sums take the fragments' elements instead), bit 1 the weight fetch (values from the lane
// and the sub-block, no weight byte read), bit 2 the activation loads. Wrong numbers on purpose.
// THE ACTIVATIONS AS HALF: the mirror the norms already dual-write beside the floats, when it
// covers this step's rows. The same values rounded once, in half the bytes and one load where the
// float path takes two. The prefill GEMM and the co-batched GEMM above eight rows read the same
// mirror, so this makes 2..8 rows agree with them rather than adding a rounding of its own; the
// exact route keeps float activations.
// TWO ENTRY POINTS OVER ONE BODY: the template argument is gone before the AIR exists, so
// imparo_blk_rows_mma carries no half code at all and compiles as it did before this arm. (A
// function constant may fold as cleanly; it was not re-measured after the brackets below were
// fixed, and this form needs no such argument.)
// THE OFFSET IS PASTED, NOT PARENTHESISED: `x + xo0 + sb * 32u` lets the compiler hoist the row
// base and walk a constant stride, while `x + (xo0 + sb * 32u)` makes it add two 64-bit values at
// every load. That one pair of brackets cost the FLOAT path 2.4 ms a step -- 123.4 against 121.1
// at 2 rows, with the half path switched off -- and was the whole of a regression first blamed on
// the arm's presence.
#define BLK_MMA_X4(off) (XHALF ? float4(*((device const half4 *)(xh + off))) \
                               : *((device const float4 *)(x + off)))
constant uint BLK_MMA_SKIP_FC [[function_constant(44)]];
constant uint BLK_MMA_SKIP = is_function_constant_defined(BLK_MMA_SKIP_FC) ? BLK_MMA_SKIP_FC : 0u;

// ROUTED: the rows this computes are WORK ROWS of one expert, so a row's activations live
// at perm[row] and not at row. That is the only difference -- two offsets below -- and it is
// why the routed feed-forward does not need a kernel of its own. The template argument is
// gone before the AIR exists, so imparo_blk_rows_mma still compiles with no routed code in
// it, the same reason the half mirror is a template argument and not a function constant.
// PAIR: two weight stacks in ONE K loop, `act(x . w) * (x . w2)` written once.
//
// TWO ACCUMULATORS, NOT TWO PASSES. Running the body twice and multiplying the second
// result into the first works and is bit-identical, but it is SLOWER than the two
// dispatches it replaces (+0.76% on a 128-step decode, measured): two dispatches of
// independent work overlap under concurrent dispatch, and a barrier between two passes of
// one threadgroup does not. This form has no barrier between the stacks -- every
// activation fragment feeds both -- so it removes a dispatch AND halves the activation
// loads instead of trading the overlap away.
template <bool XHALF, bool ROUTED, bool PAIR, uint TF = 1u>
static inline __attribute__((always_inline)) void blk_rows_mma_body(
    device const uchar * weights, device const float * x, device float * y,
    device const half * xh, device const uint * perm, uint src_work_rows,
    constant uint & n_in, constant uint & n_out, uint n_tok,
    // BY VALUE, not `constant uint &`: the routed pair below hands the gate projection a
    // literal 0 and the up projection the caller's flag, and a literal has no constant-space
    // address. The function is always_inline, so a value and a reference generate the same AIR.
    uint src_row, uint epilogue, ulong w_offset, ulong w_offset2,
    threadgroup float * partial,
    uint3 tgid, uint lane, uint sgid, uint nsg)
{
    // 1 when there is one stack, so every `s` loop below unrolls away and the single-stack
    // kernels compile to what they compiled to before the pair existed.
    constexpr uint NS = PAIR ? 2u : 1u;
    const uint g0 = tgid.x * BLK_MMA_TILES;              // the threadgroup's first 8-row unit
    if (g0 * TM_UNIT_ROWS >= n_out || n_tok == 0u) { return; }
    // The 8x8 layout MLX's steel GEMM stores by (BaseMMAFrag::get_coord): lane holds row fr and
    // the two columns fc, fc + 1; q is its pair of columns.
    const uint qd = lane / 4u;
    const uint fr = (qd & 4u) + ((lane / 2u) % 4u);
    const uint fc = (qd & 2u) * 2u + (lane % 2u) * 2u;
    const uint q = fc / 2u;
    const uint spb = tm_block_elems() / 32u;
    const uint bb = tm_block_bytes();
    const uint scb = tm_scale_bytes();
    const uint payb = bb - scb;
    const uint blocks = n_in / tm_block_elems();
    const uint subs = n_in / 32u;
    const uint so = tm_scale_src_off();
    const uint unit = TM_UNIT_ROWS * bb;
    // Row fr of each unit: its row-major blocks, or its unit's blocks (slot fr) in tile-major.
    // A row-major tensor's last unit may be short; its missing rows read the last row.
    device const uchar * base[2][BLK_MMA_MAX_TILES];
    #pragma unroll
    for (uint s = 0; s < NS; ++s) {
        const ulong wo = (s == 0u) ? w_offset : w_offset2;
        #pragma unroll
        for (uint u = 0; u < BLK_MMA_TILES; ++u) {
            const uint r = min((g0 + u) * TM_UNIT_ROWS + fr, n_out - 1u);
            base[s][u] = WFMT_ROW ? weights + wo + (ulong)r * blocks * bb
                                  : weights + wo + (ulong)(r / TM_UNIT_ROWS) * blocks * unit;
        }
    }
    // The activations' k for slot fr of fragment f is k0 + f in either order.
    const uint k0 = tm_run_order() == 1u ? ((fr & 1u) != 0u ? 14u + 2u * fr : 2u * fr) : 4u * fr;
    const uint last = n_tok - 1u;
    // THE ONE ROUTED DIFFERENCE. A work row's activations are the TOKEN's, at perm[row] --
    // except on the down projection, whose input already has one row per work row.
    // TF TOKEN FRAGMENTS of 8 rows: fragment tf holds rows 8 tf + fc and 8 tf + fc + 1, and every
    // weight fragment decoded below multiplies all of them -- one weight read for up to 8 TF rows.
    ulong xo0[TF], xo1[TF];
    BLK_TF_UNROLL
    for (uint tf = 0; tf < TF; ++tf) {
        const uint w0 = src_row + min(8u * tf + fc, last);
        const uint w1 = src_row + min(8u * tf + fc + 1u, last);
        const uint a0 = (ROUTED && src_work_rows == 0u) ? perm[w0] : w0;
        const uint a1 = (ROUTED && src_work_rows == 0u) ? perm[w1] : w1;
        xo0[tf] = (ulong)a0 * n_in + k0;
        xo1[tf] = (ulong)a1 * n_in + k0;
    }
    simdgroup_float8x8 acc[TF][2][BLK_MMA_MAX_TILES];
    float2 skip_acc[TF][2][BLK_MMA_MAX_TILES];
    BLK_TF_UNROLL
    for (uint tf = 0; tf < TF; ++tf) {
    #pragma unroll
    for (uint s = 0; s < NS; ++s) {
        #pragma unroll
        for (uint u = 0; u < BLK_MMA_TILES; ++u) {
            acc[tf][s][u] = simdgroup_float8x8(0.0f);
            skip_acc[tf][s][u] = float2(0.0f);
        }
    }
    }
    // A HALF-BLOCK A STEP (Q4_K, Q5_K, Q3_K): four sub-blocks, their four scales unpacked from the
    // header words together instead of once a run -- the packed six-bit fields cost 4.85 ms a step
    // at 2 rows when each run unpacks its own, and this takes 2.9 of it with no change to the
    // bytes on disk. The tail below serves a row whose sub-blocks are not a multiple of four (no
    // served shape has one).
    if (tm_run_quads()) {
        for (uint qb = sgid; 4u * qb < subs; qb += nsg) {
            const uint sb0 = 4u * qb;
            const uint blk = sb0 / spb, sub0 = sb0 % spb;
            float4 qsa[2][BLK_MMA_MAX_TILES], qsb[2][BLK_MMA_MAX_TILES];
            device const uchar * scu[2][BLK_MMA_MAX_TILES];
            device const uchar * payu[2][BLK_MMA_MAX_TILES];
            #pragma unroll
            for (uint s = 0; s < NS; ++s) {
            #pragma unroll
            for (uint u = 0; u < BLK_MMA_TILES; ++u) {
                if (WFMT_ROW) {
                    device const uchar * b = base[s][u] + (ulong)blk * bb;
                    scu[s][u]  = b + so;
                    payu[s][u] = b + (so == 0u ? scb : 0u);
                } else {
                    device const uchar * un = base[s][u] + (ulong)blk * unit;
                    scu[s][u]  = un + fr * scb;
                    payu[s][u] = un + TM_UNIT_ROWS * scb + fr * payb;
                }
                tm_quad_scales(scu[s][u], sub0, q, qsa[s][u], qsb[s][u]);
            }
            }
            #pragma unroll
            for (uint i = 0; i < 4u; i += (tm_run_pairs() ? 2u : 1u)) {
                const uint sb = sb0 + i;
                simdgroup_float8x8 xm0[TF][4], xm1[TF][4];
                BLK_TF_UNROLL
                for (uint tf = 0; tf < TF; ++tf) {
                float4 xa0, xa1, xb0, xb1;
                if (BLK_MMA_SKIP & 4u) {
                    xa0 = float4(float(sb + lane));
                    xa1 = float4(float(sb) * 0.5f);
                    xb0 = xa0 + 1.0f;
                    xb1 = xa1 + 1.0f;
                } else {
                    xa0 = BLK_MMA_X4(xo0[tf] + sb * 32u);
                    xa1 = BLK_MMA_X4(xo1[tf] + sb * 32u);
                    xb0 = tm_run_pairs() ? BLK_MMA_X4(xo0[tf] + sb * 32u + 32u) : float4(0.0f);
                    xb1 = tm_run_pairs() ? BLK_MMA_X4(xo1[tf] + sb * 32u + 32u) : float4(0.0f);
                }
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) {
                    rm_frag_set(xm0[tf][f], xa0[f], xa1[f]);
                    rm_frag_set(xm1[tf][f], xb0[f], xb1[f]);
                }
                }
                #pragma unroll
                for (uint s = 0; s < NS; ++s) {
                #pragma unroll
                for (uint u = 0; u < BLK_MMA_TILES; ++u) {
                    float4 lo0, hi0, lo1, hi1;
                    if (BLK_MMA_SKIP & 2u) {
                        lo0 = float4(float(lane + u), float(sb), float(sub0 + i + lane), 1.0f);
                        hi0 = float4(float(sb + u), float(lane), 2.0f, float(sub0));
                        lo1 = lo0 + 1.0f;
                        hi1 = hi0 + 1.0f;
                    } else if (tm_run_pairs()) {
                        tm_run8_pair_q(qsa[s][u], qsb[s][u], i, payu[s][u], sub0 + i, q,
                                       lo0, hi0, lo1, hi1);
                    } else {
                        tm_run8_q(qsa[s][u], qsb[s][u], i, payu[s][u], sub0 + i, q, !WFMT_ROW,
                                  lo0, hi0);
                        lo1 = float4(0.0f);
                        hi1 = float4(0.0f);
                    }
                    simdgroup_float8x8 wm0[4], wm1[4];
                    #pragma unroll
                    for (uint f = 0; f < 4u; ++f) {
                        rm_frag_set(wm0[f], lo0[f], hi0[f]);
                        rm_frag_set(wm1[f], lo1[f], hi1[f]);
                    }
                    if (BLK_MMA_SKIP & 1u) {
                        BLK_TF_UNROLL
                        for (uint tf = 0; tf < TF; ++tf) {
                        #pragma unroll
                        for (uint f = 0; f < 4u; ++f) {
                            skip_acc[tf][s][u] += rm_frag_get(wm0[f]) * rm_frag_get(xm0[tf][f]);
                            if (tm_run_pairs()) {
                                skip_acc[tf][s][u] += rm_frag_get(wm1[f]) * rm_frag_get(xm1[tf][f]);
                            }
                        }
                        }
                        continue;
                    }
                    BLK_TF_UNROLL
                    for (uint tf = 0; tf < TF; ++tf) {
                    #pragma unroll
                    for (uint f = 0; f < 4u; ++f) {
                        simdgroup_multiply_accumulate(acc[tf][s][u], wm0[f], xm0[tf][f], acc[tf][s][u]);
                    }
                    if (tm_run_pairs()) {
                        #pragma unroll
                        for (uint f = 0; f < 4u; ++f) {
                            simdgroup_multiply_accumulate(acc[tf][s][u], wm1[f], xm1[tf][f], acc[tf][s][u]);
                        }
                    }
                    }
                }
                }
            }
        }
        // The sub-blocks past the last whole half-block, one at a time.
        for (uint sb = (subs & ~3u) + sgid; sb < subs; sb += nsg) {
            const uint blk = sb / spb, sub = sb % spb;
            simdgroup_float8x8 xm[TF][4];
            BLK_TF_UNROLL
            for (uint tf = 0; tf < TF; ++tf) {
                const float4 xa0 = BLK_MMA_X4(xo0[tf] + sb * 32u);
                const float4 xa1 = BLK_MMA_X4(xo1[tf] + sb * 32u);
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) { rm_frag_set(xm[tf][f], xa0[f], xa1[f]); }
            }
            #pragma unroll
            for (uint s = 0; s < NS; ++s) {
            #pragma unroll
            for (uint u = 0; u < BLK_MMA_TILES; ++u) {
                device const uchar * sc;
                device const uchar * pay;
                if (WFMT_ROW) {
                    device const uchar * b = base[s][u] + (ulong)blk * bb;
                    sc  = b + so;
                    pay = b + (so == 0u ? scb : 0u);
                } else {
                    device const uchar * un = base[s][u] + (ulong)blk * unit;
                    sc  = un + fr * scb;
                    pay = un + TM_UNIT_ROWS * scb + fr * payb;
                }
                float4 lo, hi;
                tm_run8(sc, pay, sub, q, !WFMT_ROW, lo, hi);
                simdgroup_float8x8 wm[4];
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) { rm_frag_set(wm[f], lo[f], hi[f]); }
                BLK_TF_UNROLL
                for (uint tf = 0; tf < TF; ++tf) {
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) {
                    simdgroup_multiply_accumulate(acc[tf][s][u], wm[f], xm[tf][f], acc[tf][s][u]);
                }
                }
            }
            }
        }
    } else if (tm_run_pairs()) {
        for (uint pb = sgid; 2u * pb < subs; pb += nsg) {
            const uint sb0 = 2u * pb;
            const uint blk = sb0 / spb, sub0 = sb0 % spb;
            simdgroup_float8x8 xm0[TF][4], xm1[TF][4];
            BLK_TF_UNROLL
            for (uint tf = 0; tf < TF; ++tf) {
            float4 xa0, xa1, xb0, xb1;
            if (BLK_MMA_SKIP & 4u) {
                xa0 = float4(float(sb0 + lane));
                xa1 = float4(float(sb0) * 0.5f);
                xb0 = float4(float(sb0 + 1u + lane));
                xb1 = float4(float(sb0 + 1u) * 0.5f);
            } else {
                xa0 = BLK_MMA_X4(xo0[tf] + sb0 * 32u);
                xa1 = BLK_MMA_X4(xo1[tf] + sb0 * 32u);
                xb0 = BLK_MMA_X4(xo0[tf] + sb0 * 32u + 32u);
                xb1 = BLK_MMA_X4(xo1[tf] + sb0 * 32u + 32u);
            }
            #pragma unroll
            for (uint f = 0; f < 4u; ++f) {
                rm_frag_set(xm0[tf][f], xa0[f], xa1[f]);
                rm_frag_set(xm1[tf][f], xb0[f], xb1[f]);
            }
            }
            #pragma unroll
            for (uint s = 0; s < NS; ++s) {
            #pragma unroll
            for (uint u = 0; u < BLK_MMA_TILES; ++u) {
                device const uchar * sc;
                device const uchar * pay;
                if (WFMT_ROW) {
                    device const uchar * b = base[s][u] + (ulong)blk * bb;
                    sc  = b + so;
                    pay = b + (so == 0u ? scb : 0u);
                } else {
                    device const uchar * un = base[s][u] + (ulong)blk * unit;
                    sc  = un + fr * scb;
                    pay = un + TM_UNIT_ROWS * scb + fr * payb;
                }
                float4 lo0, hi0, lo1, hi1;
                if (BLK_MMA_SKIP & 2u) {
                    lo0 = float4(float(lane + u), float(sb0), float(sub0 + lane), 1.0f);
                    hi0 = float4(float(sb0 + u), float(lane), 2.0f, float(sub0));
                    lo1 = lo0 + 1.0f;
                    hi1 = hi0 + 1.0f;
                } else {
                    tm_run8_pair(sc, pay, sub0, q, lo0, hi0, lo1, hi1);
                }
                simdgroup_float8x8 wm0[4], wm1[4];
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) {
                    rm_frag_set(wm0[f], lo0[f], hi0[f]);
                    rm_frag_set(wm1[f], lo1[f], hi1[f]);
                }
                if (BLK_MMA_SKIP & 1u) {
                    BLK_TF_UNROLL
                    for (uint tf = 0; tf < TF; ++tf) {
                    #pragma unroll
                    for (uint f = 0; f < 4u; ++f) {
                        skip_acc[tf][s][u] += rm_frag_get(wm0[f]) * rm_frag_get(xm0[tf][f])
                                            + rm_frag_get(wm1[f]) * rm_frag_get(xm1[tf][f]);
                    }
                    }
                    continue;
                }
                BLK_TF_UNROLL
                for (uint tf = 0; tf < TF; ++tf) {
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) {
                    simdgroup_multiply_accumulate(acc[tf][s][u], wm0[f], xm0[tf][f], acc[tf][s][u]);
                }
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) {
                    simdgroup_multiply_accumulate(acc[tf][s][u], wm1[f], xm1[tf][f], acc[tf][s][u]);
                }
                }
            }
            }
        }
    } else {
        for (uint sb = sgid; sb < subs; sb += nsg) {
            const uint blk = sb / spb, sub = sb % spb;
            simdgroup_float8x8 xm[TF][4];
            BLK_TF_UNROLL
            for (uint tf = 0; tf < TF; ++tf) {
            float4 xa0, xa1;
            if (BLK_MMA_SKIP & 4u) {
                xa0 = float4(float(sb + lane));
                xa1 = float4(float(sb) * 0.5f);
            } else {
                xa0 = BLK_MMA_X4(xo0[tf] + sb * 32u);
                xa1 = BLK_MMA_X4(xo1[tf] + sb * 32u);
            }
            #pragma unroll
            for (uint f = 0; f < 4u; ++f) { rm_frag_set(xm[tf][f], xa0[f], xa1[f]); }
            }
            #pragma unroll
            for (uint sk = 0; sk < NS; ++sk) {
            #pragma unroll
            for (uint u = 0; u < BLK_MMA_TILES; ++u) {
                device const uchar * sc;
                device const uchar * pay;
                if (WFMT_ROW) {
                    device const uchar * b = base[sk][u] + (ulong)blk * bb;
                    sc  = b + so;
                    pay = b + (so == 0u ? scb : 0u);
                } else {
                    device const uchar * un = base[sk][u] + (ulong)blk * unit;
                    sc  = un + fr * scb;
                    pay = un + TM_UNIT_ROWS * scb + fr * payb;
                }
                float4 lo, hi;
                if (BLK_MMA_SKIP & 2u) {
                    lo = float4(float(lane + u), float(sb), float(sub + lane), 1.0f);
                    hi = float4(float(sb + u), float(lane), 2.0f, float(sub));
                } else {
                    tm_run8(sc, pay, sub, q, !WFMT_ROW, lo, hi);
                }
                simdgroup_float8x8 wm[4];
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) { rm_frag_set(wm[f], lo[f], hi[f]); }
                if (BLK_MMA_SKIP & 1u) {
                    BLK_TF_UNROLL
                    for (uint tf = 0; tf < TF; ++tf) {
                    #pragma unroll
                    for (uint f = 0; f < 4u; ++f) {
                        skip_acc[tf][sk][u] += rm_frag_get(wm[f]) * rm_frag_get(xm[tf][f]);
                    }
                    }
                    continue;
                }
                BLK_TF_UNROLL
                for (uint tf = 0; tf < TF; ++tf) {
                #pragma unroll
                for (uint f = 0; f < 4u; ++f) {
                    simdgroup_multiply_accumulate(acc[tf][sk][u], wm[f], xm[tf][f], acc[tf][sk][u]);
                }
                }
            }
            }
        }
    }
    if (BLK_MMA_SKIP & 1u) {
        BLK_TF_UNROLL
        for (uint tf = 0; tf < TF; ++tf) {
        #pragma unroll
        for (uint s = 0; s < NS; ++s) {
            #pragma unroll
            for (uint u = 0; u < BLK_MMA_TILES; ++u) {
                rm_frag_set(acc[tf][s][u], skip_acc[tf][s][u].x, skip_acc[tf][s][u].y);
            }
        }
        }
    }
    // The simdgroups' sums through threadgroup memory, added in simdgroup order: 64 floats a unit.
    // THE SECOND STACK'S SUMS SIT AFTER EVERY SIMDGROUP'S FIRST, so the read below walks the
    // same stride for both and the single-stack layout is untouched.
    // A simdgroup's sums for fragment tf sit after its sums for the fragments before it.
    const uint per_frag = BLK_MMA_TILES * 64u;
    const uint per_sg = TF * per_frag;
    const uint stack_span = nsg * per_sg;
    BLK_TF_UNROLL
    for (uint tf = 0; tf < TF; ++tf) {
    #pragma unroll
    for (uint s = 0; s < NS; ++s) {
        #pragma unroll
        for (uint u = 0; u < BLK_MMA_TILES; ++u) {
            threadgroup float * o =
                partial + s * stack_span + sgid * per_sg + tf * per_frag + u * 64u + fr * 8u + fc;
            const float2 e = rm_frag_get(acc[tf][s][u]);
            o[0] = e.x;
            o[1] = e.y;
        }
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = sgid * 32u + lane; i < per_sg; i += nsg * 32u) {
        const uint j = i % per_frag;
        const uint u = j / 64u, row = (j / 8u) % 8u, tok = 8u * (i / per_frag) + j % 8u;
        const uint orow = (g0 + u) * TM_UNIT_ROWS + row;
        if (tok >= n_tok || orow >= n_out) { continue; }
        float total = 0.0f;
        for (uint s = 0; s < nsg; ++s) { total += partial[s * per_sg + i]; }
        device float * dst = y + (ulong)tok * n_out + orow;
        if (PAIR) {
            // The second stack's own sum, added in the same simdgroup order. The product is
            // imparo_act_mul's expression over the two values the three dispatches wrote to
            // device memory, so the row is bit-identical to them.
            float total2 = 0.0f;
            for (uint s = 0; s < nsg; ++s) { total2 += partial[stack_span + s * per_sg + i]; }
            *dst = epilogue ? (imparo_act_f(total) * total2) : total;
        } else {
            *dst = epilogue ? (imparo_act_f(*dst) * total) : total;
        }
    }
}
// THE HALF HANDOFF. The staged GEMM rounds its activation tile to half in threadgroup memory,
// so a source that already holds half(the float) -- the norm's mirror of the FFN input, or
// the pair's own half output -- gives the same tile at half the device bytes. SRC_HALF reads
// the activation rows as half; DST_HALF writes the tile's result as half (the pair's
// act(gate) * up, which the down projection only ever stages as half).
constant bool MOE_ST_SRC_HALF_FC [[function_constant(61)]];
constant bool MOE_ST_SRC_HALF = is_function_constant_defined(MOE_ST_SRC_HALF_FC) && MOE_ST_SRC_HALF_FC;
constant bool MOE_ST_DST_HALF_FC [[function_constant(62)]];
constant bool MOE_ST_DST_HALF = is_function_constant_defined(MOE_ST_DST_HALF_FC) && MOE_ST_DST_HALF_FC;


// One (row, sub-block) of the weight tile into ws, EIGHT VALUES AT A TIME (tm_run8, the
// decoder the block-rows kernel uses, bit-exact against tm_sub32): at most eight decoded
// floats are live, where a whole sub-block held 32 registers beside the accumulators. At a
// 5955-token prefill: gate|up pair 1640 -> 1624 ms (Q4_K_M), 1617 -> 1607 (Q4_0); the down
// projection 818 -> 810 on Q4_0, a tie on Q4_K_M.
// Full unrolls in the routed staged GEMM, not the "#pragma unroll" hint: at a 17115-token
// prefill the pair 4529 -> 4476 ms and the down projection 2340 -> 2320 (Q4_0: 4496 -> 4441,
// 2281 -> 2256), same bits.
#define MOE_ST_UNROLL _Pragma("clang loop unroll(full)")
template <uint ROW_TILES>
static inline __attribute__((always_inline)) void moe_st_stage_sub(
    threadgroup half * wt, device const uchar * sc, device const uchar * pay, uint sub,
    uint sc_, uint rr) {
    // Q6_K TILE-MAJOR: the sub-block's 32 low-bit and 32 high-bit bytes as four 16-byte words
    // and its two scales, loaded once for the four runs; each value is tm_run8_t's expression.
    // Down projection at a 17115-token prefill -0.5..-0.8%, same bits; the same form for Q4_K
    // tied.
    const bool Q6W = WFMT == WF_Q6_K && !WFMT_ROW;
    uint4 q6l0 = 0u, q6l1 = 0u, q6h0 = 0u, q6h1 = 0u;
    float q6ds0 = 0.0f, q6ds1 = 0.0f;
    const uint q6c = sub & 3u;
    if (Q6W) {
        const uint n = sub >> 2;
        const half d = as_type<half>((ushort)(sc[16] | (sc[17] << 8)));
        device const char * s8 = (device const char *)sc + n * 8u;
        q6ds0 = float(d) * float((int)s8[2u * q6c]);
        q6ds1 = float(d) * float((int)s8[2u * q6c + 1u]);
        device const uint4 * l4 = (device const uint4 *)(pay + n * 64u + (q6c & 1u) * 32u);
        device const uint4 * h4 = (device const uint4 *)(pay + 128u + n * 32u);
        q6l0 = l4[0]; q6l1 = l4[1]; q6h0 = h4[0]; q6h1 = h4[1];
    }
    MOE_ST_UNROLL
    for (uint q = 0; q < 4u; ++q) {
        float4 lo, hi;
        if (Q6W) {
            const float ds = (q >> 1) == 0u ? q6ds0 : q6ds1;
            const float cc = -32.0f * ds;
            const uint shift = 2u * q6c, shift4 = (q6c >= 2u) ? 4u : 0u;
            const uint4 lq = (q >> 1) == 0u ? q6l0 : q6l1, hq = (q >> 1) == 0u ? q6h0 : q6h1;
            const uint2 lw = (q & 1u) == 0u ? lq.xy : lq.zw, hw = (q & 1u) == 0u ? hq.xy : hq.zw;
            const uchar4 a = as_type<uchar4>(((lw.x >> shift4) & 0x0F0F0F0Fu)
                                           | (((hw.x >> shift) & 0x03030303u) << 4));
            const uchar4 b = as_type<uchar4>(((lw.y >> shift4) & 0x0F0F0F0Fu)
                                           | (((hw.y >> shift) & 0x03030303u) << 4));
            lo = float4(fma(ds, float(a.x), cc), fma(ds, float(a.y), cc),
                        fma(ds, float(a.z), cc), fma(ds, float(a.w), cc));
            hi = float4(fma(ds, float(b.x), cc), fma(ds, float(b.y), cc),
                        fma(ds, float(b.z), cc), fma(ds, float(b.w), cc));
        } else {
            tm_run8(sc, pay, sub, q, !WFMT_ROW, lo, hi);
        }
        // THE RUN'S K POSITIONS ARE THE FORMAT'S (tm_run_order): order 0 is k 8q .. 8q+7,
        // order 1 (the nibble formats) k 4q .. 4q+3 then 16+4q .. 16+4q+3. Treating every run
        // as eight consecutive positions put Q4_0's values at the wrong K.
        const bool o1 = tm_run_order() == 1u;
        const uint klo = sc_ * 32u + (o1 ? 4u * q : 8u * q);
        const uint khi = sc_ * 32u + (o1 ? 16u + 4u * q : 8u * q + 4u);
        MOE_ST_UNROLL
        for (uint i = 0; i < 4u; ++i) {
            const uint k0 = klo + i, k1 = khi + i;
            wt[((k0 / 8u) * ROW_TILES + rr / 8u) * 64u + (k0 % 8u) * 8u + rr % 8u] = half(lo[i]);
            wt[((k1 / 8u) * ROW_TILES + rr / 8u) * 64u + (k1 % 8u) * 8u + rr % 8u] = half(hi[i]);
        }
    }
}

// THE ROUTED FEED-FORWARD AS A STAGED GEMM -- the design k-quants never had.
//
// WHY THIS EXISTS. This engine has two prefill GEMM designs. st_gemm stages BOTH operands as
// half in threadgroup memory and is the default for Q8_0 because it wins; rt_gemm keeps the
// weights staged and the activations in a register prefetch, and it is the ONLY design a
// k-quant has. Measured on Q8_0 at one expert's shape, where the two can be switched with
// IMPARO_Q8_DESIGN:
//
//     rows        staged   register-tiled   ratio
//       64       0.1547         0.2522      1.63x
//      128       0.2861         0.3572      1.25x
//      512       0.7777         0.8716      1.12x
//
// The staged design wins MOST at the thin shapes, and a routed layer is nothing but thin
// shapes: 2048 work rows over 32 experts is 64 rows each. So every k-quant routed matmul --
// 78.9% of a 512-token prefill chunk on LFM2.5-8B-A1B -- runs the losing design today.
//
// THE SHAPE. One threadgroup owns ROWS output rows x TOKENS work rows OF ONE EXPERT and walks
// K at a time. The weight tile is dequantised into threadgroup memory through the same brick
// decoder every other kernel uses, so this adds no format knowledge; the activation tile is
// gathered through perm, because a work row's activations are its TOKEN's.
//
// THE GRID IS HOST-COMPUTABLE AND NEEDS NO SYNC. A token never picks the same expert twice,
// so an expert holds at most n_tok work rows: ceil(n_tok / TOKENS) token groups covers any
// routing. A group past its expert's segment reads seg and returns, which is what llama.cpp's
// mul_mm_id does with tpe[e] and what vLLM does with expert_ids == -1.
// THE SHAPE IS A TEMPLATE, and the names below are its instantiations, so a shape is swept
// by picking a kernel and not by rebuilding. ROWS x TOKENS is the output tile of ONE expert,
// K is what one chunk stages, and (RT_SG, TT_SG) is how a simdgroup's share of the tile is
// cut -- RT_SG * TT_SG accumulators per simdgroup, NSG simdgroups covering the whole tile.
// ONE PASS of the staged routed GEMM over a K chunk: stage the weight tile of `wsrc` rows
// rbase .. rbase + ROWS (and, on the first pass, the activation tile), then multiply it into
// `acc`. A function called once per pass with a NAMED accumulator array, not a loop over an
// array of them: a pass loop indexing acc[p] left the index dynamic, the accumulators went to
// memory, and the kernel ran 2.8x slower with the same bits.
// The k loop over a staged tile for the first LIVE_TT of a simdgroup's TT_SG token fragments.
template <uint K, uint ROW_TILES, uint TOK_TILES, uint RT_SG, uint TT_SG, uint LIVE_TT>
static inline __attribute__((always_inline)) void moe_st_mma(
    thread simdgroup_float8x8 * acc, threadgroup half * ws, threadgroup half * as,
    uint row_tile0, uint tok_tile0)
{
    MOE_ST_UNROLL
    for (uint kk = 0; kk < K; kk += 8u) {
        simdgroup_half8x8 wm[RT_SG];
        simdgroup_half8x8 am[LIVE_TT];
        // A THREADGROUP BARRIER IS NOT ENOUGH BEFORE simdgroup_load FROM THREADGROUP MEMORY.
        // st_gemm carries the same fence (Q8_MMA_FENCE) around these loads. Without it this
        // kernel read the tile while it was still being written: the output varied run to run
        // by whole logits. Once a staged tile (kk == 0) instead of every k-step measured +2.2%
        // SLOWER; no fence at all +3.4% (2026-09-24).
        simdgroup_barrier(mem_flags::mem_threadgroup);
        MOE_ST_UNROLL
        for (uint ri = 0; ri < RT_SG; ++ri) {
            simdgroup_load(wm[ri], ws + ((kk / 8u) * ROW_TILES + row_tile0 + ri) * 64u, 8u);
        }
        MOE_ST_UNROLL
        for (uint ti = 0; ti < LIVE_TT; ++ti) {
            simdgroup_load(am[ti], as + ((kk / 8u) * TOK_TILES + tok_tile0 + ti) * 64u, 8u);
        }
        MOE_ST_UNROLL
        for (uint ti = 0; ti < LIVE_TT; ++ti) {
            MOE_ST_UNROLL
            for (uint ri = 0; ri < RT_SG; ++ri) {
                simdgroup_multiply_accumulate(acc[ti * RT_SG + ri], am[ti], wm[ri],
                                              acc[ti * RT_SG + ri]);
            }
        }
    }
}

template <uint ROWS, uint TOKENS, uint K, uint NSG, uint RT_SG, uint TT_SG>
static inline __attribute__((always_inline)) void moe_st_pass(
    thread simdgroup_float8x8 * acc, bool stage_act, device const uchar * wsrc, ulong wb,
    uint rbase, uint c00, device const float * x, threadgroup half * ws, threadgroup half * as,
    threadgroup ulong * abase, uint n_out, uint spb, uint bb, uint scb, uint payb, uint blocks,
    uint so, ulong unit, uint tid, uint row_tile0, uint tok_tile0, uint ntok)
{
    constexpr uint THREADS   = NSG * 32u;
    constexpr uint ROW_TILES = ROWS / 8u;
    constexpr uint TOK_TILES = TOKENS / 8u;
    constexpr uint subs_chk  = K / 32u;              // sub-blocks of K one row stages
    // ---- the weight tile, dequantised once per (row, sub-block) ------------------------
    // One work item per (row, sub-block): ROWS * subs_chk == the thread count at the shipped
    // shape, so no thread idles in the staging that every multiply waits on.
    for (uint it = tid; it < ROWS * subs_chk; it += THREADS) {
        const uint rr  = it / subs_chk;
        const uint sc_ = it % subs_chk;
        const uint sb  = (c00 / 32u) + sc_;             // this row's sub-block index
        const uint blk = sb / spb, sub = sb % spb;
        // Clamped rather than branched: an out-of-range row stages a duplicate and the
        // write-back masks it, so the tile's shape never depends on the edge.
        const uint rsrc = min(rbase + rr, n_out - 1u);
        device const uchar * sc;
        device const uchar * pay;
        if (WFMT_ROW) {
            device const uchar * bk = wsrc + wb + ((ulong)rsrc * blocks + blk) * bb;
            sc  = bk + so;
            pay = bk + (so == 0u ? scb : 0u);
        } else {
            device const uchar * un =
                wsrc + wb + ((ulong)(rsrc / TM_UNIT_ROWS) * blocks + blk) * unit;
            const uint slot = rsrc % TM_UNIT_ROWS;
            sc  = un + slot * scb;
            pay = un + TM_UNIT_ROWS * scb + slot * payb;
        }
        // FLOAT INTERMEDIATES, ONE ROUNDING AT THE END: the dequant runs in float and the
        // half appears only on the store, which is the rule every other decode here follows.
        moe_st_stage_sub<ROW_TILES>(ws, sc, pay, sub, sc_, rr);
    }
    // ---- the activation tile, once a K chunk, from the resolved bases ------------------
    // n_in, c00 and the k span are all multiples of four, so every one of these is a 16-byte
    // aligned float4 rather than four scalar loads.
    if (stage_act) {
        static_assert((TOKENS * (K / 4u)) % THREADS == 0u, "the activation tile divides");
        MOE_ST_UNROLL
        for (uint f = 0; f < (TOKENS * (K / 4u)) / THREADS; ++f) {
            const uint it = tid + f * THREADS;
            const uint tt = it / (K / 4u);
            const uint k4 = (it % (K / 4u)) * 4u;
            // TOKEN-MAJOR WITHIN THE TILE, which is what the left operand of `am * wm` has to
            // be: am is A[token][k] and wm is W[k][row]. The weight tile is k-major because it
            // is the RIGHT operand. Staging this one k-major too -- which it was -- transposes
            // A and computes a different product. st_gemm stages the same two orientations.
            threadgroup half * at = as + ((k4 / 8u) * TOK_TILES + (tt / 8u)) * 64u
                                  + (tt % 8u) * 8u + (k4 % 8u);
            if (MOE_ST_SRC_HALF) {
                *(threadgroup half4 *)at =
                    *(device const half4 *)((device const half *)x + abase[tt] + c00 + k4);
                continue;
            }
            const float4 xv = *(device const float4 *)(x + abase[tt] + c00 + k4);
            *(threadgroup half4 *)at = half4(xv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // A SIMDGROUP WHOSE TOKEN ROWS ARE ALL PAST THE EXPERT'S LAST WORK ROW STORES NOTHING, so
    // it skips the multiply -- decided once a pass, uniform across the simdgroup. An expert's
    // last tile is on average half padding: pair 1614 -> 1590 ms, down 836 -> 821, prefill
    // 3819 -> 3780 ms at 5955 tokens (Q4_K_M), bits identical. With it, 64-token tiles still
    // lose (down 903 / 856 with two row blocks, pair 1635).
    // ONE WHOSE SECOND TOKEN FRAGMENT IS DEAD runs the one-fragment loop: pair 4492 -> 4459,
    // down 2317 -> 2300 ms at 17115 tokens (Q4_0 alike), bits identical. The choice is ONE
    // branch before two loops of constant trip count; the same test inside the loop cost 10%,
    // and a bound written `kk < (sg_live ? K : 0u)` ran as slow as the loop with its unroll
    // disabled (pair 1722 against 1569 at 5955 tokens).
    const bool sg_live = tok_tile0 * 8u < ntok;
    const bool tail1 = TT_SG > 1u && sg_live && (tok_tile0 + 1u) * 8u >= ntok;
    if (tail1) {
        moe_st_mma<K, ROW_TILES, TOK_TILES, RT_SG, TT_SG, 1u>(acc, ws, as, row_tile0, tok_tile0);
    } else if (sg_live) {
        moe_st_mma<K, ROW_TILES, TOK_TILES, RT_SG, TT_SG, TT_SG>(acc, ws, as, row_tile0, tok_tile0);
    }
    // The next pass restages ws (and the next chunk as): every read of them ends here.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One row block's result straight from its fragments (MLX steel's store_result): each lane
// writes its two elements of every 8x8 tile to device memory, masked by row. `u` is the up
// stack's accumulators when paired -- act(gate) * up in registers, the expression
// imparo_act_mul evaluates over the same two floats -- and ignored otherwise.
template <uint RT_SG, uint TT_SG, bool PAIR>
static inline __attribute__((always_inline)) void moe_st_store_block(
    thread simdgroup_float8x8 * g, thread simdgroup_float8x8 * u, uint rowoff, uint nrows,
    device float * y, uint n_out, uint c0, uint r0, uint ntok, uint row_tile0, uint tok_tile0,
    uint lane)
{
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    MOE_ST_UNROLL
    for (uint ti = 0; ti < TT_SG; ++ti) {
        const uint tt = (tok_tile0 + ti) * 8u + fm;
        if (tt >= ntok) { continue; }
        MOE_ST_UNROLL
        for (uint ri = 0; ri < RT_SG; ++ri) {
            const uint rr = rowoff + (row_tile0 + ri) * 8u + fn;
            float2 v = rm_frag_get(g[ti * RT_SG + ri]);
            if (PAIR) {
                const float2 uv = rm_frag_get(u[ti * RT_SG + ri]);
                v = float2(imparo_act_f(v.x) * uv.x, imparo_act_f(v.y) * uv.y);
            }
            const ulong o = (ulong)(c0 + tt) * n_out + r0 + rr;
            if (rr + 1u < nrows) {
                if (MOE_ST_DST_HALF) { *(device half2 *)((device half *)y + o) = half2(v); }
                else                 { *(device float2 *)(y + o) = v; }
            } else if (rr < nrows) {
                if (MOE_ST_DST_HALF) { ((device half *)y)[o] = half(v.x); }
                else                 { y[o] = v.x; }
            }
        }
    }
}

// RB: the single projection over RB (1 or 2) row blocks a threadgroup, the pair's loop with
// the next ROWS rows of the same weight in place of the up stack. The activation tile is
// staged once for RB * ROWS rows, as the pair stages it once for gate and up.
template <uint ROWS, uint TOKENS, uint K, uint NSG, uint RT_SG, uint TT_SG,
          bool PAIR = false, uint RB = 1u>
static inline __attribute__((always_inline)) void moe_st_gemm_body(
    device const uchar * weights, device const uchar * weights2, device const float * x,
    device float * y, device const uint * perm, device const uint * seg,
    constant ulong & w_offset, constant ulong & w_offset2, constant ulong & expert_stride,
    constant uint & n_in, constant uint & n_out, constant uint & src_work_rows,
    constant uint & n_expert,
    threadgroup half * shared, uint3 tgid, uint tid, uint sgid, uint lane)
{
    constexpr uint THREADS   = NSG * 32u;
    constexpr uint ROW_TILES = ROWS / 8u;
    constexpr uint TOK_TILES = TOKENS / 8u;
    constexpr uint K_TILES   = K / 8u;
    static_assert(ROW_TILES * TOK_TILES == NSG * RT_SG * TT_SG,
                  "the simdgroups must cover the output tile exactly, once");
    // The grid spans ACTIVE experts (moe_plan's compacted tail), not all n_expert.
    if (tgid.z >= seg[n_expert + 1u]) { return; }
    const uint e  = seg[n_expert + 2u + tgid.z];
    const uint lo = seg[e];
    const uint hi = seg[e + 1u];
    const uint c0 = lo + tgid.y * TOKENS;      // this group's first work row
    if (c0 >= hi) { return; }                          // past this expert: a launch, nothing else
    const uint ntok = min(TOKENS, hi - c0);
    const uint r0   = tgid.x * ROWS * RB;
    if (r0 >= n_out) { return; }
    const uint nrow = min(ROWS, n_out - r0);

    // ONE THREADGROUP BUFFER, PARTITIONED HERE. Three separate [[threadgroup(n)]] bindings
    // is the obvious way to say "these never alias", and it is the assumption this kernel
    // raced under; one buffer with offsets this file can check removes the question.
    //   ws     K_TILES * ROW_TILES * 64 halves   the weight tile
    //   as     K_TILES * TOK_TILES * 64 halves   the activation tile
    //   abase  TOKENS ulongs                     (8-byte aligned: the tiles are whole tiles)
    // The result leaves from the fragments (below), so nothing is spilled here.
    // ONE WEIGHT TILE, STAGED TWICE when paired. Staging gate and up side by side doubled
    // the buffer to 20480 B and cost more occupancy than the fused activation saved
    // (+5.66% measured). Restaging the SAME tile for the up pass keeps the threadgroup
    // budget identical to the single-projection shape, and the activation tile is still
    // staged once for both.
    constexpr uint STAGE_H = K_TILES * (ROW_TILES + TOK_TILES) * 64u;   // halves
    threadgroup half * ws = shared;
    threadgroup half * as = shared + K_TILES * ROW_TILES * 64u;
    threadgroup ulong * abase = (threadgroup ulong *)(shared + STAGE_H);
    // THE WORK ROWS' ACTIVATION BASES, RESOLVED ONCE. A work row's token is fixed for the
    // whole kernel, so perm belongs before the K loop -- reading it per staged element cost
    // 2048 lookups a chunk and made this kernel twice the rows kernel it replaces.
    for (uint t = tid; t < TOKENS; t += THREADS) {
        const uint w = c0 + min(t, ntok - 1u);
        abase[t] = (ulong)((src_work_rows != 0u) ? w : perm[w]) * n_in;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The simdgroups tile the output: ROW_TILES / RT_SG of them across the rows.
    constexpr uint SG_COLS = ROW_TILES / RT_SG;
    const uint row_tile0 = (sgid % SG_COLS) * RT_SG;
    const uint tok_tile0 = (sgid / SG_COLS) * TT_SG;
    // ONE NAMED ACCUMULATOR ARRAY A PASS (see moe_st_pass): pass 0 is the gate (or the only)
    // stack's first row block, pass 1 the up stack when paired, else the second row block.
    // TWO PASSES AT MOST: 16 fragments a simdgroup fit in registers, 32 do not -- the pair
    // over two row blocks read 7330 ms against 1612 at a 5955-token prefill (Q4_K_M), four
    // row blocks of the down projection 3554 against 840; three row blocks (24) ran 874.
    constexpr uint NP = (PAIR ? 2u : 1u) * RB;
    constexpr uint F  = TT_SG * RT_SG;
    static_assert(NP >= 1u && NP <= 2u, "two accumulator sets fit in registers; more spill");
    simdgroup_float8x8 acc0[F], acc1[NP > 1u ? F : 1u];
    MOE_ST_UNROLL
    for (uint i = 0; i < F; ++i) {
        acc0[i] = simdgroup_float8x8(0.0f);
        if (NP > 1u) { acc1[i] = simdgroup_float8x8(0.0f); }
    }

    const uint  spb      = tm_block_elems() / 32u;      // 32-value sub-blocks per block
    const uint  bb       = tm_block_bytes();
    const uint  scb      = tm_scale_bytes();
    const uint  payb     = bb - scb;
    const uint  blocks   = n_in / tm_block_elems();
    const uint  so       = tm_scale_src_off();
    const ulong unit     = (ulong)TM_UNIT_ROWS * bb;
    const ulong wbase    = w_offset + (ulong)e * expert_stride;
    const ulong wbase2   = w_offset2 + (ulong)e * expert_stride;

    for (uint c00 = 0; c00 < n_in; c00 += K) {
        // Pass p's rows and stack: unpaired, block p; paired, block p / 2, gate then up.
        #define MOE_ST_PASS(ACC, P)                                                             \
            moe_st_pass<ROWS, TOKENS, K, NSG, RT_SG, TT_SG>(                                    \
                ACC, (P) == 0u, (PAIR && ((P) % 2u) == 1u) ? weights2 : weights,                \
                (PAIR && ((P) % 2u) == 1u) ? wbase2 : wbase,                                    \
                r0 + (PAIR ? (P) / 2u : (P)) * ROWS, c00, x, ws, as, abase, n_out, spb, bb,     \
                scb, payb, blocks, so, unit, tid, row_tile0, tok_tile0, ntok)
        MOE_ST_PASS(acc0, 0u);
        if (NP > 1u) { MOE_ST_PASS(acc1, 1u); }
        #undef MOE_ST_PASS
    }

    // ---- THE RESULT STRAIGHT FROM THE FRAGMENTS, one row block at a time -------------------
    // A threadgroup spill and a coalesced copy-out measured a tie against this, and RB > 1 has
    // row blocks the spill never held.
    const uint nrows = min(RB * ROWS, n_out - r0);
    #define MOE_ST_STORE(G, U, B)                                                               \
        moe_st_store_block<RT_SG, TT_SG, PAIR>(G, U, (B) * ROWS, nrows, y, n_out, c0, r0, ntok, \
                                               row_tile0, tok_tile0, lane)
    if (PAIR) {
        MOE_ST_STORE(acc0, acc1, 0u);
    } else {
        MOE_ST_STORE(acc0, acc0, 0u);
        if (RB > 1u) { MOE_ST_STORE(acc1, acc1, 1u); }
    }
    #undef MOE_ST_STORE
}

// The shapes this ships. A kernel cannot be a template in Metal, so these are the names --
// the same form IMPARO_RT_KERNEL uses for the register-tiled GEMM.
#define IMPARO_MOE_ST_KERNEL(NAME, R_, T_, K_, NSG_, RT_, TT_, PAIR_, RB_)                  \
kernel void NAME(                                                                           \
    device const uchar * weights [[buffer(0)]],                                             \
    device const float * x       [[buffer(1)]],                                             \
    device float       * y       [[buffer(2)]],                                             \
    device const uint  * perm    [[buffer(3)]],                                             \
    device const uint  * seg     [[buffer(4)]],                                             \
    constant ulong & w_offset      [[buffer(5)]],                                           \
    constant ulong & expert_stride [[buffer(6)]],                                           \
    constant uint  & n_in          [[buffer(7)]],                                           \
    constant uint  & n_out         [[buffer(8)]],                                           \
    constant uint  & src_work_rows [[buffer(9)]],                                           \
    constant uint  & n_expert      [[buffer(14)]],                                          \
    constant ulong & w_offset2     [[buffer(15)]],                                          \
    device const uchar * weights2  [[buffer(16)]],                                          \
    threadgroup half  * shared [[threadgroup(0)]],                                          \
    uint3 tgid [[threadgroup_position_in_grid]],                                            \
    uint3 tid3 [[thread_position_in_threadgroup]],                                          \
    uint sgid [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])  \
{                                                                                           \
    moe_st_gemm_body<R_, T_, K_, NSG_, RT_, TT_, PAIR_, RB_>(                               \
        weights, weights2, x, y, perm, seg, w_offset, w_offset2, expert_stride, n_in, n_out,\
        src_work_rows, n_expert, shared, tgid, tid3.x, sgid, lane);                         \
}
// THE DOWN PROJECTION'S SHAPE: two 64-row blocks a threadgroup over one staged activation
// tile, the pair's loop with the next 64 rows in place of the up stack. At a 5955-token
// prefill (Q4_K_M) the down projection took 868 ms on the one-block shape and 839 on this,
// bit-identical. MLX's own shape at a 2048-token chunk -- 64 work rows x 64 rows x K 64 on 4
// simdgroups of 4x4 fragments (gather_qmm_rhs) -- took 947 here, the 64-token one 937.
IMPARO_MOE_ST_KERNEL(imparo_moe_st_gemm_r2,    64u, 32u, 64u, 4u, 4u, 2u, false, 2u)
// GATE, UP AND THE ACTIVATION IN ONE STAGED DISPATCH. The weight tile is restaged for the up
// pass rather than held twice, so the paired shape costs the single one's threadgroup memory.
//
// THE SHAPE WAS SWEPT AT THE 2048-TOKEN CHUNK (bracket's 5955-token prompt, prefill wall,
// 2026-09-24) and this one wins: a 64-token pair on 8 simdgroups +2.6%; 4x4 fragments a
// simdgroup 3-4x slower (32 accumulators a thread); 16 work rows x 32 rows x K 32 on 2
// simdgroups (MLX's gather_qmm_rhs shape for few rows an expert, not the one it runs at this
// chunk) +8.2%; 32 rows x 32 tokens x K 32 +16%; 64 x 16 x 32 +8.2%; K 32 at this tile
// +3.9%. Most of the kernel's time is the multiply loop, not the weight decode. Against MLX
// at a 2048-token chunk the pair is 25.7 ms, MLX's gate + up + activation 26.1.
IMPARO_MOE_ST_KERNEL(imparo_moe_st_gemm_pair, 64u, 32u, 64u, 4u, 4u, 2u, true, 1u)
// NARROW CHUNKS (under 32 tokens, the verify batch and a resumed tail): 8 tokens a tile, so a
// chunk of a few work rows an expert is not paid for as 32. Same per-element arithmetic as
// the wide shape -- half operands, K in the same 8-wide steps in the same order -- so a
// position's result does not depend on which shape its chunk took.
IMPARO_MOE_ST_KERNEL(imparo_moe_st_gemm_n8,       64u, 8u, 64u, 2u, 4u, 1u, false, 1u)
IMPARO_MOE_ST_KERNEL(imparo_moe_st_gemm_pair_n8,  64u, 8u, 64u, 2u, 4u, 1u, true, 1u)

// THE ROUTED FEED-FORWARD ON THE MATRIX UNIT. One threadgroup owns
// TM_UNIT_ROWS * BLK_MMA_TILES output rows of ONE expert and walks that expert's segment
// eight work rows at a time -- the same eight imparo_moe_grouped holds, but as an 8x8
// fragment, so one activation fetch feeds every output row of the tile.
//
// WHY, IN ONE NUMBER. imparo_moe_grouped gives one output row to one simdgroup, so every
// activation value it loads feeds exactly one dot product and is read again for each of the
// n_out output rows:
//
//     each activation value is read     activation bytes a 512-token chunk   B per MAC
//     imparo_moe_grouped   n_out = 1792            1.94 TB                     3.9
//     a 64-wide tile       n_out/64 = 28             31 GB                     0.06
//
// against 496 GMAC of routed work and a 16 MB working set. Measured on LFM2.5-8B-A1B: the
// scalar kernel prefills at 109 tok/s where llama.cpp does 1390 on the same file, and
// IMPARO_MOE_SKIP=2 puts 88% of the chunk in those reads.
//
// The eight work rows are NOT eight consecutive activation rows -- they are eight of this
// expert's picks -- which the body handles by reading perm[row]. Everything else, the quad
// scales, the pair loads, the fragment order, the simdgroup reduction, is the rows matmul's.
kernel void imparo_moe_grouped_mma(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    device const uint  * perm    [[buffer(3)]],
    device const uint  * seg     [[buffer(4)]],
    constant ulong & w_offset      [[buffer(5)]],
    constant ulong & expert_stride [[buffer(6)]],
    constant uint  & n_in          [[buffer(7)]],
    constant uint  & n_out         [[buffer(8)]],
    constant uint  & src_work_rows [[buffer(9)]],
    constant uint  & epilogue      [[buffer(13)]],
    constant uint  & n_expert      [[buffer(14)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    // THE GRID SPANS ACTIVE EXPERTS. It used to span all n_expert and let the empty ones
    // return on seg, which at decode is 28 of every 32 expert groups -- about 207k empty
    // threadgroups a token across 66 routed dispatches.
    if (tgid.y >= seg[n_expert + 1u]) { return; }
    const uint e = seg[n_expert + 2u + tgid.y];
    const uint lo = seg[e];
    const uint hi = seg[e + 1u];
    if (lo >= hi) { return; }   // kept: a compacted entry is non-empty, this costs nothing
    const ulong wbase = w_offset + (ulong)e * expert_stride;
    // EIGHT, because eight is the fragment's token side -- not TM_UNIT_ROWS, which is the
    // weight side and happens to equal it. Two different eights.
    for (uint c = lo; c < hi; c += 8u) {
        const uint live = min(8u, hi - c);
        blk_rows_mma_body<false, true, false>(weights, x, y + (ulong)c * n_out,
                                       (device const half *)nullptr, perm, src_work_rows,
                                       n_in, n_out, live, c, epilogue, wbase, 0ul,
                                       partial, tgid, lane, sgid, nsg);
        // `partial` is written by every simdgroup and read by every simdgroup inside the
        // body, so the next group's writes would race this group's reads without this.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// GATE, UP AND THE ACTIVATION AS ONE ROUTED DISPATCH.
//
// gate and up read the SAME normalised hidden state and neither is an input to the other --
// `h = act(gate(x)) * up(x)` -- so running them in one kernel is a GRID MERGE and not a fold:
// it introduces no dependency the three dispatches did not have. That is the difference from
// folding act_mul into the up projection alone, which WOULD make up wait for gate.
//
//   was    gate -> G        up -> U       act_mul(G, U) -> G      3 dispatches
//   now    gate, up, act(gate)*up -> G                            1
//
// ONE K LOOP, TWO ACCUMULATORS -- not two passes. Two passes over the body with a barrier
// between them is also bit-identical and measured SLOWER than the three dispatches (+0.76%
// on a 128-step decode): two independent dispatches overlap under concurrent dispatch, and
// two passes of one threadgroup do not. Here every activation fragment feeds both stacks
// before it is dropped, so the pair removes two dispatches AND halves the activation loads.
//
// Bit-identical to the three dispatches: each stack accumulates in the order it did as its
// own dispatch, and `imparo_act_f(gate) * up` is imparo_act_mul's expression over the two
// values those dispatches wrote.
kernel void imparo_moe_grouped_mma_pair(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    device const uint  * perm    [[buffer(3)]],
    device const uint  * seg     [[buffer(4)]],
    constant ulong & w_offset      [[buffer(5)]],
    constant ulong & expert_stride [[buffer(6)]],
    constant uint  & n_in          [[buffer(7)]],
    constant uint  & n_out         [[buffer(8)]],
    constant uint  & src_work_rows [[buffer(9)]],
    constant uint  & epilogue      [[buffer(13)]],
    constant uint  & n_expert      [[buffer(14)]],
    constant ulong & up_offset     [[buffer(15)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    if (tgid.y >= seg[n_expert + 1u]) { return; }
    const uint e = seg[n_expert + 2u + tgid.y];
    const uint lo = seg[e];
    const uint hi = seg[e + 1u];
    if (lo >= hi) { return; }
    const ulong gbase = w_offset  + (ulong)e * expert_stride;
    const ulong ubase = up_offset + (ulong)e * expert_stride;
    for (uint c = lo; c < hi; c += 8u) {
        const uint live = min(8u, hi - c);
        blk_rows_mma_body<false, true, true>(weights, x, y + (ulong)c * n_out,
                                       (device const half *)nullptr, perm, src_work_rows,
                                       n_in, n_out, live, c, epilogue, gbase, ubase,
                                       partial, tgid, lane, sgid, nsg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The rows matmul AS SERVED: float activations. Every fast-route pick builds this pipeline.
kernel void imparo_blk_rows_mma(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    blk_rows_mma_body<false, false, false>(weights, x, y, (device const half *)nullptr,
                                    (device const uint *)nullptr, 1u, n_in, n_out, n_tok,
                                    src_row, epilogue, w_offset, 0ul,
                                    partial, tgid, lane, sgid, nsg);
}

// TWO TOKEN FRAGMENTS (9..16 rows): each weight fragment decoded once for both, so the weights are
// read once for up to 16 rows. Float activations; each row's sums are the 8-row kernel's.
kernel void imparo_blk_rows_mma_t2(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    blk_rows_mma_body<false, false, false, 2u>(weights, x, y, (device const half *)nullptr,
                                    (device const uint *)nullptr, 1u, n_in, n_out, n_tok,
                                    src_row, epilogue, w_offset, 0ul,
                                    partial, tgid, lane, sgid, nsg);
}

// The same body over the half mirror at buffer 14, bound only when it covers this step's rows.
kernel void imparo_blk_rows_mma_xh(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in [[buffer(4)]],
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],
    constant uint & src_row  [[buffer(7)]],
    constant uint & epilogue [[buffer(13)]],
    device const half  * xh      [[buffer(14)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    blk_rows_mma_body<true, false, false>(weights, x, y, xh, (device const uint *)nullptr, 1u,
                                   n_in, n_out, n_tok,
                                   src_row, epilogue, w_offset, 0ul,
                                   partial, tgid, lane, sgid, nsg);
}

// Row gather for a ROW-MAJOR block format: one row of the embedding table per token.
// `token_embd` is read BY ROW and never goes through the GEMM, so it keeps the row-major
// layout (ROW_MAJOR_ROLES) and needs this rather than a tile-major reader. Same brick.
kernel void imparo_blk_gather_rows(
    device const uchar * weights [[buffer(0)]], device float * y [[buffer(1)]],
    device const uint * indices [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],
    constant uint & width [[buffer(4)]], constant uint & table_rows [[buffer(5)]],
    constant float & scale [[buffer(6)]], constant uint & dst_off [[buffer(7)]],
    constant uint & n_rows [[buffer(8)]], uint2 gid [[thread_position_in_grid]])
{
    // One thread per (32-value sub-block, token): the brick's unit.
    const uint sb = gid.x, t = gid.y;
    const uint subs = width / 32u;
    if (sb >= subs || t >= n_rows) { return; }
    const uint index = indices[t];
    if (index >= table_rows) { return; }

    const uint sub_per_block = tm_block_elems() / 32u;
    const uint bb = tm_block_bytes();
    const uint blocks = width / tm_block_elems();
    const uint blk = sb / sub_per_block, sub = sb % sub_per_block;
    device const uchar * b = weights + w_offset + ((ulong)index * blocks + blk) * bb;
    const uint so = tm_scale_src_off();
    device const uchar * sc = b + so;
    device const uchar * pay = b + (so == 0u ? tm_scale_bytes() : 0u);

    float v[32];
    tm_sub32(sc, pay, sub, v, false);
    device float * out = y + (ulong)dst_off + (ulong)t * width + sb * 32u;
    for (uint l = 0; l < 32u; ++l) { out[l] = v[l] * scale; }
}

kernel void imparo_ple_combine(
    device float * proj [[buffer(0)]], device const float * emb [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant float & emb_scale [[buffer(3)]],
    constant float & comb_scale [[buffer(4)]], constant uint & n_tok [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.x, t = gid.y;
    if (i >= width || t >= n_tok) { return; }
    const ulong o = (ulong)t * width + i;
    proj[o] = (proj[o] + emb[(ulong)t * width + i] * emb_scale) * comb_scale;
}


// ---- Load-time repack: any row-major block format -> its tile-major twin ---------------
// The kernel is FORMAT-AGNOSTIC: it moves the byte spans the host gives it. The spans come
// from imparo_gguf::weights::TmRule through Backend::transform_weights, so no layout is
// stated twice -- scale spans first, then payload spans in source order, and the unit
// addresses are the same formulas TmRule::payload_offset / scale_offset state.
//
// One thread per (K block, row). Run once at startup per fast-tier tensor; the destination
// is a private buffer, the source the mapped file.
struct RepackSpans {
    uint block_elems;
    uint block_bytes;
    uint unit_rows;
    uint n_spans;
    uint n_scale_spans;
    uint span_off[5];
    uint span_len[5];
};

kernel void imparo_repack_tm(
    device const uchar * src   [[buffer(0)]],
    device uchar * dst         [[buffer(1)]],
    constant uint & n_in       [[buffer(2)]],
    constant uint & n_out      [[buffer(3)]],
    constant RepackSpans & R   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint blocks = n_in / R.block_elems;
    const uint b = gid.x, r = gid.y;
    if (b >= blocks || r >= n_out) { return; }

    uint scale_bytes = 0;
    for (uint i = 0; i < R.n_scale_spans; ++i) { scale_bytes += R.span_len[i]; }
    const uint payload_bytes = R.block_bytes - scale_bytes;

    device const uchar * blk = src + ((ulong)r * blocks + b) * R.block_bytes;
    const ulong unit  = ((ulong)(r / R.unit_rows) * blocks + b) * (ulong)(R.unit_rows * R.block_bytes);
    const uint  slot  = r % R.unit_rows;
    device uchar * sd = dst + unit + (ulong)slot * scale_bytes;
    device uchar * pd = dst + unit + (ulong)R.unit_rows * scale_bytes + (ulong)slot * payload_bytes;

    for (uint i = 0; i < R.n_spans; ++i) {
        device uchar * out = (i < R.n_scale_spans) ? sd : pd;
        for (uint k = 0; k < R.span_len[i]; ++k) { out[k] = blk[R.span_off[i] + k]; }
        out += R.span_len[i];
        if (i < R.n_scale_spans) { sd = out; } else { pd = out; }
    }
}

// ---- Load-time repack: row-major Q8_0 -> tile-major Q8_0_TM ----------------------------
// One thread per (K block, row): copy the block's 32 payload bytes to the unit address and
// its 2 scale bytes to the scale array (q8_tm_payload / q8_tm_scale are the same formulas
// imparo_gguf::weights::TM_RULES states; the host verifies the result against that rule
// under IMPARO_LOAD_REPACK_VERIFY=1). Run once at startup per fast-tier tensor; the
// destination is a private buffer, the source the mapped file.
kernel void imparo_repack_q8_0_tm(
    device const uchar * src [[buffer(0)]],
    device uchar * dst       [[buffer(1)]],
    constant uint & n_in     [[buffer(2)]],
    constant uint & n_out    [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint blocks = n_in / QK8_0;
    const uint b = gid.x, r = gid.y;
    if (b >= blocks || r >= n_out) { return; }
    device const uchar * blk = src + ((ulong)r * blocks + b) * Q8_0_BYTES;
    device uchar * p = dst + q8_tm_payload(r, b, blocks);
    for (uint i = 0; i < QK8_0; ++i) { p[i] = blk[2u + i]; }
    device uchar * sc = dst + q8_tm_scale(r, b, blocks, n_out);
    sc[0] = blk[0]; sc[1] = blk[1];
}




// ===================================================================================
// MEGA FFN BLOCK (task #141, stage 1): gate|up GEMVs -> act*mul -> down GEMV as ONE
// persistent dispatch: a fixed grid of threadgroups that are all resident at once loops over the
// rows; the three phases are separated by software grid barriers on device counters.
//
// Why: per decode token ~2.2 ms of sub-microsecond work sits in 5-7 us dispatches and each
// GEMV stage starts its weight stream only after the previous barrier (evidence 2026-09-05
// section 22). A fuse cannot cross a full-vector boundary on Metal; this crosses it.
//
// Arithmetic is the decode GEMV fast path's, in its order (LANES_PER_ROW lanes per row,
// one row per lane group, the same lane-group reduce), and the activation is
// imparo_act_mul's expression -- so the block is bit-identical to the three dispatches.
//
// Synchronisation: Metal has no grid barrier and Apple GPUs do not guarantee linear
// progress (Sorensen et al. 2021), so the ONLY safe wait is on threadgroups that are all
// resident: the host sizes the grid to a measured capacity. Counters: [0] barrier 1,
// [1] barrier 2, [2] exit, [3] error. The last threadgroup out resets [0..2], so the host
// keeps no epoch and nothing wraps; the next dispatch is ordered after this one by the
// hazard tracker (its inputs depend on this output). Every spin is bounded: past
// MEGA_SPIN_MAX it sets [3] and continues; the host reads [3] after the region and falls
// back to the dispatch path. Needs MSL 3.2 for the device-scope fence; below that the
// kernel is absent and the host route stays off.
#if __METAL_VERSION__ >= 320
// A legitimate wait is one phase (< 300 us). The cap must also outlast a threadgroup that the
// GPU deschedules for another client (the window server, a browser's GPU process): 400k spins
// (~20 ms) timed out on such stalls (task #148, 2026-09-06); 4M spins (~200 ms) still fails a
// deliberate over-capacity dispatch in a fraction of a second rather than minutes.
// The stall budget of one barrier wait, in spin iterations. Measured 2026-09-06 on the M3 Pro:
// one iteration is ~1.25 us when 18..31 threadgroups spin on the same word, so 1M iterations
// is ~1.2 s (a threadgroup descheduled by another GPU client comes back well inside that).
// The budget is paid at most ONCE per region: a timeout sets the error word, every other
// spinner sees it within 256 iterations and gives up too, the dispatch runs to its exit
// without further barriers, and every later dispatch leaves at entry (the word is sticky).
constant uint MEGA_SPIN_MAX = 1000000u;
// IMPARO_MEGA_DEBUG=1 (function constant 18): the attention phase and the combine record
// what they wrote and what they read, per (dispatch, head, part), into the sync buffer
// past the scratch -- 8 words per record: [PA M, PA L, PA input non-finite bits (1 = q),
// PA span|split<<16|heads<<24, combine-read M, combine-read L, combine L, combine flags
// (1 non-finite output, 2 L<=0, 4 o_proj saw a non-finite attn value)]. The host scans
// after the region. Off, none of this exists in the pipeline.
constant bool MEGA_DBG_FC [[function_constant(18)]];
constant bool MEGA_DBG = is_function_constant_defined(MEGA_DBG_FC) ? MEGA_DBG_FC : false;
// MEGA_DEEP: the layer kernels' deep attention body (task #151) is compiled into a SEPARATE
// pipeline variant, dispatched at one threadgroup per core; the plain variants keep the
// footprint the two-per-core grid was measured on (design doc, constraint 8).
constant bool MEGA_DEEP_FC [[function_constant(19)]];
constant bool MEGA_DEEP = is_function_constant_defined(MEGA_DEEP_FC) ? MEGA_DEEP_FC : false;
// MEGA_PROGRAM (task #153): the entry comes from a program of entries in a buffer, indexed at
// runtime (tok.entry_index) -- the read the footprint cliff (constraint 8) charges for, so the
// variant is dispatched at one threadgroup per core. Off, the entry is prog[0] at a compile-time
// index: the argument path the plain pipelines were measured on.
constant bool MEGA_PROGRAM_FC [[function_constant(20)]];
constant bool MEGA_PROGRAM = is_function_constant_defined(MEGA_PROGRAM_FC) ? MEGA_PROGRAM_FC : false;
// The vec phase's items stride the grid on every pipeline that runs at one threadgroup per core
// by construction (typed, program); the f16 plain pipelines keep one item per threadgroup.
constant bool MEGA_STRIDE = MEGA_KVQ || MEGA_PROGRAM;
constant uint MEGA_DBG_REC = 8u;
// One Q capture per region slot (MEGA_DBG): [0] claimed, [1] seq, [2] head, [3] part, [4] span,
// [5] tgid, [8..8+HD) the q values the flagged threadgroup loaded, as bits.
constant uint MEGA_DBG_REC_WORDS = 4u * 64u * 32u * 8u;   // 4 region slots x 64 dispatches x 32 items
// Sync buffer layout (words): [0, 16) barrier counters + error; [16, MEGA_SYNC_HDR) unused, so
// the counters' cache lines carry nothing else; [MEGA_SYNC_HDR, +scratch) the attention
// partials; then the debug records. The host mirrors MEGA_SYNC_HDR.
constant uint MEGA_SYNC_HDR = 1024u;
constant uint MEGA_DBG_CAP_WORDS = 8u + 512u;
inline bool mega_nonfinite(float x) { return (as_type<uint>(x) & 0x7f800000u) == 0x7f800000u; }

// THE CACHE BASIS (task #156). A quantized cache is stored in a rotated basis: the engine's
// quantized modes rotate Q, K and V by an orthonormal blockwise Hadamard before the store
// (llama.cpp's attn_rot defense: scores are invariant, (Hq).(Hk) = q.k, and the rotation
// spreads a 32-value block's energy so the shared scale stops starving small values) and
// rotate the attention output back (H is symmetric and orthonormal, so the same transform
// inverts). The dispatch path does this with imparo_hadamard64 around its kernels; inside the
// mega-kernel the head phase rotates the rows it forms and the combine rotates the output it
// writes, so the cache bytes and the o_proj input are the dispatch path's. The transform is
// imparo_hadamard64's: Sylvester butterflies over consecutive `nrot` values, then 1/sqrt(nrot).
inline float mega_had_scale(uint nrot) {
    // 1/sqrt(nrot) as the host computes it (1.0f / sqrtf): the same float for every legal width.
    switch (nrot) {
        case 32u:  return 0.17677669529663687f;
        case 64u:  return 0.125f;
        case 128u: return 0.08838834764831845f;
        case 256u: return 0.0625f;
        case 512u: return 0.044194173824159216f;
        default:   return 1.0f / precise::sqrt((float)nrot);
    }
}
// One head row (HD floats in device memory) rotated in place, one threadgroup: HD/nrot blocks,
// the butterflies on the row itself with a device barrier per stage (log2(nrot) stages).
template <uint HD>
inline void mega_hadamard_row(device float * row, uint nrot, uint tid, uint tcount) {
    const uint half_blk = nrot / 2u;
    for (uint stride = 1u; stride < nrot; stride <<= 1u) {
        for (uint q = tid; q < HD / 2u; q += tcount) {
            const uint blk = q / half_blk, p = q - blk * half_blk;
            const uint i = blk * nrot + (((p & ~(stride - 1u)) << 1u) | (p & (stride - 1u)));
            const float a = row[i], b = row[i | stride];
            row[i] = a + b;
            row[i | stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    const float scale = mega_had_scale(nrot);
    for (uint i = tid; i < HD; i += tcount) { row[i] = row[i] * scale; }
    threadgroup_barrier(mem_flags::mem_device);
}
// The same transform on a head row held by one simdgroup, lane `lane` owning dims
// [lane*DPL, lane*DPL + DPL): strides inside a lane are register pairs, strides across lanes
// are simd_shuffle_xor partners (nrot >= DPL, nrot a power of two: a block is a lane group).
template <uint HD>
inline void mega_hadamard_lanes(thread float * v, uint nrot, uint lane) {
    constexpr uint DPL = HD / 32u;
    for (uint stride = 1u; stride < DPL && stride < nrot; stride <<= 1u) {
        for (uint i = 0; i < DPL; ++i) {
            if ((i & stride) == 0u) {
                const float a = v[i], b = v[i | stride];
                v[i] = a + b;
                v[i | stride] = a - b;
            }
        }
    }
    for (uint lstride = 1u; lstride * DPL < nrot; lstride <<= 1u) {
        const bool upper = (lane & lstride) != 0u;
        for (uint i = 0; i < DPL; ++i) {
            const float mine = v[i];
            const float other = simd_shuffle_xor(mine, (ushort)lstride);
            v[i] = upper ? (other - mine) : (mine + other);
        }
    }
    const float scale = mega_had_scale(nrot);
    for (uint i = 0; i < DPL; ++i) { v[i] = v[i] * scale; }
}


inline void mega_barrier(device atomic_uint * ctr, uint target, device atomic_uint * err,
                         uint tid, uint tag = 0u) {   // tag: (barrier index << 4) | (tgid << 24), stored with the error
    // Release: the threadgroup barrier (mem_device) completes every thread's device writes,
    // then thread 0 fences at device scope and arrives; acquire: thread 0 fences after the
    // wait, then the threadgroup barrier releases the others. Per-thread fences (every thread
    // fencing before the arrival and after the wait) were tried on 2026-09-06 against a
    // once-in-40 wrong result: they did not remove it (the cause was the attention partials
    // aliasing Q, task #148) and they cost +0.7% decode (21.69/21.63 vs 21.53/21.46 ms per
    // token, two interleaved rounds). If a cross-threadgroup visibility failure is ever
    // observed with this form, per-thread fences are the first thing to try.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        // A failure costs ONE spin cap: once the error word is set (by this barrier's timeout
        // or an earlier one's), no barrier of the dispatch spins again -- the threadgroups
        // that never arrived will not arrive later either.
        uint spins = 0u;
        if (atomic_load_explicit(err, memory_order_relaxed) == 0u) {
            while (atomic_load_explicit(ctr, memory_order_relaxed) < target) {
                if (++spins > MEGA_SPIN_MAX) {
                    // Error word: bit 0, barrier index (bits 4..15), arrivals seen (bits 16..23), tgid (24..31).
                    const uint seen = min(atomic_load_explicit(ctr, memory_order_relaxed), 255u);
                    atomic_store_explicit(err, 1u | tag | (seen << 16), memory_order_relaxed);
                    break;
                }
                if ((spins & 255u) == 0u && atomic_load_explicit(err, memory_order_relaxed) != 0u) { break; }
            }
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// THE BLOCK'S BARRIER: one monotonic counter per dispatch -- the k-th barrier waits for
// k * n_tg arrivals, the last threadgroup out resets the counter at exit. A timeout writes the
// error word (bit 0, phase in bits 4..15, arrivals in the phase 16..23, threadgroup 24..31);
// the word is sticky and every later dispatch leaves at entry, so a failed region costs one
// spin cap (design doc, constraint 4).
inline void mega_step(device atomic_uint * ctr, thread uint & phase, uint n_tg, device atomic_uint * err,
                      uint tid, uint tgid, uint spin_max = MEGA_SPIN_MAX) {
    phase += 1u;
    const uint target = phase * n_tg;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        // A failure costs ONE spin cap: once the error word is set (by this barrier's timeout
        // or an earlier one's), no barrier of the dispatch spins again -- the threadgroups
        // that never arrived will not arrive later either -- and a spinner that sees another
        // threadgroup's error gives up within 256 iterations. Measured 2026-09-07: without
        // this a timed-out layer dispatch paid the cap at every one of its later barriers.
        uint spins = 0u;
        if (atomic_load_explicit(err, memory_order_relaxed) == 0u) {
            while (atomic_load_explicit(ctr, memory_order_relaxed) < target) {
                if (++spins > spin_max) {
                    const uint seen = atomic_load_explicit(ctr, memory_order_relaxed);
                    const uint arrived = (seen + n_tg >= target) ? min(seen + n_tg - target, 255u) : 0u;
                    atomic_store_explicit(err, 1u | ((phase & 0xfffu) << 4) | (arrived << 16) | (tgid << 24), memory_order_relaxed);
                    break;
                }
                if ((spins & 255u) == 0u && atomic_load_explicit(err, memory_order_relaxed) != 0u) { break; }
            }
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// One Q4_0 row, this lane's share (blocks sub, sub+LANES, ...): the decode fast path's
// per-block arithmetic in its order. The caller reduces across the lane group.
inline float mega_q4_row_partial(device const uchar * row, device const float4 * xs,
                                 uint blocks, uint sub) {
    // The q4_rows_partial brick at one row (the mega pipelines are built with NR0 == 1; the
    // arrays carry the brick's bound so a wider NR0 stays in bounds).
    ulong off[8] = { 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul };
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    q4_rows_partial(row, off, xs, blocks, sub, acc);
    return acc[0];
}

// DECODE ATTENTION AS A MEGA PHASE: imparo_attention_decode_vec_t's body for one query head,
// run by one threadgroup of the persistent grid (nsg simdgroups stride the span, lanes own
// HD/32 dims, online softmax per simdgroup). The merge differs from the vec kernel's
// two-round form: the scratch is capped at MEGA_ATTN_SLOTS states so it fits the block's
// n_embd-float threadgroup row at any head dim, and the simdgroups fold in rounds of
// MEGA_ATTN_SLOTS (nsg must be a multiple of it and at least twice it -- the host checks).
constant uint MEGA_ATTN_SLOTS = 4u;
template <uint HD, uint KVW>
inline void mega_attn_vec(device const float * q, device const half * kc, device const half * vc,
                          device float * partials, uint n_heads, uint n_kv, uint kv_width,
                          uint start_pos, uint window, uint ring_mask, device const uint * pt,
                          threadgroup float * red, uint h, uint part, uint split,
                          uint lane, uint sgid, uint nsg, device atomic_uint * dbg,
                          device atomic_uint * cap, uint seq, uint tgid_dbg) {
    constexpr uint DPL = HD / 32u;
    uint dbg_bad = 0u;
    const uint kvw = (KVW != 0u) ? KVW : kv_width;
    const uint kvh = h / (n_heads / n_kv);
    const uint pos = start_pos;
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;
    float qr[DPL];
    {
        device const float * qp = q + (ulong)h * HD + lane * DPL;
        for (uint i = 0; i < DPL; ++i) { qr[i] = qp[i]; }
        if (MEGA_DBG) { for (uint i = 0; i < DPL; ++i) { dbg_bad |= mega_nonfinite(qr[i]) ? 1u : 0u; } }
    }
    float m = -INFINITY, l = 0.0f;
    float o[DPL];
    for (uint i = 0; i < DPL; ++i) { o[i] = 0.0f; }
    // MEGA_DBG KV capture (task #156 probe): the region slot's first head-0 threadgroup records
    // the K and V values the typed loader returns for the span's first position: [0]=2 (a KV
    // capture), [1] seq, [2] cache slot, [3] kvw, [4] n, [5] tgid, [6] HD, [7] kvh; then K as
    // bits at [8, 8+HD) and V at [8+HD, 8+2HD) (HD <= 256). The host prints them next to its
    // own dequantisation of the same cache bytes.
    if (MEGA_DBG && HD <= 256u && h == 0u && part == 0u && sgid == 0u && n > 0u) {
        uint won = 0u;
        if (lane == 0u) {
            uint expected = 0u;
            won = atomic_compare_exchange_weak_explicit(cap, &expected, 2u, memory_order_relaxed, memory_order_relaxed) ? 1u : 0u;
        }
        won = simd_broadcast_first(won);
        if (won != 0u) {
            const uint ps = kv_slot(lo, ring_mask, pt);
            float kf[DPL], vf[DPL];
            attn_load_kv<HD>(kc, vc, kvw, ps, kvh, lane, kf, vf);
            for (uint i = 0; i < DPL; ++i) {
                atomic_store_explicit(cap + 8 + lane * DPL + i, as_type<uint>(kf[i]), memory_order_relaxed);
                atomic_store_explicit(cap + 8 + HD + lane * DPL + i, as_type<uint>(vf[i]), memory_order_relaxed);
            }
            if (lane == 0u) {
                atomic_store_explicit(cap + 1, seq, memory_order_relaxed);
                atomic_store_explicit(cap + 2, ps, memory_order_relaxed);
                atomic_store_explicit(cap + 3, kvw, memory_order_relaxed);
                atomic_store_explicit(cap + 4, n, memory_order_relaxed);
                atomic_store_explicit(cap + 5, tgid_dbg, memory_order_relaxed);
                atomic_store_explicit(cap + 6, HD, memory_order_relaxed);
                atomic_store_explicit(cap + 7, kvh, memory_order_relaxed);
            }
        }
    }
    // This threadgroup's simdgroups take positions part*nsg + sgid, stepping split*nsg.
    attn_span_online<HD, KVW>(kc, vc, kvw, kvh, lo, n, part * nsg + sgid, split * nsg, ring_mask, pt,
                              lane, qr, m, l, o);
    if (MEGA_DBG) {
        dbg_bad = simd_or(dbg_bad);
        if (lane == 0u && dbg_bad != 0u) { atomic_fetch_or_explicit(dbg + 2, dbg_bad, memory_order_relaxed); }
        if ((dbg_bad & 1u) != 0u && sgid == 0u) {
            // First flagged threadgroup of the region slot keeps its q vector for the host.
            uint won = 0u;
            if (lane == 0u) {
                uint expected = 0u;
                won = atomic_compare_exchange_weak_explicit(cap, &expected, 1u, memory_order_relaxed, memory_order_relaxed) ? 1u : 0u;
            }
            won = simd_broadcast_first(won);
            if (won != 0u) {
                for (uint i = 0; i < DPL; ++i) { atomic_store_explicit(cap + 8 + lane * DPL + i, as_type<uint>(qr[i]), memory_order_relaxed); }
                if (lane == 0u) {
                    atomic_store_explicit(cap + 1, seq, memory_order_relaxed);
                    atomic_store_explicit(cap + 2, h, memory_order_relaxed);
                    atomic_store_explicit(cap + 3, part, memory_order_relaxed);
                    atomic_store_explicit(cap + 4, n, memory_order_relaxed);
                    atomic_store_explicit(cap + 5, tgid_dbg, memory_order_relaxed);
                    atomic_store_explicit(cap + 6, HD, memory_order_relaxed);
                }
            }
        }
    }
    // Fold the simdgroups in rounds of MEGA_ATTN_SLOTS: [base, base+S) hand their state to
    // [base-S, base), until simdgroups 0..S-1 hold everything; then 1..S-1 hand theirs to 0.
    for (uint base = nsg - MEGA_ATTN_SLOTS; base >= MEGA_ATTN_SLOTS; base -= MEGA_ATTN_SLOTS) {
        if (sgid >= base && sgid < base + MEGA_ATTN_SLOTS) {
            attn_store_slot<HD>(red + (sgid - base) * (HD + 2u), lane, m, l, o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid >= base - MEGA_ATTN_SLOTS && sgid < base) {
            attn_merge_slot<HD>(red + (sgid - (base - MEGA_ATTN_SLOTS)) * (HD + 2u), lane, m, l, o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sgid > 0u && sgid < MEGA_ATTN_SLOTS) { attn_store_slot<HD>(red + sgid * (HD + 2u), lane, m, l, o); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float M, L, acc[DPL];
        attn_fold_slots<HD>(red, MEGA_ATTN_SLOTS, lane, m, l, o, M, L, acc);
        // The threadgroup's partial (unnormalised): the combine phase merges the parts.
        device float * pp = partials + ((ulong)h * split + part) * (HD + 2u);
        for (uint i = 0; i < DPL; ++i) { pp[lane * DPL + i] = acc[i]; }
        if (lane == 0u) { pp[HD] = M; pp[HD + 1u] = L; }
        if (MEGA_DBG && lane == 0u) {
            atomic_store_explicit(dbg + 0, as_type<uint>(M), memory_order_relaxed);
            atomic_store_explicit(dbg + 1, as_type<uint>(L), memory_order_relaxed);
            atomic_store_explicit(dbg + 3, n | (split << 16) | (n_heads << 24), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // red (the block's row) is reused next
}

// Merge a head's `split` partials onto a common maximum and normalise: one simdgroup.
template <uint HD>
inline void mega_attn_combine(device const float * partials, device float * out, uint h,
                              uint split, uint had_v, uint lane, device atomic_uint * dbg) {
    constexpr uint DPL = HD / 32u;
    device const float * base = partials + (ulong)h * split * (HD + 2u);
    float M = -INFINITY;
    for (uint p = 0; p < split; ++p) { M = max(M, base[p * (HD + 2u) + HD]); }
    float L = 0.0f;
    float acc[DPL];
    for (uint i = 0; i < DPL; ++i) { acc[i] = 0.0f; }
    for (uint p = 0; p < split; ++p) {
        device const float * pp = base + p * (HD + 2u);
        const float mp = pp[HD];
        const float f = (mp == -INFINITY) ? 0.0f : exp(mp - M);
        L += f * pp[HD + 1u];
        for (uint i = 0; i < DPL; ++i) { acc[i] += f * pp[lane * DPL + i]; }
        if (MEGA_DBG && lane == 0u) {
            atomic_store_explicit(dbg + p * MEGA_DBG_REC + 4, as_type<uint>(mp), memory_order_relaxed);
            atomic_store_explicit(dbg + p * MEGA_DBG_REC + 5, as_type<uint>(pp[HD + 1u]), memory_order_relaxed);
        }
    }
    device float * op = out + (ulong)h * HD + lane * DPL;
    const float inv = 1.0f / L;
    for (uint i = 0; i < DPL; ++i) { acc[i] = acc[i] * inv; }
    // A rotated (quantized) V basis: the output back to model space (task #156).
    if (MEGA_KVQ && had_v != 0u) { mega_hadamard_lanes<HD>(acc, had_v, lane); }
    for (uint i = 0; i < DPL; ++i) { op[i] = acc[i]; }
    if (MEGA_DBG) {
        uint bad = 0u;
        for (uint i = 0; i < DPL; ++i) { bad |= mega_nonfinite(acc[i]) ? 1u : 0u; }
        bad = simd_or(bad);
        if (lane == 0u) {
            if (!(L > 0.0f)) { bad |= 2u; }
            for (uint p = 0; p < split; ++p) {
                atomic_store_explicit(dbg + p * MEGA_DBG_REC + 6, as_type<uint>(L), memory_order_relaxed);
                atomic_fetch_or_explicit(dbg + p * MEGA_DBG_REC + 7, bad, memory_order_relaxed);
            }
        }
    }
}

// THE DEEP BODY AS A MEGA PHASE (task #151): work item = (KV head, head sub-group, key slice),
// `slices` per (KV head, sub-group) -- the host's attn_split -- one item per threadgroup. The
// slice is split into contiguous runs, one per simdgroup, walked by the grouped brick (K/V
// once per item); then, head by head, the simdgroups fold through `red` in the vec body's
// rounds of MEGA_ATTN_SLOTS, and the (head, slice) partial goes to the scratch in the vec
// body's layout, so mega_attn_combine merges either body's partials.
template <uint HD, uint KVW>
inline void mega_attn_group(device const float * q, device const half * kc, device const half * vc,
                            device float * partials, uint n_heads, uint n_kv, uint kv_width,
                            uint start_pos, uint window, uint ring_mask, device const uint * pt,
                            threadgroup float * red, uint item, uint slices,
                            uint lane, uint sgid, uint nsg) {
    constexpr uint HQ = attn_group_hq(HD);
    constexpr uint DPL = HD / 32u;
    const uint kvw = (KVW != 0u) ? KVW : kv_width;
    const uint share = n_heads / n_kv;
    const uint n_sub = (share + HQ - 1u) / HQ;
    const uint kvh = item / (n_sub * slices);
    const uint sub = (item / slices) % n_sub;
    const uint sl  = item % slices;
    const uint h0  = kvh * share + sub * HQ;
    const uint hq  = min(HQ, share - sub * HQ);
    const uint pos = start_pos;
    const uint lo = (window > 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
    const uint n = pos + 1u - lo;
    // Slice sl covers [sb, se) of the span; simdgroup sgid a contiguous run of it.
    const uint per = (n + slices - 1u) / slices;
    const uint sb = min(n, sl * per), se = min(n, sb + per);
    const uint run = (se - sb + nsg - 1u) / nsg;
    const uint s0 = min(se, sb + sgid * run), s1 = min(se, s0 + run);
    float qr[HQ * DPL], m[HQ], l[HQ], o[HQ * DPL];
#pragma unroll
    for (uint j = 0; j < HQ; ++j) {
        m[j] = -INFINITY; l[j] = 0.0f;
        for (uint i = 0; i < DPL; ++i) { o[j * DPL + i] = 0.0f; qr[j * DPL + i] = 0.0f; }
        if (j < hq) {
            device const float * qp = q + (ulong)(h0 + j) * HD + lane * DPL;
            for (uint i = 0; i < DPL; ++i) { qr[j * DPL + i] = qp[i]; }
        }
    }
    attn_group_run_online<HD, KVW>(kc, vc, kvw, kvh, lo, s0, s1, ring_mask, pt, lane, hq, qr, m, l, o);
    // Fold head by head (the slots hold one head's states; `red` is the block's row).
#pragma unroll
    for (uint j = 0; j < HQ; ++j) {
        if (j < hq) {
            for (uint base = nsg - MEGA_ATTN_SLOTS; base >= MEGA_ATTN_SLOTS; base -= MEGA_ATTN_SLOTS) {
                if (sgid >= base && sgid < base + MEGA_ATTN_SLOTS) {
                    attn_store_slot<HD>(red + (sgid - base) * (HD + 2u), lane, m[j], l[j], o + j * DPL);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sgid >= base - MEGA_ATTN_SLOTS && sgid < base) {
                    attn_merge_slot<HD>(red + (sgid - (base - MEGA_ATTN_SLOTS)) * (HD + 2u), lane, m[j], l[j], o + j * DPL);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (sgid > 0u && sgid < MEGA_ATTN_SLOTS) { attn_store_slot<HD>(red + sgid * (HD + 2u), lane, m[j], l[j], o + j * DPL); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sgid == 0u) {
                float M, L, acc[DPL];
                attn_fold_slots<HD>(red, MEGA_ATTN_SLOTS, lane, m[j], l[j], o + j * DPL, M, L, acc);
                device float * pp = partials + ((ulong)(h0 + j) * slices + sl) * (HD + 2u);
                for (uint i = 0; i < DPL; ++i) { pp[lane * DPL + i] = acc[i]; }
                if (lane == 0u) { pp[HD] = M; pp[HD + 1u] = L; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);   // red is reused by the next head
        }
    }
}

// The same row partial with the input row in threadgroup memory (a row every threadgroup
// formed for itself). Same arithmetic in the same order; only the load address space differs.
inline float mega_q4_row_partial_tg(device const uchar * row, threadgroup const float4 * xs,
                                    uint blocks, uint sub) {
    ulong off[8] = { 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul };
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    q4_rows_partial(row, off, xs, blocks, sub, acc);
    return acc[0];
}

#endif


// ===================================================================================
// MEGA FFN+PLE BLOCK (task #141, stage 2a): from the FFN input to the end of the layer as ONE
// persistent dispatch -- gate|up, act*mul, down, post-FFN norm+add, PLE gate, act*row, PLE
// proj, and the tail (post_norm + residual*scale, then the next layer's input norm). Ten
// dispatches today, seven software barriers here. Barrier/counter contract: words 4..15 of
// the sync buffer (barriers 4..12, exit 13, error 15), see mega_step / mega_exit.
//
// The two norm phases run on threadgroup 0 and reproduce imparo_rms_norm_add_row exactly:
// that kernel is launched with `norm_t` threads for the width (the host's rule), each thread
// summing float4 indices t, t+norm_t, ... then simd_sum per simdgroup then simd_sum over the
// simdgroup partials. Here virtual simdgroup vg (0 <= vg < norm_t/32) is served by a real
// simdgroup with the same lane -> index mapping, so every partial sum sees the same values in
// the same order and the block stays bit-identical to the dispatch path. The staged row
// (rounded normalised values before the add) lives in a free device scratch row instead of
// threadgroup memory, so the dispatch's threadgroup-memory footprint stays small and its
// residency is not reduced.
// ---- MIXTURE OF EXPERTS ----------------------------------------------------------------
//
// A routed feed-forward picks `k` of `n_expert` experts PER TOKEN, so the one thing the rest
// of this file never needs appears here: a matmul whose weight matrix is chosen per row.
// llama.cpp calls it MUL_MAT_ID. Four kernels, and the shape of the middle two is the whole
// design:
//
//   imparo_moe_gate      router scores -> probabilities, and probabilities + bias -> the
//                        scores the pick reads. THE BIAS SELECTS AND DOES NOT WEIGH: the
//                        weights come from the UNBIASED probabilities (llama.cpp says so at
//                        the line that does it), and mixing the two silently runs the wrong
//                        experts with plausible text as the only symptom.
//   imparo_moe_plan      a counting sort of the n_tok * k (token, slot) pairs BY EXPERT, in
//                        one threadgroup: positions from threadgroup atomics, a 32-entry
//                        prefix sum, then the scatter. It produces `perm` (work row -> token),
//                        `wgt` (its routing weight), `seg` (each expert's half-open range) and
//                        `inv` ((token, slot) -> work row, which the combine reads back).
//   imparo_moe_grouped   the matmul. A threadgroup owns one expert and one output row, decodes
//                        each weight sub-block ONCE, and multiplies it into up to MOE_ROWS of
//                        that expert's work rows -- the reuse that makes this affordable.
//                        WHERE ITS INPUT ROW LIVES IS AN ARGUMENT: the gate and up
//                        projections read the token's activations, so work row w reads
//                        `perm[w]` and nothing is gathered into a staging buffer first; the
//                        down projection reads what the gate and up just wrote, which is
//                        already one row per WORK ROW, so it reads row w itself. Taking
//                        `perm[w]` there reads another token's hidden state -- a wrong answer
//                        that still looks like an answer.
//   imparo_moe_combine   sums a token's k weighted rows in slot order. NOT float atomics: an
//                        atomic accumulation orders itself differently every run and this
//                        engine's determinism gate would fail on the next step.
//
// The expert stack is a 3-D tensor and the tile-major transform refuses 3-D, so these read
// ROW-MAJOR blocks -- the same brick (`tm_sub32`) the row-major GEMV reads -- with expert e's
// matrix at `w_offset + e * expert_stride`.
constant uint MOE_ROWS = 8u;          // work rows one pass of the grouped matmul holds

// DIAGNOSTIC, 0 in every served pipeline (IMPARO_MOE_SKIP). Prices the grouped matmul's
// parts by difference, the way IMPARO_BLK_ROWS_SKIP prices the decode-rows GEMV's:
//   bit 0  the weight decode and its loads are gone (v[] gets a cheap sequence)
//   bit 1  the activation row is not read at all (the products run on v[] alone)
//   bit 2  the 32 products become one (7 of the 8 float4 loads go with them)
// At 0 the code below is exactly what shipped before this constant existed, so the
// baseline arm does not move: a branch that never runs still costs, which is why this is
constant uint MOE_SKIP_FC [[function_constant(45)]];
constant uint MOE_SKIP = is_function_constant_defined(MOE_SKIP_FC) ? MOE_SKIP_FC : 0u;

// THE ACTIVATION ROW, ONCE PER THREADGROUP INSTEAD OF ONCE PER OUTPUT ROW (1 on, 0 off,
// IMPARO_MOE_STAGE). Every simdgroup in a threadgroup reads the SAME activation: lane L
// of each one wants sub-block L of the same work row, because the activation depends on
// (work row, sub-block) and not on the output row a simdgroup owns. Unstaged, that is
// eight identical load streams, and IMPARO_MOE_SKIP=2 prices them at 4121 ms of a 4700 ms
// 512-token prefill chunk -- 88%.
//
// So one threadgroup loads each work row's K-slice once into threadgroup memory and all
// MOE_GROUPED_SGS simdgroups read it there. STAGE_ROWS work rows at a time, 32 lanes x 32
// floats each = 16 KiB, which leaves room for a second threadgroup on a core; the weight
// decode stays OUTSIDE the staging rounds, so it still runs once per (output row, K-step)
// and the MOE_ROWS weight reuse is untouched.
//
// Lane L still sums its own sub-blocks in the same order, so a row's value is bit-identical
// to the unstaged kernel's. That is what makes this an A/B and not a new answer.
// MEASURED NEGATIVE, so it is OFF (LFM2.5-8B-A1B, M3 Pro, 2026-09-21): a 512-token prefill
// chunk 4785 -> 4090 ms (-14.5%), and a decode step 14.7 -> 22.9 ms (+56%) for the barriers
// at live == 1. THE REASON, and it is the useful part: IMPARO_MOE_SKIP=4 keeps one float4
// load of eight and costs 745 ms against the full kernel's 4700, so the time follows the
// NUMBER OF LOADS, not the bytes. Staging moves the loads nearer; it does not remove any.
// A simdgroup still issues 32 reads per (output row, work row, sub-block). What removes
// loads is more arithmetic per loaded value -- the matrix unit, as in imparo_blk_rows_mma.
constant uint MOE_STAGE_FC [[function_constant(46)]];
constant uint MOE_STAGE = is_function_constant_defined(MOE_STAGE_FC) ? MOE_STAGE_FC : 0u;
constant uint MOE_STAGE_ROWS = 4u;
#define MOE_STAGE_FLOATS (MOE_STAGE_ROWS * 1024u)
#define MOE_MAX_EXPERTS 256u
#define MOE_MAX_K 8u

kernel void imparo_moe_gate(
    device const uchar * weights [[buffer(0)]],
    device const float * scores  [[buffer(1)]],
    device float       * probs   [[buffer(2)]],
    device float       * sel     [[buffer(3)]],
    constant uint & n_expert [[buffer(4)]],
    constant uint & gating   [[buffer(5)]],   // 0 softmax, 1 sigmoid
    constant ulong & bias_off [[buffer(6)]],  // IMPARO_NO_WEIGHT when the file has none
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    device const float * s = scores + (ulong)tg * n_expert;
    device float * p = probs + (ulong)tg * n_expert;
    device float * q = sel   + (ulong)tg * n_expert;
    if (gating == 0u) {
        // ONE THREAD, so the sum has one order. n_expert is 32 here; a tree reduction would
        // buy nothing and would make the result depend on the threadgroup width.
        if (tid == 0u) {
            float m = -INFINITY;
            for (uint e = 0u; e < n_expert; ++e) { m = max(m, s[e]); }
            float sum = 0.0f;
            for (uint e = 0u; e < n_expert; ++e) { const float v = exp(s[e] - m); p[e] = v; sum += v; }
            for (uint e = 0u; e < n_expert; ++e) { p[e] = p[e] / sum; }
        }
    } else {
        for (uint e = tid; e < n_expert; e += tcount) { p[e] = 1.0f / (1.0f + exp(-s[e])); }
    }
    threadgroup_barrier(mem_flags::mem_device);
    device const float * bias = (device const float *)(weights + bias_off);
    const bool has_bias = bias_off != IMPARO_NO_WEIGHT;
    for (uint e = tid; e < n_expert; e += tcount) {
        q[e] = p[e] + (has_bias ? bias[e] : 0.0f);
    }
}

kernel void imparo_moe_plan(
    device const uint  * topk  [[buffer(0)]],   // [n_tok * k] expert ids, top_k_rows' layout
    device const float * probs [[buffer(1)]],   // [n_tok, n_expert] UNBIASED
    device uint  * perm        [[buffer(2)]],
    device float * wgt         [[buffer(3)]],
    device uint  * seg         [[buffer(4)]],   // [n_expert + 1]
    device uint  * inv         [[buffer(5)]],   // [n_tok * k]
    constant uint & n_tok     [[buffer(6)]],
    constant uint & n_expert  [[buffer(7)]],
    constant uint & k         [[buffer(8)]],
    constant uint & normalise [[buffer(9)]],
    constant float & scale    [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]], uint tcount [[threads_per_threadgroup]])
{
    threadgroup atomic_uint count[MOE_MAX_EXPERTS];
    threadgroup uint base[MOE_MAX_EXPERTS];
    for (uint e = tid; e < n_expert; e += tcount) {
        atomic_store_explicit(&count[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each token's position WITHIN its expert's segment. Parked in `inv` so a thread can serve
    // many tokens without holding a register per token.
    for (uint t = tid; t < n_tok; t += tcount) {
        for (uint j = 0u; j < k; ++j) {
            const uint e = topk[t * k + j];
            inv[t * k + j] = (e < n_expert)
                ? atomic_fetch_add_explicit(&count[e], 1u, memory_order_relaxed)
                : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        // THE ACTIVE EXPERTS, COMPACTED, in the tail of the same buffer:
        //   seg[0 .. n_expert]          segment offsets, as before
        //   seg[n_expert + 1]           how many experts hold a work row
        //   seg[n_expert + 2 + i]       the i-th of them, ascending
        // A token picks k DISTINCT experts, so at most min(n_expert, n_tok * k) can be
        // active -- a bound the host knows without reading this back, which is what lets
        // it size the grid by the active count instead of by n_expert.
        uint run = 0u;
        uint nact = 0u;
        for (uint e = 0u; e < n_expert; ++e) {
            base[e] = run;
            seg[e] = run;
            const uint c = atomic_load_explicit(&count[e], memory_order_relaxed);
            if (c != 0u) { seg[n_expert + 2u + nact] = e; nact += 1u; }
            run += c;
        }
        seg[n_expert] = run;
        seg[n_expert + 1u] = nact;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < n_tok; t += tcount) {
        device const float * p = probs + (ulong)t * n_expert;
        // The picked weights, renormalised over the k picks when the architecture says so.
        // The clamp is the reference's: the smallest normal half, so a row of zeros divides
        // by that rather than by zero.
        float sum = 0.0f;
        for (uint j = 0u; j < k; ++j) {
            const uint e = topk[t * k + j];
            sum += (e < n_expert) ? p[e] : 0.0f;
        }
        const float denom = (normalise != 0u) ? max(sum, 6.103515625e-5f) : 1.0f;
        for (uint j = 0u; j < k; ++j) {
            const uint e = topk[t * k + j];
            if (e >= n_expert) { continue; }
            const uint dst = base[e] + inv[t * k + j];
            perm[dst] = t;
            wgt[dst] = (p[e] / denom) * scale;
            inv[t * k + j] = dst;
        }
    }
}

// THE WHOLE ROUTE IN ONE DISPATCH: gate, the pick, and the plan.
//
//   was   moe_gate -> top_k_rows -> moe_plan                         3 dispatches
//   now   imparo_moe_route                                           1
//
// ONE THREADGROUP, because the plan already needs one -- its counting sort takes positions
// from threadgroup atomics and bases from a single prefix sum over them, so it cannot span
// threadgroups anyway. The gate and the pick are folded INTO that threadgroup rather than
// the other way round, and the two dispatches they cost disappear with them.
//
// This is a fold of ALREADY DEPENDENT work, which is the only kind worth folding: the pick
// reads what the gate wrote and the plan reads what the pick wrote, so no dispatch that
// could have overlapped is being serialised. It is also the one fold in the routed layer
// with NO weight stream to disturb -- the three kernels together read one n_expert-wide row
// per token and nothing else.
//
// Bit-identical to the three by construction: each phase is the body of the kernel it
// replaces, in the same order, over the same buffers. The gate's softmax stays ONE THREAD
// per token so its sum keeps one order, and the pick stays the one-pass top-8 list.
//
// THE GRID IS THE COST. At one token this threadgroup does a few hundred operations; at a
// 512-token prefill chunk the gate alone would be 512 softmaxes on one core where the split
// form spreads them over every core. So the host admits this only below a token threshold.
kernel void imparo_moe_route(
    device const uchar * weights   [[buffer(0)]],
    device const float * scores    [[buffer(1)]],
    device float       * probs     [[buffer(2)]],
    device float       * sel       [[buffer(3)]],
    device uint        * topk_ids  [[buffer(4)]],
    device float       * topk_vals [[buffer(5)]],   // the same buffer as topk_ids
    device uint        * perm      [[buffer(6)]],
    device float       * wgt       [[buffer(7)]],
    device uint        * seg       [[buffer(8)]],
    device uint        * inv       [[buffer(9)]],
    constant uint  & n_tok     [[buffer(10)]],
    constant uint  & n_expert  [[buffer(11)]],
    constant uint  & k         [[buffer(12)]],
    constant uint  & gating    [[buffer(13)]],
    constant ulong & bias_off  [[buffer(14)]],
    constant uint  & normalise [[buffer(15)]],
    constant float & scale     [[buffer(16)]],
    uint tid [[thread_position_in_threadgroup]], uint tcount [[threads_per_threadgroup]])
{
    // ---- 1. the gate: scores -> probabilities -------------------------------------------
    // ONE THREAD A TOKEN for softmax, so the sum has one order -- imparo_moe_gate gives the
    // token a threadgroup and lets its thread 0 do this; here the token is the thread.
    for (uint t = tid; t < n_tok; t += tcount) {
        device const float * s = scores + (ulong)t * n_expert;
        device float * p = probs + (ulong)t * n_expert;
        if (gating == 0u) {
            float m = -INFINITY;
            for (uint e = 0u; e < n_expert; ++e) { m = max(m, s[e]); }
            float sum = 0.0f;
            for (uint e = 0u; e < n_expert; ++e) { const float v = exp(s[e] - m); p[e] = v; sum += v; }
            for (uint e = 0u; e < n_expert; ++e) { p[e] = p[e] / sum; }
        } else {
            for (uint e = 0u; e < n_expert; ++e) { p[e] = 1.0f / (1.0f + exp(-s[e])); }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- 2. the pick reads probabilities PLUS the file's bias ---------------------------
    // THE BIAS SELECTS AND DOES NOT WEIGH: the weights below come from the unbiased
    // probabilities, which is why `sel` is a second buffer and not `probs` overwritten.
    device const float * bias = (device const float *)(weights + bias_off);
    const bool has_bias = bias_off != IMPARO_NO_WEIGHT;
    for (uint i = tid; i < n_tok * n_expert; i += tcount) {
        sel[i] = probs[i] + (has_bias ? bias[i % n_expert] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- 3. the top-k pick, the one-pass list -------------------------------------------
    for (uint r = tid; r < n_tok; r += tcount) {
        const ulong row = (ulong)r * n_expert;
        top8_list t = top8_empty();
        for (uint i = 0u; i < n_expert; i++) {
            const float x = sel[row + i];
            if (t.i7 != TOP_K_EMPTY && !(x > t.v7)) { continue; }
            top8_insert(t, x, i);
        }
        top8_write(t, k, topk_ids, (ulong)r * k, topk_vals,
                   (ulong)n_tok * k + (ulong)r * k);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- 4. the plan: a counting sort of the (token, slot) pairs BY EXPERT ---------------
    threadgroup atomic_uint count[MOE_MAX_EXPERTS];
    threadgroup uint base[MOE_MAX_EXPERTS];
    for (uint e = tid; e < n_expert; e += tcount) {
        atomic_store_explicit(&count[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < n_tok; t += tcount) {
        for (uint j = 0u; j < k; ++j) {
            const uint e = topk_ids[t * k + j];
            inv[t * k + j] = (e < n_expert)
                ? atomic_fetch_add_explicit(&count[e], 1u, memory_order_relaxed)
                : 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint run = 0u;
        uint nact = 0u;
        for (uint e = 0u; e < n_expert; ++e) {
            base[e] = run;
            seg[e] = run;
            const uint c = atomic_load_explicit(&count[e], memory_order_relaxed);
            if (c != 0u) { seg[n_expert + 2u + nact] = e; nact += 1u; }
            run += c;
        }
        seg[n_expert] = run;
        seg[n_expert + 1u] = nact;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < n_tok; t += tcount) {
        device const float * p = probs + (ulong)t * n_expert;
        float sum = 0.0f;
        for (uint j = 0u; j < k; ++j) {
            const uint e = topk_ids[t * k + j];
            sum += (e < n_expert) ? p[e] : 0.0f;
        }
        const float denom = (normalise != 0u) ? max(sum, 6.103515625e-5f) : 1.0f;
        for (uint j = 0u; j < k; ++j) {
            const uint e = topk_ids[t * k + j];
            if (e >= n_expert) { continue; }
            const uint dst = base[e] + inv[t * k + j];
            perm[dst] = t;
            wgt[dst] = (p[e] / denom) * scale;
            inv[t * k + j] = dst;
        }
    }
}

kernel void imparo_moe_grouped(
    device const uchar * weights [[buffer(0)]],
    device const float * x       [[buffer(1)]],
    device float       * y       [[buffer(2)]],
    device const uint  * perm    [[buffer(3)]],
    device const uint  * seg     [[buffer(4)]],
    constant ulong & w_offset      [[buffer(5)]],
    constant ulong & expert_stride [[buffer(6)]],
    constant uint  & n_in          [[buffer(7)]],
    constant uint  & n_out         [[buffer(8)]],
    // 0: work row w reads activation row `perm[w]` (a token). 1: it reads row w.
    constant uint  & src_work_rows [[buffer(9)]],
    constant uint  & n_expert      [[buffer(14)]],
    threadgroup float * xstage [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]])
{
    // THE GRID SPANS ACTIVE EXPERTS, NOT ALL OF THEM. moe_plan compacted the ones that
    // hold a work row into seg's tail, so a decode step -- 4 work rows over 32 experts --
    // launches 4 expert groups and not 32.
    if (tgid.y >= seg[n_expert + 1u]) { return; }
    const uint e = seg[n_expert + 2u + tgid.y];
    const uint lo = seg[e];
    const uint hi = seg[e + 1u];
    if (lo >= hi) { return; }                       // an expert no token picked costs nothing
    const uint r = tgid.x * nsg + sgid;             // this simdgroup's output row
    // NOT `return`: the staged path has threadgroup barriers, and a simdgroup that left
    // early would hang the ones that stayed. n_out is a multiple of MOE_GROUPED_SGS on
    // every routed tensor this serves, so no simdgroup is idle today -- but a kernel whose
    // correctness depends on that is a hang waiting for the next model.
    const bool row_live = (r < n_out);
    if (!row_live && MOE_STAGE == 0u) { return; }

    const uint sub_per_block = tm_block_elems() / 32u;
    const uint bb = tm_block_bytes();
    const uint sc_bytes = tm_scale_bytes();
    const uint pay_bytes = bb - sc_bytes;
    const uint blocks = n_in / tm_block_elems();
    const uint subs = n_in / 32u;
    const uint so = tm_scale_src_off();
    const ulong wbase = w_offset + (ulong)e * expert_stride;

    const uint tid = sgid * 32u + lane;
    const uint tcount = nsg * 32u;

    for (uint c = lo; c < hi; c += MOE_ROWS) {
        const uint live = min(MOE_ROWS, hi - c);
        float acc[MOE_ROWS];
        for (uint i = 0u; i < MOE_ROWS; ++i) { acc[i] = 0.0f; }
        if (MOE_STAGE != 0u) {
        // ONE K-STEP: the 32 sub-blocks this threadgroup's lanes hold at once, lane L
        // owning sub-block t0 + L exactly as the unstaged loop does.
        for (uint t0 = 0u; t0 < subs; t0 += 32u) {
            const uint sb = t0 + lane;
            const bool sb_live = (sb < subs);
            float v[32];
            if (sb_live && row_live) {
                const uint blk = sb / sub_per_block;
                const uint sub = sb % sub_per_block;
                device const uchar * b = weights + wbase + ((ulong)r * blocks + blk) * bb;
                tm_sub32(b + so, b + (so == 0u ? sc_bytes : 0u), sub, v, false);
            } else {
                for (uint j = 0u; j < 32u; ++j) { v[j] = 0.0f; }
            }
            // The floats of this K-step that exist. Short on the last step when n_in is not
            // a multiple of 1024 -- the down projection's 1792 is, with subs = 56.
            const uint span = min(1024u, n_in - t0 * 32u);
            for (uint base = 0u; base < MOE_ROWS; base += MOE_STAGE_ROWS) {
                if (base >= live) { break; }   // uniform in the threadgroup: all break together
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0u; i < MOE_STAGE_ROWS; ++i) {
                    const uint w = base + i;
                    if (w >= live) { continue; }
                    const uint arow = (src_work_rows != 0u) ? (c + w) : perm[c + w];
                    device const float * src = x + (ulong)arow * n_in + (ulong)t0 * 32u;
                    threadgroup float * dst = xstage + i * 1024u;
                    for (uint q = tid; q < span; q += tcount) { dst[q] = src[q]; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0u; i < MOE_STAGE_ROWS; ++i) {
                    const uint w = base + i;
                    if (w >= live || !sb_live) { continue; }
                    // THE SAME FOUR-AT-A-TIME SUM the unstaged path runs. A plain j loop
                    // would be a different order and a different contraction, and the row
                    // would stop being bit-identical -- which is the whole point of the A/B.
                    const threadgroup float4 * xv4 =
                        (const threadgroup float4 *)(xstage + i * 1024u + lane * 32u);
                    float part = 0.0f;
                    for (uint j = 0u; j < 8u; ++j) {
                        const float4 t = xv4[j];
                        part += v[4u * j] * t.x + v[4u * j + 1u] * t.y
                              + v[4u * j + 2u] * t.z + v[4u * j + 3u] * t.w;
                    }
                    acc[w] += part;
                }
            }
        }
        } else {
        for (uint sb = lane; sb < subs; sb += 32u) {
            const uint blk = sb / sub_per_block;
            const uint sub = sb % sub_per_block;
            float v[32];
            // The block's two pointers, named once so both the dequantising fetch and the
            // deferred one can take them.
            device const uchar * sc_p;
            device const uchar * pay_p;
            if (WFMT_ROW) {
                device const uchar * b = weights + wbase + ((ulong)r * blocks + blk) * bb;
                sc_p  = b + so;
                pay_p = b + (so == 0u ? sc_bytes : 0u);
            } else {
                // TILE-MAJOR, the layout 91% of an MoE file's bytes are in after the load-time
                // repack. A unit is TM_UNIT_ROWS rows: its scale headers first, then its
                // payload rows -- the same addressing imparo_blk_gemv uses, so this kernel no
                // longer has to refuse the stack it is actually given.
                const ulong unit = ((ulong)(r / TM_UNIT_ROWS) * blocks + blk)
                                 * (ulong)(TM_UNIT_ROWS * bb);
                const uint slot = r % TM_UNIT_ROWS;
                sc_p  = weights + wbase + unit + (ulong)slot * sc_bytes;
                pay_p = weights + wbase + unit + (ulong)TM_UNIT_ROWS * sc_bytes
                      + (ulong)slot * pay_bytes;
            }
            if ((MOE_SKIP & 1u) != 0u) {
                for (uint j = 0u; j < 32u; ++j) { v[j] = float(j) * 0.01f; }
            } else {
                // A row-major block is not 16-byte aligned for every format, so the flag
                // follows the layout.
                tm_sub32(sc_p, pay_p, sub, v, !WFMT_ROW);
            }
            // THE REUSE. One decode, every work row of this expert in this pass.
            for (uint i = 0u; i < MOE_ROWS; ++i) {
                if (i >= live) { continue; }        // a compile-time loop with a guard: a
                                                    // runtime bound would index acc at run time
                const uint arow = (src_work_rows != 0u) ? (c + i) : perm[c + i];
                float part = 0.0f;
                if ((MOE_SKIP & 2u) != 0u) {
                    for (uint j = 0u; j < 32u; ++j) { part += v[j]; }
                } else {
                    device const float4 * xs4 =
                        (device const float4 *)(x + (ulong)arow * n_in + sb * 32u);
                    if ((MOE_SKIP & 4u) != 0u) {
                        part = v[0] * xs4[0].x;
                    } else {
                        for (uint j = 0u; j < 8u; ++j) {
                            const float4 t = xs4[j];
                            part += v[4u * j] * t.x + v[4u * j + 1u] * t.y
                                  + v[4u * j + 2u] * t.z + v[4u * j + 3u] * t.w;
                        }
                    }
                }
                acc[i] += part;
            }
        }
        }
        for (uint i = 0u; i < MOE_ROWS; ++i) {
            const float total = simd_sum(acc[i]);
            if (lane == 0u && i < live && row_live) { y[(ulong)(c + i) * n_out + r] = total; }
        }
    }
}

kernel void imparo_moe_combine(
    device const float * ydown [[buffer(0)]],
    device const float * wgt   [[buffer(1)]],
    device const uint  * inv   [[buffer(2)]],
    device float       * out   [[buffer(3)]],
    constant uint & n_embd [[buffer(4)]],
    constant uint & k      [[buffer(5)]],
    // How many threadgroups share one token's embedding. The grid is flattened into x:
    // Metal wants every grid-position attribute scalar or all the same vector width, and
    // the rest of this kernel's are scalar.
    constant uint & chunks [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    const uint tok = tg / chunks, chunk = tg % chunks;
    device float * o = out + (ulong)tok * n_embd;
    device const uint * rows = inv + (ulong)tok * k;
    // THE EMBEDDING SPLIT ACROSS THREADGROUPS. Every output element is an independent sum,
    // so one threadgroup per token left 2048 of them on ONE core: at decode the grid was a
    // single threadgroup, reading 40 KB in 12.5 us. The sum order per element is the slot
    // order either way, so this moves no bit.
    for (uint d = chunk * tcount + tid; d < n_embd; d += tcount * chunks) {
        // SLOT ORDER, every run. The k picks are summed in the order the router ranked them,
        // which is what makes a routed layer reproduce itself bit for bit.
        float s = 0.0f;
        for (uint j = 0u; j < k; ++j) {
            const uint row = rows[j];
            s += wgt[row] * ydown[(ulong)row * n_embd + d];
        }
        o[d] = s;
    }
}

// THE COMBINE FOLDED INTO THE NEXT NORM. A routed layer ends with the combine writing each
// token's weighted sum O, and the next block's norm reading it straight back to add it to the
// residual (imparo_rms_norm's pre-add mode): 16 MB written and read again per 2048-token
// chunk, and a dispatch. Here the norm forms that sum itself.
//
// THE SAME FLOATS AS THE TWO KERNELS: O is the combine's slot-order sum (`c += w * y`, the
// combine's expression per element), t = resid + O as in the pre-add sum pass, the squares
// summed in the same strided order and reduced by the same rms_finish, and the scale pass's
// `t * inv * w`. The residual is written in the sum pass and re-read by the thread that wrote
// it, so the combine is not recomputed. The float row and its half mirror are both written,
// as the norm writes them (skipping the float store once woke a visibility wobble; see
// imparo_rms_norm).
kernel void imparo_moe_combine_add_rms_norm(
    device const uchar * weights [[buffer(0)]],
    device float       * x       [[buffer(1)]],
    constant ulong & w_offset [[buffer(2)]], constant uint & width [[buffer(3)]],
    constant float & eps      [[buffer(4)]], constant uint & n_row [[buffer(5)]],
    device float       * resid   [[buffer(8)]],
    device const float * ydown   [[buffer(10)]],
    device half        * xh      [[buffer(11)]], constant uint & xh_on [[buffer(12)]],
    device const float * wgt     [[buffer(13)]],
    device const uint  * inv_rows [[buffer(14)]],
    constant uint & k [[buffer(15)]],
    threadgroup float * partial [[threadgroup(0)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  nsg   [[simdgroups_per_threadgroup]])
{
    const uint r = tgid.x;
    if (r >= n_row) { return; }
    const uint tid = tid3.x, tcount = tcnt3.x;
    const uint w4 = width / 4u;
    device float4 * r4 = (device float4 *)(resid + (ulong)r * width);
    device const uint * rows = inv_rows + (ulong)r * k;
    float sq = 0.0f;
    // k == 0: THE RESIDUAL ALONE, already holding resid + O (a mixer's output projection
    // added into it, RT_RESID). t is read, not re-added: `x + 0.0f` would turn a -0.0 into
    // +0.0, which the two-step form never does.
    for (uint i = tid; i < w4; i += tcount) {
        if (k == 0u) {
            const float4 t = r4[i];
            sq += dot(t, t);
            continue;
        }
        float4 c = float4(0.0f);
        for (uint j = 0u; j < k; ++j) {
            const uint row = rows[j];
            c += wgt[row] * ((device const float4 *)(ydown + (ulong)row * width))[i];
        }
        const float4 t = r4[i] + c;
        r4[i] = t;
        sq += dot(t, t);
    }
    sq = simd_sum(sq);
    float inv;
    if (nsg == 1u) {
        inv = rsqrt(simd_broadcast_first(sq) / float(width) + eps);
    } else {
        if (lane == 0) { partial[sgid] = sq; }
        inv = rms_finish(nsg, width, eps, partial, lane, sgid, false);
    }
    device float4 * row4 = (device float4 *)(x + (ulong)r * width);
    device const float4 * w4p = (device const float4 *)(weights + w_offset);
    device half4 * x4 = (device half4 *)(xh + (ulong)r * width);
    for (uint i = tid; i < w4; i += tcount) {
        const float4 v = r4[i] * inv * w4p[i];
        row4[i] = v;
        if (xh_on != 0u) { x4[i] = half4(v); }
    }
}

#if __METAL_VERSION__ >= 320
// THE BLOCK'S ENTRY AND EXIT (shared by every composed kernel; task #158 step 1).
// mega_enter: true = leave. Uniform across the threadgroup by broadcast through `flag`. The
// error word is sticky (FAIL-SAFE 1): once any dispatch of this process set it, every later
// dispatch leaves here, so a failed region costs one spin cap, never barriers x cap (the form
// that pinned the GPU for minutes on 2026-09-06); the host disables the route when it reads
// the word after the region. A host/kernel disagreement on the entry layout fails loudly.
inline bool mega_enter(device atomic_uint * err, uint entry_bytes, uint entry_size,
                       threadgroup uint * flag, uint tid) {
    if (tid == 0u) { *flag = atomic_load_explicit(err, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*flag != 0u) { return true; }
    if (entry_bytes != entry_size) {
        if (tid == 0u) { atomic_store_explicit(err, 1u | (0xfffu << 4), memory_order_relaxed); }
        return true;
    }
    return false;
}
// An entry this threadgroup cannot run (its n_tg is not the dispatch's, a zero width, a deep
// entry on a plain pipeline, the heads per item not the kernel's): report which entry and
// which threadgroup, sticky; the caller leaves.
inline void mega_fail_entry(device atomic_uint * err, uint dbg_seq, uint tgid, uint tid) {
    if (tid == 0u) { atomic_store_explicit(err, 1u | (0xeu << 28) | (dbg_seq << 16) | (tgid << 4), memory_order_relaxed); }
}
// The exit: the last threadgroup out resets the barrier and exit counters for the next dispatch.
inline void mega_exit(device atomic_uint * ctr, device atomic_uint * exitc, uint n_tg, uint tid) {
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(exitc, 1u, memory_order_relaxed) == n_tg - 1u) {
            atomic_store_explicit(ctr, 0u, memory_order_relaxed);
            atomic_store_explicit(exitc, 0u, memory_order_relaxed);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
    }
}

// THE ATTENTION PHASE over the cache (shared by every composed kernel; task #158 step 1).
// Vec body: n_heads x attn_split work items (head h, part), each folding its share of the
// span to a partial in the block's own scratch. Items stride the grid on the pipelines that
// run at one threadgroup per core (MEGA_STRIDE: a model may have more heads than
// threadgroups, and with one part per head item h stays on threadgroup h % n_tg, which the
// combine's stride matches); the f16 plain pipelines keep one item per threadgroup (the code
// they were measured with). Grouped deep body (task #151): n_kv x sub-groups x attn_split
// items (KV head, sub-group, slice), K/V once per item, only on a MEGA_DEEP pipeline whose
// heads per item are the kernel's own rule. Returns false when the entry cannot run on this
// pipeline (a deep entry on a plain pipeline, or attn_hq not the kernel's); the caller
// reports and leaves.
template <uint HD, uint KVW>
inline bool mega_attn_phase(device const float * qin, device const half * kc, device const half * vc,
                            device float * apart, uint n_heads, uint n_kv, uint kv_width, uint start_pos,
                            uint window, uint ring, device const uint * kv_pt, threadgroup float * red,
                            uint attn_body, uint attn_split, uint attn_hq, uint n_tg,
                            device atomic_uint * sync, uint scratch, uint dbg_slot, uint dbg_seq,
                            device atomic_uint * dbg_rec, uint tgid, uint tid, uint lane, uint sgid, uint nsg) {
    if (attn_body == 0u) {
        const uint n_items = n_heads * attn_split;
        device atomic_uint * cap = sync + MEGA_SYNC_HDR + scratch + MEGA_DBG_REC_WORDS + dbg_slot * MEGA_DBG_CAP_WORDS;
        if (MEGA_STRIDE) {
            for (uint item = tgid; item < n_items; item += n_tg) {
                mega_attn_vec<HD, KVW>(qin, kc, vc, apart, n_heads, n_kv, kv_width, start_pos, window, ring, kv_pt, red,
                                       item / attn_split, item % attn_split, attn_split, lane, sgid, nsg,
                                       dbg_rec + (ulong)item * MEGA_DBG_REC, cap, dbg_seq, tgid);
            }
        } else if (tgid < n_items) {
            mega_attn_vec<HD, KVW>(qin, kc, vc, apart, n_heads, n_kv, kv_width, start_pos, window, ring, kv_pt, red,
                                   tgid / attn_split, tgid % attn_split, attn_split, lane, sgid, nsg,
                                   dbg_rec + (ulong)tgid * MEGA_DBG_REC, cap, dbg_seq, tgid);
        }
        return true;
    }
    constexpr uint HQ = attn_group_hq(HD);
    if (!MEGA_DEEP || attn_hq != HQ) { return false; }
    const uint n_sub = (n_heads / n_kv + HQ - 1u) / HQ;
    // Debug builds keep an entered / finished mask of the deep body's threadgroups in sync
    // words 11 / 12 (the host prints them with a timeout).
    if (MEGA_DBG && tid == 0u) { atomic_fetch_or_explicit(sync + 11, 1u << (tgid & 31u), memory_order_relaxed); }
    if (tgid < n_kv * n_sub * attn_split) {
        mega_attn_group<HD, KVW>(qin, kc, vc, apart, n_heads, n_kv, kv_width, start_pos, window, ring, kv_pt, red,
                                 tgid, attn_split, lane, sgid, nsg);
    }
    if (MEGA_DBG && tid == 0u) { atomic_fetch_or_explicit(sync + 12, 1u << (tgid & 31u), memory_order_relaxed); }
    return true;
}
// Head h's combine on threadgroup h (striding the grid: the deep grid is smaller than the
// plain one and a model may have more heads than threadgroups), simdgroup 0. The assignment
// must stay threadgroup h: with one part per head the vec body wrote head h's partial from
// this very threadgroup, which is what lets a caller skip the grid barrier before this.
template <uint HD>
inline void mega_attn_combine_phase(device const float * apart, device float * out, uint n_heads, uint attn_split,
                                    uint had_v, uint n_tg, device atomic_uint * dbg_rec, uint tgid, uint sgid, uint lane) {
    for (uint h = tgid; h < n_heads; h += n_tg) {
        if (sgid == 0u) {
            mega_attn_combine<HD>(apart, out, h, attn_split, had_v, lane, dbg_rec + (ulong)h * attn_split * MEGA_DBG_REC);
        }
    }
}


// THE ROW IN THREADGROUP MEMORY (shared by every composed kernel; task #158 step 1): every
// threadgroup forms the same n_embd row in xn (the projection kernel's staged form, so the bits
// match the standalone norm kernels). Each brick ends with the threadgroup barrier its consumer
// needs; mega_tg_keep does not (its reader is past a later barrier).
// xn = rms(src) * w: src a device row; the weight at any float offset (the float4 path when it is
// 16-byte aligned, the scalar path otherwise -- the same arithmetic per element).
inline void mega_norm_to_tg(threadgroup float4 * xn, device const float * src, device const uchar * w_b, ulong w_off,
                            uint width, float eps, uint norm_t, threadgroup float * partial,
                            uint tid, uint tcount, uint lane, uint sgid, uint nsg) {
    const uint w4 = width / 4u;
    device const float4 * s4 = (device const float4 *)src;
    const float inv = rms_inv(s4, w4, width, eps, norm_t, partial, lane, sgid, nsg);
    device const float * wn = (device const float *)(w_b + w_off);
    if (((w_off / 4u) % 4u) == 0u) {
        device const float4 * wn4 = (device const float4 *)wn;
        for (uint i = tid; i < w4; i += tcount) { xn[i] = s4[i] * inv * wn4[i]; }
    } else {
        threadgroup float * xn1 = (threadgroup float *)xn;
        for (uint i = tid; i < width; i += tcount) { xn1[i] = src[i] * inv * wn[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// xn = rms(xn) * w, in place.
inline void mega_norm_tg(threadgroup float4 * xn, device const uchar * w_b, ulong w_off,
                         uint width, float eps, uint norm_t, threadgroup float * partial,
                         uint tid, uint tcount, uint lane, uint sgid, uint nsg) {
    const uint w4 = width / 4u;
    const float inv = rms_inv(xn, w4, width, eps, norm_t, partial, lane, sgid, nsg);
    device const float * wn = (device const float *)(w_b + w_off);
    if (((w_off / 4u) % 4u) == 0u) {
        device const float4 * wn4 = (device const float4 *)wn;
        for (uint i = tid; i < w4; i += tcount) { xn[i] = xn[i] * inv * wn4[i]; }
    } else {
        threadgroup float * xn1 = (threadgroup float *)xn;
        for (uint i = tid; i < width; i += tcount) { xn1[i] = xn1[i] * inv * wn[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// xn = src.
inline void mega_tg_load(threadgroup float4 * xn, device const float * src, uint w4, uint tid, uint tcount) {
    device const float4 * s4 = (device const float4 *)src;
    for (uint i = tid; i < w4; i += tcount) { xn[i] = s4[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// xn = a + xn (the residual joins the formed row).
inline void mega_tg_add(threadgroup float4 * xn, device const float * a, uint w4, uint tid, uint tcount) {
    device const float4 * a4 = (device const float4 *)a;
    for (uint i = tid; i < w4; i += tcount) { xn[i] = a4[i] + xn[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// xn = a + b.
inline void mega_tg_sum(threadgroup float4 * xn, device const float * a, device const float * b, uint w4, uint tid, uint tcount) {
    device const float4 * a4 = (device const float4 *)a;
    device const float4 * b4 = (device const float4 *)b;
    for (uint i = tid; i < w4; i += tcount) { xn[i] = a4[i] + b4[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Threadgroup 0 keeps the formed row in a device buffer (its reader is past a later barrier).
inline void mega_tg_keep(device float * dst, threadgroup const float4 * xn, uint w4, uint tgid, uint tid, uint tcount) {
    if (tgid == 0u) {
        device float4 * d4 = (device float4 *)dst;
        for (uint i = tid; i < w4; i += tcount) { d4[i] = xn[i]; }
    }
}

// THE SAME ROW WITHOUT HOLDING IT (task #175): cur = rms(a + b) * w and x' = a + b, both in
// DEVICE memory. The sum of squares is formed on the fly, so a + b never has to be
// materialised first and no extra grid barrier is needed. Every threadgroup writes the whole
// of `cur` -- the same inputs in the same order give the same bytes, so the overlap is
// idempotent, and it is what lets the consumer read `cur` after a threadgroup barrier alone.
// x' is written by threadgroup 0 only: its reader (the tail) is past a grid barrier.
inline float rms_sumsq_sum(device const float4 * a4, device const float4 * b4,
                           uint w4, uint norm_t, uint vg, uint lane) {
    float sq = 0.0f;
    for (uint i = vg * 32u + lane; i < w4; i += norm_t) {
        const float4 v = a4[i] + b4[i];
        sq += dot(v, v);
    }
    return simd_sum(sq);
}
inline void mega_dev_resid_norm(device float * cur, device float * keep,
                                device const float * a, device const float * b,
                                device const uchar * w_b, ulong w_off, uint width, float eps,
                                uint norm_t, threadgroup float * partial, uint tgid,
                                uint tid, uint tcount, uint lane, uint sgid, uint nsg) {
    const uint w4 = width / 4u;
    device const float4 * a4 = (device const float4 *)a;
    device const float4 * b4 = (device const float4 *)b;
    const uint vt = norm_t / 32u;
    for (uint vg = sgid; vg < vt; vg += nsg) {
        const float sq = rms_sumsq_sum(a4, b4, w4, norm_t, vg, lane);
        if (lane == 0u) { partial[vg] = sq; }
    }
    const float inv = rms_finish(vt, width, eps, partial, lane, sgid, true);
    device const float * wn = (device const float *)(w_b + w_off);
    if (((w_off / 4u) % 4u) == 0u) {
        device const float4 * wn4 = (device const float4 *)wn;
        device float4 * c4 = (device float4 *)cur;
        device float4 * k4 = (device float4 *)keep;
        for (uint i = tid; i < w4; i += tcount) {
            const float4 v = a4[i] + b4[i];
            c4[i] = v * inv * wn4[i];
            if (tgid == 0u) { k4[i] = v; }
        }
    } else {
        for (uint i = tid; i < width; i += tcount) {
            const float v = a[i] + b[i];
            cur[i] = v * inv * wn[i];
            if (tgid == 0u) { keep[i] = v; }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// Q4 ROW PHASES from the row in threadgroup memory (shared; task #158 step 1): the decode GEMV's
// fast path per row (mega_q4_row_partial_tg), the lane group's reduce, one store; rows striding
// the grid's simdgroups (LANES_PER_ROW lanes per row, 32 / LANES_PER_ROW rows per simdgroup).
// Up to three destinations in one balanced stripe: rows [0, n_a) -> a, [n_a, n_a + n_b) -> b,
// then c (a destination with 0 rows is absent). The q/k/v projection.
inline void mega_q4_rows3_tg(threadgroup const float4 * xs, uint blocks,
                             device const uchar * wa, device float * a, uint n_a,
                             device const uchar * wb, device float * b, uint n_b,
                             device const uchar * wc, device float * c, uint n_c,
                             uint sg_global, uint sg_total, uint sub, uint slot) {
    const uint rows_per_simd = 32u / LANES_PER_ROW;
    const uint total = n_a + n_b + n_c;
    for (uint r = sg_global * rows_per_simd + slot; r < total; r += sg_total * rows_per_simd) {
        device const uchar * w; device float * dst; uint rr;
        if (r < n_a)            { w = wa; dst = a; rr = r; }
        else if (r < n_a + n_b) { w = wb; dst = b; rr = r - n_a; }
        else                    { w = wc; dst = c; rr = r - n_a - n_b; }
        float acc = mega_q4_row_partial_tg(w + (ulong)rr * blocks * Q4_0_BYTES, xs, blocks, sub);
        acc = lane_group_sum(acc);
        if (sub == 0u) { dst[rr] = acc; }
    }
}
// The gated pair: g[r] = act(gate_r . x) * (up_r . x), both rows by the same simdgroup (the
// producer applies the activation, so no act*mul phase and no barrier follow).
inline void mega_q4_gated_rows_tg(threadgroup const float4 * xs, uint blocks, device const uchar * wg, device const uchar * wu,
                                  device float * g, uint n_rows, uint sg_global, uint sg_total, uint sub, uint slot) {
    const uint rows_per_simd = 32u / LANES_PER_ROW;
    for (uint r = sg_global * rows_per_simd + slot; r < n_rows; r += sg_total * rows_per_simd) {
        float ag = mega_q4_row_partial_tg(wg + (ulong)r * blocks * Q4_0_BYTES, xs, blocks, sub);
        float au = mega_q4_row_partial_tg(wu + (ulong)r * blocks * Q4_0_BYTES, xs, blocks, sub);
        ag = lane_group_sum(ag);
        au = lane_group_sum(au);
        if (sub == 0u) { g[r] = imparo_act_f(ag) * au; }
    }
}
// Scaled rows: dst[r] = act(w_r . x) * scale[r] (gemma4's per-layer gate).
inline void mega_q4_scaled_rows_tg(threadgroup const float4 * xs, uint blocks, device const uchar * w, device float * dst,
                                   uint n_rows, device const float * scale, uint sg_global, uint sg_total, uint sub, uint slot) {
    const uint rows_per_simd = 32u / LANES_PER_ROW;
    for (uint r = sg_global * rows_per_simd + slot; r < n_rows; r += sg_total * rows_per_simd) {
        float acc = mega_q4_row_partial_tg(w + (ulong)r * blocks * Q4_0_BYTES, xs, blocks, sub);
        acc = lane_group_sum(acc);
        if (sub == 0u) { dst[r] = imparo_act_f(acc) * scale[r]; }
    }
}

// Threadgroup 0: dst = a + b (device rows; the tail's residual).
inline void mega_dev_sum_tg0(device float * dst, device const float * a, device const float * b, uint w4, uint tgid, uint tid, uint tcount) {
    if (tgid == 0u) {
        device const float4 * a4 = (device const float4 *)a;
        device const float4 * b4 = (device const float4 *)b;
        device float4 * d4 = (device float4 *)dst;
        for (uint i = tid; i < w4; i += tcount) { d4[i] = a4[i] + b4[i]; }
    }
}
// MEGA_DBG: a non-finite value in the row o_proj is about to read marks the head's part-0
// debug record (flag 4).
template <uint HD>
inline void mega_dbg_nonfinite_row(device const float * row, uint n, uint attn_split, device atomic_uint * dbg_rec, uint tid, uint tcount) {
    for (uint i = tid; i < n; i += tcount) {
        if (mega_nonfinite(row[i])) {
            atomic_fetch_or_explicit(dbg_rec + (ulong)(i / HD) * attn_split * MEGA_DBG_REC + 7, 4u, memory_order_relaxed);
        }
    }
}


// The row norms of the block are the rms_inv brick (imparo_rms_norm's reduction at the
// virtual geometry norm_t, which the host derives from the row width as the kernel does).

// dst = (addend + rms(src) * w1) * out_scale; dual: out = rms(dst) * w2. One threadgroup.
inline void mega_norm_add_row(device const float * src, device const float * addend,
                              device float * dst, device const uchar * w1b, ulong w1_off,
                              uint width, float eps, float out_scale, bool dual,
                              device const uchar * w2b, ulong w2_off, device float * out,
                              uint norm_t, threadgroup float * partial, device float * stage,
                              uint tid, uint tcount, uint lane, uint sgid, uint nsg) {
    const uint w4 = width / 4u;
    device const float4 * s4 = (device const float4 *)src;
    device const float4 * a4 = (device const float4 *)addend;
    device float4       * d4 = (device float4 *)dst;
    device float4       * st4 = (device float4 *)stage;
    const float inv = rms_inv(s4, w4, width, eps, norm_t, partial, lane, sgid, nsg);
    device const float * w1 = (device const float *)(w1b + w1_off);
    if (((w1_off / 4u) % 4u) == 0u) {
        device const float4 * w14 = (device const float4 *)w1;
        for (uint i = tid; i < w4; i += tcount) { st4[i] = s4[i] * inv * w14[i]; }
    } else {
        for (uint i = tid; i < width; i += tcount) { stage[i] = src[i] * inv * w1[i]; }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (out_scale == 1.0f) {
        for (uint i = tid; i < w4; i += tcount) { d4[i] = a4[i] + st4[i]; }
    } else {
        for (uint i = tid; i < w4; i += tcount) { st4[i] = a4[i] + st4[i]; }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < w4; i += tcount) { d4[i] = st4[i] * out_scale; }
    }
    if (!dual) { return; }
    threadgroup_barrier(mem_flags::mem_device);
    const float inv2 = rms_inv(d4, w4, width, eps, norm_t, partial, lane, sgid, nsg);
    device const float * w2 = (device const float *)(w2b + w2_off);
    device float4 * o4 = (device float4 *)out;
    if (((w2_off / 4u) % 4u) == 0u) {
        device const float4 * w24 = (device const float4 *)w2;
        for (uint i = tid; i < w4; i += tcount) { o4[i] = d4[i] * inv2 * w24[i]; }
    } else {
        for (uint i = tid; i < width; i += tcount) { out[i] = dst[i] * inv2 * w2[i]; }
    }
}

// One GEMV phase over rows [0, n_out): the lane group's row, the decode fast path's
// arithmetic, the lane-group reduce, one store.
inline void mega_gemv_phase(device const uchar * w, ulong w_off, device const float * xin,
                            device float * yout, uint n_in, uint n_out,
                            uint sg_global, uint sg_total, uint sub, uint slot) {
    const uint rows_per_simd = 32u / LANES_PER_ROW;
    const uint blocks = n_in / QK4_0;
    device const float4 * xs = (device const float4 *)xin;
    for (uint r = sg_global * rows_per_simd + slot; r < n_out; r += sg_total * rows_per_simd) {
        device const uchar * row = w + w_off + (ulong)r * blocks * Q4_0_BYTES;
        float acc = mega_q4_row_partial(row, xs, blocks, sub);
        acc = lane_group_sum(acc);
        if (sub == 0u) { yout[r] = acc; }
    }
}

// A head row into this layer's cache at slot `ps`: f16 as half4; q4_0 / q8_0 with
// imparo_kv_store_q4 / _q8's arithmetic (llama.cpp's quantizers), one thread per 32-value
// block, under the same safe math mode and without contraction so the bytes match the
// dispatch path's store. The row's storage type is the pipeline's constant (KVT_K / KVT_V).
template <uint HD>
inline void mega_kv_store_row_f16(device half * cache, uint kv_width, uint ps, uint hidx,
                                  device const float * row, uint tid, uint tcount) {
    constexpr uint hw4 = HD / 4u;
    device half4 * c4 = (device half4 *)(cache + (ulong)ps * kv_width + hidx * HD);
    device const float4 * r4 = (device const float4 *)row;
    for (uint i = tid; i < hw4; i += tcount) { c4[i] = half4(r4[i]); }
}
// The quantized forms, in their own function so the safe-math pragmas scope to them alone.
template <uint HD>
inline void mega_kv_store_row_q(uint kvt, device half * cache, uint kv_width, uint ps, uint hidx,
                                device const float * row, uint tid, uint tcount) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    if (kvt == 1u) { mega_kv_store_row_f16<HD>(cache, kv_width, ps, hidx, row, tid, tcount); return; }
    constexpr uint blocks = HD / 32u;
    device uchar * base = (device uchar *)cache + imparo_kv_row_off(kvt, kv_width, ps, hidx * HD);
    for (uint b = tid; b < blocks; b += tcount) {
        device const float * x = row + b * 32u;
        if (kvt == 2u) {
            device uchar * blk = base + b * 18u;
            float amax = 0.0f, vmax = 0.0f;
            for (uint j = 0; j < 32u; ++j) {
                const float v = x[j];
                if (amax < fabs(v)) { amax = fabs(v); vmax = v; }
            }
            const float d = vmax / -8.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            const half dh = half(d);
            blk[0] = as_type<ushort>(dh) & 0xFFu;
            blk[1] = (as_type<ushort>(dh) >> 8) & 0xFFu;
            for (uint j = 0; j < 16u; ++j) {
                const float x0 = x[j] * id;
                const float x1 = x[16u + j] * id;
                const uchar xi0 = (uchar)min(15, (int)(char)(x0 + 8.5f));
                const uchar xi1 = (uchar)min(15, (int)(char)(x1 + 8.5f));
                blk[2u + j] = xi0 | (xi1 << 4);
            }
        } else {
            device uchar * blk = base + b * 34u;
            float amax = 0.0f;
            for (uint j = 0; j < 32u; ++j) { amax = max(amax, fabs(x[j])); }
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            const half dh = half(d);
            blk[0] = as_type<ushort>(dh) & 0xFFu;
            blk[1] = (as_type<ushort>(dh) >> 8) & 0xFFu;
            for (uint j = 0; j < 32u; ++j) { blk[2u + j] = (uchar)(char)rint(x[j] * id); }
        }
    }
}

// OPERAND TABLE: every pointer the layer block reads or writes, as GPU addresses the host
// writes (MTLBuffer.gpuAddress + offset; Metal's argument-buffer form). The block forms its
// pointers from it, so a layer's tensors may sit in any segment or tier (memory-fit tiers) and
// the 31-slot buffer limit is not a limit: the block binds this table, the sync buffer and
// the params. Weight pointers are SEGMENT bases; the params' *_off are local to them.
// THE HEAD ROW (shared by every composed kernel; task #158 step 1): one Q, K or V head row
// after the projection. Q and K: rms(w) at the head-dim thread rule (imparo_rms_norm's form),
// NEOX rope (imparo_rope_neox's rotation), the score scale when the model applies it to Q
// (LFM2; 1.0 = none). V: an unweighted rms when the model has one (norm_v; gemma4), nothing
// otherwise. Then the cache basis (task #156: Q/K rotated by had_k, V by had_v before a
// quantized store, 0 = plain) and the cache row for K and V (imparo_kv_store's conversion; the
// storage type is the pipeline's KVT_K / KVT_V). The caller owns the row loop and places any
// barrier the next row needs (`partial` is reused).
template <uint HD>
inline void mega_head_row(bool is_q, bool is_k, uint hidx, device float * row,
                          device const uchar * wn_b, ulong wn_off, bool norm_v, uint norm_t_head, float eps,
                          uint n_rot, float rope_base, constant float * freqs, uint n_freqs, uint start_pos, float q_scale,
                          uint had_k, uint had_v, device half * kc_w, device half * vc_w, uint kv_width, uint ring,
                          device const uint * kv_pt, threadgroup float * partial,
                          uint tid, uint tcount, uint lane, uint sgid, uint nsg) {
    device float4 * r4 = (device float4 *)row;
    constexpr uint hw4 = HD / 4u;
    if (is_q || is_k || norm_v) {
        const float inv = rms_inv((device const float4 *)row, hw4, HD, eps, norm_t_head, partial, lane, sgid, nsg);
        if (is_q || is_k) {
            device const float * wn = (device const float *)(wn_b + wn_off);
            if (((wn_off / 4u) % 4u) == 0u) {
                device const float4 * wn4 = (device const float4 *)wn;
                for (uint i = tid; i < hw4; i += tcount) { r4[i] = r4[i] * inv * wn4[i]; }
            } else {
                for (uint i = tid; i < HD; i += tcount) { row[i] = row[i] * inv * wn[i]; }
            }
        } else {
            for (uint i = tid; i < hw4; i += tcount) { r4[i] = r4[i] * inv; }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (is_q || is_k) {
        const uint half_rot = n_rot / 2u;
        for (uint i = tid; i < half_rot; i += tcount) {
            rope_neox_pair(row, i, half_rot, n_rot, rope_base, freqs, n_freqs, start_pos);
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (is_q && q_scale != 1.0f) {
            for (uint i = tid; i < hw4; i += tcount) { r4[i] = r4[i] * q_scale; }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    const uint had = (is_q || is_k) ? had_k : had_v;
    if (MEGA_KVQ && had != 0u) { mega_hadamard_row<HD>(row, had, tid, tcount); }
    if (!is_q) {
        const uint ps = kv_slot(start_pos, ring, kv_pt);
        if (MEGA_KVQ) { mega_kv_store_row_q<HD>(is_k ? KVT_K : KVT_V, is_k ? kc_w : vc_w, kv_width, ps, hidx, row, tid, tcount); }
        else         { mega_kv_store_row_f16<HD>(is_k ? kc_w : vc_w, kv_width, ps, hidx, row, tid, tcount); }
    }
}


// THE ENTRY: what one dispatch of the block needs -- its operand table (GPU addresses) and
// the layer's params -- as one struct the host passes as a kernel argument, at a compile-time
// offset in the constant address space. That is the form that fits two threadgroups per
// core; read through a device pointer or a runtime index, the same fields are ordinary loads
// the compiler keeps live and only one threadgroup fits (design doc, constraint 8). Per-token
// values travel in MegaToken.
// THE ENTRY (task #158 step 2): ONE shape for every architecture -- a table of GPU pointers
// (weights as their segment base, activation rows, cache views, the page table), a table of
// weight offsets local to the pointer's segment, a table of words and a table of floats,
// indexed by the program's slot constants below. The indices are compile-time literals, so
// every read is a constant-argument load at a compile-time offset (constraint 8), as the typed
// structs were. Words 0..3 and float 0 are the header every program fills: the dispatch's
// threadgroup count, n_embd, the scratch floats, the norm's virtual thread count; eps.
// The slot constants (MEGA_NP.., MEGA_U_*, G4_*, L2_*) are emitted here by build.rs from mega_slots.rs.
// @@MEGA_SLOTS@@
struct MegaEntryG {
    device const uchar * ptr[MEGA_NP];
    ulong off[MEGA_NO];
    uint  u[MEGA_NU];
    float f[MEGA_NF];
};
struct MegaToken {
    uint start_pos, dbg_slot, n_tg, entry_bytes;   // entry_bytes: the host's sizeof(entry), refused on drift
    uint dbg_seq, entry_index, n_entries, probe;   // dbg_seq: this dispatch's index in the region (debug records);
                                                   // entry_index / n_entries: the program form's entry (task #153);
                                                   // probe: non-zero = the admission probe (task #203), and its
                                                   // spin budget in units of 1024 iterations
};


// THE PHASE BUNDLE (task #158 step 3): every phase of every architecture takes the same
// arguments, so ONE frame (emitted from mega_program.rs) serves them all and a phase list is
// something the host can write down.
// THE GEMMA4 PROGRAM'S PHASES: one inline function per phase. A phase
// never takes a grid barrier: the barriers belong to the sequence (MEGA_STEP), which is what
// makes the sequence data rather than code. Each phase casts the operands it needs out of the
// entry's pointer table, so the kernel frame carries no architecture-specific unpack.
#define MEGA_PH_ARGS constant MegaEntryG & ent, constant float * freqs, constant MegaToken & tok, \
    device atomic_uint * sync, device float * apart, device atomic_uint * dbg_rec, uint dbg_seq, \
    threadgroup float4 * xn, threadgroup float * partial, uint tgid, uint tid, uint lane, uint sgid, \
    uint nsg, uint tcount, uint sg_global, uint sg_total, uint sub, uint slot, uint w4
#define MEGA_PH_CALL ent, freqs, tok, sync, apart, dbg_rec, dbg_seq, xn, partial, tgid, tid, lane, sgid, \
    nsg, tcount, sg_global, sg_total, sub, slot, w4

// PQ0 (every threadgroup): xn = rms(x) * w_in -- the layer's input norm, consumed from threadgroup
// memory by the rows below. PQ1: the q rows -> qin, k rows -> kbuf, v rows -> vbuf, one balanced stripe.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_in_norm_qkv_rows(MEGA_PH_ARGS) {
    mega_norm_to_tg(xn, (device float *)ent.ptr[G4_X], (device const uchar *)ent.ptr[G4_W_IN], ent.off[G4_O_IN_OFF],
                    ent.u[G4_U_N_EMBD], ent.f[G4_F_EPS], ent.u[G4_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
    mega_q4_rows3_tg(xn, ent.u[G4_U_N_EMBD] / QK4_0,
                     (device const uchar *)ent.ptr[G4_W_Q] + ent.off[G4_O_WQ_OFF], (device float *)ent.ptr[G4_QIN], ent.u[G4_U_Q_ROWS],
                     (device const uchar *)ent.ptr[G4_W_K] + ent.off[G4_O_WK_OFF], (device float *)ent.ptr[G4_KBUF], ent.u[G4_U_HAS_KV] != 0u ? ent.u[G4_U_KV_WIDTH] : 0u,
                     (device const uchar *)ent.ptr[G4_W_V] + ent.off[G4_O_WV_OFF], (device float *)ent.ptr[G4_VBUF], ent.u[G4_U_HAS_KV] != 0u ? ent.u[G4_U_KV_WIDTH] : 0u,
                     sg_global, sg_total, sub, slot);
}
// PQ2: one head row per threadgroup, grid stride (the shared head-row brick: rms(w_qn) / rms(w_kn)
// then rope for Q / K, the unweighted rms for V, the cache basis, the cache row).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_head_rows(MEGA_PH_ARGS) {
    const uint n_rows = ent.u[G4_U_N_HEADS] + (ent.u[G4_U_HAS_KV] != 0u ? 2u * ent.u[G4_U_N_KV] : 0u);
    for (uint r = tgid; r < n_rows; r += ent.u[G4_U_N_TG]) {
        const bool is_q = r < ent.u[G4_U_N_HEADS];
        const bool is_k = !is_q && r < ent.u[G4_U_N_HEADS] + ent.u[G4_U_N_KV];
        const uint hidx = is_q ? r : (is_k ? r - ent.u[G4_U_N_HEADS] : r - ent.u[G4_U_N_HEADS] - ent.u[G4_U_N_KV]);
        device float * row = (is_q ? (device float *)ent.ptr[G4_QIN] : (is_k ? (device float *)ent.ptr[G4_KBUF] : (device float *)ent.ptr[G4_VBUF])) + (ulong)hidx * HD;
        mega_head_row<HD>(is_q, is_k, hidx, row, (device const uchar *)(is_q ? ent.ptr[G4_W_QN] : ent.ptr[G4_W_KN]),
                          is_q ? ent.off[G4_O_QN_OFF] : ent.off[G4_O_KN_OFF], true, ent.u[G4_U_NORM_T_HEAD], ent.f[G4_F_EPS],
                          ent.u[G4_U_N_ROT], ent.f[G4_F_ROPE_BASE], freqs, ent.u[G4_U_N_FREQS], tok.start_pos, 1.0f, ent.u[G4_U_HAD_K], ent.u[G4_U_HAD_V],
                          (device half *)ent.ptr[G4_KC_W], (device half *)ent.ptr[G4_VC_W], ent.u[G4_U_KV_WIDTH], ent.u[G4_U_RING],
                          (device const uint *)ent.ptr[G4_KV_PT], partial, tid, tcount, lane, sgid, nsg);
    }
}
// PA: attention over the cache (the shared phase brick). False = this entry cannot run on this
// pipeline (a deep entry on a plain one, or the heads per item disagree); the caller reports it.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) bool g4_ph_attn(MEGA_PH_ARGS) {
    return mega_attn_phase<HD, KVW>((device float *)ent.ptr[G4_QIN], (device const half *)ent.ptr[G4_KC], (device const half *)ent.ptr[G4_VC], apart,
                                    ent.u[G4_U_N_HEADS], ent.u[G4_U_N_KV], ent.u[G4_U_KV_WIDTH], tok.start_pos, ent.u[G4_U_WINDOW], ent.u[G4_U_RING],
                                    (device const uint *)ent.ptr[G4_KV_PT], (threadgroup float *)xn, ent.u[G4_U_ATTN_BODY], ent.u[G4_U_ATTN_SPLIT],
                                    ent.u[G4_U_ATTN_HQ], ent.u[G4_U_N_TG], sync, ent.u[G4_U_SCRATCH], tok.dbg_slot, dbg_seq, dbg_rec, tgid, tid, lane, sgid, nsg);
}
// Head h's combine on threadgroup h.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_attn_combine(MEGA_PH_ARGS) {
    mega_attn_combine_phase<HD>(apart, (device float *)ent.ptr[G4_ATTN_OUT], ent.u[G4_U_N_HEADS], ent.u[G4_U_ATTN_SPLIT],
                                ent.u[G4_U_HAD_V], ent.u[G4_U_N_TG], dbg_rec, tgid, sgid, lane);
}
// P0 (front): o = W_o . attn (the o_proj rows). MEGA_DBG first: what o_proj is about to read
// through the const view; a non-finite value marks the head's part-0 record (flag 4).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_oproj(MEGA_PH_ARGS) {
    if (MEGA_DBG && tgid == 0u && ent.u[G4_U_ATTN_PHASE] != 0u) {
        mega_dbg_nonfinite_row<HD>((device const float *)ent.ptr[G4_ATTN], ent.u[G4_U_ATTN_IN], ent.u[G4_U_ATTN_SPLIT], dbg_rec, tid, tcount);
    }
    mega_gemv_phase((device const uchar *)ent.ptr[G4_W_O], ent.off[G4_O_WO_OFF], (device const float *)ent.ptr[G4_ATTN],
                    (device float *)ent.ptr[G4_O], ent.u[G4_U_ATTN_IN], ent.u[G4_U_N_EMBD], sg_global, sg_total, sub, slot);
}
// P0b (every threadgroup): the sandwich -- mid = x + rms(o) * w_npa in threadgroup memory (the dual
// norm kernel's staged form), threadgroup 0 keeps mid in u for the post-FFN residual (u is read only
// after two more barriers; o and x are only READ here), then xn = rms(mid) * w_nf is this layer's
// FFN input, consumed from threadgroup memory.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_sandwich(MEGA_PH_ARGS) {
    mega_norm_to_tg(xn, (device float *)ent.ptr[G4_O], (device const uchar *)ent.ptr[G4_W_NPA], ent.off[G4_O_NPA_OFF],
                    ent.u[G4_U_N_EMBD], ent.f[G4_F_EPS], ent.u[G4_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
    mega_tg_add(xn, (device float *)ent.ptr[G4_X], w4, tid, tcount);
    mega_tg_keep((device float *)ent.ptr[G4_U], xn, w4, tgid, tid, tcount);
    mega_norm_tg(xn, (device const uchar *)ent.ptr[G4_W_NF], ent.off[G4_O_NF_OFF], ent.u[G4_U_N_EMBD], ent.f[G4_F_EPS],
                 ent.u[G4_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
}
// No front: the FFN input was formed by the previous dispatch; every threadgroup takes its own
// copy so the gated rows have one form.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_ffn_input_copy(MEGA_PH_ARGS) {
    mega_tg_load(xn, (device const float *)ent.ptr[G4_CUR], w4, tid, tcount);
}
// P1: g[r] = act(gate row r) * (up row r), both rows by the same simdgroup, from xn; the two
// reductions are the dispatch GEMV's shuffle tree, the product is act_mul's expression.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_ffn_gated(MEGA_PH_ARGS) {
    mega_q4_gated_rows_tg(xn, ent.u[G4_U_N_EMBD] / QK4_0, (device const uchar *)ent.ptr[G4_W_GATE] + ent.off[G4_O_GATE_OFF],
                          (device const uchar *)ent.ptr[G4_W_UP] + ent.off[G4_O_UP_OFF], (device float *)ent.ptr[G4_G],
                          ent.u[G4_U_N_MID], sg_global, sg_total, sub, slot);
}
// P3: down rows -> x (x's previous content, the layer's input residual, was consumed by the sandwich).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_ffn_down(MEGA_PH_ARGS) {
    mega_gemv_phase((device const uchar *)ent.ptr[G4_W_DOWN], ent.off[G4_O_DOWN_OFF], (device const float *)ent.ptr[G4_G],
                    (device float *)ent.ptr[G4_X], ent.u[G4_U_N_MID], ent.u[G4_U_N_EMBD], sg_global, sg_total, sub, slot);
}
// P4+P5 (every threadgroup): xn = o + rms(x) * w_n1 in threadgroup memory -- the same staged form as
// mega_norm_add_row, so the bits match -- then this threadgroup's PLE gate rows read xn:
// gate[r] = act(row r . xn) * per_layer[pl_off + r]. No grid barrier between the norm and its
// consumer; x (the down output) is only READ here, and the formed row is written back to x by
// threadgroup 0 after the next barrier.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_ple_gate(MEGA_PH_ARGS) {
    mega_norm_to_tg(xn, (device float *)ent.ptr[G4_X], (device const uchar *)ent.ptr[G4_W_N1], ent.off[G4_O_N1_OFF],
                    ent.u[G4_U_N_EMBD], ent.f[G4_F_EPS], ent.u[G4_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
    mega_tg_add(xn, (device float *)(ent.u[G4_U_FRONT] != 0u ? ent.ptr[G4_U] : ent.ptr[G4_O]), w4, tid, tcount);   // the post-attention residual
    mega_q4_scaled_rows_tg(xn, ent.u[G4_U_N_EMBD] / QK4_0, (device const uchar *)ent.ptr[G4_W_PG] + ent.off[G4_O_PG_OFF],
                           (device float *)ent.ptr[G4_GATE], ent.u[G4_U_PLE],
                           (device const float *)ent.ptr[G4_PER_LAYER] + ent.u[G4_U_PL_OFF], sg_global, sg_total, sub, slot);
}
// P7: threadgroup 0 writes the formed row back to x (every reader of the down output has passed the
// barrier); PLE proj rows [0, n_embd) from gate -> back.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_ple_proj(MEGA_PH_ARGS) {
    mega_tg_keep((device float *)ent.ptr[G4_X], xn, w4, tgid, tid, tcount);
    mega_gemv_phase((device const uchar *)ent.ptr[G4_W_PP], ent.off[G4_O_PP_OFF], (device const float *)ent.ptr[G4_GATE],
                    (device float *)ent.ptr[G4_BACK], ent.u[G4_U_PLE], ent.u[G4_U_N_EMBD], sg_global, sg_total, sub, slot);
}
// P8 (threadgroup 0): x = (x + rms(back) * w_n2) * out_scale; cur = rms(x) * w_n3 (staged in g).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void g4_ph_tail(MEGA_PH_ARGS) {
    if (tgid == 0u) {
        mega_norm_add_row((device const float *)ent.ptr[G4_BACK], (device float *)ent.ptr[G4_X], (device float *)ent.ptr[G4_X],
                          (device const uchar *)ent.ptr[G4_W_N2], ent.off[G4_O_N2_OFF], ent.u[G4_U_N_EMBD], ent.f[G4_F_EPS],
                          ent.f[G4_F_OUT_SCALE], ent.u[G4_U_HAS_NEXT] != 0u, (device const uchar *)ent.ptr[G4_W_N3], ent.off[G4_O_N3_OFF],
                          (device float *)ent.ptr[G4_NXT], ent.u[G4_U_NORM_T], partial, (device float *)ent.ptr[G4_G], tid, tcount, lane, sgid, nsg);
    }
}

// The gemma4 kernel is emitted from its program (task #158 step 3): see the marker below.
// ===================================================================================
// LFM2: one decode layer as one persistent dispatch (task #147) -- the bricks its phases are
// built from, then the phases. The entry names the layer's operands; `mixer` says what
// precedes the tail:
//   MIXER_SHORTCONV  cur = rms(x) * w_op (every threadgroup, threadgroup memory); bcx = W_in . cur
//                    | barrier | the conv step's row from bcx, the history and w_conv (every
//                    threadgroup); o = W_out . row | barrier | threadgroup 0 shifts the history
//                    and, with a checkpoint armed at this token, writes the same row to `snap`
//   MIXER_ATTENTION  cur = rms(x) * w_op; the q/k/v rows (one tile-major unit per simdgroup, a
//                    head row's units inside one threadgroup) then, threadgroup-locally, Q:
//                    rms(w_qn), rope, the score scale, K: rms(w_kn), rope, the cache row, V: the
//                    cache row | barrier | attention over the cache (the vec body, one
//                    threadgroup per (head, part)), the parts merged in place when a head has
//                    one | barrier | o = W_o . attn | barrier
//   MIXER_NONE       o already holds the mixer's output (the dispatch path's)
// then the tail: x' = x + o; cur = rms(x') * w_ffn (every threadgroup; threadgroup 0 keeps x'
// in u); g = act(gate . cur) * (up . cur) | barrier | o = down . g | barrier | threadgroup 0:
// x = x' + o. The next layer's block forms its own operator norm from x.
// Weights are Q8_0 tile-major (Q8_TM on this pipeline); the activation is the model's (EPI_ACT
// stamped at build); norm weights are float rows at 16-byte-aligned offsets (the host checks).
// Same block rules as the gemma4 kernel: the entry is a constant-space kernel argument, the
// error word is sticky, one monotonic barrier counter reset by the last threadgroup out.

// One unit of a tile-major Q8_0 matrix (Q8_TM_UNIT_ROWS = 8 rows, every K block) by one
// simdgroup: the decode GEMV's lane roles (u = block parity, rp = row pair, il = the quad of
// 8 values in a block), its per-block arithmetic, reduced over il (xor 1, 2) and u (xor 16).
// On return lanes with (lane & 19) == 0 hold rows r0 + 2 rp and r0 + 2 rp + 1 in (acca, accb).
// XS4 is a float4 pointer into the x vector, threadgroup or device.
template <typename XS4>
inline void q8_tm_unit_rows(device const uchar * w, uint blocks, uint r0, XS4 x4,
                            uint lane, thread float & acca, thread float & accb) {
    const uint u = lane / 16u, rp = (lane / 4u) % 4u, il = lane % 4u;
    const uint rowa = r0 + rp * 2u, rowb = rowa + 1u;
    acca = 0.0f; accb = 0.0f;
    for (uint ib = u; ib < blocks; ib += 2u) {
        const uint i4 = ib * (QK8_0 / 4u) + il * 2u;
        const float4 xa = x4[i4], xb4 = x4[i4 + 1u];
        device const uchar * pa = w + q8_tm_payload(rowa, ib, blocks) + il * 8u;
        device const uchar * pb = w + q8_tm_payload(rowb, ib, blocks) + il * 8u;
        const float da = float(*(device const half *)(w + q8_tm_scale(rowa, ib, blocks, 0u)));
        const float db = float(*(device const half *)(w + q8_tm_scale(rowb, ib, blocks, 0u)));
        const char4 qa0 = as_type<char4>(*(device const uint *)pa);
        const char4 qa1 = as_type<char4>(*(device const uint *)(pa + 4u));
        const char4 qb0 = as_type<char4>(*(device const uint *)pb);
        const char4 qb1 = as_type<char4>(*(device const uint *)(pb + 4u));
        acca += (dot(float4(qa0), xa) + dot(float4(qa1), xb4)) * da;
        accb += (dot(float4(qb0), xa) + dot(float4(qb1), xb4)) * db;
    }
    acca += simd_shuffle_xor(acca, 1u);  accb += simd_shuffle_xor(accb, 1u);
    acca += simd_shuffle_xor(acca, 2u);  accb += simd_shuffle_xor(accb, 2u);
    acca += simd_shuffle_xor(acca, 16u); accb += simd_shuffle_xor(accb, 16u);
}
// Rows [0, n_out) of a tile-major matrix: units strided over the grid's simdgroups.
template <typename XS4>
inline void mega_q8tm_phase(device const uchar * w, ulong w_off, XS4 x4, device float * y,
                            uint n_in, uint n_out, uint sg_global, uint sg_total, uint lane) {
    const uint blocks = n_in / QK8_0, units = n_out / Q8_TM_UNIT_ROWS;
    device const uchar * wb = w + w_off;
    for (uint un = sg_global; un < units; un += sg_total) {
        float a, b;
        q8_tm_unit_rows(wb, blocks, un * Q8_TM_UNIT_ROWS, x4, lane, a, b);
        if ((lane & 19u) == 0u) {
            const uint r = un * Q8_TM_UNIT_ROWS + ((lane / 4u) % 4u) * 2u;
            y[r] = a; y[r + 1u] = b;
        }
    }
}
// The gated pair over one x: g[r] = act(gate_r . x) * (up_r . x).
template <typename XS4>
inline void mega_q8tm_gated_phase(device const uchar * wg, ulong g_off, device const uchar * wu, ulong u_off,
                                  XS4 x4, device float * g, uint n_in, uint n_out,
                                  uint sg_global, uint sg_total, uint lane) {
    const uint blocks = n_in / QK8_0, units = n_out / Q8_TM_UNIT_ROWS;
    device const uchar * gb = wg + g_off;
    device const uchar * ub = wu + u_off;
    for (uint un = sg_global; un < units; un += sg_total) {
        float ga, gbv, ua, ubv;
        q8_tm_unit_rows(gb, blocks, un * Q8_TM_UNIT_ROWS, x4, lane, ga, gbv);
        q8_tm_unit_rows(ub, blocks, un * Q8_TM_UNIT_ROWS, x4, lane, ua, ubv);
        if ((lane & 19u) == 0u) {
            const uint r = un * Q8_TM_UNIT_ROWS + ((lane / 4u) % 4u) * 2u;
            g[r] = imparo_act_f(ga) * ua; g[r + 1u] = imparo_act_f(gbv) * ubv;
        }
    }
}

// Rows of up to three tile-major matrices in one balanced stripe of units: units [0, u_a) -> a,
// [u_a, u_a + u_b) -> b, then c (a destination with 0 units is absent). The q/k/v projection.
template <typename XS4>
inline void mega_q8tm_rows3(XS4 x4, uint blocks, device const uchar * wa, device float * a, uint u_a,
                            device const uchar * wb, device float * b, uint u_b,
                            device const uchar * wc, device float * c, uint u_c,
                            uint sg_global, uint sg_total, uint lane) {
    for (uint un = sg_global; un < u_a + u_b + u_c; un += sg_total) {
        device const uchar * w; device float * dst; uint r0;
        if (un < u_a)            { w = wa; dst = a; r0 = un * Q8_TM_UNIT_ROWS; }
        else if (un < u_a + u_b) { w = wb; dst = b; r0 = (un - u_a) * Q8_TM_UNIT_ROWS; }
        else                     { w = wc; dst = c; r0 = (un - u_a - u_b) * Q8_TM_UNIT_ROWS; }
        float va, vb;
        q8_tm_unit_rows(w, blocks, r0, x4, lane, va, vb);
        if ((lane & 19u) == 0u) {
            const uint r = r0 + ((lane / 4u) % 4u) * 2u;
            dst[r] = va; dst[r + 1u] = vb;
        }
    }
}

// The conv step's row into threadgroup memory (one thread per channel; the standalone step
// kernel's channel body), then the barrier its consumer needs.
inline void mega_shortconv_row_tg(threadgroup float4 * xn, device const float * bcx, device const uchar * w_conv, ulong conv_off,
                                  device const float * state, uint width, uint kern, uint tid, uint tcount) {
    threadgroup float * xn1 = (threadgroup float *)xn;
    device const float * cw = (device const float *)(w_conv + conv_off);
    for (uint ch = tid; ch < width; ch += tcount) { xn1[ch] = shortconv_channel_out<CONV_GATED>(bcx, cw, state, width, kern, ch); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Threadgroup 0 advances the history (every threadgroup's reads of it ended at the barrier
// before this); the checkpoint copy when one is armed.
inline void mega_shortconv_shift(device const float * bcx, device const float * state_in,
                                 device float * state_out, device float * snap, bool has_snap,
                                 uint width, uint history, uint tgid, uint tid, uint tcount) {
    if (tgid == 0u) {
        for (uint ch = tid; ch < width; ch += tcount) {
            shortconv_channel_shift<CONV_GATED>(bcx, state_in, state_out, snap, has_snap, width, history, ch);
        }
    }
}

constant uint MEGA_LFM2_MIXER_NONE = 0u;       // the tail alone: o holds the mixer's output
constant uint MEGA_LFM2_MIXER_SHORTCONV = 1u;  // in_proj, the conv step, out_proj precede the tail
constant uint MEGA_LFM2_MIXER_ATTENTION = 2u;  // q/k/v, head norm + rope, the cache row, attention, o_proj

// THE LFM2 PROGRAM'S PHASES (task #158 step 3b): the same bundle as gemma4's, so the two
// architectures share one emitted frame. The entry's mixer word selects the mixer phases; it is
// an entry word, so every threadgroup of the dispatch takes the same branch and the sequence's
// barriers stay uniform.
#define L2_IS_CONV (ent.u[L2_U_MIXER] == MEGA_LFM2_MIXER_SHORTCONV)
#define L2_IS_ATTN (ent.u[L2_U_MIXER] == MEGA_LFM2_MIXER_ATTENTION)

// C0 / A0 (every threadgroup): cur = rms(x) * w_op in threadgroup memory -- the mixer's input.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_op_norm(MEGA_PH_ARGS) {
    mega_tg_load(xn, (device const float *)ent.ptr[L2_X], w4, tid, tcount);
    mega_norm_tg(xn, ent.ptr[L2_W_OP], ent.off[L2_O_OP_OFF], ent.u[L2_U_N_EMBD], ent.f[L2_F_EPS],
                 ent.u[L2_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
}
// C1: bcx = W_in . cur   (3 n_embd rows: b, c, x).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_conv_in(MEGA_PH_ARGS) {
    mega_q8tm_phase(ent.ptr[L2_W_IN], ent.off[L2_O_IN_OFF], (threadgroup const float4 *)xn, (device float *)ent.ptr[L2_BCX],
                    ent.u[L2_U_N_EMBD], 3u * ent.u[L2_U_N_EMBD], sg_global, sg_total, lane);
}
// C2 (every threadgroup): the conv step's row into threadgroup memory (the step kernel's channel
// body, one thread per channel), then o = W_out . row.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_conv_step_out(MEGA_PH_ARGS) {
    mega_shortconv_row_tg(xn, (device const float *)ent.ptr[L2_BCX], ent.ptr[L2_W_CONV], ent.off[L2_O_CONV_OFF],
                          (device const float *)ent.ptr[L2_STATE], ent.u[L2_U_N_EMBD], ent.u[L2_U_KERN], tid, tcount);
    mega_q8tm_phase(ent.ptr[L2_W_OUT], ent.off[L2_O_OUT_OFF], (threadgroup const float4 *)xn, (device float *)ent.ptr[L2_O],
                    ent.u[L2_U_N_EMBD], ent.u[L2_U_N_EMBD], sg_global, sg_total, lane);
}
// C3 (threadgroup 0): shift the history -- every threadgroup's reads of it ended at the barrier
// before this phase. A checkpoint armed at this token takes the same row (at one token the
// boundary is this token, so the history as of the boundary is the advanced history).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_conv_shift(MEGA_PH_ARGS) {
    mega_shortconv_shift((device const float *)ent.ptr[L2_BCX],
                         (device const float *)ent.ptr[L2_STATE], (device float *)ent.ptr[L2_STATE_OUT],
                         (device float *)ent.ptr[L2_SNAP],
                         ent.u[L2_U_HAS_SNAP] != 0u, ent.u[L2_U_N_EMBD], ent.u[L2_U_KERN] - 1u, tgid, tid, tcount);
}
// A1: the q/k/v rows from cur -- one tile-major unit per simdgroup, the grid's simdgroups striding
// the units (nsg a multiple of the units per head row and the stride a multiple of nsg, so a head
// row's units all sit in one threadgroup and the head phase below needs only a threadgroup barrier;
// the deep grid is smaller than the plain one, task #151).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_qkv_rows(MEGA_PH_ARGS) {
    constexpr uint UPR = HD / Q8_TM_UNIT_ROWS;
    const uint q_units = ent.u[L2_U_N_HEADS] * UPR, kv_units = ent.u[L2_U_N_KV] * UPR;
    mega_q8tm_rows3((threadgroup const float4 *)xn, ent.u[L2_U_N_EMBD] / QK8_0,
                    ent.ptr[L2_W_Q] + ent.off[L2_O_WQ_OFF], (device float *)ent.ptr[L2_QIN], q_units,
                    ent.ptr[L2_W_K] + ent.off[L2_O_WK_OFF], (device float *)ent.ptr[L2_KBUF], kv_units,
                    ent.ptr[L2_W_V] + ent.off[L2_O_WV_OFF], (device float *)ent.ptr[L2_VBUF], kv_units,
                    sg_global, sg_total, lane);
    threadgroup_barrier(mem_flags::mem_device);   // this threadgroup's units feed its own head rows
}
// A2 (this threadgroup's head rows, every pass of A1's stride; the shared head-row brick): Q:
// rms(w_qn), rope, the score scale; K: rms(w_kn), rope, then the cache row; V: the cache row. A
// threadgroup barrier between rows: the next row reuses `partial`.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_head_rows(MEGA_PH_ARGS) {
    constexpr uint UPR = HD / Q8_TM_UNIT_ROWS;
    const uint rows_per_tg = nsg / UPR;
    const uint n_rows = ent.u[L2_U_N_HEADS] + 2u * ent.u[L2_U_N_KV];
    for (uint r0 = tgid * rows_per_tg; r0 < n_rows; r0 += ent.u[L2_U_N_TG] * rows_per_tg)
    for (uint r = r0; r < min(r0 + rows_per_tg, n_rows); ++r) {
        const bool is_q = r < ent.u[L2_U_N_HEADS];
        const bool is_k = !is_q && r < ent.u[L2_U_N_HEADS] + ent.u[L2_U_N_KV];
        const uint hidx = is_q ? r : (is_k ? r - ent.u[L2_U_N_HEADS] : r - ent.u[L2_U_N_HEADS] - ent.u[L2_U_N_KV]);
        device float * row = (device float *)(is_q ? ent.ptr[L2_QIN] : (is_k ? ent.ptr[L2_KBUF] : ent.ptr[L2_VBUF])) + (ulong)hidx * HD;
        mega_head_row<HD>(is_q, is_k, hidx, row, is_q ? ent.ptr[L2_W_QN] : ent.ptr[L2_W_KN],
                          is_q ? ent.off[L2_O_QN_OFF] : ent.off[L2_O_KN_OFF], false, ent.u[L2_U_NORM_T_HEAD], ent.f[L2_F_EPS],
                          ent.u[L2_U_N_ROT], ent.f[L2_F_ROPE_BASE], freqs, 0u, tok.start_pos, ent.f[L2_F_Q_SCALE],
                          ent.u[L2_U_HAD_K], ent.u[L2_U_HAD_V], (device half *)ent.ptr[L2_KC_W], (device half *)ent.ptr[L2_VC_W],
                          ent.u[L2_U_KV_WIDTH], ent.u[L2_U_RING], (device const uint *)ent.ptr[L2_KV_PT],
                          partial, tid, tcount, lane, sgid, nsg);
        threadgroup_barrier(mem_flags::mem_device);
    }
}
// A3: attention over the cache (the shared phase brick). False = this entry cannot run on this
// pipeline; the caller reports it.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) bool l2_ph_attn(MEGA_PH_ARGS) {
    return mega_attn_phase<HD, KVW>((device float *)ent.ptr[L2_QIN], (device const half *)ent.ptr[L2_KC], (device const half *)ent.ptr[L2_VC],
                                    apart, ent.u[L2_U_N_HEADS], ent.u[L2_U_N_KV], ent.u[L2_U_KV_WIDTH], tok.start_pos, ent.u[L2_U_WINDOW],
                                    ent.u[L2_U_RING], (device const uint *)ent.ptr[L2_KV_PT], (threadgroup float *)xn, ent.u[L2_U_ATTN_BODY],
                                    ent.u[L2_U_ATTN_SPLIT], ent.u[L2_U_ATTN_HQ], ent.u[L2_U_N_TG], sync, ent.u[L2_U_SCRATCH],
                                    tok.dbg_slot, dbg_seq, dbg_rec, tgid, tid, lane, sgid, nsg);
}
// Head h's parts merged by threadgroup h.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_attn_combine(MEGA_PH_ARGS) {
    mega_attn_combine_phase<HD>(apart, (device float *)ent.ptr[L2_ATTN], ent.u[L2_U_N_HEADS], ent.u[L2_U_ATTN_SPLIT],
                                ent.u[L2_U_HAD_V], ent.u[L2_U_N_TG], dbg_rec, tgid, sgid, lane);
}
// A4: o = W_o . attn
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_oproj(MEGA_PH_ARGS) {
    mega_q8tm_phase(ent.ptr[L2_W_O], ent.off[L2_O_WO_OFF], (device const float4 *)ent.ptr[L2_ATTN], (device float *)ent.ptr[L2_O],
                    ent.u[L2_U_N_HEADS] * HD, ent.u[L2_U_N_EMBD], sg_global, sg_total, lane);
}
// P0 (every threadgroup): xn = x + o; threadgroup 0 keeps x' in u; xn = rms(xn) * w_fn.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_ffn_norm(MEGA_PH_ARGS) {
    mega_tg_sum(xn, (device const float *)ent.ptr[L2_X], (device const float *)ent.ptr[L2_O], w4, tid, tcount);
    mega_tg_keep((device float *)ent.ptr[L2_U], xn, w4, tgid, tid, tcount);
    mega_norm_tg(xn, ent.ptr[L2_W_FN], ent.off[L2_O_FN_OFF], ent.u[L2_U_N_EMBD], ent.f[L2_F_EPS],
                 ent.u[L2_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
}
// P1: g = act(gate . cur) * (up . cur), cur from threadgroup memory.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_ffn_gated(MEGA_PH_ARGS) {
    mega_q8tm_gated_phase(ent.ptr[L2_W_GATE], ent.off[L2_O_GATE_OFF], ent.ptr[L2_W_UP], ent.off[L2_O_UP_OFF],
                          (threadgroup const float4 *)xn, (device float *)ent.ptr[L2_G],
                          ent.u[L2_U_N_EMBD], ent.u[L2_U_N_FF], sg_global, sg_total, lane);
}
// P2: o = down . g   (every threadgroup finished reading o in P0, before the barrier above).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_ffn_down(MEGA_PH_ARGS) {
    mega_q8tm_phase(ent.ptr[L2_W_DOWN], ent.off[L2_O_DOWN_OFF], (device const float4 *)ent.ptr[L2_G], (device float *)ent.ptr[L2_O],
                    ent.u[L2_U_N_FF], ent.u[L2_U_N_EMBD], sg_global, sg_total, lane);
}
// P3 (threadgroup 0): x = x' + o. The next layer's block forms its own operator norm from x.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2_ph_tail(MEGA_PH_ARGS) {
    mega_dev_sum_tg0((device float *)ent.ptr[L2_X], (device const float *)ent.ptr[L2_U], (device const float *)ent.ptr[L2_O],
                     w4, tgid, tid, tcount);
}

// The LFM2 kernel is emitted from its program (task #158 step 3b): see the marker below.

// ===================================================================================
// QWEN3.8: the third architecture (task #165). 48 gated delta-net mixers and 16 full
// attention blocks over one SwiGLU tail. What is new here, and why the bricks below exist:
//
//   the weight format is PER TENSOR   one UD file gave 53 distinct per-block signatures
//                                     over 65 blocks, so a pipeline per signature is not a
//                                     design; the format is an entry WORD and the row brick
//                                     switches on it ONCE per phase, outside the row loop
//   the work item is a TM UNIT        the tile-major layout puts a unit's eight scale runs
//                                     at the unit head and its payloads after them, so ONE
//                                     base address serves all eight rows -- which is what
//                                     makes a grid-strided unit loop cheaper than eight
//                                     independent rows, and it is the same item LFM2's
//                                     mega_q8tm_phase walks for its single format

// One tile-major UNIT (TM_UNIT_ROWS rows, every K block) by one simdgroup, at a format
// known at compile time. Lanes stride the unit's 32-value sub-blocks; each lane holds the
// x values of its sub-block once and multiplies them into all eight rows, which is the
// decode GEMV's rule (rows per simdgroup give the loop independent chains and read x once).
template <uint F, typename XS>
inline void mega_blk_unit(device const uchar * wb, uint blocks, uint un, XS x,
                          thread float acc[TM_UNIT_ROWS], uint lane) {
    const uint sub_per_block = tm_block_elems_t<F>() / 32u;
    const uint bb = tm_block_bytes_t<F>();
    const uint sc_bytes = tm_scale_bytes_t<F>();
    const uint pay_bytes = bb - sc_bytes;
    const uint subs = blocks * sub_per_block;
    for (uint r = 0u; r < TM_UNIT_ROWS; ++r) { acc[r] = 0.0f; }
    for (uint sb = lane; sb < subs; sb += 32u) {
        const uint blk = sb / sub_per_block, sub = sb % sub_per_block;
        // THE UNIT BASE, once for eight rows: scales first, then payloads.
        device const uchar * ub = wb + ((ulong)un * blocks + blk) * (ulong)(TM_UNIT_ROWS * bb);
        device const uchar * pay0 = ub + (ulong)TM_UNIT_ROWS * sc_bytes;
        float xv[32];
        for (uint j = 0u; j < 32u; ++j) { xv[j] = float(x[sb * 32u + j]); }
        for (uint r = 0u; r < TM_UNIT_ROWS; ++r) {
            float v[32];
            tm_sub32_t<F, float>(ub + (ulong)r * sc_bytes, pay0 + (ulong)r * pay_bytes, sub, v, true);
            float part = 0.0f;
            for (uint j = 0u; j < 32u; ++j) { part += v[j] * xv[j]; }
            acc[r] += part;
        }
    }
}
// Rows [0, n_out) of a tile-major matrix, units strided over the grid's simdgroups. `epi`
// makes the write-back `y = act(y) * total` -- the gated pair's second half, the same
// expression act_mul computes, so the gate needs no second buffer.
template <uint F, typename XS>
inline void mega_blk_phase_t(device const uchar * w, ulong w_off, XS x, device float * y,
                             uint n_in, uint n_out, bool epi,
                             uint sg_global, uint sg_total, uint lane) {
    const uint blocks = n_in / tm_block_elems_t<F>();
    const uint units = n_out / TM_UNIT_ROWS;
    device const uchar * wb = w + w_off;
    for (uint un = sg_global; un < units; un += sg_total) {
        float acc[TM_UNIT_ROWS];
        mega_blk_unit<F, XS>(wb, blocks, un, x, acc, lane);
        for (uint r = 0u; r < TM_UNIT_ROWS; ++r) {
            const float total = simd_sum(acc[r]);
            if (lane == 0u) {
                device float * slot = y + un * TM_UNIT_ROWS + r;
                *slot = epi ? (imparo_act_f(*slot) * total) : total;
            }
        }
    }
}
// The same phase with the format from the ENTRY. The switch is uniform across the grid (an
// entry word) and sits outside every loop; the arm it picks is compiled for one format.
template <typename XS>
inline void mega_blk_phase(uint fmt, device const uchar * w, ulong w_off, XS x, device float * y,
                           uint n_in, uint n_out, bool epi,
                           uint sg_global, uint sg_total, uint lane) {
    TM_BY_FORMAT(fmt, mega_blk_phase_t<FMT, XS>(w, w_off, x, y, n_in, n_out, epi,
                                                sg_global, sg_total, lane); return;)
}

constant uint MEGA_Q35_KIND_NONE = 0u;   // the tail alone: o holds the mixer's output
constant uint MEGA_Q35_KIND_ATTN = 1u;   // full attention with a packed [query | gate] projection
constant uint MEGA_Q35_KIND_DELTA = 2u;  // the gated delta net

// THE QWEN3.8 PROGRAM'S PHASES. The tail is shared by both mixers, and it is LFM2's tail:
// x' = x + o; cur = rms(x') * w_ffn; g = act(gate . cur) * (up . cur); o = down . g;
// x = x' + o.
#define Q35_IS_ATTN  (ent.u[Q35_U_KIND] == MEGA_Q35_KIND_ATTN)
#define Q35_IS_DELTA (ent.u[Q35_U_KIND] == MEGA_Q35_KIND_DELTA)

// T0 (every threadgroup): xn = x + o; threadgroup 0 keeps x' in u; xn = rms(xn) * w_fn.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void q35_ph_ffn_norm(MEGA_PH_ARGS) {
    mega_dev_resid_norm((device float *)ent.ptr[Q35_CUR], (device float *)ent.ptr[Q35_U],
                        (device const float *)ent.ptr[Q35_X], (device const float *)ent.ptr[Q35_O],
                        ent.ptr[Q35_W_FN], ent.off[Q35_O_FN_OFF], ent.u[Q35_U_N_EMBD], ent.f[Q35_F_EPS],
                        ent.u[Q35_U_NORM_T], partial, tgid, tid, tcount, lane, sgid, nsg);
}
// T1: g = gate . cur, then g = act(g) * (up . cur). TWO passes, NOT one loop: the two
// matrices can carry DIFFERENT formats (blk.0 of one UD file is gate IQ2_XS, up IQ2_S), and
// one loop over both would need the format switch inside it. No barrier between them --
// unit `un` belongs to simdgroup `un % sg_total` in both passes, so each simdgroup reads
// back exactly the rows it wrote.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void q35_ph_ffn_gated(MEGA_PH_ARGS) {
    mega_blk_phase(ent.u[Q35_U_F_GATE], ent.ptr[Q35_W_GATE], ent.off[Q35_O_GATE_OFF],
                   (device const float *)ent.ptr[Q35_CUR], (device float *)ent.ptr[Q35_G],
                   ent.u[Q35_U_N_EMBD], ent.u[Q35_U_N_FF], false, sg_global, sg_total, lane);
    mega_blk_phase(ent.u[Q35_U_F_UP], ent.ptr[Q35_W_UP], ent.off[Q35_O_UP_OFF],
                   (device const float *)ent.ptr[Q35_CUR], (device float *)ent.ptr[Q35_G],
                   ent.u[Q35_U_N_EMBD], ent.u[Q35_U_N_FF], true, sg_global, sg_total, lane);
}
// T2: o = down . g  (every threadgroup finished reading o in T0, before the barrier above).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void q35_ph_ffn_down(MEGA_PH_ARGS) {
    mega_blk_phase(ent.u[Q35_U_F_DOWN], ent.ptr[Q35_W_DOWN], ent.off[Q35_O_DOWN_OFF],
                   (device const float *)ent.ptr[Q35_G], (device float *)ent.ptr[Q35_O],
                   ent.u[Q35_U_N_FF], ent.u[Q35_U_N_EMBD], false, sg_global, sg_total, lane);
}
// T3 (threadgroup 0): x = x' + o. The next layer forms its own operator norm from x.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void q35_ph_tail(MEGA_PH_ARGS) {
    mega_dev_sum_tg0((device float *)ent.ptr[Q35_X], (device const float *)ent.ptr[Q35_U],
                     (device const float *)ent.ptr[Q35_O], w4, tgid, tid, tcount);
}

// The qwen35 kernel is emitted from its program: see the marker below.

// ===================================================================================
// LFM2-MOE: the fourth architecture. Its ROUTED feed-forward is the part of a decode token
// the dispatch path runs as a chain of seven dependent dispatches (residual + FFN norm,
// router, route, gate and up, the activation, down, the combine), and the chain's barriers
// are 16% of the token (docs/moe-decode-attribution.md). Here it is three phases and two
// grid barriers. What is new against the other programs:
//
//   the ROUTE IS FORMED BY EVERY THREADGROUP   from the router's rows, one thread, in the
//                                              dispatch path's order -- the same inputs in
//                                              the same order give every threadgroup the
//                                              same experts, so no barrier publishes them
//   an EXPERT IS A BASE ADDRESS                a stack converts to tile-major slice by slice,
//                                              so expert e's units start at e * stride
//   THE COMBINE LIVES IN THE DOWN PHASE        an output unit sums its k experts' rows in
//                                              slot order, the combine kernel's order, and
//                                              adds the residual; no buffer of k rows exists

// Experts the route's threadgroup scratch holds probabilities for (the host refuses more).
constant uint L2M_MAX_EXPERTS = 256u;

// The route at one token into threadgroup memory: `p` = n_expert probabilities, then k ids
// (as uint bits) and k weights -- imparo_moe_route's arithmetic at n_tok = 1, and every
// threadgroup forms it from the same scores in the same order, so every threadgroup picks the
// same experts and weighs them the same.
//
// ONE SIMDGROUP, a lane per expert, for the sigmoid gate: each probability is its own
// expression, so the lane computes the serial kernel's bits. The pick is k rounds of "the
// largest remaining value, the smallest id among equals" -- the order the one-pass list gives
// (an equal value never displaces an earlier id) -- and the k-term weight sum stays on lane 0
// in slot order. The serial form was ONE thread walking the experts while the threadgroup
// waited: 12 us a routed layer, 0.26 ms a token (priced by skipping the route). Softmax keeps
// the serial walk, because a lane-parallel sum would reorder its additions.
inline void l2m_route_tg(device const float * s, threadgroup float * p, threadgroup uint * ids,
                         threadgroup float * wts, uint n_e, uint k, uint gating,
                         device const float * bias, bool has_bias, bool normalise, float scale,
                         uint tid, uint sgid, uint lane) {
    if (gating == 0u) {
        if (tid == 0u) {
            float m = -INFINITY;
            for (uint e = 0u; e < n_e; ++e) { m = max(m, s[e]); }
            float sum = 0.0f;
            for (uint e = 0u; e < n_e; ++e) { const float v = exp(s[e] - m); p[e] = v; sum += v; }
            for (uint e = 0u; e < n_e; ++e) { p[e] = p[e] / sum; }
        }
    } else if (sgid == 0u) {
        for (uint e = lane; e < n_e; e += 32u) { p[e] = 1.0f / (1.0f + exp(-s[e])); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        // THE BIAS SELECTS AND DOES NOT WEIGH, as on the dispatch path. A lane holds the
        // experts lane, lane + 32, ...; `taken` marks the ones already picked.
        uint taken = 0u;
        uint pick[MOE_MAX_K];
        for (uint j = 0u; j < k; ++j) {
            float best = -INFINITY;
            uint best_e = TOP_K_EMPTY;
            for (uint i = 0u, e = lane; e < n_e; ++i, e += 32u) {
                if ((taken >> i) & 1u) { continue; }
                const float x = p[e] + (has_bias ? bias[e] : 0.0f);
                if (best_e == TOP_K_EMPTY || x > best) { best = x; best_e = e; }
            }
            const float top = simd_max(best_e == TOP_K_EMPTY ? -INFINITY : best);
            const uint e_top = simd_min((best_e != TOP_K_EMPTY && best == top) ? best_e : TOP_K_EMPTY);
            pick[j] = e_top;
            if (best_e == e_top && e_top != TOP_K_EMPTY) { taken |= 1u << ((e_top - lane) / 32u); }
        }
        if (lane == 0u) {
            float sum = 0.0f;
            for (uint j = 0u; j < k; ++j) { sum += (pick[j] < n_e) ? p[pick[j]] : 0.0f; }
            const float denom = normalise ? max(sum, 6.103515625e-5f) : 1.0f;
            for (uint j = 0u; j < k; ++j) {
                const uint e = pick[j];
                // An empty slot (fewer experts than k, which the host refuses) weighs nothing
                // and reads expert 0's rows rather than an address past the stack.
                ids[j] = (e < n_e) ? e : 0u;
                wts[j] = (e < n_e) ? (p[e] / denom) * scale : 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One projection of the k picked experts, units strided over the grid's simdgroups: unit `un`
// is expert slot un / upe, unit un % upe of that expert's slice. `epi` makes the write-back
// y = act(y) * total, the gated pair's second half (the q35 tail's rule: unit un belongs to
// simdgroup un % sg_total in both passes, so each simdgroup reads back only what it wrote).
template <uint F>
inline void l2m_experts_t(device const uchar * w, ulong stride, threadgroup const uint * ids,
                          threadgroup const float * x, device float * y, uint n_in, uint n_ff,
                          uint k, bool epi, uint sg_global, uint sg_total, uint lane) {
    const uint blocks = n_in / tm_block_elems_t<F>();
    const uint upe = n_ff / TM_UNIT_ROWS;
    const uint units = k * upe;
    for (uint un = sg_global; un < units; un += sg_total) {
        const uint j = un / upe, lu = un - j * upe;
        float acc[TM_UNIT_ROWS];
        mega_blk_unit<F, threadgroup const float *>(w + (ulong)ids[j] * stride, blocks, lu, x, acc, lane);
        for (uint r = 0u; r < TM_UNIT_ROWS; ++r) {
            const float total = simd_sum(acc[r]);
            if (lane == 0u) {
                device float * slot = y + (ulong)j * n_ff + lu * TM_UNIT_ROWS + r;
                *slot = epi ? (imparo_act_f(*slot) * total) : total;
            }
        }
    }
}
inline void l2m_experts(uint fmt, device const uchar * w, ulong stride, threadgroup const uint * ids,
                        threadgroup const float * x, device float * y, uint n_in, uint n_ff,
                        uint k, bool epi, uint sg_global, uint sg_total, uint lane) {
    TM_BY_FORMAT(fmt, l2m_experts_t<FMT>(w, stride, ids, x, y, n_in, n_ff, k, epi,
                                         sg_global, sg_total, lane); return;)
}

// The down projection of the k picked experts and the combine: output unit `un` (8 rows of
// n_embd) sums w_j * row_j . g_j over the k experts, then x = (x + o) + s, the residual the
// dispatch path forms in two adds, in the same order.
//
// THE k EXPERTS' SUB-BLOCKS ARE ONE LANE-STRIDED SPACE, each part weighed by its expert as it
// is added: a unit's k * n_ff / 32 sub-blocks then fill every lane. One expert at a time --
// the combine kernel's order, `s += wgt * ydown` per expert -- ran n_ff = 1792 as 56
// sub-blocks, two rounds of 32 lanes with the second a quarter idle. Decode 10.635 / 10.651
// -> 10.486 / 10.469 ms a token at 5955 keys (Q4_K_M; Q4_0 10.108 -> 10.055). Not
// bit-equal to the dispatch path's combine any more: cpu_agree reads it no further from the
// CPU backend (0.157072 against 0.157228 on Q4_K_M, 0.364016 against 0.363910 on Q4_0).
//
// The activation stays in DEVICE memory, read by every lane per sub-block. Staging each
// expert's row in the threadgroup's memory first was built and measured SLOWER (the down
// phase 1.90 -> 2.08 ms a token at 5962 keys): each stage is a load-then-barrier pause the
// unit-major loop overlaps with its weight loads.
template <uint F>
inline void l2m_down_t(device const uchar * w, ulong stride, threadgroup const uint * ids,
                       threadgroup const float * wts, device const float * g, device float * x,
                       device const float * o, uint n_ff, uint n_embd, uint k,
                       uint sg_global, uint sg_total, uint lane) {
    const uint blocks = n_ff / tm_block_elems_t<F>();
    const uint units = n_embd / TM_UNIT_ROWS;
    const uint sub_per_block = tm_block_elems_t<F>() / 32u;
    const uint bb = tm_block_bytes_t<F>();
    const uint sc_bytes = tm_scale_bytes_t<F>();
    const uint pay_bytes = bb - sc_bytes;
    const uint subs = blocks * sub_per_block;
    for (uint un = sg_global; un < units; un += sg_total) {
        float sl[TM_UNIT_ROWS];
        for (uint r = 0u; r < TM_UNIT_ROWS; ++r) { sl[r] = 0.0f; }
        for (uint it = lane; it < k * subs; it += 32u) {
            const uint j = it / subs, sb = it - j * subs;
            const uint blk = sb / sub_per_block, sub = sb % sub_per_block;
            device const uchar * ub = w + (ulong)ids[j] * stride
                                    + ((ulong)un * blocks + blk) * (ulong)(TM_UNIT_ROWS * bb);
            device const uchar * pay0 = ub + (ulong)TM_UNIT_ROWS * sc_bytes;
            device const float * xs = g + (ulong)j * n_ff + sb * 32u;
            float xv[32];
            for (uint jj = 0u; jj < 32u; ++jj) { xv[jj] = xs[jj]; }
            const float wj = wts[j];
            for (uint r = 0u; r < TM_UNIT_ROWS; ++r) {
                float v[32];
                tm_sub32_t<F, float>(ub + (ulong)r * sc_bytes, pay0 + (ulong)r * pay_bytes,
                                     sub, v, true);
                float part = 0.0f;
                for (uint jj = 0u; jj < 32u; ++jj) { part += v[jj] * xv[jj]; }
                sl[r] += wj * part;
            }
        }
        float s[TM_UNIT_ROWS];
        for (uint r = 0u; r < TM_UNIT_ROWS; ++r) { s[r] = simd_sum(sl[r]); }
        if (lane == 0u) {
            for (uint r = 0u; r < TM_UNIT_ROWS; ++r) {
                const uint row = un * TM_UNIT_ROWS + r;
                x[row] = (x[row] + o[row]) + s[r];
            }
        }
    }
}
inline void l2m_down(uint fmt, device const uchar * w, ulong stride, threadgroup const uint * ids,
                     threadgroup const float * wts, device const float * g, device float * x,
                     device const float * o, uint n_ff, uint n_embd, uint k,
                     uint sg_global, uint sg_total, uint lane) {
    TM_BY_FORMAT(fmt, l2m_down_t<FMT>(w, stride, ids, wts, g, x, o, n_ff, n_embd, k,
                                      sg_global, sg_total, lane); return;)
}

// The route's threadgroup scratch, after the row: n_expert probabilities, k ids, k weights.
#define L2M_ROUTE_P   ((threadgroup float *)xn + ent.u[L2M_U_N_EMBD])
#define L2M_ROUTE_IDS ((threadgroup uint *)(L2M_ROUTE_P + ent.u[L2M_U_N_EXPERT]))
#define L2M_ROUTE_W   (L2M_ROUTE_P + ent.u[L2M_U_N_EXPERT] + MOE_MAX_K)

// R0 (every threadgroup): xn = rms(x + o) * w_fn.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2m_ph_ffn_norm(MEGA_PH_ARGS) {
    mega_tg_sum(xn, (device const float *)ent.ptr[L2M_X], (device const float *)ent.ptr[L2M_O], w4, tid, tcount);
    mega_norm_tg(xn, ent.ptr[L2M_W_FN], ent.off[L2M_O_FN_OFF], ent.u[L2M_U_N_EMBD], ent.f[L2M_F_EPS],
                 ent.u[L2M_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
}
// R1: the router's rows (F32, row-major), dealt round the threadgroups, each row split over
// ALL of its threadgroup's threads -- one float4 a thread at n_embd 2048 -- and reduced
// through `partial`. One simdgroup a row left 32 simdgroups streaming 8 KB each while the
// rest of the grid waited at the barrier: 8 us a routed layer (priced by skipping the rows).
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2m_ph_router(MEGA_PH_ARGS) {
    const uint n_e = ent.u[L2M_U_N_EXPERT];
    const uint n_tg = ent.u[L2M_U_N_TG];
    device const float4 * wr = (device const float4 *)(ent.ptr[L2M_W_ROUTER] + ent.off[L2M_O_ROUTER_OFF]);
    device float * scores = (device float *)ent.ptr[L2M_SCORES];
    for (uint r = tgid; r < n_e; r += n_tg) {
        device const float4 * row = wr + (ulong)r * w4;
        float acc = 0.0f;
        for (uint i = tid; i < w4; i += tcount) { acc += dot(row[i], xn[i]); }
        acc = simd_sum(acc);
        if (lane == 0u) { partial[sgid] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float t = 0.0f;
            for (uint g = 0u; g < nsg; ++g) { t += partial[g]; }
            scores[r] = t;
        }
        // `partial` is the next row's.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// R2 (every threadgroup): the route, then the k experts' gate and up units.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2m_ph_gated(MEGA_PH_ARGS) {
    const uint k = ent.u[L2M_U_K];
    l2m_route_tg((device const float *)ent.ptr[L2M_SCORES], L2M_ROUTE_P, L2M_ROUTE_IDS, L2M_ROUTE_W,
                 ent.u[L2M_U_N_EXPERT], k, ent.u[L2M_U_GATING],
                 (device const float *)(ent.ptr[L2M_W_BIAS] + ent.off[L2M_O_BIAS_OFF]),
                 ent.u[L2M_U_HAS_BIAS] != 0u, ent.u[L2M_U_NORMALISE] != 0u, ent.f[L2M_F_W_SCALE], tid, sgid, lane);
    threadgroup const float * xs = (threadgroup const float *)xn;
    device float * g = (device float *)ent.ptr[L2M_G];
    l2m_experts(ent.u[L2M_U_F_GATE], ent.ptr[L2M_W_GATE] + ent.off[L2M_O_GATE_OFF], ent.u[L2M_U_STRIDE_GATE],
                L2M_ROUTE_IDS, xs, g, ent.u[L2M_U_N_EMBD], ent.u[L2M_U_N_FF], k, false, sg_global, sg_total, lane);
    l2m_experts(ent.u[L2M_U_F_UP], ent.ptr[L2M_W_UP] + ent.off[L2M_O_UP_OFF], ent.u[L2M_U_STRIDE_UP],
                L2M_ROUTE_IDS, xs, g, ent.u[L2M_U_N_EMBD], ent.u[L2M_U_N_FF], k, true, sg_global, sg_total, lane);
}
// R3: the k experts' down units summed in slot order, and the residual.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2m_ph_down(MEGA_PH_ARGS) {
    l2m_down(ent.u[L2M_U_F_DOWN], ent.ptr[L2M_W_DOWN] + ent.off[L2M_O_DOWN_OFF], ent.u[L2M_U_STRIDE_DOWN],
             L2M_ROUTE_IDS, L2M_ROUTE_W, (device const float *)ent.ptr[L2M_G], (device float *)ent.ptr[L2M_X],
             (device const float *)ent.ptr[L2M_O], ent.u[L2M_U_N_FF], ent.u[L2M_U_N_EMBD], ent.u[L2M_U_K],
             sg_global, sg_total, lane);
}

// R4 (every threadgroup, after the grid barrier that ends R3): the NEXT layer's operator norm
// of the finished x -- the mega norm brick, whose reduction and scale are imparo_rms_norm's --
// and threadgroup t writes its slice of cur.
template <uint HD, uint KVW>
inline __attribute__((always_inline)) void l2m_ph_next_norm(MEGA_PH_ARGS) {
    mega_tg_load(xn, (device const float *)ent.ptr[L2M_X], w4, tid, tcount);
    mega_norm_tg(xn, ent.ptr[L2M_W_NEXT], ent.off[L2M_O_NEXT_OFF], ent.u[L2M_U_N_EMBD], ent.f[L2M_F_EPS],
                 ent.u[L2M_U_NORM_T], partial, tid, tcount, lane, sgid, nsg);
    const uint n_tg = ent.u[L2M_U_N_TG];
    const uint per = (w4 + n_tg - 1u) / n_tg;
    const uint lo = tgid * per, hi = min(lo + per, w4);
    device float4 * c4 = (device float4 *)ent.ptr[L2M_CUR];
    for (uint i = lo + tid; i < hi; i += tcount) { c4[i] = xn[i]; }
}

// The lfm2moe kernel is emitted from its program: see the marker below.

// @@MEGA_KERNELS@@
#endif
