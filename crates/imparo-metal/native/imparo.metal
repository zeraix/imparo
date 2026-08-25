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
constant uint EPI_ACT_FC [[function_constant(11)]];
constant uint EPI_ACT = is_function_constant_defined(EPI_ACT_FC) ? EPI_ACT_FC : EPI_GELU;

inline float imparo_act_f(float t) {
    return EPI_ACT == EPI_SILU ? imparo_silu_f(t) : imparo_gelu_f(t);
}

constant uint Q4_0_BYTES = 18u;

// Q8_0: 32 values per block, an f16 scale then 32 SIGNED bytes. Element i is byte 2+i.
constant uint QK8_0 = 32u;
constant uint Q8_0_BYTES = 34u;

// "this norm has no weight tensor". It used to be 0xFFFFFFFF, which collided with any
// real offset congruent to it mod 2^32 while the offset was 32 bits, and with exactly
// 4294967295 once it was widened. All-ones in 64 bits cannot be a real offset.
constant ulong IMPARO_NO_WEIGHT = ~0ul;

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
constant uint TOKEN_TILE    = 4u;

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
        // NR0 accumulators, NR0 row pointers, ONE activation read shared by all of them.
        float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        device const uchar * rp[8];
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            // Clamp rather than branch: a tail thread reads a valid row and throws the
            // result away at the store, which keeps the inner loop uniform.
            const uint rr = min(r + i, n_out - 1u);
            rp[i] = weights + w_offset + (ulong)rr * blocks * Q4_0_BYTES;
        }
        for (uint b = sub; b < blocks; b += LANES_PER_ROW) {
            const uint base4 = b * (QK4_0 / 4u);
            float d[8], part[8];
            #pragma unroll
            for (uint i = 0; i < NR0; ++i) {
                device const uchar * blk = rp[i] + b * Q4_0_BYTES;
                d[i] = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
                part[i] = 0.0f;
            }
            #pragma unroll
            for (uint g = 0; g < 4; ++g) {
                const float4 xa = xs[base4 + g];
                const float4 xb = xs[base4 + g + 4u];
                #pragma unroll
                for (uint i = 0; i < NR0; ++i) {
                    const uchar4 pk = *(device const uchar4 *)(rp[i] + b * Q4_0_BYTES
                                                               + 2u + g * 4u);
                    part[i] += dot(unpack_lo(pk), xa) + dot(unpack_hi(pk), xb);
                }
            }
            #pragma unroll
            for (uint i = 0; i < NR0; ++i) { acc[i] += part[i] * d[i]; }
        }
        #pragma unroll
        for (uint i = 0; i < NR0; ++i) {
            #pragma unroll
            for (uint off = LANES_PER_ROW / 2u; off > 0u; off >>= 1u) {
                acc[i] += simd_shuffle_down(acc[i], off);
            }
        }
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
constant uint SG_ROWS   = 64u;
constant uint SG_TOKENS = 32u;
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
template<uint NA, uint NB, uint SGX, uint SGY, bool HALF_A = false>
static void rt_gemm(
    device const uchar * weights, device const float * x, device float * y,
    ulong w_offset, uint n_in, uint n_out, uint n_tok, uint src_row,
    threadgroup float * shared, uint3 tgid, uint tid, uint tcount, uint sgid,
    uint skip, uint epilogue, device half * xh2, uint epi_half)
{
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
    // would be 33.8 and would not fit. Half is exact for Q4_0 -- (x-8)*d with d already
    // fp16 and |x-8| <= 8 only shifts the exponent.
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
    // DOUBLE BUFFERED: chunk c+1 is staged while chunk c is being multiplied. A single
    // buffer forces two barriers per chunk that serialise the two phases -- every thread
    // waits for staging to finish before any multiply starts, and again before restaging.
    // llama.cpp pays for 14 dequantisation passes over a 440-token batch where this kernel
    // pays for 7, and its whole MUL_MAT still costs what this kernel's multiplies alone do,
    // which is what overlap buys.
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
    const uint r0 = tgid.x * RT_ROWS;
    const uint t0 = tgid.y * RT_TOKENS;
    if (r0 >= n_out || t0 >= n_tok) { return; }
    const uint nrow = min(RT_ROWS, n_out - r0);
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
                    blk = weights + w_offset
                        + (ulong)(r0 + rr) * blocks * Q4_0_BYTES + (bi + bb) * Q4_0_BYTES;
                    d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
                }
                const uint k0 = bb * QK4_0;
                #pragma unroll
                for (uint g = 0; g < 4u; ++g) {
                    uchar4 quad = uchar4(0x88);
                    if (read) {
                        quad = uchar4(*(device const packed_uchar4 *)(blk + 2u + g * 4u));
                    }
                    const float4 lo4 = (float4(quad & uchar4(0x0F)) - 8.0f) * d;
                    const float4 hi4 = (float4(quad >> 4)           - 8.0f) * d;
                    #pragma unroll
                    for (uint j = 0; j < 4u; ++j) {
                        const uint kk = k0 + g * 4u + j;
                        dst[kk * RT_WS + rr]         = half(lo4[j]);
                        dst[(kk + 16u) * RT_WS + rr] = half(hi4[j]);
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
                if (k + BPF * 8u < KSTEPS) {
                    #pragma unroll
                    for (uint j = 0; j < NB; ++j) {
                        simdgroup_load(b[BPF][j],
                                       cur + (k + BPF * 8u) * RT_WS + rx + j * 8u, RT_WS);
                    }
                }
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
    // Rows never straddle the end: n_out is a multiple of 64 for every matmul in this
    // model. Tokens can, so the caller rounds the activation buffers up to a whole token
    // tile; rows past the token count receive values nothing reads.
    if (epilogue == 0u) {
        #pragma unroll
        for (uint i = 0; i < NA; ++i) {
            #pragma unroll
            for (uint j = 0; j < NB; ++j) {
                simdgroup_store(mc[i * NB + j],
                                y + (ulong)(t0 + ty + i * 8u) * n_out + r0 + rx + j * 8u,
                                n_out);
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
    #pragma unroll
    for (uint i = 0; i < NA; ++i) {
        #pragma unroll
        for (uint j = 0; j < NB; ++j) {
            simdgroup_store(mc[i * NB + j], ob + (ty + i * 8u) * OB_S + rx + j * 8u, OB_S);
        }
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
#define IMPARO_RT_KERNEL(NAME, NA, NB, SGX, SGY)                                          \
kernel void NAME(                                                                          \
    device const uchar * weights [[buffer(0)]],                                            \
    device const float * x       [[buffer(1)]],                                            \
    device float       * y       [[buffer(2)]],                                            \
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in  [[buffer(4)]],           \
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],           \
    constant uint & src_row  [[buffer(7)]], constant uint & skip [[buffer(12)]],           \
    constant uint & epilogue [[buffer(13)]],                                               \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],               \
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  sgid  [[simdgroup_index_in_threadgroup]])                                        \
{                                                                                          \
    rt_gemm<NA, NB, SGX, SGY>(weights, x, y, w_offset, n_in, n_out, n_tok, src_row,        \
                              shared, tgid, tid3.x, tcnt3.x, sgid, skip, epilogue,        \
                              xh2, epi_half);                                             \
}

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
#define IMPARO_RT_KERNEL_H(NAME, NA, NB, SGX, SGY)                                         \
kernel void NAME(                                                                          \
    device const uchar * weights [[buffer(0)]],                                            \
    device const float * x       [[buffer(1)]],                                            \
    device float       * y       [[buffer(2)]],                                            \
    constant ulong & w_offset [[buffer(3)]], constant uint & n_in  [[buffer(4)]],           \
    constant uint & n_out    [[buffer(5)]], constant uint & n_tok [[buffer(6)]],           \
    constant uint & src_row  [[buffer(7)]], constant uint & skip [[buffer(12)]],           \
    constant uint & epilogue [[buffer(13)]],                                               \
    device half * xh2 [[buffer(8)]], constant uint & epi_half [[buffer(9)]],               \
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  sgid  [[simdgroup_index_in_threadgroup]])                                        \
{                                                                                          \
    rt_gemm<NA, NB, SGX, SGY, true>(weights, x, y, w_offset, n_in, n_out, n_tok, src_row,  \
                              shared, tgid, tid3.x, tcnt3.x, sgid, skip, epilogue,         \
                              xh2, epi_half);                                              \
}

IMPARO_RT_KERNEL_H(imparo_rt_0_h, 2, 4, 2, 2)
IMPARO_RT_KERNEL_H(imparo_rt_1_h, 4, 4, 2, 2)
IMPARO_RT_KERNEL_H(imparo_rt_2_h, 2, 4, 2, 4)
IMPARO_RT_KERNEL_H(imparo_rt_3_h, 2, 4, 4, 2)
IMPARO_RT_KERNEL_H(imparo_rt_4_h, 4, 2, 4, 2)
IMPARO_RT_KERNEL_H(imparo_rt_5_h, 2, 2, 4, 4)
IMPARO_RT_KERNEL_H(imparo_rt_6_h, 4, 4, 2, 4)
IMPARO_RT_KERNEL_H(imparo_rt_7_h, 8, 2, 2, 2)

// Narrow-N tile for 2..15-token batches (task #11, modeled on the fork's
// kernel_mul_mm_nb8, author liuliquan): 64 rows x 8 tokens, 2 simdgroups, 64
// threads, 4 accumulators each. The weight stage (RT_K x 66 halves) is the same
// 8.4 KB whatever the token width, so cost is FLAT in n_tok while a 64-token
// tile wastes up to 56/64 of its arithmetic on a narrow batch. Same k-order into
// every output as any other shape, so routing through it is BIT-IDENTICAL.
// This is also the #5 story's end state: with 2..15 covered here, no multi-token
// batch can reach the GEMV in shipping (the wobble's only reachable expression).
IMPARO_RT_KERNEL(imparo_rt_nb8, 1, 4, 2, 1)
IMPARO_RT_KERNEL_H(imparo_rt_nb8_h, 1, 4, 2, 1)
// Variant b: the same 64x8 tile on ONE simdgroup (32 threads, 8 accumulators).
// Which occupancy shape wins is a device property, so it is a tuned knob.
IMPARO_RT_KERNEL(imparo_rt_nb8b, 1, 8, 1, 1)
IMPARO_RT_KERNEL_H(imparo_rt_nb8b_h, 1, 8, 1, 1)


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
    if (n_tok != 1u) { return; }
    const uint r0 = tgid.x * Q8_DECODE_ROWS;
    if (r0 >= n_out) { return; }
    const uint blocks = n_in / QK8_0;
    const uint ix = lane / 4u;
    const uint il = lane % 4u;
    const uint ib0 = sgid * 8u + ix;
    device const float * xb = x + (ulong)src_row * n_in;

    device const uchar * rows[4];
    #pragma unroll
    for (uint rr = 0; rr < Q8_DECODE_ROWS; ++rr) {
        const uint row = min(r0 + rr, n_out - 1u);
        rows[rr] = weights + w_offset + (ulong)row * blocks * Q8_0_BYTES;
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
    device const uchar * row = weights + w_offset + (ulong)r * blocks * Q8_0_BYTES;
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
        device const uchar * blk = row + (ulong)b * Q8_0_BYTES;
        const float d = float(as_type<half>((ushort)(blk[0] | (blk[1] << 8))));
        const float wv = float(as_type<char>(blk[2u + lane])) * d;
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
template<uint ROWS, uint TOKENS, uint NSG, uint SGR, bool FULL = false,
         bool HALF_A = false>
static void q8_0_gemm(
    device const uchar * weights, device const uchar * x_bytes, device float * y,
    uint n_in, uint n_out, uint n_tok, uint src_row, uint epilogue,
    threadgroup half * shared, uint3 tgid, uint tid, uint sgid)
{
    static_assert(ROWS > 0u && (ROWS % 8u) == 0u,
                  "Q8 GEMM rows must be a non-zero multiple of one MMA tile");
    static_assert(TOKENS > 0u && (TOKENS % 8u) == 0u,
                  "Q8 GEMM tokens must be a non-zero multiple of one MMA tile");
    static_assert(NSG > 0u && SGR > 0u && (NSG % SGR) == 0u,
                  "Q8 GEMM simdgroups must split evenly across rows and tokens");
    static_assert(((ROWS / 8u) % SGR) == 0u,
                  "Q8 GEMM row tiles must split evenly across row simdgroups");
    static_assert(((TOKENS / 8u) % (NSG / SGR)) == 0u,
                  "Q8 GEMM token tiles must split evenly across token simdgroups");
    constexpr uint K = QK8_0;
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
    const uint r0 = row_group * ROWS, t0 = token_group * TOKENS;
    if (!FULL && (r0 >= n_out || t0 >= n_tok)) { return; }
    const uint nrow = FULL ? ROWS : min(ROWS, n_out - r0);
    const uint ntok = FULL ? TOKENS : min(TOKENS, n_tok - t0);
    const uint row_tile0 = (sgid % SGR) * ROW_TILES_PER_SG;
    const uint token_tile0 = (sgid / SGR) * TOKEN_TILES_PER_SG;
    const uint blocks = n_in / QK8_0;
    threadgroup half * ws = shared;
    threadgroup half * as = shared + K * ROWS;

    simdgroup_float8x8 mc[ACC];
    #pragma unroll
    for (uint i = 0; i < ACC; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint c00 = 0; c00 < n_in; c00 += K) {
        const uint bi = c00 / QK8_0;
        // Two threads dequantise one row: each handles sixteen contiguous signed bytes.
        #pragma unroll
        for (uint e = tid; e < ROWS * 2u; e += THREADS_PER_TG) {
            const uint rr = e / 2u, h = e % 2u;
            const bool live = FULL || rr < nrow;
            device const uchar * blk = weights;
            float d = 0.0f;
            if (live) {
                blk = weights + (ulong)(r0 + rr) * blocks * Q8_0_BYTES
                    + (ulong)bi * Q8_0_BYTES;
                const ushort scale_bits = Q8_TYPED_SCALE
                    ? *(device const ushort *)blk
                    : (ushort)(blk[0] | (blk[1] << 8));
                d = float(as_type<half>(scale_bits));
            }
            #pragma unroll
            for (uint g4 = 0; g4 < 4u; ++g4) {
                char4 q = char4(0);
                if (live) {
                    q = char4(*(device const packed_char4 *)(blk + 2u + h * 16u + g4 * 4u));
                }
                const half4 v = half4(float4(q) * d);
                #pragma unroll
                for (uint j = 0; j < 4u; ++j) {
                    const uint k = h * 16u + g4 * 4u + j;
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
        for (uint e = tid; e < TOKENS * K_TILES; e += THREADS_PER_TG) {
            const uint tt = e / K_TILES, kt = e % K_TILES;
            // K-major activation tiles keep the two token operands consumed by one
            // simdgroup adjacent for each K step. This is only a threadgroup-memory
            // transpose: every output keeps the same K/MMA accumulation order.
            const uint tile = kt * TOKEN_TILES + tt / 8u;
            threadgroup half4 * dst4 =
                (threadgroup half4 *)(as + tile * 64u + (tt & 7u) * 8u);
            if (FULL || tt < ntok) {
                const ulong element = (ulong)(src_row + t0 + tt) * n_in
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
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < K; kk += 8u) {
            // A simdgroup owns a rectangular set of output tiles. Load each of its row
            // and token operands once, then form their outer product. A flat tile loop
            // loads two matrices for every output tile -- 16 loads for the 64x32/4-SG
            // shape -- where this is the same per-output K order with four weight and
            // two activation loads.
            simdgroup_half8x8 wm[ROW_TILES_PER_SG];
            simdgroup_half8x8 am[TOKEN_TILES_PER_SG];
            #pragma unroll
            for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
                const uint tile = (kk / 8u) * ROW_TILES + row_tile0 + ri;
                simdgroup_load(wm[ri], ws + tile * 64u, 8u);
            }
            #pragma unroll
            for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
                const uint tile = (kk / 8u) * TOKEN_TILES + token_tile0 + ti;
                simdgroup_load(am[ti], as + tile * 64u, 8u);
            }
            #pragma unroll
            for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
                #pragma unroll
                for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
                    const uint ai = ti * ROW_TILES_PER_SG + ri;
                    simdgroup_multiply_accumulate(mc[ai], am[ti], wm[ri], mc[ai]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Full tiles need no masked epilogue: write each cooperative result matrix straight
    // to device memory. An unconditional shared-memory spill would add another barrier
    // and a scalar copy to every Q8 projection. Partial rows/tokens and the fused gated
    // epilogue take the masked path below.
    if (FULL || (epilogue == 0u && nrow == ROWS && ntok == TOKENS)) {
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

    // The operand stage is dead now, so its bytes become a masked float output tile.
    threadgroup float * out = (threadgroup float *)shared;
    #pragma unroll
    for (uint ti = 0; ti < TOKEN_TILES_PER_SG; ++ti) {
        #pragma unroll
        for (uint ri = 0; ri < ROW_TILES_PER_SG; ++ri) {
            const uint ai = ti * ROW_TILES_PER_SG + ri;
            const uint tt = (token_tile0 + ti) * 8u;
            const uint rr = (row_tile0 + ri) * 8u;
            simdgroup_store(mc[ai], out + tt * ROWS + rr, ROWS);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < ntok * nrow; e += THREADS_PER_TG) {
        const uint tt = e / nrow, rr = e % nrow;
        device float * slot = y + (ulong)(t0 + tt) * n_out + r0 + rr;
        const float v = out[tt * ROWS + rr];
        *slot = epilogue ? (imparo_act_f(*slot) * v) : v;
    }
}

// The launch contract is checked in the wrapper, before any barrier: a threadgroup that
// is not exactly NSG*32 wide would desynchronise the staging loops.
#define IMPARO_Q8_GEMM(NAME, ROWS, TOKENS, NSG, SGR, FULL)                                \
kernel void NAME(                                                                         \
    device const uchar * weights [[buffer(0)]], device const float * x [[buffer(1)]],     \
    device float * y [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],              \
    constant uint & n_in [[buffer(4)]], constant uint & n_out [[buffer(5)]],              \
    constant uint & n_tok [[buffer(6)]], constant uint & src_row [[buffer(7)]],           \
    constant uint & epilogue [[buffer(13)]],                                              \
    threadgroup half * shared [[threadgroup(0)]],                                         \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint3 tcnt3 [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) \
{                                                                                         \
    if (tcnt3.x != NSG * 32u || tcnt3.y != 1u || tcnt3.z != 1u) { return; }                \
    q8_0_gemm<ROWS, TOKENS, NSG, SGR, FULL, false>(                                       \
        weights + w_offset, (device const uchar *)x, y, n_in, n_out, n_tok, src_row,       \
        epilogue, shared, tgid, tid, sgid);                                                \
}

#define IMPARO_Q8_GEMM_HALF(NAME, ROWS, TOKENS, NSG, SGR, FULL)                           \
kernel void NAME(                                                                         \
    device const uchar * weights [[buffer(0)]], device const half * x [[buffer(1)]],      \
    device float * y [[buffer(2)]], constant ulong & w_offset [[buffer(3)]],              \
    constant uint & n_in [[buffer(4)]], constant uint & n_out [[buffer(5)]],              \
    constant uint & n_tok [[buffer(6)]], constant uint & src_row [[buffer(7)]],           \
    constant uint & epilogue [[buffer(13)]],                                              \
    threadgroup half * shared [[threadgroup(0)]],                                         \
    uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint3 tcnt3 [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) \
{                                                                                         \
    if (tcnt3.x != NSG * 32u || tcnt3.y != 1u || tcnt3.z != 1u) { return; }                \
    q8_0_gemm<ROWS, TOKENS, NSG, SGR, FULL, true>(                                        \
        weights + w_offset, (device const uchar *)x, y, n_in, n_out, n_tok, src_row,       \
        epilogue, shared, tgid, tid, sgid);                                                \
}

// The shape table. Keep it in step with Q8_SHAPES in the bridge -- the host sizes the
// grid and the threadgroup from that table, and a disagreement is a wrong-size launch,
// not a slow one. Shape 9 (128x32) is retained because the LFM branch measured it exact
// and slower, and a rejected candidate that is still reachable is evidence; the host
// builds only the selected shapes.
IMPARO_Q8_GEMM(imparo_q8_gemm_0,      32, 16,  4, 2, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_0_full, 32, 16,  4, 2, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_1,      32, 32,  4, 2, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_1_full, 32, 32,  4, 2, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_2,      64, 16,  4, 2, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_2_full, 64, 16,  4, 2, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_3,      64, 32,  4, 2, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_3_full, 64, 32,  4, 2, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_4,      64, 32,  8, 4, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_4_full, 64, 32,  8, 4, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_5,      64, 32, 16, 4, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_5_full, 64, 32, 16, 4, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_6,      64, 64,  8, 4, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_6_full, 64, 64,  8, 4, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_7,      64, 64,  4, 2, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_7_full, 64, 64,  4, 2, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_8,      32,128,  4, 2, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_8_full, 32,128,  4, 2, true)
IMPARO_Q8_GEMM(imparo_q8_gemm_9,     128, 32,  4, 4, false)
IMPARO_Q8_GEMM(imparo_q8_gemm_9_full,128, 32,  4, 4, true)

IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_0,      32, 16,  4, 2, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_0_full, 32, 16,  4, 2, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_1,      32, 32,  4, 2, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_1_full, 32, 32,  4, 2, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_2,      64, 16,  4, 2, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_2_full, 64, 16,  4, 2, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_3,      64, 32,  4, 2, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_3_full, 64, 32,  4, 2, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_4,      64, 32,  8, 4, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_4_full, 64, 32,  8, 4, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_5,      64, 32, 16, 4, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_5_full, 64, 32, 16, 4, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_6,      64, 64,  8, 4, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_6_full, 64, 64,  8, 4, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_7,      64, 64,  4, 2, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_7_full, 64, 64,  4, 2, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_8,      32,128,  4, 2, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_8_full, 32,128,  4, 2, true)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_9,     128, 32,  4, 4, false)
IMPARO_Q8_GEMM_HALF(imparo_q8_gemm_h_9_full,128, 32,  4, 4, true)

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
kernel void imparo_rms_norm(
    device const uchar * weights [[buffer(0)]],
    device float       * x       [[buffer(1)]],
    constant ulong & w_offset [[buffer(2)]], constant uint & width [[buffer(3)]],
    constant float & eps     [[buffer(4)]], constant uint & n_row [[buffer(5)]],
    constant uint & row_stride [[buffer(6)]], constant uint & base_off [[buffer(7)]],
    device const float * addend [[buffer(8)]], constant uint & has_add [[buffer(9)]],
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
    const bool vec4 = (width % 4u) == 0u && ((base_off + r * row_stride) % 4u) == 0u;
    float sq = 0.0f;
    if (vec4) {
        device const float4 * s4 = (device const float4 *)srow;
        const uint w4 = width / 4u;
        for (uint i = tid; i < w4; i += tcount) { sq += dot(s4[i], s4[i]); }
    } else {
        for (uint i = tid; i < width; i += tcount) { sq += srow[i] * srow[i]; }
    }
    sq = simd_sum(sq);

    float inv;
    if (nsg == 1u) {
        // One simdgroup needs NO threadgroup barrier: simd_sum has already reduced across
        // every thread, and the result is broadcast in a register. At decode this kernel is
        // pure latency -- 40 KB of traffic, ~0.4 us of work, 5.8 us measured -- and the two
        // barriers are most of the difference. 294 of these run per token.
        inv = rsqrt(simd_broadcast_first(sq) / float(width) + eps);
    } else {
        if (lane == 0) { partial[sgid] = sq; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Second stage by simd_sum, not a serial loop on thread 0.
        //
        // The thread count is now derived from the row width, so a 2560-wide row runs 640
        // threads = 20 simdgroups. Summing 20 partials one at a time on a single thread put
        // a 20-iteration serial loop on the critical path with 639 threads idle behind it.
        if (sgid == 0) {
            const float v = (lane < nsg) ? partial[lane] : 0.0f;
            const float total = simd_sum(v);
            if (lane == 0) { partial[0] = rsqrt(total / float(width) + eps); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv = partial[0];
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
    if (has_add) {
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
    const float ff = (n_freqs > 0u && i < n_freqs) ? freqs[i] : 1.0f;
    const float inv = pow(base, -2.0f * float(i) / float(n_rot)) / ff;
    const float theta = float(start_pos + t) * inv;
    const float c = cos(theta), s = sin(theta);
    const float x0 = head[i], x1 = head[i + half_rot];
    head[i]            = x0 * c - x1 * s;
    head[i + half_rot] = x0 * s + x1 * c;
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
constant bool KV_PAGED_FC [[function_constant(5)]];
constant bool KV_PAGED = is_function_constant_defined(KV_PAGED_FC) ? KV_PAGED_FC : true;

inline uint kv_slot(uint gp, uint ring_mask, device const uint * pt) {
    if (ring_mask > 0u) { return gp & ring_mask; }
    return KV_PAGED ? ((pt[gp >> 6] << 6) | (gp & 63u)) : gp;
}
// An n-run starting at gp0 reads contiguous cache bytes unless it straddles the
// ring wrap (windowed) or a non-adjacent page boundary (paged full layers). With
// an identity table adjacent pages ARE adjacent, so stage one keeps every fast
// path it had.
// WHOLE-RANGE wrap test: does [gp0, gp0+n) straddle the ring seam? Ring semantics
// only. On paged full layers this must stay FALSE: their 8-runs are 8-aligned and a
// 64-cell page cannot be straddled by an aligned 8-run, so nothing is ever skipped
// to the tail -- and routing whole tiles to the device-spill tail makes threadgroups
// race on its scratch (the collision its indexing comment warns about; found the
// hard way by the pair-swap placement test).
inline bool kv_range_wraps(uint gp0, uint n, uint ring_mask) {
    return ring_mask > 0u && (gp0 & ring_mask) + n > ring_mask + 1u;
}
inline bool kv_run_breaks(uint gp0, uint n, uint ring_mask, device const uint * pt) {
    if (ring_mask > 0u) { return (gp0 & ring_mask) + n > ring_mask + 1u; }
    if (!KV_PAGED) { return false; } // identity: full-layer runs are contiguous
    if ((gp0 & 63u) + n <= 64u) { return false; }
    return pt[gp0 >> 6] + 1u != pt[(gp0 + n - 1u) >> 6];
}

kernel void imparo_kv_store(
    device const float * src [[buffer(0)]],
    device half        * cache [[buffer(1)]],
    constant uint & width [[buffer(2)]], constant uint & start_pos [[buffer(3)]],
    constant uint & n_tok [[buffer(4)]], constant uint & ring_mask [[buffer(5)]],
    device const uint * pt [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Four values per thread. `width` is n_kv_heads * head_dim, a multiple of four for
    // every layer in this model, and both buffers start page aligned. The scalar tail
    // covers anything that does not divide.
    const uint w4 = width / 4u;
    const uint i = gid.x, t = gid.y;
    if (t >= n_tok) { return; }
    const uint pos = start_pos + t;
    const uint slot = kv_slot(pos, ring_mask, pt);
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
    for (uint r = 0u; r < reps; ++r) {
        for (uint i = gid; i < n4; i += gsz) { acc += src[i]; }
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
template<uint QT, uint HD_C, uint BLK, uint PT>
static void attention_prefill_qtile_body(
    device const float * q, device const half * kc, device const half * vc,
    device float * out, uint head_dim, uint n_heads, uint n_kv, uint kv_width,
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
        for (uint qi = sgid; qi < QT; qi += nsg) {
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
        const uint tail_gp0 = lo + p0;
        const bool tail_wrap = kv_range_wraps(tail_gp0, np, ring_mask);
        if (full_np < np || tail_wrap) {
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

#define IMPARO_QTILE_KERNEL(NAME, QT_N, HD_C, BLK_N, PT_N)                                                    \
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
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  lane  [[thread_index_in_simdgroup]],                                             \
    uint  sgid  [[simdgroup_index_in_threadgroup]],                                        \
    uint  nsg   [[simdgroups_per_threadgroup]])                                            \
{                                                                                          \
    attention_prefill_qtile_body<QT_N, HD_C, BLK_N, PT_N>(q, kc, vc, out, hd, n_heads, n_kv, kv_width,    \
                               start_pos, window, ring_mask, pt, n_tok, stage, shared, tgid,   \
                               tid3.x, tcnt3.x, lane, sgid, nsg);                          \
}

IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile, 8, 0, 4, 128)
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16, 16, 0, 4, 128)
// One pipeline per blocking depth, the way rt_shape has one per shape. The ceiling is the
// register file -- a DEVICE property -- and how much is needed depends on QT and head_dim,
// which are MODEL properties, so neither alone decides it and it has to be benched on the
// pair. 4 measures 496 tok/s here, 8 measures 434; 2 has never been tried.
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16h2, 16, 256, 2, 128)
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16h, 16, 256, 4, 128)
IMPARO_QTILE_KERNEL(imparo_attention_prefill_qtile16h8, 16, 256, 8, 128)

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
template<uint QT, uint HD_C, uint BLK, uint PT, uint NSG, bool DSPILL, bool HALF_Q = false,
         bool KT = false>
static void attention_prefill_qcomb_body(
    device const float * q, device const half * kc, device const half * vc,
    device float * out, device float * dspill,
    device const half * kt, uint kt_stride,
    uint head_dim, uint n_heads, uint n_kv, uint kv_width,
    uint start_pos, uint window, uint ring_mask, device const uint * pt,
    uint n_tok, uint stage,
    device half * xh, uint xh_on,
    threadgroup float * shared, uint3 tgid, uint tid, uint tcount,
    uint lane, uint sgid, uint nsg)
{
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
                #pragma unroll
                for (uint pb = 0; pb < PB; ++pb) {
                    const uint gp = lo + p0 + sp8b + pb * 8u;
                    const uint ps = kv_slot(gp, ring_mask, pt);
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
        for (uint sp8 = 0; sp8 < full_np; sp8 += 8u) {
            const uint gp = lo + p0 + sp8;
            if (kv_run_breaks(gp, 8u, ring_mask, pt)) { continue; }
            const uint ps = kv_slot(gp, ring_mask, pt);
            device const half * vrow = vc + (ulong)ps * kv_width + kvh * hd;
            simdgroup_float8x8 pa[QROWS];
            #pragma unroll
            for (uint g = 0; g < QROWS; ++g) {
                simdgroup_load(pa[g], sc + (ulong)(g * 8u) * PT + sp8, PT);
            }
            #pragma unroll
            for (uint k = 0; k < NDB; ++k) {
                simdgroup_half8x8 vb;
                simdgroup_load(vb, vrow + (db0 + k) * 8u, kv_width);
                #pragma unroll
                for (uint g = 0; g < QROWS; ++g) {
                    simdgroup_multiply_accumulate(o[g][k], pa[g], vb, o[g][k]);
                }
            }
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

#define IMPARO_QCOMB_KERNEL(NAME, QT_N, HD_N, BLK_N, PT_N, NSG_N, DS_N, ...)  \
    IMPARO_QCOMB_KERNEL_KT(NAME, QT_N, HD_N, BLK_N, PT_N, NSG_N, DS_N, ##__VA_ARGS__)
#define IMPARO_QCOMB_KERNEL_KT(NAME, QT_N, HD_N, BLK_N, PT_N, NSG_N, DS_N, ...)                       \
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
    threadgroup float * shared [[threadgroup(0)]],                                         \
    uint3 tgid  [[threadgroup_position_in_grid]],                                          \
    uint3 tid3  [[thread_position_in_threadgroup]],                                        \
    uint3 tcnt3 [[threads_per_threadgroup]],                                               \
    uint  lane  [[thread_index_in_simdgroup]],                                             \
    uint  sgid  [[simdgroup_index_in_threadgroup]],                                        \
    uint  nsg   [[simdgroups_per_threadgroup]])                                            \
{                                                                                          \
    attention_prefill_qcomb_body<QT_N, HD_N, BLK_N, PT_N, NSG_N, DS_N, ##__VA_ARGS__>(            \
                               q, kc, vc, out, dspill, ktb, kt_stride,                     \
                               hd, n_heads, n_kv,                                          \
                               kv_width, start_pos, window, ring_mask, pt, n_tok, stage,   \
                               xh, xh_on,                                                  \
                               shared, tgid, tid3.x, tcnt3.x, lane, sgid, nsg);            \
}

IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb256, 8, 256, 4, 128, 8, false)
// BLK 2 variants of the QT-8 kernels. Their score phase splits a PT-128 position tile into
// ceil(PT / (8*BLK)) work units and hands one to each simdgroup, so at BLK 4 that is 4
// units over NSG 8 -- HALF THE SIMDGROUPS IDLE. At BLK 2 it is 8 units over 8. The trade is
// arithmetic intensity: (QROWS + BLK) / (QROWS * BLK) with QROWS 1 goes from 1.25 loads per
// MAC to 1.50, against double the parallelism. BLK also does not touch threadgroup memory
// (only QT and PT do) and needs FEWER accumulator registers, so the wider occupancy costs
// nothing structural. These are the kernels q4 and q8 prefill run on.
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb256b2, 8, 256, 2, 128, 8, false)
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
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb256x, 16, 256, IMPARO_BLK_256X, IMPARO_PT_256X,
                    IMPARO_NSG_256X, false, true)
// QT 16 was tried at head size 256 (the WINDOW layers, 35 of 42, whose
// accumulator is half the hd-512 one so the wider tile fits): 32472 ms against
// 32672 on a 16k prefill, 0.6%. It stages Q as half to fit the budget, so it
// moves the logits -- not worth regenerating pins and giving up numeric ground
// for 0.6%.
// head_dim 512 (the full-attention layers): the tail scratch moves to device memory.
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb512, 8, 512, 4, 128, 8, true)
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb512b2, 8, 512, 2, 128, 8, true)
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
IMPARO_QCOMB_KERNEL(imparo_attention_prefill_qcomb512x, 16, 512, IMPARO_BLK_512X, IMPARO_PT_512X,
                    IMPARO_NSG_512X, true, true)
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
        const ulong prow = KV_PAGED ? (ulong)((pt[gp >> 6] << 6) | (gp & 63u))
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


// Merge the split-KV slices' unnormalised partials onto a common maximum and normalise.
kernel void imparo_attention_decode_combine(
    device const float * part [[buffer(0)]], device float * out [[buffer(1)]],
    constant uint & head_dim [[buffer(2)]], constant uint & n_heads [[buffer(3)]],
    constant uint & slices   [[buffer(4)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid3  [[thread_position_in_threadgroup]],
    uint3 tcnt3 [[threads_per_threadgroup]])
{
    const uint tid = tid3.x, tcount = tcnt3.x;
    const uint h = tgid.x, t = tgid.y;
    device const float * pb = part + (ulong)(t * n_heads + h) * slices * (head_dim + 2u);

    float mx = -INFINITY;
    for (uint sp = 0; sp < slices; ++sp) { mx = max(mx, pb[sp * (head_dim + 2u) + head_dim]); }
    float denom = 0.0f;
    for (uint sp = 0; sp < slices; ++sp) {
        const float m = pb[sp * (head_dim + 2u) + head_dim];
        denom += exp(m - mx) * pb[sp * (head_dim + 2u) + head_dim + 1u];
    }
    const float inv = 1.0f / denom;

    device float * o = out + ((ulong)t * n_heads + h) * head_dim;
    for (uint i = tid; i < head_dim; i += tcount) {
        float acc = 0.0f;
        for (uint sp = 0; sp < slices; ++sp) {
            const float m = pb[sp * (head_dim + 2u) + head_dim];
            acc += exp(m - mx) * pb[sp * (head_dim + 2u) + i];
        }
        o[i] = acc * inv;
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
// LFM2's gated short convolution. `imparo_cpu::ops::shortconv` is the oracle this is
// checked against, and carries the derivation:
//
//   bx     = b * x                        elementwise, per token per channel
//   seq    = state ++ bx                  causal: the state is PREPENDED
//   out[t] = c[t] * sum_k conv_w[ch][k] * seq[t + k]
//   state' = the last (kernel - 1) values of seq
//
// TWO kernels, because the new state is the TAIL of the same sequence the outputs read.
// One dispatch would have threads writing state slots other threads still need, and a
// kernel cannot barrier its own grid -- the dispatch boundary is the barrier.
//
// `conv_w` is channel-major with the tap fastest: element (k, ch) is at ch * kernel + k,
// which is what the GGUF dims (l_cache, n_embd) mean. Tap 0 multiplies the OLDEST value.
kernel void imparo_shortconv(
    device const float * bcx   [[buffer(0)]],
    device const float * cw    [[buffer(1)]],
    device const float * state [[buffer(2)]],
    device float * out         [[buffer(3)]],
    constant uint & width      [[buffer(4)]],
    constant uint & kern       [[buffer(5)]],
    constant uint & n_tok      [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_tok * width) { return; }
    const uint t = gid / width, ch = gid % width;
    const uint history = kern - 1u;
    // Accumulated in tap order, oldest first, the same order the CPU reference uses.
    float acc = 0.0f;
    for (uint k = 0; k < kern; ++k) {
        const uint e = t + k;
        float v;
        if (e < history) {
            v = state[e * width + ch];
        } else {
            const ulong row = (ulong)(e - history) * 3u * width;
            v = bcx[row + ch] * bcx[row + 2u * width + ch];
        }
        acc += cw[ch * kern + k] * v;
    }
    out[(ulong)t * width + ch] = bcx[(ulong)t * 3u * width + width + ch] * acc;
}

// The tail of `seq` becomes the new state. ONE THREAD PER CHANNEL, reading every value it
// needs before writing any: for a batch SHORTER than the history the new state is a shift
// of the old one, and a thread that wrote first would clobber a slot it still has to read.
// One thread per channel makes that a register question instead of a cross-thread one.
constant uint SHORTCONV_MAX_HISTORY = 8u;

// SEPARATE in and out, which is what lets the same kernel serve two jobs:
//
//   advance    state_in == state_out, n_tok = the whole chunk  -> the live state moves on
//   snapshot   state_out is a scratch buffer, n_tok = tokens up to a BOUNDARY inside the
//              chunk -> the state as of that boundary, without stopping the batch there
//
// The second is why the batch does not have to be cut at a unit boundary. The state at
// any position is just the last (kernel - 1) values of b*x ending there, and those are in
// `bcx` already -- so it is computed, not stood on. The reference reaches the same place
// from the other side: its scan kernel writes rollback planes as it goes rather than
// having the caller stop.
kernel void imparo_shortconv_state(
    device const float * bcx       [[buffer(0)]],
    device const float * state_in  [[buffer(1)]],
    device float * state_out       [[buffer(2)]],
    constant uint & width          [[buffer(3)]],
    constant uint & kern           [[buffer(4)]],
    constant uint & n_tok          [[buffer(5)]],
    uint ch [[thread_position_in_grid]])
{
    if (ch >= width) { return; }
    const uint history = min(kern - 1u, SHORTCONV_MAX_HISTORY);
    float next[SHORTCONV_MAX_HISTORY];
    for (uint sI = 0; sI < history; ++sI) {
        const uint e = n_tok + sI;
        if (e < history) {
            next[sI] = state_in[e * width + ch];
        } else {
            const ulong row = (ulong)(e - history) * 3u * width;
            next[sI] = bcx[row + ch] * bcx[row + 2u * width + ch];
        }
    }
    // Still read-all-then-write-all: with in == out and a batch shorter than the history
    // this is a shift, and a thread that wrote first would clobber a slot it still needs.
    for (uint sI = 0; sI < history; ++sI) { state_out[sI * width + ch] = next[sI]; }
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
    for (uint i = tid; i < n; i += tw) {
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
    device const uchar * blk = weights + w_offset
                             + ((ulong)tokens[t] * blocks + bi) * Q4_0_BYTES;
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
