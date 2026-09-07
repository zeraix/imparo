#pragma once

#include "attention_decode_vec_d256_plan.cuh"

// Ampere D256 single-token vector attention for quantized sliding KV.  This is
// intentionally a separate family from the small-query MMA kernels: pinned llama
// quantizes Q to Q8_1, uses Q4_0 x Q8_1 DP4A for QK, and accumulates dequantized V
// in FP32 on NVIDIA. Trying to hide that contract behind the staged-F16 MMA route makes
// both the numeric receipt and future contributor ownership ambiguous.
namespace imparo_sm80_d256_vec {

constexpr uint32_t kHeadDim = imparo_sm80_d256_vec_plan::kHeadDim;
constexpr uint32_t kGqaHeads = imparo_sm80_d256_vec_plan::kGqaHeads;
constexpr uint32_t kWarps = imparo_sm80_d256_vec_plan::kWarps;
constexpr uint32_t kThreads = imparo_sm80_d256_vec_plan::kThreads;
constexpr uint32_t kWindowSpan = imparo_sm80_d256_vec_plan::kWindowSpan;
constexpr uint32_t kScheduleQuantum =
    imparo_sm80_d256_vec_plan::kScheduleQuantum;
constexpr uint32_t kStripesPerPartial =
    imparo_sm80_d256_vec_plan::kStripesPerPartial;
constexpr uint32_t kMaxPartials = imparo_sm80_d256_vec_plan::kMaxPartials;
constexpr uint32_t kMaxPhysicalSpan =
    imparo_sm80_d256_vec_plan::kMaxPhysicalSpan;
constexpr uint32_t kQ4BlocksPerHead = kHeadDim / 32;
constexpr uint32_t kPartialStride = kHeadDim + 2;
// The GQA-sharing laboratory path stores the pre-rescale numerator, maximum,
// and every lane's running denominator.  Keeping the 32 lane values lets the
// second kernel replay the production warp reduction tree instead of changing
// the vector-FA numerical class.
constexpr uint32_t kGqaChildStride = kHeadDim + 2 + 32;
constexpr float kNegInf = -3.402823466e+38F;
constexpr float kMaxOffset = 3.0f * 0.6931f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value,
            __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ int load_q4_int(const uint8_t * block,
                                             uint32_t iqs) {
    // Q4_0 records are 18 bytes, so their payload is only two-byte aligned.
    // Two 16-bit loads match ggml-cuda's get_int_b2 without undefined alignment.
    const auto * values = reinterpret_cast<const uint16_t *>(block + 2);
    return int(values[2 * iqs]) | (int(values[2 * iqs + 1]) << 16);
}

__device__ __forceinline__ float dot_q4_q8_1_d256(
        const uint8_t * row, const int * q_i32, const float2 * q_ds,
        uint32_t lane) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t base = 0; base < 64; base += 32) {
        const uint32_t k = base + lane;
        const uint32_t block_index = k / 8;
        const uint32_t iqs = k & 3;
        const uint32_t shift = k & 4;
        const uint8_t * block = row + uint64_t(block_index) * 18;
        int packed = load_q4_int(block, iqs);
        packed = (packed >> shift) & 0x0f0f0f0f;
        const int sumi = __dp4a(packed, q_i32[k], 0);
        const float d4 = __half2float(
            *reinterpret_cast<const __half *>(block));
        const float2 ds = q_ds[block_index];
        // Each lane owns one eighth of a Q4 block.  The eight lane corrections
        // sum to the block's -8*d*sum(Q), matching the Q4_0 zero point.
        sum += d4 * (float(sumi) * ds.x - ds.y);
    }
    return warp_sum(sum);
}

__device__ __forceinline__ float2 q4_float2(
        const uint8_t * row, uint32_t index) {
    const uint8_t * block = row + uint64_t(index / 32) * 18;
    const uint32_t j = index & 31;
    const uint8_t p0 = block[2 + (j & 15)];
    const uint8_t p1 = block[2 + ((j + 1) & 15)];
    const int q0 = j < 16 ? (p0 & 0x0f) : (p0 >> 4);
    const int q1 = j + 1 < 16 ? (p1 & 0x0f) : (p1 >> 4);
    const float d = __half2float(
        *reinterpret_cast<const __half *>(block));
    return make_float2(d * float(q0 - 8), d * float(q1 - 8));
}

__global__ __launch_bounds__(kThreads, 1) void partial_q4(
        const float * q, const uint8_t * kc, const uint8_t * vc,
        float * partials, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t ring, uint32_t schedule_span, float qk_scale,
        const uint32_t * decode_control) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) start_pos = decode_control[0];
    const uint32_t head = blockIdx.x;
    const uint32_t partial = blockIdx.y;
    const uint32_t partial_count = gridDim.y;
    if (head >= n_heads || partial >= partial_count
        || partial_count == 0 || partial_count > kMaxPartials
        || schedule_span != partial_count * kScheduleQuantum
        || ring == UINT32_MAX || schedule_span > ring + 1) return;
    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t kvh = head / (n_heads / n_kv);
    const uint32_t q4_blocks_per_row = kv_width / 32;
    const float * qr = q + uint64_t(head) * kHeadDim;
    const uint32_t window_lo = start_pos + 1 > kWindowSpan
        ? start_pos + 1 - kWindowSpan : 0;
    const uint32_t valid_span = min(start_pos + 1, ring + 1);

    __shared__ int q_i32[64];
    __shared__ float2 q_ds[kQ4BlocksPerHead];
    __shared__ float probabilities[kThreads];
    __shared__ float warp_maxima[32];
    __shared__ float warp_sums[32];
    __shared__ float2 warp_numerator[kWarps][kHeadDim / 2];

    // Pinned vector FA lets warp 0 publish the Q8_1 representation.  A Q8_1
    // block is four float values per lane across an eight-lane XOR subgroup.
    if (warp == 0) {
#pragma unroll
        for (uint32_t base = 0; base < 64; base += 32) {
            const float4 raw = reinterpret_cast<const float4 *>(qr)[base + lane];
            const float4 value = make_float4(
                qk_scale * raw.x, qk_scale * raw.y,
                qk_scale * raw.z, qk_scale * raw.w);
            float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                fmaxf(fabsf(value.z), fabsf(value.w)));
            float sum = value.x + value.y + value.z + value.w;
#pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                amax = fmaxf(amax,
                    __shfl_xor_sync(0xffffffffu, amax, offset));
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            const float d = amax / 127.0f;
            // Match pinned ggml-cuda's quantize_q8_1_to_shared operation order.
            // Multiplying by a precomputed reciprocal changes boundary rounding and
            // can move a Q8 code by one, which is amplified by sharp attention rows.
            const int8_t q0 = d != 0.0f ? int8_t(roundf(value.x / d)) : 0;
            const int8_t q1 = d != 0.0f ? int8_t(roundf(value.y / d)) : 0;
            const int8_t q2 = d != 0.0f ? int8_t(roundf(value.z / d)) : 0;
            const int8_t q3 = d != 0.0f ? int8_t(roundf(value.w / d)) : 0;
            q_i32[base + lane] = int(uint8_t(q0))
                | (int(uint8_t(q1)) << 8) | (int(uint8_t(q2)) << 16)
                | (int(uint8_t(q3)) << 24);
            if ((lane & 7) == 0) {
                q_ds[base / 8 + lane / 8] = make_float2(d, sum);
            }
        }
    }
    __syncthreads();

    float2 numerator[4] = {
        make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f),
        make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
    float running_max = kNegInf * 0.5f;
    float running_sum = 0.0f;

    for (uint32_t stripe = 0; stripe < kStripesPerPartial; ++stripe) {
        const uint32_t tile = imparo_sm80_d256_vec_plan::partial_stripe_begin(
            partial, stripe, partial_count);
        float score_owned = kNegInf;
        float next_max = running_max;
#pragma unroll
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t slot = tile + warp * 32 + row;
            const uint32_t logical_pos = slot < valid_span
                ? slot + ((start_pos - slot) & ~ring) : 0u;
            const bool valid = slot < valid_span && logical_pos >= window_lo
                && logical_pos <= start_pos;
            const uint8_t * krow = kc
                + (uint64_t(slot) * q4_blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18;
            const float score = valid ? dot_q4_q8_1_d256(
                krow, q_i32, q_ds, lane) : kNegInf;
            next_max = fmaxf(next_max, score + kMaxOffset);
            if (lane == row) score_owned = score;
        }
        const float rescale = expf(running_max - next_max);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            numerator[i].x *= rescale;
            numerator[i].y *= rescale;
        }
        running_max = next_max;
        const float probability = score_owned > -3.0e38F
            ? expf(score_owned - running_max) : 0.0f;
        running_sum = running_sum * rescale + probability;
        probabilities[warp * 32 + lane] = probability;
        __syncwarp();

#pragma unroll
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t slot = tile + warp * 32 + row;
            const uint32_t logical_pos = slot < valid_span
                ? slot + ((start_pos - slot) & ~ring) : 0u;
            const bool valid = slot < valid_span && logical_pos >= window_lo
                && logical_pos <= start_pos;
            const uint8_t * vrow = vc
                + (uint64_t(slot) * q4_blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18;
            const float p = probabilities[warp * 32 + row];
            if (valid && p != 0.0f) {
                const float2 v0 = q4_float2(vrow, 4 * lane);
                const float2 v1 = q4_float2(vrow, 4 * lane + 2);
                const float2 v2 = q4_float2(vrow, 128 + 4 * lane);
                const float2 v3 = q4_float2(vrow, 128 + 4 * lane + 2);
                numerator[0].x += p * v0.x; numerator[0].y += p * v0.y;
                numerator[1].x += p * v1.x; numerator[1].y += p * v1.y;
                numerator[2].x += p * v2.x; numerator[2].y += p * v2.y;
                numerator[3].x += p * v3.x; numerator[3].y += p * v3.y;
            }
        }
        __syncwarp();
    }

    if (warp == 0) {
        warp_maxima[lane] = kNegInf * 0.5f;
        warp_sums[lane] = 0.0f;
    }
    __syncthreads();
    if (lane == 0) warp_maxima[warp] = running_max;
    __syncthreads();
    float combined_max = lane < kWarps ? warp_maxima[lane]
                                       : kNegInf * 0.5f;
    combined_max = warp_max(combined_max);
    const float warp_scale = expf(running_max - combined_max);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        numerator[i].x *= warp_scale;
        numerator[i].y *= warp_scale;
    }
    float sum_scaled = warp_sum(running_sum * warp_scale);
    if (lane == 0) warp_sums[warp] = sum_scaled;
    warp_numerator[warp][2 * lane] = numerator[0];
    warp_numerator[warp][2 * lane + 1] = numerator[1];
    warp_numerator[warp][64 + 2 * lane] = numerator[2];
    warp_numerator[warp][64 + 2 * lane + 1] = numerator[3];
    __syncthreads();

    float * dst = partials
        + (uint64_t(head) * kMaxPartials + partial) * kPartialStride;
    const uint32_t pair = warp * 32 + lane;
    float2 value = make_float2(0.0f, 0.0f);
#pragma unroll
    for (uint32_t source_warp = 0; source_warp < kWarps; ++source_warp) {
                const float2 add = warp_numerator[source_warp][pair];
        value.x += add.x;
        value.y += add.y;
    }
    dst[2 * pair] = value.x;
    dst[2 * pair + 1] = value.y;
    if (warp == 0) {
        float denominator = lane < kWarps ? warp_sums[lane] : 0.0f;
        denominator = warp_sum(denominator);
        if (lane == 0) {
            dst[kHeadDim] = combined_max;
            dst[kHeadDim + 1] = denominator;
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)partials; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos;
    (void)ring; (void)schedule_span; (void)qk_scale; (void)decode_control;
#endif
}

// SM86 laboratory variant for Gemma-4's fixed GQA4 D256 Decode shape.  The
// production kernel assigns four source stripes to the four warps of each
// query-head CTA.  This mapping instead assigns the four query heads sharing a
// KV head to those warps, so one staged Q4 K/V row services all four heads.
//
// The grid still contains n_heads CTAs per partial.  A child record preserves
// the exact per-source-warp state, and combine_q4_gqa4 replays the old
// cross-warp reduction before the unchanged cross-partial combine_q4 kernel.
__global__ __launch_bounds__(kThreads, 1) void partial_q4_gqa4(
        const float * q, const uint8_t * kc, const uint8_t * vc,
        float * children, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t ring, uint32_t schedule_span, float qk_scale,
        const uint32_t * decode_control) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) start_pos = decode_control[0];
    const uint32_t source_warp = blockIdx.x & (kWarps - 1);
    const uint32_t kvh = blockIdx.x / kWarps;
    const uint32_t partial = blockIdx.y;
    const uint32_t partial_count = gridDim.y;
    if (n_kv == 0 || n_heads != n_kv * kGqaHeads || kvh >= n_kv
        || partial >= partial_count || partial_count == 0
        || partial_count > kMaxPartials
        || schedule_span != partial_count * kScheduleQuantum
        || ring == UINT32_MAX || schedule_span > ring + 1) return;
    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t head = kvh * kGqaHeads + warp;
    const uint32_t q4_blocks_per_row = kv_width / 32;
    const float * qr = q + uint64_t(head) * kHeadDim;
    const uint32_t window_lo = start_pos + 1 > kWindowSpan
        ? start_pos + 1 - kWindowSpan : 0;
    const uint32_t valid_span = min(start_pos + 1, ring + 1);

    __shared__ int q_i32[kGqaHeads][64];
    __shared__ float2 q_ds[kGqaHeads][kQ4BlocksPerHead];
    __shared__ float probabilities[kGqaHeads][32];
    // One Q4 D256 row is 8 records x 18 bytes = 9 uint4 values.
    __shared__ uint4 kv_tile[32 * 9];

#pragma unroll
    for (uint32_t base = 0; base < 64; base += 32) {
        const float4 raw = reinterpret_cast<const float4 *>(qr)[base + lane];
        const float4 value = make_float4(
            qk_scale * raw.x, qk_scale * raw.y,
            qk_scale * raw.z, qk_scale * raw.w);
        float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                            fmaxf(fabsf(value.z), fabsf(value.w)));
        float sum = value.x + value.y + value.z + value.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffffu, amax, offset));
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        const float d = amax / 127.0f;
        const int8_t q0 = d != 0.0f ? int8_t(roundf(value.x / d)) : 0;
        const int8_t q1 = d != 0.0f ? int8_t(roundf(value.y / d)) : 0;
        const int8_t q2 = d != 0.0f ? int8_t(roundf(value.z / d)) : 0;
        const int8_t q3 = d != 0.0f ? int8_t(roundf(value.w / d)) : 0;
        q_i32[warp][base + lane] = int(uint8_t(q0))
            | (int(uint8_t(q1)) << 8) | (int(uint8_t(q2)) << 16)
            | (int(uint8_t(q3)) << 24);
        if ((lane & 7) == 0) {
            q_ds[warp][base / 8 + lane / 8] = make_float2(d, sum);
        }
    }
    __syncthreads();

    float2 numerator[4] = {
        make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f),
        make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
    float running_max = kNegInf * 0.5f;
    float running_sum = 0.0f;

    for (uint32_t stripe = 0; stripe < kStripesPerPartial; ++stripe) {
        const uint32_t tile = imparo_sm80_d256_vec_plan::partial_stripe_begin(
            partial, stripe, partial_count);
        const uint32_t first_slot = tile + source_warp * 32;

        // Stage K cooperatively. Physical ring rows are interleaved by KV head,
        // so copy each 144-byte head row independently rather than assuming
        // that 32 rows are contiguous.
        for (uint32_t item = threadIdx.y * 32 + lane;
             item < 32 * 9; item += kThreads) {
            const uint32_t row = item / 9;
            const uint32_t chunk = item - row * 9;
            const uint8_t * src = kc
                + (uint64_t(first_slot + row) * q4_blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18
                + uint64_t(chunk) * sizeof(uint4);
            kv_tile[item] = *reinterpret_cast<const uint4 *>(src);
        }
        __syncthreads();

        float score_owned = kNegInf;
        float next_max = running_max;
#pragma unroll
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t slot = first_slot + row;
            const uint32_t logical_pos = slot < valid_span
                ? slot + ((start_pos - slot) & ~ring) : 0u;
            const bool valid = slot < valid_span && logical_pos >= window_lo
                && logical_pos <= start_pos;
            const uint8_t * krow =
                reinterpret_cast<const uint8_t *>(kv_tile + row * 9);
            const float score = valid ? dot_q4_q8_1_d256(
                krow, q_i32[warp], q_ds[warp], lane) : kNegInf;
            next_max = fmaxf(next_max, score + kMaxOffset);
            if (lane == row) score_owned = score;
        }
        const float rescale = expf(running_max - next_max);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            numerator[i].x *= rescale;
            numerator[i].y *= rescale;
        }
        running_max = next_max;
        const float probability = score_owned > -3.0e38F
            ? expf(score_owned - running_max) : 0.0f;
        running_sum = running_sum * rescale + probability;
        probabilities[warp][lane] = probability;
        __syncthreads();

        for (uint32_t item = threadIdx.y * 32 + lane;
             item < 32 * 9; item += kThreads) {
            const uint32_t row = item / 9;
            const uint32_t chunk = item - row * 9;
            const uint8_t * src = vc
                + (uint64_t(first_slot + row) * q4_blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18
                + uint64_t(chunk) * sizeof(uint4);
            kv_tile[item] = *reinterpret_cast<const uint4 *>(src);
        }
        __syncthreads();

#pragma unroll
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t slot = first_slot + row;
            const uint32_t logical_pos = slot < valid_span
                ? slot + ((start_pos - slot) & ~ring) : 0u;
            const bool valid = slot < valid_span && logical_pos >= window_lo
                && logical_pos <= start_pos;
            const uint8_t * vrow =
                reinterpret_cast<const uint8_t *>(kv_tile + row * 9);
            const float p = probabilities[warp][row];
            if (valid && p != 0.0f) {
                const float2 v0 = q4_float2(vrow, 4 * lane);
                const float2 v1 = q4_float2(vrow, 4 * lane + 2);
                const float2 v2 = q4_float2(vrow, 128 + 4 * lane);
                const float2 v3 = q4_float2(vrow, 128 + 4 * lane + 2);
                numerator[0].x += p * v0.x; numerator[0].y += p * v0.y;
                numerator[1].x += p * v1.x; numerator[1].y += p * v1.y;
                numerator[2].x += p * v2.x; numerator[2].y += p * v2.y;
                numerator[3].x += p * v3.x; numerator[3].y += p * v3.y;
            }
        }
        __syncthreads();
    }

    float * dst = children
        + (((uint64_t(head) * kMaxPartials + partial) * kWarps
            + source_warp) * kGqaChildStride);
    dst[4 * lane] = numerator[0].x;
    dst[4 * lane + 1] = numerator[0].y;
    dst[4 * lane + 2] = numerator[1].x;
    dst[4 * lane + 3] = numerator[1].y;
    dst[128 + 4 * lane] = numerator[2].x;
    dst[128 + 4 * lane + 1] = numerator[2].y;
    dst[128 + 4 * lane + 2] = numerator[3].x;
    dst[128 + 4 * lane + 3] = numerator[3].y;
    dst[kHeadDim + 2 + lane] = running_sum;
    if (lane == 0) {
        dst[kHeadDim] = running_max;
        dst[kHeadDim + 1] = 0.0f;
    }
#else
    (void)q; (void)kc; (void)vc; (void)children; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos;
    (void)ring; (void)schedule_span; (void)qk_scale; (void)decode_control;
#endif
}

__global__ __launch_bounds__(kHeadDim, 1) void combine_q4_gqa4(
        const float * children, float * partials,
        uint32_t n_heads, uint32_t partial_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t partial = blockIdx.y;
    const uint32_t i = threadIdx.x;
    if (head >= n_heads || partial >= partial_count || partial_count == 0
        || partial_count > kMaxPartials) return;
    const float * base = children
        + uint64_t(head * kMaxPartials + partial)
            * kWarps * kGqaChildStride;
    __shared__ float combined_max_shared;
    __shared__ float scales[kWarps];
    __shared__ float source_denominators[kWarps];

    if (i < 32) {
        float combined_max = i < kWarps
            ? base[uint64_t(i) * kGqaChildStride + kHeadDim]
            : kNegInf * 0.5f;
        combined_max = warp_max(combined_max);
        if (i == 0) combined_max_shared = combined_max;
        if (i < kWarps) {
            scales[i] = expf(
                base[uint64_t(i) * kGqaChildStride + kHeadDim]
                - combined_max);
        }
    }
    __syncthreads();

    if (i < 32) {
#pragma unroll
        for (uint32_t source = 0; source < kWarps; ++source) {
            const float lane_sum =
                base[uint64_t(source) * kGqaChildStride
                    + kHeadDim + 2 + i] * scales[source];
            const float sum = warp_sum(lane_sum);
            if (i == 0) source_denominators[source] = sum;
        }
    }
    // The production kernel rounds each source-warp scaling multiply when it
    // stores to shared memory, then performs a separate FP32 add.  Prevent the
    // compiler from contracting this replay into FFMA: even a one-ULP change
    // here can be amplified by the recurrent stack over multiple Decode steps.
    float numerator = __fadd_rn(
        0.0f, __fmul_rn(scales[0], base[i]));
#pragma unroll
    for (uint32_t source = 1; source < kWarps; ++source) {
        const float scaled = __fmul_rn(
            scales[source],
            base[uint64_t(source) * kGqaChildStride + i]);
        numerator = __fadd_rn(numerator, scaled);
    }
    float * dst = partials
        + uint64_t(head * kMaxPartials + partial) * kPartialStride;
    dst[i] = numerator;
    __syncthreads();
    if (i < 32) {
        float denominator = i < kWarps ? source_denominators[i] : 0.0f;
        denominator = warp_sum(denominator);
        if (i == 0) {
            dst[kHeadDim] = combined_max_shared;
            dst[kHeadDim + 1] = denominator;
        }
    }
}

// Production-shaped fused epilogue for the GQA-sharing experiment.  It first
// reconstructs each schedule partial exactly as combine_q4_gqa4 would, then
// performs the unchanged cross-partial online-softmax merge in the same kernel.
// This removes one launch and the 33 KiB partial write/read round trip per layer.
__global__ __launch_bounds__(kHeadDim, 1) void combine_q4_gqa4_final(
        const float * children, float * out,
        uint32_t n_heads, uint32_t partial_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (head >= n_heads || partial_count == 0
        || partial_count > kMaxPartials) return;
    const float * head_base = children
        + uint64_t(head) * kMaxPartials * kWarps * kGqaChildStride;
    __shared__ float partial_maxima[kMaxPartials];
    __shared__ float source_scales[kMaxPartials][kWarps];
    __shared__ float source_denominators[kWarps];
    __shared__ float partial_denominators[kMaxPartials];
    __shared__ float combined_max_shared;

    if (i < 32) {
        for (uint32_t part = 0; part < partial_count; ++part) {
            const float * base = head_base
                + uint64_t(part) * kWarps * kGqaChildStride;
            float combined_max = i < kWarps
                ? base[uint64_t(i) * kGqaChildStride + kHeadDim]
                : kNegInf * 0.5f;
            combined_max = warp_max(combined_max);
            if (i == 0) partial_maxima[part] = combined_max;
            if (i < kWarps) {
                source_scales[part][i] = expf(
                    base[uint64_t(i) * kGqaChildStride + kHeadDim]
                    - combined_max);
            }
            __syncwarp();
#pragma unroll
            for (uint32_t source = 0; source < kWarps; ++source) {
                const float lane_sum =
                    base[uint64_t(source) * kGqaChildStride
                        + kHeadDim + 2 + i] * source_scales[part][source];
                const float sum = warp_sum(lane_sum);
                if (i == 0) source_denominators[source] = sum;
            }
            __syncwarp();
            float denominator =
                i < kWarps ? source_denominators[i] : 0.0f;
            denominator = warp_sum(denominator);
            if (i == 0) partial_denominators[part] = denominator;
            __syncwarp();
        }
        float combined_max = i < partial_count
            ? partial_maxima[i] : kNegInf * 0.5f;
        combined_max = warp_max(combined_max);
        if (i == 0) combined_max_shared = combined_max;
    }
    __syncthreads();

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (uint32_t part = 0; part < partial_count; ++part) {
        const float * base = head_base
            + uint64_t(part) * kWarps * kGqaChildStride;
        float partial_numerator = __fadd_rn(
            0.0f, __fmul_rn(source_scales[part][0], base[i]));
#pragma unroll
        for (uint32_t source = 1; source < kWarps; ++source) {
            const float scaled = __fmul_rn(
                source_scales[part][source],
                base[uint64_t(source) * kGqaChildStride + i]);
            partial_numerator = __fadd_rn(partial_numerator, scaled);
        }
        // Match combine_q4's compiler-visible multiply-add expression here:
        // unlike the source-warp boundary, the old path has no intervening
        // memory round before accumulating schedule partials.
        const float scale = expf(
            partial_maxima[part] - combined_max_shared);
        numerator += scale * partial_numerator;
        denominator += scale * partial_denominators[part];
    }
    out[uint64_t(head) * kHeadDim + i] = numerator / denominator;
}

__global__ void combine_q4(const float * partials, float * out,
                           uint32_t n_heads, uint32_t partial_count) {
    const uint32_t head = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (head >= n_heads || i >= kHeadDim || partial_count == 0
        || partial_count > kMaxPartials) return;
    const float * base = partials
        + uint64_t(head) * kMaxPartials * kPartialStride;
    float combined_max = base[kHeadDim];
    for (uint32_t part = 1; part < partial_count; ++part) {
        combined_max = fmaxf(combined_max,
            base[uint64_t(part) * kPartialStride + kHeadDim]);
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (uint32_t part = 0; part < partial_count; ++part) {
        const float * src = base + uint64_t(part) * kPartialStride;
        const float scale = expf(src[kHeadDim] - combined_max);
        numerator += scale * src[i];
        denominator += scale * src[kHeadDim + 1];
    }
    out[uint64_t(head) * kHeadDim + i] = numerator / denominator;
}

} // namespace imparo_sm80_d256_vec
