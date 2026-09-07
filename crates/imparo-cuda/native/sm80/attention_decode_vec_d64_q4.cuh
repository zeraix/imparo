#pragma once

#include "attention_decode_vec_d64_q4_plan.h"

// SM86 D64/GQA4 single-token vector attention for packed Q4_0 K/V.  Its
// arithmetic mirrors pinned llama's NVIDIA vector FA contract: scaled Q is
// quantized to Q8_1 in 32-value blocks, QK uses Q4_0 x Q8_1 DP4A, V is
// dequantized directly to FP32, and 128-key stripes retain independent online
// softmax state before a deterministic FP32 partition combine.
namespace imparo_sm80_d64_q4_vec {

namespace plan = imparo_sm80_d64_q4_vec_plan;
constexpr uint32_t kHeadDim = plan::kHeadDim;
constexpr uint32_t kGqaHeads = plan::kGqaHeads;
constexpr uint32_t kWarps = plan::kWarps;
constexpr uint32_t kThreads = plan::kThreads;
constexpr uint32_t kQ4BlocksPerHead = kHeadDim / 32;
constexpr uint32_t kPartialStride = plan::kPartialStride;
constexpr float kNegInf = -3.402823466e+38F;
constexpr float kMaxOffset = 3.0f * 0.6931f;

__device__ __forceinline__ float subgroup_sum16(float value) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, 16);
    }
    return value;
}

__device__ __forceinline__ float warp_sum32(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ int load_q4_int(
        const uint8_t * block, uint32_t iqs) {
    // A Q4_0 record is 18 bytes.  The payload is two-byte aligned but not
    // necessarily four-byte aligned, so use the same pair of 16-bit loads as
    // ggml-cuda's vector path.
    const auto * values = reinterpret_cast<const uint16_t *>(block + 2);
    return int(values[2 * iqs]) | (int(values[2 * iqs + 1]) << 16);
}

__device__ __forceinline__ float dot_q4_q8_1(
        const uint8_t * row, const int * q_i32, const float2 * q_ds,
        uint32_t subgroup_lane) {
    const uint32_t block_index = subgroup_lane / 8;
    const uint32_t iqs = subgroup_lane & 3;
    const uint32_t shift = subgroup_lane & 4;
    const uint8_t * block = row + uint64_t(block_index) * 18;
    int packed = load_q4_int(block, iqs);
    packed = (packed >> shift) & 0x0f0f0f0f;
    const int sumi = __dp4a(packed, q_i32[subgroup_lane], 0);
    const float d4 = __half2float(
        *reinterpret_cast<const __half *>(block));
    const float2 ds = q_ds[block_index];
    // Eight lanes contribute to each Q4 block.  Subtracting sum(Q) in every
    // lane reconstructs the block's -8 zero-point correction after reduction.
    return subgroup_sum16(d4 * (float(sumi) * ds.x - ds.y));
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
        uint32_t kv_width, uint32_t start_pos, float qk_scale,
        uint32_t window, uint32_t ring, uint32_t schedule_span,
        const uint32_t * decode_control, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) start_pos = decode_control[0];
    const uint32_t head = blockIdx.x;
    const uint32_t part = blockIdx.y;
    const uint32_t parts = gridDim.y;
    if (head >= n_heads || !n_kv || n_heads / n_kv != kGqaHeads
        || kv_width != n_kv * kHeadDim || !parts || part >= parts) return;

    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t subgroup = lane >> 4;
    const uint32_t subgroup_lane = lane & 15;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t kvh = head / kGqaHeads;
    const uint32_t blocks_per_row = kv_width / 32;
    const float * qr = q + uint64_t(head) * kHeadDim;
    const uint32_t initialized = start_pos + 1;
    const uint32_t capacity = ring ? ring + 1 : 0;
    const uint32_t valid_span = ring && initialized > capacity
        ? capacity : initialized;
    const uint32_t logical_base = ring && initialized > capacity
        ? initialized - capacity : 0;
    const uint32_t window_lo = window && initialized > window
        ? initialized - window : 0;
    const uint32_t stripes = uint32_t(
        (uint64_t(schedule_span) + plan::kKeysPerStripe - 1)
        / plan::kKeysPerStripe);

    __shared__ int q_i32[16];
    __shared__ float2 q_ds[kQ4BlocksPerHead];
    __shared__ float probabilities[kThreads];
    __shared__ float warp_maxima[kWarps];
    __shared__ float warp_denominators[kWarps];
    __shared__ float combined_max_shared;
    __shared__ float2 warp_numerators[kWarps][2][kHeadDim / 2];

    if (warp == 0) {
        float4 raw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (lane < 16) raw = reinterpret_cast<const float4 *>(qr)[lane];
        const float4 value = make_float4(
            qk_scale * raw.x, qk_scale * raw.y,
            qk_scale * raw.z, qk_scale * raw.w);
        float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                            fmaxf(fabsf(value.z), fabsf(value.w)));
        float sum = value.x + value.y + value.z + value.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffffu, amax, offset, 8));
            sum += __shfl_xor_sync(0xffffffffu, sum, offset, 8);
        }
        const float d = amax / 127.0f;
        const int8_t q0 = d != 0.0f ? int8_t(roundf(value.x / d)) : 0;
        const int8_t q1 = d != 0.0f ? int8_t(roundf(value.y / d)) : 0;
        const int8_t q2 = d != 0.0f ? int8_t(roundf(value.z / d)) : 0;
        const int8_t q3 = d != 0.0f ? int8_t(roundf(value.w / d)) : 0;
        if (lane < 16) {
            q_i32[lane] = int(uint8_t(q0)) | (int(uint8_t(q1)) << 8)
                | (int(uint8_t(q2)) << 16) | (int(uint8_t(q3)) << 24);
            if ((lane & 7) == 0) q_ds[lane / 8] = make_float2(d, sum);
        }
    }
    __syncthreads();

    float2 numerator[2] = {
        make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
    float running_max = kNegInf * 0.5f;
    float running_sum = 0.0f;

    for (uint32_t stripe_round = 0;
         part + stripe_round * parts < stripes; ++stripe_round) {
        const uint32_t tile = plan::stripe_begin(part, stripe_round, parts);
        float score_owned = kNegInf;
        bool owned_valid = false;
        float next_max = running_max;
#pragma unroll
        for (uint32_t iter = 0; iter < 16; ++iter) {
            const uint32_t row = subgroup * 16 + iter;
            const uint32_t offset = tile + warp * 32 + row;
            const bool valid = offset < valid_span;
            const uint32_t logical = logical_base + offset;
            const bool attended = valid && logical >= window_lo
                && logical <= start_pos;
            // Every lane named by subgroup_sum16's full warp mask must execute
            // the shuffle, including a half-warp whose padded row is inactive.
            // Read row zero as a safe dummy and discard its score after the
            // collective; branching around the collective deadlocks SM86.
            const uint32_t safe_logical = attended ? logical : logical_base;
            const uint32_t physical = imparo_cuda_kv::physical_row(
                safe_logical, ring, page_table);
            const uint8_t * krow = kc
                + (uint64_t(physical) * blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18;
            const float candidate = dot_q4_q8_1(
                krow, q_i32, q_ds, subgroup_lane);
            const float score = attended ? candidate : kNegInf;
            next_max = fmaxf(next_max, score + kMaxOffset);
            if (subgroup_lane == iter) {
                score_owned = score;
                owned_valid = attended;
            }
        }
        next_max = fmaxf(next_max,
            __shfl_xor_sync(0xffffffffu, next_max, 16));
        const float rescale = expf(running_max - next_max);
        numerator[0].x *= rescale;
        numerator[0].y *= rescale;
        numerator[1].x *= rescale;
        numerator[1].y *= rescale;
        running_max = next_max;
        const float probability = owned_valid
            ? expf(score_owned - running_max) : 0.0f;
        running_sum = running_sum * rescale + probability;
        probabilities[warp * 32 + lane] = probability;
        __syncwarp();

#pragma unroll
        for (uint32_t row0 = 0; row0 < 32; row0 += 2) {
            const uint32_t row = row0 + subgroup;
            const uint32_t offset = tile + warp * 32 + row;
            if (offset >= valid_span) continue;
            const uint32_t logical = logical_base + offset;
            if (logical < window_lo || logical > start_pos) continue;
            const float p = probabilities[warp * 32 + row];
            if (p == 0.0f) continue;
            const uint32_t physical = imparo_cuda_kv::physical_row(
                logical, ring, page_table);
            const uint8_t * vrow = vc
                + (uint64_t(physical) * blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18;
            const float2 value0 = q4_float2(vrow, 4 * subgroup_lane);
            const float2 value1 = q4_float2(vrow, 4 * subgroup_lane + 2);
            numerator[0].x += p * value0.x;
            numerator[0].y += p * value0.y;
            numerator[1].x += p * value1.x;
            numerator[1].y += p * value1.y;
        }
        __syncwarp();
    }

    if (lane == 0) warp_maxima[warp] = running_max;
    __syncthreads();
    if (tid == 0) {
        float value = warp_maxima[0];
#pragma unroll
        for (uint32_t source = 1; source < kWarps; ++source) {
            value = fmaxf(value, warp_maxima[source]);
        }
        combined_max_shared = value;
    }
    __syncthreads();

    const float warp_scale = expf(running_max - combined_max_shared);
    numerator[0].x *= warp_scale;
    numerator[0].y *= warp_scale;
    numerator[1].x *= warp_scale;
    numerator[1].y *= warp_scale;
    const float denominator = warp_sum32(running_sum * warp_scale);
    if (lane == 0) warp_denominators[warp] = denominator;
    warp_numerators[warp][subgroup][2 * subgroup_lane] = numerator[0];
    warp_numerators[warp][subgroup][2 * subgroup_lane + 1] = numerator[1];
    __syncthreads();

    // Match the pinned vector-FA denominator tree exactly: four live warp
    // totals occupy lanes 0..3 and the remaining lanes are explicit zero
    // padding before the full 32-lane XOR reduction.  Every warp executes the
    // collective so the existing tid == kHeadDim writer can consume lane 0.
    const float combined_denominator = warp_sum32(
        lane < kWarps ? warp_denominators[lane] : 0.0f);

    float * dst = partials
        + (uint64_t(head) * parts + part) * kPartialStride;
    if (tid < kHeadDim) {
        const uint32_t pair = tid / 2;
        float value = 0.0f;
#pragma unroll
        for (uint32_t source_warp = 0; source_warp < kWarps; ++source_warp) {
#pragma unroll
            for (uint32_t source_group = 0; source_group < 2; ++source_group) {
                const float2 add =
                    warp_numerators[source_warp][source_group][pair];
                value += (tid & 1) ? add.y : add.x;
            }
        }
        dst[tid] = value;
    }
    if (tid == kHeadDim) {
        dst[kHeadDim] = combined_max_shared;
        dst[kHeadDim + 1] = combined_denominator;
    }
#else
    (void)q; (void)kc; (void)vc; (void)partials; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos; (void)qk_scale;
    (void)window; (void)ring; (void)schedule_span;
    (void)decode_control; (void)page_table;
#endif
}

__global__ void combine_q4(
        const float * partials, float * out,
        uint32_t n_heads, uint32_t parts) {
    const uint32_t head = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (head >= n_heads || i >= kHeadDim || !parts) return;
    const float * base = partials
        + uint64_t(head) * parts * kPartialStride;
    float combined_max = base[kHeadDim];
    for (uint32_t part = 1; part < parts; ++part) {
        combined_max = fmaxf(combined_max,
            base[uint64_t(part) * kPartialStride + kHeadDim]);
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (uint32_t part = 0; part < parts; ++part) {
        const float * src = base + uint64_t(part) * kPartialStride;
        const float scale = expf(src[kHeadDim] - combined_max);
        numerator += scale * src[i];
        denominator += scale * src[kHeadDim + 1];
    }
    out[uint64_t(head) * kHeadDim + i] = denominator > 0.0f
        ? numerator / denominator : 0.0f;
}

} // namespace imparo_sm80_d64_q4_vec
