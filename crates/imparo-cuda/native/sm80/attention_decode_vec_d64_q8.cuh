#pragma once

#include "attention_decode_vec_d64_q4.cuh"

// SM86 D64/GQA4 single-token vector attention for packed Q8_0 K/V. The
// scheduler and deterministic FP32 partition combine match the established
// Q4 route; only the packed cache codec differs.
namespace imparo_sm80_d64_q8_vec {

namespace base = imparo_sm80_d64_q4_vec;
namespace plan = imparo_sm80_d64_q4_vec_plan;
constexpr uint32_t kHeadDim = plan::kHeadDim;
constexpr uint32_t kGqaHeads = plan::kGqaHeads;
constexpr uint32_t kWarps = plan::kWarps;
constexpr uint32_t kThreads = plan::kThreads;
constexpr uint32_t kBlocksPerHead = kHeadDim / 32;
constexpr uint32_t kPartialStride = plan::kPartialStride;
constexpr float kNegInf = -3.402823466e+38F;
constexpr float kMaxOffset = 3.0f * 0.6931f;
// The GQA-sharing child record keeps one source warp's unscaled numerator,
// maximum, and all lane-local denominator sums.  Parts are deliberately a
// runtime dimension: [head][part][source_warp][kGqa4ChildStride].
constexpr uint32_t kGqa4ChildMax = kHeadDim;
constexpr uint32_t kGqa4ChildLaneSums = kHeadDim + 1;
constexpr uint32_t kGqa4ChildStride = kHeadDim + 1 + 32;
constexpr uint32_t kQ8BytesPerHeadRow = kBlocksPerHead * 34;
constexpr uint32_t kQ8WordsPerHeadRow = kQ8BytesPerHeadRow / sizeof(uint32_t);
static_assert(kGqa4ChildStride == 97, "D64 GQA4 child ABI");
static_assert(kQ8BytesPerHeadRow == 68 && kQ8WordsPerHeadRow == 17,
              "D64 Q8 cache row copy ABI");

constexpr uint64_t gqa4_final_shared_bytes(uint32_t parts) {
    // partial maxima + source scales + partial denominators + scratch + max
    return sizeof(float) * (uint64_t(parts) * (kWarps + 2) + kWarps + 1);
}

__device__ __forceinline__ int load_q8_int(
        const uint8_t * block, uint32_t iqs) {
    const auto * values = reinterpret_cast<const uint16_t *>(block + 2);
    return int(values[2 * iqs]) | (int(values[2 * iqs + 1]) << 16);
}

__device__ __forceinline__ float dot_q8_q8_1(
        const uint8_t * row, const int * q_i32, const float2 * q_ds,
        uint32_t subgroup_lane) {
    const uint32_t block_index = subgroup_lane / 8;
    const uint32_t iqs = subgroup_lane & 7;
    const uint8_t * block = row + uint64_t(block_index) * 34;
    const int sumi = __dp4a(
        load_q8_int(block, iqs), q_i32[subgroup_lane], 0);
    const float d8 = __half2float(
        *reinterpret_cast<const __half *>(block));
    return base::subgroup_sum16(
        float(sumi) * d8 * q_ds[block_index].x);
}

__device__ __forceinline__ float2 q8_float2(
        const uint8_t * row, uint32_t index) {
    const uint8_t * block = row + uint64_t(index / 32) * 34;
    const uint32_t j = index & 31;
    const float d = __half2float(
        *reinterpret_cast<const __half *>(block));
    return make_float2(
        d * float(int8_t(block[2 + j])),
        d * float(int8_t(block[2 + j + 1])));
}

__global__ __launch_bounds__(kThreads, 1) void partial_q8(
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
    __shared__ float2 q_ds[kBlocksPerHead];
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
            const uint32_t safe_logical = attended ? logical : logical_base;
            const uint32_t physical = imparo_cuda_kv::physical_row(
                safe_logical, ring, page_table);
            const uint8_t * krow = kc
                + (uint64_t(physical) * blocks_per_row
                    + uint64_t(kvh) * kBlocksPerHead) * 34;
            const float candidate = dot_q8_q8_1(
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
                    + uint64_t(kvh) * kBlocksPerHead) * 34;
            const float2 value0 = q8_float2(vrow, 4 * subgroup_lane);
            const float2 value1 = q8_float2(vrow, 4 * subgroup_lane + 2);
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
    const float denominator = base::warp_sum32(running_sum * warp_scale);
    if (lane == 0) warp_denominators[warp] = denominator;
    warp_numerators[warp][subgroup][2 * subgroup_lane] = numerator[0];
    warp_numerators[warp][subgroup][2 * subgroup_lane + 1] = numerator[1];
    __syncthreads();

    const float combined_denominator = base::warp_sum32(
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

// SM86 laboratory mapping for D64/GQA4.  blockIdx.x retains the production
// n_heads extent, but its low two bits select one original source warp and the
// remaining bits select the shared KV head.  The four CTA warps then execute
// the four query heads against one cooperatively staged 32-row K or V tile.
// Each Q8 D64 row is exactly 68 bytes, copied as seventeen aligned uint32_t
// words.  Invalid graph-bucket rows borrow a valid physical row only for the
// load and remain masked from the online-softmax state.
__global__ __launch_bounds__(kThreads, 1) void partial_q8_gqa4(
        const float * q, const uint8_t * kc, const uint8_t * vc,
        float * children, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos, float qk_scale,
        uint32_t window, uint32_t ring, uint32_t schedule_span,
        const uint32_t * decode_control, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) start_pos = decode_control[0];
    const uint32_t source_warp = blockIdx.x & (kWarps - 1);
    const uint32_t kvh = blockIdx.x / kWarps;
    const uint32_t part = blockIdx.y;
    const uint32_t parts = gridDim.y;
    if (!n_kv || n_heads != n_kv * kGqaHeads || kvh >= n_kv
        || kv_width != n_kv * kHeadDim || !parts || part >= parts) return;

    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t subgroup = lane >> 4;
    const uint32_t subgroup_lane = lane & 15;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t head = kvh * kGqaHeads + warp;
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

    __shared__ int q_i32[kGqaHeads][16];
    __shared__ float2 q_ds[kGqaHeads][kBlocksPerHead];
    __shared__ float probabilities[kGqaHeads][32];
    __shared__ uint32_t kv_tile[32 * kQ8WordsPerHeadRow];

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
        q_i32[warp][lane] = int(uint8_t(q0)) | (int(uint8_t(q1)) << 8)
            | (int(uint8_t(q2)) << 16) | (int(uint8_t(q3)) << 24);
        if ((lane & 7) == 0) {
            q_ds[warp][lane / 8] = make_float2(d, sum);
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
        const uint32_t first_offset = tile + source_warp * 32;

        for (uint32_t item = tid; item < 32 * kQ8WordsPerHeadRow;
             item += kThreads) {
            const uint32_t row = item / kQ8WordsPerHeadRow;
            const uint32_t word = item % kQ8WordsPerHeadRow;
            const uint32_t offset = first_offset + row;
            const bool valid = offset < valid_span;
            const uint32_t logical = logical_base + (valid ? offset : 0);
            const bool attended = valid && logical >= window_lo
                && logical <= start_pos;
            const uint32_t safe_logical = attended ? logical : logical_base;
            const uint32_t physical = imparo_cuda_kv::physical_row(
                safe_logical, ring, page_table);
            const uint8_t * src = kc
                + (uint64_t(physical) * blocks_per_row
                    + uint64_t(kvh) * kBlocksPerHead) * 34
                + uint64_t(word) * sizeof(uint32_t);
            kv_tile[item] = *reinterpret_cast<const uint32_t *>(src);
        }
        __syncthreads();

        float score_owned = kNegInf;
        bool owned_valid = false;
        float next_max = running_max;
#pragma unroll
        for (uint32_t iter = 0; iter < 16; ++iter) {
            const uint32_t row = subgroup * 16 + iter;
            const uint32_t offset = first_offset + row;
            const bool valid = offset < valid_span;
            const uint32_t logical = logical_base + (valid ? offset : 0);
            const bool attended = valid && logical >= window_lo
                && logical <= start_pos;
            const uint8_t * krow = reinterpret_cast<const uint8_t *>(
                kv_tile + row * kQ8WordsPerHeadRow);
            const float candidate = dot_q8_q8_1(
                krow, q_i32[warp], q_ds[warp], subgroup_lane);
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
        probabilities[warp][lane] = probability;
        __syncthreads();

        for (uint32_t item = tid; item < 32 * kQ8WordsPerHeadRow;
             item += kThreads) {
            const uint32_t row = item / kQ8WordsPerHeadRow;
            const uint32_t word = item % kQ8WordsPerHeadRow;
            const uint32_t offset = first_offset + row;
            const bool valid = offset < valid_span;
            const uint32_t logical = logical_base + (valid ? offset : 0);
            const bool attended = valid && logical >= window_lo
                && logical <= start_pos;
            const uint32_t safe_logical = attended ? logical : logical_base;
            const uint32_t physical = imparo_cuda_kv::physical_row(
                safe_logical, ring, page_table);
            const uint8_t * src = vc
                + (uint64_t(physical) * blocks_per_row
                    + uint64_t(kvh) * kBlocksPerHead) * 34
                + uint64_t(word) * sizeof(uint32_t);
            kv_tile[item] = *reinterpret_cast<const uint32_t *>(src);
        }
        __syncthreads();

#pragma unroll
        for (uint32_t row0 = 0; row0 < 32; row0 += 2) {
            const uint32_t row = row0 + subgroup;
            const uint32_t offset = first_offset + row;
            if (offset >= valid_span) continue;
            const uint32_t logical = logical_base + offset;
            if (logical < window_lo || logical > start_pos) continue;
            const float p = probabilities[warp][row];
            if (p == 0.0f) continue;
            const uint8_t * vrow = reinterpret_cast<const uint8_t *>(
                kv_tile + row * kQ8WordsPerHeadRow);
            const float2 value0 = q8_float2(vrow, 4 * subgroup_lane);
            const float2 value1 = q8_float2(vrow, 4 * subgroup_lane + 2);
            numerator[0].x += p * value0.x;
            numerator[0].y += p * value0.y;
            numerator[1].x += p * value1.x;
            numerator[1].y += p * value1.y;
        }
        __syncthreads();
    }

    float * dst = children
        + (((uint64_t(head) * parts + part) * kWarps + source_warp)
            * kGqa4ChildStride);
    const float n0x = __shfl_xor_sync(
        0xffffffffu, numerator[0].x, 16);
    const float n0y = __shfl_xor_sync(
        0xffffffffu, numerator[0].y, 16);
    const float n1x = __shfl_xor_sync(
        0xffffffffu, numerator[1].x, 16);
    const float n1y = __shfl_xor_sync(
        0xffffffffu, numerator[1].y, 16);
    if (subgroup == 0) {
        dst[4 * subgroup_lane] = __fadd_rn(numerator[0].x, n0x);
        dst[4 * subgroup_lane + 1] = __fadd_rn(numerator[0].y, n0y);
        dst[4 * subgroup_lane + 2] = __fadd_rn(numerator[1].x, n1x);
        dst[4 * subgroup_lane + 3] = __fadd_rn(numerator[1].y, n1y);
    }
    dst[kGqa4ChildLaneSums + lane] = running_sum;
    if (lane == 0) dst[kGqa4ChildMax] = running_max;
#else
    (void)q; (void)kc; (void)vc; (void)children; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos; (void)qk_scale;
    (void)window; (void)ring; (void)schedule_span;
    (void)decode_control; (void)page_table;
#endif
}

// Two-node companion for partial_q8_gqa4.  Dynamic shared storage is supplied
// by gqa4_final_shared_bytes(parts).  It reconstructs each part's four-source
// reduction, then applies the established cross-part online-softmax merge.
__global__ __launch_bounds__(kHeadDim, 1) void combine_q8_gqa4_final(
        const float * children, float * out,
        uint32_t n_heads, uint32_t parts) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t head = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (head >= n_heads || i >= kHeadDim || !parts) return;
    const float * head_base = children
        + uint64_t(head) * parts * kWarps * kGqa4ChildStride;
    extern __shared__ __align__(16) float shared[];
    float * partial_maxima = shared;
    float * source_scales = partial_maxima + parts;
    float * partial_denominators = source_scales + uint64_t(parts) * kWarps;
    float * source_denominators = partial_denominators + parts;
    float * combined_max_shared = source_denominators + kWarps;
    const uint32_t lane = i & 31;
    const uint32_t warp = i >> 5;

    if (warp == 0) {
        for (uint32_t part = 0; part < parts; ++part) {
            const float * part_base = head_base
                + uint64_t(part) * kWarps * kGqa4ChildStride;
            float combined_max = lane < kWarps
                ? part_base[uint64_t(lane) * kGqa4ChildStride
                    + kGqa4ChildMax]
                : kNegInf * 0.5f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                combined_max = fmaxf(combined_max,
                    __shfl_xor_sync(0xffffffffu, combined_max, offset));
            }
            if (lane == 0) partial_maxima[part] = combined_max;
            if (lane < kWarps) {
                source_scales[uint64_t(part) * kWarps + lane] = expf(
                    part_base[uint64_t(lane) * kGqa4ChildStride + kGqa4ChildMax]
                    - combined_max);
            }
            __syncwarp();
#pragma unroll
            for (uint32_t source = 0; source < kWarps; ++source) {
                const float lane_sum =
                    part_base[uint64_t(source) * kGqa4ChildStride
                        + kGqa4ChildLaneSums + lane]
                    * source_scales[uint64_t(part) * kWarps + source];
                const float sum = base::warp_sum32(lane_sum);
                if (lane == 0) source_denominators[source] = sum;
            }
            __syncwarp();
            float denominator = lane < kWarps
                ? source_denominators[lane] : 0.0f;
            denominator = base::warp_sum32(denominator);
            if (lane == 0) partial_denominators[part] = denominator;
            __syncwarp();
        }
        if (lane == 0) {
            float combined_max = partial_maxima[0];
            for (uint32_t part = 1; part < parts; ++part) {
                combined_max = fmaxf(combined_max, partial_maxima[part]);
            }
            *combined_max_shared = combined_max;
        }
    }
    __syncthreads();

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (uint32_t part = 0; part < parts; ++part) {
        const float * part_base = head_base
            + uint64_t(part) * kWarps * kGqa4ChildStride;
        float partial_numerator = __fadd_rn(0.0f, __fmul_rn(
            source_scales[uint64_t(part) * kWarps], part_base[i]));
#pragma unroll
        for (uint32_t source = 1; source < kWarps; ++source) {
            const float scaled = __fmul_rn(
                source_scales[uint64_t(part) * kWarps + source],
                part_base[uint64_t(source) * kGqa4ChildStride + i]);
            partial_numerator = __fadd_rn(partial_numerator, scaled);
        }
        const float scale = expf(
            partial_maxima[part] - *combined_max_shared);
        numerator += scale * partial_numerator;
        denominator += scale * partial_denominators[part];
    }
    out[uint64_t(head) * kHeadDim + i] = denominator > 0.0f
        ? numerator / denominator : 0.0f;
#else
    (void)children; (void)out; (void)n_heads; (void)parts;
#endif
}

} // namespace imparo_sm80_d64_q8_vec
