#pragma once

#include "attention_prefill_d512_f16.cuh"

// Fused SM80 D256 prefill attention.
//
// One CTA owns a complete 64-column GQA tile. Scores and probabilities never
// leave the CTA; only the half-accumulator numerator at an actual Stream-K seam
// is staged to the caller-provided bounded workspace. Physical and stable virtual
// schedules share the staged implementation's exact seam locator. This is kept
// separate so later SM families can replace the schedule without changing common
// dispatch or the exact fallback.
namespace imparo_sm80_fa_d256 {

using imparo_sm80_mma::Float16x16;
using imparo_sm80_mma::Half16x8;
using imparo_sm80_mma::fragment_key_row;
using imparo_sm80_mma::fragment_q_column;
using imparo_sm80_mma::half_acc_output_pair;
using imparo_sm80_mma::half_acc_query_column;
using imparo_sm80_mma::load_half16x8;
using imparo_sm80_mma::load_half16x8_trans;
using imparo_sm80_mma::mma_pv;
using imparo_sm80_mma::mma_qk;

constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kColumns = 64;
constexpr uint32_t kColumnTiles = kColumns / 16;
constexpr uint32_t kOutputTiles = kHeadDim / 16;
constexpr uint32_t kCacheStride = kHeadDim / 2 + 4;
constexpr float kNegInf = -3.402823466e+38F;
constexpr float kMaxOffset = 3.0f * 0.6931f;

template <bool F32>
struct NumeratorFragment;

template <>
struct NumeratorFragment<false> {
    Half16x8 value;
};

template <>
struct NumeratorFragment<true> {
    Float16x16 value;
};

__device__ __forceinline__ void prefetch_cache_batch(
        const __half * cache, __half2 * dst, uint32_t group,
        uint32_t kvh, uint32_t kv_width, uint32_t valid_span,
        uint32_t ring, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 32 rows x 256 half values, copied as naturally aligned 16-byte sectors.
    for (uint32_t sector = threadIdx.x; sector < 32 * 32;
         sector += blockDim.x) {
        const uint32_t key_row = sector >> 5;
        const uint32_t row_sector = sector & 31;
        const uint32_t key = group * 32 + key_row;
        const uint32_t physical_key =
            imparo_cuda_kv::physical_row(key, ring, page_table);
        const __half * src = key < valid_span
            ? cache + uint64_t(physical_key) * kv_width + kvh * kHeadDim
                + row_sector * 8
            : cache;
        __half2 * target = dst
            + key_row * kCacheStride + row_sector * 4;
        const uint32_t shared = static_cast<uint32_t>(
            __cvta_generic_to_shared(target));
        const uint32_t bytes = key < valid_span ? 16u : 0u;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(shared), "l"(src), "r"(bytes));
    }
#else
    (void)cache; (void)dst; (void)group; (void)kvh;
    (void)kv_width; (void)valid_span; (void)ring; (void)page_table;
#endif
}

template <bool Ringed /* compile-time ring policy */>
__device__ __forceinline__ float masked_score(
        float score, uint32_t qcol, uint32_t key, uint32_t query_tile,
        uint32_t query_tokens, uint32_t n_tok, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span) {
    const uint32_t token = query_tile * query_tokens
        + qcol / imparo_sm80_prefill::kGqaHeads;
    const uint32_t pos = start_pos + token;
    if constexpr (Ringed) {
        const uint32_t key_pos = key
            + ((start_pos + n_tok - 1 - key) & ~ring);
        // Unsigned age folds the causal and lower-window bounds into one
        // comparison: a key newer than this query underflows to a large value.
        return token < n_tok && key < valid_span
                && uint32_t(pos - key_pos) < window
            ? score : kNegInf;
    }
    const uint32_t lo = window > 0 && pos + 1 > window
        ? pos + 1 - window : 0;
    const uint32_t key_pos = key < valid_span
        ? imparo_sm80_prefill::physical_key_position(
            key, ring, start_pos + n_tok - 1)
        : 0;
    return token < n_tok && key < valid_span
            && key_pos >= lo && key_pos <= pos
        ? score : kNegInf;
}

template <bool Ringed, bool F32Numerator = false>
__global__ __launch_bounds__(128, 2) void flash(
        const __half * q, const __half * kc, const __half * vc,
        float * workspace, float * out, uint32_t block_base,
        uint32_t query_tiles, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos, uint32_t window,
        uint32_t n_tok, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t virtual_stream_blocks, uint32_t segment_slots,
        uint32_t query_tokens,
        uint64_t block_stride, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t local_block = blockIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;
    const uint32_t absolute_query_start =
        start_pos + query_tile * query_tokens;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t subgroup = lane >> 2;
    const uint32_t sublane = lane & 3;
    const uint32_t subgroup_mask = 0x0000000fu << (4 * subgroup);
    const uint32_t subgroup_base = 4 * subgroup;

    __shared__ __align__(16) __half2 q_tile[kColumnTiles][16 * 8];
    __shared__ __align__(16) __half2 k_tile[32][kCacheStride];
    __shared__ __align__(16) __half2 p_tile[2][kColumnTiles][16 * 8];
    // A 32-key D256 batch is 16 KiB. Staging it whole removes the barrier
    // between each of the 16 output MMA tiles.
    __shared__ __align__(16) __half2 v_tile[32][kCacheStride];

    float * partial_num = workspace + uint64_t(local_block) * block_stride;
    float * partial_max = partial_num + uint64_t(kColumns) * kHeadDim;
    float * partial_sum = partial_max + kColumns;

    // Ampere's D256/64-column FA shape keeps Q resident. Loading these 16 MMA
    // fragments once is essential: rereading 32 KiB of Q for every key batch
    // overwhelms the traffic saved by fusion.
    Half16x8 q_fragments[kHeadDim / 16];
#pragma unroll
    for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
        // Each warp consumes exactly one Q column tile. Let it also publish
        // that tile so the reusable transpose buffer has no cross-warp
        // dependency; warp synchronization replaces two CTA-wide barriers per
        // 16-dimension fragment without changing the ldmatrix layout.
        for (uint32_t item = lane; item < 16 * 8; item += 32) {
            const uint32_t column_tile = warp;
            const uint32_t qrow = item >> 3;
            const uint32_t pair = item & 7;
            const uint32_t qcol = column_tile * 16 + qrow;
            const uint32_t token = query_tile * query_tokens
                + qcol / imparo_sm80_prefill::kGqaHeads;
            const uint32_t head = kvh * imparo_sm80_prefill::kGqaHeads
                + qcol % imparo_sm80_prefill::kGqaHeads;
            __half2 value = __float2half2_rn(0.0f);
            if (token < n_tok && head < n_heads) {
                value = reinterpret_cast<const __half2 *>(q
                    + (uint64_t(token) * n_heads + head) * kHeadDim
                    + d0)[pair];
            }
            q_tile[column_tile][item] = value;
        }
        __syncwarp();
        load_half16x8(q_fragments[d0 / 16], q_tile[warp], 8, lane);
        __syncwarp();
    }

    uint32_t valid_segments = 0;
    for (uint32_t slot = 0; slot < segment_slots; ++slot) {
        uint32_t begin = 0;
        uint32_t end = 0;
        const bool have_bounds = virtual_stream_blocks > 0
            ? imparo_sm80_prefill::stable_virtual_stream_segment_bounds(
                absolute_query_start, kvh, n_kv, query_tokens, ring,
                schedule_groups, virtual_stream_blocks, slot, &begin, &end)
            : imparo_sm80_prefill::stream_segment_bounds(
                logical_block, n_kv * query_tiles, schedule_groups,
                physical_blocks, slot, &begin, &end);
        valid_segments += have_bounds ? 1u : 0u;
    }

    NumeratorFragment<F32Numerator> numerator[kOutputTiles];
    float segment_max[2] = {kNegInf * 0.5f, kNegInf * 0.5f};
    float segment_sum[2] = {0.0f, 0.0f};
    uint32_t segment_index = 0;

    for (uint32_t slot_rev = segment_slots; slot_rev > 0; --slot_rev) {
        const uint32_t slot = slot_rev - 1;
        uint32_t group_begin = 0;
        uint32_t group_end = 0;
        const bool have_bounds = virtual_stream_blocks > 0
            ? imparo_sm80_prefill::stable_virtual_stream_segment_bounds(
                absolute_query_start, kvh, n_kv, query_tokens, ring,
                schedule_groups, virtual_stream_blocks, slot,
                &group_begin, &group_end)
            : imparo_sm80_prefill::stream_segment_bounds(
                logical_block, n_kv * query_tiles, schedule_groups,
                physical_blocks, slot, &group_begin, &group_end);
        if (!have_bounds) {
            continue;
        }

#pragma unroll
        for (uint32_t tile = 0; tile < kOutputTiles; ++tile) {
            if constexpr (F32Numerator) {
#pragma unroll
                for (uint32_t l = 0; l < 8; ++l) {
                    numerator[tile].value.x[l] = 0.0f;
                }
            } else {
#pragma unroll
                for (uint32_t l = 0; l < 4; ++l) {
                    numerator[tile].value.x[l] = __float2half2_rn(0.0f);
                }
            }
        }
        float running_max[2] = {kNegInf * 0.5f, kNegInf * 0.5f};
        float running_sum[2] = {0.0f, 0.0f};

        prefetch_cache_batch(kc, &k_tile[0][0], group_begin,
            kvh, kv_width, valid_span, ring, page_table);
        asm volatile("cp.async.commit_group;");

        for (uint32_t group = group_begin; group < group_end; ++group) {
            Float16x16 scores[2]{};
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            // K is resident for QK, so start V immediately. Its 16 KiB transfer
            // overlaps the tensor-core dot products and online softmax below.
            prefetch_cache_batch(vc, &v_tile[0][0], group,
                kvh, kv_width, valid_span, ring, page_table);
            asm volatile("cp.async.commit_group;");

#pragma unroll
            for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
                Half16x8 k_fragment;
#pragma unroll
                for (uint32_t partition = 0; partition < 2; ++partition) {
                    load_half16x8(k_fragment,
                        &k_tile[partition * 16][d0 / 2],
                        kCacheStride, lane);
                    mma_qk(scores[partition], q_fragments[d0 / 16], k_fragment);
                }
            }
            const bool have_next_group = group + 1 < group_end;

#pragma unroll
            for (uint32_t partition = 0; partition < 2; ++partition) {
#pragma unroll
                for (uint32_t l = 0; l < 8; ++l) {
                    const uint32_t qcol = warp * 16
                        + fragment_q_column(lane, l);
                    const uint32_t key = group * 32 + partition * 16
                        + fragment_key_row(lane, l);
                    scores[partition].x[l] = masked_score<Ringed>(
                        scores[partition].x[l], qcol, key, query_tile,
                        query_tokens, n_tok, start_pos, window, ring, valid_span);
                }
            }

            float next_max[2] = {running_max[0], running_max[1]};
#pragma unroll
            for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
                const uint32_t l0 = 2 * qhalf;
#pragma unroll
                for (uint32_t partition = 0; partition < 2; ++partition) {
                    const float * s = scores[partition].x;
                    if (s[l0] > -3.0e38F) {
                        next_max[qhalf] = fmaxf(next_max[qhalf], s[l0] + kMaxOffset);
                    }
                    if (s[l0 + 1] > -3.0e38F) {
                        next_max[qhalf] = fmaxf(next_max[qhalf], s[l0 + 1] + kMaxOffset);
                    }
                    if (s[l0 + 4] > -3.0e38F) {
                        next_max[qhalf] = fmaxf(next_max[qhalf], s[l0 + 4] + kMaxOffset);
                    }
                    if (s[l0 + 5] > -3.0e38F) {
                        next_max[qhalf] = fmaxf(next_max[qhalf], s[l0 + 5] + kMaxOffset);
                    }
                }
#pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    next_max[qhalf] = fmaxf(next_max[qhalf],
                        __shfl_xor_sync(subgroup_mask, next_max[qhalf], offset));
                }
                next_max[qhalf] = __shfl_sync(
                    subgroup_mask, next_max[qhalf], subgroup_base);
            }

            float rescale[2];
            float probability_add[2] = {0.0f, 0.0f};
#pragma unroll
            for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
                const float diff = running_max[qhalf] - next_max[qhalf];
                rescale[qhalf] = diff >= -20.0f ? expf(diff) : 0.0f;
                running_max[qhalf] = next_max[qhalf];
            }
            if (group > group_begin) {
#pragma unroll
                for (uint32_t tile = 0; tile < kOutputTiles; ++tile) {
                    if constexpr (F32Numerator) {
#pragma unroll
                        for (uint32_t l = 0; l < 8; ++l) {
                            const uint32_t qhalf =
                                fragment_q_column(lane, l) >> 3;
                            numerator[tile].value.x[l] *= rescale[qhalf];
                        }
                    } else {
#pragma unroll
                        for (uint32_t l = 0; l < 4; ++l) {
                            numerator[tile].value.x[l] = __hmul2(
                                numerator[tile].value.x[l],
                                __float2half2_rn(rescale[l & 1]));
                        }
                    }
                }
            }

#pragma unroll
            for (uint32_t partition = 0; partition < 2; ++partition) {
                float probability[2][4];
#pragma unroll
                for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
                    const uint32_t l0 = 2 * qhalf;
                    const float * s = scores[partition].x;
                    probability[qhalf][0] = s[l0] > -3.0e38F
                        ? expf(s[l0] - next_max[qhalf]) : 0.0f;
                    probability[qhalf][1] = s[l0 + 1] > -3.0e38F
                        ? expf(s[l0 + 1] - next_max[qhalf]) : 0.0f;
                    probability[qhalf][2] = s[l0 + 4] > -3.0e38F
                        ? expf(s[l0 + 4] - next_max[qhalf]) : 0.0f;
                    probability[qhalf][3] = s[l0 + 5] > -3.0e38F
                        ? expf(s[l0 + 5] - next_max[qhalf]) : 0.0f;
                    // Preserve the reference FA reduction boundary: accumulate all
                    // eight lane-owned probabilities for the 32-key batch first,
                    // then update the running denominator once.
                    probability_add[qhalf] += probability[qhalf][0];
                    probability_add[qhalf] += probability[qhalf][1];
                    probability_add[qhalf] += probability[qhalf][2];
                    probability_add[qhalf] += probability[qhalf][3];
                    const uint32_t qrow = qhalf * 8 + subgroup;
                    p_tile[partition][warp][qrow * 8 + sublane] = __floats2half2_rn(
                        probability[qhalf][0], probability[qhalf][1]);
                    p_tile[partition][warp][qrow * 8 + 4 + sublane] = __floats2half2_rn(
                        probability[qhalf][2], probability[qhalf][3]);
                }
            }
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            // The current K tile is dead and V is now visible. Refill K while
            // the PV MMA chain consumes the independent V buffer.
            if (have_next_group) {
                prefetch_cache_batch(kc, &k_tile[0][0], group + 1,
                    kvh, kv_width, valid_span, ring, page_table);
                asm volatile("cp.async.commit_group;");
            }
#pragma unroll
            for (uint32_t partition = 0; partition < 2; ++partition) {
                Half16x8 p_fragment;
                load_half16x8(
                    p_fragment, p_tile[partition][warp], 8, lane);
#pragma unroll
                for (uint32_t output_tile = 0;
                     output_tile < kOutputTiles; ++output_tile) {
                    Half16x8 v_fragment;
                    load_half16x8_trans(v_fragment,
                        &v_tile[partition * 16][output_tile * 8],
                        kCacheStride, lane);
                    if constexpr (F32Numerator) {
                        mma_qk(numerator[output_tile].value,
                               p_fragment, v_fragment);
                    } else {
                        mma_pv(numerator[output_tile].value,
                               p_fragment, v_fragment);
                    }
                }

            }
            __syncthreads();
#pragma unroll
            for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
                running_sum[qhalf] = running_sum[qhalf] * rescale[qhalf]
                    + probability_add[qhalf];
            }
        }

#pragma unroll
        for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
            float sum = running_sum[qhalf];
#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(subgroup_mask, sum, offset);
            }
            segment_sum[qhalf] = sum;
            segment_max[qhalf] = running_max[qhalf];
            if (valid_segments > 1 && segment_index == 0 && sublane == 0) {
                const uint32_t qcol = warp * 16 + qhalf * 8 + subgroup;
                partial_max[qcol] = running_max[qhalf];
                partial_sum[qcol] = sum;
            }
        }

        if (valid_segments > 1 && segment_index == 0) {
#pragma unroll
            for (uint32_t output_tile = 0;
                 output_tile < kOutputTiles; ++output_tile) {
                if constexpr (F32Numerator) {
#pragma unroll
                    for (uint32_t l = 0; l < 8; ++l) {
                        const uint32_t qcol = warp * 16
                            + fragment_q_column(lane, l);
                        const uint32_t output = output_tile * 16
                            + fragment_key_row(lane, l);
                        partial_num[uint64_t(qcol) * kHeadDim + output] =
                            numerator[output_tile].value.x[l];
                    }
                } else {
#pragma unroll
                    for (uint32_t l = 0; l < 4; ++l) {
                        const uint32_t qcol = warp * 16
                            + half_acc_query_column(lane, l);
                        const uint32_t output_pair =
                            half_acc_output_pair(lane, l);
                        const uint32_t output0 =
                            output_tile * 16 + 2 * output_pair;
                        partial_num[uint64_t(qcol) * kHeadDim + output0] =
                            __half2float(__low2half(
                                numerator[output_tile].value.x[l]));
                        partial_num[uint64_t(qcol) * kHeadDim + output0 + 1] =
                            __half2float(__high2half(
                                numerator[output_tile].value.x[l]));
                    }
                }
            }
        } else {
#pragma unroll
            for (uint32_t output_tile = 0;
                 output_tile < kOutputTiles; ++output_tile) {
                if constexpr (F32Numerator) {
#pragma unroll
                    for (uint32_t l = 0; l < 8; ++l) {
                        const uint32_t q_in_tile =
                            fragment_q_column(lane, l);
                        const uint32_t qcol = warp * 16 + q_in_tile;
                        const uint32_t qhalf = q_in_tile >> 3;
                        const uint32_t output = output_tile * 16
                            + fragment_key_row(lane, l);
                        float value = numerator[output_tile].value.x[l];
                        float denominator = segment_sum[qhalf];
                        if (valid_segments > 1) {
                            const float first_max = partial_max[qcol];
                            const float combined_max = fmaxf(first_max,
                                segment_max[qhalf]);
                            const float scale_value =
                                first_max - combined_max >= -20.0f
                                ? expf(first_max - combined_max) : 0.0f;
                            const float scale_add = segment_max[qhalf]
                                    - combined_max >= -20.0f
                                ? expf(segment_max[qhalf] - combined_max) : 0.0f;
                            value = fmaf(scale_value,
                                partial_num[uint64_t(qcol) * kHeadDim + output],
                                scale_add * value);
                            denominator = fmaf(scale_value, partial_sum[qcol],
                                scale_add * denominator);
                        }
                        const uint32_t token = query_tile * query_tokens
                            + qcol / imparo_sm80_prefill::kGqaHeads;
                        const uint32_t head =
                            kvh * imparo_sm80_prefill::kGqaHeads
                            + qcol % imparo_sm80_prefill::kGqaHeads;
                        if (token < n_tok && head < n_heads) {
                            float * dst = out
                                + (uint64_t(token) * n_heads + head) * kHeadDim
                                + output;
                            dst[0] = denominator > 0.0f
                                ? value / denominator : 0.0f;
                        }
                    }
                } else {
#pragma unroll
                    for (uint32_t l = 0; l < 4; ++l) {
                        const uint32_t q_in_tile =
                            half_acc_query_column(lane, l);
                        const uint32_t qcol = warp * 16 + q_in_tile;
                        const uint32_t qhalf = q_in_tile >> 3;
                        const uint32_t output_pair =
                            half_acc_output_pair(lane, l);
                        const uint32_t output0 =
                            output_tile * 16 + 2 * output_pair;
                        float value0 = __half2float(__low2half(
                            numerator[output_tile].value.x[l]));
                        float value1 = __half2float(__high2half(
                            numerator[output_tile].value.x[l]));
                        float denominator = segment_sum[qhalf];
                        if (valid_segments > 1) {
                            const float first_max = partial_max[qcol];
                            const float combined_max = fmaxf(first_max,
                                segment_max[qhalf]);
                            const float scale_value =
                                first_max - combined_max >= -20.0f
                                ? expf(first_max - combined_max) : 0.0f;
                            const float scale_add = segment_max[qhalf]
                                    - combined_max >= -20.0f
                                ? expf(segment_max[qhalf] - combined_max) : 0.0f;
                            value0 = fmaf(scale_value,
                                partial_num[uint64_t(qcol) * kHeadDim + output0],
                                scale_add * value0);
                            value1 = fmaf(scale_value,
                                partial_num[uint64_t(qcol) * kHeadDim + output0 + 1],
                                scale_add * value1);
                            denominator = fmaf(scale_value, partial_sum[qcol],
                                scale_add * denominator);
                        }
                        const uint32_t token = query_tile * query_tokens
                            + qcol / imparo_sm80_prefill::kGqaHeads;
                        const uint32_t head =
                            kvh * imparo_sm80_prefill::kGqaHeads
                            + qcol % imparo_sm80_prefill::kGqaHeads;
                        if (token < n_tok && head < n_heads) {
                            float * dst = out
                                + (uint64_t(token) * n_heads + head) * kHeadDim
                                + output0;
                            dst[0] = denominator > 0.0f
                                ? value0 / denominator : 0.0f;
                            dst[1] = denominator > 0.0f
                                ? value1 / denominator : 0.0f;
                        }
                    }
                }
            }
        }
        ++segment_index;
    }
#else
    (void)q; (void)kc; (void)vc; (void)workspace; (void)out;
    (void)block_base; (void)query_tiles; (void)n_heads; (void)n_kv;
    (void)kv_width; (void)start_pos; (void)window; (void)n_tok;
    (void)ring; (void)valid_span; (void)schedule_groups;
    (void)physical_blocks; (void)virtual_stream_blocks;
    (void)segment_slots; (void)query_tokens;
    (void)block_stride; (void)page_table;
#endif
}

} // namespace imparo_sm80_fa_d256
