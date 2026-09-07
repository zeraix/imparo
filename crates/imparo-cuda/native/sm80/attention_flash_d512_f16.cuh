#pragma once

#include "attention_prefill_d512_f16.cuh"

// Fused SM80 D512 prefill attention. The 64-column shape is intentionally a
// separate architecture kernel: unlike D256, Q cannot coexist with the full PV
// accumulator in registers. Q stays in opt-in shared memory while one padded
// 16 KiB cache tile is reused for K and V.
namespace imparo_sm80_fa_d512 {

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

constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kColumns = 64;
constexpr uint32_t kQStride = kHeadDim / 2 + 4;
constexpr uint32_t kCacheChunk = 256;
constexpr uint32_t kCacheStride = kCacheChunk / 2 + 4;
constexpr uint32_t kOutputTiles = kHeadDim / 16;
constexpr uint32_t kQHalf2 = kColumns * kQStride;
constexpr uint32_t kCacheHalf2 = 32 * kCacheStride;
constexpr uint32_t kScratchBytes = 8 * 16 * 8 * sizeof(__half2);
constexpr uint32_t kMetaFloats = 640;
constexpr uint32_t kSharedBytes =
    (kQHalf2 + kCacheHalf2) * sizeof(__half2)
    + kScratchBytes + kMetaFloats * sizeof(float);
constexpr float kNegInf = -3.402823466e+38F;
constexpr float kMaxOffset = 3.0f * 0.6931f;

template <uint32_t ColumnParts>
__device__ __forceinline__ void prefetch_cache_partition(
        const __half * cache, __half2 * dst, uint32_t group,
        uint32_t kvh, uint32_t kv_width, uint32_t valid_span,
        uint32_t dim_base, uint32_t partition, uint32_t local_column_tile,
        uint32_t lane, uint32_t ring, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Four warps consume each independent 16-key partition. Let those warps
    // publish exactly the rows they consume, then synchronize only that
    // 128-thread group. This preserves the cooperative 16-byte copy geometry
    // without forcing the other partition through eight CTA barriers per key
    // batch.
    for (uint32_t sector = local_column_tile * 32 + lane;
         sector < 16 * 32; sector += (4 / ColumnParts) * 32) {
        const uint32_t key_row = partition * 16 + (sector >> 5);
        const uint32_t row_sector = sector & 31;
        const uint32_t key = group * 32 + key_row;
        const uint32_t physical_key =
            imparo_cuda_kv::physical_row(key, ring, page_table);
        const __half * src = key < valid_span
            ? cache + uint64_t(physical_key) * kv_width + kvh * kHeadDim
                + dim_base + row_sector * 8
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
    (void)kv_width; (void)valid_span; (void)dim_base;
    (void)partition; (void)local_column_tile; (void)lane;
    (void)ring; (void)page_table;
#endif
}

template <uint32_t ColumnParts>
__device__ __forceinline__ void sync_cache_partition(uint32_t partition) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t barrier = 1 + partition;
    asm volatile("bar.sync %0, %1;" :: "r"(barrier),
        "n"(128 / ColumnParts) : "memory");
#else
    (void)partition;
#endif
}

__device__ __forceinline__ void sync_column_pair(uint32_t column_tile) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Each column tile is owned by one even/odd warp pair. Distinct named
    // barriers let the four pairs publish and combine their accumulator tiles
    // independently instead of serializing the whole CTA.
    const uint32_t barrier = 3 + column_tile;
    asm volatile("bar.sync %0, 64;" :: "r"(barrier) : "memory");
#else
    (void)column_tile;
#endif
}

__device__ __forceinline__ float masked_score(
        float score, uint32_t qcol, uint32_t key, uint32_t query_tile,
        uint32_t query_tokens, uint32_t n_tok, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span) {
    const uint32_t token = query_tile * query_tokens
        + qcol / imparo_sm80_prefill::kGqaHeads;
    const uint32_t pos = start_pos + token;
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

template <uint32_t OutputParts, uint32_t ColumnParts>
__global__ __launch_bounds__(
    256 / ColumnParts,
    (OutputParts > ColumnParts ? OutputParts : ColumnParts)) void flash(
        const __half * q, const __half * kc, const __half * vc,
        float * workspace, float * out, uint32_t block_base,
        uint32_t query_tiles, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos, uint32_t window,
        uint32_t n_tok, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t segment_slots, uint32_t query_tokens,
        uint32_t canonical_parts, uint32_t virtual_stream_blocks,
        uint64_t block_stride, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    __half2 * q_tile = reinterpret_cast<__half2 *>(shared_bytes);
    __half2 * cache_tile = q_tile + kQHalf2;
    unsigned char * scratch_bytes = reinterpret_cast<unsigned char *>(
        cache_tile + kCacheHalf2);
    __half2 * p_tile = reinterpret_cast<__half2 *>(scratch_bytes);
    __half * c_tile = reinterpret_cast<__half *>(scratch_bytes);
    float * meta = reinterpret_cast<float *>(scratch_bytes + kScratchBytes);
    float * part_max = meta;
    float * part_sum = part_max + 2 * kColumns;
    float * part_scale0 = part_sum + 2 * kColumns;
    float * part_scale1 = part_scale0 + kColumns;
    float * segment_max_shared = part_scale1 + kColumns;
    float * segment_sum_shared = segment_max_shared + kColumns;
    float * combine_value = segment_sum_shared + kColumns;
    float * combine_add = combine_value + kColumns;

    static_assert(OutputParts == 1 || OutputParts == 2,
        "D512 fused output partition count must be one or two");
    static_assert(ColumnParts == 1 || ColumnParts == 2,
        "D512 fused column partition count must be one or two");
    constexpr uint32_t kLocalOutputTiles = kOutputTiles / OutputParts;
    const uint32_t launch_block = blockIdx.x;
    const uint32_t local_block =
        launch_block / (OutputParts * ColumnParts);
    const uint32_t output_part =
        (launch_block / ColumnParts) % OutputParts;
    const uint32_t column_part = launch_block % ColumnParts;
    const uint32_t output_tile_base = output_part * kLocalOutputTiles;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp & 1;
    const uint32_t local_column_tile = warp >> 1;
    const uint32_t column_tile =
        column_part * (4 / ColumnParts) + local_column_tile;
    const uint32_t subgroup = lane >> 2;
    const uint32_t sublane = lane & 3;
    const uint32_t subgroup_mask = 0x0000000fu << (4 * subgroup);
    const uint32_t subgroup_base = 4 * subgroup;

    // Q is already converted and scaled by cache_scaled_q. Each even/odd warp
    // pair publishes only its own 16 query columns, so independent column tiles
    // can leave the initial CTA barrier behind without changing copy geometry.
    const uint32_t pair_thread = partition * 32 + lane;
    for (uint32_t sector = pair_thread; sector < 16 * 64; sector += 64) {
        const uint32_t qcol = column_tile * 16 + (sector >> 6);
        const uint32_t row_sector = sector & 63;
        const uint32_t token = query_tile * query_tokens
            + qcol / imparo_sm80_prefill::kGqaHeads;
        const uint32_t head = kvh * imparo_sm80_prefill::kGqaHeads
            + qcol % imparo_sm80_prefill::kGqaHeads;
        const __half * src = token < n_tok && head < n_heads
            ? q + (uint64_t(token) * n_heads + head) * kHeadDim
                + row_sector * 8
            : q;
        __half2 * target = q_tile + qcol * kQStride + row_sector * 4;
        const uint32_t shared = static_cast<uint32_t>(
            __cvta_generic_to_shared(target));
        const uint32_t bytes = token < n_tok && head < n_heads ? 16u : 0u;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(shared), "l"(src), "r"(bytes));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    sync_column_pair(column_tile);

    float * partial_num = workspace
        + uint64_t(local_block * OutputParts + output_part) * block_stride;
    float * partial_max = partial_num + uint64_t(kColumns) * kHeadDim;
    float * partial_sum = partial_max + kColumns;
    const uint32_t logical_blocks = n_kv * query_tiles;
    const uint32_t absolute_query_start =
        start_pos + query_tile * query_tokens;
    auto segment_bounds = [&](uint32_t slot, uint32_t * begin,
                              uint32_t * end) {
        return virtual_stream_blocks > 0
            ? imparo_sm80_prefill::stable_virtual_stream_segment_bounds(
                absolute_query_start, kvh, n_kv, query_tokens, ring,
                schedule_groups, virtual_stream_blocks, slot, begin, end)
            : (canonical_parts > 0
                ? imparo_sm80_prefill::canonical_segment_bounds(
                    absolute_query_start, schedule_groups, canonical_parts,
                    slot, begin, end)
                : imparo_sm80_prefill::stream_segment_bounds(
                    logical_block, logical_blocks, schedule_groups,
                    physical_blocks, slot, begin, end));
    };
    uint32_t valid_segments = 0;
    for (uint32_t slot = 0; slot < segment_slots; ++slot) {
        uint32_t begin = 0;
        uint32_t end = 0;
        valid_segments += segment_bounds(slot, &begin, &end) ? 1u : 0u;
    }

    Half16x8 numerator[kLocalOutputTiles];
    uint32_t segment_index = 0;
    for (uint32_t slot_rev = segment_slots; slot_rev > 0; --slot_rev) {
        const uint32_t slot = slot_rev - 1;
        uint32_t group_begin = 0;
        uint32_t group_end = 0;
        if (!segment_bounds(slot, &group_begin, &group_end)) {
            continue;
        }
#pragma unroll
        for (uint32_t tile = 0; tile < kLocalOutputTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                numerator[tile].x[l] = __float2half2_rn(0.0f);
            }
        }
        float running_max[2] = {kNegInf * 0.5f, kNegInf * 0.5f};
        float running_sum[2] = {0.0f, 0.0f};

        for (uint32_t group = group_begin; group < group_end; ++group) {
            Float16x16 scores{};
            // D512 wide FA accumulates the upper 256 dimensions first.
            prefetch_cache_partition<ColumnParts>(
                kc, cache_tile, group, kvh, kv_width,
                valid_span, 256, partition, local_column_tile,
                lane, ring, page_table);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            sync_cache_partition<ColumnParts>(partition);
#pragma unroll
            for (uint32_t local_d = 0; local_d < kCacheChunk; local_d += 16) {
                Half16x8 q_fragment;
                Half16x8 k_fragment;
                load_half16x8(q_fragment,
                    q_tile + (column_tile * 16) * kQStride
                        + (256 + local_d) / 2,
                    kQStride, lane);
                load_half16x8(k_fragment,
                    cache_tile + (partition * 16) * kCacheStride
                        + local_d / 2,
                    kCacheStride, lane);
                mma_qk(scores, q_fragment, k_fragment);
            }
            sync_cache_partition<ColumnParts>(partition);
            prefetch_cache_partition<ColumnParts>(
                kc, cache_tile, group, kvh, kv_width,
                valid_span, 0, partition, local_column_tile,
                lane, ring, page_table);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            sync_cache_partition<ColumnParts>(partition);
#pragma unroll
            for (uint32_t local_d = 0; local_d < kCacheChunk; local_d += 16) {
                Half16x8 q_fragment;
                Half16x8 k_fragment;
                load_half16x8(q_fragment,
                    q_tile + (column_tile * 16) * kQStride + local_d / 2,
                    kQStride, lane);
                load_half16x8(k_fragment,
                    cache_tile + (partition * 16) * kCacheStride
                        + local_d / 2,
                    kCacheStride, lane);
                mma_qk(scores, q_fragment, k_fragment);
            }
            sync_cache_partition<ColumnParts>(partition);
            // The lower K chunk is dead. Start V0 before mask/softmax so its
            // transfer is hidden by exponentiation and numerator rescaling.
            prefetch_cache_partition<ColumnParts>(
                vc, cache_tile, group, kvh, kv_width,
                valid_span, output_tile_base * 16, partition,
                local_column_tile, lane, ring, page_table);
            asm volatile("cp.async.commit_group;");

#pragma unroll
            for (uint32_t l = 0; l < 8; ++l) {
                const uint32_t qcol = column_tile * 16
                    + fragment_q_column(lane, l);
                const uint32_t key = group * 32 + partition * 16
                    + fragment_key_row(lane, l);
                scores.x[l] = masked_score(scores.x[l], qcol, key,
                    query_tile, query_tokens, n_tok, start_pos,
                    window, ring, valid_span);
            }
            float next_max[2] = {running_max[0], running_max[1]};
#pragma unroll
            for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
                const uint32_t l0 = 2 * qhalf;
                if (scores.x[l0] > -3.0e38F) {
                    next_max[qhalf] = fmaxf(next_max[qhalf],
                        scores.x[l0] + kMaxOffset);
                }
                if (scores.x[l0 + 1] > -3.0e38F) {
                    next_max[qhalf] = fmaxf(next_max[qhalf],
                        scores.x[l0 + 1] + kMaxOffset);
                }
                if (scores.x[l0 + 4] > -3.0e38F) {
                    next_max[qhalf] = fmaxf(next_max[qhalf],
                        scores.x[l0 + 4] + kMaxOffset);
                }
                if (scores.x[l0 + 5] > -3.0e38F) {
                    next_max[qhalf] = fmaxf(next_max[qhalf],
                        scores.x[l0 + 5] + kMaxOffset);
                }
#pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    next_max[qhalf] = fmaxf(next_max[qhalf],
                        __shfl_xor_sync(subgroup_mask,
                            next_max[qhalf], offset));
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
                for (uint32_t tile = 0; tile < kLocalOutputTiles; ++tile) {
#pragma unroll
                    for (uint32_t l = 0; l < 4; ++l) {
                        numerator[tile].x[l] = __hmul2(
                            numerator[tile].x[l],
                            __float2half2_rn(rescale[l & 1]));
                    }
                }
            }
#pragma unroll
            for (uint32_t qhalf = 0; qhalf < 2; ++qhalf) {
                const uint32_t l0 = 2 * qhalf;
                const float p0 = scores.x[l0] > -3.0e38F
                    ? expf(scores.x[l0] - next_max[qhalf]) : 0.0f;
                const float p1 = scores.x[l0 + 1] > -3.0e38F
                    ? expf(scores.x[l0 + 1] - next_max[qhalf]) : 0.0f;
                const float p2 = scores.x[l0 + 4] > -3.0e38F
                    ? expf(scores.x[l0 + 4] - next_max[qhalf]) : 0.0f;
                const float p3 = scores.x[l0 + 5] > -3.0e38F
                    ? expf(scores.x[l0 + 5] - next_max[qhalf]) : 0.0f;
                probability_add[qhalf] += p0;
                probability_add[qhalf] += p1;
                probability_add[qhalf] += p2;
                probability_add[qhalf] += p3;
                const uint32_t qrow = qhalf * 8 + subgroup;
                p_tile[(warp * 16 + qrow) * 8 + sublane] =
                    __floats2half2_rn(p0, p1);
                p_tile[(warp * 16 + qrow) * 8 + 4 + sublane] =
                    __floats2half2_rn(p2, p3);
            }

            asm volatile("cp.async.wait_group 0;");
            sync_cache_partition<ColumnParts>(partition);
            Half16x8 p_fragment;
            load_half16x8(p_fragment, p_tile + warp * 16 * 8, 8, lane);
#pragma unroll
            for (uint32_t output_tile = 0; output_tile < 16; ++output_tile) {
                Half16x8 v_fragment;
                load_half16x8_trans(v_fragment,
                    cache_tile + (partition * 16) * kCacheStride
                        + output_tile * 8,
                    kCacheStride, lane);
                mma_pv(numerator[output_tile], p_fragment, v_fragment);
            }
            sync_cache_partition<ColumnParts>(partition);
            if constexpr (OutputParts == 1) {
                prefetch_cache_partition<ColumnParts>(
                    vc, cache_tile, group, kvh, kv_width,
                    valid_span, 256, partition, local_column_tile,
                    lane, ring, page_table);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
                sync_cache_partition<ColumnParts>(partition);
#pragma unroll
                for (uint32_t output_tile = 0; output_tile < 16; ++output_tile) {
                    Half16x8 v_fragment;
                    load_half16x8_trans(v_fragment,
                        cache_tile + (partition * 16) * kCacheStride
                            + output_tile * 8,
                        kCacheStride, lane);
                    mma_pv(numerator[16 + output_tile], p_fragment, v_fragment);
                }
                sync_cache_partition<ColumnParts>(partition);
            }
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
            if (sublane == 0) {
                const uint32_t qcol = column_tile * 16 + qhalf * 8 + subgroup;
                part_max[partition * kColumns + qcol] = running_max[qhalf];
                part_sum[partition * kColumns + qcol] = sum;
            }
        }
        sync_column_pair(column_tile);
        if (partition == 0 && lane < 16) {
            const uint32_t qcol = column_tile * 16 + lane;
            const float max0 = part_max[qcol];
            const float max1 = part_max[kColumns + qcol];
            const float max_value = fmaxf(max0, max1);
            const float scale0 = max0 - max_value >= -20.0f
                ? expf(max0 - max_value) : 0.0f;
            const float scale1 = max1 - max_value >= -20.0f
                ? expf(max1 - max_value) : 0.0f;
            part_scale0[qcol] = scale0;
            part_scale1[qcol] = scale1;
            segment_max_shared[qcol] = max_value;
            segment_sum_shared[qcol] = __fadd_rn(
                __fmul_rn(scale0, part_sum[qcol]),
                __fmul_rn(scale1, part_sum[kColumns + qcol]));
            if (segment_index == 0 && valid_segments > 1) {
                partial_max[qcol] = max_value;
                partial_sum[qcol] = segment_sum_shared[qcol];
            } else if (segment_index > 0) {
                const float first_max = partial_max[qcol];
                const float combined_max = fmaxf(first_max, max_value);
                const float scale_value = first_max - combined_max >= -20.0f
                    ? expf(first_max - combined_max) : 0.0f;
                const float scale_add = max_value - combined_max >= -20.0f
                    ? expf(max_value - combined_max) : 0.0f;
                combine_value[qcol] = scale_value;
                combine_add[qcol] = scale_add;
                segment_sum_shared[qcol] = fmaf(scale_value,
                    partial_sum[qcol], scale_add * segment_sum_shared[qcol]);
            }
        }
        sync_column_pair(column_tile);

#pragma unroll
        for (uint32_t output_tile = 0;
             output_tile < kLocalOutputTiles; ++output_tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t qrow = half_acc_query_column(lane, l);
                const uint32_t output_pair = half_acc_output_pair(lane, l);
                const uint32_t index = (warp * 16 + qrow) * 16
                    + 2 * output_pair;
                c_tile[index] = __low2half(numerator[output_tile].x[l]);
                c_tile[index + 1] = __high2half(numerator[output_tile].x[l]);
            }
            sync_column_pair(column_tile);
            if (partition == 0) {
                for (uint32_t e = lane; e < 16 * 16; e += 32) {
                    const uint32_t row = e & 15;
                    const uint32_t col = e >> 4;
                    const uint32_t qcol = column_tile * 16 + col;
                    const uint32_t c0 = (warp * 16 + col) * 16 + row;
                    const uint32_t c1 = ((warp + 1) * 16 + col) * 16 + row;
                    float value = fmaf(part_scale0[qcol],
                        __half2float(c_tile[c0]), 0.0f);
                    value = fmaf(part_scale1[qcol],
                        __half2float(c_tile[c1]), value);
                    const uint32_t output =
                        (output_tile_base + output_tile) * 16 + row;
                    if (segment_index == 0 && valid_segments > 1) {
                        partial_num[uint64_t(qcol) * kHeadDim + output] = value;
                    } else {
                        if (segment_index > 0) {
                            value = fmaf(combine_value[qcol],
                                partial_num[uint64_t(qcol) * kHeadDim + output],
                                combine_add[qcol] * value);
                        }
                        const uint32_t token = query_tile * query_tokens
                            + qcol / imparo_sm80_prefill::kGqaHeads;
                        const uint32_t head = kvh * imparo_sm80_prefill::kGqaHeads
                            + qcol % imparo_sm80_prefill::kGqaHeads;
                        if (token < n_tok && head < n_heads) {
                            const float denominator = segment_sum_shared[qcol];
                            out[(uint64_t(token) * n_heads + head) * kHeadDim
                                + output] = denominator > 0.0f
                                ? value / denominator : 0.0f;
                        }
                    }
                }
            }
            sync_column_pair(column_tile);
        }
        ++segment_index;
    }
#else
    (void)q; (void)kc; (void)vc; (void)workspace; (void)out;
    (void)block_base; (void)query_tiles; (void)n_heads; (void)n_kv;
    (void)kv_width; (void)start_pos; (void)window; (void)n_tok;
    (void)ring; (void)valid_span; (void)schedule_groups;
    (void)physical_blocks; (void)segment_slots; (void)query_tokens;
    (void)canonical_parts; (void)virtual_stream_blocks;
    (void)block_stride; (void)page_table;
#endif
}

template <uint32_t OutputParts, uint32_t ColumnParts>
inline bool configure() {
    static const bool configured = cudaFuncSetAttribute(
        flash<OutputParts, ColumnParts>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(kSharedBytes)) == cudaSuccess;
    return configured;
}

} // namespace imparo_sm80_fa_d512
