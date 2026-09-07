#pragma once

#include "mma_f16.cuh"

// SM80+ staged prefill attention for Gemma 4's D=256/D=512, GQA=4 layers.
//
// The common backend owns policy and bounded allocation; this architecture file owns
// only tile geometry. Logical tiles select 8 or 16 query tokens at dispatch time,
// matching the reference FA's 32- and 64-column layouts without duplicating kernels.
// Scores are staged in bounded chunks so workspace is O(chunk * context), not
// O(batch * context), which keeps long-context and model-larger-than-VRAM operation
// viable.

namespace imparo_sm80_prefill {

constexpr uint32_t kMediumQueryTokens = 8;
constexpr uint32_t kWideQueryTokens = 16;
constexpr uint32_t kQueryTokens = kWideQueryTokens;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kMaxColumns = kWideQueryTokens * kGqaHeads;
constexpr uint32_t kColumns = kMaxColumns;
constexpr uint32_t kKeyBatch = 32;
constexpr uint32_t kScheduleKeys = 256;
// Numerical-route identity for canonical SM80 D512 prefill. A query belongs to
// an absolute 512-token scheduling cell and that cell's visible key domain is
// split into two stable parts. Caller batch width and sibling query tiles never
// move the boundary for the same absolute query.
constexpr uint32_t kCanonicalPrefillCellTokens = 512;
constexpr uint32_t kCanonicalD512Parts = 2;

enum class D512PartitionPolicy : uint32_t {
    PhysicalStream,
    StableCell2,
};

struct D512AttentionGeometry {
    uint32_t valid_keys;
    uint32_t physical_keys;
    uint32_t query_tokens;
    uint32_t schedule_groups;
    D512PartitionPolicy partition_policy;
};

__host__ __device__ constexpr uint32_t d512_physical_keys(
        uint32_t valid_keys) {
    const uint32_t padded = uint32_t(
        (uint64_t(valid_keys) + kScheduleKeys - 1) / kScheduleKeys
        * kScheduleKeys);
    return padded < kScheduleKeys ? kScheduleKeys : padded;
}

__host__ __device__ constexpr uint32_t d512_query_tokens(uint32_t n_tok) {
    return n_tok <= 2 ? 2u
        : (n_tok <= 4 ? 4u : (n_tok <= 8 ? 8u : 16u));
}

__host__ __device__ constexpr bool d512_staged_tile_supported(
        uint32_t n_tok) {
    return n_tok >= 3 && d512_query_tokens(n_tok) >= 4;
}

__host__ __device__ constexpr D512AttentionGeometry d512_geometry(
        uint32_t valid_keys, uint32_t n_tok,
        D512PartitionPolicy partition_policy) {
    const uint32_t physical_keys = d512_physical_keys(valid_keys);
    return {
        valid_keys,
        physical_keys,
        d512_query_tokens(n_tok),
        physical_keys / kKeyBatch,
        partition_policy,
    };
}

static_assert(d512_physical_keys(1) == 256);
static_assert(d512_physical_keys(128) == 256);
static_assert(d512_physical_keys(256) == 256);
static_assert(d512_physical_keys(257) == 512);
static_assert(d512_physical_keys(512) == 512);
static_assert(d512_physical_keys(513) == 768);
static_assert(d512_physical_keys(700) == 768);
static_assert(d512_physical_keys(768) == 768);
static_assert(d512_physical_keys(769) == 1024);
static_assert(d512_physical_keys(1996) == 2048);
static_assert(d512_physical_keys(2000) == 2048);
static_assert(d512_query_tokens(1) == 2);
static_assert(d512_query_tokens(2) == 2);
static_assert(d512_query_tokens(3) == 4);
static_assert(d512_query_tokens(4) == 4);
static_assert(d512_query_tokens(5) == 8);
static_assert(d512_query_tokens(8) == 8);
static_assert(d512_query_tokens(9) == 16);
static_assert(d512_query_tokens(2000) == 16);
static_assert(!d512_staged_tile_supported(0));
static_assert(!d512_staged_tile_supported(1));
static_assert(!d512_staged_tile_supported(2));
static_assert(d512_staged_tile_supported(3));
static_assert(d512_staged_tile_supported(4));
static_assert(d512_staged_tile_supported(8));
static_assert(d512_staged_tile_supported(16));
constexpr D512AttentionGeometry kGeometryContract =
    d512_geometry(700, 8, D512PartitionPolicy::StableCell2);
static_assert(kGeometryContract.valid_keys == 700);
static_assert(kGeometryContract.physical_keys == 768);
static_assert(kGeometryContract.query_tokens == 8);
static_assert(kGeometryContract.schedule_groups == 24);
static_assert(kGeometryContract.partition_policy
    == D512PartitionPolicy::StableCell2);

// Grouping reuses one staged Q fragment across independent key groups. Keep the
// policy in the architecture layer: register pressure and CTA residency vary by
// SM family, while the common dispatcher only supplies an optional tuned value.
inline uint32_t select_score_key_groups(
        uint32_t sm_version, uint32_t tuned) {
    if (tuned == 1 || tuned == 2 || tuned == 4) return tuned;
    return sm_version == 86 ? 4u : 1u;
}

// Ring-cache attention scans physical slots, but masking is expressed in absolute
// positions. Map each resident slot to its newest position at the end of this batch.
__device__ __forceinline__ uint32_t physical_key_position(
        uint32_t key, uint32_t ring, uint32_t end_pos) {
    if (ring == 0) return key;
    // `ring` is the mask produced by the common workflow, so ring + 1 is a
    // power of two. Recover the newest absolute cycle with the same mask rather
    // than issuing an integer divide for every score element.
    return key + ((end_pos - key) & ~ring);
}

__host__ __device__ constexpr uint32_t canonical_active_groups(
        uint32_t absolute_query_start) {
    const uint32_t cell = absolute_query_start / kCanonicalPrefillCellTokens;
    return (cell + 1) * (kCanonicalPrefillCellTokens / kKeyBatch);
}

__host__ __device__ constexpr uint32_t canonical_partition_begin(
        uint32_t active_groups, uint32_t parts, uint32_t segment) {
    return uint32_t(uint64_t(segment) * active_groups / parts);
}

__host__ __device__ constexpr uint32_t canonical_partition_end(
        uint32_t active_groups, uint32_t parts, uint32_t segment) {
    return uint32_t(uint64_t(segment + 1) * active_groups / parts);
}

struct D512GroupRange {
    uint32_t begin;
    uint32_t end;
};

__host__ __device__ constexpr D512GroupRange stable_cell_partition(
        uint32_t active_groups, uint32_t schedule_groups,
        uint32_t parts, uint32_t segment) {
    const uint32_t raw_begin = canonical_partition_begin(
        active_groups, parts, segment);
    const uint32_t raw_end = canonical_partition_end(
        active_groups, parts, segment);
    return {
        raw_begin < schedule_groups ? raw_begin : schedule_groups,
        raw_end < schedule_groups ? raw_end : schedule_groups,
    };
}

static_assert(canonical_active_groups(0) == 16);
static_assert(canonical_active_groups(511) == 16);
static_assert(canonical_active_groups(512) == 32);
static_assert(canonical_partition_begin(64, 2, 1) == 32);
static_assert(canonical_partition_end(64, 2, 1) == 64);
static_assert(stable_cell_partition(32, 24, 2, 0).begin == 0);
static_assert(stable_cell_partition(32, 24, 2, 0).end == 16);
static_assert(stable_cell_partition(32, 24, 2, 1).begin == 16);
static_assert(stable_cell_partition(32, 24, 2, 1).end == 24);
static_assert(stable_cell_partition(16, 8, 2, 0).begin == 0);
static_assert(stable_cell_partition(16, 8, 2, 0).end == 8);
static_assert(stable_cell_partition(16, 8, 2, 1).begin == 8);
static_assert(stable_cell_partition(16, 8, 2, 1).end == 8);

__device__ __forceinline__ bool canonical_segment_bounds(
        uint32_t absolute_query_start, uint32_t schedule_groups,
        uint32_t parts, uint32_t segment,
        uint32_t * group_begin, uint32_t * group_end) {
    if (parts == 0 || segment >= parts) return false;
    const D512GroupRange range = stable_cell_partition(
        canonical_active_groups(absolute_query_start),
        schedule_groups, parts, segment);
    if (range.begin >= range.end) return false;
    *group_begin = range.begin;
    *group_end = range.end;
    return true;
}
__device__ __forceinline__ bool stream_segment_bounds(
        uint32_t logical_block, uint32_t logical_blocks,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t segment, uint32_t * group_begin, uint32_t * group_end) {
    const uint64_t total_work = uint64_t(logical_blocks) * schedule_groups;
    const uint64_t tile_begin = uint64_t(logical_block) * schedule_groups;
    const uint64_t tile_end = tile_begin + schedule_groups;
    uint32_t first = uint32_t(tile_begin * physical_blocks / total_work);
    while (first < physical_blocks
            && uint64_t(first + 1) * total_work / physical_blocks <= tile_begin) {
        ++first;
    }
    const uint32_t block = first + segment;
    if (block >= physical_blocks) return false;
    const uint64_t block_begin = uint64_t(block) * total_work / physical_blocks;
    const uint64_t block_end = uint64_t(block + 1) * total_work / physical_blocks;
    const uint64_t begin = tile_begin > block_begin ? tile_begin : block_begin;
    const uint64_t end = tile_end < block_end ? tile_end : block_end;
    if (begin >= end) return false;
    *group_begin = uint32_t(begin - tile_begin);
    *group_end = uint32_t(end - tile_begin);
    return true;
}

// Reproduce the reference Stream-K partition inside an absolute 512-token
// scheduling cell. A caller may submit the whole cell or an aligned suffix;
// deriving the virtual tile from the absolute query position keeps the seam
// identical for cold, split, and resumed execution.
__device__ __forceinline__ bool stable_virtual_stream_segment_bounds(
        uint32_t absolute_query_start, uint32_t kv_head, uint32_t n_kv,
        uint32_t query_tokens, uint32_t ring, uint32_t runtime_schedule_groups,
        uint32_t virtual_physical_blocks, uint32_t segment,
        uint32_t * group_begin, uint32_t * group_end) {
    if (n_kv == 0 || query_tokens == 0 || virtual_physical_blocks == 0
        || kCanonicalPrefillCellTokens % query_tokens != 0
        || absolute_query_start % query_tokens != 0) {
        return false;
    }
    const uint32_t virtual_query_tiles =
        kCanonicalPrefillCellTokens / query_tokens;
    const uint32_t query_in_cell =
        (absolute_query_start % kCanonicalPrefillCellTokens) / query_tokens;
    if (query_in_cell >= virtual_query_tiles || kv_head >= n_kv) return false;
    const uint32_t virtual_logical_blocks = virtual_query_tiles * n_kv;
    const uint32_t virtual_logical_block =
        kv_head * virtual_query_tiles + query_in_cell;
    uint32_t virtual_schedule_groups = canonical_active_groups(absolute_query_start);
    if (ring > 0) {
        const uint32_t ring_capacity_groups =
            ((ring + 1 + kScheduleKeys - 1) / kScheduleKeys)
            * (kScheduleKeys / kKeyBatch);
        virtual_schedule_groups = min(virtual_schedule_groups, ring_capacity_groups);
    }
    if (virtual_schedule_groups == 0 || runtime_schedule_groups == 0
        || !stream_segment_bounds(
            virtual_logical_block, virtual_logical_blocks, virtual_schedule_groups,
            virtual_physical_blocks, segment, group_begin, group_end)) {
        return false;
    }
    *group_end = min(*group_end, runtime_schedule_groups);
    *group_begin = min(*group_begin, *group_end);
    return *group_begin < *group_end;
}

__host__ __device__ constexpr uint32_t stream_segment_slot_bound(
        uint32_t logical_blocks, uint32_t physical_blocks) {
    return logical_blocks == 0 || physical_blocks == 0 ? 0u
        : uint32_t((uint64_t(physical_blocks) + logical_blocks - 1)
            / logical_blocks) + 1u;
}
static_assert(stream_segment_slot_bound(16, 15) == 2);
static_assert(stream_segment_slot_bound(16, 16) == 2);
static_assert(stream_segment_slot_bound(16, 17) == 3);
static_assert(stream_segment_slot_bound(16, 56) == 5);
static_assert(stream_segment_slot_bound(64, 56) == 2);
static_assert(stream_segment_slot_bound(58, 56) == 2);

// Reproduce the reference Stream-K seam for a logical attention tile. When the
// physical grid has no more blocks than output tiles, each tile has at most one
// interior seam. Returning `groups` means the tile is whole.
__device__ __forceinline__ uint32_t stream_seam_group(
        uint32_t logical_block, uint32_t logical_blocks,
        uint32_t groups, uint32_t stream_blocks) {
    if (stream_blocks == 0 || stream_blocks > logical_blocks) return groups;
    const uint64_t total_work = uint64_t(logical_blocks) * groups;
    const uint64_t tile_start = uint64_t(logical_block) * groups;
    const uint64_t tile_stop = tile_start + groups;
    uint32_t block = uint32_t(tile_start * stream_blocks / total_work) + 1;
    while (block < stream_blocks) {
        const uint64_t boundary = uint64_t(block) * total_work / stream_blocks;
        if (boundary > tile_start) {
            return boundary < tile_stop ? uint32_t(boundary - tile_start) : groups;
        }
        ++block;
    }
    return groups;
}

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

struct IdentityKvRows {
    __device__ __forceinline__ uint32_t physical(uint32_t logical) const {
        return logical;
    }
};

struct PagedKvRows {
    const uint32_t * table;
    uint32_t ring;
    __device__ __forceinline__ uint32_t physical(uint32_t logical) const {
        return imparo_cuda_kv::physical_row(logical, ring, table);
    }
};

// Eight warps cover (2 key tiles) x (4 query-column tiles). Each warp computes one
// 16x16 Q*K^T tile with f16 inputs and an f32 accumulator. Keep the native fragment
// lane layout through the score write: materializing WMMA as row-major changes the
// column-specific cancellation used by the reference FA softmax.
__global__ void cache_scaled_q(
        const float * src, __half2 * dst, uint64_t pairs, float scale) {
    const uint64_t pair = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const float2 value = reinterpret_cast<const float2 *>(src)[pair];
    dst[pair] = __hmul2(__floats2half2_rn(value.x, value.y),
        __float2half2_rn(scale));
}

template <uint32_t HeadDim, bool LoadQuery = true, typename KvRows>
__device__ __forceinline__ void prefetch_score_qk_tile(
        const __half * q, const __half * kc, __half2 * q_dst, __half2 * k_dst,
        uint32_t key0, uint32_t lane, uint32_t column_tile,
        uint32_t query_tile, uint32_t query_tokens, uint32_t n_tok,
        uint32_t n_heads, uint32_t kvh, uint32_t kv_width,
        uint32_t valid_span, uint32_t d0, KvRows kv_rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (LoadQuery) {
        const uint32_t q_row = lane >> 1;
        const uint32_t q_half = lane & 1;
        const uint32_t qcol = column_tile * 16 + q_row;
        const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
        const uint32_t head = kvh * kGqaHeads + qcol % kGqaHeads;
        const bool valid_query = token < n_tok && head < n_heads;
        const __half * q_src = valid_query
            ? q + (uint64_t(token) * n_heads + head) * HeadDim
                + d0 + q_half * 8
            : q;
        const uint32_t q_shared =
            static_cast<uint32_t>(__cvta_generic_to_shared(q_dst));
        const uint32_t q_bytes = valid_query ? 16u : 0u;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(q_shared), "l"(q_src), "r"(q_bytes));
    }

    // Every column warp in a partition consumes the same 16x16 K tile. Split its
    // 32 naturally aligned 16-byte sectors across the actual number of column
    // warps: medium (8-query) tiles have two, wide (16-query) tiles have four,
    // and the allocation fallback may reach the one-warp 4-query geometry.
    // A block barrier below makes every producer's cp.async result visible.
    const uint32_t column_warps = query_tokens * kGqaHeads / 16;
    const uint32_t sectors_per_warp = 32 / column_warps;
    if (lane < sectors_per_warp) {
        const uint32_t sector = column_tile * sectors_per_warp + lane;
        const uint32_t key_row = sector >> 1;
        const uint32_t key_half = sector & 1;
        const uint32_t key = key0 + key_row;
        const bool valid_key = key < valid_span;
        const uint32_t physical_key = kv_rows.physical(key);
        const __half * k_src = valid_key
            ? kc + uint64_t(physical_key) * kv_width + kvh * HeadDim
                + d0 + key_half * 8
            : kc;
        const uint32_t k_shared = static_cast<uint32_t>(
            __cvta_generic_to_shared(k_dst + key_row * 8 + key_half * 4));
        const uint32_t k_bytes = valid_key ? 16u : 0u;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(k_shared), "l"(k_src), "r"(k_bytes));
    }
#else
    (void)q; (void)kc; (void)q_dst; (void)k_dst; (void)key0; (void)lane;
    (void)column_tile; (void)query_tile; (void)query_tokens; (void)n_tok;
    (void)n_heads; (void)kvh; (void)kv_width; (void)valid_span; (void)d0;
    (void)kv_rows;
#endif
}

template <uint32_t HeadDim, uint32_t KeyGroupsPerCta, typename KvRows>
__device__ __forceinline__ void scores_body(
        const __half * q, const __half * kc, float * workspace,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t n_tok, uint32_t ring,
        uint32_t valid_span, uint32_t kv_span,
        uint32_t query_tokens, uint32_t columns,
        uint64_t block_stride, KvRows kv_rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t local_block = blockIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp & 1;
    const uint32_t column_tile = warp >> 1;
    static_assert(KeyGroupsPerCta == 1 || KeyGroupsPerCta == 2
        || KeyGroupsPerCta == 4);
    const uint32_t first_group = blockIdx.y * KeyGroupsPerCta;

    constexpr uint32_t stride = 8;
    __shared__ __align__(16) __half2 q_tile[2][8][16 * stride];
    __shared__ __align__(16) __half2
        k_tile[2][KeyGroupsPerCta][2][16 * stride];

    Float16x16 c[KeyGroupsPerCta]{};
    // The pinned Ampere D512 FA configurations use one 512-wide traversal for the
    // <=4-token tail and two reverse-ordered 256-wide traversals for wider queries.
    // D256 is one forward traversal. F32 MMA accumulation is not associative.
    const uint32_t iterations = HeadDim / 16;
    uint32_t k_stage = 0;
    const uint32_t first_d0 = HeadDim == 512 && n_tok > 4 ? 256u : 0u;
    prefetch_score_qk_tile<HeadDim, true>(q, kc,
        &q_tile[k_stage][warp][(lane >> 1) * stride + (lane & 1) * 4],
        k_tile[k_stage][0][partition],
        first_group * kKeyBatch + partition * 16, lane, column_tile,
        query_tile, query_tokens, n_tok, n_heads,
        kvh, kv_width, valid_span, first_d0, kv_rows);
#pragma unroll
    for (uint32_t key_group = 1; key_group < KeyGroupsPerCta; ++key_group) {
        prefetch_score_qk_tile<HeadDim, false>(q, kc,
            &q_tile[k_stage][warp][(lane >> 1) * stride + (lane & 1) * 4],
            k_tile[k_stage][key_group][partition],
            (first_group + key_group) * kKeyBatch + partition * 16,
            lane, column_tile, query_tile, query_tokens, n_tok, n_heads,
            kvh, kv_width, valid_span, first_d0, kv_rows);
    }
    asm volatile("cp.async.commit_group;");
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const uint32_t next_stage = k_stage ^ 1;
        if (iteration + 1 < iterations) {
            const uint32_t next_iteration = iteration + 1;
            const uint32_t next_d0 = HeadDim == 512 && n_tok > 4
                ? (next_iteration < 16 ? 256 + 16 * next_iteration
                    : 16 * (next_iteration - 16))
                : 16 * next_iteration;
            prefetch_score_qk_tile<HeadDim, true>(q, kc,
                &q_tile[next_stage][warp][(lane >> 1) * stride + (lane & 1) * 4],
                k_tile[next_stage][0][partition],
                first_group * kKeyBatch + partition * 16, lane, column_tile,
                query_tile, query_tokens, n_tok, n_heads,
                kvh, kv_width, valid_span, next_d0, kv_rows);
#pragma unroll
            for (uint32_t key_group = 1;
                 key_group < KeyGroupsPerCta; ++key_group) {
                prefetch_score_qk_tile<HeadDim, false>(q, kc,
                    &q_tile[next_stage][warp]
                        [(lane >> 1) * stride + (lane & 1) * 4],
                    k_tile[next_stage][key_group][partition],
                    (first_group + key_group) * kKeyBatch + partition * 16,
                    lane, column_tile, query_tile, query_tokens, n_tok, n_heads,
                    kvh, kv_width, valid_span, next_d0, kv_rows);
            }
            asm volatile("cp.async.commit_group;");
        }
        Half16x8 q_frag;
        Half16x8 k_frag;
        load_half16x8(q_frag, q_tile[k_stage][warp], stride, lane);
#pragma unroll
        for (uint32_t key_group = 0;
             key_group < KeyGroupsPerCta; ++key_group) {
            load_half16x8(
                k_frag, k_tile[k_stage][key_group][partition], stride, lane);
            mma_qk(c[key_group], q_frag, k_frag);
        }
        k_stage = next_stage;
    }

#pragma unroll
    for (uint32_t key_group = 0;
         key_group < KeyGroupsPerCta; ++key_group) {
        const uint32_t key0 =
            (first_group + key_group) * kKeyBatch + partition * 16;
#pragma unroll
        for (uint32_t l = 0; l < 8; ++l) {
            const uint32_t qcol = column_tile * 16 + fragment_q_column(lane, l);
            const uint32_t key = key0 + fragment_key_row(lane, l);
            const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
            const uint32_t pos = start_pos + token;
            const uint32_t lo =
                window > 0 && pos + 1 > window ? pos + 1 - window : 0;
            const uint32_t key_pos = key < valid_span
                ? physical_key_position(key, ring, start_pos + n_tok - 1) : 0;
            float value = -3.402823466e+38F;
            if (token < n_tok && key < valid_span
                && key_pos >= lo && key_pos <= pos) {
                value = c[key_group].x[l];
            }
            if (key < kv_span) {
                workspace[(uint64_t)local_block * block_stride
                    + (uint64_t)qcol * kv_span + key] = value;
            }
        }
    }
#else
    (void)q; (void)kc; (void)workspace; (void)block_base; (void)query_tiles;
    (void)n_heads; (void)n_kv; (void)kv_width; (void)start_pos; (void)window;
    (void)n_tok; (void)ring; (void)valid_span; (void)kv_span;
    (void)query_tokens; (void)columns;
    (void)block_stride; (void)kv_rows;
#endif
}

template <uint32_t HeadDim, uint32_t KeyGroupsPerCta = 1>
__global__ void scores(
        const __half * q, const __half * kc, float * workspace,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t n_tok, uint32_t ring,
        uint32_t valid_span, uint32_t kv_span, uint32_t query_tokens,
        uint32_t columns, uint64_t block_stride, const uint32_t * page_table) {
    scores_body<HeadDim, KeyGroupsPerCta>(q, kc, workspace, block_base,
        query_tiles, n_heads, n_kv, kv_width, start_pos, window, n_tok, ring,
        valid_span, kv_span, query_tokens, columns, block_stride,
        PagedKvRows{page_table, ring});
}

template <uint32_t KeyGroupsPerCta = 1>
__global__ void scores_d512(
        const __half * q, const __half * kc, float * workspace,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t n_tok, uint32_t ring,
        uint32_t valid_span, uint32_t kv_span, uint32_t query_tokens,
        uint32_t columns, uint64_t block_stride) {
    scores_body<512, KeyGroupsPerCta>(q, kc, workspace, block_base,
        query_tiles, n_heads, n_kv, kv_width, start_pos, window, n_tok, ring,
        valid_span, kv_span, query_tokens, columns, block_stride,
        IdentityKvRows{});
}

template <uint32_t KeyGroupsPerCta = 1>
__global__ void scores_d512_paged(
        const __half * q, const __half * kc, float * workspace,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t n_tok, uint32_t ring,
        uint32_t valid_span, uint32_t kv_span, uint32_t query_tokens,
        uint32_t columns, uint64_t block_stride, const uint32_t * page_table) {
    scores_body<512, KeyGroupsPerCta>(q, kc, workspace, block_base,
        query_tiles, n_heads, n_kv, kv_width, start_pos, window, n_tok, ring,
        valid_span, kv_span, query_tokens, columns, block_stride,
        PagedKvRows{page_table, ring});
}

// Convert one query column to the reference FA probability representation. Four lanes
// reproduce the MMA fragment's row ownership and xor(2,1) reduction. Earlier groups are
// rescaled in place when the online max changes, leaving one final half-PV operand matrix.
__global__ void softmax(float * workspace, uint32_t kv_span,
                        uint32_t block_base, uint32_t logical_blocks,
                        uint32_t stream_blocks, uint32_t columns) {
    const uint32_t local_block = blockIdx.x;
    const uint32_t partition = blockIdx.y & 1;
    const uint32_t column = blockIdx.y >> 1;
    const uint32_t lane = threadIdx.x;
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    const uint64_t block_stride = uint64_t(columns) * kv_span
        + uint64_t(2 * columns) * groups + 8 * columns;
    float * row = workspace + (uint64_t)local_block * block_stride
        + (uint64_t)column * kv_span;
    constexpr float max_offset = 3.0f * 0.6931f;
    const uint32_t seam = stream_seam_group(
        block_base + local_block, logical_blocks, groups, stream_blocks);
    const uint32_t segments = seam < groups ? 2u : 1u;
    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t group_begin = segment == 0 ? 0 : seam;
        const uint32_t group_end = segment == 0 && segments == 2 ? seam : groups;
        float running_max = -3.402823466e+38F;
        float partial_sum = 0.0f;
        for (uint32_t group = group_begin; group < group_end; ++group) {
            const uint32_t gb = group * kKeyBatch;
            float next_max = running_max;
            if (lane < 4) {
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t key = gb + partition * 16 + half8 + 2 * lane + pair;
                        if (key < kv_span && row[key] > -3.0e38F) {
                            next_max = fmaxf(next_max, row[key] + max_offset);
                        }
                    }
                }
            }
            if (lane < 4) {
#pragma unroll
                for (int offset = 2; offset > 0; offset >>= 1) {
                    next_max = fmaxf(next_max,
                        __shfl_xor_sync(0x0000000f, next_max, offset));
                }
            }
            next_max = __shfl_sync(0xffffffff, next_max, 0);
            const float diff = running_max - next_max;
            float rescale = expf(diff);
            if (diff < -20.0f) rescale = 0.0f;
            if (lane == 0) {
                workspace[(uint64_t)local_block * block_stride +
                    (uint64_t)columns * kv_span +
                    ((uint64_t)partition * columns + column) * groups + group] = rescale;
            }

            if (lane < 4) {
                float add = 0.0f;
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t key = gb + partition * 16 + half8 + 2 * lane + pair;
                        if (key < kv_span && row[key] > -3.0e38F) {
                            const float p = expf(row[key] - next_max);
                            row[key] = p;
                            add += p;
                        } else if (key < kv_span) {
                            row[key] = 0.0f;
                        }
                    }
                }
                partial_sum = partial_sum * rescale + add;
            }
            running_max = next_max;
            __syncwarp();
        }
        if (lane < 4) {
            float sum = partial_sum;
#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0x0000000f, sum, offset);
            }
            if (lane == 0) {
                const uint64_t meta = (uint64_t)local_block * block_stride
                    + (uint64_t)columns * kv_span + uint64_t(2 * columns) * groups
                    + uint64_t(segment) * 4 * columns
                    + (uint64_t)partition * 2 * columns;
                workspace[meta + column] = running_max;
                workspace[meta + columns + column] = sum;
            }
        }
    }
}

// D512 assigns the two 16-key halves of every 32-key group to independent
// probability fragments. Keep that partitioning while using the same physical
// Stream-K schedule as the reference kernel. The per-partition metadata is
// combined only after the half-PV MMA, preserving the reference's f32 boundary.
__global__ void softmax_d512_stream(
        float * workspace, uint32_t kv_span, uint64_t block_stride,
        uint32_t block_base, uint32_t query_tiles, uint32_t start_pos,
        uint32_t ring, uint32_t logical_blocks, uint32_t schedule_groups,
        uint32_t physical_blocks, uint32_t segment_slots,
        uint32_t query_tokens, uint32_t columns,
        uint32_t canonical_parts, uint32_t virtual_stream_blocks) {
    const uint32_t local_block = blockIdx.x;
    const uint32_t partition = blockIdx.y & 1;
    const uint32_t lane = threadIdx.x;
    const uint32_t subgroup = lane >> 2;
    const uint32_t sublane = lane & 3;
    const uint32_t column = (blockIdx.y >> 1) * 8 + subgroup;
    const uint32_t subgroup_mask = 0x0000000fu << (4 * subgroup);
    const uint32_t subgroup_base = 4 * subgroup;
    const uint32_t segment = blockIdx.z;
    if (column >= columns || segment >= segment_slots) return;
    uint32_t group_begin = 0;
    uint32_t group_end = 0;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t query_tile = logical_block % query_tiles;
    const uint32_t kv_head = logical_block / query_tiles;
    const uint32_t absolute_query_start = start_pos + query_tile * query_tokens;
    const bool have_bounds = virtual_stream_blocks > 0
        ? stable_virtual_stream_segment_bounds(
            absolute_query_start, kv_head, logical_blocks / query_tiles,
            query_tokens, ring, schedule_groups, virtual_stream_blocks, segment,
            &group_begin, &group_end)
        : (canonical_parts > 0
            ? canonical_segment_bounds(absolute_query_start, schedule_groups,
                canonical_parts, segment, &group_begin, &group_end)
            : stream_segment_bounds(logical_block, logical_blocks,
                schedule_groups, physical_blocks, segment,
                &group_begin, &group_end));
    if (!have_bounds) return;

    float * row = workspace + uint64_t(local_block) * block_stride
        + uint64_t(column) * kv_span;
    const uint64_t probability_floats =
        (uint64_t(columns) * kv_span + 1) / 2;
    __half * probabilities = reinterpret_cast<__half *>(workspace
        + uint64_t(local_block) * block_stride
        + uint64_t(columns) * kv_span);
    constexpr float max_offset = 3.0f * 0.6931f;
    float running_max = -3.402823466e+38F;
    float partial_sum = 0.0f;
    for (uint32_t group = group_begin; group < group_end; ++group) {
        const uint32_t group_base = group * kKeyBatch;
        float next_max = running_max;
        float scores[4];
        if (sublane < 4) {
            for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t item = (half8 / 8) * 2 + pair;
                    const uint32_t key = group_base + partition * 16
                        + half8 + 2 * sublane + pair;
                    scores[item] = key < kv_span
                        ? row[key] : -3.402823466e+38F;
                    if (scores[item] > -3.0e38F) {
                        next_max = fmaxf(next_max, scores[item] + max_offset);
                    }
                }
            }
#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                next_max = fmaxf(next_max,
                    __shfl_xor_sync(subgroup_mask, next_max, offset));
            }
        }
        next_max = __shfl_sync(subgroup_mask, next_max, subgroup_base);
        const float diff = running_max - next_max;
        float rescale = expf(diff);
        if (diff < -20.0f) rescale = 0.0f;
        if (sublane == 0) {
            const uint64_t scales = uint64_t(local_block) * block_stride
                + uint64_t(columns) * kv_span + probability_floats;
            workspace[scales
                + (uint64_t(partition) * columns + column) * schedule_groups
                + group] = rescale;
        }

        if (sublane < 4) {
            float add = 0.0f;
            for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t item = (half8 / 8) * 2 + pair;
                    const uint32_t key = group_base + partition * 16
                        + half8 + 2 * sublane + pair;
                    if (key < kv_span && scores[item] > -3.0e38F) {
                        const float probability = expf(scores[item] - next_max);
                        probabilities[uint64_t(column) * kv_span + key] =
                            __float2half_rn(probability);
                        add += probability;
                    } else if (key < kv_span) {
                        probabilities[uint64_t(column) * kv_span + key] =
                            __float2half_rn(0.0f);
                    }
                }
            }
            partial_sum = partial_sum * rescale + add;
        }
        running_max = next_max;
        __syncwarp();
    }
    if (sublane < 4) {
        float sum = partial_sum;
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(subgroup_mask, sum, offset);
        }
        if (sublane == 0) {
            const uint64_t meta = uint64_t(local_block) * block_stride
                + uint64_t(columns) * kv_span + probability_floats
                + uint64_t(2 * columns) * schedule_groups
                + uint64_t(segment) * 4 * columns
                + uint64_t(partition) * 2 * columns;
            workspace[meta + column] = running_max;
            workspace[meta + columns + column] = sum;
        }
    }
}

// D256's pinned Ampere configuration reduces each 32-key group across one
// four-lane fragment group. D512 uses two independently normalized 16-key
// partitions above. Keep both contracts explicit: their real-number result is
// equivalent, but their phase metadata differs at Stream-K seams.
__global__ void softmax_d256_stream(
        float * workspace, uint32_t kv_span, uint64_t block_stride,
        uint32_t block_base, uint32_t query_tiles, uint32_t start_pos,
        uint32_t ring, uint32_t logical_blocks, uint32_t schedule_groups,
        uint32_t physical_blocks, uint32_t segment_slots,
        uint32_t query_tokens, uint32_t columns,
        uint32_t virtual_stream_blocks, uint32_t keep_f32_probabilities) {
    const uint32_t local_block = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t subgroup = lane >> 2;
    const uint32_t sublane = lane & 3;
    const uint32_t column = blockIdx.y * 8 + subgroup;
    if (column >= columns) return;
    const uint32_t subgroup_mask = 0x0000000fu << (4 * subgroup);
    const uint32_t subgroup_base = 4 * subgroup;
    const uint32_t segment = blockIdx.z;
    if (segment >= segment_slots) return;
    uint32_t group_begin = 0;
    uint32_t group_end = 0;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t query_tile = logical_block % query_tiles;
    const uint32_t kv_head = logical_block / query_tiles;
    const uint32_t absolute_query_start = start_pos + query_tile * query_tokens;
    const bool have_bounds = virtual_stream_blocks > 0
        ? stable_virtual_stream_segment_bounds(
            absolute_query_start, kv_head, logical_blocks / query_tiles,
            query_tokens, ring, schedule_groups, virtual_stream_blocks, segment,
            &group_begin, &group_end)
        : stream_segment_bounds(logical_block, logical_blocks,
            schedule_groups, physical_blocks, segment, &group_begin, &group_end);
    if (!have_bounds) return;
    float * row = workspace + uint64_t(local_block) * block_stride
        + uint64_t(column) * kv_span;
    const uint64_t probability_floats =
        (uint64_t(columns) * kv_span + 1) / 2;
    __half * probabilities = reinterpret_cast<__half *>(workspace
        + uint64_t(local_block) * block_stride
        + uint64_t(columns) * kv_span);
    constexpr float max_offset = 3.0f * 0.6931f;
    float running_max = -1.701411733e+38F;
    float partial_sum = 0.0f;
    for (uint32_t group = group_begin; group < group_end; ++group) {
        const uint32_t gb = group * kKeyBatch;
        float next_max = running_max;
        float scores[8];
        if (sublane < 4) {
            for (uint32_t block16 = 0; block16 < 32; block16 += 16) {
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t item = (block16 / 16) * 4
                            + (half8 / 8) * 2 + pair;
                        const uint32_t key = gb + block16 + half8
                            + 2 * sublane + pair;
                        scores[item] = key < kv_span
                            ? row[key] : -3.402823466e+38F;
                        if (scores[item] > -3.0e38F) {
                            next_max = fmaxf(next_max,
                                scores[item] + max_offset);
                        }
                    }
                }
            }
            for (int offset = 2; offset > 0; offset >>= 1) {
                next_max = fmaxf(next_max,
                    __shfl_xor_sync(subgroup_mask, next_max, offset));
            }
        }
        next_max = __shfl_sync(subgroup_mask, next_max, subgroup_base);
        const float diff = running_max - next_max;
        float rescale = expf(diff);
        if (diff < -20.0f) rescale = 0.0f;
        if (sublane == 0) {
            const uint64_t scales = uint64_t(local_block) * block_stride
                + uint64_t(columns) * kv_span + probability_floats;
            workspace[scales + uint64_t(column) * schedule_groups + group] = rescale;
        }

        if (sublane < 4) {
            float add = 0.0f;
            for (uint32_t block16 = 0; block16 < 32; block16 += 16) {
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t item = (block16 / 16) * 4
                            + (half8 / 8) * 2 + pair;
                        const uint32_t key = gb + block16 + half8
                            + 2 * sublane + pair;
                        if (key < kv_span && scores[item] > -3.0e38F) {
                            const float probability = expf(scores[item] - next_max);
                            if (keep_f32_probabilities) {
                                row[key] = probability;
                            } else {
                                probabilities[uint64_t(column) * kv_span + key] =
                                    __float2half_rn(probability);
                            }
                            add += probability;
                        } else if (key < kv_span) {
                            if (keep_f32_probabilities) {
                                row[key] = 0.0f;
                            } else {
                                probabilities[uint64_t(column) * kv_span + key] =
                                    __float2half_rn(0.0f);
                            }
                        }
                    }
                }
            }
            partial_sum = partial_sum * rescale + add;
        }
        running_max = next_max;
        __syncwarp();
    }
    if (sublane < 4) {
        float sum = partial_sum;
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(subgroup_mask, sum, offset);
        }
        if (sublane == 0) {
            const uint64_t meta = uint64_t(local_block) * block_stride
                + uint64_t(columns) * kv_span + probability_floats
                + uint64_t(columns) * schedule_groups
                + uint64_t(segment) * 2 * columns;
            workspace[meta + column] = running_max;
            workspace[meta + columns + column] = sum;
        }
    }
}

// Laboratory bridge between the high-throughput tiled QK producer and the
// receipt-backed scalar-F32 P*V arithmetic. Softmax leaves each group-relative
// probability in the original float score row, so this kernel can preserve the
// exact online-rescale order without repeating the expensive QK dot products.
__global__ __launch_bounds__(256) void values_d256_stream_f32(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t valid_span, uint32_t kv_span,
        uint64_t block_stride, uint32_t logical_blocks,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t segment_slots, uint32_t start_pos,
        uint32_t query_tokens, uint32_t columns,
        uint32_t virtual_stream_blocks, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t HeadDim = 256;
    const uint32_t local_block = blockIdx.x;
    const uint32_t qcol = blockIdx.y;
    const uint32_t output = threadIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv || qcol >= columns || output >= HeadDim) return;

    const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
    const uint32_t head = kvh * kGqaHeads + qcol % kGqaHeads;
    if (token >= n_tok || head >= n_heads) return;

    const uint64_t probability_floats =
        (uint64_t(columns) * kv_span + 1) / 2;
    const float * probability = workspace
        + uint64_t(local_block) * block_stride + uint64_t(qcol) * kv_span;
    const float * scales = workspace + uint64_t(local_block) * block_stride
        + uint64_t(columns) * kv_span + probability_floats
        + uint64_t(qcol) * schedule_groups;

    bool have_segment = false;
    float combined_num = 0.0f;
    float combined_max = -3.402823466e+38F;
    float combined_sum = 0.0f;
    for (uint32_t slot_rev = segment_slots; slot_rev > 0; --slot_rev) {
        const uint32_t segment = slot_rev - 1;
        uint32_t group_begin = 0;
        uint32_t group_end = 0;
        const uint32_t absolute_query_start =
            start_pos + query_tile * query_tokens;
        const bool have_bounds = virtual_stream_blocks > 0
            ? stable_virtual_stream_segment_bounds(
                absolute_query_start, kvh, n_kv, query_tokens, ring,
                schedule_groups, virtual_stream_blocks, segment,
                &group_begin, &group_end)
            : stream_segment_bounds(logical_block, logical_blocks,
                schedule_groups, physical_blocks, segment,
                &group_begin, &group_end);
        if (!have_bounds) continue;

        float numerator = 0.0f;
        for (uint32_t group = group_begin; group < group_end; ++group) {
            if (group > group_begin) numerator *= scales[group];
            const uint32_t key_end = min((group + 1) * kKeyBatch, kv_span);
            for (uint32_t key = group * kKeyBatch; key < key_end; ++key) {
                const float p = probability[key];
                if (p != 0.0f && key < valid_span) {
                    const uint32_t physical =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    numerator = fmaf(p, __half2float(vc[
                        uint64_t(physical) * kv_width + kvh * HeadDim + output]),
                        numerator);
                }
            }
        }

        const uint64_t meta = uint64_t(local_block) * block_stride
            + uint64_t(columns) * kv_span + probability_floats
            + uint64_t(columns) * schedule_groups
            + uint64_t(segment) * 2 * columns;
        const float segment_max = workspace[meta + qcol];
        const float segment_sum = workspace[meta + columns + qcol];
        if (!have_segment) {
            combined_num = numerator;
            combined_max = segment_max;
            combined_sum = segment_sum;
        } else {
            const float next_max = fmaxf(combined_max, segment_max);
            const float diff_value = combined_max - next_max;
            const float diff_add = segment_max - next_max;
            const float scale_value = diff_value >= -20.0f
                ? expf(diff_value) : 0.0f;
            const float scale_add = diff_add >= -20.0f
                ? expf(diff_add) : 0.0f;
            combined_num = fmaf(
                scale_value, combined_num, scale_add * numerator);
            combined_sum = fmaf(
                scale_value, combined_sum, scale_add * segment_sum);
            combined_max = next_max;
        }
        have_segment = true;
    }

    out[(uint64_t(token) * n_heads + head) * HeadDim + output] =
        have_segment && combined_sum > 0.0f
            ? combined_num / combined_sum : 0.0f;
#else
    (void)vc; (void)workspace; (void)out; (void)block_base;
    (void)query_tiles; (void)n_heads; (void)n_kv; (void)kv_width;
    (void)n_tok; (void)ring; (void)valid_span; (void)kv_span;
    (void)block_stride; (void)logical_blocks; (void)schedule_groups;
    (void)physical_blocks; (void)segment_slots; (void)start_pos;
    (void)query_tokens; (void)columns; (void)virtual_stream_blocks;
    (void)page_table;
#endif
}

// Four warps cover the four 16-column query fragments for one 16-row output tile.
// The half accumulator matches the reference MMA-F16 arithmetic boundary.
template <uint32_t HeadDim>
__global__ void values(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t kv_span, uint32_t logical_blocks,
        uint32_t stream_blocks, uint32_t query_tokens, uint32_t columns,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t local_block = blockIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp & 1;
    const uint32_t column_tile = warp >> 1;
    const uint32_t out0 = blockIdx.y * 16;
    __shared__ __align__(16) __half2 p_tile[8][16 * 8];
    __shared__ __align__(16) __half2 v_tile[8][16 * 8];
    __shared__ __align__(16) __half c_tile[8][16 * 16];
    __shared__ float first_num[4][16 * 16];
    __shared__ float first_max[kMaxColumns];
    __shared__ float first_sum[kMaxColumns];

    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    const uint64_t block_stride = uint64_t(columns) * kv_span
        + uint64_t(2 * columns) * groups + 8 * columns;
    const uint32_t seam = stream_seam_group(
        logical_block, logical_blocks, groups, stream_blocks);
    const uint32_t segments = seam < groups ? 2u : 1u;
    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t group_begin = segment == 0 ? 0 : seam;
        const uint32_t group_end = segment == 0 && segments == 2 ? seam : groups;
        Half16x8 c;
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) c.x[l] = __float2half2_rn(0.0f);
        for (uint32_t group = group_begin; group < group_end; ++group) {
            if (group > group_begin) {
#pragma unroll
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t qcol = column_tile * 16
                        + half_acc_query_column(lane, l);
                    const float scale = workspace[(uint64_t)local_block * block_stride +
                        (uint64_t)columns * kv_span +
                        ((uint64_t)partition * columns + qcol) * groups + group];
                    c.x[l] = __hmul2(c.x[l], __float2half2_rn(scale));
                }
            }
            const uint32_t key0 = group * kKeyBatch + partition * 16;
            for (uint32_t e = lane; e < 16 * 8; e += 32) {
                // Match the wide CUDA FA operand order: probability is the row-major
                // A operand and V is the column-major B operand. The old transposed
                // formulation is mathematically equivalent but not a half-MMA numeric
                // equivalent.
                const uint32_t q_in_tile = e >> 3;
                const uint32_t pair = e & 7;
                const uint32_t pkey = key0 + 2 * pair;
                const uint32_t qcol = column_tile * 16 + q_in_tile;
                const float p0 = pkey < kv_span
                    ? workspace[(uint64_t)local_block * block_stride +
                                (uint64_t)qcol * kv_span + pkey]
                    : 0.0f;
                const float p1 = pkey + 1 < kv_span
                    ? workspace[(uint64_t)local_block * block_stride +
                                (uint64_t)qcol * kv_span + pkey + 1]
                    : 0.0f;
                p_tile[warp][q_in_tile * 8 + pair] = __floats2half2_rn(p0, p1);

                const uint32_t key_row = e >> 3;
                const uint32_t out_pair = e & 7;
                const uint32_t key = key0 + key_row;
                __half2 v = __float2half2_rn(0.0f);
                if (key < kv_span) {
                    const uint32_t physical_key =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    const __half * vr = vc + (uint64_t)physical_key * kv_width
                        + kvh * HeadDim + out0 + 2 * out_pair;
                    v = __halves2half2(vr[0], vr[1]);
                }
                v_tile[warp][key_row * 8 + out_pair] = v;
            }
            __syncwarp();
            Half16x8 probability;
            Half16x8 value;
            load_half16x8(probability, p_tile[warp], 8, lane);
            load_half16x8_trans(value, v_tile[warp], 8, lane);
            mma_pv(c, probability, value);
            __syncwarp();
        }
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            const uint32_t q_in_tile = half_acc_query_column(lane, l);
            const uint32_t out_pair = half_acc_output_pair(lane, l);
            c_tile[warp][q_in_tile * 16 + 2 * out_pair] = __low2half(c.x[l]);
            c_tile[warp][q_in_tile * 16 + 2 * out_pair + 1] = __high2half(c.x[l]);
        }
        __syncthreads();
        if (partition == 0) {
        const uint64_t meta = (uint64_t)local_block * block_stride
            + (uint64_t)columns * kv_span + uint64_t(2 * columns) * groups
            + uint64_t(segment) * 4 * columns;
            for (uint32_t e = lane; e < 16 * 16; e += 32) {
                const uint32_t row = e & 15;
                const uint32_t col = e >> 4;
                const uint32_t qcol = column_tile * 16 + col;
                const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
                const uint32_t head = kvh * kGqaHeads + qcol % kGqaHeads;
                if (token < n_tok && head < n_heads) {
                    const float max0 = workspace[meta + qcol];
                    const float sum0 = workspace[meta + columns + qcol];
                    const float max1 = workspace[meta + 2 * columns + qcol];
                    const float sum1 = workspace[meta + 3 * columns + qcol];
                    const float max_value = fmaxf(max0, max1);
                    const float scale0 = max0 - max_value >= -20.0f
                        ? expf(max0 - max_value) : 0.0f;
                    const float scale1 = max1 - max_value >= -20.0f
                        ? expf(max1 - max_value) : 0.0f;
                    float numerator = fmaf(
                        scale0, __half2float(c_tile[warp][e]), 0.0f);
                    numerator = fmaf(
                        scale1, __half2float(c_tile[warp + 1][e]), numerator);
                    float denominator = __fmul_rn(scale0, sum0);
                    denominator = __fadd_rn(
                        denominator, __fmul_rn(scale1, sum1));
                    if (segments == 1) {
                        const float result = denominator > 0.0f
                            ? numerator / denominator : 0.0f;
                        out[((uint64_t)token * n_heads + head) * HeadDim
                            + out0 + row] = result;
                    } else if (segment == 0) {
                        first_num[column_tile][e] = numerator;
                        first_max[qcol] = max_value;
                        first_sum[qcol] = denominator;
                    } else {
                        const float max_add = first_max[qcol];
                        const float max_combined = fmaxf(max_value, max_add);
                        const float scale_value = max_value - max_combined >= -20.0f
                            ? expf(max_value - max_combined) : 0.0f;
                        const float scale_add = max_add - max_combined >= -20.0f
                            ? expf(max_add - max_combined) : 0.0f;
                        if constexpr (HeadDim == 256) {
                            numerator = scale_value * numerator
                                + scale_add * first_num[column_tile][e];
                            denominator = scale_value * denominator
                                + scale_add * first_sum[qcol];
                        } else {
                            numerator = fmaf(scale_value, numerator,
                                scale_add * first_num[column_tile][e]);
                            denominator = fmaf(scale_value, denominator,
                                scale_add * first_sum[qcol]);
                        }
                        const float result = denominator > 0.0f
                            ? numerator / denominator : 0.0f;
                        out[((uint64_t)token * n_heads + head) * HeadDim
                            + out0 + row] = result;
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)query_tiles;
    (void)n_heads; (void)n_kv; (void)kv_width; (void)n_tok; (void)ring;
    (void)kv_span; (void)logical_blocks; (void)stream_blocks;
    (void)query_tokens; (void)columns; (void)page_table;
#endif
}

__host__ __device__ constexpr uint32_t bounded_half8_count(
        uint32_t base, uint32_t limit) {
    if (base >= limit) return 0;
    const uint32_t remaining = limit - base;
    return remaining < 8 ? remaining : 8;
}
static_assert(bounded_half8_count(0, 128) == 8);
static_assert(bounded_half8_count(120, 128) == 8);
static_assert(bounded_half8_count(124, 128) == 4);
static_assert(bounded_half8_count(128, 128) == 0);
static_assert(bounded_half8_count(136, 128) == 0);

template <uint32_t HeadDim, typename KvRows>
__device__ __forceinline__ void prefetch_partitioned_pv_tile(
        const __half * vc, const __half * probabilities,
        __half2 * probability_dst, __half2 * value_dst,
        uint32_t group, uint32_t partition, uint32_t probability_column,
        uint32_t key_row, uint32_t lane_half, uint32_t kvh, uint32_t kv_width,
        uint32_t valid_span, uint32_t kv_span, uint32_t output0,
        KvRows kv_rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t key0 = group * kKeyBatch + partition * 16;
    const uint32_t probability_key = key0 + lane_half * 8;
    const uint32_t probability_count =
        bounded_half8_count(probability_key, kv_span);
    const __half * probability_src = probabilities
        + uint64_t(probability_column) * kv_span
        + (probability_key < kv_span ? probability_key : kv_span);
    const uint32_t key = key0 + key_row;
    const bool valid_key = key < valid_span;
    const uint32_t physical_key = kv_rows.physical(key);
    const __half * value_src = valid_key
        ? vc + uint64_t(physical_key) * kv_width + kvh * HeadDim
            + output0 + lane_half * 8
        : vc;
    const uint32_t probability_shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(probability_dst));
    const uint32_t value_shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(value_dst));
    // Sliding-window rings expose their physical span without padding, so adjacent
    // probability rows are not necessarily 16-byte aligned while the ring fills.
    // Keep the asynchronous fast path for aligned rows and use scalar half copies
    // for the short unaligned prefix; V remains asynchronously staged in both cases.
    if (probability_count == 8
        && (reinterpret_cast<uintptr_t>(probability_src) & 15u) == 0) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(probability_shared), "l"(probability_src));
    } else {
        __half * probability_half = reinterpret_cast<__half *>(probability_dst);
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            probability_half[i] = i < probability_count
                ? probability_src[i] : __float2half(0.0f);
        }
    }
    const uint32_t value_bytes = valid_key ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(value_shared), "l"(value_src), "r"(value_bytes));
#else
    (void)vc; (void)probabilities; (void)probability_dst; (void)value_dst;
    (void)group; (void)partition; (void)probability_column; (void)key_row;
    (void)lane_half; (void)kvh;
    (void)kv_width; (void)valid_span; (void)kv_span; (void)output0;
    (void)kv_rows;
#endif
}

template <uint32_t HeadDim, typename KvRows>
__device__ __forceinline__ void prefetch_partitioned_value_tile(
        const __half * vc, __half2 * value_dst, uint32_t group,
        uint32_t partition, uint32_t key_row, uint32_t lane_half,
        uint32_t kvh, uint32_t kv_width, uint32_t valid_span,
        uint32_t output0, KvRows kv_rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t key = group * kKeyBatch + partition * 16 + key_row;
    const bool valid_key = key < valid_span;
    const uint32_t physical_key = kv_rows.physical(key);
    const __half * value_src = valid_key
        ? vc + uint64_t(physical_key) * kv_width + kvh * HeadDim
            + output0 + lane_half * 8
        : vc;
    const uint32_t value_shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(value_dst));
    const uint32_t value_bytes = valid_key ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
        :: "r"(value_shared), "l"(value_src), "r"(value_bytes));
#else
    (void)vc; (void)value_dst; (void)group; (void)partition;
    (void)key_row; (void)lane_half; (void)kvh; (void)kv_width;
    (void)valid_span; (void)output0; (void)kv_rows;
#endif
}

template <uint32_t HeadDim, typename KvRows>
__device__ __forceinline__ void values_partitioned_stream_body(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t valid_span, uint32_t kv_span,
        uint64_t block_stride,
        uint32_t logical_blocks, uint32_t schedule_groups,
        uint32_t physical_blocks, uint32_t segment_slots,
        uint32_t start_pos, uint32_t query_tokens, uint32_t columns,
        uint32_t canonical_parts, uint32_t virtual_stream_blocks,
        KvRows kv_rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(HeadDim == 256 || HeadDim == 512);
    const uint32_t local_block = blockIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp & 1;
    const uint32_t column_tile = warp >> 1;
    constexpr uint32_t output_tiles_per_cta = 4;
    const uint32_t out0 = blockIdx.y * output_tiles_per_cta * 16;
    // P/V staging is dead before the segment-combine phase begins. Overlay its
    // storage with the half result tile and scale vectors instead of reserving
    // both lifetimes concurrently; this brings the kernel below SM86's 3-CTA
    // shared-memory threshold without changing any arithmetic or synchronization.
    struct PvIoScratch {
        __half2 p_tile[2][8][16 * 8];
        __half2 v_tile[2][8][16 * 8];
    };
    struct PvCombineScratch {
        __half c_tile[8][16 * 16];
        float partition_scale0[kMaxColumns];
        float partition_scale1[kMaxColumns];
        float combine_scale_value[kMaxColumns];
        float combine_scale_add[kMaxColumns];
    };
    union PvScratch {
        PvIoScratch io;
        PvCombineScratch combine;
    };
    __shared__ __align__(16) PvScratch scratch;
    auto & p_tile = scratch.io.p_tile;
    auto & v_tile = scratch.io.v_tile;
    auto & c_tile = scratch.combine.c_tile;
    auto & partition_scale0 = scratch.combine.partition_scale0;
    auto & partition_scale1 = scratch.combine.partition_scale1;
    auto & combine_scale_value = scratch.combine.combine_scale_value;
    auto & combine_scale_add = scratch.combine.combine_scale_add;
    __shared__ float combined_num[output_tiles_per_cta][4][16 * 16];
    __shared__ float combined_max[kMaxColumns];
    __shared__ float combined_sum[kMaxColumns];
    const uint64_t probability_floats =
        (uint64_t(columns) * kv_span + 1) / 2;
    const __half * probabilities = reinterpret_cast<const __half *>(
        workspace + uint64_t(local_block) * block_stride
            + uint64_t(columns) * kv_span);

    bool have_segment = false;
    for (uint32_t slot_rev = segment_slots; slot_rev > 0; --slot_rev) {
        const uint32_t segment = slot_rev - 1;
        uint32_t group_begin = 0;
        uint32_t group_end = 0;
        const uint32_t absolute_query_start =
            start_pos + query_tile * query_tokens;
        const bool have_bounds = virtual_stream_blocks > 0
            ? stable_virtual_stream_segment_bounds(
                absolute_query_start, kvh, n_kv, query_tokens, ring,
                schedule_groups, virtual_stream_blocks, segment,
                &group_begin, &group_end)
            : (canonical_parts > 0
                ? canonical_segment_bounds(absolute_query_start, schedule_groups,
                    canonical_parts, segment, &group_begin, &group_end)
                : stream_segment_bounds(logical_block, logical_blocks,
                    schedule_groups, physical_blocks, segment,
                    &group_begin, &group_end));
        if (!have_bounds) continue;

        Half16x8 c[output_tiles_per_cta];
#pragma unroll
        for (uint32_t tile = 0; tile < output_tiles_per_cta; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                c[tile].x[l] = __float2half2_rn(0.0f);
            }
        }
        uint32_t pv_stage = 0;
        const uint32_t copy_qcol = lane >> 1;
        const uint32_t copy_half = lane & 1;
        prefetch_partitioned_pv_tile<HeadDim>(
            vc, probabilities, &p_tile[pv_stage][warp][lane * 4],
            &v_tile[pv_stage][warp][lane * 4], group_begin, partition,
            column_tile * 16 + copy_qcol, copy_qcol, copy_half,
            kvh, kv_width, valid_span, kv_span, out0, kv_rows);
        asm volatile("cp.async.commit_group;");
        for (uint32_t group = group_begin; group < group_end; ++group) {
            asm volatile("cp.async.wait_group 0;");
            __syncwarp();
            const uint32_t next_group = group + 1;
            const uint32_t next_stage = pv_stage ^ 1;
            if (group > group_begin) {
#pragma unroll
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t qcol = column_tile * 16
                        + half_acc_query_column(lane, l);
                    const float scale = workspace[uint64_t(local_block) * block_stride
                        + uint64_t(columns) * kv_span + probability_floats
                        + (uint64_t(partition) * columns + qcol) * schedule_groups
                        + group];
                    const __half2 scale2 = __float2half2_rn(scale);
#pragma unroll
                    for (uint32_t tile = 0; tile < output_tiles_per_cta; ++tile) {
                        c[tile].x[l] = __hmul2(c[tile].x[l], scale2);
                    }
                }
            }
            Half16x8 probability;
            Half16x8 value;
            load_half16x8(probability, p_tile[pv_stage][warp], 8, lane);
            const bool have_next = next_group < group_end;
#pragma unroll
            for (uint32_t tile = 0; tile < output_tiles_per_cta; ++tile) {
                load_half16x8_trans(value, v_tile[pv_stage][warp], 8, lane);
                mma_pv(c[tile], probability, value);
                if (tile + 1 < output_tiles_per_cta) {
                    prefetch_partitioned_value_tile<HeadDim>(
                        vc, &v_tile[pv_stage][warp][lane * 4], group, partition,
                        copy_qcol, copy_half, kvh, kv_width, valid_span,
                        out0 + (tile + 1) * 16, kv_rows);
                    asm volatile("cp.async.commit_group;");
                    if (tile + 2 == output_tiles_per_cta && have_next) {
                        prefetch_partitioned_pv_tile<HeadDim>(
                            vc, probabilities, &p_tile[next_stage][warp][lane * 4],
                            &v_tile[next_stage][warp][lane * 4], next_group, partition,
                            column_tile * 16 + copy_qcol, copy_qcol, copy_half,
                            kvh, kv_width, valid_span, kv_span, out0, kv_rows);
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 1;");
                    } else {
                        asm volatile("cp.async.wait_group 0;");
                    }
                    __syncwarp();
                }
            }
            pv_stage = next_stage;
        }
        __syncthreads();

        const uint64_t meta = uint64_t(local_block) * block_stride
            + uint64_t(columns) * kv_span + probability_floats
            + uint64_t(2 * columns) * schedule_groups
            + uint64_t(segment) * 4 * columns;
        for (uint32_t qcol = threadIdx.x; qcol < columns; qcol += blockDim.x) {
            const float max0 = workspace[meta + qcol];
            const float sum0 = workspace[meta + columns + qcol];
            const float max1 = workspace[meta + 2 * columns + qcol];
            const float sum1 = workspace[meta + 3 * columns + qcol];
            const float segment_max = fmaxf(max0, max1);
            const float scale0 = max0 - segment_max >= -20.0f
                ? expf(max0 - segment_max) : 0.0f;
            const float scale1 = max1 - segment_max >= -20.0f
                ? expf(max1 - segment_max) : 0.0f;
            // The warp-pair reduction crosses a shuffle boundary in the reference:
            // each product is rounded before the pair is added.
            const float segment_sum = __fadd_rn(
                __fmul_rn(scale0, sum0), __fmul_rn(scale1, sum1));
            partition_scale0[qcol] = scale0;
            partition_scale1[qcol] = scale1;
            if (!have_segment) {
                combine_scale_value[qcol] = 0.0f;
                combine_scale_add[qcol] = 1.0f;
                combined_max[qcol] = segment_max;
                combined_sum[qcol] = segment_sum;
            } else {
                const float max_value = combined_max[qcol];
                const float max_new = fmaxf(max_value, segment_max);
                const float diff_value = max_value - max_new;
                const float diff_add = segment_max - max_new;
                const float scale_value = diff_value >= -20.0f
                    ? expf(diff_value) : 0.0f;
                const float scale_add = diff_add >= -20.0f
                    ? expf(diff_add) : 0.0f;
                combine_scale_value[qcol] = scale_value;
                combine_scale_add[qcol] = scale_add;
                // Match the reference fixup's contracted multiply-add. NVCC does not
                // reliably recover this contraction after the values pass through
                // shared memory, so keep it explicit.
                combined_sum[qcol] = fmaf(scale_value, combined_sum[qcol],
                    scale_add * segment_sum);
                combined_max[qcol] = max_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t tile = 0; tile < output_tiles_per_cta; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t q_in_tile = half_acc_query_column(lane, l);
                const uint32_t out_pair = half_acc_output_pair(lane, l);
                c_tile[warp][q_in_tile * 16 + 2 * out_pair]
                    = __low2half(c[tile].x[l]);
                c_tile[warp][q_in_tile * 16 + 2 * out_pair + 1]
                    = __high2half(c[tile].x[l]);
            }
            __syncthreads();
            if (partition == 0) {
                for (uint32_t e = lane; e < 16 * 16; e += 32) {
                    const uint32_t col = e >> 4;
                    const uint32_t qcol = column_tile * 16 + col;
                    float numerator = fmaf(partition_scale0[qcol],
                        __half2float(c_tile[warp][e]), 0.0f);
                    numerator = fmaf(partition_scale1[qcol],
                        __half2float(c_tile[warp + 1][e]), numerator);
                    if (!have_segment) {
                        combined_num[tile][column_tile][e] = numerator;
                    } else {
                        combined_num[tile][column_tile][e] = fmaf(
                            combine_scale_value[qcol],
                            combined_num[tile][column_tile][e],
                            combine_scale_add[qcol] * numerator);
                    }
                }
            }
            __syncthreads();
        }
        have_segment = true;
    }

    if (partition == 0) {
#pragma unroll
        for (uint32_t tile = 0; tile < output_tiles_per_cta; ++tile) {
            for (uint32_t e = lane; e < 16 * 16; e += 32) {
                const uint32_t row = e & 15;
                const uint32_t col = e >> 4;
                const uint32_t qcol = column_tile * 16 + col;
                const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
                const uint32_t head = kvh * kGqaHeads + qcol % kGqaHeads;
                if (have_segment && token < n_tok && head < n_heads) {
                    const float rowsum = combined_sum[qcol];
                    out[(uint64_t(token) * n_heads + head) * HeadDim
                            + out0 + tile * 16 + row]
                        = rowsum > 0.0f
                            ? combined_num[tile][column_tile][e] / rowsum : 0.0f;
                }
            }
        }
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)query_tiles;
    (void)n_heads; (void)n_kv; (void)kv_width; (void)n_tok; (void)ring;
    (void)valid_span; (void)kv_span; (void)block_stride; (void)logical_blocks;
    (void)schedule_groups; (void)physical_blocks; (void)segment_slots;
    (void)start_pos; (void)query_tokens; (void)columns; (void)canonical_parts;
    (void)virtual_stream_blocks; (void)kv_rows;
#endif
}

template <uint32_t HeadDim>
__global__ void values_partitioned_stream(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t valid_span, uint32_t kv_span,
        uint64_t block_stride, uint32_t logical_blocks,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t segment_slots, uint32_t start_pos, uint32_t query_tokens,
        uint32_t columns, uint32_t canonical_parts,
        uint32_t virtual_stream_blocks, const uint32_t * page_table) {
    values_partitioned_stream_body<HeadDim>(vc, workspace, out, block_base,
        query_tiles, n_heads, n_kv, kv_width, n_tok, ring, valid_span, kv_span,
        block_stride, logical_blocks, schedule_groups, physical_blocks,
        segment_slots, start_pos, query_tokens, columns, canonical_parts,
        virtual_stream_blocks, PagedKvRows{page_table, ring});
}

__global__ void values_partitioned_stream_d512(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t valid_span, uint32_t kv_span,
        uint64_t block_stride, uint32_t logical_blocks,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t segment_slots, uint32_t start_pos, uint32_t query_tokens,
        uint32_t columns, uint32_t canonical_parts,
        uint32_t virtual_stream_blocks) {
    values_partitioned_stream_body<512>(vc, workspace, out, block_base,
        query_tiles, n_heads, n_kv, kv_width, n_tok, ring, valid_span, kv_span,
        block_stride, logical_blocks, schedule_groups, physical_blocks,
        segment_slots, start_pos, query_tokens, columns, canonical_parts,
        virtual_stream_blocks, IdentityKvRows{});
}

__global__ void values_partitioned_stream_d512_paged(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t valid_span, uint32_t kv_span,
        uint64_t block_stride, uint32_t logical_blocks,
        uint32_t schedule_groups, uint32_t physical_blocks,
        uint32_t segment_slots, uint32_t start_pos, uint32_t query_tokens,
        uint32_t columns, uint32_t canonical_parts,
        uint32_t virtual_stream_blocks, const uint32_t * page_table) {
    values_partitioned_stream_body<512>(vc, workspace, out, block_base,
        query_tiles, n_heads, n_kv, kv_width, n_tok, ring, valid_span, kv_span,
        block_stride, logical_blocks, schedule_groups, physical_blocks,
        segment_slots, start_pos, query_tokens, columns, canonical_parts,
        virtual_stream_blocks, PagedKvRows{page_table, ring});
}

// D256 accumulates both 16-key halves of each 32-key group into one half MMA
// accumulator. D512 assigns them to separate warps and combines in f32. The
// distinction first becomes observable at causal token 16.
__global__ void values_d256(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t kv_span, uint32_t logical_blocks,
        uint32_t stream_blocks, uint32_t query_tokens, uint32_t columns,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t HeadDim = 256;
    const uint32_t local_block = blockIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t column_tile = warp;
    const uint32_t out0 = blockIdx.y * 16;
    __shared__ __align__(16) __half2 p_tile[4][16 * 8];
    __shared__ __align__(16) __half2 v_tile[4][16 * 8];
    __shared__ __align__(16) __half c_tile[4][16 * 16];
    __shared__ float first_num[4][16 * 16];
    __shared__ float first_max[kMaxColumns];
    __shared__ float first_sum[kMaxColumns];

    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    const uint64_t block_stride = uint64_t(columns) * kv_span
        + uint64_t(2 * columns) * groups + 8 * columns;
    const uint32_t seam = stream_seam_group(
        logical_block, logical_blocks, groups, stream_blocks);
    const uint32_t segments = seam < groups ? 2u : 1u;
    for (uint32_t segment = 0; segment < segments; ++segment) {
        const uint32_t group_begin = segment == 0 ? 0 : seam;
        const uint32_t group_end = segment == 0 && segments == 2 ? seam : groups;
        Half16x8 c;
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) c.x[l] = __float2half2_rn(0.0f);
        for (uint32_t group = group_begin; group < group_end; ++group) {
            if (group > group_begin) {
#pragma unroll
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t qcol = column_tile * 16
                        + half_acc_query_column(lane, l);
                    const float scale = workspace[uint64_t(local_block) * block_stride
                        + uint64_t(columns) * kv_span
                        + uint64_t(qcol) * groups + group];
                    c.x[l] = __hmul2(c.x[l], __float2half2_rn(scale));
                }
            }
#pragma unroll
            for (uint32_t partition = 0; partition < 2; ++partition) {
                const uint32_t key0 = group * kKeyBatch + partition * 16;
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t q_in_tile = e >> 3;
                    const uint32_t pair = e & 7;
                    const uint32_t pkey = key0 + 2 * pair;
                    const uint32_t qcol = column_tile * 16 + q_in_tile;
                    const float p0 = pkey < kv_span
                        ? workspace[uint64_t(local_block) * block_stride
                            + uint64_t(qcol) * kv_span + pkey] : 0.0f;
                    const float p1 = pkey + 1 < kv_span
                        ? workspace[uint64_t(local_block) * block_stride
                            + uint64_t(qcol) * kv_span + pkey + 1] : 0.0f;
                    p_tile[warp][q_in_tile * 8 + pair] =
                        __floats2half2_rn(p0, p1);

                    const uint32_t key_row = e >> 3;
                    const uint32_t out_pair = e & 7;
                    const uint32_t key = key0 + key_row;
                    __half2 value = __float2half2_rn(0.0f);
                    if (key < kv_span) {
                        const uint32_t physical_key =
                            imparo_cuda_kv::physical_row(key, ring, page_table);
                        const __half * vr = vc + uint64_t(physical_key) * kv_width
                            + kvh * HeadDim + out0 + 2 * out_pair;
                        value = __halves2half2(vr[0], vr[1]);
                    }
                    v_tile[warp][key_row * 8 + out_pair] = value;
                }
                __syncwarp();
                Half16x8 probability;
                Half16x8 value;
                load_half16x8(probability, p_tile[warp], 8, lane);
                load_half16x8_trans(value, v_tile[warp], 8, lane);
                mma_pv(c, probability, value);
                __syncwarp();
            }
        }
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            const uint32_t q_in_tile = half_acc_query_column(lane, l);
            const uint32_t out_pair = half_acc_output_pair(lane, l);
            c_tile[warp][q_in_tile * 16 + 2 * out_pair] = __low2half(c.x[l]);
            c_tile[warp][q_in_tile * 16 + 2 * out_pair + 1] = __high2half(c.x[l]);
        }
        __syncthreads();

        const uint64_t meta = uint64_t(local_block) * block_stride
            + uint64_t(columns) * kv_span + uint64_t(2 * columns) * groups
            + uint64_t(segment) * 4 * columns;
        for (uint32_t e = lane; e < 16 * 16; e += 32) {
            const uint32_t row = e & 15;
            const uint32_t col = e >> 4;
            const uint32_t qcol = column_tile * 16 + col;
            const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + qcol % kGqaHeads;
            if (token < n_tok && head < n_heads) {
                const float numerator = __half2float(c_tile[warp][e]);
                const float max_value = workspace[meta + qcol];
                const float denominator = workspace[meta + columns + qcol];
                if (segments == 1) {
                    out[(uint64_t(token) * n_heads + head) * HeadDim + out0 + row]
                        = denominator > 0.0f ? numerator / denominator : 0.0f;
                } else if (segment == 0) {
                    first_num[column_tile][e] = numerator;
                    first_max[qcol] = max_value;
                    first_sum[qcol] = denominator;
                } else {
                    const float max_add = first_max[qcol];
                    const float max_combined = fmaxf(max_value, max_add);
                    const float scale_value = max_value - max_combined >= -20.0f
                        ? expf(max_value - max_combined) : 0.0f;
                    const float scale_add = max_add - max_combined >= -20.0f
                        ? expf(max_add - max_combined) : 0.0f;
                    const float value = scale_value * numerator
                        + scale_add * first_num[column_tile][e];
                    const float rowsum = scale_value * denominator
                        + scale_add * first_sum[qcol];
                    out[(uint64_t(token) * n_heads + head) * HeadDim + out0 + row]
                        = value / rowsum;
                }
            }
        }
        __syncthreads();
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)query_tiles;
    (void)n_heads; (void)n_kv; (void)kv_width; (void)n_tok; (void)ring;
    (void)kv_span; (void)logical_blocks; (void)stream_blocks;
    (void)query_tokens; (void)columns; (void)page_table;
#endif
}

__global__ void values_d256_stream(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t query_tiles, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t n_tok,
        uint32_t ring, uint32_t valid_span, uint32_t kv_span,
        uint64_t block_stride,
        uint32_t logical_blocks, uint32_t schedule_groups,
        uint32_t physical_blocks, uint32_t segment_slots,
        uint32_t start_pos, uint32_t query_tokens, uint32_t columns,
        uint32_t virtual_stream_blocks, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t HeadDim = 256;
    const uint32_t local_block = blockIdx.x;
    const uint32_t logical_block = block_base + local_block;
    const uint32_t kvh = logical_block / query_tiles;
    const uint32_t query_tile = logical_block % query_tiles;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t column_tile = warp;
    constexpr uint32_t OutputTiles = 2;
    const uint32_t out0 = blockIdx.y * (16 * OutputTiles);
    struct PvIoScratch {
        __half2 p_tile[2][4][16 * 8];
        __half2 v_tile[2][4][16 * 8];
    };
    struct PvCombineScratch {
        __half c_tile[OutputTiles][4][16 * 16];
        float combine_scale_value[kMaxColumns];
        float combine_scale_add[kMaxColumns];
    };
    union PvScratch {
        PvIoScratch io;
        PvCombineScratch combine;
    };
    __shared__ __align__(16) PvScratch scratch;
    auto & p_tile = scratch.io.p_tile;
    auto & v_tile = scratch.io.v_tile;
    auto & c_tile = scratch.combine.c_tile;
    auto & combine_scale_value = scratch.combine.combine_scale_value;
    auto & combine_scale_add = scratch.combine.combine_scale_add;
    __shared__ float combined_num[OutputTiles][4][16 * 16];
    __shared__ float combined_max[kMaxColumns];
    __shared__ float combined_sum[kMaxColumns];
    const uint64_t probability_floats =
        (uint64_t(columns) * kv_span + 1) / 2;
    const __half * probabilities = reinterpret_cast<const __half *>(
        workspace + uint64_t(local_block) * block_stride
            + uint64_t(columns) * kv_span);

    bool have_segment = false;
    for (uint32_t slot_rev = segment_slots; slot_rev > 0; --slot_rev) {
        const uint32_t segment = slot_rev - 1;
        uint32_t group_begin = 0;
        uint32_t group_end = 0;
        const uint32_t absolute_query_start =
            start_pos + query_tile * query_tokens;
        const bool have_bounds = virtual_stream_blocks > 0
            ? stable_virtual_stream_segment_bounds(
                absolute_query_start, kvh, n_kv, query_tokens, ring,
                schedule_groups, virtual_stream_blocks, segment,
                &group_begin, &group_end)
            : stream_segment_bounds(logical_block, logical_blocks,
                schedule_groups, physical_blocks, segment,
                &group_begin, &group_end);
        if (!have_bounds) continue;

        Half16x8 c[OutputTiles];
#pragma unroll
        for (uint32_t output_tile = 0; output_tile < OutputTiles; ++output_tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                c[output_tile].x[l] = __float2half2_rn(0.0f);
            }
        }
        uint32_t pv_stage = 0;
        const uint32_t copy_qcol = lane >> 1;
        const uint32_t copy_half = lane & 1;
        const uint32_t work_begin = 2 * group_begin;
        const uint32_t work_end = 2 * group_end;
        prefetch_partitioned_pv_tile<HeadDim>(
            vc, probabilities, &p_tile[pv_stage][warp][lane * 4],
            &v_tile[pv_stage][warp][lane * 4], group_begin, 0,
            column_tile * 16 + copy_qcol, copy_qcol, copy_half,
            kvh, kv_width, valid_span, kv_span, out0,
            PagedKvRows{page_table, ring});
        asm volatile("cp.async.commit_group;");
        for (uint32_t work = work_begin; work < work_end; ++work) {
            asm volatile("cp.async.wait_group 0;");
            __syncwarp();
            const uint32_t group = work >> 1;
            const uint32_t partition = work & 1;
            if (partition == 0 && group > group_begin) {
#pragma unroll
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t qcol = column_tile * 16
                        + half_acc_query_column(lane, l);
                    const float scale = workspace[uint64_t(local_block) * block_stride
                        + uint64_t(columns) * kv_span + probability_floats
                        + uint64_t(qcol) * schedule_groups + group];
                    const __half2 half_scale = __float2half2_rn(scale);
#pragma unroll
                    for (uint32_t output_tile = 0;
                         output_tile < OutputTiles; ++output_tile) {
                        c[output_tile].x[l] =
                            __hmul2(c[output_tile].x[l], half_scale);
                    }
                }
            }
            Half16x8 probability;
            Half16x8 value;
            load_half16x8(probability, p_tile[pv_stage][warp], 8, lane);
            const uint32_t next_work = work + 1;
            const uint32_t next_stage = pv_stage ^ 1;
#pragma unroll
            for (uint32_t output_tile = 0;
                 output_tile < OutputTiles; ++output_tile) {
                load_half16x8_trans(value, v_tile[pv_stage][warp], 8, lane);
                mma_pv(c[output_tile], probability, value);
                if (output_tile + 1 < OutputTiles) {
                    prefetch_partitioned_value_tile<HeadDim>(
                        vc, &v_tile[pv_stage][warp][lane * 4], group, partition,
                        copy_qcol, copy_half, kvh, kv_width, valid_span,
                        out0 + (output_tile + 1) * 16,
                        PagedKvRows{page_table, ring});
                    asm volatile("cp.async.commit_group;");
                    if (output_tile + 2 == OutputTiles && next_work < work_end) {
                        const uint32_t next_group = next_work >> 1;
                        const uint32_t next_partition = next_work & 1;
                        prefetch_partitioned_pv_tile<HeadDim>(
                            vc, probabilities, &p_tile[next_stage][warp][lane * 4],
                            &v_tile[next_stage][warp][lane * 4], next_group,
                            next_partition, column_tile * 16 + copy_qcol,
                            copy_qcol, copy_half, kvh, kv_width, valid_span,
                            kv_span, out0, PagedKvRows{page_table, ring});
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 1;");
                    } else {
                        asm volatile("cp.async.wait_group 0;");
                    }
                    __syncwarp();
                }
            }
            pv_stage = next_stage;
        }
#pragma unroll
        for (uint32_t output_tile = 0;
             output_tile < OutputTiles; ++output_tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t q_in_tile = half_acc_query_column(lane, l);
                const uint32_t out_pair = half_acc_output_pair(lane, l);
                c_tile[output_tile][warp][q_in_tile * 16 + 2 * out_pair] =
                    __low2half(c[output_tile].x[l]);
                c_tile[output_tile][warp][q_in_tile * 16 + 2 * out_pair + 1] =
                    __high2half(c[output_tile].x[l]);
            }
        }
        __syncthreads();

        const uint64_t meta = uint64_t(local_block) * block_stride
            + uint64_t(columns) * kv_span + probability_floats
            + uint64_t(columns) * schedule_groups
            + uint64_t(segment) * 2 * columns;
        for (uint32_t qcol = threadIdx.x; qcol < columns; qcol += blockDim.x) {
            const float segment_max = workspace[meta + qcol];
            const float segment_sum = workspace[meta + columns + qcol];
            if (!have_segment) {
                combine_scale_value[qcol] = 0.0f;
                combine_scale_add[qcol] = 1.0f;
                combined_max[qcol] = segment_max;
                combined_sum[qcol] = segment_sum;
            } else {
                const float max_value = combined_max[qcol];
                const float max_new = fmaxf(max_value, segment_max);
                const float diff_value = max_value - max_new;
                const float diff_add = segment_max - max_new;
                const float scale_value = diff_value >= -20.0f
                    ? expf(diff_value) : 0.0f;
                const float scale_add = diff_add >= -20.0f
                    ? expf(diff_add) : 0.0f;
                combine_scale_value[qcol] = scale_value;
                combine_scale_add[qcol] = scale_add;
                // The reference fixup kernel keeps the first product in a register,
                // so NVCC contracts its final add.  This state lives in shared memory;
                // spell out the same contraction or a few Stream-K seam rows drift by
                // one ULP depending on token/head data.
                combined_sum[qcol] = fmaf(
                    scale_value, combined_sum[qcol], scale_add * segment_sum);
                combined_max[qcol] = max_new;
            }
        }
        __syncthreads();
        for (uint32_t output_tile = 0;
             output_tile < OutputTiles; ++output_tile) {
            for (uint32_t e = lane; e < 16 * 16; e += 32) {
                const uint32_t col = e >> 4;
                const uint32_t qcol = column_tile * 16 + col;
                const float numerator =
                    __half2float(c_tile[output_tile][warp][e]);
                if (!have_segment) {
                    combined_num[output_tile][column_tile][e] = numerator;
                } else {
                    combined_num[output_tile][column_tile][e] = fmaf(
                        combine_scale_value[qcol],
                        combined_num[output_tile][column_tile][e],
                        combine_scale_add[qcol] * numerator);
                }
            }
        }
        have_segment = true;
        __syncthreads();
    }

    for (uint32_t output_tile = 0;
         output_tile < OutputTiles; ++output_tile) {
        for (uint32_t e = lane; e < 16 * 16; e += 32) {
            const uint32_t row = e & 15;
            const uint32_t col = e >> 4;
            const uint32_t qcol = column_tile * 16 + col;
            const uint32_t token = query_tile * query_tokens + qcol / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + qcol % kGqaHeads;
            if (have_segment && token < n_tok && head < n_heads) {
                const float rowsum = combined_sum[qcol];
                out[(uint64_t(token) * n_heads + head) * HeadDim + out0
                    + output_tile * 16 + row] = rowsum > 0.0f
                        ? combined_num[output_tile][column_tile][e] / rowsum
                            : 0.0f;
            }
        }
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)query_tiles;
    (void)n_heads; (void)n_kv; (void)kv_width; (void)n_tok; (void)ring;
    (void)valid_span; (void)kv_span; (void)block_stride; (void)logical_blocks;
    (void)schedule_groups; (void)physical_blocks; (void)segment_slots;
    (void)start_pos; (void)query_tokens; (void)columns;
    (void)virtual_stream_blocks; (void)page_table;
#endif
}

} // namespace imparo_sm80_prefill
