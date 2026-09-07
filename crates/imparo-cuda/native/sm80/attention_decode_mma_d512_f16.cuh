#pragma once

#include "attention_prefill_d512_f16.cuh"

// SM80 D=512, single-token, staged-F16 decode attention.
//
// This numerical class deliberately mirrors the pinned llama D512 decode shape
// instead of extending the direct-Q4 D256 vector kernel:
//   * ncols1=2, ncols2=4 (8 physical columns, 4 live GQA columns),
//   * 64 threads / two warps, one 16-key partition per warp,
//   * 32 keys per online-softmax batch,
//   * F16 Q/K/V and probability operands, FP32 QK/max/rowsum/fixup state,
//   * a half-accumulator PV MMA, followed by FP32 partition/fixup combines.
//
// Launch policy, allocation and fallback remain outside this header.  The
// partial kernel consumes an externally selected physical Stream-K grid.  A
// physical block can overlap at most two logical tiles when physical_blocks is
// at least the tile count; unsupported schedules must use the caller's safe
// fallback.  `combine_reverse` handles both uniform and non-uniform block counts
// and always associates partials from the highest key range to the lowest.

namespace imparo_sm80_d512_decode {

using imparo_sm80_mma::Half16x8;
using imparo_sm80_mma::load_half16x8;
using imparo_sm80_mma::load_half16x8_trans;

constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kQueryRows = 2;
constexpr uint32_t kColumns = kQueryRows * kGqaHeads;
constexpr uint32_t kLiveColumns = kGqaHeads;
constexpr uint32_t kThreads = 64;
constexpr uint32_t kWarps = 2;
constexpr uint32_t kKeyBatch = 32;
constexpr uint32_t kKeysPerWarp = 16;
constexpr uint32_t kOutputTiles = kHeadDim / 16;
constexpr uint32_t kSegmentsPerPhysicalBlock = 2;
constexpr uint64_t kSlotNumeratorFloats = uint64_t(kColumns) * kHeadDim;
constexpr uint64_t kSlotMetaFloats = uint64_t(2) * kColumns;
constexpr uint64_t kSlotStrideFloats = kSlotNumeratorFloats + kSlotMetaFloats;
constexpr float kInitialMax = -1.701411733e+38F;
constexpr float kMaskedScore = -3.402823466e+38F;
constexpr float kMaxOffset = 3.0f * 0.6931f;
constexpr float kFtzThreshold = -20.0f;

inline uint64_t workspace_floats(uint32_t physical_blocks) {
    return uint64_t(physical_blocks) * kSegmentsPerPhysicalBlock
        * kSlotStrideFloats;
}

__host__ __device__ inline bool supports_schedule(
        uint32_t n_heads, uint32_t n_kv, uint32_t schedule_groups,
        uint32_t physical_blocks) {
    return n_kv > 0 && n_heads == n_kv * kGqaHeads
        && schedule_groups > 0 && physical_blocks >= n_kv;
}

// Return the intersection between one physical Stream-K block and one of the
// (at most two) logical tiles touched by that block.  Work is linearized as
// [tile][32-key group], matching llama's continuous ijk scheduler.
__host__ __device__ inline bool physical_segment_bounds(
        uint32_t physical_block, uint32_t physical_blocks,
        uint32_t logical_tiles, uint32_t schedule_groups, uint32_t slot,
        uint32_t * logical_tile, uint32_t * group_begin,
        uint32_t * group_end) {
    if (physical_blocks == 0 || logical_tiles == 0 || schedule_groups == 0
            || physical_blocks < logical_tiles
            || slot >= kSegmentsPerPhysicalBlock
            || physical_block >= physical_blocks) {
        return false;
    }
    const uint64_t total_work = uint64_t(logical_tiles) * schedule_groups;
    const uint64_t work_begin = uint64_t(physical_block) * total_work
        / physical_blocks;
    const uint64_t work_end = uint64_t(physical_block + 1) * total_work
        / physical_blocks;
    if (work_begin >= work_end) return false;

    const uint32_t first_tile = uint32_t(work_begin / schedule_groups);
    const uint32_t last_tile = uint32_t((work_end - 1) / schedule_groups);
    const uint32_t tile = first_tile + slot;
    if (tile > last_tile || tile >= logical_tiles
            || last_tile - first_tile >= kSegmentsPerPhysicalBlock) {
        return false;
    }
    const uint64_t tile_begin = uint64_t(tile) * schedule_groups;
    const uint64_t tile_end = tile_begin + schedule_groups;
    const uint64_t begin = work_begin > tile_begin ? work_begin : tile_begin;
    const uint64_t end = work_end < tile_end ? work_end : tile_end;
    if (begin >= end) return false;
    *logical_tile = tile;
    *group_begin = uint32_t(begin - tile_begin);
    *group_end = uint32_t(end - tile_begin);
    return true;
}

__device__ __forceinline__ float exp_ftz(float difference) {
    return difference >= kFtzThreshold ? expf(difference) : 0.0f;
}

// Native ncols=8 fragments used by pinned Ampere FA.  Physical tile J is
// measured in 32-bit elements, so B<8,8,half2> is a logical 16x8 operand and
// C<16,4,half2> is a logical 16x8 result.
struct KqB8x8 {
    __half2 x[2];
};

struct KqC16x8 {
    float x[4];
};

struct PvC16x4 {
    __half2 x[2];
};

__device__ __forceinline__ void load_b8x8(
        KqB8x8 & tile, const __half2 * src, uint32_t stride,
        uint32_t lane) {
    int * dst = reinterpret_cast<int *>(tile.x);
    const int * address = reinterpret_cast<const int *>(src)
        + (lane % 8) * stride + ((lane / 8) * 4) % 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(dst[0]), "=r"(dst[1]) : "l"(address));
}

// K_A is row-major 16 keys x 16 dimensions. Q_B is column-major
// 16 dimensions x 8 query columns. The result is 16 keys x 8 columns in FP32.
__device__ __forceinline__ void mma_kq_ncols8(
        KqC16x8 & dst, const Half16x8 & key_a, const KqB8x8 & query_b) {
    int * d = reinterpret_cast<int *>(dst.x);
    const int * a = reinterpret_cast<const int *>(key_a.x);
    const int * b = reinterpret_cast<const int *>(query_b.x);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ __half2 movmatrix_transpose(__half2 value) {
    int result;
    const int bits = *reinterpret_cast<const int *>(&value);
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(result) : "r"(bits));
    return *reinterpret_cast<__half2 *>(&result);
}

// Pair adjacent Q columns, then transpose the 16x8 KQ result into the
// column-major logical 16x8 probability B operand used by V_A x P_B.
__device__ __forceinline__ KqB8x8 probability_b(KqC16x8 scores) {
    KqB8x8 result;
    result.x[0] = movmatrix_transpose(
        __floats2half2_rn(scores.x[0], scores.x[1]));
    result.x[1] = movmatrix_transpose(
        __floats2half2_rn(scores.x[2], scores.x[3]));
    return result;
}

// V_A is row-major 16 output values x 16 keys. P_B is column-major
// 16 keys x 8 query columns. Accumulation and output are FP16, exactly as in
// pinned tile<16,4,half2>.
__device__ __forceinline__ void mma_vp_ncols8(
        PvC16x4 & dst, const Half16x8 & value_a,
        const KqB8x8 & probability_b_fragment) {
    int * d = reinterpret_cast<int *>(dst.x);
    const int * a = reinterpret_cast<const int *>(value_a.x);
    const int * b = reinterpret_cast<const int *>(
        probability_b_fragment.x);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t kq_key_row(
        uint32_t lane, uint32_t element) {
    return (element / 2) * 8 + lane / 4;
}

__device__ __forceinline__ uint32_t kq_column(
        uint32_t lane, uint32_t element) {
    return (lane % 4) * 2 + element % 2;
}

__device__ __forceinline__ uint32_t pv_output_row(
        uint32_t lane, uint32_t element) {
    return element * 8 + lane / 4;
}

// `q` is already FP32-scaled and rounded to F16, laid out [head][512].
// K/V are physical-cache-major F16
// rows with `kv_width` elements; `page_table` maps only their logical row
// address. `schedule_groups * 32` is the padded logical scan span, while
// `valid_span` bounds resident logical rows. The launch must be
// <<<physical_blocks, 64>>> and must first pass `supports_schedule`.
// Keep the pre-paging 13-argument identity kernel as a separate CUDA entry
// point. Appended page/control arguments change SM86 register allocation and
// have failed strict decode agreement. Paged symbols expand the same arithmetic
// body but map only K/V row addresses.
__global__ __launch_bounds__(kThreads, 4) void partial_f16(
        const __half * q, const __half * kc, const __half * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks) {
#include "attention_decode_mma_d512_f16_body.inc"
}

__global__ __launch_bounds__(kThreads, 4) void partial_f16_paged(
        const __half * q, const __half * kc, const __half * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks,
        const uint32_t * page_table) {
#define IMPARO_D512_PAGED 1
#include "attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PAGED
}

__global__ __launch_bounds__(kThreads, 4) void partial_f16_controlled(
        const __half * q, const __half * kc, const __half * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks,
        const uint32_t * decode_control) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        start_pos = decode_control[0];
        valid_span = min(start_pos + 1, valid_span);
    }
#else
    (void)decode_control;
#endif
#include "attention_decode_mma_d512_f16_body.inc"
}

__global__ __launch_bounds__(kThreads, 4) void partial_f16_controlled_paged(
        const __half * q, const __half * kc, const __half * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks,
        const uint32_t * decode_control,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        start_pos = decode_control[0];
        valid_span = min(start_pos + 1, valid_span);
    }
#else
    (void)decode_control;
#endif
#define IMPARO_D512_PAGED 1
#include "attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PAGED
}
// Launch as <<<n_kv, 4, 256>>>.  Each block owns one live query head and
// combines all physical partials in descending key-work order.  This is the
// general (non-uniform-safe) counterpart of llama's reverse uniform fixup.
__global__ void combine_reverse(
        const float * workspace, float * out, uint32_t n_heads,
        uint32_t n_kv, uint32_t schedule_groups,
        uint32_t physical_blocks) {
    const uint32_t logical_tile = blockIdx.x;
    const uint32_t column = blockIdx.y;
    if (logical_tile >= n_kv || column >= kLiveColumns
            || n_heads != n_kv * kGqaHeads
            || physical_blocks < n_kv || schedule_groups == 0) {
        return;
    }
    const uint32_t head = logical_tile * kGqaHeads + column;
    bool have_partial = false;
    float combined_max = kInitialMax;
    float combined_sum = 0.0f;
    float numerator[2] = {0.0f, 0.0f};
    const uint32_t output0 = 2 * threadIdx.x;

    for (uint32_t block_rev = physical_blocks; block_rev > 0; --block_rev) {
        const uint32_t physical_block = block_rev - 1;
        for (uint32_t slot_rev = kSegmentsPerPhysicalBlock;
             slot_rev > 0; --slot_rev) {
            const uint32_t slot = slot_rev - 1;
            uint32_t owner = 0;
            uint32_t group_begin = 0;
            uint32_t group_end = 0;
            if (!physical_segment_bounds(physical_block, physical_blocks,
                    n_kv, schedule_groups, slot, &owner,
                    &group_begin, &group_end)
                    || owner != logical_tile) {
                continue;
            }
            const float * partial = workspace
                + (uint64_t(physical_block) * kSegmentsPerPhysicalBlock + slot)
                    * kSlotStrideFloats;
            const float partial_max = partial[kSlotNumeratorFloats + column];
            const float partial_sum = partial[
                kSlotNumeratorFloats + kColumns + column];
            if (!have_partial) {
                if (output0 < kHeadDim) {
                    numerator[0] = partial[
                        uint64_t(column) * kHeadDim + output0];
                }
                if (output0 + 1 < kHeadDim) {
                    numerator[1] = partial[
                        uint64_t(column) * kHeadDim + output0 + 1];
                }
                combined_max = partial_max;
                combined_sum = partial_sum;
                have_partial = true;
                continue;
            }
            const float max_new = fmaxf(combined_max, partial_max);
            const float scale_value = exp_ftz(combined_max - max_new);
            const float scale_add = exp_ftz(partial_max - max_new);
            if (output0 < kHeadDim) {
                numerator[0] = fmaf(scale_value, numerator[0],
                    scale_add * partial[
                        uint64_t(column) * kHeadDim + output0]);
            }
            if (output0 + 1 < kHeadDim) {
                numerator[1] = fmaf(scale_value, numerator[1],
                    scale_add * partial[
                        uint64_t(column) * kHeadDim + output0 + 1]);
            }
            combined_sum = fmaf(
                scale_value, combined_sum, scale_add * partial_sum);
            combined_max = max_new;
        }
    }

    if (!have_partial) return;
    if (output0 < kHeadDim) {
        out[uint64_t(head) * kHeadDim + output0] = combined_sum > 0.0f
            ? numerator[0] / combined_sum : 0.0f;
    }
    if (output0 + 1 < kHeadDim) {
        out[uint64_t(head) * kHeadDim + output0 + 1]
            = combined_sum > 0.0f ? numerator[1] / combined_sum : 0.0f;
    }
}

} // namespace imparo_sm80_d512_decode
