#pragma once

#include "attention_decode_mma_d512_q4.cuh"

// Isolated two-query experiment: original eight-column MMA tile, shared Q4
// K/V loads, four GQA heads per query. No target or helper dispatch changes.
// Caller must preserve EACH original M1 schedule_groups/physical_blocks and
// verify both queries use the same schedule. Linear full attention only:
// window=ring=0, valid_span=start_pos+2, Q has 2*n_heads*512 half elements.
// Workspace is the original workspace_floats(physical_blocks), no extra plane.
// Launch partial <<<physical_blocks,64>>>; combine <<<dim3(n_kv,8),256>>>.
namespace imparo_sm80_d512_pair {
using namespace imparo_sm80_d512_decode;

__global__ __launch_bounds__(kThreads, 4) void partial_q4_pair(
        const __half * q, const uint8_t * kc, const uint8_t * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks) {
#define IMPARO_D512_Q4 1
#include "attention_decode_mma_d512_pair_q4_body.inc"
#undef IMPARO_D512_Q4
}

// Original reverse combine, with eight live columns and query-major output.
__global__ void combine_pair(
        const float * workspace, float * out, uint32_t n_heads,
        uint32_t n_kv, uint32_t schedule_groups,
        uint32_t physical_blocks) {
    const uint32_t logical_tile = blockIdx.x;
    const uint32_t column = blockIdx.y;
    if (logical_tile >= n_kv || column >= kColumns
            || n_heads != n_kv * kGqaHeads
            || physical_blocks < n_kv || schedule_groups == 0) {
        return;
    }
    const uint32_t query_row = column / kGqaHeads;
    const uint32_t head = logical_tile * kGqaHeads + column % kGqaHeads;
    bool have_partial = false;
    float combined_max = kInitialMax;
    float combined_sum = 0.0f;
    float numerator[2] = {0.0f, 0.0f};
    const uint32_t output0 = 2 * threadIdx.x;

    // Divisible nonempty schedules assign a contiguous block range to each
    // logical tile, with only slot zero live. Enumerate that range directly;
    // all other schedules keep the general reverse intersection walk below.
    const uint32_t blocks_per_tile = physical_blocks / n_kv;
    const bool direct = physical_blocks % n_kv == 0
        && schedule_groups >= blocks_per_tile;
    const uint32_t first_block = direct ? logical_tile * blocks_per_tile : 0;
    const uint32_t end_block = direct
        ? (logical_tile + 1) * blocks_per_tile : physical_blocks;
    for (uint32_t block_rev = end_block; block_rev > first_block; --block_rev) {
        const uint32_t physical_block = block_rev - 1;
        for (uint32_t slot_rev = direct ? 1 : kSegmentsPerPhysicalBlock;
             slot_rev > 0; --slot_rev) {
            const uint32_t slot = slot_rev - 1;
            uint32_t owner = 0;
            uint32_t group_begin = 0;
            uint32_t group_end = 0;
            if (!direct && (!physical_segment_bounds(physical_block, physical_blocks,
                    n_kv, schedule_groups, slot, &owner,
                    &group_begin, &group_end)
                    || owner != logical_tile)) {
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
        out[(uint64_t(query_row) * n_heads + head) * kHeadDim + output0] = combined_sum > 0.0f
            ? numerator[0] / combined_sum : 0.0f;
    }
    if (output0 + 1 < kHeadDim) {
        out[(uint64_t(query_row) * n_heads + head) * kHeadDim + output0 + 1]
            = combined_sum > 0.0f ? numerator[1] / combined_sum : 0.0f;
    }
}

} // namespace imparo_sm80_d512_pair
