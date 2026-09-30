#pragma once
#include "sm86/attention_decode_mma_d512_m3_shared_q4.cuh"

namespace imparo_sm86_d512_register_fragments {
using namespace imparo_sm80_d512_decode;
using imparo_sm86_d512_pipeline::prefetch_packed_tile;

// Match load_half16x8's four non-transposed 8x8 fragment slots using the
// unchanged scalar F32-fmaf -> half Q4 decoder. Every warp decodes its own
// operands; compressed K/V are still fetched only once by the original pair.
__device__ __forceinline__ Half16x8 direct_packed_q4_fragment(
        const uint8_t * packed, uint32_t group, uint32_t subwarp,
        uint32_t lane, uint32_t valid_span, uint32_t d0) {
    const uint32_t row = lane >> 2;
    const uint32_t pair_d = 2 * (lane & 3u);
    const uint32_t key_base = group * kKeyBatch + subwarp * kKeysPerWarp;
    const __half2 zero = __float2half2_rn(0.0f);
    Half16x8 value;
    value.x[0] = zero;
    value.x[1] = zero;
    value.x[2] = zero;
    value.x[3] = zero;
    if (key_base + row < valid_span) {
        value.x[0] = load_q4_pair(packed, kHeadDim, row, d0 + pair_d);
        value.x[2] = load_q4_pair(packed, kHeadDim, row, d0 + pair_d + 8);
    }
    if (key_base + row + 8 < valid_span) {
        value.x[1] = load_q4_pair(packed, kHeadDim, row + 8, d0 + pair_d);
        value.x[3] = load_q4_pair(packed, kHeadDim, row + 8, d0 + pair_d + 8);
    }
    return value;
}

__global__ __launch_bounds__(128, 2) void partial(
        const __half * q, const uint8_t * kc, const uint8_t * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos, uint32_t window, uint32_t ring,
        uint32_t valid_span, uint32_t schedule_groups, uint32_t physical_blocks,
        float * workspace_single) {
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PACKED_PIPELINE 1
#define IMPARO_D512_REGISTER_FRAGMENTS 1
#include "attention_decode_mma_d512_m3_shared_q4_body.inc"
#undef IMPARO_D512_REGISTER_FRAGMENTS
#undef IMPARO_D512_PACKED_PIPELINE
#undef IMPARO_D512_Q4
}
} // namespace imparo_sm86_d512_register_fragments
