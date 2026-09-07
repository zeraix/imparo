#pragma once

#include "../sm80/attention_decode_mma_d512_f16.cuh"

// SM80+ D=512 single-token attention which preserves the verified F16 MMA,
// softmax and reverse-combine order while expanding Q4_0 cache pairs directly
// into the existing shared-memory K/V tiles. This removes the global F16 cache
// mirror without changing its numerical class: every scalar uses the same
// fmaf(d, q, -8*d) followed by an independent round-to-half as kv_dequant.cuh.
namespace imparo_sm80_d512_decode {

__device__ __forceinline__ __half2 load_q4_pair(
        const uint8_t * cache, uint32_t width, uint32_t physical_row,
        uint64_t element) {
    const uint64_t blocks_per_row = width / 32;
    const uint64_t block = element / 32;
    const uint32_t lane0 = uint32_t(element & 31);
    const uint32_t lane1 = lane0 + 1;
    const uint8_t * packed = cache
        + (uint64_t(physical_row) * blocks_per_row + block) * 18;
    const float d = __half2float(
        *reinterpret_cast<const __half *>(packed));
    const uint8_t nibble0 = packed[2 + (lane0 & 15)];
    const uint8_t nibble1 = packed[2 + (lane1 & 15)];
    const int q0 = lane0 < 16 ? (nibble0 & 0x0f) : (nibble0 >> 4);
    const int q1 = lane1 < 16 ? (nibble1 & 0x0f) : (nibble1 >> 4);
    const __half value0 = __float2half(fmaf(d, float(q0), -8.0f * d));
    const __half value1 = __float2half(fmaf(d, float(q1), -8.0f * d));
    return __halves2half2(value0, value1);
}

__global__ __launch_bounds__(kThreads, 4) void partial_q4(
        const __half * q, const uint8_t * kc, const uint8_t * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks) {
#define IMPARO_D512_Q4 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_Q4
}

__global__ __launch_bounds__(kThreads, 4) void partial_q4_paged(
        const __half * q, const uint8_t * kc, const uint8_t * vc,
        float * workspace, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos,
        uint32_t window, uint32_t ring, uint32_t valid_span,
        uint32_t schedule_groups, uint32_t physical_blocks,
        const uint32_t * page_table) {
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PAGED 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PAGED
#undef IMPARO_D512_Q4
}

__global__ __launch_bounds__(kThreads, 4) void partial_q4_controlled(
        const __half * q, const uint8_t * kc, const uint8_t * vc,
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
#define IMPARO_D512_Q4 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_Q4
}

__global__ __launch_bounds__(kThreads, 4) void partial_q4_controlled_paged(
        const __half * q, const uint8_t * kc, const uint8_t * vc,
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
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PAGED 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PAGED
#undef IMPARO_D512_Q4
}

} // namespace imparo_sm80_d512_decode
