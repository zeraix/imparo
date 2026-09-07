#pragma once

// Research-only SM86 Q8 producer A1. Four warps each own a token pair and the
// same K4 fragment for both tokens, so the final K-major uint32 can be emitted
// directly without a shared-memory transpose or CTA barrier. The eight-lane
// amax reduction tree, int8 rounding, and f16 scale round-trip match A0.
#include "q8_mma_ready_a0.cuh"

namespace imparo_q8_mma_ready_a1_pairwarp_research_lab {

constexpr uint32_t kProducerThreads = 128;

__device__ __forceinline__ float subgroup8_q8_amax(float4 value) {
    float amax = fabsf(value.x);
    amax = fmaxf(amax, fabsf(value.y));
    amax = fmaxf(amax, fabsf(value.z));
    amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(
            amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    return amax;
}

__device__ __forceinline__ char4 quantize_float4_pairwarp(
        float4 value, float d_inv) {
    char4 quant{};
    quant.x = int8_t(roundf(value.x * d_inv));
    quant.y = int8_t(roundf(value.y * d_inv));
    quant.z = int8_t(roundf(value.z * d_inv));
    quant.w = int8_t(roundf(value.w * d_inv));
    return quant;
}

__device__ __forceinline__ uint16_t pack_q8_pair(int8_t lo, int8_t hi) {
    return uint16_t(uint8_t(lo)) | (uint16_t(uint8_t(hi)) << 8);
}

__launch_bounds__(kProducerThreads, 1)
__global__ void quantize_q8_mma_ready_ds4_a1_pairwarp(
        const float * __restrict__ input,
        uint16_t * __restrict__ quant_u16,
        float * __restrict__ d8_sideplane,
        uint32_t n_in, uint32_t n_tok, uint32_t token_tiles) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
    const uint32_t tid = threadIdx.x;
    const uint32_t token_pair = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t qblock = lane >> 3;
    const uint32_t lane_in_qblock = lane & 7u;
    const uint32_t group = blockIdx.y;
    const uint32_t token_tile = blockIdx.x;
    const uint32_t token0 = token_tile * kTokenTile + token_pair * 2;
    const uint32_t token1 = token0 + 1;
    const uint32_t local_k = qblock * kValuesPerQBlock
        + lane_in_qblock * 4;

    float4 value0{};
    float4 value1{};
    if (token0 < n_tok) {
        const uint64_t input_base0 = uint64_t(token0) * n_in
            + uint64_t(group) * kValuesPerGroup + local_k;
        value0 = reinterpret_cast<const float4 *>(input + input_base0)[0];
    }
    if (token1 < n_tok) {
        const uint64_t input_base1 = uint64_t(token1) * n_in
            + uint64_t(group) * kValuesPerGroup + local_k;
        value1 = reinterpret_cast<const float4 *>(input + input_base1)[0];
    }

    const float amax0 = subgroup8_q8_amax(value0);
    const float amax1 = subgroup8_q8_amax(value1);
    const float d_inv0 = 127.0f / amax0;
    const float d_inv1 = 127.0f / amax1;
    const char4 quant0 = token0 < n_tok
        ? quantize_float4_pairwarp(value0, d_inv0) : char4{};
    const char4 quant1 = token1 < n_tok
        ? quantize_float4_pairwarp(value1, d_inv1) : char4{};

    const uint16_t pair00 = pack_q8_pair(quant0.x, quant0.y);
    const uint16_t pair10 = pack_q8_pair(quant1.x, quant1.y);
    const uint16_t pair01 = pack_q8_pair(quant0.z, quant0.w);
    const uint16_t pair11 = pack_q8_pair(quant1.z, quant1.w);
    const uint32_t packed0 = uint32_t(pair00) | (uint32_t(pair10) << 16);
    const uint32_t packed1 = uint32_t(pair01) | (uint32_t(pair11) << 16);
    const uint32_t kpair0 = lane_in_qblock * 2;
    const uint64_t q16_base0 = quant_u16_index(
        token_tiles, group, token_tile, qblock, kpair0, token_pair * 2);
    const uint64_t q16_base1 = quant_u16_index(
        token_tiles, group, token_tile, qblock, kpair0 + 1, token_pair * 2);
    reinterpret_cast<uint32_t *>(quant_u16 + q16_base0)[0] = packed0;
    reinterpret_cast<uint32_t *>(quant_u16 + q16_base1)[0] = packed1;

    if (lane_in_qblock == 0) {
        const float rounded_scale0 = token0 < n_tok
            ? __half2float(__float2half(1.0f / d_inv0)) : 0.0f;
        const float rounded_scale1 = token1 < n_tok
            ? __half2float(__float2half(1.0f / d_inv1)) : 0.0f;
        d8_sideplane[scale_index(
            token_tiles, group, token_tile, qblock, token_pair * 2)]
            = rounded_scale0;
        d8_sideplane[scale_index(
            token_tiles, group, token_tile, qblock, token_pair * 2 + 1)]
            = rounded_scale1;
    }
#else
    (void)input; (void)quant_u16; (void)d8_sideplane;
    (void)n_in; (void)n_tok; (void)token_tiles;
#endif
}

inline bool launch_quantize_q8_mma_ready_ds4_a1_pairwarp(
        const float *input, uint16_t *quant_u16, float *d8_sideplane,
        uint32_t n_in, uint32_t n_tok, cudaStream_t stream,
        imparo_q8_mma_ready_a0_v1_authority_lab::Layout *layout_out = nullptr) {
    using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
    Layout layout{};
    if (!input || !quant_u16 || !d8_sideplane
            || !make_layout(n_in, n_tok, &layout)) {
        return false;
    }
    quantize_q8_mma_ready_ds4_a1_pairwarp
        <<<dim3(layout.token_tiles, layout.groups), kProducerThreads, 0,
           stream>>>(input, quant_u16, d8_sideplane,
                     n_in, n_tok, layout.token_tiles);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (layout_out) *layout_out = layout;
    return true;
}

} // namespace imparo_q8_mma_ready_a1_pairwarp_research_lab
