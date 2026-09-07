#pragma once

// Research-only SM86 Q8 producer A2. Eight warps each own one token pair,
// covering two adjacent 8-token layout tiles per CTA. This preserves A1's
// direct K-major stores while halving the producer CTA count.
#include "q8_mma_ready_a1.cuh"

namespace imparo_q8_mma_ready_a2_pairwarp16_research_lab {

constexpr uint32_t kProducerThreads = 256;

__launch_bounds__(kProducerThreads, 1)
__global__ void quantize_q8_mma_ready_ds4_a2_pairwarp16(
        const float * __restrict__ input,
        uint16_t * __restrict__ quant_u16,
        float * __restrict__ d8_sideplane,
        uint32_t n_in, uint32_t n_tok, uint32_t token_tiles) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t token_tile = blockIdx.x * 2 + (warp >> 2);
    if (token_tile >= token_tiles) return;
    const uint32_t local_pair = warp & 3u;
    const uint32_t qblock = lane >> 3;
    const uint32_t lane_in_qblock = lane & 7u;
    const uint32_t group = blockIdx.y;
    const uint32_t token0 = token_tile * kTokenTile + local_pair * 2;
    const uint32_t token1 = token0 + 1;
    const uint32_t local_k = qblock * kValuesPerQBlock
        + lane_in_qblock * 4;

    float4 value0{};
    float4 value1{};
    if (token0 < n_tok) {
        const uint64_t base = uint64_t(token0) * n_in
            + uint64_t(group) * kValuesPerGroup + local_k;
        value0 = reinterpret_cast<const float4 *>(input + base)[0];
    }
    if (token1 < n_tok) {
        const uint64_t base = uint64_t(token1) * n_in
            + uint64_t(group) * kValuesPerGroup + local_k;
        value1 = reinterpret_cast<const float4 *>(input + base)[0];
    }

    using namespace imparo_q8_mma_ready_a1_pairwarp_research_lab;
    const float amax0 = subgroup8_q8_amax(value0);
    const float amax1 = subgroup8_q8_amax(value1);
    const float d_inv0 = 127.0f / amax0;
    const float d_inv1 = 127.0f / amax1;
    const char4 quant0 = token0 < n_tok
        ? quantize_float4_pairwarp(value0, d_inv0) : char4{};
    const char4 quant1 = token1 < n_tok
        ? quantize_float4_pairwarp(value1, d_inv1) : char4{};

    const uint32_t packed0 = uint32_t(pack_q8_pair(quant0.x, quant0.y))
        | (uint32_t(pack_q8_pair(quant1.x, quant1.y)) << 16);
    const uint32_t packed1 = uint32_t(pack_q8_pair(quant0.z, quant0.w))
        | (uint32_t(pack_q8_pair(quant1.z, quant1.w)) << 16);
    const uint32_t kpair0 = lane_in_qblock * 2;
    const uint64_t q16_base0 = quant_u16_index(
        token_tiles, group, token_tile, qblock, kpair0, local_pair * 2);
    const uint64_t q16_base1 = quant_u16_index(
        token_tiles, group, token_tile, qblock, kpair0 + 1, local_pair * 2);
    reinterpret_cast<uint32_t *>(quant_u16 + q16_base0)[0] = packed0;
    reinterpret_cast<uint32_t *>(quant_u16 + q16_base1)[0] = packed1;

    if (lane_in_qblock == 0) {
        const float scale0 = token0 < n_tok
            ? __half2float(__float2half(1.0f / d_inv0)) : 0.0f;
        const float scale1 = token1 < n_tok
            ? __half2float(__float2half(1.0f / d_inv1)) : 0.0f;
        d8_sideplane[scale_index(
            token_tiles, group, token_tile, qblock, local_pair * 2)] = scale0;
        d8_sideplane[scale_index(
            token_tiles, group, token_tile, qblock, local_pair * 2 + 1)] = scale1;
    }
#else
    (void)input; (void)quant_u16; (void)d8_sideplane;
    (void)n_in; (void)n_tok; (void)token_tiles;
#endif
}

inline bool launch_quantize_q8_mma_ready_ds4_a2_pairwarp16(
        const float *input, uint16_t *quant_u16, float *d8_sideplane,
        uint32_t n_in, uint32_t n_tok, cudaStream_t stream,
        imparo_q8_mma_ready_a0_v1_authority_lab::Layout *layout_out = nullptr) {
    using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
    Layout layout{};
    const uint32_t padded_for_consumer =
        ((n_tok + 127u) / 128u) * 128u;
    if (!input || !quant_u16 || !d8_sideplane
            || padded_for_consumer < n_tok
            || !make_layout(n_in, padded_for_consumer, &layout)) return false;
    const uint32_t super_tiles = (layout.token_tiles + 1) / 2;
    quantize_q8_mma_ready_ds4_a2_pairwarp16
        <<<dim3(super_tiles, layout.groups), kProducerThreads, 0, stream>>>(
            input, quant_u16, d8_sideplane,
            n_in, n_tok, layout.token_tiles);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    layout.n_tok = n_tok;
    if (layout_out) *layout_out = layout;
    return true;
}

} // namespace imparo_q8_mma_ready_a2_pairwarp16_research_lab
