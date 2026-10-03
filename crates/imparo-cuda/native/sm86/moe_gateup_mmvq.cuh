#pragma once
// The existing gated-decode traversal/reductions are unchanged. Only expert,
// work-row and PERM pointer aliases differ; Silu selects the original helper.
__global__ void k_moe_gateup_mmvq_lab(
        const uint8_t * gate_weights, const uint8_t * up_weights, uint64_t stride,
        const BlockQ8_1 * x_q8, float * output,
        const uint32_t * perm, const uint32_t * seg,
        uint32_t ni, uint32_t no, uint32_t nt, uint32_t rows, uint32_t ne) {
    constexpr uint32_t NWarps = 4;
    constexpr bool Silu = true;
    if (ni != 2048 || no != 1792 || nt != 1 || rows != 4 || ne != 32 ||
        blockDim.x != 32 || blockDim.y != NWarps || blockDim.z != 1 ||
        blockIdx.z != 0 || blockIdx.y >= rows || blockIdx.x >= no ||
        stride != uint64_t(no) * (ni / 32) * 18) return;
    const uint32_t work_row = blockIdx.y;
    if (seg[0] != 0) return;
    uint32_t expert = ne, previous = 0;
    for (uint32_t e = 0; e < ne; ++e) {
        const uint32_t hi = seg[e + 1];
        if (hi < previous || hi > rows) return;
        if (previous <= work_row && work_row < hi) expert = e;
        previous = hi;
    }
    // Invalid picks may leave a shorter valid prefix; uncovered rows do no work.
    if (expert == ne) return;
    const uint32_t token = perm[work_row];
    if (token >= nt) {
        // Match grouped<Paired>: an invalid token has zero accumulators.
        if (threadIdx.x == 0 && threadIdx.y == 0)
            output[uint64_t(work_row) * no + blockIdx.x] = 0.0f;
        return;
    }
    const uint8_t * gate_w = gate_weights + uint64_t(expert) * stride;
    const uint8_t * up_w = up_weights + uint64_t(expert) * stride;
    const BlockQ8_1 * x = x_q8 + uint64_t(token) * (ni / 32);
    float * y = output + uint64_t(work_row) * no;
    const uint32_t n_in = ni, n_out = no;
#include "sm80/mmvq_q4_q8_1_gated_decode_body.inc"
}
