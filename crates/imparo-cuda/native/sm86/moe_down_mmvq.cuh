#pragma once
// Only expert/work-row pointers differ; the included arithmetic body is verbatim.
__launch_bounds__(128, 1)
__global__ void k_moe_down_mmvq_lab(
        const uint8_t * weights, uint64_t stride, const BlockQ8_1 * x_q8,
        float * output, const uint32_t * seg,
        uint32_t ni, uint32_t no, uint32_t rows, uint32_t ne) {
    constexpr uint32_t NWarps = 4, NRows = 2;
    if (ni != 1792 || no != 2048 || rows != 4 || ne != 32 ||
        blockDim.x != 32 || blockDim.y != NWarps || blockDim.z != 1 ||
        blockIdx.z != 0 || blockIdx.y >= rows ||
        blockIdx.x >= (no + NRows - 1) / NRows ||
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
    // A plan may contain fewer valid picks; process its valid prefix exactly
    // like grouped(), leaving unreferenced work rows untouched.
    if (expert == ne) return;
    const uint8_t * w = weights + uint64_t(expert) * stride;
    const BlockQ8_1 * x = x_q8 + uint64_t(work_row) * (ni / 32);
    float * y = output + uint64_t(work_row) * no;
    const uint32_t n_in = ni, n_out = no, row_base = 0;
#include "sm80/mmvq_q4_q8_1_decode_rows_body.inc"
}
