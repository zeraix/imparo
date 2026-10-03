#pragma once

// Lab only. Caller supplies the canonical Q4/Q8 MMQ definitions and launches
// grid (14, 32, 4), block (32, 8), kHalfKSharedBytes (36,864 bytes).
__global__ __launch_bounds__(256, 2) void k_moe_gateup_mmq_projection_lab(
        const uint8_t * __restrict__ weights, uint64_t stride,
        const BlockQ8_1Mmq * __restrict__ x_q8, float * __restrict__ y,
        const uint32_t * __restrict__ seg,
        uint32_t ni, uint32_t no, uint32_t rows, uint32_t ne) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    // FullK/FullRows below require the fixed lab shape and block geometry.
    if (ni != 2048 || no != 1792 || rows != 512 || ne != 32
            || blockDim.x != 32 || blockDim.y != kWarps || blockDim.z != 1)
        return;
    const uint32_t expert = blockIdx.y;
    const uint32_t tile_row = blockIdx.x * kRows;
    if (expert >= ne || tile_row >= no) return;
    const uint32_t lo = seg[expert], hi = seg[expert + 1];
    if (lo > hi || hi > rows) return;
    const uint32_t blocks = ni / 32;
    const uint8_t * expert_weights = weights + uint64_t(expert) * stride;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x, warp = threadIdx.y;
    float partial[64];
    for (uint32_t c = lo + blockIdx.z * kTokens; c < hi;
            c += gridDim.z * kTokens) {
        // rows is the physical group-major Q8 pitch; hi trims this expert only.
        compute_segment<128, 2, false, true, true, 4>(
            expert_weights, x_q8, sx, sy, partial, no, rows, hi, blocks,
            tile_row, c, 0, blocks);
#pragma unroll
        for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
            for (uint32_t token_fragment = 0; token_fragment < 2; ++token_fragment) {
#pragma unroll
                for (uint32_t row_fragment = 0; row_fragment < 2; ++row_fragment) {
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t local_row = (warp >> 1) * 32
                            + row_fragment * 16 + accumulator_row(lane, item);
                        const uint32_t local_token = token_group * 32
                            + (warp & 1) * 16 + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t token = c + local_token;
                        if (token >= hi) continue;
                        const uint32_t sum_index =
                            (((token_group * 2 + token_fragment) * 2
                                + row_fragment) * 4) + item;
                        y[uint64_t(token) * no + tile_row + local_row] =
                            partial[sum_index];
                    }
                }
            }
        }
    }
#else
    (void)weights; (void)stride; (void)x_q8; (void)y; (void)seg;
    (void)ni; (void)no; (void)rows; (void)ne;
#endif
}
