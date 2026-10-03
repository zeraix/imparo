
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered decode MMVQ warp count");
    static_assert(NRows == 2 || NRows == 4,
                  "registered small-K output row group");
    const uint32_t row0 = NRows * blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);
    float sums[NRows] = {};
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        const BlockQ8_1 * xb = x + block;
#pragma unroll
        for (uint32_t local = 0; local < NRows; ++local) {
            if (row0 + local < n_out) {
                sums[local] += dot_q4_0_q8_1_half(
                    w + (uint64_t(row0 + local) * blocks + block) * 18,
                    xb, iqs);
            }
        }
    }
    __shared__ float partial[NWarps - 1][NRows][32];
    if (warp > 0) {
#pragma unroll
        for (uint32_t local = 0; local < NRows; ++local) {
            partial[warp - 1][local][lane] = sums[local];
        }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (uint32_t other = 0; other < NWarps - 1; ++other) {
#pragma unroll
        for (uint32_t local = 0; local < NRows; ++local) {
            sums[local] += partial[other][local][lane];
        }
    }
#pragma unroll
    for (uint32_t local = 0; local < NRows; ++local) {
        sums[local] = warp_sum_xor(sums[local]);
        if (lane == 0 && row0 + local < n_out) {
            y[row_base + row0 + local] = sums[local];
        }
    }
