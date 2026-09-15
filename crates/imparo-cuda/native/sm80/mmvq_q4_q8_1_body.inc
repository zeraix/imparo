// Shared numerical body; included by the legacy and full-tile kernels.

    static_assert(NCols >= 2 && NCols <= kMaxTokens, "MMVQ column count");
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered batched MMVQ warp count");
    static_assert(NRows == 1 || NRows == 2 || NRows == 4,
                  "registered batched MMVQ output-row group");

    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row0 = NRows * blockIdx.x;
    if constexpr (FullTiles) {
        if (row0 + NRows > n_out) return; // Uniform, before any CTA barrier.
    }
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);

    float partial[NCols][NRows] = {};
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        #pragma unroll
        for (uint32_t token = 0; token < NCols; ++token) {
            #pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                const uint32_t row = row0 + local_row;
                if (FullTiles || row < n_out) {
                    const uint8_t * q4 = w + (uint64_t(row) * blocks + block) * 18;
                    partial[token][local_row] += dot_q4_0_q8_1_half(
                        q4, x + uint64_t(token) * blocks + block, iqs);
                }
            }
        }
    }

    __shared__ float warp_partial[NWarps - 1][NCols][NRows][32];
    if (warp > 0) {
        #pragma unroll
        for (uint32_t token = 0; token < NCols; ++token) {
            #pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                warp_partial[warp - 1][token][local_row][lane] = partial[token][local_row];
            }
        }
    }
    __syncthreads();
    if (warp > 0) return;

    #pragma unroll
    for (uint32_t token = 0; token < NCols; ++token) {
        #pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            #pragma unroll
            for (uint32_t other_warp = 0; other_warp < NWarps - 1; ++other_warp) {
                partial[token][local_row] +=
                    warp_partial[other_warp][token][local_row][lane];
            }
            const float result = warp_sum_xor(partial[token][local_row]);
            const uint32_t row = row0 + local_row;
            if (lane == local_row && row < n_out) {
                float * slot = y + uint64_t(token) * out_stride + row_base + row;
                if (epilogue) {
                    const float gate = *slot;
                    *slot = cuda_gelu(gate) * result;
                } else {
                    *slot = result;
                }
            }
        }
    }
