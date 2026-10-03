
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered gated MMVQ warp count");
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);

    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        const BlockQ8_1 * xb = x + block;
        gate += dot_q4_0_q8_1_half(
            gate_w + (uint64_t(row) * blocks + block) * 18, xb, iqs);
        up += dot_q4_0_q8_1_half(
            up_w + (uint64_t(row) * blocks + block) * 18, xb, iqs);
    }

    __shared__ float gate_partial[NWarps - 1][32];
    __shared__ float up_partial[NWarps - 1][32];
    if (warp > 0) {
        gate_partial[warp - 1][lane] = gate;
        up_partial[warp - 1][lane] = up;
    }
    __syncthreads();
    if (warp > 0) return;

#pragma unroll
    for (uint32_t other = 0; other < NWarps - 1; ++other) {
        gate += gate_partial[other][lane];
        up += up_partial[other][lane];
    }
    gate = warp_sum_xor(gate);
    up = warp_sum_xor(up);
    if (lane == 0 && row < n_out) {
        if constexpr (Silu) y[row] = imparo_cuda_lfm2::silu(gate) * up;
        else y[row] = cuda_gelu(gate) * up;
    }
