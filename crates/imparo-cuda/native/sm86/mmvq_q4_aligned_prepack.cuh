#pragma once
// Laboratory consumers for the byte-neutral aligned Q4 layout. They preserve
// canonical MMVQ activation quantization, block traversal and reduction order.
// Residency/ownership is handled by the caller; never pass raw Q4 bytes here.
namespace imparo_sm86_q4_aligned_mmvq {
__device__ __forceinline__ float dot_packed(const uint8_t* w,uint32_t row,uint32_t kb,uint32_t k,uint32_t m,const BlockQ8_1* q8,uint32_t iqs){
 const uint64_t record=imparo_sm86_q4_aligned_prepack::packed_record_index(row,kb,k), records=uint64_t(m)*(k/32);
 const uint32_t* q4i=reinterpret_cast<const uint32_t*>(w+records*2+record*16);
 const int* q8i=reinterpret_cast<const int*>(q8->qs); int sumi=0;
 #pragma unroll
 for(uint32_t i=0;i<2;++i){const int packed=int(q4i[iqs+i]); const int lo=packed&0x0F0F0F0F;const int hi=(packed>>4)&0x0F0F0F0F;sumi=__dp4a(lo,q8i[iqs+i],sumi);sumi=__dp4a(hi,q8i[iqs+i+4],sumi);}
 const float d4=__half2float(reinterpret_cast<const __half*>(w)[record]);
 return d4*(float(sumi)*__half2float(q8->d)-4.0f*__half2float(q8->s));
}

template <uint32_t NWarps>
__launch_bounds__(32 * NWarps, 1)
__global__ void q4_q8_1_decode(
        const uint8_t * __restrict__ w, const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y, uint32_t n_in, uint32_t n_out,
        uint32_t row_base) {
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered decode MMVQ warp count");
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);

    float sum = 0.0f;
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        sum += dot_packed(
            w, row, block, n_in, n_out, x + block, iqs);
    }

    __shared__ float partial[NWarps - 1][32];
    if (warp > 0) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (uint32_t other = 0; other < NWarps - 1; ++other) {
        sum += partial[other][lane];
    }
    sum = warp_sum_xor(sum);
    if (lane == 0 && row < n_out) y[row_base + row] = sum;
}

// Small-K projections are launch/latency limited. Adjacent output rows share one
// CTA and one activation traversal while retaining the standalone per-row
// accumulation and reduction order.
template <uint32_t NWarps, uint32_t NRows>
__launch_bounds__(32 * NWarps, 1)
__global__ void q4_q8_1_decode_rows(
        const uint8_t * __restrict__ w, const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y, uint32_t n_in, uint32_t n_out,
        uint32_t row_base) {
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
                sums[local] += dot_packed(
                    w, row0 + local, block, n_in, n_out,
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
}

inline void launch_decode(
        const uint8_t * w, const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t row_base,
        uint32_t nwarps, uint32_t rows_per_cta, cudaStream_t stream) {
    const uint32_t blocks = rows_per_cta == 4 ? (n_out + 3) / 4
                          : rows_per_cta == 2 ? (n_out + 1) / 2 : n_out;
    if (nwarps == 2) {
        if (rows_per_cta == 4) {
            q4_q8_1_decode_rows<2, 4><<<blocks, dim3(32, 2), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        } else if (rows_per_cta == 2) {
            q4_q8_1_decode_rows<2, 2><<<blocks, dim3(32, 2), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        } else {
            q4_q8_1_decode<2><<<blocks, dim3(32, 2), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        }
    } else if (nwarps == 8) {
        if (rows_per_cta == 4) {
            q4_q8_1_decode_rows<8, 4><<<blocks, dim3(32, 8), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        } else if (rows_per_cta == 2) {
            q4_q8_1_decode_rows<8, 2><<<blocks, dim3(32, 8), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        } else {
            q4_q8_1_decode<8><<<blocks, dim3(32, 8), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        }
    } else {
        if (rows_per_cta == 4) {
            q4_q8_1_decode_rows<4, 4><<<blocks, dim3(32, 4), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        } else if (rows_per_cta == 2) {
            q4_q8_1_decode_rows<4, 2><<<blocks, dim3(32, 4), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        } else {
            q4_q8_1_decode<4><<<blocks, dim3(32, 4), 0, stream>>>(
                w, x, y, n_in, n_out, row_base);
        }
    }
}

template <uint32_t NWarps>
__global__ void q4_q8_1_gated_decode(
        const uint8_t * gate_w, const uint8_t * up_w,
        const BlockQ8_1 * x, float * y, uint32_t n_in, uint32_t n_out) {
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
        gate += dot_packed(
            gate_w, row, block, n_in, n_out, xb, iqs);
        up += dot_packed(
            up_w, row, block, n_in, n_out, xb, iqs);
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
    if (lane == 0 && row < n_out) y[row] = cuda_gelu(gate) * up;
}

inline void launch_gated_decode(
        const uint8_t * gate_w, const uint8_t * up_w,
        const BlockQ8_1 * x, float * y, uint32_t n_in, uint32_t n_out,
        uint32_t nwarps, cudaStream_t stream) {
    if (nwarps == 2) {
        q4_q8_1_gated_decode<2><<<n_out, dim3(32, 2), 0, stream>>>(
            gate_w, up_w, x, y, n_in, n_out);
    } else if (nwarps == 8) {
        q4_q8_1_gated_decode<8><<<n_out, dim3(32, 8), 0, stream>>>(
            gate_w, up_w, x, y, n_in, n_out);
    } else {
        q4_q8_1_gated_decode<4><<<n_out, dim3(32, 4), 0, stream>>>(
            gate_w, up_w, x, y, n_in, n_out);
    }
}


 constexpr uint32_t kMaxTokens=8;
template <uint32_t NCols, uint32_t NWarps, uint32_t NRows>
__global__ void q4_q8_1(const uint8_t * w, const BlockQ8_1 * x, float * y,
                        uint32_t n_in, uint32_t n_out, uint32_t epilogue,
                        uint32_t out_stride, uint32_t row_base) {
    static_assert(NCols >= 2 && NCols <= kMaxTokens, "MMVQ column count");
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered batched MMVQ warp count");
    static_assert(NRows == 1 || NRows == 2 || NRows == 4,
                  "registered batched MMVQ output-row group");

    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row0 = NRows * blockIdx.x;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);

    float partial[NCols][NRows] = {};
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        #pragma unroll
        for (uint32_t token = 0; token < NCols; ++token) {
            #pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                const uint32_t row = row0 + local_row;
                if (row < n_out) {

                    partial[token][local_row] += dot_packed(
                        w, row, block, n_in, n_out, x + uint64_t(token) * blocks + block, iqs);
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
}

template <uint32_t NCols, uint32_t NRows>
inline void launch_rows(const uint8_t * w, const BlockQ8_1 * x, float * y,
                        uint32_t n_in, uint32_t n_out, uint32_t epilogue,
                        uint32_t out_stride, uint32_t row_base,
                        uint32_t nwarps, cudaStream_t stream) {
    const uint32_t blocks = (n_out + NRows - 1) / NRows;
    if (nwarps == 2) {
        q4_q8_1<NCols, 2, NRows><<<blocks, dim3(32, 2), 0, stream>>>(
            w, x, y, n_in, n_out, epilogue, out_stride, row_base);
    } else if (nwarps == 8) {
        q4_q8_1<NCols, 8, NRows><<<blocks, dim3(32, 8), 0, stream>>>(
            w, x, y, n_in, n_out, epilogue, out_stride, row_base);
    } else {
        q4_q8_1<NCols, 4, NRows><<<blocks, dim3(32, 4), 0, stream>>>(
            w, x, y, n_in, n_out, epilogue, out_stride, row_base);
    }
}

template <uint32_t NCols>
inline void launch(const uint8_t * w, const BlockQ8_1 * x, float * y,
                   uint32_t n_in, uint32_t n_out, uint32_t epilogue,
                   uint32_t out_stride, uint32_t row_base, uint32_t nwarps,
                   uint32_t rows_per_cta, cudaStream_t stream) {
    if (rows_per_cta == 1) {
        launch_rows<NCols, 1>(w, x, y, n_in, n_out, epilogue,
                              out_stride, row_base, nwarps, stream);
    } else if (rows_per_cta == 4) {
        launch_rows<NCols, 4>(w, x, y, n_in, n_out, epilogue,
                              out_stride, row_base, nwarps, stream);
    } else {
        launch_rows<NCols, 2>(w, x, y, n_in, n_out, epilogue,
                              out_stride, row_base, nwarps, stream);
    }
}


}
