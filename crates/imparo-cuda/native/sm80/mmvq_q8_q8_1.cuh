#pragma once

// SM80+ Q8_0 x Q8_1 MMVQ numerical route.
//
// The common runtime owns GGUF paging and Q8_1 scratch lifetime. This file owns the
// architecture-family scheduling: Ampere uses llama's generic MMVQ table (4 warps for
// 1--4 columns, 2 for 5--8; one output row at one column, two otherwise). Keeping it
// separate means later SM89/SM120 routes can change their launch geometry without
// changing the backend ABI or the Metal workflow.

// Included after BlockQ8_1 and warp_sum_xor are defined by imparo_cuda.cu.

namespace imparo_sm80_q8_mmvq {

__device__ __forceinline__ int load_q8_0_i32(const uint8_t * values,
                                              uint32_t word) {
    const uint8_t * p = values + 4 * word;
    return int(uint32_t(p[0]) | (uint32_t(p[1]) << 8)
        | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24));
}

__device__ __forceinline__ float dot_q8_0_q8_1_half(
        const uint8_t * values, __half weight_d,
        const BlockQ8_1 * q8_1, uint32_t iqs) {
    const int * activation = reinterpret_cast<const int *>(q8_1->qs);
    int sumi = 0;
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        sumi = __dp4a(load_q8_0_i32(values, iqs + i),
                      activation[iqs + i], sumi);
    }
    const float weight_scale = __half2float(weight_d);
    const float activation_scale = __half2float(q8_1->d);
    return weight_scale * activation_scale * float(sumi);
}

template <uint32_t NCols, uint32_t NWarps, uint32_t NRows,
          bool TileMajor = false>
__global__ void q8_0_q8_1(const uint8_t * w, const BlockQ8_1 * x, float * y,
                           uint32_t n_in, uint32_t n_out,
                           uint32_t out_stride, uint32_t row_base) {
    static_assert(NCols >= 1 && NCols <= 8, "MMVQ column count");
    static_assert(NWarps == 2 || NWarps == 4, "SM80 MMVQ warp count");
    static_assert(NRows == 1 || NRows == 2, "SM80 MMVQ row group");

    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row0 = NRows * blockIdx.x;
    const uint32_t blocks = n_in / 32;
    // Q8_0 has eight packed int32 words per wire block. Each MMVQ thread owns two.
    const uint32_t iqs = 2 * (tid & 3);

    float partial[NCols][NRows] = {};
    for (uint32_t block = tid / 4; block < blocks;
         block += 8 * NWarps) {
#pragma unroll
        for (uint32_t token = 0; token < NCols; ++token) {
#pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                const uint32_t row = row0 + local_row;
                if (row < n_out) {
                    const uint8_t * values = nullptr;
                    __half weight_d;
                    if constexpr (TileMajor) {
                        const uint64_t unit = uint64_t(row / 8) * blocks
                            + block;
                        values = w + unit * 256 + (row & 7) * 32;
                        const auto * scales = reinterpret_cast<const __half *>(
                            w + uint64_t(n_out) * n_in);
                        weight_d = scales[unit * 8 + (row & 7)];
                    } else {
                        const uint8_t * q8 = w
                            + (uint64_t(row) * blocks + block) * 34;
                        values = q8 + 2;
                        weight_d = *reinterpret_cast<const __half *>(q8);
                    }
                    partial[token][local_row] += dot_q8_0_q8_1_half(
                        values, weight_d,
                        x + uint64_t(token) * blocks + block, iqs);
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
                warp_partial[warp - 1][token][local_row][lane]
                    = partial[token][local_row];
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
            for (uint32_t other = 0; other < NWarps - 1; ++other) {
                partial[token][local_row] +=
                    warp_partial[other][token][local_row][lane];
            }
            const float result = warp_sum_xor(partial[token][local_row]);
            const uint32_t row = row0 + local_row;
            if (lane == local_row && row < n_out) {
                y[uint64_t(token) * out_stride + row_base + row] = result;
            }
        }
    }
}

// Decode-specialized TM reader. Four warps retain the row-major kernel's
// 32-way block partition and reduction tree, but one CTA owns a complete
// eight-row file tile. Every stage loads consecutive 256-byte units, and one
// activation block feeds all eight output rows.
__global__ void q8_0_tm_q8_1_single(
        const uint8_t * __restrict__ w,
        const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y,
        uint32_t n_in, uint32_t n_out,
        uint32_t out_stride, uint32_t row_base) {
    constexpr uint32_t NWarps = 4;
    constexpr uint32_t Rows = 8;
    constexpr uint32_t StageBlocks = 32;
    constexpr uint32_t PayloadBytes = StageBlocks * Rows * 32;

    extern __shared__ __align__(16) uint8_t shared[];
    uint8_t * staged_values = shared;
    auto * staged_scales = reinterpret_cast<__half *>(
        shared + PayloadBytes);
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row_tile = blockIdx.x;
    const uint32_t row0 = row_tile * Rows;
    const uint32_t blocks = n_in / 32;
    const uint64_t first_unit = uint64_t(row_tile) * blocks;
    const auto * source_scales = reinterpret_cast<const __half *>(
        w + uint64_t(n_out) * n_in);
    float partial[Rows] = {};

    for (uint32_t stage = 0; stage < blocks; stage += StageBlocks) {
        const uint32_t valid = min(StageBlocks, blocks - stage);
        const auto * source_values = reinterpret_cast<const uint4 *>(
            w + (first_unit + stage) * Rows * 32);
        auto * destination_values = reinterpret_cast<uint4 *>(staged_values);
        for (uint32_t vector = tid; vector < valid * 16;
             vector += NWarps * 32) {
            destination_values[vector] = source_values[vector];
        }
        const auto * source_scale_vectors = reinterpret_cast<const uint4 *>(
            source_scales + (first_unit + stage) * Rows);
        auto * destination_scale_vectors =
            reinterpret_cast<uint4 *>(staged_scales);
        for (uint32_t unit = tid; unit < valid;
             unit += NWarps * 32) {
            destination_scale_vectors[unit] = source_scale_vectors[unit];
        }
        __syncthreads();

        const uint32_t iqs = 2 * (tid & 3);
        for (uint32_t local_base = 0; local_base < StageBlocks;
             local_base += 8 * NWarps) {
            const uint32_t local_block = local_base + tid / 4;
            if (local_block < valid) {
                const uint32_t block = stage + local_block;
                const BlockQ8_1 * activation = x + block;
                const int * activation_words =
                    reinterpret_cast<const int *>(activation->qs);
                const float activation_scale = __half2float(activation->d);
#pragma unroll
                for (uint32_t local_row = 0; local_row < Rows; ++local_row) {
                    const uint8_t * values = staged_values
                        + local_block * Rows * 32 + local_row * 32;
                    int sumi = 0;
#pragma unroll
                    for (uint32_t item = 0; item < 2; ++item) {
                        sumi = __dp4a(load_q8_0_i32(values, iqs + item),
                                      activation_words[iqs + item], sumi);
                    }
                    partial[local_row] += __half2float(
                        staged_scales[local_block * Rows + local_row])
                        * activation_scale * float(sumi);
                }
            }
        }
        __syncthreads();
    }

    auto * warp_partial = reinterpret_cast<float (*)[Rows][32]>(shared);
    if (warp > 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < Rows; ++local_row) {
            warp_partial[warp - 1][local_row][lane] = partial[local_row];
        }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (uint32_t local_row = 0; local_row < Rows; ++local_row) {
#pragma unroll
        for (uint32_t other = 0; other < NWarps - 1; ++other) {
            partial[local_row] += warp_partial[other][local_row][lane];
        }
        const float result = warp_sum_xor(partial[local_row]);
        const uint32_t row = row0 + local_row;
        if (lane == 0 && row < n_out) {
            y[row_base + row] = result;
        }
    }
}

template <uint32_t NCols, uint32_t NWarps, uint32_t NRows, bool TileMajor>
inline void launch_shape(const uint8_t * w, const BlockQ8_1 * x, float * y,
                         uint32_t n_in, uint32_t n_out,
                         uint32_t out_stride, uint32_t row_base,
                         cudaStream_t stream) {
    q8_0_q8_1<NCols, NWarps, NRows, TileMajor>
        <<<(n_out + NRows - 1) / NRows, dim3(32, NWarps), 0, stream>>>(
            w, x, y, n_in, n_out, out_stride, row_base);
}

template <bool TileMajor>
inline void launch_layout(const uint8_t * w, const BlockQ8_1 * x, float * y,
                   uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                   uint32_t out_stride, uint32_t row_base,
                   cudaStream_t stream) {
    switch (n_tok) {
#define IMPARO_Q8_MMVQ_CASE(C, W, R) case C: \
        launch_shape<C, W, R, TileMajor>(w, x, y, n_in, n_out, \
            out_stride, row_base, stream); break
        IMPARO_Q8_MMVQ_CASE(1, 4, 1); IMPARO_Q8_MMVQ_CASE(2, 4, 2);
        IMPARO_Q8_MMVQ_CASE(3, 4, 2); IMPARO_Q8_MMVQ_CASE(4, 4, 2);
        IMPARO_Q8_MMVQ_CASE(5, 2, 2); IMPARO_Q8_MMVQ_CASE(6, 2, 2);
        IMPARO_Q8_MMVQ_CASE(7, 2, 2); IMPARO_Q8_MMVQ_CASE(8, 2, 2);
#undef IMPARO_Q8_MMVQ_CASE
    }
}

inline void launch(const uint8_t * w, const BlockQ8_1 * x, float * y,
                   uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                   uint32_t out_stride, uint32_t row_base,
                   cudaStream_t stream) {
    launch_layout<false>(w, x, y, n_in, n_out, n_tok,
                         out_stride, row_base, stream);
}

inline void launch_tile_major(const uint8_t * w, const BlockQ8_1 * x,
                              float * y, uint32_t n_in, uint32_t n_out,
                              uint32_t n_tok, uint32_t out_stride,
                              uint32_t row_base, cudaStream_t stream) {
    if (n_tok == 1) {
        constexpr uint32_t shared_bytes = 32 * 8 * 32
            + 32 * 8 * sizeof(__half);
        q8_0_tm_q8_1_single<<<(n_out + 7) / 8, dim3(32, 4),
            shared_bytes, stream>>>(w, x, y, n_in, n_out,
                                    out_stride, row_base);
        return;
    }
    launch_layout<true>(w, x, y, n_in, n_out, n_tok,
                        out_stride, row_base, stream);
}

} // namespace imparo_sm80_q8_mmvq
