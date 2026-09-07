#pragma once

// SM80+ small-batch Q4_0 x Q8_1 projection. This mirrors llama CUDA's MMVQ
// scheduling for 2..8 columns: one block owns two output rows, all warps visit
// the same activation columns, and warp 0 combines the per-warp partial sums
// before the XOR tree reduction. The schedule is part of the numerical contract;
// routing these shapes through the tensor-core MMQ kernel changes final logits.
namespace imparo_sm80_mmvq {

// This boundary is a numerical dispatch contract, not a performance guess:
// pinned llama CUDA uses MMVQ through eight columns and MMQ above it. Keep the
// architecture-owned limit next to the kernels so common routing does not copy
// an SM-specific literal.
constexpr uint32_t kMaxTokens = 8;

// Single-token projection with an architecture-selected, compile-time warp count.
// The generic fallback accepts a runtime warp count, which prevents nvcc from
// specializing the weight stride and cross-warp reduction.  Keep the arithmetic
// order identical: one CTA owns one row, each thread visits the same Q4 blocks,
// warp 0 adds partials in ascending warp order, then performs the XOR tree.
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
        sum += dot_q4_0_q8_1_half(
            w + (uint64_t(row) * blocks + block) * 18, x + block, iqs);
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

__device__ __forceinline__ float dot_q4_tm_q8_1_half(
        const uint8_t * payload, __half scale,
        const BlockQ8_1 * q8, uint32_t iqs) {
    const uint16_t * q4i = reinterpret_cast<const uint16_t *>(payload);
    const int * q8i = reinterpret_cast<const int *>(q8->qs);
    int sumi = 0;
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t j = iqs + i;
        const int packed = int(q4i[2 * j]) | (int(q4i[2 * j + 1]) << 16);
        const int lo = packed & 0x0F0F0F0F;
        const int hi = (packed >> 4) & 0x0F0F0F0F;
        sumi = __dp4a(lo, q8i[iqs + i], sumi);
        sumi = __dp4a(hi, q8i[iqs + i + 4], sumi);
    }
    return __half2float(scale)
        * (float(sumi) * __half2float(q8->d) - 4.0f * __half2float(q8->s));
}

// Q4 shadow layout mirrors the source Q8 TileMajor row grouping while splitting
// nibbles and scales into contiguous planes. One CTA computes eight rows and
// cooperatively stages 64 K32 blocks; this preserves the canonical MMVQ block
// traversal and reduction order while amortizing activation and launch overhead.
__launch_bounds__(128, 1)
__global__ void q4_tm_q8_1_decode_rows8(
        const uint8_t * __restrict__ w, const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y, uint32_t n_in, uint32_t n_out,
        uint32_t row_base) {
    constexpr uint32_t NRows = 8;
    constexpr uint32_t NWarps = 4;
    constexpr uint32_t StageBlocks = 64;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row0 = blockIdx.x * NRows;
    const uint32_t blocks = n_in / 32;
    const uint64_t records = uint64_t(n_out) * blocks;
    const uint8_t * payload_src =
        w + uint64_t(blockIdx.x) * blocks * NRows * 16;
    const uint8_t * scale_src = w + records * 16
        + uint64_t(blockIdx.x) * blocks * NRows * sizeof(__half);
    extern __shared__ uint8_t shared[];
    uint8_t * payload = shared;
    auto * scales = reinterpret_cast<__half *>(
        shared + StageBlocks * NRows * 16);
    float sums[NRows] = {};
    const uint32_t iqs = 2 * (tid & 1);

    for (uint32_t stage = 0; stage < blocks; stage += StageBlocks) {
        const uint32_t valid = min(StageBlocks, blocks - stage);
        auto * payload4 = reinterpret_cast<uint4 *>(payload);
        const auto * payload_src4 = reinterpret_cast<const uint4 *>(
            payload_src + uint64_t(stage) * NRows * 16);
        for (uint32_t item = tid; item < valid * NRows; item += 128) {
            payload4[item] = payload_src4[item];
        }
        auto * scale4 = reinterpret_cast<uint4 *>(scales);
        const auto * scale_src4 = reinterpret_cast<const uint4 *>(
            scale_src + uint64_t(stage) * NRows * sizeof(__half));
        for (uint32_t item = tid; item < valid; item += 128) {
            scale4[item] = scale_src4[item];
        }
        __syncthreads();

        const uint32_t local_block = tid / 2;
        if (local_block < valid) {
            const BlockQ8_1 * xb = x + stage + local_block;
#pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                sums[local_row] += dot_q4_tm_q8_1_half(
                    payload + (uint64_t(local_block) * NRows + local_row) * 16,
                    scales[local_block * NRows + local_row], xb, iqs);
            }
        }
        __syncthreads();
    }

    auto * partial = reinterpret_cast<float (*)[NRows][32]>(shared);
    if (warp > 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            partial[warp - 1][local_row][lane] = sums[local_row];
        }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (uint32_t other = 0; other < NWarps - 1; ++other) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            sums[local_row] += partial[other][local_row][lane];
        }
    }
#pragma unroll
    for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
        sums[local_row] = warp_sum_xor(sums[local_row]);
        if (lane == 0 && row0 + local_row < n_out) {
            y[row_base + row0 + local_row] = sums[local_row];
        }
    }
}

inline void launch_tile_major_decode(
        const uint8_t * w, const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t row_base,
        cudaStream_t stream) {
    constexpr uint32_t shared_bytes = 64 * 8 * (16 + sizeof(__half));
    q4_tm_q8_1_decode_rows8<<<(n_out + 7) / 8, dim3(32, 4),
        shared_bytes, stream>>>(w, x, y, n_in, n_out, row_base);
}

__device__ __forceinline__ float dot_q5_parts_q8_1_half(
        const uint8_t * low, uint32_t high, __half scale,
        const BlockQ8_1 * q8, uint32_t iqs) {
    const uint16_t * q5i = reinterpret_cast<const uint16_t *>(low);
    const int * q8i = reinterpret_cast<const int *>(q8->qs);
    int sumi = 0;
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t j = iqs + i;
        const int packed = int(q5i[2 * j]) | (int(q5i[2 * j + 1]) << 16);
        const int vh = int(high >> (4 * j));
        int lo = packed & 0x0F0F0F0F;
        lo |= (vh << 4) & 0x00000010;
        lo |= (vh << 11) & 0x00001000;
        lo |= (vh << 18) & 0x00100000;
        lo |= (vh << 25) & 0x10000000;
        int hi = (packed >> 4) & 0x0F0F0F0F;
        hi |= (vh >> 12) & 0x00000010;
        hi |= (vh >> 5) & 0x00001000;
        hi |= (vh << 2) & 0x00100000;
        hi |= (vh << 9) & 0x10000000;
        sumi = __dp4a(lo, q8i[iqs + i], sumi);
        sumi = __dp4a(hi, q8i[iqs + i + 4], sumi);
    }
    return __half2float(scale)
        * (float(sumi) * __half2float(q8->d) - 8.0f * __half2float(q8->s));
}

__device__ __forceinline__ uint32_t load_q5_high(const uint8_t * qh) {
    return uint32_t(qh[0]) | (uint32_t(qh[1]) << 8)
        | (uint32_t(qh[2]) << 16) | (uint32_t(qh[3]) << 24);
}

__launch_bounds__(128, 1)
__global__ void q5_q8_1_decode(
        const uint8_t * __restrict__ w, const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y, uint32_t n_in, uint32_t n_out,
        uint32_t row_base) {
    constexpr uint32_t NWarps = 4;
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);
    float sum = 0.0f;
    for (uint32_t block = tid / 2; block < blocks; block += 64) {
        const uint8_t * q5 =
            w + (uint64_t(row) * blocks + block) * 22;
        sum += dot_q5_parts_q8_1_half(
            q5 + 6, load_q5_high(q5 + 2),
            *reinterpret_cast<const __half *>(q5), x + block, iqs);
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

inline void launch_q5_decode(
        const uint8_t * w, const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t row_base,
        cudaStream_t stream) {
    q5_q8_1_decode<<<n_out, dim3(32, 4), 0, stream>>>(
        w, x, y, n_in, n_out, row_base);
}

__launch_bounds__(128, 1)
__global__ void q5_tm_q8_1_decode_rows8(
        const uint8_t * __restrict__ w, const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y, uint32_t n_in, uint32_t n_out,
        uint32_t row_base) {
    constexpr uint32_t NRows = 8;
    constexpr uint32_t NWarps = 4;
    constexpr uint32_t StageBlocks = 64;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row0 = blockIdx.x * NRows;
    const uint32_t blocks = n_in / 32;
    const uint64_t records = uint64_t(n_out) * blocks;
    const uint8_t * low_src =
        w + uint64_t(blockIdx.x) * blocks * NRows * 16;
    const uint8_t * high_src = w + records * 16
        + uint64_t(blockIdx.x) * blocks * NRows * 4;
    const uint8_t * scale_src = w + records * 20
        + uint64_t(blockIdx.x) * blocks * NRows * sizeof(__half);
    extern __shared__ uint8_t shared[];
    uint8_t * low = shared;
    uint8_t * high = low + StageBlocks * NRows * 16;
    auto * scales = reinterpret_cast<__half *>(
        high + StageBlocks * NRows * 4);
    float sums[NRows] = {};
    const uint32_t iqs = 2 * (tid & 1);

    for (uint32_t stage = 0; stage < blocks; stage += StageBlocks) {
        const uint32_t valid = min(StageBlocks, blocks - stage);
        auto * low4 = reinterpret_cast<uint4 *>(low);
        const auto * low_src4 = reinterpret_cast<const uint4 *>(
            low_src + uint64_t(stage) * NRows * 16);
        for (uint32_t item = tid; item < valid * NRows; item += 128) {
            low4[item] = low_src4[item];
        }
        auto * high4 = reinterpret_cast<uint4 *>(high);
        const auto * high_src4 = reinterpret_cast<const uint4 *>(
            high_src + uint64_t(stage) * NRows * 4);
        for (uint32_t item = tid; item < valid * 2; item += 128) {
            high4[item] = high_src4[item];
        }
        auto * scale4 = reinterpret_cast<uint4 *>(scales);
        const auto * scale_src4 = reinterpret_cast<const uint4 *>(
            scale_src + uint64_t(stage) * NRows * sizeof(__half));
        for (uint32_t item = tid; item < valid; item += 128) {
            scale4[item] = scale_src4[item];
        }
        __syncthreads();

        const uint32_t local_block = tid / 2;
        if (local_block < valid) {
            const BlockQ8_1 * xb = x + stage + local_block;
#pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                const uint32_t local_record =
                    local_block * NRows + local_row;
                sums[local_row] += dot_q5_parts_q8_1_half(
                    low + uint64_t(local_record) * 16,
                    load_q5_high(high + uint64_t(local_record) * 4),
                    scales[local_record], xb, iqs);
            }
        }
        __syncthreads();
    }

    auto * partial = reinterpret_cast<float (*)[NRows][32]>(shared);
    if (warp > 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            partial[warp - 1][local_row][lane] = sums[local_row];
        }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (uint32_t other = 0; other < NWarps - 1; ++other) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            sums[local_row] += partial[other][local_row][lane];
        }
    }
#pragma unroll
    for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
        sums[local_row] = warp_sum_xor(sums[local_row]);
        if (lane == 0 && row0 + local_row < n_out) {
            y[row_base + row0 + local_row] = sums[local_row];
        }
    }
}

inline void launch_q5_tile_major_decode(
        const uint8_t * w, const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t row_base,
        cudaStream_t stream) {
    constexpr uint32_t shared_bytes = 64 * 8 * (16 + 4 + sizeof(__half));
    q5_tm_q8_1_decode_rows8<<<(n_out + 7) / 8, dim3(32, 4),
        shared_bytes, stream>>>(w, x, y, n_in, n_out, row_base);
}

// Decode-only TileMajor gated projection. Gate and Up reuse one CTA and one
// staging allocation sequentially, preserving each projection's established
// K32 traversal and reduction order while removing the second launch and the
// materialized Up buffer.
template <bool Q5>
__device__ __forceinline__ void accumulate_tm_rows8(
        const uint8_t * __restrict__ w,
        const BlockQ8_1 * __restrict__ x,
        uint32_t n_in, uint32_t n_out, uint8_t * shared,
        float (&sums)[8]) {
    constexpr uint32_t NRows = 8;
    constexpr uint32_t StageBlocks = 64;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint64_t records = uint64_t(n_out) * blocks;
    const uint8_t * low_src =
        w + uint64_t(blockIdx.x) * blocks * NRows * 16;
    const uint8_t * high_src = Q5
        ? w + records * 16 + uint64_t(blockIdx.x) * blocks * NRows * 4
        : nullptr;
    const uint8_t * scale_src = w + records * (Q5 ? 20 : 16)
        + uint64_t(blockIdx.x) * blocks * NRows * sizeof(__half);
    uint8_t * low = shared;
    uint8_t * high = Q5 ? low + StageBlocks * NRows * 16 : nullptr;
    auto * scales = reinterpret_cast<__half *>(
        low + StageBlocks * NRows * (Q5 ? 20 : 16));
    const uint32_t iqs = 2 * (tid & 1);

    for (uint32_t stage = 0; stage < blocks; stage += StageBlocks) {
        const uint32_t valid = min(StageBlocks, blocks - stage);
        auto * low4 = reinterpret_cast<uint4 *>(low);
        const auto * low_src4 = reinterpret_cast<const uint4 *>(
            low_src + uint64_t(stage) * NRows * 16);
        for (uint32_t item = tid; item < valid * NRows; item += 128) {
            low4[item] = low_src4[item];
        }
        if constexpr (Q5) {
            auto * high4 = reinterpret_cast<uint4 *>(high);
            const auto * high_src4 = reinterpret_cast<const uint4 *>(
                high_src + uint64_t(stage) * NRows * 4);
            for (uint32_t item = tid; item < valid * 2; item += 128) {
                high4[item] = high_src4[item];
            }
        }
        auto * scale4 = reinterpret_cast<uint4 *>(scales);
        const auto * scale_src4 = reinterpret_cast<const uint4 *>(
            scale_src + uint64_t(stage) * NRows * sizeof(__half));
        for (uint32_t item = tid; item < valid; item += 128) {
            scale4[item] = scale_src4[item];
        }
        __syncthreads();

        const uint32_t local_block = tid / 2;
        if (local_block < valid) {
            const BlockQ8_1 * xb = x + stage + local_block;
#pragma unroll
            for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
                const uint32_t local_record = local_block * NRows + local_row;
                if constexpr (Q5) {
                    sums[local_row] += dot_q5_parts_q8_1_half(
                        low + uint64_t(local_record) * 16,
                        load_q5_high(high + uint64_t(local_record) * 4),
                        scales[local_record], xb, iqs);
                } else {
                    sums[local_row] += dot_q4_tm_q8_1_half(
                        low + uint64_t(local_record) * 16,
                        scales[local_record], xb, iqs);
                }
            }
        }
        __syncthreads();
    }
}

template <bool Q5>
__launch_bounds__(128, 1)
__global__ void q45_tm_q8_1_gated_decode_rows8(
        const uint8_t * __restrict__ gate_w,
        const uint8_t * __restrict__ up_w,
        const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y, uint32_t n_in, uint32_t n_out) {
    constexpr uint32_t NRows = 8;
    constexpr uint32_t NWarps = 4;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t row0 = blockIdx.x * NRows;
    extern __shared__ uint8_t shared[];
    float gate[NRows] = {};
    float up[NRows] = {};

    accumulate_tm_rows8<Q5>(gate_w, x, n_in, n_out, shared, gate);
    accumulate_tm_rows8<Q5>(up_w, x, n_in, n_out, shared, up);

    auto * partial = reinterpret_cast<float (*)[2][NRows][32]>(shared);
    if (warp > 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            partial[warp - 1][0][local_row][lane] = gate[local_row];
            partial[warp - 1][1][local_row][lane] = up[local_row];
        }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (uint32_t other = 0; other < NWarps - 1; ++other) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
            gate[local_row] += partial[other][0][local_row][lane];
            up[local_row] += partial[other][1][local_row][lane];
        }
    }
#pragma unroll
    for (uint32_t local_row = 0; local_row < NRows; ++local_row) {
        gate[local_row] = warp_sum_xor(gate[local_row]);
        up[local_row] = warp_sum_xor(up[local_row]);
        if (lane == 0 && row0 + local_row < n_out) {
            y[row0 + local_row] =
                imparo_cuda_lfm2::silu(gate[local_row]) * up[local_row];
        }
    }
}

inline void launch_tile_major_gated_decode(
        const uint8_t * gate_w, const uint8_t * up_w,
        const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, bool q5,
        cudaStream_t stream) {
    const uint32_t shared_bytes = 64 * 8
        * (16 + (q5 ? 4 : 0) + sizeof(__half));
    if (q5) {
        q45_tm_q8_1_gated_decode_rows8<true>
            <<<(n_out + 7) / 8, dim3(32, 4), shared_bytes, stream>>>(
                gate_w, up_w, x, y, n_in, n_out);
    } else {
        q45_tm_q8_1_gated_decode_rows8<false>
            <<<(n_out + 7) / 8, dim3(32, 4), shared_bytes, stream>>>(
                gate_w, up_w, x, y, n_in, n_out);
    }
}

// Decode-only paired projection. The two dot products retain the exact single-MMVQ
// traversal/reduction order, but share the Q8_1 activation and launch. This is a
// semantic gated-linear primitive: it contains no model shape or buffer identifiers.
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

// Two independent equal-shape projections share activation quantization and one launch.
// Each CTA still owns exactly one output row of one projection, so its arithmetic order
// is identical to the standalone decode MMVQ.
template <uint32_t NWarps>
__global__ void q4_q8_1_pair_decode(
        const uint8_t * first_w, const uint8_t * second_w,
        const BlockQ8_1 * x, float * first_y, float * second_y,
        uint32_t n_in, uint32_t n_out) {
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered pair MMVQ warp count");
    const bool second = blockIdx.x >= n_out;
    const uint32_t row = blockIdx.x - (second ? n_out : 0);
    const uint8_t * w = second ? second_w : first_w;
    float * y = second ? second_y : first_y;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);

    float sum = 0.0f;
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        sum += dot_q4_0_q8_1_half(
            w + (uint64_t(row) * blocks + block) * 18, x + block, iqs);
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
    if (lane == 0) y[row] = sum;
}

inline void launch_pair_decode(
        const uint8_t * first_w, const uint8_t * second_w,
        const BlockQ8_1 * x, float * first_y, float * second_y,
        uint32_t n_in, uint32_t n_out, uint32_t nwarps,
        cudaStream_t stream) {
    const uint32_t blocks = 2 * n_out;
    if (nwarps == 2) {
        q4_q8_1_pair_decode<2><<<blocks, dim3(32, 2), 0, stream>>>(
            first_w, second_w, x, first_y, second_y, n_in, n_out);
    } else if (nwarps == 8) {
        q4_q8_1_pair_decode<8><<<blocks, dim3(32, 8), 0, stream>>>(
            first_w, second_w, x, first_y, second_y, n_in, n_out);
    } else {
        q4_q8_1_pair_decode<4><<<blocks, dim3(32, 4), 0, stream>>>(
            first_w, second_w, x, first_y, second_y, n_in, n_out);
    }
}

// First half of the per-layer-embedding projection. It retains the standalone
// MMVQ reduction order, then folds the two pointwise operations into the store.
// The layer vector is supplied explicitly; no model shape or buffer slot is encoded
// in this SM-specific kernel.
template <uint32_t NWarps>
__global__ void q4_q8_1_ple_gate_decode(
        const uint8_t * gate_w, const BlockQ8_1 * x, const float * per_layer,
        float * gate_y, uint32_t n_in, uint32_t ple_width,
        uint32_t per_layer_off) {
    static_assert(NWarps == 2 || NWarps == 4 || NWarps == 8,
                  "registered PLE MMVQ warp count");
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (tid & 1);

    float sum = 0.0f;
    for (uint32_t block = tid / 2; block < blocks; block += 16 * NWarps) {
        sum += dot_q4_0_q8_1_half(
            gate_w + (uint64_t(row) * blocks + block) * 18, x + block, iqs);
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
    if (lane == 0 && row < ple_width) {
        gate_y[row] = cuda_gelu(sum) * per_layer[per_layer_off + row];
    }
}

inline void launch_ple_gate_decode(
        const uint8_t * gate_w, const BlockQ8_1 * x, const float * per_layer,
        float * gate_y, uint32_t n_in, uint32_t ple_width,
        uint32_t per_layer_off, uint32_t nwarps, cudaStream_t stream) {
    if (nwarps == 2) {
        q4_q8_1_ple_gate_decode<2><<<ple_width, dim3(32, 2), 0, stream>>>(
            gate_w, x, per_layer, gate_y, n_in, ple_width, per_layer_off);
    } else if (nwarps == 8) {
        q4_q8_1_ple_gate_decode<8><<<ple_width, dim3(32, 8), 0, stream>>>(
            gate_w, x, per_layer, gate_y, n_in, ple_width, per_layer_off);
    } else {
        q4_q8_1_ple_gate_decode<4><<<ple_width, dim3(32, 4), 0, stream>>>(
            gate_w, x, per_layer, gate_y, n_in, ple_width, per_layer_off);
    }
}

// MMVQ uses the general row quantizer, not MMQ's four-values-per-thread
// quantizer. One warp owns one 32-value Q8_1 block and therefore fixes both the
// max and sum reduction order consumed by the projection below.
__global__ void quantize_q8_1(const float * x, BlockQ8_1 * y,
                              uint32_t n_in, uint32_t n_tok,
                              uint32_t src_row) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = blockIdx.y;
    if (i >= n_in || token >= n_tok) return;

    const float value = x[uint64_t(src_row + token) * n_in + i];
    float amax = fabsf(value);
    float sum = value;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? int8_t(0) : int8_t(roundf(value / d));
    const uint32_t block = i / 32;
    const uint32_t lane = i & 31;
    BlockQ8_1 * out = y + uint64_t(token) * (n_in / 32) + block;
    out->qs[lane] = q;
    if (lane == 0) {
        out->d = __float2half(d);
        out->s = __float2half(sum);
    }
}

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

// Endpoint-invariant Q4_0 x Q8_1 projection for an arbitrary prefill width.
// Each template below keeps the same per-token MMVQ reduction tree; grouping
// independent token columns only amortizes launch and weight traffic. Consume
// the largest registered group first and dispatch the final 1..7 columns to
// the matching specialization, so a token's arithmetic never depends on being
// the endpoint of its caller's batch.
inline void launch_microbatches(
        const uint8_t * w, const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base, uint32_t nwarps,
        uint32_t rows_per_cta, cudaStream_t stream) {
    const uint32_t blocks = n_in / 32;
    uint32_t token_base = 0;
    while (token_base < n_tok) {
        const uint32_t cols = min(kMaxTokens, n_tok - token_base);
        const BlockQ8_1 * xb = x + uint64_t(token_base) * blocks;
        float * yb = y + uint64_t(token_base) * out_stride;
        switch (cols) {
            case 1:
                launch_decode(w, xb, yb, n_in, n_out, row_base,
                              nwarps, rows_per_cta, stream);
                break;
            case 2: launch<2>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
            case 3: launch<3>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
            case 4: launch<4>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
            case 5: launch<5>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
            case 6: launch<6>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
            case 7: launch<7>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
            case 8: launch<8>(w, xb, yb, n_in, n_out, 0, out_stride,
                              row_base, nwarps, rows_per_cta, stream); break;
        }
        token_base += cols;
    }
}


} // namespace imparo_sm80_mmvq
