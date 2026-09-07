#pragma once

// SM80+ large-batch Q4_0 x F32 Tensor-Core GEMM.
//
// This architecture header owns only physical tile geometry and the numerical
// route.  Common CUDA code remains responsible for policy, buffer ownership and
// error reporting.  The Q4_0 conversion deliberately matches pinned llama
// 4695f001 convert.cu:
//
//     d  = half_to_float(block.d)
//     dm = -8 * d
//     h  = half_rn(d * q + dm)
//
// Activations are independently rounded F32 -> F16.  Tensor Cores then perform
// FP16 multiply with FP16 accumulation/output.  Even when the public destination
// is F32, the fragment is first stored as F16 and only then widened; this is part
// of the numerical contract and must not be replaced by an F32 accumulator.
// Q4_0 rows are complete 32-value ABI blocks: n_in must be a multiple of 32.
// Token and output-row tile tails are padded, but a partial Q4_0 block is invalid.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace imparo_sm80_gemm_q4_f16 {

constexpr uint32_t kTileTokens = 64;
constexpr uint32_t kTileRows = 64;
constexpr uint32_t kStageK = 32;
constexpr uint32_t kSharedStride = kStageK + 8; // ldmatrix-friendly bank skew
constexpr uint32_t kWarps = 8;
constexpr uint32_t kThreads = 32 * kWarps;
constexpr uint32_t kQ4BlockValues = 32;
constexpr uint32_t kQ4BlockBytes = 18;

// 64x32 activation + 64x32 weight stages (both padded), plus two private
// 16x16 F16 output tiles per warp.  Kept explicit for launch-policy/resource
// accounting in the architecture registry.
constexpr uint32_t kSharedBytes =
    2 * kTileTokens * kSharedStride * sizeof(__half)
    + kWarps * 2 * 16 * 16 * sizeof(__half);

struct LaunchInfo {
    uint32_t tile_tokens = kTileTokens;
    uint32_t tile_rows = kTileRows;
    uint32_t stage_k = kStageK;
    uint32_t threads = kThreads;
    uint32_t shared_bytes = kSharedBytes;
};

inline constexpr LaunchInfo launch_info() {
    return {};
}

__device__ __forceinline__ __half dequantize_q4_0_half(
        const uint8_t * __restrict__ block, uint32_t index) {
    const __half dh = *reinterpret_cast<const __half *>(block);
    const float d = __half2float(dh);
    const float dm = -8.0f * d;
    const uint8_t packed = block[2 + (index & 15u)];
    const uint32_t q = index < 16u ? (packed & 0x0fu) : (packed >> 4);
    // Keep the expression identical to llama's conversion kernel.  In
    // particular, do not rewrite this as (q - 8) * d: its rounding differs.
    return __float2half_rn(d * static_cast<float>(q) + dm);
}

template <typename Dst>
__device__ __forceinline__ void store_output_value(
        Dst * __restrict__ dst, uint64_t index, __half value);

template <>
__device__ __forceinline__ void store_output_value<__half>(
        __half * __restrict__ dst, uint64_t index, __half value) {
    dst[index] = value;
}

template <>
__device__ __forceinline__ void store_output_value<float>(
        float * __restrict__ dst, uint64_t index, __half value) {
    // The widening happens after the mandated F16 output rounding.
    dst[index] = __half2float(value);
}

template <typename Dst>
__global__ __launch_bounds__(kThreads, 1) void gemm_q4_f16_kernel(
        const uint8_t * __restrict__ weights,
        const float * __restrict__ activations,
        Dst * __restrict__ output,
        uint32_t n_in,
        uint32_t n_out,
        uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;

    __shared__ __align__(16) __half activation_stage[kTileTokens][kSharedStride];
    __shared__ __align__(16) __half weight_stage[kTileRows][kSharedStride];
    __shared__ __align__(16) __half output_stage[kWarps][2][16][16];

    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t token0 = blockIdx.y * kTileTokens;
    const uint32_t row0 = blockIdx.x * kTileRows;
    const uint32_t q4_blocks_per_row = n_in / kQ4BlockValues;

    // Four token tiles x four output-row tiles.  Eight warps each retain two
    // fragments, so every shared K-stage is consumed by the whole CTA before it
    // is overwritten.  No warp reloads the same Q4 block from global memory.
    const uint32_t token_tile = warp >> 1;
    const uint32_t row_tile0 = (warp & 1u) * 2u;

    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> accum[2];
    wmma::fill_fragment(accum[0], __float2half(0.0f));
    wmma::fill_fragment(accum[1], __float2half(0.0f));

    for (uint32_t k0 = 0; k0 < n_in; k0 += kStageK) {
        for (uint32_t linear = tid;
             linear < kTileTokens * kStageK;
             linear += kThreads) {
            const uint32_t local_token = linear / kStageK;
            const uint32_t local_k = linear - local_token * kStageK;
            const uint32_t token = token0 + local_token;
            const uint32_t k = k0 + local_k;
            activation_stage[local_token][local_k] =
                token < n_tok && k < n_in
                    ? __float2half_rn(activations[uint64_t(token) * n_in + k])
                    : __float2half(0.0f);
        }

        for (uint32_t linear = tid;
             linear < kTileRows * kStageK;
             linear += kThreads) {
            const uint32_t local_row = linear / kStageK;
            const uint32_t local_k = linear - local_row * kStageK;
            const uint32_t row = row0 + local_row;
            const uint32_t k = k0 + local_k;
            __half value = __float2half(0.0f);
            if (row < n_out && k < n_in) {
                const uint64_t block_index = uint64_t(row) * q4_blocks_per_row
                    + k / kQ4BlockValues;
                value = dequantize_q4_0_half(
                    weights + block_index * kQ4BlockBytes,
                    k & (kQ4BlockValues - 1));
            }
            weight_stage[local_row][local_k] = value;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t k_sub = 0; k_sub < kStageK; k_sub += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16,
                           __half, wmma::row_major> a;
            wmma::load_matrix_sync(
                a, &activation_stage[token_tile * 16][k_sub], kSharedStride);

#pragma unroll
            for (uint32_t part = 0; part < 2; ++part) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               __half, wmma::col_major> b;
                const uint32_t local_row = (row_tile0 + part) * 16;
                // weight_stage is physically [output row][K].  Interpreting
                // that storage as col-major [K][output row] gives W^T without
                // a shared-memory transpose.
                wmma::load_matrix_sync(
                    b, &weight_stage[local_row][k_sub], kSharedStride);
                wmma::mma_sync(accum[part], a, b, accum[part]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t part = 0; part < 2; ++part) {
        wmma::store_matrix_sync(
            &output_stage[warp][part][0][0], accum[part], 16,
            wmma::mem_row_major);
        __syncwarp();

        for (uint32_t linear = lane; linear < 16 * 16; linear += 32) {
            const uint32_t local_token = linear >> 4;
            const uint32_t local_row = linear & 15u;
            const uint32_t token = token0 + token_tile * 16 + local_token;
            const uint32_t row = row0 + (row_tile0 + part) * 16 + local_row;
            if (token < n_tok && row < n_out) {
                store_output_value(
                    output, uint64_t(token) * n_out + row,
                    output_stage[warp][part][local_token][local_row]);
            }
        }
        __syncwarp();
    }
#else
    (void)weights;
    (void)activations;
    (void)output;
    (void)n_in;
    (void)n_out;
    (void)n_tok;
#endif
}

inline bool valid_shape(uint32_t n_in, uint32_t n_out, uint32_t n_tok) {
    return n_in != 0 && n_out != 0 && n_tok != 0
        && (n_in & (kQ4BlockValues - 1)) == 0
        && (n_out & 15u) == 0;
}

template <typename Dst>
inline bool launch_impl(
        const uint8_t * weights,
        const float * activations,
        Dst * output,
        uint32_t n_in,
        uint32_t n_out,
        uint32_t n_tok,
        uint32_t sm_version,
        cudaStream_t stream) {
    if (sm_version < 80 || !weights || !activations || !output
            || !valid_shape(n_in, n_out, n_tok)) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid(
        (n_out + kTileRows - 1) / kTileRows,
        (n_tok + kTileTokens - 1) / kTileTokens);
    gemm_q4_f16_kernel<<<grid, block, 0, stream>>>(
        weights, activations, output, n_in, n_out, n_tok);
    return cudaPeekAtLastError() == cudaSuccess;
}

// F16 destination preserves the Tensor-Core result bit pattern directly.
inline bool launch_f16(
        const uint8_t * weights,
        const float * activations,
        __half * output,
        uint32_t n_in,
        uint32_t n_out,
        uint32_t n_tok,
        uint32_t sm_version,
        cudaStream_t stream) {
    return launch_impl(weights, activations, output,
                       n_in, n_out, n_tok, sm_version, stream);
}

// F32 destination widens the already-rounded F16 result.  It is intended for
// Imparo's current F32 buffer ABI and is numerically identical to launch_f16
// followed by a half-to-float conversion.
inline bool launch_f32(
        const uint8_t * weights,
        const float * activations,
        float * output,
        uint32_t n_in,
        uint32_t n_out,
        uint32_t n_tok,
        uint32_t sm_version,
        cudaStream_t stream) {
    return launch_impl(weights, activations, output,
                       n_in, n_out, n_tok, sm_version, stream);
}

} // namespace imparo_sm80_gemm_q4_f16
