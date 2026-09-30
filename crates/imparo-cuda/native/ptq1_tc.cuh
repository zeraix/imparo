#pragma once

#include "ptq1.cuh"
#include "sm80/mma_f16.cuh"

// Experimental PTQ1_0 prefill provider. The codec and on-file weight remain
// unchanged. Only an SM80+ CTA's current K128 tile is expanded to shared half.
// Activations are already transformed FP32; conversion to half is an explicit
// new numerical contract. MMA accumulation and output remain FP32.
//
// There is one configuration: M32/N64/K128, four warps, single shared stage.
// The caller owns basis, placement, lifetime, stream and capability admission.
// M<=8 is deliberately NotSupported so the existing PTQ decode path stays in use.
namespace imparo_cuda_ptq1_tc {

constexpr uint32_t kTokens = 32;
constexpr uint32_t kRows = 64;
constexpr uint32_t kStage = 128;
constexpr uint32_t kStride = 136; // half elements: every row remains 16B aligned.
constexpr uint32_t kThreads = 128;
constexpr uint32_t kSharedBytes = (kTokens + kRows) * kStride * sizeof(__half);
static_assert(kSharedBytes == 26112, "single PTQ half stage");
static_assert(kStride % 8 == 0, "ldmatrix row alignment");

__global__ __launch_bounds__(kThreads) void matmat_kernel(
        const uint8_t * __restrict__ weights,
        const float * __restrict__ input,
        float * __restrict__ output,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) __half sx[kTokens][kStride];
    __shared__ __align__(16) __half sw[kRows][kStride];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31;
    const uint32_t warp = tid >> 5;
    const uint32_t token_base = blockIdx.y * kTokens;
    const uint32_t output_base = blockIdx.x * kRows;
    const uint32_t warp_token = (warp / 2) * 16;
    const uint32_t warp_output = (warp % 2) * 32;
    const uint32_t blocks_per_row = n_in / imparo_cuda_ptq1::block_elements;
    imparo_sm80_mma::Float16x16 accum[2]{};

    for (uint32_t k_base = 0; k_base < n_in; k_base += kStage) {
        // Each source value is converted only at the shared-tile boundary.
        for (uint32_t linear = tid; linear < kTokens * kStage; linear += kThreads) {
            const uint32_t token = linear / kStage;
            const uint32_t k = linear % kStage;
            const uint32_t global_token = token_base + token;
            sx[token][k] = global_token < n_tok
                ? __float2half_rn(input[uint64_t(global_token) * n_in + k_base + k])
                : __float2half_rn(0.0f);
        }

        // A warp owns one PTQ row at a time. Its 26 active lanes own the exact
        // on-file interleaving, five/four trits per byte. No 32B read over a 28B
        // block, no global expanded-weight buffer, and one scale load per warp.
        for (uint32_t local_row = warp; local_row < kRows; local_row += 4) {
            const uint32_t global_row = output_base + local_row;
            const bool valid = global_row < n_out;
            const uint8_t * block = valid
                ? weights + (uint64_t(global_row) * blocks_per_row + k_base / 128) * 28
                : nullptr;
            float d = lane == 0 && valid ? imparo_cuda_ptq1::scale(block) : 0.0f;
            d = __shfl_sync(0xffffffffu, d, 0);
            if (lane < 26) {
                const unsigned packed = valid ? block[lane] : 0;
                const uint32_t first = lane < 16 ? lane : lane < 24
                    ? 80 + lane - 16 : 120 + lane - 24;
                const uint32_t step = lane < 16 ? 16 : lane < 24 ? 8 : 2;
#pragma unroll
                for (uint32_t digit = 0; digit < 5; ++digit) {
                    if (digit < 4 || lane < 24) {
                        const float value = valid
                            ? float(imparo_cuda_ptq1::digit(packed, int(digit))) * d
                            : 0.0f;
                        sw[local_row][first + digit * step] = __float2half_rn(value);
                    }
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (uint32_t k = 0; k < kStage; k += 16) {
            imparo_sm80_mma::Half16x8 activation;
            imparo_sm80_mma::load_half16x8(activation,
                reinterpret_cast<const __half2 *>(&sx[warp_token][k]),
                kStride / 2, lane);
#pragma unroll
            for (uint32_t tile = 0; tile < 2; ++tile) {
                imparo_sm80_mma::Half16x8 weight;
                imparo_sm80_mma::load_half16x8(weight,
                    reinterpret_cast<const __half2 *>(&sw[warp_output + tile * 16][k]),
                    kStride / 2, lane);
                imparo_sm80_mma::mma_qk(accum[tile], activation, weight);
            }
        }
        __syncthreads(); // No producer overwrites a stage still being consumed.
    }

#pragma unroll
    for (uint32_t tile = 0; tile < 2; ++tile) {
#pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            const uint32_t token = token_base + warp_token
                + imparo_sm80_mma::fragment_q_column(lane, item);
            const uint32_t row = output_base + warp_output + tile * 16
                + imparo_sm80_mma::fragment_key_row(lane, item);
            if (token < n_tok && row < n_out) {
                output[uint64_t(token) * out_stride + row_base + row] = accum[tile].x[item];
            }
        }
    }
#endif
}

// weights begins at the first row of this slice; row_base only addresses output.
// The owner must have admitted SM80+ and sufficient buffer capacities. There is
// no allocation, synchronization, hidden device query, epilogue or host readback.
inline cudaError_t launch(const uint8_t * weights, const float * input,
        float * output, uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base, cudaStream_t stream) {
    if (!weights || !input || !output || !n_in || n_in % 128
            || !n_out || !n_tok || !out_stride || row_base > out_stride
            || n_out > out_stride - row_base || n_tok > 65535u * kTokens)
        return cudaErrorInvalidValue;
    if (n_tok <= 8) return cudaErrorNotSupported;
    matmat_kernel<<<dim3(n_out / kRows + uint32_t(n_out % kRows != 0),
        n_tok / kTokens + uint32_t(n_tok % kTokens != 0)), kThreads, 0, stream>>>(
        weights, input, output, n_in, n_out, n_tok, out_stride, row_base);
    return cudaGetLastError();
}

} // namespace imparo_cuda_ptq1_tc
