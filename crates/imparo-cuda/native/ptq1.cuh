#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

// PTQ1_0 format reference: PrismML-Eng/llama.cpp, prism-b10683-d8f26ee,
// ggml/src/ggml-common.h and dequantize_row_ptq1_0 in ggml-quants.c (MIT).
// Original launch/accumulation code below. GGUF type143 / Imparo wire39.
// Each row is K/128 blocks: qs[24], qh[2], fp16 scale[2]. No weight expansion.
// Callers provide already-transformed FP32 activations. This header must not
// infer or apply Hadamard metadata: embeddings and projections differ.
namespace imparo_cuda_ptq1 {

static constexpr int block_elements = 128;
static constexpr int block_bytes = 28;

__device__ __forceinline__ float scale(const uint8_t * b) {
    // Byte loads also admit an unaligned mapped tensor slice; never read past
    // this 28-byte block to obtain its trailing two-byte scale.
    const unsigned bits = unsigned(b[26]) | (unsigned(b[27]) << 8);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

__device__ __forceinline__ int digit(unsigned q, int position) {
    const unsigned power = position == 0 ? 1u : position == 1 ? 3u :
                           position == 2 ? 9u : position == 3 ? 27u : 81u;
    return int((((q * power) & 255u) * 3u) >> 8) - 1;
}

__device__ __forceinline__ float element(const uint8_t * b, int k) {
    const int byte = k < 80 ? k % 16 : k < 120 ? 16 + (k - 80) % 8 : 24 + (k - 120) % 2;
    const int pos  = k < 80 ? k / 16 : k < 120 ? (k - 80) / 8 : (k - 120) / 2;
    return float(digit(b[byte], pos)) * scale(b);
}

// Four warps produce four output channels. Each active lane owns one packed
// byte (5 trits, or 4 for the tail); its byte and scale are reused over B rows.
// This scalar-FP32 path also serves as the correctness baseline for future
// Tensor Core tiles without silently changing activation precision.
template<int B>
__global__ void matmat_kernel(const uint8_t * __restrict__ w,
                             const float * __restrict__ x,
                             float * __restrict__ y,
                             int k, int n, int m) {
    const int lane = int(threadIdx.x) & 31;
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    const int token = int(blockIdx.y) * B;
    if (row >= n) return;
    const size_t blocks = size_t(k) / block_elements;
    const uint8_t * wr = w + size_t(row) * blocks * block_bytes;
    float acc[B] = {};
    for (size_t b = 0; b < blocks; ++b) {
        if (lane < 26) {
            const uint8_t * wb = wr + b * block_bytes;
            const unsigned packed = wb[lane];
            const float d = scale(wb);
            const int start = lane < 16 ? lane : lane < 24 ? 80 + lane - 16 : 120 + lane - 24;
            const int stride = lane < 16 ? 16 : lane < 24 ? 8 : 2;
            float partial[B] = {};
            #pragma unroll
            for (int trit = 0; trit < 5; ++trit) {
                if (trit < 4 || lane < 24) {
                    const float q = float(digit(packed, trit));
                    const size_t col = b * block_elements + start + trit * stride;
                    #pragma unroll
                    for (int t = 0; t < B; ++t) {
                        if (token + t < m) partial[t] = fmaf(q, x[size_t(token + t) * k + col], partial[t]);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < B; ++t) acc[t] = fmaf(d, partial[t], acc[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < B; ++t) {
        #pragma unroll
        for (int offset = 16; offset; offset /= 2) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], offset);
        if (lane == 0 && token + t < m) y[size_t(token + t) * n + row] = acc[t];
    }
}

// Output row-major FP32, scale applied after dequantization. An out-of-range
// device row id yields zeros, never an out-of-bounds weight access. Normal
// model admission/token validation is still the caller's responsibility.
__global__ void rows_kernel(const uint8_t * __restrict__ w,
                            const uint32_t * __restrict__ ids,
                            float * __restrict__ y, int k, int n_rows,
                            int count, float multiplier, uint32_t single_row) {
    const size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(count) * k;
    if (at >= total) return;
    const int token = int(at / k);
    const int col = int(at % k);
    const uint32_t row = ids ? ids[token] : single_row;
    if (row >= uint32_t(n_rows)) { y[at] = 0.0f; return; }
    const uint8_t * b = w + (size_t(row) * (k / 128) + col / 128) * 28;
    y[at] = multiplier * element(b, col % 128);
}

inline cudaError_t launch_ptq1_matmat(const uint8_t * w, const float * x,
                                     float * y, int k, int n, int m,
                                     cudaStream_t stream) {
    if (!w || !x || !y || k <= 0 || k % 128 || n <= 0 || m <= 0 || m > 65535 * 4)
        return cudaErrorInvalidValue;
    if (m == 1) matmat_kernel<1><<<dim3((unsigned(n) + 3u) / 4u), 128, 0, stream>>>(w, x, y, k, n, m);
    else matmat_kernel<4><<<dim3((unsigned(n) + 3u) / 4u, (unsigned(m) + 3u) / 4u), 128, 0, stream>>>(w, x, y, k, n, m);
    return cudaGetLastError();
}

inline cudaError_t launch_ptq1_rows(const uint8_t * w, const uint32_t * ids,
                                   float * y, int k, int n_rows, int count,
                                   float multiplier, cudaStream_t stream) {
    if (!w || !ids || !y || k <= 0 || k % 128 || n_rows <= 0 || count <= 0)
        return cudaErrorInvalidValue;
    const size_t grid = (size_t(count) * k + 255) / 256;
    if (grid > 0x7fffffff) return cudaErrorInvalidValue;
    rows_kernel<<<unsigned(grid), 256, 0, stream>>>(w, ids, y, k, n_rows, count, multiplier, 0);
    return cudaGetLastError();
}

inline cudaError_t launch_ptq1_row(const uint8_t * w, float * y, int k,
                                  int n_rows, uint32_t row, float multiplier,
                                  cudaStream_t stream) {
    if (!w || !y || k <= 0 || k % 128 || n_rows <= 0 || row >= uint32_t(n_rows))
        return cudaErrorInvalidValue;
    rows_kernel<<<(unsigned(k) + 255u) / 256u, 256, 0, stream>>>(w, nullptr, y, k, n_rows, 1, multiplier, row);
    return cudaGetLastError();
}
} // namespace imparo_cuda_ptq1
