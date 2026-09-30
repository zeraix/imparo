#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

// Qwen-style gated DeltaNet, reusing Backend::DeltaNet's layout and CPU/Metal
// recurrence. No allocation, synchronization, host state, or graph ownership here.
// The caller resolves BufId + element offsets and validates all buffer ranges.
namespace imparo_cuda_gated_delta {

inline bool supports(uint32_t key_dim, uint32_t value_dim) {
    return key_dim == 128 && value_dim == 128;
}

__device__ __forceinline__ float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ float warp_sum(float v) {
    // Broadcast the total to every lane: each lane updates its own state columns.
    for (int mask = 16; mask; mask >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    return v;
}

// One channel owner reads all required old-history entries before advancing them.
// The raw input, never the SiLU output, enters the convolution history.
__device__ __forceinline__ float plain_value(
        const float * src, const float * state, uint64_t e,
        uint32_t ch, uint32_t width, uint32_t history) {
    return e < history ? state[e * width + ch]
                       : src[(e - history) * width + ch];
}

__global__ void plain_conv_output(
        const float * src, const float * weights, const float * state,
        float * out, uint32_t width, uint32_t kernel, uint32_t n_tok) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= uint64_t(width) * n_tok) return;
    const uint32_t t = uint32_t(index / width), ch = uint32_t(index % width);
    float sum = 0.0f;
    for (uint32_t tap = 0; tap < kernel; ++tap)
        sum += weights[uint64_t(ch) * kernel + tap]
             * plain_value(src, state, uint64_t(t) + tap, ch, width, kernel - 1);
    out[index] = sum * sigmoid(sum);
}

__global__ void plain_conv_state(
        const float * src, const float * state, float * next,
        uint32_t width, uint32_t kernel, uint32_t n_tok) {
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= width) return;
    // Ascending slots are safe when state == next: the source old slot is later.
    for (uint32_t slot = 0; slot < kernel - 1; ++slot)
        next[uint64_t(slot) * width + ch] = plain_value(
            src, state, uint64_t(n_tok) + slot, ch, width, kernel - 1);
}

inline bool conv_shape(uint32_t width, uint32_t kernel, uint32_t n_tok) {
    return width && kernel >= 2 && n_tok
        && uint64_t(width) * n_tok <= uint64_t(0x7fffffffu) * 256;
}

inline cudaError_t launch_plain_conv_snapshot(
        const float * src, const float * state_in, float * snapshot,
        uint32_t width, uint32_t kernel, uint32_t n_tok, cudaStream_t stream) {
    if (!src || !state_in || !snapshot || !conv_shape(width, kernel, n_tok))
        return cudaErrorInvalidValue;
    plain_conv_state<<<(uint64_t(width) + 255) / 256, 256, 0, stream>>>(
        src, state_in, snapshot, width, kernel, n_tok);
    return cudaPeekAtLastError();
}

inline cudaError_t launch_plain_conv(
        const float * src, const float * weights, const float * state_in,
        float * state_out, float * out, uint32_t width, uint32_t kernel,
        uint32_t n_tok, cudaStream_t stream) {
    if (!src || !weights || !state_in || !state_out || !out
        || !conv_shape(width, kernel, n_tok)) return cudaErrorInvalidValue;
    plain_conv_output<<<(uint64_t(width) * n_tok + 255) / 256, 256, 0, stream>>>(
        src, weights, state_in, out, width, kernel, n_tok);
    const cudaError_t rc = cudaPeekAtLastError();
    if (rc != cudaSuccess) return rc;
    // Stream order is the cross-grid barrier protecting every old-state read.
    return launch_plain_conv_snapshot(src, state_in, state_out,
                                      width, kernel, n_tok, stream);
}

__global__ void mul_sigmoid(
        float * a, const float * b, uint32_t width, uint32_t b_off,
        uint32_t b_stride, uint32_t a_stride, uint32_t n_rows) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= uint64_t(width) * n_rows) return;
    const uint64_t row = index / width, col = index % width;
    a[row * a_stride + col] *= sigmoid(b[row * b_stride + b_off + col]);
}

inline cudaError_t launch_mul_sigmoid(
        float * a, const float * b, uint32_t width, uint32_t b_off,
        uint32_t b_stride, uint32_t a_stride, uint32_t n_rows,
        cudaStream_t stream) {
    if (!a || !b || !width || !n_rows || a_stride < width
        || uint64_t(b_off) + width > b_stride
        || uint64_t(width) * n_rows > uint64_t(0x7fffffffu) * 256)
        return cudaErrorInvalidValue;
    mul_sigmoid<<<(uint64_t(width) * n_rows + 255) / 256, 256, 0, stream>>>(
        a, b, width, b_off, b_stride, a_stride, n_rows);
    return cudaPeekAtLastError();
}

// A block owns one complete value-head matrix. Each warp owns sixteen value rows;
// each lane holds four key columns per row. State stays in registers across tokens,
// avoiding a full state read/write per token during prefill. Four input tokens are
// staged together. This is an initial CUDA implementation, not a measured tuning
// claim about the best warp count on a particular GPU.
constexpr uint32_t kDim = 128, kWarps = 8, kRows = kDim / kWarps;
constexpr uint32_t kCols = kDim / 32, kStage = 4;

__global__ __launch_bounds__(kWarps * 32) void delta128(
        const float * qkv, const float * alpha, const float * beta,
        const float * wa, const float * dt, const float * state_in,
        float * state_out, float * out, float * snapshot, uint32_t snapshot_row,
        uint32_t k_heads, uint32_t v_heads, uint32_t n_tok, float eps) {
    const uint32_t h = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint64_t kw = uint64_t(k_heads) * kDim;
    const uint64_t vw = uint64_t(v_heads) * kDim, qkv_width = 2 * kw + vw;
    const uint64_t kb = uint64_t(h % k_heads) * kDim, vb = uint64_t(h) * kDim;
    const uint64_t sb = uint64_t(h) * kDim * kDim;
    __shared__ float tq[kStage][kDim], tk[kStage][kDim], tv[kStage][kDim];
    __shared__ float scalars[kStage][2];
    float s[kRows][kCols];
    #pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t j = warp + r * kWarps;
        #pragma unroll
        for (uint32_t c = 0; c < kCols; ++c)
            s[r][c] = state_in[sb + j * kDim + lane * kCols + c];
    }
    const float qscale = 1.0f / sqrtf(float(kDim));
    for (uint32_t t0 = 0; t0 < n_tok; t0 += kStage) {
        const uint32_t tn = min(kStage, n_tok - t0);
        // Two groups of four warps fill four tokens in two independent passes.
        for (uint32_t slot = warp / 4; slot < tn; slot += kWarps / 4) {
            const uint32_t t = t0 + slot, role = warp % 4;
            const float * row = qkv + uint64_t(t) * qkv_width;
            if (role < 2) {
                const float * x = row + (role == 0 ? kb : kw + kb);
                float vals[kCols], sq = 0.0f;
                #pragma unroll
                for (uint32_t c = 0; c < kCols; ++c) {
                    vals[c] = x[lane * kCols + c];
                    sq += vals[c] * vals[c];
                }
                const float inv = 1.0f / fmaxf(sqrtf(warp_sum(sq)), eps);
                #pragma unroll
                for (uint32_t c = 0; c < kCols; ++c) {
                    const float normed = vals[c] * inv;
                    if (role == 0) tq[slot][lane * kCols + c] = normed * qscale;
                    else tk[slot][lane * kCols + c] = normed;
                }
            } else if (role == 2) {
                #pragma unroll
                for (uint32_t c = 0; c < kCols; ++c)
                    tv[slot][lane * kCols + c] = row[2 * kw + vb + lane * kCols + c];
            } else if (lane == 0) {
                const float x = alpha[uint64_t(t) * v_heads + h] + dt[h];
                const float softplus = x > 20.0f ? x : log1pf(expf(x));
                scalars[slot][0] = expf(wa[h] * softplus);
                scalars[slot][1] = sigmoid(beta[uint64_t(t) * v_heads + h]);
            }
        }
        __syncthreads();
        for (uint32_t slot = 0; slot < tn; ++slot) {
            const float decay = scalars[slot][0], bt = scalars[slot][1];
            float q[kCols], k[kCols], acc[kRows];
            #pragma unroll
            for (uint32_t c = 0; c < kCols; ++c) {
                q[c] = tq[slot][lane * kCols + c];
                k[c] = tk[slot][lane * kCols + c];
            }
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                acc[r] = 0.0f;
                #pragma unroll
                for (uint32_t c = 0; c < kCols; ++c) {
                    s[r][c] *= decay;
                    acc[r] += s[r][c] * k[c];
                }
            }
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) acc[r] = warp_sum(acc[r]);
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                const uint32_t j = warp + r * kWarps;
                const float d = (tv[slot][j] - acc[r]) * bt;
                acc[r] = 0.0f;
                #pragma unroll
                for (uint32_t c = 0; c < kCols; ++c) {
                    s[r][c] += k[c] * d;
                    acc[r] += s[r][c] * q[c];
                }
            }
            #pragma unroll
            for (uint32_t r = 0; r < kRows; ++r) {
                acc[r] = warp_sum(acc[r]);
                if (lane == 0)
                    out[uint64_t(t0 + slot) * vw + vb + warp + r * kWarps] = acc[r];
            }
            if (snapshot && t0 + slot + 1 == snapshot_row) {
                #pragma unroll
                for (uint32_t r = 0; r < kRows; ++r) {
                    const uint32_t j = warp + r * kWarps;
                    #pragma unroll
                    for (uint32_t c = 0; c < kCols; ++c)
                        snapshot[sb + j * kDim + lane * kCols + c] = s[r][c];
                }
            }
        }
        // Every warp must finish reading this stage before its producers refill it.
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
        const uint32_t j = warp + r * kWarps;
        #pragma unroll
        for (uint32_t c = 0; c < kCols; ++c)
            state_out[sb + j * kDim + lane * kCols + c] = s[r][c];
    }
}

inline cudaError_t launch_delta(
        const float * qkv, const float * alpha, const float * beta,
        const float * wa, const float * dt, const float * state_in,
        float * state_out, float * out, float * snapshot, uint32_t snapshot_row,
        uint32_t k_heads, uint32_t v_heads, uint32_t key_dim, uint32_t value_dim,
        uint32_t n_tok, float eps, cudaStream_t stream) {
    if (!supports(key_dim, value_dim) || !qkv || !alpha || !beta || !wa || !dt
        || !state_in || !state_out || !out || !k_heads || !v_heads || !n_tok
        || v_heads > 0x7fffffffu || n_tok > 0x7fffffffu || !(eps > 0.0f) || !std::isfinite(eps)
        || (snapshot ? (!snapshot_row || snapshot_row > n_tok) : snapshot_row != 0))
        return cudaErrorInvalidValue;
    delta128<<<v_heads, kWarps * 32, 0, stream>>>(
        qkv, alpha, beta, wa, dt, state_in, state_out, out, snapshot,
        snapshot_row, k_heads, v_heads, n_tok, eps);
    return cudaPeekAtLastError();
}

} // namespace imparo_cuda_gated_delta
