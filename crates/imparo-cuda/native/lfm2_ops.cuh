#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace imparo_cuda_lfm2 {

#if defined(__CUDACC__)
#define IMPARO_CUDA_HD __host__ __device__
#else
#define IMPARO_CUDA_HD
#endif

struct ShortconvLayout {
    uint64_t bcx_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t weight_bytes = 0;
};

inline bool checked_mul(uint64_t a, uint64_t b, uint64_t * out) {
    if (!out || (a && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

inline bool checked_shortconv_layout(uint32_t width, uint32_t kernel,
                                     uint32_t n_tok, ShortconvLayout * out) {
    if (!out || !width || kernel < 2 || !n_tok) return false;
    uint64_t elements = 0;
    if (!checked_mul(n_tok, width, &elements)
        || !checked_mul(elements, 3, &elements)
        || !checked_mul(elements, sizeof(float), &out->bcx_bytes)) return false;
    if (!checked_mul(uint64_t(kernel - 1), width, &elements)
        || !checked_mul(elements, sizeof(float), &out->state_bytes)) return false;
    if (!checked_mul(n_tok, width, &elements)
        || !checked_mul(elements, sizeof(float), &out->output_bytes)) return false;
    if (!checked_mul(kernel, width, &elements)
        || !checked_mul(elements, sizeof(float), &out->weight_bytes)) return false;
    return true;
}

IMPARO_CUDA_HD inline float silu(float value) {
    return value / (1.0f + ::expf(-value));
}

template <typename State, typename Bcx>
IMPARO_CUDA_HD inline float signal_at(const State * state, const Bcx * bcx,
                                      uint64_t sequence_index, uint32_t channel,
                                      uint32_t width, uint32_t history) {
    if (sequence_index < history) {
        return state[sequence_index * width + channel];
    }
    const uint64_t token = sequence_index - history;
    const uint64_t row = token * uint64_t(3) * width;
    return bcx[row + channel] * bcx[row + uint64_t(2) * width + channel];
}

inline void shortconv_reference(const float * bcx, const float * weights,
                                const float * state, float * output,
                                uint32_t width, uint32_t kernel, uint32_t n_tok) {
    const uint32_t history = kernel - 1;
    for (uint32_t token = 0; token < n_tok; ++token) {
        for (uint32_t channel = 0; channel < width; ++channel) {
            float sum = 0.0f;
            for (uint32_t tap = 0; tap < kernel; ++tap) {
                sum += weights[uint64_t(channel) * kernel + tap]
                    * signal_at(state, bcx, uint64_t(token) + tap,
                                channel, width, history);
            }
            output[uint64_t(token) * width + channel] =
                bcx[uint64_t(token) * 3 * width + width + channel] * sum;
        }
    }
}

template <typename Bcx, typename State, typename Next>
IMPARO_CUDA_HD inline void shortconv_state_channel(
        const Bcx * bcx, const State * state, Next * next, uint32_t channel,
        uint32_t width, uint32_t kernel, uint32_t n_tok) {
    const uint32_t history = kernel - 1;
    // Ascending is load-bearing for state==next: every old-state source is a later
    // slot, so it is read before this channel-owner thread replaces that slot.
    for (uint32_t slot = 0; slot < history; ++slot) {
        next[uint64_t(slot) * width + channel] = signal_at(
            state, bcx, uint64_t(n_tok) + slot, channel, width, history);
    }
}

inline void shortconv_state_reference(const float * bcx, const float * state,
                                      float * next, uint32_t width,
                                      uint32_t kernel, uint32_t n_tok) {
    for (uint32_t channel = 0; channel < width; ++channel) {
        shortconv_state_channel(bcx, state, next, channel, width, kernel, n_tok);
    }
}

#if defined(__CUDACC__)
__global__ void shortconv_output_kernel(const float * bcx, const float * weights,
                                        const float * state, float * output,
                                        uint32_t width, uint32_t kernel,
                                        uint32_t n_tok) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = uint64_t(n_tok) * width;
    if (index >= count) return;
    const uint32_t token = uint32_t(index / width);
    const uint32_t channel = uint32_t(index % width);
    const uint32_t history = kernel - 1;
    float sum = 0.0f;
    for (uint32_t tap = 0; tap < kernel; ++tap) {
        sum += weights[uint64_t(channel) * kernel + tap]
            * signal_at(state, bcx, uint64_t(token) + tap,
                        channel, width, history);
    }
    output[index] = bcx[uint64_t(token) * 3 * width + width + channel] * sum;
}

__global__ void shortconv_state_kernel(const float * bcx, const float * state,
                                       float * next, uint32_t width,
                                       uint32_t kernel, uint32_t n_tok) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    // One thread owns a whole channel. In-place advance is therefore race-free:
    // ascending slots only read a later old-state slot, never one already replaced.
    // Snapshot uses this exact kernel with a distinct destination.
    shortconv_state_channel(bcx, state, next, channel, width, kernel, n_tok);
}

__global__ void silu_kernel(float * values, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] = silu(values[index]);
}

__global__ void silu_mul_kernel(float * values, const float * factors,
                                uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = values[index];
    values[index] = silu(value) * factors[index];
}
#endif

#undef IMPARO_CUDA_HD

} // namespace imparo_cuda_lfm2
