#pragma once

// Single-token LFM2 short-convolution output/state fusion for SM86. The
// output expression and the in-place state update deliberately retain the
// established per-channel operation order from lfm2_ops.cuh.
namespace imparo_sm86_shortconv_decode_fused {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

__global__ void output_state_single(
        const float * __restrict__ bcx,
        const float * __restrict__ weights,
        float * state,
        float * __restrict__ output,
        uint32_t width, uint32_t kernel) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= width) return;

    const uint32_t history = kernel - 1;
    float sum = 0.0f;
    for (uint32_t tap = 0; tap < kernel; ++tap) {
        sum += weights[uint64_t(channel) * kernel + tap]
            * imparo_cuda_lfm2::signal_at(
                state, bcx, tap, channel, width, history);
    }
    output[channel] = bcx[uint64_t(width) + channel] * sum;

    imparo_cuda_lfm2::shortconv_state_channel(
        bcx, state, state, channel, width, kernel, 1);
}

inline LaunchResult launch(
        const float * bcx, const float * weights, float * state, float * output,
        uint32_t width, uint32_t kernel, uint32_t sm_version,
        cudaStream_t stream) {
    if (sm_version != 86 || !bcx || !weights || !state || !output
            || !width || kernel < 2) {
        return LaunchResult::NotSupported;
    }
    const uint32_t grid = uint32_t((uint64_t(width) + 255) / 256);
    output_state_single<<<grid, 256, 0, stream>>>(
        bcx, weights, state, output, width, kernel);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_shortconv_decode_fused
