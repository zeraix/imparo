#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../sm80/mma_f16.cuh"

namespace {

#define CUDA_PROBE_OK(call)                                                        \
    do {                                                                           \
        const cudaError_t error_ = (call);                                         \
        if (error_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s failed: %s\n", #call,                       \
                         cudaGetErrorString(error_));                              \
            return 1;                                                              \
        }                                                                          \
    } while (0)

__global__ void probe_fragment_positions(const float * q, const __half * k,
                                         uint32_t width, float * out) {
    const uint32_t q_row = blockIdx.x & 15;
    const uint32_t k_row = blockIdx.x >> 4;
    const uint32_t lane = threadIdx.x;
    __shared__ __align__(16) __half2 q_tile[16 * 8];
    __shared__ __align__(16) __half2 k_tile[16 * 8];
    const float value = imparo_sm80_mma::dot_f16_fragment(
        q, k, width, 1.0f, q_row, k_row, q_tile, k_tile, lane);
    if (lane == 0) out[blockIdx.x] = value;
}

uint32_t bits(float value) {
    uint32_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    const uint32_t width = argc > 1 ? uint32_t(std::strtoul(argv[1], nullptr, 10)) : 256;
    if (!width) {
        std::fprintf(stderr, "width must be positive\n");
        return 2;
    }

    cudaDeviceProp properties{};
    CUDA_PROBE_OK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major < 8) {
        std::fprintf(stderr, "mma_f16_probe requires SM80+, found SM%d%d\n",
                     properties.major, properties.minor);
        return 2;
    }

    std::vector<float> q(width);
    std::vector<__half> k(width);
    double cpu = 0.0;
    for (uint32_t i = 0; i < width; ++i) {
        // Deterministic, non-symmetric values exercise cancellation and every FP16
        // exponent range relevant to normalized attention operands.
        q[i] = std::sin(float(i + 1) * 0.173f) * (0.25f + float(i % 11) * 0.03125f);
        const float kv = std::cos(float(i + 3) * 0.117f)
            * (0.375f + float(i % 7) * 0.046875f);
        k[i] = __float2half(kv);
        cpu += double(__half2float(__float2half(q[i]))) * double(__half2float(k[i]));
    }

    float * q_device = nullptr;
    __half * k_device = nullptr;
    float * out_device = nullptr;
    CUDA_PROBE_OK(cudaMalloc(&q_device, uint64_t(width) * sizeof(float)));
    CUDA_PROBE_OK(cudaMalloc(&k_device, uint64_t(width) * sizeof(__half)));
    CUDA_PROBE_OK(cudaMalloc(&out_device, 16 * 16 * sizeof(float)));
    CUDA_PROBE_OK(cudaMemcpy(q_device, q.data(), uint64_t(width) * sizeof(float),
                             cudaMemcpyHostToDevice));
    CUDA_PROBE_OK(cudaMemcpy(k_device, k.data(), uint64_t(width) * sizeof(__half),
                             cudaMemcpyHostToDevice));

    probe_fragment_positions<<<16 * 16, 32>>>(q_device, k_device, width, out_device);
    CUDA_PROBE_OK(cudaGetLastError());
    std::vector<float> out(16 * 16);
    CUDA_PROBE_OK(cudaMemcpy(out.data(), out_device, out.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));

    CUDA_PROBE_OK(cudaFree(out_device));
    CUDA_PROBE_OK(cudaFree(k_device));
    CUDA_PROBE_OK(cudaFree(q_device));

    const uint32_t expected = bits(out[0]);
    for (uint32_t i = 1; i < out.size(); ++i) {
        if (bits(out[i]) != expected) {
            std::fprintf(stderr,
                "fragment mismatch q_row=%u k_row=%u expected=%08x actual=%08x\n",
                i & 15, i >> 4, expected, bits(out[i]));
            return 1;
        }
    }
    const double tolerance = std::max(1e-5, std::abs(cpu) * 5e-4);
    if (std::abs(double(out[0]) - cpu) > tolerance) {
        std::fprintf(stderr, "dot mismatch gpu=%.9g cpu=%.9g tolerance=%.9g\n",
                     out[0], cpu, tolerance);
        return 1;
    }

    std::printf("PASS device=%s sm=%d%d width=%u bits=%08x value=%.9g\n",
                properties.name, properties.major, properties.minor,
                width, expected, out[0]);
    return 0;
}
