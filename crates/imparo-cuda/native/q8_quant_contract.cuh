#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// A numerical contract shared by dense and fused Q8 activation producers.
// Callers provide the maximum over the same 32-value quantization block.
namespace imparo_q8_quant {
__device__ __forceinline__ float m1_block(float4 xi, float amax, char4 & quant) {
    const float d = amax / 127.0f;
    quant.x = amax == 0.0f ? int8_t(0) : int8_t(roundf(xi.x / d));
    quant.y = amax == 0.0f ? int8_t(0) : int8_t(roundf(xi.y / d));
    quant.z = amax == 0.0f ? int8_t(0) : int8_t(roundf(xi.z / d));
    quant.w = amax == 0.0f ? int8_t(0) : int8_t(roundf(xi.w / d));
    return __half2float(__float2half(d));
}
} // namespace imparo_q8_quant
