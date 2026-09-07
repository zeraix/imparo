#pragma once

// SM80+ quantized-KV expansion. One lane owns one output value, so a Q4/Q8
// block is expanded by one warp instead of one thread serializing 32 converts
// and stores. Cache bytes and half rounding remain identical to the generic
// kernel; this layer owns only the parallel execution geometry.
namespace imparo_sm80_kv {

template <uint32_t CacheType>
__global__ void dequant_parallel(
        const uint8_t * __restrict__ src, __half * __restrict__ dst,
        uint32_t width, uint32_t slots, uint32_t ring,
        const uint32_t * page_table) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t slot = blockIdx.y;
    if (i >= width || slot >= slots) return;

    const uint32_t physical =
        imparo_cuda_kv::physical_row(slot, ring, page_table);
    const uint32_t block = i / 32;
    const uint32_t lane = i & 31;
    const uint32_t blocks = width / 32;
    constexpr uint32_t block_bytes = CacheType == 2 ? 18 : 34;
    const uint8_t * packed = src
        + (uint64_t(physical) * blocks + block) * block_bytes;
    const float d = __half2float(*reinterpret_cast<const __half *>(packed));
    float value;
    if constexpr (CacheType == 2) {
        const uint8_t nibble = packed[2 + (lane & 15)];
        const int q = lane < 16 ? (nibble & 0x0f) : (nibble >> 4);
        value = fmaf(d, float(q), -8.0f * d);
    } else {
        value = float(int8_t(packed[2 + lane])) * d;
    }
    dst[uint64_t(physical) * width + i] = __float2half(value);
}

inline bool launch_dequant(
        const uint8_t * src, __half * dst, uint32_t width,
        uint32_t slots, uint32_t cache_type, uint32_t ring,
        const uint32_t * page_table, cudaStream_t stream) {
    const dim3 grid((width + 255) / 256, slots);
    if (cache_type == 2) {
        dequant_parallel<2><<<grid, 256, 0, stream>>>(src, dst, width, slots, ring, page_table);
        return true;
    }
    if (cache_type == 8) {
        dequant_parallel<8><<<grid, 256, 0, stream>>>(src, dst, width, slots, ring, page_table);
        return true;
    }
    return false;
}

} // namespace imparo_sm80_kv
