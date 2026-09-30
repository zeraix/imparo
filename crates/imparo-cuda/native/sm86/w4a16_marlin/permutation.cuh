#pragma once
// Coordinate form of vLLM commit 2cf0a6915 gptq_marlin_repack.cu:122-209.
// Q4/group32, uint4b8, no act-order, is_a_8bit=false. No values/scales change.
#include <cstdint>
#include <cstddef>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#if defined(__CUDACC__)
#define IMPARO_MARLIN_COORD __host__ __device__
#else
#define IMPARO_MARLIN_COORD
#endif
namespace imparo_w4a16_marlin {
struct Coordinate { std::uint32_t k, n; };
// A packed tile has 16 K values x 64 N columns = 128 uint32 words.
IMPARO_MARLIN_COORD constexpr Coordinate unpack_coordinate(
        std::uint64_t word, std::uint32_t slot, std::uint32_t n_out) {
    const std::uint64_t tile = word / 128;
    const std::uint32_t local = std::uint32_t(word % 128);
    const std::uint32_t lane = local / 4, warp = local % 4;
    const std::uint32_t v = slot / 4 + 2 * (slot % 4);
    const std::uint32_t offset = (v % 4) / 2 * 8 + (v % 2);
    return {std::uint32_t(tile / (n_out / 64)) * 16 + 2 * (lane % 4) + offset,
        std::uint32_t(tile % (n_out / 64)) * 64 + warp * 16 + lane / 4 + (v >= 4 ? 8u : 0u)};
}
IMPARO_MARLIN_COORD constexpr std::uint64_t packed_word(
        std::uint32_t k, std::uint32_t n, std::uint32_t n_out) {
    const std::uint32_t lane = (n % 8) * 4 + (k % 8) / 2;
    const std::uint32_t warp = (n % 64) / 16;
    return (std::uint64_t(k / 16) * (n_out / 64) + n / 64) * 128 + lane * 4 + warp;
}
IMPARO_MARLIN_COORD constexpr std::uint32_t packed_shift(
        std::uint32_t k, std::uint32_t n) {
    const std::uint32_t v = (k % 2) + 2 * ((k % 16) / 8) + 4 * ((n % 16) / 8);
    return 4 * (v / 2 + 4 * (v % 2));
}
// Mini uses [group,N] scales, transpose each 8x8 chunk; the permutation is self-inverse.
IMPARO_MARLIN_COORD constexpr std::uint64_t packed_scale(
        std::uint32_t k, std::uint32_t n, std::uint32_t n_out) {
    const std::uint64_t i = std::uint64_t(k / 32) * n_out + n;
    return (i / 64) * 64 + (i % 8) * 8 + (i % 64) / 8;
}
IMPARO_MARLIN_COORD constexpr std::uint64_t canonical_q4_byte(
        std::uint32_t k, std::uint32_t n, std::uint32_t k_in) {
    return (std::uint64_t(n) * (k_in / 32) + k / 32) * 18 + 2 + k % 16;
}
IMPARO_MARLIN_COORD constexpr std::uint32_t canonical_q4_shift(std::uint32_t k) {
    return k % 32 >= 16 ? 4u : 0u;
}

#if defined(__CUDACC__)
// Same total bytes in both representations. GU's original norm gap sits between
// the two N/2 matrices; the packed representation puts that gap after scales.
// The original half scale bits and unsigned Q4 nibbles are never dequantized.
static __global__ void pack_q4_with_gap(const std::uint8_t* input,
        std::uint8_t* output, std::uint32_t k_in, std::uint32_t n_out,
        std::uint32_t norm_gap) {
    const std::uint64_t t = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t elements = std::uint64_t(k_in) * n_out;
    if (t >= elements / 8) return;
    std::uint32_t word = 0;
#pragma unroll
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
        const Coordinate c = unpack_coordinate(t, slot, n_out);
        const std::uint64_t source = canonical_q4_byte(c.k, c.n, k_in)
            + (c.n >= n_out / 2 ? norm_gap : 0);
        const std::uint32_t nibble = (input[source] >> canonical_q4_shift(c.k)) & 15u;
        word |= nibble << (4 * slot);
    }
    reinterpret_cast<std::uint32_t*>(output)[t] = word;
    const std::uint64_t records = elements / 32;
    if (t < records) {
        const std::uint32_t n = std::uint32_t(t / (k_in / 32));
        const std::uint32_t group = std::uint32_t(t % (k_in / 32));
        const std::uint64_t source = t * 18 + (n >= n_out / 2 ? norm_gap : 0);
        const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(input + source);
        auto* scales = reinterpret_cast<std::uint16_t*>(output + elements / 2);
        scales[packed_scale(group * 32, n, n_out)] = bits;
    }
}
static __global__ void unpack_q4_with_gap(const std::uint8_t* input,
        std::uint8_t* output, std::uint32_t k_in, std::uint32_t n_out,
        std::uint32_t norm_gap) {
    const std::uint64_t t = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t elements = std::uint64_t(k_in) * n_out;
    if (t >= elements / 32) return;
    const std::uint32_t n = std::uint32_t(t / (k_in / 32));
    const std::uint32_t group = std::uint32_t(t % (k_in / 32));
    std::uint8_t* record = output + t * 18 + (n >= n_out / 2 ? norm_gap : 0);
    const auto* scales = reinterpret_cast<const std::uint16_t*>(input + elements / 2);
    *reinterpret_cast<std::uint16_t*>(record) = scales[packed_scale(group * 32, n, n_out)];
    const auto* words = reinterpret_cast<const std::uint32_t*>(input);
#pragma unroll
    for (std::uint32_t i = 0; i < 16; ++i) {
        const std::uint32_t low_k = group * 32 + i, high_k = low_k + 16;
        const std::uint32_t low = (words[packed_word(low_k, n, n_out)] >> packed_shift(low_k, n)) & 15u;
        const std::uint32_t high = (words[packed_word(high_k, n, n_out)] >> packed_shift(high_k, n)) & 15u;
        record[2 + i] = std::uint8_t(low | (high << 4));
    }
}

// Existing phase owner supplies one separate scratch allocation. This never
// allocates/synchronizes or changes phase. Input/output ranges must not overlap.
// After success, caller may copy the whole range back in the SAME stream.
// GU: K2560,N20480,gap10240 -> 29501440 bytes. Down: K10240,N2560,gap0
// ->14745600 bytes. Do not publish new phase until all conversion work succeeds.
inline cudaError_t convert(const void* input, void* output, int k, int n,
        std::size_t norm_gap, bool pack, cudaStream_t stream) {
    if (!input || !output
        || !((k == 2560 && n == 20480 && norm_gap == 10240)
          || (k == 10240 && n == 2560 && norm_gap == 0)))
        return cudaErrorInvalidValue;
    const std::uintptr_t source = reinterpret_cast<std::uintptr_t>(input);
    const std::uintptr_t destination = reinterpret_cast<std::uintptr_t>(output);
    const std::size_t elements = std::size_t(k) * n;
    const std::size_t weight_bytes = elements / 32 * 18;
    const std::size_t total_bytes = weight_bytes + norm_gap;
    if (((source | destination) & 15u)
        || (source >= destination ? source - destination : destination - source) < total_bytes)
        return cudaErrorInvalidValue;
    const auto* in = static_cast<const std::uint8_t*>(input);
    auto* out = static_cast<std::uint8_t*>(output);
    if (norm_gap) {
        const std::size_t source_gap = pack ? weight_bytes / 2 : weight_bytes;
        const std::size_t destination_gap = pack ? weight_bytes : weight_bytes / 2;
        const cudaError_t rc = cudaMemcpyAsync(out + destination_gap, in + source_gap,
            norm_gap, cudaMemcpyDeviceToDevice, stream);
        if (rc != cudaSuccess) return rc;
    }
    constexpr unsigned threads = 256;
    if (pack) {
        const unsigned blocks = static_cast<unsigned>((elements / 8 + threads - 1) / threads);
        pack_q4_with_gap<<<blocks, threads, 0, stream>>>(in, out, k, n, static_cast<unsigned>(norm_gap));
    } else {
        const unsigned blocks = static_cast<unsigned>((elements / 32 + threads - 1) / threads);
        unpack_q4_with_gap<<<blocks, threads, 0, stream>>>(in, out, k, n, static_cast<unsigned>(norm_gap));
    }
    return cudaPeekAtLastError();
}
#endif // __CUDACC__
} // namespace imparo_w4a16_marlin
#undef IMPARO_MARLIN_COORD
