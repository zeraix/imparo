#pragma once
// Bit-exact tiled implementation of permutation.cuh::unpack_q4_with_gap.
// Include permutation.cuh first to share the canonical layout coordinates.
// Fixed scope: GU K2560/N20480/gap10240 and Down K10240/N2560/gap0.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace imparo_w4a16_marlin {
namespace tile_unpack_detail {
constexpr unsigned TileK = 256;
constexpr unsigned TileN = 64;
constexpr unsigned Threads = 256;
constexpr unsigned Groups = TileK / 32;
constexpr unsigned PayloadWords = TileK * TileN / 8;
constexpr unsigned ScaleHalves = Groups * TileN;
constexpr unsigned RowBytes = Groups * 18;
constexpr unsigned CanonicalBytes = TileN * RowBytes;
constexpr unsigned SharedBytes = PayloadWords * 4 + ScaleHalves * 2 + CanonicalBytes;
static_assert(RowBytes == 144 && SharedBytes == 18432);

__global__ __launch_bounds__(Threads, 1) void unpack_tile(
        const std::uint8_t* __restrict__ input,
        std::uint8_t* __restrict__ output,
        std::uint32_t k_in, std::uint32_t n_out, std::uint32_t norm_gap) {
    // Each source fragment is a contiguous 16-K x 64-N Marlin tile.
    __shared__ __align__(16) std::uint32_t payload[PayloadWords];
    __shared__ __align__(16) std::uint16_t scales[ScaleHalves];
    __shared__ __align__(16) std::uint16_t canonical[CanonicalBytes / 2];
    const unsigned tid = threadIdx.x;
    const unsigned k_base = blockIdx.x * TileK;
    const unsigned n_base = blockIdx.y * TileN;
    const auto* source_words = reinterpret_cast<const std::uint32_t*>(input);
    const auto* source_scales = reinterpret_cast<const std::uint16_t*>(
        input + std::uint64_t(k_in) * n_out / 2);

    for (unsigned v = tid; v < PayloadWords / 4; v += Threads) {
        const unsigned k16 = v / 32;
        const unsigned tile_vec = v % 32;
        const std::uint64_t source_word =
            (std::uint64_t(k_base / 16 + k16) * (n_out / 64) + n_base / 64) * 128
            + tile_vec * 4;
        reinterpret_cast<uint4*>(payload)[v] =
            *reinterpret_cast<const uint4*>(source_words + source_word);
    }
    for (unsigned v = tid; v < ScaleHalves / 8; v += Threads) {
        const unsigned group = v / 8;
        const unsigned scale_vec = v % 8;
        const std::uint64_t source_half =
            std::uint64_t(k_base / 32 + group) * n_out + n_base + scale_vec * 8;
        reinterpret_cast<uint4*>(scales)[v] =
            *reinterpret_cast<const uint4*>(source_scales + source_half);
    }
    __syncthreads();

    // Two records per thread. Global gathers become reads from the shared tile.
    // This changes only addressing, not unsigned nibbles or half-scale bits.
    for (unsigned record = tid; record < TileN * Groups; record += Threads) {
        const unsigned row = record / Groups;
        const unsigned group = record % Groups;
        auto* dst = canonical + record * 9;
        dst[0] = scales[packed_scale(group * 32, row, TileN)];
#pragma unroll
        for (unsigned pair = 0; pair < 8; ++pair) {
            const unsigned k0 = group * 32 + pair * 2;
            const unsigned k1 = k0 + 1;
            const unsigned lo0 = (payload[packed_word(k0, row, TileN)]
                >> packed_shift(k0, row)) & 15u;
            const unsigned hi0 = (payload[packed_word(k0 + 16, row, TileN)]
                >> packed_shift(k0 + 16, row)) & 15u;
            const unsigned lo1 = (payload[packed_word(k1, row, TileN)]
                >> packed_shift(k1, row)) & 15u;
            const unsigned hi1 = (payload[packed_word(k1 + 16, row, TileN)]
                >> packed_shift(k1 + 16, row)) & 15u;
            dst[1 + pair] = std::uint16_t(lo0 | (hi0 << 4) | (lo1 << 8) | (hi1 << 12));
        }
    }
    __syncthreads();

    // Eight adjacent canonical Q4 records = 144 bytes = nine aligned uint4s.
    // Both admitted K row strides and the GU norm gap are multiples of 16.
    for (unsigned v = tid; v < CanonicalBytes / 16; v += Threads) {
        const unsigned row = v / (RowBytes / 16);
        const unsigned row_vec = v % (RowBytes / 16);
        const unsigned n = n_base + row;
        const std::uint64_t destination =
            (std::uint64_t(n) * (k_in / 32) + k_base / 32) * 18
            + (n >= n_out / 2 ? norm_gap : 0) + row_vec * 16;
        *reinterpret_cast<uint4*>(output + destination) =
            reinterpret_cast<const uint4*>(canonical)[v];
    }
}
} // namespace tile_unpack_detail

// No allocation, synchronization, copy-back or owner-state mutation.
// The original production caller retains those operations and error handling.
inline cudaError_t unpack_tiled(const void* input, void* output,
        int k, int n, std::size_t norm_gap, cudaStream_t stream) {
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
        const auto rc = cudaMemcpyAsync(out + weight_bytes / 2, in + weight_bytes,
            norm_gap, cudaMemcpyDeviceToDevice, stream);
        if (rc != cudaSuccess) return rc;
    }
    tile_unpack_detail::unpack_tile<<<
        dim3(unsigned(k) / tile_unpack_detail::TileK,
             unsigned(n) / tile_unpack_detail::TileN),
        tile_unpack_detail::Threads, 0, stream>>>(in, out, unsigned(k), unsigned(n), unsigned(norm_gap));
    return cudaPeekAtLastError();
}
} // namespace imparo_w4a16_marlin
