#pragma once

// SM86 kernel-lab route for packed Q4_0 x Q8_1. Unlike the tensor-core MMQ,
// this tile never expands Q4 nibbles to int8 shared memory. It keeps the packed
// 4-bit weights, computes the unsigned-nibble dot with DP4A, and subtracts the
// exact 8 * sum(q8) zero-point term before applying the per-32-value scales.
//
// The first admission target is deliberately narrow: the 449-token
// 2560 -> 10240 gate/up pair on the aligned virtual schedule. Other shapes,
// architectures and numerical routes remain on the established MMQ.
namespace imparo_sm86_packed_dp4a {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kRows = 128;
constexpr uint32_t kTokens = 128;
constexpr uint32_t kBlocksPerStage = 4;
constexpr uint32_t kWarps = 8;

struct __align__(16) SharedTile {
    // A warp reads one packed word from 32 different weight rows. The +1
    // stride skew maps those lanes across all 32 banks instead of aliasing the
    // 64-byte unpadded row stride onto two banks.
    uint32_t w_qs[kRows][kBlocksPerStage * 4 + 1];
    // Scale loads have the same row-wise access pattern; keep an independent
    // one-float skew instead of reintroducing an eight-way conflict.
    float w_d[kRows][kBlocksPerStage + 1];
    int32_t x_qs[kTokens][kBlocksPerStage][8];
    float x_d[kTokens][kBlocksPerStage];
    int32_t x_sum[kTokens][kBlocksPerStage];
};

static_assert(sizeof(SharedTile) == 31744,
              "packed DP4A shared-memory contract");

template <bool DirectEpilogue>
__launch_bounds__(256, 1)
__global__ void q4_q8_1_packed_dp4a(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ == 860
    __shared__ SharedTile tile;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t tile_row = blockIdx.x * kRows;
    const uint32_t tile_token = blockIdx.y * kTokens;
    const uint32_t blocks = n_in / 32;

    // Match the proven DP4A register map: token groups are the major
    // compile-time dimension and four 32-row stripes are the minor one.
    // Keeping this flat is important; a doubly indexed local array makes
    // ptxas materialize the accumulator set in the thread stack.
    float sums[16 * 4] = {};
    for (uint32_t group = 0; group < blocks / kBlocksPerStage; ++group) {
        constexpr uint32_t stage_blocks = kRows * kBlocksPerStage;
        for (uint32_t linear = tid; linear < stage_blocks;
             linear += kWarps * 32) {
            const uint32_t local_row = linear / kBlocksPerStage;
            const uint32_t local_block = linear % kBlocksPerStage;
            const uint32_t row = tile_row + local_row;
            const uint32_t kb = group * kBlocksPerStage + local_block;
            uint32_t packed[4] = {};
            float scale = 0.0f;
            if (row < n_out) {
                const uint8_t * block =
                    w + (uint64_t(row) * blocks + kb) * 18;
                const uint16_t * qs =
                    reinterpret_cast<const uint16_t *>(block + 2);
#pragma unroll
                for (uint32_t word = 0; word < 4; ++word) {
                    packed[word] = uint32_t(qs[2 * word])
                        | (uint32_t(qs[2 * word + 1]) << 16);
                }
                scale = __half2float(
                    *reinterpret_cast<const __half *>(block));
            }
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
                tile.w_qs[local_row][local_block * 4 + word] = packed[word];
            }
            tile.w_d[local_row][local_block] = scale;
        }

        constexpr uint32_t activation_blocks = kTokens * kBlocksPerStage;
        for (uint32_t linear = tid; linear < activation_blocks;
             linear += kWarps * 32) {
            const uint32_t local_token = linear / kBlocksPerStage;
            const uint32_t local_block = linear % kBlocksPerStage;
            const uint32_t token = tile_token + local_token;
            int32_t packed[8] = {};
            float scale = 0.0f;
            int32_t sum = 0;
            if (token < n_tok) {
                const BlockQ8_1Mmq * record =
                    x + uint64_t(group) * n_tok + token;
                const int32_t * qs = reinterpret_cast<const int32_t *>(
                    record->qs + local_block * 32);
#pragma unroll
                for (uint32_t word = 0; word < 8; ++word) {
                    packed[word] = qs[word];
                    sum = __dp4a(qs[word], 0x01010101, sum);
                }
                scale = record->d[local_block];
            }
#pragma unroll
            for (uint32_t word = 0; word < 8; ++word) {
                tile.x_qs[local_token][local_block][word] = packed[word];
            }
            tile.x_d[local_token][local_block] = scale;
            tile.x_sum[local_token][local_block] = sum;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t token_item = 0; token_item < 16; ++token_item) {
                const uint32_t local_token = token_item * kWarps + warp;
#pragma unroll
            for (uint32_t row_item = 0; row_item < 4; ++row_item) {
                const uint32_t local_row = row_item * 32 + lane;
#pragma unroll
                for (uint32_t local_block = 0;
                     local_block < kBlocksPerStage; ++local_block) {
                    int32_t dot = 0;
#pragma unroll
                    for (uint32_t word = 0; word < 4; ++word) {
                        const uint32_t q4 =
                            tile.w_qs[local_row][local_block * 4 + word];
                        dot = __dp4a(int32_t(q4 & 0x0f0f0f0fu),
                            tile.x_qs[local_token][local_block][word], dot);
                        dot = __dp4a(int32_t((q4 >> 4) & 0x0f0f0f0fu),
                            tile.x_qs[local_token][local_block][word + 4], dot);
                    }
                    dot -= 8 * tile.x_sum[local_token][local_block];
                    sums[token_item * 4 + row_item] += float(dot)
                        * tile.w_d[local_row][local_block]
                        * tile.x_d[local_token][local_block];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t token_item = 0; token_item < 16; ++token_item) {
            const uint32_t token = tile_token + token_item * kWarps + warp;
#pragma unroll
        for (uint32_t row_item = 0; row_item < 4; ++row_item) {
            const uint32_t row = tile_row + row_item * 32 + lane;
            if (row >= n_out || token >= n_tok) continue;
            float * slot = dst + uint64_t(token) * dst_stride + row;
            if constexpr (DirectEpilogue) {
                *slot = cuda_gelu(*slot)
                    * sums[token_item * 4 + row_item];
            } else {
                *slot = sums[token_item * 4 + row_item];
            }
        }
    }
#else
    (void)w; (void)x; (void)dst; (void)n_in; (void)n_out;
    (void)n_tok; (void)dst_stride;
#endif
}

inline LaunchResult launch(
        const uint8_t * w, const BlockQ8_1Mmq * x, float * dst,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t epilogue, uint32_t out_stride, uint32_t row_base,
        uint32_t sm_version, uint32_t stream_k_numeric,
        uint32_t virtual_token_base, uint32_t virtual_schedule,
        cudaStream_t stream, imparo_sm80_mmq::LaunchInfo * info) {
    if (!w || !x || !dst || sm_version != 86
        || n_in != 2560 || n_out != 10240 || n_tok != 449
        || epilogue > 1 || out_stride != n_out || row_base != 0
        || !stream_k_numeric || !virtual_schedule
        || virtual_token_base % 512 != 0) {
        return LaunchResult::NotSupported;
    }
    const dim3 grid(n_out / kRows, (n_tok + kTokens - 1) / kTokens);
    if (epilogue) {
        q4_q8_1_packed_dp4a<true>
            <<<grid, dim3(32, kWarps), 0, stream>>>(
                w, x, dst, n_in, n_out, n_tok, out_stride);
    } else {
        q4_q8_1_packed_dp4a<false>
            <<<grid, dim3(32, kWarps), 0, stream>>>(
                w, x, dst, n_in, n_out, n_tok, out_stride);
    }
    if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;
    if (info) {
        info->route = imparo_sm80_mmq::LaunchRoute::PackedDp4aLab;
        info->tile_rows = kRows;
        info->tile_tokens = kTokens;
        info->logical_tiles = grid.x * grid.y;
        info->physical_blocks = info->logical_tiles;
        info->efficiency = 100;
        info->fused_q8 = false;
    }
    return LaunchResult::Launched;
}

} // namespace imparo_sm86_packed_dp4a
