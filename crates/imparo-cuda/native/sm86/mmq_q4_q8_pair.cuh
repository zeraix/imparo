#pragma once

// Experimental SM86 horizontal fusion for the canonical 512-token GEGLU pair.
// One CTA computes gate and up for the same output tile, keeps the gate result in
// registers, writes only the final activation and publishes its Q8_1 layout for
// the down projection. Selection stays outside the kernel and the native two-call
// route remains the fallback.
namespace imparo_sm86_q4_q8_pair {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

__launch_bounds__(256, 1)
__global__ void q4_q8_1_pair_full_tile(
        const uint8_t * __restrict__ first,
        const uint8_t * __restrict__ second,
        const BlockQ8_1Mmq * __restrict__ x,
        float * __restrict__ dst,
        BlockQ8_1Mmq * __restrict__ epilogue_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ == 860
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (uint32_t(blockIdx.x) / ntx) * kRows;
    const uint32_t tile_token = (uint32_t(blockIdx.x) % ntx) * kTokens;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    float gate[64];
    float up[64];

    compute_segment<kRows, 2, false, true, true, 4>(
        first, x, sx, sy, gate, n_out, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks);
    __syncthreads();
    compute_segment<kRows, 2, false, true, true, 4>(
        second, x, sx, sy, up, n_out, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks);

#pragma unroll
    for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2;
             ++token_fragment) {
#pragma unroll
            for (uint32_t token_item = 0; token_item < 2; ++token_item) {
                float value[4];
#pragma unroll
                for (uint32_t row_fragment = 0; row_fragment < 2;
                     ++row_fragment) {
#pragma unroll
                    for (uint32_t row_half = 0; row_half < 2; ++row_half) {
                        const uint32_t item = 2 * row_half + token_item;
                        const uint32_t value_index =
                            2 * row_fragment + row_half;
                        const uint32_t local_row = (warp >> 1) * 32
                            + row_fragment * 16 + accumulator_row(lane, item);
                        const uint32_t local_token = token_group * 32
                            + (warp & 1) * 16 + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t token = tile_token + local_token;
                        const uint32_t sum_index =
                            (((token_group * 2 + token_fragment) * 2
                                + row_fragment) * 4) + item;
                        float result = 0.0f;
                        if (token < n_tok) {
                            result = cuda_gelu(gate[sum_index]) * up[sum_index];
                            dst[uint64_t(token) * dst_stride
                                + tile_row + local_row] = result;
                        }
                        value[value_index] = result;
                    }
                }

                float amax = fabsf(value[0]);
                amax = fmaxf(amax, fabsf(value[1]));
                amax = fmaxf(amax, fabsf(value[2]));
                amax = fmaxf(amax, fabsf(value[3]));
#pragma unroll
                for (int offset = 16; offset >= 4; offset >>= 1) {
                    amax = fmaxf(amax,
                        __shfl_xor_sync(0xffffffff, amax, offset, 32));
                }
                const float d_inv = 127.0f / amax;
                const float d = 1.0f / d_inv;
                const uint32_t token = tile_token + token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8
                    + (lane & 3) * 2 + token_item;
                if (token < n_tok) {
                    BlockQ8_1Mmq * out = epilogue_q8
                        + uint64_t(tile_row / kRows) * n_tok + token;
                    const uint32_t block_in_group = warp >> 1;
                    const uint32_t row_in_block = lane >> 2;
                    out->qs[block_in_group * 32 + row_in_block] =
                        int8_t(roundf(value[0] * d_inv));
                    out->qs[block_in_group * 32 + 8 + row_in_block] =
                        int8_t(roundf(value[1] * d_inv));
                    out->qs[block_in_group * 32 + 16 + row_in_block] =
                        int8_t(roundf(value[2] * d_inv));
                    out->qs[block_in_group * 32 + 24 + row_in_block] =
                        int8_t(roundf(value[3] * d_inv));
                    if (lane < 4) {
                        out->d[block_in_group] = __half2float(__float2half(d));
                    }
                }
            }
        }
    }
#else
    (void)first; (void)second; (void)x;
    (void)dst; (void)epilogue_q8;
    (void)n_in; (void)n_out; (void)n_tok;
    (void)work_n_tok; (void)dst_stride;
#endif
}

inline LaunchResult launch(
        const uint8_t * first, const uint8_t * second,
        const BlockQ8_1Mmq * x, float * dst, BlockQ8_1Mmq * epilogue_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t sm_version, uint32_t max_grid_x, cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(first)
        | reinterpret_cast<uintptr_t>(second)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(dst)
        | reinterpret_cast<uintptr_t>(epilogue_q8);
    if (!first || !second || !x || !dst || !epilogue_q8
        || (pointers & 15u) != 0 || sm_version != 86
        || !n_in || !n_out || n_tok <= 8 || n_tok > 512
        || n_in % 128 != 0 || n_out % kRows != 0 || n_out <= n_in
        || !max_grid_x) {
        return LaunchResult::NotSupported;
    }
    const uint64_t token_tiles = (n_tok + kTokens - 1) / kTokens;
    const uint64_t logical_tiles = uint64_t(n_out / kRows) * token_tiles;
    const uint64_t flat_grid = logical_tiles;
    if (!logical_tiles || flat_grid > max_grid_x) {
        return LaunchResult::NotSupported;
    }
    static const bool configured = [] {
        const cudaError_t shared = cudaFuncSetAttribute(
            q4_q8_1_pair_full_tile,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        if (shared != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const cudaError_t carveout = cudaFuncSetAttribute(
            q4_q8_1_pair_full_tile,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (carveout != cudaSuccess) cudaGetLastError();
        return carveout == cudaSuccess;
    }();
    if (!configured) return LaunchResult::Error;

    q4_q8_1_pair_full_tile<<<uint32_t(flat_grid), dim3(32, kWarps),
        kHalfKSharedBytes, stream>>>(first, second, x, dst, epilogue_q8,
            n_in, n_out, n_tok, n_tok, n_out);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_q4_q8_pair
