#pragma once

// Laboratory SM86 gate/up schedule derived from the V19 kernel-lab result.
// Two independent projection CTAs for the same output tile are adjacent in the
// flat grid.  This preserves the admitted raw Q4_0 and Q8_1 MMQ layouts while
// giving L2 a chance to retain the shared activation tile between gate and up.
// Selection is exact and opt-in; the established two-call route is the fallback.
namespace imparo_sm86_q4_q8_interleaved {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

__launch_bounds__(256, 2)
__global__ void q4_q8_1_gate_up_interleaved_r64(
        const uint8_t * __restrict__ gate,
        const uint8_t * __restrict__ up,
        const BlockQ8_1Mmq * __restrict__ x,
        float * __restrict__ gate_out,
        float * __restrict__ up_out,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ == 860
    using namespace imparo_sm80_mmq;
    const uint32_t projection = uint32_t(blockIdx.x) & 1u;
    const uint32_t tile_index = uint32_t(blockIdx.x) >> 1u;
    const uint8_t * weights = projection ? up : gate;
    float * dst = projection ? up_out : gate_out;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kCompactRows * kWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (tile_index / ntx) * kCompactRows;
    const uint32_t tile_token = (tile_index % ntx) * kTokens;
    float partial[32];

    compute_segment<kCompactRows, 1, false, true, true>(
        weights, x, sx, sy, partial, n_out, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks);

#pragma unroll
    for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2;
             ++token_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = (warp >> 1) * 16
                    + accumulator_row(lane, item);
                const uint32_t local_token = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8
                    + accumulator_token(lane, item);
                const uint32_t token = tile_token + local_token;
                if (token >= n_tok) continue;
                const uint32_t sum_index =
                    ((token_group * 2 + token_fragment) * 4) + item;
                dst[uint64_t(token) * dst_stride + tile_row + local_row] =
                    partial[sum_index];
            }
        }
    }
#else
    (void)gate; (void)up; (void)x; (void)gate_out; (void)up_out;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

inline LaunchResult launch(
        const uint8_t * gate, const uint8_t * up,
        const BlockQ8_1Mmq * x, float * gate_out, float * up_out,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t sm_version, uint32_t max_grid_x, cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(gate)
        | reinterpret_cast<uintptr_t>(up)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(gate_out)
        | reinterpret_cast<uintptr_t>(up_out);
    if (!gate || !up || !x || !gate_out || !up_out
        || (pointers & 15u) != 0 || sm_version != 86
        || !n_in || !n_out || n_tok <= 8 || n_tok > 512
        || n_in % 128 != 0 || n_out % 128 != 0 || n_out <= n_in
        || !max_grid_x) {
        return LaunchResult::NotSupported;
    }
    const uint64_t token_tiles = (n_tok + kTokens - 1) / kTokens;
    const uint64_t logical_tiles = uint64_t(n_out / kCompactRows)
        * token_tiles;
    const uint64_t flat_grid = 2u * logical_tiles;
    if (!logical_tiles || flat_grid > max_grid_x) {
        return LaunchResult::NotSupported;
    }
    static const bool configured = [] {
        const cudaError_t shared = cudaFuncSetAttribute(
            q4_q8_1_gate_up_interleaved_r64,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kCompactSharedBytes));
        if (shared != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const cudaError_t carveout = cudaFuncSetAttribute(
            q4_q8_1_gate_up_interleaved_r64,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (carveout != cudaSuccess) cudaGetLastError();
        return carveout == cudaSuccess;
    }();
    if (!configured) return LaunchResult::Error;

    q4_q8_1_gate_up_interleaved_r64<<<uint32_t(flat_grid),
        dim3(32, kWarps), kCompactSharedBytes, stream>>>(
            gate, up, x, gate_out, up_out, n_in, n_out, n_tok, n_tok,
            n_out);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_q4_q8_interleaved
