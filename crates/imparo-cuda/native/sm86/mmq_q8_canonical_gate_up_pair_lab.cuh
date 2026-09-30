#pragma once
#include "../lfm_retained_policy.cuh"

// Default-off canonical Q8 Gate/Up experiment; not a receipted runtime route.
// Reuse the existing warp-pair MMA schedule, but read original GGUF blocks and
// write dense SiLU(Gate)*Up directly. No repack or intermediate Gate/Up tensor.
namespace canonical_q8_pair_lab {
namespace Pair = imparo_sm86_q8_tm_gate_up_row_pair_lab;
using namespace Pair;

__device__ __forceinline__ void stage_weights(
        const uint8_t *gate, const uint8_t *up, int8_t *gs, int8_t *us,
        uint32_t blocks, uint32_t row0, uint32_t kb0, uint32_t tid) {
    constexpr uint32_t records = kRows * kStageBlocks;
    for (uint32_t index = tid; index < 2 * records; index += kWarps * 32) {
        const uint32_t projection = index / records;
        const uint32_t within = index % records;
        const uint32_t row = within / kStageBlocks;
        const uint32_t kb = within % kStageBlocks;
        const uint8_t *block = (projection ? up : gate)
            + (uint64_t(row0 + row) * blocks + kb0 + kb) * 34;
        int8_t *stage = (projection ? us : gs) + row * kWeightStride;
#pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            reinterpret_cast<int *>(stage + kb * 32)[word] =
                imparo_sm80_q8_mmq::load_q8_word(block + 2, word);
        }
        reinterpret_cast<float *>(stage + imparo_sm80_q8_mmq::kStageValues)[kb]
            = __half2float(*reinterpret_cast<const __half *>(block));
    }
}

template<uint32_t LiveFragments>
__device__ __forceinline__ void fused_body(const uint8_t *gate, const uint8_t *up,
        const BlockQ8_1Mmq *x, float *y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    extern __shared__ __align__(16) int8_t shared[];
    int8_t *gs = shared;
    int8_t *us = gs + kProjectionWeightBytes;
    int8_t *xs = us + kProjectionWeightBytes;
    const uint32_t lane = threadIdx.x, warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t pair = warp / 2, projection = warp % 2;
    const uint32_t row0 = blockIdx.x * kRows;
    const uint32_t tok0 = blockIdx.y * kTokens;
    const uint32_t active = min(kTokens, n_tok - tok0);
    float partial[64] = {};
    for (uint32_t kb = 0; kb < n_in / 32; kb += kStageBlocks) {
#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase) {
            imparo_sm80_q8_mmq::stage_activation_group<kTokens>(x,
                xs + phase * kTokens * kActivationStride,
                n_tok, tok0, active, (kb + phase * 4) / 4, true, tid);
        }
        imparo_sm80_mmq::commit_async_copies();
        stage_weights(gate, up, gs, us, n_in / 32, row0, kb, tid);
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();
        Pair::accumulate_projection<LiveFragments>(projection ? us : gs, xs, partial, lane, pair);
        __syncthreads();
    }
    float *gate_tile = reinterpret_cast<float *>(shared);
    if (projection == 0) {
#pragma unroll
        for (uint32_t fragment = 0; fragment < LiveFragments; ++fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t row = pair * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t tok = fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                if (tok < active) gate_tile[tok * kRows + row]
                    = partial[fragment * 4 + item];
            }
        }
    }
    __syncthreads();
    if (projection == 1) {
#pragma unroll
        for (uint32_t fragment = 0; fragment < LiveFragments; ++fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t row = pair * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t tok = fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                if (tok < active) y[uint64_t(tok0 + tok) * n_out + row0 + row]
                    = imparo_cuda_lfm2::silu(gate_tile[tok * kRows + row])
                        * partial[fragment * 4 + item];
            }
        }
    }
#endif
}

__launch_bounds__(256, 1)
__global__ void fused(const uint8_t *gate, const uint8_t *up,
        const BlockQ8_1Mmq *x, float *y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok) {
    fused_body<16>(gate, up, x, y, n_in, n_out, n_tok);
}

#if defined(IMPARO_CUDA_SPECULATIVE)
template<uint32_t LiveFragments>
__launch_bounds__(256, 1)
__global__ void fused_tail(const uint8_t *gate, const uint8_t *up,
        const BlockQ8_1Mmq *x, float *y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok) {
    static_assert(LiveFragments == 2, "Only the fixed M9 live2 candidate is admitted");
    fused_body<LiveFragments>(gate, up, x, y, n_in, n_out, n_tok);
}
#endif

inline Pair::LaunchResult launch(const uint8_t *gate, const uint8_t *up,
        const BlockQ8_1Mmq *x, float *y, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, cudaStream_t stream, bool batch_invariant = false) {
#if defined(IMPARO_CUDA_SPECULATIVE)
    const char *live2 = std::getenv("IMPARO_LAB_Q8_CANONICAL_LIVE2");
    if ((n_tok == 9 || (batch_invariant && n_tok >= 1 && n_tok <= 16))
            && imparo_lfm_retained::common(live2 && std::strcmp(live2, "1") == 0)) {
        int live2_device = -1;
        if (cudaGetDevice(&live2_device) != cudaSuccess) return Pair::LaunchResult::Error;
        static int live2_configured_device = -1;
        static bool live2_configured = false;
        if (live2_configured_device != live2_device) {
            const auto result = cudaFuncSetAttribute(fused_tail<2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
            live2_configured_device = live2_device;
            live2_configured = result == cudaSuccess;
            if (!live2_configured) {
                if (result != cudaErrorInvalidValue && result != cudaErrorNotSupported)
                    return Pair::LaunchResult::Error;
                (void)cudaGetLastError();
            }
        }
        if (live2_configured) {
            fused_tail<2><<<dim3(n_out/kRows,(n_tok+kTokens-1)/kTokens),
                dim3(32,kWarps),kSharedBytes,stream>>>(gate,up,x,y,n_in,n_out,n_tok);
            if (cudaPeekAtLastError() != cudaSuccess) return Pair::LaunchResult::Error;
            static bool traced = false;
            const char *trace = std::getenv("IMPARO_LAB_Q8_CANONICAL_LIVE2_TRACE");
            if (!traced && trace && std::strcmp(trace, "1") == 0) {
                std::fprintf(stderr, "CANONICAL_M9_LIVE2 n_tok=%u fragments=2 n_in=%u n_out=%u\n",
                    n_tok, n_in, n_out);
                traced = true;
            }
            return Pair::LaunchResult::Launched;
        }
        if (batch_invariant) return Pair::LaunchResult::NotSupported;
    }
#endif
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return Pair::LaunchResult::Error;
    static int configured_device = -1;
    if (configured_device != device) {
        auto result = cudaFuncSetAttribute(fused,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
        if (result != cudaSuccess) {
            if (result == cudaErrorInvalidValue || result == cudaErrorNotSupported) {
                (void)cudaGetLastError();
                return Pair::LaunchResult::NotSupported;
            }
            return Pair::LaunchResult::Error;
        }
        configured_device = device;
    }
    fused<<<dim3(n_out/kRows,(n_tok+kTokens-1)/kTokens),dim3(32,kWarps),kSharedBytes,stream>>>(
        gate,up,x,y,n_in,n_out,n_tok);
    return cudaPeekAtLastError() == cudaSuccess
        ? Pair::LaunchResult::Launched : Pair::LaunchResult::Error;
}
} // namespace canonical_q8_pair_lab
