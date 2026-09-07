#pragma once

#include "q8_mma_ready_a0.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Strictly isolated SM86 prototype for the adjacent Gemma4 boundary:
//
//   mid = rms(src, post_attention_weight) + residual
//   out = rms(mid, ffn_weight)
//   out -> Q8 MMA-ready producer layout
//
// The public/backend ABI and selectors do not include this file yet.  A caller
// must explicitly include it and invoke launch(), so merely compiling this
// header cannot change the default CUDA or Metal paths.
namespace imparo_sm86_dual_rms_q8_ready {

using ReadyLayout = imparo_q8_mma_ready_a0_v1_authority_lab::Layout;

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kWidth = 2560;
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kMaxTokens = 512;
constexpr uint32_t kMidBytes = kWidth * sizeof(float);
constexpr uint32_t kReductionBytes = kWarps * sizeof(float);
constexpr uint32_t kSharedBytes = kMidBytes + kReductionBytes;

static_assert(kWidth %
                  imparo_q8_mma_ready_a0_v1_authority_lab::kValuesPerGroup
              == 0,
              "width must contain complete K128 ready-layout groups");
static_assert(kThreads == 1024, "preserve the admitted RMS reduction geometry");
static_assert(kSharedBytes == 10368, "fixed dual-RMS shared-memory budget");

// Exact namespaced copy of imparo_cuda.cu's XOR warp reduction.  The header is
// included before that translation-unit helper is declared, so it cannot call
// the global helper without introducing a declaration-order dependency.
__device__ __forceinline__ float warp_sum_xor(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Preserve imparo_cuda.cu::block_sum<1024> line-for-line: each thread owns the
// same strided columns and the same two XOR reduction levels as both standalone
// RMS kernels.
template <uint32_t BlockSize>
__device__ __forceinline__ float block_sum(float value, float *shared) {
    value = warp_sum_xor(value);
    constexpr uint32_t Warps = BlockSize / 32;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = lane < Warps ? shared[lane] : 0.0f;
    value = warp_sum_xor(value);
    // The fused kernel invokes this reduction twice with the same shared slots.
    // Do not let a faster warp publish the second reduction while warp zero is
    // still reading the first reduction's per-warp values.
    __syncthreads();
    return value;
}

__launch_bounds__(kThreads, 1)
__global__ void rms_norm_add_dual_q8_ready(
        const float *__restrict__ src,
        const float *__restrict__ residual,
        const float *__restrict__ post_attention_weight,
        const float *__restrict__ ffn_weight,
        float *__restrict__ mid_out,
        float *__restrict__ norm_out,
        uint16_t *__restrict__ quant_u16,
        float *__restrict__ d8_sideplane,
        uint32_t n_tok,
        uint32_t token_tiles,
        float first_eps,
        float second_eps) {
#if __CUDA_ARCH__ == 860
    using namespace imparo_q8_mma_ready_a0_v1_authority_lab;

    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;

    const uint64_t row_offset = uint64_t(tok) * kWidth;
    const float *src_row = src + row_offset;
    const float *residual_row = residual + row_offset;
    float *mid_out_row = mid_out + row_offset;
    float *norm_out_row = norm_out + row_offset;

    extern __shared__ __align__(16) float shared[];
    float *mid_shared = shared;
    float *reduction_shared = shared + kWidth;

    // First RMS: identical 1024-thread strided ownership and accumulation order
    // to k_rms_norm_ggml<1024>.
    float first_sum = 0.0f;
    for (uint32_t col = tid; col < kWidth; col += kThreads) {
        const float value = src_row[col];
        first_sum += value * value;
    }
    first_sum = block_sum<kThreads>(first_sum, reduction_shared);
    const float first_mean = first_sum / kWidth;
    const float first_scale = rsqrtf(first_mean + first_eps);

    // Publish the first operator's f32 seam exactly.  The global copy remains
    // mandatory because it is the later FFN residual.  The shared copy avoids
    // rereading that just-published row from global memory in the second RMS.
    // Accumulating second_sum in this same thread-strided loop preserves the
    // standalone second RMS ownership rather than changing it to float4 order.
    float second_sum = 0.0f;
    for (uint32_t col = tid; col < kWidth; col += kThreads) {
        const float value = first_scale * src_row[col]
            * post_attention_weight[col] + residual_row[col];
        mid_out_row[col] = value;
        mid_shared[col] = value;
        second_sum += value * value;
    }
    second_sum = block_sum<kThreads>(second_sum, reduction_shared);
    const float second_scale = rsqrtf(second_sum / kWidth + second_eps);

    const uint32_t token_tile = tok / kTokenTile;
    const uint32_t token_in_tile = tok % kTokenTile;

    // Second RMS + ready-layout producer: keep k_rms_norm_q8_mma_ready's
    // float4 ownership, 8-lane XOR max, roundf quantization, and f16-rounded
    // scale promotion verbatim.
    for (uint32_t i0 = tid * 4; i0 < kWidth; i0 += kThreads * 4) {
        const float4 mid = reinterpret_cast<const float4 *>(
            mid_shared + i0)[0];
        float4 value;
        value.x = second_scale * mid.x * ffn_weight[i0 + 0];
        value.y = second_scale * mid.y * ffn_weight[i0 + 1];
        value.z = second_scale * mid.z * ffn_weight[i0 + 2];
        value.w = second_scale * mid.w * ffn_weight[i0 + 3];
        reinterpret_cast<float4 *>(norm_out_row)[i0 / 4] = value;

        float amax = fabsf(value.x);
        amax = fmaxf(amax, fabsf(value.y));
        amax = fmaxf(amax, fabsf(value.z));
        amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(
                amax,
                __shfl_xor_sync(0xffffffffu, amax, offset, 32));
        }
        const float d_inv = 127.0f / amax;
        char4 quant;
        quant.x = int8_t(roundf(value.x * d_inv));
        quant.y = int8_t(roundf(value.y * d_inv));
        quant.z = int8_t(roundf(value.z * d_inv));
        quant.w = int8_t(roundf(value.w * d_inv));
        const uint32_t block = i0 / kValuesPerQBlock;
        const uint32_t group = block / kQBlocksPerGroup;
        const uint32_t qblock = block % kQBlocksPerGroup;
        const uint32_t kpair = (i0 % kValuesPerQBlock) / 2;
        quant_u16[quant_u16_index(
            token_tiles, group, token_tile, qblock, kpair,
            token_in_tile)] = uint16_t(uint8_t(quant.x))
                | (uint16_t(uint8_t(quant.y)) << 8);
        quant_u16[quant_u16_index(
            token_tiles, group, token_tile, qblock, kpair + 1,
            token_in_tile)] = uint16_t(uint8_t(quant.z))
                | (uint16_t(uint8_t(quant.w)) << 8);
        if ((i0 % kValuesPerQBlock) == 0) {
            d8_sideplane[scale_index(
                token_tiles, group, token_tile, qblock,
                token_in_tile)] =
                    __half2float(__float2half(1.0f / d_inv));
        }
    }
#else
    (void)src;
    (void)residual;
    (void)post_attention_weight;
    (void)ffn_weight;
    (void)mid_out;
    (void)norm_out;
    (void)quant_u16;
    (void)d8_sideplane;
    (void)n_tok;
    (void)token_tiles;
    (void)first_eps;
    (void)second_eps;
#endif
}

inline LaunchResult launch(
        const float *src,
        const float *residual,
        const float *post_attention_weight,
        const float *ffn_weight,
        float *mid_out,
        float *norm_out,
        uint16_t *quant_u16,
        float *d8_sideplane,
        uint32_t width,
        uint32_t n_tok,
        const ReadyLayout &layout,
        float first_eps,
        float second_eps,
        uint32_t sm_version,
        cudaStream_t stream) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(src)
        | reinterpret_cast<uintptr_t>(residual)
        | reinterpret_cast<uintptr_t>(post_attention_weight)
        | reinterpret_cast<uintptr_t>(ffn_weight)
        | reinterpret_cast<uintptr_t>(mid_out)
        | reinterpret_cast<uintptr_t>(norm_out)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane);
    const uint32_t padded_tokens = n_tok
        ? ((n_tok + 127u) / 128u) * 128u : 0u;
    if (sm_version != 86 || width != kWidth || !n_tok
            || n_tok > kMaxTokens || (pointers & 15u) != 0
            || !src || !residual || !post_attention_weight || !ffn_weight
            || !mid_out || !norm_out || !quant_u16 || !d8_sideplane
            || norm_out == mid_out || norm_out == residual
            || layout.n_in != kWidth || layout.n_tok != n_tok
            || layout.padded_tokens != padded_tokens
            || layout.token_tiles != padded_tokens /
                imparo_q8_mma_ready_a0_v1_authority_lab::kTokenTile
            || layout.groups != kWidth /
                imparo_q8_mma_ready_a0_v1_authority_lab::kValuesPerGroup) {
        return LaunchResult::NotSupported;
    }

    rms_norm_add_dual_q8_ready
        <<<n_tok, kThreads, kSharedBytes, stream>>>(
            src, residual, post_attention_weight, ffn_weight,
            mid_out, norm_out, quant_u16, d8_sideplane,
            n_tok, layout.token_tiles, first_eps, second_eps);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

}  // namespace imparo_sm86_dual_rms_q8_ready
