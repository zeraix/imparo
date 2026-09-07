// SPDX-License-Identifier: MIT
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Imparo contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Provenance and numerical contract
// ---------------------------------
// This isolated provider mirrors the large-batch Q4_0 CUDA route in llama.cpp
// commit 4695f001fece1660d8bb1b3748f50726ddcc100b:
//   ggml/src/ggml-cuda/convert.cu:84-110, 285-290
//   ggml/src/ggml-cuda/ggml-cuda.cu:1391-1402, 1547-1555
//
// The route is intentionally explicit:
//   raw Q4_0 -> F16 weights
//   token-major F32 -> F16 activations
//   cublasGemmEx(T, N, CUDA_R_16F, CUBLAS_COMPUTE_16F)
//   F16 result -> token-major F32 destination
//
// No allocation or synchronization is hidden here. The caller owns the handle,
// stream, three scratch buffers and their lifetimes. This header is not included
// by the production translation unit yet and therefore does not activate a route.

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace imparo_sm86_gemm_q4_f16_cublas {

constexpr uint32_t kQ4BlockValues = 32;
constexpr uint32_t kQ4BlockBytes = 18;
constexpr uint32_t kConvertThreads = 256;
constexpr uint32_t kDequantThreads = 32;
constexpr uint32_t kQ4BlocksPerDequantCta = 8;

enum class ProviderStatus : uint8_t {
    Ready,
    InvalidHandle,
    InvalidPointer,
    InvalidShape,
    InvalidStride,
    SizeOverflow,
    ScratchTooSmall,
    ScratchOverlap,
    HandleConfigurationFailed,
    KernelLaunchFailed,
    GemmFailed,
};

enum class LaunchStage : uint8_t {
    None,
    ConfigureHandle,
    DequantizeWeights,
    ConvertActivations,
    Gemm,
    ConvertOutput,
};

struct ScratchSizes {
    size_t weights_f16 = 0;
    size_t activations_f16 = 0;
    size_t output_f16 = 0;

    constexpr size_t total() const {
        return weights_f16 + activations_f16 + output_f16;
    }
};

struct LaunchInfo {
    uint32_t n_in = 0;
    uint32_t n_out = 0;
    uint32_t n_tok = 0;
    uint32_t dst_stride = 0;
    uint32_t row_base = 0;
    uint32_t q4_blocks = 0;
    uint32_t dequant_grid = 0;
    uint32_t activation_grid = 0;
    uint32_t output_grid = 0;
    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    size_t weights_q4_bytes = 0;
    ScratchSizes scratch{};

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cudaDataType_t a_type = CUDA_R_16F;
    cudaDataType_t b_type = CUDA_R_16F;
    cudaDataType_t c_type = CUDA_R_16F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    cublasGemmAlgo_t algorithm = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

struct ProviderArgs {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;

    const uint8_t * weights_q4_0 = nullptr;
    const float * activations_f32 = nullptr;
    float * output_f32 = nullptr;

    __half * weights_f16 = nullptr;
    size_t weights_f16_bytes = 0;
    __half * activations_f16 = nullptr;
    size_t activations_f16_bytes = 0;
    __half * output_f16 = nullptr;
    size_t output_f16_bytes = 0;

    uint32_t n_in = 0;
    uint32_t n_out = 0;
    uint32_t n_tok = 0;
    uint32_t dst_stride = 0;
    uint32_t row_base = 0;
    bool direct_gelu_mul = false;
    bool skip_activation_convert = false;
    bool skip_weight_dequant = false;
};

struct LaunchResult {
    ProviderStatus status = ProviderStatus::Ready;
    LaunchStage stage = LaunchStage::None;
    cudaError_t cuda_status = cudaSuccess;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    LaunchInfo info{};

    constexpr bool ok() const {
        return status == ProviderStatus::Ready;
    }
};

constexpr uint64_t ceil_div_u64(uint64_t value, uint32_t divisor) {
    return value / divisor + uint64_t(value % divisor != 0);
}

inline bool byte_size(uint64_t elements, size_t element_size, size_t * out) {
    if (!out || elements > std::numeric_limits<size_t>::max() / element_size) {
        return false;
    }
    *out = size_t(elements) * element_size;
    return true;
}

inline ProviderStatus make_launch_info(
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t dst_stride, uint32_t row_base,
        LaunchInfo * info) {
    if (!info) return ProviderStatus::InvalidPointer;
    *info = {};
    if (n_in == 0 || n_out == 0 || n_tok == 0
        || n_in % kQ4BlockValues != 0
        || n_in > uint32_t(INT_MAX)
        || n_out > uint32_t(INT_MAX)
        || n_tok > uint32_t(INT_MAX)) {
        return ProviderStatus::InvalidShape;
    }
    if (row_base > dst_stride || n_out > dst_stride - row_base) {
        return ProviderStatus::InvalidStride;
    }

    const uint64_t weights_elements = uint64_t(n_in) * n_out;
    const uint64_t activation_elements = uint64_t(n_in) * n_tok;
    const uint64_t output_elements = uint64_t(n_out) * n_tok;
    const uint64_t q4_blocks = uint64_t(n_in / kQ4BlockValues) * n_out;
    const uint64_t dequant_grid = ceil_div_u64(q4_blocks, kQ4BlocksPerDequantCta);
    const uint64_t activation_grid = ceil_div_u64(activation_elements, kConvertThreads);
    const uint64_t output_grid = ceil_div_u64(output_elements, kConvertThreads);
    if (q4_blocks > std::numeric_limits<uint32_t>::max()
        || dequant_grid > std::numeric_limits<uint32_t>::max()
        || activation_grid > std::numeric_limits<uint32_t>::max()
        || output_grid > std::numeric_limits<uint32_t>::max()
        || !byte_size(q4_blocks, kQ4BlockBytes, &info->weights_q4_bytes)
        || !byte_size(weights_elements, sizeof(__half), &info->scratch.weights_f16)
        || !byte_size(activation_elements, sizeof(__half), &info->scratch.activations_f16)
        || !byte_size(output_elements, sizeof(__half), &info->scratch.output_f16)) {
        return ProviderStatus::SizeOverflow;
    }

    info->n_in = n_in;
    info->n_out = n_out;
    info->n_tok = n_tok;
    info->dst_stride = dst_stride;
    info->row_base = row_base;
    info->q4_blocks = uint32_t(q4_blocks);
    info->dequant_grid = uint32_t(dequant_grid);
    info->activation_grid = uint32_t(activation_grid);
    info->output_grid = uint32_t(output_grid);
    info->gemm_m = int(n_out);
    info->gemm_n = int(n_tok);
    info->gemm_k = int(n_in);
    info->lda = int(n_in);
    info->ldb = int(n_in);
    info->ldc = int(n_out);
    return ProviderStatus::Ready;
}

inline bool ranges_overlap(const void * a, size_t a_size,
                           const void * b, size_t b_size) {
    const uintptr_t ap = reinterpret_cast<uintptr_t>(a);
    const uintptr_t bp = reinterpret_cast<uintptr_t>(b);
    if (ap > std::numeric_limits<uintptr_t>::max() - a_size
        || bp > std::numeric_limits<uintptr_t>::max() - b_size) {
        return true;
    }
    return ap < bp + b_size && bp < ap + a_size;
}

inline ProviderStatus validate(const ProviderArgs & args, LaunchInfo * info) {
    if (!args.handle) return ProviderStatus::InvalidHandle;
    if (!args.weights_q4_0 || !args.activations_f32 || !args.output_f32
        || !args.weights_f16 || !args.activations_f16 || !args.output_f16) {
        return ProviderStatus::InvalidPointer;
    }
    const ProviderStatus shape = make_launch_info(
        args.n_in, args.n_out, args.n_tok,
        args.dst_stride, args.row_base, info);
    if (shape != ProviderStatus::Ready) return shape;
    if (args.weights_f16_bytes < info->scratch.weights_f16
        || args.activations_f16_bytes < info->scratch.activations_f16
        || args.output_f16_bytes < info->scratch.output_f16) {
        return ProviderStatus::ScratchTooSmall;
    }
    if (ranges_overlap(args.weights_f16, info->scratch.weights_f16,
                       args.activations_f16, info->scratch.activations_f16)
        || ranges_overlap(args.weights_f16, info->scratch.weights_f16,
                          args.output_f16, info->scratch.output_f16)
        || ranges_overlap(args.activations_f16, info->scratch.activations_f16,
                          args.output_f16, info->scratch.output_f16)) {
        return ProviderStatus::ScratchOverlap;
    }
    return ProviderStatus::Ready;
}

#if defined(__CUDACC__)

// Byte-for-byte numerical expression from pinned convert.cu. Do not rewrite
// as `(q - 8) * d`: compiler contraction/rounding can produce a different half.
__global__ void dequantize_q4_0_f16_kernel(
        const uint8_t * __restrict__ vx,
        __half * __restrict__ yy,
        uint64_t nb32) {
    const int64_t i = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ir = tid % 8;
    const int64_t ib = 8 * i + ir;
    if (uint64_t(ib) >= nb32) return;

    __half * y = yy + 256 * i + 32 * ir + 4 * il;
    const uint8_t * x = vx + uint64_t(ib) * kQ4BlockBytes;
    const float d = __half2float(*reinterpret_cast<const __half *>(x));
    const float dm = -8.0f * d;
    const uint8_t * q = x + 2 + 4 * il;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        y[l + 0] = __float2half_rn(d * float(q[l] & 0x0f) + dm);
        y[l + 16] = __float2half_rn(d * float(q[l] >> 4) + dm);
    }
}

__global__ void convert_f32_f16_kernel(
        const float * __restrict__ src,
        __half * __restrict__ dst,
        uint64_t elements) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements) dst[i] = __float2half_rn(src[i]);
}

template <bool DirectGeluMul>
__global__ void convert_output_f16_f32_kernel(
        const __half * __restrict__ src,
        float * __restrict__ dst,
        uint32_t n_out, uint32_t n_tok,
        uint32_t dst_stride, uint32_t row_base) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = uint64_t(n_out) * n_tok;
    if (i >= elements) return;
    const uint32_t token = uint32_t(i / n_out);
    const uint32_t row = uint32_t(i - uint64_t(token) * n_out);
    float * slot = dst + uint64_t(token) * dst_stride + row_base + row;
    const float value = __half2float(src[i]);
    if constexpr (DirectGeluMul) {
        *slot = cuda_gelu(*slot) * value;
    } else {
        *slot = value;
    }
}

#endif // __CUDACC__

inline LaunchResult launch(const ProviderArgs & args) {
    LaunchResult result{};
    result.status = validate(args, &result.info);
    if (result.status != ProviderStatus::Ready) return result;

#if !defined(__CUDACC__)
    result.status = ProviderStatus::KernelLaunchFailed;
    result.stage = LaunchStage::DequantizeWeights;
    result.cuda_status = cudaErrorNotSupported;
    return result;
#else
    result.stage = LaunchStage::ConfigureHandle;
    result.cublas_status = cublasSetStream(args.handle, args.stream);
    if (result.cublas_status != CUBLAS_STATUS_SUCCESS) {
        result.status = ProviderStatus::HandleConfigurationFailed;
        return result;
    }
    result.cublas_status = cublasSetPointerMode(
        args.handle, CUBLAS_POINTER_MODE_HOST);
    if (result.cublas_status != CUBLAS_STATUS_SUCCESS) {
        result.status = ProviderStatus::HandleConfigurationFailed;
        return result;
    }

    if (!args.skip_weight_dequant) {
        result.stage = LaunchStage::DequantizeWeights;
        dequantize_q4_0_f16_kernel<<<
            result.info.dequant_grid, kDequantThreads, 0, args.stream>>>(
                args.weights_q4_0, args.weights_f16, result.info.q4_blocks);
        result.cuda_status = cudaPeekAtLastError();
        if (result.cuda_status != cudaSuccess) {
            result.status = ProviderStatus::KernelLaunchFailed;
            return result;
        }
    }

    if (!args.skip_activation_convert) {
        result.stage = LaunchStage::ConvertActivations;
        convert_f32_f16_kernel<<<
            result.info.activation_grid, kConvertThreads, 0, args.stream>>>(
                args.activations_f32, args.activations_f16,
                uint64_t(args.n_in) * args.n_tok);
        result.cuda_status = cudaPeekAtLastError();
        if (result.cuda_status != cudaSuccess) {
            result.status = ProviderStatus::KernelLaunchFailed;
            return result;
        }
    }

    result.stage = LaunchStage::Gemm;
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    result.cublas_status = cublasGemmEx(
        args.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        int(args.n_out), int(args.n_tok), int(args.n_in),
        &alpha,
        args.weights_f16, CUDA_R_16F, int(args.n_in),
        args.activations_f16, CUDA_R_16F, int(args.n_in),
        &beta,
        args.output_f16, CUDA_R_16F, int(args.n_out),
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (result.cublas_status != CUBLAS_STATUS_SUCCESS) {
        result.status = ProviderStatus::GemmFailed;
        return result;
    }

    result.stage = LaunchStage::ConvertOutput;
    if (args.direct_gelu_mul) {
        convert_output_f16_f32_kernel<true><<<
            result.info.output_grid, kConvertThreads, 0, args.stream>>>(
                args.output_f16, args.output_f32,
                args.n_out, args.n_tok, args.dst_stride, args.row_base);
    } else {
        convert_output_f16_f32_kernel<false><<<
            result.info.output_grid, kConvertThreads, 0, args.stream>>>(
                args.output_f16, args.output_f32,
                args.n_out, args.n_tok, args.dst_stride, args.row_base);
    }
    result.cuda_status = cudaPeekAtLastError();
    if (result.cuda_status != cudaSuccess) {
        result.status = ProviderStatus::KernelLaunchFailed;
        return result;
    }

    result.stage = LaunchStage::None;
    result.status = ProviderStatus::Ready;
    return result;
#endif
}

// Integration contract:
// - Inputs and all scratches are device pointers on the handle's device.
// - weights_q4_0 is contiguous [n_out][n_in/32] in the 18-byte Q4_0 ABI.
// - activations_f32 is contiguous token-major [n_tok][n_in].
// - the three scratch ranges are distinct and remain live until stream completion.
// - output_f32 may have a larger token stride; only [row_base,row_base+n_out)
//   is written. Input/output/scratch aliasing is forbidden by the caller.
// - the caller must serialize handle mutation or dedicate a handle per stream.
// - the provider sets stream and HOST pointer mode, but does not change math mode.
// - no cudaMalloc/cudaFree/cudaStreamSynchronize/cudaDeviceSynchronize occurs.

} // namespace imparo_sm86_gemm_q4_f16_cublas
