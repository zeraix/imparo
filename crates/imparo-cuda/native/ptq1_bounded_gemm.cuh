#pragma once
// Bounded form of the existing quantized-weight -> F16 -> cuBLAS organization.
// Unlike the historical Q4 lab, this provider accumulates AND stores FP32.
// Model basis, resident/streamed placement and selection belong to the caller.
// Source-build opt-in only: official driver-only plugins may omit this provider.
#include "ptq1.cuh"
#include <cublas_v2.h>
#include <algorithm>
namespace imparo_cuda_ptq1_gemm {
constexpr uint64_t weight_bytes = 64ull << 20;
constexpr uint64_t activation_bytes = 128ull * 17408 * sizeof(__half);
constexpr uint64_t workspace_bytes = 4ull << 20;
constexpr uint64_t storage_bytes = weight_bytes + activation_bytes + workspace_bytes;
struct State { cublasHandle_t handle=nullptr; void* storage=nullptr; uint64_t bytes=0; };
__global__ void dequant(const uint8_t* w, __half* out, size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) out[i]=__float2half_rn(imparo_cuda_ptq1::element(w+(i/128)*28,int(i%128)));
}
__global__ void convert(const float* x,__half* out,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) out[i]=__float2half_rn(x[i]);
}
inline bool release(State& s) {
    if(s.handle) {
        if(cublasDestroy(s.handle)!=CUBLAS_STATUS_SUCCESS) return false;
        s.handle=nullptr;
    }
    if(s.storage && cudaFree(s.storage)!=cudaSuccess) return false;
    s.storage=nullptr;s.bytes=0;return true;
}
inline cudaError_t launch(State& s,const uint8_t* w,const float* x,float* y,
        uint32_t k,uint32_t n,uint32_t m,uint32_t stride,uint32_t base,cudaStream_t stream) {
    if(!s.handle||!s.storage||s.bytes<storage_bytes||!w||!x||!y
            ||!k||k%128||k>17408||!n||!m||m>128||base>stride||n>stride-base
            ||uint64_t(m)*k*sizeof(__half)>activation_bytes) return cudaErrorInvalidValue;
    auto* bytes=static_cast<uint8_t*>(s.storage);
    auto* weights=reinterpret_cast<__half*>(bytes);
    auto* acts=reinterpret_cast<__half*>(bytes+weight_bytes);
    // cublasSetStream resets workspace; restore it after each stream selection.
    if(cublasSetStream(s.handle,stream)!=CUBLAS_STATUS_SUCCESS
       ||cublasSetPointerMode(s.handle,CUBLAS_POINTER_MODE_HOST)!=CUBLAS_STATUS_SUCCESS
       ||cublasSetMathMode(s.handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS
       ||cublasSetWorkspace(s.handle,bytes+weight_bytes+activation_bytes,workspace_bytes)!=CUBLAS_STATUS_SUCCESS)
        return cudaErrorUnknown;
    size_t count=size_t(m)*k;
    convert<<<unsigned((count+255)/256),256,0,stream>>>(x,acts,count);
    auto err=cudaGetLastError();if(err!=cudaSuccess)return err;
    uint32_t chunk=uint32_t(weight_bytes/(sizeof(__half)*k)/64)*64;
    if(!chunk)return cudaErrorInvalidValue;
    const float alpha=1,beta=0;
    for(uint32_t row=0;row<n;row+=chunk) {
        uint32_t nr=std::min(chunk,n-row);count=size_t(nr)*k;
        dequant<<<unsigned((count+255)/256),256,0,stream>>>(w+size_t(row)*(k/128)*28,weights,count);
        err=cudaGetLastError();if(err!=cudaSuccess)return err;
        if(cublasGemmEx(s.handle,CUBLAS_OP_T,CUBLAS_OP_N,int(nr),int(m),int(k),&alpha,
           weights,CUDA_R_16F,int(k),acts,CUDA_R_16F,int(k),&beta,
           y+base+row,CUDA_R_32F,int(stride),CUBLAS_COMPUTE_32F,
           CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
    }
    return cudaSuccess;
}
} // namespace imparo_cuda_ptq1_gemm
