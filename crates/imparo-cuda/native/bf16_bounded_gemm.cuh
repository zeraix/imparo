#pragma once
#include "ptq1_bounded_gemm.cuh"
namespace imparo_cuda_bf16_gemm {
__global__ void expand(const uint16_t*w,float*y,size_t count) {
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count)y[i]=__uint_as_float(uint32_t(w[i])<<16);
}
inline cudaError_t launch(imparo_cuda_ptq1_gemm::State&s,const uint8_t*w,const float*x,float*y,
 uint32_t k,uint32_t n,uint32_t m,uint32_t stride,uint32_t base,cudaStream_t stream,
 bool weights_are_f32=false) {
 using namespace imparo_cuda_ptq1_gemm;
 const uint64_t required=weights_are_f32?workspace_bytes:storage_bytes;
 if(!s.handle||!s.storage||s.bytes<required||!w||!x||!y||!k||!n||!m||m>512||base>stride||n>stride-base||uint64_t(k)*n*4>weight_bytes)return cudaErrorInvalidValue;
 auto*bytes=static_cast<uint8_t*>(s.storage);auto*expanded=reinterpret_cast<float*>(bytes);
 const float*wf=weights_are_f32?reinterpret_cast<const float*>(w):expanded;
 // Full owners retain their existing workspace offset. Already-F32 weights
 // need no expansion or activation staging, so a compact owner uses only base.
 void*workspace=s.bytes>=storage_bytes?bytes+weight_bytes+activation_bytes:bytes;
 if(cublasSetStream(s.handle,stream)!=CUBLAS_STATUS_SUCCESS||cublasSetPointerMode(s.handle,CUBLAS_POINTER_MODE_HOST)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(s.handle,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS||cublasSetWorkspace(s.handle,workspace,workspace_bytes)!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
 // Already-F32 GDN weights reuse the same pedantic GEMM without expansion.
 if(!weights_are_f32) {
  size_t count=size_t(k)*n;expand<<<unsigned((count+255)/256),256,0,stream>>>(reinterpret_cast<const uint16_t*>(w),expanded,count);
  auto err=cudaGetLastError();if(err!=cudaSuccess)return err;
 }
 const float alpha=1,beta=0;
 if(cublasGemmEx(s.handle,CUBLAS_OP_T,CUBLAS_OP_N,int(n),int(m),int(k),&alpha,wf,CUDA_R_32F,int(k),x,CUDA_R_32F,int(k),&beta,y+base,CUDA_R_32F,int(stride),CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS)return cudaErrorUnknown;
 return cudaSuccess;
}
}
