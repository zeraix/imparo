#pragma once
#include "vendor/flashinfer/attention/default_prefill_params.cuh"
#include "vendor/flashinfer/attention/prefill.cuh"
namespace imparo_d64_fa2_port {
struct PartitionParams:flashinfer::SinglePrefillParams<__half,__half,__half>{
 using flashinfer::SinglePrefillParams<__half,__half,__half>::SinglePrefillParams;
 uint32_t partition_reference_len=0;
 uint32_t partition_fixed_span=0;
};

__global__ void prepare_q(const float *q,__half *dst,unsigned n,float scale){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=__hmul(__float2half_rn(q[i]/scale),__float2half_rn(scale));}
__global__ void output_f32(const __half *src,float *dst,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=__half2float(src[i]);}
template<class Params> cudaError_t launch_linear(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,unsigned nt,unsigned nh,unsigned nk,unsigned valid,float scale,bool causal,cudaStream_t stream,unsigned fixed_span){
 const unsigned words=nt*nh*64;
 prepare_q<<<(words+255)/256,256,0,stream>>>(q,qh,words,scale);
 using Variant=flashinfer::DefaultAttention<false,false,false,false>;
 Params params(qh,const_cast<__half*>(k),const_cast<__half*>(v),nullptr,oh,nullptr,nullptr,nh,nk,nt,valid,nh*64,64,nk*64,64,64,-1,0.f,1.f,1.f,10000.f);
 if constexpr(std::is_same_v<Params,PartitionParams>)params.partition_fixed_span=fixed_span;
 cudaError_t status;
 if(causal)status=flashinfer::SinglePrefillWithKVCacheDispatched<64,64,flashinfer::PosEncodingMode::kNone,false,flashinfer::MaskMode::kCausal,Variant>(params,tmp,stream);
 else status=flashinfer::SinglePrefillWithKVCacheDispatched<64,64,flashinfer::PosEncodingMode::kNone,false,flashinfer::MaskMode::kNone,Variant>(params,tmp,stream);
 if(status!=cudaSuccess)return status;
 output_f32<<<(words+255)/256,256,0,stream>>>(oh,out,words);return cudaGetLastError();
}
cudaError_t launch(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,unsigned nt,unsigned nh,unsigned nk,unsigned valid,float scale,bool causal,cudaStream_t stream,unsigned fixed_span=0){
 if(fixed_span)return launch_linear<PartitionParams>(q,k,v,out,qh,oh,tmp,nt,nh,nk,valid,scale,causal,stream,fixed_span);
 return launch_linear<flashinfer::SinglePrefillParams<__half,__half,__half>>(q,k,v,out,qh,oh,tmp,nt,nh,nk,valid,scale,causal,stream,0);
}
cudaError_t launch_mask(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,unsigned nt,unsigned nh,unsigned nk,unsigned valid,float scale,uint8_t*mask,cudaStream_t stream,unsigned partition_reference_len,unsigned fixed_span){
 const unsigned words=nt*nh*64;prepare_q<<<(words+255)/256,256,0,stream>>>(q,qh,words,scale);
 using Params=PartitionParams;using Variant=flashinfer::DefaultAttention<true,false,false,false>;
 Params params(qh,const_cast<__half*>(k),const_cast<__half*>(v),mask,oh,nullptr,nullptr,nh,nk,nt,valid,nh*64,64,nk*64,64,64,-1,0.f,1.f,1.f,10000.f);
 if(partition_reference_len>valid || (partition_reference_len && partition_reference_len<nt))return cudaErrorInvalidValue;
 params.partition_reference_len=partition_reference_len;
 params.partition_fixed_span=fixed_span;
 auto status=flashinfer::SinglePrefillWithKVCacheDispatched<64,64,flashinfer::PosEncodingMode::kNone,false,flashinfer::MaskMode::kCustom,Variant>(params,tmp,stream);
 if(status!=cudaSuccess)return status;output_f32<<<(words+255)/256,256,0,stream>>>(oh,out,words);return cudaGetLastError();
}
}
