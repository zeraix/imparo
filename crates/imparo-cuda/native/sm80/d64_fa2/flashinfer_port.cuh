#pragma once
#include "vendor/flashinfer/attention/default_prefill_params.cuh"
#include "vendor/flashinfer/attention/prefill.cuh"
namespace imparo_d64_fa2_port {
struct PartitionParams:flashinfer::SinglePrefillParams<__half,__half,__half>{
 using flashinfer::SinglePrefillParams<__half,__half,__half>::SinglePrefillParams;
 uint32_t partition_reference_len=0;
 uint32_t partition_fixed_span=0;
 uint32_t batch_invariant_q8_v1=0;
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

// The owner contract pins the current CTA64 M9 arithmetic. Real qo_len/kv_len
// remain intact; partition count is physical storage stride, never a query mask.
namespace imparo_d64_fa2_port {
using BatchInvariantVariant=flashinfer::DefaultAttention<false,false,false,false>;
using BatchInvariantKT=flashinfer::KernelTraits<flashinfer::MaskMode::kCausal,64,1,8,4,4,4,1,flashinfer::PosEncodingMode::kNone,__half,__half,__half,float,int,BatchInvariantVariant>;

// Existing MergeStatesKernel<8> ascending arithmetic, with a per-query logical
// prefix and a separate physical stride. No future partition is read or merged.
__global__ void merge_v1(const __half* v,const float* s,__half* out,
 unsigned nt,unsigned chunks,unsigned start,unsigned span,const int* parents){
 const unsigned tx=threadIdx.x,head=threadIdx.y,row=blockIdx.x;
 unsigned depth=row;if(parents){depth=0;for(int n=int(row);parents[n]>=0;n=parents[n])++depth;}
 const unsigned count=(start+depth+1+span-1)/span;
 if(count==1){
  flashinfer::vec_t<__half,8> x;
  x.cast_load(v+uint64_t(row)*chunks*2048+head*64+tx*8);
  x.store(out+uint64_t(row)*2048+head*64+tx*8);return;
 }
 flashinfer::state_t<8> st;
 #pragma unroll 2
 for(unsigned part=0;part<count;++part){
  flashinfer::vec_t<float,8> x;
  x.cast_load(v+(uint64_t(row)*chunks+part)*2048+head*64+tx*8);
  st.merge(x,s[(uint64_t(row)*chunks+part)*32+head],1);
 }
 st.normalize();st.o.cast_store(out+uint64_t(row)*2048+head*64+tx*8);
}
cudaError_t launch_batch_invariant_v1(const float*q,const __half*k,const __half*v,float*out,
 __half*qh,__half*oh,__half*tmp,unsigned nt,unsigned valid,float scale,unsigned span,cudaStream_t stream){
 if(!nt||nt>16||valid<nt||span<256||span%128||!tmp||!qh||!oh)return cudaErrorInvalidValue;
 const unsigned chunks=(valid+span-1)/span,words=nt*2048;
 auto kernel=flashinfer::SinglePrefillWithKVCacheKernel<BatchInvariantKT,PartitionParams>;
 static bool configured=false;
 if(!configured){cudaStreamCaptureStatus capture;auto e=cudaStreamIsCapturing(stream,&capture);if(e)return e;
  if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
  e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(BatchInvariantKT::SharedStorage));if(e)return e;configured=true;}
 prepare_q<<<(words+255)/256,256,0,stream>>>(q,qh,words,scale);
 auto*ls=reinterpret_cast<float*>(tmp+uint64_t(chunks)*words);
 PartitionParams p(qh,const_cast<__half*>(k),const_cast<__half*>(v),nullptr,tmp,ls,nullptr,32,8,nt,valid,2048,64,512,64,64,-1,0.f,1.f,1.f,10000.f);
 p.partition_kv=true;p.partition_fixed_span=span;p.batch_invariant_q8_v1=1;
 void*args[]={&p};auto e=cudaLaunchKernel((void*)kernel,dim3(1,chunks,8),dim3(32,4,1),args,sizeof(BatchInvariantKT::SharedStorage),stream);if(e)return e;
 merge_v1<<<nt,dim3(8,32),0,stream>>>(tmp,ls,oh,nt,chunks,valid-nt,span,nullptr);
 output_f32<<<(words+255)/256,256,0,stream>>>(oh,out,words);return cudaGetLastError();
}
}
