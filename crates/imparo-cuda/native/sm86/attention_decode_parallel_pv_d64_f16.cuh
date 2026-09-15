#pragma once
// Scoped D64/GQA4 FP32 parallel reduction; intentionally changes PV addition order.
namespace imparo_sm86_parallel_pv {
__device__ __forceinline__ float sum_warp(float v){for(int d=16;d;d>>=1)v+=__shfl_xor_sync(0xffffffff,v,d);return v;}
__device__ __forceinline__ float max_warp(float v){for(int d=16;d;d>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));return v;}
__global__ void probabilities(const float*scores,float*prob,unsigned valid){
 const unsigned h=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;__shared__ float scratch[8];float m=-INFINITY;
 for(unsigned i=t;i<valid;i+=256)m=fmaxf(m,scores[uint64_t(h)*valid+i]);m=max_warp(m);if(lane==0)scratch[warp]=m;__syncthreads();m=max_warp(lane<8?scratch[lane]:-INFINITY);__syncthreads();
 float s=0.f;for(unsigned i=t;i<valid;i+=256)s+=expf(scores[uint64_t(h)*valid+i]-m);s=sum_warp(s);if(lane==0)scratch[warp]=s;__syncthreads();s=sum_warp(lane<8?scratch[lane]:0.f);const float inv=1.f/s;
 for(unsigned i=t;i<valid;i+=256)prob[uint64_t(h)*valid+i]=expf(scores[uint64_t(h)*valid+i]-m)*inv;
}
__global__ void values(const float*prob,const __half*v,float*out,unsigned valid,unsigned width){
 const unsigned h=blockIdx.x,base=blockIdx.y*4,t=threadIdx.x,lane=t&31,warp=t>>5;float a[4]={};__shared__ float partial[8][4];
 for(unsigned i=t;i<valid;i+=256){float p=prob[uint64_t(h)*valid+i];
 #pragma unroll
 for(unsigned d=0;d<4;d++)a[d]+=p*__half2float(v[uint64_t(i)*width+(h/4)*64+base+d]);}
 #pragma unroll
 for(unsigned d=0;d<4;d++){a[d]=sum_warp(a[d]);if(lane==0)partial[warp][d]=a[d];}
 __syncthreads();if(t<4){float s=partial[0][t];
 #pragma unroll
 for(unsigned w=1;w<8;w++)s+=partial[w][t];out[h*64+base+t]=s;}
}
}
