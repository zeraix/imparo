#pragma once
// Include after BlockQ8_1, warp_sum_xor and sm80/mmvq_q8_q8_1.cuh.
// Isolated reusable mathematical body. No owner/state/stream allocation.
namespace imparo_efficient_embedder {
constexpr unsigned H=256,VOC=262144,NC=2048,TOP=32,CS=128,SEL=TOP*CS;
// FP32 centroid projection. One warp per cluster; BF16 source has been
// expanded losslessly by fixture preparation. Does not quantize centroid scores.
__global__ void centroids(const float*c,const float*x,float*out) {
 unsigned row=blockIdx.x*4+threadIdx.y,lane=threadIdx.x;float s=0.f;
#pragma unroll
 for(unsigned i=lane;i<H;i+=32)s=fmaf(c[row*H+i],x[i],s);
 s=warp_sum_xor(s);if(lane==0)out[row]=s;
}
__device__ __forceinline__ bool better(float a,unsigned ia,float b,unsigned ib) {
 return a>b || (a==b && ia<ib);
}
__device__ __forceinline__ void warp_best(float&v,unsigned&i) {
#pragma unroll
 for(int o=16;o;o>>=1){float b=__shfl_down_sync(0xffffffff,v,o);unsigned j=__shfl_down_sync(0xffffffff,i,o);
 if((threadIdx.x&31)+o<32 && better(b,j,v,i)){v=b;i=j;}}
}
// Fixed top32, no CPU round trips. Each thread owns two clusters.
// Every iteration overwrites shared storage before consumption.
__global__ void top32(const float*s,unsigned*out) {
 __shared__ float val[32];__shared__ unsigned idx[32],selected;
 unsigned t=threadIdx.x;float a=s[t],b=s[t+1024];unsigned ai=t,bi=t+1024;
 for(unsigned rank=0;rank<TOP;++rank){
  float v=a;unsigned i=ai;if(better(b,bi,v,i)){v=b;i=bi;}warp_best(v,i);
  if((t&31)==0){val[t/32]=v;idx[t/32]=i;}__syncthreads();
  if(t<32){v=val[t];i=idx[t];warp_best(v,i);if(t==0){selected=i;out[rank]=i;}}
  __syncthreads();if(ai==selected)a=-INFINITY;if(bi==selected)b=-INFINITY;
 }
}
// Same four-rows/CTA shortK numerical body as native, only row address differs.
// ordering maps ordered cluster rows -> canonical vocabulary IDs.
__global__ void selected_head(const uint8_t*w,const BlockQ8_1*x,
 const unsigned*order,const unsigned*clusters,float*y,unsigned*ids) {
 unsigned lane=threadIdx.x,slot=blockIdx.x*4+threadIdx.y;
 unsigned row=order[clusters[slot/CS]*CS+slot%CS],block=lane/4,iqs=2*(lane&3);
 const uint8_t*q=w+(uint64_t(row)*8+block)*34;float partial=0.f;
 partial+=imparo_sm80_q8_mmvq::dot_q8_0_q8_1_half(q+2,*reinterpret_cast<const __half*>(q),x+block,iqs);
 partial=__fadd_rn(partial,0.f);partial=__fadd_rn(partial,0.f);partial=__fadd_rn(partial,0.f);
 float sum=warp_sum_xor(partial);if(lane==0){y[slot]=sum;ids[slot]=row;}
}
__global__ void argmax(const float*x,const unsigned*ids,unsigned n,unsigned*out) {
 __shared__ float val[32];__shared__ unsigned id[32];unsigned t=threadIdx.x,i=~0u;float v=-INFINITY;
 for(unsigned j=t;j<n;j+=1024){unsigned k=ids?ids[j]:j;if(better(x[j],k,v,i)){v=x[j];i=k;}}
 warp_best(v,i);if((t&31)==0){val[t/32]=v;id[t/32]=i;}__syncthreads();
 if(t<32){v=val[t];i=id[t];warp_best(v,i);if(t==0)*out=i;}
}
// Sparse confidence includes the official constant mask on unselected tokens.
// Unlike dense confidence this reduction scans only the 4096 retained logits.
__global__ void sparse_probability(const float*logits,const unsigned*ids,
 const unsigned*token,float*out) {
 __shared__ float best,mins[1024],sums[1024];
 float low=INFINITY;
 for(unsigned i=threadIdx.x;i<SEL;i+=1024){
  low=fminf(low,logits[i]);if(ids[i]==*token)best=logits[i];
 }
 mins[threadIdx.x]=low;__syncthreads();
 for(unsigned k=512;k;k>>=1){if(threadIdx.x<k)mins[threadIdx.x]=fminf(mins[threadIdx.x],mins[threadIdx.x+k]);__syncthreads();}
 float sum=0.f;for(unsigned i=threadIdx.x;i<SEL;i+=1024)sum+=__expf(logits[i]-best);
 sums[threadIdx.x]=sum;__syncthreads();
 for(unsigned k=512;k;k>>=1){if(threadIdx.x<k)sums[threadIdx.x]+=sums[threadIdx.x+k];__syncthreads();}
 if(threadIdx.x==0)*out=1.f/(sums[0]+float(VOC-SEL)*__expf((mins[0]-1.f)-best));
}

} // namespace imparo_efficient_embedder
