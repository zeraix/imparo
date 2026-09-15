#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <climits>
#include <cmath>
namespace imparo_gpu_greedy_return {
template<unsigned Threads>
__global__ void row_select(const float* logits,uint32_t* work,uint32_t vocab,uint32_t rows) {
 static_assert(Threads==1024,"qualified row reduction");
 const unsigned row=blockIdx.x,t=threadIdx.x;
 if(row>=rows)return;
 __shared__ uint32_t best[Threads],index[Threads],bad[Threads];
 uint32_t value=0,ix=UINT_MAX,invalid=0;
 for(uint32_t j=t;j<vocab;j+=Threads){
  const uint32_t bits=__float_as_uint(logits[uint64_t(row)*vocab+j]);
  const uint32_t magnitude=bits&0x7fffffffu;
  if(magnitude>=0x7f800000u){invalid=1;continue;}
  // Integer ordering preserves every finite IEEE f32 value even with FTZ enabled.
  // Both signed zeros compare equal; ties select the first original index.
  const uint32_t key=magnitude==0?0x80000000u:((bits&0x80000000u)?~bits:(bits|0x80000000u));
  if(key>value||(key==value&&j<ix)){value=key;ix=j;}
 }
 best[t]=value;index[t]=ix;bad[t]=invalid;__syncthreads();
 for(unsigned step=Threads/2;step;step>>=1){
  if(t<step){
   const uint32_t value2=best[t+step],j=index[t+step];
   if(value2>best[t]||(value2==best[t]&&j<index[t])){best[t]=value2;index[t]=j;}
   bad[t]|=bad[t+step];
  }
  __syncthreads();
 }
 if(t==0){work[2*row]=index[0];work[2*row+1]=bad[0]|uint32_t(index[0]==UINT_MAX);}
}

__global__ void finish_restore(const uint32_t* work,const uint32_t* tokens,uint32_t* out3,
 float* recur,const float* snap,uint32_t rows,uint32_t elems) {
 __shared__ uint32_t status,consumed,next;
 if(threadIdx.x==0){
  status=rows==0;consumed=0;next=0;
  for(uint32_t i=0;i<rows;++i)status|=work[2*i+1]!=0;
  if(!status){
   for(uint32_t i=0;i<rows;++i){
    const uint32_t pick=work[2*i];
    if(i+1==rows||pick!=tokens[i+1]){consumed=i+1;next=pick;break;}
   }
  }
  if(blockIdx.x==0){out3[0]=status?1:0;out3[1]=status?0:consumed;out3[2]=status?0:next;}
 }
 __syncthreads();
 if(!status&&consumed<rows){
  const uint64_t off=uint64_t(consumed+1)*elems;
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<elems;i+=blockDim.x*gridDim.x)recur[i]=snap[off+i];
 }
}
}
