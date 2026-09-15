#pragma once
#include "attention_decode_mma_d512_pair_q4.cuh"

// Opt-in SM86 linear Q4 KV pipeline. Shared numerical bodies preserve M1/pair contracts.
namespace imparo_sm86_d512_pipeline {
using namespace imparo_sm80_d512_decode;
// Two committed groups per tile (K and V); each group has nine copies/thread.
// A thread waits for its own copies, then the warp makes all lane copies visible.
__device__ __forceinline__ void prefetch_packed_tile(
 const uint8_t*kc,const uint8_t*vc,uint8_t*kd,uint8_t*vd,
 unsigned group,unsigned warp,unsigned lane,unsigned kvh,unsigned width,unsigned valid_span){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >=800
 #pragma unroll
 for(unsigned sector=lane;sector<16*18;sector+=32){
  unsigned row=sector/18,off=(sector%18)*16,key=group*32+warp*16+row;
  unsigned bytes=key<valid_span?16u:0u;
  const uint8_t*src=key<valid_span?kc+uint64_t(key)*(width/32)*18+kvh*288+off:kc;
  unsigned dst=unsigned(__cvta_generic_to_shared(kd+row*288+off));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(bytes));
 }
 asm volatile("cp.async.commit_group;");
 #pragma unroll
 for(unsigned sector=lane;sector<16*18;sector+=32){
  unsigned row=sector/18,off=(sector%18)*16,key=group*32+warp*16+row;
  unsigned bytes=key<valid_span?16u:0u;
  const uint8_t*src=key<valid_span?vc+uint64_t(key)*(width/32)*18+kvh*288+off:vc;
  unsigned dst=unsigned(__cvta_generic_to_shared(vd+row*288+off));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(bytes));
 }
 asm volatile("cp.async.commit_group;");
#endif
}

__global__ __launch_bounds__(kThreads,4) void partial_q4(const __half*q,const uint8_t*kc,const uint8_t*vc,float*workspace,uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,uint32_t window,uint32_t ring,uint32_t valid_span,uint32_t schedule_groups,uint32_t physical_blocks){
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PACKED_PIPELINE 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PACKED_PIPELINE
#undef IMPARO_D512_Q4
}
// Assistant-only FP32 PV specialization; original QK/softmax/load/schedule body is shared.
__global__ __launch_bounds__(kThreads,4) void partial_q4_pvf32(const __half*q,const uint8_t*kc,const uint8_t*vc,float*workspace,uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,uint32_t window,uint32_t ring,uint32_t valid_span,uint32_t schedule_groups,uint32_t physical_blocks){
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PACKED_PIPELINE 1
#define IMPARO_D512_PV_F32 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PV_F32
#undef IMPARO_D512_PACKED_PIPELINE
#undef IMPARO_D512_Q4
}
__global__ __launch_bounds__(kThreads,4) void partial_q4_controlled(const __half*q,const uint8_t*kc,const uint8_t*vc,float*workspace,uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,uint32_t window,uint32_t ring,uint32_t valid_span,uint32_t schedule_groups,uint32_t physical_blocks,const uint32_t*decode_control){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 if(decode_control){start_pos=decode_control[0];valid_span=min(start_pos+1,valid_span);}
#endif
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PACKED_PIPELINE 1
#include "../sm80/attention_decode_mma_d512_f16_body.inc"
#undef IMPARO_D512_PACKED_PIPELINE
#undef IMPARO_D512_Q4
}
__global__ __launch_bounds__(kThreads,4) void partial_q4_pair(const __half*q,const uint8_t*kc,const uint8_t*vc,float*workspace,uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,uint32_t window,uint32_t ring,uint32_t valid_span,uint32_t schedule_groups,uint32_t physical_blocks){
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PACKED_PIPELINE 1
#include "attention_decode_mma_d512_pair_q4_body.inc"
#undef IMPARO_D512_PACKED_PIPELINE
#undef IMPARO_D512_Q4
}
} // namespace imparo_sm86_d512_pipeline
