#pragma once
#include "attention_decode_mma_d512_pipeline_q4.cuh"
namespace imparo_sm86_d512_m3_shared {
using namespace imparo_sm80_d512_decode;
using imparo_sm86_d512_pipeline::prefetch_packed_tile;
__global__ __launch_bounds__(128,2) void partial(const __half*q,const uint8_t*kc,const uint8_t*vc,float*workspace,uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,uint32_t window,uint32_t ring,uint32_t valid_span,uint32_t schedule_groups,uint32_t physical_blocks,float*workspace_single){
#define IMPARO_D512_Q4 1
#define IMPARO_D512_PACKED_PIPELINE 1
#include "attention_decode_mma_d512_m3_shared_q4_body.inc"
#undef IMPARO_D512_PACKED_PIPELINE
#undef IMPARO_D512_Q4
}
}
