#pragma once
#include "../sm80/attention_decode_vec_d256.cuh"
namespace imparo_sm80_d256_vec {
// Packed K load/unpack is shared across three independent Q8 dot products.
__device__ __forceinline__ void dot_q4_q8_m3(const uint8_t* row,
 const int qi[3][kGqaHeads][64],const float2 ds[3][kGqaHeads][kQ4BlocksPerHead],
 unsigned warp,unsigned lane,float result[3]) {
 float sum[3]={};
 #pragma unroll
 for(unsigned base=0;base<64;base+=32){unsigned k=base+lane,bi=k/8,iqs=k&3,shift=k&4;
  const uint8_t*block=row+uint64_t(bi)*18;int packed=(load_q4_int(block,iqs)>>shift)&0x0f0f0f0f;
  float d4=__half2float(*reinterpret_cast<const __half*>(block));
  #pragma unroll
  for(unsigned query=0;query<3;query++){int sumi=__dp4a(packed,qi[query][warp][k],0);float2 scale=ds[query][warp][bi];sum[query]+=d4*(float(sumi)*scale.x-scale.y);}
 }
 #pragma unroll
 for(unsigned query=0;query<3;query++)result[query]=warp_sum(sum[query]);
}
__global__ __launch_bounds__(kThreads,1) void partial_q4_gqa4_m3(
 const float*q,const uint8_t*kc,const uint8_t*vc,float*children,
 uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,
 uint32_t ring,uint32_t schedule_span,float qk_scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t source_warp = blockIdx.x & (kWarps - 1);
    const uint32_t kvh = blockIdx.x / kWarps;
    const uint32_t partial = blockIdx.y;
    const uint32_t partial_count = gridDim.y;
    if (start_pos < ring || start_pos > UINT32_MAX - 2 || n_kv == 0 || n_heads != n_kv * kGqaHeads || kvh >= n_kv
        || partial >= partial_count || partial_count == 0
        || partial_count > kMaxPartials
        || schedule_span != partial_count * kScheduleQuantum
        || ring == UINT32_MAX || schedule_span > ring + 1) return;
    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t head = kvh * kGqaHeads + warp;
    const uint32_t q4_blocks_per_row = kv_width / 32;

 unsigned window_lo[3];
 #pragma unroll
 for(unsigned query=0;query<3;query++)window_lo[query]=start_pos+query+1-kWindowSpan;
 const unsigned valid_span=ring+1;
 __shared__ int q_i32[3][kGqaHeads][64];
 __shared__ float2 q_ds[3][kGqaHeads][kQ4BlocksPerHead];
 __shared__ float probabilities[3][kGqaHeads][32];
 __shared__ uint4 kv_tile[32*9];
 #pragma unroll
 for(unsigned query=0;query<3;query++){
 const float*qr=q+(uint64_t(query)*n_heads+head)*kHeadDim;
#pragma unroll
    for (uint32_t base = 0; base < 64; base += 32) {
        const float4 raw = reinterpret_cast<const float4 *>(qr)[base + lane];
        const float4 value = make_float4(
            qk_scale * raw.x, qk_scale * raw.y,
            qk_scale * raw.z, qk_scale * raw.w);
        float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                            fmaxf(fabsf(value.z), fabsf(value.w)));
        float sum = value.x + value.y + value.z + value.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffffu, amax, offset));
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        const float d = amax / 127.0f;
        const int8_t q0 = d != 0.0f ? int8_t(roundf(value.x / d)) : 0;
        const int8_t q1 = d != 0.0f ? int8_t(roundf(value.y / d)) : 0;
        const int8_t q2 = d != 0.0f ? int8_t(roundf(value.z / d)) : 0;
        const int8_t q3 = d != 0.0f ? int8_t(roundf(value.w / d)) : 0;
        q_i32[query][warp][base + lane] = int(uint8_t(q0))
            | (int(uint8_t(q1)) << 8) | (int(uint8_t(q2)) << 16)
            | (int(uint8_t(q3)) << 24);
        if ((lane & 7) == 0) {
            q_ds[query][warp][base / 8 + lane / 8] = make_float2(d, sum);
        }
    }

 }
 __syncthreads();
 float2 numerator[3][4]={};
 float running_max[3]={kNegInf*.5f,kNegInf*.5f,kNegInf*.5f};
 float running_sum[3]={};
 for(unsigned stripe=0;stripe<kStripesPerPartial;stripe++){
  unsigned tile=imparo_sm80_d256_vec_plan::partial_stripe_begin(partial,stripe,partial_count);
  unsigned first_slot=tile+source_warp*32;
        // Stage K cooperatively. Physical ring rows are interleaved by KV head,
        // so copy each 144-byte head row independently rather than assuming
        // that 32 rows are contiguous.
        for (uint32_t item = threadIdx.y * 32 + lane;
             item < 32 * 9; item += kThreads) {
            const uint32_t row = item / 9;
            const uint32_t chunk = item - row * 9;
            const uint8_t * src = kc
                + (uint64_t(first_slot + row) * q4_blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18
                + uint64_t(chunk) * sizeof(uint4);
            kv_tile[item] = *reinterpret_cast<const uint4 *>(src);
        }
        __syncthreads();


  float score_owned[3]={kNegInf,kNegInf,kNegInf};
  float next_max[3]={running_max[0],running_max[1],running_max[2]};
  #pragma unroll
  for(unsigned row=0;row<32;row++){
   unsigned slot=first_slot+row;bool valid[3];bool any=false;
   #pragma unroll
   for(unsigned query=0;query<3;query++){unsigned pos=start_pos+query;unsigned logical=slot<valid_span?slot+((pos-slot)&~ring):0;valid[query]=slot<valid_span&&logical>=window_lo[query]&&logical<=pos;any|=valid[query];}
   float scores[3]={};
   if(any)dot_q4_q8_m3(reinterpret_cast<const uint8_t*>(kv_tile+row*9),q_i32,q_ds,warp,lane,scores);
   #pragma unroll
   for(unsigned query=0;query<3;query++){float score=valid[query]?scores[query]:kNegInf;next_max[query]=fmaxf(next_max[query],score+kMaxOffset);if(lane==row)score_owned[query]=score;}
  }
  #pragma unroll
  for(unsigned query=0;query<3;query++){
   float rescale=expf(running_max[query]-next_max[query]);
   #pragma unroll
   for(unsigned i=0;i<4;i++){numerator[query][i].x*=rescale;numerator[query][i].y*=rescale;}
   running_max[query]=next_max[query];
   float probability=score_owned[query]>-3.0e38F?expf(score_owned[query]-running_max[query]):0.f;
   running_sum[query]=running_sum[query]*rescale+probability;probabilities[query][warp][lane]=probability;
  }
  __syncthreads();
        for (uint32_t item = threadIdx.y * 32 + lane;
             item < 32 * 9; item += kThreads) {
            const uint32_t row = item / 9;
            const uint32_t chunk = item - row * 9;
            const uint8_t * src = vc
                + (uint64_t(first_slot + row) * q4_blocks_per_row
                    + uint64_t(kvh) * kQ4BlocksPerHead) * 18
                + uint64_t(chunk) * sizeof(uint4);
            kv_tile[item] = *reinterpret_cast<const uint4 *>(src);
        }
        __syncthreads();


  #pragma unroll
  for(unsigned row=0;row<32;row++){
   unsigned slot=first_slot+row;bool active[3];bool any=false;float p[3];
   #pragma unroll
   for(unsigned query=0;query<3;query++){unsigned pos=start_pos+query;unsigned logical=slot<valid_span?slot+((pos-slot)&~ring):0;p[query]=probabilities[query][warp][row];active[query]=slot<valid_span&&logical>=window_lo[query]&&logical<=pos&&p[query]!=0.f;any|=active[query];}
   if(any){const uint8_t*vrow=reinterpret_cast<const uint8_t*>(kv_tile+row*9);
    float2 v0=q4_float2(vrow,4*lane),v1=q4_float2(vrow,4*lane+2),v2=q4_float2(vrow,128+4*lane),v3=q4_float2(vrow,128+4*lane+2);
    #pragma unroll
    for(unsigned query=0;query<3;query++)if(active[query]){
     numerator[query][0].x+=p[query]*v0.x;numerator[query][0].y+=p[query]*v0.y;
     numerator[query][1].x+=p[query]*v1.x;numerator[query][1].y+=p[query]*v1.y;
     numerator[query][2].x+=p[query]*v2.x;numerator[query][2].y+=p[query]*v2.y;
     numerator[query][3].x+=p[query]*v3.x;numerator[query][3].y+=p[query]*v3.y;
    }
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(unsigned query=0;query<3;query++){
    float * dst = children + uint64_t(query) * n_heads * kMaxPartials * kWarps * kGqaChildStride
        + (((uint64_t(head) * kMaxPartials + partial) * kWarps
            + source_warp) * kGqaChildStride);
    dst[4 * lane] = numerator[query][0].x;
    dst[4 * lane + 1] = numerator[query][0].y;
    dst[4 * lane + 2] = numerator[query][1].x;
    dst[4 * lane + 3] = numerator[query][1].y;
    dst[128 + 4 * lane] = numerator[query][2].x;
    dst[128 + 4 * lane + 1] = numerator[query][2].y;
    dst[128 + 4 * lane + 2] = numerator[query][3].x;
    dst[128 + 4 * lane + 3] = numerator[query][3].y;
    dst[kHeadDim + 2 + lane] = running_sum[query];
    if (lane == 0) {
        dst[kHeadDim] = running_max[query];
        dst[kHeadDim + 1] = 0.0f;
    }

 }
#endif
}
} // namespace
