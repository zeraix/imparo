#pragma once
#include "../sm80/attention_decode_vec_d256.cuh"
namespace imparo_sm80_d256_vec {
// Packed K load/unpack is shared across independent Q8 dot products.
// Each query keeps its own initialized prefix and causal window. Slots outside
// the producer prefix are zero-staged; no uninitialized KV row is read.
template<unsigned Queries>
__device__ __forceinline__ void dot_q4_q8_queries(const uint8_t* row,
 const int qi[Queries][kGqaHeads][64],const float2 ds[Queries][kGqaHeads][kQ4BlocksPerHead],
 unsigned warp,unsigned lane,float result[Queries]) {
 float sum[Queries]={};
 #pragma unroll
 for(unsigned base=0;base<64;base+=32){unsigned k=base+lane,bi=k/8,iqs=k&3,shift=k&4;
  const uint8_t*block=row+uint64_t(bi)*18;int packed=(load_q4_int(block,iqs)>>shift)&0x0f0f0f0f;
  float d4=__half2float(*reinterpret_cast<const __half*>(block));
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){int sumi=__dp4a(packed,qi[query][warp][k],0);float2 scale=ds[query][warp][bi];sum[query]+=d4*(float(sumi)*scale.x-scale.y);}
 }
 #pragma unroll
 for(unsigned query=0;query<Queries;query++)result[query]=warp_sum(sum[query]);
}

// The hot prefix has no RowLayout or global-tail pointers. Only a tile containing
// the tiny speculative tail takes the non-unrolled specialization. Both use the
// same Q8/Q4 arithmetic and each key's original M1 slot/warp/partition.
template<unsigned Queries, bool Tail>
__device__ __forceinline__ void d256_scores_tile(
 const uint4*kv_tile,const int qi[Queries][kGqaHeads][64],
 const float2 ds[Queries][kGqaHeads][kQ4BlocksPerHead],
 const unsigned positions[Queries],const unsigned window_lo[Queries],
 const float tail_scores[Queries][Queries][kGqaHeads],
 unsigned first_slot,unsigned start_pos,unsigned ring,unsigned warp,unsigned lane,
 float score_owned[Queries],float next_max[Queries]) {
 #pragma unroll (Tail ? 1 : 32)
 for(unsigned row=0;row<32;row++){
  const unsigned slot=first_slot+row;bool valid[Queries];bool any=false;
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){
   const unsigned pos=positions[query],valid_span=min(pos+1,ring+1);
   const unsigned logical=slot<valid_span?slot+((pos-slot)&~ring):0;
   valid[query]=slot<valid_span&&logical>=window_lo[query]&&logical<=pos;
   any|=valid[query];
  }
  float scores[Queries]={};
  if(any){
   if constexpr(Tail){
    const unsigned d=(slot-start_pos)&ring;
    if(d<Queries){
     #pragma unroll
     for(unsigned query=0;query<Queries;query++)if(valid[query])scores[query]=tail_scores[query][d][warp];
    }else dot_q4_q8_queries<Queries>(reinterpret_cast<const uint8_t*>(kv_tile+row*9),qi,ds,warp,lane,scores);
   }else dot_q4_q8_queries<Queries>(reinterpret_cast<const uint8_t*>(kv_tile+row*9),qi,ds,warp,lane,scores);
  }
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){
   const float score=valid[query]?scores[query]:kNegInf;
   next_max[query]=fmaxf(next_max[query],score+kMaxOffset);
   if(lane==row)score_owned[query]=score;
  }
 }
}
template<unsigned Queries, bool Tail>
__device__ __forceinline__ void d256_values_tile(
 const uint4*kv_tile,const float probabilities[Queries][kGqaHeads][32],
 const unsigned positions[Queries],const unsigned window_lo[Queries],
 const unsigned tail_nodes[Queries][Queries],const float tail_values[Queries][kHeadDim],
 unsigned first_slot,unsigned start_pos,unsigned ring,unsigned warp,unsigned lane,
 float2 numerator[Queries][4]) {
 #pragma unroll (Tail ? 1 : 32)
 for(unsigned row=0;row<32;row++){
  const unsigned slot=first_slot+row;bool active[Queries];bool any=false;float p[Queries];
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){
   const unsigned pos=positions[query],valid_span=min(pos+1,ring+1);
   const unsigned logical=slot<valid_span?slot+((pos-slot)&~ring):0;
   p[query]=probabilities[query][warp][row];
   active[query]=slot<valid_span&&logical>=window_lo[query]&&logical<=pos&&p[query]!=0.f;
   any|=active[query];
  }
  if constexpr(Tail) if(any&&((slot-start_pos)&ring)<Queries){
   const unsigned d=(slot-start_pos)&ring;
   #pragma unroll
   for(unsigned query=0;query<Queries;query++)if(active[query]){
    const float* v=tail_values[tail_nodes[query][d]];
    float2 v0=make_float2(v[4*lane],v[4*lane+1]),v1=make_float2(v[4*lane+2],v[4*lane+3]);
    float2 v2=make_float2(v[128+4*lane],v[128+4*lane+1]),v3=make_float2(v[128+4*lane+2],v[128+4*lane+3]);
    numerator[query][0].x+=p[query]*v0.x;numerator[query][0].y+=p[query]*v0.y;
    numerator[query][1].x+=p[query]*v1.x;numerator[query][1].y+=p[query]*v1.y;
    numerator[query][2].x+=p[query]*v2.x;numerator[query][2].y+=p[query]*v2.y;
    numerator[query][3].x+=p[query]*v3.x;numerator[query][3].y+=p[query]*v3.y;
   }
   any=false;
  }
  if(any){
   const uint8_t*vrow=reinterpret_cast<const uint8_t*>(kv_tile+row*9);
   float2 v0=q4_float2(vrow,4*lane),v1=q4_float2(vrow,4*lane+2),v2=q4_float2(vrow,128+4*lane),v3=q4_float2(vrow,128+4*lane+2);
   #pragma unroll
   for(unsigned query=0;query<Queries;query++)if(active[query]){
    numerator[query][0].x+=p[query]*v0.x;numerator[query][0].y+=p[query]*v0.y;
    numerator[query][1].x+=p[query]*v1.x;numerator[query][1].y+=p[query]*v1.y;
    numerator[query][2].x+=p[query]*v2.x;numerator[query][2].y+=p[query]*v2.y;
    numerator[query][3].x+=p[query]*v3.x;numerator[query][3].y+=p[query]*v3.y;
   }
  }
 }
}

template<unsigned Queries, bool Tree>
__device__ __forceinline__ void partial_q4_gqa4_body(
 const float*q,const uint8_t*kc,const uint8_t*vc,float*children,
 uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,
 uint32_t ring,uint32_t schedule_span,float qk_scale,const uint32_t* layout) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t source_warp = blockIdx.x & (kWarps - 1);
    const uint32_t kvh = blockIdx.x / kWarps;
    const uint32_t partial = blockIdx.y;
    const uint32_t partial_count = gridDim.y;
    if (start_pos < kWindowSpan - 1 || start_pos > UINT32_MAX - Queries || n_kv == 0 || n_heads != n_kv * kGqaHeads || kvh >= n_kv
        || partial >= partial_count || partial_count == 0
        || partial_count > kMaxPartials
        || schedule_span != partial_count * kScheduleQuantum
        || ring == UINT32_MAX || schedule_span > ring + 1) return;
    const uint32_t warp = threadIdx.y;
    const uint32_t lane = threadIdx.x;
    const uint32_t head = kvh * kGqaHeads + warp;
    const uint32_t q4_blocks_per_row = kv_width / 32;

 // BufId::RowLayout's existing 12-word contract: position, depth, two
 // visibility words, then the nearest eight ancestors. No second topology.
 if constexpr(Tree) { if(!layout) return; }
 unsigned positions[Queries],window_lo[Queries];
 #pragma unroll
 for(unsigned query=0;query<Queries;query++){
  positions[query]=Tree?layout[query*12]:start_pos+query;
  window_lo[query]=positions[query]+1-kWindowSpan;
 }
 const unsigned loaded_span=min(start_pos+Queries,ring+1);
 __shared__ int q_i32[Queries][kGqaHeads][64];
 __shared__ float2 q_ds[Queries][kGqaHeads][kQ4BlocksPerHead];
 __shared__ float probabilities[Queries][kGqaHeads][32];
 __shared__ uint4 kv_tile[32*9];
 // The tree-only shared arrays disappear in the old M3 specialization.
 __shared__ unsigned tree_tail_nodes[Queries][Queries];
 __shared__ float tree_tail_scores[Queries][Queries][kGqaHeads];
 __shared__ float tree_tail_values[Queries][kHeadDim];
 if constexpr(Tree) if(threadIdx.x==0&&threadIdx.y==0){
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){
   const uint32_t* r=layout+query*12;
   #pragma unroll
   for(unsigned d=0;d<Queries;d++){
    const unsigned delta=d<=r[1]?r[1]-d:0;
    tree_tail_nodes[query][d]=d<=r[1]?(delta?r[4+delta-1]:query):0;
   }
  }
 }
 #pragma unroll
 for(unsigned query=0;query<Queries;query++){
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
 if constexpr(Tree){
  // Load/decode the packed tail once per CTA, before any prefix stripe.
  // These are the existing single-query dot and Q4-to-float operations.
  #pragma unroll
  for(unsigned node=0;node<Queries;node++){
   const uint8_t* value=vc+(uint64_t((start_pos+node)&ring)*q4_blocks_per_row+uint64_t(kvh)*kQ4BlocksPerHead)*18;
   const unsigned i=2*(warp*32+lane);
   const float2 v=q4_float2(value,i);
   tree_tail_values[node][i]=v.x;tree_tail_values[node][i+1]=v.y;
  }
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){
   #pragma unroll
   for(unsigned d=0;d<Queries;d++)if(d<=positions[query]-start_pos){
    const unsigned node=tree_tail_nodes[query][d];
    const uint8_t* key=kc+(uint64_t((start_pos+node)&ring)*q4_blocks_per_row+uint64_t(kvh)*kQ4BlocksPerHead)*18;
    const float score=dot_q4_q8_1_d256(key,q_i32[query][warp],q_ds[query][warp],lane);
    if(lane==0)tree_tail_scores[query][d][warp]=score;
   }
  }
  __syncthreads();
 }
 float2 numerator[Queries][4]={};
 float running_max[Queries];
 #pragma unroll
 for(unsigned query=0;query<Queries;query++)running_max[query]=kNegInf*.5f;
 float running_sum[Queries]={};
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
            kv_tile[item] = first_slot + row < loaded_span ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0,0,0,0);
        }
        __syncthreads();


  float score_owned[Queries],next_max[Queries];
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){score_owned[query]=kNegInf;next_max[query]=running_max[query];}
  // This tile intersects [start, start+Queries) in the circular M1 slot
  // order. Prefix-only tiles retain a fully static, shared load/compute loop.
  const bool tail_tile=Tree&&(((start_pos-first_slot)&ring)<32||((first_slot-start_pos)&ring)<Queries);
  if constexpr(Tree){
   if(tail_tile)d256_scores_tile<Queries,true>(kv_tile,q_i32,q_ds,positions,window_lo,tree_tail_scores,first_slot,start_pos,ring,warp,lane,score_owned,next_max);
   else d256_scores_tile<Queries,false>(kv_tile,q_i32,q_ds,positions,window_lo,tree_tail_scores,first_slot,start_pos,ring,warp,lane,score_owned,next_max);
  }else d256_scores_tile<Queries,false>(kv_tile,q_i32,q_ds,positions,window_lo,tree_tail_scores,first_slot,start_pos,ring,warp,lane,score_owned,next_max);
  #pragma unroll
  for(unsigned query=0;query<Queries;query++){
   float rescale=expf(running_max[query]-next_max[query]);
   #pragma unroll
   for(unsigned i=0;i<4;i++){numerator[query][i].x*=rescale;numerator[query][i].y*=rescale;}
   running_max[query]=next_max[query];
   float probability=score_owned[query]>-3.0e38F?expf(score_owned[query]-running_max[query]):0.f;
   // P3 must preserve partial_q4's separately rounded multiply and add.
   // Keep the admitted saturated P4 arithmetic unchanged.
   running_sum[query]=schedule_span==3*kScheduleQuantum
       ? __fadd_rn(__fmul_rn(running_sum[query],rescale),probability)
       : running_sum[query]*rescale+probability;probabilities[query][warp][lane]=probability;
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
            kv_tile[item] = first_slot + row < loaded_span ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0,0,0,0);
        }
        __syncthreads();


  if constexpr(Tree){
   if(tail_tile)d256_values_tile<Queries,true>(kv_tile,probabilities,positions,window_lo,tree_tail_nodes,tree_tail_values,first_slot,start_pos,ring,warp,lane,numerator);
   else d256_values_tile<Queries,false>(kv_tile,probabilities,positions,window_lo,tree_tail_nodes,tree_tail_values,first_slot,start_pos,ring,warp,lane,numerator);
  }else d256_values_tile<Queries,false>(kv_tile,probabilities,positions,window_lo,tree_tail_nodes,tree_tail_values,first_slot,start_pos,ring,warp,lane,numerator);
  __syncthreads();
 }
 #pragma unroll
 for(unsigned query=0;query<Queries;query++){
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
// Keep the production global symbol and its 11-argument Graph ABI unchanged.
__global__ __launch_bounds__(kThreads,1) void partial_q4_gqa4_m3(
 const float*q,const uint8_t*kc,const uint8_t*vc,float*children,
 uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,
 uint32_t ring,uint32_t schedule_span,float qk_scale) {
 partial_q4_gqa4_body<3,false>(q,kc,vc,children,n_heads,n_kv,kv_width,
     start_pos,ring,schedule_span,qk_scale,nullptr);
}
// Laboratory entry only: four validated RowLayout rows; no runtime dispatch.
// All rows use the same admitted P3/P4 partition schedule as their M1 paths.
__global__ __launch_bounds__(kThreads,1) void partial_q4_gqa4_tree4(
 const float*q,const uint8_t*kc,const uint8_t*vc,float*children,
 uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,
 uint32_t ring,uint32_t schedule_span,float qk_scale,const uint32_t*layout) {
 partial_q4_gqa4_body<4,true>(q,kc,vc,children,n_heads,n_kv,kv_width,
     start_pos,ring,schedule_span,qk_scale,layout);
}
} // namespace
