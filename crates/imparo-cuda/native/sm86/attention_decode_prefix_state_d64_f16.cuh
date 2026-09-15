#pragma once
// Isolated laboratory candidate: exact keytile16 QK plus prefix-max scalar pairs.
namespace imparo_sm86_prefix_d64 {
__global__ void k_attention_direct(const float *q,const void *kc,const void *vc,float *out,
 uint32_t head_dim,uint32_t n_heads,uint32_t n_kv,uint32_t kv_width,uint32_t start_pos,
 float qk_scale,uint32_t window,uint32_t ring,uint32_t n_tok,uint32_t ktype,uint32_t vtype,
 uint32_t f32_v_accum,const uint32_t *page_table) {
 const unsigned h=blockIdx.x,tid=threadIdx.x,kvh=h/4,valid=start_pos+1;
 // Caller contract: D64/GQA4/single query/F16 staged KV/f32 PV, no window/ring/pages.
 __shared__ float m_shared,s_shared,alpha[16],prob[16];
 if(tid==0){m_shared=-1e30f;s_shared=0.f;}
 __syncthreads();
 float acc=0.f,denom=0.f;
 imparo_sm80_mma::Half16x8 qfrag[4];
 if(tid<32){
  #pragma unroll
  for(unsigned g=0;g<4;g++)qfrag[g]=imparo_sm86_keytile_d64::query_direct(q+h*64,g*16,qk_scale,h%4,tid);
 }
 for(unsigned base=0;base<valid;base+=16){
  if(tid<32){
   const auto frag=imparo_sm86_keytile_d64::dot_keytile16(qfrag,static_cast<const __half*>(kc),kv_width,kvh*64,base,valid,tid);
   const unsigned row=tid&15,owner=(h%4)*4+(row%8)/2;
   const float a0=__shfl_sync(0xffffffffu,frag.x[0],owner);
   const float a1=__shfl_sync(0xffffffffu,frag.x[1],owner);
   const float a4=__shfl_sync(0xffffffffu,frag.x[4],owner);
   const float a5=__shfl_sync(0xffffffffu,frag.x[5],owner);
   float score=row<8?((row&1)?a1:a0):((row&1)?a5:a4);
   if(tid>=16||base+tid>=valid)score=-1e30f;
   constexpr float offset=3.f*.6931f;
   const float oldm=m_shared;
   float m=fmaxf(oldm,score+offset);
   #pragma unroll
   for(unsigned step=1;step<=8;step*=2){float other=__shfl_up_sync(0xffffffffu,m,step);if(tid>=step)m=fmaxf(m,other);}
   float prev=__shfl_up_sync(0xffffffffu,m,1);if(tid==0)prev=oldm;
   const float a=expf(prev-m),p=expf(score-m);
   if(tid<16){alpha[tid]=a;prob[tid]=p;}
   if(tid==15)m_shared=m;
  }
  __syncthreads();
  if(tid<64){
   for(unsigned j=0;j<16&&base+j<valid;j++){
    const float a=alpha[j],p=prob[j];
    const float v=__half2float(static_cast<const __half*>(vc)[uint64_t(base+j)*kv_width+kvh*64+tid]);
    acc=__fmaf_rn(p,v,__fmul_rn(acc,a));
    if(tid==0)denom=fmaf(denom,a,p);
   }
  }
  __syncthreads();
 }
 if(tid==0)s_shared=denom;
 __syncthreads();
 if(tid<64){const float inv=s_shared>0.f?1.f/s_shared:0.f;out[h*64+tid]=acc*inv;}
}
}
