#pragma once
#include "sm86/moe_down_mmq.cuh"
#include "sm86/moe_down_mmvq.cuh"
#include "sm86/moe_gateup_mmq.cuh"
#include "sm86/moe_gateup_mmvq.cuh"

// Routed FFN execution, independent of model and SM. Weight storage/streaming stays
// with the existing owner; canonical Q4/Q8 readers are shared with dense kernels.
namespace imparo_moe {
constexpr uint32_t max_experts = 256, max_k = 8;

__device__ __forceinline__ void gate_row(const float *s,float *p,float *sel,const float *bias,
                        uint32_t experts,uint32_t gating,uint64_t base) {
    if (gating == 0) {
        if (threadIdx.x == 0) {
            float m = -INFINITY, total = 0;
            for (uint32_t e=0;e<experts;++e) m=fmaxf(m,s[base+e]);
            for (uint32_t e=0;e<experts;++e) { float v=expf(s[base+e]-m); p[base+e]=v; total+=v; }
            for (uint32_t e=0;e<experts;++e) p[base+e]/=total;
        }
    } else {
        for (uint32_t e=threadIdx.x;e<experts;e+=blockDim.x)
            p[base+e]=1.0f/(1.0f+expf(-s[base+e]));
    }
    __syncthreads();
    for (uint32_t e=threadIdx.x;e<experts;e+=blockDim.x)
        sel[base+e]=p[base+e]+(bias ? bias[e] : 0.0f);
}

__global__ void gate(const float *s,float *p,float *sel,const float *bias,
                     uint32_t experts,uint32_t gating) {
    gate_row(s,p,sel,bias,experts,gating,uint64_t(blockIdx.x)*experts);
}

// Deliberately stable: equal scores pick the lower expert id, including -inf.
// A single warp scans one routing row (at most 256 experts).
__device__ __forceinline__ void topk_row(const float *s,uint32_t *out,uint32_t width,
                        uint32_t rows,uint32_t k,uint32_t row) {
    const uint32_t lane=threadIdx.x;
    uint32_t picked[max_k];
    for (uint32_t rank=0;rank<k;++rank) {
        float best=-INFINITY; uint32_t idx=UINT32_MAX;
        for(uint32_t e=lane;e<width;e+=32) {
            bool used=false;
            for(uint32_t j=0;j<rank;++j) used|=picked[j]==e;
            const float v=s[uint64_t(row)*width+e];
            if(!used && (v>best || (v==best && e<idx))) {best=v;idx=e;}
        }
        for(int d=16;d;d>>=1) {
            const float v=__shfl_down_sync(0xffffffff,best,d);
            const uint32_t i=__shfl_down_sync(0xffffffff,idx,d);
            if(lane+uint32_t(d)<32 && (v>best || (v==best && i<idx))) {best=v;idx=i;}
        }
        picked[rank]=__shfl_sync(0xffffffff,idx,0);
        if(!lane) {
            out[uint64_t(row)*k+rank]=idx;
            reinterpret_cast<float*>(out)[uint64_t(rows)*k+uint64_t(row)*k+rank]=best;
        }
    }
}

__global__ void topk(const float *s,uint32_t *out,uint32_t width,
                     uint32_t rows,uint32_t k) {
    topk_row(s,out,width,rows,k,blockIdx.x);
}

// One CTA counting sort. Integer atomics only assign work rows; combine always
// restores original token/pick order, so routing sums never use float atomics.
__device__ __forceinline__ void plan_rows(const uint32_t *picks,const float *p,uint32_t *perm,float *w,
                         uint32_t *seg,uint32_t *inv,uint32_t nt,uint32_t ne,
                         uint32_t k,bool normalise,float scale,
                         uint32_t *counts,uint32_t *base,bool active_tail) {
    for(uint32_t e=threadIdx.x;e<ne;e+=blockDim.x) counts[e]=0;
    __syncthreads();
    for(uint32_t t=threadIdx.x;t<nt;t+=blockDim.x)
        for(uint32_t j=0;j<k;++j) {
            const uint32_t e=picks[t*k+j];
            inv[t*k+j]= e<ne ? atomicAdd(counts+e,1u) : UINT32_MAX;
        }
    __syncthreads();
    if(!threadIdx.x) {
        uint32_t sum=0,nactive=0;
        for(uint32_t e=0;e<ne;++e) {
            base[e]=seg[e]=sum;sum+=counts[e];
            if(active_tail && counts[e])seg[ne+2+nactive++]=e;
        }
        seg[ne]=sum;
        if(active_tail)seg[ne+1]=nactive;
    }
    __syncthreads();
    for(uint32_t t=threadIdx.x;t<nt;t+=blockDim.x) {
        float sum=0;
        for(uint32_t j=0;j<k;++j) {uint32_t e=picks[t*k+j];if(e<ne)sum+=p[uint64_t(t)*ne+e];}
        const float denom=normalise?fmaxf(sum,6.103515625e-5f):1.0f;
        for(uint32_t j=0;j<k;++j) {
            const uint32_t e=picks[t*k+j];if(e>=ne)continue;
            const uint32_t dst=base[e]+inv[t*k+j];
            perm[dst]=t;w[dst]=(p[uint64_t(t)*ne+e]/denom)*scale;inv[t*k+j]=dst;
        }
    }
}

__global__ void plan(const uint32_t *picks,const float *p,uint32_t *perm,float *w,
                     uint32_t *seg,uint32_t *inv,uint32_t nt,uint32_t ne,
                     uint32_t k,bool normalise,float scale,bool active_tail) {
    __shared__ uint32_t counts[max_experts],base[max_experts];
    plan_rows(picks,p,perm,w,seg,inv,nt,ne,k,normalise,scale,counts,base,active_tail);
}

// The same three phases for one token, with CTA barriers replacing stream
// launch boundaries. Only the first warp enters the unchanged stable top-k.
__global__ void route_one(const float *s,float *p,float *sel,uint32_t *top,
                         uint32_t *perm,float *w,uint32_t *seg,uint32_t *inv,
                         const float *bias,uint32_t ne,uint32_t k,uint32_t gating,
                         bool normalise,float scale,bool active_tail) {
    __shared__ uint32_t counts[max_experts],base[max_experts];
    gate_row(s,p,sel,bias,ne,gating,0);
    __syncthreads();
    if(threadIdx.x<32)topk_row(sel,top,ne,1,k,0);
    __syncthreads();
    plan_rows(top,p,perm,w,seg,inv,1,ne,k,normalise,scale,counts,base,active_tail);
}

template<uint32_t Kind> __device__ float value(const uint8_t *r,uint32_t i) {
    if constexpr(Kind==0) return reinterpret_cast<const float*>(r)[i];
    if constexpr(Kind==1) return q4_value(r+uint64_t(i/32)*18,i%32);
    if constexpr(Kind==2) return q8_0_value(r+uint64_t(i/32)*34,i%32);
    if constexpr(Kind==7) return q6_k_value(r+uint64_t(i/256)*210,i%256);
    return 0;
}

// One warp per output dimension, sharing one decoded weight across four work
// rows. Resident stacks dispatch together; offloaded slices use the same kernel.
template<uint32_t Kind, bool Paired=false, bool Active=false> __global__ void grouped(const uint8_t *weights,
        uint64_t stride,uint64_t rowbytes,const float *x,float *y,
        const uint32_t *perm,const uint32_t *seg,uint32_t ni,uint32_t no,
        uint32_t nt,uint32_t rows,bool work,uint32_t expert_base,
        const uint8_t *up_weights,uint32_t ne) {
    uint32_t e=expert_base+blockIdx.y;
    if constexpr(Active) {
        if(blockIdx.y>=seg[ne+1])return;
        e=seg[ne+2+blockIdx.y];
        if(e>=ne)return;
    }
    const uint32_t lane=threadIdx.x%32, r=blockIdx.x*4+threadIdx.x/32;
    const uint32_t lo=seg[e],hi=seg[e+1];
    if(r>=no || lo>hi || hi>rows)return;
    const uint32_t weight_expert=Active?e:blockIdx.y;
    const uint8_t *wr=weights+uint64_t(weight_expert)*stride+uint64_t(r)*rowbytes;
    const uint8_t *up_row=nullptr;
    if constexpr(Paired)
        up_row=up_weights+uint64_t(weight_expert)*stride+uint64_t(r)*rowbytes;
    for(uint32_t c=lo;c<hi;c+=4) {
        float acc[4]={0,0,0,0}, up_acc[4]={0,0,0,0}; uint32_t ar[4];
        #pragma unroll
        for(uint32_t z=0;z<4;++z) ar[z]=c+z<hi?(work?c+z:perm[c+z]):UINT32_MAX;
        for(uint32_t i=lane;i<ni;i+=32) {
            const float v=value<Kind>(wr,i);
            float up_v=0;
            if constexpr(Paired) up_v=value<Kind>(up_row,i);
            #pragma unroll
            for(uint32_t z=0;z<4;++z) {
                if(c+z<hi && ar[z]<(work?rows:nt)) {
                    const float input=x[uint64_t(ar[z])*ni+i];
                    acc[z]=fmaf(v,input,acc[z]);
                    if constexpr(Paired) up_acc[z]=fmaf(up_v,input,up_acc[z]);
                }
            }
        }
        #pragma unroll
        for(uint32_t z=0;z<4;++z) {
            for(int d=16;d;d>>=1)acc[z]+=__shfl_down_sync(0xffffffff,acc[z],d);
            if constexpr(Paired) {
                for(int d=16;d;d>>=1)up_acc[z]+=__shfl_down_sync(0xffffffff,up_acc[z],d);
                if(!lane && c+z<hi)
                    y[uint64_t(c+z)*no+r]=imparo_cuda_lfm2::silu(acc[z])*up_acc[z];
            } else {
                if(!lane && c+z<hi)y[uint64_t(c+z)*no+r]=acc[z];
            }
        }
    }
}

__global__ void combine(const float *src,const float *w,const uint32_t *inv,
                        float *out,uint32_t width,uint32_t k,uint32_t nt) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    if(i>=width)return;
    float sum=0;
    for(uint32_t j=0;j<k;++j) {
        const uint32_t row=inv[t*k+j];
        if(row<nt*k)sum+=w[row]*src[uint64_t(row)*width+i];
    }
    out[uint64_t(t)*width+i]=sum;
}

static bool buffer(uint32_t id,uint64_t elems) {
    return id<B_COUNT && elems<=UINT64_MAX/4 && execution().bufs[id]
        && execution().sizes[id]>=elems*4;
}
static bool distinct(std::initializer_list<uint32_t> ids) {
    for(auto i=ids.begin();i!=ids.end();++i)for(auto j=i+1;j!=ids.end();++j)if(*i==*j)return false;
    return true;
}
static bool gateup_overlap(const void *a,uint64_t na,const void *b,uint64_t nb) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x && y && na && nb && (x<=y ? y-x<na : x-y<nb);
}
static bool gateup_disjoint(std::initializer_list<uint32_t> ids) {
    const auto& s=execution();
    for(auto i=ids.begin();i!=ids.end();++i) {
        if(*i>=B_COUNT || !s.bufs[*i] || !s.sizes[*i])return false;
        for(auto j=i+1;j!=ids.end();++j) {
            if(*j>=B_COUNT || !s.bufs[*j] || !s.sizes[*j]
               || gateup_overlap(s.bufs[*i],s.sizes[*i],s.bufs[*j],s.sizes[*j]))return false;
        }
    }
    return true;
}
static bool gateup_plan_domain(uint32_t nt,uint32_t ne,uint32_t k) {
    const auto& s=execution();
    return s.moe_gateup_mmq && g.sm_version==86 && nt==128 && ne==32 && k==4
        && s.forward_open && s.forward_active && !s.forward_decode && !s.cobatch_rows
        && !s.pending_error && !s.graph_capturing && !s.tree_capture_active
        && !s.prefill_capture_active && !s.verification_capture_active && !s.graph_leases;
}
static void own_gateup_plan(uint32_t perm,uint32_t seg,uint32_t w,uint32_t inv,
                           uint32_t nt,uint32_t ne,uint32_t rows) {
    auto& s=execution();const uint32_t ids[4]={perm,seg,w,inv};
    for(uint32_t i=0;i<4;++i) {
        s.moe_gateup_plan_ids[i]=ids[i];s.moe_gateup_plan_ptrs[i]=s.bufs[ids[i]];
        s.moe_gateup_plan_bytes[i]=s.sizes[ids[i]];s.moe_gateup_plan_epochs[i]=s.buf_epoch[ids[i]];
    }
    s.moe_gateup_plan_nt=nt;s.moe_gateup_plan_ne=ne;s.moe_gateup_plan_rows=rows;
    s.moe_gateup_plan_valid=true;
}
static bool gateup_plan_matches(uint32_t perm,uint32_t seg,uint32_t nt,uint32_t ne,uint32_t rows) {
    const auto& s=execution();
    if(!gateup_plan_domain(nt,ne,4) || !s.moe_gateup_plan_valid
       || s.moe_gateup_plan_ids[0]!=perm || s.moe_gateup_plan_ids[1]!=seg
       || s.moe_gateup_plan_nt!=nt || s.moe_gateup_plan_ne!=ne || s.moe_gateup_plan_rows!=rows)return false;
    for(uint32_t i=0;i<4;++i) {
        const uint32_t id=s.moe_gateup_plan_ids[i];
        if(id>=B_COUNT || s.bufs[id]!=s.moe_gateup_plan_ptrs[i]
           || s.sizes[id]!=s.moe_gateup_plan_bytes[i] || s.buf_epoch[id]!=s.moe_gateup_plan_epochs[i])return false;
    }
    return true;
}
static int done(std::initializer_list<uint32_t> outputs) {
    if(cudaGetLastError()!=cudaSuccess) {set_pending(CUDA_RC_ERROR,"MoE launch");return CUDA_RC_ERROR;}
    for(auto id:outputs)mark_buf_written(id);
    return 0;
}
static bool active_capture_allowed() {
    const auto & s=execution();
    return !s.tree_capture_active && !s.prefill_capture_active
        && (!s.graph_capturing || (s.forward_decode && !s.verification_capture_active
            && s.graph_capture_generation!=0));
}
static bool active_tail_wanted(uint32_t seg,uint32_t nt,uint32_t ne) {
    return execution().moe_active_experts && !execution().pending_error && nt==1 && active_capture_allowed()
        && buffer(seg,uint64_t(ne)*2+2);
}
static bool active_tail_disjoint(uint32_t seg,uint32_t ne,std::initializer_list<uint32_t> ids) {
    const uintptr_t tail=reinterpret_cast<uintptr_t>(execution().bufs[seg]);
    const uint64_t bytes=(uint64_t(ne)*2+2)*sizeof(uint32_t);
    for(const uint32_t id:ids) {
        if(id>=B_COUNT || !execution().bufs[id])return false;
        const uintptr_t other=reinterpret_cast<uintptr_t>(execution().bufs[id]);
        if(other<=tail ? tail-other<execution().sizes[id] : other-tail<bytes)return false;
    }
    return true;
}
static void own_active_tail(uint32_t seg,uint32_t ne,uint32_t rows) {
    auto & s=execution();
    s.moe_active_seg=seg;s.moe_active_ne=ne;s.moe_active_rows=rows;
    s.moe_active_seg_ptr=s.bufs[seg];s.moe_active_seg_epoch=s.buf_epoch[seg];
    s.moe_active_capture_generation=s.graph_capturing?s.graph_capture_generation:0;
}
static bool active_tail_matches(uint32_t seg,uint32_t nt,uint32_t ne,uint32_t rows) {
    const auto & s=execution();
    return active_tail_wanted(seg,nt,ne) && s.moe_active_seg==seg
        && s.moe_active_ne==ne && s.moe_active_rows==rows
        && s.moe_active_seg_ptr==s.bufs[seg] && s.moe_active_seg_epoch==s.buf_epoch[seg]
        && s.moe_active_capture_generation==(s.graph_capturing?s.graph_capture_generation:0);
}
static void active_route_log(uint32_t ne,uint32_t rows,bool paired) {
    static bool logged=false;
    if(!logged) {
        std::fprintf(stderr,"[imparo] moe-active-experts actual route nt=1 ne=%u groups=%u paired=%u\n",
            ne,std::min(rows,ne),unsigned(paired));
        logged=true;
    }
}
}

extern "C" uint32_t imparo_cuda_moe_version() {return 1;}
extern "C" int imparo_cuda_moe_router_f32_v1(uint32_t enabled) {
    if(enabled>1)return CUDA_RC_INVALID;
#if defined(IMPARO_CUDA_ENABLE_PTQ_CUBLAS)
    auto& s=execution();
    if(s.moe_router_f32==(enabled!=0))return 0;
    if(!execution_boundary_closed() || s.graph_leases)return CUDA_RC_INVALID;
    if(!destroy_decode_graph_checked(s))return CUDA_RC_ERROR;
    s.moe_router_f32=enabled!=0;
    return 0;
#else
    return enabled ? -70 : 0;
#endif
}
extern "C" int imparo_cuda_moe_down_mmq_v1(uint32_t enabled) {
    if(enabled>1)return CUDA_RC_INVALID;
    auto& s=execution();
    if(s.moe_down_mmq==(enabled!=0))return 0;
    if(!execution_boundary_closed() || s.graph_leases)return CUDA_RC_INVALID;
    if(!destroy_decode_graph_checked(s))return CUDA_RC_ERROR;
    invalidate_q8_cache();
    s.moe_down_mmq=enabled!=0;
    return 0;
}
extern "C" int imparo_cuda_moe_down_mmvq_v1(uint32_t enabled) {
    auto& s=execution();
    if(enabled>1){set_pending(CUDA_RC_INVALID,"MoE Down MMVQ selector");return CUDA_RC_INVALID;}
    if(s.moe_down_mmvq==(enabled!=0))return 0;
    if(!execution_boundary_closed() || s.graph_leases) {
        set_pending(CUDA_RC_INVALID,"MoE Down MMVQ policy requires closed unleased boundary");
        return CUDA_RC_INVALID;
    }
    if(!destroy_decode_graph_checked(s)) {
        set_pending(CUDA_RC_ERROR,"MoE Down MMVQ graph teardown");return CUDA_RC_ERROR;
    }
    invalidate_q8_cache();s.moe_down_mmvq=enabled!=0;
    return 0;
}
extern "C" int imparo_cuda_moe_gateup_mmq_v1(uint32_t enabled) {
    auto& s=execution();
    if(enabled>1){set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMQ selector");return CUDA_RC_INVALID;}
    if(s.moe_gateup_mmq==(enabled!=0)) {
        if(execution_boundary_closed())s.moe_gateup_plan_valid=false;
        return 0;
    }
    if(!execution_boundary_closed() || s.graph_leases) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMQ policy requires closed unleased boundary");
        return CUDA_RC_INVALID;
    }
    s.moe_gateup_plan_valid=false;
    if(!destroy_decode_graph_checked(s)){set_pending(CUDA_RC_ERROR,"MoE Gate/Up MMQ graph teardown");return CUDA_RC_ERROR;}
    invalidate_q8_cache();s.moe_gateup_mmq=enabled!=0;
    return 0;
}
extern "C" int imparo_cuda_moe_gateup_mmvq_v1(uint32_t enabled) {
    auto& s=execution();
    if(enabled>1){set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMVQ selector");return CUDA_RC_INVALID;}
    if(s.moe_gateup_mmvq==(enabled!=0))return 0;
    if(!execution_boundary_closed() || s.graph_leases) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMVQ policy requires closed unleased boundary");
        return CUDA_RC_INVALID;
    }
    if(!destroy_decode_graph_checked(s)) {
        set_pending(CUDA_RC_ERROR,"MoE Gate/Up MMVQ graph teardown");return CUDA_RC_ERROR;
    }
    invalidate_q8_cache();s.moe_gateup_mmvq=enabled!=0;
    return 0;
}
extern "C" int imparo_cuda_moe_active_experts_v1(uint32_t enabled) {
    if(enabled>1)return CUDA_RC_INVALID;
    auto & s=execution();
    if(s.moe_active_experts==(enabled!=0)) {
        // Includes decode/prefill prepare, whose Graph replay skips begin_forward.
        if(execution_boundary_closed())s.moe_active_seg=UINT32_MAX;
        return 0;
    }
    if(!execution_boundary_closed() || s.graph_leases)return CUDA_RC_INVALID;
    s.moe_active_seg=UINT32_MAX;
    if(!destroy_decode_graph_checked(s))return CUDA_RC_ERROR;
    s.moe_active_experts=enabled!=0;
    return 0;
}
extern "C" int imparo_cuda_top_k_rows(uint32_t s,uint32_t dst,uint32_t width,uint32_t rows,uint32_t k) {
    using namespace imparo_moe;
    if(!rows || !width || !k || k>max_k || k>width || s==dst
       || !buffer(s,uint64_t(width)*rows) || !buffer(dst,uint64_t(rows)*k*2))return CUDA_RC_INVALID;
    topk<<<rows,32,0,g.stream>>>((const float*)execution().bufs[s],(uint32_t*)execution().bufs[dst],width,rows,k);
    return done({dst});
}
extern "C" int imparo_cuda_moe_gate(uint32_t s,uint32_t p,uint32_t sel,uint64_t bias,
        uint32_t nt,uint32_t ne,uint32_t gating) {
    using namespace imparo_moe;
    if(!nt || !ne || ne>max_experts || gating>1 || !distinct({s,p,sel})
       || !buffer(s,uint64_t(nt)*ne) || !buffer(p,uint64_t(nt)*ne) || !buffer(sel,uint64_t(nt)*ne))return CUDA_RC_INVALID;
    const uint8_t *b=nullptr;
    if(bias!=UINT64_MAX) {if(bias%4)return CUDA_RC_INVALID;int rc=weight_slice(bias,uint64_t(ne)*4,&b);if(rc)return rc;}
    gate<<<nt,128,0,g.stream>>>((const float*)execution().bufs[s],(float*)execution().bufs[p],
        (float*)execution().bufs[sel],(const float*)b,ne,gating);
    return done({p,sel});
}
extern "C" int imparo_cuda_moe_plan(uint32_t top,uint32_t p,uint32_t perm,uint32_t w,
        uint32_t seg,uint32_t inv,uint32_t nt,uint32_t ne,uint32_t k,uint32_t norm,float scale) {
    using namespace imparo_moe;
    execution().moe_gateup_plan_valid=false;
    const uint64_t rows=uint64_t(nt)*k;
    if(!nt || !ne || ne>max_experts || !k || k>max_k || k>ne || rows>UINT32_MAX
       || norm>1 || !std::isfinite(scale) || !distinct({top,p,perm,w,seg,inv})
       || !buffer(top,rows) || !buffer(p,uint64_t(nt)*ne) || !buffer(perm,rows)
       || !buffer(w,rows) || !buffer(seg,ne+1) || !buffer(inv,rows))return CUDA_RC_INVALID;
    const bool gateup=gateup_plan_domain(nt,ne,k) && gateup_disjoint({top,p,perm,w,seg,inv});
    if(gateup) {
        // Invalid expert picks are skipped by plan_rows. Zero its unused PERM
        // tail so every row read by the original unbounded gather is in [0,nt).
        if(cudaMemsetAsync(execution().bufs[perm],0,rows*sizeof(uint32_t),g.stream)!=cudaSuccess) {
            set_pending(CUDA_RC_ERROR,"MoE Gate/Up PERM initialization");return CUDA_RC_ERROR;
        }
        mark_buf_written(perm);
    }
    const bool active=active_tail_wanted(seg,nt,ne) && active_tail_disjoint(seg,ne,{top,p,perm,w,inv});
    plan<<<1,128,0,g.stream>>>((const uint32_t*)execution().bufs[top],(const float*)execution().bufs[p],
        (uint32_t*)execution().bufs[perm],(float*)execution().bufs[w],(uint32_t*)execution().bufs[seg],
        (uint32_t*)execution().bufs[inv],nt,ne,k,norm!=0,scale,active);
    const int rc=done({perm,w,seg,inv});
    if(!rc && active)own_active_tail(seg,ne,uint32_t(rows));
    if(!rc && gateup)own_gateup_plan(perm,seg,w,inv,nt,ne,uint32_t(rows));
    return rc;
}
// Independently optional extension. Admission cannot stage weights or write any
// output: missing/nonresident bias and all unsupported shapes use the old route.
extern "C" int imparo_cuda_moe_route_v1(uint32_t s,uint32_t p,uint32_t sel,
        uint32_t top,uint32_t perm,uint32_t w,uint32_t seg,uint32_t inv,
        uint64_t bias,uint32_t nt,uint32_t ne,uint32_t k,uint32_t gating,
        uint32_t norm,float scale) {
    using namespace imparo_moe;
    if(nt!=1 || !ne || ne>max_experts || !k || k>max_k || k>ne
       || gating>1 || norm>1 || !std::isfinite(scale)
       || !distinct({s,p,sel,top,perm,w,seg,inv})
       || !buffer(s,ne) || !buffer(p,ne) || !buffer(sel,ne) || !buffer(top,uint64_t(k)*2)
       || !buffer(perm,k) || !buffer(w,k) || !buffer(seg,ne+1) || !buffer(inv,k))return CUDA_RC_INVALID;
    const uint8_t *bias_data=nullptr;
    if(bias!=UINT64_MAX) {
        const uint64_t bytes=uint64_t(ne)*sizeof(float);
        if(bias%sizeof(float) || bias>g.weights_len || bytes>g.weights_len-bias)return CUDA_RC_INVALID;
        bias_data=resident_weight_range(bias,bytes);
        if(!bias_data)return CUDA_RC_INVALID;
    }
    const bool active=active_tail_wanted(seg,nt,ne) && active_tail_disjoint(seg,ne,{s,p,sel,top,perm,w,inv});
    route_one<<<1,128,0,g.stream>>>((const float*)execution().bufs[s],
        (float*)execution().bufs[p],(float*)execution().bufs[sel],
        (uint32_t*)execution().bufs[top],(uint32_t*)execution().bufs[perm],
        (float*)execution().bufs[w],(uint32_t*)execution().bufs[seg],
        (uint32_t*)execution().bufs[inv],(const float*)bias_data,ne,k,gating,norm!=0,scale,active);
    const int rc=done({p,sel,top,perm,w,seg,inv});
    if(!rc && active)own_active_tail(seg,ne,k);
    return rc;
}
extern "C" int imparo_cuda_moe_grouped(uint32_t kind,uint64_t off,uint64_t stride,
        uint32_t src,uint32_t dst,uint32_t perm,uint32_t seg,uint32_t ni,uint32_t no,
        uint32_t ne,uint32_t nt,uint32_t rows,uint32_t work) {
    using namespace imparo_moe;
    if(!distinct({src,dst,perm,seg}) && execution().moe_down_mmvq
       && g.sm_version==86 && kind==1 && ni==1792 && no==2048 && nt==1
       && rows==4 && ne==32 && stride==2064384 && work==1
       && execution().forward_open && execution().forward_active
       && execution().forward_decode && !execution().cobatch_rows) {
        set_pending(CUDA_RC_INVALID,"MoE Down MMVQ aliased buffer ids");return CUDA_RC_INVALID;
    }
    uint64_t rb= kind==0?uint64_t(ni)*4:
        kind==1 && ni%32==0?uint64_t(ni/32)*18:kind==2 && ni%32==0?uint64_t(ni/32)*34:kind==7 && ni%256==0?uint64_t(ni/256)*210:0;
    if(!rb || !ni || !no || !ne || ne>max_experts || !nt || !rows || work>1
       || rb>UINT64_MAX/no || !distinct({src,dst,perm,seg})
       || !buffer(src,uint64_t(work?rows:nt)*ni) || !buffer(dst,uint64_t(rows)*no)
       || !buffer(perm,rows) || !buffer(seg,ne+1))return CUDA_RC_INVALID;
    const uint64_t bytes=rb*no,align=kind==0?4:2;
    if(stride<bytes || off%align || stride%align || stride>(UINT64_MAX-bytes)/(ne-1?ne-1:1))return CUDA_RC_INVALID;
    const uint64_t total=uint64_t(ne-1)*stride+bytes;
    if(off>g.weights_len || total>g.weights_len-off)return CUDA_RC_INVALID;
    const uint8_t *resident=resident_weight_range(off,total);
    // Eager fallback uses the existing bounded weight owner, not a second full
    // expanded stack. Capturing an offloaded stack is refused before output.
    if(!resident && execution().graph_capturing)return CUDA_RC_INVALID;
    auto& s=execution();
    const bool down_mmq=s.moe_down_mmq && resident && g.sm_version==86
        && kind==1 && ni==1792 && no==2048 && nt==128 && rows==512 && ne==32
        && work==1 && !s.forward_decode && !s.cobatch_rows
        && reinterpret_cast<uintptr_t>(s.bufs[src])%alignof(float4)==0;
    if(down_mmq) {
        // This numerical policy is eager-only; never silently record the old path.
        if(s.graph_capturing || s.tree_capture_active || s.prefill_capture_active
           || s.verification_capture_active || s.graph_leases) {
            set_pending(CUDA_RC_INVALID,"MoE Down MMQ requires eager unleased storage");
            return CUDA_RC_INVALID;
        }
        if(s.pending_error)return s.pending_error;
        constexpr uint64_t q8_bytes=14ull*512*sizeof(BlockQ8_1Mmq);
        static_assert(q8_bytes==1032192,"fixed MoE Down Q8 capacity");
        int rc=ensure_q8_scratch(q8_bytes);
        if(rc){set_pending(rc,"MoE Down MMQ Q8 scratch");return rc;}
        // A same-capacity overwrite must also revoke any previous dense Q8 owner.
        invalidate_q8_cache();
        auto* q8=static_cast<BlockQ8_1Mmq*>(s.q8_scratch);
        k_quantize_q8_1_mmq<<<dim3(512,4),128,0,g.stream>>>(
            static_cast<const float*>(s.bufs[src]),q8,1792,512,0,false,false);
        if(cudaGetLastError()!=cudaSuccess) {
            set_pending(CUDA_RC_ERROR,"MoE Down MMQ quantization launch");
            return CUDA_RC_ERROR;
        }
        k_moe_down_mmq_lab<<<dim3(16,32,4),dim3(32,8),
            imparo_sm80_mmq::kHalfKSharedBytes,g.stream>>>(resident,stride,q8,
            static_cast<float*>(s.bufs[dst]),static_cast<const uint32_t*>(s.bufs[seg]),
            ni,no,rows,ne);
        rc=done({dst});
        if(!rc && !s.moe_down_mmq_logged) {
            std::fprintf(stderr,"[imparo] moe-down-mmq actual route nt=128 ne=32 k=1792 n=2048 rows=512 q8_bytes=1032192 tile=128x128x128\n");
            s.moe_down_mmq_logged=true;
        }
        return rc;
    }
    const bool down_mmvq=s.moe_down_mmvq && resident && g.sm_version==86
        && kind==1 && ni==1792 && no==2048 && nt==1 && rows==4 && ne==32
        && stride==2064384 && work==1 && s.forward_open && s.forward_active
        && s.forward_decode && !s.cobatch_rows;
    if(down_mmvq) {
        if(s.graph_capturing || s.tree_capture_active || s.prefill_capture_active
           || s.verification_capture_active || s.graph_leases) {
            set_pending(CUDA_RC_INVALID,"MoE Down MMVQ requires eager unleased storage");
            return CUDA_RC_INVALID;
        }
        if(s.pending_error)return s.pending_error;
        if(!gateup_disjoint({src,dst,perm,seg})
           || gateup_overlap(s.bufs[dst],s.sizes[dst],resident,total)) {
            set_pending(CUDA_RC_INVALID,"MoE Down MMVQ aliased storage");return CUDA_RC_INVALID;
        }
        constexpr uint64_t q8_bytes=4ull*(1792/32)*sizeof(BlockQ8_1);
        static_assert(q8_bytes==8064,"fixed MoE Down MMVQ Q8 capacity");
        int rc=ensure_q8_scratch(q8_bytes);
        if(rc){set_pending(rc,"MoE Down MMVQ Q8 scratch");return rc;}
        // Also revoke dense Q8 ownership when existing capacity is reused.
        invalidate_q8_cache();
        auto* q8=static_cast<BlockQ8_1*>(s.q8_scratch);
        imparo_sm80_mmvq::quantize_q8_1<<<dim3(7,4),256,0,g.stream>>>(
            static_cast<const float*>(s.bufs[src]),q8,1792,4,0);
        if(cudaGetLastError()!=cudaSuccess) {
            set_pending(CUDA_RC_ERROR,"MoE Down MMVQ quantization launch");
            return CUDA_RC_ERROR;
        }
        k_moe_down_mmvq_lab<<<dim3(1024,4),dim3(32,4),0,g.stream>>>(resident,stride,q8,
            static_cast<float*>(s.bufs[dst]),static_cast<const uint32_t*>(s.bufs[seg]),
            ni,no,rows,ne);
        rc=done({dst});
        if(!rc && !s.moe_down_mmvq_logged) {
            std::fprintf(stderr,"[imparo] moe-down-mmvq actual route nt=1 ne=32 k=1792 n=2048 rows=4 q8_bytes=8064 warps=4 rows_per_cta=2\n");
            s.moe_down_mmvq_logged=true;
        }
        return rc;
    }
    const bool active=resident && kind==1 && active_tail_matches(seg,nt,ne,rows)
        && active_tail_disjoint(seg,ne,{src,dst,perm});
    if(active) {
        grouped<1,false,true><<<dim3((no+3)/4,std::min(rows,ne)),128,0,g.stream>>>(resident,stride,rb,
            (const float*)execution().bufs[src],(float*)execution().bufs[dst],
            (const uint32_t*)execution().bufs[perm],(const uint32_t*)execution().bufs[seg],
            ni,no,nt,rows,work!=0,0,nullptr,ne);
        const int rc=done({dst});
        if(!rc)active_route_log(ne,rows,false);
        return rc;
    }
    auto launch=[&](const uint8_t *weights,uint32_t count,uint32_t base) {
        const dim3 grid((no+3)/4,count);
        #define IMPARO_MOE_DISPATCH(K) grouped<K><<<grid,128,0,g.stream>>>(weights,stride,rb,\
            (const float*)execution().bufs[src],(float*)execution().bufs[dst],\
            (const uint32_t*)execution().bufs[perm],(const uint32_t*)execution().bufs[seg],\
            ni,no,nt,rows,work!=0,base,nullptr,ne)
        switch(kind) {case 0:IMPARO_MOE_DISPATCH(0);break;case 1:IMPARO_MOE_DISPATCH(1);break;
            case 2:IMPARO_MOE_DISPATCH(2);break;case 7:IMPARO_MOE_DISPATCH(7);break;}
        #undef IMPARO_MOE_DISPATCH
    };
    if(resident)launch(resident,ne,0);
    else {
        int rc=ensure_weight_cache(bytes);if(rc)return rc;
        for(uint32_t e=0;e<ne;++e) {
            const uint8_t *w=nullptr;rc=weight_slice(off+uint64_t(e)*stride,bytes,&w);
            if(rc){set_pending(rc,"MoE expert weight slice");return rc;}
            launch(w,1,e);
        }
    }
    return done({dst});
}
// Optional v1 pair extension. Refuse unsupported/nonresident stacks before any
// launch or weight-cache mutation; callers retain the ordinary grouped fallback.
// The paired instantiation preserves each projection's FMA/reduction order and
// uses exactly the standalone SiLU-multiply helper after those two reductions.
extern "C" int imparo_cuda_moe_grouped_pair_v1(uint32_t kind,
        uint64_t gate_off,uint64_t up_off,uint64_t stride,uint32_t src,uint32_t dst,
        uint32_t perm,uint32_t seg,uint32_t ni,uint32_t no,uint32_t ne,
        uint32_t nt,uint32_t rows) {
    using namespace imparo_moe;
    if(kind!=1 || !ni || ni%32 || !no || !ne || ne>max_experts || !nt
       || !rows || rows>UINT32_MAX-3 || rows%nt || rows/nt>max_k || rows/nt>ne
       || uint64_t(rows)*no>UINT32_MAX || !distinct({src,dst,perm,seg})
       || !buffer(src,uint64_t(nt)*ni) || !buffer(dst,uint64_t(rows)*no)
       || !buffer(perm,rows) || !buffer(seg,ne+1))return CUDA_RC_INVALID;
    const uint64_t rb=uint64_t(ni/32)*18;
    if(rb>UINT64_MAX/no)return CUDA_RC_INVALID;
    const uint64_t bytes=rb*no;
    if(stride<bytes || gate_off%2 || up_off%2 || stride%2
       || (ne>1 && stride>(UINT64_MAX-bytes)/(ne-1)))return CUDA_RC_INVALID;
    const uint64_t total=uint64_t(ne-1)*stride+bytes;
    if(gate_off>g.weights_len || total>g.weights_len-gate_off
       || up_off>g.weights_len || total>g.weights_len-up_off)return CUDA_RC_INVALID;
    const uint8_t *gate=resident_weight_range(gate_off,total);
    const uint8_t *up=resident_weight_range(up_off,total);
    if(!gate || !up)return CUDA_RC_INVALID;
    const bool active=active_tail_matches(seg,nt,ne,rows) && active_tail_disjoint(seg,ne,{src,dst,perm});
    #define IMPARO_MOE_PAIR(ACTIVE,GROUPS) grouped<1,true,ACTIVE><<<dim3((uint64_t(no)+3)/4,GROUPS),128,0,g.stream>>>(gate,stride,rb,\
        (const float*)execution().bufs[src],(float*)execution().bufs[dst],\
        (const uint32_t*)execution().bufs[perm],(const uint32_t*)execution().bufs[seg],\
        ni,no,nt,rows,false,0,up,ne)
    if(active) { IMPARO_MOE_PAIR(true,std::min(rows,ne)); }
    else { IMPARO_MOE_PAIR(false,ne); }
    #undef IMPARO_MOE_PAIR
    const int rc=done({dst});
    if(!rc && active)active_route_log(ne,rows,true);
    return rc;
}
// Optional exact-shape Gate/Up transaction. Unsupported admission returns -70
// before any output/scratch write; after admission every failure is pending.
extern "C" int imparo_cuda_moe_gateup_mmq_pair_v1(uint32_t kind,
        uint64_t gate_off,uint64_t up_off,uint64_t stride,uint32_t src,uint32_t dst,
        uint32_t scratch,uint32_t perm,uint32_t seg,uint32_t ni,uint32_t no,
        uint32_t ne,uint32_t nt,uint32_t rows) {
    using namespace imparo_moe;auto& s=execution();
    if(!s.moe_gateup_mmq || g.sm_version!=86 || kind!=1 || ni!=2048 || no!=1792
       || nt!=128 || rows!=512 || ne!=32 || s.forward_decode || s.cobatch_rows)return -70;
    if(s.pending_error)return s.pending_error;
    if(s.graph_capturing || s.tree_capture_active || s.prefill_capture_active
       || s.verification_capture_active || s.graph_leases) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMQ requires eager unleased storage");return CUDA_RC_INVALID;
    }
    if(!buffer(src,uint64_t(nt)*ni) || !buffer(dst,uint64_t(rows)*no)
       || !buffer(scratch,uint64_t(rows)*ni) || !buffer(perm,rows) || !buffer(seg,ne+1)
       || !gateup_plan_matches(perm,seg,nt,ne,rows))return -70;
    const uint32_t w=s.moe_gateup_plan_ids[2],inv=s.moe_gateup_plan_ids[3];
    if(!gateup_disjoint({src,dst,scratch,perm,seg,w,inv})
       || reinterpret_cast<uintptr_t>(s.bufs[scratch])%alignof(float4))return -70;
    constexpr uint64_t expert_bytes=uint64_t(2048/32)*18*1792;
    if(stride<expert_bytes || (gate_off|up_off|stride)%2
       || stride>(UINT64_MAX-expert_bytes)/31) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMQ weight strides");return CUDA_RC_INVALID;
    }
    const uint64_t total=31*stride+expert_bytes;
    if(gate_off>g.weights_len || total>g.weights_len-gate_off
       || up_off>g.weights_len || total>g.weights_len-up_off) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMQ weight bounds");return CUDA_RC_INVALID;
    }
    const uint8_t *gate=resident_weight_range(gate_off,total),*up=resident_weight_range(up_off,total);
    if(!gate || !up)return -70;
    for(const uint32_t id:{dst,scratch}) {
        if(gateup_overlap(s.bufs[id],s.sizes[id],gate,total)
           || gateup_overlap(s.bufs[id],s.sizes[id],up,total))return -70;
    }
    // One consumer only, including failure after scratch allocation or launch.
    s.moe_gateup_plan_valid=false;
    constexpr uint64_t q8_bytes=16ull*512*sizeof(BlockQ8_1Mmq);
    static_assert(q8_bytes==1179648,"fixed MoE Gate/Up Q8 capacity");
    int rc=ensure_q8_scratch(q8_bytes);
    if(rc){set_pending(rc,"MoE Gate/Up MMQ Q8 scratch");return rc;}
    invalidate_q8_cache();
    auto* gathered=static_cast<float*>(s.bufs[scratch]);
    auto* q8=static_cast<BlockQ8_1Mmq*>(s.q8_scratch);
    tree_commit_gather<<<(512u*2048u*sizeof(float)+255)/256,256,0,g.stream>>>(
        static_cast<const uint8_t*>(s.bufs[src]),reinterpret_cast<uint8_t*>(gathered),
        static_cast<const int*>(s.bufs[perm]),0,2048*sizeof(float),512,nullptr);
    rc=done({scratch});if(rc)return rc;
    k_quantize_q8_1_mmq<<<dim3(512,4),128,0,g.stream>>>(gathered,q8,2048,512,0,false,false);
    if(cudaGetLastError()!=cudaSuccess){set_pending(CUDA_RC_ERROR,"MoE Gate/Up MMQ quantization launch");return CUDA_RC_ERROR;}
    k_moe_gateup_mmq_projection_lab<<<dim3(14,32,4),dim3(32,8),
        imparo_sm80_mmq::kHalfKSharedBytes,g.stream>>>(gate,stride,q8,
        static_cast<float*>(s.bufs[dst]),static_cast<const uint32_t*>(s.bufs[seg]),ni,no,rows,ne);
    rc=done({dst});if(rc)return rc;
    // The gather has already been quantized on this stream; its U storage can
    // now hold the smaller Up output without a second activation allocation.
    k_moe_gateup_mmq_projection_lab<<<dim3(14,32,4),dim3(32,8),
        imparo_sm80_mmq::kHalfKSharedBytes,g.stream>>>(up,stride,q8,gathered,
        static_cast<const uint32_t*>(s.bufs[seg]),ni,no,rows,ne);
    rc=done({scratch});if(rc)return rc;
    imparo_cuda_lfm2::silu_mul_kernel<<<(512u*1792u+255)/256,256,0,g.stream>>>(
        static_cast<float*>(s.bufs[dst]),gathered,512u*1792u);
    rc=done({dst});
    if(!rc && !s.moe_gateup_mmq_logged) {
        std::fprintf(stderr,"[imparo] moe-gateup-mmq actual route nt=128 ne=32 k=2048 n=1792 rows=512 q8_bytes=1179648 gather_bytes=4194304 tile=128x128x128\n");
        s.moe_gateup_mmq_logged=true;
    }
    return rc;
}
// Optional exact-shape Decode Gate/Up transaction. The retained scratch argument
// is ABI-compatible with the Prefill extension; this fused provider never uses U.
extern "C" int imparo_cuda_moe_gateup_mmvq_pair_v1(uint32_t kind,
        uint64_t gate_off,uint64_t up_off,uint64_t stride,uint32_t src,uint32_t dst,
        uint32_t scratch,uint32_t perm,uint32_t seg,uint32_t ni,uint32_t no,
        uint32_t ne,uint32_t nt,uint32_t rows) {
    using namespace imparo_moe;auto& s=execution();(void)scratch;
    constexpr uint64_t expert_bytes=uint64_t(2048/32)*18*1792;
    static_assert(expert_bytes==2064384,"fixed MoE Gate/Up MMVQ expert stride");
    if(!s.moe_gateup_mmvq || g.sm_version!=86 || kind!=1 || ni!=2048 || no!=1792
       || nt!=1 || rows!=4 || ne!=32 || stride!=expert_bytes
       || !s.forward_open || !s.forward_active || !s.forward_decode || s.cobatch_rows)return -70;
    if(s.pending_error)return s.pending_error;
    if(s.graph_capturing || s.tree_capture_active || s.prefill_capture_active
       || s.verification_capture_active || s.graph_leases) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMVQ requires eager unleased storage");return CUDA_RC_INVALID;
    }
    if(!buffer(src,ni) || !buffer(dst,uint64_t(rows)*no)
       || !buffer(perm,rows) || !buffer(seg,ne+1)) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMVQ buffer bounds");return CUDA_RC_INVALID;
    }
    constexpr uint64_t total=32*expert_bytes;
    if((gate_off|up_off)%2 || gate_off>g.weights_len || total>g.weights_len-gate_off
       || up_off>g.weights_len || total>g.weights_len-up_off) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMVQ weight bounds");return CUDA_RC_INVALID;
    }
    const uint8_t *gate=resident_weight_range(gate_off,total),*up=resident_weight_range(up_off,total);
    if(!gate || !up)return -70;
    if(!gateup_disjoint({src,dst,perm,seg})
       || gateup_overlap(s.bufs[dst],s.sizes[dst],gate,total)
       || gateup_overlap(s.bufs[dst],s.sizes[dst],up,total)) {
        set_pending(CUDA_RC_INVALID,"MoE Gate/Up MMVQ aliased storage");return CUDA_RC_INVALID;
    }
    constexpr uint64_t q8_bytes=(2048ull/32)*sizeof(BlockQ8_1);
    static_assert(q8_bytes==2304,"fixed MoE Gate/Up MMVQ Q8 capacity");
    int rc=ensure_q8_scratch(q8_bytes);
    if(rc){set_pending(rc,"MoE Gate/Up MMVQ Q8 scratch");return rc;}
    // One Cur row supplies all four experts; overwrites revoke dense Q8 ownership.
    invalidate_q8_cache();
    auto* q8=static_cast<BlockQ8_1*>(s.q8_scratch);
    imparo_sm80_mmvq::quantize_q8_1<<<dim3(8,1),256,0,g.stream>>>(
        static_cast<const float*>(s.bufs[src]),q8,2048,1,0);
    if(cudaGetLastError()!=cudaSuccess) {
        set_pending(CUDA_RC_ERROR,"MoE Gate/Up MMVQ quantization launch");return CUDA_RC_ERROR;
    }
    k_moe_gateup_mmvq_lab<<<dim3(1792,4),dim3(32,4),0,g.stream>>>(gate,up,stride,q8,
        static_cast<float*>(s.bufs[dst]),static_cast<const uint32_t*>(s.bufs[perm]),
        static_cast<const uint32_t*>(s.bufs[seg]),ni,no,nt,rows,ne);
    rc=done({dst});
    if(!rc && !s.moe_gateup_mmvq_logged) {
        std::fprintf(stderr,"[imparo] moe-gateup-mmvq actual route nt=1 ne=32 k=2048 n=1792 rows=4 q8_bytes=2304 warps=4 rows_per_cta=1 fused_silu=1\n");
        s.moe_gateup_mmvq_logged=true;
    }
    return rc;
}
extern "C" int imparo_cuda_moe_combine(uint32_t src,uint32_t w,uint32_t inv,uint32_t dst,
        uint32_t width,uint32_t k,uint32_t nt) {
    using namespace imparo_moe;
    const uint64_t rows=uint64_t(nt)*k;
    if(!width || !nt || !k || k>max_k || rows>UINT32_MAX || !distinct({src,w,inv,dst})
       || !buffer(src,rows*width) || !buffer(w,rows) || !buffer(inv,rows)
       || !buffer(dst,uint64_t(nt)*width))return CUDA_RC_INVALID;
    combine<<<dim3((width+255)/256,nt),256,0,g.stream>>>((const float*)execution().bufs[src],
        (const float*)execution().bufs[w],(const uint32_t*)execution().bufs[inv],(float*)execution().bufs[dst],width,k,nt);
    return done({dst});
}
