#pragma once
#include "ptq1.cuh"
#include "ptq1_tc.cuh"
#include "weight_basis.cuh"
#include "gated_delta.cuh"

static bool buffer_slice(uint32_t id, uint64_t byte_offset, uint64_t bytes, uint8_t **out);

// Adapter onto the existing allocation, stream, execution-owner and weight-tier
// services. Model-specific basis data is immutable after load; temporary tensors
// use the execution owner's existing scratch and invalidate its previous format.
struct WeightBasisNative {
    uint64_t offset; uint32_t width, block, inverse, head_dim, key_heads, value_heads;
    const int8_t *signs;
};
struct WeightBasisSigns { std::vector<int8_t> values; void *device=nullptr; };
static std::vector<WeightBasisNative> weight_bases;
static std::vector<WeightBasisSigns> weight_basis_signs;
static const WeightBasisNative *find_weight_basis(uint64_t offset) {
    auto i=std::lower_bound(weight_bases.begin(),weight_bases.end(),offset,
        [](const WeightBasisNative&a,uint64_t b){return a.offset<b;});
    return i!=weight_bases.end()&&i->offset==offset?&*i:nullptr;
}
extern "C" int imparo_cuda_weight_basis_register(uint64_t offset,uint32_t width,
        uint32_t block,uint32_t inverse,uint32_t hd,uint32_t kh,uint32_t vh,
        const int8_t *signs) {
    if(!g.weights_host||!signs||!width||width>1048576||!block||block>1024
            ||(block&(block-1))||width%block||inverse>1||offset>=g.weights_len
            ||find_weight_basis(offset)||!execution_owners.empty()
            ||(hd && (inverse||!kh||!vh||vh%kh||uint64_t(hd)*vh!=width))
            ||(!hd&&(kh||vh)))return CUDA_RC_INVALID;
    for(uint32_t i=0;i<width;++i)if(signs[i]!=1&&signs[i]!=-1)return CUDA_RC_INVALID;
    const int8_t *device=nullptr;
    for(const auto &s:weight_basis_signs) {
        if(s.values.size()==width&&std::memcmp(s.values.data(),signs,width)==0) {
            device=static_cast<const int8_t*>(s.device);break;
        }
    }
    if(!device) {
        WeightBasisSigns s; s.values.assign(signs,signs+width);
        int rc=alloc_raw(&s.device,width,"weight basis signs");if(rc)return rc;
        if(cudaMemcpy(s.device,signs,width,cudaMemcpyHostToDevice)!=cudaSuccess) {
            cudaFree(s.device);return CUDA_RC_ERROR;
        }
        device=static_cast<const int8_t*>(s.device);weight_basis_signs.push_back(std::move(s));
    }
    WeightBasisNative basis{offset,width,block,inverse,hd,kh,vh,device};
    auto pos=std::lower_bound(weight_bases.begin(),weight_bases.end(),offset,
        [](const WeightBasisNative&a,uint64_t b){return a.offset<b;});
    weight_bases.insert(pos,basis);return 0;
}
static int launch_weight_basis(const WeightBasisNative &b,const float*x,float*y,uint32_t rows) {
    auto rc=imparo_cuda_weight_basis::launch(x,y,b.signs,b.width,b.block,rows,
        b.inverse!=0,b.head_dim,b.key_heads,b.value_heads,g.stream);
    return rc==cudaSuccess?0:CUDA_RC_ERROR;
}
static void bonsai_matmat(uint32_t kind,uint64_t off,uint32_t k,uint32_t n,
        uint32_t src,uint32_t dst,uint32_t m,uint32_t src_row) {
    uint8_t *xb=nullptr,*yb=nullptr;
    if(!k||!n||!m||m>65535||n>0x7fffffffu||k>0x7fffffffu||execution().epilogue
            ||(kind==39&&k%128)
            ||!buffer_slice(src,uint64_t(src_row)*k*4,uint64_t(m)*k*4,&xb)
            ||!buffer_slice(dst,0,uint64_t(m)*n*4,&yb)) {
        set_pending(CUDA_RC_INVALID,"Bonsai matmat geometry");return;
    }
    const float*x=reinterpret_cast<const float*>(xb);
    const auto*basis=find_weight_basis(off);
    if(kind==39&&!basis){set_pending(CUDA_RC_INVALID,"PTQ matrix missing input basis");return;}
    if (kind == 39) g.ptq_prefill_tensorcore_frozen = true;
    if(basis) {
        if(basis->inverse||basis->width!=k){set_pending(CUDA_RC_INVALID,"matrix basis geometry");return;}
        int rc=ensure_q8_scratch(uint64_t(m)*k*4);
        if(rc){set_pending(rc,"weight basis scratch");return;}
        invalidate_q8_cache();
        auto*rotated=static_cast<float*>(execution().q8_scratch);
        rc=launch_weight_basis(*basis,x,rotated,m);
        if(rc){set_pending(rc,"weight basis transform");return;}x=rotated;
    }
    const uint64_t row_bytes=kind==39?uint64_t(k/128)*28:uint64_t(k)*2;
    const uint64_t bytes=row_bytes*n;
    const bool resident=resident_weight_range(off,bytes)!=nullptr;
    const uint64_t limit=resident?bytes:weight_transfer_slice_limit();
    if(row_bytes>limit){set_pending(CUDA_RC_OOM,"Bonsai weight row exceeds cache");return;}
    const uint32_t step=uint32_t(std::min<uint64_t>(n,limit/row_bytes));
    // Policy 1 preserves the previously checked FFN-only domain. Policy 2
    // reuses that same provider for the independently checked non-FFN shapes.
    // Model basis/permutation, compact placement and output owner stay common;
    // M<=8, BF16 and unlisted shapes retain their existing implementation.
    uint32_t tc_route = 0;
    if (g.ptq_prefill_tensorcore && g.sm_version >= 80
        && kind == 39 && m > 8 && m <= 128) {
        if (k == 5120 && n == 17408) tc_route = 1u;
        else if (k == 17408 && n == 5120) tc_route = 2u;
        else if (g.ptq_prefill_tensorcore >= 2 && g.sm_version == 86) {
            if (k == 5120 && n == 10240) tc_route = 4u;
            else if (k == 5120 && n == 6144) tc_route = 8u;
            else if (k == 6144 && n == 5120) tc_route = 16u;
            else if (k == 5120 && n == 12288) tc_route = 32u;
            else if (k == 5120 && n == 1024) tc_route = 64u;
        }
    }
#if defined(IMPARO_CUDA_ENABLE_PTQ_CUBLAS)
    // Policy 3 changes only the two FFN geometries at substantial M; all other
    // policy-2 projections and small/tail batches retain their checked provider.
    const bool bounded_gemm=g.ptq_prefill_tensorcore==3 && (tc_route==1u||tc_route==2u)
        && m>=96 && m<=128;
    if(bounded_gemm) {
        auto& scratch=execution().ptq_gemm;
        if(execution().graph_capturing||execution().prefill_capture_active||execution().tree_capture_active) {
            set_pending(CUDA_RC_INVALID,"bounded PTQ GEMM capture not admitted");return;
        }
        if(!scratch.handle && cublasCreate(&scratch.handle)!=CUBLAS_STATUS_SUCCESS) {
            scratch.handle=nullptr;set_pending(CUDA_RC_ERROR,"bounded PTQ GEMM handle");return;
        }
        if(!scratch.storage) {
            const int rc=alloc_raw(&scratch.storage,imparo_cuda_ptq1_gemm::storage_bytes,"bounded PTQ GEMM scratch");
            if(rc){set_pending(rc,"bounded PTQ GEMM scratch");return;}
            scratch.bytes=imparo_cuda_ptq1_gemm::storage_bytes;
        }
    }
#endif
    MatmatEventScope profile(kind,k,n,m,0);
    mark_tuner_dispatch(1,(uint64_t(kind)<<32)|m);
    for(uint32_t base=0;base<n;base+=step) {
        const uint32_t rows=std::min(step,n-base);const uint8_t*w=nullptr;
        int rc=weight_slice(off+uint64_t(base)*row_bytes,uint64_t(rows)*row_bytes,&w);
        if(rc){set_pending(rc,"Bonsai packed weight slice");return;}
#if defined(IMPARO_CUDA_ENABLE_PTQ_CUBLAS)
        if(bounded_gemm) {
            const auto err=imparo_cuda_ptq1_gemm::launch(execution().ptq_gemm,w,x,
                reinterpret_cast<float*>(yb),k,rows,m,n,base,g.stream);
            if(err!=cudaSuccess){set_pending(CUDA_RC_ERROR,"bounded PTQ GEMM launch");return;}
            const uint32_t route=tc_route<<8;
            if(!(g.ptq_prefill_tensorcore_trace_mask&route)) {
                std::fprintf(stderr,"[imparo] ptq_prefill_tensorcore actual route k=%u n=%u m=%u bounded-cublas=f16-f32 scratch=%llu weight-tile=67108864\n",k,n,m,
                    static_cast<unsigned long long>(execution().ptq_gemm.bytes));
                g.ptq_prefill_tensorcore_trace_mask|=route;
            }
            continue;
        }
#endif
        if (tc_route) {
            const auto err = imparo_cuda_ptq1_tc::launch(w,x,
                reinterpret_cast<float*>(yb),k,rows,m,n,base,g.stream);
            if (err != cudaSuccess) {
                set_pending(CUDA_RC_ERROR,"PTQ prefill tensorcore launch");return;
            }
            const uint32_t route = tc_route;
            if (!(g.ptq_prefill_tensorcore_trace_mask & route)) {
                std::fprintf(stderr,"[imparo] ptq_prefill_tensorcore actual route k=%u n=%u m=%u tile=32x64x128 activation=f16 accumulate=f32\n",k,n,m);
                g.ptq_prefill_tensorcore_trace_mask |= route;
            }
            continue;
        }
        // A paged row chunk has a different destination stride; issue each token
        // against that chunk while it is resident. Full matrices batch normally.
        const uint32_t batch=rows==n?m:1;
        for(uint32_t token=0;token<m;token+=batch) {
            auto*y=reinterpret_cast<float*>(yb)+uint64_t(token)*n+base;
            const auto*xrow=x+uint64_t(token)*k;
            auto err=kind==39?imparo_cuda_ptq1::launch_ptq1_matmat(w,xrow,y,k,rows,batch,g.stream)
                :imparo_cuda_weight_basis::launch_bf16(w,xrow,y,k,rows,batch,g.stream);
            if(err!=cudaSuccess){set_pending(CUDA_RC_ERROR,"Bonsai matrix launch");return;}
        }
    }
    mark_buf_written(dst);
}
static void bonsai_row(uint64_t off,uint32_t width,uint32_t index,float scale,
        uint32_t dst,uint32_t dst_off) {
    uint8_t*yb=nullptr;const auto*basis=find_weight_basis(off);
    if(!width||width%128||!basis||!basis->inverse||basis->width!=width
            ||!buffer_slice(dst,uint64_t(dst_off)*4,uint64_t(width)*4,&yb)) {
        set_pending(CUDA_RC_INVALID,"Bonsai embedding basis/range");return;
    }
    const uint64_t rb=uint64_t(width/128)*28;
    if(uint64_t(index)>(UINT64_MAX-off)/rb){set_pending(CUDA_RC_INVALID,"embedding overflow");return;}
    const uint64_t row_off=off+uint64_t(index)*rb;
    if(row_off>g.weights_len||rb>g.weights_len-row_off){set_pending(CUDA_RC_INVALID,"embedding row bounds");return;}
    int rc=ensure_q8_scratch(uint64_t(width)*4);
    if(rc){set_pending(rc,"embedding basis scratch");return;}
    invalidate_q8_cache();const uint8_t*w=nullptr;
    const bool pinned=execution().forward_decode&&decode_pinned_input_enabled()
        &&(decode_graph_candidate()||!resident_weight_range(row_off,rb));
    if(pinned) {
        rc=ensure_weight_cache(rb);if(!rc)rc=ensure_decode_row_host_stage(rb);
        if(!rc) {
            std::memcpy(execution().decode_row_stage_host,g.weights_host+row_off,size_t(rb));
            if(cudaMemcpyAsync(execution().weight_cache,execution().decode_row_stage_host,
                    size_t(rb),cudaMemcpyHostToDevice,g.stream)!=cudaSuccess)rc=CUDA_RC_ERROR;
            w=static_cast<const uint8_t*>(execution().weight_cache);
            if(decode_graph_candidate()) {
                execution().decode_row_desc_valid=true;execution().decode_row_base=off;
                execution().decode_row_bytes=rb;
            }
        }
    }else rc=weight_slice(row_off,rb,&w);
    if(rc){set_pending(rc,"Bonsai embedding upload");return;}
    auto*raw=static_cast<float*>(execution().q8_scratch);
    if(imparo_cuda_ptq1::launch_ptq1_row(w,raw,width,1,0,scale,g.stream)!=cudaSuccess
            ||launch_weight_basis(*basis,raw,reinterpret_cast<float*>(yb),1)) {
        set_pending(CUDA_RC_ERROR,"Bonsai embedding inverse basis");return;
    }
    mark_buf_written(dst);
}
static void bonsai_rows(uint64_t off,uint32_t width,uint32_t table_rows,uint32_t ids,
        float scale,uint32_t dst,uint32_t count) {
    uint8_t*yb=nullptr;const auto*basis=find_weight_basis(off);
    if(!width||width%128||!count||count>65535||!table_rows||!basis
            ||!basis->inverse||basis->width!=width
            ||!buffer_slice(dst,0,uint64_t(count)*width*4,&yb)) {
        set_pending(CUDA_RC_INVALID,"Bonsai embedding batch geometry");return;
    }
    std::vector<uint32_t> fallback;const uint32_t*tokens=host_u32_values(ids,count,fallback);
    if(!tokens){set_pending(CUDA_RC_INVALID,"Bonsai embedding token IDs");return;}
    for(uint32_t i=0;i<count;++i)if(tokens[i]>=table_rows){set_pending(CUDA_RC_INVALID,"embedding token bounds");return;}
    const uint64_t rb=uint64_t(width/128)*28,bytes=rb*count;
    int rc=ensure_q8_scratch(uint64_t(count)*width*4);
    if(!rc)rc=ensure_weight_cache(bytes);
    if(!rc)rc=ensure_ple_host_stage(bytes);
    if(rc){set_pending(rc,"Bonsai embedding batch scratch");return;}
    invalidate_q8_cache();auto*staged=static_cast<uint8_t*>(execution().ple_stage_host);
    for(uint32_t i=0;i<count;++i) {
        if(uint64_t(tokens[i])>(UINT64_MAX-off)/rb){set_pending(CUDA_RC_INVALID,"embedding token overflow");return;}
        const uint64_t at=off+uint64_t(tokens[i])*rb;
        if(at>g.weights_len||rb>g.weights_len-at){set_pending(CUDA_RC_INVALID,"embedding batch bounds");return;}
        std::memcpy(staged+uint64_t(i)*rb,g.weights_host+at,size_t(rb));
    }
    if(cudaMemcpyAsync(execution().weight_cache,staged,size_t(bytes),cudaMemcpyHostToDevice,g.stream)!=cudaSuccess) {
        set_pending(CUDA_RC_ERROR,"embedding batch upload");return;
    }
    auto*raw=static_cast<float*>(execution().q8_scratch);
    for(uint32_t i=0;i<count;++i) {
        if(imparo_cuda_ptq1::launch_ptq1_row(static_cast<const uint8_t*>(execution().weight_cache)+uint64_t(i)*rb,
                raw+uint64_t(i)*width,width,1,0,scale,g.stream)!=cudaSuccess) {
            set_pending(CUDA_RC_ERROR,"embedding batch decode");return;
        }
    }
    if(launch_weight_basis(*basis,raw,reinterpret_cast<float*>(yb),count)) {
        set_pending(CUDA_RC_ERROR,"embedding batch inverse basis");return;
    }
    mark_buf_written(dst);
}

struct GatedDeltaWire {
    uint64_t a_off,dt_off;
    uint32_t qkv,alpha,beta,state,state_off,state_out_off,out,snap,snap_off,snap_row;
    uint32_t k_heads,v_heads,key_dim,value_dim,n_tok;float eps;
};
static_assert(sizeof(GatedDeltaWire)==80,"DeltaNet wire layout");
extern "C" int imparo_cuda_delta_net_run(const GatedDeltaWire*op) {
    if(!op||!imparo_cuda_gated_delta::supports(op->key_dim,op->value_dim)
            ||!op->k_heads||op->k_heads>4096||!op->v_heads||op->v_heads>4096
            ||op->v_heads%op->k_heads||!op->n_tok||op->n_tok>65535) {
        set_pending(CUDA_RC_INVALID,"CUDA DeltaNet shape");return CUDA_RC_INVALID;
    }
    const auto&o=*op;const uint64_t width=2ull*o.k_heads*o.key_dim+uint64_t(o.v_heads)*o.value_dim;
    const uint64_t state_bytes=uint64_t(o.v_heads)*o.value_dim*o.key_dim*4;
    uint8_t *q=nullptr,*a=nullptr,*bt=nullptr,*sin=nullptr,*sout=nullptr,*y=nullptr,*snap=nullptr;
    if(!buffer_slice(o.qkv,0,width*o.n_tok*4,&q)
            ||!buffer_slice(o.alpha,0,uint64_t(o.v_heads)*o.n_tok*4,&a)
            ||!buffer_slice(o.beta,0,uint64_t(o.v_heads)*o.n_tok*4,&bt)
            ||!buffer_slice(o.state,uint64_t(o.state_off)*4,state_bytes,&sin)
            ||!buffer_slice(o.state,uint64_t(o.state_out_off)*4,state_bytes,&sout)
            ||!buffer_slice(o.out,0,uint64_t(o.v_heads)*o.value_dim*o.n_tok*4,&y)
            ||(o.snap_row&&(!buffer_slice(o.snap,uint64_t(o.snap_off)*4,state_bytes,&snap)))) {
        set_pending(CUDA_RC_INVALID,"CUDA DeltaNet buffers");return CUDA_RC_INVALID;
    }
    // Each nonresident weight_slice reuses one staging address. Preserve A before
    // staging dt so both scalar vectors remain valid until the asynchronous kernel.
    int rc=ensure_q8_scratch(uint64_t(o.v_heads)*8);
    if(rc){set_pending(rc,"DeltaNet scalar scratch");return rc;}
    invalidate_q8_cache();auto*scalar=static_cast<float*>(execution().q8_scratch);
    const uint8_t*w=nullptr;rc=weight_slice(o.a_off,uint64_t(o.v_heads)*4,&w);
    if(!rc&&cudaMemcpyAsync(scalar,w,size_t(o.v_heads)*4,cudaMemcpyDeviceToDevice,g.stream)!=cudaSuccess)rc=CUDA_RC_ERROR;
    if(!rc)rc=weight_slice(o.dt_off,uint64_t(o.v_heads)*4,&w);
    if(!rc&&cudaMemcpyAsync(scalar+o.v_heads,w,size_t(o.v_heads)*4,cudaMemcpyDeviceToDevice,g.stream)!=cudaSuccess)rc=CUDA_RC_ERROR;
    if(rc){set_pending(rc,"DeltaNet scalar weights");return rc;}
    OpEventScope profile("gated_delta",o.v_heads,o.n_tok);
    auto err=imparo_cuda_gated_delta::launch_delta(reinterpret_cast<const float*>(q),
        reinterpret_cast<const float*>(a),reinterpret_cast<const float*>(bt),scalar,scalar+o.v_heads,
        reinterpret_cast<const float*>(sin),reinterpret_cast<float*>(sout),reinterpret_cast<float*>(y),
        reinterpret_cast<float*>(snap),o.snap_row,o.k_heads,o.v_heads,o.key_dim,o.value_dim,o.n_tok,o.eps,g.stream);
    if(err!=cudaSuccess){set_pending(CUDA_RC_ERROR,"DeltaNet launch");return CUDA_RC_ERROR;}
    mark_buf_written(o.out);mark_buf_written(o.state);if(snap)mark_buf_written(o.snap);return 0;
}
extern "C" void imparo_cuda_plain_conv(uint32_t src,uint64_t w_off,uint32_t state,
        uint32_t state_off,uint32_t state_out_off,uint32_t out,uint32_t width,uint32_t kernel,uint32_t tokens) {
    if(!width||!tokens||kernel<2||kernel>32){set_pending(CUDA_RC_INVALID,"plain conv shape");return;}
    const uint64_t history=uint64_t(width)*(kernel-1)*4,count=uint64_t(width)*tokens*4;
    uint8_t*x=nullptr,*sin=nullptr,*sout=nullptr,*y=nullptr;
    if(!buffer_slice(src,0,count,&x)||!buffer_slice(state,uint64_t(state_off)*4,history,&sin)
            ||!buffer_slice(state,uint64_t(state_out_off)*4,history,&sout)||!buffer_slice(out,0,count,&y)) {
        set_pending(CUDA_RC_INVALID,"plain conv buffers");return;
    }
    const uint8_t*w=nullptr;int rc=weight_slice(w_off,uint64_t(width)*kernel*4,&w);
    if(rc){set_pending(rc,"plain conv weights");return;}
    auto err=imparo_cuda_gated_delta::launch_plain_conv(reinterpret_cast<const float*>(x),
        reinterpret_cast<const float*>(w),reinterpret_cast<const float*>(sin),reinterpret_cast<float*>(sout),
        reinterpret_cast<float*>(y),width,kernel,tokens,g.stream);
    if(err!=cudaSuccess){set_pending(CUDA_RC_ERROR,"plain conv launch");return;}
    mark_buf_written(out);mark_buf_written(state);
}
extern "C" void imparo_cuda_plain_conv_snapshot(uint32_t src,uint32_t state,uint32_t state_off,
        uint32_t snap,uint32_t snap_off,uint32_t width,uint32_t kernel,uint32_t tokens) {
    if(!width||!tokens||kernel<2||kernel>32){set_pending(CUDA_RC_INVALID,"plain conv snapshot shape");return;}
    const uint64_t history=uint64_t(width)*(kernel-1)*4;
    uint8_t*x=nullptr,*sin=nullptr,*y=nullptr;
    if(!buffer_slice(src,0,uint64_t(width)*tokens*4,&x)||!buffer_slice(state,uint64_t(state_off)*4,history,&sin)
            ||!buffer_slice(snap,uint64_t(snap_off)*4,history,&y)) {set_pending(CUDA_RC_INVALID,"plain conv snapshot buffers");return;}
    auto err=imparo_cuda_gated_delta::launch_plain_conv_snapshot(reinterpret_cast<const float*>(x),
        reinterpret_cast<const float*>(sin),reinterpret_cast<float*>(y),width,kernel,tokens,g.stream);
    if(err!=cudaSuccess){set_pending(CUDA_RC_ERROR,"plain conv snapshot launch");return;}mark_buf_written(snap);
}
extern "C" void imparo_cuda_mul_strided_sigmoid(uint32_t a,uint32_t b,uint32_t width,
        uint32_t b_off,uint32_t b_stride,uint32_t a_stride,uint32_t rows) {
    uint8_t*x=nullptr,*y=nullptr;
    if(!width||!rows||a_stride<width||b_stride<width
            ||!buffer_slice(a,0,(uint64_t(rows-1)*a_stride+width)*4,&x)
            ||!buffer_slice(b,0,(uint64_t(rows-1)*b_stride+b_off+width)*4,&y)) {
        set_pending(CUDA_RC_INVALID,"strided sigmoid buffers");return;
    }
    auto err=imparo_cuda_gated_delta::launch_mul_sigmoid(reinterpret_cast<float*>(x),reinterpret_cast<const float*>(y),
        width,b_off,b_stride,a_stride,rows,g.stream);
    if(err!=cudaSuccess){set_pending(CUDA_RC_ERROR,"strided sigmoid launch");return;}mark_buf_written(a);
}
