#pragma once

extern "C" uint32_t imparo_cuda_cobatch_version(void) { return 1; }

struct CudaSlotRowWire { uint32_t slot, pos; };
struct CudaRowCache {
    void * k = nullptr; void * v = nullptr;
    const uint32_t * pages = nullptr;
};

// Temporary host views only: the owner and selected conversation never change.
// All queued CUDA work captures concrete pointers before these bindings restore.
struct CobatchBufferView {
    ExecutionState & e; uint32_t id; void * pointer; uint64_t bytes;
    CobatchBufferView(uint32_t bid, uint64_t offset, uint64_t size)
        : e(execution()), id(bid), pointer(e.bufs[bid]), bytes(e.sizes[bid]) {
        e.bufs[id] = static_cast<uint8_t *>(pointer) + offset;
        e.sizes[id] = size;
    }
    ~CobatchBufferView() { e.bufs[id] = pointer; e.sizes[id] = bytes; }
};

// Attention is an independent M1 decode for each row, even though projections
// belong to a multi-row forward. Restore the outer phase on every return.
struct CobatchDecodeView {
    ExecutionState & e; bool decode;
    CobatchDecodeView() : e(execution()), decode(e.forward_decode) { e.forward_decode = true; }
    ~CobatchDecodeView() { e.forward_decode = decode; }
};

static uint64_t cobatch_stride(uint32_t width, uint32_t type) {
    if (type == 1) return uint64_t(width)*2;
    if (width % 32) return 0;
    return type == 2 ? uint64_t(width/32)*18 : type == 8 ? uint64_t(width/32)*34 : 0;
}

static bool cobatch_cache(uint32_t slot, uint32_t layer, uint32_t pos,
        uint32_t width, uint32_t ring, CudaRowCache & out) {
    auto & e = execution(); auto & slots = e.conversation_slots;
    if (slot >= slots.inactive.size() || !slots.inactive[slot].made
            || layer >= e.kv_layout.layers || !width || pos == UINT32_MAX) return false;
    const bool local = slot == slots.selected;
    const auto & saved = slots.inactive[slot];
    const auto & tables = local ? e.kv_page_tables : saved.pages;
    const void * arena = local ? e.kv_page_table_arena : saved.pages_device;
    const bool is_ring = std::find(slots.rings.begin(), slots.rings.end(), layer) != slots.rings.end();
    const uint64_t ks = cobatch_stride(width,g.kv_type_k), vs = cobatch_stride(width,g.kv_type_v);
    if (!ks || !vs || bool(ring) != is_ring || (ring && (
            (uint64_t(ring)+1)*ks > e.kv_bytes[layer]
            || (uint64_t(ring)+1)*vs > e.kv_bytes[layer]))) return false;
    out.k = is_ring && !local ? saved.ring_k[layer] : e.kv_k[layer];
    out.v = is_ring && !local ? saved.ring_v[layer] : e.kv_v[layer];
    if (!out.k || !out.v || layer >= tables.layers) return false;
    const auto & table = tables.layer[layer];
    if (!ring) {
        if (pos/64 >= table.capacity || pos/64 >= table.host_shadow.size() || !arena) return false;
        const uint64_t end = uint64_t(table.host_shadow[pos/64])*64 + pos%64 + 1;
        if (end*ks > e.kv_bytes[layer] || end*vs > e.kv_bytes[layer]) return false;
        out.pages = static_cast<const uint32_t *>(arena) + table.arena_offset;
    }
    return true;
}

struct CobatchCacheView {
    ExecutionState & e; uint32_t layer; void * k; void * v; void * pages;
    CudaConversationSlot * foreign;
    CobatchCacheView(uint32_t slot, uint32_t l, const CudaRowCache & cache)
        : e(execution()), layer(l), k(e.kv_k[l]), v(e.kv_v[l]), pages(e.kv_page_table_arena),
          foreign(slot == e.conversation_slots.selected ? nullptr : &e.conversation_slots.inactive[slot]) {
        e.kv_k[l] = cache.k; e.kv_v[l] = cache.v;
        if (foreign) {
            std::swap(e.kv_page_tables.layer[l], foreign->pages.layer[l]);
            e.kv_page_table_arena = foreign->pages_device;
        }
        invalidate();
    }
    void invalidate() {
        e.kdq.clear(); e.vdq.clear(); e.attention_q_src = UINT32_MAX;
    }
    ~CobatchCacheView() {
        e.kv_k[layer] = k; e.kv_v[layer] = v;
        if (foreign) std::swap(e.kv_page_tables.layer[layer], foreign->pages.layer[layer]);
        e.kv_page_table_arena = pages;
        invalidate();
    }
};

extern "C" int imparo_cuda_cobatch_route(uint32_t route) {
    auto & e = execution();
    if (route > 1 || e.graph_capturing || e.tree.nodes || (route && e.pending_error)) return CUDA_RC_INVALID;
    e.cobatch_rows = route != 0;
    return 0;
}

static bool cobatch_head_valid(uint32_t hd,float eps,uint32_t heads,const uint32_t * pos,
        uint32_t count,uint32_t rd,float base,uint32_t nrot) {
    if (!execution().cobatch_rows || !count || count>64 || !pos || !heads || !hd || hd>1024
            || uint64_t(heads)*hd>UINT32_MAX || !rd || rd>hd || rd%2
            || !std::isfinite(base) || base<=0 || !std::isfinite(eps) || eps<=0
            || (nrot && (nrot>hd || (nrot&(nrot-1)) || hd%nrot))) return false;
    for (uint32_t r=0;r<count;++r) if (pos[r]==UINT32_MAX) return false;
    return true;
}

extern "C" int imparo_cuda_cobatch_head(uint32_t buf,uint64_t weight,
        uint32_t hd,float eps,uint32_t heads,const uint32_t * pos,uint32_t count,
        uint32_t rd,float base,const float * freqs,uint32_t nrot) {
    uint8_t * data=nullptr;
    const uint64_t bytes=uint64_t(heads)*hd*4;
    if (!cobatch_head_valid(hd,eps,heads,pos,count,rd,base,nrot)
            || !buffer_slice(buf,0,count*bytes,&data) || weight==UINT64_MAX) return CUDA_RC_INVALID;
    const uint8_t * norm=nullptr;
    const int rc=weight_slice(weight,uint64_t(hd)*4,&norm); if(rc) return rc;
    for (uint32_t r=0;r<count;++r) {
        CobatchBufferView view(buf,r*bytes,bytes);
        imparo_cuda_head_norm_rope_hadamard(buf,weight,hd,eps,heads,pos[r],1,rd,base,freqs,nrot);
        if (execution().pending_error) return execution().pending_error;
    }
    mark_buf_written(buf);
    return cudaPeekAtLastError()==cudaSuccess ? 0 : CUDA_RC_ERROR;
}

extern "C" int imparo_cuda_cobatch_kv_head(uint32_t k,uint32_t v,uint64_t weight,
        uint32_t hd,float eps,uint32_t heads,const uint32_t * pos,uint32_t count,
        uint32_t rd,float base,const float * freqs,uint32_t hk,uint32_t hv) {
    uint8_t * data=nullptr;
    const uint64_t bytes=uint64_t(heads)*hd*4;
    if (k==v || !cobatch_head_valid(hd,eps,heads,pos,count,rd,base,hk)
            || !cobatch_head_valid(hd,eps,heads,pos,count,rd,base,hv)
            || !buffer_slice(k,0,count*bytes,&data) || !buffer_slice(v,0,count*bytes,&data)
            || weight==UINT64_MAX) return CUDA_RC_INVALID;
    const uint8_t * norm=nullptr;
    const int rc=weight_slice(weight,uint64_t(hd)*4,&norm); if(rc) return rc;
    for(uint32_t r=0;r<count;++r) {
        CobatchBufferView key(k,r*bytes,bytes), value(v,r*bytes,bytes);
        imparo_cuda_kv_head_postprocess(k,v,weight,hd,eps,heads,pos[r],1,rd,base,freqs,hk,hv);
        if(execution().pending_error) return execution().pending_error;
    }
    mark_buf_written(k); mark_buf_written(v);
    return cudaPeekAtLastError()==cudaSuccess ? 0 : CUDA_RC_ERROR;
}

extern "C" int imparo_cuda_cobatch_store(uint32_t src,uint32_t layer,uint32_t width,
        const CudaSlotRowWire * rows,uint32_t count,uint32_t is_v,uint32_t ring) {
    if (!execution().cobatch_rows || !count || count>64 || !rows || is_v>1) return CUDA_RC_INVALID;
    uint8_t * data=nullptr; CudaRowCache cache[64];
    if (!buffer_slice(src,0,uint64_t(count)*width*4,&data)) return CUDA_RC_INVALID;
    for(uint32_t r=0;r<count;++r) {
        if(!cobatch_cache(rows[r].slot,layer,rows[r].pos,width,ring,cache[r])) return CUDA_RC_INVALID;
        for(uint32_t j=0;j<r;++j) if(rows[j].slot==rows[r].slot) return CUDA_RC_INVALID;
    }
    for(uint32_t r=0;r<count;++r) {
        CobatchBufferView view(src,uint64_t(r)*width*4,uint64_t(width)*4);
        CobatchCacheView owner(rows[r].slot,layer,cache[r]);
        imparo_cuda_kv_store(src,layer,width,rows[r].pos,1,is_v,ring);
        if(execution().pending_error) return execution().pending_error;
    }
    return cudaPeekAtLastError()==cudaSuccess ? 0 : CUDA_RC_ERROR;
}

extern "C" int imparo_cuda_cobatch_attention(uint32_t layer,uint32_t hd,uint32_t heads,
        uint32_t kvheads,uint32_t width,float scale,uint32_t window,
        const CudaSlotRowWire * rows,uint32_t count,uint32_t ring) {
    auto & e=execution();
    if(!e.cobatch_rows || !rows || !count || count>64 || !kvheads || !heads
            || heads%kvheads || !hd || hd>1024 || uint64_t(kvheads)*hd!=width
            || uint64_t(heads)*hd>UINT32_MAX || !std::isfinite(scale) || scale<=0) return CUDA_RC_INVALID;
    constexpr uint32_t qbuf=2,obuf=5,kdq=17,vdq=18;
    const uint64_t bytes=uint64_t(heads)*hd*4;
    uint8_t * data=nullptr; CudaRowCache cache[64];
    if(!buffer_slice(qbuf,0,count*bytes,&data) || !buffer_slice(obuf,0,count*bytes,&data)) return CUDA_RC_INVALID;
    for(uint32_t r=0;r<count;++r)
        if(!cobatch_cache(rows[r].slot,layer,rows[r].pos,width,ring,cache[r])) return CUDA_RC_INVALID;
    for(uint32_t r=0;r<count;++r) {
        CobatchBufferView query(qbuf,r*bytes,bytes), output(obuf,r*bytes,bytes);
        CobatchCacheView owner(rows[r].slot,layer,cache[r]);
        CobatchDecodeView decode;
        // Mirror Backend::attention: scale Q before native conversion-order policy.
        if(scale!=1.0f) imparo_cuda_scale(qbuf,scale,heads*hd);
        imparo_cuda_attention(layer,hd,heads,kvheads,width,rows[r].pos,scale,window,1,ring,qbuf,obuf,kdq,vdq);
        if(e.pending_error) return e.pending_error;
    }
    mark_buf_written(obuf);
    return cudaPeekAtLastError()==cudaSuccess ? 0 : CUDA_RC_ERROR;
}


struct CudaSlotStateRowWire { uint32_t slot, state_off, state_out_off; };
struct CobatchStateBinding { void * pointer; uint64_t bytes; };

static bool cobatch_states(uint32_t state, const CudaSlotStateRowWire * rows,
        uint32_t count, uint64_t bytes, CobatchStateBinding * bindings) {
    auto & e=execution(); const auto & slots=e.conversation_slots;
    if (!e.cobatch_rows || !rows || !count || count>64 || !bytes
            || (state!=kConversationBuffers[0] && state!=kConversationBuffers[1])) return false;
    const unsigned index=state==kConversationBuffers[0] ? 0 : 1;
    for (uint32_t r=0;r<count;++r) {
        if(rows[r].slot>=slots.inactive.size() || !slots.inactive[rows[r].slot].made) return false;
        for(uint32_t j=0;j<r;++j) if(rows[j].slot==rows[r].slot) return false;
        const bool local=rows[r].slot==slots.selected;
        const auto & slot=slots.inactive[rows[r].slot];
        bindings[r]={local ? e.bufs[state] : slot.buffers[index],
                     local ? e.sizes[state] : slot.sizes[index]};
        const uint64_t start=uint64_t(rows[r].state_off)*4, out=uint64_t(rows[r].state_out_off)*4;
        if(!bindings[r].pointer || start>bindings[r].bytes || bytes>bindings[r].bytes-start
                || out>bindings[r].bytes || bytes>bindings[r].bytes-out
                || (start!=out && (start<out ? out-start : start-out)<bytes)) return false;
    }
    return true;
}

struct CobatchStateView {
    ExecutionState & e; uint32_t id; CobatchStateBinding saved;
    CobatchStateView(uint32_t state,const CobatchStateBinding & binding)
        : e(execution()),id(state),saved{e.bufs[state],e.sizes[state]} {
        e.bufs[state]=binding.pointer;e.sizes[state]=binding.bytes;
    }
    ~CobatchStateView(){e.bufs[id]=saved.pointer;e.sizes[id]=saved.bytes;}
};

extern "C" int imparo_cuda_cobatch_conv(uint32_t form,uint32_t src,uint64_t weight,
        uint32_t state,const CudaSlotStateRowWire * rows,uint32_t count,uint32_t out,
        uint32_t width,uint32_t kernel) {
    imparo_cuda_lfm2::ShortconvLayout layout;
    if(form>1 || (form==1 && kernel>32) || src==out || src==state || out==state
            || !imparo_cuda_lfm2::checked_shortconv_layout(width,kernel,1,&layout)
            || !valid_launch_count(width) || !valid_launch_count(layout.state_bytes/4)) return CUDA_RC_INVALID;
    const uint64_t input=form==0 ? layout.bcx_bytes : layout.output_bytes;
    CobatchStateBinding bindings[64];uint8_t * data=nullptr;
    if(!cobatch_states(state,rows,count,layout.state_bytes,bindings)
            || !buffer_slice(src,0,count*input,&data)
            || !buffer_slice(out,0,count*layout.output_bytes,&data)) return CUDA_RC_INVALID;
    // Preflight every row before even the first history plane is copied/advanced.
    const uint8_t * weights=nullptr;
    const int rc=weight_slice(weight,layout.weight_bytes,&weights);if(rc)return rc;
    for(uint32_t r=0;r<count;++r) {
        CobatchBufferView x(src,r*input,input), y(out,r*layout.output_bytes,layout.output_bytes);
        CobatchStateView history(state,bindings[r]);
        if(form==1) imparo_cuda_plain_conv(src,weight,state,rows[r].state_off,
            rows[r].state_out_off,out,width,kernel,1);
        else {
            if(rows[r].state_off!=rows[r].state_out_off)
                imparo_cuda_copy_range(state,rows[r].state_out_off,state,rows[r].state_off,uint32_t(layout.state_bytes/4));
            imparo_cuda_shortconv(src,weight,state,rows[r].state_out_off,out,width,kernel,1);
        }
        if(execution().pending_error)return execution().pending_error;
    }
    mark_buf_written(out);mark_buf_written(state);
    return cudaPeekAtLastError()==cudaSuccess ? 0 : CUDA_RC_ERROR;
}

extern "C" int imparo_cuda_cobatch_delta(const GatedDeltaWire * op,
        const CudaSlotStateRowWire * rows,uint32_t count) {
    if(!op || !imparo_cuda_gated_delta::supports(op->key_dim,op->value_dim)
            || !op->k_heads || op->k_heads>4096 || !op->v_heads || op->v_heads>4096
            || op->v_heads%op->k_heads || !std::isfinite(op->eps) || op->eps<=0) return CUDA_RC_INVALID;
    // Distinct bindings, as in the existing workflow's QKV/alpha/beta/output contract.
    const uint32_t ids[]={op->qkv,op->alpha,op->beta,op->out,op->state};
    for(unsigned i=0;i<5;++i)for(unsigned j=0;j<i;++j)if(ids[i]==ids[j])return CUDA_RC_INVALID;
    const uint64_t qbytes=(2ull*op->k_heads*op->key_dim+uint64_t(op->v_heads)*op->value_dim)*4;
    const uint64_t scalars=uint64_t(op->v_heads)*4, output=scalars*op->value_dim;
    CobatchStateBinding bindings[64];uint8_t * data=nullptr;
    if(!cobatch_states(op->state,rows,count,output*op->key_dim,bindings)
            || !buffer_slice(op->qkv,0,count*qbytes,&data)
            || !buffer_slice(op->alpha,0,count*scalars,&data)
            || !buffer_slice(op->beta,0,count*scalars,&data)
            || !buffer_slice(op->out,0,count*output,&data)) return CUDA_RC_INVALID;
    const uint8_t * weights=nullptr;
    int rc=weight_slice(op->a_off,scalars,&weights);if(rc)return rc;
    rc=weight_slice(op->dt_off,scalars,&weights);if(rc)return rc;
    for(uint32_t r=0;r<count;++r) {
        CobatchBufferView q(op->qkv,r*qbytes,qbytes),a(op->alpha,r*scalars,scalars),
            b(op->beta,r*scalars,scalars),y(op->out,r*output,output);
        CobatchStateView state(op->state,bindings[r]);
        GatedDeltaWire row=*op;row.n_tok=1;row.state_off=rows[r].state_off;
        row.state_out_off=rows[r].state_out_off;row.snap=UINT32_MAX;row.snap_off=row.snap_row=0;
        rc=imparo_cuda_delta_net_run(&row);if(rc)return rc;
    }
    mark_buf_written(op->out);mark_buf_written(op->state);
    return cudaPeekAtLastError()==cudaSuccess ? 0 : CUDA_RC_ERROR;
}
