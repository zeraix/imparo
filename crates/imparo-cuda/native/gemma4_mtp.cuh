#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include "native_replay_graph.cuh"
#include "sm86/mtp_cluster_head_lab.cuh"

extern "C" uint32_t imparo_cuda_e4b_retained_decode_policy();
extern "C" uint32_t imparo_cuda_e4b_retained_decode_domain();

// Gemma4 assistant: the target owns every KV byte. Only ordinary activations,
// captured post-output-norm target rows and assistant continuation hidden belong here.
// All mathematical operations use the existing native backend and tuner choices.
namespace imparo_gemma4_mtp {
// Rust registers authenticated include_bytes slices with process-long lifetime.
// These host addresses never become owners of an execution buffer.
struct ClusterAssetBytes {
    const uint8_t *centroids=nullptr;
    const uint8_t *ordering=nullptr;
};
static ClusterAssetBytes embedded_cluster_assets;

// Optional assistant confidence only; token selection remains the original argmax.
// Reuse its return buffer and host synchronization. No target logits/state change.
__global__ void first_probability(const float *logits, const uint32_t *token,
                                 float *probability, uint32_t n) {
    __shared__ float sums[1024];
    const float best=logits[*token];
    float sum=0.0f;
    for(uint32_t i=threadIdx.x;i<n;i+=blockDim.x)sum+=__expf(logits[i]-best);
    sums[threadIdx.x]=sum;__syncthreads();
    for(uint32_t stride=512;stride;stride>>=1){
        if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0)*probability=1.0f/sums[0];
}

struct Config {
    uint64_t target_embedding, pre_proj, post_proj, out_norm, head;
    uint32_t hidden, target_hidden, ffn, vocab, heads, kv_heads;
    uint32_t batch_capacity, target_final_layer, target_embedding_kind, reserved;
    float eps, embedding_scale;
};
struct Layer {
    uint64_t attn_norm, q, qn, o, attn_post_norm, ffn_norm;
    uint64_t gate, up, down, ffn_post_norm, rope_freqs;
    uint32_t target_kv_layer, head_dim, rope_dim, window, ring, had_k, had_v, has_rope_freqs;
    float rope_theta, out_scale;
};
static_assert(sizeof(Config)==88 && sizeof(Layer)==128,"Gemma4 MTP wire layout");
constexpr uint32_t X=0, CUR=1, Q=2, ATTN=5, O=6, GATE=7, UP=8;
constexpr uint32_t LOGITS=12, TOK=13, FEATURE=20, COMMITTED=21, NEXT=22, CONCAT=23, KDQ=24, VDQ=25, STEP_SAVED=26;
// Slots below 28 include the shared BufId::Pick appended by v2.
// These three slots belong only to the isolated assistant execution owner.
constexpr uint32_t CLUSTER_DATA=28, CLUSTER_SCORES=29, CLUSTER_IDS=30;
struct Session {
    Config cfg;
    std::vector<Layer> layers;
    uint64_t target_owner=0, draft_owner=0, generation=0, feature_generation=0;
    const uint8_t *weights_host=nullptr;
    const void *weights_device=nullptr;
    uint64_t weights_len=0, choice_epoch=0;
    uint32_t kv_type_k=0, kv_type_v=0, S=0, feature_start=0, feature_rows=0, hidden_position=0;
    bool capture_enabled=false, poisoned=false, hidden_valid=false, borrowed=false;
    NativeReplayGraph step_graph;
    bool cluster_head_enabled=false;
    bool step_proof_done=false;
    uint64_t step_updates=0,step_instantiates=0;
    uint32_t step_min=0,step_max=0;
    std::vector<const void*> step_storage;
    Session(const Config &c,const Layer *l,uint32_t n):cfg(c),layers(l,l+n){}
    void clear_step_graphs(){step_graph.clear();step_storage.clear();}
    bool step_graph_enabled()const{const char*v=std::getenv("IMPARO_LAB_MTP_CACHED_GRAPH");return imparo_cuda_e4b_retained_decode_policy()==1||(v&&std::strcmp(v,"1")==0);}
    std::vector<const void*> storage_signature(){
        auto&e=execution();std::vector<const void*> p;
        for(auto v:e.bufs)p.push_back(v);
        for(const auto&l:layers){p.push_back(e.kv_k[l.target_kv_layer]);p.push_back(e.kv_v[l.target_kv_layer]);}
        for(auto v:{e.q8_scratch,e.q8_scratch_next,e.attention_scratch,e.attention_q_cache,e.rope_freqs})p.push_back(v);
        p.push_back(e.kv_page_table_arena);p.push_back(g.weights);return p;
    }
    void configure_step(uint32_t start){
        step_min=start;step_max=UINT32_MAX-1;unsigned heads=0,scores=0,values=0,softmax=0;uint32_t attention_ring=0,attention_span=0;
        for(auto node:step_graph.ordered_nodes()){
            cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel){req(type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeEmpty,"unexpected assistant graph node");continue;}
            cudaKernelNodeParams p={};ck(cudaGraphKernelNodeGetParams(node,&p));req(p.kernelParams,"missing kernel arguments");
            auto is=[&](void*f){return graph_kernel_is(p.func,f);};
            auto u=[&](uint32_t i){return *static_cast<const uint32_t*>(p.kernelParams[i]);};
            unsigned argc=0,si=UINT32_MAX,vi=UINT32_MAX,span=UINT32_MAX;uint32_t ring=0;bool dq=false;
            if(is((void*)k_head_norm_rope_hadamard<256,true>)||is((void*)k_head_norm_rope_hadamard<256,false>)){
                argc=13;si=6;++heads;req(u(si)==start,"assistant rope start");
            }else if(is((void*)imparo_sm86_d512_pipeline::partial_q4)
                    ||is((void*)imparo_sm86_d512_pipeline::partial_q4_pvf32)){
                // This provider fuses score, softmax and value accumulation; only start/valid vary.
                argc=13;si=7;vi=10;
                req(u(4)==8&&u(5)==2&&u(6)==1024&&u(7)==start-1
                    &&u(8)==0&&u(9)==0&&u(10)==start,"assistant fused geometry");
                req(u(12)==2*unsigned(g.sm_count)&&u(11)*32>=start,"assistant fused schedule");
                const uint64_t needed=4096*sizeof(__half)+4096*sizeof(float)
                    +imparo_sm80_d512_decode::workspace_floats(u(12))*sizeof(float);
                req(needed<=execution().attention_scratch_bytes,"assistant fused workspace");
                step_max=std::min(step_max,u(11)*32);
                ++scores;++softmax;++values;
            }else if(is((void*)imparo_sm80_d512_small::scores<256,1,2,false>)||is((void*)imparo_sm80_d512_small::scores<512,2,2,false>)){
                argc=16;si=7;vi=12;span=13;ring=u(11);++scores;
                req(u(7)==start-1&&u(10)==1&&p.gridDim.z==1,"assistant score geometry");
                req(u(vi)==(ring?std::min(start,ring+1):start),"assistant valid span");
                attention_ring=ring;attention_span=u(span);
                const uint32_t upper=ring&&start<ring+1?std::min(((start+31)/32)*32,ring+1):ring?UINT32_MAX-1:u(span);
                step_max=std::min(step_max,upper);
                const uint32_t maxspan=ring?std::min(upper,ring+1):u(span);
                const uint64_t needed=imparo_sm80_d512_small::block_stride(maxspan,u(14))*sizeof(float)*p.gridDim.x;
                req(needed<=execution().attention_scratch_bytes,"assistant workspace bucket capacity");
            }else if(is((void*)imparo_sm80_d512_small::softmax_parts)){
                argc=3;span=1;ring=attention_ring;++softmax;req(scores==softmax&&u(1)==attention_span,"assistant softmax dependency");
            }else if(is((void*)imparo_sm80_d512_small::values_combine<256,1,1,2,false>)
                    ||is((void*)imparo_sm80_d512_small::values_combine<512,2,1,2,false>)
                    ||is((void*)imparo_sm80_d512_small::values_combine<512,2,1,2,false,true>)
                    ||is((void*)imparo_sm80_d512_small::values_combine<512,2,2,2,false,true>)){
                argc=13;vi=9;span=10;ring=u(8);++values;
                req(scores==values&&ring==attention_ring&&u(span)==attention_span,"assistant values dependency");
            }else if(is((void*)imparo_sm80_kv::dequant_parallel<2>)){
                argc=6;vi=3;ring=u(4);dq=true;req(u(vi)==(ring?std::min(start,ring+1):start),"assistant dequant extent");
            }
            if(!argc)continue;
            DynamicGraphNode d;d.node=node;d.params=p;d.args.assign(p.kernelParams,p.kernelParams+argc);d.ring=ring;
            d.start_index=si;d.valid_index=vi;d.span_index=span;d.span_follows_valid=ring!=0;
            if(si!=UINT32_MAX)d.start_delta=u(si)-start;
            if(span!=UINT32_MAX)d.span_value=u(span);
            if(dq)d.native_update=NativeGraphUpdate::ValidGridY;
            step_graph.nodes.push_back(std::move(d));
        }
        req(heads==layers.size()&&scores==layers.size()&&values==layers.size()&&softmax==layers.size(),"assistant graph dynamic coverage");
        req(step_max>=step_min,"assistant graph range");
    }
    template<class Body>void submit_step(uint32_t start,bool eager,Body body){
        if(eager||!step_graph_enabled()){body();return;}
        if(step_graph.exec&&start>=step_min&&start<=step_max&&step_storage==storage_signature()){
            ck(step_graph.replay(start,g.stream));++step_updates;
        }else{
            clear_step_graphs();
            // Eager warmup reserves ordinary owner scratch before stream capture.
            // NEXT is a loop-carried input; save/restore it around that warmup.
            imparo_cuda_copy_range(STEP_SAVED,0,NEXT,0,cfg.target_hidden);
            body();rc(imparo_cuda_end());
            imparo_cuda_begin();imparo_cuda_copy_range(NEXT,0,STEP_SAVED,0,cfg.target_hidden);
            // Token is also loop-carried; the caller staged it before this method.
            // body argmax overwrote TOK, so restore the original device token below.
            req(step_input<cfg.vocab,"assistant graph input token");imparo_cuda_write_u32(TOK,0,&step_input,1);
            execution().kdq.clear();execution().vdq.clear();invalidate_q8_cache();
            ck(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal));
            cudaGraph_t graph=nullptr;
            try{body();}catch(...){cudaStreamEndCapture(g.stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
            const auto ended=cudaStreamEndCapture(g.stream,&graph);
            if(ended!=cudaSuccess||!graph){if(graph)cudaGraphDestroy(graph);ck(ended);req(false,"assistant graph capture failed");}
            step_graph.graph=graph;configure_step(start);ck(cudaGraphInstantiate(&step_graph.exec,graph,nullptr,nullptr,0));++step_instantiates;
            step_storage=storage_signature();ck(step_graph.replay(start,g.stream));
        }
        invalidate_q8_cache();execution().kdq.clear();execution().vdq.clear();execution().attention_q_src=UINT32_MAX;
        for(uint32_t i=0;i<B_COUNT;++i)mark_buf_written(i);
        if(step_updates==1&&std::getenv("IMPARO_DEVICE_GREEDY_VERIFY_TRACE"))std::fprintf(stderr,"[mtp-cached-graph] replay=1 dynamic=%zu range=%u..%u\n",step_graph.nodes.size(),step_min,step_max);
    }
    uint32_t step_input=0;
    void req(bool b,const char *m) { if(!b) throw std::runtime_error(m); }
    void rc(int v) { if(v) throw std::runtime_error("native status "+std::to_string(v)); }
    void ck(cudaError_t v) { if(v!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(v)); }
    ExecutionState *target() { return find_execution_owner(target_owner); }
    ExecutionState *draft() { return draft_owner ? find_execution_owner(draft_owner) : nullptr; }
    void identity() {
        req(target() && draft(),"execution owner lifetime changed");
        req(g.weights_host==weights_host && g.weights==weights_device && g.weights_len==weights_len,
            "composite weight lifetime changed");
        req(g.choice_epoch==choice_epoch && g.kv_type_k==kv_type_k && g.kv_type_v==kv_type_v,
            "tuner/KV generation changed; detach and reattach");
    }
    void boundary() {
        identity();
        req(active_execution_owner_id==target_owner && execution_boundary_closed(),"target boundary is not closed");
    }
    void span(uint64_t off,uint64_t bytes) {
        req(off<=weights_len && bytes<=weights_len-off,"weight span outside composite mapping");
    }
    void q8(uint64_t off,uint32_t ni,uint32_t no) {
        req(ni && no && ni%32==0,"invalid Q8 matrix");
        const uint64_t bytes=uint64_t(ni/32)*34*no; span(off,bytes);
        req(resident_weight_range(off,bytes)!=nullptr,"assistant Q8 weights must be resident");
    }
    void mm(uint64_t off,uint32_t ni,uint32_t no,uint32_t src,uint32_t dst) {
        imparo_cuda_matmat(2,off,ni,no,src,dst,1,0);
    }
    void norm(uint32_t dst,uint32_t src,uint64_t off) {
        imparo_cuda_rms_norm_project(dst,src,off,cfg.hidden,cfg.eps,1,cfg.hidden,0);
    }
    void dense_norm(uint32_t dst,uint32_t src,uint64_t off) {
        imparo_cuda_rms_norm(dst,src,off,cfg.hidden,cfg.eps,1,cfg.hidden,0,1);
    }
    void clear_borrow() noexcept {
        auto *d=draft(); if(!d)return;
        // No borrowed pointer ever becomes a storage owner, including error paths.
        for(uint32_t i=0;i<MAX_LAYERS;++i) { d->kv_k[i]=nullptr;d->kv_v[i]=nullptr;d->kv_bytes[i]=0; }
        d->kv_page_table_arena=nullptr;
        d->kv_page_tables.layers=0;d->kv_page_tables.arena_entries=0;
        d->kv_layout.layers=0;d->kv_layout.arena_bytes=0;d->kv_layout.live_bytes=0;
        d->kdq.clear();d->vdq.clear();
        borrowed=false;
    }
    void borrow_kv(uint32_t start) {
        auto *t=target();auto *d=draft();req(t&&d,"missing owner for KV borrow");
        req(!d->kv_arena && !d->kv_page_table_arena && !borrowed,"draft unexpectedly owns KV");
        // Copy only host descriptors; arena pointers are temporary read-only views.
        // Native attention dequant writes assistant-owned KDQ/VDQ, never source KV.
        d->kv_page_tables=t->kv_page_tables;
        d->kv_page_table_arena=t->kv_page_table_arena;
        d->kv_layout.layers=t->kv_layout.layers;
        borrowed=true;
        for(const auto &l:layers) {
            const auto k=l.target_kv_layer;
            req(k<t->kv_layout.layers && t->kv_k[k] && t->kv_v[k],"target KV source missing");
            const uint64_t width=uint64_t(cfg.kv_heads)*l.head_dim;
            const auto row_bytes=[&](uint32_t type)->uint64_t {
                return type==1 ? width*2 : width/32*(type==2?18:34);
            };
            const uint64_t positions=l.ring ? std::min<uint64_t>(start,uint64_t(l.ring)+1) : start;
            req(positions*row_bytes(kv_type_k)<=t->kv_bytes[k]
                && positions*row_bytes(kv_type_v)<=t->kv_bytes[k],"target KV visible range exceeds allocation");
            if(l.window)req(l.ring && l.window<=uint64_t(l.ring)+1,"window/ring mismatch");
            d->kv_k[k]=t->kv_k[k];d->kv_v[k]=t->kv_v[k];d->kv_bytes[k]=t->kv_bytes[k];
        }
        d->kdq.clear();d->vdq.clear();
    }
    void load_cluster_head() {
        const bool retained=imparo_cuda_e4b_retained_decode_policy()==1;
        const uint32_t domain=retained?imparo_cuda_e4b_retained_decode_domain():0;
        if(retained&&domain==3)return;
        const char* root=retained?nullptr:std::getenv("IMPARO_LAB_MTP_CLUSTER_HEAD_DIR");
        if(!retained&&(!root||!*root))return;
        req(!retained||domain==1||domain==2,"cluster head retained domain");
        req(g.sm_version==86&&cfg.hidden==256&&cfg.vocab==262144,"cluster head capability");
        if(!retained) {
            const char* minimum=std::getenv("IMPARO_LAB_MTP_MIN_PROB");
            req(minimum&&std::strcmp(minimum,"0")==0,"cluster head requires current ungated proposal policy");
        }
        req(!std::getenv("IMPARO_LAB_MTP_STEP_GRAPH_PROOF"),"dense logits proof incompatible with sparse head");
        std::vector<float> centers(2048*256);std::vector<uint32_t> order(cfg.vocab);
        if(retained) {
            req(embedded_cluster_assets.centroids&&embedded_cluster_assets.ordering,"embedded cluster assets missing");
            std::memcpy(centers.data(),embedded_cluster_assets.centroids,centers.size()*4);
            std::memcpy(order.data(),embedded_cluster_assets.ordering,order.size()*4);
        } else {
            auto read=[&](const char* name,void*dst,size_t bytes){
                std::ifstream in(std::string(root)+"/"+name,std::ios::binary|std::ios::ate);
                req(in&&size_t(in.tellg())==bytes,"cluster sidecar file size");in.seekg(0);in.read(static_cast<char*>(dst),bytes);req(bool(in),"cluster sidecar read");
            };
            read("centroids.f32",centers.data(),centers.size()*4);read("ordering.u32",order.data(),order.size()*4);
        }
        for(float v:centers)req(std::isfinite(v),"cluster centroid nonfinite");
        std::vector<unsigned char>seen(cfg.vocab,0);for(uint32_t id:order){req(id<cfg.vocab&&!seen[id],"cluster ordering is not permutation");seen[id]=1;}
        rc(imparo_cuda_alloc(CLUSTER_DATA,(centers.size()+order.size())*4));
        rc(imparo_cuda_alloc(CLUSTER_SCORES,(2048+32)*4));rc(imparo_cuda_alloc(CLUSTER_IDS,4096*4));
        ck(cudaMemcpy(execution().bufs[CLUSTER_DATA],centers.data(),centers.size()*4,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(static_cast<float*>(execution().bufs[CLUSTER_DATA])+centers.size(),order.data(),order.size()*4,cudaMemcpyHostToDevice));
        cluster_head_enabled=true;
        std::fprintf(stderr,"[mtp-cluster-head] clusters=32 candidates=4096 source=65892304 owner=private target_verify=unchanged\n");
    }
    void cluster_head() {
        using namespace imparo_efficient_embedder;
        const uint8_t* head=resident_weight_range(cfg.head,uint64_t(cfg.hidden/32)*34*cfg.vocab);
        req(head,"cluster head resident weights");
        if(!q8_cache_matches(CUR,cfg.hidden,1,0,Q8_LAYOUT_MMVQ)){
            rc(ensure_q8_scratch(8*sizeof(BlockQ8_1)));
            imparo_sm80_mmvq::quantize_q8_1<<<1,256,0,g.stream>>>(static_cast<const float*>(execution().bufs[CUR]),static_cast<BlockQ8_1*>(execution().q8_scratch),256,1,0);
            own_q8_cache(CUR,cfg.hidden,1,0,Q8_LAYOUT_MMVQ);
        }
        auto*centers=static_cast<const float*>(execution().bufs[CLUSTER_DATA]);
        auto*order=reinterpret_cast<const uint32_t*>(centers+2048*256);
        auto*scores=static_cast<float*>(execution().bufs[CLUSTER_SCORES]);auto*groups=reinterpret_cast<uint32_t*>(scores+2048);
        auto*ids=static_cast<uint32_t*>(execution().bufs[CLUSTER_IDS]);auto*logits=static_cast<float*>(execution().bufs[LOGITS]);auto*token=static_cast<uint32_t*>(execution().bufs[TOK]);
        centroids<<<512,dim3(32,4),0,g.stream>>>(centers,static_cast<const float*>(execution().bufs[CUR]),scores);
        top32<<<1,1024,0,g.stream>>>(scores,groups);
        selected_head<<<1024,dim3(32,4),0,g.stream>>>(head,static_cast<const BlockQ8_1*>(execution().q8_scratch),order,groups,logits,ids);
        imparo_efficient_embedder::argmax<<<1,1024,0,g.stream>>>(logits,ids,4096,token);
        sparse_probability<<<1,1024,0,g.stream>>>(logits,ids,token,reinterpret_cast<float*>(token+1));
        mark_buf_written(LOGITS);mark_buf_written(TOK);
    }
    void allocate() {
        uint32_t qwidth=0;
        for(const auto&l:layers)qwidth=std::max(qwidth,cfg.heads*l.head_dim);
        for(uint32_t b:{X,CUR,O})rc(imparo_cuda_alloc(b,uint64_t(cfg.hidden)*4));
        for(uint32_t b:{Q,ATTN})rc(imparo_cuda_alloc(b,uint64_t(qwidth)*4));
        for(uint32_t b:{GATE,UP})rc(imparo_cuda_alloc(b,uint64_t(cfg.ffn)*4));
        rc(imparo_cuda_alloc(LOGITS,uint64_t(cfg.vocab)*4));rc(imparo_cuda_alloc(TOK,8));
        rc(imparo_cuda_alloc(FEATURE,uint64_t(cfg.batch_capacity)*cfg.target_hidden*4));
        for(uint32_t b:{COMMITTED,NEXT,STEP_SAVED})rc(imparo_cuda_alloc(b,uint64_t(cfg.target_hidden)*4));
        rc(imparo_cuda_alloc(CONCAT,uint64_t(cfg.target_hidden)*8));
        load_cluster_head();
    }
    void dequant_capacity(uint32_t start) {
        uint64_t bytes=0;
        for(const auto&l:layers) {
            const uint64_t rows=l.ring?std::min<uint64_t>(start,uint64_t(l.ring)+1):start;
            const uint64_t padded=(rows+255)/256*256;
            bytes=std::max(bytes,padded*cfg.kv_heads*l.head_dim*2);
        }
        rc(imparo_cuda_alloc(KDQ,bytes));rc(imparo_cuda_alloc(VDQ,bytes));
    }
    uint32_t step(uint32_t start,uint32_t token,float *probability=nullptr,bool eager=false) {
        imparo_cuda_begin();
        // This owner has no target-token stores and no existing decode Graph body.
        // RoPE uses start; attention alone uses start-1 to stop before anchor KV.
        rc(imparo_cuda_set_batch_geometry(start,1,0,start,1));
        imparo_cuda_write_u32(TOK,0,&token,1);
        step_input=token;submit_step(start,eager,[&]() {
        imparo_cuda_rows(cfg.target_embedding_kind,cfg.target_embedding,cfg.target_hidden,cfg.vocab,
            TOK,cfg.embedding_scale,CONCAT,1);
        imparo_cuda_copy_range(CONCAT,cfg.target_hidden,NEXT,0,cfg.target_hidden);
        mm(cfg.pre_proj,2*cfg.target_hidden,cfg.hidden,CONCAT,X);
        for(const auto&l:layers) {
            norm(CUR,X,l.attn_norm);
            mm(l.q,cfg.hidden,cfg.heads*l.head_dim,CUR,Q);
            const float *freqs=l.has_rope_freqs?reinterpret_cast<const float*>(weights_host+l.rope_freqs):nullptr;
            imparo_cuda_head_norm_rope_hadamard(Q,l.qn,l.head_dim,cfg.eps,cfg.heads,start,1,
                l.rope_dim,l.rope_theta,freqs,l.had_k);
            // A query at p=start reads existing keys j<start. Native attention uses
            // inclusive p': start-1 and W': W-1 preserve j>=start-W+1 exactly.
            imparo_cuda_attention(l.target_kv_layer,l.head_dim,cfg.heads,cfg.kv_heads,
                cfg.kv_heads*l.head_dim,start-1,1.0f,l.window?l.window-1:0,1,l.ring,Q,ATTN,KDQ,VDQ);
            if(l.had_v)imparo_cuda_hadamard(ATTN,cfg.heads*l.head_dim,l.had_v);
            mm(l.o,cfg.heads*l.head_dim,cfg.hidden,ATTN,O);
            dense_norm(O,O,l.attn_post_norm);imparo_cuda_add(X,O,cfg.hidden);
            norm(CUR,X,l.ffn_norm);
            // Q8 generic GELU: native fused_epilogue=2 means SiLU; reuse exact
            // existing Q8 projections plus GELU multiply instead.
            mm(l.gate,cfg.hidden,cfg.ffn,CUR,GATE);
            mm(l.up,cfg.hidden,cfg.ffn,CUR,UP);
            imparo_cuda_gelu_mul(GATE,UP,cfg.ffn);
            mm(l.down,cfg.ffn,cfg.hidden,GATE,O);
            dense_norm(O,O,l.ffn_post_norm);imparo_cuda_add(X,O,cfg.hidden);
            imparo_cuda_scale(X,l.out_scale,cfg.hidden);
        }
        dense_norm(CUR,X,cfg.out_norm);
        if(cluster_head_enabled){
            // Independent continuation projection produces the same reusable Q8 input.
            mm(cfg.post_proj,cfg.hidden,cfg.target_hidden,CUR,NEXT);
            cluster_head();
        }else{
            mm(cfg.head,cfg.hidden,cfg.vocab,CUR,LOGITS);
            mm(cfg.post_proj,cfg.hidden,cfg.target_hidden,CUR,NEXT);
            imparo_cuda_argmax(LOGITS,TOK,cfg.vocab);
        }
        });
        if(probability&&!cluster_head_enabled)first_probability<<<1,1024,0,g.stream>>>(
            static_cast<const float*>(execution().bufs[LOGITS]),
            static_cast<const uint32_t*>(execution().bufs[TOK]),
            static_cast<float*>(execution().bufs[TOK])+1,cfg.vocab);
        rc(imparo_cuda_end());
        uint32_t returned[2]={0,0};
        imparo_cuda_read(TOK,0,reinterpret_cast<float*>(returned),probability?2:1);rc(imparo_cuda_end());
        req(returned[0]<cfg.vocab,"assistant argmax outside vocabulary");
        if(probability){std::memcpy(probability,&returned[1],sizeof(float));
            req(std::isfinite(*probability)&&*probability>0&&*probability<=1.0f,"assistant probability invalid");}
        return returned[0];
    }
    void proof_step(uint32_t start,uint32_t anchor) {
        const char *path=std::getenv("IMPARO_LAB_MTP_STEP_GRAPH_PROOF");
        if(!path || step_proof_done)return;
        step_proof_done=true;
        const auto reset=[&]() {imparo_cuda_copy_range(NEXT,0,COMMITTED,0,cfg.target_hidden);rc(imparo_cuda_end());};
        float probability=0;
        bool dynamic_exact=true;
        for(uint32_t pos:{start-16,start-15,start-1,start}){
            std::vector<uint32_t>a(cfg.vocab+cfg.target_hidden),b(a.size());float pa=0,pb=0;
            reset();uint32_t ta=step(pos,anchor,&pa,true);imparo_cuda_read(LOGITS,0,reinterpret_cast<float*>(a.data()),cfg.vocab);imparo_cuda_read(NEXT,0,reinterpret_cast<float*>(a.data()+cfg.vocab),cfg.target_hidden);
            reset();uint32_t tb=step(pos,anchor,&pb,false);imparo_cuda_read(LOGITS,0,reinterpret_cast<float*>(b.data()),cfg.vocab);imparo_cuda_read(NEXT,0,reinterpret_cast<float*>(b.data()+cfg.vocab),cfg.target_hidden);
            dynamic_exact=dynamic_exact&&a==b&&ta==tb&&std::memcmp(&pa,&pb,4)==0;
        }
        req(dynamic_exact,"assistant cached position update mismatch");
        reset();step(start,anchor,&probability,true);
        reset();step(start,anchor,&probability,false);
        std::vector<uint32_t> reference(cfg.vocab+cfg.target_hidden), current(reference.size());
        uint32_t reference_token=0,reference_probability=0;bool exact=true;
        double ms[4]={};
        for(unsigned leg=0;leg<4;++leg) {
            const bool eager=leg==0||leg==3;
            for(unsigned repeat=0;repeat<8;++repeat) {
                reset();const auto t=std::chrono::steady_clock::now();
                const uint32_t token=step(start,anchor,&probability,eager);
                ms[leg]+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/8;
                imparo_cuda_read(LOGITS,0,reinterpret_cast<float*>(current.data()),cfg.vocab);
                imparo_cuda_read(NEXT,0,reinterpret_cast<float*>(current.data()+cfg.vocab),cfg.target_hidden);
                rc(imparo_cuda_end());uint32_t prob=0;std::memcpy(&prob,&probability,4);
                if(leg==0&&repeat==0){reference=current;reference_token=token;reference_probability=prob;}
                exact=exact&&reference==current&&reference_token==token&&reference_probability==prob;
            }
        }
        std::ofstream out(path);out<<"{\"start\":"<<start<<",\"exact_all_logits_hidden_token_probability\":"<<(exact?"true":"false")
            <<",\"mean_step_ms_abba\":["<<ms[0]<<","<<ms[1]<<","<<ms[2]<<","<<ms[3]<<"]"
            <<",\"updates\":"<<step_updates<<",\"instantiates\":"<<step_instantiates<<"}";out.close();
        std::fprintf(stderr,"[mtp-step-proof] exact=%u updates=%llu instantiates=%llu\n",unsigned(exact),(unsigned long long)step_updates,(unsigned long long)step_instantiates);
        std::exit(exact?0:73); // Explicit diagnostic: no continuation with proof state.
    }
};
static std::unique_ptr<Session> attached;
static uint64_t next_generation=1;

template<class Fn>int invoke(Fn fn) {
    if(!attached || attached->poisoned)return CUDA_RC_INVALID;
    try { fn(*attached);return 0; }
    catch(const std::exception&e) {
        std::fprintf(stderr,"[gemma4-mtp] %s\n",e.what());
        auto&s=*attached;s.poisoned=true;
        if(active_execution_owner_id==s.draft_owner) {
            (void)imparo_cuda_end();s.clear_borrow();
            if(execution_owner_select(s.target_owner))std::fprintf(stderr,"[gemma4-mtp] target restore failed\n");
        } else { s.clear_borrow(); }
        return CUDA_RC_ERROR;
    }
}
} // namespace imparo_gemma4_mtp

extern "C" int imparo_cuda_gemma4_mtp_cluster_assets(const uint8_t*centroids,
        uint64_t centroids_bytes,const uint8_t*ordering,uint64_t ordering_bytes) {
    using namespace imparo_gemma4_mtp;
    if(imparo_cuda_e4b_retained_decode_policy()!=1||attached||!execution_boundary_closed()
        ||!centroids||!ordering||centroids_bytes!=2097152||ordering_bytes!=1048576)
        return CUDA_RC_INVALID;
    if(embedded_cluster_assets.centroids
        &&(embedded_cluster_assets.centroids!=centroids||embedded_cluster_assets.ordering!=ordering))
        return CUDA_RC_INVALID;
    embedded_cluster_assets.centroids=centroids;
    embedded_cluster_assets.ordering=ordering;
    return 0;
}

extern "C" int imparo_cuda_gemma4_mtp_attach(const imparo_gemma4_mtp::Config*c,
        const imparo_gemma4_mtp::Layer*l,uint32_t nl) {
    using namespace imparo_gemma4_mtp;
    if(!c||!l||nl!=4||attached||!next_generation||!execution_boundary_closed()||!g.weights_host
        ||c->hidden!=256||c->target_hidden!=2560||c->ffn!=2048||c->vocab!=262144
        ||c->heads!=4||c->kv_heads!=2||c->batch_capacity<3||c->batch_capacity>512
        ||c->target_final_layer>=MAX_LAYERS||c->target_embedding_kind>2||c->reserved
        ||!std::isfinite(c->eps)||c->eps<=0||!std::isfinite(c->embedding_scale)
        ||std::abs(c->embedding_scale-std::sqrt(float(c->target_hidden)))>1e-5f
        ||(g.kv_type_k!=1&&g.kv_type_k!=2&&g.kv_type_k!=8)
        ||(g.kv_type_v!=1&&g.kv_type_v!=2&&g.kv_type_v!=8))return CUDA_RC_INVALID;
    try { attached=std::make_unique<Session>(*c,l,nl); }
    catch(const std::bad_alloc&) { return CUDA_RC_OOM; }
    auto&s=*attached;s.target_owner=active_execution_owner_id;s.generation=next_generation++;
    s.weights_host=g.weights_host;s.weights_device=g.weights;s.weights_len=g.weights_len;
    s.choice_epoch=g.choice_epoch;s.kv_type_k=g.kv_type_k;s.kv_type_v=g.kv_type_v;
    return invoke([&](Session&x){
        x.q8(c->pre_proj,2*c->target_hidden,c->hidden);x.q8(c->post_proj,c->hidden,c->target_hidden);
        x.q8(c->head,c->hidden,c->vocab);x.span(c->out_norm,uint64_t(c->hidden)*4);
        const uint64_t erow=c->target_embedding_kind==0?uint64_t(c->target_hidden)*4:
            uint64_t(c->target_hidden/32)*(c->target_embedding_kind==1?18:34);
        x.span(c->target_embedding,erow*c->vocab);
        for(uint32_t i=0;i<nl;++i){const auto&w=l[i];
            x.req(w.head_dim==(i<3?256u:512u)&&w.target_kv_layer<MAX_LAYERS
                &&w.rope_dim&&w.rope_dim<=w.head_dim&&w.rope_dim%2==0
                &&((i<3&&w.window>1&&w.ring)||(i==3&&!w.window&&!w.ring))
                &&std::isfinite(w.rope_theta)&&w.rope_theta>0&&std::isfinite(w.out_scale)
                &&w.has_rope_freqs<=1,"assistant layer geometry");
            for(uint32_t h:{w.had_k,w.had_v})x.req(!h||(h<=w.head_dim&&(h&(h-1))==0&&w.head_dim%h==0),"Hadamard geometry");
            x.q8(w.q,c->hidden,c->heads*w.head_dim);x.q8(w.o,c->heads*w.head_dim,c->hidden);
            x.q8(w.gate,c->hidden,c->ffn);x.q8(w.up,c->hidden,c->ffn);x.q8(w.down,c->ffn,c->hidden);
            for(uint64_t off:{w.attn_norm,w.attn_post_norm,w.ffn_norm,w.ffn_post_norm})x.span(off,uint64_t(c->hidden)*4);
            x.span(w.qn,uint64_t(w.head_dim)*4);if(w.has_rope_freqs)x.span(w.rope_freqs,uint64_t(w.rope_dim/2)*4);
        }
        x.rc(execution_owner_create(&x.draft_owner));x.rc(execution_owner_select(x.draft_owner));
        const char *pv_f32=std::getenv("IMPARO_LAB_MTP_PV_F32");
        execution().assistant_pv_f32=pv_f32&&std::strcmp(pv_f32,"1")==0;
        x.allocate();x.rc(imparo_cuda_end());x.rc(execution_owner_select(x.target_owner));
    });
}
extern "C" int imparo_cuda_gemma4_mtp_capture_mode(uint32_t enabled) {
    using namespace imparo_gemma4_mtp;
    if(enabled>1)return CUDA_RC_INVALID;
    return invoke([&](Session&s){s.boundary();
        if(s.capture_enabled!=(enabled!=0))s.req(destroy_decode_graph_checked(),"target Graph borrowed during capture toggle");
        s.capture_enabled=enabled!=0;s.feature_rows=0;s.feature_generation=0;
    });
}
extern "C" int imparo_cuda_gemma4_mtp_reset() {
    using namespace imparo_gemma4_mtp;
    return invoke([](Session&s){s.boundary();s.req(next_generation!=0,"generation exhausted");
        s.clear_step_graphs();s.generation=next_generation++;s.S=0;s.hidden_valid=false;s.feature_rows=0;s.feature_generation=0;
    });
}
extern "C" int imparo_cuda_gemma4_mtp_capture_layer(uint32_t layer,uint32_t start,uint32_t n,uint32_t src) {
    using namespace imparo_gemma4_mtp;
    if(!attached||!attached->capture_enabled||layer!=attached->cfg.target_final_layer)return 0;
    return invoke([&](Session&s){s.identity();
        s.req(active_execution_owner_id==s.target_owner&&src<B_COUNT&&n&&n<=s.cfg.batch_capacity
            &&uint64_t(start)+n<=UINT32_MAX,"feature geometry");
        // Mainline observer binding executes the loop while capture is active.
        // Reject an unqualified captured graph rather than leave stale host ranges.
        s.req(!execution().graph_capturing&&!execution().prefill_capture_active,"feature capture requires active observer loop");
        // Target callback is AFTER output RMSNorm: do not normalize FEATURE again.
        // LastToken may initialize only the final row; append selects its absolute position.
        const uint64_t bytes=uint64_t(n)*s.cfg.target_hidden*4;
        s.req(execution().bufs[src]&&execution().sizes[src]>=bytes&&s.draft()->sizes[FEATURE]>=bytes,"feature storage");
        s.ck(cudaMemcpyAsync(s.draft()->bufs[FEATURE],execution().bufs[src],bytes,cudaMemcpyDeviceToDevice,g.stream));
        s.feature_start=start;s.feature_rows=n;s.feature_generation=s.generation;
    });
}
extern "C" int imparo_cuda_gemma4_mtp_append(uint32_t start,uint32_t consumed) {
    using namespace imparo_gemma4_mtp;
    return invoke([&](Session&s){s.boundary();
        s.req(s.capture_enabled&&start==s.S&&consumed&&uint64_t(start)+consumed<=UINT32_MAX,"commit boundary");
        const uint32_t wanted=start+consumed-1;
        const bool present=s.feature_generation==s.generation&&s.feature_rows
            &&wanted>=s.feature_start&&uint64_t(wanted)<uint64_t(s.feature_start)+s.feature_rows;
        if(present){
            const uint64_t row=wanted-s.feature_start;
            s.ck(cudaMemcpyAsync(s.draft()->bufs[COMMITTED],
                static_cast<const float*>(s.draft()->bufs[FEATURE])+row*s.cfg.target_hidden,
                uint64_t(s.cfg.target_hidden)*4,cudaMemcpyDeviceToDevice,g.stream));
            s.ck(cudaStreamSynchronize(g.stream));s.hidden_position=wanted;s.hidden_valid=true;
        }else{
            // State-only prefill chunks can omit the final layer. Advancing the
            // cursor is valid; manufacturing an assistant hidden state is not.
            s.hidden_valid=false;
        }
        s.S=start+consumed;s.feature_rows=0;s.feature_generation=0;
    });
}
extern "C" int imparo_cuda_gemma4_mtp_generate(uint32_t start,uint32_t anchor,uint32_t*ids) {
    using namespace imparo_gemma4_mtp;
    return invoke([&](Session&s){s.boundary();
        s.req(ids&&s.capture_enabled&&start&&start==s.S&&anchor<s.cfg.vocab
            &&s.hidden_valid&&s.hidden_position==start-1,"proposal lacks committed target hidden");
        // Borrow after a closed target boundary. Refresh descriptors each call so
        // target KV growth/page-table generations never leave cached stale pointers.
        s.borrow_kv(start);s.rc(execution_owner_select(s.draft_owner));s.dequant_capacity(start);
        imparo_cuda_copy_range(NEXT,0,COMMITTED,0,s.cfg.target_hidden);
        const uint32_t first=s.step(start,anchor);const uint32_t second=s.step(start,first);
        s.clear_borrow();s.rc(execution_owner_select(s.target_owner));ids[0]=first;ids[1]=second;
    });
}
// A zero count is a temporary abstention, not a failed or empty proposal.
// NEXT is scratch; COMMITTED/S/target KV are advanced only by target commit.
extern "C" int imparo_cuda_gemma4_mtp_generate_gated(uint32_t start,uint32_t anchor,
        float minimum,uint32_t*ids,uint32_t*count,float*probability) {
    using namespace imparo_gemma4_mtp;
    if(!ids||!count||!probability||!std::isfinite(minimum)||minimum<0||minimum>1)return CUDA_RC_INVALID;
    return invoke([&](Session&s){s.boundary();
        s.req(s.capture_enabled&&start&&start==s.S&&anchor<s.cfg.vocab
            &&s.hidden_valid&&s.hidden_position==start-1,"proposal lacks committed target hidden");
        s.borrow_kv(start);s.rc(execution_owner_select(s.draft_owner));s.dequant_capacity(start);
        imparo_cuda_copy_range(NEXT,0,COMMITTED,0,s.cfg.target_hidden);
        s.proof_step(start,anchor);
        float confidence=0;const uint32_t first=s.step(start,anchor,&confidence);
        const bool propose=confidence>=minimum;
        const uint32_t second=propose?s.step(start,first):0;
        s.clear_borrow();s.rc(execution_owner_select(s.target_owner));
        *count=propose?2u:0u;*probability=confidence;
        if(propose){ids[0]=first;ids[1]=second;}
    });
}
extern "C" int imparo_cuda_gemma4_mtp_detach() {
    using namespace imparo_gemma4_mtp;
    if(!attached)return 0;auto&s=*attached;
    if(active_execution_owner_id!=s.target_owner||!execution_boundary_closed())return CUDA_RC_INVALID;
    if(!destroy_decode_graph_checked())return CUDA_RC_INVALID;
    s.clear_step_graphs();s.capture_enabled=false;s.clear_borrow();
    const int rc=s.draft_owner?execution_owner_release(s.draft_owner):0;
    if(!rc)attached.reset();return rc;
}
