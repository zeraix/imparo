#pragma once
#include <stdexcept>
#include <string>
#include "dspark_confidence_host.h"
#include "sm86/dspark_noncausal_d64.cuh"
#include "sm86/dspark_frontier_lab.cuh"
namespace imparo_dspark {
struct DsparkConfig {
 uint64_t embedding,fc,enc_norm,out_norm,markov1,markov2,confidence,confidence_bias;
 uint32_t hidden,ffn,vocab,heads,kv_heads,head_dim,rank,block_size,mask_token,kv_capacity,batch_capacity,target_hidden;
 float eps,rope_theta;
};
struct DsparkLayer {uint64_t attn_norm,q,k,v,o,qn,kn,ffn_norm,gate,up,down;};
static_assert(sizeof(DsparkConfig)==120 && sizeof(DsparkLayer)==88,"DSpark wire layout");
struct FrontierGroup {uint32_t parent,tokens[4];float logp[4],confidence;};
static_assert(sizeof(FrontierGroup)==40,"DSpark frontier group wire layout");

// The existing chain confidence formula, shared with optional frontier features.
// Keep the original bias/hidden/rank order and double accumulation.
static float confidence_half(uint16_t bits){__half value;memcpy(&value,&bits,sizeof(value));return __half2float(value);}
static float confidence_head(const float*hidden,const float*rank,const __half*weights,float bias,unsigned h,unsigned r){
 return imparo_dspark_confidence::head(hidden,rank,reinterpret_cast<const uint8_t*>(weights),bias,h,r,confidence_half);
}

constexpr unsigned X=0,CUR=1,Q=2,K=3,V=4,ATTN=5,O=6,GATE=7,UP=8,IDS=11,LOGITS=12,TOK=13,FEATURE=20,FUSED=21,RANK=22,BIAS=23,COL=24;
struct Session {
 DsparkConfig cfg;std::vector<DsparkLayer> dw;std::vector<uint32_t> target_layers;
 std::vector<uint32_t> tree_secondary;
 std::vector<imparo_dspark_frontier::Node> tree_frontier;
 // The same already-read beam pool, before the existing 16-row selection.
 std::vector<imparo_dspark_frontier::Node> tree_candidates;
 std::vector<FrontierGroup> tree_groups;
 unsigned H,F,M,VOC,KW,S=0,current_anchor=0;uint64_t draft_owner=0,target_owner=0;
 bool capture_enabled=false,poisoned=false,parked=false;uint64_t ticket=0;unsigned feature_mask=0,feature_start=0,feature_rows=0;
 const uint8_t* host_weights=nullptr;uint64_t host_weights_len=0;
 Session(const DsparkConfig& c,const DsparkLayer* w,unsigned nw,const uint32_t* t,unsigned nt):cfg(c),dw(w,w+nw),target_layers(t,t+nt),H(c.hidden),F(c.ffn),M(c.block_size),VOC(c.vocab),KW(c.kv_heads*c.head_dim){}
void clear_tree_candidates(){tree_secondary.clear();tree_frontier.clear();tree_candidates.clear();tree_groups.clear();}
void req(bool b,const char* m){if(!b)throw std::runtime_error(m);}
void rc(int v){if(v)throw std::runtime_error("native status "+std::to_string(v));}
void ck(cudaError_t v){if(v!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(v));}
std::vector<float> get(unsigned b,size_t n){std::vector<float>v(n);imparo_cuda_read(b,0,v.data(),n);rc(imparo_cuda_end());for(auto x:v)req(std::isfinite(x),"nonfinite activation");return v;}
void mm(unsigned long long off,unsigned ni,unsigned no,unsigned src,unsigned dst,unsigned n){imparo_cuda_matmat(2,off,ni,no,src,dst,n,0);}
void norm(unsigned dst,unsigned src,unsigned long long w,unsigned n){imparo_cuda_rms_norm_project(dst,src,w,H,cfg.eps,n,H,0);}
void dense_norm(unsigned dst,unsigned src,unsigned long long w,unsigned n){imparo_cuda_rms_norm(dst,src,w,H,cfg.eps,n,H,0,1);}
void head(unsigned b,unsigned long long w,unsigned heads,unsigned start,unsigned n){imparo_cuda_head_norm_rope_hadamard(b,w,64,cfg.eps,heads,start,n,cfg.head_dim,cfg.rope_theta,nullptr,0);}

void store(unsigned layer,unsigned start,unsigned n){
 for(unsigned isv=0;isv<2;++isv){auto src=isv?V:K; auto dst=isv?execution().kv_v[layer]:execution().kv_k[layer];invalidate_kv_dequant(layer,isv);
 k_kv_store_f16<<<dim3((KW+255)/256,n),256,0,g.stream>>>((const float*)execution().bufs[src],(__half*)dst,KW,start,n,0,nullptr,nullptr);}
}
void history(unsigned start,unsigned n){imparo_cuda_begin();rc(imparo_cuda_set_batch_geometry(start,n,0,start,n));mm(cfg.fc,unsigned(target_layers.size())*cfg.target_hidden,H,FEATURE,FUSED,n);dense_norm(FUSED,FUSED,cfg.enc_norm,n);for(unsigned l=0;l<dw.size();++l){mm(dw[l].k,H,KW,FUSED,K,n);mm(dw[l].v,H,KW,FUSED,V,n);head(K,dw[l].kn,cfg.kv_heads,start,n);store(l,start,n);}rc(imparo_cuda_end());}
struct Proposal{std::vector<uint32_t> ids;std::vector<float> conf;};
Proposal frontier_generate(bool accept_features){
 namespace bf=imparo_dspark_frontier;
 req(cfg.rank==256&&VOC==128000&&M==9,"frontier diagnostic geometry");
 auto* state=reinterpret_cast<bf::State*>(execution().bufs[IDS]);
 auto* tokens=reinterpret_cast<uint32_t*>(execution().bufs[TOK]);
 auto* ranks=static_cast<float*>(execution().bufs[O]);
 auto* bias=static_cast<float*>(execution().bufs[GATE]);
 auto* col=static_cast<float*>(execution().bufs[UP]);
 auto* quant=reinterpret_cast<BlockQ8_1*>(execution().bufs[ATTN]);
 req(execution().sizes[IDS]>=sizeof(bf::State)&&execution().sizes[O]>=4*256*4&&execution().sizes[GATE]>=4*VOC*4&&execution().sizes[UP]>=4*VOC*4&&execution().sizes[ATTN]>=4*8*sizeof(BlockQ8_1),"frontier scratch capacity");
 auto* w1=static_cast<const uint8_t*>(resident_weight_range(cfg.markov1,uint64_t(VOC)*8*34));auto*w2=static_cast<const uint8_t*>(resident_weight_range(cfg.markov2,uint64_t(VOC)*8*34));req(w1&&w2,"frontier weights resident");
 bf::initialize<<<1,1,0,g.stream>>>(state,tokens,current_anchor);
 for(unsigned depth=0;depth<8;++depth){unsigned n=depth?4:1;k_rows_q8_0<<<dim3(1,n),256,0,g.stream>>>(w1,tokens,ranks,256,1.f,n);k_quantize_q8_1<<<dim3(n,1),128,0,g.stream>>>(ranks,quant,256,n,0);
 if(n==1)imparo_sm80_q8_mmvq::q8_0_q8_1_short_k<<<(VOC+3)/4,dim3(32,4),0,g.stream>>>(w2,quant,bias,256,VOC,VOC,0);else bf::shortk_batch<4><<<(VOC+3)/4,dim3(32,4),0,g.stream>>>(w2,quant,bias);
 bf::add_base<<<(n*VOC+255)/256,256,0,g.stream>>>(static_cast<const float*>(execution().bufs[LOGITS]),bias,col,depth,n);bf::top_parts<<<dim3(64,n),256,0,g.stream>>>(col,state);bf::merge_top<<<n,256,0,g.stream>>>(state);bf::choose_frontier<<<1,1,0,g.stream>>>(state,tokens,depth,n,accept_features);}

 bf::choose_budget<<<1,1,0,g.stream>>>(state);bf::Node nodes[16],all[33];bf::Top groups[29][4];std::vector<float> hidden;
 ck(cudaMemcpyAsync(nodes,&state->out[0],sizeof(nodes),cudaMemcpyDeviceToHost,g.stream));ck(cudaMemcpyAsync(all,&state->nodes[0],sizeof(all),cudaMemcpyDeviceToHost,g.stream));
 if(accept_features){
  const uint64_t hidden_bytes=uint64_t(8)*H*sizeof(float);
  req(execution().bufs[CUR]&&execution().sizes[CUR]>=hidden_bytes,"frontier confidence hidden capacity");
  hidden.resize(size_t(8)*H);
  ck(cudaMemcpyAsync(groups,&state->groups[0][0],sizeof(groups),cudaMemcpyDeviceToHost,g.stream));
  ck(cudaMemcpyAsync(hidden.data(),execution().bufs[CUR],hidden_bytes,cudaMemcpyDeviceToHost,g.stream));
 }
 rc(imparo_cuda_end());
 // Validate every cached candidate, including those outside the default tree.
 // Scores remain the native cumulative log probabilities; no repricing here.
 for(unsigned i=0;i<33;++i){const auto n=all[i];req(n.token<VOC&&std::isfinite(n.score)&&n.depth>=0&&n.depth<=8,"frontier candidate value");if(!i){req(n.token==current_anchor&&n.parent==-1&&n.depth==0&&n.score==0.f,"frontier candidate root");}else{req(n.parent>=0&&unsigned(n.parent)<i&&all[n.parent].depth+1==n.depth&&n.score<=all[n.parent].score,"frontier candidate topology");}}
 tree_frontier.assign(nodes,nodes+16);for(unsigned i=0;i<16;++i){auto n=nodes[i];req(n.token<VOC&&std::isfinite(n.score)&&(!i||(n.parent>=0&&unsigned(n.parent)<i&&nodes[n.parent].depth+1==n.depth)),"frontier topology");}
 int best=29;for(int i=30;i<33;++i)if(all[i].score>all[best].score)best=i;
 Proposal result;result.ids.assign(M,current_anchor);result.conf.assign(M,1.f);for(int i=best;i>0;i=all[i].parent){req(all[i].depth>0&&all[i].depth<=8&&all[i].parent<i,"frontier chain projection");result.ids[all[i].depth-1]=all[i].token;}
 tree_candidates.assign(all,all+33);
 if(accept_features){
  for(float x:hidden)req(std::isfinite(x),"frontier confidence nonfinite hidden");
  const uint64_t conf_bytes=(uint64_t(H)+cfg.rank)*sizeof(__half),row_bytes=uint64_t(cfg.rank/32)*34;
  req(host_weights&&cfg.confidence<=host_weights_len&&conf_bytes<=host_weights_len-cfg.confidence
   &&cfg.confidence_bias<=host_weights_len&&sizeof(float)<=host_weights_len-cfg.confidence_bias
   &&cfg.markov1<=host_weights_len&&uint64_t(VOC)*row_bytes<=host_weights_len-cfg.markov1,"frontier confidence weight range");
  const auto*conf=reinterpret_cast<const __half*>(host_weights+cfg.confidence);float conf_bias;memcpy(&conf_bias,host_weights+cfg.confidence_bias,sizeof(conf_bias));
  req(std::isfinite(conf_bias),"frontier confidence bias");for(unsigned j=0;j<H+cfg.rank;++j)req(std::isfinite(__half2float(conf[j])),"frontier confidence weight");
  std::vector<float> rank(cfg.rank);tree_groups.reserve(29);
  for(unsigned parent=0;parent<29;++parent){
   req(all[parent].depth>=0&&all[parent].depth<8,"frontier confidence parent depth");
   imparo_dspark_confidence::parent_rank(host_weights+cfg.markov1,size_t(uint64_t(VOC)*row_bytes),all[parent].token,cfg.rank,rank.data(),confidence_half);
   FrontierGroup group{};group.parent=parent;
   group.confidence=imparo_dspark_confidence::at_depth(hidden.data(),8,H,unsigned(all[parent].depth),rank.data(),cfg.rank,reinterpret_cast<const uint8_t*>(conf),conf_bias,confidence_half);
   req(std::isfinite(group.confidence)&&group.confidence>=0.f&&group.confidence<=1.f,"frontier confidence probability");
   for(unsigned j=0;j<4;++j){const auto top=groups[parent][j];req(top.token<VOC&&std::isfinite(top.logp)&&top.logp<=0.f,"frontier group value");
    if(j)req(groups[parent][j-1].logp>=top.logp,"frontier group order");for(unsigned k=0;k<j;++k)req(groups[parent][k].token!=top.token,"frontier group duplicate");group.tokens[j]=top.token;group.logp[j]=top.logp;}
   tree_groups.push_back(group);
  }
  // Every retained beam edge must belong to its real parent's complete group.
  for(unsigned i=1;i<33;++i){req(all[i].parent>=0&&all[i].parent<29,"frontier group parent range");const auto&group=tree_groups[unsigned(all[i].parent)];bool found=false;for(unsigned j=0;j<4;++j)if(group.tokens[j]==all[i].token)found=true;req(found,"frontier group edge identity");}
 }
 for(unsigned b:{O,GATE,UP,ATTN,IDS,TOK})mark_buf_written(b);
 static bool seen=false;if(!seen){fprintf(stderr,"[dspark-frontier] gpu=1 beam=4 nodes=16 depth=8 cached_backbone=1\n");seen=true;}
 return result;

}

Proposal generate(bool accept_features=false){const char* tree_flag=std::getenv("IMPARO_LAB_DSPARK_TREE16");const bool tree=imparo_lfm_retained::common(tree_flag&&std::strcmp(tree_flag,"1")==0);clear_tree_candidates();imparo_cuda_begin();rc(imparo_cuda_set_batch_geometry(S,M,0,S,M));std::vector<uint32_t> ids(M,cfg.mask_token);ids[0]=current_anchor;imparo_cuda_write_u32(TOK,0,ids.data(),M);imparo_cuda_rows(2,cfg.embedding,H,VOC,TOK,1.0f,X,M);
for(unsigned l=0;l<dw.size();++l){norm(CUR,X,dw[l].attn_norm,M);mm(dw[l].q,H,H,CUR,Q,M);mm(dw[l].k,H,KW,CUR,K,M);mm(dw[l].v,H,KW,CUR,V,M);head(Q,dw[l].qn,cfg.heads,S,M);head(K,dw[l].kn,cfg.kv_heads,S,M);store(l,S,M);imparo_cuda_scale(Q,0.125f,M*H);auto plan=imparo_sm80_d64_mma_plan::make(M,cfg.kv_heads,S+M);if(!launch_attention_d64_ordered_qk((const float*)execution().bufs[Q],(const __half*)execution().kv_k[l],(const __half*)execution().kv_v[l],(float*)execution().bufs[ATTN],cfg.heads,cfg.kv_heads,KW,S,0,M,0,0.125f,S+M,plan.key_updates,nullptr,false,S+M))imparo_sm80_d64_mma_noncausal::whole_k_tile<false><<<plan.blocks,128,0,g.stream>>>((const float*)execution().bufs[Q],(const __half*)execution().kv_k[l],(const __half*)execution().kv_v[l],(float*)execution().bufs[ATTN],cfg.heads,cfg.kv_heads,KW,S,0,M,0,0.125f,S+M,plan.key_updates,nullptr,false,S+M);mark_buf_written(ATTN);mm(dw[l].o,H,H,ATTN,O,M);imparo_cuda_add(X,O,M*H);norm(CUR,X,dw[l].ffn_norm,M);imparo_cuda_matmat_gated(2,dw[l].gate,2,dw[l].up,H,F,CUR,GATE,UP,M,2);mm(dw[l].down,F,H,GATE,O,M);imparo_cuda_add(X,O,M*H);}
dense_norm(CUR,X,cfg.out_norm,M);mm(cfg.embedding,H,VOC,CUR,LOGITS,M);
const char*frontier_flag=std::getenv("IMPARO_LAB_DSPARK_FRONTIER16");
if(tree&&imparo_lfm_retained::common(frontier_flag&&std::strcmp(frontier_flag,"1")==0)&&16<=cfg.batch_capacity-S%cfg.batch_capacity&&uint64_t(S)+16<=cfg.kv_capacity)return frontier_generate(accept_features);
// GPU Markov chain: each argmax becomes the next position's table row index.
// Token shadow is irrelevant to the resident-weight device row kernel.
uint32_t anchor=current_anchor;imparo_cuda_write_u32(TOK,0,&anchor,1);
for(unsigned i=0;i<M;++i){imparo_cuda_rows(2,cfg.markov1,cfg.rank,VOC,TOK,1.0f,RANK,1);mm(cfg.markov2,cfg.rank,VOC,RANK,BIAS,1);imparo_cuda_copy_range(COL,0,LOGITS,i*VOC,VOC);imparo_cuda_add(COL,BIAS,VOC);if(tree&&i<7){k_argmax_two<1024><<<1,1024,0,g.stream>>>((const float*)execution().bufs[COL],(uint32_t*)execution().bufs[TOK],(uint32_t*)execution().bufs[IDS]+M+i,VOC,1);mark_buf_written(TOK);mark_buf_written(IDS);}else{imparo_cuda_argmax(COL,TOK,VOC);}imparo_cuda_copy_range(IDS,i,TOK,0,1);imparo_cuda_copy_range(UP,i*cfg.rank,RANK,0,cfg.rank);}
rc(imparo_cuda_end());Proposal result;result.ids.resize(M);imparo_cuda_read(IDS,0,reinterpret_cast<float*>(result.ids.data()),M);rc(imparo_cuda_end());if(tree){tree_secondary.resize(7);imparo_cuda_read(IDS,M,reinterpret_cast<float*>(tree_secondary.data()),7);rc(imparo_cuda_end());for(unsigned i=0;i<7;++i)req(tree_secondary[i]<VOC&&tree_secondary[i]!=result.ids[i],"tree secondary invalid");}auto hidden=get(CUR,M*H);auto rank=get(UP,M*cfg.rank);const __half* conf=reinterpret_cast<const __half*>(host_weights+cfg.confidence);float bias;memcpy(&bias,host_weights+cfg.confidence_bias,4);for(unsigned i=0;i<M;++i){req(result.ids[i]<VOC,"invalid draft token");result.conf.push_back(confidence_head(hidden.data()+i*H,rank.data()+i*cfg.rank,conf,bias,H,cfg.rank));}return result;}
};
static std::unique_ptr<Session> attached;
// Issue a fresh ticket for every parking operation; zero means exhausted/invalid.
// A stale cache entry cannot resume a later Session or a later park of this Session.
static uint64_t next_park_ticket=1;

template<class Fn>int invoke(Fn fn){try{if(!attached||attached->poisoned||attached->parked)return CUDA_RC_INVALID;fn(*attached);return 0;}catch(const std::exception&e){fprintf(stderr,"[dspark] %s\n",e.what());if(attached){attached->poisoned=true;imparo_cuda_end();if(active_execution_owner_id!=attached->target_owner)execution_owner_select(attached->target_owner);}return CUDA_RC_ERROR;}}
} // namespace imparo_dspark
extern "C" int imparo_cuda_dspark_detach();
extern "C" int imparo_cuda_dspark_attach(const imparo_dspark::DsparkConfig* c,const imparo_dspark::DsparkLayer* layers,uint32_t nl,const uint32_t* targets,uint32_t nt){using namespace imparo_dspark;
 if(!c||!layers||!targets||!nl||nl>MAX_LAYERS||!nt||nt>16||!execution_boundary_closed()||!g.weights_host||g.kv_type_k!=8||g.kv_type_v!=8)return CUDA_RC_INVALID;
 if(c->heads!=32||c->kv_heads!=8||c->head_dim!=64||c->hidden!=c->heads*c->head_dim||c->target_hidden!=c->hidden||c->block_size!=9||!c->ffn||!c->rank||!c->vocab||c->mask_token>=c->vocab||c->batch_capacity<c->block_size||c->kv_capacity<c->block_size)return CUDA_RC_INVALID;
 if(c->rank%32 || !resident_weight_range(c->markov1,uint64_t(c->vocab)*(c->rank/32)*34))return CUDA_RC_INVALID;
 if(attached){if(!attached->parked)return CUDA_RC_INVALID;const int old_rc=imparo_cuda_dspark_detach();if(old_rc)return old_rc;}
 try{attached=std::make_unique<Session>(*c,layers,nl,targets,nt);}catch(const std::bad_alloc&){return CUDA_RC_OOM;}
 attached->target_owner=active_execution_owner_id;
 return invoke([&](Session&s){s.host_weights=g.weights_host;s.host_weights_len=g.weights_len;s.rc(execution_owner_create(&s.draft_owner));s.rc(execution_owner_select(s.draft_owner));
 for(unsigned b=0;b<25;++b){uint64_t elems=uint64_t(c->batch_capacity)*c->hidden;if(b==GATE||b==UP)elems=uint64_t(c->batch_capacity)*std::max(c->ffn,c->rank);if(b==FEATURE)elems=uint64_t(c->batch_capacity)*nt*c->target_hidden;if(b==RANK)elems=c->rank;if(b==LOGITS)elems=uint64_t(c->block_size)*c->vocab;if(b==BIAS||b==COL)elems=c->vocab;s.rc(imparo_cuda_alloc(b,elems*4));s.ck(cudaMemsetAsync(execution().bufs[b],0,elems*4,g.stream));}
 std::vector<uint64_t> kv(nl,uint64_t(c->kv_capacity)*s.KW*2);s.rc(imparo_cuda_alloc_kv(nl,kv.data()));s.rc(imparo_cuda_end());s.rc(execution_owner_select(s.target_owner));});
}
// These control operations validate before mutation and do not use invoke:
// a stale resume ticket is a cache miss, not a failure of another live Session.
extern "C" int imparo_cuda_dspark_suspend(uint64_t* ticket,uint32_t* history){using namespace imparo_dspark;
 if(!ticket||!history||!attached||attached->poisoned||attached->parked||!next_park_ticket)return CUDA_RC_INVALID;
 auto& s=*attached;
 if(active_execution_owner_id!=s.target_owner||!execution_boundary_closed()||!find_execution_owner(s.draft_owner)||g.weights_host!=s.host_weights||g.weights_len!=s.host_weights_len)return CUDA_RC_INVALID;
 s.capture_enabled=false;s.feature_mask=0;s.feature_start=0;s.feature_rows=0;s.current_anchor=0;s.clear_tree_candidates();
 s.ticket=next_park_ticket++;s.parked=true;*ticket=s.ticket;*history=s.S;return 0;
}
extern "C" int imparo_cuda_dspark_resume(uint64_t ticket,uint32_t start,const imparo_dspark::DsparkConfig* c,const imparo_dspark::DsparkLayer* layers,uint32_t nl,const uint32_t* targets,uint32_t nt){using namespace imparo_dspark;
 if(!attached||attached->poisoned||!attached->parked||!ticket||!c||!layers||!targets)return CUDA_RC_INVALID;
 auto& s=*attached;
 if(ticket!=s.ticket||active_execution_owner_id!=s.target_owner||!execution_boundary_closed()||!find_execution_owner(s.draft_owner)||g.weights_host!=s.host_weights||g.weights_len!=s.host_weights_len||g.kv_type_k!=8||g.kv_type_v!=8||nl!=s.dw.size()||nt!=s.target_layers.size()||start>s.S||uint64_t(start)+s.M>s.cfg.kv_capacity)return CUDA_RC_INVALID;
 // Wire layouts have no padding (120/88-byte static assertions above).
 if(std::memcmp(c,&s.cfg,sizeof(*c))||std::memcmp(layers,s.dw.data(),size_t(nl)*sizeof(*layers))||!std::equal(s.target_layers.begin(),s.target_layers.end(),targets))return CUDA_RC_INVALID;
 // The caller proves token-prefix identity and restores the matching target
 // checkpoint. Existing committed draft KV below start remains valid; the suffix
 // is overwritten by ordinary history/generate stores, never reconstructed.
 s.S=start;s.feature_mask=0;s.feature_start=0;s.feature_rows=0;s.current_anchor=0;s.capture_enabled=false;s.ticket=0;s.parked=false;s.clear_tree_candidates();return 0;
}
extern "C" int imparo_cuda_dspark_reset(){using namespace imparo_dspark;return invoke([](Session&s){s.req(active_execution_owner_id==s.target_owner&&execution_boundary_closed(),"reset boundary");s.S=0;s.feature_mask=0;s.capture_enabled=false;s.current_anchor=0;s.clear_tree_candidates();});}
extern "C" int imparo_cuda_dspark_capture_mode(uint32_t enabled){using namespace imparo_dspark;return invoke([&](Session&s){s.req(active_execution_owner_id==s.target_owner&&execution_boundary_closed(),"capture boundary");s.capture_enabled=enabled!=0;});}
extern "C" int imparo_cuda_dspark_capture_layer(uint32_t layer,uint32_t start,uint32_t n,uint32_t src){using namespace imparo_dspark;if(!attached||!attached->capture_enabled)return 0;return invoke([&](Session&s){
 auto it=std::find(s.target_layers.begin(),s.target_layers.end(),layer);if(it==s.target_layers.end())return;unsigned slot=unsigned(it-s.target_layers.begin());s.req(active_execution_owner_id==s.target_owner&&n&&n<=s.cfg.batch_capacity&&src<B_COUNT,"feature geometry");
 if(slot==0){s.feature_start=start;s.feature_rows=n;s.feature_mask=0;}s.req(start==s.feature_start&&n==s.feature_rows&&s.feature_mask==((1u<<slot)-1),"feature order");
 auto* dest=find_execution_owner(s.draft_owner);uint64_t width=s.cfg.target_hidden;s.req(dest&&dest->bufs[FEATURE]&&execution().sizes[src]>=uint64_t(n)*width*4,"feature storage");
 s.ck(cudaMemcpy2DAsync((float*)dest->bufs[FEATURE]+slot*width,s.target_layers.size()*width*4,execution().bufs[src],width*4,width*4,n,cudaMemcpyDeviceToDevice,g.stream));s.feature_mask|=1u<<slot;});}
extern "C" int imparo_cuda_dspark_append(uint32_t start,uint32_t n){using namespace imparo_dspark;return invoke([&](Session&s){
 s.req(active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.capture_enabled,"append boundary");s.req(s.feature_mask==((1u<<s.target_layers.size())-1)&&s.feature_start==start&&n&&n<=s.feature_rows&&start==s.S&&uint64_t(start)+n+s.M<=s.cfg.kv_capacity,"accepted feature range");
 s.clear_tree_candidates();
 s.rc(execution_owner_select(s.draft_owner));mark_buf_written(FEATURE);s.history(start,n);s.S=start+n;s.feature_mask=0;s.rc(execution_owner_select(s.target_owner));});}
static int dspark_generate_mode(uint32_t start,uint32_t anchor,uint32_t* ids,float* conf,bool accept_features){using namespace imparo_dspark;return invoke([&](Session&s){
 s.req(active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.capture_enabled&&s.S==start&&uint64_t(start)+s.M<=s.cfg.kv_capacity&&ids&&conf&&anchor<s.VOC,"generate boundary");s.current_anchor=anchor;
 s.rc(execution_owner_select(s.draft_owner));auto p=s.generate(accept_features);memcpy(ids,p.ids.data(),s.M*4);memcpy(conf,p.conf.data(),s.M*4);s.rc(execution_owner_select(s.target_owner));});}
extern "C" int imparo_cuda_dspark_generate(uint32_t start,uint32_t anchor,uint32_t* ids,float* conf){return dspark_generate_mode(start,anchor,ids,conf,false);}
extern "C" int imparo_cuda_dspark_generate_with_accept_features(uint32_t start,uint32_t anchor,uint32_t* ids,float* conf){return dspark_generate_mode(start,anchor,ids,conf,true);}
extern "C" int imparo_cuda_dspark_detach(){using namespace imparo_dspark;if(!attached)return 0;auto& s=*attached;if(active_execution_owner_id!=s.target_owner||!execution_boundary_closed())return CUDA_RC_INVALID;s.capture_enabled=false;auto rc=s.draft_owner?execution_owner_release(s.draft_owner):0;if(!rc)attached.reset();return rc;}

extern "C" int imparo_cuda_dspark_tree_leaves(unsigned start,unsigned anchor,unsigned*ids){using namespace imparo_dspark;return invoke([&](Session&s){s.req(ids&&active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.S==start&&s.current_anchor==anchor&&s.tree_secondary.size()==7,"tree secondary ownership");memcpy(ids,s.tree_secondary.data(),28);});}
extern "C" int imparo_cuda_dspark_compact_features(const int*path,unsigned count){using namespace imparo_dspark;return invoke([&](Session&s){s.req(path&&count&&count<=9&&active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.feature_rows>=2&&s.feature_rows<=16&&count<=s.feature_rows&&s.feature_mask==((1u<<s.target_layers.size())-1),"tree feature ownership");for(unsigned i=0;i<count;++i)s.req(path[i]>=0&&unsigned(path[i])<s.feature_rows,"tree feature row");auto*owner=find_execution_owner(s.draft_owner);s.req(owner&&owner->bufs[FEATURE],"tree feature buffer");auto&t=execution().tree;const unsigned rowbytes=s.target_layers.size()*s.cfg.target_hidden*4;const uint64_t bytes=uint64_t(count)*rowbytes;s.req(execution().device_task_scratch&&bytes<=2*1024*1024&&t.commit_offset+bytes<=execution().device_task_scratch_bytes,"tree feature scratch");auto*base=static_cast<uint8_t*>(execution().device_task_scratch);auto*ids=reinterpret_cast<int*>(base+128);auto*tmp=base+t.commit_offset;s.ck(cudaMemcpyAsync(ids,path,count*4,cudaMemcpyHostToDevice,g.stream));tree_commit_gather<<<(bytes+255)/256,256,0,g.stream>>>(static_cast<const uint8_t*>(owner->bufs[FEATURE]),tmp,ids,0,rowbytes,count,nullptr);s.ck(cudaMemcpyAsync(owner->bufs[FEATURE],tmp,bytes,cudaMemcpyDeviceToDevice,g.stream));s.ck(cudaStreamSynchronize(g.stream));s.feature_rows=count;});}

extern "C" int imparo_cuda_dspark_frontier_tree(unsigned start,unsigned anchor,unsigned*ids,int*parents){using namespace imparo_dspark;return invoke([&](Session&s){s.req(ids&&parents&&active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.S==start&&s.current_anchor==anchor&&s.tree_frontier.size()==16,"frontier ownership");for(unsigned i=0;i<16;++i){ids[i]=s.tree_frontier[i].token;parents[i]=s.tree_frontier[i].parent;}});}

// Scores are already on the host from frontier_generate; this adds no GPU work.
extern "C" int imparo_cuda_dspark_frontier_scores(unsigned start,unsigned anchor,float*scores){using namespace imparo_dspark;return invoke([&](Session&s){s.req(scores&&active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.S==start&&s.current_anchor==anchor&&s.tree_frontier.size()==16,"frontier score ownership");for(unsigned i=0;i<16;++i)scores[i]=s.tree_frontier[i].score;});}

// Static-provider bridge only: the full beam pool is already on the host.
// Clearing at generate/reset/append/suspend/resume binds it to this live round.
extern "C" int imparo_cuda_dspark_frontier_candidates(unsigned start,unsigned anchor,unsigned*ids,int*parents,float*scores){using namespace imparo_dspark;return invoke([&](Session&s){s.req(ids&&parents&&scores&&active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.S==start&&s.current_anchor==anchor&&s.tree_frontier.size()==16&&s.tree_candidates.size()==33,"frontier candidate ownership");for(unsigned i=0;i<33;++i){ids[i]=s.tree_candidates[i].token;parents[i]=s.tree_candidates[i].parent;scores[i]=s.tree_candidates[i].score;}});}

extern "C" int imparo_cuda_dspark_frontier_groups(unsigned start,unsigned anchor,imparo_dspark::FrontierGroup*groups){using namespace imparo_dspark;return invoke([&](Session&s){s.req(groups&&active_execution_owner_id==s.target_owner&&execution_boundary_closed()&&s.S==start&&s.current_anchor==anchor&&s.tree_candidates.size()==33&&s.tree_groups.size()==29,"frontier group ownership");memcpy(groups,s.tree_groups.data(),29*sizeof(FrontierGroup));});}
