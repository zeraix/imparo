#pragma once
// Include in imparo_cuda.cu after tree/DSpark/NativeReplayGraph definitions.
// Also include at end of sm80/d64_fa2/dispatch.cu with
// IMPARO_TREE_REPLAY_FA2_BRIDGE defined. No new device arithmetic.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>
static bool tree_replay_reject(unsigned line){std::fprintf(stderr,"[tree-graph-plan] rejected line=%u\n",line);return false;}
struct TreeReplayGeometry {
 uint32_t start=0,fixed_span=0,chunks=0,begin=0,tail=0,leaf_mask=0,leaves=0;
 uint64_t ordinary_bytes=0;
 bool batch_invariant=false;
};
struct TreeReplayStorage {uint8_t*attention=nullptr,*task=nullptr;uint64_t mask_offset=0;};
struct TreeReplayFa2Kernels {
 void*custom=nullptr,*pack=nullptr,*tail=nullptr,*splice=nullptr;
 void*merge_small=nullptr,*merge_large=nullptr,*prepare_q=nullptr,*output=nullptr;
 size_t params_bytes=0;bool mapped_tail=false;bool batch_invariant=false;
};
TreeReplayFa2Kernels tree_replay_fa2_kernels(bool batch_invariant=false);
bool tree_replay_fa2_params(void*,bool,const TreeReplayGeometry&,const TreeReplayStorage&,bool);

#if defined(IMPARO_TREE_REPLAY_FA2_BRIDGE)
TreeReplayFa2Kernels tree_replay_fa2_kernels(bool batch_invariant){
 using namespace imparo_d64_fa2_port;using namespace tree_tail_detail;
 if(batch_invariant)return {(void*)flashinfer::SinglePrefillWithKVCacheKernel<CT,PP>,
 (void*)mapped_tail_detail::pack_mapped,(void*)batch_invariant_tree::tail,(void*)batch_invariant_tree::splice_v1,
 (void*)merge_v1,nullptr,(void*)prepare_q,(void*)output_f32,sizeof(PP),true,true};
 return {(void*)flashinfer::SinglePrefillWithKVCacheKernel<CT,PP>,
 tree_tail_shared_kv_enabled()?(void*)mapped_tail_detail::pack_mapped:(void*)pack_tails,
 tree_tail_shared_kv_enabled()?(void*)mapped_tail_detail::tail_mapped:(void*)tail_batch,(void*)splice,
 (void*)flashinfer::MergeStatesKernel<8,__half,__half>,
 (void*)flashinfer::MergeStatesLargeNumIndexSetsKernel<8,8,16,4,__half,__half>,
 (void*)prepare_q,(void*)output_f32,sizeof(PP),tree_tail_shared_kv_enabled()};
}
// validate=true compares the captured argument; false patches an owned copy.
// The actual Params definition belongs to this TU, never an ABI reconstruction.
bool tree_replay_fa2_params(void*blob,bool tail,const TreeReplayGeometry&g,
 const TreeReplayStorage&s,bool validate){
 using PP=imparo_d64_fa2_port::PartitionParams;PP p;std::memcpy(&p,blob,sizeof(p));
 constexpr uint64_t QW=2048,KW=512,QBYTES=16*QW*sizeof(__half);
 const uint32_t ordinary_chunks=uint32_t((uint64_t(g.start)+16+g.fixed_span-1)/g.fixed_span);
 __half*q,*k=p.k,*v=p.v,*o;float*lse;
 if(tail){
  q=reinterpret_cast<__half*>(s.attention+g.ordinary_bytes);
  k=q+uint64_t(g.leaves)*9*QW;v=k+uint64_t(g.leaves)*g.tail*KW;
  o=v+uint64_t(g.leaves)*g.tail*KW;lse=reinterpret_cast<float*>(o+uint64_t(g.leaves)*9*QW*(g.batch_invariant?(g.tail+g.fixed_span-1)/g.fixed_span:1));
 }else{
  q=reinterpret_cast<__half*>(s.attention);o=reinterpret_cast<__half*>(s.attention+2*QBYTES);
  lse=reinterpret_cast<float*>(o+uint64_t(ordinary_chunks)*16*QW);
 }
 if(p.qo_len!=(tail?9u:16u)||p.num_qo_heads!=32||p.num_kv_heads!=8||p.head_dim!=64
  ||p.q_stride_n!=QW||p.q_stride_h!=64||p.k_stride_n!=KW||p.v_stride_n!=KW
  ||p.k_stride_h!=64||p.v_stride_h!=64||!p.partition_kv||p.window_left!=-1
  ||p.maybe_alibi_slopes||p.logits_soft_cap!=0.f||p.sm_scale!=1.f)return tree_replay_reject(__LINE__);
 if(validate)return p.q==q&&p.k==k&&p.v==v&&p.o==o&&p.lse==lse
  &&p.kv_len==(tail?g.tail:g.start+16)&&p.partition_fixed_span==g.fixed_span
  &&p.batch_invariant_q8_v1==unsigned(g.batch_invariant)
  &&p.partition_reference_len==(tail||g.batch_invariant?0u:g.start+9)
  &&p.maybe_custom_mask==(tail?nullptr:s.task+s.mask_offset);
 p.q=q;p.k=k;p.v=v;p.o=o;p.lse=lse;p.kv_len=tail?g.tail:g.start+16;
 p.partition_fixed_span=g.fixed_span;p.partition_reference_len=tail||g.batch_invariant?0u:g.start+9;
 p.batch_invariant_q8_v1=unsigned(g.batch_invariant);
 p.maybe_custom_mask=tail?nullptr:s.task+s.mask_offset;
 std::memcpy(blob,&p,sizeof(p));return true;
}
#else
// Caller owns graph/memory leases, target rollback, topology/token H2D, scratch
// roles/epochs, feature publication and final D2H. All allocation is preflighted
// OUTSIDE capture. Do not invoke NativeReplayGraph::replay on this plan: FA2
// Params are by-value structures, beyond its start/valid scalar updater.
class TreeReplayPlan{
 enum class Kind{Rope,Head,Store,Dequant,Custom,Pack,Tail,Splice,Merge};
 struct Node{
  cudaGraphNode_t node=nullptr;cudaKernelNodeParams p{};Kind kind=Kind::Rope;
  std::vector<std::vector<uint8_t>>bytes;std::vector<void*>args;bool ordinary=false;
  bool bind(const std::vector<size_t>&sizes){
   if(!p.kernelParams||p.extra)return tree_replay_reject(__LINE__);bytes.resize(sizes.size());args.resize(sizes.size());
   for(size_t i=0;i<sizes.size();++i){if(!p.kernelParams[i])return tree_replay_reject(__LINE__);
    bytes[i].resize(sizes[i]);std::memcpy(bytes[i].data(),p.kernelParams[i],sizes[i]);}
   return true;
  }
  template<class T>T get(size_t i)const{T v{};std::memcpy(&v,bytes[i].data(),sizeof(v));return v;}
  template<class T>void put(size_t i,T v){std::memcpy(bytes[i].data(),&v,sizeof(v));}
  void refresh(){for(size_t i=0;i<bytes.size();++i)args[i]=bytes[i].data();p.kernelParams=args.data();p.extra=nullptr;}
 };
 std::vector<Node>nodes_;TreeReplayGeometry captured_{};TreeReplayStorage storage_{};
 cudaGraph_t source_=nullptr;cudaGraphExec_t exec_=nullptr;TreeReplayFa2Kernels fa_{};
 uint32_t attention_layers_=0,ordinary_chunks_=0;bool ready_=false;
 static constexpr size_t P=sizeof(void*),U=sizeof(uint32_t),F=sizeof(float);
 static bool page_identity(const void*cache,const void*pages){
  for(unsigned layer=0;layer<MAX_LAYERS;++layer){
   if(cache==execution().kv_k[layer]||cache==execution().kv_v[layer])
    return cache&&pages==kv_device_page_table(layer)&&!kv_page_table_requires_mapping(layer);
  }
  return false;
 }
 static uint32_t popcount(uint32_t x){uint32_t n=0;while(x){x&=x-1;++n;}return n;}
 static uint32_t ordinary_chunks(const TreeReplayGeometry&g){return uint32_t((uint64_t(g.start)+16+g.fixed_span-1)/g.fixed_span);}
 static bool geometry_valid(const TreeReplayGeometry&g){
  if(g.batch_invariant){
   if(g.start>UINT32_MAX-16||g.fixed_span<256||g.fixed_span%128||!g.chunks
    ||!g.leaf_mask||(g.leaf_mask&~0xffffu)||g.leaves!=popcount(g.leaf_mask)||g.leaves>16
    ||g.chunks!=(uint64_t(g.start)+16+g.fixed_span-1)/g.fixed_span
    ||g.begin!=(g.start/g.fixed_span)*g.fixed_span
    ||uint64_t(g.begin)+g.tail!=uint64_t(g.start)+9)return tree_replay_reject(__LINE__);
   return g.ordinary_bytes==imparo_d64_fa2_port::workspace_bytes(16,32,g.start+16,true);
  }
  if(g.start<4096||g.start>UINT32_MAX-16||g.fixed_span<256||g.fixed_span%128
   ||g.chunks<=1||!g.leaf_mask||(g.leaf_mask&~0xffffu)||g.leaves!=popcount(g.leaf_mask)||g.leaves>16
   ||g.chunks!=(uint64_t(g.start)+9+g.fixed_span-1)/g.fixed_span
   ||uint64_t(g.begin)!=uint64_t(g.chunks-1)*g.fixed_span||g.begin>g.start
   ||uint64_t(g.begin)+g.tail!=uint64_t(g.start)+9)return tree_replay_reject(__LINE__);
  return g.ordinary_bytes==imparo_d64_fa2_port::workspace_bytes(16,32,g.start+16,true);
 }
 bool storage_valid(const TreeReplayGeometry&g)const{
  const auto&e=execution();
  const unsigned tail_chunks=g.batch_invariant?(g.tail+g.fixed_span-1)/g.fixed_span:1;
  const uint64_t extra=uint64_t(g.leaves)*(9ull*2048*sizeof(__half)+2ull*g.tail*512*sizeof(__half)
   +9ull*tail_chunks*(2048*sizeof(__half)+32*sizeof(float)));
  uint32_t mask=0xffffu;for(unsigned i=1;i<16;++i){
   if(e.tree.parents[i]<0||unsigned(e.tree.parents[i])>=i)return tree_replay_reject(__LINE__);
   mask&=~(1u<<unsigned(e.tree.parents[i]));
  }
  return e.tree.nodes==16&&e.tree.start==g.start&&mask==g.leaf_mask
   &&e.attention_scratch==storage_.attention&&e.device_task_scratch==storage_.task
   &&e.tree.mask_offset==storage_.mask_offset
   &&g.ordinary_bytes<=e.attention_scratch_bytes&&extra<=e.attention_scratch_bytes-g.ordinary_bytes
   &&storage_.mask_offset<=e.device_task_scratch_bytes
   &&(16ull*(uint64_t(g.start)+16)+7)/8<=e.device_task_scratch_bytes-storage_.mask_offset;
 }
 bool patch(Node&n,const TreeReplayGeometry&g){
  constexpr uint64_t QW=2048,KW=512,QBYTES=16*QW*sizeof(__half);
  auto*tq=reinterpret_cast<__half*>(storage_.attention+g.ordinary_bytes);
  auto*tk=tq+uint64_t(g.leaves)*9*QW;auto*tv=tk+uint64_t(g.leaves)*g.tail*KW;
  auto*to=tv+uint64_t(g.leaves)*g.tail*KW;
  const unsigned tail_chunks=g.batch_invariant?(g.tail+g.fixed_span-1)/g.fixed_span:1;
  auto*tl=reinterpret_cast<float*>(to+uint64_t(g.leaves)*9*tail_chunks*QW);
  auto*tmp=reinterpret_cast<__half*>(storage_.attention+2*QBYTES);
  switch(n.kind){
  case Kind::Rope:case Kind::Head:n.put<uint32_t>(6,g.start);break;
  case Kind::Store:n.put<uint32_t>(3,g.start);break;
  case Kind::Dequant:n.put<uint32_t>(3,g.start+16);n.p.gridDim.y=g.start+16;break;
  case Kind::Custom:
   if(!tree_replay_fa2_params(n.bytes[0].data(),false,g,storage_,false))return tree_replay_reject(__LINE__);
   n.p.gridDim.y=ordinary_chunks(g);break;
  case Kind::Pack:
   n.put<void*>(3,tq);n.put<void*>(4,tk);n.put<void*>(5,tv);n.put<uint32_t>(7,g.leaf_mask);
   n.put<uint32_t>(8,g.start);n.put<uint32_t>(9,g.begin);n.put<uint32_t>(10,g.tail);
   n.p.gridDim.x=fa_.mapped_tail?uint32_t((9*QW+255)/256):uint32_t((std::max(9*QW,uint64_t(g.tail)*KW)+255)/256);n.p.gridDim.y=g.leaves;break;
  case Kind::Tail:
   if(!tree_replay_fa2_params(n.bytes[0].data(),true,g,storage_,false))return tree_replay_reject(__LINE__);
   n.put<uint32_t>(1,g.tail);n.p.gridDim.y=g.leaves;if(g.batch_invariant)n.p.gridDim.x=tail_chunks;break;
  case Kind::Splice:
   n.put<void*>(0,to);n.put<void*>(1,tl);n.put<void*>(3,reinterpret_cast<float*>(tmp+uint64_t(g.chunks)*16*QW));
   n.put<uint32_t>(5,g.leaf_mask);n.put<uint32_t>(6,g.chunks);
   if(g.batch_invariant){n.put<uint32_t>(7,g.begin/g.fixed_span);n.put<uint32_t>(8,tail_chunks);}break;
  case Kind::Merge:{
   if(g.batch_invariant){
    n.put<void*>(1,reinterpret_cast<float*>(tmp+uint64_t(g.chunks)*16*QW));
    n.put<uint32_t>(4,g.chunks);n.put<uint32_t>(5,g.start);break;
   }
   const uint32_t c=n.ordinary?ordinary_chunks(g):g.chunks;
   n.put<void*>(1,reinterpret_cast<float*>(tmp+uint64_t(c)*16*QW));n.put<uint32_t>(4,c);break;
  }}
  return true;
 }
public:
 // Unknown kernel identities fail closed. static_kernels is the caller's explicit
 // reviewed numerical-route whitelist, NOT a list of everything observed in a graph.
 // Fixed D2D feature nodes retain original addresses/extents; their memory leases,
 // including draft FEATURE storage, belong in the caller's full storage signature.
 bool configure(NativeReplayGraph&graph,const TreeReplayGeometry&g,
  uint32_t expected_attention_layers,const std::vector<void*>&static_kernels)noexcept{
  clear();try{
   if(!graph.graph||!graph.exec||!graph.nodes.empty()||!expected_attention_layers||!geometry_valid(g))return tree_replay_reject(__LINE__);
   captured_=g;storage_={static_cast<uint8_t*>(execution().attention_scratch),static_cast<uint8_t*>(execution().device_task_scratch),execution().tree.mask_offset};
   if(!storage_.attention||!storage_.task||!storage_valid(g))return tree_replay_reject(__LINE__);
   fa_=tree_replay_fa2_kernels(g.batch_invariant);if(!fa_.params_bytes)return tree_replay_reject(__LINE__);
   uint32_t heads=0,stores=0,dequants=0,customs=0,packs=0,tails=0,splices=0,merges=0,phase=0;
   for(auto handle:graph.ordered_nodes()){
    cudaGraphNodeType type;if(cudaGraphNodeGetType(handle,&type)!=cudaSuccess)return tree_replay_reject(__LINE__);
    if(type==cudaGraphNodeTypeEmpty)continue;
    if(type==cudaGraphNodeTypeMemcpy){
     cudaMemcpy3DParms c{};
     if(cudaGraphMemcpyNodeGetParams(handle,&c)!=cudaSuccess||c.kind!=cudaMemcpyDeviceToDevice
      ||c.srcArray||c.dstArray||!c.srcPtr.ptr||!c.dstPtr.ptr)return tree_replay_reject(__LINE__);continue;
    }
    if(type!=cudaGraphNodeTypeKernel)return tree_replay_reject(__LINE__);
    Node n;n.node=handle;
    if(cudaGraphKernelNodeGetParams(handle,&n.p)!=cudaSuccess||!n.p.kernelParams||n.p.extra)return tree_replay_reject(__LINE__);
    auto is=[&](void*p){return graph_kernel_is(n.p.func,p);};
    if(is((void*)k_tree_rope)){
     n.kind=Kind::Rope;if(!n.bind({P,P,U,F,U,U,U,U,P,P}))return tree_replay_reject(__LINE__);
     if(n.get<uint32_t>(6)!=g.start||n.get<uint32_t>(7)!=16||n.get<void*>(8)
      ||n.get<void*>(9)!=storage_.task+64)return tree_replay_reject(__LINE__);++heads;
    }else if(is((void*)k_tree_head_norm_rope_hadamard<64,true>)||is((void*)k_tree_head_norm_rope_hadamard<128,true>)
     ||is((void*)k_tree_head_norm_rope_hadamard<256,true>)||is((void*)k_tree_head_norm_rope_hadamard<1024,true>)){
     n.kind=Kind::Head;if(!n.bind({P,P,P,U,F,U,U,U,F,U,F,P,P,P}))return tree_replay_reject(__LINE__);
     if(n.get<uint32_t>(3)!=64||(n.get<uint32_t>(5)!=8&&n.get<uint32_t>(5)!=32)
      ||n.p.gridDim.x!=16*n.get<uint32_t>(5)||n.get<uint32_t>(6)!=g.start
      ||n.get<void*>(11)||n.get<void*>(12)||n.get<void*>(13)!=storage_.task+64)return tree_replay_reject(__LINE__);++heads;
    }else if(is((void*)k_kv_store_q8)){
     n.kind=Kind::Store;if(!n.bind({P,P,U,U,U,U,P,P}))return tree_replay_reject(__LINE__);
     if(n.get<uint32_t>(2)!=512||n.get<uint32_t>(3)!=g.start||n.get<uint32_t>(4)!=16
      ||n.get<uint32_t>(5)||n.get<void*>(6)||!page_identity(n.get<void*>(1),n.get<void*>(7)))return tree_replay_reject(__LINE__);++stores;
    }else if(is((void*)imparo_sm80_kv::dequant_parallel<8>)||is((void*)k_kv_dequant)){
     n.kind=Kind::Dequant;const bool fallback=is((void*)k_kv_dequant);
     if(!n.bind(fallback?std::vector<size_t>{P,P,U,U,U,U,P}:std::vector<size_t>{P,P,U,U,U,P}))return tree_replay_reject(__LINE__);
     if(n.get<uint32_t>(2)!=512||n.get<uint32_t>(3)!=g.start+16||n.p.gridDim.y!=g.start+16
      ||n.get<uint32_t>(fallback?5:4)||!page_identity(n.get<void*>(0),n.get<void*>(fallback?6:5))||(fallback&&n.get<uint32_t>(4)!=8))return tree_replay_reject(__LINE__);++dequants;
    }else if(is(fa_.custom)){
     n.kind=Kind::Custom;if(phase||!n.bind({fa_.params_bytes})
      ||!tree_replay_fa2_params(n.bytes[0].data(),false,g,storage_,true)||n.p.gridDim.y!=ordinary_chunks(g))return tree_replay_reject(__LINE__);
     phase=1;++customs;
    }else if(is(fa_.pack)){
     n.kind=Kind::Pack;if(phase!=(g.batch_invariant?1u:2u)||!n.bind({P,P,P,P,P,P,P,U,U,U,U}))return tree_replay_reject(__LINE__);
     if(n.get<void*>(0)!=storage_.attention||n.get<void*>(6)!=storage_.task)return tree_replay_reject(__LINE__);
     phase=3;++packs;
    }else if(is(fa_.tail)){
     n.kind=Kind::Tail;if(phase!=3||!n.bind({fa_.params_bytes,U})
      ||!tree_replay_fa2_params(n.bytes[0].data(),true,g,storage_,true))return tree_replay_reject(__LINE__);phase=4;++tails;
    }else if(is(fa_.splice)){
     n.kind=Kind::Splice;if(phase!=4||!n.bind(g.batch_invariant?std::vector<size_t>{P,P,P,P,P,U,U,U,U}:std::vector<size_t>{P,P,P,P,P,U,U})
      ||n.get<void*>(2)!=storage_.attention+2*16ull*2048*sizeof(__half)||n.get<void*>(4)!=storage_.task)return tree_replay_reject(__LINE__);
     phase=5;++splices;
    }else if(g.batch_invariant&&is(fa_.merge_small)){
     n.kind=Kind::Merge;if(phase!=5||!n.bind({P,P,P,U,U,U,U,P}))return tree_replay_reject(__LINE__);
     if(n.get<void*>(0)!=storage_.attention+2*16ull*2048*sizeof(__half)
      ||n.get<void*>(2)!=storage_.attention+16ull*2048*sizeof(__half)
      ||n.get<uint32_t>(3)!=16||n.get<uint32_t>(4)!=g.chunks||n.get<uint32_t>(5)!=g.start
      ||n.get<uint32_t>(6)!=g.fixed_span||n.get<void*>(7)!=storage_.task)return tree_replay_reject(__LINE__);
     phase=0;++merges;
    }else if(is(fa_.merge_small)||(fa_.merge_large&&is(fa_.merge_large))){
     n.kind=Kind::Merge;if(phase!=1&&phase!=5)return tree_replay_reject(__LINE__);n.ordinary=phase==1;
     const uint32_t c=n.ordinary?ordinary_chunks(g):g.chunks;const bool large=is(fa_.merge_large);
     if(large!=(c>=16)||!n.bind(large?std::vector<size_t>{P,P,P,P,U,U}:std::vector<size_t>{P,P,P,P,U,U,U}))return tree_replay_reject(__LINE__);
     if(n.get<void*>(0)!=storage_.attention+2*16ull*2048*sizeof(__half)
      ||n.get<void*>(2)!=storage_.attention+16ull*2048*sizeof(__half)||n.get<void*>(3)
      ||n.get<uint32_t>(4)!=c||n.get<uint32_t>(5)!=32||(!large&&n.get<uint32_t>(6)!=64))return tree_replay_reject(__LINE__);
     phase=n.ordinary?2:0;++merges;
    }else{
     if(!is(fa_.prepare_q)&&!is(fa_.output)&&std::none_of(static_kernels.begin(),static_kernels.end(),is))return tree_replay_reject(__LINE__);
     continue;
    }
    // Every rewritten field must reproduce the original capture byte-for-byte,
    // including leaf-mask, computed scratch pointers and full grid dimensions.
    const auto old=n.bytes;const dim3 grid=n.p.gridDim;
    if(!patch(n,g)||old!=n.bytes||grid.x!=n.p.gridDim.x||grid.y!=n.p.gridDim.y||grid.z!=n.p.gridDim.z)return tree_replay_reject(__LINE__);
    nodes_.push_back(std::move(n));
   }
   if(phase||customs!=expected_attention_layers||packs!=customs||tails!=customs||splices!=customs
    ||merges!=(g.batch_invariant?customs:2*customs)||heads!=2*customs||stores!=2*customs||dequants!=2*customs)return tree_replay_reject(__LINE__);
   attention_layers_=customs;ordinary_chunks_=ordinary_chunks(g);source_=graph.graph;exec_=graph.exec;ready_=true;return true;
  }catch(...){clear();return tree_replay_reject(__LINE__);}
 }
 cudaError_t replay(NativeReplayGraph&graph,const TreeReplayGeometry&g,cudaStream_t stream)noexcept{
  if(!ready_||graph.graph!=source_||graph.exec!=exec_||!graph.nodes.empty()||!stream||!geometry_valid(g)
   ||g.batch_invariant!=captured_.batch_invariant||g.fixed_span!=captured_.fixed_span||g.chunks!=captured_.chunks||ordinary_chunks(g)!=ordinary_chunks_
   ||!storage_valid(g))return cudaErrorInvalidValue;
  // No GPU execution occurs until all executable-node updates have succeeded.
  // Invalid geometry requires recapture; CUDA errors must propagate, never double-run.
  for(auto&n:nodes_){
   if(!patch(n,g))return cudaErrorInvalidValue;n.refresh();
   const auto rc=cudaGraphExecKernelNodeSetParams(exec_,n.node,&n.p);if(rc!=cudaSuccess)return rc;
  }
  return cudaGraphLaunch(exec_,stream);
 }
 void clear()noexcept{nodes_.clear();source_=nullptr;exec_=nullptr;ready_=false;attention_layers_=0;ordinary_chunks_=0;}
 size_t dynamic_nodes()const noexcept{return nodes_.size();}
 uint32_t attention_layers()const noexcept{return attention_layers_;}
};
// Exact current target16 symbols, mapped from the first complete tree transaction
// in lfm-current-long-decode-cost-2026-09-12-001/trace.sqlite:
// 15272506709..15295368580 ns (embedding16 through argmax16). This does not
// admit arbitrary future template variants. A changed route fails configure.
static std::vector<void*> tree_replay_static_kernels(){
 std::vector<void*> kernels = {
  (void*)k_add, (void*)k_scale, (void*)k_quantize_q8_1_mmq, (void*)k_rows_q8_0,
  (void*)tree_shortconv,
  (void*)imparo_sm80_q8_mmq::m9_parallel_k3_partials<false>,
  (void*)imparo_sm80_q8_mmq::m9_parallel_k3_combine,
  (void*)imparo_sm80_q8_mmq::q8_0_q8_1_mma<16,true,false,0,false,4,4,64,true>,
  (void*)imparo_sm86_q8_tm_gate_up_row_pair_lab::q8_tm_gate_up_row_pair_tail<2>,
  (void*)k_argmax_rows<1024>, (void*)k_hadamard64_warp<false>,
  (void*)k_rms_norm_ggml<1024,0,false,false,true,true,true>,
  (void*)k_rms_norm_q8_1_mmq<1024,true,false,false>
 };
 if(batch_invariant_q8_v1_active()){
  // Extend only this owner's replay admission to retained Q8 specializations.
  kernels.push_back((void*)imparo_sm80_q8_mmq::m9_parallel_k3_partials<true>);
  kernels.push_back((void*)imparo_sm80_q8_mmq::q8_0_q8_1_mma<16,true,true,0,false,4,4,64,true>);
  kernels.push_back((void*)imparo_sm80_q8_mmq::q8_0_q8_1_mma<16,true,false,0,false,4,2,64,true>);
  kernels.push_back((void*)imparo_sm80_q8_mmq::q8_0_q8_1_mma<16,true,true,0,false,4,2,64,true>);
  kernels.push_back((void*)k_rms_norm_q8_1_mmq<256,true,false,false>);
  kernels.push_back((void*)k_add_rms_norm_q8_1_mmq<1024>);
  kernels.push_back((void*)k_add_rms_norm_q8_1_mmq<256>);
  kernels.push_back((void*)k_add_rms_norm_q8_1_mmq_vec4_512);
 }
 return kernels;
}

#endif
