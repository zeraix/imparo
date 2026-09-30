#pragma once
// Private target-tree capture preserves the ordinary Q8/RMS/FFN dispatch flags.
// The owner retains graph argument storage; every replay rechecks live allocations.
#include "tree_replay_plan.cuh"
struct TreeReplayState {
 NativeReplayGraph graph;
 TreeReplayPlan plan;
 TreeReplayGeometry geometry;
 std::vector<uint64_t> storage;
 void* q8_in=nullptr;void* q8_next_in=nullptr;
 uint64_t q8_in_bytes=0,q8_next_in_bytes=0;
 void* q8_out=nullptr;void* q8_next_out=nullptr;
 uint64_t q8_out_bytes=0,q8_next_out_bytes=0;
 unsigned attention_layers=0;
 bool warmed=false,proof_done=false,blocked=false;
 uint64_t captures=0,replays=0;
 void clear_graph(){plan.clear();graph.clear();storage.clear();}
};
static bool tree_replay_flag(const char*name){
 const char*p=std::getenv(name);return p&&std::strcmp(p,"1")==0;
}
static std::vector<uint64_t> tree_replay_storage(){
 auto&e=execution();auto*s=imparo_dspark::attached.get();
 std::vector<uint64_t> out;out.reserve(500);
 auto p=[&](const void*v){out.push_back(reinterpret_cast<uintptr_t>(v));};
 auto u=[&](uint64_t v){out.push_back(v);};
 p(&e);u(active_execution_owner_id);p(g.stream);p(g.weights);p(g.weights_host);
 u(g.weights_len);u(g.weights_device_bytes);u(g.choice_epoch);u(g.sm_version);
 u(g.kv_type_k);u(g.kv_type_v);u(batch_invariant_q8_v1_active());for(auto k:g.knobs)u(k);
  u(imparo_d64_fa2_port::tree_tail_shared_kv_enabled()); // transient tail layout and kernel identity
 for(unsigned i=0;i<B_COUNT;++i){p(e.bufs[i]);u(e.sizes[i]);}
 for(unsigned i=0;i<MAX_LAYERS;++i){p(e.kv_k[i]);p(e.kv_v[i]);u(e.kv_bytes[i]);p(kv_device_page_table(i));}
 // Slots may exchange logical ownership. Physical identities and capacities cannot.
 if(reinterpret_cast<uintptr_t>(e.q8_scratch)<reinterpret_cast<uintptr_t>(e.q8_scratch_next)){
  p(e.q8_scratch);u(e.q8_scratch_bytes);p(e.q8_scratch_next);u(e.q8_scratch_next_bytes);
 }else{p(e.q8_scratch_next);u(e.q8_scratch_next_bytes);p(e.q8_scratch);u(e.q8_scratch_bytes);}
 p(e.attention_scratch);u(e.attention_scratch_bytes);
 p(e.attention_q_cache);u(e.attention_q_cache_bytes);
 p(e.device_task_scratch);u(e.device_task_scratch_bytes);
 p(e.rope_freqs);u(e.rope_freqs_bytes);p(e.rope_freqs_host);u(e.rope_freqs_host_bytes);
 u(e.tree.recurrent);u(e.tree.state_offset);u(e.tree.mask_offset);
 p(s);if(s){
  u(s->draft_owner);u(s->target_owner);u(s->cfg.target_hidden);u(s->cfg.batch_capacity);
  auto*d=find_execution_owner(s->draft_owner);p(d);
  if(d){p(d->bufs[imparo_dspark::FEATURE]);u(d->sizes[imparo_dspark::FEATURE]);}
  for(auto l:s->target_layers)u(l);
 }
 return out;
}
static void tree_replay_postconditions(TreeReplayState&r){
 auto&e=execution();auto*s=imparo_dspark::attached.get();
 e.q8_scratch=r.q8_out;e.q8_scratch_bytes=r.q8_out_bytes;
 e.q8_scratch_next=r.q8_next_out;e.q8_scratch_next_bytes=r.q8_next_out_bytes;
 invalidate_q8_cache();e.kdq.clear();e.vdq.clear();e.attention_q_src=UINT32_MAX;
 for(unsigned i=0;i<B_COUNT;++i)mark_buf_written(i);
 e.epilogue=0; // Current BatchGeometry was freshly published by the model.
 s->feature_start=e.tree.start;s->feature_rows=16;
 s->feature_mask=(1u<<s->target_layers.size())-1;
}
static bool tree_replay_feature_nodes(NativeReplayGraph&graph){
 auto*s=imparo_dspark::attached.get();auto*d=find_execution_owner(s->draft_owner);
 const size_t width=size_t(s->cfg.target_hidden)*4,pitch=s->target_layers.size()*width;
 size_t slot=0;
 for(auto node:graph.ordered_nodes()){
  cudaGraphNodeType type;if(cudaGraphNodeGetType(node,&type)!=cudaSuccess)return false;
  if(type!=cudaGraphNodeTypeMemcpy)continue;
  cudaMemcpy3DParms c{};
  if(slot>=s->target_layers.size()||cudaGraphMemcpyNodeGetParams(node,&c)!=cudaSuccess
   ||c.kind!=cudaMemcpyDeviceToDevice||c.srcArray||c.dstArray
   ||c.srcPtr.ptr!=execution().bufs[0]||c.srcPtr.pitch!=width
   ||c.dstPtr.ptr!=static_cast<uint8_t*>(d->bufs[imparo_dspark::FEATURE])+slot*width
   ||c.dstPtr.pitch!=pitch||c.extent.width!=width||c.extent.height!=16||c.extent.depth!=1
   ||c.srcPos.x||c.srcPos.y||c.srcPos.z||c.dstPos.x||c.dstPos.y||c.dstPos.z)return false;
  ++slot;
 }
 return slot==s->target_layers.size();
}
static void tree_replay_check(cudaError_t e){
 if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
static std::vector<uint8_t> tree_replay_image(bool committed_only){
 auto&e=execution();auto*s=imparo_dspark::attached.get();auto*d=find_execution_owner(s->draft_owner);
 std::vector<uint8_t> v;
 auto read=[&](const void*p,uint64_t n){
  const auto at=v.size();v.resize(at+size_t(n));
  if(n)tree_replay_check(cudaMemcpyAsync(v.data()+at,p,size_t(n),cudaMemcpyDeviceToHost,g.stream));
  tree_replay_check(cudaStreamSynchronize(g.stream));
 };
 read(e.bufs[25],uint64_t(e.tree.recurrent)*4);
 const uint64_t prefix=uint64_t(e.tree.start)*544,tail=prefix+16*544;
 for(unsigned i=0;i<MAX_LAYERS;++i)if(e.kv_k[i]){
  for(auto*p:{e.kv_k[i],e.kv_v[i]}){
   if(committed_only){read(p,prefix);read(static_cast<uint8_t*>(p)+tail,e.kv_bytes[i]-tail);}
   else read(p,e.kv_bytes[i]);
  }
 }
 if(!committed_only){
  read(e.bufs[12],16ull*s->cfg.vocab*4);read(e.bufs[14],16*4);
  read(static_cast<uint8_t*>(e.device_task_scratch)+e.tree.state_offset,16ull*e.tree.recurrent*4);
  read(d->bufs[imparo_dspark::FEATURE],16ull*s->target_layers.size()*s->cfg.target_hidden*4);
 }
 return v;
}
struct TreeEagerNode {
 cudaGraphNodeType type{};cudaKernelNodeParams kernel{};cudaMemcpy3DParms copy{};
};
static std::vector<TreeEagerNode> tree_replay_eager_plan(NativeReplayGraph&graph){
 std::vector<TreeEagerNode> plan;
 for(auto node:graph.ordered_nodes()){
  TreeEagerNode n;tree_replay_check(cudaGraphNodeGetType(node,&n.type));
  if(n.type==cudaGraphNodeTypeKernel)tree_replay_check(cudaGraphKernelNodeGetParams(node,&n.kernel));
  else if(n.type==cudaGraphNodeTypeMemcpy)tree_replay_check(cudaGraphMemcpyNodeGetParams(node,&n.copy));
  else if(n.type!=cudaGraphNodeTypeEmpty)throw std::runtime_error("unexpected eager tree node");
  plan.push_back(n);
 }
 return plan;
}
static void tree_replay_eager_nodes(const std::vector<TreeEagerNode>&nodes){
 for(const auto&n:nodes){
  if(n.type==cudaGraphNodeTypeKernel){
   const auto&p=n.kernel;
   tree_replay_check(cudaLaunchKernel(p.func,p.gridDim,p.blockDim,p.kernelParams,p.sharedMemBytes,g.stream));
  }else if(n.type==cudaGraphNodeTypeMemcpy)tree_replay_check(cudaMemcpy3DAsync(&n.copy,g.stream));
 }
}
// One real complete target transaction, identical inputs and original kernels.
// This compares ordinary node submission against Graph, not full-model throughput.
static void tree_replay_proof(TreeReplayState&r){
 auto&e=execution();const auto nodes=tree_replay_eager_plan(r.graph);
 const auto committed=tree_replay_image(true);
 auto reset=[&]{
  tree_replay_check(cudaMemsetAsync(static_cast<uint8_t*>(e.device_task_scratch)+e.tree.state_offset,
   0xff,16ull*e.tree.recurrent*4,g.stream));
  tree_replay_check(cudaStreamSynchronize(g.stream));
 };
 reset();tree_replay_eager_nodes(nodes);tree_replay_check(cudaStreamSynchronize(g.stream));
 if(tree_replay_image(true)!=committed)throw std::runtime_error("tree eager changed committed state");
 const auto reference=tree_replay_image(false);
 reset();tree_replay_check(r.plan.replay(r.graph,r.geometry,g.stream));
 tree_replay_check(cudaStreamSynchronize(g.stream));
 if(tree_replay_image(true)!=committed||tree_replay_image(false)!=reference)
  throw std::runtime_error("tree graph numerical/state mismatch");
 double ms[4]={};const bool graph_arm[4]={false,true,true,false};
 for(unsigned arm=0;arm<4;++arm){
  for(unsigned repeat=0;repeat<2;++repeat){
   reset();const auto start=std::chrono::steady_clock::now();
   if(graph_arm[arm])tree_replay_check(r.plan.replay(r.graph,r.geometry,g.stream));
   else tree_replay_eager_nodes(nodes);
   tree_replay_check(cudaStreamSynchronize(g.stream));
   ms[arm]+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/2;
  }
 }
 const bool exact=tree_replay_image(true)==committed&&tree_replay_image(false)==reference;
 if(!exact)throw std::runtime_error("tree ABBA changed state");
 const double a=(ms[0]+ms[3])/2,b=(ms[1]+ms[2])/2;
 std::fprintf(stderr,"[tree-graph-proof] start=%u nodes=%zu dynamic=%zu exact=1 A1_ms=%.6f B1_ms=%.6f B2_ms=%.6f A2_ms=%.6f A_ms=%.6f B_ms=%.6f improvement=%.6f\n",
  r.geometry.start,nodes.size(),r.plan.dynamic_nodes(),ms[0],ms[1],ms[2],ms[3],a,b,(a-b)/a);
 r.proof_done=true;
}
extern "C" int imparo_cuda_tree_graph_abort(){
 auto&e=execution();
 if(e.tree_capture_active){
  cudaGraph_t graph=nullptr;const auto rc=cudaStreamEndCapture(g.stream,&graph);
  e.tree_capture_active=false;
  if(graph)cudaGraphDestroy(graph);
  if(e.tree_replay)e.tree_replay->clear_graph();
  invalidate_q8_cache();e.kdq.clear();e.vdq.clear();
  return rc==cudaSuccess||rc==cudaErrorStreamCaptureInvalidated?0:CUDA_RC_ERROR;
 }
 return 0;
}
extern "C" int imparo_cuda_tree_graph_prepare(unsigned*action){
 if(!action)return CUDA_RC_INVALID;*action=0;
 if(!imparo_lfm_retained::long_only(tree_replay_flag("IMPARO_LAB_LFM_TREE_GRAPH")))return 0;
 try{
  auto&e=execution();auto*s=imparo_dspark::attached.get();
  if(e.pending_error)return e.pending_error;
  if(!s||s->poisoned||s->parked||!s->capture_enabled||s->target_owner!=active_execution_owner_id
   ||!e.forward_open||!e.forward_active||e.forward_decode||e.graph_capturing||e.prefill_capture_active
   ||e.tree_capture_active||e.tree.nodes!=16||(!batch_invariant_q8_v1_active()&&e.tree.start<4096)||e.graph_leases
   ||g.sm_version!=86||g.kv_type_k!=8||g.kv_type_v!=8||!g.weights_resident
   ||!g.stream||g.tuner_mode||g.tuner_lab||g.nsys_capture_active||e.forward_timing_armed
   ||s->cfg.target_hidden!=2048||s->target_layers.empty()||s->target_layers.size()>=32
   ||e.sizes[25]<uint64_t(e.tree.recurrent)*4||e.sizes[12]<16ull*s->cfg.vocab*4
   ||e.sizes[14]<64||e.sizes[13]<64
   ||(!batch_invariant_q8_v1_active()&&(!tree_replay_flag("IMPARO_LAB_TREE_TAIL_REPAIR")
   ||!tree_replay_flag("IMPARO_LAB_D64_FIXED_PARTITION")))
   ||std::getenv("IMPARO_CUDA_Q8_L2_PERSIST_LAB"))return 0;
  auto*d=find_execution_owner(s->draft_owner);
  if(!d||!d->bufs[imparo_dspark::FEATURE]
   ||d->sizes[imparo_dspark::FEATURE]<16ull*s->target_layers.size()*s->cfg.target_hidden*4)return 0;
  unsigned span=0,layers=0,capacity=0;
  for(unsigned i=0;i<MAX_LAYERS;++i)if(e.kv_k[i]||e.kv_v[i]){
   if(!e.kv_k[i]||!e.kv_v[i]||kv_page_table_requires_mapping(i))return 0;
   const auto rows=e.kv_bytes[i]/544;
   if(rows>UINT32_MAX||rows<uint64_t(e.tree.start)+16)return 0;
   const unsigned this_span=d64_fixed_partition_span(e.tree.start,16,8,unsigned(rows));
   if(layers&&(span!=this_span||capacity!=rows))return 0;
   span=this_span;capacity=unsigned(rows);++layers;
  }
  if(!span||!layers)return 0;
  TreeAttentionPlan p;tree_replay_check(tree_attention_plan(e.tree,span,p));
  if(!p.repair)return 0;
  TreeReplayGeometry geo;geo.batch_invariant=batch_invariant_q8_v1_active();geo.start=e.tree.start;geo.fixed_span=span;geo.chunks=p.chunks;
  geo.begin=p.begin;geo.tail=p.tail;geo.leaf_mask=p.leaf_mask;geo.ordinary_bytes=p.ordinary_bytes;
  for(unsigned mask=p.leaf_mask;mask;mask&=mask-1)++geo.leaves;
  if(!e.tree_replay)e.tree_replay=std::make_shared<TreeReplayState>();
  auto&r=*e.tree_replay;if(r.blocked)return 0;
  if(!r.warmed){r.warmed=true;return 0;}
  // Reserve the same existing workspace for the maximum tail of this physical
  // capacity. No new scratch layout or weight format is introduced.
  const uint64_t extra=geo.batch_invariant
   ?16ull*(9ull*2048*2+2ull*(span+8)*512*2+18ull*(2048*2+32*4))
   :16ull*(2ull*9*2048*2+2ull*span*512*2+9ull*32*4);
  const uint64_t worst=imparo_d64_fa2_port::workspace_bytes(16,32,capacity,true)+extra;
  if(worst>attention_workspace_budget(64)||!ensure_attention_scratch(worst))return 0;
  const uint64_t qbytes=std::max(e.q8_scratch_bytes,e.q8_scratch_next_bytes);
  if(!qbytes||ensure_q8_scratch(qbytes)||!ensure_q8_scratch_next(qbytes))return CUDA_RC_ERROR;
  const auto storage=tree_replay_storage();
  const bool same_bucket=geo.batch_invariant==r.geometry.batch_invariant&&geo.fixed_span==r.geometry.fixed_span&&geo.chunks==r.geometry.chunks
   &&(uint64_t(geo.start)+16+span-1)/span==(uint64_t(r.geometry.start)+16+span-1)/span;
  if(r.graph.exec&&storage==r.storage&&same_bucket){
   tree_replay_check(r.plan.replay(r.graph,geo,g.stream));
   tree_replay_postconditions(r);++r.replays;
   if(r.replays==1)std::fprintf(stderr,"[tree-graph] replay=1 start=%u dynamic=%zu\n",geo.start,r.plan.dynamic_nodes());
   *action=2;return 0;
  }
  r.clear_graph();r.geometry=geo;r.storage=storage;r.attention_layers=layers;
  r.q8_in=e.q8_scratch;r.q8_in_bytes=e.q8_scratch_bytes;
  r.q8_next_in=e.q8_scratch_next;r.q8_next_in_bytes=e.q8_scratch_next_bytes;
  invalidate_q8_cache();e.kdq.clear();e.vdq.clear();e.attention_q_src=UINT32_MAX;
  e.tree_capture_allocation_blocked=false;
  tree_replay_check(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal));
  e.tree_capture_active=true;*action=1;return 0;
 }catch(const std::exception&ex){
  std::fprintf(stderr,"[tree-graph] prepare: %s\n",ex.what());return CUDA_RC_ERROR;
 }
}
extern "C" int imparo_cuda_tree_graph_finish(){
 auto&e=execution();if(!e.tree_capture_active)return 0;
 auto&r=*e.tree_replay;
 cudaGraph_t graph=nullptr;const auto ended=cudaStreamEndCapture(g.stream,&graph);
 e.tree_capture_active=false;
 if(ended!=cudaSuccess||!graph||e.pending_error||e.tree_capture_allocation_blocked){
  if(graph)cudaGraphDestroy(graph);r.clear_graph();
  return e.pending_error?e.pending_error:CUDA_RC_ERROR;
 }
 r.graph.graph=graph;
 try{
  r.q8_out=e.q8_scratch;r.q8_out_bytes=e.q8_scratch_bytes;
  r.q8_next_out=e.q8_scratch_next;r.q8_next_out_bytes=e.q8_scratch_next_bytes;
  if(tree_replay_storage()!=r.storage)throw std::runtime_error("capture storage changed");
  tree_replay_check(cudaGraphInstantiate(&r.graph.exec,graph,nullptr,nullptr,0));
  if(!tree_replay_feature_nodes(r.graph)
   ||!r.plan.configure(r.graph,r.geometry,r.attention_layers,tree_replay_static_kernels()))
   throw std::runtime_error("tree dynamic coverage or feature contract");
  ++r.captures;
  if(tree_replay_flag("IMPARO_LAB_LFM_TREE_GRAPH_PROOF")&&!r.proof_done)tree_replay_proof(r);
  else tree_replay_check(r.plan.replay(r.graph,r.geometry,g.stream));
  tree_replay_postconditions(r);
  std::fprintf(stderr,"[tree-graph] capture=%llu start=%u dynamic=%zu attention=%u\n",
   (unsigned long long)r.captures,r.geometry.start,r.plan.dynamic_nodes(),r.attention_layers);
  return 0;
 }catch(const std::exception&ex){
  std::fprintf(stderr,"[tree-graph] finish: %s\n",ex.what());r.clear_graph();return CUDA_RC_ERROR;
 }
}
