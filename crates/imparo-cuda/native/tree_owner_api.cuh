// Laboratory admission on the caller's existing serialized execution owner.
static int prepare_tree_context(unsigned start,const int*parents,unsigned nodes,unsigned recurrent,TreeContext&t,uint64_t&maskbytes){
 if(!parents||nodes!=16||start>UINT_MAX-nodes||!recurrent||!execution_boundary_closed()||execution().tree.nodes||g.sm_version!=86||g.kv_type_k!=8||g.kv_type_v!=8)return CUDA_RC_INVALID;
 if(execution().pending_error)return execution().pending_error;
 t.start=start;t.nodes=nodes;t.recurrent=recurrent;
 for(unsigned i=0;i<nodes;++i){if((i==0&&parents[i]!=-1)||(i&&(parents[i]<0||unsigned(parents[i])>=i)))return CUDA_RC_INVALID;t.parents[i]=parents[i];t.depths[i]=i?t.depths[parents[i]]+1:0;if(t.depths[i]>8)return CUDA_RC_INVALID;}
 t.mask_offset=(t.state_offset+uint64_t(nodes)*recurrent*4+255)&~uint64_t(255);
 maskbytes=(uint64_t(nodes)*(start+nodes)+7)/8;t.commit_offset=(t.mask_offset+maskbytes+255)&~uint64_t(255);
 return 0;
}
static void tree_admission_witness(bool admitted,const char*reason,unsigned start,uint64_t needed,uint64_t budget,uint64_t capacity){
 static unsigned seen=0;const unsigned bit=admitted?1u:2u;if(seen&bit)return;seen|=bit;
 std::fprintf(stderr,"[tree-admission] admitted=%u reason=%s start=%u required=%llu budget=%llu physical_min=%llu target_writes=0\n",unsigned(admitted),reason,start,(unsigned long long)needed,(unsigned long long)budget,(unsigned long long)capacity);
}
// A false result performs no target-state writes and publishes no tree transaction.
// Allocation/driver failures are errors, never silently converted into a chain.
extern "C" int imparo_cuda_tree_prepare(unsigned start,const int*parents,unsigned nodes,unsigned recurrent,unsigned*admitted){
 if(!admitted)return CUDA_RC_INVALID;*admitted=0;
 TreeContext t;uint64_t maskbytes=0;int rc=prepare_tree_context(start,parents,nodes,recurrent,t,maskbytes);if(rc)return rc;
 if(!tuner_knob(37)||std::getenv("IMPARO_CUDA_NO_ATTN_D64_MMA_PREFILL"))return CUDA_RC_INVALID;
 // A selected ordinary shared-KV kernel does not implement the tree mask.
 const unsigned shared_min=tuner_knob(46);
 if(shared_min&&nodes>=shared_min&&std::getenv("IMPARO_CUDA_NO_ATTN_D64_SHARED_KV")==nullptr){tree_admission_witness(false,"attention_route",start,0,0,0);return 0;}
 const uint64_t budget=attention_workspace_budget(64);uint64_t needed=0,capacity_min=UINT_MAX;unsigned layers=0;
 for(unsigned l=0;l<MAX_LAYERS;++l){
  if(!execution().kv_k[l]&&!execution().kv_v[l])continue;
  if(!execution().kv_k[l]||!execution().kv_v[l]||kv_page_table_requires_mapping(l))return CUDA_RC_INVALID;
  const uint64_t capacity=execution().kv_bytes[l]/(uint64_t(512/32)*34);
  if(capacity>UINT_MAX)return CUDA_RC_INVALID;capacity_min=std::min(capacity_min,capacity);++layers;
  if(capacity<uint64_t(start)+nodes){tree_admission_witness(false,"physical_capacity",start,uint64_t(start)+nodes,capacity,capacity_min);return 0;}
  TreeAttentionPlan plan;const auto e=tree_attention_plan(t,d64_fixed_partition_span(start,nodes,8,unsigned(capacity)),plan);
  if(e!=cudaSuccess)return CUDA_RC_ERROR;
  needed=std::max(needed,plan.bytes);
 }
 if(!layers)return CUDA_RC_INVALID;
 if(t.commit_offset>64*1024*1024){tree_admission_witness(false,"task_budget",start,t.commit_offset,64*1024*1024,capacity_min);return 0;}
 if(needed>budget){tree_admission_witness(false,"attention_budget",start,needed,budget,capacity_min);return 0;}
 // Reserve only after every pure capacity/budget check passed. Existing bool
 // allocation helpers cannot identify all driver failures: conservatively stop.
 if(!ensure_attention_scratch(needed)||!ensure_device_task_scratch(t.commit_offset+2*1024*1024))return CUDA_RC_ERROR;
 *admitted=1;tree_admission_witness(true,"ready",start,needed,budget,capacity_min);return 0;
}
extern "C" int imparo_cuda_tree_begin(unsigned start,const int*parents,unsigned nodes,unsigned recurrent){
 TreeContext t;uint64_t maskbytes=0;int rc=prepare_tree_context(start,parents,nodes,recurrent,t,maskbytes);if(rc)return rc;
 // Existing task scratch owns topology, all recurrent node states, mask and gather scratch.
 if(t.commit_offset>64*1024*1024||!ensure_device_task_scratch(t.commit_offset+2*1024*1024))return CUDA_RC_ERROR;
 t.mask.assign(size_t(maskbytes),0);for(unsigned i=0;i<nodes;++i){for(unsigned j=0;j<start;++j){uint64_t bit=uint64_t(i)*(start+nodes)+j;t.mask[bit/8]|=1u<<(bit%8);}for(int j=int(i);j>=0;j=t.parents[j]){uint64_t bit=uint64_t(i)*(start+nodes)+start+j;t.mask[bit/8]|=1u<<(bit%8);}}
 execution().tree=std::move(t);auto&v=execution().tree;auto*base=static_cast<uint8_t*>(execution().device_task_scratch);
 auto e=cudaMemcpyAsync(base,v.parents,64,cudaMemcpyHostToDevice,g.stream);if(e==cudaSuccess)e=cudaMemcpyAsync(base+64,v.depths,64,cudaMemcpyHostToDevice,g.stream);if(e==cudaSuccess)e=cudaMemcpyAsync(base+v.mask_offset,v.mask.data(),v.mask.size(),cudaMemcpyHostToDevice,g.stream);
 if(e==cudaSuccess)e=cudaMemsetAsync(base+v.state_offset,0xff,uint64_t(nodes)*recurrent*4,g.stream);
 if(e==cudaSuccess)e=cudaStreamSynchronize(g.stream);if(e!=cudaSuccess){execution().tree.nodes=0;return CUDA_RC_ERROR;}return 0;
}
extern "C" int imparo_cuda_tree_commit(const int*path,unsigned count,unsigned kv_width,unsigned recur_buf){
 auto&t=execution().tree;
 if(!t.nodes||!path||!count||count>t.nodes||!execution_boundary_closed()||recur_buf>=B_COUNT||execution().sizes[recur_buf]<uint64_t(t.recurrent)*4||kv_width!=512)return CUDA_RC_INVALID;
 for(unsigned i=0;i<count;++i)if(path[i]<0||unsigned(path[i])>=t.nodes||(i==0?path[i]!=0:t.parents[path[i]]!=path[i-1]))return CUDA_RC_INVALID;
 auto*base=static_cast<uint8_t*>(execution().device_task_scratch);auto*scratch=base+t.commit_offset;auto*ids=reinterpret_cast<int*>(base+128);const unsigned rowbytes=kv_width/32*34;
 auto e=cudaMemcpyAsync(ids,path,count*4,cudaMemcpyHostToDevice,g.stream);if(e!=cudaSuccess)return CUDA_RC_ERROR;
 for(unsigned l=0;l<MAX_LAYERS;++l){if(!execution().kv_k[l]&&!execution().kv_v[l])continue;if(!execution().kv_k[l]||!execution().kv_v[l])return CUDA_RC_INVALID;const auto*pages=kv_device_page_table(l);
  // Both gathers precede any overwrite within each buffer, so rejected slots
  // cannot overwrite a source needed by a later accepted node.
  for(unsigned v=0;v<2;++v){auto*cache=static_cast<uint8_t*>(v?execution().kv_v[l]:execution().kv_k[l]);tree_commit_gather<<<(count*rowbytes+255)/256,256,0,g.stream>>>(cache,scratch,ids,t.start,rowbytes,count,pages);tree_commit_scatter<<<(count*rowbytes+255)/256,256,0,g.stream>>>(scratch,cache,t.start,rowbytes,count,pages);invalidate_kv_dequant(l,v);}
 }
 e=cudaMemcpyAsync(execution().bufs[recur_buf],base+t.state_offset+uint64_t(path[count-1])*t.recurrent*4,uint64_t(t.recurrent)*4,cudaMemcpyDeviceToDevice,g.stream);mark_buf_written(recur_buf);
 if(e==cudaSuccess)e=cudaStreamSynchronize(g.stream);if(e!=cudaSuccess||cudaPeekAtLastError()!=cudaSuccess)return CUDA_RC_ERROR;
 t.nodes=0;return 0;
}
extern "C" int imparo_cuda_tree_end(){if(!execution_boundary_closed())return CUDA_RC_INVALID;execution().tree.nodes=0;return 0;}
