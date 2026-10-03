#include "kv_row_move.cuh"

// Publish accepted KV rows with the model's actual owner/codec/ring geometry.
// This is a static laboratory interface; it does not admit a new tree forward.
static int tree_commit_kv_rows_impl(
        const imparo_cuda_kv::RowMoveGeometry* geometry, uint32_t layers,
        const uint32_t* from, const uint32_t* to, uint32_t count,
        bool e4b_transaction) {
    using namespace imparo_cuda_kv;
    if (!geometry || !layers || layers > MAX_LAYERS || count > kMoveRows
        || (count && (!from || !to)) || !execution_boundary_closed()
        || execution().graph_capturing || execution().tree_capture_active
        || (e4b_transaction ? (!execution().tree.nodes || !execution().tree.e4b)
                            : execution().tree.nodes != 0))
        return CUDA_RC_INVALID;
    if (execution().pending_error) return execution().pending_error;
    std::vector<RowMovePlan> plans;
    try { plans.resize(layers); }
    catch (const std::bad_alloc&) { return CUDA_RC_OOM; }
    bool seen[MAX_LAYERS]{};
    uint64_t scratch_bytes = 0;
    // Validate all owners and all addresses before reserving memory or writing KV.
    for (uint32_t i = 0; i < layers; ++i) {
        const auto& spec = geometry[i];
        if (spec.layer >= MAX_LAYERS || seen[spec.layer]
            || !execution().kv_k[spec.layer] || !execution().kv_v[spec.layer])
            return CUDA_RC_INVALID;
        seen[spec.layer] = true;
        const PageTableLayer* table = nullptr;
        if (!spec.ring && spec.layer < execution().kv_page_tables.layers
            && execution().kv_page_tables.layer[spec.layer].capacity)
            table = &execution().kv_page_tables.layer[spec.layer];
        if (!plan_row_moves(spec, execution().kv_bytes[spec.layer], table,
                from, to, count, &plans[i])) return CUDA_RC_INVALID;
        const uint64_t stride = std::max(spec.k_stride, spec.v_stride);
        if (stride > 64 * 1024 * 1024ULL
            || (plans[i].count && stride > (64 * 1024 * 1024ULL) / plans[i].count))
            return CUDA_RC_INVALID;
        scratch_bytes = std::max(scratch_bytes, stride * plans[i].count);
    }
    if (!scratch_bytes) return 0;
    // Attention workspace is dead after verification's closed output boundary.
    // Reusing it preserves the live tree topology/recurrent task-scratch region.
    if (!ensure_attention_scratch(scratch_bytes)) return CUDA_RC_ERROR;
    auto* scratch = static_cast<uint8_t*>(execution().attention_scratch);
    for (const auto& plan : plans) {
        if (!plan.count) continue;
        for (uint32_t is_v = 0; is_v < 2; ++is_v) {
            auto* cache = static_cast<uint8_t*>(is_v
                ? execution().kv_v[plan.geometry.layer]
                : execution().kv_k[plan.geometry.layer]);
            const uint64_t stride = is_v ? plan.geometry.v_stride : plan.geometry.k_stride;
            const auto error = copy_row_moves(plan, stride, cache, scratch, g.stream);
            // Invalidate even on a partial driver failure: the owner is poisoned.
            invalidate_kv_dequant(plan.geometry.layer, is_v);
            if (error != cudaSuccess) {
                set_pending(CUDA_RC_ERROR, "tree KV path publication");
                return CUDA_RC_ERROR;
            }
        }
    }
    if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
        set_pending(CUDA_RC_ERROR, "tree KV path publication completion");
        return CUDA_RC_ERROR;
    }
    return 0;
}

extern "C" int imparo_cuda_tree_commit_kv_rows(
        const imparo_cuda_kv::RowMoveGeometry* geometry, uint32_t layers,
        const uint32_t* from, const uint32_t* to, uint32_t count) {
    return tree_commit_kv_rows_impl(geometry,layers,from,to,count,false);
}

// The E4B route is deliberately a separate admission case on the SAME owner.
// Existing LFM2 entry points below retain their Q8/recurrent/numerical guards.
static int prepare_e4b_tree_context(uint32_t start,const int32_t*parents,
        uint32_t nodes,const uint32_t*layout,uint32_t words,
        const E4bTreeGeometry*geometry,uint32_t layers,TreeContext&t) {
    const char*flag=std::getenv("IMPARO_DSPARK_TREE");
    const char*value_tiles=std::getenv("IMPARO_CUDA_ATTN_VALUE_TILES");
    if(!flag || std::strcmp(flag,"4")!=0 || !parents || !layout || !geometry
        || nodes!=4 || words!=48 || !layers || layers>MAX_LAYERS
        || start>UINT_MAX-4 || !execution_boundary_closed()
        || execution().tree.nodes || execution().graph_capturing
        || execution().prefill_capture_active || execution().tree_capture_active
        || execution().cobatch_rows || !g.stream
        || !e4b_retained_decode_policy_enabled() || g.e4b_retained_domain!=1
        || !g.weights || !g.weights_resident
        || g.e4b_retained_weights_identity!=g.weights
        || g.kv_type_k!=2 || g.kv_type_v!=2 || g.tuner_mode || g.tuner_lab
        || !tuner_knob(36) || !tuner_knob(30) || tuner_knob(17)>1
        || (value_tiles && std::strcmp(value_tiles,"1")!=0)
        || std::getenv("IMPARO_CUDA_NO_ATTN_D256_VEC")
        || std::getenv("IMPARO_CUDA_NO_ATTN_D256_GQA4")
        || std::getenv("IMPARO_CUDA_ATTN_D256_GQA4_COMPARE")
        || std::getenv("IMPARO_CUDA_NO_Q4_ATTN_DIRECT"))
        return CUDA_RC_INVALID;
    if(execution().pending_error)return execution().pending_error;
    const char*branch=std::getenv("IMPARO_LAB_E4B_TREE_DELAYED");
    const bool delayed=branch&&std::strcmp(branch,"1")==0;
    const int32_t expected_parent[4]={-1,0,1,delayed?1:0};
    const uint32_t expected_depth[4]={0,1,2,delayed?2u:1u};
    const uint32_t expected_mask[4]={1,3,7,delayed?11u:9u};
    for(uint32_t i=0;i<4;++i){
        const uint32_t*row=layout+12*i;
        if(parents[i]!=expected_parent[i] || row[0]!=start+expected_depth[i]
            || row[1]!=expected_depth[i] || row[2]!=expected_mask[i] || row[3])
            return CUDA_RC_INVALID;
        uint32_t ancestor=i;
        for(uint32_t back=0;back<8;++back){
            ancestor=ancestor ? uint32_t(expected_parent[ancestor]) : 0;
            if(row[4+back]!=ancestor)return CUDA_RC_INVALID;
        }
        t.parents[i]=expected_parent[i];t.depths[i]=expected_depth[i];
    }
    bool seen[MAX_LAYERS]{};
    bool full=false,windowed=false;
    for(uint32_t i=0;i<layers;++i){
        const auto&spec=geometry[i];
        const bool sliding=spec.head_dim==256 && spec.window==512 && spec.ring_slots==1024;
        const bool global=spec.head_dim==512 && !spec.window && !spec.ring_slots;
        const uint64_t stride=uint64_t(2*spec.head_dim/32)*18;
        if(spec.layer>=MAX_LAYERS || seen[spec.layer] || (!sliding && !global)
            || spec.k_stride!=stride || spec.v_stride!=stride
            || !execution().kv_k[spec.layer] || !execution().kv_v[spec.layer]
            || kv_page_table_requires_mapping(spec.layer))return CUDA_RC_INVALID;
        // Logical context capacity is not committed storage. Full KV grows
        // through the common allocator (initially 576 rows); only the four
        // physical writes need backing. The window's ring is fully allocated.
        const uint64_t required_rows=sliding ? spec.ring_slots : uint64_t(start)+nodes;
        if(execution().kv_bytes[spec.layer]<required_rows*stride)return CUDA_RC_INVALID;
        seen[spec.layer]=true;full|=global;windowed|=sliding;
    }
    if(!full || !windowed)return CUDA_RC_INVALID;
    // All allocated owners must be declared. SharedWith layers allocate no KV
    // and therefore are neither copied twice nor mistaken for distinct owners.
    for(uint32_t layer=0;layer<MAX_LAYERS;++layer)
        if((execution().kv_k[layer] || execution().kv_v[layer]) && !seen[layer])
            return CUDA_RC_INVALID;
    try {
        t.row_layout.assign(layout,layout+words);
        t.e4b_geometry.assign(geometry,geometry+layers);
    } catch(const std::bad_alloc&) { return CUDA_RC_OOM; }
    t.nodes=nodes;t.start=start;t.recurrent=0;t.e4b=true;
    t.mask_offset=t.state_offset;t.commit_offset=t.state_offset;
    return 0;
}

static int reserve_e4b_tree_layout(){
    constexpr uint32_t row_layout_buf=28;
    constexpr uint64_t bytes=48*sizeof(uint32_t);
    if(!ensure_device_task_scratch(256))return CUDA_RC_ERROR;
    if(!execution().bufs[row_layout_buf] || execution().sizes[row_layout_buf]<bytes){
        const int rc=imparo_cuda_alloc(row_layout_buf,bytes);
        if(rc)return rc;
    }
    return 0;
}

extern "C" int imparo_cuda_tree_prepare_e4b(uint32_t start,const int32_t*parents,
        uint32_t nodes,const uint32_t*layout,uint32_t words,
        const E4bTreeGeometry*geometry,uint32_t layers,uint32_t*admitted){
    if(!admitted)return CUDA_RC_INVALID;
    *admitted=0;
    TreeContext next;
    const int rc=prepare_e4b_tree_context(start,parents,nodes,layout,words,geometry,layers,next);
    if(rc)return rc;
    // Outside the one qualified absolute-position bucket, keep ordinary MTP.
    if(start<512 || start>765)return 0;
    const int reserve=reserve_e4b_tree_layout();
    if(reserve)return reserve;
    *admitted=1;
    return 0;
}

extern "C" int imparo_cuda_tree_begin_e4b(uint32_t start,const int32_t*parents,
        uint32_t nodes,const uint32_t*layout,uint32_t words,
        const E4bTreeGeometry*geometry,uint32_t layers){
    if(start<512 || start>765)return CUDA_RC_INVALID;
    TreeContext next;
    const int rc=prepare_e4b_tree_context(start,parents,nodes,layout,words,geometry,layers,next);
    if(rc)return rc;
    const int reserve=reserve_e4b_tree_layout();
    if(reserve)return reserve;
    try { execution().u32_shadow[28]=next.row_layout; }
    catch(const std::bad_alloc&) { return CUDA_RC_OOM; }
    execution().tree=std::move(next);
    auto&t=execution().tree;
    auto*base=static_cast<uint8_t*>(execution().device_task_scratch);
    // The public layout and the compatibility parent/depth views have one
    // lifetime, owned by this transaction. No host source outlives its storage.
    auto error=cudaMemcpyAsync(base,t.parents,sizeof(t.parents),cudaMemcpyHostToDevice,g.stream);
    if(error==cudaSuccess)error=cudaMemcpyAsync(base+64,t.depths,sizeof(t.depths),cudaMemcpyHostToDevice,g.stream);
    if(error==cudaSuccess)error=cudaMemcpyAsync(execution().bufs[28],t.row_layout.data(),
        t.row_layout.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,g.stream);
    // Always drain submitted uploads before error cleanup releases host data.
    const auto drained=cudaStreamSynchronize(g.stream);
    if(error!=cudaSuccess || drained!=cudaSuccess){
        t.nodes=0;t.e4b=false;
        set_pending(CUDA_RC_ERROR,"E4B tree layout upload");
        return CUDA_RC_ERROR;
    }
    mark_buf_written(28);
    return 0;
}

extern "C" int imparo_cuda_tree_commit_e4b(const int32_t*path,uint32_t count){
    auto&t=execution().tree;
    if(!t.e4b || t.nodes!=4 || !path || !count || count>3
        || !execution_boundary_closed() || t.recurrent
        || !e4b_retained_decode_policy_enabled() || g.e4b_retained_domain!=1
        || g.e4b_retained_weights_identity!=g.weights
        || g.kv_type_k!=2 || g.kv_type_v!=2)return CUDA_RC_INVALID;
    uint32_t from[3]{},to[3]{};
    for(uint32_t i=0;i<count;++i){
        if(path[i]<0 || uint32_t(path[i])>=t.nodes
            || (i==0 ? path[i]!=0 : t.parents[path[i]]!=path[i-1]))return CUDA_RC_INVALID;
        from[i]=t.start+uint32_t(path[i]);to[i]=t.start+i;
    }
    std::vector<imparo_cuda_kv::RowMoveGeometry> geometry;
    try {
        geometry.reserve(t.e4b_geometry.size());
        for(const auto&spec:t.e4b_geometry)
            geometry.push_back({spec.layer,spec.ring_slots ? spec.ring_slots-1 : 0,
                spec.k_stride,spec.v_stride});
    } catch(const std::bad_alloc&) { return CUDA_RC_OOM; }
    // Keep the live transaction installed while the checked internal helper
    // publishes its stored geometry. The public row-move API rejects live trees.
    const int rc=tree_commit_kv_rows_impl(geometry.data(),uint32_t(geometry.size()),
        from,to,count,true);
    if(rc)return rc;
    t.nodes=0;t.e4b=false;
    return 0;
}

// Laboratory admission on the caller's existing serialized execution owner.
static int prepare_tree_context(unsigned start,const int*parents,unsigned nodes,unsigned recurrent,TreeContext&t,uint64_t&maskbytes){
 if(!parents||nodes<2||nodes>16||start>UINT_MAX-16||!recurrent||!execution_boundary_closed()||execution().tree.nodes||g.sm_version!=86||g.kv_type_k!=8||g.kv_type_v!=8)return CUDA_RC_INVALID;
 // Variable rows reuse only the admitted batch-invariant Q8 arithmetic.
 if(nodes!=16&&!batch_invariant_q8_v1_active())return CUDA_RC_INVALID;
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
  if(!execution().kv_k[l]||!execution().kv_v[l])return CUDA_RC_INVALID;
  const uint64_t capacity=execution().kv_bytes[l]/(uint64_t(512/32)*34);
  if(capacity>UINT_MAX)return CUDA_RC_INVALID;capacity_min=std::min(capacity_min,capacity);++layers;
  if(capacity<uint64_t(start)+nodes){tree_admission_witness(false,"physical_capacity",start,uint64_t(start)+nodes,capacity,capacity_min);return 0;}
  const unsigned logical_capacity=d64_attention_context_capacity(unsigned(capacity));
  if(logical_capacity<uint64_t(start)+nodes){tree_admission_witness(false,"logical_capacity",start,uint64_t(start)+nodes,logical_capacity,capacity_min);return 0;}
  if(kv_page_table_requires_mapping(l)) {
   if(!batch_invariant_q8_v1_active()){tree_admission_witness(false,"paged_provider",start,0,0,capacity_min);return 0;}
   uint64_t physical=0;
   if(l>=execution().kv_page_tables.layers
      ||!imparo_cuda_kv::mapped_prefix_rows(execution().kv_page_tables.layer[l],start+nodes,&physical)
      ||physical>capacity)return CUDA_RC_INVALID;
  }
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
 if(!t.nodes||t.e4b||!path||!count||count>t.nodes||!execution_boundary_closed()||recur_buf>=B_COUNT||execution().sizes[recur_buf]<uint64_t(t.recurrent)*4||kv_width!=512)return CUDA_RC_INVALID;
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
extern "C" int imparo_cuda_tree_end(){if(!execution_boundary_closed())return CUDA_RC_INVALID;execution().tree.nodes=0;execution().tree.e4b=false;return 0;}
