#pragma once
// Transport only: immutable mapped weights, actual per-owner access order, and
// two slots inside the existing cache budget. No model names or numerical code.
// Policy 2 reuses the same packed pipeline in Prefill and Decode. Load-time
// operations stay outside the forward and retain the ordinary scratch path.
static bool weight_transfer_active() {
    return g.weight_transfer_policy && (execution().forward_decode
        || (g.weight_transfer_policy == 2 && execution().forward_open));
}
static uint64_t weight_transfer_slice_limit() {
    return weight_transfer_active()
        ? (g.weight_cache_limit / 2) & ~uint64_t(255) : g.weight_cache_limit;
}
static bool release_weight_transfer(WeightTransferState &s) {
    if (s.stream && cudaStreamSynchronize(s.stream)!=cudaSuccess) return false;
    for(int i=0;i<2;++i) {
        if(s.ready[i] && cudaEventDestroy(s.ready[i])!=cudaSuccess)return false;
        s.ready[i]=nullptr;
        if(s.consumed[i] && cudaEventDestroy(s.consumed[i])!=cudaSuccess)return false;
        s.consumed[i]=nullptr;
    }
    if(s.stream && cudaStreamDestroy(s.stream)!=cudaSuccess)return false;
    s.stream=nullptr; return true;
}
// All unstructured users of weight_cache call ensure_weight_cache. Drain pending
// copies onto the compute stream before their writes; never return a stale hit.
static int weight_transfer_barrier(bool invalidate) {
    auto &s=execution().weight_transfer;
    if(!s.started)return 0;
    for(int i=0;i<2;++i)if(s.queued[i]!=SIZE_MAX) {
        if(cudaStreamWaitEvent(g.stream,s.ready[i],0)!=cudaSuccess)return CUDA_RC_ERROR;
    }
    s.started=false;s.previous_slot=-1;
    s.previous_group=SIZE_MAX;s.previous_group_last=false;
    s.queued[0]=s.queued[1]=SIZE_MAX;
    if(invalidate)s.blocked=true;
    return 0;
}
static int weight_transfer_pin_sources() {
    if(g.weight_pins_ready)return 0;
    if(!g.weights_host || !g.weights_len)return CUDA_RC_INVALID;
    int device=0,readonly=0;
    if(cudaGetDevice(&device)!=cudaSuccess ||
       cudaDeviceGetAttribute(&readonly,cudaDevAttrHostRegisterReadOnlySupported,device)!=cudaSuccess)
        return CUDA_RC_ERROR;
    if(!readonly) {
        std::fprintf(stderr,"[imparo] weight_transfer: read-only host registration unsupported\n");
        return CUDA_RC_INVALID;
    }
    std::vector<WeightTransferSlice> spans;
    const auto add=[&](uint64_t off,uint64_t bytes) {
        const uint64_t p=uint64_t(reinterpret_cast<uintptr_t>(g.weights_host))+off;
        const uint64_t lo=p&~uint64_t(4095),hi=(p+bytes+4095)&~uint64_t(4095);
        if(!spans.empty() && lo<=spans.back().offset+spans.back().bytes)
            spans.back().bytes=std::max(hi,spans.back().offset+spans.back().bytes)-spans.back().offset;
        else spans.push_back({lo,hi-lo});
    };
    if(g.weights_resident)for(const auto &r:g.streamed_weights)add(r.offset,r.bytes);
    else add(0,g.weights_len);
    const auto start=std::chrono::steady_clock::now();uint64_t total=0;
    for(const auto &r:spans) {
        auto err=cudaHostRegister(reinterpret_cast<void*>(uintptr_t(r.offset)),size_t(r.bytes),cudaHostRegisterReadOnly);
        if(err!=cudaSuccess) {
            std::fprintf(stderr,"[imparo] weight_transfer registration failed: %s\n",cudaGetErrorString(err));
            for(const auto &p:g.weight_pins)cudaHostUnregister(reinterpret_cast<void*>(uintptr_t(p.offset)));
            g.weight_pins.clear();return CUDA_RC_ERROR;
        }
        g.weight_pins.push_back(r);total+=r.bytes;
    }
    g.weight_pins_ready=true;
    std::fprintf(stderr,"[imparo] weight_transfer registered bytes=%llu ranges=%zu ms=%.3f cache_budget=%llu\n",
        (unsigned long long)total,spans.size(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),
        (unsigned long long)g.weight_cache_limit);
    return 0;
}
static void weight_transfer_begin(bool decode) {
    auto &s=execution().weight_transfer;
    s.observed.clear();s.cursor=0;s.blocked=false;s.previous_slot=-1;
    s.started=false;s.queued[0]=s.queued[1]=SIZE_MAX;
    s.previous_group=SIZE_MAX;s.previous_group_last=false;
    // Prefill chunks can replay the observed order. Phase changes must relearn:
    // the last-row head and decoder may have a different slice sequence.
    if(s.plan_decode!=decode || (!decode && g.weight_transfer_policy!=2)) {
        s.plan.clear();s.packed.clear();s.reported=false;
    }
    s.plan_decode=decode;
}
static int weight_transfer_queue(size_t index) {
    auto &s=execution().weight_transfer;
    if(index>=s.plan.size())return 0;
    const int slot=int(index%2);if(s.queued[slot]==index)return 0;
    const auto &r=s.plan[index];const uint64_t capacity=weight_transfer_slice_limit();
    if(!r.bytes||r.bytes>capacity||r.offset>g.weights_len||r.bytes>g.weights_len-r.offset)return CUDA_RC_INVALID;
    auto *dst=static_cast<uint8_t*>(execution().weight_cache)+slot*capacity;
    if(cudaStreamWaitEvent(s.stream,s.consumed[slot],0)!=cudaSuccess ||
       cudaMemcpyAsync(dst,g.weights_host+r.offset,size_t(r.bytes),cudaMemcpyHostToDevice,s.stream)!=cudaSuccess ||
       cudaEventRecord(s.ready[slot],s.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    s.queued[slot]=index;return 0;
}
static int weight_transfer_start() {
    auto &s=execution().weight_transfer;
    const uint64_t capacity=weight_transfer_slice_limit();
    int rc=0;
    if(!s.started) {
        rc=ensure_weight_cache(2*capacity);if(rc)return rc;
        if(!s.stream && cudaStreamCreateWithFlags(&s.stream,cudaStreamNonBlocking)!=cudaSuccess)return CUDA_RC_ERROR;
        for(int i=0;i<2;++i) {
            if(!s.ready[i]&&cudaEventCreateWithFlags(&s.ready[i],cudaEventDisableTiming)!=cudaSuccess)return CUDA_RC_ERROR;
            if(!s.consumed[i]&&cudaEventCreateWithFlags(&s.consumed[i],cudaEventDisableTiming)!=cudaSuccess)return CUDA_RC_ERROR;
            if(cudaEventRecord(s.consumed[i],g.stream)!=cudaSuccess)return CUDA_RC_ERROR;
        }
        s.started=true;
    }
    return 0;
}
static int weight_transfer_queue_group(size_t group) {
    auto &s=execution().weight_transfer;
    if(group>=s.packed.groups.size())return 0;
    const int slot=int(group%2);if(s.queued[slot]==group)return 0;
    const uint64_t capacity=weight_transfer_slice_limit();
    if(cudaStreamWaitEvent(s.stream,s.consumed[slot],0)!=cudaSuccess)return CUDA_RC_ERROR;
    const auto &pack=s.packed.groups[group];
    for(size_t i=pack.begin;i<pack.end;++i) {
        const auto &r=s.plan[i]; const auto at=s.packed.offset[i];
        if(!r.bytes || at>capacity || r.bytes>capacity-at || r.offset>g.weights_len || r.bytes>g.weights_len-r.offset)return CUDA_RC_INVALID;
        auto *dst=static_cast<uint8_t*>(execution().weight_cache)+slot*capacity+at;
        if(cudaMemcpyAsync(dst,g.weights_host+r.offset,size_t(r.bytes),cudaMemcpyHostToDevice,s.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    }
    if(cudaEventRecord(s.ready[slot],s.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    s.queued[slot]=group;return 0;
}
static int weight_transfer_group_advance() {
    auto &s=execution().weight_transfer;
    if(!s.started || !s.previous_group_last)return 0;
    // This is called on the NEXT weight access, after the last user's kernels.
    // A slot is never overwritten merely because its last pointer was returned.
    const size_t group=s.previous_group;
    if(cudaEventRecord(s.consumed[group%2],g.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    s.previous_group_last=false;
    return weight_transfer_queue_group(group+2);
}
static int weight_transfer_group_touch() {
    auto &s=execution().weight_transfer;
    if(g.weight_transfer_policy!=2 || !weight_transfer_active() || s.blocked || s.plan.empty())return 0;
    if(execution().graph_capturing || execution().tree_capture_active)return CUDA_RC_INVALID;
    if(s.packed.group.size()!=s.plan.size())return CUDA_RC_INVALID;
    int rc=weight_transfer_group_advance();if(rc)return rc;
    if(s.started)return 0;
    rc=weight_transfer_pin_sources();if(rc)return rc;
    rc=weight_transfer_start();if(rc)return rc;
    rc=weight_transfer_queue_group(0);if(rc)return rc;
    return weight_transfer_queue_group(1);
}
static int weight_transfer_slice(uint64_t off,uint64_t bytes,const uint8_t **out) {
    auto &s=execution().weight_transfer;
    if(execution().graph_capturing||execution().tree_capture_active)return CUDA_RC_INVALID;
    g.weight_transfer_frozen=true;
    int rc=weight_transfer_pin_sources();if(rc)return rc;
    const uint64_t capacity=weight_transfer_slice_limit();
    const size_t index=s.cursor++;
    if(bytes>capacity||!bytes||index>=8192)s.blocked=true;
    if(!s.blocked)s.observed.push_back({off,bytes});
    bool match=!s.blocked&&index<s.plan.size()&&s.plan[index].offset==off&&s.plan[index].bytes==bytes;
    if(!match) {
        rc=weight_transfer_barrier();if(rc)return rc;
        // Empty plan means the first real forward learns the order. Any mismatch
        // discards replay for this forward; its new valid order can be learned next.
        if(!s.plan.empty())s.blocked=true;
        rc=ensure_weight_cache(bytes);if(rc)return rc;
        if(cudaMemcpyAsync(execution().weight_cache,g.weights_host+off,size_t(bytes),cudaMemcpyHostToDevice,g.stream)!=cudaSuccess)return CUDA_RC_ERROR;
        *out=static_cast<const uint8_t*>(execution().weight_cache);return 0;
    }
    if(g.weight_transfer_policy==2) {
        rc=weight_transfer_group_touch();if(rc)return rc;
        const size_t group=s.packed.group[index];const int slot=int(group%2);
        rc=weight_transfer_queue_group(group);if(rc)return rc;
        if(cudaStreamWaitEvent(g.stream,s.ready[slot],0)!=cudaSuccess)return CUDA_RC_ERROR;
        s.previous_group=group;s.previous_group_last=index+1==s.packed.groups[group].end;
        *out=static_cast<const uint8_t*>(execution().weight_cache)+slot*capacity+s.packed.offset[index];
        return 0;
    }
    rc=weight_transfer_start();if(rc)return rc;
    // The previous weight user's kernels have now been enqueued. This event is
    // the exact overwrite fence, without a CPU synchronization per matrix.
    if(s.previous_slot>=0&&cudaEventRecord(s.consumed[s.previous_slot],g.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    rc=weight_transfer_queue(index);if(rc)return rc;
    rc=weight_transfer_queue(index+1);if(rc)return rc;
    const int slot=int(index%2);
    if(cudaStreamWaitEvent(g.stream,s.ready[slot],0)!=cudaSuccess)return CUDA_RC_ERROR;
    s.previous_slot=slot;
    *out=static_cast<const uint8_t*>(execution().weight_cache)+slot*capacity;
    return 0;
}
static int weight_transfer_finish() {
    auto &s=execution().weight_transfer;
    const bool replay=s.started&&!s.blocked&&s.cursor==s.plan.size();
    const int rc=weight_transfer_barrier(false);if(rc)return rc;
    if(weight_transfer_active()) {
        if(!s.blocked)s.plan=s.observed;else s.plan.clear();
        if(g.weight_transfer_policy==2) {
            if(!imparo_weight_transfer_plan::build(s.plan,weight_transfer_slice_limit(),s.packed))s.plan.clear();
        } else s.packed.clear();
        if(replay&&!s.reported) {
            std::fprintf(stderr,"[imparo] weight_transfer actual replay slices=%zu slots=2 bounded_cache=%llu policy=%u groups=%zu phase=%s\n",s.cursor,(unsigned long long)execution().weight_cache_bytes,g.weight_transfer_policy,s.packed.groups.size(),execution().forward_decode?"decode":"prefill");
            s.reported=true;
        }
    }
    return 0;
}
