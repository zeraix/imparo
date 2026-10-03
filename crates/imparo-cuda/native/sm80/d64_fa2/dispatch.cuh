#pragma once
#include <cstdint>
#include "decode_replay_bridge.cuh"

namespace imparo_d64_fa2_port {
// Capacity bound for the admitted D64/GQA4 M9 or M512 shapes. The vendored
// dispatcher uses chunk_size >= 256, so its actual partition count cannot
// exceed ceil(valid/256), independently of the selected kernel's occupancy.
// Each partition stores half outputs followed by one FP32 LSE per query/head.
// With head_dim=64 the LSE offset is naturally 4-byte aligned. Wide Prefill
// passes nullptr for tmp and needs only the two half Q/output adapters.
constexpr uint64_t workspace_bytes(unsigned nt, unsigned nh, unsigned valid,
                                   bool split_kv) {
    const uint64_t query_heads = uint64_t(nt) * nh;
    const uint64_t chunks = (uint64_t(valid) + 255) / 256;
    return 2 * query_heads * 64 * sizeof(__half)
        + (split_kv ? chunks * query_heads * (64 * sizeof(__half) + sizeof(float)) : 0);
}
static_assert(workspace_bytes(9, 32, 16640, true) == 2544768);
static_assert(workspace_bytes(512, 32, 16384, false) == 4194304);
cudaError_t launch_batch_invariant_v1(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,unsigned nt,unsigned valid,float scale,unsigned span,cudaStream_t stream);
cudaError_t tree_tail_plan_v1(unsigned start,unsigned span,unsigned leaves,unsigned*chunks,unsigned*begin,unsigned*tail,uint64_t*extra,unsigned nodes);
cudaError_t launch_tree_tail_v1(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,void*scratch,uint8_t*mask,const int*parents,unsigned start,unsigned span,unsigned chunks,unsigned begin,unsigned tail,unsigned leaf_mask,float scale,cudaStream_t stream,unsigned nodes);
cudaError_t launch(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,unsigned nt,unsigned nh,unsigned nk,unsigned valid,float scale,bool causal,cudaStream_t stream,unsigned fixed_span=0);
cudaError_t launch_mask(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,unsigned nt,unsigned nh,unsigned nk,unsigned valid,float scale,uint8_t*mask,cudaStream_t stream,unsigned partition_reference_len,unsigned fixed_span);
cudaError_t tree_tail_plan(unsigned start,unsigned fixed_span,unsigned leaves,unsigned*chunks,unsigned*begin,unsigned*tail,uint64_t*extra);
cudaError_t launch_tree_tail(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,void*scratch,uint8_t*mask,const int*parents,unsigned start,unsigned fixed_span,unsigned chunks,unsigned begin,unsigned tail,unsigned leaf_mask,float scale,cudaStream_t stream);
bool tree_tail_shared_kv_enabled();
cudaError_t launch_tree_tail_mapped(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,void*scratch,uint8_t*mask,const int*parents,unsigned start,unsigned fixed_span,unsigned chunks,unsigned begin,unsigned tail,unsigned leaf_mask,float scale,cudaStream_t stream);
}
