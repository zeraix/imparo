#include "flashinfer_port.cuh"
#include "../../lfm_retained_policy.cuh"
#include "decode_replay_bridge.cuh"

#include "tree_tail.cuh"
#include "tree_tail_mapped.cuh"
#include <cstdlib>
#include <cstring>
namespace imparo_d64_fa2_port {
bool tree_tail_shared_kv_enabled(){const char*p=std::getenv("IMPARO_LAB_TREE_TAIL_SHARED_KV");return imparo_lfm_retained::long_only(p&&std::strcmp(p,"1")==0);}
}

#define IMPARO_TREE_REPLAY_FA2_BRIDGE
#include "../../tree_replay_plan.cuh"

namespace imparo_d64_fa2_port {
DecodeReplayFa2Kernels decode_replay_fa2_kernels(){
 return {(void*)flashinfer::SinglePrefillWithKVCacheKernel<BatchInvariantKT,PartitionParams>,
  (void*)merge_v1,(void*)prepare_q,(void*)output_f32,
  sizeof(PartitionParams),uint32_t(sizeof(BatchInvariantKT::SharedStorage))};
}
bool decode_replay_fa2_params(void*blob,uint32_t start,void*scratch,uint64_t bytes,
 DecodeReplayFa2Binding&b,bool validate){
 if(!blob||!scratch||start==UINT32_MAX)return false;
 PartitionParams p;std::memcpy(&p,blob,sizeof(p));
 const uint32_t span=validate?p.partition_fixed_span:b.fixed_span;
 if(span<256||span%128||p.partition_fixed_span!=span)return false;
 constexpr uint64_t QW=2048,QB=QW*sizeof(__half);
 const uint64_t chunks=(uint64_t(start)+1+span-1)/span;
 if(!chunks||chunks>UINT32_MAX||2*QB+chunks*(QB+32*sizeof(float))>bytes)return false;
 auto*base=static_cast<uint8_t*>(scratch);auto*tmp=reinterpret_cast<__half*>(base+2*QB);
 auto*ls=reinterpret_cast<float*>(tmp+chunks*QW);
 if(p.qo_len!=1||p.num_qo_heads!=32||p.num_kv_heads!=8||uint32_t(p.group_size)!=4
  ||p.head_dim!=64||p.q_stride_n!=QW||p.q_stride_h!=64
  ||p.k_stride_n!=512||p.v_stride_n!=512||p.k_stride_h!=64||p.v_stride_h!=64
  ||p.partition_kv!=1||p.batch_invariant_q8_v1!=1||p.partition_reference_len!=0
  ||p.window_left!=-1||p.maybe_custom_mask||p.maybe_alibi_slopes
  ||p.logits_soft_cap!=0.f||p.sm_scale!=1.f
  ||p.q!=reinterpret_cast<__half*>(base)||p.o!=tmp||!p.k||!p.v)return false;
 if(validate){if(p.kv_len!=start+1||p.lse!=ls)return false;b={p.k,p.v,span};return true;}
 if(p.k!=b.k||p.v!=b.v)return false;
 p.kv_len=start+1;p.lse=ls;std::memcpy(blob,&p,sizeof(p));return true;
}
}
