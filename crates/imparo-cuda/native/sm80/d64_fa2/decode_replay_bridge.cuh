#pragma once
#include <cstdint>

namespace imparo_d64_fa2_port {
// Internal typed-TU bridge. The dispatch TU owns the real by-value Params ABI.
struct DecodeReplayFa2Kernels {
 void *causal=nullptr,*merge=nullptr,*prepare_q=nullptr,*output=nullptr;
 uint64_t params_bytes=0;uint32_t shared_bytes=0;
};
struct DecodeReplayFa2Binding {
 const void *k=nullptr,*v=nullptr;uint32_t fixed_span=0;
};
DecodeReplayFa2Kernels decode_replay_fa2_kernels();
bool decode_replay_fa2_params(void*,uint32_t,void*,uint64_t,DecodeReplayFa2Binding&,bool);
}
