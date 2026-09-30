#pragma once
#include <cstdint>
// Static provider only. Model preparation binds these choices to resident
// weights and a checked domain before any target or draft forward.
extern "C" uint32_t imparo_cuda_lfm_retained_domain();
namespace imparo_lfm_retained {
inline uint32_t domain() {
#if defined(IMPARO_CUDA_SPECULATIVE) && defined(_WIN32)
    return imparo_cuda_lfm_retained_domain();
#else
    return 0;
#endif
}
inline bool enabled() { return domain() != 0; }
inline bool common(bool legacy) { return enabled() || legacy; }
inline bool short_only(bool legacy) { const auto d=domain(); return d ? d==1 : legacy; }
inline bool wide_only(bool legacy) { const auto d=domain(); return d ? d>=2 : legacy; }
inline bool long_only(bool legacy) { const auto d=domain(); return d ? d==3 : legacy; }
}
