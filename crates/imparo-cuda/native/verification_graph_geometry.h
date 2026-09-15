#pragma once
#include <cstdint>
#include <algorithm>
namespace imparo_verify_geometry {
// Query-grid kernels derive exact per-query row pitches from valid_span.
// Reuse only while all queries keep their original key-group count.
inline bool query_bucket_upper(uint32_t start, uint32_t ring, uint32_t span,
        uint32_t queries, uint32_t * upper) {
    if (!upper || queries!=3 || start>UINT32_MAX-queries || !span || ring==UINT32_MAX) return false;
    if (!ring) {
        if (span<start+queries) return false;
        *upper=span-queries;
        return true;
    }
    if (start>=ring) { *upper=UINT32_MAX-queries; return span==ring+1; }
    if (span!=start+1 || start/32!=(start+queries-1)/32) return false;
    const uint64_t group_end=(uint64_t(start)/32+1)*32;
    *upper=std::min(uint32_t(group_end-queries),ring>=queries ? ring-queries : 0u);
    return *upper>=start;
}
}
