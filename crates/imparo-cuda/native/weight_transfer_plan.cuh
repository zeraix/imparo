#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace imparo_weight_transfer_plan {
struct Slice { uint64_t offset, bytes; };
struct Group { size_t begin, end; uint64_t bytes; };
struct Plan {
    std::vector<Group> groups;
    std::vector<size_t> group;
    std::vector<uint64_t> offset;
    void clear() { groups.clear(); group.clear(); offset.clear(); }
};
// Pack the observed access order, without changing it or exceeding either slot.
// Groups use half-open slice indices; offsets are aligned for every provider.
inline bool build(const std::vector<Slice>& slices, uint64_t capacity, Plan& out) {
    out.clear();
    for (size_t i=0; i<slices.size(); ++i) {
        const uint64_t bytes=slices[i].bytes;
        if (!bytes || bytes>capacity) { out.clear(); return false; }
        uint64_t at=0;
        if (!out.groups.empty()) {
            const uint64_t used=out.groups.back().bytes;
            const uint64_t pad=(256-used%256)%256;
            if (used<=capacity && pad<=capacity-used) at=used+pad;
            else at=capacity;
        }
        if (out.groups.empty() || bytes>capacity-at) {
            out.groups.push_back({i,i,0}); at=0;
        }
        out.group.push_back(out.groups.size()-1); out.offset.push_back(at);
        out.groups.back().end=i+1; out.groups.back().bytes=at+bytes;
    }
    return true;
}
}
