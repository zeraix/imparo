#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace imparo_cuda_kv {

// Persisted engine addressing quantum. It is intentionally not a tuner knob.
constexpr uint32_t kPageCells = 64;
static_assert(kPageCells == 64, "the common KV pool persists 64-cell blocks");

#if defined(__CUDACC__)
#define IMPARO_KV_HD __host__ __device__
#else
#define IMPARO_KV_HD
#endif

/// The single logical-row mapping used by store, dequant and (in Step 8.3-C)
/// attention. A page-table entry is already a 64-row block id; shift exactly once.
IMPARO_KV_HD inline uint32_t physical_row(
        uint32_t logical, uint32_t ring, const uint32_t * page_table) {
    if (ring) return logical & ring;
    if (!page_table) return logical;
    return (page_table[logical >> 6] << 6) | (logical & (kPageCells - 1));
}

#undef IMPARO_KV_HD

struct PagingLayout {
    uint32_t layer = 0;
    uint32_t reserved = 0;
    uint64_t logical_slots = 0;
    uint64_t k_stride = 0;
    uint64_t v_stride = 0;
};
static_assert(sizeof(PagingLayout) == 32, "paging layout wire must stay ABI-stable");

inline bool checked_page_count(uint64_t slots, uint32_t * out) {
    if (!out) return false;
    const uint64_t pages = slots / kPageCells + uint64_t(slots % kPageCells != 0);
    if (pages > UINT32_MAX) return false;
    *out = uint32_t(pages);
    return true;
}

inline bool checked_bytes(uint64_t slots, uint64_t stride, uint64_t available) {
    return !slots || (stride && slots <= UINT64_MAX / stride
        && slots * stride <= available);
}

struct PageTableLayer {
    std::vector<uint32_t> host_shadow;
    uint32_t capacity = 0;
    uint32_t installed_len = 0;
    uint32_t max_entry = 0;
    uint64_t generation = 0;
    uint64_t arena_offset = 0; // uint32 entries, not bytes
};

inline bool page_table_requires_mapping(
        const std::vector<uint32_t> & shadow, uint32_t capacity) {
    if (shadow.size() != capacity) return true;
    for (uint32_t entry = 0; entry < capacity; ++entry) {
        if (shadow[entry] != entry) return true;
    }
    return false;
}

template <uint32_t MaxLayers>
struct PageTables {
    uint32_t layers = 0;
    uint64_t arena_entries = 0;
    PageTableLayer layer[MaxLayers];
};

enum class PagingRc : uint8_t { Ok, Invalid, OutOfMemory };

template <uint32_t MaxLayers>
PagingRc build_page_tables(uint32_t n_layers, const uint64_t * bytes,
                           const PagingLayout * layouts, uint32_t layout_count,
                           const PageTables<MaxLayers> * old, bool preserve,
                           PageTables<MaxLayers> * out) {
    if (!out || n_layers > MaxLayers || layout_count != n_layers
        || (n_layers && (!bytes || !layouts)) || (preserve && !old)) {
        return PagingRc::Invalid;
    }
    PageTables<MaxLayers> next;
    next.layers = n_layers;
    try {
        for (uint32_t layer = 0; layer < n_layers; ++layer) {
            const PagingLayout & spec = layouts[layer];
            if (spec.layer != layer || spec.reserved != 0) return PagingRc::Invalid;
            uint32_t capacity = 0;
            if (!checked_page_count(spec.logical_slots, &capacity)
                || !checked_bytes(spec.logical_slots, spec.k_stride, bytes[layer])
                || !checked_bytes(spec.logical_slots, spec.v_stride, bytes[layer])) {
                return PagingRc::Invalid;
            }
            if (next.arena_entries > UINT64_MAX - capacity) return PagingRc::Invalid;
            PageTableLayer & table = next.layer[layer];
            table.capacity = capacity;
            table.max_entry = capacity ? capacity - 1 : 0;
            table.arena_offset = next.arena_entries;
            table.host_shadow.resize(capacity);
            for (uint32_t entry = 0; entry < capacity; ++entry) {
                table.host_shadow[entry] = entry;
            }
            if (preserve && layer < old->layers) {
                const PageTableLayer & prior = old->layer[layer];
                if (prior.capacity > capacity || prior.installed_len > prior.capacity
                    || prior.host_shadow.size() != prior.capacity) {
                    return PagingRc::Invalid;
                }
                std::copy(prior.host_shadow.begin(), prior.host_shadow.end(),
                          table.host_shadow.begin());
                table.installed_len = prior.installed_len;
                table.generation = prior.generation;
            }
            next.arena_entries += capacity;
        }
    } catch (const std::bad_alloc &) {
        return PagingRc::OutOfMemory;
    } catch (const std::length_error &) {
        return PagingRc::Invalid;
    }
    *out = std::move(next);
    return PagingRc::Ok;
}

template <uint32_t MaxLayers>
PagingRc prepare_page_update(const PageTables<MaxLayers> & tables, uint32_t layer,
                             const uint32_t * entries, uint32_t n,
                             std::vector<uint32_t> * candidate, bool * changed) {
    if (!candidate || !changed || layer >= tables.layers || (n && !entries)) {
        return PagingRc::Invalid;
    }
    const PageTableLayer & table = tables.layer[layer];
    if (n > table.capacity || table.host_shadow.size() != table.capacity) {
        return PagingRc::Invalid;
    }
    try {
        candidate->resize(table.capacity);
    } catch (const std::bad_alloc &) {
        return PagingRc::OutOfMemory;
    } catch (const std::length_error &) {
        return PagingRc::Invalid;
    }
    for (uint32_t entry = 0; entry < table.capacity; ++entry) {
        (*candidate)[entry] = entry;
    }
    for (uint32_t entry = 0; entry < n; ++entry) {
        if (entries[entry] > table.max_entry) return PagingRc::Invalid;
        (*candidate)[entry] = entries[entry];
    }
    *changed = table.installed_len != n || *candidate != table.host_shadow;
    if (*changed && table.generation == UINT64_MAX) return PagingRc::Invalid;
    return PagingRc::Ok;
}

template <uint32_t MaxLayers>
void commit_page_update(PageTables<MaxLayers> * tables, uint32_t layer,
                        std::vector<uint32_t> candidate, uint32_t installed_len) {
    PageTableLayer & table = tables->layer[layer];
    table.host_shadow = std::move(candidate);
    table.installed_len = installed_len;
    ++table.generation;
}

} // namespace imparo_cuda_kv
