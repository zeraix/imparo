#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

// CUDA KV arena bookkeeping is independent of model geometry. Arena slices use
// CUDA-friendly alignment, while ownership learns the host pool's logical block
// quantum from the first advise call for each (layer, K/V) side.
namespace imparo_cuda_kv {

constexpr uint64_t kArenaAlignment = 4096;
constexpr uint32_t kStageSlots = 3;

enum BlockState : uint8_t {
    NeverClaimed = 0,
    Live = 1,
    Free = 2,
};

inline bool checked_align_up(uint64_t value, uint64_t * out) {
    if (!out || value > UINT64_MAX - (kArenaAlignment - 1)) return false;
    *out = (value + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
    return true;
}

struct Ownership {
    uint64_t quantum = 0;
    std::vector<uint8_t> blocks;
};

template <uint32_t MaxLayers>
struct Layout {
    uint32_t layers = 0;
    uint64_t arena_bytes = 0;
    uint64_t live_bytes = 0;
    uint64_t logical_bytes[MaxLayers] = {};
    uint64_t span_bytes[MaxLayers] = {};
    uint64_t k_offset[MaxLayers] = {};
    uint64_t v_offset[MaxLayers] = {};
    Ownership k_owner[MaxLayers];
    Ownership v_owner[MaxLayers];
};

template <uint32_t MaxLayers>
bool build_layout(uint32_t layers, const uint64_t * bytes,
                  Layout<MaxLayers> * out) {
    if (!out || layers > MaxLayers || (layers && !bytes)) return false;
    Layout<MaxLayers> next;
    next.layers = layers;
    uint64_t cursor = 0;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        uint64_t span = 0;
        if (!checked_align_up(bytes[layer], &span)
            || cursor > UINT64_MAX - span) return false;
        next.logical_bytes[layer] = bytes[layer];
        next.span_bytes[layer] = span;
        next.k_offset[layer] = cursor;
        cursor += span;
        if (cursor > UINT64_MAX - span) return false;
        next.v_offset[layer] = cursor;
        cursor += span;
    }
    next.arena_bytes = cursor;
    *out = std::move(next);
    return true;
}

template <uint32_t MaxLayers>
Ownership * ownership(Layout<MaxLayers> * layout, uint32_t layer, uint32_t is_v) {
    if (!layout || layer >= layout->layers || is_v > 1) return nullptr;
    return is_v ? &layout->v_owner[layer] : &layout->k_owner[layer];
}

template <uint32_t MaxLayers>
const Ownership * ownership(const Layout<MaxLayers> & layout,
                            uint32_t layer, uint32_t is_v) {
    if (layer >= layout.layers || is_v > 1) return nullptr;
    return is_v ? &layout.v_owner[layer] : &layout.k_owner[layer];
}

enum class OwnershipRc : uint8_t { Ok, Invalid, OutOfMemory };

template <uint32_t MaxLayers>
OwnershipRc preserve_ownership(const Layout<MaxLayers> & old_layout,
                               Layout<MaxLayers> * next) {
    if (!next) return OwnershipRc::Invalid;
    Layout<MaxLayers> working;
    try {
        working = *next;
        const uint32_t layers = std::min(old_layout.layers, working.layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (uint32_t is_v = 0; is_v < 2; ++is_v) {
                const Ownership * old_owner = ownership(old_layout, layer, is_v);
                Ownership * new_owner = ownership(&working, layer, is_v);
                if (!old_owner || !new_owner || !old_owner->quantum) continue;
                new_owner->quantum = old_owner->quantum;
                const uint64_t logical = working.logical_bytes[layer];
                const uint64_t count = logical / new_owner->quantum
                    + uint64_t(logical % new_owner->quantum != 0);
                if (count > SIZE_MAX || count > new_owner->blocks.max_size()) {
                    return OwnershipRc::Invalid;
                }
                new_owner->blocks.assign(size_t(count), NeverClaimed);
                const size_t common = std::min(old_owner->blocks.size(),
                                               new_owner->blocks.size());
                std::copy_n(old_owner->blocks.begin(), common,
                            new_owner->blocks.begin());
                for (size_t block = 0; block < common; ++block) {
                    if (new_owner->blocks[block] == Live) {
                        if (working.live_bytes > UINT64_MAX - new_owner->quantum) {
                            return OwnershipRc::Invalid;
                        }
                        working.live_bytes += new_owner->quantum;
                    }
                }
            }
        }
    } catch (const std::bad_alloc &) {
        return OwnershipRc::OutOfMemory;
    } catch (const std::length_error &) {
        return OwnershipRc::Invalid;
    }
    *next = std::move(working);
    return OwnershipRc::Ok;
}

template <uint32_t MaxLayers>
bool range_owned(const Layout<MaxLayers> & layout, uint32_t layer, uint32_t is_v,
                 uint64_t off, uint64_t len) {
    if (layer >= layout.layers || is_v > 1
        || off > layout.logical_bytes[layer]
        || len > layout.logical_bytes[layer] - off) return false;
    if (!len) return true;
    const Ownership * owner = ownership(layout, layer, is_v);
    if (!owner || !owner->quantum) return true;
    const uint64_t first = off / owner->quantum;
    const uint64_t last = (off + len - 1) / owner->quantum;
    if (last >= owner->blocks.size()) return false;
    for (uint64_t block = first; block <= last; ++block) {
        if (owner->blocks[size_t(block)] == Free) return false;
    }
    return true;
}

template <uint32_t MaxLayers>
bool range_live(const Layout<MaxLayers> & layout, uint32_t layer, uint32_t is_v,
                uint64_t off, uint64_t len) {
    if (layer >= layout.layers || is_v > 1 || !len
        || off > layout.logical_bytes[layer]
        || len > layout.logical_bytes[layer] - off) return false;
    const Ownership * owner = ownership(layout, layer, is_v);
    // Host-tier movement is a pool lifecycle operation. Unlike legacy raw KV
    // I/O it must not treat NeverClaimed storage as resident ownership.
    if (!owner || !owner->quantum) return false;
    const uint64_t first = off / owner->quantum;
    const uint64_t last = (off + len - 1) / owner->quantum;
    if (last >= owner->blocks.size()) return false;
    for (uint64_t block = first; block <= last; ++block) {
        if (owner->blocks[size_t(block)] != Live) return false;
    }
    return true;
}

template <uint32_t MaxLayers>
OwnershipRc transition(Layout<MaxLayers> * layout, uint32_t layer, uint32_t is_v,
                       uint64_t off, uint64_t len, bool reuse) {
    if (!layout || layer >= layout->layers || is_v > 1 || !len
        || off > layout->logical_bytes[layer]
        || len > layout->logical_bytes[layer] - off) {
        return OwnershipRc::Invalid;
    }
    Ownership * owner = ownership(layout, layer, is_v);
    if (!owner) return OwnershipRc::Invalid;
    if (!owner->quantum) {
        if (off % len) return OwnershipRc::Invalid;
        const uint64_t logical = layout->logical_bytes[layer];
        const uint64_t count = logical / len + uint64_t(logical % len != 0);
        std::vector<uint8_t> fresh;
        if (count > SIZE_MAX || count > fresh.max_size()) {
            return OwnershipRc::Invalid;
        }
        try {
            fresh.assign(size_t(count), NeverClaimed);
        } catch (const std::bad_alloc &) {
            return OwnershipRc::OutOfMemory;
        } catch (const std::length_error &) {
            return OwnershipRc::Invalid;
        }
        owner->quantum = len;
        owner->blocks = std::move(fresh);
    }
    if (len != owner->quantum || off % owner->quantum) {
        return OwnershipRc::Invalid;
    }
    const uint64_t index = off / owner->quantum;
    if (index >= owner->blocks.size()) return OwnershipRc::Invalid;
    const uint8_t state = owner->blocks[size_t(index)];
    if ((!reuse && state != Live) || (reuse && state == Live)) {
        return OwnershipRc::Invalid;
    }
    if (reuse && layout->live_bytes > UINT64_MAX - len) {
        return OwnershipRc::Invalid;
    }
    if (!reuse && layout->live_bytes < len) {
        return OwnershipRc::Invalid;
    }
    owner->blocks[size_t(index)] = reuse ? Live : Free;
    if (reuse) layout->live_bytes += len;
    else layout->live_bytes -= len;
    return OwnershipRc::Ok;
}

} // namespace imparo_cuda_kv
