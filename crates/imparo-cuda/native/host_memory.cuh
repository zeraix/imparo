#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

// Backend-owned Host-tier allocation metadata. CUDA pointers deliberately stay
// outside this header and are never encoded in public handles. A handle combines
// a one-based slot with a nonzero generation, so stale and double-free operations
// fail after a slot is recycled.
namespace imparo_cuda_host {

enum class Rc : uint8_t { Ok, Invalid, OutOfMemory };

struct TransferSpan {
    uint64_t host_handle;
    uint32_t layer;
    uint32_t is_v;
    uint64_t device_offset;
    uint64_t host_offset;
    uint64_t len;
};
static_assert(sizeof(TransferSpan) == 40, "stable C transfer-span layout");
static_assert(offsetof(TransferSpan, host_handle) == 0, "stable handle offset");
static_assert(offsetof(TransferSpan, layer) == 8, "stable layer offset");
static_assert(offsetof(TransferSpan, is_v) == 12, "stable side offset");
static_assert(offsetof(TransferSpan, device_offset) == 16, "stable device offset");
static_assert(offsetof(TransferSpan, host_offset) == 24, "stable host offset");
static_assert(offsetof(TransferSpan, len) == 32, "stable length offset");

struct Allocation {
    uint64_t bytes = 0;
    uint32_t generation = 0;
    bool live = false;
};

class Registry {
public:
    Rc claim(uint64_t bytes, uint64_t * handle_out, uint32_t * index_out) {
        if (!bytes || !handle_out || !index_out
            || live_bytes_ > UINT64_MAX - bytes) {
            return Rc::Invalid;
        }
        uint32_t index = UINT32_MAX;
        for (size_t i = 0; i < allocations_.size(); ++i) {
            if (!allocations_[i].live && allocations_[i].generation != UINT32_MAX) {
                if (i >= UINT32_MAX) return Rc::Invalid;
                index = uint32_t(i);
                break;
            }
        }
        if (index == UINT32_MAX) {
            if (allocations_.size() >= UINT32_MAX) return Rc::Invalid;
            try {
                allocations_.push_back({});
            } catch (const std::bad_alloc &) {
                return Rc::OutOfMemory;
            } catch (const std::length_error &) {
                return Rc::Invalid;
            }
            index = uint32_t(allocations_.size() - 1);
        }
        Allocation & allocation = allocations_[index];
        const uint32_t generation = allocation.generation + 1;
        allocation.bytes = bytes;
        allocation.generation = generation;
        allocation.live = true;
        live_bytes_ += bytes;
        *handle_out = uint64_t(generation) << 32 | uint64_t(index + 1);
        *index_out = index;
        return Rc::Ok;
    }

    Rc release(uint64_t handle, uint32_t * index_out = nullptr) {
        uint32_t index = 0;
        Allocation * allocation = resolve_mut(handle, &index);
        if (!allocation || live_bytes_ < allocation->bytes) return Rc::Invalid;
        live_bytes_ -= allocation->bytes;
        allocation->bytes = 0;
        allocation->live = false;
        if (index_out) *index_out = index;
        return Rc::Ok;
    }

    const Allocation * resolve(uint64_t handle,
                               uint32_t * index_out = nullptr) const {
        if (!handle) return nullptr;
        const uint32_t encoded = uint32_t(handle);
        const uint32_t generation = uint32_t(handle >> 32);
        if (!encoded || !generation) return nullptr;
        const uint32_t index = encoded - 1;
        if (index >= allocations_.size()) return nullptr;
        const Allocation & allocation = allocations_[index];
        if (!allocation.live || allocation.generation != generation) return nullptr;
        if (index_out) *index_out = index;
        return &allocation;
    }

    uint64_t live_bytes() const { return live_bytes_; }
    size_t slots() const { return allocations_.size(); }

private:
    Allocation * resolve_mut(uint64_t handle, uint32_t * index_out) {
        if (!handle) return nullptr;
        const uint32_t encoded = uint32_t(handle);
        const uint32_t generation = uint32_t(handle >> 32);
        if (!encoded || !generation) return nullptr;
        const uint32_t index = encoded - 1;
        if (index >= allocations_.size()) return nullptr;
        Allocation & allocation = allocations_[index];
        if (!allocation.live || allocation.generation != generation) return nullptr;
        if (index_out) *index_out = index;
        return &allocation;
    }

    std::vector<Allocation> allocations_;
    uint64_t live_bytes_ = 0;
};

struct PlannedSpan {
    TransferSpan span;
    uint64_t staging_offset;
    uint32_t host_index;
};

inline bool intervals_overlap(uint64_t a, uint64_t a_len,
                              uint64_t b, uint64_t b_len) {
    return a < b + b_len && b < a + a_len;
}

// Validate every host-side property before CUDA sees a pointer or command. Device
// bounds and ownership remain backend-layout responsibilities. Overlapping host
// spans are rejected even when their device sources differ; overlapping device
// destinations are rejected within one (layer, K/V) side.
inline Rc build_plan(const Registry & registry, const TransferSpan * spans,
                     uint32_t count, std::vector<PlannedSpan> * out,
                     uint64_t * total_out) {
    if (!spans || !count || !out || !total_out) {
        return Rc::Invalid;
    }
    std::vector<PlannedSpan> plan;
    try {
        plan.reserve(count);
        uint64_t total = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const TransferSpan & span = spans[i];
            uint32_t host_index = 0;
            const Allocation * allocation = registry.resolve(
                span.host_handle, &host_index);
            if (!allocation || span.is_v > 1 || !span.len
                || span.host_offset > allocation->bytes
                || span.len > allocation->bytes - span.host_offset
                || span.device_offset > UINT64_MAX - span.len
                || total > UINT64_MAX - span.len
                || span.len > SIZE_MAX || total + span.len > SIZE_MAX) {
                return Rc::Invalid;
            }
            for (const PlannedSpan & prior : plan) {
                if (span.host_handle == prior.span.host_handle
                    && intervals_overlap(span.host_offset, span.len,
                                         prior.span.host_offset,
                                         prior.span.len)) {
                    return Rc::Invalid;
                }
                if (span.layer == prior.span.layer
                    && span.is_v == prior.span.is_v
                    && intervals_overlap(span.device_offset, span.len,
                                         prior.span.device_offset,
                                         prior.span.len)) {
                    return Rc::Invalid;
                }
            }
            plan.push_back({span, total, host_index});
            total += span.len;
        }
        *out = std::move(plan);
        *total_out = total;
    } catch (const std::bad_alloc &) {
        return Rc::OutOfMemory;
    } catch (const std::length_error &) {
        return Rc::Invalid;
    }
    return Rc::Ok;
}

} // namespace imparo_cuda_host
