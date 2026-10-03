#pragma once
#include "kv_paging.cuh"

namespace imparo_cuda_kv {

// The model's StateGeometry supplies byte strides and ring masks for real KV
// owners. Moving opaque rows does not dequantize or assume a particular codec.
struct RowMoveGeometry {
    uint32_t layer;
    uint32_t ring;
    uint64_t k_stride;
    uint64_t v_stride;
};
static_assert(sizeof(RowMoveGeometry) == 24, "row move wire layout");

constexpr uint32_t kMoveRows = 64;
struct RowMovePlan {
    RowMoveGeometry geometry{};
    uint32_t count = 0;
    uint32_t from[kMoveRows]{};
    uint32_t to[kMoveRows]{};
};

inline bool resolve_move_row(uint32_t logical, uint32_t ring,
        const PageTableLayer* table, uint32_t* physical) {
    if (ring) {
        if (ring == UINT32_MAX || (ring & (ring + 1))) return false;
        *physical = physical_row(logical, ring, nullptr);
        return true;
    }
    if (!table) {
        *physical = logical;
        return true;
    }
    const uint32_t page = logical / kPageCells;
    if (table->host_shadow.size() != table->capacity
        || page >= table->capacity) return false;
    const uint32_t entry = table->host_shadow[page];
    if (entry > table->max_entry || entry > UINT32_MAX / kPageCells)
        return false;
    *physical = physical_row(logical, 0, table->host_shadow.data());
    return true;
}

// Pure preflight. Failure leaves the output plan unchanged and touches no KV.
inline bool plan_row_moves(const RowMoveGeometry& geometry, uint64_t available,
        const PageTableLayer* table, const uint32_t* from, const uint32_t* to,
        uint32_t count, RowMovePlan* out) {
    if (!out || count > kMoveRows || (count && (!from || !to))
        || !geometry.k_stride || !geometry.v_stride
        || geometry.ring == UINT32_MAX
        || (geometry.ring && (geometry.ring & (geometry.ring + 1))))
        return false;
    RowMovePlan next{};
    next.geometry = geometry;
    uint32_t all_destinations[kMoveRows]{};
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t src = 0, dst = 0;
        if (!resolve_move_row(from[i], geometry.ring, table, &src)
            || !resolve_move_row(to[i], geometry.ring, table, &dst)) return false;
        const uint64_t end = uint64_t(std::max(src, dst)) + 1;
        if (!checked_bytes(end, geometry.k_stride, available)
            || !checked_bytes(end, geometry.v_stride, available)) return false;
        // Aliased destinations would make publication order dependent.
        for (uint32_t j = 0; j < i; ++j)
            if (all_destinations[j] == dst) return false;
        all_destinations[i] = dst;
        if (src != dst) {
            next.from[next.count] = src;
            next.to[next.count++] = dst;
        }
    }
    *out = next;
    return true;
}

#if defined(__CUDACC__)
// Both phases use existing CUDA D2D copies. Gather every source before any
// destination write, so even overlapping/ring-wrapped paths are safe.
inline cudaError_t copy_row_moves(const RowMovePlan& plan, uint64_t stride,
        uint8_t* cache, uint8_t* scratch, cudaStream_t stream) {
    for (uint32_t i = 0; i < plan.count; ++i) {
        const auto error = cudaMemcpyAsync(scratch + uint64_t(i) * stride,
            cache + uint64_t(plan.from[i]) * stride, size_t(stride),
            cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return error;
    }
    for (uint32_t i = 0; i < plan.count; ++i) {
        const auto error = cudaMemcpyAsync(cache + uint64_t(plan.to[i]) * stride,
            scratch + uint64_t(i) * stride, size_t(stride),
            cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}
#endif
} // namespace imparo_cuda_kv
