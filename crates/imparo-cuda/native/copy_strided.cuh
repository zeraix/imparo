#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

// Backend::copy_strided, without ownership or allocation. Reuse the same pitched
// D2D primitive as DSpark feature collection. The row fallback is supplied by the
// caller so production preserves the existing k_copy behavior and stream order.
namespace imparo_cuda_copy_strided_detail {
struct Layout {
    uint64_t src_bytes = 0, dst_bytes = 0, src_offset_bytes = 0;
    uint64_t row_bytes = 0, src_pitch_bytes = 0;
    uint32_t width = 0, src_off = 0, src_stride = 0, rows = 0;
};
inline bool checked_bytes(uint64_t elements, uint64_t * bytes) {
    if (elements > std::numeric_limits<uint64_t>::max() / sizeof(float)) return false;
    *bytes = elements * sizeof(float);
    return *bytes <= std::numeric_limits<size_t>::max();
}
inline bool make_layout(uint32_t width, uint32_t src_off, uint32_t src_stride,
                        uint32_t rows, Layout * out) {
    if (!out) return false;
    *out = {};
    if (!rows || !width) return true;
    // All input factors are u32. The product plus the two remaining u32 terms
    // may still overflow u64 at the extreme, so check the addition explicitly.
    const uint64_t base = uint64_t(rows - 1) * src_stride;
    const uint64_t tail = uint64_t(src_off) + width;
    if (tail > std::numeric_limits<uint64_t>::max() - base) return false;
    Layout l;
    if (!checked_bytes(base + tail, &l.src_bytes)
        || !checked_bytes(uint64_t(rows) * width, &l.dst_bytes)
        || !checked_bytes(src_off, &l.src_offset_bytes)
        || !checked_bytes(width, &l.row_bytes)
        || !checked_bytes(src_stride, &l.src_pitch_bytes)) return false;
    l.width = width; l.src_off = src_off; l.src_stride = src_stride; l.rows = rows;
    *out = l;
    return true;
}
inline bool ranges_overlap(const void * a, uint64_t an, const void * b, uint64_t bn) {
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    // Subtraction, not end-pointer addition: it cannot wrap at the address-space end.
    return aa <= bb ? uint64_t(bb - aa) < an : uint64_t(aa - bb) < bn;
}
template<class CopyRow>
inline cudaError_t launch(float * dst, const float * src, const Layout& l,
                           cudaStream_t stream, CopyRow copy_row) {
    if (!l.rows || !l.width) return cudaSuccess;
    if (!dst || !src) return cudaErrorInvalidValue;
    const float * first = src + l.src_off;
    const uint64_t source_span = l.src_bytes - l.src_offset_bytes;
    if (l.src_stride >= l.width
        && !ranges_overlap(dst, l.dst_bytes, first, source_span)) {
        // A one-row copy never addresses a second row: an arbitrary unused source
        // stride must not become an invalid CUDA pitch for this valid contract.
        const size_t pitch = size_t(l.rows == 1 ? l.row_bytes : l.src_pitch_bytes);
        return cudaMemcpy2DAsync(dst, size_t(l.row_bytes), first, pitch,
                                 size_t(l.row_bytes), l.rows, cudaMemcpyDeviceToDevice, stream);
    }
    // Preserve the original increasing-row copy semantics for overlapping buffers
    // or a source stride shorter than width. Never pass overlapping regions to
    // cudaMemcpy2DAsync. Within-row overlap retains the old k_copy contract.
    for (uint32_t row = 0; row < l.rows; ++row) {
        const auto rc = copy_row(dst + uint64_t(row) * l.width,
                                 first + uint64_t(row) * l.src_stride, l.width, stream);
        if (rc != cudaSuccess) return rc;
    }
    return cudaSuccess;
}
} // namespace imparo_cuda_copy_strided_detail
