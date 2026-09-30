#pragma once
// Internal to imparo_sm80_mmq, after constants and unpack_q4_nibbles.
// Experimental packed-resident consumer; host admission owns all bounds.
// Activation, MMA, epilogue and shared-memory ownership remain unchanged.

struct PackedWeightView {
    const uint32_t* words; // Full joined Gate+Up payload, or full Down payload.
    const __half* scales; // Same base + uint64_t(K)*packed_n/2 bytes.
    uint32_t packed_n;    // GU:20480; Down:2560. Never local GU N=10240.
    uint32_t row_offset;  // Gate:0; Up:10240; Down:0.
};

namespace packed_stage {

template <uint32_t Rows, uint32_t StageBlocks, bool FullK, bool FullRows>
__device__ __forceinline__ void stage_weights(
        PackedWeightView view, int8_t* sx, uint32_t n_out, uint32_t blocks,
        uint32_t tile_row, uint32_t stage_block, uint32_t segment_end,
        uint32_t tid) {
    static_assert(Rows == 128, "experiment supports the existing Rows128 full tile");
    static_assert(StageBlocks == 4 || StageBlocks == 8, "existing K128/K256 stages");
    // FullRows=false is used by the existing direct Stream-K epilogue.
    // launch_packed still requires complete 128-row tiles in both cases.
    constexpr uint32_t Threads = imparo_sm80_mmq::kWarps * 32;
    constexpr uint32_t WeightValues = StageBlocks * 32;
    constexpr uint32_t WeightStride = StageBlocks == 4
        ? imparo_sm80_mmq::kHalfKWeightStride : imparo_sm80_mmq::kWeightStride;
    constexpr uint32_t RowTiles = Rows / 64;
    constexpr uint32_t PayloadWords = Rows * WeightValues / 8;
    static_assert(WeightStride >= WeightValues + StageBlocks * sizeof(float),
                  "scale slots must follow values within each original row");
    static_assert(WeightStride % 4 == 0, "original shared row alignment");

#if defined(IMPARO_CUDA_PACKED_STAGE_DEBUG)
    assert(view.words && view.scales && sx);
    assert((reinterpret_cast<uintptr_t>(view.words) & 3u) == 0);
    assert((reinterpret_cast<uintptr_t>(view.scales) & 1u) == 0);
    assert((reinterpret_cast<uintptr_t>(sx) & 3u) == 0);
    assert(blockDim.x == 32 && blockDim.y == imparo_sm80_mmq::kWarps
           && blockDim.z == 1);
    assert(tid == threadIdx.y * 32 + threadIdx.x && tid < Threads);
    assert(view.packed_n != 0 && view.packed_n % 64 == 0);
    assert(view.row_offset % 64 == 0 && tile_row % 64 == 0);
    assert(view.row_offset <= view.packed_n
           && n_out <= view.packed_n - view.row_offset);
    assert(tile_row <= n_out && Rows <= n_out - tile_row);
    assert(stage_block <= segment_end && segment_end <= blocks);
    if constexpr (FullK) {
        assert(StageBlocks <= segment_end - stage_block);
    }

#endif

    const uint32_t first_row = view.row_offset + tile_row;
    const uint32_t row_tiles_total = view.packed_n / 64;
    // Each 16K x 64N tile is 128 consecutive words. A CTA wave loads the
    // two adjacent N tiles contiguously before moving to the next 16K tile.
    for (uint32_t linear = tid; linear < PayloadWords; linear += Threads) {
        const uint32_t tile = linear / 128;
        const uint32_t local_word = linear % 128;
        const uint32_t local_k16 = tile / RowTiles;
        const uint32_t local_n64 = tile % RowTiles;
        const uint32_t qblock = local_k16 / 2;
        const uint32_t kb = stage_block + qblock;
        uint32_t packed = 0x88888888u; // Original MMQ zero-value nibble.
        if (FullK || kb < segment_end) {
            const uint64_t word_index =
                (uint64_t(stage_block * 2 + local_k16) * row_tiles_total
                 + first_row / 64 + local_n64) * 128 + local_word;
            packed = view.words[word_index];
        }
        const uint32_t lane = local_word / 4;
        const uint32_t warp = local_word % 4;
        const uint32_t row0 = local_n64 * 64 + warp * 16 + lane / 4;
        const uint32_t k0 = local_k16 * 16 + 2 * (lane % 4);
        // a bytes: (row0,k0), (row0+8,k0), (row0,k0+1), (row0+8,k0+1).
        // b has the same coordinates with K increased by eight.
        const uint32_t a = uint32_t(imparo_sm80_mmq::unpack_q4_nibbles(packed));
        const uint32_t b = uint32_t(imparo_sm80_mmq::unpack_q4_nibbles(packed >> 4));
        const uint16_t a0 = uint16_t((a & 0xffu) | ((a >> 8) & 0xff00u));
        const uint16_t a8 = uint16_t(((a >> 8) & 0xffu) | ((a >> 16) & 0xff00u));
        const uint16_t b0 = uint16_t((b & 0xffu) | ((b >> 8) & 0xff00u));
        const uint16_t b8 = uint16_t(((b >> 8) & 0xffu) | ((b >> 16) & 0xff00u));
        *reinterpret_cast<uint16_t*>(sx + row0 * WeightStride + k0) = a0;
        *reinterpret_cast<uint16_t*>(sx + (row0 + 8) * WeightStride + k0) = a8;
        *reinterpret_cast<uint16_t*>(sx + row0 * WeightStride + k0 + 8) = b0;
        *reinterpret_cast<uint16_t*>(sx + (row0 + 8) * WeightStride + k0 + 8) = b8;
    }

    // Packed scales are [K/32,N], with every 8x8 N tile transposed.
    // Reading them in stored order is contiguous; the transpose is its own
    // inverse. Load the unchanged half bits and use the original half2float.
    for (uint32_t linear = tid; linear < Rows * StageBlocks; linear += Threads) {
        const uint32_t qblock = linear / Rows;
        const uint32_t packed_row = linear % Rows;
        const uint32_t row = (packed_row / 64) * 64
            + (packed_row % 8) * 8 + (packed_row % 64) / 8;
        const uint32_t kb = stage_block + qblock;
        float scale = 0.0f;
        if (FullK || kb < segment_end) {
            scale = __half2float(view.scales[uint64_t(kb) * view.packed_n
                                            + first_row + packed_row]);
        }
        reinterpret_cast<float*>(sx + row * WeightStride + WeightValues)[qblock] = scale;
    }
    // Caller performs the existing wait_async_copies()/__syncthreads().
}
} // namespace packed_stage
