#pragma once

// Ampere Q4_0 x Q8_1 MMQ. One CTA owns a 128x128 output tile and stages 256
// K values at a time. The layout intentionally mirrors the hardware hierarchy:
// eight warps each own two 16-row fragments and alternating 16-token fragments.
// This removes repeated Q4 unpacking across tiny 16x16 CTAs while preserving the
// per-32-value integer dot and f32 scaling boundary required by the llama oracle.
namespace imparo_sm80_mmq {

enum class LaunchRoute : uint8_t {
    None,
    GridTile,
    FullTile,
    PhysicalStreamK,
    VirtualStreamK,
    VirtualNoSeam,
    VirtualNoSeamRows64,
    VirtualNoSeamRows64Tokens256Lab,
    VirtualSingleActivationRows128Lab,
    VirtualFullKRows64Lab,
    VirtualStreamRows64Lab,
    PackedDp4aLab,
    VirtualNoSeamAsyncActivation,
    VirtualDirectSeam,
    Exact128Token64SingleSeamLab,
};

enum class FullTileVariant : uint8_t {
    Rows128K256,
    Rows64K256,
    Rows128K128,
};

// Keep architecture policy beside the kernels it selects. A tuned row override
// remains backward compatible, while the automatic SM86 path can choose a
// different K-stage without leaking its shared-memory geometry into common code.
inline FullTileVariant select_full_tile_variant(
        uint32_t sm_version, uint32_t tuned_rows) {
    if (tuned_rows == 64) return FullTileVariant::Rows64K256;
    if (tuned_rows == 128) return FullTileVariant::Rows128K256;
    return sm_version == 86 ? FullTileVariant::Rows128K128
                            : FullTileVariant::Rows128K256;
}

inline uint32_t select_full_tile_min_efficiency(
        uint32_t sm_version, uint32_t tuned_percent) {
    if (tuned_percent >= 50 && tuned_percent <= 100) return tuned_percent;
    // SM86's K128 tile can replay the reference Stream-K seam inside a full
    // CTA.  Its low-occupancy projections also win by assigning one physical
    // block per logical tile as soon as the tile grid reaches half a wave;
    // this avoids cross-CTA fixup without encoding any model shape here.
    return sm_version == 86 ? 50u : 90u;
}

// Architecture-owned launch description. Common CUDA policy and diagnostics may
// inspect this result without copying SM80 tile constants or route thresholds.
// Future tile implementations can return the same contract while keeping their
// compile-time geometry local to the architecture layer.
struct LaunchInfo {
    LaunchRoute route = LaunchRoute::None;
    uint32_t tile_rows = 0;
    uint32_t tile_tokens = 0;
    uint32_t logical_tiles = 0;
    uint32_t physical_blocks = 0;
    uint32_t efficiency = 0;
    bool fused_q8 = false;
};

inline const char * launch_route_name(LaunchRoute route) {
    switch (route) {
        case LaunchRoute::GridTile: return "grid-tile";
        case LaunchRoute::FullTile: return "full-tile";
        case LaunchRoute::PhysicalStreamK: return "physical-stream-k";
        case LaunchRoute::VirtualStreamK: return "virtual-stream-k";
        case LaunchRoute::VirtualNoSeam: return "virtual-no-seam";
        case LaunchRoute::VirtualNoSeamRows64:
            return "virtual-no-seam-r64";
        case LaunchRoute::VirtualNoSeamRows64Tokens256Lab:
            return "virtual-no-seam-r64-j256-lab";
        case LaunchRoute::VirtualSingleActivationRows128Lab:
            return "virtual-single-activation-r128-lab";
        case LaunchRoute::VirtualFullKRows64Lab:
            return "virtual-full-k-r64-lab";
        case LaunchRoute::VirtualStreamRows64Lab:
            return "virtual-stream-r64-lab";
        case LaunchRoute::PackedDp4aLab:
            return "packed-dp4a-lab";
        case LaunchRoute::VirtualNoSeamAsyncActivation:
            return "virtual-no-seam-async-activation";
        case LaunchRoute::VirtualDirectSeam: return "virtual-direct-seam";
        case LaunchRoute::Exact128Token64SingleSeamLab:
            return "exact128-token64-single-seam-lab";
        default: return "none";
    }
}

constexpr uint32_t kRows = 128;
constexpr uint32_t kTokens = 128;
constexpr uint32_t kHalfTokens = 64;
constexpr uint32_t kKStage = 256;
constexpr uint32_t kWarps = 8;
// Keep Q4 scales as f32 in shared memory. Converting each source half once while
// staging avoids repeating half->float conversion for every token fragment.
// Four additional int words retain the Ampere ldmatrix bank skew: the complete
// row is 76 int words, so its stride is 4 modulo 8 rather than conflict-prone 0.
constexpr uint32_t kWeightStride = kKStage + 32 + 16;
constexpr uint32_t kActivationStage = 128;
constexpr uint32_t kActivationStride = kActivationStage + 16;
constexpr uint32_t kSharedBytes = kRows * kWeightStride
    + 2 * kTokens * kActivationStride;
constexpr uint32_t kSingleActivationSharedBytes = kRows * kWeightStride
    + kTokens * kActivationStride;
constexpr uint32_t kHalfTokenSharedBytes = kRows * kWeightStride
    + kHalfTokens * kActivationStride;
constexpr uint32_t kCompactRows = 64;
constexpr uint32_t kCompactSharedBytes = kCompactRows * kWeightStride
    + kTokens * kActivationStride;
constexpr uint32_t kWideTokens = 256;
constexpr uint32_t kWideCompactSharedBytes = kCompactRows * kWeightStride
    + kWideTokens * kActivationStride;
constexpr uint32_t kHalfKBlocks = 4;
constexpr uint32_t kHalfKValues = kHalfKBlocks * 32;
// The 128 values plus four f32 scales already leave the row stride at four
// words modulo eight.  Unlike K256, this tile needs no extra bank-skew tail.
constexpr uint32_t kHalfKWeightStride =
    kHalfKValues + kHalfKBlocks * sizeof(float);
constexpr uint32_t kHalfKSharedBytes = kRows * kHalfKWeightStride
    + kTokens * kActivationStride;

// Batched PLE gate epilogue. Keep the semantic GELU then per-layer multiply in
// f32, publish that common result, and prepare the exact group-major Q8 layout
// consumed by the following MMQ projection. One CTA owns a token row and one
// lane owns four adjacent values, so the semantic f32 result and Q8 record are
// produced in one coalesced pass. The common CUDA boundary selects this
// capability without learning the architecture's record mapping.
template <bool VectorLoads>
__global__ void ple_gate_q8_mmq(
        float * __restrict__ gate, const float * __restrict__ per_layer,
        BlockQ8_1Mmq * __restrict__ q8,
        uint32_t width, uint32_t per_layer_off, uint32_t per_layer_stride,
        uint32_t n_tok) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tok) return;
    float * row = gate + uint64_t(tok) * width;
    const float * scale = per_layer + per_layer_off
        + uint64_t(tok) * per_layer_stride;
    for (uint32_t i0 = threadIdx.x * 4; i0 < width;
         i0 += blockDim.x * 4) {
        float4 gate4;
        float4 scale4;
        if constexpr (VectorLoads) {
            gate4 = *reinterpret_cast<const float4 *>(row + i0);
            scale4 = *reinterpret_cast<const float4 *>(scale + i0);
        } else {
            gate4 = make_float4(
                row[i0 + 0], row[i0 + 1], row[i0 + 2], row[i0 + 3]);
            scale4 = make_float4(
                scale[i0 + 0], scale[i0 + 1], scale[i0 + 2], scale[i0 + 3]);
        }
        float4 xi;
        xi.x = cuda_gelu(gate4.x) * scale4.x;
        xi.y = cuda_gelu(gate4.y) * scale4.y;
        xi.z = cuda_gelu(gate4.z) * scale4.z;
        xi.w = cuda_gelu(gate4.w) * scale4.w;
        *reinterpret_cast<float4 *>(row + i0) = xi;
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffff, amax, off, 32));
        }
        const float d_inv = 127.0f / amax;
        char4 quant;
        quant.x = int8_t(roundf(xi.x * d_inv));
        quant.y = int8_t(roundf(xi.y * d_inv));
        quant.z = int8_t(roundf(xi.z * d_inv));
        quant.w = int8_t(roundf(xi.w * d_inv));
        const float d = 1.0f / d_inv;
        const uint32_t block = i0 / 32;
        const uint32_t block_in_group = block % 4;
        const uint32_t iqs = i0 % 32;
        BlockQ8_1Mmq * out = q8
            + uint64_t(block / 4) * n_tok + tok;
        reinterpret_cast<char4 *>(
            out->qs + block_in_group * 32)[iqs / 4] = quant;
        if (iqs == 0) {
            out->d[block_in_group] = __half2float(__float2half(d));
        }
    }
}

inline void launch_ple_gate_q8_mmq(
        float * gate, const float * per_layer, BlockQ8_1Mmq * q8,
        uint32_t width, uint32_t per_layer_off, uint32_t per_layer_stride,
        uint32_t n_tok, cudaStream_t stream) {
    // One lane owns four adjacent values. Avoid launching idle warps for narrow
    // PLE widths while retaining enough lanes to cover wide rows in a few
    // coalesced iterations. The override is a diagnostic/tuning hook; the
    // default remains shape-derived and portable across models.
    uint32_t threads = min(256u, max(32u, width / 4));
    if (const char * forced = std::getenv("IMPARO_CUDA_PLE_THREADS")) {
        const uint32_t parsed = uint32_t(std::strtoul(forced, nullptr, 10));
        if (parsed >= 32 && parsed <= 256 && parsed % 32 == 0) {
            threads = parsed;
        }
    }
    if ((per_layer_off | per_layer_stride) % 4 == 0) {
        ple_gate_q8_mmq<true><<<n_tok, threads, 0, stream>>>(
            gate, per_layer, q8, width, per_layer_off, per_layer_stride, n_tok);
    } else {
        ple_gate_q8_mmq<false><<<n_tok, threads, 0, stream>>>(
            gate, per_layer, q8, width, per_layer_off, per_layer_stride, n_tok);
    }
}

__device__ __forceinline__ uint32_t accumulator_row(uint32_t lane, uint32_t item) {
    return (item / 2) * 8 + lane / 4;
}

__device__ __forceinline__ uint32_t accumulator_token(uint32_t lane, uint32_t item) {
    return (lane % 4) * 2 + item % 2;
}

// Load one 16x32 row-major int8 A fragment from shared memory in the register
// layout consumed by mma.m16n8k32.  ldmatrix reads 16-bit lanes; four registers
// per thread represent the full 32-byte K dimension.
__device__ __forceinline__ void load_a_m16n8k32(
        int (&a)[4], const int8_t * base, uint32_t stride) {
    const int * address = reinterpret_cast<const int *>(base)
        + (threadIdx.x % 16) * (stride / 4) + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "l"(address));
}

// B is already arranged as eight token rows by eight packed int32 K words.
// A generic shared-memory load is faster here than transposing through ldmatrix.
__device__ __forceinline__ void load_b_m16n8k32(
        int (&b)[2], const int8_t * base, uint32_t stride) {
    const uint32_t lane = threadIdx.x;
    const int * row = reinterpret_cast<const int *>(
        base + (lane / 4) * stride);
    b[0] = row[lane % 4];
    b[1] = row[4 + lane % 4];
}

__device__ __forceinline__ void mma_m16n8k32(
        int (&c)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ int unpack_q4_nibbles(uint32_t nibbles) {
    // Map four unsigned nibbles to four signed bytes in [-8, 7]. For each
    // byte, bits 0..2 pass through and inverted bit 3 fills bits 3..7. The
    // multiply is carry-free because each byte lane is either 0 or 8 and
    // 8 * 31 == 248, so this is an exact packed-byte transform.
    const uint32_t sign_fill = (~nibbles & 0x08080808u) * 31u;
    return int((nibbles & 0x07070707u) | sign_fill);
}

// Move one aligned activation vector directly from global to shared memory.
// Avoiding the four-register round trip is especially valuable in Stream-K,
// whose accumulator set already puts substantial pressure on SM86 registers.
__device__ __forceinline__ void copy_global_to_shared_16(
        uint4 * dst, const uint4 * src) {
    const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    // SM86's compact K128 tile issues enough paired-row activation traffic that
    // retaining it in L1 evicts more useful working data. Keep the established
    // L1+L2 policy for other, independently shipped SM binaries until measured.
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        : : "r"(shared), "l"(src));
}

// Predicated cp.async zero-fills a reused shared tile when a virtual token lies
// outside the compact request. The source address remains a legal allocation
// even when src_size is zero; passing nullptr here is not a valid CUDA contract.
__device__ __forceinline__ void copy_global_to_shared_16_zfill(
        uint4 * dst, const uint4 * src, uint32_t src_size) {
    const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
        : : "r"(shared), "l"(src), "r"(src_size));
}

__device__ __forceinline__ void commit_async_copies() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void wait_async_copies() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void stage_activation_async(
        const BlockQ8_1Mmq * __restrict__ x, int8_t * sy,
        uint32_t n_tok, uint32_t tile_token, uint32_t active_tokens,
        uint32_t group_index, uint32_t tid) {
    static_assert(sizeof(BlockQ8_1Mmq) % sizeof(uint4) == 0,
                  "MMQ activation record vector alignment");
    constexpr uint32_t vectors_per_record =
        sizeof(BlockQ8_1Mmq) / sizeof(uint4);
    const uint32_t copy_count = active_tokens * vectors_per_record;
    const uint4 * src = reinterpret_cast<const uint4 *>(
        x + uint64_t(group_index) * n_tok + tile_token);
    uint4 * dst = reinterpret_cast<uint4 *>(sy);
    for (uint32_t linear = tid; linear < copy_count;
         linear += kWarps * 32) {
        copy_global_to_shared_16(dst + linear, src + linear);
    }
}

// Compute one contiguous K segment for one 128x128 output tile. Keeping the
// segment primitive independent of grid ownership lets the regular tiled route
// and the physical Stream-K route share exactly the same arithmetic contract.
template <uint32_t Rows, uint32_t RowFragments, bool DoubleBuffer,
          bool FullK, bool FullRows, uint32_t StageBlocks = 8,
          bool CooperativeQ4 = false, uint32_t TokenTile = kTokens>
__device__ __forceinline__ void compute_segment(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        int8_t * sx, int8_t * sy,
        float (&partial)[(TokenTile / 4) * RowFragments],
        uint32_t n_out, uint32_t n_tok, uint32_t work_n_tok, uint32_t blocks,
        uint32_t tile_row, uint32_t tile_token,
        uint32_t segment_begin, uint32_t segment_end) {
    static_assert(Rows == 64 || Rows == 128, "supported MMQ row tile");
    static_assert(Rows == 64 * RowFragments, "warp/row-fragment mapping");
    static_assert(StageBlocks == 4 || StageBlocks == 8,
                  "supported MMQ K-stage width");
    static_assert(TokenTile == 64 || TokenTile == 128 || TokenTile == 256,
                  "supported MMQ token tile");
    static_assert(!DoubleBuffer || StageBlocks == 8,
                  "double buffering requires two activation records");
    constexpr uint32_t WeightValues = StageBlocks * 32;
    constexpr uint32_t WeightStride = StageBlocks == 4
        ? kHalfKWeightStride : kWeightStride;
    constexpr uint32_t ActivationPhases = StageBlocks / 4;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    // The final token tile is often only partly occupied (for example 445 tokens
    // in four 128-token tiles). Keep the output bounds tied to the real token
    // count, but let production launches stop staging and issuing MMA work once
    // every lane of a token fragment is outside the valid tail. The diagnostic
    // fallback passes a padded work count and reproduces the former zero-filled
    // work exactly.
    const uint32_t active_tokens = tile_token < work_n_tok
        ? min(TokenTile, work_n_tok - tile_token) : 0;
#pragma unroll
    for (uint32_t item = 0; item < (TokenTile / 4) * RowFragments; ++item) {
        partial[item] = 0.0f;
    }

    for (uint32_t stage_block = segment_begin;
         stage_block < segment_end; stage_block += StageBlocks) {
        // Start activation traffic before unpacking the independent Q4 tile.
        // The two shared regions do not overlap, so Ampere can hide much of the
        // global-memory latency underneath the integer transform below.
        if constexpr (DoubleBuffer) {
#pragma unroll
            for (uint32_t phase = 0; phase < ActivationPhases; ++phase) {
                stage_activation_async(
                    x, sy + phase * TokenTile * kActivationStride,
                    n_tok, tile_token, active_tokens,
                    (stage_block + phase * 4) / 4, tid);
            }
        } else {
            stage_activation_async(
                x, sy, n_tok, tile_token, active_tokens,
                stage_block / 4, tid);
        }
        commit_async_copies();

        // Assign a complete Q4 block to each logical loader.  The old word-wise
        // mapping repeated the 64-bit row/block address calculation four times
        // and needed a per-word scale predicate.  Keeping all four packed words
        // together preserves the exact shared layout and MMA order while making
        // the global-memory transform substantially cheaper.
        constexpr uint32_t packed_block_count = Rows * StageBlocks;
        if constexpr (CooperativeQ4) {
            constexpr uint32_t packed_word_count = packed_block_count * 4;
            for (uint32_t linear = tid; linear < packed_word_count;
                 linear += kWarps * 32) {
                const uint32_t packed_block = linear / 4;
                const uint32_t word = linear % 4;
                const uint32_t qblock = packed_block % StageBlocks;
                const uint32_t local_row = packed_block / StageBlocks;
                const uint32_t row = tile_row + local_row;
                const uint32_t kb = stage_block + qblock;
                uint32_t packed = 0x88888888u;
                float scale = 0.0f;
                if ((FullRows || row < n_out) && (FullK || kb < segment_end)) {
                    const uint8_t * block =
                        w + (uint64_t(row) * blocks + kb) * 18;
                    const uint16_t * qs =
                        reinterpret_cast<const uint16_t *>(block + 2);
                    packed = uint32_t(qs[2 * word])
                        | (uint32_t(qs[2 * word + 1]) << 16);
                    if (word == 0) {
                        scale = __half2float(
                            *reinterpret_cast<const __half *>(block));
                    }
                }
                int8_t * dst = sx + local_row * WeightStride + qblock * 32;
                reinterpret_cast<int *>(dst)[word] =
                    unpack_q4_nibbles(packed);
                reinterpret_cast<int *>(dst + 16)[word] =
                    unpack_q4_nibbles(packed >> 4);
                if (word == 0) {
                    reinterpret_cast<float *>(sx + local_row * WeightStride
                        + WeightValues)[qblock] = scale;
                }
            }
        } else {
            for (uint32_t linear = tid; linear < packed_block_count;
                 linear += kWarps * 32) {
                const uint32_t qblock = linear % StageBlocks;
                const uint32_t local_row = linear / StageBlocks;
                const uint32_t row = tile_row + local_row;
                const uint32_t kb = stage_block + qblock;
                float scale = 0.0f;
                const uint16_t * qs = nullptr;
                if ((FullRows || row < n_out) && (FullK || kb < segment_end)) {
                    const uint8_t * block =
                        w + (uint64_t(row) * blocks + kb) * 18;
                    qs = reinterpret_cast<const uint16_t *>(block + 2);
                    scale = __half2float(
                        *reinterpret_cast<const __half *>(block));
                }
                int8_t * dst = sx + local_row * WeightStride + qblock * 32;
#pragma unroll
                for (uint32_t word = 0; word < 4; ++word) {
                    uint32_t packed = 0x88888888u;
                    if (qs) {
                        packed = uint32_t(qs[2 * word])
                            | (uint32_t(qs[2 * word + 1]) << 16);
                    }
                    reinterpret_cast<int *>(dst)[word] =
                        unpack_q4_nibbles(packed);
                    reinterpret_cast<int *>(dst + 16)[word] =
                        unpack_q4_nibbles(packed >> 4);
                }
                reinterpret_cast<float *>(
                    sx + local_row * WeightStride + WeightValues)[qblock] = scale;
            }
        }

        // Large row tiles preload both phases and halve the barrier count. The
        // compact tile deliberately reuses one buffer so it remains small enough
        // for two resident CTAs on SM86.
        wait_async_copies();
        __syncthreads();

#pragma unroll
        for (uint32_t phase = 0; phase < ActivationPhases; ++phase) {
            if constexpr (!DoubleBuffer) {
                if (phase != 0) {
                    stage_activation_async(
                        x, sy, n_tok, tile_token, active_tokens,
                        (stage_block + phase * 4) / 4, tid);
                    commit_async_copies();
                    wait_async_copies();
                    __syncthreads();
                }
            }
            const int8_t * sy_phase = sy
                + (DoubleBuffer ? phase * TokenTile * kActivationStride : 0);
#pragma unroll
            for (uint32_t qblock = 0; qblock < 4; ++qblock) {
                const uint32_t local_kb = phase * 4 + qblock;
                int af[RowFragments][4];
                float d4[RowFragments][2];
#pragma unroll
                for (uint32_t row_fragment = 0; row_fragment < RowFragments;
                     ++row_fragment) {
                    const uint32_t local_row0 = (warp >> 1) * (16 * RowFragments)
                        + row_fragment * 16;
                    load_a_m16n8k32(
                        af[row_fragment], sx + local_row0 * WeightStride
                            + local_kb * 32,
                        WeightStride);
#pragma unroll
                    for (uint32_t scale_item = 0; scale_item < 2;
                         ++scale_item) {
                        const uint32_t local_row = local_row0
                            + accumulator_row(lane, scale_item * 2);
                        d4[row_fragment][scale_item] =
                            reinterpret_cast<const float *>(
                                sx + local_row * WeightStride
                                    + WeightValues)[local_kb];
                    }
                }
#pragma unroll
                for (uint32_t token_group = 0;
                     token_group < TokenTile / 32; ++token_group) {
#pragma unroll
                    for (uint32_t token_fragment = 0; token_fragment < 2;
                         ++token_fragment) {
                        const uint32_t local_token0 = token_group * 32
                            + (warp & 1) * 16 + token_fragment * 8;
                        // Uniform within a warp: no thread in this fragment owns
                        // a valid output, so skip both B loads and tensor-core work.
                        if (local_token0 >= active_tokens) continue;
                        int bf[2];
                        load_b_m16n8k32(
                            bf, sy_phase + local_token0 * kActivationStride
                                + qblock * 32,
                            kActivationStride);
                        float d8[2];
#pragma unroll
                        for (uint32_t scale_item = 0; scale_item < 2;
                             ++scale_item) {
                            const uint32_t local_token = local_token0
                                + accumulator_token(lane, scale_item);
                            d8[scale_item] = reinterpret_cast<const float *>(
                                sy_phase + local_token * kActivationStride
                                    + kActivationStage)[qblock];
                        }
#pragma unroll
                        for (uint32_t row_fragment = 0;
                             row_fragment < RowFragments;
                             ++row_fragment) {
                            int cf[4] = {};
                            mma_m16n8k32(cf, af[row_fragment], bf);
#pragma unroll
                            for (uint32_t item = 0; item < 4; ++item) {
                                const uint32_t sum_index =
                                    (((token_group * 2 + token_fragment)
                                        * RowFragments
                                        + row_fragment) * 4) + item;
                                partial[sum_index] += float(cf[item])
                                    * d4[row_fragment][item / 2] * d8[item % 2];
                            }
                        }
                    }
                }
            }
            if constexpr (!DoubleBuffer) {
                __syncthreads();
            }
        }
        if constexpr (DoubleBuffer) {
            __syncthreads();
        }
    }
}

template <bool NumericSplit, bool SingleSeam = false,
          bool AsyncActivation = false, uint32_t TokenTile = kTokens>
__global__ void q4_q8_1_mma_128(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t epilogue, uint32_t out_stride, uint32_t row_base,
        uint32_t sm_count, uint32_t stream_k_numeric,
        uint32_t virtual_token_base, uint32_t virtual_schedule,
        uint32_t no_seam_scan) {
#if __CUDA_ARCH__ >= 800
    static_assert(!SingleSeam || NumericSplit,
        "single-seam replay requires numeric splitting");
    using namespace nvcuda;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kWeightStride;

    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t tile_row = blockIdx.x * kRows;
    static_assert(TokenTile == kHalfTokens || TokenTile == kTokens,
                  "supported streamed MMQ token tile");
    constexpr uint32_t SumItems = (TokenTile / 4) * 2;
    const uint32_t tile_token = blockIdx.y * TokenTile;
    const uint32_t blocks = n_in / 32;
    const uint32_t token_leading = virtual_schedule
        ? virtual_token_base % kTokens : 0u;
    const int32_t compact_token_base = int32_t(tile_token)
        - int32_t(token_leading);

    float suffix[NumericSplit ? SumItems : 1] = {};
    float fixup[NumericSplit && !SingleSeam ? SumItems : 1] = {};
    float partial[SumItems] = {};
    const uint32_t ntx = virtual_schedule
        ? 512u / kTokens : (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = (out_stride + kRows - 1) / kRows;
    const uint32_t ntiles = ntx * nty;
    const uint32_t nwaves = (ntiles + sm_count - 1) / sm_count;
    const uint32_t efficiency = 100 * ntiles / (sm_count * nwaves);
    const uint32_t stream_grid = efficiency >= 90 ? ntiles : sm_count;
    const uint32_t virtual_token_tile = virtual_schedule
        ? ((virtual_token_base / kTokens) + tile_token / kTokens) % ntx
        : tile_token / kTokens;
    const uint32_t logical_tile = ((row_base + tile_row) / kRows) * ntx
        + virtual_token_tile;
    const uint64_t total_work = uint64_t(ntiles) * blocks;
    uint32_t segment_end = blocks;
    bool suffix_phase = true;
    bool single_seam_split = false;
    if constexpr (!NumericSplit) {
        (void)stream_grid;
        (void)logical_tile;
        (void)total_work;
        asm volatile("" : : "r"(int(suffix_phase)));
    }
    while (segment_end > 0) {
        uint32_t segment_begin = 0;
        if constexpr (NumericSplit) {
        if (stream_k_numeric && !no_seam_scan && lane == 0) {
            for (uint32_t bi = 1; bi < stream_grid; ++bi) {
                uint64_t boundary = uint64_t(bi) * total_work / stream_grid;
                boundary -= (boundary % blocks) % 8;
                if (boundary / blocks == logical_tile) {
                    const uint32_t kb = uint32_t(boundary % blocks);
                    if (kb < segment_end) segment_begin = max(segment_begin, kb);
                }
            }
        }
        }
        segment_begin = __shfl_sync(0xffffffff, segment_begin, 0);
        if constexpr (SingleSeam) {
            if (suffix_phase) single_seam_split = segment_begin != 0;
        }
#pragma unroll
        for (uint32_t item = 0; item < SumItems; ++item) partial[item] = 0.0f;

        for (uint32_t stage_block = segment_begin;
             stage_block < segment_end; stage_block += 8) {
            // For the SM86 gate/up laboratory route, start phase-0 activation
            // traffic before the independent packed-Q4 transform. The single
            // shared activation tile prevents overwriting phase 0 early, so
            // phase 1 is issued after phase 0's MMA below.
            if constexpr (AsyncActivation) {
                constexpr uint32_t vectors_per_record =
                    sizeof(BlockQ8_1Mmq) / sizeof(uint4);
                constexpr uint32_t copy_count =
                    TokenTile * vectors_per_record;
                const uint32_t group_index = stage_block / 4;
                uint4 * dst = reinterpret_cast<uint4 *>(sy);
                const uint4 * legal_base =
                    reinterpret_cast<const uint4 *>(x);
                for (uint32_t linear = tid; linear < copy_count;
                     linear += kWarps * 32) {
                    const uint32_t local_token =
                        linear / vectors_per_record;
                    const uint32_t vector =
                        linear % vectors_per_record;
                    const int32_t compact_token =
                        compact_token_base + int32_t(local_token);
                    const bool valid = compact_token >= 0
                        && uint32_t(compact_token) < n_tok;
                    const uint4 * src = valid
                        ? reinterpret_cast<const uint4 *>(
                            x + uint64_t(group_index) * n_tok
                                + uint32_t(compact_token)) + vector
                        : legal_base + vector;
                    copy_global_to_shared_16_zfill(
                        dst + linear, src, valid ? 16u : 0u);
                }
                commit_async_copies();
            }
            // Transform eight Q4 blocks for all 128 rows into signed-int8 tiles.
            constexpr uint32_t packed_count = kRows * 8 * 4;
            for (uint32_t linear = tid; linear < packed_count;
                 linear += kWarps * 32) {
                const uint32_t word = linear % 4;
                const uint32_t qblock = (linear / 4) % 8;
                const uint32_t local_row = linear / (8 * 4);
                const uint32_t row = tile_row + local_row;
                const uint32_t kb = stage_block + qblock;
                uint32_t packed = 0x88888888u;
                float scale = 0.0f;
                if (row < n_out && kb < segment_end) {
                    const uint8_t * block =
                        w + (uint64_t(row) * blocks + kb) * 18;
                    const uint16_t * qs = reinterpret_cast<const uint16_t *>(block + 2);
                    packed = uint32_t(qs[2 * word])
                        | (uint32_t(qs[2 * word + 1]) << 16);
                    if (word == 0) {
                        scale = __half2float(
                            *reinterpret_cast<const __half *>(block));
                    }
                }
                int8_t * dst = sx + local_row * kWeightStride + qblock * 32;
                reinterpret_cast<int *>(dst)[word] =
                    unpack_q4_nibbles(packed);
                reinterpret_cast<int *>(dst + 16)[word] =
                    unpack_q4_nibbles(packed >> 4);
                if (word == 0) {
                    reinterpret_cast<float *>(
                        sx + local_row * kWeightStride + kKStage)[qblock] =
                            scale;
                }
            }
#pragma unroll
            for (uint32_t phase = 0; phase < 2; ++phase) {
                constexpr uint32_t vectors_per_record =
                    sizeof(BlockQ8_1Mmq) / sizeof(uint4);
                constexpr uint32_t copy_count = TokenTile * vectors_per_record;
                const uint32_t group_index = (stage_block + phase * 4) / 4;
                uint4 * dst = reinterpret_cast<uint4 *>(sy);
                if constexpr (AsyncActivation) {
                    if (phase != 0) {
                        const uint4 * legal_base =
                            reinterpret_cast<const uint4 *>(x);
                        for (uint32_t linear = tid; linear < copy_count;
                             linear += kWarps * 32) {
                            const uint32_t local_token =
                                linear / vectors_per_record;
                            const uint32_t vector =
                                linear % vectors_per_record;
                            const int32_t compact_token =
                                compact_token_base + int32_t(local_token);
                            const bool valid = compact_token >= 0
                                && uint32_t(compact_token) < n_tok
                                && stage_block + phase * 4 < segment_end;
                            const uint4 * src = valid
                                ? reinterpret_cast<const uint4 *>(
                                    x + uint64_t(group_index) * n_tok
                                        + uint32_t(compact_token)) + vector
                                : legal_base + vector;
                            copy_global_to_shared_16_zfill(
                                dst + linear, src, valid ? 16u : 0u);
                        }
                        commit_async_copies();
                    }
                    wait_async_copies();
                } else {
                    for (uint32_t linear = tid; linear < copy_count;
                         linear += kWarps * 32) {
                        const uint32_t local_token =
                            linear / vectors_per_record;
                        const uint32_t vector =
                            linear % vectors_per_record;
                        const int32_t compact_token =
                            compact_token_base + int32_t(local_token);
                        const bool valid = compact_token >= 0
                            && uint32_t(compact_token) < n_tok
                            && stage_block + phase * 4 < segment_end;
                        const uint4 * src = valid
                            ? reinterpret_cast<const uint4 *>(
                                x + uint64_t(group_index) * n_tok
                                    + uint32_t(compact_token))
                            : nullptr;
                        dst[linear] =
                            valid ? src[vector] : make_uint4(0, 0, 0, 0);
                    }
                }
                __syncthreads();

#pragma unroll
                for (uint32_t qblock = 0; qblock < 4; ++qblock) {
                    const uint32_t local_kb = phase * 4 + qblock;
                    int af[2][4];
                    float d4[2][2];
#pragma unroll
                    for (uint32_t row_fragment = 0; row_fragment < 2;
                         ++row_fragment) {
                        const uint32_t local_row0 = (warp >> 1) * 32
                            + row_fragment * 16;
                        load_a_m16n8k32(
                            af[row_fragment], sx + local_row0 * kWeightStride
                                + local_kb * 32,
                            kWeightStride);
#pragma unroll
                        for (uint32_t scale_item = 0; scale_item < 2;
                             ++scale_item) {
                            const uint32_t local_row = local_row0
                                + accumulator_row(lane, scale_item * 2);
                            d4[row_fragment][scale_item] =
                                reinterpret_cast<const float *>(
                                    sx + local_row * kWeightStride
                                        + kKStage)[local_kb];
                        }
                    }
#pragma unroll
                    for (uint32_t token_group = 0;
                         token_group < TokenTile / 32; ++token_group) {
#pragma unroll
                        for (uint32_t token_fragment = 0; token_fragment < 2;
                             ++token_fragment) {
                            const uint32_t local_token0 = token_group * 32
                                + (warp & 1) * 16 + token_fragment * 8;
                            int bf[2];
                            load_b_m16n8k32(
                                bf, sy + local_token0 * kActivationStride
                                    + qblock * 32,
                                kActivationStride);
                            float d8[2];
#pragma unroll
                            for (uint32_t scale_item = 0; scale_item < 2;
                                 ++scale_item) {
                                const uint32_t local_token = local_token0
                                    + accumulator_token(lane, scale_item);
                                d8[scale_item] = reinterpret_cast<const float *>(
                                    sy + local_token * kActivationStride
                                        + kActivationStage)[qblock];
                            }

#pragma unroll
                            for (uint32_t row_fragment = 0; row_fragment < 2;
                                 ++row_fragment) {
                                int cf[4] = {};
                                mma_m16n8k32(cf, af[row_fragment], bf);
#pragma unroll
                                for (uint32_t item = 0; item < 4; ++item) {
                                    const uint32_t sum_index =
                                        (((token_group * 2 + token_fragment) * 2
                                            + row_fragment) * 4) + item;
                                    // Keep the pinned llama Q4_0 MMQ source order.
                                    // The compiler owns fast-math fusion for this route.
                                    partial[sum_index] += float(cf[item])
                                        * d4[row_fragment][item / 2]
                                        * d8[item % 2];
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

#pragma unroll
        for (uint32_t item = 0; item < SumItems; ++item) {
            if constexpr (NumericSplit) {
                if (suffix_phase) suffix[item] = partial[item];
                else if constexpr (!SingleSeam) fixup[item] += partial[item];
            }
        }
        if constexpr (NumericSplit) suffix_phase = false;
        segment_end = segment_begin;
    }

#pragma unroll
    for (uint32_t token_group = 0;
                         token_group < TokenTile / 32; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2; ++token_fragment) {
#pragma unroll
            for (uint32_t row_fragment = 0; row_fragment < 2; ++row_fragment) {
#pragma unroll
                for (uint32_t item = 0; item < 4; ++item) {
                    const uint32_t local_row = (warp >> 1) * 32
                        + row_fragment * 16 + accumulator_row(lane, item);
                    const uint32_t local_token = token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8
                        + accumulator_token(lane, item);
                    const uint32_t row = tile_row + local_row;
                    const int32_t compact_token =
                        compact_token_base + int32_t(local_token);
                    if (row >= n_out || compact_token < 0
                        || uint32_t(compact_token) >= n_tok) continue;
                    const uint32_t token = uint32_t(compact_token);
                    const uint32_t sum_index =
                        (((token_group * 2 + token_fragment) * 2
                            + row_fragment) * 4) + item;
                    float result = partial[sum_index];
                    if constexpr (NumericSplit) {
                        if constexpr (SingleSeam) {
                            result = single_seam_split
                                ? suffix[sum_index] + partial[sum_index]
                                : suffix[sum_index];
                        } else {
                            result = suffix[sum_index] + fixup[sum_index];
                        }
                    }
                    float * slot = y + uint64_t(token) * out_stride + row_base + row;
                    if (epilogue) {
                        const float gate = *slot;
                        *slot = cuda_gelu(gate) * result;
                    } else {
                        *slot = result;
                    }
                }
            }
        }
    }
#else
    (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)n_tok;
    (void)epilogue; (void)out_stride; (void)row_base; (void)sm_count;
    (void)stream_k_numeric; (void)virtual_token_base; (void)virtual_schedule;
    (void)no_seam_scan;
#endif
}

__device__ __forceinline__ uint64_t stream_boundary(
        uint32_t index, uint64_t total_work, uint32_t grid,
        uint32_t blocks) {
    uint64_t boundary = uint64_t(index) * total_work / grid;
    boundary -= (boundary % blocks) % 8;
    return boundary;
}

// Boundary-free full-tile route. Model shapes with a partial output row tile
// stay on physical Stream-K below; only the common aligned projection shape
// enters this specialization. Token bounds remain because prefill lengths are
// intentionally arbitrary.
template <bool DirectEpilogue, uint32_t StageBlocks = 8,
          bool QuantizeEpilogue = false, bool StoreEpilogue = true,
          bool NumericSeams = false, bool CooperativeQ4 = false,
          bool SingleActivation = false>
__launch_bounds__(256, StageBlocks == 4 ? 2 : 1)
__global__ void q4_q8_1_full_tile(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        BlockQ8_1Mmq * __restrict__ epilogue_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok, uint32_t work_n_tok,
        uint32_t dst_stride, uint32_t numeric_stream_grid) {
#if __CUDA_ARCH__ >= 800
    static_assert(!QuantizeEpilogue || DirectEpilogue,
                  "only a complete epilogue value may be quantized");
    static_assert(!NumericSeams || (!DirectEpilogue && !QuantizeEpilogue),
                  "numeric seam replay is for plain projections");
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    constexpr uint32_t WeightStride = StageBlocks == 4
        ? kHalfKWeightStride : kWeightStride;
    int8_t * sy = sx + kRows * WeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    float partial[64];

    uint32_t numeric_seam = 0;
    if constexpr (NumericSeams) {
        const uint64_t total_work = uint64_t(gridDim.x) * blocks;
        // Full-tile routing is restricted to at least two logical tiles per
        // physical worker, so a tile can contain at most one Stream-K boundary.
        // Locate that candidate directly instead of making every CTA scan all
        // workers (and execute a 64-bit division for each one).
        uint32_t worker = uint32_t(
            (uint64_t(blockIdx.x) * numeric_stream_grid + gridDim.x - 1)
            / gridDim.x);
        worker = max(worker, 1u);
        if (worker < numeric_stream_grid) {
            const uint64_t boundary = stream_boundary(
                worker, total_work, numeric_stream_grid, blocks);
            if (boundary / blocks == blockIdx.x && boundary % blocks) {
                numeric_seam = uint32_t(boundary % blocks);
            }
        }
    }
    const uint32_t first_end = numeric_seam ? numeric_seam : blocks;
    if constexpr (NumericSeams) {
        if (numeric_seam) {
            compute_segment<kRows, 2, StageBlocks == 8 && !SingleActivation,
                false, true,
                StageBlocks, CooperativeQ4>(w, x, sx, sy, partial, n_out, n_tok,
                    work_n_tok, blocks, tile_row, tile_token, 0, first_end);
        } else {
            compute_segment<kRows, 2, StageBlocks == 8 && !SingleActivation,
                true, true,
                StageBlocks, CooperativeQ4>(w, x, sx, sy, partial, n_out, n_tok,
                    work_n_tok, blocks, tile_row, tile_token, 0, blocks);
        }
    } else {
        compute_segment<kRows, 2, StageBlocks == 8 && !SingleActivation,
            true, true, StageBlocks, CooperativeQ4>(
            w, x, sx, sy, partial, n_out, n_tok, work_n_tok, blocks,
            tile_row, tile_token, 0, blocks);
    }
    if constexpr (NumericSeams) {
        if (numeric_seam) {
#pragma unroll
            for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
#pragma unroll
                    for (uint32_t row_fragment = 0; row_fragment < 2;
                         ++row_fragment) {
#pragma unroll
                        for (uint32_t item = 0; item < 4; ++item) {
                            const uint32_t local_row = (warp >> 1) * 32
                                + row_fragment * 16
                                + accumulator_row(lane, item);
                            const uint32_t local_token = token_group * 32
                                + (warp & 1) * 16 + token_fragment * 8
                                + accumulator_token(lane, item);
                            const uint32_t token = tile_token + local_token;
                            if (token >= n_tok) continue;
                            const uint32_t sum_index =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + item;
                            dst[uint64_t(token) * dst_stride
                                + tile_row + local_row] = partial[sum_index];
                        }
                    }
                }
            }
            __syncthreads();
            compute_segment<kRows, 2, StageBlocks == 8 && !SingleActivation,
                false, true,
                StageBlocks, CooperativeQ4>(w, x, sx, sy, partial, n_out, n_tok,
                    work_n_tok, blocks, tile_row, tile_token,
                    numeric_seam, blocks);
        }
    }
    if constexpr (NumericSeams) {
#pragma unroll
        for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
            for (uint32_t token_fragment = 0; token_fragment < 2;
                 ++token_fragment) {
#pragma unroll
                for (uint32_t row_fragment = 0; row_fragment < 2;
                     ++row_fragment) {
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t local_row = (warp >> 1) * 32
                            + row_fragment * 16
                            + accumulator_row(lane, item);
                        const uint32_t local_token = token_group * 32
                            + (warp & 1) * 16 + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t token = tile_token + local_token;
                        if (token >= n_tok) continue;
                        const uint32_t sum_index =
                            (((token_group * 2 + token_fragment) * 2
                                + row_fragment) * 4) + item;
                        float * slot = dst + uint64_t(token) * dst_stride
                            + tile_row + local_row;
                        *slot = numeric_seam
                            ? partial[sum_index] + *slot : partial[sum_index];
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2;
             ++token_fragment) {
            if constexpr (QuantizeEpilogue) {
#pragma unroll
                for (uint32_t token_item = 0; token_item < 2; ++token_item) {
                    float value[4];
#pragma unroll
                    for (uint32_t row_fragment = 0; row_fragment < 2;
                         ++row_fragment) {
#pragma unroll
                        for (uint32_t row_half = 0; row_half < 2; ++row_half) {
                            const uint32_t item = 2 * row_half + token_item;
                            const uint32_t value_index =
                                2 * row_fragment + row_half;
                            const uint32_t local_row = (warp >> 1) * 32
                                + row_fragment * 16
                                + accumulator_row(lane, item);
                            const uint32_t local_token = token_group * 32
                                + (warp & 1) * 16 + token_fragment * 8
                                + accumulator_token(lane, item);
                            const uint32_t token = tile_token + local_token;
                            const uint32_t sum_index =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + item;
                            float result = 0.0f;
                            if (token < n_tok) {
                                float * slot = dst + uint64_t(token) * dst_stride
                                    + tile_row + local_row;
                                result = cuda_gelu(*slot) * partial[sum_index];
                                if constexpr (StoreEpilogue) *slot = result;
                            }
                            value[value_index] = result;
                        }
                    }

                    float amax = fabsf(value[0]);
                    amax = fmaxf(amax, fabsf(value[1]));
                    amax = fmaxf(amax, fabsf(value[2]));
                    amax = fmaxf(amax, fabsf(value[3]));
#pragma unroll
                    for (int offset = 16; offset >= 4; offset >>= 1) {
                        amax = fmaxf(amax,
                            __shfl_xor_sync(0xffffffff, amax, offset, 32));
                    }
                    const float d_inv = 127.0f / amax;
                    const float d = 1.0f / d_inv;
                    const uint32_t token = tile_token + token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8
                        + (lane & 3) * 2 + token_item;
                    if (token < n_tok) {
                        BlockQ8_1Mmq * out = epilogue_q8
                            + uint64_t(tile_row / kRows) * n_tok + token;
                        const uint32_t block_in_group = warp >> 1;
                        const uint32_t row_in_block = lane >> 2;
                        out->qs[block_in_group * 32 + row_in_block] =
                            int8_t(roundf(value[0] * d_inv));
                        out->qs[block_in_group * 32 + 8 + row_in_block] =
                            int8_t(roundf(value[1] * d_inv));
                        out->qs[block_in_group * 32 + 16 + row_in_block] =
                            int8_t(roundf(value[2] * d_inv));
                        out->qs[block_in_group * 32 + 24 + row_in_block] =
                            int8_t(roundf(value[3] * d_inv));
                        if (lane < 4) {
                            out->d[block_in_group] =
                                __half2float(__float2half(d));
                        }
                    }
                }
                continue;
            }
#pragma unroll
            for (uint32_t row_fragment = 0; row_fragment < 2;
                 ++row_fragment) {
#pragma unroll
                for (uint32_t item = 0; item < 4; ++item) {
                    const uint32_t local_row = (warp >> 1) * 32
                        + row_fragment * 16 + accumulator_row(lane, item);
                    const uint32_t local_token = token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8
                        + accumulator_token(lane, item);
                    const uint32_t token = tile_token + local_token;
                    if (token >= n_tok) continue;
                    const uint32_t sum_index =
                        (((token_group * 2 + token_fragment) * 2
                            + row_fragment) * 4) + item;
                    float * slot = dst + uint64_t(token) * dst_stride
                        + tile_row + local_row;
                    if constexpr (DirectEpilogue) {
                        *slot = cuda_gelu(*slot) * partial[sum_index];
                    } else {
                        *slot = partial[sum_index];
                    }
                }
            }
        }
    }
#else
    (void)w; (void)x; (void)dst; (void)epilogue_q8;
    (void)n_in; (void)n_out; (void)n_tok;
    (void)work_n_tok; (void)dst_stride; (void)numeric_stream_grid;
#endif
}

// Compact row tile for reference-full-grid projections. K traversal and every
// per-output FMA remain identical to the 128-row kernel; only independent output
// rows are split across more CTAs. Its smaller register result set and shared tile
// allow two resident CTAs on SM86. Physical Stream-K never enters this variant,
// because changing that grid would also change the reference reduction seams.
template <bool DirectEpilogue, bool QuantizeEpilogue = false>
__launch_bounds__(256, 2)
__global__ void q4_q8_1_full_tile_r64(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        BlockQ8_1Mmq * __restrict__ epilogue_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok, uint32_t work_n_tok,
        uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kCompactRows * kWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kCompactRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    float partial[32];

    compute_segment<kCompactRows, 1, false, true, true>(
        w, x, sx, sy, partial, n_out, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks);
#pragma unroll
    for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2;
             ++token_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = (warp >> 1) * 16
                    + accumulator_row(lane, item);
                const uint32_t local_token = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8
                    + accumulator_token(lane, item);
                const uint32_t token = tile_token + local_token;
                if (token >= n_tok) continue;
                const uint32_t sum_index =
                    ((token_group * 2 + token_fragment) * 4) + item;
                float * slot = dst + uint64_t(token) * dst_stride
                    + tile_row + local_row;
                if constexpr (DirectEpilogue) {
                    *slot = cuda_gelu(*slot) * partial[sum_index];
                } else {
                    *slot = partial[sum_index];
                }
            }
        }
    }
    if constexpr (QuantizeEpilogue) {
        static_assert(DirectEpilogue,
            "only a completed gate/up value may be quantized");
        __syncthreads();
        for (uint32_t record_linear = warp; record_linear < 2 * kTokens;
             record_linear += kWarps) {
            const uint32_t local_token = record_linear / 2;
            const uint32_t local_block = record_linear % 2;
            const uint32_t token = tile_token + local_token;
            float value = 0.0f;
            if (token < n_tok) {
                value = dst[uint64_t(token) * dst_stride + tile_row
                    + local_block * 32 + lane];
            }
            float amax = fabsf(value);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax,
                    __shfl_xor_sync(0xffffffff, amax, offset, 32));
            }
            const float d_inv = 127.0f / amax;
            const float d = 1.0f / d_inv;
            if (token < n_tok) {
                BlockQ8_1Mmq * out = epilogue_q8
                    + uint64_t(tile_row / kRows) * n_tok + token;
                const uint32_t block_in_record = (tile_row % kRows) / 32
                    + local_block;
                out->qs[block_in_record * 32 + lane] =
                    int8_t(roundf(value * d_inv));
                if (lane == 0) {
                    out->d[block_in_record] = __half2float(__float2half(d));
                }
            }
        }
    }
#else
    (void)w; (void)x; (void)dst; (void)epilogue_q8;
    (void)n_in; (void)n_out; (void)n_tok;
    (void)work_n_tok; (void)dst_stride;
#endif
}

// Laboratory J256 companion for the dominant SM86 gate/up shapes. A CTA keeps
// the proven 64-row ownership but consumes two adjacent J128 token tiles, which
// halves repeated Q4 weight staging. The larger accumulator/shared footprint is
// intentionally isolated until exact-shape end-to-end evidence prices the
// reduced weight traffic against its one-CTA-per-SM occupancy.
template <bool DirectEpilogue>
__launch_bounds__(256, 1)
__global__ void q4_q8_1_full_tile_r64_j256(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kCompactRows * kWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kWideTokens - 1) / kWideTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kCompactRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kWideTokens;
    float partial[kWideTokens / 4];

    compute_segment<kCompactRows, 1, false, true, true, 8, false,
                    kWideTokens>(
        w, x, sx, sy, partial, n_out, n_tok, n_tok, blocks,
        tile_row, tile_token, 0, blocks);
#pragma unroll
    for (uint32_t token_group = 0;
         token_group < kWideTokens / 32; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2;
             ++token_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = (warp >> 1) * 16
                    + accumulator_row(lane, item);
                const uint32_t local_token = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8
                    + accumulator_token(lane, item);
                const uint32_t token = tile_token + local_token;
                if (token >= n_tok) continue;
                const uint32_t sum_index =
                    ((token_group * 2 + token_fragment) * 4) + item;
                float * slot = dst + uint64_t(token) * dst_stride
                    + tile_row + local_row;
                if constexpr (DirectEpilogue) {
                    *slot = cuda_gelu(*slot) * partial[sum_index];
                } else {
                    *slot = partial[sum_index];
                }
            }
        }
    }
#else
    (void) w; (void) x; (void) dst; (void) n_in; (void) n_out;
    (void) n_tok; (void) dst_stride;
#endif
}

// Physical Stream-K: each resident CTA owns a contiguous range in flattened
// tile/K space. Complete (or suffix) tiles go directly to the destination; a
// final prefix is written to the bounded fixup area owned by this CTA.
template <bool DirectEpilogue, bool FullRows>
__launch_bounds__(256, 1)
__global__ void q4_q8_1_stream(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        float * __restrict__ fixup, uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = (n_out + kRows - 1) / kRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    uint64_t work = stream_boundary(blockIdx.x, total_work, gridDim.x, blocks);
    const uint64_t work_stop = stream_boundary(
        blockIdx.x + 1, total_work, gridDim.x, blocks);
    float partial[64];

    while (work < work_stop) {
        const uint32_t logical_tile = uint32_t(work / blocks);
        const uint32_t segment_begin = uint32_t(work % blocks);
        const uint64_t tile_stop = uint64_t(logical_tile + 1) * blocks;
        const uint64_t segment_work_stop = min(work_stop, tile_stop);
        const uint32_t segment_end = uint32_t(segment_work_stop
            - uint64_t(logical_tile) * blocks);
        const uint32_t tile_row = (logical_tile / ntx) * kRows;
        const uint32_t tile_token = (logical_tile % ntx) * kTokens;

        compute_segment<kRows, 2, true, false, FullRows>(
            w, x, sx, sy, partial, n_out, n_tok, work_n_tok, blocks,
            tile_row, tile_token, segment_begin, segment_end);
        const bool direct = segment_end == blocks;
#pragma unroll
        for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
            for (uint32_t token_fragment = 0; token_fragment < 2;
                 ++token_fragment) {
#pragma unroll
                for (uint32_t row_fragment = 0; row_fragment < 2;
                     ++row_fragment) {
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t local_row = (warp >> 1) * 32
                            + row_fragment * 16 + accumulator_row(lane, item);
                        const uint32_t local_token = token_group * 32
                            + (warp & 1) * 16 + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t row = tile_row + local_row;
                        const uint32_t token = tile_token + local_token;
                        if ((!FullRows && row >= n_out) || token >= n_tok) continue;
                        const uint32_t sum_index =
                            (((token_group * 2 + token_fragment) * 2
                                + row_fragment) * 4) + item;
                        if (direct) {
                            float * slot = dst + uint64_t(token) * dst_stride + row;
                            if constexpr (DirectEpilogue) {
                                *slot = cuda_gelu(*slot) * partial[sum_index];
                            } else {
                                *slot = partial[sum_index];
                            }
                        } else {
                            fixup[(uint64_t(blockIdx.x) * kTokens + local_token)
                                * kRows + local_row] = partial[sum_index];
                        }
                    }
                }
            }
        }
        __syncthreads();
        work = segment_work_stop;
    }
#else
    (void)w; (void)x; (void)dst; (void)fixup; (void)n_in; (void)n_out;
    (void)n_tok; (void)work_n_tok; (void)dst_stride;
#endif
}

// Preserve each 128-row physical Stream-K worker and its exact K seam, but split
// the independent output rows across a paired 64-row CTA. This cuts the register
// accumulator set in half and admits two resident CTAs on SM86. Both CTAs use the
// same logical-worker fixup slot and write disjoint row halves.
__launch_bounds__(256, 2)
__global__ void q4_q8_1_stream_r64(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        float * __restrict__ fixup, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kCompactRows * kWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t worker = blockIdx.x / 2;
    const uint32_t row_half = blockIdx.x & 1;
    const uint32_t workers = gridDim.x / 2;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = (n_out + kRows - 1) / kRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    uint64_t work = stream_boundary(worker, total_work, workers, blocks);
    const uint64_t work_stop = stream_boundary(
        worker + 1, total_work, workers, blocks);
    float partial[32];

    while (work < work_stop) {
        const uint32_t logical_tile = uint32_t(work / blocks);
        const uint32_t segment_begin = uint32_t(work % blocks);
        const uint64_t tile_stop = uint64_t(logical_tile + 1) * blocks;
        const uint64_t segment_work_stop = min(work_stop, tile_stop);
        const uint32_t segment_end = uint32_t(segment_work_stop
            - uint64_t(logical_tile) * blocks);
        const uint32_t tile_row = (logical_tile / ntx) * kRows
            + row_half * kCompactRows;
        const uint32_t tile_token = (logical_tile % ntx) * kTokens;

        compute_segment<kCompactRows, 1, false, false, true>(
            w, x, sx, sy, partial, n_out, n_tok, work_n_tok, blocks,
            tile_row, tile_token, segment_begin, segment_end);
        const bool direct = segment_end == blocks;
#pragma unroll
        for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
            for (uint32_t token_fragment = 0; token_fragment < 2;
                 ++token_fragment) {
#pragma unroll
                for (uint32_t item = 0; item < 4; ++item) {
                    const uint32_t local_row = (warp >> 1) * 16
                        + accumulator_row(lane, item);
                    const uint32_t local_token = token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8
                        + accumulator_token(lane, item);
                    const uint32_t row = tile_row + local_row;
                    const uint32_t token = tile_token + local_token;
                    if (token >= n_tok) continue;
                    const uint32_t sum_index =
                        ((token_group * 2 + token_fragment) * 4) + item;
                    if (direct) {
                        dst[uint64_t(token) * dst_stride + row] =
                            partial[sum_index];
                    } else {
                        const uint32_t worker_row =
                            row_half * kCompactRows + local_row;
                        fixup[(uint64_t(worker) * kTokens + local_token)
                            * kRows + worker_row] = partial[sum_index];
                    }
                }
            }
        }
        __syncthreads();
        work = segment_work_stop;
    }
#else
    (void)w; (void)x; (void)dst; (void)fixup; (void)n_in; (void)n_out;
    (void)n_tok; (void)work_n_tok; (void)dst_stride;
#endif
}

// Combine prefix segments in the same nearest-to-farthest order used by the
// pinned llama Stream-K fixup. The suffix already resides in dst.
__global__ void q4_q8_1_stream_fixup(
        float * dst, const float * fixup, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride) {
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = (n_out + kRows - 1) / kRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    const uint64_t work = stream_boundary(
        blockIdx.x, total_work, gridDim.x, blocks);
    const uint64_t work_stop = stream_boundary(
        blockIdx.x + 1, total_work, gridDim.x, blocks);
    const bool did_not_write_last = work / blocks == work_stop / blocks
        && work_stop % blocks != 0;
    if (work == work_stop || work % blocks == 0 || did_not_write_last) return;

    const uint32_t logical_tile = uint32_t(work / blocks);
    const uint32_t tile_row = (logical_tile / ntx) * kRows;
    const uint32_t tile_token = (logical_tile % ntx) * kTokens;
    const uint32_t local_row = threadIdx.x;
    if (local_row >= kRows || tile_row + local_row >= n_out) return;

    for (uint32_t local_token = blockIdx.y * 32;
         local_token < min(kTokens, (blockIdx.y + 1) * 32u);
         ++local_token) {
        if (tile_token + local_token >= n_tok) break;
        float sum = 0.0f;
        int previous = int(blockIdx.x) - 1;
        uint64_t previous_stop = work;
        while (previous >= 0) {
            const uint64_t previous_work = stream_boundary(
                uint32_t(previous), total_work, gridDim.x, blocks);
            if (previous_work == previous_stop) {
                --previous;
                previous_stop = previous_work;
                continue;
            }
            sum += fixup[(uint64_t(previous) * kTokens + local_token)
                * kRows + local_row];
            if (previous_work % blocks == 0
                || previous_work / blocks < logical_tile) break;
            --previous;
            previous_stop = previous_work;
        }
        dst[uint64_t(tile_token + local_token) * dst_stride
            + tile_row + local_row] += sum;
    }
}

__global__ void q4_q8_1_stream_epilogue(
        const float * projection, float * y, uint32_t n_out,
        uint32_t n_tok, uint32_t out_stride, uint32_t row_base) {
    const uint64_t linear = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = uint64_t(n_out) * n_tok;
    if (linear >= count) return;
    const uint32_t token = uint32_t(linear / n_out);
    const uint32_t row = uint32_t(linear % n_out);
    float * slot = y + uint64_t(token) * out_stride + row_base + row;
    *slot = cuda_gelu(*slot) * projection[linear];
}

inline uint64_t stream_workspace_bytes(
        uint32_t n_out, uint32_t n_tok, uint32_t sm_count,
        uint32_t epilogue) {
    const uint64_t output = epilogue ? uint64_t(n_out) * n_tok : 0;
    return (output + uint64_t(sm_count) * kRows * kTokens) * sizeof(float);
}

inline bool launch(const uint8_t * w, const BlockQ8_1Mmq * x, float * y,
                   uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                   uint32_t epilogue, uint32_t out_stride,
                   uint32_t row_base, uint32_t sm_count,
                   uint32_t stream_k_numeric, uint32_t virtual_token_base,
                   uint32_t virtual_schedule, uint32_t llama_compat_requested,
                   uint32_t canonical_full_tile_requested,
                   uint32_t virtual_no_seam_requested,
                   uint32_t virtual_direct_seam_requested,
                   uint32_t virtual_async_activation_requested,
                   uint32_t exact128_token64_requested,
                   FullTileVariant full_tile_variant,
                   uint32_t full_tile_min_efficiency,
                   float * workspace, BlockQ8_1Mmq * epilogue_q8,
                   cudaStream_t stream, LaunchInfo * info = nullptr) {
    if (info) *info = {};
    if (n_in % 128) return false;
    static const bool configured = [] {
        const cudaError_t fast = cudaFuncSetAttribute(
            q4_q8_1_mma_128<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t split = cudaFuncSetAttribute(
            q4_q8_1_mma_128<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t single_seam_split = cudaFuncSetAttribute(
            q4_q8_1_mma_128<true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
        const cudaError_t async_no_seam = cudaFuncSetAttribute(
            q4_q8_1_mma_128<true, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
        const cudaError_t stream = cudaFuncSetAttribute(
            q4_q8_1_stream<false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t stream_epilogue = cudaFuncSetAttribute(
            q4_q8_1_stream<true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t stream_full_rows = cudaFuncSetAttribute(
            q4_q8_1_stream<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t stream_compact_rows = cudaFuncSetAttribute(
            q4_q8_1_stream_r64,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kCompactSharedBytes));
        const cudaError_t full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t full_tile_epilogue = cudaFuncSetAttribute(
            q4_q8_1_full_tile<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        const cudaError_t single_activation_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false, 8, false, true, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSingleActivationSharedBytes));
        const cudaError_t single_activation_full_tile_epilogue =
            cudaFuncSetAttribute(
                q4_q8_1_full_tile<true, 8, false, true, false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(kSingleActivationSharedBytes));
        const cudaError_t half_k_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false, 4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        const cudaError_t half_k_numeric_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false, 4, false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        const cudaError_t coop_half_k_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false, 4, false, true, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        const cudaError_t coop_half_k_numeric_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false, 4, false, true, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        const cudaError_t full_k_numeric_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile<false, 8, false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));

        const cudaError_t half_k_full_tile_epilogue = cudaFuncSetAttribute(
            q4_q8_1_full_tile<true, 4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        const cudaError_t half_k_full_tile_epilogue_q8 = cudaFuncSetAttribute(
            q4_q8_1_full_tile<true, 4, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kHalfKSharedBytes));
        const cudaError_t coop_half_k_full_tile_epilogue_q8 =
            cudaFuncSetAttribute(
                q4_q8_1_full_tile<true, 4, true, true, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(kHalfKSharedBytes));
        const cudaError_t half_k_full_tile_epilogue_q8_only =
            cudaFuncSetAttribute(
                q4_q8_1_full_tile<true, 4, true, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(kHalfKSharedBytes));
        const cudaError_t compact_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile_r64<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kCompactSharedBytes));
        const cudaError_t compact_full_tile_epilogue = cudaFuncSetAttribute(
            q4_q8_1_full_tile_r64<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kCompactSharedBytes));
        const cudaError_t compact_full_tile_epilogue_q8 = cudaFuncSetAttribute(
            q4_q8_1_full_tile_r64<true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kCompactSharedBytes));
        const cudaError_t compact_j256_full_tile = cudaFuncSetAttribute(
            q4_q8_1_full_tile_r64_j256<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kWideCompactSharedBytes));
        const cudaError_t compact_j256_full_tile_epilogue =
            cudaFuncSetAttribute(
                q4_q8_1_full_tile_r64_j256<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(kWideCompactSharedBytes));
        if (fast != cudaSuccess || split != cudaSuccess
            || single_seam_split != cudaSuccess
            || async_no_seam != cudaSuccess
            || stream != cudaSuccess
            || stream_epilogue != cudaSuccess || stream_full_rows != cudaSuccess
            || stream_compact_rows != cudaSuccess
            || full_tile != cudaSuccess
            || full_tile_epilogue != cudaSuccess
            || single_activation_full_tile != cudaSuccess
            || single_activation_full_tile_epilogue != cudaSuccess
            || half_k_full_tile != cudaSuccess
            || half_k_numeric_full_tile != cudaSuccess
            || coop_half_k_full_tile != cudaSuccess
            || coop_half_k_numeric_full_tile != cudaSuccess
            || full_k_numeric_full_tile != cudaSuccess

            || half_k_full_tile_epilogue != cudaSuccess
            || half_k_full_tile_epilogue_q8 != cudaSuccess
            || coop_half_k_full_tile_epilogue_q8 != cudaSuccess
            || half_k_full_tile_epilogue_q8_only != cudaSuccess
            || compact_full_tile != cudaSuccess
            || compact_full_tile_epilogue != cudaSuccess
            || compact_full_tile_epilogue_q8 != cudaSuccess
            || compact_j256_full_tile != cudaSuccess
            || compact_j256_full_tile_epilogue != cudaSuccess) {
            cudaGetLastError();
        }
        return fast == cudaSuccess && split == cudaSuccess
            && single_seam_split == cudaSuccess
            && async_no_seam == cudaSuccess
            && stream == cudaSuccess
            && stream_epilogue == cudaSuccess && stream_full_rows == cudaSuccess
            && stream_compact_rows == cudaSuccess
            && full_tile == cudaSuccess
            && full_tile_epilogue == cudaSuccess
            && single_activation_full_tile == cudaSuccess
            && single_activation_full_tile_epilogue == cudaSuccess
            && half_k_full_tile == cudaSuccess
            && half_k_numeric_full_tile == cudaSuccess
            && coop_half_k_full_tile == cudaSuccess
            && coop_half_k_numeric_full_tile == cudaSuccess
            && full_k_numeric_full_tile == cudaSuccess

            && half_k_full_tile_epilogue == cudaSuccess
            && half_k_full_tile_epilogue_q8 == cudaSuccess
            && coop_half_k_full_tile_epilogue_q8 == cudaSuccess
            && half_k_full_tile_epilogue_q8_only == cudaSuccess
            && compact_full_tile == cudaSuccess
            && compact_full_tile_epilogue == cudaSuccess
            && compact_full_tile_epilogue_q8 == cudaSuccess
            && compact_j256_full_tile == cudaSuccess
            && compact_j256_full_tile_epilogue == cudaSuccess;
    }();
    if (!configured) return false;
    static const bool trim_tail =
        std::getenv("IMPARO_CUDA_NO_MMQ_TAIL_TRIM") == nullptr;
    const uint32_t work_n_tok = trim_tail ? n_tok
        : ((n_tok + kTokens - 1) / kTokens) * kTokens;
    const dim3 grid((n_out + kRows - 1) / kRows,
                    (n_tok + kTokens - 1) / kTokens);
    const uint32_t virtual_leading = virtual_schedule
        ? virtual_token_base % kTokens : 0u;
    const dim3 virtual_grid(grid.x,
        (virtual_leading + n_tok + kTokens - 1) / kTokens);
    const uint32_t ntiles = grid.x * grid.y;
    const uint32_t nwaves = (ntiles + sm_count - 1) / sm_count;
    const uint32_t efficiency = 100 * ntiles / (sm_count * nwaves);
    const bool canonical_full_tile = canonical_full_tile_requested
        && n_tok == 512 && virtual_token_base % 512 == 0;
    const bool llama_compat = llama_compat_requested && !canonical_full_tile;
    const uint32_t route_efficiency = llama_compat
        ? 90u : full_tile_min_efficiency;
    if (info) {
        info->tile_rows = kRows;
        info->tile_tokens = kTokens;
        info->logical_tiles = ntiles;
        info->efficiency = efficiency;
    }
    if (stream_k_numeric && virtual_schedule) {
        const uint32_t virtual_ntiles =
            ((out_stride + kRows - 1) / kRows) * (512u / kTokens);
        const uint32_t virtual_waves =
            (virtual_ntiles + sm_count - 1) / sm_count;
        const uint32_t virtual_efficiency =
            100u * virtual_ntiles / (sm_count * virtual_waves);
        const uint32_t virtual_blocks = virtual_efficiency >= 90u
            ? virtual_ntiles : sm_count;
        const uint32_t qblocks = n_in / 32;
        // A boundary is rounded down by at most seven Q4 blocks.  When adjacent
        // rounded boundaries remain more than one tile apart, a logical tile can
        // contain at most one numeric seam.  That lets the specialized replay keep
        // only suffix + partial instead of suffix + fixup + partial, avoiding the
        // 255-register/local-stack cliff without changing reduction order.
        const uint64_t boundary_stride =
            uint64_t(virtual_ntiles) * qblocks / virtual_blocks;
        static const bool single_seam_enabled =
            std::getenv("IMPARO_CUDA_MMQ_LEGACY_VIRTUAL") == nullptr;
        const bool single_seam = single_seam_enabled
            && (virtual_blocks == virtual_ntiles
                || boundary_stride >= uint64_t(qblocks) + 8u);
        // Laboratory-only occupancy candidate for exact 128-token SM86
        // projections. Two 64-token CTAs retain one parent J128 logical-tile
        // identity, so both halves replay the same canonical Stream-K seam and
        // preserve every token's K/FMA order. The split raises the local-Q and
        // FFN-down grids from 16/20 to 32/40 blocks while halving each thread's
        // result state. Promotion belongs to the tuner/receipt path; this selector
        // is deliberately exact-shape and default-off. The caller supplies either
        // the versioned tuner decision or the explicit laboratory override.
        const bool exact128_token64 = exact128_token64_requested
            && single_seam && full_tile_variant == FullTileVariant::Rows128K128
            && virtual_leading == 0 && virtual_token_base % 512 == 0
            && n_tok == 128 && !epilogue && row_base == 0
            && out_stride == n_out && n_out % kRows == 0
            && ((n_in == 2560 && n_out == 2048)
                || (n_in == 10240 && n_out == 2560));
        // When every virtual logical tile owns one physical CTA there is no
        // cross-CTA K boundary to replay. Keep the established NumericSplit and
        // SingleSeam arithmetic instance, but bypass its otherwise redundant
        // host-derived boundary scan at runtime. This preserves the exact floating
        // point code path while removing O(physical_blocks) integer work per CTA.
        const bool virtual_no_seam = virtual_no_seam_requested
            && virtual_blocks == virtual_ntiles;
        // Admitted SM86 occupancy route. Splitting independent output
        // rows into 64-row CTAs preserves the complete per-output K traversal
        // and FMA order while halving the accumulator set and shared weight
        // tile. Restrict it to aligned no-seam virtual batches so neither the
        // reference Stream-K ownership nor token coordinates can change.
        static const bool virtual_no_seam_r64_requested =
            std::getenv("IMPARO_CUDA_NO_MMQ_VIRTUAL_R64") == nullptr;
        const bool virtual_no_seam_r64 =
            virtual_no_seam_r64_requested && virtual_no_seam
            && virtual_leading == 0 && virtual_token_base % 512 == 0
            // The 64-row geometry wins on the dominant wide gate/up pair but
            // regresses the smaller projections through duplicated activation
            // staging. Keep the laboratory selector shape-scoped until a tuner
            // can carry per-shape receipts.
            && n_in == 2560 && n_out == 10240
            && n_out % kCompactRows == 0;
        static const bool virtual_no_seam_r64_j256_requested =
            std::getenv("IMPARO_CUDA_MMQ_GATE_R64_J256_LAB") != nullptr;
        const bool virtual_no_seam_r64_j256 =
            virtual_no_seam_r64_j256_requested && virtual_no_seam_r64
            && n_tok > kTokens && n_tok <= 2 * kWideTokens
            && row_base == 0 && out_stride == n_out;
        // Price llama's successful SM86 architecture directly: one K128
        // activation stage in shared memory, with a complete per-output K walk.
        // This remains opt-in until whole-model numerical and performance gates
        // accept it; the admitted 64-row route is the immediate rollback.
        static const bool virtual_single_activation_r128_requested =
            std::getenv("IMPARO_CUDA_MMQ_VIRTUAL_SINGLE_ACTIVATION_R128_LAB")
                != nullptr;
        const bool virtual_single_activation_r128 =
            virtual_single_activation_r128_requested && virtual_no_seam
            && virtual_leading == 0 && virtual_token_base % 512 == 0
            && n_in == 2560 && n_out == 10240
            && n_out % kRows == 0;
        // Laboratory-only occupancy experiment for the dominant down
        // projection. The established 128-row route replays a reference
        // Stream-K seam at this grid size and therefore carries the large
        // accumulator set at one CTA/SM. This candidate gives each 64-row
        // tile the complete K range, preserving the per-output K/FMA order but
        // intentionally changing the cross-seam grouping. Keep the selector
        // exact and opt-in until both whole-model numerical gates and an
        // end-to-end A/B receipt accept that different numeric route.
        static const bool virtual_full_k_r64_lab_requested =
            std::getenv("IMPARO_CUDA_MMQ_DOWN_FULL_K_R64_LAB") != nullptr;
        const bool virtual_full_k_r64_lab =
            virtual_full_k_r64_lab_requested
            && virtual_leading == 0 && virtual_token_base % 512 == 0
            && n_in == 10240 && n_out == 2560 && n_tok == 449
            && !epilogue && row_base == 0 && out_stride == n_out
            && n_out % kCompactRows == 0;
        // Preserve the established 128-row Stream-K worker boundaries while
        // splitting each worker into two disjoint 64-row CTAs. Unlike the
        // full-K laboratory route above, both row halves replay the exact same
        // reference seam and share the existing fixup contract.
        static const bool virtual_stream_r64_lab_requested =
            std::getenv("IMPARO_CUDA_MMQ_DOWN_STREAM_R64_LAB") != nullptr;
        const bool virtual_stream_r64_lab =
            virtual_stream_r64_lab_requested && workspace
            && virtual_leading == 0 && virtual_token_base % 512 == 0
            && n_in == 10240 && n_out == 2560 && n_tok == 449
            && !epilogue && row_base == 0 && out_stride == n_out
            && n_out % kRows == 0 && virtual_blocks < virtual_ntiles;
        const bool virtual_async_activation =
            virtual_async_activation_requested && virtual_no_seam
            && virtual_leading == 0 && n_in == 2560 && n_out == 10240
            // Full 512-token tiles showed no stable event win. Compact tails
            // benefit because cp.async performs their mandatory zero fill
            // without the old global-load/register/shared-store round trip.
            && n_tok > 0 && n_tok < 512;
        // An aligned compact batch has the same token-tile coordinates as its
        // 512-token virtual schedule.  For sparse virtual grids with at most one
        // reference seam per tile, reuse the cp.async/double-buffered full-tile
        // primitive and locate that seam directly.  This changes neither the
        // reference Stream-K boundary nor the suffix + prefix addition order;
        // unaligned tails stay on the established compact-token mapper above.
        const bool virtual_direct_seam = virtual_direct_seam_requested
            && single_seam && virtual_blocks < virtual_ntiles
            && virtual_leading == 0 && grid.x * grid.y == virtual_ntiles
            && n_out % kRows == 0 && !epilogue;
        if (info) {
            info->route = exact128_token64
                ? LaunchRoute::Exact128Token64SingleSeamLab
                : (virtual_single_activation_r128
                ? LaunchRoute::VirtualSingleActivationRows128Lab
                : (virtual_no_seam_r64_j256
                ? LaunchRoute::VirtualNoSeamRows64Tokens256Lab
                : (virtual_no_seam_r64
                ? LaunchRoute::VirtualNoSeamRows64
                : (virtual_full_k_r64_lab
                    ? LaunchRoute::VirtualFullKRows64Lab
                    : (virtual_stream_r64_lab
                        ? LaunchRoute::VirtualStreamRows64Lab
                        : (virtual_async_activation
                    ? LaunchRoute::VirtualNoSeamAsyncActivation
                    : (virtual_no_seam ? LaunchRoute::VirtualNoSeam
                    : (virtual_direct_seam ? LaunchRoute::VirtualDirectSeam
                                           : LaunchRoute::VirtualStreamK))))))));
            if (exact128_token64) {
                info->tile_rows = kRows;
                info->tile_tokens = kHalfTokens;
                info->logical_tiles = grid.x * 2;
                info->physical_blocks = info->logical_tiles;
                const uint32_t exact_waves =
                    (info->logical_tiles + sm_count - 1) / sm_count;
                info->efficiency = 100u * info->logical_tiles
                    / (sm_count * exact_waves);
            } else if (virtual_single_activation_r128) {
                info->tile_rows = kRows;
                info->logical_tiles = ntiles;
                info->physical_blocks = ntiles;
                info->efficiency = efficiency;
            } else if (virtual_no_seam_r64_j256) {
                info->tile_rows = kCompactRows;
                info->logical_tiles =
                    ((n_out + kCompactRows - 1) / kCompactRows)
                    * ((n_tok + kWideTokens - 1) / kWideTokens);
                info->physical_blocks = info->logical_tiles;
                const uint32_t compact_waves =
                    (info->logical_tiles + sm_count - 1) / sm_count;
                info->efficiency = 100u * info->logical_tiles
                    / (sm_count * compact_waves);
            } else if (virtual_no_seam_r64 || virtual_full_k_r64_lab) {
                info->tile_rows = kCompactRows;
                info->logical_tiles =
                    ((n_out + kCompactRows - 1) / kCompactRows) * grid.y;
                info->physical_blocks = info->logical_tiles;
                const uint32_t compact_waves =
                    (info->logical_tiles + sm_count - 1) / sm_count;
                info->efficiency = 100u * info->logical_tiles
                    / (sm_count * compact_waves);
            } else if (virtual_stream_r64_lab) {
                info->tile_rows = kCompactRows;
                info->logical_tiles = virtual_ntiles * 2;
                info->physical_blocks = virtual_blocks * 2;
                info->efficiency = virtual_efficiency;
            } else {
                info->logical_tiles = virtual_ntiles;
                info->physical_blocks = virtual_blocks;
                info->efficiency = virtual_efficiency;
            }
        }
        if (exact128_token64) {
            const dim3 exact_grid(grid.x, 2);
            q4_q8_1_mma_128<true, true, false, kHalfTokens>
                <<<exact_grid, dim3(32, kWarps),
                    kHalfTokenSharedBytes, stream>>>(
                        w, x, y, n_in, n_out, n_tok, 0u, out_stride,
                        row_base, sm_count, 1u, virtual_token_base, 1u, 0u);
        } else if (virtual_single_activation_r128) {
            if (epilogue) {
                q4_q8_1_full_tile<true, 8, false, true, false, false, true>
                    <<<ntiles, dim3(32, kWarps),
                        kSingleActivationSharedBytes, stream>>>(
                            w, x, y + row_base, nullptr, n_in, n_out, n_tok,
                            n_tok, out_stride, 0);
            } else {
                q4_q8_1_full_tile<false, 8, false, true, false, false, true>
                    <<<ntiles, dim3(32, kWarps),
                        kSingleActivationSharedBytes, stream>>>(
                            w, x, y + row_base, nullptr, n_in, n_out, n_tok,
                            n_tok, out_stride, 0);
            }
        } else if (virtual_no_seam_r64_j256) {
            const uint32_t compact_tiles =
                ((n_out + kCompactRows - 1) / kCompactRows)
                * ((n_tok + kWideTokens - 1) / kWideTokens);
            if (epilogue) {
                q4_q8_1_full_tile_r64_j256<true><<<compact_tiles,
                    dim3(32, kWarps), kWideCompactSharedBytes, stream>>>(
                        w, x, y + row_base, n_in, n_out, n_tok, out_stride);
            } else {
                q4_q8_1_full_tile_r64_j256<false><<<compact_tiles,
                    dim3(32, kWarps), kWideCompactSharedBytes, stream>>>(
                        w, x, y + row_base, n_in, n_out, n_tok, out_stride);
            }
        } else if (virtual_stream_r64_lab) {
            q4_q8_1_stream_r64<<<virtual_blocks * 2,
                dim3(32, kWarps), kCompactSharedBytes, stream>>>(
                    w, x, y + row_base, workspace, n_in, n_out, n_tok, n_tok,
                    out_stride);
            if (virtual_ntiles % virtual_blocks != 0) {
                q4_q8_1_stream_fixup<<<dim3(virtual_blocks, 4), 128, 0,
                    stream>>>(y + row_base, workspace, n_in, n_out, n_tok,
                              out_stride);
            }
        } else if (virtual_no_seam_r64 || virtual_full_k_r64_lab) {
            const uint32_t compact_tiles =
                ((n_out + kCompactRows - 1) / kCompactRows) * grid.y;
            if (epilogue) {
                static const bool fused_q8_lab =
                    std::getenv("IMPARO_CUDA_MMQ_R64_FUSED_Q8_LAB") != nullptr;
                if (fused_q8_lab && epilogue_q8) {
                    q4_q8_1_full_tile_r64<true, true><<<compact_tiles,
                        dim3(32, kWarps), kCompactSharedBytes, stream>>>(
                            w, x, y + row_base, epilogue_q8, n_in, n_out,
                            n_tok, n_tok, out_stride);
                    if (info) info->fused_q8 = true;
                } else {
                    q4_q8_1_full_tile_r64<true><<<compact_tiles,
                        dim3(32, kWarps), kCompactSharedBytes, stream>>>(
                            w, x, y + row_base, nullptr, n_in, n_out, n_tok,
                            n_tok, out_stride);
                }
            } else {
                q4_q8_1_full_tile_r64<false><<<compact_tiles,
                    dim3(32, kWarps), kCompactSharedBytes, stream>>>(
                        w, x, y + row_base, nullptr, n_in, n_out, n_tok, n_tok,
                        out_stride);
            }
        } else if (virtual_direct_seam) {
            q4_q8_1_full_tile<false, 8, false, true, true>
                <<<virtual_ntiles, dim3(32, kWarps), kSharedBytes, stream>>>(
                    w, x, y + row_base, nullptr, n_in, n_out, n_tok, n_tok,
                    out_stride, virtual_blocks);
        } else if (virtual_async_activation) {
            q4_q8_1_mma_128<true, true, true><<<virtual_grid,
                dim3(32, kWarps), kSharedBytes, stream>>>(
                    w, x, y, n_in, n_out, n_tok, epilogue, out_stride,
                    row_base, sm_count, 1u, virtual_token_base, 1u, 1u);
        } else if (virtual_no_seam) {
            q4_q8_1_mma_128<true, true><<<virtual_grid,
                dim3(32, kWarps), kSharedBytes, stream>>>(
                    w, x, y, n_in, n_out, n_tok, epilogue, out_stride,
                    row_base, sm_count, 1u, virtual_token_base, 1u, 1u);
        } else if (single_seam) {
            q4_q8_1_mma_128<true, true><<<virtual_grid,
                dim3(32, kWarps), kSharedBytes, stream>>>(
                    w, x, y, n_in, n_out, n_tok, epilogue, out_stride,
                    row_base, sm_count, 1u, virtual_token_base, 1u, 0u);
        } else {
            q4_q8_1_mma_128<true><<<virtual_grid,
                dim3(32, kWarps), kSharedBytes, stream>>>(
                    w, x, y, n_in, n_out, n_tok, epilogue, out_stride,
                    row_base, sm_count, 1u, virtual_token_base, 1u, 0u);
        }
        return true;
    }
    if (stream_k_numeric && workspace) {
        // Numeric seam replay stores one physical Stream-K boundary per logical
        // tile.  Sparse grids can place boundaries too close together for that
        // representation, so they must stay on the physical Stream-K path.
        if (!llama_compat && efficiency >= route_efficiency
            && ntiles >= 2 * sm_count
            && n_out % kRows == 0) {
            const bool compact =
                full_tile_variant == FullTileVariant::Rows64K256;
            const bool half_k =
                full_tile_variant == FullTileVariant::Rows128K128;
            static const bool cooperative_q4 =
                std::getenv("IMPARO_CUDA_COOP_Q4") != nullptr;
            const uint32_t compact_tiles =
                ((n_out + kCompactRows - 1) / kCompactRows) * grid.y;
            if (info) {
                info->route = LaunchRoute::FullTile;
                info->tile_rows = compact ? kCompactRows : kRows;
                info->logical_tiles = compact ? compact_tiles : ntiles;
                info->physical_blocks = info->logical_tiles;
                const uint32_t compact_waves =
                    (info->logical_tiles + sm_count - 1) / sm_count;
                info->efficiency = 100 * info->logical_tiles
                    / (sm_count * compact_waves);
            }
            if (half_k && epilogue && epilogue_q8) {
                if (epilogue == 2) {
                    q4_q8_1_full_tile<true, 4, true, false>
                        <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes, stream>>>(
                            w, x, y + row_base, epilogue_q8, n_in, n_out, n_tok,
                            work_n_tok, out_stride, 0);
                } else {
                    if (cooperative_q4) {
                        q4_q8_1_full_tile<true, 4, true, true, false, true>
                            <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes,
                                stream>>>(w, x, y + row_base, epilogue_q8,
                                    n_in, n_out, n_tok, work_n_tok, out_stride, 0);
                    } else {
                        q4_q8_1_full_tile<true, 4, true>
                            <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes,
                                stream>>>(w, x, y + row_base, epilogue_q8,
                                    n_in, n_out, n_tok, work_n_tok, out_stride, 0);
                    }
                }
                if (info) info->fused_q8 = true;
            } else if (half_k && epilogue) {
                q4_q8_1_full_tile<true, 4><<<ntiles, dim3(32, kWarps),
                    kHalfKSharedBytes, stream>>>(w, x, y + row_base, nullptr,
                        n_in, n_out, n_tok, work_n_tok, out_stride, 0);
            } else if (half_k) {
                if (efficiency < 90) {
                    if (cooperative_q4) {
                        q4_q8_1_full_tile<false, 4, false, true, true, true>
                            <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes,
                                stream>>>(w, x, y + row_base, nullptr, n_in,
                                    n_out, n_tok, work_n_tok, out_stride, sm_count);
                    } else {
                        q4_q8_1_full_tile<false, 4, false, true, true>
                            <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes,
                                stream>>>(w, x, y + row_base, nullptr, n_in,
                                    n_out, n_tok, work_n_tok, out_stride, sm_count);
                    }
                } else {
                    if (cooperative_q4) {
                        q4_q8_1_full_tile<false, 4, false, true, false, true>
                            <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes,
                                stream>>>(w, x, y + row_base, nullptr, n_in,
                                    n_out, n_tok, work_n_tok, out_stride, 0);
                    } else {
                        q4_q8_1_full_tile<false, 4>
                            <<<ntiles, dim3(32, kWarps), kHalfKSharedBytes,
                                stream>>>(w, x, y + row_base, nullptr, n_in,
                                    n_out, n_tok, work_n_tok, out_stride, 0);
                    }
                }
            } else if (compact && epilogue) {
                q4_q8_1_full_tile_r64<true><<<compact_tiles,
                    dim3(32, kWarps), kCompactSharedBytes, stream>>>(
                        w, x, y + row_base, nullptr, n_in, n_out, n_tok,
                        work_n_tok,
                        out_stride);
            } else if (compact) {
                q4_q8_1_full_tile_r64<false><<<compact_tiles,
                    dim3(32, kWarps), kCompactSharedBytes, stream>>>(
                        w, x, y + row_base, nullptr, n_in, n_out, n_tok,
                        work_n_tok,
                        out_stride);
            } else if (epilogue) {
                q4_q8_1_full_tile<true><<<ntiles, dim3(32, kWarps),
                    kSharedBytes, stream>>>(w, x, y + row_base, nullptr, n_in,
                                           n_out, n_tok, work_n_tok, out_stride, 0);
            } else {
                if (efficiency < 90) {
                    q4_q8_1_full_tile<false, 8, false, true, true>
                        <<<ntiles, dim3(32, kWarps), kSharedBytes, stream>>>(
                            w, x, y + row_base, nullptr, n_in, n_out, n_tok,
                            work_n_tok, out_stride, sm_count);
                } else {
                    q4_q8_1_full_tile<false><<<ntiles, dim3(32, kWarps),
                        kSharedBytes, stream>>>(w, x, y + row_base, nullptr, n_in,
                            n_out, n_tok, work_n_tok, out_stride, 0);
                }
            }
            return true;
        }
        // The lower architecture-owned threshold admits eligible full-tile grids;
        // it must not also change the ownership of projections that cannot use that
        // route. In canonical replay mode those fallbacks stay on the accepted
        // llama-compatible 90% physical Stream-K policy.
        const uint32_t physical_route_efficiency = canonical_full_tile
            ? 90u : route_efficiency;
        const uint32_t stream_grid = efficiency >= physical_route_efficiency
            ? ntiles : sm_count;
        if (info) {
            info->route = LaunchRoute::PhysicalStreamK;
            info->physical_blocks = stream_grid;
        }
        const bool direct_epilogue = epilogue && stream_grid == ntiles;
        float * output = epilogue && !direct_epilogue ? workspace : y + row_base;
        const uint32_t output_stride = epilogue && !direct_epilogue
            ? n_out : out_stride;
        float * fixup = workspace + (epilogue && !direct_epilogue
            ? uint64_t(n_out) * n_tok : 0);
        const bool compact_stream = !llama_compat && !epilogue && n_out % kRows == 0
            && full_tile_variant == FullTileVariant::Rows128K128;
        if (compact_stream) {
            q4_q8_1_stream_r64<<<stream_grid * 2, dim3(32, kWarps),
                kCompactSharedBytes, stream>>>(w, x, output, fixup, n_in,
                    n_out, n_tok, work_n_tok, output_stride);
        } else if (direct_epilogue) {
            q4_q8_1_stream<true, false><<<stream_grid, dim3(32, kWarps),
                kSharedBytes, stream>>>(w, x, output, fixup, n_in, n_out,
                                       n_tok, work_n_tok, output_stride);
        } else if (n_out % kRows == 0) {
            q4_q8_1_stream<false, true><<<stream_grid, dim3(32, kWarps),
                kSharedBytes, stream>>>(w, x, output, fixup, n_in, n_out,
                                       n_tok, work_n_tok, output_stride);
        } else {
            q4_q8_1_stream<false, false><<<stream_grid, dim3(32, kWarps),
                kSharedBytes, stream>>>(w, x, output, fixup, n_in, n_out,
                                       n_tok, work_n_tok, output_stride);
        }
        if (ntiles % stream_grid != 0) {
            q4_q8_1_stream_fixup<<<dim3(stream_grid, 4), 128, 0, stream>>>(
                output, fixup, n_in, n_out, n_tok, output_stride);
        }
        if (epilogue && !direct_epilogue) {
            const uint64_t count = uint64_t(n_out) * n_tok;
            q4_q8_1_stream_epilogue<<<uint32_t((count + 255) / 256), 256, 0,
                stream>>>(output, y, n_out, n_tok, out_stride, row_base);
        }
        return true;
    }
    if (info) {
        info->route = LaunchRoute::GridTile;
        info->physical_blocks = ntiles;
    }
    if (stream_k_numeric && efficiency < 90) {
        q4_q8_1_mma_128<true><<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
            w, x, y, n_in, n_out, n_tok, epilogue, out_stride, row_base,
            sm_count, stream_k_numeric, 0u, 0u, 0u);
    } else {
        q4_q8_1_mma_128<false><<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
            w, x, y, n_in, n_out, n_tok, epilogue, out_stride, row_base,
            sm_count, stream_k_numeric, 0u, 0u, 0u);
    }
    return true;
}

} // namespace imparo_sm80_mmq
