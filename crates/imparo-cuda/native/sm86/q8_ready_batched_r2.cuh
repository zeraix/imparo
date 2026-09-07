#pragma once

// Research-only SM86 R2 batched/interleaved Cursor V4 kernel.
// Keeps V1 math and launch order while requesting a 100% shared/L1 carveout.
// Two projections share one launch but retain one 256-thread, 8-warp CTA per
// projection/tile. Adjacent flat-grid CTAs process the same activation tile for
// projection 0 and 1, preserving 2 CTA/SM occupancy and improving Q8 L2 reuse.
// Isolated under native/tests; not a production selector or ABI surface.
#include "q8_mma_ready_a0.cuh"

// Research-only 2-A screen for the SM86 Q4_0 x Q8_1 kernel lab.
//
// This header deliberately lives under native/tests and is only included by the
// standalone lab executable.  It is not a production weight format or selector.
// The input is a byte-neutral split of canonical Q4 blocks into a uint16 scale
// plane and a 16-byte nibble plane.  Four consecutive K blocks for one row are
// adjacent so one typed uint4 load supplies the complete nibble payload.

namespace imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab {

constexpr uint32_t kPackM = 128;
constexpr uint32_t kPackKBlocks = 4;
constexpr uint32_t kReadyTokenTile = 8;
constexpr uint32_t kReadyTokenTilesPerCta = 16;
constexpr uint32_t kReadyQuantBytes = 16384;
constexpr uint32_t kReadyScaleBytes = 2048;
static_assert(kReadyQuantBytes + kReadyScaleBytes == 18432,
              "byte-neutral K128 x N128 Q8 activation stage");

constexpr uint64_t kReadyQuantVectorsPerTokenTile =
    uint64_t(kPackKBlocks) * 16 * kReadyTokenTile
    * sizeof(uint16_t) / sizeof(uint4);
constexpr uint64_t kReadyScaleVectorsPerTokenTile =
    uint64_t(kPackKBlocks) * kReadyTokenTile
    * sizeof(float) / sizeof(uint4);
static_assert(kReadyQuantVectorsPerTokenTile == 64,
              "one N8 tile has 64 quant uint4 vectors");
static_assert(kReadyScaleVectorsPerTokenTile == 8,
              "one N8 tile has 8 scale uint4 vectors");

template <uint32_t CopyThreads = imparo_sm80_mmq::kWarps * 32>
__device__ __forceinline__ void stage_ready_activation_cursor_async(
        const uint4 * __restrict__ quant_cursor,
        const uint4 * __restrict__ scale_cursor, int8_t * sy,
        uint32_t tid) {
    constexpr uint32_t quant_vectors = kReadyQuantBytes / sizeof(uint4);
    constexpr uint32_t scale_vectors = kReadyScaleBytes / sizeof(uint4);
    uint4 * dst_quant = reinterpret_cast<uint4 *>(sy);
    uint4 * dst_scale = reinterpret_cast<uint4 *>(sy + kReadyQuantBytes);
    for (uint32_t linear = tid; linear < quant_vectors;
         linear += CopyThreads) {
        imparo_sm80_mmq::copy_global_to_shared_16(
            dst_quant + linear, quant_cursor + linear);
    }
    for (uint32_t linear = tid; linear < scale_vectors;
         linear += CopyThreads) {
        imparo_sm80_mmq::copy_global_to_shared_16(
            dst_scale + linear, scale_cursor + linear);
    }
}

__device__ __forceinline__ void load_ready_b_m16n8k32(
        int (&b)[2], const int8_t * sy, uint32_t local_token0,
        uint32_t qblock) {
    const uint32_t local_tile = local_token0 / kReadyTokenTile;
    const uint16_t * tile = reinterpret_cast<const uint16_t *>(sy)
        + ((local_tile * 4 + qblock) * 16 * kReadyTokenTile);
    imparo_q8_mma_ready_a0_v1_authority_lab::
        load_b_ldmatrix_trans_a0_v1(b, tile);
}

__device__ __forceinline__ float load_ready_d8(
        const int8_t * sy, uint32_t local_token, uint32_t qblock) {
    const uint32_t local_tile = local_token / kReadyTokenTile;
    const uint32_t token_in_tile = local_token % kReadyTokenTile;
    const float * scales = reinterpret_cast<const float *>(
        sy + kReadyQuantBytes);
    return scales[(local_tile * 4 + qblock) * kReadyTokenTile
        + token_in_tile];
}

__host__ __device__ __forceinline__ uint64_t packed_record_index(
        uint32_t row, uint32_t kb, uint32_t n_in) {
    const uint64_t k_groups = uint64_t(n_in / 32) / kPackKBlocks;
    const uint64_t row_tile = row / kPackM;
    const uint64_t local_row = row % kPackM;
    const uint64_t k_group = kb / kPackKBlocks;
    const uint64_t local_kblock = kb % kPackKBlocks;
    return (((row_tile * k_groups + k_group) * kPackM + local_row)
        * kPackKBlocks + local_kblock);
}

// Four independent signed-nibble conversions in packed byte lanes.  __vsub4
// prevents borrow propagation between lanes, so this is exactly equivalent to
// unpack_q4_nibbles while giving SM86 a chance to replace the multiply-based
// sign fill with its packed-byte ALU sequence.
__device__ __forceinline__ int unpack_q4_nibbles_vsub4(uint32_t nibbles) {
    constexpr uint32_t low_nibbles = 0x0f0f0f0fu;
    constexpr uint32_t sign = 0x08080808u;
    const uint32_t biased = (nibbles & low_nibbles) ^ sign;
    return int(__vsub4(biased, sign));
}

template <uint32_t Rows, uint32_t CopyThreads>
__device__ __forceinline__ void stage_pair_packed_weights_async(
        const uint16_t * __restrict__ gate_scales16,
        const uint4 * __restrict__ gate_nibbles16,
        const uint16_t * __restrict__ up_scales16,
        const uint4 * __restrict__ up_nibbles16,
        uint8_t * packed_stage, uint32_t tid, uint32_t tile_row,
        uint32_t stage_block, uint32_t n_in) {
    constexpr uint32_t Records = Rows * kPackKBlocks;
    static_assert(Records % 8 == 0,
                  "pair packed scale planes require uint4 alignment");
    uint4 * gate_nibbles_stage = reinterpret_cast<uint4 *>(packed_stage);
    uint4 * up_nibbles_stage = gate_nibbles_stage + Records;
    uint16_t * gate_scales_stage = reinterpret_cast<uint16_t *>(
        up_nibbles_stage + Records);
    uint16_t * up_scales_stage = gate_scales_stage + Records;
    const uint64_t record_base = packed_record_index(
        tile_row, stage_block, n_in);
    for (uint32_t linear = tid; linear < Records; linear += CopyThreads) {
        imparo_sm80_mmq::copy_global_to_shared_16(
            gate_nibbles_stage + linear, gate_nibbles16 + record_base + linear);
        imparo_sm80_mmq::copy_global_to_shared_16(
            up_nibbles_stage + linear, up_nibbles16 + record_base + linear);
    }
    constexpr uint32_t ScaleVectors = Records / 8;
    uint4 * gate_scale_vectors = reinterpret_cast<uint4 *>(gate_scales_stage);
    uint4 * up_scale_vectors = reinterpret_cast<uint4 *>(up_scales_stage);
    const uint4 * gate_scale_source = reinterpret_cast<const uint4 *>(
        gate_scales16 + record_base);
    const uint4 * up_scale_source = reinterpret_cast<const uint4 *>(
        up_scales16 + record_base);
    for (uint32_t linear = tid; linear < ScaleVectors;
         linear += CopyThreads) {
        imparo_sm80_mmq::copy_global_to_shared_16(
            gate_scale_vectors + linear, gate_scale_source + linear);
        imparo_sm80_mmq::copy_global_to_shared_16(
            up_scale_vectors + linear, up_scale_source + linear);
    }
}

template <bool FullK,
          uint32_t Rows = imparo_sm80_mmq::kRows,
          uint32_t KernelWarps = imparo_sm80_mmq::kWarps,
          bool GuardRows = false,
          bool PartitionTokenGroups = false>
__device__ __forceinline__ void compute_segment(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        int8_t * sx, int8_t * sy,
        float (&partial)[(PartitionTokenGroups ? 8 : 32)
            * (Rows <= 80 ? 1 : 2)],
        uint32_t n_in, uint32_t n_tok, uint32_t work_n_tok,
        uint32_t blocks, uint32_t tile_row, uint32_t tile_token,
        uint32_t segment_begin, uint32_t segment_end,
        uint32_t active_rows = Rows) {
    using namespace imparo_sm80_mmq;
    constexpr uint32_t StageBlocks = kPackKBlocks;
    constexpr uint32_t WeightValues = StageBlocks * 32;
    constexpr uint32_t WeightStride = kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint64_t first_tile = tile_token / kReadyTokenTile;
    const uint64_t first_group_tile =
        uint64_t(segment_begin / StageBlocks) * token_tiles + first_tile;
    const uint64_t quant_group_stride =
        uint64_t(token_tiles) * kReadyQuantVectorsPerTokenTile;
    const uint64_t scale_group_stride =
        uint64_t(token_tiles) * kReadyScaleVectorsPerTokenTile;
    const uint4 * quant_cursor =
        reinterpret_cast<const uint4 *>(quant_u16)
        + first_group_tile * kReadyQuantVectorsPerTokenTile;
    const uint4 * scale_cursor =
        reinterpret_cast<const uint4 *>(d8_sideplane)
        + first_group_tile * kReadyScaleVectorsPerTokenTile;
    const uint32_t active_tokens = tile_token < work_n_tok
        ? min(kTokens, work_n_tok - tile_token) : 0;
    static_assert(Rows == 64 || Rows == 80 || Rows == 96 || Rows == 128,
                  "Q8-ready rows must preserve an MMA-aligned row mapping");
    constexpr uint32_t RowFragments = Rows <= 80 ? 1 : 2;
    constexpr uint32_t WarpPairs = Rows / (16 * RowFragments);
    constexpr uint32_t TokenGroups = kTokens / 32;
    static_assert(kTokens % 32 == 0,
                  "partitioned Q8-ready scheduling requires complete token groups");
    constexpr uint32_t ExpectedWarps = 2 * WarpPairs
        * (PartitionTokenGroups ? TokenGroups : 1);
    static_assert(KernelWarps == ExpectedWarps,
                  "Q8-ready schedule must assign one warp pair per work owner");
    constexpr uint32_t PartialItems =
        (PartitionTokenGroups ? 8 : 32) * RowFragments;
    const uint32_t warp_pair = warp >> 1;
    const uint32_t row_pair = PartitionTokenGroups
        ? warp_pair % WarpPairs : warp_pair;
    const uint32_t assigned_token_group = PartitionTokenGroups
        ? warp_pair / WarpPairs : 0;
    constexpr uint32_t TokenGroupIterations =
        PartitionTokenGroups ? 1 : TokenGroups;
#pragma unroll
    for (uint32_t item = 0; item < PartialItems; ++item) {
        partial[item] = 0.0f;
    }

    for (uint32_t stage_block = segment_begin;
         stage_block < segment_end; stage_block += StageBlocks) {
        stage_ready_activation_cursor_async<KernelWarps * 32>(
            quant_cursor, scale_cursor, sy, tid);
        quant_cursor += quant_group_stride;
        scale_cursor += scale_group_stride;
        commit_async_copies();

        // A warp consumes 32 consecutive 16-byte records.  The typed uint4 load
        // is intentional: four scalar uint32 loads do not establish a single
        // vector memory operation and may be scheduled/split independently.
        constexpr uint32_t packed_block_count = Rows * StageBlocks;
        for (uint32_t linear = tid; linear < packed_block_count;
             linear += KernelWarps * 32) {
            const uint32_t qblock = linear % StageBlocks;
            const uint32_t local_row = linear / StageBlocks;
            const uint32_t row = tile_row + local_row;
            const uint32_t kb = stage_block + qblock;
            const bool valid = (FullK || kb < segment_end)
                && (!GuardRows || local_row < active_rows);
            const uint64_t record = packed_record_index(row, kb, n_in);
            const uint4 packed4 = valid
                ? nibbles16[record] : make_uint4(
                    0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
            const uint16_t scale_bits = valid ? scales16[record] : 0;
            const __half scale_half = *reinterpret_cast<const __half *>(
                &scale_bits);
            const float scale = __half2float(scale_half);
            int8_t * dst = sx + local_row * WeightStride + qblock * 32;
            const uint32_t packed_words[4] = {
                packed4.x, packed4.y, packed4.z, packed4.w};
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
                const uint32_t packed = packed_words[word];
                reinterpret_cast<int *>(dst)[word] =
                    unpack_q4_nibbles(packed);
                reinterpret_cast<int *>(dst + 16)[word] =
                    unpack_q4_nibbles(packed >> 4);
            }
            reinterpret_cast<float *>(
                sx + local_row * WeightStride + WeightValues)[qblock] = scale;
        }

        wait_async_copies();
        __syncthreads();

#pragma unroll
        for (uint32_t qblock = 0; qblock < 4; ++qblock) {
            int af[2][4];
            float d4[2][2];
#pragma unroll
            for (uint32_t row_fragment = 0;
                 row_fragment < RowFragments;
                 ++row_fragment) {
                const uint32_t local_row0 = row_pair
                    * (16 * RowFragments) + row_fragment * 16;
                load_a_m16n8k32(
                    af[row_fragment], sx + local_row0 * WeightStride
                        + qblock * 32,
                    WeightStride);
#pragma unroll
                for (uint32_t scale_item = 0; scale_item < 2; ++scale_item) {
                    const uint32_t local_row = local_row0
                        + accumulator_row(lane, scale_item * 2);
                    d4[row_fragment][scale_item] =
                        reinterpret_cast<const float *>(
                            sx + local_row * WeightStride
                                + WeightValues)[qblock];
                }
            }
#pragma unroll
            for (uint32_t token_group_offset = 0;
                 token_group_offset < TokenGroupIterations;
                 ++token_group_offset) {
                const uint32_t token_group = assigned_token_group
                    + token_group_offset;
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
                    const uint32_t local_token0 = token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8;
                    if (local_token0 >= active_tokens) continue;
                    int bf[2];
                    load_ready_b_m16n8k32(
                        bf, sy, local_token0, qblock);
                    float d8[2];
#pragma unroll
                    for (uint32_t scale_item = 0; scale_item < 2;
                         ++scale_item) {
                        const uint32_t local_token = local_token0
                            + accumulator_token(lane, scale_item);
                        d8[scale_item] =
                            load_ready_d8(sy, local_token, qblock);
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
                                (((token_group_offset * 2 + token_fragment)
                                    * RowFragments
                                    + row_fragment) * 4) + item;
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

constexpr uint32_t kReadyCompactRows = 64;
constexpr uint32_t kReadyCompactSharedBytes =
    kReadyCompactRows * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kReadyR80Rows = 80;
constexpr uint32_t kReadyR80Warps = 10;
constexpr uint32_t kReadyR80SharedBytes =
    kReadyR80Rows * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kReadyBalancedRows = 96;
constexpr uint32_t kReadyBalancedWarps = 6;
constexpr uint32_t kReadyPartitionedWarps = 24;
constexpr uint32_t kReadyBalancedSharedBytes =
    kReadyBalancedRows * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kReadyPairCtaSharedBytes =
    2 * kReadyCompactRows * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kReadyPairCtaPrefetchSharedBytes =
    kReadyPairCtaSharedBytes
    + 2 * kReadyCompactRows * kPackKBlocks
        * (sizeof(uint4) + sizeof(uint16_t));
constexpr uint32_t kReadyPairCtaR128SharedBytes =
    2 * 128 * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kReadyPairCtaT256StageBytes =
    2 * kReadyCompactRows * imparo_sm80_mmq::kHalfKWeightStride
    + 2 * imparo_sm80_mmq::kTokens
        * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kReadyPairCtaT256GateBytes =
    2 * imparo_sm80_mmq::kTokens * kReadyCompactRows * sizeof(float);
constexpr uint32_t kReadyPairCtaT256SharedBytes =
    kReadyPairCtaT256StageBytes > kReadyPairCtaT256GateBytes
        ? kReadyPairCtaT256StageBytes : kReadyPairCtaT256GateBytes;

template <bool NumericSeams>
__launch_bounds__(256, 2)
__global__ void q8_ready_batched_r2_interleaved_carveout100_v17(
        const uint16_t * __restrict__ scales0,
        const uint4 * __restrict__ nibbles0,
        const uint16_t * __restrict__ scales1,
        const uint4 * __restrict__ nibbles1,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst0, float * __restrict__ dst1,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t numeric_stream_grid) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    const uint32_t projection = uint32_t(blockIdx.x) & 1u;
    const uint32_t tile_index = uint32_t(blockIdx.x) >> 1u;
    const uint16_t *scales16 = projection ? scales1 : scales0;
    const uint4 *nibbles16 = projection ? nibbles1 : nibbles0;
    float *dst = projection ? dst1 : dst0;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (tile_index / ntx) * kRows;
    const uint32_t tile_token = (tile_index % ntx) * kTokens;
    float partial[64];

    uint32_t numeric_seam = 0;
    if constexpr (NumericSeams) {
        const uint32_t logical_grid = uint32_t(gridDim.x) / 2u;
        const uint64_t total_work = uint64_t(logical_grid) * blocks;
        uint32_t worker = uint32_t(
            (uint64_t(tile_index) * numeric_stream_grid + logical_grid - 1)
            / logical_grid);
        worker = max(worker, 1u);
        if (worker < numeric_stream_grid) {
            const uint64_t boundary = stream_boundary(
                worker, total_work, numeric_stream_grid, blocks);
            if (boundary / blocks == tile_index && boundary % blocks) {
                numeric_seam = uint32_t(boundary % blocks);
            }
        }
    }

    const uint32_t first_end = numeric_seam ? numeric_seam : blocks;
    if constexpr (NumericSeams) {
        if (numeric_seam) {
            compute_segment<false>(scales16, nibbles16, quant_u16, d8_sideplane, token_tiles, sx, sy, partial,
                n_in, n_tok, work_n_tok, blocks, tile_row, tile_token,
                0, first_end);
        } else {
            compute_segment<true>(scales16, nibbles16, quant_u16, d8_sideplane, token_tiles, sx, sy, partial,
                n_in, n_tok, work_n_tok, blocks, tile_row, tile_token,
                0, blocks);
        }
    } else {
        compute_segment<true>(scales16, nibbles16, quant_u16, d8_sideplane, token_tiles, sx, sy, partial,
            n_in, n_tok, work_n_tok, blocks, tile_row, tile_token, 0, blocks);
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
            compute_segment<false>(scales16, nibbles16, quant_u16, d8_sideplane, token_tiles, sx, sy, partial,
                n_in, n_tok, work_n_tok, blocks, tile_row, tile_token,
                numeric_seam, blocks);
        }
    }

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
                    const uint32_t token = tile_token + local_token;
                    if (token >= n_tok) continue;
                    const uint32_t sum_index =
                        (((token_group * 2 + token_fragment) * 2
                            + row_fragment) * 4) + item;
                    float * slot = dst + uint64_t(token) * dst_stride
                        + tile_row + local_row;
                    *slot = NumericSeams && numeric_seam
                        ? partial[sum_index] + *slot : partial[sum_index];
                }
            }
        }
    }
#else
    (void)scales0; (void)nibbles0; (void)scales1; (void)nibbles1;
    (void)quant_u16; (void)d8_sideplane; (void)token_tiles;
    (void)dst0; (void)dst1; (void)n_in; (void)n_out; (void)n_tok;
    (void)work_n_tok; (void)dst_stride; (void)numeric_stream_grid;
#endif
}

template <bool DirectEpilogue>
__launch_bounds__(256, 2)
__global__ void q8_ready_full_tile_direct(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    float partial[64];
    compute_segment<true>(
        scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
        sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks);
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
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

// Full logical-grid execution with the exact physical Stream-K numeric seam.
// The CTA owns a complete output tile, but rounds the prefix and suffix at the
// same K-block boundary as the established physical-grid route. This retains
// the Q8-ready data path without changing the native reduction contract.
__launch_bounds__(256, 2)
__global__ void q8_ready_full_tile_seamed(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t numeric_stream_grid) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_index = uint32_t(blockIdx.x);
    const uint32_t tile_row = (tile_index / ntx) * kRows;
    const uint32_t tile_token = (tile_index % ntx) * kTokens;
    const uint32_t logical_grid = uint32_t(gridDim.x);
    const uint64_t total_work = uint64_t(logical_grid) * blocks;
    uint32_t worker = uint32_t(
        (uint64_t(tile_index) * numeric_stream_grid + logical_grid - 1)
        / logical_grid);
    worker = max(worker, 1u);
    uint32_t numeric_seam = 0;
    if (worker < numeric_stream_grid) {
        const uint64_t boundary = stream_boundary(
            worker, total_work, numeric_stream_grid, blocks);
        if (boundary / blocks == tile_index && boundary % blocks) {
            numeric_seam = uint32_t(boundary % blocks);
        }
    }
    float partial[64];
    if (numeric_seam) {
        compute_segment<false>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
            tile_row, tile_token, 0, numeric_seam);
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
                        if (token < n_tok) {
                            const uint32_t sum_index =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + item;
                            dst[uint64_t(token) * dst_stride
                                + tile_row + local_row] = partial[sum_index];
                        }
                    }
                }
            }
        }
        __syncthreads();
        compute_segment<false>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
            tile_row, tile_token, numeric_seam, blocks);
    } else {
        compute_segment<true>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
            tile_row, tile_token, 0, blocks);
    }
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
                    const uint32_t token = tile_token + local_token;
                    if (token >= n_tok) continue;
                    const uint32_t sum_index =
                        (((token_group * 2 + token_fragment) * 2
                            + row_fragment) * 4) + item;
                    float * slot = dst + uint64_t(token) * dst_stride
                        + tile_row + local_row;
                    *slot = numeric_seam
                        ? *slot + partial[sum_index] : partial[sum_index];
                }
            }
        }
    }
#else
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride; (void)numeric_stream_grid;
#endif
}

inline bool launch_q8_ready_seamed(
        const uint16_t * scales16, const uint4 * nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t work_n_tok,
        uint32_t dst_stride, uint32_t numeric_stream_grid,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(scales16)
        | reinterpret_cast<uintptr_t>(nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!scales16 || !nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || !numeric_stream_grid || work_n_tok != n_tok
        || n_in % (32 * kPackKBlocks) != 0 || n_out % kPackM != 0
        || dst_stride < n_out) return false;
    static const bool configured = [] {
        const cudaError_t rc = cudaFuncSetAttribute(
            q8_ready_full_tile_seamed,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kHalfKSharedBytes);
        if (rc != cudaSuccess) cudaGetLastError();
        return rc == cudaSuccess;
    }();
    if (!configured) return false;
    const uint64_t grid = uint64_t(n_out / kRows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_full_tile_seamed<<<uint32_t(grid), dim3(32, kWarps),
        kHalfKSharedBytes, stream>>>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            dst, n_in, n_out, n_tok, work_n_tok, dst_stride,
            numeric_stream_grid);
    return cudaPeekAtLastError() == cudaSuccess;
}

// Exact physical Stream-K ownership with Q8-ready activations. The scheduling,
// prefix slots, and fixup order are intentionally identical to the admitted
// native route; only the packed Q4 and activation staging are replaced.
__launch_bounds__(256, 2)
__global__ void q8_ready_physical_stream(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, float * __restrict__ fixup,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = n_out / kRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    uint64_t work = stream_boundary(
        blockIdx.x, total_work, gridDim.x, blocks);
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
        compute_segment<false>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
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
                        if (direct) {
                            dst[uint64_t(token) * dst_stride
                                + tile_row + local_row] = partial[sum_index];
                        } else {
                            fixup[(uint64_t(blockIdx.x) * kTokens
                                + local_token) * kRows + local_row]
                                = partial[sum_index];
                        }
                    }
                }
            }
        }
        __syncthreads();
        work = segment_work_stop;
    }
#else
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst; (void)fixup;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

inline bool configure_q8_ready_physical_stream() {
    static const bool configured = [] {
        const cudaError_t rc = cudaFuncSetAttribute(
            q8_ready_physical_stream,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            imparo_sm80_mmq::kHalfKSharedBytes);
        if (rc != cudaSuccess) cudaGetLastError();
        return rc == cudaSuccess;
    }();
    return configured;
}

inline bool launch_q8_ready_physical_stream(
        const uint16_t * scales16, const uint4 * nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, float * fixup,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t stream_grid, cudaStream_t stream,
        bool * public_output_committed = nullptr) {
    using namespace imparo_sm80_mmq;
    if (public_output_committed) *public_output_committed = false;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(scales16)
        | reinterpret_cast<uintptr_t>(nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst)
        | reinterpret_cast<uintptr_t>(fixup);
    if (!scales16 || !nibbles16 || !quant_u16 || !d8_sideplane
        || !dst || !fixup || (bits & 15u) != 0 || !token_tiles
        || !n_in || !n_out || !n_tok || !stream_grid
        || work_n_tok != n_tok || n_in % (32 * kPackKBlocks) != 0
        || n_out % kPackM != 0 || dst_stride < n_out
        || uint64_t(token_tiles) * kReadyTokenTile < n_tok) return false;
    if (!configure_q8_ready_physical_stream()) return false;
    const uint64_t ntiles = uint64_t(n_out / kRows)
        * ((uint64_t(n_tok) + kTokens - 1) / kTokens);
    if (!ntiles) return false;
    q8_ready_physical_stream<<<stream_grid, dim3(32, kWarps),
        kHalfKSharedBytes, stream>>>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            dst, fixup, n_in, n_out, n_tok, work_n_tok, dst_stride);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (public_output_committed) *public_output_committed = true;
    if (ntiles % stream_grid != 0) {
        q4_q8_1_stream_fixup<<<dim3(stream_grid, 4), 128, 0, stream>>>(
            dst, fixup, n_in, n_out, n_tok, dst_stride);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

// Shape-specialized AOT replay of the admitted R128 physical Stream-K route.
// For N=449, K=10240, M=2560 and grid=60, stream_boundary() has an exact
// three-worker period.  Every group of three workers owns one 128-row tile and
// the four token tiles as follows (K is measured in Q4/Q8 blocks):
//
//   worker 3g+0: T0 [0,320) suffix, T1 [0,104) prefix
//   worker 3g+1: T1 [104,320) suffix, T2 [0,208) prefix
//   worker 3g+2: T2 [208,320) suffix, T3 [0,320) suffix
//
// These are exactly stream_boundary(i, 25600, 60, 320), including its K8
// alignment.  Encoding the plan in template arguments removes the runtime
// 64-bit boundary/div/mod loop without changing any K32 MMA accumulation or
// suffix-plus-nearest-prefix reassociation.  T0..T2 are the full 384 tokens;
// T3 is the independently compiled 65-token tail.
constexpr uint32_t kExact449NIn = 10240;
constexpr uint32_t kExact449NOut = 2560;
constexpr uint32_t kExact449NTok = 449;
constexpr uint32_t kExact449PaddedTokens = 512;
constexpr uint32_t kExact449TokenTiles =
    kExact449PaddedTokens / kReadyTokenTile;
constexpr uint32_t kExact449Blocks = kExact449NIn / 32;
constexpr uint32_t kExact449PhysicalGrid = 60;
constexpr uint32_t kExact449WorkerGroups = 20;
constexpr uint32_t kExact449FixupSeams = 40;
static_assert(imparo_sm80_mmq::kRows == 128,
              "exact-449 schedule is receipt-bound to R128");
static_assert(imparo_sm80_mmq::kTokens == 128,
              "exact-449 schedule is receipt-bound to T128");
static_assert(kExact449Blocks == 320,
              "exact-449 schedule is receipt-bound to K320 blocks");
static_assert(kExact449WorkerGroups * 3 == kExact449PhysicalGrid,
              "exact-449 worker grouping must cover grid60");
static_assert(kExact449WorkerGroups * imparo_sm80_mmq::kRows
                  == kExact449NOut,
              "exact-449 worker groups must cover all output rows");

__launch_bounds__(256, 2)
__global__ void q8_ready_physical_stream_exact449_grid60(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane,
        float * __restrict__ dst, float * __restrict__ fixup) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t worker = uint32_t(blockIdx.x);
    const uint32_t worker_group = worker / 3;
    const uint32_t phase = worker - worker_group * 3;
    const uint32_t tile_row = worker_group * kRows;
    const uint32_t first_tile_token = phase * kTokens;
    const uint32_t first_segment_begin = phase * 104;
    const uint32_t second_tile_token = first_tile_token + kTokens;
    const uint32_t second_segment_end = phase == 2
        ? kExact449Blocks : (phase + 1) * 104;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    float partial[64];

#pragma unroll
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const uint32_t tile_token = pass == 0
            ? first_tile_token : second_tile_token;
        const uint32_t segment_begin = pass == 0
            ? first_segment_begin : 0;
        const uint32_t segment_end = pass == 0
            ? kExact449Blocks : second_segment_end;
        const bool direct = pass == 0 || phase == 2;
        compute_segment<false>(
            scales16, nibbles16, quant_u16, d8_sideplane,
            kExact449TokenTiles, sx, sy, partial,
            kExact449NIn, kExact449NTok, kExact449NTok,
            kExact449Blocks, tile_row, tile_token,
            segment_begin, segment_end);
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
                        if (token >= kExact449NTok) continue;
                        const uint32_t sum_index =
                            (((token_group * 2 + token_fragment) * 2
                                + row_fragment) * 4) + item;
                        if (direct) {
                            dst[uint64_t(token) * kExact449NOut
                                + tile_row + local_row] = partial[sum_index];
                        } else {
                            fixup[(uint64_t(worker) * kTokens + local_token)
                                * kRows + local_row] = partial[sum_index];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)dst; (void)fixup;
#endif
}

// Only the 40 real seams are launched.  Every exact-449 worker spans more than
// one tile, so a split tile has exactly one earlier prefix; the generic
// nearest-to-farthest scan therefore reduces to this same single addition.
__global__ void q8_ready_stream_fixup_exact449_grid60(
        float * dst, const float * fixup) {
    using namespace imparo_sm80_mmq;
    const uint32_t seam = uint32_t(blockIdx.x);
    const uint32_t worker_group = seam / 2;
    const uint32_t phase = seam - worker_group * 2;
    const uint32_t prefix_worker = worker_group * 3 + phase;
    const uint32_t tile_token = phase == 0 ? 128u : 256u;
    const uint32_t tile_row = worker_group * kRows;
    const uint32_t local_row = threadIdx.x;
    const uint32_t token_begin = uint32_t(blockIdx.y) * 32;
#pragma unroll
    for (uint32_t local_token = token_begin;
         local_token < token_begin + 32; ++local_token) {
        const float prefix = fixup[
            (uint64_t(prefix_worker) * kTokens + local_token)
                * kRows + local_row];
        dst[uint64_t(tile_token + local_token) * kExact449NOut
            + tile_row + local_row] += prefix;
    }
}

inline bool configure_q8_ready_physical_stream_exact449_grid60() {
    static const bool configured = [] {
        const cudaError_t rc = cudaFuncSetAttribute(
            q8_ready_physical_stream_exact449_grid60,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            imparo_sm80_mmq::kHalfKSharedBytes);
        if (rc != cudaSuccess) cudaGetLastError();
        return rc == cudaSuccess;
    }();
    return configured;
}

inline bool launch_q8_ready_physical_stream_exact449_grid60(
        const uint16_t * scales16, const uint4 * nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, float * fixup,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t stream_grid, cudaStream_t stream,
        bool * public_output_committed = nullptr) {
    using namespace imparo_sm80_mmq;
    if (public_output_committed) *public_output_committed = false;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(scales16)
        | reinterpret_cast<uintptr_t>(nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst)
        | reinterpret_cast<uintptr_t>(fixup);
    if (!scales16 || !nibbles16 || !quant_u16 || !d8_sideplane
            || !dst || !fixup || (bits & 15u) != 0
            || token_tiles != kExact449TokenTiles
            || n_in != kExact449NIn || n_out != kExact449NOut
            || n_tok != kExact449NTok || work_n_tok != kExact449NTok
            || dst_stride != kExact449NOut
            || stream_grid != kExact449PhysicalGrid) {
        return false;
    }
    if (!configure_q8_ready_physical_stream_exact449_grid60()) return false;
    q8_ready_physical_stream_exact449_grid60
        <<<kExact449PhysicalGrid, dim3(32, kWarps),
        kHalfKSharedBytes, stream>>>(
            scales16, nibbles16, quant_u16, d8_sideplane, dst, fixup);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (public_output_committed) *public_output_committed = true;
    q8_ready_stream_fixup_exact449_grid60
        <<<dim3(kExact449FixupSeams, 4), kRows, 0, stream>>>(dst, fixup);
    return cudaPeekAtLastError() == cudaSuccess;
}

// R64 physical Stream-K laboratory route.  It keeps the established K32
// accumulation order and nearest-to-farthest prefix fixup, but halves the
// independent output-row ownership per CTA.  The smaller accumulator and Q4
// stage are intended to expose a third resident CTA on SM86; it is a distinct
// numerical route because its physical worker seams are receipt-bound.
__launch_bounds__(256, 3)
__global__ void q8_ready_physical_stream_r64(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, float * __restrict__ fixup,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kReadyCompactRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = n_out / kReadyCompactRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    uint64_t work = stream_boundary(
        blockIdx.x, total_work, gridDim.x, blocks);
    const uint64_t work_stop = stream_boundary(
        blockIdx.x + 1, total_work, gridDim.x, blocks);
    float partial[32];
    while (work < work_stop) {
        const uint32_t logical_tile = uint32_t(work / blocks);
        const uint32_t segment_begin = uint32_t(work % blocks);
        const uint64_t tile_stop = uint64_t(logical_tile + 1) * blocks;
        const uint64_t segment_work_stop = min(work_stop, tile_stop);
        const uint32_t segment_end = uint32_t(segment_work_stop
            - uint64_t(logical_tile) * blocks);
        const uint32_t tile_row =
            (logical_tile / ntx) * kReadyCompactRows;
        const uint32_t tile_token = (logical_tile % ntx) * kTokens;
        compute_segment<false, kReadyCompactRows>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
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
                    const uint32_t token = tile_token + local_token;
                    if (token >= n_tok) continue;
                    const uint32_t sum_index =
                        ((token_group * 2 + token_fragment) * 4) + item;
                    if (direct) {
                        dst[uint64_t(token) * dst_stride
                            + tile_row + local_row] = partial[sum_index];
                    } else {
                        fixup[(uint64_t(blockIdx.x) * kTokens
                            + local_token) * kReadyCompactRows + local_row]
                            = partial[sum_index];
                    }
                }
            }
        }
        __syncthreads();
        work = segment_work_stop;
    }
#else
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst; (void)fixup;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

__global__ void q8_ready_stream_fixup_r64(
        float * dst, const float * fixup, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride) {
    using namespace imparo_sm80_mmq;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t nty = n_out / kReadyCompactRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    const uint64_t work = stream_boundary(
        blockIdx.x, total_work, gridDim.x, blocks);
    const uint64_t work_stop = stream_boundary(
        blockIdx.x + 1, total_work, gridDim.x, blocks);
    const bool did_not_write_last = work / blocks == work_stop / blocks
        && work_stop % blocks != 0;
    if (work == work_stop || work % blocks == 0 || did_not_write_last) return;

    const uint32_t logical_tile = uint32_t(work / blocks);
    const uint32_t tile_row =
        (logical_tile / ntx) * kReadyCompactRows;
    const uint32_t tile_token = (logical_tile % ntx) * kTokens;
    const uint32_t local_row = threadIdx.x;
    if (local_row >= kReadyCompactRows
            || tile_row + local_row >= n_out) return;

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
                * kReadyCompactRows + local_row];
            if (previous_work % blocks == 0
                    || previous_work / blocks < logical_tile) break;
            --previous;
            previous_stop = previous_work;
        }
        dst[uint64_t(tile_token + local_token) * dst_stride
            + tile_row + local_row] += sum;
    }
}

inline bool launch_q8_ready_physical_stream_r64(
        const uint16_t * scales16, const uint4 * nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, float * fixup,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t stream_grid, cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(scales16)
        | reinterpret_cast<uintptr_t>(nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst)
        | reinterpret_cast<uintptr_t>(fixup);
    if (!scales16 || !nibbles16 || !quant_u16 || !d8_sideplane
        || !dst || !fixup || (bits & 15u) != 0 || !token_tiles
        || !n_in || !n_out || !n_tok || !stream_grid
        || work_n_tok != n_tok || n_in % (32 * kPackKBlocks) != 0
        || n_out % kReadyCompactRows != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t rc = cudaFuncSetAttribute(
            q8_ready_physical_stream_r64,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyCompactSharedBytes);
        if (rc != cudaSuccess) cudaGetLastError();
        return rc == cudaSuccess;
    }();
    if (!configured) return false;
    const uint32_t ntiles = (n_out / kReadyCompactRows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!ntiles || stream_grid > ntiles) return false;
    q8_ready_physical_stream_r64<<<stream_grid, dim3(32, kWarps),
        kReadyCompactSharedBytes, stream>>>(
            scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
            dst, fixup, n_in, n_out, n_tok, work_n_tok, dst_stride);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (ntiles % stream_grid != 0) {
        q8_ready_stream_fixup_r64<<<dim3(stream_grid, 4),
            kReadyCompactRows, 0, stream>>>(
                dst, fixup, n_in, n_out, n_tok, dst_stride);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

// Occupancy laboratory companion for the direct gate/up path.  It preserves
// every K-stage and per-output accumulation in the 128-row kernel, but splits
// independent output rows into 64-row CTAs.  The smaller accumulator and Q4
// stage let ptxas lower the register/shared footprint without changing logits.
template <bool DirectEpilogue>
__launch_bounds__(256, 3)
__global__ void q8_ready_full_tile_direct_r64(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kReadyCompactRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kReadyCompactRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    float partial[32];
    compute_segment<true, kReadyCompactRows>(
        scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
        sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
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
#else
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

// Exact-divisor laboratory for E4B Down.  Eighty rows divide 2,560 exactly,
// yielding 32 independent CTAs for the 30-SM verification GPU.  One warp pair
// owns each 16-row group and keeps all four token groups, so the CTA has ten
// warps, 32 f32 accumulators per thread, no tail CTA and no duplicated global
// Q4/Q8 staging.
template <bool DirectEpilogue>
__launch_bounds__(320, 1)
__global__ void q8_ready_full_tile_direct_r80(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kReadyR80Rows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kReadyR80Rows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    const uint32_t active_rows = min(kReadyR80Rows, n_out - tile_row);
    float partial[32];
    compute_segment<true, kReadyR80Rows, kReadyR80Warps, true>(
        scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
        sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks, active_rows);
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
                if (token >= n_tok || local_row >= active_rows) continue;
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
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

// Balanced occupancy laboratory for exact-width Down.  R96 yields 27 CTAs
// for E4B's 2560 output rows, filling 27 of SM86's 30 SMs while preserving
// one full-K owner and the established K32 accumulation order per output row.
// The final CTA guards its 64 live rows; inactive rows are zero-staged and
// never write public output.
template <bool DirectEpilogue>
__launch_bounds__(192, 1)
__global__ void q8_ready_full_tile_direct_r96(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kReadyBalancedRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kReadyBalancedRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    const uint32_t active_rows = min(kReadyBalancedRows, n_out - tile_row);
    float partial[64];
    compute_segment<true, kReadyBalancedRows, kReadyBalancedWarps, true>(
        scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
        sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks, active_rows);
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
                    const uint32_t token = tile_token + local_token;
                    if (token >= n_tok || local_row >= active_rows) continue;
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
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

// Register-pressure laboratory for exact-width Down.  The CTA stages each Q4
// weight record and each Q8 activation record once, as in R96, but assigns one
// warp pair to each (32-row group, 32-token group).  This preserves every K32
// accumulation order while reducing the live accumulator array from 64 to 16
// floats per thread and exposing 24 resident warps for latency hiding.
template <bool DirectEpilogue>
__launch_bounds__(768, 1)
__global__ void q8_ready_full_tile_direct_r96_w24(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t work_n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx = reinterpret_cast<int8_t *>(storage);
    int8_t * sy = sx + kReadyBalancedRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t warp_pair = warp >> 1;
    const uint32_t row_pair = warp_pair % 3;
    const uint32_t token_group = warp_pair / 3;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + kTokens - 1) / kTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * kReadyBalancedRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    const uint32_t active_rows = min(kReadyBalancedRows, n_out - tile_row);
    float partial[16];
    compute_segment<true, kReadyBalancedRows, kReadyPartitionedWarps,
                    true, true>(
        scales16, nibbles16, quant_u16, d8_sideplane, token_tiles,
        sx, sy, partial, n_in, n_tok, work_n_tok, blocks,
        tile_row, tile_token, 0, blocks, active_rows);
#pragma unroll
    for (uint32_t token_fragment = 0; token_fragment < 2;
         ++token_fragment) {
#pragma unroll
        for (uint32_t row_fragment = 0; row_fragment < 2;
             ++row_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = row_pair * 32
                    + row_fragment * 16 + accumulator_row(lane, item);
                const uint32_t local_token = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8
                    + accumulator_token(lane, item);
                const uint32_t token = tile_token + local_token;
                if (token >= n_tok || local_row >= active_rows) continue;
                const uint32_t sum_index =
                    ((token_fragment * 2 + row_fragment) * 4) + item;
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
    (void)scales16; (void)nibbles16; (void)quant_u16;
    (void)d8_sideplane; (void)token_tiles; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}
// Cooperative gate/up CTA laboratory. Four warps accumulate one 64-row gate
// tile and four warps accumulate the matching up tile. Both projections share
// one Q8 activation stage, while each output keeps the exact full-K FMA order.
// The gate warps publish first; after a CTA barrier the up warps apply the
// direct GELU(gate) * up epilogue without a temporary matrix or extra launch.
template <uint32_t Rows, bool SharedGate, bool PrefetchPacked,
          bool QuantizeOutput, uint32_t TokenHalves = 1,
          bool PackedByteSubtract = false, bool WriteDense = true>
__launch_bounds__(512, 1)
__global__ void q8_ready_paircta_r64(
        const uint16_t * __restrict__ gate_scales16,
        const uint4 * __restrict__ gate_nibbles16,
        const uint16_t * __restrict__ up_scales16,
        const uint4 * __restrict__ up_nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst,
        uint16_t * __restrict__ output_quant_u16,
        float * __restrict__ output_d8_sideplane,
        uint32_t output_token_tiles, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t dst_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    static_assert(Rows == 64 || Rows == 128,
                  "pair-CTA row tile must be R64 or R128");
    static_assert(TokenHalves == 1 || TokenHalves == 2,
                  "pair-CTA token tile must be T128 or T256");
    static_assert(TokenHalves == 1 || (Rows == 64 && SharedGate
                      && !PrefetchPacked && !QuantizeOutput),
                  "T256 is isolated to the R64 shared-gate screen");
    static_assert(WriteDense || (Rows == 64 && SharedGate && QuantizeOutput),
                  "sidecar-only output requires the proven shared-gate Q8 epilogue");
    constexpr uint32_t KernelWarps = (Rows / 8) * TokenHalves;
    constexpr uint32_t WarpsPerProjection = KernelWarps / 2;
    constexpr uint32_t BaseWarpsPerProjection = Rows / 16;
    constexpr uint32_t WideTokens = TokenHalves * kTokens;
    constexpr uint32_t ActivationBytes =
        kReadyQuantBytes + kReadyScaleBytes;
    constexpr uint32_t StageBlocks = kPackKBlocks;
    constexpr uint32_t WeightValues = StageBlocks * 32;
    constexpr uint32_t WeightStride = kHalfKWeightStride;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx_gate = reinterpret_cast<int8_t *>(storage);
    int8_t * sx_up = sx_gate + Rows * WeightStride;
    int8_t * sy = sx_up + Rows * WeightStride;
    uint8_t * packed_stage = reinterpret_cast<uint8_t *>(
        sy + TokenHalves * ActivationBytes);
    float * gate_tile = reinterpret_cast<float *>(storage);
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t projection = warp / WarpsPerProjection;
    const uint32_t local_warp = warp % WarpsPerProjection;
    const uint32_t token_half = local_warp / BaseWarpsPerProjection;
    const uint32_t compute_warp =
        local_warp % BaseWarpsPerProjection;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + WideTokens - 1) / WideTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * Rows;
    const uint32_t tile_token = (blockIdx.x % ntx) * WideTokens;
    const uint32_t active_tokens = tile_token < n_tok
        ? min(WideTokens, n_tok - tile_token) : 0;
    const uint32_t half_base = token_half * kTokens;
    const uint32_t active_half_tokens = active_tokens > half_base
        ? min(kTokens, active_tokens - half_base) : 0;
    const uint64_t first_tile = tile_token / kReadyTokenTile;
    const uint4 * quant_cursor = reinterpret_cast<const uint4 *>(quant_u16)
        + first_tile * kReadyQuantVectorsPerTokenTile;
    const uint4 * scale_cursor = reinterpret_cast<const uint4 *>(d8_sideplane)
        + first_tile * kReadyScaleVectorsPerTokenTile;
    const uint64_t quant_group_stride =
        uint64_t(token_tiles) * kReadyQuantVectorsPerTokenTile;
    const uint64_t scale_group_stride =
        uint64_t(token_tiles) * kReadyScaleVectorsPerTokenTile;
    constexpr uint64_t quant_half_stride =
        uint64_t(kReadyTokenTilesPerCta)
        * kReadyQuantVectorsPerTokenTile;
    constexpr uint64_t scale_half_stride =
        uint64_t(kReadyTokenTilesPerCta)
        * kReadyScaleVectorsPerTokenTile;
    float partial[64];
#pragma unroll
    for (uint32_t item = 0; item < 64; ++item) partial[item] = 0.0f;

    if constexpr (PrefetchPacked) {
        stage_pair_packed_weights_async<Rows, KernelWarps * 32>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            packed_stage, tid, tile_row, 0, n_in);
        commit_async_copies();
        wait_async_copies();
        __syncthreads();
    }

    for (uint32_t stage_block = 0; stage_block < blocks;
         stage_block += StageBlocks) {
        for (uint32_t half = 0; half < TokenHalves; ++half) {
            stage_ready_activation_cursor_async<KernelWarps * 32>(
                quant_cursor + half * quant_half_stride,
                scale_cursor + half * scale_half_stride,
                sy + half * ActivationBytes, tid);
        }
        quant_cursor += quant_group_stride;
        scale_cursor += scale_group_stride;
        commit_async_copies();

        constexpr uint32_t records_per_projection = Rows * StageBlocks;
        constexpr uint32_t packed_block_count =
            2 * records_per_projection;
        for (uint32_t linear = tid; linear < packed_block_count;
             linear += KernelWarps * 32) {
            const uint32_t weight_projection =
                linear / records_per_projection;
            const uint32_t record_linear =
                linear % records_per_projection;
            const uint32_t qblock = record_linear % StageBlocks;
            const uint32_t local_row = record_linear / StageBlocks;
            const uint32_t row = tile_row + local_row;
            const uint32_t kb = stage_block + qblock;
            int8_t * sx = weight_projection ? sx_up : sx_gate;
            uint4 packed4;
            uint16_t scale_bits;
            if constexpr (PrefetchPacked) {
                constexpr uint32_t Records = Rows * StageBlocks;
                const uint4 * gate_nibbles_stage =
                    reinterpret_cast<const uint4 *>(packed_stage);
                const uint4 * up_nibbles_stage =
                    gate_nibbles_stage + Records;
                const uint16_t * gate_scales_stage =
                    reinterpret_cast<const uint16_t *>(
                        up_nibbles_stage + Records);
                const uint16_t * up_scales_stage =
                    gate_scales_stage + Records;
                packed4 = (weight_projection
                    ? up_nibbles_stage : gate_nibbles_stage)[record_linear];
                scale_bits = (weight_projection
                    ? up_scales_stage : gate_scales_stage)[record_linear];
            } else {
                const uint16_t * scales16 = weight_projection
                    ? up_scales16 : gate_scales16;
                const uint4 * nibbles16 = weight_projection
                    ? up_nibbles16 : gate_nibbles16;
                const uint64_t record = packed_record_index(row, kb, n_in);
                packed4 = nibbles16[record];
                scale_bits = scales16[record];
            }
            const __half scale_half =
                *reinterpret_cast<const __half *>(&scale_bits);
            int8_t * out = sx + local_row * WeightStride + qblock * 32;
            const uint32_t packed_words[4] = {
                packed4.x, packed4.y, packed4.z, packed4.w};
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
                const uint32_t packed = packed_words[word];
                if constexpr (PackedByteSubtract) {
                    reinterpret_cast<int *>(out)[word] =
                        unpack_q4_nibbles_vsub4(packed);
                    reinterpret_cast<int *>(out + 16)[word] =
                        unpack_q4_nibbles_vsub4(packed >> 4);
                } else {
                    reinterpret_cast<int *>(out)[word] =
                        unpack_q4_nibbles(packed);
                    reinterpret_cast<int *>(out + 16)[word] =
                        unpack_q4_nibbles(packed >> 4);
                }
            }
            reinterpret_cast<float *>(
                sx + local_row * WeightStride + WeightValues)[qblock] =
                    __half2float(scale_half);
        }

        wait_async_copies();
        __syncthreads();
        const bool has_next_stage = stage_block + StageBlocks < blocks;
        if constexpr (PrefetchPacked) {
            if (has_next_stage) {
                stage_pair_packed_weights_async<Rows, KernelWarps * 32>(
                    gate_scales16, gate_nibbles16,
                    up_scales16, up_nibbles16,
                    packed_stage, tid, tile_row,
                    stage_block + StageBlocks, n_in);
                commit_async_copies();
            }
        }
        int8_t * sx = projection ? sx_up : sx_gate;
        const int8_t * sy_half = sy + token_half * ActivationBytes;
#pragma unroll
        for (uint32_t qblock = 0; qblock < StageBlocks; ++qblock) {
            int af[2][4];
            float d4[2][2];
#pragma unroll
            for (uint32_t row_fragment = 0; row_fragment < 2;
                 ++row_fragment) {
                const uint32_t local_row0 = (compute_warp >> 1) * 32
                    + row_fragment * 16;
                load_a_m16n8k32(
                    af[row_fragment], sx + local_row0 * WeightStride
                        + qblock * 32,
                    WeightStride);
#pragma unroll
                for (uint32_t scale_item = 0; scale_item < 2;
                     ++scale_item) {
                    const uint32_t local_row = local_row0
                        + accumulator_row(lane, scale_item * 2);
                    d4[row_fragment][scale_item] =
                        reinterpret_cast<const float *>(
                            sx + local_row * WeightStride
                                + WeightValues)[qblock];
                }
            }
#pragma unroll
            for (uint32_t token_group = 0; token_group < 4;
                 ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
                    const uint32_t local_token0 = token_group * 32
                        + (compute_warp & 1) * 16 + token_fragment * 8;
                    if (local_token0 >= active_half_tokens) continue;
                    int bf[2];
                    load_ready_b_m16n8k32(
                        bf, sy_half, local_token0, qblock);
                    float d8[2];
#pragma unroll
                    for (uint32_t scale_item = 0; scale_item < 2;
                         ++scale_item) {
                        d8[scale_item] = load_ready_d8(
                            sy_half, local_token0
                                + accumulator_token(lane, scale_item),
                            qblock);
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
                            partial[sum_index] += float(cf[item])
                                * d4[row_fragment][item / 2]
                                * d8[item % 2];
                        }
                    }
                }
            }
        }
        if constexpr (PrefetchPacked) {
            if (has_next_stage) wait_async_copies();
        }
        __syncthreads();
    }

    if (projection == 0) {
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
                        const uint32_t local_row = (compute_warp >> 1) * 32
                            + row_fragment * 16
                            + accumulator_row(lane, item);
                        const uint32_t half_local_token = token_group * 32
                            + (compute_warp & 1) * 16 + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t local_token =
                            half_base + half_local_token;
                        const uint32_t token = tile_token + local_token;
                        if (token < n_tok) {
                            const uint32_t sum_index =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + item;
                            if constexpr (SharedGate) {
                                gate_tile[local_token * Rows + local_row] =
                                    partial[sum_index];
                            } else {
                                dst[uint64_t(token) * dst_stride
                                    + tile_row + local_row] =
                                        partial[sum_index];
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    if (projection == 1) {
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
                        const uint32_t local_row = (compute_warp >> 1) * 32
                            + row_fragment * 16
                            + accumulator_row(lane, item);
                        const uint32_t half_local_token = token_group * 32
                            + (compute_warp & 1) * 16 + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t local_token =
                            half_base + half_local_token;
                        const uint32_t token = tile_token + local_token;
                        if (token < n_tok) {
                            const uint32_t sum_index =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + item;
                            float * slot = dst + uint64_t(token) * dst_stride
                                + tile_row + local_row;
                            const float gate = SharedGate
                                ? gate_tile[local_token * Rows + local_row]
                                : *slot;
                            const float result =
                                cuda_gelu(gate) * partial[sum_index];
                            // Dense G remains the authoritative buffer contract.
                            // The fused Q8 sidecar is optional acceleration for the
                            // immediate down projection, not a replacement for G:
                            // probes, fallback routes and cache misses may still read
                            // the dense buffer.
                            if constexpr (WriteDense) {
                                *slot = result;
                            }
                            if constexpr (QuantizeOutput) {
                                // The up warp already owns all 32 rows required by
                                // one Q8 scale for each of its tokens. Reuse the
                                // accumulator registers for the exact dense result;
                                // the direct Q8 epilogue below avoids a shared-tile
                                // round trip and a second CTA barrier.
                                partial[sum_index] = result;
                            }
                        }
                    }
                }
            }
        }
    }
    if constexpr (QuantizeOutput) {
        static_assert(SharedGate && Rows == 64,
            "MMA-ready fused Q8 requires the R64 shared-gate tile");
        static_assert(TokenHalves == 1,
            "MMA-ready fused Q8 T256 requires a separate ownership proof");
        if (projection == 1) {
            using namespace
                imparo_q8_mma_ready_a0_v1_authority_lab;
            const uint32_t lane_row = lane >> 2;
            const uint32_t lane_token_pair = lane & 3u;
            const uint32_t row_block = tile_row / kValuesPerQBlock
                + (compute_warp >> 1);
            const uint32_t group = row_block / kQBlocksPerGroup;
            const uint32_t qblock = row_block % kQBlocksPerGroup;
#pragma unroll
            for (uint32_t token_group = 0; token_group < 4;
                 ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
                    const uint32_t local_token0 = token_group * 32
                        + (compute_warp & 1) * 16
                        + token_fragment * 8 + lane_token_pair * 2;
                    const uint32_t token0 = tile_token + local_token0;
                    const uint32_t token1 = token0 + 1;
                    const bool active0 = token0 < n_tok;
                    const bool active1 = token1 < n_tok;
                    float amax0 = 0.0f;
                    float amax1 = 0.0f;
#pragma unroll
                    for (uint32_t row_fragment = 0; row_fragment < 2;
                         ++row_fragment) {
#pragma unroll
                        for (uint32_t row_half = 0; row_half < 2;
                             ++row_half) {
                            const uint32_t sum_base =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + row_half * 2;
                            amax0 = fmaxf(
                                amax0, fabsf(partial[sum_base]));
                            amax1 = fmaxf(
                                amax1, fabsf(partial[sum_base + 1]));
                        }
                    }
#pragma unroll
                    for (int offset = 4; offset <= 16; offset <<= 1) {
                        amax0 = fmaxf(amax0, __shfl_xor_sync(
                            0xffffffffu, amax0, offset, 32));
                        amax1 = fmaxf(amax1, __shfl_xor_sync(
                            0xffffffffu, amax1, offset, 32));
                    }
                    const float d_inv0 = active0 ? 127.0f / amax0 : 0.0f;
                    const float d_inv1 = active1 ? 127.0f / amax1 : 0.0f;
                    const uint32_t token_tile = token0 / kTokenTile;
                    const uint32_t token_in_tile = token0 % kTokenTile;
#pragma unroll
                    for (uint32_t row_fragment = 0; row_fragment < 2;
                         ++row_fragment) {
#pragma unroll
                        for (uint32_t row_half = 0; row_half < 2;
                             ++row_half) {
                            const uint32_t sum_base =
                                (((token_group * 2 + token_fragment) * 2
                                    + row_fragment) * 4) + row_half * 2;
                            const int q0 = active0
                                ? int(int8_t(roundf(
                                    partial[sum_base] * d_inv0))) : 0;
                            const int q1 = active1
                                ? int(int8_t(roundf(
                                    partial[sum_base + 1] * d_inv1))) : 0;
                            const int q0_next = __shfl_down_sync(
                                0xffffffffu, q0, 4, 32);
                            const int q1_next = __shfl_down_sync(
                                0xffffffffu, q1, 4, 32);
                            if ((lane_row & 1u) == 0) {
                                const uint32_t row_in_block =
                                    row_fragment * 16 + row_half * 8
                                    + lane_row;
                                const uint16_t pair0 = uint16_t(uint8_t(q0))
                                    | (uint16_t(uint8_t(q0_next)) << 8);
                                const uint16_t pair1 = uint16_t(uint8_t(q1))
                                    | (uint16_t(uint8_t(q1_next)) << 8);
                                const uint64_t q16_base = quant_u16_index(
                                    output_token_tiles, group, token_tile,
                                    qblock, row_in_block / 2,
                                    token_in_tile);
                                reinterpret_cast<uint32_t *>(
                                    output_quant_u16 + q16_base)[0] =
                                        uint32_t(pair0)
                                        | (uint32_t(pair1) << 16);
                            }
                        }
                    }
                    if (lane_row == 0) {
                        const float d0 = active0 ? 1.0f / d_inv0 : 0.0f;
                        const float d1 = active1 ? 1.0f / d_inv1 : 0.0f;
                        output_d8_sideplane[scale_index(
                            output_token_tiles, group, token_tile,
                            qblock, token_in_tile)] =
                                __half2float(__float2half(d0));
                        output_d8_sideplane[scale_index(
                            output_token_tiles, group, token_tile,
                            qblock, token_in_tile + 1)] =
                                __half2float(__float2half(d1));
                    }
                }
            }
        }
    }
#else
    (void)gate_scales16; (void)gate_nibbles16;
    (void)up_scales16; (void)up_nibbles16;
    (void)quant_u16; (void)d8_sideplane; (void)token_tiles;
    (void)dst; (void)output_quant_u16; (void)output_d8_sideplane;
    (void)output_token_tiles; (void)n_in; (void)n_out; (void)n_tok;
    (void)dst_stride;
#endif
}

// Wide-token cooperative gate/up candidate.  The established R64xT128 CTA
// reloads the complete gate/up weight tile once for every 128 prompt tokens.
// TokenHalves=2 is the original R32xT256 weight-reuse screen. TokenHalves=1 is
// the complementary occupancy screen: R32xT128 keeps only 32 accumulators per
// thread and trades duplicate activation staging for a materially smaller
// register footprint. Both variants retain the exact K32 accumulation order.
template <uint32_t TokenHalves, bool SidecarDirectQ8 = false>
__launch_bounds__(256, 2)
__global__ void q8_ready_paircta_r32_t256(
        const uint16_t * __restrict__ gate_scales16,
        const uint4 * __restrict__ gate_nibbles16,
        const uint16_t * __restrict__ up_scales16,
        const uint4 * __restrict__ up_nibbles16,
        const uint16_t * __restrict__ quant_u16,
        const float * __restrict__ d8_sideplane, uint32_t token_tiles,
        float * __restrict__ dst, uint32_t n_in, uint32_t n_out,
        uint32_t n_tok, uint32_t dst_stride,
        uint16_t * __restrict__ output_quant_u16,
        float * __restrict__ output_d8_sideplane,
        uint32_t output_token_tiles) {
#if __CUDA_ARCH__ >= 800
    using namespace imparo_sm80_mmq;
    constexpr uint32_t Rows = 32;
    static_assert(TokenHalves == 1 || TokenHalves == 2,
                  "R32 pair-CTA supports T128 or T256 only");
    static_assert(!SidecarDirectQ8 || TokenHalves == 1,
                  "R32 direct-Q8 is isolated to the T128 sidecar producer");
    constexpr uint32_t WideTokens = TokenHalves * kTokens;
    constexpr uint32_t StageBlocks = kPackKBlocks;
    constexpr uint32_t WeightValues = StageBlocks * 32;
    constexpr uint32_t WeightStride = kHalfKWeightStride;
    extern __shared__ __align__(16) uint8_t storage[];
    int8_t * sx_gate = reinterpret_cast<int8_t *>(storage);
    int8_t * sx_up = sx_gate + Rows * WeightStride;
    int8_t * sy = sx_up + Rows * WeightStride;
    // The final 16 KiB tile aliases the now-dead K-stage storage. Gate warps
    // publish f32 values, then up warps replace them with GELU(gate)*up.
    float * sidecar_tile = reinterpret_cast<float *>(storage);
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t projection = warp / 4;
    const uint32_t local_warp = warp % 4;
    const uint32_t blocks = n_in / 32;
    const uint32_t ntx = (n_tok + WideTokens - 1) / WideTokens;
    const uint32_t tile_row = (blockIdx.x / ntx) * Rows;
    const uint32_t tile_token = (blockIdx.x % ntx) * WideTokens;
    const uint32_t active_tokens = tile_token < n_tok
        ? min(WideTokens, n_tok - tile_token) : 0;
    const uint64_t first_tile = tile_token / kReadyTokenTile;
    const uint4 * quant_cursor = reinterpret_cast<const uint4 *>(quant_u16)
        + first_tile * kReadyQuantVectorsPerTokenTile;
    const uint4 * scale_cursor = reinterpret_cast<const uint4 *>(d8_sideplane)
        + first_tile * kReadyScaleVectorsPerTokenTile;
    const uint64_t quant_group_stride =
        uint64_t(token_tiles) * kReadyQuantVectorsPerTokenTile;
    const uint64_t scale_group_stride =
        uint64_t(token_tiles) * kReadyScaleVectorsPerTokenTile;
    constexpr uint64_t quant_half_stride =
        uint64_t(kReadyTokenTilesPerCta)
        * kReadyQuantVectorsPerTokenTile;
    constexpr uint64_t scale_half_stride =
        uint64_t(kReadyTokenTilesPerCta)
        * kReadyScaleVectorsPerTokenTile;
    float partial[32 * TokenHalves];
#pragma unroll
    for (uint32_t item = 0; item < 32 * TokenHalves; ++item) {
        partial[item] = 0.0f;
    }

    for (uint32_t stage_block = 0; stage_block < blocks;
         stage_block += StageBlocks) {
        stage_ready_activation_cursor_async(
            quant_cursor, scale_cursor, sy, tid);
        commit_async_copies();

        constexpr uint32_t records_per_projection = Rows * StageBlocks;
        constexpr uint32_t packed_block_count =
            2 * records_per_projection;
        for (uint32_t linear = tid; linear < packed_block_count;
             linear += kWarps * 32) {
            const uint32_t weight_projection =
                linear / records_per_projection;
            const uint32_t record_linear =
                linear % records_per_projection;
            const uint32_t qblock = record_linear % StageBlocks;
            const uint32_t local_row = record_linear / StageBlocks;
            const uint32_t row = tile_row + local_row;
            const uint32_t kb = stage_block + qblock;
            const uint16_t * scales16 = weight_projection
                ? up_scales16 : gate_scales16;
            const uint4 * nibbles16 = weight_projection
                ? up_nibbles16 : gate_nibbles16;
            int8_t * sx = weight_projection ? sx_up : sx_gate;
            const uint64_t record = packed_record_index(row, kb, n_in);
            const uint4 packed4 = nibbles16[record];
            const uint16_t scale_bits = scales16[record];
            const __half scale_half =
                *reinterpret_cast<const __half *>(&scale_bits);
            int8_t * out = sx + local_row * WeightStride + qblock * 32;
            const uint32_t packed_words[4] = {
                packed4.x, packed4.y, packed4.z, packed4.w};
#pragma unroll
            for (uint32_t word = 0; word < 4; ++word) {
                const uint32_t packed = packed_words[word];
                reinterpret_cast<int *>(out)[word] =
                    unpack_q4_nibbles(packed);
                reinterpret_cast<int *>(out + 16)[word] =
                    unpack_q4_nibbles(packed >> 4);
            }
            reinterpret_cast<float *>(
                sx + local_row * WeightStride + WeightValues)[qblock] =
                    __half2float(scale_half);
        }

        wait_async_copies();
        __syncthreads();
        int8_t * sx = projection ? sx_up : sx_gate;
#pragma unroll
        for (uint32_t token_half = 0; token_half < TokenHalves;
             ++token_half) {
            if (token_half > 0) {
                stage_ready_activation_cursor_async(
                    quant_cursor + token_half * quant_half_stride,
                    scale_cursor + token_half * scale_half_stride,
                    sy, tid);
                commit_async_copies();
                wait_async_copies();
                __syncthreads();
            }
            const uint32_t half_base = token_half * kTokens;
            const uint32_t active_half_tokens = active_tokens > half_base
                ? min(kTokens, active_tokens - half_base) : 0;
#pragma unroll
            for (uint32_t qblock = 0; qblock < StageBlocks; ++qblock) {
                int af[4];
                float d4[2];
                const uint32_t local_row0 =
                    (local_warp >> 1) * 16;
                load_a_m16n8k32(
                    af, sx + local_row0 * WeightStride + qblock * 32,
                    WeightStride);
#pragma unroll
                for (uint32_t scale_item = 0; scale_item < 2;
                     ++scale_item) {
                    const uint32_t local_row = local_row0
                        + accumulator_row(lane, scale_item * 2);
                    d4[scale_item] = reinterpret_cast<const float *>(
                        sx + local_row * WeightStride
                            + WeightValues)[qblock];
                }
#pragma unroll
                for (uint32_t token_group = 0; token_group < 4;
                     ++token_group) {
#pragma unroll
                    for (uint32_t token_fragment = 0; token_fragment < 2;
                         ++token_fragment) {
                        const uint32_t local_token0 = token_group * 32
                            + (local_warp & 1) * 16
                            + token_fragment * 8;
                        if (local_token0 >= active_half_tokens) continue;
                        int bf[2];
                        load_ready_b_m16n8k32(
                            bf, sy, local_token0, qblock);
                        float d8[2];
#pragma unroll
                        for (uint32_t scale_item = 0; scale_item < 2;
                             ++scale_item) {
                            d8[scale_item] = load_ready_d8(
                                sy, local_token0
                                    + accumulator_token(lane, scale_item),
                                qblock);
                        }
                        int cf[4] = {};
                        mma_m16n8k32(cf, af, bf);
#pragma unroll
                        for (uint32_t item = 0; item < 4; ++item) {
                            const uint32_t sum_index = token_half * 32
                                + (token_group * 2 + token_fragment) * 4
                                + item;
                            partial[sum_index] += float(cf[item])
                                * d4[item / 2] * d8[item % 2];
                        }
                    }
                }
            }
            __syncthreads();
        }
        quant_cursor += quant_group_stride;
        scale_cursor += scale_group_stride;
    }

    if (projection == 0) {
#pragma unroll
        for (uint32_t token_half = 0; token_half < TokenHalves;
             ++token_half) {
#pragma unroll
            for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t local_row =
                            (local_warp >> 1) * 16
                            + accumulator_row(lane, item);
                        const uint32_t local_token = token_half * kTokens
                            + token_group * 32 + (local_warp & 1) * 16
                            + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t token = tile_token + local_token;
                        if (token < n_tok) {
                            const uint32_t sum_index = token_half * 32
                                + (token_group * 2 + token_fragment) * 4
                                + item;
                            if constexpr (SidecarDirectQ8) {
                                sidecar_tile[local_token * Rows + local_row] =
                                    partial[sum_index];
                            } else {
                                dst[uint64_t(token) * dst_stride
                                    + tile_row + local_row] =
                                        partial[sum_index];
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    if (projection == 1) {
#pragma unroll
        for (uint32_t token_half = 0; token_half < TokenHalves;
             ++token_half) {
#pragma unroll
            for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t local_row =
                            (local_warp >> 1) * 16
                            + accumulator_row(lane, item);
                        const uint32_t local_token = token_half * kTokens
                            + token_group * 32 + (local_warp & 1) * 16
                            + token_fragment * 8
                            + accumulator_token(lane, item);
                        const uint32_t token = tile_token + local_token;
                        if (token < n_tok) {
                            const uint32_t sum_index = token_half * 32
                                + (token_group * 2 + token_fragment) * 4
                                + item;
                            if constexpr (SidecarDirectQ8) {
                                const uint32_t tile_index =
                                    local_token * Rows + local_row;
                                sidecar_tile[tile_index] = cuda_gelu(
                                    sidecar_tile[tile_index])
                                    * partial[sum_index];
                            } else {
                                float * slot = dst
                                    + uint64_t(token) * dst_stride
                                    + tile_row + local_row;
                                *slot = cuda_gelu(*slot) * partial[sum_index];
                            }
                        }
                    }
                }
            }
        }
    }
    if constexpr (SidecarDirectQ8) {
        static_assert(Rows == 32 && TokenHalves == 1,
                      "R32 direct-Q8 owns exactly one K32 output block");
        __syncthreads();
        using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
        const uint32_t row = lane;
        const uint32_t row_block = tile_row / kValuesPerQBlock;
        const uint32_t group = row_block / kQBlocksPerGroup;
        const uint32_t qblock = row_block % kQBlocksPerGroup;
        for (uint32_t local_token = warp; local_token < kTokens;
             local_token += kWarps) {
            const uint32_t token = tile_token + local_token;
            const bool active = token < n_tok;
            const float value = active
                ? sidecar_tile[local_token * Rows + row] : 0.0f;
            float amax = active ? fabsf(value) : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(
                    0xffffffffu, amax, offset, 32));
            }
            const float d_inv = active ? 127.0f / amax : 0.0f;
            const int q = active
                ? int(int8_t(roundf(value * d_inv))) : 0;
            const int q_next = __shfl_down_sync(
                0xffffffffu, q, 1, 32);
            const uint32_t token_tile = token / kTokenTile;
            const uint32_t token_in_tile = token % kTokenTile;
            if ((row & 1u) == 0) {
                const uint16_t pair = uint16_t(uint8_t(q))
                    | (uint16_t(uint8_t(q_next)) << 8);
                const uint64_t q16_base = quant_u16_index(
                    output_token_tiles, group, token_tile, qblock,
                    row / 2, token_in_tile);
                output_quant_u16[q16_base] = pair;
            }
            if (row == 0) {
                const float d = active ? 1.0f / d_inv : 0.0f;
                output_d8_sideplane[scale_index(
                    output_token_tiles, group, token_tile,
                    qblock, token_in_tile)] =
                        __half2float(__float2half(d));
            }
        }
    }
#else
    (void)gate_scales16; (void)gate_nibbles16;
    (void)up_scales16; (void)up_nibbles16;
    (void)quant_u16; (void)d8_sideplane; (void)token_tiles;
    (void)dst; (void)n_in; (void)n_out; (void)n_tok;
    (void)dst_stride; (void)output_quant_u16;
    (void)output_d8_sideplane; (void)output_token_tiles;
#endif
}

constexpr uint32_t kReadyPairCtaR32SharedBytes =
    2 * 32 * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;

inline bool configure_q8_ready_paircta_r32_sidecar_q8() {
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r32_t256<1, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaR32SharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    return configured;
}

inline bool launch_q8_ready_paircta_r32_sidecar_q8(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst,
        uint16_t * output_quant_u16, float * output_d8_sideplane,
        uint32_t output_token_tiles, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst)
        | reinterpret_cast<uintptr_t>(output_quant_u16)
        | reinterpret_cast<uintptr_t>(output_d8_sideplane);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || !output_quant_u16 || !output_d8_sideplane
        || (bits & 15u) != 0 || !token_tiles || !output_token_tiles
        || !n_in || !n_out || !n_tok || n_tok > 512
        || uint64_t(token_tiles) * kReadyTokenTile < n_tok
        || uint64_t(output_token_tiles) * kReadyTokenTile < n_tok
        // Each R32 CTA stages and publishes a complete 128-token tile, even
        // when the logical tail is shorter.  Require the physical sidecars to
        // carry that padding rather than accepting only ceil(n_tok / 8).
        || uint64_t(token_tiles)
            < ((uint64_t(n_tok) + kTokens - 1) / kTokens)
                * (kTokens / kReadyTokenTile)
        || uint64_t(output_token_tiles)
            < ((uint64_t(n_tok) + kTokens - 1) / kTokens)
                * (kTokens / kReadyTokenTile)
        || n_in % (32 * kPackKBlocks) != 0
        // The published MMA-ready output groups K in 128-value records.  The
        // current FFN selector already guarantees this, but keep the launcher
        // independently fail-closed so a future caller cannot create a
        // partially addressable sidecar.
        || n_out % (32 * kPackKBlocks) != 0 || dst_stride < n_out) {
        return false;
    }
    if (!configure_q8_ready_paircta_r32_sidecar_q8()) return false;
    const uint64_t grid = uint64_t(n_out / 32)
        * ((uint64_t(n_tok) + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r32_t256<1, true>
        <<<uint32_t(grid), dim3(32, kWarps),
        kReadyPairCtaR32SharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            n_in, n_out, n_tok, dst_stride,
            output_quant_u16, output_d8_sideplane, output_token_tiles);
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch_q8_ready_paircta_t256(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % 32 != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r32_t256<2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaR32SharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    constexpr uint32_t wide_tokens = 2 * kTokens;
    const uint64_t grid = uint64_t(n_out / 32)
        * ((n_tok + wide_tokens - 1) / wide_tokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r32_t256<2><<<uint32_t(grid), dim3(32, kWarps),
        kReadyPairCtaR32SharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst, n_in, n_out,
            n_tok, dst_stride, nullptr, nullptr, 0);
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch_q8_ready_paircta_r32_t128(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % 32 != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r32_t256<1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaR32SharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    const uint64_t grid = uint64_t(n_out / 32)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r32_t256<1><<<uint32_t(grid), dim3(32, kWarps),
        kReadyPairCtaR32SharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst, n_in, n_out,
            n_tok, dst_stride, nullptr, nullptr, 0);
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch_q8_ready_paircta_r128(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % 128 != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r64<128, false, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaR128SharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    const uint64_t grid = uint64_t(n_out / 128)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r64<128, false, false, false>
        <<<uint32_t(grid), dim3(32, 16),
        kReadyPairCtaR128SharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            nullptr, nullptr, 0, n_in, n_out,
            n_tok, dst_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch_q8_ready_paircta_shared_gate(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % kReadyCompactRows != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r64<64, true, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaSharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    const uint64_t grid = uint64_t(n_out / kReadyCompactRows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r64<64, true, false, false>
        <<<uint32_t(grid), dim3(32, kWarps),
        kReadyPairCtaSharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            nullptr, nullptr, 0, n_in, n_out,
            n_tok, dst_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

// R64xT256 weight-reuse screen. Sixteen warps keep the same resident warp
// count as two established R64xT128 CTAs, but one CTA stages each gate/up
// weight tile for two token halves. The full-width gate tile reuses shared
// storage only after the K loop has finished.
inline bool launch_q8_ready_paircta_shared_gate_t256(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % kReadyCompactRows != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r64<64, true, false, false, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaT256SharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    constexpr uint32_t wide_tokens = 2 * kTokens;
    const uint64_t grid = uint64_t(n_out / kReadyCompactRows)
        * ((n_tok + wide_tokens - 1) / wide_tokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r64<64, true, false, false, 2>
        <<<uint32_t(grid), dim3(32, 16),
        kReadyPairCtaT256SharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            nullptr, nullptr, 0, n_in, n_out,
            n_tok, dst_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <bool PrefetchPacked = false, bool WriteDense = true>
inline bool configure_q8_ready_paircta_shared_gate_q8() {
    static const bool configured = [] {
        constexpr uint32_t shared_bytes = PrefetchPacked
            ? kReadyPairCtaPrefetchSharedBytes
            : kReadyPairCtaSharedBytes;
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r64<64, true, PrefetchPacked, true, 1, false,
                WriteDense>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    return configured;
}

template <bool PrefetchPacked = false, bool WriteDense = true>
inline bool launch_q8_ready_paircta_shared_gate_q8(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst,
        uint16_t * output_quant_u16, float * output_d8_sideplane,
        uint32_t output_token_tiles, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst)
        | reinterpret_cast<uintptr_t>(output_quant_u16)
        | reinterpret_cast<uintptr_t>(output_d8_sideplane);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || !output_quant_u16 || !output_d8_sideplane
        || (bits & 15u) != 0 || !token_tiles || !output_token_tiles
        || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % kPackM != 0 || dst_stride < n_out
        || uint64_t(output_token_tiles) * kReadyTokenTile < n_tok) {
        return false;
    }
    if (!configure_q8_ready_paircta_shared_gate_q8<
            PrefetchPacked, WriteDense>()) return false;
    const uint64_t grid = uint64_t(n_out / kReadyCompactRows)
        * ((uint64_t(n_tok) + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    constexpr uint32_t shared_bytes = PrefetchPacked
        ? kReadyPairCtaPrefetchSharedBytes
        : kReadyPairCtaSharedBytes;
    q8_ready_paircta_r64<64, true, PrefetchPacked, true, 1, false,
        WriteDense>
        <<<uint32_t(grid), dim3(32, kWarps),
        shared_bytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            output_quant_u16, output_d8_sideplane, output_token_tiles,
            n_in, n_out, n_tok, dst_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <bool PackedByteSubtract = false>
inline bool launch_q8_ready_paircta_shared_gate_prefetch(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % kReadyCompactRows != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r64<64, true, true, false, 1,
                PackedByteSubtract>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaPrefetchSharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    const uint64_t grid = uint64_t(n_out / kReadyCompactRows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r64<64, true, true, false, 1, PackedByteSubtract>
        <<<uint32_t(grid), dim3(32, kWarps),
        kReadyPairCtaPrefetchSharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            nullptr, nullptr, 0, n_in, n_out,
            n_tok, dst_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch_q8_ready_paircta(
        const uint16_t * gate_scales16, const uint4 * gate_nibbles16,
        const uint16_t * up_scales16, const uint4 * up_nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t dst_stride,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(gate_scales16)
        | reinterpret_cast<uintptr_t>(gate_nibbles16)
        | reinterpret_cast<uintptr_t>(up_scales16)
        | reinterpret_cast<uintptr_t>(up_nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!gate_scales16 || !gate_nibbles16 || !up_scales16
        || !up_nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0
        || n_out % kReadyCompactRows != 0 || dst_stride < n_out) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t dynamic = cudaFuncSetAttribute(
            q8_ready_paircta_r64<64, false, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyPairCtaSharedBytes);
        if (dynamic != cudaSuccess) cudaGetLastError();
        return dynamic == cudaSuccess;
    }();
    if (!configured) return false;
    const uint64_t grid = uint64_t(n_out / kReadyCompactRows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    q8_ready_paircta_r64<64, false, false, false>
        <<<uint32_t(grid), dim3(32, kWarps),
        kReadyPairCtaSharedBytes, stream>>>(
            gate_scales16, gate_nibbles16, up_scales16, up_nibbles16,
            quant_u16, d8_sideplane, token_tiles, dst,
            nullptr, nullptr, 0, n_in, n_out,
            n_tok, dst_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

enum class DirectRows : uint32_t {
    Environment = 0,
    Rows128 = 128,
    Rows80 = 80,
    Rows96 = 96,
    // Schedule sentinel: 0x60 rows and 0x18 (24) warps.
    Rows96Warp24 = 0x6018,
    Rows64 = 64,
};

// One authority for laboratory spelling, profiler identity and launch geometry.
// Candidate discovery may add descriptors here, but a descriptor alone never
// enters the tuning space or changes the receipt-backed production selector.
struct DirectScheduleDescriptor {
    DirectRows value;
    const char * selector;
    const char * profile_label;
    uint32_t tile_rows;
    uint32_t warps;
    uint32_t dynamic_shared_bytes;
};

static constexpr DirectScheduleDescriptor kDirectSchedules[] = {
    {DirectRows::Rows128, "r128", "ffn_sidecar_down_full_k_r128",
        imparo_sm80_mmq::kRows, imparo_sm80_mmq::kWarps,
        imparo_sm80_mmq::kHalfKSharedBytes},
    {DirectRows::Rows80, "r80", "ffn_sidecar_down_full_k_r80",
        kReadyR80Rows, kReadyR80Warps, kReadyR80SharedBytes},
    {DirectRows::Rows96, "r96", "ffn_sidecar_down_full_k_r96",
        kReadyBalancedRows, kReadyBalancedWarps, kReadyBalancedSharedBytes},
    {DirectRows::Rows96Warp24, "r96w24",
        "ffn_sidecar_down_full_k_r96_w24", kReadyBalancedRows,
        kReadyPartitionedWarps, kReadyBalancedSharedBytes},
    {DirectRows::Rows64, "r64", "ffn_sidecar_down_full_k_r64",
        kReadyCompactRows, imparo_sm80_mmq::kWarps,
        kReadyCompactSharedBytes},
};

inline const DirectScheduleDescriptor * find_direct_schedule(DirectRows value) {
    for (const auto & schedule : kDirectSchedules) {
        if (schedule.value == value) return &schedule;
    }
    return nullptr;
}

inline DirectRows parse_direct_schedule(const char * selector) {
    if (!selector) return DirectRows::Environment;
    for (const auto & schedule : kDirectSchedules) {
        if (std::strcmp(selector, schedule.selector) == 0) {
            return schedule.value;
        }
    }
    return DirectRows::Environment;
}

inline const char * direct_schedule_profile_label(DirectRows value) {
    const auto * schedule = find_direct_schedule(value);
    return schedule ? schedule->profile_label
                    : "ffn_sidecar_down_full_k_invalid";
}

inline bool configure_q8_ready_direct_r128() {
    using namespace imparo_sm80_mmq;
    static const bool configured = [] {
        const cudaError_t plain = cudaFuncSetAttribute(
            q8_ready_full_tile_direct<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kHalfKSharedBytes);
        const cudaError_t direct = cudaFuncSetAttribute(
            q8_ready_full_tile_direct<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kHalfKSharedBytes);
        if (plain != cudaSuccess || direct != cudaSuccess) {
            cudaGetLastError();
        }
        return plain == cudaSuccess && direct == cudaSuccess;
    }();
    return configured;
}

inline bool configure_q8_ready_direct_r64() {
    using namespace imparo_sm80_mmq;
    static const bool configured = [] {
        const cudaError_t plain = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r64<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyCompactSharedBytes);
        const cudaError_t direct = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r64<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyCompactSharedBytes);
        const cudaError_t plain_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r64<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        const cudaError_t direct_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r64<true>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (plain != cudaSuccess || direct != cudaSuccess
                || plain_carveout != cudaSuccess
                || direct_carveout != cudaSuccess) {
            cudaGetLastError();
        }
        return plain == cudaSuccess && direct == cudaSuccess
            && plain_carveout == cudaSuccess
            && direct_carveout == cudaSuccess;
    }();
    return configured;
}

inline bool configure_q8_ready_direct_r96() {
    static const bool configured = [] {
        const cudaError_t plain = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyBalancedSharedBytes);
        const cudaError_t direct = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyBalancedSharedBytes);
        const cudaError_t plain_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        const cudaError_t direct_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96<true>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (plain != cudaSuccess || direct != cudaSuccess
                || plain_carveout != cudaSuccess
                || direct_carveout != cudaSuccess) {
            cudaGetLastError();
        }
        return plain == cudaSuccess && direct == cudaSuccess
            && plain_carveout == cudaSuccess
            && direct_carveout == cudaSuccess;
    }();
    return configured;
}

inline bool configure_q8_ready_direct_r96_w24() {
    static const bool configured = [] {
        const cudaError_t plain = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96_w24<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyBalancedSharedBytes);
        const cudaError_t direct = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96_w24<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyBalancedSharedBytes);
        const cudaError_t plain_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96_w24<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        const cudaError_t direct_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r96_w24<true>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (plain != cudaSuccess || direct != cudaSuccess
                || plain_carveout != cudaSuccess
                || direct_carveout != cudaSuccess) {
            cudaGetLastError();
        }
        return plain == cudaSuccess && direct == cudaSuccess
            && plain_carveout == cudaSuccess
            && direct_carveout == cudaSuccess;
    }();
    return configured;
}

inline bool configure_q8_ready_direct_r80() {
    static const bool configured = [] {
        const cudaError_t plain = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r80<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyR80SharedBytes);
        const cudaError_t direct = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r80<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kReadyR80SharedBytes);
        const cudaError_t plain_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r80<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        const cudaError_t direct_carveout = cudaFuncSetAttribute(
            q8_ready_full_tile_direct_r80<true>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (plain != cudaSuccess || direct != cudaSuccess
                || plain_carveout != cudaSuccess
                || direct_carveout != cudaSuccess) {
            cudaGetLastError();
        }
        return plain == cudaSuccess && direct == cudaSuccess
            && plain_carveout == cudaSuccess
            && direct_carveout == cudaSuccess;
    }();
    return configured;
}

inline bool configure_q8_ready_direct_schedule(DirectRows rows) {
    switch (rows) {
        case DirectRows::Rows128:
            return configure_q8_ready_direct_r128();
        case DirectRows::Rows80:
            return configure_q8_ready_direct_r80();
        case DirectRows::Rows96:
            return configure_q8_ready_direct_r96();
        case DirectRows::Rows96Warp24:
            return configure_q8_ready_direct_r96_w24();
        case DirectRows::Rows64:
            return configure_q8_ready_direct_r64();
        case DirectRows::Environment:
            return false;
    }
    return false;
}

inline bool launch_q8_ready_direct(
        const uint16_t * scales16, const uint4 * nibbles16,
        const uint16_t * quant_u16, const float * d8_sideplane,
        uint32_t token_tiles, float * dst, uint32_t n_in,
        uint32_t n_out, uint32_t n_tok, uint32_t work_n_tok,
        uint32_t dst_stride, bool direct_epilogue, cudaStream_t stream,
        DirectRows rows = DirectRows::Environment) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(scales16)
        | reinterpret_cast<uintptr_t>(nibbles16)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst);
    if (!scales16 || !nibbles16 || !quant_u16 || !d8_sideplane || !dst
        || (bits & 15u) != 0 || !token_tiles || !n_in || !n_out || !n_tok
        || work_n_tok != n_tok
        || n_in % (32 * kPackKBlocks) != 0 || n_out % kPackM != 0
        || dst_stride < n_out) return false;
    DirectRows resolved_rows = rows;
    if (resolved_rows == DirectRows::Environment) {
        resolved_rows = std::getenv("IMPARO_CUDA_PREFILL_Q8_READY_R64_LAB")
            ? DirectRows::Rows64 : DirectRows::Rows128;
    }
    const auto * schedule = find_direct_schedule(resolved_rows);
    if (!schedule || !configure_q8_ready_direct_schedule(resolved_rows)) {
        return false;
    }
    const bool compact = resolved_rows == DirectRows::Rows64;
    const bool dense = resolved_rows == DirectRows::Rows80;
    const bool balanced = resolved_rows == DirectRows::Rows96;
    const bool partitioned = resolved_rows == DirectRows::Rows96Warp24;
    const uint32_t tile_rows = schedule->tile_rows;
    const uint64_t grid = uint64_t((n_out + tile_rows - 1) / tile_rows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    if (dense && direct_epilogue) {
        q8_ready_full_tile_direct_r80<true><<<uint32_t(grid),
            dim3(32, kReadyR80Warps), kReadyR80SharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (dense) {
        q8_ready_full_tile_direct_r80<false><<<uint32_t(grid),
            dim3(32, kReadyR80Warps), kReadyR80SharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (partitioned && direct_epilogue) {
        q8_ready_full_tile_direct_r96_w24<true><<<uint32_t(grid),
            dim3(32, kReadyPartitionedWarps),
            kReadyBalancedSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (partitioned) {
        q8_ready_full_tile_direct_r96_w24<false><<<uint32_t(grid),
            dim3(32, kReadyPartitionedWarps),
            kReadyBalancedSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (balanced && direct_epilogue) {
        q8_ready_full_tile_direct_r96<true><<<uint32_t(grid),
            dim3(32, kReadyBalancedWarps), kReadyBalancedSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (balanced) {
        q8_ready_full_tile_direct_r96<false><<<uint32_t(grid),
            dim3(32, kReadyBalancedWarps), kReadyBalancedSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (compact && direct_epilogue) {
        q8_ready_full_tile_direct_r64<true><<<uint32_t(grid),
            dim3(32, kWarps), kReadyCompactSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (compact) {
        q8_ready_full_tile_direct_r64<false><<<uint32_t(grid),
            dim3(32, kWarps), kReadyCompactSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else if (direct_epilogue) {
        q8_ready_full_tile_direct<true><<<uint32_t(grid),
            dim3(32, kWarps), kHalfKSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    } else {
        q8_ready_full_tile_direct<false><<<uint32_t(grid),
            dim3(32, kWarps), kHalfKSharedBytes, stream>>>(
                scales16, nibbles16, quant_u16, d8_sideplane,
                token_tiles, dst, n_in, n_out, n_tok, work_n_tok,
                dst_stride);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

enum class RouteDecision : uint32_t { Invalid = 0, Candidate = 1 };

inline RouteDecision validate_batched_r2_plan(
        const uint16_t *scales0, const uint4 *nibbles0,
        const uint16_t *scales1, const uint4 *nibbles1,
        const uint16_t *quant_u16, const float *d8_sideplane,
        const float *dst0, const float *dst1,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t numeric_stream_grid, uint32_t max_grid_x,
        uint64_t *tile_count_out = nullptr,
        uint64_t *flat_grid_out = nullptr) {
    using namespace imparo_sm80_mmq;
    const uintptr_t bits = reinterpret_cast<uintptr_t>(scales0)
        | reinterpret_cast<uintptr_t>(nibbles0)
        | reinterpret_cast<uintptr_t>(scales1)
        | reinterpret_cast<uintptr_t>(nibbles1)
        | reinterpret_cast<uintptr_t>(quant_u16)
        | reinterpret_cast<uintptr_t>(d8_sideplane)
        | reinterpret_cast<uintptr_t>(dst0)
        | reinterpret_cast<uintptr_t>(dst1);
    if (!scales0 || !nibbles0 || !scales1 || !nibbles1
            || !quant_u16 || !d8_sideplane || !dst0 || !dst1
            || (bits & 15u) != 0 || !n_in || !n_out || !n_tok
            || work_n_tok != n_tok
            || n_out <= n_in || numeric_stream_grid != 0
            || n_in % (32 * kPackKBlocks) != 0 || n_out % kPackM != 0
            || dst_stride < n_out || !max_grid_x) {
        return RouteDecision::Invalid;
    }
    const uint64_t tile_count =
        uint64_t(n_out / kRows) * ((n_tok + kTokens - 1) / kTokens);
    const uint64_t flat_grid = 2u * tile_count;
    if (!tile_count || flat_grid > max_grid_x) return RouteDecision::Invalid;
    if (tile_count_out) *tile_count_out = tile_count;
    if (flat_grid_out) *flat_grid_out = flat_grid;
    return RouteDecision::Candidate;
}

inline bool launch_batched_r2(
        const uint16_t *scales0, const uint4 *nibbles0,
        const uint16_t *scales1, const uint4 *nibbles1,
        const uint16_t *quant_u16, const float *d8_sideplane,
        float *dst0, float *dst1,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t numeric_stream_grid, uint32_t max_grid_x,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    uint64_t tile_count = 0, flat_grid = 0;
    if (validate_batched_r2_plan(scales0, nibbles0, scales1, nibbles1,
            quant_u16, d8_sideplane, dst0, dst1,
            n_in, n_out, n_tok, work_n_tok, dst_stride,
            numeric_stream_grid, max_grid_x, &tile_count, &flat_grid)
            != RouteDecision::Candidate) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t status = cudaFuncSetAttribute(
            q8_ready_batched_r2_interleaved_carveout100_v17<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kHalfKSharedBytes);
        if (status != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const cudaError_t carveout_status = cudaFuncSetAttribute(
            q8_ready_batched_r2_interleaved_carveout100_v17<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (carveout_status != cudaSuccess) cudaGetLastError();
        return carveout_status == cudaSuccess;
    }();
    if (!configured) return false;
    const uint32_t padded_tokens =
        ((n_tok + kTokens - 1) / kTokens) * kTokens;
    const uint32_t q8_token_tiles = padded_tokens / kReadyTokenTile;
    q8_ready_batched_r2_interleaved_carveout100_v17<false>
        <<<uint32_t(flat_grid), dim3(32, kWarps),
           kHalfKSharedBytes, stream>>>(
            scales0, nibbles0, scales1, nibbles1,
            quant_u16, d8_sideplane, q8_token_tiles,
            dst0, dst1, n_in, n_out, n_tok, work_n_tok,
            dst_stride, 0);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab
