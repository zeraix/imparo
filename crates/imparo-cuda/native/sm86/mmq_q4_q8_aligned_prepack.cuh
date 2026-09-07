#pragma once

// Laboratory SM86 equal-size Q4_0 layout prototype derived from the validated 2-A kernel lab.
//
// Selection remains exact and opt-in; transformed spans are never sent to the
// canonical raw-Q4 kernels. This is not a production weight format or selector.
// The input is a byte-neutral split of canonical Q4 blocks into a uint16 scale
// plane and a 16-byte nibble plane.  Four consecutive K blocks for one row are
// adjacent so one typed uint4 load supplies the complete nibble payload.

namespace imparo_sm86_q4_aligned_prepack {

constexpr uint32_t kPackM = 128;
constexpr uint32_t kPackKBlocks = 4;

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

__global__ void pack_q4_equal_size(
        const uint8_t * __restrict__ raw,
        uint8_t * __restrict__ packed,
        uint32_t n_in, uint32_t n_out) {
    const uint32_t blocks = n_in / 32;
    const uint64_t records = uint64_t(n_out) * blocks;
    const uint64_t logical = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (logical >= records) return;
    const uint32_t row = uint32_t(logical / blocks);
    const uint32_t kb = uint32_t(logical - uint64_t(row) * blocks);
    const uint64_t physical = packed_record_index(row, kb, n_in);
    const uint8_t * source = raw + logical * 18;
    uint8_t * scale_plane = packed;
    uint8_t * nibble_plane = packed + records * 2;
    scale_plane[physical * 2] = source[0];
    scale_plane[physical * 2 + 1] = source[1];
#pragma unroll
    for (uint32_t byte = 0; byte < 16; ++byte) {
        nibble_plane[physical * 16 + byte] = source[2 + byte];
    }
}

inline bool launch_pack_q4_equal_size(
        const uint8_t * raw, uint8_t * packed,
        uint32_t n_in, uint32_t n_out, cudaStream_t stream) {
    if (!raw || !packed || !n_in || !n_out
        || n_in % 128 != 0 || n_out % 128 != 0) return false;
    const uint64_t records = uint64_t(n_out) * (n_in / 32);
    const uint64_t grid = (records + 255) / 256;
    if (!grid || grid > uint64_t(UINT32_MAX)) return false;
    pack_q4_equal_size<<<uint32_t(grid), 256, 0, stream>>>(
        raw, packed, n_in, n_out);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <bool FullK>
__device__ __forceinline__ void compute_segment(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const BlockQ8_1Mmq * __restrict__ x,
        int8_t * sx, int8_t * sy, float (&partial)[64],
        uint32_t n_in, uint32_t n_tok, uint32_t work_n_tok,
        uint32_t blocks, uint32_t tile_row, uint32_t tile_token,
        uint32_t segment_begin, uint32_t segment_end) {
    using namespace imparo_sm80_mmq;
    constexpr uint32_t StageBlocks = kPackKBlocks;
    constexpr uint32_t WeightValues = StageBlocks * 32;
    constexpr uint32_t WeightStride = kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t active_tokens = tile_token < work_n_tok
        ? min(kTokens, work_n_tok - tile_token) : 0;
#pragma unroll
    for (uint32_t item = 0; item < 64; ++item) partial[item] = 0.0f;

    for (uint32_t stage_block = segment_begin;
         stage_block < segment_end; stage_block += StageBlocks) {
        stage_activation_async(
            x, sy, n_tok, tile_token, active_tokens, stage_block / 4, tid);
        commit_async_copies();

        // A warp consumes 32 consecutive 16-byte records.  The typed uint4 load
        // is intentional: four scalar uint32 loads do not establish a single
        // vector memory operation and may be scheduled/split independently.
        constexpr uint32_t packed_block_count = kPackM * StageBlocks;
        for (uint32_t linear = tid; linear < packed_block_count;
             linear += imparo_sm80_mmq::kWarps * 32) {
            const uint32_t qblock = linear % StageBlocks;
            const uint32_t local_row = linear / StageBlocks;
            const uint32_t row = tile_row + local_row;
            const uint32_t kb = stage_block + qblock;
            const bool valid = FullK || kb < segment_end;
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
            for (uint32_t row_fragment = 0; row_fragment < 2;
                 ++row_fragment) {
                const uint32_t local_row0 = (warp >> 1) * 32
                    + row_fragment * 16;
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
            for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 2;
                     ++token_fragment) {
                    const uint32_t local_token0 = token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8;
                    if (local_token0 >= active_tokens) continue;
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

template <bool NumericSeams, bool DirectEpilogue>
__launch_bounds__(256, 2)
__global__ void aligned_q4_full_tile(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
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
    const uint32_t tile_row = (blockIdx.x / ntx) * kRows;
    const uint32_t tile_token = (blockIdx.x % ntx) * kTokens;
    float partial[64];

    uint32_t numeric_seam = 0;
    if constexpr (NumericSeams) {
        const uint64_t total_work = uint64_t(gridDim.x) * blocks;
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
            compute_segment<false>(scales16, nibbles16, x, sx, sy, partial,
                n_in, n_tok, work_n_tok, blocks, tile_row, tile_token,
                0, first_end);
        } else {
            compute_segment<true>(scales16, nibbles16, x, sx, sy, partial,
                n_in, n_tok, work_n_tok, blocks, tile_row, tile_token,
                0, blocks);
        }
    } else {
        compute_segment<true>(scales16, nibbles16, x, sx, sy, partial,
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
            compute_segment<false>(scales16, nibbles16, x, sx, sy, partial,
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
                    const float value = NumericSeams && numeric_seam
                        ? partial[sum_index] + *slot : partial[sum_index];
                    if constexpr (DirectEpilogue) {
                        *slot = cuda_gelu(*slot) * value;
                    } else {
                        *slot = value;
                    }
                }
            }
        }
    }
#else
    (void)scales16; (void)nibbles16; (void)x; (void)dst;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride; (void)numeric_stream_grid;
#endif
}

inline bool launch_aligned_q4(
        const uint16_t * scales16, const uint4 * nibbles16,
        const BlockQ8_1Mmq * x, float * dst,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t numeric_stream_grid, bool direct_epilogue,
        cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    if (!scales16 || !nibbles16 || !x || !dst || !n_in || !n_out || !n_tok
        || n_in % (32 * kPackKBlocks) != 0 || n_out % kPackM != 0
        || dst_stride < n_out || (numeric_stream_grid && direct_epilogue)) {
        return false;
    }
    static const bool configured = [] {
        const cudaError_t plain = cudaFuncSetAttribute(
            aligned_q4_full_tile<false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kHalfKSharedBytes);
        const cudaError_t direct = cudaFuncSetAttribute(
            aligned_q4_full_tile<false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kHalfKSharedBytes);
        const cudaError_t seams = cudaFuncSetAttribute(
            aligned_q4_full_tile<true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kHalfKSharedBytes);
        if (plain != cudaSuccess || direct != cudaSuccess
            || seams != cudaSuccess) cudaGetLastError();
        return plain == cudaSuccess && direct == cudaSuccess
            && seams == cudaSuccess;
    }();
    if (!configured) return false;
    const uint32_t token_tiles = (n_tok + kTokens - 1) / kTokens;
    const uint32_t grid = (n_out / kRows) * token_tiles;
    if (numeric_stream_grid) {
        aligned_q4_full_tile<true, false><<<grid, dim3(32, kWarps),
            kHalfKSharedBytes, stream>>>(scales16, nibbles16, x, dst,
                n_in, n_out, n_tok, work_n_tok, dst_stride,
                numeric_stream_grid);
    } else {
        if (direct_epilogue) {
            aligned_q4_full_tile<false, true><<<grid, dim3(32, kWarps),
                kHalfKSharedBytes, stream>>>(scales16, nibbles16, x, dst,
                    n_in, n_out, n_tok, work_n_tok, dst_stride, 0);
        } else {
            aligned_q4_full_tile<false, false><<<grid, dim3(32, kWarps),
                kHalfKSharedBytes, stream>>>(scales16, nibbles16, x, dst,
                    n_in, n_out, n_tok, work_n_tok, dst_stride, 0);
        }
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

// Exact physical Stream-K ownership for contraction projections. This changes
// only Q4 staging: worker boundaries, K traversal, partial stores, and fixup
// order remain identical to imparo_sm80_mmq::q4_q8_1_stream.
__launch_bounds__(256, 2)
__global__ void aligned_q4_physical_stream(
        const uint16_t * __restrict__ scales16,
        const uint4 * __restrict__ nibbles16,
        const BlockQ8_1Mmq * __restrict__ x, float * __restrict__ dst,
        float * __restrict__ fixup, uint32_t n_in, uint32_t n_out,
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
    const uint32_t nty = n_out / kRows;
    const uint64_t total_work = uint64_t(ntx) * nty * blocks;
    uint64_t work =
        stream_boundary(blockIdx.x, total_work, gridDim.x, blocks);
    const uint64_t work_stop =
        stream_boundary(blockIdx.x + 1, total_work, gridDim.x, blocks);
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
            scales16, nibbles16, x, sx, sy, partial,
            n_in, n_tok, work_n_tok, blocks, tile_row, tile_token,
            segment_begin, segment_end);
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
    (void)scales16; (void)nibbles16; (void)x; (void)dst; (void)fixup;
    (void)n_in; (void)n_out; (void)n_tok; (void)work_n_tok;
    (void)dst_stride;
#endif
}

inline bool launch_aligned_q4_physical_stream(
        const uint16_t * scales16, const uint4 * nibbles16,
        const BlockQ8_1Mmq * x, float * dst, float * fixup,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t work_n_tok, uint32_t dst_stride,
        uint32_t stream_grid, cudaStream_t stream) {
    using namespace imparo_sm80_mmq;
    if (!scales16 || !nibbles16 || !x || !dst || !fixup
        || !n_in || !n_out || !n_tok || !stream_grid
        || n_in % (32 * kPackKBlocks) != 0 || n_out % kRows != 0
        || dst_stride < n_out) return false;
    static const bool configured = [] {
        const cudaError_t status = cudaFuncSetAttribute(
            aligned_q4_physical_stream,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kHalfKSharedBytes);
        if (status != cudaSuccess) cudaGetLastError();
        return status == cudaSuccess;
    }();
    if (!configured) return false;
    const uint32_t ntiles = (n_out / kRows)
        * ((n_tok + kTokens - 1) / kTokens);
    if (!ntiles) return false;
    aligned_q4_physical_stream<<<stream_grid, dim3(32, kWarps),
        kHalfKSharedBytes, stream>>>(scales16, nibbles16, x, dst, fixup,
            n_in, n_out, n_tok, work_n_tok, dst_stride);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (ntiles % stream_grid != 0) {
        q4_q8_1_stream_fixup<<<dim3(stream_grid, 4), 128, 0, stream>>>(
            dst, fixup, n_in, n_out, n_tok, dst_stride);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace imparo_sm86_q4_aligned_prepack
