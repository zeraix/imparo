#pragma once

// Strictly opt-in SM86 packed-weight pipeline for one admitted down-projection
// shape. The kernel retains the established virtual-direct-seam ownership and
// arithmetic order; only Q4 byte transport and unpack staging differ.
namespace imparo_sm86_mmq_down_pipe {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kPackedBlocks = 8;
constexpr uint32_t kPackedRowBytes = kPackedBlocks * 18;
constexpr uint32_t kPackedVectors = kPackedRowBytes / sizeof(uint4);
constexpr uint32_t kRawStageBytes =
    imparo_sm80_mmq::kRows * kPackedRowBytes;
constexpr uint32_t kPipeSharedBytes = 2 * kRawStageBytes
    + imparo_sm80_mmq::kRows * imparo_sm80_mmq::kHalfKWeightStride
    + imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;

static_assert(kPackedRowBytes % sizeof(uint4) == 0,
              "packed Q4 stage vector alignment");
static_assert(kPipeSharedBytes == 73728,
              "SM86 down-pipeline shared-memory budget");

struct DownShapeV1 {
    static constexpr uint32_t kInput = 10240;
    static constexpr uint32_t kOutput = 2560;
    static constexpr uint32_t kTokens = 449;
    static constexpr uint32_t kSms = 30;
    static constexpr uint32_t kBlocks = kInput / 32;
    static constexpr uint32_t kTokenTiles = 4;
    static constexpr uint32_t kLogicalTiles =
        (kOutput / imparo_sm80_mmq::kRows) * kTokenTiles;

    static bool supports(
            const uint8_t * w, const BlockQ8_1Mmq * x, const float * y,
            uint32_t n_in, uint32_t n_out, uint32_t n_tok,
            uint32_t epilogue, uint32_t out_stride, uint32_t row_base,
            uint32_t sm_count, uint32_t stream_k_numeric,
            uint32_t virtual_token_base, uint32_t virtual_schedule,
            uint32_t max_grid_x) {
        const uintptr_t pointers = reinterpret_cast<uintptr_t>(w)
            | reinterpret_cast<uintptr_t>(x)
            | reinterpret_cast<uintptr_t>(y);
        if (!w || !x || !y || (pointers & 15u) != 0
            || n_in != kInput || n_out != kOutput || n_tok != kTokens
            || epilogue != 0 || out_stride != kOutput || row_base != 0
            || sm_count != kSms || !stream_k_numeric || !virtual_schedule
            || virtual_token_base % imparo_sm80_mmq::kTokens != 0
            || max_grid_x < kLogicalTiles) {
            return false;
        }
        const uint32_t actual_token_tiles =
            (n_tok + imparo_sm80_mmq::kTokens - 1)
            / imparo_sm80_mmq::kTokens;
        const uint32_t actual_tiles =
            (n_out / imparo_sm80_mmq::kRows) * actual_token_tiles;
        const uint32_t waves =
            (kLogicalTiles + sm_count - 1) / sm_count;
        const uint32_t efficiency =
            100u * kLogicalTiles / (sm_count * waves);
        const uint32_t numeric_grid =
            efficiency >= 90u ? kLogicalTiles : sm_count;
        const uint64_t boundary_stride =
            uint64_t(kLogicalTiles) * kBlocks / numeric_grid;
        return actual_tiles == kLogicalTiles
            && numeric_grid == kSms
            && numeric_grid < kLogicalTiles
            && boundary_stride >= uint64_t(kBlocks) + kPackedBlocks;
    }
};

__device__ __forceinline__ void wait_async_copies_one_group() {
    asm volatile("cp.async.wait_group 1;");
}

__device__ __forceinline__ void stage_q4_raw_k256_async(
        const uint8_t * __restrict__ w, uint8_t * raw,
        uint32_t blocks, uint32_t tile_row, uint32_t stage_block,
        uint32_t tid) {
    constexpr uint32_t copies = imparo_sm80_mmq::kRows * kPackedVectors;
    for (uint32_t linear = tid; linear < copies;
         linear += imparo_sm80_mmq::kWarps * 32) {
        const uint32_t local_row = linear / kPackedVectors;
        const uint32_t vector = linear % kPackedVectors;
        const uint8_t * src = w
            + (uint64_t(tile_row + local_row) * blocks + stage_block) * 18
            + vector * sizeof(uint4);
        uint8_t * dst = raw
            + local_row * kPackedRowBytes + vector * sizeof(uint4);
        imparo_sm80_mmq::copy_global_to_shared_16(
            reinterpret_cast<uint4 *>(dst),
            reinterpret_cast<const uint4 *>(src));
    }
}

template <uint32_t Phase>
__device__ __forceinline__ void expand_raw_q4_k128(
        const uint8_t * raw, int8_t * sx, uint32_t tid) {
    static_assert(Phase < 2, "K256 has exactly two K128 phases");
    constexpr uint32_t packed_block_count =
        imparo_sm80_mmq::kRows * imparo_sm80_mmq::kHalfKBlocks;
    for (uint32_t linear = tid; linear < packed_block_count;
         linear += imparo_sm80_mmq::kWarps * 32) {
        const uint32_t qblock = linear % imparo_sm80_mmq::kHalfKBlocks;
        const uint32_t local_row = linear / imparo_sm80_mmq::kHalfKBlocks;
        const uint8_t * block = raw + local_row * kPackedRowBytes
            + (Phase * imparo_sm80_mmq::kHalfKBlocks + qblock) * 18;
        const uint16_t * qs = reinterpret_cast<const uint16_t *>(block + 2);
        int8_t * dst = sx
            + local_row * imparo_sm80_mmq::kHalfKWeightStride
            + qblock * 32;
#pragma unroll
        for (uint32_t word = 0; word < 4; ++word) {
            const uint32_t packed = uint32_t(qs[2 * word])
                | (uint32_t(qs[2 * word + 1]) << 16);
            reinterpret_cast<int *>(dst)[word] =
                imparo_sm80_mmq::unpack_q4_nibbles(packed);
            reinterpret_cast<int *>(dst + 16)[word] =
                imparo_sm80_mmq::unpack_q4_nibbles(packed >> 4);
        }
        reinterpret_cast<float *>(
            sx + local_row * imparo_sm80_mmq::kHalfKWeightStride
                + imparo_sm80_mmq::kHalfKValues)[qblock] =
            __half2float(*reinterpret_cast<const __half *>(block));
    }
}

template <uint32_t Phase>
__device__ __forceinline__ void mma_phase_k128(
        const int8_t * sx, const int8_t * sy, float (&partial)[64],
        uint32_t active_tokens, uint32_t lane, uint32_t warp) {
    static_assert(Phase < 2, "K256 has exactly two K128 phases");
#pragma unroll
    for (uint32_t qblock = 0;
         qblock < imparo_sm80_mmq::kHalfKBlocks; ++qblock) {
        int af[2][4];
        float d4[2][2];
#pragma unroll
        for (uint32_t row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const uint32_t local_row0 =
                (warp >> 1) * 32 + row_fragment * 16;
            imparo_sm80_mmq::load_a_m16n8k32(
                af[row_fragment],
                sx + local_row0 * imparo_sm80_mmq::kHalfKWeightStride
                    + qblock * 32,
                imparo_sm80_mmq::kHalfKWeightStride);
#pragma unroll
            for (uint32_t scale_item = 0; scale_item < 2; ++scale_item) {
                const uint32_t local_row = local_row0
                    + imparo_sm80_mmq::accumulator_row(lane, scale_item * 2);
                d4[row_fragment][scale_item] =
                    reinterpret_cast<const float *>(
                        sx + local_row * imparo_sm80_mmq::kHalfKWeightStride
                        + imparo_sm80_mmq::kHalfKValues)[qblock];
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
                imparo_sm80_mmq::load_b_m16n8k32(
                    bf,
                    sy + local_token0 * imparo_sm80_mmq::kActivationStride
                        + qblock * 32,
                    imparo_sm80_mmq::kActivationStride);
                float d8[2];
#pragma unroll
                for (uint32_t scale_item = 0; scale_item < 2; ++scale_item) {
                    const uint32_t local_token = local_token0
                        + imparo_sm80_mmq::accumulator_token(lane, scale_item);
                    d8[scale_item] = reinterpret_cast<const float *>(
                        sy + local_token * imparo_sm80_mmq::kActivationStride
                        + imparo_sm80_mmq::kActivationStage)[qblock];
                }
#pragma unroll
                for (uint32_t row_fragment = 0; row_fragment < 2;
                     ++row_fragment) {
                    int cf[4] = {};
                    imparo_sm80_mmq::mma_m16n8k32(cf, af[row_fragment], bf);
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t sum_index =
                            (((token_group * 2 + token_fragment) * 2
                                + row_fragment) * 4) + item;
                        partial[sum_index] += float(cf[item])
                            * d4[row_fragment][item / 2] * d8[item % 2];
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void compute_segment_pipe_v1(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        uint8_t * raw0, uint8_t * raw1, int8_t * sx, int8_t * sy,
        float (&partial)[64], uint32_t n_tok, uint32_t blocks,
        uint32_t tile_row, uint32_t tile_token,
        uint32_t segment_begin, uint32_t segment_end) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t active_tokens = tile_token < n_tok
        ? min(imparo_sm80_mmq::kTokens, n_tok - tile_token) : 0;
#pragma unroll
    for (uint32_t item = 0; item < 64; ++item) partial[item] = 0.0f;

    stage_q4_raw_k256_async(w, raw0, blocks, tile_row, segment_begin, tid);
    imparo_sm80_mmq::commit_async_copies();
    imparo_sm80_mmq::wait_async_copies();
    __syncthreads();

    uint8_t * current_raw = raw0;
    uint8_t * next_raw = raw1;
    for (uint32_t stage_block = segment_begin;
         stage_block < segment_end; stage_block += kPackedBlocks) {
        const bool has_next = stage_block + kPackedBlocks < segment_end;
        imparo_sm80_mmq::stage_activation_async(
            x, sy, n_tok, tile_token, active_tokens, stage_block / 4, tid);
        imparo_sm80_mmq::commit_async_copies();
        if (has_next) {
            stage_q4_raw_k256_async(w, next_raw, blocks, tile_row,
                stage_block + kPackedBlocks, tid);
            imparo_sm80_mmq::commit_async_copies();
        }
        expand_raw_q4_k128<0>(current_raw, sx, tid);
        if (has_next) wait_async_copies_one_group();
        else imparo_sm80_mmq::wait_async_copies();
        __syncthreads();
        mma_phase_k128<0>(sx, sy, partial, active_tokens, lane, warp);
        __syncthreads();

        imparo_sm80_mmq::stage_activation_async(
            x, sy, n_tok, tile_token, active_tokens,
            (stage_block + imparo_sm80_mmq::kHalfKBlocks) / 4, tid);
        imparo_sm80_mmq::commit_async_copies();
        expand_raw_q4_k128<1>(current_raw, sx, tid);
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();
        mma_phase_k128<1>(sx, sy, partial, active_tokens, lane, warp);
        __syncthreads();

        uint8_t * swap = current_raw;
        current_raw = next_raw;
        next_raw = swap;
    }
}

__launch_bounds__(256, 1)
__global__ void q4_q8_1_down_direct_seam_pipe_v1(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        float * __restrict__ dst, uint32_t n_tok,
        uint32_t dst_stride, uint32_t numeric_stream_grid) {
#if __CUDA_ARCH__ == 860
    using namespace imparo_sm80_mmq;
    extern __shared__ __align__(16) uint8_t storage[];
    uint8_t * raw0 = storage;
    uint8_t * raw1 = raw0 + kRawStageBytes;
    int8_t * sx = reinterpret_cast<int8_t *>(raw1 + kRawStageBytes);
    int8_t * sy = sx + kRows * kHalfKWeightStride;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    constexpr uint32_t blocks = DownShapeV1::kBlocks;
    constexpr uint32_t ntx = DownShapeV1::kTokenTiles;
    const uint32_t tile_row = (uint32_t(blockIdx.x) / ntx) * kRows;
    const uint32_t tile_token = (uint32_t(blockIdx.x) % ntx) * kTokens;
    float partial[64];

    const uint64_t total_work = uint64_t(gridDim.x) * blocks;
    uint32_t worker = uint32_t(
        (uint64_t(blockIdx.x) * numeric_stream_grid + gridDim.x - 1)
        / gridDim.x);
    worker = max(worker, 1u);
    uint32_t numeric_seam = 0;
    if (worker < numeric_stream_grid) {
        const uint64_t boundary = stream_boundary(
            worker, total_work, numeric_stream_grid, blocks);
        if (boundary / blocks == blockIdx.x && boundary % blocks) {
            numeric_seam = uint32_t(boundary % blocks);
        }
    }
    const uint32_t first_end = numeric_seam ? numeric_seam : blocks;
    compute_segment_pipe_v1(w, x, raw0, raw1, sx, sy, partial, n_tok,
        blocks, tile_row, tile_token, 0, first_end);

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
                            + row_fragment * 16 + accumulator_row(lane, item);
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
        compute_segment_pipe_v1(w, x, raw0, raw1, sx, sy, partial,
            n_tok, blocks, tile_row, tile_token, numeric_seam, blocks);
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
                        ? partial[sum_index] + *slot
                        : partial[sum_index];
                }
            }
        }
    }
#else
    (void)w; (void)x; (void)dst; (void)n_tok;
    (void)dst_stride; (void)numeric_stream_grid;
#endif
}

inline LaunchResult launch(
        const uint8_t * w, const BlockQ8_1Mmq * x, float * dst,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t epilogue, uint32_t out_stride, uint32_t row_base,
        uint32_t sm_version, uint32_t sm_count,
        uint32_t stream_k_numeric, uint32_t virtual_token_base,
        uint32_t virtual_schedule, uint32_t max_grid_x,
        cudaStream_t stream, imparo_sm80_mmq::LaunchInfo * info) {
    if (sm_version != 86 || !DownShapeV1::supports(
            w, x, dst, n_in, n_out, n_tok, epilogue, out_stride, row_base,
            sm_count, stream_k_numeric, virtual_token_base,
            virtual_schedule, max_grid_x)) {
        return LaunchResult::NotSupported;
    }
    static const bool configured = [] {
        const cudaError_t shared = cudaFuncSetAttribute(
            q4_q8_1_down_direct_seam_pipe_v1,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kPipeSharedBytes));
        if (shared != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const cudaError_t carveout = cudaFuncSetAttribute(
            q4_q8_1_down_direct_seam_pipe_v1,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (carveout != cudaSuccess) cudaGetLastError();
        return carveout == cudaSuccess;
    }();
    if (!configured) return LaunchResult::NotSupported;

    q4_q8_1_down_direct_seam_pipe_v1
        <<<DownShapeV1::kLogicalTiles,
            dim3(32, imparo_sm80_mmq::kWarps),
            kPipeSharedBytes, stream>>>(
                w, x, dst, n_tok, out_stride, DownShapeV1::kSms);
    if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;

    if (info) {
        info->route = imparo_sm80_mmq::LaunchRoute::VirtualDirectSeam;
        info->tile_rows = imparo_sm80_mmq::kRows;
        info->tile_tokens = imparo_sm80_mmq::kTokens;
        info->logical_tiles = DownShapeV1::kLogicalTiles;
        info->physical_blocks = DownShapeV1::kSms;
        info->efficiency = 100u * DownShapeV1::kLogicalTiles
            / (DownShapeV1::kSms
                * ((DownShapeV1::kLogicalTiles + DownShapeV1::kSms - 1)
                    / DownShapeV1::kSms));
        info->fused_q8 = false;
    }
    return LaunchResult::Launched;
}

} // namespace imparo_sm86_mmq_down_pipe
