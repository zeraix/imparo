#pragma once

// Dormant SM86 laboratory path for the canonical gate/up pair.  A compact
// R64xT128 CTA keeps two sidecar K128 Q4 stages in shared memory and expands
// only the A fragments actually consumed by mma.m16n8k32 into registers.
namespace imparo_sm86_mmq_gate_up_pipe {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kInput = 2560;
constexpr uint32_t kOutput = 10240;
constexpr uint32_t kPaddedTokens = 512;
constexpr uint32_t kCtaRows = 64;
constexpr uint32_t kQ8RecordRows = imparo_sm80_mmq::kRows;
constexpr uint32_t kStageBlocks = imparo_sm80_mmq::kHalfKBlocks;
constexpr uint32_t kBlocks = kInput / 32;
constexpr uint32_t kStageRecords = kCtaRows * kStageBlocks;
constexpr uint32_t kNibbleStageBytes = kStageRecords * sizeof(uint4);
constexpr uint32_t kScaleStageBytes = kStageRecords * sizeof(uint16_t);
constexpr uint32_t kActivationBytes =
    imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kSharedBytes =
    2 * (kNibbleStageBytes + kScaleStageBytes) + kActivationBytes;
constexpr uint32_t kTokenTiles =
    kPaddedTokens / imparo_sm80_mmq::kTokens;
constexpr uint32_t kLogicalTiles = (kOutput / kCtaRows) * kTokenTiles;

static_assert(kNibbleStageBytes == 4096, "one R64 K128 nibble slot");
static_assert(kScaleStageBytes == 512, "one R64 K128 scale slot");
static_assert(kSharedBytes == 27648, "R64 double-slot shared budget");
static_assert(kSharedBytes < 50 * 1024, "hard shared-memory gate");
static_assert(kLogicalTiles == 640, "canonical R64 full grid");
static_assert(kQ8RecordRows == 2 * kCtaRows,
              "two R64 CTAs complete one Q8 MMQ record");

__device__ __forceinline__ void stage_weight_async(
        const uint16_t * __restrict__ scales,
        const uint4 * __restrict__ nibbles,
        uint16_t * scale_slot, uint4 * nibble_slot,
        uint32_t tile_row, uint32_t stage_block, uint32_t tid) {
    const uint64_t record =
        imparo_sm86_q4_aligned_prepack::packed_record_index(
            tile_row, stage_block, kInput);
    if (tid < kStageRecords) {
        imparo_sm80_mmq::copy_global_to_shared_16(
            nibble_slot + tid, nibbles + record + tid);
    }
    constexpr uint32_t scale_bundles = kScaleStageBytes / sizeof(uint4);
    if (tid < scale_bundles) {
        imparo_sm80_mmq::copy_global_to_shared_16(
            reinterpret_cast<uint4 *>(scale_slot) + tid,
            reinterpret_cast<const uint4 *>(scales + record) + tid);
    }
}

__device__ __forceinline__ void stage_activation_zfill_async(
        const BlockQ8_1Mmq * __restrict__ x, int8_t * sy,
        uint32_t n_tok, uint32_t tile_token,
        uint32_t group_index, uint32_t tid) {
    constexpr uint32_t vectors_per_record =
        sizeof(BlockQ8_1Mmq) / sizeof(uint4);
    constexpr uint32_t copies =
        imparo_sm80_mmq::kTokens * vectors_per_record;
    uint4 * dst = reinterpret_cast<uint4 *>(sy);
    for (uint32_t linear = tid; linear < copies;
         linear += imparo_sm80_mmq::kWarps * 32) {
        const uint32_t local_token = linear / vectors_per_record;
        const uint32_t vector = linear % vectors_per_record;
        const bool valid = tile_token + local_token < n_tok;
        const uint32_t safe_token = valid ? tile_token + local_token : 0;
        const uint4 * src = reinterpret_cast<const uint4 *>(
            x + uint64_t(group_index) * n_tok + safe_token) + vector;
        imparo_sm80_mmq::copy_global_to_shared_16_zfill(
            dst + linear, src, valid ? sizeof(uint4) : 0);
    }
}

__device__ __forceinline__ void load_a_packed(
        int (&a)[4], const uint4 * nibbles,
        uint32_t local_row0, uint32_t qblock, uint32_t lane) {
    const uint32_t row_in_half = lane >> 2;
    const uint32_t packed_word = lane & 3u;
    const uint32_t upper_row = local_row0 + row_in_half;
    const uint32_t lower_row = upper_row + 8;
    const uint32_t * upper = reinterpret_cast<const uint32_t *>(
        nibbles + upper_row * kStageBlocks + qblock);
    const uint32_t * lower = reinterpret_cast<const uint32_t *>(
        nibbles + lower_row * kStageBlocks + qblock);
    const uint32_t upper_nibbles = upper[packed_word];
    const uint32_t lower_nibbles = lower[packed_word];
    a[0] = imparo_sm80_mmq::unpack_q4_nibbles(upper_nibbles);
    a[1] = imparo_sm80_mmq::unpack_q4_nibbles(lower_nibbles);
    a[2] = imparo_sm80_mmq::unpack_q4_nibbles(upper_nibbles >> 4);
    a[3] = imparo_sm80_mmq::unpack_q4_nibbles(lower_nibbles >> 4);
}

__device__ __forceinline__ float load_scale(
        const uint16_t * scales, uint32_t local_row, uint32_t qblock) {
    const uint16_t bits = scales[local_row * kStageBlocks + qblock];
    return __half2float(*reinterpret_cast<const __half *>(&bits));
}

__device__ __forceinline__ void mma_stage(
        const uint16_t * scales, const uint4 * nibbles,
        const int8_t * sy, float (&partial)[32],
        uint32_t lane, uint32_t warp) {
    const uint32_t local_row0 = (warp >> 1) * 16;
#pragma unroll
    for (uint32_t qblock = 0; qblock < kStageBlocks; ++qblock) {
        int af[4];
        load_a_packed(af, nibbles, local_row0, qblock, lane);
        float d4[2];
#pragma unroll
        for (uint32_t scale_item = 0; scale_item < 2; ++scale_item) {
            const uint32_t local_row = local_row0
                + imparo_sm80_mmq::accumulator_row(lane, scale_item * 2);
            d4[scale_item] = load_scale(scales, local_row, qblock);
        }
#pragma unroll
        for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
            for (uint32_t token_fragment = 0; token_fragment < 2;
                 ++token_fragment) {
                const uint32_t local_token0 = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8;
                int bf[2];
                imparo_sm80_mmq::load_b_m16n8k32(bf,
                    sy + local_token0 * imparo_sm80_mmq::kActivationStride
                        + qblock * 32,
                    imparo_sm80_mmq::kActivationStride);
                float d8[2];
#pragma unroll
                for (uint32_t scale_item = 0; scale_item < 2; ++scale_item) {
                    const uint32_t local_token = local_token0
                        + imparo_sm80_mmq::accumulator_token(lane, scale_item);
                    d8[scale_item] = reinterpret_cast<const float *>(sy
                        + local_token * imparo_sm80_mmq::kActivationStride
                        + imparo_sm80_mmq::kActivationStage)[qblock];
                }
                int cf[4] = {};
                imparo_sm80_mmq::mma_m16n8k32(cf, af, bf);
#pragma unroll
                for (uint32_t item = 0; item < 4; ++item) {
                    const uint32_t sum_index =
                        ((token_group * 2 + token_fragment) * 4) + item;
                    partial[sum_index] += float(cf[item])
                        * d4[item / 2] * d8[item % 2];
                }
            }
        }
    }
}

__device__ __forceinline__ void compute(
        const uint16_t * __restrict__ scales,
        const uint4 * __restrict__ nibbles,
        const BlockQ8_1Mmq * __restrict__ x,
        uint16_t * scale0, uint16_t * scale1,
        uint4 * nibble0, uint4 * nibble1, int8_t * sy,
        float (&partial)[32], uint32_t n_tok,
        uint32_t tile_row, uint32_t tile_token) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
#pragma unroll
    for (uint32_t item = 0; item < 32; ++item) partial[item] = 0.0f;

    stage_weight_async(scales, nibbles, scale0, nibble0,
        tile_row, 0, tid);
    imparo_sm80_mmq::commit_async_copies();
    imparo_sm80_mmq::wait_async_copies();
    __syncthreads();

    uint16_t * current_scale = scale0;
    uint16_t * next_scale = scale1;
    uint4 * current_nibble = nibble0;
    uint4 * next_nibble = nibble1;
#pragma unroll
    for (uint32_t stage_block = 0; stage_block < kBlocks;
         stage_block += kStageBlocks) {
        stage_activation_zfill_async(
            x, sy, n_tok, tile_token, stage_block / kStageBlocks, tid);
        imparo_sm80_mmq::commit_async_copies();
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();

        constexpr uint32_t final_stage = kBlocks - kStageBlocks;
        if (stage_block != final_stage) {
            stage_weight_async(scales, nibbles,
                next_scale, next_nibble, tile_row,
                stage_block + kStageBlocks, tid);
            imparo_sm80_mmq::commit_async_copies();
        }
        mma_stage(current_scale, current_nibble, sy, partial, lane, warp);
        if (stage_block != final_stage) {
            imparo_sm80_mmq::wait_async_copies();
            __syncthreads();
            uint16_t * scale_swap = current_scale;
            current_scale = next_scale;
            next_scale = scale_swap;
            uint4 * nibble_swap = current_nibble;
            current_nibble = next_nibble;
            next_nibble = nibble_swap;
        }
    }
}

template <bool DirectEpilogue, bool QuantizeEpilogue>
__launch_bounds__(256, 2)
__global__ void q4_q8_1_gate_up_pipe(
        const uint16_t * __restrict__ scales,
        const uint4 * __restrict__ nibbles,
        const BlockQ8_1Mmq * __restrict__ x,
        float * __restrict__ dst,
        BlockQ8_1Mmq * __restrict__ epilogue_q8,
        uint32_t n_tok) {
#if __CUDA_ARCH__ == 860
    static_assert(!QuantizeEpilogue || DirectEpilogue,
                  "quantization requires the complete direct epilogue");
    extern __shared__ __align__(16) uint8_t storage[];
    uint4 * nibble0 = reinterpret_cast<uint4 *>(storage);
    uint4 * nibble1 = nibble0 + kStageRecords;
    uint16_t * scale0 = reinterpret_cast<uint16_t *>(
        nibble1 + kStageRecords);
    uint16_t * scale1 = scale0 + kStageRecords;
    int8_t * sy = reinterpret_cast<int8_t *>(scale1 + kStageRecords);
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tile_row =
        (uint32_t(blockIdx.x) / kTokenTiles) * kCtaRows;
    const uint32_t tile_token =
        (uint32_t(blockIdx.x) % kTokenTiles) * imparo_sm80_mmq::kTokens;
    float partial[32];
    compute(scales, nibbles, x, scale0, scale1, nibble0, nibble1,
        sy, partial, n_tok, tile_row, tile_token);

#pragma unroll
    for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 2;
             ++token_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = (warp >> 1) * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t local_token = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                const uint32_t token = tile_token + local_token;
                if (token >= n_tok) continue;
                const uint32_t sum_index =
                    ((token_group * 2 + token_fragment) * 4) + item;
                float * slot = dst + uint64_t(token) * kOutput
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
        __syncthreads();
        for (uint32_t record_linear = warp;
             record_linear < 2 * imparo_sm80_mmq::kTokens;
             record_linear += imparo_sm80_mmq::kWarps) {
            const uint32_t local_token = record_linear / 2;
            const uint32_t local_block = record_linear % 2;
            const uint32_t token = tile_token + local_token;
            float value = 0.0f;
            if (token < n_tok) {
                value = dst[uint64_t(token) * kOutput + tile_row
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
                    + uint64_t(tile_row / kQ8RecordRows) * n_tok + token;
                const uint32_t block_in_record =
                    (tile_row % kQ8RecordRows) / 32 + local_block;
                out->qs[block_in_record * 32 + lane] =
                    int8_t(roundf(value * d_inv));
                if (lane == 0) {
                    out->d[block_in_record] = __half2float(__float2half(d));
                }
            }
        }
    }
#else
    (void)scales; (void)nibbles; (void)x; (void)dst;
    (void)epilogue_q8; (void)n_tok;
#endif
}

inline bool supports(
        const uint16_t * scales, const uint4 * nibbles,
        const BlockQ8_1Mmq * x, const float * dst,
        const BlockQ8_1Mmq * epilogue_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t epilogue, uint32_t out_stride, uint32_t row_base,
        uint32_t sm_version, uint32_t max_grid_x) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(scales)
        | reinterpret_cast<uintptr_t>(nibbles)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(dst);
    return scales && nibbles && x && dst && (pointers & 15u) == 0
        && n_in == kInput && n_out == kOutput
        && (n_tok == 449 || n_tok == kPaddedTokens)
        && out_stride == kOutput && row_base == 0 && sm_version == 86
        && max_grid_x >= kLogicalTiles
        && ((epilogue == 0 && epilogue_q8 == nullptr)
            || epilogue == 1);
}

inline LaunchResult launch(
        const uint16_t * scales, const uint4 * nibbles,
        const BlockQ8_1Mmq * x, float * dst,
        BlockQ8_1Mmq * epilogue_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t epilogue, uint32_t out_stride, uint32_t row_base,
        uint32_t sm_version, uint32_t sm_count, uint32_t max_grid_x,
        cudaStream_t stream, imparo_sm80_mmq::LaunchInfo * info) {
    if (!supports(scales, nibbles, x, dst, epilogue_q8,
            n_in, n_out, n_tok,
            epilogue, out_stride, row_base, sm_version, max_grid_x)) {
        return LaunchResult::NotSupported;
    }
    static const bool configured = [] {
        const cudaError_t gate_shared = cudaFuncSetAttribute(
            q4_q8_1_gate_up_pipe<false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
        const cudaError_t up_shared = cudaFuncSetAttribute(
            q4_q8_1_gate_up_pipe<true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
        const cudaError_t up_direct_shared = cudaFuncSetAttribute(
            q4_q8_1_gate_up_pipe<true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
        if (gate_shared != cudaSuccess || up_shared != cudaSuccess
                || up_direct_shared != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        return true;
    }();
    if (!configured) return LaunchResult::NotSupported;

    if (epilogue == 0) {
        q4_q8_1_gate_up_pipe<false, false>
            <<<kLogicalTiles, dim3(32, imparo_sm80_mmq::kWarps),
                kSharedBytes, stream>>>(
                    scales, nibbles, x, dst, nullptr, n_tok);
    } else if (epilogue_q8) {
        q4_q8_1_gate_up_pipe<true, true>
            <<<kLogicalTiles, dim3(32, imparo_sm80_mmq::kWarps),
                kSharedBytes, stream>>>(
                    scales, nibbles, x, dst, epilogue_q8, n_tok);
    } else {
        q4_q8_1_gate_up_pipe<true, false>
            <<<kLogicalTiles, dim3(32, imparo_sm80_mmq::kWarps),
                kSharedBytes, stream>>>(
                    scales, nibbles, x, dst, nullptr, n_tok);
    }
    if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;
    if (info) {
        info->route = imparo_sm80_mmq::LaunchRoute::FullTile;
        info->tile_rows = kCtaRows;
        info->tile_tokens = imparo_sm80_mmq::kTokens;
        info->logical_tiles = kLogicalTiles;
        info->physical_blocks = kLogicalTiles;
        const uint32_t waves = sm_count
            ? (kLogicalTiles + sm_count - 1) / sm_count : 0;
        info->efficiency = waves
            ? 100u * kLogicalTiles / (sm_count * waves) : 0;
        info->fused_q8 = epilogue == 1 && epilogue_q8 != nullptr;
    }
    return LaunchResult::Launched;
}

} // namespace imparo_sm86_mmq_gate_up_pipe
