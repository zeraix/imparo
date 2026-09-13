#pragma once
#include "../q8_quant_contract.cuh"

// SM86 Q8_0 TileMajor gate/up transaction selected by the existing TM policy.
// Four warp pairs share one R64 output tile.  The even warp in each pair owns
// Gate and the odd warp owns Up for the same sixteen rows.  Consequently every
// thread carries one accumulator set rather than simultaneous Gate and Up
// arrays.  Once K is complete, Gate crosses the warp pair through storage that
// is no longer needed by the MMQ stages; Up applies SiLU and publishes the
// ordinary four-block BlockQ8_1Mmq sidecar consumed by Down.
//
// The shared fragment body preserves the existing full-tile entry. A narrow
// opt-in tail specialization omits only entirely unused token fragments.
namespace imparo_sm86_q8_tm_gate_up_row_pair_lab {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kRows = 64;
constexpr uint32_t kTokens = 128;
constexpr uint32_t kWarps = 8;
constexpr uint32_t kBlockValues = imparo_sm80_q8_mmq::kBlockValues;
constexpr uint32_t kStageBlocks = imparo_sm80_q8_mmq::kStageBlocks;
constexpr uint32_t kWeightStride = imparo_sm80_q8_mmq::kWeightStride;
constexpr uint32_t kActivationStride =
    imparo_sm80_q8_mmq::kActivationStride;
constexpr uint32_t kActivationRecordValues =
    imparo_sm80_q8_mmq::kActivationRecordValues;
constexpr uint32_t kProjectionWeightBytes = kRows * kWeightStride;
constexpr uint32_t kActivationBytes =
    2 * kTokens * kActivationStride;
constexpr uint32_t kSharedBytes =
    2 * kProjectionWeightBytes + kActivationBytes;
constexpr uint32_t kOutputTileValues = kRows * kTokens;

static_assert(kStageBlocks == 8, "Q8 TileMajor stage ABI");
static_assert(kWeightStride == 304, "Q8 TileMajor shared stride ABI");
static_assert(kSharedBytes == 75776, "paired Q8 MMQ shared budget");
static_assert(kOutputTileValues * sizeof(float) <= kSharedBytes,
              "completed MMQ storage must hold the fused output tile");
static_assert(imparo_sm80_q8_mmq::kRows == 2 * kRows,
              "two R64 CTAs must complete one D4 sidecar record");

__device__ __forceinline__ void stage_tm_projection_weights(
        const uint8_t * __restrict__ gate,
        const uint8_t * __restrict__ up,
        int8_t * __restrict__ gate_stage,
        int8_t * __restrict__ up_stage,
        uint32_t n_in, uint32_t n_out,
        uint32_t tile_row, uint32_t stage_block,
        uint32_t tid) {
    const uint32_t blocks = n_in / kBlockValues;
    constexpr uint32_t projection_records = kRows * kStageBlocks;
    constexpr uint32_t records = 2 * projection_records;
    for (uint32_t linear = tid; linear < records;
         linear += kWarps * 32) {
        const uint32_t projection = linear / projection_records;
        const uint32_t within = linear % projection_records;
        const uint32_t qblock = within / kRows;
        const uint32_t local_row = within % kRows;
        const uint32_t row = tile_row + local_row;
        const uint32_t kb = stage_block + qblock;
        const uint8_t * base = projection == 0 ? gate : up;
        int8_t * stage = projection == 0 ? gate_stage : up_stage;
        const uint64_t unit = uint64_t(row / 8) * blocks + kb;
        const uint8_t * values = base + unit * 256 + (row & 7u) * 32;
        int8_t * staged_values = stage + local_row * kWeightStride
            + qblock * kBlockValues;
        imparo_sm80_mmq::copy_global_to_shared_16(
            reinterpret_cast<uint4 *>(staged_values),
            reinterpret_cast<const uint4 *>(values));
        imparo_sm80_mmq::copy_global_to_shared_16(
            reinterpret_cast<uint4 *>(staged_values) + 1,
            reinterpret_cast<const uint4 *>(values) + 1);

        const auto * scales = reinterpret_cast<const __half *>(
            base + uint64_t(n_out) * n_in);
        reinterpret_cast<float *>(stage + local_row * kWeightStride
            + imparo_sm80_q8_mmq::kStageValues)[qblock] =
                __half2float(scales[unit * 8 + (row & 7u)]);
    }
}

template<uint32_t LiveFragments = 16>
__device__ __forceinline__ void accumulate_projection(
        const int8_t * __restrict__ weight_stage,
        const int8_t * __restrict__ activation_stage,
        float (&partial)[64], uint32_t lane, uint32_t warp_pair) {
    const uint32_t local_row0 = warp_pair * 16;
#pragma unroll
    for (uint32_t phase = 0; phase < 2; ++phase) {
        const int8_t * phase_activation = activation_stage
            + phase * kTokens * kActivationStride;
#pragma unroll
        for (uint32_t qblock = 0; qblock < 4; ++qblock) {
            const uint32_t weight_qblock = phase * 4 + qblock;
            int af[4];
            imparo_sm80_mmq::load_a_m16n8k32(
                af, weight_stage + local_row0 * kWeightStride
                    + weight_qblock * kBlockValues,
                kWeightStride);
            float d8w[2];
#pragma unroll
            for (uint32_t scale_item = 0; scale_item < 2; ++scale_item) {
                const uint32_t local_row = local_row0
                    + imparo_sm80_mmq::accumulator_row(lane, scale_item * 2);
                d8w[scale_item] = reinterpret_cast<const float *>(
                    weight_stage + local_row * kWeightStride
                        + imparo_sm80_q8_mmq::kStageValues)[weight_qblock];
            }

#pragma unroll
            for (uint32_t token_group = 0; token_group < 4; ++token_group) {
#pragma unroll
                for (uint32_t token_fragment = 0; token_fragment < 4;
                     ++token_fragment) {
                    const uint32_t local_token0 =
                        token_group * 32 + token_fragment * 8;
                    if (local_token0 >= LiveFragments * 8) continue;
                    int bf[2];
                    imparo_sm80_mmq::load_b_m16n8k32(
                        bf, phase_activation
                            + local_token0 * kActivationStride
                            + qblock * kBlockValues,
                        kActivationStride);
                    float d8a[2];
#pragma unroll
                    for (uint32_t scale_item = 0; scale_item < 2;
                         ++scale_item) {
                        const uint32_t local_token = local_token0
                            + imparo_sm80_mmq::accumulator_token(
                                lane, scale_item);
                        d8a[scale_item] = reinterpret_cast<const float *>(
                            phase_activation + local_token * kActivationStride
                                + kActivationRecordValues)[qblock];
                    }
                    int cf[4] = {};
                    imparo_sm80_mmq::mma_m16n8k32(cf, af, bf);
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item) {
                        const uint32_t sum_index =
                            ((token_group * 4 + token_fragment) * 4) + item;
                        partial[sum_index] += float(cf[item])
                            * d8w[item / 2] * d8a[item % 2];
                    }
                }
            }
        }
    }
}

template<uint32_t LiveFragments>
__device__ __forceinline__ void gate_up_body(
        const uint8_t * __restrict__ gate,
        const uint8_t * __restrict__ up,
        const BlockQ8_1Mmq * __restrict__ x,
        BlockQ8_1Mmq * __restrict__ output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        bool match_m1_quant = false) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    static_assert(LiveFragments > 0 && LiveFragments <= 16, "token fragment bound");
    extern __shared__ __align__(16) int8_t shared[];
    int8_t * gate_stage = shared;
    int8_t * up_stage = gate_stage + kProjectionWeightBytes;
    int8_t * activation_stage = up_stage + kProjectionWeightBytes;

    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t projection = warp & 1u;
    const uint32_t warp_pair = warp >> 1;
    const uint32_t tile_row = blockIdx.x * kRows;
    const uint32_t tile_token = blockIdx.y * kTokens;
    const uint32_t active_tokens = tile_token < n_tok
        ? min(kTokens, n_tok - tile_token) : 0;
    const uint32_t blocks = n_in / kBlockValues;

    float partial[64];
#pragma unroll
    for (uint32_t item = 0; item < 64; ++item) partial[item] = 0.0f;

    for (uint32_t stage_block = 0; stage_block < blocks;
         stage_block += kStageBlocks) {
#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase) {
            const uint32_t group_block = stage_block + phase * 4;
            imparo_sm80_q8_mmq::stage_activation_group<kTokens>(
                x, activation_stage + phase * kTokens * kActivationStride,
                n_tok, tile_token, active_tokens, group_block / 4,
                true, tid);
        }
        imparo_sm80_mmq::commit_async_copies();
        stage_tm_projection_weights(gate, up, gate_stage, up_stage,
            n_in, n_out, tile_row, stage_block, tid);
        imparo_sm80_mmq::commit_async_copies();
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();

        const int8_t * selected_weights =
            projection == 0 ? gate_stage : up_stage;
        accumulate_projection<LiveFragments>(selected_weights, activation_stage,
            partial, lane, warp_pair);
        __syncthreads();
    }

    // The MMQ stages are dead after K completes, so the same allocation can
    // carry Gate across each warp pair without increasing shared memory.
    float * fused_tile = reinterpret_cast<float *>(shared);
    if (projection == 0) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 16;
             ++token_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = warp_pair * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t local_token = token_fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                if (local_token < active_tokens) {
                    const uint32_t sum_index = token_fragment * 4 + item;
                    fused_tile[uint64_t(local_token) * kRows + local_row] =
                        partial[sum_index];
                }
            }
        }
    }
    __syncthreads();

    if (projection == 1) {
#pragma unroll
        for (uint32_t token_fragment = 0; token_fragment < 16;
             ++token_fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t local_row = warp_pair * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t local_token = token_fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                if (local_token < active_tokens) {
                    const uint32_t sum_index = token_fragment * 4 + item;
                    const uint64_t tile_index =
                        uint64_t(local_token) * kRows + local_row;
                    const float value = imparo_cuda_lfm2::silu(
                        fused_tile[tile_index]) * partial[sum_index];
                    fused_tile[tile_index] = value;
                }
            }
        }
    }
    __syncthreads();

    // Match the established Epilogue4 arithmetic and its eight-lane groups:
    // eight float4 vectors make one independently scaled 32-value Q8 block.
    // Adjacent R64 CTAs own disjoint block pairs in the same D4 record.
    constexpr uint32_t blocks_per_token = kRows / 32;
    constexpr uint32_t vectors_per_block = 32 / 4;
    const uint32_t quant_units = active_tokens * blocks_per_token;
    const uint32_t quant_slot = lane / vectors_per_block;
    const uint32_t vector_in_block = lane % vectors_per_block;
    for (uint32_t unit_base = warp * 4; unit_base < quant_units;
         unit_base += kWarps * 4) {
        const uint32_t unit = unit_base + quant_slot;
        const bool valid_unit = unit < quant_units;
        // All 32 lanes must execute the full-mask shuffle.  Unused slots in
        // the final warp borrow unit zero for arithmetic and suppress stores.
        const uint32_t safe_unit = valid_unit ? unit : 0;
        const uint32_t local_token = safe_unit / blocks_per_token;
        const uint32_t local_block = safe_unit % blocks_per_token;
        const uint32_t vector =
            local_block * vectors_per_block + vector_in_block;
        const uint32_t i0 = vector_in_block * 4;
        const float4 xi = reinterpret_cast<const float4 *>(
            fused_tile + uint64_t(local_token) * kRows)[vector];
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffff, amax, offset, 32));
        }
        char4 quant;
        float d;
        if (match_m1_quant) {
            d = imparo_q8_quant::m1_block(xi, amax, quant);
        } else {
            const float d_inv = 127.0f / amax;
            quant.x = int8_t(roundf(xi.x * d_inv));
            quant.y = int8_t(roundf(xi.y * d_inv));
            quant.z = int8_t(roundf(xi.z * d_inv));
            quant.w = int8_t(roundf(xi.w * d_inv));
            d = 1.0f / d_inv;
        }
        const uint32_t iqs = i0 % 32;
        if (valid_unit) {
            const uint32_t token = tile_token + local_token;
            BlockQ8_1Mmq * out = output_q8
                + uint64_t(tile_row / imparo_sm80_q8_mmq::kRows) * n_tok
                + token;
            const uint32_t block_in_record =
                ((tile_row % imparo_sm80_q8_mmq::kRows) / 32)
                + local_block;
            reinterpret_cast<char4 *>(
                out->qs + block_in_record * 32)[iqs / 4] = quant;
            if (iqs == 0) out->d[block_in_record] = d;
        }
    }
#else
    (void)gate; (void)up; (void)x; (void)output_q8;
    (void)n_in; (void)n_out; (void)n_tok; (void)match_m1_quant;
#endif
}

__launch_bounds__(kWarps * 32, 1)
__global__ void q8_tm_gate_up_row_pair(
        const uint8_t * __restrict__ gate, const uint8_t * __restrict__ up,
        const BlockQ8_1Mmq * __restrict__ x, BlockQ8_1Mmq * __restrict__ output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        bool match_m1_quant = false) {
    gate_up_body<16>(gate, up, x, output_q8, n_in, n_out, n_tok, match_m1_quant);
}

template<uint32_t LiveFragments>
__launch_bounds__(kWarps * 32, 1)
__global__ void q8_tm_gate_up_row_pair_tail(
        const uint8_t * __restrict__ gate, const uint8_t * __restrict__ up,
        const BlockQ8_1Mmq * __restrict__ x, BlockQ8_1Mmq * __restrict__ output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        bool match_m1_quant = false) {
    gate_up_body<LiveFragments>(gate, up, x, output_q8, n_in, n_out, n_tok, match_m1_quant);
}

// Called only after the existing TM route has admitted the layout and shape
// and configured the original full-tile kernel. Both TM consumers share this
// selection; no model-specific code or additional tuner slot is required.
inline cudaError_t launch_selected(
        const uint8_t * gate, const uint8_t * up,
        const BlockQ8_1Mmq * x, BlockQ8_1Mmq * output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok, cudaStream_t stream,
        bool match_m1_quant = false) {
    // The existing TM route has admitted this shape; only live token fragments differ.
    if ((n_tok == 9 || n_tok == 16)) {
        int live2_device = -1;
        cudaError_t err = cudaGetDevice(&live2_device);
        if (err != cudaSuccess) return err;
        static int live2_configured_device = -1;
        static bool live2_configured = false;
        if (live2_configured_device != live2_device) {
            err = cudaFuncSetAttribute(q8_tm_gate_up_row_pair_tail<2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
            live2_configured_device = live2_device;
            live2_configured = err == cudaSuccess;
            if (!live2_configured) {
                if (err != cudaErrorInvalidValue && err != cudaErrorNotSupported) return err;
                (void)cudaGetLastError();
            }
        }
        if (live2_configured) {
            if (const char * trace = std::getenv("IMPARO_LAB_Q8_TM_STATIC_TAIL_TRACE")) {
                if (trace[0] == '1') std::fprintf(stderr, "TM_STATIC_TAIL n_tok=%u fragments=2\n", n_tok);
            }
            q8_tm_gate_up_row_pair_tail<2><<<dim3(n_out / kRows, 1), dim3(32, kWarps), kSharedBytes, stream>>>(
                gate, up, x, output_q8, n_in, n_out, n_tok, match_m1_quant);
            return cudaPeekAtLastError();
        }
    }
    const char * flag = std::getenv("IMPARO_LAB_Q8_TM_STATIC_TAIL");
    bool tail = n_tok >= 49 && n_tok <= 56 && flag && flag[0] == '1';
    if (tail) {
        int device = -1;
        cudaError_t error = cudaGetDevice(&device);
        if (error != cudaSuccess) return error;
        static int configured_device = -1;
        static bool configured = false;
        if (configured_device != device) {
            error = cudaFuncSetAttribute(q8_tm_gate_up_row_pair_tail<7>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
            configured = error == cudaSuccess;
            configured_device = device;
            if (!configured) {
                if (error != cudaErrorInvalidValue && error != cudaErrorNotSupported)
                    return error;
                (void)cudaGetLastError();
            }
        }
        tail = configured;
    }
    const dim3 grid(n_out / kRows, uint32_t((uint64_t(n_tok) + kTokens - 1) / kTokens));
    if (tail) {
        if (const char * trace = std::getenv("IMPARO_LAB_Q8_TM_STATIC_TAIL_TRACE")) {
            if (trace[0] == '1') std::fprintf(stderr, "TM_STATIC_TAIL n_tok=%u fragments=7\n", n_tok);
        }
        q8_tm_gate_up_row_pair_tail<7><<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
            gate, up, x, output_q8, n_in, n_out, n_tok, match_m1_quant);
    } else {
        q8_tm_gate_up_row_pair<<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
            gate, up, x, output_q8, n_in, n_out, n_tok, match_m1_quant);
    }
    return cudaPeekAtLastError();
}

inline bool supports(
        const uint8_t * gate, const uint8_t * up,
        const BlockQ8_1Mmq * x, const BlockQ8_1Mmq * output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t sm_version, uint32_t max_grid_x,
        uint32_t max_grid_y) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(gate)
        | reinterpret_cast<uintptr_t>(up)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(output_q8);
    const uint64_t grid_x = uint64_t(n_out) / kRows;
    const uint64_t grid_y =
        (uint64_t(n_tok) + uint64_t(kTokens) - 1) / kTokens;
    return gate && up && x && output_q8
        && (pointers & 15u) == 0 && sm_version == 86
        && n_in != 0 && n_in % (kStageBlocks * kBlockValues) == 0
        && n_out != 0 && n_out % imparo_sm80_q8_mmq::kRows == 0
        && n_tok > 8 && grid_x != 0 && grid_x <= max_grid_x
        && grid_y != 0 && grid_y <= max_grid_y;
}

inline LaunchResult launch(
        const uint8_t * gate, const uint8_t * up,
        const BlockQ8_1Mmq * x, BlockQ8_1Mmq * output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t sm_version, uint32_t max_grid_x,
        uint32_t max_grid_y, cudaStream_t stream, bool match_m1_quant = false) {
    if (!supports(gate, up, x, output_q8, n_in, n_out, n_tok,
            sm_version, max_grid_x, max_grid_y)) {
        return LaunchResult::NotSupported;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return LaunchResult::Error;
    static int configured_device = -1;
    static bool configured = false;
    if (configured_device != device) {
        const cudaError_t attr = cudaFuncSetAttribute(
            q8_tm_gate_up_row_pair,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
        configured = attr == cudaSuccess;
        configured_device = device;
        if (!configured) {
            if (attr == cudaErrorInvalidValue
                    || attr == cudaErrorNotSupported) {
                (void)cudaGetLastError();
            } else {
                return LaunchResult::Error;
            }
        }
    }
    if (!configured) return LaunchResult::NotSupported;

    return launch_selected(gate, up, x, output_q8, n_in, n_out, n_tok, stream, match_m1_quant) == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_q8_tm_gate_up_row_pair_lab
