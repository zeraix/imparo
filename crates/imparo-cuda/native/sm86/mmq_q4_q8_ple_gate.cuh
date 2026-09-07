#pragma once

#include "q8_mma_ready_a0.cuh"

// SM86 laboratory route for the exact-128 Direct-K and 449..512 Stream-K PLE gate:
//   Q4(2560 -> 256) -> GELU * per-layer -> Q8_1 MMQ layout.
//
// Exact 128 gives each 32-row x 64-token CTA the complete K dimension.  The long
// route replays the established 30-worker Stream-K seams inside each original
// 128x128 logical tile; prefixes are summed nearest-to-farthest before the suffix.
namespace imparo_sm86_ple_gate {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kInput = 2560;
constexpr uint32_t kOutput = 256;
constexpr uint32_t kTokens = 512;
constexpr uint32_t kRows = 32;
constexpr uint32_t kTileTokens = 64;
constexpr uint32_t kWarps = 4;
constexpr uint32_t kBlocks = kInput / 32;
constexpr uint32_t kPackedBlocks = 4;
constexpr uint32_t kPackedRowBytes = kPackedBlocks * 18;
constexpr uint32_t kRawBytes = kRows * kPackedRowBytes;
constexpr uint32_t kWeightStride =
    imparo_sm80_mmq::kHalfKWeightStride;
constexpr uint32_t kWeightBytes = kRows * kWeightStride;
constexpr uint32_t kActivationStride =
    imparo_sm80_mmq::kActivationStride;
constexpr uint32_t kActivationBytes = kTileTokens * kActivationStride;
constexpr uint32_t kTileValues = kRows * kTileTokens;
constexpr uint32_t kValueBytes = kTileValues * sizeof(float);
constexpr uint32_t kSharedBytes =
    kRawBytes + kWeightBytes + kActivationBytes + 2 * kValueBytes;
constexpr uint32_t kGrid =
    (kOutput / kRows) * (kTokens / kTileTokens);

static_assert(kPackedRowBytes == 72, "K128 packed row");
static_assert(kRawBytes == 2304, "PLE raw stage");
static_assert(kSharedBytes == 32512, "PLE shared-memory budget");
static_assert(kGrid == 64, "PLE fixed grid");

__device__ __forceinline__ void copy_global_to_shared_8(
        uint64_t * dst, const uint64_t * src) {
    const uint32_t shared =
        static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
        : : "r"(shared), "l"(src));
}

__device__ __forceinline__ void stage_weight_async(
        const uint8_t * __restrict__ w, uint8_t * raw,
        uint32_t tile_row, uint32_t stage_block, uint32_t tid) {
    constexpr uint32_t vectors = kPackedRowBytes / sizeof(uint64_t);
    constexpr uint32_t copies = kRows * vectors;
    for (uint32_t linear = tid; linear < copies;
         linear += kWarps * 32) {
        const uint32_t local_row = linear / vectors;
        const uint32_t vector = linear % vectors;
        const uint8_t * src = w
            + (uint64_t(tile_row + local_row) * kBlocks + stage_block) * 18
            + vector * sizeof(uint64_t);
        uint8_t * dst = raw
            + local_row * kPackedRowBytes + vector * sizeof(uint64_t);
        copy_global_to_shared_8(
            reinterpret_cast<uint64_t *>(dst),
            reinterpret_cast<const uint64_t *>(src));
    }
}

__device__ __forceinline__ void stage_activation_async(
        const BlockQ8_1Mmq * __restrict__ x, int8_t * sy,
        uint32_t tile_token, uint32_t group, uint32_t n_tok,
        uint32_t tid) {
    constexpr uint32_t vectors =
        sizeof(BlockQ8_1Mmq) / sizeof(uint4);
    constexpr uint32_t copies = kTileTokens * vectors;
    uint4 * dst = reinterpret_cast<uint4 *>(sy);
    for (uint32_t linear = tid; linear < copies;
         linear += kWarps * 32) {
        const uint32_t local_token = linear / vectors;
        const uint32_t vector = linear % vectors;
        const uint32_t token = tile_token + local_token;
        if (token < n_tok) {
            const uint4 * src = reinterpret_cast<const uint4 *>(
                x + uint64_t(group) * n_tok + token);
            imparo_sm80_mmq::copy_global_to_shared_16(
                dst + linear, src + vector);
        } else {
            dst[linear] = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void expand_weight(
        const uint8_t * raw, int8_t * sx, uint32_t tid) {
    constexpr uint32_t count =
        kRows * imparo_sm80_mmq::kHalfKBlocks;
    for (uint32_t linear = tid; linear < count;
         linear += kWarps * 32) {
        const uint32_t qblock =
            linear % imparo_sm80_mmq::kHalfKBlocks;
        const uint32_t local_row =
            linear / imparo_sm80_mmq::kHalfKBlocks;
        const uint8_t * block =
            raw + local_row * kPackedRowBytes + qblock * 18;
        const uint16_t * qs =
            reinterpret_cast<const uint16_t *>(block + 2);
        int8_t * dst =
            sx + local_row * kWeightStride + qblock * 32;
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
            sx + local_row * kWeightStride
                + imparo_sm80_mmq::kHalfKValues)[qblock] =
            __half2float(*reinterpret_cast<const __half *>(block));
    }
}

__device__ __forceinline__ void mma_stage(
        const int8_t * sx, const int8_t * sy, float (&partial)[16],
        uint32_t lane, uint32_t warp) {
#pragma unroll
    for (uint32_t qblock = 0;
         qblock < imparo_sm80_mmq::kHalfKBlocks; ++qblock) {
        int af[4];
        const uint32_t local_row0 = (warp >> 1) * 16;
        imparo_sm80_mmq::load_a_m16n8k32(
            af, sx + local_row0 * kWeightStride + qblock * 32,
            kWeightStride);
        float d4[2];
#pragma unroll
        for (uint32_t scale_item = 0; scale_item < 2;
             ++scale_item) {
            const uint32_t local_row = local_row0
                + imparo_sm80_mmq::accumulator_row(
                    lane, scale_item * 2);
            d4[scale_item] = reinterpret_cast<const float *>(
                sx + local_row * kWeightStride
                    + imparo_sm80_mmq::kHalfKValues)[qblock];
        }
#pragma unroll
        for (uint32_t token_group = 0; token_group < 2;
             ++token_group) {
#pragma unroll
            for (uint32_t token_fragment = 0; token_fragment < 2;
                 ++token_fragment) {
                const uint32_t local_token0 = token_group * 32
                    + (warp & 1) * 16 + token_fragment * 8;
                int bf[2];
                imparo_sm80_mmq::load_b_m16n8k32(
                    bf, sy + local_token0 * kActivationStride
                        + qblock * 32,
                    kActivationStride);
                float d8[2];
#pragma unroll
                for (uint32_t scale_item = 0; scale_item < 2;
                     ++scale_item) {
                    const uint32_t local_token = local_token0
                        + imparo_sm80_mmq::accumulator_token(
                            lane, scale_item);
                    d8[scale_item] = reinterpret_cast<const float *>(
                        sy + local_token * kActivationStride
                            + imparo_sm80_mmq::kActivationStage)[qblock];
                }
                int cf[4] = {};
                imparo_sm80_mmq::mma_m16n8k32(cf, af, bf);
#pragma unroll
                for (uint32_t item = 0; item < 4; ++item) {
                    const uint32_t index =
                        ((token_group * 2 + token_fragment) * 4) + item;
                    partial[index] += float(cf[item])
                        * d4[item / 2] * d8[item % 2];
                }
            }
        }
    }
}

__device__ __forceinline__ void compute_segment(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        uint8_t * raw, int8_t * sx, int8_t * sy,
        float (&partial)[16], uint32_t tile_row, uint32_t tile_token,
        uint32_t n_tok, uint32_t begin, uint32_t end) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) partial[i] = 0.0f;
    for (uint32_t stage = begin; stage < end;
         stage += kPackedBlocks) {
        stage_weight_async(w, raw, tile_row, stage, tid);
        imparo_sm80_mmq::commit_async_copies();
        stage_activation_async(
            x, sy, tile_token, stage / kPackedBlocks, n_tok, tid);
        imparo_sm80_mmq::commit_async_copies();
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();
        expand_weight(raw, sx, tid);
        __syncthreads();
        mma_stage(sx, sy, partial, lane, warp);
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t seam_count(uint32_t logical_tile) {
    return (logical_tile & 3u) == 2u ? 6u : 5u;
}

__device__ __forceinline__ uint32_t seam(
        uint32_t logical_tile, uint32_t index) {
    switch (logical_tile & 3u) {
        case 0:
            switch (index) {
                case 0: return 0;
                case 1: return 16;
                case 2: return 40;
                case 3: return 64;
                default: return 80;
            }
        case 1:
            switch (index) {
                case 0: return 0;
                case 1: return 24;
                case 2: return 48;
                case 3: return 64;
                default: return 80;
            }
        case 2:
            switch (index) {
                case 0: return 0;
                case 1: return 8;
                case 2: return 32;
                case 3: return 48;
                case 4: return 72;
                default: return 80;
            }
        default:
            switch (index) {
                case 0: return 0;
                case 1: return 16;
                case 2: return 32;
                case 3: return 56;
                default: return 80;
            }
    }
}

__device__ __forceinline__ uint32_t value_index(
        uint32_t lane, uint32_t warp, uint32_t linear) {
    const uint32_t token_group = linear / 8;
    const uint32_t rem = linear % 8;
    const uint32_t token_fragment = rem / 4;
    const uint32_t item = rem % 4;
    const uint32_t local_row = (warp >> 1) * 16
        + imparo_sm80_mmq::accumulator_row(lane, item);
    const uint32_t local_token = token_group * 32
        + (warp & 1) * 16 + token_fragment * 8
        + imparo_sm80_mmq::accumulator_token(lane, item);
    return local_token * kRows + local_row;
}

template <bool ReadyOutput, bool DirectK = false>
__launch_bounds__(128, 2)
__global__ void fused_gate(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        const float * __restrict__ per_layer,
        float * __restrict__ gate,
        BlockQ8_1Mmq * __restrict__ q8,
        uint16_t * __restrict__ ready_quant,
        float * __restrict__ ready_scale,
        uint32_t ready_token_tiles,
        uint32_t n_tok, uint32_t per_layer_off,
        uint32_t per_layer_stride) {
#if __CUDA_ARCH__ == 860
    extern __shared__ __align__(16) uint8_t storage[];
    uint8_t * raw = storage;
    int8_t * sx = reinterpret_cast<int8_t *>(raw + kRawBytes);
    int8_t * sy = sx + kWeightBytes;
    float * suffix = reinterpret_cast<float *>(sy + kActivationBytes);
    float * sum = suffix + kTileValues;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t token_tiles = DirectK
        ? (n_tok + kTileTokens - 1) / kTileTokens
        : kTokens / kTileTokens;
    const uint32_t row_tile = blockIdx.x / token_tiles;
    const uint32_t token_tile = blockIdx.x % token_tiles;
    const uint32_t tile_row = row_tile * kRows;
    const uint32_t tile_token = token_tile * kTileTokens;
    float partial[16];

    if constexpr (DirectK) {
        compute_segment(w, x, raw, sx, sy, partial, tile_row, tile_token,
            n_tok, 0, kBlocks);
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t index = value_index(lane, warp, i);
            suffix[index] = partial[i];
            sum[index] = 0.0f;
        }
        __syncthreads();
    } else {
        const uint32_t logical_token_tiles =
            (n_tok + imparo_sm80_mmq::kTokens - 1)
                / imparo_sm80_mmq::kTokens;
        const uint32_t logical_tile =
            (tile_row / imparo_sm80_mmq::kRows) * logical_token_tiles
                + tile_token / imparo_sm80_mmq::kTokens;
        const uint32_t count = seam_count(logical_tile);
        compute_segment(w, x, raw, sx, sy, partial, tile_row, tile_token,
            n_tok, seam(logical_tile, count - 2),
            seam(logical_tile, count - 1));
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t index = value_index(lane, warp, i);
            suffix[index] = partial[i];
            sum[index] = 0.0f;
        }
        __syncthreads();

        for (int segment = int(count) - 3; segment >= 0; --segment) {
            compute_segment(w, x, raw, sx, sy, partial, tile_row, tile_token,
                n_tok, seam(logical_tile, uint32_t(segment)),
                seam(logical_tile, uint32_t(segment + 1)));
#pragma unroll
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t index = value_index(lane, warp, i);
                sum[index] = sum[index] + partial[i];
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t index = value_index(lane, warp, i);
        const uint32_t local_token = index / kRows;
        const uint32_t local_row = index % kRows;
        const uint32_t token = tile_token + local_token;
        const uint32_t row = tile_row + local_row;
        const float result = token < n_tok
            ? cuda_gelu(suffix[index] + sum[index])
                * per_layer[uint64_t(token) * per_layer_stride
                    + per_layer_off + row]
            : 0.0f;
        suffix[index] = result;
        if (token < n_tok) {
            gate[uint64_t(token) * kOutput + row] = result;
        }
    }
    __syncthreads();

    if constexpr (!ReadyOutput) {
      constexpr uint32_t quant_items = kTileTokens * (kRows / 4);
      for (uint32_t linear = tid; linear < quant_items;
           linear += kWarps * 32) {
        const uint32_t local_token = linear / (kRows / 4);
        const uint32_t lane8 = linear % (kRows / 4);
        const uint32_t row0 = lane8 * 4;
        const float4 value = *reinterpret_cast<const float4 *>(
            suffix + local_token * kRows + row0);
        float amax = fabsf(value.x);
        amax = fmaxf(amax, fabsf(value.y));
        amax = fmaxf(amax, fabsf(value.z));
        amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffff, amax, offset, 32));
        }
        const float d_inv = 127.0f / amax;
        const float d = 1.0f / d_inv;
        char4 quant;
        quant.x = int8_t(roundf(value.x * d_inv));
        quant.y = int8_t(roundf(value.y * d_inv));
        quant.z = int8_t(roundf(value.z * d_inv));
        quant.w = int8_t(roundf(value.w * d_inv));
        const uint32_t block = tile_row / 32;
        const uint32_t token = tile_token + local_token;
        if (token >= n_tok) continue;
        BlockQ8_1Mmq * out =
            q8 + uint64_t(block / 4) * n_tok + token;
        reinterpret_cast<char4 *>(
            out->qs + (block % 4) * 32)[lane8] = quant;
        if (lane8 == 0) {
            out->d[block % 4] =
                __half2float(__float2half(d));
        }
      }
    } else {
      using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
      constexpr uint32_t ready_items =
          (kTileTokens / 2) * (kRows / 4);
      for (uint32_t linear = tid; linear < ready_items;
           linear += kWarps * 32) {
        const uint32_t local_pair = linear / (kRows / 4);
        const uint32_t lane8 = linear % (kRows / 4);
        const uint32_t local_token0 = local_pair * 2;
        const uint32_t local_token1 = local_token0 + 1;
        const uint32_t row0 = lane8 * 4;
        const float4 value0 = *reinterpret_cast<const float4 *>(
            suffix + local_token0 * kRows + row0);
        const float4 value1 = *reinterpret_cast<const float4 *>(
            suffix + local_token1 * kRows + row0);
        const float amax0 = warp_q8_amax(value0);
        const float amax1 = warp_q8_amax(value1);
        const float d_inv0 = 127.0f / amax0;
        const float d_inv1 = 127.0f / amax1;
        char4 quant0{};
        char4 quant1{};
        quant0.x = int8_t(roundf(value0.x * d_inv0));
        quant0.y = int8_t(roundf(value0.y * d_inv0));
        quant0.z = int8_t(roundf(value0.z * d_inv0));
        quant0.w = int8_t(roundf(value0.w * d_inv0));
        quant1.x = int8_t(roundf(value1.x * d_inv1));
        quant1.y = int8_t(roundf(value1.y * d_inv1));
        quant1.z = int8_t(roundf(value1.z * d_inv1));
        quant1.w = int8_t(roundf(value1.w * d_inv1));

        const uint32_t packed0 =
            uint32_t(uint8_t(quant0.x))
            | (uint32_t(uint8_t(quant0.y)) << 8)
            | (uint32_t(uint8_t(quant1.x)) << 16)
            | (uint32_t(uint8_t(quant1.y)) << 24);
        const uint32_t packed1 =
            uint32_t(uint8_t(quant0.z))
            | (uint32_t(uint8_t(quant0.w)) << 8)
            | (uint32_t(uint8_t(quant1.z)) << 16)
            | (uint32_t(uint8_t(quant1.w)) << 24);
        const uint32_t group = tile_row / kValuesPerGroup;
        const uint32_t qblock =
            (tile_row % kValuesPerGroup) / kValuesPerQBlock;
        const uint32_t token0 = tile_token + local_token0;
        const uint32_t token_tile = token0 / kTokenTile;
        const uint32_t token_in_tile = token0 % kTokenTile;
        const uint32_t kpair0 = lane8 * 2;
        const uint64_t q16_base0 = quant_u16_index(
            ready_token_tiles, group, token_tile, qblock,
            kpair0, token_in_tile);
        const uint64_t q16_base1 = quant_u16_index(
            ready_token_tiles, group, token_tile, qblock,
            kpair0 + 1, token_in_tile);
        reinterpret_cast<uint32_t *>(ready_quant + q16_base0)[0] = packed0;
        reinterpret_cast<uint32_t *>(ready_quant + q16_base1)[0] = packed1;
        if (lane8 == 0) {
            const float scale0 = token0 < n_tok
                ? __half2float(__float2half(1.0f / d_inv0)) : 0.0f;
            const float scale1 = token0 + 1 < n_tok
                ? __half2float(__float2half(1.0f / d_inv1)) : 0.0f;
            ready_scale[scale_index(
                ready_token_tiles, group, token_tile, qblock,
                token_in_tile)] = scale0;
            ready_scale[scale_index(
                ready_token_tiles, group, token_tile, qblock,
                token_in_tile + 1)] = scale1;
        }
      }
    }
#else
    (void)w; (void)x; (void)per_layer; (void)gate; (void)q8;
    (void)ready_quant; (void)ready_scale; (void)ready_token_tiles;
    (void)n_tok; (void)per_layer_off; (void)per_layer_stride;
#endif
}

template <bool ReadyOutput, bool DirectK>
inline bool configure_fused_gate() {
    static const bool configured = [] {
        const cudaError_t shared = cudaFuncSetAttribute(
            fused_gate<ReadyOutput, DirectK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(kSharedBytes));
        if (shared != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const cudaError_t carveout = cudaFuncSetAttribute(
            fused_gate<ReadyOutput, DirectK>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (carveout != cudaSuccess) cudaGetLastError();
        return carveout == cudaSuccess;
    }();
    return configured;
}

inline LaunchResult launch(
        const uint8_t * w, const BlockQ8_1Mmq * x,
        const float * per_layer, float * gate, BlockQ8_1Mmq * q8,
        uint32_t n_embd, uint32_t ple_width, uint32_t n_tok,
        uint32_t per_layer_off, uint32_t per_layer_stride,
        uint32_t sm_version, uint32_t canonical,
        cudaStream_t stream) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(w)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(per_layer)
        | reinterpret_cast<uintptr_t>(gate)
        | reinterpret_cast<uintptr_t>(q8);
    const bool direct_k = n_tok == 128;
    const bool canonical_stream_k = n_tok > kTokens - kTileTokens
        && n_tok <= kTokens;
    if (sm_version != 86 || !canonical || !w || !x || !per_layer
        || !gate || !q8 || (pointers & 15u) != 0
        || n_embd != kInput || ple_width != kOutput
        || (!direct_k && !canonical_stream_k)
        || per_layer_stride < kOutput
        || per_layer_off > per_layer_stride
        || kOutput > per_layer_stride - per_layer_off) {
        return LaunchResult::NotSupported;
    }
    const uint32_t grid = (kOutput / kRows)
        * ((n_tok + kTileTokens - 1) / kTileTokens);
    if (direct_k) {
        if (!configure_fused_gate<false, true>()) {
            return LaunchResult::NotSupported;
        }
        fused_gate<false, true>
            <<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
                w, x, per_layer, gate, q8, nullptr, nullptr, 0,
                n_tok, per_layer_off, per_layer_stride);
    } else {
        if (!configure_fused_gate<false, false>()) {
            return LaunchResult::NotSupported;
        }
        fused_gate<false, false>
            <<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
                w, x, per_layer, gate, q8, nullptr, nullptr, 0,
                n_tok, per_layer_off, per_layer_stride);
    }
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

inline LaunchResult launch_ready(
        const uint8_t * w, const BlockQ8_1Mmq * x,
        const float * per_layer, float * gate, uint16_t * ready_quant,
        float * ready_scale, uint32_t ready_token_tiles,
        uint32_t n_embd, uint32_t ple_width, uint32_t n_tok,
        uint32_t per_layer_off, uint32_t per_layer_stride,
        uint32_t sm_version, uint32_t canonical,
        cudaStream_t stream) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(w)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(per_layer)
        | reinterpret_cast<uintptr_t>(gate)
        | reinterpret_cast<uintptr_t>(ready_quant)
        | reinterpret_cast<uintptr_t>(ready_scale);
    const bool direct_k = n_tok == 128;
    const bool canonical_stream_k = n_tok > kTokens - kTileTokens
        && n_tok <= kTokens;
    if (sm_version != 86 || !canonical || !w || !x || !per_layer || !gate
        || !ready_quant || !ready_scale || (pointers & 15u) != 0
        || n_embd != kInput || ple_width != kOutput
        || (!direct_k && !canonical_stream_k)
        || (direct_k
            ? ready_token_tiles != 16u
            : ready_token_tiles < (n_tok + 7u) / 8u)
        || per_layer_stride < kOutput
        || per_layer_off > per_layer_stride
        || kOutput > per_layer_stride - per_layer_off) {
        return LaunchResult::NotSupported;
    }
    const uint32_t grid = (kOutput / kRows)
        * ((n_tok + kTileTokens - 1) / kTileTokens);
    if (direct_k) {
        if (!configure_fused_gate<true, true>()) {
            return LaunchResult::NotSupported;
        }
        fused_gate<true, true>
            <<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
                w, x, per_layer, gate, nullptr,
                ready_quant, ready_scale, ready_token_tiles,
                n_tok, per_layer_off, per_layer_stride);
    } else {
        if (!configure_fused_gate<true, false>()) {
            return LaunchResult::NotSupported;
        }
        fused_gate<true, false>
            <<<grid, dim3(32, kWarps), kSharedBytes, stream>>>(
                w, x, per_layer, gate, nullptr,
                ready_quant, ready_scale, ready_token_tiles,
                n_tok, per_layer_off, per_layer_stride);
    }
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_ple_gate
