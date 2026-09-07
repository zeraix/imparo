// SPDX-License-Identifier: MIT
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Imparo contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Derived from the Ampere Q8_0 MMQ arithmetic and selector contract in llama.cpp
// commit 4695f001fece1660d8bb1b3748f50726ddcc100b: mmq.cuh,
// mmq-load-tiles.cuh, mmq-vec-dot.cuh, mma.cuh and mmq-config-ampere.cuh.

#pragma once

#include "mmq_q8_replay_plan.h"

// Ampere Q8_0 x Q8_1 MMQ for batches above the MMVQ boundary.
//
// This header deliberately owns only the architecture tile.  Weight paging,
// activation-scratch lifetime, and the <= 8 column MMVQ decision remain common
// runtime policy.  It is included after mmq_q4_q8_1.cuh and reuses that file's
// Ampere MMA fragment ABI, accumulator mapping, and asynchronous 16-byte copy.
//
// The host selector mirrors pinned llama.cpp 4695f001:
//   * I = 128 output rows, K_vram = 256 values, 256 threads;
//   * J is selected from the registered Ampere Q8_0 table by minimizing the
//     number of output-column tiles, retaining the first J on a tie;
//   * a non-multiple-of-128 output row count uses llama's fallback J set.
// No caller/model token count is special-cased.
namespace imparo_sm80_q8_mmq {

constexpr uint32_t kRows = 128;
constexpr uint32_t kWarps = 8;
constexpr uint32_t kBlockValues = 32;
constexpr uint32_t kStageBlocks = 8;
constexpr uint32_t kStageValues = kBlockValues * kStageBlocks;
constexpr uint32_t kActivationRecordValues = 4 * kBlockValues;
constexpr uint32_t kActivationStride = sizeof(BlockQ8_1Mmq);

// 256 signed bytes, eight f32 scales, and the same 16-byte bank-skew tail used
// by the existing Ampere MMQ A-fragment loader.  304 bytes is 4 modulo the
// 32-byte ldmatrix bank period.
constexpr uint32_t kWeightStride = kStageValues
    + kStageBlocks * sizeof(float) + 16;
static_assert(kWeightStride == 304, "Q8 MMQ shared weight stride");
static_assert(kActivationStride == 144, "Q8 MMQ activation record size");
static_assert(kRows == imparo_sm80_q8_replay::kRows, "planner row ABI");
enum class LaunchResult : uint8_t {
    Launched,
    NotSupported,
    Error,
};

static_assert(kBlockValues == imparo_sm80_q8_replay::kBlockValues,
    "planner block ABI");
static_assert(kStageBlocks == imparo_sm80_q8_replay::kStageBlocks,
    "planner stage ABI");

template <uint32_t Tokens>
constexpr uint32_t shared_bytes() {
    static_assert(Tokens >= 8 && Tokens <= 128 && Tokens % 8 == 0,
                  "registered Q8 MMQ token tile");
    return kRows * kWeightStride
        + 2 * Tokens * kActivationStride;
}

__device__ __forceinline__ int load_q8_word(const uint8_t * values,
                                             uint32_t word) {
    const uint8_t * p = values + 4 * word;
    return int(uint32_t(p[0]) | (uint32_t(p[1]) << 8)
        | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24));
}

enum : uint32_t { kReplaySuffix = 0, kReplayFirstPrefix = 1 };


template <uint32_t Tokens, uint32_t Warps = kWarps>
__device__ __forceinline__ void stage_activation_group(
        const BlockQ8_1Mmq * __restrict__ x, int8_t * __restrict__ dst,
        uint32_t n_tok, uint32_t tile_token, uint32_t active_tokens,
        uint32_t group_index, bool group_valid, uint32_t tid) {
    static_assert(sizeof(BlockQ8_1Mmq) % sizeof(uint4) == 0,
                  "MMQ activation vector alignment");
    constexpr uint32_t vectors_per_record =
        sizeof(BlockQ8_1Mmq) / sizeof(uint4);
    uint4 * out = reinterpret_cast<uint4 *>(dst);
    for (uint32_t linear = tid; linear < Tokens * vectors_per_record;
         linear += Warps * 32) {
        const uint32_t token = linear / vectors_per_record;
        const uint32_t vector = linear % vectors_per_record;
        if (group_valid && token < active_tokens) {
            const uint4 * in = reinterpret_cast<const uint4 *>(
                x + uint64_t(group_index) * n_tok + tile_token + token);
            imparo_sm80_mmq::copy_global_to_shared_16(
                out + linear, in + vector);
        } else {
            out[linear] = make_uint4(0, 0, 0, 0);
        }
    }
}

// One regular grid tile.  This is the reusable arithmetic primitive; a later
// physical Stream-K owner can split the ascending [segment_begin, segment_end)
// block interval without changing weight unpack, MMA, scale, or tail semantics.
template <uint32_t Tokens, bool AlignedWholeK = false, bool TileMajor = false,
          uint32_t Epilogue = 0, bool AsyncTileMajor = false,
          uint32_t Warps = kWarps>
__global__ __launch_bounds__(Warps * 32, 1) void q8_0_q8_1_mma(
        const uint8_t * __restrict__ w,
        const BlockQ8_1Mmq * __restrict__ x,
        float * __restrict__ y,
        BlockQ8_1Mmq * __restrict__ output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base,
        uint32_t physical_grid, uint32_t replay_phase,
        float * __restrict__ replay_prefix, uint32_t multi_seam) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t token_groups = (Tokens + 31) / 32;
    constexpr uint32_t row_groups = kRows / 32;
    constexpr uint32_t token_group_partitions = (Warps / 2) / row_groups;
    constexpr uint32_t local_token_groups =
        (token_groups + token_group_partitions - 1) / token_group_partitions;
    constexpr uint32_t row_fragments = 2;
    constexpr uint32_t partial_count =
        local_token_groups * 2 * row_fragments * 4;
    static_assert(Warps == 8 || Warps == 16,
                  "registered Q8 MMQ warp count");
    static_assert((Warps / 2) % row_groups == 0,
                  "warp pairs must partition complete row groups");
    static_assert(Epilogue <= 4, "registered Q8 MMQ epilogue");
    static_assert(!AsyncTileMajor || TileMajor,
                  "async weight staging is a tile-major specialization");

    extern __shared__ __align__(16) int8_t shared[];
    int8_t * sx = shared;
    int8_t * sy = sx + kRows * kWeightStride;

    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t token_tiles = (n_tok + Tokens - 1) / Tokens;
    const uint32_t first_global_tile = row_base / kRows;
    const uint32_t global_tile_index = first_global_tile + blockIdx.x;
    const uint32_t global_tile_row = global_tile_index * kRows;
    const int64_t slice_tile_row = int64_t(global_tile_row) - row_base;
    const uint32_t tile_token = blockIdx.y * Tokens;
    const uint32_t blocks = n_in / kBlockValues;
    uint32_t segment_begin = 0;
    uint32_t segment_end = blocks;
    if constexpr (!AlignedWholeK) {
        const uint64_t logical_tile =
            uint64_t(global_tile_index) * token_tiles + blockIdx.y;
        const uint64_t logical_tiles =
            uint64_t((out_stride + kRows - 1) / kRows) * token_tiles;
        const uint64_t total_work = logical_tiles * blocks;
        __shared__ uint32_t replay_segment[3];
        if (tid == 0) {
            const imparo_sm80_q8_replay::Segment segment =
                imparo_sm80_q8_replay::segment_for_phase(
                    logical_tile, blocks, total_work, physical_grid, replay_phase);
            replay_segment[0] = segment.begin;
            replay_segment[1] = segment.end;
            replay_segment[2] = segment.valid ? 1u : 0u;
        }
        __syncthreads();
        segment_begin = replay_segment[0];
        segment_end = replay_segment[1];
        if (!replay_segment[2]) return;
    }
    const uint32_t active_tokens = tile_token < n_tok
        ? min(Tokens, n_tok - tile_token) : 0;

    float partial[partial_count];
#pragma unroll
    for (uint32_t item = 0; item < partial_count; ++item) {
        partial[item] = 0.0f;
    }

    // Each source Q8_0 block contributes one independently scaled 32-value
    // integer dot.  Keep blocks and phases ascending to match the pinned route.
    for (uint32_t stage_block = segment_begin; stage_block < segment_end;
         stage_block += kStageBlocks) {
#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase) {
            const uint32_t group_block = stage_block + phase * 4;
            stage_activation_group<Tokens, Warps>(
                x, sy + phase * Tokens * kActivationStride,
                n_tok, tile_token, active_tokens, group_block / 4,
                AlignedWholeK || group_block < segment_end, tid);
        }
        imparo_sm80_mmq::commit_async_copies();

        constexpr uint32_t staged_blocks = kRows * kStageBlocks;
        for (uint32_t linear = tid; linear < staged_blocks;
             linear += Warps * 32) {
            // TM assigns consecutive lanes to the eight rows of one on-file
            // unit. Row-major keeps consecutive lanes on adjacent K blocks.
            const uint32_t qblock = TileMajor
                ? linear / kRows : linear % kStageBlocks;
            const uint32_t local_row = TileMajor
                ? linear % kRows : linear / kStageBlocks;
            const int64_t slice_row = slice_tile_row + local_row;
            const uint32_t kb = stage_block + qblock;
            int8_t * values = sx + local_row * kWeightStride
                + qblock * kBlockValues;
            float scale = 0.0f;
            if (AlignedWholeK || (slice_row >= 0 && uint64_t(slice_row) < n_out
                && kb < segment_end)) {
                const uint8_t * weight_values = nullptr;
                __half weight_d;
                if constexpr (TileMajor) {
                    const uint64_t row = uint64_t(slice_row);
                    const uint64_t unit = (row / 8) * blocks + kb;
                    weight_values = w + unit * 256 + (row & 7) * 32;
                    const auto * scales = reinterpret_cast<const __half *>(
                        w + uint64_t(n_out) * n_in);
                    weight_d = scales[unit * 8 + (row & 7)];
                } else {
                    const uint8_t * block =
                        w + (uint64_t(slice_row) * blocks + kb) * 34;
                    weight_values = block + 2;
                    weight_d = *reinterpret_cast<const __half *>(block);
                }
                if constexpr (TileMajor) {
                    const auto * packed =
                        reinterpret_cast<const uint4 *>(weight_values);
                    auto * staged = reinterpret_cast<uint4 *>(values);
                    if constexpr (AsyncTileMajor) {
                        imparo_sm80_mmq::copy_global_to_shared_16(staged, packed);
                        imparo_sm80_mmq::copy_global_to_shared_16(
                            staged + 1, packed + 1);
                    } else {
                        staged[0] = packed[0];
                        staged[1] = packed[1];
                    }
                } else {
#pragma unroll
                    for (uint32_t word = 0; word < 8; ++word) {
                        reinterpret_cast<int *>(values)[word] =
                            load_q8_word(weight_values, word);
                    }
                }
                scale = __half2float(weight_d);
            } else {
#pragma unroll
                for (uint32_t word = 0; word < 8; ++word) {
                    reinterpret_cast<int *>(values)[word] = 0;
                }
            }
            reinterpret_cast<float *>(
                sx + local_row * kWeightStride + kStageValues)[qblock] = scale;
        }
        if constexpr (AsyncTileMajor) {
            imparo_sm80_mmq::commit_async_copies();
        }

        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();

#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase) {
            const int8_t * sy_phase =
                sy + phase * Tokens * kActivationStride;
#pragma unroll
            for (uint32_t qblock = 0; qblock < 4; ++qblock) {
                const uint32_t local_kb = phase * 4 + qblock;
                if constexpr (!AlignedWholeK) {
                    if (stage_block + local_kb >= segment_end) continue;
                }

                int af[row_fragments][4];
                float d8w[row_fragments][2];
#pragma unroll
                for (uint32_t row_fragment = 0;
                     row_fragment < row_fragments; ++row_fragment) {
                    const uint32_t local_row0 =
                        (warp / (2 * token_group_partitions)) * 32
                            + row_fragment * 16;
                    imparo_sm80_mmq::load_a_m16n8k32(
                        af[row_fragment],
                        sx + local_row0 * kWeightStride
                            + local_kb * kBlockValues,
                        kWeightStride);
#pragma unroll
                    for (uint32_t scale_item = 0; scale_item < 2;
                         ++scale_item) {
                        const uint32_t local_row = local_row0
                            + imparo_sm80_mmq::accumulator_row(
                                lane, scale_item * 2);
                        d8w[row_fragment][scale_item] =
                            reinterpret_cast<const float *>(
                                sx + local_row * kWeightStride
                                    + kStageValues)[local_kb];
                    }
                }

#pragma unroll
                for (uint32_t token_group =
                         (warp >> 1) % token_group_partitions;
                     token_group < token_groups;
                     token_group += token_group_partitions) {
#pragma unroll
                    for (uint32_t token_fragment = 0;
                         token_fragment < 2; ++token_fragment) {
                        const uint32_t local_token0 = token_group * 32
                            + (warp & 1) * 16 + token_fragment * 8;
                        if (local_token0 >= active_tokens) continue;

                        int bf[2];
                        imparo_sm80_mmq::load_b_m16n8k32(
                            bf, sy_phase + local_token0 * kActivationStride
                                + qblock * kBlockValues,
                            kActivationStride);
                        float d8a[2];
#pragma unroll
                        for (uint32_t scale_item = 0; scale_item < 2;
                             ++scale_item) {
                            const uint32_t local_token = local_token0
                                + imparo_sm80_mmq::accumulator_token(
                                    lane, scale_item);
                            d8a[scale_item] =
                                reinterpret_cast<const float *>(
                                    sy_phase + local_token * kActivationStride
                                        + kActivationRecordValues)[qblock];
                        }
#pragma unroll
                        for (uint32_t row_fragment = 0;
                             row_fragment < row_fragments; ++row_fragment) {
                            int cf[4] = {};
                            imparo_sm80_mmq::mma_m16n8k32(
                                cf, af[row_fragment], bf);
#pragma unroll
                            for (uint32_t item = 0; item < 4; ++item) {
                                const uint32_t local_token_group =
                                    token_group / token_group_partitions;
                                const uint32_t sum_index =
                                    (((local_token_group * 2 + token_fragment)
                                        * row_fragments + row_fragment) * 4)
                                    + item;
                                partial[sum_index] += float(cf[item])
                                    * d8w[row_fragment][item / 2]
                                    * d8a[item % 2];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t token_group = (warp >> 1) % token_group_partitions;
         token_group < token_groups;
         token_group += token_group_partitions) {
#pragma unroll
        for (uint32_t token_fragment = 0;
             token_fragment < 2; ++token_fragment) {
#pragma unroll
            for (uint32_t row_fragment = 0;
                 row_fragment < row_fragments; ++row_fragment) {
#pragma unroll
                for (uint32_t item = 0; item < 4; ++item) {
                    const uint32_t local_row =
                        (warp / (2 * token_group_partitions)) * 32
                        + row_fragment * 16
                        + imparo_sm80_mmq::accumulator_row(lane, item);
                    const uint32_t local_token = token_group * 32
                        + (warp & 1) * 16 + token_fragment * 8
                        + imparo_sm80_mmq::accumulator_token(lane, item);
                    const int64_t slice_row = slice_tile_row + local_row;
                    const uint32_t global_row = global_tile_row + local_row;
                    const uint32_t token = tile_token + local_token;
                    const bool valid_output = token < n_tok
                        && (AlignedWholeK || (slice_row >= 0
                            && uint64_t(slice_row) < n_out
                            && global_row < out_stride));
                    if (!valid_output) continue;
                    const uint32_t local_token_group =
                        token_group / token_group_partitions;
                    const uint32_t sum_index =
                        (((local_token_group * 2 + token_fragment) * row_fragments
                            + row_fragment) * 4) + item;
                    float * slot = y + uint64_t(token) * out_stride + global_row;
                    if constexpr (AlignedWholeK) {
                        if constexpr (Epilogue >= 2 && Epilogue <= 4) {
                            const float gate = *slot;
                            const float value = imparo_cuda_lfm2::silu(gate)
                                * partial[sum_index];
                            if constexpr (Epilogue != 4) {
                                *slot = value;
                            }
                            if constexpr (Epilogue == 3 || Epilogue == 4) {
                                reinterpret_cast<float *>(shared)[
                                    local_token * kRows + local_row] = value;
                            }
                        } else {
                            *slot = partial[sum_index];
                        }
                    } else {
                        if (replay_phase == kReplaySuffix) {
                            *slot = partial[sum_index];
                        } else if (!multi_seam) {
                            *slot += partial[sum_index];
                        } else {
                            float * prefix = replay_prefix
                                + uint64_t(token) * n_out + uint64_t(slice_row);
                            if (replay_phase == kReplayFirstPrefix) {
                                *prefix = partial[sum_index];
                            } else {
                                *prefix += partial[sum_index];
                            }
                        }
                    }
                }
            }
        }
    }
    if constexpr (Epilogue == 3 || Epilogue == 4) {
        __syncthreads();
        const float * tile = reinterpret_cast<const float *>(shared);
        constexpr uint32_t vectors_per_token = kRows / 4;
        for (uint32_t linear = tid;
             linear < active_tokens * vectors_per_token;
             linear += Warps * 32) {
            const uint32_t local_token = linear / vectors_per_token;
            const uint32_t vector = linear % vectors_per_token;
            const uint32_t i0 = vector * 4;
            const float4 xi = reinterpret_cast<const float4 *>(
                tile + uint64_t(local_token) * kRows)[vector];
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
            const uint32_t token = tile_token + local_token;
            const uint32_t output_group = global_tile_row / kRows;
            BlockQ8_1Mmq * out = output_q8
                + uint64_t(output_group) * n_tok + token;
            reinterpret_cast<char4 *>(
                out->qs + block_in_group * 32)[iqs / 4] = quant;
            if (iqs == 0) {
                out->d[block_in_group] = d;
            }
        }
    }
#else
    (void)w; (void)x; (void)y; (void)output_q8;
    (void)n_in; (void)n_out; (void)n_tok;
    (void)out_stride; (void)row_base; (void)physical_grid;
    (void)replay_phase; (void)replay_prefix; (void)multi_seam;
#endif
}

__global__ void add_replay_prefix(
        float * y, const float * prefix, uint32_t slice_n_out,
        uint32_t n_tok, uint32_t out_stride, uint32_t row_base) {
    const uint32_t slice_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (slice_row >= slice_n_out) return;
    for (uint32_t token = blockIdx.y; token < n_tok; token += gridDim.y) {
        const uint64_t linear = uint64_t(token) * slice_n_out + slice_row;
        y[uint64_t(token) * out_stride + row_base + slice_row] += prefix[linear];
    }
}

template <uint32_t Tokens, bool TileMajor = false>
inline LaunchResult launch_tile(
        const uint8_t * w, const BlockQ8_1Mmq * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base,
        const imparo_sm80_q8_replay::ReplayPlan & plan,
        float * replay_prefix, cudaStream_t stream) {
    // A paged tensor can call this once per row slice. Configure each template
    // once per CUDA device instead of paying a host-driver call for every slice.
    // A configuration failure returns to the safe F32 route for this dispatch.
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return LaunchResult::Error;
    static int configured_device = -1;
    static bool configured = false;
    if (configured_device != device) {
        const cudaError_t attr = cudaFuncSetAttribute(
            q8_0_q8_1_mma<Tokens, false, TileMajor>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(shared_bytes<Tokens>()));
        configured = attr == cudaSuccess;
        configured_device = device;
        if (!configured) {
            if (attr == cudaErrorInvalidValue || attr == cudaErrorNotSupported) {
                (void)cudaGetLastError();
            } else {
                return LaunchResult::Error;
            }
        }
    }
    if (!configured) return LaunchResult::NotSupported;
    if (plan.status != imparo_sm80_q8_replay::PlanStatus::Ok
        || plan.tile_tokens != Tokens) return LaunchResult::NotSupported;
    const uint32_t token_tiles = plan.token_tiles;
    const uint32_t physical_grid = plan.physical_grid;
    const bool replay = plan.replay;
    const bool multi_seam = plan.multi_seam;
    if (multi_seam && !replay_prefix) return LaunchResult::NotSupported;
    const auto slice = imparo_sm80_q8_replay::slice_tiles(
        row_base, n_out, out_stride);
    if (!slice.valid || !slice.count) return LaunchResult::Error;
    const dim3 grid(slice.count, token_tiles);
    if (multi_seam) {
        uint64_t bytes = 0;
        if (!imparo_sm80_q8_replay::slice_workspace_bytes(plan, n_out, &bytes)
            || cudaMemsetAsync(replay_prefix, 0, size_t(bytes), stream)
                != cudaSuccess) {
            return LaunchResult::Error;
        }
    }
    q8_0_q8_1_mma<Tokens, false, TileMajor><<<grid, dim3(32, kWarps),
        shared_bytes<Tokens>(), stream>>>(
            w, x, y, nullptr, n_in, n_out, n_tok, out_stride, row_base,
            replay ? physical_grid : 0, kReplaySuffix,
            replay_prefix, multi_seam ? 1u : 0u);
    if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;
    if (replay) {
        for (uint32_t phase = 1; phase <= plan.prefix_phases; ++phase) {
            q8_0_q8_1_mma<Tokens, false, TileMajor><<<grid, dim3(32, kWarps),
                shared_bytes<Tokens>(), stream>>>(
                    w, x, y, nullptr, n_in, n_out, n_tok, out_stride, row_base,
                    physical_grid, phase, replay_prefix, multi_seam ? 1u : 0u);
            if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;
        }
        if (multi_seam) {
            const uint32_t add_grid_x = uint32_t(
                (uint64_t(n_out) + 255) / 256);
            add_replay_prefix<<<dim3(add_grid_x, token_tiles), 256, 0, stream>>>(
                y, replay_prefix, n_out, n_tok, out_stride, row_base);
            if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;
        }
    }
    return LaunchResult::Launched;
}

// Specialized path for resident, fully aligned projections. It preserves the
// regular 128x128 arithmetic tile while compiling out paging, replay, K-tail,
// and row-tail predicates. Dispatch is default-off and versioned by the tuner;
// ineligible shapes return to the common MMQ route without changing semantics.
template <bool TileMajor = false, uint32_t Epilogue = 0,
          bool AsyncTileMajor = false, uint32_t Warps = kWarps>
inline LaunchResult launch_aligned_whole_k(
        const uint8_t * w, const BlockQ8_1Mmq * x, float * y,
        BlockQ8_1Mmq * output_q8,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base, cudaStream_t stream,
        imparo_sm80_mmq::LaunchInfo * info = nullptr) {
    constexpr uint32_t Tokens = 128;
    static_assert(Epilogue == 0 || Epilogue == 2 || Epilogue == 3
        || Epilogue == 4,
        "aligned Q8 MMQ supports only registered SiLU sidecar stores");
    if (!w || !x || !y || n_tok <= 8 || n_in == 0 || n_in % 256 != 0
        || n_out == 0 || n_out % kRows != 0 || out_stride == 0
        || out_stride % kRows != 0 || row_base % kRows != 0
        || row_base > out_stride || n_out > out_stride - row_base
        || ((Epilogue == 3 || Epilogue == 4) && !output_q8)) {
        return LaunchResult::NotSupported;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return LaunchResult::Error;
    static int configured_device = -1;
    static bool configured = false;
    if (configured_device != device) {
        const cudaError_t attr = cudaFuncSetAttribute(
            q8_0_q8_1_mma<
                Tokens, true, TileMajor, Epilogue, AsyncTileMajor, Warps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(shared_bytes<Tokens>()));
        configured = attr == cudaSuccess;
        configured_device = device;
        if (!configured) {
            if (attr == cudaErrorInvalidValue || attr == cudaErrorNotSupported) {
                (void)cudaGetLastError();
            } else {
                return LaunchResult::Error;
            }
        }
    }
    if (!configured) return LaunchResult::NotSupported;

    const uint32_t token_tiles = (n_tok + Tokens - 1) / Tokens;
    const uint32_t row_tiles = n_out / kRows;
    q8_0_q8_1_mma<Tokens, true, TileMajor, Epilogue, AsyncTileMajor, Warps>
        <<<dim3(row_tiles, token_tiles),
        dim3(32, Warps), shared_bytes<Tokens>(), stream>>>(
            w, x, y, output_q8, n_in, n_out, n_tok, out_stride, row_base,
            0, kReplaySuffix, nullptr, 0);
    if (cudaPeekAtLastError() != cudaSuccess) return LaunchResult::Error;
    if (info) {
        *info = {};
        info->route = imparo_sm80_mmq::LaunchRoute::GridTile;
        info->tile_rows = kRows;
        info->tile_tokens = Tokens;
        info->logical_tiles = row_tiles * token_tiles;
        info->physical_blocks = info->logical_tiles;
        info->efficiency = 100;
    }
    return LaunchResult::Launched;
}

// `out_stride` is the complete tensor's row count even when `n_out` is one
// low-memory weight slice. It supplies llama's logical fallback-row predicate:
// paging must not change J or the numerical route. The kernel separately masks
// the current slice's physical row tail before every weight read and write.
template <bool TileMajor = false>
inline LaunchResult launch(
        const uint8_t * w, const BlockQ8_1Mmq * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base,
        const imparo_sm80_q8_replay::ReplayPlan & plan,
        float * replay_prefix, cudaStream_t stream,
        imparo_sm80_mmq::LaunchInfo * info = nullptr) {
    if (!w || !x || !y || n_tok <= 8 || n_in == 0 || n_in % 128 != 0
        || n_out == 0 || out_stride == 0 || row_base > out_stride
        || n_out > out_stride - row_base) {
        return LaunchResult::Error;
    }
    const uint32_t tokens = plan.tile_tokens;
    LaunchResult result = LaunchResult::NotSupported;
    switch (tokens) {
#define IMPARO_Q8_LAUNCH_CASE(T) case T: result = launch_tile<T, TileMajor>(w, x, y, \
            n_in, n_out, n_tok, out_stride, row_base, plan, \
            replay_prefix, stream); break
        IMPARO_Q8_LAUNCH_CASE(8);
        IMPARO_Q8_LAUNCH_CASE(16);
        IMPARO_Q8_LAUNCH_CASE(24);
        IMPARO_Q8_LAUNCH_CASE(32);
        IMPARO_Q8_LAUNCH_CASE(40);
        IMPARO_Q8_LAUNCH_CASE(48);
        IMPARO_Q8_LAUNCH_CASE(64);
        IMPARO_Q8_LAUNCH_CASE(80);
        IMPARO_Q8_LAUNCH_CASE(96);
        IMPARO_Q8_LAUNCH_CASE(112);
        IMPARO_Q8_LAUNCH_CASE(128);
#undef IMPARO_Q8_LAUNCH_CASE
        default: break;
    }
    if (info) {
        *info = {};
        if (result == LaunchResult::Launched) {
            info->route = imparo_sm80_mmq::LaunchRoute::GridTile;
            info->tile_rows = kRows;
            info->tile_tokens = tokens;
            info->logical_tiles = plan.logical_tiles;
            info->physical_blocks = plan.physical_grid;
            const uint64_t waves = (uint64_t(plan.logical_tiles)
                + plan.physical_grid - 1) / plan.physical_grid;
            info->efficiency = uint32_t(
                100 * plan.logical_tiles / (uint64_t(plan.physical_grid) * waves));
        }
    }
    return result;
}

} // namespace imparo_sm80_q8_mmq
