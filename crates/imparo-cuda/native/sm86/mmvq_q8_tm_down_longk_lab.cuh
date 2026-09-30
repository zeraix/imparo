#pragma once

// Laboratory-only SM86 long-K Q8_0 TileMajor decode reader. Four compute
// warps retain the incumbent ownership and reduction tree while four loader
// warps prefetch the next Stage32 weight tile into the inactive shared buffer.
namespace imparo_sm86_q8_tm_down_longk_lab {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kComputeWarps = 4;
constexpr uint32_t kLoaderWarps = 4;
constexpr uint32_t kTotalWarps = kComputeWarps + kLoaderWarps;
constexpr uint32_t kRows = 8;
constexpr uint32_t kStageBlocks = 32;
constexpr uint32_t kPayloadBytes = kStageBlocks * kRows * 32;
constexpr uint32_t kScaleBytes = kStageBlocks * kRows * sizeof(__half);
constexpr uint32_t kStageBytes = kPayloadBytes + kScaleBytes;
constexpr uint32_t kSharedBytes = 2 * kStageBytes;
constexpr uint32_t kReductionBytes =
    (kComputeWarps - 1) * kRows * 32 * sizeof(float);

static_assert(kStageBytes == 8704, "Q8_0_TM Stage32 byte budget");
static_assert(kSharedBytes == 17408, "Q8_0_TM double-buffer byte budget");
static_assert(kReductionBytes <= kSharedBytes,
              "final reduction must reuse the completed pipeline storage");

__device__ __forceinline__ void copy_global_to_shared_16(
        void * destination, const void * source) {
    const uint32_t shared_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
        : : "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void commit_async_copies() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void wait_async_copies() {
    asm volatile("cp.async.wait_all;");
}

__device__ __forceinline__ void stage_weight_async(
        const uint8_t * __restrict__ payload,
        const __half * __restrict__ scales,
        uint8_t * stage, uint32_t valid, uint32_t loader_tid) {
    const auto * source_values = reinterpret_cast<const uint4 *>(payload);
    auto * destination_values = reinterpret_cast<uint4 *>(stage);
    for (uint32_t vector = loader_tid; vector < valid * 16;
         vector += kLoaderWarps * 32) {
        copy_global_to_shared_16(destination_values + vector,
                                 source_values + vector);
    }

    const auto * source_scale_vectors =
        reinterpret_cast<const uint4 *>(scales);
    auto * destination_scale_vectors = reinterpret_cast<uint4 *>(
        stage + kPayloadBytes);
    for (uint32_t unit = loader_tid; unit < valid;
         unit += kLoaderWarps * 32) {
        copy_global_to_shared_16(destination_scale_vectors + unit,
                                 source_scale_vectors + unit);
    }
}

__launch_bounds__(256)
__global__ void q8_0_tm_q8_1_single_longk(
        const uint8_t * __restrict__ w,
        const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y,
        uint32_t n_in, uint32_t n_out,
        uint32_t out_stride, uint32_t row_base) {
    extern __shared__ __align__(16) uint8_t shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const bool loader = warp >= kComputeWarps;
    const uint32_t loader_tid = (warp - kComputeWarps) * 32 + lane;
    const uint32_t compute_tid = warp * 32 + lane;
    const uint32_t row_tile = blockIdx.x;
    const uint32_t row0 = row_tile * kRows;
    const uint32_t blocks = n_in / 32;
    const uint64_t first_unit = uint64_t(row_tile) * blocks;
    const auto * source_scales = reinterpret_cast<const __half *>(
        w + uint64_t(n_out) * n_in);
    uint8_t * current = shared;
    uint8_t * next = shared + kStageBytes;
    float partial[kRows] = {};

    if (loader) {
        const uint32_t valid = min(kStageBlocks, blocks);
        stage_weight_async(
            w + first_unit * kRows * 32,
            source_scales + first_unit * kRows,
            current, valid, loader_tid);
        commit_async_copies();
        wait_async_copies();
    }
    __syncthreads();

    for (uint32_t stage = 0; stage < blocks; stage += kStageBlocks) {
        const uint32_t valid = min(kStageBlocks, blocks - stage);
        const bool has_next = stage + kStageBlocks < blocks;
        if (loader && has_next) {
            const uint32_t next_stage = stage + kStageBlocks;
            const uint32_t next_valid =
                min(kStageBlocks, blocks - next_stage);
            stage_weight_async(
                w + (first_unit + next_stage) * kRows * 32,
                source_scales + (first_unit + next_stage) * kRows,
                next, next_valid, loader_tid);
            commit_async_copies();
        }

        if (!loader) {
            const uint32_t local_block = compute_tid / 4;
            const uint32_t iqs = 2 * (compute_tid & 3);
            if (local_block < valid) {
                const uint32_t block = stage + local_block;
                const BlockQ8_1 * activation = x + block;
                const int * activation_words =
                    reinterpret_cast<const int *>(activation->qs);
                const float activation_scale =
                    __half2float(activation->d);
                const auto * staged_scales =
                    reinterpret_cast<const __half *>(
                        current + kPayloadBytes);
#pragma unroll
                for (uint32_t local_row = 0; local_row < kRows; ++local_row) {
                    const uint8_t * values = current
                        + local_block * kRows * 32 + local_row * 32;
                    int sumi = 0;
#pragma unroll
                    for (uint32_t item = 0; item < 2; ++item) {
                        sumi = __dp4a(
                            imparo_sm80_q8_mmvq::load_q8_0_i32(
                                values, iqs + item),
                            activation_words[iqs + item], sumi);
                    }
                    partial[local_row] +=
                        __half2float(staged_scales[
                            local_block * kRows + local_row])
                        * activation_scale * float(sumi);
                }
            }
        } else if (has_next) {
            wait_async_copies();
        }

        // The barrier both publishes the prefetched tile and prevents loaders
        // from recycling the just-consumed buffer before all compute warps finish.
        __syncthreads();
        uint8_t * swap = current;
        current = next;
        next = swap;
    }

    auto * warp_partial = reinterpret_cast<float (*)[kRows][32]>(shared);
    if (!loader && warp > 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < kRows; ++local_row) {
            warp_partial[warp - 1][local_row][lane] = partial[local_row];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < kRows; ++local_row) {
#pragma unroll
            for (uint32_t other = 0; other < kComputeWarps - 1; ++other) {
                partial[local_row] += warp_partial[other][local_row][lane];
            }
            const float result = warp_sum_xor(partial[local_row]);
            const uint32_t row = row0 + local_row;
            if (lane == 0 && row < n_out) {
                y[uint64_t(0) * out_stride + row_base + row] = result;
            }
        }
    }
}

inline LaunchResult launch(
        const uint8_t * w, const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok,
        uint32_t out_stride, uint32_t row_base,
        uint32_t sm_version, cudaStream_t stream) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(w)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(y);
    if (sm_version != 86 || !w || !x || !y || (pointers & 15u) != 0
            || n_tok != 1 || !n_in || n_in % 32 != 0
            || !n_out || n_out % kRows != 0
            || out_stride < row_base
            || uint64_t(row_base) + n_out > out_stride) {
        return LaunchResult::NotSupported;
    }
    q8_0_tm_q8_1_single_longk
        <<<(n_out + kRows - 1) / kRows, dim3(32, kTotalWarps),
            kSharedBytes, stream>>>(
                w, x, y, n_in, n_out, out_stride, row_base);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_q8_tm_down_longk_lab
