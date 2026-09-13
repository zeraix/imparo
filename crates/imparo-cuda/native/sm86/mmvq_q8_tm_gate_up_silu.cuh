#pragma once

// SM86 single-token Q8_0 TileMajor Gate/Up transaction. Two four-warp
// cohorts retain the established projection traversal and reduction tree,
// while sharing one launch and eliminating both dense intermediate writes.
namespace imparo_sm86_q8_tm_gate_up_silu {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kCohortWarps = 4;
constexpr uint32_t kTotalWarps = 2 * kCohortWarps;
constexpr uint32_t kRows = 8;
constexpr uint32_t kStageBlocks = 32;
constexpr uint32_t kPayloadBytes = kStageBlocks * kRows * 32;
constexpr uint32_t kScaleBytes = kStageBlocks * kRows * sizeof(__half);
constexpr uint32_t kStageBytes = kPayloadBytes + kScaleBytes;
constexpr uint32_t kReductionBytes =
    2 * (kCohortWarps - 1) * kRows * 32 * sizeof(float);

constexpr uint32_t kSharedBytes = kReductionBytes + 2 * kRows * sizeof(float);

static_assert(kStageBytes == 8704, "Q8_0_TM projection stage budget");
static_assert(kSharedBytes == 6208, "Q8 gated reduction-only shared budget");
static_assert(kReductionBytes + 2 * kRows * sizeof(float) <= kSharedBytes,
              "gated MMVQ reduction and results must fit shared memory");

__launch_bounds__(256)
__global__ void q8_0_tm_q8_1_gate_up_silu(
        const uint8_t * __restrict__ gate_w,
        const uint8_t * __restrict__ up_w,
        const BlockQ8_1 * __restrict__ x,
        float * __restrict__ y,
        uint32_t n_in, uint32_t n_out) {
    extern __shared__ __align__(16) uint8_t shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t projection = warp / kCohortWarps;
    const uint32_t projection_warp = warp % kCohortWarps;
    const uint32_t projection_tid = projection_warp * 32 + lane;
    const uint32_t row_tile = blockIdx.x;
    const uint32_t row0 = row_tile * kRows;
    const uint32_t blocks = n_in / 32;
    const uint64_t first_unit = uint64_t(row_tile) * blocks;
    const uint8_t * weights = projection == 0 ? gate_w : up_w;
    const auto * scales = reinterpret_cast<const __half *>(
        weights + uint64_t(n_out) * n_in);
    const uint32_t local_block = projection_tid / 4;
    const uint32_t iqs = 2 * (projection_tid & 3);
    float partial[kRows] = {};

    for (uint32_t stage = 0; stage < blocks; stage += kStageBlocks) {
        const uint32_t valid = min(kStageBlocks, blocks - stage);
        if (local_block < valid) {
            const uint32_t block = stage + local_block;
            const BlockQ8_1 * activation = x + block;
            const int * activation_words =
                reinterpret_cast<const int *>(activation->qs);
            const float activation_scale = __half2float(activation->d);
            const uint64_t unit = first_unit + block;
#pragma unroll
            for (uint32_t local_row = 0; local_row < kRows; ++local_row) {
                // TM payload and each row/word offset are word aligned. The
                // canonical Q8 byte-pack helper would emit scalar byte loads here.
                const uint8_t * values = weights
                    + unit * kRows * 32 + local_row * 32;
                int sumi = 0;
#pragma unroll
                for (uint32_t item = 0; item < 2; ++item) {
                    sumi = __dp4a(
                        reinterpret_cast<const int *>(values)[iqs + item],
                        activation_words[iqs + item], sumi);
                }
                partial[local_row] +=
                    __half2float(
                        scales[unit * kRows + local_row])
                    * activation_scale * float(sumi);
            }
        }
    }

    auto * reductions = reinterpret_cast<
        float (*)[kCohortWarps - 1][kRows][32]>(shared);
    if (projection_warp > 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < kRows; ++local_row) {
            reductions[projection][projection_warp - 1][local_row][lane] =
                partial[local_row];
        }
    }
    __syncthreads();

    auto * results = reinterpret_cast<float *>(shared + kReductionBytes);
    if (projection_warp == 0) {
#pragma unroll
        for (uint32_t local_row = 0; local_row < kRows; ++local_row) {
#pragma unroll
            for (uint32_t other = 0; other < kCohortWarps - 1; ++other) {
                partial[local_row] +=
                    reductions[projection][other][local_row][lane];
            }
            const float result = warp_sum_xor(partial[local_row]);
            if (lane == 0) results[projection * kRows + local_row] = result;
        }
    }
    __syncthreads();

    const uint32_t tid = warp * 32 + lane;
    if (tid < kRows) {
        const uint32_t row = row0 + tid;
        if (row < n_out) {
            const float gate = results[tid];
            const float up = results[kRows + tid];
            y[row] = imparo_cuda_lfm2::silu(gate) * up;
        }
    }
}

inline LaunchResult launch(
        const uint8_t * gate_w, const uint8_t * up_w,
        const BlockQ8_1 * x, float * y,
        uint32_t n_in, uint32_t n_out,
        uint32_t sm_version, cudaStream_t stream) {
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(gate_w)
        | reinterpret_cast<uintptr_t>(up_w)
        | reinterpret_cast<uintptr_t>(x)
        | reinterpret_cast<uintptr_t>(y);
    if (sm_version != 86 || !gate_w || !up_w || !x || !y
            || (pointers & 15u) != 0 || !n_in || n_in % 32 != 0
            || !n_out || n_out % kRows != 0) {
        return LaunchResult::NotSupported;
    }
    q8_0_tm_q8_1_gate_up_silu
        <<<(n_out + kRows - 1) / kRows, dim3(32, kTotalWarps),
            kSharedBytes, stream>>>(gate_w, up_w, x, y, n_in, n_out);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_q8_tm_gate_up_silu
