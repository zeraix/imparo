#pragma once

// Experimental SM86 single-token Q8_0 TileMajor Gate/Up producer. One CTA
// owns exactly one 32-value SwiGLU block, preserving the established projection
// accumulation order before emitting the incumbent BlockQ8_1 wire layout.
namespace imparo_sm86_q8_tm_gate_up_q8_sidecar_lab {

enum class LaunchResult : uint8_t { NotSupported, Launched, Error };

constexpr uint32_t kCohortWarps = 4;
constexpr uint32_t kTotalWarps = 2 * kCohortWarps;
constexpr uint32_t kRowsPerRound = 8;
constexpr uint32_t kRounds = 4;
constexpr uint32_t kRows = kRowsPerRound * kRounds;
constexpr uint32_t kStageBlocks = 32;
constexpr uint32_t kPayloadBytes =
    kStageBlocks * kRowsPerRound * 32;
constexpr uint32_t kScaleBytes =
    kStageBlocks * kRowsPerRound * sizeof(__half);
constexpr uint32_t kStageBytes = kPayloadBytes + kScaleBytes;
constexpr uint32_t kProjectionBytes = 2 * kStageBytes;
constexpr uint32_t kReductionBytes =
    2 * (kCohortWarps - 1) * kRowsPerRound * 32 * sizeof(float);
constexpr uint32_t kResultBytes =
    2 * kRowsPerRound * sizeof(float);
constexpr uint32_t kGatedBytes = kRows * sizeof(float);
constexpr uint32_t kSharedBytes = kProjectionBytes + kGatedBytes;

static_assert(kStageBytes == 8704, "Q8_0_TM projection stage budget");
static_assert(kProjectionBytes == 17408,
              "Q8_0_TM dual-projection stage budget");
static_assert(kReductionBytes + kResultBytes <= kProjectionBytes,
              "reduction storage must reuse the projection stages");
static_assert(kSharedBytes == 17536, "private sidecar shared budget");

__device__ __forceinline__ void stage_weight(
        const uint8_t * __restrict__ payload,
        const __half * __restrict__ scales,
        uint8_t * stage, uint32_t valid, uint32_t projection_tid) {
    const auto * source_values = reinterpret_cast<const uint4 *>(payload);
    auto * destination_values = reinterpret_cast<uint4 *>(stage);
    for (uint32_t vector = projection_tid; vector < valid * 16;
         vector += kCohortWarps * 32) {
        destination_values[vector] = source_values[vector];
    }

    const auto * source_scale_vectors =
        reinterpret_cast<const uint4 *>(scales);
    auto * destination_scale_vectors = reinterpret_cast<uint4 *>(
        stage + kPayloadBytes);
    for (uint32_t unit = projection_tid; unit < valid;
         unit += kCohortWarps * 32) {
        destination_scale_vectors[unit] = source_scale_vectors[unit];
    }
}

__launch_bounds__(256)
__global__ void q8_0_tm_q8_1_gate_up_sidecar(
        const uint8_t * __restrict__ gate_w,
        const uint8_t * __restrict__ up_w,
        const BlockQ8_1 * __restrict__ x,
        BlockQ8_1 * __restrict__ y,
        uint32_t n_in, uint32_t n_out) {
    extern __shared__ __align__(16) uint8_t shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t projection = warp / kCohortWarps;
    const uint32_t projection_warp = warp % kCohortWarps;
    const uint32_t projection_tid = projection_warp * 32 + lane;
    const uint32_t blocks = n_in / 32;
    const uint8_t * weights = projection == 0 ? gate_w : up_w;
    const auto * scales = reinterpret_cast<const __half *>(
        weights + uint64_t(n_out) * n_in);
    uint8_t * weight_stage = shared + projection * kStageBytes;
    auto * reductions = reinterpret_cast<
        float (*)[kCohortWarps - 1][kRowsPerRound][32]>(shared);
    auto * results = reinterpret_cast<float *>(shared + kReductionBytes);
    auto * gated = reinterpret_cast<float *>(shared + kProjectionBytes);
    const uint32_t local_block = projection_tid / 4;
    const uint32_t iqs = 2 * (projection_tid & 3);

#pragma unroll 1
    for (uint32_t round = 0; round < kRounds; ++round) {
        float partial[kRowsPerRound] = {};
        const uint32_t row_tile = blockIdx.x * kRounds + round;
        const uint64_t first_unit = uint64_t(row_tile) * blocks;

        for (uint32_t stage = 0; stage < blocks; stage += kStageBlocks) {
            const uint32_t valid = min(kStageBlocks, blocks - stage);
            stage_weight(
                weights + (first_unit + stage) * kRowsPerRound * 32,
                scales + (first_unit + stage) * kRowsPerRound,
                weight_stage, valid, projection_tid);
            __syncthreads();

            if (local_block < valid) {
                const uint32_t block = stage + local_block;
                const BlockQ8_1 * activation = x + block;
                const int * activation_words =
                    reinterpret_cast<const int *>(activation->qs);
                const float activation_scale = __half2float(activation->d);
                const auto * staged_scales = reinterpret_cast<const __half *>(
                    weight_stage + kPayloadBytes);
#pragma unroll
                for (uint32_t local_row = 0;
                     local_row < kRowsPerRound; ++local_row) {
                    const uint8_t * values = weight_stage
                        + local_block * kRowsPerRound * 32
                        + local_row * 32;
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
                            local_block * kRowsPerRound + local_row])
                        * activation_scale * float(sumi);
                }
            }
            __syncthreads();
        }

        if (projection_warp > 0) {
#pragma unroll
            for (uint32_t local_row = 0;
                 local_row < kRowsPerRound; ++local_row) {
                reductions[projection][projection_warp - 1]
                    [local_row][lane] = partial[local_row];
            }
        }
        __syncthreads();

        if (projection_warp == 0) {
#pragma unroll
            for (uint32_t local_row = 0;
                 local_row < kRowsPerRound; ++local_row) {
#pragma unroll
                for (uint32_t other = 0;
                     other < kCohortWarps - 1; ++other) {
                    partial[local_row] +=
                        reductions[projection][other][local_row][lane];
                }
                const float result = warp_sum_xor(partial[local_row]);
                if (lane == 0) {
                    results[projection * kRowsPerRound + local_row] = result;
                }
            }
        }
        __syncthreads();

        const uint32_t tid = warp * 32 + lane;
        if (tid < kRowsPerRound) {
            const float gate = results[tid];
            const float up = results[kRowsPerRound + tid];
            gated[round * kRowsPerRound + tid] =
                imparo_cuda_lfm2::silu(gate) * up;
        }
        __syncthreads();
    }

    // Every eight-lane subgroup repeats the incumbent quantizer's arithmetic.
    // Only subgroup zero commits, but all 32 lanes remain active for full-mask
    // shuffles so the reduction instruction sequence is identical.
    if (warp == 0) {
        const uint32_t qlane = lane & 7u;
        const float4 xi = reinterpret_cast<const float4 *>(gated)[qlane];
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffff, amax, off, 32));
        }
        float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
        }
        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = int8_t(roundf(xi.x * d_inv));
        q.y = int8_t(roundf(xi.y * d_inv));
        q.z = int8_t(roundf(xi.z * d_inv));
        q.w = int8_t(roundf(xi.w * d_inv));
        const float d = 1.0f / d_inv;
        BlockQ8_1 * out = y + blockIdx.x;
        if (lane < 8) reinterpret_cast<char4 *>(out->qs)[lane] = q;
        if (lane == 0) {
            out->d = __float2half(d);
            out->s = __float2half(sum);
        }
    }
}

inline LaunchResult launch(
        const uint8_t * gate_w, const uint8_t * up_w,
        const BlockQ8_1 * x, BlockQ8_1 * y,
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
    q8_0_tm_q8_1_gate_up_sidecar
        <<<(n_out + kRows - 1) / kRows, dim3(32, kTotalWarps),
            kSharedBytes, stream>>>(gate_w, up_w, x, y, n_in, n_out);
    return cudaPeekAtLastError() == cudaSuccess
        ? LaunchResult::Launched : LaunchResult::Error;
}

} // namespace imparo_sm86_q8_tm_gate_up_q8_sidecar_lab
