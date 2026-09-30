#pragma once

// Research-only SM86 packed-shared, single-s8-MMA Gate A lab.
//
// This header is intentionally disconnected from every production launcher,
// selector, ABI, workflow, and Metal path.  Both kernels consume the exact same
// qblock-major Q4 payload:
//
//   packed_q4[projection][qblock][row][4 words]
//   q4_scale[projection][qblock][row]              (f32)
//   q8[qblock][token][32 bytes]
//   q8_scale[qblock][token]                        (f32)
//
// A CTA owns 16 rows x 64 tokens.  The control expands packed Q4 into shared
// s8.  The candidate keeps four packed words per row in shared, expands one A
// fragment into registers, and reuses that fragment across eight N8 token
// fragments.  Both routes then issue exactly one m16n8k32 s8xs8 MMA per token
// fragment and retain identical per-K32 f32 scale/FMA ordering.

#include <cstdint>

#include <cuda_runtime.h>

namespace imparo_sm86_packed_shared_s8_lab {

constexpr int kRowsPerCta = 16;
constexpr int kTokensPerCta = 64;
constexpr int kKPerQblock = 32;
constexpr int kWordsPerQblock = 4;
constexpr int kTokenFragments = 8;
constexpr int kOutputsPerFragment = 4;
constexpr int kThreads = 32;

struct alignas(16) ControlShared {
    std::int8_t expanded_a[kRowsPerCta][kKPerQblock];
    float a_scale[kRowsPerCta];
    alignas(16) std::int8_t b[kTokensPerCta][kKPerQblock];
    float b_scale[kTokensPerCta];
};

struct alignas(16) CandidateShared {
    std::uint32_t packed_a[kRowsPerCta][kWordsPerQblock];
    float a_scale[kRowsPerCta];
    alignas(16) std::int8_t b[kTokensPerCta][kKPerQblock];
    float b_scale[kTokensPerCta];
};

static_assert(sizeof(ControlShared) == 2880,
    "control shared footprint is part of the Gate A receipt");
static_assert(sizeof(CandidateShared) == 2624,
    "candidate shared footprint is part of the Gate A receipt");

__device__ __forceinline__ std::int8_t q4_storage_value(
        std::uint32_t packed, int nibble) {
    return static_cast<std::int8_t>(
        int((packed >> (4 * nibble)) & 0x0fu) - 8);
}

__device__ __forceinline__ std::uint32_t unpack_four_q4_to_s8(
        std::uint32_t packed_word, int half) {
    const std::uint32_t packed = packed_word >> (half * 16);
    std::uint32_t expanded = 0;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const std::int8_t value = q4_storage_value(packed, item);
        expanded |= std::uint32_t(static_cast<std::uint8_t>(value))
            << (8 * item);
    }
    return expanded;
}

__device__ __forceinline__ void load_control_a_fragment(
        std::uint32_t (&fragment)[4],
        const std::int8_t * expanded_a, int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int row = (reg & 1) != 0 ? group + 8 : group;
        const int k_base = thread_in_group * 4 + (reg >= 2 ? 16 : 0);
        fragment[reg] = *reinterpret_cast<const std::uint32_t *>(
            expanded_a + row * kKPerQblock + k_base);
    }
}

__device__ __forceinline__ void load_candidate_a_fragment(
        std::uint32_t (&fragment)[4],
        const std::uint32_t * packed_a, int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int row = (reg & 1) != 0 ? group + 8 : group;
        const int k_base = thread_in_group * 4 + (reg >= 2 ? 16 : 0);
        const int word = k_base / 8;
        const int half = (k_base & 4) != 0 ? 1 : 0;
        fragment[reg] = unpack_four_q4_to_s8(
            packed_a[row * kWordsPerQblock + word], half);
    }
}

__device__ __forceinline__ void load_b_fragment(
        std::uint32_t (&fragment)[2], const std::int8_t * b,
        int token_fragment, int lane) {
    const int column = lane >> 2;
    const int thread_in_group = lane & 3;
    const int token = token_fragment * 8 + column;
#pragma unroll
    for (int reg = 0; reg < 2; ++reg) {
        const int k_base = thread_in_group * 4 + reg * 16;
        fragment[reg] = *reinterpret_cast<const std::uint32_t *>(
            b + token * kKPerQblock + k_base);
    }
}

__device__ __forceinline__ void mma_m16n8k32_s8_s8(
        std::int32_t (&accumulator)[4],
        const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+r"(accumulator[0]), "+r"(accumulator[1]),
          "+r"(accumulator[2]), "+r"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ int accumulator_row(int lane, int item) {
    return (item >> 1) * 8 + lane / 4;
}

__device__ __forceinline__ int accumulator_column(int lane, int item) {
    return (lane & 3) * 2 + (item & 1);
}

template <bool PackedShared>
__device__ __forceinline__ void run_tile(
        const std::uint32_t * packed_q4,
        const float * q4_scale,
        const std::int8_t * q8,
        const float * q8_scale,
        float * output,
        int projections,
        int qblocks,
        int rows,
        int tokens,
        std::int8_t * expanded_a,
        std::uint32_t * packed_a,
        float * shared_a_scale,
        std::int8_t * shared_b,
        float * shared_b_scale) {
    const int lane = int(threadIdx.x);
    const int row_base = int(blockIdx.x) * kRowsPerCta;
    const int token_base = int(blockIdx.y) * kTokensPerCta;
    const int projection = int(blockIdx.z);
    if (projection >= projections || row_base >= rows || token_base >= tokens) {
        return;
    }

    float accumulator[kTokenFragments][kOutputsPerFragment];
#pragma unroll
    for (int fragment = 0; fragment < kTokenFragments; ++fragment) {
#pragma unroll
        for (int item = 0; item < kOutputsPerFragment; ++item) {
            accumulator[fragment][item] = 0.0f;
        }
    }

#pragma unroll 1
    for (int qblock = 0; qblock < qblocks; ++qblock) {
        // Both routes read the same 4-word raw payload.  The control writes 8
        // expanded bytes per word; the candidate writes the word unchanged.
        for (int flat = lane; flat < kRowsPerCta * kWordsPerQblock;
                flat += kThreads) {
            const int local_row = flat / kWordsPerQblock;
            const int word = flat % kWordsPerQblock;
            const int global_row = row_base + local_row;
            std::uint32_t value = 0;
            if (global_row < rows) {
                const std::uint64_t index =
                    ((std::uint64_t(projection) * qblocks + qblock) * rows
                        + global_row) * kWordsPerQblock + word;
                value = packed_q4[index];
            }
            if constexpr (PackedShared) {
                packed_a[local_row * kWordsPerQblock + word] = value;
            } else {
#pragma unroll
                for (int nibble = 0; nibble < 8; ++nibble) {
                    expanded_a[local_row * kKPerQblock + word * 8 + nibble]
                        = q4_storage_value(value, nibble);
                }
            }
        }
        if (lane < kRowsPerCta) {
            const int global_row = row_base + lane;
            shared_a_scale[lane] = global_row < rows
                ? q4_scale[(std::uint64_t(projection) * qblocks + qblock)
                    * rows + global_row]
                : 0.0f;
        }
        for (int flat = lane; flat < kTokensPerCta * kKPerQblock;
                flat += kThreads) {
            const int local_token = flat / kKPerQblock;
            const int k = flat % kKPerQblock;
            const int global_token = token_base + local_token;
            shared_b[flat] = global_token < tokens
                ? q8[(std::uint64_t(qblock) * tokens + global_token)
                    * kKPerQblock + k]
                : 0;
        }
        for (int local_token = lane; local_token < kTokensPerCta;
                local_token += kThreads) {
            const int global_token = token_base + local_token;
            shared_b_scale[local_token] = global_token < tokens
                ? q8_scale[std::uint64_t(qblock) * tokens + global_token]
                : 0.0f;
        }
        __syncwarp();

        std::uint32_t a[4];
        if constexpr (PackedShared) {
            load_candidate_a_fragment(a, packed_a, lane);
        } else {
            load_control_a_fragment(a, expanded_a, lane);
        }

#pragma unroll
        for (int fragment = 0; fragment < kTokenFragments; ++fragment) {
            std::uint32_t b[2];
            load_b_fragment(b, shared_b, fragment, lane);
            std::int32_t dot[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8_s8(dot, a, b);
#pragma unroll
            for (int item = 0; item < kOutputsPerFragment; ++item) {
                const int local_row = accumulator_row(lane, item);
                const int column = accumulator_column(lane, item);
                const int local_token = fragment * 8 + column;
                const float scale = shared_a_scale[local_row]
                    * shared_b_scale[local_token];
                accumulator[fragment][item] = fmaf(float(dot[item]), scale,
                    accumulator[fragment][item]);
            }
        }
        __syncwarp();
    }

#pragma unroll
    for (int fragment = 0; fragment < kTokenFragments; ++fragment) {
#pragma unroll
        for (int item = 0; item < kOutputsPerFragment; ++item) {
            const int local_row = accumulator_row(lane, item);
            const int column = accumulator_column(lane, item);
            const int global_row = row_base + local_row;
            const int global_token = token_base + fragment * 8 + column;
            if (global_row < rows && global_token < tokens) {
                output[(std::uint64_t(projection) * rows + global_row) * tokens
                    + global_token] = accumulator[fragment][item];
            }
        }
    }
}

__global__ __launch_bounds__(kThreads, 2) void expanded_shared_control(
        const std::uint32_t * packed_q4,
        const float * q4_scale,
        const std::int8_t * q8,
        const float * q8_scale,
        float * output,
        int projections,
        int qblocks,
        int rows,
        int tokens) {
    __shared__ ControlShared shared;
    run_tile<false>(packed_q4, q4_scale, q8, q8_scale, output,
        projections, qblocks, rows, tokens,
        &shared.expanded_a[0][0], nullptr, shared.a_scale,
        &shared.b[0][0], shared.b_scale);
}

__global__ __launch_bounds__(kThreads, 2) void packed_shared_candidate(
        const std::uint32_t * packed_q4,
        const float * q4_scale,
        const std::int8_t * q8,
        const float * q8_scale,
        float * output,
        int projections,
        int qblocks,
        int rows,
        int tokens) {
    __shared__ CandidateShared shared;
    run_tile<true>(packed_q4, q4_scale, q8, q8_scale, output,
        projections, qblocks, rows, tokens,
        nullptr, &shared.packed_a[0][0], shared.a_scale,
        &shared.b[0][0], shared.b_scale);
}

}  // namespace imparo_sm86_packed_shared_s8_lab
