#pragma once

// Research-only SM86 W4A8 digit-sliced kernel lab.
//
// This file deliberately has no production launcher, selector, ABI surface, or
// dependency on the runtime workflow.  It proves one exact K32 arithmetic
// building block before any engine integration is considered:
//
//   q4 = s4(q4_storage_nibble ^ 8)
//   q8 = u4(q8 & 0x0f) + 16 * s4((q8 >> 4) & 0x0f)
//   dot(q4, q8) = mma.s4.u4(q4, low) + 16 * mma.s4.s4(q4, high)
//
// The integer halves are combined before the existing per-K32 Q4/Q8 scale is
// applied and before FP32 accumulation.  That boundary is part of the lab
// contract: moving either scale inside an MMA changes numerical semantics.

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define IMPARO_W4A8_LAB_HD __host__ __device__
#else
#define IMPARO_W4A8_LAB_HD
#endif

namespace imparo_sm86_w4a8_digit_lab {

constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;
constexpr int kOutputs = kM * kN;
constexpr int kPackedQ4Bytes = kM * kK / 2;
constexpr int kBenchmarkRawTiles = 256;

IMPARO_W4A8_LAB_HD constexpr std::uint8_t low_nibble(std::uint8_t value) {
    return value & 0x0fu;
}

IMPARO_W4A8_LAB_HD constexpr int signed_nibble(std::uint8_t bits) {
    const int value = int(bits & 0x0fu);
    return value >= 8 ? value - 16 : value;
}

// Q4_0 stores an unsigned nibble and subtracts eight at dequantization. XORing
// bit three converts that representation to the two's-complement s4 bits that
// mma.sync consumes, without changing any value.
IMPARO_W4A8_LAB_HD constexpr std::uint8_t q4_storage_to_s4_bits(
        std::uint8_t storage) {
    return low_nibble(storage) ^ 0x08u;
}

IMPARO_W4A8_LAB_HD constexpr int q4_storage_value(std::uint8_t storage) {
    return int(low_nibble(storage)) - 8;
}

IMPARO_W4A8_LAB_HD constexpr std::uint8_t q8_low_u4(std::int8_t value) {
    return low_nibble(static_cast<std::uint8_t>(value));
}

IMPARO_W4A8_LAB_HD constexpr std::uint8_t q8_high_s4_bits(
        std::int8_t value) {
    return (static_cast<std::uint8_t>(value) >> 4) & 0x0fu;
}

IMPARO_W4A8_LAB_HD constexpr int q8_high_s4(std::int8_t value) {
    return signed_nibble(q8_high_s4_bits(value));
}

IMPARO_W4A8_LAB_HD constexpr int reconstruct_q8(std::int8_t value) {
    return int(q8_low_u4(value)) + 16 * q8_high_s4(value);
}

IMPARO_W4A8_LAB_HD constexpr bool scalar_identity_holds(
        std::uint8_t q4_storage, std::int8_t q8) {
    const int q4 = q4_storage_value(q4_storage);
    const int direct = q4 * int(q8);
    const int sliced = q4 * int(q8_low_u4(q8))
        + 16 * q4 * q8_high_s4(q8);
    return signed_nibble(q4_storage_to_s4_bits(q4_storage)) == q4
        && reconstruct_q8(q8) == int(q8)
        && direct == sliced;
}

#if defined(__CUDACC__)

__device__ __forceinline__ std::uint32_t pack_nibble(
        std::uint32_t packed, std::uint8_t nibble, int index) {
    return packed | (std::uint32_t(nibble & 0x0fu) << (4 * index));
}

__device__ __forceinline__ std::uint32_t pack_byte(
        std::uint32_t packed, std::int8_t value, int index) {
    return packed
        | (std::uint32_t(static_cast<std::uint8_t>(value)) << (8 * index));
}

__device__ __forceinline__ std::uint8_t load_packed_q4_storage(
        const std::uint8_t * packed_q4, int row, int k) {
    const std::uint8_t byte = packed_q4[row * (kK / 2) + k / 2];
    return (k & 1) != 0 ? (byte >> 4) & 0x0fu : byte & 0x0fu;
}

// PTX m16n8k32 s4 A-fragment layout:
//   group = lane / 4, thread = lane % 4
//   register 0: A[group][thread*8 + 0..7]
//   register 1: A[group+8][thread*8 + 0..7]
__device__ __forceinline__ void load_q4_a_fragment(
        std::uint32_t (&fragment)[2], const std::uint8_t * q4_storage,
        int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int k_base = thread_in_group * 8;
    fragment[0] = 0;
    fragment[1] = 0;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        fragment[0] = pack_nibble(fragment[0], q4_storage_to_s4_bits(
            q4_storage[group * kK + k_base + item]), item);
        fragment[1] = pack_nibble(fragment[1], q4_storage_to_s4_bits(
            q4_storage[(group + 8) * kK + k_base + item]), item);
    }
}

// PTX m16n8k32 s4/u4 B-fragment layout for column-major B.  The lab stores
// Q8 as [N][K], so group selects N and the packed register walks eight K rows.
__device__ __forceinline__ void load_q8_digit_fragments(
        std::uint32_t & low_u4, std::uint32_t & high_s4,
        const std::int8_t * q8, int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int k_base = thread_in_group * 8;
    low_u4 = 0;
    high_s4 = 0;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const std::int8_t value = q8[group * kK + k_base + item];
        low_u4 = pack_nibble(low_u4, q8_low_u4(value), item);
        high_s4 = pack_nibble(high_s4, q8_high_s4_bits(value), item);
    }
}

__device__ __forceinline__ void load_q4_digit_fragment_from_packed(
        std::uint32_t (&fragment)[2],
        const std::uint8_t * packed_q4, int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int k_base = thread_in_group * 8;
    fragment[0] = 0;
    fragment[1] = 0;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        fragment[0] = pack_nibble(fragment[0], q4_storage_to_s4_bits(
            load_packed_q4_storage(
                packed_q4, group, k_base + item)), item);
        fragment[1] = pack_nibble(fragment[1], q4_storage_to_s4_bits(
            load_packed_q4_storage(
                packed_q4, group + 8, k_base + item)), item);
    }
}

// PTX m16n8k32 s8 A-fragment layout.  This is the exact expanded control for
// the digit-sliced route: both represent the same 16x32 signed matrix.
__device__ __forceinline__ void load_q4_control_fragment(
        std::uint32_t (&fragment)[4], const std::uint8_t * q4_storage,
        int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int row = (reg & 1) != 0 ? group + 8 : group;
        const int k_base = thread_in_group * 4 + (reg >= 2 ? 16 : 0);
        fragment[reg] = 0;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            fragment[reg] = pack_byte(fragment[reg],
                static_cast<std::int8_t>(q4_storage_value(
                    q4_storage[row * kK + k_base + item])), item);
        }
    }
}

__device__ __forceinline__ void load_q4_control_fragment_from_packed(
        std::uint32_t (&fragment)[4],
        const std::uint8_t * packed_q4, int lane) {
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
        const int row = (reg & 1) != 0 ? group + 8 : group;
        const int k_base = thread_in_group * 4 + (reg >= 2 ? 16 : 0);
        fragment[reg] = 0;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            fragment[reg] = pack_byte(fragment[reg],
                static_cast<std::int8_t>(q4_storage_value(
                    load_packed_q4_storage(
                        packed_q4, row, k_base + item))), item);
        }
    }
}

__device__ __forceinline__ void load_q8_control_fragment(
        std::uint32_t (&fragment)[2],
        const std::int8_t * q8, int lane) {
    const int column = lane >> 2;
    const int thread_in_group = lane & 3;
#pragma unroll
    for (int reg = 0; reg < 2; ++reg) {
        const int k_base = thread_in_group * 4 + reg * 16;
        fragment[reg] = 0;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            fragment[reg] = pack_byte(fragment[reg],
                q8[column * kK + k_base + item], item);
        }
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

__device__ __forceinline__ void mma_m16n8k32_s4_u4(
        std::int32_t (&accumulator)[4],
        const std::uint32_t (&a)[2], std::uint32_t b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(accumulator[0]), "+r"(accumulator[1]),
          "+r"(accumulator[2]), "+r"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ void mma_m16n8k32_s4_s4(
        std::int32_t (&accumulator)[4],
        const std::uint32_t (&a)[2], std::uint32_t b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(accumulator[0]), "+r"(accumulator[1]),
          "+r"(accumulator[2]), "+r"(accumulator[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ int accumulator_row(int lane, int item) {
    return (item >> 1) * 8 + lane / 4;
}

__device__ __forceinline__ int accumulator_column(int lane, int item) {
    return (lane & 3) * 2 + (item & 1);
}

__global__ __launch_bounds__(32, 1) void single_k32_control_oracle(
        const std::uint8_t * q4_storage,
        const std::int8_t * q8,
        const float * q4_scale,
        const float * q8_scale,
        const float * initial_output,
        std::int32_t * integer_output,
        float * scaled_output) {
    const int lane = int(threadIdx.x) & 31;
    std::uint32_t a[4];
    std::uint32_t b[2];
    load_q4_control_fragment(a, q4_storage, lane);
    load_q8_control_fragment(b, q8, lane);

    std::int32_t accumulator[4] = {0, 0, 0, 0};
    mma_m16n8k32_s8_s8(accumulator, a, b);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int row = accumulator_row(lane, item);
        const int column = accumulator_column(lane, item);
        const int index = row * kN + column;
        integer_output[index] = accumulator[item];
        const float block_scale = q4_scale[row] * q8_scale[column];
        scaled_output[index] = fmaf(
            float(accumulator[item]), block_scale, initial_output[index]);
    }
}

// One-warp, one-K32 oracle kernel.  It is intentionally not a throughput
// kernel: direct global fragment construction isolates digit arithmetic and
// PTX register mapping from any future staging/pipeline design.
__global__ __launch_bounds__(32, 1) void single_k32_fragment_oracle(
        const std::uint8_t * q4_storage,
        const std::int8_t * q8,
        const float * q4_scale,
        const float * q8_scale,
        const float * initial_output,
        std::int32_t * integer_output,
        float * scaled_output) {
    const int lane = int(threadIdx.x) & 31;
    std::uint32_t a[2];
    std::uint32_t b_low;
    std::uint32_t b_high;
    load_q4_a_fragment(a, q4_storage, lane);
    load_q8_digit_fragments(b_low, b_high, q8, lane);

    std::int32_t low[4] = {0, 0, 0, 0};
    std::int32_t high[4] = {0, 0, 0, 0};
    mma_m16n8k32_s4_u4(low, a, b_low);
    mma_m16n8k32_s4_s4(high, a, b_high);

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int row = accumulator_row(lane, item);
        const int column = accumulator_column(lane, item);
        const int index = row * kN + column;
        const std::int32_t combined = low[item] + 16 * high[item];
        integer_output[index] = combined;

        // Exact boundary under test: combine the two integer MMAs first, then
        // apply one d4*d8 scale for this K32 block, then accumulate in FP32.
        const float block_scale = q4_scale[row] * q8_scale[column];
        scaled_output[index] = fmaf(
            float(combined), block_scale, initial_output[index]);
    }
}

constexpr int kBenchmarkWarps = 8;
constexpr int kBenchmarkThreads = kBenchmarkWarps * 32;

__device__ __forceinline__ void initialize_benchmark_accumulators(
        float (&accumulator)[4], float (&block_scale)[4],
        const float * q4_scale, const float * q8_scale,
        const float * initial_output, int lane) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int row = accumulator_row(lane, item);
        const int column = accumulator_column(lane, item);
        const int index = row * kN + column;
        accumulator[item] = initial_output[index];
        block_scale[item] = q4_scale[row] * q8_scale[column];
    }
}

__device__ __forceinline__ void store_benchmark_accumulators(
        const float (&accumulator)[4], float * output, int lane) {
    const int warp = int(threadIdx.x) >> 5;
    const std::uint64_t global_warp = std::uint64_t(blockIdx.x)
        * (blockDim.x / 32) + warp;
    const std::uint64_t output_base = global_warp * kOutputs;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int row = accumulator_row(lane, item);
        const int column = accumulator_column(lane, item);
        output[output_base + row * kN + column] = accumulator[item];
    }
}

// Consumer-only upper-bound control: all expanded fragments are prepared by
// the host.  The timed loop contains exactly one s8xs8 MMA and the unchanged
// K32 scale/FP32 accumulation boundary.
__global__ __launch_bounds__(kBenchmarkThreads, 1)
void consumer_control_benchmark(
        const std::uint32_t * ready_a,
        const std::uint32_t * ready_b,
        const float * q4_scale,
        const float * q8_scale,
        const float * initial_output,
        int iterations,
        float * output) {
    const int lane = int(threadIdx.x) & 31;
    std::uint32_t a[4];
    std::uint32_t b[2];
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        a[index] = ready_a[lane * 4 + index];
    }
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        b[index] = ready_b[lane * 2 + index];
    }
    float accumulator[4];
    float block_scale[4];
    initialize_benchmark_accumulators(
        accumulator, block_scale, q4_scale, q8_scale, initial_output, lane);

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::int32_t dot[4] = {0, 0, 0, 0};
        mma_m16n8k32_s8_s8(dot, a, b);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            accumulator[item] = fmaf(
                float(dot[item]), block_scale[item], accumulator[item]);
        }
    }
    store_benchmark_accumulators(accumulator, output, lane);
}

// Consumer-only digit candidate: the two compact fragments and two Q8 digit
// fragments are prepared.  Integer combination precedes the same scale/FMA.
__global__ __launch_bounds__(kBenchmarkThreads, 1)
void consumer_digit_benchmark(
        const std::uint32_t * ready_a,
        const std::uint32_t * ready_b_low,
        const std::uint32_t * ready_b_high,
        const float * q4_scale,
        const float * q8_scale,
        const float * initial_output,
        int iterations,
        float * output) {
    const int lane = int(threadIdx.x) & 31;
    std::uint32_t a[2] = {
        ready_a[lane * 2], ready_a[lane * 2 + 1]};
    const std::uint32_t b_low = ready_b_low[lane];
    const std::uint32_t b_high = ready_b_high[lane];
    float accumulator[4];
    float block_scale[4];
    initialize_benchmark_accumulators(
        accumulator, block_scale, q4_scale, q8_scale, initial_output, lane);

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::int32_t low[4] = {0, 0, 0, 0};
        std::int32_t high[4] = {0, 0, 0, 0};
        mma_m16n8k32_s4_u4(low, a, b_low);
        mma_m16n8k32_s4_s4(high, a, b_high);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const std::int32_t combined = low[item] + 16 * high[item];
            accumulator[item] = fmaf(
                float(combined), block_scale[item], accumulator[item]);
        }
    }
    store_benchmark_accumulators(accumulator, output, lane);
}

// Honest control and candidate start each timed K32 iteration from exactly the
// same raw packed Q4 bytes and raw Q8 bytes in a 128-KiB rotating tile pool.
// The iteration/warp-dependent address prevents packing from being hoisted,
// while normal cached LDG semantics avoid turning the lab into a system-memory
// serialization benchmark.  The pool exceeds one SM's L1 but remains L2-hot.
__global__ __launch_bounds__(kBenchmarkThreads, 1)
void honest_control_benchmark(
        const std::uint8_t * packed_q4,
        const std::int8_t * q8,
        const float * q4_scale,
        const float * q8_scale,
        const float * initial_output,
        int iterations,
        float * output) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int warp_seed = int(blockIdx.x) * kBenchmarkWarps + warp;
    float accumulator[4];
    float block_scale[4];
    initialize_benchmark_accumulators(
        accumulator, block_scale, q4_scale, q8_scale, initial_output, lane);

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int tile = (warp_seed + iteration)
            & (kBenchmarkRawTiles - 1);
        const std::uint8_t * raw_q4 =
            packed_q4 + tile * kPackedQ4Bytes;
        const std::int8_t * raw_q8 = q8 + tile * (kN * kK);
        std::uint32_t a[4];
        std::uint32_t b[2];
        load_q4_control_fragment_from_packed(a, raw_q4, lane);
        load_q8_control_fragment(b, raw_q8, lane);
        std::int32_t dot[4] = {0, 0, 0, 0};
        mma_m16n8k32_s8_s8(dot, a, b);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            accumulator[item] = fmaf(
                float(dot[item]), block_scale[item], accumulator[item]);
        }
    }
    store_benchmark_accumulators(accumulator, output, lane);
}

__global__ __launch_bounds__(kBenchmarkThreads, 1)
void honest_digit_benchmark(
        const std::uint8_t * packed_q4,
        const std::int8_t * q8,
        const float * q4_scale,
        const float * q8_scale,
        const float * initial_output,
        int iterations,
        float * output) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int warp_seed = int(blockIdx.x) * kBenchmarkWarps + warp;
    float accumulator[4];
    float block_scale[4];
    initialize_benchmark_accumulators(
        accumulator, block_scale, q4_scale, q8_scale, initial_output, lane);

#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int tile = (warp_seed + iteration)
            & (kBenchmarkRawTiles - 1);
        const std::uint8_t * raw_q4 =
            packed_q4 + tile * kPackedQ4Bytes;
        const std::int8_t * raw_q8 = q8 + tile * (kN * kK);
        std::uint32_t a[2];
        std::uint32_t b_low;
        std::uint32_t b_high;
        load_q4_digit_fragment_from_packed(a, raw_q4, lane);
        load_q8_digit_fragments(b_low, b_high, raw_q8, lane);
        std::int32_t low[4] = {0, 0, 0, 0};
        std::int32_t high[4] = {0, 0, 0, 0};
        mma_m16n8k32_s4_u4(low, a, b_low);
        mma_m16n8k32_s4_s4(high, a, b_high);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const std::int32_t combined = low[item] + 16 * high[item];
            accumulator[item] = fmaf(
                float(combined), block_scale[item], accumulator[item]);
        }
    }
    store_benchmark_accumulators(accumulator, output, lane);
}

#endif  // defined(__CUDACC__)

}  // namespace imparo_sm86_w4a8_digit_lab

#undef IMPARO_W4A8_LAB_HD
