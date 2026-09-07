#pragma once

#include <algorithm>
#include <cstdint>

namespace imparo_cuda_host_measure {

constexpr uint64_t kMib = 1ull << 20;
constexpr uint64_t kMinimumSampleBytes = 8ull * kMib;
constexpr uint64_t kMaximumSampleBytes = 64ull * kMib;
constexpr uint64_t kTargetMeasuredBytes = 256ull * kMib;
constexpr uint64_t kSampleAlignment = 4096;

struct BenchmarkPlan {
    uint64_t sample_bytes = 0;
    uint32_t repetitions = 0;
};

// Keep the probe a small, bounded fraction of current free resources. Constants
// describe measurement quality and pressure limits, never a model or GPU shape.
inline BenchmarkPlan benchmark_plan(uint64_t device_free,
                                    uint64_t host_available) {
    const uint64_t candidate = std::min(device_free / 32, host_available / 128);
    if (candidate < kMinimumSampleBytes) return {};
    uint64_t bytes = std::min(candidate, kMaximumSampleBytes);
    bytes = bytes / kSampleAlignment * kSampleAlignment;
    if (bytes < kMinimumSampleBytes) return {};
    const uint64_t repetitions64 = std::clamp(
        kTargetMeasuredBytes / bytes
            + uint64_t(kTargetMeasuredBytes % bytes != 0),
        uint64_t(4), uint64_t(32));
    return {bytes, uint32_t(repetitions64)};
}

inline uint64_t bytes_per_second(uint64_t sample_bytes,
                                 uint32_t repetitions,
                                 float elapsed_ms) {
    if (!sample_bytes || !repetitions || !(elapsed_ms > 0.0f)) return 0;
    if (sample_bytes > UINT64_MAX / repetitions) return 0;
    const long double total = static_cast<long double>(sample_bytes)
        * static_cast<long double>(repetitions);
    const long double rate = total * 1000.0L
        / static_cast<long double>(elapsed_ms);
    if (!(rate > 0.0L) || rate > static_cast<long double>(UINT64_MAX)) return 0;
    return static_cast<uint64_t>(rate);
}

} // namespace imparo_cuda_host_measure
