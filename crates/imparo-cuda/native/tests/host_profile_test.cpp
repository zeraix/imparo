#include "../host_profile.cuh"

#include <cassert>
#include <cstdint>

int main() {
    using namespace imparo_cuda_host_measure;
    assert(benchmark_plan(0, 1ull << 40).sample_bytes == 0);
    assert(benchmark_plan(1ull << 40, 0).sample_bytes == 0);
    assert(benchmark_plan(255ull << 20, 1ull << 40).sample_bytes == 0);
    assert(benchmark_plan(1ull << 40, 1023ull << 20).sample_bytes == 0);

    const BenchmarkPlan minimum = benchmark_plan(256ull << 20, 1ull << 40);
    assert(minimum.sample_bytes == 8ull << 20);
    assert(minimum.repetitions == 32);

    const BenchmarkPlan bounded = benchmark_plan(1ull << 40, 1ull << 40);
    assert(bounded.sample_bytes == 64ull << 20);
    assert(bounded.repetitions == 4);
    assert(bounded.sample_bytes % kSampleAlignment == 0);

    const BenchmarkPlan host_bound = benchmark_plan(1ull << 40, 2ull << 30);
    assert(host_bound.sample_bytes == 16ull << 20);
    assert(host_bound.repetitions == 16);

    assert(bytes_per_second(0, 4, 1.0f) == 0);
    assert(bytes_per_second(1, 0, 1.0f) == 0);
    assert(bytes_per_second(1, 1, 0.0f) == 0);
    assert(bytes_per_second(UINT64_MAX, 2, 1.0f) == 0);
    assert(bytes_per_second(1ull << 20, 4, 4.0f) == 1048576000ull);
    return 0;
}
