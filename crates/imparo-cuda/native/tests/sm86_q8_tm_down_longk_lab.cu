// Standalone SM86 component and performance lab for the long-K Q8_0
// TileMajor decode reader. This file is never linked into the backend library.

#if !defined(IMPARO_CUDA_KERNEL_LAB) || IMPARO_CUDA_KERNEL_LAB != 1
#error "sm86_q8_tm_down_longk_lab.cu is test-only"
#endif

#include "../imparo_cuda.cu"
#include "../sm86/mmvq_q8_tm_down_longk_lab.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

namespace Candidate = imparo_sm86_q8_tm_down_longk_lab;

constexpr std::size_t kGuardBytes = 256;
constexpr uint8_t kGuardValue = 0xa5;
constexpr uint32_t kInput = 10752;
constexpr uint32_t kOutput = 2048;

[[noreturn]] void fail(const char * operation, cudaError_t error) {
    std::fprintf(stderr, "q8-tm-down-longk-lab: %s: %s\n",
                 operation, cudaGetErrorString(error));
    std::exit(3);
}

void cuda_ok(cudaError_t error, const char * operation) {
    if (error != cudaSuccess) fail(operation, error);
}

struct GuardedBuffer {
    uint8_t * allocation = nullptr;
    uint8_t * data = nullptr;
    std::size_t bytes = 0;

    explicit GuardedBuffer(std::size_t requested) : bytes(requested) {
        cuda_ok(cudaMalloc(reinterpret_cast<void **>(&allocation),
                           bytes + 2 * kGuardBytes),
                "cudaMalloc guarded buffer");
        data = allocation + kGuardBytes;
        cuda_ok(cudaMemset(allocation, kGuardValue,
                           bytes + 2 * kGuardBytes),
                "initialize guarded buffer");
    }
    GuardedBuffer(const GuardedBuffer &) = delete;
    GuardedBuffer & operator=(const GuardedBuffer &) = delete;
    ~GuardedBuffer() {
        if (allocation) (void)cudaFree(allocation);
    }

    uint64_t guard_errors() const {
        std::vector<uint8_t> before(kGuardBytes);
        std::vector<uint8_t> after(kGuardBytes);
        cuda_ok(cudaMemcpy(before.data(), allocation, kGuardBytes,
                           cudaMemcpyDeviceToHost),
                "download leading guard");
        cuda_ok(cudaMemcpy(after.data(), data + bytes, kGuardBytes,
                           cudaMemcpyDeviceToHost),
                "download trailing guard");
        uint64_t errors = 0;
        for (uint8_t value : before) errors += value != kGuardValue;
        for (uint8_t value : after) errors += value != kGuardValue;
        return errors;
    }
};

template <typename T>
void upload(GuardedBuffer & destination, const std::vector<T> & source,
            const char * operation) {
    const std::size_t bytes = source.size() * sizeof(T);
    if (bytes != destination.bytes) {
        std::fprintf(stderr, "q8-tm-down-longk-lab: %s size mismatch\n",
                     operation);
        std::exit(3);
    }
    cuda_ok(cudaMemcpy(destination.data, source.data(), bytes,
                       cudaMemcpyHostToDevice), operation);
}

template <typename T>
std::vector<T> download(const GuardedBuffer & source, std::size_t count,
                        const char * operation) {
    if (count * sizeof(T) != source.bytes) {
        std::fprintf(stderr, "q8-tm-down-longk-lab: %s size mismatch\n",
                     operation);
        std::exit(3);
    }
    std::vector<T> result(count);
    cuda_ok(cudaMemcpy(result.data(), source.data, source.bytes,
                       cudaMemcpyDeviceToHost), operation);
    return result;
}

uint64_t byte_mismatches(const GuardedBuffer & source,
                         const void * expected) {
    const std::vector<uint8_t> actual = download<uint8_t>(
        source, source.bytes, "download immutable input");
    const auto * reference = static_cast<const uint8_t *>(expected);
    uint64_t mismatches = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        mismatches += actual[index] != reference[index];
    }
    return mismatches;
}

std::vector<uint8_t> make_tm_weights(uint32_t seed) {
    const uint32_t blocks = kInput / 32;
    const uint64_t payload_bytes = uint64_t(kOutput) * kInput;
    const uint64_t scale_bytes =
        uint64_t(kOutput) * blocks * sizeof(__half);
    std::vector<uint8_t> result(payload_bytes + scale_bytes);
    for (uint32_t row = 0; row < kOutput; ++row) {
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint64_t unit = uint64_t(row / 8) * blocks + block;
            uint8_t * values = result.data() + unit * 256
                + uint64_t(row & 7u) * 32;
            for (uint32_t lane = 0; lane < 32; ++lane) {
                const uint32_t mixed = seed + row * 19u + block * 37u
                    + lane * 53u + (row ^ block) * 7u;
                values[lane] = uint8_t(
                    int8_t(int32_t(mixed % 255u) - 127));
            }
            const __half scale = __float2half_rn(
                0.001953125f
                * float(1u + ((seed + row * 5u + block * 11u) % 31u)));
            const uint64_t scale_index = unit * 8 + (row & 7u);
            std::memcpy(result.data() + payload_bytes
                            + scale_index * sizeof(__half),
                        &scale, sizeof(scale));
        }
    }
    result.front() = uint8_t(int8_t(-127));
    result[payload_bytes - 1] = uint8_t(int8_t(126));
    return result;
}

std::vector<BlockQ8_1> make_activation(bool zero) {
    const uint32_t blocks = kInput / 32;
    std::vector<BlockQ8_1> result(blocks);
    for (uint32_t block = 0; block < blocks; ++block) {
        BlockQ8_1 & record = result[block];
        int32_t sum = 0;
        for (uint32_t lane = 0; lane < 32; ++lane) {
            const int8_t value = zero ? int8_t(0)
                : int8_t(int32_t((block * 29u + lane * 43u + 17u) % 255u)
                         - 127);
            record.qs[lane] = value;
            sum += int32_t(value);
        }
        const float scale = zero ? 0.0078125f
            : 0.00390625f * float(1u + ((block * 7u + 3u) % 17u));
        record.d = __float2half_rn(scale);
        record.s = __float2half_rn(scale * float(sum));
    }
    if (!zero) {
        result.front().qs[0] = -127;
        result.back().qs[31] = 126;
    }
    return result;
}

void launch_control(const GuardedBuffer & weights,
                    const GuardedBuffer & activation,
                    GuardedBuffer & output, cudaStream_t stream) {
    imparo_sm80_q8_mmvq::launch_tile_major(
        weights.data, reinterpret_cast<const BlockQ8_1 *>(activation.data),
        reinterpret_cast<float *>(output.data), kInput, kOutput,
        1, kOutput, 0, stream);
}

void launch_candidate(const GuardedBuffer & weights,
                      const GuardedBuffer & activation,
                      GuardedBuffer & output, cudaStream_t stream) {
    const Candidate::LaunchResult result = Candidate::launch(
        weights.data, reinterpret_cast<const BlockQ8_1 *>(activation.data),
        reinterpret_cast<float *>(output.data), kInput, kOutput,
        1, kOutput, 0, 86, stream);
    if (result != Candidate::LaunchResult::Launched) {
        std::fprintf(stderr,
            "q8-tm-down-longk-lab: candidate launch result=%u\n",
            unsigned(result));
        std::exit(3);
    }
}

struct Comparison {
    bool bitwise = false;
    uint64_t different = 0;
    double max_abs = 0.0;
    uint64_t non_finite = 0;
};

Comparison compare(const std::vector<float> & candidate,
                   const std::vector<float> & control) {
    Comparison result;
    if (candidate.size() != control.size()) return result;
    for (std::size_t index = 0; index < candidate.size(); ++index) {
        const float a = candidate[index];
        const float b = control[index];
        if (!std::isfinite(a) || !std::isfinite(b)) ++result.non_finite;
        uint32_t a_bits = 0;
        uint32_t b_bits = 0;
        std::memcpy(&a_bits, &a, sizeof(a_bits));
        std::memcpy(&b_bits, &b, sizeof(b_bits));
        result.different += a_bits != b_bits;
        result.max_abs = std::max(
            result.max_abs, std::fabs(double(a) - double(b)));
    }
    result.bitwise = result.different == 0;
    return result;
}

struct CaseResult {
    Comparison comparison{};
    uint64_t guard_errors = 0;
    uint64_t input_mismatches = 0;
    uint64_t nonzero_bits = 0;
};

CaseResult run_correctness_case(bool zero, cudaStream_t stream) {
    const std::vector<uint8_t> host_weights = make_tm_weights(47);
    const std::vector<BlockQ8_1> host_activation = make_activation(zero);
    GuardedBuffer weights(host_weights.size());
    GuardedBuffer activation(host_activation.size() * sizeof(BlockQ8_1));
    GuardedBuffer control_output(uint64_t(kOutput) * sizeof(float));
    GuardedBuffer candidate_output(uint64_t(kOutput) * sizeof(float));
    upload(weights, host_weights, "upload correctness weights");
    upload(activation, host_activation, "upload correctness activation");

    launch_control(weights, activation, control_output, stream);
    launch_candidate(weights, activation, candidate_output, stream);
    cuda_ok(cudaPeekAtLastError(), "correctness launch");
    cuda_ok(cudaStreamSynchronize(stream), "correctness synchronize");

    const std::vector<float> control = download<float>(
        control_output, kOutput, "download control output");
    const std::vector<float> candidate = download<float>(
        candidate_output, kOutput, "download candidate output");
    CaseResult result;
    result.comparison = compare(candidate, control);
    for (float value : candidate) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        result.nonzero_bits += bits != 0;
    }
    result.guard_errors = weights.guard_errors() + activation.guard_errors()
        + control_output.guard_errors() + candidate_output.guard_errors();
    result.input_mismatches = byte_mismatches(weights, host_weights.data())
        + byte_mismatches(activation, host_activation.data());
    return result;
}

double median(std::vector<float> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() & 1u ? values[middle]
        : 0.5 * (double(values[middle - 1]) + double(values[middle]));
}

double coefficient_of_variation(const std::vector<float> & values) {
    if (values.size() < 2) return 0.0;
    const double average = std::accumulate(
        values.begin(), values.end(), 0.0) / double(values.size());
    if (!(average > 0.0)) return 0.0;
    double sum_sq = 0.0;
    for (float value : values) {
        const double delta = double(value) - average;
        sum_sq += delta * delta;
    }
    return std::sqrt(sum_sq / double(values.size() - 1)) / average;
}

class EventTimer {
public:
    EventTimer() {
        cuda_ok(cudaEventCreate(&start_), "create start event");
        cuda_ok(cudaEventCreate(&stop_), "create stop event");
    }
    ~EventTimer() {
        if (start_) (void)cudaEventDestroy(start_);
        if (stop_) (void)cudaEventDestroy(stop_);
    }
    template <typename Launch>
    float measure(cudaStream_t stream, uint32_t repetitions,
                  Launch && launch) {
        cuda_ok(cudaEventRecord(start_, stream), "record start event");
        for (uint32_t repeat = 0; repeat < repetitions; ++repeat) launch();
        cuda_ok(cudaEventRecord(stop_, stream), "record stop event");
        cuda_ok(cudaEventSynchronize(stop_), "synchronize stop event");
        float milliseconds = 0.0f;
        cuda_ok(cudaEventElapsedTime(&milliseconds, start_, stop_),
                "measure elapsed event time");
        return milliseconds * 1000.0f / float(repetitions);
    }
private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

struct BenchmarkResult {
    std::vector<float> control_us;
    std::vector<float> candidate_us;
    Comparison post_comparison{};
    uint64_t guard_errors = 0;
    uint64_t input_mismatches = 0;
};

BenchmarkResult run_benchmark(uint32_t warmup, uint32_t pairs,
                              uint32_t repetitions, cudaStream_t stream) {
    const std::vector<uint8_t> host_weights = make_tm_weights(113);
    const std::vector<BlockQ8_1> host_activation = make_activation(false);
    GuardedBuffer weights(host_weights.size());
    GuardedBuffer activation(host_activation.size() * sizeof(BlockQ8_1));
    GuardedBuffer control_output(uint64_t(kOutput) * sizeof(float));
    GuardedBuffer candidate_output(uint64_t(kOutput) * sizeof(float));
    upload(weights, host_weights, "upload benchmark weights");
    upload(activation, host_activation, "upload benchmark activation");

    const auto control = [&] {
        launch_control(weights, activation, control_output, stream);
    };
    const auto candidate = [&] {
        launch_candidate(weights, activation, candidate_output, stream);
    };
    for (uint32_t index = 0; index < warmup; ++index) {
        control();
        candidate();
    }
    cuda_ok(cudaStreamSynchronize(stream), "benchmark warmup synchronize");

    EventTimer timer;
    BenchmarkResult result;
    result.control_us.reserve(uint64_t(pairs) * 2);
    result.candidate_us.reserve(uint64_t(pairs) * 2);
    const auto sample_control = [&] {
        result.control_us.push_back(timer.measure(stream, repetitions, control));
    };
    const auto sample_candidate = [&] {
        result.candidate_us.push_back(
            timer.measure(stream, repetitions, candidate));
    };
    for (uint32_t pair = 0; pair < pairs; ++pair) {
        if ((pair & 1u) == 0) {
            sample_control(); sample_candidate();
            sample_candidate(); sample_control();
        } else {
            sample_candidate(); sample_control();
            sample_control(); sample_candidate();
        }
    }

    cuda_ok(cudaStreamSynchronize(stream), "benchmark final synchronize");
    result.post_comparison = compare(
        download<float>(candidate_output, kOutput,
                        "download benchmark candidate"),
        download<float>(control_output, kOutput,
                        "download benchmark control"));
    result.guard_errors = weights.guard_errors() + activation.guard_errors()
        + control_output.guard_errors() + candidate_output.guard_errors();
    result.input_mismatches = byte_mismatches(weights, host_weights.data())
        + byte_mismatches(activation, host_activation.data());
    return result;
}

uint32_t parse_u32(const char * text, const char * name) {
    char * end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr, "q8-tm-down-longk-lab: invalid %s: %s\n",
                     name, text);
        std::exit(2);
    }
    return uint32_t(value);
}

} // namespace

int main(int argc, char ** argv) {
    bool benchmark = false;
    uint32_t warmup = 8;
    uint32_t pairs = 16;
    uint32_t repetitions = 4;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--smoke") benchmark = false;
        else if (argument == "--benchmark") benchmark = true;
        else if (argument == "--warmup" && index + 1 < argc)
            warmup = parse_u32(argv[++index], "warmup");
        else if (argument == "--pairs" && index + 1 < argc)
            pairs = parse_u32(argv[++index], "pairs");
        else if (argument == "--launches-per-sample" && index + 1 < argc)
            repetitions = parse_u32(argv[++index], "launches-per-sample");
        else {
            std::fprintf(stderr,
                "usage: sm86_q8_tm_down_longk_lab [--smoke|--benchmark] "
                "[--warmup N] [--pairs N] [--launches-per-sample N]\n");
            return 2;
        }
    }

    int device = 0;
    cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_ok(cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");
    if (properties.major != 8 || properties.minor != 6) {
        std::fprintf(stderr,
            "q8-tm-down-longk-lab: exact SM86 required, found SM%d%d\n",
            properties.major, properties.minor);
        return 2;
    }
    cudaStream_t stream = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");

    const CaseResult ordinary = run_correctness_case(false, stream);
    const CaseResult zero = run_correctness_case(true, stream);
    const bool correctness_ok = ordinary.comparison.bitwise
        && ordinary.comparison.non_finite == 0
        && ordinary.guard_errors == 0 && ordinary.input_mismatches == 0
        && zero.comparison.bitwise && zero.comparison.non_finite == 0
        && zero.nonzero_bits == 0
        && zero.guard_errors == 0 && zero.input_mismatches == 0;
    std::printf(
        "{\"mode\":\"correctness\",\"sm\":86,\"k\":%u,\"n\":%u,"
        "\"tokens\":1,\"ordinary_bitwise\":%s,"
        "\"ordinary_different\":%llu,\"ordinary_max_abs\":%.9g,"
        "\"ordinary_non_finite\":%llu,\"ordinary_guard_errors\":%llu,"
        "\"ordinary_input_mismatches\":%llu,\"zero_bitwise\":%s,"
        "\"zero_nonzero_bits\":%llu,\"zero_guard_errors\":%llu,"
        "\"zero_input_mismatches\":%llu,\"pass\":%s}\n",
        kInput, kOutput,
        ordinary.comparison.bitwise ? "true" : "false",
        static_cast<unsigned long long>(ordinary.comparison.different),
        ordinary.comparison.max_abs,
        static_cast<unsigned long long>(ordinary.comparison.non_finite),
        static_cast<unsigned long long>(ordinary.guard_errors),
        static_cast<unsigned long long>(ordinary.input_mismatches),
        zero.comparison.bitwise ? "true" : "false",
        static_cast<unsigned long long>(zero.nonzero_bits),
        static_cast<unsigned long long>(zero.guard_errors),
        static_cast<unsigned long long>(zero.input_mismatches),
        correctness_ok ? "true" : "false");

    bool benchmark_ok = true;
    if (benchmark && correctness_ok) {
        const BenchmarkResult result = run_benchmark(
            warmup, pairs, repetitions, stream);
        const double control = median(result.control_us);
        const double candidate = median(result.candidate_us);
        const double speedup = candidate > 0.0 ? control / candidate : 0.0;
        benchmark_ok = result.post_comparison.bitwise
            && result.post_comparison.non_finite == 0
            && result.guard_errors == 0 && result.input_mismatches == 0
            && control > 0.0 && candidate > 0.0;
        std::printf(
            "{\"mode\":\"benchmark\",\"sm\":86,\"k\":%u,\"n\":%u,"
            "\"tokens\":1,\"warmup\":%u,\"pairs\":%u,"
            "\"launches_per_sample\":%u,\"samples_per_route\":%llu,"
            "\"control_median_us\":%.9g,\"candidate_median_us\":%.9g,"
            "\"control_cv\":%.9g,\"candidate_cv\":%.9g,"
            "\"speedup\":%.9g,\"post_bitwise\":%s,"
            "\"post_different\":%llu,\"post_max_abs\":%.9g,"
            "\"guard_errors\":%llu,\"input_mismatches\":%llu,"
            "\"pass\":%s}\n",
            kInput, kOutput, warmup, pairs, repetitions,
            static_cast<unsigned long long>(result.control_us.size()),
            control, candidate,
            coefficient_of_variation(result.control_us),
            coefficient_of_variation(result.candidate_us), speedup,
            result.post_comparison.bitwise ? "true" : "false",
            static_cast<unsigned long long>(result.post_comparison.different),
            result.post_comparison.max_abs,
            static_cast<unsigned long long>(result.guard_errors),
            static_cast<unsigned long long>(result.input_mismatches),
            benchmark_ok ? "true" : "false");
    }

    cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return correctness_ok && benchmark_ok ? 0 : 1;
}
