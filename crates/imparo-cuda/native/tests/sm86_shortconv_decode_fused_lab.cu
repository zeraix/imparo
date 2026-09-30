// Isolated SM86 lab for fusing the two single-token LFM2 short-convolution
// kernels. This translation unit is never linked into the CUDA backend.

#if !defined(IMPARO_CUDA_KERNEL_LAB) || IMPARO_CUDA_KERNEL_LAB != 1
#error "sm86_shortconv_decode_fused_lab.cu is test-only"
#endif

#include "../lfm2_ops.cuh"
#include "../sm86/shortconv_decode_fused.cuh"

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

namespace Candidate = imparo_sm86_shortconv_decode_fused;

constexpr uint32_t kWidth = 2048;
constexpr uint32_t kKernel = 3;
constexpr uint32_t kAlternateKernel = 5;
constexpr std::size_t kGuardBytes = 256;
constexpr uint8_t kGuardValue = 0xa5;

[[noreturn]] void fail(const char * operation, cudaError_t error) {
    std::fprintf(stderr, "shortconv-fused-lab: %s: %s\n", operation,
                 cudaGetErrorString(error));
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
                "copy leading guard");
        cuda_ok(cudaMemcpy(after.data(), data + bytes, kGuardBytes,
                           cudaMemcpyDeviceToHost),
                "copy trailing guard");
        uint64_t errors = 0;
        for (uint8_t value : before) errors += value != kGuardValue;
        for (uint8_t value : after) errors += value != kGuardValue;
        return errors;
    }
};

template <typename T>
T * device(GuardedBuffer & buffer) {
    return reinterpret_cast<T *>(buffer.data);
}

template <typename T>
const T * device(const GuardedBuffer & buffer) {
    return reinterpret_cast<const T *>(buffer.data);
}

template <typename T>
void upload(GuardedBuffer & destination, const std::vector<T> & source,
            const char * operation) {
    cuda_ok(cudaMemcpy(destination.data, source.data(),
                       source.size() * sizeof(T), cudaMemcpyHostToDevice),
            operation);
}

template <typename T>
std::vector<T> download(const GuardedBuffer & source, std::size_t count,
                        const char * operation) {
    std::vector<T> result(count);
    cuda_ok(cudaMemcpy(result.data(), source.data, count * sizeof(T),
                       cudaMemcpyDeviceToHost), operation);
    return result;
}

uint64_t byte_mismatches(const GuardedBuffer & device_buffer,
                         const void * expected) {
    std::vector<uint8_t> actual(device_buffer.bytes);
    cuda_ok(cudaMemcpy(actual.data(), device_buffer.data, actual.size(),
                       cudaMemcpyDeviceToHost), "download immutable input");
    const auto * reference = static_cast<const uint8_t *>(expected);
    uint64_t count = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        count += actual[i] != reference[i];
    }
    return count;
}

struct Comparison {
    uint64_t different = 0;
    uint64_t non_finite = 0;
    float max_abs = 0.0f;

    bool bitwise() const { return different == 0; }
};

Comparison compare(const std::vector<float> & candidate,
                   const std::vector<float> & control) {
    Comparison result;
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        uint32_t a = 0;
        uint32_t b = 0;
        std::memcpy(&a, &candidate[i], sizeof(a));
        std::memcpy(&b, &control[i], sizeof(b));
        result.different += a != b;
        result.non_finite += !std::isfinite(candidate[i]);
        result.max_abs = std::max(
            result.max_abs, std::fabs(candidate[i] - control[i]));
    }
    return result;
}

uint32_t next_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_float(uint32_t & state, float scale) {
    const int32_t centered = int32_t(next_random(state) & 0xffffu) - 32768;
    return float(centered) * (scale / 32768.0f);
}

void launch_control(const GuardedBuffer & bcx, const GuardedBuffer & weights,
                    GuardedBuffer & state, GuardedBuffer & output,
                    cudaStream_t stream, uint32_t kernel = kKernel) {
    imparo_cuda_lfm2::shortconv_output_kernel<<<
        (kWidth + 255) / 256, 256, 0, stream>>>(
            device<float>(bcx), device<float>(weights), device<float>(state),
            device<float>(output), kWidth, kernel, 1);
    imparo_cuda_lfm2::shortconv_state_kernel<<<
        (kWidth + 255) / 256, 256, 0, stream>>>(
            device<float>(bcx), device<float>(state), device<float>(state),
            kWidth, kernel, 1);
}

void launch_candidate(const GuardedBuffer & bcx,
                      const GuardedBuffer & weights, GuardedBuffer & state,
                      GuardedBuffer & output, cudaStream_t stream,
                      uint32_t kernel = kKernel) {
    const Candidate::LaunchResult result = Candidate::launch(
        device<float>(bcx), device<float>(weights), device<float>(state),
        device<float>(output), kWidth, kernel, 86, stream);
    if (result != Candidate::LaunchResult::Launched) {
        std::fprintf(stderr, "shortconv-fused-lab: candidate launch rejected\n");
        std::exit(3);
    }
}

struct CorrectnessResult {
    Comparison output;
    Comparison state;
    uint64_t guard_errors = 0;
    uint64_t input_mismatches = 0;
};

CorrectnessResult correctness(cudaStream_t stream, uint32_t kernel = kKernel) {
    const std::size_t bcx_count = 3 * std::size_t(kWidth);
    const std::size_t weight_count = std::size_t(kernel) * kWidth;
    const std::size_t state_count = std::size_t(kernel - 1) * kWidth;
    std::vector<float> host_bcx(bcx_count);
    std::vector<float> host_weights(weight_count);
    std::vector<float> host_state(state_count);
    uint32_t rng = 0x93457abdu;
    for (float & value : host_bcx) value = random_float(rng, 1.25f);
    for (float & value : host_weights) value = random_float(rng, 0.2f);
    for (float & value : host_state) value = random_float(rng, 0.75f);

    GuardedBuffer bcx(bcx_count * sizeof(float));
    GuardedBuffer weights(weight_count * sizeof(float));
    GuardedBuffer control_state(state_count * sizeof(float));
    GuardedBuffer candidate_state(state_count * sizeof(float));
    GuardedBuffer control_output(std::size_t(kWidth) * sizeof(float));
    GuardedBuffer candidate_output(std::size_t(kWidth) * sizeof(float));
    upload(bcx, host_bcx, "upload bcx");
    upload(weights, host_weights, "upload weights");
    upload(control_state, host_state, "upload control state");
    upload(candidate_state, host_state, "upload candidate state");

    launch_control(bcx, weights, control_state, control_output, stream, kernel);
    launch_candidate(
        bcx, weights, candidate_state, candidate_output, stream, kernel);
    cuda_ok(cudaStreamSynchronize(stream), "synchronize correctness");

    CorrectnessResult result;
    result.output = compare(
        download<float>(candidate_output, kWidth, "download candidate output"),
        download<float>(control_output, kWidth, "download control output"));
    result.state = compare(
        download<float>(candidate_state, state_count,
                        "download candidate state"),
        download<float>(control_state, state_count, "download control state"));
    result.guard_errors = bcx.guard_errors() + weights.guard_errors()
        + control_state.guard_errors() + candidate_state.guard_errors()
        + control_output.guard_errors() + candidate_output.guard_errors();
    result.input_mismatches = byte_mismatches(bcx, host_bcx.data())
        + byte_mismatches(weights, host_weights.data());
    return result;
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
        : 0.5 * (double(values[middle - 1]) + values[middle]);
}

double coefficient_of_variation(const std::vector<float> & values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0)
        / double(values.size());
    double sum = 0.0;
    for (float value : values) {
        const double delta = double(value) - mean;
        sum += delta * delta;
    }
    return mean > 0.0 ? std::sqrt(sum / double(values.size())) / mean : 0.0;
}

template <typename Launch>
float timed(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream,
            uint32_t repetitions, const Launch & launch) {
    cuda_ok(cudaEventRecord(start, stream), "record timing start");
    for (uint32_t i = 0; i < repetitions; ++i) launch();
    cuda_ok(cudaEventRecord(stop, stream), "record timing stop");
    cuda_ok(cudaEventSynchronize(stop), "synchronize timing stop");
    float milliseconds = 0.0f;
    cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop),
            "elapsed timing");
    return milliseconds * 1000.0f / float(repetitions);
}

struct BenchmarkResult {
    std::vector<float> control_us;
    std::vector<float> candidate_us;
    Comparison output;
    Comparison state;
    uint64_t guard_errors = 0;
    uint64_t input_mismatches = 0;
};

BenchmarkResult benchmark(cudaStream_t stream, uint32_t warmup,
                          uint32_t pairs, uint32_t repetitions) {
    const std::size_t bcx_count = 3 * std::size_t(kWidth);
    const std::size_t weight_count = std::size_t(kKernel) * kWidth;
    const std::size_t state_count = std::size_t(kKernel - 1) * kWidth;
    std::vector<float> host_bcx(bcx_count);
    std::vector<float> host_weights(weight_count);
    std::vector<float> host_state(state_count);
    uint32_t rng = 0x62b4e91fu;
    for (float & value : host_bcx) value = random_float(rng, 0.8f);
    for (float & value : host_weights) value = random_float(rng, 0.125f);
    for (float & value : host_state) value = random_float(rng, 0.5f);

    GuardedBuffer bcx(bcx_count * sizeof(float));
    GuardedBuffer weights(weight_count * sizeof(float));
    GuardedBuffer control_state(state_count * sizeof(float));
    GuardedBuffer candidate_state(state_count * sizeof(float));
    GuardedBuffer control_output(std::size_t(kWidth) * sizeof(float));
    GuardedBuffer candidate_output(std::size_t(kWidth) * sizeof(float));
    upload(bcx, host_bcx, "upload benchmark bcx");
    upload(weights, host_weights, "upload benchmark weights");
    upload(control_state, host_state, "upload benchmark control state");
    upload(candidate_state, host_state, "upload benchmark candidate state");

    const auto control = [&] {
        launch_control(bcx, weights, control_state, control_output, stream);
    };
    const auto candidate = [&] {
        launch_candidate(bcx, weights, candidate_state, candidate_output, stream);
    };
    for (uint32_t i = 0; i < warmup; ++i) {
        control();
        candidate();
    }
    cuda_ok(cudaStreamSynchronize(stream), "synchronize warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "create timing start");
    cuda_ok(cudaEventCreate(&stop), "create timing stop");
    BenchmarkResult result;
    result.control_us.reserve(2 * pairs);
    result.candidate_us.reserve(2 * pairs);
    for (uint32_t pair = 0; pair < pairs; ++pair) {
        if ((pair & 1u) == 0) {
            result.control_us.push_back(
                timed(start, stop, stream, repetitions, control));
            result.candidate_us.push_back(
                timed(start, stop, stream, repetitions, candidate));
            result.candidate_us.push_back(
                timed(start, stop, stream, repetitions, candidate));
            result.control_us.push_back(
                timed(start, stop, stream, repetitions, control));
        } else {
            result.candidate_us.push_back(
                timed(start, stop, stream, repetitions, candidate));
            result.control_us.push_back(
                timed(start, stop, stream, repetitions, control));
            result.control_us.push_back(
                timed(start, stop, stream, repetitions, control));
            result.candidate_us.push_back(
                timed(start, stop, stream, repetitions, candidate));
        }
    }
    cuda_ok(cudaEventDestroy(stop), "destroy timing stop");
    cuda_ok(cudaEventDestroy(start), "destroy timing start");
    cuda_ok(cudaStreamSynchronize(stream), "synchronize benchmark");

    result.output = compare(
        download<float>(candidate_output, kWidth,
                        "download benchmark candidate output"),
        download<float>(control_output, kWidth,
                        "download benchmark control output"));
    result.state = compare(
        download<float>(candidate_state, state_count,
                        "download benchmark candidate state"),
        download<float>(control_state, state_count,
                        "download benchmark control state"));
    result.guard_errors = bcx.guard_errors() + weights.guard_errors()
        + control_state.guard_errors() + candidate_state.guard_errors()
        + control_output.guard_errors() + candidate_output.guard_errors();
    result.input_mismatches = byte_mismatches(bcx, host_bcx.data())
        + byte_mismatches(weights, host_weights.data());
    return result;
}

uint32_t parse_u32(const char * text, const char * name) {
    char * end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr, "shortconv-fused-lab: invalid %s: %s\n", name,
                     text);
        std::exit(2);
    }
    return uint32_t(value);
}

} // namespace

int main(int argc, char ** argv) {
    bool run_benchmark = false;
    uint32_t warmup = 32;
    uint32_t pairs = 16;
    uint32_t repetitions = 256;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--smoke") {
            run_benchmark = false;
        } else if (argument == "--benchmark") {
            run_benchmark = true;
        } else if (argument == "--warmup" && index + 1 < argc) {
            warmup = parse_u32(argv[++index], "warmup");
        } else if (argument == "--pairs" && index + 1 < argc) {
            pairs = parse_u32(argv[++index], "pairs");
        } else if (argument == "--launches-per-sample" && index + 1 < argc) {
            repetitions = parse_u32(argv[++index], "launches-per-sample");
        } else {
            std::fprintf(stderr,
                "usage: sm86_shortconv_decode_fused_lab "
                "[--smoke|--benchmark] [--warmup N] [--pairs N] "
                "[--launches-per-sample N]\n");
            return 2;
        }
    }

    int device_id = 0;
    cuda_ok(cudaGetDevice(&device_id), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_ok(cudaGetDeviceProperties(&properties, device_id),
            "cudaGetDeviceProperties");
    if (properties.major != 8 || properties.minor != 6) {
        std::fprintf(stderr,
            "shortconv-fused-lab: exact SM86 required, found SM%d%d\n",
            properties.major, properties.minor);
        return 2;
    }
    cudaStream_t stream = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "create nonblocking stream");

    const CorrectnessResult check = correctness(stream);
    const CorrectnessResult alternate_check =
        correctness(stream, kAlternateKernel);
    const bool correctness_ok = check.output.bitwise()
        && check.state.bitwise() && check.output.non_finite == 0
        && check.state.non_finite == 0 && check.guard_errors == 0
        && check.input_mismatches == 0
        && alternate_check.output.bitwise()
        && alternate_check.state.bitwise()
        && alternate_check.output.non_finite == 0
        && alternate_check.state.non_finite == 0
        && alternate_check.guard_errors == 0
        && alternate_check.input_mismatches == 0;
    std::printf(
        "{\"mode\":\"correctness\",\"sm\":86,\"width\":%u,\"kernel\":%u,"
        "\"output_bitwise\":%s,\"output_different\":%llu,"
        "\"output_max_abs\":%.9g,\"state_bitwise\":%s,"
        "\"state_different\":%llu,\"state_max_abs\":%.9g,"
        "\"guard_errors\":%llu,\"input_mismatches\":%llu,\"pass\":%s}\n",
        kWidth, kKernel, check.output.bitwise() ? "true" : "false",
        static_cast<unsigned long long>(check.output.different),
        check.output.max_abs, check.state.bitwise() ? "true" : "false",
        static_cast<unsigned long long>(check.state.different),
        check.state.max_abs,
        static_cast<unsigned long long>(check.guard_errors),
        static_cast<unsigned long long>(check.input_mismatches),
        correctness_ok ? "true" : "false");
    std::printf(
        "{\"mode\":\"correctness\",\"sm\":86,\"width\":%u,\"kernel\":%u,"
        "\"output_bitwise\":%s,\"output_different\":%llu,"
        "\"output_max_abs\":%.9g,\"state_bitwise\":%s,"
        "\"state_different\":%llu,\"state_max_abs\":%.9g,"
        "\"guard_errors\":%llu,\"input_mismatches\":%llu,\"pass\":%s}\n",
        kWidth, kAlternateKernel,
        alternate_check.output.bitwise() ? "true" : "false",
        static_cast<unsigned long long>(alternate_check.output.different),
        alternate_check.output.max_abs,
        alternate_check.state.bitwise() ? "true" : "false",
        static_cast<unsigned long long>(alternate_check.state.different),
        alternate_check.state.max_abs,
        static_cast<unsigned long long>(alternate_check.guard_errors),
        static_cast<unsigned long long>(alternate_check.input_mismatches),
        correctness_ok ? "true" : "false");

    bool benchmark_ok = true;
    if (run_benchmark && correctness_ok) {
        const BenchmarkResult result = benchmark(
            stream, warmup, pairs, repetitions);
        const double control_us = median(result.control_us);
        const double candidate_us = median(result.candidate_us);
        const double speedup = control_us / candidate_us;
        benchmark_ok = result.output.bitwise() && result.state.bitwise()
            && result.output.non_finite == 0 && result.state.non_finite == 0
            && result.guard_errors == 0 && result.input_mismatches == 0;
        std::printf(
            "{\"mode\":\"benchmark\",\"samples_per_route\":%llu,"
            "\"launches_per_sample\":%u,\"control_us\":%.6f,"
            "\"candidate_us\":%.6f,\"speedup\":%.6f,"
            "\"control_cv\":%.6f,\"candidate_cv\":%.6f,"
            "\"post_output_bitwise\":%s,\"post_state_bitwise\":%s,"
            "\"guard_errors\":%llu,\"input_mismatches\":%llu,\"pass\":%s}\n",
            static_cast<unsigned long long>(result.control_us.size()),
            repetitions, control_us, candidate_us, speedup,
            coefficient_of_variation(result.control_us),
            coefficient_of_variation(result.candidate_us),
            result.output.bitwise() ? "true" : "false",
            result.state.bitwise() ? "true" : "false",
            static_cast<unsigned long long>(result.guard_errors),
            static_cast<unsigned long long>(result.input_mismatches),
            benchmark_ok ? "true" : "false");
    }

    cuda_ok(cudaStreamDestroy(stream), "destroy stream");
    return correctness_ok && benchmark_ok ? 0 : 1;
}
