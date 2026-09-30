// Standalone SM86 lab for a single-token Q8_0 TileMajor FFN transaction:
// fused Gate/Up/SiLU -> private Q8_1 sidecar -> established Down MMVQ.
// This file is never linked into the backend library.

#if !defined(IMPARO_CUDA_KERNEL_LAB) || IMPARO_CUDA_KERNEL_LAB != 1
#error "sm86_q8_tm_private_down_lab.cu is test-only"
#endif

#include "../imparo_cuda.cu"
#include "../sm86/mmvq_q8_tm_gate_up_q8_sidecar.cuh"

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

namespace Dense = imparo_sm86_q8_tm_gate_up_silu;
namespace Sidecar = imparo_sm86_q8_tm_gate_up_q8_sidecar_lab;

constexpr std::size_t kGuardBytes = 256;
constexpr uint8_t kGuardValue = 0xa5;
constexpr uint32_t kCorrectnessInput = 1056;
constexpr uint32_t kCorrectnessMid = 64;
constexpr uint32_t kCorrectnessOutput = 64;
constexpr uint32_t kBenchmarkInput = 2048;
constexpr uint32_t kBenchmarkMid = 10752;
constexpr uint32_t kBenchmarkOutput = 2048;

[[noreturn]] void fail(const char * operation, cudaError_t error) {
    std::fprintf(stderr, "q8-tm-private-down-lab: %s: %s\n",
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
std::vector<T> download(const GuardedBuffer & buffer, std::size_t count,
                        const char * operation) {
    std::vector<T> result(count);
    cuda_ok(cudaMemcpy(result.data(), buffer.data, count * sizeof(T),
                       cudaMemcpyDeviceToHost),
            operation);
    return result;
}

template <typename T>
void upload(GuardedBuffer & buffer, const std::vector<T> & values,
            const char * operation) {
    cuda_ok(cudaMemcpy(buffer.data, values.data(), values.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            operation);
}

uint64_t byte_mismatches(const GuardedBuffer & buffer,
                         const void * expected) {
    const std::vector<uint8_t> actual =
        download<uint8_t>(buffer, buffer.bytes, "download input bytes");
    const auto * reference = static_cast<const uint8_t *>(expected);
    uint64_t mismatches = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        mismatches += actual[index] != reference[index];
    }
    return mismatches;
}

std::vector<uint8_t> make_tm_weights(
        uint32_t n_in, uint32_t n_out, uint32_t seed) {
    const uint32_t blocks = n_in / 32;
    std::vector<uint8_t> result(
        uint64_t(n_out) * n_in + uint64_t(n_out) * blocks * sizeof(__half));
    uint8_t * payload = result.data();
    auto * scales = reinterpret_cast<__half *>(
        result.data() + uint64_t(n_out) * n_in);
    for (uint32_t tile = 0; tile < n_out / 8; ++tile) {
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint64_t unit = uint64_t(tile) * blocks + block;
            for (uint32_t row = 0; row < 8; ++row) {
                for (uint32_t item = 0; item < 32; ++item) {
                    const uint32_t mix = seed + tile * 131u + block * 37u
                        + row * 17u + item * 11u;
                    payload[unit * 256 + row * 32 + item] =
                        uint8_t(int8_t(int32_t(mix % 255u) - 127));
                }
                const float scale = 0.001953125f
                    * float(1u + ((seed + tile * 5u + block * 3u + row) % 31u));
                scales[unit * 8 + row] = __float2half_rn(scale);
            }
        }
    }
    return result;
}

std::vector<BlockQ8_1> make_activation(uint32_t n_in, bool zero) {
    const uint32_t blocks = n_in / 32;
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
    return result;
}

void launch_dense_producer(
        const GuardedBuffer & gate, const GuardedBuffer & up,
        const GuardedBuffer & activation, GuardedBuffer & dense,
        uint32_t n_in, uint32_t n_mid, cudaStream_t stream) {
    const Dense::LaunchResult result = Dense::launch(
        gate.data, up.data,
        reinterpret_cast<const BlockQ8_1 *>(activation.data),
        reinterpret_cast<float *>(dense.data),
        n_in, n_mid, 86, stream);
    if (result != Dense::LaunchResult::Launched) {
        std::fprintf(stderr,
            "q8-tm-private-down-lab: dense producer result=%u\n",
            unsigned(result));
        std::exit(3);
    }
}

void launch_control(
        const GuardedBuffer & gate, const GuardedBuffer & up,
        const GuardedBuffer & down, const GuardedBuffer & activation,
        GuardedBuffer & dense, GuardedBuffer & q8,
        GuardedBuffer & output, uint32_t n_in, uint32_t n_mid,
        uint32_t n_out, cudaStream_t stream) {
    launch_dense_producer(gate, up, activation, dense, n_in, n_mid, stream);
    k_quantize_q8_1<<<dim3(1, (n_mid + 511) / 512), 128, 0, stream>>>(
        reinterpret_cast<const float *>(dense.data),
        reinterpret_cast<BlockQ8_1 *>(q8.data), n_mid, 1, 0);
    imparo_sm80_q8_mmvq::launch_tile_major(
        down.data, reinterpret_cast<const BlockQ8_1 *>(q8.data),
        reinterpret_cast<float *>(output.data), n_mid, n_out,
        1, n_out, 0, stream);
}

void launch_candidate(
        const GuardedBuffer & gate, const GuardedBuffer & up,
        const GuardedBuffer & down, const GuardedBuffer & activation,
        GuardedBuffer & q8, GuardedBuffer & output,
        uint32_t n_in, uint32_t n_mid, uint32_t n_out,
        cudaStream_t stream) {
    const Sidecar::LaunchResult result = Sidecar::launch(
        gate.data, up.data,
        reinterpret_cast<const BlockQ8_1 *>(activation.data),
        reinterpret_cast<BlockQ8_1 *>(q8.data),
        n_in, n_mid, 86, stream);
    if (result != Sidecar::LaunchResult::Launched) {
        std::fprintf(stderr,
            "q8-tm-private-down-lab: sidecar producer result=%u\n",
            unsigned(result));
        std::exit(3);
    }
    imparo_sm80_q8_mmvq::launch_tile_major(
        down.data, reinterpret_cast<const BlockQ8_1 *>(q8.data),
        reinterpret_cast<float *>(output.data), n_mid, n_out,
        1, n_out, 0, stream);
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
    Comparison output{};
    uint64_t sidecar_mismatches = 0;
    uint64_t guard_errors = 0;
    uint64_t input_mismatches = 0;
    uint64_t nonzero_output_bits = 0;
};

CaseResult run_correctness_case(
        uint32_t n_in, uint32_t n_mid, uint32_t n_out,
        bool zero, cudaStream_t stream) {
    const std::vector<uint8_t> host_gate = make_tm_weights(n_in, n_mid, 11);
    const std::vector<uint8_t> host_up = make_tm_weights(n_in, n_mid, 97);
    const std::vector<uint8_t> host_down = make_tm_weights(n_mid, n_out, 173);
    const std::vector<BlockQ8_1> host_activation = make_activation(n_in, zero);
    const uint64_t q8_bytes = uint64_t(n_mid / 32) * sizeof(BlockQ8_1);

    GuardedBuffer gate(host_gate.size());
    GuardedBuffer up(host_up.size());
    GuardedBuffer down(host_down.size());
    GuardedBuffer activation(host_activation.size() * sizeof(BlockQ8_1));
    GuardedBuffer dense(uint64_t(n_mid) * sizeof(float));
    GuardedBuffer control_q8(q8_bytes);
    GuardedBuffer candidate_q8(q8_bytes);
    GuardedBuffer control_output(uint64_t(n_out) * sizeof(float));
    GuardedBuffer candidate_output(uint64_t(n_out) * sizeof(float));
    upload(gate, host_gate, "upload gate weights");
    upload(up, host_up, "upload up weights");
    upload(down, host_down, "upload down weights");
    upload(activation, host_activation, "upload activation");

    launch_control(gate, up, down, activation, dense, control_q8,
                   control_output, n_in, n_mid, n_out, stream);
    launch_candidate(gate, up, down, activation, candidate_q8,
                     candidate_output, n_in, n_mid, n_out, stream);
    cuda_ok(cudaPeekAtLastError(), "correctness launch");
    cuda_ok(cudaStreamSynchronize(stream), "correctness synchronize");

    const std::vector<uint8_t> control_sidecar = download<uint8_t>(
        control_q8, q8_bytes, "download control sidecar");
    const std::vector<uint8_t> candidate_sidecar = download<uint8_t>(
        candidate_q8, q8_bytes, "download candidate sidecar");
    const std::vector<float> control_values = download<float>(
        control_output, n_out, "download control output");
    const std::vector<float> candidate_values = download<float>(
        candidate_output, n_out, "download candidate output");

    CaseResult result;
    for (std::size_t index = 0; index < control_sidecar.size(); ++index) {
        result.sidecar_mismatches +=
            control_sidecar[index] != candidate_sidecar[index];
    }
    result.output = compare(candidate_values, control_values);
    for (float value : candidate_values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        result.nonzero_output_bits += bits != 0;
    }
    result.guard_errors = gate.guard_errors() + up.guard_errors()
        + down.guard_errors() + activation.guard_errors()
        + dense.guard_errors() + control_q8.guard_errors()
        + candidate_q8.guard_errors() + control_output.guard_errors()
        + candidate_output.guard_errors();
    result.input_mismatches = byte_mismatches(gate, host_gate.data())
        + byte_mismatches(up, host_up.data())
        + byte_mismatches(down, host_down.data())
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
    Comparison output{};
    uint64_t sidecar_mismatches = 0;
    uint64_t guard_errors = 0;
    uint64_t input_mismatches = 0;
};

BenchmarkResult run_benchmark(
        uint32_t warmup, uint32_t pairs, uint32_t repetitions,
        cudaStream_t stream) {
    const std::vector<uint8_t> host_gate =
        make_tm_weights(kBenchmarkInput, kBenchmarkMid, 23);
    const std::vector<uint8_t> host_up =
        make_tm_weights(kBenchmarkInput, kBenchmarkMid, 149);
    const std::vector<uint8_t> host_down =
        make_tm_weights(kBenchmarkMid, kBenchmarkOutput, 211);
    const std::vector<BlockQ8_1> host_activation =
        make_activation(kBenchmarkInput, false);
    const uint64_t q8_bytes =
        uint64_t(kBenchmarkMid / 32) * sizeof(BlockQ8_1);
    GuardedBuffer gate(host_gate.size());
    GuardedBuffer up(host_up.size());
    GuardedBuffer down(host_down.size());
    GuardedBuffer activation(host_activation.size() * sizeof(BlockQ8_1));
    GuardedBuffer dense(uint64_t(kBenchmarkMid) * sizeof(float));
    GuardedBuffer control_q8(q8_bytes);
    GuardedBuffer candidate_q8(q8_bytes);
    GuardedBuffer control_output(uint64_t(kBenchmarkOutput) * sizeof(float));
    GuardedBuffer candidate_output(uint64_t(kBenchmarkOutput) * sizeof(float));
    upload(gate, host_gate, "upload benchmark gate weights");
    upload(up, host_up, "upload benchmark up weights");
    upload(down, host_down, "upload benchmark down weights");
    upload(activation, host_activation, "upload benchmark activation");

    const auto control = [&] {
        launch_control(gate, up, down, activation, dense, control_q8,
            control_output, kBenchmarkInput, kBenchmarkMid,
            kBenchmarkOutput, stream);
    };
    const auto candidate = [&] {
        launch_candidate(gate, up, down, activation, candidate_q8,
            candidate_output, kBenchmarkInput, kBenchmarkMid,
            kBenchmarkOutput, stream);
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

    const std::vector<uint8_t> control_sidecar = download<uint8_t>(
        control_q8, q8_bytes, "download benchmark control sidecar");
    const std::vector<uint8_t> candidate_sidecar = download<uint8_t>(
        candidate_q8, q8_bytes, "download benchmark candidate sidecar");
    for (std::size_t index = 0; index < control_sidecar.size(); ++index) {
        result.sidecar_mismatches +=
            control_sidecar[index] != candidate_sidecar[index];
    }
    result.output = compare(
        download<float>(candidate_output, kBenchmarkOutput,
                        "download benchmark candidate output"),
        download<float>(control_output, kBenchmarkOutput,
                        "download benchmark control output"));
    result.guard_errors = gate.guard_errors() + up.guard_errors()
        + down.guard_errors() + activation.guard_errors()
        + dense.guard_errors() + control_q8.guard_errors()
        + candidate_q8.guard_errors() + control_output.guard_errors()
        + candidate_output.guard_errors();
    result.input_mismatches = byte_mismatches(gate, host_gate.data())
        + byte_mismatches(up, host_up.data())
        + byte_mismatches(down, host_down.data())
        + byte_mismatches(activation, host_activation.data());
    return result;
}

uint32_t parse_u32(const char * text, const char * name) {
    char * end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr,
            "q8-tm-private-down-lab: invalid %s: %s\n", name, text);
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
                "usage: sm86_q8_tm_private_down_lab "
                "[--smoke|--benchmark] [--warmup N] [--pairs N] "
                "[--launches-per-sample N]\n");
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
            "q8-tm-private-down-lab: exact SM86 required, found SM%d%d\n",
            properties.major, properties.minor);
        return 2;
    }
    cudaStream_t stream = nullptr;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");

    const CaseResult ordinary = run_correctness_case(
        kCorrectnessInput, kCorrectnessMid, kCorrectnessOutput,
        false, stream);
    const CaseResult zero = run_correctness_case(
        256, 32, 32, true, stream);
    const bool correctness_ok = ordinary.sidecar_mismatches == 0
        && ordinary.output.bitwise && ordinary.output.non_finite == 0
        && ordinary.guard_errors == 0 && ordinary.input_mismatches == 0
        && zero.sidecar_mismatches == 0 && zero.output.bitwise
        && zero.output.non_finite == 0 && zero.nonzero_output_bits == 0
        && zero.guard_errors == 0 && zero.input_mismatches == 0;
    std::printf(
        "{\"mode\":\"correctness\",\"sm\":86,"
        "\"input\":%u,\"mid\":%u,\"output\":%u,"
        "\"sidecar_mismatches\":%llu,\"output_bitwise\":%s,"
        "\"output_different\":%llu,\"output_max_abs\":%.9g,"
        "\"guard_errors\":%llu,\"input_mismatches\":%llu,"
        "\"zero_sidecar_mismatches\":%llu,"
        "\"zero_output_nonzero_bits\":%llu,\"pass\":%s}\n",
        kCorrectnessInput, kCorrectnessMid, kCorrectnessOutput,
        static_cast<unsigned long long>(ordinary.sidecar_mismatches),
        ordinary.output.bitwise ? "true" : "false",
        static_cast<unsigned long long>(ordinary.output.different),
        ordinary.output.max_abs,
        static_cast<unsigned long long>(ordinary.guard_errors),
        static_cast<unsigned long long>(ordinary.input_mismatches),
        static_cast<unsigned long long>(zero.sidecar_mismatches),
        static_cast<unsigned long long>(zero.nonzero_output_bits),
        correctness_ok ? "true" : "false");

    bool benchmark_ok = true;
    if (benchmark && correctness_ok) {
        const BenchmarkResult result = run_benchmark(
            warmup, pairs, repetitions, stream);
        const double control = median(result.control_us);
        const double candidate = median(result.candidate_us);
        const double speedup = candidate > 0.0 ? control / candidate : 0.0;
        benchmark_ok = result.sidecar_mismatches == 0
            && result.output.bitwise && result.output.non_finite == 0
            && result.guard_errors == 0 && result.input_mismatches == 0
            && control > 0.0 && candidate > 0.0;
        std::printf(
            "{\"mode\":\"benchmark\",\"sm\":86,"
            "\"input\":%u,\"mid\":%u,\"output\":%u,\"tokens\":1,"
            "\"warmup\":%u,\"pairs\":%u,"
            "\"launches_per_sample\":%u,\"samples_per_route\":%llu,"
            "\"control_median_us\":%.9g,\"candidate_median_us\":%.9g,"
            "\"control_cv\":%.9g,\"candidate_cv\":%.9g,"
            "\"speedup\":%.9g,\"sidecar_mismatches\":%llu,"
            "\"output_bitwise\":%s,\"output_different\":%llu,"
            "\"output_max_abs\":%.9g,\"guard_errors\":%llu,"
            "\"input_mismatches\":%llu,\"pass\":%s}\n",
            kBenchmarkInput, kBenchmarkMid, kBenchmarkOutput,
            warmup, pairs, repetitions,
            static_cast<unsigned long long>(result.control_us.size()),
            control, candidate,
            coefficient_of_variation(result.control_us),
            coefficient_of_variation(result.candidate_us), speedup,
            static_cast<unsigned long long>(result.sidecar_mismatches),
            result.output.bitwise ? "true" : "false",
            static_cast<unsigned long long>(result.output.different),
            result.output.max_abs,
            static_cast<unsigned long long>(result.guard_errors),
            static_cast<unsigned long long>(result.input_mismatches),
            benchmark_ok ? "true" : "false");
    }

    cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return correctness_ok && benchmark_ok ? 0 : 1;
}
