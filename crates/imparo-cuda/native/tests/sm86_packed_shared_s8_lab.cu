#include "../sm86/mmq_q4_q8_packed_shared_s8_lab.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace imparo_sm86_packed_shared_s8_lab;

constexpr int kInputPoolCount = 4;

struct Options {
    enum class Mode {
        GpuOracle,
        CpuOnly,
        Benchmark,
        Profile,
    } mode = Mode::GpuOracle;
    bool profile_candidate = false;
    int profile_tokens = 449;
    int cycles = 9;
    int warmup = 4;
};

int parse_positive(const char * name, const char * value) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || parsed <= 0 || parsed > 10000) {
        std::fprintf(stderr, "invalid %s: %s\n", name, value);
        std::exit(2);
    }
    return int(parsed);
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--cpu-only") == 0) {
            options.mode = Options::Mode::CpuOnly;
        } else if (std::strcmp(argv[index], "--gpu-oracle") == 0) {
            options.mode = Options::Mode::GpuOracle;
        } else if (std::strcmp(argv[index], "--benchmark") == 0) {
            options.mode = Options::Mode::Benchmark;
        } else if (std::strcmp(argv[index], "--profile") == 0) {
            if (++index >= argc) {
                std::fprintf(stderr, "missing route after --profile\n");
                std::exit(2);
            }
            options.mode = Options::Mode::Profile;
            if (std::strcmp(argv[index], "candidate") == 0) {
                options.profile_candidate = true;
            } else if (std::strcmp(argv[index], "control") == 0) {
                options.profile_candidate = false;
            } else {
                std::fprintf(stderr, "invalid profile route: %s\n", argv[index]);
                std::exit(2);
            }
        } else if (std::strcmp(argv[index], "--tokens") == 0
                || std::strcmp(argv[index], "--cycles") == 0
                || std::strcmp(argv[index], "--warmup") == 0) {
            if (++index >= argc) {
                std::fprintf(stderr, "missing numeric option value\n");
                std::exit(2);
            }
            const int parsed = parse_positive(argv[index - 1], argv[index]);
            if (std::strcmp(argv[index - 1], "--tokens") == 0) {
                if (parsed != 449 && parsed != 512) {
                    std::fprintf(stderr, "--tokens must be 449 or 512\n");
                    std::exit(2);
                }
                options.profile_tokens = parsed;
            } else if (std::strcmp(argv[index - 1], "--cycles") == 0) {
                options.cycles = parsed;
            } else {
                options.warmup = parsed;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
            std::exit(2);
        }
    }
    return options;
}

void check_cuda(cudaError_t status, const char * expression, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure line=%d expression=%s error=%s\n",
            line, expression, cudaGetErrorString(status));
        std::exit(3);
    }
}

#define CHECK_CUDA(expression) check_cuda((expression), #expression, __LINE__)

std::int8_t unpack_q4(std::uint32_t word, int nibble) {
    return static_cast<std::int8_t>(
        int((word >> (4 * nibble)) & 0x0fu) - 8);
}

int run_cpu_layout_exhaustive() {
    int mismatches = 0;
    for (int storage = 0; storage < 16; ++storage) {
        std::uint32_t word = 0;
        for (int nibble = 0; nibble < 8; ++nibble) {
            word |= std::uint32_t(storage) << (4 * nibble);
        }
        for (int nibble = 0; nibble < 8; ++nibble) {
            if (int(unpack_q4(word, nibble)) != storage - 8) {
                ++mismatches;
            }
        }
    }
    return mismatches;
}

struct Shape {
    int projections;
    int qblocks;
    int rows;
    int tokens;
};

struct HostData {
    explicit HostData(Shape shape_value) : shape(shape_value),
        packed_q4(std::size_t(shape.projections) * shape.qblocks * shape.rows
            * kWordsPerQblock),
        q4_scale(std::size_t(shape.projections) * shape.qblocks * shape.rows),
        q8(std::size_t(shape.qblocks) * shape.tokens * kKPerQblock),
        q8_scale(std::size_t(shape.qblocks) * shape.tokens),
        oracle(std::size_t(shape.projections) * shape.rows * shape.tokens) {
        fill();
    }

    void fill() {
        for (int projection = 0; projection < shape.projections; ++projection) {
            for (int qblock = 0; qblock < shape.qblocks; ++qblock) {
                for (int row = 0; row < shape.rows; ++row) {
                    for (int word = 0; word < kWordsPerQblock; ++word) {
                        std::uint32_t packed = 0;
                        for (int nibble = 0; nibble < 8; ++nibble) {
                            const int k = word * 8 + nibble;
                            const int storage = (projection * 11 + qblock * 7
                                + row * 5 + k * 13) & 15;
                            packed |= std::uint32_t(storage) << (4 * nibble);
                        }
                        packed_q4[weight_word_index(
                            projection, qblock, row, word)] = packed;
                    }
                    q4_scale[weight_scale_index(projection, qblock, row)] =
                        shape.qblocks <= 3
                        ? 1.0f
                        : std::ldexp(1.0f, ((projection + qblock + row) % 7) - 5);
                }
            }
        }
        for (int qblock = 0; qblock < shape.qblocks; ++qblock) {
            for (int token = 0; token < shape.tokens; ++token) {
                for (int k = 0; k < kKPerQblock; ++k) {
                    const std::uint32_t hash = std::uint32_t(
                        qblock * 131 + token * 37 + k * 17) * 1664525u
                        + 1013904223u;
                    q8[(std::size_t(qblock) * shape.tokens + token)
                        * kKPerQblock + k] = static_cast<std::int8_t>(
                            int((hash >> 16) & 255u) - 128);
                }
                q8_scale[std::size_t(qblock) * shape.tokens + token] =
                    shape.qblocks <= 3
                    ? 1.0f
                    : std::ldexp(1.0f, ((qblock + token) % 5) - 4);
            }
        }
    }

    std::size_t weight_word_index(
            int projection, int qblock, int row, int word) const {
        return ((std::size_t(projection) * shape.qblocks + qblock) * shape.rows
            + row) * kWordsPerQblock + word;
    }

    std::size_t weight_scale_index(
            int projection, int qblock, int row) const {
        return (std::size_t(projection) * shape.qblocks + qblock) * shape.rows
            + row;
    }

    void make_cpu_oracle() {
        std::fill(oracle.begin(), oracle.end(), 0.0f);
        for (int projection = 0; projection < shape.projections; ++projection) {
            for (int row = 0; row < shape.rows; ++row) {
                for (int token = 0; token < shape.tokens; ++token) {
                    float accumulator = 0.0f;
                    for (int qblock = 0; qblock < shape.qblocks; ++qblock) {
                        std::int32_t dot = 0;
                        for (int k = 0; k < kKPerQblock; ++k) {
                            const std::uint32_t word = packed_q4[weight_word_index(
                                projection, qblock, row, k / 8)];
                            const std::int8_t a = unpack_q4(word, k & 7);
                            const std::int8_t b = q8[
                                (std::size_t(qblock) * shape.tokens + token)
                                    * kKPerQblock + k];
                            dot += int(a) * int(b);
                        }
                        const float scale = q4_scale[weight_scale_index(
                            projection, qblock, row)]
                            * q8_scale[std::size_t(qblock) * shape.tokens + token];
                        accumulator = std::fma(float(dot), scale, accumulator);
                    }
                    oracle[(std::size_t(projection) * shape.rows + row)
                        * shape.tokens + token] = accumulator;
                }
            }
        }
    }

    Shape shape;
    std::vector<std::uint32_t> packed_q4;
    std::vector<float> q4_scale;
    std::vector<std::int8_t> q8;
    std::vector<float> q8_scale;
    std::vector<float> oracle;
};

template <typename T>
T * allocate_device(std::size_t count) {
    T * pointer = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&pointer),
        count * sizeof(T)));
    return pointer;
}

template <typename T>
void copy_to_device(T * destination, const std::vector<T> & source) {
    CHECK_CUDA(cudaMemcpy(destination, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

struct DeviceData {
    explicit DeviceData(const HostData & host)
        : control_output(allocate_device<float>(host.oracle.size())),
          candidate_output(allocate_device<float>(host.oracle.size())) {
        for (int pool = 0; pool < kInputPoolCount; ++pool) {
            packed_q4[pool] = allocate_device<std::uint32_t>(
                host.packed_q4.size());
            q4_scale[pool] = allocate_device<float>(host.q4_scale.size());
            q8[pool] = allocate_device<std::int8_t>(host.q8.size());
            q8_scale[pool] = allocate_device<float>(host.q8_scale.size());
            copy_to_device(packed_q4[pool], host.packed_q4);
            copy_to_device(q4_scale[pool], host.q4_scale);
            copy_to_device(q8[pool], host.q8);
            copy_to_device(q8_scale[pool], host.q8_scale);
        }
    }

    ~DeviceData() {
        cudaFree(candidate_output);
        cudaFree(control_output);
        for (int pool = kInputPoolCount - 1; pool >= 0; --pool) {
            cudaFree(q8_scale[pool]);
            cudaFree(q8[pool]);
            cudaFree(q4_scale[pool]);
            cudaFree(packed_q4[pool]);
        }
    }

    std::array<std::uint32_t *, kInputPoolCount> packed_q4{};
    std::array<float *, kInputPoolCount> q4_scale{};
    std::array<std::int8_t *, kInputPoolCount> q8{};
    std::array<float *, kInputPoolCount> q8_scale{};
    float * control_output;
    float * candidate_output;
};

std::size_t input_bytes_per_copy(const HostData & host) {
    return host.packed_q4.size() * sizeof(std::uint32_t)
        + host.q4_scale.size() * sizeof(float)
        + host.q8.size() * sizeof(std::int8_t)
        + host.q8_scale.size() * sizeof(float);
}

dim3 grid_for(const Shape & shape) {
    return dim3((shape.rows + kRowsPerCta - 1) / kRowsPerCta,
        (shape.tokens + kTokensPerCta - 1) / kTokensPerCta,
        shape.projections);
}

void launch_route(bool candidate, const HostData & host,
        const DeviceData & device, cudaStream_t stream, float * output,
        int pool_index = 0) {
    const dim3 grid = grid_for(host.shape);
    if (candidate) {
        packed_shared_candidate<<<grid, kThreads, 0, stream>>>(
            device.packed_q4[pool_index], device.q4_scale[pool_index],
            device.q8[pool_index], device.q8_scale[pool_index],
            output, host.shape.projections, host.shape.qblocks,
            host.shape.rows, host.shape.tokens);
    } else {
        expanded_shared_control<<<grid, kThreads, 0, stream>>>(
            device.packed_q4[pool_index], device.q4_scale[pool_index],
            device.q8[pool_index], device.q8_scale[pool_index],
            output, host.shape.projections, host.shape.qblocks,
            host.shape.rows, host.shape.tokens);
    }
    CHECK_CUDA(cudaGetLastError());
}

int compare_bitwise(const std::vector<float> & lhs,
        const std::vector<float> & rhs, const char * label) {
    int mismatches = 0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (std::memcmp(&lhs[index], &rhs[index], sizeof(float)) != 0) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                    "%s mismatch index=%zu lhs=%.9g rhs=%.9g\n",
                    label, index, lhs[index], rhs[index]);
            }
            ++mismatches;
        }
    }
    return mismatches;
}

int run_gpu_oracle(const cudaDeviceProp & properties) {
    int control_occupancy = 0;
    int candidate_occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &control_occupancy, expanded_shared_control, kThreads, 0));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &candidate_occupancy, packed_shared_candidate, kThreads, 0));
    const Shape shape{2, 3, 32, 65};
    HostData host(shape);
    host.make_cpu_oracle();
    DeviceData device(host);
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    launch_route(false, host, device, stream, device.control_output);
    launch_route(true, host, device, stream, device.candidate_output);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> control(host.oracle.size());
    std::vector<float> candidate(host.oracle.size());
    CHECK_CUDA(cudaMemcpy(control.data(), device.control_output,
        control.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(candidate.data(), device.candidate_output,
        candidate.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamDestroy(stream));
    const int control_cpu = compare_bitwise(control, host.oracle, "control/cpu");
    const int candidate_cpu = compare_bitwise(candidate, host.oracle,
        "candidate/cpu");
    const int route = compare_bitwise(control, candidate, "control/candidate");
    std::printf(
        "{\"oracle\":{\"device\":\"%s\",\"sm\":86,"
        "\"shape\":{\"projections\":2,\"qblocks\":3,\"rows\":32,"
        "\"tokens\":65},\"outputs\":%zu,\"control_cpu_mismatches\":%d,"
        "\"candidate_cpu_mismatches\":%d,\"route_mismatches\":%d,"
        "\"control_active_cta_per_sm\":%d,"
        "\"candidate_active_cta_per_sm\":%d}}\n",
        properties.name, host.oracle.size(), control_cpu, candidate_cpu, route,
        control_occupancy, candidate_occupancy);
    return control_cpu == 0 && candidate_cpu == 0 && route == 0
        && control_occupancy >= 2 && candidate_occupancy >= 2 ? 0 : 1;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const std::size_t size = values.size();
    return (size & 1u) != 0
        ? values[size / 2]
        : 0.5f * (values[size / 2 - 1] + values[size / 2]);
}

float mad(const std::vector<float> & values, float center) {
    std::vector<float> deviations;
    deviations.reserve(values.size());
    for (const float value : values) {
        deviations.push_back(std::fabs(value - center));
    }
    return median(std::move(deviations));
}

void print_samples(const std::vector<float> & values) {
    std::printf("[");
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::printf(index == 0 ? "%.6f" : ",%.6f", values[index]);
    }
    std::printf("]");
}

void print_samples(const std::vector<int> & values) {
    std::printf("[");
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::printf(index == 0 ? "%d" : ",%d", values[index]);
    }
    std::printf("]");
}

struct TimingResult {
    std::vector<float> raw;
    std::vector<int> raw_pool_index;
    std::vector<float> control;
    std::vector<float> candidate;
    float control_median = 0.0f;
    float control_mad = 0.0f;
    float candidate_median = 0.0f;
    float candidate_mad = 0.0f;
};

TimingResult time_shape(const HostData & host, const DeviceData & device,
        const Options & options, cudaStream_t stream) {
    for (int warmup = 0; warmup < options.warmup; ++warmup) {
        const int pool_index = warmup % kInputPoolCount;
        launch_route(false, host, device, stream, device.control_output,
            pool_index);
        launch_route(true, host, device, stream, device.candidate_output,
            pool_index);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    constexpr bool kCandidateSchedule[8] = {
        false, true, true, false, true, false, false, true};
    TimingResult result;
    result.raw.reserve(std::size_t(options.cycles) * 8);
    result.raw_pool_index.reserve(std::size_t(options.cycles) * 8);
    for (int cycle = 0; cycle < options.cycles; ++cycle) {
        for (int leg = 0; leg < 8; ++leg) {
            const bool candidate = kCandidateSchedule[leg];
            const int pool_index = (cycle + leg / 2) % kInputPoolCount;
            CHECK_CUDA(cudaEventRecord(start, stream));
            launch_route(candidate, host, device, stream,
                candidate ? device.candidate_output : device.control_output,
                pool_index);
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            const float microseconds = milliseconds * 1000.0f;
            result.raw.push_back(microseconds);
            result.raw_pool_index.push_back(pool_index);
            (candidate ? result.candidate : result.control).push_back(
                microseconds);
        }
    }
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));
    result.control_median = median(result.control);
    result.control_mad = mad(result.control, result.control_median);
    result.candidate_median = median(result.candidate);
    result.candidate_mad = mad(result.candidate, result.candidate_median);
    return result;
}

int validate_routes(const HostData & host, const DeviceData & device,
        cudaStream_t stream) {
    launch_route(false, host, device, stream, device.control_output);
    launch_route(true, host, device, stream, device.candidate_output);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> control(host.oracle.size());
    std::vector<float> candidate(host.oracle.size());
    CHECK_CUDA(cudaMemcpy(control.data(), device.control_output,
        control.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(candidate.data(), device.candidate_output,
        candidate.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return compare_bitwise(control, candidate, "benchmark routes");
}

void print_timing(int tokens, const HostData & host,
        const TimingResult & result, const Options & options,
        int control_occupancy, int candidate_occupancy) {
    const dim3 grid = grid_for(host.shape);
    std::printf(
        "{\"benchmark\":{\"tokens\":%d,\"clock\":\"cuda_event\","
        "\"same_stream\":true,\"schedule_unit\":\"ABBABAAB\","
        "\"cycles\":%d,\"warmup_per_route\":%d,"
        "\"input_pool_count\":%d,\"input_bytes_per_copy\":%zu,"
        "\"input_pool_total_bytes\":%zu,"
        "\"rotation_rule\":\"(cycle+floor(leg/2))%%4\","
        "\"warmup_rotation_rule\":\"warmup%%4_A_then_B\","
        "\"pair_route_order\":[\"AB\",\"BA\",\"BA\",\"AB\"],"
        "\"pair_pool_offsets\":[0,1,2,3],"
        "\"grid\":[%u,%u,%u],\"ctas\":%u,"
        "\"control_active_cta_per_sm\":%d,"
        "\"candidate_active_cta_per_sm\":%d,\"raw_us\":",
        tokens, options.cycles, options.warmup, kInputPoolCount,
        input_bytes_per_copy(host),
        input_bytes_per_copy(host) * kInputPoolCount,
        grid.x, grid.y, grid.z,
        grid.x * grid.y * grid.z, control_occupancy, candidate_occupancy);
    print_samples(result.raw);
    std::printf(",\"raw_pool_index\":");
    print_samples(result.raw_pool_index);
    std::printf(",\"control_us\":");
    print_samples(result.control);
    std::printf(",\"candidate_us\":");
    print_samples(result.candidate);
    std::printf(
        ",\"control_median_us\":%.6f,\"control_mad_us\":%.6f,"
        "\"candidate_median_us\":%.6f,\"candidate_mad_us\":%.6f,"
        "\"control_over_candidate\":%.6f}}\n",
        result.control_median, result.control_mad,
        result.candidate_median, result.candidate_mad,
        result.control_median / result.candidate_median);
}

int run_benchmark(const Options & options) {
    int control_occupancy = 0;
    int candidate_occupancy = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &control_occupancy, expanded_shared_control, kThreads, 0));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &candidate_occupancy, packed_shared_candidate, kThreads, 0));
    if (control_occupancy < 2 || candidate_occupancy < 2) {
        std::fprintf(stderr, "occupancy stop rule failed control=%d candidate=%d\n",
            control_occupancy, candidate_occupancy);
        return 1;
    }
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const int tokens : {449, 512}) {
        HostData host(Shape{2, 80, 256, tokens});
        DeviceData device(host);
        const int mismatches = validate_routes(host, device, stream);
        if (mismatches != 0) {
            CHECK_CUDA(cudaStreamDestroy(stream));
            return 1;
        }
        const TimingResult result = time_shape(host, device, options, stream);
        print_timing(tokens, host, result, options,
            control_occupancy, candidate_occupancy);
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}

int run_profile(bool candidate, int tokens) {
    HostData host(Shape{2, 80, 256, tokens});
    DeviceData device(host);
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    launch_route(candidate, host, device, stream,
        candidate ? device.candidate_output : device.control_output);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> output(host.oracle.size());
    CHECK_CUDA(cudaMemcpy(output.data(),
        candidate ? device.candidate_output : device.control_output,
        output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (const float value : output) {
        checksum += value;
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::printf(
        "{\"profile\":{\"route\":\"%s\",\"tokens\":%d,"
        "\"checksum\":%.17g}}\n",
        candidate ? "candidate" : "control", tokens, checksum);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const Options options = parse_options(argc, argv);
    const int cpu_mismatches = run_cpu_layout_exhaustive();
    std::printf(
        "{\"cpu\":{\"layout\":\"projection_qblock_row_4words\","
        "\"cases\":128,\"mismatches\":%d}}\n",
        cpu_mismatches);
    if (cpu_mismatches != 0) {
        return 4;
    }
    if (options.mode == Options::Mode::CpuOnly) {
        return 0;
    }
    int device_index = 0;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device_index));
    if (properties.major != 8 || properties.minor != 6) {
        std::fprintf(stderr, "exact SM86 required, got SM%d%d\n",
            properties.major, properties.minor);
        return 5;
    }
    if (options.mode == Options::Mode::GpuOracle) {
        return run_gpu_oracle(properties) == 0 ? 0 : 6;
    }
    if (options.mode == Options::Mode::Benchmark) {
        if (run_gpu_oracle(properties) != 0) {
            return 6;
        }
        return run_benchmark(options) == 0 ? 0 : 7;
    }
    return run_profile(options.profile_candidate, options.profile_tokens);
}
