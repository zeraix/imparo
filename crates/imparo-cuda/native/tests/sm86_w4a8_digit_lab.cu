#include "../sm86/mmq_q4_q8_digit_sliced_lab.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace imparo_sm86_w4a8_digit_lab;

struct Options {
    bool gpu_oracle = true;
    bool benchmark = false;
    int iterations = 2048;
    int cycles = 9;
    int warmup = 4;
    int blocks_per_sm = 4;
};

int parse_positive(const char * name, const char * value) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || parsed <= 0 || parsed > 1'000'000) {
        std::fprintf(stderr, "invalid %s: %s\n", name, value);
        std::exit(2);
    }
    return int(parsed);
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--cpu-only") == 0) {
            options.gpu_oracle = false;
        } else if (std::strcmp(argv[index], "--gpu-oracle") == 0) {
            options.gpu_oracle = true;
        } else if (std::strcmp(argv[index], "--benchmark") == 0) {
            options.gpu_oracle = true;
            options.benchmark = true;
        } else if (std::strcmp(argv[index], "--iterations") == 0
                || std::strcmp(argv[index], "--cycles") == 0
                || std::strcmp(argv[index], "--warmup") == 0
                || std::strcmp(argv[index], "--blocks-per-sm") == 0) {
            if (++index >= argc) {
                std::fprintf(stderr, "missing value for %s\n", argv[index - 1]);
                std::exit(2);
            }
            const char * name = argv[index - 1];
            const int value = parse_positive(name, argv[index]);
            if (std::strcmp(name, "--iterations") == 0) {
                options.iterations = value;
            } else if (std::strcmp(name, "--cycles") == 0) {
                options.cycles = value;
            } else if (std::strcmp(name, "--warmup") == 0) {
                options.warmup = value;
            } else {
                options.blocks_per_sm = value;
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
        std::fprintf(stderr, "CUDA failure at line %d for %s: %s\n",
            line, expression, cudaGetErrorString(status));
        std::exit(3);
    }
}

#define CHECK_CUDA(expression) check_cuda((expression), #expression, __LINE__)

int run_cpu_exhaustive_identity() {
    int mismatches = 0;
    int cases = 0;
    for (int storage = 0; storage < 16; ++storage) {
        for (int q8_integer = -128; q8_integer <= 127; ++q8_integer) {
            const auto q8 = static_cast<std::int8_t>(q8_integer);
            ++cases;
            if (!scalar_identity_holds(
                    static_cast<std::uint8_t>(storage), q8)) {
                if (mismatches < 8) {
                    std::fprintf(stderr,
                        "scalar mismatch q4_storage=%d q8=%d q4=%d low=%u "
                        "high=%d reconstructed=%d\n",
                        storage, q8_integer, q4_storage_value(storage),
                        unsigned(q8_low_u4(q8)), q8_high_s4(q8),
                        reconstruct_q8(q8));
                }
                ++mismatches;
            }
        }
    }
    if (cases != 4096) {
        std::fprintf(stderr, "internal exhaustive-case count mismatch: %d\n",
            cases);
        return 1;
    }
    return mismatches;
}

void fill_case(int case_index,
        std::array<std::uint8_t, kM * kK> & q4,
        std::array<std::int8_t, kN * kK> & q8,
        std::array<float, kM> & q4_scale,
        std::array<float, kN> & q8_scale,
        std::array<float, kOutputs> & initial) {
    for (int row = 0; row < kM; ++row) {
        for (int k = 0; k < kK; ++k) {
            int storage;
            if (case_index == 0) {
                storage = (row * 11 + k * 13) & 15;
            } else if (case_index == 1) {
                storage = ((row + k) & 1) ? 15 : 0;
            } else {
                const std::uint32_t hash = std::uint32_t(row * kK + k)
                    * 1664525u + 1013904223u;
                storage = int((hash >> 19) & 15u);
            }
            q4[row * kK + k] = static_cast<std::uint8_t>(storage);
        }
        q4_scale[row] = std::ldexp(1.0f, (row % 7) - 5);
    }

    for (int column = 0; column < kN; ++column) {
        for (int k = 0; k < kK; ++k) {
            int value;
            if (case_index == 0) {
                // N*K is exactly 256: cover every possible signed Q8 value.
                value = column * kK + k - 128;
            } else if (case_index == 1) {
                value = ((column + k) & 1) ? 127 : -128;
            } else {
                const std::uint32_t hash = std::uint32_t(column * kK + k)
                    * 22695477u + 1u;
                value = int((hash >> 16) & 255u) - 128;
            }
            q8[column * kK + k] = static_cast<std::int8_t>(value);
        }
        q8_scale[column] = std::ldexp(1.0f, (column % 5) - 4);
    }

    for (int index = 0; index < kOutputs; ++index) {
        initial[index] = float((index % 17) - 8) * 0.125f;
    }
}

void make_oracle(
        const std::array<std::uint8_t, kM * kK> & q4,
        const std::array<std::int8_t, kN * kK> & q8,
        const std::array<float, kM> & q4_scale,
        const std::array<float, kN> & q8_scale,
        const std::array<float, kOutputs> & initial,
        std::array<std::int32_t, kOutputs> & integer_oracle,
        std::array<float, kOutputs> & scaled_oracle) {
    integer_oracle.fill(0);
    for (int row = 0; row < kM; ++row) {
        for (int column = 0; column < kN; ++column) {
            std::int32_t dot = 0;
            for (int k = 0; k < kK; ++k) {
                dot += q4_storage_value(q4[row * kK + k])
                    * int(q8[column * kK + k]);
            }
            const int index = row * kN + column;
            integer_oracle[index] = dot;
            const float block_scale = q4_scale[row] * q8_scale[column];
            scaled_oracle[index] = std::fma(
                float(dot), block_scale, initial[index]);
        }
    }
}

template <typename T>
T * allocate_device(std::size_t count) {
    T * pointer = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&pointer),
        count * sizeof(T)));
    return pointer;
}

struct DeviceBuffers {
    std::uint8_t * q4 = allocate_device<std::uint8_t>(kM * kK);
    std::int8_t * q8 = allocate_device<std::int8_t>(kN * kK);
    float * q4_scale = allocate_device<float>(kM);
    float * q8_scale = allocate_device<float>(kN);
    float * initial = allocate_device<float>(kOutputs);
    std::int32_t * integer_output = allocate_device<std::int32_t>(kOutputs);
    float * scaled_output = allocate_device<float>(kOutputs);
    std::int32_t * integer_control = allocate_device<std::int32_t>(kOutputs);
    float * scaled_control = allocate_device<float>(kOutputs);

    ~DeviceBuffers() {
        cudaFree(scaled_control);
        cudaFree(integer_control);
        cudaFree(scaled_output);
        cudaFree(integer_output);
        cudaFree(initial);
        cudaFree(q8_scale);
        cudaFree(q4_scale);
        cudaFree(q8);
        cudaFree(q4);
    }
};

template <typename T, std::size_t Size>
void copy_to_device(T * destination, const std::array<T, Size> & source) {
    CHECK_CUDA(cudaMemcpy(destination, source.data(), sizeof(source),
        cudaMemcpyHostToDevice));
}

int run_gpu_fragment_oracle(const cudaDeviceProp & properties) {
    if (properties.major != 8 || properties.minor != 6) {
        std::fprintf(stderr, "exact SM86 device required, got SM%d%d\n",
            properties.major, properties.minor);
        return 1;
    }

    DeviceBuffers device;
    int control_integer_mismatches = 0;
    int digit_integer_mismatches = 0;
    int integer_route_mismatches = 0;
    int control_scaled_bit_mismatches = 0;
    int digit_scaled_bit_mismatches = 0;
    int scaled_route_mismatches = 0;
    float maximum_absolute_error = 0.0f;
    constexpr int kCases = 3;

    for (int case_index = 0; case_index < kCases; ++case_index) {
        std::array<std::uint8_t, kM * kK> q4{};
        std::array<std::int8_t, kN * kK> q8{};
        std::array<float, kM> q4_scale{};
        std::array<float, kN> q8_scale{};
        std::array<float, kOutputs> initial{};
        std::array<std::int32_t, kOutputs> integer_oracle{};
        std::array<float, kOutputs> scaled_oracle{};
        std::array<std::int32_t, kOutputs> integer_output{};
        std::array<float, kOutputs> scaled_output{};
        std::array<std::int32_t, kOutputs> integer_control{};
        std::array<float, kOutputs> scaled_control{};

        fill_case(case_index, q4, q8, q4_scale, q8_scale, initial);
        make_oracle(q4, q8, q4_scale, q8_scale, initial,
            integer_oracle, scaled_oracle);
        copy_to_device(device.q4, q4);
        copy_to_device(device.q8, q8);
        copy_to_device(device.q4_scale, q4_scale);
        copy_to_device(device.q8_scale, q8_scale);
        copy_to_device(device.initial, initial);

        single_k32_control_oracle<<<1, 32>>>(
            device.q4, device.q8, device.q4_scale, device.q8_scale,
            device.initial, device.integer_control, device.scaled_control);
        single_k32_fragment_oracle<<<1, 32>>>(
            device.q4, device.q8, device.q4_scale, device.q8_scale,
            device.initial, device.integer_output, device.scaled_output);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(integer_output.data(), device.integer_output,
            sizeof(integer_output), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(scaled_output.data(), device.scaled_output,
            sizeof(scaled_output), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(integer_control.data(), device.integer_control,
            sizeof(integer_control), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(scaled_control.data(), device.scaled_control,
            sizeof(scaled_control), cudaMemcpyDeviceToHost));

        for (int index = 0; index < kOutputs; ++index) {
            if (integer_control[index] != integer_oracle[index]) {
                if (control_integer_mismatches < 8) {
                    std::fprintf(stderr,
                        "control integer mismatch case=%d index=%d expected=%d "
                        "got=%d\n",
                        case_index, index, integer_oracle[index],
                        integer_control[index]);
                }
                ++control_integer_mismatches;
            }
            if (integer_output[index] != integer_oracle[index]) {
                if (digit_integer_mismatches < 8) {
                    std::fprintf(stderr,
                        "digit integer mismatch case=%d index=%d expected=%d "
                        "got=%d\n",
                        case_index, index, integer_oracle[index],
                        integer_output[index]);
                }
                ++digit_integer_mismatches;
            }
            if (integer_output[index] != integer_control[index]) {
                ++integer_route_mismatches;
            }
            const float absolute_error = std::fabs(
                scaled_output[index] - scaled_oracle[index]);
            maximum_absolute_error = std::max(
                maximum_absolute_error, absolute_error);
            if (std::memcmp(&scaled_control[index], &scaled_oracle[index],
                    sizeof(float)) != 0) {
                if (control_scaled_bit_mismatches < 8) {
                    std::fprintf(stderr,
                        "control scaled mismatch case=%d index=%d expected=%.9g "
                        "got=%.9g abs=%.9g\n",
                        case_index, index, scaled_oracle[index],
                        scaled_control[index], std::fabs(
                            scaled_control[index] - scaled_oracle[index]));
                }
                ++control_scaled_bit_mismatches;
            }
            if (std::memcmp(&scaled_output[index], &scaled_oracle[index],
                    sizeof(float)) != 0) {
                if (digit_scaled_bit_mismatches < 8) {
                    std::fprintf(stderr,
                        "digit scaled mismatch case=%d index=%d expected=%.9g "
                        "got=%.9g "
                        "abs=%.9g\n",
                        case_index, index, scaled_oracle[index],
                        scaled_output[index], absolute_error);
                }
                ++digit_scaled_bit_mismatches;
            }
            if (std::memcmp(&scaled_output[index], &scaled_control[index],
                    sizeof(float)) != 0) {
                ++scaled_route_mismatches;
            }
        }
    }

    std::printf(
        "{\"gpu\":{\"device\":\"%s\",\"sm\":86,\"cases\":%d,"
        "\"outputs\":%d,\"control_integer_mismatches\":%d,"
        "\"digit_integer_mismatches\":%d,\"integer_route_mismatches\":%d,"
        "\"control_scaled_bit_mismatches\":%d,"
        "\"digit_scaled_bit_mismatches\":%d,\"scaled_route_mismatches\":%d,"
        "\"max_abs_error\":%.9g}}\n",
        properties.name, kCases, kCases * kOutputs,
        control_integer_mismatches, digit_integer_mismatches,
        integer_route_mismatches, control_scaled_bit_mismatches,
        digit_scaled_bit_mismatches, scaled_route_mismatches,
        maximum_absolute_error);
    return control_integer_mismatches == 0
        && digit_integer_mismatches == 0
        && integer_route_mismatches == 0
        && control_scaled_bit_mismatches == 0
        && digit_scaled_bit_mismatches == 0
        && scaled_route_mismatches == 0 ? 0 : 1;
}

std::uint32_t host_pack_nibble(
        std::uint32_t packed, std::uint8_t value, int index) {
    return packed | (std::uint32_t(value & 0x0fu) << (4 * index));
}

std::uint32_t host_pack_byte(
        std::uint32_t packed, std::int8_t value, int index) {
    return packed
        | (std::uint32_t(static_cast<std::uint8_t>(value)) << (8 * index));
}

void make_packed_q4(
        const std::array<std::uint8_t, kM * kK> & q4,
        std::array<std::uint8_t, kPackedQ4Bytes> & packed_q4) {
    packed_q4.fill(0);
    for (int row = 0; row < kM; ++row) {
        for (int k = 0; k < kK; k += 2) {
            packed_q4[row * (kK / 2) + k / 2] = std::uint8_t(
                (q4[row * kK + k] & 0x0fu)
                | ((q4[row * kK + k + 1] & 0x0fu) << 4));
        }
    }
}

void prepare_fragments(
        const std::array<std::uint8_t, kM * kK> & q4,
        const std::array<std::int8_t, kN * kK> & q8,
        std::array<std::uint32_t, 32 * 4> & control_a,
        std::array<std::uint32_t, 32 * 2> & control_b,
        std::array<std::uint32_t, 32 * 2> & digit_a,
        std::array<std::uint32_t, 32> & digit_b_low,
        std::array<std::uint32_t, 32> & digit_b_high) {
    for (int lane = 0; lane < 32; ++lane) {
        const int group = lane >> 2;
        const int thread_in_group = lane & 3;
        for (int reg = 0; reg < 4; ++reg) {
            const int row = (reg & 1) != 0 ? group + 8 : group;
            const int k_base = thread_in_group * 4 + (reg >= 2 ? 16 : 0);
            std::uint32_t packed = 0;
            for (int item = 0; item < 4; ++item) {
                packed = host_pack_byte(packed,
                    static_cast<std::int8_t>(q4_storage_value(
                        q4[row * kK + k_base + item])), item);
            }
            control_a[lane * 4 + reg] = packed;
        }
        for (int reg = 0; reg < 2; ++reg) {
            const int k_base = thread_in_group * 4 + reg * 16;
            std::uint32_t packed = 0;
            for (int item = 0; item < 4; ++item) {
                packed = host_pack_byte(packed,
                    q8[group * kK + k_base + item], item);
            }
            control_b[lane * 2 + reg] = packed;
        }

        const int digit_k_base = thread_in_group * 8;
        std::uint32_t a0 = 0;
        std::uint32_t a1 = 0;
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        for (int item = 0; item < 8; ++item) {
            a0 = host_pack_nibble(a0, q4_storage_to_s4_bits(
                q4[group * kK + digit_k_base + item]), item);
            a1 = host_pack_nibble(a1, q4_storage_to_s4_bits(
                q4[(group + 8) * kK + digit_k_base + item]), item);
            const std::int8_t value =
                q8[group * kK + digit_k_base + item];
            low = host_pack_nibble(low, q8_low_u4(value), item);
            high = host_pack_nibble(high, q8_high_s4_bits(value), item);
        }
        digit_a[lane * 2] = a0;
        digit_a[lane * 2 + 1] = a1;
        digit_b_low[lane] = low;
        digit_b_high[lane] = high;
    }
}

struct BenchmarkDeviceBuffers {
    explicit BenchmarkDeviceBuffers(std::size_t output_count)
        : packed_q4(allocate_device<std::uint8_t>(
              kBenchmarkRawTiles * kPackedQ4Bytes)),
          q8(allocate_device<std::int8_t>(
              kBenchmarkRawTiles * kN * kK)),
          q4_scale(allocate_device<float>(kM)),
          q8_scale(allocate_device<float>(kN)),
          initial(allocate_device<float>(kOutputs)),
          control_a(allocate_device<std::uint32_t>(32 * 4)),
          control_b(allocate_device<std::uint32_t>(32 * 2)),
          digit_a(allocate_device<std::uint32_t>(32 * 2)),
          digit_b_low(allocate_device<std::uint32_t>(32)),
          digit_b_high(allocate_device<std::uint32_t>(32)),
          output_a(allocate_device<float>(output_count)),
          output_b(allocate_device<float>(output_count)) {}

    ~BenchmarkDeviceBuffers() {
        cudaFree(output_b);
        cudaFree(output_a);
        cudaFree(digit_b_high);
        cudaFree(digit_b_low);
        cudaFree(digit_a);
        cudaFree(control_b);
        cudaFree(control_a);
        cudaFree(initial);
        cudaFree(q8_scale);
        cudaFree(q4_scale);
        cudaFree(q8);
        cudaFree(packed_q4);
    }

    std::uint8_t * packed_q4;
    std::int8_t * q8;
    float * q4_scale;
    float * q8_scale;
    float * initial;
    std::uint32_t * control_a;
    std::uint32_t * control_b;
    std::uint32_t * digit_a;
    std::uint32_t * digit_b_low;
    std::uint32_t * digit_b_high;
    float * output_a;
    float * output_b;
};

enum class BenchmarkKind {
    Consumer,
    Honest,
};

void launch_benchmark_kernel(BenchmarkKind kind, bool digit,
        const BenchmarkDeviceBuffers & device, int blocks, int iterations,
        cudaStream_t stream, float * output) {
    const dim3 threads(kBenchmarkThreads);
    if (kind == BenchmarkKind::Consumer) {
        if (digit) {
            consumer_digit_benchmark<<<blocks, threads, 0, stream>>>(
                device.digit_a, device.digit_b_low, device.digit_b_high,
                device.q4_scale, device.q8_scale, device.initial,
                iterations, output);
        } else {
            consumer_control_benchmark<<<blocks, threads, 0, stream>>>(
                device.control_a, device.control_b, device.q4_scale,
                device.q8_scale, device.initial, iterations, output);
        }
    } else if (digit) {
        honest_digit_benchmark<<<blocks, threads, 0, stream>>>(
            device.packed_q4, device.q8, device.q4_scale, device.q8_scale,
            device.initial, iterations, output);
    } else {
        honest_control_benchmark<<<blocks, threads, 0, stream>>>(
            device.packed_q4, device.q8, device.q4_scale, device.q8_scale,
            device.initial, iterations, output);
    }
    CHECK_CUDA(cudaGetLastError());
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const std::size_t size = values.size();
    return (size & 1u) != 0
        ? values[size / 2]
        : 0.5f * (values[size / 2 - 1] + values[size / 2]);
}

float median_absolute_deviation(
        const std::vector<float> & values, float center) {
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

struct BenchmarkResult {
    bool output_bit_exact = false;
    std::vector<float> raw;
    std::vector<float> control;
    std::vector<float> digit;
    float control_median = 0.0f;
    float control_mad = 0.0f;
    float digit_median = 0.0f;
    float digit_mad = 0.0f;
};

template <typename T>
void copy_vector_to_device(T * destination, const std::vector<T> & source) {
    CHECK_CUDA(cudaMemcpy(destination, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

BenchmarkResult benchmark_kind(BenchmarkKind kind,
        const BenchmarkDeviceBuffers & device, int blocks,
        std::size_t output_count, const Options & options,
        cudaStream_t stream) {
    BenchmarkResult result;
    constexpr int kValidationIterations = 7;
    launch_benchmark_kernel(kind, false, device, blocks,
        kValidationIterations, stream, device.output_a);
    launch_benchmark_kernel(kind, true, device, blocks,
        kValidationIterations, stream, device.output_b);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> control_output(output_count);
    std::vector<float> digit_output(output_count);
    CHECK_CUDA(cudaMemcpy(control_output.data(), device.output_a,
        output_count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(digit_output.data(), device.output_b,
        output_count * sizeof(float), cudaMemcpyDeviceToHost));
    result.output_bit_exact = std::memcmp(control_output.data(),
        digit_output.data(), output_count * sizeof(float)) == 0;
    if (!result.output_bit_exact) {
        int mismatches = 0;
        for (std::size_t index = 0; index < output_count; ++index) {
            if (std::memcmp(&control_output[index], &digit_output[index],
                    sizeof(float)) != 0) {
                if (mismatches < 8) {
                    std::fprintf(stderr,
                        "benchmark output mismatch kind=%s index=%zu "
                        "control=%.9g digit=%.9g\n",
                        kind == BenchmarkKind::Consumer ? "consumer" : "honest",
                        index, control_output[index], digit_output[index]);
                }
                ++mismatches;
            }
        }
        std::fprintf(stderr, "benchmark route mismatches: %d\n", mismatches);
        return result;
    }

    for (int index = 0; index < options.warmup; ++index) {
        launch_benchmark_kernel(kind, false, device, blocks,
            options.iterations, stream, device.output_a);
        launch_benchmark_kernel(kind, true, device, blocks,
            options.iterations, stream, device.output_b);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    constexpr bool kDigitSchedule[8] = {
        false, true, true, false, true, false, false, true};
    result.raw.reserve(std::size_t(options.cycles) * 8);
    result.control.reserve(std::size_t(options.cycles) * 4);
    result.digit.reserve(std::size_t(options.cycles) * 4);
    for (int cycle = 0; cycle < options.cycles; ++cycle) {
        for (const bool digit : kDigitSchedule) {
            CHECK_CUDA(cudaEventRecord(start, stream));
            launch_benchmark_kernel(kind, digit, device, blocks,
                options.iterations, stream,
                digit ? device.output_b : device.output_a);
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            const float microseconds = milliseconds * 1000.0f;
            result.raw.push_back(microseconds);
            (digit ? result.digit : result.control).push_back(microseconds);
        }
    }
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));
    result.control_median = median(result.control);
    result.control_mad = median_absolute_deviation(
        result.control, result.control_median);
    result.digit_median = median(result.digit);
    result.digit_mad = median_absolute_deviation(
        result.digit, result.digit_median);
    return result;
}

void print_benchmark_result(const char * name,
        const BenchmarkResult & result, int blocks, const Options & options,
        float threshold) {
    const float ratio = result.control_median / result.digit_median;
    std::printf(
        "{\"benchmark\":\"%s\",\"lab_only\":true,"
        "\"production_enabled\":false,\"clock\":\"cuda_event\","
        "\"same_stream\":true,\"shape\":\"m16n8k32\","
        "\"threads\":%d,\"warps_per_block\":%d,\"blocks\":%d,"
        "\"iterations\":%d,\"warmup_per_route\":%d,\"cycles\":%d,"
        "\"schedule_unit\":\"ABBABAAB\",\"raw_routes_repeated\":%d,"
        "\"output_bit_exact\":%s,\"raw_us\":",
        name, kBenchmarkThreads, kBenchmarkWarps, blocks,
        options.iterations, options.warmup, options.cycles, options.cycles,
        result.output_bit_exact ? "true" : "false");
    print_samples(result.raw);
    std::printf(",\"control_us\":");
    print_samples(result.control);
    std::printf(",\"digit_us\":");
    print_samples(result.digit);
    std::printf(
        ",\"control_median_us\":%.6f,\"control_mad_us\":%.6f,"
        "\"digit_median_us\":%.6f,\"digit_mad_us\":%.6f,"
        "\"control_over_digit\":%.6f,\"threshold\":%.6f,"
        "\"gate_pass\":%s}\n",
        result.control_median, result.control_mad,
        result.digit_median, result.digit_mad, ratio, threshold,
        ratio >= threshold ? "true" : "false");
}

int run_benchmarks(const cudaDeviceProp & properties, const Options & options) {
    const int blocks = properties.multiProcessorCount * options.blocks_per_sm;
    const std::size_t output_count = std::size_t(blocks)
        * kBenchmarkWarps * kOutputs;
    BenchmarkDeviceBuffers device(output_count);
    std::array<std::uint8_t, kM * kK> q4{};
    std::array<std::int8_t, kN * kK> q8{};
    std::array<float, kM> q4_scale{};
    std::array<float, kN> q8_scale{};
    std::array<float, kOutputs> initial{};
    std::array<std::uint8_t, kPackedQ4Bytes> packed_q4{};
    std::array<std::uint32_t, 32 * 4> control_a{};
    std::array<std::uint32_t, 32 * 2> control_b{};
    std::array<std::uint32_t, 32 * 2> digit_a{};
    std::array<std::uint32_t, 32> digit_b_low{};
    std::array<std::uint32_t, 32> digit_b_high{};
    fill_case(2, q4, q8, q4_scale, q8_scale, initial);
    make_packed_q4(q4, packed_q4);
    std::vector<std::uint8_t> packed_q4_pool(
        kBenchmarkRawTiles * kPackedQ4Bytes);
    std::vector<std::int8_t> q8_pool(
        kBenchmarkRawTiles * kN * kK);
    for (int tile = 0; tile < kBenchmarkRawTiles; ++tile) {
        std::array<std::uint8_t, kM * kK> tile_q4{};
        std::array<std::int8_t, kN * kK> tile_q8{};
        for (int index = 0; index < kM * kK; ++index) {
            tile_q4[index] = std::uint8_t(
                (int(q4[index]) + tile * 7 + (index >> 5)) & 15);
        }
        for (int index = 0; index < kN * kK; ++index) {
            const std::uint8_t bits = std::uint8_t(q8[index]);
            tile_q8[index] = static_cast<std::int8_t>(std::uint8_t(
                unsigned(bits) + unsigned(tile * 29 + (index >> 4))));
        }
        std::array<std::uint8_t, kPackedQ4Bytes> tile_packed{};
        make_packed_q4(tile_q4, tile_packed);
        std::copy(tile_packed.begin(), tile_packed.end(),
            packed_q4_pool.begin() + std::size_t(tile) * kPackedQ4Bytes);
        std::copy(tile_q8.begin(), tile_q8.end(),
            q8_pool.begin() + std::size_t(tile) * kN * kK);
    }
    prepare_fragments(q4, q8, control_a, control_b,
        digit_a, digit_b_low, digit_b_high);
    copy_vector_to_device(device.packed_q4, packed_q4_pool);
    copy_vector_to_device(device.q8, q8_pool);
    copy_to_device(device.q4_scale, q4_scale);
    copy_to_device(device.q8_scale, q8_scale);
    copy_to_device(device.initial, initial);
    copy_to_device(device.control_a, control_a);
    copy_to_device(device.control_b, control_b);
    copy_to_device(device.digit_a, digit_a);
    copy_to_device(device.digit_b_low, digit_b_low);
    copy_to_device(device.digit_b_high, digit_b_high);

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const BenchmarkResult consumer = benchmark_kind(
        BenchmarkKind::Consumer, device, blocks, output_count, options, stream);
    if (!consumer.output_bit_exact) {
        CHECK_CUDA(cudaStreamDestroy(stream));
        return 1;
    }
    const BenchmarkResult honest = benchmark_kind(
        BenchmarkKind::Honest, device, blocks, output_count, options, stream);
    CHECK_CUDA(cudaStreamDestroy(stream));
    if (!honest.output_bit_exact) {
        return 1;
    }
    print_benchmark_result("consumer_only", consumer, blocks, options, 1.10f);
    print_benchmark_result("honest_operator", honest, blocks, options, 1.05f);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const Options options = parse_options(argc, argv);
    const int cpu_mismatches = run_cpu_exhaustive_identity();
    std::printf(
        "{\"cpu\":{\"cases\":4096,\"mismatches\":%d,"
        "\"q4_transform\":\"nibble_xor_8_to_s4\","
        "\"q8_identity\":\"low_u4_plus_16_high_s4\"}}\n",
        cpu_mismatches);
    if (cpu_mismatches != 0) {
        return 4;
    }
    if (!options.gpu_oracle) {
        return 0;
    }

    int device_index = 0;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device_index));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device_index));
    if (run_gpu_fragment_oracle(properties) != 0) {
        return 5;
    }
    if (options.benchmark && run_benchmarks(properties, options) != 0) {
        return 6;
    }
    return 0;
}
