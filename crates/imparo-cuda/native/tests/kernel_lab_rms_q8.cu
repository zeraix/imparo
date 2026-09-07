// Phase A2 only: standalone, deletable RMSNorm -> Q8_1 MMQ comparison harness.
//
// This translation unit intentionally includes the native backend so the control is
// the exact production k_rms_norm_q8_1_mmq<1024> implementation, on the exact engine
// primary context and g.stream. It is never linked into the backend library and must
// be compiled explicitly with -DIMPARO_CUDA_KERNEL_LAB=1.

#if !defined(IMPARO_CUDA_KERNEL_LAB) || IMPARO_CUDA_KERNEL_LAB != 1
#error "kernel_lab_rms_q8.cu is test-only; define IMPARO_CUDA_KERNEL_LAB=1"
#endif

#include "../imparo_cuda.cu"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

namespace {

constexpr uint32_t kWidth = 2560;
constexpr uint32_t kTokens = 512;
constexpr uint32_t kNativeThreads = 1024;
constexpr uint32_t kSharedBytes = 32 * sizeof(float);
constexpr size_t kGuardBytes = 256;
constexpr uint8_t kGuard = 0xa5;

struct GuardedBuffer {
    uint8_t * allocation = nullptr;
    uint8_t * data = nullptr;
    size_t bytes = 0;

    bool allocate(size_t requested) {
        bytes = requested;
        if (cudaMalloc(reinterpret_cast<void **>(&allocation),
                       bytes + 2 * kGuardBytes) != cudaSuccess) return false;
        data = allocation + kGuardBytes;
        return cudaMemsetAsync(allocation, kGuard,
                   bytes + 2 * kGuardBytes, g.stream) == cudaSuccess;
    }

    uint64_t canary_errors() const {
        std::array<uint8_t, kGuardBytes> before{};
        std::array<uint8_t, kGuardBytes> after{};
        if (cudaMemcpy(before.data(), allocation, kGuardBytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(after.data(), data + bytes, kGuardBytes,
                          cudaMemcpyDeviceToHost) != cudaSuccess) {
            return UINT64_MAX;
        }
        uint64_t errors = 0;
        for (uint8_t value : before) errors += value != kGuard;
        for (uint8_t value : after) errors += value != kGuard;
        return errors;
    }

    void release() {
        if (allocation) (void)cudaFree(allocation);
        allocation = nullptr;
        data = nullptr;
        bytes = 0;
    }
};

struct Difference {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq = 0.0;
    uint64_t count = 0;
    uint64_t non_finite = 0;

    void add(double actual, double expected) {
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++non_finite;
            return;
        }
        const double absolute = std::fabs(actual - expected);
        const double relative = absolute / std::max(std::fabs(expected), 1.0e-30);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, relative);
        sum_sq += absolute * absolute;
        ++count;
    }

    double rms() const {
        return count ? std::sqrt(sum_sq / double(count)) : 0.0;
    }
};

struct Timing {
    std::vector<float> native_us;
    std::vector<float> triton_us;
};

struct Resources {
    int registers = -1;
    int static_shared = -1;
    int local = -1;
    int max_threads = -1;
    int max_dynamic_shared = -1;
    int binary_version = -1;
};

struct VariantResult {
    std::string symbol;
    uint32_t warps = 0;
    uint32_t dynamic_shared_bytes = 0;
    uint64_t cubin_bytes = 0;
    double module_load_wall_us = 0.0;
    uint64_t module_device_bytes_delta = 0;
    float cold_first_us = 0.0f;
    Resources resources;
    Difference normalized_vs_native;
    Difference normalized_vs_reference;
    Difference q8_scales_vs_native;
    Difference q8_dequant;
    uint64_t q8_value_mismatches = 0;
    uint64_t q8_value_total = 0;
    uint32_t q8_value_max_abs = 0;
    uint64_t canary_errors = 0;
    Timing timing;
};

[[noreturn]] void fail(const std::string & message) {
    std::cerr << "kernel-lab: " << message << "\n";
    std::exit(2);
}

bool trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("IMPARO_CUDA_KERNEL_LAB_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void trace(const char * phase) {
    if (trace_enabled()) std::cerr << "kernel-lab-phase: " << phase << "\n";
}

void cuda_ok(cudaError_t result, const char * operation) {
    if (result != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void driver_ok(CUresult result, const char * operation) {
    if (result != CUDA_SUCCESS) {
        std::ostringstream out;
        out << operation << " failed with CUresult " << int(result);
        fail(out.str());
    }
}

std::vector<uint8_t> read_file(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail("cannot open cubin");
    const std::streamsize size = input.tellg();
    if (size <= 0) fail("cubin is empty");
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char *>(bytes.data()), size)) {
        fail("cannot read complete cubin");
    }
    return bytes;
}

template <typename Launch>
float event_us(Launch && launch) {
    static uint64_t event_call = 0;
    const uint64_t call = ++event_call;
    if (trace_enabled() && call <= 3) {
        std::cerr << "kernel-lab-event-" << call << ": create\n";
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_ok(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_ok(cudaEventRecord(start, g.stream), "cudaEventRecord(start)");
    if (trace_enabled() && call <= 3) {
        std::cerr << "kernel-lab-event-" << call << ": launch\n";
    }
    launch();
    if (trace_enabled() && call <= 3) {
        std::cerr << "kernel-lab-event-" << call << ": launched\n";
    }
    cuda_ok(cudaEventRecord(stop, g.stream), "cudaEventRecord(stop)");
    if (trace_enabled() && call <= 3) {
        std::cerr << "kernel-lab-event-" << call << ": synchronize\n";
    }
    cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float milliseconds = 0.0f;
    cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop),
            "cudaEventElapsedTime");
    cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    if (trace_enabled() && call <= 3) {
        std::cerr << "kernel-lab-event-" << call << ": complete\n";
    }
    return milliseconds * 1000.0f;
}

void launch_native(const GuardedBuffer & src, GuardedBuffer & dst,
                   const GuardedBuffer & mul, GuardedBuffer & q8, float eps) {
    k_rms_norm_q8_1_mmq<kNativeThreads>
        <<<kTokens, kNativeThreads, kSharedBytes, g.stream>>>(
            reinterpret_cast<const float *>(src.data),
            reinterpret_cast<float *>(dst.data),
            reinterpret_cast<const float *>(mul.data),
            reinterpret_cast<BlockQ8_1Mmq *>(q8.data),
            kWidth, kTokens, eps);
    cuda_ok(cudaPeekAtLastError(), "native RMSNorm-Q8 launch");
}

void launch_triton(CUfunction function, uint32_t warps, uint32_t dynamic_shared,
                   const GuardedBuffer & src, GuardedBuffer & dst,
                   const GuardedBuffer & mul, GuardedBuffer & q8, float eps) {
    void * src_pointer = src.data;
    void * dst_pointer = dst.data;
    void * mul_pointer = mul.data;
    void * q8_bytes_pointer = q8.data;
    void * q8_scales_pointer = q8.data + 128;
    CUdeviceptr global_scratch = 0;
    CUdeviceptr profile_scratch = 0;
    void * arguments[] = {
        &src_pointer, &dst_pointer, &mul_pointer,
        &q8_bytes_pointer, &q8_scales_pointer, &eps,
        &global_scratch, &profile_scratch,
    };
    if (!program_catalog.driver.launch_kernel) {
        fail("CUDA Driver launch function pointer is null");
    }
    static uint64_t launch_count = 0;
    const uint64_t call = ++launch_count;
    if (trace_enabled() && call <= 2) {
        std::cerr << "kernel-lab-driver-launch-" << call << ": enter\n";
    }
    const CUresult launch_result = program_catalog.driver.launch_kernel(
        function, kTokens, 1, 1, warps * 32, 1, 1, dynamic_shared,
        reinterpret_cast<CUstream>(g.stream), arguments, nullptr);
    if (trace_enabled() && call <= 2) {
        std::cerr << "kernel-lab-driver-launch-" << call << ": return\n";
    }
    driver_ok(launch_result, "Triton RMSNorm-Q8 launch");
}

Resources query_resources(CUfunction function) {
    Resources result;
    auto attribute = [&](int kind, int * value, const char * operation) {
        driver_ok(program_catalog.driver.function_get_attribute(value,
                      CUfunction_attribute(kind), function), operation);
    };
    attribute(CU_FUNC_ATTRIBUTE_NUM_REGS, &result.registers, "query registers");
    attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, &result.static_shared,
              "query static shared memory");
    attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, &result.local,
              "query local memory");
    attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &result.max_threads,
              "query max threads");
    attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
              &result.max_dynamic_shared, "query max dynamic shared memory");
    attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION, &result.binary_version,
              "query binary version");
    return result;
}

std::vector<float> make_source() {
    std::vector<float> values(uint64_t(kWidth) * kTokens);
    for (uint64_t index = 0; index < values.size(); ++index) {
        const int32_t integer = int32_t((index * 37 + (index / kWidth) * 13) % 257) - 128;
        values[index] = float(integer) / 64.0f;
    }
    values.front() = -3.75f;
    values.back() = 4.25f;
    return values;
}

std::vector<float> make_multiplier() {
    std::vector<float> values(kWidth);
    for (uint32_t index = 0; index < kWidth; ++index) {
        values[index] = 1.0f + float(int32_t((index * 17) % 31) - 15) / 64.0f;
    }
    values.front() = 0.5f;
    values.back() = 1.5f;
    return values;
}

std::vector<float> reference_normalized(const std::vector<float> & src,
                                        const std::vector<float> & mul,
                                        float eps) {
    std::vector<float> output(src.size());
    for (uint32_t token = 0; token < kTokens; ++token) {
        double square_sum = 0.0;
        const uint64_t row = uint64_t(token) * kWidth;
        for (uint32_t column = 0; column < kWidth; ++column) {
            const double value = src[row + column];
            square_sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(float(square_sum / kWidth) + eps);
        for (uint32_t column = 0; column < kWidth; ++column) {
            output[row + column] = scale * src[row + column] * mul[column];
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_device(const GuardedBuffer & buffer, size_t count) {
    std::vector<T> result(count);
    cuda_ok(cudaMemcpy(result.data(), buffer.data, count * sizeof(T),
                       cudaMemcpyDeviceToHost), "copy result from device");
    return result;
}

Difference compare_floats(const std::vector<float> & actual,
                          const std::vector<float> & expected) {
    if (actual.size() != expected.size()) fail("float comparison size mismatch");
    Difference result;
    for (size_t index = 0; index < actual.size(); ++index) {
        result.add(actual[index], expected[index]);
    }
    return result;
}

void compare_q8(const std::vector<uint8_t> & triton,
                const std::vector<uint8_t> & native,
                const std::vector<float> & triton_normalized,
                VariantResult * result) {
    const uint64_t groups = kWidth / 128;
    for (uint64_t group = 0; group < groups; ++group) {
        for (uint32_t token = 0; token < kTokens; ++token) {
            const uint64_t record = (group * kTokens + token) * sizeof(BlockQ8_1Mmq);
            for (uint32_t block = 0; block < 4; ++block) {
                float actual_scale = 0.0f;
                float native_scale = 0.0f;
                std::memcpy(&actual_scale,
                            triton.data() + record + 128 + block * sizeof(float),
                            sizeof(float));
                std::memcpy(&native_scale,
                            native.data() + record + 128 + block * sizeof(float),
                            sizeof(float));
                result->q8_scales_vs_native.add(actual_scale, native_scale);
                for (uint32_t lane = 0; lane < 32; ++lane) {
                    const uint64_t byte = record + block * 32 + lane;
                    const int32_t actual = int8_t(triton[byte]);
                    const int32_t expected = int8_t(native[byte]);
                    const uint32_t difference = uint32_t(std::abs(actual - expected));
                    ++result->q8_value_total;
                    result->q8_value_mismatches += difference != 0;
                    result->q8_value_max_abs =
                        std::max(result->q8_value_max_abs, difference);
                    const uint64_t column = group * 128 + block * 32 + lane;
                    const uint64_t normalized = uint64_t(token) * kWidth + column;
                    result->q8_dequant.add(double(actual) * actual_scale,
                                           triton_normalized[normalized]);
                }
            }
        }
    }
}

template <typename NativeLaunch, typename TritonLaunch>
Timing interleaved_timing(uint32_t warmup, uint32_t pairs,
                          uint32_t launches_per_sample,
                          NativeLaunch && native, TritonLaunch && triton) {
    for (uint32_t index = 0; index < warmup; ++index) {
        native();
        triton();
    }
    cuda_ok(cudaStreamSynchronize(g.stream), "warmup synchronize");
    Timing result;
    result.native_us.reserve(uint64_t(pairs) * 2);
    result.triton_us.reserve(uint64_t(pairs) * 2);
    auto sample = [&](auto && launch) {
        return event_us([&] {
            for (uint32_t repeat = 0; repeat < launches_per_sample; ++repeat) {
                launch();
            }
        }) / float(launches_per_sample);
    };
    for (uint32_t pair = 0; pair < pairs; ++pair) {
        if ((pair & 1) == 0) {
            result.native_us.push_back(sample(native));
            result.triton_us.push_back(sample(triton));
            result.triton_us.push_back(sample(triton));
            result.native_us.push_back(sample(native));
        } else {
            result.triton_us.push_back(sample(triton));
            result.native_us.push_back(sample(native));
            result.native_us.push_back(sample(native));
            result.triton_us.push_back(sample(triton));
        }
    }
    return result;
}

double median(std::vector<float> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() & 1 ? values[middle]
                             : 0.5 * (values[middle - 1] + values[middle]);
}

double mean(const std::vector<float> & values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double coefficient_of_variation(const std::vector<float> & values) {
    if (values.size() < 2) return std::numeric_limits<double>::infinity();
    const double average = mean(values);
    if (!(average > 0.0)) return std::numeric_limits<double>::infinity();
    double squared = 0.0;
    for (float value : values) {
        const double delta = double(value) - average;
        squared += delta * delta;
    }
    return std::sqrt(squared / double(values.size() - 1)) / average;
}

double paired_mad_fraction(const Timing & value) {
    if (value.native_us.size() != value.triton_us.size()
        || value.native_us.empty()) return std::numeric_limits<double>::infinity();
    std::vector<float> ratios;
    ratios.reserve(value.native_us.size());
    for (size_t index = 0; index < value.native_us.size(); ++index) {
        if (!(value.triton_us[index] > 0.0f)) {
            return std::numeric_limits<double>::infinity();
        }
        ratios.push_back(value.native_us[index] / value.triton_us[index]);
    }
    const double center = median(ratios);
    if (!(center > 0.0)) return std::numeric_limits<double>::infinity();
    std::vector<float> deviations;
    deviations.reserve(ratios.size());
    for (float ratio : ratios) {
        deviations.push_back(float(std::fabs(double(ratio) - center)));
    }
    return median(deviations) / center;
}

void emit_difference(const Difference & value) {
    std::cout << "{\"max_abs\":" << value.max_abs
              << ",\"max_rel\":" << value.max_rel
              << ",\"rms\":" << value.rms()
              << ",\"count\":" << value.count
              << ",\"non_finite\":" << value.non_finite << "}";
}

void emit_samples(const std::vector<float> & values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void emit_timing(const Timing & value) {
    const double native_median = median(value.native_us);
    const double triton_median = median(value.triton_us);
    std::cout << "{\"native_median_us\":" << native_median
              << ",\"triton_median_us\":" << triton_median
              << ",\"native_mean_us\":" << mean(value.native_us)
              << ",\"triton_mean_us\":" << mean(value.triton_us)
              << ",\"native_cv\":" << coefficient_of_variation(value.native_us)
              << ",\"triton_cv\":" << coefficient_of_variation(value.triton_us)
              << ",\"paired_mad_fraction\":" << paired_mad_fraction(value)
              << ",\"speedup_median\":"
              << (triton_median > 0.0 ? native_median / triton_median : 0.0)
              << ",\"native_samples_us\":";
    emit_samples(value.native_us);
    std::cout << ",\"triton_samples_us\":";
    emit_samples(value.triton_us);
    std::cout << '}';
}

void emit_resources(const Resources & value) {
    std::cout << "{\"registers_per_thread\":" << value.registers
              << ",\"static_shared_bytes\":" << value.static_shared
              << ",\"local_bytes\":" << value.local
              << ",\"max_threads_per_block\":" << value.max_threads
              << ",\"max_dynamic_shared_bytes\":" << value.max_dynamic_shared
              << ",\"binary_version\":" << value.binary_version << "}";
}

void emit_variant(const VariantResult & value) {
    std::cout << "{\"symbol\":\"" << value.symbol << "\",\"warps\":"
              << value.warps
              << ",\"dynamic_shared_bytes\":" << value.dynamic_shared_bytes
              << ",\"cubin_bytes\":" << value.cubin_bytes
              << ",\"module_load_wall_us\":" << value.module_load_wall_us
              << ",\"module_device_bytes_delta\":"
              << value.module_device_bytes_delta
              << ",\"cold_first_us\":" << value.cold_first_us
              << ",\"resources\":";
    emit_resources(value.resources);
    std::cout << ",\"normalized_vs_native\":";
    emit_difference(value.normalized_vs_native);
    std::cout << ",\"normalized_vs_reference\":";
    emit_difference(value.normalized_vs_reference);
    std::cout << ",\"q8_scales_vs_native\":";
    emit_difference(value.q8_scales_vs_native);
    std::cout << ",\"q8_value_mismatches\":" << value.q8_value_mismatches
              << ",\"q8_value_total\":" << value.q8_value_total
              << ",\"q8_byte_agreement\":"
              << (value.q8_value_total
                      ? 1.0 - double(value.q8_value_mismatches)
                            / double(value.q8_value_total)
                      : 0.0)
              << ",\"q8_value_max_abs\":" << value.q8_value_max_abs
              << ",\"q8_dequant\":";
    emit_difference(value.q8_dequant);
    std::cout << ",\"canary_errors\":" << value.canary_errors
              << ",\"timing\":";
    emit_timing(value.timing);
    std::cout << '}';
}

VariantResult run_variant(CUfunction function, const char * symbol, uint32_t warps,
                          uint32_t dynamic_shared, uint64_t cubin_bytes,
                          double module_load_us, uint64_t module_device_bytes,
                          uint32_t warmup, uint32_t pairs,
                          uint32_t launches_per_sample, float eps,
                          const GuardedBuffer & src, const GuardedBuffer & mul,
                          GuardedBuffer & native_dst, GuardedBuffer & native_q8,
                          const std::vector<float> & native_output,
                          const std::vector<uint8_t> & native_q8_output,
                          const std::vector<float> & reference,
                          GuardedBuffer & triton_dst, GuardedBuffer & triton_q8) {
    auto phase = [&](const char * name) {
        if (trace_enabled()) {
            std::cerr << "kernel-lab-variant-w" << warps << ": " << name << "\n";
        }
    };
    VariantResult result;
    result.symbol = symbol;
    result.warps = warps;
    result.dynamic_shared_bytes = dynamic_shared;
    result.cubin_bytes = cubin_bytes;
    result.module_load_wall_us = module_load_us;
    result.module_device_bytes_delta = module_device_bytes;
    phase("query-resources");
    result.resources = query_resources(function);
    if (dynamic_shared > uint32_t(std::max(result.resources.max_dynamic_shared, 0))) {
        if (dynamic_shared > uint32_t(std::numeric_limits<int>::max())) {
            fail("Triton dynamic shared memory does not fit Driver attribute");
        }
        driver_ok(program_catalog.driver.function_set_attribute(
                      function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                      int(dynamic_shared)),
                  "set Triton max dynamic shared memory");
        result.resources = query_resources(function);
    }
    if (result.resources.binary_version != 86) fail("Triton cubin is not exact SM86");
    if (result.resources.max_threads < int(warps * 32)) {
        fail("Triton function cannot launch requested block size");
    }
    if (dynamic_shared > uint32_t(std::max(result.resources.max_dynamic_shared, 0))) {
        fail("Triton dynamic shared memory exceeds Driver function limit");
    }
    phase("cold-launch");
    result.cold_first_us = event_us([&] {
        launch_triton(function, warps, dynamic_shared,
                      src, triton_dst, mul, triton_q8, eps);
    });
    cuda_ok(cudaStreamSynchronize(g.stream), "Triton cold launch synchronize");
    phase("copy-normalized");
    const std::vector<float> triton_output =
        copy_from_device<float>(triton_dst, uint64_t(kWidth) * kTokens);
    phase("copy-q8");
    const std::vector<uint8_t> triton_q8_output =
        copy_from_device<uint8_t>(triton_q8, triton_q8.bytes);
    phase("compare-normalized");
    result.normalized_vs_native = compare_floats(triton_output, native_output);
    result.normalized_vs_reference = compare_floats(triton_output, reference);
    phase("compare-q8");
    compare_q8(triton_q8_output, native_q8_output, triton_output, &result);
    auto native_launch = [&] { launch_native(src, native_dst, mul, native_q8, eps); };
    auto triton_launch = [&] {
        launch_triton(function, warps, dynamic_shared,
                      src, triton_dst, mul, triton_q8, eps);
    };
    phase("timing");
    result.timing = interleaved_timing(
        warmup, pairs, launches_per_sample, native_launch, triton_launch);
    phase("canaries");
    result.canary_errors = src.canary_errors() + mul.canary_errors()
        + native_dst.canary_errors() + native_q8.canary_errors()
        + triton_dst.canary_errors() + triton_q8.canary_errors();
    phase("complete");
    return result;
}

uint32_t parse_u32(const char * text, const char * name) {
    char * end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > UINT32_MAX) {
        fail(std::string("invalid ") + name);
    }
    return uint32_t(value);
}

float parse_f32(const char * text, const char * name) {
    char * end = nullptr;
    const float value = std::strtof(text, &end);
    if (!text[0] || !end || *end || !std::isfinite(value) || !(value > 0.0f)) {
        fail(std::string("invalid ") + name);
    }
    return value;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 11) {
        std::cerr << "usage: kernel_lab_rms_q8 <w4.cubin> <w4.symbol> "
                     "<w8.cubin> <w8.symbol> <w4.dynamic_shared> "
                     "<w8.dynamic_shared> <eps> <warmup> <abba_baab_pairs> "
                     "<launches_per_sample>\n";
        return 2;
    }
    const uint32_t dynamic_shared_w4 = parse_u32(argv[5], "w4 dynamic shared bytes");
    const uint32_t dynamic_shared_w8 = parse_u32(argv[6], "w8 dynamic shared bytes");
    const float eps = parse_f32(argv[7], "epsilon");
    const uint32_t warmup = parse_u32(argv[8], "warmup");
    const uint32_t pairs = parse_u32(argv[9], "ABBA/BAAB pair count");
    const uint32_t launches_per_sample =
        parse_u32(argv[10], "launches per sample");
    if (!pairs) fail("ABBA/BAAB pair count must be nonzero");
    if (!launches_per_sample) fail("launches per sample must be nonzero");

    trace("initialize-context");
    if (ensure_cuda_runtime() != 0 || ensure_program_context(program_catalog) != 0) {
        fail("initialize engine primary context and stream");
    }
    trace("context-ready");
    if (g.sm_version != 86) fail("LAB-A local evidence requires exact SM86");
    size_t free_before_module = 0;
    size_t total_bytes = 0;
    cuda_ok(cudaMemGetInfo(&free_before_module, &total_bytes),
            "cudaMemGetInfo before module");

    trace("read-cubins");
    const std::vector<uint8_t> cubin_w4 = read_file(argv[1]);
    const std::vector<uint8_t> cubin_w8 = read_file(argv[3]);
    CUmodule module_w4 = nullptr;
    CUmodule module_w8 = nullptr;
    const auto module_w4_start = std::chrono::steady_clock::now();
    trace("load-w4-module");
    driver_ok(program_catalog.driver.module_load_data_ex(
                  &module_w4, cubin_w4.data(), 0, nullptr, nullptr),
              "cuModuleLoadDataEx(w4)");
    const double module_w4_load_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - module_w4_start).count();
    size_t free_after_w4 = 0;
    cuda_ok(cudaMemGetInfo(&free_after_w4, &total_bytes),
            "cudaMemGetInfo after w4 module");
    const auto module_w8_start = std::chrono::steady_clock::now();
    trace("load-w8-module");
    driver_ok(program_catalog.driver.module_load_data_ex(
                  &module_w8, cubin_w8.data(), 0, nullptr, nullptr),
              "cuModuleLoadDataEx(w8)");
    const double module_w8_load_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - module_w8_start).count();
    size_t free_after_w8 = 0;
    cuda_ok(cudaMemGetInfo(&free_after_w8, &total_bytes),
            "cudaMemGetInfo after w8 module");

    CUfunction function_w4 = nullptr;
    CUfunction function_w8 = nullptr;
    trace("resolve-functions");
    driver_ok(program_catalog.driver.module_get_function(
                  &function_w4, module_w4, argv[2]),
              "cuModuleGetFunction(w4)");
    driver_ok(program_catalog.driver.module_get_function(
                  &function_w8, module_w8, argv[4]),
              "cuModuleGetFunction(w8)");

    const uint64_t element_count = uint64_t(kWidth) * kTokens;
    const uint64_t q8_bytes = uint64_t(kWidth / 128) * kTokens
        * sizeof(BlockQ8_1Mmq);
    GuardedBuffer src;
    GuardedBuffer mul;
    GuardedBuffer native_dst;
    GuardedBuffer native_q8;
    GuardedBuffer triton_dst;
    GuardedBuffer triton_q8;
    trace("allocate-buffers");
    if (!src.allocate(element_count * sizeof(float))
        || !mul.allocate(kWidth * sizeof(float))
        || !native_dst.allocate(element_count * sizeof(float))
        || !native_q8.allocate(q8_bytes)
        || !triton_dst.allocate(element_count * sizeof(float))
        || !triton_q8.allocate(q8_bytes)) {
        fail("allocate guarded LAB-A buffers");
    }
    for (const GuardedBuffer * buffer :
         {&src, &mul, &native_dst, &native_q8, &triton_dst, &triton_q8}) {
        if ((reinterpret_cast<uintptr_t>(buffer->data) & 15u) != 0) {
            fail("LAB-A logical buffer is not 16-byte aligned");
        }
    }
    if ((reinterpret_cast<uintptr_t>(triton_q8.data + 128) & 15u) != 0) {
        fail("LAB-A q8 scale alias is not 16-byte aligned");
    }

    trace("prepare-host-inputs");
    const std::vector<float> source = make_source();
    const std::vector<float> multiplier = make_multiplier();
    const std::vector<float> reference =
        reference_normalized(source, multiplier, eps);
    trace("upload-inputs");
    cuda_ok(cudaMemcpyAsync(src.data, source.data(), src.bytes,
                           cudaMemcpyHostToDevice, g.stream), "upload source");
    cuda_ok(cudaMemcpyAsync(mul.data, multiplier.data(), mul.bytes,
                           cudaMemcpyHostToDevice, g.stream), "upload multiplier");
    cuda_ok(cudaStreamSynchronize(g.stream), "input upload synchronize");
    size_t free_after_buffers = 0;
    cuda_ok(cudaMemGetInfo(&free_after_buffers, &total_bytes),
            "cudaMemGetInfo after buffers");

    trace("native-control");
    const float native_cold_first_us = event_us([&] {
        launch_native(src, native_dst, mul, native_q8, eps);
    });
    cuda_ok(cudaStreamSynchronize(g.stream), "native cold launch synchronize");
    const std::vector<float> native_output =
        copy_from_device<float>(native_dst, element_count);
    const std::vector<uint8_t> native_q8_output =
        copy_from_device<uint8_t>(native_q8, q8_bytes);
    const Difference native_vs_reference =
        compare_floats(native_output, reference);

    trace("candidate-w4");
    VariantResult w4 = run_variant(function_w4, argv[2], 4, dynamic_shared_w4,
        cubin_w4.size(), module_w4_load_us,
        free_before_module > free_after_w4 ? free_before_module - free_after_w4 : 0,
        warmup, pairs, launches_per_sample, eps,
        src, mul, native_dst, native_q8, native_output, native_q8_output,
        reference, triton_dst, triton_q8);
    cuda_ok(cudaMemsetAsync(triton_dst.allocation, kGuard,
                           triton_dst.bytes + 2 * kGuardBytes, g.stream),
            "reset Triton dst canaries");
    cuda_ok(cudaMemsetAsync(triton_q8.allocation, kGuard,
                           triton_q8.bytes + 2 * kGuardBytes, g.stream),
            "reset Triton q8 canaries");
    cuda_ok(cudaStreamSynchronize(g.stream), "reset Triton buffers");
    trace("candidate-w8");
    VariantResult w8 = run_variant(function_w8, argv[4], 8, dynamic_shared_w8,
        cubin_w8.size(), module_w8_load_us,
        free_after_w4 > free_after_w8 ? free_after_w4 - free_after_w8 : 0,
        warmup, pairs, launches_per_sample, eps,
        src, mul, native_dst, native_q8, native_output, native_q8_output,
        reference, triton_dst, triton_q8);

    const uint64_t all_canary_errors = w4.canary_errors + w8.canary_errors;
    const bool structural_ok = all_canary_errors == 0
        && native_vs_reference.non_finite == 0
        && w4.normalized_vs_native.non_finite == 0
        && w8.normalized_vs_native.non_finite == 0
        && w4.q8_scales_vs_native.non_finite == 0
        && w8.q8_scales_vs_native.non_finite == 0
        && w4.q8_dequant.non_finite == 0
        && w8.q8_dequant.non_finite == 0;

    trace("emit-result");
    std::cout << std::setprecision(10)
              << "{\"schema\":1,\"phase\":\"A2\",\"production_enabled\":false"
              << ",\"same_primary_context\":true,\"same_stream\":true"
              << ",\"separate_outputs\":true,\"timing_clock\":\"cuda-events\""
              << ",\"interleaved_schedule\":\"abba-baab\""
              << ",\"launches_per_sample\":" << launches_per_sample
              << ",\"target_sm\":86,\"width\":" << kWidth
              << ",\"n_tok\":" << kTokens
              << ",\"eps\":" << eps
              << ",\"buffers_device_bytes_delta\":"
              << (free_after_w8 > free_after_buffers
                      ? free_after_w8 - free_after_buffers : 0)
              << ",\"native\":{\"symbol\":\"k_rms_norm_q8_1_mmq<1024>\""
              << ",\"cold_first_us\":" << native_cold_first_us
              << ",\"normalized_vs_reference\":";
    emit_difference(native_vs_reference);
    std::cout << "},\"variants\":[";
    emit_variant(w4);
    std::cout << ',';
    emit_variant(w8);
    std::cout << "],\"all_canary_errors\":" << all_canary_errors
              << ",\"structural_ok\":" << (structural_ok ? "true" : "false")
              << "}\n";

    trace("cleanup");
    cuda_ok(cudaStreamSynchronize(g.stream), "final synchronize");
    driver_ok(program_catalog.driver.module_unload(module_w8), "cuModuleUnload(w8)");
    driver_ok(program_catalog.driver.module_unload(module_w4), "cuModuleUnload(w4)");
    triton_q8.release();
    triton_dst.release();
    native_q8.release();
    native_dst.release();
    mul.release();
    src.release();
    if (g.stream) {
        cuda_ok(cudaStreamDestroy(g.stream), "cudaStreamDestroy");
        g.stream = nullptr;
        g.runtime_initialized = false;
    }
    trace("complete");
    return structural_ok ? 0 : 3;
}
