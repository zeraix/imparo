// Phase A2 only: standalone, deletable Q4_0 x Q8_1 MMQ comparison harness.
//
// This translation unit includes the native backend so the control launch is the
// accepted production SM86 FullTile route. It is never linked into the backend
// library and must be compiled explicitly with IMPARO_CUDA_KERNEL_LAB=1.

#if !defined(IMPARO_CUDA_KERNEL_LAB) || IMPARO_CUDA_KERNEL_LAB != 1
#error "kernel_lab_q4_q8.cu is test-only; define IMPARO_CUDA_KERNEL_LAB=1"
#endif

#include "../imparo_cuda.cu"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kEvidenceTokens = 512;
constexpr uint32_t kStridePadding = 17;
constexpr uint32_t kExpectedSmCount = 30;
constexpr uint32_t kBaselineTritonBm = 64;
constexpr uint32_t kBaselineTritonBn = 64;
constexpr uint32_t kBaselineTritonThreads = 128;
constexpr uint32_t kBaselineTritonSharedBytes = 16384;
constexpr size_t kGuardBytes = 256;
constexpr uint8_t kGuard = 0xa5;
constexpr uint64_t kExpectedWorkspaceBytes = 1966080;

struct Shape {
    const char * id;
    uint32_t n_in;
    uint32_t n_out;
    uint32_t expected_physical;
    uint32_t expected_efficiency;
    const char * cubin;
    const char * symbol;
    uint32_t triton_bm;
    uint32_t triton_bn;
    uint32_t triton_threads;
    uint32_t triton_dynamic_shared_bytes;
};

struct GuardedBuffer {
    uint8_t * allocation = nullptr;
    uint8_t * data = nullptr;
    size_t bytes = 0;

    void allocate(size_t requested) {
        bytes = requested;
        cudaError_t rc = cudaMalloc(reinterpret_cast<void **>(&allocation),
                                    bytes + 2 * kGuardBytes);
        if (rc != cudaSuccess) {
            std::cerr << "kernel-lab: cudaMalloc: " << cudaGetErrorString(rc) << "\n";
            std::exit(2);
        }
        data = allocation + kGuardBytes;
        rc = cudaMemsetAsync(allocation, kGuard, bytes + 2 * kGuardBytes, g.stream);
        if (rc != cudaSuccess) {
            std::cerr << "kernel-lab: initialize canaries: "
                      << cudaGetErrorString(rc) << "\n";
            std::exit(2);
        }
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
    double max_normalized_rel = 0.0;
    double sum_sq = 0.0;
    uint64_t finite = 0;
    uint64_t non_finite = 0;
    uint64_t bitwise_different = 0;

    void add(float actual, float expected) {
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++non_finite;
            return;
        }
        const double absolute = std::fabs(double(actual) - double(expected));
        const double relative = absolute / std::max(std::fabs(double(expected)), 1.0e-30);
        const double normalized = absolute / std::max(std::fabs(double(expected)), 1.0);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, relative);
        max_normalized_rel = std::max(max_normalized_rel, normalized);
        sum_sq += absolute * absolute;
        ++finite;
        uint32_t a = 0;
        uint32_t b = 0;
        std::memcpy(&a, &actual, sizeof(a));
        std::memcpy(&b, &expected, sizeof(b));
        bitwise_different += a != b;
    }

    double rms() const { return finite ? std::sqrt(sum_sq / double(finite)) : 0.0; }
};

struct ReferenceDifference {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double max_normalized_rel = 0.0;
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
        const double normalized = absolute / std::max(std::fabs(expected), 1.0);
        max_abs = std::max(max_abs, absolute);
        max_rel = std::max(max_rel, relative);
        max_normalized_rel = std::max(max_normalized_rel, normalized);
        sum_sq += absolute * absolute;
        ++count;
    }

    double rms() const { return count ? std::sqrt(sum_sq / double(count)) : 0.0; }
};

struct OracleEvidence {
    bool present = false;
    uint32_t samples = 0;
    uint32_t seam_104_samples = 0;
    uint32_t seam_208_samples = 0;
    uint32_t no_seam_samples = 0;
    uint64_t wrong_grid_bitwise_different = 0;
    uint64_t seam_minus_one_bitwise_different = 0;
    uint64_t seam_plus_one_bitwise_different = 0;
    bool wrong_grid_gpu_launched = false;
    uint64_t wrong_grid_gpu_canary_errors = 0;
    uint64_t wrong_grid_gpu_input_mismatches = 0;
    uint64_t wrong_grid_gpu_padding_errors = 0;
    Difference native_vs_strict_float;
    Difference triton_vs_strict_float;
    Difference wrong_grid_gpu_vs_correct;
    Difference wrong_grid_gpu_vs_native;
    ReferenceDifference native_vs_f64;
    ReferenceDifference triton_vs_f64;
};

struct FunctionResources {
    int registers_per_thread = -1;
    int static_shared_bytes = -1;
    int local_bytes = -1;
    int max_threads_per_block = -1;
    int max_dynamic_shared_bytes = -1;
    int binary_version = -1;
    uint32_t launch_threads = 0;
    uint32_t launch_dynamic_shared_bytes = 0;
    uint32_t block_m = 0;
    uint32_t block_n = 0;
};

struct TimingEvidence {
    bool present = false;
    uint32_t warmup = 0;
    uint32_t pairs = 0;
    uint32_t launches_per_sample = 0;
    float native_first_launch_us = 0.0f;
    float triton_first_launch_us = 0.0f;
    std::vector<float> native_us;
    std::vector<float> triton_us;
};

struct ArtifactEvidence {
    bool present = false;
    uint64_t cubin_bytes = 0;
    double module_load_wall_us = 0.0;
    uint64_t free_before_buffers = 0;
    uint64_t free_after_buffers = 0;
    uint64_t free_before_module = 0;
    uint64_t free_after_module = 0;
    uint64_t free_after_first_launches = 0;
    uint64_t free_after_timing = 0;
    uint64_t total_device_bytes = 0;
    uint64_t observed_buffer_delta = 0;
    uint64_t observed_module_delta = 0;
    uint64_t observed_first_launch_delta = 0;
    uint64_t observed_timing_delta = 0;
    uint64_t observed_peak_delta = 0;
    uint64_t weights_logical_bytes = 0;
    uint64_t q8_logical_bytes = 0;
    uint64_t output_logical_bytes = 0;
    uint64_t workspace_logical_bytes = 0;
    uint64_t guarded_allocation_bytes = 0;
    FunctionResources native_resources{};
    FunctionResources triton_resources{};
};

struct PostTimingEvidence {
    bool present = false;
    Difference comparison{};
    uint64_t native_input_mismatches = 0;
    uint64_t triton_input_mismatches = 0;
    uint64_t native_padding_errors = 0;
    uint64_t triton_padding_errors = 0;
    uint64_t canary_errors = 0;
};

struct ShapeResult {
    Shape shape{};
    uint32_t n_tok = 0;
    uint32_t out_stride = 0;
    uint32_t numeric_stream_grid = 0;
    imparo_sm80_mmq::LaunchInfo launch{};
    Difference difference{};
    uint64_t canary_errors = 0;
    uint64_t native_input_mismatches = 0;
    uint64_t triton_input_mismatches = 0;
    uint64_t native_padding_errors = 0;
    uint64_t triton_padding_errors = 0;
    OracleEvidence oracle{};
    TimingEvidence timing{};
    ArtifactEvidence artifact{};
    PostTimingEvidence post_timing{};
};

bool has_numeric_seams(const imparo_sm80_mmq::LaunchInfo & info) {
    return (info.route == imparo_sm80_mmq::LaunchRoute::FullTile
                && info.efficiency < 90)
        || (info.route == imparo_sm80_mmq::LaunchRoute::PhysicalStreamK
            && info.physical_blocks && info.logical_tiles % info.physical_blocks != 0);
}

[[noreturn]] void fail(const std::string & message) {
    std::cerr << "kernel-lab: " << message << "\n";
    std::exit(2);
}

uint32_t parse_u32(const char * text, const char * name) {
    try {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed, 10);
        if (text[consumed] != '\0' || value == 0
            || value > std::numeric_limits<uint32_t>::max()) {
            fail(std::string("invalid ") + name);
        }
        return uint32_t(value);
    } catch (...) {
        fail(std::string("invalid ") + name);
    }
}

uint32_t parse_u32_allow_zero(const char * text, const char * name) {
    try {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed, 10);
        if (text[consumed] != '\0'
            || value > std::numeric_limits<uint32_t>::max()) {
            fail(std::string("invalid ") + name);
        }
        return uint32_t(value);
    } catch (...) {
        fail(std::string("invalid ") + name);
    }
}

std::string uuid_hex(const cudaUUID_t & uuid) {
    std::ostringstream value;
    value << std::hex << std::setfill('0');
    for (unsigned char byte : uuid.bytes) {
        value << std::setw(2) << unsigned(byte);
    }
    return value.str();
}

bool trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("IMPARO_CUDA_KERNEL_LAB_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void trace(const std::string & message) {
    if (trace_enabled()) std::cerr << "kernel-lab-q4: " << message << "\n";
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

template <typename Launch>
float event_us(Launch && launch) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_ok(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_ok(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    cuda_ok(cudaEventRecord(start, g.stream), "cudaEventRecord(start)");
    launch();
    cuda_ok(cudaEventRecord(stop, g.stream), "cudaEventRecord(stop)");
    cuda_ok(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float milliseconds = 0.0f;
    cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop),
            "cudaEventElapsedTime");
    cuda_ok(cudaEventDestroy(start), "cudaEventDestroy(start)");
    cuda_ok(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return milliseconds * 1000.0f;
}

double median(std::vector<float> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() & 1 ? values[middle]
        : 0.5 * (values[middle - 1] + values[middle]);
}

double mean(const std::vector<float> & values) {
    return values.empty() ? 0.0
        : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
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

double paired_mad_fraction(const TimingEvidence & value) {
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
    std::vector<float> deviations;
    deviations.reserve(ratios.size());
    for (float ratio : ratios) {
        deviations.push_back(float(std::fabs(double(ratio) - center)));
    }
    return center > 0.0 ? median(deviations) / center
                        : std::numeric_limits<double>::infinity();
}

std::vector<uint8_t> read_file(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) fail(std::string("cannot open cubin: ") + path);
    const std::streamsize size = input.tellg();
    if (size <= 0) fail("cubin is empty");
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char *>(bytes.data()), size)) {
        fail("cannot read complete cubin");
    }
    return bytes;
}

std::vector<uint8_t> make_weights(uint32_t n_in, uint32_t n_out) {
    const uint64_t blocks = n_in / 32;
    std::vector<uint8_t> result(uint64_t(n_out) * blocks * 18);
    for (uint32_t row = 0; row < n_out; ++row) {
        for (uint32_t block = 0; block < blocks; ++block) {
            uint8_t * record = result.data() + (uint64_t(row) * blocks + block) * 18;
            const float scale = 0.0137f * float(1 + ((row * 5 + block * 3) % 7));
            const __half half_scale = __float2half_rn(scale);
            std::memcpy(record, &half_scale, sizeof(half_scale));
            for (uint32_t lane = 0; lane < 16; ++lane) {
                const uint8_t low = uint8_t((row * 3 + block * 5 + lane * 7) & 15);
                const uint8_t high = uint8_t((row * 11 + block * 13 + lane * 3 + 1) & 15);
                record[2 + lane] = uint8_t(low | (high << 4));
            }
        }
    }
    // Explicit first/last record boundary vectors.
    result[2] = 0xf0;
    result[result.size() - 1] = 0x0f;
    return result;
}

std::vector<BlockQ8_1Mmq> make_activations(uint32_t n_in, uint32_t n_tok) {
    const uint64_t groups = n_in / 128;
    std::vector<BlockQ8_1Mmq> result(groups * n_tok);
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t token = 0; token < n_tok; ++token) {
            BlockQ8_1Mmq & record = result[uint64_t(group) * n_tok + token];
            for (uint32_t lane = 0; lane < 128; ++lane) {
                record.qs[lane] = int8_t(int32_t((group * 17 + token * 7 + lane * 11) % 127) - 63);
            }
            for (uint32_t block = 0; block < 4; ++block) {
                const float raw = 0.0069f * float(1 + ((group * 3 + token + block * 5) % 9));
                record.d[block] = __half2float(__float2half_rn(raw));
            }
        }
    }
    result.front().qs[0] = -127;
    result.front().qs[127] = 126;
    result.back().qs[0] = 125;
    result.back().qs[127] = -126;
    return result;
}

template <typename T>
std::vector<T> copy_from_device(const GuardedBuffer & buffer, size_t count) {
    std::vector<T> result(count);
    cuda_ok(cudaMemcpy(result.data(), buffer.data, count * sizeof(T),
                       cudaMemcpyDeviceToHost), "copy output from device");
    return result;
}

Difference compare(const std::vector<float> & actual,
                   const std::vector<float> & expected,
                   uint32_t n_out, uint32_t n_tok, uint32_t out_stride) {
    if (actual.size() != uint64_t(n_tok) * out_stride
        || expected.size() != actual.size()) {
        fail("output comparison size mismatch");
    }
    Difference result;
    for (uint32_t token = 0; token < n_tok; ++token) {
        for (uint32_t row = 0; row < n_out; ++row) {
            const uint64_t index = uint64_t(token) * out_stride + row;
            result.add(actual[index], expected[index]);
        }
    }
    return result;
}

uint64_t byte_mismatches(const GuardedBuffer & buffer,
                         const void * expected, size_t bytes) {
    if (buffer.bytes != bytes) fail("input immutability size mismatch");
    std::vector<uint8_t> actual(bytes);
    cuda_ok(cudaMemcpy(actual.data(), buffer.data, bytes, cudaMemcpyDeviceToHost),
            "copy immutable input from device");
    const uint8_t * expected_bytes = static_cast<const uint8_t *>(expected);
    uint64_t mismatches = 0;
    for (size_t index = 0; index < bytes; ++index) {
        mismatches += actual[index] != expected_bytes[index];
    }
    return mismatches;
}

uint64_t padding_errors(const std::vector<float> & output,
                        uint32_t n_out, uint32_t n_tok, uint32_t out_stride) {
    constexpr uint32_t sentinel = 0xcdcdcdcd;
    uint64_t errors = 0;
    for (uint32_t token = 0; token < n_tok; ++token) {
        for (uint32_t row = n_out; row < out_stride; ++row) {
            uint32_t bits = 0;
            std::memcpy(&bits, &output[uint64_t(token) * out_stride + row],
                        sizeof(bits));
            errors += bits != sentinel;
        }
    }
    return errors;
}

struct OraclePoint {
    uint32_t row;
    uint32_t token;
};

uint32_t numeric_seam_for(uint32_t row, uint32_t token,
                          uint32_t n_out, uint32_t n_tok, uint32_t blocks,
                          uint32_t numeric_stream_grid, bool numeric_seams) {
    if (!numeric_seams || !numeric_stream_grid) return 0;
    const uint32_t token_tiles = (n_tok + 127) / 128;
    const uint32_t tile = (row / 128) * token_tiles + token / 128;
    const uint32_t logical_tiles = ((n_out + 127) / 128) * token_tiles;
    uint32_t worker = uint32_t(
        (uint64_t(tile) * numeric_stream_grid + logical_tiles - 1)
        / logical_tiles);
    worker = std::max(worker, 1u);
    if (worker >= numeric_stream_grid) return 0;
    const uint64_t total_work = uint64_t(logical_tiles) * blocks;
    uint64_t boundary = uint64_t(worker) * total_work / numeric_stream_grid;
    boundary -= (boundary % blocks) % 8;
    return boundary / blocks == tile && boundary % blocks
        ? uint32_t(boundary % blocks) : 0;
}

int32_t block_dot(const std::vector<uint8_t> & weights,
                  const std::vector<BlockQ8_1Mmq> & q8,
                  uint32_t n_in, uint32_t n_tok, uint32_t row,
                  uint32_t token, uint32_t block) {
    const uint32_t blocks = n_in / 32;
    const uint8_t * weight = weights.data()
        + (uint64_t(row) * blocks + block) * 18;
    const BlockQ8_1Mmq & activation =
        q8[uint64_t(block / 4) * n_tok + token];
    int32_t dot = 0;
    for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint8_t packed = weight[2 + (lane & 15)];
        const int32_t q4 = int32_t(lane < 16 ? packed & 15 : packed >> 4) - 8;
        dot += q4 * int32_t(activation.qs[(block & 3) * 32 + lane]);
    }
    return dot;
}

float weight_scale(const std::vector<uint8_t> & weights,
                   uint32_t n_in, uint32_t row, uint32_t block) {
    const uint32_t blocks = n_in / 32;
    __half value{};
    std::memcpy(&value, weights.data()
        + (uint64_t(row) * blocks + block) * 18, sizeof(value));
    return __half2float(value);
}

float strict_segment(const std::vector<uint8_t> & weights,
                     const std::vector<BlockQ8_1Mmq> & q8,
                     uint32_t n_in, uint32_t n_tok, uint32_t row,
                     uint32_t token, uint32_t begin, uint32_t end) {
    float sum = 0.0f;
    for (uint32_t block = begin; block < end; ++block) {
        const BlockQ8_1Mmq & activation =
            q8[uint64_t(block / 4) * n_tok + token];
        const float scaled = float(block_dot(
            weights, q8, n_in, n_tok, row, token, block))
            * weight_scale(weights, n_in, row, block);
        sum = std::fma(scaled, activation.d[block & 3], sum);
    }
    return sum;
}

float strict_value_at_seam(const std::vector<uint8_t> & weights,
                           const std::vector<BlockQ8_1Mmq> & q8,
                           uint32_t n_in, uint32_t n_tok,
                           uint32_t row, uint32_t token, uint32_t seam) {
    const uint32_t blocks = n_in / 32;
    if (!seam) {
        return strict_segment(weights, q8, n_in, n_tok, row, token, 0, blocks);
    }
    if (seam >= blocks) fail("CPU oracle seam lies outside K blocks");
    const float prefix = strict_segment(
        weights, q8, n_in, n_tok, row, token, 0, seam);
    const float suffix = strict_segment(
        weights, q8, n_in, n_tok, row, token, seam, blocks);
    return suffix + prefix;
}

float strict_value(const std::vector<uint8_t> & weights,
                   const std::vector<BlockQ8_1Mmq> & q8,
                   uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                   uint32_t row, uint32_t token,
                   uint32_t numeric_stream_grid, bool numeric_seams,
                   uint32_t * seam_out = nullptr) {
    const uint32_t blocks = n_in / 32;
    const uint32_t seam = numeric_seam_for(row, token, n_out, n_tok, blocks,
                                           numeric_stream_grid, numeric_seams);
    if (seam_out) *seam_out = seam;
    return strict_value_at_seam(
        weights, q8, n_in, n_tok, row, token, seam);
}

double f64_value(const std::vector<uint8_t> & weights,
                 const std::vector<BlockQ8_1Mmq> & q8,
                 uint32_t n_in, uint32_t n_tok,
                 uint32_t row, uint32_t token) {
    const uint32_t blocks = n_in / 32;
    double sum = 0.0;
    for (uint32_t block = 0; block < blocks; ++block) {
        const BlockQ8_1Mmq & activation =
            q8[uint64_t(block / 4) * n_tok + token];
        sum += double(block_dot(weights, q8, n_in, n_tok, row, token, block))
            * double(weight_scale(weights, n_in, row, block))
            * double(activation.d[block & 3]);
    }
    return sum;
}

OracleEvidence build_oracle(const Shape & shape, uint32_t n_tok,
                            uint32_t out_stride, uint32_t numeric_stream_grid,
                            bool numeric_seams,
                            const std::vector<uint8_t> & weights,
                            const std::vector<BlockQ8_1Mmq> & q8,
                            const std::vector<float> & native,
                            const std::vector<float> & triton) {
    OracleEvidence result;
    if (n_tok != kEvidenceTokens) return result;
    result.present = true;
    std::vector<OraclePoint> points;
    if (shape.n_in == 10240 && shape.n_out == 2560) {
        // Tile 2 has seam 208, tile 5 has seam 104. Tiles 1/3 and 4/6 are
        // their adjacent no-seam controls under the generic runtime formula.
        constexpr uint32_t tiles[] = {1, 2, 3, 4, 5, 6};
        for (uint32_t tile : tiles) {
            points.push_back({(tile / 4) * 128 + (tile * 17) % 128,
                              (tile % 4) * 128 + (tile * 23) % 128});
        }
        points.push_back({0, 256});
        points.push_back({128, 128});
    } else {
        points = {{0, 0}, {127, 127}, {128, 128},
                  {shape.n_out - 1, n_tok - 1}};
    }
    for (const OraclePoint & point : points) {
        uint32_t seam = 0;
        const float strict = strict_value(
            weights, q8, shape.n_in, shape.n_out, n_tok,
            point.row, point.token, numeric_stream_grid, numeric_seams, &seam);
        const double reference = f64_value(
            weights, q8, shape.n_in, n_tok, point.row, point.token);
        const uint64_t index = uint64_t(point.token) * out_stride + point.row;
        result.native_vs_strict_float.add(native[index], strict);
        result.triton_vs_strict_float.add(triton[index], strict);
        result.native_vs_f64.add(native[index], reference);
        result.triton_vs_f64.add(triton[index], reference);
        ++result.samples;
        result.seam_104_samples += seam == 104;
        result.seam_208_samples += seam == 208;
        result.no_seam_samples += seam == 0;
        if (seam == 104 || seam == 208) {
            const float minus_one = strict_value_at_seam(
                weights, q8, shape.n_in, n_tok,
                point.row, point.token, seam - 1);
            const float plus_one = strict_value_at_seam(
                weights, q8, shape.n_in, n_tok,
                point.row, point.token, seam + 1);
            uint32_t strict_bits = 0;
            uint32_t minus_bits = 0;
            uint32_t plus_bits = 0;
            std::memcpy(&strict_bits, &strict, sizeof(strict_bits));
            std::memcpy(&minus_bits, &minus_one, sizeof(minus_bits));
            std::memcpy(&plus_bits, &plus_one, sizeof(plus_bits));
            result.seam_minus_one_bitwise_different += strict_bits != minus_bits;
            result.seam_plus_one_bitwise_different += strict_bits != plus_bits;
        }
        if (shape.n_in == 10240) {
            const float wrong_grid = strict_value(
                weights, q8, shape.n_in, shape.n_out, n_tok,
                point.row, point.token, 80, true);
            uint32_t correct_bits = 0;
            uint32_t wrong_bits = 0;
            std::memcpy(&correct_bits, &strict, sizeof(correct_bits));
            std::memcpy(&wrong_bits, &wrong_grid, sizeof(wrong_bits));
            result.wrong_grid_bitwise_different += correct_bits != wrong_bits;
        }
    }
    return result;
}

void launch_triton(CUfunction function, const Shape & shape,
                   uint32_t n_tok_value, uint32_t out_stride_value,
                   uint32_t numeric_stream_grid_value,
                   GuardedBuffer & weights, GuardedBuffer & q8,
                   GuardedBuffer & output) {
    void * w_qs = weights.data + 2;
    void * w_d = weights.data;
    void * x_qs = q8.data;
    void * x_d = q8.data + 128;
    void * y = output.data;
    int32_t n_tok = int32_t(n_tok_value);
    int32_t out_stride = int32_t(out_stride_value);
    int32_t numeric_stream_grid = int32_t(numeric_stream_grid_value);
    CUdeviceptr global_scratch = 0;
    CUdeviceptr profile_scratch = 0;
    void * arguments[] = {
        &w_qs, &w_d, &x_qs, &x_d, &y, &n_tok, &out_stride,
        &numeric_stream_grid, &global_scratch, &profile_scratch,
    };
    driver_ok(program_catalog.driver.launch_kernel(
                  function,
                  (shape.n_out + shape.triton_bm - 1) / shape.triton_bm,
                  (n_tok_value + shape.triton_bn - 1) / shape.triton_bn, 1,
                  shape.triton_threads, 1, 1,
                  shape.triton_dynamic_shared_bytes,
                  reinterpret_cast<CUstream>(g.stream), arguments, nullptr),
              "Triton Q4-Q8 launch");
}

bool launch_native_once(const Shape & shape, uint32_t n_tok,
                        uint32_t out_stride, GuardedBuffer & weights,
                        GuardedBuffer & q8, GuardedBuffer & output,
                        GuardedBuffer & workspace,
                        imparo_sm80_mmq::LaunchInfo * info) {
    const auto variant = imparo_sm80_mmq::select_full_tile_variant(86, 0);
    const uint32_t min_efficiency =
        imparo_sm80_mmq::select_full_tile_min_efficiency(86, 0);
    return imparo_sm80_mmq::launch(
        weights.data, reinterpret_cast<const BlockQ8_1Mmq *>(q8.data),
        reinterpret_cast<float *>(output.data), shape.n_in, shape.n_out,
        n_tok, 0, out_stride, 0, uint32_t(g.sm_count), 1, 0, 0, 1, 1,
        variant, min_efficiency, reinterpret_cast<float *>(workspace.data),
        nullptr, g.stream, info);
}

FunctionResources query_triton_resources(CUfunction function,
                                         const Shape & shape) {
    FunctionResources result;
    auto get = [&](CUfunction_attribute kind, int * value, const char * name) {
        driver_ok(program_catalog.driver.function_get_attribute(
                      value, kind, function), name);
    };
    get(CU_FUNC_ATTRIBUTE_NUM_REGS, &result.registers_per_thread,
        "query Triton registers");
    get(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, &result.static_shared_bytes,
        "query Triton static shared");
    get(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, &result.local_bytes,
        "query Triton local memory");
    get(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &result.max_threads_per_block,
        "query Triton max threads");
    get(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        &result.max_dynamic_shared_bytes, "query Triton max dynamic shared");
    get(CU_FUNC_ATTRIBUTE_BINARY_VERSION, &result.binary_version,
        "query Triton binary version");
    result.launch_threads = shape.triton_threads;
    result.launch_dynamic_shared_bytes =
        shape.triton_dynamic_shared_bytes;
    result.block_m = shape.triton_bm;
    result.block_n = shape.triton_bn;
    return result;
}

FunctionResources query_native_resources(const Shape & shape) {
    cudaFuncAttributes attributes{};
    cudaError_t rc = cudaSuccess;
    if (shape.n_in == 2560) {
        rc = cudaFuncGetAttributes(&attributes,
            imparo_sm80_mmq::q4_q8_1_full_tile<false, 4>);
    } else {
        rc = cudaFuncGetAttributes(&attributes,
            imparo_sm80_mmq::q4_q8_1_full_tile<false, 4, false, true, true>);
    }
    cuda_ok(rc, "query native Q4-Q8 resources");
    FunctionResources result;
    result.registers_per_thread = attributes.numRegs;
    result.static_shared_bytes = int(attributes.sharedSizeBytes);
    result.local_bytes = int(attributes.localSizeBytes);
    result.max_threads_per_block = attributes.maxThreadsPerBlock;
    result.max_dynamic_shared_bytes = int(attributes.maxDynamicSharedSizeBytes);
    result.binary_version = attributes.binaryVersion;
    result.launch_threads = 256;
    result.launch_dynamic_shared_bytes =
        uint32_t(imparo_sm80_mmq::kHalfKSharedBytes);
    return result;
}

template <typename NativeLaunch, typename TritonLaunch>
TimingEvidence measure_interleaved(uint32_t warmup, uint32_t pairs,
                                   uint32_t launches_per_sample,
                                   NativeLaunch && native,
                                   TritonLaunch && triton) {
    TimingEvidence result;
    result.present = true;
    result.warmup = warmup;
    result.pairs = pairs;
    result.launches_per_sample = launches_per_sample;
    for (uint32_t index = 0; index < warmup; ++index) {
        native();
        triton();
    }
    cuda_ok(cudaStreamSynchronize(g.stream), "timing warmup synchronize");
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

ShapeResult run_case(const Shape & shape, uint32_t n_tok, uint32_t out_stride,
                     uint32_t warmup, uint32_t pairs,
                     uint32_t launches_per_sample) {
    if (!shape.triton_bm || !shape.triton_bn || !shape.triton_threads
        || shape.triton_threads > 1024) {
        fail("invalid per-shape Triton launch configuration");
    }
    std::ostringstream case_name;
    case_name << shape.id << " n_tok=" << n_tok << " stride=" << out_stride;
    trace(std::string("begin ") + case_name.str());
    const uint64_t weight_bytes = uint64_t(shape.n_out) * (shape.n_in / 32) * 18;
    const uint64_t q8_bytes = uint64_t(shape.n_in / 128) * n_tok
        * sizeof(BlockQ8_1Mmq);
    const uint64_t output_count = uint64_t(out_stride) * n_tok;
    const uint64_t workspace_bytes = imparo_sm80_mmq::stream_workspace_bytes(
        shape.n_out, n_tok, uint32_t(g.sm_count), 0);
    if (workspace_bytes != kExpectedWorkspaceBytes) {
        fail("native workspace contract is not 1,966,080 bytes");
    }
    size_t free_before_buffers = 0;
    size_t total_device_bytes = 0;
    cuda_ok(cudaMemGetInfo(&free_before_buffers, &total_device_bytes),
            "cudaMemGetInfo before buffers");

    GuardedBuffer weights;
    GuardedBuffer q8;
    GuardedBuffer native_output;
    GuardedBuffer triton_output;
    GuardedBuffer workspace;
    weights.allocate(weight_bytes);
    q8.allocate(q8_bytes);
    native_output.allocate(output_count * sizeof(float));
    triton_output.allocate(output_count * sizeof(float));
    workspace.allocate(workspace_bytes);
    if ((reinterpret_cast<uintptr_t>(weights.data) & 15u) != 0
        || (reinterpret_cast<uintptr_t>(weights.data + 2) & 1u) != 0
        || (reinterpret_cast<uintptr_t>(q8.data) & 15u) != 0
        || (reinterpret_cast<uintptr_t>(q8.data + 128) & 15u) != 0
        || (reinterpret_cast<uintptr_t>(native_output.data) & 15u) != 0
        || (reinterpret_cast<uintptr_t>(triton_output.data) & 15u) != 0) {
        fail("LAB-A Q4/Q8 alias alignment contract failed");
    }

    const std::vector<uint8_t> host_weights = make_weights(shape.n_in, shape.n_out);
    const std::vector<BlockQ8_1Mmq> host_q8 = make_activations(shape.n_in, n_tok);
    cuda_ok(cudaMemcpyAsync(weights.data, host_weights.data(), weight_bytes,
                           cudaMemcpyHostToDevice, g.stream), "upload Q4 weights");
    cuda_ok(cudaMemcpyAsync(q8.data, host_q8.data(), q8_bytes,
                           cudaMemcpyHostToDevice, g.stream), "upload Q8 activations");
    cuda_ok(cudaMemsetAsync(native_output.data, 0xcd, native_output.bytes, g.stream),
            "initialize native output sentinel");
    cuda_ok(cudaMemsetAsync(triton_output.data, 0xcd, triton_output.bytes, g.stream),
            "initialize Triton output sentinel");
    cuda_ok(cudaMemsetAsync(workspace.data, 0, workspace.bytes, g.stream),
            "clear native workspace");
    cuda_ok(cudaStreamSynchronize(g.stream), "input upload synchronize");
    size_t free_after_buffers = 0;
    cuda_ok(cudaMemGetInfo(&free_after_buffers, &total_device_bytes),
            "cudaMemGetInfo after buffers");

    imparo_sm80_mmq::LaunchInfo info;
    const auto variant = imparo_sm80_mmq::select_full_tile_variant(86, 0);
    const uint32_t min_efficiency =
        imparo_sm80_mmq::select_full_tile_min_efficiency(86, 0);
    if (variant != imparo_sm80_mmq::FullTileVariant::Rows128K128
        || min_efficiency != 50) {
        fail("SM86 automatic FullTile policy is not Rows128K128/min50");
    }
    float native_first_launch_us = 0.0f;
    bool accepted = false;
    auto first_native_launch = [&] {
        accepted = launch_native_once(shape, n_tok, out_stride, weights, q8,
                                      native_output, workspace, &info);
    };
    if (n_tok == kEvidenceTokens) {
        native_first_launch_us = event_us(first_native_launch);
    } else {
        first_native_launch();
    }
    if (!accepted) fail("native Q4-Q8 launch was rejected");
    const uint32_t logical_tiles = ((shape.n_out + 127) / 128)
        * ((n_tok + 127) / 128);
    const uint32_t waves = (logical_tiles + uint32_t(g.sm_count) - 1)
        / uint32_t(g.sm_count);
    const uint32_t expected_efficiency = 100 * logical_tiles
        / (uint32_t(g.sm_count) * waves);
    const uint32_t expected_physical = expected_efficiency >= 90
        ? logical_tiles : uint32_t(g.sm_count);
    const auto expected_route = n_tok == kEvidenceTokens
        ? imparo_sm80_mmq::LaunchRoute::FullTile
        : imparo_sm80_mmq::LaunchRoute::PhysicalStreamK;
    const uint32_t route_physical = n_tok == kEvidenceTokens
        ? shape.expected_physical : expected_physical;
    const uint32_t route_efficiency = n_tok == kEvidenceTokens
        ? shape.expected_efficiency : expected_efficiency;
    if (info.route != expected_route
        || info.tile_rows != 128 || info.tile_tokens != 128
        || info.logical_tiles != logical_tiles
        || info.physical_blocks != route_physical
        || info.efficiency != route_efficiency || info.fused_q8) {
        std::ostringstream out;
        out << "unexpected native route="
            << imparo_sm80_mmq::launch_route_name(info.route)
            << " tile=" << info.tile_rows << 'x' << info.tile_tokens
            << " logical=" << info.logical_tiles
            << " physical=" << info.physical_blocks
            << " efficiency=" << info.efficiency
            << " fused=" << info.fused_q8;
        fail(out.str());
    }
    cuda_ok(cudaStreamSynchronize(g.stream), "native launch synchronize");
    const std::vector<float> native =
        copy_from_device<float>(native_output, output_count);
    const uint64_t native_input_mismatches =
        byte_mismatches(weights, host_weights.data(), host_weights.size())
        + byte_mismatches(q8, host_q8.data(), host_q8.size() * sizeof(BlockQ8_1Mmq));
    const uint64_t native_padding_errors =
        padding_errors(native, shape.n_out, n_tok, out_stride);

    const std::vector<uint8_t> cubin = read_file(shape.cubin);
    size_t free_before_module = 0;
    cuda_ok(cudaMemGetInfo(&free_before_module, &total_device_bytes),
            "cudaMemGetInfo before module");
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    const auto module_load_start = std::chrono::steady_clock::now();
    driver_ok(program_catalog.driver.module_load_data_ex(
                  &module, cubin.data(), 0, nullptr, nullptr),
              "cuModuleLoadDataEx(Q4-Q8)");
    const double module_load_wall_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - module_load_start).count();
    size_t free_after_module = 0;
    cuda_ok(cudaMemGetInfo(&free_after_module, &total_device_bytes),
            "cudaMemGetInfo after module");
    driver_ok(program_catalog.driver.module_get_function(
                  &function, module, shape.symbol),
              "cuModuleGetFunction(Q4-Q8)");
    int binary_version = 0;
    driver_ok(program_catalog.driver.function_get_attribute(
                  &binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, function),
              "query Q4-Q8 binary version");
    if (binary_version != 86) fail("Q4-Q8 cubin is not exact SM86");
    int max_dynamic = 0;
    driver_ok(program_catalog.driver.function_get_attribute(
                  &max_dynamic, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                  function), "query Q4-Q8 dynamic shared limit");
    if (shape.triton_dynamic_shared_bytes
        > uint32_t(std::max(max_dynamic, 0))) {
        driver_ok(program_catalog.driver.function_set_attribute(
                      function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                      int(shape.triton_dynamic_shared_bytes)),
                  "set Q4-Q8 dynamic shared limit");
    }
    const FunctionResources triton_function_resources =
        query_triton_resources(function, shape);
    if (shape.triton_threads
            > uint32_t(std::max(triton_function_resources.max_threads_per_block, 0))
        || shape.triton_dynamic_shared_bytes
            > uint32_t(std::max(
                triton_function_resources.max_dynamic_shared_bytes, 0))) {
        fail("requested Triton launch resources exceed function attributes");
    }
    const uint32_t numeric_stream_grid =
        info.route == imparo_sm80_mmq::LaunchRoute::FullTile
            && info.efficiency < 90
        ? uint32_t(g.sm_count) : info.physical_blocks;
    float triton_first_launch_us = 0.0f;
    auto triton_launch = [&] {
        launch_triton(function, shape, n_tok, out_stride, numeric_stream_grid,
                      weights, q8, triton_output);
    };
    if (n_tok == kEvidenceTokens) {
        triton_first_launch_us = event_us(triton_launch);
    } else {
        triton_launch();
    }
    cuda_ok(cudaStreamSynchronize(g.stream), "Triton launch synchronize");
    size_t free_after_first_launches = 0;
    cuda_ok(cudaMemGetInfo(&free_after_first_launches, &total_device_bytes),
            "cudaMemGetInfo after first launches");
    const std::vector<float> triton =
        copy_from_device<float>(triton_output, output_count);
    TimingEvidence timing;
    ArtifactEvidence artifact;
    PostTimingEvidence post_timing;
    if (n_tok == kEvidenceTokens) {
        auto native_launch = [&] {
            imparo_sm80_mmq::LaunchInfo repeated_info;
            if (!launch_native_once(shape, n_tok, out_stride, weights, q8,
                                    native_output, workspace, &repeated_info)) {
                fail("timed native Q4-Q8 launch was rejected");
            }
        };
        timing = measure_interleaved(
            warmup, pairs, launches_per_sample, native_launch, triton_launch);
        size_t free_after_timing = 0;
        cuda_ok(cudaMemGetInfo(&free_after_timing, &total_device_bytes),
                "cudaMemGetInfo after timing");
        timing.native_first_launch_us = native_first_launch_us;
        timing.triton_first_launch_us = triton_first_launch_us;
        artifact.present = true;
        artifact.cubin_bytes = cubin.size();
        artifact.module_load_wall_us = module_load_wall_us;
        artifact.free_before_buffers = free_before_buffers;
        artifact.free_after_buffers = free_after_buffers;
        artifact.free_before_module = free_before_module;
        artifact.free_after_module = free_after_module;
        artifact.free_after_first_launches = free_after_first_launches;
        artifact.free_after_timing = free_after_timing;
        artifact.total_device_bytes = total_device_bytes;
        artifact.observed_buffer_delta = free_before_buffers > free_after_buffers
            ? free_before_buffers - free_after_buffers : 0;
        artifact.observed_module_delta = free_before_module > free_after_module
            ? free_before_module - free_after_module : 0;
        artifact.observed_first_launch_delta =
            free_after_module > free_after_first_launches
                ? free_after_module - free_after_first_launches : 0;
        artifact.observed_timing_delta =
            free_after_first_launches > free_after_timing
                ? free_after_first_launches - free_after_timing : 0;
        const uint64_t minimum_free = std::min<uint64_t>(
            {uint64_t(free_after_buffers), uint64_t(free_after_module),
             uint64_t(free_after_first_launches), uint64_t(free_after_timing)});
        artifact.observed_peak_delta = free_before_buffers > minimum_free
            ? free_before_buffers - minimum_free : 0;
        artifact.weights_logical_bytes = weight_bytes;
        artifact.q8_logical_bytes = q8_bytes;
        artifact.output_logical_bytes = output_count * sizeof(float);
        artifact.workspace_logical_bytes = workspace_bytes;
        artifact.guarded_allocation_bytes = weight_bytes + q8_bytes
            + 2 * output_count * sizeof(float) + workspace_bytes
            + 5 * 2 * kGuardBytes;
        artifact.native_resources = query_native_resources(shape);
        artifact.triton_resources = triton_function_resources;
        const std::vector<float> post_native =
            copy_from_device<float>(native_output, output_count);
        const std::vector<float> post_triton =
            copy_from_device<float>(triton_output, output_count);
        post_timing.present = true;
        post_timing.comparison = compare(
            post_triton, post_native, shape.n_out, n_tok, out_stride);
        const uint64_t post_input_mismatches =
            byte_mismatches(weights, host_weights.data(), host_weights.size())
            + byte_mismatches(q8, host_q8.data(),
                              host_q8.size() * sizeof(BlockQ8_1Mmq));
        post_timing.native_input_mismatches = post_input_mismatches;
        post_timing.triton_input_mismatches = post_input_mismatches;
        post_timing.native_padding_errors =
            padding_errors(post_native, shape.n_out, n_tok, out_stride);
        post_timing.triton_padding_errors =
            padding_errors(post_triton, shape.n_out, n_tok, out_stride);
        post_timing.canary_errors = weights.canary_errors()
            + q8.canary_errors() + native_output.canary_errors()
            + triton_output.canary_errors() + workspace.canary_errors();
    }
    const uint64_t triton_input_mismatches =
        byte_mismatches(weights, host_weights.data(), host_weights.size())
        + byte_mismatches(q8, host_q8.data(), host_q8.size() * sizeof(BlockQ8_1Mmq));
    const uint64_t triton_padding_errors =
        padding_errors(triton, shape.n_out, n_tok, out_stride);
    OracleEvidence oracle = build_oracle(
        shape, n_tok, out_stride, numeric_stream_grid,
        has_numeric_seams(info), host_weights, host_q8, native, triton);
    GuardedBuffer wrong_grid_output;
    if (shape.n_in == 10240 && shape.n_out == 2560
        && n_tok == kEvidenceTokens) {
        wrong_grid_output.allocate(output_count * sizeof(float));
        cuda_ok(cudaMemsetAsync(wrong_grid_output.data, 0xcd,
                               wrong_grid_output.bytes, g.stream),
                "initialize wrong-grid output sentinel");
        launch_triton(function, shape, n_tok, out_stride, 80,
                      weights, q8, wrong_grid_output);
        cuda_ok(cudaStreamSynchronize(g.stream),
                "wrong-grid Triton launch synchronize");
        const std::vector<float> wrong_grid =
            copy_from_device<float>(wrong_grid_output, output_count);
        oracle.wrong_grid_gpu_launched = true;
        oracle.wrong_grid_gpu_vs_correct = compare(
            wrong_grid, triton, shape.n_out, n_tok, out_stride);
        oracle.wrong_grid_gpu_vs_native = compare(
            wrong_grid, native, shape.n_out, n_tok, out_stride);
        oracle.wrong_grid_gpu_canary_errors = wrong_grid_output.canary_errors();
        oracle.wrong_grid_gpu_input_mismatches =
            byte_mismatches(weights, host_weights.data(), host_weights.size())
            + byte_mismatches(q8, host_q8.data(),
                              host_q8.size() * sizeof(BlockQ8_1Mmq));
        oracle.wrong_grid_gpu_padding_errors = padding_errors(
            wrong_grid, shape.n_out, n_tok, out_stride);
    }

    ShapeResult result;
    result.shape = shape;
    result.n_tok = n_tok;
    result.out_stride = out_stride;
    result.numeric_stream_grid = numeric_stream_grid;
    result.launch = info;
    result.difference = compare(triton, native, shape.n_out, n_tok, out_stride);
    result.native_input_mismatches = native_input_mismatches;
    result.triton_input_mismatches = triton_input_mismatches;
    result.native_padding_errors = native_padding_errors;
    result.triton_padding_errors = triton_padding_errors;
    result.oracle = oracle;
    result.timing = timing;
    result.artifact = artifact;
    result.post_timing = post_timing;
    result.canary_errors = weights.canary_errors() + q8.canary_errors()
        + native_output.canary_errors() + triton_output.canary_errors()
        + workspace.canary_errors();

    driver_ok(program_catalog.driver.module_unload(module), "cuModuleUnload(Q4-Q8)");
    workspace.release();
    wrong_grid_output.release();
    triton_output.release();
    native_output.release();
    q8.release();
    weights.release();
    trace(std::string("complete ") + case_name.str());
    return result;
}

void emit_difference(const Difference & value) {
    std::cout << "{\"max_abs\":" << value.max_abs
              << ",\"max_rel\":" << value.max_rel
              << ",\"max_normalized_rel\":" << value.max_normalized_rel
              << ",\"rms\":" << value.rms()
              << ",\"finite\":" << value.finite
              << ",\"non_finite\":" << value.non_finite
              << ",\"bitwise_different\":" << value.bitwise_different << '}';
}

void emit_reference_difference(const ReferenceDifference & value) {
    std::cout << "{\"max_abs\":" << value.max_abs
              << ",\"max_rel\":" << value.max_rel
              << ",\"max_normalized_rel\":" << value.max_normalized_rel
              << ",\"rms\":" << value.rms()
              << ",\"count\":" << value.count
              << ",\"non_finite\":" << value.non_finite << '}';
}

void emit_samples(const std::vector<float> & values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void emit_function_resources(const FunctionResources & value) {
    std::cout << "{\"registers_per_thread\":" << value.registers_per_thread
              << ",\"static_shared_bytes\":" << value.static_shared_bytes
              << ",\"local_bytes\":" << value.local_bytes
              << ",\"max_threads_per_block\":" << value.max_threads_per_block
              << ",\"max_dynamic_shared_bytes\":"
              << value.max_dynamic_shared_bytes
              << ",\"binary_version\":" << value.binary_version
              << ",\"launch_threads\":" << value.launch_threads
              << ",\"launch_dynamic_shared_bytes\":"
              << value.launch_dynamic_shared_bytes
              << ",\"block_m\":" << value.block_m
              << ",\"block_n\":" << value.block_n << '}';
}

void emit_timing(const TimingEvidence & value) {
    if (!value.present) {
        std::cout << "null";
        return;
    }
    const double native_median = median(value.native_us);
    const double triton_median = median(value.triton_us);
    std::cout << "{\"clock\":\"cuda_event\",\"schedule\":\"ABBA_BAAB\""
              << ",\"warmup\":" << value.warmup
              << ",\"pairs\":" << value.pairs
              << ",\"launches_per_sample\":" << value.launches_per_sample
              << ",\"samples_per_route\":" << value.native_us.size()
              << ",\"native_first_launch_us\":"
              << value.native_first_launch_us
              << ",\"triton_first_launch_us\":"
              << value.triton_first_launch_us
              << ",\"native_median_us\":" << native_median
              << ",\"triton_median_us\":" << triton_median
              << ",\"native_mean_us\":" << mean(value.native_us)
              << ",\"triton_mean_us\":" << mean(value.triton_us)
              << ",\"native_cv\":" << coefficient_of_variation(value.native_us)
              << ",\"triton_cv\":" << coefficient_of_variation(value.triton_us)
              << ",\"paired_mad_fraction\":" << paired_mad_fraction(value)
              << ",\"median_speedup_native_over_triton\":"
              << (triton_median > 0.0 ? native_median / triton_median : 0.0)
              << ",\"native_samples_us\":";
    emit_samples(value.native_us);
    std::cout << ",\"triton_samples_us\":";
    emit_samples(value.triton_us);
    std::cout << '}';
}

void emit_artifact(const ArtifactEvidence & value) {
    if (!value.present) {
        std::cout << "null";
        return;
    }
    std::cout << "{\"cubin_bytes\":" << value.cubin_bytes
              << ",\"module_load_wall_us\":" << value.module_load_wall_us
              << ",\"cuda_mem_free_before_buffers\":"
              << value.free_before_buffers
              << ",\"cuda_mem_free_after_buffers\":" << value.free_after_buffers
              << ",\"cuda_mem_free_before_module\":" << value.free_before_module
              << ",\"cuda_mem_free_after_module\":" << value.free_after_module
              << ",\"cuda_mem_free_after_first_launches\":"
              << value.free_after_first_launches
              << ",\"cuda_mem_free_after_timing\":" << value.free_after_timing
              << ",\"cuda_mem_total_bytes\":" << value.total_device_bytes
              << ",\"observed_buffer_delta_bytes\":"
              << value.observed_buffer_delta
              << ",\"observed_module_delta_bytes\":"
              << value.observed_module_delta
              << ",\"observed_first_launch_delta_bytes\":"
              << value.observed_first_launch_delta
              << ",\"observed_timing_delta_bytes\":"
              << value.observed_timing_delta
              << ",\"observed_peak_delta_bytes\":"
              << value.observed_peak_delta
              << ",\"weights_logical_bytes\":" << value.weights_logical_bytes
              << ",\"q8_logical_bytes\":" << value.q8_logical_bytes
              << ",\"output_logical_bytes_per_buffer\":"
              << value.output_logical_bytes
              << ",\"output_buffer_count\":2"
              << ",\"workspace_logical_bytes\":"
              << value.workspace_logical_bytes
              << ",\"guarded_allocation_bytes\":"
              << value.guarded_allocation_bytes
              << ",\"native_function\":";
    emit_function_resources(value.native_resources);
    std::cout << ",\"triton_function\":";
    emit_function_resources(value.triton_resources);
    std::cout << '}';
}

void emit_oracle(const ShapeResult & result) {
    const OracleEvidence & value = result.oracle;
    if (!value.present) {
        std::cout << "null";
        return;
    }
    std::cout << "{\"samples\":" << value.samples
              << ",\"seam_104_samples\":" << value.seam_104_samples
              << ",\"seam_208_samples\":" << value.seam_208_samples
              << ",\"no_seam_samples\":" << value.no_seam_samples
              << ",\"wrong_grid_bitwise_different\":"
              << value.wrong_grid_bitwise_different
              << ",\"seam_minus_one_bitwise_different\":"
              << value.seam_minus_one_bitwise_different
              << ",\"seam_plus_one_bitwise_different\":"
              << value.seam_plus_one_bitwise_different
              << ",\"wrong_grid_gpu_launched\":"
              << (value.wrong_grid_gpu_launched ? "true" : "false")
              << ",\"wrong_grid_gpu_canary_errors\":"
              << value.wrong_grid_gpu_canary_errors
              << ",\"wrong_grid_gpu_input_mismatches\":"
              << value.wrong_grid_gpu_input_mismatches
              << ",\"wrong_grid_gpu_padding_errors\":"
              << value.wrong_grid_gpu_padding_errors;
    if (result.shape.n_in == 10240) {
        std::cout << ",\"directed_seam_tiles\":[2,5]"
                  << ",\"adjacent_no_seam_tiles\":[1,3,4,6]"
                  << ",\"directed_block_boundaries\":[103,104,207,208]";
    } else {
        std::cout << ",\"directed_seam_tiles\":[]"
                  << ",\"adjacent_no_seam_tiles\":[]"
                  << ",\"directed_block_boundaries\":[]";
    }
    std::cout << ",\"native_vs_strict_float\":";
    emit_difference(value.native_vs_strict_float);
    std::cout << ",\"triton_vs_strict_float\":";
    emit_difference(value.triton_vs_strict_float);
    std::cout << ",\"wrong_grid_gpu_vs_correct\":";
    emit_difference(value.wrong_grid_gpu_vs_correct);
    std::cout << ",\"wrong_grid_gpu_vs_native\":";
    emit_difference(value.wrong_grid_gpu_vs_native);
    std::cout << ",\"native_vs_f64\":";
    emit_reference_difference(value.native_vs_f64);
    std::cout << ",\"triton_vs_f64\":";
    emit_reference_difference(value.triton_vs_f64);
    std::cout << '}';
}

void emit_result(const ShapeResult & result) {
    std::cout << "{\"shape_id\":\"" << result.shape.id
              << "\",\"n_in\":" << result.shape.n_in
              << ",\"n_out\":" << result.shape.n_out
              << ",\"n_tok\":" << result.n_tok
              << ",\"out_stride\":" << result.out_stride
              << ",\"native_route\":\""
              << imparo_sm80_mmq::launch_route_name(result.launch.route)
              << "\",\"native_tiles\":{\"rows\":" << result.launch.tile_rows
              << ",\"tokens\":" << result.launch.tile_tokens << '}'
              << ",\"logical_tiles\":" << result.launch.logical_tiles
              << ",\"physical_tiles\":" << result.launch.physical_blocks
              << ",\"efficiency\":" << result.launch.efficiency
              << ",\"numeric_seams\":"
              << (has_numeric_seams(result.launch) ? "true" : "false")
              << ",\"fused_epilogue\":"
              << (result.launch.fused_q8 ? "true" : "false")
              << ",\"numeric_stream_grid\":" << result.numeric_stream_grid
              << ",\"triton_launch_config\":{\"bm\":"
              << result.shape.triton_bm
              << ",\"bn\":" << result.shape.triton_bn
              << ",\"threads\":" << result.shape.triton_threads
              << ",\"dynamic_shared_bytes\":"
              << result.shape.triton_dynamic_shared_bytes << '}'
              << ",\"workspace_bytes\":" << kExpectedWorkspaceBytes
              << ",\"comparison\":";
    emit_difference(result.difference);
    std::cout << ",\"native_input_mismatches\":"
              << result.native_input_mismatches
              << ",\"triton_input_mismatches\":"
              << result.triton_input_mismatches
              << ",\"native_padding_errors\":" << result.native_padding_errors
              << ",\"triton_padding_errors\":" << result.triton_padding_errors
              << ",\"canary_errors\":" << result.canary_errors
              << ",\"cpu_oracle\":";
    emit_oracle(result);
    std::cout << ",\"timing\":";
    emit_timing(result.timing);
    std::cout << ",\"artifact_resources\":";
    emit_artifact(result.artifact);
    std::cout << ",\"post_timing_comparison\":";
    if (result.post_timing.present) {
        emit_difference(result.post_timing.comparison);
    } else {
        std::cout << "null";
    }
    std::cout << ",\"post_timing_native_input_mismatches\":"
              << result.post_timing.native_input_mismatches
              << ",\"post_timing_triton_input_mismatches\":"
              << result.post_timing.triton_input_mismatches
              << ",\"post_timing_native_padding_errors\":"
              << result.post_timing.native_padding_errors
              << ",\"post_timing_triton_padding_errors\":"
              << result.post_timing.triton_padding_errors
              << ",\"post_timing_canary_errors\":"
              << result.post_timing.canary_errors;
    std::cout << '}';
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 8 && argc != 16) {
        std::cerr << "usage: kernel_lab_q4_q8 <expand.cubin> <expand.symbol> "
                     "<contract.cubin> <contract.symbol> "
                     "<warmup> <abba_baab_pairs> <launches_per_sample> "
                     "[<expand_bm> <expand_bn> <expand_threads> "
                     "<expand_dynamic_shared> <contract_bm> <contract_bn> "
                     "<contract_threads> <contract_dynamic_shared>]\n";
        return 2;
    }
    const uint32_t warmup = parse_u32(argv[5], "warmup");
    const uint32_t pairs = parse_u32(argv[6], "abba_baab_pairs");
    const uint32_t launches_per_sample =
        parse_u32(argv[7], "launches_per_sample");
    const uint32_t expand_bm = argc == 16
        ? parse_u32(argv[8], "expand_bm") : kBaselineTritonBm;
    const uint32_t expand_bn = argc == 16
        ? parse_u32(argv[9], "expand_bn") : kBaselineTritonBn;
    const uint32_t expand_threads = argc == 16
        ? parse_u32(argv[10], "expand_threads") : kBaselineTritonThreads;
    const uint32_t expand_shared = argc == 16
        ? parse_u32_allow_zero(argv[11], "expand_dynamic_shared")
        : kBaselineTritonSharedBytes;
    const uint32_t contract_bm = argc == 16
        ? parse_u32(argv[12], "contract_bm") : kBaselineTritonBm;
    const uint32_t contract_bn = argc == 16
        ? parse_u32(argv[13], "contract_bn") : kBaselineTritonBn;
    const uint32_t contract_threads = argc == 16
        ? parse_u32(argv[14], "contract_threads") : kBaselineTritonThreads;
    const uint32_t contract_shared = argc == 16
        ? parse_u32_allow_zero(argv[15], "contract_dynamic_shared")
        : kBaselineTritonSharedBytes;
    if (ensure_cuda_runtime() != 0 || ensure_program_context(program_catalog) != 0) {
        fail("initialize engine primary context and stream");
    }
    if (g.sm_version != 86) fail("LAB-A local evidence requires exact SM86");
    if (g.sm_count != int(kExpectedSmCount)) {
        fail("C0 route evidence requires the RTX 3060 30-SM device");
    }
    cudaDeviceProp device_properties{};
    cuda_ok(cudaGetDeviceProperties(&device_properties, g.device),
            "cudaGetDeviceProperties for provenance");
    int driver_version = 0;
    int runtime_version = 0;
    cuda_ok(cudaDriverGetVersion(&driver_version),
            "cudaDriverGetVersion for provenance");
    cuda_ok(cudaRuntimeGetVersion(&runtime_version),
            "cudaRuntimeGetVersion for provenance");
    Shape expansion{"k2560-m10240", 2560, 10240, 320, 96,
                    argv[1], argv[2], expand_bm, expand_bn, expand_threads,
                    expand_shared};
    Shape contraction{"k10240-m2560", 10240, 2560, 80, 88,
                      argv[3], argv[4], contract_bm, contract_bn,
                      contract_threads, contract_shared};
    constexpr uint32_t execution_cases[] = {512, 1, 127, 128, 129, 511};
    constexpr uint32_t canonical_cases[] = {1, 127, 128, 129, 511, 512};
    std::vector<ShapeResult> results;
    results.reserve(2 * (sizeof(execution_cases) / sizeof(execution_cases[0])));
    for (const Shape * shape : {&expansion, &contraction}) {
        for (uint32_t n_tok : execution_cases) {
            const uint32_t out_stride = shape->n_out
                + (n_tok == kEvidenceTokens ? 0u : kStridePadding);
            results.push_back(run_case(*shape, n_tok, out_stride, warmup, pairs,
                                       launches_per_sample));
        }
    }
    auto token_rank = [&](uint32_t value) {
        for (size_t index = 0;
             index < sizeof(canonical_cases) / sizeof(canonical_cases[0]);
             ++index) {
            if (canonical_cases[index] == value) return index;
        }
        return sizeof(canonical_cases) / sizeof(canonical_cases[0]);
    };
    std::stable_sort(results.begin(), results.end(),
        [&](const ShapeResult & left, const ShapeResult & right) {
            if (left.shape.n_in != right.shape.n_in) {
                return left.shape.n_in < right.shape.n_in;
            }
            return token_rank(left.n_tok) < token_rank(right.n_tok);
        });
    bool structural_ok = true;
    bool debug_exact_triton_native_pass = true;
    for (const ShapeResult & result : results) {
        structural_ok = structural_ok && result.canary_errors == 0
            && result.native_input_mismatches == 0
            && result.triton_input_mismatches == 0
            && result.native_padding_errors == 0
            && result.triton_padding_errors == 0
            && result.difference.non_finite == 0;
        debug_exact_triton_native_pass = debug_exact_triton_native_pass
            && result.difference.bitwise_different == 0;
        if (result.n_tok == kEvidenceTokens) {
            auto valid_samples = [](const std::vector<float> & values) {
                return !values.empty()
                    && std::all_of(values.begin(), values.end(), [](float value) {
                           return std::isfinite(value) && value > 0.0f;
                       });
            };
            structural_ok = structural_ok && result.timing.present
                && result.artifact.present
                && result.timing.native_us.size() == uint64_t(pairs) * 2
                && result.timing.triton_us.size() == uint64_t(pairs) * 2
                && valid_samples(result.timing.native_us)
                && valid_samples(result.timing.triton_us)
                && result.timing.native_first_launch_us > 0.0f
                && result.timing.triton_first_launch_us > 0.0f
                && result.artifact.cubin_bytes > 0
                && result.artifact.native_resources.binary_version == 86
                && result.artifact.triton_resources.binary_version == 86
                && result.artifact.native_resources.launch_threads == 256
                && result.artifact.triton_resources.launch_threads
                    == result.shape.triton_threads
                && result.artifact.triton_resources.launch_dynamic_shared_bytes
                    == result.shape.triton_dynamic_shared_bytes
                && result.artifact.triton_resources.block_m
                    == result.shape.triton_bm
                && result.artifact.triton_resources.block_n
                    == result.shape.triton_bn
                && result.post_timing.present
                && result.post_timing.comparison.non_finite == 0
                && result.post_timing.native_input_mismatches == 0
                && result.post_timing.triton_input_mismatches == 0
                && result.post_timing.native_padding_errors == 0
                && result.post_timing.triton_padding_errors == 0
                && result.post_timing.canary_errors == 0;
        } else {
            structural_ok = structural_ok && !result.timing.present
                && !result.artifact.present && !result.post_timing.present;
        }
        if (result.oracle.present) {
            structural_ok = structural_ok && result.oracle.samples > 0
                && result.oracle.native_vs_strict_float.non_finite == 0
                && result.oracle.triton_vs_strict_float.non_finite == 0
                && result.oracle.native_vs_f64.non_finite == 0
                && result.oracle.triton_vs_f64.non_finite == 0;
            if (result.shape.n_in == 10240) {
                structural_ok = structural_ok
                    && result.oracle.seam_104_samples > 0
                    && result.oracle.seam_208_samples > 0
                    && result.oracle.no_seam_samples > 0
                    && result.oracle.wrong_grid_bitwise_different > 0
                    && result.oracle.seam_minus_one_bitwise_different > 0
                    && result.oracle.seam_plus_one_bitwise_different > 0
                    && result.oracle.wrong_grid_gpu_launched
                    && result.oracle.wrong_grid_gpu_vs_correct.non_finite == 0
                    && result.oracle.wrong_grid_gpu_vs_correct.bitwise_different > 0
                    && result.oracle.wrong_grid_gpu_canary_errors == 0
                    && result.oracle.wrong_grid_gpu_input_mismatches == 0
                    && result.oracle.wrong_grid_gpu_padding_errors == 0;
            }
        }
    }
    std::cout << std::setprecision(10)
              << "{\"schema\":1,\"phase\":\"A2\",\"milestone\":\"C0\""
              << ",\"production_enabled\":false,\"target_sm\":86"
              << ",\"device\":{\"uuid\":\""
              << uuid_hex(device_properties.uuid)
              << "\",\"name\":\"" << device_properties.name
              << "\",\"sm\":" << g.sm_version
              << ",\"sm_count\":" << g.sm_count
              << ",\"driver_version\":" << driver_version
              << ",\"cuda_runtime_version\":" << runtime_version << '}'
              << ",\"same_primary_context\":true,\"same_stream\":true"
              << ",\"visible_arguments\":8,\"hidden_arguments\":2"
              << ",\"tail_tokens\":[1,127,128,129,511,512]"
              << ",\"measurement_request\":{\"warmup\":" << warmup
              << ",\"pairs\":" << pairs
              << ",\"launches_per_sample\":" << launches_per_sample
              << ",\"samples_per_route\":" << uint64_t(pairs) * 2
              << ",\"formal_contract\":"
              << (warmup == 20 && pairs == 50 && launches_per_sample == 16
                      ? "true" : "false")
              << '}'
              << ",\"policy_admissible\":false"
              << ",\"metadata_kparam_preflight\":false"
              << ",\"debug_exact_triton_native_required\":false"
              << ",\"debug_exact_triton_native_observed\":true"
              << ",\"debug_exact_triton_native_pass\":"
              << (debug_exact_triton_native_pass ? "true" : "false")
              << ",\"direct_debug_exit_requires_exact\":false"
              << ",\"formal_pass\":false,\"cases\":[";
    for (size_t index = 0; index < results.size(); ++index) {
        if (index) std::cout << ',';
        emit_result(results[index]);
    }
    std::cout << "],\"structural_ok\":" << (structural_ok ? "true" : "false")
              << "}\n";
    cuda_ok(cudaStreamSynchronize(g.stream), "final synchronize");
    if (g.stream) {
        cuda_ok(cudaStreamDestroy(g.stream), "cudaStreamDestroy");
        g.stream = nullptr;
        g.runtime_initialized = false;
    }
    return structural_ok ? 0 : 3;
}
