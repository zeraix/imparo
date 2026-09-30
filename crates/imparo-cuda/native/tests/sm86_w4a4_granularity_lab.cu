#include "../sm86/mmq_q4_a4_granularity_lab.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace {

using namespace imparo_sm86_w4a4_gate0;

constexpr int kSyntheticRecords = 384;
constexpr int kRandomIdentityRecords = 256;
constexpr double kRealGateCosineMin = 0.999;
constexpr double kRealGateRelativeL2Max = 0.03;
constexpr double kRealGateApproximateCoverageMin = 0.70;

int failures = 0;

void expect(bool condition, const char * label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

struct Lcg {
    explicit Lcg(std::uint32_t seed) : state(seed) {}

    std::uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }

    std::uint32_t state;
};

void fill_codes(AuthorityRecord & record, std::uint32_t seed) {
    Lcg random(seed);
    for (int block = 0; block < kBlocksPerRecord; ++block) {
        for (int index = 0; index < kK32; ++index) {
            const int q4 = int((random.next() >> 28) & 15u) - 8;
            set_q4_code(record.weights[std::size_t(block)], index, q4);
            record.activations[std::size_t(block)].qs[std::size_t(index)] =
                static_cast<std::int8_t>(
                    int((random.next() >> 24) & 255u) - 128);
        }
    }
}

AuthorityRecord make_record(int mode, std::uint32_t seed) {
    AuthorityRecord record;
    fill_codes(record, seed);
    static constexpr std::array<float, 4> uniform = {
        0.03125f, 0.03125f, 0.03125f, 0.03125f};
    static constexpr std::array<float, 4> split = {
        0.015625f, 0.015625f, 0.078125f, 0.078125f};
    static constexpr std::array<float, 4> ragged = {
        0.0078125f, 0.0703125f, 0.015625f, 0.125f};
    const auto & scales = mode == 0 ? uniform : (mode == 1 ? split : ragged);
    for (int block = 0; block < kBlocksPerRecord; ++block) {
        record.weights[std::size_t(block)].d = scales[std::size_t(block)];
        record.activations[std::size_t(block)].d = 0.0078125f;
    }
    return record;
}

void test_q4_authority_layout() {
    Q4K32 block;
    block.d = 0.25f;
    int cases = 0;
    for (int index = 0; index < kK32; ++index) {
        for (int code = -8; code <= 7; ++code) {
            set_q4_code(block, index, code);
            expect(q4_code(block, index) == code, "Q4 code round trip");
            expect(q4_authority_value(block, index) == float(code) * block.d,
                "Q4 authority value");
            ++cases;
        }
    }
    int nibble_preservation_cases = 0;
    for (int byte = 0; byte < kQ4BytesPerBlock; ++byte) {
        for (int low = -8; low <= 7; ++low) {
            for (int high = -8; high <= 7; ++high) {
                block.qs[std::size_t(byte)] = 0;
                set_q4_code(block, byte, low);
                set_q4_code(block, byte + 16, high);
                expect(q4_code(block, byte) == low,
                    "Q4 low nibble preserved by high write");
                expect(q4_code(block, byte + 16) == high,
                    "Q4 high nibble preserved by low write");
                ++nibble_preservation_cases;
            }
        }
    }
    std::printf(
        "{\"q4_authority\":{\"layout\":\"low_0_15_high_16_31\","
        "\"k32_cases\":%d,\"nibble_preservation_cases\":%d,"
        "\"mismatches\":%d}}\n",
        cases, nibble_preservation_cases, failures);
}

void test_closed_form_weight_scale() {
    AuthorityRecord record;
    for (int block = 0; block < 2; ++block) {
        record.weights[std::size_t(block)].d = block == 0 ? 2.0f : 4.0f;
        record.activations[std::size_t(block)].d = 1.0f;
        for (int index = 0; index < kK32; ++index) {
            set_q4_code(record.weights[std::size_t(block)], index,
                block == 0 ? 1 : 2);
            record.activations[std::size_t(block)].qs[std::size_t(index)] =
                static_cast<std::int8_t>(index - 16);
        }
    }
    const WeightPool first = pool_weights_least_squares(record, 0, kG64);
    // S2_0=32, S2_1=128, so dG=(2*32+4*128)/(32+128)=3.6.
    expect(first.valid, "closed-form pool valid");
    expect(std::fabs(double(first.d_g) - 3.6) < 1.0e-6,
        "closed-form analytic dG");
    expect(first.sum_q == 96, "closed-form analytic sum_q");
    for (int item = 0; item < kG64; ++item) {
        expect(int(first.r_w[std::size_t(item)]) == (item < 32 ? 1 : 2),
            "r_w preserves authority code");
    }
    for (int block = 0; block < 2; ++block) {
        record.activations[std::size_t(block)].d = 123.0f + float(block);
        for (int index = 0; index < kK32; ++index) {
            record.activations[std::size_t(block)].qs[std::size_t(index)] =
                static_cast<std::int8_t>(127 - index);
        }
    }
    const WeightPool changed_activation =
        pool_weights_least_squares(record, 0, kG64);
    expect(std::memcmp(&first.d_g, &changed_activation.d_g,
        sizeof(first.d_g)) == 0, "dG activation independent");
    expect(first.sum_q == changed_activation.sum_q,
        "sum_q activation independent");
    expect(first.r_w == changed_activation.r_w,
        "r_w activation independent");
    std::printf(
        "{\"closed_form\":{\"formula\":\"sum(d_j*S2_j)/sum(S2_j)\","
        "\"expected_dG\":3.6,\"actual_dG\":%.9g,"
        "\"activation_independent\":true,\"sum_q\":%d}}\n",
        first.d_g, first.sum_q);
}

void test_zero_point_identity() {
    int exhaustive = 0;
    int mismatches = 0;
    for (int weight = -8; weight <= 7; ++weight) {
        for (int activation = 0; activation <= 15; ++activation) {
            for (int zero = 0; zero <= 15; ++zero) {
                const int corrected = weight * activation - zero * weight;
                const int direct = weight * (activation - zero);
                if (corrected != direct) ++mismatches;
                ++exhaustive;
            }
        }
    }
    expect(mismatches == 0, "scalar zero-point correction exhaustive");

    int random_segment_mismatches = 0;
    for (int sample = 0; sample < kRandomIdentityRecords; ++sample) {
        const AuthorityRecord record = make_record(
            sample % 3, 0x91e10da5u + std::uint32_t(sample) * 17u);
        for (const int begin : {0, 64}) {
            const SegmentApproximation approximation = approximate_segment(
                record, begin, kG64, GateThresholds{1.0, 1.0});
            if (!approximation.zero_point_identity
                    || approximation.corrected_dot
                        != approximation.direct_centered_dot) {
                ++random_segment_mismatches;
            }
        }
        const SegmentApproximation approximation = approximate_segment(
            record, 0, kG128, GateThresholds{1.0, 1.0});
        if (!approximation.zero_point_identity
                || approximation.corrected_dot
                    != approximation.direct_centered_dot) {
            ++random_segment_mismatches;
        }
    }
    expect(random_segment_mismatches == 0,
        "random vector zero-point correction");
    std::printf(
        "{\"zero_point\":{\"scalar_cases\":%d,\"scalar_mismatches\":%d,"
        "\"random_segments\":%d,\"random_mismatches\":%d}}\n",
        exhaustive, mismatches, kRandomIdentityRecords * 3,
        random_segment_mismatches);
}

void test_ladder_order_and_fallback() {
    AuthorityRecord uniform = make_record(0, 0x12345678u);
    const LadderResult g128 = evaluate_ladder(
        uniform, GateThresholds{1.0e-6, 1.0});
    expect(g128.valid, "G128 ladder finite");
    expect(g128.level == LadderLevel::G128, "G128 ladder first");

    AuthorityRecord split = make_record(1, 0x2468ace0u);
    // The two K64 halves each have one scale, while the G128 scale pools two
    // different values. This fixed threshold separates the analytic errors.
    const LadderResult g64 = evaluate_ladder(
        split, GateThresholds{1.0e-6, 1.0});
    expect(g64.valid, "G64 ladder finite");
    expect(g64.level == LadderLevel::G64, "G64 ladder second");

    AuthorityRecord fallback_record = make_record(2, 0x0badf00du);
    const LadderResult exact = evaluate_ladder(
        fallback_record, GateThresholds{-1.0, -1.0});
    expect(exact.valid, "exact ladder finite");
    expect(exact.level == LadderLevel::Exact, "exact ladder fallback");
    expect(std::memcmp(&exact.selected, &exact.exact,
        sizeof(exact.exact)) == 0, "exact fallback bit identity");
    std::printf(
        "{\"ladder_oracle\":{\"order\":[\"g128\",\"g64\",\"exact\"],"
        "\"forced_levels\":[\"%s\",\"%s\",\"%s\"],"
        "\"fallback_bit_exact\":true}}\n",
        ladder_level_name(g128.level), ladder_level_name(g64.level),
        ladder_level_name(exact.level));
}

void put_le16(std::vector<std::uint8_t> & bytes,
        std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

void put_le32(std::vector<std::uint8_t> & bytes,
        std::size_t offset, std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) {
        bytes[offset + std::size_t(byte)] =
            static_cast<std::uint8_t>(value >> (byte * 8));
    }
}

void put_le64(std::vector<std::uint8_t> & bytes,
        std::size_t offset, std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        bytes[offset + std::size_t(byte)] =
            static_cast<std::uint8_t>(value >> (byte * 8));
    }
}

std::vector<std::uint8_t> make_valid_capture(
        std::array<std::uint8_t, 32> & fingerprint) {
    constexpr std::size_t payload_bytes = 96;
    std::vector<std::uint8_t> bytes(kCaptureHeaderBytes + payload_bytes, 0);
    std::copy(kCaptureMagic.begin(), kCaptureMagic.end(), bytes.begin());
    put_le16(bytes, 8, kCaptureVersion);
    put_le16(bytes, 10, kCaptureHeaderBytes);
    put_le32(bytes, 12, kCaptureEndianTag);
    put_le32(bytes, 16, kCaptureOpaqueSchemaV1);
    put_le32(bytes, 20, 2);
    put_le64(bytes, 24, 2);
    put_le64(bytes, 32, 3);
    put_le64(bytes, 56, 6);
    put_le64(bytes, 64, 16);
    put_le64(bytes, 72, 32);
    put_le64(bytes, 80, 48);
    put_le64(bytes, 88, 16);
    for (std::size_t index = 0; index < payload_bytes; ++index) {
        bytes[kCaptureHeaderBytes + index] =
            static_cast<std::uint8_t>((index * 37 + 11) & 255);
    }
    fingerprint = capture_identity_sha256(bytes.data(), bytes.size());
    std::copy(fingerprint.begin(), fingerprint.end(), bytes.begin() + 96);
    return bytes;
}

void expect_capture_error(const std::vector<std::uint8_t> & bytes,
        const std::array<std::uint8_t, 32> * fingerprint,
        CaptureError expected, const char * label, int & cases,
        std::size_t validation_size = 0) {
    const std::size_t size =
        validation_size == 0 ? bytes.size() : validation_size;
    const CaptureValidation validation = validate_capture_envelope(
        bytes.data(), size, fingerprint);
    expect(validation.error == expected, label);
    ++cases;
}

void test_capture_reader() {
    const std::string abc = "abc";
    const auto abc_digest = sha256(
        reinterpret_cast<const std::uint8_t *>(abc.data()), abc.size());
    expect(bytes_to_hex(abc_digest)
        == "ba7816bf8f01cfea414140de5dae2223"
           "b00361a396177a9cb410ff61f20015ad",
        "SHA256 known vector");

    std::array<std::uint8_t, 32> fingerprint{};
    const std::vector<std::uint8_t> valid = make_valid_capture(fingerprint);
    int positive_cases = 0;
    int negative_cases = 0;
    expect_capture_error(valid, &fingerprint, CaptureError::None,
        "valid capture envelope", positive_cases);
    expect_capture_error(valid, nullptr,
        CaptureError::MissingExpectedFingerprint,
        "capture requires external fingerprint", negative_cases);
    const CaptureValidation null_input = validate_capture_envelope(
        nullptr, kCaptureHeaderBytes, &fingerprint);
    expect(null_input.error == CaptureError::NullInput,
        "capture null input");
    ++negative_cases;

    auto mutate = [&](std::size_t offset, std::uint8_t value,
            CaptureError error, const char * label) {
        std::vector<std::uint8_t> copy = valid;
        copy[offset] = value;
        expect_capture_error(copy, &fingerprint, error, label, negative_cases);
    };
    mutate(0, 'X', CaptureError::BadMagic, "capture magic");
    mutate(8, 2, CaptureError::BadVersion, "capture version");
    mutate(10, 64, CaptureError::BadHeaderBytes, "capture header bytes");
    mutate(12, 0, CaptureError::BadEndianness, "capture endianness");
    mutate(16, 2, CaptureError::UnsupportedSchema, "capture schema");
    mutate(20, 0, CaptureError::BadShape, "capture rank");

    std::vector<std::uint8_t> truncated(valid.begin(), valid.begin() + 127);
    expect_capture_error(truncated, &fingerprint, CaptureError::Truncated,
        "capture truncated", negative_cases);
    expect_capture_error(valid, &fingerprint, CaptureError::TooLarge,
        "capture maximum size",
        negative_cases, std::size_t(kCaptureMaxBytes + 1));
    std::vector<std::uint8_t> short_payload = valid;
    short_payload.pop_back();
    expect_capture_error(short_payload, &fingerprint, CaptureError::SizeMismatch,
        "capture exact size", negative_cases);
    std::vector<std::uint8_t> bad_shape = valid;
    put_le64(bad_shape, 24, 0);
    expect_capture_error(bad_shape, &fingerprint, CaptureError::BadShape,
        "capture nonzero shape", negative_cases);
    std::vector<std::uint8_t> inactive_dimension = valid;
    put_le64(inactive_dimension, 40, 1);
    expect_capture_error(inactive_dimension, &fingerprint,
        CaptureError::BadShape, "capture inactive dimension",
        negative_cases);
    std::vector<std::uint8_t> shape_overflow = valid;
    put_le64(shape_overflow, 24, std::numeric_limits<std::uint64_t>::max());
    put_le64(shape_overflow, 32, 2);
    expect_capture_error(shape_overflow, &fingerprint,
        CaptureError::ArithmeticOverflow, "capture shape overflow",
        negative_cases);
    std::vector<std::uint8_t> zero_records = valid;
    put_le64(zero_records, 56, 0);
    expect_capture_error(zero_records, &fingerprint,
        CaptureError::BadRecordCount, "capture record count", negative_cases);
    std::vector<std::uint8_t> bad_lengths = valid;
    put_le64(bad_lengths, 72, 0);
    expect_capture_error(bad_lengths, &fingerprint,
        CaptureError::BadByteLengths, "capture byte lengths", negative_cases);
    std::vector<std::uint8_t> length_overflow = valid;
    put_le64(length_overflow, 72, std::numeric_limits<std::uint64_t>::max());
    put_le64(length_overflow, 80, 1);
    expect_capture_error(length_overflow, &fingerprint,
        CaptureError::ArithmeticOverflow, "capture byte overflow",
        negative_cases);
    std::vector<std::uint8_t> record_multiply_overflow = valid;
    put_le64(record_multiply_overflow, 64,
        std::numeric_limits<std::uint64_t>::max());
    expect_capture_error(record_multiply_overflow, &fingerprint,
        CaptureError::ArithmeticOverflow, "capture record multiply overflow",
        negative_cases);
    std::array<std::uint8_t, 32> wrong_fingerprint = fingerprint;
    wrong_fingerprint[0] ^= 1;
    expect_capture_error(valid, &wrong_fingerprint,
        CaptureError::FingerprintMismatch, "capture expected fingerprint",
        negative_cases);
    std::vector<std::uint8_t> changed_payload = valid;
    changed_payload.back() ^= 1;
    expect_capture_error(changed_payload, &fingerprint,
        CaptureError::FingerprintMismatch, "capture payload identity",
        negative_cases);
    std::vector<std::uint8_t> changed_header_hash = valid;
    changed_header_hash[96] ^= 1;
    expect_capture_error(changed_header_hash, &fingerprint,
        CaptureError::FingerprintMismatch, "capture embedded identity",
        negative_cases);

    std::vector<std::uint8_t> swapped_sections = valid;
    put_le64(swapped_sections, 72, 48);
    put_le64(swapped_sections, 80, 32);
    expect_capture_error(swapped_sections, &fingerprint,
        CaptureError::FingerprintMismatch,
        "capture legal section metadata identity", negative_cases);

    std::vector<std::uint8_t> reshaped = valid;
    put_le64(reshaped, 24, 1);
    put_le64(reshaped, 32, 6);
    expect_capture_error(reshaped, &fingerprint,
        CaptureError::FingerprintMismatch,
        "capture legal shape metadata identity", negative_cases);

    expect(positive_cases == 1, "capture positive case count");
    expect(negative_cases == 23, "capture negative case count");
    std::printf(
        "{\"capture_reader\":{\"mode\":\"opaque_envelope_v1\","
        "\"positive_cases\":%d,\"baseline_negative_cases\":17,"
        "\"additional_negative_cases\":6,\"negative_cases\":%d,"
        "\"identity\":\"canonical_header_plus_payload_v1\","
        "\"sha256_known_vector\":true,"
        "\"real_record_schema\":\"unbound\"}}\n",
        positive_cases, negative_cases);
}

enum class StatsError : std::uint8_t {
    None,
    LengthMismatch,
    Empty,
    NonFinite,
};

struct ErrorStats {
    bool valid = false;
    StatsError error = StatsError::None;
    std::size_t count = 0;
    double nrmse = 0.0;
    double cosine = 1.0;
    double p99_absolute = 0.0;
    // v2 is descriptive only. Its denominator floor is 1% of reference RMS,
    // so near-zero dot products can dominate it; it is not an admission gate.
    double p99_floor_relative_v2 = 0.0;
};

ErrorStats calculate_error_stats(const std::vector<double> & exact,
        const std::vector<double> & approximate) {
    ErrorStats result;
    result.count = exact.size();
    if (exact.size() != approximate.size()) {
        result.error = StatsError::LengthMismatch;
        return result;
    }
    if (exact.empty()) {
        result.error = StatsError::Empty;
        return result;
    }
    for (std::size_t index = 0; index < exact.size(); ++index) {
        if (!std::isfinite(exact[index])
                || !std::isfinite(approximate[index])) {
            result.error = StatsError::NonFinite;
            return result;
        }
    }
    double squared_error = 0.0;
    double exact_energy = 0.0;
    double approximate_energy = 0.0;
    double dot = 0.0;
    std::vector<double> absolute;
    absolute.reserve(exact.size());
    for (std::size_t index = 0; index < exact.size(); ++index) {
        const double error = approximate[index] - exact[index];
        squared_error += error * error;
        exact_energy += exact[index] * exact[index];
        approximate_energy += approximate[index] * approximate[index];
        dot += exact[index] * approximate[index];
        absolute.push_back(std::fabs(error));
        if (!std::isfinite(error) || !std::isfinite(squared_error)
                || !std::isfinite(exact_energy)
                || !std::isfinite(approximate_energy)
                || !std::isfinite(dot) || !std::isfinite(absolute.back())) {
            result.error = StatsError::NonFinite;
            return result;
        }
    }
    result.nrmse = normalized_rmse(squared_error, exact_energy);
    if (exact_energy == 0.0 && approximate_energy == 0.0) {
        result.cosine = 1.0;
    } else if (exact_energy == 0.0 || approximate_energy == 0.0) {
        result.cosine = 0.0;
    } else {
        const double denominator =
            std::sqrt(exact_energy) * std::sqrt(approximate_energy);
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            result.error = StatsError::NonFinite;
            return result;
        }
        result.cosine = dot / denominator;
    }
    const double rms_exact = std::sqrt(exact_energy / double(exact.size()));
    const double floor = std::max(1.0e-12, rms_exact * 0.01);
    if (!std::isfinite(result.nrmse) || !std::isfinite(result.cosine)
            || !std::isfinite(rms_exact) || !std::isfinite(floor)) {
        result.error = StatsError::NonFinite;
        return result;
    }
    std::vector<double> normalized;
    normalized.reserve(exact.size());
    for (std::size_t index = 0; index < exact.size(); ++index) {
        const double value = absolute[index]
            / std::max(std::fabs(exact[index]), floor);
        if (!std::isfinite(value)) {
            result.error = StatsError::NonFinite;
            return result;
        }
        normalized.push_back(value);
    }
    auto p99 = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const std::size_t rank = std::min(values.size() - 1,
            std::size_t(std::ceil(0.99 * double(values.size()))) - 1);
        return values[rank];
    };
    result.p99_absolute = p99(std::move(absolute));
    result.p99_floor_relative_v2 = p99(std::move(normalized));
    if (!std::isfinite(result.p99_absolute)
            || !std::isfinite(result.p99_floor_relative_v2)) {
        result.error = StatsError::NonFinite;
        return result;
    }
    result.valid = true;
    result.error = StatsError::None;
    return result;
}

struct LevelSamples {
    std::vector<double> exact;
    std::vector<double> approximate;
};

void test_minmax_affine_and_finite_guards() {
    int cases = 0;
    AuthorityRecord record;
    for (int block = 0; block < kBlocksPerRecord; ++block) {
        record.weights[std::size_t(block)].d = 0.5f;
        record.activations[std::size_t(block)].d = 1.0f;
        for (int index = 0; index < kK32; ++index) {
            set_q4_code(record.weights[std::size_t(block)], index, 1);
            record.activations[std::size_t(block)].qs[std::size_t(index)] =
                static_cast<std::int8_t>((index & 1) == 0 ? -15 : 15);
        }
    }

    const ActivationPool affine =
        quantize_q8_authority_to_a4(record, 0, kG64);
    expect(affine.valid, "fixed minmax affine valid");
    expect(affine.d_a == 2.0f, "fixed minmax da=(max-min)/15");
    expect(affine.z == 8, "fixed minmax rounded zero");
    for (int item = 0; item < kG64; ++item) {
        const int expected = (item & 1) == 0 ? 0 : 15;
        expect(int(affine.r_a[std::size_t(item)]) == expected,
            "fixed minmax centered rounding");
    }
    ++cases;

    AuthorityRecord all_zero = record;
    for (Q8K32 & block : all_zero.activations) {
        block.qs.fill(0);
    }
    const ActivationPool zero_pool =
        quantize_q8_authority_to_a4(all_zero, 0, kG64);
    expect(zero_pool.valid && zero_pool.d_a == 0.0f && zero_pool.z == 8
        && zero_pool.nrmse == 0.0, "all-zero affine degeneracy");
    ++cases;

    AuthorityRecord constant_nonzero = record;
    for (Q8K32 & block : constant_nonzero.activations) {
        block.qs.fill(7);
    }
    expect(!quantize_q8_authority_to_a4(
        constant_nonzero, 0, kG64).valid,
        "constant nonzero affine fails closed");
    ++cases;

    int code = -1;
    expect(round_centered_affine_u4(
        std::numeric_limits<double>::max(), 1.0, 8, code) && code == 15,
        "finite huge positive saturates before int cast");
    ++cases;
    expect(round_centered_affine_u4(
        -std::numeric_limits<double>::max(), 1.0, 8, code) && code == 0,
        "finite huge negative saturates before int cast");
    ++cases;
    expect(!round_centered_affine_u4(
        std::numeric_limits<double>::quiet_NaN(), 1.0, 8, code),
        "NaN affine input rejected");
    ++cases;
    expect(round_centered_affine_u4(-0.5, 1.0, 8, code) && code == 7,
        "negative tie rounds away from zero");
    expect(round_centered_affine_u4(0.5, 1.0, 8, code) && code == 9,
        "positive tie rounds away from zero");
    ++cases;

    const int huge_begin = std::numeric_limits<int>::max();
    expect(!pool_weights_least_squares(record, huge_begin, kG64).valid,
        "weight begin/count overflow rejected");
    expect(!quantize_q8_authority_to_a4(record, huge_begin, kG64).valid,
        "activation begin/count overflow rejected");
    ++cases;

    AuthorityRecord huge_activation = record;
    for (Q8K32 & block : huge_activation.activations) {
        block.d = std::numeric_limits<float>::max();
        for (int index = 0; index < kK32; ++index) {
            block.qs[std::size_t(index)] =
                static_cast<std::int8_t>((index & 1) == 0 ? -128 : 127);
        }
    }
    expect(!quantize_q8_authority_to_a4(
        huge_activation, 0, kG64).valid,
        "unrepresentable A4 scale rejected");
    ++cases;

    AuthorityRecord scale_overflow = record;
    for (int block = 0; block < kBlocksPerRecord; ++block) {
        scale_overflow.weights[std::size_t(block)].d =
            std::numeric_limits<float>::max();
        scale_overflow.activations[std::size_t(block)].d =
            std::numeric_limits<float>::max();
    }
    expect(!exact_q4_q8_dot(scale_overflow).valid,
        "exact scale product overflow rejected");
    expect(!evaluate_ladder(scale_overflow).valid,
        "nonfinite exact output fails ladder closed");
    ++cases;

    AuthorityRecord segment_scale_overflow = record;
    for (Q4K32 & block : segment_scale_overflow.weights) {
        block.d = std::numeric_limits<float>::max();
    }
    const SegmentApproximation overflow_segment = approximate_segment(
        segment_scale_overflow, 0, kG64, GateThresholds{1.0, 1.0});
    expect(!overflow_segment.finite && !overflow_segment.quality_pass,
        "dG*dA overflow rejects approximate segment");
    const LadderResult segment_fallback =
        evaluate_ladder(segment_scale_overflow, GateThresholds{1.0, 1.0});
    expect(segment_fallback.valid
        && segment_fallback.level == LadderLevel::Exact,
        "segment scale overflow falls back to finite exact");
    ++cases;

    AuthorityRecord fma_overflow = record;
    for (int block = 0; block < kBlocksPerRecord; ++block) {
        fma_overflow.weights[std::size_t(block)].d =
            std::numeric_limits<float>::max();
        fma_overflow.activations[std::size_t(block)].d = 1.0f;
        for (int index = 0; index < kK32; ++index) {
            set_q4_code(fma_overflow.weights[std::size_t(block)], index, 7);
            fma_overflow.activations[std::size_t(block)].qs[
                std::size_t(index)] = 127;
        }
    }
    expect(!exact_q4_q8_dot(fma_overflow).valid,
        "exact FMA overflow rejected");
    ++cases;

    AuthorityRecord nan_scale = record;
    nan_scale.activations[0].d =
        std::numeric_limits<float>::quiet_NaN();
    expect(!exact_q4_q8_dot(nan_scale).valid,
        "NaN exact scale rejected");
    expect(!evaluate_ladder(nan_scale).valid,
        "NaN record fails ladder closed");
    ++cases;

    const LadderResult normal = evaluate_ladder(record);
    expect(normal.valid && std::isfinite(normal.exact)
        && std::isfinite(normal.selected), "normal finite chain accepted");
    ++cases;

    std::printf(
        "{\"minmax_finite\":{\"estimator\":\"fixed_unsigned_affine\","
        "\"formula\":\"da=(max-min)/15\","
        "\"rounding\":\"nearest_ties_away_then_u4_saturate\","
        "\"cases\":%d,\"fail_closed\":true}}\n", cases);
}

void test_stats_fail_closed() {
    int cases = 0;
    const ErrorStats finite = calculate_error_stats(
        {1.0, -2.0, 3.0}, {1.0, -2.0, 3.0});
    expect(finite.valid && finite.error == StatsError::None,
        "finite equal stats valid");
    ++cases;

    const ErrorStats mismatch = calculate_error_stats({1.0}, {1.0, 2.0});
    expect(!mismatch.valid && mismatch.error == StatsError::LengthMismatch,
        "stats length mismatch rejected");
    ++cases;
    const ErrorStats empty = calculate_error_stats({}, {});
    expect(!empty.valid && empty.error == StatsError::Empty,
        "empty stats rejected");
    ++cases;
    const ErrorStats nan_exact = calculate_error_stats(
        {std::numeric_limits<double>::quiet_NaN()}, {0.0});
    expect(!nan_exact.valid && nan_exact.error == StatsError::NonFinite,
        "NaN exact stats rejected");
    ++cases;
    const ErrorStats inf_approximate = calculate_error_stats(
        {0.0}, {std::numeric_limits<double>::infinity()});
    expect(!inf_approximate.valid
        && inf_approximate.error == StatsError::NonFinite,
        "Inf approximate stats rejected");
    ++cases;
    const double huge = std::numeric_limits<double>::max();
    const ErrorStats overflow = calculate_error_stats(
        {huge, huge}, {-huge, -huge});
    expect(!overflow.valid && overflow.error == StatsError::NonFinite,
        "stats accumulation overflow rejected");
    ++cases;

    std::printf(
        "{\"stats_fail_closed\":{\"cases\":%d,"
        "\"length_checked_before_index\":true,"
        "\"finite_checked_before_sort\":true}}\n", cases);
}

int run_self_test() {
    std::printf(
        "{\"self_test_contract\":{\"weight_nrmse_max\":%.6f,"
        "\"activation_nrmse_max\":%.6f,"
        "\"zero_point_exhaustive_cases\":4096,"
        "\"random_identity_records\":%d,"
        "\"activation_estimator\":\"fixed_minmax_unsigned_affine_v1\","
        "\"capture_reader\":\"canonical_identity_fail_closed_v1\"}}\n",
        kSyntheticGateThresholds.weight_nrmse_max,
        kSyntheticGateThresholds.activation_nrmse_max,
        kRandomIdentityRecords);
    test_q4_authority_layout();
    test_closed_form_weight_scale();
    test_zero_point_identity();
    test_ladder_order_and_fallback();
    test_capture_reader();
    test_minmax_affine_and_finite_guards();
    test_stats_fail_closed();
    std::printf(
        "{\"self_test\":{\"failures\":%d,\"synthetic_screen_run\":false,"
        "\"real_capture_evaluated\":false}}\n", failures);
    return failures == 0 ? 0 : 1;
}

int run_synthetic_screen() {
    std::printf(
        "{\"screen_contract\":{\"selection_weight_nrmse_max\":%.6f,"
        "\"selection_activation_nrmse_max\":%.6f,"
        "\"real_gate_cosine_min\":%.6f,"
        "\"real_gate_relative_l2_max\":%.6f,"
        "\"real_gate_approximate_coverage_min\":%.6f,"
        "\"records\":%d,\"seed_formula\":"
        "\"0x6d2b79f5 xor sample*0x9e3779b9\","
        "\"classification\":\"synthetic_screen_not_real_gate0\"}}\n",
        kSyntheticGateThresholds.weight_nrmse_max,
        kSyntheticGateThresholds.activation_nrmse_max,
        kRealGateCosineMin, kRealGateRelativeL2Max,
        kRealGateApproximateCoverageMin, kSyntheticRecords);
    std::array<int, 3> level_count{};
    std::array<LevelSamples, 3> by_level{};
    std::vector<double> all_exact;
    std::vector<double> all_selected;
    int fallback_bit_mismatches = 0;
    int zero_point_mismatches = 0;
    int invalid_records = 0;
    for (int sample = 0; sample < kSyntheticRecords; ++sample) {
        const int mode = sample % 3;
        const AuthorityRecord record = make_record(
            mode, 0x6d2b79f5u ^ (std::uint32_t(sample) * 0x9e3779b9u));
        const LadderResult result = evaluate_ladder(record);
        if (!result.valid) {
            ++invalid_records;
            continue;
        }
        const int level = result.level == LadderLevel::G128
            ? 0 : (result.level == LadderLevel::G64 ? 1 : 2);
        ++level_count[std::size_t(level)];
        by_level[std::size_t(level)].exact.push_back(result.exact);
        by_level[std::size_t(level)].approximate.push_back(result.selected);
        all_exact.push_back(result.exact);
        all_selected.push_back(result.selected);
        if (!result.g128.zero_point_identity
                || !result.g64[0].zero_point_identity
                || !result.g64[1].zero_point_identity) {
            ++zero_point_mismatches;
        }
        if (result.level == LadderLevel::Exact
                && std::memcmp(&result.selected, &result.exact,
                    sizeof(result.exact)) != 0) {
            ++fallback_bit_mismatches;
        }
    }

    const ErrorStats overall = calculate_error_stats(all_exact, all_selected);
    const ErrorStats g128 = calculate_error_stats(
        by_level[0].exact, by_level[0].approximate);
    const ErrorStats g64 = calculate_error_stats(
        by_level[1].exact, by_level[1].approximate);
    const double approximate_coverage =
        double(level_count[0] + level_count[1]) / double(kSyntheticRecords);
    const bool structural_pass = level_count[0] > 0 && level_count[1] > 0
        && level_count[2] > 0 && zero_point_mismatches == 0
        && fallback_bit_mismatches == 0 && invalid_records == 0
        && overall.valid && g128.valid && g64.valid;
    const bool cosine_pass = overall.valid
        && overall.cosine >= kRealGateCosineMin;
    const bool relative_l2_pass = overall.valid
        && overall.nrmse <= kRealGateRelativeL2Max;
    const bool coverage_pass =
        approximate_coverage >= kRealGateApproximateCoverageMin;
    const bool reference_thresholds_met = structural_pass && cosine_pass
        && relative_l2_pass && coverage_pass;

    std::printf(
        "{\"synthetic_screen\":{\"records\":%d,"
        "\"coverage\":{\"g128\":%d,\"g64\":%d,\"exact\":%d,"
        "\"approximate_fraction\":%.9f},"
        "\"overall\":{\"nrmse\":%.9g,\"cosine\":%.9g,"
        "\"p99_abs\":%.9g,\"p99_floor_relative_v2\":%.9g},"
        "\"g128\":{\"count\":%zu,\"nrmse\":%.9g,\"cosine\":%.9g,"
        "\"p99_abs\":%.9g,\"p99_floor_relative_v2\":%.9g},"
        "\"g64\":{\"count\":%zu,\"nrmse\":%.9g,\"cosine\":%.9g,"
        "\"p99_abs\":%.9g,\"p99_floor_relative_v2\":%.9g},"
        "\"p99_v2_note\":"
        "\"descriptive_only; denominator=max(abs(ref),0.01*rms_ref);"
        "near_zero_sensitive\","
        "\"gate_checks\":{\"structural\":%s,\"cosine\":%s,"
        "\"relative_l2\":%s,\"coverage\":%s},"
        "\"zero_point_mismatches\":%d,"
        "\"fallback_bit_mismatches\":%d,"
        "\"invalid_records\":%d,"
        "\"reference_thresholds_met\":%s,"
        "\"decision\":\"continue_to_real_capture\","
        "\"is_gate\":false,\"real_capture_evaluated\":false}}\n",
        kSyntheticRecords, level_count[0], level_count[1], level_count[2],
        approximate_coverage, overall.nrmse, overall.cosine,
        overall.p99_absolute, overall.p99_floor_relative_v2,
        g128.count, g128.nrmse, g128.cosine,
        g128.p99_absolute, g128.p99_floor_relative_v2,
        g64.count, g64.nrmse, g64.cosine,
        g64.p99_absolute, g64.p99_floor_relative_v2,
        structural_pass ? "true" : "false",
        cosine_pass ? "true" : "false",
        relative_l2_pass ? "true" : "false",
        coverage_pass ? "true" : "false",
        zero_point_mismatches, fallback_bit_mismatches,
        invalid_records, reference_thresholds_met ? "true" : "false");
    if (!structural_pass) return 1;
    // Synthetic observations can never admit Gate0 or return success.
    return 10;
}

int run_capture(const std::string & path, const std::string & expected_hex) {
    std::array<std::uint8_t, 32> expected{};
    if (!hex_to_fingerprint(expected_hex, expected)) {
        std::fprintf(stderr, "expected SHA256 must be exactly 64 hex digits\n");
        return 2;
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open capture: %s\n", path.c_str());
        return 3;
    }
    const std::streamoff end = input.tellg();
    if (end < 0 || std::uint64_t(end) > kCaptureMaxBytes) {
        std::fprintf(stderr, "capture size is invalid or exceeds limit\n");
        return 4;
    }
    std::vector<std::uint8_t> bytes;
    try {
        bytes.resize(static_cast<std::size_t>(end));
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "capture allocation failed under provisional cap\n");
        return 4;
    }
    input.seekg(0);
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!input) {
        std::fprintf(stderr, "capture read failed\n");
        return 4;
    }
    const CaptureValidation validation = validate_capture_envelope(
        bytes.data(), bytes.size(), &expected);
    if (validation.error != CaptureError::None) {
        std::printf(
            "{\"capture\":{\"validated\":false,\"error\":\"%s\"}}\n",
            capture_error_name(validation.error));
        return 5;
    }
    std::printf(
        "{\"capture\":{\"validated\":true,\"version\":%u,\"schema\":%u,"
        "\"rank\":%u,\"record_count\":%llu,\"record_bytes\":%llu,"
        "\"fingerprint\":\"%s\","
        "\"real_gate_thresholds\":{\"cosine_min\":%.6f,"
        "\"relative_l2_max\":%.6f,\"approximate_coverage_min\":%.6f},"
        "\"analyzed\":false,"
        "\"reason\":\"real_record_schema_not_audited\"}}\n",
        unsigned(validation.header.version), validation.header.schema,
        validation.header.rank,
        static_cast<unsigned long long>(validation.header.record_count),
        static_cast<unsigned long long>(validation.header.record_bytes),
        bytes_to_hex(validation.header.identity_sha256).c_str(),
        kRealGateCosineMin, kRealGateRelativeL2Max,
        kRealGateApproximateCoverageMin);
    std::printf(
        "{\"capture_schema_todo\":{"
        "\"stratify_by\":[\"layer\",\"shape\",\"ladder_level\"],"
        "\"aggregate\":\"work_weighted\","
        "\"admission_requires\":\"worst_layer_bound\","
        "\"reader\":\"stream_records_before_schema_binding\"}}\n");
    // A valid envelope is not a valid Gate0 result until the real record schema
    // is audited and bound to the authority structs above.
    return 6;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        return run_self_test();
    }
    if (argc == 2 && std::strcmp(argv[1], "--synthetic-screen") == 0) {
        return run_synthetic_screen();
    }
    if (argc == 5 && std::strcmp(argv[1], "--capture") == 0
            && std::strcmp(argv[3], "--expect-sha256") == 0) {
        return run_capture(argv[2], argv[4]);
    }
    std::fprintf(stderr,
        "usage: %s [--self-test | --synthetic-screen | --capture PATH "
        "--expect-sha256 HEX]\n", argv[0]);
    return 2;
}
