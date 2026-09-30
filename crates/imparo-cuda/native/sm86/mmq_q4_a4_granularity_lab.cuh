#pragma once

// CPU-only W4A4 granularity laboratory.
//
// This header deliberately defines no CUDA kernel, launcher, selector, ABI, or
// workflow hook. It measures whether Q4_0 K32 blocks and Q8 authority
// activations can share G64/G128 scales before any SM86 implementation exists.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace imparo_sm86_w4a4_gate0 {

constexpr int kK32 = 32;
constexpr int kG64 = 64;
constexpr int kG128 = 128;
constexpr int kBlocksPerRecord = 4;
constexpr int kQ4BytesPerBlock = 16;
constexpr int kMaxGroupValues = kG128;

// These are feasibility thresholds, not production admission thresholds and
// not claims about real captures. They are printed by the synthetic test.
struct GateThresholds {
    double weight_nrmse_max = 0.04;
    double activation_nrmse_max = 0.12;
};

constexpr GateThresholds kSyntheticGateThresholds{};

struct Q4K32 {
    float d = 0.0f;
    std::array<std::uint8_t, kQ4BytesPerBlock> qs{};
};

struct Q8K32 {
    float d = 0.0f;
    std::array<std::int8_t, kK32> qs{};
};

struct AuthorityRecord {
    std::array<Q4K32, kBlocksPerRecord> weights{};
    std::array<Q8K32, kBlocksPerRecord> activations{};
};

inline int q4_storage(const Q4K32 & block, int index) {
    const std::uint8_t packed = block.qs[std::size_t(index & 15)];
    return index < 16 ? int(packed & 0x0fu) : int(packed >> 4);
}

inline int q4_code(const Q4K32 & block, int index) {
    return q4_storage(block, index) - 8;
}

inline void set_q4_code(Q4K32 & block, int index, int code) {
    const std::uint8_t storage = static_cast<std::uint8_t>(code + 8);
    std::uint8_t & packed = block.qs[std::size_t(index & 15)];
    if (index < 16) {
        packed = static_cast<std::uint8_t>((packed & 0xf0u) | storage);
    } else {
        packed = static_cast<std::uint8_t>(
            (packed & 0x0fu) | (storage << 4));
    }
}

inline float q4_authority_value(const Q4K32 & block, int index) {
    return float(q4_code(block, index)) * block.d;
}

inline float q8_authority_value(const Q8K32 & block, int index) {
    return float(block.qs[std::size_t(index)]) * block.d;
}

inline bool finite_float_from_double(double value, float & converted) {
    if (!std::isfinite(value)
            || std::fabs(value) > double(std::numeric_limits<float>::max())) {
        return false;
    }
    converted = static_cast<float>(value);
    return std::isfinite(converted);
}

inline double normalized_rmse(double squared_error, double energy) {
    if (!std::isfinite(squared_error) || !std::isfinite(energy)
            || squared_error < 0.0 || energy < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (energy == 0.0) {
        return squared_error == 0.0
            ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return std::sqrt(squared_error / energy);
}

struct WeightPool {
    bool valid = false;
    int begin = 0;
    int count = 0;
    float d_g = 0.0f;
    std::array<std::int8_t, kMaxGroupValues> r_w{};
    std::int32_t sum_q = 0;
    double squared_error = std::numeric_limits<double>::infinity();
    double energy = 0.0;
    double nrmse = std::numeric_limits<double>::infinity();
};

struct ActivationPool {
    bool valid = false;
    int begin = 0;
    int count = 0;
    float d_a = 0.0f;
    std::uint8_t z = 0;
    std::array<std::uint8_t, kMaxGroupValues> r_a{};
    double squared_error = std::numeric_limits<double>::infinity();
    double energy = 0.0;
    double nrmse = std::numeric_limits<double>::infinity();
};

inline std::array<double, kMaxGroupValues> gather_weights(
        const AuthorityRecord & record, int begin, int count) {
    std::array<double, kMaxGroupValues> values{};
    for (int item = 0; item < count; ++item) {
        const int global = begin + item;
        values[std::size_t(item)] = q4_authority_value(
            record.weights[std::size_t(global / kK32)], global % kK32);
    }
    return values;
}

inline std::array<double, kMaxGroupValues> gather_activations(
        const AuthorityRecord & record, int begin, int count) {
    std::array<double, kMaxGroupValues> values{};
    for (int item = 0; item < count; ++item) {
        const int global = begin + item;
        values[std::size_t(item)] = q8_authority_value(
            record.activations[std::size_t(global / kK32)], global % kK32);
    }
    return values;
}

inline WeightPool pool_weights_least_squares(
        const AuthorityRecord & record, int begin, int count) {
    WeightPool result;
    result.begin = begin;
    result.count = count;
    if ((count != kG64 && count != kG128) || begin < 0
            || begin > kG128 - count || begin % kK32 != 0) {
        return result;
    }
    // r_w is the authority signed Q4 code; it is never re-trained or made
    // activation-dependent. With r_w fixed, minimizing
    //   sum_j sum_i (d_j*q_ji - dG*q_ji)^2
    // has the unique closed form below, where S2_j=sum_i(q_ji^2).
    double weighted_scale = 0.0;
    double sum_s2 = 0.0;
    const int first_block = begin / kK32;
    const int block_count = count / kK32;
    int output = 0;
    for (int local_block = 0; local_block < block_count; ++local_block) {
        const Q4K32 & block = record.weights[
            std::size_t(first_block + local_block)];
        if (!std::isfinite(block.d)) {
            return result;
        }
        double block_s2 = 0.0;
        for (int index = 0; index < kK32; ++index) {
            const int code = q4_code(block, index);
            result.r_w[std::size_t(output++)] =
                static_cast<std::int8_t>(code);
            result.sum_q += code;
            block_s2 += double(code * code);
            const double value = double(block.d) * double(code);
            result.energy += value * value;
            if (!std::isfinite(result.energy)) {
                return result;
            }
        }
        weighted_scale += double(block.d) * block_s2;
        sum_s2 += block_s2;
        if (!std::isfinite(weighted_scale) || !std::isfinite(sum_s2)) {
            return result;
        }
    }
    if (sum_s2 == 0.0) {
        result.valid = true;
        result.d_g = 0.0f;
        result.squared_error = 0.0;
        result.nrmse = 0.0;
        return result;
    }
    if (!finite_float_from_double(weighted_scale / sum_s2, result.d_g)) {
        return result;
    }
    result.squared_error = 0.0;
    output = 0;
    for (int local_block = 0; local_block < block_count; ++local_block) {
        const Q4K32 & block = record.weights[
            std::size_t(first_block + local_block)];
        for (int index = 0; index < kK32; ++index) {
            const int code = int(result.r_w[std::size_t(output++)]);
            const double error = (double(block.d) - double(result.d_g))
                * double(code);
            result.squared_error += error * error;
            if (!std::isfinite(result.squared_error)) {
                return result;
            }
        }
    }
    result.nrmse = normalized_rmse(result.squared_error, result.energy);
    result.valid = std::isfinite(result.d_g)
        && std::isfinite(result.squared_error)
        && std::isfinite(result.energy) && std::isfinite(result.nrmse);
    return result;
}

// Fixed deployable unsigned-affine A4. Rounding is nearest with ties away from
// zero in the centered domain, followed by saturation to the u4 code range.
// Every comparison happens in double before the final, already-bounded cast.
inline bool round_centered_affine_u4(
        double value, double scale, int zero, int & code) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0
            || zero < 0 || zero > 15) {
        return false;
    }
    const double centered = value / scale;
    if (std::isnan(centered)) {
        return false;
    }
    if (centered == std::numeric_limits<double>::infinity()) {
        code = 15;
        return true;
    }
    if (centered == -std::numeric_limits<double>::infinity()) {
        code = 0;
        return true;
    }
    const double low = -double(zero);
    const double high = double(15 - zero);
    if (centered <= low - 0.5) {
        code = 0;
        return true;
    }
    if (centered >= high + 0.5) {
        code = 15;
        return true;
    }
    const double rounded = centered >= 0.0
        ? std::floor(centered + 0.5) : std::ceil(centered - 0.5);
    if (!std::isfinite(rounded) || rounded < low || rounded > high) {
        return false;
    }
    const int centered_code = static_cast<int>(rounded);
    code = centered_code + zero;
    return code >= 0 && code <= 15;
}

inline bool round_nonnegative_u4(double value, int & code) {
    if (!std::isfinite(value)) return false;
    if (value <= 0.0) {
        code = 0;
        return true;
    }
    if (value >= 15.0) {
        code = 15;
        return true;
    }
    const double rounded = std::floor(value + 0.5);
    if (!std::isfinite(rounded) || rounded < 0.0 || rounded > 15.0) {
        return false;
    }
    code = static_cast<int>(rounded);
    return true;
}

inline ActivationPool quantize_q8_authority_to_a4(
        const AuthorityRecord & record, int begin, int count) {
    ActivationPool result;
    result.begin = begin;
    result.count = count;
    if ((count != kG64 && count != kG128) || begin < 0
            || begin > kG128 - count || begin % kK32 != 0) {
        return result;
    }
    const auto values = gather_activations(record, begin, count);
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double energy = 0.0;
    for (int item = 0; item < count; ++item) {
        const double value = values[std::size_t(item)];
        if (!std::isfinite(value)) {
            return result;
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        energy += value * value;
        if (!std::isfinite(energy)) {
            return result;
        }
    }
    result.energy = energy;
    if (minimum == 0.0 && maximum == 0.0) {
        result.valid = true;
        result.d_a = 0.0f;
        result.z = 8;
        result.r_a.fill(result.z);
        result.squared_error = 0.0;
        result.nrmse = 0.0;
        return result;
    }

    const double range = maximum - minimum;
    if (!std::isfinite(range) || range <= 0.0) {
        return result;
    }
    const double scale = range / 15.0;
    if (!finite_float_from_double(scale, result.d_a)
            || result.d_a <= 0.0f) {
        return result;
    }
    const double deployed_scale = double(result.d_a);
    int zero = 0;
    if (!round_nonnegative_u4(-minimum / deployed_scale, zero)) {
        return result;
    }
    result.z = static_cast<std::uint8_t>(zero);
    result.squared_error = 0.0;
    for (int item = 0; item < count; ++item) {
        int code = 0;
        if (!round_centered_affine_u4(
                values[std::size_t(item)], deployed_scale, zero, code)) {
            return result;
        }
        result.r_a[std::size_t(item)] = static_cast<std::uint8_t>(code);
        const double reconstruction =
            deployed_scale * double(code - zero);
        const double error = values[std::size_t(item)] - reconstruction;
        result.squared_error += error * error;
        if (!std::isfinite(reconstruction) || !std::isfinite(error)
                || !std::isfinite(result.squared_error)) {
            return result;
        }
    }
    result.nrmse = normalized_rmse(result.squared_error, result.energy);
    result.valid = std::isfinite(result.d_a)
        && std::isfinite(result.squared_error)
        && std::isfinite(result.energy) && std::isfinite(result.nrmse);
    return result;
}

struct CheckedFloat {
    bool valid = false;
    float value = 0.0f;
};

inline CheckedFloat exact_q4_q8_dot(const AuthorityRecord & record) {
    CheckedFloat result;
    float accumulator = 0.0f;
    for (int block = 0; block < kBlocksPerRecord; ++block) {
        const Q4K32 & weight = record.weights[std::size_t(block)];
        const Q8K32 & activation = record.activations[std::size_t(block)];
        if (!std::isfinite(weight.d) || !std::isfinite(activation.d)) {
            return result;
        }
        std::int32_t integer_dot = 0;
        for (int index = 0; index < kK32; ++index) {
            integer_dot += q4_code(weight, index)
                * int(activation.qs[std::size_t(index)]);
        }
        float scale = 0.0f;
        if (!finite_float_from_double(
                double(weight.d) * double(activation.d), scale)) {
            return result;
        }
        accumulator = std::fma(float(integer_dot), scale, accumulator);
        if (!std::isfinite(accumulator)) {
            return result;
        }
    }
    result.valid = true;
    result.value = accumulator;
    return result;
}

struct SegmentApproximation {
    WeightPool weights;
    ActivationPool activations;
    std::int32_t raw_dot = 0;
    std::int32_t corrected_dot = 0;
    std::int32_t direct_centered_dot = 0;
    float scale = 0.0f;
    float value = 0.0f;
    bool finite = false;
    bool zero_point_identity = false;
    bool quality_pass = false;
};

inline SegmentApproximation approximate_segment(
        const AuthorityRecord & record, int begin, int count,
        const GateThresholds & thresholds) {
    SegmentApproximation result;
    result.weights = pool_weights_least_squares(record, begin, count);
    result.activations = quantize_q8_authority_to_a4(record, begin, count);
    if (!result.weights.valid || !result.activations.valid) {
        return result;
    }
    for (int item = 0; item < count; ++item) {
        const int w = int(result.weights.r_w[std::size_t(item)]);
        const int a = int(result.activations.r_a[std::size_t(item)]);
        result.raw_dot += w * a;
        result.direct_centered_dot += w * (a - int(result.activations.z));
    }
    result.corrected_dot = result.raw_dot
        - int(result.activations.z) * result.weights.sum_q;
    result.zero_point_identity =
        result.corrected_dot == result.direct_centered_dot;
    if (!finite_float_from_double(
            double(result.weights.d_g) * double(result.activations.d_a),
            result.scale)) {
        return result;
    }
    result.value = std::fma(float(result.corrected_dot), result.scale, 0.0f);
    if (!std::isfinite(result.value)) {
        return result;
    }
    result.finite = true;
    result.quality_pass = result.finite && result.zero_point_identity
        && std::isfinite(thresholds.weight_nrmse_max)
        && std::isfinite(thresholds.activation_nrmse_max)
        && result.weights.nrmse <= thresholds.weight_nrmse_max
        && result.activations.nrmse <= thresholds.activation_nrmse_max;
    return result;
}

enum class LadderLevel : std::uint8_t {
    G128,
    G64,
    Exact,
};

inline const char * ladder_level_name(LadderLevel level) {
    switch (level) {
        case LadderLevel::G128: return "g128";
        case LadderLevel::G64: return "g64";
        case LadderLevel::Exact: return "exact";
    }
    return "invalid";
}

struct LadderResult {
    bool valid = false;
    LadderLevel level = LadderLevel::Exact;
    float exact = 0.0f;
    float selected = 0.0f;
    float g128_value = 0.0f;
    float g64_value = 0.0f;
    SegmentApproximation g128;
    std::array<SegmentApproximation, 2> g64{};
};

inline LadderResult evaluate_ladder(
        const AuthorityRecord & record,
        const GateThresholds & thresholds = kSyntheticGateThresholds) {
    LadderResult result;
    const CheckedFloat exact = exact_q4_q8_dot(record);
    if (!exact.valid) {
        return result;
    }
    result.exact = exact.value;
    result.g128 = approximate_segment(record, 0, kG128, thresholds);
    result.g128_value = result.g128.value;
    result.g64[0] = approximate_segment(record, 0, kG64, thresholds);
    result.g64[1] = approximate_segment(record, kG64, kG64, thresholds);
    bool g64_finite = result.g64[0].finite && result.g64[1].finite;
    if (g64_finite) {
        result.g64_value = std::fma(float(result.g64[0].corrected_dot),
            result.g64[0].scale, 0.0f);
        result.g64_value = std::fma(float(result.g64[1].corrected_dot),
            result.g64[1].scale, result.g64_value);
        g64_finite = std::isfinite(result.g64_value);
    }
    if (result.g128.quality_pass) {
        result.level = LadderLevel::G128;
        result.selected = result.g128_value;
    } else if (g64_finite && result.g64[0].quality_pass
            && result.g64[1].quality_pass) {
        result.level = LadderLevel::G64;
        result.selected = result.g64_value;
    } else {
        result.level = LadderLevel::Exact;
        result.selected = result.exact;
    }
    result.valid = std::isfinite(result.exact)
        && std::isfinite(result.selected);
    return result;
}

// Versioned, fail-closed capture envelope. The v1 payload record semantics are
// intentionally opaque until a real capture producer is audited. The reader
// validates transport and identity but never interprets bytes as Q4/A4.
constexpr std::array<std::uint8_t, 8> kCaptureMagic = {
    'I', 'M', 'W', '4', 'A', '4', 'G', '0'};
constexpr std::uint16_t kCaptureVersion = 1;
constexpr std::uint16_t kCaptureHeaderBytes = 128;
constexpr std::uint32_t kCaptureEndianTag = 0x01020304u;
constexpr std::uint32_t kCaptureOpaqueSchemaV1 = 1;
// The provisional reader buffers the envelope, so cap it well below 1 GiB.
// A production schema reader must stream records instead of raising this cap.
constexpr std::uint64_t kCaptureMaxBytes = UINT64_C(64) << 20;
constexpr std::uint64_t kCaptureMaxRecords = UINT64_C(1) << 28;

class Sha256 {
public:
    void update(const std::uint8_t * data, std::size_t size) {
        for (std::size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            ++total_bytes_;
            if (buffer_size_ == 64) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bit_length = total_bytes_ * 8;
        buffer_[buffer_size_++] = 0x80u;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) buffer_[buffer_size_++] = 0;
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) buffer_[buffer_size_++] = 0;
        for (int byte = 7; byte >= 0; --byte) {
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(
                bit_length >> (byte * 8));
        }
        transform(buffer_.data());
        std::array<std::uint8_t, 32> digest{};
        for (std::size_t word = 0; word < state_.size(); ++word) {
            for (int byte = 0; byte < 4; ++byte) {
                digest[word * 4 + std::size_t(byte)] =
                    static_cast<std::uint8_t>(
                        state_[word] >> (24 - byte * 8));
            }
        }
        return digest;
    }

private:
    static std::uint32_t rotate_right(std::uint32_t value, int bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void transform(const std::uint8_t * block) {
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        std::array<std::uint32_t, 64> words{};
        for (int word = 0; word < 16; ++word) {
            words[std::size_t(word)] =
                (std::uint32_t(block[word * 4]) << 24)
                | (std::uint32_t(block[word * 4 + 1]) << 16)
                | (std::uint32_t(block[word * 4 + 2]) << 8)
                | std::uint32_t(block[word * 4 + 3]);
        }
        for (int word = 16; word < 64; ++word) {
            const std::uint32_t x = words[std::size_t(word - 15)];
            const std::uint32_t y = words[std::size_t(word - 2)];
            const std::uint32_t s0 = rotate_right(x, 7)
                ^ rotate_right(x, 18) ^ (x >> 3);
            const std::uint32_t s1 = rotate_right(y, 17)
                ^ rotate_right(y, 19) ^ (y >> 10);
            words[std::size_t(word)] = words[std::size_t(word - 16)] + s0
                + words[std::size_t(word - 7)] + s1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (int round = 0; round < 64; ++round) {
            const std::uint32_t s1 = rotate_right(e, 6)
                ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choose
                + k[std::size_t(round)] + words[std::size_t(round)];
            const std::uint32_t s0 = rotate_right(a, 2)
                ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

inline std::array<std::uint8_t, 32> sha256(
        const std::uint8_t * data, std::size_t size) {
    Sha256 hasher;
    hasher.update(data, size);
    return hasher.finish();
}

// Canonical envelope identity: all interpreted header bytes plus payload, with
// the embedded digest field [96,128) treated as zero. This prevents a valid
// payload digest from being replayed under different shape/section metadata.
inline std::array<std::uint8_t, 32> capture_identity_sha256(
        const std::uint8_t * bytes, std::size_t size) {
    if (bytes == nullptr || size < kCaptureHeaderBytes) {
        return {};
    }
    Sha256 hasher;
    hasher.update(bytes, 96);
    const std::array<std::uint8_t, 32> zero_digest{};
    hasher.update(zero_digest.data(), zero_digest.size());
    hasher.update(bytes + kCaptureHeaderBytes, size - kCaptureHeaderBytes);
    return hasher.finish();
}

inline std::uint16_t read_le16(const std::uint8_t * data) {
    return std::uint16_t(data[0]) | (std::uint16_t(data[1]) << 8);
}

inline std::uint32_t read_le32(const std::uint8_t * data) {
    return std::uint32_t(data[0]) | (std::uint32_t(data[1]) << 8)
        | (std::uint32_t(data[2]) << 16) | (std::uint32_t(data[3]) << 24);
}

inline std::uint64_t read_le64(const std::uint8_t * data) {
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
        value |= std::uint64_t(data[byte]) << (byte * 8);
    }
    return value;
}

inline bool checked_add(std::uint64_t lhs, std::uint64_t rhs,
        std::uint64_t & result) {
    if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) return false;
    result = lhs + rhs;
    return true;
}

inline bool checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
        std::uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

enum class CaptureError : std::uint8_t {
    None, NullInput, Truncated, TooLarge, BadMagic, BadVersion, BadEndianness,
    BadHeaderBytes, UnsupportedSchema, BadShape, BadRecordCount,
    BadByteLengths, ArithmeticOverflow, SizeMismatch,
    MissingExpectedFingerprint, FingerprintMismatch,
};

inline const char * capture_error_name(CaptureError error) {
    switch (error) {
        case CaptureError::None: return "none";
        case CaptureError::NullInput: return "null_input";
        case CaptureError::Truncated: return "truncated";
        case CaptureError::TooLarge: return "too_large";
        case CaptureError::BadMagic: return "bad_magic";
        case CaptureError::BadVersion: return "bad_version";
        case CaptureError::BadEndianness: return "bad_endianness";
        case CaptureError::BadHeaderBytes: return "bad_header_bytes";
        case CaptureError::UnsupportedSchema: return "unsupported_schema";
        case CaptureError::BadShape: return "bad_shape";
        case CaptureError::BadRecordCount: return "bad_record_count";
        case CaptureError::BadByteLengths: return "bad_byte_lengths";
        case CaptureError::ArithmeticOverflow: return "arithmetic_overflow";
        case CaptureError::SizeMismatch: return "size_mismatch";
        case CaptureError::MissingExpectedFingerprint:
            return "missing_expected_fingerprint";
        case CaptureError::FingerprintMismatch: return "fingerprint_mismatch";
    }
    return "invalid";
}

struct CaptureHeader {
    std::uint16_t version = 0;
    std::uint32_t schema = 0;
    std::uint32_t rank = 0;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint64_t record_count = 0;
    std::uint64_t record_bytes = 0;
    std::uint64_t q4_bytes = 0;
    std::uint64_t q8_bytes = 0;
    std::uint64_t auxiliary_bytes = 0;
    std::array<std::uint8_t, 32> identity_sha256{};
};

struct CaptureValidation {
    CaptureError error = CaptureError::None;
    CaptureHeader header{};
};

inline CaptureValidation validate_capture_envelope(
        const std::uint8_t * bytes, std::size_t size,
        const std::array<std::uint8_t, 32> * expected_fingerprint) {
    CaptureValidation result;
    if (bytes == nullptr) {
        result.error = CaptureError::NullInput;
        return result;
    }
    if (size < kCaptureHeaderBytes) {
        result.error = CaptureError::Truncated;
        return result;
    }
    if (std::uint64_t(size) > kCaptureMaxBytes) {
        result.error = CaptureError::TooLarge;
        return result;
    }
    if (!std::equal(kCaptureMagic.begin(), kCaptureMagic.end(), bytes)) {
        result.error = CaptureError::BadMagic;
        return result;
    }
    result.header.version = read_le16(bytes + 8);
    if (result.header.version != kCaptureVersion) {
        result.error = CaptureError::BadVersion;
        return result;
    }
    if (read_le16(bytes + 10) != kCaptureHeaderBytes) {
        result.error = CaptureError::BadHeaderBytes;
        return result;
    }
    if (read_le32(bytes + 12) != kCaptureEndianTag) {
        result.error = CaptureError::BadEndianness;
        return result;
    }
    result.header.schema = read_le32(bytes + 16);
    if (result.header.schema != kCaptureOpaqueSchemaV1) {
        result.error = CaptureError::UnsupportedSchema;
        return result;
    }
    result.header.rank = read_le32(bytes + 20);
    if (result.header.rank == 0 || result.header.rank > 4) {
        result.error = CaptureError::BadShape;
        return result;
    }
    std::uint64_t shape_product = 1;
    for (std::size_t index = 0; index < 4; ++index) {
        result.header.dimensions[index] = read_le64(bytes + 24 + index * 8);
        const bool active = index < result.header.rank;
        if ((active && result.header.dimensions[index] == 0)
                || (!active && result.header.dimensions[index] != 0)) {
            result.error = CaptureError::BadShape;
            return result;
        }
        if (active && !checked_multiply(shape_product,
                result.header.dimensions[index], shape_product)) {
            result.error = CaptureError::ArithmeticOverflow;
            return result;
        }
    }
    result.header.record_count = read_le64(bytes + 56);
    result.header.record_bytes = read_le64(bytes + 64);
    result.header.q4_bytes = read_le64(bytes + 72);
    result.header.q8_bytes = read_le64(bytes + 80);
    result.header.auxiliary_bytes = read_le64(bytes + 88);
    std::copy(bytes + 96, bytes + 128,
        result.header.identity_sha256.begin());
    if (result.header.record_count == 0
            || result.header.record_count > kCaptureMaxRecords
            || shape_product != result.header.record_count) {
        result.error = CaptureError::BadRecordCount;
        return result;
    }
    if (result.header.record_bytes == 0 || result.header.q4_bytes == 0
            || result.header.q8_bytes == 0) {
        result.error = CaptureError::BadByteLengths;
        return result;
    }
    std::uint64_t payload_bytes = 0;
    if (!checked_add(result.header.q4_bytes, result.header.q8_bytes,
            payload_bytes)
            || !checked_add(payload_bytes, result.header.auxiliary_bytes,
                payload_bytes)) {
        result.error = CaptureError::ArithmeticOverflow;
        return result;
    }
    std::uint64_t records_bytes = 0;
    if (!checked_multiply(result.header.record_count,
            result.header.record_bytes, records_bytes)) {
        result.error = CaptureError::ArithmeticOverflow;
        return result;
    }
    if (records_bytes != payload_bytes || payload_bytes > kCaptureMaxBytes) {
        result.error = CaptureError::BadByteLengths;
        return result;
    }
    std::uint64_t total_bytes = 0;
    if (!checked_add(kCaptureHeaderBytes, payload_bytes, total_bytes)) {
        result.error = CaptureError::ArithmeticOverflow;
        return result;
    }
    if (total_bytes != std::uint64_t(size)) {
        result.error = CaptureError::SizeMismatch;
        return result;
    }
    if (expected_fingerprint == nullptr) {
        result.error = CaptureError::MissingExpectedFingerprint;
        return result;
    }
    const bool fingerprint_nonzero = std::any_of(
        result.header.identity_sha256.begin(),
        result.header.identity_sha256.end(),
        [](std::uint8_t value) { return value != 0; });
    if (!fingerprint_nonzero
            || result.header.identity_sha256 != *expected_fingerprint) {
        result.error = CaptureError::FingerprintMismatch;
        return result;
    }
    const auto actual = capture_identity_sha256(bytes, size);
    if (actual != result.header.identity_sha256) {
        result.error = CaptureError::FingerprintMismatch;
        return result;
    }
    result.error = CaptureError::None;
    return result;
}

inline std::string bytes_to_hex(
        const std::array<std::uint8_t, 32> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const std::uint8_t value : bytes) {
        result.push_back(digits[value >> 4]);
        result.push_back(digits[value & 15]);
    }
    return result;
}

inline bool hex_to_fingerprint(const std::string & text,
        std::array<std::uint8_t, 32> & fingerprint) {
    if (text.size() != 64) return false;
    auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        if (character >= 'A' && character <= 'F') return character - 'A' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < fingerprint.size(); ++index) {
        const int high = nibble(text[index * 2]);
        const int low = nibble(text[index * 2 + 1]);
        if (high < 0 || low < 0) return false;
        fingerprint[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

}  // namespace imparo_sm86_w4a4_gate0
