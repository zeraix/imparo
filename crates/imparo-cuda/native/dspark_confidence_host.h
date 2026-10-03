#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// CUDA-independent host arithmetic. The caller supplies the existing F16
// conversion, so the ordinary chain and frontier use the same confidence head.
namespace imparo_dspark_confidence {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class HalfToFloat>
float half_at(const std::uint8_t* bytes, std::size_t index, HalfToFloat convert) {
    std::uint16_t bits;
    std::memcpy(&bits, bytes + index * 2, sizeof(bits));
    return convert(bits);
}

template<class HalfToFloat>
void parent_rank(const std::uint8_t* table, std::size_t bytes,
                 std::uint32_t token, unsigned rank, float* out,
                 HalfToFloat convert) {
    require(table && out && rank && rank % 32 == 0, "confidence rank shape");
    const std::uint64_t row_bytes = std::uint64_t(rank / 32) * 34;
    require(row_bytes <= bytes && token <= (bytes - row_bytes) / row_bytes,
            "confidence parent row range");
    const std::uint64_t offset = std::uint64_t(token) * row_bytes;
    const auto* row = table + offset;
    for (unsigned j = 0; j < rank; ++j) {
        const auto* block = row + std::uint64_t(j / 32) * 34;
        const float scale = half_at(block, 0, convert);
        std::int8_t value;
        std::memcpy(&value, block + 2 + j % 32, sizeof(value));
        out[j] = float(value) * scale;
        require(std::isfinite(out[j]), "confidence nonfinite rank");
    }
}

template<class HalfToFloat>
float head(const float* hidden, const float* rank, const std::uint8_t* weights,
           float bias, unsigned h, unsigned r, HalfToFloat convert) {
    require(hidden && rank && weights && std::isfinite(bias), "confidence head inputs");
    double x = bias;
    for (unsigned j = 0; j < h; ++j) {
        const float w = half_at(weights, j, convert);
        require(std::isfinite(hidden[j]) && std::isfinite(w), "confidence nonfinite hidden/head");
        x += double(hidden[j]) * w;
    }
    for (unsigned j = 0; j < r; ++j) {
        const float w = half_at(weights, std::size_t(h) + j, convert);
        require(std::isfinite(rank[j]) && std::isfinite(w), "confidence nonfinite rank/head");
        x += double(rank[j]) * w;
    }
    require(std::isfinite(x), "confidence nonfinite logit");
    return float(1.0 / (1.0 + std::exp(-x)));
}

template<class HalfToFloat>
float at_depth(const float* hidden, unsigned hidden_rows, unsigned h,
               unsigned depth, const float* rank, unsigned r,
               const std::uint8_t* weights, float bias, HalfToFloat convert) {
    require(hidden && depth < hidden_rows, "confidence hidden depth");
    return head(hidden + std::size_t(depth) * h, rank, weights, bias, h, r, convert);
}
}
