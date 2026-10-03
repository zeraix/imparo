#include "../dspark_confidence_host.h"
#include <array>
#include <cstdio>
#include <limits>

namespace {
// Independent IEEE binary16 decoder for the portable CPU contract test.
float decode_half(std::uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    float value = exponent == 31
        ? (fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
        : exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                   : std::ldexp(float(fraction), -24);
    return bits & 0x8000 ? -value : value;
}
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F fn) {
    bool caught = false;
    try { fn(); } catch (const std::runtime_error&) { caught = true; }
    check(caught, "invalid input was accepted");
}
void half(std::uint8_t* at, std::uint16_t bits) { std::memcpy(at, &bits, 2); }
}

int main() {
    try {
        using namespace imparo_dspark_confidence;
        std::array<std::uint8_t, 68> table{};
        half(table.data(), 0x3400);       // Parent 0: +0.25.
        half(table.data() + 34, 0xb800); // Parent 1: -0.5.
        for (unsigned base : {0u, 34u}) {
            table[base + 2] = 0x80;      // Signed -128, never +128.
            table[base + 3] = 0x7f;      // Signed +127.
            table[base + 33] = 2;
        }
        std::array<float, 32> rank{};
        parent_rank(table.data(), table.size(), 0, 32, rank.data(), decode_half);
        check(rank[0] == -32.f && rank[1] == 31.75f && rank[31] == .5f, "positive Q8 scale/extremes");
        parent_rank(table.data(), table.size(), 1, 32, rank.data(), decode_half);
        check(rank[0] == 64.f && rank[1] == -63.5f && rank[31] == -1.f, "negative Q8 scale/parent selection");

        std::array<std::uint8_t, 68> weights{}; // H=2, R=32, F16 head.
        half(weights.data(), 0x3c00);     // Hidden coefficient +1.
        half(weights.data() + 2, 0xc000);// Hidden coefficient -2.
        half(weights.data() + 4, 0x3800);// Rank 0 coefficient +0.5.
        half(weights.data() + 6, 0xb400);// Rank 1 coefficient -0.25.
        half(weights.data() + 66, 0x3c00);// Rank 31 coefficient +1.
        std::array<float, 4> hidden{{-100.f, 100.f, 3.f, 5.f}};
        // Independently hand-computed logit:
        // -39.375 + 3 - 2*5 + .5*64 - .25*(-63.5) - 1 = 0.5.
        const float actual = at_depth(hidden.data(), 2, 2, 1, rank.data(), 32,
                                      weights.data(), -39.375f, decode_half);
        check(std::abs(double(actual) - 0.6224593312018545646) < 3e-8,
              "real parent and depth confidence differs from independent reference");
        check(at_depth(hidden.data(), 2, 2, 0, rank.data(), 32,
                       weights.data(), -39.375f, decode_half) < 1e-20f,
              "hidden depth was ignored");

        rank.fill(0.f);
        std::array<float, 2> saturated{{1000.f, 0.f}};
        check(head(saturated.data(), rank.data(), weights.data(), 0.f, 2, 32, decode_half) == 1.f,
              "positive sigmoid saturation");
        saturated[0] = -1000.f;
        check(head(saturated.data(), rank.data(), weights.data(), 0.f, 2, 32, decode_half) == 0.f,
              "negative sigmoid saturation");
        rejects([&] { parent_rank(table.data(), table.size(), 2, 32, rank.data(), decode_half); });
        rejects([&] { parent_rank(table.data(), table.size()-1, 1, 32, rank.data(), decode_half); });
        rejects([&] { at_depth(hidden.data(), 2, 2, 2, rank.data(), 32, weights.data(), 0.f, decode_half); });
        rejects([&] { head(hidden.data(), rank.data(), weights.data(), std::numeric_limits<float>::infinity(), 2, 32, decode_half); });
        hidden[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { head(hidden.data(), rank.data(), weights.data(), 0.f, 2, 32, decode_half); });
        hidden[0] = 0.f; rank[0] = std::numeric_limits<float>::infinity();
        rejects([&] { head(hidden.data(), rank.data(), weights.data(), 0.f, 2, 32, decode_half); });
        rank[0] = 0.f; half(weights.data(), 0x7e00);
        rejects([&] { head(hidden.data(), rank.data(), weights.data(), 0.f, 2, 32, decode_half); });
        half(table.data(), 0x7e00);
        rejects([&] { parent_rank(table.data(), table.size(), 0, 32, rank.data(), decode_half); });
        std::puts("PASS: parent Q8 decode, depth/head reference, saturation and nonfinite/range rejection");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
