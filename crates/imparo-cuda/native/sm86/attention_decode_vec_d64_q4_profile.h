#pragma once

#include <cstdint>

// Pinned RTX 3060 oracle profile: the D64/Q4 vector kernel uses 144 registers
// and 2304 shared bytes per 128-thread block, yielding three resident blocks.
namespace imparo_sm86_d64_q4_vec_profile {

constexpr uint32_t kSmVersion = 86;
constexpr uint32_t kMaxBlocksPerSm = 3;

constexpr bool applies_to(uint32_t sm_version) {
    return sm_version == kSmVersion;
}

} // namespace imparo_sm86_d64_q4_vec_profile
