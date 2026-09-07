#pragma once

#include <cstdint>

// Host/device-neutral launch planning for the D64/GQA4 whole-K prefill family.
// The common plan owns semantic tiling. SM-family headers own occupancy and
// device-selection policy, so adding a later architecture never changes this ABI.
namespace imparo_sm80_d64_mma_plan {

constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kQueryTokens = 16;
constexpr uint32_t kKeysPerUpdate = 64;
constexpr uint32_t kThreads = 128;

struct Plan {
    uint32_t query_tiles = 0;
    uint32_t key_updates = 0;
    uint32_t blocks = 0;
    bool valid = false;
};

constexpr Plan make(uint32_t n_tok, uint32_t n_kv, uint32_t valid_span) {
    if (n_tok == 0 || n_kv == 0 || valid_span == 0) return {};
    const uint64_t query_tiles =
        (uint64_t(n_tok) + kQueryTokens - 1) / kQueryTokens;
    const uint64_t key_updates =
        (uint64_t(valid_span) + kKeysPerUpdate - 1) / kKeysPerUpdate;
    const uint64_t blocks = query_tiles * n_kv;
    if (query_tiles > UINT32_MAX || key_updates > UINT32_MAX
        || blocks > UINT32_MAX) {
        return {};
    }
    return {
        uint32_t(query_tiles), uint32_t(key_updates), uint32_t(blocks), true,
    };
}

static_assert(make(512, 8, 512).query_tiles == 32);
static_assert(make(512, 8, 512).key_updates == 8);
static_assert(make(512, 8, 512).blocks == 256);
static_assert(make(464, 8, 2000).blocks == 232);

} // namespace imparo_sm80_d64_mma_plan
