#pragma once

#include <cstdint>

namespace imparo_sm80_d64_q4_vec_plan {

constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kWarps = 4;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kKeysPerStripe = 128;
constexpr uint32_t kKvTile = 64;
constexpr uint32_t kScheduleQuantum = 256;
constexpr uint32_t kPartialStride = kHeadDim + 2;

struct Geometry {
    uint32_t capacity = 0;
    uint32_t valid_span = 0;
    uint32_t schedule_span = 0;
    uint32_t logical_base = 0;
    uint32_t stripes = 0;
    uint32_t kv_tiles = 0;
    uint32_t parts = 0;
};

#if defined(__CUDACC__)
#define IMPARO_D64_Q4_VEC_HD __host__ __device__
#else
#define IMPARO_D64_Q4_VEC_HD
#endif

IMPARO_D64_Q4_VEC_HD constexpr uint32_t min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

// The selection loop matches pinned non-Stream-K launch_fattn for caller-supplied
// kv_tiles and occupancy. geometry() deliberately supplies graph-bucket-padded
// kv_tiles so captured launches stay stable; it is not an all-length claim that
// the pinned host scheduler uses the same padding. Empty y partitions are valid.
IMPARO_D64_Q4_VEC_HD constexpr uint32_t parallel_blocks(
        uint32_t kv_tiles, uint32_t tiles_dst, uint32_t sm_count,
        uint32_t max_blocks_per_sm) {
    if (!kv_tiles || !tiles_dst || !sm_count || !max_blocks_per_sm) return 0;
    uint32_t selected = min_u32(max_blocks_per_sm, kv_tiles);
    const uint64_t blocks_per_wave = uint64_t(sm_count) * max_blocks_per_sm;
    uint64_t best_waves = 0;
    uint32_t best_efficiency = 0;
    for (uint32_t test = selected; test <= kv_tiles; ++test) {
        const uint64_t total = uint64_t(tiles_dst) * test;
        const uint64_t waves = (total + blocks_per_wave - 1) / blocks_per_wave;
        const uint32_t efficiency = uint32_t(
            100 * total / (waves * blocks_per_wave));
        if (best_efficiency >= 95 && waves > best_waves) break;
        if (efficiency > best_efficiency) {
            best_waves = waves;
            best_efficiency = efficiency;
            selected = test;
        }
    }
    return selected;
}

IMPARO_D64_Q4_VEC_HD constexpr bool geometry(
        uint32_t start_pos, uint32_t ring, uint32_t sm_count,
        uint32_t max_blocks_per_sm, uint32_t tiles_dst,
        uint32_t diagnostic_part_cap, Geometry * out) {
    if (!out || start_pos == UINT32_MAX) return false;
    const uint32_t initialized = start_pos + 1;
    const uint32_t capacity = ring ? ring + 1 : 0;
    if (ring && (!capacity || (capacity & (capacity - 1)) != 0)) return false;
    const uint32_t valid_span = ring ? min_u32(initialized, capacity) : initialized;
    const uint32_t logical_base = ring && initialized > capacity
        ? initialized - capacity : 0;
    const uint64_t rounded = (uint64_t(valid_span) + kScheduleQuantum - 1)
        / kScheduleQuantum * kScheduleQuantum;
    if (rounded > UINT32_MAX) return false;
    uint32_t schedule_span = uint32_t(rounded);
    if (ring && schedule_span > capacity) schedule_span = capacity;
    if (!schedule_span || schedule_span < valid_span) return false;
    const uint32_t stripes = uint32_t(
        (uint64_t(schedule_span) + kKeysPerStripe - 1) / kKeysPerStripe);
    const uint32_t kv_tiles = uint32_t(
        (uint64_t(schedule_span) + kKvTile - 1) / kKvTile);
    uint32_t parts = parallel_blocks(
        kv_tiles, tiles_dst, sm_count, max_blocks_per_sm);
    if (diagnostic_part_cap) parts = min_u32(parts, diagnostic_part_cap);
    if (!parts) return false;
    *out = Geometry{capacity, valid_span, schedule_span, logical_base,
                    stripes, kv_tiles, parts};
    return true;
}

IMPARO_D64_Q4_VEC_HD constexpr uint32_t stripe_begin(
        uint32_t part, uint32_t stripe_round, uint32_t parts) {
    return (part + stripe_round * parts) * kKeysPerStripe;
}

IMPARO_D64_Q4_VEC_HD constexpr uint32_t graph_bucket_min(
        const Geometry & value) {
    return value.schedule_span > kScheduleQuantum
        ? value.schedule_span - kScheduleQuantum : 0;
}

IMPARO_D64_Q4_VEC_HD constexpr uint32_t graph_bucket_max(
        const Geometry & value) {
    if (value.capacity && value.schedule_span == value.capacity) return UINT32_MAX;
    return value.schedule_span - 1;
}

IMPARO_D64_Q4_VEC_HD constexpr bool active_mapping_is_exact_partition(
        uint32_t valid_span, uint32_t schedule_span, uint32_t parts) {
    if (!valid_span || valid_span > schedule_span || !parts) return false;
    const uint32_t stripes = uint32_t(
        (uint64_t(schedule_span) + kKeysPerStripe - 1) / kKeysPerStripe);
    for (uint32_t key = 0; key < valid_span; ++key) {
        uint32_t owners = 0;
        for (uint32_t part = 0; part < parts; ++part) {
            for (uint32_t round = 0; part + round * parts < stripes; ++round) {
                const uint32_t begin = stripe_begin(part, round, parts);
                owners += key >= begin && key < begin + kKeysPerStripe;
            }
        }
        if (owners != 1) return false;
    }
    return true;
}

static_assert(active_mapping_is_exact_partition(2001, 2048, 11),
    "P11 graph-bucket mapping assigns every active n2001 key once");

#undef IMPARO_D64_Q4_VEC_HD

} // namespace imparo_sm80_d64_q4_vec_plan
