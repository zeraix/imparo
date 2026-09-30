#pragma once

#include <cstdint>

namespace imparo_sm80_d256_vec_plan {

constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kWarps = 4;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kWindowSpan = 512;
constexpr uint32_t kScheduleQuantum = 256;
constexpr uint32_t kKeysPerPartialTile = 128;
constexpr uint32_t kStripesPerPartial = 2;
constexpr uint32_t kMaxPartials = 4;
constexpr uint32_t kMaxPhysicalSpan = kMaxPartials * kScheduleQuantum;

struct Geometry {
    uint32_t capacity = 0;
    uint32_t valid_span = 0;
    uint32_t schedule_span = 0;
    uint32_t partials = 0;
};

#if defined(__CUDACC__)
#define IMPARO_D256_PLAN_HD __host__ __device__
#else
#define IMPARO_D256_PLAN_HD
#endif

IMPARO_D256_PLAN_HD constexpr bool is_power_of_two(uint32_t value) {
    return value && (value & (value - 1)) == 0;
}

IMPARO_D256_PLAN_HD constexpr uint32_t partial_count(uint32_t schedule_span) {
    return (schedule_span + kScheduleQuantum - 1) / kScheduleQuantum;
}

IMPARO_D256_PLAN_HD constexpr uint32_t partial_stripe_begin(
        uint32_t partial, uint32_t stripe, uint32_t partials) {
    return (partial + stripe * partials) * kKeysPerPartialTile;
}

IMPARO_D256_PLAN_HD constexpr bool geometry(
        uint32_t start_pos, uint32_t n_tok, uint32_t ring, Geometry * out) {
    if (!out || !n_tok || ring == UINT32_MAX) return false;
    const uint32_t capacity = ring + 1;
    if (!is_power_of_two(capacity) || capacity < kWindowSpan
        || capacity > kMaxPhysicalSpan
        || capacity % kScheduleQuantum != 0) return false;
    const uint64_t initialized = uint64_t(start_pos) + n_tok;
    if (initialized > UINT32_MAX) return false;
    const uint32_t valid_span = initialized < capacity
        ? uint32_t(initialized) : capacity;
    const uint64_t rounded = (uint64_t(valid_span) + kScheduleQuantum - 1)
        / kScheduleQuantum * kScheduleQuantum;
    const uint32_t schedule_span = rounded < capacity
        ? uint32_t(rounded) : capacity;
    const uint32_t partials = partial_count(schedule_span);
    if (!schedule_span || partials == 0 || partials > kMaxPartials
        || schedule_span != partials * kScheduleQuantum) return false;
    *out = Geometry{capacity, valid_span, schedule_span, partials};
    return true;
}

IMPARO_D256_PLAN_HD constexpr uint32_t graph_bucket_min(
        const Geometry & value) {
    const uint32_t schedule_min = value.schedule_span > kScheduleQuantum
        ? value.schedule_span - kScheduleQuantum : 0;
    const uint32_t route_min = kWindowSpan - 1;
    return schedule_min > route_min ? schedule_min : route_min;
}

IMPARO_D256_PLAN_HD constexpr uint32_t graph_bucket_max(
        const Geometry & value) {
    return value.schedule_span < value.capacity
        ? value.schedule_span - 1 : UINT32_MAX;
}

// The verifier captures up to three causal rows, each with its own M1 plan.
// Bound replay by the earliest row whose fixed partial geometry would change.
IMPARO_D256_PLAN_HD constexpr bool verification_bucket_max(
        uint32_t start, uint32_t row, uint32_t ring,
        uint32_t captured_span, uint32_t captured_partials, uint32_t * upper) {
    if (!upper || row > 2 || start > UINT32_MAX - 2) return false;
    Geometry value;
    if (!geometry(start + row, 1, ring, &value)
        || start + row < graph_bucket_min(value)
        || captured_span != value.schedule_span
        || captured_partials != value.partials) return false;
    const uint32_t limit = graph_bucket_max(value);
    *upper = limit == UINT32_MAX ? UINT32_MAX - 2 : limit - row;
    return *upper >= start;
}

// A shared three-query kernel may reuse a graph only while all original M1
// rows have the captured geometry. In particular P3 ends at start765, not767.
IMPARO_D256_PLAN_HD constexpr bool shared_verification_bucket_max(
        uint32_t start, uint32_t ring, uint32_t span, uint32_t partials,
        uint32_t * upper) {
    if (!upper || start > UINT32_MAX - 3) return false;
    uint32_t limit = UINT32_MAX - 3;
    for (uint32_t row = 0; row < 3; ++row) {
        uint32_t row_limit = 0;
        if (!verification_bucket_max(start, row, ring, span, partials, &row_limit))
            return false;
        if (row_limit < limit) limit = row_limit;
    }
    *upper = limit;
    return true;
}

IMPARO_D256_PLAN_HD constexpr bool mapping_is_exact_partition(
        uint32_t schedule_span, uint32_t partials) {
    if (!partials || partials > kMaxPartials
        || schedule_span != partials * kScheduleQuantum) return false;
    for (uint32_t key = 0; key < schedule_span; ++key) {
        uint32_t owners = 0;
        for (uint32_t partial = 0; partial < partials; ++partial) {
            for (uint32_t stripe = 0; stripe < kStripesPerPartial; ++stripe) {
                const uint32_t begin = partial_stripe_begin(
                    partial, stripe, partials);
                owners += key >= begin
                    && key < begin + kKeysPerPartialTile;
            }
        }
        if (owners != 1) return false;
    }
    return true;
}

static_assert(mapping_is_exact_partition(512, 2), "P2 must partition 512 keys");
static_assert(mapping_is_exact_partition(768, 3), "P3 must partition 768 keys");
static_assert(mapping_is_exact_partition(1024, 4), "P4 must partition 1024 keys");

#undef IMPARO_D256_PLAN_HD

} // namespace imparo_sm80_d256_vec_plan
