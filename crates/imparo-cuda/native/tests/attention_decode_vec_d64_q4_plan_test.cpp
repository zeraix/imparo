#include "../sm80/attention_decode_vec_d64_q4_plan.h"
#include "../sm86/attention_decode_vec_d64_q4_profile.h"

#include <cassert>
#include <cstdint>

namespace plan = imparo_sm80_d64_q4_vec_plan;
namespace profile = imparo_sm86_d64_q4_vec_profile;

static void expect_full(uint32_t start, uint32_t schedule,
                        uint32_t stripes, uint32_t kv_tiles,
                        uint32_t parts) {
    plan::Geometry value;
    assert(plan::geometry(
        start, 0, 30, profile::kMaxBlocksPerSm, 32, 0, &value));
    assert(value.valid_span == start + 1);
    assert(value.logical_base == 0);
    assert(value.schedule_span == schedule);
    assert(value.stripes == stripes);
    assert(value.kv_tiles == kv_tiles);
    assert(value.parts == parts);
    assert(plan::active_mapping_is_exact_partition(
        start + 1, schedule, parts));
    assert(plan::graph_bucket_min(value) + plan::kScheduleQuantum == schedule);
    assert(plan::graph_bucket_max(value) == schedule - 1);
}

int main() {
    assert(profile::applies_to(86));
    assert(!profile::applies_to(80));
    assert(profile::kMaxBlocksPerSm == 3);
    assert(plan::parallel_blocks(
        32, 32, 30, profile::kMaxBlocksPerSm) == 11);
    assert(plan::parallel_blocks(32, 32, 30, 2) != 11);
    assert(plan::active_mapping_is_exact_partition(2001, 2048, 11));
    expect_full(0, 256, 2, 4, 4);
    expect_full(255, 256, 2, 4, 4);
    expect_full(256, 512, 4, 8, 8);
    expect_full(520, 768, 6, 12, 11);
    expect_full(1999, 2048, 16, 32, 11);
    expect_full(2000, 2048, 16, 32, 11);

    plan::Geometry capped;
    assert(plan::geometry(2000, 0, 30, 3, 32, 8, &capped));
    assert(capped.parts == 8);

    plan::Geometry saturated;
    assert(plan::geometry(2000, 511, 30, 3, 32, 0, &saturated));
    assert(saturated.capacity == 512);
    assert(saturated.valid_span == 512);
    assert(saturated.schedule_span == 512);
    assert(saturated.logical_base == 1489);
    assert(plan::graph_bucket_max(saturated) == UINT32_MAX);
    assert(plan::active_mapping_is_exact_partition(
        saturated.valid_span, saturated.schedule_span, saturated.parts));

    plan::Geometry invalid;
    assert(!plan::geometry(UINT32_MAX, 0, 30, 3, 32, 0, &invalid));
    assert(!plan::geometry(10, 1000, 30, 3, 32, 0, &invalid));
    assert(!plan::geometry(10, 0, 0, 3, 32, 0, &invalid));
    assert(!plan::active_mapping_is_exact_partition(256, 256, 0));
    return 0;
}
