#include "../sm80/attention_decode_vec_d256_plan.cuh"

#include <cassert>
#include <cstdint>

namespace plan = imparo_sm80_d256_vec_plan;

static void expect_geometry(uint32_t start, uint32_t span, uint32_t partials,
                            uint32_t bucket_min, uint32_t bucket_max) {
    plan::Geometry value;
    assert(plan::geometry(start, 1, 1023, &value));
    assert(value.capacity == 1024);
    assert(value.schedule_span == span);
    assert(value.partials == partials);
    assert(plan::graph_bucket_min(value) == bucket_min);
    assert(plan::graph_bucket_max(value) == bucket_max);
    assert(plan::mapping_is_exact_partition(span, partials));
}

static void expect_window(uint32_t start) {
    plan::Geometry value;
    assert(plan::geometry(start, 1, 1023, &value));
    const uint32_t lo = start + 1 > plan::kWindowSpan
        ? start + 1 - plan::kWindowSpan : 0;
    uint32_t seen = 0;
    for (uint32_t slot = 0; slot < value.schedule_span; ++slot) {
        const uint32_t logical = slot < value.valid_span
            ? slot + ((start - slot) & ~uint32_t(1023)) : 0;
        const bool valid = slot < value.valid_span && logical >= lo
            && logical <= start;
        if (valid) {
            assert((logical & 1023) == slot);
            ++seen;
        }
    }
    assert(seen == plan::kWindowSpan);
}

int main() {
    expect_geometry(511, 512, 2, 511, 511);
    expect_geometry(512, 768, 3, 512, 767);
    expect_geometry(767, 768, 3, 512, 767);
    expect_geometry(768, 1024, 4, 768, UINT32_MAX);
    expect_geometry(1023, 1024, 4, 768, UINT32_MAX);
    expect_geometry(1024, 1024, 4, 768, UINT32_MAX);
    expect_geometry(2000, 1024, 4, 768, UINT32_MAX);

    const uint32_t starts[] = {511, 512, 767, 768, 1023, 1024, 2000};
    for (uint32_t start : starts) expect_window(start);

    plan::Geometry invalid;
    assert(!plan::geometry(512, 1, UINT32_MAX, &invalid));
    assert(!plan::geometry(512, 1, 767, &invalid));
    assert(!plan::geometry(512, 1, 1000, &invalid));
    assert(!plan::geometry(UINT32_MAX, 1, 1023, &invalid));
    assert(!plan::mapping_is_exact_partition(768, 2));
    assert(!plan::mapping_is_exact_partition(1024, 3));
    return 0;
}
