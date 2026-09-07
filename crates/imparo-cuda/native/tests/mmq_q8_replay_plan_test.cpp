#include "../sm80/mmq_q8_replay_plan.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <vector>

// Keep checks live under /DNDEBUG: this executable is a contract test, not a
// source file whose assertions may disappear in an optimized CI build.
#undef assert
#define assert(expression) do {                                               \
    if (!(expression)) {                                                      \
        std::fprintf(stderr, "CHECK failed at line %d: %s\n",               \
            __LINE__, #expression);                                           \
        std::abort();                                                         \
    }                                                                         \
} while (false)

using namespace imparo_sm80_q8_replay;

static PlanLimits sm86_limits() {
    return {
        uint64_t{2147483647},
        uint64_t{65535},
        uint64_t{102400},
        uint64_t{1} << 30, // Test-only bound exercises PlanStatus::WorkLimit.
        all_registered_tile_mask(),
    };
}

static void assert_segments_cover(
        uint32_t logical_tiles, uint32_t physical_grid, uint32_t blocks,
        uint32_t max_phases) {
    const uint64_t total_work = uint64_t(logical_tiles) * blocks;
    for (uint32_t tile = 0; tile < logical_tiles; ++tile) {
        std::vector<Segment> segments;
        for (uint32_t phase = 0; phase <= max_phases; ++phase) {
            const Segment segment = segment_for_phase(
                tile, blocks, total_work, physical_grid, phase);
            if (segment.valid) segments.push_back(segment);
        }
        assert(!segments.empty());
        assert(segments.front().end == blocks);
        for (size_t i = 1; i < segments.size(); ++i) {
            // Replay order is suffix, nearest prefix, then farther prefixes.
            assert(segments[i].end == segments[i - 1].begin);
        }
        assert(segments.back().begin == 0);

        std::sort(segments.begin(), segments.end(),
            [](const Segment & a, const Segment & b) {
                return a.begin < b.begin;
            });
        uint32_t cursor = 0;
        for (const Segment & segment : segments) {
            assert(segment.begin == cursor);
            assert(segment.begin < segment.end);
            cursor = segment.end;
        }
        assert(cursor == blocks);
    }
}

int main() {
    const PlanLimits sm86 = sm86_limits();

    uint64_t value = 0;
    assert(checked_ceil_div_u32_to_u64(
        std::numeric_limits<uint32_t>::max(), 128, &value));
    assert(value == 33554432);
    assert(!checked_ceil_div_u32_to_u64(1, 0, &value));
    assert(checked_mul_u64(3, 7, &value) && value == 21);
    assert(!checked_mul_u64(
        std::numeric_limits<uint64_t>::max(), 2, &value));
    assert(!checked_workspace_bytes(
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(), &value));

    assert(select_tile_tokens(0, false, sm86) == 0);
    for (uint32_t tokens : {9u, 16u, 64u, 256u, 448u, 512u, 520u}) {
        assert(select_tile_tokens(tokens, false, sm86) == 128);
        assert(select_tile_tokens(tokens, true, sm86) == 128);
    }
    assert(select_tile_tokens(520, true, sm86) == 128);
    PlanLimits only_64 = sm86;
    only_64.enabled_tile_mask = tile_bit(64);
    assert(select_tile_tokens(520, false, only_64) == 64);

    PlanLimits too_little_shared = sm86;
    too_little_shared.max_shared_bytes = shared_bytes(8) - 1;
    assert(select_tile_tokens(16, false, too_little_shared) == 0);

    const ReplayPlan n16 = make_plan(2048, 2048, 16, 28, sm86);
    assert(n16.status == PlanStatus::Ok);
    assert(n16.tile_tokens == 128);
    assert(n16.row_tiles == 16 && n16.token_tiles == 1);
    assert(n16.logical_tiles == 16 && n16.physical_grid == 28);
    assert(n16.replay && n16.multi_seam && n16.prefix_phases == 2);
    assert(n16.needs_workspace && n16.workspace_bytes_per_row == 64);
    uint64_t n16_slice_bytes = 0;
    assert(slice_workspace_bytes(n16, 2048, &n16_slice_bytes));
    assert(n16_slice_bytes == 131072);
    assert(choose_execution(n16, true) == ExecutionRoute::Mmq);
    assert(choose_execution(n16, false) == ExecutionRoute::SafeF32);

    const ReplayPlan n520 = make_plan(2048, 2048, 520, 28, sm86);
    assert(n520.status == PlanStatus::Ok);
    assert(n520.tile_tokens == 128);
    assert(n520.logical_tiles == 80 && n520.physical_grid == 80);
    assert(!n520.replay && !n520.needs_workspace);
    assert(choose_execution(n520, false) == ExecutionRoute::Mmq);

    for (uint32_t tokens : {16u, 64u, 256u, 448u, 512u}) {
        const ReplayPlan canonical = canonical_whole_k_plan(
            make_plan(2048, 2048, tokens, 28, sm86));
        assert(canonical.status == PlanStatus::Ok);
        assert(canonical.tile_tokens == 128);
        assert(canonical.physical_grid == canonical.logical_tiles);
        assert(!canonical.replay && !canonical.multi_seam);
        assert(!canonical.needs_workspace);
        assert(canonical.prefix_phases == 0);
    }

    // A physical grid narrower than the logical grid creates at most the one
    // prefix that may be added directly to the suffix.
    assert(pinned_physical_grid(32, 28) == 28);
    assert_segments_cover(32, 28, 64, 1);
    // An exact multiple has only whole-tile boundaries and needs no replay.
    assert(pinned_physical_grid(56, 28) == 56);
    assert(56 % pinned_physical_grid(56, 28) == 0);
    assert_segments_cover(56, 56, 64, 0);
    // Many physical workers on one logical tile create duplicate/empty worker
    // ranges. Seven 8-block seam slots cap replay at seven useful phases.
    assert_segments_cover(1, 28, 64, 7);

    const uint32_t out_stride = 2050;
    const SliceTiles first = slice_tiles(0, 1000, out_stride);
    const SliceTiles second = slice_tiles(1000, 1000, out_stride);
    const SliceTiles third = slice_tiles(2000, 50, out_stride);
    assert(first.valid && first.first == 0 && first.count == 8);
    assert(second.valid && second.first == 7 && second.count == 9);
    assert(third.valid && third.first == 15 && third.count == 2);
    std::vector<uint8_t> row_owners(out_stride, 0);
    const struct { uint32_t base; uint32_t rows; SliceTiles tiles; } slices[] = {
        {0, 1000, first}, {1000, 1000, second}, {2000, 50, third},
    };
    for (const auto & slice : slices) {
        for (uint32_t row = slice.base; row < slice.base + slice.rows; ++row) {
            ++row_owners[row];
            const uint32_t tile = row / kRows;
            assert(tile >= slice.tiles.first);
            assert(tile < slice.tiles.first + slice.tiles.count);
        }
    }
    for (uint8_t owners : row_owners) assert(owners == 1);
    const ReplayPlan resident = make_plan(2048, out_stride, 16, 28, sm86);
    assert(resident.status == PlanStatus::Ok && resident.fallback_rows);
    for (const auto & slice : slices) {
        for (uint32_t local = 0; local < slice.tiles.count; ++local) {
            const uint32_t global_tile = slice.tiles.first + local;
            assert(global_tile < resident.row_tiles);
            // Every paged slice uses the resident plan's logical tile identity.
            assert(uint64_t(global_tile) * resident.token_tiles
                < resident.logical_tiles);
        }
    }

    PlanLimits y_too_small = sm86;
    y_too_small.max_grid_y = 4;
    assert(make_plan(2048, 2048, 520, 28, y_too_small).status
        == PlanStatus::GridLimit);
    assert(choose_execution(
        make_plan(2048, 2048, 520, 28, y_too_small), true)
        == ExecutionRoute::Reject);
    assert(make_plan(130, 2048, 16, 28, sm86).status
        == PlanStatus::InvalidShape);
    assert(choose_execution(
        make_plan(130, 2048, 16, 28, sm86), true)
        == ExecutionRoute::Reject);
    assert(make_plan(96, 2048, 16, 28, sm86).status
        == PlanStatus::NotApplicable);
    assert(choose_execution(make_plan(96, 2048, 16, 28, sm86), true)
        == ExecutionRoute::SafeF32);
    assert(make_plan(2048, 2048, 8, 28, sm86).status
        == PlanStatus::NotApplicable);
    assert(choose_execution(
        make_plan(2048, 2048, 8, 28, sm86), true)
        == ExecutionRoute::SafeF32);
    assert(make_plan(2048, 2048, 16, 28, too_little_shared).status
        == PlanStatus::UnsupportedTile);
    assert(choose_execution(
        make_plan(2048, 2048, 16, 28, too_little_shared), true)
        == ExecutionRoute::SafeF32);

    const uint32_t huge_aligned =
        std::numeric_limits<uint32_t>::max() & ~uint32_t{127};
    assert(make_plan(huge_aligned, 2048, 520, 28, sm86).status
        == PlanStatus::WorkLimit);
    assert(make_plan(
        2048, 2048, std::numeric_limits<uint32_t>::max(), 28, sm86).status
        == PlanStatus::GridLimit);
    assert(choose_execution(make_plan(
        2048, 2048, std::numeric_limits<uint32_t>::max(), 28, sm86), true)
        == ExecutionRoute::Reject);
    ReplayPlan overflow{};
    overflow.status = PlanStatus::Overflow;
    assert(choose_execution(overflow, true) == ExecutionRoute::Reject);

    return 0;
}
