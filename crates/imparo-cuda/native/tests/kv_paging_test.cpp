#include "../kv_paging.cuh"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    using namespace imparo_cuda_kv;
    uint32_t pages = 99;
    assert(checked_page_count(0, &pages) && pages == 0);
    assert(checked_page_count(63, &pages) && pages == 1);
    assert(checked_page_count(64, &pages) && pages == 1);
    assert(checked_page_count(127, &pages) && pages == 2);

    const uint32_t pair_swap[] = {1, 0, 3, 2, 5, 4, 7, 6, 8};
    for (uint32_t logical : {0u, 63u, 64u, 127u, 511u, 512u, 520u}) {
        assert(physical_row(logical, 0, nullptr) == logical);
    }
    assert(physical_row(0, 0, pair_swap) == 64);
    assert(physical_row(63, 0, pair_swap) == 127);
    assert(physical_row(64, 0, pair_swap) == 0);
    assert(physical_row(127, 0, pair_swap) == 63);
    assert(physical_row(511, 0, pair_swap) == 447);
    assert(physical_row(512, 0, pair_swap) == 512);
    assert(physical_row(520, 0, pair_swap) == 520);
    // Ring selection is first and must not consult the non-identity table.
    assert(physical_row(511, 511, pair_swap) == 511);
    assert(physical_row(512, 511, pair_swap) == 0);
    assert(physical_row(520, 511, pair_swap) == 8);

    constexpr uint32_t MaxLayers = 4;
    const uint64_t bytes[] = {0, 127 * 16};
    const PagingLayout layout[] = {
        {0, 0, 0, 0, 0},
        {1, 0, 127, 16, 8},
    };
    PageTables<MaxLayers> tables;
    assert(build_page_tables<MaxLayers>(2, bytes, layout, 2, nullptr, false, &tables)
           == PagingRc::Ok);
    assert(tables.arena_entries == 2);
    assert(tables.layer[1].capacity == 2);
    assert(tables.layer[1].max_entry == 1);
    assert((tables.layer[1].host_shadow == std::vector<uint32_t>{0, 1}));
    assert(!page_table_requires_mapping(tables.layer[1].host_shadow, 2));
    assert(page_table_requires_mapping(std::vector<uint32_t>{0}, 2));

    std::vector<uint32_t> candidate;
    bool changed = false;
    const uint32_t explicit_identity[] = {0, 1};
    assert(prepare_page_update(
               tables, 1, explicit_identity, 2, &candidate, &changed)
           == PagingRc::Ok && changed);
    assert(!page_table_requires_mapping(candidate, 2));
    commit_page_update(&tables, 1, std::move(candidate), 2);
    assert(tables.layer[1].installed_len == 2);
    assert(!page_table_requires_mapping(tables.layer[1].host_shadow, 2));

    const uint32_t partial_identity[] = {0};
    assert(prepare_page_update(
               tables, 1, partial_identity, 1, &candidate, &changed)
           == PagingRc::Ok && changed);
    assert((candidate == std::vector<uint32_t>{0, 1}));
    assert(!page_table_requires_mapping(candidate, 2));
    commit_page_update(&tables, 1, std::move(candidate), 1);

    const uint32_t pair[] = {1, 0};
    assert(prepare_page_update(tables, 1, pair, 2, &candidate, &changed)
           == PagingRc::Ok && changed);
    assert(page_table_requires_mapping(candidate, 2));
    commit_page_update(&tables, 1, std::move(candidate), 2);

    const uint32_t different_paged[] = {0, 0};
    assert(prepare_page_update(
               tables, 1, different_paged, 2, &candidate, &changed)
           == PagingRc::Ok && changed);
    assert(page_table_requires_mapping(tables.layer[1].host_shadow, 2));
    assert(page_table_requires_mapping(candidate, 2));
    commit_page_update(&tables, 1, std::move(candidate), 2);
    assert(prepare_page_update(
               tables, 1, different_paged, 2, &candidate, &changed)
           == PagingRc::Ok && !changed);
    assert(page_table_requires_mapping(candidate, 2));

    const uint32_t short_table[] = {1};
    assert(prepare_page_update(tables, 1, short_table, 1, &candidate, &changed)
           == PagingRc::Ok && changed);
    assert((candidate == std::vector<uint32_t>{1, 1}));
    commit_page_update(&tables, 1, std::move(candidate), 1);
    assert(tables.layer[1].installed_len == 1);
    assert(tables.layer[1].generation == 5);

    const uint32_t too_long[] = {0, 1, 0};
    assert(prepare_page_update(tables, 1, too_long, 3, &candidate, &changed)
           == PagingRc::Invalid);
    const uint32_t bad_entry[] = {2};
    assert(prepare_page_update(tables, 1, bad_entry, 1, &candidate, &changed)
           == PagingRc::Invalid);

    // Empty means an explicit identity reset, not "leave the old mapping".
    assert(prepare_page_update(tables, 1, nullptr, 0, &candidate, &changed)
           == PagingRc::Ok && changed);
    assert((candidate == std::vector<uint32_t>{0, 1}));
    assert(!page_table_requires_mapping(candidate, 2));
    commit_page_update(&tables, 1, std::move(candidate), 0);
    assert(tables.layer[1].installed_len == 0);
    assert(tables.layer[1].generation == 6);

    // Reinstall a non-identity prefix, then grow. Installed entries survive and
    // the newly addressable tail is identity-filled.
    assert(prepare_page_update(tables, 1, short_table, 1, &candidate, &changed)
           == PagingRc::Ok && changed);
    commit_page_update(&tables, 1, std::move(candidate), 1);
    const uint64_t grown_bytes[] = {0, 191 * 16};
    const PagingLayout grown_layout[] = {
        {0, 0, 0, 0, 0},
        {1, 0, 191, 16, 8},
    };
    PageTables<MaxLayers> grown;
    assert(build_page_tables(2, grown_bytes, grown_layout, 2, &tables, true, &grown)
           == PagingRc::Ok);
    assert(grown.layer[1].capacity == 3);
    assert(grown.layer[1].installed_len == 1);
    assert((grown.layer[1].host_shadow == std::vector<uint32_t>{1, 1, 2}));
    assert(grown.layer[1].generation == 7);

    PagingLayout bad_layout[] = {layout[0], layout[1]};
    bad_layout[1].k_stride = 17;
    assert(build_page_tables<MaxLayers>(2, bytes, bad_layout, 2, nullptr, false, &grown)
           == PagingRc::Invalid);
    return 0;
}
