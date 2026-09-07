#include "../kv_memory.cuh"

#include <cassert>
#include <cstdint>

int main() {
    using namespace imparo_cuda_kv;
    constexpr uint32_t MaxLayers = 4;
    const uint64_t bytes[] = {0, 64 * 3, 64 * 5};
    Layout<MaxLayers> layout;
    assert(build_layout<MaxLayers>(3, bytes, &layout));
    assert(layout.logical_bytes[0] == 0);
    assert(layout.logical_bytes[1] == 192);
    assert(layout.span_bytes[1] == 4096);
    assert(layout.k_offset[1] == 0);
    assert(layout.v_offset[1] == 4096);
    assert(layout.k_offset[2] == 8192);
    assert(layout.v_offset[2] == 12288);
    assert(layout.arena_bytes == 16384);
    assert(layout.live_bytes == 0);

    // Legacy contiguous access treats NeverClaimed as available, but does not
    // falsely count it as live pool ownership.
    assert(range_owned(layout, 1, 0, 0, 192));
    assert(!range_live(layout, 1, 0, 0, 192));
    assert(!range_owned(layout, 1, 0, 192, 1));

    // The first advise establishes a model-derived 192-byte quantum. State
    // transitions are strict and a freed block is inaccessible until reuse.
    assert(transition(&layout, 1, 0, 0, 192, true) == OwnershipRc::Ok);
    assert(layout.live_bytes == 192);
    assert(range_live(layout, 1, 0, 0, 192));
    assert(transition(&layout, 1, 0, 0, 192, true) == OwnershipRc::Invalid);
    assert(transition(&layout, 1, 0, 0, 192, false) == OwnershipRc::Ok);
    assert(layout.live_bytes == 0);
    assert(!range_live(layout, 1, 0, 0, 192));
    assert(!range_owned(layout, 1, 0, 0, 1));
    assert(transition(&layout, 1, 0, 0, 192, false) == OwnershipRc::Invalid);
    assert(transition(&layout, 1, 0, 0, 192, true) == OwnershipRc::Ok);

    // A distinct side/model stride learns an independent five-byte-row quantum;
    // no 4 KiB ownership assumption leaks into the state machine.
    assert(transition(&layout, 2, 1, 0, 320, true) == OwnershipRc::Ok);
    assert(layout.live_bytes == 512);
    assert(ownership(layout, 1, 0)->quantum == 192);
    assert(ownership(layout, 2, 1)->quantum == 320);
    assert(ownership(layout, 2, 0)->quantum == 0);

    // An invalid first transition is transactional: it must not pin a bad
    // quantum into an otherwise untouched side.
    assert(transition(&layout, 2, 0, 1, 64, true) == OwnershipRc::Invalid);
    assert(ownership(layout, 2, 0)->quantum == 0);

    const uint64_t grown_bytes[] = {0, 64 * 6, 64 * 5};
    Layout<MaxLayers> grown;
    assert(build_layout<MaxLayers>(3, grown_bytes, &grown));
    assert(preserve_ownership(layout, &grown) == OwnershipRc::Ok);
    assert(grown.live_bytes == 512);
    assert(ownership(grown, 1, 0)->quantum == 192);
    assert(ownership(grown, 1, 0)->blocks.size() == 2);
    assert(ownership(grown, 1, 0)->blocks[0] == Live);
    assert(ownership(grown, 1, 0)->blocks[1] == NeverClaimed);
    assert(ownership(grown, 2, 1)->blocks[0] == Live);

    assert(!build_layout<MaxLayers>(MaxLayers + 1, bytes, &grown));
    assert(!checked_align_up(UINT64_MAX, &grown.arena_bytes));
    return 0;
}
