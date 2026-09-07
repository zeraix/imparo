#include "../host_memory.cuh"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    using namespace imparo_cuda_host;
    Registry registry;
    uint64_t first = 0;
    uint32_t index = UINT32_MAX;
    assert(registry.claim(1024, &first, &index) == Rc::Ok);
    assert(first != 0 && index == 0 && registry.live_bytes() == 1024);
    assert(registry.resolve(first)->bytes == 1024);
    assert(registry.resolve(0) == nullptr);

    const TransferSpan valid[] = {
        {first, 1, 0, 64, 0, 128},
        {first, 1, 1, 64, 128, 128},
        {first, 2, 0, 0, 256, 256},
    };
    std::vector<PlannedSpan> plan;
    uint64_t total = 0;
    assert(build_plan(registry, valid, 3, &plan, &total) == Rc::Ok);
    assert(total == 512 && plan[2].staging_offset == 256);

    const TransferSpan host_overlap[] = {
        {first, 1, 0, 0, 10, 64}, {first, 2, 0, 0, 70, 64},
    };
    assert(build_plan(registry, host_overlap, 2, &plan, &total) == Rc::Invalid);
    const TransferSpan device_overlap[] = {
        {first, 1, 0, 0, 0, 64}, {first, 1, 0, 32, 128, 64},
    };
    assert(build_plan(registry, device_overlap, 2, &plan, &total) == Rc::Invalid);
    const TransferSpan distinct_sides[] = {
        {first, 1, 0, 0, 0, 64}, {first, 1, 1, 0, 128, 64},
    };
    assert(build_plan(registry, distinct_sides, 2, &plan, &total) == Rc::Ok);
    const TransferSpan out_of_bounds[] = {{first, 1, 0, 0, 1000, 25}};
    assert(build_plan(registry, out_of_bounds, 1, &plan, &total) == Rc::Invalid);
    const TransferSpan zero[] = {{first, 1, 0, 0, 0, 0}};
    assert(build_plan(registry, zero, 1, &plan, &total) == Rc::Invalid);
    const TransferSpan overflow[] = {{first, 1, 0, UINT64_MAX, 0, 1}};
    assert(build_plan(registry, overflow, 1, &plan, &total) == Rc::Invalid);
    assert(build_plan(registry, nullptr, 1, &plan, &total) == Rc::Invalid);
    assert(build_plan(registry, valid, 0, &plan, &total) == Rc::Invalid);

    assert(registry.release(first) == Rc::Ok);
    assert(registry.live_bytes() == 0);
    assert(registry.resolve(first) == nullptr);
    assert(registry.release(first) == Rc::Invalid);
    assert(build_plan(registry, valid, 3, &plan, &total) == Rc::Invalid);

    uint64_t second = 0;
    assert(registry.claim(512, &second, &index) == Rc::Ok);
    assert(index == 0 && second != first);
    assert(registry.resolve(first) == nullptr);
    assert(registry.resolve(second)->bytes == 512);
    const TransferSpan stale[] = {{first, 1, 0, 0, 0, 64}};
    assert(build_plan(registry, stale, 1, &plan, &total) == Rc::Invalid);
    uint64_t other = 0;
    assert(registry.claim(512, &other, &index) == Rc::Ok);
    const TransferSpan multi_handle[] = {
        {second, 1, 0, 0, 0, 64},
        {other, 1, 1, 0, 0, 64},
    };
    assert(build_plan(registry, multi_handle, 2, &plan, &total) == Rc::Ok);
    assert(plan[0].host_index != plan[1].host_index);
    assert(registry.release(other) == Rc::Ok);
    assert(registry.claim(0, &first, &index) == Rc::Invalid);
    assert(registry.claim(UINT64_MAX, &first, &index) == Rc::Invalid);

    uint64_t third = 0;
    assert(registry.claim(UINT64_MAX - 512, &third, &index) == Rc::Ok);
    uint64_t impossible = 0;
    assert(registry.claim(1, &impossible, &index) == Rc::Invalid);
    assert(registry.release(third) == Rc::Ok);
    assert(registry.release(second) == Rc::Ok);
    return 0;
}