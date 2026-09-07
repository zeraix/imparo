#include "../sm80/attention_prefill_mma_d64_plan.h"

#include <cassert>
#include <cstdint>

int main() {
    using namespace imparo_sm80_d64_mma_plan;
    assert(!make(0, 8, 512).valid);
    assert(!make(512, 0, 512).valid);
    assert(!make(512, 8, 0).valid);

    const Plan first = make(512, 8, 512);
    assert(first.valid && first.query_tiles == 32);
    assert(first.key_updates == 8 && first.blocks == 256);

    const Plan tail = make(464, 8, 2000);
    assert(tail.valid && tail.query_tiles == 29);
    assert(tail.key_updates == 32 && tail.blocks == 232);

    const Plan ragged = make(17, 1, 65);
    assert(ragged.valid && ragged.query_tiles == 2);
    assert(ragged.key_updates == 2 && ragged.blocks == 2);
    return 0;
}
