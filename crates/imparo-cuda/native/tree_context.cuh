#pragma once
#include <vector>
#include <cstdint>
struct TreeContext {
 unsigned nodes=0,start=0,recurrent=0;
 int parents[16]{};unsigned depths[16]{};
 uint64_t state_offset=256,mask_offset=0,commit_offset=0;
 std::vector<uint8_t> mask;
};
