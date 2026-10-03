#pragma once
#include <vector>
#include <cstdint>
// The model derives this wire descriptor from its existing StateGeometry. Ring
// sizes are slot counts here, never the native mask used by a cache consumer.
struct E4bTreeGeometry {
 uint32_t layer,head_dim,window,ring_slots;
 uint64_t k_stride,v_stride;
};
static_assert(sizeof(E4bTreeGeometry)==32,"E4B tree geometry wire layout");
struct TreeContext {
 unsigned nodes=0,start=0,recurrent=0;
 int parents[16]{};unsigned depths[16]{};
 uint64_t state_offset=256,mask_offset=0,commit_offset=0;
 std::vector<uint8_t> mask;
 // An explicit laboratory transaction on this same execution owner. LFM2 keeps
 // its original topology/snapshot ABI; E4B consumes the public 12-word layout.
 bool e4b=false;
 std::vector<uint32_t> row_layout;
 std::vector<E4bTreeGeometry> e4b_geometry;
};
