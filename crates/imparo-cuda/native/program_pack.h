#pragma once

#include <cstddef>
#include <cstdint>

// Data-only CUDA Program ABI v1. Every top-level wire is versioned and sized;
// pointers are borrowed only for the duration of the synchronous call.
constexpr uint32_t IMPARO_CUDA_PROGRAM_ABI_V1 = 1;

struct ImparoCudaProgramModuleWire {
    uint32_t struct_size;
    uint32_t abi_version;
    uint8_t module_id[32];
    const uint8_t * image;
    uint64_t image_bytes;
};

enum ImparoCudaProgramArgumentKind : uint32_t {
    IMPARO_CUDA_PROGRAM_TENSOR_PTR = 1,
    IMPARO_CUDA_PROGRAM_STATE_PTR = 2,
    IMPARO_CUDA_PROGRAM_SCRATCH_PTR = 3,
    IMPARO_CUDA_PROGRAM_SCALAR_I32 = 4,
    IMPARO_CUDA_PROGRAM_SCALAR_U32 = 5,
    IMPARO_CUDA_PROGRAM_SCALAR_U64 = 6,
    IMPARO_CUDA_PROGRAM_SCALAR_F32 = 7,
    IMPARO_CUDA_PROGRAM_MANIFEST_U32 = 8,
};

enum ImparoCudaProgramGraphMutability : uint32_t {
    IMPARO_CUDA_PROGRAM_CAPTURE_STATIC = 0,
    IMPARO_CUDA_PROGRAM_REPLAY_UPDATE = 1,
};

enum ImparoCudaProgramUpdateSource : uint32_t {
    IMPARO_CUDA_PROGRAM_UPDATE_NONE = 0,
    IMPARO_CUDA_PROGRAM_UPDATE_DECODE_START_POS_U32 = 1,
};

// Borrowed, immutable install input. `argument_schema` below owns no storage: native
// must validate and copy every descriptor before the synchronous install returns.
// `update_source` is zero for capture-static values and a compiled-adapter-owned,
// nonzero source ID for replay-update values. It never denotes a host callback.
struct ImparoCudaProgramArgumentDescriptorWire {
    uint32_t kind;
    uint32_t graph_mutability;
    uint32_t update_source;
    uint32_t reserved;
};

struct ImparoCudaProgramFunctionWire {
    uint32_t struct_size;
    uint32_t abi_version;
    uint8_t choice_group_id[32];
    uint8_t variant_id[32];
    uint8_t module_id[32];
    const uint8_t * symbol;
    uint32_t symbol_bytes;
    uint32_t argument_count;
    const ImparoCudaProgramArgumentDescriptorWire * argument_schema;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t dynamic_shared_bytes_max;
    uint32_t registers_per_thread_max;
    uint32_t static_shared_bytes_max;
    uint64_t local_memory_bytes_max;
    uint32_t threads_per_block_max;
    // 0 = forbidden, 1 = capture-only, 2 = replay-update-safe.
    uint32_t graph_capture;
    // Exact engine-authorized built-in fallback for this choice group. Native must
    // require every function in a group to declare the same nonzero identity and may
    // accept no other unknown variant as a fallback marker.
    uint8_t builtin_variant_id[32];
};

struct ImparoCudaProgramPackWire {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t program_pack_abi;
    uint32_t kernel_contract_abi;
    uint32_t backend_abi_min;
    uint32_t backend_abi_max_exclusive;
    uint32_t target_sm;
    uint32_t driver_min;
    const ImparoCudaProgramModuleWire * modules;
    uint32_t module_count;
    const ImparoCudaProgramFunctionWire * functions;
    uint32_t function_count;
    // Reserved for PR-G's frozen identity encoding. Must be zero in PR-E.
    uint32_t identity_ready;
    uint8_t pack_set_sha256[32];
    uint8_t candidate_catalog_sha256[32];
    uint8_t eligible_candidate_set_sha256[32];
};

struct ImparoCudaProgramCatalogIdentityWire {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t generation;
    uint32_t frozen;
    uint32_t module_count;
    uint32_t function_count;
    uint32_t reserved;
    uint8_t pack_set_sha256[32];
    uint8_t candidate_catalog_sha256[32];
    uint8_t eligible_candidate_set_sha256[32];
};

struct ImparoCudaProgramArgumentWire {
    uint32_t kind;
    uint32_t reserved;
    uint64_t value;
};

struct ImparoCudaProgramLaunchWire {
    uint32_t struct_size;
    uint32_t abi_version;
    uint8_t variant_id[32];
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t dynamic_shared_bytes;
    const ImparoCudaProgramArgumentWire * arguments;
    uint32_t argument_count;
    uint32_t reserved;
};

static_assert(sizeof(ImparoCudaProgramModuleWire) == 56, "Program module wire v1");
static_assert(sizeof(void *) == 8, "Program ABI v1 requires a 64-bit host");
static_assert(sizeof(ImparoCudaProgramArgumentDescriptorWire) == 16,
              "Program argument descriptor wire v1");
static_assert(sizeof(ImparoCudaProgramFunctionWire) == 200, "Program function wire v1");
static_assert(sizeof(ImparoCudaProgramPackWire) == 160, "Program pack wire v1");
static_assert(sizeof(ImparoCudaProgramCatalogIdentityWire) == 128,
              "Program catalog identity wire v1");
static_assert(sizeof(ImparoCudaProgramArgumentWire) == 16, "Program argument wire v1");
static_assert(sizeof(ImparoCudaProgramLaunchWire) == 72, "Program launch wire v1");
static_assert(offsetof(ImparoCudaProgramFunctionWire, symbol) == 104,
              "Program function symbol offset v1");
static_assert(offsetof(ImparoCudaProgramFunctionWire, argument_schema) == 120,
              "Program function argument-schema offset v1");
static_assert(offsetof(ImparoCudaProgramFunctionWire, local_memory_bytes_max) == 152,
              "Program function local-memory offset v1");
static_assert(offsetof(ImparoCudaProgramFunctionWire, builtin_variant_id) == 168,
              "Program function built-in identity offset v1");
static_assert(offsetof(ImparoCudaProgramPackWire, modules) == 32,
              "Program pack module pointer offset v1");
static_assert(offsetof(ImparoCudaProgramPackWire, functions) == 48,
              "Program pack function pointer offset v1");
static_assert(offsetof(ImparoCudaProgramLaunchWire, arguments) == 56,
              "Program launch argument pointer offset v1");
