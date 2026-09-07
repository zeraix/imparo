// Included by imparo_cuda.cu after the core runtime surface. This remains a unity
// build so the bridge shares the one Runtime primary context, stream and Graph owner.

#include "program_pack.h"
#include <cuda.h>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace {

constexpr uint64_t kProgramModuleMaxBytes = 268435456ull;
constexpr uint32_t kProgramFunctionMaxCount = 4096;
constexpr uint32_t kProgramUpdateDecodeStartPosU32 = 1;

struct ProgramDriverApi {
    void * library = nullptr;
    bool ready = false;
    decltype(&cuInit) init = nullptr;
    decltype(&cuDevicePrimaryCtxRetain) primary_context_retain = nullptr;
    decltype(&cuCtxSetCurrent) context_set_current = nullptr;
    decltype(&cuCtxGetCurrent) context_get_current = nullptr;
    decltype(&cuCtxGetDevice) context_get_device = nullptr;
    decltype(&cuModuleLoadDataEx) module_load_data_ex = nullptr;
    decltype(&cuModuleUnload) module_unload = nullptr;
    decltype(&cuModuleGetFunction) module_get_function = nullptr;
    decltype(&cuFuncGetAttribute) function_get_attribute = nullptr;
    decltype(&cuFuncSetAttribute) function_set_attribute = nullptr;
    decltype(&cuLaunchKernel) launch_kernel = nullptr;
    decltype(&cuGraphKernelNodeGetParams) graph_kernel_node_get_params = nullptr;
    decltype(&cuGraphExecKernelNodeSetParams)
        graph_exec_kernel_node_set_params = nullptr;
};

struct ProgramModule {
    uint8_t id[32] = {};
    CUmodule module = nullptr;
};

struct ProgramFunction {
    uint8_t choice_group_id[32] = {};
    uint8_t variant_id[32] = {};
    uint8_t module_id[32] = {};
    uint8_t builtin_variant_id[32] = {};
    CUfunction function = nullptr;
    uint32_t argument_count = 0;
    std::array<ImparoCudaProgramArgumentDescriptorWire, 64> argument_schema = {};
    uint32_t block[3] = {};
    uint32_t dynamic_shared_bytes_max = 0;
    uint32_t graph_capture = 0;
};

struct ProgramCatalog {
    std::mutex mutex;
    ProgramDriverApi driver;
    CUcontext primary_context = nullptr;
    CUdevice primary_device = -1;
    std::vector<ProgramModule> modules;
    std::vector<ProgramFunction> functions;
    std::vector<std::pair<std::array<uint8_t, 32>, std::array<uint8_t, 32>>> bindings;
    uint64_t generation = 0;
    bool frozen = false;
    bool poisoned = false;
    uint32_t max_grid[3] = {};
    uint32_t max_block[3] = {};
    uint32_t max_threads_per_block = 0;
    uint64_t max_shared_bytes = 0;
    uint8_t pack_set_sha256[32] = {};
    uint8_t candidate_catalog_sha256[32] = {};
    uint8_t eligible_candidate_set_sha256[32] = {};
};

ProgramCatalog program_catalog;

bool bytes_equal(const uint8_t * a, const uint8_t * b, size_t count) {
    return std::memcmp(a, b, count) == 0;
}

bool bytes_nonzero(const uint8_t * value, size_t count) {
    for (size_t i = 0; i < count; ++i) if (value[i]) return true;
    return false;
}

bool lower_hex(uint8_t value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

bool valid_program_symbol(const uint8_t * symbol, uint32_t bytes) {
    if (!symbol || bytes != 67 || symbol[0] != 'i' || symbol[1] != 'p'
        || symbol[2] != '_') return false;
    for (uint32_t i = 3; i < bytes; ++i) if (!lower_hex(symbol[i])) return false;
    return true;
}

bool valid_program_argument_schema(
        const ImparoCudaProgramFunctionWire & wire) {
    if (!wire.argument_schema || !wire.argument_count
        || wire.argument_count > 64) return false;
    bool has_replay_update = false;
    for (uint32_t i = 0; i < wire.argument_count; ++i) {
        const auto & descriptor = wire.argument_schema[i];
        if (descriptor.kind < IMPARO_CUDA_PROGRAM_TENSOR_PTR
            || descriptor.kind > IMPARO_CUDA_PROGRAM_MANIFEST_U32
            || descriptor.reserved
            || descriptor.graph_mutability > IMPARO_CUDA_PROGRAM_REPLAY_UPDATE) {
            return false;
        }
        if (descriptor.graph_mutability == IMPARO_CUDA_PROGRAM_CAPTURE_STATIC) {
            if (descriptor.update_source) return false;
        } else {
            // ABI26 currently freezes one compiled-adapter source. Unknown
            // sources are not slot indices and must fail closed until an engine
            // adapter explicitly implements their replay semantics.
            if (descriptor.update_source != kProgramUpdateDecodeStartPosU32
                || descriptor.kind != IMPARO_CUDA_PROGRAM_SCALAR_U32) {
                return false;
            }
            has_replay_update = true;
        }
    }
    if ((wire.graph_capture == 2) != has_replay_update) return false;
    return true;
}

bool valid_program_builtin_registration(
        const ImparoCudaProgramPackWire & pack,
        const ImparoCudaProgramFunctionWire & wire) {
    if (!bytes_nonzero(wire.builtin_variant_id, 32)
        || bytes_equal(wire.builtin_variant_id, wire.variant_id, 32)) {
        return false;
    }
    for (const ProgramFunction & prior : program_catalog.functions) {
        if (bytes_equal(prior.variant_id, wire.builtin_variant_id, 32)
            || bytes_equal(prior.builtin_variant_id, wire.variant_id, 32)
            || (bytes_equal(prior.choice_group_id, wire.choice_group_id, 32)
                && !bytes_equal(prior.builtin_variant_id,
                                wire.builtin_variant_id, 32))
            || (!bytes_equal(prior.choice_group_id, wire.choice_group_id, 32)
                && bytes_equal(prior.builtin_variant_id,
                               wire.builtin_variant_id, 32))) {
            return false;
        }
    }
    for (uint32_t i = 0; i < pack.function_count; ++i) {
        const auto & candidate = pack.functions[i];
        if (bytes_equal(candidate.variant_id, wire.builtin_variant_id, 32)
            || (bytes_equal(candidate.choice_group_id, wire.choice_group_id, 32)
                && !bytes_equal(candidate.builtin_variant_id,
                                wire.builtin_variant_id, 32))
            || (!bytes_equal(candidate.choice_group_id, wire.choice_group_id, 32)
                && bytes_equal(candidate.builtin_variant_id,
                               wire.builtin_variant_id, 32))) {
            return false;
        }
    }
    return true;
}

bool looks_like_cubin(const uint8_t * image, uint64_t bytes) {
    return image && bytes >= 16 && image[0] == 0x7f && image[1] == 'E'
        && image[2] == 'L' && image[3] == 'F';
}

void close_program_driver(ProgramDriverApi & api) noexcept {
    if (!api.library) return;
#if defined(_WIN32)
    (void)FreeLibrary(static_cast<HMODULE>(api.library));
#else
    (void)dlclose(api.library);
#endif
    api = {};
}

template <typename T>
bool load_program_driver_symbol(void * library, const char * name, T * out) {
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress(static_cast<HMODULE>(library), name);
    if (!symbol || sizeof(symbol) != sizeof(*out)) return false;
    std::memcpy(out, &symbol, sizeof(*out));
#else
    void * symbol = dlsym(library, name);
    if (!symbol || sizeof(symbol) != sizeof(*out)) return false;
    std::memcpy(out, &symbol, sizeof(*out));
#endif
    return true;
}

bool ensure_program_driver(ProgramDriverApi & api) {
    if (api.ready) return true;
    if (api.library) return false;
    ProgramDriverApi candidate;
#if defined(_WIN32)
    HMODULE library = LoadLibraryExW(
        L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!library) return false;
    candidate.library = library;
#else
    void * library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) return false;
    candidate.library = library;
#endif
#define IMPARO_LOAD_DRIVER(field, symbol)                                           \
    if (!load_program_driver_symbol(candidate.library, symbol, &candidate.field)) { \
        close_program_driver(candidate);                                             \
        return false;                                                                \
    }
    IMPARO_LOAD_DRIVER(init, "cuInit");
    IMPARO_LOAD_DRIVER(primary_context_retain, "cuDevicePrimaryCtxRetain");
    IMPARO_LOAD_DRIVER(context_set_current, "cuCtxSetCurrent");
    IMPARO_LOAD_DRIVER(context_get_current, "cuCtxGetCurrent");
    IMPARO_LOAD_DRIVER(context_get_device, "cuCtxGetDevice");
    IMPARO_LOAD_DRIVER(module_load_data_ex, "cuModuleLoadDataEx");
    IMPARO_LOAD_DRIVER(module_unload, "cuModuleUnload");
    IMPARO_LOAD_DRIVER(module_get_function, "cuModuleGetFunction");
    IMPARO_LOAD_DRIVER(function_get_attribute, "cuFuncGetAttribute");
    IMPARO_LOAD_DRIVER(function_set_attribute, "cuFuncSetAttribute");
    IMPARO_LOAD_DRIVER(launch_kernel, "cuLaunchKernel");
    IMPARO_LOAD_DRIVER(graph_kernel_node_get_params,
                       "cuGraphKernelNodeGetParams_v2");
    IMPARO_LOAD_DRIVER(graph_exec_kernel_node_set_params,
                       "cuGraphExecKernelNodeSetParams_v2");
#undef IMPARO_LOAD_DRIVER
    if (candidate.init(0) != CUDA_SUCCESS) {
        close_program_driver(candidate);
        return false;
    }
    candidate.ready = true;
    api = candidate;
    return true;
}

int ensure_program_context(ProgramCatalog & catalog) {
    const bool capturing = g.graph_capturing;
    // Runtime initialization uses cudaFree(nullptr), and context switching is also
    // host lifecycle work. Neither belongs inside stream capture. Install/freeze
    // must have established the exact primary context and sole stream beforehand.
    if (capturing) {
        if (!g.runtime_initialized || !g.stream || !catalog.driver.ready
            || !catalog.primary_context) return CUDA_RC_INVALID;
    } else {
        const int runtime_rc = ensure_cuda_runtime();
        if (runtime_rc) return runtime_rc;
    }
    if (catalog.poisoned
        || (!catalog.driver.ready && !ensure_program_driver(catalog.driver))) {
        return CUDA_RC_ERROR;
    }
    if (!catalog.primary_context) {
        CUcontext primary = nullptr;
        if (catalog.driver.primary_context_retain(&primary, CUdevice(g.device))
                != CUDA_SUCCESS
            || !primary) return CUDA_RC_ERROR;
        catalog.primary_context = primary;
        catalog.primary_device = CUdevice(g.device);

        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, g.device) != cudaSuccess) {
            catalog.poisoned = true;
            return CUDA_RC_ERROR;
        }
        for (uint32_t axis = 0; axis < 3; ++axis) {
            if (prop.maxGridSize[axis] <= 0 || prop.maxThreadsDim[axis] <= 0) {
                catalog.poisoned = true;
                return CUDA_RC_ERROR;
            }
            catalog.max_grid[axis] = uint32_t(prop.maxGridSize[axis]);
            catalog.max_block[axis] = uint32_t(prop.maxThreadsDim[axis]);
        }
        catalog.max_threads_per_block = uint32_t(prop.maxThreadsPerBlock);
        catalog.max_shared_bytes = uint64_t(prop.sharedMemPerBlockOptin > 0
            ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock);
    }
    if (catalog.primary_device != CUdevice(g.device)
        || (!capturing
            && catalog.driver.context_set_current(catalog.primary_context)
                != CUDA_SUCCESS)) {
        return CUDA_RC_INVALID;
    }
    CUcontext current = nullptr;
    CUdevice device = -1;
    if (catalog.driver.context_get_current(&current) != CUDA_SUCCESS
        || catalog.driver.context_get_device(&device) != CUDA_SUCCESS
        || current != catalog.primary_context || device != catalog.primary_device) {
        return CUDA_RC_INVALID;
    }
    return 0;
}

const ProgramModule * find_program_module(
        const std::vector<ProgramModule> & modules, const uint8_t id[32]) {
    for (const ProgramModule & module : modules) {
        if (bytes_equal(module.id, id, 32)) return &module;
    }
    return nullptr;
}

ProgramFunction * find_program_function(const uint8_t id[32]) {
    for (ProgramFunction & function : program_catalog.functions) {
        if (bytes_equal(function.variant_id, id, 32)) return &function;
    }
    return nullptr;
}

bool function_handle_exists(const std::vector<ProgramFunction> & functions,
                            CUfunction handle) {
    for (const ProgramFunction & function : functions) {
        if (function.function == handle) return true;
    }
    return false;
}

bool unload_program_modules(ProgramCatalog & catalog,
                            std::vector<ProgramModule> & modules) noexcept {
    bool ok = true;
    for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
        if (!it->module) continue;
        if (catalog.driver.module_unload(it->module) == CUDA_SUCCESS) {
            it->module = nullptr;
        } else {
            ok = false;
        }
    }
    if (ok) modules.clear();
    return ok;
}

bool validate_function_resources(ProgramCatalog & catalog, CUfunction function,
                                 const ImparoCudaProgramFunctionWire & wire) {
    if (!wire.block_x || !wire.block_y || !wire.block_z
        || wire.block_x > 1024 || wire.block_y > 1024 || wire.block_z > 64
        || wire.block_x > catalog.max_block[0]
        || wire.block_y > catalog.max_block[1]
        || wire.block_z > catalog.max_block[2]
        || wire.registers_per_thread_max > 255
        || wire.static_shared_bytes_max > 262144
        || wire.dynamic_shared_bytes_max > 262144
        || wire.local_memory_bytes_max > 1073741824ull
        || !wire.threads_per_block_max || wire.threads_per_block_max > 1024) {
        return false;
    }
    const uint64_t block_threads = uint64_t(wire.block_x) * wire.block_y * wire.block_z;
    if (block_threads > catalog.max_threads_per_block
        || block_threads > wire.threads_per_block_max) return false;

    int registers = 0;
    int static_shared = 0;
    int local = 0;
    int max_threads = 0;
    int max_dynamic_shared = 0;
    int binary_version = 0;
    auto & api = catalog.driver;
    if (api.function_get_attribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function)
            != CUDA_SUCCESS
        || api.function_get_attribute(&static_shared,
               CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function) != CUDA_SUCCESS
        || api.function_get_attribute(&local,
               CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function) != CUDA_SUCCESS
        || api.function_get_attribute(&max_threads,
               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function) != CUDA_SUCCESS
        || api.function_get_attribute(&max_dynamic_shared,
               CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function) != CUDA_SUCCESS
        || api.function_get_attribute(&binary_version,
               CU_FUNC_ATTRIBUTE_BINARY_VERSION, function) != CUDA_SUCCESS) {
        return false;
    }
    if (max_dynamic_shared >= 0
        && wire.dynamic_shared_bytes_max > uint32_t(max_dynamic_shared)) {
        if (wire.dynamic_shared_bytes_max > uint32_t(std::numeric_limits<int>::max())
            || api.function_set_attribute(function,
                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                   int(wire.dynamic_shared_bytes_max)) != CUDA_SUCCESS
            || api.function_get_attribute(&max_dynamic_shared,
                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function)
                   != CUDA_SUCCESS) return false;
    }
    const uint64_t total_shared = static_shared < 0
        ? UINT64_MAX : uint64_t(static_shared) + wire.dynamic_shared_bytes_max;
    return binary_version == g.sm_version
        && registers >= 0 && uint32_t(registers) <= wire.registers_per_thread_max
        && static_shared >= 0 && uint32_t(static_shared) <= wire.static_shared_bytes_max
        && local >= 0 && uint64_t(local) <= wire.local_memory_bytes_max
        && max_threads > 0 && block_threads <= uint32_t(max_threads)
        && max_dynamic_shared >= 0
        && wire.dynamic_shared_bytes_max <= uint32_t(max_dynamic_shared)
        && total_shared <= catalog.max_shared_bytes;
}

bool fatal_driver_launch(CUresult result) {
    return result == CUDA_ERROR_LAUNCH_FAILED
        || result == CUDA_ERROR_ILLEGAL_ADDRESS
        || result == CUDA_ERROR_ASSERT;
}

int quiesce_program_catalog(ProgramCatalog & catalog) {
    if (g.graph_capturing || g.forward_active) return CUDA_RC_INVALID;
    const int context_rc = ensure_program_context(catalog);
    if (context_rc) return context_rc;
    if (g.stream && cudaStreamSynchronize(g.stream) != cudaSuccess) {
        catalog.poisoned = true;
        return CUDA_RC_ERROR;
    }
    if (!destroy_decode_graph_checked()) {
        catalog.poisoned = true;
        return CUDA_RC_ERROR;
    }
    return 0;
}

int program_pack_install_impl(
        const ImparoCudaProgramPackWire * pack, uint32_t struct_size) {
    if (!pack || struct_size != sizeof(*pack) || pack->struct_size != sizeof(*pack)
        || pack->abi_version != IMPARO_CUDA_PROGRAM_ABI_V1
        || pack->program_pack_abi != IMPARO_CUDA_PROGRAM_ABI_V1
        || pack->kernel_contract_abi != 1
        || pack->backend_abi_min > IMPARO_CUDA_BACKEND_ABI
        || pack->backend_abi_max_exclusive <= IMPARO_CUDA_BACKEND_ABI
        || !pack->modules || !pack->module_count || pack->module_count > 64
        || !pack->functions || !pack->function_count
        || pack->function_count > kProgramFunctionMaxCount
        || pack->identity_ready != 0
        || bytes_nonzero(pack->pack_set_sha256, 32)
        || bytes_nonzero(pack->candidate_catalog_sha256, 32)
        || bytes_nonzero(pack->eligible_candidate_set_sha256, 32)) {
        return CUDA_RC_INVALID;
    }
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || program_catalog.frozen
        || g.graph_capturing || g.forward_active) return CUDA_RC_INVALID;
    const int context_rc = ensure_program_context(program_catalog);
    if (context_rc) return context_rc;
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess
        || driver_version <= 0 || pack->target_sm != uint32_t(g.sm_version)
        || pack->driver_min > uint32_t(driver_version)) return CUDA_RC_INVALID;
    const int quiesce_rc = quiesce_program_catalog(program_catalog);
    if (quiesce_rc) return quiesce_rc;

    std::vector<ProgramModule> added_modules;
    std::vector<ProgramFunction> added_functions;
    added_modules.reserve(pack->module_count);
    added_functions.reserve(pack->function_count);
    program_catalog.modules.reserve(
        program_catalog.modules.size() + pack->module_count);
    program_catalog.functions.reserve(
        program_catalog.functions.size() + pack->function_count);

    for (uint32_t i = 0; i < pack->module_count; ++i) {
        const auto & wire = pack->modules[i];
        if (wire.struct_size != sizeof(wire)
            || wire.abi_version != IMPARO_CUDA_PROGRAM_ABI_V1
            || !bytes_nonzero(wire.module_id, 32)
            || !wire.image_bytes || wire.image_bytes > kProgramModuleMaxBytes
            || !looks_like_cubin(wire.image, wire.image_bytes)
            || find_program_module(program_catalog.modules, wire.module_id)
            || find_program_module(added_modules, wire.module_id)) {
            if (!unload_program_modules(program_catalog, added_modules)) {
                program_catalog.poisoned = true;
                return CUDA_RC_ERROR;
            }
            return CUDA_RC_INVALID;
        }
        ProgramModule module;
        std::memcpy(module.id, wire.module_id, 32);
        if (program_catalog.driver.module_load_data_ex(
                &module.module, wire.image, 0, nullptr, nullptr) != CUDA_SUCCESS) {
            if (!unload_program_modules(program_catalog, added_modules)) {
                program_catalog.poisoned = true;
                return CUDA_RC_ERROR;
            }
            return CUDA_RC_INVALID;
        }
        added_modules.push_back(module);
    }

    for (uint32_t i = 0; i < pack->function_count; ++i) {
        const auto & wire = pack->functions[i];
        const ProgramModule * module = find_program_module(added_modules, wire.module_id);
        bool duplicate = find_program_function(wire.variant_id) != nullptr;
        for (const ProgramFunction & prior : added_functions) {
            duplicate = duplicate || bytes_equal(prior.variant_id, wire.variant_id, 32);
        }
        if (wire.struct_size != sizeof(wire)
            || wire.abi_version != IMPARO_CUDA_PROGRAM_ABI_V1
            || !module || !bytes_nonzero(wire.choice_group_id, 32)
            || !bytes_nonzero(wire.variant_id, 32)
            || !valid_program_builtin_registration(*pack, wire)
            || !valid_program_symbol(wire.symbol, wire.symbol_bytes)
            || !valid_program_argument_schema(wire)
            || wire.graph_capture > 2 || duplicate) {
            if (!unload_program_modules(program_catalog, added_modules)) {
                program_catalog.poisoned = true;
                return CUDA_RC_ERROR;
            }
            return CUDA_RC_INVALID;
        }
        char symbol[68] = {};
        std::memcpy(symbol, wire.symbol, wire.symbol_bytes);
        ProgramFunction function;
        std::memcpy(function.choice_group_id, wire.choice_group_id, 32);
        std::memcpy(function.variant_id, wire.variant_id, 32);
        std::memcpy(function.module_id, wire.module_id, 32);
        std::memcpy(function.builtin_variant_id, wire.builtin_variant_id, 32);
        function.argument_count = wire.argument_count;
        std::copy_n(
            wire.argument_schema, wire.argument_count, function.argument_schema.begin());
        function.block[0] = wire.block_x;
        function.block[1] = wire.block_y;
        function.block[2] = wire.block_z;
        function.dynamic_shared_bytes_max = wire.dynamic_shared_bytes_max;
        function.graph_capture = wire.graph_capture;
        if (program_catalog.driver.module_get_function(
                &function.function, module->module, symbol) != CUDA_SUCCESS
            || !function.function
            || function_handle_exists(program_catalog.functions, function.function)
            || function_handle_exists(added_functions, function.function)
            || !validate_function_resources(program_catalog, function.function, wire)) {
            if (!unload_program_modules(program_catalog, added_modules)) {
                program_catalog.poisoned = true;
                return CUDA_RC_ERROR;
            }
            return CUDA_RC_INVALID;
        }
        added_functions.push_back(function);
    }

    program_catalog.modules.insert(program_catalog.modules.end(),
        added_modules.begin(), added_modules.end());
    program_catalog.functions.insert(program_catalog.functions.end(),
        added_functions.begin(), added_functions.end());
    ++program_catalog.generation;
    return 0;
}

int program_catalog_identity_impl(
        ImparoCudaProgramCatalogIdentityWire * out, uint32_t struct_size) {
    if (!out || struct_size != sizeof(*out)) return CUDA_RC_INVALID;
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    out->abi_version = IMPARO_CUDA_PROGRAM_ABI_V1;
    out->generation = program_catalog.generation;
    out->frozen = program_catalog.frozen ? 1u : 0u;
    out->module_count = uint32_t(program_catalog.modules.size());
    out->function_count = uint32_t(program_catalog.functions.size());
    std::memcpy(out->pack_set_sha256, program_catalog.pack_set_sha256, 32);
    std::memcpy(out->candidate_catalog_sha256,
                program_catalog.candidate_catalog_sha256, 32);
    std::memcpy(out->eligible_candidate_set_sha256,
                program_catalog.eligible_candidate_set_sha256, 32);
    return program_catalog.poisoned ? CUDA_RC_ERROR : 0;
}

int program_bind_impl(const uint8_t group_id[32], const uint8_t variant_id[32]) {
    if (!group_id || !variant_id || !bytes_nonzero(group_id, 32)
        || !bytes_nonzero(variant_id, 32)) return CUDA_RC_INVALID;
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || !program_catalog.frozen
        || g.graph_capturing || g.forward_active) return CUDA_RC_INVALID;
    bool group_known = false;
    for (const ProgramFunction & candidate : program_catalog.functions) {
        group_known = group_known
            || bytes_equal(candidate.choice_group_id, group_id, 32);
    }
    if (!group_known) return CUDA_RC_INVALID;
    ProgramFunction * function = find_program_function(variant_id);
    if (function && !bytes_equal(function->choice_group_id, group_id, 32)) {
        return CUDA_RC_INVALID;
    }
    bool exact_builtin = false;
    for (const ProgramFunction & candidate : program_catalog.functions) {
        if (bytes_equal(candidate.choice_group_id, group_id, 32)) {
            exact_builtin = bytes_equal(
                candidate.builtin_variant_id, variant_id, 32);
            break;
        }
    }
    // Unknown IDs carry no authority. The only non-CUfunction binding accepted is
    // the exact engine-owned fallback identity registered by every member of this
    // choice group during transactional install.
    if (!function && !exact_builtin) return CUDA_RC_INVALID;
    for (auto & binding : program_catalog.bindings) {
        if (!bytes_equal(binding.first.data(), group_id, 32)) continue;
        if (bytes_equal(binding.second.data(), variant_id, 32)) return 0;
        const int quiesce_rc = quiesce_program_catalog(program_catalog);
        if (quiesce_rc) return quiesce_rc;
        std::memcpy(binding.second.data(), variant_id, 32);
        ++program_catalog.generation;
        return 0;
    }
    // Allocate before invalidating an otherwise usable graph so allocation failure
    // leaves both the route map and the graph unchanged.
    program_catalog.bindings.reserve(program_catalog.bindings.size() + 1);
    const int quiesce_rc = quiesce_program_catalog(program_catalog);
    if (quiesce_rc) return quiesce_rc;
    std::array<uint8_t, 32> group = {};
    std::array<uint8_t, 32> variant = {};
    std::memcpy(group.data(), group_id, 32);
    std::memcpy(variant.data(), variant_id, 32);
    program_catalog.bindings.emplace_back(group, variant);
    ++program_catalog.generation;
    return 0;
}

int program_freeze_impl() {
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || g.graph_capturing || g.forward_active
        || program_catalog.modules.empty() || program_catalog.functions.empty()) {
        return CUDA_RC_INVALID;
    }
    if (program_catalog.frozen) return 0;
    const int context_rc = ensure_program_context(program_catalog);
    if (context_rc) return context_rc;
    program_catalog.frozen = true;
    ++program_catalog.generation;
    return 0;
}

union alignas(8) ProgramArgumentStorage {
    uint64_t u64;
    uint32_t u32;
};

bool add_program_dynamic_graph_node(
        cudaGraphNode_t node, bool * matched) noexcept try {
    if (!matched) return false;
    *matched = false;
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned) return false;
    if (!program_catalog.frozen) {
        // No Program Pack is the normal open-source runtime state. It contributes
        // no Driver nodes, so continue classifying ordinary Runtime kernels. A
        // populated but unfrozen catalog remains ineligible for Graph capture.
        return program_catalog.modules.empty() && program_catalog.functions.empty()
            && program_catalog.bindings.empty();
    }
    CUDA_KERNEL_NODE_PARAMS params = {};
    if (program_catalog.driver.graph_kernel_node_get_params(
            reinterpret_cast<CUgraphNode>(node), &params) != CUDA_SUCCESS) {
        return true;
    }
    ProgramFunction * function = nullptr;
    for (ProgramFunction & candidate : program_catalog.functions) {
        if (candidate.function == params.func) {
            function = &candidate;
            break;
        }
    }
    if (!function) return true;
    *matched = true;
    if (function->graph_capture == 0 || !params.kernelParams || params.extra
        || params.blockDimX != function->block[0]
        || params.blockDimY != function->block[1]
        || params.blockDimZ != function->block[2]
        || params.sharedMemBytes > function->dynamic_shared_bytes_max) {
        return false;
    }
    if (function->graph_capture == 1) return true;
    if (function->graph_capture != 2 || !function->argument_count) return false;

    DynamicGraphNode dynamic;
    dynamic.node = node;
    dynamic.params.func = reinterpret_cast<void *>(params.func);
    dynamic.params.gridDim = dim3(params.gridDimX, params.gridDimY, params.gridDimZ);
    dynamic.params.blockDim = dim3(params.blockDimX, params.blockDimY, params.blockDimZ);
    dynamic.params.sharedMemBytes = params.sharedMemBytes;
    dynamic.args.resize(function->argument_count);
    dynamic.program_arguments.resize(function->argument_count);
    dynamic.program_update_sources.resize(function->argument_count);
    bool has_replay_update = false;
    for (uint32_t i = 0; i < function->argument_count; ++i) {
        if (!params.kernelParams[i]) return false;
        const auto & descriptor = function->argument_schema[i];
        auto & storage = dynamic.program_arguments[i];
        const bool narrow = descriptor.kind == IMPARO_CUDA_PROGRAM_SCALAR_I32
            || descriptor.kind == IMPARO_CUDA_PROGRAM_SCALAR_U32
            || descriptor.kind == IMPARO_CUDA_PROGRAM_SCALAR_F32
            || descriptor.kind == IMPARO_CUDA_PROGRAM_MANIFEST_U32;
        if (narrow) {
            std::memcpy(&storage.u32, params.kernelParams[i], sizeof(storage.u32));
            dynamic.args[i] = &storage.u32;
        } else {
            std::memcpy(&storage.u64, params.kernelParams[i], sizeof(storage.u64));
            dynamic.args[i] = &storage.u64;
        }
        dynamic.program_update_sources[i] = descriptor.update_source;
        has_replay_update = has_replay_update || descriptor.update_source != 0;
    }
    if (!has_replay_update) return false;
    dynamic.params.kernelParams = dynamic.args.data();
    g.decode_graph_nodes.push_back(std::move(dynamic));
    // Rebuild argument pointers from their final owner; the scanner never keeps
    // pointers into the synchronous launch wire or driver-owned temporary storage.
    DynamicGraphNode & owned = g.decode_graph_nodes.back();
    for (uint32_t i = 0; i < function->argument_count; ++i) {
        const uint32_t kind = function->argument_schema[i].kind;
        const bool narrow = kind == IMPARO_CUDA_PROGRAM_SCALAR_I32
            || kind == IMPARO_CUDA_PROGRAM_SCALAR_U32
            || kind == IMPARO_CUDA_PROGRAM_SCALAR_F32
            || kind == IMPARO_CUDA_PROGRAM_MANIFEST_U32;
        owned.args[i] = narrow
            ? static_cast<void *>(&owned.program_arguments[i].u32)
            : static_cast<void *>(&owned.program_arguments[i].u64);
    }
    owned.params.kernelParams = owned.args.data();
    return true;
} catch (...) {
    // configure_decode_graph_nodes is called by the legacy C surface. Allocation
    // or mutex failures must reject and discard the capture, never cross that ABI.
    return false;
}

int update_program_dynamic_graph_node(
        DynamicGraphNode & dynamic, uint32_t) noexcept try {
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || !program_catalog.frozen
        || !g.decode_graph_exec || !dynamic.node || dynamic.args.empty()) {
        return CUDA_RC_INVALID;
    }
    CUDA_KERNEL_NODE_PARAMS params = {};
    params.func = reinterpret_cast<CUfunction>(dynamic.params.func);
    params.gridDimX = dynamic.params.gridDim.x;
    params.gridDimY = dynamic.params.gridDim.y;
    params.gridDimZ = dynamic.params.gridDim.z;
    params.blockDimX = dynamic.params.blockDim.x;
    params.blockDimY = dynamic.params.blockDim.y;
    params.blockDimZ = dynamic.params.blockDim.z;
    params.sharedMemBytes = dynamic.params.sharedMemBytes;
    params.kernelParams = dynamic.args.data();
    return program_catalog.driver.graph_exec_kernel_node_set_params(
        reinterpret_cast<CUgraphExec>(g.decode_graph_exec),
        reinterpret_cast<CUgraphNode>(dynamic.node), &params) == CUDA_SUCCESS
        ? 0 : CUDA_RC_ERROR;
} catch (...) {
    return CUDA_RC_ERROR;
}

int program_launch_impl(
        const ImparoCudaProgramLaunchWire * launch, uint32_t struct_size) {
    if (!launch || struct_size != sizeof(*launch)
        || launch->struct_size != sizeof(*launch)
        || launch->abi_version != IMPARO_CUDA_PROGRAM_ABI_V1
        || !launch->grid_x || !launch->grid_y || !launch->grid_z
        || !launch->arguments || !launch->argument_count
        || launch->argument_count > 64 || launch->reserved) return CUDA_RC_INVALID;
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || !program_catalog.frozen) return CUDA_RC_INVALID;
    const int context_rc = ensure_program_context(program_catalog);
    if (context_rc) return context_rc;
    ProgramFunction * function = find_program_function(launch->variant_id);
    if (!function || launch->argument_count != function->argument_count
        || launch->dynamic_shared_bytes > function->dynamic_shared_bytes_max
        || launch->grid_x > program_catalog.max_grid[0]
        || launch->grid_y > program_catalog.max_grid[1]
        || launch->grid_z > program_catalog.max_grid[2]
        || (g.graph_capturing && function->graph_capture == 0)) {
        return CUDA_RC_INVALID;
    }
    bool selected = false;
    for (const auto & binding : program_catalog.bindings) {
        selected = selected
            || (bytes_equal(binding.first.data(), function->choice_group_id, 32)
                && bytes_equal(binding.second.data(), launch->variant_id, 32));
    }
    if (!selected) return CUDA_RC_INVALID;

    ProgramArgumentStorage storage[64] = {};
    void * parameters[64] = {};
    for (uint32_t i = 0; i < launch->argument_count; ++i) {
        const auto & argument = launch->arguments[i];
        const auto & descriptor = function->argument_schema[i];
        if (argument.reserved || argument.kind < IMPARO_CUDA_PROGRAM_TENSOR_PTR
            || argument.kind > IMPARO_CUDA_PROGRAM_MANIFEST_U32
            || argument.kind != descriptor.kind) {
            return CUDA_RC_INVALID;
        }
        const bool narrow = argument.kind == IMPARO_CUDA_PROGRAM_SCALAR_I32
            || argument.kind == IMPARO_CUDA_PROGRAM_SCALAR_U32
            || argument.kind == IMPARO_CUDA_PROGRAM_SCALAR_F32
            || argument.kind == IMPARO_CUDA_PROGRAM_MANIFEST_U32;
        if (narrow && argument.value > UINT32_MAX) return CUDA_RC_INVALID;
        const bool pointer = argument.kind == IMPARO_CUDA_PROGRAM_TENSOR_PTR
            || argument.kind == IMPARO_CUDA_PROGRAM_STATE_PTR
            || argument.kind == IMPARO_CUDA_PROGRAM_SCRATCH_PTR;
        if (pointer && !argument.value) return CUDA_RC_INVALID;
        if (narrow) {
            storage[i].u32 = uint32_t(argument.value);
            parameters[i] = &storage[i].u32;
        } else {
            storage[i].u64 = argument.value;
            parameters[i] = &storage[i].u64;
        }
    }
    const CUresult result = program_catalog.driver.launch_kernel(function->function,
        launch->grid_x, launch->grid_y, launch->grid_z,
        function->block[0], function->block[1], function->block[2],
        launch->dynamic_shared_bytes, reinterpret_cast<CUstream>(g.stream),
        parameters, nullptr);
    if (result == CUDA_SUCCESS && g.graph_capturing
        && function->graph_capture == 2) {
        ++g.graph_expected_dynamic_nodes;
    }
    if (fatal_driver_launch(result)) program_catalog.poisoned = true;
    return result == CUDA_SUCCESS ? 0 : CUDA_RC_ERROR;
}

int program_reset_impl() {
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || g.graph_capturing || g.forward_active) {
        return CUDA_RC_INVALID;
    }
    if (program_catalog.modules.empty()) {
        program_catalog.functions.clear();
        program_catalog.bindings.clear();
        if (program_catalog.frozen) ++program_catalog.generation;
        program_catalog.frozen = false;
        return 0;
    }
    const int quiesce_rc = quiesce_program_catalog(program_catalog);
    if (quiesce_rc) return quiesce_rc;
    if (!unload_program_modules(program_catalog, program_catalog.modules)) {
        program_catalog.poisoned = true;
        return CUDA_RC_ERROR;
    }
    program_catalog.functions.clear();
    program_catalog.bindings.clear();
    program_catalog.frozen = false;
    std::memset(program_catalog.pack_set_sha256, 0, 32);
    std::memset(program_catalog.candidate_catalog_sha256, 0, 32);
    std::memset(program_catalog.eligible_candidate_set_sha256, 0, 32);
    ++program_catalog.generation;
    return 0;
}

template <typename Callable>
int program_abi_guard(Callable && callable) noexcept {
    try {
        return callable();
    } catch (const std::bad_alloc &) {
        return CUDA_RC_OOM;
    } catch (...) {
        return CUDA_RC_ERROR;
    }
}

} // namespace

extern "C" int imparo_cuda_program_pack_install(
        const ImparoCudaProgramPackWire * pack, uint32_t struct_size) noexcept {
    return program_abi_guard([&] { return program_pack_install_impl(pack, struct_size); });
}

extern "C" int imparo_cuda_program_catalog_identity(
        ImparoCudaProgramCatalogIdentityWire * out, uint32_t struct_size) noexcept {
    return program_abi_guard([&] { return program_catalog_identity_impl(out, struct_size); });
}

extern "C" int imparo_cuda_program_bind(
        const uint8_t group_id[32], const uint8_t variant_id[32]) noexcept {
    return program_abi_guard([&] { return program_bind_impl(group_id, variant_id); });
}

extern "C" int imparo_cuda_program_freeze(void) noexcept {
    return program_abi_guard([] { return program_freeze_impl(); });
}

extern "C" int imparo_cuda_program_launch(
        const ImparoCudaProgramLaunchWire * launch, uint32_t struct_size) noexcept {
    return program_abi_guard([&] { return program_launch_impl(launch, struct_size); });
}

extern "C" int imparo_cuda_program_reset(void) noexcept {
    return program_abi_guard([] { return program_reset_impl(); });
}
