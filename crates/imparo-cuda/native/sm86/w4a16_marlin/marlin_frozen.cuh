#pragma once
// Exact frozen vLLM 0.28.0 Marlin image. No arithmetic rebuild or new owner.
// Original kernel: upstream/marlin_template.h; Apache-2.0, upstream/LICENSE.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace imparo_w4a16_marlin {
inline constexpr const char* kSymbol = "_ZN6marlin6MarlinILl1125899906910725ELl1125899907892224ELl1125899906910725ELl1125899906910725ELi256ELi1ELi8ELi8ELb1ELi4ELi2ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiiiiiiPibbbi";
inline constexpr const char* kCubinSha256 =
    "9521406b46b918c23450e6199ed973149c88fcb2e33e5a7d8348212f7694cf37";
inline constexpr std::size_t kCubinBytes = 4348832;
inline constexpr std::size_t kReduceBytes = 30u * 16u * 256u * sizeof(float);
inline constexpr std::size_t kLockBytes = 30u * sizeof(std::int32_t);
inline constexpr int kPhysicalM = 3;

// Resolve only existing-context Driver operations. No cuda.lib link dependency,
// driver installation, context creation, allocator, synchronization, or fallback.
struct Driver {
    HMODULE dll = nullptr;
    decltype(&cuCtxGetCurrent) ctx_get_current = nullptr;
    decltype(&cuCtxGetDevice) ctx_get_device = nullptr;
    decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
    decltype(&cuModuleLoadDataEx) module_load_data = nullptr;
    decltype(&cuModuleGetFunction) module_function = nullptr;
    decltype(&cuFuncSetAttribute) function_attribute = nullptr;
    decltype(&cuLaunchKernel) launch_kernel = nullptr;
    decltype(&cuGraphKernelNodeGetParams) graph_params = nullptr;
    decltype(&cuModuleUnload) module_unload = nullptr;
};
struct Kernel {
    Driver driver{};
    CUmodule module = nullptr;
    CUfunction function = nullptr; // stable Graph identity; lifetime owned by caller
    CUcontext context = nullptr;
    int max_shared_memory = 0;
};
inline bool load_driver(Driver& d) {
    d.dll = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!d.dll) return false;
#define IMPARO_MARLIN_DRIVER(field, symbol) \
    d.field = reinterpret_cast<decltype(d.field)>(GetProcAddress(d.dll, #symbol))
    IMPARO_MARLIN_DRIVER(ctx_get_current, cuCtxGetCurrent);
    IMPARO_MARLIN_DRIVER(ctx_get_device, cuCtxGetDevice);
    IMPARO_MARLIN_DRIVER(device_attribute, cuDeviceGetAttribute);
    IMPARO_MARLIN_DRIVER(module_load_data, cuModuleLoadDataEx);
    IMPARO_MARLIN_DRIVER(module_function, cuModuleGetFunction);
    IMPARO_MARLIN_DRIVER(function_attribute, cuFuncSetAttribute);
    IMPARO_MARLIN_DRIVER(launch_kernel, cuLaunchKernel);
    d.graph_params = reinterpret_cast<decltype(d.graph_params)>(GetProcAddress(d.dll, "cuGraphKernelNodeGetParams_v2"));
    IMPARO_MARLIN_DRIVER(module_unload, cuModuleUnload);
#undef IMPARO_MARLIN_DRIVER
    if (d.ctx_get_current && d.ctx_get_device && d.device_attribute && d.module_load_data
        && d.module_function && d.function_attribute && d.launch_kernel && d.graph_params && d.module_unload)
        return true;
    FreeLibrary(d.dll); d = {}; return false;
}

// Existing owner calls this outside capture with the exact Rust-authenticated
// immutable image slice. Load synchronously from these bytes, never reread a path.
// The current CUDA context/device must be the owner's. Device is CUDA ordinal.
inline CUresult initialize_image(const uint8_t* image, std::size_t bytes, int device, Kernel* out) {
    if (!image || bytes != kCubinBytes || std::memcmp(image,"\x7f" "ELF",4) != 0
            || !out || out->module || out->function || out->driver.dll)
        return CUDA_ERROR_INVALID_VALUE;
    Kernel next{};
    if (!load_driver(next.driver)) return CUDA_ERROR_NOT_INITIALIZED;
    auto fail = [&](CUresult rc) {
        if (next.module) (void)next.driver.module_unload(next.module);
        FreeLibrary(next.driver.dll);
        return rc;
    };
    CUresult rc = next.driver.ctx_get_current(&next.context);
    if (rc != CUDA_SUCCESS) return fail(rc);
    if (!next.context) return fail(CUDA_ERROR_INVALID_CONTEXT);
    CUdevice actual_device{};
    if ((rc = next.driver.ctx_get_device(&actual_device)) != CUDA_SUCCESS) return fail(rc);
    if (actual_device != device) return fail(CUDA_ERROR_INVALID_DEVICE);
    int major = 0, minor = 0, sms = 0;
    if ((rc = next.driver.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, actual_device)) != CUDA_SUCCESS) return fail(rc);
    if ((rc = next.driver.device_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, actual_device)) != CUDA_SUCCESS) return fail(rc);
    if ((rc = next.driver.device_attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, actual_device)) != CUDA_SUCCESS) return fail(rc);
    if (major != 8 || minor != 6 || sms != 30) return fail(CUDA_ERROR_NOT_SUPPORTED);
    if ((rc = next.driver.device_attribute(&next.max_shared_memory,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, actual_device)) != CUDA_SUCCESS) return fail(rc);
    // Host planner needs 53248 B, but the tested launch uses the full device
    // opt-in limit, passed both as dynamic shared bytes and final kernel int.
    if (next.max_shared_memory < 53248) return fail(CUDA_ERROR_NOT_SUPPORTED);
    if ((rc = next.driver.module_load_data(&next.module, image, 0, nullptr, nullptr)) != CUDA_SUCCESS) return fail(rc);
    if ((rc = next.driver.module_function(&next.function, next.module, kSymbol)) != CUDA_SUCCESS) return fail(rc);
    if ((rc = next.driver.function_attribute(next.function,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, next.max_shared_memory)) != CUDA_SUCCESS) return fail(rc);
    *out = next;
    return CUDA_SUCCESS;
}

// Physical M=3 by default; M4 reuses the same frozen eight-row tile/reduction.
// lda=K half elements. Logical M1 must have two zero rows in the M3 route.
// Owner supplies buffer capacities, same context/stream, immutable packed B/S.
// Ctmp >=491520 B: no initialization required. locks >=120 B: initialize to zero
// before first use; successful non-atomic completion resets them. On any failed
// execution discard/reset the transaction, never reuse uncertain locks/fallback.
inline CUresult launch(const Kernel& kernel,
        CUdeviceptr a, CUdeviceptr packed_weight, CUdeviceptr half_scales,
        CUdeviceptr c, CUdeviceptr reduce_tmp, CUdeviceptr locks,
        int k, int n, CUstream stream, int physical_m = kPhysicalM) {
    if (!kernel.module || !kernel.function || !kernel.driver.launch_kernel
        || !a || !packed_weight || !half_scales || !c || !reduce_tmp || !locks
        || ((a | packed_weight | half_scales | c | reduce_tmp) & 15u)
        || (locks & 3u) || kernel.max_shared_memory < 53248
        || (physical_m != 3 && physical_m != 4)
        || !((k == 2560 && n == 20480) || (k == 10240 && n == 2560)))
        return CUDA_ERROR_INVALID_VALUE;
    CUdeviceptr bias = 0, a_scales = 0, global_scale = 0, zp = 0, g_idx = 0;
    int groups = k / 32, m = physical_m, lda = k;
    bool has_bias = false, use_atomic_add = false, use_fp32_reduce = true;
    int max_shared = kernel.max_shared_memory;
    static_assert(sizeof(bool) == 1, "Frozen kernel bool ABI is one byte");
    static_assert(sizeof(CUdeviceptr) == 8, "Frozen kernel pointer ABI is 64-bit");
    void* args[] = {&a, &packed_weight, &c, &reduce_tmp, &bias, &a_scales,
        &half_scales, &global_scale, &zp, &g_idx, &groups, &m, &n, &k, &lda,
        &locks, &has_bias, &use_atomic_add, &use_fp32_reduce, &max_shared};
    return kernel.driver.launch_kernel(kernel.function, 30, 1, 1, 256, 1, 1,
        static_cast<unsigned>(max_shared), stream, args, nullptr);
}

// Runtime-buffer overload; converts addresses only, retaining the exact ABI.
inline CUresult launch(const Kernel& kernel,
        const void* a, const void* packed_weight, const void* half_scales,
        void* c, void* reduce_tmp, void* locks, int k, int n, CUstream stream,
        int physical_m = kPhysicalM) {
    return launch(kernel, reinterpret_cast<CUdeviceptr>(a),
        reinterpret_cast<CUdeviceptr>(packed_weight), reinterpret_cast<CUdeviceptr>(half_scales),
        reinterpret_cast<CUdeviceptr>(c), reinterpret_cast<CUdeviceptr>(reduce_tmp),
        reinterpret_cast<CUdeviceptr>(locks), k, n, stream, physical_m);
}

// Only after all launches and Graphs referencing this module have completed.
inline CUresult dispose(Kernel* kernel) {
    if (!kernel) return CUDA_ERROR_INVALID_VALUE;
    if (kernel->module) {
        const CUresult rc = kernel->driver.module_unload(kernel->module);
        if (rc != CUDA_SUCCESS) return rc;
    }
    if (kernel->driver.dll) FreeLibrary(kernel->driver.dll);
    *kernel = {};
    return CUDA_SUCCESS;
}
} // namespace imparo_w4a16_marlin
