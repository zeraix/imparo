// Test-only hardware hooks for the Program Pack SM86 smoke. This file is included
// only when IMPARO_CUDA_PROGRAM_SMOKE=1 and its symbols are intentionally absent
// from imparo_cuda.def, so none of this surface is part of the release ABI.

namespace {

int program_test_context() {
    return ensure_program_context(program_catalog);
}

int program_test_alloc_impl(uint64_t bytes, uint64_t * pointer_out) {
    if (!bytes || !pointer_out || program_test_context()) return CUDA_RC_INVALID;
    void * pointer = nullptr;
    if (cudaMalloc(&pointer, size_t(bytes)) != cudaSuccess || !pointer) {
        return CUDA_RC_OOM;
    }
    *pointer_out = uint64_t(reinterpret_cast<uintptr_t>(pointer));
    return 0;
}

int program_test_free_impl(uint64_t pointer) {
    if (!pointer || program_test_context()) return CUDA_RC_INVALID;
    return cudaFree(reinterpret_cast<void *>(uintptr_t(pointer))) == cudaSuccess
        ? 0 : CUDA_RC_ERROR;
}

int program_test_write_u32_impl(uint64_t pointer, uint32_t value) {
    if (!pointer || program_test_context()) return CUDA_RC_INVALID;
    return cudaMemcpy(reinterpret_cast<void *>(uintptr_t(pointer)), &value,
                      sizeof(value), cudaMemcpyHostToDevice) == cudaSuccess
        ? 0 : CUDA_RC_ERROR;
}

int program_test_read_u32_impl(uint64_t pointer, uint32_t * value_out) {
    if (!pointer || !value_out || program_test_context()) return CUDA_RC_INVALID;
    if (g.stream && cudaStreamSynchronize(g.stream) != cudaSuccess) {
        return CUDA_RC_ERROR;
    }
    return cudaMemcpy(value_out, reinterpret_cast<const void *>(uintptr_t(pointer)),
                      sizeof(*value_out), cudaMemcpyDeviceToHost) == cudaSuccess
        ? 0 : CUDA_RC_ERROR;
}

int program_test_synchronize_impl() {
    if (program_test_context() || !g.stream) return CUDA_RC_INVALID;
    return cudaStreamSynchronize(g.stream) == cudaSuccess ? 0 : CUDA_RC_ERROR;
}

int program_test_graph_begin_impl() {
    std::lock_guard<std::mutex> lock(program_catalog.mutex);
    if (program_catalog.poisoned || !program_catalog.frozen
        || g.graph_capturing || g.forward_active || g.decode_graph
        || g.decode_graph_exec || program_test_context()) {
        return CUDA_RC_INVALID;
    }
    if (cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal)
            != cudaSuccess) {
        return CUDA_RC_ERROR;
    }
    g.decode_graph_nodes.clear();
    g.graph_expected_dynamic_nodes = 0;
    g.graph_capturing = true;
    return 0;
}

int program_test_graph_end_replay_impl(uint32_t replay_count) {
    if (!g.graph_capturing || !replay_count) return CUDA_RC_INVALID;
    const cudaError_t end = cudaStreamEndCapture(g.stream, &g.decode_graph);
    g.graph_capturing = false;
    if (end != cudaSuccess || !g.decode_graph) {
        (void)destroy_decode_graph_checked();
        return 101;
    }
    if (cudaGraphInstantiate(&g.decode_graph_exec, g.decode_graph,
                            nullptr, nullptr, 0) != cudaSuccess
        || !g.decode_graph_exec) {
        (void)destroy_decode_graph_checked();
        return 102;
    }
    if (!configure_decode_graph_nodes()) {
        (void)destroy_decode_graph_checked();
        return 103;
    }
    for (uint32_t replay = 0; replay < replay_count; ++replay) {
        if (cudaGraphLaunch(g.decode_graph_exec, g.stream) != cudaSuccess) {
            return 104;
        }
    }
    // Deliberately keep the graph registered in the production owner. A following
    // route change or reset must synchronize, destroy it, and only then unload cubin.
    return cudaStreamSynchronize(g.stream) == cudaSuccess ? 0 : 105;
}

int program_test_graph_replay_impl(uint32_t decode_start_pos) {
    if (!g.decode_graph || !g.decode_graph_exec || g.graph_capturing
        || g.forward_active) return CUDA_RC_INVALID;
    const int update = update_decode_graph_nodes(decode_start_pos);
    if (update) return update;
    if (cudaGraphLaunch(g.decode_graph_exec, g.stream) != cudaSuccess) {
        return CUDA_RC_ERROR;
    }
    return cudaStreamSynchronize(g.stream) == cudaSuccess ? 0 : CUDA_RC_ERROR;
}

} // namespace

extern "C" int imparo_cuda_program_test_alloc(
        uint64_t bytes, uint64_t * pointer_out) noexcept {
    return program_abi_guard([&] { return program_test_alloc_impl(bytes, pointer_out); });
}

extern "C" int imparo_cuda_program_test_free(uint64_t pointer) noexcept {
    return program_abi_guard([&] { return program_test_free_impl(pointer); });
}

extern "C" int imparo_cuda_program_test_write_u32(
        uint64_t pointer, uint32_t value) noexcept {
    return program_abi_guard([&] { return program_test_write_u32_impl(pointer, value); });
}

extern "C" int imparo_cuda_program_test_read_u32(
        uint64_t pointer, uint32_t * value_out) noexcept {
    return program_abi_guard([&] { return program_test_read_u32_impl(pointer, value_out); });
}

extern "C" int imparo_cuda_program_test_synchronize(void) noexcept {
    return program_abi_guard([] { return program_test_synchronize_impl(); });
}

extern "C" int imparo_cuda_program_test_graph_begin(void) noexcept {
    return program_abi_guard([] { return program_test_graph_begin_impl(); });
}

extern "C" int imparo_cuda_program_test_graph_end_replay(
        uint32_t replay_count) noexcept {
    return program_abi_guard([&] { return program_test_graph_end_replay_impl(replay_count); });
}

extern "C" int imparo_cuda_program_test_graph_replay(
        uint32_t decode_start_pos) noexcept {
    return program_abi_guard([&] {
        return program_test_graph_replay_impl(decode_start_pos);
    });
}

extern "C" uint32_t imparo_cuda_program_test_graph_alive(void) noexcept {
    return (g.decode_graph || g.decode_graph_exec) ? 1u : 0u;
}
