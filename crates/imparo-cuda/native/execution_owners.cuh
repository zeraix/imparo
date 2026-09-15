#pragma once

// Internal bridge, not a new public backend ABI. The model runner must serialize
// native calls. A view can change only between closed forwards; this does not yet
// expose the layer-phase yield needed for cross-request FFN grouping.
struct ExecutionOwner {
    uint64_t id;
    std::unique_ptr<ExecutionState> state;
    bool retiring = false;
};
std::vector<ExecutionOwner> execution_owners;
uint64_t next_execution_owner_id = 1;
uint64_t active_execution_owner_id = 0;

bool execution_boundary_closed() {
    const auto & s = execution();
    return !s.forward_open && !s.forward_active && !s.graph_capturing
        && !s.prefill_capture_active && !s.decode_prepared
        && !s.prefill_prepared && !s.pending_error;
}

#if defined(IMPARO_CUDA_EXECUTION_OWNER_LAB)
int verification_m1_quant_exchange_lab(int enabled) {
    auto & state = execution();
    const int previous = state.verification_m1_quant_active ? 1 : 0;
    state.verification_m1_quant_active = enabled != 0;
    return previous;
}
#endif

ExecutionState * find_execution_owner(uint64_t id) {
    if (id == 0) return &default_execution;
    for (auto & owner : execution_owners) {
        if (owner.id == id && !owner.retiring) return owner.state.get();
    }
    return nullptr;
}

int execution_owner_create(uint64_t * id) {
    if (!id || !g.runtime_initialized || !g.weights_host
            || !execution_boundary_closed() || !next_execution_owner_id) {
        return CUDA_RC_INVALID;
    }
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    // The optional lab has a separate mutable shared weight cache. Its lifecycle
    // has not been admitted for owner views; do not silently enable that path.
    return CUDA_RC_INVALID;
#else
    try {
        auto state = std::make_unique<ExecutionState>();
        const uint64_t fresh = next_execution_owner_id;
        execution_owners.push_back({fresh, std::move(state)});
        ++next_execution_owner_id; // zero is exhaustion, never reuse old tickets
        *id = fresh;
        return 0;
    } catch (const std::bad_alloc &) {
        return CUDA_RC_OOM;
    }
#endif
}

int execution_owner_select(uint64_t id) {
    if (!execution_boundary_closed()) return CUDA_RC_INVALID;
    auto * next = find_execution_owner(id);
    if (!next || next->graph_leases) return CUDA_RC_INVALID;
    // No allocation, graph reconstruction, state copy or stream wait. Every
    // graph's argument storage and pinned upload source stays with its owner.
    active_execution = next;
    active_execution_owner_id = id;
    return 0;
}

bool execution_graphs_borrowed() {
    if (default_execution.graph_leases) return true;
    for (const auto & owner : execution_owners) if (owner.state->graph_leases) return true;
    return false;
}

bool destroy_all_execution_graphs_checked() {
    if (execution_graphs_borrowed()) return false;
    if (execution().graph_capturing || execution().prefill_capture_active) return false;
    if (!destroy_decode_graph_checked(default_execution)) return false;
    for (auto & owner : execution_owners) {
        if (!destroy_decode_graph_checked(*owner.state)) return false;
    }
    return true;
}

void invalidate_all_execution_choices() {
    const auto invalidate = [](ExecutionState & s) {
        s.decode_graph_shape_blocked = false;
        s.decode_graph_capture_after = 0;
        s.q8_src = UINT32_MAX;
        s.q8_layout = UINT32_MAX;
        s.q8_owner_capture_generation = 0;
    };
    invalidate(default_execution);
    for (auto & owner : execution_owners) invalidate(*owner.state);
}

uint64_t execution_device_bytes(const ExecutionState & s) {
    uint64_t total = s.weight_cache_bytes + s.rope_freqs_bytes
        + s.q8_scratch_bytes + s.q8_scratch_next_bytes + s.device_task_scratch_bytes
        + s.attention_scratch_bytes + s.attention_q_cache_bytes + s.arena_size;
    if (s.decode_control_device) total += sizeof(uint32_t);
    for (int i = 0; i < B_COUNT; ++i) {
        if (s.bufs[i] && !s.in_arena[i]) total += s.sizes[i];
    }
    total += s.kv_layout.arena_bytes;
    total += s.kv_page_tables.arena_entries * sizeof(uint32_t);
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    total += s.cublas_activations_f16_bytes + s.cublas_output_f16_bytes;
#endif
    return total;
}

uint64_t execution_owners_device_bytes() {
    uint64_t total = execution_device_bytes(default_execution);
    for (const auto & owner : execution_owners) total += execution_device_bytes(*owner.state);
    return total;
}

bool release_execution_storage_checked(ExecutionState & s) {
    if (!destroy_decode_graph_checked(s)) return false;
    const auto device = [](void *& p) {
        if (p && cudaFree(p) != cudaSuccess) return false;
        p = nullptr;
        return true;
    };
    const auto pinned = [](void *& p) {
        if (p && cudaFreeHost(p) != cudaSuccess) return false;
        p = nullptr;
        return true;
    };
    const auto event = [](cudaEvent_t & p) {
        if (p && cudaEventDestroy(p) != cudaSuccess) return false;
        p = nullptr;
        return true;
    };
    for (int i = 0; i < B_COUNT; ++i) {
        if (!s.in_arena[i] && !device(s.bufs[i])) return false;
        s.bufs[i] = nullptr;
        s.sizes[i] = 0;
    }
    if (!device(s.arena)) return false;
    s.arena_size = 0;
    if (!device(s.kv_arena)) return false;
    s.kv_layout.arena_bytes = 0;
    if (!device(s.kv_page_table_arena)) return false;
    s.kv_page_tables.arena_entries = 0;
    const auto sized = [&](void *& p, uint64_t & bytes) {
        if (!device(p)) return false;
        bytes = 0;
        return true;
    };
    if (!sized(s.weight_cache, s.weight_cache_bytes)
        || !sized(s.rope_freqs, s.rope_freqs_bytes)
        || !sized(s.q8_scratch, s.q8_scratch_bytes)
        || !sized(s.q8_scratch_next, s.q8_scratch_next_bytes)
        || !sized(s.device_task_scratch, s.device_task_scratch_bytes)
        || !sized(s.attention_scratch, s.attention_scratch_bytes)
        || !sized(s.attention_q_cache, s.attention_q_cache_bytes)
        || !device(s.decode_control_device)) return false;
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    if (!sized(s.cublas_activations_f16, s.cublas_activations_f16_bytes)
        || !sized(s.cublas_output_f16, s.cublas_output_f16_bytes)) return false;
#endif
    return pinned(s.ple_stage_host) && pinned(s.decode_row_stage_host)
        && pinned(s.decode_token_stage_host) && pinned(s.decode_control_host)
        && event(s.forward_start) && event(s.forward_stop);
}

int execution_owner_release(uint64_t id) {
    if (!id || id == active_execution_owner_id || !execution_boundary_closed()) {
        return CUDA_RC_INVALID;
    }
    for (auto it = execution_owners.begin(); it != execution_owners.end(); ++it) {
        if (it->id != id) continue;
        if (it->state->graph_leases) return CUDA_RC_INVALID;
        // Retirement, unlike selection, waits for all queued references. Retain
        // the owner on any failure so a later attempt can finish its teardown.
        if (cudaStreamSynchronize(g.stream) != cudaSuccess) return CUDA_RC_ERROR;
        it->retiring = true;
        clear_q8_l2_window();
        if (!release_execution_storage_checked(*it->state)) return CUDA_RC_ERROR;
        execution_owners.erase(it);
        return 0;
    }
    return CUDA_RC_INVALID;
}

#if defined(IMPARO_CUDA_SPECULATIVE)
// Static execution symbols deliberately absent from imparo_cuda.def.
extern "C" int imparo_cuda_execution_work_demand_lab(
        uint32_t ffn_layers, uint32_t logits_wanted) {
    auto & s = execution();
    if (!execution_boundary_closed() || s.graph_leases
            || !s.kv_layout.layers || ffn_layers > s.kv_layout.layers
            || logits_wanted > 1
            || (logits_wanted && ffn_layers != s.kv_layout.layers)) {
        return CUDA_RC_INVALID;
    }
    const bool changed = !s.work_demand_explicit
        || s.work_ffn_layers != ffn_layers
        || s.work_logits_wanted != (logits_wanted != 0);
    if (changed) {
        // Demand changes the captured body even at identical token geometry.
        // Preserve the existing packed hot-set and decode graph lifetimes.
        // Decode capture already requires argmax=true; state-only uses false.
        const bool had_graph = s.prefill_graph_exec != nullptr;
        if (!destroy_prefill_graph_checked(s)) return CUDA_RC_ERROR;
        s.prefill_warm_forwards = 0;
        s.prefill_graph_blocked = false;
        s.work_demand_explicit = true;
        s.work_ffn_layers = ffn_layers;
        s.work_logits_wanted = logits_wanted != 0;
        if (std::getenv("IMPARO_CUDA_TRACE_GRAPH")) {
            std::fprintf(stderr,
                "[cuda-work-demand] owner=%llu ffn=%u logits=%u invalidated=%u\n",
                (unsigned long long)active_execution_owner_id,
                ffn_layers, logits_wanted, unsigned(had_graph));
        }
    }
    return 0;
}

// Static numeric identity only; no active geometry or public ABI mutation.
extern "C" int imparo_cuda_projection_reference_lab(uint32_t start,uint32_t tokens){
    auto & s=execution();
    if (!s.forward_open || s.forward_decode || s.graph_capturing || s.graph_leases
            || !s.batch_geometry_valid || !s.materialized_geometry_valid
            || g.sm_version!=86 || tokens<=8 || tokens>=128
            || s.materialized_tokens<=8 || s.materialized_tokens>=128
            || uint64_t(start)+tokens!=uint64_t(s.materialized_start)+s.materialized_tokens
            || start<s.batch_canonical_start || tokens>=s.batch_canonical_tokens)
        return CUDA_RC_INVALID;
    s.projection_reference_active=true;s.projection_reference_start=start;s.projection_reference_tokens=tokens;
    if (std::getenv("IMPARO_CUDA_TRACE_GRAPH")) std::fprintf(stderr,
        "[cuda-projection-reference] actual=%u+%u materialized=%u+%u reference=%u+%u\n",
        s.batch_geometry_start,s.batch_geometry_tokens,s.materialized_start,s.materialized_tokens,start,tokens);
    return 0;
}

extern "C" int imparo_cuda_execution_owner_create_lab(uint64_t * id) {
    return execution_owner_create(id);
}
extern "C" int imparo_cuda_execution_owner_select_lab(uint64_t id) {
    return execution_owner_select(id);
}
extern "C" int imparo_cuda_execution_owner_release_lab(uint64_t id) {
    return execution_owner_release(id);
}
#endif
