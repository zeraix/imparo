#pragma once

static bool free_conversation_slot(CudaConversationSlot & slot) {
    auto release = [](void *& p) {
        if (p && cudaFree(p) != cudaSuccess) return false;
        p = nullptr;
        return true;
    };
    for (void *& p : slot.buffers) if (!release(p)) return false;
    for (uint32_t l = 0; l < MAX_LAYERS; ++l) {
        if (slot.ring_owned[l]) {
            if (!release(slot.ring_k[l]) || !release(slot.ring_v[l])) return false;
        }
        slot.ring_k[l] = slot.ring_v[l] = nullptr;
        slot.ring_owned[l] = false;
    }
    if (!release(slot.pages_device)) return false;
    slot.pages = {};
    slot.made = false;
    return true;
}

static uint64_t conversation_ring_bytes(const ExecutionState & e) {
    uint64_t bytes = 0;
    for (uint32_t l : e.conversation_slots.rings) {
        if (e.conversation_slots.live_ring_owned[l]) {
            if (e.kv_k[l]) bytes += e.kv_bytes[l];
            if (e.kv_v[l]) bytes += e.kv_bytes[l];
        }
        for (const auto & slot : e.conversation_slots.inactive) {
            if (slot.ring_owned[l]) {
                if (slot.ring_k[l]) bytes += e.kv_bytes[l];
                if (slot.ring_v[l]) bytes += e.kv_bytes[l];
            }
        }
    }
    return bytes;
}

static bool release_conversation_slots_checked(ExecutionState & e) {
    for (auto & slot : e.conversation_slots.inactive)
        if (!free_conversation_slot(slot)) return false;
    for (uint32_t l : e.conversation_slots.rings) {
        if (!e.conversation_slots.live_ring_owned[l]) continue;
        if (e.kv_k[l] && cudaFree(e.kv_k[l]) != cudaSuccess) return false;
        e.kv_k[l] = nullptr;
        if (e.kv_v[l] && cudaFree(e.kv_v[l]) != cudaSuccess) return false;
        e.kv_v[l] = nullptr;
        e.conversation_slots.live_ring_owned[l] = false;
    }
    e.conversation_slots = {};
    return true;
}

// Only host bindings change; kernels already queued on the owner's stream retain
// their explicit arguments. This operation never switches model execution owners.
static void conversation_slot_swap(ExecutionState & e, uint32_t index) {
    auto & slot = e.conversation_slots.inactive[index];
    std::swap(e.kv_page_table_arena, slot.pages_device);
    std::swap(e.kv_page_tables, slot.pages);
    for (unsigned i = 0; i < 2; ++i) {
        const uint32_t bid = kConversationBuffers[i];
        std::swap(e.bufs[bid], slot.buffers[i]);
        std::swap(e.sizes[bid], slot.sizes[i]);
    }
    for (uint32_t l : e.conversation_slots.rings) {
        std::swap(e.kv_k[l], slot.ring_k[l]);
        std::swap(e.kv_v[l], slot.ring_v[l]);
        std::swap(e.conversation_slots.live_ring_owned[l], slot.ring_owned[l]);
    }
}

// Transactional: no live state is published until every allocation, zero-fill and
// page-table upload succeeds. New slots start with identity metadata; the pool
// installs the admitted request's mapping before any forward uses it.
static int make_conversation_slot(ExecutionState & e, uint32_t index) {
    auto & target = e.conversation_slots.inactive[index];
    if (target.made) return 0;
    CudaConversationSlot fresh;
    try {
        fresh.pages = e.kv_page_tables;
        for (uint32_t l = 0; l < fresh.pages.layers; ++l) {
            auto & table = fresh.pages.layer[l];
            table.installed_len = 0;
            table.generation = 0;
            for (uint32_t i = 0; i < table.capacity; ++i) table.host_shadow[i] = i;
        }
    } catch (const std::bad_alloc &) { return CUDA_RC_OOM; }
    int rc = 0;
    auto allocate = [&](void ** p, uint64_t bytes) {
        if (rc || !bytes) return;
        rc = alloc_raw_with_packed_q4_reclaim(p, bytes, "CoBatch conversation state");
        if (!rc && cudaMemsetAsync(*p, 0, size_t(bytes), g.stream) != cudaSuccess)
            rc = CUDA_RC_ERROR;
    };
    for (unsigned i = 0; i < 2; ++i) {
        fresh.sizes[i] = e.sizes[kConversationBuffers[i]];
        allocate(&fresh.buffers[i], fresh.sizes[i]);
    }
    for (uint32_t l : e.conversation_slots.rings) {
        fresh.ring_owned[l] = true;
        allocate(&fresh.ring_k[l], e.kv_bytes[l]);
        allocate(&fresh.ring_v[l], e.kv_bytes[l]);
    }
    allocate(&fresh.pages_device, fresh.pages.arena_entries * sizeof(uint32_t));
    for (uint32_t l = 0; l < fresh.pages.layers && !rc; ++l) {
        const auto & table = fresh.pages.layer[l];
        if (table.capacity && cudaMemcpyAsync(
                static_cast<uint32_t *>(fresh.pages_device) + table.arena_offset,
                table.host_shadow.data(), size_t(table.capacity) * sizeof(uint32_t),
                cudaMemcpyHostToDevice, g.stream) != cudaSuccess) rc = CUDA_RC_ERROR;
    }
    if (cudaStreamSynchronize(g.stream) != cudaSuccess) rc = CUDA_RC_ERROR;
    if (rc) {
        if (!free_conversation_slot(fresh)) set_pending(CUDA_RC_ERROR, "CoBatch rollback");
        return rc;
    }
    fresh.made = true;
    target = std::move(fresh);
    return 0;
}

// Already committed planner categories, for admission only. Count actual owner
// storage without aliases, conversation copies, weight caches or scratch. The
// caller caps each credit by its matching planned reserve and retains all margin.
extern "C" int imparo_cuda_cobatch_committed_runtime(uint64_t * kv, uint64_t * activations) {
    if (!kv || !activations) return CUDA_RC_INVALID;
    const auto & e = execution();
    *kv = e.kv_layout.arena_bytes;
    uint64_t bytes = e.arena_size;
    for (uint32_t i = 0; i < B_COUNT; ++i) {
        if (i == kConversationBuffers[0] || i == kConversationBuffers[1]) continue;
        if (e.bufs[i] && !e.in_arena[i]) bytes += e.sizes[i];
    }
    *activations = bytes;
    return 0;
}

extern "C" int imparo_cuda_set_slots(uint32_t n, const uint32_t * rings, uint32_t nrings) {
    auto & e = execution();
    auto & slots = e.conversation_slots;
    if (!n || n > 65536 || (nrings && !rings) || nrings > MAX_LAYERS
            || !execution_boundary_closed() || e.graph_leases) return CUDA_RC_INVALID;
    // Recurrent planes are intentionally outside shared activation arenas.
    for (uint32_t bid : kConversationBuffers)
        if (e.in_arena[bid]) return CUDA_RC_INVALID;
    for (uint32_t i = 0; i < nrings; ++i) {
        const uint32_t l = rings[i];
        if (l >= e.kv_layout.layers || !e.kv_k[l] || !e.kv_v[l]
                || std::find(rings, rings + i, l) != rings + i) return CUDA_RC_INVALID;
    }
    if (!slots.inactive.empty()) {
        if (nrings != slots.rings.size()
                || (nrings && !std::equal(slots.rings.begin(), slots.rings.end(), rings)))
            return CUDA_RC_INVALID;
    }
    try {
        if (slots.inactive.empty()) {
            std::vector<uint32_t> ring_copy;
            if (nrings) ring_copy.assign(rings, rings + nrings);
            std::vector<CudaConversationSlot> fresh(n);
            fresh[0].made = true;
            slots.rings = std::move(ring_copy);
            slots.inactive = std::move(fresh);
        } else if (n > slots.inactive.size()) slots.inactive.resize(n);
    } catch (const std::bad_alloc &) { return CUDA_RC_OOM; }
    return 0;
}

extern "C" int imparo_cuda_select_slot(uint32_t index) {
    auto & e = execution();
    auto & slots = e.conversation_slots;
    if (index == slots.selected) return 0;
    if (index >= slots.inactive.size() || !execution_boundary_closed() || e.graph_leases)
        return CUDA_RC_INVALID;
    const int rc = make_conversation_slot(e, index);
    if (rc) return rc;
    // Captured one-row graphs hold conversation pointers. Never replay a previous
    // slot's graph for a new request; batching will have its own descriptor graph.
    if (!destroy_decode_graph_checked(e)) return CUDA_RC_ERROR;
    e.tree_replay.reset();
    conversation_slot_swap(e, slots.selected);
    conversation_slot_swap(e, index);
    slots.selected = index;
    e.kdq.clear(); e.vdq.clear();
    e.decode_phase_active = false;
    e.decode_sequence_start_pos = e.decode_previous_start_pos = UINT32_MAX;
    e.q8_src = UINT32_MAX;
    e.attention_q_src = UINT32_MAX;
    for (uint32_t bid : kConversationBuffers) mark_buf_written(bid);
    return 0;
}

extern "C" int imparo_cuda_release_slot(uint32_t index) {
    auto & e = execution();
    auto & slots = e.conversation_slots;
    if (index >= slots.inactive.size() || index == slots.selected
            || !slots.inactive[index].made || !execution_boundary_closed() || e.graph_leases)
        return CUDA_RC_INVALID;
    // Releasing state is forbidden while any submitted work still references it.
    const cudaError_t ready = cudaStreamQuery(g.stream);
    if (ready != cudaSuccess) return ready == cudaErrorNotReady ? CUDA_RC_INVALID : CUDA_RC_ERROR;
    return free_conversation_slot(slots.inactive[index]) ? 0 : CUDA_RC_ERROR;
}

// Prepare every inactive page table before replacing the shared KV allocation.
// A failed grow changes no conversation; successful commit preserves each one's
// mapping, installed length and generation while expanding its physical bounds.
struct ConversationPageGrowth {
    std::vector<CudaConversationSlot> pages;
    ~ConversationPageGrowth() {
        for (auto & slot : pages)
            if (slot.pages_device && cudaFree(slot.pages_device) != cudaSuccess)
                set_pending(CUDA_RC_ERROR, "CoBatch page-table retirement");
    }
};
static int prepare_conversation_page_growth(
        ExecutionState & e, uint32_t layers, const uint64_t * bytes,
        const imparo_cuda_kv::PagingLayout * layouts, uint32_t count,
        ConversationPageGrowth & fresh) {
    if (e.conversation_slots.inactive.empty()) return 0;
    for (uint32_t l : e.conversation_slots.rings)
        if (l >= layers || bytes[l] != e.kv_bytes[l]) return CUDA_RC_INVALID;
    try { fresh.pages.resize(e.conversation_slots.inactive.size()); }
    catch (const std::bad_alloc &) { return CUDA_RC_OOM; }
    for (size_t i = 0; i < fresh.pages.size(); ++i) {
        const auto & old = e.conversation_slots.inactive[i];
        if (i == e.conversation_slots.selected || !old.made) continue;
        auto & next = fresh.pages[i];
        const auto rc = imparo_cuda_kv::build_page_tables(
            layers, bytes, layouts, count, &old.pages, true, &next.pages);
        if (rc != imparo_cuda_kv::PagingRc::Ok) return paging_rc(rc);
        if (next.pages.arena_entries) {
            const int alloc = alloc_raw_with_packed_q4_reclaim(&next.pages_device,
                next.pages.arena_entries * sizeof(uint32_t), "CoBatch grown page table");
            if (alloc) return alloc;
        }
        for (uint32_t l = 0; l < layers; ++l) {
            const auto & table = next.pages.layer[l];
            if (table.capacity && cudaMemcpyAsync(
                    static_cast<uint32_t *>(next.pages_device) + table.arena_offset,
                    table.host_shadow.data(), size_t(table.capacity) * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, g.stream) != cudaSuccess) return CUDA_RC_ERROR;
        }
        next.made = true;
    }
    return cudaStreamSynchronize(g.stream) == cudaSuccess ? 0 : CUDA_RC_ERROR;
}
static void commit_conversation_page_growth(ExecutionState & e, ConversationPageGrowth & fresh) {
    for (size_t i = 0; i < fresh.pages.size(); ++i) {
        auto & next = fresh.pages[i];
        if (!next.made) continue;
        auto & old = e.conversation_slots.inactive[i];
        std::swap(old.pages_device, next.pages_device);
        std::swap(old.pages, next.pages);
    }
    // Slot zero initially borrows its rings from the KV arena; other slots own
    // separate rings. A grow must rebase only borrowed addresses, never overwrite
    // a selected request's private ring with slot zero's bytes.
    for (uint32_t l : e.conversation_slots.rings) {
        void * k = static_cast<uint8_t *>(e.kv_arena) + e.kv_layout.k_offset[l];
        void * v = static_cast<uint8_t *>(e.kv_arena) + e.kv_layout.v_offset[l];
        if (!e.conversation_slots.live_ring_owned[l]) { e.kv_k[l] = k; e.kv_v[l] = v; }
        for (size_t i = 0; i < e.conversation_slots.inactive.size(); ++i) {
            auto & slot = e.conversation_slots.inactive[i];
            if (i != e.conversation_slots.selected && slot.made && !slot.ring_owned[l]) {
                slot.ring_k[l] = k; slot.ring_v[l] = v;
            }
        }
    }
}
