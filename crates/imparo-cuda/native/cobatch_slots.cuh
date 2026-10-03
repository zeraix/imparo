#pragma once

// Conversation state belongs to an execution owner, not to the process or a weight
// allocation. The selected slot lives in ExecutionState's existing one-row view;
// inactive slots own only their recurrent buffers, rings and page tables. Shared
// activation scratch and the full-attention KV arena are never duplicated.
struct CudaConversationSlot {
    void * buffers[2] = {};
    uint64_t sizes[2] = {};
    void * ring_k[MAX_LAYERS] = {};
    void * ring_v[MAX_LAYERS] = {};
    bool ring_owned[MAX_LAYERS] = {};
    void * pages_device = nullptr;
    imparo_cuda_kv::PageTables<MAX_LAYERS> pages;
    bool made = false;
};
struct CudaConversationSlots {
    std::vector<CudaConversationSlot> inactive;
    std::vector<uint32_t> rings;
    uint32_t selected = 0;
    bool live_ring_owned[MAX_LAYERS] = {};
};
static constexpr uint32_t kConversationBuffers[2] = {25, 26}; // Recur, RecurSnap

// Ownership accounting excludes the selected slot, which is already counted by
// ExecutionState. Borrowed slot-zero rings remain part of the shared KV arena.
static uint64_t conversation_slot_bytes(const CudaConversationSlots & slots) {
    uint64_t bytes = 0;
    for (const auto & slot : slots.inactive) {
        for (unsigned i = 0; i < 2; ++i)
            if (slot.buffers[i]) bytes += slot.sizes[i];
        if (slot.pages_device) bytes += slot.pages.arena_entries * sizeof(uint32_t);
    }
    return bytes;
}
