// CUDA backend host glue + kernels. Official releases compile this source once per
// supported SM into an independently downloadable library. The contracts mirror
// crates/imparo-metal/native/imparo_metal.mm
// operation by operation; where that file documents a bit-layout (Q4_0 18-byte
// blocks, the ring slot mapping `pos & ring`, rope-neox pairs, the online-softmax
// order), THIS file must match it exactly, because the gemma4 workflow above the
// Backend trait assumes one semantics across backends.
//
// Style follows the fork's ggml-cuda (same ops, same quants) and the salvaged
// in-repo prior art (docs_v2/reference/cuda-expert/): cuda_check + RAII allocations,
// extern "C" status-returning surface, a .def-style export list for the Windows DLL.
//
// Deliberately UNTUNED: straightforward kernels a profiler can then shape. CUDA
// performance knobs (threads-per-block, k-split, stages, vector widths, tensor-core
// paths) belong to the backend knob registry (src/knobs.rs) and the CUDA tuner
// sweeps -- not to hand-guessing on a machine without the GPU.

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <dlfcn.h>
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if defined(__has_include)
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
// Minimal CUDA toolkit installations can omit this header even though the CUDA
// Runtime API still exports and documents the profiler-control entry points.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
#else
#include <cuda_profiler_api.h>
#endif
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
#include <cublas_v2.h>
#endif
#include <mma.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include <utility>
#include <array>

#include "sm80/mma_f16.cuh"
#include "host_memory.cuh"
#include "host_profile.cuh"
#include "kv_memory.cuh"
#include "kv_paging.cuh"
#include "lfm2_ops.cuh"
#include "sm86/shortconv_decode_fused.cuh"
#include "sm80/mmq_q8_replay_plan.h"
#include "sm86/rms_norm_add_dual_q8_ready.cuh"

#ifndef IMPARO_CUDA_BACKEND_ABI
#error "define IMPARO_CUDA_BACKEND_ABI from cuda-sm.json"
#endif
#ifndef IMPARO_CUDA_BUILD_SHA256
#error "define IMPARO_CUDA_BUILD_SHA256 from the shared build-identity producer"
#endif
static_assert(sizeof(IMPARO_CUDA_BUILD_SHA256) == 65,
              "IMPARO_CUDA_BUILD_SHA256 must be 64 hex characters");

struct ImparoCudaRuntimeIdentityWire {
    uint32_t struct_bytes;
    uint32_t backend_abi;
    uint32_t device_sm;
    uint32_t driver_version;
    uint32_t runtime_version;
    uint8_t device_uuid[16];
    uint8_t backend_build_sha256[32];
};

struct ImparoCudaHostProfileWire {
    uint32_t struct_bytes;
    uint32_t reserved;
    uint64_t available_host_bytes;
    uint64_t pinned_h2d_bytes_per_second;
    uint64_t pinned_d2h_bytes_per_second;
};
struct ImparoCudaDeviceProfileWire {
    uint32_t struct_bytes;
    uint32_t max_threads;
    uint64_t threadgroup_bytes;
};
struct ImparoCudaDispatchProofWire {
    uint32_t struct_bytes;
    uint32_t family;
    uint64_t choice_epoch;
    uint64_t observed_choice_epoch;
    uint64_t dispatches;
    uint64_t variant;
    uint64_t expected_knob_mask;
    uint64_t observed_knob_mask;
};
static_assert(sizeof(ImparoCudaDispatchProofWire) == 56, "stable dispatch proof wire");
static_assert(sizeof(ImparoCudaDeviceProfileWire) == 16, "stable device profile wire");
static_assert(sizeof(ImparoCudaHostProfileWire) == 32, "stable Host profile wire");
static_assert(offsetof(ImparoCudaHostProfileWire, struct_bytes) == 0, "profile size offset");
static_assert(offsetof(ImparoCudaHostProfileWire, reserved) == 4, "profile reserved offset");
static_assert(offsetof(ImparoCudaHostProfileWire, available_host_bytes) == 8, "profile RAM offset");
static_assert(offsetof(ImparoCudaHostProfileWire, pinned_h2d_bytes_per_second) == 16, "profile H2D offset");
static_assert(offsetof(ImparoCudaHostProfileWire, pinned_d2h_bytes_per_second) == 24, "profile D2H offset");

#define CUDA_OK(call)                                                              \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "imparo cuda: %s failed: %s (%s:%d)\n", #call,    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);            \
            return 1;                                                              \
        }                                                                          \
    } while (0)

namespace {

// ---- state (the .mm's globals, CUDA edition) --------------------------------------
// Capacity, not a second copy of BufId::COUNT. Rust checks this through
// imparo_cuda_buf_count at init; headroom lets model-private slots grow without
// changing the native ABI on every appended BufId.
constexpr int B_COUNT = 30;
constexpr int MAX_LAYERS = 128;

struct WeightSpanWire {
    uint64_t offset;
    uint64_t bytes;
};

struct ResidentWeightSegment {
    uint64_t file_offset;
    uint64_t bytes;
    uint64_t device_offset;
};

struct PackedQ4Span {
    const uint8_t * base = nullptr;
    uint8_t * packed = nullptr;
    uint64_t bytes = 0;
    uint32_t n_in = 0;
    uint32_t n_out = 0;
};

struct DecodeQ4ShadowSpan {
    uint64_t offset = 0;
    uint8_t * q4 = nullptr;
    uint32_t n_in = 0;
    uint32_t n_out = 0;
    bool tile_major = false;
    bool q5 = false;
};

struct PrefillQ4ShadowSpan {
    uint64_t offset = 0;
    uint8_t * q4 = nullptr;
    uint32_t n_in = 0;
    uint32_t n_out = 0;
};

#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
struct CublasF16WeightSpan {
    const uint8_t * base = nullptr;
    __half * weights = nullptr;
    uint64_t bytes = 0;
    uint32_t n_in = 0;
    uint32_t n_out = 0;
    bool ready = false;
};
#endif

struct DynamicGraphNode {
    cudaGraphNode_t node = nullptr;
    cudaKernelNodeParams params = {};
    std::vector<void *> args;
    // Program-pack launches originate from a synchronous C wire whose argument
    // storage is not allowed to outlive the call. The graph scanner therefore
    // owns a naturally aligned copy of every argument, including capture-static
    // arguments, and patches only adapter-authorized update sources on replay.
    union alignas(8) ProgramArgumentStorage {
        uint64_t u64;
        uint32_t u32;
    };
    std::vector<ProgramArgumentStorage> program_arguments;
    std::vector<uint32_t> program_update_sources;
    uint32_t start_index = UINT32_MAX;
    uint32_t valid_index = UINT32_MAX;
    uint32_t ring = 0;
    uint32_t start_value = 0;
    uint32_t valid_value = 0;
};

struct KvDequantKey {
    uint32_t layer = UINT32_MAX;
    uint32_t width = 0;
    uint32_t slots = 0;
    uint32_t ring = 0;
    uint32_t scratch = UINT32_MAX;
    uint32_t cache_type = 0;

    bool matches(uint32_t wanted_layer, uint32_t wanted_width,
                 uint32_t wanted_slots, uint32_t wanted_ring,
                 uint32_t wanted_scratch, uint32_t wanted_type) const {
        return layer == wanted_layer && width == wanted_width
            && slots == wanted_slots && ring == wanted_ring
            && scratch == wanted_scratch && cache_type == wanted_type;
    }
    void install(uint32_t new_layer, uint32_t new_width, uint32_t new_slots,
                 uint32_t new_ring, uint32_t new_scratch, uint32_t new_type) {
        layer = new_layer;
        width = new_width;
        slots = new_slots;
        ring = new_ring;
        scratch = new_scratch;
        cache_type = new_type;
    }
    void invalidate_layer(uint32_t changed_layer) {
        if (layer == changed_layer) layer = UINT32_MAX;
    }
    void clear() { layer = UINT32_MAX; }
};

struct State {
    const uint8_t * weights_host = nullptr; // live GGUF mmap; always retained for paging
    void * weights = nullptr;        // whole device copy in resident mode
    uint64_t weights_len = 0;
    uint64_t weights_device_bytes = 0;
    bool weights_resident = false;
    std::vector<WeightSpanWire> streamed_weights;
    std::vector<ResidentWeightSegment> resident_segments;
    void * weight_cache = nullptr;   // bounded scratch in paged-weight mode
    uint64_t weight_cache_bytes = 0;
    uint64_t weight_cache_limit = 0;
    void * rope_freqs = nullptr;
    uint64_t rope_freqs_bytes = 0;
    const float * rope_freqs_host = nullptr;
    uint64_t rope_freqs_host_bytes = 0;
    // Host-originated integer control buffers retain a shadow. Paged embedding
    // staging can then select rows without a device-to-host synchronization.
    std::vector<uint32_t> u32_shadow[B_COUNT];
    void * ple_stage_host = nullptr;
    uint64_t ple_stage_host_bytes = 0;
    bool ple_stage_used = false;
    void * decode_row_stage_host = nullptr;
    uint64_t decode_row_stage_host_bytes = 0;
    void * decode_token_stage_host = nullptr;
    void * decode_control_host = nullptr;
    void * decode_control_device = nullptr;
    void * q8_scratch = nullptr;    // transient Q8_1 activations for Q4_0 MMVQ
    uint64_t q8_scratch_bytes = 0;
    const void * q8_l2_window_base = nullptr;
    uint64_t q8_l2_window_bytes = 0;
    uint64_t packed_q4_bytes = 0;
    uint64_t packed_q4_budget = 0;
    uint64_t runtime_reserve_bytes = 0;
    uint64_t ffn_sidecar_model_layer_bytes = 0;
    bool packed_q4_budget_initialized = false;
    bool ffn_sidecar_model_ready = false;
    bool ffn_sidecar_model_pack_failed = false;
    uint32_t ffn_sidecar_model_pack_calls = 0;
    bool packed_q4_force_reclaim_exercised = false;
    bool packed_q4_force_reclaim_active = false;
    std::vector<PackedQ4Span> packed_q4_spans;
    std::vector<const uint8_t *> rejected_packed_q4_spans;
    void * decode_q4_shadow_slab = nullptr;
    uint64_t decode_q4_shadow_bytes = 0;
    bool decode_q4_shadow_ready = false;
    std::vector<DecodeQ4ShadowSpan> decode_q4_shadow_spans;
    std::vector<PrefillQ4ShadowSpan> prefill_q4_shadow_spans;
    // Optional second slot lets a fused producer write the next projection's Q8
    // input while the current projection still consumes the active slot. The
    // slots swap ownership only after a kernel has produced the complete layout.
    void * q8_scratch_next = nullptr;
    uint64_t q8_scratch_next_bytes = 0;
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    cublasHandle_t cublas = nullptr;
    void * cublas_weights_f16 = nullptr;
    uint64_t cublas_weights_f16_bytes = 0;
    void * cublas_activations_f16 = nullptr;
    uint64_t cublas_activations_f16_bytes = 0;
    void * cublas_output_f16 = nullptr;
    uint64_t cublas_output_f16_bytes = 0;
    uint64_t cublas_weight_cache_bytes = 0;
    uint64_t cublas_weight_cache_budget = 0;
    std::vector<CublasF16WeightSpan> cublas_weight_spans;
#endif

    // Host-side versioned ownership for the transient activation quantization.
    // Independent projections may reuse it only while the exact source generation,
    // shape, row range, and physical layout still match.
    uint64_t buf_epoch[B_COUNT] = {};
    uint32_t q8_src = UINT32_MAX;
    uint32_t q8_n_in = 0, q8_n_tok = 0, q8_src_row = 0, q8_layout = UINT32_MAX;
    uint64_t q8_src_epoch = 0;
    uint64_t graph_capture_generation = 0;
    uint64_t q8_owner_capture_generation = 0;
    void * attention_scratch = nullptr; // bounded, reusable per-SM prefill workspace
    uint64_t attention_scratch_bytes = 0;
    void * attention_q_cache = nullptr; // post-head f16 Q, consumed by tiled attention
    uint64_t attention_q_cache_bytes = 0;
    uint32_t attention_q_src = UINT32_MAX;
    uint64_t attention_q_epoch = 0;
    uint32_t attention_q_head_dim = 0, attention_q_heads = 0, attention_q_tokens = 0;
    void * bufs[B_COUNT] = {};
    uint64_t sizes[B_COUNT] = {};
    bool in_arena[B_COUNT] = {};
    void * arena = nullptr;
    uint64_t arena_size = 0;
    void * kv_k[MAX_LAYERS] = {};
    void * kv_v[MAX_LAYERS] = {};
    uint64_t kv_bytes[MAX_LAYERS] = {};
    // A single transactionally replaced device allocation owns every K/V slice.
    // Request-time free/reuse changes only fixed-offset ownership metadata and
    // therefore never calls cudaMalloc/cudaFree.
    void * kv_arena = nullptr;
    imparo_cuda_kv::Layout<MAX_LAYERS> kv_layout;
    // One startup/grow allocation owns every per-layer page table. Request-time
    // placement updates only its contents; it never cudaMalloc/cudaFree.
    void * kv_page_table_arena = nullptr;
    imparo_cuda_kv::PageTables<MAX_LAYERS> kv_page_tables;
    imparo_cuda_host::Registry host_registry;
    std::vector<void *> host_buffers;
    bool host_profile_cached = false;
    ImparoCudaHostProfileWire host_profile = {};
    void * kv_stage_host = nullptr;
    uint64_t kv_stage_slot_bytes = 0;
    uint32_t kv_stage_next = 0;
    cudaEvent_t kv_stage_done[imparo_cuda_kv::kStageSlots] = {};
    cudaStream_t stream = nullptr;
    bool runtime_initialized = false;
    int device = 0;
    int sm_count = 1;
    int sm_version = 0;
    imparo_sm80_q8_replay::PlanLimits q8_mmq_limits = {};
    char device_name[256] = {};
    int pending_error = 0;
    bool forward_active = false;
    uint64_t ffn_sidecar_trace_attempts = 0;
    uint64_t ffn_sidecar_trace_admitted = 0;
    uint64_t ffn_sidecar_trace_input_hits = 0;
    uint64_t ffn_sidecar_trace_committed = 0;
    uint32_t ffn_sidecar_model_calls = 0;
    uint32_t ffn_sidecar_model_commits = 0;
    bool forward_decode = false;
    // `forward_decode` describes only the currently submitted forward and is reset
    // by `imparo_cuda_end`. Keep the prefill -> decode phase transition separate:
    // otherwise every token releases graph scratch and resets the warmup counter.
    bool decode_phase_active = false;
    bool decode_prepared = false;
    bool decode_argmax = false;
    uint32_t decode_token = 0;
    uint32_t decode_start_pos = 0;
    bool batch_geometry_valid = false;
    uint32_t batch_geometry_start = 0;
    uint32_t batch_geometry_tokens = 0;
    uint32_t batch_geometry_phase = 0;
    uint32_t decode_sequence_start_pos = UINT32_MAX;
    uint32_t decode_previous_start_pos = UINT32_MAX;
    uint32_t decode_warm_forwards = 0;
    bool graph_capturing = false;
    bool graph_capture_compatible = false;
    bool decode_graph_shape_blocked = false;
    uint32_t decode_graph_capture_after = 0;
    uint32_t graph_min_start = 0;
    uint32_t graph_max_start = UINT32_MAX;
    uint32_t graph_expected_dynamic_nodes = 0;
    cudaGraph_t decode_graph = nullptr;
    cudaGraphExec_t decode_graph_exec = nullptr;
    std::vector<DynamicGraphNode> decode_graph_nodes;
    // Laboratory-only exact-key prefill replay. Unlike decode graph reuse this
    // never widens a schedule bucket: token count, start position and output
    // route must match exactly, while token payloads are uploaded before replay.
    cudaGraph_t prefill_graph = nullptr;
    cudaGraphExec_t prefill_graph_exec = nullptr;
    uint32_t prefill_graph_tokens = 0;
    uint32_t prefill_graph_start = 0;
    bool prefill_graph_argmax = false;
    uint32_t prefill_warm_forwards = 0;
    // Dev-only same-process numerical mask sweep index. It is read only when
    // the explicit sweep environment is present and never affects production.
    uint32_t prefill_lab_sweep_index = UINT32_MAX;
    bool prefill_prepared = false;
    bool prefill_capture_requested = false;
    bool prefill_capture_active = false;
    bool prefill_graph_blocked = false;
    // True only between begin_forward() and end(). It lets the existing Prefill
    // prepare ABI distinguish the pre-forward admission call from the split body
    // boundary without expanding the dynamic backend ABI.
    bool forward_open = false;
    uint32_t prefill_token_buf = UINT32_MAX;
    uint64_t prefill_token_off = 0;
    std::vector<uint32_t> prefill_tokens;
    // Stable host-source descriptors populated by the normal semantic ops.
    bool decode_row_desc_valid = false;
    uint64_t decode_row_base = 0;
    uint64_t decode_row_bytes = 0;
    bool decode_ple_desc_valid = false;
    uint64_t decode_ple_table = 0;
    uint64_t decode_ple_row_bytes = 0;
    uint64_t decode_ple_norm = 0;
    uint64_t decode_ple_norm_bytes = 0;
    uint64_t decode_ple_prefix = 0;
    uint32_t decode_token_buf = UINT32_MAX;
    uint64_t decode_token_off = 0;
    // Reused diagnostic events: profiling one complete forward must not add two
    // event allocations to every token or enlarge the stable plugin ABI.
    cudaEvent_t forward_start = nullptr;
    cudaEvent_t forward_stop = nullptr;
    bool forward_timing_armed = false;
    bool tuner_mode = false;
    double last_gpu_us = 0.0;
    void * probe_input = nullptr;
    uint64_t probe_input_bytes = 0;
    void * probe_output = nullptr;
    uint64_t probe_output_bytes = 0;
    uint64_t choice_epoch = 1;
    uint64_t proof_dispatches = 0;
    uint64_t proof_observed_choice_epoch = 0;
    uint64_t proof_variant = 0;
    uint64_t proof_expected_knob_mask = 0;
    uint64_t proof_observed_knob_mask = 0;
    uint32_t proof_family = 0;
    // Static tuner-only reachability proof for the coupled exact-128 route. Bits are
    // set only after concrete kernel success/commit points; counters prevent one
    // successful layer from hiding a fallback in another layer.
    uint32_t tune_exact128_route_hits = 0;
    uint32_t tune_exact128_attn_commits = 0;
    uint32_t tune_exact128_ffn_commits = 0;
    uint32_t tune_exact128_ple_commits = 0;
    uint32_t tune_exact128_token64_commits = 0;
    bool nsys_capture_active = false;
    bool tuner_lab = false;
    std::chrono::steady_clock::time_point forward_host_start;
    uint32_t kv_type_k = 1, kv_type_v = 1;   // 1 = f16, 2 = q4_0, 8 = q8_0
    KvDequantKey kdq, vdq;
    uint32_t epilogue = 0;
    // The backend knob table (src/knobs.rs declares names/candidates; the tuner
    // sweeps them; kernels consult the slots they are wired to). UNTUNED kernels
    // above currently hard-code their geometry -- wiring each launch to its slot
    // is part of the CUDA port (marked per knob in knobs.rs).
    uint32_t knobs[64] = {};
};
State g;
std::mutex cuda_runtime_mutex;

template <int NACC>
__global__ void k_probe_accumulators(float * out, uint32_t iters) {
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = float(i + 1) * 0.0001f;
    const float x = float((blockIdx.x * blockDim.x + threadIdx.x) & 255) * 0.00001f + 1.0f;
    for (uint32_t k = 0; k < iters; ++k) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = fmaf(acc[i], x, 0.000001f);
    }
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; ++i) sum += acc[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void k_probe_bandwidth(
        volatile const float * in, float * out, uint64_t words, uint32_t reps) {
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t step = uint64_t(gridDim.x) * blockDim.x;
    float sum = 0.0f;
    for (uint32_t rep = 0; rep < reps; ++rep) {
        for (uint64_t i = tid; i < words; i += step) sum += in[i];
    }
    out[tid] = sum;
}

__global__ void k_probe_scoremix(
        const __half2 * input, float * out, uint32_t iters,
        uint32_t stride, uint32_t kspan) {
    extern __shared__ __half2 tiles[];
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    __half2 * q_tile = tiles + warp * 256;
    __half2 * k_tile = q_tile + 128;
    for (uint32_t i = lane; i < 128; i += 32) q_tile[i] = input[i];
    __syncwarp();
    imparo_sm80_mma::Float16x16 accum{};
    for (uint32_t i = 0; i < iters; ++i) {
        const uint64_t row = uint64_t(i & kspan) * (stride / 2);
        for (uint32_t j = lane; j < 128; j += 32) {
            k_tile[j] = input[row + j];
        }
        __syncwarp();
        imparo_sm80_mma::Half16x8 q_fragment;
        imparo_sm80_mma::Half16x8 k_fragment;
        imparo_sm80_mma::load_half16x8(q_fragment, q_tile, 8, lane);
        imparo_sm80_mma::load_half16x8(k_fragment, k_tile, 8, lane);
        imparo_sm80_mma::mma_qk(accum, q_fragment, k_fragment);
        __syncwarp();
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = accum.x[lane & 7];
}

__global__ void k_probe_noop(float * out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] += 1.0f;
}

bool ensure_probe_storage(uint64_t input_bytes, uint64_t output_bytes) {
    if (!g.stream || input_bytes > SIZE_MAX || output_bytes > SIZE_MAX) return false;
    if (input_bytes > g.probe_input_bytes) {
        if (g.probe_input) cudaFree(g.probe_input);
        g.probe_input = nullptr;
        g.probe_input_bytes = 0;
        if (cudaMalloc(&g.probe_input, size_t(input_bytes)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        g.probe_input_bytes = input_bytes;
        if (cudaMemsetAsync(g.probe_input, 0, size_t(input_bytes), g.stream)
            != cudaSuccess) return false;
    }
    if (output_bytes > g.probe_output_bytes) {
        if (g.probe_output) cudaFree(g.probe_output);
        g.probe_output = nullptr;
        g.probe_output_bytes = 0;
        if (cudaMalloc(&g.probe_output, size_t(output_bytes)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        g.probe_output_bytes = output_bytes;
    }
    return true;
}

bool ensure_forward_events() {
    if (g.forward_start && g.forward_stop) return true;
    if (cudaEventCreate(&g.forward_start) != cudaSuccess
        || cudaEventCreate(&g.forward_stop) != cudaSuccess) {
        if (g.forward_start) cudaEventDestroy(g.forward_start);
        if (g.forward_stop) cudaEventDestroy(g.forward_stop);
        g.forward_start = nullptr;
        g.forward_stop = nullptr;
        cudaGetLastError();
        return false;
    }
    return true;
}

bool probe_begin() {
    return ensure_forward_events()
        && cudaEventRecord(g.forward_start, g.stream) == cudaSuccess;
}

double probe_end_ms() {
    if (cudaEventRecord(g.forward_stop, g.stream) != cudaSuccess
        || cudaEventSynchronize(g.forward_stop) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, g.forward_start, g.forward_stop) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    return double(ms);
}

uint64_t current_knob_digest() {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t knob : g.knobs) {
        h ^= knob;
        h *= 1099511628211ull;
    }
    return h;
}

uint32_t tuner_knob(uint32_t slot) {
    if (slot >= 64) return 0;
    if (g.tuner_mode) g.proof_observed_knob_mask |= uint64_t(1) << slot;
    return g.knobs[slot];
}

void mark_tuner_dispatch(uint32_t family, uint64_t route_bits) {
    if (!g.tuner_mode) return;
    ++g.proof_dispatches;
    g.proof_observed_choice_epoch = g.choice_epoch;
    g.proof_family = family;
    g.proof_variant = current_knob_digest() ^ route_bits;
}

void clear_q8_l2_window();
namespace imparo_sm86_q4_aligned_prepack {
bool launch_pack_q4_equal_size(
    const uint8_t *, uint8_t *, uint32_t, uint32_t, cudaStream_t);
}

void initialize_arch_knob_defaults(int sm_version) {
    // Slots 6--22 retain zero: zero is their architecture/shape-sensitive auto mode.
    // Only versioned route slots receive compiled defaults here.
    g.knobs[23] = 1024; // D512 MMA correctness floor: 4 * 256 schedule keys.
    const uint32_t verified_sm86 = sm_version == 86 ? 1u : 0u;
    g.knobs[24] = verified_sm86; // numeric Stream-K replay
    g.knobs[25] = verified_sm86; // llama-compatible MMQ ownership
    g.knobs[26] = verified_sm86; // virtual 512-token MMQ schedule
    g.knobs[27] = verified_sm86; // canonical full-tile MMQ
    g.knobs[28] = verified_sm86; // fused RMS norm + add
    g.knobs[29] = verified_sm86; // D256 tiled attention
    g.knobs[30] = verified_sm86; // D256 vector decode
    g.knobs[31] = verified_sm86; // D256 virtual Stream-K
    g.knobs[32] = verified_sm86; // D256 fused attention with stable virtual seams
    g.knobs[33] = verified_sm86; // D512 MMA decode
    g.knobs[34] = verified_sm86; // D512 virtual Stream-K request
    g.knobs[35] = 0;             // 0 = no virtual cells; stable whole-after route
    // The D512 MMA, D256 vector and small-query kernels are one coupled numerical
    // route: neither deep specialization passes the recurrent gate in isolation,
    // while the complete SM86 route does. Other architectures fail closed to the
    // staged-half common path until their own fixed-gate receipt exists.
    g.knobs[36] = verified_sm86; // coupled, gate-verified decode specializations
    g.knobs[37] = verified_sm86; // D64/GQA4 MMA prefill
    g.knobs[38] = 0;             // complete FFN sidecar: safe-off until receipted
    g.knobs[39] = 0;             // exact-128 coupled route: safe-off until receipted
    g.knobs[40] = 0;             // exact-128 full-logits Graph: safe-off until receipted
    g.knobs[41] = 0;             // D64/GQA4 direct-Q8 vector Decode: receipt required
    g.knobs[42] = 0;             // aligned whole-K Q8 MMQ: safe-off until tuned
    g.knobs[43] = 0;             // Q8_0_TM SiLU pair: safe-off until receipted
    g.knobs[44] = 0;             // fused SwiGLU D4 sidecar: safe-off until receipted
    g.knobs[45] = 0;             // private SwiGLU-to-Down: safe-off until receipted
    g.knobs[46] = 0;             // D64 shared K/V minimum tokens: zero disables
    g.knobs[47] = 0;             // Q8_0_TM async weight staging: safe-off
    g.knobs[48] = 0;             // Q8_0_TM Gate/Up row-pair: safe-off until receipted
    g.knobs[49] = 0;             // D64/Q8 GQA4 shared-K/V Decode: safe-off
    g.knobs[50] = 0;             // Q8_0_TM Decode Gate/Up/SiLU: receipt required
    g.knobs[51] = 0;             // SM86 single-token shortconv fusion: safe-off
    g.knobs[52] = 0;             // capture-local RMS-produced Q8 reuse: safe-off
    g.knobs[53] = 0;             // Decode Gate/Up Q5 layer map: receipt required
    g.knobs[54] = 0;             // Decode Down Q4 layer map: receipt required
    g.knobs[55] = 0;             // Decode Down Q5 layer map: receipt required
    g.knobs[56] = 0;             // batched projection RMS-to-D4: receipt required
    g.knobs[57] = 0;             // Prefill Down Q4 layer map: receipt required
    g.knobs[58] = 0;             // row-local Prefill tail rows: zero keeps 64
    g.knobs[59] = 0;             // Prefill head-post threads: derived default
}

static bool tuned_ffn_sidecar_requested(uint32_t n_tok) {
    const uint32_t min_tokens = tuner_knob(38);
    return min_tokens != 0 && n_tok >= min_tokens;
}

static bool tuned_exact128_sm86_route(uint32_t n_tok) {
    // Default-off laboratory override for evaluating the complete atomic route
    // before a current-fingerprint correctness receipt can authorize knob 39.
    // Production selection remains exclusively tuner/receipt bound.
    static const bool atomic_lab = std::getenv(
        "IMPARO_CUDA_PREFILL_EXACT128_ATOMIC_LAB") != nullptr;
    return n_tok == 128
        && ((g.knobs[39] == 1 || g.knobs[39] == 2) || atomic_lab);
}

static bool tuned_exact128_token64(uint32_t n_tok) {
    return n_tok == 128 && g.knobs[39] == 2;
}

static bool tuned_exact128_staged_f32_attention(uint32_t n_tok) {
    return n_tok == 128 && g.sm_version == 86
        && (g.knobs[39] == 3 || g.knobs[39] == 4);
}

static bool tuned_exact128_fast_transaction(uint32_t n_tok) {
    return n_tok == 128 && g.sm_version == 86
        && (g.knobs[39] == 4 || g.knobs[39] == 5)
        && g.ffn_sidecar_model_ready;
}
enum : uint32_t {
    EXACT128_PLE_GATE = 1u << 0,
    EXACT128_PLE_DIRECT = 1u << 1,
    EXACT128_FFN_ADMITTED = 1u << 2,
    EXACT128_FFN_DOWN = 1u << 3,
    EXACT128_FFN_COMMITTED = 1u << 4,
    EXACT128_ATTN = 1u << 5,
};
enum : uint32_t {
    EXACT128_COUNT_MASK = 0x3fu,
    EXACT128_ATTN_COUNT_SHIFT = 6u,
    EXACT128_FFN_COUNT_SHIFT = 12u,
    EXACT128_PLE_COUNT_SHIFT = 18u,
    EXACT128_TOKEN64_COUNT_SHIFT = 24u,
};
static void exact128_record_commit(uint32_t &counter) {
    if (counter < EXACT128_COUNT_MASK) ++counter;
}
static uint32_t exact128_packed_route_evidence() {
    return g.tune_exact128_route_hits
        | ((g.tune_exact128_attn_commits & EXACT128_COUNT_MASK)
            << EXACT128_ATTN_COUNT_SHIFT)
        | ((g.tune_exact128_ffn_commits & EXACT128_COUNT_MASK)
            << EXACT128_FFN_COUNT_SHIFT)
        | ((g.tune_exact128_ple_commits & EXACT128_COUNT_MASK)
            << EXACT128_PLE_COUNT_SHIFT)
        | ((g.tune_exact128_token64_commits & EXACT128_COUNT_MASK)
            << EXACT128_TOKEN64_COUNT_SHIFT);
}
enum : int { CUDA_RC_ERROR = 1, CUDA_RC_OOM = 2, CUDA_RC_INVALID = 3 };

int selected_cuda_device(int * device_out) {
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr, "imparo cuda: cudaGetDeviceCount failed: %s\n",
                     cudaGetErrorString(count_error));
        return CUDA_RC_ERROR;
    }
    int device = 0;
    if (const char * selected = std::getenv("IMPARO_CUDA_DEVICE")) {
        char * end = nullptr;
        const long value = std::strtol(selected, &end, 10);
        if (end == selected || *end || value < 0 || value >= device_count) {
            std::fprintf(stderr,
                "imparo cuda: IMPARO_CUDA_DEVICE=%s is invalid; visible devices=%d\n",
                selected, device_count);
            return CUDA_RC_INVALID;
        }
        device = int(value);
    }
    *device_out = device;
    return 0;
}

// Initialize exactly one CUDA Runtime primary context/stream. Program Packs are
// admitted before weights, so both the Program bridge and weight initialization use
// this idempotent path. `cudaSetDevice` is repeated on every call because Rust work may
// resume on another OS thread.
int ensure_cuda_runtime() noexcept try {
    std::lock_guard<std::mutex> lock(cuda_runtime_mutex);
    int selected = 0;
    const int selected_rc = selected_cuda_device(&selected);
    if (selected_rc) return selected_rc;
    if (g.runtime_initialized && selected != g.device) return CUDA_RC_INVALID;
    g.device = selected;
    if (cudaSetDevice(g.device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        return CUDA_RC_ERROR;
    }
    if (g.runtime_initialized) return g.stream ? 0 : CUDA_RC_INVALID;

    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, g.device) != cudaSuccess) return CUDA_RC_ERROR;
    g.sm_count = std::max(1, prop.multiProcessorCount);
    g.sm_version = 10 * prop.major + prop.minor;
    g.q8_mmq_limits.max_grid_x = uint64_t(prop.maxGridSize[0]);
    g.q8_mmq_limits.max_grid_y = uint64_t(prop.maxGridSize[1]);
    g.q8_mmq_limits.max_shared_bytes = uint64_t(
        prop.sharedMemPerBlockOptin > 0
            ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock);
    g.q8_mmq_limits.max_total_work = 0;
    g.q8_mmq_limits.enabled_tile_mask =
        imparo_sm80_q8_replay::all_registered_tile_mask();
    initialize_arch_knob_defaults(g.sm_version);
    std::snprintf(g.device_name, sizeof(g.device_name), "%s", prop.name);
    cudaError_t stream_rc = cudaSuccess;
    if (g.tuner_mode) {
        int least_priority = 0;
        int greatest_priority = 0;
        stream_rc = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (stream_rc == cudaSuccess) {
            stream_rc = cudaStreamCreateWithPriority(
                &g.stream, cudaStreamNonBlocking, least_priority);
        }
    } else {
        stream_rc = cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking);
    }
    if (stream_rc != cudaSuccess || !g.stream) {
        if (g.stream) {
            (void)cudaStreamDestroy(g.stream);
            g.stream = nullptr;
        }
        return CUDA_RC_ERROR;
    }
    g.runtime_initialized = true;
    return 0;
} catch (...) {
    // This helper is reached from legacy C exports as well as guarded Program
    // exports. Mutex/runtime initialization failures must remain status codes.
    return CUDA_RC_ERROR;
}

int hex_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

bool decode_build_sha256(uint8_t (&out)[32]) {
    constexpr const char * encoded = IMPARO_CUDA_BUILD_SHA256;
    for (uint32_t i = 0; i < 32; ++i) {
        const int hi = hex_nibble(encoded[2 * i]);
        const int lo = hex_nibble(encoded[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return true;
}

enum : uint32_t {
    Q8_LAYOUT_GENERIC = 0,
    Q8_LAYOUT_MMQ = 1,
    Q8_LAYOUT_MMVQ = 2,
    Q8_LAYOUT_MMQ_D4 = 3,
    Q8_LAYOUT_MMA_READY = 4,
};

bool destroy_prefill_graph_checked() {
    if (g.prefill_graph_exec) {
        if (cudaGraphExecDestroy(g.prefill_graph_exec) != cudaSuccess) return false;
        g.prefill_graph_exec = nullptr;
    }
    if (g.prefill_graph) {
        if (cudaGraphDestroy(g.prefill_graph) != cudaSuccess) return false;
        g.prefill_graph = nullptr;
    }
    g.prefill_capture_requested = false;
    g.prefill_capture_active = false;
    return true;
}

void destroy_prefill_graph() {
    (void)destroy_prefill_graph_checked();
}

bool destroy_decode_graph_checked() {
    // A graph exec and its source graph may both retain kernel/module references.
    // Preserve a failed handle so Program lifecycle code never unloads a module
    // underneath an object CUDA may still own.
    if (g.decode_graph_exec) {
        if (cudaGraphExecDestroy(g.decode_graph_exec) != cudaSuccess) return false;
        g.decode_graph_exec = nullptr;
    }
    if (g.decode_graph) {
        if (cudaGraphDestroy(g.decode_graph) != cudaSuccess) return false;
        g.decode_graph = nullptr;
    }
    g.decode_graph_nodes.clear();
    return destroy_prefill_graph_checked();
}

void destroy_decode_graph() {
    // Existing void-returning fallback paths cannot surface teardown failure. Keep
    // failed handles live; checked Program mutations use the status helper directly.
    (void)destroy_decode_graph_checked();
}

void invalidate_q8_cache() {
    g.q8_src = UINT32_MAX;
    g.q8_layout = UINT32_MAX;
    g.q8_owner_capture_generation = 0;
}

void mark_buf_written(uint32_t id) {
    if (id >= B_COUNT) return;
    ++g.buf_epoch[id];
    if (!g.buf_epoch[id]) ++g.buf_epoch[id];
}

bool q8_cache_matches(uint32_t src, uint32_t n_in, uint32_t n_tok,
                      uint32_t src_row, uint32_t layout) {
    static const bool disabled = std::getenv("IMPARO_CUDA_NO_Q8_CACHE") != nullptr;
    if (disabled) return false;
    // Capture may omit a quantizer only when the owner is a producer already recorded
    // in this exact Decode capture. An eager owner, a previous capture generation or a
    // Prefill capture can never authorize a hit: their scratch contents would become a
    // stale constant when the graph replays for another token.
    if (g.graph_capturing
            && (g.knobs[52] == 0 || !g.forward_decode
                || g.prefill_capture_active
                || g.graph_capture_generation == 0
                || g.q8_owner_capture_generation
                    != g.graph_capture_generation)) {
        return false;
    }
    return src < B_COUNT && g.q8_src == src && g.q8_src_epoch == g.buf_epoch[src]
        && g.q8_n_in == n_in && g.q8_n_tok == n_tok
        && g.q8_src_row == src_row && g.q8_layout == layout;
}

void own_q8_cache(uint32_t src, uint32_t n_in, uint32_t n_tok,
                  uint32_t src_row, uint32_t layout) {
    g.q8_src = src;
    g.q8_src_epoch = g.buf_epoch[src];
    g.q8_n_in = n_in;
    g.q8_n_tok = n_tok;
    g.q8_src_row = src_row;
    g.q8_layout = layout;
    g.q8_owner_capture_generation =
        g.graph_capturing && g.forward_decode && !g.prefill_capture_active
        ? g.graph_capture_generation
        : 0;
}

void trace_q8_cache(bool hit, uint32_t src, uint32_t n_in, uint32_t n_tok,
                    uint32_t src_row, uint32_t layout) {
    if (!std::getenv("IMPARO_CUDA_PROFILE_Q8")) return;
    std::fprintf(stderr,
        "[cuda-q8-cache] %s src=%u epoch=%llu in=%u tokens=%u row=%u layout=%u"
        " owner=%u owner_epoch=%llu owner_in=%u owner_tokens=%u owner_row=%u"
        " owner_layout=%u\n",
        hit ? "hit" : "miss", src,
        static_cast<unsigned long long>(g.buf_epoch[src]),
        n_in, n_tok, src_row, layout, g.q8_src,
        static_cast<unsigned long long>(g.q8_src_epoch),
        g.q8_n_in, g.q8_n_tok, g.q8_src_row, g.q8_layout);
}

uint64_t env_mib(const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr;
    const unsigned long long mib = std::strtoull(value, &end, 10);
    return end != value ? uint64_t(mib) << 20 : fallback;
}

uint64_t attention_workspace_budget(uint32_t head_dim) {
    uint64_t mib = tuner_knob(head_dim == 256 ? 13 : 14);
    if (!mib) {
        mib = g.sm_version == 86 ? (head_dim == 256 ? 20u : 16u) : 32u;
    }
    uint64_t budget = env_mib("IMPARO_CUDA_ATTN_WORKSPACE_MIB", mib << 20);
    const char * shape_budget = head_dim == 64
        ? "IMPARO_CUDA_ATTN_D64_WORKSPACE_MIB"
        : (head_dim == 256
            ? "IMPARO_CUDA_ATTN_D256_WORKSPACE_MIB"
            : "IMPARO_CUDA_ATTN_D512_WORKSPACE_MIB");
    return env_mib(shape_budget, budget);
}

uint64_t tracked_bytes() {
    uint64_t total = g.weights_resident ? g.weights_device_bytes : 0;
    total += g.weight_cache_bytes + g.rope_freqs_bytes + g.q8_scratch_bytes
        + g.q8_scratch_next_bytes + g.packed_q4_bytes
        + g.attention_scratch_bytes + g.attention_q_cache_bytes + g.arena_size;
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    total += g.cublas_weights_f16_bytes + g.cublas_activations_f16_bytes
        + g.cublas_output_f16_bytes + g.cublas_weight_cache_bytes;
#endif
    if (g.decode_control_device) total += sizeof(uint32_t);
    for (int i = 0; i < B_COUNT; ++i) {
        if (g.bufs[i] && !g.in_arena[i]) total += g.sizes[i];
    }
    total += g.kv_layout.arena_bytes;
    total += g.kv_page_tables.arena_entries * sizeof(uint32_t);
    return total;
}

void set_pending(int rc, const char * what) {
    if (!g.pending_error) g.pending_error = rc;
    std::fprintf(stderr, "imparo cuda: %s failed rc=%d\n", what, rc);
}

void release_prefill_transients_for_decode() {
    static const bool keep_transients =
        std::getenv("IMPARO_CUDA_KEEP_PREFILL_TRANSIENTS") != nullptr;
    // The half-Q cache is produced only by wide prefill. Multi-token target
    // verification also enters the common non-single-token path, but it does not
    // create this cache and must not trigger another release/Graph recapture.
    if (!g.attention_q_cache) return;
    static const bool profile =
        std::getenv("IMPARO_CUDA_PROFILE_MEMORY") != nullptr;
    if (profile) {
        uint64_t buffers = 0;
        uint64_t kv = 0;
        for (int i = 0; i < B_COUNT; ++i) {
            if (g.bufs[i] && !g.in_arena[i]) buffers += g.sizes[i];
        }
        for (int i = 0; i < MAX_LAYERS; ++i) {
            if (g.kv_k[i]) kv += g.kv_bytes[i];
            if (g.kv_v[i]) kv += g.kv_bytes[i];
        }
        std::fprintf(stderr,
            "[cuda-memory] prefill-to-decode total=%.1fMiB q-cache=%.1fMiB "
            "attention=%.1fMiB q8=%.1f/%.1fMiB arena=%.1fMiB "
            "weights=%.1fMiB page=%.1fMiB buffers=%.1fMiB kv=%.1fMiB\n",
            double(tracked_bytes()) / double(1ull << 20),
            double(g.attention_q_cache_bytes) / double(1ull << 20),
            double(g.attention_scratch_bytes) / double(1ull << 20),
            double(g.q8_scratch_bytes) / double(1ull << 20),
            double(g.q8_scratch_next_bytes) / double(1ull << 20),
            double(g.arena_size) / double(1ull << 20),
            double(g.weights_device_bytes) / double(1ull << 20),
            double(g.weight_cache_bytes) / double(1ull << 20),
            double(buffers) / double(1ull << 20),
            double(kv) / double(1ull << 20));
    }
    if (keep_transients) return;
    // Prefill can retain full-tile attention and MMQ scratch. Small-query decode
    // needs only a tiny shape-derived subset, so keeping the high-water allocations
    // inflates resting VRAM. A cached graph may reference any scratch allocation;
    // discard it first and recapture after the usual warmup.
    destroy_decode_graph();
    clear_q8_l2_window();
    const auto release = [](void *& ptr, uint64_t & bytes) {
        if (!ptr) return true;
        if (cudaFree(ptr) != cudaSuccess) return false;
        ptr = nullptr;
        bytes = 0;
        return true;
    };
    bool released = release(g.attention_q_cache, g.attention_q_cache_bytes);
    released = release(g.attention_scratch, g.attention_scratch_bytes) && released;
    released = release(g.q8_scratch, g.q8_scratch_bytes) && released;
    released = release(g.q8_scratch_next, g.q8_scratch_next_bytes) && released;
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    released = release(
        g.cublas_weights_f16, g.cublas_weights_f16_bytes) && released;
    released = release(
        g.cublas_activations_f16, g.cublas_activations_f16_bytes) && released;
    released = release(
        g.cublas_output_f16, g.cublas_output_f16_bytes) && released;
#endif

    if (!released) {
        set_pending(CUDA_RC_ERROR, "prefill transient release");
    }
    // The next decode must allocate its much smaller scratch before capture.
    // Retaining a prior request's warm count would start capture immediately,
    // where CUDA allocation is intentionally rejected.
    g.decode_warm_forwards = 0;
    g.attention_q_src = UINT32_MAX;
    g.attention_q_epoch = 0;
    g.attention_q_head_dim = 0;
    g.attention_q_heads = 0;
    g.attention_q_tokens = 0;
    invalidate_q8_cache();
}

// Diagnostic-only event scope. It is deliberately local to the native CUDA layer so
// profiling does not enlarge the stable backend ABI. With the environment variable
// absent the compiler-visible work is one predictable branch and no CUDA object exists.
struct MatmatEventScope {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    uint32_t wkind, n_in, n_out, n_tok, epilogue;

    MatmatEventScope(uint32_t wk, uint32_t ni, uint32_t no, uint32_t nt, uint32_t ep)
        : wkind(wk), n_in(ni), n_out(no), n_tok(nt), epilogue(ep) {
        static const bool enabled =
            std::getenv("IMPARO_CUDA_PROFILE_MATMUL") != nullptr;
        if (g.graph_capturing || !enabled
                || cudaEventCreate(&start) != cudaSuccess) return;
        if (cudaEventCreate(&stop) != cudaSuccess) {
            cudaEventDestroy(start);
            start = nullptr;
            return;
        }
        cudaEventRecord(start, g.stream);
    }

    ~MatmatEventScope() {
        if (!start) return;
        float elapsed_ms = 0.0f;
        cudaEventRecord(stop, g.stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::fprintf(stderr,
            "[cuda-matmat-prof] kind=%u in=%u out=%u tokens=%u epilogue=%u ms=%.6f\n",
            wkind, n_in, n_out, n_tok, epilogue, elapsed_ms);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
};

struct OpEventScope {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    const char * name;
    uint64_t a;
    uint64_t b;

    OpEventScope(const char * op, uint64_t first, uint64_t second)
        : name(op), a(first), b(second) {
        static const bool enabled =
            std::getenv("IMPARO_CUDA_PROFILE_OPS") != nullptr;
        if (g.graph_capturing || !enabled
                || cudaEventCreate(&start) != cudaSuccess) return;
        if (cudaEventCreate(&stop) != cudaSuccess) {
            cudaEventDestroy(start);
            start = nullptr;
            return;
        }
        cudaEventRecord(start, g.stream);
    }

    ~OpEventScope() {
        if (!start) return;
        float elapsed_ms = 0.0f;
        cudaEventRecord(stop, g.stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::fprintf(stderr,
            "[cuda-op-prof] op=%s a=%llu b=%llu ms=%.6f\n", name,
            static_cast<unsigned long long>(a),
            static_cast<unsigned long long>(b), elapsed_ms);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
};

// More granular attention timings are opt-in because each scope synchronizes the
// stream. They expose architecture-kernel phase costs without changing the stable
// backend ABI or burdening ordinary end-to-end profiling with thousands of events.
struct AttentionStageEventScope {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    const char * name;
    uint32_t head_dim;
    uint32_t blocks;

    AttentionStageEventScope(const char * stage, uint32_t dim, uint32_t count)
        : name(stage), head_dim(dim), blocks(count) {
        static const bool enabled =
            std::getenv("IMPARO_CUDA_PROFILE_ATTN_STAGES") != nullptr;
        if (g.graph_capturing || !enabled
                || cudaEventCreate(&start) != cudaSuccess) return;
        if (cudaEventCreate(&stop) != cudaSuccess) {
            cudaEventDestroy(start);
            start = nullptr;
            return;
        }
        cudaEventRecord(start, g.stream);
    }

    ~AttentionStageEventScope() {
        if (!start) return;
        float elapsed_ms = 0.0f;
        cudaEventRecord(stop, g.stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::fprintf(stderr,
            "[cuda-attn-prof] stage=%s dim=%u blocks=%u ms=%.6f\n",
            name, head_dim, blocks, elapsed_ms);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }
};

// Route-only diagnostics: never participate in policy selection or alter the
// numerical path. Keeping the trace here makes D256 route evidence uniform
// across fused, staged, batch32 and conservative fallback implementations.
static inline void trace_d256_attention_route(
        const char * route, uint32_t layer, uint32_t start_pos,
        uint32_t n_tok, uint32_t logical_blocks, uint32_t physical_blocks,
        uint32_t segment_bound) {
    if (std::getenv("IMPARO_CUDA_ATTN_D256_TRACE") == nullptr) return;
    std::fprintf(stderr,
        "[d256-attn] route=%s layer=%u start=%u n_tok=%u logical=%u "
        "physical=%u segment_bound=%u\n",
        route, layer, start_pos, n_tok, logical_blocks, physical_blocks,
        segment_bound);
}

int alloc_raw(void ** out, uint64_t bytes, const char * what) {
    if (!bytes) { *out = nullptr; return 0; }
    cudaError_t err = cudaMalloc(out, size_t(bytes));
    if (err == cudaSuccess) return 0;
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    std::fprintf(stderr,
        "imparo cuda: %s needs %.1f MiB, free %.1f / %.1f MiB: %s\n",
        what, double(bytes) / double(1ull << 20), double(free) / double(1ull << 20),
        double(total) / double(1ull << 20), cudaGetErrorString(err));
    *out = nullptr;
    return err == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR;
}

int ensure_weight_cache(uint64_t bytes) {
    if (bytes <= g.weight_cache_bytes) return 0;
    if (!g.weights_host || bytes > g.weight_cache_limit) return CUDA_RC_INVALID;
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    if (g.weight_cache) {
        cudaFree(g.weight_cache);
        g.weight_cache = nullptr;
        g.weight_cache_bytes = 0;
    }
    // Grow geometrically but never beyond the explicit fit limit.
    uint64_t want = std::min(g.weight_cache_limit,
                             std::max(bytes, std::max<uint64_t>(64ull << 20,
                                                                g.weight_cache_bytes * 2)));
    int rc = alloc_raw(&g.weight_cache, want, "paged weight cache");
    if (rc) return rc;
    g.weight_cache_bytes = want;
    return 0;
}

void clear_q8_l2_window() {
    if (!g.q8_l2_window_base) return;
    cudaStreamAttrValue reset{};
    reset.accessPolicyWindow.base_ptr = nullptr;
    reset.accessPolicyWindow.num_bytes = 0;
    reset.accessPolicyWindow.hitRatio = 0.0f;
    reset.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    reset.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    if (cudaStreamSetAttribute(g.stream,
            cudaStreamAttributeAccessPolicyWindow, &reset) != cudaSuccess) {
        cudaGetLastError();
    }
    g.q8_l2_window_base = nullptr;
    g.q8_l2_window_bytes = 0;
}

void configure_q8_l2_window_lab() {
    if (std::getenv("IMPARO_CUDA_Q8_L2_PERSIST_LAB") == nullptr
        || !g.q8_scratch || !g.q8_scratch_bytes
        || (g.q8_l2_window_base == g.q8_scratch
            && g.q8_l2_window_bytes == g.q8_scratch_bytes)) {
        return;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, g.device) != cudaSuccess
        || !prop.persistingL2CacheMaxSize) {
        cudaGetLastError();
        return;
    }
    const uint64_t window_bytes = std::min<uint64_t>(
        g.q8_scratch_bytes, uint64_t(prop.persistingL2CacheMaxSize));
    if (!window_bytes
        || cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
            size_t(window_bytes)) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = g.q8_scratch;
    attr.accessPolicyWindow.num_bytes = size_t(window_bytes);
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    if (cudaStreamSetAttribute(g.stream,
            cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    g.q8_l2_window_base = g.q8_scratch;
    g.q8_l2_window_bytes = g.q8_scratch_bytes;
}

bool release_packed_q4_for_priority_allocation();

int ensure_q8_scratch(uint64_t bytes) {
    if (bytes <= g.q8_scratch_bytes) {
        configure_q8_l2_window_lab();
        return 0;
    }
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    void * next = nullptr;
    int rc = alloc_raw(&next, bytes, "Q8_1 projection scratch");
    if (rc == CUDA_RC_OOM && g.forward_decode) {
        cudaGetLastError();
        if (release_packed_q4_for_priority_allocation()) {
            rc = alloc_raw(&next, bytes, "Q8_1 projection scratch");
        }
    }
    if (rc) return rc;
    void * previous = g.q8_scratch;
    if (previous) clear_q8_l2_window();
    g.q8_scratch = next;
    g.q8_scratch_bytes = bytes;
    invalidate_q8_cache();
    if (previous) cudaFree(previous);
    configure_q8_l2_window_lab();
    return 0;
}

const PackedQ4Span * find_aligned_packed_q4(
        const uint8_t * base, uint32_t n_in, uint32_t n_out) {
    for (const PackedQ4Span & span : g.packed_q4_spans) {
        if (span.base == base && span.n_in == n_in && span.n_out == n_out) {
            return &span;
        }
    }
    return nullptr;
}

const DecodeQ4ShadowSpan * find_decode_q4_shadow(
        uint64_t offset, uint32_t n_in, uint32_t n_out) {
    if (!g.decode_q4_shadow_ready) return nullptr;
    for (const DecodeQ4ShadowSpan & span : g.decode_q4_shadow_spans) {
        if (span.offset == offset && span.n_in == n_in
                && span.n_out == n_out) {
            return &span;
        }
    }
    return nullptr;
}

const PrefillQ4ShadowSpan * find_prefill_q4_shadow(
        uint64_t offset, uint32_t n_in, uint32_t n_out) {
    if (!g.decode_q4_shadow_ready) return nullptr;
    for (const PrefillQ4ShadowSpan & span : g.prefill_q4_shadow_spans) {
        if (span.offset == offset && span.n_in == n_in
                && span.n_out == n_out) {
            return &span;
        }
    }
    return nullptr;
}

bool initialize_packed_q4_budget() {
    if (g.packed_q4_budget_initialized) return true;
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    // This cache is admitted only after the activation arena and current KV fit
    // are already resident.  Do not subtract the startup residency reserve a
    // second time: that made a valid fit depend on a device-specific reserve
    // override.  The model transaction below first builds the complete hot set
    // without publishing output, so an allocation race or incomplete fit remains
    // a conservative fallback rather than a half-committed forward.
    const uint64_t automatic = uint64_t(free);
    g.packed_q4_budget = std::min<uint64_t>(
        automatic, env_mib("IMPARO_CUDA_Q4_PACKED_SIDECAR_MIB", automatic));
    g.packed_q4_budget_initialized = true;
    if (std::getenv("IMPARO_CUDA_Q4_PACKED_SIDECAR_TRACE")) {
        std::fprintf(stderr,
            "[cuda-q4-sidecar] budget=%llu free=%zu total=%zu "
            "admission=complete-model-warmup\n",
            static_cast<unsigned long long>(g.packed_q4_budget), free, total);
    }
    return true;
}

const PackedQ4Span * ensure_aligned_packed_q4_lab(
        const uint8_t * base, uint32_t n_in, uint32_t n_out) {
    const bool trace = std::getenv(
        "IMPARO_CUDA_Q4_PACKED_SIDECAR_TRACE") != nullptr;
    if (!base || !n_in || !n_out || n_in % 128 || n_out % 128) return nullptr;
    if (const PackedQ4Span * found =
            find_aligned_packed_q4(base, n_in, n_out)) return found;
    if (std::find(g.rejected_packed_q4_spans.begin(),
            g.rejected_packed_q4_spans.end(), base)
            != g.rejected_packed_q4_spans.end()) return nullptr;
    const uint64_t records = uint64_t(n_out) * (n_in / 32);
    const uint64_t bytes = records * 18;
    if (!initialize_packed_q4_budget()) {
        g.rejected_packed_q4_spans.push_back(base);
        return nullptr;
    }
    if (bytes > g.packed_q4_budget -
            std::min(g.packed_q4_budget, g.packed_q4_bytes)) {
        if (trace) {
            std::fprintf(stderr,
                "[cuda-q4-sidecar] reject=budget base=%p in=%u out=%u "
                "bytes=%llu used=%llu budget=%llu\n",
                static_cast<const void *>(base), n_in, n_out,
                static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(g.packed_q4_bytes),
                static_cast<unsigned long long>(g.packed_q4_budget));
        }
        g.rejected_packed_q4_spans.push_back(base);
        return nullptr;
    }
    void * packed = nullptr;
    if (alloc_raw(&packed, bytes, "aligned Q4 sidecar")) {
        cudaGetLastError();
        g.rejected_packed_q4_spans.push_back(base);
        return nullptr;
    }
    if (!imparo_sm86_q4_aligned_prepack::launch_pack_q4_equal_size(
            base, static_cast<uint8_t *>(packed), n_in, n_out, g.stream)) {
        cudaGetLastError();
        cudaFree(packed);
        g.rejected_packed_q4_spans.push_back(base);
        return nullptr;
    }
    g.packed_q4_spans.push_back(
        {base, static_cast<uint8_t *>(packed), bytes, n_in, n_out});
    g.packed_q4_bytes += bytes;
    if (trace) {
        std::fprintf(stderr,
            "[cuda-q4-sidecar] accept base=%p in=%u out=%u bytes=%llu "
            "used=%llu budget=%llu spans=%zu\n",
            static_cast<const void *>(base), n_in, n_out,
            static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(g.packed_q4_bytes),
            static_cast<unsigned long long>(g.packed_q4_budget),
            g.packed_q4_spans.size());
    }
    return &g.packed_q4_spans.back();
}

// Packed-Q4 program data is an optional Prefill acceleration cache. A later
// activation resize, KV grow, or Decode-control allocation has higher priority:
// release the cache, invalidate its model admission, and let the next eligible
// Prefill rebuild it through the complete-model warmup transaction. Callers use
// this only at allocation boundaries where no packed pointer is being consumed.
bool release_packed_q4_for_priority_allocation() {
    if ((g.packed_q4_spans.empty() && !g.decode_q4_shadow_slab)
            || !g.packed_q4_bytes
            || g.graph_capturing || g.prefill_capture_active) return false;
    destroy_decode_graph();
    destroy_prefill_graph();
    if (g.stream && cudaStreamSynchronize(g.stream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    uint64_t released_bytes = 0;
    std::vector<PackedQ4Span> retained;
    retained.reserve(g.packed_q4_spans.size());
    for (const PackedQ4Span & span : g.packed_q4_spans) {
        if (span.packed && cudaFree(span.packed) == cudaSuccess) {
            released_bytes += span.bytes;
        } else {
            cudaGetLastError();
            retained.push_back(span);
        }
    }
    if (g.decode_q4_shadow_slab
            && cudaFree(g.decode_q4_shadow_slab) == cudaSuccess) {
        released_bytes += g.decode_q4_shadow_bytes;
        g.decode_q4_shadow_slab = nullptr;
        g.decode_q4_shadow_bytes = 0;
        g.decode_q4_shadow_ready = false;
        g.decode_q4_shadow_spans.clear();
        g.prefill_q4_shadow_spans.clear();
    } else if (g.decode_q4_shadow_slab) {
        cudaGetLastError();
    }
    if (!released_bytes) return false;
    g.packed_q4_spans = std::move(retained);
    g.packed_q4_bytes = 0;
    for (const PackedQ4Span & span : g.packed_q4_spans) {
        g.packed_q4_bytes += span.bytes;
    }
    g.packed_q4_bytes += g.decode_q4_shadow_bytes;
    g.rejected_packed_q4_spans.clear();
    g.packed_q4_budget = 0;
    g.packed_q4_budget_initialized = false;
    g.ffn_sidecar_model_layer_bytes = 0;
    g.ffn_sidecar_model_ready = false;
    g.ffn_sidecar_model_pack_failed = false;
    g.ffn_sidecar_model_pack_calls = 0;
    if (std::getenv("IMPARO_CUDA_Q4_PACKED_SIDECAR_TRACE")) {
        std::fprintf(stderr,
            "[cuda-q4-sidecar] reclaim=%llu retained=%llu "
            "reason=priority-allocation\n",
            static_cast<unsigned long long>(released_bytes),
            static_cast<unsigned long long>(g.packed_q4_bytes));
    }
    return true;
}

int alloc_raw_with_packed_q4_reclaim(
        void ** out, uint64_t bytes, const char * what) {
    const bool forced = g.packed_q4_force_reclaim_active
        && g.packed_q4_bytes;
    int rc = forced ? CUDA_RC_OOM : alloc_raw(out, bytes, what);
    if (forced) *out = nullptr;
    if (rc != CUDA_RC_OOM) return rc;
    cudaGetLastError();
    if (!release_packed_q4_for_priority_allocation()) return rc;
    return alloc_raw(out, bytes, what);
}

bool ensure_q8_scratch_next(uint64_t bytes) {
    if (bytes <= g.q8_scratch_next_bytes) return true;
    if (g.graph_capturing) return false;
    destroy_decode_graph();
    if (g.q8_scratch_next) {
        cudaFree(g.q8_scratch_next);
        g.q8_scratch_next = nullptr;
        g.q8_scratch_next_bytes = 0;
    }
    if (alloc_raw(&g.q8_scratch_next, bytes,
                  "Q8_1 fused-output scratch")) {
        // This cache is an optimization, never a fit requirement. Clear a sticky
        // allocation error and let the standalone quantizer remain authoritative.
        cudaGetLastError();
        return false;
    }
    g.q8_scratch_next_bytes = bytes;
    return true;
}

#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
int ensure_cublas_provider_scratch(
        uint64_t weights_bytes, uint64_t activations_bytes,
        uint64_t output_bytes) {
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    if (!g.cublas && cublasCreate(&g.cublas) != CUBLAS_STATUS_SUCCESS) {
        g.cublas = nullptr;
        return CUDA_RC_ERROR;
    }
    const auto grow = [](void *& pointer, uint64_t & capacity,
                         uint64_t bytes, const char * what) {
        if (bytes <= capacity) return 0;
        void * next = nullptr;
        const int rc = alloc_raw(&next, bytes, what);
        if (rc) return rc;
        void * previous = pointer;
        pointer = next;
        capacity = bytes;
        if (previous) cudaFree(previous);
        return 0;
    };
    int rc = grow(
        g.cublas_weights_f16, g.cublas_weights_f16_bytes,
        weights_bytes, "cuBLAS Q4->F16 weight scratch");
    if (rc) return rc;
    rc = grow(
        g.cublas_activations_f16, g.cublas_activations_f16_bytes,
        activations_bytes, "cuBLAS F16 activation scratch");
    if (rc) return rc;
    return grow(
        g.cublas_output_f16, g.cublas_output_f16_bytes,
        output_bytes, "cuBLAS F16 output scratch");
}

CublasF16WeightSpan * find_cublas_f16_weight(
        const uint8_t * base, uint32_t n_in, uint32_t n_out) {
    for (CublasF16WeightSpan & span : g.cublas_weight_spans) {
        if (span.base == base && span.n_in == n_in
            && span.n_out == n_out) return &span;
    }
    return nullptr;
}

CublasF16WeightSpan * ensure_cublas_f16_weight_lab(
        const uint8_t * base, uint32_t n_in, uint32_t n_out,
        uint64_t bytes) {
    if (CublasF16WeightSpan * found =
            find_cublas_f16_weight(base, n_in, n_out)) return found;
    if (!g.cublas_weight_cache_budget) {
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        const uint64_t guard = std::max<uint64_t>(
            768ull << 20, uint64_t(total) / 6);
        const uint64_t automatic = uint64_t(free) > guard
            ? std::min<uint64_t>(
                2ull << 30, uint64_t(free) - guard)
            : 0;
        g.cublas_weight_cache_budget = env_mib(
            "IMPARO_CUDA_CUBLAS_WEIGHT_CACHE_MIB", automatic);
    }
    if (!bytes || bytes > g.cublas_weight_cache_budget
            - std::min(g.cublas_weight_cache_budget,
                       g.cublas_weight_cache_bytes)) {
        return nullptr;
    }
    void * allocation = nullptr;
    if (alloc_raw(&allocation, bytes, "cuBLAS cached F16 weight")) {
        cudaGetLastError();
        return nullptr;
    }
    g.cublas_weight_spans.push_back({
        base, static_cast<__half *>(allocation), bytes,
        n_in, n_out, false});
    g.cublas_weight_cache_bytes += bytes;
    return &g.cublas_weight_spans.back();
}
#endif

void publish_q8_scratch_next() {
    std::swap(g.q8_scratch, g.q8_scratch_next);
    std::swap(g.q8_scratch_bytes, g.q8_scratch_next_bytes);
}

uint32_t narrow_gemv_max_width() {
    uint32_t narrow_max = tuner_knob(20);
    if (!narrow_max) {
        static const uint32_t forced_narrow_max = [] {
            const char * value = std::getenv(
                "IMPARO_CUDA_NARROW_GEMV_MAX_WIDTH");
            return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        narrow_max = forced_narrow_max;
    }
    if (!narrow_max) narrow_max = 512;
    return narrow_max;
}

uint32_t decode_warps(uint32_t n_in) {
    const bool narrow = n_in <= narrow_gemv_max_width();
    uint32_t requested = narrow ? tuner_knob(19) : tuner_knob(6);
    if (!requested) {
        static const uint32_t forced_narrow = [] {
            const char * value = std::getenv(
                "IMPARO_CUDA_NARROW_GEMV_WARPS");
            return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        static const uint32_t forced_general = [] {
            const char * value = std::getenv("IMPARO_CUDA_GEMV_WARPS");
            return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        requested = narrow ? forced_narrow : forced_general;
    }
    if (requested == 2 || requested == 4 || requested == 8) return requested;
    return narrow ? 2u : 4u;
}

uint32_t decode_rows_per_cta(uint32_t n_in) {
    if (n_in > narrow_gemv_max_width()) return 1;
    uint32_t requested = tuner_knob(21);
    if (!requested) {
        static const uint32_t forced = [] {
            const char * value = std::getenv("IMPARO_CUDA_NARROW_GEMV_ROWS");
            return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        requested = forced;
    }
    if (requested == 1 || requested == 4) return requested;
    return 2u;
}

uint32_t batch_mmvq_rows_per_cta() {
    uint32_t requested = tuner_knob(22);
    if (!requested) {
        static const uint32_t forced = [] {
            const char * value = std::getenv("IMPARO_CUDA_BATCH_MMVQ_ROWS");
            return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        requested = forced;
    }
    if (requested == 2 || requested == 4) return requested;
    // One row keeps enough CTAs resident when each thread already owns 2--8
    // independent token accumulators. The registry can retune this per device.
    return 1u;
}

// Attention kernels may need a bounded staging area, but a workspace allocation must
// never make an otherwise-valid fit fail. Callers progressively reduce their tile chunk
// and fall back to the common kernel if even one tile cannot be staged.
bool ensure_attention_scratch(uint64_t bytes) {
    if (bytes <= g.attention_scratch_bytes) return true;
    if (g.graph_capturing) {
        g.graph_capture_compatible = false;
        return false;
    }
    destroy_decode_graph();
    void * next = nullptr;
    cudaError_t allocation = cudaMalloc(&next, bytes);
    if (allocation == cudaErrorMemoryAllocation && g.forward_decode) {
        cudaGetLastError();
        if (release_packed_q4_for_priority_allocation()) {
            allocation = cudaMalloc(&next, bytes);
        }
    }
    if (allocation != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (g.attention_scratch) cudaFree(g.attention_scratch);
    g.attention_scratch = next;
    g.attention_scratch_bytes = bytes;
    return true;
}

bool ensure_attention_q_cache(uint64_t bytes) {
    if (bytes <= g.attention_q_cache_bytes) return true;
    if (g.graph_capturing) return false;
    void * next = nullptr;
    if (cudaMalloc(&next, bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (g.attention_q_cache) cudaFree(g.attention_q_cache);
    g.attention_q_cache = next;
    g.attention_q_cache_bytes = bytes;
    g.attention_q_src = UINT32_MAX;
    return true;
}

bool async_host_control_enabled() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_NO_ASYNC_HOST_CONTROL") == nullptr;
    return enabled;
}

bool decode_pinned_input_enabled() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_NO_DECODE_PINNED_INPUT") == nullptr;
    return enabled;
}

bool decode_graph_enabled() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_NO_DECODE_GRAPH") == nullptr;
    return enabled;
}

bool prefill_graph_lab_override() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_PREFILL_GRAPH_LAB") != nullptr
        || (std::getenv("IMPARO_CUDA_PREFILL_SIDECAR_GRAPH_LAB")
            && std::strcmp(
                std::getenv("IMPARO_CUDA_PREFILL_SIDECAR_GRAPH_LAB"), "1") == 0);
    return enabled;
}

bool prefill_sidecar_graph_lab_override(uint32_t token_count) {
    const char * value = std::getenv(
        "IMPARO_CUDA_PREFILL_SIDECAR_GRAPH_LAB");
    return token_count == 128 && value && std::strcmp(value, "1") == 0;
}

bool prefill_graph_enabled_for(uint32_t token_count) {
    return prefill_graph_lab_override()
        || (g.knobs[40] == 1
            && (tuned_exact128_sm86_route(token_count)
                || tuned_exact128_fast_transaction(token_count)));
}

bool prefill_graph_candidate(
        uint32_t token_count, uint32_t start_pos, bool /*argmax*/) {
    // Full-logits Prefill is the real server/CLI path. Requiring device argmax made
    // this route unreachable because multi-token forward_into() always requests
    // logits. Production is exact-128 only; the lab override remains explicit.
    return prefill_graph_enabled_for(token_count) && token_count > 1
        && g.weights_resident && g.kv_type_k == 2 && g.kv_type_v == 2
        && g.sm_version == 86 && g.batch_geometry_valid
        && g.batch_geometry_tokens == token_count
        && g.batch_geometry_start == start_pos
        && g.batch_geometry_phase == 0;
}

bool prefill_exact128_sidecar_capture_ready(uint32_t n_tok) {
    const bool admitted_route = tuned_exact128_sm86_route(n_tok)
        || tuned_exact128_fast_transaction(n_tok)
        || prefill_sidecar_graph_lab_override(n_tok);
    return g.graph_capturing && g.prefill_capture_active && g.prefill_prepared
        && prefill_graph_enabled_for(n_tok) && admitted_route
        && g.ffn_sidecar_model_ready && g.prefill_warm_forwards >= 2;
}

bool trace_cuda_graph() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_TRACE_GRAPH") != nullptr;
    return enabled;
}

bool decode_graph_candidate() {
    const bool graph_kv = (g.kv_type_k == 2 && g.kv_type_v == 2)
        || (g.kv_type_k == 8 && g.kv_type_v == 8
            && (g.knobs[41] != 0
                || std::getenv("IMPARO_CUDA_ATTN_D64_VEC_Q8") != nullptr)
            && std::getenv("IMPARO_CUDA_NO_ATTN_D64_VEC_Q8") == nullptr);
    return !g.tuner_mode && g.decode_prepared && g.decode_argmax && decode_graph_enabled()
        && decode_pinned_input_enabled() && async_host_control_enabled()
        && g.weights_resident && graph_kv
        && g.sm_version >= 80;
}

bool decode_device_control_enabled() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_GRAPH_NODE_UPDATES") == nullptr;
    return enabled;
}

bool decode_ring_bucket_enabled() {
    static const bool enabled =
        std::getenv("IMPARO_CUDA_NO_RING_GRAPH_BUCKETS") == nullptr;
    return enabled;
}

bool decode_device_control_active() {
    return decode_graph_candidate() && decode_device_control_enabled()
        && g.decode_warm_forwards > 0 && !g.decode_graph_shape_blocked
        && g.decode_start_pos >= g.decode_graph_capture_after;
}

const uint32_t * decode_control_arg() {
    return decode_device_control_active()
        ? static_cast<const uint32_t *>(g.decode_control_device) : nullptr;
}

int rope_frequency_buffer(const float * host, uint64_t bytes,
                          const float ** device) {
    *device = nullptr;
    if (!host) return 0;
    if (bytes > g.rope_freqs_bytes) {
        if (g.graph_capturing) return CUDA_RC_INVALID;
        destroy_decode_graph();
        if (g.rope_freqs) cudaFree(g.rope_freqs);
        g.rope_freqs = nullptr;
        g.rope_freqs_bytes = 0;
        g.rope_freqs_host = nullptr;
        g.rope_freqs_host_bytes = 0;
        int rc = alloc_raw(&g.rope_freqs, bytes, "RoPE frequency buffer");
        if (rc) return rc;
        g.rope_freqs_bytes = bytes;
    }
    if (!async_host_control_enabled()
        || g.rope_freqs_host != host || g.rope_freqs_host_bytes != bytes) {
        const cudaError_t err = cudaMemcpyAsync(
            g.rope_freqs, host, size_t(bytes), cudaMemcpyHostToDevice, g.stream);
        if (err != cudaSuccess) return CUDA_RC_ERROR;
        g.rope_freqs_host = host;
        g.rope_freqs_host_bytes = bytes;
    }
    *device = static_cast<const float *>(g.rope_freqs);
    return 0;
}

int ensure_ple_host_stage(uint64_t bytes) {
    // The common Gemma workflow stages one PLE table per forward. Preserve the
    // backend's general sequential-op semantics too: a second staging call must
    // not overwrite pinned memory while the first DMA is still consuming it.
    if (g.ple_stage_used
        && cudaStreamSynchronize(g.stream) != cudaSuccess) return CUDA_RC_ERROR;
    g.ple_stage_used = true;
    if (bytes <= g.ple_stage_host_bytes) return 0;
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    void * next = nullptr;
    const cudaError_t err = cudaHostAlloc(&next, size_t(bytes), cudaHostAllocDefault);
    if (err != cudaSuccess) return err == cudaErrorMemoryAllocation
        ? CUDA_RC_OOM : CUDA_RC_ERROR;
    if (g.ple_stage_host) cudaFreeHost(g.ple_stage_host);
    g.ple_stage_host = next;
    g.ple_stage_host_bytes = bytes;
    return 0;
}

int ensure_decode_row_host_stage(uint64_t bytes) {
    if (bytes <= g.decode_row_stage_host_bytes) return 0;
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    void * next = nullptr;
    const cudaError_t err = cudaHostAlloc(&next, size_t(bytes), cudaHostAllocDefault);
    if (err != cudaSuccess) return err == cudaErrorMemoryAllocation
        ? CUDA_RC_OOM : CUDA_RC_ERROR;
    if (g.decode_row_stage_host) cudaFreeHost(g.decode_row_stage_host);
    g.decode_row_stage_host = next;
    g.decode_row_stage_host_bytes = bytes;
    return 0;
}

int ensure_decode_token_host_stage() {
    if (g.decode_token_stage_host) return 0;
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    const cudaError_t err = cudaHostAlloc(
        &g.decode_token_stage_host, sizeof(uint32_t), cudaHostAllocDefault);
    return err == cudaSuccess ? 0 : (err == cudaErrorMemoryAllocation
        ? CUDA_RC_OOM : CUDA_RC_ERROR);
}

int ensure_decode_control() {
    if (g.decode_control_host && g.decode_control_device) return 0;
    if (g.graph_capturing) return CUDA_RC_INVALID;
    destroy_decode_graph();
    if (!g.decode_control_host) {
        const cudaError_t host = cudaHostAlloc(
            &g.decode_control_host, sizeof(uint32_t), cudaHostAllocDefault);
        if (host != cudaSuccess) return host == cudaErrorMemoryAllocation
            ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }
    if (!g.decode_control_device) {
        const int device = alloc_raw_with_packed_q4_reclaim(
            &g.decode_control_device, sizeof(uint32_t), "decode control");
        if (device) return device;
    }
    return 0;
}

const uint32_t * host_u32_values(uint32_t id, uint32_t count,
                                 std::vector<uint32_t> & fallback) {
    if (async_host_control_enabled()
        && id < B_COUNT && g.u32_shadow[id].size() >= count) {
        return g.u32_shadow[id].data();
    }
    fallback.resize(count);
    const cudaError_t copy = cudaMemcpyAsync(
        fallback.data(), g.bufs[id], size_t(count) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, g.stream);
    if (copy != cudaSuccess || cudaStreamSynchronize(g.stream) != cudaSuccess) {
        return nullptr;
    }
    return fallback.data();
}

const uint8_t * resident_weight_range(uint64_t off, uint64_t bytes) {
    if (!g.weights_resident || !g.weights || off > g.weights_len
        || bytes > g.weights_len - off) return nullptr;
    for (const ResidentWeightSegment & segment : g.resident_segments) {
        if (off >= segment.file_offset
            && off - segment.file_offset <= segment.bytes
            && bytes <= segment.bytes - (off - segment.file_offset)) {
            return static_cast<const uint8_t *>(g.weights) + segment.device_offset
                + (off - segment.file_offset);
        }
    }
    return nullptr;
}

int weight_slice(uint64_t off, uint64_t bytes, const uint8_t ** out) {
    if (!out || off > g.weights_len || bytes > g.weights_len - off) return CUDA_RC_INVALID;
    if (g.weights_resident) {
        for (const ResidentWeightSegment & segment : g.resident_segments) {
            if (off >= segment.file_offset
                && off - segment.file_offset <= segment.bytes
                && bytes <= segment.bytes - (off - segment.file_offset)) {
                *out = static_cast<const uint8_t *>(g.weights) + segment.device_offset
                    + (off - segment.file_offset);
                return 0;
            }
        }
    }
    int rc = ensure_weight_cache(bytes);
    if (rc) return rc;
    cudaError_t err = cudaMemcpyAsync(g.weight_cache, g.weights_host + off, size_t(bytes),
                                      cudaMemcpyHostToDevice, g.stream);
    if (err != cudaSuccess) return err == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR;
    *out = static_cast<const uint8_t *>(g.weight_cache);
    return 0;
}

// Materialize one self-contained Q8_0_TM row slice in the bounded page cache.
// The file stores all payload bytes before all scales, so a byte-range cut by
// row size is not a valid local tensor. Two ordered copies preserve the file's
// plane layout in scratch without increasing the configured cache bound.
int q8_tm_weight_slice(uint64_t tensor_off, uint32_t n_in,
                       uint32_t tensor_rows, uint32_t row_base,
                       uint32_t rows, const uint8_t ** out) {
    if (!out || !n_in || n_in % 32 || !rows || rows % 8
            || row_base % 8 || row_base > tensor_rows
            || rows > tensor_rows - row_base) {
        return CUDA_RC_INVALID;
    }
    const uint64_t payload_total = uint64_t(tensor_rows) * n_in;
    const uint64_t scale_total = payload_total / 16;
    const uint64_t tensor_bytes = payload_total + scale_total;
    if (const uint8_t * resident =
            resident_weight_range(tensor_off, tensor_bytes)) {
        if (row_base == 0 && rows == tensor_rows) {
            *out = resident;
            return 0;
        }
    }
    const uint64_t payload_bytes = uint64_t(rows) * n_in;
    const uint64_t scale_bytes = payload_bytes / 16;
    const uint64_t slice_bytes = payload_bytes + scale_bytes;
    if (!g.weights_host || tensor_off > g.weights_len
            || tensor_bytes > g.weights_len - tensor_off) {
        return CUDA_RC_INVALID;
    }
    int rc = ensure_weight_cache(slice_bytes);
    if (rc) return rc;
    auto * cache = static_cast<uint8_t *>(g.weight_cache);
    cudaError_t err = cudaMemcpyAsync(
        cache,
        g.weights_host + tensor_off + uint64_t(row_base) * n_in,
        size_t(payload_bytes), cudaMemcpyHostToDevice, g.stream);
    if (err != cudaSuccess) {
        return err == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }
    err = cudaMemcpyAsync(
        cache + payload_bytes,
        g.weights_host + tensor_off + payload_total
            + uint64_t(row_base) * n_in / 16,
        size_t(scale_bytes), cudaMemcpyHostToDevice, g.stream);
    if (err != cudaSuccess) {
        return err == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }
    *out = cache;
    return 0;
}

struct QuantizedWeightPrepackWire {
    uint64_t offset;
    uint32_t n_in;
    uint32_t n_out;
    uint32_t kind;
    uint32_t reserved;
};
static_assert(sizeof(QuantizedWeightPrepackWire) == 24,
    "quantized-weight prepack ABI drift");

// Admission-only conversion from Q8_0 TileMajor to canonical row-major Q4_0.
// The cache is a Decode numerical candidate; model bytes and Prefill stay Q8.
template <bool SourceTileMajor, bool OutputTileMajor>
__global__ void k_q8_tm_to_q4_shadow(
        const uint8_t * source, uint8_t * destination,
        uint32_t n_in, uint32_t n_out) {
    const uint32_t record = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t blocks = n_in / 32;
    const uint32_t records = n_out * blocks;
    if (record >= records) return;
    const uint32_t row = record / blocks;
    const uint32_t block = record - row * blocks;
    const uint64_t unit = uint64_t(row / 8) * blocks + block;
    const uint32_t local_row = row & 7;
    const uint8_t * source_record = SourceTileMajor
        ? source + unit * 8 * 32 + local_row * 32
        : source + uint64_t(record) * 34;
    const int8_t * values = reinterpret_cast<const int8_t *>(
        source_record + (SourceTileMajor ? 0 : 2));
    const float source_scale = SourceTileMajor
        ? __half2float(reinterpret_cast<const __half *>(
            source + uint64_t(n_out) * n_in)[unit * 8 + local_row])
        : __half2float(*reinterpret_cast<const __half *>(source_record));
    const uint64_t output_record = OutputTileMajor
        ? unit * 8 + local_row : uint64_t(record);
    uint8_t * output_values = OutputTileMajor
        ? destination + output_record * 16
        : destination + output_record * 18 + 2;
    __half * output_scale = OutputTileMajor
        ? reinterpret_cast<__half *>(destination + uint64_t(records) * 16)
            + output_record
        : reinterpret_cast<__half *>(destination + output_record * 18);

    float amax = 0.0f;
    float vmax = 0.0f;
#pragma unroll
    for (uint32_t item = 0; item < 32; ++item) {
        const float value = float(values[item]) * source_scale;
        if (fabsf(value) > amax) {
            amax = fabsf(value);
            vmax = value;
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *output_scale = __float2half(d);
#pragma unroll
    for (uint32_t item = 0; item < 16; ++item) {
        int q0 = int(float(values[item]) * source_scale * id + 8.5f);
        int q1 = int(float(values[item + 16]) * source_scale * id + 8.5f);
        q0 = max(0, min(15, q0));
        q1 = max(0, min(15, q1));
        output_values[item] = uint8_t(q0 | (q1 << 4));
    }
}

// Admission-only Q8 TileMajor -> split-scale Q4 conversion used by the
// batched SM86 Q4 kernel. This keeps repacking outside the request path.
__global__ void k_q8_tm_to_aligned_q4_shadow(
        const uint8_t * source, uint8_t * destination,
        uint32_t n_in, uint32_t n_out) {
    const uint32_t record = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t blocks = n_in / 32;
    const uint32_t records = n_out * blocks;
    if (record >= records) return;
    const uint32_t row = record / blocks;
    const uint32_t block = record - row * blocks;
    const uint64_t unit = uint64_t(row / 8) * blocks + block;
    const uint32_t local_row = row & 7;
    const uint8_t * source_record =
        source + unit * 8 * 32 + local_row * 32;
    const int8_t * values = reinterpret_cast<const int8_t *>(source_record);
    const float source_scale = __half2float(reinterpret_cast<const __half *>(
        source + uint64_t(n_out) * n_in)[unit * 8 + local_row]);
    constexpr uint32_t pack_rows = 128;
    constexpr uint32_t pack_kblocks = 4;
    const uint64_t k_groups = uint64_t(blocks) / pack_kblocks;
    const uint64_t output_record =
        ((((uint64_t(row / pack_rows) * k_groups
            + block / pack_kblocks) * pack_rows + row % pack_rows)
            * pack_kblocks) + block % pack_kblocks);
    auto * output_scale =
        reinterpret_cast<__half *>(destination) + output_record;
    uint8_t * output_values = destination
        + uint64_t(records) * sizeof(__half) + output_record * 16;

    float amax = 0.0f;
    float vmax = 0.0f;
#pragma unroll
    for (uint32_t item = 0; item < 32; ++item) {
        const float value = float(values[item]) * source_scale;
        if (fabsf(value) > amax) {
            amax = fabsf(value);
            vmax = value;
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *output_scale = __float2half(d);
#pragma unroll
    for (uint32_t item = 0; item < 16; ++item) {
        int q0 = int(float(values[item]) * source_scale * id + 8.5f);
        int q1 = int(float(values[item + 16]) * source_scale * id + 8.5f);
        q0 = max(0, min(15, q0));
        q1 = max(0, min(15, q1));
        output_values[item] = uint8_t(q0 | (q1 << 4));
    }
}

template <bool SourceTileMajor, bool OutputTileMajor>
__global__ void k_q8_to_q5_shadow(
        const uint8_t * source, uint8_t * destination,
        uint32_t n_in, uint32_t n_out) {
    const uint32_t record = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t blocks = n_in / 32;
    const uint32_t records = n_out * blocks;
    if (record >= records) return;
    const uint32_t row = record / blocks;
    const uint32_t block = record - row * blocks;
    const uint64_t unit = uint64_t(row / 8) * blocks + block;
    const uint32_t local_row = row & 7;
    const uint8_t * source_record = SourceTileMajor
        ? source + unit * 8 * 32 + local_row * 32
        : source + uint64_t(record) * 34;
    const int8_t * values = reinterpret_cast<const int8_t *>(
        source_record + (SourceTileMajor ? 0 : 2));
    const float source_scale = SourceTileMajor
        ? __half2float(reinterpret_cast<const __half *>(
            source + uint64_t(n_out) * n_in)[unit * 8 + local_row])
        : __half2float(*reinterpret_cast<const __half *>(source_record));
    const uint64_t output_record = OutputTileMajor
        ? unit * 8 + local_row : uint64_t(record);
    uint8_t * output_low = OutputTileMajor
        ? destination + output_record * 16
        : destination + output_record * 22 + 6;
    uint8_t * output_high = OutputTileMajor
        ? destination + uint64_t(records) * 16 + output_record * 4
        : destination + output_record * 22 + 2;
    __half * output_scale = OutputTileMajor
        ? reinterpret_cast<__half *>(destination + uint64_t(records) * 20)
            + output_record
        : reinterpret_cast<__half *>(destination + output_record * 22);

    float amax = 0.0f;
    float vmax = 0.0f;
#pragma unroll
    for (uint32_t item = 0; item < 32; ++item) {
        const float value = float(values[item]) * source_scale;
        if (fabsf(value) > amax) {
            amax = fabsf(value);
            vmax = value;
        }
    }
    const float d = vmax / -16.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    *output_scale = __float2half(d);
    uint32_t qh = 0;
#pragma unroll
    for (uint32_t item = 0; item < 16; ++item) {
        int q0 = int(float(values[item]) * source_scale * id + 16.5f);
        int q1 = int(float(values[item + 16]) * source_scale * id + 16.5f);
        q0 = max(0, min(31, q0));
        q1 = max(0, min(31, q1));
        output_low[item] = uint8_t((q0 & 15) | ((q1 & 15) << 4));
        qh |= uint32_t((q0 & 16) >> 4) << item;
        qh |= uint32_t((q1 & 16) >> 4) << (item + 16);
    }
    output_high[0] = uint8_t(qh);
    output_high[1] = uint8_t(qh >> 8);
    output_high[2] = uint8_t(qh >> 16);
    output_high[3] = uint8_t(qh >> 24);
}

extern "C" int imparo_cuda_prepare_quantized_weight_cache(
        const QuantizedWeightPrepackWire * specs, uint32_t count) {
    const char * decode_q4_shadow_mode =
        std::getenv("IMPARO_LAB_LFM2_Q8_TM_DOWN_Q4_SHADOW");
    const uint64_t tuned_ffn_q5_mask = tuner_knob(53);
    const uint64_t tuned_down_q4_mask = tuner_knob(54);
    const uint64_t tuned_down_q5_mask = tuner_knob(55);
    const uint64_t tuned_prefill_down_q4_mask = tuner_knob(57);
    const bool prefill_q4_shadow_lab = decode_q4_shadow_mode
        && std::strcmp(decode_q4_shadow_mode, "prefill") == 0;
    const bool prefill_q4_shadow_requested = prefill_q4_shadow_lab
        || tuned_prefill_down_q4_mask;
    const uint64_t prefill_q4_shadow_mask = prefill_q4_shadow_requested
        ? (tuned_prefill_down_q4_mask
            ? tuned_prefill_down_q4_mask
            : std::strtoull(std::getenv("IMPARO_LAB_LFM2_PREFILL_DOWN_Q4_MASK")
                ? std::getenv("IMPARO_LAB_LFM2_PREFILL_DOWN_Q4_MASK")
                : "0xffffffffffffffff", nullptr, 0))
        : 0;
    const bool tuned_decode_shadow = tuned_ffn_q5_mask
        || tuned_down_q4_mask || tuned_down_q5_mask;
    const bool decode_q4_shadow_requested =
        decode_q4_shadow_mode != nullptr || tuned_decode_shadow
        || tuned_prefill_down_q4_mask;
    const bool decode_q4_shadow_tile_major = decode_q4_shadow_mode
        && std::strcmp(decode_q4_shadow_mode, "tm") == 0;
    const bool decode_q4_shadow_include_head = decode_q4_shadow_mode
        && (std::strcmp(decode_q4_shadow_mode, "head") == 0
            || std::strcmp(decode_q4_shadow_mode, "down-head") == 0
            || std::strcmp(decode_q4_shadow_mode, "q5") == 0);
    const bool decode_shadow_q5 = decode_q4_shadow_mode
        && std::strcmp(decode_q4_shadow_mode, "q5") == 0;
    const bool decode_shadow_mixed_lab = decode_q4_shadow_mode
        && std::strcmp(decode_q4_shadow_mode, "mixed") == 0;
    const bool decode_shadow_mixed_ffn_lab = decode_q4_shadow_mode
        && std::strcmp(decode_q4_shadow_mode, "mixed-ffn") == 0;
    const bool decode_shadow_mixed =
        tuned_decode_shadow || decode_shadow_mixed_lab;
    const bool decode_shadow_mixed_ffn =
        tuned_ffn_q5_mask || decode_shadow_mixed_ffn_lab;
    const uint64_t decode_shadow_q4_mask = tuned_decode_shadow
        ? tuned_down_q4_mask
        : (decode_shadow_mixed || decode_shadow_mixed_ffn)
        ? std::strtoull(std::getenv("IMPARO_LAB_LFM2_DOWN_Q4_MASK")
            ? std::getenv("IMPARO_LAB_LFM2_DOWN_Q4_MASK") : "0", nullptr, 0)
        : UINT64_MAX;
    const uint64_t decode_shadow_q5_mask = tuned_decode_shadow
        ? tuned_down_q5_mask
        : (decode_shadow_mixed || decode_shadow_mixed_ffn)
        ? std::strtoull(std::getenv("IMPARO_LAB_LFM2_DOWN_Q5_MASK")
            ? std::getenv("IMPARO_LAB_LFM2_DOWN_Q5_MASK") : "0", nullptr, 0)
        : (decode_shadow_q5 ? UINT64_MAX : 0);
    const uint64_t decode_shadow_gate_q4_mask = tuned_decode_shadow
        ? 0
        : decode_shadow_mixed_ffn
        ? std::strtoull(std::getenv("IMPARO_LAB_LFM2_GATE_Q4_MASK")
            ? std::getenv("IMPARO_LAB_LFM2_GATE_Q4_MASK") : "0", nullptr, 0)
        : 0;
    const uint64_t decode_shadow_gate_q5_mask = tuned_decode_shadow
        ? tuned_ffn_q5_mask
        : decode_shadow_mixed_ffn
        ? std::strtoull(std::getenv("IMPARO_LAB_LFM2_GATE_Q5_MASK")
            ? std::getenv("IMPARO_LAB_LFM2_GATE_Q5_MASK") : "0", nullptr, 0)
        : 0;
    const uint64_t decode_shadow_up_q4_mask = tuned_decode_shadow
        ? 0
        : decode_shadow_mixed_ffn
        ? std::strtoull(std::getenv("IMPARO_LAB_LFM2_UP_Q4_MASK")
            ? std::getenv("IMPARO_LAB_LFM2_UP_Q4_MASK") : "0", nullptr, 0)
        : 0;
    const uint64_t decode_shadow_up_q5_mask = tuned_decode_shadow
        ? tuned_ffn_q5_mask
        : decode_shadow_mixed_ffn
        ? std::strtoull(std::getenv("IMPARO_LAB_LFM2_UP_Q5_MASK")
            ? std::getenv("IMPARO_LAB_LFM2_UP_Q5_MASK") : "0", nullptr, 0)
        : 0;
    const auto mask_precision = [](uint64_t q4, uint64_t q5, uint32_t index) {
        if (index >= 64) return 0u;
        const uint64_t bit = uint64_t(1) << index;
        if (q5 & bit) return 5u;
        if (q4 & bit) return 4u;
        return 0u;
    };
    const auto shadow_precision = [&](uint32_t index, bool head) {
        if (prefill_q4_shadow_lab) return 0u;
        if (head) return decode_shadow_q5 ? 5u : 4u;
        if (decode_shadow_mixed_ffn) {
            const uint32_t layer = index / 3;
            switch (index % 3) {
                case 0: return mask_precision(
                    decode_shadow_gate_q4_mask, decode_shadow_gate_q5_mask, layer);
                case 1: return mask_precision(
                    decode_shadow_up_q4_mask, decode_shadow_up_q5_mask, layer);
                default: return mask_precision(
                    decode_shadow_q4_mask, decode_shadow_q5_mask, layer);
            }
        }
        if (!decode_shadow_mixed) return decode_shadow_q5 ? 5u : 4u;
        return mask_precision(decode_shadow_q4_mask, decode_shadow_q5_mask, index);
    };
    if (decode_q4_shadow_requested) {
        if (g.sm_version != 86 || !g.weights_resident
                || !specs || !count || g.forward_open
                || g.graph_capturing || g.prefill_capture_active) return 0;
        if (g.decode_q4_shadow_ready) return 1;

        uint64_t total_bytes = 0;
        uint32_t prefill_down_index = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const auto & spec = specs[i];
            const bool down = spec.kind == 3 && spec.n_in > spec.n_out;
            const bool gate_or_up = decode_shadow_mixed_ffn
                && spec.kind == 3 && spec.n_in < spec.n_out;
            const bool head = decode_q4_shadow_include_head
                && spec.kind == 2 && spec.n_in < spec.n_out;
            const bool ordered_ffn_shape = !decode_shadow_mixed_ffn
                || ((i % 3 == 2) == down);
            if (spec.reserved != 0 || (!down && !gate_or_up && !head)
                    || !ordered_ffn_shape
                    || !spec.n_in || !spec.n_out
                    || spec.n_in % 32 || spec.n_out % 8) return 0;
            const uint32_t precision = shadow_precision(i, head);
            const bool prefill_q4 = down && prefill_down_index < 64
                && (prefill_q4_shadow_mask
                    & (uint64_t(1) << prefill_down_index));
            if (down) ++prefill_down_index;
            if (!precision && !prefill_q4) continue;
            const uint64_t records =
                uint64_t(spec.n_out) * (spec.n_in / 32);
            const uint64_t source_bytes = records * 34;
            const uint64_t q4_bytes = records * (precision == 5 ? 22 : 18);
            if (!resident_weight_range(spec.offset, source_bytes)
                    || (precision && q4_bytes > UINT64_MAX - total_bytes)) return 0;
            if (precision) total_bytes += q4_bytes;
            const uint64_t prefill_bytes = records * 18;
            if (prefill_q4) {
                if (prefill_bytes > UINT64_MAX - total_bytes) return 0;
                total_bytes += prefill_bytes;
            }
        }
        if (!initialize_packed_q4_budget()
                || total_bytes > g.packed_q4_budget
                    - std::min(g.packed_q4_budget, g.packed_q4_bytes)) {
            return 0;
        }

        void * slab = nullptr;
        if (alloc_raw(&slab, total_bytes, "Decode Down Q4 shadow")) {
            cudaGetLastError();
            return 0;
        }
        std::vector<DecodeQ4ShadowSpan> spans;
        spans.reserve(count);
        std::vector<PrefillQ4ShadowSpan> prefill_spans;
        prefill_spans.reserve(count);
        uint64_t cursor = 0;
        prefill_down_index = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const auto & spec = specs[i];
            const bool down = spec.kind == 3 && spec.n_in > spec.n_out;
            const bool head = decode_q4_shadow_include_head
                && spec.kind == 2 && spec.n_in < spec.n_out;
            const uint32_t precision = shadow_precision(i, head);
            const bool prefill_q4 = down && prefill_down_index < 64
                && (prefill_q4_shadow_mask
                    & (uint64_t(1) << prefill_down_index));
            if (down) ++prefill_down_index;
            if (!precision && !prefill_q4) continue;
            const uint64_t records =
                uint64_t(spec.n_out) * (spec.n_in / 32);
            const uint64_t source_bytes = records * 34;
            const uint64_t q4_bytes = records * (precision == 5 ? 22 : 18);
            const uint8_t * source =
                resident_weight_range(spec.offset, source_bytes);
            if (precision) {
                auto * destination = static_cast<uint8_t *>(slab) + cursor;
                const bool output_tile_major = decode_q4_shadow_tile_major
                    || decode_shadow_mixed_ffn || spec.kind == 2;
                if (precision == 5 && output_tile_major && spec.kind == 3) {
                    k_q8_to_q5_shadow<true, true><<<
                        (records + 255) / 256, 256, 0, g.stream>>>(
                            source, destination, spec.n_in, spec.n_out);
                } else if (precision == 5 && spec.kind == 2) {
                    k_q8_to_q5_shadow<false, true><<<
                        (records + 255) / 256, 256, 0, g.stream>>>(
                            source, destination, spec.n_in, spec.n_out);
                } else if (precision == 5) {
                    k_q8_to_q5_shadow<true, false><<<
                        (records + 255) / 256, 256, 0, g.stream>>>(
                            source, destination, spec.n_in, spec.n_out);
                } else if (spec.kind == 2) {
                    k_q8_tm_to_q4_shadow<false, true><<<
                        (records + 255) / 256, 256, 0, g.stream>>>(
                            source, destination, spec.n_in, spec.n_out);
                } else if (output_tile_major) {
                    k_q8_tm_to_q4_shadow<true, true><<<
                        (records + 255) / 256, 256, 0, g.stream>>>(
                            source, destination, spec.n_in, spec.n_out);
                } else {
                    k_q8_tm_to_q4_shadow<true, false><<<
                        (records + 255) / 256, 256, 0, g.stream>>>(
                            source, destination, spec.n_in, spec.n_out);
                }
                spans.push_back(
                    {spec.offset, destination, spec.n_in, spec.n_out,
                     output_tile_major, precision == 5});
                cursor += q4_bytes;
            }
            if (prefill_q4) {
                auto * destination = static_cast<uint8_t *>(slab) + cursor;
                k_q8_tm_to_aligned_q4_shadow<<<
                    (records + 255) / 256, 256, 0, g.stream>>>(
                        source, destination, spec.n_in, spec.n_out);
                prefill_spans.push_back(
                    {spec.offset, destination, spec.n_in, spec.n_out});
                cursor += records * 18;
            }
        }
        if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(slab);
            return 0;
        }
        g.decode_q4_shadow_slab = slab;
        g.decode_q4_shadow_bytes = total_bytes;
        g.decode_q4_shadow_spans = std::move(spans);
        g.prefill_q4_shadow_spans = std::move(prefill_spans);
        g.decode_q4_shadow_ready = true;
        g.packed_q4_bytes += total_bytes;
        return 1;
    }

    // Slot 39 values 4/5 are the per-model/per-device fast-transaction candidates.
    // Value 5 deliberately excludes staged Attention so the receipt can distinguish
    // component safety from a combined numerical-route interaction.
    if ((g.knobs[39] != 4 && g.knobs[39] != 5)
            || g.sm_version != 86 || !g.weights_resident) return 0;
    if (g.ffn_sidecar_model_ready) return 1;
    if (!specs || !count || g.forward_open || g.graph_capturing
            || g.prefill_capture_active) return 0;

    bool complete = true;
    for (uint32_t i = 0; i < count; ++i) {
        const auto & spec = specs[i];
        if (spec.reserved != 0 || spec.kind != 1 || !spec.n_in || !spec.n_out
                || spec.n_in % 128 || spec.n_out % 128
                || uint64_t(spec.n_out) > UINT64_MAX / (spec.n_in / 32)
                || uint64_t(spec.n_out) * (spec.n_in / 32) > UINT64_MAX / 18) {
            complete = false;
            break;
        }
        const uint64_t bytes = uint64_t(spec.n_out) * (spec.n_in / 32) * 18;
        const uint8_t * raw = nullptr;
        if (weight_slice(spec.offset, bytes, &raw)
                || !ensure_aligned_packed_q4_lab(raw, spec.n_in, spec.n_out)) {
            complete = false;
            break;
        }
    }
    if (complete && cudaStreamSynchronize(g.stream) == cudaSuccess) {
        g.ffn_sidecar_model_ready = true;
        g.ffn_sidecar_model_pack_failed = false;
        return 1;
    }
    cudaGetLastError();
    g.ffn_sidecar_model_pack_failed = true;
    release_packed_q4_for_priority_allocation();
    return 0;
}

const uint8_t * resident_weight_at(uint64_t off) {
    if (!g.weights_resident || off >= g.weights_len) return nullptr;
    for (const ResidentWeightSegment & segment : g.resident_segments) {
        if (off >= segment.file_offset && off - segment.file_offset < segment.bytes) {
            return static_cast<const uint8_t *>(g.weights) + segment.device_offset
                + (off - segment.file_offset);
        }
    }
    return nullptr;
}

// ---- kernels ----------------------------------------------------------------------
// Q4_0: 18-byte blocks, half scale then 16 packed bytes; value i in the low nibble,
// i+16 in the high one; v = (q - 8) * d. Matches the .mm and imparo-cpu bit for bit.

__device__ inline float q4_value(const uint8_t * blk, int i) {
    const __half d = *reinterpret_cast<const __half *>(blk);
    const uint8_t pk = blk[2 + (i & 15)];
    const int q = (i < 16) ? (pk & 0x0F) : (pk >> 4);
    return (float(q) - 8.0f) * __half2float(d);
}

// Q8_0: one f16 scale followed by 32 signed bytes. This is a weight format, distinct
// from the 36-byte Q8_1 activation block used by the Q4 MMVQ family below.
__device__ inline float q8_0_value(const uint8_t * blk, int i) {
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    return float(int8_t(blk[2 + i])) * d;
}

// Correctness-first Q8_0 projection. One warp owns one output row and reduces each
// 32-value wire block before advancing, so the f16 scale is applied in block order.
// An architecture-owned tiled family can replace this route later without changing the
// weight ABI or the low-memory slicing contract.
__global__ void k_gemm_q8_0_f32(const uint8_t * w, const float * x, float * y,
                                uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                                uint32_t src_row, uint32_t out_stride,
                                uint32_t row_base) {
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + warp;
    const uint32_t tok = blockIdx.y;
    if (row >= n_out || tok >= n_tok) return;
    const uint32_t blocks = n_in / 32;
    const uint8_t * wr = w + uint64_t(row) * blocks * 34;
    const float * xr = x + uint64_t(src_row + tok) * n_in;
    float acc = 0.0f;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint8_t * blk = wr + uint64_t(block) * 34;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        float partial = float(int8_t(blk[2 + lane])) * xr[block * 32 + lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) acc += partial * d;
    }
    if (lane == 0) {
        y[uint64_t(tok) * out_stride + row_base + row] = acc;
    }
}


// Q8_0_TM fallback preserves the row-major kernel's block accumulation order,
// while reading each row from 8-row tile-major payload and scale planes.
__global__ void k_gemm_q8_0_tm_f32(const uint8_t * w, const float * x,
                                   float * y, uint32_t n_in,
                                   uint32_t n_out, uint32_t n_tok,
                                   uint32_t src_row, uint32_t out_stride,
                                   uint32_t row_base) {
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + warp;
    const uint32_t tok = blockIdx.y;
    if (row >= n_out || tok >= n_tok) return;
    const uint32_t blocks = n_in / 32;
    const auto * scales = reinterpret_cast<const __half *>(
        w + uint64_t(n_out) * n_in);
    const float * xr = x + uint64_t(src_row + tok) * n_in;
    float acc = 0.0f;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint64_t unit = uint64_t(row / 8) * blocks + block;
        const uint8_t * values = w + unit * 256 + (row & 7) * 32;
        const float d = __half2float(scales[unit * 8 + (row & 7)]);
        float partial = float(int8_t(values[lane]))
            * xr[block * 32 + lane];
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, offset);
        }
        if (lane == 0) acc += partial * d;
    }
    if (lane == 0) {
        y[uint64_t(tok) * out_stride + row_base + row] = acc;
    }
}

// Correctness-first decode path. This keeps f32 activations and directly dequantizes
// Q4_0 weights. The Q8_1 MMVQ variant below remains opt-in until it passes the
// external llama logit gate as well as its speed gate.
__global__ void k_gemv_q4_f32(const uint8_t * w, const float * x, float * y,
                              uint32_t n_in, uint32_t n_out, uint32_t row_base) {
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const uint8_t * wr = w + (uint64_t)row * (n_in / 32) * 18;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_in / 32; ++b) {
        const uint8_t * blk = wr + (uint64_t)b * 18;
        acc += q4_value(blk, lane) * x[b * 32 + lane];
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) y[row_base + row] = acc;
}

// Decode projections use the same Q4_0 x Q8_1 arithmetic as llama.cpp's CUDA MMVQ.
// Quantizing activations is part of that operator's numerical contract: dequantizing
// Q4 weights and multiplying by f32 activations produces materially different Gemma4
// logits even though both are reasonable approximations of the same real matrix.
struct BlockQ8_1 {
    __half d;
    __half s;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_1) == 36, "Q8_1 layout");

// MMQ groups four adjacent Q8_1 blocks into one 144-byte record. Records are
// written K-group-major (all tokens for one 128-value K group are adjacent),
// so an MMQ CTA can copy its activation tile to shared memory with coalesced
// loads. The size is unchanged; decode/MMVQ retains BlockQ8_1 above.
struct BlockQ8_1Mmq {
    int8_t qs[128];
    // The producer selects DS4-compatible half rounding or Q8_0's full-f32 D4
    // route. Four floats occupy the same 16-byte record tail as d+s.
    float d[4];
};
static_assert(sizeof(BlockQ8_1Mmq) == 4 * sizeof(BlockQ8_1),
              "Q8_1 MMQ layout");

__device__ __forceinline__ float warp_sum_xor(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// Keep one implementation for both standalone and fused projection epilogues.
// The expression matches ggml-cuda exactly; algebraic reassociation under
// --use_fast_math can otherwise move values across Q8_1 quantization boundaries.
__device__ __forceinline__ float cuda_gelu(float x) {
    constexpr float kA = 0.044715f;
    constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * x * (1.0f + kA * x * x)));
}

#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
#include "sm86/gemm_q4_f16_cublas.cuh"
#endif

__global__ void k_quantize_q8_1(const float * x, BlockQ8_1 * y,
                                uint32_t n_in, uint32_t n_tok,
                                uint32_t src_row) {
    // Match llama CUDA's MMQ quantizer structurally, not only algebraically. Four
    // adjacent values per thread and 8-lane XOR subgroups are part of the numeric
    // contract under --use_fast_math and can move borderline values by one int8.
    const uint32_t tok = blockIdx.x;
    const uint32_t i0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (tok >= n_tok || i0 >= n_in) return;
    const float4 xi = reinterpret_cast<const float4 *>(
        x + uint64_t(src_row + tok) * n_in)[i0 / 4];
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
    }
    float sum = xi.x + xi.y + xi.z + xi.w;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = int8_t(roundf(xi.x * d_inv));
    q.y = int8_t(roundf(xi.y * d_inv));
    q.z = int8_t(roundf(xi.z * d_inv));
    q.w = int8_t(roundf(xi.w * d_inv));
    const float d = 1.0f / d_inv;
    const uint32_t block = i0 / 32;
    const uint32_t iqs = i0 % 32;
    BlockQ8_1 * out = y + uint64_t(tok) * (n_in / 32) + block;
    reinterpret_cast<char4 *>(out->qs)[iqs / 4] = q;
    if (iqs == 0) {
        out->d = __float2half(d);
        // Q8_1 stores the pre-quantization float sum. This is the ggml CUDA MMVQ
        // contract used by the normal llama CUDA/FA path (and by its Q4_0 correction).
        out->s = __float2half(sum);
    }
}

__global__ void k_quantize_q8_1_mmq(const float * x, BlockQ8_1Mmq * y,
                                    uint32_t n_in, uint32_t n_tok,
                                    uint32_t src_row,
                                    bool full_precision_scale) {
    const uint32_t tok = blockIdx.x;
    const uint32_t i0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (tok >= n_tok || i0 >= n_in) return;
    const float4 xi = reinterpret_cast<const float4 *>(
        x + uint64_t(src_row + tok) * n_in)[i0 / 4];
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = int8_t(roundf(xi.x * d_inv));
    q.y = int8_t(roundf(xi.y * d_inv));
    q.z = int8_t(roundf(xi.z * d_inv));
    q.w = int8_t(roundf(xi.w * d_inv));
    const float d = 1.0f / d_inv;
    const uint32_t block = i0 / 32;
    const uint32_t block_in_group = block % 4;
    const uint32_t iqs = i0 % 32;
    BlockQ8_1Mmq * out = y + uint64_t(block / 4) * n_tok + tok;
    reinterpret_cast<char4 *>(out->qs + block_in_group * 32)[iqs / 4] = q;
    if (iqs == 0) {
        // Q8_0 MMQ consumes llama's D4 route and keeps the scale in f32.
        // Q4_0 MMQ consumes DS4, whose scale is rounded through f16. Keep the
        // distinction explicit: mixed-quant models must not share cache entries.
        out->d[block_in_group] = full_precision_scale
            ? d : __half2float(__float2half(d));
    }
}

__device__ __forceinline__ float dot_q4_0_q8_1_half(
        const uint8_t * q4, const BlockQ8_1 * q8, uint32_t iqs) {
    const uint16_t * q4i = reinterpret_cast<const uint16_t *>(q4 + 2);
    const int * q8i = reinterpret_cast<const int *>(q8->qs);
    int sumi = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        // Q4_0 blocks are 18 bytes, so alternating blocks are only 2-byte aligned.
        // Match llama.cpp's get_int_b2 instead of issuing a misaligned 32-bit load.
        const uint32_t j = iqs + i;
        const int packed = int(q4i[2 * j]) | (int(q4i[2 * j + 1]) << 16);
        const int lo = packed & 0x0F0F0F0F;
        const int hi = (packed >> 4) & 0x0F0F0F0F;
        sumi = __dp4a(lo, q8i[iqs + i], sumi);
        sumi = __dp4a(hi, q8i[iqs + i + 4], sumi);
    }
    const float d4 = __half2float(*reinterpret_cast<const __half *>(q4));
    return d4 * (float(sumi) * __half2float(q8->d) - 4.0f * __half2float(q8->s));
}

// One output row per block. The default four-warp schedule and XOR reductions match
// the exact fork on SM86; the registered gemv_warps knob remains available to tune.
__global__ void k_gemv_q4_q8_1(const uint8_t * w, const BlockQ8_1 * x, float * y,
                               uint32_t n_in, uint32_t n_out, uint32_t row_base,
                               uint32_t nwarps) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    if (row >= n_out || warp >= nwarps) return;
    const uint32_t blocks = n_in / 32;
    const uint32_t blocks_per_iter = 16 * nwarps;
    const uint32_t iqs = 2 * (tid & 1);
    float acc = 0.0f;
    for (uint32_t b = tid / 2; b < blocks; b += blocks_per_iter) {
        const uint8_t * q4 = w + ((uint64_t)row * blocks + b) * 18;
        acc += dot_q4_0_q8_1_half(q4, x + b, iqs);
    }
    __shared__ float partial[7][32];
    if (warp > 0) partial[warp - 1][lane] = acc;
    __syncthreads();
    if (warp > 0) return;
    for (uint32_t i = 1; i < nwarps; ++i) acc += partial[i - 1][lane];
    acc = warp_sum_xor(acc);
    if (lane == 0) y[row_base + row] = acc;
}

#include "sm80/mmvq_q4_q8_1.cuh"
#include "sm80/mmvq_q8_q8_1.cuh"
#include "sm86/mmvq_q8_tm_gate_up_silu.cuh"

// Batched Q4_0 x Q8_1 projection. Each warp owns one (output row, token) pair and
// uses dp4a over the quantized activation. This is also the numerical contract of
// llama CUDA MMQ: batch projections must not silently switch back to f32 activation
// arithmetic. Consecutive warps consume consecutive weight rows while sharing the
// same activation row through L2.
__global__ void k_gemm_q4_q8_1(const uint8_t * w, const BlockQ8_1 * x, float * y,
                               uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                               uint32_t epilogue, uint32_t out_stride,
                               uint32_t row_base, uint32_t nwarps) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t row = blockIdx.x * nwarps + warp;
    const uint32_t tok = blockIdx.y;
    if (row >= n_out || tok >= n_tok) return;

    const uint32_t blocks = n_in / 32;
    const uint32_t iqs = 2 * (lane & 1);
    const BlockQ8_1 * xr = x + (uint64_t)tok * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane / 2; b < blocks; b += 16) {
        const uint8_t * q4 = w + ((uint64_t)row * blocks + b) * 18;
        acc += dot_q4_0_q8_1_half(q4, xr + b, iqs);
    }
    acc = warp_sum_xor(acc);
    if (lane != 0) return;

    float * slot = y + (uint64_t)tok * out_stride + row_base + row;
    if (epilogue) {
        const float g = *slot;
        *slot = cuda_gelu(g) * acc;
    } else {
        *slot = acc;
    }
}

// Tensor-core batch path used by the SM75+ plugins. A tile covers 16 weight rows
// and 16 tokens. For each Q4 block, two k=16 integer MMA instructions form the
// exact 32-element signed dot before the Q4/Q8 scales are applied in f32. Keeping
// the scale boundary at 32 elements is important for agreement with llama CUDA.
__global__ void k_gemm_q4_q8_1_mma(const uint8_t * w, const BlockQ8_1 * x, float * y,
                                   uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                                   uint32_t epilogue, uint32_t out_stride,
                                   uint32_t row_base, uint32_t sm_count,
                                   uint32_t stream_k_numeric,
                                   uint32_t fixed_segment_blocks,
                                   uint32_t virtual_token_base,
                                   uint32_t virtual_schedule) {
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    constexpr uint32_t tile = 16;
    const uint32_t row0 = blockIdx.x * tile;
    const uint32_t tok0 = blockIdx.y * tile;
    const uint32_t lane = threadIdx.x;
    const uint32_t blocks = n_in / 32;

    __shared__ __align__(16) int8_t a[tile * tile];
    __shared__ __align__(16) int8_t b[tile * tile];
    __shared__ __align__(16) int c[tile * tile];

    float suffix[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float fixup[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t ntx = virtual_schedule ? 4u : (n_tok + 127) / 128;
    const uint32_t nty = (out_stride + 127) / 128;
    const uint32_t ntiles = ntx * nty;
    const uint32_t nwaves = (ntiles + sm_count - 1) / sm_count;
    const uint32_t efficiency = 100 * ntiles / (sm_count * nwaves);
    const uint32_t stream_grid = efficiency >= 90 ? ntiles : sm_count;
    const uint32_t token_tile = virtual_schedule
        ? ((virtual_token_base + tok0) % 512) / 128 : tok0 / 128;
    const uint32_t llama_tile = ((row_base + row0) / 128) * ntx + token_tile;
    const uint64_t total_work = uint64_t(ntiles) * blocks;
    uint32_t chunk_end = blocks;
    bool suffix_phase = true;
    while (chunk_end > 0) {
        uint32_t chunk_start = 0;
        if (fixed_segment_blocks) {
            chunk_start = chunk_end > fixed_segment_blocks
                ? chunk_end - fixed_segment_blocks : 0;
        } else if (stream_k_numeric && lane == 0) {
            for (uint32_t bi = 1; bi < stream_grid; ++bi) {
                uint64_t boundary = uint64_t(bi) * total_work / stream_grid;
                boundary -= (boundary % blocks) % 8;
                if (boundary / blocks == llama_tile) {
                    const uint32_t kb = uint32_t(boundary % blocks);
                    if (kb < chunk_end) chunk_start = max(chunk_start, kb);
                }
            }
        }
        chunk_start = __shfl_sync(0xffffffff, chunk_start, 0);
        float partial[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t kb = chunk_start; kb < chunk_end; ++kb) {
        wmma::fragment<wmma::accumulator, tile, tile, tile, int> cf;
        wmma::fill_fragment(cf, 0);

        #pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t linear = lane; linear < tile * tile; linear += 32) {
                const uint32_t r = linear / tile;
                const uint32_t k = linear % tile;
                const uint32_t row = row0 + r;
                int8_t q = 0;
                if (row < n_out) {
                    const uint8_t * q4 = w + ((uint64_t)row * blocks + kb) * 18;
                    const uint8_t packed = q4[2 + k];
                    q = int8_t(int(half == 0 ? (packed & 0x0f) : (packed >> 4)) - 8);
                }
                a[linear] = q;

                const uint32_t t = linear / tile;
                const uint32_t bk = linear % tile;
                const uint32_t tok = tok0 + t;
                // B is column-major KxN: each token is one contiguous column.
                b[bk + t * tile] = tok < n_tok
                    ? x[(uint64_t)tok * blocks + kb].qs[half * tile + bk]
                    : int8_t(0);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, tile, tile, tile,
                           signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, tile, tile, tile,
                           signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(af, a, tile);
            wmma::load_matrix_sync(bf, b, tile);
            wmma::mma_sync(cf, af, bf, cf);
            __syncwarp();
        }

        wmma::store_matrix_sync(c, cf, tile, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            const uint32_t linear = lane + item * 32;
            const uint32_t r = linear / tile;
            const uint32_t t = linear % tile;
            const uint32_t row = row0 + r;
            const uint32_t tok = tok0 + t;
            if (row < n_out && tok < n_tok) {
                const uint8_t * q4 = w + ((uint64_t)row * blocks + kb) * 18;
                const float d4 = __half2float(*reinterpret_cast<const __half *>(q4));
                const float d8 = __half2float(x[(uint64_t)tok * blocks + kb].d);
                partial[item] += float(c[linear]) * d4 * d8;
            }
        }
        __syncwarp();
        }
        #pragma unroll
        for (uint32_t item = 0; item < 8; ++item) {
            if (suffix_phase) suffix[item] = partial[item];
            else fixup[item] += partial[item];
        }
        suffix_phase = false;
        chunk_end = chunk_start;
    }

    float sums[8];
    #pragma unroll
    for (uint32_t item = 0; item < 8; ++item) sums[item] = suffix[item] + fixup[item];

    #pragma unroll
    for (uint32_t item = 0; item < 8; ++item) {
        const uint32_t linear = lane + item * 32;
        const uint32_t r = linear / tile;
        const uint32_t t = linear % tile;
        const uint32_t row = row0 + r;
        const uint32_t tok = tok0 + t;
        if (row >= n_out || tok >= n_tok) continue;
        float * slot = y + (uint64_t)tok * out_stride + row_base + row;
        if (epilogue) {
            const float g = *slot;
            *slot = cuda_gelu(g) * sums[item];
        } else {
            *slot = sums[item];
        }
    }
#else
    (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)n_tok;
    (void)epilogue; (void)out_stride; (void)row_base; (void)sm_count;
    (void)stream_k_numeric; (void)fixed_segment_blocks;
    (void)virtual_token_base; (void)virtual_schedule;
#endif
}

#include "sm80/mmq_q4_q8_1.cuh"
#include "sm86/mmq_q4_q8_aligned_prepack.cuh"
#include "sm86/mmq_q4_q8_dp4a.cuh"
#include "sm86/mmq_q4_q8_interleaved.cuh"
#include "sm86/mmq_q4_q8_pair.cuh"
#include "sm86/mmq_q4_q8_down_pipe.cuh"
#include "sm86/mmq_q4_q8_gate_up_pipe.cuh"
#include "sm86/mmq_q4_q8_ple_gate.cuh"
#include "sm86/q8_mma_ready_a2.cuh"
#include "sm86/q8_ready_batched_r2.cuh"
#include "sm80/mmq_q8_q8_1.cuh"
#include "sm86/mmq_q8_tm_gate_up_row_pair.cuh"

// Diagnostic reference GEMM: one thread per (row, token), directly dequantizing
// weights against f32 activations. It is intentionally not selected in production.
__global__ void k_gemm_q4(const uint8_t * w, const float * x, float * y,
                          uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                          uint32_t src_row, uint32_t epilogue,
                          uint32_t out_stride, uint32_t row_base) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= n_out || tok >= n_tok) return;
    const uint8_t * wr = w + (uint64_t)row * (n_in / 32) * 18;
    const float * xr = x + (uint64_t)(src_row + tok) * n_in;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_in / 32; ++b) {
        const uint8_t * blk = wr + (uint64_t)b * 18;
        #pragma unroll 4
        for (int i = 0; i < 32; ++i) acc += q4_value(blk, i) * xr[b * 32 + i];
    }
    float * slot = y + (uint64_t)tok * out_stride + row_base + row;
    if (epilogue) {
        // gelu(gate) * up fused into the up projection's write-back (the .mm's
        // gated-activation epilogue): y currently holds the gate value.
        const float g = *slot;
        *slot = cuda_gelu(g) * acc;
    } else {
        *slot = acc;
    }
}

__global__ void k_gemm_f32(const float * w, const float * x, float * y,
                           uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                           uint32_t src_row, uint32_t out_stride,
                           uint32_t row_base) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= n_out || tok >= n_tok) return;
    const float * wr = w + (uint64_t)row * n_in;
    const float * xr = x + (uint64_t)(src_row + tok) * n_in;
    float acc = 0.0f;
    for (uint32_t i = 0; i < n_in; ++i) acc += wr[i] * xr[i];
    y[(uint64_t)tok * out_stride + row_base + row] = acc;
}

template <uint32_t BlockSize>
__device__ __forceinline__ float block_sum(float value, float * shared) {
    value = warp_sum_xor(value);
    constexpr uint32_t Warps = BlockSize / 32;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = lane < Warps ? shared[lane] : 0.0f;
    return warp_sum_xor(value);
}

static uint3 init_fastdiv_values(uint32_t d) {
    uint32_t l = 0;
    while (l < 32 && (uint32_t{1} << l) < d) ++l;
    const uint32_t mp = uint32_t((uint64_t{1} << 32) * ((uint64_t{1} << l) - d) / d + 1);
    return make_uint3(mp, l, d);
}

__device__ __forceinline__ uint32_t fastmodulo(uint32_t n, uint3 divisor) {
    const uint32_t quotient = (__umulhi(n, divisor.x) + n) >> divisor.y;
    return n - quotient * divisor.z;
}

// CUDA-agreement specialization. Its signature and indexing deliberately mirror
// ggml-cuda's public RMSNorm operator so ptxas uses the same per-thread accumulation
// schedule. Keep this compatibility kernel separate from the compact common kernel:
// other SM families remain free to select a native implementation.
template <int BlockSize, int SumUlpBias = 0, bool DoAdd = false,
          bool ApplyOutputScale = false, bool QuantizeMmq = false>
__global__ void k_rms_norm_ggml(const float * src, float * dst, int ncols,
                                int64_t stride_row, int64_t stride_channel,
                                int64_t stride_sample, float eps, const float * mul,
                                int64_t mul_stride_row, int64_t mul_stride_channel,
                                int64_t mul_stride_sample, uint3 mul_ncols_packed,
                                uint3 mul_nrows_packed, uint3 mul_nchannels_packed,
                                uint3 mul_nsamples_packed, const float * add,
                                int64_t add_stride_row, int64_t add_stride_channel,
                                int64_t add_stride_sample, uint3 add_ncols_packed,
                                uint3 add_nrows_packed, uint3 add_nchannels_packed,
                                uint3 add_nsamples_packed, float output_scale,
                                BlockQ8_1Mmq * q8, uint32_t q8_n_tok) {
    const int nrows = gridDim.x;
    const int nchannels = gridDim.y;
    const int row = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample = blockIdx.z;
    const int tid = threadIdx.x;

    src += int64_t(sample) * stride_sample + int64_t(channel) * stride_channel
        + int64_t(row) * stride_row;
    dst += ((sample * nchannels + channel) * nrows + row) * ncols;
    const uint32_t mul_row = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample = fastmodulo(sample, mul_nsamples_packed);
    mul += int64_t(mul_sample) * mul_stride_sample
        + int64_t(mul_channel) * mul_stride_channel + int64_t(mul_row) * mul_stride_row;
    if constexpr (DoAdd) {
        const uint32_t add_row = fastmodulo(row, add_nrows_packed);
        const uint32_t add_channel = fastmodulo(channel, add_nchannels_packed);
        const uint32_t add_sample = fastmodulo(sample, add_nsamples_packed);
        add += int64_t(add_sample) * add_stride_sample
            + int64_t(add_channel) * add_stride_channel + int64_t(add_row) * add_stride_row;
    } else {
        (void)add; (void)add_stride_row; (void)add_stride_channel; (void)add_stride_sample;
        (void)add_ncols_packed; (void)add_nrows_packed;
        (void)add_nchannels_packed; (void)add_nsamples_packed;
    }

    float sum = 0.0f;
    for (int col = tid; col < ncols; col += BlockSize) {
        const float value = src[col];
        sum += value * value;
    }
    extern __shared__ float shared[];
    sum = block_sum<BlockSize>(sum, shared);
    if constexpr (SumUlpBias > 0) {
        sum = __uint_as_float(__float_as_uint(sum) + SumUlpBias);
    }
    const float mean = sum / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += BlockSize) {
        const uint32_t mul_col = fastmodulo(col, mul_ncols_packed);
        if constexpr (DoAdd) {
            const uint32_t add_col = fastmodulo(col, add_ncols_packed);
            float value = scale * src[col] * mul[mul_col] + add[add_col];
            if constexpr (ApplyOutputScale) value *= output_scale;
            dst[col] = value;
        } else {
            dst[col] = scale * src[col] * mul[mul_col];
        }
    }
    if constexpr (QuantizeMmq) {
        __syncthreads();
        const volatile float * published = dst;
        for (uint32_t i0 = uint32_t(tid) * 4; i0 < uint32_t(ncols);
             i0 += BlockSize * 4) {
            float4 xi;
            xi.x = published[i0 + 0];
            xi.y = published[i0 + 1];
            xi.z = published[i0 + 2];
            xi.w = published[i0 + 3];
            float amax = fabsf(xi.x);
            amax = fmaxf(amax, fabsf(xi.y));
            amax = fmaxf(amax, fabsf(xi.z));
            amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                amax = fmaxf(amax,
                    __shfl_xor_sync(0xffffffff, amax, off, 32));
            }
            const float d_inv = 127.0f / amax;
            char4 quant;
            quant.x = int8_t(roundf(xi.x * d_inv));
            quant.y = int8_t(roundf(xi.y * d_inv));
            quant.z = int8_t(roundf(xi.z * d_inv));
            quant.w = int8_t(roundf(xi.w * d_inv));
            const float d = 1.0f / d_inv;
            const uint32_t block = i0 / 32;
            const uint32_t block_in_group = block % 4;
            const uint32_t iqs = i0 % 32;
            BlockQ8_1Mmq * out = q8
                + uint64_t(block / 4) * q8_n_tok + row;
            reinterpret_cast<char4 *>(
                out->qs + block_in_group * 32)[iqs / 4] = quant;
            if (iqs == 0) {
                out->d[block_in_group] = __half2float(__float2half(d));
            }
        }
    } else {
        (void)q8; (void)q8_n_tok;
    }
}

// Decode projection preparation: preserve the ggml RMS reduction and f32 output while
// producing the exact Q8_1 blocks consumed by the next Q4 MMVQ. One warp owns each
// contiguous 32-value block, matching quantize_q8_1's XOR max/sum order.
template <int BlockSize>
__global__ void k_rms_norm_q8_1_decode(
        const float * src, float * dst, const float * mul, BlockQ8_1 * q8,
        uint32_t width, float eps) {
    const uint32_t tid = threadIdx.x;
    float square_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += BlockSize) {
        const float value = src[col];
        square_sum += value * value;
    }
    extern __shared__ float shared[];
    square_sum = block_sum<BlockSize>(square_sum, shared);
    const float norm_scale = rsqrtf(square_sum / width + eps);

    const uint32_t lane = tid & 31;
    const uint32_t warp_base = (tid >> 5) * 32;
    for (uint32_t base = warp_base; base < width; base += BlockSize) {
        const uint32_t col = base + lane;
        const float value = norm_scale * src[col] * mul[col];
        dst[col] = value;

        float amax = fabsf(value);
        float sum = value;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        const float d = amax / 127.0f;
        const int8_t quant = amax == 0.0f
            ? int8_t(0) : int8_t(roundf(value / d));
        BlockQ8_1 * block = q8 + base / 32;
        block->qs[lane] = quant;
        if (lane == 0) {
            block->d = __float2half(d);
            block->s = __float2half(sum);
        }
    }
}

// Prefill projection preparation. Preserve the ggml RMS reduction and the
// four-values-per-thread / eight-lane Q8_1 quantizer contract while producing
// MMQ's K-group-major records in the same dispatch.
template <int BlockSize, bool FullPrecisionScale = false,
          bool WriteDense = true>
__global__ void k_rms_norm_q8_1_mmq(
        const float * src, float * dst, const float * mul, BlockQ8_1Mmq * q8,
        uint32_t width, uint32_t n_tok, float eps) {
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;
    const float * src_row = src + uint64_t(tok) * width;
    float * dst_row = dst + uint64_t(tok) * width;

    float square_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += BlockSize) {
        const float value = src_row[col];
        square_sum += value * value;
    }
    extern __shared__ float shared[];
    square_sum = block_sum<BlockSize>(square_sum, shared);
    const float norm_scale = rsqrtf(square_sum / width + eps);

    for (uint32_t i0 = tid * 4; i0 < width; i0 += BlockSize * 4) {
        float4 xi;
        xi.x = norm_scale * src_row[i0 + 0] * mul[i0 + 0];
        xi.y = norm_scale * src_row[i0 + 1] * mul[i0 + 1];
        xi.z = norm_scale * src_row[i0 + 2] * mul[i0 + 2];
        xi.w = norm_scale * src_row[i0 + 3] * mul[i0 + 3];
        if constexpr (WriteDense) {
            reinterpret_cast<float4 *>(dst_row)[i0 / 4] = xi;
        }

        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffff, amax, off, 32));
        }
        const float d_inv = 127.0f / amax;
        char4 quant;
        quant.x = int8_t(roundf(xi.x * d_inv));
        quant.y = int8_t(roundf(xi.y * d_inv));
        quant.z = int8_t(roundf(xi.z * d_inv));
        quant.w = int8_t(roundf(xi.w * d_inv));
        const float d = 1.0f / d_inv;
        const uint32_t block = i0 / 32;
        const uint32_t block_in_group = block % 4;
        const uint32_t iqs = i0 % 32;
        BlockQ8_1Mmq * out = q8 + uint64_t(block / 4) * n_tok + tok;
        reinterpret_cast<char4 *>(out->qs + block_in_group * 32)[iqs / 4] = quant;
        if (iqs == 0) {
            if constexpr (FullPrecisionScale) {
                out->d[block_in_group] = d;
            } else {
                out->d[block_in_group] = __half2float(__float2half(d));
            }
        }
    }
}

// LFM-style pre-norm boundary: update the residual and produce the normalized
// dense row plus MMQ D4 activation in one dispatch. The reduction observes the
// stored f32 sum, matching a separate add followed by RMSNorm.
template <int BlockSize>
__global__ void k_add_rms_norm_q8_1_mmq(
        float * resid, const float * other, float * dst, const float * mul,
        BlockQ8_1Mmq * q8, uint32_t width, uint32_t n_tok, float eps) {
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;
    float * resid_row = resid + uint64_t(tok) * width;
    const float * other_row = other + uint64_t(tok) * width;
    float * dst_row = dst + uint64_t(tok) * width;

    float square_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += BlockSize) {
        const float value = resid_row[col] + other_row[col];
        resid_row[col] = value;
        square_sum += value * value;
    }
    extern __shared__ float shared[];
    square_sum = block_sum<BlockSize>(square_sum, shared);
    const float norm_scale = rsqrtf(square_sum / width + eps);

    for (uint32_t i0 = tid * 4; i0 < width; i0 += BlockSize * 4) {
        float4 value;
        value.x = norm_scale * resid_row[i0 + 0] * mul[i0 + 0];
        value.y = norm_scale * resid_row[i0 + 1] * mul[i0 + 1];
        value.z = norm_scale * resid_row[i0 + 2] * mul[i0 + 2];
        value.w = norm_scale * resid_row[i0 + 3] * mul[i0 + 3];
        reinterpret_cast<float4 *>(dst_row)[i0 / 4] = value;

        float amax = fabsf(value.x);
        amax = fmaxf(amax, fabsf(value.y));
        amax = fmaxf(amax, fabsf(value.z));
        amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(
                amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
        }
        const float d_inv = 127.0f / amax;
        char4 quant;
        quant.x = int8_t(roundf(value.x * d_inv));
        quant.y = int8_t(roundf(value.y * d_inv));
        quant.z = int8_t(roundf(value.z * d_inv));
        quant.w = int8_t(roundf(value.w * d_inv));
        const float d = 1.0f / d_inv;
        const uint32_t block = i0 / 32;
        const uint32_t block_in_group = block % 4;
        const uint32_t iqs = i0 % 32;
        BlockQ8_1Mmq * out = q8 + uint64_t(block / 4) * n_tok + tok;
        reinterpret_cast<char4 *>(out->qs + block_in_group * 32)[iqs / 4] = quant;
        if (iqs == 0) {
            out->d[block_in_group] = __half2float(__float2half(d));
        }
    }
}

// Width-2048 pre-norm fast path. One 512-thread block owns one contiguous
// float4 per thread across both phases, so the residual values stay in
// registers across the reduction instead of being reloaded for normalization.
__global__ void k_add_rms_norm_q8_1_mmq_vec4_512(
        float * resid, const float * other, float * dst, const float * mul,
        BlockQ8_1Mmq * q8, uint32_t n_tok, float eps) {
    constexpr uint32_t Width = 2048;
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;
    const uint32_t i0 = tid * 4;
    float * resid_row = resid + uint64_t(tok) * Width;
    const float * other_row = other + uint64_t(tok) * Width;
    float * dst_row = dst + uint64_t(tok) * Width;
    float4 value = reinterpret_cast<const float4 *>(resid_row)[tid];
    const float4 add = reinterpret_cast<const float4 *>(other_row)[tid];
    value.x += add.x;
    value.y += add.y;
    value.z += add.z;
    value.w += add.w;
    reinterpret_cast<float4 *>(resid_row)[tid] = value;

    float square_sum = value.x * value.x + value.y * value.y
        + value.z * value.z + value.w * value.w;
    extern __shared__ float shared[];
    square_sum = block_sum<512>(square_sum, shared);
    const float norm_scale = rsqrtf(square_sum / Width + eps);
    const float4 scale = reinterpret_cast<const float4 *>(mul)[tid];
    value.x *= norm_scale * scale.x;
    value.y *= norm_scale * scale.y;
    value.z *= norm_scale * scale.z;
    value.w *= norm_scale * scale.w;
    reinterpret_cast<float4 *>(dst_row)[tid] = value;

    float amax = fabsf(value.x);
    amax = fmaxf(amax, fabsf(value.y));
    amax = fmaxf(amax, fabsf(value.z));
    amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax,
            __shfl_xor_sync(0xffffffffu, amax, off, 32));
    }
    const float d_inv = 127.0f / amax;
    char4 quant;
    quant.x = int8_t(roundf(value.x * d_inv));
    quant.y = int8_t(roundf(value.y * d_inv));
    quant.z = int8_t(roundf(value.z * d_inv));
    quant.w = int8_t(roundf(value.w * d_inv));
    const float d = 1.0f / d_inv;
    const uint32_t block = i0 / 32;
    const uint32_t block_in_group = block % 4;
    const uint32_t iqs = i0 % 32;
    BlockQ8_1Mmq * out = q8 + uint64_t(block / 4) * n_tok + tok;
    reinterpret_cast<char4 *>(out->qs + block_in_group * 32)[iqs / 4] = quant;
    if (iqs == 0) {
        out->d[block_in_group] = __half2float(__float2half(d));
    }
}

// Laboratory producer for the SM86 MMA-ready activation layout. It preserves
// the established RMS reduction and Q8 subgroup arithmetic while publishing
// the final K-major consumer layout directly, eliminating a second full read
// of the normalized activation.
template <int BlockSize>
__global__ void k_rms_norm_q8_mma_ready(
        const float * src, float * dst, const float * mul,
        uint16_t * quant_u16, float * d8_sideplane,
        uint32_t width, uint32_t n_tok, uint32_t token_tiles, float eps) {
    using namespace imparo_q8_mma_ready_a0_v1_authority_lab;
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;
    const float * src_row = src + uint64_t(tok) * width;
    float * dst_row = dst + uint64_t(tok) * width;
    float square_sum = 0.0f;
    for (uint32_t col = tid; col < width; col += BlockSize) {
        const float value = src_row[col];
        square_sum += value * value;
    }
    extern __shared__ float shared[];
    square_sum = block_sum<BlockSize>(square_sum, shared);
    const float norm_scale = rsqrtf(square_sum / width + eps);
    const uint32_t token_tile = tok / kTokenTile;
    const uint32_t token_in_tile = tok % kTokenTile;

    for (uint32_t i0 = tid * 4; i0 < width; i0 += BlockSize * 4) {
        float4 value;
        value.x = norm_scale * src_row[i0 + 0] * mul[i0 + 0];
        value.y = norm_scale * src_row[i0 + 1] * mul[i0 + 1];
        value.z = norm_scale * src_row[i0 + 2] * mul[i0 + 2];
        value.w = norm_scale * src_row[i0 + 3] * mul[i0 + 3];
        reinterpret_cast<float4 *>(dst_row)[i0 / 4] = value;

        float amax = fabsf(value.x);
        amax = fmaxf(amax, fabsf(value.y));
        amax = fmaxf(amax, fabsf(value.z));
        amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            amax = fmaxf(
                amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
        }
        const float d_inv = 127.0f / amax;
        char4 quant;
        quant.x = int8_t(roundf(value.x * d_inv));
        quant.y = int8_t(roundf(value.y * d_inv));
        quant.z = int8_t(roundf(value.z * d_inv));
        quant.w = int8_t(roundf(value.w * d_inv));
        const uint32_t block = i0 / kValuesPerQBlock;
        const uint32_t group = block / kQBlocksPerGroup;
        const uint32_t qblock = block % kQBlocksPerGroup;
        const uint32_t kpair = (i0 % kValuesPerQBlock) / 2;
        quant_u16[quant_u16_index(
            token_tiles, group, token_tile, qblock, kpair,
            token_in_tile)] = uint16_t(uint8_t(quant.x))
                | (uint16_t(uint8_t(quant.y)) << 8);
        quant_u16[quant_u16_index(
            token_tiles, group, token_tile, qblock, kpair + 1,
            token_in_tile)] = uint16_t(uint8_t(quant.z))
                | (uint16_t(uint8_t(quant.w)) << 8);
        if ((i0 % kValuesPerQBlock) == 0) {
            d8_sideplane[scale_index(
                token_tiles, group, token_tile, qblock,
                token_in_tile)] =
                    __half2float(__float2half(1.0f / d_inv));
        }
    }
}

// rms_norm: one block per row, two passes (sum of squares, then scale [*w]).
// BlockSize and HasWeight are compile-time values to keep reduction and multiply
// code generation aligned with ggml-cuda's rms_norm_f32 specializations.
template <uint32_t BlockSize, bool HasWeight>
__global__ void k_rms_norm(const float * w, float * x, const float * src,
                           uint32_t width, float eps, uint32_t n_row,
                            uint32_t row_stride, uint32_t base_off) {
    const uint32_t r = blockIdx.x;
    if (r >= n_row) return;
    float * row = x + base_off + (uint64_t)r * row_stride;
    const float * srow = src + base_off + (uint64_t)r * row_stride;
    extern __shared__ float part[];
    float sq = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += BlockSize) sq += srow[i] * srow[i];
    sq = block_sum<BlockSize>(sq, part);
    const float mean = sq / width;
    const float scale = rsqrtf(mean + eps);
    for (uint32_t i = threadIdx.x; i < width; i += BlockSize) {
        if constexpr (HasWeight) row[i] = scale * srow[i] * w[i];
        else row[i] = scale * srow[i];
    }
}

// rope, neox pairing: dim i rotates with i + n_rot/2; positions offset by start_pos.
// freqs == nullptr means theta = base^(-2i/n_rot) (the workflow uploads
// rope_freqs.weight when the model carries it).
__global__ void k_rope(float * x, const float * freqs, uint32_t n_rot, float theta_scale,
                       uint32_t head_dim, uint32_t n_heads, uint32_t start_pos,
                       uint32_t n_tok, const uint32_t * decode_control) {
    if (decode_control) start_pos = decode_control[0];
    const uint32_t half_rot = n_rot / 2;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_tok = n_heads * half_rot;
    if (idx >= n_tok * per_tok) return;
    const uint32_t t = idx / per_tok;
    const uint32_t h = (idx % per_tok) / half_rot;
    const uint32_t i = idx % half_rot;
    // Match llama.cpp's CUDA path exactly: compute one theta_scale on the host,
    // exponentiate that scale in the kernel, multiply by position, then divide by
    // the optional frequency factor.  The algebraically equivalent single powf
    // below used to differ after a few dimensions and moved long-prefill logits.
    float a = float(start_pos + t) * powf(theta_scale, float(i));
    if (freqs) a /= freqs[i];
    const float c = cosf(a), s = sinf(a);
    float * hd = x + ((uint64_t)t * n_heads + h) * head_dim;
    const float x0 = hd[i], x1 = hd[i + half_rot];
    // Preserve llama.cpp's CUDA contraction order. nvcc otherwise contracts the
    // commutative second sum around x0*s on this translation unit, while the pinned
    // kernel contracts x1*c; the resulting ULPs are visible from position one.
    hd[i] = x0 * c - x1 * s;
    hd[i + half_rot] = fmaf(x1, c, x0 * s);
}

// FWHT, natural (Sylvester) order -- the quantized-KV rotation. Matches
// imparo_hadamard64's butterfly and its fixed pair order.
__global__ void k_hadamard(float * x, uint32_t n, uint32_t nrot, float scale) {
    extern __shared__ float blk[];
    const uint32_t b0 = blockIdx.x * nrot;
    if (b0 + nrot > n) return;
    // Match ggml-cuda FWHT's rounding boundary: normalize each input before the
    // butterflies, not the final output. The transforms are algebraically equal,
    // but post-scaling moves values across Q4/Q8 quantization bins.
    for (uint32_t i = threadIdx.x; i < nrot; i += blockDim.x) {
        blk[i] = x[b0 + i] * scale;
    }
    __syncthreads();
    for (uint32_t stride = 1; stride < nrot; stride <<= 1) {
        for (uint32_t p = threadIdx.x; p < nrot / 2; p += blockDim.x) {
            const uint32_t i = ((p & ~(stride - 1)) << 1) | (p & (stride - 1));
            const float a = blk[i], b = blk[i | stride];
            blk[i] = a + b;
            blk[i | stride] = a - b;
        }
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < nrot; i += blockDim.x) x[b0 + i] = blk[i];
}

// A 64-point FWHT maps exactly to one warp when every lane owns the matching
// elements in both 32-value halves. Strides 1..16 remain shuffle butterflies;
// the final stride-32 pair is already resident in the same lane. This preserves
// the generic kernel's scale-before-transform and a+b/a-b order without shared
// memory or six block-wide barriers. Eight independent transforms share a CTA
// to avoid one-warp block scheduling overhead.
template <bool Quantize>
__global__ void k_hadamard64_warp(
        float * x, BlockQ8_1Mmq * q8, uint32_t width, uint32_t n_row,
        uint32_t blocks, float scale) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t block = blockIdx.x * blockDim.y + warp;
    if (block >= blocks) return;
    float * row = x + uint64_t(block) * 64;
    float lo = row[lane] * scale;
    float hi = row[32 + lane] * scale;
#pragma unroll
    for (uint32_t stride = 1; stride < 32; stride <<= 1) {
        const float lo_pair = __shfl_xor_sync(0xffffffff, lo, stride);
        const float hi_pair = __shfl_xor_sync(0xffffffff, hi, stride);
        if (lane & stride) {
            lo = lo_pair - lo;
            hi = hi_pair - hi;
        } else {
            lo += lo_pair;
            hi += hi_pair;
        }
    }
    const float final_lo = lo + hi;
    const float final_hi = lo - hi;
    lo = final_lo;
    hi = final_hi;
    row[lane] = lo;
    row[32 + lane] = hi;
    if constexpr (Quantize) {
        const uint32_t blocks_per_row = width / 64;
        const uint32_t token = block / blocks_per_row;
        const uint32_t block64 = block % blocks_per_row;
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
            const float value = half ? hi : lo;
            float amax = fabsf(value);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax,
                    __shfl_xor_sync(0xffffffff, amax, offset));
            }
            const float d_inv = 127.0f / amax;
            const float d = 1.0f / d_inv;
            const uint32_t block32 = 2 * block64 + half;
            const uint32_t block_in_group = block32 % 4;
            BlockQ8_1Mmq * out = q8
                + uint64_t(block32 / 4) * n_row + token;
            out->qs[block_in_group * 32 + lane] =
                int8_t(roundf(value * d_inv));
            if (lane == 0) {
                out->d[block_in_group] = __half2float(__float2half(d));
            }
        }
    } else {
        (void)q8; (void)width; (void)n_row;
    }
}

template <uint32_t BlockSize, bool HasWeight>
__global__ void k_head_norm_rope_hadamard(
        float * x, const float * norm_w, const float * freqs,
        uint32_t head_dim, float eps, uint32_t n_heads,
        uint32_t start_pos, uint32_t rope_dim, float theta_scale,
        uint32_t hadamard_nrot, float hadamard_scale,
        const uint32_t * decode_control, __half * q_cache) {
    if (decode_control) start_pos = decode_control[0];
    const uint32_t row_index = blockIdx.x;
    float * global_row = x + uint64_t(row_index) * head_dim;
    extern __shared__ float row[];

    float square_sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < head_dim; col += BlockSize) {
        const float value = global_row[col];
        square_sum += value * value;
    }
    square_sum = block_sum<BlockSize>(square_sum, row);
    // block_sum uses row[0..Warps) as reduction scratch.  Its last warp-level
    // read is not a block-wide completion point, so a faster warp could start
    // materializing the normalized row below while another warp is still
    // consuming that scratch.  Close the scratch epoch before reusing the same
    // shared allocation for the full head row.
    __syncthreads();
    const float norm_scale = rsqrtf(square_sum / head_dim + eps);
    for (uint32_t col = threadIdx.x; col < head_dim; col += BlockSize) {
        if constexpr (HasWeight) {
            row[col] = norm_scale * global_row[col] * norm_w[col];
        } else {
            row[col] = norm_scale * global_row[col];
        }
    }
    __syncthreads();

    const uint32_t half_rot = rope_dim / 2;
    const uint32_t token = row_index / n_heads;
    for (uint32_t i = threadIdx.x; i < half_rot; i += BlockSize) {
        float angle = float(start_pos + token) * powf(theta_scale, float(i));
        if (freqs) angle /= freqs[i];
        const float c = cosf(angle), s = sinf(angle);
        const float x0 = row[i], x1 = row[i + half_rot];
        row[i] = x0 * c - x1 * s;
        row[i + half_rot] = fmaf(x1, c, x0 * s);
    }
    __syncthreads();

    if (hadamard_nrot != 0) {
        for (uint32_t base = 0; base < head_dim; base += hadamard_nrot) {
            for (uint32_t i = threadIdx.x; i < hadamard_nrot; i += BlockSize) {
                row[base + i] *= hadamard_scale;
            }
            __syncthreads();
            for (uint32_t stride = 1; stride < hadamard_nrot; stride <<= 1) {
                for (uint32_t p = threadIdx.x; p < hadamard_nrot / 2;
                     p += BlockSize) {
                    const uint32_t i = ((p & ~(stride - 1)) << 1)
                        | (p & (stride - 1));
                    const float a = row[base + i], b = row[base + (i | stride)];
                    row[base + i] = a + b;
                    row[base + (i | stride)] = a - b;
                }
                __syncthreads();
            }
        }
    }

    for (uint32_t col = threadIdx.x; col < head_dim; col += BlockSize) {
        const float value = row[col];
        global_row[col] = value;
        if (q_cache) q_cache[uint64_t(row_index) * head_dim + col] = __float2half(value);
    }
}

// kv_store, f16: slot = ring ? (pos & ring) : pos. One thread per value.
__global__ void k_kv_store_f16(const float * src, __half * dst, uint32_t width,
                               uint32_t start_pos, uint32_t n_tok, uint32_t ring,
                               const uint32_t * decode_control,
                               const uint32_t * page_table) {
    if (decode_control) start_pos = decode_control[0];
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= width || t >= n_tok) return;
    const uint32_t gp = start_pos + t;
    const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
    dst[(uint64_t)ps * width + i] = __float2half(src[(uint64_t)t * width + i]);
}

// kv_store, q4_0: llama.cpp's quantize_q4_0, byte-for-byte the .mm sibling. One
// thread per 32-value block.
__global__ void k_kv_store_q4(const float * src, uint8_t * dst, uint32_t width,
                              uint32_t start_pos, uint32_t n_tok, uint32_t ring,
                              const uint32_t * decode_control,
                              const uint32_t * page_table) {
    if (decode_control) start_pos = decode_control[0];
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t blocks = width / 32;
    if (b >= blocks || t >= n_tok) return;
    const float * x = src + (uint64_t)t * width + b * 32;
    const uint32_t gp = start_pos + t;
    const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
    uint8_t * blk = dst + ((uint64_t)ps * blocks + b) * 18;
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float v = x[j];
        if (fabsf(v) > amax) { amax = fabsf(v); vmax = v; }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const __half dh = __float2half(d);
    *reinterpret_cast<__half *>(blk) = dh;
    for (int j = 0; j < 16; ++j) {
        int q0 = int(x[j] * id + 8.5f);      if (q0 > 15) q0 = 15; if (q0 < 0) q0 = 0;
        int q1 = int(x[16 + j] * id + 8.5f); if (q1 > 15) q1 = 15; if (q1 < 0) q1 = 0;
        blk[2 + j] = uint8_t(q0 | (q1 << 4));
    }
}

__global__ void k_kv_store_q8(const float * src, uint8_t * dst, uint32_t width,
                              uint32_t start_pos, uint32_t n_tok, uint32_t ring,
                              const uint32_t * decode_control,
                              const uint32_t * page_table) {
    if (decode_control) start_pos = decode_control[0];
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t blocks = width / 32;
    if (b >= blocks || t >= n_tok) return;
    const float * x = src + (uint64_t)t * width + b * 32;
    const uint32_t ps = imparo_cuda_kv::physical_row(start_pos + t, ring, page_table);
    uint8_t * blk = dst + ((uint64_t)ps * blocks + b) * 34;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const __half dh = __float2half(d);
    *reinterpret_cast<__half *>(blk) = dh;
    // ggml's CUDA Q8_0 store uses roundf (ties away from zero). rintf follows
    // the active rounding mode (normally ties-to-even), which only changes rare
    // boundary values but compounds across a long quantized KV cache.
    for (int j = 0; j < 32; ++j) {
        blk[2 + j] = uint8_t(int8_t(roundf(x[j] * id)));
    }
}

// kv_dequant: whole cache slice into a half scratch, value = (q-8)*d / q*d.
#include "sm80/kv_dequant.cuh"

__global__ void k_kv_dequant(const uint8_t * src, __half * dst, uint32_t width,
                             uint32_t slots, uint32_t ktype, uint32_t ring,
                             const uint32_t * page_table) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t s = blockIdx.y;
    const uint32_t blocks = width / 32;
    if (b >= blocks || s >= slots) return;
    const uint32_t physical = imparo_cuda_kv::physical_row(s, ring, page_table);
    __half * out = dst + (uint64_t)physical * width + b * 32;
    if (ktype == 2) {
        const uint8_t * blk = src + ((uint64_t)physical * blocks + b) * 18;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        for (int i = 0; i < 32; ++i) {
            const uint8_t pk = blk[2 + (i & 15)];
            const int q = (i < 16) ? (pk & 0x0F) : (pk >> 4);
            out[i] = __float2half(fmaf(d, float(q), -8.0f * d));
        }
    } else {
        const uint8_t * blk = src + ((uint64_t)physical * blocks + b) * 34;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        for (int i = 0; i < 32; ++i) {
            out[i] = __float2half(float(int8_t(blk[2 + i])) * d);
        }
    }
}

__device__ inline float kv_value(const void * cache, uint32_t type, uint32_t width,
                                 uint32_t slot, uint32_t i) {
    if (type == 1) {
        return __half2float(static_cast<const __half *>(cache)[(uint64_t)slot * width + i]);
    }
    const uint32_t block = i / 32;
    const uint32_t j = i & 31;
    const uint32_t blocks = width / 32;
    if (type == 2) {
        const uint8_t * blk = static_cast<const uint8_t *>(cache)
            + ((uint64_t)slot * blocks + block) * 18;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const uint8_t pk = blk[2 + (j & 15)];
        const int q = j < 16 ? (pk & 0x0f) : (pk >> 4);
        return fmaf(d, float(q), -8.0f * d);
    }
    const uint8_t * blk = static_cast<const uint8_t *>(cache)
        + ((uint64_t)slot * blocks + block) * 34;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    return float(int8_t(blk[2 + j])) * d;
}

// Attention, flash-style: ONE BLOCK per (head, query). Online softmax over position
// chunks, accumulator in registers/shared. The algorithm (bounds, masking, the
// dead-row rule, the causal window `[max(0, pos+1-window), pos]`) matches
// attention_qcomb_body; the parallel decomposition is deliberately simpler --
// per-query instead of 8-query tiles -- because correctness comes first and the
// CUDA tile geometry is a registry knob for real hardware.
__global__ void k_attention(const float * q, const void * kc, const void * vc,
                            float * out, uint32_t head_dim, uint32_t n_heads,
                            uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
                            float qk_scale, uint32_t window, uint32_t ring, uint32_t n_tok,
                            uint32_t ktype, uint32_t vtype, uint32_t f32_v_accum,
                            const uint32_t * page_table) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    const uint32_t kvh = h / (n_heads / n_kv);
    const uint32_t pos = start_pos + t;
    const uint32_t lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    extern __shared__ float sh[];              // head_dim accumulator + reductions
    float * acc = sh;                          // [head_dim]
    const float * qr = q + ((uint64_t)t * n_heads + h) * head_dim;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) acc[i] = 0.0f;
    __shared__ float m_run, s_run;
    // SM70+ QK probe tile.  llama.cpp's FA path feeds half Q/K into an f32 MMA
    // accumulator; preserving that contract matters because score-(score+offset)
    // retains the tensor-core accumulator's low bits.  A scalar f32 dot is close in
    // real numbers but crosses Q8_1 boundaries in the following WO projection.
    __shared__ __align__(16) __half mma_q[16 * 16];
    __shared__ __align__(16) __half mma_k[16 * 16];
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
    __shared__ __align__(16) float mma_c[16 * 16];
#endif
    __shared__ float mma_score;
    if (threadIdx.x == 0) { m_run = -1e30f; s_run = 0.0f; }
    __syncthreads();
    for (uint32_t gp = lo; gp <= pos; ++gp) {
        const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
        __shared__ float warp_dot[32];
        if (ktype == 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if (threadIdx.x < 32) {
                const uint32_t gqa = n_heads / n_kv;
                // Preserve the same register slot as a 16-query x GQA tiled FA:
                // mathematically equal output slots can differ by a few accumulator ULPs.
                const uint32_t q_row = (t * gqa + h % gqa) & 15;
                const uint32_t k_row = gp & 15;
                const __half * kr = static_cast<const __half *>(kc)
                    + uint64_t(ps) * kv_width + kvh * head_dim;
                const float dot = imparo_sm80_mma::dot_f16_fragment(
                    qr, kr, head_dim, qk_scale, q_row, k_row,
                    reinterpret_cast<__half2 *>(mma_q),
                    reinterpret_cast<__half2 *>(mma_k), threadIdx.x);
                if (threadIdx.x == 0) mma_score = dot;
            }
            __syncthreads();
            warp_dot[0] = mma_score;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            if (threadIdx.x < 32) {
                using namespace nvcuda;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                for (uint32_t k0 = 0; k0 < head_dim; k0 += 16) {
                    for (uint32_t j = threadIdx.x; j < 16 * 16; j += 32) {
                        mma_q[j] = j < 16 && k0 + j < head_dim
                            ? __hmul(__float2half(qr[k0 + j]), __float2half(qk_scale))
                            : __float2half(0.0f);
                        mma_k[j] = j < 16 && k0 + j < head_dim
                            ? static_cast<const __half *>(kc)[
                                  (uint64_t)ps * kv_width + kvh * head_dim + k0 + j]
                            : __float2half(0.0f);
                    }
                    __syncwarp();
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                   wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                   wmma::col_major> b;
                    wmma::load_matrix_sync(a, mma_q, 16);
                    wmma::load_matrix_sync(b, mma_k, 16);
                    wmma::mma_sync(c, a, b, c);
                    __syncwarp();
                }
                wmma::store_matrix_sync(mma_c, c, 16, wmma::mem_row_major);
                __syncwarp();
                if (threadIdx.x == 0) mma_score = mma_c[0];
            }
            __syncthreads();
            warp_dot[0] = mma_score;
#else
            if (threadIdx.x == 0) warp_dot[0] = 0.0f;
#endif
        } else {
            float dot = 0.0f;
            for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
                dot += qr[i] * kv_value(kc, ktype, kv_width, ps, kvh * head_dim + i);
            }
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, off);
            if ((threadIdx.x & 31) == 0) warp_dot[threadIdx.x >> 5] = dot;
            __syncthreads();
            if (threadIdx.x == 0) {
                float d = 0.0f;
                for (uint32_t wi = 0; wi < (blockDim.x + 31) / 32; ++wi) d += warp_dot[wi];
                warp_dot[0] = d;
            }
        }
        __syncthreads();
        const float score = warp_dot[0];
        // Match llama.cpp's f16-MMA Flash Attention numerical contract. Its
        // max is deliberately shifted by 3*log(2), extending the dynamic range
        // of the half VKQ accumulator. The denominator stays f32 while the
        // probabilities and running numerator are f16 MMA operands/accumulators.
        // Omitting either detail is mathematically equivalent in real numbers,
        // but crosses Q8_1 activation boundaries in the following projection.
        constexpr float kFaMaxOffset = 3.0f * 0.6931f;
        const float m_new = fmaxf(m_run, score + kFaMaxOffset);
        const float scale = expf(m_run - m_new);
        const float p = expf(score - m_new);
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float vv = kv_value(vc, vtype, kv_width, ps, kvh * head_dim + i);
            if (vtype == 1 && !f32_v_accum) {
                const __half ah = __float2half(acc[i]);
                const __half sh = __float2half(scale);
                const __half ph = __float2half(p);
                const __half vh = __float2half(vv);
                acc[i] = __half2float(__hfma(ph, vh, __hmul(ah, sh)));
            } else {
                acc[i] = acc[i] * scale + p * vv;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) { s_run = s_run * scale + p; m_run = m_new; }
        __syncthreads();
    }
    float * op = out + ((uint64_t)t * n_heads + h) * head_dim;
    const float inv = s_run > 0.0f ? 1.0f / s_run : 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) op[i] = acc[i] * inv;
}

// Graph-only twin of the conservative attention kernel. Keep `k_attention` itself
// untouched: its SM86 numerical code generation is already gated. A stable device
// control word replaces only the captured by-value start position on replay.
__global__ void k_attention_d64_controlled(
        const float * q, const void * kc, const void * vc,
        float * out, uint32_t head_dim, uint32_t n_heads,
        uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
        float qk_scale, uint32_t window, uint32_t ring, uint32_t n_tok,
        uint32_t ktype, uint32_t vtype, uint32_t f32_v_accum,
        const uint32_t * page_table, const uint32_t * control) {
    if (control) start_pos = *control;
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    const uint32_t kvh = h / (n_heads / n_kv);
    const uint32_t pos = start_pos + t;
    const uint32_t lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    extern __shared__ float sh[];
    float * acc = sh;
    const float * qr = q + ((uint64_t)t * n_heads + h) * head_dim;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) acc[i] = 0.0f;
    __shared__ float m_run, s_run;
    __shared__ __align__(16) __half mma_q[16 * 16];
    __shared__ __align__(16) __half mma_k[16 * 16];
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
    __shared__ __align__(16) float mma_c[16 * 16];
#endif
    __shared__ float mma_score;
    if (threadIdx.x == 0) { m_run = -1e30f; s_run = 0.0f; }
    __syncthreads();
    for (uint32_t gp = lo; gp <= pos; ++gp) {
        const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
        __shared__ float warp_dot[32];
        if (ktype == 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if (threadIdx.x < 32) {
                const uint32_t gqa = n_heads / n_kv;
                const uint32_t q_row = (t * gqa + h % gqa) & 15;
                const uint32_t k_row = gp & 15;
                const __half * kr = static_cast<const __half *>(kc)
                    + uint64_t(ps) * kv_width + kvh * head_dim;
                const float dot = imparo_sm80_mma::dot_f16_fragment(
                    qr, kr, head_dim, qk_scale, q_row, k_row,
                    reinterpret_cast<__half2 *>(mma_q),
                    reinterpret_cast<__half2 *>(mma_k), threadIdx.x);
                if (threadIdx.x == 0) mma_score = dot;
            }
            __syncthreads();
            warp_dot[0] = mma_score;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            if (threadIdx.x < 32) {
                using namespace nvcuda;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                for (uint32_t k0 = 0; k0 < head_dim; k0 += 16) {
                    for (uint32_t j = threadIdx.x; j < 16 * 16; j += 32) {
                        mma_q[j] = j < 16 && k0 + j < head_dim
                            ? __hmul(__float2half(qr[k0 + j]), __float2half(qk_scale))
                            : __float2half(0.0f);
                        mma_k[j] = j < 16 && k0 + j < head_dim
                            ? static_cast<const __half *>(kc)[
                                  (uint64_t)ps * kv_width + kvh * head_dim + k0 + j]
                            : __float2half(0.0f);
                    }
                    __syncwarp();
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                   wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                   wmma::col_major> b;
                    wmma::load_matrix_sync(a, mma_q, 16);
                    wmma::load_matrix_sync(b, mma_k, 16);
                    wmma::mma_sync(c, a, b, c);
                    __syncwarp();
                }
                wmma::store_matrix_sync(mma_c, c, 16, wmma::mem_row_major);
                __syncwarp();
                if (threadIdx.x == 0) mma_score = mma_c[0];
            }
            __syncthreads();
            warp_dot[0] = mma_score;
#else
            if (threadIdx.x == 0) warp_dot[0] = 0.0f;
#endif
        } else {
            float dot = 0.0f;
            for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
                dot += qr[i] * kv_value(kc, ktype, kv_width, ps, kvh * head_dim + i);
            }
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, off);
            if ((threadIdx.x & 31) == 0) warp_dot[threadIdx.x >> 5] = dot;
            __syncthreads();
            if (threadIdx.x == 0) {
                float d = 0.0f;
                for (uint32_t wi = 0; wi < (blockDim.x + 31) / 32; ++wi) d += warp_dot[wi];
                warp_dot[0] = d;
            }
        }
        __syncthreads();
        const float score = warp_dot[0];
        constexpr float kFaMaxOffset = 3.0f * 0.6931f;
        const float m_new = fmaxf(m_run, score + kFaMaxOffset);
        const float scale = expf(m_run - m_new);
        const float p = expf(score - m_new);
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float vv = kv_value(vc, vtype, kv_width, ps, kvh * head_dim + i);
            if (vtype == 1 && !f32_v_accum) {
                const __half ah = __float2half(acc[i]);
                const __half sh = __float2half(scale);
                const __half ph = __float2half(p);
                const __half vh = __float2half(vv);
                acc[i] = __half2float(__hfma(ph, vh, __hmul(ah, sh)));
            } else {
                acc[i] = acc[i] * scale + p * vv;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) { s_run = s_run * scale + p; m_run = m_new; }
        __syncthreads();
    }
    float * op = out + ((uint64_t)t * n_heads + h) * head_dim;
    const float inv = s_run > 0.0f ? 1.0f / s_run : 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) op[i] = acc[i] * inv;
}
template <bool ExactF32>
__global__ void k_attention_batch32_f16(const float * q, const __half * kc,
                                         const __half * vc, float * out,
                                         uint32_t head_dim, uint32_t n_heads,
                                         uint32_t n_kv, uint32_t kv_width,
                                         uint32_t start_pos, float qk_scale, uint32_t window,
                                         uint32_t ring, uint32_t n_tok,
                                         uint32_t half_v_accum,
                                         uint32_t llama_reduce,
                                         const uint32_t * page_table) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    // ExactF32 is the receipted short-prefill specialization. NVCC can erase
    // the dormant half-value and alternative-reduction branches, while the
    // generic specialization retains the environment-controlled lab surface.
    const bool use_half_v_accum = !ExactF32 && half_v_accum != 0;
    const bool use_llama_reduce = !ExactF32 && llama_reduce != 0;
    const uint32_t kvh = h / (n_heads / n_kv);
    const uint32_t pos = start_pos + t;
    const uint32_t lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    const float * qr = q + ((uint64_t)t * n_heads + h) * head_dim;
    extern __shared__ float acc[];
    // The exact short-Prefill route assigns one independent score to each of
    // four warps.  Per-warp tiles avoid cross-warp overwrite while preserving
    // the established per-score K16 WMMA accumulation order.  The generic lab
    // specialization retains the original single-warp footprint and schedule.
    constexpr uint32_t score_warps = ExactF32 ? 4u : 1u;
    // Exact D256 has sixteen immutable K16 Q tiles. Build them once per CTA
    // instead of repeating FP32->FP16 scaling and 256 shared stores per score.
    constexpr uint32_t exact_q_tile_count = ExactF32 ? 16u : 1u;
    __shared__ __align__(16) __half mma_q[1][16 * 16];
    __shared__ __align__(16) __half exact_q[exact_q_tile_count][16 * 16];
    __shared__ __align__(16) __half mma_k[score_warps][16 * 16];
    __shared__ __align__(16) float mma_c[score_warps][16 * 16];
    __shared__ __align__(16) __half prob_tile[16 * 16];
    __shared__ __align__(16) __half v_tile[8][16 * 16];
    __shared__ __align__(16) __half out_tile[8][16 * 16];
    __shared__ float scores[32];
    __shared__ uint32_t physical_rows[ExactF32 ? 32u : 1u];
    __shared__ volatile uint32_t exact_k_rounds;
    __shared__ float m_run;
    __shared__ float s_run;
    __shared__ float m_part[4];
    __shared__ float s_part[4];
    __shared__ float group_scale;

    using namespace nvcuda;
    // Each warp covers 16 output elements per accumulator round. E4B's pinned
    // SM86 D=256 geometry is four warps by four rounds.
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> v_acc[4];
    const uint32_t nwarps = blockDim.x / 32;
    const uint32_t v_tiles = head_dim / (nwarps * 16);
    if constexpr (ExactF32) {
        if (threadIdx.x == 0) exact_k_rounds = head_dim / 16u;
        // The selector proves head_dim=256 and blockDim=128. Each cached tile
        // has Q in row zero and zeros elsewhere, exactly matching the previous
        // per-score WMMA A matrix.
        const __half qk_scale_h = __float2half(qk_scale);
        for (uint32_t e = threadIdx.x;
             e < exact_q_tile_count * 16u * 16u; e += blockDim.x) {
            const uint32_t tile = e / (16u * 16u);
            const uint32_t cell = e % (16u * 16u);
            exact_q[tile][cell] = cell < 16u
                ? __hmul(__float2half(qr[tile * 16u + cell]), qk_scale_h)
                : __float2half(0.0f);
        }
        // Only the first K column changes between K16 rounds. The other 240
        // cells are invariant zeros, so initialize them once for each warp.
        for (uint32_t e = threadIdx.x;
             e < score_warps * 16u * 16u; e += blockDim.x) {
            const uint32_t cell = e % (16u * 16u);
            if (cell >= 16u) mma_k[e / (16u * 16u)][cell] = __float2half(0.0f);
        }
    }
    if (use_half_v_accum) {
        for (uint32_t tile = 0; tile < v_tiles; ++tile) {
            wmma::fill_fragment(v_acc[tile], 0.0f);
        }
    }
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) acc[i] = 0.0f;
    if (!use_llama_reduce && threadIdx.x == 0) {
        m_run = -3.402823466e+38F;
        s_run = 0.0f;
    }
    if (use_llama_reduce && threadIdx.x < 4) {
        m_part[threadIdx.x] = -3.402823466e+38F;
        s_part[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    // A volatile shared hand-off keeps ptxas from cloning all sixteen K16
    // rounds into the instruction cache while preserving their exact order.
    const uint32_t k_rounds = ExactF32 ? exact_k_rounds : 0u;

    constexpr uint32_t score_batch = 32u;
    const uint32_t group0 = (lo / score_batch) * score_batch;
    for (uint32_t gb = group0; gb <= pos; gb += score_batch) {
        if constexpr (ExactF32) {
            if (threadIdx.x < score_batch) {
                const uint32_t gp = gb + threadIdx.x;
                physical_rows[threadIdx.x] = gp >= lo && gp <= pos
                    ? imparo_cuda_kv::physical_row(gp, ring, page_table)
                    : 0u;
            }
            __syncthreads();
            const uint32_t warp = threadIdx.x >> 5;
            const uint32_t lane = threadIdx.x & 31;
            // Four scores advance together.  Each warp still performs the same
            // sixteen K16 MMA operations for its score, so this changes only
            // independent work scheduling, not arithmetic or reduction order.
            for (uint32_t j0 = 0; j0 < score_batch; j0 += score_warps) {
                const uint32_t j = j0 + warp;
                const uint32_t gp = gb + j;
                if (gp >= lo && gp <= pos) {
                    const uint32_t ps = physical_rows[j];
                    using namespace nvcuda;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                    wmma::fill_fragment(c, 0.0f);
                    for (uint32_t k_round = 0; k_round < k_rounds; ++k_round) {
                        const uint32_t k0 = k_round * 16u;
                        if (lane < 16u) {
                            mma_k[warp][lane] = kc[(uint64_t)ps * kv_width
                                + kvh * head_dim + k0 + lane];
                        }
                        __syncwarp();
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                       wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                       wmma::col_major> b;
                        wmma::load_matrix_sync(a, exact_q[k0 / 16u], 16);
                        wmma::load_matrix_sync(b, mma_k[warp], 16);
                        wmma::mma_sync(c, a, b, c);
                        __syncwarp();
                    }
                    wmma::store_matrix_sync(
                        mma_c[warp], c, 16, wmma::mem_row_major);
                    __syncwarp();
                    if (lane == 0) scores[j] = mma_c[warp][0];
                } else if (lane == 0) {
                    scores[j] = -3.402823466e+38F;
                }
                __syncthreads();
            }
        } else {
            for (uint32_t j = 0; j < score_batch; ++j) {
                const uint32_t gp = gb + j;
                if (gp < lo || gp > pos) {
                    if (threadIdx.x == 0) scores[j] = -3.402823466e+38F;
                    __syncthreads();
                    continue;
                }
                const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
                if (threadIdx.x < 32) {
                    using namespace nvcuda;
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                    wmma::fill_fragment(c, 0.0f);
                    for (uint32_t k0 = 0; k0 < head_dim; k0 += 16) {
                        for (uint32_t e = threadIdx.x; e < 16 * 16; e += 32) {
                            mma_q[0][e] = e < 16
                                ? __hmul(__float2half(qr[k0 + e]), __float2half(qk_scale))
                                : __float2half(0.0f);
                            mma_k[0][e] = e < 16
                                ? kc[(uint64_t)ps * kv_width + kvh * head_dim + k0 + e]
                                : __float2half(0.0f);
                        }
                        __syncwarp();
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                       wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                                       wmma::col_major> b;
                        wmma::load_matrix_sync(a, mma_q[0], 16);
                        wmma::load_matrix_sync(b, mma_k[0], 16);
                        wmma::mma_sync(c, a, b, c);
                        __syncwarp();
                    }
                    wmma::store_matrix_sync(mma_c[0], c, 16, wmma::mem_row_major);
                    __syncwarp();
                    if (threadIdx.x == 0) scores[j] = mma_c[0][0];
                }
                __syncthreads();
            }
        }

        if (use_llama_reduce && threadIdx.x < 4) {
            // The pinned SM86 reference selects its MMA-F16 kernel (ncols1=16,
            // ncols2=4, nbatch=32). For one query column, a 16x16 accumulator
            // fragment distributes each 16-row half-batch over four lanes. Each
            // lane owns the rows below in this exact fragment order, and the max
            // and final rowsum only fold across that four-lane group (xor 2, 1).
            const uint32_t lane = threadIdx.x;
            float m_new = m_part[lane];
            constexpr float kFaMaxOffset = 3.0f * 0.6931f;
            for (uint32_t block16 = 0; block16 < 32; block16 += 16) {
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t row = block16 + half8 + 2 * lane + pair;
                        if (gb + row >= lo && gb + row <= pos) {
                            m_new = fmaxf(m_new, scores[row] + kFaMaxOffset);
                        }
                    }
                }
            }
#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                m_new = fmaxf(m_new, __shfl_xor_sync(0x0000000f, m_new, offset));
            }
            const float max_diff = m_part[lane] - m_new;
            float scale = expf(max_diff);
            if (max_diff < -20.0f) scale = 0.0f;
            float add = 0.0f;
            for (uint32_t block16 = 0; block16 < 32; block16 += 16) {
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t row = block16 + half8 + 2 * lane + pair;
                        if (gb + row >= lo && gb + row <= pos) {
                            scores[row] = expf(scores[row] - m_new);
                            add += scores[row];
                        } else {
                            scores[row] = 0.0f;
                        }
                    }
                }
            }
            s_part[lane] = s_part[lane] * scale + add;
            m_part[lane] = m_new;
            if (lane == 0) group_scale = scale;
        } else if (!use_llama_reduce && threadIdx.x == 0) {
            constexpr float kFaMaxOffset = 3.0f * 0.6931f;
            float m_new = m_run;
            for (uint32_t j = 0; j < 32; ++j) {
                m_new = fmaxf(m_new, scores[j] + kFaMaxOffset);
            }
            group_scale = expf(m_run - m_new);
            float p_sum = 0.0f;
            for (uint32_t j = 0; j < 32; ++j) {
                const float p = expf(scores[j] - m_new);
                scores[j] = p;
                p_sum += p;
            }
            s_run = s_run * group_scale + p_sum;
            m_run = m_new;
        }
        __syncthreads();

        if (use_half_v_accum) {
            const __half hs = __float2half(group_scale);
            for (uint32_t tile = 0; tile < v_tiles; ++tile) {
                for (int e = 0; e < v_acc[tile].num_elements; ++e) {
                    v_acc[tile].x[e] = __hmul(v_acc[tile].x[e], hs);
                }
            }
            const uint32_t warp = threadIdx.x >> 5;
            const uint32_t lane = threadIdx.x & 31;
            for (uint32_t sub = 0; sub < score_batch; sub += 16) {
                for (uint32_t e = threadIdx.x; e < 16 * 16; e += blockDim.x) {
                    prob_tile[e] = __float2half(scores[sub + (e & 15)]);
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::col_major> b;
                wmma::load_matrix_sync(b, prob_tile, 16);
                for (uint32_t local = 0; local < v_tiles; ++local) {
                    const uint32_t out_block = warp + nwarps * local;
                    for (uint32_t e = lane; e < 16 * 16; e += 32) {
                        const uint32_t row = e >> 4;
                        const uint32_t k = e & 15;
                        const uint32_t gp = gb + sub + k;
                        __half vv = __float2half(0.0f);
                        if (gp >= lo && gp <= pos) {
                            const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
                            vv = vc[(uint64_t)ps * kv_width + kvh * head_dim +
                                    out_block * 16 + row];
                        }
                        v_tile[warp][e] = vv;
                    }
                    __syncwarp();
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                   wmma::row_major> a;
                    wmma::load_matrix_sync(a, v_tile[warp], 16);
                    wmma::mma_sync(v_acc[local], a, b, v_acc[local]);
                    __syncwarp();
                }
                __syncthreads();
            }
        } else {
            for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
                float v = acc[i] * group_scale;
                for (uint32_t j = 0; j < score_batch; ++j) {
                    const uint32_t gp = gb + j;
                    if (gp >= lo && gp <= pos) {
                        uint32_t ps;
                        if constexpr (ExactF32) {
                            ps = physical_rows[j];
                        } else {
                            ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
                        }
                        v += scores[j] * __half2float(
                            vc[(uint64_t)ps * kv_width + kvh * head_dim + i]);
                    }
                }
                acc[i] = v;
            }
        }
        __syncthreads();
    }

    float * op = out + ((uint64_t)t * n_heads + h) * head_dim;
    if (use_llama_reduce && threadIdx.x < 4) {
        float sum = s_part[threadIdx.x];
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0x0000000f, sum, offset);
        }
        if (threadIdx.x == 0) s_run = sum;
    }
    __syncthreads();
    if (use_half_v_accum) {
        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t lane = threadIdx.x & 31;
        for (uint32_t local = 0; local < v_tiles; ++local) {
            const uint32_t out_block = warp + nwarps * local;
            wmma::store_matrix_sync(out_tile[warp], v_acc[local], 16,
                                    wmma::mem_row_major);
            __syncwarp();
            if (lane < 16) {
                op[out_block * 16 + lane] = s_run > 0.0f
                    ? __half2float(out_tile[warp][lane * 16]) / s_run : 0.0f;
            }
            __syncwarp();
        }
    } else {
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
            op[i] = s_run > 0.0f ? acc[i] / s_run : 0.0f;
        }
    }
}

#include "sm80/attention_streamk_f16.cuh"
#include "sm80/attention_prefill_d512_f16.cuh"
#include "sm80/attention_flash_d256_f16.cuh"
#include "sm80/attention_flash_d512_f16.cuh"
#include "sm80/attention_prefill_d512_small_f16.cuh"
#include "sm80/attention_prefill_mma_d64_plan.h"
#include "sm80/attention_prefill_mma_d64_f16.cuh"
#include "sm80/attention_prefill_d64_wide_f16.cuh"
#include "sm86/attention_prefill_mma_d64_profile.h"
#include "sm80/attention_decode_mma_d256_f16.cuh"
#include "sm80/attention_decode_mma_d512_f16.cuh"
#include "sm86/attention_decode_mma_d512_q4.cuh"
#include "sm80/attention_decode_vec_d64_q4.cuh"
#include "sm80/attention_decode_vec_d64_q8.cuh"
#include "sm86/attention_decode_vec_d64_q4_profile.h"
#include "sm80/attention_decode_vec_d256.cuh"

// ---- elementwise family (contracts identical to the .mm kernels) ------------------
__global__ void k_gelu(float * a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = cuda_gelu(a[i]);
}
__global__ void k_gelu_mul(float * a, const float * b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    a[i] = cuda_gelu(x) * b[i];
}
__global__ void k_add(float * a, const float * b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
__global__ void k_add_scale(float * a, const float * b, float k, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (a[i] + b[i]) * k;
}
__global__ void k_scale(float * a, float k, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= k;
}
__global__ void k_copy(float * dst, const float * src, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void k_mul_strided(float * a, const float * b, uint32_t n, uint32_t b_off,
                              uint32_t b_stride, uint32_t a_stride, uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= n || t >= n_tok) return;
    a[(uint64_t)t * a_stride + i] *= b[b_off + (uint64_t)t * b_stride + i];
}

__global__ void k_softcap(float * a, float cap, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = cap * tanhf(a[i] / cap);
}
// Argmax with the smallest-index-at-max rule. Large vocabulary rows use a wider
// single CTA so each lane scans fewer values; the reduction contract is unchanged.
template <uint32_t Threads>
__launch_bounds__(Threads, 1)
__global__ void k_argmax_rows(const float * src, uint32_t * dst,
                              uint32_t n, uint32_t rows) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    src += uint64_t(row) * n;
    __shared__ float bv[Threads];
    __shared__ uint32_t bi[Threads];
    float best = -1e30f; uint32_t besti = 0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = src[i];
        if (v > best || (v == best && i < besti)) { best = v; besti = i; }
    }
    bv[threadIdx.x] = best; bi[threadIdx.x] = besti;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (bv[threadIdx.x + s] > bv[threadIdx.x]
                || (bv[threadIdx.x + s] == bv[threadIdx.x]
                    && bi[threadIdx.x + s] < bi[threadIdx.x])) {
                bv[threadIdx.x] = bv[threadIdx.x + s];
                bi[threadIdx.x] = bi[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) dst[row] = bi[0];
}
// embedding row: dst[dst_off .. dst_off+width) = dequant(weights row) * scale.
__global__ void k_row_q4(const uint8_t * row, float * dst, uint32_t width,
                         float scale, uint32_t dst_off) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const uint8_t * blk = row + (uint64_t)(i / 32) * 18;
    dst[dst_off + i] = q4_value(blk, i & 31) * scale;
}

__global__ void k_rows_q4(const uint8_t * table, const uint32_t * tokens,
                          float * dst, uint32_t width, float scale,
                          uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = blockIdx.y;
    if (i >= width || token >= n_tok) return;
    const uint64_t row_bytes = uint64_t(width / 32) * 18;
    const uint8_t * row = table + uint64_t(tokens[token]) * row_bytes;
    const uint8_t * blk = row + uint64_t(i / 32) * 18;
    dst[uint64_t(token) * width + i] = q4_value(blk, i & 31) * scale;
}

__global__ void k_rows_q4_packed(const uint8_t * rows, float * dst,
                                 uint32_t width, float scale,
                                 uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = blockIdx.y;
    if (i >= width || token >= n_tok) return;
    const uint64_t row_bytes = uint64_t(width / 32) * 18;
    const uint8_t * row = rows + uint64_t(token) * row_bytes;
    const uint8_t * blk = row + uint64_t(i / 32) * 18;
    dst[uint64_t(token) * width + i] = q4_value(blk, i & 31) * scale;
}

__global__ void k_row_q8_0(const uint8_t * row, float * dst, uint32_t width,
                           float scale, uint32_t dst_off) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const uint8_t * blk = row + uint64_t(i / 32) * 34;
    dst[dst_off + i] = q8_0_value(blk, i & 31) * scale;
}

__global__ void k_rows_q8_0(const uint8_t * table, const uint32_t * tokens,
                            float * dst, uint32_t width, float scale,
                            uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = blockIdx.y;
    if (i >= width || token >= n_tok) return;
    const uint64_t row_bytes = uint64_t(width / 32) * 34;
    const uint8_t * row = table + uint64_t(tokens[token]) * row_bytes;
    const uint8_t * blk = row + uint64_t(i / 32) * 34;
    dst[uint64_t(token) * width + i] = q8_0_value(blk, i & 31) * scale;
}

__global__ void k_rows_q8_0_packed(const uint8_t * rows, float * dst,
                                   uint32_t width, float scale,
                                   uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t token = blockIdx.y;
    if (i >= width || token >= n_tok) return;
    const uint64_t row_bytes = uint64_t(width / 32) * 34;
    const uint8_t * row = rows + uint64_t(token) * row_bytes;
    const uint8_t * blk = row + uint64_t(i / 32) * 34;
    dst[uint64_t(token) * width + i] = q8_0_value(blk, i & 31) * scale;
}

// One token row of the per-layer embedding table. The host launches one row at a
// time in paged-weight mode, so arbitrary token ids never require the whole table.
__global__ void k_ple_row(float * proj, const uint8_t * row, uint32_t width,
                          float emb_scale, uint32_t token_row) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const uint8_t * blk = row + (uint64_t)(i / 32) * 18;
    const float e = q4_value(blk, i & 31) * emb_scale;
    const uint64_t out = (uint64_t)token_row * width + i;
    proj[out] += e;
}

__global__ void k_ple_gather(float * proj, const uint8_t * table,
                             const uint32_t * tokens, uint32_t width,
                             float emb_scale, uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= width || t >= n_tok) return;
    const uint64_t row_bytes = (uint64_t)(width / 32) * 18;
    const uint8_t * row = table + (uint64_t)tokens[t] * row_bytes;
    const uint8_t * blk = row + (uint64_t)(i / 32) * 18;
    const float e = q4_value(blk, i & 31) * emb_scale;
    const uint64_t out = (uint64_t)t * width + i;
    proj[out] += e;
}

__global__ void k_ple_gather_packed(float * proj, const uint8_t * rows,
                                    uint32_t width, float emb_scale,
                                    uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= width || t >= n_tok) return;
    const uint64_t row_bytes = (uint64_t)(width / 32) * 18;
    const uint8_t * row = rows + (uint64_t)t * row_bytes;
    const uint8_t * blk = row + (uint64_t)(i / 32) * 18;
    const float e = q4_value(blk, i & 31) * emb_scale;
    proj[(uint64_t)t * width + i] += e;
}

// Gemma 4 projects all per-layer inputs first, then applies RMSNorm to each
// `ple_width` row and adds a token-selected row from the quantized PLE table.
// llama.cpp's default CUDA graph fuses RMSNorm, weight multiplication, and the
// add; matching that operation boundary is both faster and numerically observable.
template <uint32_t BlockSize, bool SingleValue>
__global__ void k_ple_norm_gather_combine(
    float * proj, const float * norm_w, const uint8_t * table,
    const uint32_t * tokens, uint32_t ple_width, uint32_t n_layers,
    float input_scale, float eps, float emb_scale, float comb_scale,
    uint32_t n_tok) {
    const uint32_t layer = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (layer >= n_layers || token >= n_tok) return;
    const uint64_t row_index = uint64_t(token) * n_layers + layer;
    float * row = proj + row_index * ple_width;
    float sum = 0.0f;
    float owned_value = 0.0f;
    if constexpr (SingleValue) {
        if (threadIdx.x < ple_width) {
            owned_value = __fmul_rn(row[threadIdx.x], input_scale);
            sum = owned_value * owned_value;
        }
    } else {
        for (uint32_t col = threadIdx.x; col < ple_width; col += BlockSize) {
            // Materialize the same rounded multiply formerly produced by the
            // standalone scale kernel before RMSNorm consumed the buffer.
            const float value = __fmul_rn(row[col], input_scale);
            sum += value * value;
        }
    }
    extern __shared__ float shared[];
    sum = block_sum<BlockSize>(sum, shared);
    const float scale = rsqrtf(sum / ple_width + eps);
    const uint64_t full_width = uint64_t(ple_width) * n_layers;
    const uint64_t table_row_bytes = full_width / 32 * 18;
    const uint8_t * selected = table + uint64_t(tokens[token]) * table_row_bytes;
    const uint32_t col_end = SingleValue ? min(ple_width, BlockSize) : ple_width;
    for (uint32_t col = threadIdx.x; col < col_end; col += BlockSize) {
        const uint64_t table_col = uint64_t(layer) * ple_width + col;
        const uint8_t * blk = selected + (table_col / 32) * 18;
        const float add = q4_value(blk, uint32_t(table_col & 31)) * emb_scale;
        const float value = SingleValue
            ? owned_value : __fmul_rn(row[col], input_scale);
        const float combined = scale * value * norm_w[col] + add;
        row[col] = combined * comb_scale;
    }
}

template <uint32_t BlockSize, bool SingleValue>
__global__ void k_ple_norm_gather_combine_packed(
    float * proj, const float * norm_w, const uint8_t * rows,
    uint32_t ple_width, uint32_t n_layers, float input_scale, float eps,
    float emb_scale, float comb_scale, uint32_t n_tok) {
    const uint32_t layer = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (layer >= n_layers || token >= n_tok) return;
    const uint64_t row_index = uint64_t(token) * n_layers + layer;
    float * row = proj + row_index * ple_width;
    float sum = 0.0f;
    float owned_value = 0.0f;
    if constexpr (SingleValue) {
        if (threadIdx.x < ple_width) {
            owned_value = __fmul_rn(row[threadIdx.x], input_scale);
            sum = owned_value * owned_value;
        }
    } else {
        for (uint32_t col = threadIdx.x; col < ple_width; col += BlockSize) {
            const float value = __fmul_rn(row[col], input_scale);
            sum += value * value;
        }
    }
    extern __shared__ float shared[];
    sum = block_sum<BlockSize>(sum, shared);
    const float scale = rsqrtf(sum / ple_width + eps);
    const uint64_t table_row_bytes = uint64_t(ple_width) * n_layers / 32 * 18;
    const uint8_t * selected = rows + uint64_t(token) * table_row_bytes;
    const uint32_t col_end = SingleValue ? min(ple_width, BlockSize) : ple_width;
    for (uint32_t col = threadIdx.x; col < col_end; col += BlockSize) {
        const uint64_t table_col = uint64_t(layer) * ple_width + col;
        const uint8_t * blk = selected + (table_col / 32) * 18;
        const float add = q4_value(blk, uint32_t(table_col & 31)) * emb_scale;
        const float value = SingleValue
            ? owned_value : __fmul_rn(row[col], input_scale);
        const float combined = scale * value * norm_w[col] + add;
        row[col] = combined * comb_scale;
    }
}

template <uint32_t BlockSize>
__global__ void k_ple_norm_gather_combine_row(
    float * proj, const float * norm_w, const uint8_t * selected,
    uint32_t token, uint32_t ple_width, uint32_t n_layers,
    float eps, float emb_scale) {
    const uint32_t layer = blockIdx.x;
    if (layer >= n_layers) return;
    const uint64_t row_index = uint64_t(token) * n_layers + layer;
    float * row = proj + row_index * ple_width;
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < ple_width; col += BlockSize) {
        const float value = row[col];
        sum += value * value;
    }
    extern __shared__ float shared[];
    sum = block_sum<BlockSize>(sum, shared);
    const float scale = rsqrtf(sum / ple_width + eps);
    for (uint32_t col = threadIdx.x; col < ple_width; col += BlockSize) {
        const uint64_t table_col = uint64_t(layer) * ple_width + col;
        const uint8_t * blk = selected + (table_col / 32) * 18;
        const float add = q4_value(blk, uint32_t(table_col & 31)) * emb_scale;
        row[col] = scale * row[col] * norm_w[col] + add;
    }
}

bool graph_kernel_is(void * actual, void * expected) {
    return actual == expected;
}

// Defined by the unity-included program_pack.cu. A registered CUfunction is
// recognized by opaque handle plus copied contract metadata, never by symbol.
bool add_program_dynamic_graph_node(
    cudaGraphNode_t node, bool * matched) noexcept;
int update_program_dynamic_graph_node(
    DynamicGraphNode & dynamic, uint32_t start_pos) noexcept;

bool add_dynamic_graph_node(
        cudaGraphNode_t node, const cudaKernelNodeParams & params,
        uint32_t arg_count, uint32_t start_index,
        uint32_t valid_index, uint32_t ring_index) {
    if (!params.kernelParams
        || (start_index != UINT32_MAX && start_index >= arg_count)
        || (valid_index != UINT32_MAX && valid_index >= arg_count)
        || (ring_index != UINT32_MAX && ring_index >= arg_count)) return false;
    DynamicGraphNode dynamic;
    dynamic.node = node;
    dynamic.params = params;
    dynamic.args.assign(params.kernelParams, params.kernelParams + arg_count);
    dynamic.start_index = start_index;
    dynamic.valid_index = valid_index;
    if (ring_index != UINT32_MAX) {
        dynamic.ring = *static_cast<const uint32_t *>(params.kernelParams[ring_index]);
    }
    g.decode_graph_nodes.push_back(std::move(dynamic));
    g.decode_graph_nodes.back().params.kernelParams =
        g.decode_graph_nodes.back().args.data();
    return true;
}

constexpr uint32_t kKvStoreGraphArgCount = 8;
constexpr uint32_t kKvStoreGraphStartArg = 3;
constexpr uint32_t kKvStoreGraphPageTableArg = 7;
static_assert(kKvStoreGraphPageTableArg + 1 == kKvStoreGraphArgCount,
              "the stable page-table pointer must remain the appended store arg");
constexpr uint32_t kD512IdentityPartialGraphArgCount = 13;
constexpr uint32_t kD512PartialGraphArgCount = 14;
constexpr uint32_t kD512PartialPageTableArg = 13;
constexpr uint32_t kD512IdentityControlledGraphArgCount = 14;
constexpr uint32_t kD512ControlledGraphArgCount = 15;
constexpr uint32_t kD512ControlledPageTableArg = 14;
constexpr uint32_t kSmallIdentityScoresGraphArgCount = 16;
constexpr uint32_t kSmallScoresGraphArgCount = 17;
constexpr uint32_t kSmallScoresPageTableArg = 16;
constexpr uint32_t kSmallIdentityValuesGraphArgCount = 13;
constexpr uint32_t kSmallValuesGraphArgCount = 14;
constexpr uint32_t kSmallValuesPageTableArg = 13;
constexpr uint32_t kD256FullValuesGraphArgCount = 13;
constexpr uint32_t kD256FullValuesPageTableArg = 12;
constexpr uint32_t kD64ControlledGraphArgCount = 18;
constexpr uint32_t kD64ControlledStartArg = 8;
constexpr uint32_t kD64ControlledRingArg = 11;
constexpr uint32_t kD64ControlledPageTableArg = 16;
constexpr uint32_t kD64ControlledControlArg = 17;
constexpr uint32_t kD64Q4VecGraphArgCount = 14;
constexpr uint32_t kD64Q4VecStartArg = 7;
constexpr uint32_t kD64Q4VecRingArg = 10;
constexpr uint32_t kD64Q4VecPageTableArg = 13;
static_assert(kD64ControlledPageTableArg + 1 == kD64ControlledControlArg);
static_assert(kD64ControlledControlArg + 1 == kD64ControlledGraphArgCount);
static_assert(kD64Q4VecPageTableArg + 1 == kD64Q4VecGraphArgCount);
static_assert(kD512PartialPageTableArg + 1 == kD512PartialGraphArgCount);
static_assert(kD512ControlledPageTableArg + 1 == kD512ControlledGraphArgCount);
static_assert(kSmallScoresPageTableArg + 1 == kSmallScoresGraphArgCount);
static_assert(kSmallValuesPageTableArg + 1 == kSmallValuesGraphArgCount);
static_assert(kD256FullValuesPageTableArg + 1 == kD256FullValuesGraphArgCount);

bool configure_decode_graph_nodes() noexcept try {
    size_t count = 0;
    if (cudaGraphGetNodes(g.decode_graph, nullptr, &count) != cudaSuccess) return false;
    std::vector<cudaGraphNode_t> nodes(count);
    if (count && cudaGraphGetNodes(g.decode_graph, nodes.data(), &count) != cudaSuccess) {
        return false;
    }
    g.decode_graph_nodes.clear();
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return false;
        if (type != cudaGraphNodeTypeKernel) continue;
        // Driver-launched Program kernels require the Driver graph-parameter API.
        // Identify them before asking the Runtime API to interpret the node.
        bool program_matched = false;
        if (!add_program_dynamic_graph_node(node, &program_matched)) return false;
        if (program_matched) continue;
        cudaKernelNodeParams params = {};
        if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) return false;
        const bool head =
            graph_kernel_is(params.func, (void *)k_head_norm_rope_hadamard<256, true>)
            || graph_kernel_is(params.func, (void *)k_head_norm_rope_hadamard<256, false>)
            || graph_kernel_is(params.func, (void *)k_head_norm_rope_hadamard<1024, true>)
            || graph_kernel_is(params.func, (void *)k_head_norm_rope_hadamard<1024, false>);
        if (head) {
            if (!add_dynamic_graph_node(
                    node, params, 13, 6, UINT32_MAX, UINT32_MAX)) return false;
            continue;
        }
        if (graph_kernel_is(params.func, (void *)k_rope)) {
            if (!add_dynamic_graph_node(
                    node, params, 9, 6, UINT32_MAX, UINT32_MAX)) return false;
            continue;
        }
        const bool store = graph_kernel_is(params.func, (void *)k_kv_store_f16)
            || graph_kernel_is(params.func, (void *)k_kv_store_q4)
            || graph_kernel_is(params.func, (void *)k_kv_store_q8);
        if (store) {
            if (!add_dynamic_graph_node(
                    node, params, kKvStoreGraphArgCount, kKvStoreGraphStartArg,
                    UINT32_MAX, UINT32_MAX)) return false;
            continue;
        }
        const bool d64_vector_partial = graph_kernel_is(
                params.func, (void *)imparo_sm80_d64_q4_vec::partial_q4)
            || graph_kernel_is(
                params.func, (void *)imparo_sm80_d64_q8_vec::partial_q8)
            || graph_kernel_is(
                params.func, (void *)imparo_sm80_d64_q8_vec::partial_q8_gqa4);
        if (d64_vector_partial) {
            if (!add_dynamic_graph_node(
                    node, params, kD64Q4VecGraphArgCount,
                    kD64Q4VecStartArg, UINT32_MAX,
                    kD64Q4VecRingArg)) return false;
            continue;
        }
        if (graph_kernel_is(params.func, (void *)k_attention_d64_controlled)) {
            if (!add_dynamic_graph_node(
                    node, params, kD64ControlledGraphArgCount,
                    kD64ControlledStartArg, UINT32_MAX,
                    kD64ControlledRingArg)) return false;
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d256_vec::partial_q4)) {
            // Ring addressing wins and this window-only kernel has no page-table
            // argument. Its padded schedule is fixed within each graph bucket.
            if (!add_dynamic_graph_node(
                    node, params, 12, 7, UINT32_MAX, 8)) return false;
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d256_vec::partial_q4_gqa4)) {
            if (!add_dynamic_graph_node(
                    node, params, 12, 7, UINT32_MAX, 8)) return false;
            continue;
        }
        if (graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_decode::partial_q4_controlled_paged)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512ControlledGraphArgCount, 7, 10, 9)) return false;
            continue;
        }
        if (graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_decode::partial_q4_controlled)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512IdentityControlledGraphArgCount, 7, 10, 9)) {
                return false;
            }
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d512_decode::partial_q4_paged)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512PartialGraphArgCount, 7, 10, 9)) return false;
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d512_decode::partial_q4)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512IdentityPartialGraphArgCount, 7, 10, 9)) {
                return false;
            }
            continue;
        }
        if (graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_decode::partial_f16_controlled_paged)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512ControlledGraphArgCount, 7, 10, 9)) return false;
            continue;
        }
        if (graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_decode::partial_f16_controlled)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512IdentityControlledGraphArgCount, 7, 10, 9)) {
                return false;
            }
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d512_decode::partial_f16_paged)) {
            if (!add_dynamic_graph_node(
                    node, params, kD512PartialGraphArgCount, 7, 10, 9)) return false;
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d512_decode::partial_f16)) {
            // Diagnostic node-update mode keeps the original numerical kernel and
            // patches its by-value start/valid arguments on every replay.
            if (!add_dynamic_graph_node(
                    node, params, kD512IdentityPartialGraphArgCount, 7, 10, 9)) {
                return false;
            }
            continue;
        }
        const bool scores_paged =
            graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::scores_paged<256, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::scores_paged<256, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::scores_paged<512, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::scores_paged<512, 2>);
        if (scores_paged) {
            if (!add_dynamic_graph_node(
                    node, params, kSmallScoresGraphArgCount, 7, 12, 11)) {
                return false;
            }
            continue;
        }
        const bool scores =
            graph_kernel_is(params.func, (void *)imparo_sm80_d512_small::scores<256, 1>)
            || graph_kernel_is(params.func, (void *)imparo_sm80_d512_small::scores<256, 2>)
            || graph_kernel_is(params.func, (void *)imparo_sm80_d512_small::scores<512, 1>)
            || graph_kernel_is(params.func, (void *)imparo_sm80_d512_small::scores<512, 2>);
        if (scores) {
            if (!add_dynamic_graph_node(
                    node, params, kSmallIdentityScoresGraphArgCount, 7, 12, 11)) {
                return false;
            }
            continue;
        }
        if (graph_kernel_is(
                params.func, (void *)imparo_sm80_d256_decode::values_full32_combine)) {
            if (!add_dynamic_graph_node(
                    node, params, kD256FullValuesGraphArgCount, UINT32_MAX, 8, 7)) return false;
            continue;
        }
        const bool values_paged =
            graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<256, 1, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<256, 1, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<256, 2, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<256, 2, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<512, 1, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<512, 1, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<512, 2, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine_paged<512, 2, 2>);
        if (values_paged && !add_dynamic_graph_node(
                node, params, kSmallValuesGraphArgCount, UINT32_MAX, 9, 8)) {
            return false;
        }
        if (values_paged) continue;
        const bool values =
            graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<256, 1, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<256, 1, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<256, 2, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<256, 2, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<512, 1, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<512, 1, 2>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<512, 2, 1>)
            || graph_kernel_is(params.func,
                (void *)imparo_sm80_d512_small::values_combine<512, 2, 2>);
        if (values && !add_dynamic_graph_node(
                node, params, kSmallIdentityValuesGraphArgCount, UINT32_MAX, 9, 8)) {
            return false;
        }
        if (values) continue;
    }
    return g.decode_graph_nodes.size() == g.graph_expected_dynamic_nodes;
} catch (...) {
    // Graph discovery owns temporary vectors and Program argument copies. Any
    // allocation/lock failure rejects the capture instead of crossing extern "C".
    return false;
}

int update_decode_graph_nodes(uint32_t start_pos) {
    const bool device_control = decode_device_control_enabled();
    for (DynamicGraphNode & dynamic : g.decode_graph_nodes) {
        if (!dynamic.program_arguments.empty()) {
            if (dynamic.program_arguments.size() != dynamic.args.size()
                || dynamic.program_update_sources.size() != dynamic.args.size()) {
                return CUDA_RC_ERROR;
            }
            for (size_t i = 0; i < dynamic.program_update_sources.size(); ++i) {
                switch (dynamic.program_update_sources[i]) {
                    case 0:
                        break;
                    case 1: // compiled adapter: decode_start_pos_u32
                        dynamic.program_arguments[i].u32 = start_pos;
                        break;
                    default:
                        return CUDA_RC_ERROR;
                }
            }
            const int update_rc =
                update_program_dynamic_graph_node(dynamic, start_pos);
            if (update_rc) return update_rc;
            continue;
        }
        // Device-control kernels read their position from the stable device slot,
        // but Program nodes have independent compiled-adapter update sources and
        // must still be patched above in the default device-control mode.
        if (device_control) continue;
        dynamic.start_value = start_pos;
        dynamic.valid_value = dynamic.ring
            ? std::min(start_pos + 1, dynamic.ring + 1) : start_pos + 1;
        if (dynamic.start_index != UINT32_MAX) {
            dynamic.args[dynamic.start_index] = &dynamic.start_value;
        }
        if (dynamic.valid_index != UINT32_MAX) {
            dynamic.args[dynamic.valid_index] = &dynamic.valid_value;
        }
        dynamic.params.kernelParams = dynamic.args.data();
        if (cudaGraphExecKernelNodeSetParams(
                g.decode_graph_exec, dynamic.node, &dynamic.params) != cudaSuccess) {
            return CUDA_RC_ERROR;
        }
    }
    return 0;
}

int stage_decode_inputs(uint32_t token) {
    if (!g.weights_host || !g.decode_row_desc_valid
        || g.decode_token_buf >= B_COUNT || !g.decode_token_stage_host
        || g.decode_row_bytes > g.decode_row_stage_host_bytes) {
        return CUDA_RC_INVALID;
    }
    if (g.decode_row_bytes) {
        const uint64_t row_off =
            g.decode_row_base + uint64_t(token) * g.decode_row_bytes;
        if (row_off < g.decode_row_base || row_off > g.weights_len
            || g.decode_row_bytes > g.weights_len - row_off) {
            return CUDA_RC_INVALID;
        }
        std::memcpy(g.decode_row_stage_host, g.weights_host + row_off,
                    size_t(g.decode_row_bytes));
    }
    *static_cast<uint32_t *>(g.decode_token_stage_host) = token;
    if (decode_device_control_active()) {
        if (!g.decode_control_host) return CUDA_RC_INVALID;
        *static_cast<uint32_t *>(g.decode_control_host) = g.decode_start_pos;
    }
    auto & shadow = g.u32_shadow[g.decode_token_buf];
    if (g.decode_token_off >= shadow.size()) return CUDA_RC_INVALID;
    shadow[size_t(g.decode_token_off)] = token;

    if (!g.decode_ple_desc_valid) return 0;
    const uint64_t staged_bytes = g.decode_ple_prefix + g.decode_ple_row_bytes;
    if (!g.ple_stage_host || staged_bytes > g.ple_stage_host_bytes) {
        return CUDA_RC_INVALID;
    }
    auto * staged = static_cast<uint8_t *>(g.ple_stage_host);
    if (g.decode_ple_prefix) {
        if (g.decode_ple_norm > g.weights_len
            || g.decode_ple_norm_bytes > g.weights_len - g.decode_ple_norm) {
            return CUDA_RC_INVALID;
        }
        std::memcpy(staged, g.weights_host + g.decode_ple_norm,
                    size_t(g.decode_ple_norm_bytes));
    }
    const uint64_t ple_off = g.decode_ple_table
        + uint64_t(token) * g.decode_ple_row_bytes;
    if (ple_off < g.decode_ple_table || ple_off > g.weights_len
        || g.decode_ple_row_bytes > g.weights_len - ple_off) return CUDA_RC_INVALID;
    std::memcpy(staged + size_t(g.decode_ple_prefix), g.weights_host + ple_off,
                size_t(g.decode_ple_row_bytes));
    return 0;
}

int upload_staged_decode_inputs() {
    // Keep request-dependent host transfers outside the reusable CUDA graph. The
    // token embedding table is resident for graph-capable models; a paged input
    // embedding would share weight_cache with PLE and therefore needs a dedicated
    // device slot before it can safely participate in capture.
    if (g.decode_row_bytes != 0
        || !g.decode_token_stage_host || g.decode_token_buf >= B_COUNT) {
        return CUDA_RC_INVALID;
    }
    const uint64_t ple_bytes = g.decode_ple_desc_valid
        ? g.decode_ple_prefix + g.decode_ple_row_bytes : 0;
    if (ple_bytes && (!g.ple_stage_host || !g.weight_cache
        || ple_bytes > g.ple_stage_host_bytes
        || ple_bytes > g.weight_cache_bytes)) {
        return CUDA_RC_INVALID;
    }
    if (cudaMemcpyAsync(
            static_cast<uint32_t *>(g.bufs[g.decode_token_buf]) + g.decode_token_off,
            g.decode_token_stage_host, sizeof(uint32_t), cudaMemcpyHostToDevice,
            g.stream) != cudaSuccess) {
        return CUDA_RC_ERROR;
    }
    if (ple_bytes && cudaMemcpyAsync(
            g.weight_cache, g.ple_stage_host, size_t(ple_bytes),
            cudaMemcpyHostToDevice, g.stream) != cudaSuccess) {
        return CUDA_RC_ERROR;
    }
    if (decode_device_control_active()) {
        if (!g.decode_control_host || !g.decode_control_device) {
            return CUDA_RC_INVALID;
        }
        if (cudaMemcpyAsync(
                g.decode_control_device, g.decode_control_host, sizeof(uint32_t),
                cudaMemcpyHostToDevice, g.stream) != cudaSuccess) {
            return CUDA_RC_ERROR;
        }
    }
    return 0;
}

} // namespace

// ---- extern "C" surface (mirrors the .mm's; the Rust backend_impl calls these) ----
// Every function returns 0 on success where a result is meaningful. The .def export
// list for the Windows DLL build lives next to this file.

extern "C" uint32_t imparo_cuda_abi_version(void) { return IMPARO_CUDA_BACKEND_ABI; }

extern "C" int imparo_cuda_runtime_identity(ImparoCudaRuntimeIdentityWire * out,
                                                uint32_t out_bytes) {
    if (!out || out_bytes < sizeof(ImparoCudaRuntimeIdentityWire)) {
        return CUDA_RC_INVALID;
    }
    std::memset(out, 0, sizeof(*out));
    int device = 0;
    const int selected_rc = selected_cuda_device(&device);
    if (selected_rc) return selected_rc;

    cudaDeviceProp prop = {};
    const cudaError_t prop_error = cudaGetDeviceProperties(&prop, device);
    if (prop_error != cudaSuccess) {
        std::fprintf(stderr, "imparo cuda: cudaGetDeviceProperties failed: %s\n",
                     cudaGetErrorString(prop_error));
        return CUDA_RC_ERROR;
    }
    int driver = 0;
    int runtime = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess
        || cudaRuntimeGetVersion(&runtime) != cudaSuccess
        || driver <= 0 || runtime <= 0) {
        return CUDA_RC_ERROR;
    }
    if (!decode_build_sha256(out->backend_build_sha256)) {
        return CUDA_RC_INVALID;
    }

    out->struct_bytes = uint32_t(sizeof(*out));
    out->backend_abi = IMPARO_CUDA_BACKEND_ABI;
    out->device_sm = uint32_t(10 * prop.major + prop.minor);
    out->driver_version = uint32_t(driver);
    out->runtime_version = uint32_t(runtime);
    std::memcpy(out->device_uuid, prop.uuid.bytes, sizeof(out->device_uuid));
    return 0;
}

extern "C" int imparo_cuda_init(const void * weights_host, uint64_t len,
                                  const WeightSpanWire * streamed,
                                  uint32_t streamed_count) {
    if (!weights_host || !len) return CUDA_RC_INVALID;
    if (streamed_count && !streamed) return CUDA_RC_INVALID;
    if (g.weights_host) {
        return g.weights_host == weights_host && g.weights_len == len ? 0 : CUDA_RC_INVALID;
    }
    const int runtime_rc = ensure_cuda_runtime();
    if (runtime_rc) return runtime_rc;
    g.weights_host = static_cast<const uint8_t *>(weights_host);
    g.weights_len = len;
    g.streamed_weights.clear();
    if (streamed_count) {
        g.streamed_weights.assign(streamed, streamed + streamed_count);
    }
    std::sort(g.streamed_weights.begin(), g.streamed_weights.end(),
        [](const WeightSpanWire & a, const WeightSpanWire & b) {
            return a.offset < b.offset;
        });
    uint64_t streamed_bytes = 0;
    uint64_t previous_end = 0;
    for (const WeightSpanWire & span : g.streamed_weights) {
        if (!span.bytes || span.offset < previous_end || span.offset > len
            || span.bytes > len - span.offset) {
            std::fprintf(stderr,
                "imparo cuda: invalid/overlapping streamed weight span off=%llu bytes=%llu\n",
                (unsigned long long)span.offset, (unsigned long long)span.bytes);
            return CUDA_RC_INVALID;
        }
        previous_end = span.offset + span.bytes;
        streamed_bytes += span.bytes;
    }
    const uint64_t resident_bytes = len - streamed_bytes;
    size_t free = 0, total = 0;
    CUDA_OK(cudaMemGetInfo(&free, &total));
    const uint64_t reserve = env_mib(
        "IMPARO_CUDA_RESERVE_MIB", std::max<uint64_t>(512ull << 20, uint64_t(total) / 8));
    g.runtime_reserve_bytes = reserve;

    // Fast path: only claim residency if activations/KV retain an explicit reserve.
    if (resident_bytes <= uint64_t(free)
        && resident_bytes + reserve <= uint64_t(free)) {
        int rc = alloc_raw(&g.weights, resident_bytes, "resident hot weights");
        if (!rc) {
            uint64_t file_cursor = 0;
            uint64_t device_cursor = 0;
            g.resident_segments.clear();
            for (const WeightSpanWire & span : g.streamed_weights) {
                if (span.offset > file_cursor) {
                    const uint64_t bytes = span.offset - file_cursor;
                    g.resident_segments.push_back({file_cursor, bytes, device_cursor});
                    CUDA_OK(cudaMemcpyAsync(
                        static_cast<uint8_t *>(g.weights) + device_cursor,
                        static_cast<const uint8_t *>(weights_host) + file_cursor,
                        size_t(bytes), cudaMemcpyHostToDevice, g.stream));
                    device_cursor += bytes;
                }
                file_cursor = span.offset + span.bytes;
            }
            if (file_cursor < len) {
                const uint64_t bytes = len - file_cursor;
                g.resident_segments.push_back({file_cursor, bytes, device_cursor});
                CUDA_OK(cudaMemcpyAsync(
                    static_cast<uint8_t *>(g.weights) + device_cursor,
                    static_cast<const uint8_t *>(weights_host) + file_cursor,
                    size_t(bytes), cudaMemcpyHostToDevice, g.stream));
                device_cursor += bytes;
            }
            CUDA_OK(cudaStreamSynchronize(g.stream));
            if (device_cursor != resident_bytes) return CUDA_RC_INVALID;
            g.weights_resident = true;
            g.weights_device_bytes = resident_bytes;
            // Sparse tables are staged in batches from the retained host mapping. The
            // capacity comes from the runtime reserve instead of a model-specific size.
            g.weight_cache_limit = std::max<uint64_t>(1ull << 20,
                std::min<uint64_t>(128ull << 20, reserve / 4));
            std::fprintf(stderr,
                "[imparo] cuda %s: resident hot weights %.1f MiB, streamed %.1f MiB, "
                "runtime reserve %.1f MiB\n",
                g.device_name, double(resident_bytes) / double(1ull << 20),
                double(streamed_bytes) / double(1ull << 20),
                double(reserve) / double(1ull << 20));
            return 0;
        }
        // A racing desktop allocation can invalidate cudaMemGetInfo between the check and
        // cudaMalloc. Clear the sticky OOM and continue into the bounded paging path.
        cudaGetLastError();
    }

    const uint64_t available = uint64_t(free) > reserve ? uint64_t(free) - reserve
                                                        : uint64_t(free) / 2;
    const uint64_t requested = env_mib("IMPARO_CUDA_WEIGHT_CACHE_MIB", 0);
    const uint64_t automatic = std::min<uint64_t>(512ull << 20, available / 2);
    g.weight_cache_limit = std::min(available, requested ? requested : automatic);
    if (g.weight_cache_limit < (1ull << 20)) {
        std::fprintf(stderr,
            "imparo cuda: no room for paged weights after reserve: free=%.1f MiB reserve=%.1f MiB\n",
            double(free) / double(1ull << 20), double(reserve) / double(1ull << 20));
        return CUDA_RC_OOM;
    }
    g.weights_resident = false;
    g.weights_device_bytes = 0;
    g.resident_segments.clear();
    int cache_rc = ensure_weight_cache(g.weight_cache_limit);
    if (cache_rc) return cache_rc;
    std::fprintf(stderr,
        "[imparo] cuda %s: paged-weight fit, model %.1f MiB > resident budget %.1f MiB; "
        "cache limit %.1f MiB, reserve %.1f MiB\n",
        g.device_name, double(len) / double(1ull << 20),
        double(uint64_t(free) > reserve ? uint64_t(free) - reserve : 0) / double(1ull << 20),
        double(g.weight_cache_limit) / double(1ull << 20),
        double(reserve) / double(1ull << 20));
    return 0;
}

extern "C" int imparo_cuda_memory_info(uint64_t * free_out, uint64_t * total_out,
                                        uint64_t * allocated_out) {
    size_t free = 0, total = 0;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) return CUDA_RC_ERROR;
    if (free_out) *free_out = uint64_t(free);
    if (total_out) *total_out = uint64_t(total);
    if (allocated_out) *allocated_out = tracked_bytes();
    return 0;
}

static uint64_t available_host_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? uint64_t(status.ullAvailPhys) : 0;
#elif defined(__linux__)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0
        || uint64_t(pages) > UINT64_MAX / uint64_t(page_bytes)) return 0;
    return uint64_t(pages) * uint64_t(page_bytes);
#else
    return 0;
#endif
}

static void cache_host_profile() {
    ImparoCudaHostProfileWire profile = {};
    profile.struct_bytes = uint32_t(sizeof(profile));
    profile.available_host_bytes = available_host_memory_bytes();

    size_t device_free = 0;
    size_t device_total = 0;
    if (!profile.available_host_bytes || !g.stream
        || cudaMemGetInfo(&device_free, &device_total) != cudaSuccess) {
        cudaGetLastError();
        g.host_profile = profile;
        g.host_profile_cached = true;
        return;
    }
    const auto plan = imparo_cuda_host_measure::benchmark_plan(
        uint64_t(device_free), profile.available_host_bytes);
    if (!plan.sample_bytes || plan.sample_bytes > SIZE_MAX) {
        g.host_profile = profile;
        g.host_profile_cached = true;
        return;
    }

    void * pinned = nullptr;
    void * device = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = cudaHostAlloc(
        &pinned, size_t(plan.sample_bytes), cudaHostAllocDefault) == cudaSuccess;
    if (ok) ok = cudaMalloc(&device, size_t(plan.sample_bytes)) == cudaSuccess;
    if (ok) ok = cudaEventCreate(&start) == cudaSuccess;
    if (ok) ok = cudaEventCreate(&stop) == cudaSuccess;
    if (ok) {
        std::memset(pinned, 0xa5, size_t(plan.sample_bytes));
        ok = cudaMemcpyAsync(device, pinned, size_t(plan.sample_bytes),
                             cudaMemcpyHostToDevice, g.stream) == cudaSuccess
            && cudaMemcpyAsync(pinned, device, size_t(plan.sample_bytes),
                               cudaMemcpyDeviceToHost, g.stream) == cudaSuccess
            && cudaStreamSynchronize(g.stream) == cudaSuccess;
    }

    float h2d_ms = 0.0f;
    if (ok) ok = cudaEventRecord(start, g.stream) == cudaSuccess;
    for (uint32_t i = 0; ok && i < plan.repetitions; ++i) {
        ok = cudaMemcpyAsync(device, pinned, size_t(plan.sample_bytes),
                             cudaMemcpyHostToDevice, g.stream) == cudaSuccess;
    }
    if (ok) ok = cudaEventRecord(stop, g.stream) == cudaSuccess;
    if (ok) ok = cudaEventSynchronize(stop) == cudaSuccess;
    if (ok) ok = cudaEventElapsedTime(&h2d_ms, start, stop) == cudaSuccess;

    float d2h_ms = 0.0f;
    if (ok) ok = cudaEventRecord(start, g.stream) == cudaSuccess;
    for (uint32_t i = 0; ok && i < plan.repetitions; ++i) {
        ok = cudaMemcpyAsync(pinned, device, size_t(plan.sample_bytes),
                             cudaMemcpyDeviceToHost, g.stream) == cudaSuccess;
    }
    if (ok) ok = cudaEventRecord(stop, g.stream) == cudaSuccess;
    if (ok) ok = cudaEventSynchronize(stop) == cudaSuccess;
    if (ok) ok = cudaEventElapsedTime(&d2h_ms, start, stop) == cudaSuccess;

    bool released = true;
    if (stop && cudaEventDestroy(stop) != cudaSuccess) released = false;
    if (start && cudaEventDestroy(start) != cudaSuccess) released = false;
    if (device && cudaFree(device) != cudaSuccess) released = false;
    if (pinned && cudaFreeHost(pinned) != cudaSuccess) released = false;
    if (ok && released) {
        profile.pinned_h2d_bytes_per_second =
            imparo_cuda_host_measure::bytes_per_second(
                plan.sample_bytes, plan.repetitions, h2d_ms);
        profile.pinned_d2h_bytes_per_second =
            imparo_cuda_host_measure::bytes_per_second(
                plan.sample_bytes, plan.repetitions, d2h_ms);
    } else {
        cudaGetLastError();
    }
    g.host_profile = profile;
    g.host_profile_cached = true;
}

extern "C" int imparo_cuda_host_profile(
        ImparoCudaHostProfileWire * out, uint32_t out_bytes) {
    if (!out || out_bytes < sizeof(ImparoCudaHostProfileWire)
        || g.graph_capturing || !g.stream) {
        return CUDA_RC_INVALID;
    }
    if (!g.host_profile_cached) cache_host_profile();
    std::memcpy(out, &g.host_profile, sizeof(g.host_profile));
    return 0;
}

extern "C" uint32_t imparo_cuda_weights_resident(void) {
    return g.weights_resident ? 1u : 0u;
}

extern "C" uint32_t imparo_cuda_verify_max_tokens(void) {
    // Cache-mode policy stays native, next to the kernels it guards. F16 can use
    // the full batched small-query family; Q4 uses batched projection rows and
    // the exact one-token attention schedule certified by recurrent crosschecks.
    if (g.sm_version < 80 || g.kv_type_k != g.kv_type_v) return 0u;
    if (g.kv_type_k == 1) return imparo_sm80_mmvq::kMaxTokens;
    return g.kv_type_k == 2 ? imparo_sm80_mmvq::kMaxTokens : 0u;
}

extern "C" uint32_t imparo_cuda_verify_tile_tokens(void) {
    // The architecture small-query attention kernel owns four query tokens per
    // physical tile. Wider verifier batches are valid but chain another tile.
    return imparo_cuda_verify_max_tokens()
        ? imparo_sm80_d512_small::kQueryTokens : 0u;
}

extern "C" uint32_t imparo_cuda_device_tag(uint8_t * dst, uint32_t len) {
    if (!dst || !len || !g.device_name[0]) return 0;
    const uint32_t n = std::min<uint32_t>(uint32_t(std::strlen(g.device_name)), len);
    std::memcpy(dst, g.device_name, n);
    return n;
}

static void begin_forward(bool decode) {
    g.forward_open = true;
    if (!decode) ++g.prefill_lab_sweep_index;
    static const bool profile =
        std::getenv("IMPARO_CUDA_PROFILE_FORWARD") != nullptr;
    g.forward_active = true;
    static const bool nsys_capture =
        std::getenv("IMPARO_CUDA_NSYS_CAPTURE") != nullptr;
    const bool tune_events = g.tuner_lab;
    if (!decode && nsys_capture && !g.nsys_capture_active) {
        const cudaError_t capture = cudaProfilerStart();
        g.nsys_capture_active = capture == cudaSuccess;
        if (capture != cudaSuccess) {
            std::fprintf(stderr,
                "imparo cuda: cudaProfilerStart failed: %s\n",
                cudaGetErrorString(capture));
            cudaGetLastError();
        }
    }
    if (decode && !g.decode_phase_active) {
        release_prefill_transients_for_decode();
        g.decode_phase_active = true;
    } else if (!decode) {
        g.decode_phase_active = false;
    }
    g.forward_decode = decode;
    g.ple_stage_used = false;
    const bool timing = profile || g.tuner_mode;
    g.forward_timing_armed = false;
    g.ffn_sidecar_trace_attempts = 0;
    g.ffn_sidecar_trace_admitted = 0;
    g.ffn_sidecar_trace_input_hits = 0;
    g.ffn_sidecar_trace_committed = 0;
    g.ffn_sidecar_model_calls = 0;
    g.ffn_sidecar_model_commits = 0;
    g.ffn_sidecar_model_pack_calls = 0;
    g.ffn_sidecar_model_pack_failed = false;
    const char * force_reclaim_env = std::getenv(
        "IMPARO_CUDA_Q4_PACKED_SIDECAR_FORCE_RECLAIM_LAB");
    if (!decode && g.ffn_sidecar_model_ready
            && !g.packed_q4_force_reclaim_exercised
            && force_reclaim_env && std::strcmp(force_reclaim_env, "1") == 0) {
        g.packed_q4_force_reclaim_exercised = true;
        g.packed_q4_force_reclaim_active = true;
        void * probe = nullptr;
        const int probe_rc = alloc_raw_with_packed_q4_reclaim(
            &probe, sizeof(uint32_t), "packed-Q4 reclaim probe");
        g.packed_q4_force_reclaim_active = false;
        if (probe) cudaFree(probe);
        if (probe_rc) set_pending(probe_rc, "packed-Q4 reclaim probe");
    }
    if (timing && !g.forward_start) {
        if (cudaEventCreate(&g.forward_start) != cudaSuccess
            || cudaEventCreate(&g.forward_stop) != cudaSuccess) {
            if (g.forward_start) cudaEventDestroy(g.forward_start);
            if (g.forward_stop) cudaEventDestroy(g.forward_stop);
            g.forward_start = nullptr;
            g.forward_stop = nullptr;
        }
    }
    g.tune_exact128_route_hits = 0;
    g.tune_exact128_attn_commits = 0;
    g.tune_exact128_ffn_commits = 0;
    g.tune_exact128_ple_commits = 0;
    g.tune_exact128_token64_commits = 0;
    g.forward_host_start = std::chrono::steady_clock::now();
    if (timing && g.forward_start
        && cudaEventRecord(g.forward_start, g.stream) == cudaSuccess) {
        g.forward_timing_armed = true;
    }
    const bool capture = decode && decode_graph_candidate()
        && g.decode_warm_forwards > 0 && !timing && !g.decode_graph_exec
        && !g.decode_graph_shape_blocked
        && g.decode_start_pos >= g.decode_graph_capture_after
        && g.decode_row_desc_valid && g.decode_row_bytes == 0;
    if (capture) {
        // Token ids and their PLE row are request data, not graph structure. Seed
        // both device slots before capture and keep their H2D transfers outside the
        // reusable graph; captured host-memory memcpy nodes otherwise replay the
        // first captured row after verifier batches touch the same input path.
        const int stage_rc = stage_decode_inputs(g.decode_token);
        const int upload_rc = stage_rc ? stage_rc : upload_staged_decode_inputs();
        if (upload_rc) {
            set_pending(upload_rc, "decode capture input staging");
            g.decode_graph_shape_blocked = true;
            return;
        }
        if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
            set_pending(CUDA_RC_ERROR, "decode capture input upload");
            g.decode_graph_shape_blocked = true;
            return;
        }
        g.graph_capture_compatible = true;
        g.graph_expected_dynamic_nodes = 0;
        g.graph_min_start = 0;
        g.graph_max_start = UINT32_MAX;
        invalidate_q8_cache();
        const cudaError_t err = cudaStreamBeginCapture(
            g.stream, cudaStreamCaptureModeThreadLocal);
        g.graph_capturing = err == cudaSuccess;
        if (g.graph_capturing) {
            ++g.graph_capture_generation;
            if (!g.graph_capture_generation) ++g.graph_capture_generation;
        } else {
            g.decode_graph_shape_blocked = true;
            cudaGetLastError();
        }
    }
    if (decode && decode_device_control_active() && !g.graph_capturing) {
        if (!g.decode_control_host || !g.decode_control_device) {
            set_pending(CUDA_RC_INVALID, "decode control allocation");
        } else {
            *static_cast<uint32_t *>(g.decode_control_host) = g.decode_start_pos;
            if (cudaMemcpyAsync(
                    g.decode_control_device, g.decode_control_host, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, g.stream) != cudaSuccess) {
                set_pending(CUDA_RC_ERROR, "decode control upload");
            }
        }
    }
}
extern "C" void imparo_cuda_begin(void) { begin_forward(false); }
extern "C" void imparo_cuda_begin_forward(uint32_t decode) {
    begin_forward(decode != 0);
}
extern "C" int imparo_cuda_set_batch_geometry(
        uint64_t absolute_start, uint32_t active_tokens, uint32_t phase) {
    if (active_tokens == 0 || phase > 1 || absolute_start > UINT32_MAX
        || absolute_start + uint64_t(active_tokens) > uint64_t(UINT32_MAX) + 1) {
        g.batch_geometry_valid = false;
        return CUDA_RC_INVALID;
    }
    g.batch_geometry_start = uint32_t(absolute_start);
    g.batch_geometry_tokens = active_tokens;
    g.batch_geometry_phase = phase;
    g.batch_geometry_valid = true;
    return 0;
}
static int prefill_body_capture_or_replay(
        uint32_t count, uint32_t start_pos, uint32_t argmax) {
    const bool descriptor_ok = g.forward_open && !g.forward_decode
        && g.prefill_prepared && g.prefill_token_buf < B_COUNT
        && g.prefill_tokens.size() == count
        && g.prefill_graph_tokens == count
        && g.prefill_graph_start == start_pos
        && g.prefill_graph_argmax == (argmax != 0);
    if (!descriptor_ok) {
        destroy_prefill_graph();
        g.prefill_graph_blocked = true;
        return 0;
    }
    if (g.prefill_graph_exec) {
        // The request-specific embedding/PLE prefix is already queued on this
        // stream. Graph launch is ordered after it and reuses only the model body.
        invalidate_q8_cache();
        const cudaError_t launch = cudaGraphLaunch(g.prefill_graph_exec, g.stream);
        if (trace_cuda_graph() && launch == cudaSuccess) {
            std::fprintf(stderr,
                "[cuda-prefill-graph] action=replay tokens=%u start=%u "
                "argmax=%u exact128_sidecar=%u boundary=post-embedding\n",
                count, start_pos, argmax,
                unsigned(tuned_exact128_sm86_route(count)));
        }
        return launch == cudaSuccess ? 1 : -CUDA_RC_ERROR;
    }
    if (!g.prefill_capture_requested) return 0;
    g.prefill_capture_requested = false;
    if (g.forward_start) {
        g.prefill_graph_blocked = true;
        return 0;
    }
    // Capture starts only after all request-dependent host staging has finished.
    // Synchronizing is paid once at capture, never on replay.
    const cudaError_t prefix_sync = cudaStreamSynchronize(g.stream);
    if (prefix_sync != cudaSuccess) {
        g.prefill_graph_blocked = true;
        set_pending(CUDA_RC_ERROR, "prefill body prefix sync");
        cudaGetLastError();
        return 0;
    }
    const cudaError_t capture = cudaStreamBeginCapture(
        g.stream, cudaStreamCaptureModeThreadLocal);
    g.prefill_capture_active = capture == cudaSuccess;
    g.graph_capture_compatible = g.prefill_capture_active;
    g.graph_capturing = g.prefill_capture_active;
    if (!g.prefill_capture_active) {
        g.prefill_graph_blocked = true;
        cudaGetLastError();
    }
    return 0;
}

extern "C" int imparo_cuda_prefill_prepare(
        const uint32_t * tokens, uint32_t count,
        uint32_t start_pos, uint32_t argmax) {
    if (g.forward_open && g.prefill_prepared) {
        return prefill_body_capture_or_replay(count, start_pos, argmax);
    }
    g.prefill_prepared = false;
    if (!tokens || !prefill_graph_candidate(count, start_pos, argmax != 0)) {
        return 0;
    }
    const bool same_key = g.prefill_graph_tokens == count
        && g.prefill_graph_start == start_pos
        && g.prefill_graph_argmax == (argmax != 0);
    if (!same_key) {
        destroy_prefill_graph();
        g.prefill_graph_tokens = count;
        g.prefill_graph_start = start_pos;
        g.prefill_graph_argmax = argmax != 0;
        g.prefill_warm_forwards = 0;
        g.prefill_graph_blocked = false;
        g.prefill_token_buf = UINT32_MAX;
        g.prefill_token_off = 0;
    }
    try {
        g.prefill_tokens.assign(tokens, tokens + count);
    } catch (const std::bad_alloc &) {
        return -CUDA_RC_OOM;
    } catch (const std::length_error &) {
        return -CUDA_RC_INVALID;
    }
    g.prefill_prepared = true;

    // The exact-128 sidecar needs two successful ordinary forwards: the first
    // discovers the complete packed model and deliberately falls back, the second
    // proves every sidecar/scratch/configuration path can commit without allocation.
    // Capture starts only on the third matching forward.
    const bool exact128_sidecar = tuned_exact128_sm86_route(count)
        || tuned_exact128_fast_transaction(count)
        || prefill_sidecar_graph_lab_override(count);
    const uint32_t required_warm_forwards = exact128_sidecar ? 2u : 1u;
    if (g.prefill_warm_forwards >= required_warm_forwards
        && (!exact128_sidecar || g.ffn_sidecar_model_ready)
        && !g.prefill_graph_blocked && g.prefill_token_buf < B_COUNT) {
        g.prefill_capture_requested = true;
    }
    return 0;
}
extern "C" int imparo_cuda_decode_prepare(
        uint32_t token, uint32_t start_pos, uint32_t argmax) {
    // Diagnostic override for isolating stale transient state. It intentionally
    // disables graph reuse and must never be selected by production policy.
    const bool release_each = std::getenv("IMPARO_CUDA_RELEASE_EACH_DECODE") != nullptr;
    if (!g.decode_phase_active || release_each) {
        release_prefill_transients_for_decode();
        g.decode_phase_active = true;
    }
    g.decode_prepared = true;
    g.decode_argmax = argmax != 0;
    g.decode_token = token;
    if (g.decode_previous_start_pos == UINT32_MAX
        || start_pos != g.decode_previous_start_pos + 1) {
        g.decode_sequence_start_pos = start_pos;
    }
    g.decode_previous_start_pos = start_pos;
    g.decode_start_pos = start_pos;
    if (trace_cuda_graph()) {
        static uint32_t prepare_traces = 0;
        if (prepare_traces < 4) {
            std::fprintf(stderr,
                "[cuda-graph] prepare start=%u argmax=%u enabled=%u pinned=%u "
                "async=%u resident=%u kv=%u/%u sm=%d warm=%u blocked=%u "
                "capture_after=%u row_desc=%u ple_desc=%u exec=%u\n",
                start_pos, argmax, unsigned(decode_graph_enabled()),
                unsigned(decode_pinned_input_enabled()),
                unsigned(async_host_control_enabled()), unsigned(g.weights_resident),
                g.kv_type_k, g.kv_type_v, g.sm_version, g.decode_warm_forwards,
                unsigned(g.decode_graph_shape_blocked), g.decode_graph_capture_after,
                unsigned(g.decode_row_desc_valid), unsigned(g.decode_ple_desc_valid),
                unsigned(g.decode_graph_exec != nullptr));
            ++prepare_traces;
        }
    }
    if (!decode_graph_candidate()) return 0;
    if (decode_device_control_active()) {
        const int control_rc = ensure_decode_control();
        if (control_rc) return -control_rc;
        *static_cast<uint32_t *>(g.decode_control_host) = start_pos;
    }
    if (!g.decode_graph_exec) return 0;
    if (start_pos == UINT32_MAX) {
        destroy_decode_graph();
        g.decode_warm_forwards = 0;
        return 0;
    }
    if (start_pos < g.graph_min_start || start_pos > g.graph_max_start) {
        destroy_decode_graph();
        // Run one uncaptured token so bucket-dependent workspaces can grow before
        // the next capture; CUDA allocation is deliberately forbidden in capture.
        g.decode_warm_forwards = 0;
        return 0;
    }
    const int stage_rc = stage_decode_inputs(token);
    if (stage_rc) {
        destroy_decode_graph();
        return 0;
    }
    if (upload_staged_decode_inputs()) {
        destroy_decode_graph();
        return 0;
    }
    const int update_rc = update_decode_graph_nodes(start_pos);
    if (update_rc) {
        destroy_decode_graph();
        return 0;
    }
    invalidate_q8_cache();
    g.forward_decode = true;
    g.forward_host_start = std::chrono::steady_clock::now();
    const cudaError_t launch = cudaGraphLaunch(g.decode_graph_exec, g.stream);
    if (launch == cudaSuccess) g.forward_active = true;
    if (trace_cuda_graph() && launch == cudaSuccess) {
        static uint64_t replays = 0;
        ++replays;
        if (replays == 1 || replays % 128 == 0) {
            std::fprintf(stderr,
                "[cuda-graph] replay=%llu start=%u range=%u..%u nodes=%zu\n",
                static_cast<unsigned long long>(replays), start_pos,
                g.graph_min_start, g.graph_max_start, g.decode_graph_nodes.size());
        }
    }
    return launch == cudaSuccess ? 1 : -CUDA_RC_ERROR;
}
extern "C" void imparo_cuda_flush(void) { /* async stream: kernels already queued */ }
extern "C" int imparo_cuda_end(void) {
    const bool profile = std::getenv("IMPARO_CUDA_PROFILE_FORWARD") != nullptr;
    bool discard_captured_graph = false;
    if (g.graph_capturing) {
        const bool captured_prefill = g.prefill_capture_active;
        cudaGraph_t captured = nullptr;
        const cudaError_t end_capture = cudaStreamEndCapture(g.stream, &captured);
        g.graph_capturing = false;
        invalidate_q8_cache();
        g.prefill_capture_active = false;
        if (captured_prefill) {
            if (end_capture != cudaSuccess || !captured
                    || !g.graph_capture_compatible || g.pending_error) {
                if (trace_cuda_graph()) {
                    std::fprintf(stderr,
                        "[cuda-prefill-graph] action=discard end=%d graph=%u "
                        "compatible=%u pending=%d\n",
                        int(end_capture), unsigned(captured != nullptr),
                        unsigned(g.graph_capture_compatible), g.pending_error);
                }
                if (captured) cudaGraphDestroy(captured);
                g.prefill_graph_blocked = true;
                g.pending_error = g.pending_error ? g.pending_error : CUDA_RC_ERROR;
            } else {
                destroy_decode_graph();
                g.prefill_graph = captured;
                const cudaError_t instantiate = cudaGraphInstantiate(
                    &g.prefill_graph_exec, g.prefill_graph, nullptr, nullptr, 0);
                if (instantiate != cudaSuccess) {
                    destroy_prefill_graph();
                    g.prefill_graph_blocked = true;
                    g.pending_error = g.pending_error
                        ? g.pending_error : CUDA_RC_ERROR;
                } else {
                    if (trace_cuda_graph()) {
                        std::fprintf(stderr,
                            "[cuda-prefill-graph] action=capture tokens=%u "
                            "start=%u argmax=%u exact128_sidecar=%u\n",
                            g.prefill_graph_tokens, g.prefill_graph_start,
                            unsigned(g.prefill_graph_argmax), unsigned(
                                tuned_exact128_sm86_route(g.prefill_graph_tokens)));
                    }
                    const cudaError_t graph_launch = cudaGraphLaunch(
                        g.prefill_graph_exec, g.stream);
                    if (graph_launch != cudaSuccess) {
                        g.pending_error = g.pending_error
                            ? g.pending_error : CUDA_RC_ERROR;
                    }
                }
            }
        } else if (end_capture != cudaSuccess || !captured) {
            if (captured) cudaGraphDestroy(captured);
            g.decode_graph_shape_blocked = true;
            g.pending_error = g.pending_error ? g.pending_error : CUDA_RC_ERROR;
        } else {
            destroy_decode_graph();
            g.decode_graph = captured;
            const cudaError_t instantiate = cudaGraphInstantiate(
                &g.decode_graph_exec, g.decode_graph, nullptr, nullptr, 0);
            if (instantiate != cudaSuccess) {
                destroy_decode_graph();
                g.decode_graph_shape_blocked = true;
                g.pending_error = g.pending_error ? g.pending_error : CUDA_RC_ERROR;
            } else {
                const bool nodes_ok = configure_decode_graph_nodes();
                const size_t configured_dynamic_nodes = g.decode_graph_nodes.size();
                const bool range_ok = g.decode_start_pos >= g.graph_min_start
                    && g.decode_start_pos <= g.graph_max_start;
                if (!nodes_ok || !range_ok) {
                    g.graph_capture_compatible = false;
                    g.decode_graph_shape_blocked = true;
                    g.decode_graph_nodes.clear();
                }
                if (trace_cuda_graph()) {
                    size_t total_nodes = 0;
                    const cudaError_t count_nodes = cudaGraphGetNodes(
                        g.decode_graph, nullptr, &total_nodes);
                    if (count_nodes != cudaSuccess) cudaGetLastError();
                    std::fprintf(stderr,
                        "[cuda-graph] capture start=%u range=%u..%u "
                        "dynamic=%zu/%u total=%zu total_ok=%u "
                        "nodes_ok=%u range_ok=%u %s\n",
                        g.decode_start_pos, g.graph_min_start, g.graph_max_start,
                        configured_dynamic_nodes, g.graph_expected_dynamic_nodes,
                        total_nodes, unsigned(count_nodes == cudaSuccess),
                        unsigned(nodes_ok), unsigned(range_ok),
                        g.graph_capture_compatible ? "retained" : "discarded");
                }
                const cudaError_t graph_launch = cudaGraphLaunch(
                    g.decode_graph_exec, g.stream);
                if (graph_launch != cudaSuccess) {
                    g.pending_error = g.pending_error
                        ? g.pending_error : CUDA_RC_ERROR;
                }
                discard_captured_graph = !g.graph_capture_compatible;
            }
        }
    }
    cudaError_t launch = cudaGetLastError();
    if (g.forward_timing_armed
        && cudaEventRecord(g.forward_stop, g.stream) != cudaSuccess) {
        g.forward_timing_armed = false;
        cudaGetLastError();
    }
    const auto before_sync = std::chrono::steady_clock::now();
    cudaError_t sync = cudaStreamSynchronize(g.stream);
    const auto after_sync = std::chrono::steady_clock::now();
    if (g.nsys_capture_active) {
        const cudaError_t capture = cudaProfilerStop();
        g.nsys_capture_active = false;
        if (capture != cudaSuccess) {
            std::fprintf(stderr,
                "imparo cuda: cudaProfilerStop failed: %s\n",
                cudaGetErrorString(capture));
            cudaGetLastError();
        }
    }
    if (g.forward_timing_armed) {
        float gpu_ms = 0.0f;
        if (cudaEventElapsedTime(&gpu_ms, g.forward_start, g.forward_stop)
            == cudaSuccess) {
            g.last_gpu_us = double(gpu_ms) * 1000.0;
        } else {
            g.last_gpu_us = 0.0;
            cudaGetLastError();
        }
    } else {
        g.last_gpu_us = 0.0;
    }
    if (profile && g.forward_timing_armed) {
        const double gpu_ms = g.last_gpu_us / 1000.0;
        const double enqueue_ms = std::chrono::duration<double, std::milli>(
            before_sync - g.forward_host_start).count();
        const double wait_ms = std::chrono::duration<double, std::milli>(
            after_sync - before_sync).count();
        std::fprintf(stderr,
            "[cuda-forward-prof] enqueue_ms=%.6f wait_ms=%.6f gpu_ms=%.6f\n",
            enqueue_ms, wait_ms, gpu_ms);
    }
    if (!g.forward_decode
            && std::getenv("IMPARO_CUDA_PREFILL_FFN_SIDECAR_TRACE")) {
        std::fprintf(stderr,
            "[cuda-ffn-sidecar] attempts=%llu admitted=%llu "
            "input_hits=%llu committed=%llu prepared=%u pack_calls=%u\n",
            (unsigned long long)g.ffn_sidecar_trace_attempts,
            (unsigned long long)g.ffn_sidecar_trace_admitted,
            (unsigned long long)g.ffn_sidecar_trace_input_hits,
            (unsigned long long)g.ffn_sidecar_trace_committed,
            unsigned(g.ffn_sidecar_model_ready),
            g.ffn_sidecar_model_pack_calls);
    }
    if (!g.forward_decode && !g.ffn_sidecar_model_ready
            && g.ffn_sidecar_model_calls
            && g.ffn_sidecar_model_calls == g.kv_layout.layers
            && g.ffn_sidecar_model_pack_calls == g.kv_layout.layers
            && !g.ffn_sidecar_model_pack_failed
            && launch == cudaSuccess && sync == cudaSuccess
            && !g.pending_error) {
        g.ffn_sidecar_model_ready = true;
        if (std::getenv("IMPARO_CUDA_PREFILL_FFN_SIDECAR_TRACE")) {
            std::fprintf(stderr,
                "[cuda-ffn-sidecar] complete model hot set prepared; "
                "next forward may commit\n");
        }
    }
    if (!g.forward_decode && g.ffn_sidecar_model_commits
            && (g.ffn_sidecar_model_calls != g.ffn_sidecar_model_commits
                || g.ffn_sidecar_model_commits != g.kv_layout.layers)) {
        set_pending(CUDA_RC_ERROR,
            "sidecar-only FFN model admission was not all-or-none");
    }
    int rc = g.pending_error;
    g.pending_error = 0;
    if (launch != cudaSuccess) {
        std::fprintf(stderr, "imparo cuda: kernel launch failed: %s\n", cudaGetErrorString(launch));
        rc = rc ? rc : CUDA_RC_ERROR;
    }
    if (sync != cudaSuccess) {
        std::fprintf(stderr, "imparo cuda: stream failed: %s\n", cudaGetErrorString(sync));
        rc = rc ? rc : CUDA_RC_ERROR;
    }
    if (discard_captured_graph) {
        destroy_decode_graph();
        // A rejected capture may have discovered a larger schedule bucket. Give
        // the next forward a normal allocation-capable warmup pass.
        g.decode_warm_forwards = 0;
    } else if (g.forward_decode && !rc) {
        ++g.decode_warm_forwards;
    } else if (g.prefill_prepared && !rc) {
        ++g.prefill_warm_forwards;
    }
    g.forward_open = false;
    g.forward_decode = false;
    g.forward_active = false;
    g.decode_prepared = false;
    g.forward_timing_armed = false;
    g.prefill_prepared = false;
    g.prefill_capture_requested = false;
    return rc;
}
// Static exact-route feasibility surface. Dynamic plugins return zero until a future
// ABI revision promotes this route-evidence counter; device timing itself is already
// provided by the ABI-26 imparo_cuda_last_gpu_us surface below.
extern "C" uint32_t imparo_cuda_exact128_route_hits_lab(void) {
    return g.tuner_lab ? exact128_packed_route_evidence() : 0u;
}
extern "C" void imparo_cuda_enable_tuner_lab(void) {
    g.tuner_lab = true;
}

extern "C" double imparo_cuda_last_gpu_us(void) { return g.last_gpu_us; }

extern "C" int imparo_cuda_set_tuner_mode(uint32_t enabled) {
    if (g.graph_capturing) return CUDA_RC_INVALID;
    const bool next = enabled != 0;
    if (next == g.tuner_mode) return 0;
    if (g.stream && cudaStreamSynchronize(g.stream) != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    destroy_decode_graph();
    g.decode_warm_forwards = 0;
    g.decode_graph_shape_blocked = false;
    g.decode_graph_capture_after = 0;
    g.tuner_mode = next;
    g.last_gpu_us = 0.0;
    g.proof_expected_knob_mask = 0;
    g.proof_observed_knob_mask = 0;
    return 0;
}

extern "C" int imparo_cuda_device_profile(
        ImparoCudaDeviceProfileWire * out, uint32_t out_bytes) {
    if (!out || out_bytes < sizeof(ImparoCudaDeviceProfileWire)) {
        return CUDA_RC_INVALID;
    }
    int device = 0;
    if (selected_cuda_device(&device)) return CUDA_RC_ERROR;
    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    ImparoCudaDeviceProfileWire profile = {};
    profile.struct_bytes = uint32_t(sizeof(profile));
    profile.max_threads = uint32_t(std::max(prop.maxThreadsPerBlock, 0));
    profile.threadgroup_bytes = uint64_t(prop.sharedMemPerBlockOptin > 0
        ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock);
    std::memcpy(out, &profile, sizeof(profile));
    return 0;
}

extern "C" void imparo_cuda_dispatch_proof_reset(void) {
    g.proof_dispatches = 0;
    g.proof_observed_choice_epoch = 0;
    g.proof_family = 0;
    g.proof_variant = 0;
    g.proof_observed_knob_mask = 0;
}


extern "C" int imparo_cuda_dispatch_expectation(uint64_t knob_mask) {
    if (!g.tuner_mode) return CUDA_RC_INVALID;
    g.proof_expected_knob_mask = knob_mask;
    g.proof_observed_knob_mask = 0;
    return 0;
}

extern "C" int imparo_cuda_dispatch_proof(
        ImparoCudaDispatchProofWire * out, uint32_t out_bytes) {
    if (!out || out_bytes < sizeof(ImparoCudaDispatchProofWire) || !g.tuner_mode) {
        return CUDA_RC_INVALID;
    }
    ImparoCudaDispatchProofWire proof = {};
    proof.struct_bytes = uint32_t(sizeof(proof));
    proof.family = g.proof_family;
    proof.choice_epoch = g.choice_epoch;
    proof.observed_choice_epoch = g.proof_observed_choice_epoch;
    proof.dispatches = g.proof_dispatches;
    proof.variant = g.proof_variant;
    proof.expected_knob_mask = g.proof_expected_knob_mask;
    proof.observed_knob_mask = g.proof_observed_knob_mask;
    std::memcpy(out, &proof, sizeof(proof));
    return 0;
}

extern "C" double imparo_cuda_probe(
        uint32_t kind, uint64_t a, uint32_t b, uint32_t c,
        uint32_t d, uint32_t e) {
    if (!g.tuner_mode || !g.stream) return 0.0;
    if (kind == 1) {
        static constexpr uint32_t nacc[] = {4, 8, 12, 16, 24, 32, 48, 64};
        const uint32_t idx = uint32_t(a);
        const uint32_t tgs = b;
        const uint32_t tpg = c;
        const uint32_t iters = d;
        if (idx >= 8 || !tgs || !tpg || tpg > 1024 || !iters) return 0.0;
        const uint64_t threads = uint64_t(tgs) * tpg;
        if (!ensure_probe_storage(0, threads * sizeof(float)) || !probe_begin()) {
            return 0.0;
        }
        float * out = static_cast<float *>(g.probe_output);
        switch (idx) {
            case 0: k_probe_accumulators<4><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 1: k_probe_accumulators<8><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 2: k_probe_accumulators<12><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 3: k_probe_accumulators<16><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 4: k_probe_accumulators<24><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 5: k_probe_accumulators<32><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 6: k_probe_accumulators<48><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
            case 7: k_probe_accumulators<64><<<tgs, tpg, 0, g.stream>>>(out, iters); break;
        }
        const double ms = probe_end_ms();
        const double ops = double(threads) * iters * nacc[idx] * 2.0;
        return ms > 0.0 ? ops / (ms * 1.0e9) : 0.0;
    }
    if (kind == 2) {
        const uint64_t bytes = a & ~uint64_t(3);
        const uint32_t reps = b;
        const uint32_t tgs = c;
        const uint32_t tpg = d;
        if (!bytes || !reps || !tgs || !tpg || tpg > 1024) return 0.0;
        const uint64_t threads = uint64_t(tgs) * tpg;
        if (!ensure_probe_storage(bytes, threads * sizeof(float)) || !probe_begin()) {
            return 0.0;
        }
        k_probe_bandwidth<<<tgs, tpg, 0, g.stream>>>(
            static_cast<const float *>(g.probe_input),
            static_cast<float *>(g.probe_output), bytes / sizeof(float), reps);
        const double ms = probe_end_ms();
        return ms > 0.0 ? double(bytes) * reps / (ms * 1.0e6) : 0.0;
    }
    if (kind == 3) {
        const uint32_t tgs = uint32_t(a);
        const uint32_t sgs = b;
        const uint32_t iters = c;
        const uint32_t stride = d;
        const uint32_t kspan = e;
        const uint32_t tpg = sgs * 32;
        if (!tgs || !sgs || tpg > 1024 || !iters || !stride || !kspan) return 0.0;
        if ((stride & 1) != 0 || stride < 256) return 0.0;
        const uint64_t rows = uint64_t(kspan) + 1;
        if (rows > UINT64_MAX / stride
            || rows * stride > UINT64_MAX / sizeof(__half)) return 0.0;
        const uint64_t input_bytes = rows * stride * sizeof(__half);
        const uint64_t threads = uint64_t(tgs) * tpg;
        if (!ensure_probe_storage(input_bytes, threads * sizeof(float))
            || !probe_begin()) return 0.0;
        k_probe_scoremix<<<tgs, tpg, size_t(sgs) * 1024, g.stream>>>(
            static_cast<const __half2 *>(g.probe_input),
            static_cast<float *>(g.probe_output), iters, stride, kspan);
        const double ms = probe_end_ms();
        const double ops = double(tgs) * sgs * iters * 8192.0;
        return ms > 0.0 ? ops / (ms * 1.0e9) : 0.0;
    }
    const uint32_t n = uint32_t(a);
    if (kind < 4 || kind > 6 || !n || !ensure_probe_storage(0, sizeof(float))) {
        return 0.0;
    }
    if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    const auto start = std::chrono::steady_clock::now();
    if (kind == 4) {
        if (!ensure_forward_events()) return 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            if (cudaEventRecord(g.forward_start, g.stream) != cudaSuccess) return 0.0;
        }
    } else if (kind == 5) {
        if (!ensure_forward_events()) return 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            if (cudaEventRecord(g.forward_start, g.stream) != cudaSuccess
                || cudaEventSynchronize(g.forward_start) != cudaSuccess) return 0.0;
        }
    } else {
        for (uint32_t i = 0; i < n; ++i) {
            k_probe_noop<<<1, 1, 0, g.stream>>>(static_cast<float *>(g.probe_output));
        }
    }
    const double host_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start).count();
    if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    return host_us / n;
}

extern "C" int imparo_cuda_alloc(uint32_t id, uint64_t bytes) {
    if (id >= B_COUNT) return 1;
    if (g.bufs[id] && !g.in_arena[id] && g.sizes[id] >= bytes
        && bytes * 2 >= g.sizes[id]) return 0;   // the .mm's shrink hysteresis
    destroy_decode_graph();
    if (g.bufs[id] && !g.in_arena[id]) cudaFree(g.bufs[id]);
    g.bufs[id] = nullptr; g.sizes[id] = 0; g.in_arena[id] = false;
    int rc = alloc_raw_with_packed_q4_reclaim(
        &g.bufs[id], bytes, "activation buffer");
    if (rc) return rc;
    CUDA_OK(cudaMemsetAsync(g.bufs[id], 0, bytes, g.stream));
    g.sizes[id] = bytes; g.in_arena[id] = false;
    mark_buf_written(id);
    return 0;
}
extern "C" int imparo_cuda_arena(uint64_t bytes) {
    if (g.arena && bytes <= g.arena_size && bytes * 2 >= g.arena_size) return 0;
    destroy_decode_graph();
    for (int i = 0; i < B_COUNT; ++i) if (g.in_arena[i]) {
        g.bufs[i] = nullptr; g.in_arena[i] = false; mark_buf_written(i);
    }
    if (g.arena) cudaFree(g.arena);
    g.arena = nullptr; g.arena_size = 0;
    int rc = alloc_raw_with_packed_q4_reclaim(
        &g.arena, bytes, "activation arena");
    if (rc) return rc;
    CUDA_OK(cudaMemsetAsync(g.arena, 0, bytes, g.stream));
    g.arena_size = bytes;
    return 0;
}
extern "C" int imparo_cuda_place(uint32_t id, uint64_t offset, uint64_t bytes) {
    if (id >= B_COUNT || !g.arena || offset + bytes > g.arena_size) return 1;
    void * next = static_cast<uint8_t *>(g.arena) + offset;
    if (g.bufs[id] != next || g.sizes[id] != bytes) destroy_decode_graph();
    // A slot may move between a small standalone decode allocation and a wider
    // batch-liveness placement. Release the former before replacing its pointer;
    // otherwise every 1 -> N activation-width transition leaks one device buffer.
    if (g.bufs[id] && !g.in_arena[id]) cudaFree(g.bufs[id]);
    g.bufs[id] = next;
    g.sizes[id] = bytes; g.in_arena[id] = true;
    mark_buf_written(id);
    return 0;
}
extern "C" uint64_t imparo_cuda_page_round(uint64_t n) {
    return (n + 4095) / 4096 * 4096;
}

static void invalidate_kv_dequant(uint32_t layer, uint32_t is_v) {
    (is_v ? g.vdq : g.kdq).invalidate_layer(layer);
}

static bool kv_byte_range_owned(uint32_t layer, uint32_t is_v,
                                uint64_t off, uint64_t len) {
    return imparo_cuda_kv::range_owned(g.kv_layout, layer, is_v, off, len);
}

static const uint32_t * kv_device_page_table(uint32_t layer) {
    if (layer >= g.kv_page_tables.layers || !g.kv_page_table_arena
        || !g.kv_page_tables.layer[layer].capacity) return nullptr;
    return static_cast<const uint32_t *>(g.kv_page_table_arena)
        + g.kv_page_tables.layer[layer].arena_offset;
}

static bool kv_page_table_requires_mapping(uint32_t layer) {
    return layer < g.kv_page_tables.layers
        && imparo_cuda_kv::page_table_requires_mapping(
            g.kv_page_tables.layer[layer].host_shadow,
            g.kv_page_tables.layer[layer].capacity);
}


static int paging_rc(imparo_cuda_kv::PagingRc rc) {
    if (rc == imparo_cuda_kv::PagingRc::Ok) return 0;
    return rc == imparo_cuda_kv::PagingRc::OutOfMemory
        ? CUDA_RC_OOM : CUDA_RC_INVALID;
}

static int replace_kv_arena(uint32_t n_layers, const uint64_t * bytes,
                            const imparo_cuda_kv::PagingLayout * layouts,
                            uint32_t layout_count, bool preserve) {
    if (g.graph_capturing) return CUDA_RC_INVALID;
    imparo_cuda_kv::Layout<MAX_LAYERS> next;
    if (!imparo_cuda_kv::build_layout(n_layers, bytes, &next)) {
        return CUDA_RC_INVALID;
    }

    // Legacy microbench callers do not address through page tables. Give them an
    // explicit zero-slot descriptor rather than guessing geometry from byte counts.
    std::vector<imparo_cuda_kv::PagingLayout> unpaged;
    if (!layouts && layout_count == 0) {
        try {
            unpaged.resize(n_layers);
        } catch (const std::bad_alloc &) {
            return CUDA_RC_OOM;
        }
        for (uint32_t layer = 0; layer < n_layers; ++layer) {
            unpaged[layer].layer = layer;
        }
        layouts = unpaged.data();
        layout_count = n_layers;
    }
    imparo_cuda_kv::PageTables<MAX_LAYERS> next_pages;
    const auto page_rc = imparo_cuda_kv::build_page_tables(
        n_layers, bytes, layouts, layout_count,
        preserve ? &g.kv_page_tables : nullptr, preserve, &next_pages);
    if (page_rc != imparo_cuda_kv::PagingRc::Ok) return paging_rc(page_rc);

    if (preserve) {
        if (n_layers < g.kv_layout.layers) return CUDA_RC_INVALID;
        for (uint32_t layer = 0; layer < g.kv_layout.layers; ++layer) {
            if (next.logical_bytes[layer] < g.kv_layout.logical_bytes[layer]) {
                return CUDA_RC_INVALID;
            }
        }
        const auto ownership_rc = imparo_cuda_kv::preserve_ownership(
            g.kv_layout, &next);
        if (ownership_rc != imparo_cuda_kv::OwnershipRc::Ok) {
            return ownership_rc == imparo_cuda_kv::OwnershipRc::OutOfMemory
                ? CUDA_RC_OOM : CUDA_RC_INVALID;
        }
    }

    void * fresh = nullptr;
    void * fresh_pages = nullptr;
    int alloc_rc = 0;
    if (next.arena_bytes) {
        alloc_rc = alloc_raw_with_packed_q4_reclaim(
            &fresh, next.arena_bytes, "transactional KV arena");
    }
    if (!alloc_rc && next_pages.arena_entries) {
        if (next_pages.arena_entries > SIZE_MAX / sizeof(uint32_t)) {
            alloc_rc = CUDA_RC_INVALID;
        } else {
            alloc_rc = alloc_raw_with_packed_q4_reclaim(
                &fresh_pages, next_pages.arena_entries * sizeof(uint32_t),
                "transactional KV page-table arena");
        }
    }
    if (alloc_rc) {
        if (fresh) cudaFree(fresh);
        if (fresh_pages) cudaFree(fresh_pages);
        return alloc_rc;
    }
    cudaError_t error = next.arena_bytes
        ? cudaMemsetAsync(fresh, 0, size_t(next.arena_bytes), g.stream)
        : cudaSuccess;
    if (error == cudaSuccess && preserve && g.kv_arena) {
        for (uint32_t layer = 0; layer < g.kv_layout.layers; ++layer) {
            const uint64_t copy_bytes = g.kv_layout.logical_bytes[layer];
            for (uint32_t is_v = 0; is_v < 2 && error == cudaSuccess; ++is_v) {
                const uint64_t old_off = is_v ? g.kv_layout.v_offset[layer]
                                              : g.kv_layout.k_offset[layer];
                const uint64_t new_off = is_v ? next.v_offset[layer]
                                              : next.k_offset[layer];
                if (!copy_bytes) continue;
                error = cudaMemcpyAsync(
                    static_cast<uint8_t *>(fresh) + new_off,
                    static_cast<const uint8_t *>(g.kv_arena) + old_off,
                    size_t(copy_bytes), cudaMemcpyDeviceToDevice, g.stream);
            }
        }
    }
    for (uint32_t layer = 0;
         layer < next_pages.layers && error == cudaSuccess; ++layer) {
        const auto & table = next_pages.layer[layer];
        if (!table.capacity) continue;
        error = cudaMemcpyAsync(
            static_cast<uint32_t *>(fresh_pages) + table.arena_offset,
            table.host_shadow.data(), size_t(table.capacity) * sizeof(uint32_t),
            cudaMemcpyHostToDevice, g.stream);
    }
    // One transaction boundary covers zero-fill, preserving D2D copies and table upload.
    if (error == cudaSuccess) error = cudaStreamSynchronize(g.stream);
    if (error != cudaSuccess) {
        cudaGetLastError();
        if (fresh) cudaFree(fresh);
        if (fresh_pages) cudaFree(fresh_pages);
        return error == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }

    destroy_decode_graph();
    void * old_arena = g.kv_arena;
    void * old_pages = g.kv_page_table_arena;
    g.kv_arena = fresh;
    g.kv_layout = std::move(next);
    g.kv_page_table_arena = fresh_pages;
    g.kv_page_tables = std::move(next_pages);
    for (uint32_t layer = 0; layer < MAX_LAYERS; ++layer) {
        if (layer < g.kv_layout.layers && g.kv_layout.logical_bytes[layer]) {
            g.kv_k[layer] = static_cast<uint8_t *>(g.kv_arena)
                + g.kv_layout.k_offset[layer];
            g.kv_v[layer] = static_cast<uint8_t *>(g.kv_arena)
                + g.kv_layout.v_offset[layer];
            g.kv_bytes[layer] = g.kv_layout.logical_bytes[layer];
        } else {
            g.kv_k[layer] = nullptr;
            g.kv_v[layer] = nullptr;
            g.kv_bytes[layer] = 0;
        }
    }
    g.kdq.clear();
    g.vdq.clear();
    g.decode_phase_active = false;
    g.decode_sequence_start_pos = UINT32_MAX;
    g.decode_previous_start_pos = UINT32_MAX;
    if (old_arena && cudaFree(old_arena) != cudaSuccess) {
        set_pending(CUDA_RC_ERROR, "retired KV arena release");
    }
    if (old_pages && cudaFree(old_pages) != cudaSuccess) {
        set_pending(CUDA_RC_ERROR, "retired KV page-table arena release");
    }
    return 0;
}

extern "C" int imparo_cuda_alloc_kv(uint32_t n_layers, const uint64_t * bytes) {
    return replace_kv_arena(n_layers, bytes, nullptr, 0, false);
}
extern "C" int imparo_cuda_grow_kv(uint32_t n_layers, const uint64_t * bytes) {
    return replace_kv_arena(n_layers, bytes, nullptr, 0, true);
}
extern "C" int imparo_cuda_alloc_kv_layout(
        uint32_t n_layers, const uint64_t * bytes,
        const imparo_cuda_kv::PagingLayout * layouts, uint32_t layout_count) {
    return replace_kv_arena(n_layers, bytes, layouts, layout_count, false);
}
extern "C" int imparo_cuda_grow_kv_layout(
        uint32_t n_layers, const uint64_t * bytes,
        const imparo_cuda_kv::PagingLayout * layouts, uint32_t layout_count) {
    return replace_kv_arena(n_layers, bytes, layouts, layout_count, true);
}

extern "C" int imparo_cuda_set_kv_pages(
        uint32_t layer, const uint32_t * entries, uint32_t n) {
    // A capture must see one immutable host scheduling decision. Stable table
    // contents may change between replays, but never while the graph is recorded.
    if (g.graph_capturing) return CUDA_RC_INVALID;
    std::vector<uint32_t> candidate;
    bool changed = false;
    const auto prepare = imparo_cuda_kv::prepare_page_update(
        g.kv_page_tables, layer, entries, n, &candidate, &changed);
    if (prepare != imparo_cuda_kv::PagingRc::Ok) return paging_rc(prepare);
    if (!changed) return 0;
    const auto & table = g.kv_page_tables.layer[layer];
    const bool mapped_before = imparo_cuda_kv::page_table_requires_mapping(
        table.host_shadow, table.capacity);
    const bool mapped_after = imparo_cuda_kv::page_table_requires_mapping(
        candidate, table.capacity);
    if (table.capacity) {
        if (!g.kv_page_table_arena) return CUDA_RC_INVALID;
        const cudaError_t copy = cudaMemcpyAsync(
            static_cast<uint32_t *>(g.kv_page_table_arena) + table.arena_offset,
            candidate.data(), size_t(table.capacity) * sizeof(uint32_t),
            cudaMemcpyHostToDevice, g.stream);
        const cudaError_t sync = copy == cudaSuccess
            ? cudaStreamSynchronize(g.stream) : copy;
        if (copy != cudaSuccess || sync != cudaSuccess) {
            cudaGetLastError();
            return CUDA_RC_ERROR;
        }
    }
    imparo_cuda_kv::commit_page_update(
        &g.kv_page_tables, layer, std::move(candidate), n);
    invalidate_kv_dequant(layer, 0);
    invalidate_kv_dequant(layer, 1);
    // Same-specialization content updates keep the stable pointer and captured graph.
    // Crossing identity/paged classes changes the kernel symbol and must recapture.
    if (mapped_before != mapped_after) destroy_decode_graph();
    return 0;
}
static int ensure_kv_stage(uint64_t bytes) {
    if (!bytes) return 0;
    uint64_t slot_bytes = 0;
    if (!imparo_cuda_kv::checked_align_up(bytes, &slot_bytes)
        || slot_bytes > SIZE_MAX / imparo_cuda_kv::kStageSlots) {
        return CUDA_RC_INVALID;
    }
    if (g.kv_stage_host && slot_bytes <= g.kv_stage_slot_bytes) return 0;

    void * fresh = nullptr;
    const cudaError_t host_rc = cudaHostAlloc(
        &fresh, size_t(slot_bytes * imparo_cuda_kv::kStageSlots),
        cudaHostAllocDefault);
    if (host_rc != cudaSuccess) {
        cudaGetLastError();
        return host_rc == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }
    cudaEvent_t events[imparo_cuda_kv::kStageSlots] = {};
    for (uint32_t slot = 0; slot < imparo_cuda_kv::kStageSlots; ++slot) {
        if (cudaEventCreateWithFlags(&events[slot], cudaEventDisableTiming)
            != cudaSuccess) {
            for (uint32_t prior = 0; prior < slot; ++prior) cudaEventDestroy(events[prior]);
            cudaFreeHost(fresh);
            cudaGetLastError();
            return CUDA_RC_ERROR;
        }
    }
    if (cudaStreamSynchronize(g.stream) != cudaSuccess) {
        for (cudaEvent_t event : events) cudaEventDestroy(event);
        cudaFreeHost(fresh);
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    for (cudaEvent_t & event : g.kv_stage_done) {
        if (event) cudaEventDestroy(event);
    }
    if (g.kv_stage_host) cudaFreeHost(g.kv_stage_host);
    g.kv_stage_host = fresh;
    g.kv_stage_slot_bytes = slot_bytes;
    g.kv_stage_next = 0;
    for (uint32_t slot = 0; slot < imparo_cuda_kv::kStageSlots; ++slot) {
        g.kv_stage_done[slot] = events[slot];
    }
    return 0;
}

static int kv_stage_slot(uint64_t bytes, uint8_t ** host, uint32_t * slot) {
    const int ensure_rc = ensure_kv_stage(bytes);
    if (ensure_rc) return ensure_rc;
    const uint32_t selected = g.kv_stage_next++ % imparo_cuda_kv::kStageSlots;
    const cudaError_t prior = cudaEventQuery(g.kv_stage_done[selected]);
    if (prior == cudaErrorNotReady) {
        if (cudaEventSynchronize(g.kv_stage_done[selected]) != cudaSuccess) {
            return CUDA_RC_ERROR;
        }
    } else if (prior != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    *host = static_cast<uint8_t *>(g.kv_stage_host)
        + uint64_t(selected) * g.kv_stage_slot_bytes;
    *slot = selected;
    return 0;
}

extern "C" void imparo_cuda_write(uint32_t id, uint64_t off, const float * src, uint64_t n) {
    cudaMemcpyAsync(static_cast<float *>(g.bufs[id]) + off, src, n * 4,
                    cudaMemcpyHostToDevice, g.stream);
    mark_buf_written(id);
}
extern "C" void imparo_cuda_write_u32(uint32_t id, uint64_t off, const uint32_t * src, uint64_t n) {
    const bool bounded = off <= SIZE_MAX && n <= SIZE_MAX - size_t(off);
    if (id < B_COUNT && src && bounded) {
        auto & shadow = g.u32_shadow[id];
        const size_t end = size_t(off + n);
        if (shadow.size() < end) shadow.resize(end);
        std::copy_n(src, size_t(n), shadow.data() + size_t(off));
    }
    const bool stable_decode = g.forward_decode && decode_graph_candidate()
        && id < B_COUNT && bounded && n == 1;
    const bool stable_prefill = !g.forward_decode && g.prefill_prepared
        && prefill_graph_enabled_for(g.prefill_graph_tokens) && id < B_COUNT && bounded
        && n == g.prefill_graph_tokens;
    const uint32_t * upload = src;
    if (stable_decode) {
        const int rc = ensure_decode_token_host_stage();
        if (rc) { set_pending(rc, "decode token pinned staging"); return; }
        *static_cast<uint32_t *>(g.decode_token_stage_host) = src[0];
        upload = static_cast<const uint32_t *>(g.decode_token_stage_host);
        g.decode_token_buf = id;
        g.decode_token_off = off;
    } else if (g.forward_decode && decode_pinned_input_enabled()
               && id < B_COUNT && bounded
               && g.u32_shadow[id].size() >= size_t(off + n)) {
        upload = g.u32_shadow[id].data() + size_t(off);
    } else if (stable_prefill) {
        g.prefill_token_buf = id;
        g.prefill_token_off = off;
    }
    if (!(g.graph_capturing && (stable_decode || stable_prefill))) {
        cudaMemcpyAsync(static_cast<uint32_t *>(g.bufs[id]) + off, upload, n * 4,
                        cudaMemcpyHostToDevice, g.stream);
    }
    mark_buf_written(id);
}
extern "C" void imparo_cuda_read(uint32_t id, uint64_t off, float * dst, uint64_t n) {
    cudaMemcpyAsync(dst, static_cast<const float *>(g.bufs[id]) + off, n * 4,
                    cudaMemcpyDeviceToHost, g.stream);
    cudaStreamSynchronize(g.stream);
}
extern "C" int imparo_cuda_write_kv(uint32_t layer, uint32_t is_v, uint64_t off,
                                    const uint8_t * src, uint64_t n) {
    if ((!src && n) || !kv_byte_range_owned(layer, is_v, off, n)) {
        return CUDA_RC_INVALID;
    }
    if (!n) return 0;
    void * b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    if (!b) return CUDA_RC_INVALID;
    uint8_t * stage = nullptr;
    uint32_t slot = 0;
    const int stage_rc = kv_stage_slot(n, &stage, &slot);
    if (stage_rc) return stage_rc;
    std::memcpy(stage, src, size_t(n));
    const cudaError_t copy = cudaMemcpyAsync(
        static_cast<uint8_t *>(b) + off, stage, size_t(n),
        cudaMemcpyHostToDevice, g.stream);
    if (copy != cudaSuccess
        || cudaEventRecord(g.kv_stage_done[slot], g.stream) != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    invalidate_kv_dequant(layer, is_v);
    return 0;
}
extern "C" int imparo_cuda_read_kv(uint32_t layer, uint32_t is_v, uint64_t off,
                                   uint8_t * dst, uint64_t n) {
    if ((!dst && n) || !kv_byte_range_owned(layer, is_v, off, n)) {
        return CUDA_RC_INVALID;
    }
    if (!n) return 0;
    const void * b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    if (!b) return CUDA_RC_INVALID;
    uint8_t * stage = nullptr;
    uint32_t slot = 0;
    const int stage_rc = kv_stage_slot(n, &stage, &slot);
    if (stage_rc) return stage_rc;
    const cudaError_t copy = cudaMemcpyAsync(
        stage, static_cast<const uint8_t *>(b) + off, size_t(n),
        cudaMemcpyDeviceToHost, g.stream);
    if (copy != cudaSuccess
        || cudaEventRecord(g.kv_stage_done[slot], g.stream) != cudaSuccess
        || cudaEventSynchronize(g.kv_stage_done[slot]) != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    std::memcpy(dst, stage, size_t(n));
    return 0;
}

static int host_rc(imparo_cuda_host::Rc rc) {
    if (rc == imparo_cuda_host::Rc::Ok) return 0;
    return rc == imparo_cuda_host::Rc::OutOfMemory
        ? CUDA_RC_OOM : CUDA_RC_INVALID;
}

static void * host_pointer(uint64_t handle, uint32_t * index_out = nullptr) {
    uint32_t index = 0;
    if (!g.host_registry.resolve(handle, &index)
        || index >= g.host_buffers.size() || !g.host_buffers[index]) {
        return nullptr;
    }
    if (index_out) *index_out = index;
    return g.host_buffers[index];
}

extern "C" int imparo_cuda_host_alloc(uint64_t bytes, uint64_t * handle_out) {
    if (g.graph_capturing || !g.stream || !bytes || bytes > SIZE_MAX
        || !handle_out) {
        return CUDA_RC_INVALID;
    }
    *handle_out = 0;
    void * fresh = nullptr;
    const cudaError_t allocation = cudaHostAlloc(
        &fresh, size_t(bytes), cudaHostAllocDefault);
    if (allocation != cudaSuccess) {
        cudaGetLastError();
        return allocation == cudaErrorMemoryAllocation
            ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }
    uint64_t handle = 0;
    uint32_t index = 0;
    const auto claim = g.host_registry.claim(bytes, &handle, &index);
    if (claim != imparo_cuda_host::Rc::Ok) {
        cudaFreeHost(fresh);
        return host_rc(claim);
    }
    try {
        if (g.host_buffers.size() <= index) {
            g.host_buffers.resize(size_t(index) + 1, nullptr);
        }
    } catch (const std::bad_alloc &) {
        g.host_registry.release(handle);
        cudaFreeHost(fresh);
        return CUDA_RC_OOM;
    } catch (const std::length_error &) {
        g.host_registry.release(handle);
        cudaFreeHost(fresh);
        return CUDA_RC_INVALID;
    }
    if (g.host_buffers[index]) {
        g.host_registry.release(handle);
        cudaFreeHost(fresh);
        return CUDA_RC_INVALID;
    }
    g.host_buffers[index] = fresh;
    *handle_out = handle;
    return 0;
}

extern "C" int imparo_cuda_host_free(uint64_t handle) {
    if (g.graph_capturing) return CUDA_RC_INVALID;
    uint32_t index = 0;
    void * pointer = host_pointer(handle, &index);
    if (!pointer) return CUDA_RC_INVALID;
    const cudaError_t released = cudaFreeHost(pointer);
    if (released != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    if (g.host_registry.release(handle) != imparo_cuda_host::Rc::Ok) {
        return CUDA_RC_INVALID;
    }
    g.host_buffers[index] = nullptr;
    return 0;
}

extern "C" int imparo_cuda_host_read(
        uint64_t handle, uint64_t off, uint8_t * dst, uint64_t len) {
    if (g.graph_capturing || !dst || !len || len > SIZE_MAX) {
        return CUDA_RC_INVALID;
    }
    const auto * allocation = g.host_registry.resolve(handle);
    void * pointer = host_pointer(handle);
    if (!allocation || !pointer || off > allocation->bytes
        || len > allocation->bytes - off) {
        return CUDA_RC_INVALID;
    }
    std::memcpy(dst, static_cast<const uint8_t *>(pointer) + off, size_t(len));
    return 0;
}

extern "C" int imparo_cuda_host_write(
        uint64_t handle, uint64_t off, const uint8_t * src, uint64_t len) {
    if (g.graph_capturing || !src || !len || len > SIZE_MAX) {
        return CUDA_RC_INVALID;
    }
    const auto * allocation = g.host_registry.resolve(handle);
    void * pointer = host_pointer(handle);
    if (!allocation || !pointer || off > allocation->bytes
        || len > allocation->bytes - off) {
        return CUDA_RC_INVALID;
    }
    std::memcpy(static_cast<uint8_t *>(pointer) + off, src, size_t(len));
    return 0;
}

extern "C" int imparo_cuda_host_allocated_bytes(uint64_t * out) {
    if (g.graph_capturing || !out) return CUDA_RC_INVALID;
    *out = g.host_registry.live_bytes();
    return 0;
}

static int prepare_host_transfer(
        const imparo_cuda_host::TransferSpan * spans, uint32_t count,
        std::vector<imparo_cuda_host::PlannedSpan> * plan,
        uint64_t * total) {
    if (g.graph_capturing || !g.stream || !plan || !total) {
        return CUDA_RC_INVALID;
    }
    const auto planned = imparo_cuda_host::build_plan(
        g.host_registry, spans, count, plan, total);
    if (planned != imparo_cuda_host::Rc::Ok) return host_rc(planned);
    for (const auto & item : *plan) {
        const auto & span = item.span;
        if (item.host_index >= g.host_buffers.size()
            || !g.host_buffers[item.host_index]
            || !imparo_cuda_kv::range_live(
                g.kv_layout, span.layer, span.is_v,
                span.device_offset, span.len)) {
            return CUDA_RC_INVALID;
        }
    }
    return 0;
}

extern "C" int imparo_cuda_kv_demote(
        const imparo_cuda_host::TransferSpan * spans, uint32_t count) {
    std::vector<imparo_cuda_host::PlannedSpan> plan;
    uint64_t total = 0;
    const int prepare = prepare_host_transfer(spans, count, &plan, &total);
    if (prepare) return prepare;

    // Copy into an unobservable pinned transaction buffer. The persistent Host
    // allocation changes only after every D2H transfer and the one batch sync
    // succeed; device bytes and common-pool ownership are never modified here.
    void * staging = nullptr;
    const cudaError_t allocated = cudaHostAlloc(
        &staging, size_t(total), cudaHostAllocDefault);
    if (allocated != cudaSuccess) {
        cudaGetLastError();
        return allocated == cudaErrorMemoryAllocation
            ? CUDA_RC_OOM : CUDA_RC_ERROR;
    }
    cudaError_t copy = cudaSuccess;
    for (const auto & item : plan) {
        const auto & span = item.span;
        const void * device = span.is_v ? g.kv_v[span.layer]
                                        : g.kv_k[span.layer];
        copy = cudaMemcpyAsync(
            static_cast<uint8_t *>(staging) + item.staging_offset,
            static_cast<const uint8_t *>(device) + span.device_offset,
            size_t(span.len), cudaMemcpyDeviceToHost, g.stream);
        if (copy != cudaSuccess) break;
    }
    const cudaError_t sync = cudaStreamSynchronize(g.stream);
    if (copy == cudaSuccess && sync == cudaSuccess) {
        for (const auto & item : plan) {
            std::memcpy(
                static_cast<uint8_t *>(g.host_buffers[item.host_index])
                    + item.span.host_offset,
                static_cast<const uint8_t *>(staging) + item.staging_offset,
                size_t(item.span.len));
        }
    }
    const cudaError_t freed = cudaFreeHost(staging);
    if (copy != cudaSuccess || sync != cudaSuccess || freed != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    return 0;
}

extern "C" int imparo_cuda_kv_promote(
        const imparo_cuda_host::TransferSpan * spans, uint32_t count) {
    std::vector<imparo_cuda_host::PlannedSpan> plan;
    uint64_t total = 0;
    const int prepare = prepare_host_transfer(spans, count, &plan, &total);
    if (prepare) return prepare;

    // Promotion targets were freshly claimed through the common lifecycle drain
    // before this call. Host is authoritative until this one batch sync succeeds;
    // on failure the caller emits Free for every fresh placement. Copying stale,
    // recycled destination bytes back to Host would double PCIe traffic and is not
    // a meaningful rollback, so this transfer-only API never changes Host or
    // native ownership metadata.
    cudaError_t copy = cudaSuccess;
    for (const auto & item : plan) {
        const auto & span = item.span;
        void * device = span.is_v ? g.kv_v[span.layer]
                                  : g.kv_k[span.layer];
        copy = cudaMemcpyAsync(
            static_cast<uint8_t *>(device) + span.device_offset,
            static_cast<const uint8_t *>(g.host_buffers[item.host_index])
                + span.host_offset,
            size_t(span.len), cudaMemcpyHostToDevice, g.stream);
        if (copy != cudaSuccess) break;
    }
    const cudaError_t sync = cudaStreamSynchronize(g.stream);
    if (copy != cudaSuccess || sync != cudaSuccess) {
        cudaGetLastError();
        return CUDA_RC_ERROR;
    }
    for (const auto & item : plan) {
        invalidate_kv_dequant(item.span.layer, item.span.is_v);
    }
    return 0;
}

static int kv_advise(uint32_t layer, uint32_t is_v, uint64_t off,
                     uint64_t len, bool reuse) {
    const auto rc = imparo_cuda_kv::transition(
        &g.kv_layout, layer, is_v, off, len, reuse);
    if (rc != imparo_cuda_kv::OwnershipRc::Ok) {
        return rc == imparo_cuda_kv::OwnershipRc::OutOfMemory
            ? CUDA_RC_OOM : CUDA_RC_INVALID;
    }
    invalidate_kv_dequant(layer, is_v);
    destroy_decode_graph();
    return 0;
}

extern "C" int imparo_cuda_kv_advise_free(
        uint32_t layer, uint32_t is_v, uint64_t off, uint64_t len) {
    return kv_advise(layer, is_v, off, len, false);
}

extern "C" int imparo_cuda_kv_advise_reuse(
        uint32_t layer, uint32_t is_v, uint64_t off, uint64_t len) {
    return kv_advise(layer, is_v, off, len, true);
}

extern "C" int imparo_cuda_kv_live_bytes(uint64_t * out) {
    if (!out) return CUDA_RC_INVALID;
    *out = g.kv_layout.live_bytes;
    return 0;
}
extern "C" void imparo_cuda_set_epilogue(uint32_t on) {
    // CUDA's existing fused kernels implement GELU only. A nonzero test used to
    // make SILU (wire value 2) silently execute GELU; reject it before any launch.
    if (on > 1) {
        set_pending(CUDA_RC_INVALID, "unsupported fused epilogue");
        return;
    }
    g.epilogue = on;
}
extern "C" void imparo_cuda_set_knob(uint32_t idx, uint32_t v) {
    if (idx < 64) {
        if (g.knobs[idx] != v) {
            destroy_decode_graph();
            g.decode_graph_shape_blocked = false;
            g.decode_graph_capture_after = 0;
            ++g.choice_epoch;
            invalidate_q8_cache();
            if (!g.choice_epoch) ++g.choice_epoch;
        }
        g.knobs[idx] = v;
    }
}
extern "C" uint32_t imparo_cuda_knob(uint32_t idx) {
    return tuner_knob(idx);
}
extern "C" uint32_t imparo_cuda_buf_count(void) { return uint32_t(B_COUNT); }
extern "C" void imparo_cuda_set_kv_types(uint32_t k, uint32_t v) {
    if (g.kv_type_k != k || g.kv_type_v != v) {
        destroy_decode_graph();
        g.decode_graph_shape_blocked = false;
        g.decode_graph_capture_after = 0;
        g.kdq.clear();
        g.vdq.clear();
    }
    g.kv_type_k = k; g.kv_type_v = v;
}

extern "C" void imparo_cuda_matmat(uint32_t wkind, uint64_t w_off, uint32_t n_in,
                                    uint32_t n_out, uint32_t src, uint32_t dst,
                                    uint32_t n_tok, uint32_t src_row) {
    if (src >= B_COUNT || dst >= B_COUNT || !g.bufs[src] || !g.bufs[dst]
        || !n_in || !n_out || !n_tok
        || (wkind != 0 && wkind != 1 && wkind != 2 && wkind != 3)) {
        set_pending(CUDA_RC_INVALID, "matmat arguments");
        return;
    }
    mark_tuner_dispatch(1, (uint64_t(wkind) << 32) | n_tok);
    const float * x = static_cast<const float *>(g.bufs[src]);
    float * y = static_cast<float *>(g.bufs[dst]);
    if (wkind != 0 && n_in % 32) {
        set_pending(CUDA_RC_INVALID, "quantized matmat width");
        return;
    }
    const bool q8_tm_silu_requested = wkind == 3
        && (g.epilogue == 2 || g.epilogue == 4)
        && tuner_knob(43) != 0 && tuner_knob(42) != 0
        && g.sm_version == 86 && n_tok > 8 && src_row == 0
        && n_in % 256 == 0
        && n_out % imparo_sm80_mmq::kRows == 0;
    const uint32_t q8_tm_silu_sidecar_min_tokens =
        tuner_knob(g.epilogue == 4 ? 45 : 44);
    const bool q8_tm_silu_sidecar_requested = q8_tm_silu_requested
        && q8_tm_silu_sidecar_min_tokens != 0
        && n_tok >= q8_tm_silu_sidecar_min_tokens
        && std::getenv("IMPARO_CUDA_NO_Q8_TM_SILU_Q8_SIDECAR") == nullptr;
    if ((wkind == 2 || wkind == 3) && g.epilogue != 0
            && !q8_tm_silu_requested) {
        set_pending(CUDA_RC_INVALID, "Q8_0 fused epilogue is unsupported");
        return;
    }
    if (wkind == 3 && n_out % 8 != 0) {
        set_pending(CUDA_RC_INVALID, "Q8_0_TM output rows must be tile-aligned");
        return;
    }
    MatmatEventScope profile(wkind, n_in, n_out, n_tok, g.epilogue);
    const uint64_t row_bytes = wkind == 0 ? uint64_t(n_in) * 4
        : uint64_t(n_in / 32) * (wkind == 1 ? 18 : 34);
    if (uint64_t(n_out) > UINT64_MAX / row_bytes) {
        set_pending(CUDA_RC_INVALID, "matmat weight range overflow");
        return;
    }
    const uint64_t tensor_bytes = row_bytes * n_out;
    // `weights_resident` means the hot subset was uploaded, not that this particular
    // tensor is resident. Streamed tensors remain bounded by the page cache.
    const bool tensor_resident = resident_weight_range(w_off, tensor_bytes) != nullptr;
    if (q8_tm_silu_requested && !tensor_resident) {
        set_pending(CUDA_RC_INVALID, "Q8_0_TM SiLU epilogue requires resident weights");
        return;
    }
    const uint64_t slice_limit = tensor_resident ? tensor_bytes : g.weight_cache_limit;
    if (row_bytes > slice_limit) {
        set_pending(CUDA_RC_OOM, "one weight row exceeds paged cache limit");
        return;
    }
    uint64_t row_capacity = std::max<uint64_t>(1, slice_limit / row_bytes);
    if (wkind == 3) {
        if (row_capacity < 8) {
            set_pending(CUDA_RC_OOM,
                        "Q8_0_TM cache cannot hold one 8-row weight tile");
            return;
        }
        row_capacity -= row_capacity % 8;
    }
    const uint32_t rows_per_slice = uint32_t(std::min<uint64_t>(
        n_out, row_capacity));
    const BlockQ8_1 * projection_q8 = nullptr;
    const BlockQ8_1Mmq * projection_q8_mmq = nullptr;
    bool produced_q8 = false;
    uint32_t produced_q8_layout = Q8_LAYOUT_MMQ;
    uint32_t gemv_warps = 4;
    // CUDA decode follows llama's Q8_1-activation MMVQ path by default. The direct-f32
    // kernel remains only as an explicit diagnostic/localization fallback.
    static const bool force_f32_gemv = std::getenv("IMPARO_CUDA_GEMV_F32") != nullptr;
    // Q8_0 uses the same Q8_1 activation contract as llama CUDA for decode and
    // short microbatches, then the Ampere MMQ contract above that boundary.
    // Keep the former f32-activation route only as an explicit localization
    // fallback or for architectures/alignments without a compiled MMQ route.
    const bool q8_force_f32 =
        std::getenv("IMPARO_CUDA_Q8_F32") != nullptr;
    const bool q8_weight = wkind == 2 || wkind == 3;
    const bool q8_mmvq = q8_weight && g.sm_version >= 80
        && n_tok <= 8 && !q8_force_f32;
    bool q8_mmq = q8_weight && g.sm_version >= 80 && n_tok > 8
        && n_in % 128 == 0
        && (g.epilogue == 0 || q8_tm_silu_requested) && !q8_force_f32;
    imparo_sm80_q8_replay::ReplayPlan q8_mmq_plan{};
    const bool kv_large_all = ((dst == 3 || dst == 4)
            && std::getenv("IMPARO_CUDA_KV_PROJ_LARGE_ALL") != nullptr)
        || (dst == 3 && std::getenv("IMPARO_CUDA_K_PROJ_LARGE_ALL") != nullptr)
        || (dst == 4 && std::getenv("IMPARO_CUDA_V_PROJ_LARGE_ALL") != nullptr);
    const bool kv_virtual512 = n_tok > 1 && g.batch_geometry_valid
        && g.batch_geometry_tokens == n_tok && (
        ((dst == 3 || dst == 4)
            && std::getenv("IMPARO_CUDA_KV_VIRTUAL512") != nullptr)
        || (dst == 3 && std::getenv("IMPARO_CUDA_K_VIRTUAL512") != nullptr)
        || (dst == 4 && std::getenv("IMPARO_CUDA_V_VIRTUAL512") != nullptr)
        || ((dst == 3 || dst == 4) && n_out == 512
            && std::getenv("IMPARO_CUDA_KV_VIRTUAL512_OUT512") != nullptr)
        || ((dst == 3 || dst == 4) && n_out == 1024
            && std::getenv("IMPARO_CUDA_KV_VIRTUAL512_OUT1024") != nullptr));
    const bool kv_mmvq4 = wkind == 1 && n_tok > 1 && n_tok % 4 == 0
        && (((dst == 3 || dst == 4)
                && std::getenv("IMPARO_CUDA_KV_MMVQ4_ALL") != nullptr)
            || std::getenv("IMPARO_CUDA_MMVQ4_ALL") != nullptr);
    static const uint32_t mmvq_microbatch_dst_mask = [] {
        const char * raw = std::getenv("IMPARO_CUDA_MMVQ_MICROBATCH_DST_MASK");
        return raw ? uint32_t(std::strtoul(raw, nullptr, 0)) : 0u;
    }();
    const bool mmvq_microbatch = wkind == 1 && n_tok > 1 && !g.epilogue
        && (((dst == 3 || dst == 4)
                && std::getenv("IMPARO_CUDA_KV_MMVQ_MICROBATCH") != nullptr)
            || (dst < 32 && (mmvq_microbatch_dst_mask & (1u << dst)) != 0));
    const bool mmvq_batched = kv_mmvq4 || mmvq_microbatch;
    const bool kv_tail4 = wkind == 1 && n_tok > 4
        && (dst == 3 || dst == 4)
        && std::getenv("IMPARO_CUDA_KV_TAIL4_MMVQ") != nullptr;
    const bool small_mmvq = wkind == 1 && g.sm_version >= 80
        && n_tok <= imparo_sm80_mmvq::kMaxTokens
        && !kv_large_all && !kv_virtual512 && !mmvq_batched
        && !std::getenv("IMPARO_CUDA_NO_MMVQ_SMALL");
    // Diagnostic projection-family selector. Low 32 bits address plain writes by
    // destination BufId; high 32 bits address fused-epilogue writes. Keeping the
    // epilogue class in the key prevents a Q/G/O experiment from silently changing
    // the numerically distinct fused FFN-up route that happens to share a buffer.
    static const uint64_t batch_dp4a_dst_mask = [] {
        const char * raw = std::getenv("IMPARO_CUDA_BATCH_DP4A_DST_MASK");
        return raw ? uint64_t(std::strtoull(raw, nullptr, 0)) : 0ULL;
    }();
    const uint32_t batch_dp4a_bit = dst < 32
        ? dst + (g.epilogue ? 32u : 0u) : 64u;
    const bool batch_dp4a = std::getenv("IMPARO_CUDA_BATCH_DP4A") != nullptr
        || (batch_dp4a_bit < 64
            && (batch_dp4a_dst_mask & (1ULL << batch_dp4a_bit)) != 0)
        || ((dst == 3 || dst == 4)
            && std::getenv("IMPARO_CUDA_KV_PROJ_DP4A") != nullptr);
    const bool kv_mma16 = ((dst == 3 || dst == 4)
            && std::getenv("IMPARO_CUDA_KV_PROJ_MMA16") != nullptr)
        || (dst == 3 && std::getenv("IMPARO_CUDA_K_PROJ_MMA16") != nullptr)
        || (dst == 4 && std::getenv("IMPARO_CUDA_V_PROJ_MMA16") != nullptr);
    const bool kv_fixed8 = (dst == 3 || dst == 4)
        && std::getenv("IMPARO_CUDA_KV_FIXED8") != nullptr;
    uint32_t kv_virtual_workers = uint32_t(g.sm_count);
    if (kv_virtual512) {
        if (const char * forced = std::getenv("IMPARO_CUDA_KV_VIRTUAL_WORKERS")) {
            const uint32_t parsed = uint32_t(std::strtoul(forced, nullptr, 10));
            if (parsed >= 1 && parsed <= 128) kv_virtual_workers = parsed;
        }
        const char * side_workers = std::getenv(dst == 3
            ? "IMPARO_CUDA_K_VIRTUAL_WORKERS"
            : "IMPARO_CUDA_V_VIRTUAL_WORKERS");
        if (side_workers) {
            const uint32_t parsed = uint32_t(std::strtoul(side_workers, nullptr, 10));
            if (parsed >= 1 && parsed <= 128) kv_virtual_workers = parsed;
        }
        const char * shape_workers = nullptr;
        if (dst == 3 && n_out == 512) shape_workers = std::getenv("IMPARO_CUDA_K_VIRTUAL_WORKERS_OUT512");
        if (dst == 3 && n_out == 1024) shape_workers = std::getenv("IMPARO_CUDA_K_VIRTUAL_WORKERS_OUT1024");
        if (dst == 4 && n_out == 512) shape_workers = std::getenv("IMPARO_CUDA_V_VIRTUAL_WORKERS_OUT512");
        if (dst == 4 && n_out == 1024) shape_workers = std::getenv("IMPARO_CUDA_V_VIRTUAL_WORKERS_OUT1024");
        if (shape_workers) {
            const uint32_t parsed = uint32_t(std::strtoul(shape_workers, nullptr, 10));
            if (parsed >= 1 && parsed <= 128) kv_virtual_workers = parsed;
        }
    }
    const bool large_mmq_candidate = wkind == 1 && g.sm_version >= 80
        && (n_tok > imparo_sm80_mmvq::kMaxTokens || kv_large_all)
        && n_in % 128 == 0
        && !batch_dp4a && !kv_mma16 && !kv_fixed8 && !kv_virtual512 && !mmvq_batched;
    const uint32_t projection_logical_tiles =
        (n_out / imparo_sm80_mmq::kRows)
        * ((n_tok + imparo_sm80_mmq::kTokens - 1)
            / imparo_sm80_mmq::kTokens);
    const uint32_t projection_waves = projection_logical_tiles
        ? (projection_logical_tiles + uint32_t(g.sm_count) - 1)
            / uint32_t(g.sm_count)
        : 0;
    const uint32_t projection_efficiency = projection_waves
        ? 100u * projection_logical_tiles
            / (uint32_t(g.sm_count) * projection_waves)
        : 0;
    const bool q8_ready_projection_physical_lab = std::getenv(
        "IMPARO_CUDA_PREFILL_PROJECTION_Q8_READY_PHYSICAL_LAB") != nullptr
        && projection_efficiency >= 90u;
    const bool down_projection_shape = n_in == 4 * n_out;
    const bool down_q8_ready_pack_lab =
        large_mmq_candidate && g.sm_version == 86 && tensor_resident
        && n_tok > 8 && n_tok <= 512 && src_row == 0 && g.epilogue == 0
        && n_out % 128 == 0
        && ((down_projection_shape
                && (std::getenv(
                        "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_PACK_LAB")
                        != nullptr
                    || std::getenv(
                        "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_SEAM_LAB")
                        != nullptr
                    || std::getenv(
                        "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_PHYSICAL_LAB")
                        != nullptr))
            || (!down_projection_shape
                && q8_ready_projection_physical_lab));
    if (down_q8_ready_pack_lab) {
        using ReadyLayout =
            imparo_q8_mma_ready_a0_v1_authority_lab::Layout;
        ReadyLayout layout{};
        const uint32_t padded_tokens = ((n_tok + 127u) / 128u) * 128u;
        if (imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                n_in, padded_tokens, &layout)) {
            const uint64_t quant_bytes =
                layout.quant_u16_count * sizeof(uint16_t);
            const uint64_t scale_bytes =
                layout.scale_count * sizeof(float);
            const int scratch_rc =
                ensure_q8_scratch(quant_bytes + scale_bytes);
            if (scratch_rc) {
                set_pending(
                    scratch_rc, "Q8-ready prefill down activation scratch");
                return;
            }
            auto * quant_u16 = static_cast<uint16_t *>(g.q8_scratch);
            auto * d8_sideplane = reinterpret_cast<float *>(
                static_cast<uint8_t *>(g.q8_scratch) + quant_bytes);
            ReadyLayout actual = layout;
            actual.n_tok = n_tok;
            const bool fused_pair_q8_ready = std::getenv(
                "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_Q8_LAB")
                != nullptr || std::getenv(
                    "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_Q8_PREFETCH_LAB")
                != nullptr;
            const bool fused_pair_q8_consume = fused_pair_q8_ready
                && std::getenv(
                    "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_Q8_NO_CONSUME")
                    == nullptr;
            // Laboratory bridge for a producer that already published the exact
            // MMA-ready layout (for example fused RMSNorm -> Q8).  The established
            // projection route deliberately requantizes unless this ownership
            // experiment is explicit; a cache hit must still satisfy pointer,
            // epoch, shape, row and layout identity in q8_cache_matches().
            const bool projection_q8_cache_reuse =
                q8_ready_projection_physical_lab
                && std::getenv(
                    "IMPARO_CUDA_PREFILL_PROJECTION_Q8_READY_CACHE_REUSE_LAB")
                    != nullptr;
            const bool q8_hit =
                (!q8_ready_projection_physical_lab || fused_pair_q8_consume
                    || projection_q8_cache_reuse)
                && q8_cache_matches(
                    src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
            const bool quantized = q8_hit
                || imparo_q8_mma_ready_a2_pairwarp16_research_lab::
                    launch_quantize_q8_mma_ready_ds4_a2_pairwarp16(
                        x, quant_u16, d8_sideplane, n_in, n_tok, g.stream,
                        &actual);
            if (!q8_hit && quantized) {
                own_q8_cache(src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
            }
            const uint8_t * raw = nullptr;
            int rc = weight_slice(w_off, tensor_bytes, &raw);
            if (rc) {
                set_pending(rc, "Q8-ready prefill down weights");
                return;
            }
            const bool packed_ready =
                ensure_aligned_packed_q4_lab(raw, n_in, n_out) != nullptr;
            const PackedQ4Span * packed = packed_ready
                ? find_aligned_packed_q4(raw, n_in, n_out) : nullptr;
            if (quantized && packed) {
                MatmatEventScope down_profile(
                    wkind, n_in, n_out, n_tok, 10);
                const uint64_t records = uint64_t(n_out) * (n_in / 32);
                const auto * scales = reinterpret_cast<const uint16_t *>(
                    packed->packed);
                const auto * nibbles = reinterpret_cast<const uint4 *>(
                    packed->packed + records * sizeof(uint16_t));
                const bool exact_seam = std::getenv(
                    "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_SEAM_LAB") != nullptr;
                const bool physical = std::getenv(
                    "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_PHYSICAL_LAB")
                    != nullptr || q8_ready_projection_physical_lab;
                const bool physical_r64 = down_projection_shape && physical
                    && std::getenv(
                        "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_PHYSICAL_R64_LAB")
                        != nullptr;
                const uint32_t physical_tile_rows = physical_r64
                    ? imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        kReadyCompactRows
                    : imparo_sm80_mmq::kRows;
                const uint32_t logical_tiles =
                    (n_out / physical_tile_rows)
                    * ((n_tok + imparo_sm80_mmq::kTokens - 1)
                        / imparo_sm80_mmq::kTokens);
                const uint32_t waves =
                    (logical_tiles + uint32_t(g.sm_count) - 1)
                    / uint32_t(g.sm_count);
                const uint32_t efficiency =
                    100u * logical_tiles / (uint32_t(g.sm_count) * waves);
                uint32_t tuned_efficiency = g.knobs[12];
                if (!tuned_efficiency) {
                    if (const char * forced = std::getenv(
                            "IMPARO_CUDA_MMQ_FULL_TILE_MIN_EFFICIENCY")) {
                        const uint32_t parsed = uint32_t(
                            std::strtoul(forced, nullptr, 10));
                        if (parsed >= 50 && parsed <= 100) {
                            tuned_efficiency = parsed;
                        }
                    }
                }
                const uint32_t route_efficiency = g.knobs[25]
                    ? 90u
                    : imparo_sm80_mmq::select_full_tile_min_efficiency(
                        uint32_t(g.sm_version), tuned_efficiency);
                uint32_t stream_grid = efficiency >= route_efficiency
                    ? logical_tiles : uint32_t(g.sm_count);
                if (const char * forced = std::getenv(
                        "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_STREAM_GRID_LAB")) {
                    const uint32_t parsed =
                        uint32_t(std::strtoul(forced, nullptr, 10));
                    if (parsed >= uint32_t(g.sm_count)
                            && parsed <= logical_tiles) {
                        stream_grid = parsed;
                    }
                }
                const uint64_t fixup_bytes = uint64_t(stream_grid)
                    * imparo_sm80_mmq::kTokens
                    * physical_tile_rows * sizeof(float);
                float * fixup = physical && ensure_attention_scratch(fixup_bytes)
                    ? static_cast<float *>(g.attention_scratch) : nullptr;
                const bool launched = physical
                    ? fixup && (physical_r64
                        ? imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_physical_stream_r64(
                                scales, nibbles, quant_u16, d8_sideplane,
                                actual.token_tiles, y, fixup, n_in, n_out,
                                n_tok, n_tok, n_out, stream_grid, g.stream)
                        : imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_physical_stream(
                            scales, nibbles, quant_u16, d8_sideplane,
                            actual.token_tiles, y, fixup, n_in, n_out,
                            n_tok, n_tok, n_out, stream_grid, g.stream))
                    : exact_seam
                    ? imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        launch_q8_ready_seamed(
                            scales, nibbles, quant_u16, d8_sideplane,
                            actual.token_tiles, y, n_in, n_out,
                            n_tok, n_tok, n_out,
                            uint32_t(g.q8_mmq_limits.max_grid_x), g.stream)
                    : imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        launch_q8_ready_direct(
                            scales, nibbles, quant_u16, d8_sideplane,
                            actual.token_tiles, y, n_in, n_out,
                            n_tok, n_tok, n_out, false, g.stream);
                if (launched) {
                    mark_buf_written(dst);
                    return;
                }
            }
            // The research layout is not a canonical Q8 cache entry. Clear any
            // sticky launch failure; the established path below will re-quantize.
            cudaGetLastError();
        }
    }
    float * q8_replay_prefix = nullptr;
    if (q8_mmq) {
        q8_mmq_plan = imparo_sm80_q8_replay::canonical_whole_k_plan(
            imparo_sm80_q8_replay::make_plan(
                n_in, n_out, n_tok, uint32_t(g.sm_count), g.q8_mmq_limits));
        bool workspace_available = true;
        if (q8_mmq_plan.needs_workspace) {
            uint64_t bytes = 0;
            const uint32_t max_slice_rows = std::min(rows_per_slice, n_out);
            workspace_available =
                std::getenv("IMPARO_CUDA_Q8_REPLAY_FORCE_OOM") == nullptr
                && imparo_sm80_q8_replay::slice_workspace_bytes(
                    q8_mmq_plan, max_slice_rows, &bytes)
                && ensure_attention_scratch(bytes);
            if (workspace_available) {
                q8_replay_prefix = static_cast<float *>(g.attention_scratch);
            } else {
                (void)cudaGetLastError();
            }
        }
        const imparo_sm80_q8_replay::ExecutionRoute route =
            imparo_sm80_q8_replay::choose_execution(
                q8_mmq_plan, workspace_available);
        if (route == imparo_sm80_q8_replay::ExecutionRoute::Reject) {
            set_pending(CUDA_RC_INVALID, "Q8 MMQ replay plan");
            return;
        }
        q8_mmq = route == imparo_sm80_q8_replay::ExecutionRoute::Mmq;
        if (q8_mmq) {
            const uint64_t records = uint64_t(n_tok) * (n_in / 32);
            const uint64_t bytes = records <= UINT64_MAX / sizeof(BlockQ8_1)
                ? records * sizeof(BlockQ8_1) : UINT64_MAX;
            const int scratch_rc = bytes == UINT64_MAX
                ? CUDA_RC_INVALID
                : (std::getenv("IMPARO_CUDA_Q8_SCRATCH_FORCE_OOM")
                    ? CUDA_RC_OOM : ensure_q8_scratch(bytes));
            if (scratch_rc == CUDA_RC_OOM) {
                (void)cudaGetLastError();
                q8_mmq = false;
                q8_replay_prefix = nullptr;
            } else if (scratch_rc) {
                set_pending(scratch_rc, "Q8_1 projection scratch preflight");
                return;
            }
        }
    }
    if ((wkind == 1 && !force_f32_gemv) || q8_mmvq || q8_mmq) {
        const uint64_t q8_records = uint64_t(n_tok) * (n_in / 32);
        if (q8_records > UINT64_MAX / sizeof(BlockQ8_1)) {
            set_pending(CUDA_RC_INVALID, "Q8_1 projection scratch overflow");
            return;
        }
        const uint64_t q8_bytes = q8_records * sizeof(BlockQ8_1);
        int rc = ensure_q8_scratch(q8_bytes);
        if (rc) { set_pending(rc, "Q8_1 projection scratch"); return; }
        // Quantizer/layout identity is part of the transient-cache key. MMQ's
        // packed activation tile, MMVQ's row-major blocks, and the generic
        // fallback are numerically distinct producers even when their byte
        // counts and logical shapes are identical.
        const uint32_t q8_layout = q8_mmq ? Q8_LAYOUT_MMQ_D4
            : (large_mmq_candidate ? Q8_LAYOUT_MMQ
            : ((small_mmvq || mmvq_batched || q8_mmvq) ? Q8_LAYOUT_MMVQ
                : Q8_LAYOUT_GENERIC));
        const bool q8_hit = q8_cache_matches(src, n_in, n_tok, src_row, q8_layout);
        if (large_mmq_candidate || q8_mmq) {
            projection_q8_mmq = static_cast<const BlockQ8_1Mmq *>(g.q8_scratch);
            if (!q8_hit) {
                OpEventScope quantize_profile("quantize_q8_mmq", n_in, n_tok);
                k_quantize_q8_1_mmq<<<dim3(n_tok, (n_in + 511) / 512), 128, 0,
                    g.stream>>>(x, static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                                n_in, n_tok, src_row, q8_mmq);
            }
        } else if (small_mmvq || mmvq_batched || q8_mmvq) {
            projection_q8 = static_cast<const BlockQ8_1 *>(g.q8_scratch);
            if (!q8_hit) {
                imparo_sm80_mmvq::quantize_q8_1
                    <<<dim3((n_in + 255) / 256, n_tok), 256, 0, g.stream>>>(
                        x, static_cast<BlockQ8_1 *>(g.q8_scratch), n_in, n_tok, src_row);
            }
        } else {
            projection_q8 = static_cast<const BlockQ8_1 *>(g.q8_scratch);
            if (!q8_hit) {
                k_quantize_q8_1<<<dim3(n_tok, (n_in + 511) / 512), 128, 0, g.stream>>>(
                    x, static_cast<BlockQ8_1 *>(g.q8_scratch), n_in, n_tok, src_row);
            }
        }
        if (!q8_hit) own_q8_cache(src, n_in, n_tok, src_row, q8_layout);
        trace_q8_cache(q8_hit, src, n_in, n_tok, src_row, q8_layout);
        gemv_warps = decode_warps(n_in);
    }
    if (q8_mmvq && projection_q8 && (wkind == 2 || wkind == 3)
            && n_tok == 1 && src_row == 0 && g.epilogue == 0
            && g.sm_version == 86) {
        if (const DecodeQ4ShadowSpan * shadow =
                find_decode_q4_shadow(w_off, n_in, n_out)) {
            if (shadow->q5 && shadow->tile_major) {
                imparo_sm80_mmvq::launch_q5_tile_major_decode(
                    shadow->q4, projection_q8, y, n_in, n_out, 0,
                    g.stream);
            } else if (shadow->q5) {
                imparo_sm80_mmvq::launch_q5_decode(
                    shadow->q4, projection_q8, y, n_in, n_out, 0,
                    g.stream);
            } else if (shadow->tile_major) {
                imparo_sm80_mmvq::launch_tile_major_decode(
                    shadow->q4, projection_q8, y, n_in, n_out, 0,
                    g.stream);
            } else {
                imparo_sm80_mmvq::launch_decode(
                    shadow->q4, projection_q8, y, n_in, n_out, 0,
                    gemv_warps, decode_rows_per_cta(n_in), g.stream);
            }
            mark_buf_written(dst);
            return;
        }
    }
    BlockQ8_1Mmq * q8_tm_silu_sidecar = nullptr;
    if (q8_tm_silu_sidecar_requested) {
        const uint64_t records = uint64_t(n_out / 128) * n_tok;
        if (records <= UINT64_MAX / sizeof(BlockQ8_1Mmq)
                && ensure_q8_scratch_next(
                    records * sizeof(BlockQ8_1Mmq))) {
            q8_tm_silu_sidecar = static_cast<BlockQ8_1Mmq *>(
                g.q8_scratch_next);
        }
    }
    const bool down_aligned_pack_lab =
        wkind == 1 && projection_q8_mmq && g.sm_version == 86
        && tensor_resident && n_tok > 8 && n_tok <= 512
        && src_row == 0 && g.epilogue == 0
        && n_in == 4 * n_out && n_in % 128 == 0 && n_out % 128 == 0
        && std::getenv("IMPARO_CUDA_PREFILL_DOWN_ALIGNED_PACK_LAB")
            != nullptr;
    if (down_aligned_pack_lab) {
        const uint8_t * raw = nullptr;
        int rc = weight_slice(w_off, tensor_bytes, &raw);
        if (rc) {
            set_pending(rc, "aligned-pack prefill down weights");
            return;
        }
        const bool packed_ready =
            ensure_aligned_packed_q4_lab(raw, n_in, n_out) != nullptr;
        const PackedQ4Span * packed = packed_ready
            ? find_aligned_packed_q4(raw, n_in, n_out) : nullptr;
        if (packed) {
            MatmatEventScope down_profile(
                wkind, n_in, n_out, n_tok, 9);
            const uint64_t records = uint64_t(n_out) * (n_in / 32);
            const auto * scales = reinterpret_cast<const uint16_t *>(
                packed->packed);
            const auto * nibbles = reinterpret_cast<const uint4 *>(
                packed->packed + records * sizeof(uint16_t));
            if (imparo_sm86_q4_aligned_prepack::launch_aligned_q4(
                    scales, nibbles, projection_q8_mmq, y,
                    n_in, n_out, n_tok, n_tok, n_out, 0, false, g.stream)) {
                mark_buf_written(dst);
                return;
            }
            cudaGetLastError();
        }
    }
    if (wkind == 3 && projection_q8_mmq && g.sm_version == 86
            && tensor_resident && n_tok > 8 && src_row == 0
            && g.epilogue == 0) {
        if (const PrefillQ4ShadowSpan * shadow =
                find_prefill_q4_shadow(w_off, n_in, n_out)) {
            MatmatEventScope down_profile(wkind, n_in, n_out, n_tok, 13);
            const uint64_t records = uint64_t(n_out) * (n_in / 32);
            const auto * scales = reinterpret_cast<const uint16_t *>(shadow->q4);
            const auto * nibbles = reinterpret_cast<const uint4 *>(
                shadow->q4 + records * sizeof(uint16_t));
            if (imparo_sm86_q4_aligned_prepack::launch_aligned_q4(
                    scales, nibbles, projection_q8_mmq, y,
                    n_in, n_out, n_tok, n_tok, n_out, 0, false, g.stream)) {
                mark_buf_written(dst);
                return;
            }
            cudaGetLastError();
        }
    }
    BlockQ8_1 * kv_tail_q8 = nullptr;
    if (projection_q8_mmq && kv_tail4) {
        const uint32_t blocks = n_in / 32;
        const uint64_t tail_bytes = uint64_t(4) * blocks * sizeof(BlockQ8_1);
        if (!ensure_q8_scratch_next(tail_bytes)) {
            set_pending(CUDA_RC_OOM, "K/V tail Q8_1 scratch");
            return;
        }
        kv_tail_q8 = static_cast<BlockQ8_1 *>(g.q8_scratch_next);
        imparo_sm80_mmvq::quantize_q8_1
            <<<dim3((n_in + 255) / 256, 4), 256, 0, g.stream>>>(
                x, kv_tail_q8, n_in, 4, src_row + n_tok - 4);
    }
    for (uint32_t row_base = 0; row_base < n_out; row_base += rows_per_slice) {
        const uint32_t rows = std::min(rows_per_slice, n_out - row_base);
        const uint8_t * raw = nullptr;
        int rc = wkind == 3
            ? q8_tm_weight_slice(
                w_off, n_in, n_out, row_base, rows, &raw)
            : weight_slice(w_off + uint64_t(row_base) * row_bytes,
                           uint64_t(rows) * row_bytes, &raw);
        if (rc) { set_pending(rc, "weight page upload"); return; }
        if (wkind == 0) {
            dim3 grid((rows + 255) / 256, n_tok);
            k_gemm_f32<<<grid, 256, 0, g.stream>>>(
                reinterpret_cast<const float *>(raw), x, y, n_in, rows, n_tok,
                src_row, n_out, row_base);
        } else if (q8_weight) {
            if (q8_mmvq && projection_q8) {
                if (wkind == 3) {
                    imparo_sm80_q8_mmvq::launch_tile_major(
                        raw, projection_q8, y, n_in, rows, n_tok,
                        n_out, row_base, g.stream);
                } else {
                    imparo_sm80_q8_mmvq::launch(raw, projection_q8, y, n_in,
                        rows, n_tok, n_out, row_base, g.stream);
                }
            } else if (q8_mmq && projection_q8_mmq) {
                imparo_sm80_mmq::LaunchInfo launch_info;
                const bool aligned_whole_k_requested = tensor_resident
                    && (tuner_knob(42) != 0
                        || std::getenv(
                            "IMPARO_CUDA_Q8_MMQ_ALIGNED_WHOLE_K_LAB")
                            != nullptr);
                const bool async_tm_stage = wkind == 3 && tuner_knob(47) != 0
                    && std::getenv("IMPARO_CUDA_NO_Q8_TM_ASYNC_STAGE") == nullptr;
                auto launch_result =
                    imparo_sm80_q8_mmq::LaunchResult::NotSupported;
                if (aligned_whole_k_requested) {
                    const bool w16_down = tuner_knob(47) == 2
                        && !q8_tm_silu_requested && n_in > n_out;
                    if (w16_down) {
                        launch_result = imparo_sm80_q8_mmq::launch_aligned_whole_k<
                            true, 0, true, 16>(
                                raw, projection_q8_mmq, y, nullptr,
                                n_in, rows, n_tok, n_out, row_base,
                                g.stream, &launch_info);
                    } else if (async_tm_stage) {
                        launch_result = q8_tm_silu_requested
                            ? (q8_tm_silu_sidecar
                                ? (g.epilogue == 4
                                    ? imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 4, true>(
                                        raw, projection_q8_mmq, y,
                                        q8_tm_silu_sidecar, n_in, rows, n_tok,
                                        n_out, row_base, g.stream, &launch_info)
                                    : imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 3, true>(
                                        raw, projection_q8_mmq, y,
                                        q8_tm_silu_sidecar, n_in, rows, n_tok,
                                        n_out, row_base, g.stream, &launch_info))
                                : imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 2, true>(
                                    raw, projection_q8_mmq, y, nullptr,
                                    n_in, rows, n_tok, n_out, row_base,
                                    g.stream, &launch_info))
                            : imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 0, true>(
                                raw, projection_q8_mmq, y, nullptr,
                                n_in, rows, n_tok,
                                n_out, row_base, g.stream, &launch_info);
                    } else {
                        launch_result = wkind == 3
                        ? (q8_tm_silu_requested
                            ? (q8_tm_silu_sidecar
                                ? (g.epilogue == 4
                                    ? imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 4>(
                                        raw, projection_q8_mmq, y,
                                        q8_tm_silu_sidecar, n_in, rows, n_tok,
                                        n_out, row_base, g.stream, &launch_info)
                                    : imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 3>(
                                        raw, projection_q8_mmq, y,
                                        q8_tm_silu_sidecar, n_in, rows, n_tok,
                                        n_out, row_base, g.stream, &launch_info))
                                : imparo_sm80_q8_mmq::launch_aligned_whole_k<true, 2>(
                                    raw, projection_q8_mmq, y, nullptr,
                                    n_in, rows, n_tok, n_out, row_base,
                                    g.stream, &launch_info))
                            : imparo_sm80_q8_mmq::launch_aligned_whole_k<true>(
                                raw, projection_q8_mmq, y, nullptr,
                                n_in, rows, n_tok,
                                n_out, row_base, g.stream, &launch_info))
                        : imparo_sm80_q8_mmq::launch_aligned_whole_k(
                            raw, projection_q8_mmq, y, nullptr,
                            n_in, rows, n_tok,
                            n_out, row_base, g.stream, &launch_info);
                    }
                    if (launch_result
                            == imparo_sm80_q8_mmq::LaunchResult::Launched
                            && q8_tm_silu_sidecar) {
                        produced_q8 = true;
                        produced_q8_layout = Q8_LAYOUT_MMQ_D4;
                    }
                }
                if (launch_result
                    == imparo_sm80_q8_mmq::LaunchResult::NotSupported) {
                    launch_result = wkind == 3
                        ? imparo_sm80_q8_mmq::launch<true>(raw,
                            projection_q8_mmq, y, n_in, rows, n_tok, n_out,
                            row_base, q8_mmq_plan, q8_replay_prefix, g.stream,
                            &launch_info)
                        : imparo_sm80_q8_mmq::launch(raw, projection_q8_mmq,
                            y, n_in, rows, n_tok, n_out, row_base, q8_mmq_plan,
                            q8_replay_prefix, g.stream, &launch_info);
                }
                static const bool trace_q8_mmq =
                    std::getenv("IMPARO_CUDA_MMQ_TRACE") != nullptr;
                if (trace_q8_mmq) {
                    std::fprintf(stderr,
                        "[cuda-q8-mmq] route=%s sm=%d in=%u rows=%u "
                        "row_base=%u out=%u tokens=%u tile=%ux%u "
                        "logical=%u physical=%u weights=%s async_tm=%u\n",
                        imparo_sm80_mmq::launch_route_name(launch_info.route),
                        g.sm_version, n_in, rows, row_base, n_out, n_tok,
                        launch_info.tile_rows, launch_info.tile_tokens,
                        launch_info.logical_tiles, launch_info.physical_blocks,
                        tensor_resident ? "resident" : "paged",
                        unsigned(async_tm_stage));
                }
                if (launch_result == imparo_sm80_q8_mmq::LaunchResult::Error) {
                    set_pending(CUDA_RC_ERROR, "Q8 MMQ setup or kernel launch");
                    return;
                }
                if (launch_result
                    == imparo_sm80_q8_mmq::LaunchResult::NotSupported) {
                    if (q8_tm_silu_requested) {
                        set_pending(CUDA_RC_INVALID,
                            "Q8_0_TM SiLU aligned MMQ launch rejected");
                        return;
                    }
                    // Capability/configuration failures occur before an MMQ
                    // output kernel is enqueued. Preserve fit and correctness
                    // by running the established f32 activation route.
                    (void)cudaGetLastError();
                    constexpr uint32_t warps = 4;
                    dim3 grid((rows + warps - 1) / warps, n_tok);
                    if (wkind == 3) {
                        k_gemm_q8_0_tm_f32<<<grid, warps * 32, 0, g.stream>>>(
                            raw, x, y, n_in, rows, n_tok, src_row, n_out, row_base);
                    } else {
                        k_gemm_q8_0_f32<<<grid, warps * 32, 0, g.stream>>>(
                            raw, x, y, n_in, rows, n_tok, src_row, n_out, row_base);
                    }
                    continue;
                }
            } else {
                constexpr uint32_t warps = 4;
                dim3 grid((rows + warps - 1) / warps, n_tok);
                if (wkind == 3) {
                    k_gemm_q8_0_tm_f32<<<grid, warps * 32, 0, g.stream>>>(
                        raw, x, y, n_in, rows, n_tok, src_row, n_out, row_base);
                } else {
                    k_gemm_q8_0_f32<<<grid, warps * 32, 0, g.stream>>>(
                        raw, x, y, n_in, rows, n_tok, src_row, n_out, row_base);
                }
            }

        } else if (projection_q8 && mmvq_microbatch) {
            const uint32_t batch_warps = decode_warps(n_in);
            const uint32_t batch_rows = batch_mmvq_rows_per_cta();
            imparo_sm80_mmvq::launch_microbatches(
                raw, projection_q8, y, n_in, rows, n_tok, n_out, row_base,
                batch_warps, batch_rows, g.stream);
        } else if (projection_q8 && kv_mmvq4) {
            const uint32_t blocks = n_in / 32;
            const uint32_t batch_warps = decode_warps(n_in);
            const uint32_t batch_rows = batch_mmvq_rows_per_cta();
            for (uint32_t token_base = 0; token_base < n_tok; token_base += 4) {
                imparo_sm80_mmvq::launch<4>(
                    raw, projection_q8 + uint64_t(token_base) * blocks,
                    y + uint64_t(token_base) * n_out, n_in, rows, g.epilogue,
                    n_out, row_base, batch_warps, batch_rows, g.stream);
            }
        } else if (projection_q8 && kv_fixed8) {
            dim3 grid((rows + 15) / 16, (n_tok + 15) / 16);
            k_gemm_q4_q8_1_mma<<<grid, 32, 0, g.stream>>>(
                raw, projection_q8, y, n_in, rows, n_tok, g.epilogue,
                n_out, row_base, uint32_t(g.sm_count), 0u, 8u, 0u, 0u);
        } else if (projection_q8 && kv_virtual512) {
            dim3 grid((rows + 15) / 16, (n_tok + 15) / 16);
            k_gemm_q4_q8_1_mma<<<grid, 32, 0, g.stream>>>(
                raw, projection_q8, y, n_in, rows, n_tok, g.epilogue,
                n_out, row_base, kv_virtual_workers, 1u, 0u,
                g.batch_geometry_start, 1u);
            const bool virtual_tail4 = n_tok > 4
                && (std::getenv("IMPARO_CUDA_KV_VIRTUAL_TAIL4") != nullptr
                    || (dst == 3 && std::getenv("IMPARO_CUDA_K_VIRTUAL_TAIL4") != nullptr)
                    || (dst == 4 && std::getenv("IMPARO_CUDA_V_VIRTUAL_TAIL4") != nullptr));
            if (virtual_tail4) {
                const uint32_t blocks = n_in / 32;
                const uint32_t batch_warps = decode_warps(n_in);
                const uint32_t batch_rows = batch_mmvq_rows_per_cta();
                imparo_sm80_mmvq::launch<4>(
                    raw, projection_q8 + uint64_t(n_tok - 4) * blocks,
                    y + uint64_t(n_tok - 4) * n_out,
                    n_in, rows, g.epilogue, n_out, row_base,
                    batch_warps, batch_rows, g.stream);
            }
        } else if (projection_q8 && n_tok == 1 && !g.epilogue) {
            if (g.sm_version >= 80
                && std::getenv("IMPARO_CUDA_NO_SM80_DECODE_MMVQ") == nullptr) {
                imparo_sm80_mmvq::launch_decode(
                    raw, projection_q8, y, n_in, rows, row_base,
                    gemv_warps, decode_rows_per_cta(n_in), g.stream);
            } else {
                k_gemv_q4_q8_1<<<rows, dim3(32, gemv_warps), 0, g.stream>>>(
                    raw, projection_q8, y, n_in, rows, row_base, gemv_warps);
            }
        } else if (projection_q8 && small_mmvq && n_tok >= 2) {
            // llama CUDA routes these widths through MMVQ rather than MMQ. Preserve
            // its per-warp accumulation order; tensor-core reassociation is visible
            // in recurrent model logits even when every local error is tiny.
            const uint32_t batch_warps = decode_warps(n_in);
            const uint32_t batch_rows = batch_mmvq_rows_per_cta();
            switch (n_tok) {
                case 2: imparo_sm80_mmvq::launch<2>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
                case 3: imparo_sm80_mmvq::launch<3>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
                case 4: imparo_sm80_mmvq::launch<4>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
                case 5: imparo_sm80_mmvq::launch<5>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
                case 6: imparo_sm80_mmvq::launch<6>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
                case 7: imparo_sm80_mmvq::launch<7>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
                case 8: imparo_sm80_mmvq::launch<8>(raw, projection_q8, y, n_in, rows, g.epilogue, n_out, row_base, batch_warps, batch_rows, g.stream); break;
            }
        } else if (projection_q8_mmq) {
            // Physical Stream-K changes its reduction ownership with the caller's
            // token width. That made the same logical row differ between a cold
            // n=464 tail and an externally resumed n=80 tail, so it is not a safe
            // default numerical route. Keep it diagnostic-only until a fixed logical
            // partition is admitted by the versioned correctness policy.
            const uint32_t stream_k_numeric = tuner_knob(24) ? 1u : 0u;
            float * mmq_workspace = nullptr;
            if (stream_k_numeric && g.sm_version >= 80) {
                const uint64_t workspace_bytes =
                    imparo_sm80_mmq::stream_workspace_bytes(
                        rows, n_tok, uint32_t(g.sm_count), g.epilogue);
                if (ensure_attention_scratch(workspace_bytes)) {
                    mmq_workspace = static_cast<float *>(g.attention_scratch);
                }
            }
            static const bool trace_mmq =
                std::getenv("IMPARO_CUDA_MMQ_TRACE") != nullptr;
            // Architecture-owned policy selects the internal shared-memory/K-stage
            // geometry. Common CUDA only supplies an optional tuned row override.
            uint32_t tuned_full_tile_rows = tuner_knob(11);
            if (!tuner_knob(11)) {
                if (const char * forced = std::getenv("IMPARO_CUDA_MMQ_FULL_ROWS")) {
                    const uint32_t parsed = uint32_t(std::strtoul(forced, nullptr, 10));
                    if (parsed == 64 || parsed == 128) tuned_full_tile_rows = parsed;
                }
            }
            const auto full_tile_variant =
                imparo_sm80_mmq::select_full_tile_variant(
                    uint32_t(g.sm_version), tuned_full_tile_rows);
            uint32_t tuned_full_tile_efficiency = tuner_knob(12);
            if (!tuned_full_tile_efficiency) {
                if (const char * forced = std::getenv(
                        "IMPARO_CUDA_MMQ_FULL_TILE_MIN_EFFICIENCY")) {
                    const uint32_t parsed = uint32_t(
                        std::strtoul(forced, nullptr, 10));
                    if (parsed >= 50 && parsed <= 100) {
                        tuned_full_tile_efficiency = parsed;
                    }
                }
            }
            const uint32_t full_tile_min_efficiency =
                imparo_sm80_mmq::select_full_tile_min_efficiency(
                    uint32_t(g.sm_version), tuned_full_tile_efficiency);
            BlockQ8_1Mmq * epilogue_q8 = nullptr;
            if (g.epilogue && row_base == 0 && rows == n_out
                && full_tile_variant
                    == imparo_sm80_mmq::FullTileVariant::Rows128K128
                && !std::getenv("IMPARO_CUDA_NO_FUSED_EPILOGUE_Q8")) {
                const uint64_t output_q8_bytes = uint64_t(n_tok)
                    * (n_out / 32) * sizeof(BlockQ8_1);
                if (ensure_q8_scratch_next(output_q8_bytes)) {
                    epilogue_q8 = static_cast<BlockQ8_1Mmq *>(
                        g.q8_scratch_next);
                }
            }
            imparo_sm80_mmq::LaunchInfo launch_info;
            const uint32_t large_tokens = n_tok;
            const bool canonical_full_mmq = n_tok == 512
                && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
                && g.batch_geometry_start % 512 == 0 && tuner_knob(27)
                && std::getenv("IMPARO_CUDA_NO_MMQ_CANONICAL_FULL") == nullptr;
            static const bool virtual_no_seam_mmq =
                std::getenv("IMPARO_CUDA_NO_MMQ_VIRTUAL_NO_SEAM") == nullptr;
            static const bool virtual_direct_seam_mmq =
                std::getenv("IMPARO_CUDA_NO_MMQ_VIRTUAL_DIRECT_SEAM") == nullptr;
            static const bool virtual_async_activation_mmq =
                std::getenv("IMPARO_CUDA_MMQ_VIRTUAL_ASYNC_ACTIVATION") != nullptr;
            // Diagnostic route isolation for whole-engine A/B. The persisted knob
            // remains authoritative in production; this opt-out exists only to price
            // the virtual schedule against the native full/physical schedule before
            // changing any versioned numerical policy.
            const bool virtual_mmq_enabled =
                std::getenv("IMPARO_CUDA_NO_MMQ_VIRTUAL") == nullptr;
            // Kernel-lab selector derived from exact-449 per-shape events:
            // virtual r64 remains materially faster for the wide gate/up pair,
            // while the pinned physical Stream-K scheduler wins on the down,
            // PLE and smaller attention projections. Keep this opt-in until the
            // combined numerical and end-to-end gate is complete.
            const bool hybrid_physical_lab =
                tuned_exact128_fast_transaction(n_tok)
                || std::getenv("IMPARO_CUDA_MMQ_HYBRID_PHYSICAL_LAB") != nullptr;
            static const bool hybrid_down_virtual_r64_lab =
                std::getenv("IMPARO_CUDA_MMQ_DOWN_FULL_K_R64_LAB") != nullptr
                || std::getenv("IMPARO_CUDA_MMQ_DOWN_STREAM_R64_LAB") != nullptr;
            const bool hybrid_virtual_shape =
                (n_in == 2560 && rows == 10240)
                || (hybrid_down_virtual_r64_lab
                    && n_in == 10240 && rows == 2560 && n_tok == 449);
            const uint32_t exact128_token64_mmq =
                (tuned_exact128_token64(n_tok)
                    || std::getenv("IMPARO_CUDA_MMQ_EXACT128_TOKEN64_LAB") != nullptr)
                ? 1u : 0u;
            const uint32_t virtual_mmq = n_tok > 1 && g.batch_geometry_valid
                && g.batch_geometry_tokens == n_tok && !canonical_full_mmq
                && tuner_knob(26) && virtual_mmq_enabled
                && (!hybrid_physical_lab || hybrid_virtual_shape) ? 1u : 0u;
            const bool gate_up_449_no_seam_lab = n_tok == 449
                && g.sm_version == 86 && g.batch_geometry_valid
                && g.batch_geometry_tokens == n_tok
                && g.batch_geometry_start % 512 == 0
                && virtual_mmq && virtual_no_seam_mmq;
            static const bool down_pipe_v1 =
                std::getenv("IMPARO_CUDA_MMQ_DOWN_PACKED_PIPE_V1") != nullptr;
            static const bool gate_up_pipe_v1 =
                std::getenv("IMPARO_CUDA_MMQ_GATE_UP_PACKED_K128") != nullptr;
            static const bool packed_dp4a_lab =
                std::getenv("IMPARO_CUDA_MMQ_PACKED_DP4A_LAB") != nullptr;
            bool aligned_stream_launched = false;
            const bool aligned_down_stream_lab =
                std::getenv(
                    "IMPARO_CUDA_PREFILL_DOWN_ALIGNED_STREAM_LAB") != nullptr;
            const bool aligned_projection_stream_lab =
                std::getenv(
                    "IMPARO_CUDA_PREFILL_PROJECTION_ALIGNED_STREAM_LAB")
                    != nullptr;
            const bool down_projection = n_in == 4 * n_out;
            const bool selected_aligned_stream =
                (aligned_down_stream_lab && down_projection)
                || (aligned_projection_stream_lab && !down_projection);
            if (selected_aligned_stream && g.sm_version == 86
                    && tensor_resident && stream_k_numeric && mmq_workspace
                    && !virtual_mmq && !g.epilogue && row_base == 0
                    && rows == n_out
                    && n_out % imparo_sm80_mmq::kRows == 0
                    && n_tok > 8 && n_tok <= 512) {
                const uint32_t logical_tiles =
                    (n_out / imparo_sm80_mmq::kRows)
                    * ((n_tok + imparo_sm80_mmq::kTokens - 1)
                        / imparo_sm80_mmq::kTokens);
                const uint32_t waves =
                    (logical_tiles + uint32_t(g.sm_count) - 1)
                    / uint32_t(g.sm_count);
                const uint32_t efficiency =
                    100u * logical_tiles / (uint32_t(g.sm_count) * waves);
                const bool llama_compat =
                    tuner_knob(25) && !canonical_full_mmq;
                const uint32_t route_efficiency = llama_compat
                    ? 90u : full_tile_min_efficiency;
                const bool default_full_tile = !llama_compat
                    && efficiency >= route_efficiency
                    && logical_tiles >= 2 * uint32_t(g.sm_count);
                if (!default_full_tile) {
                    const uint32_t stream_grid =
                        efficiency >= route_efficiency
                        ? logical_tiles : uint32_t(g.sm_count);
                    const bool packed_ready =
                        ensure_aligned_packed_q4_lab(
                            raw, n_in, n_out) != nullptr;
                    const PackedQ4Span * packed = packed_ready
                        ? find_aligned_packed_q4(raw, n_in, n_out) : nullptr;
                    if (packed) {
                        MatmatEventScope down_profile(
                            wkind, n_in, n_out, n_tok, 11);
                        const uint64_t records =
                            uint64_t(n_out) * (n_in / 32);
                        const auto * scales =
                            reinterpret_cast<const uint16_t *>(packed->packed);
                        const auto * nibbles =
                            reinterpret_cast<const uint4 *>(
                                packed->packed
                                    + records * sizeof(uint16_t));
                        aligned_stream_launched =
                            imparo_sm86_q4_aligned_prepack::
                                launch_aligned_q4_physical_stream(
                                    scales, nibbles, projection_q8_mmq,
                                    y + row_base, mmq_workspace,
                                    n_in, n_out, n_tok, n_tok, n_out,
                                    stream_grid, g.stream);
                        if (aligned_stream_launched) {
                            launch_info.route =
                                imparo_sm80_mmq::LaunchRoute::PhysicalStreamK;
                            launch_info.tile_rows = imparo_sm80_mmq::kRows;
                            launch_info.tile_tokens = imparo_sm80_mmq::kTokens;
                            launch_info.logical_tiles = logical_tiles;
                            launch_info.physical_blocks = stream_grid;
                            launch_info.efficiency = efficiency;
                        } else {
                            cudaGetLastError();
                        }
                    }
                }
            }
            imparo_sm86_packed_dp4a::LaunchResult packed_dp4a_result =
                imparo_sm86_packed_dp4a::LaunchResult::NotSupported;
            if (!aligned_stream_launched
                    && packed_dp4a_lab && g.weights_resident) {
                packed_dp4a_result = imparo_sm86_packed_dp4a::launch(
                    raw, projection_q8_mmq, y,
                    n_in, rows, large_tokens,
                    g.epilogue, n_out, row_base,
                    uint32_t(g.sm_version), stream_k_numeric,
                    g.batch_geometry_start, virtual_mmq,
                    g.stream, &launch_info);
            }
            if (packed_dp4a_result
                    == imparo_sm86_packed_dp4a::LaunchResult::Error) {
                set_pending(CUDA_RC_ERROR,
                    "SM86 packed-Q4 DP4A laboratory launch");
                return;
            }
            imparo_sm86_mmq_down_pipe::LaunchResult down_pipe_result =
                imparo_sm86_mmq_down_pipe::LaunchResult::NotSupported;
            imparo_sm86_mmq_gate_up_pipe::LaunchResult gate_up_pipe_result =
                imparo_sm86_mmq_gate_up_pipe::LaunchResult::NotSupported;
            if (packed_dp4a_result
                    != imparo_sm86_packed_dp4a::LaunchResult::Launched
                    && gate_up_pipe_v1
                    && (canonical_full_mmq || gate_up_449_no_seam_lab)
                    && g.weights_resident) {
                const bool gate_up_packed_ready =
                    ensure_aligned_packed_q4_lab(raw, n_in, rows) != nullptr;
                const PackedQ4Span * gate_up_packed = gate_up_packed_ready
                    ? find_aligned_packed_q4(raw, n_in, rows) : nullptr;
                if (gate_up_packed) {
                    const uint64_t records =
                        uint64_t(rows) * (n_in / 32);
                    const auto * scales =
                        reinterpret_cast<const uint16_t *>(
                            gate_up_packed->packed);
                    const auto * nibbles =
                        reinterpret_cast<const uint4 *>(
                            gate_up_packed->packed
                                + records * sizeof(uint16_t));
                    gate_up_pipe_result =
                        imparo_sm86_mmq_gate_up_pipe::launch(
                            scales, nibbles, projection_q8_mmq,
                            y, epilogue_q8, n_in, rows, large_tokens,
                            g.epilogue, n_out, row_base,
                            uint32_t(g.sm_version), uint32_t(g.sm_count),
                            uint32_t(g.q8_mmq_limits.max_grid_x),
                            g.stream, &launch_info);
                }
            }
            if (gate_up_pipe_result
                    == imparo_sm86_mmq_gate_up_pipe::LaunchResult::Error) {
                set_pending(CUDA_RC_ERROR,
                    "SM86 packed-Q4 gate/up K128 pipeline launch");
                return;
            }
            if (packed_dp4a_result
                    != imparo_sm86_packed_dp4a::LaunchResult::Launched
                    && gate_up_pipe_result
                        != imparo_sm86_mmq_gate_up_pipe::LaunchResult::Launched
                    && down_pipe_v1 && g.weights_resident) {
                down_pipe_result = imparo_sm86_mmq_down_pipe::launch(
                    raw, projection_q8_mmq, y,
                    n_in, rows, large_tokens,
                    g.epilogue, n_out, row_base,
                    uint32_t(g.sm_version), uint32_t(g.sm_count),
                    stream_k_numeric, g.batch_geometry_start, virtual_mmq,
                    uint32_t(g.q8_mmq_limits.max_grid_x),
                    g.stream, &launch_info);
            }
            if (down_pipe_result
                    == imparo_sm86_mmq_down_pipe::LaunchResult::Error) {
                set_pending(CUDA_RC_ERROR,
                    "SM86 packed-Q4 down-pipeline launch");
                return;
            }
            const bool large_mmq = aligned_stream_launched
                || packed_dp4a_result
                    == imparo_sm86_packed_dp4a::LaunchResult::Launched
                || gate_up_pipe_result
                    == imparo_sm86_mmq_gate_up_pipe::LaunchResult::Launched
                || down_pipe_result
                    == imparo_sm86_mmq_down_pipe::LaunchResult::Launched
                || imparo_sm80_mmq::launch(
                    raw, projection_q8_mmq, y, n_in, rows, large_tokens, g.epilogue,
                    n_out, row_base, uint32_t(g.sm_count), stream_k_numeric,
                    g.batch_geometry_start, virtual_mmq,
                    tuner_knob(25), canonical_full_mmq,
                    virtual_no_seam_mmq && g.sm_version == 86 ? 1u : 0u,
                    virtual_direct_seam_mmq && g.sm_version == 86 ? 1u : 0u,
                    virtual_async_activation_mmq && g.sm_version == 86 ? 1u : 0u,
                    exact128_token64_mmq,
                    full_tile_variant, full_tile_min_efficiency,
                    mmq_workspace, epilogue_q8, g.stream, &launch_info);
            if (g.tuner_lab && tuned_exact128_token64(n_tok)
                    && launch_info.route
                        == imparo_sm80_mmq::LaunchRoute::Exact128Token64SingleSeamLab
                    && n_in == 2560 && rows == 2048) {
                exact128_record_commit(g.tune_exact128_token64_commits);
            }
            produced_q8 = produced_q8 || launch_info.fused_q8;
            if (trace_mmq) {
                std::fprintf(stderr,
                    "[cuda-mmq] route=%s sm=%d in=%u rows=%u tokens=%u "
                    "epilogue=%u tile=%ux%u logical=%u physical=%u efficiency=%u\n",
                    imparo_sm80_mmq::launch_route_name(launch_info.route),
                    g.sm_version, n_in, rows, n_tok, g.epilogue,
                    launch_info.tile_rows, launch_info.tile_tokens,
                    launch_info.logical_tiles, launch_info.physical_blocks,
                    launch_info.efficiency);
            }
            if (!large_mmq) {
                set_pending(CUDA_RC_INVALID, "large MMQ packed layout dispatch");
                return;
            }
            if (kv_tail4) {
                const uint32_t batch_warps = decode_warps(n_in);
                const uint32_t batch_rows = batch_mmvq_rows_per_cta();
                imparo_sm80_mmvq::launch<4>(
                    raw, kv_tail_q8, y + uint64_t(n_tok - 4) * n_out,
                    n_in, rows, g.epilogue, n_out, row_base,
                    batch_warps, batch_rows, g.stream);
            }
        } else if (projection_q8 && !batch_dp4a) {
            const uint32_t stream_k_numeric = tuner_knob(24) ? 1u : 0u;
            if (kv_mma16) {
                const uint32_t blocks = n_in / 32;
                for (uint32_t token_base = 0; token_base < n_tok; token_base += 16) {
                    const uint32_t token_count = std::min(16u, n_tok - token_base);
                    dim3 grid((rows + 15) / 16, 1);
                    k_gemm_q4_q8_1_mma<<<grid, 32, 0, g.stream>>>(
                        raw, projection_q8 + uint64_t(token_base) * blocks,
                        y + uint64_t(token_base) * n_out,
                        n_in, rows, token_count, g.epilogue,
                        n_out, row_base, uint32_t(g.sm_count), stream_k_numeric, 0u,
                        0u, 0u);
                }
            } else {
                dim3 grid((rows + 15) / 16, (n_tok + 15) / 16);
                k_gemm_q4_q8_1_mma<<<grid, 32, 0, g.stream>>>(
                    raw, projection_q8, y, n_in, rows, n_tok, g.epilogue,
                    n_out, row_base, uint32_t(g.sm_count), stream_k_numeric, 0u,
                    0u, 0u);
            }
        } else if (projection_q8) {
            constexpr uint32_t batch_warps = 8;
            dim3 grid((rows + batch_warps - 1) / batch_warps, n_tok);
            k_gemm_q4_q8_1<<<grid, dim3(32, batch_warps), 0, g.stream>>>(
                raw, projection_q8, y, n_in, rows, n_tok, g.epilogue,
                n_out, row_base, batch_warps);
        } else if (n_tok == 1 && !g.epilogue) {
            constexpr uint32_t warps = 4;
            k_gemv_q4_f32<<<(rows + warps - 1) / warps, warps * 32, 0, g.stream>>>(
                raw, x + (uint64_t)src_row * n_in, y, n_in, rows, row_base);
        } else {
            dim3 grid((rows + 255) / 256, n_tok);
            k_gemm_q4<<<grid, 256, 0, g.stream>>>(
                raw, x, y, n_in, rows, n_tok, src_row, g.epilogue, n_out, row_base);
        }
    }
    mark_buf_written(dst);
    if (produced_q8) {
        publish_q8_scratch_next();
        own_q8_cache(dst, n_out, n_tok, 0, produced_q8_layout);
    }
}

extern "C" void imparo_cuda_matmat_gated(
        uint32_t gate_kind, uint64_t gate_off,
        uint32_t up_kind, uint64_t up_off,
        uint32_t n_in, uint32_t n_out, uint32_t src,
        uint32_t dst, uint32_t tmp, uint32_t n_tok,
        uint32_t fused_epilogue) {
#if defined(IMPARO_CUDA_ENABLE_CUBLAS_LAB)
    const bool cublas_gated_lab = gate_kind == 1 && up_kind == 1
        && fused_epilogue == 1 && n_tok > 128 && n_tok <= 512
        && g.sm_version == 86 && g.sm_count > 0 && g.weights_resident
        && src < B_COUNT && dst < B_COUNT
        && g.bufs[src] && g.bufs[dst]
        && n_in && n_out && n_in % 32 == 0 && n_out % 32 == 0
        && n_out > n_in
        && std::getenv("IMPARO_CUDA_PREFILL_CUBLAS_GATED_LAB") != nullptr;
    if (cublas_gated_lab) {
        using namespace imparo_sm86_gemm_q4_f16_cublas;
        LaunchInfo info{};
        const ProviderStatus shape = make_launch_info(
            n_in, n_out, n_tok, n_out, 0, &info);
        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * gate_weights = nullptr;
        const uint8_t * up_weights = nullptr;
        int rc = shape == ProviderStatus::Ready
            ? weight_slice(
                gate_off, row_bytes * n_out, &gate_weights)
            : CUDA_RC_INVALID;
        if (!rc) {
            rc = weight_slice(
                up_off, row_bytes * n_out, &up_weights);
        }
        CublasF16WeightSpan * gate_cache = nullptr;
        CublasF16WeightSpan * up_cache = nullptr;
        if (!rc) {
            // Admission is deliberately bounded and non-evicting. Repeated
            // prefill requests then see a stable hot set instead of spending
            // every layer copying/dequantizing a different weight matrix.
            (void)ensure_cublas_f16_weight_lab(
                gate_weights, n_in, n_out, info.scratch.weights_f16);
            (void)ensure_cublas_f16_weight_lab(
                up_weights, n_in, n_out, info.scratch.weights_f16);
            // The second insertion may reallocate the vector; never retain a
            // span pointer across it.
            gate_cache = find_cublas_f16_weight(
                gate_weights, n_in, n_out);
            up_cache = find_cublas_f16_weight(
                up_weights, n_in, n_out);
            const uint64_t transient_weight_bytes =
                gate_cache && up_cache ? 0 : info.scratch.weights_f16;
            rc = ensure_cublas_provider_scratch(
                transient_weight_bytes,
                info.scratch.activations_f16,
                info.scratch.output_f16);
        }
        if (!rc) {
            ProviderArgs args{};
            args.handle = g.cublas;
            args.stream = g.stream;
            args.weights_q4_0 = gate_weights;
            args.activations_f32 =
                static_cast<const float *>(g.bufs[src]);
            args.output_f32 = static_cast<float *>(g.bufs[dst]);
            args.weights_f16 = gate_cache
                ? gate_cache->weights
                : static_cast<__half *>(g.cublas_weights_f16);
            args.weights_f16_bytes = gate_cache
                ? gate_cache->bytes : g.cublas_weights_f16_bytes;
            args.activations_f16 =
                static_cast<__half *>(g.cublas_activations_f16);
            args.activations_f16_bytes =
                g.cublas_activations_f16_bytes;
            args.output_f16 =
                static_cast<__half *>(g.cublas_output_f16);
            args.output_f16_bytes = g.cublas_output_f16_bytes;
            args.n_in = n_in;
            args.n_out = n_out;
            args.n_tok = n_tok;
            args.dst_stride = n_out;
            args.row_base = 0;
            args.skip_weight_dequant = gate_cache && gate_cache->ready;
            MatmatEventScope profile(
                1, n_in, 2 * n_out, n_tok, 14);
            const LaunchResult gate_result = launch(args);
            if (gate_result.ok()) {
                if (gate_cache) gate_cache->ready = true;
                args.weights_q4_0 = up_weights;
                args.weights_f16 = up_cache
                    ? up_cache->weights
                    : static_cast<__half *>(g.cublas_weights_f16);
                args.weights_f16_bytes = up_cache
                    ? up_cache->bytes : g.cublas_weights_f16_bytes;
                args.direct_gelu_mul = true;
                args.skip_activation_convert = true;
                args.skip_weight_dequant = up_cache && up_cache->ready;
                const LaunchResult up_result = launch(args);
                if (up_result.ok()) {
                    if (up_cache) up_cache->ready = true;
                    mark_buf_written(dst);
                    return;
                }
            }
            cudaGetLastError();
        } else {
            cudaGetLastError();
        }
    }
#endif

    const bool q8_ready_pack_prefill = gate_kind == 1 && up_kind == 1
        && n_tok > 8 && n_tok <= 512
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_phase == 0
        && src < B_COUNT && dst < B_COUNT && g.bufs[src] && g.bufs[dst]
        && n_in && n_out && n_in % 128 == 0 && n_out % 128 == 0
        && n_out > n_in
        && std::getenv(
            "IMPARO_CUDA_NO_PREFILL_Q8_READY_GATED_LAB") == nullptr
        && std::getenv("IMPARO_CUDA_PREFILL_Q8_READY_PACK_LAB") != nullptr;
    if (q8_ready_pack_prefill) {
        MatmatEventScope profile(1, n_in, 2 * n_out, n_tok, 8);
        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * gate = nullptr;
        const uint8_t * up = nullptr;
        int rc = weight_slice(gate_off, row_bytes * n_out, &gate);
        if (rc) {
            set_pending(rc, "Q8-ready prefill gate weights");
            return;
        }
        rc = weight_slice(up_off, row_bytes * n_out, &up);
        if (rc) {
            set_pending(rc, "Q8-ready prefill up weights");
            return;
        }

        // The ensure call may grow the vector. Re-find both spans only after
        // all insertions so a reallocation cannot leave a dangling pointer.
        const bool gate_ready =
            ensure_aligned_packed_q4_lab(gate, n_in, n_out) != nullptr;
        const bool up_ready =
            ensure_aligned_packed_q4_lab(up, n_in, n_out) != nullptr;
        const PackedQ4Span * packed_gate = gate_ready && up_ready
            ? find_aligned_packed_q4(gate, n_in, n_out) : nullptr;
        const PackedQ4Span * packed_up = gate_ready && up_ready
            ? find_aligned_packed_q4(up, n_in, n_out) : nullptr;
        if (packed_gate && packed_up) {
            using ReadyLayout =
                imparo_q8_mma_ready_a0_v1_authority_lab::Layout;
            ReadyLayout layout{};
            const uint32_t padded_tokens = ((n_tok + 127u) / 128u) * 128u;
            if (imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                    n_in, padded_tokens, &layout)) {
                const uint64_t quant_bytes =
                    layout.quant_u16_count * sizeof(uint16_t);
                const uint64_t scale_bytes =
                    layout.scale_count * sizeof(float);
                rc = ensure_q8_scratch(quant_bytes + scale_bytes);
                if (rc) {
                    set_pending(rc, "Q8-ready prefill activation scratch");
                    return;
                }
                auto * quant_u16 = static_cast<uint16_t *>(g.q8_scratch);
                auto * d8_sideplane = reinterpret_cast<float *>(
                    static_cast<uint8_t *>(g.q8_scratch) + quant_bytes);
                const bool q8_hit = q8_cache_matches(
                    src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
                ReadyLayout actual = layout;
                actual.n_tok = n_tok;
                const bool quantized = q8_hit
                    || imparo_q8_mma_ready_a2_pairwarp16_research_lab::
                        launch_quantize_q8_mma_ready_ds4_a2_pairwarp16(
                            static_cast<const float *>(g.bufs[src]),
                            quant_u16, d8_sideplane, n_in, n_tok, g.stream,
                            &actual);
                if (!q8_hit && quantized) {
                    own_q8_cache(
                        src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
                }
                const uint64_t records = uint64_t(n_out) * (n_in / 32);
                const auto * gate_scales = reinterpret_cast<const uint16_t *>(
                    packed_gate->packed);
                const auto * gate_nibbles = reinterpret_cast<const uint4 *>(
                    packed_gate->packed + records * sizeof(uint16_t));
                const auto * up_scales = reinterpret_cast<const uint16_t *>(
                    packed_up->packed);
                const auto * up_nibbles = reinterpret_cast<const uint4 *>(
                    packed_up->packed + records * sizeof(uint16_t));
                const bool paircta_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_LAB") != nullptr;
                const bool paircta_t256_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_T256_LAB")
                        != nullptr;
                const bool paircta_r32_t128_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_R32_T128_LAB")
                        != nullptr;
                const bool paircta_r128_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_R128_LAB")
                        != nullptr;
                const bool paircta_shared_gate_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_LAB")
                        != nullptr;
                const bool paircta_shared_gate_t256_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_T256_LAB")
                        != nullptr;
                const bool paircta_shared_gate_prefetch_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_PREFETCH_LAB")
                        != nullptr;
                const bool paircta_shared_gate_vsub_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_VSUB_LAB")
                        != nullptr;
                const bool paircta_shared_gate_q8_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_Q8_LAB")
                        != nullptr;
                const bool paircta_shared_gate_q8_prefetch_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_SHARED_GATE_Q8_PREFETCH_LAB")
                        != nullptr;
                if (quantized
                        && (paircta_shared_gate_q8_ready
                            || paircta_shared_gate_q8_prefetch_ready)
                        && !g.graph_capturing && !g.prefill_capture_active
                        && std::getenv("IMPARO_GPU_PROBE") == nullptr) {
                    ReadyLayout output_layout{};
                    const bool output_layout_ready =
                        imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                            n_out, padded_tokens, &output_layout);
                    if (output_layout_ready) {
                        const uint64_t output_quant_bytes =
                            output_layout.quant_u16_count * sizeof(uint16_t);
                        const uint64_t output_scale_bytes =
                            output_layout.scale_count * sizeof(float);
                        if (ensure_q8_scratch_next(
                                output_quant_bytes + output_scale_bytes)) {
                            auto * output_quant_u16 =
                                static_cast<uint16_t *>(g.q8_scratch_next);
                            auto * output_d8_sideplane =
                                reinterpret_cast<float *>(
                                    static_cast<uint8_t *>(g.q8_scratch_next)
                                        + output_quant_bytes);
                            const bool launched =
                                paircta_shared_gate_q8_prefetch_ready
                                ? imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                                    launch_q8_ready_paircta_shared_gate_q8<true>(
                                        gate_scales, gate_nibbles,
                                        up_scales, up_nibbles,
                                        quant_u16, d8_sideplane,
                                        actual.token_tiles,
                                        static_cast<float *>(g.bufs[dst]),
                                        output_quant_u16,
                                        output_d8_sideplane,
                                        output_layout.token_tiles,
                                        n_in, n_out, n_tok, n_out, g.stream)
                                : imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                                    launch_q8_ready_paircta_shared_gate_q8(
                                        gate_scales, gate_nibbles,
                                        up_scales, up_nibbles,
                                        quant_u16, d8_sideplane,
                                        actual.token_tiles,
                                        static_cast<float *>(g.bufs[dst]),
                                        output_quant_u16,
                                        output_d8_sideplane,
                                        output_layout.token_tiles,
                                        n_in, n_out, n_tok, n_out, g.stream);
                            if (launched) {
                                mark_buf_written(dst);
                                const bool consume_fused_q8 = std::getenv(
                                    "IMPARO_CUDA_PREFILL_Q8_READY_PAIRCTA_Q8_NO_CONSUME")
                                    == nullptr;
                                if (consume_fused_q8) {
                                    publish_q8_scratch_next();
                                    own_q8_cache(
                                        dst, n_out, n_tok, 0,
                                        Q8_LAYOUT_MMA_READY);
                                }
                                return;
                            }
                            cudaGetLastError();
                        }
                    }
                }
                if (quantized && paircta_shared_gate_t256_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_shared_gate_t256(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_shared_gate_vsub_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_shared_gate_prefetch<true>(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_shared_gate_prefetch_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_shared_gate_prefetch(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_shared_gate_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_shared_gate(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_r128_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_r128(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_r32_t128_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_r32_t128(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_t256_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta_t256(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                if (quantized && paircta_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_paircta(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                actual.token_tiles,
                                static_cast<float *>(g.bufs[dst]),
                                n_in, n_out, n_tok, n_out, g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                const bool batched_ready =
                    std::getenv(
                        "IMPARO_CUDA_PREFILL_Q8_READY_BATCHED_LAB") != nullptr
                    && tmp < B_COUNT && dst != tmp && g.bufs[tmp];
                if (quantized && batched_ready) {
                    const bool launched =
                        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_batched_r2(
                                gate_scales, gate_nibbles,
                                up_scales, up_nibbles,
                                quant_u16, d8_sideplane,
                                static_cast<float *>(g.bufs[dst]),
                                static_cast<float *>(g.bufs[tmp]),
                                n_in, n_out, n_tok, n_tok, n_out,
                                0, uint32_t(g.q8_mmq_limits.max_grid_x),
                                g.stream);
                    if (launched) {
                        mark_buf_written(dst);
                        mark_buf_written(tmp);
                        k_gelu_mul<<<
                            (uint64_t(n_tok) * n_out + 255) / 256,
                            256, 0, g.stream>>>(
                                static_cast<float *>(g.bufs[dst]),
                                static_cast<const float *>(g.bufs[tmp]),
                                uint64_t(n_tok) * n_out);
                        mark_buf_written(dst);
                        return;
                    }
                    cudaGetLastError();
                }
                const bool gate_launched = quantized
                    && imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        launch_q8_ready_direct(
                            gate_scales, gate_nibbles, quant_u16,
                            d8_sideplane, actual.token_tiles,
                            static_cast<float *>(g.bufs[dst]),
                            n_in, n_out, n_tok, n_tok, n_out, false, g.stream);
                const bool up_launched = gate_launched
                    && imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        launch_q8_ready_direct(
                            up_scales, up_nibbles, quant_u16,
                            d8_sideplane, actual.token_tiles,
                            static_cast<float *>(g.bufs[dst]),
                            n_in, n_out, n_tok, n_tok, n_out, true, g.stream);
                if (up_launched) {
                    mark_buf_written(dst);
                    return;
                }
                cudaGetLastError();
            }
        }
    }
    const bool aligned_pack_prefill = gate_kind == 1 && up_kind == 1
        && n_tok > 8 && n_tok <= 512
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_start % 512 == 0 && fused_epilogue == 1
        && src < B_COUNT && dst < B_COUNT && g.bufs[src] && g.bufs[dst]
        && n_in && n_out && n_in % 128 == 0 && n_out % 128 == 0
        && std::getenv("IMPARO_CUDA_PREFILL_ALIGNED_PACK_LAB") != nullptr;
    if (aligned_pack_prefill) {
        MatmatEventScope profile(1, n_in, 2 * n_out, n_tok, 7);
        const uint64_t q8_records = uint64_t(n_tok) * (n_in / 32);
        const uint64_t q8_bytes = q8_records * sizeof(BlockQ8_1Mmq);
        int rc = ensure_q8_scratch(q8_bytes);
        if (rc) {
            set_pending(rc, "aligned-pack prefill Q8_1 projection scratch");
            return;
        }
        const bool q8_hit =
            q8_cache_matches(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);
        if (!q8_hit) {
            k_quantize_q8_1_mmq<<<dim3(n_tok, (n_in + 511) / 512),
                128, 0, g.stream>>>(static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                    n_in, n_tok, 0, false);
            own_q8_cache(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);
        }
        trace_q8_cache(q8_hit, src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);

        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * gate = nullptr;
        const uint8_t * up = nullptr;
        rc = weight_slice(gate_off, row_bytes * n_out, &gate);
        if (rc) {
            set_pending(rc, "aligned-pack prefill gate weights");
            return;
        }
        rc = weight_slice(up_off, row_bytes * n_out, &up);
        if (rc) {
            set_pending(rc, "aligned-pack prefill up weights");
            return;
        }
        const bool gate_ready =
            ensure_aligned_packed_q4_lab(gate, n_in, n_out) != nullptr;
        const bool up_ready =
            ensure_aligned_packed_q4_lab(up, n_in, n_out) != nullptr;
        const PackedQ4Span * packed_gate = gate_ready && up_ready
            ? find_aligned_packed_q4(gate, n_in, n_out) : nullptr;
        const PackedQ4Span * packed_up = gate_ready && up_ready
            ? find_aligned_packed_q4(up, n_in, n_out) : nullptr;
        if (packed_gate && packed_up) {
            const uint64_t records = uint64_t(n_out) * (n_in / 32);
            const uint16_t * gate_scales =
                reinterpret_cast<const uint16_t *>(packed_gate->packed);
            const uint4 * gate_nibbles = reinterpret_cast<const uint4 *>(
                packed_gate->packed + records * 2);
            const uint16_t * up_scales =
                reinterpret_cast<const uint16_t *>(packed_up->packed);
            const uint4 * up_nibbles = reinterpret_cast<const uint4 *>(
                packed_up->packed + records * 2);
            const bool gate_launched =
                imparo_sm86_q4_aligned_prepack::launch_aligned_q4(
                    gate_scales, gate_nibbles,
                    static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
                    static_cast<float *>(g.bufs[dst]), n_in, n_out, n_tok,
                    n_tok, n_out, 0, false, g.stream);
            const bool up_launched = gate_launched
                && imparo_sm86_q4_aligned_prepack::launch_aligned_q4(
                    up_scales, up_nibbles,
                    static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
                    static_cast<float *>(g.bufs[dst]), n_in, n_out, n_tok,
                    n_tok, n_out, 0, true, g.stream);
            if (up_launched) {
                mark_buf_written(dst);
                return;
            }
            cudaGetLastError();
        }
    }
    const bool interleaved_prefill = gate_kind == 1 && up_kind == 1
        && n_tok > 8 && n_tok <= 512
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_start % 512 == 0 && fused_epilogue == 1
        && src < B_COUNT && dst < B_COUNT && tmp < B_COUNT
        && dst != tmp && g.bufs[src] && g.bufs[dst] && g.bufs[tmp]
        && n_in && n_out && n_in % 128 == 0 && n_out % 128 == 0
        && std::getenv("IMPARO_CUDA_PREFILL_INTERLEAVED_R2_LAB") != nullptr;
    if (interleaved_prefill) {
        MatmatEventScope profile(1, n_in, 2 * n_out, n_tok, 6);
        const uint64_t q8_records = uint64_t(n_tok) * (n_in / 32);
        const uint64_t q8_bytes = q8_records * sizeof(BlockQ8_1Mmq);
        int rc = ensure_q8_scratch(q8_bytes);
        if (rc) {
            set_pending(rc, "interleaved prefill Q8_1 projection scratch");
            return;
        }
        const bool q8_hit =
            q8_cache_matches(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);
        if (!q8_hit) {
            k_quantize_q8_1_mmq<<<dim3(n_tok, (n_in + 511) / 512),
                128, 0, g.stream>>>(static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                    n_in, n_tok, 0, false);
            own_q8_cache(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);
        }
        trace_q8_cache(q8_hit, src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);

        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * gate = nullptr;
        const uint8_t * up = nullptr;
        rc = weight_slice(gate_off, row_bytes * n_out, &gate);
        if (rc) {
            set_pending(rc, "interleaved prefill gate weights");
            return;
        }
        rc = weight_slice(up_off, row_bytes * n_out, &up);
        if (rc) {
            set_pending(rc, "interleaved prefill up weights");
            return;
        }
        const auto launched = imparo_sm86_q4_q8_interleaved::launch(
            gate, up, static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
            static_cast<float *>(g.bufs[dst]),
            static_cast<float *>(g.bufs[tmp]), n_in, n_out, n_tok,
            uint32_t(g.sm_version), uint32_t(g.q8_mmq_limits.max_grid_x),
            g.stream);
        if (launched == imparo_sm86_q4_q8_interleaved::LaunchResult::Error) {
            set_pending(CUDA_RC_ERROR,
                "interleaved prefill Q4 x Q8 launch");
            return;
        }
        if (launched
                == imparo_sm86_q4_q8_interleaved::LaunchResult::Launched) {
            mark_buf_written(dst);
            mark_buf_written(tmp);
            k_gelu_mul<<<(uint64_t(n_tok) * n_out + 255) / 256, 256,
                0, g.stream>>>(static_cast<float *>(g.bufs[dst]),
                    static_cast<const float *>(g.bufs[tmp]),
                    uint64_t(n_tok) * n_out);
            mark_buf_written(dst);
            return;
        }
    }
    const bool pair_prefill = gate_kind == 1 && up_kind == 1
        && n_tok > 8 && n_tok <= 512
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_start % 512 == 0 && fused_epilogue == 1
        && src < B_COUNT && dst < B_COUNT && tmp < B_COUNT
        && g.bufs[src] && g.bufs[dst] && g.bufs[tmp]
        && n_in && n_out && n_in % 128 == 0 && n_out % 128 == 0
        && std::getenv("IMPARO_CUDA_PREFILL_PAIR_V1") != nullptr;
    if (pair_prefill) {
        MatmatEventScope profile(1, n_in, 2 * n_out, n_tok, 5);
        const uint64_t q8_records = uint64_t(n_tok) * (n_in / 32);
        const uint64_t q8_bytes = q8_records * sizeof(BlockQ8_1Mmq);
        int rc = ensure_q8_scratch(q8_bytes);
        if (rc) {
            set_pending(rc, "paired prefill Q8_1 projection scratch");
            return;
        }
        const bool q8_hit =
            q8_cache_matches(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);
        if (!q8_hit) {
            k_quantize_q8_1_mmq<<<dim3(n_tok, (n_in + 511) / 512),
                128, 0, g.stream>>>(static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                    n_in, n_tok, 0, false);
            own_q8_cache(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);
        }
        trace_q8_cache(q8_hit, src, n_in, n_tok, 0, Q8_LAYOUT_MMQ);

        const uint64_t output_q8_bytes = uint64_t(n_tok)
            * (n_out / 32) * sizeof(BlockQ8_1Mmq);
        if (!ensure_q8_scratch_next(output_q8_bytes)) {
            set_pending(CUDA_RC_OOM, "paired prefill epilogue Q8_1 scratch");
            return;
        }

        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * gate = nullptr;
        const uint8_t * up = nullptr;
        rc = weight_slice(gate_off, row_bytes * n_out, &gate);
        if (rc) {
            set_pending(rc, "paired prefill gate weights");
            return;
        }
        rc = weight_slice(up_off, row_bytes * n_out, &up);
        if (rc) {
            set_pending(rc, "paired prefill up weights");
            return;
        }
        const auto launched = imparo_sm86_q4_q8_pair::launch(
            gate, up, static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
            static_cast<float *>(g.bufs[dst]),
            static_cast<BlockQ8_1Mmq *>(g.q8_scratch_next),
            n_in, n_out, n_tok,
            uint32_t(g.sm_version),
            uint32_t(g.q8_mmq_limits.max_grid_x), g.stream);
        if (launched == imparo_sm86_q4_q8_pair::LaunchResult::Error) {
            set_pending(CUDA_RC_ERROR, "paired prefill Q4 x Q8 launch");
            return;
        }
        if (launched == imparo_sm86_q4_q8_pair::LaunchResult::Launched) {
            mark_buf_written(dst);
            publish_q8_scratch_next();
            own_q8_cache(dst, n_out, n_tok, 0, Q8_LAYOUT_MMQ);
            return;
        }
    }

    const bool shadow_decode_requested = fused_epilogue == 2
        && gate_kind == 3 && up_kind == 3 && n_tok == 1
        && g.sm_version == 86 && g.epilogue == 0
        && src < B_COUNT && dst < B_COUNT && tmp < B_COUNT
        && src != dst && src != tmp && dst != tmp
        && g.bufs[src] && g.bufs[dst] && g.bufs[tmp]
        && n_in != 0 && n_out != 0 && n_in % 32 == 0;
    if (shadow_decode_requested) {
        const DecodeQ4ShadowSpan * gate_shadow =
            find_decode_q4_shadow(gate_off, n_in, n_out);
        const DecodeQ4ShadowSpan * up_shadow =
            find_decode_q4_shadow(up_off, n_in, n_out);
        const bool compatible = gate_shadow && up_shadow
            && gate_shadow->tile_major && up_shadow->tile_major
            && gate_shadow->q5 == up_shadow->q5;
        if (compatible) {
            const uint64_t q8_bytes =
                uint64_t(n_in / 32) * sizeof(BlockQ8_1);
            const int rc = ensure_q8_scratch(q8_bytes);
            if (rc) {
                set_pending(rc, "mixed FFN Gate/Up Q8 activation scratch");
                return;
            }
            const bool q8_hit = q8_cache_matches(
                src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
            if (!q8_hit) {
                imparo_sm80_mmvq::quantize_q8_1
                    <<<(n_in + 255) / 256, 256, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<BlockQ8_1 *>(g.q8_scratch),
                        n_in, 1, 0);
                own_q8_cache(src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
            }
            {
                OpEventScope profile(
                    gate_shadow->q5 ? "ffn_gate_up_q5_shadow"
                                    : "ffn_gate_up_q4_shadow",
                    n_out, 1);
                imparo_sm80_mmvq::launch_tile_major_gated_decode(
                    gate_shadow->q4, up_shadow->q4,
                    static_cast<const BlockQ8_1 *>(g.q8_scratch),
                    static_cast<float *>(g.bufs[dst]),
                    n_in, n_out, gate_shadow->q5, g.stream);
            }
            mark_buf_written(dst);
            return;
        }
    }

    const bool q8_tm_decode_requested = fused_epilogue == 2
        && gate_kind == 3 && up_kind == 3 && n_tok == 1
        && tuner_knob(50) != 0
        && std::getenv("IMPARO_CUDA_NO_Q8_TM_GATED_MMVQ") == nullptr;
    if (q8_tm_decode_requested) {
        const uint64_t row_bytes = n_in % 32 == 0
            ? uint64_t(n_in / 32) * 34 : 0;
        const bool size_valid = row_bytes != 0 && n_out != 0
            && uint64_t(n_out) <= UINT64_MAX / row_bytes;
        const uint64_t tensor_bytes = size_valid
            ? row_bytes * uint64_t(n_out) : 0;
        const uint8_t * gate = size_valid
            ? resident_weight_range(gate_off, tensor_bytes) : nullptr;
        const uint8_t * up = size_valid
            ? resident_weight_range(up_off, tensor_bytes) : nullptr;
        const bool direct = g.sm_version == 86 && g.epilogue == 0
            && src < B_COUNT && dst < B_COUNT && src != dst
            && g.bufs[src] && g.bufs[dst]
            && n_in != 0 && n_out % 8 == 0
            && uint64_t(n_in) <= g.sizes[src] / sizeof(float)
            && uint64_t(n_out) <= g.sizes[dst] / sizeof(float)
            && gate && up
            && std::getenv("IMPARO_GPU_PROBE") == nullptr;
        if (direct) {
            MatmatEventScope profile(3, n_in, 2 * n_out, 1, 15);
            const uint64_t q8_bytes =
                uint64_t(n_in / 32) * sizeof(BlockQ8_1);
            const int rc = ensure_q8_scratch(q8_bytes);
            if (rc) {
                set_pending(rc, "Q8_0_TM gated Decode activation scratch");
                return;
            }
            const bool q8_hit = q8_cache_matches(
                src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
            if (!q8_hit) {
                imparo_sm80_mmvq::quantize_q8_1
                    <<<dim3((n_in + 255) / 256, 1), 256, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<BlockQ8_1 *>(g.q8_scratch),
                        n_in, 1, 0);
                own_q8_cache(src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
            }
            trace_q8_cache(q8_hit, src, n_in, 1, 0, Q8_LAYOUT_MMVQ);

            const auto result = imparo_sm86_q8_tm_gate_up_silu::launch(
                gate, up, static_cast<const BlockQ8_1 *>(g.q8_scratch),
                static_cast<float *>(g.bufs[dst]),
                n_in, n_out, uint32_t(g.sm_version), g.stream);
            if (result
                    == imparo_sm86_q8_tm_gate_up_silu::LaunchResult::Error) {
                set_pending(CUDA_RC_ERROR,
                    "Q8_0_TM gated Decode MMVQ launch");
                return;
            }
            if (result
                    == imparo_sm86_q8_tm_gate_up_silu::LaunchResult::Launched) {
                mark_tuner_dispatch(3, (uint64_t(15) << 32) | n_out);
                mark_buf_written(dst);
                return;
            }
        }
        if (g.tuner_mode) {
            set_pending(CUDA_RC_INVALID,
                "Q8_0_TM gated Decode candidate did not dispatch");
            return;
        }
    }
    const bool direct = gate_kind == 1 && up_kind == 1 && n_tok == 1
        && g.sm_version >= 80 && g.weights_resident
        && src < B_COUNT && dst < B_COUNT && g.bufs[src] && g.bufs[dst]
        && n_in && n_out && n_in % 32 == 0
        && std::getenv("IMPARO_CUDA_NO_GATED_MMVQ") == nullptr;
    if (direct) {
        MatmatEventScope profile(1, n_in, 2 * n_out, 1, 2);
        const uint64_t q8_bytes = uint64_t(n_in / 32) * sizeof(BlockQ8_1);
        int rc = ensure_q8_scratch(q8_bytes);
        if (rc) { set_pending(rc, "gated Q8_1 projection scratch"); return; }
        const bool q8_hit = q8_cache_matches(src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
        if (!q8_hit) {
            imparo_sm80_mmvq::quantize_q8_1
                <<<dim3((n_in + 255) / 256, 1), 256, 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1 *>(g.q8_scratch), n_in, 1, 0);
            own_q8_cache(src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
        }
        trace_q8_cache(q8_hit, src, n_in, 1, 0, Q8_LAYOUT_MMVQ);

        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * gate = nullptr;
        const uint8_t * up = nullptr;
        rc = weight_slice(gate_off, row_bytes * n_out, &gate);
        if (rc) { set_pending(rc, "gated gate weights"); return; }
        rc = weight_slice(up_off, row_bytes * n_out, &up);
        if (rc) { set_pending(rc, "gated up weights"); return; }
        imparo_sm80_mmvq::launch_gated_decode(
            gate, up, static_cast<const BlockQ8_1 *>(g.q8_scratch),
            static_cast<float *>(g.bufs[dst]), n_in, n_out,
            decode_warps(n_in), g.stream);
        mark_buf_written(dst);
        return;
    }

    // Portable/paged and batched fallback: retain the exact established sequence.
    const uint64_t q8_tm_row_bytes = n_in % 32 == 0
        ? uint64_t(n_in / 32) * 34 : 0;
    const bool q8_tm_silu = fused_epilogue == 2
        && gate_kind == 3 && up_kind == 3
        && tuner_knob(43) != 0 && tuner_knob(42) != 0
        && g.sm_version == 86 && n_tok > 8
        && n_in % 256 == 0
        && n_out % imparo_sm80_mmq::kRows == 0
        && q8_tm_row_bytes != 0
        && uint64_t(n_out) <= UINT64_MAX / q8_tm_row_bytes
        && resident_weight_range(
            up_off, q8_tm_row_bytes * uint64_t(n_out)) != nullptr;
    imparo_cuda_matmat(gate_kind, gate_off, n_in, n_out, src, dst, n_tok, 0);
    if (g.pending_error) return;
    if (q8_tm_silu) {
        const uint32_t previous = g.epilogue;
        g.epilogue = 2;
        imparo_cuda_matmat(up_kind, up_off, n_in, n_out, src, dst, n_tok, 0);
        g.epilogue = previous;
    } else if (fused_epilogue == 1) {
        const uint32_t previous = g.epilogue;
        g.epilogue = 1;
        imparo_cuda_matmat(up_kind, up_off, n_in, n_out, src, dst, n_tok, 0);
        g.epilogue = previous;
    } else {
        imparo_cuda_matmat(up_kind, up_off, n_in, n_out, src, tmp, n_tok, 0);
        if (!g.pending_error) {
            const uint64_t count = uint64_t(n_tok) * n_out;
            if (fused_epilogue == 2) {
                imparo_cuda_lfm2::silu_mul_kernel
                    <<<(count + 255) / 256, 256, 0, g.stream>>>(
                        static_cast<float *>(g.bufs[dst]),
                        static_cast<const float *>(g.bufs[tmp]), count);
            } else {
                k_gelu_mul<<<(count + 255) / 256, 256, 0, g.stream>>>(
                    static_cast<float *>(g.bufs[dst]),
                    static_cast<const float *>(g.bufs[tmp]), count);
            }
            mark_buf_written(dst);
        }
    }
}

static uint32_t try_q8_tm_silu_private_down(
        uint32_t gate_kind, uint64_t gate_off,
        uint32_t up_kind, uint64_t up_off,
        uint32_t down_kind, uint64_t down_off,
        uint32_t n_in, uint32_t n_mid, uint32_t n_out,
        uint32_t src, uint32_t gated_tmp, uint32_t dst,
        uint32_t n_tok) {
    const uint32_t min_tokens = tuner_knob(45);
    const bool requested = min_tokens != 0 && n_tok >= min_tokens
        && gate_kind == 3 && up_kind == 3 && down_kind == 3
        && tuner_knob(42) != 0 && tuner_knob(43) != 0;
    if (!requested) return 0;

    const bool eligible = n_tok > 8 && g.epilogue == 0
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_phase == 0
        && !g.graph_capturing && !g.prefill_capture_active
        && std::getenv("IMPARO_GPU_PROBE") == nullptr
        && src < B_COUNT && gated_tmp < B_COUNT && dst < B_COUNT
        && src != gated_tmp && src != dst && gated_tmp != dst
        && g.bufs[src] && g.bufs[gated_tmp] && g.bufs[dst]
        && n_in != 0 && n_mid != 0 && n_out != 0 && n_in == n_out
        && n_in % 256 == 0 && n_mid % 128 == 0 && n_out % 128 == 0
        && uint64_t(n_tok) * n_in <= g.sizes[src] / sizeof(float)
        && uint64_t(n_tok) * n_mid <= g.sizes[gated_tmp] / sizeof(float)
        && uint64_t(n_tok) * n_out <= g.sizes[dst] / sizeof(float);
    if (!eligible || g.pending_error) return 0;

    const auto resident_tm = [](uint64_t offset, uint32_t width,
                                uint32_t rows) -> const uint8_t * {
        if (width % 32 != 0) return nullptr;
        const uint64_t row_bytes = uint64_t(width / 32) * 34;
        if (rows != 0 && row_bytes > UINT64_MAX / rows) return nullptr;
        return resident_weight_range(offset, row_bytes * rows);
    };
    const uint8_t * gate_weights = resident_tm(gate_off, n_in, n_mid);
    const uint8_t * up_weights = resident_tm(up_off, n_in, n_mid);
    if (!gate_weights || !up_weights
            || !resident_tm(down_off, n_mid, n_out)) return 0;

    const uint64_t input_records = uint64_t(n_tok) * (n_in / 128);
    const uint64_t sidecar_records = uint64_t(n_tok) * (n_mid / 128);
    if (input_records > UINT64_MAX / sizeof(BlockQ8_1Mmq)
            || sidecar_records > UINT64_MAX / sizeof(BlockQ8_1Mmq)) return 0;
    if (ensure_q8_scratch(input_records * sizeof(BlockQ8_1Mmq))
            || !ensure_q8_scratch_next(
                sidecar_records * sizeof(BlockQ8_1Mmq))) {
        (void)cudaGetLastError();
        return 0;
    }
    const uint32_t pair_min_tokens = tuner_knob(48);
    const bool pair_requested = pair_min_tokens != 0
        && n_tok >= pair_min_tokens
        && std::getenv("IMPARO_CUDA_NO_Q8_TM_GATE_UP_ROW_PAIR") == nullptr;
    if (pair_requested) {
        const bool q8_hit = q8_cache_matches(
            src, n_in, n_tok, 0, Q8_LAYOUT_MMQ_D4);
        if (!q8_hit) {
            k_quantize_q8_1_mmq<<<dim3(n_tok, (n_in + 511) / 512),
                128, 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                    n_in, n_tok, 0, true);
            own_q8_cache(src, n_in, n_tok, 0, Q8_LAYOUT_MMQ_D4);
        }
        trace_q8_cache(q8_hit, src, n_in, n_tok, 0, Q8_LAYOUT_MMQ_D4);

        bool pair_launched = false;
        bool pair_error = false;
        namespace Pair = imparo_sm86_q8_tm_gate_up_row_pair_lab;
        const auto result = Pair::launch(
            gate_weights, up_weights,
            static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
            static_cast<BlockQ8_1Mmq *>(g.q8_scratch_next),
            n_in, n_mid, n_tok, uint32_t(g.sm_version),
            uint32_t(g.q8_mmq_limits.max_grid_x),
            uint32_t(g.q8_mmq_limits.max_grid_y), g.stream);
        pair_launched = result == Pair::LaunchResult::Launched;
        pair_error = result == Pair::LaunchResult::Error;
        if (pair_error) {
            set_pending(CUDA_RC_ERROR, "Q8_0_TM Gate/Up row-pair launch");
            return 1;
        }
        if (pair_launched) {
            mark_buf_written(gated_tmp);
            publish_q8_scratch_next();
            own_q8_cache(
                gated_tmp, n_mid, n_tok, 0, Q8_LAYOUT_MMQ_D4);
            imparo_cuda_matmat(down_kind, down_off, n_mid, n_out,
                gated_tmp, dst, n_tok, 0);
            return 1;
        }
    }

    imparo_cuda_matmat(gate_kind, gate_off, n_in, n_mid,
        src, gated_tmp, n_tok, 0);
    if (g.pending_error) return 1;

    const uint32_t previous_epilogue = g.epilogue;
    g.epilogue = 4;
    imparo_cuda_matmat(up_kind, up_off, n_in, n_mid,
        src, gated_tmp, n_tok, 0);
    g.epilogue = previous_epilogue;
    if (g.pending_error) return 1;

    imparo_cuda_matmat(down_kind, down_off, n_mid, n_out,
        gated_tmp, dst, n_tok, 0);
    return 1;
}

// Static-build laboratory transaction for the common gated-FFN sequence.  It is
// intentionally absent from the dynamic DLL export table: the experiment does
// not expand or freeze the production backend ABI.  Returning zero guarantees
// that no public buffer was written, so the workflow may issue its established
// gate/up/down sequence.  Once the first public-output launch is enqueued, CUDA
// errors are forward-fatal and this routine returns success for end() to report.
extern "C" uint32_t imparo_cuda_ffn_gated_down(
        uint32_t gate_kind, uint64_t gate_off,
        uint32_t up_kind, uint64_t up_off,
        uint32_t down_kind, uint64_t down_off,
        uint32_t n_in, uint32_t n_mid, uint32_t n_out,
        uint32_t src, uint32_t gated_tmp, uint32_t dst,
        uint32_t n_tok) {
    if (try_q8_tm_silu_private_down(
            gate_kind, gate_off, up_kind, up_off, down_kind, down_off,
            n_in, n_mid, n_out, src, gated_tmp, dst, n_tok)) {
        return 1;
    }

    using ReadyLayout =
        imparo_q8_mma_ready_a0_v1_authority_lab::Layout;
    namespace Ready =
        imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab;

    const char * lab_requested_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB");
    const bool lab_requested = lab_requested_env
        && std::strcmp(lab_requested_env, "1") == 0;
    const bool kernel_lab = g.tuner_lab;
    const bool tuned_requested = tuned_ffn_sidecar_requested(n_tok)
        || tuned_exact128_sm86_route(n_tok)
        || tuned_exact128_fast_transaction(n_tok);
    const bool requested = lab_requested || tuned_requested;
    const bool trace = requested && std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_TRACE") != nullptr;
    const char * fail_precommit_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_PRECOMMIT_LAB");
    const bool fail_precommit = fail_precommit_env
        && std::strcmp(fail_precommit_env, "1") == 0;
    const char * fail_postcommit_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_POSTCOMMIT_LAB");
    const bool fail_postcommit = fail_postcommit_env
        && std::strcmp(fail_postcommit_env, "1") == 0;
    const bool exact128_graph_capture =
        prefill_exact128_sidecar_capture_ready(n_tok);
    const auto reject_exact128_graph_capture = [&](const char * what) {
        if (!exact128_graph_capture) return;
        g.graph_capture_compatible = false;
        set_pending(CUDA_RC_INVALID, what);
    };
    if (trace) ++g.ffn_sidecar_trace_attempts;
    if (requested && !kernel_lab) ++g.ffn_sidecar_model_calls;
    const bool eligible = requested
        && gate_kind == 1 && up_kind == 1 && down_kind == 1
        && n_tok > 8 && n_tok <= 512 && g.epilogue == 0
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_phase == 0
        && ((!g.graph_capturing && !g.prefill_capture_active)
            || exact128_graph_capture)
        && std::getenv("IMPARO_GPU_PROBE") == nullptr
        && src < B_COUNT && gated_tmp < B_COUNT && dst < B_COUNT
        && src != gated_tmp && src != dst && gated_tmp != dst
        && g.bufs[src] && g.bufs[gated_tmp] && g.bufs[dst]
        && n_in && n_mid && n_out && n_in == n_out
        && uint64_t(n_mid) == 4ull * uint64_t(n_out)
        && uint64_t(n_tok) * n_in <= g.sizes[src] / sizeof(float)
        && uint64_t(n_tok) * n_out <= g.sizes[dst] / sizeof(float)
        && n_in % 128 == 0 && n_mid % 128 == 0 && n_out % 128 == 0;
    if (!eligible || g.pending_error) return 0;

    // Validate the logical grid before allocating or packing any sidecar.  Do not
    // require one logical tile per SM: the physical Stream-K schedule is deliberately
    // allowed to split a smaller logical grid across all SMs.  The receipt-bound token
    // selector, measured by the static tuner, decides where that schedule wins.
    const uint64_t logical_tiles64 = uint64_t(
            n_out / imparo_sm80_mmq::kRows)
        * ((uint64_t(n_tok) + imparo_sm80_mmq::kTokens - 1)
            / imparo_sm80_mmq::kTokens);
    if (!logical_tiles64 || logical_tiles64 > UINT32_MAX) return 0;
    const uint32_t logical_tiles = uint32_t(logical_tiles64);

    const uint64_t gate_row_bytes = uint64_t(n_in / 32) * 18;
    const uint64_t down_row_bytes = uint64_t(n_mid / 32) * 18;
    if (gate_row_bytes > UINT64_MAX / n_mid
            || down_row_bytes > UINT64_MAX / n_out) return 0;
    const uint64_t gate_bytes = gate_row_bytes * n_mid;
    const uint64_t down_bytes = down_row_bytes * n_out;
    if (!kernel_lab) {
        if (gate_bytes > (UINT64_MAX - down_bytes) / 2) return 0;
        const uint64_t layer_bytes = gate_bytes * 2 + down_bytes;
        if (!g.ffn_sidecar_model_layer_bytes) {
            g.ffn_sidecar_model_layer_bytes = layer_bytes;
        } else if (g.ffn_sidecar_model_layer_bytes != layer_bytes) {
            set_pending(CUDA_RC_INVALID,
                "sidecar-only FFN heterogeneous model plan");
            return 1;
        }
        const uint64_t layers = g.kv_layout.layers;
        if (!layers || layer_bytes > UINT64_MAX / layers
                || !initialize_packed_q4_budget()
                || layer_bytes * layers > g.packed_q4_budget) {
            reject_exact128_graph_capture(
                "exact128 FFN Graph packed-Q4 budget");
            return 0;
        }
    }
    const uint8_t * gate = nullptr;
    const uint8_t * up = nullptr;
    const uint8_t * down = nullptr;
    if (weight_slice(gate_off, gate_bytes, &gate)
            || weight_slice(up_off, gate_bytes, &up)
            || weight_slice(down_off, down_bytes, &down)) {
        reject_exact128_graph_capture(
            "exact128 FFN Graph weight span");
        return 0;
    }

    // All three sidecars are admitted before any public output.  Each insertion
    // may grow the vector, so pointers are re-found only after the final ensure.
    // Capture may consume only the complete hot set proven by the two ordinary
    // warm forwards; it must never allocate or mutate the packed-span vector.
    const bool gate_ready = exact128_graph_capture
        ? find_aligned_packed_q4(gate, n_in, n_mid) != nullptr
        : ensure_aligned_packed_q4_lab(gate, n_in, n_mid) != nullptr;
    const bool up_ready = gate_ready
        && (exact128_graph_capture
            ? find_aligned_packed_q4(up, n_in, n_mid) != nullptr
            : ensure_aligned_packed_q4_lab(up, n_in, n_mid) != nullptr);
    const bool down_ready = up_ready
        && (exact128_graph_capture
            ? find_aligned_packed_q4(down, n_mid, n_out) != nullptr
            : ensure_aligned_packed_q4_lab(down, n_mid, n_out) != nullptr);
    const PackedQ4Span * packed_gate = down_ready
        ? find_aligned_packed_q4(gate, n_in, n_mid) : nullptr;
    const PackedQ4Span * packed_up = down_ready
        ? find_aligned_packed_q4(up, n_in, n_mid) : nullptr;
    const PackedQ4Span * packed_down = down_ready
        ? find_aligned_packed_q4(down, n_mid, n_out) : nullptr;
    if (!packed_gate || !packed_up || !packed_down) {
        reject_exact128_graph_capture(
            "exact128 FFN Graph incomplete packed-Q4 hot set");
        if (!kernel_lab) g.ffn_sidecar_model_pack_failed = true;
        cudaGetLastError();
        return 0;
    }

    // Model execution uses a two-stage transaction.  The first eligible forward
    // discovers and packs every layer but deliberately publishes no sidecar
    // output; the established workflow remains authoritative for that forward.
    // end() promotes the cache only after all model layers prepared and the stream
    // synchronized successfully.  A later forward can therefore never discover
    // an incomplete model hot set after its first public Down write.
    if (!kernel_lab && !g.ffn_sidecar_model_ready) {
        ++g.ffn_sidecar_model_pack_calls;
        return 0;
    }

    const uint32_t padded_tokens = ((n_tok + 127u) / 128u) * 128u;
    ReadyLayout input_layout{};
    ReadyLayout output_layout{};
    if (!imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                n_in, padded_tokens, &input_layout)
            || !imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                n_mid, padded_tokens, &output_layout)) return 0;
    const uint64_t input_quant_bytes =
        input_layout.quant_u16_count * sizeof(uint16_t);
    const uint64_t input_scale_bytes =
        input_layout.scale_count * sizeof(float);
    const uint64_t output_quant_bytes =
        output_layout.quant_u16_count * sizeof(uint16_t);
    const uint64_t output_scale_bytes =
        output_layout.scale_count * sizeof(float);

    const uint32_t waves =
        (logical_tiles + uint32_t(g.sm_count) - 1) / uint32_t(g.sm_count);
    const uint32_t efficiency =
        100u * logical_tiles / (uint32_t(g.sm_count) * waves);
    uint32_t tuned_efficiency = g.knobs[12];
    if (!tuned_efficiency) {
        if (const char * forced = std::getenv(
                "IMPARO_CUDA_MMQ_FULL_TILE_MIN_EFFICIENCY")) {
            const uint32_t parsed =
                uint32_t(std::strtoul(forced, nullptr, 10));
            if (parsed >= 50 && parsed <= 100) tuned_efficiency = parsed;
        }
    }
    const uint32_t route_efficiency = g.knobs[25]
        ? 90u
        : imparo_sm80_mmq::select_full_tile_min_efficiency(
            uint32_t(g.sm_version), tuned_efficiency);
    uint32_t stream_grid = efficiency >= route_efficiency
        ? logical_tiles : uint32_t(g.sm_count);
    if (const char * forced = std::getenv(
            "IMPARO_CUDA_PREFILL_DOWN_Q8_READY_STREAM_GRID_LAB")) {
        const uint32_t parsed =
            uint32_t(std::strtoul(forced, nullptr, 10));
        if (parsed >= uint32_t(g.sm_count) && parsed <= logical_tiles) {
            stream_grid = parsed;
        }
    }
    const uint64_t fixup_bytes = uint64_t(stream_grid)
        * imparo_sm80_mmq::kTokens * imparo_sm80_mmq::kRows * sizeof(float);
    // Exact-shape Gate-A screen for removing the physical Stream-K seam and
    // fixup launch.  The selector names the concrete row schedule instead of
    // inheriting the older R64 environment flag, so benchmark identity cannot
    // drift.  Other shapes remain on the established physical route.
    const char * full_k_449_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_449_LAB");
    const char * full_k_128_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_128_LAB");
    const char * full_k_env = n_tok == 128 ? full_k_128_env
        : (n_tok == 449 ? full_k_449_env : nullptr);
    const bool tuned_exact128 = tuned_exact128_sm86_route(n_tok);
    Ready::DirectRows full_k_rows = tuned_exact128
        ? Ready::DirectRows::Rows128 : Ready::DirectRows::Environment;
    if (full_k_env) {
        const Ready::DirectRows parsed_rows =
            Ready::parse_direct_schedule(full_k_env);
        if (parsed_rows != Ready::DirectRows::Environment) {
            full_k_rows = parsed_rows;
        }
    }
    constexpr uint32_t kValidatedFullKWidth = 2560;
    constexpr uint32_t kValidatedFullKMid = 10240;
    const bool full_k_shape = tuned_exact128
        || (n_tok == 128 && full_k_128_env)
        || (n_tok == 449 && full_k_449_env);
    const bool full_k_requested = full_k_shape
        && full_k_rows != Ready::DirectRows::Environment
        && n_in == kValidatedFullKWidth
        && n_mid == kValidatedFullKMid && n_out == kValidatedFullKWidth;
    const bool full_k_configured = !full_k_requested
        || Ready::configure_q8_ready_direct_schedule(full_k_rows);
    const char * exact449_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_DOWN_STREAMK_AOT_449_LAB");
    const bool exact449_requested = !full_k_requested
        && exact449_env && std::strcmp(exact449_env, "1") == 0
        && n_tok == Ready::kExact449NTok
        && n_mid == Ready::kExact449NIn
        && n_out == Ready::kExact449NOut
        && stream_grid == Ready::kExact449PhysicalGrid
        && output_layout.token_tiles == Ready::kExact449TokenTiles;
    const bool exact449_configured = exact449_requested
        && Ready::configure_q8_ready_physical_stream_exact449_grid60();
    // Preflight the established physical route even when the AOT candidate is
    // configured.  It is the byte-safe fallback for an exact-kernel launch
    // rejection before the public-output commit point.
    const bool physical_configured = full_k_requested
        || Ready::configure_q8_ready_physical_stream();
    const char * r32_env = std::getenv(
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_R32_LAB");
    const bool r32_requested = r32_env && std::strcmp(r32_env, "1") == 0;
    const bool r32_configured = r32_requested
        && Ready::configure_q8_ready_paircta_r32_sidecar_q8();
    const bool r64_configured =
        Ready::configure_q8_ready_paircta_shared_gate_q8<false, false>();

    // Complete all recoverable allocation and launch-attribute preflight before
    // the sidecar-only producer.  Failure leaves Cur/G/X byte-authoritative.
    if (ensure_q8_scratch(input_quant_bytes + input_scale_bytes)
            || !ensure_q8_scratch_next(
                output_quant_bytes + output_scale_bytes)
            || (!full_k_requested && !ensure_attention_scratch(fixup_bytes))
            || (!r32_configured && !r64_configured)
            || !full_k_configured
            || !physical_configured) {
        reject_exact128_graph_capture(
            "exact128 FFN Graph preflight");
        cudaGetLastError();
        return 0;
    }
    // "Admitted" means the transaction has crossed every recoverable precommit
    // check: exact weights, all three packed sidecars, layouts, launch attributes,
    // scratch and workspace. Earlier counting at basic shape eligibility overstated
    // route coverage and could not distinguish a real candidate from a fallback.
    if (trace) ++g.ffn_sidecar_trace_admitted;
    if (g.tuner_lab && tuned_exact128) {
        g.tune_exact128_route_hits |= EXACT128_FFN_ADMITTED;
    }

    auto * input_quant = static_cast<uint16_t *>(g.q8_scratch);
    auto * input_scales = reinterpret_cast<float *>(
        static_cast<uint8_t *>(g.q8_scratch) + input_quant_bytes);
    auto * output_quant = static_cast<uint16_t *>(g.q8_scratch_next);
    auto * output_scales = reinterpret_cast<float *>(
        static_cast<uint8_t *>(g.q8_scratch_next) + output_quant_bytes);
    auto * fixup = static_cast<float *>(g.attention_scratch);

    const bool input_hit = q8_cache_matches(
        src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
    if (trace && input_hit) ++g.ffn_sidecar_trace_input_hits;
    trace_q8_cache(input_hit, src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
    ReadyLayout actual_input = input_layout;
    actual_input.n_tok = n_tok;
    const bool input_ready = input_hit
        || imparo_q8_mma_ready_a2_pairwarp16_research_lab::
            launch_quantize_q8_mma_ready_ds4_a2_pairwarp16(
                static_cast<const float *>(g.bufs[src]),
                input_quant, input_scales, n_in, n_tok, g.stream,
                &actual_input);
    if (!input_ready) {
        reject_exact128_graph_capture(
            "exact128 FFN Graph input quantizer");
        cudaGetLastError();
        return 0;
    }
    if (!input_hit) {
        own_q8_cache(src, n_in, n_tok, 0, Q8_LAYOUT_MMA_READY);
    }

    const uint64_t gate_records = uint64_t(n_mid) * (n_in / 32);
    const auto * gate_scales = reinterpret_cast<const uint16_t *>(
        packed_gate->packed);
    const auto * gate_nibbles = reinterpret_cast<const uint4 *>(
        packed_gate->packed + gate_records * sizeof(uint16_t));
    const auto * up_scales = reinterpret_cast<const uint16_t *>(
        packed_up->packed);
    const auto * up_nibbles = reinterpret_cast<const uint4 *>(
        packed_up->packed + gate_records * sizeof(uint16_t));

    // Both producers keep G byte-authoritative. R32 is selected only by the
    // exact lab value "1"; unsupported configuration or a pre-launch failure
    // falls back to the established R64 specialization before any public write.
    bool gate_up_launched = false;
    if (r32_configured) {
        {
            OpEventScope gate_up_profile(
                "ffn_sidecar_gate_up_r32", n_mid, n_tok);
            gate_up_launched =
                Ready::launch_q8_ready_paircta_r32_sidecar_q8(
                    gate_scales, gate_nibbles, up_scales, up_nibbles,
                    input_quant, input_scales, actual_input.token_tiles,
                    static_cast<float *>(g.bufs[gated_tmp]),
                    output_quant, output_scales,
                    output_layout.token_tiles,
                    n_in, n_mid, n_tok, n_mid, g.stream);
        }
        if (!gate_up_launched) cudaGetLastError();
    }
    if (!gate_up_launched && r64_configured) {
        {
            // Keep the fallback in its own scope.  A failed R32 launch must
            // never make a successful R64 fallback look like R32 evidence.
            OpEventScope gate_up_profile(
                "ffn_sidecar_gate_up_r64", n_mid, n_tok);
            gate_up_launched =
                Ready::launch_q8_ready_paircta_shared_gate_q8<false, false>(
                    gate_scales, gate_nibbles, up_scales, up_nibbles,
                    input_quant, input_scales, actual_input.token_tiles,
                    static_cast<float *>(g.bufs[gated_tmp]),
                    output_quant, output_scales,
                    output_layout.token_tiles,
                    n_in, n_mid, n_tok, n_mid, g.stream);
        }
    }
    if (!gate_up_launched) {
        reject_exact128_graph_capture(
            "exact128 FFN Graph Gate/Up producer");
        cudaGetLastError();
        return 0;
    }

    // Transaction-test hook: the Gate/Up producer has populated only private
    // sidecar storage (`WriteDense=false`).  Returning zero here must enqueue
    // the established workflow fallback without observing or publishing any
    // partial public output.
    if (fail_precommit) {
        reject_exact128_graph_capture(
            "exact128 FFN Graph injected precommit failure");
        return 0;
    }

    const uint64_t down_records = uint64_t(n_out) * (n_mid / 32);
    const auto * down_scales = reinterpret_cast<const uint16_t *>(
        packed_down->packed);
    const auto * down_nibbles = reinterpret_cast<const uint4 *>(
        packed_down->packed + down_records * sizeof(uint16_t));
    bool public_output_committed = false;
    bool down_launched = false;
    if (full_k_requested) {
        {
            OpEventScope down_profile(
                Ready::direct_schedule_profile_label(full_k_rows),
                n_out, n_tok);
            down_launched = Ready::launch_q8_ready_direct(
                down_scales, down_nibbles, output_quant, output_scales,
                output_layout.token_tiles, static_cast<float *>(g.bufs[dst]),
                n_mid, n_out, n_tok, n_tok, n_out, false, g.stream,
                full_k_rows);
        }
        // A successful direct launch is the transaction commit point.  There
        // is no fixup launch; later asynchronous failure is reported by end().
        public_output_committed = down_launched;
        if (down_launched && g.tuner_lab && tuned_exact128) {
            g.tune_exact128_route_hits |= EXACT128_FFN_DOWN;
        }
    } else if (exact449_configured) {
        {
            OpEventScope down_profile(
                "ffn_sidecar_down_streamk_aot_449", n_out, n_tok);
            down_launched =
                Ready::launch_q8_ready_physical_stream_exact449_grid60(
                    down_scales, down_nibbles, output_quant, output_scales,
                    output_layout.token_tiles,
                    static_cast<float *>(g.bufs[dst]), fixup,
                    n_mid, n_out, n_tok, n_tok, n_out, stream_grid, g.stream,
                    &public_output_committed);
        }
        // A launch-configuration failure cannot have written dst, so retain the
        // established physical route.  Once the exact main kernel is enqueued,
        // a fixup failure is forward-fatal and must never double-write dst.
        if (!down_launched && !public_output_committed) {
            cudaGetLastError();
            OpEventScope down_profile(
                "ffn_sidecar_down_aot_449_fallback", n_out, n_tok);
            down_launched = Ready::launch_q8_ready_physical_stream(
                down_scales, down_nibbles, output_quant, output_scales,
                output_layout.token_tiles, static_cast<float *>(g.bufs[dst]),
                fixup, n_mid, n_out, n_tok, n_tok, n_out, stream_grid, g.stream,
                &public_output_committed);
        }
    } else {
        OpEventScope down_profile(
            "ffn_sidecar_down", n_out, n_tok);
        down_launched = Ready::launch_q8_ready_physical_stream(
            down_scales, down_nibbles, output_quant, output_scales,
            output_layout.token_tiles, static_cast<float *>(g.bufs[dst]),
            fixup, n_mid, n_out, n_tok, n_tok, n_out, stream_grid, g.stream,
            &public_output_committed);
    }
    // Transaction-test hook: turn a successful launch sequence into a
    // synthetic failure strictly after the public destination commit.  The
    // common fatal branch below must return success to the workflow (so it
    // cannot double-write dst) while end() reports the pending error.
    if (down_launched && fail_postcommit) {
        public_output_committed = true;
        down_launched = false;
    }
    if (!down_launched) {
        cudaGetLastError();
        if (public_output_committed) {
            // The first public-output kernel is the transaction commit point.
            // A later fixup launch failure must surface as fatal at end(); it
            // can never return false and run the byte-authoritative fallback.
            mark_buf_written(dst);
            set_pending(CUDA_RC_ERROR,
                fail_postcommit
                    ? "sidecar-only FFN injected failure after commit"
                    : "sidecar-only FFN down fixup launch after commit");
            if (trace) ++g.ffn_sidecar_trace_committed;
            if (!kernel_lab) ++g.ffn_sidecar_model_commits;
            return 1;
        }
        reject_exact128_graph_capture(
            "exact128 FFN Graph Down producer");
        return 0;
    }
    mark_buf_written(dst);
    if (trace) ++g.ffn_sidecar_trace_committed;
    if (!kernel_lab) ++g.ffn_sidecar_model_commits;
    if (g.tuner_lab && tuned_exact128) {
        g.tune_exact128_route_hits |= EXACT128_FFN_COMMITTED;
        exact128_record_commit(g.tune_exact128_ffn_commits);
    }
    return 1;
}

extern "C" void imparo_cuda_matmat_pair(
        uint32_t first_kind, uint64_t first_off, uint32_t first_dst,
        uint32_t second_kind, uint64_t second_off, uint32_t second_dst,
        uint32_t n_in, uint32_t n_out, uint32_t src, uint32_t n_tok) {
    const bool prefill_q8_ready = first_kind == 1 && second_kind == 1
        && n_tok > 8 && n_tok <= 512 && g.epilogue == 0
        && g.sm_version == 86 && g.weights_resident
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && src < B_COUNT && first_dst < B_COUNT && second_dst < B_COUNT
        && g.bufs[src] && g.bufs[first_dst] && g.bufs[second_dst]
        && first_dst != second_dst && n_in && n_out
        && n_in % 128 == 0 && n_out % 128 == 0
        && std::getenv(
            "IMPARO_CUDA_PREFILL_Q8_READY_PAIR_PROJECTION_LAB") != nullptr;
    if (prefill_q8_ready) {
        MatmatEventScope profile(1, n_in, 2 * n_out, n_tok, 12);
        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * first = nullptr;
        const uint8_t * second = nullptr;
        int rc = weight_slice(first_off, row_bytes * n_out, &first);
        if (rc) { set_pending(rc, "first prefill pair weights"); return; }
        rc = weight_slice(second_off, row_bytes * n_out, &second);
        if (rc) { set_pending(rc, "second prefill pair weights"); return; }
        const bool first_ready =
            ensure_aligned_packed_q4_lab(first, n_in, n_out) != nullptr;
        const bool second_ready =
            ensure_aligned_packed_q4_lab(second, n_in, n_out) != nullptr;
        const PackedQ4Span * packed_first = first_ready && second_ready
            ? find_aligned_packed_q4(first, n_in, n_out) : nullptr;
        const PackedQ4Span * packed_second = first_ready && second_ready
            ? find_aligned_packed_q4(second, n_in, n_out) : nullptr;
        if (packed_first && packed_second) {
            using ReadyLayout =
                imparo_q8_mma_ready_a0_v1_authority_lab::Layout;
            ReadyLayout layout{};
            const uint32_t padded_tokens =
                ((n_tok + 127u) / 128u) * 128u;
            if (imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                    n_in, padded_tokens, &layout)) {
                const uint64_t quant_bytes =
                    layout.quant_u16_count * sizeof(uint16_t);
                const uint64_t scale_bytes =
                    layout.scale_count * sizeof(float);
                rc = ensure_q8_scratch(quant_bytes + scale_bytes);
                if (rc) {
                    set_pending(rc, "Q8-ready prefill pair scratch");
                    return;
                }
                auto * quant_u16 = static_cast<uint16_t *>(g.q8_scratch);
                auto * d8_sideplane = reinterpret_cast<float *>(
                    static_cast<uint8_t *>(g.q8_scratch) + quant_bytes);
                ReadyLayout actual = layout;
                actual.n_tok = n_tok;
                const bool quantized =
                    imparo_q8_mma_ready_a2_pairwarp16_research_lab::
                        launch_quantize_q8_mma_ready_ds4_a2_pairwarp16(
                            static_cast<const float *>(g.bufs[src]),
                            quant_u16, d8_sideplane, n_in, n_tok,
                            g.stream, &actual);
                const uint32_t logical_tiles =
                    (n_out / imparo_sm80_mmq::kRows)
                    * ((n_tok + imparo_sm80_mmq::kTokens - 1)
                        / imparo_sm80_mmq::kTokens);
                const uint32_t waves =
                    (logical_tiles + uint32_t(g.sm_count) - 1)
                    / uint32_t(g.sm_count);
                const uint32_t efficiency =
                    100u * logical_tiles
                    / (uint32_t(g.sm_count) * waves);
                uint32_t tuned_efficiency = g.knobs[12];
                if (!tuned_efficiency) {
                    if (const char * forced = std::getenv(
                            "IMPARO_CUDA_MMQ_FULL_TILE_MIN_EFFICIENCY")) {
                        const uint32_t parsed = uint32_t(
                            std::strtoul(forced, nullptr, 10));
                        if (parsed >= 50 && parsed <= 100) {
                            tuned_efficiency = parsed;
                        }
                    }
                }
                const uint32_t route_efficiency = g.knobs[25]
                    ? 90u
                    : imparo_sm80_mmq::select_full_tile_min_efficiency(
                        uint32_t(g.sm_version), tuned_efficiency);
                const uint32_t stream_grid = efficiency >= route_efficiency
                    ? logical_tiles : uint32_t(g.sm_count);
                const uint64_t fixup_bytes = uint64_t(stream_grid)
                    * imparo_sm80_mmq::kTokens
                    * imparo_sm80_mmq::kRows * sizeof(float);
                float * fixup = ensure_attention_scratch(fixup_bytes)
                    ? static_cast<float *>(g.attention_scratch) : nullptr;
                const uint64_t records = uint64_t(n_out) * (n_in / 32);
                const auto * first_scales =
                    reinterpret_cast<const uint16_t *>(packed_first->packed);
                const auto * first_nibbles = reinterpret_cast<const uint4 *>(
                    packed_first->packed + records * sizeof(uint16_t));
                const auto * second_scales =
                    reinterpret_cast<const uint16_t *>(packed_second->packed);
                const auto * second_nibbles = reinterpret_cast<const uint4 *>(
                    packed_second->packed + records * sizeof(uint16_t));

                const bool first_launched = quantized && fixup
                    && imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        launch_q8_ready_physical_stream(
                            first_scales, first_nibbles,
                            quant_u16, d8_sideplane, actual.token_tiles,
                            static_cast<float *>(g.bufs[first_dst]), fixup,
                            n_in, n_out, n_tok, n_tok, n_out,
                            stream_grid, g.stream);
                const bool second_launched = first_launched
                    && imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                        launch_q8_ready_physical_stream(
                            second_scales, second_nibbles,
                            quant_u16, d8_sideplane, actual.token_tiles,
                            static_cast<float *>(g.bufs[second_dst]), fixup,
                            n_in, n_out, n_tok, n_tok, n_out,
                            stream_grid, g.stream);
                if (second_launched) {
                    mark_buf_written(first_dst);
                    mark_buf_written(second_dst);
                    return;
                }
                cudaGetLastError();
            }
        }
    }
    const bool direct = first_kind == 1 && second_kind == 1 && n_tok == 1
        && g.epilogue == 0 && g.sm_version >= 80 && g.weights_resident
        && src < B_COUNT && first_dst < B_COUNT && second_dst < B_COUNT
        && g.bufs[src] && g.bufs[first_dst] && g.bufs[second_dst]
        && n_in && n_out && n_in % 32 == 0
        && std::getenv("IMPARO_CUDA_NO_PAIR_MMVQ") == nullptr;
    if (direct) {
        MatmatEventScope profile(1, n_in, 2 * n_out, 1, 3);
        const uint64_t q8_bytes = uint64_t(n_in / 32) * sizeof(BlockQ8_1);
        int rc = ensure_q8_scratch(q8_bytes);
        if (rc) { set_pending(rc, "pair Q8_1 projection scratch"); return; }
        const bool q8_hit = q8_cache_matches(src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
        if (!q8_hit) {
            imparo_sm80_mmvq::quantize_q8_1
                <<<dim3((n_in + 255) / 256, 1), 256, 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1 *>(g.q8_scratch), n_in, 1, 0);
            own_q8_cache(src, n_in, 1, 0, Q8_LAYOUT_MMVQ);
        }
        trace_q8_cache(q8_hit, src, n_in, 1, 0, Q8_LAYOUT_MMVQ);

        const uint64_t row_bytes = uint64_t(n_in / 32) * 18;
        const uint8_t * first = nullptr;
        const uint8_t * second = nullptr;
        rc = weight_slice(first_off, row_bytes * n_out, &first);
        if (rc) { set_pending(rc, "first pair weights"); return; }
        rc = weight_slice(second_off, row_bytes * n_out, &second);
        if (rc) { set_pending(rc, "second pair weights"); return; }
        imparo_sm80_mmvq::launch_pair_decode(
            first, second, static_cast<const BlockQ8_1 *>(g.q8_scratch),
            static_cast<float *>(g.bufs[first_dst]),
            static_cast<float *>(g.bufs[second_dst]), n_in, n_out,
            decode_warps(n_in), g.stream);
        mark_buf_written(first_dst);
        mark_buf_written(second_dst);
        return;
    }

    imparo_cuda_matmat(
        first_kind, first_off, n_in, n_out, src, first_dst, n_tok, 0);
    if (!g.pending_error) {
        imparo_cuda_matmat(
            second_kind, second_off, n_in, n_out, src, second_dst, n_tok, 0);
    }
}

extern "C" void imparo_cuda_ple_project(
        uint32_t gate_kind, uint64_t gate_off,
        uint32_t proj_kind, uint64_t proj_off,
        uint32_t n_embd, uint32_t ple_width, uint32_t src, uint32_t gate,
        uint32_t per_layer, uint32_t per_layer_off, uint32_t per_layer_stride,
        uint32_t back, uint32_t n_tok) {
    OpEventScope profile("ple_project", ple_width, n_tok);
    const bool direct = gate_kind == 1 && proj_kind == 1 && n_tok == 1
        && g.sm_version >= 80 && g.weights_resident
        && src < B_COUNT && gate < B_COUNT && per_layer < B_COUNT && back < B_COUNT
        && g.bufs[src] && g.bufs[gate] && g.bufs[per_layer] && g.bufs[back]
        && n_embd && ple_width && n_embd % 32 == 0 && ple_width % 32 == 0
        && std::getenv("IMPARO_CUDA_NO_PLE_EPILOGUE") == nullptr;
    if (direct) {
        {
            MatmatEventScope profile(1, n_embd, ple_width, 1, 4);
            const uint64_t q8_bytes = uint64_t(n_embd / 32) * sizeof(BlockQ8_1);
            int rc = ensure_q8_scratch(q8_bytes);
            if (rc) { set_pending(rc, "PLE gate Q8_1 scratch"); return; }
            const bool q8_hit = q8_cache_matches(src, n_embd, 1, 0, Q8_LAYOUT_MMVQ);
            if (!q8_hit) {
                imparo_sm80_mmvq::quantize_q8_1
                    <<<dim3((n_embd + 255) / 256, 1), 256, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<BlockQ8_1 *>(g.q8_scratch), n_embd, 1, 0);
                own_q8_cache(src, n_embd, 1, 0, Q8_LAYOUT_MMVQ);
            }
            trace_q8_cache(q8_hit, src, n_embd, 1, 0, Q8_LAYOUT_MMVQ);

            const uint64_t row_bytes = uint64_t(n_embd / 32) * 18;
            const uint8_t * gate_weights = nullptr;
            rc = weight_slice(gate_off, row_bytes * ple_width, &gate_weights);
            if (rc) { set_pending(rc, "PLE gate weights"); return; }
            imparo_sm80_mmvq::launch_ple_gate_decode(
                gate_weights, static_cast<const BlockQ8_1 *>(g.q8_scratch),
                static_cast<const float *>(g.bufs[per_layer]),
                static_cast<float *>(g.bufs[gate]), n_embd, ple_width,
                per_layer_off, decode_warps(n_embd), g.stream);
            mark_buf_written(gate);
        }
        imparo_cuda_matmat(
            proj_kind, proj_off, ple_width, n_embd, gate, back, 1, 0);
        return;
    }

    static const bool ple_fused_sm86 =
        std::getenv("IMPARO_CUDA_PLE_FUSED_SM86") != nullptr;
    static const bool ple_fused_short_direct_k_sm86 =
        std::getenv("IMPARO_CUDA_PLE_FUSED_SHORT_DIRECT_K_LAB") != nullptr;
    const bool ple_fused_shape =
        (ple_fused_sm86 && n_tok > 448 && n_tok <= 512)
        || (ple_fused_short_direct_k_sm86 && n_tok == 128)
        || tuned_exact128_fast_transaction(n_tok)
        || tuned_exact128_sm86_route(n_tok);
    const bool ple_fused_candidate = ple_fused_shape
        && gate_kind == 1 && proj_kind == 1 && g.sm_version == 86
        && g.weights_resident && n_embd == 2560 && ple_width == 256
        && src < B_COUNT && gate < B_COUNT
        && per_layer < B_COUNT && back < B_COUNT
        && g.bufs[src] && g.bufs[gate] && g.bufs[per_layer] && g.bufs[back]
        && g.batch_geometry_valid && g.batch_geometry_tokens == n_tok
        && g.batch_geometry_start % 512 == 0;
    const bool exact128_graph_capture =
        prefill_exact128_sidecar_capture_ready(n_tok);
    if (ple_fused_candidate) {
        const uint64_t input_q8_bytes = uint64_t(n_tok) * (n_embd / 32)
            * sizeof(BlockQ8_1);
        int rc = ensure_q8_scratch(input_q8_bytes);
        if (rc) { set_pending(rc, "PLE fused input Q8_1 scratch"); return; }
        const bool q8_hit =
            q8_cache_matches(src, n_embd, n_tok, 0, Q8_LAYOUT_MMQ);
        if (!q8_hit) {
            k_quantize_q8_1_mmq<<<dim3(n_tok, (n_embd + 511) / 512),
                128, 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[src]),
                    static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                    n_embd, n_tok, 0, false);
            own_q8_cache(src, n_embd, n_tok, 0, Q8_LAYOUT_MMQ);
        }
        trace_q8_cache(q8_hit, src, n_embd, n_tok, 0, Q8_LAYOUT_MMQ);

        const bool ple_ready_route = tuned_exact128_sm86_route(n_tok)
            || tuned_exact128_fast_transaction(n_tok)
            || std::getenv("IMPARO_CUDA_PLE_FUSED_Q8_READY_LAB") != nullptr;
        if (ple_ready_route) {
            using ReadyLayout =
                imparo_q8_mma_ready_a0_v1_authority_lab::Layout;
            ReadyLayout layout{};
            const uint32_t padded_tokens =
                ((n_tok + 127u) / 128u) * 128u;
            if (imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                    ple_width, padded_tokens, &layout)) {
                layout.n_tok = n_tok;
                const uint64_t quant_bytes =
                    layout.quant_u16_count * sizeof(uint16_t);
                const uint64_t scale_bytes =
                    layout.scale_count * sizeof(float);
                const uint64_t gate_row_bytes =
                    uint64_t(n_embd / 32) * 18;
                const uint64_t proj_row_bytes =
                    uint64_t(ple_width / 32) * 18;
                const uint8_t * gate_weights = nullptr;
                const uint8_t * proj_weights = nullptr;
                rc = weight_slice(
                    gate_off, gate_row_bytes * ple_width, &gate_weights);
                if (rc) {
                    set_pending(rc, "PLE ready gate weights");
                    return;
                }
                rc = weight_slice(
                    proj_off, proj_row_bytes * n_embd, &proj_weights);
                if (rc) {
                    set_pending(rc, "PLE ready projection weights");
                    return;
                }
                // Graph capture is read-only with respect to the complete packed
                // model hot set established by the ordinary warm forwards.
                const bool packed_ready = exact128_graph_capture
                    ? find_aligned_packed_q4(
                        proj_weights, ple_width, n_embd) != nullptr
                    : ensure_aligned_packed_q4_lab(
                        proj_weights, ple_width, n_embd) != nullptr;
                const PackedQ4Span * packed = packed_ready
                    ? find_aligned_packed_q4(
                        proj_weights, ple_width, n_embd)
                    : nullptr;
                const uint32_t logical_tiles =
                    (n_embd / imparo_sm80_mmq::kRows)
                    * ((n_tok + imparo_sm80_mmq::kTokens - 1)
                        / imparo_sm80_mmq::kTokens);
                const uint32_t waves =
                    (logical_tiles + uint32_t(g.sm_count) - 1)
                    / uint32_t(g.sm_count);
                const uint32_t efficiency =
                    100u * logical_tiles
                    / (uint32_t(g.sm_count) * waves);
                uint32_t tuned_efficiency = g.knobs[12];
                if (!tuned_efficiency) {
                    if (const char * forced = std::getenv(
                            "IMPARO_CUDA_MMQ_FULL_TILE_MIN_EFFICIENCY")) {
                        const uint32_t parsed = uint32_t(
                            std::strtoul(forced, nullptr, 10));
                        if (parsed >= 50 && parsed <= 100) {
                            tuned_efficiency = parsed;
                        }
                    }
                }
                const uint32_t route_efficiency = g.knobs[25]
                    ? 90u
                    : imparo_sm80_mmq::select_full_tile_min_efficiency(
                        uint32_t(g.sm_version), tuned_efficiency);
                const uint32_t stream_grid =
                    efficiency >= route_efficiency
                    ? logical_tiles : uint32_t(g.sm_count);
                const uint64_t fixup_bytes = uint64_t(stream_grid)
                    * imparo_sm80_mmq::kTokens
                    * imparo_sm80_mmq::kRows * sizeof(float);
                const bool direct_projection = n_tok == 128;
                const bool scratch_ready =
                    ensure_q8_scratch_next(quant_bytes + scale_bytes)
                    && (direct_projection
                        || ensure_attention_scratch(fixup_bytes));
                if (packed && scratch_ready) {
                    auto * ready_quant =
                        static_cast<uint16_t *>(g.q8_scratch_next);
                    auto * ready_scale = reinterpret_cast<float *>(
                        static_cast<uint8_t *>(g.q8_scratch_next)
                            + quant_bytes);
                    imparo_sm86_ple_gate::LaunchResult result;
                    {
                        OpEventScope fused_profile(
                            "ple_gate_ready_sm86", ple_width, n_tok);
                        result = imparo_sm86_ple_gate::launch_ready(
                            gate_weights,
                            static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
                            static_cast<const float *>(g.bufs[per_layer]),
                            static_cast<float *>(g.bufs[gate]),
                            ready_quant, ready_scale, layout.token_tiles,
                            n_embd, ple_width, n_tok,
                            per_layer_off, per_layer_stride,
                            uint32_t(g.sm_version), 1u, g.stream);
                    }
                    if (result == imparo_sm86_ple_gate::LaunchResult::Error) {
                        set_pending(
                            CUDA_RC_ERROR, "SM86 fused PLE ready launch");
                        return;
                    }
                    if (result
                        == imparo_sm86_ple_gate::LaunchResult::Launched) {
                        if (g.tuner_lab && tuned_exact128_sm86_route(n_tok)) {
                            g.tune_exact128_route_hits |= EXACT128_PLE_GATE;
                        }
                        if (std::getenv("IMPARO_CUDA_PLE_FUSED_TRACE")) {
                            std::fprintf(stderr,
                                "[cuda-ple-fused] route=ready tokens=%u "
                                "direct_k=%u gate_grid=%u\n",
                                n_tok, unsigned(n_tok == 128),
                                (ple_width / 32u) * ((n_tok + 63u) / 64u));
                        }
                        mark_buf_written(gate);
                        const uint64_t records =
                            uint64_t(n_embd) * (ple_width / 32);
                        const auto * scales =
                            reinterpret_cast<const uint16_t *>(
                                packed->packed);
                        const auto * nibbles =
                            reinterpret_cast<const uint4 *>(
                                packed->packed
                                    + records * sizeof(uint16_t));
                        MatmatEventScope projection_profile(
                            proj_kind, ple_width, n_embd, n_tok, 13);
                        bool public_output_committed = false;
                        const bool launched = direct_projection
                            ? imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_direct(
                                scales, nibbles, ready_quant, ready_scale,
                                layout.token_tiles,
                                static_cast<float *>(g.bufs[back]),
                                ple_width, n_embd, n_tok, n_tok, n_embd,
                                false, g.stream)
                            : imparo_q8_ready_batched_r2_interleaved_carveout100_v17_research_lab::
                            launch_q8_ready_physical_stream(
                                scales, nibbles, ready_quant, ready_scale,
                                layout.token_tiles,
                                static_cast<float *>(g.bufs[back]),
                                static_cast<float *>(g.attention_scratch),
                                ple_width, n_embd, n_tok, n_tok, n_embd,
                                stream_grid, g.stream,
                                &public_output_committed);
                        if (direct_projection && launched) {
                            public_output_committed = true;
                            if (g.tuner_lab
                                    && tuned_exact128_sm86_route(n_tok)) {
                                g.tune_exact128_route_hits |=
                                    EXACT128_PLE_DIRECT;
                                exact128_record_commit(g.tune_exact128_ple_commits);
                            }
                        }
                        if (launched) {
                            if (std::getenv("IMPARO_CUDA_PLE_FUSED_TRACE")) {
                                std::fprintf(stderr,
                                    "[cuda-ple-fused] consumer=%s tokens=%u "
                                    "local_ready=1 committed=1\n",
                                    direct_projection ? "direct" : "streamk",
                                    n_tok);
                            }
                            mark_buf_written(back);
                            return;
                        }
                        cudaGetLastError();
                        if (public_output_committed) {
                            mark_buf_written(back);
                            set_pending(CUDA_RC_ERROR,
                                "SM86 fused PLE projection failed after commit");
                            return;
                        }
                        if (exact128_graph_capture) {
                            g.graph_capture_compatible = false;
                            set_pending(CUDA_RC_INVALID,
                                "exact128 PLE Graph projection route");
                            return;
                        }
                        if (std::getenv("IMPARO_CUDA_PLE_FUSED_TRACE")) {
                            std::fprintf(stderr,
                                "[cuda-ple-fused] consumer=fallback tokens=%u "
                                "local_ready=1 committed=0\n", n_tok);
                        }
                        imparo_cuda_matmat(
                            proj_kind, proj_off, ple_width, n_embd,
                            gate, back, n_tok, 0);
                        return;
                    }
                }
            }
            if (exact128_graph_capture) {
                g.graph_capture_compatible = false;
                set_pending(CUDA_RC_INVALID,
                    "exact128 PLE Graph ready route unavailable");
                return;
            }
            cudaGetLastError();
        }

        const uint64_t output_q8_bytes = uint64_t(n_tok) * (ple_width / 32)
            * sizeof(BlockQ8_1);
        if (ensure_q8_scratch_next(output_q8_bytes)) {
            const uint64_t gate_row_bytes = uint64_t(n_embd / 32) * 18;
            const uint8_t * gate_weights = nullptr;
            rc = weight_slice(
                gate_off, gate_row_bytes * ple_width, &gate_weights);
            if (rc) { set_pending(rc, "PLE fused gate weights"); return; }
            imparo_sm86_ple_gate::LaunchResult result;
            {
                OpEventScope fused_profile(
                    "ple_gate_fused_sm86", ple_width, n_tok);
                result = imparo_sm86_ple_gate::launch(
                    gate_weights,
                    static_cast<const BlockQ8_1Mmq *>(g.q8_scratch),
                    static_cast<const float *>(g.bufs[per_layer]),
                    static_cast<float *>(g.bufs[gate]),
                    static_cast<BlockQ8_1Mmq *>(g.q8_scratch_next),
                    n_embd, ple_width, n_tok,
                    per_layer_off, per_layer_stride,
                    uint32_t(g.sm_version), 1u, g.stream);
            }
            if (result == imparo_sm86_ple_gate::LaunchResult::Error) {
                set_pending(CUDA_RC_ERROR, "SM86 fused PLE gate launch");
                return;
            }
            if (result == imparo_sm86_ple_gate::LaunchResult::Launched) {
                if (std::getenv("IMPARO_CUDA_PLE_FUSED_TRACE")) {
                    std::fprintf(stderr,
                        "[cuda-ple-fused] route=mmq tokens=%u direct_k=%u\n",
                        n_tok, unsigned(n_tok == 128));
                }
                mark_buf_written(gate);
                publish_q8_scratch_next();
                own_q8_cache(gate, ple_width, n_tok, 0, Q8_LAYOUT_MMQ);
                imparo_cuda_matmat(
                    proj_kind, proj_off, ple_width, n_embd,
                    gate, back, n_tok, 0);
                return;
            }
        }
    }

    // Portable, paged-weight, and batched fallback: exact common semantic order.
    imparo_cuda_matmat(
        gate_kind, gate_off, n_embd, ple_width, src, gate, n_tok, 0);
    if (g.pending_error) return;
    const bool fused_q8 = proj_kind == 1 && n_tok > 8
        && ple_width % 128 == 0 && g.sm_version >= 80
        && std::getenv("IMPARO_CUDA_NO_PLE_GATE_Q8") == nullptr;
    if (fused_q8) {
        const uint64_t q8_bytes = uint64_t(n_tok) * (ple_width / 32)
            * sizeof(BlockQ8_1);
        const int rc = ensure_q8_scratch(q8_bytes);
        if (rc) { set_pending(rc, "PLE gate Q8_1 scratch"); return; }
        imparo_sm80_mmq::launch_ple_gate_q8_mmq(
            static_cast<float *>(g.bufs[gate]),
            static_cast<const float *>(g.bufs[per_layer]),
            static_cast<BlockQ8_1Mmq *>(g.q8_scratch), ple_width,
            per_layer_off, per_layer_stride, n_tok, g.stream);
        mark_buf_written(gate);
        own_q8_cache(gate, ple_width, n_tok, 0, Q8_LAYOUT_MMQ);
    } else {
        k_gelu<<<(uint64_t(n_tok) * ple_width + 255) / 256, 256, 0, g.stream>>>(
            static_cast<float *>(g.bufs[gate]), uint64_t(n_tok) * ple_width);
        dim3 grid((ple_width + 255) / 256, n_tok);
        k_mul_strided<<<grid, 256, 0, g.stream>>>(
            static_cast<float *>(g.bufs[gate]),
            static_cast<const float *>(g.bufs[per_layer]), ple_width,
            per_layer_off, per_layer_stride, ple_width, n_tok);
        mark_buf_written(gate);
    }
    imparo_cuda_matmat(
        proj_kind, proj_off, ple_width, n_embd, gate, back, n_tok, 0);
}

extern "C" void imparo_cuda_rms_norm(uint32_t buf, uint32_t src, uint64_t w_off,
                                     uint32_t width, float eps, uint32_t n_row,
                                      uint32_t row_stride, uint32_t base_off,
                                      uint32_t has_w) {
    OpEventScope profile("rms_norm", width, n_row);
    const float * w = nullptr;
    if (has_w) {
        const uint8_t * raw = nullptr;
        int rc = weight_slice(w_off, uint64_t(width) * 4, &raw);
        if (rc) { set_pending(rc, "rms weight upload"); return; }
        w = reinterpret_cast<const float *>(raw);
    }
    const uint32_t requested = tuner_knob(8);
    const uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (width < 1024 ? 256 : 1024);
    constexpr uint32_t shared_bytes = 32 * sizeof(float);
    if (has_w && row_stride == width && base_off == 0) {
        const uint3 cols = init_fastdiv_values(width);
        const uint3 one = init_fastdiv_values(1);
        if (threads == 1024) {
            k_rms_norm_ggml<1024><<<dim3(n_row, 1, 1), 1024, shared_bytes, g.stream>>>(
                static_cast<const float *>(g.bufs[src]), static_cast<float *>(g.bufs[buf]),
                int(width), int64_t(row_stride), 0, 0, eps, w, 0, 0, 0,
                cols, one, one, one, nullptr, 0, 0, 0, one, one, one, one, 1.0f,
                nullptr, 0);
        } else {
            k_rms_norm_ggml<256><<<dim3(n_row, 1, 1), 256, shared_bytes, g.stream>>>(
                static_cast<const float *>(g.bufs[src]), static_cast<float *>(g.bufs[buf]),
                int(width), int64_t(row_stride), 0, 0, eps, w, 0, 0, 0,
                cols, one, one, one, nullptr, 0, 0, 0, one, one, one, one, 1.0f,
                nullptr, 0);
        }
        mark_buf_written(buf);
        return;
    }
    if (threads == 1024) {
        if (has_w) {
            k_rms_norm<1024, true><<<n_row, 1024, shared_bytes, g.stream>>>(
                w, static_cast<float *>(g.bufs[buf]), static_cast<const float *>(g.bufs[src]),
                width, eps, n_row, row_stride, base_off);
        } else {
            k_rms_norm<1024, false><<<n_row, 1024, shared_bytes, g.stream>>>(
                w, static_cast<float *>(g.bufs[buf]), static_cast<const float *>(g.bufs[src]),
                width, eps, n_row, row_stride, base_off);
        }
    } else {
        if (has_w) {
            k_rms_norm<256, true><<<n_row, 256, shared_bytes, g.stream>>>(
                w, static_cast<float *>(g.bufs[buf]), static_cast<const float *>(g.bufs[src]),
                width, eps, n_row, row_stride, base_off);
        } else {
            k_rms_norm<256, false><<<n_row, 256, shared_bytes, g.stream>>>(
                w, static_cast<float *>(g.bufs[buf]), static_cast<const float *>(g.bufs[src]),
                width, eps, n_row, row_stride, base_off);
        }
    }
    mark_buf_written(buf);
}

extern "C" void imparo_cuda_rms_norm_project(
        uint32_t buf, uint32_t src, uint64_t w_off, uint32_t width,
        float eps, uint32_t n_row, uint32_t row_stride, uint32_t base_off) {
    const bool projection_shape = buf < B_COUNT && src < B_COUNT
        && g.bufs[buf] && g.bufs[src] && width && width % 32 == 0
        && n_row > 0 && row_stride == width && (base_off == 0 || n_row == 1)
        && g.sm_version >= 80
        && std::getenv("IMPARO_CUDA_NO_RMS_Q8") == nullptr;
    const bool decode_direct = projection_shape && n_row == 1;
    const bool prefill_direct = projection_shape && n_row > 1
        && width % 128 == 0;
    if (!decode_direct && !prefill_direct) {
        imparo_cuda_rms_norm(
            buf, src, w_off, width, eps, n_row, row_stride, base_off, 1);
        return;
    }

    OpEventScope profile("rms_norm_project", width, n_row);
    const uint8_t * raw = nullptr;
    int rc = weight_slice(w_off, uint64_t(width) * sizeof(float), &raw);
    if (rc) { set_pending(rc, "projection RMS weight upload"); return; }
    if (prefill_direct
            && std::getenv("IMPARO_CUDA_PREFILL_RMS_FLOAT_ONLY_LAB")
                != nullptr) {
        imparo_cuda_rms_norm(
            buf, src, w_off, width, eps, n_row, row_stride, base_off, 1);
        return;
    }
    const bool mma_ready_prefill = prefill_direct && g.sm_version == 86
        && std::getenv("IMPARO_CUDA_PREFILL_RMS_Q8_READY_LAB") != nullptr;
    if (mma_ready_prefill) {
        using ReadyLayout =
            imparo_q8_mma_ready_a0_v1_authority_lab::Layout;
        ReadyLayout layout{};
        const uint32_t padded_tokens =
            ((n_row + 127u) / 128u) * 128u;
        if (imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
                width, padded_tokens, &layout)) {
            const uint64_t quant_bytes =
                layout.quant_u16_count * sizeof(uint16_t);
            const uint64_t scale_bytes =
                layout.scale_count * sizeof(float);
            rc = ensure_q8_scratch(quant_bytes + scale_bytes);
            if (rc) {
                set_pending(rc, "projection RMS MMA-ready Q8 scratch");
                return;
            }
            auto * quant_u16 = static_cast<uint16_t *>(g.q8_scratch);
            auto * d8_sideplane = reinterpret_cast<float *>(
                static_cast<uint8_t *>(g.q8_scratch) + quant_bytes);
            const uint32_t requested = g.knobs[8];
            const uint32_t threads =
                (requested == 256 || requested == 1024)
                ? requested : (width < 1024 ? 256 : 1024);
            constexpr uint32_t shared_bytes = 32 * sizeof(float);
            if (threads == 1024) {
                k_rms_norm_q8_mma_ready<1024>
                    <<<n_row, 1024, shared_bytes, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<float *>(g.bufs[buf]),
                        reinterpret_cast<const float *>(raw),
                        quant_u16, d8_sideplane, width, n_row,
                        layout.token_tiles, eps);
            } else {
                k_rms_norm_q8_mma_ready<256>
                    <<<n_row, 256, shared_bytes, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<float *>(g.bufs[buf]),
                        reinterpret_cast<const float *>(raw),
                        quant_u16, d8_sideplane, width, n_row,
                        layout.token_tiles, eps);
            }
            mark_buf_written(buf);
            own_q8_cache(
                buf, width, n_row, 0, Q8_LAYOUT_MMA_READY);
            return;
        }
    }
    const uint64_t q8_bytes = uint64_t(n_row) * (width / 32)
        * sizeof(BlockQ8_1);
    rc = ensure_q8_scratch(q8_bytes);
    if (rc) { set_pending(rc, "projection RMS Q8_1 scratch"); return; }

    const uint32_t requested = tuner_knob(8);
    const uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (width < 1024 ? 256 : 1024);
    constexpr uint32_t shared_bytes = 32 * sizeof(float);
    const bool q8_d4_prefill = prefill_direct
        && (tuner_knob(56) != 0
            || std::getenv("IMPARO_LAB_PREFILL_RMS_Q8_D4") != nullptr);
    if (prefill_direct) {
        if (threads == 1024) {
            if (q8_d4_prefill) {
                k_rms_norm_q8_1_mmq<1024, true, false>
                    <<<n_row, 1024, shared_bytes, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<float *>(g.bufs[buf]),
                        reinterpret_cast<const float *>(raw),
                        static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                        width, n_row, eps);
            } else {
                k_rms_norm_q8_1_mmq<1024>
                    <<<n_row, 1024, shared_bytes, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<float *>(g.bufs[buf]),
                        reinterpret_cast<const float *>(raw),
                        static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                        width, n_row, eps);
            }
        } else {
            if (q8_d4_prefill) {
                k_rms_norm_q8_1_mmq<256, true, false>
                    <<<n_row, 256, shared_bytes, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<float *>(g.bufs[buf]),
                        reinterpret_cast<const float *>(raw),
                        static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                        width, n_row, eps);
            } else {
                k_rms_norm_q8_1_mmq<256>
                    <<<n_row, 256, shared_bytes, g.stream>>>(
                        static_cast<const float *>(g.bufs[src]),
                        static_cast<float *>(g.bufs[buf]),
                        reinterpret_cast<const float *>(raw),
                        static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                        width, n_row, eps);
            }
        }
        mark_buf_written(buf);
        own_q8_cache(buf, width, n_row, 0,
            q8_d4_prefill ? Q8_LAYOUT_MMQ_D4 : Q8_LAYOUT_MMQ);
        return;
    }
    if (threads == 1024) {
        k_rms_norm_q8_1_decode<1024>
            <<<1, 1024, shared_bytes, g.stream>>>(
                static_cast<const float *>(g.bufs[src]) + base_off,
                static_cast<float *>(g.bufs[buf]),
                reinterpret_cast<const float *>(raw),
                static_cast<BlockQ8_1 *>(g.q8_scratch), width, eps);
    } else {
        k_rms_norm_q8_1_decode<256>
            <<<1, 256, shared_bytes, g.stream>>>(
                static_cast<const float *>(g.bufs[src]) + base_off,
                static_cast<float *>(g.bufs[buf]),
                reinterpret_cast<const float *>(raw),
                static_cast<BlockQ8_1 *>(g.q8_scratch), width, eps);
    }
    mark_buf_written(buf);
    own_q8_cache(buf, width, 1, 0, Q8_LAYOUT_MMVQ);
    if (std::getenv("IMPARO_CUDA_PROFILE_Q8")) {
        std::fprintf(stderr,
            "[cuda-q8-cache] produce src=%u epoch=%llu in=%u tokens=1 row=0 layout=%u\n",
            buf, static_cast<unsigned long long>(g.buf_epoch[buf]), width,
            Q8_LAYOUT_MMVQ);
    }
}

extern "C" uint32_t imparo_cuda_rms_norm_add_dual_project(
        uint32_t src, uint32_t residual, uint64_t first_w_off,
        uint32_t mid, uint64_t second_w_off, uint32_t out,
        uint32_t width, float eps, uint32_t n_row) {
    if (g.sm_version != 86 || g.graph_capturing
            || g.prefill_capture_active
            || std::getenv("IMPARO_GPU_PROBE") != nullptr
            || src >= B_COUNT || residual >= B_COUNT
            || mid >= B_COUNT || out >= B_COUNT
            || !g.bufs[src] || !g.bufs[residual]
            || !g.bufs[mid] || !g.bufs[out]
            || src != mid || out == mid || out == residual
            || width != imparo_sm86_dual_rms_q8_ready::kWidth
            || n_row <= 8
            || n_row > imparo_sm86_dual_rms_q8_ready::kMaxTokens) {
        return 0;
    }
    const uint8_t * first_raw = nullptr;
    const uint8_t * second_raw = nullptr;
    int rc = weight_slice(
        first_w_off, uint64_t(width) * sizeof(float), &first_raw);
    if (rc) return 0;
    rc = weight_slice(
        second_w_off, uint64_t(width) * sizeof(float), &second_raw);
    if (rc) return 0;

    using ReadyLayout = imparo_sm86_dual_rms_q8_ready::ReadyLayout;
    ReadyLayout layout{};
    const uint32_t padded_tokens =
        ((n_row + 127u) / 128u) * 128u;
    if (!imparo_q8_mma_ready_a0_v1_authority_lab::make_layout(
            width, padded_tokens, &layout)) {
        return 0;
    }
    const uint64_t quant_bytes =
        layout.quant_u16_count * sizeof(uint16_t);
    const uint64_t scale_bytes =
        layout.scale_count * sizeof(float);
    if (ensure_q8_scratch(quant_bytes + scale_bytes)) {
        cudaGetLastError();
        return 0;
    }
    layout.n_tok = n_row;
    auto * quant_u16 = static_cast<uint16_t *>(g.q8_scratch);
    auto * d8_sideplane = reinterpret_cast<float *>(
        static_cast<uint8_t *>(g.q8_scratch) + quant_bytes);
    OpEventScope profile("dual_rms_q8_ready", width, n_row);
    const auto launched = imparo_sm86_dual_rms_q8_ready::launch(
        static_cast<const float *>(g.bufs[src]),
        static_cast<const float *>(g.bufs[residual]),
        reinterpret_cast<const float *>(first_raw),
        reinterpret_cast<const float *>(second_raw),
        static_cast<float *>(g.bufs[mid]),
        static_cast<float *>(g.bufs[out]),
        quant_u16, d8_sideplane, width, n_row, layout,
        eps, eps, uint32_t(g.sm_version), g.stream);
    if (launched == imparo_sm86_dual_rms_q8_ready::LaunchResult::NotSupported) {
        return 0;
    }
    if (launched == imparo_sm86_dual_rms_q8_ready::LaunchResult::Error) {
        set_pending(CUDA_RC_ERROR, "dual RMS Q8-ready launch");
        return 0;
    }
    mark_buf_written(mid);
    mark_buf_written(out);
    own_q8_cache(out, width, n_row, 0, Q8_LAYOUT_MMA_READY);
    if (std::getenv("IMPARO_CUDA_PREFILL_DUAL_RMS_TRACE")) {
        std::fprintf(stderr,
            "[cuda-dual-rms-q8] tokens=%u width=%u committed=1\n",
            n_row, width);
    }
    return 1;
}

extern "C" void imparo_cuda_rms_norm_add(uint32_t dst, uint32_t src, uint64_t w_off,
                                          uint32_t width, float eps,
                                          uint32_t n_row, uint32_t add_buf,
                                          float output_scale) {
    OpEventScope profile("rms_norm_add", width, n_row);
    const uint8_t * raw = nullptr;
    int rc = weight_slice(w_off, uint64_t(width) * 4, &raw);
    if (rc) { set_pending(rc, "rms-add weight upload"); return; }
    const uint32_t requested = tuner_knob(8);
    const uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (width < 1024 ? 256 : 1024);
    constexpr uint32_t shared_bytes = 32 * sizeof(float);
    const uint3 cols = init_fastdiv_values(width);
    const uint3 one = init_fastdiv_values(1);
    const uint3 rows = init_fastdiv_values(n_row);
    const bool apply_scale = output_scale != 1.0f;
#define LAUNCH_RMS_ADD(BLOCK_SIZE, APPLY_SCALE)                                      \
    k_rms_norm_ggml<BLOCK_SIZE, 0, true, APPLY_SCALE>                               \
        <<<dim3(n_row, 1, 1), BLOCK_SIZE, shared_bytes, g.stream>>>(                 \
            static_cast<const float *>(g.bufs[src]),                                 \
            static_cast<float *>(g.bufs[dst]), int(width), int64_t(width), 0, 0, eps,\
            reinterpret_cast<const float *>(raw), 0, 0, 0,                           \
            cols, one, one, one, static_cast<const float *>(g.bufs[add_buf]),         \
            int64_t(width), 0, 0, cols, rows, one, one, output_scale, nullptr, 0)
    if (threads == 1024) {
        if (apply_scale) { LAUNCH_RMS_ADD(1024, true); }
        else { LAUNCH_RMS_ADD(1024, false); }
    } else {
        if (apply_scale) { LAUNCH_RMS_ADD(256, true); }
        else { LAUNCH_RMS_ADD(256, false); }
    }
#undef LAUNCH_RMS_ADD
    mark_buf_written(dst);
}

extern "C" void imparo_cuda_rms_norm_add_project(
        uint32_t buf, uint64_t w_off, uint32_t width, float eps,
        uint32_t n_row, uint32_t add_buf) {
    const bool direct = buf < B_COUNT && add_buf < B_COUNT
        && g.bufs[buf] && g.bufs[add_buf] && buf != add_buf
        && width && width % 128 == 0 && n_row > 8 && g.sm_version >= 80
        && std::getenv("IMPARO_CUDA_NO_RMS_ADD_Q8") == nullptr;
    if (!direct) {
        imparo_cuda_rms_norm_add(
            buf, buf, w_off, width, eps, n_row, add_buf, 1.0f);
        return;
    }
    OpEventScope profile("rms_norm_add_q8", width, n_row);
    const uint8_t * raw = nullptr;
    int rc = weight_slice(w_off, uint64_t(width) * sizeof(float), &raw);
    if (rc) { set_pending(rc, "rms-add weight upload"); return; }
    const uint64_t q8_bytes = uint64_t(n_row) * (width / 32)
        * sizeof(BlockQ8_1);
    rc = ensure_q8_scratch(q8_bytes);
    if (rc) { set_pending(rc, "rms-add Q8_1 scratch"); return; }
    const uint32_t requested = tuner_knob(8);
    const uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (width < 1024 ? 256 : 1024);
    constexpr uint32_t shared_bytes = 32 * sizeof(float);
    const uint3 cols = init_fastdiv_values(width);
    const uint3 one = init_fastdiv_values(1);
    const uint3 rows = init_fastdiv_values(n_row);
#define LAUNCH_RMS_ADD_Q8(BLOCK_SIZE)                                               \
    k_rms_norm_ggml<BLOCK_SIZE, 0, true, false, true>                              \
        <<<dim3(n_row, 1, 1), BLOCK_SIZE, shared_bytes, g.stream>>>(                \
            static_cast<const float *>(g.bufs[buf]),                               \
            static_cast<float *>(g.bufs[buf]), int(width), int64_t(width), 0, 0,    \
            eps, reinterpret_cast<const float *>(raw), 0, 0, 0,                    \
            cols, one, one, one, static_cast<const float *>(g.bufs[add_buf]),       \
            int64_t(width), 0, 0, cols, rows, one, one, 1.0f,                      \
            static_cast<BlockQ8_1Mmq *>(g.q8_scratch), n_row)
    if (threads == 1024) {
        LAUNCH_RMS_ADD_Q8(1024);
    } else {
        LAUNCH_RMS_ADD_Q8(256);
    }
#undef LAUNCH_RMS_ADD_Q8
    mark_buf_written(buf);
    own_q8_cache(buf, width, n_row, 0, Q8_LAYOUT_MMQ);
}

extern "C" uint32_t imparo_cuda_add_rms_norm_project(
        uint32_t dst, uint32_t resid, uint32_t other, uint64_t w_off,
        uint32_t width, float eps, uint32_t n_row) {
    if (g.sm_version < 80 || dst >= B_COUNT || resid >= B_COUNT
            || other >= B_COUNT || !g.bufs[dst] || !g.bufs[resid]
            || !g.bufs[other] || dst == resid || dst == other
            || resid == other || !width || width % 128 != 0 || n_row <= 8
            || g.graph_capturing || g.prefill_capture_active) {
        return 0;
    }
    const uint8_t * raw = nullptr;
    int rc = weight_slice(w_off, uint64_t(width) * sizeof(float), &raw);
    if (rc) return 0;
    const uint64_t q8_bytes = uint64_t(n_row) * (width / 32)
        * sizeof(BlockQ8_1Mmq);
    rc = ensure_q8_scratch(q8_bytes);
    if (rc) {
        cudaGetLastError();
        return 0;
    }
    OpEventScope profile("add_rms_norm_q8_d4", width, n_row);
    const uint32_t requested = tuner_knob(8);
    const bool vec4_512 = width == 2048 && tuner_knob(56) == 2;
    const uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (width < 1024 ? 256 : 1024);
    constexpr uint32_t shared_bytes = 32 * sizeof(float);
    if (vec4_512) {
        k_add_rms_norm_q8_1_mmq_vec4_512
            <<<n_row, 512, shared_bytes, g.stream>>>(
                static_cast<float *>(g.bufs[resid]),
                static_cast<const float *>(g.bufs[other]),
                static_cast<float *>(g.bufs[dst]),
                reinterpret_cast<const float *>(raw),
                static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                n_row, eps);
    } else if (threads == 1024) {
        k_add_rms_norm_q8_1_mmq<1024>
            <<<n_row, 1024, shared_bytes, g.stream>>>(
                static_cast<float *>(g.bufs[resid]),
                static_cast<const float *>(g.bufs[other]),
                static_cast<float *>(g.bufs[dst]),
                reinterpret_cast<const float *>(raw),
                static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                width, n_row, eps);
    } else {
        k_add_rms_norm_q8_1_mmq<256>
            <<<n_row, 256, shared_bytes, g.stream>>>(
                static_cast<float *>(g.bufs[resid]),
                static_cast<const float *>(g.bufs[other]),
                static_cast<float *>(g.bufs[dst]),
                reinterpret_cast<const float *>(raw),
                static_cast<BlockQ8_1Mmq *>(g.q8_scratch),
                width, n_row, eps);
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    mark_buf_written(resid);
    mark_buf_written(dst);
    own_q8_cache(dst, width, n_row, 0, Q8_LAYOUT_MMQ_D4);
    return 1;
}

extern "C" void imparo_cuda_rope(uint32_t buf, uint32_t n_rot, float base,
                                  uint32_t head_dim, uint32_t n_heads,
                                  uint32_t start_pos, uint32_t n_tok,
                                  const float * freqs_host) {
    OpEventScope profile("rope", n_rot, n_tok);
    const float * freqs_dev = nullptr;
    if (freqs_host) {
        const uint64_t bytes = uint64_t(n_rot / 2) * 4;
        const int rc = rope_frequency_buffer(freqs_host, bytes, &freqs_dev);
        if (rc) { set_pending(rc, "RoPE frequency upload"); return; }
    }
    const uint32_t total = n_tok * n_heads * (n_rot / 2);
    const float theta_scale = powf(base, -2.0f / float(n_rot));
    k_rope<<<(total + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[buf]), freqs_dev, n_rot, theta_scale, head_dim,
        n_heads, start_pos, n_tok, decode_control_arg());
    if (g.graph_capturing) ++g.graph_expected_dynamic_nodes;
    mark_buf_written(buf);
}

extern "C" void imparo_cuda_hadamard(uint32_t buf, uint32_t n, uint32_t nrot) {
    OpEventScope profile("hadamard", n, nrot);
    // The workflow's public rotation contract uses nrot=0 for "disabled".  The
    // fused norm/rope/Hadamard entry points already treat it as a no-op; keep the
    // standalone entry point consistent so diagnostic and fallback routes do too.
    if (nrot == 0) return;
    if (buf >= B_COUNT || !g.bufs[buf] || (nrot & (nrot - 1)) || n % nrot) {
        set_pending(CUDA_RC_INVALID, "Hadamard shape");
        return;
    }
    const float scale = 1.0f / sqrtf(float(nrot));
    if (nrot == 64 && std::getenv("IMPARO_CUDA_NO_HADAMARD64_WARP") == nullptr) {
        constexpr uint32_t warps = 8;
        const uint32_t blocks = n / 64;
        k_hadamard64_warp<false><<<(blocks + warps - 1) / warps,
            dim3(32, warps), 0, g.stream>>>(
                static_cast<float *>(g.bufs[buf]), nullptr, 0, 0,
                blocks, scale);
        mark_buf_written(buf);
        return;
    }
    const uint32_t thr = nrot < 256 ? nrot : 256;
    k_hadamard<<<n / nrot, thr, nrot * 4, g.stream>>>(
        static_cast<float *>(g.bufs[buf]), n, nrot, scale);
    mark_buf_written(buf);
}

extern "C" void imparo_cuda_hadamard_project(
        uint32_t buf, uint32_t width, uint32_t n_row, uint32_t nrot) {
    const uint64_t elements = uint64_t(width) * n_row;
    const bool direct = buf < B_COUNT && g.bufs[buf] && width && n_row > 8
        && width % 128 == 0 && nrot == 64 && g.sm_version >= 80
        && elements <= UINT32_MAX
        && std::getenv("IMPARO_CUDA_NO_HADAMARD_Q8") == nullptr;
    if (!direct) {
        if (elements > UINT32_MAX) {
            set_pending(CUDA_RC_INVALID, "Hadamard projection shape");
            return;
        }
        imparo_cuda_hadamard(buf, uint32_t(elements), nrot);
        return;
    }
    OpEventScope profile("hadamard_q8", width, n_row);
    const uint64_t q8_bytes = uint64_t(n_row) * (width / 32)
        * sizeof(BlockQ8_1);
    const int rc = ensure_q8_scratch(q8_bytes);
    if (rc) { set_pending(rc, "Hadamard Q8_1 scratch"); return; }
    constexpr uint32_t warps = 8;
    const uint32_t blocks = n_row * (width / 64);
    const float scale = rsqrtf(64.0f);
    k_hadamard64_warp<true><<<(blocks + warps - 1) / warps,
        dim3(32, warps), 0, g.stream>>>(
            static_cast<float *>(g.bufs[buf]),
            static_cast<BlockQ8_1Mmq *>(g.q8_scratch), width, n_row,
            blocks, scale);
    mark_buf_written(buf);
    own_q8_cache(buf, width, n_row, 0, Q8_LAYOUT_MMQ);
}

extern "C" void imparo_cuda_head_norm_rope_hadamard(
        uint32_t buf, uint64_t w_off, uint32_t head_dim, float eps,
        uint32_t n_heads, uint32_t start_pos, uint32_t n_tok,
        uint32_t rope_dim, float rope_base, const float * freqs_host,
        uint32_t hadamard_nrot) {
    const bool valid_hadamard = hadamard_nrot == 0
        || (hadamard_nrot <= head_dim && (hadamard_nrot & (hadamard_nrot - 1)) == 0
            && head_dim % hadamard_nrot == 0);
    const bool direct = buf < B_COUNT && g.bufs[buf] && head_dim && head_dim <= 1024
        && n_heads && n_tok && rope_dim && rope_dim <= head_dim && rope_dim % 2 == 0
        && valid_hadamard
        && std::getenv("IMPARO_CUDA_NO_HEAD_POST_FUSION") == nullptr;
    if (!direct) {
        imparo_cuda_rms_norm(
            buf, buf, w_off, head_dim, eps, n_tok * n_heads, head_dim, 0, 1);
        if (!g.pending_error) {
            imparo_cuda_rope(
                buf, rope_dim, rope_base, head_dim, n_heads,
                start_pos, n_tok, freqs_host);
        }
        if (!g.pending_error && hadamard_nrot != 0) {
            imparo_cuda_hadamard(
                buf, n_tok * n_heads * head_dim, hadamard_nrot);
        }
        return;
    }

    OpEventScope profile("head_post", head_dim, n_tok * n_heads);
    const uint8_t * norm_raw = nullptr;
    int rc = weight_slice(w_off, uint64_t(head_dim) * sizeof(float), &norm_raw);
    if (rc) { set_pending(rc, "head postprocess norm weights"); return; }

    const float * freqs_dev = nullptr;
    if (freqs_host) {
        const uint64_t bytes = uint64_t(rope_dim / 2) * sizeof(float);
        rc = rope_frequency_buffer(freqs_host, bytes, &freqs_dev);
        if (rc) { set_pending(rc, "RoPE frequency upload"); return; }
    }

    const uint32_t requested = tuner_knob(8);
    uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (head_dim < 1024 ? 256 : 1024);
    if (n_tok > 8) {
        const uint32_t candidate = tuner_knob(59);
        if (candidate == 64 || candidate == 128 || candidate == 256) {
            threads = candidate;
        }
    }
    const float theta_scale = powf(rope_base, -2.0f / float(rope_dim));
    const float hadamard_scale = hadamard_nrot ? 1.0f / sqrtf(float(hadamard_nrot)) : 1.0f;
    const uint32_t rows = n_tok * n_heads;
    __half * produced_q = nullptr;
    const uint64_t q_bytes = uint64_t(rows) * head_dim * sizeof(__half);
    if (n_tok > imparo_sm80_d512_small::kQueryTokens
        && g.sm_version >= 80
        && std::getenv("IMPARO_CUDA_NO_HEAD_Q_CACHE") == nullptr
        && ensure_attention_q_cache(q_bytes)) {
        produced_q = static_cast<__half *>(g.attention_q_cache);
    }
    if (threads == 1024) {
        k_head_norm_rope_hadamard<1024, true>
            <<<rows, 1024, head_dim * sizeof(float), g.stream>>>(
                static_cast<float *>(g.bufs[buf]),
                reinterpret_cast<const float *>(norm_raw), freqs_dev,
                head_dim, eps, n_heads, start_pos, rope_dim, theta_scale,
                hadamard_nrot, hadamard_scale, decode_control_arg(), produced_q);
    } else if (threads == 128) {
        k_head_norm_rope_hadamard<128, true>
            <<<rows, 128, head_dim * sizeof(float), g.stream>>>(
                static_cast<float *>(g.bufs[buf]),
                reinterpret_cast<const float *>(norm_raw), freqs_dev,
                head_dim, eps, n_heads, start_pos, rope_dim, theta_scale,
                hadamard_nrot, hadamard_scale, decode_control_arg(), produced_q);
    } else if (threads == 64) {
        k_head_norm_rope_hadamard<64, true>
            <<<rows, 64, head_dim * sizeof(float), g.stream>>>(
                static_cast<float *>(g.bufs[buf]),
                reinterpret_cast<const float *>(norm_raw), freqs_dev,
                head_dim, eps, n_heads, start_pos, rope_dim, theta_scale,
                hadamard_nrot, hadamard_scale, decode_control_arg(), produced_q);
    } else {
        k_head_norm_rope_hadamard<256, true>
            <<<rows, 256, head_dim * sizeof(float), g.stream>>>(
                static_cast<float *>(g.bufs[buf]),
                reinterpret_cast<const float *>(norm_raw), freqs_dev,
                head_dim, eps, n_heads, start_pos, rope_dim, theta_scale,
                hadamard_nrot, hadamard_scale, decode_control_arg(), produced_q);
    }
    if (g.graph_capturing) ++g.graph_expected_dynamic_nodes;
    mark_buf_written(buf);
    if (produced_q) {
        g.attention_q_src = buf;
        g.attention_q_epoch = g.buf_epoch[buf];
        g.attention_q_head_dim = head_dim;
        g.attention_q_heads = n_heads;
        g.attention_q_tokens = n_tok;
    }
}

extern "C" void imparo_cuda_kv_head_postprocess(
        uint32_t k_buf, uint32_t v_buf, uint64_t k_norm_off,
        uint32_t head_dim, float eps, uint32_t n_kv,
        uint32_t start_pos, uint32_t n_tok, uint32_t rope_dim,
        float rope_base, const float * freqs_host,
        uint32_t k_hadamard_nrot, uint32_t v_hadamard_nrot) {
    const auto valid_hadamard = [head_dim](uint32_t nrot) {
        return nrot == 0
            || (nrot <= head_dim && (nrot & (nrot - 1)) == 0
                && head_dim % nrot == 0);
    };
    const bool direct = k_buf < B_COUNT && v_buf < B_COUNT && k_buf != v_buf
        && g.bufs[k_buf] && g.bufs[v_buf] && head_dim && head_dim <= 1024
        && n_kv && n_tok && rope_dim && rope_dim <= head_dim && rope_dim % 2 == 0
        && valid_hadamard(k_hadamard_nrot) && valid_hadamard(v_hadamard_nrot)
        && std::getenv("IMPARO_CUDA_NO_KV_HEAD_FUSION") == nullptr;
    if (!direct) {
        imparo_cuda_rms_norm(
            k_buf, k_buf, k_norm_off, head_dim, eps,
            n_tok * n_kv, head_dim, 0, 1);
        if (!g.pending_error) {
            imparo_cuda_rms_norm(
                v_buf, v_buf, UINT64_MAX, head_dim, eps,
                n_tok * n_kv, head_dim, 0, 0);
        }
        if (!g.pending_error) {
            imparo_cuda_rope(
                k_buf, rope_dim, rope_base, head_dim, n_kv,
                start_pos, n_tok, freqs_host);
        }
        if (!g.pending_error && k_hadamard_nrot != 0) {
            imparo_cuda_hadamard(
                k_buf, n_tok * n_kv * head_dim, k_hadamard_nrot);
        }
        if (!g.pending_error && v_hadamard_nrot != 0) {
            imparo_cuda_hadamard(
                v_buf, n_tok * n_kv * head_dim, v_hadamard_nrot);
        }
        return;
    }

    OpEventScope profile("kv_head_post", head_dim, n_tok * n_kv);
    const uint8_t * norm_raw = nullptr;
    int rc = weight_slice(k_norm_off, uint64_t(head_dim) * sizeof(float), &norm_raw);
    if (rc) { set_pending(rc, "KV head postprocess norm weights"); return; }

    const float * freqs_dev = nullptr;
    if (freqs_host) {
        const uint64_t bytes = uint64_t(rope_dim / 2) * sizeof(float);
        rc = rope_frequency_buffer(freqs_host, bytes, &freqs_dev);
        if (rc) { set_pending(rc, "RoPE frequency upload"); return; }
    }

    const uint32_t requested = tuner_knob(8);
    const uint32_t threads = (requested == 256 || requested == 1024)
        ? requested : (head_dim < 1024 ? 256 : 1024);
    const float theta_scale = powf(rope_base, -2.0f / float(rope_dim));
    const float k_hadamard_scale = k_hadamard_nrot
        ? 1.0f / sqrtf(float(k_hadamard_nrot)) : 1.0f;
    const float v_hadamard_scale = v_hadamard_nrot
        ? 1.0f / sqrtf(float(v_hadamard_nrot)) : 1.0f;
    const uint32_t rows = n_tok * n_kv;
#define LAUNCH_KV_HEAD_POST(BLOCK_SIZE)                                             \
    k_head_norm_rope_hadamard<BLOCK_SIZE, true>                                    \
        <<<rows, BLOCK_SIZE, head_dim * sizeof(float), g.stream>>>(                 \
            static_cast<float *>(g.bufs[k_buf]),                                   \
            reinterpret_cast<const float *>(norm_raw), freqs_dev,                  \
            head_dim, eps, n_kv, start_pos, rope_dim, theta_scale,                  \
            k_hadamard_nrot, k_hadamard_scale, decode_control_arg(), nullptr);     \
    k_head_norm_rope_hadamard<BLOCK_SIZE, false>                                   \
        <<<rows, BLOCK_SIZE, head_dim * sizeof(float), g.stream>>>(                 \
            static_cast<float *>(g.bufs[v_buf]), nullptr, nullptr,                 \
            head_dim, eps, n_kv, start_pos, 0, 1.0f,                               \
            v_hadamard_nrot, v_hadamard_scale, decode_control_arg(), nullptr)
    if (threads == 1024) {
        LAUNCH_KV_HEAD_POST(1024);
    } else {
        LAUNCH_KV_HEAD_POST(256);
    }
#undef LAUNCH_KV_HEAD_POST
    if (g.graph_capturing) g.graph_expected_dynamic_nodes += 2;
    mark_buf_written(k_buf);
    mark_buf_written(v_buf);
}

extern "C" void imparo_cuda_kv_store(uint32_t src, uint32_t layer, uint32_t width,
                                     uint32_t start_pos, uint32_t n_tok, uint32_t is_v,
                                     uint32_t ring) {
    OpEventScope profile("kv_store", width, n_tok);
    const float * s = static_cast<const float *>(g.bufs[src]);
    void * dstb = is_v ? g.kv_v[layer] : g.kv_k[layer];
    const uint32_t kt = is_v ? g.kv_type_v : g.kv_type_k;
    const uint32_t * page_table = kv_device_page_table(layer);
    invalidate_kv_dequant(layer, is_v);
    if (kt == 2) {
        dim3 grid((width / 32 + 63) / 64, n_tok);
        k_kv_store_q4<<<grid, 64, 0, g.stream>>>(s, static_cast<uint8_t *>(dstb),
                                                 width, start_pos, n_tok, ring,
                                                 decode_control_arg(), page_table);
    } else if (kt == 8) {
        dim3 grid((width / 32 + 63) / 64, n_tok);
        k_kv_store_q8<<<grid, 64, 0, g.stream>>>(s, static_cast<uint8_t *>(dstb),
                                                 width, start_pos, n_tok, ring,
                                                 decode_control_arg(), page_table);
    } else {
        dim3 grid((width + 255) / 256, n_tok);
        k_kv_store_f16<<<grid, 256, 0, g.stream>>>(s, static_cast<__half *>(dstb),
                                                   width, start_pos, n_tok, ring,
                                                   decode_control_arg(), page_table);
    }
    if (g.graph_capturing) ++g.graph_expected_dynamic_nodes;
}

extern "C" void imparo_cuda_kv_dequant(uint32_t layer, uint32_t width, uint32_t slots,
                                       uint32_t is_v, uint32_t scratch_buf,
                                       uint32_t ring) {
    OpEventScope profile("kv_dequant", width, slots);
    const uint32_t kt = is_v ? g.kv_type_v : g.kv_type_k;
    if (kt == 1) return;
    const uint64_t required = uint64_t(width) * slots * sizeof(__half);
    if (layer >= MAX_LAYERS || scratch_buf >= B_COUNT || width == 0
        || width % 32 != 0 || slots == 0 || (kt != 2 && kt != 8)
        || !(is_v ? g.kv_v[layer] : g.kv_k[layer])
        || !g.bufs[scratch_buf] || required > g.sizes[scratch_buf]) {
        set_pending(CUDA_RC_INVALID, "kv_dequant scratch");
        return;
    }
    KvDequantKey & key = is_v ? g.vdq : g.kdq;
    if (key.matches(layer, width, slots, ring, scratch_buf, kt)) return;
    const auto * src = static_cast<const uint8_t *>(
        is_v ? g.kv_v[layer] : g.kv_k[layer]);
    auto * dst = static_cast<__half *>(g.bufs[scratch_buf]);
    const uint32_t * page_table = kv_device_page_table(layer);
    const bool parallel = g.sm_version >= 80
        && std::getenv("IMPARO_CUDA_NO_PARALLEL_KV_DEQUANT") == nullptr
        && imparo_sm80_kv::launch_dequant(
            src, dst, width, slots, kt, ring, page_table, g.stream);
    if (!parallel) {
        const dim3 grid((width / 32 + 63) / 64, slots);
        k_kv_dequant<<<grid, 64, 0, g.stream>>>(
            src, dst, width, slots, kt, ring, page_table);
    }
    mark_buf_written(scratch_buf);
    key.install(layer, width, slots, ring, scratch_buf, kt);
}

// Full-attention tensors in the pinned FA contract expose an architecture-owned
// padded key extent. Keep that execution geometry separate from the number of
// initialized cache rows: padded rows participate in scheduling and masking but
// must never be read from K/V storage. Ring caches already have a physical extent.
static uint32_t attention_schedule_span(
        uint32_t valid_span, uint32_t ring, uint32_t alignment) {
    if (ring || alignment <= 1) return valid_span;
    return uint32_t((uint64_t(valid_span) + alignment - 1) / alignment * alignment);
}

// Shared SM80 small-query scheduler for D64/D256/D512 attention families.
// The per-SM residency is a property of the pinned FA shapes; partition count is
// derived from the actual GPU topology and capped by available 32-key groups.
template <uint32_t CacheType>
static bool launch_attention_small(
        const void * kc, const void * vc, uint32_t head_dim,
        uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, float qk_scale, uint32_t window,
        uint32_t n_tok, uint32_t ring,
        uint32_t q_buf, uint32_t out_buf,
        const uint32_t * page_table,
        uint32_t q_token_offset = 0, uint32_t out_token_offset = 0) {
    const uint32_t valid_span = ring
        ? std::min(start_pos + n_tok, ring + 1) : start_pos + n_tok;
    uint32_t kv_span = attention_schedule_span(
        valid_span, ring, imparo_sm80_prefill::kScheduleKeys);
    // Unsaturated ring attention normally sizes its workspace to the exact valid
    // prefix. A graph is reused for the whole 32-key launch bucket, so retaining
    // that exact stride makes later replays silently drop newly appended keys even
    // though their store and valid-span control are correct. Capture the complete
    // bucket; score masking still excludes rows beyond the replay's valid prefix.
    if (g.graph_capturing && ring && valid_span != ring + 1) {
        const uint32_t bucket = imparo_sm80_d512_small::kKeyBatch;
        kv_span = std::min(
            uint32_t((uint64_t(valid_span) + bucket - 1) / bucket * bucket),
            ring + 1);
    }
    if (g.forward_decode && decode_graph_candidate() && ring
        && valid_span != ring + 1 && !decode_ring_bucket_enabled()) {
        g.decode_graph_capture_after = std::max(g.decode_graph_capture_after, ring);
    }
    const uint32_t key_groups =
        (kv_span + imparo_sm80_d512_small::kKeyBatch - 1)
        / imparo_sm80_d512_small::kKeyBatch;
    if (g.graph_capturing) {
        if (n_tok != 1
            || (ring && valid_span != ring + 1
                && !decode_ring_bucket_enabled())) {
            g.graph_capture_compatible = false;
        } else if (ring && valid_span != ring + 1) {
            // Before saturation, ring attention grows one row per token and its
            // physical grid changes only when a 32-key group is added. Restrict
            // replay to that exact schedule bucket; decode_prepare discards and
            // recaptures at the next boundary.
            const uint32_t bucket_min = key_groups
                ? (key_groups - 1) * imparo_sm80_d512_small::kKeyBatch : 0;
            const uint32_t bucket_max = std::min(
                key_groups * imparo_sm80_d512_small::kKeyBatch - 1,
                ring - 1);
            g.graph_min_start = std::max(g.graph_min_start, bucket_min);
            g.graph_max_start = std::min(g.graph_max_start, bucket_max);
        } else if (ring) {
            // A saturated graph has a fixed physical span, but it must never be
            // replayed for a new unsaturated conversation on the same backend.
            g.graph_min_start = std::max(g.graph_min_start, ring);
        } else if (!ring) {
            const uint32_t bucket_min = kv_span > imparo_sm80_prefill::kScheduleKeys
                ? kv_span - imparo_sm80_prefill::kScheduleKeys : 0;
            const uint32_t bucket_max = kv_span ? kv_span - 1 : 0;
            g.graph_min_start = std::max(g.graph_min_start, bucket_min);
            g.graph_max_start = std::min(g.graph_max_start, bucket_max);
        }
    }
    // Achieved occupancy is shape-owned: the pinned D256/ncols=16 tile reaches
    // two blocks per SM, as does D512/ncols=8 after its 41,760-byte dynamic
    // shared-memory footprint is accounted for. Stream-K seams are
    // numerical boundaries, so model achieved residency rather than confusing
    // launch-bound hints with the runtime occupancy of the exact kernel shape.
    const uint32_t resident_blocks_per_sm = 2u;
    const uint32_t resident_blocks = resident_blocks_per_sm
        * uint32_t(std::max(1, g.sm_count));
    const uint32_t topology_parts = std::max(1u, std::min(
        key_groups, resident_blocks / n_kv));
    const bool d256_full32 = CacheType == 1 && head_dim == 256 && n_tok == 1
        && std::getenv("IMPARO_CUDA_ATTN_D256_FULL32") != nullptr;
    uint32_t stream_part_cap = tuner_knob(18);
    const char * shape_cap_name = head_dim == 512
        ? "IMPARO_CUDA_ATTN_D512_STREAM_PART_CAP"
        : (head_dim == 256
            ? "IMPARO_CUDA_ATTN_D256_STREAM_PART_CAP"
            : "IMPARO_CUDA_ATTN_D64_STREAM_PART_CAP");
    const char * shape_cap = std::getenv(shape_cap_name);
    if (shape_cap) {
        // Diagnostic shape override: partition seams are part of the numerical
        // route, so each head dimension is isolatable without changing a shared knob.
        stream_part_cap = uint32_t(std::strtoul(shape_cap, nullptr, 10));
    } else if (!stream_part_cap) {
        static const uint32_t forced_stream_part_cap = [] {
            const char * value = std::getenv(
                "IMPARO_CUDA_ATTN_STREAM_PART_CAP");
            return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        stream_part_cap = forced_stream_part_cap;
    }
    if (!stream_part_cap) stream_part_cap = d256_full32 ? topology_parts : 8;
    const uint32_t stream_parts =
        std::max(1u, std::min(topology_parts, stream_part_cap));
    const uint64_t floats_per_block =
        uint64_t(imparo_sm80_d512_small::kColumns) * kv_span
        + uint64_t(2 * imparo_sm80_d512_small::kColumns) * key_groups
        + uint64_t(4 * imparo_sm80_d512_small::kColumns) * stream_parts;
    const uint64_t bytes_per_block = floats_per_block * sizeof(float);
    const uint64_t workspace_cap = std::max<uint64_t>(
        bytes_per_block, attention_workspace_budget(head_dim));
    uint32_t chunk_blocks = uint32_t(std::min<uint64_t>(
        n_kv, workspace_cap / bytes_per_block));
    chunk_blocks = std::max(1u, chunk_blocks);
    while (!ensure_attention_scratch(uint64_t(chunk_blocks) * bytes_per_block)
           && chunk_blocks > 1) {
        chunk_blocks = (chunk_blocks + 1) / 2;
    }
    if (!g.attention_scratch
        || g.attention_scratch_bytes < uint64_t(chunk_blocks) * bytes_per_block) {
        return false;
    }

    float * workspace = static_cast<float *>(g.attention_scratch);
    for (uint32_t base = 0; base < n_kv; base += chunk_blocks) {
        const uint32_t count = std::min(chunk_blocks, n_kv - base);
        {
            AttentionStageEventScope stage("small-scores", head_dim, count);
            if (head_dim == 64) {
            if (page_table) {
                imparo_sm80_d512_small::scores_paged<64, CacheType>
                    <<<dim3(count, key_groups), 64, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf])
                            + uint64_t(q_token_offset) * n_heads * head_dim,
                        kc, workspace, base,
                        n_heads, n_kv, kv_width, start_pos, qk_scale, window, n_tok,
                        ring, valid_span, kv_span, stream_parts, decode_control_arg(),
                        page_table);
            } else {
                imparo_sm80_d512_small::scores<64, CacheType>
                    <<<dim3(count, key_groups), 64, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf])
                            + uint64_t(q_token_offset) * n_heads * head_dim,
                        kc, workspace, base,
                        n_heads, n_kv, kv_width, start_pos, qk_scale, window, n_tok,
                        ring, valid_span, kv_span, stream_parts, decode_control_arg());
            }
            } else if (head_dim == 256) {
            if (page_table) {
                imparo_sm80_d512_small::scores_paged<256, CacheType>
                    <<<dim3(count, key_groups), 64, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf])
                            + uint64_t(q_token_offset) * n_heads * head_dim,
                        kc, workspace, base,
                        n_heads, n_kv, kv_width, start_pos, qk_scale, window, n_tok,
                        ring, valid_span, kv_span, stream_parts, decode_control_arg(),
                        page_table);
            } else {
                imparo_sm80_d512_small::scores<256, CacheType>
                    <<<dim3(count, key_groups), 64, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf])
                            + uint64_t(q_token_offset) * n_heads * head_dim,
                        kc, workspace, base,
                        n_heads, n_kv, kv_width, start_pos, qk_scale, window, n_tok,
                        ring, valid_span, kv_span, stream_parts, decode_control_arg());
            }
            } else {
            if (page_table) {
                imparo_sm80_d512_small::scores_paged<512, CacheType>
                    <<<dim3(count, key_groups), 64, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf])
                            + uint64_t(q_token_offset) * n_heads * head_dim,
                        kc, workspace, base,
                        n_heads, n_kv, kv_width, start_pos, qk_scale, window, n_tok,
                        ring, valid_span, kv_span, stream_parts, decode_control_arg(),
                        page_table);
            } else {
                imparo_sm80_d512_small::scores<512, CacheType>
                    <<<dim3(count, key_groups), 64, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf])
                            + uint64_t(q_token_offset) * n_heads * head_dim,
                        kc, workspace, base,
                        n_heads, n_kv, kv_width, start_pos, qk_scale, window, n_tok,
                        ring, valid_span, kv_span, stream_parts, decode_control_arg());
            }
            }
        }
        if constexpr (CacheType == 1) {
            if (d256_full32) {
                const uint32_t tasks =
                    imparo_sm80_d512_small::kColumns * stream_parts;
                {
                    AttentionStageEventScope stage("d256-full32-softmax", head_dim, count);
                    imparo_sm80_d256_decode::softmax_full32_parts
                        <<<dim3(count, (tasks + 3) / 4), 128, 0, g.stream>>>(
                            workspace, kv_span, stream_parts);
                }
                {
                    AttentionStageEventScope stage("d256-full32-values", head_dim, count);
                    imparo_sm80_d256_decode::values_full32_combine
                        <<<dim3(count, head_dim / 16), 32, 0, g.stream>>>(
                            static_cast<const __half *>(vc), workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base, n_heads, n_kv, kv_width, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                }
                if (g.graph_capturing) g.graph_expected_dynamic_nodes += 2;
                continue;
            }
        }
        uint32_t softmax_warps = tuner_knob(16);
        if (!softmax_warps) {
            static const uint32_t forced_softmax_warps = [] {
                const char * value = std::getenv(
                    "IMPARO_CUDA_ATTN_SOFTMAX_WARPS");
                return value ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
            }();
            softmax_warps = forced_softmax_warps;
        }
        if (softmax_warps != 1 && softmax_warps != 2
            && softmax_warps != 4 && softmax_warps != 8) {
            softmax_warps = 2;
        }
        const uint32_t softmax_tasks =
            2 * imparo_sm80_d512_small::kColumns * stream_parts;
        {
            AttentionStageEventScope stage("small-softmax", head_dim, count);
            imparo_sm80_d512_small::softmax_parts
                <<<dim3(count, (softmax_tasks + softmax_warps - 1)
                    / softmax_warps), 32 * softmax_warps, 0, g.stream>>>(
                        workspace, kv_span, stream_parts);
        }
        {
            AttentionStageEventScope stage("small-values", head_dim, count);
            uint32_t value_tiles = tuner_knob(17);
            if (!value_tiles) {
                static const uint32_t forced_value_tiles = [] {
                    const char * value = std::getenv(
                        "IMPARO_CUDA_ATTN_VALUE_TILES");
                    return value
                        ? uint32_t(std::strtoul(value, nullptr, 10)) : 0u;
                }();
                value_tiles = forced_value_tiles;
            }
            if (value_tiles != 1 && value_tiles != 2) value_tiles = 1;
            const dim3 value_grid(count, head_dim / (16 * value_tiles));
            if (head_dim == 64) {
                if (value_tiles == 2) {
                if (page_table) {
                    imparo_sm80_d512_small::values_combine_paged<64, CacheType, 2>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                } else {
                    imparo_sm80_d512_small::values_combine<64, CacheType, 2>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg());
                }
                } else {
                if (page_table) {
                    imparo_sm80_d512_small::values_combine_paged<64, CacheType, 1>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                } else {
                    imparo_sm80_d512_small::values_combine<64, CacheType, 1>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg());
                }
                }
            } else if (head_dim == 256) {
                if (value_tiles == 2) {
                if (page_table) {
                    imparo_sm80_d512_small::values_combine_paged<256, CacheType, 2>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                } else {
                    imparo_sm80_d512_small::values_combine<256, CacheType, 2>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg());
                }
                } else {
                if (page_table) {
                    imparo_sm80_d512_small::values_combine_paged<256, CacheType, 1>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                } else {
                    imparo_sm80_d512_small::values_combine<256, CacheType, 1>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg());
                }
                }
            } else {
                if (value_tiles == 2) {
                if (page_table) {
                    imparo_sm80_d512_small::values_combine_paged<512, CacheType, 2>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                } else {
                    imparo_sm80_d512_small::values_combine<512, CacheType, 2>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg());
                }
                } else {
                if (page_table) {
                    imparo_sm80_d512_small::values_combine_paged<512, CacheType, 1>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg(), page_table);
                } else {
                    imparo_sm80_d512_small::values_combine<512, CacheType, 1>
                        <<<value_grid, 64, 0, g.stream>>>(
                            vc, workspace,
                            static_cast<float *>(g.bufs[out_buf])
                                + uint64_t(out_token_offset) * n_heads * head_dim,
                            base,
                            n_heads, n_kv, kv_width, n_tok, ring, valid_span,
                            kv_span, stream_parts, decode_control_arg());
                }
                }
            }
        }
        if (g.graph_capturing) g.graph_expected_dynamic_nodes += 2;
    }
    return true;
}

// Native D64/GQA4 wide-prefill route.  Its bounded workspace is partitioned by
// KV head, so allocation pressure reduces concurrency rather than model fit.
// Returning false is a deliberate contract: the caller retains the proven
// small-query chain as a numerically conservative fallback.
static bool launch_attention_d64_wide(
        const void * kc, const void * vc,
        uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, float qk_scale, uint32_t window,
        uint32_t n_tok, uint32_t ring,
        uint32_t q_buf, uint32_t out_buf,
        const uint32_t * page_table) {
    if (n_kv == 0 || n_heads != n_kv * imparo_sm80_d64_wide::kGqaHeads
        || kv_width != n_kv * imparo_sm80_d64_wide::kHeadDim
        || n_tok == 0 || n_tok > imparo_sm80_d64_wide::kQueryTokens
        || q_buf >= B_COUNT || out_buf >= B_COUNT
        || !g.bufs[q_buf] || !g.bufs[out_buf] || !kc || !vc) {
        return false;
    }
    const uint64_t initialized = uint64_t(start_pos) + n_tok;
    if (initialized > UINT32_MAX || ring == UINT32_MAX) return false;
    const uint32_t valid_span = ring
        ? std::min(uint32_t(initialized), ring + 1)
        : uint32_t(initialized);
    if (!ring && uint64_t(valid_span)
            + imparo_sm80_prefill::kScheduleKeys - 1 > UINT32_MAX) {
        return false;
    }
    const uint32_t kv_span = attention_schedule_span(
        valid_span, ring, imparo_sm80_prefill::kScheduleKeys);
    const uint32_t blocks = imparo_sm80_d64_wide::key_blocks(kv_span);
    if (blocks == 0) return false;

    const uint64_t numerator_bytes =
        imparo_sm80_d64_wide::numerator_halves_per_head(blocks)
        * sizeof(__half);
    const uint64_t meta_bytes =
        imparo_sm80_d64_wide::meta_floats_per_head(blocks)
        * sizeof(float);
    if (numerator_bytes > UINT64_MAX - meta_bytes) return false;
    const uint64_t bytes_per_head = numerator_bytes + meta_bytes;
    if (bytes_per_head == 0 || uint64_t(n_kv) > UINT64_MAX / bytes_per_head)
        return false;
    const uint64_t workspace_cap = std::max<uint64_t>(
        bytes_per_head, attention_workspace_budget(
            imparo_sm80_d64_wide::kHeadDim));
    uint32_t chunk_heads = uint32_t(std::min<uint64_t>(
        n_kv, workspace_cap / bytes_per_head));
    chunk_heads = std::max(1u, chunk_heads);
    while (!ensure_attention_scratch(uint64_t(chunk_heads) * bytes_per_head)
           && chunk_heads > 1) {
        chunk_heads = (chunk_heads + 1) / 2;
    }
    if (!g.attention_scratch
        || g.attention_scratch_bytes
            < uint64_t(chunk_heads) * bytes_per_head) {
        return false;
    }

    for (uint32_t base = 0; base < n_kv; base += chunk_heads) {
        const uint32_t count = std::min(chunk_heads, n_kv - base);
        auto * numerators = static_cast<__half *>(g.attention_scratch);
        auto * meta = reinterpret_cast<float *>(
            static_cast<uint8_t *>(g.attention_scratch)
            + uint64_t(count) * numerator_bytes);
        {
            AttentionStageEventScope stage("d64-wide-partials", 64, count);
            imparo_sm80_d64_wide::partials
                <<<dim3(count, blocks), imparo_sm80_d64_wide::kThreads,
                    0, g.stream>>>(
                    static_cast<const float *>(g.bufs[q_buf]),
                    static_cast<const __half *>(kc),
                    static_cast<const __half *>(vc), numerators, meta, base,
                    n_heads, n_kv, kv_width, start_pos, window, n_tok, ring,
                    qk_scale, valid_span, blocks, page_table);
        }
        {
            AttentionStageEventScope stage("d64-wide-combine", 64, count);
            imparo_sm80_d64_wide::combine
                <<<count, 256, 0, g.stream>>>(
                    numerators, meta, static_cast<float *>(g.bufs[out_buf]),
                    base, n_heads, n_kv, n_tok, blocks);
        }
    }
    return true;
}

// Canonical SM86 LFM2 prefill: one CTA owns the complete K reduction for one
// 16-token x GQA4 tile. It needs no global fixup workspace; returning false keeps
// the established bounded Stream-K and scalar routes available as safe fallbacks.
static bool launch_attention_d64_mma_prefill(
        const void * kc, const void * vc,
        uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, float applied_q_scale, uint32_t window,
        uint32_t n_tok, uint32_t ring,
        uint32_t q_buf, uint32_t out_buf,
        const uint32_t * page_table) {
    if (n_kv == 0
        || n_heads != n_kv * imparo_sm80_d64_mma_plan::kGqaHeads
        || kv_width != n_kv * imparo_sm80_d64_mma_plan::kHeadDim
        || n_tok == 0 || q_buf >= B_COUNT || out_buf >= B_COUNT
        || !g.bufs[q_buf] || !g.bufs[out_buf] || !kc || !vc
        || !std::isfinite(applied_q_scale) || applied_q_scale <= 0.0f) {
        return false;
    }
    const uint64_t initialized = uint64_t(start_pos) + n_tok;
    if (initialized > UINT32_MAX || ring == UINT32_MAX) return false;
    const uint32_t valid_span = ring
        ? std::min(uint32_t(initialized), ring + 1)
        : uint32_t(initialized);
    const auto plan = imparo_sm80_d64_mma_plan::make(
        n_tok, n_kv, valid_span);
    if (!plan.valid) return false;
    const float kernel_applied_q_scale =
        std::getenv("IMPARO_CUDA_ATTN_D64_MMA_Q_AS_IS")
        ? 1.0f : applied_q_scale;
    const uint32_t shared_kv_min_tokens = tuner_knob(46);
    const bool shared_kv = shared_kv_min_tokens != 0
        && n_tok >= shared_kv_min_tokens
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_SHARED_KV") == nullptr;

    if (shared_kv) {
        imparo_sm80_d64_mma::whole_k_tile<true>
            <<<plan.blocks, imparo_sm86_d64_mma_profile::kThreads,
                imparo_sm86_d64_mma_profile::kDynamicSharedBytes, g.stream>>>(
                static_cast<const float *>(g.bufs[q_buf]),
                static_cast<const __half *>(kc), static_cast<const __half *>(vc),
                static_cast<float *>(g.bufs[out_buf]),
                n_heads, n_kv, kv_width, start_pos, window, n_tok, ring,
                kernel_applied_q_scale, valid_span, plan.key_updates, page_table);
    } else {
        imparo_sm80_d64_mma::whole_k_tile<false>
            <<<plan.blocks, imparo_sm86_d64_mma_profile::kThreads,
                imparo_sm86_d64_mma_profile::kDynamicSharedBytes, g.stream>>>(
                static_cast<const float *>(g.bufs[q_buf]),
                static_cast<const __half *>(kc), static_cast<const __half *>(vc),
                static_cast<float *>(g.bufs[out_buf]),
                n_heads, n_kv, kv_width, start_pos, window, n_tok, ring,
                kernel_applied_q_scale, valid_span, plan.key_updates, page_table);
    }
    if (std::getenv("IMPARO_CUDA_ATTN_D64_MMA_TRACE")) {
        std::fprintf(stderr,
            "[d64-mma-prefill] start=%u tokens=%u qtiles=%u keys=%u grid=%u scale=%g sm=%d shared_kv=%u\n",
            start_pos, n_tok, plan.query_tiles, plan.key_updates,
            plan.blocks, double(kernel_applied_q_scale), g.sm_version,
            unsigned(shared_kv));
    }
    return true;
}

// SM80 prefill can be wider than the architecture kernel's 16-column (four-token x
// four-GQA-head) ownership tile. Keep that proven fragment unchanged and cover the
// registered 32/64-column families with consecutive launches. Each tile advances its
// logical position and Q/output row base; the kernel's existing causal bound therefore
// sees exactly the prefix available to those rows.
template <uint32_t CacheType>
static bool launch_attention_small_chain(
        const void * kc, const void * vc, uint32_t head_dim,
        uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, float qk_scale, uint32_t window,
        uint32_t n_tok, uint32_t ring,
        uint32_t q_buf, uint32_t out_buf,
        const uint32_t * page_table) {
    for (uint32_t token = 0; token < n_tok;
         token += imparo_sm80_d512_small::kQueryTokens) {
        const uint32_t rows = std::min(
            imparo_sm80_d512_small::kQueryTokens, n_tok - token);
        if (!launch_attention_small<CacheType>(
                kc, vc, head_dim, n_heads, n_kv, kv_width,
                start_pos + token, qk_scale, window, rows, ring,
                q_buf, out_buf, page_table, token, token)) {
            return false;
        }
    }
    return true;
}

extern "C" void imparo_cuda_attention(uint32_t kv_layer, uint32_t head_dim,
                                       uint32_t n_heads, uint32_t n_kv,
                                       uint32_t kv_width, uint32_t start_pos,
                                       float applied_q_scale, uint32_t window, uint32_t n_tok, uint32_t ring,
                                        uint32_t q_buf, uint32_t out_buf,
                                        uint32_t kdq_buf, uint32_t vdq_buf) {
    if (kv_layer >= MAX_LAYERS || q_buf >= B_COUNT || out_buf >= B_COUNT
        || n_kv == 0 || n_heads == 0 || n_heads % n_kv != 0
        || head_dim == 0 || kv_width == 0 || n_tok == 0
        || uint64_t(start_pos) + n_tok > UINT32_MAX
        || !std::isfinite(applied_q_scale) || applied_q_scale <= 0.0f) {
        set_pending(CUDA_RC_INVALID, "attention shape");
        return;
    }
    mark_tuner_dispatch(2, (uint64_t(head_dim) << 32) | n_tok);
    // Q is pre-scaled by the shared workflow. Established kernels consume it as-is;
    // only a route that explicitly reconstructs conversion order uses this metadata.
    constexpr float qk_scale = 1.0f;
    OpEventScope profile("attention", head_dim, n_tok);
    const uint32_t * stable_page_table = kv_device_page_table(kv_layer);
    const bool page_mapping = ring == 0
        && kv_page_table_requires_mapping(kv_layer);
    const uint32_t * page_table = page_mapping ? stable_page_table : nullptr;
    // Prefill dequantizes quantized cache rows into bounded half scratch in the
    // common workflow. Consume that scratch here so all KV types share the exact
    // SM80 FA scheduler.
    // Quantized decode stages the initialized cache slice to half, matching llama's
    // FA contract and feeding the parallel SM80 tail family. Global-attention Q4 decode
    // instead dequantizes while loading K/V: its full history amortizes that work better
    // than materializing a second cache. The 512-token sliding path remains staged because
    // direct loads regress that workload. The policy is semantic rather than device/model
    // named, and the diagnostic opt-out keeps both paths measurable.
    const bool specialized_decode = tuner_knob(36) != 0;
    const bool d512_mma_requested = specialized_decode && g.sm_version == 86
        && head_dim == imparo_sm80_d512_decode::kHeadDim
        && n_tok == 1 && n_heads / n_kv == imparo_sm80_d512_decode::kGqaHeads
        && window == 0 && ring == 0
        && kv_width == n_kv * imparo_sm80_d512_decode::kHeadDim
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && std::getenv("IMPARO_CUDA_NO_ATTN_D512_MMA") == nullptr
        && tuner_knob(33);
    uint32_t d512_valid_span = 0;
    uint32_t d512_schedule_span = 0;
    uint32_t d512_materialized_slots = 0;
    uint32_t d512_bucket_min = 0;
    uint32_t d512_bucket_max = 0;
    bool d512_mma_f16 = false;
    bool d512_mma_q4 = false;
    if (d512_mma_requested && kv_layer < MAX_LAYERS
        && kdq_buf < B_COUNT && vdq_buf < B_COUNT
        && g.kv_k[kv_layer] && g.kv_v[kv_layer]
        && g.bufs[kdq_buf] && g.bufs[vdq_buf]
        && kv_width % 32 == 0) {
        const uint64_t q4_row_bytes = uint64_t(kv_width / 32) * 18;
        const uint64_t f16_row_bytes = uint64_t(kv_width) * sizeof(__half);
        const uint64_t valid = uint64_t(start_pos) + 1;
        const uint64_t schedule = (valid
            + imparo_sm80_prefill::kScheduleKeys - 1)
            / imparo_sm80_prefill::kScheduleKeys
            * imparo_sm80_prefill::kScheduleKeys;
        const uint64_t raw_capacity = q4_row_bytes
            ? g.kv_bytes[kv_layer] / q4_row_bytes : 0;
        const uint64_t kdq_capacity = f16_row_bytes
            ? g.sizes[kdq_buf] / f16_row_bytes : 0;
        const uint64_t vdq_capacity = f16_row_bytes
            ? g.sizes[vdq_buf] / f16_row_bytes : 0;
        const uint64_t capacity = std::min(
            raw_capacity, std::min(kdq_capacity, vdq_capacity));
        const uint64_t materialized = std::min(schedule, capacity);
        // Slot 23 owns the minimum absolute schedule span for this numerical
        // route. The conservative default keeps the verified direct-Q4 path
        // through the first three 256-key buckets; unlike final-chunk routing,
        // this decision depends only on absolute position and is prefix-stable.
        // Step 7 exposes the slot through the backend registry so tuning can
        // raise (but never silently lower) the correctness floor.
        const uint64_t d512_mma_min_schedule = tuner_knob(23)
            ? uint64_t(tuner_knob(23))
            : uint64_t(4) * imparo_sm80_prefill::kScheduleKeys;
        if (schedule >= d512_mma_min_schedule
            && valid > 0 && valid <= materialized
            && schedule <= UINT32_MAX && materialized <= UINT32_MAX) {
            d512_valid_span = uint32_t(valid);
            d512_schedule_span = uint32_t(schedule);
            d512_materialized_slots = uint32_t(materialized);
            d512_bucket_min = d512_schedule_span
                - imparo_sm80_prefill::kScheduleKeys;
            d512_bucket_max = d512_materialized_slots - 1;
            d512_mma_f16 = true;
        }
    }
    // Keep the staged-F16 implementation as the exact fallback and switch only
    // the cache-load side of its verified D512 numerical class. The direct route
    // is deliberately SM86/Q4/decode-only through d512_mma_requested above.
    if (d512_mma_f16
        && std::getenv("IMPARO_CUDA_NO_ATTN_D512_Q4_MMA") == nullptr) {
        d512_mma_q4 = true;
        d512_mma_f16 = false;
    }
    const bool direct_q4_decode = specialized_decode && n_tok == 1
        && (window == 0 || std::getenv("IMPARO_CUDA_Q4_ATTN_DIRECT_ALL"))
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && !d512_mma_f16
        && std::getenv("IMPARO_CUDA_NO_Q4_ATTN_DIRECT") == nullptr;
    // LFM2's D64 full-attention decode may retain the exact conservative arithmetic
    // in a separate graph symbol. Fail closed for dequant/staged routes: their launch
    // extent grows with the absolute position and is not a stable replay contract.
    const bool d64_graph_q4 = n_kv != 0 && direct_q4_decode
        && head_dim == 64 && n_tok == 1 && window == 0 && ring == 0
        && n_heads / n_kv == 4 && kv_width == n_kv * 64;
    const bool d64_q8_vec_requested = tuner_knob(41) != 0
        || std::getenv("IMPARO_CUDA_ATTN_D64_VEC_Q8") != nullptr;
    const bool d64_graph_q8 = n_kv != 0 && specialized_decode
        && d64_q8_vec_requested
        && imparo_sm86_d64_q4_vec_profile::applies_to(g.sm_version)
        && head_dim == 64 && n_tok == 1 && window == 0 && ring == 0
        && n_heads / n_kv == 4 && kv_width == n_kv * 64
        && g.kv_type_k == 8 && g.kv_type_v == 8
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_VEC_Q8") == nullptr;
    if (std::getenv("IMPARO_CUDA_ATTN_D64_VEC_TRACE")) {
        static uint32_t checks = 0;
        if (checks++ < 32) {
            std::fprintf(stderr,
                "[d64-q8-check] sm=%d specialized=%u tok=%u hd=%u heads=%u/%u "
                "kv_width=%u window=%u ring=%u graph=%u kv=%u/%u vec=%u gqa4=%u\n",
                g.sm_version, unsigned(specialized_decode), n_tok, head_dim,
                n_heads, n_kv, kv_width, window, ring,
                unsigned(g.graph_capturing), g.kv_type_k, g.kv_type_v,
                unsigned(d64_q8_vec_requested), unsigned(g.knobs[49] != 0));
        }
    }
    // D256 quantized decode is a vector-FA contract, not a small-MMA shape.
    // Keep the first implementation opt-in until strict decode and determinism
    if (head_dim == 256 && n_tok == 1
        && std::getenv("IMPARO_CUDA_ATTN_D256_VEC_TRACE")) {
        std::fprintf(stderr,
            "[d256-vec] sm=%d layer=%u start=%u window=%u ring=%u heads=%u/%u kv=%u/%u\n",
            g.sm_version, kv_layer, start_pos, window, ring, n_heads, n_kv,
            g.kv_type_k, g.kv_type_v);
    }
    imparo_sm80_d256_vec_plan::Geometry d256_vec_geometry;
    const bool d256_vec_plan = imparo_sm80_d256_vec_plan::geometry(
        start_pos, n_tok, ring, &d256_vec_geometry);
    const bool d256_vec_q4 = specialized_decode && g.sm_version == 86
        && head_dim == imparo_sm80_d256_vec::kHeadDim
        && n_tok == 1
        && n_heads / n_kv == imparo_sm80_d256_vec::kGqaHeads
        && window == imparo_sm80_d256_vec::kWindowSpan
        && d256_vec_plan
        // P2/P3 geometry now has an exact ownership/Graph contract, but the
        // vector kernel's arithmetic is not yet certified against pinned llama.
        // Keep the historical P4 production floor; the early buckets remain an
        // explicit diagnostic until their strict eight-step gate passes.
        && (d256_vec_geometry.schedule_span
                == imparo_sm80_d256_vec::kMaxPhysicalSpan
            || std::getenv("IMPARO_CUDA_ATTN_D256_VEC_EARLY"))
        && start_pos + n_tok >= imparo_sm80_d256_vec::kWindowSpan
        && kv_width == n_kv * imparo_sm80_d256_vec::kHeadDim
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && tuner_knob(30)
        && std::getenv("IMPARO_CUDA_NO_ATTN_D256_VEC") == nullptr;
    if (d256_vec_q4) {
        if (std::getenv("IMPARO_CUDA_ATTN_D256_VEC_TRACE")) {
            std::fprintf(stderr,
                "[d256-vec] HIT partials=%u stripes=%u schedule=%u valid=%u "
                "capacity=%u window=%u ring=%u\n",
                d256_vec_geometry.partials,
                imparo_sm80_d256_vec::kStripesPerPartial,
                d256_vec_geometry.schedule_span, d256_vec_geometry.valid_span,
                d256_vec_geometry.capacity, window, ring);
        }
        const bool d256_gqa4 = n_heads == n_kv * imparo_sm80_d256_vec::kGqaHeads
            && d256_vec_geometry.schedule_span
                == imparo_sm80_d256_vec::kMaxPhysicalSpan
            // Admitted only for the fixed SM86 P4/GQA4/Q4 route above. Keep the
            // established per-head kernel as an immediate operational rollback.
            && std::getenv("IMPARO_CUDA_NO_ATTN_D256_GQA4") == nullptr;
        const bool d256_gqa4_compare = d256_gqa4
            && std::getenv("IMPARO_CUDA_ATTN_D256_GQA4_COMPARE") != nullptr
            && !g.graph_capturing;
        const uint64_t partial_floats = uint64_t(n_heads)
            * imparo_sm80_d256_vec::kMaxPartials
            * imparo_sm80_d256_vec::kPartialStride;
        const uint64_t child_floats = d256_gqa4
            ? uint64_t(n_heads) * imparo_sm80_d256_vec::kMaxPartials
                * imparo_sm80_d256_vec::kWarps
                * imparo_sm80_d256_vec::kGqaChildStride
            : 0;
        const uint64_t compare_floats =
            d256_gqa4_compare ? partial_floats : 0;
        if (ensure_attention_scratch(
                (child_floats + partial_floats + compare_floats)
                    * sizeof(float))) {
            auto * scratch = static_cast<float *>(g.attention_scratch);
            auto * partials = scratch + child_floats;
            if (d256_gqa4) {
                imparo_sm80_d256_vec::partial_q4_gqa4
                    <<<dim3(n_kv * imparo_sm80_d256_vec::kWarps,
                             d256_vec_geometry.partials),
                        dim3(32, imparo_sm80_d256_vec::kWarps), 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf]),
                        static_cast<const uint8_t *>(g.kv_k[kv_layer]),
                        static_cast<const uint8_t *>(g.kv_v[kv_layer]), scratch,
                        n_heads, n_kv, kv_width, start_pos, ring,
                        d256_vec_geometry.schedule_span, qk_scale,
                        decode_control_arg());
                if (d256_gqa4_compare) {
                    imparo_sm80_d256_vec::combine_q4_gqa4
                        <<<dim3(n_heads, d256_vec_geometry.partials),
                            imparo_sm80_d256_vec::kHeadDim, 0, g.stream>>>(
                            scratch, partials, n_heads,
                            d256_vec_geometry.partials);
                    auto * reference_partials = partials + partial_floats;
                    imparo_sm80_d256_vec::partial_q4
                        <<<dim3(n_heads, d256_vec_geometry.partials),
                            dim3(32, imparo_sm80_d256_vec::kWarps),
                            0, g.stream>>>(
                            static_cast<const float *>(g.bufs[q_buf]),
                            static_cast<const uint8_t *>(g.kv_k[kv_layer]),
                            static_cast<const uint8_t *>(g.kv_v[kv_layer]),
                            reference_partials, n_heads, n_kv, kv_width,
                            start_pos, ring, d256_vec_geometry.schedule_span,
                            qk_scale, decode_control_arg());
                    std::vector<float> candidate(
                        static_cast<size_t>(partial_floats), 0.0f);
                    std::vector<float> reference(
                        static_cast<size_t>(partial_floats), 0.0f);
                    const cudaError_t c0 = cudaMemcpyAsync(
                        candidate.data(), partials,
                        size_t(partial_floats) * sizeof(float),
                        cudaMemcpyDeviceToHost, g.stream);
                    const cudaError_t c1 = cudaMemcpyAsync(
                        reference.data(), reference_partials,
                        size_t(partial_floats) * sizeof(float),
                        cudaMemcpyDeviceToHost, g.stream);
                    const cudaError_t cs = cudaStreamSynchronize(g.stream);
                    if (c0 == cudaSuccess && c1 == cudaSuccess
                        && cs == cudaSuccess) {
                        uint64_t mismatches = 0;
                        float max_abs = 0.0f;
                        uint64_t max_index = 0;
                        for (uint64_t i = 0; i < partial_floats; ++i) {
                            if (std::memcmp(&candidate[size_t(i)],
                                            &reference[size_t(i)],
                                            sizeof(float)) != 0) {
                                ++mismatches;
                                const float delta = std::fabs(
                                    candidate[size_t(i)]
                                    - reference[size_t(i)]);
                                if (delta > max_abs) {
                                    max_abs = delta;
                                    max_index = i;
                                }
                            }
                        }
                        std::fprintf(stderr,
                            "[d256-gqa4-compare] layer=%u start=%u "
                            "mismatch=%llu/%llu max_abs=%.9g index=%llu "
                            "candidate=%.9g reference=%.9g\n",
                            kv_layer, start_pos,
                            static_cast<unsigned long long>(mismatches),
                            static_cast<unsigned long long>(partial_floats),
                            double(max_abs),
                            static_cast<unsigned long long>(max_index),
                            double(candidate[size_t(max_index)]),
                            double(reference[size_t(max_index)]));
                    } else {
                        std::fprintf(stderr,
                            "[d256-gqa4-compare] CUDA copy/sync failed "
                            "layer=%u start=%u errors=%d/%d/%d\n",
                            kv_layer, start_pos, int(c0), int(c1), int(cs));
                    }
                }
            } else {
                imparo_sm80_d256_vec::partial_q4
                    <<<dim3(n_heads, d256_vec_geometry.partials),
                        dim3(32, imparo_sm80_d256_vec::kWarps), 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[q_buf]),
                    static_cast<const uint8_t *>(g.kv_k[kv_layer]),
                    static_cast<const uint8_t *>(g.kv_v[kv_layer]), partials,
                    n_heads, n_kv, kv_width, start_pos, ring,
                    d256_vec_geometry.schedule_span, qk_scale,
                    decode_control_arg());
            }
            if (g.graph_capturing) {
                g.graph_min_start = std::max(g.graph_min_start,
                    imparo_sm80_d256_vec_plan::graph_bucket_min(
                        d256_vec_geometry));
                g.graph_max_start = std::min(g.graph_max_start,
                    imparo_sm80_d256_vec_plan::graph_bucket_max(
                        d256_vec_geometry));
                ++g.graph_expected_dynamic_nodes;
            }
            if (d256_gqa4 && !d256_gqa4_compare) {
                imparo_sm80_d256_vec::combine_q4_gqa4_final
                    <<<n_heads, imparo_sm80_d256_vec::kHeadDim,
                        0, g.stream>>>(
                        scratch, static_cast<float *>(g.bufs[out_buf]),
                        n_heads, d256_vec_geometry.partials);
            } else {
                imparo_sm80_d256_vec::combine_q4
                    <<<n_heads, imparo_sm80_d256_vec::kHeadDim, 0, g.stream>>>(
                        partials, static_cast<float *>(g.bufs[out_buf]), n_heads,
                        d256_vec_geometry.partials);
            }
            mark_buf_written(out_buf);
            return;
        }
    }
    const bool d64_vec_q4 =
        imparo_sm86_d64_q4_vec_profile::applies_to(g.sm_version)
        && direct_q4_decode
        && n_kv != 0 && head_dim == imparo_sm80_d64_q4_vec::kHeadDim
        && n_heads / n_kv == imparo_sm80_d64_q4_vec::kGqaHeads
        && kv_width == n_kv * imparo_sm80_d64_q4_vec::kHeadDim
        && (!g.graph_capturing || d64_graph_q4)
        && std::getenv("IMPARO_CUDA_ATTN_D64_VEC_Q4") != nullptr
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_VEC_Q4") == nullptr;
    if (d64_vec_q4) {
        uint32_t part_cap = 0;
        if (const char * value = std::getenv(
                "IMPARO_CUDA_ATTN_D64_VEC_PART_CAP")) {
            part_cap = uint32_t(std::strtoul(value, nullptr, 10));
        }
        imparo_sm80_d64_q4_vec_plan::Geometry geometry;
        if (imparo_sm80_d64_q4_vec_plan::geometry(
                start_pos, ring, uint32_t(std::max(1, g.sm_count)),
                imparo_sm86_d64_q4_vec_profile::kMaxBlocksPerSm,
                n_heads, part_cap, &geometry)) {
            const uint64_t partial_floats = uint64_t(n_heads)
                * geometry.parts * imparo_sm80_d64_q4_vec::kPartialStride;
            if (partial_floats <= UINT64_MAX / sizeof(float)
                && ensure_attention_scratch(partial_floats * sizeof(float))) {
                auto * partials = static_cast<float *>(g.attention_scratch);
                imparo_sm80_d64_q4_vec::partial_q4
                    <<<dim3(n_heads, geometry.parts),
                        dim3(32, imparo_sm80_d64_q4_vec::kWarps), 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf]),
                        static_cast<const uint8_t *>(g.kv_k[kv_layer]),
                        static_cast<const uint8_t *>(g.kv_v[kv_layer]),
                        partials, n_heads, n_kv, kv_width, start_pos, qk_scale,
                        window, ring, geometry.schedule_span,
                        decode_control_arg(), page_table);
                imparo_sm80_d64_q4_vec::combine_q4
                    <<<n_heads, imparo_sm80_d64_q4_vec::kHeadDim, 0, g.stream>>>(
                        partials, static_cast<float *>(g.bufs[out_buf]),
                        n_heads, geometry.parts);
                if (g.graph_capturing) {
                    g.graph_min_start = std::max(g.graph_min_start,
                        imparo_sm80_d64_q4_vec_plan::graph_bucket_min(geometry));
                    g.graph_max_start = std::min(g.graph_max_start,
                        imparo_sm80_d64_q4_vec_plan::graph_bucket_max(geometry));
                    ++g.graph_expected_dynamic_nodes;
                }
                if (std::getenv("IMPARO_CUDA_ATTN_D64_VEC_TRACE")) {
                    std::fprintf(stderr,
                        "[d64-q4-vec] layer=%u start=%u valid=%u schedule=%u "
                        "stripes=%u parts=%u ring=%u window=%u paged=%u\n",
                        kv_layer, start_pos, geometry.valid_span,
                        geometry.schedule_span, geometry.stripes, geometry.parts,
                        ring, window, page_table ? 1u : 0u);
                }
                mark_buf_written(out_buf);
                return;
            }
        }
    }
    const bool d64_vec_q8 =
        imparo_sm86_d64_q4_vec_profile::applies_to(g.sm_version)
        && specialized_decode && n_tok == 1 && n_kv != 0
        && head_dim == imparo_sm80_d64_q8_vec::kHeadDim
        && n_heads / n_kv == imparo_sm80_d64_q8_vec::kGqaHeads
        && kv_width == n_kv * imparo_sm80_d64_q8_vec::kHeadDim
        && window == 0 && ring == 0
        && (!g.graph_capturing || d64_graph_q8)
        && g.kv_type_k == 8 && g.kv_type_v == 8
        && d64_q8_vec_requested
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_VEC_Q8") == nullptr;
    if (d64_vec_q8) {
        const bool gqa4_shared_kv = tuner_knob(49) != 0
            && std::getenv("IMPARO_CUDA_NO_ATTN_D64_Q8_GQA4") == nullptr;
        uint32_t part_cap = 0;
        if (const char * value = std::getenv(
                "IMPARO_CUDA_ATTN_D64_VEC_Q8_PART_CAP")) {
            part_cap = uint32_t(std::strtoul(value, nullptr, 10));
        }
        imparo_sm80_d64_q4_vec_plan::Geometry geometry;
        if (imparo_sm80_d64_q4_vec_plan::geometry(
                start_pos, ring, uint32_t(std::max(1, g.sm_count)),
                imparo_sm86_d64_q4_vec_profile::kMaxBlocksPerSm,
                n_heads, part_cap, &geometry)) {
            const uint64_t child_floats = uint64_t(n_heads)
                * geometry.parts * imparo_sm80_d64_q8_vec::kWarps
                * imparo_sm80_d64_q8_vec::kGqa4ChildStride;
            const uint64_t final_shared_bytes =
                imparo_sm80_d64_q8_vec::gqa4_final_shared_bytes(
                    geometry.parts);
            if (gqa4_shared_kv
                && child_floats <= UINT64_MAX / sizeof(float)
                && final_shared_bytes <= UINT32_MAX
                && ensure_attention_scratch(child_floats * sizeof(float))) {
                auto * children = static_cast<float *>(g.attention_scratch);
                imparo_sm80_d64_q8_vec::partial_q8_gqa4
                    <<<dim3(n_heads, geometry.parts),
                        dim3(32, imparo_sm80_d64_q8_vec::kWarps),
                        0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf]),
                        static_cast<const uint8_t *>(g.kv_k[kv_layer]),
                        static_cast<const uint8_t *>(g.kv_v[kv_layer]),
                        children, n_heads, n_kv, kv_width, start_pos, qk_scale,
                        window, ring, geometry.schedule_span,
                        decode_control_arg(), page_table);
                imparo_sm80_d64_q8_vec::combine_q8_gqa4_final
                    <<<n_heads, imparo_sm80_d64_q8_vec::kHeadDim,
                        size_t(final_shared_bytes), g.stream>>>(
                        children, static_cast<float *>(g.bufs[out_buf]),
                        n_heads, geometry.parts);
                if (g.graph_capturing) {
                    g.graph_min_start = std::max(g.graph_min_start,
                        imparo_sm80_d64_q4_vec_plan::graph_bucket_min(geometry));
                    g.graph_max_start = std::min(g.graph_max_start,
                        imparo_sm80_d64_q4_vec_plan::graph_bucket_max(geometry));
                    ++g.graph_expected_dynamic_nodes;
                }
                if (std::getenv("IMPARO_CUDA_ATTN_D64_VEC_TRACE")) {
                    std::fprintf(stderr,
                        "[d64-q8-gqa4] layer=%u start=%u valid=%u schedule=%u "
                        "stripes=%u parts=%u child_kib=%.1f\n",
                        kv_layer, start_pos, geometry.valid_span,
                        geometry.schedule_span, geometry.stripes, geometry.parts,
                        double(child_floats * sizeof(float)) / 1024.0);
                }
                mark_buf_written(out_buf);
                return;
            }
            const uint64_t partial_floats = uint64_t(n_heads)
                * geometry.parts * imparo_sm80_d64_q8_vec::kPartialStride;
            if (partial_floats <= UINT64_MAX / sizeof(float)
                && ensure_attention_scratch(partial_floats * sizeof(float))) {
                auto * partials = static_cast<float *>(g.attention_scratch);
                imparo_sm80_d64_q8_vec::partial_q8
                    <<<dim3(n_heads, geometry.parts),
                        dim3(32, imparo_sm80_d64_q8_vec::kWarps),
                        0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf]),
                        static_cast<const uint8_t *>(g.kv_k[kv_layer]),
                        static_cast<const uint8_t *>(g.kv_v[kv_layer]),
                        partials, n_heads, n_kv, kv_width, start_pos, qk_scale,
                        window, ring, geometry.schedule_span,
                        decode_control_arg(), page_table);
                imparo_sm80_d64_q4_vec::combine_q4
                    <<<n_heads, imparo_sm80_d64_q8_vec::kHeadDim,
                        0, g.stream>>>(
                        partials, static_cast<float *>(g.bufs[out_buf]),
                        n_heads, geometry.parts);
                if (g.graph_capturing) {
                    g.graph_min_start = std::max(g.graph_min_start,
                        imparo_sm80_d64_q4_vec_plan::graph_bucket_min(geometry));
                    g.graph_max_start = std::min(g.graph_max_start,
                        imparo_sm80_d64_q4_vec_plan::graph_bucket_max(geometry));
                    ++g.graph_expected_dynamic_nodes;
                }
                if (std::getenv("IMPARO_CUDA_ATTN_D64_VEC_TRACE")) {
                    std::fprintf(stderr,
                        "[d64-q8-vec] layer=%u start=%u valid=%u schedule=%u "
                        "stripes=%u parts=%u\n",
                        kv_layer, start_pos, geometry.valid_span,
                        geometry.schedule_span, geometry.stripes, geometry.parts);
                }
                mark_buf_written(out_buf);
                return;
            }
        }
    }
    if (g.graph_capturing && d64_graph_q4) {
        const uint32_t requested = tuner_knob(7);
        const uint32_t threads =
            (requested == 64 || requested == 128 || requested == 256)
                ? requested : 128;
        // D64 is LFM2's attention route. The pinned CUDA/FA oracle keeps the
        // probability-weighted V reduction in f32; half accumulation misses the
        // fixed n=2000 decode distribution gate even with an f16 cache. Keep the
        // environment switch as a diagnostic override for the wider families.
        const uint32_t f32_v_accum =
            std::getenv("IMPARO_CUDA_ATTN_D64_HALF") ? 0u : 1u;
        k_attention_d64_controlled<<<dim3(n_heads, 1), threads,
                head_dim * sizeof(float), g.stream>>>(
            static_cast<const float *>(g.bufs[q_buf]), g.kv_k[kv_layer],
            g.kv_v[kv_layer], static_cast<float *>(g.bufs[out_buf]),
            head_dim, n_heads, n_kv, kv_width, start_pos, qk_scale,
            window, ring, n_tok, 2, 2, f32_v_accum, page_table,
            decode_control_arg());
        ++g.graph_expected_dynamic_nodes;
        mark_buf_written(out_buf);
        return;
    }
    const bool d256_tiled4 = g.sm_version >= 80
        && head_dim == 256 && n_tok == 4 && n_heads / n_kv == 4
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && tuner_knob(29);
    // Full attention rounds its physical schedule to a fixed key bucket and has no
    // ring overwrite, so consecutive four-row architecture tiles can evaluate the
    // backend's complete verification width directly from Q4 while retaining every
    // per-row MMA/reduction.
    // Sliding attention uses the staged-half route below.
    const bool grouped_q4_verify = n_tok > 1
        && n_tok <= imparo_sm80_mmvq::kMaxTokens
        && window == 0 && ring == 0
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && std::getenv("IMPARO_CUDA_NO_Q4_ATTN_DIRECT") == nullptr
        && std::getenv("IMPARO_CUDA_NO_Q4_VERIFY_GROUPED") == nullptr;
    if (grouped_q4_verify
        && launch_attention_small_chain<2>(
            g.kv_k[kv_layer], g.kv_v[kv_layer], head_dim,
            n_heads, n_kv, kv_width, start_pos, qk_scale, window,
            n_tok, ring, q_buf, out_buf, page_table)) {
        mark_buf_written(out_buf);
        return;
    }
    // Sliding layers consume the workflow's staged-half cache. The small-query
    // kernel keeps one causal mask per row and maps every physical ring slot from
    // the tile's final batch position; unsaturated and saturated schedules are therefore
    // grouped without changing the reusable Q4 storage contract.
    const bool grouped_q4_verify_sliding = n_tok > 1
        && n_tok <= imparo_sm80_mmvq::kMaxTokens && !d256_tiled4
        && window > 0
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && kdq_buf < B_COUNT && vdq_buf < B_COUNT
        && g.bufs[kdq_buf] && g.bufs[vdq_buf]
        && std::getenv("IMPARO_CUDA_NO_Q4_VERIFY_GROUPED") == nullptr
        && std::getenv("IMPARO_CUDA_NO_Q4_ATTN_DIRECT") == nullptr;
    if (grouped_q4_verify_sliding
        && launch_attention_small_chain<1>(
            g.bufs[kdq_buf], g.bufs[vdq_buf], head_dim,
            n_heads, n_kv, kv_width, start_pos, qk_scale, window,
            n_tok, ring, q_buf, out_buf, page_table)) {
        mark_buf_written(out_buf);
        return;
    }
    // Remaining quantized target verification keeps projections batched but
    // executes each causal query with the exact one-token Q4 attention schedule.
    const bool sequential_q4_verify = n_tok > 1
        && n_tok <= imparo_sm80_mmvq::kMaxTokens && !d256_tiled4
        && g.kv_type_k == 2 && g.kv_type_v == 2
        && std::getenv("IMPARO_CUDA_NO_Q4_ATTN_DIRECT") == nullptr;
    if (sequential_q4_verify) {
        const bool direct_cache = window == 0
            || std::getenv("IMPARO_CUDA_Q4_ATTN_DIRECT_ALL");
        const void * verify_k = direct_cache
            ? g.kv_k[kv_layer] : g.bufs[kdq_buf];
        const void * verify_v = direct_cache
            ? g.kv_v[kv_layer] : g.bufs[vdq_buf];
        if (!verify_k || !verify_v) {
            set_pending(CUDA_RC_INVALID, "Q4 verifier attention cache");
            return;
        }
        bool launched = true;
        for (uint32_t token = 0; token < n_tok; ++token) {
            launched = direct_cache
                ? launch_attention_small<2>(
                    verify_k, verify_v, head_dim, n_heads, n_kv, kv_width,
                    start_pos + token, qk_scale, window, 1, ring,
                    q_buf, out_buf, page_table, token, token)
                : launch_attention_small<1>(
                    verify_k, verify_v, head_dim, n_heads, n_kv, kv_width,
                    start_pos + token, qk_scale, window, 1, ring,
                    q_buf, out_buf, page_table, token, token);
            if (!launched) break;
        }
        if (launched) { mark_buf_written(out_buf); return; }
    }
    const bool decode_dequant = n_tok == 1 && !direct_q4_decode;
    if (decode_dequant) {
        uint32_t slots = d512_mma_f16 ? d512_materialized_slots
            : (ring ? std::min(start_pos + n_tok, ring + 1)
                    : start_pos + n_tok);
        // A decode graph is reusable across one 32-key schedule bucket. Capture
        // enough dequantized rows for the whole bucket: retaining the capture
        // position's shorter grid leaves later replay positions reading stale half
        // rows even though scores/values correctly advance their valid span.
        if (g.graph_capturing && ring) {
            const uint32_t bucket = imparo_sm80_d512_small::kKeyBatch;
            slots = std::min(
                uint32_t((uint64_t(slots) + bucket - 1) / bucket * bucket),
                ring + 1);
        }
        if (g.kv_type_k != 1 && kdq_buf < B_COUNT && g.bufs[kdq_buf]
            && (g.graph_capturing
                || !g.kdq.matches(kv_layer, kv_width, slots, ring,
                                  kdq_buf, g.kv_type_k))) {
            const auto * src = static_cast<const uint8_t *>(g.kv_k[kv_layer]);
            auto * dst = static_cast<__half *>(g.bufs[kdq_buf]);
            const bool parallel = g.sm_version >= 80
                && std::getenv("IMPARO_CUDA_NO_PARALLEL_KV_DEQUANT") == nullptr
                && imparo_sm80_kv::launch_dequant(
                    src, dst, kv_width, slots, g.kv_type_k, ring,
                    stable_page_table, g.stream);
            if (!parallel) {
                k_kv_dequant<<<dim3((kv_width / 32 + 63) / 64, slots), 64, 0,
                    g.stream>>>(src, dst, kv_width, slots, g.kv_type_k, ring,
                    stable_page_table);
            }
            mark_buf_written(kdq_buf);
            g.kdq.install(kv_layer, kv_width, slots, ring, kdq_buf, g.kv_type_k);
        }
        if (g.kv_type_v != 1 && vdq_buf < B_COUNT && g.bufs[vdq_buf]
            && (g.graph_capturing
                || !g.vdq.matches(kv_layer, kv_width, slots, ring,
                                  vdq_buf, g.kv_type_v))) {
            const auto * src = static_cast<const uint8_t *>(g.kv_v[kv_layer]);
            auto * dst = static_cast<__half *>(g.bufs[vdq_buf]);
            const bool parallel = g.sm_version >= 80
                && std::getenv("IMPARO_CUDA_NO_PARALLEL_KV_DEQUANT") == nullptr
                && imparo_sm80_kv::launch_dequant(
                    src, dst, kv_width, slots, g.kv_type_v, ring,
                    stable_page_table, g.stream);
            if (!parallel) {
                k_kv_dequant<<<dim3((kv_width / 32 + 63) / 64, slots), 64, 0,
                    g.stream>>>(src, dst, kv_width, slots, g.kv_type_v, ring,
                    stable_page_table);
            }
            mark_buf_written(vdq_buf);
            g.vdq.install(kv_layer, kv_width, slots, ring, vdq_buf, g.kv_type_v);
        }
    }
    // Diagnostic: keep the absolute prefill schedule unchanged while consuming the
    // reusable Q4 cache directly. This isolates Q4 attention arithmetic from the
    // common staged-half route; it is opt-in until the numerical gate and performance
    // both justify a dedicated SM-family kernel.
    const bool direct_q4_prefill = n_tok > 1
        && std::getenv("IMPARO_CUDA_Q4_PREFILL_DIRECT") != nullptr;
    const bool use_kdq = (n_tok > 1 || decode_dequant) && g.kv_type_k != 1
        && !(direct_q4_prefill && g.kv_type_k == 2)
        && kdq_buf < B_COUNT && g.bufs[kdq_buf];
    const bool use_vdq = (n_tok > 1 || decode_dequant) && g.kv_type_v != 1
        && !(direct_q4_prefill && g.kv_type_v == 2)
        && vdq_buf < B_COUNT && g.bufs[vdq_buf];
    const void * kc = use_kdq ? g.bufs[kdq_buf] : g.kv_k[kv_layer];
    const void * vc = use_vdq ? g.bufs[vdq_buf] : g.kv_v[kv_layer];
    const uint32_t ktype = use_kdq ? 1u : g.kv_type_k;
    const uint32_t vtype = use_vdq ? 1u : g.kv_type_v;
    const bool d64_mma_prefill = tuner_knob(37) != 0
        && imparo_sm86_d64_mma_profile::applies_to(g.sm_version)
        && head_dim == imparo_sm80_d64_mma_plan::kHeadDim
        && n_kv != 0
        && n_heads / n_kv == imparo_sm80_d64_mma_plan::kGqaHeads
        && n_tok > 1
        && window == 0 && ring == 0 && ktype == 1 && vtype == 1
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_MMA_PREFILL") == nullptr;
    if (d64_mma_prefill
        && launch_attention_d64_mma_prefill(
            kc, vc, n_heads, n_kv, kv_width, start_pos,
            applied_q_scale, window, n_tok, ring,
            q_buf, out_buf, page_table)) {
        mark_buf_written(out_buf);
        return;
    }
    // The pinned D64/64-column route owns all four query fragments in one CTA and
    // combines fixed 64-key partitions from last to first.  Preserve that seam
    // contract for 9--16 token prefill; the existing small-query chain remains the
    // bounded-workspace and diagnostic fallback.
    const bool d64_f16_wide = g.sm_version >= 80 && n_kv != 0
        && head_dim == imparo_sm80_d64_wide::kHeadDim
        && n_heads / n_kv == imparo_sm80_d64_wide::kGqaHeads
        && n_tok > 2 * imparo_sm80_d512_small::kQueryTokens
        && n_tok <= imparo_sm80_d64_wide::kQueryTokens
        && ktype == 1 && vtype == 1
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_WIDE") == nullptr;
    if (d64_f16_wide
        && launch_attention_d64_wide(
            kc, vc, n_heads, n_kv, kv_width, start_pos, qk_scale, window,
            n_tok, ring, q_buf, out_buf, page_table)) {
        mark_buf_written(out_buf);
        return;
    }
    // Preserve the registered 16-column fragments when the native route is
    // disabled or cannot acquire even a one-head scratch slice.
    const bool d64_f16_chain = g.sm_version >= 80
        && head_dim == 64 && n_heads / n_kv == 4
        && n_tok > imparo_sm80_d512_small::kQueryTokens
        && n_tok <= imparo_sm80_prefill::kWideQueryTokens
        && ktype == 1 && vtype == 1
        && std::getenv("IMPARO_CUDA_NO_ATTN_D64_CHAIN") == nullptr;
    if (d64_f16_chain
        && launch_attention_small_chain<1>(
            kc, vc, head_dim, n_heads, n_kv, kv_width,
            start_pos, qk_scale, window, n_tok, ring,
            q_buf, out_buf, page_table)) {
        mark_buf_written(out_buf);
        return;
    }
    if (d512_mma_q4 && ktype == 2 && vtype == 2) {
        const uint32_t * control = decode_control_arg();
        const bool graph_controlled = g.graph_capturing && control;
        const uint32_t valid_span = graph_controlled
            ? d512_materialized_slots : d512_valid_span;
        const uint32_t kv_span = d512_schedule_span;
        const uint32_t schedule_groups = (kv_span
            + imparo_sm80_d512_decode::kKeyBatch - 1)
            / imparo_sm80_d512_decode::kKeyBatch;
        const uint32_t resident_blocks = 2u
            * uint32_t(std::max(1, g.sm_count));
        const uint32_t physical_blocks = std::min(
            resident_blocks, schedule_groups * n_kv);
        const uint64_t q_elements = uint64_t(n_heads) * head_dim;
        const uint64_t q_bytes = q_elements * sizeof(__half);
        const uint64_t workspace_bytes =
            imparo_sm80_d512_decode::workspace_floats(physical_blocks)
            * sizeof(float);
        if (imparo_sm80_d512_decode::supports_schedule(
                n_heads, n_kv, schedule_groups, physical_blocks)
                && ensure_attention_q_cache(q_bytes)
                && ensure_attention_scratch(workspace_bytes)) {
            auto * q_cache = static_cast<__half *>(g.attention_q_cache);
            const uint64_t q_pairs = q_elements / 2;
            imparo_sm80_prefill::cache_scaled_q
                <<<(q_pairs + 255) / 256, 256, 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[q_buf]),
                    reinterpret_cast<__half2 *>(q_cache), q_pairs, qk_scale);
            auto * workspace = static_cast<float *>(g.attention_scratch);
            const auto * q4_k = static_cast<const uint8_t *>(g.kv_k[kv_layer]);
            const auto * q4_v = static_cast<const uint8_t *>(g.kv_v[kv_layer]);
            if (g.graph_capturing) {
                g.graph_min_start = std::max(
                    g.graph_min_start, d512_bucket_min);
                g.graph_max_start = std::min(
                    g.graph_max_start, d512_bucket_max);
                ++g.graph_expected_dynamic_nodes;
            }
            if (graph_controlled) {
                if (page_table) {
                    imparo_sm80_d512_decode::partial_q4_controlled_paged
                        <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                            0, g.stream>>>(
                            q_cache, q4_k, q4_v, workspace,
                            n_heads, n_kv, kv_width, start_pos, window, ring,
                            valid_span, schedule_groups, physical_blocks, control,
                            page_table);
                } else {
                    imparo_sm80_d512_decode::partial_q4_controlled
                        <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                            0, g.stream>>>(
                            q_cache, q4_k, q4_v, workspace,
                            n_heads, n_kv, kv_width, start_pos, window, ring,
                            valid_span, schedule_groups, physical_blocks, control);
                }
            } else if (page_table) {
                imparo_sm80_d512_decode::partial_q4_paged
                    <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                        0, g.stream>>>(
                        q_cache, q4_k, q4_v, workspace,
                        n_heads, n_kv, kv_width, start_pos, window, ring,
                        valid_span, schedule_groups, physical_blocks, page_table);
            } else {
                imparo_sm80_d512_decode::partial_q4
                    <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                        0, g.stream>>>(
                        q_cache, q4_k, q4_v, workspace,
                        n_heads, n_kv, kv_width, start_pos, window, ring,
                        valid_span, schedule_groups, physical_blocks);
            }
            imparo_sm80_d512_decode::combine_reverse
                <<<dim3(n_kv, imparo_sm80_d512_decode::kLiveColumns),
                    256, 0, g.stream>>>(
                    workspace, static_cast<float *>(g.bufs[out_buf]),
                    n_heads, n_kv, schedule_groups, physical_blocks);
            if (std::getenv("IMPARO_CUDA_ATTN_D512_MMA_TRACE")) {
                std::fprintf(stderr,
                    "[d512-mma-q4] layer=%u start=%u valid=%u groups=%u blocks=%u sm=%d\n",
                    kv_layer, start_pos, d512_valid_span, schedule_groups,
                    physical_blocks, g.sm_version);
            }
            mark_buf_written(out_buf);
            return;
        }
    }
    if (d512_mma_f16 && ktype == 1 && vtype == 1) {
        const uint32_t * control = decode_control_arg();
        const bool graph_controlled = g.graph_capturing && control;
        const uint32_t valid_span = graph_controlled
            ? d512_materialized_slots : d512_valid_span;
        const uint32_t kv_span = d512_schedule_span;
        const uint32_t schedule_groups = (kv_span
            + imparo_sm80_d512_decode::kKeyBatch - 1)
            / imparo_sm80_d512_decode::kKeyBatch;
        const uint32_t resident_blocks = 2u
            * uint32_t(std::max(1, g.sm_count));
        const uint32_t physical_blocks = std::min(
            resident_blocks, schedule_groups * n_kv);
        const uint64_t q_elements = uint64_t(n_heads) * head_dim;
        const uint64_t q_bytes = q_elements * sizeof(__half);
        const uint64_t workspace_bytes =
            imparo_sm80_d512_decode::workspace_floats(physical_blocks)
            * sizeof(float);
        if (imparo_sm80_d512_decode::supports_schedule(
                n_heads, n_kv, schedule_groups, physical_blocks)
                && ensure_attention_q_cache(q_bytes)
                && ensure_attention_scratch(workspace_bytes)) {
            auto * q_cache = static_cast<__half *>(g.attention_q_cache);
            const uint64_t q_pairs = q_elements / 2;
            imparo_sm80_prefill::cache_scaled_q
                <<<(q_pairs + 255) / 256, 256, 0, g.stream>>>(
                    static_cast<const float *>(g.bufs[q_buf]),
                    reinterpret_cast<__half2 *>(q_cache), q_pairs, qk_scale);
            auto * workspace = static_cast<float *>(g.attention_scratch);
            if (g.graph_capturing) {
                g.graph_min_start = std::max(
                    g.graph_min_start, d512_bucket_min);
                g.graph_max_start = std::min(
                    g.graph_max_start, d512_bucket_max);
                ++g.graph_expected_dynamic_nodes;
            }
            if (graph_controlled) {
                if (page_table) {
                    imparo_sm80_d512_decode::partial_f16_controlled_paged
                        <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                            0, g.stream>>>(
                            q_cache, static_cast<const __half *>(kc),
                            static_cast<const __half *>(vc), workspace,
                            n_heads, n_kv, kv_width, start_pos, window, ring,
                            valid_span, schedule_groups, physical_blocks, control,
                            page_table);
                } else {
                    imparo_sm80_d512_decode::partial_f16_controlled
                        <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                            0, g.stream>>>(
                            q_cache, static_cast<const __half *>(kc),
                            static_cast<const __half *>(vc), workspace,
                            n_heads, n_kv, kv_width, start_pos, window, ring,
                            valid_span, schedule_groups, physical_blocks, control);
                }
            } else if (page_table) {
                imparo_sm80_d512_decode::partial_f16_paged
                    <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                        0, g.stream>>>(
                        q_cache, static_cast<const __half *>(kc),
                        static_cast<const __half *>(vc), workspace,
                        n_heads, n_kv, kv_width, start_pos, window, ring,
                        valid_span, schedule_groups, physical_blocks, page_table);
            } else {
                imparo_sm80_d512_decode::partial_f16
                    <<<physical_blocks, imparo_sm80_d512_decode::kThreads,
                        0, g.stream>>>(
                        q_cache, static_cast<const __half *>(kc),
                        static_cast<const __half *>(vc), workspace,
                        n_heads, n_kv, kv_width, start_pos, window, ring,
                        valid_span, schedule_groups, physical_blocks);
            }
            imparo_sm80_d512_decode::combine_reverse
                <<<dim3(n_kv, imparo_sm80_d512_decode::kLiveColumns),
                    256, 0, g.stream>>>(
                    workspace, static_cast<float *>(g.bufs[out_buf]),
                    n_heads, n_kv, schedule_groups, physical_blocks);
            if (std::getenv("IMPARO_CUDA_ATTN_D512_MMA_TRACE")) {
                std::fprintf(stderr,
                    "[d512-mma] layer=%u start=%u valid=%u groups=%u blocks=%u sm=%d\n",
                    kv_layer, start_pos, d512_valid_span, schedule_groups,
                    physical_blocks, g.sm_version);
            }
            mark_buf_written(out_buf);
            return;
        }
    }
    dim3 grid(n_heads, n_tok);
    const uint32_t requested = tuner_knob(7);
    const uint32_t threads = (requested == 64 || requested == 128 || requested == 256)
        ? requested : 128;
    const bool small_shape = g.sm_version >= 80
        && (!g.forward_decode || specialized_decode)
        && (head_dim == 64 || head_dim == 256 || head_dim == 512)
        && n_heads / n_kv == 4
        && n_tok > 0 && n_tok <= imparo_sm80_d512_small::kQueryTokens
        // The fixed small scheduler owns a 4-token x 4-GQA (16-column) tile.
        // D64 n=1/2 uses the reference's narrower 8-column family instead.
        && (head_dim != 64 || n_tok >= 3)
        // D512 prefill follows the pinned 2/4/8/16-token tile selector. The staged
        // implementation begins at four tokens, so <=2 remains on the conservative
        // fallback while prompt tails from 3 onward are padded and masked below.
        && (head_dim != 512 || g.forward_decode)
        && !d256_tiled4
        && std::getenv("IMPARO_CUDA_NO_ATTN_SMALL") == nullptr
        && ktype == vtype && (ktype == 1 || (ktype == 2 && direct_q4_decode));
    if (small_shape) {
        const bool launched = ktype == 1
            ? launch_attention_small<1>(
                kc, vc, head_dim, n_heads, n_kv, kv_width, start_pos, qk_scale,
                window, n_tok, ring, q_buf, out_buf, page_table)
            : launch_attention_small<2>(
                kc, vc, head_dim, n_heads, n_kv, kv_width, start_pos, qk_scale,
                window, n_tok, ring, q_buf, out_buf, page_table);
        if (launched) { mark_buf_written(out_buf); return; }
    }
    if (g.forward_decode && decode_graph_candidate()
        && !d64_graph_q4 && !d64_graph_q8) {
        g.decode_graph_shape_blocked = true;
    }
    // Reaching the tiled Prefill family is expected for a multi-token Graph.
    // This point remains incompatible only for Decode capture, whose dynamic
    // node updater supports the dedicated small-query symbols above.
    if (g.graph_capturing && !g.prefill_capture_active)
        g.graph_capture_compatible = false;
    // The SM80 tiled family owns every prefill beyond the <=4-query tail kernel.
    // Shape-specific opt-outs are diagnostics only; the bounded-workspace fallback
    // remains available if a specialization is disabled or cannot allocate scratch.
    const bool tiled_medium = n_tok > imparo_sm80_d512_small::kQueryTokens
        && n_tok <= imparo_sm80_prefill::kMediumQueryTokens
        && !std::getenv("IMPARO_CUDA_NO_ATTN_MEDIUM_TILED");
    const bool tiled_wide = n_tok > imparo_sm80_prefill::kMediumQueryTokens
        && n_tok <= imparo_sm80_prefill::kWideQueryTokens
        && !std::getenv("IMPARO_CUDA_NO_ATTN_WIDE_TILED");
    const uint32_t selected_d512_query_tokens = head_dim == 512
        ? imparo_sm80_prefill::d512_query_tokens(n_tok) : 0u;
    const bool stable_d512_candidate = head_dim == 512 && !g.forward_decode
        && n_tok > imparo_sm80_prefill::kMediumQueryTokens && ring == 0
        // A stable cell is selected from the tile's absolute first query. If a
        // caller submits a tile crossing that cell boundary, the second half
        // would require a different active-key domain. Fail closed to the
        // physical policy unless the complete tile is naturally aligned.
        && start_pos % selected_d512_query_tokens == 0
        && !std::getenv("IMPARO_CUDA_ATTN_LEGACY_PHYSICAL_STREAM");
    const bool tiled_d512 = head_dim == 512
        && imparo_sm80_prefill::d512_staged_tile_supported(n_tok)
        && selected_d512_query_tokens >= 4
        && !std::getenv("IMPARO_CUDA_NO_ATTN_D512_TILED");
    // D256 changes its MMA contract with tile width. At 32/64 columns the
    // reference uses two warps per query fragment and combines the independent
    // 16-key halves; at 64 columns it uses one warp and accumulates both halves
    // into the same half fragment. The <=4 family above owns 16 columns.
    // The receipted exact-128 bundle uses the established batch32 fallback with
    // float value accumulation. This is part of one atomic numerical route with
    // its PLE/FFN selectors: the standalone tiled kernel remains available for
    // every other shape and for the safe-off configuration.
    // Lab-only layer mask for finding the smallest correctness-preserving
    // exact subset. Absence keeps the receipt-bound all-layer atomic route;
    // production never reads a partial mask. The sweep is parsed by the CUDA
    // CRT once, avoiding Windows cross-CRT environment mutation.
    static const std::vector<uint64_t> exact128_batch32_mask_sweep = [] {
        std::vector<uint64_t> masks;
        const char * cursor = std::getenv(
            "IMPARO_CUDA_ATTN_D256_EXACT128_BATCH32_MASK_SWEEP_LAB");
        while (cursor && *cursor) {
            char * end = nullptr;
            const uint64_t mask = std::strtoull(cursor, &end, 0);
            if (end == cursor) break;
            masks.push_back(mask);
            cursor = end;
            while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
                ++cursor;
            }
        }
        return masks;
    }();
    const char * exact128_batch32_mask_env = std::getenv(
        "IMPARO_CUDA_ATTN_D256_EXACT128_BATCH32_MASK_LAB");
    const uint64_t exact128_batch32_mask =
        !exact128_batch32_mask_sweep.empty()
        ? exact128_batch32_mask_sweep[std::min<size_t>(
            g.prefill_lab_sweep_index,
            exact128_batch32_mask_sweep.size() - 1)]
        : (exact128_batch32_mask_env
            ? std::strtoull(exact128_batch32_mask_env, nullptr, 0)
            : UINT64_MAX);
    const bool exact128_batch32_layer = kv_layer < 64
        && ((exact128_batch32_mask >> kv_layer) & 1u) != 0;
    const bool exact128_partial_mask_lab = exact128_batch32_mask_env
        || !exact128_batch32_mask_sweep.empty();
    const bool exact128_d256_batch32 = head_dim == 256 && n_tok == 128
        && g.sm_version == 86 && tuned_exact128_sm86_route(n_tok)
        && exact128_batch32_layer
        && (std::getenv("IMPARO_CUDA_ATTN_D256_F32_FA_LAB") == nullptr
            || exact128_partial_mask_lab);
    const bool d256_f32_fa_lab = head_dim == 256 && n_tok == 128
        && g.sm_version == 86
        && std::getenv("IMPARO_CUDA_ATTN_D256_F32_FA_LAB") != nullptr;
    const bool d256_staged_f32_values_lab = head_dim == 256 && n_tok == 128
        && g.sm_version == 86
        && (tuned_exact128_staged_f32_attention(n_tok)
            || std::getenv("IMPARO_CUDA_ATTN_D256_STAGED_F32_VALUES_LAB") != nullptr);
    const bool tiled_d256 = head_dim == 256
        && n_tok >= imparo_sm80_d512_small::kQueryTokens
        // Quarantined after the onto-v2 n=128 gate: this route changed the layer-0
        // attention output enough to change top-1, while the bounded fallback agreed
        // with both the CPU oracle and upstream llama. Keep the implementation for
        // repair/A-B work, but correctness must be opt-in until it is recertified.
        && (tuner_knob(29) || d256_f32_fa_lab)
        && !exact128_d256_batch32
        && std::getenv("IMPARO_CUDA_NO_ATTN_D256_TILED") == nullptr;
    const bool tiled_d512_shape = head_dim == 512
        && (tiled_medium || tiled_wide || tiled_d512);
    if (g.sm_version >= 80 && (tiled_d512_shape || tiled_d256)
        && n_heads / n_kv == 4
        && ktype == 1 && vtype == 1) {
        const uint32_t valid_span = ring
            ? std::min(start_pos + n_tok, ring + 1) : start_pos + n_tok;
        const imparo_sm80_prefill::D512PartitionPolicy d512_partition_policy =
            stable_d512_candidate
                ? imparo_sm80_prefill::D512PartitionPolicy::StableCell2
                : imparo_sm80_prefill::D512PartitionPolicy::PhysicalStream;
        const imparo_sm80_prefill::D512AttentionGeometry geometry_d512 =
            imparo_sm80_prefill::d512_geometry(
                valid_span, n_tok, d512_partition_policy);
        const bool stable_cell_d512 = head_dim == 512
            && geometry_d512.partition_policy
                == imparo_sm80_prefill::D512PartitionPolicy::StableCell2;
        const uint32_t kv_span = head_dim == 512 && ring == 0
            ? geometry_d512.physical_keys
            : attention_schedule_span(
                valid_span, ring, imparo_sm80_prefill::kScheduleKeys);
        const uint32_t query_tokens = head_dim == 512
            ? geometry_d512.query_tokens
            : (n_tok <= 4 ? 4
                : (n_tok <= imparo_sm80_prefill::kMediumQueryTokens
                    ? imparo_sm80_prefill::kMediumQueryTokens
                    : imparo_sm80_prefill::kWideQueryTokens));
        const uint32_t columns = query_tokens * imparo_sm80_prefill::kGqaHeads;
        const uint32_t column_tiles = columns / 16;
        const uint32_t query_tiles = (n_tok + query_tokens - 1) / query_tokens;
        const uint32_t logical_blocks = query_tiles * n_kv;
        const uint32_t key_groups = (kv_span + imparo_sm80_prefill::kKeyBatch - 1)
            / imparo_sm80_prefill::kKeyBatch;
        const uint32_t schedule_groups = head_dim == 512 && ring == 0
            ? geometry_d512.schedule_groups
            : ((kv_span + imparo_sm80_prefill::kScheduleKeys - 1)
                / imparo_sm80_prefill::kScheduleKeys)
                * (imparo_sm80_prefill::kScheduleKeys
                    / imparo_sm80_prefill::kKeyBatch);
        // Ampere's pinned D512 kernel has one resident block per SM. Mirror the
        // reference Stream-K decision and its physical-grid seams; later SM
        // families own their different residency in their own architecture layer.
        const uint32_t occupancy = head_dim == 256 && query_tokens == 4 ? 2u
            : (head_dim == 512
                && query_tokens == imparo_sm80_prefill::kWideQueryTokens ? 1u : 2u);
        const uint32_t resident_blocks = uint32_t(std::max(1, g.sm_count)) * occupancy;
        const bool virtual_stream_requested = stable_cell_d512 && tuner_knob(34);
        // This is an enum, not a truncated u64 mask: 0 selects no virtual cells and
        // therefore the verified whole-after route; 1 selects every logical cell.
        const uint64_t virtual_stream_cell_mask = tuner_knob(35) == 1
            ? UINT64_MAX : uint64_t(0);
        const uint32_t absolute_stream_cell =
            start_pos / imparo_sm80_prefill::kCanonicalPrefillCellTokens;
        const bool stable_virtual_stream_d512 = virtual_stream_requested
            && absolute_stream_cell < 64
            && (virtual_stream_cell_mask & (uint64_t(1) << absolute_stream_cell));
        const bool stable_virtual_stream_d256 = head_dim == 256
            && query_tokens == imparo_sm80_prefill::kWideQueryTokens
            && start_pos % query_tokens == 0 && tuner_knob(31);
        const bool stable_whole_after_virtual = virtual_stream_requested
            && !stable_virtual_stream_d512;
        uint32_t virtual_stream_blocks = 0u;
        uint32_t virtual_logical_blocks = 0u;
        if (stable_virtual_stream_d512 || stable_virtual_stream_d256) {
            const uint32_t virtual_query_tiles =
                imparo_sm80_prefill::kCanonicalPrefillCellTokens / query_tokens;
            virtual_logical_blocks = virtual_query_tiles * n_kv;
            uint32_t virtual_schedule_groups =
                imparo_sm80_prefill::canonical_active_groups(start_pos);
            if (stable_virtual_stream_d256 && ring > 0) {
                const uint32_t ring_capacity_groups =
                    ((ring + 1 + imparo_sm80_prefill::kScheduleKeys - 1)
                        / imparo_sm80_prefill::kScheduleKeys)
                    * (imparo_sm80_prefill::kScheduleKeys
                        / imparo_sm80_prefill::kKeyBatch);
                virtual_schedule_groups =
                    std::min(virtual_schedule_groups, ring_capacity_groups);
            }
            const uint64_t virtual_total_work =
                uint64_t(virtual_logical_blocks) * virtual_schedule_groups;
            const uint32_t raw = uint32_t(std::min<uint64_t>(
                resident_blocks, virtual_total_work));
            const uint32_t rounded = virtual_logical_blocks > 0
                ? (raw / virtual_logical_blocks) * virtual_logical_blocks : 0u;
            const uint32_t loss = raw > 0
                ? 100u * (raw - rounded) / raw : 100u;
            virtual_stream_blocks = loss <= 5u ? rounded : raw;
        }
        const uint32_t tile_waves = (logical_blocks + resident_blocks - 1)
            / resident_blocks;
        const uint32_t tile_efficiency = tile_waves > 0
            ? 100u * logical_blocks / (resident_blocks * tile_waves) : 100u;
        const bool stream_k = tile_efficiency < 75u;
        const uint32_t stream_blocks = stream_k && resident_blocks <= logical_blocks
            ? std::min(resident_blocks, key_groups * logical_blocks) : 0u;
        const uint32_t stream_raw = std::min(
            resident_blocks, schedule_groups * logical_blocks);
        const uint32_t stream_rounded = logical_blocks > 0
            ? (stream_raw / logical_blocks) * logical_blocks : 0u;
        const uint32_t efficiency_loss = stream_raw > 0
            ? 100u * (stream_raw - stream_rounded) / stream_raw : 100u;
        // Canonical D512 uses two partitions derived from each absolute query's
        // 512-token scheduling cell. The part count is numerical-route identity;
        // caller batch width and occupancy may change launch order but not seams.
        uint32_t canonical_parts = stable_cell_d512
            ? imparo_sm80_prefill::kCanonicalD512Parts : 0u;
        if (stable_virtual_stream_d512) canonical_parts = 0u;
        if (stable_whole_after_virtual) canonical_parts = 1u;
        if (stable_cell_d512 && !virtual_stream_requested) {
            if (const char * forced = std::getenv(
                    "IMPARO_CUDA_ATTN_PARTS_PER_TILE")) {
                const uint32_t parsed = uint32_t(std::strtoul(forced, nullptr, 10));
                if (parsed > 0) canonical_parts = parsed;
            }
            canonical_parts = std::min(canonical_parts, schedule_groups);
        }
        const bool use_physical_stream = !stable_cell_d512
            && std::getenv("IMPARO_CUDA_NO_ATTN_PHYSICAL_STREAM") == nullptr
            && (g.sm_version >= 89 || tile_efficiency < 75u);
        const uint32_t physical_blocks = use_physical_stream
            ? (efficiency_loss <= 5u ? stream_rounded : stream_raw)
            : logical_blocks;
        if (head_dim == 512
            && std::getenv("IMPARO_CUDA_ATTN_D512_PREFILL_TRACE") != nullptr) {
            std::fprintf(stderr,
                "[d512-prefill] layer=%u start=%u n_tok=%u policy=%s "
                "virtual=%u whole_after=%u parts=%u logical=%u physical=%u "
                "groups=%u\n",
                kv_layer, start_pos, n_tok,
                stable_cell_d512 ? "stable-cell" : "physical-stream",
                stable_virtual_stream_d512 ? 1u : 0u,
                stable_whole_after_virtual ? 1u : 0u,
                canonical_parts, logical_blocks, physical_blocks,
                schedule_groups);
        }
        const uint32_t segment_slots =
            (stable_virtual_stream_d512 || stable_virtual_stream_d256)
            ? imparo_sm80_prefill::stream_segment_slot_bound(
                virtual_logical_blocks, virtual_stream_blocks)
            : (stable_cell_d512
                ? canonical_parts
                : (physical_blocks + logical_blocks - 1) / logical_blocks + 1);
        const bool partitioned_values = head_dim == 512
            || (head_dim == 256
                && query_tokens <= imparo_sm80_prefill::kMediumQueryTokens);
        const uint32_t d256_segment_bound = head_dim == 256
            ? imparo_sm80_prefill::stream_segment_slot_bound(
                logical_blocks, physical_blocks)
            : 0u;
        const bool fused_d256 = head_dim == 256
            && g.sm_version == 86
            && query_tokens == imparo_sm80_prefill::kWideQueryTokens
            // The fused kernel currently keeps at most two physical or stable
            // virtual Stream-K segments per logical tile. Wider schedules must
            // use the staged N-way combiner rather than dropping middle segments.
            && segment_slots <= 2
            && (tuner_knob(32) || d256_f32_fa_lab)
            && !d256_staged_f32_values_lab
            && std::getenv("IMPARO_CUDA_NO_ATTN_D256_FUSED") == nullptr;
        const bool fused_d512_stable = head_dim == 512
            && g.sm_version == 86 && stable_cell_d512 && ring == 0
            && canonical_parts > 0 && virtual_stream_blocks == 0
            && segment_slots <= imparo_sm80_prefill::kCanonicalD512Parts;
        const bool fused_d512_physical = head_dim == 512
            && !stable_cell_d512;
        const bool fused_d512 = head_dim == 512
            && (fused_d512_stable || fused_d512_physical)
            && query_tokens == imparo_sm80_prefill::kWideQueryTokens
            // The receipt-backed SM86 StableCell route is default-on. Keep the
            // older physical-stream laboratory route explicit opt-in, and keep
            // one kill switch that restores the staged scores/softmax/values path.
            && (fused_d512_stable
                || std::getenv("IMPARO_CUDA_ENABLE_ATTN_D512_FUSED") != nullptr)
            && std::getenv("IMPARO_CUDA_NO_ATTN_D512_FUSED") == nullptr
            && imparo_sm80_fa_d512::configure<1, 1>();
        const bool fused_d512_output_split_lab = fused_d512
            && std::getenv("IMPARO_CUDA_ATTN_D512_OUTPUT_SPLIT_LAB") != nullptr
            && imparo_sm80_fa_d512::configure<2, 1>();
        const bool fused_d512_column_split_lab = fused_d512
            && !fused_d512_output_split_lab
            && std::getenv("IMPARO_CUDA_ATTN_D512_COLUMN_SPLIT_LAB") != nullptr
            && imparo_sm80_fa_d512::configure<1, 2>();
        const uint64_t probability_floats =
            (uint64_t(columns) * kv_span + 1) / 2;
        const uint64_t d256_block_stride = uint64_t(columns) * kv_span
            + probability_floats
            + uint64_t(columns) * schedule_groups
            + uint64_t(2 * columns) * segment_slots;
        const uint64_t d512_block_stride = uint64_t(columns) * kv_span
            + probability_floats
            + uint64_t(2 * columns) * schedule_groups
            + uint64_t(4 * columns) * segment_slots;
        // Both fused kernels keep scores/probabilities inside the CTA. Their
        // only per-logical-block workspace is a possible Stream-K numerator
        // seam followed by max/sum metadata. Keep this contract shape-derived
        // so adding another fused head dimension cannot silently inherit the
        // much larger staged-attention workspace and fragment its launches.
        const uint64_t fused_block_stride =
            uint64_t(columns) * head_dim + 2 * columns;
        const uint64_t floats_per_block = fused_d256 || fused_d512
            ? fused_block_stride * (fused_d512_output_split_lab ? 2u : 1u)
            : (partitioned_values ? d512_block_stride : d256_block_stride);
        const uint64_t active_block_stride = fused_d512_output_split_lab
            ? fused_block_stride : floats_per_block;
        const uint64_t bytes_per_block = floats_per_block * sizeof(float);
        const uint64_t q_elements = uint64_t(n_tok) * n_heads * head_dim;
        const uint64_t q_cache_bytes = q_elements * sizeof(__half);
        const bool produced_q_hit = qk_scale == 1.0f
            && g.attention_q_src == q_buf
            && g.attention_q_epoch == g.buf_epoch[q_buf]
            && g.attention_q_head_dim == head_dim
            && g.attention_q_heads == n_heads
            && g.attention_q_tokens == n_tok;
        const uint64_t workspace_offset = produced_q_hit ? 0
            : (q_cache_bytes + 255u) & ~uint64_t(255u);
        const uint64_t workspace_cap = std::max<uint64_t>(
            bytes_per_block, attention_workspace_budget(head_dim));
        uint32_t chunk_blocks = uint32_t(std::min<uint64_t>(
            logical_blocks, workspace_cap / bytes_per_block));
        chunk_blocks = std::max(1u, chunk_blocks);
        while (!ensure_attention_scratch(
                   workspace_offset + uint64_t(chunk_blocks) * bytes_per_block)
               && chunk_blocks > 1) {
            chunk_blocks = (chunk_blocks + 1) / 2;
        }
        if (g.attention_scratch &&
            g.attention_scratch_bytes >= workspace_offset
                + uint64_t(chunk_blocks) * bytes_per_block) {
            if (head_dim == 256) {
                trace_d256_attention_route(
                    fused_d256
                        ? (d256_f32_fa_lab ? "fused-f32-lab" : "fused")
                        : "staged", kv_layer, start_pos,
                    n_tok, logical_blocks, physical_blocks,
                    d256_segment_bound);
            }
            __half * q_cache = produced_q_hit
                ? static_cast<__half *>(g.attention_q_cache)
                : static_cast<__half *>(g.attention_scratch);
            float * workspace = reinterpret_cast<float *>(
                static_cast<uint8_t *>(g.attention_scratch) + workspace_offset);
            const uint64_t q_pairs = q_elements / 2;
            if (!produced_q_hit) {
                AttentionStageEventScope stage("q_cache", head_dim, 1);
                imparo_sm80_prefill::cache_scaled_q
                    <<<(q_pairs + 255) / 256, 256, 0, g.stream>>>(
                        static_cast<const float *>(g.bufs[q_buf]),
                        reinterpret_cast<__half2 *>(q_cache), q_pairs, qk_scale);
            }
            const uint32_t score_threads = 2 * column_tiles * 32;
            uint32_t tuned_score_key_groups = tuner_knob(15);
            if (!tuned_score_key_groups) {
                if (const char * forced = std::getenv(
                        "IMPARO_CUDA_ATTN_SCORE_KEY_GROUPS")) {
                    tuned_score_key_groups = uint32_t(
                        std::strtoul(forced, nullptr, 10));
                }
            }
            const uint32_t score_key_groups =
                imparo_sm80_prefill::select_score_key_groups(
                    uint32_t(g.sm_version), tuned_score_key_groups);
            for (uint32_t base = 0; base < logical_blocks; base += chunk_blocks) {
                const uint32_t count = std::min(chunk_blocks, logical_blocks - base);
                if (fused_d512) {
                    AttentionStageEventScope stage("flash-d512", head_dim, count);
                    if (fused_d512_output_split_lab) {
                        imparo_sm80_fa_d512::flash<2, 1>
                            <<<count * 2, 256, imparo_sm80_fa_d512::kSharedBytes,
                                g.stream>>>(
                                q_cache, static_cast<const __half *>(kc),
                                static_cast<const __half *>(vc), workspace,
                                static_cast<float *>(g.bufs[out_buf]), base,
                                query_tiles, n_heads, n_kv, kv_width, start_pos,
                                window, n_tok, ring, valid_span, schedule_groups,
                                physical_blocks, segment_slots, query_tokens,
                                canonical_parts, virtual_stream_blocks,
                                active_block_stride, page_table);
                    } else if (fused_d512_column_split_lab) {
                        imparo_sm80_fa_d512::flash<1, 2>
                            <<<count * 2, 128,
                                imparo_sm80_fa_d512::kSharedBytes,
                                g.stream>>>(
                                q_cache, static_cast<const __half *>(kc),
                                static_cast<const __half *>(vc), workspace,
                                static_cast<float *>(g.bufs[out_buf]), base,
                                query_tiles, n_heads, n_kv, kv_width, start_pos,
                                window, n_tok, ring, valid_span, schedule_groups,
                                physical_blocks, segment_slots, query_tokens,
                                canonical_parts, virtual_stream_blocks,
                                active_block_stride, page_table);
                    } else {
                        imparo_sm80_fa_d512::flash<1, 1>
                            <<<count, 256, imparo_sm80_fa_d512::kSharedBytes,
                                g.stream>>>(
                                q_cache, static_cast<const __half *>(kc),
                                static_cast<const __half *>(vc), workspace,
                                static_cast<float *>(g.bufs[out_buf]), base,
                                query_tiles, n_heads, n_kv, kv_width, start_pos,
                                window, n_tok, ring, valid_span, schedule_groups,
                                physical_blocks, segment_slots, query_tokens,
                                canonical_parts, virtual_stream_blocks,
                                active_block_stride, page_table);
                    }
                    continue;
                }
                if (fused_d256) {
                    AttentionStageEventScope stage("flash-d256", head_dim, count);
#define IMPARO_LAUNCH_D256_FA(RINGED, F32_NUMERATOR)                      \
                    imparo_sm80_fa_d256::flash<RINGED, F32_NUMERATOR>    \
                        <<<count, 128, 0, g.stream>>>(                    \
                            q_cache, static_cast<const __half *>(kc),     \
                            static_cast<const __half *>(vc), workspace,   \
                            static_cast<float *>(g.bufs[out_buf]), base,  \
                            query_tiles, n_heads, n_kv, kv_width,         \
                            start_pos, window, n_tok, ring, valid_span,   \
                            schedule_groups, physical_blocks,            \
                            virtual_stream_blocks, segment_slots,         \
                            query_tokens, active_block_stride, page_table)
                    if (ring) {
                        if (d256_f32_fa_lab) {
                            IMPARO_LAUNCH_D256_FA(true, true);
                        } else {
                            IMPARO_LAUNCH_D256_FA(true, false);
                        }
                    } else {
                        if (d256_f32_fa_lab) {
                            IMPARO_LAUNCH_D256_FA(false, true);
                        } else {
                            IMPARO_LAUNCH_D256_FA(false, false);
                        }
                    }
#undef IMPARO_LAUNCH_D256_FA
                    continue;
                }
                {
                    AttentionStageEventScope stage("scores", head_dim, count);
                    if (head_dim == 256) {
                        if (score_key_groups == 4) {
                            imparo_sm80_prefill::scores<256, 4>
                                <<<dim3(count, (key_groups + 3) / 4), score_threads,
                                    0, g.stream>>>(
                                q_cache,
                                static_cast<const __half *>(kc), workspace, base,
                                query_tiles, n_heads, n_kv, kv_width, start_pos,
                                window, n_tok, ring, valid_span, kv_span,
                                query_tokens, columns, active_block_stride, page_table);
                        } else if (score_key_groups == 2) {
                            imparo_sm80_prefill::scores<256, 2>
                                <<<dim3(count, (key_groups + 1) / 2), score_threads,
                                    0, g.stream>>>(
                                q_cache,
                                static_cast<const __half *>(kc), workspace, base,
                                query_tiles, n_heads, n_kv, kv_width, start_pos,
                                window, n_tok, ring, valid_span, kv_span,
                                query_tokens, columns, active_block_stride, page_table);
                        } else {
                            imparo_sm80_prefill::scores<256, 1>
                                <<<dim3(count, key_groups), score_threads,
                                    0, g.stream>>>(
                                q_cache,
                                static_cast<const __half *>(kc), workspace, base,
                                query_tiles, n_heads, n_kv, kv_width, start_pos,
                                window, n_tok, ring, valid_span, kv_span,
                                query_tokens, columns, active_block_stride, page_table);
                        }
                    } else {
                        if (score_key_groups == 4) {
                            if (page_table) {
                                imparo_sm80_prefill::scores_d512_paged<4>
                                    <<<dim3(count, (key_groups + 3) / 4), score_threads,
                                        0, g.stream>>>(
                                    q_cache, static_cast<const __half *>(kc),
                                    workspace, base, query_tiles, n_heads, n_kv,
                                    kv_width, start_pos, window, n_tok, ring,
                                    valid_span, kv_span, query_tokens, columns,
                                    active_block_stride, page_table);
                            } else {
                                imparo_sm80_prefill::scores_d512<4>
                                    <<<dim3(count, (key_groups + 3) / 4), score_threads,
                                        0, g.stream>>>(
                                    q_cache, static_cast<const __half *>(kc),
                                    workspace, base, query_tiles, n_heads, n_kv,
                                    kv_width, start_pos, window, n_tok, ring,
                                    valid_span, kv_span, query_tokens, columns,
                                    active_block_stride);
                            }
                        } else if (score_key_groups == 2) {
                            if (page_table) {
                                imparo_sm80_prefill::scores_d512_paged<2>
                                    <<<dim3(count, (key_groups + 1) / 2), score_threads,
                                        0, g.stream>>>(
                                    q_cache, static_cast<const __half *>(kc),
                                    workspace, base, query_tiles, n_heads, n_kv,
                                    kv_width, start_pos, window, n_tok, ring,
                                    valid_span, kv_span, query_tokens, columns,
                                    active_block_stride, page_table);
                            } else {
                                imparo_sm80_prefill::scores_d512<2>
                                    <<<dim3(count, (key_groups + 1) / 2), score_threads,
                                        0, g.stream>>>(
                                    q_cache, static_cast<const __half *>(kc),
                                    workspace, base, query_tiles, n_heads, n_kv,
                                    kv_width, start_pos, window, n_tok, ring,
                                    valid_span, kv_span, query_tokens, columns,
                                    active_block_stride);
                            }
                        } else {
                            if (page_table) {
                                imparo_sm80_prefill::scores_d512_paged<1>
                                    <<<dim3(count, key_groups), score_threads,
                                        0, g.stream>>>(
                                    q_cache, static_cast<const __half *>(kc),
                                    workspace, base, query_tiles, n_heads, n_kv,
                                    kv_width, start_pos, window, n_tok, ring,
                                    valid_span, kv_span, query_tokens, columns,
                                    active_block_stride, page_table);
                            } else {
                                imparo_sm80_prefill::scores_d512<1>
                                    <<<dim3(count, key_groups), score_threads,
                                        0, g.stream>>>(
                                    q_cache, static_cast<const __half *>(kc),
                                    workspace, base, query_tiles, n_heads, n_kv,
                                    kv_width, start_pos, window, n_tok, ring,
                                    valid_span, kv_span, query_tokens, columns,
                                    active_block_stride);
                            }
                        }
                    }
                }
                {
                    AttentionStageEventScope stage("softmax", head_dim, count);
                    if (!partitioned_values) {
                        imparo_sm80_prefill::softmax_d256_stream
                            <<<dim3(count, (columns + 7) / 8, segment_slots),
                                32, 0, g.stream>>>(
                                workspace, kv_span, d256_block_stride, base,
                                query_tiles, start_pos, ring, logical_blocks,
                                schedule_groups, physical_blocks, segment_slots,
                                query_tokens, columns, virtual_stream_blocks,
                                d256_staged_f32_values_lab ? 1u : 0u);
                    } else {
                        imparo_sm80_prefill::softmax_d512_stream
                            <<<dim3(count, 2 * ((columns + 7) / 8), segment_slots),
                                32, 0, g.stream>>>(
                                workspace, kv_span, d512_block_stride, base,
                                query_tiles, start_pos, ring, logical_blocks,
                                schedule_groups, physical_blocks, segment_slots,
                                query_tokens, columns, canonical_parts,
                                virtual_stream_blocks);
                    }
                }
                {
                    AttentionStageEventScope stage("values", head_dim, count);
                    if (!partitioned_values) {
                        if (d256_staged_f32_values_lab) {
                            imparo_sm80_prefill::values_d256_stream_f32
                                <<<dim3(count, columns), 256, 0, g.stream>>>(
                                    static_cast<const __half *>(vc), workspace,
                                    static_cast<float *>(g.bufs[out_buf]), base,
                                    query_tiles, n_heads, n_kv, kv_width, n_tok,
                                    ring, valid_span, kv_span, d256_block_stride,
                                    logical_blocks, schedule_groups, physical_blocks,
                                    segment_slots, start_pos, query_tokens, columns,
                                    virtual_stream_blocks, page_table);
                        } else {
                            imparo_sm80_prefill::values_d256_stream
                                <<<dim3(count, head_dim / 32), column_tiles * 32, 0,
                                    g.stream>>>(
                                    static_cast<const __half *>(vc), workspace,
                                    static_cast<float *>(g.bufs[out_buf]), base,
                                    query_tiles, n_heads, n_kv, kv_width, n_tok, ring,
                                    valid_span, kv_span, d256_block_stride,
                                    logical_blocks, schedule_groups, physical_blocks,
                                    segment_slots, start_pos, query_tokens, columns,
                                    virtual_stream_blocks, page_table);
                        }
                    } else if (head_dim == 256) {
                        imparo_sm80_prefill::values_partitioned_stream<256>
                            <<<dim3(count, head_dim / 64), score_threads, 0, g.stream>>>(
                                static_cast<const __half *>(vc), workspace,
                                static_cast<float *>(g.bufs[out_buf]), base,
                                query_tiles, n_heads, n_kv, kv_width, n_tok, ring,
                                valid_span, kv_span, d512_block_stride,
                                logical_blocks, schedule_groups, physical_blocks,
                                segment_slots, start_pos, query_tokens, columns, 0u, 0u,
                                page_table);
                    } else {
                        if (page_table) {
                            imparo_sm80_prefill::values_partitioned_stream_d512_paged
                                <<<dim3(count, head_dim / 64), score_threads,
                                    0, g.stream>>>(
                                    static_cast<const __half *>(vc), workspace,
                                    static_cast<float *>(g.bufs[out_buf]), base,
                                    query_tiles, n_heads, n_kv, kv_width, n_tok,
                                    ring, valid_span, kv_span, d512_block_stride,
                                    logical_blocks, schedule_groups, physical_blocks,
                                    segment_slots, start_pos, query_tokens, columns,
                                    canonical_parts, virtual_stream_blocks, page_table);
                        } else {
                            imparo_sm80_prefill::values_partitioned_stream_d512
                                <<<dim3(count, head_dim / 64), score_threads,
                                    0, g.stream>>>(
                                    static_cast<const __half *>(vc), workspace,
                                    static_cast<float *>(g.bufs[out_buf]), base,
                                    query_tiles, n_heads, n_kv, kv_width, n_tok,
                                    ring, valid_span, kv_span, d512_block_stride,
                                    logical_blocks, schedule_groups, physical_blocks,
                                    segment_slots, start_pos, query_tokens, columns,
                                    canonical_parts, virtual_stream_blocks);
                        }
                    }
                }
            }
            mark_buf_written(out_buf);
            return;
        }
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "[imparo] cuda: D512 tiled workspace unavailable; using common attention\n");
            warned = true;
        }
    }
    if ((exact128_d256_batch32
            || std::getenv("IMPARO_CUDA_ATTN_BATCH32")) &&
        head_dim == 256 &&
        ktype == 1 && vtype == 1) {
        // Receipt-bound exact128 never inherits ambient laboratory arithmetic.
        // The fixed gate accepted the f32 numerator without the extra reduction.
        const uint32_t half_v_accum = exact128_d256_batch32 ? 0u
            : (std::getenv("IMPARO_CUDA_ATTN_BATCH32_HALF") ? 1u : 0u);
        const uint32_t llama_reduce = exact128_d256_batch32 ? 0u
            : (std::getenv("IMPARO_CUDA_ATTN_LLAMA_REDUCE") ? 1u : 0u);
        trace_d256_attention_route(
            exact128_d256_batch32 ? "batch32-exact128" : "batch32",
            kv_layer, start_pos, n_tok, 0, 0, 0);
#define IMPARO_LAUNCH_BATCH32(EXACT_F32, HALF_V, LLAMA_REDUCE)              \
        k_attention_batch32_f16<EXACT_F32>                                \
            <<<grid, EXACT_F32 ? 128u : threads, head_dim * 4, g.stream>>>(\
                static_cast<const float *>(g.bufs[q_buf]),                 \
                static_cast<const __half *>(kc),                          \
                static_cast<const __half *>(vc),                          \
                static_cast<float *>(g.bufs[out_buf]), head_dim, n_heads,  \
                n_kv, kv_width, start_pos, qk_scale, window, ring, n_tok,  \
                HALF_V, LLAMA_REDUCE, page_table)
        if (exact128_d256_batch32) {
            // The receipted route is a distinct compile-time specialization.
            // It must not carry the dormant half-V accumulator or alternative
            // reduction's stack/shared-memory footprint into short prefill.
            IMPARO_LAUNCH_BATCH32(true, 0u, 0u);
        } else {
            IMPARO_LAUNCH_BATCH32(false, half_v_accum, llama_reduce);
        }
        if (g.tuner_lab && exact128_d256_batch32) {
            g.tune_exact128_route_hits |= EXACT128_ATTN;
            exact128_record_commit(g.tune_exact128_attn_commits);
        }
#undef IMPARO_LAUNCH_BATCH32
        if (llama_reduce && half_v_accum) {
            k_attention_streamk_fixup_sm80_f16<<<grid, 128, 0, g.stream>>>(
                static_cast<const float *>(g.bufs[q_buf]),
                static_cast<const __half *>(kc), static_cast<const __half *>(vc),
                static_cast<float *>(g.bufs[out_buf]), head_dim, n_heads, n_kv,
                kv_width, start_pos, qk_scale, window, ring, n_tok, uint32_t(g.sm_count),
                page_table);
        }
        mark_buf_written(out_buf);
        return;
    }
    const bool d64_f32 = head_dim == 64
        && std::getenv("IMPARO_CUDA_ATTN_D64_HALF") == nullptr;
    const uint32_t f32_v_accum =
        (d64_f32 || std::getenv("IMPARO_CUDA_ATTN_F32")) ? 1u : 0u;
    if (head_dim == 256) {
        trace_d256_attention_route(
            "fallback", kv_layer, start_pos, n_tok, 0, 0, 0);
    }
    k_attention<<<grid, threads, head_dim * 4, g.stream>>>(
        static_cast<const float *>(g.bufs[q_buf]), kc, vc,
        static_cast<float *>(g.bufs[out_buf]), head_dim, n_heads, n_kv, kv_width,
        start_pos, qk_scale, window, ring, n_tok, ktype, vtype,
        f32_v_accum, page_table);
    mark_buf_written(out_buf);
}

static bool buffer_slice(uint32_t id, uint64_t byte_offset, uint64_t bytes,
                         uint8_t ** out) {
    if (!out || id >= B_COUNT || !g.bufs[id] || byte_offset > g.sizes[id]
        || bytes > g.sizes[id] - byte_offset) return false;
    *out = static_cast<uint8_t *>(g.bufs[id]) + byte_offset;
    return true;
}

static bool byte_ranges_overlap(const void * first, uint64_t first_bytes,
                                const void * second, uint64_t second_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(first);
    const uintptr_t b = reinterpret_cast<uintptr_t>(second);
    if (first_bytes > UINTPTR_MAX - a || second_bytes > UINTPTR_MAX - b) return true;
    return a < b + second_bytes && b < a + first_bytes;
}

static bool valid_launch_count(uint64_t count) {
    return count && count <= uint64_t(UINT32_MAX) * 256;
}

extern "C" void imparo_cuda_shortconv(
        uint32_t bcx, uint64_t w_off, uint32_t state, uint32_t state_off,
        uint32_t out, uint32_t width, uint32_t kernel, uint32_t n_tok) {
    imparo_cuda_lfm2::ShortconvLayout layout;
    if (!imparo_cuda_lfm2::checked_shortconv_layout(
            width, kernel, n_tok, &layout)) {
        set_pending(CUDA_RC_INVALID, "shortconv shape");
        return;
    }
    const uint64_t state_byte_off = uint64_t(state_off) * sizeof(float);
    uint8_t * bcx_bytes = nullptr;
    uint8_t * state_bytes = nullptr;
    uint8_t * out_bytes = nullptr;
    if (!buffer_slice(bcx, 0, layout.bcx_bytes, &bcx_bytes)
        || !buffer_slice(state, state_byte_off, layout.state_bytes, &state_bytes)
        || !buffer_slice(out, 0, layout.output_bytes, &out_bytes)
        || !valid_launch_count(layout.output_bytes / sizeof(float))
        || !valid_launch_count(width)) {
        set_pending(CUDA_RC_INVALID, "shortconv buffer range");
        return;
    }
    if (byte_ranges_overlap(bcx_bytes, layout.bcx_bytes, state_bytes, layout.state_bytes)
        || byte_ranges_overlap(bcx_bytes, layout.bcx_bytes, out_bytes, layout.output_bytes)
        || byte_ranges_overlap(state_bytes, layout.state_bytes, out_bytes,
                               layout.output_bytes)) {
        set_pending(CUDA_RC_INVALID, "shortconv alias");
        return;
    }
    const uint8_t * raw_weights = nullptr;
    const int weight_rc = weight_slice(w_off, layout.weight_bytes, &raw_weights);
    if (weight_rc) {
        set_pending(weight_rc, "shortconv weights");
        return;
    }
    OpEventScope profile("shortconv", width, n_tok);
    const uint32_t fused_selector = tuner_knob(51);
    mark_tuner_dispatch(4, (uint64_t(kernel) << 32) | width);
    if (fused_selector != 0 && g.sm_version == 86 && n_tok == 1) {
        using Fused = imparo_sm86_shortconv_decode_fused::LaunchResult;
        const Fused fused = imparo_sm86_shortconv_decode_fused::launch(
            reinterpret_cast<const float *>(bcx_bytes),
            reinterpret_cast<const float *>(raw_weights),
            reinterpret_cast<float *>(state_bytes),
            reinterpret_cast<float *>(out_bytes),
            width, kernel, g.sm_version, g.stream);
        if (fused == Fused::Error) {
            set_pending(CUDA_RC_ERROR, "shortconv fused launch");
            return;
        }
        if (fused == Fused::Launched) {
            mark_buf_written(out);
            mark_buf_written(state);
            return;
        }
    }
    if (fused_selector != 0 && g.tuner_mode) {
        set_pending(CUDA_RC_INVALID, "shortconv fused candidate did not dispatch");
        return;
    }
    const uint64_t output_count = layout.output_bytes / sizeof(float);
    imparo_cuda_lfm2::shortconv_output_kernel<<<
        uint32_t((output_count + 255) / 256), 256, 0, g.stream>>>(
            reinterpret_cast<const float *>(bcx_bytes),
            reinterpret_cast<const float *>(raw_weights),
            reinterpret_cast<const float *>(state_bytes),
            reinterpret_cast<float *>(out_bytes), width, kernel, n_tok);
    if (cudaPeekAtLastError() != cudaSuccess) {
        set_pending(CUDA_RC_ERROR, "shortconv output launch");
        return;
    }
    // Stream ordering is the cross-grid barrier: every output reads the old state
    // before the second dispatch advances it in place.
    imparo_cuda_lfm2::shortconv_state_kernel<<<
        (width + 255) / 256, 256, 0, g.stream>>>(
            reinterpret_cast<const float *>(bcx_bytes),
            reinterpret_cast<const float *>(state_bytes),
            reinterpret_cast<float *>(state_bytes), width, kernel, n_tok);
    mark_buf_written(out);
    mark_buf_written(state);
}

extern "C" void imparo_cuda_shortconv_snapshot(
        uint32_t bcx, uint32_t state, uint32_t state_off,
        uint32_t snap, uint32_t snap_off, uint32_t width,
        uint32_t kernel, uint32_t n_tok) {
    imparo_cuda_lfm2::ShortconvLayout layout;
    if (!imparo_cuda_lfm2::checked_shortconv_layout(
            width, kernel, n_tok, &layout)) {
        set_pending(CUDA_RC_INVALID, "shortconv snapshot shape");
        return;
    }
    const uint64_t state_byte_off = uint64_t(state_off) * sizeof(float);
    const uint64_t snap_byte_off = uint64_t(snap_off) * sizeof(float);
    uint8_t * bcx_bytes = nullptr;
    uint8_t * state_bytes = nullptr;
    uint8_t * snap_bytes = nullptr;
    if (!buffer_slice(bcx, 0, layout.bcx_bytes, &bcx_bytes)
        || !buffer_slice(state, state_byte_off, layout.state_bytes, &state_bytes)
        || !buffer_slice(snap, snap_byte_off, layout.state_bytes, &snap_bytes)
        || !valid_launch_count(width)) {
        set_pending(CUDA_RC_INVALID, "shortconv snapshot buffer range");
        return;
    }
    if (byte_ranges_overlap(bcx_bytes, layout.bcx_bytes, state_bytes, layout.state_bytes)
        || byte_ranges_overlap(bcx_bytes, layout.bcx_bytes, snap_bytes, layout.state_bytes)
        || byte_ranges_overlap(state_bytes, layout.state_bytes, snap_bytes,
                               layout.state_bytes)) {
        set_pending(CUDA_RC_INVALID, "shortconv snapshot alias");
        return;
    }
    imparo_cuda_lfm2::shortconv_state_kernel<<<
        (width + 255) / 256, 256, 0, g.stream>>>(
            reinterpret_cast<const float *>(bcx_bytes),
            reinterpret_cast<const float *>(state_bytes),
            reinterpret_cast<float *>(snap_bytes), width, kernel, n_tok);
    mark_buf_written(snap);
}

extern "C" void imparo_cuda_silu(uint32_t a, uint32_t n) {
    uint8_t * values = nullptr;
    const uint64_t bytes = uint64_t(n) * sizeof(float);
    if (!n || !buffer_slice(a, 0, bytes, &values)) {
        set_pending(CUDA_RC_INVALID, "silu arguments");
        return;
    }
    OpEventScope profile("silu", n, 0);
    imparo_cuda_lfm2::silu_kernel<<<(n + 255) / 256, 256, 0, g.stream>>>(
        reinterpret_cast<float *>(values), n);
    mark_buf_written(a);
}

extern "C" void imparo_cuda_silu_mul(uint32_t a, uint32_t b, uint32_t n) {
    uint8_t * values = nullptr;
    uint8_t * factors = nullptr;
    const uint64_t bytes = uint64_t(n) * sizeof(float);
    if (!n || !buffer_slice(a, 0, bytes, &values)
        || !buffer_slice(b, 0, bytes, &factors)) {
        set_pending(CUDA_RC_INVALID, "silu_mul arguments");
        return;
    }
    OpEventScope profile("silu_mul", n, 0);
    imparo_cuda_lfm2::silu_mul_kernel<<<(n + 255) / 256, 256, 0, g.stream>>>(
        reinterpret_cast<float *>(values), reinterpret_cast<const float *>(factors), n);
    mark_buf_written(a);
}

extern "C" void imparo_cuda_gelu(uint32_t a, uint32_t n) {
    OpEventScope profile("gelu", n, 0);
    k_gelu<<<(n + 255) / 256, 256, 0, g.stream>>>(static_cast<float *>(g.bufs[a]), n);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_gelu_mul(uint32_t a, uint32_t b, uint32_t n) {
    OpEventScope profile("gelu_mul", n, 0);
    k_gelu_mul<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]), n);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_add(uint32_t a, uint32_t b, uint32_t n) {
    OpEventScope profile("add", n, 0);
    k_add<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]), n);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_add_scale(uint32_t a, uint32_t b, float k, uint32_t n) {
    OpEventScope profile("add_scale", n, 0);
    k_add_scale<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]), k, n);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_scale(uint32_t a, float k, uint32_t n) {
    OpEventScope profile("scale", n, 0);
    k_scale<<<(n + 255) / 256, 256, 0, g.stream>>>(static_cast<float *>(g.bufs[a]), k, n);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_copy(uint32_t dst, uint32_t src, uint32_t n) {
    OpEventScope profile("copy", n, 0);
    k_copy<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[dst]), static_cast<const float *>(g.bufs[src]), n);
    mark_buf_written(dst);
}
// n floats from src[src_off..] to dst[dst_off..] (element offsets); the caller keeps the
// two ranges disjoint when dst == src.
extern "C" void imparo_cuda_copy_range(uint32_t dst, uint32_t dst_off, uint32_t src,
                                       uint32_t src_off, uint32_t n) {
    OpEventScope profile("copy_range", n, 0);
    k_copy<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[dst]) + dst_off,
        static_cast<const float *>(g.bufs[src]) + src_off, n);
    mark_buf_written(dst);
}
extern "C" void imparo_cuda_mul_strided(uint32_t a, uint32_t b, uint32_t n,
                                        uint32_t b_off, uint32_t b_stride,
                                        uint32_t a_stride, uint32_t n_tok) {
    OpEventScope profile("mul_strided", n, n_tok);
    dim3 grid((n + 255) / 256, n_tok);
    k_mul_strided<<<grid, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]),
        n, b_off, b_stride, a_stride, n_tok);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_softcap(uint32_t a, float cap, uint32_t n) {
    OpEventScope profile("softcap", n, 0);
    k_softcap<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), cap, n);
    mark_buf_written(a);
}
extern "C" void imparo_cuda_argmax(uint32_t src, uint32_t dst, uint32_t n) {
    OpEventScope profile("argmax", n, 0);
    if (n >= 65536 && std::getenv("IMPARO_CUDA_NO_WIDE_ARGMAX") == nullptr) {
        k_argmax_rows<1024><<<1, 1024, 0, g.stream>>>(
            static_cast<const float *>(g.bufs[src]),
            static_cast<uint32_t *>(g.bufs[dst]), n, 1);
    } else {
        k_argmax_rows<256><<<1, 256, 0, g.stream>>>(
            static_cast<const float *>(g.bufs[src]),
            static_cast<uint32_t *>(g.bufs[dst]), n, 1);
    }
    mark_buf_written(dst);
}
extern "C" void imparo_cuda_argmax_rows(uint32_t src, uint32_t dst,
                                         uint32_t row_width, uint32_t rows) {
    OpEventScope profile("argmax_rows", row_width, rows);
    if (row_width >= 65536
        && std::getenv("IMPARO_CUDA_NO_WIDE_ARGMAX") == nullptr) {
        k_argmax_rows<1024><<<rows, 1024, 0, g.stream>>>(
            static_cast<const float *>(g.bufs[src]),
            static_cast<uint32_t *>(g.bufs[dst]), row_width, rows);
    } else {
        k_argmax_rows<256><<<rows, 256, 0, g.stream>>>(
            static_cast<const float *>(g.bufs[src]),
            static_cast<uint32_t *>(g.bufs[dst]), row_width, rows);
    }
    mark_buf_written(dst);
}
extern "C" void imparo_cuda_row(uint32_t wkind, uint64_t w_off, uint32_t width,
                                 uint32_t index, float scale, uint32_t dst,
                                 uint32_t dst_off) {
    OpEventScope profile("row", width, 0);
    if ((wkind != 1 && wkind != 2) || dst >= B_COUNT || !g.bufs[dst]
        || !width || width % 32 || uint64_t(dst_off) + width > g.sizes[dst] / 4) {
        set_pending(CUDA_RC_INVALID, "embedding row arguments");
        return;
    }
    const uint64_t row_bytes = uint64_t(width / 32) * (wkind == 1 ? 18 : 34);
    if (uint64_t(index) > (UINT64_MAX - w_off) / row_bytes) {
        set_pending(CUDA_RC_INVALID, "embedding row offset overflow");
        return;
    }
    const uint8_t * row = nullptr;
    const uint64_t row_off = w_off + uint64_t(index) * row_bytes;
    int rc = 0;
    const bool pinned_decode = g.forward_decode && decode_pinned_input_enabled()
        && (decode_graph_candidate() || !resident_weight_range(row_off, row_bytes));
    if (pinned_decode) {
        if (row_off > g.weights_len || row_bytes > g.weights_len - row_off) {
            set_pending(CUDA_RC_INVALID, "embedding row range");
            return;
        }
        rc = ensure_weight_cache(row_bytes);
        if (!rc) rc = ensure_decode_row_host_stage(row_bytes);
        if (!rc) {
            std::memcpy(g.decode_row_stage_host, g.weights_host + row_off,
                        size_t(row_bytes));
            const cudaError_t err = cudaMemcpyAsync(
                g.weight_cache, g.decode_row_stage_host, size_t(row_bytes),
                cudaMemcpyHostToDevice, g.stream);
            if (err != cudaSuccess) rc = CUDA_RC_ERROR;
            row = static_cast<const uint8_t *>(g.weight_cache);
            if (decode_graph_candidate()) {
                g.decode_row_desc_valid = true;
                g.decode_row_base = w_off;
                g.decode_row_bytes = row_bytes;
            }
        }
    } else {
        rc = weight_slice(row_off, row_bytes, &row);
    }
    if (rc) { set_pending(rc, "embedding row upload"); return; }
    if (wkind == 1) {
        k_row_q4<<<(width + 255) / 256, 256, 0, g.stream>>>(
            row, static_cast<float *>(g.bufs[dst]), width, scale, dst_off);
    } else {
        k_row_q8_0<<<(width + 255) / 256, 256, 0, g.stream>>>(
            row, static_cast<float *>(g.bufs[dst]), width, scale, dst_off);
    }
    mark_buf_written(dst);
}

extern "C" void imparo_cuda_rows(uint32_t wkind, uint64_t w_off, uint32_t width,
                                  uint32_t table_rows, uint32_t tokens_buf,
                                  float scale, uint32_t dst, uint32_t n_tok) {
    OpEventScope profile("rows", width, n_tok);
    if ((wkind != 1 && wkind != 2) || tokens_buf >= B_COUNT || dst >= B_COUNT
        || !g.bufs[tokens_buf] || !g.bufs[dst] || !width || width % 32
        || !table_rows || !n_tok || uint64_t(n_tok) > g.sizes[tokens_buf] / 4
        || uint64_t(n_tok) * width > g.sizes[dst] / 4) {
        set_pending(CUDA_RC_INVALID, "embedding rows arguments");
        return;
    }
    const uint64_t row_bytes = uint64_t(width / 32) * (wkind == 1 ? 18 : 34);
    if (uint64_t(table_rows) > UINT64_MAX / row_bytes) {
        set_pending(CUDA_RC_INVALID, "embedding table range overflow");
        return;
    }
    std::vector<uint32_t> fallback_tokens;
    const uint32_t * tokens = host_u32_values(tokens_buf, n_tok, fallback_tokens);
    if (!tokens) {
        set_pending(CUDA_RC_ERROR, "embedding token readback");
        return;
    }
    for (uint32_t t = 0; t < n_tok; ++t) {
        if (tokens[t] >= table_rows) {
            set_pending(CUDA_RC_INVALID, "embedding token outside table");
            return;
        }
    }
    const uint64_t table_bytes = uint64_t(table_rows) * row_bytes;
    const uint8_t * table = resident_weight_range(w_off, table_bytes);
    if (table) {
        // The resident embedding kernel reads the token buffer directly; graph
        // replay therefore needs only a fresh pinned token id, not a copied row.
        if (g.forward_decode && n_tok == 1 && decode_graph_candidate()) {
            g.decode_row_desc_valid = true;
            g.decode_row_base = w_off;
            g.decode_row_bytes = 0;
        }
        dim3 grid((width + 255) / 256, n_tok);
        if (wkind == 1) {
            k_rows_q4<<<grid, 256, 0, g.stream>>>(
                table, static_cast<const uint32_t *>(g.bufs[tokens_buf]),
                static_cast<float *>(g.bufs[dst]), width, scale, n_tok);
        } else {
            k_rows_q8_0<<<grid, 256, 0, g.stream>>>(
                table, static_cast<const uint32_t *>(g.bufs[tokens_buf]),
                static_cast<float *>(g.bufs[dst]), width, scale, n_tok);
        }
        mark_buf_written(dst);
        return;
    }
    if (uint64_t(n_tok) > UINT64_MAX / row_bytes) {
        set_pending(CUDA_RC_INVALID, "embedding staged row size");
        return;
    }
    const uint64_t staged_bytes = uint64_t(n_tok) * row_bytes;
    int rc = ensure_weight_cache(staged_bytes);
    if (!rc) rc = ensure_ple_host_stage(staged_bytes);
    if (rc) {
        set_pending(rc, "embedding row staging");
        return;
    }
    auto * staged = static_cast<uint8_t *>(g.ple_stage_host);
    for (uint32_t t = 0; t < n_tok; ++t) {
        if (uint64_t(tokens[t]) > (UINT64_MAX - w_off) / row_bytes) {
            set_pending(CUDA_RC_INVALID, "embedding row offset overflow");
            return;
        }
        const uint64_t row_off = w_off + uint64_t(tokens[t]) * row_bytes;
        if (row_off > g.weights_len || row_bytes > g.weights_len - row_off) {
            set_pending(CUDA_RC_INVALID, "embedding row range");
            return;
        }
        std::memcpy(staged + uint64_t(t) * row_bytes,
                    g.weights_host + row_off, size_t(row_bytes));
    }
    const cudaError_t copy = cudaMemcpyAsync(
        g.weight_cache, staged, size_t(staged_bytes),
        cudaMemcpyHostToDevice, g.stream);
    if (copy != cudaSuccess) {
        set_pending(copy == cudaErrorMemoryAllocation ? CUDA_RC_OOM : CUDA_RC_ERROR,
                    "embedding row upload");
        return;
    }
    dim3 grid((width + 255) / 256, n_tok);
    if (wkind == 1) {
        k_rows_q4_packed<<<grid, 256, 0, g.stream>>>(
            static_cast<const uint8_t *>(g.weight_cache),
            static_cast<float *>(g.bufs[dst]), width, scale, n_tok);
    } else {
        k_rows_q8_0_packed<<<grid, 256, 0, g.stream>>>(
            static_cast<const uint8_t *>(g.weight_cache),
            static_cast<float *>(g.bufs[dst]), width, scale, n_tok);
    }
    mark_buf_written(dst);
}

extern "C" void imparo_cuda_ple_gather_combine(
    uint32_t proj, uint32_t tokens_buf, uint64_t w_offset, uint32_t width,
    float emb_scale, float comb_scale, uint32_t n_tok) {
    OpEventScope profile("ple_gather", width, n_tok);
    if (proj >= B_COUNT || tokens_buf >= B_COUNT || !g.bufs[proj]
        || !g.bufs[tokens_buf] || width % 32) {
        set_pending(CUDA_RC_INVALID, "PLE gather arguments");
        return;
    }
    const uint64_t row_bytes = uint64_t(width / 32) * 18;
    if (const uint8_t * table = resident_weight_at(w_offset)) {
        dim3 grid((width + 255) / 256, n_tok);
        k_ple_gather<<<grid, 256, 0, g.stream>>>(
            static_cast<float *>(g.bufs[proj]), table,
            static_cast<const uint32_t *>(g.bufs[tokens_buf]), width,
            emb_scale, n_tok);
        const uint32_t n = width * n_tok;
        k_scale<<<(n + 255) / 256, 256, 0, g.stream>>>(
            static_cast<float *>(g.bufs[proj]), comb_scale, n);
        mark_buf_written(proj);
        return;
    }
    std::vector<uint32_t> fallback_tokens;
    const uint32_t * tokens = host_u32_values(tokens_buf, n_tok, fallback_tokens);
    if (!tokens) {
        set_pending(CUDA_RC_ERROR, "PLE token readback");
        return;
    }
    if (uint64_t(n_tok) > UINT64_MAX / row_bytes) {
        set_pending(CUDA_RC_INVALID, "PLE staged row size");
        return;
    }
    const uint64_t staged_bytes = uint64_t(n_tok) * row_bytes;
    int rc = ensure_weight_cache(staged_bytes);
    if (rc) { set_pending(rc, "PLE row cache"); return; }
    rc = ensure_ple_host_stage(staged_bytes);
    if (rc) { set_pending(rc, "PLE pinned row staging"); return; }
    if (g.forward_decode && decode_graph_candidate() && n_tok == 1) {
        g.decode_ple_desc_valid = true;
        g.decode_ple_table = w_offset;
        g.decode_ple_row_bytes = row_bytes;
        g.decode_ple_norm = 0;
        g.decode_ple_norm_bytes = 0;
        g.decode_ple_prefix = 0;
    }
    auto * staged = static_cast<uint8_t *>(g.ple_stage_host);
    for (uint32_t t = 0; t < n_tok; ++t) {
        const uint64_t row_off = w_offset + uint64_t(tokens[t]) * row_bytes;
        if (row_off < w_offset || row_off > g.weights_len
            || row_bytes > g.weights_len - row_off) {
            set_pending(CUDA_RC_INVALID, "PLE row range");
            return;
        }
        std::memcpy(staged + size_t(uint64_t(t) * row_bytes),
                    g.weights_host + row_off, size_t(row_bytes));
    }
    const cudaError_t err = g.graph_capturing && g.forward_decode && n_tok == 1
        ? cudaSuccess
        : cudaMemcpyAsync(g.weight_cache, staged, size_t(staged_bytes),
                          cudaMemcpyHostToDevice, g.stream);
    const cudaError_t sync = !async_host_control_enabled() && err == cudaSuccess
        ? cudaStreamSynchronize(g.stream) : cudaSuccess;
    if (err != cudaSuccess || sync != cudaSuccess) {
        set_pending(CUDA_RC_ERROR, "PLE batched row upload");
        return;
    }
    dim3 grid((width + 255) / 256, n_tok);
    k_ple_gather_packed<<<grid, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[proj]),
        static_cast<const uint8_t *>(g.weight_cache), width, emb_scale, n_tok);
    const uint32_t n = width * n_tok;
    k_scale<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[proj]), comb_scale, n);
    mark_buf_written(proj);
}

extern "C" void imparo_cuda_ple_norm_gather_combine(
    uint32_t proj, uint32_t tokens_buf, uint64_t norm_w_offset,
    uint64_t table_offset, uint32_t ple_width, uint32_t n_layers,
    float input_scale, float eps, float emb_scale, float comb_scale,
    uint32_t n_tok) {
    OpEventScope profile("ple_norm_gather", ple_width, n_tok);
    if (proj >= B_COUNT || tokens_buf >= B_COUNT || !g.bufs[proj]
        || !g.bufs[tokens_buf] || !ple_width || !n_layers
        || ple_width % 32) {
        set_pending(CUDA_RC_INVALID, "PLE norm-gather arguments");
        return;
    }
    const uint64_t norm_bytes = uint64_t(ple_width) * sizeof(float);
    const uint8_t * norm_raw = resident_weight_at(norm_w_offset);
    constexpr uint32_t threads = 256;
    constexpr uint32_t shared_bytes = 32 * sizeof(float);
    const uint32_t width = ple_width * n_layers;
    const uint64_t table_row_bytes = uint64_t(width / 32) * 18;
    if (const uint8_t * table = resident_weight_at(table_offset)) {
        if (g.forward_decode && decode_graph_candidate()) {
            g.decode_graph_shape_blocked = true;
        }
        // Resident norm/table pointers and the exact-key token buffer are stable
        // throughout Prefill replay. Decode still needs its separately staged row
        // descriptors and therefore keeps the conservative rejection.
        if (g.graph_capturing && !g.prefill_capture_active)
            g.graph_capture_compatible = false;
        if (!norm_raw) {
            int rc = weight_slice(norm_w_offset, norm_bytes, &norm_raw);
            if (rc) { set_pending(rc, "PLE norm weight upload"); return; }
        }
        if (ple_width <= threads) {
            k_ple_norm_gather_combine<threads, true>
                <<<dim3(n_layers, n_tok), threads, shared_bytes, g.stream>>>(
                    static_cast<float *>(g.bufs[proj]),
                    reinterpret_cast<const float *>(norm_raw), table,
                    static_cast<const uint32_t *>(g.bufs[tokens_buf]), ple_width,
                    n_layers, input_scale, eps, emb_scale, comb_scale, n_tok);
        } else {
            k_ple_norm_gather_combine<threads, false>
                <<<dim3(n_layers, n_tok), threads, shared_bytes, g.stream>>>(
                    static_cast<float *>(g.bufs[proj]),
                    reinterpret_cast<const float *>(norm_raw), table,
                    static_cast<const uint32_t *>(g.bufs[tokens_buf]), ple_width,
                    n_layers, input_scale, eps, emb_scale, comb_scale, n_tok);
        }
    } else {
        std::vector<uint32_t> fallback_tokens;
        const uint32_t * tokens = host_u32_values(
            tokens_buf, n_tok, fallback_tokens);
        if (!tokens) {
            set_pending(CUDA_RC_ERROR, "PLE token readback");
            return;
        }
        if (uint64_t(n_tok) > (UINT64_MAX - norm_bytes) / table_row_bytes) {
            set_pending(CUDA_RC_INVALID, "PLE norm staged row size");
            return;
        }
        const uint64_t prefix = norm_raw ? 0 : norm_bytes;
        const uint64_t staged_bytes = prefix + uint64_t(n_tok) * table_row_bytes;
        int rc = ensure_weight_cache(staged_bytes);
        if (rc) { set_pending(rc, "PLE norm row cache"); return; }
        rc = ensure_ple_host_stage(staged_bytes);
        if (rc) { set_pending(rc, "PLE norm pinned row staging"); return; }
        if (g.forward_decode && decode_graph_candidate() && n_tok == 1) {
            g.decode_ple_desc_valid = true;
            g.decode_ple_table = table_offset;
            g.decode_ple_row_bytes = table_row_bytes;
            g.decode_ple_norm = norm_w_offset;
            g.decode_ple_norm_bytes = norm_bytes;
            g.decode_ple_prefix = prefix;
        }
        auto * staged = static_cast<uint8_t *>(g.ple_stage_host);
        if (prefix) {
            if (norm_w_offset > g.weights_len
                || norm_bytes > g.weights_len - norm_w_offset) {
                set_pending(CUDA_RC_INVALID, "PLE norm range");
                return;
            }
            std::memcpy(staged, g.weights_host + norm_w_offset, size_t(norm_bytes));
        }
        for (uint32_t token = 0; token < n_tok; ++token) {
            const uint64_t row_off = table_offset
                + uint64_t(tokens[token]) * table_row_bytes;
            if (row_off < table_offset || row_off > g.weights_len
                || table_row_bytes > g.weights_len - row_off) {
                set_pending(CUDA_RC_INVALID, "PLE norm row range");
                return;
            }
            std::memcpy(staged + size_t(prefix + uint64_t(token) * table_row_bytes),
                        g.weights_host + row_off, size_t(table_row_bytes));
        }
        const cudaError_t err = g.graph_capturing && g.forward_decode && n_tok == 1
            ? cudaSuccess
            : cudaMemcpyAsync(g.weight_cache, staged, size_t(staged_bytes),
                              cudaMemcpyHostToDevice, g.stream);
        const cudaError_t sync = !async_host_control_enabled() && err == cudaSuccess
            ? cudaStreamSynchronize(g.stream) : cudaSuccess;
        if (err != cudaSuccess || sync != cudaSuccess) {
            set_pending(CUDA_RC_ERROR, "PLE norm batched row upload");
            return;
        }
        const uint8_t * cache = static_cast<const uint8_t *>(g.weight_cache);
        const float * norm_w = norm_raw
            ? reinterpret_cast<const float *>(norm_raw)
            : reinterpret_cast<const float *>(cache);
        const uint8_t * rows = cache + prefix;
        if (ple_width <= threads) {
            k_ple_norm_gather_combine_packed<threads, true>
                <<<dim3(n_layers, n_tok), threads, shared_bytes, g.stream>>>(
                    static_cast<float *>(g.bufs[proj]), norm_w, rows, ple_width,
                    n_layers, input_scale, eps, emb_scale, comb_scale, n_tok);
        } else {
            k_ple_norm_gather_combine_packed<threads, false>
                <<<dim3(n_layers, n_tok), threads, shared_bytes, g.stream>>>(
                    static_cast<float *>(g.bufs[proj]), norm_w, rows, ple_width,
                    n_layers, input_scale, eps, emb_scale, comb_scale, n_tok);
        }
    }
    mark_buf_written(proj);
}

// Data-only CUDA Driver module bridge. Kept in a separate source file for review,
// included here so existing source and per-SM DLL builds remain one translation unit.
#include "program_pack.cu"
#if defined(IMPARO_CUDA_PROGRAM_SMOKE)
#include "program_pack_test.cu"
#endif
