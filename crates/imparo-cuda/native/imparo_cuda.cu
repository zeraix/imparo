// CUDA backend host glue + kernels. WRITTEN ON A MAC, NEVER COMPILED BY NVCC HERE --
// reviewable best-effort code for the Windows/CUDA developer to build and verify
// (task #19). The contracts mirror crates/imparo-metal/native/imparo_metal.mm
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

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

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
// Capacity for imparo_backend::BufId::COUNT ids. Metal caps at 30 because its hazard mask
// is a uint32_t with two bits reserved for the KV sides; this backend has no such mask, so
// the number here is only a table size -- keep it at least BufId::COUNT.
constexpr int B_COUNT = 30;
constexpr int MAX_LAYERS = 128;

struct State {
    void * weights = nullptr;        // device copy of the GGUF tensor blob
    uint64_t weights_len = 0;
    void * bufs[B_COUNT] = {};
    uint64_t sizes[B_COUNT] = {};
    bool in_arena[B_COUNT] = {};
    void * arena = nullptr;
    uint64_t arena_size = 0;
    void * kv_k[MAX_LAYERS] = {};
    void * kv_v[MAX_LAYERS] = {};
    uint64_t kv_bytes[MAX_LAYERS] = {};
    cudaStream_t stream = nullptr;
    uint32_t kv_type_k = 1, kv_type_v = 1;   // 1 = f16, 2 = q4_0, 8 = q8_0
    uint32_t epilogue = 0;
    // The backend knob table (src/knobs.rs declares names/candidates; the tuner
    // sweeps them; kernels consult the slots they are wired to). UNTUNED kernels
    // above currently hard-code their geometry -- wiring each launch to its slot
    // is part of the CUDA port (marked per knob in knobs.rs).
    uint32_t knobs[32] = {};
};
State g;

// ---- kernels ----------------------------------------------------------------------
// Q4_0: 18-byte blocks, half scale then 16 packed bytes; value i in the low nibble,
// i+16 in the high one; v = (q - 8) * d. Matches the .mm and imparo-cpu bit for bit.

__device__ inline float q4_value(const uint8_t * blk, int i) {
    const __half d = *reinterpret_cast<const __half *>(blk);
    const uint8_t pk = blk[2 + (i & 15)];
    const int q = (i < 16) ? (pk & 0x0F) : (pk >> 4);
    return (float(q) - 8.0f) * __half2float(d);
}

// GEMV: one warp per output row, lanes stride the input. n_tok == 1 decode shape.
__global__ void k_gemv_q4(const uint8_t * w, const float * x, float * y,
                          uint32_t n_in, uint32_t n_out) {
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const uint8_t * wr = w + (uint64_t)row * (n_in / 32) * 18;
    float acc = 0.0f;
    for (uint32_t b = 0; b < n_in / 32; ++b) {
        const uint8_t * blk = wr + (uint64_t)b * 18;
        for (uint32_t i = lane; i < 32; i += 32) {
            acc += q4_value(blk, i) * x[b * 32 + i];
        }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[row] = acc;
}

// GEMM: one thread per (row, token) with a shared-memory staged x tile. Untuned by
// design; the tile/threads/k-split geometry is registry-swept on real hardware.
__global__ void k_gemm_q4(const uint8_t * w, const float * x, float * y,
                          uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                          uint32_t src_row, uint32_t epilogue) {
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
    float * slot = y + (uint64_t)tok * n_out + row;
    if (epilogue) {
        // gelu(gate) * up fused into the up projection's write-back (the .mm's
        // gated-activation epilogue): y currently holds the gate value.
        const float g = *slot;
        const float gg = 0.5f * g * (1.0f + tanhf(0.7978845608028654f
                                    * (g + 0.044715f * g * g * g)));
        *slot = gg * acc;
    } else {
        *slot = acc;
    }
}

__global__ void k_gemm_f32(const float * w, const float * x, float * y,
                           uint32_t n_in, uint32_t n_out, uint32_t n_tok,
                           uint32_t src_row) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= n_out || tok >= n_tok) return;
    const float * wr = w + (uint64_t)row * n_in;
    const float * xr = x + (uint64_t)(src_row + tok) * n_in;
    float acc = 0.0f;
    for (uint32_t i = 0; i < n_in; ++i) acc += wr[i] * xr[i];
    y[(uint64_t)tok * n_out + row] = acc;
}

// rms_norm: one block per row, two passes (sum of squares, then scale [*w] [+add]).
// Matches imparo_rms_norm's arithmetic: inv = rsqrt(mean(sq) + eps).
__global__ void k_rms_norm(const float * weights_blob, float * x, const float * src,
                           uint64_t w_off, uint32_t width, float eps, uint32_t n_row,
                           uint32_t row_stride, uint32_t base_off, int has_w) {
    const uint32_t r = blockIdx.x;
    if (r >= n_row) return;
    float * row = x + base_off + (uint64_t)r * row_stride;
    const float * srow = src + base_off + (uint64_t)r * row_stride;
    __shared__ float part[32];
    float sq = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) sq += srow[i] * srow[i];
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (uint32_t i = 0; i < (blockDim.x + 31) / 32; ++i) t += part[i];
        part[0] = rsqrtf(t / float(width) + eps);
    }
    __syncthreads();
    const float inv = part[0];
    const float * w = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(weights_blob) + w_off);
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        row[i] = has_w ? srow[i] * inv * w[i] : srow[i] * inv;
    }
}

// rope, neox pairing: dim i rotates with i + n_rot/2; positions offset by start_pos.
// freqs == nullptr means theta = base^(-2i/n_rot) (the workflow uploads
// rope_freqs.weight when the model carries it).
__global__ void k_rope(float * x, const float * freqs, uint32_t n_rot, float base,
                       uint32_t head_dim, uint32_t n_heads, uint32_t start_pos,
                       uint32_t n_tok) {
    const uint32_t half_rot = n_rot / 2;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t per_tok = n_heads * half_rot;
    if (idx >= n_tok * per_tok) return;
    const uint32_t t = idx / per_tok;
    const uint32_t h = (idx % per_tok) / half_rot;
    const uint32_t i = idx % half_rot;
    float theta = powf(base, -2.0f * float(i) / float(n_rot));
    if (freqs) theta /= freqs[i];
    const float a = float(start_pos + t) * theta;
    const float c = cosf(a), s = sinf(a);
    float * hd = x + ((uint64_t)t * n_heads + h) * head_dim;
    const float x0 = hd[i], x1 = hd[i + half_rot];
    hd[i] = x0 * c - x1 * s;
    hd[i + half_rot] = x0 * s + x1 * c;
}

// FWHT, natural (Sylvester) order -- the quantized-KV rotation. Matches
// imparo_hadamard64's butterfly and its fixed pair order.
__global__ void k_hadamard(float * x, uint32_t n, uint32_t nrot, float scale) {
    extern __shared__ float blk[];
    const uint32_t b0 = blockIdx.x * nrot;
    if (b0 + nrot > n) return;
    for (uint32_t i = threadIdx.x; i < nrot; i += blockDim.x) blk[i] = x[b0 + i];
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
    for (uint32_t i = threadIdx.x; i < nrot; i += blockDim.x) x[b0 + i] = blk[i] * scale;
}

// kv_store, f16: slot = ring ? (pos & ring) : pos. One thread per value.
__global__ void k_kv_store_f16(const float * src, __half * dst, uint32_t width,
                               uint32_t start_pos, uint32_t n_tok, uint32_t ring) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= width || t >= n_tok) return;
    const uint32_t gp = start_pos + t;
    const uint32_t ps = ring ? (gp & ring) : gp;
    dst[(uint64_t)ps * width + i] = __float2half(src[(uint64_t)t * width + i]);
}

// kv_store, q4_0: llama.cpp's quantize_q4_0, byte-for-byte the .mm sibling. One
// thread per 32-value block.
__global__ void k_kv_store_q4(const float * src, uint8_t * dst, uint32_t width,
                              uint32_t start_pos, uint32_t n_tok, uint32_t ring) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t blocks = width / 32;
    if (b >= blocks || t >= n_tok) return;
    const float * x = src + (uint64_t)t * width + b * 32;
    const uint32_t gp = start_pos + t;
    const uint32_t ps = ring ? (gp & ring) : gp;
    uint8_t * blk = dst + ((uint64_t)ps * blocks + b) * 18;
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float v = x[j];
        if (fabsf(v) > amax) { amax = fabsf(v); vmax = v; }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const __half dh = __float2half(d);
    std::memcpy(blk, &dh, 2);
    for (int j = 0; j < 16; ++j) {
        int q0 = int(x[j] * id + 8.5f);      if (q0 > 15) q0 = 15; if (q0 < 0) q0 = 0;
        int q1 = int(x[16 + j] * id + 8.5f); if (q1 > 15) q1 = 15; if (q1 < 0) q1 = 0;
        blk[2 + j] = uint8_t(q0 | (q1 << 4));
    }
}

__global__ void k_kv_store_q8(const float * src, uint8_t * dst, uint32_t width,
                              uint32_t start_pos, uint32_t n_tok, uint32_t ring) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t blocks = width / 32;
    if (b >= blocks || t >= n_tok) return;
    const float * x = src + (uint64_t)t * width + b * 32;
    const uint32_t ps = ring ? ((start_pos + t) & ring) : (start_pos + t);
    uint8_t * blk = dst + ((uint64_t)ps * blocks + b) * 34;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const __half dh = __float2half(d);
    std::memcpy(blk, &dh, 2);
    for (int j = 0; j < 32; ++j) blk[2 + j] = uint8_t(int8_t(rintf(x[j] * id)));
}

// kv_dequant: whole cache slice into a half scratch, value = (q-8)*d / q*d.
__global__ void k_kv_dequant(const uint8_t * src, __half * dst, uint32_t width,
                             uint32_t slots, uint32_t ktype) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t s = blockIdx.y;
    const uint32_t blocks = width / 32;
    if (b >= blocks || s >= slots) return;
    __half * out = dst + (uint64_t)s * width + b * 32;
    if (ktype == 2) {
        const uint8_t * blk = src + ((uint64_t)s * blocks + b) * 18;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        for (int i = 0; i < 32; ++i) {
            const uint8_t pk = blk[2 + (i & 15)];
            const int q = (i < 16) ? (pk & 0x0F) : (pk >> 4);
            out[i] = __float2half((float(q) - 8.0f) * d);
        }
    } else {
        const uint8_t * blk = src + ((uint64_t)s * blocks + b) * 34;
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        for (int i = 0; i < 32; ++i) {
            out[i] = __float2half(float(int8_t(blk[2 + i])) * d);
        }
    }
}

// Attention, flash-style: ONE BLOCK per (head, query). Online softmax over position
// chunks, accumulator in registers/shared. The algorithm (bounds, masking, the
// dead-row rule, the causal window `[max(0, pos+1-window), pos]`) matches
// attention_qcomb_body; the parallel decomposition is deliberately simpler --
// per-query instead of 8-query tiles -- because correctness comes first and the
// CUDA tile geometry is a registry knob for real hardware.
__global__ void k_attention(const float * q, const __half * kc, const __half * vc,
                            float * out, uint32_t head_dim, uint32_t n_heads,
                            uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
                            uint32_t window, uint32_t ring, uint32_t n_tok) {
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
    if (threadIdx.x == 0) { m_run = -1e30f; s_run = 0.0f; }
    __syncthreads();
    for (uint32_t gp = lo; gp <= pos; ++gp) {
        const uint32_t ps = ring ? (gp & ring) : gp;
        const __half * kr = kc + (uint64_t)ps * kv_width + kvh * head_dim;
        float dot = 0.0f;
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
            dot += qr[i] * __half2float(kr[i]);
        }
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, off);
        __shared__ float warp_dot[32];
        if ((threadIdx.x & 31) == 0) warp_dot[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float d = 0.0f;
            for (uint32_t wi = 0; wi < (blockDim.x + 31) / 32; ++wi) d += warp_dot[wi];
            warp_dot[0] = d;
        }
        __syncthreads();
        const float score = warp_dot[0];
        const float m_new = fmaxf(m_run, score);
        const float scale = expf(m_run - m_new);
        const float p = expf(score - m_new);
        const __half * vr = vc + (uint64_t)ps * kv_width + kvh * head_dim;
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
            acc[i] = acc[i] * scale + p * __half2float(vr[i]);
        }
        __syncthreads();
        if (threadIdx.x == 0) { s_run = s_run * scale + p; m_run = m_new; }
        __syncthreads();
    }
    float * op = out + ((uint64_t)t * n_heads + h) * head_dim;
    const float inv = s_run > 0.0f ? 1.0f / s_run : 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) op[i] = acc[i] * inv;
}

// ---- elementwise family (contracts identical to the .mm kernels) ------------------
__global__ void k_gelu(float * a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    a[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
__global__ void k_gelu_mul(float * a, const float * b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    a[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f
             * (x + 0.044715f * x * x * x))) * b[i];
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
// argmax with the smallest-index-at-max rule, single block tree reduce.
__global__ void k_argmax(const float * src, uint32_t * dst, uint32_t n) {
    __shared__ float bv[256];
    __shared__ uint32_t bi[256];
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
    if (threadIdx.x == 0) dst[0] = bi[0];
}
// embedding row: dst[dst_off .. dst_off+width) = dequant(weights row) * scale.
__global__ void k_row_q4(const uint8_t * w, float * dst, uint64_t w_off,
                         uint32_t width, uint32_t index, float scale, uint32_t dst_off) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const uint8_t * blk = w + w_off + ((uint64_t)index * (width / 32) + i / 32) * 18;
    dst[dst_off + i] = q4_value(blk, i & 31) * scale;
}

} // namespace

// ---- extern "C" surface (mirrors the .mm's; the Rust backend_impl calls these) ----
// Every function returns 0 on success where a result is meaningful. The .def export
// list for the Windows DLL build lives next to this file.

extern "C" int imparo_cuda_init(const void * weights_host, uint64_t len) {
    CUDA_OK(cudaSetDevice(0));
    CUDA_OK(cudaStreamCreate(&g.stream));
    // No unified memory mapping as on Apple silicon: the tensor blob is UPLOADED.
    // A Windows pinned-memory staging path is a later optimization.
    CUDA_OK(cudaMalloc(&g.weights, len));
    CUDA_OK(cudaMemcpyAsync(g.weights, weights_host, len, cudaMemcpyHostToDevice, g.stream));
    CUDA_OK(cudaStreamSynchronize(g.stream));
    g.weights_len = len;
    return 0;
}

extern "C" void imparo_cuda_begin(void) { /* stream model: nothing to open */ }
extern "C" void imparo_cuda_flush(void) { /* async stream: kernels already queued */ }
extern "C" int imparo_cuda_end(void) { CUDA_OK(cudaStreamSynchronize(g.stream)); return 0; }

extern "C" int imparo_cuda_alloc(uint32_t id, uint64_t bytes) {
    if (id >= B_COUNT) return 1;
    if (g.bufs[id] && !g.in_arena[id] && g.sizes[id] >= bytes
        && bytes * 2 >= g.sizes[id]) return 0;   // the .mm's shrink hysteresis
    if (g.bufs[id] && !g.in_arena[id]) cudaFree(g.bufs[id]);
    CUDA_OK(cudaMalloc(&g.bufs[id], bytes));
    CUDA_OK(cudaMemsetAsync(g.bufs[id], 0, bytes, g.stream));
    g.sizes[id] = bytes; g.in_arena[id] = false;
    return 0;
}
extern "C" int imparo_cuda_arena(uint64_t bytes) {
    if (g.arena && bytes <= g.arena_size && bytes * 2 >= g.arena_size) return 0;
    for (int i = 0; i < B_COUNT; ++i) if (g.in_arena[i]) { g.bufs[i] = nullptr; g.in_arena[i] = false; }
    if (g.arena) cudaFree(g.arena);
    CUDA_OK(cudaMalloc(&g.arena, bytes));
    CUDA_OK(cudaMemsetAsync(g.arena, 0, bytes, g.stream));
    g.arena_size = bytes;
    return 0;
}
extern "C" int imparo_cuda_place(uint32_t id, uint64_t offset, uint64_t bytes) {
    if (id >= B_COUNT || !g.arena || offset + bytes > g.arena_size) return 1;
    g.bufs[id] = static_cast<uint8_t *>(g.arena) + offset;
    g.sizes[id] = bytes; g.in_arena[id] = true;
    return 0;
}
extern "C" uint64_t imparo_cuda_page_round(uint64_t n) {
    return (n + 4095) / 4096 * 4096;
}
extern "C" int imparo_cuda_alloc_kv(uint32_t n_layers, const uint64_t * bytes) {
    for (uint32_t i = 0; i < n_layers && i < MAX_LAYERS; ++i) {
        if (!bytes[i]) continue;
        CUDA_OK(cudaMalloc(&g.kv_k[i], bytes[i]));
        CUDA_OK(cudaMalloc(&g.kv_v[i], bytes[i]));
        CUDA_OK(cudaMemsetAsync(g.kv_k[i], 0, bytes[i], g.stream));
        CUDA_OK(cudaMemsetAsync(g.kv_v[i], 0, bytes[i], g.stream));
        g.kv_bytes[i] = bytes[i];
    }
    return 0;
}
extern "C" int imparo_cuda_grow_kv(uint32_t n_layers, const uint64_t * bytes) {
    for (uint32_t i = 0; i < n_layers && i < MAX_LAYERS; ++i) {
        if (!bytes[i] || bytes[i] <= g.kv_bytes[i]) continue;
        for (int kv = 0; kv < 2; ++kv) {
            void ** slot = kv ? &g.kv_v[i] : &g.kv_k[i];
            void * fresh = nullptr;
            CUDA_OK(cudaMalloc(&fresh, bytes[i]));
            CUDA_OK(cudaMemsetAsync(fresh, 0, bytes[i], g.stream));
            if (*slot) {
                CUDA_OK(cudaMemcpyAsync(fresh, *slot, g.kv_bytes[i],
                                        cudaMemcpyDeviceToDevice, g.stream));
                CUDA_OK(cudaStreamSynchronize(g.stream));
                cudaFree(*slot);
            }
            *slot = fresh;
        }
        g.kv_bytes[i] = bytes[i];
    }
    return 0;
}
extern "C" void imparo_cuda_write(uint32_t id, uint64_t off, const float * src, uint64_t n) {
    cudaMemcpyAsync(static_cast<float *>(g.bufs[id]) + off, src, n * 4,
                    cudaMemcpyHostToDevice, g.stream);
}
extern "C" void imparo_cuda_write_u32(uint32_t id, uint64_t off, const uint32_t * src, uint64_t n) {
    cudaMemcpyAsync(static_cast<uint32_t *>(g.bufs[id]) + off, src, n * 4,
                    cudaMemcpyHostToDevice, g.stream);
}
extern "C" void imparo_cuda_read(uint32_t id, uint64_t off, float * dst, uint64_t n) {
    cudaMemcpyAsync(dst, static_cast<const float *>(g.bufs[id]) + off, n * 4,
                    cudaMemcpyDeviceToHost, g.stream);
    cudaStreamSynchronize(g.stream);
}
extern "C" void imparo_cuda_read_kv(uint32_t layer, uint32_t is_v, uint64_t off,
                                    uint8_t * dst, uint64_t n) {
    const void * b = is_v ? g.kv_v[layer] : g.kv_k[layer];
    if (!b) { std::memset(dst, 0xEE, n); return; }
    cudaMemcpy(dst, static_cast<const uint8_t *>(b) + off, n, cudaMemcpyDeviceToHost);
}
extern "C" void imparo_cuda_set_epilogue(uint32_t on) { g.epilogue = on; }
extern "C" void imparo_cuda_set_knob(uint32_t idx, uint32_t v) {
    if (idx < 32) g.knobs[idx] = v;
}
extern "C" uint32_t imparo_cuda_knob(uint32_t idx) { return idx < 32 ? g.knobs[idx] : 0; }
extern "C" void imparo_cuda_set_kv_types(uint32_t k, uint32_t v) {
    g.kv_type_k = k; g.kv_type_v = v;
}

extern "C" void imparo_cuda_matmat(uint32_t wkind, uint64_t w_off, uint32_t n_in,
                                   uint32_t n_out, uint32_t src, uint32_t dst,
                                   uint32_t n_tok, uint32_t src_row) {
    const float * x = static_cast<const float *>(g.bufs[src]);
    float * y = static_cast<float *>(g.bufs[dst]);
    if (wkind == 0) {
        const float * w = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(g.weights) + w_off);
        dim3 grid((n_out + 255) / 256, n_tok);
        k_gemm_f32<<<grid, 256, 0, g.stream>>>(w, x, y, n_in, n_out, n_tok, src_row);
        return;
    }
    // FAIL CLOSED on any kind this backend has no kernel for. Everything below reads
    // Q4_0's 18-byte blocks; a Q8_0 tensor (wkind 2, 34-byte blocks) fed through it
    // returns plausible garbage instead of an error, which is exactly what the
    // WeightKind table exists to prevent. The shared table admits Q8_0 because Metal
    // implements it; CUDA does not yet.
    if (wkind != 1u) {
        std::fprintf(stderr,
                     "imparo cuda: matmat got weight kind %u (n_out=%u); this backend "
                     "implements F32 and Q4_0 only -- refusing the dispatch\n",
                     wkind, n_out);
        return;
    }
    const uint8_t * w = static_cast<const uint8_t *>(g.weights) + w_off;
    if (n_tok == 1 && !g.epilogue) {
        const uint32_t warps = 4;
        k_gemv_q4<<<(n_out + warps - 1) / warps, warps * 32, 0, g.stream>>>(
            w, x + (uint64_t)src_row * n_in, y, n_in, n_out);
        return;
    }
    dim3 grid((n_out + 255) / 256, n_tok);
    k_gemm_q4<<<grid, 256, 0, g.stream>>>(w, x, y, n_in, n_out, n_tok, src_row,
                                          g.epilogue);
}

extern "C" uint32_t imparo_cuda_buf_count(void) { return (uint32_t)B_COUNT; }

extern "C" void imparo_cuda_rms_norm(uint32_t buf, uint32_t src, uint64_t w_off,
                                     uint32_t width, float eps, uint32_t n_row,
                                     uint32_t row_stride, uint32_t base_off,
                                     uint32_t has_w) {
    k_rms_norm<<<n_row, 256, 0, g.stream>>>(
        static_cast<const float *>(g.weights), static_cast<float *>(g.bufs[buf]),
        static_cast<const float *>(g.bufs[src]), w_off, width, eps, n_row,
        row_stride, base_off, int(has_w));
}

extern "C" void imparo_cuda_rope(uint32_t buf, uint32_t n_rot, float base,
                                 uint32_t head_dim, uint32_t n_heads,
                                 uint32_t start_pos, uint32_t n_tok,
                                 const float * freqs_dev) {
    const uint32_t total = n_tok * n_heads * (n_rot / 2);
    k_rope<<<(total + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[buf]), freqs_dev, n_rot, base, head_dim,
        n_heads, start_pos, n_tok);
}

extern "C" void imparo_cuda_hadamard(uint32_t buf, uint32_t n, uint32_t nrot) {
    const float scale = rsqrtf(float(nrot));
    const uint32_t thr = nrot < 256 ? nrot : 256;
    k_hadamard<<<n / nrot, thr, nrot * 4, g.stream>>>(
        static_cast<float *>(g.bufs[buf]), n, nrot, scale);
}

extern "C" void imparo_cuda_kv_store(uint32_t src, uint32_t layer, uint32_t width,
                                     uint32_t start_pos, uint32_t n_tok, uint32_t is_v,
                                     uint32_t ring) {
    const float * s = static_cast<const float *>(g.bufs[src]);
    void * dstb = is_v ? g.kv_v[layer] : g.kv_k[layer];
    const uint32_t kt = is_v ? g.kv_type_v : g.kv_type_k;
    if (kt == 2) {
        dim3 grid((width / 32 + 63) / 64, n_tok);
        k_kv_store_q4<<<grid, 64, 0, g.stream>>>(s, static_cast<uint8_t *>(dstb),
                                                 width, start_pos, n_tok, ring);
    } else if (kt == 8) {
        dim3 grid((width / 32 + 63) / 64, n_tok);
        k_kv_store_q8<<<grid, 64, 0, g.stream>>>(s, static_cast<uint8_t *>(dstb),
                                                 width, start_pos, n_tok, ring);
    } else {
        dim3 grid((width + 255) / 256, n_tok);
        k_kv_store_f16<<<grid, 256, 0, g.stream>>>(s, static_cast<__half *>(dstb),
                                                   width, start_pos, n_tok, ring);
    }
}

extern "C" void imparo_cuda_kv_dequant(uint32_t layer, uint32_t width, uint32_t slots,
                                       uint32_t is_v, uint32_t scratch_buf) {
    const uint32_t kt = is_v ? g.kv_type_v : g.kv_type_k;
    if (kt == 1) return;
    dim3 grid((width / 32 + 63) / 64, slots);
    k_kv_dequant<<<grid, 64, 0, g.stream>>>(
        static_cast<const uint8_t *>(is_v ? g.kv_v[layer] : g.kv_k[layer]),
        static_cast<__half *>(g.bufs[scratch_buf]), width, slots, kt);
}

extern "C" void imparo_cuda_attention(uint32_t kv_layer, uint32_t head_dim,
                                      uint32_t n_heads, uint32_t n_kv,
                                      uint32_t kv_width, uint32_t start_pos,
                                      uint32_t window, uint32_t n_tok, uint32_t ring,
                                      uint32_t q_buf, uint32_t out_buf,
                                      uint32_t kdq_buf, uint32_t vdq_buf) {
    const __half * kc = g.kv_type_k != 1
        ? static_cast<const __half *>(g.bufs[kdq_buf])
        : static_cast<const __half *>(g.kv_k[kv_layer]);
    const __half * vc = g.kv_type_v != 1
        ? static_cast<const __half *>(g.bufs[vdq_buf])
        : static_cast<const __half *>(g.kv_v[kv_layer]);
    dim3 grid(n_heads, n_tok);
    k_attention<<<grid, 128, head_dim * 4, g.stream>>>(
        static_cast<const float *>(g.bufs[q_buf]), kc, vc,
        static_cast<float *>(g.bufs[out_buf]), head_dim, n_heads, n_kv, kv_width,
        start_pos, window, ring, n_tok);
}

extern "C" void imparo_cuda_gelu(uint32_t a, uint32_t n) {
    k_gelu<<<(n + 255) / 256, 256, 0, g.stream>>>(static_cast<float *>(g.bufs[a]), n);
}
extern "C" void imparo_cuda_gelu_mul(uint32_t a, uint32_t b, uint32_t n) {
    k_gelu_mul<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]), n);
}
extern "C" void imparo_cuda_add(uint32_t a, uint32_t b, uint32_t n) {
    k_add<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]), n);
}
extern "C" void imparo_cuda_add_scale(uint32_t a, uint32_t b, float k, uint32_t n) {
    k_add_scale<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]), k, n);
}
extern "C" void imparo_cuda_scale(uint32_t a, float k, uint32_t n) {
    k_scale<<<(n + 255) / 256, 256, 0, g.stream>>>(static_cast<float *>(g.bufs[a]), k, n);
}
extern "C" void imparo_cuda_copy(uint32_t dst, uint32_t src, uint32_t n) {
    k_copy<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[dst]), static_cast<const float *>(g.bufs[src]), n);
}
extern "C" void imparo_cuda_mul_strided(uint32_t a, uint32_t b, uint32_t n,
                                        uint32_t b_off, uint32_t b_stride,
                                        uint32_t a_stride, uint32_t n_tok) {
    dim3 grid((n + 255) / 256, n_tok);
    k_mul_strided<<<grid, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), static_cast<const float *>(g.bufs[b]),
        n, b_off, b_stride, a_stride, n_tok);
}
extern "C" void imparo_cuda_softcap(uint32_t a, float cap, uint32_t n) {
    k_softcap<<<(n + 255) / 256, 256, 0, g.stream>>>(
        static_cast<float *>(g.bufs[a]), cap, n);
}
extern "C" void imparo_cuda_argmax(uint32_t src, uint32_t dst, uint32_t n) {
    k_argmax<<<1, 256, 0, g.stream>>>(static_cast<const float *>(g.bufs[src]),
                                      static_cast<uint32_t *>(g.bufs[dst]), n);
}
extern "C" void imparo_cuda_row(uint64_t w_off, uint32_t width, uint32_t index,
                                float scale, uint32_t dst, uint32_t dst_off) {
    k_row_q4<<<(width + 255) / 256, 256, 0, g.stream>>>(
        static_cast<const uint8_t *>(g.weights), static_cast<float *>(g.bufs[dst]),
        w_off, width, index, scale, dst_off);
}
