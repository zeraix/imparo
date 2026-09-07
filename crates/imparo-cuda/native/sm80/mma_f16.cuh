#pragma once

// Reusable SM80+ FP16 Tensor Core primitives.
//
// Keep the PTX fragment ABI in the architecture layer: common kernels express tile
// semantics, while each SM family remains free to replace the physical instruction and
// lane mapping.  The mapping follows mma.sync m16n8k16 and is shared by prefill and the
// correctness fallback; it is not tied to a model, head size, or launch policy.

namespace imparo_sm80_mma {

struct Half16x8 {
    __half2 x[4];
};

struct Float16x16 {
    float x[8];
};

__device__ __forceinline__ void load_half16x8(
        Half16x8 & tile, const __half2 * src, uint32_t stride, uint32_t lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    int * xi = reinterpret_cast<int *>(tile.x);
    const int * xs = reinterpret_cast<const int *>(src)
        + (lane % 16) * stride + (lane / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
#else
    (void)tile; (void)src; (void)stride; (void)lane;
#endif
}

__device__ __forceinline__ void load_half16x8_trans(
        Half16x8 & tile, const __half2 * src, uint32_t stride, uint32_t lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    int * xi = reinterpret_cast<int *>(tile.x);
    const int * xs = reinterpret_cast<const int *>(src)
        + (lane % 16) * stride + (lane / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(xs));
#else
    (void)tile; (void)src; (void)stride; (void)lane;
#endif
}

__device__ __forceinline__ void mma_qk(Float16x16 & dst,
                                        const Half16x8 & q,
                                        const Half16x8 & k) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int * di = reinterpret_cast<int *>(dst.x);
    const int * qi = reinterpret_cast<const int *>(q.x);
    const int * ki = reinterpret_cast<const int *>(k.x);
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
        : "r"(qi[0]), "r"(qi[1]), "r"(qi[2]), "r"(qi[3]),
          "r"(ki[0]), "r"(ki[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(di[4]), "+r"(di[5]), "+r"(di[6]), "+r"(di[7])
        : "r"(qi[0]), "r"(qi[1]), "r"(qi[2]), "r"(qi[3]),
          "r"(ki[1]), "r"(ki[3]));
#else
    (void)dst; (void)q; (void)k;
#endif
}

__device__ __forceinline__ void mma_pv(Half16x8 & dst,
                                        const Half16x8 & probability,
                                        const Half16x8 & value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int * di = reinterpret_cast<int *>(dst.x);
    const int * pi = reinterpret_cast<const int *>(probability.x);
    const int * vi = reinterpret_cast<const int *>(value.x);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(di[0]), "+r"(di[1])
        : "r"(pi[0]), "r"(pi[1]), "r"(pi[2]), "r"(pi[3]),
          "r"(vi[0]), "r"(vi[2]));
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(di[2]), "+r"(di[3])
        : "r"(pi[0]), "r"(pi[1]), "r"(pi[2]), "r"(pi[3]),
          "r"(vi[1]), "r"(vi[3]));
#else
    (void)dst; (void)probability; (void)value;
#endif
}

__device__ __forceinline__ uint32_t half_acc_query_column(uint32_t lane, uint32_t l) {
    return (l % 2) * 8 + lane / 4;
}

__device__ __forceinline__ uint32_t half_acc_output_pair(uint32_t lane, uint32_t l) {
    return (l / 2) * 4 + lane % 4;
}

__device__ __forceinline__ uint32_t fragment_q_column(uint32_t lane, uint32_t l) {
    return ((l / 2) % 2) * 8 + lane / 4;
}

__device__ __forceinline__ uint32_t fragment_key_row(uint32_t lane, uint32_t l) {
    return (l / 4) * 8 + (lane % 4) * 2 + l % 2;
}

// Computes one FP16 dot while retaining the m16n8 fragment position selected by the
// caller.  Filling only the selected row/column is intentional: outputs of an MMA tile
// are independent, and doing this lets a scalar correctness fallback use the exact same
// accumulation instruction and register slot as a tiled FA kernel without duplicating
// the whole prefill scheduler.
__device__ __forceinline__ float dot_f16_fragment(
        const float * q, const __half * k, uint32_t width, float scale,
        uint32_t q_row, uint32_t k_row, __half2 * q_tile, __half2 * k_tile,
        uint32_t lane) {
    Float16x16 accum{};
    for (uint32_t d0 = 0; d0 < width; d0 += 16) {
        for (uint32_t e = lane; e < 16 * 8; e += 32) {
            const uint32_t row = e >> 3;
            const uint32_t pair = e & 7;
            const uint32_t index = d0 + 2 * pair;
            const float q0 = index < width ? q[index] : 0.0f;
            const float q1 = index + 1 < width ? q[index + 1] : 0.0f;
            const __half k0 = index < width ? k[index] : __float2half(0.0f);
            const __half k1 = index + 1 < width ? k[index + 1] : __float2half(0.0f);
            q_tile[e] = row == q_row
                ? __hmul2(__floats2half2_rn(q0, q1), __float2half2_rn(scale))
                : __float2half2_rn(0.0f);
            k_tile[e] = row == k_row
                ? __halves2half2(k0, k1) : __float2half2_rn(0.0f);
        }
        __syncwarp();
        Half16x8 q_fragment;
        Half16x8 k_fragment;
        load_half16x8(q_fragment, q_tile, 8, lane);
        load_half16x8(k_fragment, k_tile, 8, lane);
        mma_qk(accum, q_fragment, k_fragment);
        __syncwarp();
    }

    bool owns = false;
    float selected = 0.0f;
#pragma unroll
    for (uint32_t l = 0; l < 8; ++l) {
        if (fragment_q_column(lane, l) == q_row
                && fragment_key_row(lane, l) == k_row) {
            owns = true;
            selected = accum.x[l];
        }
    }
    const uint32_t owners = __ballot_sync(0xffffffffu, owns);
    const int owner = __ffs(int(owners)) - 1;
    return owner >= 0 ? __shfl_sync(0xffffffffu, selected, owner) : 0.0f;
}

} // namespace imparo_sm80_mma
