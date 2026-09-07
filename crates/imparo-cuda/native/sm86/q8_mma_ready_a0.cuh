#pragma once

// Research-only SM86 A0 experiment for a quantizer-emitted Q8 MMA-ready
// activation layout.  This file is isolated under native/tests: it is not a
// production selector, public/backend ABI, program-pack input, release asset,
// or commercial surface.
//
// Unlike a free post-quantization prepack, the producer below performs the
// dynamic float -> Q8 quantization and writes the final K-major quant plane and
// d8 side plane itself.  Any later performance experiment must time this
// producer together with every consumer in the projection group.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace imparo_q8_mma_ready_a0_v1_authority_lab {

constexpr uint32_t kValuesPerQBlock = 32;
constexpr uint32_t kQBlocksPerGroup = 4;
constexpr uint32_t kValuesPerGroup =
    kValuesPerQBlock * kQBlocksPerGroup;
constexpr uint32_t kTokenTile = 8;
constexpr uint32_t kKPairsPerQBlock = kValuesPerQBlock / 2;
constexpr uint32_t kProducerThreads = 256;
constexpr uint32_t kFragmentThreads = 32;
constexpr uint32_t kQuantBytesPerTile =
    kQBlocksPerGroup * kTokenTile * kValuesPerQBlock;
constexpr uint32_t kScaleBytesPerTile =
    kQBlocksPerGroup * kTokenTile * sizeof(float);
constexpr uint32_t kProducerSharedBytes =
    kQuantBytesPerTile + kScaleBytesPerTile;
constexpr uint32_t kFragmentSharedBytes =
    kTokenTile * kValuesPerQBlock;
constexpr uint32_t kPortableGridYLimit = 65535;

static_assert(kQuantBytesPerTile == 1024, "one K128 x N8 quant tile");
static_assert(kScaleBytesPerTile == 128, "four Q8 scales x eight tokens");
static_assert(kProducerSharedBytes == 1152, "byte-neutral Q8 tile");
static_assert(kFragmentSharedBytes == 256, "one K32 x N8 B fragment");
static_assert(kProducerThreads == kTokenTile * 32,
              "one complete warp must own each token in the N8 tile");

struct Layout {
    uint32_t n_in = 0;
    uint32_t n_tok = 0;
    uint32_t padded_tokens = 0;
    uint32_t token_tiles = 0;
    uint32_t groups = 0;
    uint64_t quant_u16_count = 0;
    uint64_t scale_count = 0;
};

inline bool make_layout(uint32_t n_in, uint32_t n_tok, Layout *layout) {
    if (!layout || !n_in || !n_tok || n_in % kValuesPerGroup != 0
            || n_tok > std::numeric_limits<uint32_t>::max()
                - (kTokenTile - 1)) {
        return false;
    }
    const uint32_t padded =
        ((n_tok + kTokenTile - 1) / kTokenTile) * kTokenTile;
    const uint32_t groups = n_in / kValuesPerGroup;
    if (groups > kPortableGridYLimit) return false;
    const uint64_t q16 = uint64_t(groups) * (padded / kTokenTile)
        * kQBlocksPerGroup * kKPairsPerQBlock * kTokenTile;
    const uint64_t scales = uint64_t(groups) * (padded / kTokenTile)
        * kQBlocksPerGroup * kTokenTile;
    if (q16 > std::numeric_limits<size_t>::max() / sizeof(uint16_t)
            || scales > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    *layout = {n_in, n_tok, padded, padded / kTokenTile, groups,
               q16, scales};
    return true;
}

__host__ __device__ __forceinline__ uint64_t quant_u16_index(
        uint32_t token_tiles, uint32_t group, uint32_t token_tile,
        uint32_t qblock, uint32_t kpair, uint32_t token_in_tile) {
    // [K128 group][N8 tile][Q8 block][K-pair row][token column]
    return (((uint64_t(group) * token_tiles + token_tile)
                * kQBlocksPerGroup + qblock)
                * kKPairsPerQBlock + kpair)
                * kTokenTile + token_in_tile;
}

__host__ __device__ __forceinline__ uint64_t scale_index(
        uint32_t token_tiles, uint32_t group, uint32_t token_tile,
        uint32_t qblock, uint32_t token_in_tile) {
    // [K128 group][N8 tile][Q8 block][token column]
    return (((uint64_t(group) * token_tiles + token_tile)
                * kQBlocksPerGroup + qblock)
                * kTokenTile + token_in_tile);
}

__device__ __forceinline__ float warp_q8_amax(float4 value) {
    float amax = fabsf(value.x);
    amax = fmaxf(amax, fabsf(value.y));
    amax = fmaxf(amax, fabsf(value.z));
    amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(
            amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    return amax;
}

// A CTA owns one K128 x N8 tile.  Each warp owns one token and each eight-lane
// subgroup retains the established Q8 quantization order for one K32 block.
// Shared memory performs the dynamic token-major -> K-major transpose before
// coalesced 32-bit quant-plane stores.  It is producer work, not a free pack.
__launch_bounds__(kProducerThreads, 1)
__global__ void quantize_q8_mma_ready_ds4_a0_v1(
        const float * __restrict__ input,
        uint16_t * __restrict__ quant_u16,
        float * __restrict__ d8_sideplane,
        uint32_t n_in, uint32_t n_tok, uint32_t token_tiles) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) uint8_t token_major_quant[kQuantBytesPerTile];
    __shared__ __align__(16) float token_major_scales[
        kTokenTile * kQBlocksPerGroup];

    const uint32_t tid = threadIdx.x;
    const uint32_t token_in_tile = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t qblock = lane >> 3;
    const uint32_t lane_in_qblock = lane & 7u;
    const uint32_t group = blockIdx.y;
    const uint32_t token_tile = blockIdx.x;
    const uint32_t token = token_tile * kTokenTile + token_in_tile;
    const uint32_t local_k = lane * 4;

    char4 quant = {};
    float rounded_scale = 0.0f;
    if (token < n_tok) {
        const uint64_t input_base = uint64_t(token) * n_in
            + uint64_t(group) * kValuesPerGroup + local_k;
        const float4 value = reinterpret_cast<const float4 *>(
            input + input_base)[0];
        const float amax = warp_q8_amax(value);
        const float d_inv = 127.0f / amax;
        quant.x = int8_t(roundf(value.x * d_inv));
        quant.y = int8_t(roundf(value.y * d_inv));
        quant.z = int8_t(roundf(value.z * d_inv));
        quant.w = int8_t(roundf(value.w * d_inv));
        const float d = 1.0f / d_inv;
        // Q4_0 MMQ's DS4 contract rounds the scale through f16, then carries
        // the exact promoted f32 bits in the side plane.
        rounded_scale = __half2float(__float2half(d));
    }

    reinterpret_cast<char4 *>(
        token_major_quant + token_in_tile * kValuesPerGroup)[lane] = quant;
    if (lane_in_qblock == 0) {
        token_major_scales[token_in_tile * kQBlocksPerGroup + qblock]
            = rounded_scale;
    }
    __syncthreads();

    // 256 threads write 256 contiguous uint32 words.  A word contains two
    // adjacent K bytes for each of two adjacent tokens.
    const uint32_t output_qblock = tid / 64;
    const uint32_t within_qblock = tid % 64;
    const uint32_t kpair = within_qblock / 4;
    const uint32_t token_pair = within_qblock % 4;
    const uint32_t token0 = token_pair * 2;
    const uint32_t byte_in_token = output_qblock * kValuesPerQBlock
        + kpair * 2;
    const uint16_t pair0 = uint16_t(
        token_major_quant[token0 * kValuesPerGroup + byte_in_token])
        | (uint16_t(token_major_quant[
                token0 * kValuesPerGroup + byte_in_token + 1]) << 8);
    const uint16_t pair1 = uint16_t(
        token_major_quant[(token0 + 1) * kValuesPerGroup + byte_in_token])
        | (uint16_t(token_major_quant[
                (token0 + 1) * kValuesPerGroup + byte_in_token + 1]) << 8);
    const uint32_t packed = uint32_t(pair0) | (uint32_t(pair1) << 16);
    const uint64_t q16_base = quant_u16_index(
        token_tiles, group, token_tile, output_qblock, kpair, token0);
    reinterpret_cast<uint32_t *>(quant_u16 + q16_base)[0] = packed;

    if (tid < kTokenTile * kQBlocksPerGroup) {
        const uint32_t scale_qblock = tid / kTokenTile;
        const uint32_t scale_token = tid % kTokenTile;
        d8_sideplane[scale_index(
            token_tiles, group, token_tile, scale_qblock, scale_token)]
            = token_major_scales[
                scale_token * kQBlocksPerGroup + scale_qblock];
    }
#else
    (void)input; (void)quant_u16; (void)d8_sideplane;
    (void)n_in; (void)n_tok; (void)token_tiles;
#endif
}

inline bool launch_quantize_q8_mma_ready_ds4_a0_v1(
        const float *input, uint16_t *quant_u16, float *d8_sideplane,
        uint32_t n_in, uint32_t n_tok, cudaStream_t stream,
        Layout *layout_out = nullptr) {
    Layout layout{};
    if (!input || !quant_u16 || !d8_sideplane
            || !make_layout(n_in, n_tok, &layout)) {
        return false;
    }
    quantize_q8_mma_ready_ds4_a0_v1
        <<<dim3(layout.token_tiles, layout.groups), kProducerThreads, 0,
           stream>>>(input, quant_u16, d8_sideplane,
                     n_in, n_tok, layout.token_tiles);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (layout_out) *layout_out = layout;
    return true;
}

__device__ __forceinline__ uint32_t shared_u32_address_a0_v1(
        const void *pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void load_b_ldmatrix_trans_a0_v1(
        int (&b)[2], const uint16_t *kmajor_tile) {
    // Rows 0..7 are K-pairs 0..7 and rows 8..15 are K-pairs 8..15.
    // Every row contains the eight token columns expected by .trans.
    const uint32_t address_lane = threadIdx.x & 15u;
    const uint32_t address = shared_u32_address_a0_v1(
        kmajor_tile + address_lane * kTokenTile);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
}

__device__ __forceinline__ uint8_t kmajor_qbyte_a0_v1(
        const uint16_t *tile, uint32_t k, uint32_t token) {
    const uint16_t pair = tile[(k / 2) * kTokenTile + token];
    return uint8_t(k & 1u ? pair >> 8 : pair);
}

__device__ __forceinline__ void load_b_scalar_reference_a0_v1(
        int (&b)[2], const uint16_t *tile) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token = lane / 4;
    const uint32_t word = lane % 4;
    uint32_t lo = 0;
    uint32_t hi = 0;
#pragma unroll
    for (uint32_t byte = 0; byte < 4; ++byte) {
        lo |= uint32_t(kmajor_qbyte_a0_v1(
            tile, word * 4 + byte, token)) << (8 * byte);
        hi |= uint32_t(kmajor_qbyte_a0_v1(
            tile, 16 + word * 4 + byte, token)) << (8 * byte);
    }
    b[0] = int(lo);
    b[1] = int(hi);
}

template <bool Candidate>
__device__ __forceinline__ void stage_and_load_b_fragment_a0_v1(
        const uint16_t *quant_u16, uint32_t token_tiles,
        uint32_t group, uint32_t token_tile, uint32_t qblock,
        uint32_t *output) {
    __shared__ __align__(16) uint16_t tile[
        kKPairsPerQBlock * kTokenTile];
    const uint32_t lane = threadIdx.x;
    const uint64_t source_base = quant_u16_index(
        token_tiles, group, token_tile, qblock, 0, 0);
    if (lane < 16) {
        reinterpret_cast<uint4 *>(tile)[lane] =
            reinterpret_cast<const uint4 *>(quant_u16 + source_base)[lane];
    }
    __syncthreads();
    int b[2] = {};
    if constexpr (Candidate) {
        load_b_ldmatrix_trans_a0_v1(b, tile);
    } else {
        load_b_scalar_reference_a0_v1(b, tile);
    }
    const uint64_t block =
        (uint64_t(group) * token_tiles + token_tile)
            * kQBlocksPerGroup + qblock;
    const uint64_t output_base =
        (block * kFragmentThreads + lane) * 2;
    output[output_base] = uint32_t(b[0]);
    output[output_base + 1] = uint32_t(b[1]);
}

template <bool Candidate>
__launch_bounds__(kFragmentThreads, 1)
__global__ void b_fragment_probe_a0_v1(
        const uint16_t *quant_u16, uint32_t groups,
        uint32_t token_tiles, uint32_t *output) {
#if __CUDA_ARCH__ >= 800
    const uint32_t flat = blockIdx.x;
    const uint32_t qblock = flat % kQBlocksPerGroup;
    const uint32_t group_tile = flat / kQBlocksPerGroup;
    const uint32_t token_tile = group_tile % token_tiles;
    const uint32_t group = group_tile / token_tiles;
    if (group >= groups) return;
    stage_and_load_b_fragment_a0_v1<Candidate>(
        quant_u16, token_tiles, group, token_tile, qblock, output);
#else
    (void)quant_u16; (void)groups; (void)token_tiles; (void)output;
#endif
}

} // namespace imparo_q8_mma_ready_a0_v1_authority_lab
