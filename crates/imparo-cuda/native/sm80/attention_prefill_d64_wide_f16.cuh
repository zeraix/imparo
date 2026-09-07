#pragma once

// Native SM80 D64/GQA4 wide attention.
//
// This is an Imparo-owned implementation of the public numerical contract used
// by the D64 64-column family: one CTA owns one KV head and one 64-key Stream-K
// block, and its four warps own consecutive 16-column query fragments.  Keeping
// the 64-key softmax and half-PV numerator inside the same CTA avoids the
// additional 16-column/32-key seams introduced by chaining the small-query
// kernel.  A separate bounded fixup pass combines blocks from last to first.
namespace imparo_sm80_d64_wide {

constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kQueryTokens = 16;
constexpr uint32_t kColumns = kQueryTokens * kGqaHeads;
constexpr uint32_t kKeysPerBlock = 64;
constexpr uint32_t kKeyTiles = kKeysPerBlock / 16;
constexpr uint32_t kWarps = 4;
constexpr uint32_t kThreads = kWarps * 32;
constexpr uint32_t kOutputTiles = kHeadDim / 16;

static_assert(kColumns == 64, "D64 wide owns 64 query/head columns");
static_assert(kKeysPerBlock == 64, "D64 wide Stream-K block is 64 keys");
static_assert(kThreads == 128, "D64 wide uses four warps");

__host__ __device__ constexpr uint32_t key_blocks(uint32_t kv_span) {
    return (kv_span + kKeysPerBlock - 1) / kKeysPerBlock;
}

__host__ __device__ constexpr uint64_t numerator_halves_per_head(
        uint32_t blocks) {
    return uint64_t(blocks) * kColumns * kHeadDim;
}

__host__ __device__ constexpr uint64_t meta_floats_per_head(uint32_t blocks) {
    return uint64_t(blocks) * kColumns * 2;
}

__host__ __device__ constexpr uint64_t workspace_bytes_per_head(
        uint32_t blocks) {
    return numerator_halves_per_head(blocks) * sizeof(__half)
        + meta_floats_per_head(blocks) * sizeof(float);
}

static_assert(key_blocks(1) == 1);
static_assert(key_blocks(64) == 1);
static_assert(key_blocks(65) == 2);
static_assert(key_blocks(256) == 4);
static_assert(numerator_halves_per_head(4) == 16384);
static_assert(meta_floats_per_head(4) == 512);

__device__ __forceinline__ uint64_t numerator_index(
        uint32_t local_head, uint32_t blocks, uint32_t block,
        uint32_t column, uint32_t output) {
    return ((uint64_t(local_head) * blocks + block) * kColumns + column)
        * kHeadDim + output;
}

__device__ __forceinline__ uint64_t meta_index(
        uint32_t local_head, uint32_t blocks, uint32_t block,
        uint32_t column, uint32_t field) {
    return ((uint64_t(local_head) * blocks + block) * kColumns + column) * 2
        + field;
}

// One CTA evaluates one (KV head, 64-key block).  Each warp owns the same
// 16-column fragment identity as the corresponding warp of the native 64-column
// family.  Scores stay in f32 fragments; probabilities and PV accumulation use
// the project's shared half-MMA ABI.
__global__ void partials(
        const float * q, const __half * kc, const __half * vc,
        __half * numerators, float * meta, uint32_t block_base,
        uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, uint32_t window, uint32_t n_tok, uint32_t ring,
        float qk_scale, uint32_t valid_span, uint32_t blocks,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t local_head = blockIdx.x;
    const uint32_t kvh = block_base + local_head;
    const uint32_t key_block = blockIdx.y;
    if (kvh >= n_kv || key_block >= blocks) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    constexpr uint32_t stride = 12;
    __shared__ __align__(16) __half2 q_tile[kWarps][16 * stride];
    __shared__ __align__(16) __half2 k_tile[kWarps][16 * stride];
    __shared__ __align__(16) __half probability_tile[kKeyTiles][kWarps][16 * 16];
    __shared__ __align__(16) __half2 value_tile[kWarps][16 * 8];

    imparo_sm80_mma::Float16x16 score[kKeyTiles]{};
    for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
        for (uint32_t e = lane; e < 16 * 8; e += 32) {
            const uint32_t row = e >> 3;
            const uint32_t pair = e & 7;
            const uint32_t column = warp * 16 + row;
            const uint32_t token = column / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + column % kGqaHeads;
            __half2 value = __float2half2_rn(0.0f);
            if (token < n_tok && head < n_heads) {
                const float * qr = q
                    + (uint64_t(token) * n_heads + head) * kHeadDim
                    + d0 + 2 * pair;
                value = __hmul2(__floats2half2_rn(qr[0], qr[1]),
                    __float2half2_rn(qk_scale));
            }
            q_tile[warp][row * stride + pair] = value;
        }
        __syncwarp();
        imparo_sm80_mma::Half16x8 q_fragment;
        imparo_sm80_mma::load_half16x8(
            q_fragment, q_tile[warp], stride, lane);

#pragma unroll
        for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
            const uint32_t key0 = key_block * kKeysPerBlock + tile * 16;
            for (uint32_t e = lane; e < 16 * 8; e += 32) {
                const uint32_t row = e >> 3;
                const uint32_t pair = e & 7;
                const uint32_t key = key0 + row;
                __half2 value = __float2half2_rn(0.0f);
                if (key < valid_span) {
                    const uint32_t physical =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    value = *reinterpret_cast<const __half2 *>(
                        kc + uint64_t(physical) * kv_width + kvh * kHeadDim
                            + d0 + 2 * pair);
                }
                k_tile[warp][row * stride + pair] = value;
            }
            __syncwarp();
            imparo_sm80_mma::Half16x8 k_fragment;
            imparo_sm80_mma::load_half16x8(
                k_fragment, k_tile[warp], stride, lane);
            imparo_sm80_mma::mma_qk(score[tile], q_fragment, k_fragment);
            __syncwarp();
        }
    }

    float row_max[2] = {-1.701411733e+38F, -1.701411733e+38F};
    bool valid_score[kKeyTiles][8];
#pragma unroll
    for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
#pragma unroll
        for (uint32_t l = 0; l < 8; ++l) {
            const uint32_t local_column =
                imparo_sm80_mma::fragment_q_column(lane, l);
            const uint32_t column = warp * 16 + local_column;
            const uint32_t token = column / kGqaHeads;
            const uint32_t pos = start_pos + token;
            const uint32_t lo = window > 0 && pos + 1 > window
                ? pos + 1 - window : 0;
            const uint32_t key = key_block * kKeysPerBlock + tile * 16
                + imparo_sm80_mma::fragment_key_row(lane, l);
            const uint32_t key_pos = key < valid_span
                ? imparo_sm80_prefill::physical_key_position(
                    key, ring, start_pos + n_tok - 1)
                : 0;
            const bool valid = token < n_tok && key < valid_span
                && key_pos >= lo && key_pos <= pos;
            valid_score[tile][l] = valid;
            if (valid) {
                const uint32_t owned_column = (l / 2) % 2;
                row_max[owned_column] = fmaxf(
                    row_max[owned_column], score[tile].x[l]
                        + 3.0f * 0.6931f);
            }
        }
    }
#pragma unroll
    for (uint32_t owned_column = 0; owned_column < 2; ++owned_column) {
#pragma unroll
        for (int offset = 2; offset >= 1; offset >>= 1) {
            row_max[owned_column] = fmaxf(
                row_max[owned_column],
                __shfl_xor_sync(0xffffffffu, row_max[owned_column], offset));
        }
    }

    float row_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
#pragma unroll
        for (uint32_t l = 0; l < 8; ++l) {
            const uint32_t owned_column = (l / 2) % 2;
            const float probability = valid_score[tile][l]
                ? expf(score[tile].x[l] - row_max[owned_column]) : 0.0f;
            score[tile].x[l] = probability;
            row_sum[owned_column] += probability;
            const uint32_t local_column =
                imparo_sm80_mma::fragment_q_column(lane, l);
            const uint32_t key_row =
                imparo_sm80_mma::fragment_key_row(lane, l);
            probability_tile[tile][warp][local_column * 16 + key_row] =
                __float2half(probability);
        }
    }
#pragma unroll
    for (uint32_t owned_column = 0; owned_column < 2; ++owned_column) {
#pragma unroll
        for (int offset = 2; offset >= 1; offset >>= 1) {
            row_sum[owned_column] += __shfl_xor_sync(
                0xffffffffu, row_sum[owned_column], offset);
        }
    }
    if ((lane & 3) == 0) {
        const uint32_t local_column = lane >> 2;
        const uint32_t column0 = warp * 16 + local_column;
        const uint32_t column1 = column0 + 8;
        meta[meta_index(local_head, blocks, key_block, column0, 0)] = row_max[0];
        meta[meta_index(local_head, blocks, key_block, column0, 1)] = row_sum[0];
        meta[meta_index(local_head, blocks, key_block, column1, 0)] = row_max[1];
        meta[meta_index(local_head, blocks, key_block, column1, 1)] = row_sum[1];
    }
    __syncwarp();

    imparo_sm80_mma::Half16x8 numerator[kOutputTiles];
#pragma unroll
    for (uint32_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            numerator[output_tile].x[l] = __float2half2_rn(0.0f);
        }
    }
#pragma unroll
    for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
        const uint32_t key0 = key_block * kKeysPerBlock + tile * 16;
        const __half2 * probability = reinterpret_cast<const __half2 *>(
            probability_tile[tile][warp]);
        imparo_sm80_mma::Half16x8 probability_fragment;
        imparo_sm80_mma::load_half16x8(
            probability_fragment, probability, 8, lane);
#pragma unroll
        for (uint32_t output_tile = 0;
             output_tile < kOutputTiles; ++output_tile) {
            for (uint32_t e = lane; e < 16 * 8; e += 32) {
                const uint32_t key = key0 + (e >> 3);
                const uint32_t pair = e & 7;
                __half2 value = __float2half2_rn(0.0f);
                if (key < valid_span) {
                    const uint32_t physical =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    value = *reinterpret_cast<const __half2 *>(
                        vc + uint64_t(physical) * kv_width + kvh * kHeadDim
                            + output_tile * 16 + 2 * pair);
                }
                value_tile[warp][(e >> 3) * 8 + pair] = value;
            }
            __syncwarp();
            imparo_sm80_mma::Half16x8 value_fragment;
            imparo_sm80_mma::load_half16x8_trans(
                value_fragment, value_tile[warp], 8, lane);
            imparo_sm80_mma::mma_pv(
                numerator[output_tile], probability_fragment, value_fragment);
            __syncwarp();
        }
    }

#pragma unroll
    for (uint32_t output_tile = 0;
         output_tile < kOutputTiles; ++output_tile) {
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            const uint32_t local_column =
                imparo_sm80_mma::half_acc_query_column(lane, l);
            const uint32_t column = warp * 16 + local_column;
            const uint32_t output_pair =
                imparo_sm80_mma::half_acc_output_pair(lane, l);
            const uint32_t output = output_tile * 16 + 2 * output_pair;
            numerators[numerator_index(
                local_head, blocks, key_block, column, output)] =
                __low2half(numerator[output_tile].x[l]);
            numerators[numerator_index(
                local_head, blocks, key_block, column, output + 1)] =
                __high2half(numerator[output_tile].x[l]);
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)numerators; (void)meta;
    (void)block_base; (void)n_heads; (void)n_kv; (void)kv_width;
    (void)start_pos; (void)window; (void)n_tok; (void)ring;
    (void)qk_scale; (void)valid_span; (void)blocks; (void)page_table;
#endif
}

// Uniform Stream-K fixup.  Reverse traversal and the contracted side of each
// update are explicit because they are part of the numerical route identity.
__global__ void combine(
        const __half * numerators, const float * meta, float * out,
        uint32_t block_base, uint32_t n_heads, uint32_t n_kv,
        uint32_t n_tok, uint32_t blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t local_head = blockIdx.x;
    const uint32_t kvh = block_base + local_head;
    if (kvh >= n_kv || blocks == 0) return;
    for (uint32_t element = threadIdx.x;
         element < kColumns * kHeadDim; element += blockDim.x) {
        const uint32_t column = element / kHeadDim;
        const uint32_t output = element % kHeadDim;
        const uint32_t last = blocks - 1;
        float max_value = meta[meta_index(
            local_head, blocks, last, column, 0)];
        float rowsum = meta[meta_index(
            local_head, blocks, last, column, 1)];
        float numerator = __half2float(numerators[numerator_index(
            local_head, blocks, last, column, output)]);
        for (int32_t block = int32_t(last) - 1; block >= 0; --block) {
            const float add_max = meta[meta_index(
                local_head, blocks, uint32_t(block), column, 0)];
            const float add_sum = meta[meta_index(
                local_head, blocks, uint32_t(block), column, 1)];
            const float add_numerator = __half2float(numerators[numerator_index(
                local_head, blocks, uint32_t(block), column, output)]);
            const float max_new = fmaxf(max_value, add_max);
            const float diff_value = max_value - max_new;
            const float diff_add = add_max - max_new;
            const float scale_value = diff_value >= -20.0f
                ? expf(diff_value) : 0.0f;
            const float scale_add = diff_add >= -20.0f
                ? expf(diff_add) : 0.0f;
            numerator = fmaf(
                scale_value, numerator, scale_add * add_numerator);
            rowsum = fmaf(scale_value, rowsum, scale_add * add_sum);
            max_value = max_new;
        }
        const uint32_t token = column / kGqaHeads;
        const uint32_t head = kvh * kGqaHeads + column % kGqaHeads;
        if (token < n_tok && head < n_heads) {
            out[(uint64_t(token) * n_heads + head) * kHeadDim + output] =
                rowsum > 0.0f ? numerator / rowsum : 0.0f;
        }
    }
#else
    (void)numerators; (void)meta; (void)out; (void)block_base;
    (void)n_heads; (void)n_kv; (void)n_tok; (void)blocks;
#endif
}

} // namespace imparo_sm80_d64_wide
