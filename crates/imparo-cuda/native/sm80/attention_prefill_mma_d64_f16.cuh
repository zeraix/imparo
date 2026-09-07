#pragma once

// D64/GQA4 whole-K prefill for SM80+ Tensor Cores.
//
// One CTA owns one (16 query-token, 4 GQA-head, KV-head) tile and walks every
// 64-key update in source order.  The online max/rowsum and half PV accumulator
// therefore never cross a CTA seam.  This is the numerical distinction from the
// bounded Stream-K fallback in attention_prefill_d64_wide_f16.cuh.
namespace imparo_sm80_d64_mma {

using imparo_sm80_d64_mma_plan::kGqaHeads;
using imparo_sm80_d64_mma_plan::kHeadDim;
using imparo_sm80_d64_mma_plan::kKeysPerUpdate;
using imparo_sm80_d64_mma_plan::kQueryTokens;
using imparo_sm80_d64_mma_plan::kThreads;

constexpr uint32_t kWarps = 4;
constexpr uint32_t kColumns = kQueryTokens * kGqaHeads;
constexpr uint32_t kKeyTiles = kKeysPerUpdate / 16;
constexpr uint32_t kOutputTiles = kHeadDim / 16;
constexpr float kMaxOffset = 3.0f * 0.6931f;
constexpr float kSoftmaxFtzThreshold = -20.0f;

static_assert(kColumns == 64);
static_assert(kKeyTiles == 4);
static_assert(kWarps * 32 == kThreads);

template <bool SharedKv>
__launch_bounds__(kThreads, 2) __global__ void whole_k_tile(
        const float * q, const __half * kc, const __half * vc, float * out,
        uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, uint32_t window, uint32_t n_tok, uint32_t ring,
        float applied_q_scale, uint32_t valid_span, uint32_t key_updates,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t query_tile = blockIdx.x / n_kv;
    const uint32_t kvh = blockIdx.x - query_tile * n_kv;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t token_base = query_tile * kQueryTokens;
    constexpr uint32_t stride = 12;
    constexpr uint32_t per_warp_k_half2 = kWarps * 16 * stride;
    constexpr uint32_t shared_kv_half2 = kKeysPerUpdate * (kHeadDim / 2);
    constexpr uint32_t kv_storage_half2 =
        SharedKv ? shared_kv_half2 : per_warp_k_half2;
    __shared__ __align__(16) __half2 q_tile[kWarps][16 * stride];
    // K and V lifetimes do not overlap. One storage serves either the established
    // per-warp staging or the cooperative 64x64 CTA tile.
    __shared__ __align__(16) __half2 kv_tile[kv_storage_half2];

    // The workflow already applied the semantic Q scale in f32.  Reconstructing
    // raw Q is exact for LFM2's power-of-two 1/8 scale, then this half conversion
    // and half multiply preserve the pinned MMA-F16 operation order.
    const float inverse_q_scale = 1.0f / applied_q_scale;
    const __half2 q_scale_h2 = __float2half2_rn(applied_q_scale);
    imparo_sm80_mma::Half16x8 q_fragment[kOutputTiles];
#pragma unroll
    for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
        for (uint32_t e = lane; e < 16 * 8; e += 32) {
            const uint32_t row = e >> 3;
            const uint32_t pair = e & 7;
            const uint32_t column = warp * 16 + row;
            const uint32_t token = token_base + column / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + column % kGqaHeads;
            __half2 value = __float2half2_rn(0.0f);
            if (token < n_tok && head < n_heads) {
                const float * qr = q
                    + (uint64_t(token) * n_heads + head) * kHeadDim
                    + d0 + 2 * pair;
                value = __hmul2(
                    __floats2half2_rn(
                        qr[0] * inverse_q_scale,
                        qr[1] * inverse_q_scale),
                    q_scale_h2);
            }
            q_tile[warp][row * stride + pair] = value;
        }
        __syncwarp();
        imparo_sm80_mma::load_half16x8(
            q_fragment[d0 / 16], q_tile[warp], stride, lane);
        __syncwarp();
    }

    float row_max[2] = {-1.701411733e+38F, -1.701411733e+38F};
    float row_sum[2] = {0.0f, 0.0f};
    imparo_sm80_mma::Half16x8 numerator[kOutputTiles];
#pragma unroll
    for (uint32_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            numerator[output_tile].x[l] = __float2half2_rn(0.0f);
        }
    }

    for (uint32_t key_update = 0; key_update < key_updates; ++key_update) {
        imparo_sm80_mma::Float16x16 score[kKeyTiles]{};
        if constexpr (SharedKv) {
            // All four query warps consume the same K rows. Stage the complete
            // 64-key x D64 update once instead of issuing four identical global
            // reads, while retaining the same per-warp ldmatrix/MMA order.
            for (uint32_t e = threadIdx.x; e < shared_kv_half2;
                 e += kThreads) {
                const uint32_t key = key_update * kKeysPerUpdate
                    + e / (kHeadDim / 2);
                const uint32_t pair = e % (kHeadDim / 2);
                __half2 value = __float2half2_rn(0.0f);
                if (key < valid_span) {
                    const uint32_t physical =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    value = *reinterpret_cast<const __half2 *>(
                        kc + uint64_t(physical) * kv_width + kvh * kHeadDim
                            + 2 * pair);
                }
                kv_tile[e] = value;
            }
            __syncthreads();
#pragma unroll
            for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
#pragma unroll
                for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
                    imparo_sm80_mma::Half16x8 k_fragment;
                    imparo_sm80_mma::load_half16x8(
                        k_fragment,
                        kv_tile + tile * 16 * (kHeadDim / 2) + d0 / 2,
                        kHeadDim / 2, lane);
                    imparo_sm80_mma::mma_qk(
                        score[tile], q_fragment[d0 / 16], k_fragment);
                }
            }
            // No warp may overwrite the common tile with V while another warp
            // is still issuing its final K ldmatrix.
            __syncthreads();
        } else {
            __half2 * k_tile = kv_tile + warp * 16 * stride;
#pragma unroll
            for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
#pragma unroll
                for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
                    const uint32_t key0 =
                        key_update * kKeysPerUpdate + tile * 16;
                    for (uint32_t e = lane; e < 16 * 8; e += 32) {
                        const uint32_t row = e >> 3;
                        const uint32_t pair = e & 7;
                        const uint32_t key = key0 + row;
                        __half2 value = __float2half2_rn(0.0f);
                        if (key < valid_span) {
                            const uint32_t physical =
                                imparo_cuda_kv::physical_row(
                                    key, ring, page_table);
                            value = *reinterpret_cast<const __half2 *>(
                                kc + uint64_t(physical) * kv_width
                                    + kvh * kHeadDim + d0 + 2 * pair);
                        }
                        k_tile[row * stride + pair] = value;
                    }
                    __syncwarp();
                    imparo_sm80_mma::Half16x8 k_fragment;
                    imparo_sm80_mma::load_half16x8(
                        k_fragment, k_tile, stride, lane);
                    imparo_sm80_mma::mma_qk(
                        score[tile], q_fragment[d0 / 16], k_fragment);
                    __syncwarp();
                }
            }
        }

        float max_new[2] = {row_max[0], row_max[1]};
#pragma unroll
        for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 8; ++l) {
                const uint32_t local_column =
                    imparo_sm80_mma::fragment_q_column(lane, l);
                const uint32_t column = warp * 16 + local_column;
                const uint32_t token = token_base + column / kGqaHeads;
                const uint32_t pos = start_pos + token;
                const uint32_t lo = window > 0 && pos + 1 > window
                    ? pos + 1 - window : 0;
                const uint32_t key = key_update * kKeysPerUpdate + tile * 16
                    + imparo_sm80_mma::fragment_key_row(lane, l);
                const uint32_t key_pos = key < valid_span
                    ? imparo_sm80_prefill::physical_key_position(
                        key, ring, start_pos + n_tok - 1)
                    : 0;
                const bool valid = token < n_tok && key < valid_span
                    && key_pos >= lo && key_pos <= pos;
                if (valid) {
                    const uint32_t owned_column = (l / 2) % 2;
                    max_new[owned_column] = fmaxf(
                        max_new[owned_column], score[tile].x[l] + kMaxOffset);
                }
            }
        }
#pragma unroll
        for (uint32_t owned_column = 0; owned_column < 2; ++owned_column) {
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                max_new[owned_column] = fmaxf(
                    max_new[owned_column],
                    __shfl_xor_sync(0xffffffffu, max_new[owned_column], offset));
            }
        }

        float sum_add[2] = {0.0f, 0.0f};
#pragma unroll
        for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 8; ++l) {
                const uint32_t local_column =
                    imparo_sm80_mma::fragment_q_column(lane, l);
                const uint32_t column = warp * 16 + local_column;
                const uint32_t token = token_base + column / kGqaHeads;
                const uint32_t pos = start_pos + token;
                const uint32_t lo = window > 0 && pos + 1 > window
                    ? pos + 1 - window : 0;
                const uint32_t key = key_update * kKeysPerUpdate + tile * 16
                    + imparo_sm80_mma::fragment_key_row(lane, l);
                const uint32_t key_pos = key < valid_span
                    ? imparo_sm80_prefill::physical_key_position(
                        key, ring, start_pos + n_tok - 1)
                    : 0;
                const bool valid = token < n_tok && key < valid_span
                    && key_pos >= lo && key_pos <= pos;
                const uint32_t owned_column = (l / 2) % 2;
                const float probability = valid
                    ? expf(score[tile].x[l] - max_new[owned_column]) : 0.0f;
                score[tile].x[l] = probability;
                sum_add[owned_column] += probability;
            }
        }
        float previous_scale[2];
#pragma unroll
        for (uint32_t owned_column = 0; owned_column < 2; ++owned_column) {
            const float diff = row_max[owned_column] - max_new[owned_column];
            previous_scale[owned_column] = diff >= kSoftmaxFtzThreshold
                ? expf(diff) : 0.0f;
            row_sum[owned_column] = previous_scale[owned_column]
                * row_sum[owned_column] + sum_add[owned_column];
            row_max[owned_column] = max_new[owned_column];
        }
#pragma unroll
        for (uint32_t output_tile = 0;
             output_tile < kOutputTiles; ++output_tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                numerator[output_tile].x[l] = __hmul2(
                    numerator[output_tile].x[l],
                    __float2half2_rn(previous_scale[l % 2]));
            }
        }
        __syncwarp();
        if constexpr (SharedKv) {
            // Reuse the same shared allocation for V. The cooperative load is
            // coalesced across the 64 rows and reused by every query warp.
            for (uint32_t e = threadIdx.x; e < shared_kv_half2;
                 e += kThreads) {
                const uint32_t key = key_update * kKeysPerUpdate
                    + e / (kHeadDim / 2);
                const uint32_t pair = e % (kHeadDim / 2);
                __half2 value = __float2half2_rn(0.0f);
                if (key < valid_span) {
                    const uint32_t physical =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    value = *reinterpret_cast<const __half2 *>(
                        vc + uint64_t(physical) * kv_width + kvh * kHeadDim
                            + 2 * pair);
                }
                kv_tile[e] = value;
            }
            __syncthreads();
        }

#pragma unroll
        for (uint32_t tile = 0; tile < kKeyTiles; ++tile) {
            const uint32_t key0 = key_update * kKeysPerUpdate + tile * 16;
            // Upstream converts the FP32 KQ fragment to half registers directly;
            // a logical shared-memory round trip has the same coordinates but a
            // different physical conversion path.
            imparo_sm80_mma::Half16x8 probability_fragment;
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                probability_fragment.x[l] = __floats2half2_rn(
                    score[tile].x[2 * l], score[tile].x[2 * l + 1]);
            }
#pragma unroll
            for (uint32_t output_tile = 0;
                 output_tile < kOutputTiles; ++output_tile) {
                imparo_sm80_mma::Half16x8 value_fragment;
                if constexpr (SharedKv) {
                    imparo_sm80_mma::load_half16x8_trans(
                        value_fragment,
                        kv_tile + tile * 16 * (kHeadDim / 2)
                            + output_tile * 8,
                        kHeadDim / 2, lane);
                } else {
                    __half2 * value_tile = kv_tile + warp * 16 * 8;
                    for (uint32_t e = lane; e < 16 * 8; e += 32) {
                        const uint32_t key = key0 + (e >> 3);
                        const uint32_t pair = e & 7;
                        __half2 value = __float2half2_rn(0.0f);
                        if (key < valid_span) {
                            const uint32_t physical =
                                imparo_cuda_kv::physical_row(
                                    key, ring, page_table);
                            value = *reinterpret_cast<const __half2 *>(
                                vc + uint64_t(physical) * kv_width
                                    + kvh * kHeadDim + output_tile * 16
                                    + 2 * pair);
                        }
                        value_tile[(e >> 3) * 8 + pair] = value;
                    }
                    __syncwarp();
                    imparo_sm80_mma::load_half16x8_trans(
                        value_fragment, value_tile, 8, lane);
                }
                imparo_sm80_mma::mma_pv(
                    numerator[output_tile], probability_fragment, value_fragment);
                if constexpr (!SharedKv) __syncwarp();
            }
        }
        if constexpr (SharedKv) __syncthreads();
    }

    // The pinned MMA route keeps denominator partials lane-local across every
    // 64-key online update and applies one XOR tree only after the whole K walk.
    // Reducing each update first is algebraically equivalent but changes f32
    // association at long context.
    float denominator[2] = {row_sum[0], row_sum[1]};
#pragma unroll
    for (uint32_t owned_column = 0; owned_column < 2; ++owned_column) {
#pragma unroll
        for (int offset = 2; offset >= 1; offset >>= 1) {
            denominator[owned_column] += __shfl_xor_sync(
                0xffffffffu, denominator[owned_column], offset);
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
            const uint32_t token = token_base + column / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + column % kGqaHeads;
            const uint32_t output_pair =
                imparo_sm80_mma::half_acc_output_pair(lane, l);
            const uint32_t output = output_tile * 16 + 2 * output_pair;
            const float row_denominator = denominator[l % 2];
            if (token < n_tok && head < n_heads) {
                float * dst = out
                    + (uint64_t(token) * n_heads + head) * kHeadDim + output;
                dst[0] = row_denominator > 0.0f
                    ? __half2float(__low2half(numerator[output_tile].x[l]))
                        / row_denominator
                    : 0.0f;
                dst[1] = row_denominator > 0.0f
                    ? __half2float(__high2half(numerator[output_tile].x[l]))
                        / row_denominator
                    : 0.0f;
            }
        }
    }
#else
    (void)q; (void)kc; (void)vc; (void)out; (void)n_heads; (void)n_kv;
    (void)kv_width; (void)start_pos; (void)window; (void)n_tok; (void)ring;
    (void)applied_q_scale; (void)valid_span; (void)key_updates; (void)page_table;
#endif
}

} // namespace imparo_sm80_d64_mma
