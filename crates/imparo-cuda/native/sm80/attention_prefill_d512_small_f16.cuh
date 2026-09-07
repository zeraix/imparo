#pragma once

// Ampere D=64/D=256/D=512 Flash Attention for a tail of up to four query tokens.
// chunking. The 16 query/head columns map to two cooperating warps. Key groups
// are divided into topology-sized stream-K partials, evaluated online in each
// partial, then combined from the last partial to the first.
//
// The fragment ABI and lane mapping live in attention_prefill_d512_f16.cuh and
// are intentionally reused here. This file owns only the small-query schedule.
namespace imparo_sm80_d512_small {

constexpr uint32_t kQueryTokens = 4;
constexpr uint32_t kGqaHeads = 4;
constexpr uint32_t kColumns = kQueryTokens * kGqaHeads;
constexpr uint32_t kKeyBatch = 32;
constexpr uint32_t kWarps = 2;

template <uint32_t CacheType>
__device__ __forceinline__ __half cache_half(
        const void * cache, uint32_t width, uint32_t slot, uint32_t index) {
    if constexpr (CacheType == 1) {
        return static_cast<const __half *>(cache)[uint64_t(slot) * width + index];
    } else {
        const uint32_t block = index / 32;
        const uint32_t lane = index & 31;
        const uint32_t blocks = width / 32;
        if constexpr (CacheType == 2) {
            const uint8_t * q = static_cast<const uint8_t *>(cache)
                + (uint64_t(slot) * blocks + block) * 18;
            const float d = __half2float(*reinterpret_cast<const __half *>(q));
            const uint8_t packed = q[2 + (lane & 15)];
            const int value = lane < 16 ? (packed & 0x0f) : (packed >> 4);
            return __float2half(fmaf(d, float(value), -8.0f * d));
        } else {
            const uint8_t * q = static_cast<const uint8_t *>(cache)
                + (uint64_t(slot) * blocks + block) * 34;
            const float d = __half2float(*reinterpret_cast<const __half *>(q));
            return __float2half(float(int8_t(q[2 + lane])) * d);
        }
    }
}

template <uint32_t CacheType>
__device__ __forceinline__ __half2 cache_half2(
        const void * cache, uint32_t width, uint32_t slot, uint32_t index) {
    // Every caller requests an aligned adjacent pair for an MMA fragment. Load
    // the shared quantization scale and block address once instead of expanding
    // two scalar cache_half calls. Each lane still rounds independently to f16.
    if constexpr (CacheType == 1) {
        const __half * values = static_cast<const __half *>(cache)
            + uint64_t(slot) * width + index;
        return *reinterpret_cast<const __half2 *>(values);
    } else {
        const uint32_t block = index / 32;
        const uint32_t lane = index & 31;
        const uint32_t blocks = width / 32;
        if constexpr (CacheType == 2) {
            const uint8_t * q = static_cast<const uint8_t *>(cache)
                + (uint64_t(slot) * blocks + block) * 18;
            const float d = __half2float(*reinterpret_cast<const __half *>(q));
            // index is even, so the adjacent packed bytes are naturally aligned.
            // One 16-bit transaction serves both output values.
            const uint16_t packed = *reinterpret_cast<const uint16_t *>(
                q + 2 + (lane & 15));
            const int value0 = lane < 16
                ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
            const int value1 = lane < 16
                ? ((packed >> 8) & 0x0f) : ((packed >> 12) & 0x0f);
            return __halves2half2(
                __float2half(fmaf(d, float(value0), -8.0f * d)),
                __float2half(fmaf(d, float(value1), -8.0f * d)));
        } else {
            const uint8_t * q = static_cast<const uint8_t *>(cache)
                + (uint64_t(slot) * blocks + block) * 34;
            const float d = __half2float(*reinterpret_cast<const __half *>(q));
            return __halves2half2(
                __float2half(float(int8_t(q[2 + lane])) * d),
                __float2half(float(int8_t(q[3 + lane])) * d));
        }
    }
}

__device__ __forceinline__ __half2 cache_half2_q4_scaled(
        const void * cache, uint32_t width, uint32_t slot, uint32_t index,
        __half scale) {
    const uint32_t block = index / 32;
    const uint32_t lane = index & 31;
    const uint32_t blocks = width / 32;
    const uint8_t * q = static_cast<const uint8_t *>(cache)
        + (uint64_t(slot) * blocks + block) * 18;
    const uint16_t packed = *reinterpret_cast<const uint16_t *>(
        q + 2 + (lane & 15));
    const int value0 = lane < 16
        ? (packed & 0x0f) : ((packed >> 4) & 0x0f);
    const int value1 = lane < 16
        ? ((packed >> 8) & 0x0f) : ((packed >> 12) & 0x0f);
    const float d = __half2float(scale);
    return __halves2half2(
        __float2half(fmaf(d, float(value0), -8.0f * d)),
        __float2half(fmaf(d, float(value1), -8.0f * d)));
}

__device__ __forceinline__ uint64_t block_stride(
        uint32_t kv_span, uint32_t parts) {
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    return uint64_t(kColumns) * kv_span
        + uint64_t(2 * kColumns) * groups
        + uint64_t(4 * kColumns) * parts;
}

__device__ __forceinline__ uint64_t rescale_index(
        uint32_t kv_span, uint32_t partition, uint32_t column,
        uint32_t group) {
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    return uint64_t(kColumns) * kv_span
        + (uint64_t(partition) * kColumns + column) * groups + group;
}

__device__ __forceinline__ uint64_t meta_index(
        uint32_t kv_span, uint32_t parts, uint32_t partition,
        uint32_t column, uint32_t part, uint32_t field) {
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    return uint64_t(kColumns) * kv_span + uint64_t(2 * kColumns) * groups
        + ((uint64_t(partition) * kColumns + column) * parts + part) * 2
        + field;
}

// Two warps cover the two 16-key halves for all 16 query/head columns. The
// small reference configuration stages the complete D=512 K dimension at once,
// so the MMA accumulator walks dimensions 0..511 in increasing order.
template <uint32_t HeadDim, uint32_t CacheType>
__global__ void scores(
        const float * q, const void * kc, float * workspace,
        uint32_t block_base, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos, float qk_scale, uint32_t window,
        uint32_t n_tok, uint32_t ring, uint32_t valid_span,
        uint32_t kv_span, uint32_t parts, const uint32_t * decode_control) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        start_pos = decode_control[0];
        valid_span = ring ? min(start_pos + 1, ring + 1) : start_pos + 1;
    }
    const uint32_t local_block = blockIdx.x;
    const uint32_t kvh = block_base + local_block;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp;
    const uint32_t key0 = blockIdx.y * kKeyBatch + partition * 16;
    constexpr uint32_t stride = 12;
    __shared__ __align__(16) __half2 q_tile[kWarps][16 * stride];
    __shared__ __align__(16) __half2 k_tile[kWarps][16 * stride];

    imparo_sm80_prefill::Float16x16 c{};
    static_assert(HeadDim == 64 || HeadDim == 256 || HeadDim == 512,
        "small FA head dimension");
    for (uint32_t d0 = 0; d0 < HeadDim; d0 += 16) {
        for (uint32_t e = lane; e < 16 * 8; e += 32) {
            const uint32_t qrow = e >> 3;
            const uint32_t pair = e & 7;
            const uint32_t token = qrow / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + qrow % kGqaHeads;
            __half2 qv = __float2half2_rn(0.0f);
            if (token < n_tok && head < n_heads) {
                const float * qr = q + ((uint64_t)token * n_heads + head) * HeadDim
                    + d0 + 2 * pair;
                qv = __hmul2(__floats2half2_rn(qr[0], qr[1]),
                              __float2half2_rn(qk_scale));
            }
            q_tile[warp][qrow * stride + pair] = qv;

            const uint32_t key = key0 + qrow;
            __half2 kval = __float2half2_rn(0.0f);
            if (key < valid_span) {
                kval = cache_half2<CacheType>(
                    kc, kv_width, key,
                    kvh * HeadDim + d0 + 2 * pair);
            }
            k_tile[warp][qrow * stride + pair] = kval;
        }
        __syncwarp();
        imparo_sm80_prefill::Half16x8 q_frag;
        imparo_sm80_prefill::Half16x8 k_frag;
        imparo_sm80_prefill::load_half16x8(q_frag, q_tile[warp], stride, lane);
        imparo_sm80_prefill::load_half16x8(k_frag, k_tile[warp], stride, lane);
        imparo_sm80_prefill::mma_qk(c, q_frag, k_frag);
        __syncwarp();
    }

#pragma unroll
    for (uint32_t l = 0; l < 8; ++l) {
        const uint32_t column = imparo_sm80_prefill::fragment_q_column(lane, l);
        const uint32_t key = key0 + imparo_sm80_prefill::fragment_key_row(lane, l);
        const uint32_t token = column / kGqaHeads;
        const uint32_t pos = start_pos + token;
        const uint32_t lo = window > 0 && pos + 1 > window ? pos + 1 - window : 0;
        const uint32_t key_pos = key < valid_span
            ? imparo_sm80_prefill::physical_key_position(
                key, ring, start_pos + n_tok - 1) : 0;
        float value = -3.402823466e+38F;
        if (token < n_tok && key < valid_span && key_pos >= lo && key_pos <= pos) {
            value = c.x[l];
        }
        if (key < kv_span) {
            workspace[uint64_t(local_block) * block_stride(kv_span, parts)
                + uint64_t(column) * kv_span + key] = value;
        }
    }
#else
    (void)q; (void)kc; (void)workspace; (void)block_base; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos; (void)qk_scale; (void)window; (void)n_tok;
    (void)ring; (void)valid_span; (void)kv_span; (void)parts; (void)decode_control;
#endif
}

template <uint32_t HeadDim, uint32_t CacheType>
__global__ void scores_paged(
        const float * q, const void * kc, float * workspace,
        uint32_t block_base, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t start_pos, float qk_scale, uint32_t window,
        uint32_t n_tok, uint32_t ring, uint32_t valid_span,
        uint32_t kv_span, uint32_t parts, const uint32_t * decode_control,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        start_pos = decode_control[0];
        valid_span = ring ? min(start_pos + 1, ring + 1) : start_pos + 1;
    }
    const uint32_t local_block = blockIdx.x;
    const uint32_t kvh = block_base + local_block;
    if (kvh >= n_kv) return;

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp;
    const uint32_t key0 = blockIdx.y * kKeyBatch + partition * 16;
    constexpr uint32_t stride = 12;
    __shared__ __align__(16) __half2 q_tile[kWarps][16 * stride];
    __shared__ __align__(16) __half2 k_tile[kWarps][16 * stride];

    imparo_sm80_prefill::Float16x16 c{};
    static_assert(HeadDim == 64 || HeadDim == 256 || HeadDim == 512,
        "small FA head dimension");
    for (uint32_t d0 = 0; d0 < HeadDim; d0 += 16) {
        for (uint32_t e = lane; e < 16 * 8; e += 32) {
            const uint32_t qrow = e >> 3;
            const uint32_t pair = e & 7;
            const uint32_t token = qrow / kGqaHeads;
            const uint32_t head = kvh * kGqaHeads + qrow % kGqaHeads;
            __half2 qv = __float2half2_rn(0.0f);
            if (token < n_tok && head < n_heads) {
                const float * qr = q + ((uint64_t)token * n_heads + head) * HeadDim
                    + d0 + 2 * pair;
                qv = __hmul2(__floats2half2_rn(qr[0], qr[1]),
                              __float2half2_rn(qk_scale));
            }
            q_tile[warp][qrow * stride + pair] = qv;

            const uint32_t key = key0 + qrow;
            __half2 kval = __float2half2_rn(0.0f);
            if (key < valid_span) {
                const uint32_t physical_key =
                    imparo_cuda_kv::physical_row(key, ring, page_table);
                kval = cache_half2<CacheType>(
                    kc, kv_width, physical_key,
                    kvh * HeadDim + d0 + 2 * pair);
            }
            k_tile[warp][qrow * stride + pair] = kval;
        }
        __syncwarp();
        imparo_sm80_prefill::Half16x8 q_frag;
        imparo_sm80_prefill::Half16x8 k_frag;
        imparo_sm80_prefill::load_half16x8(q_frag, q_tile[warp], stride, lane);
        imparo_sm80_prefill::load_half16x8(k_frag, k_tile[warp], stride, lane);
        imparo_sm80_prefill::mma_qk(c, q_frag, k_frag);
        __syncwarp();
    }

#pragma unroll
    for (uint32_t l = 0; l < 8; ++l) {
        const uint32_t column = imparo_sm80_prefill::fragment_q_column(lane, l);
        const uint32_t key = key0 + imparo_sm80_prefill::fragment_key_row(lane, l);
        const uint32_t token = column / kGqaHeads;
        const uint32_t pos = start_pos + token;
        const uint32_t lo = window > 0 && pos + 1 > window ? pos + 1 - window : 0;
        const uint32_t key_pos = key < valid_span
            ? imparo_sm80_prefill::physical_key_position(
                key, ring, start_pos + n_tok - 1) : 0;
        float value = -3.402823466e+38F;
        if (token < n_tok && key < valid_span && key_pos >= lo && key_pos <= pos) {
            value = c.x[l];
        }
        if (key < kv_span) {
            workspace[uint64_t(local_block) * block_stride(kv_span, parts)
                + uint64_t(column) * kv_span + key] = value;
        }
    }
#else
    (void)q; (void)kc; (void)workspace; (void)block_base; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos; (void)qk_scale; (void)window; (void)n_tok;
    (void)ring; (void)valid_span; (void)kv_span; (void)parts; (void)decode_control;
    (void)page_table;
#endif
}

// Each block owns one (partial, query column, 16-key warp partition). It scans
// that partial's key groups online, recording the half-accumulator rescale used
// before every subsequent PV MMA.
__global__ void softmax_parts(
        float * workspace, uint32_t kv_span, uint32_t parts) {
    const uint32_t local_block = blockIdx.x;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t flat = blockIdx.y * (blockDim.x / 32) + warp;
    if (flat >= 2 * kColumns * parts) return;
    const uint32_t part = flat % parts;
    const uint32_t pc = flat / parts;
    const uint32_t partition = pc & 1;
    const uint32_t column = pc >> 1;
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    const uint32_t group_start = uint64_t(part) * groups / parts;
    const uint32_t group_stop = uint64_t(part + 1) * groups / parts;
    const uint64_t base = uint64_t(local_block) * block_stride(kv_span, parts);
    float * row = workspace + base + uint64_t(column) * kv_span;
    constexpr float max_offset = 3.0f * 0.6931f;
    float running_max = -3.402823466e+38F;
    float partial_sum = 0.0f;

    for (uint32_t group = group_start; group < group_stop; ++group) {
        float next_max = running_max;
        if (lane < 4) {
            for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t key = group * kKeyBatch + partition * 16
                        + half8 + 2 * lane + pair;
                    if (key < kv_span && row[key] > -3.0e38F) {
                        next_max = fmaxf(next_max, row[key] + max_offset);
                    }
                }
            }
#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                next_max = fmaxf(next_max,
                    __shfl_xor_sync(0x0000000f, next_max, offset));
            }
        }
        next_max = __shfl_sync(0xffffffff, next_max, 0);
        const float diff = running_max - next_max;
        float rescale = expf(diff);
        if (diff < -20.0f) rescale = 0.0f;
        if (lane == 0) {
            workspace[base + rescale_index(
                kv_span, partition, column, group)] = rescale;
        }

        if (lane < 4) {
            float add = 0.0f;
            for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t key = group * kKeyBatch + partition * 16
                        + half8 + 2 * lane + pair;
                    if (key < kv_span && row[key] > -3.0e38F) {
                        const float probability = expf(row[key] - next_max);
                        row[key] = probability;
                        add += probability;
                    } else if (key < kv_span) {
                        row[key] = 0.0f;
                    }
                }
            }
            partial_sum = partial_sum * rescale + add;
        }
        running_max = next_max;
        __syncwarp();
    }

    if (lane < 4) {
        float sum = partial_sum;
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0x0000000f, sum, offset);
        }
        if (lane == 0) {
            workspace[base + meta_index(
                kv_span, parts, partition, column, part, 0)] = running_max;
            workspace[base + meta_index(
                kv_span, parts, partition, column, part, 1)] = sum;
        }
    }
}

// Compute one half-MMA numerator per stream-K partial, combine its two warp
// partitions, then reproduce the reference uniform fixup from the last partial
// to the first. Each thread in warp 0 owns eight (query, output-row) cells.
template <uint32_t HeadDim, uint32_t CacheType, uint32_t OutputTiles>
__global__ void values_combine(
        const void * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t n_tok, uint32_t ring,
        uint32_t valid_span, uint32_t kv_span, uint32_t parts,
        const uint32_t * decode_control) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        const uint32_t start_pos = decode_control[0];
        valid_span = ring ? min(start_pos + 1, ring + 1) : start_pos + 1;
    }
    static_assert(HeadDim == 64 || HeadDim == 256 || HeadDim == 512,
        "small FA head dimension");
    static_assert(OutputTiles == 1 || OutputTiles == 2,
                  "small FA value output tiles");
    const uint32_t local_block = blockIdx.x;
    const uint32_t kvh = block_base + local_block;
    if (kvh >= n_kv) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp;
    const uint32_t out0 = blockIdx.y * (16 * OutputTiles);
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    const uint64_t base = uint64_t(local_block) * block_stride(kv_span, parts);
    __shared__ __align__(16) __half2 p_tile[kWarps][16 * 8];
    __shared__ __align__(16) __half2 v_tile[kWarps][16 * 8];
    __shared__ __align__(16) __half c_tile[kWarps][OutputTiles * 16 * 16];
    // Rescale and partial-softmax metadata are column properties, but every
    // output row consumes them. Stage each value once instead of issuing the
    // same global workspace load for every MMA accumulator cell.
    __shared__ float rescale_tile[kWarps][kColumns];
    __shared__ float meta_tile[kWarps][kColumns][2];
    __shared__ __half v_scale[kWarps][16];
    float accum_num[OutputTiles][8]{};
    float accum_sum[OutputTiles][8]{};
    float accum_max[OutputTiles][8]{};

    for (int32_t part = int32_t(parts) - 1; part >= 0; --part) {
        imparo_sm80_prefill::Half16x8 c[OutputTiles];
#pragma unroll
        for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                c[tile].x[l] = __float2half2_rn(0.0f);
            }
        }
        const uint32_t group_start = uint64_t(uint32_t(part)) * groups / parts;
        const uint32_t group_stop = uint64_t(uint32_t(part) + 1) * groups / parts;
        if constexpr (CacheType == 2) {
            if (lane < 16) {
                const uint32_t key = group_start * kKeyBatch
                    + partition * 16 + lane;
                __half scale = __float2half(0.0f);
                if (key < valid_span) {
                    const uint32_t index = kvh * HeadDim + out0;
                    const uint32_t block = index / 32;
                    const uint32_t blocks = kv_width / 32;
                    const uint8_t * q = static_cast<const uint8_t *>(vc)
                        + (uint64_t(key) * blocks + block) * 18;
                    scale = *reinterpret_cast<const __half *>(q);
                }
                v_scale[warp][lane] = scale;
            }
            __syncwarp();
        }
        for (uint32_t group = group_start; group < group_stop; ++group) {
            if (group > group_start) {
                if (lane < kColumns) {
                    rescale_tile[partition][lane] = workspace[
                        base + rescale_index(
                            kv_span, partition, lane, group)];
                }
                __syncwarp();
#pragma unroll
                for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
                    for (uint32_t l = 0; l < 4; ++l) {
                        const uint32_t column =
                            imparo_sm80_prefill::half_acc_query_column(lane, l);
                        const float scale = rescale_tile[partition][column];
                        c[tile].x[l] = __hmul2(
                            c[tile].x[l], __float2half2_rn(scale));
                    }
                }
            }
            const uint32_t key0 = group * kKeyBatch + partition * 16;
            for (uint32_t e = lane; e < 16 * 8; e += 32) {
                const uint32_t query = e >> 3;
                const uint32_t pair = e & 7;
                const uint32_t pkey = key0 + 2 * pair;
                const float p0 = pkey < kv_span
                    ? workspace[base + uint64_t(query) * kv_span + pkey] : 0.0f;
                const float p1 = pkey + 1 < kv_span
                    ? workspace[base + uint64_t(query) * kv_span + pkey + 1] : 0.0f;
                p_tile[warp][query * 8 + pair] = __floats2half2_rn(p0, p1);
            }
#pragma unroll
            for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t key = key0 + (e >> 3);
                    const uint32_t out_pair = e & 7;
                    __half2 value = __float2half2_rn(0.0f);
                    if (key < valid_span) {
                        const uint32_t index = kvh * HeadDim + out0
                            + tile * 16 + 2 * out_pair;
                        if constexpr (CacheType == 2) {
                            value = cache_half2_q4_scaled(
                                vc, kv_width, key, index,
                                v_scale[warp][e >> 3]);
                        } else {
                            value = cache_half2<CacheType>(
                                vc, kv_width, key, index);
                        }
                    }
                    v_tile[warp][(e >> 3) * 8 + out_pair] = value;
                }
                __syncwarp();
                imparo_sm80_prefill::Half16x8 probability;
                imparo_sm80_prefill::Half16x8 value;
                imparo_sm80_prefill::load_half16x8(
                    probability, p_tile[warp], 8, lane);
                imparo_sm80_prefill::load_half16x8_trans(
                    value, v_tile[warp], 8, lane);
                imparo_sm80_prefill::mma_pv(c[tile], probability, value);
                if constexpr (OutputTiles > 1) {
                    if (tile + 1 < OutputTiles) __syncwarp();
                }
            }
            if constexpr (CacheType == 2) {
                if (group + 1 < group_stop && lane < 16) {
                    const uint32_t key = (group + 1) * kKeyBatch
                        + partition * 16 + lane;
                    __half scale = __float2half(0.0f);
                    if (key < valid_span) {
                        const uint32_t index = kvh * HeadDim + out0;
                        const uint32_t block = index / 32;
                        const uint32_t blocks = kv_width / 32;
                        const uint8_t * q = static_cast<const uint8_t *>(vc)
                            + (uint64_t(key) * blocks + block) * 18;
                        scale = *reinterpret_cast<const __half *>(q);
                    }
                    v_scale[warp][lane] = scale;
                }
            }
            __syncwarp();
        }

#pragma unroll
        for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t query =
                    imparo_sm80_prefill::half_acc_query_column(lane, l);
                const uint32_t out_pair =
                    imparo_sm80_prefill::half_acc_output_pair(lane, l);
                const uint32_t offset = tile * 16 * 16
                    + query * 16 + 2 * out_pair;
                c_tile[warp][offset] = __low2half(c[tile].x[l]);
                c_tile[warp][offset + 1] = __high2half(c[tile].x[l]);
            }
        }
        if (lane < kColumns) {
            meta_tile[partition][lane][0] = workspace[base + meta_index(
                kv_span, parts, partition, lane, uint32_t(part), 0)];
            meta_tile[partition][lane][1] = workspace[base + meta_index(
                kv_span, parts, partition, lane, uint32_t(part), 1)];
        }
        __syncthreads();

        if (partition == 0) {
#pragma unroll
            for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
                for (uint32_t slot = 0; slot < 8; ++slot) {
                    const uint32_t e = lane + slot * 32;
                    const uint32_t output_row = e & 15;
                    const uint32_t column = e >> 4;
                    const float max0 = meta_tile[0][column][0];
                    const float sum0 = meta_tile[0][column][1];
                    const float max1 = meta_tile[1][column][0];
                    const float sum1 = meta_tile[1][column][1];
                    const float part_max = fmaxf(max0, max1);
                    const float scale0 = expf(max0 - part_max);
                    const float scale1 = expf(max1 - part_max);
                    const uint32_t offset = tile * 16 * 16 + e;
                    float part_num = fmaf(
                        scale0, __half2float(c_tile[0][offset]), 0.0f);
                    part_num = fmaf(
                        scale1, __half2float(c_tile[1][offset]), part_num);
                    float part_sum = scale0 * sum0;
                    part_sum = fmaf(scale1, sum1, part_sum);

                    if (part == int32_t(parts) - 1) {
                        accum_num[tile][slot] = part_num;
                        accum_sum[tile][slot] = part_sum;
                        accum_max[tile][slot] = part_max;
                    } else {
                        const float max_new = fmaxf(
                            accum_max[tile][slot], part_max);
                        const float diff_value = accum_max[tile][slot] - max_new;
                        const float diff_add = part_max - max_new;
                        const float scale_value = diff_value >= -20.0f
                            ? expf(diff_value) : 0.0f;
                        const float scale_add = diff_add >= -20.0f
                            ? expf(diff_add) : 0.0f;
                        // Match the pinned stream-K fixup instruction order: round
                        // the addend product, then contract scale_value*accumulator.
                        accum_num[tile][slot] = fmaf(
                            scale_value, accum_num[tile][slot], scale_add * part_num);
                        // Keep the denominator on the same contracted side as the
                        // pinned uniform Stream-K fixup for both supported head sizes.
                        accum_sum[tile][slot] = fmaf(
                            scale_value, accum_sum[tile][slot], scale_add * part_sum);
                        accum_max[tile][slot] = max_new;
                    }
                    if (part == 0) {
                        const uint32_t token = column / kGqaHeads;
                        const uint32_t head = kvh * kGqaHeads + column % kGqaHeads;
                        if (token < n_tok && head < n_heads) {
                            out[((uint64_t)token * n_heads + head) * HeadDim
                                + out0 + tile * 16 + output_row]
                                = accum_sum[tile][slot] > 0.0f
                                ? accum_num[tile][slot] / accum_sum[tile][slot]
                                : 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)n_tok; (void)ring; (void)valid_span; (void)kv_span;
    (void)parts; (void)decode_control;
#endif
}

template <uint32_t HeadDim, uint32_t CacheType, uint32_t OutputTiles>
__global__ void values_combine_paged(
        const void * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t n_tok, uint32_t ring,
        uint32_t valid_span, uint32_t kv_span, uint32_t parts,
        const uint32_t * decode_control, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        const uint32_t start_pos = decode_control[0];
        valid_span = ring ? min(start_pos + 1, ring + 1) : start_pos + 1;
    }
    static_assert(HeadDim == 64 || HeadDim == 256 || HeadDim == 512,
        "small FA head dimension");
    static_assert(OutputTiles == 1 || OutputTiles == 2,
                  "small FA value output tiles");
    const uint32_t local_block = blockIdx.x;
    const uint32_t kvh = block_base + local_block;
    if (kvh >= n_kv) return;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t partition = warp;
    const uint32_t out0 = blockIdx.y * (16 * OutputTiles);
    const uint32_t groups = (kv_span + kKeyBatch - 1) / kKeyBatch;
    const uint64_t base = uint64_t(local_block) * block_stride(kv_span, parts);
    __shared__ __align__(16) __half2 p_tile[kWarps][16 * 8];
    __shared__ __align__(16) __half2 v_tile[kWarps][16 * 8];
    __shared__ __align__(16) __half c_tile[kWarps][OutputTiles * 16 * 16];
    // Rescale and partial-softmax metadata are column properties, but every
    // output row consumes them. Stage each value once instead of issuing the
    // same global workspace load for every MMA accumulator cell.
    __shared__ float rescale_tile[kWarps][kColumns];
    __shared__ float meta_tile[kWarps][kColumns][2];
    __shared__ __half v_scale[kWarps][16];
    float accum_num[OutputTiles][8]{};
    float accum_sum[OutputTiles][8]{};
    float accum_max[OutputTiles][8]{};

    for (int32_t part = int32_t(parts) - 1; part >= 0; --part) {
        imparo_sm80_prefill::Half16x8 c[OutputTiles];
#pragma unroll
        for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                c[tile].x[l] = __float2half2_rn(0.0f);
            }
        }
        const uint32_t group_start = uint64_t(uint32_t(part)) * groups / parts;
        const uint32_t group_stop = uint64_t(uint32_t(part) + 1) * groups / parts;
        if constexpr (CacheType == 2) {
            if (lane < 16) {
                const uint32_t key = group_start * kKeyBatch
                    + partition * 16 + lane;
                __half scale = __float2half(0.0f);
                if (key < valid_span) {
                    const uint32_t index = kvh * HeadDim + out0;
                    const uint32_t block = index / 32;
                    const uint32_t blocks = kv_width / 32;
                    const uint32_t physical_key =
                        imparo_cuda_kv::physical_row(key, ring, page_table);
                    const uint8_t * q = static_cast<const uint8_t *>(vc)
                        + (uint64_t(physical_key) * blocks + block) * 18;
                    scale = *reinterpret_cast<const __half *>(q);
                }
                v_scale[warp][lane] = scale;
            }
            __syncwarp();
        }
        for (uint32_t group = group_start; group < group_stop; ++group) {
            if (group > group_start) {
                if (lane < kColumns) {
                    rescale_tile[partition][lane] = workspace[
                        base + rescale_index(
                            kv_span, partition, lane, group)];
                }
                __syncwarp();
#pragma unroll
                for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
                    for (uint32_t l = 0; l < 4; ++l) {
                        const uint32_t column =
                            imparo_sm80_prefill::half_acc_query_column(lane, l);
                        const float scale = rescale_tile[partition][column];
                        c[tile].x[l] = __hmul2(
                            c[tile].x[l], __float2half2_rn(scale));
                    }
                }
            }
            const uint32_t key0 = group * kKeyBatch + partition * 16;
            for (uint32_t e = lane; e < 16 * 8; e += 32) {
                const uint32_t query = e >> 3;
                const uint32_t pair = e & 7;
                const uint32_t pkey = key0 + 2 * pair;
                const float p0 = pkey < kv_span
                    ? workspace[base + uint64_t(query) * kv_span + pkey] : 0.0f;
                const float p1 = pkey + 1 < kv_span
                    ? workspace[base + uint64_t(query) * kv_span + pkey + 1] : 0.0f;
                p_tile[warp][query * 8 + pair] = __floats2half2_rn(p0, p1);
            }
#pragma unroll
            for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t key = key0 + (e >> 3);
                    const uint32_t out_pair = e & 7;
                    __half2 value = __float2half2_rn(0.0f);
                    if (key < valid_span) {
                        const uint32_t physical_key =
                            imparo_cuda_kv::physical_row(key, ring, page_table);
                        const uint32_t index = kvh * HeadDim + out0
                            + tile * 16 + 2 * out_pair;
                        if constexpr (CacheType == 2) {
                            value = cache_half2_q4_scaled(
                                vc, kv_width, physical_key, index,
                                v_scale[warp][e >> 3]);
                        } else {
                            value = cache_half2<CacheType>(
                                vc, kv_width, physical_key, index);
                        }
                    }
                    v_tile[warp][(e >> 3) * 8 + out_pair] = value;
                }
                __syncwarp();
                imparo_sm80_prefill::Half16x8 probability;
                imparo_sm80_prefill::Half16x8 value;
                imparo_sm80_prefill::load_half16x8(
                    probability, p_tile[warp], 8, lane);
                imparo_sm80_prefill::load_half16x8_trans(
                    value, v_tile[warp], 8, lane);
                imparo_sm80_prefill::mma_pv(c[tile], probability, value);
                if constexpr (OutputTiles > 1) {
                    if (tile + 1 < OutputTiles) __syncwarp();
                }
            }
            if constexpr (CacheType == 2) {
                if (group + 1 < group_stop && lane < 16) {
                    const uint32_t key = (group + 1) * kKeyBatch
                        + partition * 16 + lane;
                    __half scale = __float2half(0.0f);
                    if (key < valid_span) {
                        const uint32_t index = kvh * HeadDim + out0;
                        const uint32_t block = index / 32;
                        const uint32_t blocks = kv_width / 32;
                        const uint32_t physical_key =
                            imparo_cuda_kv::physical_row(key, ring, page_table);
                        const uint8_t * q = static_cast<const uint8_t *>(vc)
                            + (uint64_t(physical_key) * blocks + block) * 18;
                        scale = *reinterpret_cast<const __half *>(q);
                    }
                    v_scale[warp][lane] = scale;
                }
            }
            __syncwarp();
        }

#pragma unroll
        for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t query =
                    imparo_sm80_prefill::half_acc_query_column(lane, l);
                const uint32_t out_pair =
                    imparo_sm80_prefill::half_acc_output_pair(lane, l);
                const uint32_t offset = tile * 16 * 16
                    + query * 16 + 2 * out_pair;
                c_tile[warp][offset] = __low2half(c[tile].x[l]);
                c_tile[warp][offset + 1] = __high2half(c[tile].x[l]);
            }
        }
        if (lane < kColumns) {
            meta_tile[partition][lane][0] = workspace[base + meta_index(
                kv_span, parts, partition, lane, uint32_t(part), 0)];
            meta_tile[partition][lane][1] = workspace[base + meta_index(
                kv_span, parts, partition, lane, uint32_t(part), 1)];
        }
        __syncthreads();

        if (partition == 0) {
#pragma unroll
            for (uint32_t tile = 0; tile < OutputTiles; ++tile) {
#pragma unroll
                for (uint32_t slot = 0; slot < 8; ++slot) {
                    const uint32_t e = lane + slot * 32;
                    const uint32_t output_row = e & 15;
                    const uint32_t column = e >> 4;
                    const float max0 = meta_tile[0][column][0];
                    const float sum0 = meta_tile[0][column][1];
                    const float max1 = meta_tile[1][column][0];
                    const float sum1 = meta_tile[1][column][1];
                    const float part_max = fmaxf(max0, max1);
                    const float scale0 = expf(max0 - part_max);
                    const float scale1 = expf(max1 - part_max);
                    const uint32_t offset = tile * 16 * 16 + e;
                    float part_num = fmaf(
                        scale0, __half2float(c_tile[0][offset]), 0.0f);
                    part_num = fmaf(
                        scale1, __half2float(c_tile[1][offset]), part_num);
                    float part_sum = scale0 * sum0;
                    part_sum = fmaf(scale1, sum1, part_sum);

                    if (part == int32_t(parts) - 1) {
                        accum_num[tile][slot] = part_num;
                        accum_sum[tile][slot] = part_sum;
                        accum_max[tile][slot] = part_max;
                    } else {
                        const float max_new = fmaxf(
                            accum_max[tile][slot], part_max);
                        const float diff_value = accum_max[tile][slot] - max_new;
                        const float diff_add = part_max - max_new;
                        const float scale_value = diff_value >= -20.0f
                            ? expf(diff_value) : 0.0f;
                        const float scale_add = diff_add >= -20.0f
                            ? expf(diff_add) : 0.0f;
                        // Match the pinned stream-K fixup instruction order: round
                        // the addend product, then contract scale_value*accumulator.
                        accum_num[tile][slot] = fmaf(
                            scale_value, accum_num[tile][slot], scale_add * part_num);
                        // Keep the denominator on the same contracted side as the
                        // pinned uniform Stream-K fixup for both supported head sizes.
                        accum_sum[tile][slot] = fmaf(
                            scale_value, accum_sum[tile][slot], scale_add * part_sum);
                        accum_max[tile][slot] = max_new;
                    }
                    if (part == 0) {
                        const uint32_t token = column / kGqaHeads;
                        const uint32_t head = kvh * kGqaHeads + column % kGqaHeads;
                        if (token < n_tok && head < n_heads) {
                            out[((uint64_t)token * n_heads + head) * HeadDim
                                + out0 + tile * 16 + output_row]
                                = accum_sum[tile][slot] > 0.0f
                                ? accum_num[tile][slot] / accum_sum[tile][slot]
                                : 0.0f;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)n_tok; (void)ring; (void)valid_span; (void)kv_span;
    (void)parts; (void)decode_control; (void)page_table;
#endif
}

} // namespace imparo_sm80_d512_small
