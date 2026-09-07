#pragma once

// D=256, one-token staged-f16 attention numerical route. The generic small
// scheduler splits each 32-key batch into independent 16-key softmax/PV
// partitions. This route keeps the complete 32-key batch under one running
// max/rowsum and feeds both 16-key halves into the same half MMA accumulator.
namespace imparo_sm80_d256_decode {

__global__ void softmax_full32_parts(
        float * workspace, uint32_t kv_span, uint32_t parts) {
    const uint32_t local_block = blockIdx.x;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t flat = blockIdx.y * (blockDim.x / 32) + warp;
    if (flat >= imparo_sm80_d512_small::kColumns * parts) return;
    const uint32_t part = flat % parts;
    const uint32_t column = flat / parts;
    const uint32_t groups = (kv_span + imparo_sm80_d512_small::kKeyBatch - 1)
        / imparo_sm80_d512_small::kKeyBatch;
    const uint32_t group_start = uint64_t(part) * groups / parts;
    const uint32_t group_stop = uint64_t(part + 1) * groups / parts;
    const uint64_t base = uint64_t(local_block)
        * imparo_sm80_d512_small::block_stride(kv_span, parts);
    float * row = workspace + base + uint64_t(column) * kv_span;
    constexpr float max_offset = 3.0f * 0.6931f;
    float running_max = -3.402823466e+38F;
    float partial_sum = 0.0f;

    for (uint32_t group = group_start; group < group_stop; ++group) {
        float next_max = running_max;
        if (lane < 4) {
#pragma unroll
            for (uint32_t half = 0; half < 32; half += 16) {
#pragma unroll
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
#pragma unroll
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t key = group * 32 + half + half8
                            + 2 * lane + pair;
                        if (key < kv_span && row[key] > -3.0e38F) {
                            next_max = fmaxf(next_max, row[key] + max_offset);
                        }
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
            workspace[base + imparo_sm80_d512_small::rescale_index(
                kv_span, 0, column, group)] = rescale;
        }

        if (lane < 4) {
            float add = 0.0f;
#pragma unroll
            for (uint32_t half = 0; half < 32; half += 16) {
#pragma unroll
                for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
#pragma unroll
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        const uint32_t key = group * 32 + half + half8
                            + 2 * lane + pair;
                        if (key < kv_span && row[key] > -3.0e38F) {
                            const float probability = expf(row[key] - next_max);
                            row[key] = probability;
                            add += probability;
                        } else if (key < kv_span) {
                            row[key] = 0.0f;
                        }
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
            workspace[base + imparo_sm80_d512_small::meta_index(
                kv_span, parts, 0, column, part, 0)] = running_max;
            workspace[base + imparo_sm80_d512_small::meta_index(
                kv_span, parts, 0, column, part, 1)] = sum;
        }
    }
}

__global__ void values_full32_combine(
        const __half * vc, const float * workspace, float * out,
        uint32_t block_base, uint32_t n_heads, uint32_t n_kv,
        uint32_t kv_width, uint32_t ring, uint32_t valid_span,
        uint32_t kv_span, uint32_t parts, const uint32_t * decode_control,
        const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (decode_control) {
        const uint32_t start_pos = decode_control[0];
        valid_span = ring ? min(start_pos + 1, ring + 1) : start_pos + 1;
    }
    const uint32_t local_block = blockIdx.x;
    const uint32_t kvh = block_base + local_block;
    if (kvh >= n_kv) return;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t out0 = blockIdx.y * 16;
    const uint32_t groups = (kv_span + 31) / 32;
    const uint64_t base = uint64_t(local_block)
        * imparo_sm80_d512_small::block_stride(kv_span, parts);
    __shared__ __align__(16) __half2 p_tile[16 * 8];
    __shared__ __align__(16) __half2 v_tile[16 * 8];
    __shared__ uint32_t physical_keys[16];
    __shared__ __align__(16) __half c_tile[16 * 16];
    __shared__ float rescale_tile[16];
    __shared__ float meta_tile[16][2];
    float accum_num[8]{};
    float accum_sum[8]{};
    float accum_max[8]{};

    for (int32_t part = int32_t(parts) - 1; part >= 0; --part) {
        imparo_sm80_prefill::Half16x8 c;
#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            c.x[l] = __float2half2_rn(0.0f);
        }
        const uint32_t group_start = uint64_t(uint32_t(part)) * groups / parts;
        const uint32_t group_stop = uint64_t(uint32_t(part) + 1) * groups / parts;
        for (uint32_t group = group_start; group < group_stop; ++group) {
            if (group > group_start) {
                if (lane < 16) {
                    rescale_tile[lane] = workspace[base
                        + imparo_sm80_d512_small::rescale_index(
                            kv_span, 0, lane, group)];
                }
                __syncwarp();
#pragma unroll
                for (uint32_t l = 0; l < 4; ++l) {
                    const uint32_t column =
                        imparo_sm80_prefill::half_acc_query_column(lane, l);
                    c.x[l] = __hmul2(c.x[l],
                        __float2half2_rn(rescale_tile[column]));
                }
            }

#pragma unroll
            for (uint32_t half = 0; half < 32; half += 16) {
                const uint32_t key0 = group * 32 + half;
                if (lane < 16) {
                    const uint32_t logical = key0 + lane;
                    physical_keys[lane] = logical < valid_span
                        ? imparo_cuda_kv::physical_row(logical, ring, page_table)
                        : 0;
                }
                __syncwarp();
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t query = e >> 3;
                    const uint32_t pair = e & 7;
                    const uint32_t key = key0 + 2 * pair;
                    const float p0 = key < kv_span
                        ? workspace[base + uint64_t(query) * kv_span + key] : 0.0f;
                    const float p1 = key + 1 < kv_span
                        ? workspace[base + uint64_t(query) * kv_span + key + 1] : 0.0f;
                    p_tile[query * 8 + pair] = __floats2half2_rn(p0, p1);
                }
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t key = key0 + (e >> 3);
                    const uint32_t out_pair = e & 7;
                    __half2 value = __float2half2_rn(0.0f);
                    if (key < valid_span) {
                        const uint32_t index = kvh * 256 + out0 + 2 * out_pair;
                        value = *reinterpret_cast<const __half2 *>(vc
                            + uint64_t(physical_keys[e >> 3]) * kv_width + index);
                    }
                    v_tile[(e >> 3) * 8 + out_pair] = value;
                }
                __syncwarp();
                imparo_sm80_prefill::Half16x8 probability;
                imparo_sm80_prefill::Half16x8 value;
                imparo_sm80_prefill::load_half16x8(probability, p_tile, 8, lane);
                imparo_sm80_prefill::load_half16x8_trans(value, v_tile, 8, lane);
                imparo_sm80_prefill::mma_pv(c, probability, value);
                __syncwarp();
            }
        }

#pragma unroll
        for (uint32_t l = 0; l < 4; ++l) {
            const uint32_t column =
                imparo_sm80_prefill::half_acc_query_column(lane, l);
            const uint32_t out_pair =
                imparo_sm80_prefill::half_acc_output_pair(lane, l);
            const uint32_t offset = column * 16 + 2 * out_pair;
            c_tile[offset] = __low2half(c.x[l]);
            c_tile[offset + 1] = __high2half(c.x[l]);
        }
        if (lane < 16) {
            meta_tile[lane][0] = workspace[base
                + imparo_sm80_d512_small::meta_index(
                    kv_span, parts, 0, lane, uint32_t(part), 0)];
            meta_tile[lane][1] = workspace[base
                + imparo_sm80_d512_small::meta_index(
                    kv_span, parts, 0, lane, uint32_t(part), 1)];
        }
        __syncwarp();

#pragma unroll
        for (uint32_t slot = 0; slot < 8; ++slot) {
            const uint32_t e = lane + slot * 32;
            const uint32_t output_row = e & 15;
            const uint32_t column = e >> 4;
            const float part_max = meta_tile[column][0];
            const float part_sum = meta_tile[column][1];
            const float part_num = __half2float(c_tile[e]);
            if (part == int32_t(parts) - 1) {
                accum_num[slot] = part_num;
                accum_sum[slot] = part_sum;
                accum_max[slot] = part_max;
            } else {
                const float max_new = fmaxf(accum_max[slot], part_max);
                const float diff_value = accum_max[slot] - max_new;
                const float diff_add = part_max - max_new;
                const float scale_value = diff_value >= -20.0f
                    ? expf(diff_value) : 0.0f;
                const float scale_add = diff_add >= -20.0f
                    ? expf(diff_add) : 0.0f;
                accum_num[slot] = fmaf(
                    scale_value, accum_num[slot], scale_add * part_num);
                accum_sum[slot] = fmaf(
                    scale_value, accum_sum[slot], scale_add * part_sum);
                accum_max[slot] = max_new;
            }
            if (part == 0) {
                const uint32_t token = column / 4;
                const uint32_t head = kvh * 4 + column % 4;
                if (token == 0 && head < n_heads) {
                    out[uint64_t(head) * 256 + out0 + output_row]
                        = accum_sum[slot] > 0.0f
                        ? accum_num[slot] / accum_sum[slot] : 0.0f;
                }
            }
        }
        __syncwarp();
    }
#else
    (void)vc; (void)workspace; (void)out; (void)block_base; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)ring; (void)valid_span; (void)kv_span;
    (void)parts; (void)decode_control; (void)page_table;
#endif
}

} // namespace imparo_sm80_d256_decode
