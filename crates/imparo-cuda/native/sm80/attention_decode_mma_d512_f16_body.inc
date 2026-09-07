#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    if (blockDim.x != kThreads || blockIdx.x >= physical_blocks
            || !supports_schedule(
                n_heads, n_kv, schedule_groups, physical_blocks)) {
        return;
    }

    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t query_pair = lane & 3;
    const uint32_t low = window > 0 && start_pos + 1 > window
        ? start_pos + 1 - window : 0;

    // Q_B is 8 query columns x 16 dimensions in shared-memory row-major
    // storage; load_b8x8 interprets it as the column-major MMA B operand.
    __shared__ __align__(16) __half2 q_tile[kWarps][8 * 8];
    __shared__ __align__(16) __half2 k_tile[kWarps][16 * 8];
    __shared__ __align__(16) __half2 value_tile[kWarps][16 * 8];
#if defined(IMPARO_D512_PAGED)
    __shared__ uint32_t physical_keys[kWarps][kKeysPerWarp];
#endif
    __shared__ __align__(16) __half result_tile[kWarps][kColumns * 16];
    __shared__ float partition_max[kWarps][kColumns];
    __shared__ float partition_sum[kWarps][kColumns];
    __shared__ float partition_scale[kWarps][kColumns];
    __shared__ float segment_max[kColumns];
    __shared__ float segment_sum[kColumns];

#pragma unroll
    for (uint32_t slot = 0; slot < kSegmentsPerPhysicalBlock; ++slot) {
        uint32_t logical_tile = 0;
        uint32_t group_begin = 0;
        uint32_t group_end = 0;
        if (!physical_segment_bounds(blockIdx.x, physical_blocks, n_kv,
                schedule_groups, slot, &logical_tile,
                &group_begin, &group_end)) {
            continue;
        }
        const uint32_t kvh = logical_tile;

        PvC16x4 numerator[kOutputTiles];
#pragma unroll
        for (uint32_t output_tile = 0;
             output_tile < kOutputTiles; ++output_tile) {
            numerator[output_tile].x[0] = __float2half2_rn(0.0f);
            numerator[output_tile].x[1] = __float2half2_rn(0.0f);
        }
        float running_max[2] = {kInitialMax, kInitialMax};
        float running_sum[2] = {0.0f, 0.0f};

        for (uint32_t group = group_begin; group < group_end; ++group) {
#if defined(IMPARO_D512_PAGED)
            if (lane < kKeysPerWarp) {
                const uint32_t logical = group * kKeyBatch
                    + warp * kKeysPerWarp + lane;
                physical_keys[warp][lane] = logical < valid_span
                    ? imparo_cuda_kv::physical_row(logical, ring, page_table)
                    : 0;
            }
            __syncwarp();
#endif
            KqC16x8 scores{};
#pragma unroll
            for (uint32_t d0 = 0; d0 < kHeadDim; d0 += 16) {
                for (uint32_t e = lane; e < 8 * 8; e += 32) {
                    const uint32_t column = e >> 3;
                    const uint32_t pair = e & 7;
                    const uint32_t head = kvh * kGqaHeads + column;
                    __half2 value = __float2half2_rn(0.0f);
                    if (column < kLiveColumns && head < n_heads) {
                        value = *reinterpret_cast<const __half2 *>(q
                            + uint64_t(head) * kHeadDim + d0 + 2 * pair);
                    }
                    q_tile[warp][e] = value;
                }
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t key_row = e >> 3;
                    const uint32_t pair = e & 7;
                    const uint32_t key = group * kKeyBatch
                        + warp * kKeysPerWarp + key_row;
                    __half2 value = __float2half2_rn(0.0f);
                    if (key < valid_span) {
#if defined(IMPARO_D512_Q4)
                        const uint32_t physical_key =
#if defined(IMPARO_D512_PAGED)
                            physical_keys[warp][key_row];
#else
                            key;
#endif
                        value = load_q4_pair(kc, kv_width, physical_key,
                            uint64_t(kvh) * kHeadDim + d0 + 2 * pair);
#else
                        value = *reinterpret_cast<const __half2 *>(kc
#if defined(IMPARO_D512_PAGED)
                            + uint64_t(physical_keys[warp][key_row]) * kv_width
#else
                            + uint64_t(key) * kv_width
#endif
                            + uint64_t(kvh) * kHeadDim + d0 + 2 * pair);
#endif
                    }
                    k_tile[warp][e] = value;
                }
                __syncwarp();
                KqB8x8 query_b;
                Half16x8 key_a;
                load_b8x8(query_b, q_tile[warp], 8, lane);
                load_half16x8(key_a, k_tile[warp], 8, lane);
                mma_kq_ncols8(scores, key_a, query_b);
                __syncwarp();
            }

            float max_new[2] = {running_max[0], running_max[1]};
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t key_row = kq_key_row(lane, l);
                const uint32_t column = kq_column(lane, l);
                const uint32_t key = group * kKeyBatch
                    + warp * kKeysPerWarp + key_row;
                const uint32_t key_position = key < valid_span
                    ? imparo_sm80_prefill::physical_key_position(
                        key, ring, start_pos)
                    : 0;
                const bool valid = column < kLiveColumns && key < valid_span
                    && key_position >= low && key_position <= start_pos;
                if (!valid) scores.x[l] = kMaskedScore;
                const uint32_t index = l & 1;
                if (valid) {
                    max_new[index] = fmaxf(
                        max_new[index], scores.x[l] + kMaxOffset);
                }
            }
#pragma unroll
            for (uint32_t index = 0; index < 2; ++index) {
#pragma unroll
                for (int offset = 16; offset >= 4; offset >>= 1) {
                    max_new[index] = fmaxf(max_new[index],
                        __shfl_xor_sync(
                            0xffffffffu, max_new[index], offset));
                }
            }

            float rowsum_add[2] = {0.0f, 0.0f};
#pragma unroll
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t index = l & 1;
                const float probability = scores.x[l] > -3.0e38F
                    ? expf(scores.x[l] - max_new[index]) : 0.0f;
                scores.x[l] = probability;
                rowsum_add[index] += probability;
            }
            const float scale0 = exp_ftz(running_max[0] - max_new[0]);
            const float scale1 = exp_ftz(running_max[1] - max_new[1]);
            running_max[0] = max_new[0];
            running_max[1] = max_new[1];
            running_sum[0] = running_sum[0] * scale0 + rowsum_add[0];
            running_sum[1] = running_sum[1] * scale1 + rowsum_add[1];

            const __half2 half_scale = __floats2half2_rn(scale0, scale1);
#pragma unroll
            for (uint32_t output_tile = 0;
                 output_tile < kOutputTiles; ++output_tile) {
                numerator[output_tile].x[0] = __hmul2(
                    numerator[output_tile].x[0], half_scale);
                numerator[output_tile].x[1] = __hmul2(
                    numerator[output_tile].x[1], half_scale);
            }

            const KqB8x8 probabilities = probability_b(scores);
#pragma unroll
            for (uint32_t output_tile = 0;
                 output_tile < kOutputTiles; ++output_tile) {
                for (uint32_t e = lane; e < 16 * 8; e += 32) {
                    const uint32_t key_row = e >> 3;
                    const uint32_t output_pair = e & 7;
                    const uint32_t key = group * kKeyBatch
                        + warp * kKeysPerWarp + key_row;
                    __half2 value = __float2half2_rn(0.0f);
                    if (key < valid_span) {
#if defined(IMPARO_D512_Q4)
                        const uint32_t physical_key =
#if defined(IMPARO_D512_PAGED)
                            physical_keys[warp][key_row];
#else
                            key;
#endif
                        value = load_q4_pair(vc, kv_width, physical_key,
                            uint64_t(kvh) * kHeadDim
                                + output_tile * 16 + 2 * output_pair);
#else
                        value = *reinterpret_cast<const __half2 *>(vc
#if defined(IMPARO_D512_PAGED)
                            + uint64_t(physical_keys[warp][key_row]) * kv_width
#else
                            + uint64_t(key) * kv_width
#endif
                            + uint64_t(kvh) * kHeadDim
                            + output_tile * 16 + 2 * output_pair);
#endif
                    }
                    value_tile[warp][e] = value;
                }
                __syncwarp();
                Half16x8 value_a;
                load_half16x8_trans(value_a, value_tile[warp], 8, lane);
                mma_vp_ncols8(
                    numerator[output_tile], value_a, probabilities);
                __syncwarp();
            }
        }

#pragma unroll
        for (uint32_t index = 0; index < 2; ++index) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                running_sum[index] += __shfl_xor_sync(
                    0xffffffffu, running_sum[index], offset);
            }
        }
        if (lane < 4) {
            const uint32_t column0 = 2 * lane;
            partition_max[warp][column0] = running_max[0];
            partition_sum[warp][column0] = running_sum[0];
            partition_max[warp][column0 + 1] = running_max[1];
            partition_sum[warp][column0 + 1] = running_sum[1];
        }
        __syncthreads();

        if (warp == 0 && lane < kColumns) {
            const float max0 = partition_max[0][lane];
            const float max1 = partition_max[1][lane];
            const float max_value = fmaxf(max0, max1);
            const float scale_partition0 = expf(max0 - max_value);
            const float scale_partition1 = expf(max1 - max_value);
            partition_scale[0][lane] = scale_partition0;
            partition_scale[1][lane] = scale_partition1;
            segment_max[lane] = max_value;
            segment_sum[lane] = fmaf(
                scale_partition1, partition_sum[1][lane],
                scale_partition0 * partition_sum[0][lane]);
        }
        __syncthreads();

        float * slot_base = workspace
            + (uint64_t(blockIdx.x) * kSegmentsPerPhysicalBlock + slot)
                * kSlotStrideFloats;
#pragma unroll
        for (uint32_t output_tile = 0;
             output_tile < kOutputTiles; ++output_tile) {
            const uint32_t output_row0 = pv_output_row(lane, 0);
            const uint32_t output_row1 = pv_output_row(lane, 1);
            const uint32_t column0 = 2 * query_pair;
            result_tile[warp][column0 * 16 + output_row0]
                = __low2half(numerator[output_tile].x[0]);
            result_tile[warp][(column0 + 1) * 16 + output_row0]
                = __high2half(numerator[output_tile].x[0]);
            result_tile[warp][column0 * 16 + output_row1]
                = __low2half(numerator[output_tile].x[1]);
            result_tile[warp][(column0 + 1) * 16 + output_row1]
                = __high2half(numerator[output_tile].x[1]);
            __syncthreads();
            if (warp == 0) {
                for (uint32_t e = lane; e < kColumns * 16; e += 32) {
                    const uint32_t column = e >> 4;
                    const uint32_t output = output_tile * 16 + (e & 15);
                    float value = fmaf(partition_scale[0][column],
                        __half2float(result_tile[0][e]), 0.0f);
                    value = fmaf(partition_scale[1][column],
                        __half2float(result_tile[1][e]), value);
                    slot_base[uint64_t(column) * kHeadDim + output] = value;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x < kColumns) {
            slot_base[kSlotNumeratorFloats + threadIdx.x]
                = segment_max[threadIdx.x];
            slot_base[kSlotNumeratorFloats + kColumns + threadIdx.x]
                = segment_sum[threadIdx.x];
        }
        __syncthreads();
    }
#else
    (void)q; (void)kc; (void)vc; (void)workspace; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos;
    (void)window; (void)ring; (void)valid_span; (void)schedule_groups;
    (void)physical_blocks;
#if defined(IMPARO_D512_PAGED)
    (void)page_table;
#endif
#endif
