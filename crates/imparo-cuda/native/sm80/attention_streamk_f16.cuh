#pragma once

// SM80+ numerical compatibility fixup for the D=256, GQA=4 MMA-F16
// attention shapes used by Gemma4 E4B. The common CUDA workflow remains
// architecture-neutral; this file owns Ampere launch geometry and fragment
// reduction contracts.
__global__ void k_attention_streamk_fixup_sm80_f16(
        const float * q, const __half * kc, const __half * vc, float * out,
        uint32_t head_dim, uint32_t n_heads, uint32_t n_kv, uint32_t kv_width,
        uint32_t start_pos, float qk_scale, uint32_t window, uint32_t ring, uint32_t n_tok,
        uint32_t sm_count, const uint32_t * page_table) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (head_dim != 256 || n_heads / n_kv != 4 || blockDim.x != 128) return;
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;

    constexpr uint32_t score_batch = 32;
    constexpr uint32_t query_tile = 16;
    constexpr uint32_t gqa_tile = 4;
    // Pinned SM86 ncols64 MMA-F16 config uses parallel_blocks=2 for D256.
    constexpr uint32_t occupancy = 2;
    const uint32_t kvh = h / 4;
    const uint32_t pos = start_pos + t;
    const uint32_t lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    const uint32_t groups = (start_pos + n_tok + score_batch - 1) / score_batch;
    const uint32_t query_tiles = (n_tok + query_tile - 1) / query_tile;
    const uint32_t gqa_tiles = ((n_heads / n_kv) + gqa_tile - 1) / gqa_tile;
    const uint32_t output_tiles = query_tiles * gqa_tiles * n_kv;
    if (groups == 0 || output_tiles == 0) return;

    // Mirror launch_fattn's Ampere Stream-K selection and general-fixup grid.
    const uint32_t max_blocks = max(1u, sm_count) * occupancy;
    const uint32_t tile_waves = (output_tiles + max_blocks - 1) / max_blocks;
    const uint32_t tile_efficiency = 100 * output_tiles / (max_blocks * tile_waves);
    if (tile_efficiency >= 75) return;
    const uint64_t total_work = uint64_t(groups) * output_tiles;
    const uint32_t raw_grid = total_work < max_blocks
        ? uint32_t(total_work) : max_blocks;
    const uint32_t rounded_grid = (raw_grid / output_tiles) * output_tiles;
    const uint32_t efficiency_loss = rounded_grid > 0
        ? 100 * (raw_grid - rounded_grid) / raw_grid : 100;
    const uint32_t stream_grid = efficiency_loss <= 5 ? rounded_grid : raw_grid;
    if (stream_grid == 0) return;

    const uint32_t zt_gqa = (h - kvh * 4) / gqa_tile;
    const uint32_t tile = (kvh * gqa_tiles + zt_gqa) * query_tiles + t / query_tile;
    uint32_t seam = 0;
    uint32_t seams = 0;
    for (uint32_t b = 1; b < stream_grid; ++b) {
        const uint64_t boundary = uint64_t(b) * total_work / stream_grid;
        if (boundary / groups == tile && boundary % groups != 0) {
            seam = uint32_t(boundary % groups);
            ++seams;
        }
    }
    // The current SM86 E4B gate has one seam per affected tile. More complex
    // schedules safely keep the already-written sequential result until a generic
    // N-way fixup implementation is registered for that shape.
    if (seams != 1) return;

    const float * qr = q + ((uint64_t)t * n_heads + h) * head_dim;
    __shared__ __align__(16) __half mma_q[16 * 16];
    __shared__ __align__(16) __half mma_k[16 * 16];
    __shared__ __align__(16) float mma_c[16 * 16];
    __shared__ __align__(16) __half prob_tile[16 * 16];
    __shared__ __align__(16) __half v_tile[8][16 * 16];
    __shared__ __align__(16) __half out_tile[8][16 * 16];
    __shared__ float scores[score_batch];
    __shared__ float m_part[4];
    __shared__ float s_part[4];
    __shared__ float group_scale;
    __shared__ float phase_max[2];
    __shared__ float phase_sum[2];
    __shared__ float prefix[512];

    const uint32_t nwarps = blockDim.x / 32;
    const uint32_t v_tiles = head_dim / (nwarps * 16);

    using namespace nvcuda;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, __half> v_acc[8];
        for (uint32_t tile_i = 0; tile_i < v_tiles; ++tile_i) {
            wmma::fill_fragment(v_acc[tile_i], 0.0f);
        }
        if (threadIdx.x < 4) {
            m_part[threadIdx.x] = -1.701411733e+38F;
            s_part[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        const uint32_t group_begin = phase == 0 ? 0 : seam;
        const uint32_t group_end = phase == 0 ? seam : groups;
        for (uint32_t group = group_begin; group < group_end; ++group) {
            const uint32_t gb = group * score_batch;
            for (uint32_t j = 0; j < score_batch; ++j) {
                const uint32_t gp = gb + j;
                if (gp < lo || gp > pos) {
                    if (threadIdx.x == 0) scores[j] = -3.402823466e+38F;
                    __syncthreads();
                    continue;
                }
                const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
                if (threadIdx.x < 32) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                    wmma::fill_fragment(c, 0.0f);
                    for (uint32_t k0 = 0; k0 < head_dim; k0 += 16) {
                        for (uint32_t e = threadIdx.x; e < 16 * 16; e += 32) {
                            mma_q[e] = e < 16
                                ? __hmul(__float2half(qr[k0 + e]), __float2half(qk_scale))
                                : __float2half(0.0f);
                            mma_k[e] = e < 16
                                ? kc[(uint64_t)ps * kv_width + kvh * head_dim + k0 + e]
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
                    if (threadIdx.x == 0) scores[j] = mma_c[0];
                }
                __syncthreads();
            }

            if (threadIdx.x < 4) {
                const uint32_t lane = threadIdx.x;
                float m_new = m_part[lane];
                constexpr float max_offset = 3.0f * 0.6931f;
                for (uint32_t block16 = 0; block16 < 32; block16 += 16) {
                    for (uint32_t half8 = 0; half8 < 16; half8 += 8) {
                        for (uint32_t pair = 0; pair < 2; ++pair) {
                            const uint32_t row = block16 + half8 + 2 * lane + pair;
                            if (gb + row >= lo && gb + row <= pos) {
                                m_new = fmaxf(m_new, scores[row] + max_offset);
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
            }
            __syncthreads();

            const __half hs = __float2half(group_scale);
            for (uint32_t tile_i = 0; tile_i < v_tiles; ++tile_i) {
                for (int e = 0; e < v_acc[tile_i].num_elements; ++e) {
                    v_acc[tile_i].x[e] = __hmul(v_acc[tile_i].x[e], hs);
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
                        __half value = __float2half(0.0f);
                        if (gp >= lo && gp <= pos) {
                            const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
                            value = vc[(uint64_t)ps * kv_width + kvh * head_dim
                                     + out_block * 16 + row];
                        }
                        v_tile[warp][e] = value;
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
        }

        if (threadIdx.x < 4) {
            float sum = s_part[threadIdx.x];
#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0x0000000f, sum, offset);
            }
            if (threadIdx.x == 0) {
                phase_max[phase] = m_part[0];
                phase_sum[phase] = sum;
            }
        }
        __syncthreads();

        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t lane = threadIdx.x & 31;
        for (uint32_t local = 0; local < v_tiles; ++local) {
            const uint32_t out_block = warp + nwarps * local;
            wmma::store_matrix_sync(out_tile[warp], v_acc[local], 16,
                                    wmma::mem_row_major);
            __syncwarp();
            if (lane < 16) {
                const uint32_t index = out_block * 16 + lane;
                const float numerator = __half2float(out_tile[warp][lane * 16]);
                if (phase == 0) {
                    prefix[index] = numerator;
                } else {
                    const float max_new = fmaxf(phase_max[1], phase_max[0]);
                    const float diff_val = phase_max[1] - max_new;
                    const float diff_add = phase_max[0] - max_new;
                    const float scale_val = diff_val >= -20.0f ? expf(diff_val) : 0.0f;
                    const float scale_add = diff_add >= -20.0f ? expf(diff_add) : 0.0f;
                    const float value = scale_val * numerator + scale_add * prefix[index];
                    const float rowsum = scale_val * phase_sum[1] + scale_add * phase_sum[0];
                    out[((uint64_t)t * n_heads + h) * head_dim + index] = value / rowsum;
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
#else
    (void)q; (void)kc; (void)vc; (void)out; (void)head_dim; (void)n_heads;
    (void)n_kv; (void)kv_width; (void)start_pos; (void)window; (void)ring;
    (void)n_tok; (void)sm_count; (void)page_table;
#endif
}
