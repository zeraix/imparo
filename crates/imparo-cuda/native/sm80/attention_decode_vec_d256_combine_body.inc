
#if defined(IMPARO_D256_PARALLEL_COMBINE)
    const uint64_t row = blockIdx.y;
    children += row * n_heads * kMaxPartials * kWarps * kGqaChildStride;
    out += row * n_heads * kHeadDim;
#else
#endif
    const uint32_t head = blockIdx.x;
    const uint32_t i = threadIdx.x;
#if defined(IMPARO_D256_PARALLEL_COMBINE)
    const uint32_t lane = i & 31u;
    const uint32_t warp = i >> 5;
#else
#endif
    if (head >= n_heads || partial_count == 0
        || partial_count > kMaxPartials) return;
    const float * head_base = children
        + uint64_t(head) * kMaxPartials * kWarps * kGqaChildStride;
    __shared__ float partial_maxima[kMaxPartials];
    __shared__ float source_scales[kMaxPartials][kWarps];
#if defined(IMPARO_D256_PARALLEL_COMBINE)
    __shared__ float source_denominators[kMaxPartials][kWarps];
#else
    __shared__ float source_denominators[kWarps];
#endif
    __shared__ float partial_denominators[kMaxPartials];
#if defined(IMPARO_D256_PARALLEL_COMBINE)
    __shared__ float partial_scales[kMaxPartials];
    __shared__ float denominator_shared;
#else
    __shared__ float combined_max_shared;
#endif

#if defined(IMPARO_D256_PARALLEL_COMBINE)
    // Each independent schedule partial keeps its original lane/reduction order.
    if (warp < partial_count) {
        const uint32_t part = warp;
#else
    if (i < 32) {
        for (uint32_t part = 0; part < partial_count; ++part) {
#endif
            const float * base = head_base
                + uint64_t(part) * kWarps * kGqaChildStride;
#if defined(IMPARO_D256_PARALLEL_COMBINE)
            float combined_max = lane < kWarps
                ? base[uint64_t(lane) * kGqaChildStride + kHeadDim]
#else
            float combined_max = i < kWarps
                ? base[uint64_t(i) * kGqaChildStride + kHeadDim]
#endif
                : kNegInf * 0.5f;
            combined_max = warp_max(combined_max);
#if defined(IMPARO_D256_PARALLEL_COMBINE)
            if (lane == 0) partial_maxima[part] = combined_max;
            if (lane < kWarps) {
                source_scales[part][lane] = expf(
                    base[uint64_t(lane) * kGqaChildStride + kHeadDim]
#else
            if (i == 0) partial_maxima[part] = combined_max;
            if (i < kWarps) {
                source_scales[part][i] = expf(
                    base[uint64_t(i) * kGqaChildStride + kHeadDim]
#endif
                    - combined_max);
            }
            __syncwarp();
#pragma unroll
            for (uint32_t source = 0; source < kWarps; ++source) {
                const float lane_sum =
                    base[uint64_t(source) * kGqaChildStride
#if defined(IMPARO_D256_PARALLEL_COMBINE)
                        + kHeadDim + 2 + lane] * source_scales[part][source];
#else
                        + kHeadDim + 2 + i] * source_scales[part][source];
#endif
                const float sum = warp_sum(lane_sum);
#if defined(IMPARO_D256_PARALLEL_COMBINE)
                if (lane == 0) source_denominators[part][source] = sum;
#else
                if (i == 0) source_denominators[source] = sum;
#endif
            }
            __syncwarp();
            float denominator =
#if defined(IMPARO_D256_PARALLEL_COMBINE)
                lane < kWarps ? source_denominators[part][lane] : 0.0f;
#else
                i < kWarps ? source_denominators[i] : 0.0f;
#endif
            denominator = warp_sum(denominator);
#if defined(IMPARO_D256_PARALLEL_COMBINE)
            if (lane == 0) partial_denominators[part] = denominator;
#else
            if (i == 0) partial_denominators[part] = denominator;
#endif
            __syncwarp();
#if defined(IMPARO_D256_PARALLEL_COMBINE)
    }
    __syncthreads();
    if (warp == 0) {
        float combined_max = lane < partial_count
            ? partial_maxima[lane] : kNegInf * 0.5f;
        combined_max = warp_max(combined_max);
        if (lane < partial_count)
            partial_scales[lane] = expf(partial_maxima[lane] - combined_max);
        __syncwarp();
        if (lane == 0) {
            float denominator = 0.0f;
            for (uint32_t part = 0; part < partial_count; ++part)
                denominator += partial_scales[part] * partial_denominators[part];
            denominator_shared = denominator;
#else
#endif
        }
#if defined(IMPARO_D256_PARALLEL_COMBINE)
#else
        float combined_max = i < partial_count
            ? partial_maxima[i] : kNegInf * 0.5f;
        combined_max = warp_max(combined_max);
        if (i == 0) combined_max_shared = combined_max;
#endif
    }
    __syncthreads();

    float numerator = 0.0f;
#if defined(IMPARO_D256_PARALLEL_COMBINE)
#else
    float denominator = 0.0f;
#endif
    for (uint32_t part = 0; part < partial_count; ++part) {
        const float * base = head_base
            + uint64_t(part) * kWarps * kGqaChildStride;
        float partial_numerator = __fadd_rn(
            0.0f, __fmul_rn(source_scales[part][0], base[i]));
#pragma unroll
        for (uint32_t source = 1; source < kWarps; ++source) {
            const float scaled = __fmul_rn(
                source_scales[part][source],
                base[uint64_t(source) * kGqaChildStride + i]);
            partial_numerator = __fadd_rn(partial_numerator, scaled);
        }
        // Match combine_q4's compiler-visible multiply-add expression here:
        // unlike the source-warp boundary, the old path has no intervening
        // memory round before accumulating schedule partials.
#if defined(IMPARO_D256_PARALLEL_COMBINE)
        const float scale = partial_scales[part];
#else
        const float scale = expf(
            partial_maxima[part] - combined_max_shared);
#endif
        numerator += scale * partial_numerator;
#if defined(IMPARO_D256_PARALLEL_COMBINE)
#else
        denominator += scale * partial_denominators[part];
#endif
    }
#if defined(IMPARO_D256_PARALLEL_COMBINE)
    out[uint64_t(head) * kHeadDim + i] = numerator / denominator_shared;
#else
    out[uint64_t(head) * kHeadDim + i] = numerator / denominator;
#endif
