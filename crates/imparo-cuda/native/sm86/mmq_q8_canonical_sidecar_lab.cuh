#pragma once

// Canonical Q8 candidate. Reuse canonical weight staging and the existing paired
// MMA schedule; publish D4 records directly for Down. Runtime admission is owned
// by the safe-off CUDA tuner selection, not by inclusion of this header.
namespace canonical_q8_sidecar_lab {
namespace C = canonical_q8_pair_lab;
namespace P = imparo_sm86_q8_tm_gate_up_row_pair_lab;

// Adjacent lanes fetch one 32-value record together. This changes only
// global-load ownership; the shared layout and MMA arithmetic remain identical.
template <uint32_t LoadLanes = 8, bool HalfwordLoads = false,
          bool WarpWordLoads = false, uint32_t Warps = P::kWarps>
__device__ __forceinline__ void stage_cooperative(
        const uint8_t *gate, const uint8_t *up, int8_t *gs, int8_t *us,
        uint32_t blocks, uint32_t row0, uint32_t kb0, uint32_t tid) {
    constexpr uint32_t records = P::kRows * P::kStageBlocks;
    static_assert(LoadLanes == 2 || LoadLanes == 4 || LoadLanes == 8);
    static_assert(!WarpWordLoads || (LoadLanes == 8 && !HalfwordLoads));
    const uint32_t word_base = tid % LoadLanes;
    for (uint32_t index = tid / LoadLanes; index < 2 * records;
            index += Warps * 32 / LoadLanes) {
        const uint32_t projection = index / records;
        const uint32_t within = index % records;
        const uint32_t row = within / P::kStageBlocks;
        const uint32_t kb = within % P::kStageBlocks;
        const uint8_t *block = (projection ? up : gate)
            + (uint64_t(row0 + row) * blocks + kb0 + kb) * 34;
        int8_t *stage = (projection ? us : gs) + row * P::kWeightStride;
        if constexpr (WarpWordLoads) {
            // Each eight-lane group owns one complete 34-byte Q8 record.
            // Read aligned words; exchange the next word for payloads at +2.
            // The final halfword is loaded inside the record, never beyond it.
            const bool shifted=(reinterpret_cast<uintptr_t>(block+2)&2u)!=0;
            const auto *aligned=reinterpret_cast<const uint32_t *>(block+(shifted?0:2));
            const uint32_t lo=aligned[word_base];
            uint32_t hi=__shfl_down_sync(0xffffffff,lo,1,8);
            if(word_base==7 && shifted) hi=*reinterpret_cast<const uint16_t *>(block+32);
            const uint32_t packed=shifted?((lo>>16)|(hi<<16)):lo;
            reinterpret_cast<int *>(stage+kb*32)[word_base]=int(packed);
        } else {
#pragma unroll
            for (uint32_t item = 0; item < 8 / LoadLanes; ++item) {
                const uint32_t word = word_base + item * LoadLanes;
                reinterpret_cast<int *>(stage + kb * 32)[word] = HalfwordLoads
                    ? imparo_sm80_q8_mmq::load_q8_halfwords(block + 2, word)
                    : imparo_sm80_q8_mmq::load_q8_word(block + 2, word);
            }
        }
        if (word_base == 0) {
            reinterpret_cast<float *>(stage + imparo_sm80_q8_mmq::kStageValues)[kb]
                = __half2float(*reinterpret_cast<const __half *>(block));
        }
    }
}

template <bool Cooperative = false, uint32_t LoadLanes = 8, bool HalfwordLoads = false,
          bool WarpWordLoads = false>
__launch_bounds__(256, 1)
__global__ void fused(const uint8_t *gate, const uint8_t *up,
        const BlockQ8_1Mmq *x, BlockQ8_1Mmq *out,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    static_assert(!WarpWordLoads || Cooperative);
    extern __shared__ __align__(16) int8_t shared[];
    int8_t *gs = shared, *us = gs + P::kProjectionWeightBytes;
    int8_t *xs = us + P::kProjectionWeightBytes;
    const uint32_t lane = threadIdx.x, warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane, pair = warp / 2, projection = warp % 2;
    const uint32_t row0 = blockIdx.x * P::kRows, tok0 = blockIdx.y * P::kTokens;
    const uint32_t active = min(P::kTokens, n_tok - tok0);
    float partial[64] = {};
    for (uint32_t kb = 0; kb < n_in / 32; kb += P::kStageBlocks) {
#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase)
            imparo_sm80_q8_mmq::stage_activation_group<P::kTokens>(x,
                xs + phase * P::kTokens * P::kActivationStride,
                n_tok, tok0, active, (kb + phase * 4) / 4, true, tid);
        imparo_sm80_mmq::commit_async_copies();
        if constexpr (Cooperative)
            stage_cooperative<LoadLanes, HalfwordLoads, WarpWordLoads>(gate, up, gs, us, n_in / 32, row0, kb, tid);
        else
            C::stage_weights(gate, up, gs, us, n_in / 32, row0, kb, tid);
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();
        P::accumulate_projection(projection ? us : gs, xs, partial, lane, pair);
        __syncthreads();
    }
    // Reuse dead staging storage for the Gate/Up handoff and quantization.
    float *tile = reinterpret_cast<float *>(shared);
    if (projection == 0) {
#pragma unroll
        for (uint32_t fragment = 0; fragment < 16; ++fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t row = pair * 16 + imparo_sm80_mmq::accumulator_row(lane,item);
                const uint32_t tok = fragment * 8 + imparo_sm80_mmq::accumulator_token(lane,item);
                if (tok < active) tile[tok * P::kRows + row] = partial[fragment*4+item];
            }
        }
    }
    __syncthreads();
    if (projection == 1) {
#pragma unroll
        for (uint32_t fragment = 0; fragment < 16; ++fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t row = pair * 16 + imparo_sm80_mmq::accumulator_row(lane,item);
                const uint32_t tok = fragment * 8 + imparo_sm80_mmq::accumulator_token(lane,item);
                if (tok < active) {
                    const uint32_t offset = tok * P::kRows + row;
                    tile[offset] = imparo_cuda_lfm2::silu(tile[offset]) * partial[fragment*4+item];
                }
            }
        }
    }
    __syncthreads();
    // Match k_quantize_q8_1_mmq's float4 loads, eight-lane max and rounding.
    // Adjacent R64 CTAs own disjoint 32-value blocks, not overlapping stores.
    const uint32_t units = active * (P::kRows / 32);
    for (uint32_t base = warp * 4; base < units; base += P::kWarps * 4) {
        const uint32_t unit = base + lane / 8;
        const bool valid = unit < units;
        const uint32_t safe = valid ? unit : 0;
        const uint32_t tok = safe / (P::kRows / 32), block = safe % (P::kRows / 32);
        const float4 v = reinterpret_cast<const float4 *>(tile + tok * P::kRows)[block*8+lane%8];
        float amax = fabsf(v.x);
        amax = fmaxf(amax, fabsf(v.y));
        amax = fmaxf(amax, fabsf(v.z));
        amax = fmaxf(amax, fabsf(v.w));
#pragma unroll
        for (int shift=4; shift>0; shift>>=1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff,amax,shift,32));
        const float inverse = 127.0f / amax;
        char4 q;
        q.x = int8_t(roundf(v.x*inverse)); q.y = int8_t(roundf(v.y*inverse));
        q.z = int8_t(roundf(v.z*inverse)); q.w = int8_t(roundf(v.w*inverse));
        if (valid) {
            BlockQ8_1Mmq *record = out + uint64_t(row0/128)*n_tok + tok0 + tok;
            const uint32_t subblock = (row0%128)/32 + block;
            reinterpret_cast<char4 *>(record->qs+subblock*32)[lane%8] = q;
            if (lane%8 == 0) record->d[subblock] = 1.0f/inverse;
        }
    }
#endif
}

// Stage four K blocks at a time. This halves dynamic shared memory and asks
// ptxas to fit two CTAs per SM without changing the per-output accumulation
// order. It is instantiated only by the standalone laboratory executable.
namespace stage4 {
constexpr uint32_t kRows=64,kTokens=128,kTokens64=64,kWarps=8,kStageBlocks=4;
constexpr uint32_t kWeightStride=4*32+4*sizeof(float);
constexpr uint32_t kActivationStride=imparo_sm80_q8_mmq::kActivationStride;
constexpr uint32_t kActivationRecordValues=imparo_sm80_q8_mmq::kActivationRecordValues;
constexpr uint32_t kProjectionBytes=kRows*kWeightStride;
template<uint32_t Rows,uint32_t Tokens>
constexpr uint32_t shared_bytes() {
    constexpr uint32_t staging=2*Rows*kWeightStride+Tokens*kActivationStride;
    constexpr uint32_t handoff=Rows*Tokens*sizeof(float);
    return staging>handoff?staging:handoff;
}
constexpr uint32_t kSharedBytes=shared_bytes<kRows,kTokens>();
constexpr uint32_t kSharedBytesT64=shared_bytes<kRows,kTokens64>();
constexpr uint32_t kSharedBytesR128=shared_bytes<128,kTokens>();
constexpr uint32_t kSharedBytesR32=shared_bytes<32,kTokens>();
static_assert(kSharedBytes==36864);
static_assert(kSharedBytesT64==27648);
static_assert(kSharedBytesR128==65536);
static_assert(kSharedBytesR32==27648);

template<uint32_t Rows,uint32_t Warps>
__device__ __forceinline__ void stage_weights(const uint8_t *gate,const uint8_t *up,
        int8_t *gs,int8_t *us,uint32_t blocks,uint32_t row0,uint32_t kb0,uint32_t tid) {
    constexpr uint32_t records=Rows*kStageBlocks;
    const uint32_t word_base=tid%8;
    for(uint32_t index=tid/8;index<2*records;index+=Warps*4) {
        const uint32_t projection=index/records,within=index%records;
        const uint32_t row=within/kStageBlocks,kb=within%kStageBlocks;
        const uint8_t *block=(projection?up:gate)
            +(uint64_t(row0+row)*blocks+kb0+kb)*34;
        int8_t *stage=(projection?us:gs)+row*kWeightStride;
        reinterpret_cast<int *>(stage+kb*32)[word_base]=
            imparo_sm80_q8_mmq::load_q8_word(block+2,word_base);
        if(word_base==0)
            reinterpret_cast<float *>(stage+4*32)[kb]=
                __half2float(*reinterpret_cast<const __half *>(block));
    }
}

template<uint32_t Tokens>
__device__ __forceinline__ void accumulate(const int8_t *weights,const int8_t *activations,
        float (&partial)[Tokens/2],uint32_t lane,uint32_t pair) {
    const uint32_t row0=pair*16;
#pragma unroll 1
    for(uint32_t qblock=0;qblock<4;++qblock) {
        int af[4];
        imparo_sm80_mmq::load_a_m16n8k32(af,weights+row0*kWeightStride+qblock*32,kWeightStride);
        float dw[2];
#pragma unroll
        for(uint32_t s=0;s<2;++s) {
            const uint32_t row=row0+imparo_sm80_mmq::accumulator_row(lane,s*2);
            dw[s]=reinterpret_cast<const float *>(weights+row*kWeightStride+4*32)[qblock];
        }
#pragma unroll
        for(uint32_t group=0;group<Tokens/32;++group) {
#pragma unroll
            for(uint32_t fragment=0;fragment<4;++fragment) {
                const uint32_t token0=group*32+fragment*8;
                int bf[2];
                imparo_sm80_mmq::load_b_m16n8k32(bf,
                    activations+token0*kActivationStride+qblock*32,kActivationStride);
                float da[2];
#pragma unroll
                for(uint32_t s=0;s<2;++s) {
                    const uint32_t token=token0+imparo_sm80_mmq::accumulator_token(lane,s);
                    da[s]=reinterpret_cast<const float *>(activations+token*kActivationStride
                        +kActivationRecordValues)[qblock];
                }
                int cf[4]={}; imparo_sm80_mmq::mma_m16n8k32(cf,af,bf);
#pragma unroll
                for(uint32_t item=0;item<4;++item)
                    partial[(group*4+fragment)*4+item]+=
                        float(cf[item])*dw[item/2]*da[item%2];
            }
        }
    }
}

template<uint32_t Rows,uint32_t Tokens,uint32_t Warps>
__device__ __forceinline__ void body(int8_t *shared,const uint8_t *gate,const uint8_t *up,
        const BlockQ8_1Mmq *x,BlockQ8_1Mmq *out,
        uint32_t n_in,uint32_t n_out,uint32_t n_tok) {
    static_assert(Warps==2*(Rows/16));
    constexpr uint32_t projection_bytes=Rows*kWeightStride;
    int8_t *gs=shared,*us=gs+projection_bytes,*xs=us+projection_bytes;
    const uint32_t lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;
    const uint32_t pair=warp/2,projection=warp%2;
    const uint32_t row0=blockIdx.x*Rows,tok0=blockIdx.y*Tokens;
    const uint32_t active=min(Tokens,n_tok-tok0);
    float partial[Tokens/2]={};
    for(uint32_t kb=0;kb<n_in/32;kb+=kStageBlocks) {
        imparo_sm80_q8_mmq::stage_activation_group<Tokens,Warps>(
            x,xs,n_tok,tok0,active,kb/4,true,tid);
        imparo_sm80_mmq::commit_async_copies();
        stage_weights<Rows,Warps>(gate,up,gs,us,n_in/32,row0,kb,tid);
        imparo_sm80_mmq::wait_async_copies(); __syncthreads();
        accumulate<Tokens>(projection?us:gs,xs,partial,lane,pair); __syncthreads();
    }
    float *tile=reinterpret_cast<float *>(shared);
    if(projection==0) {
#pragma unroll
        for(uint32_t f=0;f<Tokens/8;++f) {
#pragma unroll
            for(uint32_t item=0;item<4;++item) {
                const uint32_t row=pair*16+imparo_sm80_mmq::accumulator_row(lane,item);
                const uint32_t tok=f*8+imparo_sm80_mmq::accumulator_token(lane,item);
                if(tok<active) tile[tok*Rows+row]=partial[f*4+item];
            }
        }
    }
    __syncthreads();
    if(projection==1) {
#pragma unroll
        for(uint32_t f=0;f<Tokens/8;++f) {
#pragma unroll
            for(uint32_t item=0;item<4;++item) {
                const uint32_t row=pair*16+imparo_sm80_mmq::accumulator_row(lane,item);
                const uint32_t tok=f*8+imparo_sm80_mmq::accumulator_token(lane,item);
                if(tok<active) {const uint32_t off=tok*Rows+row;
                    tile[off]=imparo_cuda_lfm2::silu(tile[off])*partial[f*4+item];}
            }
        }
    }
    __syncthreads();
    const uint32_t units=active*(Rows/32);
    for(uint32_t base=warp*4;base<units;base+=Warps*4) {
        const uint32_t unit=base+lane/8; const bool valid=unit<units;
        const uint32_t safe=valid?unit:0,tok=safe/(Rows/32),block=safe%(Rows/32);
        const float4 v=reinterpret_cast<const float4 *>(tile+tok*Rows)[block*8+lane%8];
        float amax=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
#pragma unroll
        for(int shift=4;shift>0;shift>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,shift,32));
        const float inverse=127.0f/amax;
        const char4 q={int8_t(roundf(v.x*inverse)),int8_t(roundf(v.y*inverse)),
            int8_t(roundf(v.z*inverse)),int8_t(roundf(v.w*inverse))};
        if(valid) {BlockQ8_1Mmq *record=out+uint64_t(row0/128)*n_tok+tok0+tok;
            const uint32_t sub=(row0%128)/32+block;
            reinterpret_cast<char4 *>(record->qs+sub*32)[lane%8]=q;
            if(lane%8==0) record->d[sub]=1.0f/inverse;}
    }
}

__launch_bounds__(256,2)
__global__ void fused(const uint8_t *gate,const uint8_t *up,const BlockQ8_1Mmq *x,
        BlockQ8_1Mmq *out,uint32_t n_in,uint32_t n_out,uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==860
    extern __shared__ __align__(16) int8_t shared[];
    body<kRows,kTokens,kWarps>(shared,gate,up,x,out,n_in,n_out,n_tok);
#endif
}

// T64 duplicates weight staging but exposes twice as many CTAs at short
// prefill. Build both occupancy constraints so the lab can distinguish tile
// geometry from register-pressure effects before any runtime integration.
__launch_bounds__(256,2)
__global__ void fused_t64_b2(const uint8_t *gate,const uint8_t *up,const BlockQ8_1Mmq *x,
        BlockQ8_1Mmq *out,uint32_t n_in,uint32_t n_out,uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==860
    extern __shared__ __align__(16) int8_t shared[];
    body<kRows,kTokens64,kWarps>(shared,gate,up,x,out,n_in,n_out,n_tok);
#endif
}

__launch_bounds__(256,3)
__global__ void fused_t64_b3(const uint8_t *gate,const uint8_t *up,const BlockQ8_1Mmq *x,
        BlockQ8_1Mmq *out,uint32_t n_in,uint32_t n_out,uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==860
    extern __shared__ __align__(16) int8_t shared[];
    body<kRows,kTokens64,kWarps>(shared,gate,up,x,out,n_in,n_out,n_tok);
#endif
}

// R128 retains one T128 tile and the same total resident warp count as two
// R64 CTAs, while halving duplicated activation staging and CTA bookkeeping.
__launch_bounds__(512,1)
__global__ void fused_r128(const uint8_t *gate,const uint8_t *up,const BlockQ8_1Mmq *x,
        BlockQ8_1Mmq *out,uint32_t n_in,uint32_t n_out,uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==860
    extern __shared__ __align__(16) int8_t shared[];
    body<128,kTokens,16>(shared,gate,up,x,out,n_in,n_out,n_tok);
#endif
}

// R32/T128 preserves complete-token weight reuse. Unlike T64 it duplicates
// only the much smaller activation stage across output-row CTAs, while the
// reduced block exposes up to three resident four-warp CTAs on SM86.
__launch_bounds__(128,3)
__global__ void fused_r32(const uint8_t *gate,const uint8_t *up,const BlockQ8_1Mmq *x,
        BlockQ8_1Mmq *out,uint32_t n_in,uint32_t n_out,uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==860
    extern __shared__ __align__(16) int8_t shared[];
    body<32,kTokens,4>(shared,gate,up,x,out,n_in,n_out,n_tok);
#endif
}
} // namespace stage4

// Keep the incumbent R64/T128 CTA and its single canonical-weight traversal,
// but split each projection's token tile across two warps.  Each thread owns
// only 64 tokens (32 accumulators) instead of 128 (64 accumulators).  This is a
// laboratory-only schedule until exact D4 and complete-FFN evidence selects it.
namespace token_half_w16 {
constexpr uint32_t kWarps = 16;

__device__ __forceinline__ void accumulate_projection(
        const int8_t *weights, const int8_t *activations,
        float (&partial)[32], uint32_t lane, uint32_t row_group,
        uint32_t token_half) {
    const uint32_t local_row0 = row_group * 16;
#pragma unroll
    for (uint32_t phase = 0; phase < 2; ++phase) {
        const int8_t *phase_activation = activations
            + phase * P::kTokens * P::kActivationStride
            + token_half * 64 * P::kActivationStride;
#pragma unroll
        for (uint32_t qblock = 0; qblock < 4; ++qblock) {
            const uint32_t weight_qblock = phase * 4 + qblock;
            int af[4];
            imparo_sm80_mmq::load_a_m16n8k32(
                af, weights + local_row0 * P::kWeightStride
                    + weight_qblock * P::kBlockValues,
                P::kWeightStride);
            float d8w[2];
#pragma unroll
            for (uint32_t item = 0; item < 2; ++item) {
                const uint32_t row = local_row0
                    + imparo_sm80_mmq::accumulator_row(lane, item * 2);
                d8w[item] = reinterpret_cast<const float *>(
                    weights + row * P::kWeightStride
                        + imparo_sm80_q8_mmq::kStageValues)[weight_qblock];
            }
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group) {
#pragma unroll
                for (uint32_t fragment = 0; fragment < 4; ++fragment) {
                    const uint32_t token0 = group * 32 + fragment * 8;
                    int bf[2];
                    imparo_sm80_mmq::load_b_m16n8k32(
                        bf, phase_activation + token0 * P::kActivationStride
                            + qblock * P::kBlockValues,
                        P::kActivationStride);
                    float d8a[2];
#pragma unroll
                    for (uint32_t item = 0; item < 2; ++item) {
                        const uint32_t token = token0
                            + imparo_sm80_mmq::accumulator_token(lane, item);
                        d8a[item] = reinterpret_cast<const float *>(
                            phase_activation + token * P::kActivationStride
                                + P::kActivationRecordValues)[qblock];
                    }
                    int cf[4] = {};
                    imparo_sm80_mmq::mma_m16n8k32(cf, af, bf);
#pragma unroll
                    for (uint32_t item = 0; item < 4; ++item)
                        partial[(group * 4 + fragment) * 4 + item]
                            += float(cf[item]) * d8w[item / 2] * d8a[item % 2];
                }
            }
        }
    }
}

template <uint32_t LoadLanes = 2>
__launch_bounds__(512, 1)
__global__ void fused(const uint8_t *gate, const uint8_t *up,
        const BlockQ8_1Mmq *x, BlockQ8_1Mmq *out,
        uint32_t n_in, uint32_t n_out, uint32_t n_tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    extern __shared__ __align__(16) int8_t shared[];
    int8_t *gs = shared;
    int8_t *us = gs + P::kProjectionWeightBytes;
    int8_t *xs = us + P::kProjectionWeightBytes;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32 + lane;
    const uint32_t row_group = warp / 4;
    const uint32_t projection = (warp / 2) & 1u;
    const uint32_t token_half = warp & 1u;
    const uint32_t row0 = blockIdx.x * P::kRows;
    const uint32_t tok0 = blockIdx.y * P::kTokens;
    const uint32_t active = min(P::kTokens, n_tok - tok0);
    float partial[32] = {};
    for (uint32_t kb = 0; kb < n_in / 32; kb += P::kStageBlocks) {
#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase)
            imparo_sm80_q8_mmq::stage_activation_group<P::kTokens, kWarps>(
                x, xs + phase * P::kTokens * P::kActivationStride,
                n_tok, tok0, active, (kb + phase * 4) / 4, true, tid);
        imparo_sm80_mmq::commit_async_copies();
        stage_cooperative<LoadLanes, false, false, kWarps>(
            gate, up, gs, us, n_in / 32, row0, kb, tid);
        imparo_sm80_mmq::wait_async_copies();
        __syncthreads();
        accumulate_projection(projection ? us : gs, xs, partial,
            lane, row_group, token_half);
        __syncthreads();
    }
    float *tile = reinterpret_cast<float *>(shared);
    const uint32_t token_base = token_half * 64;
    if (projection == 0) {
#pragma unroll
        for (uint32_t fragment = 0; fragment < 8; ++fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t row = row_group * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t tok = token_base + fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                if (tok < active)
                    tile[tok * P::kRows + row] = partial[fragment * 4 + item];
            }
        }
    }
    __syncthreads();
    if (projection == 1) {
#pragma unroll
        for (uint32_t fragment = 0; fragment < 8; ++fragment) {
#pragma unroll
            for (uint32_t item = 0; item < 4; ++item) {
                const uint32_t row = row_group * 16
                    + imparo_sm80_mmq::accumulator_row(lane, item);
                const uint32_t tok = token_base + fragment * 8
                    + imparo_sm80_mmq::accumulator_token(lane, item);
                if (tok < active) {
                    const uint32_t offset = tok * P::kRows + row;
                    tile[offset] = imparo_cuda_lfm2::silu(tile[offset])
                        * partial[fragment * 4 + item];
                }
            }
        }
    }
    __syncthreads();
    const uint32_t units = active * (P::kRows / 32);
    for (uint32_t base = warp * 4; base < units; base += kWarps * 4) {
        const uint32_t unit = base + lane / 8;
        const bool valid = unit < units;
        const uint32_t safe = valid ? unit : 0;
        const uint32_t tok = safe / (P::kRows / 32);
        const uint32_t block = safe % (P::kRows / 32);
        const float4 value = reinterpret_cast<const float4 *>(
            tile + tok * P::kRows)[block * 8 + lane % 8];
        float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                           fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
        for (int shift = 4; shift > 0; shift >>= 1)
            amax = fmaxf(amax,
                __shfl_xor_sync(0xffffffff, amax, shift, 32));
        const float inverse = 127.0f / amax;
        const char4 q = {
            int8_t(roundf(value.x * inverse)),
            int8_t(roundf(value.y * inverse)),
            int8_t(roundf(value.z * inverse)),
            int8_t(roundf(value.w * inverse))};
        if (valid) {
            BlockQ8_1Mmq *record = out
                + uint64_t(row0 / 128) * n_tok + tok0 + tok;
            const uint32_t subblock = (row0 % 128) / 32 + block;
            reinterpret_cast<char4 *>(record->qs + subblock * 32)[lane % 8] = q;
            if (lane % 8 == 0) record->d[subblock] = 1.0f / inverse;
        }
    }
#endif
}
} // namespace token_half_w16
} // namespace canonical_q8_sidecar_lab
