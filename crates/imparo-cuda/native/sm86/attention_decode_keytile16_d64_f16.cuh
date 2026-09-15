#pragma once
// Laboratory D64 single-query path; same MMA slots and PV contract as generic attention.
namespace imparo_sm86_keytile_d64 {

__device__ __forceinline__ imparo_sm80_mma::Half16x8 query_direct(const float*q,uint32_t d0,float scale,uint32_t row,uint32_t lane){
 using imparo_sm80_mma::Half16x8;Half16x8 a;unsigned r=lane/4,col=2*(lane%4);
 __half2 lo=__hmul2(__floats2half2_rn(q[d0+col],q[d0+col+1]),__float2half2_rn(scale));
 __half2 hi=__hmul2(__floats2half2_rn(q[d0+col+8],q[d0+col+9]),__float2half2_rn(scale));
 __half2 z=__float2half2_rn(0.f);a.x[0]=row==r?lo:z;a.x[1]=row==r+8?lo:z;a.x[2]=row==r?hi:z;a.x[3]=row==r+8?hi:z;return a;
}
__device__ __forceinline__ float dot_direct(const imparo_sm80_mma::Half16x8*q,const __half*k,uint32_t qr,uint32_t kr,uint32_t lane){
 using namespace imparo_sm80_mma;Float16x16 accum{};unsigned r=lane/4,col=2*(lane%4);__half2 z=__float2half2_rn(0.f);
 #pragma unroll
 for(unsigned g=0;g<4;g++){unsigned d0=g*16;Half16x8 b;__half2 lo=__halves2half2(k[d0+col],k[d0+col+1]),hi=__halves2half2(k[d0+col+8],k[d0+col+9]);b.x[0]=kr==r?lo:z;b.x[1]=kr==r+8?lo:z;b.x[2]=kr==r?hi:z;b.x[3]=kr==r+8?hi:z;mma_qk(accum,q[g],b);}
 bool owns=false;float selected=0.f;
 #pragma unroll
 for(unsigned l=0;l<8;l++){if(fragment_q_column(lane,l)==qr&&fragment_key_row(lane,l)==kr){owns=true;selected=accum.x[l];}}
 unsigned owners=__ballot_sync(0xffffffffu,owns);int owner=__ffs(int(owners))-1;return owner>=0?__shfl_sync(0xffffffffu,selected,owner):0.f;
}

__device__ __forceinline__ imparo_sm80_mma::Float16x16 dot_keytile16(
 const imparo_sm80_mma::Half16x8*q,const __half*k,unsigned width,unsigned offset,unsigned base,unsigned valid,unsigned lane){
 using namespace imparo_sm80_mma;Float16x16 accum{};unsigned r0=base+lane/4,r1=r0+8,col=2*(lane%4);__half2 z=__float2half2_rn(0.f);
 #pragma unroll
 for(unsigned g=0;g<4;g++){
  unsigned d0=g*16+offset+col;Half16x8 b;
  b.x[0]=r0<valid?__halves2half2(k[(uint64_t)r0*width+d0],k[(uint64_t)r0*width+d0+1]):z;
  b.x[1]=r1<valid?__halves2half2(k[(uint64_t)r1*width+d0],k[(uint64_t)r1*width+d0+1]):z;
  b.x[2]=r0<valid?__halves2half2(k[(uint64_t)r0*width+d0+8],k[(uint64_t)r0*width+d0+9]):z;
  b.x[3]=r1<valid?__halves2half2(k[(uint64_t)r1*width+d0+8],k[(uint64_t)r1*width+d0+9]):z;
  mma_qk(accum,q[g],b);
 }return accum;
}
__device__ __forceinline__ float select_keytile16(const imparo_sm80_mma::Float16x16&a,unsigned qr,unsigned kr,unsigned lane){
 using namespace imparo_sm80_mma;bool owns=false;float selected=0.f;
 #pragma unroll
 for(unsigned l=0;l<8;l++){if(fragment_q_column(lane,l)==qr&&fragment_key_row(lane,l)==kr){owns=true;selected=a.x[l];}}
 unsigned owners=__ballot_sync(0xffffffffu,owns);int owner=__ffs(int(owners))-1;return owner>=0?__shfl_sync(0xffffffffu,selected,owner):0.f;
}
__global__ void k_attention_direct(const float * q, const void * kc, const void * vc,
                            float * out, uint32_t head_dim, uint32_t n_heads,
                            uint32_t n_kv, uint32_t kv_width, uint32_t start_pos,
                            float qk_scale, uint32_t window, uint32_t ring, uint32_t n_tok,
                            uint32_t ktype, uint32_t vtype, uint32_t f32_v_accum,
                            const uint32_t * page_table) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    const uint32_t kvh = h / (n_heads / n_kv);
    const uint32_t pos = start_pos + t;
    const uint32_t lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    extern __shared__ float sh[];              // head_dim accumulator + reductions
    float * acc = sh;                          // [head_dim]
    const float * qr = q + ((uint64_t)t * n_heads + h) * head_dim;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) acc[i] = 0.0f;
    __shared__ float m_run, s_run;
    // SM70+ QK probe tile.  llama.cpp's FA path feeds half Q/K into an f32 MMA
    // accumulator; preserving that contract matters because score-(score+offset)
    // retains the tensor-core accumulator's low bits.  A scalar f32 dot is close in
    // real numbers but crosses Q8_1 boundaries in the following WO projection.
    __shared__ __align__(16) __half mma_q[16 * 16];
    __shared__ __align__(16) __half mma_k[16 * 16];
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
    __shared__ __align__(16) float mma_c[16 * 16];
#endif
    __shared__ float mma_score;
    if (threadIdx.x == 0) { m_run = -1e30f; s_run = 0.0f; }
    __syncthreads();
    imparo_sm80_mma::Half16x8 q_cached[4];
    if(ktype==1 && threadIdx.x<32){
        const uint32_t query_row=(t*(n_heads/n_kv)+h%(n_heads/n_kv))&15;
        #pragma unroll
        for(uint32_t g=0;g<4;g++)q_cached[g]=query_direct(qr,g*16,qk_scale,query_row,threadIdx.x);
    }
    imparo_sm80_mma::Float16x16 keytile_cached{};
    for (uint32_t gp = lo; gp <= pos; ++gp) {
        const uint32_t ps = imparo_cuda_kv::physical_row(gp, ring, page_table);
        __shared__ float warp_dot[32];
        if (ktype == 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            if (threadIdx.x < 32) {
                const uint32_t gqa = n_heads / n_kv;
                // Preserve the same register slot as a 16-query x GQA tiled FA:
                // mathematically equal output slots can differ by a few accumulator ULPs.
                const uint32_t q_row = (t * gqa + h % gqa) & 15;
                const uint32_t k_row = gp & 15;
                const __half * kr = static_cast<const __half *>(kc)
                    + uint64_t(ps) * kv_width + kvh * head_dim;
                if ((gp & 15) == 0) keytile_cached=dot_keytile16(q_cached,
                    static_cast<const __half*>(kc),kv_width,kvh*head_dim,gp,pos+1,threadIdx.x);
                const float dot=select_keytile16(keytile_cached,q_row,k_row,threadIdx.x);
                if (threadIdx.x == 0) mma_score = dot;
            }
            __syncthreads();
            warp_dot[0] = mma_score;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            if (threadIdx.x < 32) {
                using namespace nvcuda;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
                wmma::fill_fragment(c, 0.0f);
                for (uint32_t k0 = 0; k0 < head_dim; k0 += 16) {
                    for (uint32_t j = threadIdx.x; j < 16 * 16; j += 32) {
                        mma_q[j] = j < 16 && k0 + j < head_dim
                            ? __hmul(__float2half(qr[k0 + j]), __float2half(qk_scale))
                            : __float2half(0.0f);
                        mma_k[j] = j < 16 && k0 + j < head_dim
                            ? static_cast<const __half *>(kc)[
                                  (uint64_t)ps * kv_width + kvh * head_dim + k0 + j]
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
                if (threadIdx.x == 0) mma_score = mma_c[0];
            }
            __syncthreads();
            warp_dot[0] = mma_score;
#else
            if (threadIdx.x == 0) warp_dot[0] = 0.0f;
#endif
        } else {
            float dot = 0.0f;
            for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
                dot += qr[i] * kv_value(kc, ktype, kv_width, ps, kvh * head_dim + i);
            }
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, off);
            if ((threadIdx.x & 31) == 0) warp_dot[threadIdx.x >> 5] = dot;
            __syncthreads();
            if (threadIdx.x == 0) {
                float d = 0.0f;
                for (uint32_t wi = 0; wi < (blockDim.x + 31) / 32; ++wi) d += warp_dot[wi];
                warp_dot[0] = d;
            }
        }
        __syncthreads();
        const float score = warp_dot[0];
        // Match llama.cpp's f16-MMA Flash Attention numerical contract. Its
        // max is deliberately shifted by 3*log(2), extending the dynamic range
        // of the half VKQ accumulator. The denominator stays f32 while the
        // probabilities and running numerator are f16 MMA operands/accumulators.
        // Omitting either detail is mathematically equivalent in real numbers,
        // but crosses Q8_1 activation boundaries in the following projection.
        constexpr float kFaMaxOffset = 3.0f * 0.6931f;
        const float m_new = fmaxf(m_run, score + kFaMaxOffset);
        const float scale = expf(m_run - m_new);
        const float p = expf(score - m_new);
        for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float vv = kv_value(vc, vtype, kv_width, ps, kvh * head_dim + i);
            if (vtype == 1 && !f32_v_accum) {
                const __half ah = __float2half(acc[i]);
                const __half sh = __float2half(scale);
                const __half ph = __float2half(p);
                const __half vh = __float2half(vv);
                acc[i] = __half2float(__hfma(ph, vh, __hmul(ah, sh)));
            } else {
                acc[i] = __fmaf_rn(p,vv,__fmul_rn(acc[i],scale));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) { s_run = s_run * scale + p; m_run = m_new; }
        __syncthreads();
    }
    float * op = out + ((uint64_t)t * n_heads + h) * head_dim;
    const float inv = s_run > 0.0f ? 1.0f / s_run : 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) op[i] = acc[i] * inv;
}


} // namespace imparo_sm86_keytile_d64
