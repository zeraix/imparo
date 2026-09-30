#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// The codec stores matrix rows in an orthogonal basis. This transform is part of
// model semantics, separate from attention's optional KV-cache Hadamard basis.
namespace imparo_cuda_weight_basis {
__global__ void transform(const float *src, float *dst, const int8_t *signs,
        uint32_t width, uint32_t block, uint32_t inverse,
        uint32_t head_dim, uint32_t key_heads, uint32_t value_heads) {
    extern __shared__ float values[];
    const uint32_t lane = threadIdx.x;
    const uint32_t base = blockIdx.x * block;
    const uint64_t row = uint64_t(blockIdx.y) * width;
    for (uint32_t i=lane; i<block; i+=blockDim.x) {
        uint32_t logical = base+i, source=logical;
        if (head_dim) {
            const uint32_t repeats=value_heads/key_heads;
            const uint32_t h=logical/head_dim;
            source=((h%repeats)*key_heads+h/repeats)*head_dim+logical%head_dim;
        }
        values[i]=src[row+source]*(inverse ? 1.0f : float(signs[logical]));
    }
    __syncthreads();
    for (uint32_t stride=1; stride<block; stride*=2) {
        float next[4];
        for (uint32_t j=0,i=lane; i<block; ++j,i+=blockDim.x) {
            const float a=values[i], b=values[i^stride];
            next[j]=(i&stride) ? b-a : a+b;
        }
        __syncthreads();
        for (uint32_t j=0,i=lane; i<block; ++j,i+=blockDim.x) values[i]=next[j];
        __syncthreads();
    }
    const float scale=rsqrtf(float(block));
    for (uint32_t i=lane; i<block; i+=blockDim.x) {
        const uint32_t logical=base+i;
        dst[row+logical]=values[i]*scale*(inverse ? float(signs[logical]) : 1.0f);
    }
}
inline cudaError_t launch(const float *src,float *dst,const int8_t *signs,
        uint32_t width,uint32_t block,uint32_t rows,bool inverse,
        uint32_t head_dim,uint32_t key_heads,uint32_t value_heads,cudaStream_t stream) {
    if (!src||!dst||!signs||!rows||!width||!block||block>1024
            ||(block&(block-1))||width%block||src==dst
            ||(head_dim && (inverse||!key_heads||!value_heads
                ||value_heads%key_heads||uint64_t(head_dim)*value_heads!=width)))
        return cudaErrorInvalidValue;
    transform<<<dim3(width/block,rows),256,block*sizeof(float),stream>>>(
        src,dst,signs,width,block,inverse,head_dim,key_heads,value_heads);
    return cudaGetLastError();
}

// Only the tiny alpha/beta projections use BF16 in this model. Read them directly;
// expanding the full ternary backbone would destroy the memory advantage.
__global__ void bf16_matmat(const uint16_t *weights,const float *x,float *y,
        uint32_t n_in,uint32_t n_out) {
    const uint32_t output=blockIdx.x,token=blockIdx.y,lane=threadIdx.x;
    float sum=0.0f;
    for(uint32_t k=lane;k<n_in;k+=blockDim.x) {
        const float w=__uint_as_float(uint32_t(weights[uint64_t(output)*n_in+k])<<16);
        sum=fmaf(w,x[uint64_t(token)*n_in+k],sum);
    }
    for(uint32_t mask=16;mask;mask>>=1) sum+=__shfl_down_sync(0xffffffff,sum,mask);
    __shared__ float sums[8];
    if((lane&31)==0)sums[lane>>5]=sum;
    __syncthreads();
    if(lane<32) {
        sum=lane<8?sums[lane]:0;
        for(uint32_t mask=16;mask;mask>>=1)sum+=__shfl_down_sync(0xffffffff,sum,mask);
        if(lane==0)y[uint64_t(token)*n_out+output]=sum;
    }
}
inline cudaError_t launch_bf16(const uint8_t*w,const float*x,float*y,
        uint32_t n_in,uint32_t n_out,uint32_t rows,cudaStream_t stream) {
    if(!w||!x||!y||!n_in||!n_out||!rows)return cudaErrorInvalidValue;
    bf16_matmat<<<dim3(n_out,rows),256,0,stream>>>(
        reinterpret_cast<const uint16_t*>(w),x,y,n_in,n_out);
    return cudaGetLastError();
}
} // namespace imparo_cuda_weight_basis
