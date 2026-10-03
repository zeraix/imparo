#pragma once
// Canonical Q6_K reader, identical byte geometry to imparo-cpu/quants.rs.
// Required by the official Q4_0 LFM2-MoE file's tied embedding/output tensor.
__device__ inline float q6_k_value(const uint8_t *b,uint32_t i) {
    const uint32_t half=i/128,l=i%32,group=(i%128)/32;
    const uint8_t low=b[half*64+(group%2)*32+l];
    const uint8_t high=b[128+half*32+l];
    const int q=int(((low>>(group>=2?4:0))&15)|(((high>>(group*2))&3)<<4))-32;
    const int scale=int(int8_t(b[192+half*8+group*2+l/16]));
    const float d=__half2float(*reinterpret_cast<const __half*>(b+208));
    return (d*float(scale))*float(q);
}
__global__ void k_rows_q6_k(const uint8_t *table,const uint32_t *tokens,float *dst,
        uint32_t width,float scale,uint32_t dst_off) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    if(i>=width)return;
    const uint64_t row=tokens?tokens[t]:t;
    const uint8_t *b=table+(row*(width/256)+i/256)*210;
    dst[uint64_t(t)*width+dst_off+i]=q6_k_value(b,i%256)*scale;
}
__global__ void k_gemm_q6_k(const uint8_t *weights,const float *x,float *y,
        uint32_t ni,uint32_t no,uint32_t src_row,uint32_t row_base,uint32_t out_stride) {
    const uint32_t lane=threadIdx.x%32,r=blockIdx.x*4+threadIdx.x/32,t=blockIdx.y;
    if(r>=no)return;
    const uint8_t *wr=weights+uint64_t(r)*(ni/256)*210;
    const float *xr=x+uint64_t(src_row+t)*ni;
    float sum=0;
    for(uint32_t i=lane;i<ni;i+=32)sum=fmaf(q6_k_value(wr+uint64_t(i/256)*210,i%256),xr[i],sum);
    for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!lane)y[uint64_t(t)*out_stride+row_base+r]=sum;
}
static void q6_k_matmat(uint64_t off,uint32_t ni,uint32_t no,uint32_t src,
        uint32_t dst,uint32_t nt,uint32_t src_row) {
    if(!ni || ni%256 || !no || !nt || nt>65535 || src>=B_COUNT || dst>=B_COUNT
       || src==dst || !execution().bufs[src] || !execution().bufs[dst]
       || (uint64_t(src_row)+nt)*ni>execution().sizes[src]/4
       || uint64_t(nt)*no>execution().sizes[dst]/4 || execution().epilogue) {
        set_pending(CUDA_RC_INVALID,"Q6_K projection shape/epilogue");return;
    }
    const uint64_t rb=uint64_t(ni/256)*210,bytes=rb*no;
    if(off%2 || off>g.weights_len || bytes>g.weights_len-off) {
        set_pending(CUDA_RC_INVALID,"Q6_K weight range");return;
    }
    const uint8_t *resident=resident_weight_range(off,bytes);
    const uint32_t chunk=resident?no:uint32_t(std::max<uint64_t>(1,std::min<uint64_t>(no,(16ull<<20)/rb)));
    for(uint32_t row=0;row<no;row+=chunk) {
        const uint32_t count=std::min(chunk,no-row);const uint8_t *w=resident?resident+uint64_t(row)*rb:nullptr;
        if(!resident) {int rc=weight_slice(off+uint64_t(row)*rb,uint64_t(count)*rb,&w);if(rc){set_pending(rc,"Q6_K weight slice");return;}}
        k_gemm_q6_k<<<dim3((count+3)/4,nt),128,0,g.stream>>>(w,(const float*)execution().bufs[src],
            (float*)execution().bufs[dst],ni,count,src_row,row,no);
    }
    mark_buf_written(dst);
}
