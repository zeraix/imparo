// SPDX-License-Identifier: MIT
#pragma once
// Scheduling over the existing narrow Q8 MMQ primitive. The caller owns the
// bounded partial planes in its existing scratch and orders all work on one stream.
namespace imparo_sm80_q8_mmq {
template<bool TileMajor=false>
__global__ __launch_bounds__(128,1) void m9_parallel_k3_partials(
        const uint8_t *w,const BlockQ8_1Mmq *x,float *p,
        uint32_t ni,uint32_t no,uint32_t nt) {
    const uint32_t part=blockIdx.z,blocks=ni/32/3;
    q8_0_q8_1_mma_tile<16,true,TileMajor,0,false,4,2,64,true>(w,x,
        p+uint64_t(part)*nt*no,nullptr,ni,no,nt,no,0,0,0,nullptr,0,
        blockIdx.x,part*blocks,(part+1)*blocks);
}
__global__ void m9_parallel_k3_combine(const float *p,float *out,uint32_t n) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=(p[i]+p[uint64_t(n)+i])+p[uint64_t(2)*n+i];
}
template<bool TileMajor=false>
inline LaunchResult launch_m9_parallel_k3(const uint8_t *w,
        const BlockQ8_1Mmq *x,float *out,float *p,uint32_t ni,uint32_t no,
        uint32_t nt,cudaStream_t stream,bool batch_invariant=false) {
    if(!w||!x||!out||!p
            ||((nt!=9&&nt!=16)&&!(batch_invariant&&nt>=1&&nt<=16))
            ||ni==0||ni%768||!no||no%64
            ||uint64_t(nt)*no>UINT_MAX)return LaunchResult::NotSupported;
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return LaunchResult::Error;
    static int checked=-1;static bool supported=false;
    if(checked!=device){int major=0,minor=0;
        if(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)!=cudaSuccess
           ||cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)!=cudaSuccess)
            return LaunchResult::Error;
        supported=major==8&&minor==6;checked=device;
    }
    if(!supported)return LaunchResult::NotSupported;
    m9_parallel_k3_partials<TileMajor><<<dim3(no/64,1,3),dim3(32,4),24064,stream>>>(w,x,p,ni,no,nt);
    if(cudaPeekAtLastError()!=cudaSuccess)return LaunchResult::Error;
    m9_parallel_k3_combine<<<(nt*no+255)/256,256,0,stream>>>(p,out,nt*no);
    return cudaPeekAtLastError()==cudaSuccess?LaunchResult::Launched:LaunchResult::Error;
}
} // namespace imparo_sm80_q8_mmq
