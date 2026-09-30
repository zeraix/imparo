#include "../copy_strided.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace cp = imparo_cuda_copy_strided_detail;
static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(rc)); std::exit(2); }
}
struct Device {
    float * p=nullptr; size_t n;
    explicit Device(size_t count):n(count){ck(cudaMalloc(&p,n*4));}
    explicit Device(const std::vector<float>&v):Device(v.size()){ck(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice));}
    ~Device(){cudaFree(p);}
    std::vector<float> read(){std::vector<float>v(n);ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
};
// Exact existing native k_copy body, supplied to the shared helper's fallback.
__global__ void legacy_copy(float*d,const float*s,uint32_t n){
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=s[i];
}
struct Rows {
    uint32_t * calls;
    cudaError_t operator()(float*d,const float*s,uint32_t n,cudaStream_t stream) const {
        ++*calls;
        legacy_copy<<<uint32_t((uint64_t(n)+255)/256),256,0,stream>>>(d,s,n);
        return cudaPeekAtLastError();
    }
};
static std::vector<float> data(size_t n) {
    std::vector<float>v(n);uint32_t x=17;
    for(auto&a:v){x=x*1664525u+1013904223u;uint32_t bits=0x3e800000u|(x&0x007fffffu);std::memcpy(&a,&bits,4);}
    return v;
}
static bool exact(const std::vector<float>&a,const std::vector<float>&b){
    return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0;
}
static bool separate(uint32_t width,uint32_t rows,uint32_t stride,uint32_t off,bool timing=false){
    cp::Layout l;if(!cp::make_layout(width,off,stride,rows,&l))return false;
    const size_t pad=37;
    auto input=data(size_t(l.src_bytes/4)+2*pad);
    std::vector<float> target(size_t(l.dst_bytes/4)+2*pad,-9999.0f),ref=target;
    for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<width;++c)
        ref[pad+size_t(r)*width+c]=input[pad+size_t(r)*stride+off+c];
    Device src(input),dst(target),old(target);uint32_t calls=0;
    ck(cp::launch(dst.p+pad,src.p+pad,l,nullptr,Rows{&calls}));ck(cudaDeviceSynchronize());
    bool ok=exact(dst.read(),ref)&&exact(src.read(),input)&&calls==(stride<width?rows:0);
    std::printf("width=%u rows=%u stride=%u offset=%u bitexact=%u fallback_rows=%u\n",width,rows,stride,off,unsigned(ok),calls);
    if(timing){
        // One small A/B after correctness. Events bracket the complete copy stage;
        // there is no per-row synchronization. This is not whole-model timing.
        cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
        auto run=[&](bool batch){
            uint32_t ncopy=0;ck(cudaEventRecord(a));auto start=std::chrono::steady_clock::now();
            if(batch)ck(cp::launch(dst.p+pad,src.p+pad,l,nullptr,Rows{&ncopy}));
            else for(uint32_t r=0;r<rows;++r)ck(Rows{&ncopy}(old.p+pad+size_t(r)*width,src.p+pad+size_t(r)*stride+off,width,nullptr));
            ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
            double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::printf("copy-stage path=%s gpu_ms=%.6f host_to_done_ms=%.6f row_launches=%u\n",batch?"2d":"old-row",ms,wall,ncopy);
        };
        run(false);run(true);ok=exact(old.read(),ref)&&ok;cudaEventDestroy(a);cudaEventDestroy(b);
    }
    return ok;
}
static bool alias(){
    cp::Layout l;if(!cp::make_layout(16,32,48,7,&l))return false;
    const size_t pad=19;auto original=data(size_t(l.src_bytes/4)+pad*2),ref=original;
    for(uint32_t r=0;r<l.rows;++r)for(uint32_t c=0;c<l.width;++c)
        ref[pad+size_t(r)*l.width+c]=ref[pad+size_t(r)*l.src_stride+l.src_off+c];
    Device memory(original);uint32_t calls=0;
    ck(cp::launch(memory.p+pad,memory.p+pad,l,nullptr,Rows{&calls}));ck(cudaDeviceSynchronize());
    bool ok=calls==l.rows&&exact(memory.read(),ref);
    std::printf("same-buffer-overlap bitexact=%u fallback_rows=%u\n",unsigned(ok),calls);return ok;
}
int main(){
    cp::Layout l;uint32_t calls=0;
    bool ok=cp::make_layout(256,0,512,0,&l)&&l.rows==0
        &&cp::launch(nullptr,nullptr,l,nullptr,Rows{&calls})==cudaSuccess&&calls==0;
    ok=cp::make_layout(0,UINT32_MAX,UINT32_MAX,100,&l)&&l.rows==0&&ok;
    ok=!cp::make_layout(UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,&l)&&ok;
    std::printf("empty-overflow-guards=%u\n",unsigned(ok));
    ok=separate(256,128*24,512,0,true)&&ok;
    ok=separate(256,115*24,512,0)&&ok;
    ok=separate(256,24,512,0)&&ok;
    ok=separate(7,11,19,3)&&ok;
    ok=separate(256,1,UINT32_MAX,7)&&ok;
    ok=separate(8,5,4,2)&&ok;
    ok=alias()&&ok;
    std::printf("copy-strided-probe %s\n",ok?"PASS":"FAIL");return ok?0:1;
}
