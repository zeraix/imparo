// Standalone admission probe. It does not load a model or claim model quality.
// Packed base-3 inputs and direct FP64 matrix/Hadamard oracles are generated
// independently of the CUDA decoders and butterfly implementation under test.
#include "../ptq1.cuh"
#include "../weight_basis.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(rc)); std::exit(2); }
}
template<class T> struct Device {
    T *p=nullptr;
    explicit Device(size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&p),n*sizeof(T))); }
    ~Device() { if(p) cudaFree(p); }
    Device(const Device&)=delete;
    Device&operator=(const Device&)=delete;
};
static uint8_t pack(const std::vector<int> &v) {
    unsigned code=0;
    for(int t:v) code=3*code+unsigned(t+1);
    if(v.size()==4)code*=3;
    return uint8_t((code*256+242)/243);
}
static int ternary(int row,int k) { return int((uint64_t(row+3)*131+uint64_t(k+5)*17+uint64_t(k/7)*29)%3)-1; }
static float weight_scale(int row,int b) { return float(1+((row+b)%4))/32.0f; }
static std::vector<uint8_t> weights(int k,int n) {
    std::vector<uint8_t>w(size_t(k/128)*n*28);
    for(int row=0;row<n;++row)for(int b=0;b<k/128;++b) {
        uint8_t*o=w.data()+(size_t(row)*(k/128)+b)*28;
        for(int j=0;j<16;++j){std::vector<int>v;for(int t=0;t<5;++t)v.push_back(ternary(row,b*128+j+t*16));o[j]=pack(v);}
        for(int j=0;j<8;++j){std::vector<int>v;for(int t=0;t<5;++t)v.push_back(ternary(row,b*128+80+j+t*8));o[16+j]=pack(v);}
        for(int j=0;j<2;++j){std::vector<int>v;for(int t=0;t<4;++t)v.push_back(ternary(row,b*128+120+j+t*2));o[24+j]=pack(v);}
        __half_raw d=__float2half(weight_scale(row,b));o[26]=uint8_t(d.x);o[27]=uint8_t(d.x>>8);
    }
    return w;
}
static void compare(const char*label,const std::vector<float>&got,const std::vector<double>&want,double atol=1e-4,double rtol=2e-5) {
    double worst=0;size_t wi=0;
    for(size_t i=0;i<got.size();++i) {
        const double error=std::abs(double(got[i])-want[i]);if(error>worst){worst=error;wi=i;}
        if(!std::isfinite(got[i])||error>atol+rtol*std::abs(want[i])) {
            std::fprintf(stderr,"FAIL %s index=%zu got=%.9g want=%.17g error=%.9g\n",label,i,got[i],want[i],error);std::exit(1);
        }
    }
    std::printf("PASS %s count=%zu max_abs=%.9g at=%zu\n",label,got.size(),worst,wi);
}
static void matmul_case(int k,int n,int m) {
    auto w=weights(k,n);std::vector<float>x(size_t(k)*m),got(size_t(n)*m);std::vector<double>want(got.size());
    for(size_t i=0;i<x.size();++i)x[i]=float(int((i*19+i/11)%127)-63)/64.0f;
    for(int t=0;t<m;++t)for(int row=0;row<n;++row) {
        double sum=0;for(int j=0;j<k;++j)sum+=double(ternary(row,j))*weight_scale(row,j/128)*x[size_t(t)*k+j];want[size_t(t)*n+row]=sum;
    }
    Device<uint8_t>dw(w.size());Device<float>dx(x.size()),dy(got.size());
    check(cudaMemcpy(dw.p,w.data(),w.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    check(imparo_cuda_ptq1::launch_ptq1_matmat(dw.p,dx.p,dy.p,k,n,m,nullptr));
    check(cudaMemcpy(got.data(),dy.p,got.size()*4,cudaMemcpyDeviceToHost));
    char name[100];std::snprintf(name,sizeof(name),"PTQ1_0 matmul M=%d N=%d K=%d",m,n,k);compare(name,got,want);
    // Nonsequential rows and an invalid id verify addressing and the gather guard.
    std::vector<uint32_t>ids={uint32_t(n-1),0,uint32_t(n)};Device<uint32_t>di(ids.size());
    std::vector<float>rows(ids.size()*k);std::vector<double>rw(rows.size());Device<float>dr(rows.size());
    check(cudaMemcpy(di.p,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
    check(imparo_cuda_ptq1::launch_ptq1_rows(dw.p,di.p,dr.p,k,n,int(ids.size()),0.5f,nullptr));
    check(cudaMemcpy(rows.data(),dr.p,rows.size()*4,cudaMemcpyDeviceToHost));
    for(size_t t=0;t<ids.size();++t)for(int j=0;j<k;++j)rw[t*k+j]=ids[t]<unsigned(n)?0.5*ternary(ids[t],j)*weight_scale(ids[t],j/128):0;
    compare("PTQ1_0 row gather",rows,rw,0,0);
    check(imparo_cuda_ptq1::launch_ptq1_row(dw.p,dr.p,k,n,n-1,0.5f,nullptr));
    check(cudaMemcpy(rows.data(),dr.p,size_t(k)*4,cudaMemcpyDeviceToHost));rows.resize(k);rw.resize(k);compare("PTQ1_0 single row",rows,rw,0,0);
}
static int parity(unsigned v) {int bit=0;while(v){bit^=int(v&1u);v>>=1;}return bit;}
static void transform_case(int width,bool inverse,bool grouped) {
    const int rows=2,block=1024;std::vector<float>x(size_t(width)*rows),got(x.size());std::vector<int8_t>signs(width);std::vector<double>want(x.size());
    for(size_t i=0;i<x.size();++i)x[i]=float(int((i*29+i/3)%251)-125)/128.0f;
    for(int j=0;j<width;++j)signs[j]=((j*13+j/7)%3)?1:-1;
    // Construct the permutation by explicit [rep][key][lane] -> [key][rep][lane]
    // scatter, independent of the GPU's destination-to-source index formula.
    std::vector<float>permuted=x;
    if(grouped)for(int t=0;t<rows;++t)for(int rep=0;rep<3;++rep)for(int key=0;key<16;++key)for(int d=0;d<128;++d)
        permuted[size_t(t)*width+(key*3+rep)*128+d]=x[size_t(t)*width+(rep*16+key)*128+d];
    for(int t=0;t<rows;++t)for(int base=0;base<width;base+=block)for(int i=0;i<block;++i) {
        double sum=0;for(int j=0;j<block;++j)sum+=(parity(unsigned(i&j))?-1.0:1.0)*permuted[size_t(t)*width+base+j]*(inverse?1:signs[base+j]);
        want[size_t(t)*width+base+i]=sum/32.0*(inverse?signs[base+i]:1);
    }
    Device<float>dx(x.size()),dy(x.size());Device<int8_t>ds(signs.size());
    check(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(ds.p,signs.data(),signs.size(),cudaMemcpyHostToDevice));
    check(imparo_cuda_weight_basis::launch(dx.p,dy.p,ds.p,width,block,rows,inverse,grouped?128:0,grouped?16:0,grouped?48:0,nullptr));
    check(cudaMemcpy(got.data(),dy.p,got.size()*4,cudaMemcpyDeviceToHost));
    char name[100];std::snprintf(name,sizeof(name),"basis width=%d inverse=%d grouped=%d",width,int(inverse),int(grouped));compare(name,got,want,1e-5,1e-6);
}
static void bf16_case() {
    const int k=5120,n=48,m=3;std::vector<uint16_t>w(size_t(k)*n);std::vector<float>x(size_t(k)*m),got(size_t(n)*m);std::vector<double>want(got.size());
    for(size_t i=0;i<w.size();++i)w[i]=uint16_t((i%2?0xbf00:0x3e80)+(i%8));
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%37)-18)/32;
    for(int t=0;t<m;++t)for(int row=0;row<n;++row){double s=0;for(int j=0;j<k;++j){union {uint32_t u;float f;}v;v.u=uint32_t(w[size_t(row)*k+j])<<16;s+=double(v.f)*x[size_t(t)*k+j];}want[size_t(t)*n+row]=s;}
    Device<uint16_t>dw(w.size());Device<float>dx(x.size()),dy(got.size());check(cudaMemcpy(dw.p,w.data(),w.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    check(imparo_cuda_weight_basis::launch_bf16(reinterpret_cast<const uint8_t*>(dw.p),dx.p,dy.p,k,n,m,nullptr));check(cudaMemcpy(got.data(),dy.p,got.size()*4,cudaMemcpyDeviceToHost));compare("BF16 alpha/beta real shape",got,want);
}
int main() {
    matmul_case(17408,5120,1); // Actual Down projection dimensions.
    matmul_case(5120,17,3); // Same real input width, row and token tails.
    for(int width:{5120,6144,17408})transform_case(width,false,false);
    transform_case(5120,true,false); // Latent embedding restores H then signs.
    transform_case(6144,false,true); // GDN ssm_out feature permutation.
    bf16_case();
    check(cudaDeviceSynchronize());std::puts("PTQ1/BF16/basis probe completed");return 0;
}
