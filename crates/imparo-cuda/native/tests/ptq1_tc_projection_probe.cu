#include "../ptq1_tc.cuh"
#include "../weight_basis.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
constexpr unsigned M=128; constexpr float GUARD=1234567.0f;
void ck(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
void req(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T>std::vector<T> read(const fs::path&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);req(bool(f)&&f.tellg()==std::streamoff(n*sizeof(T)),"fixture length");f.seekg(0);std::vector<T>v(n);f.read((char*)v.data(),n*sizeof(T));req(bool(f),"fixture read");return v;}
template<class T>struct Dev{T*p=nullptr;size_t n;Dev(size_t n):n(n){ck(cudaMalloc((void**)&p,n*sizeof(T)));}~Dev(){cudaFree(p);}Dev(const Dev&)=delete;void put(const std::vector<T>&v){req(v.size()==n,"upload length");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
struct Proj{
 std::string name;unsigned k,n;bool perm;Dev<uint8_t>w;Dev<int8_t>signs;Dev<float>x,basis,out;std::vector<uint8_t>hw;std::vector<int8_t>hs;std::vector<float>hx,fullA,fullB;
 Proj(fs::path f,std::string s,unsigned k,unsigned n,bool perm):name(s),k(k),n(n),perm(perm),w(size_t(k/128)*28*n),signs(k),x(size_t(M)*k),basis(x.n),out(size_t(M)*(n+64)+64){hw=read<uint8_t>(f/(s+".ptq.bin"),w.n);hs=read<int8_t>(f/("signs"+std::to_string(k)+".i8.bin"),k);hx=read<float>(f/("input"+std::to_string(k)+".f32.bin"),x.n);w.put(hw);signs.put(hs);x.put(hx);}
 void run(bool tc,unsigned m,bool strided=false){
  ck(imparo_cuda_weight_basis::launch(x.p,basis.p,signs.p,k,1024,m,false,perm?128:0,perm?16:0,perm?48:0,nullptr));
  if(tc)ck(imparo_cuda_ptq1_tc::launch(w.p,basis.p,out.p,k,n,m,strided?n+64:n,strided?32:0,nullptr));
  else ck(imparo_cuda_ptq1::launch_ptq1_matmat(w.p,basis.p,out.p,k,n,m,nullptr));
 }
 void correctness(bool tc,unsigned m,const fs::path&d){
  bool strided=tc&&m==115;unsigned stride=strided?n+64:n,base=strided?32:0;out.put(std::vector<float>(out.n,GUARD));run(tc,m,strided);ck(cudaDeviceSynchronize());auto raw=out.get();std::vector<float>valid(size_t(m)*n);
  for(size_t i=0;i<raw.size();i++){size_t row=i/stride,col=i%stride;if(row<m&&col>=base&&col<base+n){req(std::isfinite(raw[i]),"nonfinite output");valid[row*n+col-base]=raw[i];}else req(std::memcmp(&raw[i],&GUARD,4)==0,"output guard");}
  auto bx=basis.get();for(size_t i=0;i<size_t(m)*k;i++)req(std::isfinite(bx[i])&&std::abs(bx[i])<=65504,"half overflow/basis nonfinite");
  req(w.get()==hw&&signs.get()==hs&&x.get()==hx,"input modified");
  auto&full=tc?fullB:fullA;if(m==M)full=valid;else req(std::memcmp(full.data(),valid.data(),valid.size()*4)==0,"M115 strided prefix differs");
  std::ofstream f(d/((tc?"candidate.":"baseline.")+name+".m"+std::to_string(m)+".f32.bin"),std::ios::binary);f.write((char*)valid.data(),valid.size()*4);req(bool(f),"write result");
 }
};
int main(int argc,char**argv){try{
 req(argc==3,"usage: PROBE FIXTURE OUTPUT");fs::path f=argv[1],d=argv[2];fs::create_directories(d);cudaDeviceProp dev{};ck(cudaGetDeviceProperties(&dev,0));req(dev.major==8&&dev.minor==6,"SM86 fixed candidate");
 std::vector<std::unique_ptr<Proj>> p;
 p.emplace_back(new Proj(f,"gdn_qkv",5120,10240,false));p.emplace_back(new Proj(f,"gdn_gate",5120,6144,false));p.emplace_back(new Proj(f,"gdn_out",6144,5120,true));p.emplace_back(new Proj(f,"attn_q",5120,12288,false));p.emplace_back(new Proj(f,"attn_k",5120,1024,false));p.emplace_back(new Proj(f,"attn_v",5120,1024,false));p.emplace_back(new Proj(f,"attn_out",6144,5120,false));
 auto&first=*p[0];req(imparo_cuda_ptq1_tc::launch(first.w.p,first.basis.p,first.out.p,first.k,first.n,1,first.n,0,nullptr)==cudaErrorNotSupported,"M1 retained");
 for(auto&x:p){x->correctness(false,M,d);x->correctness(true,M,d);}
 auto bundle=[&](bool tc){for(unsigned r=0;r<3;r++)for(unsigned i=0;i<3;i++)p[i]->run(tc,M);for(unsigned i=3;i<p.size();i++)p[i]->run(tc,M);};
 bundle(false);bundle(true);ck(cudaDeviceSynchronize());cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
 auto time=[&](bool tc){ck(cudaEventRecord(a));bundle(tc);ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));return ms;};float ams=time(false),bms=time(true);ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
 for(auto&x:p){x->correctness(false,115,d);x->correctness(true,115,d);}
 std::ofstream r(d/"result.json");r<<std::setprecision(12)<<"{\"scope\":\"one fixed projection bundle; 3 GDN sets and1 attention set; basis included, resident fixture, not complete decoder\",\"tile\":[32,64,128],\"warmups_each\":1,\"timed_calls_each\":1,\"baseline_ms\":"<<ams<<",\"candidate_ms\":"<<bms<<",\"speedup\":"<<ams/bms<<",\"guards_and_finite\":true,\"inputs_immutable\":true,\"m115_prefix_equal\":true,\"m1_rejected\":true,\"independent_oracle\":\"pending frozen CPU check\"}";req(bool(r),"report write");std::cout<<"Projection bundle: "<<ams<<" -> "<<bms<<" ms\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
