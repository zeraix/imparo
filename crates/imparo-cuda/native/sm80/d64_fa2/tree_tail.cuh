#pragma once
namespace imparo_d64_fa2_port {
namespace tree_tail_detail {
using PP=PartitionParams;
using OP=flashinfer::SinglePrefillParams<__half,__half,__half>;
using VV=flashinfer::DefaultAttention<false,false,false,false>;
using CV=flashinfer::DefaultAttention<true,false,false,false>;
using KT=flashinfer::KernelTraits<flashinfer::MaskMode::kCausal,64,1,8,4,4,4,1,flashinfer::PosEncodingMode::kNone,__half,__half,__half,float,int,VV>;
using CT=flashinfer::KernelTraits<flashinfer::MaskMode::kCustom,64,1,8,4,4,4,1,flashinfer::PosEncodingMode::kNone,__half,__half,__half,float,int,CV>;
constexpr unsigned QW=2048,KW=512;

// One canonical nine-row tail per leaf; ancestors share that path's result.
// The leaf mask is derived from the checked host topology, not inferred from row order.
__global__ void pack_tails(const __half*q,const __half*k,const __half*v,__half*qs,__half*ks,__half*vs,const int*parents,unsigned leaf_mask,unsigned start,unsigned begin,unsigned tail){
 unsigned j=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
 __shared__ int path[9];__shared__ unsigned depth;
 if(threadIdx.x==0){unsigned mask=leaf_mask;for(unsigned n=0;n<j;++n)mask&=mask-1;int leaf=__ffs(mask)-1;unsigned d=0;for(int n=leaf;parents[n]>=0;n=parents[n])++d;depth=d;for(int n=leaf;n>=0;n=parents[n])path[d--]=n;}
 __syncthreads();
 if(i<9*QW){unsigned row=i/QW;unsigned node=unsigned(path[min(row,depth)]);qs[j*9*QW+i]=q[node*QW+i%QW];}
 if(i<tail*KW){unsigned row=i/KW,src=begin+row;if(src>=start)src=start+unsigned(path[min(src-start,depth)]);ks[j*tail*KW+i]=k[uint64_t(src)*KW+i%KW];vs[j*tail*KW+i]=v[uint64_t(src)*KW+i%KW];}
}
__global__ __launch_bounds__(KT::NUM_THREADS) void tail_batch(PP p,unsigned tail){
 extern __shared__ uint8_t smem[];unsigned j=blockIdx.y;p.q+=j*9*QW;p.k+=j*tail*KW;p.v+=j*tail*KW;p.o+=j*9*QW;p.lse+=j*9*32;
 flashinfer::SinglePrefillWithKVCacheDevice<KT>(p,*reinterpret_cast<KT::SharedStorage*>(smem),threadIdx,blockIdx.x,0,blockIdx.z,1,8);
}

// Each tree node has one deterministic leaf owner, avoiding concurrent writes
// to a shared ancestor even when several leaf paths contain it.
__global__ void splice(const __half*o,const float*ls,__half*tmp,float*tl,const int*parents,unsigned leaf_mask,unsigned chunks){
 unsigned node=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
 __shared__ unsigned source_row;
 if(threadIdx.x==0){unsigned depth=0;for(int n=int(node);parents[n]>=0;n=parents[n])++depth;unsigned mask=leaf_mask,bucket=0;
  while(mask){int leaf=__ffs(mask)-1;bool owns=false;for(int n=leaf;n>=0;n=parents[n])if(unsigned(n)==node)owns=true;if(owns){source_row=bucket*9+depth;break;}mask&=mask-1;++bucket;}}
 __syncthreads();
 if(i<QW)tmp[(node*chunks+chunks-1)*QW+i]=o[source_row*QW+i];
 if(i<32)tl[(node*chunks+chunks-1)*32+i]=ls[source_row*32+i];
}

}
cudaError_t tree_tail_plan(unsigned start,unsigned fixed_span,unsigned leaves,unsigned*chunks,unsigned*begin,unsigned*tail,uint64_t*extra){
 using namespace tree_tail_detail;
 if(!leaves||leaves>16)return cudaErrorInvalidValue;
 int dev=0,sm=0,nb=0,ntree=0;auto e=cudaGetDevice(&dev);if(e)return e;
 e=cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,dev);if(e)return e;
 auto ordinary_kernel=flashinfer::SinglePrefillWithKVCacheKernel<KT,OP>;auto tree_kernel=flashinfer::SinglePrefillWithKVCacheKernel<CT,PP>;
 e=cudaFuncSetAttribute(ordinary_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(KT::SharedStorage));if(e)return e;
 e=cudaFuncSetAttribute(tree_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(CT::SharedStorage));if(e)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,ordinary_kernel,KT::NUM_THREADS,sizeof(KT::SharedStorage));if(e)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ntree,tree_kernel,CT::NUM_THREADS,sizeof(CT::SharedStorage));if(e)return e;
 if(nb!=ntree||nb*sm<8||start>UINT32_MAX-16)return cudaErrorInvalidValue;
 unsigned maxchunks=unsigned(nb*sm)/8,ref=start+9,chunk_size=std::max((ref+maxchunks-1)/maxchunks,256u);
 *chunks=(ref+chunk_size-1)/chunk_size;if(*chunks<=1)return cudaErrorInvalidValue;
 unsigned span=(ref+*chunks-1)/ *chunks;if(fixed_span){span=fixed_span;*chunks=(ref+span-1)/span;}*begin=(*chunks-1)*span;*tail=ref-*begin;
 if(*begin>start)return cudaErrorInvalidValue;
 *extra=uint64_t(leaves)*(2*9*QW*sizeof(__half)+2*uint64_t(*tail)*KW*sizeof(__half)+9*32*sizeof(float));
 return cudaFuncSetAttribute(tail_batch,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(KT::SharedStorage));
}
cudaError_t launch_tree_tail(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,void*scratch,uint8_t*mask,const int*parents,unsigned start,unsigned fixed_span,unsigned chunks,unsigned begin,unsigned tail,unsigned leaf_mask,float scale,cudaStream_t stream){
 using namespace tree_tail_detail;
 unsigned leaves=0;for(unsigned m=leaf_mask;m;m&=m-1)++leaves;if(!leaves||leaves>16)return cudaErrorInvalidValue;
 auto e=launch_mask(q,k,v,out,qh,oh,tmp,16,32,8,start+16,scale,mask,stream,start+9,fixed_span);if(e)return e;
 auto*tq=static_cast<__half*>(scratch);auto*tk=tq+leaves*9*QW;auto*tv=tk+leaves*tail*KW;auto*to=tv+leaves*tail*KW;auto*tl=reinterpret_cast<float*>(to+leaves*9*QW);
 pack_tails<<<dim3((std::max(9*QW,tail*KW)+255)/256,leaves),256,0,stream>>>(qh,k,v,tq,tk,tv,parents,leaf_mask,start,begin,tail);
 PP p(tq,tk,tv,nullptr,to,tl,nullptr,32,8,9,tail,QW,64,KW,64,64,-1,0.f,1.f,1.f,10000.f);p.partition_kv=true;p.partition_fixed_span=fixed_span;
 tail_batch<<<dim3(1,leaves,8),dim3(32,4,1),sizeof(KT::SharedStorage),stream>>>(p,tail);e=cudaGetLastError();if(e)return e;
 auto*ls=reinterpret_cast<float*>(tmp+chunks*16*QW);splice<<<dim3((QW+255)/256,16),256,0,stream>>>(to,tl,tmp,ls,parents,leaf_mask,chunks);
 e=flashinfer::MergeStates(tmp,ls,oh,(float*)nullptr,chunks,16,32,64,stream);if(e)return e;
 output_f32<<<(16*QW+255)/256,256,0,stream>>>(oh,out,16*QW);return cudaGetLastError();
}
}
