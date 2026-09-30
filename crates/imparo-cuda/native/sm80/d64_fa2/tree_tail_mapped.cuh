#pragma once
namespace imparo_d64_fa2_port {
namespace mapped_tail_detail {
using namespace tree_tail_detail;
struct alignas(8) TailDescriptor {const __half* k;const __half* v;unsigned begin;unsigned rows[9];};
struct TailAddress {
 const TailDescriptor* d;unsigned prefix;unsigned begin;
 template<class T> __device__ __forceinline__ T* operator()(T*p,unsigned row,unsigned len,unsigned stride)const {
  if(row>=len)return p-int64_t(row)*stride;
  if(row<prefix)return p;
  return p+(int64_t(d->rows[row-prefix])-int64_t(begin+row))*stride;
 }
};
__global__ void pack_mapped(const __half*q,const __half*k,const __half*v,__half*qs,__half*ks,__half*vs,const int*parents,unsigned leaf_mask,unsigned start,unsigned begin,unsigned tail){
 unsigned j=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
 __shared__ int path[9];__shared__ unsigned depth;
 if(threadIdx.x==0){unsigned mask=leaf_mask;for(unsigned n=0;n<j;++n)mask&=mask-1;int leaf=__ffs(mask)-1;unsigned d=0;for(int n=leaf;parents[n]>=0;n=parents[n])++d;depth=d;for(int n=leaf;n>=0;n=parents[n])path[d--]=n;}
 __syncthreads();
 if(i<9*QW){unsigned row=i/QW;unsigned node=unsigned(path[min(row,depth)]);qs[j*9*QW+i]=q[node*QW+i%QW];}
 if(blockIdx.x==0&&threadIdx.x==0){auto*d=reinterpret_cast<TailDescriptor*>(ks+uint64_t(j)*tail*KW);d->k=k;d->v=v;d->begin=begin;for(unsigned row=0;row<9;++row)d->rows[row]=start+unsigned(path[min(row,depth)]);}
}
__global__ __launch_bounds__(KT::NUM_THREADS) void tail_mapped(PP p,unsigned tail){
 static_assert(KT::HEAD_DIM_QK==64&&KT::HEAD_DIM_VO==64&&KT::SWIZZLE_MODE_KV==flashinfer::SwizzleMode::k128B);
 extern __shared__ uint8_t smem[];unsigned j=blockIdx.y;
 auto*d=reinterpret_cast<const TailDescriptor*>(p.k+uint64_t(j)*tail*KW);
 const unsigned begin=d->begin;
 p.q+=j*9*QW;p.k=const_cast<__half*>(d->k)+uint64_t(begin)*KW;p.v=const_cast<__half*>(d->v)+uint64_t(begin)*KW;p.o+=j*9*QW;p.lse+=j*9*32;
 flashinfer::SinglePrefillWithKVCacheDevice<KT>(p,*reinterpret_cast<KT::SharedStorage*>(smem),threadIdx,blockIdx.x,0,blockIdx.z,1,8,TailAddress{d,tail-9,begin});
}
}
cudaError_t launch_tree_tail_mapped(const float*q,const __half*k,const __half*v,float*out,__half*qh,__half*oh,__half*tmp,void*scratch,uint8_t*mask,const int*parents,unsigned start,unsigned fixed_span,unsigned chunks,unsigned begin,unsigned tail,unsigned leaf_mask,float scale,cudaStream_t stream){
 using namespace tree_tail_detail;
 unsigned leaves=0;for(unsigned m=leaf_mask;m;m&=m-1)++leaves;if(!leaves||leaves>16||tail<9)return cudaErrorInvalidValue;
 auto e=launch_mask(q,k,v,out,qh,oh,tmp,16,32,8,start+16,scale,mask,stream,start+9,fixed_span);if(e)return e;
 auto*tq=static_cast<__half*>(scratch);auto*tk=tq+leaves*9*QW;auto*tv=tk+leaves*tail*KW;auto*to=tv+leaves*tail*KW;auto*tl=reinterpret_cast<float*>(to+leaves*9*QW);
 mapped_tail_detail::pack_mapped<<<dim3((9*QW+255)/256,leaves),256,0,stream>>>(qh,k,v,tq,tk,tv,parents,leaf_mask,start,begin,tail);
 PP p(tq,tk,tv,nullptr,to,tl,nullptr,32,8,9,tail,QW,64,KW,64,64,-1,0.f,1.f,1.f,10000.f);p.partition_kv=true;p.partition_fixed_span=fixed_span;
 mapped_tail_detail::tail_mapped<<<dim3(1,leaves,8),dim3(32,4,1),sizeof(KT::SharedStorage),stream>>>(p,tail);e=cudaGetLastError();if(e)return e;
 auto*ls=reinterpret_cast<float*>(tmp+chunks*16*QW);splice<<<dim3((QW+255)/256,16),256,0,stream>>>(to,tl,tmp,ls,parents,leaf_mask,chunks);
 e=flashinfer::MergeStates(tmp,ls,oh,(float*)nullptr,chunks,16,32,64,stream);if(e)return e;
 output_f32<<<(16*QW+255)/256,256,0,stream>>>(oh,out,16*QW);return cudaGetLastError();
}
}

namespace imparo_d64_fa2_port {
namespace batch_invariant_tree {
using namespace tree_tail_detail;
struct PartitionTailAddress {
 const mapped_tail_detail::TailDescriptor*d;unsigned prefix,begin,chunk_start;
 template<class T> __device__ __forceinline__ T* operator()(T*p,unsigned row,unsigned len,unsigned stride)const {
  // produce_kv row/len are chunk-relative while p already includes chunk_start.
  // Rebase both before the retained map. For a predicated-off row its existing
  // subtraction now returns the safe physical begin row, without indexing rows[].
  return mapped_tail_detail::TailAddress{d,prefix,begin}(p,row+chunk_start,len+chunk_start,stride);
 }
};
// Same KT and address map as retained shared-KV tail. x now names the one or
// two tail partitions, and all output/LSE strides carry that physical count.
__global__ __launch_bounds__(KT::NUM_THREADS) void tail(PP p,unsigned rows){
 extern __shared__ uint8_t smem[];const unsigned j=blockIdx.y,parts=(rows+p.partition_fixed_span-1)/p.partition_fixed_span;
 auto*d=reinterpret_cast<const mapped_tail_detail::TailDescriptor*>(p.k+uint64_t(j)*rows*KW);
 const unsigned begin=d->begin;
 p.q+=uint64_t(j)*9*QW;p.k=const_cast<__half*>(d->k)+uint64_t(begin)*KW;p.v=const_cast<__half*>(d->v)+uint64_t(begin)*KW;
 p.o+=uint64_t(j)*9*parts*QW;p.lse+=uint64_t(j)*9*parts*32;
 flashinfer::SinglePrefillWithKVCacheDevice<KT>(p,*reinterpret_cast<KT::SharedStorage*>(smem),threadIdx,0,blockIdx.x,blockIdx.z,parts,8,PartitionTailAddress{d,rows-9,begin,blockIdx.x*p.partition_fixed_span});
}
__global__ void splice_v1(const __half*o,const float*ls,__half*tmp,float*tl,const int*parents,
 unsigned leaf_mask,unsigned chunks,unsigned prefix_chunks,unsigned tail_chunks){
 const unsigned node=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
 __shared__ unsigned source_row;
 if(threadIdx.x==0){unsigned depth=0;for(int n=int(node);parents[n]>=0;n=parents[n])++depth;unsigned mask=leaf_mask,bucket=0;
  while(mask){int leaf=__ffs(mask)-1;bool owns=false;for(int n=leaf;n>=0;n=parents[n])if(unsigned(n)==node)owns=true;if(owns){source_row=bucket*9+depth;break;}mask&=mask-1;++bucket;}}
 __syncthreads();
 for(unsigned c=0;c<tail_chunks;++c){
  if(i<QW)tmp[(uint64_t(node)*chunks+prefix_chunks+c)*QW+i]=o[(uint64_t(source_row)*tail_chunks+c)*QW+i];
  if(i<32)tl[(uint64_t(node)*chunks+prefix_chunks+c)*32+i]=ls[(uint64_t(source_row)*tail_chunks+c)*32+i];
 }
}
}
cudaError_t tree_tail_plan_v1(unsigned start,unsigned span,unsigned leaves,unsigned*chunks,unsigned*begin,unsigned*rows,uint64_t*extra){
 using namespace tree_tail_detail;
 if(!span||span<256||span%128||!leaves||leaves>16||start>UINT32_MAX-16)return cudaErrorInvalidValue;
 *begin=(start/span)*span;*rows=start-*begin+9;*chunks=(start+16+span-1)/span;
 const unsigned parts=(*rows+span-1)/span;
 *extra=uint64_t(leaves)*(9ull*QW*sizeof(__half)+2ull*(*rows)*KW*sizeof(__half)+9ull*parts*(QW*sizeof(__half)+32*sizeof(float)));
 // Attributes must be installed by eager preflight, before a replay capture.
 static bool configured=false;
 if(!configured){auto e=cudaFuncSetAttribute(batch_invariant_tree::tail,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(KT::SharedStorage));if(e)return e;
  auto kernel=flashinfer::SinglePrefillWithKVCacheKernel<CT,PP>;
  e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(CT::SharedStorage));if(e)return e;configured=true;}
 return cudaSuccess;
}
cudaError_t launch_tree_tail_v1(const float*q,const __half*k,const __half*v,float*out,
 __half*qh,__half*oh,__half*tmp,void*scratch,uint8_t*mask,const int*parents,
 unsigned start,unsigned span,unsigned chunks,unsigned begin,unsigned rows,unsigned leaf_mask,float scale,cudaStream_t stream){
 using namespace tree_tail_detail;
 unsigned leaves=0;for(unsigned m=leaf_mask;m;m&=m-1)++leaves;
 if(!leaves||leaves>16||begin>start||begin%span||rows!=start-begin+9||chunks!=(start+16+span-1)/span)return cudaErrorInvalidValue;
 const unsigned prefix=begin/span,parts=(rows+span-1)/span;
 auto*ls=reinterpret_cast<float*>(tmp+uint64_t(chunks)*16*QW);
 prepare_q<<<(16*QW+255)/256,256,0,stream>>>(q,qh,16*QW,scale);
 PP p(qh,const_cast<__half*>(k),const_cast<__half*>(v),mask,tmp,ls,nullptr,32,8,16,start+16,QW,64,KW,64,64,-1,0.f,1.f,1.f,10000.f);
 p.partition_kv=true;p.partition_fixed_span=span;p.batch_invariant_q8_v1=1;
 auto kernel=flashinfer::SinglePrefillWithKVCacheKernel<CT,PP>;void*args[]={&p};
 // The ordinary kernel's output stride is the total number of launched chunks.
 // Its tail values are always replaced; prefix chunks contain only common history.
 auto e=cudaLaunchKernel((void*)kernel,dim3(1,chunks,8),dim3(32,4,1),args,sizeof(CT::SharedStorage),stream);if(e)return e;
 auto*tq=static_cast<__half*>(scratch);auto*tk=tq+uint64_t(leaves)*9*QW;auto*tv=tk+uint64_t(leaves)*rows*KW;
 auto*to=tv+uint64_t(leaves)*rows*KW;auto*tl=reinterpret_cast<float*>(to+uint64_t(leaves)*9*parts*QW);
 mapped_tail_detail::pack_mapped<<<dim3((9*QW+255)/256,leaves),256,0,stream>>>(qh,k,v,tq,tk,tv,parents,leaf_mask,start,begin,rows);
 PP tp(tq,tk,tv,nullptr,to,tl,nullptr,32,8,9,rows,QW,64,KW,64,64,-1,0.f,1.f,1.f,10000.f);
 tp.partition_kv=true;tp.partition_fixed_span=span;tp.batch_invariant_q8_v1=1;
 batch_invariant_tree::tail<<<dim3(parts,leaves,8),dim3(32,4,1),sizeof(KT::SharedStorage),stream>>>(tp,rows);
 batch_invariant_tree::splice_v1<<<dim3((QW+255)/256,16),256,0,stream>>>(to,tl,tmp,ls,parents,leaf_mask,chunks,prefix,parts);
 merge_v1<<<16,dim3(8,32),0,stream>>>(tmp,ls,oh,16,chunks,start,span,parents);
 output_f32<<<(16*QW+255)/256,256,0,stream>>>(oh,out,16*QW);return cudaGetLastError();
}
}
