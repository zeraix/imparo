#pragma once
// Node ids are packed rows; parent precedes child; depth is a logical position.
// No state is shared between speculative branches. -1 denotes committed prefix.
__device__ float tree_signal(const float*bcx,const float*state,const int*parent,int node,unsigned back,unsigned ch,unsigned width,unsigned history){
 while(back && node>=0){node=parent[node];--back;}
 if(node<0)return state[uint64_t(history-1-back)*width+ch];
 const float*row=bcx+uint64_t(node)*3*width;
 return row[ch]*row[2*width+ch];
}
__global__ void tree_shortconv(const float*bcx,const float*w,const float*state,const int*parent,float*out,float*snap,unsigned width,unsigned kernel,unsigned nt,unsigned state_stride,unsigned state_off){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*nt)return;unsigned node=i/width,ch=i%width,h=kernel-1;float sum=0.f;
 for(unsigned tap=0;tap<kernel;++tap)sum+=w[uint64_t(ch)*kernel+tap]*tree_signal(bcx,state,parent,node,h-tap,ch,width,h);
 out[i]=bcx[uint64_t(node)*3*width+width+ch]*sum;
 for(unsigned slot=0;slot<h;++slot)snap[uint64_t(node)*state_stride+state_off+slot*width+ch]=tree_signal(bcx,state,parent,node,h-1-slot,ch,width,h);
}

__global__ void tree_commit_gather(const uint8_t*src,uint8_t*dst,const int*ids,unsigned start,unsigned rowbytes,unsigned nt,const unsigned*pages){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*rowbytes){unsigned row=imparo_cuda_kv::physical_row(start+ids[i/rowbytes],0,pages);dst[i]=src[uint64_t(row)*rowbytes+i%rowbytes];}}
__global__ void tree_commit_scatter(const uint8_t*src,uint8_t*dst,unsigned start,unsigned rowbytes,unsigned nt,const unsigned*pages){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nt*rowbytes){unsigned row=imparo_cuda_kv::physical_row(start+i/rowbytes,0,pages);dst[uint64_t(row)*rowbytes+i%rowbytes]=src[i];}}
