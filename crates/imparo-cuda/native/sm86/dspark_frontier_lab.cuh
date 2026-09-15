#pragma once
namespace imparo_dspark_frontier {
constexpr unsigned K=256,V=128000,M=9,B=4;
struct Node{uint32_t token;int parent,depth;float score;};
struct Top{uint32_t token;float logp;};
struct Part{uint32_t ids[4];float values[4],sum;};
struct State{Part parts[256];Node nodes[33],out[16];Top top[16];int front[4];};
static_assert(sizeof(Node)==16,"node ABI");
template<unsigned C>__global__ void shortk_batch(const uint8_t*w,const BlockQ8_1*x,float*y){
 const unsigned lane=threadIdx.x,row=blockIdx.x*4+threadIdx.y,block=lane/4,iqs=2*(lane&3);float p[C]={};
 if(row<V){const uint8_t*q=w+(uint64_t(row)*8+block)*34;const __half d=*reinterpret_cast<const __half*>(q);
 #pragma unroll
 for(unsigned j=0;j<C;++j)p[j]+=imparo_sm80_q8_mmvq::dot_q8_0_q8_1_half(q+2,d,x+j*8+block,iqs);}
 #pragma unroll
 for(unsigned j=0;j<C;++j){p[j]=__fadd_rn(p[j],0.f);p[j]=__fadd_rn(p[j],0.f);p[j]=__fadd_rn(p[j],0.f);const float v=warp_sum_xor(p[j]);if(lane==0&&row<V)y[j*V+row]=v;}
}
__global__ void add_base(const float*base,const float*bias,float*col,unsigned depth,unsigned count){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count*V)col[i]=base[depth*V+i%V]+bias[i];}
__device__ bool better(float a,unsigned ia,float b,unsigned ib){return a>b||(a==b&&ia<ib);}
__global__ void top_four(const float*col,State*s){
 unsigned row=blockIdx.x,t=threadIdx.x;const float*a=col+row*V;float local[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};unsigned ids[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX};
 for(unsigned i=t;i<V;i+=1024){float v=a[i];for(int j=0;j<4;++j)if(better(v,i,local[j],ids[j])){for(int k=3;k>j;--k){local[k]=local[k-1];ids[k]=ids[k-1];}local[j]=v;ids[j]=i;break;}}
 __shared__ float val[1024],chosen[4],maxv,logz;__shared__ unsigned ix[1024];
 for(unsigned rank=0;rank<4;++rank){val[t]=local[0];ix[t]=ids[0];__syncthreads();for(unsigned d=512;d;d>>=1){if(t<d&&better(val[t+d],ix[t+d],val[t],ix[t])){val[t]=val[t+d];ix[t]=ix[t+d];}__syncthreads();}
  unsigned winner=ix[0];if(t==0){chosen[rank]=val[0];s->top[row*4+rank].token=winner;if(rank==0)maxv=val[0];}__syncthreads();if(ids[0]==winner){for(unsigned j=0;j<3;++j){ids[j]=ids[j+1];local[j]=local[j+1];}ids[3]=UINT32_MAX;local[3]=-INFINITY;}}
 float sum=0;for(unsigned i=t;i<V;i+=1024)sum+=__expf(a[i]-maxv);val[t]=sum;__syncthreads();for(unsigned d=512;d;d>>=1){if(t<d)val[t]+=val[t+d];__syncthreads();}
 if(t==0){logz=logf(val[0]);for(unsigned j=0;j<4;++j)s->top[row*4+j].logp=fminf(0.f,(chosen[j]-maxv)-logz);}
}

__global__ void top_parts(const float*col,State*s){
 unsigned row=blockIdx.y,part=blockIdx.x,t=threadIdx.x,lo=part*2000,hi=min(lo+2000,V);const float*a=col+row*V;float local[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};unsigned ids[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX};
 for(unsigned i=lo+t;i<hi;i+=256){float v=a[i];for(int j=0;j<4;++j)if(better(v,i,local[j],ids[j])){for(int k=3;k>j;--k){local[k]=local[k-1];ids[k]=ids[k-1];}local[j]=v;ids[j]=i;break;}}
 __shared__ float val[256],maxv;__shared__ unsigned ix[256];auto&out=s->parts[row*64+part];
 for(unsigned rank=0;rank<4;++rank){val[t]=local[0];ix[t]=ids[0];__syncthreads();for(unsigned d=128;d;d>>=1){if(t<d&&better(val[t+d],ix[t+d],val[t],ix[t])){val[t]=val[t+d];ix[t]=ix[t+d];}__syncthreads();}unsigned winner=ix[0];if(t==0){out.values[rank]=val[0];out.ids[rank]=winner;if(rank==0)maxv=val[0];}__syncthreads();if(ids[0]==winner){for(unsigned j=0;j<3;++j){ids[j]=ids[j+1];local[j]=local[j+1];}ids[3]=UINT32_MAX;local[3]=-INFINITY;}}
 float sum=0;for(unsigned i=lo+t;i<hi;i+=256)sum+=__expf(a[i]-maxv);val[t]=sum;__syncthreads();for(unsigned d=128;d;d>>=1){if(t<d)val[t]+=val[t+d];__syncthreads();}if(t==0)out.sum=val[0];
}
__global__ void merge_top(State*s){
 unsigned row=blockIdx.x,t=threadIdx.x;const Part*parts=s->parts+row*64;float v=parts[t/4].values[t%4];unsigned id=parts[t/4].ids[t%4];__shared__ float val[256],chosen[4],maxv;__shared__ unsigned ix[256];
 for(unsigned rank=0;rank<4;++rank){val[t]=v;ix[t]=id;__syncthreads();for(unsigned d=128;d;d>>=1){if(t<d&&better(val[t+d],ix[t+d],val[t],ix[t])){val[t]=val[t+d];ix[t]=ix[t+d];}__syncthreads();}unsigned winner=ix[0];if(t==0){chosen[rank]=val[0];s->top[row*4+rank].token=winner;if(rank==0)maxv=val[0];}__syncthreads();if(id==winner){id=UINT32_MAX;v=-INFINITY;}}
 val[t]=t<64?parts[t].sum*__expf(parts[t].values[0]-maxv):0;__syncthreads();for(unsigned d=128;d;d>>=1){if(t<d)val[t]+=val[t+d];__syncthreads();}if(t==0){float z=logf(val[0]);for(unsigned j=0;j<4;++j)s->top[row*4+j].logp=fminf(0.f,(chosen[j]-maxv)-z);}
}

__global__ void initialize(State*s,uint32_t*tok,unsigned anchor){if(threadIdx.x==0){s->nodes[0]={anchor,-1,0,0.f};s->front[0]=0;tok[0]=anchor;}}
__global__ void choose_frontier(State*s,uint32_t*tok,unsigned depth,unsigned rows){if(threadIdx.x)return;int old[4];for(unsigned i=0;i<rows;++i)old[i]=s->front[i];bool used[16]={};
 for(unsigned k=0;k<4;++k){int best=-1;float score=-INFINITY;for(unsigned j=0;j<rows*4;++j)if(!used[j]){float z=s->nodes[old[j/4]].score+s->top[j].logp;if(best<0||z>score||(z==score&&j<unsigned(best))){best=j;score=z;}}used[best]=true;int at=1+depth*4+k;auto top=s->top[best];s->nodes[at]={top.token,old[best/4],int(depth)+1,score};s->front[k]=at;tok[k]=top.token;}}
__global__ void choose_budget(State*s){if(threadIdx.x)return;int map[33];for(int i=0;i<33;++i)map[i]=-1;map[0]=0;s->out[0]=s->nodes[0];
 for(int k=1;k<16;++k){int best=-1;for(int j=1;j<33;++j)if(map[j]<0&&(best<0||s->nodes[j].score>s->nodes[best].score))best=j;map[best]=k;Node n=s->nodes[best];n.parent=map[n.parent];s->out[k]=n;}}

}
