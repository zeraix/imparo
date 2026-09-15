#pragma once
// Static experiment: preserve the captured native model and replace only
// model-declared, numerically matched FFN spans. No forward/capture guard bypass.
namespace owner_ffn_lab {
constexpr uint32_t MaxOwners = 5, MaxReady = 4;
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    Graph() = default;
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;
    Graph(Graph && v) noexcept : graph(std::exchange(v.graph,nullptr)),
        exec(std::exchange(v.exec,nullptr)) {}
    bool clear() {
        if (exec && cudaGraphExecDestroy(exec) != cudaSuccess) return false;
        exec = nullptr;
        if (graph && cudaGraphDestroy(graph) != cudaSuccess) return false;
        graph = nullptr;
        return true;
    }
    ~Graph() { (void)clear(); }
};
struct Member {
    uint64_t id = 0, generation = 0;
    ExecutionState * state = nullptr;
    cudaGraph_t parent = nullptr;
    std::vector<Graph> phases;
    Graph head_pre, head_post;
};
struct Rows { const float * input[MaxOwners]{}; float * output[MaxOwners]{}; };
struct Group {
    uint64_t id = 0, choice_epoch = 0;
    const void * weights = nullptr;
    std::vector<Member> members;
    std::vector<std::array<Graph,3>> ffn; // B2/B3/B4; B1 keeps complete parent.
    float * x = nullptr, * gate = nullptr, * output = nullptr;
    BlockQ8_1 * q8 = nullptr;
    uint32_t * map = nullptr, * host_map = nullptr, * host_output = nullptr;
    uint32_t max_in = 0, max_mid = 0, max_out = 0, max_quant = 0;
    bool leased = false, poisoned = false;
    bool fused_b2_gate_up = false;
    bool paired_head = false;
    Graph head;
    Rows head_rows;
    const uint8_t * head_weights = nullptr;
    float * head_output = nullptr;
    size_t parent_nodes = 0, phase_nodes = 0, ffn_nodes = 0;
    bool clear() {
        if (!head.clear()) return false;
        for (auto & m : members) {
            if (!m.head_pre.clear() || !m.head_post.clear()) return false;
            for (auto & p : m.phases) if (!p.clear()) return false;
        }
        for (auto & layer : ffn) for (auto & p : layer) if (!p.clear()) return false;
        const auto device = [](auto *& p) { if(p && cudaFree(p)!=cudaSuccess)return false;p=nullptr;return true; };
        const auto host = [](auto *& p) { if(p && cudaFreeHost(p)!=cudaSuccess)return false;p=nullptr;return true; };
        if (!device(head_output)||!device(x)||!device(gate)||!device(output)||!device(q8)||!device(map)
                ||!host(host_map)||!host(host_output)) return false;
        if (leased) { for(auto & m:members)--m.state->graph_leases; leased=false; }
        return true;
    }
    ~Group() { (void)clear(); }
};
std::unique_ptr<Group> group;
uint64_t next_group_id = 1;

__global__ void gather(Rows rows,const uint32_t * map,float * dst,uint32_t width) {
    uint32_t i=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(i<width)dst[uint64_t(t)*width+i]=rows.input[map[t]][i];
}
__global__ void scatter(Rows rows,const uint32_t * map,const float * src,uint32_t width) {
    uint32_t i=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(i<width)rows.output[map[t]][i]=src[uint64_t(t)*width+i];
}
// Default-off exact E4B B2 schedule; miniature gate recorded in the structural ledger.
__global__ void fused_b2_gate_up(const uint8_t*g,const uint8_t*u,const BlockQ8_1*x,float*y){
 const uint32_t lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane,row0=2*blockIdx.x,iqs=2*(tid&1);float pg[2][2]={},pu[2][2]={};
 for(uint32_t block=tid/2;block<80;block+=64){
 #pragma unroll
 for(int t=0;t<2;++t){
 #pragma unroll
 for(int r=0;r<2;++r){uint64_t off=(uint64_t(row0+r)*80+block)*18;pg[t][r]+=dot_q4_0_q8_1_half(g+off,x+t*80+block,iqs);pu[t][r]+=dot_q4_0_q8_1_half(u+off,x+t*80+block,iqs);}}}
 __shared__ float sg[3][2][2][32],su[3][2][2][32];
 if(warp>0){
 #pragma unroll
 for(int t=0;t<2;++t){
 #pragma unroll
 for(int r=0;r<2;++r){sg[warp-1][t][r][lane]=pg[t][r];su[warp-1][t][r][lane]=pu[t][r];}}}
 __syncthreads();if(warp>0)return;
 #pragma unroll
 for(int t=0;t<2;++t){
 #pragma unroll
 for(int r=0;r<2;++r){
 #pragma unroll
 for(int w=0;w<3;++w){pg[t][r]+=sg[w][t][r][lane];pu[t][r]+=su[w][t][r][lane];}
 float a=warp_sum_xor(pg[t][r]),b=warp_sum_xor(pu[t][r]);if(lane==r)y[t*10240+row0+r]=cuda_gelu(a)*b;}}
}

template<uint32_t B>
void launch_ffn(Group & c,const FfnPhaseWire & w,Rows rows,
        const uint8_t * gate,const uint8_t * up,const uint8_t * down) {
    namespace Raw = imparo_sm80_mmvq;
    gather<<<dim3((w.n_in+255)/256,B),256,0,g.stream>>>(rows,c.map,c.x,w.n_in);
    Raw::quantize_q8_1<<<dim3((w.n_in+255)/256,B),256,0,g.stream>>>(c.x,c.q8,w.n_in,B,0);
    if (B == 2 && c.fused_b2_gate_up && w.n_in == 2560 && w.n_mid == 10240) {
        fused_b2_gate_up<<<5120,dim3(32,4),0,g.stream>>>(gate,up,c.q8,c.gate);
        if (std::getenv("IMPARO_GROUP_B2_GATE_UP_TRACE"))
            std::fprintf(stderr,"[group-b2-gate-up] captured fused exact B2 2560->10240\n");
    } else {
    Raw::launch<B>(gate,c.q8,c.gate,w.n_in,w.n_mid,0,w.n_mid,0,4,2,g.stream);
    Raw::launch<B>(up,c.q8,c.gate,w.n_in,w.n_mid,1,w.n_mid,0,4,2,g.stream);
    }
    Raw::quantize_q8_1<<<dim3((w.n_mid+255)/256,B),256,0,g.stream>>>(c.gate,c.q8,w.n_mid,B,0);
    Raw::launch<B>(down,c.q8,c.output,w.n_mid,w.n_out,0,w.n_out,0,4,2,g.stream);
    scatter<<<dim3((w.n_out+255)/256,B),256,0,g.stream>>>(rows,c.map,c.output,w.n_out);
}
bool linear_nodes(cudaGraph_t graph,std::vector<cudaGraphNode_t> & ordered) {
    size_t count=0,roots=0;
    if(cudaGraphGetNodes(graph,nullptr,&count)!=cudaSuccess || !count
        ||cudaGraphGetRootNodes(graph,nullptr,&roots)!=cudaSuccess || roots!=1)return false;
    cudaGraphNode_t at=nullptr;
    if(cudaGraphGetRootNodes(graph,&at,&roots)!=cudaSuccess)return false;
    ordered.reserve(count);
    while(at && ordered.size()<count) {
        ordered.push_back(at);size_t children=0;
        if(cudaGraphNodeGetDependentNodes(at,nullptr,&children)!=cudaSuccess||children>1)return false;
        cudaGraphNode_t next=nullptr;
        if(children && cudaGraphNodeGetDependentNodes(at,&next,&children)!=cudaSuccess)return false;
        at=next;
    }
    return !at && ordered.size()==count;
}
bool clone_span(Graph & out,const std::vector<cudaGraphNode_t> & nodes,size_t first,size_t last) {
    if(first>=last || last>nodes.size() || cudaGraphCreate(&out.graph,0)!=cudaSuccess)return false;
    cudaGraphNode_t prior=nullptr;
    for(size_t i=first;i<last;++i) {
        cudaGraphNodeType type;cudaGraphNode_t fresh=nullptr;
        if(cudaGraphNodeGetType(nodes[i],&type)!=cudaSuccess)return false;
        auto * deps=prior?&prior:nullptr;size_t n=prior?1:0;cudaError_t err=cudaErrorInvalidValue;
        if(type==cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams p{};
            if(cudaGraphKernelNodeGetParams(nodes[i],&p)!=cudaSuccess)return false;
            err=cudaGraphAddKernelNode(&fresh,out.graph,deps,n,&p);
        } else if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms p{};
            if(cudaGraphMemcpyNodeGetParams(nodes[i],&p)!=cudaSuccess)return false;
            err=cudaGraphAddMemcpyNode(&fresh,out.graph,deps,n,&p);
        } else if(type==cudaGraphNodeTypeMemset) {
            cudaMemsetParams p{};
            if(cudaGraphMemsetNodeGetParams(nodes[i],&p)!=cudaSuccess)return false;
            err=cudaGraphAddMemsetNode(&fresh,out.graph,deps,n,&p);
        } else if(type==cudaGraphNodeTypeEmpty) {
            err=cudaGraphAddEmptyNode(&fresh,out.graph,deps,n);
        }
        if(err!=cudaSuccess)return false;
        prior=fresh;
    }
    return cudaGraphInstantiate(&out.exec,out.graph,nullptr,nullptr,0)==cudaSuccess;
}
bool matched_ffn(const std::vector<cudaGraphNode_t>& nodes,size_t first,size_t last) {
    namespace Raw=imparo_sm80_mmvq;
    uint32_t gated=0,down=0,quant=0;
    for(size_t i=first;i<last;++i) {
        cudaGraphNodeType t;cudaKernelNodeParams p{};
        if(cudaGraphNodeGetType(nodes[i],&t)!=cudaSuccess||t!=cudaGraphNodeTypeKernel
                ||cudaGraphKernelNodeGetParams(nodes[i],&p)!=cudaSuccess)return false;
        if(graph_kernel_is(p.func,(void*)Raw::q4_q8_1_gated_decode<4>))++gated;
        else if(graph_kernel_is(p.func,(void*)Raw::q4_q8_1_decode<4>))++down;
        else if(graph_kernel_is(p.func,(void*)Raw::quantize_q8_1))++quant;
        else {std::fprintf(stderr,"[owner-ffn] unsupported FFN kernel=%p\n",p.func);return false;}
    }
    return gated==1&&down==1&&quant<=2;
}

// Exact existing head path only. The private pre-phase keeps final norm/PLE;
// the private post-phase keeps native logits, softcap, argmax and Tmp identity.
template<class T> T head_arg(const cudaKernelNodeParams & p,size_t i) {
    return *reinterpret_cast<const T*>(p.kernelParams[i]);
}
bool split_head(Member & m,const std::vector<cudaGraphNode_t>& nodes,size_t from,
        Group & c,uint32_t owner) {
    namespace Raw=imparo_sm80_mmvq;
    if(nodes.size()<from+5)return false;
    const size_t first=nodes.size()-4;
    cudaKernelNodeParams p[4]{};
    for(size_t j=0;j<4;++j) {
        cudaGraphNodeType type;
        if(cudaGraphNodeGetType(nodes[first+j],&type)!=cudaSuccess||type!=cudaGraphNodeTypeKernel
            ||cudaGraphKernelNodeGetParams(nodes[first+j],&p[j])!=cudaSuccess
            ||!p[j].kernelParams||p[j].extra)return false;
    }
    if(!graph_kernel_is(p[0].func,(void*)Raw::quantize_q8_1)
        ||!graph_kernel_is(p[1].func,(void*)Raw::q4_q8_1_decode<4>)
        ||!graph_kernel_is(p[2].func,(void*)k_softcap)
        ||!graph_kernel_is(p[3].func,(void*)k_argmax_rows<1024>))return false;
    const auto * x=head_arg<const float*>(p[0],0);
    const auto * q=head_arg<const BlockQ8_1*>(p[0],1);
    const auto * w=head_arg<const uint8_t*>(p[1],0);
    auto * y=head_arg<float*>(p[1],2);
    if(x!=m.state->bufs[0]||y!=m.state->bufs[12]||!q||!w
        ||m.state->sizes[0]<2560*4||m.state->sizes[12]<262144*4||m.state->sizes[14]<4
        ||head_arg<uint32_t>(p[0],2)!=2560||head_arg<uint32_t>(p[0],3)!=1||head_arg<uint32_t>(p[0],4)!=0
        ||head_arg<const BlockQ8_1*>(p[1],1)!=q||head_arg<uint32_t>(p[1],3)!=2560
        ||head_arg<uint32_t>(p[1],4)!=262144||head_arg<uint32_t>(p[1],5)!=0
        ||head_arg<float*>(p[2],0)!=y||head_arg<float>(p[2],1)!=30.f||head_arg<uint32_t>(p[2],2)!=262144
        ||head_arg<const float*>(p[3],0)!=y||head_arg<uint32_t*>(p[3],1)!=m.state->bufs[14]
        ||head_arg<uint32_t>(p[3],2)!=262144||head_arg<uint32_t>(p[3],3)!=1
        ||(c.head_weights&&c.head_weights!=w))return false;
    if(p[0].gridDim.x!=10||p[0].gridDim.y!=1||p[0].gridDim.z!=1
        ||p[0].blockDim.x!=256||p[0].blockDim.y!=1||p[0].blockDim.z!=1
        ||p[1].gridDim.x!=262144||p[1].gridDim.y!=1||p[1].gridDim.z!=1
        ||p[1].blockDim.x!=32||p[1].blockDim.y!=4||p[1].blockDim.z!=1)return false;
    c.head_weights=w;c.head_rows.input[owner]=x;c.head_rows.output[owner]=y;
    return clone_span(m.head_pre,nodes,from,first)&&clone_span(m.head_post,nodes,first+2,nodes.size());
}
void launch_head(Group & c) {
    namespace Raw=imparo_sm80_mmvq;
    gather<<<dim3(10,2),256,0,g.stream>>>(c.head_rows,c.map,c.x,2560);
    Raw::quantize_q8_1<<<dim3(10,2),256,0,g.stream>>>(c.x,c.q8,2560,2,0);
    Raw::launch<2>(c.head_weights,c.q8,c.head_output,2560,262144,0,262144,0,4,2,g.stream);
    scatter<<<dim3(1024,2),256,0,g.stream>>>(c.head_rows,c.map,c.head_output,262144);
}

int create(const uint64_t * ids,uint32_t count,uint64_t * out) noexcept try {
    if(!ids||!out||count<2||count>MaxOwners||group||!next_group_id
            ||active_execution_owner_id!=0||!execution_boundary_closed()
            ||g.sm_version!=86||!g.weights_resident||!decode_device_control_enabled()
            ||program_catalog.poisoned||!program_catalog.modules.empty()
            ||!program_catalog.functions.empty()||!program_catalog.bindings.empty())return CUDA_RC_INVALID;
    auto c=std::make_unique<Group>();c->id=next_group_id;c->choice_epoch=g.choice_epoch;c->weights=g.weights;
    const char * paired = std::getenv("IMPARO_GROUP_B2_GATE_UP_LAB");
    c->fused_b2_gate_up = paired && std::strcmp(paired,"1") == 0;
    const char * head_mode = std::getenv("IMPARO_GROUP_B2_HEAD_LAB");
    c->paired_head = head_mode && std::strcmp(head_mode,"1") == 0;
    size_t layers=0;
    for(uint32_t i=0;i<count;++i) {
        for(uint32_t j=0;j<i;++j)if(ids[i]==ids[j])return CUDA_RC_INVALID;
        auto * s=find_execution_owner(ids[i]);
        if(!ids[i]||!s||s->graph_leases||s->forward_open||s->forward_active||s->pending_error
            ||!s->decode_graph_exec||!s->decode_graph||!s->decode_argmax||s->decode_row_bytes
            ||!s->graph_capture_compatible||s->captured_ffn_phases.empty())return CUDA_RC_INVALID;
        if(i==0)layers=s->captured_ffn_phases.size();
        if(layers!=s->captured_ffn_phases.size())return CUDA_RC_INVALID;
        Member m;m.id=ids[i];m.state=s;m.generation=s->graph_capture_generation;m.parent=s->decode_graph;
        std::vector<cudaGraphNode_t> nodes;
        if(!linear_nodes(s->decode_graph,nodes)) {
            std::fprintf(stderr,"[owner-ffn] captured graph is not a supported single-stream chain\n");return CUDA_RC_INVALID;
        }
        c->parent_nodes+=nodes.size();m.phases.resize(layers+1);size_t from=0;
        for(size_t l=0;l<layers;++l) {
            const auto & f=s->captured_ffn_phases[l];const auto & w=f.wire;
            if(w.layer!=l||w.tokens!=1||w.activation!=1||w.gate_kind!=1||w.up_kind!=1||w.down_kind!=1
                ||!w.n_in||!w.n_mid||!w.n_out||w.n_in%128||w.n_mid%128||w.n_out%128
                ||w.src>=B_COUNT||w.dst>=B_COUNT||s->sizes[w.src]<uint64_t(w.n_in)*4
                ||s->sizes[w.dst]<uint64_t(w.n_out)*4)return CUDA_RC_INVALID;
            if(i && std::memcmp(&w,&c->members[0].state->captured_ffn_phases[l].wire,sizeof(w)))return CUDA_RC_INVALID;
            auto pre=std::find(nodes.begin()+from,nodes.end(),f.before);
            auto post=std::find(nodes.begin()+from,nodes.end(),f.after);
            if(pre==nodes.end()||post==nodes.end()||pre>=post)return CUDA_RC_INVALID;
            size_t begin=size_t(pre-nodes.begin())+1,end=size_t(post-nodes.begin())+1;
            if(!matched_ffn(nodes,begin,end))return CUDA_RC_INVALID;
            if(!clone_span(m.phases[l],nodes,from,begin))return CUDA_RC_ERROR;
            c->phase_nodes+=begin-from;from=end;
            c->max_in=std::max(c->max_in,w.n_in);c->max_mid=std::max(c->max_mid,w.n_mid);
            c->max_out=std::max(c->max_out,w.n_out);c->max_quant=std::max(c->max_quant,std::max(w.n_in,w.n_mid));
        }
        if(c->paired_head && !split_head(m,nodes,from,*c,i)) {
            std::fprintf(stderr,"[group-b2-head] unsupported final capture owner=%u\n",i);return CUDA_RC_INVALID;
        }
        if(!clone_span(m.phases[layers],nodes,from,nodes.size()))return CUDA_RC_ERROR;
        c->phase_nodes+=nodes.size()-from;c->members.push_back(std::move(m));
    }
    if(cudaStreamSynchronize(g.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    if(cudaMalloc((void**)&c->x,MaxReady*uint64_t(c->max_in)*4)!=cudaSuccess
        ||cudaMalloc((void**)&c->gate,MaxReady*uint64_t(c->max_mid)*4)!=cudaSuccess
        ||cudaMalloc((void**)&c->output,MaxReady*uint64_t(c->max_out)*4)!=cudaSuccess
        ||cudaMalloc((void**)&c->q8,MaxReady*uint64_t(c->max_quant/32)*sizeof(BlockQ8_1))!=cudaSuccess
        ||cudaMalloc((void**)&c->map,MaxReady*4)!=cudaSuccess
        ||cudaHostAlloc((void**)&c->host_map,MaxReady*4,cudaHostAllocDefault)!=cudaSuccess
        ||cudaHostAlloc((void**)&c->host_output,MaxReady*4,cudaHostAllocDefault)!=cudaSuccess)return CUDA_RC_OOM;
    c->ffn.resize(layers);
    for(size_t l=0;l<layers;++l) {
        const auto & w=c->members[0].state->captured_ffn_phases[l].wire;
        const uint8_t * weights[3]{};
        uint64_t offsets[3]={w.gate_off,w.up_off,w.down_off};
        uint64_t bytes[3]={uint64_t(w.n_in/32)*w.n_mid*18,uint64_t(w.n_in/32)*w.n_mid*18,uint64_t(w.n_mid/32)*w.n_out*18};
        for(int j=0;j<3;++j) {
            if(!resident_weight_range(offsets[j],bytes[j])
                ||weight_slice(offsets[j],bytes[j],&weights[j]))return CUDA_RC_INVALID;
        }
        Rows rows;for(uint32_t i=0;i<count;++i) {
            rows.input[i]=(const float*)c->members[i].state->bufs[w.src];
            rows.output[i]=(float*)c->members[i].state->bufs[w.dst];
        }
        for(int b=2;b<=4;++b) {
            auto & dst=c->ffn[l][b-2];
            if(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return CUDA_RC_ERROR;
            if(b==2)launch_ffn<2>(*c,w,rows,weights[0],weights[1],weights[2]);
            if(b==3)launch_ffn<3>(*c,w,rows,weights[0],weights[1],weights[2]);
            if(b==4)launch_ffn<4>(*c,w,rows,weights[0],weights[1],weights[2]);
            cudaError_t e=cudaStreamEndCapture(g.stream,&dst.graph);
            if(e!=cudaSuccess||!dst.graph||cudaGraphInstantiate(&dst.exec,dst.graph,nullptr,nullptr,0)!=cudaSuccess)return CUDA_RC_ERROR;
            size_t n=0;if(cudaGraphGetNodes(dst.graph,nullptr,&n)!=cudaSuccess)return CUDA_RC_ERROR;c->ffn_nodes+=n;
        }
    }

    if(c->paired_head) {
        if(c->max_in<2560||c->max_quant<2560)return CUDA_RC_INVALID;
        if(cudaMalloc((void**)&c->head_output,2*uint64_t(262144)*4)!=cudaSuccess)return CUDA_RC_OOM;
        if(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess)return CUDA_RC_ERROR;
        launch_head(*c);
        auto e=cudaStreamEndCapture(g.stream,&c->head.graph);
        if(e!=cudaSuccess||!c->head.graph||cudaGraphInstantiate(&c->head.exec,c->head.graph,nullptr,nullptr,0)!=cudaSuccess)return CUDA_RC_ERROR;
        if(std::getenv("IMPARO_GROUP_B2_HEAD_TRACE"))std::fprintf(stderr,"[group-b2-head] captured owners=%u scratch=2097152\n",count);
    }
    for(auto & m:c->members)++m.state->graph_leases;c->leased=true;
    std::fprintf(stderr,"[owner-ffn] owners=%u layers=%zu parent_nodes=%zu non_ffn_nodes=%zu grouped_nodes_all_widths=%zu phases_per_owner=%zu\n",
        count,layers,c->parent_nodes,c->phase_nodes,c->ffn_nodes,layers+1);
    *out=c->id;++next_group_id;group=std::move(c);return 0;
} catch(const std::bad_alloc&) {return CUDA_RC_OOM;} catch(...) {return CUDA_RC_ERROR;}

int step(uint64_t id,const uint32_t * members,const uint32_t * tokens,
        const uint32_t * positions,uint32_t count,uint32_t grouped,uint32_t * output) {
    if(!group||group->id!=id||group->poisoned||!members||!tokens||!positions||!output
        ||!count||count>MaxReady||grouped>1||active_execution_owner_id!=0
        ||!execution_boundary_closed())return CUDA_RC_INVALID;
    auto & c=*group;
    if(c.choice_epoch!=g.choice_epoch||c.weights!=g.weights)return CUDA_RC_INVALID;
    for(uint32_t i=0;i<count;++i) {
        if(members[i]>=c.members.size())return CUDA_RC_INVALID;
        for(uint32_t j=0;j<i;++j)if(members[i]==members[j])return CUDA_RC_INVALID;
        auto & m=c.members[members[i]];auto & s=*m.state;
        if(s.decode_graph!=m.parent||s.graph_capture_generation!=m.generation||s.forward_active
            ||s.forward_open||s.pending_error||!s.graph_leases||positions[i]==UINT32_MAX
            ||positions[i]<s.graph_min_start||positions[i]>s.graph_max_start)return CUDA_RC_INVALID;
    }
    // Every member was checked before modifying input/control state. Once uploads
    // or kernels begin, any error poisons the group; never silently retry a token.
    auto fail=[&](int rc){c.poisoned=true;active_execution=&default_execution;return rc;};
    for(uint32_t i=0;i<count;++i) {
        auto & s=*c.members[members[i]].state;active_execution=&s;
        s.decode_prepared=true;s.decode_argmax=true;s.decode_token=tokens[i];
        s.decode_start_pos=positions[i];
        if(s.decode_previous_start_pos==UINT32_MAX||positions[i]!=s.decode_previous_start_pos+1)s.decode_sequence_start_pos=positions[i];
        s.decode_previous_start_pos=positions[i];
        int rc=stage_decode_inputs(tokens[i]);if(!rc)rc=upload_staged_decode_inputs();
        if(rc)return fail(rc);
        invalidate_q8_cache();
    }
    active_execution=&default_execution;
    if(grouped&&count>1) {
        std::copy(members,members+count,c.host_map);
        if(cudaMemcpyAsync(c.map,c.host_map,count*4,cudaMemcpyHostToDevice,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
    }
    for(uint32_t i=0;i<count;++i) {auto & s=*c.members[members[i]].state;s.forward_active=true;s.forward_decode=true;}
    if(!grouped||count==1) {
        for(uint32_t i=0;i<count;++i)
            if(cudaGraphLaunch(c.members[members[i]].state->decode_graph_exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
    } else {
        for(size_t l=0;l<c.ffn.size();++l) {
            for(uint32_t i=0;i<count;++i)
                if(cudaGraphLaunch(c.members[members[i]].phases[l].exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
            if(cudaGraphLaunch(c.ffn[l][count-2].exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
        }
        if(c.paired_head&&count==2) {
            for(uint32_t i=0;i<count;++i)
                if(cudaGraphLaunch(c.members[members[i]].head_pre.exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
            if(cudaGraphLaunch(c.head.exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
            for(uint32_t i=0;i<count;++i)
                if(cudaGraphLaunch(c.members[members[i]].head_post.exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
            if(std::getenv("IMPARO_GROUP_B2_HEAD_TRACE"))std::fprintf(stderr,"[group-b2-head] step members=%u,%u positions=%u,%u\n",members[0],members[1],positions[0],positions[1]);
        } else {
            for(uint32_t i=0;i<count;++i)
                if(cudaGraphLaunch(c.members[members[i]].phases.back().exec,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
        }
    }
    for(uint32_t i=0;i<count;++i)
        if(cudaMemcpyAsync(c.host_output+i,c.members[members[i]].state->bufs[14],4,cudaMemcpyDeviceToHost,g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
    if(cudaStreamSynchronize(g.stream)!=cudaSuccess)return fail(CUDA_RC_ERROR);
    std::copy(c.host_output,c.host_output+count,output);
    for(uint32_t i=0;i<count;++i) {
        auto & s=*c.members[members[i]].state;
        s.forward_active=false;s.forward_decode=false;s.forward_open=false;s.decode_prepared=false;s.forward_timing_armed=false;
        ++s.decode_warm_forwards;
    }
    return 0;
}
} // namespace owner_ffn_lab

namespace {
uint64_t grouped_ffn_device_bytes() {
    const auto * c=owner_ffn_lab::group.get();if(!c)return 0;
    uint64_t n=0;
    if(c->head_output)n+=2*uint64_t(262144)*4;
    if(c->x)n+=4*uint64_t(c->max_in)*4;
    if(c->gate)n+=4*uint64_t(c->max_mid)*4;
    if(c->output)n+=4*uint64_t(c->max_out)*4;
    if(c->q8)n+=4*uint64_t(c->max_quant/32)*sizeof(BlockQ8_1);
    if(c->map)n+=16;return n;
}

} // anonymous namespace

extern "C" int imparo_cuda_record_ffn_phase_lab(const FfnPhaseWire * wire,uint32_t entering) noexcept try {
    if(!wire||entering>1)return CUDA_RC_INVALID;
    auto & s=execution();
    if(!s.graph_capturing||!s.forward_decode||s.prefill_capture_active)return 0;
    cudaStreamCaptureStatus status;unsigned long long id=0;cudaGraph_t graph=nullptr;
    const cudaGraphNode_t * deps=nullptr;size_t n=0;
    if(cudaStreamGetCaptureInfo(g.stream,&status,&id,&graph,&deps,&n)!=cudaSuccess
        ||status!=cudaStreamCaptureStatusActive||!graph||n!=1)return CUDA_RC_INVALID;
    if(entering) {
        if(s.recording_ffn_open||wire->layer!=s.recording_ffn_phases.size())return CUDA_RC_INVALID;
        CapturedFfnPhase phase;phase.wire=*wire;phase.before=deps[0];
        s.recording_ffn_phases.push_back(phase);s.recording_ffn_open=true;
    } else {
        if(!s.recording_ffn_open||s.recording_ffn_phases.empty()
            ||std::memcmp(wire,&s.recording_ffn_phases.back().wire,sizeof(*wire)))return CUDA_RC_INVALID;
        s.recording_ffn_phases.back().after=deps[0];s.recording_ffn_open=false;
    }
    return 0;
} catch(...) {return CUDA_RC_OOM;}

extern "C" int imparo_cuda_ffn_group_create_lab(const uint64_t * ids,uint32_t count,uint64_t * out) {
    return owner_ffn_lab::create(ids,count,out);
}
extern "C" int imparo_cuda_ffn_group_step_lab(uint64_t id,const uint32_t * members,
        const uint32_t * tokens,const uint32_t * positions,uint32_t count,uint32_t grouped,uint32_t * output) {
    return owner_ffn_lab::step(id,members,tokens,positions,count,grouped,output);
}
extern "C" int imparo_cuda_ffn_group_read_lab(uint64_t id,uint32_t member,uint32_t buf,float * out,uint64_t n) {
    auto * c=owner_ffn_lab::group.get();
    if(!c||c->id!=id||c->poisoned||member>=c->members.size()||buf>=B_COUNT||!out||n>UINT64_MAX/4)return CUDA_RC_INVALID;
    auto & s=*c->members[member].state;
    if(!s.bufs[buf]||n*4>s.sizes[buf]||s.forward_active)return CUDA_RC_INVALID;
    if(cudaMemcpyAsync(out,s.bufs[buf],size_t(n*4),cudaMemcpyDeviceToHost,g.stream)!=cudaSuccess
        ||cudaStreamSynchronize(g.stream)!=cudaSuccess)return CUDA_RC_ERROR;
    return 0;
}
extern "C" int imparo_cuda_ffn_group_release_lab(uint64_t id) {
    auto * c=owner_ffn_lab::group.get();
    if(!c||c->id!=id||active_execution_owner_id!=0||!execution_boundary_closed())return CUDA_RC_INVALID;
    if(cudaStreamSynchronize(g.stream)!=cudaSuccess||!c->clear()) {c->poisoned=true;return CUDA_RC_ERROR;}
    owner_ffn_lab::group.reset();return 0;
}
extern "C" int imparo_cuda_ffn_group_read_kv_lab(uint64_t id,uint32_t member,uint32_t layer,
        uint32_t is_v,uint64_t off,uint8_t * out,uint64_t n) {
    auto * c=owner_ffn_lab::group.get();
    if(!c||c->id!=id||c->poisoned||member>=c->members.size()||!out
        ||active_execution_owner_id!=0||!execution_boundary_closed())return CUDA_RC_INVALID;
    auto & s=*c->members[member].state;
    if(s.forward_active||s.forward_open)return CUDA_RC_INVALID;
    active_execution=&s;
    int rc=imparo_cuda_read_kv(layer,is_v,off,out,n);
    active_execution=&default_execution;
    return rc;
}

// Read-only readiness before any submission: 1=ready,0=ordinary native path,
// negative=invalid call. The request executor may fall back only from this query,
// never from a failed step whose work may already have been submitted.
extern "C" int imparo_cuda_execution_owner_graph_range_lab(uint64_t id,uint32_t * lo,uint32_t * hi) {
    if(!lo||!hi||active_execution_owner_id!=0||!execution_boundary_closed())return -CUDA_RC_INVALID;
    auto * s=find_execution_owner(id);
    if(!s||s->pending_error||s->forward_open||s->forward_active)return -CUDA_RC_INVALID;
    if(!s->decode_graph||!s->decode_graph_exec||!s->decode_argmax||s->decode_row_bytes
        ||!s->graph_capture_compatible||s->captured_ffn_phases.empty())return 0;
    *lo=s->graph_min_start;*hi=s->graph_max_start;return 1;
}
