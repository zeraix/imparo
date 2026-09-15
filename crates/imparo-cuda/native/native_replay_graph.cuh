#pragma once
// A source graph owns the Runtime argument storage referenced by DynamicGraphNode.
// Model adapters admit dynamic fields and ranges; this helper only manages lifetime,
// dependency order and the existing native node parameter update mechanism.
struct NativeReplayGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<DynamicGraphNode> nodes;
    ~NativeReplayGraph() { clear(); }
    NativeReplayGraph() = default;
    NativeReplayGraph(const NativeReplayGraph &) = delete;
    NativeReplayGraph &operator=(const NativeReplayGraph &) = delete;
    void clear() noexcept {
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
        exec=nullptr;graph=nullptr;nodes.clear();
    }
    std::vector<cudaGraphNode_t> ordered_nodes() {
        size_t n=0;if(cudaGraphGetNodes(graph,nullptr,&n)!=cudaSuccess)throw std::runtime_error("graph node count");
        std::vector<cudaGraphNode_t> all(n),out;std::vector<bool> used(n,false);
        if(cudaGraphGetNodes(graph,all.data(),&n)!=cudaSuccess)throw std::runtime_error("graph nodes");
        while(out.size()<n){bool changed=false;
            for(size_t i=0;i<n;++i)if(!used[i]){
                size_t count=0;if(cudaGraphNodeGetDependencies(all[i],nullptr,&count)!=cudaSuccess)throw std::runtime_error("graph dependencies");
                std::vector<cudaGraphNode_t> deps(count);
                if(count&&cudaGraphNodeGetDependencies(all[i],deps.data(),&count)!=cudaSuccess)throw std::runtime_error("graph dependency list");
                bool ready=true;for(auto dep:deps)if(std::find(out.begin(),out.end(),dep)==out.end())ready=false;
                if(ready){out.push_back(all[i]);used[i]=true;changed=true;}
            }
            if(!changed)throw std::runtime_error("graph dependency cycle");
        }return out;
    }
    cudaError_t replay(uint32_t start,cudaStream_t stream) {
        for(auto &d:nodes){
            d.start_value=start+d.start_delta;
            const uint64_t valid=uint64_t(start)+d.valid_delta;
            d.valid_value=d.ring?uint32_t(std::min<uint64_t>(valid,uint64_t(d.ring)+1)):uint32_t(valid);
            if(d.start_index!=UINT32_MAX)d.args[d.start_index]=&d.start_value;
            if(d.valid_index!=UINT32_MAX)d.args[d.valid_index]=&d.valid_value;
            if(d.span_index!=UINT32_MAX&&d.span_follows_valid){d.span_value=d.valid_value;d.args[d.span_index]=&d.span_value;}
            if(d.native_update==NativeGraphUpdate::ValidGridY)d.params.gridDim.y=d.valid_value;
            d.params.kernelParams=d.args.data();
            const auto rc=cudaGraphExecKernelNodeSetParams(exec,d.node,&d.params);if(rc!=cudaSuccess)return rc;
        }return cudaGraphLaunch(exec,stream);
    }
};
