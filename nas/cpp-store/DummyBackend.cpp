#include "DummyBackend.hpp"
#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <cuda.h>
#include<cuda_runtime.h>

using namespace std::chrono;

TMCI_REGISTER_BACKEND(dummy, DummyBackend);

std::vector<std::string> global_servers;
std::ostream *logger = &std::cerr;

namespace tl = thallium;
void __attribute__ ((constructor)) DummyBackendConstructorAttr() {
}

void __attribute__ ((destructor)) DummyBackendDestructorAttr() {

}

bool DummyBackend::isGPUPtr(const void *ptr)
{
    cudaPointerAttributes atts;
    const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);

    // clear last error so other error checking does
    // not pick it up
    cudaError_t error = cudaGetLastError();
#if CUDART_VERSION >= 10000
    return perr == cudaSuccess &&
        (atts.type == cudaMemoryTypeDevice ||
         atts.type == cudaMemoryTypeManaged);
#else
    return perr == cudaSuccess && atts.memoryType == cudaMemoryTypeDevice;
#endif
}


void DummyBackend::getServers(std::vector<std::string>&servers, std::vector<int>&providers){
    servers = global_servers;
    providers.resize(servers.size());
    std::iota(providers.begin(), providers.end(), 0);
}

uint32_t DummyBackend::Signed32ToUnsigned32(int32_t ele){
    int32_t int32_max = std::numeric_limits<std::int32_t>::max();
    uint32_t ret = (ele>0) ? (uint32_t)ele + (uint32_t)int32_max : (uint32_t)(ele + int32_max);
    return ret; 
}

uint64_t DummyBackend::ConcatUnsigned32ToUnsigned64(uint32_t first, uint32_t second){
    uint64_t combined_ele = ((uint64_t)first) << 32 | second;
    return combined_ele;
}

std::vector<uint64_t>  DummyBackend::ConcatSigned32ToUnsigned64(std::vector<int32_t>&elements){
    std::vector<uint64_t> result;
    int siz = elements.size();
    for(int i=0; i<siz; i+=2){
        //TODO: just a regular conversion is well defined. 
        uint32_t temp_u32_first = Signed32ToUnsigned32(elements[i]);
        uint32_t temp_u32_second = Signed32ToUnsigned32(elements[i+1]);
        result.push_back(ConcatUnsigned32ToUnsigned64(temp_u32_first, temp_u32_second));
    }
    return result;
}


int DummyBackend::Save(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32 ) {
    uint64_t model_id = ConcatSigned32ToUnsigned64(modelids_int32)[0];
    std::vector<uint64_t> lids_uint64  = ConcatSigned32ToUnsigned64(lids_int32);
    vertex_list_t layer_id(lids_uint64.data(), lids_uint64.data() + lids_uint64.size());
    std::pmr::vector<char> temp{dbg};
    std::pmr::vector<char> ptrs[tensors.size()];
    for(int i=0 ; i<tensors.size(); ++i)
	    ptrs[i] = temp;
    std::vector<segment_t> segments;
    //Note: we assume that all the tensors are on the same device (GPU or CPU)
    if (isGPUPtr(tensors.begin()->get().tensor_data().data())){
	int iter=0;
        for(auto &t: tensors){
            auto size = (size_t)t.get().tensor_data().size();
	    ptrs[iter].resize(size);
	    cudaMemcpy((char*)ptrs[iter].data(), (char*)t.get().tensor_data().data(), size, cudaMemcpyDeviceToHost);
	    segments.emplace_back((void*)ptrs[iter].data(), size);
            iter++;
        }
    }
    else{
        for(auto &t: tensors)
            segments.emplace_back((void*)t.get().tensor_data().data(), (size_t)t.get().tensor_data().size());
    }
    auto ret = client->store_layers(model_id, layer_id, segments,  profile_time_stamps);
    return 0;
}


int DummyBackend::Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32,  std::vector<int32_t>&lowners_int32) {
    uint64_t model_id = ConcatSigned32ToUnsigned64(modelids_int32)[0];
    std::vector<uint64_t> lids_uint64  = ConcatSigned32ToUnsigned64(lids_int32);
    std::vector<uint64_t> lowners_uint64 = ConcatSigned32ToUnsigned64(lowners_int32);
    vertex_list_t layer_id(lids_uint64.data(), lids_uint64.data() + lids_uint64.size());
    std::pmr::vector<char>temp{dbg};
    std::pmr::vector<char> ptrs[tensors.size()];
    std::vector<segment_t> segments;
    for(int i=0 ; i<tensors.size(); ++i)
	    ptrs[i] = temp;
    
    //Note: we assume that all the tensors are on the same device (GPU or CPU)
    bool is_gpu = isGPUPtr(tensors.begin()->get().tensor_data().data());
    
    if (is_gpu){
        int iter=0;
        for(auto &t: tensors){
            size_t size = (size_t)t.get().tensor_data().size();
            ptrs[iter].resize(size);
            segments.emplace_back((void*)ptrs[iter].data(), size);
            iter++;
        }
    }else{
        for(auto &t: tensors)
            segments.emplace_back((void*)t.get().tensor_data().data(), (size_t)t.get().tensor_data().size());
    }
    int ret =  client->read_layers(model_id, layer_id, segments, lowners_uint64, profile_time_stamps);

    if (is_gpu){
        int iter=0;
        for(auto &t: tensors){
            cudaMemcpy((char*)t.get().tensor_data().data(), (char*)segments[iter].first, segments[iter].second, cudaMemcpyHostToDevice);
            ++iter;
        }
    }
    return 0;
}


bool DummyBackend::store_meta(uint64_t id, uint64_t *edges, int m, uint64_t *lids, uint64_t *owners, uint64_t *sizes, int n, const float val_acc) {
    if (n < 2)
	return false;
    
    digraph_t g;
    g.root = edges[0];
    g.id = id;
    for (int i = 0; i < m; i += 2) {
        g.out_edges[edges[i]].insert(edges[i+1]);
        g.in_degree[edges[i+1]]++;
    }
    composition_t comp;
    for (int i = 0; i < n; i++)
        comp.emplace(lids[i], std::make_pair(owners[i], sizes[i]));

    bool ret =  client->store_meta(g, comp, val_acc, profile_time_stamps);
    return ret;
}

int DummyBackend::get_composition(uint64_t id, uint64_t *lids, uint64_t *owners, int n) {
    auto &comp = client->get_composition(id, profile_time_stamps);
    int count = 0;
    for (int i = 0; i < n; i++) {
        auto it = comp.find(lids[i]);
        if (it != comp.end()) {
            owners[i] = it->second.first;
            count++;
        } else{
            owners[i] = 0;
        }
    }
    return count;
}

int DummyBackend::get_prefix(uint64_t *edges, int n, uint64_t *id, uint64_t *result) {
    if (n < 2)
        return 0;
    digraph_t g;
    g.root = edges[0];
    for (int i = 0; i < n; i += 2) {
        g.out_edges[edges[i]].insert(edges[i+1]);
        g.in_degree[edges[i+1]]++;
    }
    prefix_t reply = client->get_prefix(g, profile_time_stamps);
    *id = reply.first;
    std::copy(reply.second.begin(), reply.second.end(), result);
    return reply.second.size();
}


bool DummyBackend::update_ref_counter(uint64_t id, int value) {
    bool ret =  client->update_ref_counter(id, value);
    return ret;
}

int DummyBackend::shutdown() {
    return client->shutdown();
}

extern "C" bool store_meta(uint64_t id, uint64_t *edges, int m, uint64_t *lids, uint64_t *owners, uint64_t *sizes, int n, const float val_acc) {
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->store_meta(id, edges, m, lids, owners, sizes, n, val_acc);
}

extern "C" int get_composition(uint64_t id, uint64_t *lids, uint64_t *owners, int n) {
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->get_composition(id, lids, owners, n);
}

extern "C" int get_prefix(uint64_t *edges, int n, uint64_t *id, uint64_t *result) {
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->get_prefix(edges, n, id, result);
}

extern "C" bool update_ref_counter(uint64_t id, int value) {
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->update_ref_counter(id, value);	
}

extern "C" int get_time_stamps(uint64_t *ts){
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->getTimeStamps(ts);
}
extern "C" int get_time_stamps_by_key(uint64_t *ts, char *function_str){
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->getTimeStampsByKey(ts, function_str);
}

extern "C" void clear_time_stamps(){
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    ptr->clearTimeStamps();
}

extern "C" void clear_time_stamps_by_key(char *function_str){
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    ptr->clearTimeStampsByKey(function_str);
}


extern "C" int get_num_time_stamps(){
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->getNumTimeStamps();
}

extern "C" int get_num_time_stamps_by_key(char *function_str){
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->getNumTimeStampsByKey(function_str);
}

extern "C" int shutdown() {
    std::string backend = "dummy";
    std::string config  = ".";
    auto ptr = tmci::Backend::Create(backend.c_str(), config.c_str());
    return ptr->shutdown();
}

static 
std::vector<std::string> split(std::string const& s, char c) {
  std::vector<std::string> ret;
  std::istringstream ss(s);
  std::string tmp;
  while(std::getline(ss,tmp, c)) {
    if(!tmp.empty() && tmp.front() != '\0')
    ret.emplace_back(std::move(tmp));
  }
  return ret;
}

extern "C" int set_servers(const char* servers, size_t size) {
  try {
    std::string s(servers, size);
    global_servers = split(s, '\0');
    return 0;
  } catch(...) {
      std::cout << "ERROR " << std::endl;
    return 1;
  }
}
