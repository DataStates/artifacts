#ifndef __DUMMY_BACKEND_HPP
#define __DUMMY_BACKEND_HPP
#include <tmci/backend.hpp>
#include "model_client.hpp"
#include "model_utils.hpp"
#include <string>
#include <list>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <memory>
#include<memory_resource>
#include "debug_resource.cpp"
#include "debug.hpp"
#define __DEBUG


class DummyBackend : public tmci::Backend {

    std::unique_ptr<model_client_t> client;
    timestamp_map_t profile_time_stamps;
    
    public:
	std::pmr::monotonic_buffer_resource* monotonic_upstream;
	std::pmr::unsynchronized_pool_resource* sync_upstream;
	debug_resource* dbg;
	char *buffer;
    DummyBackend(const char* config) {
        std::vector<std::string>servers;
        std::vector<int>providers;
	getServers(servers,providers);
        std::cout << "client(cpp) ";
        for (auto i : servers) {
          std::cout << i << ' ' << std::endl;
        }
        std::cout <<  std::endl;
        client.reset(new model_client_t(servers, providers));
	buffer = new char[client_pinned_buffer_size];
    	monotonic_upstream =  new std::pmr::monotonic_buffer_resource(buffer,client_pinned_buffer_size, std::pmr::null_memory_resource());
        sync_upstream = new std::pmr::unsynchronized_pool_resource(monotonic_upstream);
        dbg = new debug_resource("pool", sync_upstream, buffer);
    
    }

    virtual inline int getTimeStamps(uint64_t* ts){
        int k=0;
        for(auto &kv: profile_time_stamps){
            std::vector<uint64_t> ts_temp = kv.second;
            for(int i=0; i<ts_temp.size(); ++i){
                ts[k] = ts_temp[i];
                k++;
            }
        }
        return k;
    }
    virtual inline int getTimeStampsByKey(uint64_t* ts, char* function_string){
        if (profile_time_stamps.find(std::string(function_string)) == profile_time_stamps.end()){
            return -1;
        }
        
        std::vector<uint64_t> ts_temp = profile_time_stamps[std::string(function_string)];
        int k=0;
        for(int i=0; i<ts_temp.size(); ++i){
            ts[k] = ts_temp[i];
            ++k;
        }
        return k;
    }
    virtual inline int getNumTimeStamps(){
        int count=0;
        for(auto &kv: profile_time_stamps)
            count+=kv.second.size();
        return count;
    }
    
    virtual inline int getNumTimeStampsByKey(char* function_string){
        if (profile_time_stamps.find(std::string(function_string)) == profile_time_stamps.end())
            return -1;
        return profile_time_stamps[std::string(function_string)].size();
    }

    virtual inline void clearTimeStamps(){
        for(auto &kv: profile_time_stamps){
            kv.second.clear();
        }
    }
    virtual inline bool clearTimeStampsByKey(char *function_string){
        if (profile_time_stamps.find(std::string(function_string)) == profile_time_stamps.end())
            return false;
        profile_time_stamps[std::string(function_string)].clear();
        return true;
    }

    bool isGPUPtr(const void* ptr);

    void getServers(std::vector<std::string>&servers, std::vector<int>&providers);

    uint32_t Signed32ToUnsigned32(int32_t ele);

    uint64_t ConcatUnsigned32ToUnsigned64(uint32_t first, uint32_t second);

    std::vector<uint64_t> ConcatSigned32ToUnsigned64(std::vector<int32_t>&elements);
    virtual int Save(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32);

    virtual int Load(const std::vector<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32, std::vector<int32_t>&lowners_int32 );

    virtual bool store_meta(uint64_t id, uint64_t *edges, int m, uint64_t *lids, uint64_t *owners, uint64_t *sizes, int n, const float val_acc);

    virtual int get_composition(uint64_t id, uint64_t *lids, uint64_t *owners, int n);

    virtual int get_prefix(uint64_t *edges, int n, uint64_t *id, uint64_t *result) ;

    virtual bool update_ref_counter(uint64_t id, int value); 

    virtual int shutdown();

};
#endif
