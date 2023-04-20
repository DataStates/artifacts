#ifndef __MODEL_UTILS
#define __MODEL_UTILS

#include <utility>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include<map>

#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <thallium/serialization/stl/pair.hpp>
#include <thallium/serialization/stl/vector.hpp>
#include <thallium/serialization/stl/unordered_set.hpp>
#include <thallium/serialization/stl/unordered_map.hpp>

typedef std::pair<void *, size_t> segment_t;
typedef uint64_t vertex_t;
typedef uint64_t model_id_t;
typedef std::vector<vertex_t> vertex_list_t;
typedef std::vector<model_id_t> id_list_t;
typedef std::unordered_map<model_id_t, std::pair<model_id_t, size_t>> composition_t;
typedef std::pair<model_id_t, vertex_list_t> prefix_t;
typedef std::pair<uint64_t, uint64_t> timestamp_t;
typedef std::map<std::string, std::vector<uint64_t>> timestamp_map_t;
struct tensor_access_t 
{
    uint64_t count;
    std::vector<double> elapsed_timestamps;
    std::chrono::steady_clock::time_point first_timestamp; 
     template<typename A> void serialize(A& ar) {
         ar & count;
         ar & elapsed_timestamps;
     }
};

struct digraph_t {
    typedef std::unordered_set<vertex_t> vset_t;

    model_id_t id = 0;
    vertex_t root;
    std::unordered_map<vertex_t, vset_t> out_edges;
    std::unordered_map<vertex_t, int> in_degree;

    template<typename A> void serialize(A& ar) {
        ar & id;
        ar & root;
        ar & out_edges;
        ar & in_degree;
    }
};
const size_t rdma_transfer_size=400000000;
const size_t client_pinned_buffer_size=5000000000;
const size_t server_pinned_buffer_size=50000000000;
#endif
