#include <iostream>

#include "model_client.hpp"

#define __DEBUG
#include "debug.hpp"

std::ostream *logger = &std::cout;

namespace tl = thallium;

int main(int argc, char **argv) {
    if (argc < 2)
        FATAL("usage: " << argv[0] << " <thallium_backend1> <thallium_backend2> ...");

    digraph_t m1;
    std::vector<vertex_t> m1_edges{0, 7, 0, 1, 1, 2, 1, 3, 1, 4, 2, 5, 3, 5, 4, 6, 5, 7, 5, 8, 6, 8, 8, 9};
    for (int i = 0; i < m1_edges.size(); i += 2) {
        m1.out_edges[m1_edges[i]].insert(m1_edges[i+1]);
        m1.in_degree[m1_edges[i+1]]++;
    }
    m1.root = 0;

    digraph_t m2;
    std::vector<vertex_t> m2_edges{0, 7, 0, 1, 1, 2, 1, 3, 1, 4, 2, 5, 3, 5, 4, 8, 5, 7, 5, 8, 8, 9};
    for (int i = 0; i < m2_edges.size(); i += 2) {
        m2.out_edges[m2_edges[i]].insert(m2_edges[i+1]);
        m2.in_degree[m2_edges[i+1]]++;
    }
    m2.root = 0;

    std::vector<std::string> servers;
    for (unsigned int i = 1; i < argc; i++)
        servers.push_back(argv[i]);
    model_client_t model_client(servers);
    //typedef std::unordered_map<model_id_t, std::pair<model_id_t, size_t>> composition_t;
    composition_t temp_c;
    temp_c[0] = std::pair<model_id_t, size_t>(0, 10);
    model_client.store_meta(m1, temp_c);
    prefix_t result = model_client.get_prefix(m2);
    vertex_list_t result_vertices = result.second;
    for (auto &i : result_vertices)
        std::cout << i << " ";
    
    std::cout << std::endl;
    
    DBG("get_prefix done");

    return 0;
}
