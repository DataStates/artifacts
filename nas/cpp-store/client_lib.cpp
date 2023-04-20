#include "model_client.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
//#include <boost/algorithm/string.hpp>

#define __DEBUG
#include "debug.hpp"

std::ostream *logger = &std::cerr;
    model_client_t *client = NULL;

namespace tl = thallium;

void __attribute__ ((constructor)) client_constructor() {
    /*std::vector<std::string> servers;
    std::fstream fin;
    std::string server_fname = "/home/mmadhya1/experiments/cpp-store/server_str.txt";
    fin.open(server_fname.c_str(), std::ios::in);
    
   if (fin.is_open()){   //checking whether the file is open
      std::string s1;
      while(std::getline(fin, s1)){ //read data from file object and put it into string.
         servers.emplace_back(s1);
      }
      fin.close(); //close the file object.
   }
    
//boost::split(servers, getenv("MODEL_SERVERS"), boost::is_any_of(" "));
    if (servers.empty())
        FATAL("MODEL_SERVERS environment variable does not contain a valid list of connection strings");
    std::stringstream buff;
    for (auto &server : servers)
        buff << server << " ";
    DBG("client connecting to " << servers.size() << " servers: " << buff.str());
    client = new model_client_t(servers);
    */

}




/*#include "model_client.hpp"

#include <iostream>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

#define __DEBUG
#include "debug.hpp"

std::ostream *logger = &std::cerr;
model_client_t *client = NULL;

namespace tl = thallium;

void __attribute__ ((constructor)) client_constructor() {
    std::vector<std::string> servers;
    boost::split(servers, getenv("MODEL_SERVERS"), boost::is_any_of(" "));
    if (servers.empty())
        FATAL("MODEL_SERVERS environment variable does not contain a valid list of connection strings");
    std::stringstream buff;
    for (auto &server : servers)
        buff << server << " ";
    //DBG("client connecting to " << servers.size() << " servers: " << buff.str());
    client = new model_client_t(servers);
}
*/

void __attribute__ ((destructor)) client_destructor() {
    //delete client;
}

extern "C" bool store_meta(uint64_t id, uint64_t *edges, int m, uint64_t *lids, uint64_t *owners, uint64_t *sizes, int n) {
    if (n < 2)
        return false;
    digraph_t g;
    g.root = edges[0];
    g.id = id;
    for (int i = 0; i < m; i += 2) {
        g.out_edges[edges[i]].insert(edges[i+1]);
        g.in_degree[edges[i+1]]++;
    }
    //Composition is a map from tensor id: (owner model id, size)
    //Tensor id is the layer hash + tensor id in the layer
    composition_t comp;
    for (int i = 0; i < n; i++)
        comp.emplace(lids[i], std::make_pair(owners[i], sizes[i]));
    return client->store_meta(g, comp);
}

extern "C" bool store_layers(uint64_t id, uint64_t *lids, uint64_t *sizes, unsigned char **ptrs, int n) {
    //std::cout<<"id in client lib: "<<id<<"\n";
    vertex_list_t layer_id(lids, lids + n);
    return client->store_layers(id, layer_id, sizes, ptrs);
}

extern "C" bool read_layers(uint64_t id, uint64_t *lids, unsigned char **ptrs, int n) {
    //std::cout<<"id here in client_lib: "<<id<<"\n";
    //fflush(stdout);
    vertex_list_t layer_id(lids, lids + n);
    bool ret =  client->read_layers(id, layer_id, ptrs);
    return ret;
}

extern "C" int get_composition(uint64_t id, uint64_t *lids, uint64_t *owners, int n) {
    auto &comp = client->get_composition(id);
    int count = 0;
    for (int i = 0; i < n; i++) {
        auto it = comp.find(lids[i]);
        if (it != comp.end()) {
            owners[i] = it->second.first;
            count++;
        } else{
            //std::cout<<"setting to zero!!\n";
            //fflush(stdout);
            owners[i] = 0;
        }
    }
    return count;
}

extern "C" int get_prefix(uint64_t *edges, int n, uint64_t *id, uint64_t *result) {
    if (n < 2)
        return 0;
    digraph_t g;
    g.root = edges[0];
    for (int i = 0; i < n; i += 2) {
        g.out_edges[edges[i]].insert(edges[i+1]);
        g.in_degree[edges[i+1]]++;
    }
    prefix_t reply = client->get_prefix(g);
    *id = reply.first;
    std::copy(reply.second.begin(), reply.second.end(), result);
    return reply.second.size();
}

extern "C" int get_tensor_access_statistics(uint64_t *model_ids, uint64_t *layer_ids, double **elapsed,  uint64_t *count){
    std::map<std::pair<model_id_t, vertex_t>, tensor_access_t> result = client->get_tensor_access_statistics();
    int i=0;
    for(auto &it: result){
        model_ids[i] = it.first.first; 
        layer_ids[i] = it.first.second; 
        count[i] = it.second.count;
        std::vector<double> timestamps = it.second.elapsed_timestamps;
        for(int j=0; j<timestamps.size(); ++j){
            elapsed[i][j]=timestamps[j] ;
        }
        i++;
    }
    //std::cout<<"clientlib HERE44\n";
    //fflush(stdout);
    return (int)result.size();
}

extern "C" bool update_ref_counter(uint64_t id, int value) {
    return client->update_ref_counter(id, value);
}

extern "C" int shutdown() {
    return client->shutdown();
}
