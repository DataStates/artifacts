#include "model_server.hpp"
#include <iostream>
#include <fstream>
#include <string>
#define __DEBUG
#include "debug.hpp"
#include "cxxopts.hpp"
#include "mpi.h"

std::ostream *logger = &std::cout;
struct maxloc_dtype {
  int value;
  int pos;
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm work_comm, intercomm;
    int rank, size, lsize, lrank;
    cxxopts::ParseResult result;
    try {
      cxxopts::Options options("server", "One line description of MyProgram");
      options.add_options()
          ("thallium_connection_string", "Thallium connection string (eg ofi+verbs)",  cxxopts::value<std::string>()->default_value("ofi+verbs"))
          ("num_threads", "Number of argobots threads for handling RPC requests", cxxopts::value<int>()->default_value("8"))
          ("num_servers", "Number of servers", cxxopts::value<int>()->default_value("1"))
          ("storage_backend", "rocksdb/map", cxxopts::value<std::string>()->default_value("rocksdb"))
          ("ds_colocated", "colocated for datastates ?", cxxopts::value<int>()->default_value("0")) 
          ;
      result = options.parse(argc, argv);
    } catch(std::exception const& ex) {
      std::cerr << ex.what() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &work_comm);
    MPI_Comm_rank(work_comm, &lrank);
    MPI_Comm_size(work_comm, &lsize);
   
    maxloc_dtype remote_leader_rank =  {};
    maxloc_dtype is_remote_leader = {lrank == 0 && rank != 0, rank};
    MPI_Allreduce(&is_remote_leader, &remote_leader_rank, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Intercomm_create(work_comm, 0, MPI_COMM_WORLD, 0, 0, &intercomm);
    auto thallium_connection_string = result["thallium_connection_string"].as<std::string>();
    auto num_threads = result["num_threads"].as<int>();
    auto storage_backend = result["storage_backend"].as<std::string>();

    tl::engine engine(thallium_connection_string.c_str(), THALLIUM_SERVER_MODE);
    engine.enable_remote_shutdown();
    std::string addr = engine.self();
    std::vector<char> addr_v(addr.begin(), addr.end());
    addr_v.emplace_back('\0');

    int max_length = addr_v.size() + 1;
    int global_max_len = 0;
    
    MPI_Allreduce(&max_length, &global_max_len, 1, MPI_INT, MPI_MAX, work_comm);
    addr_v.resize(global_max_len, '\0');
    std::vector<char> alladdr_v(global_max_len*lsize);
    MPI_Gather(
        addr_v.data(), addr_v.size(), MPI_CHAR,
        alladdr_v.data(), addr_v.size(), MPI_CHAR, 0,
        work_comm
        );
    std::vector<std::string> alladdr;
    for (int i = 0; i < lsize; ++i) {
      alladdr.emplace_back(alladdr_v.data() + i*global_max_len);
    }

    int64_t alldata_size = alladdr_v.size();
    MPI_Bcast(&alldata_size, 1 , MPI_INT64_T, ((lrank==0)? MPI_ROOT : MPI_PROC_NULL), intercomm);
    MPI_Bcast(alladdr_v.data(), alladdr_v.size(), MPI_CHAR, ((lrank==0)? MPI_ROOT : MPI_PROC_NULL), intercomm);
    MPI_Comm_free(&intercomm);

    model_server_t* provider = new model_server_t(engine, lrank, num_threads, storage_backend);
    engine.wait_for_finalize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
