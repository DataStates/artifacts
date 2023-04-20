import prefix_utils
import json, sys
import networkx as nx

from timeit import default_timer as timer
from datetime import timedelta
from mpi4py import MPI

start = timer()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    p = prefix_utils.GraphCache()
    p.start()

    if rank == 0:
        count = 0
        f = open(sys.argv[1])
        for line in f.readlines():
            model = json.loads(line)
            g = nx.DiGraph()
            for layer in model['layers']:
                u = layer['name']
                for in_node in layer['inbound_nodes']:
                    v = in_node[0][0]
                    g.add_edge(v, u)
            count += 1
            p.add_graph(count, 'input_0', g)
    comm.barrier()
    print("rank %d init complete, starting experiment" % rank)
    start = timer()
    if rank == 0:
        id, prefix = p.reduce_query(prefix_utils.GraphCache.PREFIX, g)
        print("selected candidate: %s, prefix length: %d" % (id, len(prefix)))
    end = timer()
    print("rank %d, duration: %s" % (rank, timedelta(seconds=end-start)))
