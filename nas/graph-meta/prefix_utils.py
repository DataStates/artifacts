import rpc_utils
import networkx as nx
import collections

class GraphCache(rpc_utils.MPIRPC):
    PREFIX = 1
    STORE = 2
    def __init__(self):
        super().__init__()
        self.graphs = {}

    def add_graph(self, id, root, graph):
        self.single_query(GraphCache.STORE, (id, root, graph), hash(id) % self.size)
        
    def match(self, root, parent, child):
        visits = collections.defaultdict(int)
        frontier = collections.deque([root])
        prefix = []
        while len(frontier) > 0:
            u = frontier.popleft()
            prefix.append(u)
            for e in child.out_edges(u):
                if e in parent.out_edges(u):
                    u, v = e
                    visits[v] += 1
                    if visits[v] == max(parent.in_degree(v), child.in_degree(v)):
                        frontier.append(v)
        return prefix

    def aggregate(self, result, reply):
        return reply if len(reply[1]) > len(result[1]) else result
    
    def handler(self, cmd, args):
        if cmd == GraphCache.STORE:
            id, root, graph = args
            print("rank %d storing id: %s" % (self.rank, id))
            self.graphs[id] = (root, graph)
            return None
        elif cmd == GraphCache.PREFIX:            
            max_prefix = []
            max_id = None
            for (id, (root, parent)) in self.graphs.items():
                prefix = self.match(root, parent, args)
                if len(prefix) > len(max_prefix):
                    max_prefix = prefix
                    max_id = id
            return (max_id, max_prefix)
            
if __name__ == "__main__":    
    g1 = nx.DiGraph()
    g1.add_edge(0, 7)
    g1.add_edge(0, 1)
    g1.add_edge(1, 2)
    g1.add_edge(1, 3)
    g1.add_edge(1, 4)
    g1.add_edge(2, 5)
    g1.add_edge(3, 5)
    g1.add_edge(4, 6)
    g1.add_edge(5, 7)
    g1.add_edge(5, 8)
    g1.add_edge(6, 8)
    g1.add_edge(8, 9)

    g2 = g1.copy()
    g2.remove_edge(4, 6)
    g2.remove_edge(6, 8)
    g2.add_edge(4, 8)

    p = GraphCache()
    print(p.match(0, g2, g1))
    
