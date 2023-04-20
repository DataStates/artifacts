#!/usr/bin/env python

from deepspace.tabular import DenseSkipCoSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import operation, Zero, Connect, AddByProjecting, Identity
from networkx import shortest_path, topological_generations, descendants, DiGraph, topological_sort, transitive_closure
from pprint import pprint
from collections import Counter, defaultdict
from numpy import linspace
import time

space_args = {
    "input_shape": (32,),
    "output_shape": (1,),
    "num_layers": 100,
}
space = DenseSkipCoSpace(**space_args)
space.build()



g = DiGraph()
g.add_edge(1,5)
g.add_edge(2,5)
g.add_edge(3,5)
g.add_edge(3,6)
g.add_edge(4,7)
g.add_edge(5,7)
g.add_edge(6,7)


def compute_weights(graph, filter_fn = lambda x: True, beta=1):
    """compute the weights to use when determining where to allocate a mutation

    First, we determine generations, by counting how many nodes would be invalidated by changing this node
    Next, we assign weights to each generation as a linear weight between 1 and β
    Then within a generation we weigh each option equally.

    Choosing β=1 weighs all generations equally, larger values of β result in greater weight to the end layers
    """
    invalidate_count_to_nodes = defaultdict(list)
    for node in filter(filter_fn, graph):
        invalidate_count_to_nodes[len(descendants(graph, node))].append(node)
    invalidate_count_to_gen_size = {gen:len(nodes) for gen,nodes in  invalidate_count_to_nodes.items()}
    generations = len(invalidate_count_to_nodes)
    #compute the linearly distributed weights weights
    weights = linspace(1, beta, generations, endpoint=True)
    weights /= weights.sum()
    invalidate_count_to_weight = {
        invalidate_count:weight for invalidate_count, weight in
        zip(sorted(invalidate_count_to_nodes, reverse=True), weights)
    }
    node_probablility = {}
    for invalidate_count, nodes in invalidate_count_to_nodes.items():
        gen_size = invalidate_count_to_gen_size[invalidate_count]
        weight = invalidate_count_to_weight[invalidate_count]
        propability = 1/gen_size * weight
        for node in nodes:
            node_probablility[node] = propability
    return node_probablility

pprint(compute_weights(g))


b = time.perf_counter()
is_vnode = lambda x: isinstance(x, VariableNode)
node_weights = compute_weights(space.graph, is_vnode, 1)
e = time.perf_counter()
print(e-b)
    
