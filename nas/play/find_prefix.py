#!/usr/bin/env python
import tensorflow as tf
import networkx as nx
from tensorflow import keras
from tensorflow.keras import layers
import copy

model_seq = keras.Sequential([
    layers.Input(4),
    layers.Dense(2),
    layers.Dense(3),
    layers.Dense(4)
])
model_seq.build()

inputs = layers.Input(4)
out1 = layers.Dense(2)(inputs)
out2 = layers.Dense(3)(out1)
out3 = layers.Dense(4)(out2)
full = layers.Concatenate()([out3,out1])
out4 = layers.Dense(4)(full)
model_fun = keras.Model(inputs=inputs, outputs=out4, name="foo")

model_fun.get_config()

#a model_clone clone_function that adds the suffix foo

def add_suffix(layer):
    layer_config = copy.deepcopy(layer.get_config())
    #print(layer_config)
    print(layer.__class__)
    layer_config['name'] = layer_config['name'] + "_foo"
    return layer.__class__.from_config(layer_config)


def serialized_layer_order(model):
    dot = tf.keras.utils.model_to_dot(model)
    graph = nx.DiGraph()
    prov_ids = {}
    root = None
    for node in dot.get_nodes():
        label = node.get_attributes().get('label')
        if label is None:
            continue
        prov_id = node.get_name()
        if root is None:
            root = prov_id
        prov_ids[prov_id] = label.partition('|')[0]
    for edge in dot.get_edges():
        graph.add_edge(edge.get_source(), edge.get_destination())
    return map(lambda x: prov_ids[x], nx.dfs_preorder_nodes(graph, source=root))

from collections import defaultdict
def standardize_names():
    counts = defaultdict(int)
    def inner(layer):
        nonlocal counts
        layer_config = copy.deepcopy(layer.get_config())
        layer_config['name'] = layer.__class__.__name__ + str(counts[layer.__class__.__name__])
        counts[layer.__class__.__name__] += 1
        return layer.__class__.from_config(layer_config)
    return inner


# testing serialized_layer_order to see what the output is for equivalent configurations
list(serialized_layer_order(model_seq))
list(serialized_layer_order(model_fun))

# playing around with clone_model
list(serialized_layer_order(keras.models.clone_model(model_seq, clone_function=add_suffix)))

list(serialized_layer_order(keras.models.clone_model(model_fun, clone_function=standardize_names())))


