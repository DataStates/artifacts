#!/usr/bin/env python
from deephyper.benchmark.nas.linearReg.load_data import load_data
from deephyper.contrib.callbacks import import_callback
from deephyper.core.exceptions import DeephyperRuntimeError
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback, SearchEarlyStopping
from deephyper.nas.node import VariableNode
from deephyper.nas.preprocessing import minmaxstdscaler
from deephyper.nas.run._util import (HistorySaver, compute_objective,
                                     default_callbacks_config,
                                     get_search_space, load_config,
                                     preproc_trainer, setup_data)
from deephyper.nas.trainer import BaseTrainer
from deephyper.problem import NaProblem

from deephyper.search.nas import RegularizedEvolution
from transfer_methods import TransferHDF5, TransferMethod, TransferNoop, TransferSimpleHDF5
from pathlib import Path
import ray
import os, sys
import pickle
from deephyper.nas.metrics import r2, acc
import keras
import transfer_methods
import time
import traceback
import argparse
from collections import defaultdict
import hashlib
from typing import List
import random
import copy
import tensorflow as tf
from deepspace.tabular import FeedForwardSpace
import collections
import json
from pathlib import Path
import os, sys
import pickle
import networkx as nx

'''
import logging
import os
#Uncomment if running on a single GPU on thetaGPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import traceback
import argparse
from collections import defaultdict
import hashlib
from typing import List
import random
import copy
import networkx as nx
import numpy as np
import tensorflow as tf
from deepspace.tabular import FeedForwardSpace
import collections
import json
from transfer_methods import TransferHDF5, TransferMethod, TransferNoop, TransferSimpleHDF5
from pathlib import Path
import os, sys
import pickle
from deephyper.nas.metrics import r2, acc
import keras
import transfer_methods
'''
def _standardize_names(model):
    # first build the graph of the layers
    graph = collections.defaultdict(list)
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)
    if isinstance(model, tf.keras.Sequential):
        #functional model doesn't have 'inbound_nodes', simply iterate an build graph linearly
        for last_layer, current_layer in zip(model.layers, model.layers[1:]):
            graph[current_layer.name] = [last_layer.name]
    elif isinstance(model, tf.keras.Model):
        #assume instance of functional model
        model_config = model.get_config()
        for layer in model_config['layers']:
            if len(layer['inbound_nodes']) > 0:
                graph[layer['name']] = [
                            inbound[0] for inbound in layer['inbound_nodes'][0]
                        ]

    def inner(layer):
        #we can't assume that layers aren't shared, so force a copy
        layer_config = copy.deepcopy(layer.get_config())

        #we need the old name to lookup predecessors
        old_name = layer_config['name']

        #we don't want to hash the name since this would make things recursively defined
        del layer_config['name']

        begin = time.perf_counter()
        layer_hash = hashlib.sha3_512()
        layer_hash.update(json.dumps(layer_config, sort_keys=True).encode())
        for pred_name in graph[old_name]:
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()
        end = time.perf_counter()

        base_name = layer.__class__.__name__ + layer_hash
        hashed_names[old_name] = base_name + "_" + str(counts[base_name])
        counts[base_name]+=1
        #create the new layer with the appropriate name
        layer_config['name'] = hashed_names[old_name]
        return layer.__class__.from_config(layer_config)
    return inner

def standardize_names(model: tf.keras.Model):
    return tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_standardize_names(model))

def reset_trainable_clone_model(model):
    all_layer_dict = model.get_config()['layers']
    modified_all_layer_dict = []
    for layer in all_layer_dict:
        config = copy.deepcopy(layer)
        if 'trainable' in config['config']:
            del config['config']['trainable']

        modified_all_layer_dict.append(config)

    config_modified = copy.deepcopy(model.get_config())
    del config_modified['layers']
    config_modified['layers'] = copy.deepcopy(modified_all_layer_dict)
    model_ret = keras.Model.from_config(config_modified)
    return model_ret


def print_single_layer(layer_dict, ind, attribute=False):
    if attribute is not False:
        print(layer_dict[ind]['config'][attribute])
    else:
        print(layer_dict[ind]['config'])

def print_layers(model, attribute=False):
    all_layers = model.get_config()['layers']
    for it, layer in enumerate(all_layers):
        print_single_layer(all_layers, it, attribute=attribute)

path_main='/home/mmadhyastha/experiments/models_cpp_store_test/'
with open(path_main+'parent_dict.pickle', 'rb') as handle:
    global_parent_dict = pickle.load(handle)
with open(path_main+'child_dict.pickle', 'rb') as handle:
    global_child_dict = pickle.load(handle)

new_parent_dict = {}
new_child_dict = {}
for k, v in global_child_dict.items():
    loaded_parent = tf.keras.models.load_model(path_main + k+'.h5', custom_objects={'r2': r2, 'acc':acc}, compile=False)
    loaded_parent.compile()
    modified_parent = reset_trainable_clone_model(loaded_parent)
    modified_parent_coded = standardize_names(modified_parent)
    new_child_dict[modified_parent_coded] = []
    for child in v:
        loaded_child = tf.keras.models.load_model(path_main + child+'.h5', custom_objects={'r2': r2, 'acc':acc}, compile=False)
        modified_child = reset_trainable_clone_model(loaded_child)
        modified_child_coded =  standardize_names(modified_child)
        new_parent_dict[modified_child_coded] = modified_parent_coded
        new_child_dict[modified_parent_coded].append(modified_child_coded)


repo = transfer_methods.DataStatesModelRepo()
itid=0
for child, parent in new_parent_dict.items():
    repo.store(itid, parent, None)
    itid+=1
    print('stored', flush=True)
    if itid == 5:
        break

itid=0
for child, parent in new_parent_dict.items():
    prefix = repo.transfer(50+itid, child)
    itid+=1
    if itid == 5:
        break

m_ids, l_ids, counts, times = repo.get_tensor_access_statistics()

print('length /number of items: ', flush=True)
print(len(counts))
print('printing access statistics', flush=True)

'''
print('Model ids: ', flush=True)
for i in m_ids:
    print(i, flush=True, end=', ')
print('')

print('layer/tensor ids: ', flush=True)
for i in l_ids:
    print(i, flush=True, end=', ')
print('')

print('layer/tensor counts: ', flush=True)
for i in counts:
    print(i, flush=True, end=', ')
print('')
print('timestamps: ', flush=True)
for timestamp in times:
    for ele in timestamp:
        print(ele, flush=True, end=',')
    print('')

print('DONE')
for child in new_parent_dict:
    parent = new_parent_dict[child]
    print('printing child')
    print_layers(child)
    print(child.summary())
    print('printing parent')
    print_layers(parent)
    print(parent.summary())
'''

