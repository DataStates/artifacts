import os
import tensorflow as tf
from tensorflow import keras
import transfer_learning as tl
import collections
import requests
import json
requests.packages.urllib3.disable_warnings()
import ssl
from tensorflow import keras
import time
import traceback
import argparse
from collections import defaultdict
from typing import List
import gc
import uuid
import networkx as nx
import numpy as np
import tensorflow as tf
import pickle
import os, logging
import hashlib
import copy
from datetime import datetime
import random
import ctypes
import argparse
import math
import sys
import multiprocessing as mp
from mpi4py import MPI

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def _transfer_name():
    return np.uint64(uuid.uuid4().int>>64)


def perc_to_layers(perc, total_size, size_per_layer, total_num_layers):
    if perc == 100:
        return total_num_layers
    
    perc_decimal = float(perc) / 100.0
    sol =  1 + ((perc_decimal * total_size) / 4.0  - 32*size_per_layer) / (size_per_layer + size_per_layer**2) 
    sol = int(sol)
    return sol


def get_model_size(model):
    siz=0
    dtype_size=0
    for layer in model.layers:
        weights = layer.get_weights()
        for i in weights:
            siz+=i.size

    print("Model size: ", end='', flush=True)
    print(siz * 4 / 1000000, flush=True)


def from_human(s: str) -> int:
    if s.endswith("g"):
        return int(s.split("g")[0]) * 1024**3
    elif s.endswith("m"):
        return int(s.split("m")[0]) * 1024**2
    elif s.endswith("k"):
        return int(s.split("k")[0]) * 1024
    else:
        raise RuntimeError(f"unable to parse \"{s}\"")


def str_to_int(name):
    h = int.from_bytes(hashlib.md5(name.encode()).digest()[8:], "big")
    return h

def create_model(size_per_layer=1024, num_layers=1, variance=0, num_transferred=0, model_id=0):
    current_limit = sys.getrecursionlimit()
    target_limit = 20000
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)
    assert size_per_layer - variance > 0
    inputs = tf.keras.Input(32, name="input")
    ik=0
    for i in range(num_layers+1):
        layer_size = random.randint(
                size_per_layer - variance, size_per_layer + variance
            )
        if i < num_transferred:
            name = f'transfer{i}{ik}'
            #name = f'transfer{i}{ik}{variance}{num_layers}'
        else:
            name = f'notransfer{i}{ik}'
            #name = f'notransfer{i}{ik}{variance}{num_layers}'
            random_number = random.randint(1, 100000)
            name+=str(random_number)
        #name = str_to_int(name) 
        if i==0:
            x = tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal', activation="tanh")(inputs)
        elif i==num_layers:
            outputs = tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal', activation="tanh")(x)
        else:
            x= tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal', activation="tanh")(x)
  
    model = keras.Model(inputs, outputs)
    return model


def test(model, perc_str, transfer_method=None, transferred=None ):
    transfer_method.store(id=model_id, model=model, prefix=transferred, val_acc=0)


def setup_gpus(ds_colocated):
    gpus = tf.config.list_physical_devices("GPU")
    gpu_per_node = len(gpus)
    if ds_colocated:
        rank = MPI.COMM_WORLD.Get_rank()
        num_nodes = MPI.COMM_WORLD.Get_size()  / 5
        gpu_local_idx = (int(rank/num_nodes)-1) % gpu_per_node
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        num_nodes = MPI.COMM_WORLD.Get_size()  / 5
        gpu_local_idx = (int(rank/num_nodes)-1) % gpu_per_node
        #rank = MPI.COMM_WORLD.Get_rank()
        #gpu_local_idx = (rank-1) % gpu_per_node
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            logging.info(
                f"[r={rank}]: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU"
            )
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            logging.info(f"{e}")




if not MPI.Is_initialized():
    MPI.Init_thread()
parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--total_size", type=from_human, default=1024**2)
parser.add_argument("--size_relative", type=str, default='small')
parser.add_argument("--variance", type=int, default=0)
parser.add_argument("--action", type=str, default='both')
parser.add_argument("--partial_store", action="store_true")
parser.add_argument("--ds_colocated", action="store_true")
parser.add_argument("--allow_partial_stores", action="store_true")
parser.add_argument("--store_format", type=str, default='datastates')
parser.add_argument("--result_dir", type=str, default='./')
parser.add_argument("--file_dir", type=str, default='./')
parser.add_argument(
    "--store_weights",
    action="store_true",
)


workcomm = MPI.COMM_WORLD
rank = workcomm.Get_rank()
workcomm_size = workcomm.Get_size()
args = parser.parse_args()
store_format = args.store_format
result_dir = args.result_dir 

setup_gpus(args.ds_colocated)

num_stores = 2
num_stores_all = num_stores
perc_transferred_random_trials = np.tile([100], num_stores)

action='store'
size_per_layer = int( (-32 - args.num_layers + math.sqrt(32**2 + args.num_layers*args.total_size)) / (2*args.num_layers))
models = []
layers_to_store = []
for i in range(num_stores_all):
    num_tr = perc_to_layers(perc_transferred_random_trials[i], args.total_size, size_per_layer, args.num_layers)
    models.append(create_model(size_per_layer, args.num_layers, args.variance, num_tr,rank*1111 + i))
    layers_to_store.append([layer.name for iter1,layer in enumerate(models[-1].layers) if iter1 > num_tr])
    
store_dict={'100':0}
for i in range(num_stores_all):
    workcomm.Barrier()
    t1 = MPI.Wtime()
    models[i].save_weights(args.file_dir + str(rank*1111 + i) + '.h5')
    t2 = MPI.Wtime()
    elapsed_arr = [t1, t2]
    #print(elapsed_arr, flush=True)
    workcomm.Barrier()
    if workcomm.rank==0:
        min_time=t1
        max_time=t2
        for j in range(1, workcomm_size, 1):
            data = workcomm.recv(source=j, tag=11)
            if data[0] < min_time:
                min_time = data[0]
            if data[1] > max_time:
                max_time = data[1]
        store_dict[str(100)] += (t2 - t1)
    else:
        workcomm.send(elapsed_arr, dest=0, tag=11)


if workcomm.rank == 0:
    for i in store_dict:
        store_dict[i] = store_dict[i] / (float(workcomm_size) * num_stores)
    print(store_dict, flush=True)
    with open(result_dir + 'hdf5_store.pickle', 'wb') as handle:
        pickle.dump(store_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
