import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
    print("number of layers transferred: ", end='', flush=True)
    print(sol)
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


def create_model_and_test( action='store', transfer_method=None, size_per_layer=1024, num_layers=1, variance=0, num_transferred=0, model_id=0 ):
    random.seed(2)
    current_limit = sys.getrecursionlimit()
    target_limit = 20000
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)
    assert size_per_layer - variance > 0
    inputs = tf.keras.Input(32, name="input")
    for i in range(num_layers+1):
        layer_size = random.randint(
                size_per_layer - variance, size_per_layer + variance
            )
        if i < num_transferred:
            name = f'transfer{i}{model_id}{variance}{num_layers}'
        else:
            name = f'notransfer{i}{model_id}{variance}{num_layers}' 
        #name = str_to_int(name) 
        if i==0:
            x = tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal', activation="tanh")(inputs)
        elif i==num_layers:
            outputs = tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal', activation="tanh")(x)
        else:
            x= tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal', activation="tanh")(x)
   
    model = keras.Model(inputs, outputs) 
    if action == 'store':
        transfer_method.store(id=model_id, model=model, prefix=[])
    else:
        transfer_method.transfer(id=model_id, model=model)
    
    transfer_method.clear_time_stamps()
    del model
    tf.keras.backend.clear_session()
    gc.collect




parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--total_size", type=from_human, default=1024**2)
parser.add_argument("--size_relative", type=str, default='small')
parser.add_argument("--variance", type=int, default=0)
parser.add_argument("--action", type=str, default='both')
args = parser.parse_args()

transfer_method_cls = tl.transfer_methods.make_transfer_method('datastates')
workcomm, conn_strs = transfer_method_cls.startup_server()

transfer_method = transfer_method_cls(
        hosts=conn_strs,
        trace_path = '.',
        bulk_storage_path='.'
        )

num_stores = 1
num_transfers = 1
perc_transferred_random_trials = np.tile([25,50,75,100], num_transfers)

store_model=True
load_model=True

if args.action == 'store':
    load_model=False

if args.action == 'load':
    store_model=False

size_per_layer = int( (-32 - args.num_layers + math.sqrt(32**2 + args.num_layers*args.total_size)) / (2*args.num_layers))
if store_model:
    ####STORE FULL MODELS######
    for i in range(num_stores):
        create_model_and_test( action='store', transfer_method=transfer_method, size_per_layer=size_per_layer, num_layers=args.num_layers, variance=args.variance, num_transferred=args.num_layers, model_id=i)
        print("stored", flush=True)

time.sleep(10)

if load_model:
    ####TRANSFER DIFFERENT SIZED LAYERS######
    for i, perc in enumerate(perc_transferred_random_trials):
        create_model_and_test( action='transfer', transfer_method=transfer_method, size_per_layer=size_per_layer, num_layers=args.num_layers, variance=args.variance, num_transferred=perc_to_layers(perc, args.total_size, size_per_layer, args.num_layers), model_id=i)
        print("transferred", True)

transfer_method.teardown_server(workcomm)
