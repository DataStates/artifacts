#!/usr/bin/env python

from mpi4py import MPI
from transfer_learning.transfer_methods import TransferHDF5, TransferNoop, TransferSimpleHDF5
from numpy.random import SeedSequence, default_rng
import uuid
import itertools
import tensorflow as tf
import pandas as pd
import collections
import argparse
import gc
import time
Timings = collections.namedtuple('Timings', ['standardize_time', 'store_time', 'transfer_time'])

rank = MPI.COMM_WORLD.rank
comm_size = MPI.COMM_WORLD.size
devices = [dev for _,dev in zip(range(comm_size), itertools.cycle(tf.config.list_logical_devices("GPU")))]
print(devices, comm_size, rank)

with tf.device(devices[rank].name):

    TRANSFER = {
            "simple_hdf5": TransferSimpleHDF5(),
            "hdf5": TransferHDF5(port=7000),
            "noop": TransferNoop(),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter",type=int, default=100)
    parser.add_argument("--max_layer_size",type=int, default=100)
    parser.add_argument("--max_layers",type=int, default=5)
    parser.add_argument("--layers_scale_factor",type=int, default=1)
    parser.add_argument("--method",type=str, default='shallow', choices=["small", "shallow"])
    parser.add_argument("--transfer_method",type=str, default='noop', choices=list(TRANSFER))
    args = parser.parse_args()


    global_seed = 0
    seed = SeedSequence([rank,global_seed]).spawn(1)[0]
    rng = default_rng(seed)


    def make_small_models(max_layer_size:int, layers_scale_factor:int):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(1),
            tf.keras.layers.Dense(rng.integers(1,max_layer_size) * layers_scale_factor)
        ])
        model.build()
        return model

    def make_shallow_models(max_layer_size:int, max_layers: int, layers_scale_factor: int):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(1),
            *[tf.keras.layers.Dense(rng.integers(1,max_layer_size)* layers_scale_factor) for _ in range(rng.integers(1, max_layers))]
        ])
        model.build()
        return model

    GEN_FNS = {
        "small": lambda: make_small_models(args.max_layer_size, args.layers_scale_factor),
        "shallow": lambda: make_shallow_models(args.max_layer_size, args.max_layers, args.layers_scale_factor)
    }

    def fake_key():
        return str(uuid.uuid4())

    transfer_method = TRANSFER[args.transfer_method]
    gen_fn = GEN_FNS[args.method]

    timings = []
    for i in range(args.n_iter):
        model = gen_fn()
        key = f"{i}-{rank}-{fake_key()}"

        begin = time.perf_counter()
        standard_model = standardize_names(model)
        end = time.perf_counter()
        standardize_time = end-begin

        begin = time.perf_counter()
        transfer_method.transfer(standard_model)
        end = time.perf_counter()
        transfer_time = end-begin

        begin = time.perf_counter()
        transfer_method.store(key, model=standard_model,  prefix=[])
        end = time.perf_counter()
        store_time = end-begin


        timings.append(Timings(standardize_time=standardize_time, store_time=store_time, transfer_time=transfer_time))
        del model
        del standard_model
        gc.collect()

    all_results = MPI.COMM_WORLD.gather(timings, root=0)

    if rank == 0:
        df = pd.DataFrame(itertools.chain(*all_results))
        df.to_csv("timings.csv")


