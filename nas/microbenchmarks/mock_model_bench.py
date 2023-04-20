#!/usr/bin/env python
from mock_model import MockModel
import transfer_methods
import json
import time
import itertools
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--skip_init", action="store_true")
parser.add_argument("--replicas", type=int, default=10)
args = parser.parse_args()


method = transfer_methods.TransferHDF5(port=7000)


#python itertools recipe
def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))

times = []
with open("./model_traces/models_1.json") as model_traces:
    if not args.skip_init:
        for idx, line in enumerate(ncycles(model_traces,8)):
            m = MockModel.from_json(line)
            begin = time.perf_counter()
            method.store(str(idx), m, [])
            end = time.perf_counter()
            times.append(end-begin)
        times = pd.Series(times)
        times.to_csv("model_traces/redis-store.csv")
        print(times.describe())
    else:
        m = MockModel.from_json(next(model_traces))
            
times = []
for i in range(args.replicas):
    begin = time.perf_counter()
    method._best_match(m)
    end = time.perf_counter()
    times.append(end-begin)

times = pd.Series(times)
times.to_csv("model_traces/redis-query.csv")
print(times.describe())
