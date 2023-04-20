#!/usr/bin/env python
from __future__ import annotations
from mpi4py import MPI
from transfer_learning.mock_model import MockModel
from transfer_learning import transfer_methods
import time
import sys
import itertools
import argparse
import socket
import math
import pandas as pd
from contextlib import ExitStack
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--skip_init",
    action="store_true",
    help="reuse existing redis db and do not add models",
)
parser.add_argument("--replicas", type=int, default=10, help="query replicas per rank")
parser.add_argument(
    "--models_10k",
    type=int,
    default=1,
    help="approximately how many 10,000 models to add across all ranks",
)
parser.add_argument("--host", default="localhost", help="redis host to connect to")
parser.add_argument("--port", type=int, default=7000, help="redis port to connect to")
parser.add_argument(
    "--evostore_servers", type=int, default=1, help="number of evostore_servers"
)
parser.add_argument(
    "--evostore_threads",
    type=int,
    default=1,
    help="number of worker threads for evostore_servers",
)
parser.add_argument(
    "--transfer_method",
    default="redis",
    help="which transfer method to use",
    choices=transfer_methods.transfer_methods(),
)
parser.add_argument(
    "--trace_dir",
    type=Path,
    default=Path("./model_traces/"),
    help="directory containing traces to train from",
)
parser.add_argument(
    "--query_trace_dir",
    type=Path,
    default=Path("./model_traces/"),
    help="directory containing traces to query from",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("./model_traces/"),
    help="directory to output training results to",
)
parser.add_argument(
    "--use_full_model",
    action="store_true",
    help="use tf.Model instead of MockModel",
)
args = parser.parse_args()

class FullModel:
    @staticmethod
    def from_json(s: str):
        return tf.keras.models.from_json(s, custom_objects={})

model_cls = FullModel if args.use_full_model else MockModel

if args.use_full_model:
    import tensorflow as tf


transfer_method_cls = transfer_methods.make_transfer_method(args.transfer_method)
workcomm, conn_strs = transfer_method_cls.startup_server()
rank = workcomm.rank
size = workcomm.size
print("client(py)", workcomm, conn_strs, flush=True)

def gather_and_store_timings(timings, filename):
    all_timings = workcomm.gather(timings)
    if rank == 0:
        all_timings = pd.Series(itertools.chain(*all_timings))
        print(all_timings.describe(), flush=True)
        all_timings.to_csv(filename)

workcomm.Barrier()
if rank == 0:
    print("starting storing", file=sys.stderr, flush=True)

method = transfer_method_cls(hosts=conn_strs)

traces_per_rank = int(math.ceil(10000 * args.models_10k / size))
if traces_per_rank < 1:
    print("traces_per_rank must be >=1", file=sys.stderr, flush=True)
    workcomm.Abort(1)

try:
    store_times = []
    with ExitStack() as stack:
        trace_files = list(sorted(args.trace_dir.glob("models*.json")))
        trace_files_for_rank = list(
            itertools.islice(
                itertools.islice(
                    itertools.cycle(trace_files), max(len(trace_files), size)
                ),  # ensure we have >=1 file per process
                rank,
                None,
                size,
            )
        )  # each process takes rank, rank+size, ... , rank+k*size
        print(
            "training rank ",
            rank,
            " takes files ",
            trace_files_for_rank,
            file=sys.stderr,
            flush=True,
        )
        model_trace_lines = itertools.chain(
            *[stack.enter_context(f.open()) for f in trace_files]
        )
        if not args.skip_init:
            for idx, line in enumerate(
                itertools.islice(itertools.cycle(model_trace_lines), traces_per_rank)
            ):
                m = model_cls.from_json(line)
                begin = time.perf_counter()
                method.store(str(idx + rank * traces_per_rank), m, [])
                end = time.perf_counter()
                store_times.append(end - begin)
        else:
            m = model_cls.from_json(next(model_trace_lines))

    gather_and_store_timings(store_times, args.output_dir / "redis-store.csv")

    workcomm.Barrier()
    if rank == 0:
        print("starting queries", file=sys.stderr, flush=True)

    query_times = []
    with ExitStack() as stack:
        trace_files = list(sorted(args.query_trace_dir.glob("models*.json")))
        trace_files_for_rank = list(
            itertools.islice(
                itertools.islice(
                    itertools.cycle(trace_files), max(len(trace_files), size)
                ),  # ensure we have >=1 file per process
                rank,
                None,
                size,
            )
        )  # each process takes rank, rank+size, ... , rank+k*size
        print(
            "query rank ",
            rank,
            " takes files ",
            trace_files_for_rank,
            file=sys.stderr,
            flush=True,
        )
        model_trace_lines = itertools.chain(
            *[stack.enter_context(f.open()) for f in trace_files]
        )
        for idx, line in enumerate(
            itertools.islice(itertools.cycle(model_trace_lines), args.replicas)
        ):
            m = model_cls.from_json(line)
            begin = time.perf_counter()
            print("query start", flush=True)
            method._best_match(m)
            print("query end", flush=True)
            end = time.perf_counter()
            query_times.append(end - begin)

    gather_and_store_timings(query_times, args.output_dir / "redis-query.csv")
except Exception as ex:
    print("Caught Exception", ex, flush=True)

workcomm.Barrier()
method.teardown_server(workcomm)
MPI.Finalize()
print("finalized client", rank, flush=True)
