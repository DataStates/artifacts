#!/usr/bin/env python3
from deepspace.tabular import DenseSkipCoSpace, FeedForwardSpace
import argparse
import gc
import os
import copy
import json
import sys
import tensorflow as tf
import numpy as np
from typing import Any
from mpi4py import MPI
from pathlib import Path
from transfer_learning.transfer_methods._common import standardize_names


rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size


def infer_space_arg(arg: str) -> (str, Any):
    name, value = arg.split("=")
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return name, value


parser = argparse.ArgumentParser()
parser.add_argument("--population_size", type=int, default=10)
parser.add_argument("--evolve_count", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output_dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp"))
)
parser.add_argument("--initial_population_dir", type=Path)
parser.add_argument("--space", choices=["skip", "seq"], default="skip")
parser.add_argument(
    "--space_extra_arg", type=infer_space_arg, action="append", default=[]
)
args = parser.parse_args()
args.seed += rank
args.population_size //= size
args.evolve_count //= size
assert (
    args.population_size >= args.evolve_count
), f"the evolve_count={args.evolve_count} must be ≤ pop_size={args.population_size}"


def sample_pop_randomly(space_list, rng):
    return [rng.choice(upper_bound + 1) for (_, upper_bound) in space_list]


def evolve(arch, space_list, rng):
    mutant = copy.deepcopy(arch)
    layer = rng.choice(len(space_list))
    mutant[layer] = rng.choice(space_list[layer][1])

    return mutant


with tf.device("/gpu:%d" % rank):
    space_args = {
        "input_shape": (32,),
        "output_shape": (1,),
    }
    space_args.update(args.space_extra_arg)
    if rank == 0:
        print("space_args", space_args, file=sys.stderr)

    if args.space == "skip":
        space = DenseSkipCoSpace(**space_args)
    elif args.space == "seq":
        space = FeedForwardSpace(**space_args)
    else:
        raise NotImplementedError(f"unexpected space type {args.space}")
    space.build()

    rng = np.random.RandomState(args.seed)

    space_list = space.choices()
    if args.initial_population_dir is None:
        population = [
            sample_pop_randomly(space_list, rng) for i in range(args.population_size)
        ]
    else:
        with open(
            args.initial_population_dir / ("metadata_%d.json" % rank)
        ) as pop_file:
            population = json.load(pop_file)["population"]

    for i in range(args.evolve_count):
        population[i] = evolve(population[i], space_list, rng)

    with open(args.output_dir / ("metadata_%d.json" % rank), "w") as metadata_file:
        metadata = {
            "seed": args.seed,
            "population": population,
        }
        json.dump(metadata, metadata_file)

    with open(args.output_dir / ("models_%d.json" % rank), "w") as model_file:
        for i, arch_seq in enumerate(population):
            print(f"rank {rank} generating model: {i}")
            s = standardize_names(space.sample(arch_seq))
            print(s.to_json(), file=model_file)
            del s
            gc.collect()
