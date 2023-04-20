#!/usr/bin/env python

import os
import sys
import argparse
import math
import random
from pathlib import Path
import tensorflow.keras as kf


def to_human(i: int) -> str:
    if i < 1024:
        return f"{i}B"
    elif i < 1024**2:
        return f"{i//1024}kB"
    elif i < 1024**3:
        return f"{i//(1024**2)}MB"
    else:
        return f"{i//(1024**3)}GB"


def from_human(s: str) -> int:
    if s.endswith("g"):
        return int(s.split("g")[0]) * 1024**3
    elif s.endswith("m"):
        return int(s.split("m")[0]) * 1024**2
    elif s.endswith("k"):
        return int(s.split("k")[0]) * 1024
    else:
        raise RuntimeError(f"unable to parse \"{s}\"")


parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--num_transfer", type=int, default=1)
parser.add_argument("--total_size", type=from_human, default=1024**2)
parser.add_argument("--variance", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--outdir", type=Path, default=Path(os.environ.get('TMPDIR', "/tmp")))
args = parser.parse_args()
print(args)

size_per_layer = int(args.total_size / (args.num_layers * 4))
size_per_layer = int((-1 + math.sqrt(-1 + 1+4*size_per_layer)) / 2)
print(size_per_layer)
assert size_per_layer - args.variance > 0


#i = 0
#m = kf.Sequential()
#m.add(kf.Input(32, name="input"))
#
#random.seed(args.seed)
#for i in range(args.num_layers+1):
#    layer_size = random.randint(
#            size_per_layer - args.variance, size_per_layer + args.variance
#        )
#    m.add(kf.layers.Dense(layer_size, name=f"{'transfer' if i < args.num_transfer else 'notransfer'}{i}"))
#
#outfile = args.outdir / f"{to_human(args.total_size)}s-{args.num_layers}l-{args.variance}v-{args.num_transfer}t-{args.seed}seed.h5"
#m.save(outfile)
#print(m.summary(), file=sys.stderr)
#st = Path(outfile).stat()
#print(outfile, to_human(st.st_size))
