from deephyper.problem import NaProblem
from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
import json
import requests
from tqdm import tqdm
import os
from typing import List
import random


def get_arch_seq_ranges(ss: KSearchSpace):
    opranges = []
    for n in ss.nodes:
        if isinstance(n, VariableNode):
            opranges.append(n.num_ops)

    return opranges


def get_search_space_size(ss: KSearchSpace) -> int:
    opranges = get_arch_seq_ranges(ss)
    res = functools.reduce(lambda x, y: x * y, opranges)
    print(res)
    return res


def generate_a_random_archseq(ss: KSearchSpace) -> List[int]:
    opranges = get_arch_seq_ranges(ss)
    for k in opranges:
        assert k > 1

    return [random.randint(0, k - 1) for k in opranges]
