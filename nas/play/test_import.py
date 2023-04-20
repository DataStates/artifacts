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
from nt3_problem import NT3Problem
from synthetic_problem import SyntheticProblem
from uno_problem import UnoProblem
from pathlib import Path
import ray
import os, sys
import pickle
from deephyper.nas.metrics import r2, acc
import keras
import transfer_methods
