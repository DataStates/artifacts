import logging
import os
from deephyper.problem import NaProblem
from deephyper.nas.run._util import setup_data
from ._base import ExperimentSetup, train_data_global


class MPIExperimentSetup(ExperimentSetup):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def setup_gpus(self, ds_colocated=False):
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.threads = True
        mpi4py.rc.thread_level = "multiple"
        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init_thread()

        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        gpu_per_node = len(gpus)
        if ds_colocated:
            rank = MPI.COMM_WORLD.Get_rank()
            num_nodes = MPI.COMM_WORLD.Get_size()  / 6
            gpu_local_idx = (int(rank/num_nodes)-1) % gpu_per_node
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            gpu_local_idx = (rank-1) % gpu_per_node
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

        pass

    def load_data(self, problem: NaProblem):
        global train_data_global
        train_data_global = setup_data(problem.space)

    def evaluator_method_kwargs(self):
        return {"comm": self.workcomm}

    def trainer_method_kwargs(self):
        return {}

    def teardown(self):
        pass
