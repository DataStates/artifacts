from mpi4py import MPI
from typing import Optional, List, Tuple, Any
import tensorflow as tf
import numpy as np
import abc

maxloc_dtype = np.dtype([('value', 'i4'), ('pos', 'i4')])

class TransferMethod(metaclass=abc.ABCMeta):
    @staticmethod
    def startup_server(ds_colocated=False, **kwargs) -> Tuple[List[str], List[Any]]:
        """returns the MPI communicator and host addresses"""
        print("rank in client: ", MPI.COMM_WORLD.rank)
        print("", flush=True)
        print("size in client: ", MPI.COMM_WORLD.size)
        print("", flush=True)
        work_comm = MPI.COMM_WORLD.Split(0, MPI.COMM_WORLD.rank)
        rank = np.zeros(1, dtype=maxloc_dtype)
        rank[0]['value'] = (work_comm.rank == 0 and MPI.COMM_WORLD.rank != 0)
        rank[0]['pos'] = MPI.COMM_WORLD.rank
        remote_leader = np.zeros(1, dtype=maxloc_dtype)
        MPI.COMM_WORLD.Allreduce([rank, MPI.INT_INT], [remote_leader, MPI.INT_INT],
                                 op=MPI.MAXLOC)
        intercomm = work_comm.Create_intercomm(0, MPI.COMM_WORLD,
                                               remote_leader[0]['pos'], tag=0)
        
        total_size = np.zeros(1, dtype=np.int64)
        intercomm.Bcast(total_size, root=0)
        addrs = np.zeros(total_size[0], dtype=np.dtype('c'))
        intercomm.Bcast(addrs, root=0)
        intercomm.Free()
        addrs = list(map(lambda x: x.decode(), filter(lambda x: len(x), bytes(addrs).split(b'\0'))))
        
        return work_comm, addrs

    @abc.abstractmethod
    def teardown_server(self):
        MPI.COMM_WORLD.Barrier()

    @abc.abstractmethod
    def transfer(
        self, model: tf.keras.Model, id: str, hint: Optional[str] = None
    ) -> Tuple[tf.keras.Model, List[str]]:
        """transfer weights into the model, returning the transfer model and list of layer id's transfered"""
        pass

    @abc.abstractmethod
    def store(self, id: str, model: tf.keras.Model, prefix: List[str]) -> str:
        """store weights; empty prefix means store everything; returns the id of a model"""
        pass

    @abc.abstractmethod
    def retire_model(self, id: str):
        """removes a model and its weights"""
        pass

    @abc.abstractmethod
    def retain(self, parent: str, child: str):
        """allow explicit retention of a model"""
        pass

    def _best_match(self, model: tf.keras.Model) -> Tuple[Optional[str], List[str]]:
        """returns the model that best matches a given (model, list_of_tranfered_layers)"""
        pass
