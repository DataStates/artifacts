from mpi4py import MPI
import threading

class MPIRPC(threading.Thread):
    QUERY = 1
    REPLY = 2
    def __init__(self):
        super().__init__()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def handler(self, args):
        return None

    def aggregate(self, result, reply):
        return None

    def run(self):
        status = MPI.Status()
        while True:
            cmd, msg = self.comm.recv(status=status, tag=MPIRPC.QUERY)
            self.comm.send(self.handler(cmd, msg), dest=status.Get_source(), tag=MPIRPC.REPLY)

    def single_query(self, cmd, args, rank):
        self.comm.send((cmd, args), dest=rank, tag=MPIRPC.QUERY)
        return self.comm.recv(source=rank, tag=MPIRPC.REPLY)
            
    def reduce_query(self, cmd, args):
        reqs = []
        result = None
        for i in range(self.comm.size):
            reqs.append(self.comm.isend((cmd, args), dest=i, tag=MPIRPC.QUERY))
        MPI.Request.waitall(reqs)
        for i in range(self.comm.size):
            reqs[i] = self.comm.irecv(source=i, tag=MPIRPC.REPLY)
        completed = self.comm.size
        while completed > 0:
            _, reply = MPI.Request.waitany(reqs)
            result = reply if result is None else self.aggregate(result, reply)
            completed -= 1
        return result
