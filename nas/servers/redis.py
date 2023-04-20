#!/usr/bin/env python
import argparse
import subprocess as sp
import numpy as np
import socket
import time
from mpi4py import MPI
parser = argparse.ArgumentParser()
parser.add_argument("--port", '-p', type=int, default=6500)
args = parser.parse_args()


print("pre split", flush=True)
work_comm = MPI.COMM_WORLD.Split(1, MPI.COMM_WORLD.rank)

maxloc_dtype = np.dtype([('value', 'i4'), ('pos', 'i4')])
rank = np.zeros(1, dtype=maxloc_dtype)
rank[0]['value'] = (work_comm.rank == 0 and MPI.COMM_WORLD.rank != 0)
rank[0]['pos'] = MPI.COMM_WORLD.rank
remote_leader = np.zeros(1, dtype=maxloc_dtype)
MPI.COMM_WORLD.Allreduce([rank, MPI.INT_INT], [remote_leader, MPI.INT_INT],
                         op=MPI.MAXLOC)
intercomm = work_comm.Create_intercomm(0, MPI.COMM_WORLD, 0, tag=0)
print(f"server(py)  global={MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size},"
      f" local={work_comm.rank}/{work_comm.size}", flush=True)

if work_comm.rank == 0:
    proc = sp.Popen(
        [
            "redis-server",
            "--port",
            str(args.port),
            "--protected-mode",
            "no"
        ]
    )
    time.sleep(30)

total_size = np.zeros(1, dtype=np.int64)
addr_b = b"\0".join(
        [b"".join([
            b"redis://",
            socket.gethostbyname(socket.gethostname()).encode(),
            f":{args.port}".encode()
            ])]
        ) + b"\0"
total_size[0] = len(addr_b)
intercomm.Bcast(total_size, root=(MPI.ROOT if work_comm.rank == 0 else MPI.PROC_NULL))
intercomm.Bcast(addr_b, root=(MPI.ROOT if work_comm.rank == 0 else MPI.PROC_NULL))
intercomm.Free()

MPI.COMM_WORLD.Barrier()
print("shutdown server", MPI.COMM_WORLD.rank)
if work_comm.rank == 0:
    proc.kill()
