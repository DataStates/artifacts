#!/usr/bin/env python
"""main entry point for transfer_learning"""
import logging
import tensorflow as tf
import transfer_learning as tl
import socket
from deephyper.evaluator import Evaluator
from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

logger = logging.getLogger(__name__)


args = tl.cli.parse_args()
if args.debug:
    logging.basicConfig(level=logging.DEBUG)

setup = tl.setup.make_experimental_setup(args.load_data_mode)(args)
setup.setup_gpus()
if args.rand_seed is not None:
    tf.random.set_seed(args.rand_seed)

transfer_method_cls = tl.transfer_methods.make_transfer_method(args.transfer_method)
'''
hosts = MPI.COMM_WORLD.gather(socket.gethostbyname(socket.gethostname()))

if rank == 0:
    conn_strs, procs = transfer_method_cls.startup_server(hosts=hosts, port=args.port)
else:
    conn_strs = [None]
    procs = [None]
conn_strs = MPI.COMM_WORLD.bcast(conn_strs)
print("RANK", rank, conn_strs)

my_transfer_method = transfer_method_cls(
        hosts=conn_strs,
        port=args.port,
        bulk_storage_path=args.bulk_storage_path,
        debug=args.debug
        )
'''
my_transfer_method = transfer_method_cls()
try:
    problem = tl.models.make_problem(args.application)(
            train_path=args.train_data_path,
            test_path=args.train_data_path,
            synthetic_space=args.synthetic_space
            ).setup_problem()
    setup.load_data(problem)

    with Evaluator.create(
        tl.trainer.build_transfer_trainer(
            my_transfer_method,
            tl.trainer.TrasferTrainer,
            args.save_result_dir,
            args.num_epochs,
            args.load_data_mode,
            **setup.trainer_method_kwargs()
        ),
        method=args.load_data_mode,
        method_kwargs=setup.evaluator_method_kwargs(),
    ) as evaluator:
        if evaluator is not None:
            search = tl.search.make_search(args.search_method)(
                problem,
                evaluator,
                log_dir=args.save_result_dir,
                random_state=args.rand_seed,
                transfer_method=my_transfer_method,
                sample_size=args.sample_size,
                population_size=args.population_size,
                beta=args.beta,
            )
            results = search.search(args.search_attempts)

finally:
    procs = 0
    transfer_method_cls.teardown_server(procs)
    setup.teardown()
