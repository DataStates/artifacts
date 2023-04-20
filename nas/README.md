# experiments

Ad-hoc code repository needed to run initial evaluations

## `bench_writes.py`

This benchmark looks at the walltime of writing to a filesystem including GPU setup time (no sync per task)

## `bench_writes-2.py`

This benchmark looks at the walltime of writing to a filesystem not including GPU setup time and sync each task

## `bench_writes-3.py`

This benchmark looks at the walltime of writing to memory not including GPU setup time and syncs after each task

## `keras-freeze.py`

an example of how to set layers as frozen

## `load_single.py`

Looks at the runtime of reading a subset of layers from the filesystem


## `save_bench.py`

An earlier version of bench_writes.py


## `testing.py`

A minimal example of distributed tracing using some simplistic NAS benchmark from deephyper.

We'll use this to start writing the nas search with transfer-learning.

## `transfer_learn.py`

A minimal example of distributed tracing where we've started to implement transfer learning

We've added `TransferTrainer` which transfers the model weights  from a prior run.
It will have a method `transfer` and `store` that uses a `TrasferMethod` to transfer the weights before/after training.
The `TransferTrainer` will handle freezing and unfreezing of the layers after they are transferred.

This requires `run_transfer_trainer` which is almost exactly `run_base_trainer` except that it uses `TransferTrainer` instead of `BaseTrainer`.

## `find_prefix.py`

Code that explores how to name the model layers in a stable way

## `bench_transfer.py`

Code that benchmarks transfer methods

## `birthday.py`

Code that estimates the number of layers expected to transfer

## `transfer_methods.py`

Implementations of the transfer learning code

## `gen_models.py`

Code that simply generates a large number of models as quickly as possible

## graph-meta

MPI-based metadata service. Implements the graph-based prefix calculation in distributed fashion.

## cpp-store

Distributed C++ implementation based on Thallium. Servers need to be started using a connection string of the form: ofi+tcp://<ip>:port>. 
Clients need to define the MODEL_SERVERS environment variable that contains a list of server connection strings separated by spaces.


## `gen_models_with_pop.py`

Code that generates a large number of models as quickly as possible and then preforms a process that resembles regularized evolution to make small changes to the model

To get the initial population:

```bash
mpiexec -np $count python -m mpi4py ./gen_models_with_pop.py --output_dir=/foo  --population_size=60000
```

You can then run it again later with to evolve some count of the population

```bash
mpiexec -np $count python -m mpi4py ./gen_models_with_pop.py --output_dir=/foo_evolved --initial_population_dir=/foo --evolve_count=10000
```

Download Candle datasets:
https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/normal-tumor/
