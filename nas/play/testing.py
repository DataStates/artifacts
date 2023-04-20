#!/usr/bin/env python
from deephyper.problem import NaProblem
from deephyper.benchmark.nas.linearReg.load_data import load_data
from deephyper.nas.preprocessing import minmaxstdscaler
from deephyper.nas.run import run_base_trainer
from deephyper.search.nas import RegularizedEvolution
from deepspace.tabular import OneLayerSpace
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback


def create_search_space(input_shape, output_shape, **kwargs):
    return OneLayerSpace(input_shape, output_shape)


problem = NaProblem()
problem.load_data(load_data)
problem.preprocessing(minmaxstdscaler)
problem.search_space(create_search_space)
problem.hyperparameters(
    batch_size=100,
    learning_rate=0.1,
    optimizer='adam',
    num_epochs=10,
    callbacks=dict(
       EarlyStopping=dict(
            monitor='val_r2',
            mode='max',
            verbose=0,
            patience=5
        )
    )
)
problem.loss('mse')
problem.metrics(['r2'])
problem.objective('val_r2__last')

evaluator = Evaluator.create(
        run_base_trainer,
        method="ray",
        method_kwargs={
             "address": None,
             "num_cpus": 6,
             "num_cpus_per_task": 1,
             "callbacks": [LoggerCallback()]
        })

search = RegularizedEvolution(problem, evaluator)
results = search.search(10)
