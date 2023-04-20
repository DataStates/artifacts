import os
import tensorflow as tf
from .combo_search_space import create_combo_search_space
from deephyper.problem import NaProblem
from deephyper.nas.operation import operation
from .. import misc_utils
import os.path
from .global_load_data import combo_load_data 


class ComboProblem:
    def __init__(self, train_path, batch_size=32, num_epochs=1, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.training_data_path = train_path
        return

    def setup_problem(self, **kwargs):
        Problem = NaProblem()

        Problem.load_data(
                combo_load_data,
                train_path=self.training_data_path
        )
        Problem.search_space(create_combo_search_space, num_layers=5)

        Problem.hyperparameters(
            lsr_batch_size=True,
            lsr_learning_rate=True,
            batch_size=self.batch_size,
            learning_rate=0.001,
            optimizer="adam",
            num_epochs=self.num_epochs,
            verbose=0,
            callbacks=dict(
                ReduceLROnPlateau=dict(monitor="val_r2", mode="max", verbose=0, patience=5),
                EarlyStopping=dict(
                monitor="val_r2", min_delta=0, mode="max", verbose=0, patience=10
                ),
            )
        )
        Problem.loss("mse")
        Problem.metrics(["r2"])
        Problem.objective("val_r2")
        return Problem

    def test_problem(self):
        return

    def get_baseline_model(self):
        return
