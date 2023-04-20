import os
import tensorflow as tf
from .attn_search_space import create_attn_search_space
from deephyper.problem import NaProblem
from deephyper.nas.operation import operation
from .. import misc_utils
import os.path
from .global_load_data import attn_load_data 


class AttnProblem:
    def __init__(self, train_path, batch_size=32, num_epochs=1, rel_size='small', **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.training_data_path = train_path
        self.rel_size = rel_size
        return

    def setup_problem(self, **kwargs):
        Problem = NaProblem()
        Problem.load_data(
                attn_load_data, 
                train_path=self.training_data_path
        )
        
        Problem.search_space(create_attn_search_space, num_layers=5, size=self.rel_size)

        Problem.hyperparameters(
            lsr_batch_size=True,
            lsr_learning_rate=True,
            batch_size=self.batch_size,
            learning_rate=0.001,
            optimizer="adam",
            num_epochs=self.num_epochs,
            verbose=0,
            callbacks=dict(
                ReduceLROnPlateau=dict(monitor="val_aucpr", mode="max", verbose=0, patience=5),
                EarlyStopping=dict(
                monitor="val_aucpr", min_delta=0, mode="max", verbose=0, patience=10
                ),
            )
        )

        Problem.loss(
            "categorical_crossentropy",
            class_weights={0: 0.5186881480859765, 1: 13.877462488516892}
        )

        Problem.metrics(["acc", "auroc", "aucpr"])

        Problem.objective("val_aucpr")
        return Problem

    def test_problem(self):
        return

    def get_baseline_model(self):
        return
