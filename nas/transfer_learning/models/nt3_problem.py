import os
import tensorflow as tf
from .nt3_search_space import create_nt3_search_space
from deephyper.problem import NaProblem
from deephyper.nas.operation import operation
from .. import misc_utils
import os.path
from .global_load_data import load_preproc_nt3_data_from_file

Dense = operation(tf.keras.layers.Dense)


class NT3Problem:
    def __init__(
        self,
        train_path,
        test_path,
        batch_size=100,
        num_epochs=50,
        download=False,
        problem_size="small",
        **_kwargs
    ):
        assert train_path is not None, "You must provide a training file path"
        assert test_path is not None, "You must provide a test file path"
        assert os.path.exists(
            train_path
        ), f"Cannot find training data file path {train_path}"
        assert os.path.exists(
            test_path
        ), f"Cannot find training data file path {test_path}"

        self.training_data_path = train_path
        self.test_data_path = test_path
        self.num_classes = 2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.problem_size = problem_size
        return

    def setup_problem(self, *args, **kwargs):
        problem = NaProblem()
        problem.load_data(
            load_preproc_nt3_data_from_file,
            train_path=self.training_data_path,
            test_path=self.test_data_path,
            num_classes=self.num_classes,
        )
        problem.search_space(create_nt3_search_space, problem_size=self.problem_size)
        # TODO: Read this from file or cmdargs instead of hard-coding
        problem.hyperparameters(
            batch_size=self.batch_size,
            learning_rate=0.001,
            optimizer="adam",
            num_epochs=self.num_epochs,
        )
        problem.loss("categorical_crossentropy")

        problem.metrics(["acc"])

        problem.objective("val_acc__last")
        self.problem = problem
        return problem

    def test_problem(self):
        if self.problem is None:
            self.setup_problem()

        prob = self.problem.build_search_space()

        arch_seq = misc_utils.generate_a_random_archseq(prob)
        prob.set_ops(arch_seq)
        model = prob.create_model()
        model.summary()

    def get_baseline_model(self):
        return
