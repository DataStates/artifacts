from deephyper.nas.operation import operation
from deephyper.problem import NaProblem
from global_load_data import load_uno_data_from_file, load_uno_data_fake
from uno_search_space import create_uno_search_space
import copy
import misc_utils
import tensorflow as tf

Dense = operation(tf.keras.layers.Dense)


class UnoProblem:
    def __init__(
        self, dataset_path=None, batch_size=100, num_epochs=50, download=False
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def setup_problem(self, *args, **kwargs):
        problem = NaProblem()
        if self.dataset_path is None:
            problem.load_data(load_uno_data_fake)
        else:
            problem.load_data(load_uno_data_from_file, self.dataset_path)
        problem.search_space(create_uno_search_space)
        problem.hyperparameters(
            batch_size=self.batch_size,
            learning_rate=0.001,
            optimizer="adam",
            num_epochs=self.num_epochs,
        )
        problem.loss("mae")
        problem.metrics(["r2"])
        problem.objective("val_r2__last")
        self.problem = problem
        return problem

    def test_problem(self):
        if self.problem is None:
            self.setup_problem()

        problem = copy.deepcopy(self.problem)
        prob = problem.build_search_space()

        arch_seq = misc_utils.generate_a_random_archseq(prob)
        prob.set_ops(arch_seq)
        model = prob.create_model()
        model.summary()

    def get_baseline_model(self):
        return
