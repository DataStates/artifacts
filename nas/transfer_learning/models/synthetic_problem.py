import tensorflow as tf
from deephyper.problem import NaProblem
from deephyper.nas.preprocessing import minmaxstdscaler
from deephyper.nas.operation import operation
from deepspace.tabular import FeedForwardSpace, DenseSkipCoSpace

Dense = operation(tf.keras.layers.Dense)
try:
    from deephyper.benchmark.nas.linearReg import load_data
except ImportError:
    from .benchmark_linearReg import load_data


def create_feed_forward_space(input_shape, output_shape, *args, **kwargs):
    return FeedForwardSpace(
        input_shape, output_shape, num_units=(1, 5), *args, **kwargs
    )


def create_dense_skip_co_space(input_shape, output_shape, *args, **kwargs):
    return DenseSkipCoSpace(input_shape, output_shape, *args, **kwargs)


class SyntheticProblem:
    def __init__(self, batch_size=100, num_epochs=30, space="feed_forward", **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.space = space
        return

    def __name__(self):
        return str("SyntheticProblem")

    def setup_problem(self):
        problem = NaProblem()
        problem.load_data(load_data)
        problem.preprocessing(minmaxstdscaler)
        if self.space == "feed_forward":
            problem.search_space(create_feed_forward_space)
        else:
            problem.search_space(create_dense_skip_co_space)

        problem.hyperparameters(
            batch_size=self.batch_size,  # TODO: change
            learning_rate=0.1,
            optimizer="adam",
            num_epochs=self.num_epochs,  # TODO: change
            callbacks=dict(
                EarlyStopping=dict(monitor="val_r2", mode="max", verbose=0, patience=5)
            ),
        )
        problem.loss("mse")
        problem.metrics(["r2"])
        problem.objective("val_r2__last")
        self.problem = problem
        return problem

    def test_problem(self):
        """
        if self.problem is None:
            setup_problem()

        prob = self.problem.build_search_space(seed=self.seed)

        arch_seq = misc_utils.generate_a_random_archseq(prob)
        prob.set_ops(arch_seq)
        model=prob.create_model()
        model.summary()
        """
