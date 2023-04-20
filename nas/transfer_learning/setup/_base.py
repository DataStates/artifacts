import abc
from deephyper.problem import NaProblem

# global for use by setup classes to store a global id for training
train_data_global = None


class ExperimentSetup(metaclass=abc.ABCMeta):
    def setup_gpus(self):
        """setup gpus for the runtime envionment"""
        pass

    def load_data(self, problem: NaProblem):
        """load the data into a fast storage tier"""
        pass

    def evaluator_method_kwargs(self):
        """return args for the evaluator"""
        pass

    def trainer_method_kwargs(self):
        """return args for the trainer"""
        pass

    def teardown(self):
        """teardown the experiment"""
        pass
