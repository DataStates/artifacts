from ._base import ExperimentSetup
from deephyper.evaluator.callback import LoggerCallback
from deephyper.problem import NaProblem


class RayExperimentSetup(ExperimentSetup):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args

    def setup_gpus(self):
        pass

    def load_data(self, problem: NaProblem):
        pass

    def evaluator_method_kwargs(self):
        return {
            "address": "auto",
            "num_cpus": self.args.ncpus,
            "num_gpus": self.args.ngpus,
            "num_cpus_per_task": self.args.num_cpus_per_task,
            "num_gpus_per_task": self.args.num_gpus_per_task,
            "callbacks": [LoggerCallback()],
            # , SearchEarlyStopping(patience=args.patience)],
        }

    def trainer_method_kwargs(self):
        return {"dat_id": self.dat_id}

    def teardown(self):
        pass
