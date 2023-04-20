from deephyper.problem import NaProblem
import abc


class Application(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def setup_problem(self) -> NaProblem:
        pass
