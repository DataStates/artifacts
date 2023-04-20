from .application import Application
from .nt3_problem import NT3Problem
from .mnist import MNISTProblem
from .synthetic_problem import SyntheticProblem
from .attn_problem import AttnProblem
from .combo_problem import ComboProblem

_PROBLEMS = {
        "nt3": NT3Problem,
        "mnist": MNISTProblem,
        "synthetic": SyntheticProblem,
        "attn": AttnProblem,
        "combo": ComboProblem
}


def make_problem(problem: str) -> Application:
    return _PROBLEMS[problem]
