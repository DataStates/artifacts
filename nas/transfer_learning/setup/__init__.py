from ._mpi import MPIExperimentSetup
from ._ray import RayExperimentSetup
from ._base import ExperimentSetup, train_data_global

_METHODS = {
    "mpicomm": MPIExperimentSetup,
    "ray": RayExperimentSetup,
}


def make_experimental_setup(method: str) -> ExperimentSetup:
    return _METHODS[method]
