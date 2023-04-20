from ._datastates import DataStatesModelRepo
from ._redis import TransferHDF5
from ._simplehdf5 import TransferSimpleHDF5
from ._noop import TransferNoop
from ._base import TransferMethod

_METHODS = {
    "datastates": DataStatesModelRepo,
    "redis": TransferHDF5,
    "simplehdf5": TransferSimpleHDF5,
    "noop": TransferNoop,
}


def transfer_methods():
    return list(_METHODS.keys())


def make_transfer_method(method: str) -> TransferMethod:
    return _METHODS[method]
