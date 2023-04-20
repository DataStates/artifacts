from deephyper.search.nas import NeuralArchitectureSearch, RegularizedEvolution
from .expiring import ExpiringRegularizedEvolution
from typing import List

_METHODS = {
    "expiring": ExpiringRegularizedEvolution,
    "regularized": RegularizedEvolution,
}


def search_methods() -> List[str]:
    """returns the list of search methods supported"""
    return list(_METHODS.keys())


def make_search(search_method: str) -> NeuralArchitectureSearch:
    """returns a constructor function for a search method"""
    return _METHODS[search_method]
