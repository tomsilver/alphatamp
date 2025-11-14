"""A naive parameter scorer that only returns 1."""

from typing import Any, List, Tuple, TypeVar

from alphatamp.approaches.parameter_scorers.base_parameter_scorer import ParameterScorer

_X = TypeVar("_X")  # state
Datastore = List[Tuple[Any]]
Labels = List[Any]


class NaiveScorer(ParameterScorer):
    """A naive parameter scorer that only returns 1."""

    def __init__(self):
        pass

    def train(self, features: Datastore, labels: Labels):
        """Given training data, update parameter scorer."""

    def score(self, x: _X, params: Any) -> float:
        """Score the parameter given the low-level state."""
        return 1
