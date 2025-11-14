"""Base class for a parameter scorer that returns how good a certain parameter is."""

import abc
from typing import Any, TypeVar

_X = TypeVar("_X")  # state
Datastore = list[tuple[Any]]
Labels = list[Any]


class ParameterScorer(abc.ABC):
    """Base class for a parameter scorer that returns how good a certain parameter
    is."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, features: Datastore, labels: Labels):
        """Given training data, update parameter scorer."""

    @abc.abstractmethod
    def score(self, x: _X, params: Any) -> float:
        """Score the parameter given the low-level state."""
