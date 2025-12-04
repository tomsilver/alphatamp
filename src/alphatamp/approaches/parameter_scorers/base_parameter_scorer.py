"""Base class for a parameter scorer that returns how good a certain parameter is."""

import abc
from typing import Any, TypeVar

import numpy as np

_O = TypeVar("_O")  # observation


class ParameterScorer(abc.ABC):
    """Base class for a parameter scorer that returns how good a certain parameter
    is."""

    def __init__(self, configs: dict):
        pass

    @abc.abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray):
        """Given training data, update parameter scorer."""

    @abc.abstractmethod
    def score(self, obs: _O, parameter: Any) -> float:
        """Score the parameter given the low-level observation."""
