"""Base class for a scoring function."""

import abc
from typing import Any, TypeVar

import numpy as np

_O = TypeVar("_O")  # observation


class BaseScorer(abc.ABC):
    """Base class for a scoring function."""

    def __init__(self, configs: dict):
        pass

    @abc.abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray):
        """Given training data, update scorer."""

    @abc.abstractmethod
    def score(self, *args, **kwargs) -> float:
        """Score the input."""
