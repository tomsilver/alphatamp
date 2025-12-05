"""A naive parameter scorer that only returns 1."""

from typing import Any, TypeVar

import numpy as np

from alphatamp.approaches.scorers.base_scorer import BaseScorer

_O = TypeVar("_O")  # observation


class NaiveParameterScorer(BaseScorer):
    """A naive parameter scorer that only returns 1."""

    def __init__(self, configs: dict):
        pass

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Given training data, update parameter scorer."""

    def score(self, obs: _O, parameter: Any, *args, **kwargs) -> float:
        """Score the parameter given the low-level observation."""
        return 1
