"""A naive parameter scorer that only returns 1."""

from typing import Any, TypeVar

import numpy as np

from alphatamp.approaches.parameter_scorers.base_parameter_scorer import ParameterScorer

_X = TypeVar("_X")  # state


class NaiveScorer(ParameterScorer):
    """A naive parameter scorer that only returns 1."""

    def __init__(self, configs: dict):
        pass

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Given training data, update parameter scorer."""

    def score(self, x: _X, parameter: Any) -> float:
        """Score the parameter given the low-level state."""
        return 1
