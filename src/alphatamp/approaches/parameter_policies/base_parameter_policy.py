"""A base class for a parameter policy wrapper over a ParameterizedController."""

from typing import Any, TypeVar

import numpy as np
from bilevel_planning.structs import ParameterizedController
from numpy.random import Generator

from alphatamp.approaches.scorers.base_scorer import BaseScorer

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state


class ParameterPolicy:
    """A base class for a parameter policy wrapper over a ParameterizedController.

    Generates candidate parameters and selects the one with the highest score
    (greedy exploitation). Exploration is handled externally via epsilon-greedy
    in the approach class.
    """

    def __init__(
        self,
        controller: ParameterizedController,
        scoring_function: BaseScorer,
        param_sample_count=10,
    ) -> None:
        self._controller = controller
        self._scoring_function = scoring_function
        self._param_sample_count = param_sample_count

    def sample_parameters(self, x: _X, obs: _O, rng: Generator) -> Any:
        """Select the highest-scoring candidate parameter (argmax)."""

        candidates = []
        scores = []
        for _ in range(self._param_sample_count):
            params = self._controller.sample_parameters(x, rng)
            score = self._scoring_function.score(obs, params)
            candidates.append(params)
            scores.append(score)

        best_idx = int(np.argmax(scores))
        return candidates[best_idx]
