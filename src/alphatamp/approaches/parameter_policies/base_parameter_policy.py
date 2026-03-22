"""A base class for a parameter policy wrapper over a ParameterizedController."""

import math
from typing import Any, TypeVar

from bilevel_planning.structs import ParameterizedController
from numpy.random import Generator

from alphatamp.approaches.scorers.base_scorer import BaseScorer

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state


class ParameterPolicy:
    """A base class for a parameter policy wrapper over a ParameterizedController."""

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
        """Sample controller parameter given low-level state."""

        optimal_params = None
        optimal_score = -math.inf
        for _ in range(self._param_sample_count):
            # Get initial parameters from controller
            params = self._controller.sample_parameters(x, rng)

            # Now we score the params based on the energy function
            energy_score = self._scoring_function.score(obs, params)

            if energy_score > optimal_score:
                optimal_score = energy_score
                optimal_params = params

        return optimal_params
