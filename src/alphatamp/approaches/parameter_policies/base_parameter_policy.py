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

    Uses Boltzmann (softmax) sampling over candidate parameters weighted by scorer
    outputs, controlled by a temperature parameter.  High temperature → nearly uniform
    (preserves diversity); low temperature → approaches argmax (exploits scorer
    confidence).
    """

    def __init__(
        self,
        controller: ParameterizedController,
        scoring_function: BaseScorer,
        param_sample_count=10,
        temperature: float = 1.0,
    ) -> None:
        self._controller = controller
        self._scoring_function = scoring_function
        self._param_sample_count = param_sample_count
        self._temperature = temperature

    def sample_parameters(self, x: _X, obs: _O, rng: Generator) -> Any:
        """Sample controller parameter using Boltzmann sampling over scores."""

        candidates = []
        scores = []
        for _ in range(self._param_sample_count):
            params = self._controller.sample_parameters(x, rng)
            score = self._scoring_function.score(obs, params)
            candidates.append(params)
            scores.append(score)

        scores_arr = np.array(scores)

        # Boltzmann weights with numerical stability
        logits = scores_arr / self._temperature
        logits -= logits.max()
        weights = np.exp(logits)
        probs = weights / weights.sum()

        idx = rng.choice(len(candidates), p=probs)
        return candidates[idx]
