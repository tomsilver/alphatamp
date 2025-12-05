"""An abstract action scorer that uses a MLP for scoring."""

from typing import Any, TypeVar

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted

from alphatamp.approaches.scorers.base_scorer import BaseScorer
from alphatamp.structs import Skeleton


class AbstractActionScorer(BaseScorer):
    """A abstract action scorer that uses a MLP for scoring."""

    def __init__(self, configs: dict, saved_classifier=None):
        self._regressor = (
            MLPRegressor(hidden_layer_sizes=configs["hidden_layer_sizes"])
            if not saved_classifier
            else saved_classifier
        )

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Given training data, update scorer."""
        self._regressor.fit(features, labels)

    def _create_abstract_plan_embedding(self, abstract_plan: Skeleton) -> np.ndarray:
        """Create a embedding for an abstract plan."""
        return np.array(
            [hash(state) for state in abstract_plan[0]]
            + [hash(action) for action in abstract_plan[1]]
        )

    def score(self, previous_abstract_plan: Skeleton) -> float:
        """Score the action given the previous abstract plan."""

        try:
            check_is_fitted(self._regressor)
            abstract_plan_embedding = self._create_abstract_plan_embedding(
                previous_abstract_plan
            )
            return self._regressor.predict(abstract_plan_embedding)[0]
        except NotFittedError:
            return 1.0
