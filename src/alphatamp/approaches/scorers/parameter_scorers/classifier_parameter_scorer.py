"""A parameter scorer that uses a MLP classifier for scoring."""

from typing import Any, TypeVar

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

from alphatamp.approaches.scorers.base_scorer import BaseScorer

_O = TypeVar("_O")  # observation


class ClassifierParameterScorer(BaseScorer):
    """A parameter scorer that uses a MLP classifier for scoring."""

    def __init__(self, configs: dict, saved_classifier=None):
        self._classifier = (
            MLPClassifier(
                hidden_layer_sizes=configs["hidden_layer_sizes"],
                max_iter=configs.get("max_iter", 1000),
            )
            if not saved_classifier
            else saved_classifier
        )

    def train(self, features: np.ndarray, labels: np.ndarray):
        """Given training data, update parameter scorer."""
        sample_weights = compute_sample_weight("balanced", labels)
        self._classifier.fit(features, labels, sample_weight=sample_weights)

    def score(self, obs: _O, parameter: Any) -> float:
        """Score the parameter given the low-level observation."""
        try:
            check_is_fitted(self._classifier)
            state_arr = np.array(obs)
            parameter_arr = np.array(parameter)
            feature_arr = np.append(state_arr, parameter_arr)
            features = feature_arr.reshape(1, -1)
            return self._classifier.predict_proba(features)[:, 1][0]
        except NotFittedError:
            return 1.0
