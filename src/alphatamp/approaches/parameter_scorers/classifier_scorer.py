"""A parameter scorer that uses a MLP classifier for scoring."""

from typing import Any, TypeVar

from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_is_fitted

from alphatamp.approaches.parameter_scorers.base_parameter_scorer import ParameterScorer

_X = TypeVar("_X")  # state
Datastore = list[tuple[Any]]
Labels = list[Any]


class ClassifierScorer(ParameterScorer):
    """A parameter scorer that uses a MLP classifier for scoring."""

    def __init__(self, configs, saved_classifier=None):
        self._classifier = (
            MLPClassifier(hidden_layer_sizes=configs["hidden_layer_sizes"])
            if not saved_classifier
            else saved_classifier
        )

    def train(self, features: Datastore, labels: Labels):
        """Given training data, update parameter scorer."""
        self._classifier.fit(features, labels)

    def score(self, x: _X, params: Any) -> float:
        """Score the parameter given the low-level state."""
        try:
            check_is_fitted(self._classifier)
            return self._classifier.predict((x, params))[0]
        except NotFittedError:
            return 1.0
