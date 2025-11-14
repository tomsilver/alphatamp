"""A parameter scorer that uses a MLP classifier for scoring."""

from typing import Any, TypeVar

from sklearn.neural_network import MLPClassifier

from alphatamp.approaches.parameter_scorers.base_parameter_scorer import ParameterScorer

_X = TypeVar("_X")  # state
Datastore = list[tuple[Any]]
Labels = list[Any]


class ClassifierScorer(ParameterScorer):
    """A parameter scorer that uses a MLP classifier for scoring."""

    def __init__(self, configs):
        self._classifier = MLPClassifier(
            hidden_layer_sizes=configs["hidden_layer_sizes"]
        )

    def train(self, features: Datastore, labels: Labels):
        """Given training data, update parameter scorer."""
        self._classifier.fit(features, labels)

    def score(self, x: _X, params: Any) -> float:
        """Score the parameter given the low-level state."""
        prediction = self._classifier.predict((x, params))[0]

        return prediction
