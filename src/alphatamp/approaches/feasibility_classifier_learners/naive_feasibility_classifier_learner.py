"""Naive learner that simply updates classifier based on most recent experience."""

from typing import Tuple

from alphatamp.approaches.feasibility_classifier_learners.base_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (
    BaseFeasibilityClassifier,
)
from alphatamp.approaches.feasibility_classifiers.naive_feasibility_classifier import (
    NaiveFeasibilityClassifier,
)
from alphatamp.structs import FrozenSkeleton


class NaiveFeasibilityClassifierLearner(BaseFeasibilityClassifierLearner):
    """Naive learner that simply updates classifier based on most recent experience."""

    def __init__(self, classifier: NaiveFeasibilityClassifier):
        self.classifier = classifier

    def train_classifier(self, experience: Tuple[FrozenSkeleton, str]):
        """Train the classifier given most recent experience."""
        self.classifier.update_classifier([experience])

    def get_classifier(self) -> BaseFeasibilityClassifier:
        return self.classifier
