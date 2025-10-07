"""Base Classifier Learner that trains the feasibility classifier based on past
experiences."""

import abc

from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (
    BaseFeasibilityClassifier,
)


class FeasibilityClassifierLearner:
    """Base Classifier Learner that trains the feasibility classifier based on past
    experiences."""

    @abc.abstractmethod
    def get_classifier(self) -> BaseFeasibilityClassifier:
        """Produce an action to execute now."""
