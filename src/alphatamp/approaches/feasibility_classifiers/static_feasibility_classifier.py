"""Simply return the given base feasibility classifer with no learning."""

from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (
    BaseFeasibilityClassifier,
)
from alphatamp.approaches.feasibility_classifiers.feasibility_classifier_learner import (
    FeasibilityClassifierLearner,
)


class StaticFeasibilityClassifier(FeasibilityClassifierLearner):
    """Simply return the given base feasibility classifer with no learning."""

    def __init__(self, classifier: BaseFeasibilityClassifier):
        self.classifier = classifier

    def get_classifier(self) -> BaseFeasibilityClassifier:
        return self.classifier
