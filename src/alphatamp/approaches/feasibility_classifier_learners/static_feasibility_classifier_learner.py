"""Simply return the given base feasibility classifer with no learning."""

from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (
    BaseFeasibilityClassifier,
)
from alphatamp.approaches.feasibility_classifier_learners.base_feasibility_classifier_learner import (
    BaseFeasibilityClassifierLearner,
)


class StaticFeasibilityClassifierLearner(BaseFeasibilityClassifierLearner):
    """Simply return the given base feasibility classifer with no learning."""

    def __init__(self, classifier: BaseFeasibilityClassifier):
        self.classifier = classifier

    def get_classifier(self) -> BaseFeasibilityClassifier:
        return self.classifier
