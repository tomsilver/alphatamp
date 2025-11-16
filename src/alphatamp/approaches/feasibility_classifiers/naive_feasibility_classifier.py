"""Non-ML classifier that simply stores a dictionary of success and failures."""

from typing import Tuple

from bilevel_planning.structs import (
    RelationalAbstractState,
)
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.pddl import GroundOperator

from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifier,
)
from alphatamp.structs import FrozenSkeleton


class NaiveFeasibilityClassifier(BaseFeasibilityClassifier):
    """Non-ML classifier that simply stores a dictionary of success and failures."""

    def __init__(self, success_freq_threshold: float, success_str: str):
        self._skeleton_success_frequency: dict[FrozenSkeleton, Tuple[int, int]] = {}
        self._success_freq_threshold = success_freq_threshold
        self._success_str = success_str

    def update_classifier(self, data: list[Tuple[FrozenSkeleton, str]]):
        """Update the skeleton success frequency dictionary given data."""
        for skeleton, status in data:
            successes, failures = self._skeleton_success_frequency.get(skeleton, (0, 0))
            if status == self._success_str:
                successes += 1
            else:
                failures += 1
            self._skeleton_success_frequency[skeleton] = (successes, failures)

    def validate_plan(
        self,
        x0: ObjectCentricState,
        abstract_states: list[RelationalAbstractState],
        abstract_actions: list[GroundOperator],
    ) -> bool:
        """Validate the feasibility of the given abstract plan."""

        skeleton_plan: FrozenSkeleton = (
            tuple(abstract_states),
            tuple(abstract_actions),
        )

        # If classifier hasn't seen plan before, optimistically return true
        if skeleton_plan not in self._skeleton_success_frequency:
            return True

        # Otherwise, check if success frequency is above threshold
        successes, failures = self._skeleton_success_frequency[skeleton_plan]
        return successes / (successes + failures) >= self._success_freq_threshold
