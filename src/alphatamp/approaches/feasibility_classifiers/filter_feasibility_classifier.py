"""Hardcoded classifier that filters out specific types of abstract plans."""

from collections import defaultdict

from bilevel_planning.structs import (
    RelationalAbstractState,
)
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.pddl import GroundOperator

from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifier,
)


class FilterFeasibilityClassifier(BaseFeasibilityClassifier):
    """Hardcoded classifier that filters out specific types of abstract plans."""

    def __init__(self) -> None:
        self._invalid_abstract_states: defaultdict[int, set] = defaultdict(set)
        self._invalid_abstract_actions: defaultdict[int, set] = defaultdict(set)

    def update_classifier(
        self,
        filter_abstract_states: list[tuple[int, int]] | None = None,
        filter_abstract_actions_str: list[tuple[str, int]] | None = None,
    ):
        """Add abstract states and actions to filter list."""
        if filter_abstract_states:
            for abstract_state_hash, plan_position in filter_abstract_states:
                self._invalid_abstract_states[plan_position].add(abstract_state_hash)

        if filter_abstract_actions_str:
            for abstract_action_str, plan_position in filter_abstract_actions_str:
                self._invalid_abstract_actions[plan_position].add(abstract_action_str)

    def validate_plan(
        self,
        x0: ObjectCentricState,
        abstract_states: list[RelationalAbstractState],
        abstract_actions: list[GroundOperator],
    ) -> bool:
        """Validate the feasibility of the given abstract plan."""

        # Iterate over each plan step and check if
        # the abstract state is in the filter.
        for plan_position, abstract_state in enumerate(abstract_states):
            if hash(abstract_state) in self._invalid_abstract_states[plan_position]:
                return False

        # Do the same for abstract actions.
        for plan_position, abstract_action in enumerate(abstract_actions):
            name = abstract_action.name
            parameters = abstract_action.parameters
            parameter_names = set(object.name for object in parameters)

            if name in self._invalid_abstract_actions[
                plan_position
            ] or not parameter_names.isdisjoint(
                self._invalid_abstract_actions[plan_position]
            ):
                return False

        return True
