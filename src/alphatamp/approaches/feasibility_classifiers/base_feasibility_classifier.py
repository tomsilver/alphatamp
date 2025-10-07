"""A base classifier that classifies whether or not a given abstract plan is
feasible."""

import abc

from bilevel_planning.structs import (
    RelationalAbstractState,
)
from relational_structs import GroundOperator
from relational_structs.object_centric_state import ObjectCentricState


class BaseFeasibilityClassifier:
    """A base classifier that classifies whether or not a given abstract plan is
    feasible."""

    @abc.abstractmethod
    def validate_plan(
        self,
        x0: ObjectCentricState,
        abstract_states: list[RelationalAbstractState],
        abstract_actions: list[GroundOperator],
    ) -> bool:
        """Validate the feasibility of the given abstract plan."""
