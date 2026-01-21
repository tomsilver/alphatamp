"""Defines an abstract explorer that can utilize a feasiblity classifier to generate
entire abstract skeletons."""

import abc
from typing import Generic, TypeVar, Optional

from bilevel_planning.structs import (
    RelationalAbstractState, RelationalAbstractGoal
)
from relational_structs.pddl import GroundOperator

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class BaseAbstractExplorer(Generic[_O, _X, _U], abc.ABC):
    """Defines an abstract explorer that can utilize a feasiblity classifier to generate
    entire abstract skeletons."""

    @abc.abstractmethod
    def generate_abstract_plan(
        self, obs: _O, goal: Optional[RelationalAbstractGoal] = None
    ) -> tuple[list[RelationalAbstractState], list[GroundOperator]]:
        """Generates an abstract plan given initial observation."""
