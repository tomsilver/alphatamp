"""A base approach that does not have access to a simulator."""

import abc
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from gymnasium.spaces import Space
from prpl_utils.gym_agent import Agent
from relational_structs import (
    LiftedOperator,
    Predicate,
    Type,
)

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


@dataclass(frozen=True)
class SimulatorFreeSesameModels(Generic[_O, _X, _U]):
    """Container for common models used in SeSamE, but with NO simulator."""

    observation_space: Space[_O]
    state_space: Space[_X]
    action_space: Space[_U]
    types: set[Type]
    predicates: set[Predicate]
    observation_to_state: Callable[[_O], _X]
    state_abstractor: Callable[[_X], RelationalAbstractState]
    goal_deriver: Callable[[_X], RelationalAbstractGoal]
    skills: set[LiftedSkill]

    @property
    def operators(self) -> set[LiftedOperator]:
        """Access the lifted operators from the lifted skills."""
        return {s.operator for s in self.skills}


class SimulatorFreeBaseApproach(abc.ABC, Generic[_O, _X, _U], Agent[_O, _U]):
    """A base approach that does not have access to a simulator."""

    def __init__(
        self,
        env_models: SimulatorFreeSesameModels[_O, _X, _U],
        seed: int,
    ) -> None:
        self._env_models = env_models
        super().__init__(seed)

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Produce an action to execute now."""
