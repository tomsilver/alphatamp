"""A base approach that does not have access to a simulator."""

import abc
from typing import Generic, TypeVar

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    Plan,
    PlanningProblem,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from prpl_utils.gym_agent import Agent

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action

import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar

import numpy as np
from gymnasium.spaces import Space
from prpl_utils.utils import consistent_hash
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)


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
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Produce an action to execute now."""
