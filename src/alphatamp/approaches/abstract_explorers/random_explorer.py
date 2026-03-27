"""An random abstract plan explorer that returns a random length plan with random
actions."""

from typing import TypeVar

import numpy as np
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalAbstractSuccessorGenerator,
)
from bilevel_planning.structs import RelationalAbstractGoal, RelationalAbstractState

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.simulator_free_base_approach import (
    SimulatorFreeSesameModels,
)
from relational_structs.pddl import GroundOperator

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class RandomExplorer(BaseAbstractExplorer[_O, _X, _U]):
    """An random abstract plan explorer that returns a random length plan with random
    actions whose preconditions are satisfied at each step."""

    def __init__(
        self,
        env_models: SimulatorFreeSesameModels[_O, _X, _U],
        seed: int,
        planning_timeout: float = 100,
        max_plan_length: int = 5,
    ):
        super().__init__()
        self._env_models = env_models
        self._seed = seed

        self._planning_timeout = planning_timeout
        self._max_plan_length = max_plan_length
        self._rng = np.random.default_rng(seed=seed)
        self._successor_fn = RelationalAbstractSuccessorGenerator(
            env_models.operators
        )

    def generate_abstract_plan(
        self, obs: _O, goal: RelationalAbstractGoal | None = None
    ) -> tuple[list[RelationalAbstractState], list[GroundOperator]]:
        # Get the initial abstract state.
        x0 = self._env_models.observation_to_state(obs)
        s0 = self._env_models.state_abstractor(x0)

        # Create random abstract plan with random length
        s_plan = [s0]
        a_plan = []

        plan_length = self._rng.integers(1, self._max_plan_length)
        for _ in range(plan_length):
            # Only consider operators whose preconditions hold in the current state.
            feasible = list(self._successor_fn(s_plan[-1]))
            if not feasible:
                break

            action, next_state = feasible[self._rng.integers(len(feasible))]
            s_plan.append(next_state)
            a_plan.append(action)

        return (s_plan, a_plan)
