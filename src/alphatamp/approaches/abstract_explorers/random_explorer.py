"""An random abstract plan explorer that returns a fixed length plan with random
actions."""

from typing import TypeVar

import numpy as np
from bilevel_planning.structs import (
    RelationalAbstractState,
)
from bilevel_planning.utils import (
    cached_all_ground_operators,
)
from relational_structs.pddl import GroundOperator

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)

from alphatamp.approaches.simulator_free_base_approach import (
    SimulatorFreeSesameModels,
)

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class RandomExplorer(BaseAbstractExplorer[_O, _X, _U]):
    """An random abstract plan explorer that returns a fixed length plan with random
    actions."""

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

    def generate_abstract_plan(
        self, obs: _O
    ) -> tuple[list[RelationalAbstractState], list[GroundOperator]]:
        # Randomly create abstract plan.

        # Get the initial abstract state.
        x0 = self._env_models.observation_to_state(obs)
        s0 = self._env_models.state_abstractor(x0)

        # Get set of lifted operators and ground them
        operators = self._env_models.operators
        grounded_operators = cached_all_ground_operators(operators, s0.objects)

        # Create random abstract plan with random length
        s_plan = [s0]
        a_plan = []

        plan_length = self._rng.integers(1, self._max_plan_length)
        for _ in range(plan_length):

            next_random_abstract_action: GroundOperator = self._rng.choice(
                np.array(list(grounded_operators), dtype=object)
            )

            next_atoms = (
                s_plan[-1].atoms - next_random_abstract_action.delete_effects
            ) | next_random_abstract_action.add_effects

            next_random_abstract_state = RelationalAbstractState(next_atoms, s0.objects)

            s_plan.append(next_random_abstract_state)
            a_plan.append(next_random_abstract_action)

        return (s_plan, a_plan)
