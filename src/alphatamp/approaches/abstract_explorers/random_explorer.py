"""An random abstract plan explorer that returns a fixed length plan with random
actions."""

import random
import time
from typing import TypeVar

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    RelationalAbstractState,
)
from bilevel_planning.utils import (
    cached_all_ground_operators,
)
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.pddl import GroundOperator

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.feasibility_classifiers.feasibility_classifier_learner import (
    FeasibilityClassifierLearner,
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
        feasibility_classifier_learner: FeasibilityClassifierLearner,
        seed: int,
        heuristic_name: str = "hff",
        planning_timeout: float = 100,
        max_plan_length: int = 5,
    ):
        super().__init__()
        self._env_models = env_models
        self._feasibility_classifier_learner = feasibility_classifier_learner

        self._abstract_plan_generator: (
            RelationalHeuristicSearchAbstractPlanGenerator
        ) = RelationalHeuristicSearchAbstractPlanGenerator(
            self._env_models.types,
            self._env_models.predicates,
            self._env_models.operators,
            heuristic_name,
            seed=seed,
        )

        self._planning_timeout = planning_timeout
        self._max_plan_length = max_plan_length

    def generate_abstract_plan(
        self, obs: _O
    ) -> tuple[list[RelationalAbstractState], list[GroundOperator]]:
        # Randomly create abstract plan.

        # Get the initial abstract state.
        x0 = self._env_models.observation_to_state(obs)
        s0 = self._env_models.state_abstractor(x0)
        goal = self._env_models.goal_deriver(x0)

        # Get set of lifted operators and ground them
        operators = self._env_models.operators
        grounded_operators = cached_all_ground_operators(operators, s0.objects)

        # Create random abstract plan with max_plan_length
        s_plan = [s0]
        a_plan = []
        for _ in range(self._max_plan_length):
            next_random_abstract_action = random.choice(list(grounded_operators))

            # import ipdb
            # ipdb.set_trace()

            next_atoms = (
                s_plan[-1].atoms - next_random_abstract_action.delete_effects
            ) | next_random_abstract_action.add_effects

            next_random_abstract_state = RelationalAbstractState(next_atoms, s0.objects)

            s_plan.append(next_random_abstract_state)
            a_plan.append(next_random_abstract_action)

        return (s_plan, a_plan)
