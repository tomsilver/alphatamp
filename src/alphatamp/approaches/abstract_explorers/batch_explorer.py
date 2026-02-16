"""An batched abstract plan explorer that returns a specified number of valid plans."""

import time
from typing import TypeVar

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractGoal, RelationalAbstractState
from relational_structs.pddl import GroundOperator

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.simulator_free_base_approach import (
    SimulatorFreeSesameModels,
)
from alphatamp.approaches.utils.abstract_plan_generation_error import (
    AbstractPlanGenerationError,
)

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class BatchExplorer(BaseAbstractExplorer[_O, _X, _U]):
    """An batched abstract plan explorer that returns a specified number of valid
    plans."""

    def __init__(
        self,
        env_models: SimulatorFreeSesameModels[_O, _X, _U],
        seed: int,
        heuristic_name: str = "hff",
        planning_timeout: float = 100,
        max_abstract_plans: int = 10,
    ):
        super().__init__()
        self._env_models = env_models

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
        self._max_abstract_plans = max_abstract_plans

    def generate_abstract_plan(
        self, obs: _O, goal: RelationalAbstractGoal | None = None
    ) -> tuple[list[RelationalAbstractState], list[GroundOperator]]:
        """Unnecessary function."""
        return ([], [])

    def generate_batched_abstract_plan(
        self,
        obs: _O,
        goal: RelationalAbstractGoal | None = None,
    ) -> list[tuple[list[RelationalAbstractState], list[GroundOperator]]]:
        """Returns a list of unique abstract plans."""
        start_time = time.perf_counter()

        # Get the initial abstract state.
        x0 = self._env_models.observation_to_state(obs)
        s0 = self._env_models.state_abstractor(x0)

        if goal is None:
            goal = self._env_models.goal_deriver(x0)

        # Initialize the bilevel planning graph.
        bpg: BilevelPlanningGraph[_X, _U, RelationalAbstractState, GroundOperator] = (
            BilevelPlanningGraph()
        )
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # Generate abstract plans and DO NOT attempt to refine them.
        gen = self._abstract_plan_generator(
            x0,
            s0,
            goal,
            self._planning_timeout,
            bpg,
        )
        num_abstract_plans = 0

        abstract_plans = []
        while (
            num_abstract_plans < self._max_abstract_plans
            and time.perf_counter() - start_time < self._planning_timeout
        ):
            # Get the next abstract plan.
            try:
                s_plan, a_plan = next(gen)
                abstract_plans.append((s_plan, a_plan))
                num_abstract_plans += 1
            except StopIteration:
                break
            # Quit early if timeout.
            remaining_time = self._planning_timeout - (time.perf_counter() - start_time)
            if remaining_time < 0:
                break

        if len(abstract_plans) == 0:
            raise AbstractPlanGenerationError("No abstract plan found.")
        return abstract_plans
