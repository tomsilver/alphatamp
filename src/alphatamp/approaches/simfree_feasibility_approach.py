"""A simulator-free approach that learns and uses a feasibility classifier."""

import time
from typing import Any, TypeVar

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    GroundOperator,
    ParameterizedController,
    RelationalAbstractState,
)
from bilevel_planning.utils import (
    RelationalControllerGenerator,
)

from alphatamp.approaches.simulator_free_base_approach import (
    SimulatorFreeBaseApproach,
    SimulatorFreeSesameModels,
)
from alphatamp.structs import Skeleton

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class SimFreeFeasiblityApproach(SimulatorFreeBaseApproach[_O, _X, _U]):
    """A simulator-free approach that learns and uses a feasibility classifier."""

    def __init__(
        self,
        env_models: SimulatorFreeSesameModels[_O, _X, _U],
        feasibility_classifier_learner: FeasibilityClassifierLearner,
        seed: int,
        heuristic_name: str = "hff",
        eval_planning_timeout: float = 100,
        max_abstract_plans: int = 10,
    ) -> None:
        super().__init__(env_models, seed)
        self._feasibility_classifier_learner = feasibility_classifier_learner
        self._abstract_plan_generator = RelationalHeuristicSearchAbstractPlanGenerator(
            self._env_models.types,
            self._env_models.predicates,
            self._env_models.operators,
            heuristic_name,
            seed=seed,
        )
        self._controller_generator = RelationalControllerGenerator(
            self._env_models.skills
        )
        self._max_abstract_plans = max_abstract_plans
        self._eval_planning_timeout = eval_planning_timeout

        # Maintain a current abstract plan.
        self._current_abstract_plan: Skeleton | None = None
        self._current_abstract_plan_step: int = 0
        self._current_controller: ParameterizedController | None = None

    def reset(
        self,
        obs: _O,
        info: dict[str, Any],
    ) -> None:
        start_time = time.perf_counter()

        # During training, use the explorer to choose actions.
        if self._train_or_eval == "train":
            import ipdb

            ipdb.set_trace()

        # During evaluation, use the feasibility classifier to plan.
        assert self._train_or_eval == "eval"

        # Get the current feasibility classifier.
        abstract_plan_classifier = self._feasibility_classifier_learner.get_classifier()

        # Plan with the feasibility classifier.

        # Get the initial abstract state.
        x0 = self._last_observation
        s0 = self._env_models.state_abstractor(x0)
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
            self._eval_planning_timeout,
            bpg,
        )
        num_abstract_plans = 0

        while (
            num_abstract_plans < self._max_abstract_plans
            and time.perf_counter() - start_time < self._eval_planning_timeout
        ):
            # Get the next abstract plan.
            try:
                s_plan, a_plan = next(gen)
                num_abstract_plans += 1
            except StopIteration:
                break
            # Quit early if timeout.
            remaining_time = self._eval_planning_timeout - (
                time.perf_counter() - start_time
            )
            if remaining_time < 0:
                break

            # Try to classify whether or not this abstract plan is valid
            if abstract_plan_classifier.validate_plan(x0, s_plan, a_plan):
                self._current_abstract_plan_step = 0
                self._current_abstract_plan = (s_plan, a_plan)
                self._current_controller = None

        raise RuntimeError("No abstract plan found.")

    def _get_action(self) -> _U:
        assert self._current_abstract_plan is not None

        # Advance until we are at an abstract action that has not yet completed.
        advanced = False
        while True:
            # If we ran out of actions, raise an error.
            if self._current_abstract_plan_step >= len(self._current_abstract_plan[1]):
                raise RuntimeError("Abstract planning ran out of errors.")

            # Get the current abstract action.
            ns = self._current_abstract_plan[0][self._current_abstract_plan_step + 1]

            # If we have reached the next abstract state, advance the current plan step.
            s = self._env_models.state_abstractor(self._last_observation)
            if s == ns:
                self._current_abstract_plan_step += 1
                advanced = True
            # We have found a step in the plan where the next state is not yet reached.
            else:
                break

        # If we advanced, we need to reset a new parameterized controller.
        if advanced:
            # Get the current abstract action and controller.
            a = self._current_abstract_plan[1][self._current_abstract_plan_step]
            self._current_controller = self._controller_generator(a)
            # Resample parameters.
            params = self._current_controller.sample_parameters(
                self._last_observation, self._rng
            )
            self._current_controller.reset(self._last_observation, params)
        # We are using the same controller as before.
        else:
            self._current_controller.observe(self._last_observation)

        # Take one more low-level action.
        return self._current_controller.step()
