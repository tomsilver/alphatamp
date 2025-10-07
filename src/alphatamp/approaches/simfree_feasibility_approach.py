"""A simulator-free approach that learns and uses a feasibility classifier."""

import time
from typing import Any, TypeVar

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    ParameterizedController,
    RelationalAbstractState,
)
from bilevel_planning.utils import (
    RelationalControllerGenerator,
)
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.pddl import GroundOperator

from alphatamp.approaches.feasibility_classifiers.feasibility_classifier_learner import (
    FeasibilityClassifierLearner,
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
        self._abstract_plan_generator: (
            RelationalHeuristicSearchAbstractPlanGenerator
        ) = RelationalHeuristicSearchAbstractPlanGenerator(
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
        self._last_observation: _O | None = None
        self._timestep = 0

    def reset(
        self,
        obs: _O,
        info: dict[str, Any],
    ) -> None:
        start_time = time.perf_counter()

        # During training, use the explorer to choose actions.
        if self._train_or_eval == "train":
            # Will need to implement in future PRs
            pass

        # During evaluation, use the feasibility classifier to plan.
        assert self._train_or_eval == "eval"

        # Get the current feasibility classifier.
        abstract_plan_classifier = self._feasibility_classifier_learner.get_classifier()

        # Plan with the feasibility classifier.

        # Get the initial abstract state.
        x0 = self._env_models.observation_to_state(obs)
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

            # Try to classify whether or not this abstract plan is valid.
            assert isinstance(x0, ObjectCentricState)
            if abstract_plan_classifier.validate_plan(x0, s_plan, a_plan):

                # If abstract plan is valid, return the plan.
                self._current_abstract_plan_step = 0
                self._current_abstract_plan = (s_plan, a_plan)
                self._current_controller = None
                self._last_observation = obs
                return

        raise RuntimeError("No abstract plan found.")

    def _get_action(self) -> _U:
        assert self._current_abstract_plan is not None

        # Advance until we are at an abstract action that has not yet completed.
        advanced = False
        while True:
            # If we ran out of actions, raise an error.
            if self._current_abstract_plan_step >= len(self._current_abstract_plan[1]):
                raise RuntimeError("Abstract planning ran out of actions.")

            # Get the next abstract state.
            ns = self._current_abstract_plan[0][self._current_abstract_plan_step + 1]

            # If we have reached the next abstract state, advance the current plan step.
            assert self._last_observation is not None
            x = self._env_models.observation_to_state(self._last_observation)
            s = self._env_models.state_abstractor(x)

            if s == ns:
                self._current_abstract_plan_step += 1
                advanced = True
            # We have found a step in the plan where the next state is not yet reached.
            else:
                # if it is the first step, we also need to reset a new controller
                if self._timestep == 0:
                    advanced = True
                break

        # Get the last observed state.
        x = self._env_models.observation_to_state(self._last_observation)
        # If we advanced, we need to reset a new parameterized controller.
        if advanced:
            # Get the current abstract action and controller.
            a = self._current_abstract_plan[1][self._current_abstract_plan_step]

            self._current_controller = self._controller_generator(a)

            # Resample parameters.
            params = self._current_controller.sample_parameters(x, self._rng)
            self._current_controller.reset(x, params)
        # We are using the same controller as before.
        else:
            assert self._current_controller is not None
            self._current_controller.observe(x)

        # Take one more low-level action.
        self._last_action = self._current_controller.step()
        assert self._last_action is not None

        return self._last_action

    def step(self) -> _U:
        """Get the next action to take."""
        self._last_action = self._get_action()
        self._timestep += 1
        return self._last_action
