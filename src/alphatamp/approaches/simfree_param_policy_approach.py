"""A simulator-free approach that learns parameter policies in its free time."""

import math
from typing import Any, TypeVar

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.structs import (
    ParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from bilevel_planning.utils import (
    RelationalControllerGenerator,
)

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.base_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifierLearner,
)
from alphatamp.approaches.simulator_free_base_approach import (
    SimulatorFreeBaseApproach,
    SimulatorFreeSesameModels,
)
from alphatamp.approaches.parameter_policies.base_parameter_policy import ParameterPolicy
from prbench_models.geom2d.utils import Geom2dRobotController

from alphatamp.approaches.utils.approach_step_error import ApproachStepError
from alphatamp.structs import Skeleton

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class SimFreeParamPolicyApproach(SimulatorFreeBaseApproach[_O, _X, _U]):
    """A simulator-free approach that learns parameter policies in its free time."""

    def __init__(
        self,
        env_models: SimulatorFreeSesameModels[_O, _X, _U],
        feasibility_classifier_learner: BaseFeasibilityClassifierLearner,
        train_explorer: BaseAbstractExplorer[_O, _X, _U],
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

        # Explorers.
        self._train_explorer = train_explorer
        self._exploit_explorer: ExploitExplorer = ExploitExplorer(
            self._env_models, self._feasibility_classifier_learner, seed
        )

        self._abstract_action_to_energy_function = {}

    def reset(
        self,
        obs: _O,
        info: dict[str, Any],
    ) -> None:

        explorer = None
        # During training, use the explorer to choose actions.
        if self._train_or_eval == "train":
            explorer = self._train_explorer
        else:
            # During evaluation, use the feasibility classifier to plan.
            assert self._train_or_eval == "eval"
            explorer = self._exploit_explorer

        # Use the explorer to create the current abstract plan.
        self._current_abstract_plan_step = 0
        self._current_controller = None
        self._last_observation = obs
        self._current_abstract_plan = explorer.generate_abstract_plan(obs)
        self._timestep = 0

    def _update_parameter_policies(self):
        pass

    def _plan_to_practice(self):
        pass

    
    def _select_skill(self):
        skills = self._env_models.skills

        pass
    
    def _estimate_competence(self, skills):
        for skill in skills:
            pass
        pass

    def _extrapolate_competence(self):
        pass

    def _situate_competence(self):
        pass

    def _energy_function(self, x, params) -> float:
        return 0.0
    
    def _resample_controller(self, x):
        """Resample parameters and reset the controller with the specified observation."""

        assert self._current_abstract_plan is not None

        # Get the current abstract action and controller.
        a = self._current_abstract_plan[1][self._current_abstract_plan_step]

        # Recreate controller
        self._current_controller = self._controller_generator(a)
        energy_function = self._abstract_action_to_energy_function[a]

        # Sample new params
        parameter_policy = ParameterPolicy(self._current_controller, energy_function)
        optimal_params = parameter_policy.sample_parameters(x, self._rng)

        # Reset + observe
        self._current_controller.reset(x, optimal_params)
        self._current_controller.observe(x)

    def _get_action(self) -> _U:
        assert self._current_abstract_plan is not None

        # Advance until we are at an abstract action that has not yet completed.
        advanced = False
        while True:
            # If we ran out of actions, raise an error.
            if self._current_abstract_plan_step >= len(self._current_abstract_plan[1]):
                out_of_actions_error = RuntimeError(
                    "Abstract planning ran out of actions."
                )
                raise ApproachStepError(
                    "Abstract planning ran out of actions.", out_of_actions_error
                )

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
            energy_function = self._abstract_action_to_energy_function[a]

            # Get parameter policy
            parameter_policy = ParameterPolicy(self._current_controller, energy_function)
            optimal_params = parameter_policy.sample_parameters(x, self._rng)

            # Reset the controller on the optimal parameters
            self._current_controller.reset(x, optimal_params)
        # We are using the same controller as before.
        else:
            assert self._current_controller is not None
            self._current_controller.observe(x)

        # Take one more low-level action.
        while True:
            try:
                self._last_action = self._current_controller.step()
                assert self._last_action is not None

                return self._last_action
            # if low level action failed, resample parameters!
            except TrajectorySamplingFailure as e:
                self._resample_controller(x)
                continue

            except IndexError as e:
                self._resample_controller(x)
                raise ApproachStepError("Index Error!", e)


    def step(self) -> _U:
        """Get the next action to take."""
        self._last_action = self._get_action()
        self._timestep += 1
        return self._last_action

    def get_abstract_plan(self) -> Skeleton | None:
        """Return the current abstract plan."""
        return self._current_abstract_plan
