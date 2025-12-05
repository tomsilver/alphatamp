"""A simulator-free approach that learns parameter policies in its free time."""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
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
    cached_all_ground_operators,
)

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.base_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifierLearner,
)
from alphatamp.approaches.parameter_policies.base_parameter_policy import (
    ParameterPolicy,
)
from alphatamp.approaches.scorers.base_scorer import BaseScorer
from alphatamp.approaches.simulator_free_base_approach import (
    SimulatorFreeBaseApproach,
    SimulatorFreeSesameModels,
)
from alphatamp.approaches.utils.approach_step_error import ApproachStepError
from alphatamp.structs import FrozenSkeleton, GroundOperator, Skeleton

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
        parameter_scorer_class: type[BaseScorer],
        parameter_scorer_configs: dict,
        seed: int,
        heuristic_name: str = "hff",
        eval_planning_timeout: float = 100,
        max_abstract_plans: int = 10,
        max_resamples: int = 5,
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

        # Parameter policy.
        self._max_resamples = max_resamples
        self._abstract_action_to_scoring_function: dict[GroundOperator, BaseScorer] = {}
        self._parameter_scorer_class = parameter_scorer_class
        self._parameter_scorer_configs = parameter_scorer_configs
        self._parameter_dataset: defaultdict[str, list] = defaultdict(list)
        self._most_recent_parameter: Any | None = None
        self._most_recent_abstract_action_descriptor: str | None = None

        # Abstract Plan Dataset
        self._abstract_plan_dataset: list = []

        # Abstract Skill Dataset
        self._abstract_skill_dataset: defaultdict[
            str, defaultdict[FrozenSkeleton, int]
        ] = defaultdict(lambda: defaultdict(int))

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
        self._most_recent_parameter = None
        self._most_recent_abstract_action_descriptor = None

        x0 = self._env_models.observation_to_state(obs)
        s0 = self._env_models.state_abstractor(x0)

        # Get set of lifted operators and ground them
        operators = self._env_models.operators
        grounded_operators = cached_all_ground_operators(operators, s0.objects)

        for grounded_operator in grounded_operators:
            # Create new parameter scorer instances per grounded operators.
            self._abstract_action_to_scoring_function[grounded_operator] = (
                self._parameter_scorer_class(**self._parameter_scorer_configs)
            )

    def _learn_from_transition(
        self,
        obs: _O,
        act: _U,
        next_obs: _O,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if done:
            # Store last successful parameter
            self._add_most_recent_parameter_to_dataset("success")
            self._add_abstract_plan_to_dataset("success")

    def update(self, obs: _O, reward: float, done: bool, info: dict[str, Any]) -> None:
        """Record the reward and next observation following an action."""
        assert self._last_observation is not None
        assert self._last_action is not None
        if self._train_or_eval == "train":
            self._learn_from_transition(
                self._last_observation, self._last_action, obs, reward, done, info
            )
        self._last_observation = obs
        self._last_info = info

    def _generate_parameter_scorer_training_data(
        self, features_and_labels: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reformat training data into numpy arrays."""

        features_list = []
        labels_list = []

        # Generate a row in the training dataset.
        for datapoint in features_and_labels:
            state, parameter, label = datapoint
            state_arr = np.array(state)
            parameter_arr = np.array(parameter)

            # The features are the state observation and the parameter.
            feature_arr = np.append(state_arr, parameter_arr)
            label_arr = np.array(label)

            features_list.append(feature_arr)
            labels_list.append(label_arr)

        features = np.vstack(features_list)
        labels = np.vstack(labels_list)

        return (features, labels)

    def train_parameter_policy(self, parameter_dataset: defaultdict[str, list]):
        """Train each abstract action's parameter policy given dataset."""

        for (
            abstract_action,
            scoring_function,
        ) in self._abstract_action_to_scoring_function.items():
            # Segment data for each ground operator.

            abstract_action_descriptor = abstract_action.short_str
            if abstract_action_descriptor in parameter_dataset:
                features_and_labels = parameter_dataset[abstract_action_descriptor]

                # Generate training data.
                features, labels = self._generate_parameter_scorer_training_data(
                    features_and_labels
                )

                # Train the scoring function for each grounded skill.
                scoring_function.train(features, labels)

    def _add_most_recent_parameter_to_dataset(self, training_label: str):
        """Label the parameter as successful (1) or failure (0)."""
        assert (
            self._most_recent_parameter and self._most_recent_abstract_action_descriptor
        )
        assert self._last_observation is not None
        label = 1 if training_label == "success" else 0

        self._parameter_dataset[self._most_recent_abstract_action_descriptor].append(
            (self._last_observation, self._most_recent_parameter, label)
        )

    def _add_most_recent_abstract_action_to_dataset(self, training_label: str):
        """Label the abstract action as successful (1) or failure (0)."""
        assert self._most_recent_abstract_action_descriptor
        assert self._current_abstract_plan

        label = 0 if training_label == "success" else 1

        prev_abstract_states = tuple(
            self._current_abstract_plan[0][: self._current_abstract_plan_step]
        )
        prev_abstract_actions = tuple(
            self._current_abstract_plan[1][: self._current_abstract_plan_step]
        )

        # Store the number of times the abstract action
        # given the previous abstract plan needed to be resampled.
        self._abstract_skill_dataset[self._most_recent_abstract_action_descriptor][
            (prev_abstract_states, prev_abstract_actions)
        ] += label

    def _add_abstract_plan_to_dataset(self, training_label: str):
        assert self._current_abstract_plan

        label = 1 if training_label == "success" else 0

        # Add the completed abstract plan up to the point where this function is called
        abstract_states = tuple(
            self._current_abstract_plan[0][: self._current_abstract_plan_step + 1]
        )
        abstract_actions = tuple(
            self._current_abstract_plan[1][: self._current_abstract_plan_step + 1]
        )
        self._abstract_plan_dataset.append(((abstract_states, abstract_actions), label))

    def _resample_controller(self, x: _X, obs: _O) -> None:
        """Resample parameters and reset the controller with the specified
        observation."""

        assert self._current_abstract_plan is not None

        # Get the current abstract action and controller.
        a = self._current_abstract_plan[1][self._current_abstract_plan_step]

        assert a is not None

        # Recreate controller and query scoring function
        self._current_controller = self._controller_generator(a)
        scoring_function = self._abstract_action_to_scoring_function[a]

        # Sample new params from the Parameter Policy
        parameter_policy = ParameterPolicy(self._current_controller, scoring_function)
        optimal_params = parameter_policy.sample_parameters(x, obs, self._rng)
        self._most_recent_parameter = optimal_params
        self._most_recent_abstract_action_descriptor = a.short_str

        # Reset controller
        self._current_controller.reset(x, optimal_params)

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

            assert self._last_observation is not None
            x = self._env_models.observation_to_state(self._last_observation)
            s = self._env_models.state_abstractor(x)

            # If we have reached the next abstract state, advance the current plan step.
            if s == ns:
                self._add_most_recent_abstract_action_to_dataset("success")
                self._add_abstract_plan_to_dataset("success")
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
            self._resample_controller(x, self._last_observation)

        # We are using the same controller as before.
        else:
            assert self._current_controller
            self._current_controller.observe(x)

        assert self._current_controller

        for attempt_num in range(self._max_resamples):
            # Try to take a low-level action from the controller.
            try:
                # Take one more low-level action.
                self._last_action = self._current_controller.step()
                assert self._last_action is not None

                # If low-level action is successful, store it.
                if self._train_or_eval == "train":
                    self._add_most_recent_parameter_to_dataset("success")

                return self._last_action
            # If low level action failed, store the parameter that failed!
            except (TrajectorySamplingFailure, IndexError) as e:
                # If training, store the previous parameter.
                if self._train_or_eval == "train":
                    self._add_most_recent_abstract_action_to_dataset("failure")
                    self._add_most_recent_parameter_to_dataset("failure")
                    self._add_abstract_plan_to_dataset("failure")

                if attempt_num == self._max_resamples - 1:
                    # Raise ApproachStepError
                    raise ApproachStepError("Trajectory Error!", e)

        raise RuntimeError("Should not reach this point")

    def step(self) -> _U:
        """Get the next action to take."""
        self._last_action = self._get_action()
        self._timestep += 1
        return self._last_action

    def get_abstract_plan(self) -> Skeleton | None:
        """Return the current abstract plan."""
        return self._current_abstract_plan

    def get_parameter_dataset(self) -> defaultdict[str, list]:
        """Return the collected parameter dataset."""
        return self._parameter_dataset

    def get_abstract_plan_dataset(self) -> list:
        """Return the collected abstract plan dataset."""
        return self._abstract_plan_dataset

    def get_abstract_skill_dataset(
        self,
    ) -> defaultdict[str, defaultdict[FrozenSkeleton, int]]:
        """Return the collected abstract skill dataset."""
        return self._abstract_skill_dataset

    def _create_abstract_plan_embedding(
        self, abstract_plan: Skeleton | FrozenSkeleton
    ) -> np.ndarray:
        """Create a embedding for an abstract plan."""
        return np.array(
            [hash(state) for state in abstract_plan[0]]
            + [hash(action) for action in abstract_plan[1]]
        )

    def save_datasets(self, directory: str | Path) -> None:
        """Save the collected dataset to disk as a pickle."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        datasets = {
            "parameter_dataset.pkl": dict(self._parameter_dataset),
            "abstract_plan_dataset.pkl": list(
                (self._create_abstract_plan_embedding(abstract_plan), training_label)
                for abstract_plan, training_label in self._abstract_plan_dataset
            ),
            "abstract_skill_dataset.pkl": {
                k: list(
                    (
                        self._create_abstract_plan_embedding(abstract_plan),
                        resample_count,
                    )
                    for abstract_plan, resample_count in v.items()
                )
                for k, v in self._abstract_skill_dataset.items()
            },
        }

        for filename, data in datasets.items():
            with (directory / filename).open("wb") as f:
                pickle.dump(data, f)

    @staticmethod
    def load_parameter_dataset(path: str | Path) -> defaultdict[str, list]:
        """Load a parameter dataset pickle from disk and return as defaultdict.

        Raises FileNotFoundError if the path does not exist.
        """
        p = Path(path)
        with p.open("rb") as f:
            raw = pickle.load(f)

        dd: defaultdict[str, list] = defaultdict(list)
        if isinstance(raw, dict):
            dd.update(raw)
        return dd

    @staticmethod
    def load_abstract_plan_dataset(path: str | Path) -> list:
        """Load a abstract plan dataset pickle from disk and return as list.

        Raises FileNotFoundError if the path does not exist.
        """
        p = Path(path)
        with p.open("rb") as f:
            raw = pickle.load(f)

        l: list = []
        if isinstance(raw, list):
            l = raw
        return l
