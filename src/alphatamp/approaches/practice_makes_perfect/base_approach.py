"""An implementation of Kumar, Silver et al's paper Practice Makes Perfect approach."""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.structs import ParameterizedController, RelationalAbstractGoal
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from bilevel_planning.utils import (
    RelationalControllerGenerator,
    cached_all_ground_operators,
    get_all_ground_atoms_for_predicate,
)
from relational_structs.pddl import GroundAtom

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.abstract_plan_classifiers.q_network import (
    create_abstract_plan_sequence,
)
from alphatamp.approaches.feasibility_classifier_learners.base_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifierLearner,
)
from alphatamp.approaches.parameter_policies.base_parameter_policy import (
    ParameterPolicy,
)
from alphatamp.approaches.practice_makes_perfect.competence_models import (
    SkillCompetenceModel,
    create_competence_model,
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


class PracticeMakesPerfectApproach(SimulatorFreeBaseApproach[_O, _X, _U]):
    """A simulator-free approach that estimates, extrapoltates, and situates abstract
    action competencies in its free time (training mode)."""

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
        max_resamples: int = 100,
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

        # Ground Atoms and Operators
        self._all_ground_atoms: tuple[GroundAtom, ...] = ()
        self._all_ground_operators: tuple[GroundOperator, ...] = ()

        # Competence Models
        self._current_competence_model: SkillCompetenceModel | None = None
        self._abstract_action_to_competence_model: dict[
            GroundOperator, SkillCompetenceModel
        ] = {}

        # State distributions
        self._initial_state_distribution: list = []

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
        self._current_competence_model = None

        self._timestep = 0
        self._most_recent_parameter = None
        self._most_recent_abstract_action_descriptor = None

        x0 = self._env_models.observation_to_state(obs)
        s0 = self._env_models.state_abstractor(x0)

        # Get set of lifted operators and ground them
        operators = self._env_models.operators
        grounded_operators = cached_all_ground_operators(operators, s0.objects)

        self._all_ground_operators = tuple(sorted(grounded_operators))

        # Get all the ground atoms in environment
        predicates = self._env_models.predicates
        all_ground_atoms = set()

        for predicate in predicates:
            all_ground_atoms.update(
                get_all_ground_atoms_for_predicate(predicate, s0.objects)
            )

        self._all_ground_atoms = tuple(sorted(all_ground_atoms))

        for grounded_operator in grounded_operators:
            # Create new parameter scorer instances per grounded operators.
            self._abstract_action_to_scoring_function[grounded_operator] = (
                self._parameter_scorer_class(**self._parameter_scorer_configs)
            )

            self._abstract_action_to_competence_model[grounded_operator] = (
                create_competence_model("optimistic", grounded_operator.name)
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
            pass

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
        labels = np.vstack(labels_list).ravel()

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

        if self._train_or_eval == "eval":
            return

        label = 1 if training_label == "success" else 0

        self._parameter_dataset[self._most_recent_abstract_action_descriptor].append(
            (self._last_observation, self._most_recent_parameter, label)
        )

    def _update_competence_model(self, action_outcome: bool) -> None:
        """Update the current abstract_actions's competence model with the observed outcome."""
        assert self._current_competence_model is not None

        self._current_competence_model.observe(action_outcome)


    def _extrapolate_abstract_action(self) -> list[tuple[GroundOperator, float]]:
        """Use the abstract action's competence model to determine which abstract action to practice.

        We do this by calculating the expected probability of successfully completing
        the desired goal assuming, for each abstract action, that we've updated the abstract actions's
        competency. We then return each abstract_action alongside its extrapolated competency.
        """

        abstract_action_scores: list[tuple[GroundOperator, float]] = []

        # Precompute the competencies for each abstract action
        abstract_action_competences = {
            abs_a: comp_model.get_current_competence()
            for abs_a, comp_model in self._abstract_action_to_competence_model.items()
        }
        for (
            extrapolated_abstract_action,
            competency_model,
        ) in self._abstract_action_to_competence_model.items():
            expected_task_success = 0.0

            curr_competency = abstract_action_competences[extrapolated_abstract_action]

            # Update the extrapolated abstract action's competency
            # to use the predicted competency
            abstract_action_competences[extrapolated_abstract_action] = (
                competency_model.predict_competence(1)
            )

            # Calculate the expected probability of task success given
            # updated competency
            for initial_state in self._initial_state_distribution:
                abstract_plan = self._train_explorer.generate_abstract_plan(
                    initial_state
                )

                if abstract_plan is not None:
                    _, abstract_actions = abstract_plan
                    task_success = 1.0
                    for abstract_action in abstract_actions:
                        task_success *= abstract_action_competences[abstract_action]

                    expected_task_success += task_success

            expected_task_success /= len(self._initial_state_distribution)

            abstract_action_scores.append(
                (extrapolated_abstract_action, expected_task_success)
            )

            # Revert abstract action competency back to baseline
            abstract_action_competences[extrapolated_abstract_action] = curr_competency
        assert abstract_action_scores

        abstract_action_scores.sort(key=lambda x: x[1], reverse=True)
        return abstract_action_scores

    def _generate_new_task_goal(self) -> None:
        """Given sorted list of extrapolated abstract action task successes, generate the abstract
        plan that transitions the agent to an abstract state where it can perform the
        abstract action that improves the task success the most."""

        assert self._last_observation is not None
        abstract_action_scores = self._extrapolate_abstract_action()

        for abstract_action, _ in abstract_action_scores:
            # Determine the required preconditions for the desired abstract action to practice
            preconditions = abstract_action.preconditions
            goal = RelationalAbstractGoal(
                preconditions, self._env_models.state_abstractor
            )

            try:
                self._current_abstract_plan = (
                    self._train_explorer.generate_abstract_plan(
                        self._last_observation, goal
                    )
                )
                return
            except RuntimeError:
                # If no abstract plan is found, try next abstract action.
                continue

        # Reset pointer
        self._current_abstract_plan_step = 0
        # If no abstract plan is found for all abstract actions, throw error
        raise RuntimeError("Unable to plan for any skill")

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
        # If we advanced, we need to update the current competence model.
        if advanced:
            a = self._current_abstract_plan[1][self._current_abstract_plan_step]
            self._current_competence_model = self._abstract_action_to_competence_model[
                a
            ]
            # Successful execution of abstract action.
            self._update_competence_model(True)
            self._resample_controller(x, self._last_observation)

        # We are using the same controller as before.
        else:
            assert self._current_controller
            self._current_controller.observe(x)

        assert self._current_controller

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
        except (TrajectorySamplingFailure, IndexError):
            # If training, store the previous parameter.
            if self._train_or_eval == "train":
                self._update_competence_model(False)
                self._add_most_recent_parameter_to_dataset("failure")

                # Determine new task to execute
                self._generate_new_task_goal()

                # Return dummy action
                assert self._env_models.action_space.shape
                action_shape = self._env_models.action_space.shape
                stationary_action = np.zeros(action_shape)
                return cast(_U, stationary_action)

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

    def _make_data(self, abstract_plan: Skeleton | FrozenSkeleton, label: int):
        sequence, sequence_length = create_abstract_plan_sequence(
            self._all_ground_atoms, self._all_ground_operators, abstract_plan
        )
        return (sequence, sequence_length, label)

    def save_datasets(self, directory: str | Path) -> None:
        """Save the collected dataset to disk as a pickle."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        datasets = {
            "parameter_dataset.pkl": dict(self._parameter_dataset),
            "abstract_plan_dataset.pkl": list(
                self._make_data(abstract_plan, training_label)
                for abstract_plan, training_label in self._abstract_plan_dataset
            ),
        }

        for filename, data in datasets.items():
            with (directory / filename).open("wb") as f:
                pickle.dump(data, f)

    @staticmethod
    def load_abstract_action_level_dataset(path: str | Path) -> defaultdict[str, list]:
        """Load a dataset keyed by abstract action from disk and return as defaultdict.

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
