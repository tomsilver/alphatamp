"""A simulator-free approach that learns parameter policies in its free time."""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.structs import (
    ParameterizedController,
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from bilevel_planning.utils import (
    RelationalControllerGenerator,
    cached_all_ground_operators,
    get_all_ground_atoms_for_predicate,
)
from relational_structs.pddl import GroundAtom
from torch import FloatTensor, Tensor, nn

from alphatamp.approaches.abstract_explorers.base_abstract_explorer import (
    BaseAbstractExplorer,
)
from alphatamp.approaches.abstract_explorers.batch_explorer import BatchExplorer
from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.abstract_plan_classifiers.q_network import (
    PerActionQNetwork,
    create_abstract_plan_sequence,
)
from alphatamp.approaches.abstract_plan_classifiers.utils import (
    calculate_bald_objective,
    convert_q_value_to_probability,
    train_q_network,
)
from alphatamp.approaches.feasibility_classifier_learners.base_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    BaseFeasibilityClassifierLearner,
)
from alphatamp.approaches.parameter_policies.base_parameter_policy import (
    ParameterPolicy,
)
from alphatamp.approaches.scorers.abstract_action_scorers.regressor_abstract_action_scorer import (  # pylint:disable=line-too-long
    AbstractActionScorer,
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
        abstract_action_scorer_class: type[AbstractActionScorer],
        abstract_action_scorer_configs: dict,
        q_network_configs: dict,
        seed: int,
        heuristic_name: str = "hff",
        eval_planning_timeout: float = 100,
        max_abstract_plans: int = 10,
        max_resamples: int = 100,
        num_candidate_plans: int = 10,
        train_every: int = 1,
        param_sample_count: int = 10,
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
        self._reset_controller = True
        self._last_observation: _O | None = None

        # Explorers.
        self._train_explorer = train_explorer
        self._exploit_explorer: ExploitExplorer = ExploitExplorer(
            self._env_models, self._feasibility_classifier_learner, seed
        )
        self._batch_explorer: BatchExplorer = BatchExplorer(
            self._env_models, seed, max_abstract_plans=num_candidate_plans
        )

        # Global resample count
        self._num_resamples = 0

        # Parameter policy.
        self._max_resamples = max_resamples
        self._param_sample_count = param_sample_count
        self._abstract_action_to_scoring_function: dict[GroundOperator, BaseScorer] = {}
        self._parameter_scorer_class = parameter_scorer_class
        self._parameter_scorer_configs = parameter_scorer_configs
        self._parameter_dataset: defaultdict[str, list] = defaultdict(list)
        self._most_recent_parameter: Any | None = None
        self._most_recent_abstract_action_descriptor: str | None = None

        # Abstract Plan Dataset — keyed by (abstract_states, abstract_actions) to
        # prevent duplicate entries from repeated resample failures
        # on the same plan step.
        # A success label (1) always overrides a prior failure label (0).
        self._abstract_plan_dataset: dict[tuple, int] = {}

        # Abstract Action inits.
        # Each entry stores (num_failures, num_attempts)
        # for a given (states, actions) key.
        self._abstract_action_dataset: defaultdict[
            str, defaultdict[FrozenSkeleton, tuple[int, int]]
        ] = defaultdict(lambda: defaultdict(lambda: (0, 0)))
        self._abstract_action_to_action_scorer: dict[
            GroundOperator, AbstractActionScorer
        ] = {}
        self._abstract_action_scorer_class = abstract_action_scorer_class
        self._abstract_action_scorer_configs = abstract_action_scorer_configs
        self._all_ground_atoms: tuple[GroundAtom, ...] = ()
        self._all_ground_operators: tuple[GroundOperator, ...] = ()

        # Loss metrics
        self._loss_metrics: dict[str, list] = {}
        self._q_loss_metrics: list[float] = []

        # Per-Action Resample Q Network
        self._q_network_configs: dict = q_network_configs
        self._ensemble_nets: list[PerActionQNetwork] = []
        self._num_ensemble_nets: int = q_network_configs.get("num_ensemble_nets", 5)
        self._resample_exhaustion_count: int = 0

        self._train_every: int = train_every
        self._completed_task = False
        self._plan_from_exploit: bool = False

    def reset_episode(self, obs: _O) -> None:
        """Reset only episode-level state for a new environment episode.

        Unlike reset(), this preserves all learned state: trained Q networks, parameter
        scorers, action scorers, and all collected datasets. Use this when the
        environment is reset mid-training to avoid getting stuck in a terminal state.
        """
        explorer = (
            self._train_explorer
            if self._train_or_eval == "train"
            else self._exploit_explorer
        )
        self._current_abstract_plan_step = 0
        self._current_controller = None
        self._last_observation = obs
        self._completed_task = False  # must be set before generate_candidate_plans

        if self._resample_exhaustion_count > 0:
            # Q-networks have been trained at least once: exploit learned plan scores
            # to start each episode with the plan expected to need fewest resamples.
            candidate_plans = self.generate_candidate_plans()
            self._current_abstract_plan = self._score_candidate_plans_exploit(
                candidate_plans
            )
            self._plan_from_exploit = True
            logging.info(
                "[BALD] Generated Exploit plan at Reset Episode: %s",
                [a.short_str for a in self._current_abstract_plan[1]],
            )
        else:
            self._current_abstract_plan = explorer.generate_abstract_plan(obs)
            self._plan_from_exploit = False

            logging.info(
                "[BALD] Generated Initial plan at Reset Episode: %s",
                [a.short_str for a in self._current_abstract_plan[1]],
            )

        self._timestep = 0
        self._num_resamples = 0
        self._most_recent_parameter = None
        self._most_recent_abstract_action_descriptor = None
        self._reset_controller = True

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
        self._num_resamples = 0
        self._most_recent_parameter = None
        self._most_recent_abstract_action_descriptor = None
        self._completed_task = False
        self._reset_controller = True

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

            # Create new abstract action scorer instances per grounded operators.
            self._abstract_action_to_action_scorer[grounded_operator] = (
                self._abstract_action_scorer_class(
                    self._all_ground_atoms,
                    self._all_ground_operators,
                    **self._abstract_action_scorer_configs,
                )
            )

        self._reinitialize_ensemble_nets()

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
            # Store last successful parameter, action, plan
            self._add_most_recent_parameter_to_dataset("success")
            self._add_most_recent_abstract_action_to_dataset("success")
            self._add_abstract_plan_to_dataset("success")
            self._completed_task = True

            # Retrain scorers
            self._update_scorers()

            # Generate new candidate plan

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

    def train_parameter_policy(self, parameter_dataset: dict[str, list]):
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

    def _generate_abstract_action_scorer_training_data(
        self, features_and_labels: list
    ) -> tuple[list[FloatTensor], Tensor, Tensor]:
        """Reformat training data into tensors."""

        abstract_plan_list = []
        abstract_plan_lengths_list = []
        failure_rate_list = []

        # Generate a row in the training dataset.
        for datapoint in features_and_labels:
            abstract_plan, plan_len, failure_rate = datapoint

            # Get the features
            abstract_plan_list.append(torch.FloatTensor(abstract_plan))
            abstract_plan_lengths_list.append(plan_len)

            # Get the targets
            failure_rate_list.append(failure_rate)

        # Convert targets to tensor
        failure_rates = torch.FloatTensor(np.array(failure_rate_list)).unsqueeze(1)

        # Convert lengths to tensor
        abstract_plan_lengths = torch.tensor(abstract_plan_lengths_list)

        return (abstract_plan_list, abstract_plan_lengths, failure_rates)

    def train_abstract_action_scorer(self, abstract_action_dataset: dict[str, list]):
        """Train each abstract action scorer given dataset."""
        for (
            abstract_action,
            abstract_action_scorer,
        ) in self._abstract_action_to_action_scorer.items():
            # Segment data for each ground operator.

            abstract_action_descriptor = abstract_action.short_str
            if abstract_action_descriptor in abstract_action_dataset:
                features_and_labels = abstract_action_dataset[
                    abstract_action_descriptor
                ]

                # Generate training data.
                abstract_plan_list, abstract_plan_lengths, failure_rates = (
                    self._generate_abstract_action_scorer_training_data(
                        features_and_labels
                    )
                )

                loss_fn = nn.BCEWithLogitsLoss()
                # Train the scoring function for each grounded skill.
                losses = abstract_action_scorer.train(
                    abstract_plan_list, failure_rates, abstract_plan_lengths, loss_fn
                )

                self._loss_metrics[abstract_action_descriptor] = losses

    def get_abstract_action_score(self, abstract_action_str: str) -> float:
        """Evaluate the predicted failure rate for the abstract action given current
        task plan."""

        assert self._current_abstract_plan is not None
        for (
            abstract_action,
            abstract_action_scorer,
        ) in self._abstract_action_to_action_scorer.items():

            abstract_action_descriptor = abstract_action.short_str
            if abstract_action_descriptor == abstract_action_str:

                score = abstract_action_scorer.score(self._current_abstract_plan)
                return score
        return -1

    def _train_q_function(self, q_net: PerActionQNetwork) -> None:
        """Train the Per-Action Resample Q Function.

        The network learns to predict, for each action a_i in a plan, the failure
        rate for that action conditioned on the history (s_0, a_1, ..., a_{i-1}).

        Args:
            q_network: The PerActionQNetwork to train
            abstract_plans: List of previously executed abstract plans
            all_ground_atoms: All possible ground atoms in the environment
            all_ground_operators: All possible ground operators in the environment
            trained_abstract_action_scorers:
                Dictionary mapping abstract actions to trained abstract action scorers
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            verbose: Whether to log training progress
        Returns:
            List of average losses per epoch
        """

        assert q_net

        # Reformat abstract plan dataset
        abstract_plans: list[Skeleton] = []
        for abstract_plan, _ in self._abstract_plan_dataset.items():
            # Convert tuple of tuples into list of tuples
            abstract_states, abstract_actions = abstract_plan
            abstract_states_list = list(abstract_states)
            abstract_actions_list = list(abstract_actions)
            abstract_plans.append((abstract_states_list, abstract_actions_list))

        num_epochs = self._q_network_configs.get("num_epochs", 20)
        losses = train_q_network(
            q_net,
            abstract_plans,
            self._all_ground_atoms,
            self._all_ground_operators,
            self._abstract_action_to_action_scorer,
            batch_size=4,
            num_epochs=num_epochs,
            verbose=True,
        )
        self._q_loss_metrics.extend(losses)

    def _reinitialize_ensemble_nets(self) -> None:
        """Create fresh PerActionQNetwork instances, discarding any trained weights.

        Called both during reset() and at the start of each train_ensemble_nets() to
        avoid chasing non-stationary targets produced by the abstract action scorers,
        which are themselves retrained before every Q-network update.
        """
        _net_kwargs = {
            k: self._q_network_configs[k]
            for k in ("hidden_dim", "num_layers")
            if k in self._q_network_configs
        }
        self._ensemble_nets = [
            PerActionQNetwork(
                self._all_ground_atoms,
                self._all_ground_operators,
                **_net_kwargs,
            )
            for _ in range(self._num_ensemble_nets)
        ]

    def train_ensemble_nets(self) -> None:
        """Trains each of the Per-Action Resample Q Function in the approach's
        ensemble."""

        self._reinitialize_ensemble_nets()
        for q_net in self._ensemble_nets:
            self._train_q_function(q_net)

    def generate_candidate_plans(self) -> list[Skeleton]:
        """Use the training explorer to generate candidate plans for next execution
        using the last observation.

        This forces the explorer to generate a plan that tries to achieve the goal.
        However, the agent successfully completed the prior plan, the goal is to reset
        the environment first
        """

        assert self._last_observation is not None
        candidate_plans: list[Skeleton] = []

        goal = None
        if self._completed_task:
            # Need to reset the environment
            goal = RelationalAbstractGoal(set(), self._env_models.state_abstractor)
            self._completed_task = False

        candidate_plans = self._batch_explorer.generate_batched_abstract_plan(
            self._last_observation, goal
        )
        return candidate_plans

    def score_candidate_plans(self, candidate_plans: list[Skeleton]) -> Skeleton:
        """Given a list of candidate plans, score each plan based on the BALD objective
        and return the plan with the highest score."""

        best_bald_score = float("-inf")
        best_candidate_plan = None

        assert candidate_plans, "No Candidate Plans!"
        for candidate_plan in candidate_plans:

            candidate_probabilities = []
            # Use ensemble of Q networks to predict per-action failure rates
            for q_net in self._ensemble_nets:
                per_action_failure_rates = q_net.predict(candidate_plan)

                # Convert per-action failure rates to overall plan success probability
                probability = convert_q_value_to_probability(
                    per_action_failure_rates.tolist(), self._max_resamples
                )
                candidate_probabilities.append(probability)

            bald_score = calculate_bald_objective(candidate_probabilities)

            if bald_score > best_bald_score:
                best_bald_score = bald_score
                best_candidate_plan = candidate_plan

        assert best_candidate_plan
        return best_candidate_plan

    def _score_candidate_plans_exploit(
        self, candidate_plans: list[Skeleton]
    ) -> Skeleton:
        """Select the plan with the highest average predicted success probability across
        the Q-network ensemble (exploitation: highest expected success probability).

        Unlike score_candidate_plans(), which maximises epistemic uncertainty (BALD) for
        exploration, this method exploits what has been learned to start each episode
        with the plan that the ensemble collectively believes is easiest.
        """
        best_avg_prob = float("-inf")
        best_candidate_plan = None

        assert candidate_plans, "No Candidate Plans!"
        for candidate_plan in candidate_plans:
            probs = []
            ensemble_failure_rates = []
            for q_net in self._ensemble_nets:
                per_action_failure_rates = q_net.predict(candidate_plan)
                prob = convert_q_value_to_probability(
                    per_action_failure_rates.tolist(), self._max_resamples
                )
                probs.append(prob)
                ensemble_failure_rates.append(per_action_failure_rates.tolist())

            avg_prob = float(np.mean(probs))
            plan_str = [a.short_str for a in candidate_plan[1]]
            mean_failure_rates = np.mean(ensemble_failure_rates, axis=0).tolist()
            logging.info(
                "[Exploit] plan=%s  mean_failure_rates=%s  avg_prob=%.4f",
                plan_str,
                [f"{r:.4f}" for r in mean_failure_rates],
                avg_prob,
            )
            if avg_prob > best_avg_prob:
                best_avg_prob = avg_prob
                best_candidate_plan = candidate_plan

        assert best_candidate_plan
        return best_candidate_plan

    def _add_most_recent_parameter_to_dataset(self, training_label: str):
        """Label the parameter as successful (1) or failure (0)."""
        assert (
            self._most_recent_parameter is not None
            and self._most_recent_abstract_action_descriptor is not None
        )
        assert self._last_observation is not None

        if self._train_or_eval == "eval":
            return

        label = 1 if training_label == "success" else 0

        self._parameter_dataset[self._most_recent_abstract_action_descriptor].append(
            (self._last_observation, self._most_recent_parameter, label)
        )

    def _add_most_recent_abstract_action_to_dataset(self, training_label: str):
        """Label the abstract action as successful (1) or failure (0)."""
        assert self._most_recent_abstract_action_descriptor
        assert self._current_abstract_plan

        if self._train_or_eval == "eval":
            return

        is_failure = training_label != "success"

        prev_abstract_states = tuple(
            self._current_abstract_plan[0][: self._current_abstract_plan_step + 1]
        )
        prev_abstract_actions = tuple(
            self._current_abstract_plan[1][: self._current_abstract_plan_step]
        )

        key = (prev_abstract_states, prev_abstract_actions)
        failures, attempts = self._abstract_action_dataset[
            self._most_recent_abstract_action_descriptor
        ][key]
        self._abstract_action_dataset[self._most_recent_abstract_action_descriptor][
            key
        ] = (failures + int(is_failure), attempts + 1)

    def _add_abstract_plan_to_dataset(self, training_label: str):
        assert self._current_abstract_plan

        if self._train_or_eval == "eval":
            return

        label = 1 if training_label == "success" else 0

        # Add the completed abstract plan up to the point where this function is called
        abstract_states = tuple(
            self._current_abstract_plan[0][: self._current_abstract_plan_step + 1]
        )
        abstract_actions = tuple(
            self._current_abstract_plan[1][: self._current_abstract_plan_step + 1]
        )
        key = (abstract_states, abstract_actions)

        # Only upgrade from failure (0) to success (1), never downgrade.
        self._abstract_plan_dataset[key] = max(
            self._abstract_plan_dataset.get(key, 0), label
        )

    def _update_scorers(self) -> None:
        """Retrain the parameter and abstract action scorers given the current stored
        dataset."""

        # First reformat dataset, converting (failures, attempts) → failure rate
        abstract_action_dataset = {
            k: list(
                self._make_data(
                    abstract_plan,
                    failures / attempts if attempts > 0 else 0.0,
                )
                for abstract_plan, (failures, attempts) in v.items()
            )
            for k, v in self._abstract_action_dataset.items()
        }

        self.train_abstract_action_scorer(abstract_action_dataset)
        self.train_parameter_policy(self._parameter_dataset)

        # Log predicted vs actual failure rates after training to verify convergence.
        assert self._last_observation is not None
        x0 = self._env_models.observation_to_state(self._last_observation)
        s0 = self._env_models.state_abstractor(x0)
        for op, scorer in self._abstract_action_to_action_scorer.items():
            action_data: defaultdict | dict = self._abstract_action_dataset.get(
                op.short_str, {}
            )
            counts = list(action_data.values())
            total_failures = sum(f for f, _ in counts)
            total_attempts = sum(a for _, a in counts)
            actual_rate = (
                total_failures / total_attempts if total_attempts > 0 else float("nan")
            )
            predicted = scorer.score(([s0], []))
            logging.info(
                "[Scorer] %s predicted=%.4f actual=%.4f (failures=%d attempts=%d)",
                op.short_str,
                predicted,
                actual_rate,
                total_failures,
                total_attempts,
            )
            # Log per-history-key predictions to diagnose whether the scorer
            # distinguishes different contexts (e.g. Reach vs Reach-after-Widen).
            for (key_states, key_actions), (failures, attempts) in action_data.items():
                if attempts == 0:
                    continue
                key_actual = failures / attempts
                key_predicted = scorer.score((list(key_states), list(key_actions)))
                action_str = [a.short_str for a in key_actions]
                logging.info(
                    "[Scorer-ctx] %s history=%s  predicted=%.4f actual=%.4f"
                    " (failures=%d attempts=%d)",
                    op.short_str,
                    action_str,
                    key_predicted,
                    key_actual,
                    failures,
                    attempts,
                )

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
        parameter_policy = ParameterPolicy(
            self._current_controller,
            scoring_function,
            param_sample_count=self._param_sample_count,
        )
        optimal_params = parameter_policy.sample_parameters(x, obs, self._rng)
        self._most_recent_parameter = optimal_params
        self._most_recent_abstract_action_descriptor = a.short_str

        # Reset controller
        self._current_controller.reset(x, optimal_params)

    def _return_dummy_action(self) -> _U:
        assert self._env_models.action_space.shape
        action_shape = self._env_models.action_space.shape
        stationary_action = np.zeros(
            action_shape, dtype=self._env_models.action_space.dtype
        )
        dummy_action = cast(_U, stationary_action)
        return dummy_action

    def _abstract_action_completed(
        self, a: GroundOperator, s: RelationalAbstractState, ns: RelationalAbstractState
    ) -> bool:
        """Return True if the current abstract action has completed.

        Actions that change the abstract state are complete when the current state
        matches the planned next state. Actions with no effects (e.g. pure moves) are
        complete when the controller signals termination.
        """
        if a.add_effects or a.delete_effects:
            return s == ns
        return (
            self._current_controller is not None
            and self._current_controller.terminated()
        )

    def _record_action_success(self) -> None:
        """Record that the current abstract action and its parameter succeeded."""
        self._add_most_recent_abstract_action_to_dataset("success")
        self._add_most_recent_parameter_to_dataset("success")

    def _get_action(self) -> _U:
        assert self._current_abstract_plan is not None

        # Advance past any already-completed abstract actions.
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

            assert self._last_observation is not None
            x = self._env_models.observation_to_state(self._last_observation)
            s = self._env_models.state_abstractor(x)

            a = self._current_abstract_plan[1][self._current_abstract_plan_step]
            ns = self._current_abstract_plan[0][self._current_abstract_plan_step + 1]

            if self._abstract_action_completed(a, s, ns):
                self._record_action_success()
                self._current_abstract_plan_step += 1
                advanced = True
                continue

            # If we haven't reached the next state,
            # determine if we still need a new controller.
            if self._timestep == 0 or self._reset_controller:
                advanced = True

            break

        # Get the last observed state.
        x = self._env_models.observation_to_state(self._last_observation)
        # If we advanced, we need to reset a new parameterized controller.
        if advanced:
            self._resample_controller(x, self._last_observation)
            self._reset_controller = False

        # We are using the same controller as before.
        else:
            assert self._current_controller
            self._current_controller.observe(x)

        assert self._current_controller

        while self._num_resamples < self._max_resamples:
            # Try to take a low-level action from the controller.
            try:
                # Take one more low-level action.
                self._last_action = self._current_controller.step()
                assert self._last_action is not None

                return self._last_action
            # If low level action failed, store the parameter that failed!
            except (TrajectorySamplingFailure, IndexError):
                # If training, store the previous parameter.
                if self._train_or_eval == "train":
                    self._add_most_recent_abstract_action_to_dataset("failure")
                    self._add_most_recent_parameter_to_dataset("failure")
                    self._add_abstract_plan_to_dataset("failure")

                # Resample Controller
                self._resample_controller(x, self._last_observation)

                self._num_resamples += 1

        # After trying a certain number of resamples, update the scorers
        # and Q-function only every train_every exhaustion events.
        self._resample_exhaustion_count += 1
        current_plan_str = (
            [a.short_str for a in self._current_abstract_plan[1]]
            if self._current_abstract_plan
            else []
        )
        logging.info(
            "[BALD] Exhaustion #%d (timestep %d), failed plan: %s",
            self._resample_exhaustion_count,
            self._timestep,
            current_plan_str,
        )
        if self._resample_exhaustion_count % self._train_every == 0:
            self._update_scorers()
            self.train_ensemble_nets()

        # Generate candidate plans
        candidate_plans = self.generate_candidate_plans()

        # Score candidate plans and return best plan
        plan_to_execute = self.score_candidate_plans(candidate_plans)
        logging.info(
            "[BALD] Selected plan: %s",
            [a.short_str for a in plan_to_execute[1]],
        )
        self._plan_from_exploit = False

        # Set new plan as the plan to execute
        self._current_abstract_plan = plan_to_execute
        self._current_abstract_plan_step = 0

        # Reset the current controller so it will be reinitialized for the new plan
        self._current_controller = None
        self._reset_controller = True

        # Reset num_resamples
        self._num_resamples = 0

        # Return dummy action
        return self._return_dummy_action()

    def step(self) -> _U:
        """Get the next action to take."""
        self._last_action = self._get_action()
        self._timestep += 1
        return self._last_action

    def get_resample_exhaustion_count(self) -> int:
        """Return cumulative number of resample exhaustion events (triggers scorer/net updates)."""
        return self._resample_exhaustion_count

    def get_abstract_plan(self) -> Skeleton | None:
        """Return the current abstract plan."""
        return self._current_abstract_plan

    def is_plan_from_exploit(self) -> bool:
        """Return True if the current plan was selected by the exploit planner."""
        return self._plan_from_exploit

    def get_current_abstract_plan_step(self) -> int:
        """Return the current step in the abstract plan."""
        return self._current_abstract_plan_step

    def get_most_recent_parameter(self) -> Any:
        """Returns most recent parameter from the controller."""
        return self._most_recent_parameter

    def get_most_recent_abstract_action_str(self) -> str:
        """Returns name of current abstract action."""
        assert self._most_recent_abstract_action_descriptor is not None
        return self._most_recent_abstract_action_descriptor

    def get_parameter_dataset(self) -> defaultdict[str, list]:
        """Return the collected parameter dataset."""
        return self._parameter_dataset

    def get_abstract_plan_dataset(self) -> dict[tuple, int]:
        """Return the collected abstract plan dataset."""
        return self._abstract_plan_dataset

    def get_abstract_action_dataset(
        self,
    ) -> defaultdict[str, defaultdict[FrozenSkeleton, tuple[int, int]]]:
        """Return the collected abstract action dataset."""
        return self._abstract_action_dataset

    def get_loss_metrics(
        self,
    ) -> dict[str, list]:
        """Return the loss metrics for each abstract action."""
        return self._loss_metrics

    def get_q_network_loss_metrics(self) -> list[float]:
        """Return the per-epoch training losses across all Q-network training runs."""
        return self._q_loss_metrics

    def _make_data(self, abstract_plan: Skeleton | FrozenSkeleton, label: float):
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
                for abstract_plan, training_label in self._abstract_plan_dataset.items()
            ),
            "abstract_action_dataset.pkl": {
                k: list(
                    self._make_data(
                        abstract_plan,
                        failures / attempts if attempts > 0 else 0.0,
                    )
                    for abstract_plan, (failures, attempts) in v.items()
                )
                for k, v in self._abstract_action_dataset.items()
            },
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
