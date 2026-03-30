import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypeAlias, TypeVar, cast

import dill
import numpy as np
from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Plan,
    PlanningProblem,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from relational_structs import GroundOperator

from alphatamp.approaches.base_approach import BaseApproach
from alphatamp.scoring_utils.batch_ranking import BatchRankingAbstractPlanGenerator

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]
FrozenGroundOpSequence: TypeAlias = tuple[GroundOperator, ...]
SkeletonCandidate: TypeAlias = FrozenSkeleton | FrozenGroundOpSequence


class BoxApproach(BaseApproach[_O, _X, _U]):
    """An approach that implements the BOX algorithm for skeleton selection."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        skeleton_batch_size: int = 100,
        num_training_skeletons_per_problem: int = 10,
        training_planning_timeout: float = 5,
        exploration_constant: float = math.sqrt(2),
        training_label_mode: str = "effort",
        failure_penalty_multiplier: float = 1.0,
    ):
        super().__init__(env_models, seed)
        self._seed = seed
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._skeleton_batch_size = skeleton_batch_size
        self._num_training_skeletons_per_problem = num_training_skeletons_per_problem
        self._training_planning_timeout = training_planning_timeout
        self._exploration_constant = exploration_constant  # c parameter in UCB
        if training_label_mode not in {"binary", "effort"}:
            raise ValueError("training_label_mode must be either 'binary' or 'effort'.")
        if failure_penalty_multiplier < 1.0:
            raise ValueError("failure_penalty_multiplier must be >= 1.0.")
        self._training_label_mode = training_label_mode
        self._failure_penalty_multiplier = failure_penalty_multiplier

        # Create the trajectory sampler for refinement
        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the abstract plan generator.
        self._base_abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
            )
        )

        self._batched_abstract_plan_generator: AbstractPlanGenerator = (
            BatchRankingAbstractPlanGenerator(
                self._base_abstract_plan_generator,
                score_fn=self._score_skeleton,
                batch_size=self._skeleton_batch_size,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        self._planner = SesamePlanner(
            self._batched_abstract_plan_generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

        # Use the same refiner at training time that we will use at test time.
        self._refiner = self._planner._refiner  # pylint: disable=protected-access

        # Store training data.
        # List of dicts, where each dict maps FrozenSkeleton -> (utility score, is_success).
        # Higher is better for utility score.
        self._data: list[dict[FrozenSkeleton, tuple[float, bool]]] = []
        self._training_initial_states: list[_X] = []

        # BOX Model parameters (will init once after training)
        self._skeletons_vocab: list[SkeletonCandidate] = []
        self._skeleton_to_idx: dict[SkeletonCandidate, int] = {}
        self._prior_mu: Optional[np.ndarray] = None
        self._prior_sigma: Optional[np.ndarray] = None
        self._score_matrix: Optional[np.ndarray] = None
        self._model_uses_op_sequence_vocab = False
        self._offline_seed_ids: Optional[list[int]] = None
        self._offline_seed_id_to_row_idx: Dict[int, int] = {}
        self._offline_applicability: Optional[np.ndarray] = None
        self._offline_success: Optional[np.ndarray] = None
        self._offline_refinement_time: Optional[np.ndarray] = None
        self._offline_dataset_timeout: Optional[float] = None
        self._model_built = False  # ensure we don't rebuild multiple times

    @staticmethod
    def _freeze_ground_op_sequence(skeleton: Skeleton) -> FrozenGroundOpSequence:
        return tuple(skeleton[1])

    @staticmethod
    def _load_encoder_dataset_payload(
        artifact_path: str | Path,
    ) -> dict[str, Any]:
        with open(Path(artifact_path), "rb") as file:
            payload = dill.load(file)
        if not isinstance(payload, dict):
            raise TypeError(
                "Expected dict payload in encoder dataset artifact, got "
                f"{type(payload)}"
            )
        return payload

    @staticmethod
    def _validate_encoder_dataset_payload(
        payload: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[FrozenGroundOpSequence], float]:
        dataset = payload.get("dataset")
        if not isinstance(dataset, dict):
            raise TypeError("Encoder dataset payload missing dict 'dataset' field")

        required = {"op_sequence_vocab", "applicability", "success", "refinement_time"}
        missing = required - set(dataset)
        if missing:
            raise KeyError(
                "Encoder dataset payload missing dataset keys: " f"{sorted(missing)}"
            )

        vocab = [tuple(op_sequence) for op_sequence in dataset["op_sequence_vocab"]]
        applicability = np.asarray(dataset["applicability"], dtype=float)
        success = np.asarray(dataset["success"], dtype=float)
        refinement_time = np.asarray(dataset["refinement_time"], dtype=float)

        if applicability.ndim != 2 or success.ndim != 2 or refinement_time.ndim != 2:
            raise ValueError(
                "Encoder dataset arrays applicability/success/refinement_time must be rank-2"
            )
        if (
            applicability.shape != success.shape
            or applicability.shape != refinement_time.shape
        ):
            raise ValueError(
                "Encoder dataset shape mismatch: "
                f"A{applicability.shape} Y{success.shape} T{refinement_time.shape}"
            )
        if applicability.shape[1] != len(vocab):
            raise ValueError(
                "Encoder dataset vocab size mismatch: "
                f"num_cols={applicability.shape[1]} vocab={len(vocab)}"
            )
        if np.any(success > applicability):
            raise ValueError(
                "Encoder dataset has success=1 entries where applicability=0"
            )
        if np.any(refinement_time < 0.0):
            raise ValueError("Encoder dataset refinement_time must be non-negative")

        config = payload.get("config")
        if not isinstance(config, dict):
            raise TypeError("Encoder dataset payload missing dict 'config' field")
        if "training_planning_timeout" not in config:
            raise KeyError(
                "Encoder dataset payload config missing 'training_planning_timeout'"
            )
        training_timeout = float(config["training_planning_timeout"])
        if training_timeout <= 0:
            raise ValueError("training_planning_timeout must be > 0")

        return applicability, success, refinement_time, vocab, training_timeout

    def _score_from_dataset_entry(
        self,
        is_success: bool,
        recorded_refinement_time: float,
        timeout: float,
    ) -> float:
        effective_time = float(recorded_refinement_time)
        if not is_success:
            effective_time = float(timeout) * self._failure_penalty_multiplier
        return -1.0 * effective_time

    @staticmethod
    def _compute_prior_statistics(
        score_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if score_matrix.ndim != 2:
            raise ValueError("score_matrix must be rank-2")

        num_rows, num_cols = score_matrix.shape
        if num_cols == 0:
            return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)

        mu = np.mean(score_matrix, axis=0)
        if num_rows <= 1:
            sigma = np.zeros((num_cols, num_cols), dtype=float)
        else:
            sigma = np.cov(score_matrix, rowvar=False)
            if np.ndim(sigma) == 0:
                sigma = np.asarray([[float(sigma)]], dtype=float)
            sigma = np.asarray(sigma, dtype=float)
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(mu, dtype=float), sigma

    def build_box_model_from_encoder_dataset_payload(
        self,
        payload: dict[str, Any],
    ) -> None:
        """Initialize BOX priors directly from a build_encoder_dataset artifact.

        The encoder dataset provides a fixed grounded-op-sequence vocabulary and a
        complete matrix of applicability / success / refinement_time values over
        training problems. We treat each vocabulary column as a BOX candidate and
        convert each matrix entry into the same utility score used by BOX training.
        """
        applicability, success, refinement_time, vocab, training_timeout = (
            self._validate_encoder_dataset_payload(payload)
        )

        num_rows, num_cols = success.shape
        score_matrix = np.zeros((num_rows, num_cols), dtype=float)
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                score_matrix[row_idx, col_idx] = self._score_from_dataset_entry(
                    is_success=bool(success[row_idx, col_idx] > 0.5),
                    recorded_refinement_time=float(refinement_time[row_idx, col_idx]),
                    timeout=training_timeout,
                )

        self._data = []
        self._training_initial_states = []
        self._skeletons_vocab = list(vocab)
        self._skeleton_to_idx = {
            op_sequence: idx for idx, op_sequence in enumerate(self._skeletons_vocab)
        }
        self._score_matrix = score_matrix
        self._prior_mu, self._prior_sigma = self._compute_prior_statistics(score_matrix)
        self._model_uses_op_sequence_vocab = True
        self._model_built = True

        unique_rows = np.unique(score_matrix, axis=0)
        print(
            "[BoxApproach] Built artifact-backed BOX model with score matrix "
            f"shape {score_matrix.shape}, {len(unique_rows)} unique rows, and "
            f"{len(self._skeletons_vocab)} grounded-op-sequence candidates."
        )

    def build_box_model_from_encoder_dataset_artifact(
        self,
        artifact_path: str | Path,
    ) -> None:
        """Load a build_encoder_dataset artifact and initialize BOX priors from it."""
        payload = self._load_encoder_dataset_payload(artifact_path)
        self.build_box_model_from_encoder_dataset_payload(payload)

    def load_offline_planning_dataset_from_encoder_dataset_payload(
        self,
        payload: dict[str, Any],
    ) -> None:
        """Load an offline replay dataset for BOX planning without refinement calls.

        The loaded payload is typically a validation/test artifact from
        `build_encoder_dataset.py`. Once loaded, `run_offline_planning_*()` can
        replay BOX's candidate selection using the stored applicability, success,
        and refinement-time matrices instead of invoking the refiner.
        """
        applicability, success, refinement_time, vocab, training_timeout = (
            self._validate_encoder_dataset_payload(payload)
        )

        if not self._model_built:
            raise RuntimeError(
                "Build the BOX model before loading an offline planning dataset. "
                "Call build_box_model_from_encoder_dataset_artifact() first."
            )
        if not self._model_uses_op_sequence_vocab:
            raise RuntimeError(
                "Offline encoder-dataset planning requires an op-sequence-backed "
                "BOX model. Build the model from an encoder dataset artifact first."
            )

        payload_vocab = list(vocab)
        if len(payload_vocab) != len(self._skeletons_vocab):
            raise ValueError(
                "Offline dataset vocab size mismatch with BOX model: "
                f"dataset={len(payload_vocab)} model={len(self._skeletons_vocab)}"
            )
        for idx, (dataset_entry, model_entry) in enumerate(
            zip(payload_vocab, self._skeletons_vocab)
        ):
            if dataset_entry != model_entry:
                raise ValueError(
                    "Offline dataset vocab mismatch with BOX model at index " f"{idx}"
                )

        seed_ids_raw = payload.get("seed_ids")
        seed_ids: list[int] | None = None
        if seed_ids_raw is not None:
            seed_ids = [int(seed_id) for seed_id in seed_ids_raw]
            if len(seed_ids) != applicability.shape[0]:
                raise ValueError(
                    "Offline dataset seed_ids length mismatch with number of rows: "
                    f"len(seed_ids)={len(seed_ids)} rows={applicability.shape[0]}"
                )

        self._offline_applicability = np.asarray(applicability, dtype=float)
        self._offline_success = np.asarray(success, dtype=float)
        self._offline_refinement_time = np.asarray(refinement_time, dtype=float)
        self._offline_dataset_timeout = float(training_timeout)
        self._offline_seed_ids = seed_ids
        self._offline_seed_id_to_row_idx = (
            {seed_id: idx for idx, seed_id in enumerate(seed_ids)}
            if seed_ids is not None
            else {}
        )

    def load_offline_planning_dataset_from_encoder_dataset_artifact(
        self,
        artifact_path: str | Path,
    ) -> None:
        """Load an offline replay dataset from a build_encoder_dataset artifact."""
        payload = self._load_encoder_dataset_payload(artifact_path)
        self.load_offline_planning_dataset_from_encoder_dataset_payload(payload)

    @staticmethod
    def _offline_attempt_allowed(attempt_time: float, remaining_time: float) -> bool:
        return float(attempt_time) <= float(remaining_time) + 1e-9

    def _run_offline_planning_by_row_index(
        self,
        row_idx: int,
        timeout: float,
    ) -> dict[str, Any]:
        if not self._model_built:
            raise RuntimeError("Build the BOX model before offline planning.")
        if not self._model_uses_op_sequence_vocab:
            raise RuntimeError(
                "Offline encoder-dataset planning requires an op-sequence-backed "
                "BOX model."
            )
        if self._offline_applicability is None:
            raise RuntimeError(
                "Offline planning dataset not loaded. Call "
                "load_offline_planning_dataset_from_encoder_dataset_artifact() first."
            )
        assert self._prior_mu is not None
        assert self._prior_sigma is not None
        assert self._offline_success is not None
        assert self._offline_refinement_time is not None
        assert self._offline_dataset_timeout is not None

        num_rows = int(self._offline_applicability.shape[0])
        if row_idx < 0 or row_idx >= num_rows:
            raise IndexError(f"row_idx {row_idx} out of range for offline dataset")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")

        applicable_row = self._offline_applicability[row_idx] > 0.5
        success_row = self._offline_success[row_idx] > 0.5
        refinement_time_row = self._offline_refinement_time[row_idx]

        candidate_indices = np.flatnonzero(applicable_row).astype(int)
        observed_indices: List[int] = []
        observed_scores: List[float] = []
        attempted_indices: List[int] = []
        attempted_op_sequences: List[FrozenGroundOpSequence] = []
        attempted_scores: List[float] = []

        elapsed = 0.0
        success_found = False

        while len(candidate_indices) > len(observed_indices):
            if elapsed >= timeout:
                break

            untried_indices = np.asarray(
                [idx for idx in candidate_indices if idx not in observed_indices],
                dtype=int,
            )
            if len(untried_indices) == 0:
                break

            if len(observed_indices) == 0:
                mu_t = self._prior_mu[untried_indices]
                sigma_t = self._prior_sigma[np.ix_(untried_indices, untried_indices)]
            else:
                idx_1 = untried_indices
                idx_2 = np.asarray(observed_indices, dtype=int)
                mu_1 = self._prior_mu[idx_1]
                mu_2 = self._prior_mu[idx_2]
                sigma_11 = self._prior_sigma[np.ix_(idx_1, idx_1)]
                sigma_12 = self._prior_sigma[np.ix_(idx_1, idx_2)]
                sigma_21 = self._prior_sigma[np.ix_(idx_2, idx_1)]
                sigma_22 = self._prior_sigma[np.ix_(idx_2, idx_2)]
                sigma_22_reg = sigma_22 + 1e-6 * np.eye(len(idx_2))
                try:
                    term_t = np.linalg.solve(sigma_22_reg, sigma_21)
                    term = term_t.T
                except np.linalg.LinAlgError:
                    sigma_22_inv = np.linalg.pinv(sigma_22_reg)
                    term = sigma_12 @ sigma_22_inv

                j_observed = np.asarray(observed_scores, dtype=float)
                mu_t = mu_1 + term @ (j_observed - mu_2)
                sigma_t = sigma_11 - term @ sigma_21

            sigma_diag = np.diag(sigma_t)
            sigma_diag = np.maximum(sigma_diag, 0.0)
            ucb_scores = mu_t + self._exploration_constant * np.sqrt(sigma_diag)
            best_idx = int(untried_indices[int(np.argmax(ucb_scores))])

            remaining = timeout - elapsed
            attempt_time = float(refinement_time_row[best_idx])
            if not self._offline_attempt_allowed(attempt_time, remaining):
                break

            elapsed += attempt_time
            observed_indices.append(best_idx)
            attempted_indices.append(best_idx)

            op_sequence = cast(FrozenGroundOpSequence, self._skeletons_vocab[best_idx])
            attempted_op_sequences.append(op_sequence)

            score = self._score_from_dataset_entry(
                is_success=bool(success_row[best_idx]),
                recorded_refinement_time=attempt_time,
                timeout=self._offline_dataset_timeout,
            )
            observed_scores.append(score)
            attempted_scores.append(score)

            if bool(success_row[best_idx]):
                success_found = True
                break

        final_elapsed = float(elapsed if success_found else timeout)
        result: dict[str, Any] = {
            "row_idx": int(row_idx),
            "seed_id": (
                int(self._offline_seed_ids[row_idx])
                if self._offline_seed_ids is not None
                else None
            ),
            "success": bool(success_found),
            "elapsed_time": final_elapsed,
            "attempted_indices": list(attempted_indices),
            "attempted_op_sequences": list(attempted_op_sequences),
            "attempted_scores": list(attempted_scores),
            "num_attempts": int(len(attempted_indices)),
            "timeout": float(timeout),
        }
        return result

    def run_offline_planning_by_row_index(
        self,
        row_idx: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Replay BOX planning on an offline dataset row without refinement calls."""
        return self._run_offline_planning_by_row_index(row_idx, timeout)

    def run_offline_planning_by_seed_id(
        self,
        seed_id: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Replay BOX planning on an offline dataset seed id without refinement
        calls."""
        if not self._offline_seed_id_to_row_idx:
            raise RuntimeError(
                "Offline dataset seed ids are unavailable. Ensure the loaded artifact "
                "contains a 'seed_ids' field."
            )
        if int(seed_id) not in self._offline_seed_id_to_row_idx:
            raise KeyError(f"seed_id {seed_id} not found in offline dataset")
        row_idx = self._offline_seed_id_to_row_idx[int(seed_id)]
        return self._run_offline_planning_by_row_index(row_idx, timeout)

    def _max_effort_for_skeleton(self, num_ops: int) -> float:
        """Upper-bound effort baseline in unit of sampler calls."""
        return float(max(1, num_ops * self._samples_per_step))

    def _score_from_refinement_legacy_normalized(
        self, plan_found: bool, num_sampler_calls: int, num_ops: int
    ) -> float:
        """Legacy normalized effort scoring kept for reference."""
        if self._training_label_mode == "binary":
            return 1.0 if plan_found else 0.0

        max_effort = self._max_effort_for_skeleton(num_ops)
        failure_effort = self._failure_penalty_multiplier * max_effort
        effort = float(num_sampler_calls) if plan_found else failure_effort
        normalized_effort = min(effort, failure_effort) / max_effort
        return 1.0 / (1.0 + normalized_effort)

    def _score_from_refinement_legacy_linear(
        self, plan_found: bool, num_sampler_calls: int, num_ops: int
    ) -> float:
        """Legacy linear effort scoring kept for reference."""
        if self._training_label_mode == "binary":
            return 1.0 if plan_found else 0.0

        max_effort = self._max_effort_for_skeleton(num_ops)
        failure_effort = self._failure_penalty_multiplier * max_effort
        effort = float(num_sampler_calls) if plan_found else failure_effort
        return max(0.0, failure_effort - effort)

    def _score_from_refinement(
        self, elapsed_wall_time: float, timeout: float, timed_out: bool
    ) -> float:
        """Convert refinement outcome into a utility score (higher is better)."""
        effective_time = float(elapsed_wall_time)
        if timed_out:
            effective_time = float(timeout) * self._failure_penalty_multiplier
        return -1.0 * effective_time

    def _refine_with_score(
        self,
        x0: _X,
        skel_states: list[RelationalAbstractState],
        skel_ops: list[GroundOperator],
        timeout: float,
        bpg: BilevelPlanningGraph,
    ) -> tuple[Plan | None, float]:
        """Run refiner and score with wall-clock elapsed time (higher is better)."""
        wall_start_time = time.perf_counter()
        try:
            plan = self._refiner(x0, skel_states, skel_ops, timeout, bpg)
        except Exception:
            plan = None

        elapsed_wall_time = time.perf_counter() - wall_start_time
        timed_out = plan is None

        score = self._score_from_refinement(elapsed_wall_time, timeout, timed_out)
        return plan, score

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        """Collect training data by generating skeletons and checking refinability."""
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        problem_data: dict[FrozenSkeleton, tuple[float, bool]] = {}

        # Generate a fixed number of skeletons for this training problem
        gen = self._base_abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            self._training_planning_timeout,
            bpg,
        )
        print("[BoxApproach] Generating training skeletons...")

        count = 0
        for skeleton in gen:
            # Only generate up to _num_training_skeletons_per_problem skeletons
            if count >= self._num_training_skeletons_per_problem:
                break

            # Attempt refinement
            plan, score = self._refine_with_score(
                x0,
                skeleton[0],
                skeleton[1],
                self._training_planning_timeout,
                bpg,
            )
            frozen_skeleton = (tuple(skeleton[0]), tuple(skeleton[1]))
            problem_data[frozen_skeleton] = (score, plan is not None)
            count += 1

        self._data.append(problem_data)
        self._training_initial_states.append(x0)
        self._score_matrix = None
        self._model_built = False

    def _backfill_missing_training_labels(self) -> None:
        """Evaluate missing (problem, skeleton) pairs before building BOX priors."""
        if not self._data:
            return

        if len(self._data) != len(self._training_initial_states):
            raise RuntimeError("Training data/state mismatch in BoxApproach backfill.")

        all_skeletons: Set[FrozenSkeleton] = set()
        for problem_data in self._data:
            all_skeletons.update(problem_data.keys())

        for i, (problem_data, x0) in enumerate(
            zip(self._data, self._training_initial_states)
        ):
            missing_skeletons = all_skeletons - set(problem_data.keys())
            if not missing_skeletons:
                continue

            print(
                "[BoxApproach] Backfilling "
                f"{len(missing_skeletons)} missing skeleton labels for "
                f"training problem {i}."
            )

            s0 = self._env_models.state_abstractor(x0)
            bpg: BilevelPlanningGraph = BilevelPlanningGraph()
            bpg.add_state_node(x0)
            bpg.add_abstract_state_node(s0)
            bpg.add_state_abstractor_edge(x0, s0)

            for frozen_skeleton in sorted(missing_skeletons, key=str):
                skel_states = list(frozen_skeleton[0])
                skel_ops = list(frozen_skeleton[1])
                plan, score = self._refine_with_score(
                    x0,
                    skel_states,
                    skel_ops,
                    self._training_planning_timeout,
                    bpg,
                )
                problem_data[frozen_skeleton] = (score, plan is not None)

    def _build_box_model(self) -> None:
        """Builds the prior mu and sigma from collected training data."""
        # Don't rebuild if already built
        if self._model_built:
            return

        self._backfill_missing_training_labels()

        # Identify fixed set of skeletons from training
        # Take the union of all skeletons seen during training as analogy to "constraints" from BOX paper
        all_skeletons: Set[FrozenSkeleton] = set()
        for problem_data in self._data:
            all_skeletons.update(problem_data.keys())

        for i, problem_data in enumerate(self._data):
            missing_skeletons = all_skeletons - set(problem_data.keys())
            if len(missing_skeletons) > 0:
                raise RuntimeError(
                    "Backfilling failed: training problem "
                    f"{i} is still missing {len(missing_skeletons)} skeletons."
                )

        self._skeletons_vocab = sorted(list(all_skeletons), key=lambda s: str(s))
        self._skeleton_to_idx = {s: i for i, s in enumerate(self._skeletons_vocab)}

        N = len(self._data)
        M = len(self._skeletons_vocab)

        if M == 0 or N == 0:
            # Fallback if no data
            self._prior_mu = np.zeros(M)
            self._prior_sigma = np.eye(M)
            self._score_matrix = np.zeros((N, M), dtype=float)
            self._model_built = True
            return

        # Construct score matrix D (N x M) from explicit utility labels.
        D = np.zeros((N, M))
        for i, problem_data in enumerate(self._data):
            for skel, (score, _) in problem_data.items():
                j = self._skeleton_to_idx[skel]
                D[i, j] = float(score)

        self._score_matrix = D

        # debugging: show the number of unique rows in D
        unique_rows = np.unique(D, axis=0)
        print(
            f"[BoxApproach] Built score matrix D with shape {D.shape}, {len(unique_rows)} unique rows from {N} training problems and {M} skeletons."
        )

        self._prior_mu, self._prior_sigma = self._compute_prior_statistics(D)

        self._model_uses_op_sequence_vocab = False
        self._model_built = True

    def _run_planning_with_full_skeleton_vocab(
        self,
        problem: PlanningProblem[_X, _U],
        timeout: float,
    ) -> Plan[_X, _U] | None:
        assert self._prior_mu is not None
        assert self._prior_sigma is not None

        num_candidates = len(self._skeletons_vocab)
        all_indices = np.arange(num_candidates)
        observed_indices: List[int] = []
        observed_scores: List[float] = []
        tried_skeletons: Set[FrozenSkeleton] = set()

        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        print("Beginning BOX approach with timeout of {:.2f} seconds.".format(timeout))

        start_time = time.perf_counter()

        if num_candidates > 0:
            for _ in range(num_candidates):
                elapsed = time.perf_counter() - start_time
                if elapsed >= timeout:
                    break

                untried_mask = np.ones(num_candidates, dtype=bool)
                untried_mask[observed_indices] = False
                untried_indices = all_indices[untried_mask]
                if len(untried_indices) == 0:
                    break

                if len(observed_indices) == 0:
                    mu_t = self._prior_mu[untried_indices]
                    sigma_t = self._prior_sigma[
                        np.ix_(untried_indices, untried_indices)
                    ]
                else:
                    idx_1 = untried_indices
                    idx_2 = observed_indices
                    mu_1 = self._prior_mu[idx_1]
                    mu_2 = self._prior_mu[idx_2]
                    sigma_11 = self._prior_sigma[np.ix_(idx_1, idx_1)]
                    sigma_12 = self._prior_sigma[np.ix_(idx_1, idx_2)]
                    sigma_21 = self._prior_sigma[np.ix_(idx_2, idx_1)]
                    sigma_22 = self._prior_sigma[np.ix_(idx_2, idx_2)]
                    sigma_22_reg = sigma_22 + 1e-6 * np.eye(len(idx_2))
                    try:
                        term_T = np.linalg.solve(sigma_22_reg, sigma_21)
                        term = term_T.T
                    except np.linalg.LinAlgError:
                        sigma_22_inv = np.linalg.pinv(sigma_22_reg)
                        term = sigma_12 @ sigma_22_inv

                    j_observed = np.array(observed_scores)
                    mu_t = mu_1 + term @ (j_observed - mu_2)
                    sigma_t = sigma_11 - term @ sigma_21

                sigma_diag = np.diag(sigma_t)
                sigma_diag = np.maximum(sigma_diag, 0.0)
                ucb_scores = mu_t + self._exploration_constant * np.sqrt(sigma_diag)
                best_idx = untried_indices[np.argmax(ucb_scores)]

                skeleton = self._skeletons_vocab[best_idx]
                assert isinstance(skeleton, tuple)
                tried_skeletons.add(skeleton)
                skel_states = list(skeleton[0])
                skel_ops = list(skeleton[1])

                elapsed = time.perf_counter() - start_time
                if elapsed >= timeout:
                    break
                remaining = timeout - elapsed

                plan, score = self._refine_with_score(
                    x0,
                    skel_states,
                    skel_ops,
                    remaining,
                    bpg,
                )

                elapsed = time.perf_counter() - start_time
                if plan is not None:
                    print(
                        f"BOX found a plan with score {score:.4f} after {elapsed:.2f} seconds."
                    )
                    return plan

                observed_indices.append(best_idx)
                observed_scores.append(score)

        elapsed = time.perf_counter() - start_time
        if elapsed < timeout:
            remaining = timeout - elapsed
            gen = self._base_abstract_plan_generator(
                x0, s0, problem.goal, remaining, bpg
            )
            for skeleton in gen:
                elapsed = time.perf_counter() - start_time
                if elapsed >= timeout:
                    break
                remaining = timeout - elapsed

                frozen = (tuple(skeleton[0]), tuple(skeleton[1]))
                if frozen in tried_skeletons:
                    continue

                tried_skeletons.add(frozen)
                plan, _ = self._refine_with_score(
                    x0,
                    skeleton[0],
                    skeleton[1],
                    remaining,
                    bpg,
                )
                if plan is not None:
                    print(
                        f"BOX fallback found a plan with skeleton {frozen} after {elapsed:.2f} seconds."
                    )
                    return plan

        return None

    def _run_planning_with_op_sequence_vocab(
        self,
        problem: PlanningProblem[_X, _U],
        timeout: float,
    ) -> Plan[_X, _U] | None:
        assert self._prior_mu is not None
        assert self._prior_sigma is not None

        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        start_time = time.perf_counter()
        print(
            "Beginning artifact-backed BOX approach with timeout of "
            f"{timeout:.2f} seconds."
        )

        vocab_op_sequences = set(self._skeletons_vocab)
        matched_skeletons: Dict[FrozenGroundOpSequence, Skeleton] = {}
        tried_op_sequences: Set[FrozenGroundOpSequence] = set()

        generator = self._base_abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            timeout,
            bpg,
        )
        for skeleton in generator:
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                break

            op_sequence = self._freeze_ground_op_sequence(skeleton)
            if (
                op_sequence in vocab_op_sequences
                and op_sequence not in matched_skeletons
            ):
                matched_skeletons[op_sequence] = skeleton
                if len(matched_skeletons) == len(vocab_op_sequences):
                    break

        candidate_indices = np.asarray(
            [self._skeleton_to_idx[op_sequence] for op_sequence in matched_skeletons],
            dtype=int,
        )
        observed_indices: List[int] = []
        observed_scores: List[float] = []

        while len(candidate_indices) > len(observed_indices):
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                break

            untried_indices = np.asarray(
                [idx for idx in candidate_indices if idx not in observed_indices],
                dtype=int,
            )
            if len(untried_indices) == 0:
                break

            if len(observed_indices) == 0:
                mu_t = self._prior_mu[untried_indices]
                sigma_t = self._prior_sigma[np.ix_(untried_indices, untried_indices)]
            else:
                idx_1 = untried_indices
                idx_2 = np.asarray(observed_indices, dtype=int)
                mu_1 = self._prior_mu[idx_1]
                mu_2 = self._prior_mu[idx_2]
                sigma_11 = self._prior_sigma[np.ix_(idx_1, idx_1)]
                sigma_12 = self._prior_sigma[np.ix_(idx_1, idx_2)]
                sigma_21 = self._prior_sigma[np.ix_(idx_2, idx_1)]
                sigma_22 = self._prior_sigma[np.ix_(idx_2, idx_2)]
                sigma_22_reg = sigma_22 + 1e-6 * np.eye(len(idx_2))
                try:
                    term_T = np.linalg.solve(sigma_22_reg, sigma_21)
                    term = term_T.T
                except np.linalg.LinAlgError:
                    sigma_22_inv = np.linalg.pinv(sigma_22_reg)
                    term = sigma_12 @ sigma_22_inv

                j_observed = np.array(observed_scores)
                mu_t = mu_1 + term @ (j_observed - mu_2)
                sigma_t = sigma_11 - term @ sigma_21

            sigma_diag = np.diag(sigma_t)
            sigma_diag = np.maximum(sigma_diag, 0.0)
            ucb_scores = mu_t + self._exploration_constant * np.sqrt(sigma_diag)
            best_idx = int(untried_indices[np.argmax(ucb_scores)])

            op_sequence = cast(FrozenGroundOpSequence, self._skeletons_vocab[best_idx])
            skeleton = matched_skeletons[op_sequence]
            tried_op_sequences.add(op_sequence)

            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            plan, score = self._refine_with_score(
                x0,
                skeleton[0],
                skeleton[1],
                remaining,
                bpg,
            )

            elapsed = time.perf_counter() - start_time
            if plan is not None:
                print(
                    f"Artifact-backed BOX found a plan with score {score:.4f} "
                    f"after {elapsed:.2f} seconds."
                )
                return plan

            observed_indices.append(best_idx)
            observed_scores.append(score)

        elapsed = time.perf_counter() - start_time
        if elapsed < timeout:
            remaining = timeout - elapsed
            gen = self._base_abstract_plan_generator(
                x0, s0, problem.goal, remaining, bpg
            )
            for skeleton in gen:
                elapsed = time.perf_counter() - start_time
                if elapsed >= timeout:
                    break
                remaining = timeout - elapsed

                op_sequence = self._freeze_ground_op_sequence(skeleton)
                if op_sequence in tried_op_sequences:
                    continue

                tried_op_sequences.add(op_sequence)
                plan, _ = self._refine_with_score(
                    x0,
                    skeleton[0],
                    skeleton[1],
                    remaining,
                    bpg,
                )
                if plan is not None:
                    print(
                        "Artifact-backed BOX fallback found a plan with op sequence "
                        f"{op_sequence} after {elapsed:.2f} seconds."
                    )
                    return plan

        return None

    def get_score_matrix_copy(self) -> np.ndarray:
        """Return a defensive copy of the BOX score matrix D."""
        if not self._model_built or self._score_matrix is None:
            raise RuntimeError(
                "BOX score matrix is not available. Build the model first."
            )
        return np.array(self._score_matrix, copy=True)

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        # Ensure model is built
        self._build_box_model()
        self._model_built = True  # redundant but explicit

        if self._model_uses_op_sequence_vocab:
            plan = self._run_planning_with_op_sequence_vocab(problem, timeout)
        else:
            plan = self._run_planning_with_full_skeleton_vocab(problem, timeout)
        if plan is None:
            raise TimeoutError("No plan found within timeout")
        return plan

    def run_planning_filtered(self, init_obs: _O, timeout: float) -> Plan[_X, _U]:
        """Use the base generator but filters out skeletons that have consistently
        failed in training."""

        start_time = time.perf_counter()
        problem = self._observation_to_planning_problem(init_obs)

        # Identify skeletons that have been tried but never succeeded
        tried_skeletons: Set[FrozenSkeleton] = set()
        successful_skeletons: Set[FrozenSkeleton] = set()

        for problem_data in self._data:
            for skel, (_, is_success) in problem_data.items():
                tried_skeletons.add(skel)
                if is_success:
                    successful_skeletons.add(skel)

        # Skeletons that were tried but never succeeded
        always_failed_skeletons = tried_skeletons - successful_skeletons

        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout:
            raise TimeoutError("No plan found within timeout")
        remaining = timeout - elapsed

        gen = self._base_abstract_plan_generator(x0, s0, problem.goal, remaining, bpg)

        for skeleton in gen:
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            frozen = (tuple(skeleton[0]), tuple(skeleton[1]))

            if frozen in always_failed_skeletons:
                continue

            plan, _ = self._refine_with_score(
                x0,
                skeleton[0],
                skeleton[1],
                remaining,
                bpg,
            )

            if plan is not None:
                return plan

        raise TimeoutError("No plan found within timeout")

    def run_planning_successful_first(
        self, init_obs: _O, timeout: float
    ) -> Plan[_X, _U]:
        """Use previously successful skeletons first, then fall back to the
        generator."""

        start_time = time.perf_counter()
        problem = self._observation_to_planning_problem(init_obs)
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # Collect successful skeletons from training
        successful_skeletons: List[FrozenSkeleton] = []
        seen_successful: Set[FrozenSkeleton] = set()

        for problem_data in self._data:
            for skel, (_, is_success) in problem_data.items():
                if is_success and skel not in seen_successful:
                    successful_skeletons.append(skel)
                    seen_successful.add(skel)

        tried_skeletons: Set[FrozenSkeleton] = set()

        # Try successful skeletons first
        for successful_skeleton in successful_skeletons:
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            tried_skeletons.add(successful_skeleton)

            # Convert FrozenSkeleton back to lists
            skel_states = list(successful_skeleton[0])
            skel_ops = list(successful_skeleton[1])

            plan, _ = self._refine_with_score(
                x0,
                skel_states,
                skel_ops,
                remaining,
                bpg,
            )

            if plan is not None:
                return plan

        # 3. Fallback to generator
        print("[SuccessfulFirst] Fallback to generator.")

        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout:
            raise TimeoutError("No plan found within timeout")

        # recalculate remaining time
        remaining = timeout - elapsed

        gen = self._base_abstract_plan_generator(x0, s0, problem.goal, remaining, bpg)

        # TODO: could refactor to avoid code duplication
        for generated_skeleton in gen:
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            frozen = (
                tuple(generated_skeleton[0]),
                tuple(generated_skeleton[1]),
            )

            if frozen in tried_skeletons:
                continue

            tried_skeletons.add(frozen)

            plan, _ = self._refine_with_score(
                x0,
                generated_skeleton[0],
                generated_skeleton[1],
                remaining,
                bpg,
            )

            if plan is not None:
                return plan

        raise TimeoutError("No plan found within timeout")

    def _score_skeleton(
        self, skeleton: Skeleton, failed_skeletons: list[Skeleton]
    ) -> float:
        """Score skeletons.

        In this BOX implementation, scoring is handled dynamically in _run_planning.
        """
        if not self._model_built:
            return 0.0
        assert self._prior_mu is not None

        frozen = (tuple(skeleton[0]), tuple(skeleton[1]))
        if frozen in self._skeleton_to_idx:
            idx = self._skeleton_to_idx[frozen]
            return float(self._prior_mu[idx])

        if self._model_uses_op_sequence_vocab:
            op_sequence = self._freeze_ground_op_sequence(skeleton)
            if op_sequence in self._skeleton_to_idx:
                idx = self._skeleton_to_idx[op_sequence]
                return float(self._prior_mu[idx])

        return 0.0
