"""Encoder approach phase 1: build a skeleton vocabulary and plan from it."""

# mypy: disable-error-code=var-annotated

from __future__ import annotations

import os
import sys
import time
from typing import Any, TypeVar
import numpy as np
from tqdm.auto import tqdm

import kinder
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

from alphatamp.approaches.base_approach import BaseApproach
from alphatamp.structs import FrozenGroundOpSequence, Skeleton

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class EncoderApproach(BaseApproach[_O, _X, _U]):
    """Build a frequent grounded-op vocabulary and plan using matching skeletons.

    This implements phase 1 of the encoder roadmap:
    - collect skeletons during training,
    - count occurrences of grounded operator sequences,
    - keep top-k sequences as a vocabulary,
    - restrict planning-time refinement to sequences in the vocabulary.
    """

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 20,
        samples_per_step: int = 5,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        num_training_skeletons_per_problem: int = 20,
        training_planning_timeout: float = 10.0,
        vocabulary_size: int = 25,
        env_id: str | None = None,
        prune_cycles: bool = True,
    ) -> None:
        super().__init__(env_models, seed)
        if num_training_skeletons_per_problem < 1:
            raise ValueError("num_training_skeletons_per_problem must be >= 1")
        if training_planning_timeout <= 0:
            raise ValueError("training_planning_timeout must be > 0")
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be >= 1")

        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._num_training_skeletons_per_problem = num_training_skeletons_per_problem
        self._training_planning_timeout = training_planning_timeout
        self._vocabulary_size = vocabulary_size
        self._env_id = env_id
        self._prune_cycles = prune_cycles

        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        self._base_abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
            )
        )

        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        self._planner = SesamePlanner(
            self._base_abstract_plan_generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )
        self._refiner = self._planner._refiner  # pylint: disable=protected-access

        self._op_sequence_counts: dict[FrozenGroundOpSequence, int] = {}
        self._op_sequence_vocab: list[FrozenGroundOpSequence] = []
        self._op_sequence_to_idx: dict[FrozenGroundOpSequence, int] = {}
        self._num_training_problems = 0

    @staticmethod
    def _freeze_ground_op_sequence(skeleton: Skeleton) -> FrozenGroundOpSequence:
        return tuple(skeleton[1])

    def reconstruct_abstract_state_sequence(
        self,
        initial_abstract_state: RelationalAbstractState,
        grounded_operator_sequence: FrozenGroundOpSequence,
    ) -> list[RelationalAbstractState] | None:
        """Roll out abstract states under grounded operators.

        Returns a full abstract-state sequence starting at `initial_abstract_state`.
        If any grounded operator is not applicable from the current abstract state,
        returns None.
        """
        state_sequence: list[RelationalAbstractState] = [initial_abstract_state]
        current_state = initial_abstract_state

        for grounded_operator in grounded_operator_sequence:
            next_state: RelationalAbstractState | None = None
            for candidate_operator, candidate_next_state in self._abstract_successor_fn(
                current_state
            ):
                if candidate_operator == grounded_operator:
                    next_state = candidate_next_state
                    break

            if next_state is None:
                return None

            state_sequence.append(next_state)
            current_state = next_state

        return state_sequence

    def _refresh_vocabulary(self) -> None:
        ranked = sorted(
            self._op_sequence_counts.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
        self._op_sequence_vocab = [
            op_sequence for op_sequence, _ in ranked[: self._vocabulary_size]
        ]
        self._op_sequence_to_idx = {
            op_sequence: index
            for index, op_sequence in enumerate(self._op_sequence_vocab)
        }

    def build_full_vocab(self, seed_ids: list[int]) -> None:
        """Build grounded-op sequence counts from explicit seed IDs."""
        if self._env_id is None:
            raise ValueError("env_id must be set to build vocabulary from seeds.")

        self._op_sequence_counts = {}
        self._op_sequence_vocab = []
        self._op_sequence_to_idx = {}

        if not seed_ids:
            return

        kinder.register_all_environments()
        env = kinder.make(self._env_id)
        try:
            for seed_id in seed_ids:
                if seed_id % 10 == 0:
                    print(f"Building vocab: processing seed {seed_id}...")
                obs, _ = env.reset(seed=int(seed_id))
                problem = self._observation_to_planning_problem(obs)

                x0 = problem.initial_state
                s0 = self._env_models.state_abstractor(x0)

                bpg: BilevelPlanningGraph = BilevelPlanningGraph()  # type: ignore[var-annotated]
                bpg.add_state_node(x0)
                bpg.add_abstract_state_node(s0)
                bpg.add_state_abstractor_edge(x0, s0)

                generated = 0
                generator = self._base_abstract_plan_generator(
                    x0,
                    s0,
                    problem.goal,
                    self._training_planning_timeout,
                    bpg,
                )
                for skeleton in generator:
                    if generated >= self._num_training_skeletons_per_problem:
                        break

                    if self._prune_cycles and self._has_abstract_cycle(skeleton):
                        generated += 1
                        continue
                    op_sequence = self._freeze_ground_op_sequence(skeleton)
                    self._op_sequence_counts[op_sequence] = (
                        self._op_sequence_counts.get(op_sequence, 0) + 1
                    )
                    generated += 1
        finally:
            env.close()  # type: ignore[no-untyped-call]

    def _has_abstract_cycle(self, skeleton: Skeleton) -> bool:
        """Detect if skeleton has a cycle in the abstract state sequence."""
        abstract_states = skeleton[0]
        seen = set()
        for state in abstract_states:
            if state in seen:
                return True
            seen.add(state)
        return False

    @staticmethod
    def _abstract_state_sequence_has_cycle(
        abstract_state_sequence: list[Any],
    ) -> bool:
        """Return True iff any abstract state appears more than once."""
        seen = set()
        for state in abstract_state_sequence:
            if state in seen:
                return True
            seen.add(state)
        return False

    def build_vocab(
        self,
        seed_ids: list[int],
        k: int,
        prune_cycles: bool | None = None,
    ) -> list[FrozenGroundOpSequence]:
        """Build op-sequence vocabulary from explicit seed IDs and return top-k."""
        if k < 1:
            raise ValueError("k must be >= 1")
        self._vocabulary_size = k
        prev_prune_cycles = self._prune_cycles
        if prune_cycles is not None:
            self._prune_cycles = prune_cycles
        self.build_full_vocab(seed_ids)
        self._prune_cycles = prev_prune_cycles
        self._refresh_vocabulary()
        return self.get_op_sequence_vocabulary()

    def build_dataset(
        self,
        seed_ids: list[int],
        show_progress: bool | None = None,
        log_every_seeds: int = 5,
    ) -> dict[str, object]:
        """Build applicability/success/runtime matrices over (seed, op-sequence).

        Returns a dict with keys:
        - "seed_ids": list[int]
        - "op_sequence_vocab": list[FrozenGroundOpSequence]
        - "applicability": np.ndarray shape (N, M), binary float32
        - "success": np.ndarray shape (N, M), binary float32
        - "refinement_time": np.ndarray shape (N, M), float32 seconds

        Semantics:
        - If applicability[i, j] == 0, refinement is skipped and
          success[i, j] = 0, refinement_time[i, j] = timeout.
        - If applicable, refinement_time stores measured wall-clock duration,
          capped at timeout.
        """
        if self._env_id is None:
            raise ValueError("env_id must be set to build a dataset.")
        if not self._op_sequence_vocab:
            raise ValueError(
                "Vocabulary is empty. Call build_vocab() or build_full_vocab()+_refresh_vocabulary() first."
            )
        if log_every_seeds < 1:
            raise ValueError("log_every_seeds must be >= 1")

        is_tty = sys.stderr.isatty()
        in_slurm = "SLURM_JOB_ID" in os.environ
        if show_progress is None:
            show_progress = is_tty and not in_slurm

        seed_id_list = [int(seed_id) for seed_id in seed_ids]

        num_seeds = len(seed_id_list)
        num_vocab = len(self._op_sequence_vocab)

        applicability = np.zeros((num_seeds, num_vocab), dtype=np.float32)
        success_matrix = np.zeros((num_seeds, num_vocab), dtype=np.float32)
        refinement_time = np.zeros((num_seeds, num_vocab), dtype=np.float32)
        steps_completed_fraction = np.zeros((num_seeds, num_vocab), dtype=np.float32)

        initial_low_level_states = []
        initial_abstract_states = []
        problem_goals = []

        dataset_start_time = time.perf_counter()

        kinder.register_all_environments()
        env = kinder.make(self._env_id)
        try:
            seed_iter = tqdm(
                enumerate(seed_id_list),
                total=num_seeds,
                desc="Building dataset (seeds)",
                unit="seed",
                leave=True,
                dynamic_ncols=True,
                disable=not show_progress,
                mininterval=1.0,
            )
            for seed_idx, seed_id in seed_iter:
                obs, _ = env.reset(seed=seed_id)
                problem = self._observation_to_planning_problem(obs)

                x0 = problem.initial_state
                s0 = self._env_models.state_abstractor(x0)

                initial_low_level_states.append(x0)
                initial_abstract_states.append(s0)
                problem_goals.append(problem.goal)

                op_iter = tqdm(
                    enumerate(self._op_sequence_vocab),
                    total=num_vocab,
                    desc=f"Seed {seed_id} (op seq)",
                    unit="seq",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not show_progress,
                    mininterval=1.0,
                )
                for op_sequence_idx, op_sequence in op_iter:
                    abstract_state_sequence = self.reconstruct_abstract_state_sequence(
                        s0, op_sequence
                    )

                    # incompatible with initial state, skip refinement entirely.
                    # applicability, success_matrix, steps_completed_fraction,
                    # and refinement_time all stay 0.0.
                    if abstract_state_sequence is None:
                        continue

                    applicability[seed_idx, op_sequence_idx] = 1.0

                    bpg: BilevelPlanningGraph = BilevelPlanningGraph()  # type: ignore[var-annotated]
                    bpg.add_state_node(x0)
                    bpg.add_abstract_state_node(s0)
                    bpg.add_state_abstractor_edge(x0, s0)

                    wall_start_time = time.perf_counter()
                    try:
                        plan = self._refiner(
                            x0,
                            abstract_state_sequence,
                            list(op_sequence),
                            self._training_planning_timeout,
                            bpg,
                        )
                    except Exception:
                        plan = None

                    elapsed_wall_time = time.perf_counter() - wall_start_time
                    measured_time = min(
                        elapsed_wall_time,
                        self._training_planning_timeout,
                    )

                    success_matrix[seed_idx, op_sequence_idx] = (
                        1.0 if plan is not None else 0.0
                    )
                    refinement_time[seed_idx, op_sequence_idx] = measured_time

                    # Count how many abstract states from the skeleton were
                    # successfully reached during refinement (including backtracking).
                    # The trajectory sampler only adds an abstract state to the BPG
                    # when a trajectory successfully reaches it, so set membership
                    # directly encodes per-step refinement success.
                    abstract_state_set = set(bpg.abstract_states)
                    n_steps = len(abstract_state_sequence) - 1  # exclude initial s0
                    if n_steps > 0:
                        steps_done = sum(
                            1 for s in abstract_state_sequence[1:]
                            if s in abstract_state_set
                        )
                        steps_completed_fraction[seed_idx, op_sequence_idx] = (
                            float(steps_done) / n_steps
                        )

                # Helpful live summary while outer tqdm estimates ETA.
                seed_applicable = int(np.sum(applicability[seed_idx]))
                seed_success = int(np.sum(success_matrix[seed_idx]))
                seed_iter.set_postfix(
                    applicable=seed_applicable,
                    success=seed_success,
                )

                # Batch-friendly periodic logs (single-line snapshots) when bars are disabled.
                if (not show_progress) and (
                    (seed_idx + 1) % log_every_seeds == 0 or (seed_idx + 1) == num_seeds
                ):
                    elapsed = time.perf_counter() - dataset_start_time
                    done = seed_idx + 1
                    avg_per_seed = elapsed / done
                    remaining = max(0.0, (num_seeds - done) * avg_per_seed)
                    print(
                        "[build_dataset] "
                        f"{done}/{num_seeds} seeds "
                        f"({100.0 * done / max(1, num_seeds):.1f}%) | "
                        f"elapsed={elapsed/60.0:.1f}m | eta={remaining/60.0:.1f}m | "
                        f"last_seed={seed_id} applicable={seed_applicable} success={seed_success}"
                    )
        finally:
            env.close()  # type: ignore[no-untyped-call]

        return {
            "seed_ids": seed_id_list,
            "op_sequence_vocab": list(self._op_sequence_vocab),
            "applicability": applicability,
            "success": success_matrix,
            "refinement_time": refinement_time,
            "steps_completed_fraction": steps_completed_fraction,
            "skeleton_lengths": np.array(
                [len(seq) for seq in self._op_sequence_vocab], dtype=np.int16
            ),
            "initial_low_level_states": initial_low_level_states,
            "initial_abstract_states": initial_abstract_states,
            "problem_goals": problem_goals,
        }

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        del problem
        self._num_training_problems += 1

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        if not self._op_sequence_vocab:
            raise TimeoutError(
                "No vocabulary skeletons available. Call build_vocab() first."
            )

        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()  # type: ignore[var-annotated]
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        start_time = time.perf_counter()
        vocab_set = set(self._op_sequence_vocab)
        matched_skeletons: dict[FrozenGroundOpSequence, Skeleton] = {}

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
            if op_sequence in vocab_set and op_sequence not in matched_skeletons:
                matched_skeletons[op_sequence] = skeleton
                if len(matched_skeletons) == len(vocab_set):
                    break

        for op_sequence in self._op_sequence_vocab:
            if op_sequence not in matched_skeletons:
                continue
            elapsed = time.perf_counter() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                break

            skeleton = matched_skeletons[op_sequence]
            skel_states = skeleton[0]
            skel_ops = skeleton[1]
            plan = self._refiner(x0, skel_states, skel_ops, remaining, bpg)
            if plan is not None:
                return plan

        raise TimeoutError("No plan found from skeleton vocabulary")

    def get_op_sequence_counts(self) -> dict[FrozenGroundOpSequence, int]:
        """Return a copy of grounded operator sequence counts."""
        return dict(self._op_sequence_counts)

    def get_skeleton_counts(self) -> dict[FrozenGroundOpSequence, int]:
        """Backward-compatible alias for grounded op-sequence counts."""
        return self.get_op_sequence_counts()

    @staticmethod
    def filter_vocab_by_success_rate(
        dataset: dict[str, Any],
        threshold: float,
    ) -> tuple[list[FrozenGroundOpSequence], list[int], dict[str, Any]]:
        """Identify vocabulary entries that meet a minimum success rate.

        Sequences where ``applicable_count == 0`` across all filter seeds are
        treated as having an undefined success rate and removed.

        Args:
            dataset: dict returned by ``build_dataset()``, must contain keys
                ``op_sequence_vocab``, ``applicability``, and ``success``.
            threshold: exclusive lower bound on success_rate required to keep
                a sequence (i.e. a sequence is kept iff
                ``success_rate > threshold``) among sequences that are
                applicable at least once. ``0.0`` therefore removes both
                always-fail and never-applicable sequences; ``0.1`` requires
                a success rate strictly above 10% for applicable sequences.

        Returns:
            A 3-tuple of:

            - ``filtered_vocab``: sequences passing the threshold, ranked by
              success_rate descending.
            - ``keep_indices``: column indices into the original vocab that were
              kept, in the same (success_rate descending) order.
            - ``stats``: dict with keys ``original_size``, ``filtered_size``,
                            ``removed_count``, ``threshold``, ``never_applicable_count``,
                            and ``success_rates`` (list of per-column rates; ``None`` where
                            applicable_count == 0).
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        vocab: list[FrozenGroundOpSequence] = list(dataset["op_sequence_vocab"])
        applicability = np.asarray(dataset["applicability"], dtype=np.float32)
        success = np.asarray(dataset["success"], dtype=np.float32)

        applicable_counts = applicability.sum(axis=0)  # shape (M,)
        success_counts = success.sum(axis=0)           # shape (M,)

        # Per-column success rate; NaN where never applicable.
        success_rates = np.where(
            applicable_counts > 0,
            success_counts / np.maximum(applicable_counts, 1.0),
            np.nan,
        )

        # Keep applicable columns only if they strictly exceed the
        # success-rate threshold. Never-applicable columns are removed.
        applicable_mask = applicable_counts > 0
        keep_applicable_mask = applicable_mask & (success_rates > threshold)

        candidate_applicable_indices = [
            i for i, keep in enumerate(keep_applicable_mask) if keep
        ]
        never_applicable_indices = [
            i for i, is_applicable in enumerate(applicable_mask) if not is_applicable
        ]

        # Sort applicable candidates by descending success_rate.
        candidate_applicable_indices.sort(key=lambda i: float(-success_rates[i]))
        candidate_indices = candidate_applicable_indices

        filtered_vocab = [vocab[i] for i in candidate_indices]

        # Build a human-readable per-column success_rate list (None for never-applicable).
        readable_rates: list[float | None] = [
            float(success_rates[i]) if np.isfinite(success_rates[i]) else None
            for i in range(len(vocab))
        ]
        stats: dict[str, Any] = {
            "original_size": len(vocab),
            "filtered_size": len(filtered_vocab),
            "removed_count": len(vocab) - len(filtered_vocab),
            "threshold": threshold,
            "never_applicable_count": len(never_applicable_indices),
            "success_rates": readable_rates,
        }
        return filtered_vocab, candidate_indices, stats

    @staticmethod
    def apply_vocab_filter_to_dataset(
        dataset: dict[str, Any],
        keep_indices: list[int],
    ) -> dict[str, Any]:
        """Derive a filtered dataset by dropping unwanted vocabulary columns.

        This is a pure offline operation — no simulation is performed.  The
        returned dict has the same structure as ``build_dataset()`` output but
        with ``op_sequence_vocab``, ``applicability``, ``success``, and
        ``refinement_time`` restricted to the kept columns in the given order.

        Args:
            dataset: original wide dataset dict (output of ``build_dataset()``).
            keep_indices: column indices to retain, in the order they should
                appear in the filtered dataset (typically success_rate
                descending, as returned by ``filter_vocab_by_success_rate``).

        Returns:
            A new dict with the same keys; matrix-valued entries are sliced
            along the vocabulary axis; per-seed list entries are copied
            unchanged.
        """
        if not keep_indices:
            raise ValueError("keep_indices must be non-empty")

        vocab: list[FrozenGroundOpSequence] = list(dataset["op_sequence_vocab"])
        applicability = np.asarray(dataset["applicability"], dtype=np.float32)
        success = np.asarray(dataset["success"], dtype=np.float32)
        refinement_time = np.asarray(dataset["refinement_time"], dtype=np.float32)

        idx = list(keep_indices)
        filtered: dict[str, Any] = {
            "seed_ids": list(dataset["seed_ids"]),
            "op_sequence_vocab": [vocab[i] for i in idx],
            "applicability": applicability[:, idx],
            "success": success[:, idx],
            "refinement_time": refinement_time[:, idx],
        }
        # Propagate optional per-seed fields unchanged.
        for key in ("initial_low_level_states", "initial_abstract_states", "problem_goals"):
            if key in dataset:
                filtered[key] = list(dataset[key])
        return filtered

    def set_vocab(self, vocab: list[FrozenGroundOpSequence]) -> None:
        """Inject a pre-built vocabulary without re-running collection.

        Useful when the vocabulary was computed by another process or a
        preceding step and should be shared across multiple workers without
        each worker rebuilding it from scratch.
        """
        self._op_sequence_vocab = list(vocab)
        self._op_sequence_to_idx = {
            op_sequence: index
            for index, op_sequence in enumerate(self._op_sequence_vocab)
        }

    def get_op_sequence_vocabulary(self) -> list[FrozenGroundOpSequence]:
        """Return a copy of the current top-k grounded-op sequence vocabulary."""
        return list(self._op_sequence_vocab)

    def get_skeleton_vocabulary(self) -> list[FrozenGroundOpSequence]:
        """Backward-compatible alias for grounded op-sequence vocabulary."""
        return self.get_op_sequence_vocabulary()
