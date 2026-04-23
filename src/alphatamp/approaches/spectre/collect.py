"""Per-episode collection: pool → per-skeleton refinement → serialized record.

Non-short-circuiting: every skeleton in the pool is refined regardless of
earlier successes. This is the defining difference from the standard SeSaMe
planner behavior (which stops at first success). See ``SPECTRE_METHOD_SPEC.md``
§5.1 and ``SPECTRE_TRAINING_PIPELINE_SPEC.md`` §6.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import time
from pathlib import Path

import kinder
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import RelationalControllerGenerator
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from relational_structs import GroundAtom, GroundOperator, Object

from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz
from alphatamp.approaches.spectre.schema import (
    EpisodeRecord,
    OutcomeRecord,
    ProvenanceBlock,
    SkeletonRecord,
    SummaryBlock,
)


def _refinement_seed(rule: str, problem_id: int, skeleton_idx: int) -> int:
    """Stable blake2b-8 hash.

    §6.3 of the pipeline spec.
    """
    payload = f"{rule}:{problem_id}:{skeleton_idx}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFF_FFFFFFFF


def _collect_all_objects(
    initial_state: RelationalAbstractState,
    skeleton_pool: list[tuple[list[RelationalAbstractState], list[GroundOperator]]],
    goal_atoms: set[GroundAtom],
) -> dict[str, str]:
    """Build a name→type_name registry covering every object referenced anywhere."""
    registry: dict[str, str] = {}

    def _add(obj: Object) -> None:
        existing = registry.get(obj.name)
        if existing is not None and existing != obj.type.name:
            raise AssertionError(
                f"Object {obj.name} appears with conflicting types"
                f" {existing} vs {obj.type.name}"
            )
        registry[obj.name] = obj.type.name

    for obj in initial_state.objects:
        _add(obj)
    for atom in initial_state.atoms:
        for e in atom.entities:
            _add(e)
    for atom in goal_atoms:
        for e in atom.entities:
            _add(e)
    for state_plan, action_plan in skeleton_pool:
        for s in state_plan:
            for obj in s.objects:
                _add(obj)
            for atom in s.atoms:
                for e in atom.entities:
                    _add(e)
        for op in action_plan:
            for obj in op.parameters:
                _add(obj)
    return registry


def collect_episode(
    cfg: CollectionConfig,
    problem_id: int,
) -> EpisodeRecord:
    """Collect one episode: pool generation + per-skeleton refinement.

    The pool size is capped by ``cfg.K_max`` via ``itertools.islice``. Each
    skeleton is refined independently with a deterministic seed; outcomes are
    recorded for every skeleton regardless of earlier successes.
    """
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=problem_id)
        env_models = create_bilevel_planning_models(
            cfg.model_name,
            env.observation_space,
            env.action_space,
            **cfg.model_kwargs,
        )

        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        assert isinstance(goal, RelationalAbstractGoal)

        # Shared per-problem scratchpad; planner and refiner both mutate.
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        plan_generator: RelationalHeuristicSearchAbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                env_models.types,
                env_models.predicates,
                env_models.operators,
                heuristic_name=cfg.heuristic_name,
                seed=problem_id,
            )
        )

        # Draw the pool (lazy A*; islice caps at K_max).
        pool_iter = plan_generator(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg)
        skeleton_pool: list[
            tuple[list[RelationalAbstractState], list[GroundOperator]]
        ] = list(itertools.islice(pool_iter, cfg.K_max))

        # Refiner reused across every skeleton for this problem.
        trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(env_models.skills),
            transition_function=env_models.transition_fn,
            state_abstractor=env_models.state_abstractor,
            max_trajectory_steps=cfg.max_trajectory_steps,
        )

        skeleton_records: list[SkeletonRecord] = []
        outcome_records: list[OutcomeRecord] = []
        first_success_idx: int | None = None
        total_wall_clock = 0.0

        for idx, (state_plan, action_plan) in enumerate(skeleton_pool):
            skeleton_records.append(
                SkeletonRecord(
                    skeleton_idx=idx,
                    operator_seq=tuple(action_plan),
                    final_abstract_state=state_plan[-1],
                )
            )

            seed = _refinement_seed(cfg.refinement_seed_rule, problem_id, idx)
            refiner: BacktrackingRefiner = BacktrackingRefiner(
                trajectory_sampler=trajectory_sampler,
                num_sampling_attempts_per_step=cfg.num_sampling_attempts_per_step,
                seed=seed,
            )

            start = time.perf_counter()
            outcome: str
            error_info: dict[str, str] | None = None
            try:
                plan = refiner(
                    x0, state_plan, action_plan, cfg.refinement_timeout_s, bpg
                )
                outcome = "success" if plan is not None else "fail"
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                outcome = "error"
                error_info = {"cls": type(exc).__name__, "msg": str(exc)}
            wall_clock = time.perf_counter() - start
            total_wall_clock += wall_clock

            if outcome == "success" and first_success_idx is None:
                first_success_idx = idx

            outcome_records.append(
                OutcomeRecord(
                    skeleton_idx=idx,
                    outcome=outcome,  # type: ignore[arg-type]
                    refinement_wall_clock_s=wall_clock,
                    refinement_seed=seed,
                    error_info=error_info,
                )
            )

        summary = SummaryBlock(
            num_skeletons=len(skeleton_records),
            num_success=sum(1 for o in outcome_records if o.outcome == "success"),
            num_fail=sum(1 for o in outcome_records if o.outcome == "fail"),
            num_error=sum(1 for o in outcome_records if o.outcome == "error"),
            first_success_idx=first_success_idx,
            total_wall_clock_s=total_wall_clock,
            pool_truncated=len(skeleton_records) >= cfg.K_max,
        )

        provenance = ProvenanceBlock(
            problem_id=problem_id,
            env_id=cfg.env_id,
            env_variant=cfg.env_variant,
            split=cfg.split,
            config_hash=cfg.config_hash,
            problem_seed=problem_id,
            git_sha=cfg.git_sha,
            collection_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            package_versions=dict(cfg.package_versions),
        )

        object_registry = _collect_all_objects(s0, skeleton_pool, goal.atoms)

        return EpisodeRecord(
            provenance=provenance,
            initial_abstract_state=s0,
            goal_atoms=frozenset(goal.atoms),
            object_registry=object_registry,
            skeleton_pool=tuple(skeleton_records),
            outcomes=tuple(outcome_records),
            summary=summary,
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]


def episode_path(
    data_root: Path,
    env_variant: str,
    split: str,
    problem_id: int,
) -> Path:
    """Canonical on-disk path for one episode file."""
    return (
        data_root
        / "raw"
        / env_variant
        / split
        / "episodes"
        / f"ep_{problem_id:05d}.pkl.gz"
    )


def collect_and_save(
    cfg: CollectionConfig,
    data_root: Path,
    problem_id: int,
    overwrite: bool = False,
) -> Path:
    """Collect one episode and atomically write it.

    Skip if already present.
    """
    path = episode_path(data_root, cfg.env_variant, cfg.split, problem_id)
    if path.exists() and not overwrite:
        return path
    record = collect_episode(cfg, problem_id)
    atomic_write_pickle_gz(record, path)
    return path


def save_config_yaml(cfg: CollectionConfig, data_root: Path) -> Path:
    """Persist a config YAML under ``<data_root>/configs/collection_<hash>.yaml``."""
    out = data_root / "configs" / f"collection_{cfg.config_hash}.yaml"
    if not out.exists():
        cfg.to_yaml(out)
    return out


def collect_and_save_result(
    cfg: CollectionConfig,
    data_root: Path,
    problem_id: int,
) -> tuple[int, Path | None, str | None]:
    """Worker-safe wrapper around :func:`collect_and_save`.

    Returns ``(problem_id, path, error_message)``. Exceptions are captured so
    one bad problem doesn't kill a worker pool. Lives in the package (not in
    the Hydra entrypoint script) so it has a stable importable qualname under
    ``multiprocessing`` ``spawn`` start method.
    """
    try:
        path = collect_and_save(cfg, data_root, problem_id)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        return problem_id, None, f"{type(exc).__name__}: {exc}"
    return problem_id, path, None
