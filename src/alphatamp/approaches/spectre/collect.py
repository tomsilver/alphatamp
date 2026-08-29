"""Per-episode collection: pool → per-skeleton refinement → serialized record.

Non-short-circuiting: every skeleton in the pool is refined regardless of earlier
successes. This is the defining difference from the standard SeSaMe planner behavior
(which stops at first success). See ``docs/archive/SPECTRE_METHOD_SPEC.md`` §5.1 and
``docs/archive/SPECTRE_TRAINING_PIPELINE_SPEC.md`` §6.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import random
import time
from pathlib import Path
from typing import Callable

import kinder
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import RelationalControllerGenerator
from gymnasium.spaces import Space
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

_STICK_BUTTON_MODEL_NAME = "stickbutton2d"
_RESTOCK3D_MODEL_NAME = "restock3d"
# v2 continuous-packing restock: place_tall/place_short over two shelf sections. Distinct
# model_name from v1 (different models/sampler/geometry), so its collection is a separate
# env_variant and cannot be mixed with restock3d_v1.
_RESTOCK3D_V2_MODEL_NAME = "restock3d_v2"
_RESTOCK3D_V3_MODEL_NAME = "restock3d_v3"

# Restock3D's recording sampler needs the models' internal sim + region_infos, which the
# frozen SesameModels cannot carry. `_make_env_models` stashes them here for
# `_make_trajectory_sampler`; collection is per-process sequential (episode N's models
# are built before its sampler), so a module-level holder is safe (separate workers).
_restock_extras: dict[str, object] = {}

# hybrid_prune (Restock3D-v3 real collection): fraction of the analytic-INFEASIBLE candidates
# that are still real-refined, to audit "analytic-infeasible => real-infeasible" and catch rare
# analytic false negatives. Analytic-FEASIBLE candidates are always real-refined (they are the
# analytic false positives we most need real labels for).
_HYBRID_AUDIT_FRAC = 0.25


def _refinement_seed(rule: str, problem_id: int, skeleton_idx: int) -> int:
    """Stable blake2b-8 hash.

    §6.3 of the pipeline spec.
    """
    payload = f"{rule}:{problem_id}:{skeleton_idx}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) & 0x7FFFFFFF_FFFFFFFF


def _make_env_models(
    cfg: CollectionConfig, observation_space: Space, action_space: Space
) -> SesameModels:
    """Dispatch on ``cfg.model_name`` to build the SesameModels for this env.

    Restock3D builds its own kinematic models (custom env, not a kinder factory) and
    stashes the internal sim + region_infos for the recording sampler. All other envs
    fall through to the kinder factory. Imports are deferred so callers that never build
    env models do not pay the cost.
    """
    # pylint: disable=import-outside-toplevel
    if cfg.model_name == _RESTOCK3D_MODEL_NAME:
        from alphatamp.approaches.spectre.envs.restock3d.models import (
            create_restock3d_models,
        )

        bundle = create_restock3d_models(
            observation_space, action_space, int(cfg.model_kwargs["stratum"])
        )
        _restock_extras["sim"] = bundle.sim
        _restock_extras["region_infos"] = bundle.region_infos
        _restock_extras["goal_names"] = bundle.abstractor.goal_object_names()
        return bundle.models

    if cfg.model_name == _RESTOCK3D_V2_MODEL_NAME:
        from alphatamp.approaches.spectre.envs.restock3d.models_v2 import (
            create_restock3d_v2_models,
        )

        bundle_v2 = create_restock3d_v2_models(
            observation_space, action_space, int(cfg.model_kwargs["stratum"])
        )
        _restock_extras["sim"] = bundle_v2.sim
        # The v2 recording sampler takes the 2 section bands via the same `region_infos`
        # channel (keys `section_0`/`section_1`); the probe dispatches on op name.
        _restock_extras["region_infos"] = bundle_v2.section_infos
        _restock_extras["goal_names"] = bundle_v2.abstractor.goal_object_names()
        return bundle_v2.models

    if cfg.model_name == _RESTOCK3D_V3_MODEL_NAME:
        from alphatamp.approaches.spectre.envs.restock3d.models_v3 import (
            create_restock3d_v3_models,
        )

        bundle_v3 = create_restock3d_v3_models(
            observation_space, action_space, int(cfg.model_kwargs["stratum"])
        )
        _restock_extras["sim"] = bundle_v3.sim
        _restock_extras["region_infos"] = bundle_v3.section_infos
        _restock_extras["goal_names"] = bundle_v3.abstractor.goal_object_names()
        return bundle_v3.models

    from kinder_bilevel_planning.env_models import (
        create_bilevel_planning_models,
    )

    return create_bilevel_planning_models(
        cfg.model_name,
        observation_space,
        action_space,
        **cfg.model_kwargs,
    )


def _make_plan_generator(
    cfg: CollectionConfig,
    env_models: SesameModels,
    obs: dict[str, object] | object,
    problem_id: int,
    x0: object = None,
):  # pragma: no cover — return type union widens to whatever the impl supports
    """Build the abstract plan generator.

    Two-way dispatch:

    - StickButton2D → A* over a **geometry-aware** heuristic. Required, not an
      optimization: kinder's symbolic model lets ``RobotPressButton*`` apply to any
      button, including ones past the robot's reach, so hff ranks physically
      unrefinable stick-free plans first and they crowd out the pool. Opt out with
      ``plan_generator="heuristic_search"`` to get the stock hff ordering.
    - Any other kinder env → A*+FF (the ``plan_generator`` field is ignored; those
      envs have no closed-form option).
    """
    if (
        cfg.model_name == _STICK_BUTTON_MODEL_NAME
        and cfg.plan_generator != "heuristic_search"
    ):
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (
            make_plan_generator,
        )

        state = x0 if x0 is not None else env_models.observation_to_state(obs)
        return make_plan_generator(env_models, state, seed=problem_id)
    if cfg.model_name == _RESTOCK3D_MODEL_NAME and cfg.plan_generator == "astar_eager":
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.envs.restock3d.eager_search import (
            EagerValidityPlanGenerator,
        )
        from alphatamp.approaches.spectre.envs.restock3d.eager_tables import (
            EagerWeights,
            build_tables,
            make_penalty,
        )
        from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
            RegionInfo,
        )

        region_infos: dict[str, RegionInfo]
        region_infos = _restock_extras["region_infos"]  # type: ignore[assignment]
        goal_names: list[str] = _restock_extras["goal_names"]  # type: ignore[assignment]
        # Pass the sim + this problem's initial state so build_tables computes the F1 blockers map
        # via grasp_blockers -- the T5 penalty needs it on the clutter strata (r1/r3).
        tables = build_tables(
            region_infos, goal_names, sim=_restock_extras.get("sim"), state=x0
        )
        return EagerValidityPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            heuristic_name=cfg.heuristic_name,
            seed=problem_id,
            penalty_fn=make_penalty(tables, EagerWeights()),
        )
    if (
        cfg.model_name in (_RESTOCK3D_V2_MODEL_NAME, _RESTOCK3D_V3_MODEL_NAME)
        and cfg.plan_generator != "heuristic_search"
    ):
        # v2 default = the geometry-informed nearest-first plan-gen prior (the DEPLOYED
        # generator). It orders the pool feasible-pick-first while still enumerating the
        # place_short-on-tall (F3) variants, so a small K_max pool contains both a feasible
        # skeleton and labelled negatives -- tractable AND diverse, unlike a stock-hff pool
        # whose feasible plan is buried thousands deep on the hard configs. Opt out with
        # ``plan_generator="heuristic_search"`` for the stock hff ordering.
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.envs.restock3d.plan_generator_v2 import (
            GeometryGuidedRestockPlanGenerator,
            pick_distance_from_state,
        )

        state = x0 if x0 is not None else env_models.observation_to_state(obs)
        v2_goal_names: list[str] = _restock_extras["goal_names"]  # type: ignore[assignment]
        return GeometryGuidedRestockPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            seed=problem_id,
            pick_distance=pick_distance_from_state(state, v2_goal_names),
            lam=1.0,
        )
    del obs  # heuristic-search path takes its inputs from env_models alone
    return RelationalHeuristicSearchAbstractPlanGenerator(
        env_models.types,
        env_models.predicates,
        env_models.operators,
        heuristic_name=cfg.heuristic_name,
        seed=problem_id,
    )


def time_pool_generation(cfg: CollectionConfig, problem_id: int) -> float:
    """Wall-clock (s) to draw the capped skeleton pool for one problem.

    The abstract-plan-generation cost the pool-ranking methods share, measured on its
    own by mirroring :func:`collect_episode`'s setup up to — and timing only — the
    ``islice`` pool draw, then stopping before any refinement. Env-agnostic: it
    dispatches the same env models and generator the collection used, so on
    StickButton2D it times the geometry-aware ``AcyclicPlanGenerator`` exactly as
    collected. Used by the §2b wall-clock breakdown
    (``precompute_dd2d_cache._measure_plan_gen``) to supply a per-stratum plan-gen
    constant, the analog of DD2D's ``planner.plan`` timing.
    """
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=problem_id)
        env_models = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        plan_generator = _make_plan_generator(cfg, env_models, obs, problem_id, x0)
        start = time.perf_counter()
        list(
            itertools.islice(
                plan_generator(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg),
                cfg.K_max,
            )
        )
        return time.perf_counter() - start
    finally:
        env.close()


def _make_trajectory_sampler(
    cfg: CollectionConfig, env_models: SesameModels
) -> ParameterizedControllerTrajectorySampler | None:
    """Build the trajectory sampler for this env, or ``None`` where one is not used.

    StickButton2D gets :class:`RecordingSampler` instead of the stock sampler. That is a
    deliberate exception to "wrap kinder, do not reimplement it": upstream's sampler
    computes the achieved abstract state in order to decide accept-or-reject and then
    throws it away behind a payload-free ``TrajectorySamplingFailure``, with no hook to
    read it. Without that state there is no class-2 evidence, and ``coverage``/``waste``
    -- the features v3's margin rests on -- are identically zero on this environment. The
    subclass re-runs the same loop and keeps what it discarded; that labels are unchanged
    is a same-seed differential measurement, in
    ``tests/approaches/spectre/test_stickbutton2d_observational.py``.
    """
    kwargs: dict[str, object] = {
        "controller_generator": RelationalControllerGenerator(env_models.skills),
        "transition_function": env_models.transition_fn,
        "state_abstractor": env_models.state_abstractor,
        "max_trajectory_steps": cfg.max_trajectory_steps,
    }
    if cfg.model_name == _STICK_BUTTON_MODEL_NAME:
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.envs.stickbutton2d.instrumented_refiner import (  # pylint: disable=line-too-long
            RecordingSampler,
        )

        return RecordingSampler(**kwargs)
    if cfg.model_name in (
        _RESTOCK3D_MODEL_NAME,
        _RESTOCK3D_V2_MODEL_NAME,
        _RESTOCK3D_V3_MODEL_NAME,
    ):
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
            RestockRecordingSampler,
        )

        # v3 enables the arm-insertion F3 attribution (Phase 3): a block in (cutoff, clearance]
        # fits under the board but the arm can't insert it; the analytic classifier calls that a
        # provable F3, so the real refiner must too (Gate G1). v1/v2 pass None (block-vs-board only).
        section_cutoffs = None
        if cfg.model_name == _RESTOCK3D_V3_MODEL_NAME:
            from alphatamp.approaches.spectre.envs.restock3d.feasibility_v3 import (
                SHORT_CUTOFF,
                TALL_CUTOFF,
            )

            section_cutoffs = {"section_0": TALL_CUTOFF, "section_1": SHORT_CUTOFF}
        return RestockRecordingSampler(
            sim=_restock_extras["sim"],
            region_infos=_restock_extras["region_infos"],  # type: ignore[arg-type]
            section_height_cutoffs=section_cutoffs,
            **kwargs,  # type: ignore[arg-type]
        )
    return ParameterizedControllerTrajectorySampler(**kwargs)  # type: ignore[arg-type]


def _make_refiner(
    cfg: CollectionConfig,
    obs: dict[str, object] | object,
    trajectory_sampler: ParameterizedControllerTrajectorySampler | None,
    seed: int,
):  # pragma: no cover — return type union widens to whatever the impl supports
    """Build a per-skeleton refiner.

    All supported envs use kinder's backtracking refiner.
    """
    del obs  # kept for signature stability; no env-specific refiner needs it
    assert (
        trajectory_sampler is not None
    ), "kinder envs require a non-None trajectory_sampler"
    return BacktrackingRefiner(
        trajectory_sampler=trajectory_sampler,
        num_sampling_attempts_per_step=cfg.num_sampling_attempts_per_step,
        seed=seed,
    )


def _failure_metadata_fn(
    model_name: str,
) -> Callable[..., list[dict[str, object]]] | None:
    """The env's observation-only failure-harvest fn (SB2D / Restock3D), else None."""
    # pylint: disable=import-outside-toplevel
    if model_name == _STICK_BUTTON_MODEL_NAME:
        from alphatamp.approaches.spectre.envs.stickbutton2d.instrumented_refiner import (
            failure_metadata as sb_fm,
        )  # pylint: disable=line-too-long

        return sb_fm
    if model_name in (
        _RESTOCK3D_MODEL_NAME,
        _RESTOCK3D_V2_MODEL_NAME,
        _RESTOCK3D_V3_MODEL_NAME,
    ):
        from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
            failure_metadata as rs_fm,
        )

        return rs_fm
    return None


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


def _restock3d_analytic_inputs(
    x0: object,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Per-block ``(width, height)`` and ``(x, y)`` for the goal blocks, read from
    ``x0``.

    The pure-geometry inputs to ``feasibility_v3.classify_skeleton``; mirrors
    ``restock3d_v3_plan_attempts._dims_pos``. Constant across a problem's skeleton pool.
    """
    block_dims: dict[str, tuple[float, float]] = {}
    positions: dict[str, tuple[float, float]] = {}
    for o in x0:  # type: ignore[attr-defined]
        if o.name.startswith("obj_goal"):
            block_dims[o.name] = (
                2 * x0.get(o, "half_extent_x"),  # type: ignore[attr-defined]
                2 * x0.get(o, "half_extent_z"),  # type: ignore[attr-defined]
            )
            p = x0.get_object_pose(o.name).position  # type: ignore[attr-defined]
            positions[o.name] = (float(p[0]), float(p[1]))
    return block_dims, positions


def _restock3d_classify(
    action_plan: list[GroundOperator],
    block_dims: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    num_attempts: int,
) -> dict | None:
    """Run the pure-geometry ``feasibility_v3.classify_skeleton`` on one candidate.

    Returns ``None`` if analytically feasible, else the first-violation failure dict
    (byte-compatible in shape with the real failure harvest). Split out so the hybrid-prune
    path can classify once and reuse the dict for the analytic-trusted branch.
    """
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.restock3d.feasibility_v3 import (
        classify_skeleton,
    )

    steps = [(op.name, [p.name for p in op.parameters]) for op in action_plan]
    return classify_skeleton(steps, block_dims, positions, num_attempts=num_attempts)


def _restock3d_outcome_from_fm(
    fm: dict | None,
    seed: int,
    r_cap: float,
) -> tuple[str, float, dict[str, object], int | None]:
    """Analytic label + synthetic wall-clock from an already-computed classify dict ``fm``.

    ``fm is None`` -> a ``success`` costing ``U[0.6,0.8]*r_cap`` (deterministic in ``seed``);
    else a ``fail`` costing the full ``r_cap`` with the first-violation dict under
    ``refiner_metadata["failures"]``. Returns ``(outcome, wall_clock, refiner_metadata,
    stuck_step_index)``.
    """
    refiner_metadata: dict[str, object] = {}
    if fm is None:
        wall_clock = random.Random(seed).uniform(0.6, 0.8) * r_cap
        return "success", wall_clock, refiner_metadata, None
    refiner_metadata["failures"] = [fm]
    return "fail", r_cap, refiner_metadata, int(fm["step_index"])


def _restock3d_analytic_outcome(
    action_plan: list[GroundOperator],
    block_dims: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    num_attempts: int,
    seed: int,
    r_cap: float,
) -> tuple[str, float, dict[str, object], int | None]:
    """Analytic (geometry-only) label + synthetic wall-clock for one skeleton.

    Thin wrapper over ``_restock3d_classify`` + ``_restock3d_outcome_from_fm``. A feasible
    skeleton is a ``success`` costing ``U[0.6,0.8]*r_cap``; an infeasible one is a ``fail``
    costing the full ``r_cap`` with the first-violation dict under ``refiner_metadata[
    "failures"]`` -- indistinguishable in shape from a real run downstream.
    """
    fm = _restock3d_classify(action_plan, block_dims, positions, num_attempts)
    return _restock3d_outcome_from_fm(fm, seed, r_cap)


def _real_refine_candidate(
    cfg: CollectionConfig,
    obs: dict[str, object] | object,
    trajectory_sampler: ParameterizedControllerTrajectorySampler | None,
    bpg: BilevelPlanningGraph,
    x0: object,
    state_plan: list[RelationalAbstractState],
    action_plan: list[GroundOperator],
    seed: int,
) -> tuple[str, float, dict[str, object], int | None, dict[str, str] | None]:
    """One real (motion-planned) refinement of a candidate + observed-failure harvest.

    Lifted from the ``real`` branch of ``collect_episode`` so the hybrid-prune path can
    real-refine a candidate identically. Returns ``(outcome, wall_clock, refiner_metadata,
    stuck_step_index, error_info)``. Observation-only harvest -- every failure field was
    computed by the acceptance check the refiner already ran.
    """
    refiner = _make_refiner(cfg, obs, trajectory_sampler, seed)
    # Rejections accumulate on the sampler, which outlives the candidate loop.
    if hasattr(trajectory_sampler, "clear"):
        trajectory_sampler.clear()  # type: ignore[union-attr]

    refiner_metadata: dict[str, object] = {}
    error_info: dict[str, str] | None = None
    stuck_step_index: int | None = None
    start = time.perf_counter()
    try:
        plan = refiner(x0, state_plan, action_plan, cfg.refinement_timeout_s, bpg)
        outcome = "success" if plan is not None else "fail"
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        outcome = "error"
        error_info = {"cls": type(exc).__name__, "msg": str(exc)}

    fm_fn = _failure_metadata_fn(cfg.model_name) if outcome == "fail" else None
    if fm_fn is not None:
        failures = fm_fn(
            trajectory_sampler,  # type: ignore[arg-type]
            action_plan,
            cfg.num_sampling_attempts_per_step,
            budget_exhausted=(
                time.perf_counter() - start >= cfg.refinement_timeout_s
            ),
        )
        if failures:
            refiner_metadata["failures"] = failures
            stuck_step_index = int(failures[0]["step_index"])  # type: ignore[call-overload]

    wall_clock = time.perf_counter() - start
    return outcome, wall_clock, refiner_metadata, stuck_step_index, error_info


def collect_episode(
    cfg: CollectionConfig,
    problem_id: int,
) -> EpisodeRecord:
    """Collect one episode: pool generation + per-skeleton refinement.

    The pool size is capped by ``cfg.K_max`` via ``itertools.islice``. Each skeleton is
    refined independently with a deterministic seed; outcomes are recorded for every
    skeleton regardless of earlier successes.
    """
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=problem_id)
        env_models = _make_env_models(cfg, env.observation_space, env.action_space)

        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        assert isinstance(goal, RelationalAbstractGoal)

        # Shared per-problem scratchpad; planner and refiner both mutate.
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        plan_generator = _make_plan_generator(cfg, env_models, obs, problem_id, x0)

        # Draw the pool (lazy; islice caps at K_max).
        pool_iter = plan_generator(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg)
        skeleton_pool: list[
            tuple[list[RelationalAbstractState], list[GroundOperator]]
        ] = list(itertools.islice(pool_iter, cfg.K_max))

        # Refiner mode (Restock3D-v3): "analytic" labels every skeleton by pure geometry (no
        # sampler/refiner); "hybrid_prune" classifies all K_max analytically, then real-refines
        # only the analytic-feasible candidates + a 25% audit sample of the analytic-infeasible
        # ones, trusting the analytic label for the rest; "real" (default, all other envs)
        # motion-plans every candidate. The env was reset above for x0/s0/goal/scene_geometry.
        mode = cfg.refiner_mode
        analytic = mode == "analytic"
        hybrid = mode == "hybrid_prune"
        trajectory_sampler = (
            _make_trajectory_sampler(cfg, env_models)
            if mode in ("real", "hybrid_prune")
            else None
        )
        block_dims: dict[str, tuple[float, float]] = {}
        positions: dict[str, tuple[float, float]] = {}
        if analytic or hybrid:
            block_dims, positions = _restock3d_analytic_inputs(x0)

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
            outcome: str
            wall_clock: float
            error_info: dict[str, str] | None = None
            refiner_metadata: dict[str, object] = {}
            stuck_step_index: int | None = None
            label_source: str | None = None

            if analytic:
                outcome, wall_clock, refiner_metadata, stuck_step_index = (
                    _restock3d_analytic_outcome(
                        action_plan,
                        block_dims,
                        positions,
                        cfg.num_sampling_attempts_per_step,
                        seed,
                        cfg.refinement_timeout_s,
                    )
                )
                label_source = "analytic"
            elif hybrid:
                # Classify once; real-refine iff analytic-feasible OR drawn into the 25% audit
                # of analytic-infeasible candidates; else trust the analytic label. The audit
                # draw is keyed on a string (deterministic across processes -- a tuple's hash
                # would be PYTHONHASHSEED-dependent), independent of the refiner seed.
                fm = _restock3d_classify(
                    action_plan,
                    block_dims,
                    positions,
                    cfg.num_sampling_attempts_per_step,
                )
                analytic_feasible = fm is None
                audit = (
                    random.Random(
                        f"prune_audit:{cfg.refinement_seed_rule}:{problem_id}:{idx}"
                    ).random()
                    < _HYBRID_AUDIT_FRAC
                )
                if analytic_feasible or audit:
                    (
                        outcome,
                        wall_clock,
                        refiner_metadata,
                        stuck_step_index,
                        error_info,
                    ) = _real_refine_candidate(
                        cfg,
                        obs,
                        trajectory_sampler,
                        bpg,
                        x0,
                        state_plan,
                        action_plan,
                        seed,
                    )
                    label_source = "real"
                    refiner_metadata["prune_reason"] = (
                        "analytic_feasible" if analytic_feasible else "audit_sample"
                    )
                else:
                    outcome, wall_clock, refiner_metadata, stuck_step_index = (
                        _restock3d_outcome_from_fm(fm, seed, cfg.refinement_timeout_s)
                    )
                    label_source = "analytic"
                    refiner_metadata["prune_reason"] = "analytic_trusted"
            else:
                (
                    outcome,
                    wall_clock,
                    refiner_metadata,
                    stuck_step_index,
                    error_info,
                ) = _real_refine_candidate(
                    cfg,
                    obs,
                    trajectory_sampler,
                    bpg,
                    x0,
                    state_plan,
                    action_plan,
                    seed,
                )
                label_source = "real"

            total_wall_clock += wall_clock

            if outcome == "success" and first_success_idx is None:
                first_success_idx = idx

            outcome_records.append(
                OutcomeRecord(
                    skeleton_idx=idx,
                    outcome=outcome,  # type: ignore[arg-type]
                    refinement_wall_clock_s=wall_clock,
                    refinement_seed=seed,
                    stuck_step_index=stuck_step_index,
                    error_info=error_info,
                    refiner_metadata=refiner_metadata,
                    label_source=label_source,  # type: ignore[arg-type]
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

        # Kinder collections have no per-episode scene latent.
        scene_latent: dict[str, str] | None = None

        # Audit trail for the pooled StickButton2D collection, where the stratum is the
        # button count and is otherwise recoverable only by decoding the problem id
        # arithmetically (`envs/stickbutton2d/strata.py`). Recording it independently is
        # what makes a broken encoding detectable instead of silently mislabelling every
        # stratum. Never a model input -- nothing in the tensorizer reads provenance.
        gen_params: dict[str, object] | None = None
        if cfg.model_name == _STICK_BUTTON_MODEL_NAME:
            # pylint: disable=import-outside-toplevel
            from alphatamp.approaches.spectre.envs.stickbutton2d.strata import (
                slot_of,
            )

            n_buttons = int(cfg.model_kwargs.get("num_buttons", 0))
            gen_params = {
                "num_buttons": n_buttons,
                "stratum": slot_of(n_buttons),
                "split": cfg.split,
                "acyclic_pool": True,
            }
        elif cfg.model_name in (
            _RESTOCK3D_MODEL_NAME,
            _RESTOCK3D_V2_MODEL_NAME,
            _RESTOCK3D_V3_MODEL_NAME,
        ):
            gen_params = {
                "stratum": int(cfg.model_kwargs["stratum"]),
                "split": cfg.split,
            }

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
            scene_latent=scene_latent,
            gen_params=gen_params,
        )

        object_registry = _collect_all_objects(s0, skeleton_pool, goal.atoms)

        # Ground-truth geometry. Required by v3 (`train._trainable` drops episodes
        # without it, silently) and by the later PIGINet / VLMPlan comparators. Only
        # StickButton2D has a builder; the other kinder envs stay abstract-only
        # as before.
        scene_geometry = None
        if cfg.model_name == _STICK_BUTTON_MODEL_NAME:
            # pylint: disable=import-outside-toplevel
            from alphatamp.approaches.spectre.envs.stickbutton2d.scene_geometry import (
                build_scene_geometry,
            )

            scene_geometry = build_scene_geometry(x0)
        elif cfg.model_name in (_RESTOCK3D_V2_MODEL_NAME, _RESTOCK3D_V3_MODEL_NAME):
            # pylint: disable=import-outside-toplevel
            from alphatamp.approaches.spectre.envs.restock3d.scene_geometry import (
                build_scene_geometry as build_restock3d_scene_geometry,
            )

            scene_geometry = build_restock3d_scene_geometry(x0)

        return EpisodeRecord(
            provenance=provenance,
            initial_abstract_state=s0,
            goal_atoms=frozenset(goal.atoms),
            object_registry=object_registry,
            skeleton_pool=tuple(skeleton_records),
            outcomes=tuple(outcome_records),
            summary=summary,
            scene_geometry=scene_geometry,
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

    Returns ``(problem_id, path, error_message)``. Exceptions are captured so one bad
    problem doesn't kill a worker pool. Lives in the package (not in the Hydra
    entrypoint script) so it has a stable importable qualname under ``multiprocessing``
    ``spawn`` start method.
    """
    try:
        path = collect_and_save(cfg, data_root, problem_id)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        return problem_id, None, f"{type(exc).__name__}: {exc}"
    return problem_id, path, None
