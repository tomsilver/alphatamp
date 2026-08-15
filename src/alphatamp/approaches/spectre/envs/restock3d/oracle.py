"""Privileged oracle solver for the **kinematic** Restock3D (no-clutter v1).

Constructs a correct skeleton directly from region geometry (bipartite assignment + FFD
order — no relocation, since v1 has no clutter) and refines it through the **standard
refiner and samplers**, so its wall-clock is the same quantity every method pays.
Primary deliverable: **budget calibration** — the per-candidate refinement cap is set
from measured *feasible*-refinement time, which only a solver that reliably produces
feasible skeletons can sample. Secondary: per-instance feasibility certification and the
P5 oracle-FP reference.

Boundary (oracle_solver.md §1.2): oracle plans feed budgets / certification only — never
skeleton pool membership or training labels beyond feasibility.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import kinder  # noqa: E402  (after package imports; keeps BLAS shim ordering upstream)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundOperator, Object

from alphatamp.approaches.spectre.collect import (
    _make_env_models,
    _make_refiner,
    _make_trajectory_sampler,
    _refinement_seed,
    _restock_extras,
)
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs

from .eager_tables import build_tables, is_feasible_skeleton
from .place_controller import RegionType
from .region_geometry import RegionInfo

_TALL_PREFIX = "block_goal"
_CUBE_PREFIX = "cube_goal"


@dataclass(frozen=True)
class OracleResult:
    """Per-instance oracle outcome (feasibility certificate + timing)."""

    stratum: int
    problem_id: int
    certified_feasible: bool
    t_oracle: Optional[float]  # wall-clock (s) of the successful refiner call
    n_refiner_calls: int
    plan_len: int


def solve_assignment(
    region_infos: dict[str, RegionInfo], goal_object_names: Iterable[str]
) -> list[tuple[str, str]]:
    """Bipartite object→region assignment: talls to distinct tall regions, cubes to
    distinct remaining regions (short first, overflow to leftover tall).

    Central-first per section (shelf edges place unreliably), mirroring the demo
    assignment. Returns (obj, region) pairs, talls first.
    """
    tall_regions = [n for n, i in region_infos.items() if i.shelf == 0]
    short_regions = [n for n, i in region_infos.items() if i.shelf == 1]
    blocks = sorted(n for n in goal_object_names if n.startswith(_TALL_PREFIX))
    cubes = sorted(n for n in goal_object_names if n.startswith(_CUBE_PREFIX))
    center_x = statistics.median([i.center_xy[0] for i in region_infos.values()])

    def central(names: list[str], n: int) -> list[str]:
        ordered = sorted(
            names, key=lambda r: abs(region_infos[r].center_xy[0] - center_x)
        )
        return sorted(ordered[:n], key=lambda r: region_infos[r].center_xy[0])

    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for b, reg in zip(blocks, central(tall_regions, len(blocks))):
        pairs.append((b, reg))
        used.add(reg)
    n_short = min(len(cubes), len(short_regions))
    for c, reg in zip(cubes[:n_short], central(short_regions, n_short)):
        pairs.append((c, reg))
        used.add(reg)
    leftover_tall = [t for t in tall_regions if t not in used]
    for c, reg in zip(cubes[n_short:], central(leftover_tall, len(cubes) - n_short)):
        pairs.append((c, reg))
        used.add(reg)
    return pairs


def build_skeleton(
    x0: object,
    s0: RelationalAbstractState,
    assignment: list[tuple[str, str]],
    lifted_ops: dict[str, object],
) -> tuple[list[RelationalAbstractState], list[GroundOperator]]:
    """Ground pick+place for each (obj, region) pair (FFD order) and STRIPS-progress the
    abstract state to the interleaved ``(state_plan, action_plan)`` the refiner
    consumes."""
    pick = lifted_ops["pick"]
    place = lifted_ops["place"]
    robot = x0.get_object_from_name("robot")  # type: ignore[attr-defined]
    state_plan: list[RelationalAbstractState] = [s0]
    action_plan: list[GroundOperator] = []
    state = s0
    for obj_name, region_name in assignment:
        obj = x0.get_object_from_name(obj_name)  # type: ignore[attr-defined]
        region = Object(region_name, RegionType)
        for op in (
            pick.ground((robot, obj)),  # type: ignore[attr-defined]
            place.ground((robot, obj, region)),  # type: ignore[attr-defined]
        ):
            ns_atoms = (state.atoms - op.delete_effects) | op.add_effects
            state = RelationalAbstractState(ns_atoms, state.objects)
            action_plan.append(op)
            state_plan.append(state)
    return state_plan, action_plan


def refine_oracle(
    cfg: CollectionConfig,
    problem_id: int,
    budget_s: float = 300.0,
    max_retries: int = 8,
) -> OracleResult:
    """Build the oracle skeleton for ``problem_id`` and refine it (retry fresh seeds
    until success or budget).

    Records the successful call's wall-clock as ``t_oracle``.
    """
    register_extra_envs()
    stratum = int(cfg.model_kwargs["stratum"])
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=problem_id)
        env_models = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        region_infos = _restock_extras["region_infos"]
        goal_names = _restock_extras["goal_names"]
        lifted = {op.name: op for op in env_models.operators}
        assignment = solve_assignment(region_infos, goal_names)  # type: ignore[arg-type]
        skeleton = build_skeleton(x0, s0, assignment, lifted)  # type: ignore[arg-type]
        state_plan, action_plan = skeleton
        tables = build_tables(region_infos, goal_names)  # type: ignore[arg-type]
        assert is_feasible_skeleton(
            action_plan, tables
        ), "oracle constructed an abstractly-infeasible skeleton"

        t0 = time.perf_counter()
        for attempt in range(max_retries):
            remaining = budget_s - (time.perf_counter() - t0)
            if remaining <= 0:
                break
            sampler = _make_trajectory_sampler(cfg, env_models)
            if hasattr(sampler, "clear"):
                sampler.clear()  # type: ignore[union-attr]
            seed = _refinement_seed(cfg.refinement_seed_rule, problem_id, attempt)
            refiner = _make_refiner(cfg, obs, sampler, seed)
            bpg: BilevelPlanningGraph = BilevelPlanningGraph()
            bpg.add_abstract_state_node(s0)
            bpg.add_state_node(x0)
            bpg.add_state_abstractor_edge(x0, s0)
            start = time.perf_counter()
            try:
                plan = refiner(x0, state_plan, action_plan, remaining, bpg)
            except BaseException:  # noqa: BLE001
                plan = None
            if plan is not None:
                return OracleResult(
                    stratum,
                    problem_id,
                    True,
                    time.perf_counter() - start,
                    attempt + 1,
                    len(action_plan),
                )
        return OracleResult(
            stratum, problem_id, False, None, max_retries, len(action_plan)
        )
    finally:
        env.close()
