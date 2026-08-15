"""Oracle solver tests: assignment/skeleton construction (fast) + real certification
(slow).

The oracle builds a feasible-by-construction skeleton (bipartite assignment + FFD) and
refines it through the standard refiner. See ``envs/restock3d/oracle.py``.
"""

from __future__ import annotations

import kinder
import pytest

from alphatamp.approaches.spectre.collect import _make_env_models, _restock_extras
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import strata as S
from alphatamp.approaches.spectre.envs.restock3d.eager_tables import (
    build_tables,
    is_feasible_skeleton,
)
from alphatamp.approaches.spectre.envs.restock3d.oracle import (
    build_skeleton,
    refine_oracle,
    solve_assignment,
)


def _cfg(stratum: int) -> CollectionConfig:
    start = S.problem_id("train", stratum, 0)
    return CollectionConfig(
        env_id=f"spectre/Restock3D-r{stratum}-v0",
        env_variant="restock3d_v1",
        model_name="restock3d",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=start,
        problem_seed_end=start + 1,
        K_max=1,
        num_sampling_attempts_per_step=10,
        refinement_timeout_s=200.0,
        max_trajectory_steps=500,
    )


@pytest.mark.parametrize("stratum", [0, 1, 2, 3])
def test_oracle_builds_feasible_skeleton(stratum: int) -> None:
    """The constructed skeleton is abstractly feasible (no tall→short, no region
    reused)."""
    cfg = _cfg(stratum)
    pid = S.problem_id("train", stratum, 0)
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=pid)
        env_models = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = env_models.observation_to_state(obs)
        s0 = env_models.state_abstractor(x0)
        region_infos = _restock_extras["region_infos"]
        goal_names: list[str] = _restock_extras["goal_names"]  # type: ignore[assignment]
        lifted = {op.name: op for op in env_models.operators}
        assignment = solve_assignment(region_infos, goal_names)  # type: ignore[arg-type]
        # Every goal object is assigned exactly once, to distinct regions.
        assert len(assignment) == len(goal_names)
        assert len({r for _, r in assignment}) == len(assignment)
        skeleton = build_skeleton(x0, s0, assignment, lifted)  # type: ignore[arg-type]
        _, action_plan = skeleton
        tables = build_tables(region_infos, goal_names)  # type: ignore[arg-type]
        assert is_feasible_skeleton(action_plan, tables)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("stratum", [0, 2])
def test_oracle_certifies(stratum: int) -> None:
    """The oracle certifies a real problem by refining its skeleton through the standard
    refiner."""
    cfg = _cfg(stratum)
    pid = S.problem_id("train", stratum, 0)
    result = refine_oracle(cfg, pid, budget_s=200.0, max_retries=8)
    assert result.certified_feasible
    assert result.t_oracle is not None and result.t_oracle > 0
