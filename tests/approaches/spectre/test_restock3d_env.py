"""Slow MuJoCo integration test for Restock3D: generate a scene, enumerate the pool, and confirm
the geometric feasibility gate classifies candidates (F2 over-assign, F3 height, feasible).

Skipped by default (needs kinder/MuJoCo + ikfast); run with ``-m slow``.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import pathlib
import tempfile

import pytest

pytestmark = pytest.mark.slow


def _setup_3d_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.pop("PYOPENGL_PLATFORM", None)
    blas = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
    os.environ.setdefault("LAPACK_DIR", blas)
    os.environ.setdefault("BLAS_DIR", blas)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    pathlib.Path(blas).mkdir(parents=True, exist_ok=True)
    for archive, (subdir, pat) in {
        "liblapack.a": ("lapack", "liblapack.so.3*"),
        "libblas.a": ("blas", "libblas.so.3*"),
    }.items():
        link = pathlib.Path(blas) / archive
        if link.exists() or link.is_symlink():
            continue
        root = "/usr/lib/x86_64-linux-gnu"
        cands = sorted(
            glob.glob(os.path.join(root, subdir, pat))
            + glob.glob(os.path.join(root, pat))
        )
        real = next((c for c in cands if os.path.isfile(c)), None)
        if real is not None:
            link.symlink_to(real)


def test_r2_scene_gate_classifies_f2_f3_and_feasible() -> None:
    pytest.importorskip("kinder")
    pytest.importorskip("mujoco")
    _setup_3d_env()
    import gymnasium
    import kinder
    from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
        AbstractPlanGenerator,
    )
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    from alphatamp.approaches.spectre.envs.restock3d import generator as gen_mod
    from alphatamp.approaches.spectre.envs.restock3d.models import (
        CubeType,
        create_restock3d_models,
    )
    from alphatamp.approaches.spectre.envs.restock3d.refine import (
        evaluate_skeleton,
        object_dims,
    )
    from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
        load_region_infos,
    )

    # r2 has tall blocks + a short cell, so both F2 (over-assign) and F3 (height) can occur.
    spec = gen_mod.build_spec(seed=0, stratum=2)
    cfg = gen_mod.build_task_config(spec)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        task_path = tf.name
    eid = "kinder/Restock3D-test-r2-v0"
    if eid not in gymnasium.registry:
        gymnasium.register(
            id=eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={"task_config_path": task_path, "scene_render_camera": "task_view"},
        )
    env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    try:
        obs, _ = env.reset(seed=0)
        n_obj = len(cfg["goal_objects"])
        models = create_restock3d_models(
            env.observation_space, env.action_space, task_path, num_objects=n_obj
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        region_infos = load_region_infos(task_path, x0)
        dims = object_dims(x0, CubeType)

        # every goal object starts OnFloor; none stored
        assert all(a.predicate.name != "Stored" for a in s0.atoms)
        assert sum(a.predicate.name == "OnFloor" for a in s0.atoms) == n_obj
        assert len(goal.atoms) == n_obj

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen: AbstractPlanGenerator = RelationalHeuristicSearchAbstractPlanGenerator(
            models.types, models.predicates, models.operators, "hff", seed=0
        )
        pool = list(itertools.islice(gen(x0, s0, goal, 30.0, bpg), 200))
    finally:
        env.close()
        os.unlink(task_path)

    assert len(pool) > 20, "pool should enumerate many diverse assignments"
    families = {"feasible": 0, "F2": 0, "F3": 0}
    f2_culprit_seen = f3_proves_failure = False
    for state_plan, action_plan in pool:
        v = evaluate_skeleton(state_plan, action_plan, region_infos, dims)
        if v.feasible:
            families["feasible"] += 1
        else:
            assert v.family is not None
            families[v.family] += 1
            if v.family == "F2":
                assert (
                    v.failure is not None and v.failure["culprits"]
                ), "F2 names culprits"
                f2_culprit_seen = True
            if v.family == "F3":
                assert (
                    v.failure is not None and not v.failure["culprits"]
                ), "F3 culprit-free"
                assert v.failure["exhausted"] and not v.failure["budget_exhausted"]
                f3_proves_failure = True

    assert (
        families["feasible"] >= 1
    ), "r2 must be solvable (a feasible candidate exists)"
    assert (
        families["F2"] >= 1 and f2_culprit_seen
    ), "r2 pool should contain F2 over-assignments"
    assert (
        families["F3"] >= 1 and f3_proves_failure
    ), "r2 pool should contain F3 height mismatches"
