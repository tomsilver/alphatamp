"""Render demo videos of ShelfObstruct3D with successful clearing plans.

For each demo scene (the canonical o1 task, plus generated 1-/2-target scenes) this
plans the first pooled clearing skeleton, executes its controllers LIVE in the gym env
(continuous env.step -> faithful render, no set_state phantoms), verifies the goal is
reached, and writes an mp4. Videos go to ``envs/shelf3d/demos/``.

python experiments/spectre/shelf3d_demos.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault(
    "LAPACK_DIR", os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
)
os.environ.setdefault("BLAS_DIR", os.path.expanduser("~/.cache/alphatamp_ikfast_blas"))
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, _SRC)

import gymnasium  # noqa: E402
import imageio  # noqa: E402
import kinder  # noqa: E402
import numpy as np  # noqa: E402
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (  # noqa: E402
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph  # noqa: E402
from bilevel_planning.utils import RelationalControllerGenerator  # noqa: E402

from alphatamp.approaches.spectre.envs.shelf3d import generator as G  # noqa: E402
from alphatamp.approaches.spectre.envs.shelf3d.models import (  # noqa: E402
    create_obstruction_models,
)

_TASKS = os.path.join(
    _SRC, "alphatamp", "approaches", "spectre", "envs", "shelf3d", "tasks"
)
_DEMOS = os.path.join(
    _SRC, "alphatamp", "approaches", "spectre", "envs", "shelf3d", "demos"
)


def _n_cubes(task_path: str) -> int:
    with open(task_path, encoding="utf-8") as f:
        return len(json.load(f)["objects"]["cube"])


def render_demo(name: str, task_path: str, out_path: str) -> bool:
    """Plan + live-execute the first clearing skeleton; render to mp4.

    Returns goal-reached.
    """
    eid = f"kinder/ShelfDemo-{name}-v0"
    if eid not in gymnasium.registry:
        gymnasium.register(
            id=eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={"task_config_path": task_path, "scene_render_camera": "task_view"},
        )
    env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    obs, _ = env.reset(seed=0)
    models = create_obstruction_models(
        env.observation_space,
        env.action_space,
        task_path,
        num_objects=_n_cubes(task_path),
    )
    x = models.observation_to_state(obs)
    s0 = models.state_abstractor(x)
    goal = models.goal_deriver(x)
    bpg = BilevelPlanningGraph()
    bpg.add_abstract_state_node(s0)
    bpg.add_state_node(x)
    bpg.add_state_abstractor_edge(x, s0)
    gen = RelationalHeuristicSearchAbstractPlanGenerator(
        models.types, models.predicates, models.operators, "hff", seed=0
    )
    _, action_plan = list(itertools.islice(gen(x, s0, goal, 30.0, bpg), 1))[0]
    print(f"[{name}] plan: " + " -> ".join(op.name for op in action_plan))
    gen_ctrl = RelationalControllerGenerator(models.skills)
    rng = np.random.default_rng(0)
    frames = [env.render()]
    for a in action_plan:
        ctrl = gen_ctrl(a)
        ctrl.reset(x, ctrl.sample_parameters(x, rng))
        k = 0
        for _ in range(6000):
            if ctrl.terminated():
                break
            obs, *_ = env.step(ctrl.step())
            x = models.observation_to_state(obs)
            ctrl.observe(x)
            k += 1
            if k % 3 == 0:
                frames.append(env.render())
        frames.append(env.render())
    reached = goal.check_abstract_state(models.state_abstractor(x))
    imageio.mimsave(out_path, frames, fps=30)
    print(f"[{name}] wrote {out_path} ({len(frames)} frames); goal_reached={reached}")
    env.close()
    return bool(reached)


def _write_generated(name: str, spec: G.ShelfObstructSpec) -> str:
    path = os.path.join(_TASKS, f"_demo_{name}.json")
    return G.write_task(G.build_task_config(spec), path)


def main() -> None:
    os.makedirs(_DEMOS, exist_ok=True)
    scenes: list[tuple[str, str]] = []
    # 1. The canonical hand-authored scene: 1 blocker relocated, 1 ground target placed.
    scenes.append(("o1_clearing", os.path.join(_TASKS, "ShelfObstruct3D-o1.json")))
    # 2. A generated 1-target scene with three free regions (richer relocation pool). Positions
    #    are set directly to keep the blocker central and graspable (build_spec centres the row,
    #    which pushes a lone target to the ungraspable edge).
    scenes.append(
        (
            "gen_1target",
            _write_generated(
                "1target",
                G.ShelfObstructSpec(
                    target_region_y=[-0.10],
                    free_region_y=[-0.24, 0.06, 0.20],
                    obstructed_free=[],
                    obstructor_y=[],
                ),
            ),
        )
    )
    # 3. A generated 2-target scene: two blockers relocated, two targets placed (8-step plan).
    #    Positions kept in the reliable lateral grasp/place window.
    scenes.append(
        (
            "gen_2target",
            _write_generated(
                "2target",
                G.ShelfObstructSpec(
                    target_region_y=[-0.10, 0.06],
                    free_region_y=[-0.24, 0.20],
                    obstructed_free=[],
                    obstructor_y=[],
                ),
            ),
        )
    )

    results = []
    for name, task in scenes:
        try:
            ok = render_demo(name, task, os.path.join(_DEMOS, f"demo_{name}.mp4"))
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            ok = False
        results.append((name, ok))
        if task.startswith(os.path.join(_TASKS, "_demo_")):
            os.remove(task)
    print("\n=== demo summary ===")
    for name, ok in results:
        print(f"  {name}: {'goal reached' if ok else 'GOAL NOT REACHED'}")


if __name__ == "__main__":
    main()
