"""Render a MuJoCo demo of Restock3D: pick floor cubes and store them into shelf regions.

Plans the first feasible restock skeleton (the geometric gate rules out over-assign / height
violations) and executes its controllers LIVE in the gym TidyBot3DEnv (continuous env.step ->
faithful render), then writes an mp4 to ``envs/restock3d/demos/``. Uses the physics place
controller (geometric_place is a data-collection device, not used here) on a short-cell scene —
the currently-proven placement geometry.

    python experiments/spectre/restock3d_demos.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault("LAPACK_DIR", os.path.expanduser("~/.cache/alphatamp_ikfast_blas"))
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

from alphatamp.approaches.spectre.envs.restock3d.models import (  # noqa: E402
    CubeType,
    create_restock3d_models,
)
from alphatamp.approaches.spectre.envs.restock3d.refine import (  # noqa: E402
    evaluate_skeleton,
    object_dims,
)
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (  # noqa: E402
    load_region_infos,
)

_DEMOS = os.path.join(
    _SRC, "alphatamp", "approaches", "spectre", "envs", "restock3d", "demos"
)

# A short-cell restock scene (shelf 1, the proven placement geometry): 2 floor cubes, 3 central
# single-object regions. The planner may over-assign, but the first *feasible* skeleton stores
# each cube in a distinct region.
_SHORT_SURFACE_Z, _SHORT_CLEARANCE = 0.537, 0.241
_DEMO_SCENE = {
    "description": "Restock3D demo: store 2 floor cubes into shelf regions",
    "robots": {"tidybot": {"robot": {}}},
    "scene": "lab2",
    "fixtures": {
        "cupboard": {
            "cupboard_1": {
                "length": 0.60198,
                "depth": 0.254,
                "shelf_heights": [0.254, 0.254, 0.254],
                "shelf_partitions": [[], [], []],
                "shelf_thickness": 0.0127,
                "side_and_back_open": False,
            }
        }
    },
    "regions": {
        "ground_cupboard_init_region": {
            "target": "ground",
            "ranges": [[1.5, 0.0, 1.5, 0.0]],
            "yaw_ranges": [[90, 90]],
        },
        "robot_0_task_init_region": {
            "target": "ground",
            "ranges": [[-0.1, -0.1, 0.1, 0.1]],
            "yaw_ranges": [[0, 0]],
        },
        "cube_goal1_init_region": {
            "target": "ground",
            "ranges": [[0.5, -0.17, 0.56, -0.13]],
            "yaw_ranges": [[0, 0]],
        },
        "cube_goal2_init_region": {
            "target": "ground",
            "ranges": [[0.5, 0.13, 0.56, 0.17]],
            "yaw_ranges": [[0, 0]],
        },
        "region_2_1": {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [[-0.08, 0.085, 0.0, -0.02, 0.105, 0.03]],
            "rgba": [0.0, 1.0, 1.0, 0.3],
            "yaw_ranges": [[0, 0]],
        },
        "region_2_2": {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [[0.02, 0.085, 0.0, 0.08, 0.105, 0.03]],
            "rgba": [0.0, 1.0, 1.0, 0.3],
            "yaw_ranges": [[0, 0]],
        },
        "region_2_3": {
            "target": "cupboard_1",
            "shelf": 2,
            "ranges": [[-0.14, 0.085, 0.0, -0.08, 0.105, 0.03]],
            "rgba": [0.0, 1.0, 1.0, 0.3],
            "yaw_ranges": [[0, 0]],
        },
    },
    "region_meta": {
        "region_2_1": {"cell_clearance": _SHORT_CLEARANCE, "surface_z": 0.55},
        "region_2_2": {"cell_clearance": _SHORT_CLEARANCE, "surface_z": 0.55},
        "region_2_3": {"cell_clearance": _SHORT_CLEARANCE, "surface_z": 0.55},
    },
    "objects": {
        "cube": {
            "cube_goal1": {"size": 0.02, "rgba": [0.1, 0.5, 0.1, 1], "mass": 0.02},
            "cube_goal2": {"size": 0.02, "rgba": [0.1, 0.3, 0.6, 1], "mass": 0.02},
        }
    },
    "cameras": {
        "task_view": {
            "position": [-1, 1, 2],
            "lookat": [2, 0, 0],
            "fovy": 42,
            "resolution": [640, 480],
        }
    },
    "initial_state": [
        ["on", "cupboard_1", "ground_cupboard_init_region"],
        ["on", "cube_goal1", "cube_goal1_init_region"],
        ["on", "cube_goal2", "cube_goal2_init_region"],
        ["on", "robot", "robot_0_task_init_region"],
    ],
    "goal_objects": ["cube_goal1", "cube_goal2"],
    "goal_state": [],
}


def render_demo(task_path: str, out_path: str) -> bool:
    """Refine the first feasible restock skeleton (physics pick + geometric place) and render its
    states via ``set_state`` (drift-free). Physics picks show the real arm motion; the geometric
    places store each cube deterministically (the flaky physics-insertion MP is a data device,
    not shown). Returns whether the goal is reached.
    """
    # pylint: disable=import-outside-toplevel
    from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner

    from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
        make_recording_sampler,
    )

    eid = "kinder/Restock3D-demo-v0"
    if eid not in gymnasium.registry:
        gymnasium.register(
            id=eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={"task_config_path": task_path, "scene_render_camera": "task_view"},
        )
    env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    obs, _ = env.reset(seed=0)
    with open(task_path, encoding="utf-8") as f:
        n_obj = len(json.load(f)["goal_objects"])
    models = create_restock3d_models(
        env.observation_space, env.action_space, task_path, num_objects=n_obj
    )
    x0 = models.observation_to_state(obs)
    s0 = models.state_abstractor(x0)
    goal = models.goal_deriver(x0)
    region_infos = load_region_infos(task_path, x0)
    dims = object_dims(x0, CubeType)
    bpg = BilevelPlanningGraph()
    bpg.add_abstract_state_node(s0)
    bpg.add_state_node(x0)
    bpg.add_state_abstractor_edge(x0, s0)
    gen = RelationalHeuristicSearchAbstractPlanGenerator(
        models.types, models.predicates, models.operators, "hff", seed=0
    )
    pool = list(itertools.islice(gen(x0, s0, goal, 30.0, bpg), 40))
    chosen = None
    for sp, ap in pool:
        if evaluate_skeleton(sp, ap, region_infos, dims).feasible:
            chosen = (sp, ap)
            break
    assert chosen is not None, "no feasible skeleton in the demo pool"
    print("[restock3d] plan: " + " -> ".join(op.name for op in chosen[1]))

    sampler = make_recording_sampler(
        controller_generator=RelationalControllerGenerator(models.skills),
        transition_function=models.transition_fn,
        state_abstractor=models.state_abstractor,
        max_trajectory_steps=500,
        region_infos=region_infos,
        robot_name="robot",
        geometric_place=True,
    )
    refiner = BacktrackingRefiner(
        trajectory_sampler=sampler, num_sampling_attempts_per_step=8, seed=0
    )
    plan = refiner(x0, chosen[0], chosen[1], 90.0, bpg)
    assert plan is not None, "feasible skeleton failed to refine (physics pick)"

    # Render the refined states via set_state (drift-free; cf. shelf3d_collect._render_plan_video).
    render_env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    render_env.reset(seed=0)
    space = render_env.unwrapped.observation_space
    frames = []
    for state in plan.states:
        render_env.unwrapped.set_state(space.vectorize(state))  # type: ignore[attr-defined]
        frames.append(render_env.render())
    reached = goal.check_abstract_state(models.state_abstractor(plan.states[-1]))
    imageio.mimsave(out_path, frames, fps=20)
    print(f"[restock3d] wrote {out_path} ({len(frames)} frames); goal_reached={reached}")
    render_env.close()
    env.close()
    return bool(reached)


def main() -> None:
    os.makedirs(_DEMOS, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(_DEMO_SCENE, tf)
        task_path = tf.name
    try:
        ok = render_demo(task_path, os.path.join(_DEMOS, "demo_restock.mp4"))
    finally:
        os.unlink(task_path)
    print(f"\n=== demo: {'goal reached' if ok else 'GOAL NOT REACHED'} ===")


if __name__ == "__main__":
    main()
