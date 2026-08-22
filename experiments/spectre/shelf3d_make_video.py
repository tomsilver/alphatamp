"""Render the ShelfObstruct3D clearing episode to mp4 by executing the refined
controllers LIVE in the gym env (continuous env.step, no set_state -> faithful, no
phantom cubes)."""

import itertools
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.setdefault(
    "LAPACK_DIR", os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
)
os.environ.setdefault("BLAS_DIR", os.path.expanduser("~/.cache/alphatamp_ikfast_blas"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import gymnasium
import imageio
import kinder
import numpy as np
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.utils import RelationalControllerGenerator

from alphatamp.approaches.spectre.envs.shelf3d.models import create_obstruction_models

TASK = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/josephxu/Projects/alphatamp/src/alphatamp/approaches/spectre/envs/shelf3d/tasks/ShelfObstruct3D-o1.json"
)
OUT = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "/home/josephxu/Projects/alphatamp/src/alphatamp/approaches/spectre/envs/shelf3d/demo_o1_clearing.mp4"
)
EID = f"kinder/ShelfObstruct3D-vid-{os.path.basename(TASK).split('.')[0]}-v0"
if EID not in gymnasium.registry:
    gymnasium.register(
        id=EID,
        entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
        kwargs={"task_config_path": TASK, "scene_render_camera": "task_view"},
    )
env = kinder.make(EID, render_mode="rgb_array", allow_state_access=True)
obs, _ = env.reset(seed=0)
space = env.observation_space
models = create_obstruction_models(
    env.observation_space, env.action_space, TASK, num_objects=2
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
sp, ap = list(itertools.islice(gen(x, s0, goal, 30.0, bpg), 1))[0]
print(
    "plan:",
    " -> ".join(f"{op.name}({','.join(o.name for o in op.parameters)})" for op in ap),
)
gen_ctrl = RelationalControllerGenerator(models.skills)
rng = np.random.default_rng(0)
frames = [env.render()]
for a in ap:
    ctrl = gen_ctrl(a)
    ctrl.reset(x, ctrl.sample_parameters(x, rng))
    k = 0
    for _ in range(6000):
        if ctrl.terminated():
            break
        u = ctrl.step()
        obs, *_ = env.step(u)
        x = models.observation_to_state(obs)
        ctrl.observe(x)
        k += 1
        if k % 3 == 0:
            frames.append(env.render())
    frames.append(env.render())
imageio.mimsave(OUT, frames, fps=30)
print(f"wrote {OUT} ({len(frames)} frames)")
env.close()
print("VIDEO DONE")
