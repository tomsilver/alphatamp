"""Render successful MuJoCo demo videos of Restock3D — one per stratum (r0, r1, r2).

For each stratum, generate the actual generator scene, plan the first feasible restock skeleton
(the geometric gate rules out over-assign / height violations), refine it with physics picks +
deterministic geometric places, and render the refined states via ``set_state`` (drift-free) to
``envs/restock3d/demos/demo_r{stratum}.mp4``. Small cubes are physics-picked (real arm motion off
the floor); tall blocks are geometrically picked (kinder's small-cube pick controller cannot grasp
a ~0.29 m block); every place is geometric (the flaky physics-insertion MP is a data device, not
shown — DD-6). A seed-retry loop takes the first seed whose feasible skeleton refines to the goal.

Complexity rises with stratum: r0 = 3 small cubes, r1 = 5 small cubes (short-cell capacity
pressure), r2 = 3 small cubes + 1 tall block (a tall block routed to the tall cell).

    python experiments/spectre/restock3d_demos.py
    python experiments/spectre/restock3d_demos.py --strata 0,1,2 --max-seeds 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile

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
from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (  # noqa: E402
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (  # noqa: E402
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph  # noqa: E402
from bilevel_planning.refiners.backtracking_refiner import (  # noqa: E402
    BacktrackingRefiner,
)
from bilevel_planning.utils import RelationalControllerGenerator  # noqa: E402

from alphatamp.approaches.spectre.envs.restock3d import generator  # noqa: E402
from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (  # noqa: E402
    make_recording_sampler,
)
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
_REFINE_BUDGET_S = 180.0


def _plan_str(action_plan) -> str:
    def one(op) -> str:
        if op.name == "place":
            return f"place({op.parameters[1].name}->{op.parameters[2].name})"
        return f"pick({op.parameters[1].name})"

    return " ".join(one(op) for op in action_plan)


def _refine_and_render(stratum: int, seed: int, out_path: str) -> tuple[bool, str]:
    """Generate a (stratum, seed) scene, refine the first feasible skeleton, render to
    mp4."""
    spec = generator.build_spec(seed, stratum)
    cfg = generator.build_task_config(spec)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        task_path = tf.name
    eid = f"kinder/Restock3D-demo-r{stratum}-s{seed}-v0"
    if eid not in gymnasium.registry:
        gymnasium.register(
            id=eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={"task_config_path": task_path, "scene_render_camera": "task_view"},
        )
    env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
    try:
        obs, _ = env.reset(seed=seed)
        n_obj = len(cfg["goal_objects"])
        models = create_restock3d_models(
            env.observation_space, env.action_space, task_path, num_objects=n_obj
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        region_infos = load_region_infos(task_path, x0)
        dims = object_dims(x0, CubeType)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        gen: AbstractPlanGenerator = RelationalHeuristicSearchAbstractPlanGenerator(
            models.types, models.predicates, models.operators, "hff", seed=seed
        )
        pool = list(itertools.islice(gen(x0, s0, goal, 30.0, bpg), 60))
        chosen = None
        for state_plan, action_plan in pool:
            if evaluate_skeleton(state_plan, action_plan, region_infos, dims).feasible:
                chosen = (state_plan, action_plan)
                break
        if chosen is None:
            return False, "no feasible skeleton"
        plan_desc = _plan_str(chosen[1])

        sampler = make_recording_sampler(
            controller_generator=RelationalControllerGenerator(models.skills),
            transition_function=models.transition_fn,
            state_abstractor=models.state_abstractor,
            max_trajectory_steps=500,
            region_infos=region_infos,
            robot_name="robot",
            geometric_place=True,
            geometric_pick_tall=True,
        )
        refiner = BacktrackingRefiner(
            trajectory_sampler=sampler, num_sampling_attempts_per_step=8, seed=seed
        )
        plan = refiner(x0, chosen[0], chosen[1], _REFINE_BUDGET_S, bpg)
        if plan is None:
            return False, plan_desc

        render_env = kinder.make(eid, render_mode="rgb_array", allow_state_access=True)
        render_env.reset(seed=seed)
        space = render_env.unwrapped.observation_space
        frames = []
        for state in plan.states:
            render_env.unwrapped.set_state(space.vectorize(state))  # type: ignore[attr-defined]
            frames.append(render_env.render())
        reached = goal.check_abstract_state(models.state_abstractor(plan.states[-1]))
        render_env.close()
        if reached:
            imageio.mimsave(out_path, frames, fps=20)  # type: ignore[arg-type]
        return bool(reached), plan_desc
    finally:
        env.close()
        os.unlink(task_path)


def render_stratum_demo(stratum: int, out_path: str, max_seeds: int) -> bool:
    """Try seeds until one refines to the goal, then write its mp4.

    Returns success.
    """
    for seed in range(max_seeds):
        try:
            reached, plan_desc = _refine_and_render(stratum, seed, out_path)
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            print(f"[r{stratum}] seed {seed}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if reached:
            print(f"[r{stratum}] seed {seed}: SUCCESS -> {out_path}", flush=True)
            print(f"[r{stratum}]   plan: {plan_desc}", flush=True)
            return True
        print(
            f"[r{stratum}] seed {seed}: not solved ({plan_desc}), next seed", flush=True
        )
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--strata", default="0,1,2", help="comma-separated strata to demo")
    ap.add_argument(
        "--max-seeds", type=int, default=8, help="seed-retry budget per stratum"
    )
    args = ap.parse_args(argv)

    os.makedirs(_DEMOS, exist_ok=True)
    results = []
    for stratum in [int(s) for s in args.strata.split(",")]:
        out = os.path.join(_DEMOS, f"demo_r{stratum}.mp4")
        results.append((stratum, render_stratum_demo(stratum, out, args.max_seeds)))

    print("\n=== demo summary ===")
    for stratum, ok in results:
        print(
            f"  r{stratum}: {'demo_r%d.mp4 written' % stratum if ok else 'FAILED (no seed solved)'}"
        )
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
