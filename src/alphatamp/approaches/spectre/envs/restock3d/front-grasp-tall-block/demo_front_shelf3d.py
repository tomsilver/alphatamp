"""Demo: front-grasp pick + translate-only place of a TALL block in Shelf3D.

Runs the kinematic3D ``KinematicShelf3D-o1`` task with a tall upright block
(full height = half a shelf section) and the front-grasp / translate-only
skills, and writes mp4 videos:

* ``front_shelf3d_skills.mp4``  -- the two controllers driven directly
  (front_pick -> front_place).
* ``front_shelf3d_planner.mp4`` -- the full SeSamE bilevel planner choosing and
  executing the skills.

Usage (in an env with ``kinder`` + ``bilevel_planning`` installed)::

    python demo_front_shelf3d.py                 # both videos
    python demo_front_shelf3d.py --mode skills   # just the controllers
    python demo_front_shelf3d.py --mode planner  # just the planner
    python demo_front_shelf3d.py --seed 7 --out /tmp

Portability: the env-model builder is imported DIRECTLY from the local
``shelf3d_front`` module (not via the ``kinder_bilevel_planning`` string
dispatcher, which only sees files inside the installed package). Change the
``shelf3d_front`` import to your package path when you vendor these files.

Note: the arm uses IKFast, which compiles once on first use. If that build
fails for missing static LAPACK/BLAS, point it at the shared libs, e.g.::

    LAPACK_DIR=<dir> BLAS_DIR=<dir> python demo_front_shelf3d.py
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import kinder
import numpy as np
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnvConfig

# ``BilevelPlanningAgent`` is only needed for the planner demo; if your repo does
# not depend on ``kinder_bilevel_planning`` you can drop the planner path and use
# ``bilevel_planning.sesame.run_sesame`` instead.
from kinder_bilevel_planning.agent import BilevelPlanningAgent
from pybullet_helpers.camera import capture_image

# --- Change these imports to your package path when you vendor the files. ---
from shelf3d_front import TALL_BLOCK_HALF_EXTENTS, create_bilevel_planning_models

ENV_ID = "kinder/KinematicShelf3D-o1-v0"
MAX_SKILL_STEPS = 600


def make_env():
    config = Shelf3DEnvConfig(block_half_extents=TALL_BLOCK_HALF_EXTENTS)
    env = kinder.make(ENV_ID, render_mode="rgb_array", config=config)
    # Call the local builder directly (NOT the env_name string dispatcher).
    models = create_bilevel_planning_models(
        env.observation_space, env.action_space, num_objects=1, config=config
    )
    return env, models


def render(env):
    """Wide view framing both the pick area and the shelf."""
    oce = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    return capture_image(
        oce.physics_client_id,
        camera_target=(1.0, 0.75, 0.35),
        camera_distance=3.9,
        camera_pitch=-31,
        camera_yaw=38,
        image_width=640,
        image_height=480,
    )


def _run_controller(env, models, skill, objects, obs, rng, frames):
    """Drive one grounded skill to termination, appending frames each step."""
    state = models.observation_to_state(obs)
    controller = skill.ground(objects).controller
    controller.reset(state, controller.sample_parameters(state, rng))
    terminated = False
    for _ in range(MAX_SKILL_STEPS):
        obs, _, terminated, _, _ = env.step(controller.step())
        frames.append(render(env))
        controller.observe(models.observation_to_state(obs))
        if controller.terminated() or terminated:
            break
    return obs, terminated


def demo_skills(env, models, seed, attempts=6):
    """Direct controller rollout with whole-episode retry (resampled params)."""
    skills = {s.operator.name: s for s in models.skills}
    frames = []
    for attempt in range(attempts):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + attempt * 777)
        frames = [render(env)]
        state = models.observation_to_state(obs)
        names = {o.name: o for o in models.state_abstractor(state).objects}
        objs = (names["robot"], names["cube0"], names["shelf"])
        try:
            obs, _ = _run_controller(
                env, models, skills["Pick"], objs[:2], obs, rng, frames
            )
            obs, done = _run_controller(
                env, models, skills["Place"], objs, obs, rng, frames
            )
        except (
            BaseException
        ) as exc:  # noqa: B036 (TrajectorySamplingFailure is a BaseException)
            print(
                f"  skills attempt {attempt}: failed ({type(exc).__name__}); retrying"
            )
            continue
        atoms = {
            str(a)
            for a in models.state_abstractor(models.observation_to_state(obs)).atoms
        }
        goal = "(OnFixture cube0 shelf)" in atoms and "(HandEmpty robot)" in atoms
        if done or goal:
            print(f"  skills: solved on attempt {attempt} ({len(frames)} frames)")
            return frames
        print(f"  skills attempt {attempt}: not solved; retrying")
    print("  skills: no success within attempts; returning last frames")
    return frames


def demo_planner(env, models, seed):
    """Full SeSamE planner rollout."""
    agent = BilevelPlanningAgent(
        models,
        seed,
        max_abstract_plans=1,
        samples_per_step=8,
        max_skill_horizon=MAX_SKILL_STEPS,
        heuristic_name="hff",
        planning_timeout=180.0,
    )
    obs, info = env.reset(seed=seed)
    frames = [render(env)]
    agent.reset(obs, info)
    for _ in range(MAX_SKILL_STEPS * 4):
        obs, reward, terminated, truncated, info = env.step(agent.step())
        frames.append(render(env))
        agent.update(obs, reward, terminated, info)
        if terminated or truncated:
            break
    print(f"  planner: terminated={terminated} ({len(frames)} frames)")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "skills", "planner"], default="both")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=".")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    kinder.register_all_environments()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("skills", "both"):
        print("Rendering direct-controller demo...")
        env, models = make_env()
        frames = demo_skills(env, models, args.seed)
        path = out / "front_shelf3d_skills.mp4"
        iio.mimsave(path, frames, fps=args.fps, macro_block_size=16)
        print("wrote", path)
        env.close()

    if args.mode in ("planner", "both"):
        print("Rendering SeSamE-planner demo...")
        env, models = make_env()
        frames = demo_planner(env, models, args.seed)
        path = out / "front_shelf3d_planner.mp4"
        iio.mimsave(path, frames, fps=args.fps, macro_block_size=16)
        print("wrote", path)
        env.close()


if __name__ == "__main__":
    main()
