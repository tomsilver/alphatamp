"""Demo: front-grasp pick + translate-only place of the SHORT 5cm cube (PORTABLE).

Runs the kinematic3D ``KinematicShelf3D-o1`` task with a symmetric 5 cm cube and
the short-cube front-grasp calibration (from the local ``shelf3d_front_small``
builder), and writes a single mp4 that concatenates several successful episodes
to show the calibration is repeatable (not a one-off). The swept calibration
solves single-attempt (12/12 across seeds); a small retry loop is kept only as a
safety net.

Usage (in an env with ``kinder`` + ``kinder_models`` + ``bilevel_planning``)::

    python demo_front_shelf3d_small.py                  # seeds 0 42 7
    python demo_front_shelf3d_small.py --seeds 0 1 2 --out /tmp

Portability: the builder is imported DIRECTLY from the local
``shelf3d_front_small`` module. Change that import to your package path when you
vendor these files.

Note: the arm uses IKFast (compiles once on first use). If that build fails for
missing static LAPACK/BLAS, point it at the shared libs, e.g.
``LAPACK_DIR=<dir> BLAS_DIR=<dir> python demo_front_shelf3d_small.py``.
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import kinder
import numpy as np
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnvConfig
from pybullet_helpers.camera import capture_image

# --- Change these imports to your package path when you vendor the files. ---
from shelf3d_front_small import (
    SMALL_CUBE_HALF_EXTENTS,
    create_bilevel_planning_models,
)

ENV_ID = "kinder/KinematicShelf3D-o1-v0"
MAX_SKILL_STEPS = 600


def make_env():
    config = Shelf3DEnvConfig(block_half_extents=SMALL_CUBE_HALF_EXTENTS)
    env = kinder.make(ENV_ID, render_mode="rgb_array", config=config)
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


def _drive(env, models, controller, obs, rng, frames):
    state = models.observation_to_state(obs)
    controller.reset(state, controller.sample_parameters(state, rng))
    terminated = False
    for _ in range(MAX_SKILL_STEPS):
        obs, _, terminated, _, _ = env.step(controller.step())
        frames.append(render(env))
        controller.observe(models.observation_to_state(obs))
        if controller.terminated() or terminated:
            break
    return obs


def run_episode(env, models, seed, attempts=6):
    """Front-grasp pick then translate-only place; retry on sampling failure."""
    oce = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    skills = {s.operator.name: s for s in models.skills}
    frames = []
    for attempt in range(attempts):
        obs, _ = env.reset(seed=seed)
        state = models.observation_to_state(obs)
        robot = state.get_objects(Kinematic3DRobotType)[0]
        cube = state.get_objects(Kinematic3DCuboidType)[0]
        shelf = state.get_objects(Kinematic3DFixtureType)[0]
        goal_atoms = models.goal_deriver(state).atoms
        pick = skills["Pick"].ground((robot, cube)).controller
        place = skills["Place"].ground((robot, cube, shelf)).controller
        rng = np.random.default_rng(seed + attempt * 777)
        frames = [render(env)]
        try:
            obs = _drive(env, models, pick, obs, rng, frames)
            if oce._grasped_object is None:  # pylint: disable=protected-access
                continue
            obs = _drive(env, models, place, obs, rng, frames)
        except BaseException:  # noqa: B036 (TrajectorySamplingFailure is BaseException)
            continue
        final_atoms = models.state_abstractor(models.observation_to_state(obs)).atoms
        if goal_atoms.issubset(final_atoms):
            print(f"  seed {seed}: SUCCESS (attempt {attempt + 1})")
            return frames
    print(f"  seed {seed}: FAILED after {attempts} attempts")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 7])
    parser.add_argument("--out", type=str, default=".")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    kinder.register_all_environments()
    env, models = make_env()
    all_frames = []
    for seed in args.seeds:
        frames = run_episode(env, models, seed)
        all_frames.extend(frames)
        all_frames.extend([frames[-1]] * 20)  # hold last frame between episodes
    env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "front_shelf3d_small_cube.mp4"
    iio.mimsave(path, all_frames, fps=args.fps, macro_block_size=16)
    print(f"wrote {path}  ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()
