"""Tests for the short-cube front-grasp Shelf3D variant (PORTABLE).

Portability: imports the env-model builder DIRECTLY from the local
``shelf3d_front_small`` module. Change that import to your package path when you vendor
the files.
"""

import kinder
import numpy as np
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnvConfig

# --- Change these imports to your package path when you vendor the files. ---
from shelf3d_front_small import SMALL_CUBE_HALF_EXTENTS, create_bilevel_planning_models

kinder.register_all_environments()


def _run_skill(ground_skill, env_models, env, obs):
    """Drive a grounded skill to termination, returning the final obs."""
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    assert ground_skill.operator.preconditions.issubset(
        env_models.state_abstractor(state).atoms
    )
    controller = ground_skill.controller
    controller.reset(state, controller.sample_parameters(state, rng))
    for _ in range(600):
        obs, _, terminated, _, _ = env.step(controller.step())
        controller.observe(env_models.observation_to_state(obs))
        if controller.terminated() or terminated:
            break
    return obs


def _pick_and_place(env, env_models, seed):
    preds = {p.name: p for p in env_models.predicates}
    Holding, HandEmpty, OnFixture = (
        preds["Holding"],
        preds["HandEmpty"],
        preds["OnFixture"],
    )
    skills = {s.operator.name: s for s in env_models.skills}

    obs, _ = env.reset(seed=seed)
    state0 = env_models.observation_to_state(obs)
    objs = {o.name: o for o in env_models.state_abstractor(state0).objects}
    robot, cube, shelf = objs["robot"], objs["cube0"], objs["shelf"]
    start_quat = [
        state0.get(cube, f) for f in ("pose_qx", "pose_qy", "pose_qz", "pose_qw")
    ]

    obs = _run_skill(skills["Pick"].ground((robot, cube)), env_models, env, obs)
    atoms1 = env_models.state_abstractor(env_models.observation_to_state(obs)).atoms
    assert Holding([robot, cube]) in atoms1

    obs = _run_skill(skills["Place"].ground((robot, cube, shelf)), env_models, env, obs)
    state2 = env_models.observation_to_state(obs)
    atoms2 = env_models.state_abstractor(state2).atoms
    assert HandEmpty([robot]) in atoms2
    assert OnFixture([cube, shelf]) in atoms2

    # The short cube must still be upright: same orientation it started with.
    final_quat = [
        state2.get(cube, f) for f in ("pose_qx", "pose_qy", "pose_qz", "pose_qw")
    ]
    assert np.allclose(final_quat, start_quat, atol=1e-2)


def test_shelf3d_front_small_pick_place():
    """Front-grasp the short cube and place it upright, repeatedly (2 seeds)."""
    config = Shelf3DEnvConfig(block_half_extents=SMALL_CUBE_HALF_EXTENTS)
    env = kinder.make(
        "kinder/KinematicShelf3D-o1-v0", render_mode="rgb_array", config=config
    )
    env_models = create_bilevel_planning_models(
        env.observation_space, env.action_space, num_objects=1, config=config
    )
    for seed in (0, 42):
        _pick_and_place(env, env_models, seed)
    env.close()
