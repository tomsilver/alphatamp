"""Bilevel-planning models for front-grasping the SHORT (5cm) Shelf3D cube (PORTABLE).

Same task/predicates/operators/abstractor/goal and the same translate-only
place as ``shelf3d_front.py``, but calibrated for a short 5 cm cube instead of
the tall block:

* the env config uses a symmetric ``0.05^3 m`` cube (half-extents 0.025), and
* the front grasp keeps the tall block's 45-deg orientation but grasps the
  cube's CENTER (``grip_height = 0``) because the cube is short, at the same
  standoff.

That calibration was found by a pitch x grip-height x standoff sweep and
verified 12/12 full pick->place across 12 seeds (single attempt, no retries) --
see ``SWEEP_FINDINGS.md``.

Portability note: this is a **standalone builder** -- call
``create_bilevel_planning_models(...)`` from this module DIRECTLY (same as
``shelf3d_front.py``; do NOT route through the package string dispatcher).
When you vendor these files, change the two local imports below to your package
path.
"""

# --- Change these two imports to your package path when you vendor the files. ---
import shelf3d_front
from bilevel_planning.structs import SesameModels
from front_grasp_skills import (
    SMALL_CUBE_FRONT_GRASP_TRANSFORM,
    SMALL_CUBE_PICK_DISTANCE_BOUNDS,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnvConfig

# Symmetric 0.05 x 0.05 x 0.05 m cube (the default Shelf3D block's height, but a
# 5 cm x-footprint so the +/-x faces fit the 85 mm Robotiq gripper).
SMALL_CUBE_HALF_EXTENTS = (0.025, 0.025, 0.025)


def _small_cube_config() -> Shelf3DEnvConfig:
    return Shelf3DEnvConfig(block_half_extents=SMALL_CUBE_HALF_EXTENTS)


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
    config: Shelf3DEnvConfig | None = None,
) -> SesameModels:
    """Create the short-cube front-grasp Shelf3D models.

    Thin wrapper over ``shelf3d_front.create_bilevel_planning_models`` that injects the
    short-cube env config + the short-cube front-grasp calibration. ``config`` (if
    given) must match the executable env's config.
    """
    if config is None:
        config = _small_cube_config()
    return shelf3d_front.create_bilevel_planning_models(
        observation_space,
        action_space,
        num_objects=num_objects,
        config=config,
        grasp_transform=SMALL_CUBE_FRONT_GRASP_TRANSFORM,
        pick_distance_bounds=SMALL_CUBE_PICK_DISTANCE_BOUNDS,
    )
