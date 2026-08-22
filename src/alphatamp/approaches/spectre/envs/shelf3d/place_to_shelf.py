"""A place-to-shelf-region controller for the obstruction Shelf3D variant.

The mirror of ``PickFromShelfController``: it brings a *held* cube to a specific shelf region's
world centre and releases it, using the same base-positioning + stand-off + straight-insertion
machinery (inherited helpers). The only differences from pick are (1) the cube is held through
the approach (gripper closed, cube attached in the motion plan), (2) at the deposit the gripper
opens instead of closing, and (3) the retract is the empty extraction back out of the shelf --
no lift.

Regions are not objects in the env state, so the destination centre comes from a
``region_centers`` map (name -> world xy) built once by the model factory via
``region_geometry``; the place height is the shelf surface plus the held cube's half-height.
"""

import os
from typing import Any

import numpy as np
from kinder_models.dynamic3d.utils import (
    ARM_MOVEMENT_CUPBOARD,
    BASE_DISTANCE_TO_CUPBOARD,
    BASE_TO_CUPBOARD_ROTATION,
    GRIPPER_OPEN_THRESHOLD,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
)
from spatialmath import SE2

from .pick_from_shelf import _INSERT_STEPS, _STANDOFF, PickFromShelfController

_CUPBOARD_NAME = "cupboard_1"


class PlaceToShelfRegionController(PickFromShelfController):
    """Release a held cube at a shelf region's centre (reverse of the shelf grasp)."""

    def __init__(
        self,
        *args: Any,
        region_centers: dict[str, tuple[float, float]] | None = None,
        shelf_surface_z: float = 0.55,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._region_centers = region_centers or {}
        self._shelf_surface_z = shelf_surface_z
        self._placed = False  # gripper opened (cube released)
        self._returned = False  # extraction finished

    def sample_parameters(self, x: Any, rng: np.random.Generator) -> Any:
        del x
        return np.array([rng.uniform(-0.02, 0.02)], dtype=np.float32)

    def reset(
        self,
        x: Any,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        del (
            extend_xy_magnitude,
            extend_rot_magnitude,
        )  # place uses fixed base-MP extents
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        self._last_state = x
        self._current_params = np.asarray(params, dtype=np.float32)
        jitter = float(self._current_params[0])
        self._placed = False
        self._returned = False

        robot, cube, region = self.objects[0], self.objects[1], self.objects[2]
        cupboard = x.get_object_from_name(_CUPBOARD_NAME)
        cupboard_pose = get_overhead_object_se2_pose(x, cupboard)
        base_rot = cupboard_pose.theta() + BASE_TO_CUPBOARD_ROTATION

        # The cube is currently held; its offset from the end-effector is the transform we must
        # reproduce at the reach pose so releasing drops it at the region centre.
        self._pybullet_sim.set_state(x)
        ee_now = self._pybullet_sim.robot.get_end_effector_pose()
        cube_world = Pose(
            (x.get(cube, "x"), x.get(cube, "y"), x.get(cube, "z")),
            (
                x.get(cube, "qx"),
                x.get(cube, "qy"),
                x.get(cube, "qz"),
                x.get(cube, "qw"),
            ),
        )
        held_offset = multiply_poses(ee_now.invert(), cube_world)
        self._pybullet_sim.base_link_to_held_obj = held_offset
        held_id = self._pybullet_sim._cubes[
            cube.name
        ]  # pylint: disable=protected-access

        cx, cy = self._region_centers[region.name]
        place_z = self._shelf_surface_z + x.get(cube, "bb_z") / 2
        target_xyz = np.array([cx, cy, place_z], dtype=float)

        # Base calibration: position so the held cube (at held_offset from the reach EE) lands at
        # the region centre. Same rigid-offset calibration as the pick grasp, with roll=0.
        plan_x = x.copy()
        trial_base = get_target_robot_pose_from_parameters(
            cupboard_pose, BASE_DISTANCE_TO_CUPBOARD, BASE_TO_CUPBOARD_ROTATION
        )
        grasp0 = self._grasp_point_for_base(
            plan_x, trial_base, robot, ARM_MOVEMENT_CUPBOARD, held_offset
        )
        z_shift = target_xyz[2] - grasp0[2]
        reach_local = Pose(
            (
                ARM_MOVEMENT_CUPBOARD.position[0],
                ARM_MOVEMENT_CUPBOARD.position[1],
                ARM_MOVEMENT_CUPBOARD.position[2] + z_shift,
            ),
            ARM_MOVEMENT_CUPBOARD.orientation,
        )
        trial_grasp = self._grasp_point_for_base(
            plan_x, trial_base, robot, reach_local, held_offset
        )
        delta = target_xyz[:2] - trial_grasp[:2]
        base_x = trial_base.x + float(delta[0]) + jitter * float(np.cos(base_rot))
        base_y = trial_base.y + float(delta[1]) + jitter * float(np.sin(base_rot))
        target_base_pose = SE2(base_x, base_y, base_rot)

        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            extend_xy_magnitude=0.025,
            extend_rot_magnitude=np.pi / 8,
        )
        assert base_motion_plan is not None, "Place base motion planning failed"
        self._current_base_motion_plan = base_motion_plan

        final_base_pose = self._current_base_motion_plan[-1]
        plan_x = x.copy()
        if not self._navigated:
            plan_x.set(robot, "pos_base_x", final_base_pose.x)
            plan_x.set(robot, "pos_base_y", final_base_pose.y)
            plan_x.set(robot, "pos_base_rot", final_base_pose.theta())
        self._pybullet_sim.set_state(plan_x, cube)  # cube held during the approach
        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()
        deposit_ee = multiply_poses(current_arm_base_pose, reach_local)
        deposit_joints = inverse_kinematics(
            self._pybullet_sim.robot, deposit_ee, set_joints=False
        )

        # Straight extraction (deposit -> front stand-off), branch-consistent; insertion is its
        # reverse (stand-off -> deposit), executed with the cube held.
        approach_dir = np.array([np.cos(base_rot), np.sin(base_rot), 0.0])
        standoff_ee = Pose(
            tuple(np.asarray(deposit_ee.position) - _STANDOFF * approach_dir),
            deposit_ee.orientation,
        )
        extraction = self._cartesian_path(
            deposit_ee, standoff_ee, list(deposit_joints), _INSERT_STEPS
        )
        standoff_joints = extraction[-1]
        insertion = list(reversed(extraction))

        self._pybullet_sim.set_state(plan_x, cube)
        start_joints = self._pybullet_sim.get_robot_joints()
        to_standoff = None
        for seed in range(8):
            self._pybullet_sim.set_state(plan_x, cube)
            to_standoff = run_motion_planning(
                self._pybullet_sim.robot,
                start_joints,
                standoff_joints,
                collision_bodies=self._pybullet_sim.get_collision_bodies(
                    held_object=held_id
                ),
                held_object=held_id,
                base_link_to_held_obj=held_offset,
                seed=seed,
                physics_client_id=self._pybullet_sim.physics_client_id,
            )
            if to_standoff is not None:
                break
        if os.environ.get("SHELF3D_DEBUG"):
            print(
                f"[PTS] region={region.name} target={np.round(target_xyz,3)} "
                f"final_base=({final_base_pose.x:.3f},{final_base_pose.y:.3f}) "
                f"toStandoff={'OK' if to_standoff else 'FAIL'}",
                flush=True,
            )
        assert to_standoff is not None, "Place approach motion planning failed"
        to_standoff = remap_joint_position_plan_to_constant_distance(
            to_standoff, self._pybullet_sim.robot, max_distance=0.1
        )

        # After releasing, tuck the (empty) arm back home so the next skill plans from a clean
        # start -- unlike pick, there is no held cube to shake loose. Plan from the released
        # standoff conf (cube now on the shelf, so it is a collision body).
        self._pybullet_sim.set_state(plan_x)  # cube released -> not held
        home_return = None
        for seed in range(8):
            self._pybullet_sim.set_state(plan_x)
            self._pybullet_sim.robot.set_joints(standoff_joints)
            home_return = run_motion_planning(
                self._pybullet_sim.robot,
                standoff_joints,
                self.home_joints.tolist(),
                collision_bodies=self._pybullet_sim.get_collision_bodies(),
                seed=seed,
                physics_client_id=self._pybullet_sim.physics_client_id,
            )
            if home_return is not None:
                break
        if home_return is not None:
            home_return = remap_joint_position_plan_to_constant_distance(
                home_return, self._pybullet_sim.robot, max_distance=0.1
            )

        # Approach brings the held cube to the deposit; retract extracts the empty gripper and
        # tucks home.
        self._current_arm_joint_plan = list(to_standoff) + insertion
        self._current_retract_plan = extraction + (
            list(home_return) if home_return is not None else []
        )
        self._approach_wp_idx = 0
        self._retract_wp_idx = 0
        self._wp_stall = 0

    def terminated(self) -> bool:
        return self._returned

    def step(self) -> Any:
        """navigate -> approach (cube held) -> open gripper (release) -> extract (empty)."""
        assert self._current_arm_joint_plan is not None
        assert self._current_retract_plan is not None
        assert self._current_base_motion_plan is not None
        from prpl_utils.utils import get_signed_angle_distance

        kp = 2.0
        # Phase 1: base navigation (cube held; gripper stays closed).
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                if self._robot_is_close_to_pose(self._current_base_motion_plan[0]):
                    self._current_base_motion_plan.pop(0)
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            nxt = self._current_base_motion_plan[0]
            action = np.zeros(11, dtype=np.float32)
            action[0] = nxt.x - robot_pose.x
            action[1] = nxt.y - robot_pose.y
            action[2] = get_signed_angle_distance(nxt.theta(), robot_pose.theta())
            action[-1] = 1  # hold the cube
            return action

        # Phase 2: bring the held cube to the deposit, following the plan; gripper stays closed.
        if self._navigated and not self._placed:
            action = self._follow_plan(
                self._current_arm_joint_plan, "_approach_wp_idx", kp
            )
            if action is not None:
                action[-1] = 1
                return action
            # Reached the deposit -- open the gripper to release the cube.
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._placed = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 0  # open
            return action

        # Phase 3: extract the empty gripper back out of the shelf.
        if self._placed and not self._returned:
            action = self._follow_plan(
                self._current_retract_plan, "_retract_wp_idx", kp
            )
            if action is None:
                self._returned = True
                action = np.zeros(11, dtype=np.float32)
            action[-1] = 0  # stay open
            return action

        raise ValueError("Invalid state")
