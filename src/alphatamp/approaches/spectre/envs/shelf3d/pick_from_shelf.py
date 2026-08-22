"""A pick-from-shelf controller for the obstruction Shelf3D variant.

kinder's stock ``PickShelfController`` is ground-only: it reaches a *top-down* grasp pose ON
the object, and the arm motion planning for that descent fails for any cube already on a
shelf (measured 0/12 on shelf 2, 0/8 on the open top shelf, vs 12/12 on the ground) -- a
top-down approach collides with the shelf above the cube. The obstruction task must
*relocate* shelf blockers, so it needs a controller that can grasp a cube off a shelf.

**Pick is the place skill run in reverse.** ``PlaceShelfController`` reaches a *fixed*
end-effector pose ``ARM_MOVEMENT_CUPBOARD`` (a horizontal reach that clears the shelf, whose
IK is reliably solvable -- this is how place deposits a cube on shelf 2), and the **base
position alone** decides where the held cube ends up: the arm config at the reach pose is
base-relative, hence identical wherever the base stands, so the grasped-cube centre is a
*fixed rigid offset* from the base. Place opens the gripper there and retracts. To pick, we
put the base where a *placed* cube would land exactly on the blocker, reach the same fixed
pose, then **close** instead of open and retract with the blocker held. No grasp-orientation
search is needed: the gripper pose that holds a cube stably at this shelf location is exactly
the one place ends in, proven by construction.

The grasp/lift helpers (``terminated``, arm-conf readers, closeness tests) are inherited from
``PickShelfController``; ``reset``, ``step`` and ``sample_parameters`` are overridden -- the
into-shelf reach needs waypoint-following execution, not the parent's straight-line profile.
"""

import os
from typing import Any

import numpy as np
from kinder_models.dynamic3d.shelf.parameterized_skills import PickShelfController
from kinder_models.dynamic3d.utils import (
    ARM_MOVEMENT_CUPBOARD,
    BASE_DISTANCE_TO_CUPBOARD,
    BASE_TO_CUPBOARD_ROTATION,
    GRASP_TRANSFORM_TO_OBJECT,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)
from prpl_utils.utils import get_signed_angle_distance
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
)
from spatialmath import SE2

_CUPBOARD_NAME = "cupboard_1"
_STANDOFF = 0.15  # metres in front of the grasp pose to stage the horizontal insertion
_INSERT_STEPS = 16  # waypoints along the straight insertion / extraction
_LIFT = 0.12  # metres to raise the held cube (gently) after clearing the shelf, before tucking


class PickFromShelfController(PickShelfController):
    """Grasp a cube off the cupboard shelf via the place skill's cupboard-reach, reversed."""

    # Grasp tuning (set on the instance before reset; defaults are the calibrated shelf grasp).
    # _grasp_z_extra raises the grasp centre above the cube centre so the fingers clear the
    # shelf surface; _grasp_roll rolls the gripper about the approach axis so its finger-close
    # axis straddles the cube laterally rather than vertically.
    _grasp_z_extra: float = 0.0
    _grasp_roll: float = (
        0.0  # 0 => fingers open laterally (Y), the shelf side-grasp axis
    )

    def sample_parameters(self, x: Any, rng: np.random.Generator) -> Any:
        # The blocker's pose fixes where to reach; the only free parameter is a small base
        # depth jitter (along the approach axis), letting the refiner retry a slightly
        # different stand-off if base/arm motion planning fails for one sample.
        del x
        return np.array([rng.uniform(-0.02, 0.02)], dtype=np.float32)

    def _grasp_point_for_base(
        self,
        plan_x: Any,
        base_pose: SE2,
        robot: Any,
        reach_local: Pose,
        held_transform: Pose | None = None,
    ) -> np.ndarray:
        """World xyz of where the held/grasped cube centre sits for a given base pose.

        The arm reaches the base-relative ``reach_local`` pose, and the cube sits at
        ``held_transform`` from the end-effector (``GRASP_TRANSFORM_TO_OBJECT.invert()`` for the
        canonical pick grasp; the actual held offset for place). So the cube centre is a rigid
        function of the base pose; we read it off pybullet rather than re-deriving the transform
        chain. (Pure kinematic forward pass -- IK on ``reach_local`` is solved separately.)
        """
        assert self._pybullet_sim is not None
        if held_transform is None:
            held_transform = GRASP_TRANSFORM_TO_OBJECT.invert()
        plan_x.set(robot, "pos_base_x", base_pose.x)
        plan_x.set(robot, "pos_base_y", base_pose.y)
        plan_x.set(robot, "pos_base_rot", base_pose.theta())
        self._pybullet_sim.set_state(plan_x)
        ee_reach = multiply_poses(self._pybullet_sim.robot.get_base_pose(), reach_local)
        grasp_point = multiply_poses(ee_reach, held_transform)
        return np.asarray(grasp_point.position, dtype=float)

    def reset(
        self,
        x: Any,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        self._last_state = x
        self._current_params = np.asarray(params, dtype=np.float32)
        jitter = float(self._current_params[0])

        robot = self.objects[0]
        blocker = self.objects[1]
        cupboard = x.get_object_from_name(_CUPBOARD_NAME)
        cupboard_pose = get_overhead_object_se2_pose(x, cupboard)
        blocker_world = Pose(
            (
                x.get(blocker, "x"),
                x.get(blocker, "y"),
                x.get(blocker, "z"),
            ),
            (
                x.get(blocker, "qx"),
                x.get(blocker, "qy"),
                x.get(blocker, "qz"),
                x.get(blocker, "qw"),
            ),
        )

        # Base heading: face the cupboard, exactly as place does.
        base_rot = cupboard_pose.theta() + BASE_TO_CUPBOARD_ROTATION
        blocker_xyz = np.asarray(blocker_world.position, dtype=float)

        # 1. Trial stance in front of the cupboard (a reachable stand-off). The place skill's
        #    fixed reach aims ~11cm above a shelf cube (place *drops* the cube; pick must grasp
        #    it in place), so lower the reach's z until the grasp centre sits at the blocker's
        #    height. base is flat and the arm-base z is constant, so grasp_z is exactly linear
        #    in reach z -- one measurement gives the shift. xy is then delta-corrected with the
        #    same lowered reach, so the base lands the grasp centre on the blocker.
        plan_x = x.copy()
        trial_base = get_target_robot_pose_from_parameters(
            cupboard_pose, BASE_DISTANCE_TO_CUPBOARD, BASE_TO_CUPBOARD_ROTATION
        )
        grasp0 = self._grasp_point_for_base(
            plan_x, trial_base, robot, ARM_MOVEMENT_CUPBOARD
        )
        z_shift = (blocker_xyz[2] + self._grasp_z_extra) - grasp0[2]
        # Roll the reach orientation about the approach (base-x) axis so the gripper's
        # finger-close axis can be aligned across the cube for a shelf side-grasp.
        roll = self._grasp_roll
        reach_orientation = multiply_poses(
            Pose((0.0, 0.0, 0.0), (np.sin(roll / 2), 0.0, 0.0, np.cos(roll / 2))),
            ARM_MOVEMENT_CUPBOARD,
        ).orientation
        reach_local = Pose(
            (
                ARM_MOVEMENT_CUPBOARD.position[0],
                ARM_MOVEMENT_CUPBOARD.position[1],
                ARM_MOVEMENT_CUPBOARD.position[2] + z_shift,
            ),
            reach_orientation,
        )
        self._reach_local = reach_local
        trial_grasp = self._grasp_point_for_base(plan_x, trial_base, robot, reach_local)
        delta = blocker_xyz[:2] - trial_grasp[:2]
        # Extra depth jitter along the approach axis (base heading), for refiner retries.
        base_x = trial_base.x + float(delta[0]) + jitter * float(np.cos(base_rot))
        base_y = trial_base.y + float(delta[1]) + jitter * float(np.sin(base_rot))
        target_base_pose = SE2(base_x, base_y, base_rot)

        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
        )
        assert base_motion_plan is not None, "Base motion planning failed"
        self._current_base_motion_plan = base_motion_plan

        # 2. Settle the base at the planned final pose and reach the SAME fixed pose place uses
        #    (reliable IK). The grasp centre now coincides with the blocker by construction.
        final_base_pose = self._current_base_motion_plan[-1]
        plan_x = x.copy()
        if not self._navigated:
            plan_x.set(robot, "pos_base_x", final_base_pose.x)
            plan_x.set(robot, "pos_base_y", final_base_pose.y)
            plan_x.set(robot, "pos_base_rot", final_base_pose.theta())
        self._pybullet_sim.set_state(plan_x)  # empty gripper during the approach
        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()
        target_end_effector_pose = multiply_poses(current_arm_base_pose, reach_local)
        target_joints = inverse_kinematics(
            self._pybullet_sim.robot, target_end_effector_pose, set_joints=False
        )
        # 3. Grasp offset for the retract: the real blocker relative to the reach EE pose (so
        #    the held cube carries its actual pose out).
        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(), blocker_world
        )

        held_id = self._pybullet_sim._cubes[
            blocker.name
        ]  # pylint: disable=protected-access

        # Reaching *into* the shelf by motion-planning straight to the grasp conf routes the arm
        # up-and-over the cupboard and then cannot descend into the narrow opening (the pybullet
        # plan clears but the MuJoCo arm stalls on the descent). Instead use the standard
        # into-shelf manipulation pattern: motion-plan to a FRONT stand-off at the grasp height,
        # then a straight horizontal INSERTION into the opening. The stand-off is outside the
        # shelf (clean MP); the insertion is a short linear EE move through the verified-clear
        # opening, which the waypoint follower tracks without contortion.
        approach_dir = np.array(
            [np.cos(base_rot), np.sin(base_rot), 0.0]
        )  # world direction the gripper reaches (base heading)
        standoff_ee = Pose(
            tuple(
                np.asarray(target_end_effector_pose.position) - _STANDOFF * approach_dir
            ),
            target_end_effector_pose.orientation,
        )
        # Build the extraction (grasp -> stand-off) as a Cartesian straight line, IK'ing each
        # step seeded from the previous conf so the whole leg stays in ONE IK branch. A plain
        # joint-space lerp between an independently-IK'd stand-off and the grasp conf can land in
        # different branches and swing the EE far off the straight line (measured 0.27 m off).
        extraction = self._cartesian_path(
            target_end_effector_pose, standoff_ee, list(target_joints), _INSERT_STEPS
        )
        standoff_joints = extraction[-1]
        insertion = list(reversed(extraction))  # stand-off -> grasp
        # After extracting the cube from the shelf, raise it straight up (gentle, branch-
        # consistent) before the big tuck-to-home swing -- the tuck alone shakes the cube loose.
        lift_ee = Pose(
            (
                standoff_ee.position[0],
                standoff_ee.position[1],
                standoff_ee.position[2] + _LIFT,
            ),
            standoff_ee.orientation,
        )
        lift = self._cartesian_path(
            standoff_ee, lift_ee, standoff_joints, _INSERT_STEPS
        )
        self._pybullet_sim.set_state(
            plan_x
        )  # restore clean conf (IK perturbs the robot)
        start_joints = self._pybullet_sim.get_robot_joints()

        # MP home -> front stand-off (empty gripper, outside the shelf); retry across seeds.
        to_standoff = None
        for seed in range(8):
            self._pybullet_sim.set_state(plan_x)
            to_standoff = run_motion_planning(
                self._pybullet_sim.robot,
                start_joints,
                standoff_joints,
                collision_bodies=self._pybullet_sim.get_collision_bodies(),
                seed=seed,
                physics_client_id=self._pybullet_sim.physics_client_id,
            )
            if to_standoff is not None:
                break

        if os.environ.get("SHELF3D_DEBUG"):
            _gp = multiply_poses(
                target_end_effector_pose, GRASP_TRANSFORM_TO_OBJECT.invert()
            )
            print(
                f"[PFS] final_base=({final_base_pose.x:.3f},{final_base_pose.y:.3f},"
                f"{final_base_pose.theta():.3f}) z_shift={z_shift:+.3f} "
                f"grasp={np.round(_gp.position,3)} blk={np.round(blocker_xyz,3)} "
                f"toStandoff={'OK' if to_standoff else 'FAIL'}",
                flush=True,
            )
        assert to_standoff is not None, "Approach (to stand-off) motion planning failed"

        # Densely resample the free-space approach so the follower tracks the true path.
        to_standoff = remap_joint_position_plan_to_constant_distance(
            to_standoff, self._pybullet_sim.robot, max_distance=0.1
        )
        # Approach = MP to the front stand-off, then the straight insertion into the opening.
        # Retract = the straight extraction back out, then a gentle vertical lift clear of the
        # shelf. The pick ends with the cube extracted and lifted, held in the gripper -- a big
        # tuck-to-home swing torques a laterally-gripped cube loose, and the downstream place
        # skill takes over from a held state anyway.
        self._current_arm_joint_plan = list(to_standoff) + insertion
        self._current_retract_plan = extraction + lift
        # Waypoint-following indices (see step()): the grasp-height reach goes *into* the shelf,
        # so the arm must track the collision-free plan rather than straight-line to its end.
        self._approach_wp_idx = 0
        self._retract_wp_idx = 0
        self._wp_stall = 0

    def step(self) -> Any:
        """Execute navigate -> approach -> close -> retract, tracking the collision-free arm
        plan waypoint-by-waypoint (the parent straight-lines to the endpoint, which crashes the
        arm into the shelf for a reach that must enter the shelf opening)."""
        assert self._current_arm_joint_plan is not None
        assert self._current_retract_plan is not None
        assert self._current_base_motion_plan is not None
        kp = 2.0

        # Phase 1: base navigation (identical to the parent).
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            next_pose = self._current_base_motion_plan[0]
            action = np.zeros(11, dtype=np.float32)
            action[0] = next_pose.x - robot_pose.x
            action[1] = next_pose.y - robot_pose.y
            action[2] = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
            action[-1] = self._get_current_robot_gripper_pose()
            return action

        # Phase 2: approach the grasp conf, following the collision-free plan waypoints.
        if self._navigated and not self._pre_grasp and not self._closed_gripper:
            action = self._follow_plan(
                self._current_arm_joint_plan, "_approach_wp_idx", kp
            )
            if action is None:
                self._pre_grasp = True
            else:
                action[-1] = (
                    self._get_current_robot_gripper_pose()
                )  # gripper stays open
                return action

        # Phase 3: close the gripper on the blocker (identical to the parent).
        if self._pre_grasp and not self._closed_gripper:
            if self._get_current_robot_gripper_pose() > 0.2 and np.isclose(
                self._get_current_robot_gripper_pose(),
                self._last_gripper_state,
                atol=0.02,
            ):
                self._closed_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 1
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action

        # Phase 4: retract to home with the blocker held, following the retract plan waypoints.
        if self._pre_grasp and self._closed_gripper:
            action = self._follow_plan(
                self._current_retract_plan, "_retract_wp_idx", kp
            )
            if action is None:
                self._lifted = True
                action = np.zeros(11, dtype=np.float32)
            action[-1] = 1  # keep squeezing so the grasp holds through the lift
            return action

        raise ValueError("Invalid state")

    def _follow_plan(self, plan: Any, idx_attr: str, kp: float) -> Any:
        """Command the arm toward the current waypoint, advancing when close. Returns the action
        (velocity command), or None once the final waypoint is reached (phase done)."""
        idx = getattr(self, idx_attr)
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        # Advance to the next waypoint once we are near the current one. Dense waypoints + a
        # tight tolerance keep the arm on the true collision-free path (no corner-cutting).
        if idx < len(plan) - 1 and self._conf_close(curr, plan[idx][:7], atol=0.06):
            idx += 1
        setattr(self, idx_attr, idx)
        target = np.array(plan[idx][:7])
        # Done when the final waypoint is reached (or the arm stalls against contact there).
        if idx >= len(plan) - 1:
            if self._conf_close(curr, target, atol=0.06):
                return None
            self._wp_stall += 1
            if self._wp_stall > 80:  # physics won't converge the last mm -- accept it
                self._wp_stall = 0
                return None
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = kp * (target - curr)
        return action

    @staticmethod
    def _conf_close(a: np.ndarray, b: Any, atol: float) -> bool:
        return bool(np.max(np.abs(np.asarray(a) - np.asarray(b)[:7])) < atol)

    def _cartesian_path(
        self, ee_from: Pose, ee_to: Pose, seed_conf: list, n: int
    ) -> list:
        """A straight-line Cartesian EE path from ee_from to ee_to as joint confs. Each step is
        IK'd seeded from the previous conf (branch-consistent), so the returned joint path keeps
        the EE on the straight line -- unlike a joint-space lerp between independently-IK'd ends.
        Returns n confs including both endpoints. Restores the sim to a clean state on exit.
        """
        assert self._pybullet_sim is not None
        p0 = np.asarray(ee_from.position, dtype=float)
        p1 = np.asarray(ee_to.position, dtype=float)
        confs: list = []
        prev = list(seed_conf)
        for t in np.linspace(0.0, 1.0, n):
            ee_t = Pose(tuple(p0 + (p1 - p0) * t), ee_from.orientation)
            self._pybullet_sim.robot.set_joints(prev)  # seed IK from the previous conf
            conf = list(
                inverse_kinematics(self._pybullet_sim.robot, ee_t, set_joints=False)
            )
            confs.append(conf)
            prev = conf
        return confs
