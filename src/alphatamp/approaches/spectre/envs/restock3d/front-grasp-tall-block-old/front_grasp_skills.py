"""Front-grasp pick + translate-only place skills for Shelf3D (PORTABLE).

Drop-in module for a repo that *imports* the kinder packages
(``kindergarden`` / ``kinder_models``) as dependencies. It defines two
parameterized controllers for placing a TALL, upright block into a shelf cell:

* ``FrontGroundPickController`` -- grasps the block from the FRONT with a
  diagonal (45 deg) approach, fingers on the block's LEFT/RIGHT (+/-x) faces.
* ``FrontGroundPlaceController`` -- inserts the block into a shelf cell WITHOUT
  rotating it (translate-only), so it ends up upright, same face down.

Portability notes:
  * The PICK controller here is **fully self-contained**: it overrides
    ``step()`` completely, so it does NOT require the ``GroundPickController``
    hook (an overridable ``self._grasp_transform`` + extracted
    ``_plan_grasp_approach``) that the kinder-baselines source uses. It works
    against a stock, pip/git-installed ``kinder_models``.
  * The PLACE controller subclasses the stock ``GroundPlaceController`` and only
    reuses inherited ``BasePlaceController`` helpers -- no patch needed either.
  * When you drop this into your package, the imports of ``kinder_*`` /
    ``bilevel_planning`` / ``pybullet_helpers`` / ``relational_structs`` stay
    as-is. Only your OWN intra-package imports (this module <-> shelf3d_front
    <-> demo/test) use your package namespace.
"""

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.shelf3d import (
    Kinematic3DRobotType,
    ObjectCentricShelf3DEnv,
    Shelf3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import Kinematic3DRobotActionSpace
from kinder_models.kinematic3d.constants import (
    GRIPPER_CLOSE_THRESHOLD,
    HOME_JOINT_POSITIONS,
)
from kinder_models.kinematic3d.shelf3d.parameterized_skills import (
    GroundPickController,
    GroundPlaceController,
)
from kinder_models.kinematic3d.utils import get_target_robot_pose_from_parameters
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs import Variable

# --- Front-grasp geometry -------------------------------------------------

# Object-frame -> end-effector transform for a FRONT grasp: a diagonal
# down-and-forward approach (tool +z pitched 45 deg from vertical toward +y /
# the shelf) with the fingers gripping the block's LEFT/RIGHT (+/-x) faces so
# the pads sit flat and the block does not slip.
#
# Two constraints shape the orientation:
#  * A *fully* horizontal grasp of a block resting on the ground is
#    kinematically infeasible for this arm (mounted ~0.4 m up, it cannot point
#    the wrist horizontal that low -- IKFast returns no solution at any base
#    distance). The most-horizontal approach feasible at BOTH the low ground
#    pick AND the shelf insertion is ~45 deg, so the approach is pitched 45 deg.
#  * The finger-closing axis is tool +x and the approach axis is tool +z (from
#    the Robotiq 2F-85 model). To grip the +/-x faces, tool +x must map to world
#    +/-x -- a pure roll about the approach axis, always reachable since joint_7
#    (the wrist roll) is a continuous joint and the IKFast free joint.
#
# The result is a pure Rx(-135 deg) orientation (x, y, z, w): tool +z -> world
# (0, +0.707, -0.707) (down-and-forward toward the shelf) and tool +x -> world
# +x (fingers straddle the +/-x faces). The block still ends up perfectly
# upright -- translate-only place keeps the object's orientation fixed
# regardless of the gripper pitch/roll.
#
# Offset (object frame == world at grasp, since blocks spawn identity): the
# marker's long axis is along the approach, so back the EE ~2 cm along -y and
# grasp near the TOP of the tall block (+0.057 above center -> tool z ~= 0.12).
# At 45 deg the gripper cannot dip below ~z=0.12 without the fingers hitting the
# floor; this height lets the straight-line reach complete and the marker
# overlap the block.
FRONT_GRASP_PITCH_DEG = 45.0
FRONT_GRASP_TRANSFORM_TO_OBJECT = Pose(
    (0.0, -0.02, 0.057),
    (-0.9238795, 0.0, 0.0, 0.3826834),
)

# Front pick base-approach bounds: stand on the -y side of the block facing +y
# (rot ~= pi/2). FURTHER than the top-down defaults (0.45-0.60) on purpose: at a
# closer standoff the arm folds its elbow back over the mobile base (the grasp
# config penetrates the base for d <= ~0.65 -- impossible on the real robot). At
# d >= 0.70 the arm extends forward and clears the base by ~28 mm. This clearance
# depends only on the standoff (not the block position). The base is also added
# to the arm collision set below as a belt-and-suspenders guarantee.
FRONT_PICK_DISTANCE_BOUNDS = (0.70, 0.75)
FRONT_PICK_ROT_BOUNDS = (np.pi / 2 - 0.05, np.pi / 2 + 0.05)

# Front place params (x/y jitter of the placement within the cell).
FRONT_PLACE_X_OFFSET_BOUNDS = (-0.1, 0.1)
FRONT_PLACE_Y_OFFSET_BOUNDS = (-0.05, 0.05)

# Which shelf board (layer index) to place the tall block on. Layer 1 is the
# lowest that satisfies the goal (cube center z >= 0.224) and the easiest reach;
# layer 2 is a higher-clearance fallback.
TARGET_LAYER = 1
# Back-off distance along the grasp approach axis for the pre-place standoff.
PLACE_STANDOFF = 0.13
# Base standoff distance in front of the shelf when placing.
PLACE_BASE_DISTANCE = 0.8
# Back-off distance along the grasp approach axis for the pre-grasp standoff.
PICK_STANDOFF = 0.12


class FrontGroundPickController(GroundPickController):
    """Pick a block with a front grasp (diagonal approach), fingers on +/-x.

    Self-contained: overrides ``step()`` entirely so it does not depend on the
    ``kinder_models`` ``GroundPickController`` hook. Uses a pre-grasp standoff +
    straight-line Cartesian reach-in (a direct joint-space plan to the low grasp
    config tends to fail because the goal config sits against the block/ground).
    """

    def __init__(self, objects, sim: ObjectCentricShelf3DEnv) -> None:
        super().__init__(objects, sim)

    def sample_parameters(self, x, rng: np.random.Generator):
        assert isinstance(x, Shelf3DObjectCentricState)
        distance = rng.uniform(*FRONT_PICK_DISTANCE_BOUNDS)
        rot = rng.uniform(*FRONT_PICK_ROT_BOUNDS)
        return np.array([distance, rot])

    def _front_plan_grasp_approach(
        self, target_end_effector_pose: Pose
    ) -> list[JointPositions]:
        """Plan the front-grasp approach: standoff + straight-line reach-in."""
        arm = self._sim.robot.arm
        # Include the robot's OWN mobile base so the arm plan cannot fold
        # through it. The env's collision set is only shelf + cubes; the base is
        # a separate PyBullet body that nothing else checks.
        collision_ids = self._sim._get_collision_object_ids() | {
            self._sim.robot.base.robot_id
        }  # pylint: disable=protected-access
        cube_id = (
            self._sim._object_name_to_pybullet_id(  # pylint: disable=protected-access
                self.objects[1].name
            )
        )
        # Pre-grasp standoff: back off along the tool approach axis (tool -z),
        # i.e. up-and-back from the grasp for the diagonal front approach.
        standoff = multiply_poses(
            target_end_effector_pose, Pose((0.0, 0.0, -PICK_STANDOFF))
        )
        plan1 = run_smooth_motion_planning_to_pose(
            standoff,
            arm,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=0,
            max_time=2.0,
            max_candidate_plans=1,
        )
        if plan1 is None:
            raise TrajectorySamplingFailure("front pick: approach to standoff failed")
        # Straight-line reach standoff -> grasp, ignoring the target cube (we are
        # intentionally moving the gripper onto it).
        joint_distance_fn = create_joint_distance_fn(arm)
        plan2 = smoothly_follow_end_effector_path(
            arm,
            [standoff, target_end_effector_pose],
            initial_joints=plan1[-1],
            collision_ids=collision_ids - {cube_id},
            joint_distance_fn=joint_distance_fn,
            include_start=False,
            seed=0,
        )
        return remap_joint_position_plan_to_constant_distance(
            plan1 + plan2,
            arm,
            max_distance=self._sim.config.max_action_mag / 2,
        )

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Shelf3DObjectCentricState)

        # Base motion plan to a standoff in front of the block.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)
            target_pose = self._current_state.get_object_pose(
                self.objects[1].name
            ).to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, self._current_params[0], self._current_params[1]
            )
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=self._sim._get_collision_object_ids(),  # pylint: disable=protected-access
                seed=0,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        # Phase: navigate the base.
        if not self._navigated:
            assert self._current_plan is not None
            target_base_pose = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
                self._navigated = True
            delta = target_base_pose - self._current_state.base_pose
            action_lst = [delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0]
            return np.array(action_lst, dtype=np.float32)

        # Phase: reach to the front-grasp pose (standoff + straight-line reach).
        if self._navigated and not self._pre_grasp:
            if self._current_arm_joint_plan is None:
                self._sim.set_state(self._current_state)
                target_grasp_pose_world = self._current_state.get_object_pose(
                    self.objects[1].name
                )
                target_end_effector_pose = multiply_poses(
                    target_grasp_pose_world, FRONT_GRASP_TRANSFORM_TO_OBJECT
                )
                joint_plan = self._front_plan_grasp_approach(target_end_effector_pose)
                self._current_arm_joint_plan = joint_plan[1:]
            assert self._current_arm_joint_plan is not None
            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._pre_grasp = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            return np.array(action_lst, dtype=np.float32)

        # Phase: close the gripper.
        if self._pre_grasp and not self._closed_gripper:
            if self._get_current_robot_gripper_pose() > GRIPPER_CLOSE_THRESHOLD and (
                np.isclose(
                    self._get_current_robot_gripper_pose(),
                    self._last_gripper_state,
                    atol=0.02,
                )
            ):
                self._closed_gripper = True
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return np.array([0.0] * 10 + [-1.0], dtype=np.float32)

        # Phase: lift/retract to HOME carrying the block.
        if self._closed_gripper and not self._lifted:
            if self._current_retract_plan is None:
                self._sim.set_state(self._current_state)
                grasped_object_id = (
                    self._sim._grasped_object_id  # pylint: disable=protected-access
                )
                grasped_object_transform = (
                    self._sim._grasped_object_transform  # pylint: disable=protected-access
                )
                # Include the mobile base so the retract cannot fold through it.
                all_collision_ids = (
                    self._sim._get_collision_object_ids()
                    | {  # pylint: disable=protected-access
                        self._sim.robot.base.robot_id
                    }
                )
                joint_plan = run_motion_planning(  # type: ignore
                    self._sim.robot.arm,
                    initial_positions=self._sim.robot.arm.get_joint_positions(),
                    target_positions=HOME_JOINT_POSITIONS.tolist(),
                    collision_bodies=all_collision_ids - {grasped_object_id},
                    seed=0,
                    physics_client_id=self._sim.physics_client_id,
                    held_object=grasped_object_id,
                    base_link_to_held_obj=grasped_object_transform,
                )
                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )
                self._current_retract_plan = joint_plan[1:]
            assert self._current_retract_plan is not None
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            return np.array(action_lst, dtype=np.float32)

        raise ValueError("Invalid state")


class FrontGroundPlaceController(GroundPlaceController):
    """Place a block into a shelf cell without rotating it (translate-only).

    Overrides only the pose-building portion of the base ``step()``; the
    ``navigate`` / ``pre_place`` (straight-line reach-in) / ``open_gripper`` /
    ``lift`` phases are inherited unchanged from the stock ``BasePlaceController``.
    """

    def _target_board_top_z(self) -> float:
        cfg = self._sim.config
        return (
            cfg.shelf_pose.position[2]
            + cfg.shelf_height / 2
            + TARGET_LAYER * (cfg.shelf_spacing + cfg.shelf_height)
        )

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Shelf3DObjectCentricState)

        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            grasped_object_id = (
                self._sim._grasped_object_id  # pylint: disable=protected-access
            )
            grasped_object_transform = (
                self._sim._grasped_object_transform  # pylint: disable=protected-access
            )
            assert grasped_object_transform is not None

            target_surface_pose = self._current_state.get_object_pose(
                self.objects[2].name
            )

            # Rest the UPRIGHT block on the target board: board top + block
            # half-height (z half-extent) + a 2 mm gap for the 5 mm release
            # tolerance.
            desired_object_z = (
                self._target_board_top_z()
                + self._sim.config.block_half_extents[2]
                + 0.002
            )
            # Keep the block's pickup (identity) orientation -> translate-only.
            desired_object_pose = Pose(
                (
                    target_surface_pose.position[0] + self._current_params[0],
                    target_surface_pose.position[1] - 0.05 + self._current_params[1],
                    desired_object_z,
                ),
                (0, 0, 0, 1),
            )

            # Use the orientation-preserving EE pose IN FULL (position AND
            # orientation), so the wrist stays at the front-grasp orientation and
            # the block is translated, not rotated.
            ee_pose_from_grasp = multiply_poses(
                desired_object_pose, grasped_object_transform.invert()
            )
            self._target_place_pose_world = ee_pose_from_grasp

            # Pre-place standoff: back off along the grasp APPROACH axis (tool -z),
            # so the straight-line reach-in comes DOWN-and-forward and the block
            # settles onto the board from above-front instead of scraping into the
            # board edge.
            self._pre_place_pose_world = multiply_poses(
                self._target_place_pose_world, Pose((0.0, 0.0, -PLACE_STANDOFF))
            )

            target_pose_temp_se2 = target_surface_pose.to_se2()
            self._target_place_pose_se2 = SE2Pose(
                target_pose_temp_se2.x + self._current_params[0],
                target_pose_temp_se2.y + self._current_params[1],
                target_pose_temp_se2.rot,
            )
            target_base_pose = get_target_robot_pose_from_parameters(
                self._target_place_pose_se2, PLACE_BASE_DISTANCE, np.pi / 2
            )
            all_collision_ids = (
                self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            )
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=all_collision_ids - {grasped_object_id},
                seed=0,
                held_object=grasped_object_id,
                base_link_to_held_obj=grasped_object_transform,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        if not self._navigated:
            return self.navigate()
        if self._navigated and not self._pre_place:
            return self.pre_place()
        if self._pre_place and not self._opened_gripper:
            return self.open_gripper()
        if self._opened_gripper and not self._lifted:
            return self.lift()
        raise ValueError("Invalid state")


def create_front_lifted_controllers(
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricShelf3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create the front-grasp pick + translate-only place lifted controllers."""
    del action_space

    class PickController(FrontGroundPickController):
        """Front pick controller bound to the sim."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    class PlaceController(FrontGroundPlaceController):
        """Front place controller bound to the sim."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    pick_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        PickController,
        Box(
            low=np.array([FRONT_PICK_DISTANCE_BOUNDS[0], FRONT_PICK_ROT_BOUNDS[0]]),
            high=np.array([FRONT_PICK_DISTANCE_BOUNDS[1], FRONT_PICK_ROT_BOUNDS[1]]),
        ),
    )

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_shelf = Variable("?target_shelf", Kinematic3DFixtureType)
    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, target_shelf],
        PlaceController,
        Box(
            low=np.array(
                [FRONT_PLACE_X_OFFSET_BOUNDS[0], FRONT_PLACE_Y_OFFSET_BOUNDS[0]]
            ),
            high=np.array(
                [FRONT_PLACE_X_OFFSET_BOUNDS[1], FRONT_PLACE_Y_OFFSET_BOUNDS[1]]
            ),
        ),
    )

    return {
        "front_pick": pick_controller,
        "front_place": place_controller,
    }
