"""Region-parameterized kinematic place controller for Restock3D.

Subclasses the substrate :class:`BasePlaceController` (whose ``navigate/pre_place/open_gripper/lift``
phases run the real motion planning), retargeting the place pose to a *region* centre + that shelf's
placement surface z, read from ``region_infos`` (regions are not PyBullet bodies / not in the state,
so unlike the stock shelf place we cannot read the target pose from the state). Feasibility is
decided by real collision inside ``pre_place``:

* **F2** — the held block collides with a resident already placed in the region.
* **F3** — a tall block collides with the board above a short shelf.

Both surface as ``run_...motion_planning(...) -> None`` -> ``TrajectorySamplingFailure``, which the
backtracking refiner turns into a refinement failure. Adapted from
``kinder_models/kinematic3d/shelf3d/parameterized_skills.py::GroundPlaceController``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.kinematic3d.shelf3d import Kinematic3DRobotType
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState
from kinder_models.kinematic3d.base_controllers import BasePlaceController
from kinder_models.kinematic3d.constants import (
    GRASP_TRANSFORM_TO_OBJECT,
    GRIPPER_CLOSE_THRESHOLD,
    HOME_JOINT_POSITIONS,
)
from kinder_models.kinematic3d.shelf3d.parameterized_skills import (
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    MOVE_TO_TARGET_ROT_BOUNDS,
    GroundPickController,
)
from kinder_models.kinematic3d.utils import get_target_robot_pose_from_parameters
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    remap_se2_pose_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs import Object, ObjectCentricState, Type, Variable
from scipy.spatial.transform import Rotation as _Rotation

from .region_geometry import RegionInfo

#: Region placement target type (a symbolic object; not a PyBullet body / not in the state).
RegionType = Type("region")

#: Small xy jitter so backtracking gives genuine retries; feasibility comes from MP collision.
_PLACE_JITTER = 0.015
#: Vertical pad so the released object's underside rests just above the board (within the env's
#: ``min_placement_dist`` so it is released, but not penetrating the board).
_PLACE_Z_PAD = 3e-3

#: Floor buffer zone for relocated F1 clutter: a disjoint x-band to the -x (LEFT) of the object
#: sampling region (the fully-lateral layout -- buffer | objects | shelf, left to right; see
#: decisions/07 2026-08-16). The base reaches it by sliding laterally along the southern corridor and
#: front-placing from the south. A cube resting here (floor height, xy in the zone) is abstracted
#: ``OnBuffer`` -- off every goal's grasp -- not ``OnFloor``, so ``Pick`` (precond ``OnFloor``) won't
#: re-pick it. The zone is DISJOINT in x from the object region so a staged goal is never mis-labeled
#: ``OnBuffer``. Buffers are controller-side placement spots, NOT abstract regions (a floor "region"
#: at surface_z ~ 0 would be surface-z-matched and wrongly emit ``Stored``; see decisions/07
#: 2026-08-15).
BUFFER_SPOTS: list[tuple[float, float]] = [
    (-1.15, 0.80),
    (-1.00, 0.80),
    (-1.15, 1.05),
    (-1.00, 1.05),
]
_BUFFER_ZONE_X = (-1.35, -0.90)
_BUFFER_ZONE_Y = (0.55, 1.25)


def in_buffer_zone(x: float, y: float) -> bool:
    """Whether a floor xy lies inside the relocation buffer band (used by the abstractor
    to emit ``OnBuffer`` instead of ``OnFloor``)."""
    return (
        _BUFFER_ZONE_X[0] <= x <= _BUFFER_ZONE_X[1]
        and _BUFFER_ZONE_Y[0] <= y <= _BUFFER_ZONE_Y[1]
    )


#: Stock top-down grasp offset above the object centre (works for short objects).
_STOCK_GRASP_Z = 0.02
#: Grasp this far below a tall object's top; a ``center + _STOCK_GRASP_Z`` grasp is buried inside a
#: tall block so the descending arm collides with its upper body (F0). Grasp height is
#: ``max(_STOCK_GRASP_Z, half_z - _GRIP_INSET)`` so short objects keep the stock (place-calibrated)
#: grasp and only tall objects switch to a top-relative grasp.
_GRIP_INSET = 0.015


def _arm_collision_ids(sim) -> set[int]:
    """Env collision bodies PLUS the robot's OWN mobile base (a separate PyBullet body
    nothing else checks), so an arm plan cannot fold the elbow through the base."""
    return sim._get_collision_object_ids() | {sim.robot.base.robot_id}


def _smooth_base_plan(plan, sim):
    """Densify an SE2 base plan to constant small steps so the base glides directly to
    the target instead of teleporting between sparse motion-planner waypoints.

    The substrate's SE2 densifier interpolates rotation along the shortest arc, which for a plan whose
    consecutive waypoints straddle the ``±π`` branch cut produces a value marginally outside
    ``[-π, π]`` and trips ``SE2Pose``'s range assertion. That is purely a smoothing artefact -- the
    raw waypoints are valid poses and the controller's per-step SE2 delta already takes the shortest
    arc -- so on that boundary failure we fall back to the un-densified plan (coarser base steps, same
    endpoints). Surfaced by the buffer place, whose base start/goal orientations can straddle the cut.
    """
    try:
        return remap_se2_pose_plan_to_constant_distance(
            plan, sim.config.max_action_mag / 2
        )
    except AssertionError:
        return plan


def _place_reach_collision_ids(sim, state) -> set[int]:
    """Collision set for the place reach-in / lift: the shelf boards (F3) + any
    shelf-*resident* movables (F2), but NOT floor movables.

    The held object travels UP to the shelf, away from the staging area, so staging
    objects must not block the reach-in -- otherwise a dense multi-object scene fails
    placements spuriously (nothing to do with region assignment). A movable resting
    above the floor is a resident; one on the floor is staging.
    """
    held = sim._grasped_object_id
    ids = set(sim.shelf_structure_ids())
    for name in sim.movable_names():
        mid = sim._object_name_to_pybullet_id(name)
        if mid != held and state.get_object_pose(name).position[2] > 0.2:
            ids.add(mid)
    return ids


def _base_nav_collision_ids(
    sim, state, exclude: frozenset[str] = frozenset()
) -> set[int]:
    """Base-navigation obstacle set: shelf boards + floor-resting movables, minus
    ``exclude`` (the object being approached or carried).

    Previously the four ``get_base_plan`` call sites passed only
    ``shelf_structure_ids()``, so floor clutter was invisible to both the base motion
    planner and (with ``check_base_collisions`` on) the step-time reversion -- the
    mobile base drove straight through floor blocks. Including floor movables here
    routes the base *around* them. A movable resting above the floor (z > 0.2) is a
    shelf resident, out of the base's swept volume and already covered by the reach-in
    set, so only floor-level movables count. ``exclude`` drops the pick/place target
    itself: nav must not avoid the thing it is reaching for or carrying.
    """
    ids = set(sim.shelf_structure_ids())
    for name in sim.movable_names():
        if name in exclude:
            continue
        if state.get_object_pose(name).position[2] < 0.2:
            ids.add(sim._object_name_to_pybullet_id(name))
    return ids


class RestockPickController(GroundPickController):
    """Top-relative grasp pick: grasp near the object's top so a tall block is
    graspable.

    Identical to the stock ``GroundPickController`` except the grasp end-effector pose targets the
    object's top (``center + half_z - _GRIP_INSET``) instead of ``center + 0.02`` — the latter is
    inside a tall object, so the descending arm collides with its upper body (F0).
    """

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        sim = self._sim

        if self._current_plan is None:
            sim.set_state(self._current_state)
            target_pose = self._current_state.get_object_pose(
                self.objects[1].name
            ).to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, self._current_params[0], self._current_params[1]
            )
            base_plan = get_base_plan(
                sim,
                target_base_pose,
                _base_nav_collision_ids(
                    sim, self._current_state, frozenset({self.objects[1].name})
                ),
                None,
                None,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        if not self._navigated:
            target_base_pose = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
                self._navigated = True
            delta = target_base_pose - self._current_state.base_pose
            return np.array(
                [delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32
            )

        if self._navigated and not self._pre_grasp:
            if self._current_arm_joint_plan is None:
                sim.set_state(self._current_state)
                target_grasp_pose_world = self._current_state.get_object_pose(
                    self.objects[1].name
                )
                half_z = self._current_state.get_object_half_extents(
                    self.objects[1].name
                )[2]
                grasp_tf = top_down_grasp_transform(half_z)
                target_end_effector_pose = multiply_poses(
                    target_grasp_pose_world, grasp_tf
                )
                try:
                    joint_positions = inverse_kinematics(
                        sim.robot.arm,
                        target_end_effector_pose,
                        validate=True,
                        set_joints=False,
                    )
                except InverseKinematicsError as e:
                    raise TrajectorySamplingFailure(
                        f"IK failed for {target_end_effector_pose}"
                    ) from e
                joint_plan = run_motion_planning(
                    sim.robot.arm,
                    initial_positions=sim.robot.arm.get_joint_positions(),
                    target_positions=joint_positions,
                    collision_bodies=_arm_collision_ids(sim),
                    seed=0,
                    physics_client_id=sim.physics_client_id,
                )
                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    sim.robot.arm,
                    max_distance=sim.config.max_action_mag / 2,
                )
                self._current_arm_joint_plan = joint_plan[1:]
            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._pre_grasp = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)

        if self._pre_grasp and not self._closed_gripper:
            g = self._get_current_robot_gripper_pose()
            if g > GRIPPER_CLOSE_THRESHOLD and np.isclose(
                g, self._last_gripper_state, atol=0.02
            ):
                self._closed_gripper = True
            self._last_gripper_state = g
            return np.array([0.0] * 10 + [-1.0], dtype=np.float32)

        if self._closed_gripper and not self._lifted:
            if self._current_retract_plan is None:
                sim.set_state(self._current_state)
                grasped_id = sim._grasped_object_id
                grasped_tf = sim._grasped_object_transform
                joint_plan = run_motion_planning(
                    sim.robot.arm,
                    initial_positions=sim.robot.arm.get_joint_positions(),
                    target_positions=HOME_JOINT_POSITIONS.tolist(),
                    collision_bodies=_arm_collision_ids(sim) - {grasped_id},
                    seed=0,
                    physics_client_id=sim.physics_client_id,
                    held_object=grasped_id,
                    base_link_to_held_obj=grasped_tf,
                )
                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    sim.robot.arm,
                    max_distance=sim.config.max_action_mag / 2,
                )
                self._current_retract_plan = joint_plan[1:]
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)

        raise ValueError("Invalid state")


class RegionPlaceController(BasePlaceController):
    """Place the held object into a named region (centre + surface z from
    ``region_infos``)."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim,
        region_infos: dict[str, RegionInfo],
    ) -> None:
        super().__init__(objects, sim)
        self._region_infos = region_infos

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x
        return rng.uniform(-_PLACE_JITTER, _PLACE_JITTER, size=2)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Kinematic3DObjectCentricState)

        if self._current_plan is None:
            self._sim.set_state(self._current_state)
            grasped_object_id = self._sim._grasped_object_id
            grasped_object_transform = self._sim._grasped_object_transform
            assert grasped_object_transform is not None

            region_name = self.objects[2].name
            info = self._region_infos[region_name]
            rx, ry = info.center_xy
            px, py = float(self._current_params[0]), float(self._current_params[1])
            half = self._current_state.get_object_half_extents(self._target.name)
            # Keep the shelf-approach EE orientation (a top-down orientation cannot reach INTO a
            # ceilinged cell), but choose the EE z so the object's underside lands on the shelf
            # surface, computed analytically from the ACTUAL grasp transform (a fixed correction
            # only worked for the near-centre small-cube grasp). The released object is rotated by
            # ``R_place @ R_grasp``; its vertical half-extent under that rotation is
            # ``sum_i |R[2,i]| * half[i]``, and the EE->object z offset is ``(R_place @ grasp_pos)_z``.
            place_orient = Pose.from_rpy(
                (0.0, 0.0, 0.0), (-np.pi / 2, np.pi, 0)
            ).orientation
            r_place = _Rotation.from_quat(place_orient).as_matrix()
            r_grasp = _Rotation.from_quat(
                grasped_object_transform.orientation
            ).as_matrix()
            r_obj = r_place @ r_grasp
            v_half = sum(abs(r_obj[2, i]) * half[i] for i in range(3))
            offset_z = float(
                (r_place @ np.asarray(grasped_object_transform.position))[2]
            )
            ee_z = info.surface_z + v_half + _PLACE_Z_PAD - offset_z
            self._target_place_pose_world = Pose((rx + px, ry + py, ee_z), place_orient)
            pre_place_height = 0.02
            self._pre_place_pose_world = Pose(
                (
                    self._target_place_pose_world.position[0],
                    self._target_place_pose_world.position[1] - 0.1,
                    self._target_place_pose_world.position[2] + pre_place_height,
                ),
                self._target_place_pose_world.orientation,
            )
            self._target_place_pose_se2 = SE2Pose(rx + px, ry + py, 0.0)
            target_base_pose = get_target_robot_pose_from_parameters(
                self._target_place_pose_se2, 0.8, np.pi / 2
            )
            # Base nav routes around floor movables (minus the carried object); the carried object
            # still must clear the shelf (via held_object/held_tf).
            base_plan = get_base_plan(
                self._sim,
                target_base_pose,
                _base_nav_collision_ids(
                    self._sim, self._current_state, frozenset({self.objects[1].name})
                ),
                grasped_object_id,
                grasped_object_transform,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        reach_ids = _place_reach_collision_ids(self._sim, self._current_state)
        if not self._navigated:
            return self.navigate()
        if self._navigated and not self._pre_place:
            return self.pre_place(collision_ids=reach_ids)
        if self._pre_place and not self._opened_gripper:
            return self.open_gripper()
        if self._opened_gripper and not self._lifted:
            return self.lift(collision_ids=reach_ids)
        raise ValueError("Invalid state")


def get_base_plan(
    sim, target_base_pose, collision_bodies, held_object, held_tf, allow_fallback=False
):
    """Densified base motion plan to ``target_base_pose`` carrying the held object (or
    None).

    The plan is remapped to constant small steps so the base glides directly to the target instead
    of teleporting between sparse motion-planner waypoints.

    ``collision_bodies`` includes floor movables (see ``_base_nav_collision_ids``) so the base routes
    *around* floor objects. **No fallback:** in the fully-lateral layout the base approaches every
    target from a clear southern corridor (objects/buffer are disjoint -x bands north of the corridor,
    the shelf is reached from the south), so a collision-free path always exists on the feasible order.
    A ``None`` therefore means the base is genuinely boxed -- an intended refinement failure -- not a
    reason to phase through: the old shelf-only straight-line fallback (which could overlap floor
    movables and forced ``check_base_collisions`` OFF) is removed (decisions/07 2026-08-16). Strict
    step-time base-collision enforcement is now ON. ``allow_fallback`` is retained for call-site
    compatibility but is inert.
    """
    del allow_fallback

    def _plan(bodies):
        plan = run_single_arm_mobile_base_motion_planning(
            sim.robot,
            sim.robot.base.get_pose(),
            target_base_pose,
            collision_bodies=bodies,
            seed=0,
            held_object=held_object,
            base_link_to_held_obj=held_tf,
            hyperparameters=MotionPlanningHyperparameters(
                birrt_smooth_amt=_BASE_SMOOTH_AMT
            ),
        )
        return None if plan is None else _smooth_base_plan(plan, sim)

    return _plan(collision_bodies)


def create_lifted_controllers(
    action_space, sim, region_infos: dict[str, RegionInfo]
) -> dict[str, LiftedParameterizedController]:
    """Height-adaptive lifted pick + region place controllers for Restock3D.

    ONE Pick / ONE Place skill: the dispatchers select the front-grasp path for tall
    blocks and the top-down / analytic path for cubes from the target height. The pick
    Box is the union of both styles' bounds (sample_parameters dispatches to the style-
    appropriate range).
    """
    del action_space

    class PickController(RestockAdaptivePickController):
        def __init__(self, objects):  # type: ignore[no-untyped-def]
            super().__init__(objects, sim)

    class PlaceController(RegionAdaptivePlaceController):
        def __init__(self, objects):  # type: ignore[no-untyped-def]
            super().__init__(objects, sim, region_infos)

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    pick: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        PickController,
        Box(
            low=np.array(
                [
                    min(
                        MOVE_TO_TARGET_DISTANCE_BOUNDS[0],
                        _FRONT_PICK_DISTANCE_BOUNDS[0],
                    ),
                    min(MOVE_TO_TARGET_ROT_BOUNDS[0], _FRONT_PICK_ROT_BOUNDS[0]),
                ]
            ),
            high=np.array(
                [
                    max(
                        MOVE_TO_TARGET_DISTANCE_BOUNDS[1],
                        _FRONT_PICK_DISTANCE_BOUNDS[1],
                    ),
                    max(MOVE_TO_TARGET_ROT_BOUNDS[1], _FRONT_PICK_ROT_BOUNDS[1]),
                ]
            ),
        ),
    )

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    region = Variable("?region", RegionType)
    place: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, region],
        PlaceController,
        Box(
            low=np.array([-_PLACE_JITTER, -_PLACE_JITTER]),
            high=np.array([_PLACE_JITTER, _PLACE_JITTER]),
        ),
    )

    class BufferPlaceControllerInner(BufferPlaceController):
        """BufferPlaceController bound to this problem's ``sim`` for the lifted
        controller."""

        def __init__(self, objects):  # type: ignore[no-untyped-def]
            super().__init__(objects, sim)

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    place_buffer: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        BufferPlaceControllerInner,
        Box(
            low=np.array([-_PLACE_JITTER, -_PLACE_JITTER]),
            high=np.array([_PLACE_JITTER, _PLACE_JITTER]),
        ),
    )
    return {"pick": pick, "place": place, "place_buffer": place_buffer}


# ==========================================================================================
# Front-grasp pick + translate-only place (for TALL blocks that must stay upright).
#
# Ported from ``envs/restock3d/front-grasp-tall-block/front_grasp_skills.py`` (proven on kinder's
# own KinematicShelf3D). A top-down grasp of a tall block, then a place into a *ceilinged* cell, is
# impossible (the gripper must sit above the block, hence above the ceiling). The front grasp
# approaches diagonally (~45deg down-and-forward) with the fingers on the block's +/-x faces, and
# the place is **translate-only** (keeps the object's orientation fixed), so the upright block is
# threaded under the shelf boards. That is exactly what makes F3 real: an upright block taller than
# a cell's clearance collides the board capping it during the reach-in -> TrajectorySamplingFailure.
# ==========================================================================================

#: Rx(-135deg) grasp orientation: tool +z -> world (0, +0.707, -0.707) (down-and-forward toward the
#: shelf), tool +x -> world +x (fingers straddle the block's +/-x faces).
_FRONT_GRASP_QUAT = (-0.9238795, 0.0, 0.0, 0.3826834)
#: Back the EE ~2cm along -y (object frame == world at grasp, since blocks spawn identity).
_FRONT_GRASP_Y_OFFSET = -0.02
#: Grasp so the EE lands at this fixed WORLD height. The arm's 45deg reach envelope tops out
#: ~0.16m, so a tall block must be gripped LOWER (toward its centre) rather than near its top --
#: near-top grasping a 0.24m block puts the EE at ~0.23m and IK/MP fail. The 0.127m demo block
#: grasped near-top happens to land right here (~0.12), so this generalises the vendored constant.
_FRONT_GRASP_TARGET_EE_Z = 0.13
#: Keep the grip this far inside the block's top/bottom faces (so short objects still grip on-face).
_FRONT_GRIP_MARGIN = 0.015
# FURTHER than the top-down defaults on purpose: at a closer standoff the arm folds its elbow back
# over the mobile base (the grasp config penetrates the base for d <= ~0.65). At d >= 0.70 the arm
# extends forward and clears the base. The base is also in the arm collision set (belt + suspenders).
_FRONT_PICK_DISTANCE_BOUNDS = (0.70, 0.75)
_FRONT_PICK_ROT_BOUNDS = (np.pi / 2 - 0.05, np.pi / 2 + 0.05)
#: Pre-grasp / pre-place back-off along the tool approach axis (tool -z), so the reach-in comes
#: down-and-forward and settles onto the surface from above-front.
_FRONT_PICK_STANDOFF = 0.12
_FRONT_PLACE_STANDOFF = 0.13
#: Base standoff in front of the shelf when placing.
_FRONT_PLACE_BASE_DISTANCE = 0.8

#: Extra BiRRT path shortcutting for base plans. The default (50) leaves visible detours; more
#: shortcutting straightens the base's route to a near-direct line to the target.
_BASE_SMOOTH_AMT = 300


def top_down_grasp_transform(half_z: float) -> Pose:
    """Top-down grasp transform: near the object top for a tall object, stock offset for
    a short one.

    Shared by the top-down pick and the refiner's F1 probe.
    """
    grasp_z = max(_STOCK_GRASP_Z, float(half_z) - _GRIP_INSET)
    return Pose((0.0, 0.0, grasp_z), GRASP_TRANSFORM_TO_OBJECT.orientation)


def front_grasp_transform(half_z: float) -> Pose:
    """Front (45deg) grasp transform, landing the EE at a fixed reachable height (grip
    lower on a tall block).

    Shared by the front pick and the refiner's F1 probe.
    """
    half_z = float(half_z)
    offset = _FRONT_GRASP_TARGET_EE_Z - half_z
    lim = max(0.0, half_z - _FRONT_GRIP_MARGIN)
    return Pose(
        (0.0, _FRONT_GRASP_Y_OFFSET, float(np.clip(offset, -lim, lim))),
        _FRONT_GRASP_QUAT,
    )


class RestockFrontPickController(GroundPickController):
    """Front grasp of an upright block (diagonal approach, fingers on +/-x faces).

    Self-contained ``step()`` (does not need the ``GroundPickController`` grasp-transform hook). The
    grasp z-offset is adaptive: it grasps ``half_z - _FRONT_GRIP_INSET`` above the block centre, i.e.
    near its top, valid for any block height.
    """

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x
        distance = rng.uniform(*_FRONT_PICK_DISTANCE_BOUNDS)
        rot = rng.uniform(*_FRONT_PICK_ROT_BOUNDS)
        return np.array([distance, rot])

    def _front_grasp_transform(self) -> Pose:
        half_z = self._current_state.get_object_half_extents(self.objects[1].name)[2]
        return front_grasp_transform(half_z)

    def _front_plan_grasp_approach(self, target_end_effector_pose: Pose):
        """Front-grasp approach: MP to a standoff, then a straight-line reach-in."""
        arm = self._sim.robot.arm
        # Include the robot's OWN base so the arm cannot fold through it (the env collision set is
        # only shelf + movables; the base is a separate body nothing else checks).
        collision_ids = _arm_collision_ids(self._sim)
        cube_id = self._sim._object_name_to_pybullet_id(self.objects[1].name)
        # Pre-grasp standoff: back off along the tool approach axis (tool -z).
        standoff = multiply_poses(
            target_end_effector_pose, Pose((0.0, 0.0, -_FRONT_PICK_STANDOFF))
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
        # Straight-line reach standoff -> grasp, ignoring the target (we mean to touch it).
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
            plan1 + plan2, arm, max_distance=self._sim.config.max_action_mag / 2
        )

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        sim = self._sim

        # Base motion plan to a standoff in front of the block.
        if self._current_plan is None:
            sim.set_state(self._current_state)
            target_pose = self._current_state.get_object_pose(
                self.objects[1].name
            ).to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, self._current_params[0], self._current_params[1]
            )
            base_plan = get_base_plan(
                sim,
                target_base_pose,
                _base_nav_collision_ids(
                    sim, self._current_state, frozenset({self.objects[1].name})
                ),
                None,
                None,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        # Phase: navigate the base.
        if not self._navigated:
            target_base_pose = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
                self._navigated = True
            delta = target_base_pose - self._current_state.base_pose
            return np.array(
                [delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32
            )

        # Phase: reach to the front-grasp pose (standoff + straight-line reach).
        if self._navigated and not self._pre_grasp:
            if self._current_arm_joint_plan is None:
                sim.set_state(self._current_state)
                target_grasp_pose_world = self._current_state.get_object_pose(
                    self.objects[1].name
                )
                target_end_effector_pose = multiply_poses(
                    target_grasp_pose_world, self._front_grasp_transform()
                )
                joint_plan = self._front_plan_grasp_approach(target_end_effector_pose)
                self._current_arm_joint_plan = joint_plan[1:]
            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._pre_grasp = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)

        # Phase: close the gripper.
        if self._pre_grasp and not self._closed_gripper:
            g = self._get_current_robot_gripper_pose()
            if g > GRIPPER_CLOSE_THRESHOLD and np.isclose(
                g, self._last_gripper_state, atol=0.02
            ):
                self._closed_gripper = True
            self._last_gripper_state = g
            return np.array([0.0] * 10 + [-1.0], dtype=np.float32)

        # Phase: retract to HOME carrying the block.
        if self._closed_gripper and not self._lifted:
            if self._current_retract_plan is None:
                sim.set_state(self._current_state)
                grasped_id = sim._grasped_object_id
                grasped_tf = sim._grasped_object_transform
                joint_plan = run_motion_planning(
                    sim.robot.arm,
                    initial_positions=sim.robot.arm.get_joint_positions(),
                    target_positions=HOME_JOINT_POSITIONS.tolist(),
                    collision_bodies=_arm_collision_ids(sim) - {grasped_id},
                    seed=0,
                    physics_client_id=sim.physics_client_id,
                    held_object=grasped_id,
                    base_link_to_held_obj=grasped_tf,
                )
                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    sim.robot.arm,
                    max_distance=sim.config.max_action_mag / 2,
                )
                self._current_retract_plan = joint_plan[1:]
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)

        raise ValueError("Invalid state")


class RestockFrontPlaceController(BasePlaceController):
    """Translate-only place of an upright block into a region (keeps its orientation fixed).

    The EE pose is derived from the ACTUAL grasp so the object is *translated*, not rotated:
    ``ee = desired_object_pose(upright) . grasp^-1``. The pre-place standoff backs off along the
    grasp approach axis so the reach-in settles from above-front. Real MP inside ``pre_place`` is
    where F2 (resident collision) and F3 (ceiling collision) surface.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        sim,
        region_infos: dict[str, RegionInfo],
    ) -> None:
        super().__init__(objects, sim)
        self._region_infos = region_infos

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x
        return rng.uniform(-_PLACE_JITTER, _PLACE_JITTER, size=2)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Kinematic3DObjectCentricState)

        if self._current_plan is None:
            self._sim.set_state(self._current_state)
            grasped_object_id = self._sim._grasped_object_id
            grasped_object_transform = self._sim._grasped_object_transform
            assert grasped_object_transform is not None

            info = self._region_infos[self.objects[2].name]
            rx, ry = info.center_xy
            px, py = float(self._current_params[0]), float(self._current_params[1])
            half = self._current_state.get_object_half_extents(self.objects[1].name)
            # Rest the UPRIGHT object on the section surface: surface + z half-extent + a small pad
            # (within the env's ``min_placement_dist`` so it is released, not penetrating).
            desired_object_z = info.surface_z + float(half[2]) + _PLACE_Z_PAD
            desired_object_pose = Pose(
                (rx + px, ry + py, desired_object_z), (0.0, 0.0, 0.0, 1.0)
            )
            # Translate-only: derive the full EE pose from the grasp (position AND orientation).
            self._target_place_pose_world = multiply_poses(
                desired_object_pose, grasped_object_transform.invert()
            )
            # Pre-place standoff: back off along the grasp approach axis (tool -z).
            self._pre_place_pose_world = multiply_poses(
                self._target_place_pose_world, Pose((0.0, 0.0, -_FRONT_PLACE_STANDOFF))
            )

            self._target_place_pose_se2 = SE2Pose(rx + px, ry + py, 0.0)
            target_base_pose = get_target_robot_pose_from_parameters(
                self._target_place_pose_se2, _FRONT_PLACE_BASE_DISTANCE, np.pi / 2
            )
            # Base nav routes around floor movables (minus the carried block); the carried block
            # still must clear the shelf (via held_object/held_tf).
            base_plan = get_base_plan(
                self._sim,
                target_base_pose,
                _base_nav_collision_ids(
                    self._sim, self._current_state, frozenset({self.objects[1].name})
                ),
                grasped_object_id,
                grasped_object_transform,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        reach_ids = _place_reach_collision_ids(self._sim, self._current_state)
        if not self._navigated:
            return self.navigate()
        if self._navigated and not self._pre_place:
            return self.pre_place(collision_ids=reach_ids)
        if self._pre_place and not self._opened_gripper:
            return self.open_gripper()
        if self._opened_gripper and not self._lifted:
            return self.lift(collision_ids=reach_ids)
        raise ValueError("Invalid state")


# ==========================================================================================
# Buffer place: relocate an F1 clutter cube to a free floor buffer spot (FRONT translate-only place
# onto the floor, symmetric to the unified front pick). Clears a target's obstructed grasp so it can
# be picked; the buffered cube is abstracted OnBuffer (off every grasp), not Stored.
# ==========================================================================================

#: Pre-place back-off along the grasp approach axis (tool -z); mirrors the front shelf place.
_BUFFER_PLACE_STANDOFF = _FRONT_PLACE_STANDOFF
#: Base standoff for the floor buffer place. Must match the FRONT-grasp envelope (>= 0.70): the
#: closer top-down envelope (~0.525) folds the arm into its own base, so the place MP fails on a
#: front-grasped cube. Same distance as the front floor pick / front shelf place.
_BUFFER_PLACE_BASE_DISTANCE = float(np.mean(_FRONT_PICK_DISTANCE_BOUNDS))


class BufferPlaceController(BasePlaceController):
    """Top-down place of a (top-down-grasped) cube onto a free floor buffer spot.

    Mirrors :class:`RestockFrontPlaceController`'s translate-only place (``ee = desired_object_pose .
    grasp^-1``) but rests the cube FLAT on the floor at a chosen buffer spot instead of a shelf
    region -- and because the cube was grasped TOP-DOWN, the derived EE pose places it top-down from
    above. Registered for the ``PlaceBuffer(robot, target)`` operator (2 params, no region).
    """

    def __init__(self, objects: Sequence[Object], sim) -> None:
        # BasePlaceController unpacks (robot, target, target_table); the buffer place has no table
        # (it rests on the floor), so pass the target as an unused stand-in -- step() computes the
        # buffer pose directly and never reads a target table.
        robot, target = objects
        super().__init__([robot, target, target], sim)

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x
        return rng.uniform(-_PLACE_JITTER, _PLACE_JITTER, size=2)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def _free_buffer_spot(self, state: ObjectCentricState) -> tuple[float, float]:
        """First buffer spot not already occupied by another movable (>= 0.12 m
        away)."""
        occupied = []
        for name in self._sim.movable_names():
            if name == self._target.name:
                continue
            pos = state.get_object_pose(name).position
            occupied.append((pos[0], pos[1]))
        for bx, by in BUFFER_SPOTS:
            if all((bx - ox) ** 2 + (by - oy) ** 2 > 0.12**2 for ox, oy in occupied):
                return bx, by
        return BUFFER_SPOTS[0]

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Kinematic3DObjectCentricState)

        if self._current_plan is None:
            self._sim.set_state(self._current_state)
            grasped_object_id = self._sim._grasped_object_id
            grasped_object_transform = self._sim._grasped_object_transform
            assert grasped_object_transform is not None

            bx, by = self._free_buffer_spot(self._current_state)
            px, py = float(self._current_params[0]), float(self._current_params[1])
            half = self._current_state.get_object_half_extents(self._target.name)
            # Rest the cube flat on the FLOOR at the buffer spot; a cube is symmetric -> identity.
            desired_object_z = float(half[2]) + _PLACE_Z_PAD
            desired_object_pose = Pose(
                (bx + px, by + py, desired_object_z), (0.0, 0.0, 0.0, 1.0)
            )
            # Translate-only from the ACTUAL (top-down) grasp -> the EE places top-down from above.
            self._target_place_pose_world = multiply_poses(
                desired_object_pose, grasped_object_transform.invert()
            )
            self._pre_place_pose_world = multiply_poses(
                self._target_place_pose_world, Pose((0.0, 0.0, -_BUFFER_PLACE_STANDOFF))
            )
            self._target_place_pose_se2 = SE2Pose(bx + px, by + py, 0.0)
            target_base_pose = get_target_robot_pose_from_parameters(
                self._target_place_pose_se2, _BUFFER_PLACE_BASE_DISTANCE, np.pi / 2
            )
            base_plan = get_base_plan(
                self._sim,
                target_base_pose,
                _base_nav_collision_ids(
                    self._sim, self._current_state, frozenset({self._target.name})
                ),
                grasped_object_id,
                grasped_object_transform,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        # Avoid all env bodies except the held cube during the (empty-buffer-spot) descent.
        reach_ids = _arm_collision_ids(self._sim) - {self._sim._grasped_object_id}
        if not self._navigated:
            return self.navigate()
        if self._navigated and not self._pre_place:
            return self.pre_place(collision_ids=reach_ids)
        if self._pre_place and not self._opened_gripper:
            return self.open_gripper()
        # lift() already motion-plans the (now empty) arm back to HOME, so no separate retract phase
        # is needed -- the same terminal phase the region place uses, leaving the next pick a clean
        # start (arm home, gripper open, nothing grasped).
        if self._opened_gripper and not self._lifted:
            return self.lift(collision_ids=reach_ids)
        raise ValueError("Invalid state")


# ==========================================================================================
# Height-adaptive dispatchers: the pipeline has ONE Pick and ONE Place operator, but a short cube
# and a tall block need different skills. A cube is grasped TOP-DOWN and placed with the analytic
# front-orientation place (a top-down-grasped cube cannot be placed upright into a ceilinged cell,
# and a cube is symmetric so its final orientation is free). A tall block is grasped from the FRONT
# and placed translate-only so it stays upright (the whole F3 mechanism). The dispatchers pick the
# concrete controller from the target's height once it is known, and delegate everything to it.
# ==========================================================================================

#: A movable with z half-extent >= this is a "tall block" (front grasp + translate place); below it
#: is a "cube" (top-down grasp + analytic place). Cube half_z 0.025, block half_z 0.12.
_FRONT_GRASP_MIN_HALF_Z = 0.08


def _target_uses_front(state: ObjectCentricState, target_name: str) -> bool:
    return (
        float(state.get_object_half_extents(target_name)[2]) >= _FRONT_GRASP_MIN_HALF_Z
    )


class RestockAdaptivePickController(GroundPickController):
    """Front grasp for tall blocks, top-down grasp for cubes (chosen from the target
    height)."""

    def __init__(self, objects: Sequence[Object], sim) -> None:
        super().__init__(objects, sim)
        self._inner = None

    def _select(self, x: ObjectCentricState):
        del x  # grasp style no longer depends on object height
        if self._inner is None:
            # Unified FRONT grasp for every pick (cubes + tall blocks). The front grasp parks the
            # base SOUTH of the target facing +y and reaches north, which is what lets inter-target
            # base motion stay lateral (the fully-lateral layout). The front grip height still
            # adapts per object height inside ``front_grasp_transform``. (Top-down ``RestockPick
            # Controller`` is retired from the dispatch but kept for reference.)
            self._inner = RestockFrontPickController(self.objects, self._sim)
        return self._inner

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        return self._select(x).sample_parameters(x, rng)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._select(x).reset(x, params)

    def step(self) -> np.ndarray:
        return self._inner.step()

    def observe(self, x: ObjectCentricState) -> None:
        self._inner.observe(x)

    def terminated(self) -> bool:
        return self._inner.terminated()


class RegionAdaptivePlaceController(BasePlaceController):
    """Translate-only place for EVERY object (tall blocks + cubes).

    Both are now FRONT-grasped (Gate A), so the translate-only place derives the EE from the actual
    grasp and preserves the object's world orientation floor→shelf — a cube that was axis-aligned on
    the floor lands **upright** on the shelf. (The analytic ``RegionPlaceController`` reoriented the
    object by ``R_place @ R_grasp``; calibrated for a top-down grasp, it leaks the front grasp's 45°
    into the symmetric cube and lands it tilted -- decisions/07 2026-08-17. Kept for reference, no
    longer dispatched.)
    """

    def __init__(
        self, objects: Sequence[Object], sim, region_infos: dict[str, RegionInfo]
    ) -> None:
        super().__init__(objects, sim)
        self._region_infos = region_infos
        self._inner = None

    def _select(self, x: ObjectCentricState):
        del x  # place style no longer depends on object height (all front-grasped)
        if self._inner is None:
            self._inner = RestockFrontPlaceController(
                self.objects, self._sim, self._region_infos
            )
        return self._inner

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        return self._select(x).sample_parameters(x, rng)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._select(x).reset(x, params)

    def step(self) -> np.ndarray:
        return self._inner.step()

    def observe(self, x: ObjectCentricState) -> None:
        self._inner.observe(x)

    def terminated(self) -> bool:
        return self._inner.terminated()
