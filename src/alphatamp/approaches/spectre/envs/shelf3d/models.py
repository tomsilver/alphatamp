"""Bilevel-planning models for the ShelfObstruct3D obstruction variant.

Mirrors kinder's ``tidybot3d_shelf3D`` factory, but adds per-region occupancy predicates
(``At`` / ``Clear``) and the clear-then-place operator set that forces the planner to relocate
obstructing blockers before placing targets. Regions are symbolic objects (a new ``region``
type); their world centres come from ``region_geometry`` and drive both the abstractor and the
place-to-region controller. The pick-from-shelf and place-to-region skills are the custom
controllers in this package; the ground-target pick reuses kinder's ``PickShelfController``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium
import kinder
import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Box, Space
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder_models.dynamic3d.shelf.parameterized_skills import (
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    MOVE_TO_TARGET_ROT_BOUNDS,
    PickShelfController,
    PyBulletSim,
)
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Type,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from .pick_from_shelf import PickFromShelfController
from .place_to_shelf import PlaceToShelfRegionController
from .region_geometry import load_region_infos, shelf_surface_z

# Types.
RobotType = MujocoTidyBotRobotObjectType
CubeType = MujocoMovableObjectType
RegionType = Type("region")

# Predicates.
HandEmpty = Predicate("HandEmpty", [RobotType])
Holding = Predicate("Holding", [RobotType, CubeType])
OnGround = Predicate("OnGround", [CubeType])
At = Predicate("At", [CubeType, RegionType])
Clear = Predicate("Clear", [RegionType])

# Tolerances.
_ON_SHELF_Z = 0.3  # a cube above this height is on a shelf, not the ground
# The At radius must be *smaller* than the cube collision distance (~2*half = 0.07) for the
# obstruction failure mode to exist at all: a cube close enough to physically block a placement
# but farther than _AT_XY_TOL from the region centre reads as "region Clear" symbolically while
# colliding geometrically -- the class-1 culprit. Placements land within ~0.01 of the centre, so
# 0.05 still marks a correctly-placed cube At its region.
_AT_XY_TOL = 0.05  # a cube within this of a region centre occupies it
_HOLDING_TOL = 0.05
_HANDEMPTY_TOL = 1e-3
_GRASP_THRESHOLD = 0.1


class ObstructionAbstractor:
    """State/goal abstractor with per-region ``At`` / ``Clear`` occupancy."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv, task_json_path: str) -> None:
        initial_state, _ = sim.reset()
        self._pybullet_sim = PyBulletSim(initial_state, rendering=False)
        self._robot_name = sim.robot_name
        self._task_json_path = task_json_path
        self._region_infos = load_region_infos(task_json_path, initial_state)
        self._region_objs = {
            name: Object(name, RegionType) for name in self._region_infos
        }

    # -- geometry helpers -------------------------------------------------
    def region_centers(self) -> dict[str, tuple[float, float]]:
        """World xy of every region (for the place controller)."""
        return {n: r.center_xy for n, r in self._region_infos.items()}

    def shelf_surface_z(self, state: ObjectCentricState) -> float:
        """Shelf surface height, read off any cube currently on a shelf."""
        for cube in state.get_objects(CubeType):
            if state.get(cube, "z") > _ON_SHELF_Z:
                return shelf_surface_z(state, cube)
        return 0.55  # fallback (no cube on a shelf yet)

    # -- abstraction ------------------------------------------------------
    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        atoms: set[GroundAtom] = set()
        self._pybullet_sim.set_state(state)
        robot = state.get_object_from_name(self._robot_name)
        movables = list(state.get_objects(CubeType))

        gripper_val = state.get(robot, "pos_gripper")
        if np.isclose(gripper_val, 0.0, atol=_HANDEMPTY_TOL):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        held: set[str] = set()
        if gripper_val > _GRASP_THRESHOLD:
            ee = self._pybullet_sim.get_ee_pose()
            for cube in movables:
                if state.get(cube, "z") > 0.1 and all(
                    abs(ee.position[i] - state.get(cube, ax)) < _HOLDING_TOL
                    for i, ax in enumerate(("x", "y", "z"))
                ):
                    atoms.add(GroundAtom(Holding, [robot, cube]))
                    held.add(cube.name)

        occupied: set[str] = set()
        for cube in movables:
            if cube.name in held:
                continue
            z = state.get(cube, "z")
            if z <= _ON_SHELF_Z:
                if np.isclose(z - state.get(cube, "bb_z") / 2, 0.0, atol=0.05):
                    atoms.add(GroundAtom(OnGround, [cube]))
                continue
            # On a shelf: assign to the nearest region within tolerance.
            cx, cy = state.get(cube, "x"), state.get(cube, "y")
            best, best_d = None, _AT_XY_TOL
            for name, info in self._region_infos.items():
                d = float(np.hypot(cx - info.center_xy[0], cy - info.center_xy[1]))
                if d < best_d:
                    best, best_d = name, d
            if best is not None:
                atoms.add(GroundAtom(At, [cube, self._region_objs[best]]))
                occupied.add(best)

        for name, region_obj in self._region_objs.items():
            if name not in occupied:
                atoms.add(GroundAtom(Clear, [region_obj]))

        objects = {robot} | set(movables) | set(self._region_objs.values())
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """Each target cube must reach its matching target region (by index suffix)."""
        atoms: set[GroundAtom] = set()
        for name, info in self._region_infos.items():
            if not info.is_target:
                continue
            idx = name.rsplit("_", 1)[-1]  # target_region_<idx>
            target_name = f"cube_target{idx}"
            if target_name in state.get_object_names():
                target = state.get_object_from_name(target_name)
                atoms.add(GroundAtom(At, [target, self._region_objs[name]]))
        return RelationalAbstractGoal(atoms, self.state_abstractor)


def create_obstruction_models(
    observation_space: Space,
    action_space: Space,
    task_json_path: str,
    num_objects: int = 1,
) -> SesameModels:
    """Create the SesameModels for a ShelfObstruct3D task."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    sim = ObjectCentricTidyBot3DEnv(
        task_config_path=task_json_path,
        num_objects=num_objects,
        allow_state_access=True,
    )
    abstractor = ObstructionAbstractor(sim, task_json_path)
    initial_state, _ = sim.reset()
    region_centers = abstractor.region_centers()
    surface_z = abstractor.shelf_surface_z(initial_state)

    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        return observation_space.devectorize(o)

    # Refinement rolls out on the gym TidyBot3DEnv, not the ObjectCentric sim: the shelf grasp
    # controllers are physically stable under the gym env's continuous stepping, whereas the
    # ObjectCentric sim's set_state-per-step rollout drops a cube resting on a thin shelf (the
    # contact solver's warm-start is lost). To keep stepping continuous we set_state only when x
    # is NOT the state we returned last (a fresh rollout or a backtrack); the sampler chains
    # x = transition_fn(x, u) with the same object, so consecutive steps run without re-syncing.
    _eid = f"kinder/ShelfObstruct3D-trans-{Path(task_json_path).stem}-v0"
    if _eid not in gymnasium.registry:
        gymnasium.register(
            id=_eid,
            entry_point="kinder.envs.dynamic3d.envs:TidyBot3DEnv",
            kwargs={
                "task_config_path": task_json_path,
                "scene_render_camera": "task_view",
            },
        )
    trans_env = kinder.make(_eid, render_mode="rgb_array", allow_state_access=True)
    trans_env.reset(seed=0)
    _last_returned: list[ObjectCentricState | None] = [None]

    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        if _last_returned[0] is not x:
            trans_env.unwrapped.set_state(  # type: ignore[attr-defined]
                observation_space.vectorize(x)
            )
        obs, _, _, _, _ = trans_env.step(u)
        nx = observation_to_state(obs)
        _last_returned[0] = nx
        return nx

    types = {RobotType, MujocoObjectType, MujocoFixtureObjectType, CubeType, RegionType}
    state_space = ObjectCentricStateSpace(
        {RobotType, MujocoObjectType, MujocoFixtureObjectType, CubeType}
    )
    predicates = {HandEmpty, Holding, OnGround, At, Clear}

    robot = Variable("?robot", RobotType)
    target = Variable("?target", CubeType)
    blocker = Variable("?blocker", CubeType)
    region = Variable("?region", RegionType)

    PickTargetOperator = LiftedOperator(
        "pick_target",
        [robot, target],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [target]),
        },
    )
    PickBlockerOperator = LiftedOperator(
        "pick_blocker",
        [robot, blocker, region],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(At, [blocker, region]),
        },
        add_effects={
            LiftedAtom(Holding, [robot, blocker]),
            LiftedAtom(Clear, [region]),
        },
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(At, [blocker, region]),
        },
    )
    PlaceOperator = LiftedOperator(
        "place",
        [robot, target, region],
        preconditions={
            LiftedAtom(Holding, [robot, target]),
            LiftedAtom(Clear, [region]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(At, [target, region]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, target]),
            LiftedAtom(Clear, [region]),
        },
    )

    pybullet_sim = PyBulletSim(initial_state, rendering=False)

    class GroundPickController(PickShelfController):
        def __init__(self, objects):  # type: ignore[no-untyped-def]
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class ShelfPickController(PickFromShelfController):
        def __init__(self, objects):  # type: ignore[no-untyped-def]
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class RegionPlaceController(PlaceToShelfRegionController):
        def __init__(self, objects):  # type: ignore[no-untyped-def]
            super().__init__(
                pybullet_sim=pybullet_sim,
                objects=objects,
                region_centers=region_centers,
                shelf_surface_z=surface_z,
            )

    jitter_space = Box(
        low=np.array([-0.02], dtype=np.float32),
        high=np.array([0.02], dtype=np.float32),
        dtype=np.float32,
    )
    ground_pick_space = Box(
        low=np.array(
            [MOVE_TO_TARGET_DISTANCE_BOUNDS[0], MOVE_TO_TARGET_ROT_BOUNDS[0]],
            dtype=np.float32,
        ),
        high=np.array(
            [MOVE_TO_TARGET_DISTANCE_BOUNDS[1], MOVE_TO_TARGET_ROT_BOUNDS[1]],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    LiftedGroundPick: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target], GroundPickController, params_space=ground_pick_space
    )
    LiftedShelfPick: LiftedParameterizedController = LiftedParameterizedController(
        [robot, blocker, region], ShelfPickController, params_space=jitter_space
    )
    LiftedRegionPlace: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, region], RegionPlaceController, params_space=jitter_space
    )

    skills = {
        LiftedSkill(PickTargetOperator, LiftedGroundPick),
        LiftedSkill(PickBlockerOperator, LiftedShelfPick),
        LiftedSkill(PlaceOperator, LiftedRegionPlace),
    }

    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        abstractor.state_abstractor,
        abstractor.goal_deriver,
        skills,
    )
