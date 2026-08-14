"""Bilevel-planning models for Restock3D.

Adapts ``envs/shelf3d/models.py`` to the restock domain: floor objects are stored into
single-object shelf regions. The deliberate difference from ShelfObstruct3D is the operator set —
``Place(robot, obj, region)`` has **no ``Clear`` precondition**, so region capacity is invisible
to the planner (proposal §2.1). A height-blind / capacity-blind planner therefore emits many
goal-reaching skeletons that over-assign a region (F2) or place a tall object under a short cell
(F3); those fail the geometric feasibility gate at refinement time (``instrumented_refiner``),
producing the false positives an oracle avoids.

Regions are symbolic objects (a ``region`` type); their world geometry comes from
``region_geometry``. Ground pick reuses kinder's ``PickShelfController``; region place reuses
``envs/shelf3d/place_to_shelf.PlaceToShelfRegionController`` (already region-parameterized).
"""

from __future__ import annotations

import json
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

from ..shelf3d.place_to_shelf import PlaceToShelfRegionController
from .region_geometry import RegionInfo, load_region_infos

# Types.
RobotType = MujocoTidyBotRobotObjectType
CubeType = MujocoMovableObjectType
RegionType = Type("region")

# Predicates.
HandEmpty = Predicate("HandEmpty", [RobotType])
Holding = Predicate("Holding", [RobotType, CubeType])
OnFloor = Predicate("OnFloor", [CubeType])
InRegion = Predicate("InRegion", [CubeType, RegionType])
Stored = Predicate("Stored", [CubeType])

# Tolerances.
_FLOOR_Z_TOL = 0.1  # a cube resting this near the ground (and in no region) is OnFloor
_SURFACE_Z_TOL = (
    0.08  # a cube rests in a region iff its base is this near the region's surface_z
)
_HANDEMPTY_TOL = 1e-3
_GRASP_THRESHOLD = 0.1
_INREGION_MARGIN = (
    0.02  # slack added to the region footprint for the InRegion containment test
)


class RestockAbstractor:
    """State/goal abstractor: HandEmpty/Holding/OnFloor/InRegion/Stored.

    ``InRegion(cube, region)`` holds when a shelf cube's world xy is inside the region
    footprint (plus a small margin); ``Stored(cube)`` is derived from any InRegion.
    Region *capacity* is not represented — the abstraction sees no reason a second cube
    cannot join a region, which is exactly where the false positives come from.
    """

    def __init__(self, sim: ObjectCentricTidyBot3DEnv, task_json_path: str) -> None:
        initial_state, _ = sim.reset()
        self._robot_name = sim.robot_name
        self._task_json_path = task_json_path
        self._region_infos: dict[str, RegionInfo] = load_region_infos(
            task_json_path, initial_state
        )
        self._region_objs = {
            name: Object(name, RegionType) for name in self._region_infos
        }
        with open(task_json_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Goal objects come from a top-level ``goal_objects`` list. The JSON ``goal_state`` is
        # kept EMPTY so kinder's own ``_check_goals`` (which only knows built-in predicates like
        # ``on``) does not choke on our ``Stored`` goal — refinement uses this goal_deriver, not
        # the gym env's terminated signal.
        self._goal_object_names: list[str] = list(cfg.get("goal_objects", []))

    # -- geometry helpers -------------------------------------------------
    def region_infos(self) -> dict[str, RegionInfo]:
        return self._region_infos

    def region_centers(self) -> dict[str, tuple[float, float]]:
        return {n: r.center_xy for n, r in self._region_infos.items()}

    def region_objs(self) -> dict[str, Object]:
        return self._region_objs

    def goal_object_names(self) -> list[str]:
        return list(self._goal_object_names)

    def shelf_surface_z(self, state: ObjectCentricState) -> float:
        """A representative shelf surface height (fallback = first region's surface_z)."""
        if self._region_infos:
            return next(iter(self._region_infos.values())).surface_z
        for cube in state.get_objects(CubeType):
            if state.get(cube, "z") > 0.3:  # a cube this high is resting on a shelf
                return float(state.get(cube, "z") - state.get(cube, "bb_z") / 2)
        return 0.55

    def _region_of(self, state: ObjectCentricState, cube: Object) -> str | None:
        """The region whose footprint contains the cube's world xy AND whose shelf
        surface the cube is resting on.

        The surface-z match disambiguates the vertically-stacked Config B cells (a tall
        cell and a short cell share the same xy footprint at different heights); xy
        alone would assign a short-cell cube to the tall region below it.
        """
        cx, cy = state.get(cube, "x"), state.get(cube, "y")
        rest_z = state.get(cube, "z") - state.get(cube, "bb_z") / 2
        best, best_slack = None, 1e9
        for name, info in self._region_infos.items():
            if abs(rest_z - info.surface_z) > _SURFACE_Z_TOL:
                continue  # not resting on this region's shelf (wrong cell of a stacked pair)
            dx = abs(cx - info.center_xy[0]) - (info.half_xy[0] + _INREGION_MARGIN)
            dy = abs(cy - info.center_xy[1]) - (info.half_xy[1] + _INREGION_MARGIN)
            if dx <= 0 and dy <= 0:
                slack = max(dx, dy)  # most-interior region wins ties
                if slack < best_slack:
                    best, best_slack = name, slack
        return best

    # -- abstraction ------------------------------------------------------
    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        atoms: set[GroundAtom] = set()
        robot = state.get_object_from_name(self._robot_name)
        movables = list(state.get_objects(CubeType))

        gripper_val = state.get(robot, "pos_gripper")
        gripper_closed = gripper_val > _GRASP_THRESHOLD
        if np.isclose(gripper_val, 0.0, atol=_HANDEMPTY_TOL):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # Each cube is InRegion (resting in a shelf region: xy + surface-z match), OnFloor (resting
        # near the ground in no region), or — if lifted with the gripper closed — Holding.
        # Position-agnostic Holding works for both a physics pick (held cube lifted near the EE)
        # and a geometric pick (cube teleported to a lifted pose).
        for cube in movables:
            region = self._region_of(state, cube)
            if region is not None:
                atoms.add(GroundAtom(InRegion, [cube, self._region_objs[region]]))
                atoms.add(GroundAtom(Stored, [cube]))
                continue
            rest_z = state.get(cube, "z") - state.get(cube, "bb_z") / 2
            if rest_z < _FLOOR_Z_TOL:
                atoms.add(GroundAtom(OnFloor, [cube]))
            elif gripper_closed:
                atoms.add(GroundAtom(Holding, [robot, cube]))

        objects = {robot} | set(movables) | set(self._region_objs.values())
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """Every goal object must be ``Stored`` (in some region — assignment is
        free)."""
        atoms: set[GroundAtom] = set()
        for name in self._goal_object_names:
            if name in state.get_object_names():
                atoms.add(GroundAtom(Stored, [state.get_object_from_name(name)]))
        return RelationalAbstractGoal(atoms, self.state_abstractor)


def create_restock3d_models(
    observation_space: Space,
    action_space: Space,
    task_json_path: str,
    num_objects: int,
) -> SesameModels:
    """Create the SesameModels for a Restock3D task."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    sim = ObjectCentricTidyBot3DEnv(
        task_config_path=task_json_path,
        num_objects=num_objects,
        allow_state_access=True,
    )
    abstractor = RestockAbstractor(sim, task_json_path)
    initial_state, _ = sim.reset()
    region_centers = abstractor.region_centers()
    surface_z = abstractor.shelf_surface_z(initial_state)

    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        return observation_space.devectorize(o)

    # Refinement rolls out on the gym TidyBot3DEnv (set_state-per-step drops cubes on thin
    # shelves in the ObjectCentric sim). set_state only on discontinuity; see shelf3d/models.py.
    _eid = f"kinder/Restock3D-trans-{Path(task_json_path).stem}-v0"
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
    predicates = {HandEmpty, Holding, OnFloor, InRegion, Stored}

    robot = Variable("?robot", RobotType)
    obj = Variable("?obj", CubeType)
    region = Variable("?region", RegionType)

    PickOperator = LiftedOperator(
        "pick",
        [robot, obj],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnFloor, [obj])},
        add_effects={LiftedAtom(Holding, [robot, obj])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnFloor, [obj])},
    )
    PlaceOperator = LiftedOperator(
        "place",
        [robot, obj, region],
        preconditions={
            LiftedAtom(Holding, [robot, obj])
        },  # NO Clear -> capacity invisible
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InRegion, [obj, region]),
            LiftedAtom(Stored, [obj]),
        },
        delete_effects={LiftedAtom(Holding, [robot, obj])},
    )

    pybullet_sim = PyBulletSim(initial_state, rendering=False)

    class GroundPickController(PickShelfController):
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

    LiftedPick: LiftedParameterizedController = LiftedParameterizedController(
        [robot, obj], GroundPickController, params_space=ground_pick_space
    )
    LiftedPlace: LiftedParameterizedController = LiftedParameterizedController(
        [robot, obj, region], RegionPlaceController, params_space=jitter_space
    )

    skills = {
        LiftedSkill(PickOperator, LiftedPick),
        LiftedSkill(PlaceOperator, LiftedPlace),
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
