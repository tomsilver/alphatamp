"""Bilevel-planning models for the **kinematic** Restock3D.

Adapts the substrate kinematic shelf3d factory
(``kinder_bilevel_planning/env_models/kinematic3d/shelf3d.py``) to the restock domain: floor objects
are stored into single-object shelf regions. The deliberate difference is the operator set —
``Place(robot, obj, region)`` has **no ``Clear`` precondition**, so region capacity / cell height are
invisible to the planner; a height-/capacity-blind A* therefore emits many goal-reaching skeletons
that over-assign a region (F2) or send a tall block to a short shelf (F3). Those genuinely fail
refinement by real PyBullet collision (:mod:`instrumented_refiner`), producing the false positives an
oracle avoids.

Regions are symbolic objects (:data:`place_controller.RegionType`); their world geometry comes from
:mod:`region_geometry`. Ground pick reuses kinder's kinematic ``GroundPickController``; region place
uses :class:`place_controller.RegionPlaceController`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    Kinematic3DRobotActionSpace,
)
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from .kinematic_env import ObjectCentricRestock3DEnv, stratum_env_args
from .place_controller import RegionType, create_lifted_controllers
from .region_geometry import RegionInfo

# Types.
RobotType = Kinematic3DRobotType
CubeType = Kinematic3DCuboidType

# Predicates.
HandEmpty = Predicate("HandEmpty", [RobotType])
Holding = Predicate("Holding", [RobotType, CubeType])
OnFloor = Predicate("OnFloor", [CubeType])
InRegion = Predicate("InRegion", [CubeType, RegionType])
Stored = Predicate("Stored", [CubeType])

_GRIPPER_OPEN_THRESHOLD = 0.01
_FLOOR_Z_TOL = 0.1
_SURFACE_Z_TOL = 0.06
_INREGION_MARGIN = 0.02


class RestockAbstractor:
    """State/goal abstractor: HandEmpty / Holding / OnFloor / InRegion / Stored.

    ``InRegion(cube, region)`` holds when a shelf cube's world xy is inside the region footprint
    (plus a margin) AND its underside rests near the region's shelf surface (surface-z match
    disambiguates the tall / short shelves). Region *capacity* is not represented — that invisibility
    is where the false positives come from.
    """

    def __init__(
        self, region_infos: dict[str, RegionInfo], goal_object_names: list[str]
    ) -> None:
        self._region_infos = region_infos
        self._region_objs = {n: Object(n, RegionType) for n in region_infos}
        self._goal_object_names = list(goal_object_names)

    def region_infos(self) -> dict[str, RegionInfo]:
        return self._region_infos

    def region_objs(self) -> dict[str, Object]:
        return self._region_objs

    def goal_object_names(self) -> list[str]:
        return list(self._goal_object_names)

    def _region_of(self, state: ObjectCentricState, cube: Object) -> str | None:
        cx, cy = state.get(cube, "pose_x"), state.get(cube, "pose_y")
        rest_z = state.get(cube, "pose_z") - state.get(cube, "half_extent_z")
        best, best_slack = None, 1e9
        for name, info in self._region_infos.items():
            if abs(rest_z - info.surface_z) > _SURFACE_Z_TOL:
                continue
            dx = abs(cx - info.center_xy[0]) - (info.half_xy[0] + _INREGION_MARGIN)
            dy = abs(cy - info.center_xy[1]) - (info.half_xy[1] + _INREGION_MARGIN)
            if dx <= 0 and dy <= 0:
                slack = max(dx, dy)
                if slack < best_slack:
                    best, best_slack = name, slack
        return best

    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        assert isinstance(state, Kinematic3DObjectCentricState)
        atoms: set[GroundAtom] = set()
        robot = state.get_object_from_name("robot")
        movables = list(state.get_objects(CubeType))
        grasped = state.grasped_object

        if grasped is None and state.finger_state < _GRIPPER_OPEN_THRESHOLD:
            atoms.add(GroundAtom(HandEmpty, [robot]))

        for cube in movables:
            if grasped == cube.name:
                atoms.add(GroundAtom(Holding, [robot, cube]))
                continue
            region = self._region_of(state, cube)
            if region is not None:
                atoms.add(GroundAtom(InRegion, [cube, self._region_objs[region]]))
                atoms.add(GroundAtom(Stored, [cube]))
                continue
            rest_z = state.get(cube, "pose_z") - state.get(cube, "half_extent_z")
            if rest_z < _FLOOR_Z_TOL:
                atoms.add(GroundAtom(OnFloor, [cube]))

        objects = {robot} | set(movables) | set(self._region_objs.values())
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        atoms: set[GroundAtom] = set()
        names = set(state.get_object_names())
        for name in self._goal_object_names:
            if name in names:
                atoms.add(GroundAtom(Stored, [state.get_object_from_name(name)]))
        return RelationalAbstractGoal(atoms, self.state_abstractor)


@dataclass
class RestockModels:
    """The SesameModels plus the internal sim + region_infos the recording sampler needs."""

    models: SesameModels
    sim: ObjectCentricRestock3DEnv
    region_infos: dict[str, RegionInfo]
    abstractor: RestockAbstractor


def create_restock3d_models(
    observation_space: Space,
    action_space: Space,
    stratum: int,
) -> RestockModels:
    """Create the kinematic Restock3D models bundle for a collection stratum."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    object_specs, pose_fn, region_infos, config = stratum_env_args(stratum)
    sim = ObjectCentricRestock3DEnv(
        object_specs, pose_fn, region_infos, config=config, allow_state_access=True
    )
    goal_names = [
        n for n, _, _ in object_specs if n.startswith(("cube_goal", "block_goal"))
    ]
    return build_restock3d_models(
        sim,
        region_infos,
        goal_names,
        observation_space,
        observation_space.devectorize,
        action_space,
    )


def build_restock3d_models(
    sim: ObjectCentricRestock3DEnv,
    region_infos: dict[str, RegionInfo],
    goal_names: list[str],
    observation_space: Space,
    observation_to_state,
    action_space: Space,
) -> RestockModels:
    """Assemble the models bundle from an already-built sim (used by collection + Stage-0)."""
    abstractor = RestockAbstractor(region_infos, goal_names)

    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        state = x.copy()
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {RobotType, CubeType, RegionType}
    state_space = ObjectCentricStateSpace({RobotType, CubeType})
    predicates = {HandEmpty, Holding, OnFloor, InRegion, Stored}

    # Variable names must match the controllers' (LiftedSkill asserts equality).
    robot = Variable("?robot", RobotType)
    target = Variable("?target", CubeType)
    region = Variable("?region", RegionType)

    PickOperator = LiftedOperator(
        "pick",
        [robot, target],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnFloor, [target])},
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnFloor, [target]),
        },
    )
    PlaceOperator = LiftedOperator(
        "place",
        [robot, target, region],
        preconditions={LiftedAtom(Holding, [robot, target])},  # NO Clear
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InRegion, [target, region]),
            LiftedAtom(Stored, [target]),
        },
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )

    lifted = create_lifted_controllers(action_space, sim, region_infos)
    skills = {
        LiftedSkill(PickOperator, lifted["pick"]),
        LiftedSkill(PlaceOperator, lifted["place"]),
    }

    models = SesameModels(
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
    return RestockModels(models, sim, region_infos, abstractor)
