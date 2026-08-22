"""Bilevel-planning models for **Restock3D v2** (continuous-packing variant).

v2 reframes placement from discrete region assignment to **continuous geometric packing**:

* **Two place operators** — ``place_tall`` (tall/bottom section) and ``place_short`` (short/top
  section) — *both legal for either object size*. They have **identical abstract effects**
  (``add {HandEmpty, Stored}``), so the section choice lives only in the operator identity and is
  validated geometrically at refinement: a tall block via ``place_short`` overhangs the short
  section's ceiling board and is rejected by real PyBullet collision (F3).
* **No ``?region`` argument and no ``InRegion``.** Placement x is sampled uniformly across the
  section's continuous x-band (:mod:`place_controller_v2`); ``Stored`` is a purely geometric section
  membership (object underside near a section surface AND xy on the shelf band). Predicates are
  ``HandEmpty / Holding / OnFloor / Stored`` (+ inert ``OnBuffer`` for the retained-but-unused buffer
  machinery). Goal is ``Stored(o)`` for every goal object — unchanged from v1.

To keep the low-level env untouched (:mod:`kinematic_env`), the two shelf sections are the wide
:class:`region_geometry.RegionInfo` bands from :func:`section_geometry.compute_section_infos`; the
abstractor reuses v1's geometric section match (``_region_of``) over exactly those two bands.
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
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from .kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
    stratum_object_specs,
    stratum_pose_fn,
)
from .place_controller import in_buffer_zone
from .place_controller_v2 import create_lifted_controllers_v2
from .region_geometry import RegionInfo
from .section_geometry import compute_section_infos

# Types (no RegionType — v2 operators carry no region argument).
RobotType = Kinematic3DRobotType
CubeType = Kinematic3DCuboidType

# Predicates.
HandEmpty = Predicate("HandEmpty", [RobotType])
Holding = Predicate("Holding", [RobotType, CubeType])
OnFloor = Predicate("OnFloor", [CubeType])
Stored = Predicate("Stored", [CubeType])
# Inert (kept for the retained-but-unused F1-clutter buffer machinery; never in the goal, and
# clutter counts are 0 on every stratum). A cube relocated to the floor buffer is OnBuffer, not
# OnFloor, so Pick (precond OnFloor) will not re-pick it.
OnBuffer = Predicate("OnBuffer", [CubeType])

_GRIPPER_OPEN_THRESHOLD = 0.01
_FLOOR_Z_TOL = 0.1
_SURFACE_Z_TOL = 0.06
_SECTION_MARGIN = 0.02


class RestockAbstractorV2:
    """State/goal abstractor: HandEmpty / Holding / OnFloor / Stored (no InRegion).

    ``Stored(cube)`` holds when a shelf cube's underside rests near a SECTION surface
    (surface-z match disambiguates tall / short) AND its xy is inside the section band
    footprint — i.e. purely geometric section membership. Section *capacity* is not
    represented; that invisibility is where the continuous-packing false positives (an
    over-full section) come from.
    """

    def __init__(
        self, section_infos: dict[str, RegionInfo], goal_object_names: list[str]
    ) -> None:
        self._section_infos = section_infos
        self._goal_object_names = list(goal_object_names)

    def section_infos(self) -> dict[str, RegionInfo]:
        return self._section_infos

    def goal_object_names(self) -> list[str]:
        return list(self._goal_object_names)

    def _section_of(self, state: ObjectCentricState, cube) -> str | None:
        cx, cy = state.get(cube, "pose_x"), state.get(cube, "pose_y")
        rest_z = state.get(cube, "pose_z") - state.get(cube, "half_extent_z")
        best, best_slack = None, 1e9
        for name, info in self._section_infos.items():
            if abs(rest_z - info.surface_z) > _SURFACE_Z_TOL:
                continue
            dx = abs(cx - info.center_xy[0]) - (info.half_xy[0] + _SECTION_MARGIN)
            dy = abs(cy - info.center_xy[1]) - (info.half_xy[1] + _SECTION_MARGIN)
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
            if self._section_of(state, cube) is not None:
                atoms.add(GroundAtom(Stored, [cube]))
                continue
            rest_z = state.get(cube, "pose_z") - state.get(cube, "half_extent_z")
            if rest_z < _FLOOR_Z_TOL:
                cx, cy = state.get(cube, "pose_x"), state.get(cube, "pose_y")
                if in_buffer_zone(cx, cy):
                    atoms.add(GroundAtom(OnBuffer, [cube]))
                else:
                    atoms.add(GroundAtom(OnFloor, [cube]))

        objects = {robot} | set(movables)
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        atoms: set[GroundAtom] = set()
        names = set(state.get_object_names())
        for name in self._goal_object_names:
            if name in names:
                atoms.add(GroundAtom(Stored, [state.get_object_from_name(name)]))
        return RelationalAbstractGoal(atoms, self.state_abstractor)


@dataclass
class RestockModelsV2:
    """The SesameModels plus the internal sim + section_infos the recording sampler
    needs."""

    models: SesameModels
    sim: ObjectCentricRestock3DEnv
    section_infos: dict[str, RegionInfo]
    abstractor: RestockAbstractorV2


def stratum_env_args_v2(stratum: int, config: Restock3DEnvConfig | None = None):
    """The ``(object_specs, pose_fn, section_infos, config)`` tuple for a v2 collection
    stratum.

    Reuses v1's object specs + floor pose function (object counts only) and swaps the
    discrete ``compute_region_infos`` for the two wide :func:`compute_section_infos`
    bands.
    """
    if config is None:
        config = Restock3DEnvConfig()
    return (
        stratum_object_specs(stratum, config),
        stratum_pose_fn(stratum),
        compute_section_infos(config),
        config,
    )


def create_restock3d_v2_models(
    observation_space: Space,
    action_space: Space,
    stratum: int,
) -> RestockModelsV2:
    """Create the Restock3D v2 models bundle for a collection stratum."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    object_specs, pose_fn, section_infos, config = stratum_env_args_v2(stratum)
    sim = ObjectCentricRestock3DEnv(
        object_specs, pose_fn, section_infos, config=config, allow_state_access=True
    )
    goal_names = [
        n for n, _, _ in object_specs if n.startswith(("cube_goal", "block_goal"))
    ]
    return build_restock3d_v2_models(
        sim,
        section_infos,
        goal_names,
        observation_space,
        observation_space.devectorize,
        action_space,
    )


def build_restock3d_v2_models(
    sim: ObjectCentricRestock3DEnv,
    section_infos: dict[str, RegionInfo],
    goal_names: list[str],
    observation_space: Space,
    observation_to_state,
    action_space: Space,
    lifted_controllers_factory=None,
) -> RestockModelsV2:
    """Assemble the v2 models bundle from an already-built sim (used by the oracle +
    Stage-0).

    ``lifted_controllers_factory`` (default ``create_lifted_controllers_v2``) lets v3 inject its
    left-to-right packing controllers (``place_controller_v3.create_lifted_controllers_v3``) while
    reusing the identical operators/predicates/abstractor; omitting it is byte-identical to v2.
    """
    abstractor = RestockAbstractorV2(section_infos, goal_names)

    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        state = x.copy()
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {RobotType, CubeType}
    state_space = ObjectCentricStateSpace({RobotType, CubeType})
    predicates = {HandEmpty, Holding, OnFloor, Stored, OnBuffer}

    robot = Variable("?robot", RobotType)
    target = Variable("?target", CubeType)

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
    # place_tall / place_short: identical abstract effects (add {HandEmpty, Stored}); the section
    # choice is validated geometrically at refinement (place_short of a tall block -> F3 collision).
    PlaceTallOperator = LiftedOperator(
        "place_tall",
        [robot, target],
        preconditions={
            LiftedAtom(Holding, [robot, target])
        },  # NO capacity/height precond
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(Stored, [target])},
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )
    PlaceShortOperator = LiftedOperator(
        "place_short",
        [robot, target],
        preconditions={LiftedAtom(Holding, [robot, target])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(Stored, [target])},
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )
    # Inert relocation operator (kept for parity; clutter counts are 0 on every stratum).
    PlaceBufferOperator = LiftedOperator(
        "place_buffer",
        [robot, target],
        preconditions={LiftedAtom(Holding, [robot, target])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnBuffer, [target])},
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )

    factory = lifted_controllers_factory or create_lifted_controllers_v2
    lifted = factory(action_space, sim, section_infos)
    skills = {
        LiftedSkill(PickOperator, lifted["pick"]),
        LiftedSkill(PlaceTallOperator, lifted["place_tall"]),
        LiftedSkill(PlaceShortOperator, lifted["place_short"]),
        LiftedSkill(PlaceBufferOperator, lifted["place_buffer"]),
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
    return RestockModelsV2(models, sim, section_infos, abstractor)
