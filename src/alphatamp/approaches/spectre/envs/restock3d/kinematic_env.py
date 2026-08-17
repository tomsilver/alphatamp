"""Kinematic PyBullet Restock3D environment (single multi-section shelf).

A custom :class:`ObjectCentricKinematic3DRobotEnv` subclass: a floor staging area with
heterogeneous movable objects (small cubes + tall blocks + clutter) at **scripted**
poses, and a **single shelf** whose boards are placed to make a **tall section
(bottom)** and a **short section (top)**. A tall block fits under the tall section but,
kept upright by the front-grasp translate-only place, collides with the board capping
the short section (F3). Feasibility is decided by real PyBullet collision (the base env
reverts colliding moves; the pick/place controllers fail motion planning when no
collision-free solution exists), NOT by a hand-written geometric gate.

Regions are placement targets (metadata in :mod:`region_geometry`), not PyBullet bodies,
so they never look like solid shelf blocks. The shelf boards ARE solid collision bodies,
which is what makes the height-mismatch failure (F3) real.

The env is constructed from explicit ``object_specs`` + a ``pose_fn`` + ``region_infos``
so a Stage-0 probe can build arbitrary micro-scenes; :func:`stratum_env_args` builds
them for a collection stratum, and :class:`Restock3DEnv` is the constant-object gym
wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Type as TypingType

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DFixtureType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.shelf3d import Shelf3DObjectCentricState
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from .generator import STRATA, build_spec
from .region_geometry import (
    RegionInfo,
    board_center_zs,
    compute_region_infos,
    section_surfaces,
)

_SHELF_HEIGHT = 0.0127  # board thickness

# Clutter blocks per stratum -- **RETIRED, all 0** (decisions/07 2026-08-16). Under the unified FRONT
# grasp a floor neighbour never obstructs a grasp at the grasp config (verified by sweep: grasp_blockers
# empty for every +/-x/-y offset), so F1 grasp-obstruction clutter cannot be realised; the difficulty is
# the depth REACH-OVER among goals instead (the front grasp reaches north over anything nearer, so a back
# object is blocked until nearer ones are cleared -- naive order fails, south-to-north succeeds). The
# clutter object SPECS + the buffer/relocation machinery are kept inert (one flag away). This drives the
# object SPECS; generator._CLUTTER_PER_STRATUM drives their POSITIONS -- the two MUST match.
CLUTTER_PER_STRATUM: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}


@dataclass(frozen=True)
class Restock3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for :class:`ObjectCentricRestock3DEnv`."""

    max_action_mag: float = 0.2
    # Step-time base-collision enforcement is ON (fully-lateral layout, decisions/07 2026-08-16). The
    # object + buffer regions are disjoint x-bands to the -x of the shelf, and the front-grasp standoff
    # (~0.72 m) keeps the base SOUTH (y <= ~0.55) of every object (y >= ~0.60), so the base never has to
    # cross the object field to reach the shelf. The base motion planner routes laterally in that clear
    # southern corridor, so strict enforcement no longer collapses solvability (the ~0% collapse was the
    # old shelf-north layout where every place drove the base through the floor). The get_base_plan
    # shelf-only fallback is removed (Gate C) so a genuinely boxed base fails instead of phasing through.
    check_base_collisions: bool = True
    realistic_bg: bool = True  # a real room backdrop, not the blank void
    gripper_open_threshold: float = 0.01

    # One shelf: boards built at cumulative per-section gaps (custom builder; the stock
    # ``create_pybullet_shelf`` only does a scalar/uniform spacing).
    shelf_pose: Pose = Pose((0.4, 1.4, 0.0))  # (x, y) used for layout; z unused
    shelf_width: float = 0.60198
    shelf_depth: float = 0.254
    shelf_height: float = _SHELF_HEIGHT
    shelf_support_width: float = 0.0127  # side-wall / back-panel thickness
    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

    # Vertical layout: the bottom (tall) section's placement surface, then per-section clearances
    # (tall bottom, short top). A tall block (0.24) fits the 0.34 tall gap but is much taller than
    # the 0.15 short gap -> it collides the board capping the short section (F3), overhanging it by
    # ~0.09 so the mismatch is unmistakable. Short surface z is set by the TALL clearance, not this,
    # so a small short gap does not raise the (reachable) short placement surface.
    bottom_surface_z: float = 0.29
    section_clearances: tuple[float, float] = (0.34, 0.15)

    # Region layout (front strip along the shelf width).
    region_front_offset: float = 0.05  # region y = shelf_y - offset (front, robot side)
    region_pitch: float = 0.12
    region_half_x: float = 0.03
    region_half_y: float = 0.04

    # Object half-extents.
    small_half: tuple[float, float, float] = (0.025, 0.025, 0.025)
    tall_half: tuple[float, float, float] = (0.025, 0.025, 0.12)
    clutter_half: tuple[float, float, float] = (0.025, 0.025, 0.05)

    def get_camera_kwargs(self) -> dict:
        # Fully-lateral layout: frame the whole horizontal strip -- buffer band (x ~ -1.1), object
        # region (x ~ -0.5), and the shelf at (0.4, 1.4) -- centred on the strip midpoint, 3/4 view,
        # tilted down. Wider than the old forward-staging framing (decisions/07 2026-08-16).
        return {
            "camera_target": (-0.3, 0.85, 0.3),
            "camera_yaw": 55,
            "camera_distance": 3.4,
            "camera_pitch": -28,
        }


class Restock3DObjectCentricState(Shelf3DObjectCentricState):
    """A state in :class:`ObjectCentricRestock3DEnv`.

    Subclasses ``Shelf3DObjectCentricState`` so the stock kinematic
    ``GroundPickController`` (which type-asserts that class) can be reused unchanged.
    """


# A pose function maps a problem seed to a dict of movable name -> world (x, y).
PoseFn = Callable[[int], dict[str, tuple[float, float]]]
# An object spec: (name, half_extents, rgba).
ObjectSpec = tuple[str, tuple[float, float, float], tuple[float, float, float, float]]


class ObjectCentricRestock3DEnv(
    ObjectCentricKinematic3DRobotEnv[Kinematic3DObjectCentricState, Restock3DEnvConfig]
):
    """Kinematic Restock3D: floor objects stored into single-object shelf regions."""

    def __init__(
        self,
        object_specs: list[ObjectSpec],
        pose_fn: PoseFn,
        region_infos: dict[str, RegionInfo],
        config: Restock3DEnvConfig = Restock3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._object_specs = list(object_specs)
        self._pose_fn = pose_fn
        self._region_infos = region_infos
        self._problem_seed = 0

        # Create movable bodies (fixed set; poses reset per seed).
        self._movable_ids: dict[str, int] = {}
        self._half_extents: dict[str, tuple[float, float, float]] = {}
        for name, half, rgba in self._object_specs:
            self._movable_ids[name] = create_pybullet_block(
                rgba, half, physics_client_id=self.physics_client_id
            )
            self._half_extents[name] = half

        # Build the single shelf as separate board bodies at cumulative z's (custom builder:
        # the stock shelf uses uniform spacing). Each board is a solid collision + surface body,
        # so a too-tall upright block genuinely collides the board above a short cell (F3).
        self._shelf_ids: dict[str, int] = {}
        board_half = (
            config.shelf_width / 2,
            config.shelf_depth / 2,
            config.shelf_height / 2,
        )
        sx, sy = config.shelf_pose.position[0], config.shelf_pose.position[1]
        board_zs = board_center_zs(config)
        for i, cz in enumerate(board_zs):
            board_id = create_pybullet_block(
                config.shelf_rgba, board_half, physics_client_id=self.physics_client_id
            )
            set_pose(board_id, Pose((sx, sy, float(cz))), self.physics_client_id)
            self._shelf_ids[f"shelf_board_{i}"] = board_id

        # Side walls (+/-x) and a back panel (+y, far side) so the shelf reads as a real cupboard
        # with an open front (the -y side the robot inserts from). Real collision bodies, but NOT
        # placement surfaces and NOT tracked state objects (they still render).
        self._shelf_support_ids: list[int] = []
        sup = config.shelf_support_width / 2
        z_bottom = board_zs[0] - config.shelf_height / 2
        z_top = board_zs[-1] + config.shelf_height / 2
        stack_half = (z_top - z_bottom) / 2
        stack_cz = (z_top + z_bottom) / 2
        half_w = config.shelf_width / 2
        half_d = config.shelf_depth / 2
        supports = [
            # left / right side walls
            ((sup, half_d, stack_half), (sx - half_w - sup, sy, stack_cz)),
            ((sup, half_d, stack_half), (sx + half_w + sup, sy, stack_cz)),
            # back panel (far -from-robot side, +y)
            ((half_w + 2 * sup, sup, stack_half), (sx, sy + half_d + sup, stack_cz)),
        ]
        for half, pos in supports:
            sid = create_pybullet_block(
                config.shelf_rgba, half, physics_client_id=self.physics_client_id
            )
            set_pose(sid, Pose(pos), self.physics_client_id)
            self._shelf_support_ids.append(sid)

    # -- geometry access --------------------------------------------------
    def region_infos(self) -> dict[str, RegionInfo]:
        return self._region_infos

    def movable_names(self) -> list[str]:
        return [name for name, _, _ in self._object_specs]

    def shelf_board_ids(self) -> set[int]:
        return set(self._shelf_ids.values())

    def shelf_structure_ids(self) -> set[int]:
        """The shelf's COLLISION bodies = the boards. The side walls + back panel are cosmetic
        (visual-only, not collision) so they don't block off-centre placements near the shelf edges;
        F3 is a *board* (ceiling) collision, not a wall collision."""
        return set(self._shelf_ids.values())

    # -- required abstract methods ---------------------------------------
    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return Restock3DObjectCentricState

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._problem_seed = seed
        return super().reset(seed=seed, options=options)

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict(
            [(name, Kinematic3DFixtureType) for name in self._shelf_ids]
        )

    def _reset_objects(self) -> None:
        poses = self._pose_fn(self._problem_seed)
        for name in self._movable_ids:
            x, y = poses[name]
            half_z = self._half_extents[name][2]
            set_pose(
                self._movable_ids[name],
                Pose((float(x), float(y), float(half_z))),
                self.physics_client_id,
            )

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        for name in self._movable_ids:
            set_pose(
                self._movable_ids[name],
                obs.get_object_pose(name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name in self._shelf_ids:
            return self._shelf_ids[object_name]
        if object_name in self._movable_ids:
            return self._movable_ids[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        # Boards + movables only; the side walls / back panel (``_shelf_support_ids``) are cosmetic
        # (they render but do not collide), so they never spuriously block a placement.
        return set(self._shelf_ids.values()) | set(self._movable_ids.values())

    def _get_movable_object_names(self) -> set[str]:
        return set(self._movable_ids)

    def _get_surface_object_names(self) -> set[str]:
        return set(self._shelf_ids)

    def _get_surfaces_supporting_object(self, object_id: int) -> set[int]:
        """The base env registers only the shelf boards as placement surfaces, so an
        object placed on the FLOOR (buffer relocation) would never satisfy the ungrasp
        condition and stay held.

        Count
        the floor explicitly: an object whose underside is within ``min_placement_dist`` of ``z=0`` is
        floor-supported (sentinel id ``-1``). Only the ungrasp check reads this, so grasping and shelf
        placement are unaffected; the sentinel only fires when the gripper opens near the floor -- the
        ``PlaceBuffer`` case -- letting the cube release onto a buffer spot. See decisions/07
        2026-08-15 (F1 clutter re-added).
        """
        import pybullet as p  # local import; only needed here

        supports = super()._get_surfaces_supporting_object(object_id)
        aabb_min, _ = p.getAABB(object_id, physicsClientId=self.physics_client_id)
        if aabb_min[2] <= self.config.min_placement_dist:
            supports = supports | {-1}
        return supports

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name in self._half_extents:
            return self._half_extents[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Restock3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [(name, Kinematic3DFixtureType) for name in self._shelf_ids]
            + [(name, Kinematic3DCuboidType) for name in self._movable_ids]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=Restock3DObjectCentricState,
        )
        assert isinstance(state, Restock3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        """Gripper closed AND every goal object rests within some region footprint.

        Used only for the demo/terminated signal; the refiner uses the goal deriver.
        """
        if self._robot_arm.get_finger_state() > self.config.gripper_open_threshold:
            return False
        for name in self._movable_ids:
            if not name.startswith(("cube_goal", "block_goal")):
                continue
            from pybullet_helpers.geometry import get_pose  # local import

            pose = get_pose(self._movable_ids[name], self.physics_client_id)
            if pose.position[2] < 0.2:  # on a shelf section, not the floor
                return False
            if not self._name_in_any_region(name, pose):
                return False
        return True

    def _name_in_any_region(self, name: str, pose) -> bool:
        if not name.startswith(("cube_goal", "block_goal")):
            return True
        cx, cy = pose.position[0], pose.position[1]
        for info in self._region_infos.values():
            if (
                abs(cx - info.center_xy[0]) <= info.half_xy[0] + 0.03
                and abs(cy - info.center_xy[1]) <= info.half_xy[1] + 0.03
            ):
                return True
        return False


def stratum_object_specs(stratum: int, config: Restock3DEnvConfig) -> list[ObjectSpec]:
    """The fixed movable bodies for a stratum (small cubes + tall blocks + clutter)."""
    n_small, n_tall, _, _ = STRATA[stratum]
    n_clutter = CLUTTER_PER_STRATUM[stratum]
    specs: list[ObjectSpec] = []
    for i in range(1, n_small + 1):
        specs.append((f"cube_goal{i}", config.small_half, (0.1, 0.5, 0.1, 1.0)))
    for i in range(1, n_tall + 1):
        specs.append((f"block_goal{i}", config.tall_half, (0.6, 0.2, 0.2, 1.0)))
    for i in range(1, n_clutter + 1):
        specs.append((f"clutter{i}", config.clutter_half, (0.3, 0.3, 0.3, 1.0)))
    return specs


def stratum_pose_fn(stratum: int) -> PoseFn:
    """A pose function that scripts the floor layout for ``(seed, stratum)``."""

    def pose_fn(seed: int) -> dict[str, tuple[float, float]]:
        spec = build_spec(seed, stratum)
        poses: dict[str, tuple[float, float]] = {}
        for i, (fx, fy) in enumerate(spec.small_floor, start=1):
            poses[f"cube_goal{i}"] = (fx, fy)
        for i, (fx, fy) in enumerate(spec.tall_floor, start=1):
            poses[f"block_goal{i}"] = (fx, fy)
        for i, (fx, fy) in enumerate(spec.clutter_floor, start=1):
            poses[f"clutter{i}"] = (fx, fy)
        return poses

    return pose_fn


def stratum_env_args(stratum: int, config: Restock3DEnvConfig | None = None):
    """The (object_specs, pose_fn, region_infos, config) tuple for a collection
    stratum."""
    if config is None:
        config = Restock3DEnvConfig()
    return (
        stratum_object_specs(stratum, config),
        stratum_pose_fn(stratum),
        compute_region_infos(config, stratum),
        config,
    )


class Restock3DEnv(ConstantObjectKinDEREnv):
    """Constant-object gym wrapper for a single stratum."""

    def __init__(self, stratum: int = 0, **kwargs) -> None:
        self._stratum = stratum
        super().__init__(stratum=stratum, **kwargs)

    def _create_object_centric_env(
        self, *args, stratum: int = 0, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        object_specs, pose_fn, region_infos, config = stratum_env_args(stratum)
        return ObjectCentricRestock3DEnv(
            object_specs, pose_fn, region_infos, config=config, *args, **kwargs
        )

    def _get_constant_object_names(self, exemplar_state: ObjectCentricState) -> list:
        names = ["robot"]
        for obj in exemplar_state:
            if obj.name.startswith(
                ("shelf_board_", "cube_goal", "block_goal", "clutter")
            ):
                names.append(obj.name)
        return names

    def _create_env_markdown_description(self) -> str:
        return "Kinematic 3D restock: store floor objects into shelf regions."

    def _create_variant_markdown_description(self) -> str:
        return "Variants differ by difficulty stratum r0-r3."

    def _create_variant_specific_description(self) -> str:
        return f"Restock3D stratum r{self._stratum}."


# Convenience re-export (kept so callers can compute section geometry without importing
# region_geometry directly).
__all__ = [
    "Restock3DEnvConfig",
    "Restock3DObjectCentricState",
    "ObjectCentricRestock3DEnv",
    "Restock3DEnv",
    "stratum_env_args",
    "stratum_object_specs",
    "stratum_pose_fn",
    "section_surfaces",
    "CLUTTER_PER_STRATUM",
]
