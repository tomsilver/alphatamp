"""Left-to-right analytic tight-packing place controllers for **Restock3D v3**.

v2's :class:`place_controller_v2.SectionFrontPlaceController` samples the placement x
**uniformly** across a section band, so a crowded section needs many samples before a free x is
hit. v3 blocks carry per-object widths, so packing must be deliberate: this controller reads the
section's current **residents** from the state and packs the held block at the **leftmost free
slot** — to the right of the rightmost resident's face (or at the left margin if the section is
empty), plus a small +-jitter. The packing region is exactly the capacity formula's region
``[cx - USABLE/2 + END_MARGIN, cx + USABLE/2 - END_MARGIN]`` (``feasibility_v3``), so a placement
succeeds iff the level's widths pass ``level_fits`` — the invariant the parity test pins.

Promoted (state-reading) from the standalone harness
``experiments/spectre/restock3d_v3_crowded_demo.py``; only ``sample_parameters`` changes vs v2.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from gymnasium.spaces import Box
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.kinematic3d.shelf3d import Kinematic3DRobotType
from relational_structs import Object, ObjectCentricState, Variable

from .feasibility_v3 import END_MARGIN, GAP, USABLE
from .instrumented_refiner import _RESIDENT_Z_TOL
from .place_controller import create_lifted_controllers
from .place_controller_v2 import SectionFrontPlaceController
from .region_geometry import RegionInfo
from .section_geometry import Y_JITTER

#: Small +-jitter around the analytic slot x (m); packing is exact, so a handful of samples suffice.
_PLACE_JITTER = 0.01


def leftmost_slot_center(
    right_edges: Sequence[float], my_half_x: float, cx: float
) -> float:
    """The center-x of the leftmost free slot for a block of half-width ``my_half_x`` on a section
    centered at ``cx`` whose residents' right-edge x's are ``right_edges``. Packs to the right of the
    rightmost resident (+ GAP), or at the left margin (``cx - USABLE/2 + END_MARGIN``) if empty. This
    is the single arithmetic shared by the controller and the packing-parity test, and it is
    consistent-by-construction with ``feasibility_v3.level_fits``."""
    left_face = (
        (max(right_edges) + GAP) if right_edges else (cx - USABLE / 2.0 + END_MARGIN)
    )
    return left_face + my_half_x


class LeftToRightSectionPlaceController(SectionFrontPlaceController):
    """Front place at the leftmost free slot, packing left-to-right against the section's
    residents (read from the state), consistent with the ``feasibility_v3`` capacity formula.
    """

    def __init__(
        self, objects: Sequence[Object], sim, section_info: RegionInfo
    ) -> None:
        super().__init__(objects, sim, section_info)
        self._target_name = objects[1].name

    def _resident_right_edges(self, x: ObjectCentricState) -> list[float]:
        """Right-edge x of each block already resting on this section (bottom ~ surface_z)."""
        info = self._section_info
        edges: list[float] = []
        for name in self._sim.movable_names():
            if name == self._target_name:
                continue
            obj = x.get_object_from_name(name)
            pose = x.get_object_pose(name)
            half_z = float(x.get(obj, "half_extent_z"))
            if abs(float(pose.position[2]) - half_z - info.surface_z) < _RESIDENT_Z_TOL:
                half_x = float(x.get(obj, "half_extent_x"))
                edges.append(float(pose.position[0]) + half_x)
        return edges

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator):
        info = self._section_info
        cx = info.center_xy[0]
        my_half_x = float(
            x.get(x.get_object_from_name(self._target_name), "half_extent_x")
        )
        edges = self._resident_right_edges(x)
        target_x = leftmost_slot_center(edges, my_half_x, cx)
        return np.array(
            [
                target_x - cx + float(rng.uniform(-_PLACE_JITTER, _PLACE_JITTER)),
                float(rng.uniform(-Y_JITTER, Y_JITTER)),
            ],
            dtype=np.float64,
        )


def create_lifted_controllers_v3(
    action_space, sim, section_infos: dict[str, RegionInfo]
) -> dict[str, LiftedParameterizedController]:
    """Lifted ``pick`` / ``place_tall`` / ``place_short`` / ``place_buffer`` for v3.

    Identical wiring to ``create_lifted_controllers_v2`` except the two section places are
    :class:`LeftToRightSectionPlaceController`s (analytic L2R packing) instead of uniform-band.
    """
    v1 = create_lifted_controllers(action_space, sim, section_infos)
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    def _make_section_place(section_info: RegionInfo) -> LiftedParameterizedController:
        class _Inner(LeftToRightSectionPlaceController):
            def __init__(self, objects):  # type: ignore[no-untyped-def]
                super().__init__(objects, sim, section_info)

        band = section_info.half_xy[0]
        return LiftedParameterizedController(
            [robot, target],
            _Inner,
            Box(low=np.array([-band, -Y_JITTER]), high=np.array([band, Y_JITTER])),
        )

    return {
        "pick": v1["pick"],
        "place_tall": _make_section_place(section_infos["section_0"]),
        "place_short": _make_section_place(section_infos["section_1"]),
        "place_buffer": v1["place_buffer"],
    }
