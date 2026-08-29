"""Section place controllers for **Restock3D v2** (continuous x-band).

v1 places a held object at a discrete region's centre (+-0.015 jitter): the region is an operator
argument and the controller reads ``region_infos[objects[2].name]``. v2 instead has TWO place
operators — ``place_tall`` / ``place_short`` — each bound to a shelf section, and samples the
placement x **uniformly across that section's continuous band** (y = front strip + a tiny jitter).

Implementation reuses v1's :class:`place_controller.RestockFrontPlaceController` **verbatim**: that
controller already places ``objects[1]`` (the target) at ``region_infos[objects[2].name]``'s centre +
surface. We simply **synthesise the section as an internal region object** (``objects[2]``) so the
inherited translate-only front place — base standoff, upright EE-from-grasp, F2/F3 real collision —
runs unchanged; only ``sample_parameters`` widens the x range from the +-0.015 jitter to the full band.
No edit to v1's place controller.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from gymnasium.spaces import Box
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.kinematic3d.shelf3d import Kinematic3DRobotType
from relational_structs import Object, ObjectCentricState, Variable

from .place_controller import (
    RegionType,
    RestockFrontPlaceController,
    create_lifted_controllers,
)
from .region_geometry import RegionInfo
from .section_geometry import Y_JITTER


class SectionFrontPlaceController(RestockFrontPlaceController):
    """Translate-only front place onto a fixed shelf SECTION, x sampled across its band.

    Bound to one ``section_info`` at construction (no ``?region`` operator arg). Reuses
    the inherited ``step`` by presenting the section as a synthetic ``objects[2]``
    region; overrides only the sampler so x spans ``+-band_half_x`` (the section's
    ``half_xy[0]``) and y spans ``+-Y_JITTER``.
    """

    def __init__(
        self, objects: Sequence[Object], sim, section_info: RegionInfo
    ) -> None:
        # Present the section as an internal region object so the inherited step's
        # ``region_infos[objects[2].name]`` lookup resolves to this section, unchanged.
        section_obj = Object(f"__{section_info.name}", RegionType)
        super().__init__(
            list(objects) + [section_obj], sim, {section_obj.name: section_info}
        )
        self._section_info = section_info

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x
        band = self._section_info.half_xy[0]
        return np.array(
            [rng.uniform(-band, band), rng.uniform(-Y_JITTER, Y_JITTER)],
            dtype=np.float64,
        )


def create_lifted_controllers_v2(
    action_space, sim, section_infos: dict[str, RegionInfo]
) -> dict[str, LiftedParameterizedController]:
    """Lifted ``pick`` / ``place_tall`` / ``place_short`` / ``place_buffer`` controllers
    for v2.

    ``pick`` and the (inert) ``place_buffer`` are reused from v1 unchanged;
    ``place_tall`` / ``place_short`` are section-bound
    :class:`SectionFrontPlaceController`s over ``[robot, target]`` (no region arg), each
    with a Box spanning its section band in x and ``+-Y_JITTER`` in y.
    """
    v1 = create_lifted_controllers(action_space, sim, section_infos)

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    def _make_section_place(section_info: RegionInfo) -> LiftedParameterizedController:
        class _Inner(SectionFrontPlaceController):
            def __init__(self, objects):  # type: ignore[no-untyped-def]
                super().__init__(objects, sim, section_info)

        band = section_info.half_xy[0]
        return LiftedParameterizedController(
            [robot, target],
            _Inner,
            Box(
                low=np.array([-band, -Y_JITTER]),
                high=np.array([band, Y_JITTER]),
            ),
        )

    return {
        "pick": v1["pick"],
        "place_tall": _make_section_place(section_infos["section_0"]),
        "place_short": _make_section_place(section_infos["section_1"]),
        "place_buffer": v1["place_buffer"],
    }
