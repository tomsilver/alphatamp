"""The v3 grasp-witness layer is observation-only (G0).

``grasps.grasp_blocker`` / ``grasps.has_grasp_witness`` exist so the DD2D refiner can
record *which* object blocked a grasp -- a culprit -- without running any geometric query
the un-instrumented refiner did not already run. That property is what lets the
instrumented ``dd2d_v4`` collection reproduce ``dd2d_v3``'s labels bit-for-bit, so it is
pinned here rather than argued: the witness variants must agree with the originals on
every input, and the culprit indices they report must be genuine blockers.
"""

from __future__ import annotations

import random

import pytest
from shapely import box as shp_box
from shapely.geometry import Point

from alphatamp.approaches.spectre.envs.dd2d.drawer.grasps import (
    grasp_blocker,
    grasp_cells,
    grasp_cfree,
    has_grasp,
    has_grasp_witness,
)
from alphatamp.approaches.spectre.envs.dd2d.drawer.shapes import FAMILIES, sample_shape


def _obstacle_battery(rng: random.Random) -> list[list]:
    """Obstacle sets spanning clear / partially blocked / fully hemmed."""
    box_far = shp_box(30, 30, 40, 40)  # miles away: never blocks
    ring = shp_box(-20, -20, 20, 20).difference(Point(0, 0).buffer(6.0))  # hems in
    near = shp_box(2.0, -6.0, 9.0, 6.0)  # blocks some directions, not others
    other = shp_box(-9.0, -6.0, -2.0, 6.0)
    return [
        [],
        [box_far],
        [near],
        [near, other],
        [box_far, near, other],
        [ring],
        [near, ring, other],
    ]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_grasp_cfree_matches_blocker_witness(family: str) -> None:
    """``grasp_cfree`` is exactly ``grasp_blocker(...) < 0`` -- the refactor that made the
    culprit fall out of the collision loop changed no answer."""
    rng = random.Random(7)
    shape = sample_shape(rng, family=family)
    cells = grasp_cells(shape)
    assert cells, f"{family} should admit at least one grasp cell"
    checked = 0
    for obstacles in _obstacle_battery(rng):
        for pose in [(0.0, 0.0, 0.0), (1.5, -0.5, 0.7), (-2.0, 1.0, 2.4)]:
            for g in cells:
                blocker = grasp_blocker(g, pose, obstacles)
                assert grasp_cfree(g, pose, obstacles) == (blocker < 0)
                if blocker >= 0:
                    # the witness must name a real obstacle, and it must be the FIRST
                    # blocking one (the short-circuit the un-instrumented loop took)
                    assert 0 <= blocker < len(obstacles)
                    assert all(
                        grasp_cfree(g, pose, [obstacles[i]]) for i in range(blocker)
                    )
                    assert not grasp_cfree(g, pose, [obstacles[blocker]])
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_has_grasp_witness_returns_the_same_grasp(family: str) -> None:
    """``has_grasp_witness(...)[0]`` is ``has_grasp(...)`` -- same cell order, same
    short-circuit -- so swapping the refiner onto the witness variant is a no-op for
    every label."""
    rng = random.Random(11)
    for trial in range(3):
        shape = sample_shape(rng, family=family)
        for obstacles in _obstacle_battery(rng):
            for pose in [(0.0, 0.0, 0.0), (0.8, 1.2, 1.1)]:
                plain = has_grasp(shape, pose, obstacles)
                witness, culprits = has_grasp_witness(shape, pose, obstacles)
                assert plain == witness, (family, trial, pose)
                assert all(0 <= i < len(obstacles) for i in culprits)
                if plain is None and obstacles:
                    # ungraspable against a non-empty obstacle set: every cell was
                    # tried, so at least one blocker was observed (unless the shape
                    # simply has no cells, excluded by the battery above)
                    assert culprits or not grasp_cells(shape)


def test_witness_reports_no_culprits_when_unobstructed() -> None:
    """A grasp found on the first cell observed nothing -- the culprit set is empty, not
    'everything nearby'."""
    shape = sample_shape(random.Random(3), family="box")
    grasp, culprits = has_grasp_witness(shape, (0.0, 0.0, 0.0), [])
    assert grasp is not None
    assert culprits == frozenset()
