"""Restock3D-v3 packing parity: the L2R controller's slot geometry is consistent-by-construction
with the ``feasibility_v3`` capacity formula.

Fast + deterministic (no motion planning): it packs widths left-to-right via the controller's own
``leftmost_slot_center`` arithmetic and asserts the packed span fits the physical packing region
**iff** ``feasibility_v3.level_fits`` says so. (The real-MP analogue of this parity is measured
end-to-end at Gate G1.)
"""

from __future__ import annotations

import random

from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d.place_controller_v3 import (
    leftmost_slot_center,
)

_CX = 0.4  # shelf center x (parity is cx-independent, but use the real value)


def _pack_right_faces(widths, cx=_CX):
    """Pack widths left-to-right via the controller's slot arithmetic; return each block's right
    face x."""
    edges: list[float] = []
    for w in widths:
        half = w / 2.0
        center = leftmost_slot_center(edges, half, cx)
        edges.append(center + half)
    return edges


def _fits_region(widths, cx=_CX):
    edges = _pack_right_faces(widths, cx)
    if not edges:
        return True
    right_limit = cx + F.USABLE / 2.0 - F.END_MARGIN
    return max(edges) <= right_limit + 1e-9


def test_parity_on_boundary_cases():
    # four cubes fit (level_used 0.46), five overflow (0.57) -> parity both ways
    assert F.level_fits([0.05] * 4) and _fits_region([0.05] * 4)
    assert (not F.level_fits([0.05] * 5)) and (not _fits_region([0.05] * 5))
    # three max-width fit (0.44), four overflow (0.58)
    assert F.level_fits([0.08] * 3) and _fits_region([0.08] * 3)
    assert (not F.level_fits([0.08] * 4)) and (not _fits_region([0.08] * 4))
    # empty level trivially fits
    assert F.level_fits([]) and _fits_region([])


def test_parity_left_margin_and_first_slot():
    # a single block's left face sits exactly at the capacity region's left margin
    cx = _CX
    w = 0.06
    center = leftmost_slot_center([], w / 2.0, cx)
    assert abs((center - w / 2.0) - (cx - F.USABLE / 2.0 + F.END_MARGIN)) < 1e-9


def test_parity_random_width_lists():
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(1, 8)
        widths = [round(rng.uniform(F.WIDTH_MIN, F.WIDTH_MAX), 4) for _ in range(n)]
        assert _fits_region(widths) == F.level_fits(widths), widths
