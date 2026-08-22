"""Tests for **Restock3D v2** section-band geometry (continuous packing).

Pure Python (config + section math), runs in CI. v2 replaces the discrete per-object
regions with two wide continuous placement bands (one per shelf section).
"""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import Restock3DEnvConfig
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import section_surfaces
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    _X_BAND_END_MARGIN,
    band_half_x,
    compute_section_infos,
    section_x_band,
)


def test_two_sections_tall_bottom_short_top() -> None:
    cfg = Restock3DEnvConfig()
    (tall_surf, tall_clr), (short_surf, short_clr) = section_surfaces(cfg)
    infos = compute_section_infos(cfg)
    assert set(infos) == {"section_0", "section_1"}
    assert infos["section_0"].shelf == 0 and infos["section_1"].shelf == 1
    assert infos["section_0"].surface_z == pytest.approx(tall_surf)  # 0.29
    assert infos["section_1"].surface_z == pytest.approx(short_surf)  # 0.6427
    assert infos["section_0"].cell_clearance == pytest.approx(tall_clr)  # 0.34
    assert infos["section_1"].cell_clearance == pytest.approx(short_clr)  # 0.15


def test_x_band_wide_and_inside_board() -> None:
    cfg = Restock3DEnvConfig()
    lo, hi = section_x_band(cfg)
    sx = cfg.shelf_pose.position[0]
    board_lo = sx - cfg.shelf_width / 2
    board_hi = sx + cfg.shelf_width / 2
    # Band is centred on the shelf, symmetric, and strictly inside the board x-extent (no overhang).
    assert lo == pytest.approx(sx - band_half_x(cfg))
    assert hi == pytest.approx(sx + band_half_x(cfg))
    assert board_lo < lo < hi < board_hi
    assert (lo - board_lo) == pytest.approx(_X_BAND_END_MARGIN)
    # The two section bands share the wide half-width and the shelf-centre / front-strip xy.
    infos = compute_section_infos(cfg)
    for info in infos.values():
        assert info.half_xy[0] == pytest.approx(band_half_x(cfg))
        assert info.center_xy[0] == pytest.approx(sx)
        assert info.center_xy[1] == pytest.approx(
            cfg.shelf_pose.position[1] - cfg.region_front_offset
        )


def test_band_wider_than_v1_reachable_row() -> None:
    """The continuous band should be at least as wide as v1's widest discrete region
    row."""
    cfg = Restock3DEnvConfig()
    lo, hi = section_x_band(cfg)
    # v1's widest short row (5 regions, pitch 0.12) spans [0.16, 0.64] centre-to-centre.
    assert lo <= 0.16 and hi >= 0.64


def test_f3_geometry_invariant() -> None:
    """A tall block overhangs the short cell (F3) but fits the tall cell; a cube fits
    either."""
    cfg = Restock3DEnvConfig()
    block_h = 2 * cfg.tall_half[2]  # 0.24
    (_, tall_clr), (_, short_clr) = section_surfaces(cfg)
    assert block_h > short_clr  # place_short(block) -> ceiling collision (F3)
    assert block_h < tall_clr  # place_tall(block) fits
    assert 2 * cfg.small_half[2] < short_clr  # cube fits either section
