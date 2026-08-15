"""Tests for the kinematic Restock3D env geometry and construction (single multi-section shelf).

The geometry tests are pure Python (config + region math) and run in CI. The env-construction test
spins up a headless PyBullet client + the kinder robot, so it is marked ``slow``.
"""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import Restock3DEnvConfig
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
    board_center_zs,
    compute_region_infos,
    section_surfaces,
)

_BOARD_T = 0.0127


def test_section_surfaces_tall_bottom_short_top() -> None:
    cfg = Restock3DEnvConfig()
    (tall_surf, tall_clr), (short_surf, short_clr) = section_surfaces(cfg)
    assert tall_surf == cfg.bottom_surface_z  # 0.29
    assert (tall_clr, short_clr) == cfg.section_clearances  # (0.34, 0.15)
    # The short surface sits one board-thickness above the tall section's ceiling board.
    assert short_surf == pytest.approx(tall_surf + tall_clr + _BOARD_T)


def test_board_center_zs_one_more_than_sections() -> None:
    cfg = Restock3DEnvConfig()
    zs = board_center_zs(cfg)
    assert len(zs) == len(cfg.section_clearances) + 1  # 3 boards for 2 sections
    assert zs == sorted(zs)  # bottom -> top


def test_f3_geometry_invariant() -> None:
    """The block fits the tall gap (with gripper headroom) but overhangs the short gap -> F3."""
    cfg = Restock3DEnvConfig()
    block_h = 2 * cfg.tall_half[2]  # 0.24
    (_, tall_clr), (_, short_clr) = section_surfaces(cfg)
    assert block_h > short_clr, "block must be taller than the short cell (F3)"
    assert block_h < tall_clr, "block must fit the tall cell"
    cube_h = 2 * cfg.small_half[2]  # 0.05
    assert cube_h < short_clr, "cube must fit either cell"


def test_region_infos_sections_and_surfaces() -> None:
    cfg = Restock3DEnvConfig()
    (tall_surf, _), (short_surf, _) = section_surfaces(cfg)
    infos = compute_region_infos(cfg, stratum=3)  # STRATA[3] = (4,2,3,5)
    tall = {n: i for n, i in infos.items() if i.shelf == 0}
    short = {n: i for n, i in infos.items() if i.shelf == 1}
    assert len(tall) == 3 and len(short) == 5
    assert all(i.surface_z == pytest.approx(tall_surf) for i in tall.values())
    assert all(i.surface_z == pytest.approx(short_surf) for i in short.values())
    assert set(infos) == {f"region_0_{i}" for i in range(1, 4)} | {
        f"region_1_{i}" for i in range(1, 6)
    }


@pytest.mark.slow
def test_env_constructs_and_resets() -> None:
    pytest.importorskip("kinder")
    from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
        ObjectCentricRestock3DEnv,
        stratum_env_args,
    )

    object_specs, pose_fn, region_infos, config = stratum_env_args(2)
    env = ObjectCentricRestock3DEnv(
        object_specs, pose_fn, region_infos, config=config, allow_state_access=True
    )
    x0, _ = env.reset(seed=0)
    # One shelf built from 3 boards + 3 support bodies; movables spawned on the floor.
    assert len(env.shelf_board_ids()) == 3
    assert env.shelf_structure_ids() >= env.shelf_board_ids()
    names = {o.name for o in x0}
    assert any(n.startswith("cube_goal") for n in names)
    for spec_name, _, _ in object_specs:
        rest_z = x0.get_object_pose(spec_name).position[2]
        assert rest_z < 0.2, f"{spec_name} should start on the floor"
