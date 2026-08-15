"""F1/F2/F3 attribution tests for the kinematic Restock3D recording sampler (probe-level).

Drives the real-collision probes directly on hand-built micro-scenes: F3 (tall block vs short-cell
ceiling, culprit-free), F2 (place onto a region resident, resident named), F1 (grasp blocked by tight
clutter, clutter named). Marked ``slow`` — needs PyBullet + kinder (F1 also uses IKFast).
"""

from __future__ import annotations

import glob
import os
import pathlib
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow


def _blas_shim() -> None:
    b = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
    os.environ.setdefault("LAPACK_DIR", b)
    os.environ.setdefault("BLAS_DIR", b)
    pathlib.Path(b).mkdir(parents=True, exist_ok=True)
    for a, (sd, pt) in {
        "liblapack.a": ("lapack", "liblapack.so.3*"),
        "libblas.a": ("blas", "libblas.so.3*"),
    }.items():
        lk = pathlib.Path(b) / a
        if not (lk.exists() or lk.is_symlink()):
            cands = sorted(
                glob.glob(f"/usr/lib/x86_64-linux-gnu/{sd}/{pt}")
                + glob.glob(f"/usr/lib/x86_64-linux-gnu/{pt}")
            )
            real = next((c for c in cands if os.path.isfile(c)), None)
            if real:
                lk.symlink_to(real)


def _scene(specs, poses):
    """Build a micro-scene sim + a tall/short region + a probe-only recording sampler."""
    pytest.importorskip("kinder")
    _blas_shim()
    from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
        make_recording_sampler,
    )
    from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
        ObjectCentricRestock3DEnv,
        Restock3DEnvConfig,
    )
    from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
        RegionInfo,
        section_surfaces,
    )

    cfg = Restock3DEnvConfig()
    (tall_surf, tall_clr), (short_surf, short_clr) = section_surfaces(cfg)
    sx = cfg.shelf_pose.position[0]
    fy = cfg.shelf_pose.position[1] - cfg.region_front_offset
    hxy = (cfg.region_half_x, cfg.region_half_y)
    regions = {
        "region_tall": RegionInfo("region_tall", 0, (sx, fy), hxy, tall_clr, tall_surf),
        "region_short": RegionInfo(
            "region_short", 1, (sx, fy), hxy, short_clr, short_surf
        ),
    }
    sim = ObjectCentricRestock3DEnv(
        specs, lambda s: poses, regions, config=cfg, allow_state_access=True
    )
    sampler = make_recording_sampler(
        controller_generator=lambda a: None,
        transition_function=lambda x, u: x,
        state_abstractor=lambda x: None,
        max_trajectory_steps=1,
        sim=sim,
        region_infos=regions,
    )
    return sim, regions, sampler


def _op(name, *obj_names, state, region=None):
    from relational_structs import Object

    from alphatamp.approaches.spectre.envs.restock3d.place_controller import RegionType

    params = [state.get_object_from_name(n) for n in obj_names]
    if region is not None:
        params.append(Object(region, RegionType))
    return SimpleNamespace(name=name, parameters=params)


def test_f3_tall_block_into_short_section() -> None:
    specs = [("block_goal1", (0.025, 0.025, 0.12), (0.6, 0.2, 0.2, 1.0))]
    sim, _, sampler = _scene(specs, {"block_goal1": (0.6, 0.12)})
    state, _ = sim.reset(seed=0)
    culprits, family = sampler._probe_place(
        state, _op("place", "robot", "block_goal1", state=state, region="region_short")
    )
    assert family == "F3"
    assert culprits == ()
    # ... and the SAME block into the tall section is not F3 (fits).
    _, fam_tall = sampler._probe_place(
        state, _op("place", "robot", "block_goal1", state=state, region="region_tall")
    )
    assert fam_tall != "F3"


def test_f2_place_onto_resident() -> None:
    from pybullet_helpers.geometry import Pose, set_pose

    specs = [
        ("cube_goal1", (0.025, 0.025, 0.025), (0.1, 0.5, 0.1, 1.0)),
        ("cube_goal2", (0.025, 0.025, 0.025), (0.1, 0.5, 0.1, 1.0)),
    ]
    sim, regions, sampler = _scene(
        specs, {"cube_goal1": (0.45, 0.05), "cube_goal2": (0.55, 0.05)}
    )
    sim.reset(seed=0)
    # Seat cube_goal1 as a resident of region_short, then probe placing cube_goal2 there.
    info = regions["region_short"]
    set_pose(
        sim._object_name_to_pybullet_id("cube_goal1"),
        Pose((info.center_xy[0], info.center_xy[1], info.surface_z + 0.025)),
        sim.physics_client_id,
    )
    state = sim._get_obs()
    culprits, family = sampler._probe_place(
        state, _op("place", "robot", "cube_goal2", state=state, region="region_short")
    )
    assert family == "F2"
    assert "cube_goal1" in culprits


def test_f1_grasp_blocked_by_clutter() -> None:
    # F1 (grasp obstruction) is DEFERRED from restock3d v1 (needs relocatable goal-block blockers,
    # a generator redesign), but the probe machinery is kept + tested one flag away. TALL clutter
    # (0.20 m) beside the cube collides the arm at the top-down grasp config (a short neighbour does
    # not — the open fingers clear it), matching what actually blocks the descent MP.
    specs = [
        ("cube_goal1", (0.025, 0.025, 0.025), (0.1, 0.5, 0.1, 1.0)),
        ("clutter1", (0.025, 0.025, 0.10), (0.3, 0.3, 0.3, 1.0)),
        ("clutter2", (0.025, 0.025, 0.10), (0.3, 0.3, 0.3, 1.0)),
    ]
    poses = {
        "cube_goal1": (0.45, 0.05),
        "clutter1": (0.52, 0.05),
        "clutter2": (0.38, 0.05),
    }
    sim, _, sampler = _scene(specs, poses)
    state, _ = sim.reset(seed=0)
    culprits, family = sampler._probe_pick(
        state, _op("pick", "robot", "cube_goal1", state=state)
    )
    assert family == "F1"
    assert set(culprits) & {"clutter1", "clutter2"}
