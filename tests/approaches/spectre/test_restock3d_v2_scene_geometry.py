"""Gate-0 tests for the Restock3D v2 3D scene-geometry producer.

The fast test exercises the analytic box point cloud (pure numpy). The slow test builds
a real v2 sim and asserts the producer covers every registry object (I5), carries the 3D
point cloud + pose_z + height, and -- the whole point of going 3D -- distinguishes a
cube from a tall block by z-extent while their 2D footprints are identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphatamp.approaches.spectre.envs.restock3d.scene_geometry import (
    POINT_CLOUD_SIZE,
    object_point_cloud,
)


def test_object_point_cloud_scales_with_half_extents() -> None:
    cube = object_point_cloud((0.025, 0.025, 0.025))
    tall = object_point_cloud((0.025, 0.025, 0.12))
    assert cube.shape == (POINT_CLOUD_SIZE, 3)
    assert tall.shape == (POINT_CLOUD_SIZE, 3)
    # x, y extents match; z extent tracks half_z (the F3 signal a footprint loses).
    assert np.isclose(cube[:, 0].ptp(), 0.05) and np.isclose(cube[:, 1].ptp(), 0.05)
    assert np.isclose(cube[:, 2].ptp(), 0.05)
    assert np.isclose(tall[:, 2].ptp(), 0.24)
    # Centred at the origin (item frame).
    assert np.allclose(cube.mean(axis=0), 0.0, atol=1e-6)
    # Deterministic.
    assert np.array_equal(
        object_point_cloud((0.03, 0.04, 0.12)), object_point_cloud((0.03, 0.04, 0.12))
    )


@pytest.mark.slow
def test_producer_covers_registry_and_distinguishes_height() -> None:
    pytest.importorskip("kinder")
    from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import build_v2_bundle
    from alphatamp.approaches.spectre.envs.restock3d.scene_geometry import (
        build_scene_geometry,
    )

    bundle = build_v2_bundle(3)  # r3 = 4 cubes + 2 tall blocks
    try:
        x0, _ = bundle.sim.reset(seed=0)
        geo = build_scene_geometry(x0)

        names = {o.name for o in geo.objects}
        # I5: every object-registry key (goal objects + robot) has geometry.
        registry = set(bundle.goal_names) | {"robot"}
        assert registry.issubset(names), f"missing geometry for {registry - names}"

        # Normalization frame present (dataset.build_example raises without it).
        assert {"frame_w", "frame_d"}.issubset(geo.frame or {})

        for o in geo.objects:
            assert o.point_cloud is not None and len(o.point_cloud) == POINT_CLOUD_SIZE
            assert all(len(p) == 3 for p in o.point_cloud)
            assert o.pose_z is not None and o.height is not None

        cube = next(o for o in geo.objects if o.family == "cube")
        tall = next(o for o in geo.objects if o.family == "tall")
        # Same 2D footprint...
        assert np.allclose(np.asarray(cube.boundary), np.asarray(tall.boundary))
        # ...but the 3D cloud and height separate them.
        assert (
            np.asarray(tall.point_cloud)[:, 2].ptp()
            > 2 * np.asarray(cube.point_cloud)[:, 2].ptp()
        )
        assert tall.height > cube.height
    finally:
        bundle.sim.close()
