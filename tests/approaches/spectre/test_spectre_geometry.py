"""Tests for the DD2D grasp-geometry reconstructor (``envs/dd2d/spectre_geometry``).

The reconstructor rebuilds the target's grasp obstacle set from a persisted
``SceneGeometry`` (boundary rings + poses + drawer W/D) instead of regenerating the
scene. Its load-bearing property is **label consistency**: computed on the same poses
the labeler used, a `blocked-after-removing` proof can never contradict a feasibility
label. These tests pin it (a) equivalent to the live env's own grasp check, and (b)
never firing on a subset that actually clears the target.
"""

from __future__ import annotations

import pytest

shapely = pytest.importorskip("shapely")

from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (  # noqa: E402
    reconstruct_wall_band,
    target_blocked_after_removing,
)
from alphatamp.approaches.spectre.schema import (  # noqa: E402
    ObjectGeometry,
    SceneGeometry,
)


def _scene_geometry_from_live(scene) -> SceneGeometry:
    """Mirror ``record_ext.build_dd2d_example`` geometry (boundary ring + pose)."""
    objs = []
    for name, st in scene.items.items():
        ring = tuple(
            (float(px), float(py))
            for px, py in list(st.shape.polygon.exterior.coords)[:-1]
        )
        objs.append(
            ObjectGeometry(
                name=name,
                pose=tuple(float(v) for v in st.pose),
                boundary=ring,
                family=st.shape.family,
                area=float(st.shape.area),
                concave=bool(st.shape.concave),
                is_target=(name == scene.target),
            )
        )
    return SceneGeometry(
        objects=tuple(objs),
        containers=(),
        frame={"drawer_w": scene.dims["W"], "drawer_d": scene.dims["D"]},
    )


def _live_blocked_after(scene, subset) -> bool:
    from alphatamp.approaches.spectre.envs.dd2d.drawer.enumerate import (
        _footprints,
        _obstacles,
    )
    from alphatamp.approaches.spectre.envs.dd2d.drawer.grasps import has_grasp

    present = set(scene.item_names()) - set(subset) - {scene.target}
    obs = _obstacles(_footprints(scene), present, scene.target, scene.wall_band)
    t = scene.target_state()
    return has_grasp(t.shape, t.pose, obs) is None


def test_wall_band_matches_live_env():
    from alphatamp.approaches.spectre.envs.dd2d.drawer.scene import generate_scene

    scene = generate_scene(seed=3, lam=0.8, crowd=5)
    frame = {"drawer_w": scene.dims["W"], "drawer_d": scene.dims["D"]}
    recon = reconstruct_wall_band(frame)
    assert recon.symmetric_difference(scene.wall_band).area < 1e-9


@pytest.mark.parametrize("seed", [1, 4, 7, 11])
def test_reconstruction_equivalent_to_live_grasp_check(seed):
    """Reconstructed blocked-after-removing == the env's own check, over many
    subsets."""
    from alphatamp.approaches.spectre.envs.dd2d.drawer.scene import generate_scene

    scene = generate_scene(seed=seed, lam=0.8, crowd=5)
    sg = _scene_geometry_from_live(scene)
    names = [n for n in scene.item_names() if n != scene.target]
    subsets = [frozenset()] + [frozenset({n}) for n in names]
    subsets += [frozenset(names[:2]), frozenset(names[:3]), frozenset(names)]
    for subset in subsets:
        assert target_blocked_after_removing(sg, subset) == _live_blocked_after(
            scene, subset
        )


def test_removing_all_clutter_opens_target():
    """Sanity: with every non-target item removed, the target is graspable (not
    blocked)."""
    from alphatamp.approaches.spectre.envs.dd2d.drawer.scene import generate_scene

    scene = generate_scene(seed=2, lam=0.8, crowd=5)
    sg = _scene_geometry_from_live(scene)
    others = frozenset(n for n in scene.item_names() if n != scene.target)
    assert target_blocked_after_removing(sg, others) is False


def test_frame_required_for_wall_band():
    sg = SceneGeometry(
        objects=(
            ObjectGeometry(
                name="target",
                pose=(0.0, 0.0, 0.0),
                boundary=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
                family="box",
                area=4.0,
                concave=False,
                is_target=True,
            ),
        ),
        containers=(),
        frame=None,
    )
    with pytest.raises(ValueError):
        target_blocked_after_removing(sg, frozenset())
