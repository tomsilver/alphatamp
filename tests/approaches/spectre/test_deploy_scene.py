"""Tests for the Restock3D-v3 hand-specified deploy scene format + model builder."""

# The deploy kit is a folder outside the package; the scene builder is imported late,
# once its folder is on sys.path (exactly as ``deploy.py`` does at runtime).
# pylint: disable=wrong-import-position,wrong-import-order
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F

_DEPLOY_DIR = (
    Path(__file__).resolve().parents[3] / "experiments/spectre/restock3d_deploy"
)
sys.path.insert(0, str(_DEPLOY_DIR))

import deploy_scene as DS  # type: ignore[import-not-found]

_DEMO6 = _DEPLOY_DIR / "scenes/demo6"


def test_load_and_validate_demo6() -> None:
    """The worked 6-object scene parses, is geometrically feasible, and clamps to
    stratum 0."""
    scene = DS.load_scene(_DEMO6)
    assert scene.n == 6
    assert [o.name for o in scene.objects] == [f"obj_goal{i}" for i in range(1, 7)]
    assert scene.stratum == 0

    # Budgets default to the restock3d_v3_real stratum-0 collection values.
    assert DS.budget_for_n(6) == (35, 60.0, 6)

    # Depth defaults to 0.05 when unspecified.
    assert all(abs(o.depth - DS.DEFAULT_DEPTH) < 1e-9 for o in scene.objects)

    warnings = DS.validate_scene(scene)
    assert not any("NO FEASIBLE" in w for w in warnings), warnings
    # A clean scene: no width/height/spacing/OOD warnings either.
    assert warnings == [], warnings

    # And the analytic classifier agrees at least one two-level split packs.
    n_feas, _total, _rho = F.feasible_ratio(scene.blocks())
    assert n_feas >= 1


def test_budget_clamps_outside_trained_range() -> None:
    """N outside 6..9 clamps to the nearest trained stratum's budget."""
    assert DS.budget_for_n(3) == DS.budget_for_n(6)  # clamp up to stratum 0
    assert DS.budget_for_n(20) == DS.budget_for_n(9)  # clamp down to stratum 3


def test_missing_objects_key_raises(tmp_path: Path) -> None:
    (tmp_path / "scene.yaml").write_text("shelf: {x: 0.4, y: 1.4}\n")
    with pytest.raises(ValueError):
        DS.load_scene(tmp_path)


def test_infeasible_scene_warns(tmp_path: Path) -> None:
    """Six TALL objects all need the single tall section and overflow it ->
    unsolvable."""
    xs = [-0.70, -0.50, -0.30]
    ys = [0.75, 1.05]
    lines = ["objects:"]
    for i in range(6):
        x, y = xs[i % 3], ys[i // 3]
        lines += [
            f"  - name: obj_goal{i + 1}",
            "    width: 0.06",
            "    height: 0.15",  # > SHORT_CUTOFF (0.12): must go in the tall section
            f"    floor: [{x:.2f}, {y:.2f}]",
        ]
    (tmp_path / "scene.yaml").write_text("\n".join(lines) + "\n")
    scene = DS.load_scene(tmp_path)
    warnings = DS.validate_scene(scene)
    assert any("NO FEASIBLE" in w for w in warnings), warnings


@pytest.mark.slow
def test_build_deploy_models_and_x0() -> None:
    """Building the sim from the hand scene yields an x0 whose object dims match the
    file."""
    scene = DS.load_scene(_DEMO6)
    bundle = DS.build_deploy_models(scene)
    x0 = DS.make_x0(bundle.sim)

    goal_objs = [o for o in x0 if o.name.startswith("obj_goal")]
    assert len(goal_objs) == 6
    by_name = {o.name: o for o in scene.objects}
    for o in goal_objs:
        hx, _hy, hz = x0.get_object_half_extents(o.name)
        spec = by_name[o.name]
        assert abs(2 * hx - spec.width) < 5e-3
        assert abs(2 * hz - spec.height) < 5e-3

    # The initial abstract state has every goal object OnFloor and none Stored yet.
    s0 = bundle.models.state_abstractor(x0)
    atom_names = {a.predicate.name for a in s0.atoms}
    assert "OnFloor" in atom_names
    assert "Stored" not in atom_names
