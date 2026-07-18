"""Step-1 gate (docs/piginet_dd2d_plan.md): DD2D geometry-extended records.

Verifies the geometry sidecar (`blocks_tamp/dd2d/record_ext.py`) that gives PIGINet its
geometric channel: per-object pose+shape on `objects[]`, one `at-pose` init fact per
object, real segmented crop PNGs (path filled), a drawer-frame normalization reference in
provenance, and a clean JSON round-trip.
"""

from __future__ import annotations

import imageio.v2 as imageio

from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import generate_dd2d_problem
from alphatamp.approaches.spectre.envs.dd2d.dd2d.record_ext import (
    build_dd2d_example,
    write_crops,
)
from alphatamp.approaches.spectre.envs.dd2d.record import PIGINetExample
from alphatamp.approaches.spectre.envs.dd2d.refine import RefineResult


def _small_problem():
    # small + certify=False keeps labeling fast; crowd=0 = naturalistic (F1/F2/F3 only).
    return generate_dd2d_problem(lam=0.8, seed=0, n_items=9, crowd=0, certify=False)


def _feasible_result(k: int) -> RefineResult:
    # record-structure test: no real refinement needed, just a well-formed feasible result.
    return RefineResult(
        status="feasible",
        steps_bound=k,
        plan_length=k,
        n_attempts=1,
        failure_action=None,
    )


def test_write_crops_one_png_per_object_matching_bbox(tmp_path):
    problem = _small_problem()
    images_dir = tmp_path / "images"
    refs = write_crops(problem, str(images_dir))

    # one ref per (object, single view); every object segmented -> path filled + file on disk
    assert len(refs) == len(problem.objects)
    assert all(r.path is not None for r in refs)
    for r in refs:
        png = images_dir / f"{r.object}__topdown.png"
        assert png.exists()
        assert r.path == f"images/{r.object}__topdown.png"  # relative + portable
        r0, c0, r1, c1 = r.bbox
        arr = imageio.imread(png)
        assert arr.shape[:2] == (r1 - r0 + 1, c1 - c0 + 1)  # crop dims == bbox extent
        assert arr.size > 0


def test_build_dd2d_example_geometry_and_roundtrip(tmp_path):
    problem = _small_problem()
    refs = write_crops(problem, str(tmp_path / "images"))
    sk = problem.intended_skeleton()
    ex = build_dd2d_example(
        problem,
        sk,
        _feasible_result(sk.length),
        planner_name="dd2d-candidates",
        images=refs,
    )

    # 2) geometry on every object
    names = {o["name"] for o in ex.objects}
    for o in ex.objects:
        assert isinstance(o["pose"], list) and len(o["pose"]) == 3
        assert all(isinstance(v, (int, float)) for v in o["pose"])
        shp = o["shape"]
        assert isinstance(shp["family"], str)
        assert all(isinstance(shp[k], (int, float)) for k in ("w", "h", "area"))
        assert isinstance(shp["concave"], bool)

    # 3) one at-pose init fact per object, well-formed
    at_pose = [f for f in ex.init_literals if f and f[0] == "at-pose"]
    assert len(at_pose) == len(ex.objects)
    for fact in at_pose:
        assert fact[1] in names
        assert isinstance(fact[2], list) and len(fact[2]) == 3

    # 4) provenance carries the drawer frame for train-time normalization
    assert (
        isinstance(ex.provenance["drawer_wh"], list)
        and len(ex.provenance["drawer_wh"]) == 2
    )
    assert all(isinstance(v, (int, float)) for v in ex.provenance["drawer_wh"])

    # 5) JSON round-trip is byte-identical
    assert PIGINetExample.from_json(ex.to_json()).to_json() == ex.to_json()
