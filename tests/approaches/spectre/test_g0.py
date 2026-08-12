"""Unit tests for the Gate G0 cheap-statistics probes and the λ* decision rule."""

from __future__ import annotations

import numpy as np
from shapely import Polygon, box
from shapely.affinity import translate

from alphatamp.approaches.spectre.envs.dd2d.drawer.shapes import Shape
from alphatamp.approaches.spectre.envs.dd2d.drawer.world import DrawerScene, ItemState
from alphatamp.approaches.spectre.g0 import (
    FEATURE_NAMES,
    G0Point,
    buffer_slack,
    candidate_features,
    choose_lambda_star,
    feature_vector,
)


def _sq(side: float) -> Polygon:
    h = side / 2.0
    return Polygon([(-h, -h), (h, -h), (h, h), (-h, h)])


def _scene(polys, buffer, margin=1.0) -> DrawerScene:
    items = {}
    for i, p in enumerate(polys):
        p = translate(p, -p.centroid.x, -p.centroid.y)
        items[f"o{i}"] = ItemState(
            name=f"o{i}",
            shape=Shape(family="test", polygon=p, concave=False),
            pose=(0.0, 0.0, 0.0),
            region="drawer",
        )
    return DrawerScene(
        drawer=buffer,
        wall_band=Polygon(),
        buffer=buffer,
        items=items,
        target="o0",
        margin=margin,
        dims={},
    )


def test_buffer_slack():
    # two side-2 squares (area 4 each) deflated by 0.5 -> side-1 (area 1 each); buffer 20.
    sc = _scene([_sq(2), _sq(2)], box(0, 0, 5, 4), margin=1.0)
    slack = buffer_slack(sc, ["o0", "o1"])
    assert slack == 20.0 - 2.0  # buffer_area 20 minus 2*1


def test_feature_vector_shape_and_keys():
    sc = _scene([_sq(3), _sq(2)], box(0, 0, 10, 8))
    feats = candidate_features(sc, ["o0", "o1"])
    assert set(feats) == set(FEATURE_NAMES)
    v = feature_vector(sc, ["o0", "o1"])
    assert v.shape == (len(FEATURE_NAMES),)
    assert feats["n_items"] == 2.0
    assert feats["slack_ratio"] <= 1.0
    # a square has circularity < 1 (a disk is 1); values are finite.
    assert 0.0 < feats["mean_circularity"] < 1.0
    assert np.isfinite(v).all()


def _pt(lam, gbdt_within, oracle):
    # cheap_degraded keys on the WITHIN-length GBDT AUROC (the size-controlled residual).
    return G0Point(
        lam=lam,
        n_scenes=30,
        oracle_solve_rate=oracle,
        feasible_frac=0.3,
        marginal_frac=0.1,
        slack_auroc=0.5,
        gbdt_auroc=0.9,  # overall may be high (length-inflated) — not the criterion
        slack_within_auroc=0.5,
        gbdt_within_auroc=gbdt_within,
        n_conf=100,
    )


def test_within_length_auroc_controls_for_size():
    # a score that is perfect BETWEEN sizes but random WITHIN size → within-length ~0.5.
    y = np.array([1, 0, 1, 0])  # size-2 pair, size-3 pair
    sizes = np.array([2, 2, 3, 3])
    # score encodes size only (2 vs 3): perfect overall separation is impossible here,
    # but within each size the feasible/infeasible get identical scores → 0.5.
    score = np.array([10.0, 10.0, 20.0, 20.0])
    from alphatamp.approaches.spectre.g0 import within_length_auroc

    assert within_length_auroc(y, score, sizes) == 0.5


def test_choose_lambda_star_maximizes_gap_within_operating_range():
    # among in-range (0.7-0.95) degraded+solving λ, pick the largest oracle−GBDT_wl gap.
    points = [
        _pt(0.9, 0.85, 1.0),  # within-length captured → not degraded
        _pt(0.8, 0.60, 1.0),  # gap 0.40  <- max (and in range)
        _pt(0.75, 0.62, 0.8),  # gap 0.18
    ]
    assert choose_lambda_star(points, degrade_thresh=0.65, oracle_thresh=0.5) == 0.8


def test_choose_lambda_star_rejects_out_of_range_even_if_best():
    # a tighter λ maximizes the gap but is off-design (3-subsets stop packing) → excluded;
    # the in-range λ=0.8 is chosen instead.
    points = [
        _pt(0.80, 0.588, 0.97),  # in range, degraded  <- selected
        _pt(0.50, 0.578, 1.00),  # bigger gap but OUT of range -> rejected
        _pt(0.40, 0.803, 0.97),  # out of range + not degraded
    ]
    assert choose_lambda_star(points, operating_range=(0.7, 0.95)) == 0.80
    # if nothing in-range degrades, off-ramp:
    assert (
        choose_lambda_star([_pt(0.80, 0.9, 1.0)], operating_range=(0.7, 0.95)) is None
    )


def test_choose_lambda_star_offramp_when_oracle_fails():
    points = [
        _pt(0.8, 0.85, 1.0),
        _pt(0.3, 0.55, 0.1),
    ]  # degrades only where oracle fails
    assert choose_lambda_star(points, degrade_thresh=0.65, oracle_thresh=0.5) is None


def test_choose_lambda_star_offramp_when_never_degrades():
    # the real DD2D fingerprint: cheap GBDT captures within-length feasibility everywhere.
    points = [_pt(0.8, 0.9, 1.0), _pt(0.4, 0.88, 0.7)]
    assert choose_lambda_star(points) is None
