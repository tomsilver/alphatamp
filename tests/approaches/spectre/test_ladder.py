"""Tests for the elimination ladder (Step 9 acceptance mechanism)."""

from __future__ import annotations

import numpy as np

from alphatamp.approaches.spectre.ladder import (
    beats_slack_paired,
    variance_ladder,
)


def test_variance_ladder_pure_length():
    # scores that ARE length → length explains everything, residual ~0.
    rng = np.random.default_rng(0)
    length = rng.integers(1, 5, size=200).astype(float)
    slack = rng.normal(size=200)
    prox = rng.normal(size=200)
    rungs = variance_ladder(length.copy(), length, slack, prox)
    assert rungs.r2_length > 0.99 and rungs.residual < 0.01


def test_variance_ladder_residual_when_beyond_cheap_stats():
    # scores driven by an "identity" signal orthogonal to length/slack/proximity → residual high.
    rng = np.random.default_rng(1)
    n = 400
    length = rng.integers(1, 5, size=n).astype(float)
    slack = rng.normal(size=n)
    prox = rng.normal(size=n)
    identity = rng.normal(size=n)  # the subset-identity signal cheap stats can't see
    scores = 0.1 * length + identity
    rungs = variance_ladder(scores, length, slack, prox)
    assert rungs.residual > 0.5  # most variance is the identity residual


def test_beats_slack_paired_v2_better():
    # construct pools where v2 ranks a feasible skeleton first but slack does not.
    v2, slack, feas = [], [], []
    for _ in range(30):
        k = 8
        feasible = np.zeros(k, dtype=bool)
        feasible[5] = True  # the feasible one is at index 5
        # v2 scores it highest; slack scores it lowest → v2 FP=0, slack FP>0.
        v2s = np.zeros(k)
        v2s[5] = 10.0
        sls = np.ones(k)
        sls[5] = -10.0
        v2.append(v2s)
        slack.append(sls)
        feas.append(feasible)
    strata = np.full(30, 3)
    res = beats_slack_paired(v2, slack, feas, strata, min_stratum=2, n_boot=2000)
    assert res["n"] == 30 and res["mean_diff"] > 0 and res["passes"]


def test_beats_slack_paired_tie_does_not_pass():
    v2, slack, feas = [], [], []
    for _ in range(20):
        feasible = np.zeros(6, dtype=bool)
        feasible[0] = True
        s = np.array([5.0, 4, 3, 2, 1, 0])  # identical ordering for both
        v2.append(s.copy())
        slack.append(s.copy())
        feas.append(feasible)
    res = beats_slack_paired(v2, slack, feas, np.full(20, 3), n_boot=2000)
    assert res["mean_diff"] == 0.0 and not res["passes"]


def test_beats_slack_paired_filters_low_strata():
    v2 = [np.array([1.0, 0.0])]
    slack = [np.array([0.0, 1.0])]
    feas = [np.array([True, False])]
    res = beats_slack_paired(v2, slack, feas, np.array([1]), min_stratum=2)
    assert res["n"] == 0  # stratum 1 < 2 → excluded
