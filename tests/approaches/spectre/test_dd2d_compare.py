"""Tests for the pure DD2D comparison-cache loader (``dd2d_compare``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from alphatamp.approaches.spectre import dd2d_compare


def test_rollout_fp_basic_and_ties() -> None:
    """FP counts negatives above the best positive, half-credit on exact ties."""
    # Best positive is top-scored -> 0 FP.
    assert dd2d_compare.rollout_fp([3.0, 1.0, 2.0], [1, 0, 0]) == 0.0
    # Two negatives outrank the single positive.
    assert dd2d_compare.rollout_fp([3.0, 2.0, 1.0], [0, 0, 1]) == 2.0
    # One negative ties the best positive -> half credit.
    assert dd2d_compare.rollout_fp([2.0, 2.0, 1.0], [0, 1, 0]) == 0.5
    # No feasible skeleton -> None.
    assert dd2d_compare.rollout_fp([1.0, 2.0], [0, 0]) is None


def test_stratum_of_bands() -> None:
    """Stratum comes from the test seed band, clamped to [0, 3]."""
    assert dd2d_compare.stratum_of(1_000_000) == 0
    assert dd2d_compare.stratum_of(1_250_000) == 1
    assert dd2d_compare.stratum_of(1_500_000) == 2
    assert dd2d_compare.stratum_of(1_999_999) == 3


def _dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_load_fp_records_aggregates_and_averages(tmp_path: Path) -> None:
    """load_fp_records derives static FPs and seed-averages SPECTRE."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000  # stratum 2
    # astar: order by score desc = [0,-1] -> positive at idx1 (score -1),
    # one negative (score 0) outranks it -> FP 1.
    _dump(
        cache / "astar" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [0.0, -1.0], "labels": [0, 1]},
    )
    _dump(
        cache / "piginet_v3" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [5.0, 1.0], "labels": [1, 0]},
    )
    # SPECTRE static: seed 0 -> FP 1 (neg outranks pos), seed 1 -> FP 0 -> mean 0.5.
    _dump(
        cache / "spectre_static" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [2.0, 1.0], "labels": [0, 1]},
    )
    _dump(
        cache / "spectre_static" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [2.0, 1.0], "labels": [1, 0]},
    )
    # SPECTRE adaptive: seeds 4 and 6 -> mean 5.
    _dump(
        cache / "spectre_adaptive" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 4.0},
    )
    _dump(
        cache / "spectre_adaptive" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 6.0},
    )

    recs = dd2d_compare.load_fp_records(cache)
    by_method = {r["method"]: r for r in recs}
    assert set(by_method) == set(dd2d_compare.METHOD_ORDER)
    assert all(r["problem_id"] == pid and r["stratum"] == 2 for r in recs)
    assert by_method["astar-dist"]["fp"] == 1.0
    assert by_method["PIGINet_v3"]["fp"] == 0.0
    assert by_method["SPECTRE-static"]["fp"] == 0.5
    assert by_method["SPECTRE-adaptive"]["fp"] == 5.0


def test_missing_cache_raises(tmp_path: Path) -> None:
    """A missing cache directory raises with the precompute command in the message."""
    with pytest.raises(FileNotFoundError, match="precompute_dd2d_cache.py"):
        dd2d_compare.load_fp_records(tmp_path / "nope")


def test_load_named_fp_records_seed_averages(tmp_path: Path) -> None:
    """load_named_fp_records seed-averages an adaptive-style fp cache."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000  # stratum 2
    _dump(
        cache / "spectre_lenctx" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 4.0, "order": [0, 1]},
    )
    _dump(
        cache / "spectre_lenctx" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 6.0, "order": [0, 2]},
    )
    recs = dd2d_compare.load_named_fp_records(
        cache, "spectre_lenctx", "SPECTRE-adaptive-lenctx"
    )
    assert len(recs) == 1
    assert recs[0] == {
        "problem_id": pid,
        "stratum": 2,
        "method": "SPECTRE-adaptive-lenctx",
        "fp": 5.0,
    }


# --- T0 length-dependence helpers ------------------------------------------


def test_length_fit_pure_length_function() -> None:
    """A ranking that is a function of length alone -> eta2=1, no within signal."""
    # scores depend only on length; same-length plans share a score exactly.
    scores = [10.0, 10.0, 5.0, 5.0, 5.0, 2.0]
    lengths = [1, 1, 3, 3, 3, 5]
    fit = dd2d_compare.length_fit(scores, lengths)
    assert fit["eta2"] == pytest.approx(1.0)
    assert fit["within_frac"] == pytest.approx(0.0)
    assert fit["n_len"] == 3 and fit["n"] == 6
    # longer -> lower score here, so the rank correlation is perfectly negative.
    assert fit["spearman"] == pytest.approx(-1.0)


def test_length_fit_within_length_signal() -> None:
    """Same-length plans that differ in score -> eta2 < 1, within_frac > 0."""
    scores = [10.0, 8.0, 5.0, 4.0]  # length 1 and 3 groups each vary
    lengths = [1, 1, 3, 3]
    fit = dd2d_compare.length_fit(scores, lengths)
    assert 0.0 < fit["eta2"] < 1.0
    assert fit["within_frac"] == pytest.approx(1.0 - fit["eta2"])


def test_length_fit_constant_scores_nan() -> None:
    """No score variance -> eta2/within/spearman undefined (NaN), not a crash."""
    fit = dd2d_compare.length_fit([1.0, 1.0, 1.0], [1, 3, 5])
    assert math.isnan(fit["eta2"]) and math.isnan(fit["within_frac"])
    assert math.isnan(fit["spearman"])


def test_spearman_sign_via_length_fit() -> None:
    """length_fit's Spearman is +1 when longer plans score higher, -1 otherwise."""
    # score increases with length -> prefers longer -> +1.
    up = dd2d_compare.length_fit([10.0, 20.0, 30.0, 40.0], [1, 3, 5, 7])
    assert up["spearman"] == pytest.approx(1.0)
    # score decreases with length -> prefers shorter -> -1.
    down = dd2d_compare.length_fit([40.0, 30.0, 20.0, 10.0], [1, 3, 5, 7])
    assert down["spearman"] == pytest.approx(-1.0)


def test_mean_position_by_length_short_first() -> None:
    """Descending-score order; a short-first ranking puts short tiers near 0."""
    # score = -length -> shortest tried first.
    lengths = [1, 3, 3, 5]
    scores = [-x for x in lengths]
    pos = dd2d_compare.mean_position_by_length(scores, lengths)
    assert pos[1] < pos[3] < pos[5]
    assert pos[1] == pytest.approx(0.0)  # the single shortest plan is tried first


def test_length_ladder_climb_and_slope() -> None:
    """A realized order that moves to longer plans -> positive spearman/slope."""
    # attempt indices [0,1,2] with lengths [1,3,5] -> climbs.
    lad = dd2d_compare.length_ladder([0, 1, 2], [1, 3, 5])
    assert lad["spearman"] == pytest.approx(1.0)
    assert lad["slope"] == pytest.approx(2.0)
    assert lad["n_steps"] == 3
    assert lad["first_len"] == 1.0 and lad["last_len"] == 5.0
    # a single-step trace has no defined trend but still reports endpoints.
    solo = dd2d_compare.length_ladder([2], [1, 3, 5])
    assert solo["n_steps"] == 1 and math.isnan(solo["spearman"])
    assert solo["first_len"] == 5.0 and solo["last_len"] == 5.0


def test_load_length_fit_records_and_ladder(tmp_path: Path) -> None:
    """End-to-end: static-score dirs -> length_fit; adaptive order -> ladder."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000  # stratum 2
    lengths = [1, 3, 3, 5]
    lengths_by_pid = {pid: lengths}
    # astar: score = -plan_idx (index order); within-length variance present.
    _dump(
        cache / "astar" / f"{pid}.json",
        {
            "problem_id": pid,
            "stratum": 2,
            "scores": [0.0, -1.0, -2.0, -3.0],
            "labels": [1, 0, 0, 1],
        },
    )
    _dump(
        cache / "piginet_v3" / f"{pid}.json",
        {
            "problem_id": pid,
            "stratum": 2,
            "scores": [0.1, 0.9, 0.2, 0.5],
            "labels": [1, 0, 0, 1],
        },
    )
    # SPECTRE static: pure length function (score depends only on length) both seeds.
    for seed in (0, 1):
        _dump(
            cache / "spectre_static" / f"seed_{seed}" / f"{pid}.json",
            {
                "problem_id": pid,
                "stratum": 2,
                "scores": [9.0, 4.0, 4.0, 1.0],
                "labels": [1, 0, 0, 1],
            },
        )
    # SPECTRE adaptive: realized attempt orders (climbing) per seed, with fp too.
    _dump(
        cache / "spectre_adaptive" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 2.0, "order": [0, 1, 3]},
    )
    _dump(
        cache / "spectre_adaptive" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 2.0, "order": [0, 2, 3]},
    )

    fit_recs = dd2d_compare.load_length_fit_records(cache, lengths_by_pid)
    by_method = {r["method"]: r for r in fit_recs}
    # adaptive is intentionally absent from the static-score loader.
    assert set(by_method) == {"astar-dist", "PIGINet_v3", "SPECTRE-static"}
    assert by_method["SPECTRE-static"]["eta2"] == pytest.approx(1.0)
    assert by_method["SPECTRE-static"]["within_frac"] == pytest.approx(0.0)

    pos_recs = dd2d_compare.load_position_by_length_records(cache, lengths_by_pid)
    # SPECTRE-static scores by length (9,4,4,1) -> length 1 tried first, 5 last.
    stat_pos = {
        r["length"]: r["mean_pos"] for r in pos_recs if r["method"] == "SPECTRE-static"
    }
    assert stat_pos[1] < stat_pos[3] < stat_pos[5]

    ladder_recs = dd2d_compare.load_adaptive_ladder_records(cache, lengths_by_pid)
    assert len(ladder_recs) == 1
    rec = ladder_recs[0]
    assert rec["method"] == "SPECTRE-adaptive" and rec["problem_id"] == pid
    # both seed orders climb from length 1 to length 5.
    assert rec["spearman"] == pytest.approx(1.0)
    assert rec["first_len"] == pytest.approx(1.0)
    assert rec["last_len"] == pytest.approx(5.0)
