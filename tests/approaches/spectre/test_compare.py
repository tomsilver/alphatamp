"""Tests for the pure DD2D comparison-cache loader (``dd2d_compare``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from alphatamp.approaches.spectre import compare


def test_rollout_fp_basic_and_ties() -> None:
    """FP counts negatives above the best positive, half-credit on exact ties."""
    # Best positive is top-scored -> 0 FP.
    assert compare.rollout_fp([3.0, 1.0, 2.0], [1, 0, 0]) == 0.0
    # Two negatives outrank the single positive.
    assert compare.rollout_fp([3.0, 2.0, 1.0], [0, 0, 1]) == 2.0
    # One negative ties the best positive -> half credit.
    assert compare.rollout_fp([2.0, 2.0, 1.0], [0, 1, 0]) == 0.5
    # No feasible skeleton -> None.
    assert compare.rollout_fp([1.0, 2.0], [0, 0]) is None


def test_stratum_of_bands() -> None:
    """Stratum comes from the seed band, clamped to [0, 3]."""
    assert compare.stratum_of(1_000_000) == 0
    assert compare.stratum_of(1_250_000) == 1
    assert compare.stratum_of(1_500_000) == 2
    assert compare.stratum_of(1_999_999) == 3


def test_stratum_of_is_split_agnostic() -> None:
    """Train/val seeds must also resolve, and test must be bit-identical to before.

    The collector gives each split a disjoint 1M-wide band and divides it into four
    250k stratum sub-bands, so ``seed % 1M`` recovers the stratum on any split. The
    earlier test-only formula returned negative strata on train (seed < 1M) — which is
    how ``s-4`` appeared in a train-split VLMPlan run.
    """
    # train band [0, 1M)
    assert compare.stratum_of(0) == 0
    assert compare.stratum_of(250_017) == 1
    assert compare.stratum_of(500_032) == 2
    assert compare.stratum_of(750_063) == 3
    # val band [2M, 3M)
    assert compare.stratum_of(2_000_000) == 0
    assert compare.stratum_of(2_750_000) == 3
    # test band unchanged vs the historical formula, so published numbers cannot move
    for seed in range(1_000_000, 2_000_000, 9_973):
        assert compare.stratum_of(seed) == min(3, (seed - 1_000_000) // 250_000)


def _dump(path: Path, obj: dict) -> None:
    """Write a cache record, creating its directory."""
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
        cache / "piginet" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [5.0, 1.0], "labels": [1, 0]},
    )
    # SPECTRE static: seed 0 -> FP 1 (neg outranks pos), seed 1 -> FP 0 -> mean 0.5.
    _dump(
        cache / "spectre3_static" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [2.0, 1.0], "labels": [0, 1]},
    )
    _dump(
        cache / "spectre3_static" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [2.0, 1.0], "labels": [1, 0]},
    )
    # SPECTRE adaptive: seeds 4 and 6 -> mean 5.
    _dump(
        cache / "spectre3_adaptive" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 4.0},
    )
    _dump(
        cache / "spectre3_adaptive" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 6.0},
    )

    recs = compare.load_fp_records(cache)
    by_method = {r["method"]: r for r in recs}
    # SPECTRE is displayed without a version; its cache dirs are `spectre3_*`.
    assert set(by_method) == {
        "astar-dist",
        "PIGINet",
        "SPECTRE-static",
        "SPECTRE-adaptive",
    }
    assert all(r["problem_id"] == pid and r["stratum"] == 2 for r in recs)
    assert by_method["astar-dist"]["fp"] == 1.0
    assert by_method["PIGINet"]["fp"] == 0.0
    assert by_method["SPECTRE-static"]["fp"] == 0.5
    assert by_method["SPECTRE-adaptive"]["fp"] == 5.0


def test_missing_cache_raises(tmp_path: Path) -> None:
    """A missing cache directory raises with the precompute command in the message."""
    with pytest.raises(FileNotFoundError, match="precompute_dd2d_cache.py"):
        compare.load_fp_records(tmp_path / "nope")


def test_load_adaptive_trace_reads_step_fields(tmp_path: Path) -> None:
    """The per-problem accessor returns order + step-aligned scores and dead sets."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000
    _dump(
        cache / "spectre2_adaptive" / "seed_0" / f"{pid}.json",
        {
            "problem_id": pid,
            "stratum": 2,
            "fp": 1.0,
            "order": [1, 0],
            # step 1 masks candidate 1 (already attempted) -> null in the cache.
            "step_scores": [[0.5, 2.0], [0.5, None]],
            "step_dead": [[], [1]],
        },
    )
    tr = compare.load_adaptive_trace(cache, "spectre2_adaptive", pid)
    assert tr is not None
    assert tr.problem_id == pid and tr.stratum == 2 and tr.fp == 1.0
    assert tr.order == [1, 0]
    assert tr.step_dead == [[], [1]]
    # Both step fields are Optional (legacy caches lack them); narrow before indexing.
    assert tr.step_scores is not None and tr.step_dead is not None
    # Step-aligned with the realized order.
    assert len(tr.step_scores) == len(tr.order) == len(tr.step_dead)
    # JSON null (unavailable at that step) reads back as NaN, not 0.0.
    assert tr.step_scores[0] == [0.5, 2.0]
    assert tr.step_scores[1][0] == 0.5
    assert math.isnan(tr.step_scores[1][1])


def test_load_adaptive_trace_legacy_record_has_no_step_scores(tmp_path: Path) -> None:
    """A pre-per-step cache still loads; the new fields come back as None."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000
    _dump(
        cache / "spectre3_adaptive" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 3.0, "order": [2, 1, 0, 4]},
    )
    tr = compare.load_adaptive_trace(cache, "spectre3_adaptive", pid)
    assert tr is not None
    assert tr.order == [2, 1, 0, 4]
    assert tr.step_scores is None
    assert tr.step_dead is None


def test_single_problem_accessors_return_none_when_absent(tmp_path: Path) -> None:
    """Missing problem / missing family -> None, not an exception.

    PIGINet is genuinely missing one DD2D test problem, and a cache may hold only one
    SPECTRE family, so the inspector skips rather than raises.
    """
    cache = tmp_path / "compare_cache"
    assert compare.load_adaptive_trace(cache, "spectre2_adaptive", 1_600_000) is None
    assert compare.load_static_scores(cache, "piginet", 1_600_000) is None


def test_load_static_scores_roundtrip(tmp_path: Path) -> None:
    """The static accessor returns the cached record verbatim."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000
    rec = {"problem_id": pid, "stratum": 2, "scores": [1.0, 2.0], "labels": [0, 1]}
    _dump(cache / "piginet" / f"{pid}.json", rec)
    assert compare.load_static_scores(cache, "piginet", pid) == rec


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
    recs = compare.load_named_fp_records(
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
    fit = compare.length_fit(scores, lengths)
    assert fit["eta2"] == pytest.approx(1.0)
    assert fit["within_frac"] == pytest.approx(0.0)
    assert fit["n_len"] == 3 and fit["n"] == 6
    # longer -> lower score here, so the rank correlation is perfectly negative.
    assert fit["spearman"] == pytest.approx(-1.0)


def test_length_fit_within_length_signal() -> None:
    """Same-length plans that differ in score -> eta2 < 1, within_frac > 0."""
    scores = [10.0, 8.0, 5.0, 4.0]  # length 1 and 3 groups each vary
    lengths = [1, 1, 3, 3]
    fit = compare.length_fit(scores, lengths)
    assert 0.0 < fit["eta2"] < 1.0
    assert fit["within_frac"] == pytest.approx(1.0 - fit["eta2"])


def test_length_fit_r2_and_pearson_linear() -> None:
    """A score linear in length -> R² = pearson² ≈ 1; pearson sign = preference."""
    up = compare.length_fit([1.0, 3.0, 5.0, 7.0], [1, 3, 5, 7])
    assert up["pearson"] == pytest.approx(1.0)
    assert up["r2"] == pytest.approx(1.0)
    down = compare.length_fit([7.0, 5.0, 3.0, 1.0], [1, 3, 5, 7])
    assert down["pearson"] == pytest.approx(-1.0)
    assert down["r2"] == pytest.approx(1.0)


def test_length_fit_nonmonotone_r2_below_eta2() -> None:
    """A non-monotone length lookup -> eta2=1 (pure length) but linear R² small.

    This is the v1-static regime: the score *is* a function of length (eta2=1) but
    V-shaped in it, so the linear R² badly understates the length dependence — the
    reason the T0 table reports R² alongside eta2.
    """
    scores = [5.0, 5.0, 1.0, 1.0, 5.0, 5.0]  # V-shape over lengths 1/3/5
    lengths = [1, 1, 3, 3, 5, 5]
    fit = compare.length_fit(scores, lengths)
    assert fit["eta2"] == pytest.approx(1.0)
    assert fit["r2"] < 0.2


def test_length_fit_constant_scores_nan() -> None:
    """No score variance -> eta2/within/spearman undefined (NaN), not a crash."""
    fit = compare.length_fit([1.0, 1.0, 1.0], [1, 3, 5])
    assert math.isnan(fit["eta2"]) and math.isnan(fit["within_frac"])
    assert math.isnan(fit["spearman"])
    assert math.isnan(fit["r2"]) and math.isnan(fit["pearson"])


def test_spearman_sign_via_length_fit() -> None:
    """length_fit's Spearman is +1 when longer plans score higher, -1 otherwise."""
    # score increases with length -> prefers longer -> +1.
    up = compare.length_fit([10.0, 20.0, 30.0, 40.0], [1, 3, 5, 7])
    assert up["spearman"] == pytest.approx(1.0)
    # score decreases with length -> prefers shorter -> -1.
    down = compare.length_fit([40.0, 30.0, 20.0, 10.0], [1, 3, 5, 7])
    assert down["spearman"] == pytest.approx(-1.0)


def test_mean_position_by_length_short_first() -> None:
    """Descending-score order; a short-first ranking puts short tiers near 0."""
    # score = -length -> shortest tried first.
    lengths = [1, 3, 3, 5]
    scores = [-x for x in lengths]
    pos = compare.mean_position_by_length(scores, lengths)
    assert pos[1] < pos[3] < pos[5]
    assert pos[1] == pytest.approx(0.0)  # the single shortest plan is tried first


def test_length_ladder_climb_and_slope() -> None:
    """A realized order that moves to longer plans -> positive spearman/slope."""
    # attempt indices [0,1,2] with lengths [1,3,5] -> climbs.
    lad = compare.length_ladder([0, 1, 2], [1, 3, 5])
    assert lad["spearman"] == pytest.approx(1.0)
    assert lad["slope"] == pytest.approx(2.0)
    assert lad["n_steps"] == 3
    assert lad["first_len"] == 1.0 and lad["last_len"] == 5.0
    # a single-step trace has no defined trend but still reports endpoints.
    solo = compare.length_ladder([2], [1, 3, 5])
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
        cache / "piginet" / f"{pid}.json",
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
            cache / "spectre3_static" / f"seed_{seed}" / f"{pid}.json",
            {
                "problem_id": pid,
                "stratum": 2,
                "scores": [9.0, 4.0, 4.0, 1.0],
                "labels": [1, 0, 0, 1],
            },
        )
    # SPECTRE adaptive: realized attempt orders (climbing) per seed, with fp too.
    _dump(
        cache / "spectre3_adaptive" / "seed_0" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 2.0, "order": [0, 1, 3]},
    )
    _dump(
        cache / "spectre3_adaptive" / "seed_1" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "fp": 2.0, "order": [0, 2, 3]},
    )

    fit_recs = compare.load_length_fit_records(cache, lengths_by_pid)
    by_method = {r["method"]: r for r in fit_recs}
    # adaptive is intentionally absent from the static-score loader; v2 family
    # is absent from this fixture (gracefully skipped).
    assert set(by_method) == {"astar-dist", "PIGINet", "SPECTRE-static"}
    assert by_method["SPECTRE-static"]["eta2"] == pytest.approx(1.0)
    assert by_method["SPECTRE-static"]["within_frac"] == pytest.approx(0.0)

    pos_recs = compare.load_position_by_length_records(cache, lengths_by_pid)
    # SPECTRE-static scores by length (9,4,4,1) -> length 1 tried first, 5 last.
    stat_pos = {
        r["length"]: r["mean_pos"] for r in pos_recs if r["method"] == "SPECTRE-static"
    }
    assert stat_pos[1] < stat_pos[3] < stat_pos[5]

    ladder_recs = compare.load_adaptive_ladder_records(cache, lengths_by_pid)
    assert len(ladder_recs) == 1
    rec = ladder_recs[0]
    assert rec["method"] == "SPECTRE-adaptive" and rec["problem_id"] == pid
    # both seed orders climb from length 1 to length 5.
    assert rec["spearman"] == pytest.approx(1.0)
    assert rec["first_len"] == pytest.approx(1.0)
    assert rec["last_len"] == pytest.approx(5.0)


def _base_cache(cache: Path, pid: int) -> None:
    """The two dirs load_fp_records requires; everything else is optional."""
    _dump(
        cache / "astar" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [0.0, -1.0], "labels": [0, 1]},
    )
    _dump(
        cache / "piginet" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [5.0, 1.0], "labels": [1, 0]},
    )


def test_vlmplan_absent_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The notebook must load with no VLMPlan cache — the arm is optional."""
    cache = tmp_path / "compare_cache"
    _base_cache(cache, 1_600_000)
    methods = {r["method"] for r in compare.load_fp_records(cache)}
    assert not any(m.startswith("VLMPlan") for m in methods)
    assert not compare.load_vlmplan_diagnostics(cache, "vlmplan_qwen32b")


def test_vlmplan_fp_is_read_verbatim_not_derived(tmp_path: Path) -> None:
    """A sequence method's FP is precomputed, because only the builder knows the
    off-pool labels. It must be read straight off the record, never recomputed from
    per-pool scores (which a sequence method does not have)."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000
    _base_cache(cache, pid)
    _dump(
        cache / "vlmplan_qwen32b" / "seed_0" / f"{pid}.json",
        {
            "problem_id": pid,
            "stratum": 2,
            "fp": 7.0,
            "order": [12, -1, 3],  # -1 == an off-pool attempt
            "attempts": [
                {"source": "vlm", "round": 0, "in_pool": True, "pool_idx": 12},
                {"source": "vlm", "round": 1, "in_pool": False, "pool_idx": None},
                {"source": "fill", "round": None, "in_pool": True, "pool_idx": 3},
            ],
            "n_attempts": 3,
            "n_offpool": 1,
            "n_fill_used": 1,
            "n_live_refines": 1,
            "censored": False,
            "first_success_source": "fill",
            "spearman_vs_published": 0.5,
            "loop": {"plans_per_round": 10},
        },
    )
    rows = [r for r in compare.load_fp_records(cache) if r["method"] == "VLMPlan-32B"]
    assert rows == [
        {"problem_id": pid, "stratum": 2, "method": "VLMPlan-32B", "fp": 7.0}
    ]
    assert "VLMPlan-32B" in compare.METHOD_ORDER
    assert "VLMPlan-GPT5.6" in compare.METHOD_ORDER


def test_vlmplan_diagnostics_expose_the_reported_fields(tmp_path: Path) -> None:
    """Every field the notebook's VLMPlan section reports is surfaced."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000
    _base_cache(cache, pid)
    _dump(
        cache / "vlmplan_qwen8b" / "seed_0" / f"{pid}.json",
        {
            "problem_id": pid,
            "stratum": 2,
            "fp": 7.0,
            "order": [12, -1, 3],
            "attempts": [
                {"source": "vlm", "round": 0, "in_pool": True, "pool_idx": 12},
                {"source": "vlm", "round": 2, "in_pool": False, "pool_idx": None},
                {"source": "fill", "round": None, "in_pool": True, "pool_idx": 3},
            ],
            "n_attempts": 3,
            "n_offpool": 1,
            "n_fill_used": 1,
            "n_live_refines": 1,
            "censored": False,
            "first_success_source": "fill",
            "spearman_vs_published": 0.5,
            "loop": {"plans_per_round": 10},
        },
    )
    (row,) = compare.load_vlmplan_diagnostics(cache, "vlmplan_qwen8b")
    assert row["first_success_source"] == "fill"
    assert row["n_offpool"] == 1
    assert row["n_vlm_attempts"] == 2
    assert row["n_rounds_used"] == 3  # highest round tag (2) + 1
    assert row["spearman_vs_published"] == 0.5


def test_load_fp_records_per_seed_preserves_the_seed_axis(tmp_path: Path) -> None:
    """The seed axis must survive loading, or "within seed noise" is unmeasurable.

    ``load_fp_records`` averages a problem's FP across seeds before returning it, so a
    std taken downstream is the across-*problem* spread of a seed-mean. Every v3 gate is
    accepted on "no stratum regresses beyond seed noise", which needs the between-*seed*
    spread -- a different number, recoverable only from per-seed records.
    """
    cache = tmp_path / "compare_cache"
    pid = 1_600_000  # stratum 2
    _dump(
        cache / "astar" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [0.0, -1.0], "labels": [0, 1]},
    )
    _dump(
        cache / "piginet" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [5.0, 1.0], "labels": [1, 0]},
    )
    for seed, fp in ((0, 4.0), (1, 6.0), (2, 8.0)):
        _dump(
            cache / "spectre3_adaptive" / f"seed_{seed}" / f"{pid}.json",
            {"problem_id": pid, "stratum": 2, "fp": fp},
        )

    records = compare.load_fp_records_per_seed(cache)
    adaptive = sorted(
        (r for r in records if r["method"] == "SPECTRE-adaptive"),
        key=lambda r: r["seed"],
    )
    assert [r["seed"] for r in adaptive] == [0, 1, 2]
    assert [r["fp"] for r in adaptive] == [4.0, 6.0, 8.0]

    # the collapsing loader still reports the mean, so existing callers are unaffected
    collapsed = [
        r for r in compare.load_fp_records(cache) if r["method"] == "SPECTRE-adaptive"
    ]
    assert len(collapsed) == 1 and collapsed[0]["fp"] == 6.0

    # deterministic baselines carry seed=None rather than a fabricated 0: reporting a
    # spread for a single deterministic run would imply stability nobody measured
    assert {r["seed"] for r in records if r["method"] == "astar-dist"} == {None}
    # PIGINet here is a FLAT cache (no seed_* layer), so it is seedless too
    assert {r["seed"] for r in records if r["method"] == "PIGINet"} == {None}


def test_static_cache_layout_is_detected_not_assumed(tmp_path: Path) -> None:
    """A seeded PIGINet cache keeps its seed axis; a flat one still reads as one run.

    PIGINet had no ``--seed`` flag until 2026-07-28, so dd2d_v2/v3 are genuinely single
    deterministic runs while dd2d_v4 has three. Both layouts live on disk at once, and
    the difference must survive loading: fabricating ``seed_0`` for a flat cache would
    report a one-sample spread for something never sampled, while collapsing a seeded
    cache to ``seed=None`` would silently discard the spread we paid to measure.
    """
    cache = tmp_path / "compare_cache"
    pid = 1_600_000  # stratum 2
    _dump(
        cache / "astar" / f"{pid}.json",
        {"problem_id": pid, "stratum": 2, "scores": [0.0, -1.0], "labels": [0, 1]},
    )
    # PIGINet, seeded: seed 0 ranks the positive first (FP 0), seed 1 ranks it last
    # (FP 1)
    for seed, scores in ((0, [5.0, 1.0]), (1, [1.0, 5.0])):
        _dump(
            cache / "piginet" / f"seed_{seed}" / f"{pid}.json",
            {"problem_id": pid, "stratum": 2, "scores": scores, "labels": [1, 0]},
        )

    records = compare.load_fp_records_per_seed(cache)
    piginet = sorted(
        (r for r in records if r["method"] == "PIGINet"), key=lambda r: r["seed"]
    )
    assert [r["seed"] for r in piginet] == [0, 1]
    assert [r["fp"] for r in piginet] == [0.0, 1.0]
    assert {r["seed"] for r in records if r["method"] == "astar-dist"} == {None}

    # and the collapsing loader averages the seeds, exactly as it does for SPECTRE
    collapsed = [r for r in compare.load_fp_records(cache) if r["method"] == "PIGINet"]
    assert len(collapsed) == 1 and collapsed[0]["fp"] == 0.5

    # build_table then reports a real between-seed spread rather than a bare mean
    _header, rows, _tidy = compare.build_table(records)
    by_method = {r[0]: r for r in rows}
    assert by_method["PIGINet"][1] == "2"  # the `seeds` column
    assert by_method["astar-dist"][1] == "-"


def test_v3_table_reports_between_seed_spread() -> None:
    """The table's ``±`` is the spread across seeds of the per-stratum mean."""
    # two problems in one stratum; per-seed means are 5, 7, 9 -> mean 7, sample std 2
    records = []
    for seed, fps in ((0, (4.0, 6.0)), (1, (6.0, 8.0)), (2, (8.0, 10.0))):
        for pid, fp in zip((1_600_000, 1_600_001), fps):
            records.append(
                {
                    "seed": seed,
                    "problem_id": pid,
                    "stratum": 2,
                    "method": "SPECTRE-adaptive",
                    "fp": fp,
                }
            )
    records.append(
        {
            "seed": None,
            "problem_id": 1_600_000,
            "stratum": 2,
            "method": "astar-dist",
            "fp": 3.0,
        }
    )

    header, rows, tidy = compare.build_table(records)
    assert header[:3] == ["method", "seeds", "ALL"]
    by_method = {r[0]: r for r in rows}
    assert by_method["SPECTRE-adaptive"][1] == "3"
    assert by_method["SPECTRE-adaptive"][2] == "7.00 ± 2.00"
    # a single deterministic run reports no spread at all rather than "± 0.00"
    assert by_method["astar-dist"][1] == "-"
    assert by_method["astar-dist"][2] == "3.00"
    entry = next(
        t for t in tidy if t["method"] == "SPECTRE-adaptive" and t["stratum"] == "ALL"
    )
    assert entry["n_seeds"] == 3 and entry["mean_fp"] == 7.0


def test_load_named_fp_records_per_seed_keeps_the_seed_axis(tmp_path: Path) -> None:
    """An ablation arm read by name preserves seeds, unlike its averaging sibling."""
    cache = tmp_path / "compare_cache"
    pid = 1_600_000
    for seed, fp in [(0, 4.0), (1, 6.0)]:
        _dump(
            cache / "abl_thing_adaptive" / f"seed_{seed}" / f"{pid}.json",
            {"problem_id": pid, "stratum": 2, "fp": fp, "order": [0]},
        )
    rows = compare.load_named_fp_records_per_seed(cache, "abl_thing_adaptive", "thing")
    assert {(r["seed"], r["fp"]) for r in rows} == {(0, 4.0), (1, 6.0)}
    assert {r["method"] for r in rows} == {"thing"}
    # the averaging sibling collapses the same data to one row
    avg = compare.load_named_fp_records(cache, "abl_thing_adaptive", "thing")
    assert [r["fp"] for r in avg] == [5.0]


def test_load_named_fp_records_per_seed_missing_dir_raises(tmp_path: Path) -> None:
    """A missing ablation arm raises: a 2x2 silently rendering as 2x1 over-reads."""
    (tmp_path / "compare_cache").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="precompute_dd2d_cache.py"):
        compare.load_named_fp_records_per_seed(
            tmp_path / "compare_cache", "abl_absent", "absent"
        )


def _rec(method: str, pid: int, fp: float, seed=0) -> dict:
    return {"seed": seed, "problem_id": pid, "stratum": 2, "method": method, "fp": fp}


def test_merge_collections_grafts_only_the_named_methods() -> None:
    """Legacy rows are taken only for methods absent from the primary collection."""
    primary = [_rec("SPECTRE-adaptive", 1, 7.0), _rec("VLMPlan-32B", 1, 14.0)]
    legacy = [
        _rec("PIGINet", 1, 18.0, seed=None),
        _rec("VLMPlan-32B", 1, 13.0),  # exists in primary -> must NOT be taken
        _rec("astar-dist", 1, 29.0),
    ]
    out = compare.merge_collections(
        primary, legacy, ["PIGINet", "astar-dist", "VLMPlan-32B"], "v4", "v3"
    )
    by_method = {r["method"]: r for r in out}
    assert set(by_method) == {
        "SPECTRE-adaptive",
        "VLMPlan-32B",
        "PIGINet",
        "astar-dist",
    }
    # primary wins a name collision, and keeps its own value
    assert by_method["VLMPlan-32B"]["fp"] == 14.0
    assert by_method["VLMPlan-32B"]["collection"] == "v4"
    assert by_method["PIGINet"]["collection"] == "v3"
    # every record is tagged, not just the grafted ones
    assert all("collection" in r for r in out)


def test_merge_time_records_grafts_only_named_absent_methods() -> None:
    """§2b timing graft: a legacy timing row is taken only when its method is named and
    not already present natively -- the timing analog of merge_collections (no
    ``collection`` tag). This is what lets the kinder SB2D §2b read SPECTRE's wall-clock
    from the ``stickbutton2d_v1`` legacy cache while astar/PIGINet stay native."""
    primary = [_rec("PIGINet", 1, 0.14), _rec("astar-dist", 1, 0.0, seed=None)]
    legacy = [
        _rec("SPECTRE-adaptive", 1, 3.0),  # named + absent -> grafted
        _rec("SPECTRE-static", 1, 2.0),  # named + absent -> grafted
        _rec("PIGINet", 1, 99.0),  # named but present natively -> NOT taken
        _rec("astar-dist", 1, 88.0),  # not named -> NOT taken
    ]
    out = compare.merge_time_records(
        primary, legacy, ["SPECTRE-static", "SPECTRE-adaptive", "PIGINet"]
    )
    by_method: dict[str, list[float]] = {}
    for r in out:
        by_method.setdefault(r["method"], []).append(r["fp"])
    assert set(by_method) == {
        "PIGINet",
        "astar-dist",
        "SPECTRE-adaptive",
        "SPECTRE-static",
    }
    assert by_method["PIGINet"] == [0.14]  # native wins the name collision
    assert by_method["astar-dist"] == [0.0]  # unnamed legacy row dropped
    assert sorted(by_method["SPECTRE-adaptive"] + by_method["SPECTRE-static"]) == [
        2.0,
        3.0,
    ]


def test_select_seed_prefers_seed_zero_and_reports_it() -> None:
    """Seed 0 is kept when cached; deterministic rows pass through."""
    records = [
        _rec("m", 1, 5.0, seed=0),
        _rec("m", 1, 9.0, seed=1),
        _rec("astar-dist", 1, 34.0, seed=None),
    ]
    kept, chosen = compare.select_seed(records, prefer=0)
    assert chosen == {"m": 0, "astar-dist": None}
    assert [r["fp"] for r in kept if r["method"] == "m"] == [5.0]
    assert [r["fp"] for r in kept if r["method"] == "astar-dist"] == [34.0]


def test_select_seed_falls_back_to_the_best_seed() -> None:
    """Without seed 0, the lowest-mean-FP seed is used -- and named, not assumed."""
    records = [
        _rec("m", 1, 9.0, seed=1),
        _rec("m", 2, 9.0, seed=1),
        _rec("m", 1, 3.0, seed=2),
        _rec("m", 2, 5.0, seed=2),  # seed 2 mean 4.0 < seed 1 mean 9.0
    ]
    kept, chosen = compare.select_seed(records, prefer=0)
    assert chosen == {"m": 2}
    assert sorted(r["fp"] for r in kept) == [3.0, 5.0]


def test_build_table_reports_across_seed_spread_not_across_problem() -> None:
    """`±` must be the between-seed spread; one seed leaves it blank, never 0.00."""
    records = [
        # seed 0: problems 1,2 -> mean 2.0 ; seed 1: problems 1,2 -> mean 4.0
        _rec("m", 1, 0.0, seed=0),
        _rec("m", 2, 4.0, seed=0),
        _rec("m", 1, 4.0, seed=1),
        _rec("m", 2, 4.0, seed=1),
        _rec("solo", 1, 1.0, seed=0),
    ]
    header, rows, tidy = compare.build_table(records)
    assert header[:3] == ["method", "seeds", "ALL"]
    by_method = {r[0]: r for r in rows}
    # across-seed sd of (2.0, 4.0) is sqrt(2) ~ 1.41 -- NOT the across-problem sd
    assert by_method["m"][2] == "3.00 ± 1.41"
    # a single seed reports the mean alone; "0.00" would imply measured stability
    assert by_method["solo"][2] == "1.00"
    solo_all = next(t for t in tidy if t["method"] == "solo" and t["stratum"] == "ALL")
    assert math.isnan(solo_all["std_fp_across_seeds"])
