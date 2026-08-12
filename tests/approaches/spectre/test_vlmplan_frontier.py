"""Tests for the frontier-VLM (gpt-5.6-luna) VLMPlan additions.

Covers the env-agnostic pieces that do not need a model or kinder: the per-env plan
formatters (the ``retrieve ?`` inspector bug), stratified problem selection (the stride-
never-truncate trap), the Responses-API usage normalisation, the wall-clock capped-
refinement accounting, and the method registration.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from alphatamp.approaches.spectre import compare, compare_envs
from alphatamp.approaches.spectre.vlmplan.loop import RoundLog, _record_usage
from alphatamp.approaches.spectre.vlmplan.runio import _stratified
from alphatamp.approaches.spectre.vlmplan.score import Attempt, _fp_refine_capped


def test_sb2d_plan_label_is_not_retrieve_question_mark() -> None:
    """The SB2D formatter renders presses, not the DD2D-hardcoded ``retrieve ?``."""
    steps = [
        ("PickStickFromNothing", ["crv_robot_0", "rectangle_0"]),
        ("StickPressButtonFromNothing", ["crv_robot_0", "rectangle_0", "circle_3"]),
        ("PlaceStick", ["crv_robot_0", "rectangle_0"]),
        ("RobotPressButtonFromNothing", ["crv_robot_0", "circle_1"]),
    ]
    label = compare_envs.SB2D.plan_label(steps)
    assert "retrieve ?" not in label
    assert "pick stick" in label
    assert "circle_3 (stick)" in label
    assert "circle_1 (arm)" in label
    assert "place stick" in label


def test_dd2d_plan_label() -> None:
    """DD2D formatter keeps the ``stage {…} → retrieve N`` shape."""
    steps = [
        ("pick", ["item_5"]),
        ("place-buffer", ["item_5"]),
        ("retrieve", ["item_10"]),
    ]
    assert compare_envs.DD2D.plan_label(steps) == "stage {5} → retrieve 10"
    # No staging -> bare retrieve.
    assert compare_envs.DD2D.plan_label([("retrieve", ["item_3"])]) == "retrieve 3"


def test_every_env_has_a_plan_label() -> None:
    """A registered environment must format its own plans (else the inspector
    breaks)."""
    for spec in compare_envs.ENVS.values():
        assert spec.plan_label is not None
        assert callable(spec.plan_label)


def test_stratified_selection_balances_and_strides() -> None:
    """``_stratified`` takes ``per_stratum`` from each stratum, not the first N
    (band)."""
    # 4 strata x 25 pids in the DD2D test band, in problem-id order.
    eps = [
        SimpleNamespace(
            provenance=SimpleNamespace(problem_id=1_000_000 + s * 250_000 + i)
        )
        for s in range(4)
        for i in range(25)
    ]
    chosen = _stratified(eps, per_stratum=10, stratum_of=compare.stratum_of)
    assert len(chosen) == 40
    by_stratum: dict[int, int] = {}
    for ep in chosen:
        s = compare.stratum_of(int(ep.provenance.problem_id))
        by_stratum[s] = by_stratum.get(s, 0) + 1
    assert by_stratum == {0: 10, 1: 10, 2: 10, 3: 10}
    # Strided, not truncated: the last stratum's max index is sampled, not just 0..9.
    s3 = [
        int(e.provenance.problem_id)
        for e in chosen
        if compare.stratum_of(int(e.provenance.problem_id)) == 3
    ]
    assert max(s3) - 1_750_000 > 9  # would be <=9 if it took the first ten only


def test_record_usage_normalises_responses_api_keys() -> None:
    """The Responses API reports input_tokens/output_tokens; both namings map in."""
    log = RoundLog(round_index=0)
    _record_usage(
        log,
        {
            "input_tokens": 2666,
            "output_tokens": 1098,
            "output_tokens_details": {"reasoning_tokens": 390},
        },
        {"max_tokens": 16384},
    )
    assert log.prompt_tokens == 2666
    assert log.completion_tokens == 1098
    assert log.reasoning_tokens == 390
    assert log.truncated is False

    # Chat-completions naming still works, and the cap comparison flags truncation.
    log2 = RoundLog(round_index=1)
    _record_usage(
        log2, {"prompt_tokens": 10, "completion_tokens": 4096}, {"max_tokens": 4096}
    )
    assert log2.prompt_tokens == 10
    assert log2.completion_tokens == 4096
    assert log2.truncated is True


def _att(label: str, refine_s: float) -> Attempt:
    return Attempt(
        members=[],
        in_pool=False,
        pool_idx=None,
        label=label,
        source="vlm",
        refine_s=refine_s,
    )


def test_fp_refine_capped_stops_at_fast_success() -> None:
    """Two failures (one slow) then a fast success: cap charges min(t, cap)."""
    attempts = [_att("fail", 1.0), _att("fail", 3.0), _att("success", 0.5)]
    fp_capped, refine_capped = _fp_refine_capped(attempts, cap=2.0)
    assert fp_capped == 2.0
    # 1.0 + min(3.0, 2.0) + 0.5
    assert abs(refine_capped - 3.5) < 1e-9


def test_fp_refine_capped_abandons_slow_feasible() -> None:
    """A feasible candidate slower than the cap is abandoned and counts against FP."""
    attempts = [_att("success", 5.0), _att("fail", 1.0), _att("success", 0.5)]
    fp_capped, refine_capped = _fp_refine_capped(attempts, cap=2.0)
    assert fp_capped == 2.0  # the 5.0-second success is skipped
    # min(5,2) + 1.0 + 0.5
    assert abs(refine_capped - 3.5) < 1e-9


def test_frontier_arm_is_registered() -> None:
    """The frontier arm (gpt-5.6-terra) is a sequence method and carries timing."""
    assert compare.SEQUENCE_METHODS.get("VLMPlan-GPT5.6") == "vlmplan_terra"
    assert compare.TIMED_METHODS.get("VLMPlan-GPT5.6") == "vlmplan_terra"
    assert "VLMPlan-GPT5.6" in compare.METHOD_ORDER


def test_build_time_table_zeroes_plan_gen_for_sequence_methods() -> None:
    """A sequence method's total is infer + refine; it never adds the pool plan_gen."""
    records = [
        {
            "seed": 0,
            "problem_id": 1_000_000,
            "stratum": 0,
            "method": "astar-dist",
            "refine_s": 4.0,
            "refine_s_capped": 4.0,
            "fp_capped": 1.0,
            "infer_s": 0.0,
        },
        {
            "seed": 0,
            "problem_id": 1_000_001,
            "stratum": 0,
            "method": "VLMPlan-GPT5.6",
            "refine_s": 2.0,
            "refine_s_capped": 2.0,
            "fp_capped": 1.0,
            "infer_s": 60.0,
        },
    ]
    plan_gen = {0: 7.0}
    _, _, tidy = compare.build_time_table(records, plan_gen, use_capped=True)
    vlm_all = next(
        t for t in tidy if t["method"] == "VLMPlan-GPT5.6" and t["stratum"] == "ALL"
    )
    astar_all = next(
        t for t in tidy if t["method"] == "astar-dist" and t["stratum"] == "ALL"
    )
    assert vlm_all["plan_gen_s"] == 0.0
    assert (
        abs(vlm_all["mean_seconds"] - 62.0) < 1e-9
    )  # 60 infer + 2 refine, no plan_gen
    assert (
        abs(astar_all["mean_seconds"] - 11.0) < 1e-9
    )  # 7 plan_gen + 0 infer + 4 refine


def test_build_time_table_reports_per_component_std() -> None:
    """`infer_std`/`refine_std` are each component's across-seed spread; NaN at 1
    seed."""
    records = [
        # a 2-seed pool method: inference 0.4/0.6, refinement 1.0/2.0 across seeds
        {
            "seed": 0,
            "problem_id": 1_000_000,
            "stratum": 0,
            "method": "PIGINet",
            "refine_s": 1.0,
            "refine_s_capped": 1.0,
            "fp_capped": 1.0,
            "infer_s": 0.4,
        },
        {
            "seed": 1,
            "problem_id": 1_000_000,
            "stratum": 0,
            "method": "PIGINet",
            "refine_s": 2.0,
            "refine_s_capped": 2.0,
            "fp_capped": 1.0,
            "infer_s": 0.6,
        },
        # a 1-seed method -> each component std is NaN, not 0
        {
            "seed": 0,
            "problem_id": 1_000_001,
            "stratum": 0,
            "method": "VLMPlan-GPT5.6",
            "refine_s": 3.0,
            "refine_s_capped": 3.0,
            "fp_capped": 1.0,
            "infer_s": 60.0,
        },
    ]
    _, _, tidy = compare.build_time_table(records, {0: 0.0}, use_capped=True)
    pig = next(t for t in tidy if t["method"] == "PIGINet" and t["stratum"] == "ALL")
    assert abs(pig["infer_s"] - 0.5) < 1e-9
    assert abs(pig["infer_std"] - math.sqrt(0.02)) < 1e-9  # sample std of [0.4, 0.6]
    assert abs(pig["refine_s"] - 1.5) < 1e-9
    assert abs(pig["refine_std"] - math.sqrt(0.5)) < 1e-9  # sample std of [1.0, 2.0]
    vlm = next(
        t for t in tidy if t["method"] == "VLMPlan-GPT5.6" and t["stratum"] == "ALL"
    )
    assert math.isnan(vlm["infer_std"]) and math.isnan(vlm["refine_std"])
