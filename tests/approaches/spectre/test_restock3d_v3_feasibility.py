"""Unit tests for the Restock3D-v3 analytic feasibility core (``feasibility_v3``).

Pure-arithmetic: capacity formula, height cutoffs, split enumeration, greedy hand-rules, and the
``classify_skeleton`` analytic classifier (no motion planning). These pin the calibrated constants
and the failure-dict shape the SPECTRE downstream consumes.
"""

from __future__ import annotations

from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F


# --------------------------------------------------------------------------- constants / formula
def test_pinned_constants() -> None:
    assert F.SECTION_CLEARANCES == (0.27, 0.22)
    assert F.TALL_CUTOFF == 0.17 and F.SHORT_CUTOFF == 0.12
    assert (F.WIDTH_MIN, F.WIDTH_MAX) == (0.02, 0.08)
    assert F.GAP == 0.06 and F.USABLE == 0.50 and F.END_MARGIN == 0.04


def test_level_used_and_fits() -> None:
    assert F.level_used([]) == 0.0
    # one cube: 0.05 + 0 gaps + 2*0.04 = 0.13
    assert abs(F.level_used([0.05]) - 0.13) < 1e-9
    # four cubes: 0.20 + 3*0.06 + 0.08 = 0.46 <= 0.50 -> fits
    assert abs(F.level_used([0.05] * 4) - 0.46) < 1e-9
    assert F.level_fits([0.05] * 4)
    # five cubes: 0.25 + 4*0.06 + 0.08 = 0.57 > 0.50 -> does not fit
    assert abs(F.level_used([0.05] * 5) - 0.57) < 1e-9
    assert not F.level_fits([0.05] * 5)
    # three max-width blocks: 0.24 + 0.12 + 0.08 = 0.44 fits; four: 0.58 does not
    assert F.level_fits([0.08] * 3)
    assert not F.level_fits([0.08] * 4)


def test_height_eligible() -> None:
    assert F.height_eligible(0.12, "short") and not F.height_eligible(0.13, "short")
    assert F.height_eligible(0.17, "tall") and not F.height_eligible(0.18, "tall")
    assert F.height_eligible(0.05, "short") and F.height_eligible(0.05, "tall")
    # a tall-band block does not fit short
    assert not F.height_eligible(0.15, "short") and F.height_eligible(0.15, "tall")


def test_is_reach_tall() -> None:
    # reach-tall iff half-height >= 0.08 (full >= 0.16), matching the refiner's threshold
    assert F.is_reach_tall(0.17) and F.is_reach_tall(0.16)
    assert not F.is_reach_tall(0.15) and not F.is_reach_tall(0.12)


# --------------------------------------------------------------------------- split enumeration
def test_enumerate_splits_two_cubes() -> None:
    blocks = [F.Block("c0", 0.05, 0.05), F.Block("c1", 0.05, 0.05)]
    # both eligible for both sections, always fit -> all 4 splits feasible
    assert len(F.enumerate_feasible_splits(blocks)) == 4
    n_feas, total, rho = F.feasible_ratio(blocks)
    assert (n_feas, total) == (4, 4) and rho == 1.0


def test_enumerate_splits_tall_only_block() -> None:
    # a 0.17 tall block can only live in the tall section -> only the 2 splits that place it
    # tall are feasible (the cube is free either way)
    blocks = [F.Block("c0", 0.05, 0.05), F.Block("t0", 0.05, 0.17)]
    splits = F.enumerate_feasible_splits(blocks)
    assert len(splits) == 2
    assert all(a["t0"] == "tall" for a in splits)


def test_split_is_feasible_rejects_overcapacity_and_height() -> None:
    # five cubes all assigned to tall overflow the capacity formula
    blocks = [F.Block(f"c{i}", 0.05, 0.05) for i in range(5)]
    assert not F.split_is_feasible({b.name: "tall" for b in blocks}, blocks)
    # a tall block assigned short violates height
    b2 = [F.Block("t", 0.05, 0.15)]
    assert not F.split_is_feasible({"t": "short"}, b2)
    assert F.split_is_feasible({"t": "tall"}, b2)


# --------------------------------------------------------------------------- greedy hand-rules
def test_greedy_rules_wellformed_and_feasible_on_easy() -> None:
    blocks = [F.Block("c0", 0.05, 0.05), F.Block("t0", 0.05, 0.17)]
    for rule in F.HAND_RULES.values():
        assignment = rule(blocks)
        assert set(assignment) == {"c0", "t0"}
        assert set(assignment.values()) <= {"tall", "short"}
        assert F.split_is_feasible(assignment, blocks)
    # send-shortest-up puts the short block up and the tall block down
    a = F.greedy_send_shortest_up(blocks)
    assert a["c0"] == "short" and a["t0"] == "tall"


def test_greedy_rules_are_distinct() -> None:
    # the two rules must be genuinely different policies (they disagree on some instance)
    blocks = [
        F.Block("t0", 0.08, 0.15),  # tall-only, wide
        F.Block("t1", 0.08, 0.15),  # tall-only, wide
        F.Block("s0", 0.08, 0.05),  # short-eligible, wide
        F.Block("s1", 0.08, 0.05),
        F.Block("s2", 0.08, 0.05),
        F.Block("s3", 0.08, 0.05),
    ]
    a1 = F.greedy_widest_best_fit(blocks)
    a2 = F.greedy_send_shortest_up(blocks)
    assert a1 != a2  # different assignments -> genuinely different rules


# --------------------------------------------------------------------------- classifier
def _pos_row(names, y=1.0, x0=0.2, dx=0.15):
    """Positions in a west-east row at a common y (no reach-over between them)."""
    return {n: (x0 + i * dx, y) for i, n in enumerate(names)}


def test_classify_feasible_returns_none() -> None:
    names = ["c0", "c1"]
    dims = {n: (0.05, 0.05) for n in names}
    pos = _pos_row(names)
    plan = [
        ("pick", ("robot", "c0")),
        ("place_tall", ("robot", "c0")),
        ("pick", ("robot", "c1")),
        ("place_short", ("robot", "c1")),
    ]
    assert F.classify_skeleton(plan, dims, pos) is None


def test_classify_height_f3() -> None:
    # a 0.17 block placed into the short section -> culprit-free height failure
    dims = {"t0": (0.05, 0.17)}
    pos = {"t0": (0.4, 1.0)}
    plan = [("pick", ("robot", "t0")), ("place_short", ("robot", "t0"))]
    rec = F.classify_skeleton(plan, dims, pos)
    assert rec is not None
    assert rec["schema"] == "place_short"
    assert rec["culprits"] == []
    assert rec["dev_added"] == [] and rec["dev_deleted"] == []
    assert rec["exhausted"] and not rec["budget_exhausted"]
    assert rec["step_index"] == 1


def test_classify_crowding_f2() -> None:
    # five cubes all into the tall section -> the 5th place overflows; culprits = the first four
    names = [f"c{i}" for i in range(5)]
    dims = {n: (0.05, 0.05) for n in names}
    pos = _pos_row(names)  # common y -> no reach-over confound
    plan = []
    for n in names:
        plan.append(("pick", ("robot", n)))
        plan.append(("place_tall", ("robot", n)))
    rec = F.classify_skeleton(plan, dims, pos)
    assert rec is not None
    assert rec["schema"] == "place_tall"
    assert rec["culprits"] == ["c0", "c1", "c2", "c3"]
    assert rec["dev_added"] is None and rec["dev_deleted"] is None  # class-1


def test_classify_reachover_f4() -> None:
    # A south of B, same x, both tall -> picking B first is a reach-over failure blamed on A
    dims = {"A": (0.05, 0.17), "B": (0.05, 0.17)}
    pos = {"A": (0.4, 0.70), "B": (0.4, 1.00)}
    plan_bad = [("pick", ("robot", "B")), ("place_tall", ("robot", "B"))]
    rec = F.classify_skeleton(plan_bad, dims, pos)
    assert rec is not None
    assert rec["schema"] == "pick"
    assert rec["culprits"] == ["A"]
    assert rec["dev_added"] is None  # class-1
    # clearing A first (south-to-north) removes the reach-over
    plan_ok = [
        ("pick", ("robot", "A")),
        ("place_tall", ("robot", "A")),
        ("pick", ("robot", "B")),
        ("place_tall", ("robot", "B")),
    ]
    assert F.classify_skeleton(plan_ok, dims, pos) is None
