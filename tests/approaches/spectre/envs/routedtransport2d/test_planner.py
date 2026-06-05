"""Closed-form planner pool size + family partition (spec §8.3 #5-6)."""

from __future__ import annotations

import itertools

import pytest

from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
    _enumerate_raw_skeletons,
    _skeleton_family,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    make_problem,
)
from alphatamp.approaches.spectre.trajectory import apply_operator


@pytest.mark.parametrize(
    "variant,raw,capped",
    [
        ("n2-v1", 12, 12),
        ("n3-v1", 36, 30),
        ("n4-v1", 144, 30),
    ],
)
def test_pool_sizes_match_spec(variant: str, raw: int, capped: int) -> None:
    """Spec §8.3 #5: pool size matches the spec's variant table."""
    p = make_problem(seed=0, variant=variant)
    raw_skeletons = _enumerate_raw_skeletons(p)
    assert len(raw_skeletons) == raw

    cap_target = capped
    gen = ClosedFormSkeletonGenerator(p, k_cap=cap_target)
    pool = list(
        itertools.islice(gen(None, p.initial_abstract_state, p.goal, 1.0, None), 200)
    )
    assert len(pool) == capped


def test_family_partition_uniform_after_capping_n3() -> None:
    """Spec §8.3 #6: 6 families × 5 = 30 after capping for N=3."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    pool = list(
        itertools.islice(gen(None, p.initial_abstract_state, p.goal, 1.0, None), 200)
    )
    counts: dict[tuple[frozenset[str], str], int] = {}
    for _state_plan, action_plan in pool:
        f = _skeleton_family(action_plan)
        counts[f] = counts.get(f, 0) + 1
    # All 6 families with exactly 5 skeletons each.
    assert len(counts) == 6
    assert all(v == 5 for v in counts.values())


def test_family_partition_uniform_n2_uncapped() -> None:
    """N=2 has 12 raw skeletons (2 per family), uncapped."""
    p = make_problem(seed=0, variant="n2-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    pool = list(
        itertools.islice(gen(None, p.initial_abstract_state, p.goal, 1.0, None), 200)
    )
    counts: dict[tuple[frozenset[str], str], int] = {}
    for _state_plan, action_plan in pool:
        counts[_skeleton_family(action_plan)] = (
            counts.get(_skeleton_family(action_plan), 0) + 1
        )
    assert len(counts) == 6
    assert all(v == 2 for v in counts.values())


def test_skeleton_family_uses_exactly_two_loaded_colors() -> None:
    """Family classification: each skeleton uses exactly 2 distinct loaded colors."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    for _state_plan, action_plan in gen(
        None, p.initial_abstract_state, p.goal, 1.0, None
    ):
        loaded_colors = {
            op.name[-1]
            for op in action_plan
            if op.name.startswith("TraverseLoadedColor")
        }
        assert len(loaded_colors) == 2


def test_skeleton_uses_exactly_one_grasp_mode() -> None:
    """No mixed-grasp skeletons (spec §9.3 — pruned at the planner level)."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    for _state_plan, action_plan in gen(
        None, p.initial_abstract_state, p.goal, 1.0, None
    ):
        top_ops = sum(
            1 for op in action_plan if op.name in ("PickItemTop", "PlaceItemTop")
        )
        side_ops = sum(
            1 for op in action_plan if op.name in ("PickItemSide", "PlaceItemSide")
        )
        # Either all top or all side; never mixed.
        assert top_ops == 0 or side_ops == 0


def test_state_plan_aligns_with_operator_progression() -> None:
    """state_plan[i+1] must equal STRIPS application of action_plan[i] to
    state_plan[i]."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    for state_plan, action_plan in gen(
        None, p.initial_abstract_state, p.goal, 1.0, None
    ):
        assert len(state_plan) == len(action_plan) + 1
        for i, op in enumerate(action_plan):
            expected = apply_operator(state_plan[i], op)
            assert state_plan[i + 1].atoms == expected.atoms


def test_final_state_entails_goal() -> None:
    """state_plan[-1] must contain every goal atom."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    for state_plan, _action_plan in gen(
        None, p.initial_abstract_state, p.goal, 1.0, None
    ):
        assert p.goal.atoms.issubset(state_plan[-1].atoms)
