"""Three-gate refiner logic across modes × tag combos (spec §8.3 #9)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from relational_structs import GroundOperator

from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
    _skeleton_family,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    ProblemInstance,
    make_problem,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.refiner import (
    ThreeGateRefiner,
    refine,
)


def _frozen_skeleton(
    p: ProblemInstance, family_target: tuple[frozenset[str], str]
) -> list[GroundOperator]:
    """Pull the first skeleton in the requested family from the canonical pool."""
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    for _state_plan, action_plan in gen(
        None, p.initial_abstract_state, p.goal, 1.0, None
    ):
        if _skeleton_family(action_plan) == family_target:
            return action_plan
    raise AssertionError(f"family {family_target} missing from pool")


@pytest.mark.parametrize("blocked_color", ["A", "B", "C"])
@pytest.mark.parametrize("blocked_grasp", ["top", "side"])
def test_right_family_succeeds_in_its_mode(
    blocked_color: str, blocked_grasp: str
) -> None:
    """A skeleton in F_z always succeeds under mode z modulo noise.

    With base_op_fail_rate=0 we get deterministic behavior: the right family has no
    firing gate, so refine returns success.
    """
    p = make_problem(seed=0, variant="n3-v1")
    other_colors = frozenset({"A", "B", "C"}) - {blocked_color}
    other_grasp = "side" if blocked_grasp == "top" else "top"
    family = (other_colors, other_grasp)
    skel = _frozen_skeleton(p, family)
    # Force tag-feasibility for this test by using all-wide / all-small:
    feasible_widths = {pn: "wide" for pn in p.passage_widths}
    feasible_sizes = {item: "small" for item in p.item_sizes}
    out = refine(
        skel,
        blocked_color=blocked_color,
        blocked_grasp=blocked_grasp,
        passage_widths=feasible_widths,
        item_sizes=feasible_sizes,
        rng=np.random.default_rng(0),
        base_op_fail_rate=0.0,
    )
    assert out.success, f"right family should succeed; got cause {out.stuck_cause}"


@pytest.mark.parametrize("color", ["A", "B", "C"])
def test_blocked_color_gate_fires(color: str) -> None:
    """Any skeleton whose loaded colors include ``color`` fails on TraverseLoaded."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    feasible_widths = {pn: "wide" for pn in p.passage_widths}
    feasible_sizes = {item: "small" for item in p.item_sizes}
    for _state_plan, action_plan in itertools.islice(
        gen(None, p.initial_abstract_state, p.goal, 1.0, None), 30
    ):
        loaded_colors, grasp = _skeleton_family(action_plan)
        if color not in loaded_colors:
            continue
        # Force a non-grasp-blocking mode so the color gate is the only one firing.
        blocked_grasp = grasp  # wait — this would fire the grasp gate before the color
        # Pick the *opposite* grasp so the grasp gate never fires.
        blocked_grasp = "top" if grasp == "side" else "side"
        out = refine(
            action_plan,
            blocked_color=color,
            blocked_grasp=blocked_grasp,
            passage_widths=feasible_widths,
            item_sizes=feasible_sizes,
            rng=np.random.default_rng(0),
            base_op_fail_rate=0.0,
        )
        assert not out.success
        assert out.stuck_cause == "blocked_color"
        assert out.stuck_op_name is not None
        assert out.stuck_op_name.endswith(color)


@pytest.mark.parametrize("blocked_grasp", ["top", "side"])
def test_blocked_grasp_gate_fires_on_pick(blocked_grasp: str) -> None:
    """A skeleton using grasp ``g`` fails its first PickItem<g> when blocked_grasp ==
    g."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    feasible_widths = {pn: "wide" for pn in p.passage_widths}
    feasible_sizes = {item: "small" for item in p.item_sizes}
    for _state_plan, action_plan in itertools.islice(
        gen(None, p.initial_abstract_state, p.goal, 1.0, None), 30
    ):
        _loaded, grasp = _skeleton_family(action_plan)
        if grasp != blocked_grasp:
            continue
        # Choose a non-blocked color so the color gate doesn't fire first.
        loaded_set = {
            op.name[-1]
            for op in action_plan
            if op.name.startswith("TraverseLoadedColor")
        }
        blocked_color = next(c for c in ("A", "B", "C") if c not in loaded_set)
        out = refine(
            action_plan,
            blocked_color=blocked_color,
            blocked_grasp=blocked_grasp,
            passage_widths=feasible_widths,
            item_sizes=feasible_sizes,
            rng=np.random.default_rng(0),
            base_op_fail_rate=0.0,
        )
        assert not out.success
        assert out.stuck_cause == "blocked_grasp"
        assert out.stuck_op_name is not None
        assert out.stuck_op_name.startswith("PickItem")


def test_size_width_gate_overrides_grasp_gate_only_when_color_clean() -> None:
    """Size-width fires before blocked_grasp because Pick comes after the first loaded
    traversal in some skeletons.

    Schedule a size-width incompatible passage on a non-blocked-color and verify
    size_width fires.
    """
    p = make_problem(seed=42, variant="n3-v1")
    # Pick a skeleton in a family that uses non-blocked colors and non-blocked grasp.
    blocked_color = p.blocked_color
    blocked_grasp = p.blocked_grasp
    safe_colors = frozenset({"A", "B", "C"}) - {blocked_color}
    safe_grasp = "side" if blocked_grasp == "top" else "top"
    skel = _frozen_skeleton(p, (safe_colors, safe_grasp))
    # Put a narrow width on every passage so EVERY loaded traversal is incompat
    # for any non-small item. Set every item to large to guarantee the trip.
    narrow_widths = {pn: "narrow" for pn in p.passage_widths}
    large_sizes = {item: "large" for item in p.item_sizes}
    out = refine(
        skel,
        blocked_color=blocked_color,
        blocked_grasp=blocked_grasp,
        passage_widths=narrow_widths,
        item_sizes=large_sizes,
        rng=np.random.default_rng(0),
        base_op_fail_rate=0.0,
    )
    assert not out.success
    assert out.stuck_cause == "size_width"


def test_three_gate_refiner_class_sets_last_outcome() -> None:
    """ThreeGateRefiner exposes its structured outcome via last_outcome."""
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    state_plan, action_plan = next(
        iter(gen(None, p.initial_abstract_state, p.goal, 1.0, None))
    )
    refiner = ThreeGateRefiner(p, seed=7)
    refiner(None, state_plan, action_plan, 1.0, None)
    assert refiner.last_outcome is not None
