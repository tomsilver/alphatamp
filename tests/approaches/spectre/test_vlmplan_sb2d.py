"""Tests for the StickButton2D VLMPlan adapter.

Every case here guards a failure that is *silent* — the run completes, a row lands in the
comparison table, and the number is wrong. Three of the four were real bugs during
bring-up:

- grounding that accepts an inapplicable plan turns a precondition violation into a
  refinement attempt, so the arm is charged for work no planner would have done;
- ``pool_index`` that fails to match a stored candidate sends an *in-pool* plan down the
  live off-pool refinement path, which is what dropped the label-agreement gate to 0.571;
- ``canonical_key`` that ignores press order collapses two genuinely different plans into
  one, deflating the proposal count §6 reports.

They use a real episode from the collection rather than a synthetic one: the thing under
test is agreement between the adapter and the stored pool, which a fixture I wrote myself
cannot demonstrate.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest

from alphatamp.approaches.spectre.baselines.vlmplan.adapter import RawPlan
from alphatamp.approaches.spectre.baselines.vlmplan.registry import make_adapter
from alphatamp.approaches.spectre.baselines.vlmplan.sb2d_adapter import _lifted_by_name

_EPISODES = Path("data/spectre/raw/stickbutton2d_v1/test/episodes")


def _episode():
    paths = sorted(_EPISODES.glob("*.pkl.gz"))
    if not paths:
        pytest.skip("stickbutton2d_v1 test collection is not present")
    # The LAST file, not the first: problem ids are banded by stratum, so `paths[0]` is
    # always b1 -- a 2-candidate pool that would pass every assertion below vacuously.
    # (*Stride, never truncate*, `decisions.md` 2026-07-27.)
    with gzip.open(paths[-1], "rb") as handle:
        return pickle.load(handle)


@pytest.fixture(name="adapter")
def _adapter():
    return make_adapter("stickbutton2d_v1", with_images=False)


@pytest.fixture(name="episode")
def _episode_fixture():
    return _episode()


def _raw(steps) -> RawPlan:
    return RawPlan(
        steps=tuple((name, tuple(args)) for name, args in steps),
        block_index=0,
        text="",
    )


def test_ground_accepts_a_stored_pool_plan(adapter, episode) -> None:
    """A planner-emitted skeleton must survive the adapter's precondition check.

    If it does not, the adapter is stricter than the collection and every VLM proposal is
    rejected for reasons the pool itself would fail.
    """
    plan = adapter.published_order(episode)[0]
    assert adapter.ground(_raw(plan), episode) is not None


def test_ground_rejects_an_inapplicable_plan(adapter, episode) -> None:
    """`RobotPressButtonFromButton` needs the robot already at the *from* button.

    Issued first, its precondition is unmet. Accepting it would let the arm bill a
    refinement attempt for a plan no skeleton generator would ever emit.
    """
    buttons = sorted(n for n, t in adapter.objects(episode).items() if t == "circle")
    bogus = [("RobotPressButtonFromButton", ["robot", buttons[0], buttons[1]])]
    assert adapter.ground(_raw(bogus), episode) is None


def test_pool_index_matches_every_stored_candidate(adapter, episode) -> None:
    """The in-pool/off-pool split must be exact.

    A key that fails to match sends a stored candidate down live refinement, which reads
    as env drift in the label-agreement gate rather than as the lookup bug it is.
    """
    index = adapter.pool_index(episode)
    order = adapter.published_order(episode)
    assert len(index) == len(order)
    for j, plan in enumerate(order):
        assert index[adapter.canonical_key(plan)] == j


def test_canonical_key_distinguishes_press_orderings(adapter, episode) -> None:
    """Reordering presses is a different plan here, not a duplicate.

    The robot presses what it drives over, so order determines feasibility; keying on the
    unordered set would merge a feasible plan with an infeasible one.
    """
    plan = adapter.published_order(episode)[0]
    if len(plan) < 2:
        pytest.skip("pool[0] is a single-step plan; nothing to reorder")
    swapped = (plan[1], plan[0]) + tuple(plan[2:])
    assert adapter.canonical_key(swapped) != adapter.canonical_key(plan)


def test_published_order_is_the_collection_order(adapter, episode) -> None:
    """Index j of the fallback must be cache index j.

    The published-order fill charges attempts against stored labels by position; an
    off-by-one here silently scores each fallback attempt against a different candidate.
    """
    order = adapter.published_order(episode)
    assert len(order) == len(episode.skeleton_pool)
    for j, plan in enumerate(order):
        expected = tuple(
            (op.parent.name, tuple(o.name for o in op.parameters))
            for op in episode.skeleton_pool[j].operator_seq
        )
        assert tuple(plan) == expected


# --------------------------------------------------------------------------- #
# Grounding must judge a proposal against the DOMAIN, not against the filtered pool.
#
# Both cases below returned "inapplicable" before 2026-08-01 and cost the b5 pilots
# every one of their proposals. Neither failure is visible in a result: the plans are
# discarded, VLMPlan falls back to published order, and the row just looks weak.
# --------------------------------------------------------------------------- #
def _buttons(adapter, episode) -> list[str]:
    return sorted(n for n, t in adapter.objects(episode).items() if t == "circle")


def test_place_stick_is_groundable_even_though_no_pooled_plan_uses_it(
    episode,
) -> None:
    """The acyclic pool filter removes every skeleton containing a stick cycle.

    Recovering lifted operators from the pool alone therefore leaves ``PlaceStick``
    undefined, and *every* proposal using it is rejected — including the
    press-far-with-stick-then-near-with-arm strategy the model actually writes.
    """
    lifted = _lifted_by_name(episode)
    assert "PlaceStick" in lifted
    pooled = {
        op.parent.name
        for skel in episode.skeleton_pool
        for op in skel.operator_seq
        if op.parent is not None
    }
    # The regression only bites when the pool really is missing it; if a future
    # collection includes stick cycles this test still passes but stops proving much.
    if "PlaceStick" in pooled:
        pytest.skip("this pool contains PlaceStick; the gap this guards is absent")


def test_mixed_stick_then_arm_plan_grounds(adapter, episode) -> None:
    """`PlaceStick` restores "over no button", so the next press is `...FromNothing`.

    Pins the corrected chaining rule the prompt states. The old rule ("every later press
    is `...FromButton`") is false across a `PlaceStick`, and a model obeying it emits
    plans that cannot ground.
    """
    buttons = _buttons(adapter, episode)
    if len(buttons) < 3:
        pytest.skip("needs >= 3 buttons to exercise a mixed plan")
    steps = [
        ("PickStickFromNothing", ["robot", "stick"]),
        ("StickPressButtonFromNothing", ["robot", "stick", buttons[0]]),
        ("PlaceStick", ["robot", "stick"]),
        ("RobotPressButtonFromNothing", ["robot", buttons[1]]),
    ]
    steps += [
        ("RobotPressButtonFromButton", ["robot", b, buttons[i + 1]])
        for i, b in enumerate(buttons[2:])
    ]
    assert adapter.ground(_raw(steps), episode) is not None


def test_from_button_after_place_stick_is_rejected(adapter, episode) -> None:
    """The converse: `...FromButton` straight after `PlaceStick` must NOT ground.

    Without this the test above would pass on an adapter that had simply stopped
    checking preconditions.
    """
    buttons = _buttons(adapter, episode)
    if len(buttons) < 2:
        pytest.skip("needs >= 2 buttons")
    steps = [
        ("PickStickFromNothing", ["robot", "stick"]),
        ("StickPressButtonFromNothing", ["robot", "stick", buttons[0]]),
        ("PlaceStick", ["robot", "stick"]),
        ("RobotPressButtonFromButton", ["robot", buttons[1], buttons[0]]),
    ]
    assert adapter.ground(_raw(steps), episode) is None
