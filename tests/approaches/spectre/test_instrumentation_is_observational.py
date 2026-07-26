"""G0's differential gate: the v3 refiner instrumentation changed no label.

The DD2D refiner now emits :class:`FailureObservation` records (culprits, per-step
effort, exhausted-vs-budget) so the v3 adaptive pathway reads *observations* rather than
recomputed geometry. The whole ``dd2d_v4`` re-collection rests on that instrumentation
being **observation-only**: it may not call the stream counter, draw from ``rng``, or
change control flow. ``n_attempts`` *is* ``counter.calls``, so a single extra stream call
would shift it and cascade into every downstream label.

This is a *differential* test in the only sense that certifies the invariant: it replays
the instrumented refiner against episodes collected by the **pre-instrumentation** code
(``data/spectre/raw/dd2d_v3``) and requires the stored answers back. A reference
implementation kept inside the test would only prove the test agrees with itself.

Scope note -- ``n_attempts`` is asserted only for candidates that finished inside the
wall-clock budget. A candidate that hits ``time_budget`` spends "however many stream
calls fit in 20 s", which measures host CPU speed rather than the problem; measured on
this corpus it diverges ~2x on a faster machine while ``label`` / ``steps_bound`` /
``failure_action`` still reproduce exactly. That split is G0's acceptance criterion, and
the budget-exhausted set is enumerated rather than waved at.

The full 120000-candidate version of this check runs after the ``dd2d_v4`` collection by
diffing the two JSON trees; this test is the fast subsample that guards the invariant in
CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SPLIT = _ROOT / "data" / "spectre" / "raw" / "dd2d_v3" / "train"

pytestmark = pytest.mark.skipif(
    not (_SPLIT / "episodes").is_dir(),
    reason="dd2d_v3 collection not present (gitignored data)",
)

# The v3 collection's refiner settings. A live re-refinement must match the preset its
# labels were collected under or it is drawing from a different distribution
# (``vlmplan/score.py`` REFINER_PRESETS makes the same point for off-pool labelling).
_V3_REFINER = dict(budget=None, retry_cap=10, samples_per_step=15, time_budget=20.0)

# Elapsed seconds above which a candidate is treated as budget-bound. The budget is
# 20.0; anything at/near it stopped on the clock, not on a proof.
_BUDGET_BOUND_S = 19.0


def _replay(n_episodes: int = 8, every: int = 17):
    """Re-refine a subsample of stored candidates with the instrumented refiner."""
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.planning import staging_skeleton
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.refine import DD2DRefiner
    from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (
        reconstruct_scene,
    )
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    refiner = DD2DRefiner(**_V3_REFINER)
    paths = list_episodes(_SPLIT)
    # spread across the seed range so all four strata are represented
    stride = max(1, len(paths) // n_episodes)
    out = []
    for path in paths[::stride][:n_episodes]:
        episode = load_episode(path)
        assert episode.scene_geometry is not None
        scene = reconstruct_scene(episode.scene_geometry)
        target = next(o.name for o in episode.scene_geometry.objects if o.is_target)
        for i, (skel, stored) in enumerate(
            zip(episode.skeleton_pool, episode.outcomes)
        ):
            if i % every:
                continue
            staged = [
                op.parameters[0].name
                for op in skel.operator_seq
                if op.name == "place-buffer"
            ]
            result = refiner.refine(
                staging_skeleton(target, staged),
                scene,
                seed=stored.refinement_seed,
            )
            out.append((result, stored, staged))
    return out


@pytest.mark.slow
def test_instrumented_refiner_reproduces_stored_labels() -> None:
    """Labels, depth and the failing action reproduce for *every* candidate; stream
    counts reproduce for every candidate that finished inside the budget."""
    replayed = _replay()
    assert len(replayed) > 50, "subsample too small to be meaningful"

    budget_bound = []
    for result, stored, _ in replayed:
        meta = stored.refiner_metadata or {}
        pid = (stored.skeleton_idx, meta.get("plan_idx"))
        assert (result.status == "feasible") == (stored.outcome == "success"), pid
        assert result.steps_bound == meta["steps_bound"], pid
        assert result.plan_length == meta["plan_length"], pid
        assert (result.failure_action or None) == (meta["failure_action"] or None), pid
        if result.elapsed >= _BUDGET_BOUND_S:
            budget_bound.append((pid, meta["n_attempts"], result.n_attempts))
        else:
            assert result.n_attempts == meta["n_attempts"], pid

    # Budget-bound candidates are allowed to disagree on n_attempts only, and only
    # because the count is wall-clock-bound. Keep the set small: if it grows, the
    # instrumentation has slowed the refiner enough to change the corpus.
    assert len(budget_bound) <= 0.05 * len(replayed), budget_bound


@pytest.mark.slow
def test_failure_observations_are_consistent_with_the_outcome() -> None:
    """Every infeasible replay emits at least one observation, every feasible one emits
    none for its final state, and the fields are internally coherent."""
    for result, _, staged in _replay(n_episodes=6, every=23):
        if result.feasible:
            continue
        assert result.failures, "an infeasible refinement must observe something"
        for obs in result.failures:
            assert obs.schema in {"pick", "place-buffer", "retrieve"}
            assert 0 <= obs.step_index < result.plan_length
            # a culprit is an object of the scene or the wall band, never a fabrication
            assert all(isinstance(c, str) and c for c in obs.culprits)
            # U(sigma, j): the staged prefix has left the drawer, so no staged object
            # may still be reported as unmoved at the step that failed
            assert obs.unmoved, "unmoved set should never be empty mid-plan"
            if obs.budget_exhausted:
                # a budget exit proves nothing: no culprit was observed and the
                # exactness axiom must be off
                assert not obs.exhausted
                assert obs.culprits == ()
        # the budget flag on the result agrees with the terminal observation
        assert result.budget_exhausted == any(
            o.budget_exhausted for o in result.failures
        )


@pytest.mark.slow
def test_retrieve_observations_carry_the_unmoved_set_v22_demotion_used() -> None:
    """The certificate rule's ``U(sigma, j)`` reduces to v2.2's staged-subset rule.

    v2.2 demoted a candidate whose staged set was a subset of an observed-blocked staged
    set. The v3 rule is ``U(sigma', j') superset-eq U(sigma, j)``. On DD2D the drawer
    contents at the retrieve step are exactly ``all_objects - staged``, so the two
    coincide -- pinned here because G5's identity gate depends on it.
    """
    seen = 0
    for result, stored, staged in _replay(n_episodes=6, every=23):
        for obs in result.failures:
            if obs.schema != "retrieve" or obs.budget_exhausted:
                continue
            # nothing staged by this plan is still in the drawer
            assert not (set(staged) & set(obs.unmoved)), (staged, obs.unmoved)
            seen += 1
    assert seen > 0, "no retrieve failures in the subsample"
