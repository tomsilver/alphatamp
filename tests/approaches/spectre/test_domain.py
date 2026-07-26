"""The domain contract reproduces DD2D's hand-written literals exactly.

v2.2 hard-codes ``place-buffer`` to derive a candidate's manipulated set and its plan
length, and ``retrieve`` to decide whether a failure licenses demotion. v3 derives all
three from the operator schema plus a per-query axiom declaration. Replacing a literal
with a derivation is only safe if the derivation gives the *same answer*, so these are
whole-corpus identity checks rather than spot checks: any drift shows up as a changed
ranking, which reads like a modelling result instead of a bug.

The corpus-wide tests are ``slow`` and skip when the gitignored collection is absent; the
semantics of the contract itself are covered by fast synthetic tests below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphatamp.approaches.spectre.domain import (
    DOMAINS,
    EMPTY_SPEC,
    QueryAxioms,
    failure_schema,
    length_key,
    manipulated,
    spec_for,
    unmoved,
)

_ROOT = Path(__file__).resolve().parents[3]
_SPLITS = [
    _ROOT / "data" / "spectre" / "raw" / "dd2d_v3" / s for s in ("train", "val", "test")
]


# --------------------------------------------------------------------------- #
# contract semantics (fast, no data)
# --------------------------------------------------------------------------- #
def test_undeclared_query_is_hint_tier() -> None:
    """Silence means "no proof", never "assume sound" -- the safe direction."""
    assert not QueryAxioms().proof_tier()
    assert not EMPTY_SPEC.axioms_for("anything").proof_tier()
    assert not EMPTY_SPEC.axioms_for(None).proof_tier()


def test_proof_tier_requires_all_three_axioms() -> None:
    """Monotone + local + exact. Dropping any one is unsound, and each has a real
    counterexample (non-monotone contact modes; same-surface declutter breaks locality;
    a sampled query is not exhaustive)."""
    assert QueryAxioms(True, True, True).proof_tier()
    for drop in range(3):
        flags = [True, True, True]
        flags[drop] = False
        assert not QueryAxioms(*flags).proof_tier()


def test_dd2d_declares_only_retrieve_as_proof_tier() -> None:
    """`pick` / `place-buffer` are sampled, so their failures are evidence, not proof."""
    spec = spec_for("dd2d_v3")
    assert spec.axioms_for("retrieve").proof_tier()
    assert not spec.axioms_for("pick").proof_tier()
    assert not spec.axioms_for("place-buffer").proof_tier()


def test_unknown_variant_degrades_to_empty_spec() -> None:
    """A new environment must be runnable before it is declared."""
    assert spec_for("some_new_env_v1") is EMPTY_SPEC
    assert set(DOMAINS) >= {"dd2d_v2", "dd2d_v3", "dd2d_v4"}


def test_failure_schema_parses_the_action_string() -> None:
    class _O:
        refiner_metadata = {"failure_action": "place-buffer(o12)"}

    assert failure_schema(_O()) == "place-buffer"

    class _Empty:
        refiner_metadata: dict = {}

    assert failure_schema(_Empty()) is None


def test_budget_exhausted_failure_never_licenses_demotion() -> None:
    """The v2.2 unsoundness, closed.

    On a budget exit the refiner still names ``retrieve(target)`` as the failing action
    although the retrieve was never tested. Trusting that is what let one dd2d_v2
    candidate demote 12 genuinely-feasible plans.
    """
    spec = spec_for("dd2d_v3")

    class _Ran:
        refiner_metadata = {"failure_action": "retrieve(target)"}

    class _TimedOut:
        refiner_metadata = {
            "failure_action": "retrieve(target)",
            "budget_exhausted": True,
        }

    assert spec.licenses_demotion(_Ran())
    assert not spec.licenses_demotion(_TimedOut())


# --------------------------------------------------------------------------- #
# whole-corpus identities (slow; these are what license deleting the literals)
# --------------------------------------------------------------------------- #
def _iter_episodes():
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    for split in _SPLITS:
        if not (split / "episodes").is_dir():
            continue
        for path in list_episodes(split):
            yield load_episode(path)


def _staged_dd2d(skeleton) -> frozenset[str]:
    """v2.2's literal, quoted from ``dataset_v2.py`` / ``evidence.py``."""
    return frozenset(
        op.parameters[0].name
        for op in skeleton.operator_seq
        if op.name == "place-buffer"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (_SPLITS[0] / "episodes").is_dir(), reason="dd2d_v3 collection absent"
)
def test_manipulated_equals_the_place_buffer_literal_on_every_skeleton() -> None:
    """``args(sigma) \\ goal_objects`` == the ``place-buffer`` filter, corpus-wide."""
    from alphatamp.approaches.spectre.domain import goal_objects as goal_objs_fn

    n = 0
    for episode in _iter_episodes():
        goal_objs = goal_objs_fn(episode)
        for skeleton in episode.skeleton_pool:
            assert manipulated(skeleton, goal_objs) == _staged_dd2d(skeleton)
            n += 1
    assert n >= 120_000, f"expected the full corpus, saw {n} skeletons"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_SPLITS[0] / "episodes").is_dir(), reason="dd2d_v3 collection absent"
)
def test_length_key_induces_the_same_buckets_as_the_v2_prior_column() -> None:
    """``len(operator_seq) == 2*|staged| + 1``, so bucketing on plan length partitions
    the pool exactly as v2.2's normalized removal-count column did."""
    n = 0
    for episode in _iter_episodes():
        for skeleton in episode.skeleton_pool:
            assert length_key(skeleton) == 2 * len(_staged_dd2d(skeleton)) + 1
            n += 1
    assert n >= 120_000, f"expected the full corpus, saw {n} skeletons"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_SPLITS[2] / "episodes").is_dir(), reason="dd2d_v3 collection absent"
)
def test_unmoved_at_the_final_step_reduces_to_the_v22_subset_rule() -> None:
    """``U(sigma', j') superset-eq U(sigma, j)`` iff ``staged' subset-eq staged``.

    Pinned because G5's certificate rule is only allowed to differ from v2.2's demotion
    *after* it has been shown to agree with it.
    """
    from alphatamp.approaches.spectre.domain import goal_objects as goal_objs_fn
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    paths = list_episodes(_SPLITS[2])
    checked = 0
    for path in paths[:: max(1, len(paths) // 8)][:8]:
        episode = load_episode(path)
        goal_objs = goal_objs_fn(episode)
        all_objects = frozenset(episode.object_registry)
        pool = episode.skeleton_pool[:40]
        us = [unmoved(s, len(s.operator_seq) - 1, all_objects, goal_objs) for s in pool]
        staged = [_staged_dd2d(s) for s in pool]
        for i, (ui, si) in enumerate(zip(us, staged)):
            for j, (uj, sj) in enumerate(zip(us, staged)):
                assert (uj >= ui) == (sj <= si), (i, j)
                checked += 1
    assert checked > 0


@pytest.mark.slow
@pytest.mark.skipif(
    not (_SPLITS[0] / "episodes").is_dir(), reason="dd2d_v3 collection absent"
)
def test_canonicalize_is_not_idempotent_so_eval_must_load_raw() -> None:
    """Pins the trap that silently skewed every cached comparison number.

    ``canonicalize_episode`` renames objects, and a *second* pass renames them again
    differently (``item_10`` -> ``item_2``). Scene poses survive, but the object->tag
    binding does not -- and tags are the join key the whole representation runs on. The
    comparison cache used to source already-canonicalized episodes from
    ``eda.load_split_episodes`` and hand them to ``build_v2_example``, which canonicalizes
    again, so evaluation ran on a different binding than training (which loads raw).

    This asserts the *non*-idempotence deliberately: the day it becomes idempotent this
    test fails, and whoever fixed it should then simplify the loaders rather than keep a
    workaround nobody can explain.
    """
    from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    def _names(ep) -> list[str]:
        return [o.name for o in ep.scene_geometry.objects]

    differing = 0
    checked = 0
    for path in list_episodes(_SPLITS[0])[:20]:
        raw = load_episode(path)
        if raw.scene_geometry is None:
            continue
        once = canonicalize_episode(raw, rng=None)
        twice = canonicalize_episode(once, rng=None)
        checked += 1
        # poses are stable; only the naming/order moves
        assert [o.pose for o in once.scene_geometry.objects] == [
            o.pose for o in twice.scene_geometry.objects
        ]
        if _names(once) != _names(twice):
            differing += 1
    assert checked > 0
    assert differing > 0, (
        "canonicalize_episode now looks idempotent -- if that is a real fix, drop the "
        "_RawSplit workaround in precompute_dd2d_cache and delete this test"
    )
