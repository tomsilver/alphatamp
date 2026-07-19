"""Tests for the proof-demotion filter (Step 10)."""

from __future__ import annotations

import numpy as np

from alphatamp.approaches.spectre.proof_demotion import (
    ProofState,
    demote,
    demote_scores,
)


def _fs(*xs):
    return frozenset(xs)


def test_demote_is_a_permutation_never_drops():
    order = [0, 1, 2, 3, 4]
    out = demote(order, dead={1, 3})
    assert sorted(out) == sorted(order)  # completeness: nothing dropped
    assert out == [0, 2, 4, 1, 3]  # live (in order) then dead (in order)


def test_demote_all_dead_keeps_full_pool():
    order = [2, 0, 1]
    out = demote(order, dead={0, 1, 2})
    assert sorted(out) == [0, 1, 2] and out == [
        2,
        0,
        1,
    ]  # unchanged order, still complete


def test_blocked_at_contents_demotes_subsets_only():
    # candidate 0 stages {a,b}; fails with target still blocked ⇒ any S ⊆ {a,b} is dead.
    subsets = [_fs("a", "b"), _fs("a"), _fs("a", "b", "c"), _fs("c")]
    st = ProofState(subsets=subsets)
    st.observe_failure(0, blocked=True, pack_impossible=False)
    assert st.is_dead(0)  # F itself
    assert st.is_dead(1)  # {a} ⊆ {a,b}
    assert not st.is_dead(2)  # {a,b,c} ⊋ {a,b} — removes MORE, may clear
    assert not st.is_dead(3)  # {c} ⊄ {a,b}


def test_pack_impossible_demotes_supersets_only():
    # subset {a,b} provably cannot pack ⇒ any S ⊇ {a,b} is dead.
    subsets = [_fs("a", "b"), _fs("a", "b", "c"), _fs("a"), _fs("a", "c")]
    st = ProofState(subsets=subsets)
    st.observe_failure(0, blocked=False, pack_impossible=True)
    assert st.is_dead(0) and st.is_dead(1)  # F and superset
    assert not st.is_dead(2) and not st.is_dead(3)  # {a},{a,c} do not contain {a,b}


def test_monotonicity_vs_bruteforce_blocked():
    rng = np.random.default_rng(0)
    items = list("abcde")
    subsets = [
        frozenset(rng.choice(items, size=rng.integers(1, 4), replace=False))
        for _ in range(30)
    ]
    st = ProofState(subsets=subsets)
    f = 7
    st.observe_failure(f, blocked=True, pack_impossible=False)
    brute = {i for i, s in enumerate(subsets) if s <= subsets[f]}
    assert st.dead == brute


def test_demote_scores_finite_offset_never_neg_inf():
    scores = np.array([1.0, 2.0, 3.0])
    out = demote_scores(scores, dead={2})
    assert np.isfinite(out).all()
    assert out[2] < out[0] and out[2] < out[1]  # ranks last but finite
