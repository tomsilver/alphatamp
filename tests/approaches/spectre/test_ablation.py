"""Frozen-context ablation tests: freeze seam + comparison metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre import eda
from alphatamp.approaches.spectre.eda import (
    BaselineResult,
    ChoiceStep,
    first_divergence_distribution,
    per_index_agreement,
    success_at_k,
    win_tie_loss,
)
from alphatamp.approaches.spectre.inference import (
    init_inference_state,
    record_failure,
    select_next_skeleton,
)
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz
from alphatamp.approaches.spectre.model import SpectreModel
from alphatamp.approaches.spectre.priors import ZeroPrior
from alphatamp.approaches.spectre.vocab import extract_vocab


def _seed_split(tmp_path: Path, outcomes: tuple[str, ...]):
    split = tmp_path / "train"
    ep = build_toy_episode(problem_id=0, num_blocks=len(outcomes), outcomes=outcomes)
    atomic_write_pickle_gz(ep, split / "episodes" / "ep_00000.pkl.gz")
    vocab = extract_vocab(split, "abc")
    return split, ep, vocab


# ---------------------------------------------------------------------------
# Freeze seam (inference.select_next_skeleton)
# ---------------------------------------------------------------------------


def test_freeze_context_equals_static_ranking(tmp_path: Path) -> None:
    """A frozen-context rollout is a static ranking by the initial logits.

    With c pinned to c_0 the per-skeleton scores never change, so the visit order must
    equal repeatedly popping the masked argmax of the episode-start logits.
    """
    _, ep, vocab = _seed_split(tmp_path, ("fail", "fail", "fail", "fail"))
    model = SpectreModel(vocab)
    model.eval()

    # Independent static ranking from the initial (empty-F) logits.
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    f_emb = torch.zeros(1, 1, state.e_S.size(-1))
    f_mask = torch.zeros(1, 1, dtype=torch.bool)
    with torch.no_grad():
        c0 = model.encode_context(f_emb, f_mask)
        logits = model.score(state.e_S.unsqueeze(0), c0, state.priors.unsqueeze(0))[0]
    mask = state.pool_mask.clone()
    expected_order = []
    while bool(mask.any().item()):
        masked = torch.where(mask, logits, torch.tensor(-float("inf")))
        j = int(masked.argmax().item())
        expected_order.append(j)
        mask[j] = False

    # Frozen-context rollout over the whole (all-fail) pool.
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    frozen_order = []
    while bool(state.pool_mask.any().item()):
        idx = select_next_skeleton(state, model, freeze_context=True)
        frozen_order.append(idx)
        record_failure(state, idx)

    assert frozen_order == expected_order


def test_freeze_context_default_behavior_unchanged(tmp_path: Path) -> None:
    """``freeze_context=False`` (and the 2-arg form) match; at |F|=0 the frozen and live
    variants agree by construction."""
    _, ep, vocab = _seed_split(tmp_path, ("fail", "fail", "success", "fail"))
    model = SpectreModel(vocab)
    model.eval()
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    # Empty F: all three calls must agree (full also scores with c_0).
    assert (
        select_next_skeleton(state, model)
        == select_next_skeleton(state, model, freeze_context=False)
        == select_next_skeleton(state, model, freeze_context=True)
    )
    # Non-empty F: default == explicit freeze_context=False.
    first = select_next_skeleton(state, model)
    record_failure(state, first)
    assert select_next_skeleton(state, model) == select_next_skeleton(
        state, model, freeze_context=False
    )


# ---------------------------------------------------------------------------
# Traced evaluator
# ---------------------------------------------------------------------------


def test_spectre_evaluate_traced_matches_untraced(tmp_path: Path) -> None:
    """Traced evaluation is a strict superset: same BaselineResult arrays,
    plus a trace consistent with the attempts count."""
    split_dir, _, vocab = _seed_split(tmp_path, ("fail", "fail", "success", "fail"))
    split = eda.load_split_episodes(split_dir)
    model = SpectreModel(vocab)
    model.eval()
    for freeze in (False, True):
        plain = eda.spectre_evaluate(
            split, model, vocab, attempt_budget=20, freeze_context=freeze
        )
        traced, traces = eda.spectre_evaluate_traced(
            split, model, vocab, attempt_budget=20, freeze_context=freeze
        )
        assert np.array_equal(plain.attempts, traced.attempts)
        assert np.array_equal(plain.wall_clock, traced.wall_clock)
        assert np.array_equal(plain.censored, traced.censored)
        assert np.array_equal(plain.problem_ids, traced.problem_ids)
        assert len(traces) == len(traced.attempts)
        # Uncensored episode: trace ends at the success, length == attempts.
        assert not traced.censored[0]
        assert len(traces[0]) == int(traced.attempts[0])
        assert traces[0][-1].outcome == "success"
        assert [c.step for c in traces[0]] == list(range(1, len(traces[0]) + 1))


# ---------------------------------------------------------------------------
# Comparison metrics (pure functions on hand-built data)
# ---------------------------------------------------------------------------


def _trace(*idxs: int) -> list[ChoiceStep]:
    steps = [
        ChoiceStep(step=t, idx=i, outcome="fail") for t, i in enumerate(idxs, start=1)
    ]
    return steps


def test_per_index_agreement_handles_unequal_lengths() -> None:
    full = [_trace(0, 1, 2), _trace(0, 2)]
    frozen = [_trace(0, 1, 3), _trace(0)]
    rows = per_index_agreement(full, frozen, max_index=4)
    # t=1: both episodes co-running, both agree (always true at empty F).
    assert rows[0] == (1, 1.0, 2)
    # t=2: only episode 0 co-running (episode 1's frozen trace ended); agree.
    assert rows[1] == (2, 1.0, 1)
    # t=3: episode 0 diverges (2 vs 3).
    assert rows[2] == (3, 0.0, 1)
    # t=4: nobody co-running → nan rate, n=0.
    t, rate, n_co = rows[3]
    assert t == 4 and n_co == 0 and np.isnan(rate)


def test_first_divergence_distribution() -> None:
    full = [_trace(0, 1, 2), _trace(0, 2), _trace(5, 1)]
    frozen = [_trace(0, 1, 3), _trace(0), _trace(5, 1, 9)]
    hist = first_divergence_distribution(full, frozen)
    # Episode 0 diverges at t=3; episode 1 is a prefix relation; episode 2's
    # shared prefix is identical (extra steps beyond the shorter trace are
    # not comparable).
    assert hist == {3: 1, "never": 2}


def _result(attempts: list[float], name: str = "x") -> BaselineResult:
    n = len(attempts)
    arr = np.array(attempts, dtype=float)
    return BaselineResult(
        name=name,
        attempts=arr,
        wall_clock=np.zeros(n),
        censored=arr > 20,
        problem_ids=np.arange(n, dtype=np.int64),
    )


def test_win_tie_loss() -> None:
    a = _result([1, 5, 21, 3])
    b = _result([2, 5, 4, 21])
    assert win_tie_loss(a, b) == (2, 1, 1)


def test_success_at_k_censored_never_solved() -> None:
    # attempts: 1, 3, 21 (censored at budget 20).
    res = _result([1, 3, 21])
    curve = success_at_k(res, k_max=20)
    assert len(curve) == 20
    assert curve[0] == 1 / 3  # K=1
    assert curve[2] == 2 / 3  # K=3
    assert curve[19] == 2 / 3  # K=20 — censored episode never counts
    assert np.all(np.diff(curve) >= 0)  # monotone non-decreasing
