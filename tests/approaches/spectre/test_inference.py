"""Test-time inference helper tests (spec §10.5)."""

from __future__ import annotations

from pathlib import Path

import torch
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre.inference import (
    init_inference_state,
    record_failure,
    select_next_skeleton,
)
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz
from alphatamp.approaches.spectre.model import SpectreModel
from alphatamp.approaches.spectre.priors import ZeroPrior
from alphatamp.approaches.spectre.vocab import extract_vocab


def _seed_train_split(tmp_path: Path):
    train = tmp_path / "train"
    ep = build_toy_episode(
        problem_id=0,
        num_blocks=4,
        outcomes=("fail", "fail", "success", "fail"),
    )
    atomic_write_pickle_gz(ep, train / "episodes" / "ep_00000.pkl.gz")
    vocab = extract_vocab(train, "abc")
    return ep, vocab


def test_select_next_skeleton_returns_valid_index(tmp_path: Path) -> None:
    """``select_next_skeleton`` picks an index that is currently in the pool."""
    ep, vocab = _seed_train_split(tmp_path)
    model = SpectreModel(vocab)
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    idx = select_next_skeleton(state, model)
    assert 0 <= idx < len(ep.skeleton_pool)
    assert state.pool_mask[idx].item() is True


def test_record_failure_moves_skeleton_into_F(tmp_path: Path) -> None:
    """After ``record_failure``, the index leaves R and the next pick differs."""
    ep, vocab = _seed_train_split(tmp_path)
    model = SpectreModel(vocab)
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    first = select_next_skeleton(state, model)
    record_failure(state, first)
    assert first in state.fail_indices
    assert not state.pool_mask[first].item()
    # Argmax over R should now skip the failed slot.
    second = select_next_skeleton(state, model)
    assert second != first
    assert state.pool_mask[second].item()


def test_empty_F_uses_c0(tmp_path: Path) -> None:
    """At |F|=0, encode_context should return c_0 — verify by direct equality."""
    ep, vocab = _seed_train_split(tmp_path)
    model = SpectreModel(vocab)
    model.eval()
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    assert not state.fail_indices
    f_emb = torch.zeros(1, 1, state.e_S.size(-1))
    f_mask = torch.zeros(1, 1, dtype=torch.bool)
    with torch.no_grad():
        c = model.encode_context(f_emb, f_mask)
    assert torch.allclose(c[0], model.empty_context.detach(), atol=1e-6)


def test_error_outcomes_excluded_from_pool(tmp_path: Path) -> None:
    """Skeletons with outcome=="error" must start out of the inference pool."""
    train = tmp_path / "train"
    ep = build_toy_episode(
        problem_id=0,
        num_blocks=4,
        outcomes=("fail", "error", "success", "fail"),
    )
    atomic_write_pickle_gz(ep, train / "episodes" / "ep_00000.pkl.gz")
    vocab = extract_vocab(train, "abc")
    model = SpectreModel(vocab)
    state = init_inference_state(model, ep, vocab, prior=ZeroPrior())
    # Skeleton 1 was the error outcome.
    assert not state.pool_mask[1].item()
    # The other three are still in the pool.
    assert state.pool_mask[0].item()
    assert state.pool_mask[2].item()
    assert state.pool_mask[3].item()
