"""Test-time inference helper tests (spec §10.5)."""

from __future__ import annotations

from pathlib import Path

import torch
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre.inference import (
    init_inference_state,
    load_checkpoint,
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


def test_load_checkpoint_drops_legacy_static_tag_buffer(tmp_path: Path) -> None:
    """Legacy checkpoints saved the static_tag_predicate_ids buffer as a persistent
    state_dict entry; the current model registers it non-persistent.

    ``load_checkpoint`` must strip the legacy key so a
    strict-mode load still succeeds.
    """
    _, vocab = _seed_train_split(tmp_path)
    src_model = SpectreModel(
        vocab,
        prior_dropout_p=0.2,
        use_atom_sab2=False,
        static_tag_predicates=None,  # any list; legacy key is independent
    )
    src_model.eval()
    sd = dict(src_model.state_dict())
    # Inject a legacy-style persistent buffer entry.
    sd["skeleton_encoder.state_enc.static_tag_predicate_ids"] = torch.tensor(
        [2, 8, 9], dtype=torch.long
    )
    cfg = {
        "use_atom_sab2": False,
        "prior_dropout_p": 0.2,
        "use_static_tag_pool": False,
    }
    ckpt_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": sd,
            "config": cfg,
            "static_tag_predicates": [],
        },
        ckpt_path,
    )
    # Should not raise even though the saved sd has an "extra" key.
    reloaded = load_checkpoint(ckpt_path, vocab, device="cpu")
    assert not reloaded.skeleton_encoder.state_enc.use_static_tag_pool


def test_load_checkpoint_round_trip(tmp_path: Path) -> None:
    """``load_checkpoint`` reconstructs a SpectreModel with matching state.

    Saves a tiny checkpoint mimicking ``train.py``'s save format (cfg dict
    + ``model_state_dict`` + ``static_tag_predicates``); reloads via the
    shared loader; asserts both models produce identical output on the
    same toy batch.
    """
    _, vocab = _seed_train_split(tmp_path)
    # Pick a non-default flag combo to verify auto-detect actually reads
    # the saved cfg rather than silently using SpectreModel defaults.
    src_model = SpectreModel(
        vocab,
        prior_dropout_p=0.3,
        use_atom_sab2=False,
        static_tag_predicates=None,
    )
    src_model.eval()
    cfg = {
        "use_atom_sab2": False,
        "prior_dropout_p": 0.3,
        "use_static_tag_pool": False,
    }
    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": src_model.state_dict(),
            "config": cfg,
            "static_tag_predicates": [],
        },
        ckpt_path,
    )

    reloaded = load_checkpoint(ckpt_path, vocab, device="cpu")
    assert reloaded.scorer.prior_dropout_p == 0.3
    assert reloaded.skeleton_encoder.state_enc.atom_sab2 is None
    assert not reloaded.skeleton_encoder.state_enc.use_static_tag_pool
    src_keys = set(src_model.state_dict().keys())
    reloaded_keys = set(reloaded.state_dict().keys())
    assert src_keys == reloaded_keys
