"""SPECTRE model smoke tests (spec §11.1).

Covers smoke tests #1–5; PL-loss saturation (#6) lives in ``test_loss.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from _fixtures import write_toy_split

from alphatamp.approaches.spectre.dataset import (
    SpectreDataset,
    collate_spectre_batch,
)
from alphatamp.approaches.spectre.model import (
    D_MODEL,
    SpectreModel,
    build_model_info,
)
from alphatamp.approaches.spectre.priors import ZeroPrior
from alphatamp.approaches.spectre.vocab import Vocab, extract_vocab


def _toy_fixture(tmp_path: Path):
    train = tmp_path / "train"
    write_toy_split(
        train,
        outcomes_per_problem=[
            ("fail", "fail", "success"),
            ("success", "fail", "fail", "fail"),
            ("fail", "success", "fail", "fail"),
        ],
    )
    vocab = extract_vocab(train, config_hash="abc")
    ds = SpectreDataset(
        split_dir=train,
        prior=ZeroPrior(),
        seed=0,
        augment=False,
        num_f_samples_per_epoch=1,
    )
    return ds, vocab


def test_forward_shape_and_grad_flow(tmp_path: Path) -> None:
    """#1: forward returns finite (B, R) and gradient flows to all params."""
    ds, vocab = _toy_fixture(tmp_path)
    batch = [ds[i] for i in range(min(2, len(ds)))]
    sb = collate_spectre_batch(batch, vocab)
    model = SpectreModel(vocab)
    logits = model(sb)
    assert logits.dim() == 2
    assert logits.shape[0] == len(batch)
    assert torch.all(torch.isfinite(logits))
    loss = logits.sum()
    loss.backward()
    has_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    # All trainable params should have received a gradient under .sum().
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(has_grad) == n_total


def test_empty_f_returns_c0(tmp_path: Path) -> None:
    """#2: encode_context returns broadcast c_0 when |F|=0 for every row."""
    _, vocab = _toy_fixture(tmp_path)
    model = SpectreModel(vocab)
    model.eval()
    f_emb = torch.zeros(3, 1, D_MODEL)  # one synthetic slot
    f_mask = torch.zeros(3, 1, dtype=torch.bool)  # all False ⇒ empty F
    with torch.no_grad():
        c = model.encode_context(f_emb, f_mask)
    expected = model.empty_context.detach()
    for i in range(3):
        assert torch.allclose(c[i], expected, atol=1e-6)


def test_augmentation_skips_non_augmentable_types(tmp_path: Path) -> None:
    """#3: with width/size frozen, those local ids never permute under aug.

    Toy domain doesn't have width/size, so we synthesize a policy that
    pins ``block`` and verify ``block`` local ids are stable across calls.
    """
    train = tmp_path / "train"
    write_toy_split(train, [("fail", "fail", "success")])
    ds = SpectreDataset(
        split_dir=train,
        prior=ZeroPrior(),
        seed=12345,
        augment=True,
        type_aug_policy={"block": False, "robot": False},  # freeze both
        num_f_samples_per_epoch=1,
    )
    seen_block_orderings: set[tuple[str, ...]] = set()
    for epoch in range(5):
        ds.set_epoch(epoch)
        ex = ds[0]
        # Object names after canonicalization look like "block_<idx>".
        block_names = tuple(
            sorted(name for name in ex.object_registry if name.startswith("block_"))
        )
        seen_block_orderings.add(block_names)
    # Frozen → only one canonical block ordering ever.
    assert len(seen_block_orderings) == 1


def test_augmentation_permutes_augmentable_types(tmp_path: Path) -> None:
    """Sanity check the inverse: with policy missing, augmentation does run.

    We verify that the *operator-arg ordering inside skeletons* changes
    across epochs even though the registry is just ``{name: type}``. We
    look at the operator_seq as a tuple of ``(op.name, parameter names)``.
    """
    train = tmp_path / "train"
    write_toy_split(train, [("fail", "fail", "fail", "success")])
    ds = SpectreDataset(
        split_dir=train,
        prior=ZeroPrior(),
        seed=12345,
        augment=True,
        type_aug_policy=None,  # ⇒ all augmentable
        num_f_samples_per_epoch=1,
    )
    seen: set[tuple[str, ...]] = set()
    for epoch in range(10):
        ds.set_epoch(epoch)
        ex = ds[0]
        sig = tuple(
            f"{op.name}({','.join(p.name for p in op.parameters)})"
            for skel in ex.r_skeletons
            for op in skel.operator_seq
        )
        seen.add(sig)
    # With augmentation on, we should see >1 distinct ordering across 10 epochs.
    assert len(seen) > 1


def test_vocab_arity_drives_mlp_input_dims(tmp_path: Path) -> None:
    """#5: ``A=7, P=4`` ⇒ op-MLP in_features=32+7*16+16=160 and atom in=32+4*24=128."""
    del tmp_path
    fake_vocab = Vocab(
        config_hash="test",
        operators={"<OOV>": 0, "Foo": 1},
        predicates={"<OOV>": {"arity": 0, "idx": 0}, "P": {"arity": 4, "idx": 1}},
        types={"<OOV>": 0, "thing": 1},
        max_operator_arity=7,
        max_predicate_arity=4,
        max_skeleton_length=4,
        max_atoms_per_state=4,
        max_objects_per_state=8,
        max_pool_size=4,
        max_objects_per_type={"thing": 8},
        type_aug_policy={},
    )
    model = SpectreModel(fake_vocab)
    info = build_model_info(model)
    assert info.op_mlp_in_features == 32 + 7 * 16 + 16  # = 160
    assert info.atom_proj_in_features == 32 + 4 * (8 + 16)  # = 128


def test_prior_dropout_zeros_priors_in_training(tmp_path: Path) -> None:
    """When prior_dropout=True and training, priors are zeroed for whole examples.

    With ``p_drop=1.0``, every row is dropped, so the σ output should be
    identical to when priors are 0.
    """
    _, vocab = _toy_fixture(tmp_path)
    model = SpectreModel(vocab, prior_dropout_p=1.0)
    model.train()
    e_R = torch.randn(2, 3, D_MODEL)
    c = torch.randn(2, D_MODEL)
    priors = torch.randn(2, 3)
    with torch.no_grad():
        out_with = model.score(e_R, c, priors, prior_dropout=True)
        out_zero = model.score(e_R, c, torch.zeros_like(priors), prior_dropout=False)
    assert torch.allclose(out_with, out_zero, atol=1e-5)
