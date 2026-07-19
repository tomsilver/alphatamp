"""Plackett-Luce loss numerical tests (spec §11.1.6)."""

from __future__ import annotations

import math

import torch

from alphatamp.approaches.spectre.loss import (
    plackett_luce_loss,
    within_length_pl_loss,
)


def test_pl_loss_zero_when_logit_concentrated_on_success() -> None:
    """``logits=[1e6, 0, 0]`` with success_mask ``[T, F, F]`` → loss ≈ 0."""
    logits = torch.tensor([[1e6, 0.0, 0.0]])
    succ = torch.tensor([[True, False, False]])
    pool = torch.tensor([[True, True, True]])
    loss = plackett_luce_loss(logits, succ, pool)
    assert float(loss) < 1e-3


def test_pl_loss_huge_when_logit_concentrated_on_failure() -> None:
    """``logits=[0, 1e6, 0]`` with success on idx 0 → loss ≈ 1e6."""
    logits = torch.tensor([[0.0, 1e6, 0.0]])
    succ = torch.tensor([[True, False, False]])
    pool = torch.tensor([[True, True, True]])
    loss = plackett_luce_loss(logits, succ, pool)
    assert float(loss) > 1e5


def test_pl_loss_uniform_logits_equals_log_k_over_succ() -> None:
    """Uniform logits → ``loss = log(K / |SUCC_R|)``."""
    k = 5
    logits = torch.zeros(1, k)
    succ = torch.tensor([[True, True, False, False, False]])  # |SUCC|=2
    pool = torch.tensor([[True] * k])
    loss = plackett_luce_loss(logits, succ, pool)
    expected = math.log(k / 2)
    assert math.isclose(float(loss), expected, rel_tol=1e-5)


def test_pl_loss_respects_pool_mask() -> None:
    """Padded R-slots (pool_mask False) must not contribute to Z."""
    logits = torch.tensor([[1e6, 0.0, 1e6]])
    succ = torch.tensor([[True, False, False]])
    pool = torch.tensor([[True, True, False]])  # last slot is padding
    loss = plackett_luce_loss(logits, succ, pool)
    # Without the pool mask, idx 2's huge logit would dominate Z and make loss
    # large; with the mask it's ignored, so the success at idx 0 dominates.
    assert float(loss) < 1e-3


def test_pl_loss_batched_mean() -> None:
    """Multi-example loss is the mean of per-example losses."""
    logits = torch.tensor([[1e6, 0.0, 0.0], [0.0, 0.0, 1e6]])
    succ = torch.tensor([[True, False, False], [False, False, True]])
    pool = torch.tensor([[True, True, True], [True, True, True]])
    loss = plackett_luce_loss(logits, succ, pool)
    assert float(loss) < 1e-3


def test_within_length_only_penalizes_within_bucket() -> None:
    """A success ranked below an infeasible of the SAME length is penalized; a success
    ranked below an infeasible of a DIFFERENT length is not (length carries no info)."""
    # buckets by length key: candidates 0,1 length=1.0 ; candidates 2,3 length=2.0
    length_key = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    pool = torch.tensor([[True, True, True, True]])
    # within bucket-1: success at idx0 ranked BELOW infeasible idx1 -> penalized.
    logits_bad = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    succ = torch.tensor([[True, False, False, False]])
    bad = within_length_pl_loss(logits_bad, succ, pool, length_key)
    # within bucket-1: success idx0 ranked ABOVE infeasible idx1 -> ~0.
    logits_good = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    good = within_length_pl_loss(logits_good, succ, pool, length_key)
    assert float(good) < float(bad)
    assert float(good) < 1e-2


def test_within_length_ignores_cross_bucket_ordering() -> None:
    """A long infeasible ranked above a short success does NOT add within-length loss
    (that is the cross-length signal the global PL handles)."""
    length_key = torch.tensor([[1.0, 2.0]])  # different lengths
    pool = torch.tensor([[True, True]])
    succ = torch.tensor([[True, False]])  # success is the short one
    # the long infeasible (idx1) is ranked far above the short success (idx0)...
    logits = torch.tensor([[0.0, 9.0]])
    # ...but they are in different buckets, each of size 1 -> no within-length term.
    assert float(within_length_pl_loss(logits, succ, pool, length_key)) == 0.0
