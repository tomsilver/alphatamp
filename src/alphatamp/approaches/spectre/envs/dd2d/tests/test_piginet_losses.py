"""Step-7 gate: PIGINet losses match hand-computed values / have the right minima."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from alphatamp.approaches.spectre.envs.dd2d.piginet.losses import (
    focal_loss,
    listwise_ranking_loss,
    weighted_bce,
)


def test_weighted_bce_matches_reference():
    logits = torch.tensor([0.5, -1.0, 2.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    got = weighted_bce(logits, labels, pos_weight=50.0)
    ref = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(50.0)
    )
    assert torch.allclose(got, ref)


def test_focal_loss_hand_computed():
    # logit 0 -> p=0.5; label 1 -> p_t=0.5, ce=-ln .5, focal weight (1-.5)^2=.25
    loss = focal_loss(torch.tensor([0.0]), torch.tensor([1.0]), gamma=2.0)
    expected = 0.25 * (-math.log(0.5))
    assert abs(loss.item() - expected) < 1e-6


def test_focal_downweights_easy_negative():
    # an easy (confident-correct) negative contributes far less than a hard one
    easy = focal_loss(torch.tensor([-8.0]), torch.tensor([0.0]))
    hard = focal_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    assert easy.item() < 0.01 * hard.item()


def test_listwise_ranking_minimised_when_positive_top():
    group = torch.tensor([0, 0, 0])
    labels = torch.tensor([0.0, 1.0, 0.0])  # positive is index 1
    top = listwise_ranking_loss(
        torch.tensor([0.0, 5.0, 0.0]), group, labels
    )  # pos highest
    bot = listwise_ranking_loss(
        torch.tensor([5.0, 0.0, 5.0]), group, labels
    )  # pos lowest
    assert top.item() < bot.item()
    assert top.item() < 0.05  # near-zero when the positive dominates


def test_listwise_skips_groups_without_positive():
    group = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])  # only group 0 has a positive
    loss = listwise_ranking_loss(torch.tensor([3.0, 0.0, 1.0, 2.0]), group, labels)
    assert torch.isfinite(loss) and loss.item() >= 0
