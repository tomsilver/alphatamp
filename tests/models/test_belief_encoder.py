"""Tests for BeliefEncoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from alphatamp.models.belief_encoder import BeliefEncoder


def test_output_shapes() -> None:
    """Forward pass produces correct output shapes and dtypes."""
    d_token = 64
    d_model = 32
    B, M = 4, 10

    model = BeliefEncoder(d_token=d_token, d_model=d_model, n_layers=2)
    model.eval()

    tokens = torch.randn(B, M, d_token)
    pad_mask = torch.zeros(B, M, dtype=torch.bool)
    # Pad last 2 slots in first batch item
    pad_mask[0, -2:] = True

    with torch.no_grad():
        ctx, belief = model(tokens, pad_mask)

    assert ctx.shape == (B, M, d_model)
    assert belief.shape == (B, d_model)
    assert ctx.dtype == torch.float32
    assert belief.dtype == torch.float32


def test_identity_proj_when_dims_match() -> None:
    """No linear projection when d_token == d_model."""
    model = BeliefEncoder(d_token=128, d_model=128)
    assert isinstance(model.input_proj, nn.Identity)

    model2 = BeliefEncoder(d_token=64, d_model=128)
    assert isinstance(model2.input_proj, nn.Linear)


def test_belief_encoder_integrates_history() -> None:
    """Train a linear probe on beta_t to predict a hidden difficulty bit.

    Generates 500 synthetic instances where a hidden binary "difficulty" bit
    determines outcome patterns.  After observing 3 history entries, a jointly
    trained BeliefEncoder + linear probe must achieve >0.85 accuracy on
    predicting the difficulty bit, proving that beta_t integrates history.
    """
    # --- Hyperparameters (small for test speed) ---
    d_model = 32
    M = 10  # skeleton slots
    n_instances = 500
    n_history = 3  # revealed entries per instance
    n_steps = 500
    batch_size = 64
    scale = 1.5  # outcome signal strength

    torch.manual_seed(0)

    # --- Generate synthetic data ---
    difficulty = torch.randint(0, 2, (n_instances,))  # hidden bit per instance

    # Fixed skeleton base embeddings (shared across instances)
    base = torch.randn(M, d_model)

    # Fixed outcome direction vector (normalized)
    direction = torch.randn(d_model)
    direction = direction / direction.norm()

    # Build token tensor: (n_instances, M, d_model)
    tokens = base.unsqueeze(0).expand(n_instances, -1, -1).clone()

    for i in range(n_instances):
        p_success = 0.8 if difficulty[i] == 0 else 0.2
        for j in range(n_history):
            outcome = (torch.rand(1) < p_success).float().item()
            tokens[i, j] += (2 * outcome - 1) * direction * scale

    pad_mask = torch.zeros(n_instances, M, dtype=torch.bool)
    labels = difficulty.float()

    # --- Train BeliefEncoder + linear probe ---
    encoder = BeliefEncoder(
        d_token=d_model,
        d_model=d_model,
        n_heads=4,
        n_layers=4,
        ffn_dim=64,
        dropout=0.0,
    )
    probe = nn.Linear(d_model, 1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(probe.parameters()), lr=1e-3
    )

    encoder.train()
    probe.train()
    for _ in range(n_steps):
        idx = torch.randint(0, n_instances, (batch_size,))
        batch_tokens = tokens[idx]
        batch_mask = pad_mask[idx]
        batch_labels = labels[idx]

        _, beta = encoder(batch_tokens, batch_mask)
        logits = probe(beta).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Evaluate ---
    encoder.eval()
    probe.eval()
    with torch.no_grad():
        _, beta = encoder(tokens, pad_mask)
        logits = probe(beta).squeeze(-1)
        preds = (logits > 0).long()
        accuracy = (preds == difficulty).float().mean().item()

    assert accuracy > 0.85, (
        f"Probe accuracy {accuracy:.3f} <= 0.85 — belief encoder "
        f"is not integrating history into beta_t"
    )
