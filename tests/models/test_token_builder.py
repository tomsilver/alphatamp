"""Tests for OutcomeEncoder and TokenBuilder."""

from __future__ import annotations

import torch

from alphatamp.models.token_builder import OutcomeEncoder, TokenBuilder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_token_builder(
    d_skel: int = 16,
    d_out: int = 8,
) -> TokenBuilder:
    torch.manual_seed(0)
    tb = TokenBuilder(d_skel=d_skel, d_out=d_out, dropout=0.0)
    tb.eval()
    return tb


def _make_inputs(
    B: int = 2,
    M: int = 6,
    d_skel: int = 16,
) -> dict[str, torch.Tensor]:
    """Return a base set of inputs.

    Layout per instance (same for both batch items):
        slots 0,1   — inapplicable  (applicability=0)
        slots 2,3   — candidates    (applicable, not revealed)
        slots 4,5   — history       (applicable, revealed)
    """
    torch.manual_seed(1)
    skeleton_embeds = torch.randn(B, M, d_skel)

    applicability = torch.ones(B, M)
    applicability[:, :2] = 0.0  # slots 0,1 inapplicable

    revealed_mask = torch.zeros(B, M, dtype=torch.bool)
    revealed_mask[:, 4:] = True  # slots 4,5 revealed

    y = torch.zeros(B, M)
    y[:, 4] = 1.0  # slot 4 succeeded
    y[:, 5] = 0.0  # slot 5 failed

    f = torch.zeros(B, M)
    f[:, 4] = 1.0
    f[:, 5] = 0.4

    t = torch.zeros(B, M)
    t[:, 4] = 2.5
    t[:, 5] = 1.1

    lengths = torch.full((B, M), 5.0)

    return dict(
        skeleton_embeds=skeleton_embeds,
        applicability=applicability,
        revealed_mask=revealed_mask,
        y=y, f=f, t=t, lengths=lengths,
    )


# ---------------------------------------------------------------------------
# OutcomeEncoder basic test
# ---------------------------------------------------------------------------

def test_outcome_encoder_shape() -> None:
    """Forward pass produces correct output shape."""
    d_out = 12
    enc = OutcomeEncoder(d_out=d_out, dropout=0.0)
    enc.eval()

    B, M = 3, 5
    y = torch.rand(B, M)
    f = torch.rand(B, M)
    t = torch.rand(B, M)
    lengths = torch.randint(1, 10, (B, M)).float()

    with torch.no_grad():
        out = enc(y, f, t, lengths)

    assert out.shape == (B, M, d_out)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# TokenBuilder shape test
# ---------------------------------------------------------------------------

def test_token_builder_output_shape() -> None:
    """Token tensor has shape (B, M, d_skel + d_out)."""
    d_skel, d_out = 16, 8
    tb = _make_token_builder(d_skel=d_skel, d_out=d_out)
    inputs = _make_inputs(d_skel=d_skel)

    with torch.no_grad():
        tokens = tb(**inputs)

    B, M = inputs["applicability"].shape
    assert tokens.shape == (B, M, d_skel + d_out)
    assert tokens.dtype == torch.float32


def test_d_token_property() -> None:
    """d_token property matches d_skel + d_out."""
    tb = _make_token_builder(d_skel=16, d_out=8)
    assert tb.d_token == 24


# ---------------------------------------------------------------------------
# (i) Candidate tokens unchanged when histories differ
# ---------------------------------------------------------------------------

def test_candidate_tokens_unchanged_by_history() -> None:
    """Candidate-position tokens must be identical across different histories.

    Slots 2,3 are candidates in both calls.  We change the revealed outcomes
    (and which slots are revealed) — candidate tokens must not change.
    """
    d_skel, d_out = 16, 8
    tb = _make_token_builder(d_skel=d_skel, d_out=d_out)
    inputs_a = _make_inputs(d_skel=d_skel)

    # Second call: reveal slot 3 as well, with different outcomes at slot 4,5
    inputs_b = _make_inputs(d_skel=d_skel)
    inputs_b["revealed_mask"][:, 3] = True
    inputs_b["y"][:, 3] = 1.0
    inputs_b["f"][:, 3] = 1.0
    inputs_b["t"][:, 3] = 3.0
    inputs_b["y"][:, 4] = 0.0
    inputs_b["f"][:, 4] = 0.3
    inputs_b["t"][:, 4] = 0.5

    with torch.no_grad():
        tokens_a = tb(**inputs_a)
        tokens_b = tb(**inputs_b)

    # Slot 2 is a candidate in both calls
    cand_a = tokens_a[:, 2, :]
    cand_b = tokens_b[:, 2, :]
    assert torch.equal(cand_a, cand_b), (
        f"Candidate tokens differ — max delta: "
        f"{(cand_a - cand_b).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# (ii) History tokens change when outcomes change
# ---------------------------------------------------------------------------

def test_history_tokens_change_with_outcomes() -> None:
    """Changing Y/F/T at revealed positions must change the token values."""
    d_skel, d_out = 16, 8
    tb = _make_token_builder(d_skel=d_skel, d_out=d_out)
    inputs_a = _make_inputs(d_skel=d_skel)

    # Same revealed_mask, different outcomes at slot 4
    inputs_b = _make_inputs(d_skel=d_skel)
    inputs_b["y"][:, 4] = 0.0
    inputs_b["f"][:, 4] = 0.2
    inputs_b["t"][:, 4] = 10.0

    with torch.no_grad():
        tokens_a = tb(**inputs_a)
        tokens_b = tb(**inputs_b)

    hist_a = tokens_a[:, 4, :]
    hist_b = tokens_b[:, 4, :]
    assert not torch.equal(hist_a, hist_b), (
        "History tokens should change when outcomes change"
    )


# ---------------------------------------------------------------------------
# (iii) Inapplicable tokens never change with t
# ---------------------------------------------------------------------------

def test_inapplicable_tokens_invariant() -> None:
    """Inapplicable tokens must be identical regardless of history state.

    We vary revealed_mask and outcome values across two calls — inapplicable
    slots (0, 1) must produce bitwise-identical tokens.
    """
    d_skel, d_out = 16, 8
    tb = _make_token_builder(d_skel=d_skel, d_out=d_out)
    inputs_a = _make_inputs(d_skel=d_skel)

    # Second call: completely different history
    inputs_b = _make_inputs(d_skel=d_skel)
    inputs_b["revealed_mask"][:, 2:] = True  # reveal everything applicable
    inputs_b["y"][:, 2] = 1.0
    inputs_b["f"][:, 2] = 1.0
    inputs_b["t"][:, 2] = 5.0
    inputs_b["y"][:, 3] = 0.0
    inputs_b["f"][:, 3] = 0.6
    inputs_b["t"][:, 3] = 0.8

    with torch.no_grad():
        tokens_a = tb(**inputs_a)
        tokens_b = tb(**inputs_b)

    # Inapplicable slots 0 and 1
    inapp_a = tokens_a[:, :2, :]
    inapp_b = tokens_b[:, :2, :]
    assert torch.equal(inapp_a, inapp_b), (
        f"Inapplicable tokens differ — max delta: "
        f"{(inapp_a - inapp_b).abs().max().item():.2e}"
    )
