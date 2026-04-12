"""Tests for prediction heads and composite NLL loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, LogNormal

from alphatamp.models.losses import PredictionNLLLoss
from alphatamp.models.prediction_heads import FHead, JointYHead, THead, YHead


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 32
B, M = 4, 10


def _random_ctx_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    ctx = torch.randn(B, M, D_MODEL)
    pad_mask = torch.zeros(B, M, dtype=torch.bool)
    pad_mask[0, -2:] = True  # pad last 2 in first batch item
    return ctx, pad_mask


# ===================================================================
# Shape tests
# ===================================================================


def test_yhead_output_shape() -> None:
    """YHead produces (B, M) float32 logits."""
    head = YHead(D_MODEL, dropout=0.0)
    head.eval()
    ctx, mask = _random_ctx_and_mask()
    with torch.no_grad():
        out = head(ctx, mask)
    assert out.shape == (B, M)
    assert out.dtype == torch.float32


def test_fhead_output_distribution() -> None:
    """FHead produces Beta distribution with batch_shape (B, M)."""
    head = FHead(D_MODEL, dropout=0.0)
    head.eval()
    ctx, mask = _random_ctx_and_mask()
    with torch.no_grad():
        dist = head(ctx, mask)
    assert isinstance(dist, Beta)
    assert dist.batch_shape == (B, M)


def test_thead_output_distribution() -> None:
    """THead produces LogNormal distribution with batch_shape (B, M)."""
    head = THead(D_MODEL, dropout=0.0)
    head.eval()
    ctx, mask = _random_ctx_and_mask()
    with torch.no_grad():
        dist = head(ctx, mask)
    assert isinstance(dist, LogNormal)
    assert dist.batch_shape == (B, M)


def test_jointyhead_output_shape() -> None:
    """JointYHead produces (B, M) float32 corrected logits."""
    head = JointYHead(D_MODEL, n_heads=4, rank=8, dropout=0.0)
    head.eval()
    ctx, mask = _random_ctx_and_mask()
    marginal = torch.randn(B, M)
    with torch.no_grad():
        out = head(ctx, marginal, mask)
    assert out.shape == (B, M)
    assert out.dtype == torch.float32


# ===================================================================
# Property tests
# ===================================================================


def test_fhead_concentrations_positive() -> None:
    """FHead α, β are strictly > concentration_floor."""
    floor = 0.05
    head = FHead(D_MODEL, dropout=0.0, concentration_floor=floor)
    head.eval()
    ctx, mask = _random_ctx_and_mask()
    with torch.no_grad():
        dist = head(ctx, mask)
    assert (dist.concentration1 > floor).all()
    assert (dist.concentration0 > floor).all()


def test_thead_sigma_positive() -> None:
    """THead σ is strictly > sigma_floor."""
    floor = 0.01
    head = THead(D_MODEL, dropout=0.0, sigma_floor=floor)
    head.eval()
    ctx, mask = _random_ctx_and_mask()
    with torch.no_grad():
        dist = head(ctx, mask)
    assert (dist.scale > floor).all()


def test_jointyhead_respects_pad_mask() -> None:
    """Padding positions do not affect non-padding outputs."""
    torch.manual_seed(7)
    head = JointYHead(D_MODEL, n_heads=4, rank=8, dropout=0.0)
    head.eval()

    ctx = torch.randn(1, 6, D_MODEL)
    marginal = torch.randn(1, 6)
    pad_mask = torch.zeros(1, 6, dtype=torch.bool)
    pad_mask[0, 4:] = True  # last 2 are padding

    with torch.no_grad():
        out1 = head(ctx, marginal, pad_mask).clone()

    # Change values at padding positions
    ctx2 = ctx.clone()
    ctx2[0, 4:] = torch.randn(2, D_MODEL) * 100
    with torch.no_grad():
        out2 = head(ctx2, marginal, pad_mask)

    # Non-padding outputs should be identical
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), (
        f"Max diff: {(out1[0, :4] - out2[0, :4]).abs().max().item():.2e}"
    )


# ===================================================================
# Loss tests
# ===================================================================


def _make_loss_inputs(
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Build synthetic inputs for PredictionNLLLoss."""
    torch.manual_seed(0)
    B_, M_ = 3, 8
    ctx = torch.randn(B_, M_, D_MODEL, device=device)

    y_head = YHead(D_MODEL, dropout=0.0).to(device)
    f_head = FHead(D_MODEL, dropout=0.0).to(device)
    t_head = THead(D_MODEL, dropout=0.0).to(device)
    pad_mask = torch.zeros(B_, M_, dtype=torch.bool, device=device)

    y_logits = y_head(ctx, pad_mask)
    f_dist = f_head(ctx, pad_mask)
    t_dist = t_head(ctx, pad_mask)

    y_true = torch.randint(0, 2, (B_, M_), device=device).float()
    f_true = torch.rand(B_, M_, device=device)
    f_true[y_true == 1] = 1.0  # Y=1 => F=1
    t_true = torch.rand(B_, M_, device=device) * 10 + 0.1

    applicable_mask = torch.ones(B_, M_, dtype=torch.bool, device=device)
    applicable_mask[:, -2:] = False  # last 2 inapplicable
    revealed_mask = torch.ones(B_, M_, dtype=torch.bool, device=device)
    revealed_mask[:, 3:5] = False  # slots 3-4 unrevealed

    return dict(
        y_logits=y_logits, f_dist=f_dist, t_dist=t_dist,
        y_true=y_true, f_true=f_true, t_true=t_true,
        applicable_mask=applicable_mask, revealed_mask=revealed_mask,
        heads=(y_head, f_head, t_head),
    )


def test_loss_output_keys() -> None:
    """PredictionNLLLoss returns dict with expected keys and scalar tensors."""
    inputs = _make_loss_inputs()
    loss_fn = PredictionNLLLoss()
    result = loss_fn(
        inputs["y_logits"], inputs["f_dist"], inputs["t_dist"],
        inputs["y_true"], inputs["f_true"], inputs["t_true"],
        inputs["applicable_mask"], inputs["revealed_mask"],
    )
    assert set(result.keys()) == {"loss", "loss_y", "loss_f", "loss_t"}
    for k, v in result.items():
        assert v.dim() == 0, f"{k} should be scalar, got shape {v.shape}"
        assert torch.isfinite(v), f"{k} is not finite: {v.item()}"


def test_loss_masks_inapplicable() -> None:
    """Changing targets on inapplicable entries does not change loss."""
    torch.manual_seed(1)
    inputs = _make_loss_inputs()
    loss_fn = PredictionNLLLoss()

    result1 = loss_fn(
        inputs["y_logits"], inputs["f_dist"], inputs["t_dist"],
        inputs["y_true"], inputs["f_true"], inputs["t_true"],
        inputs["applicable_mask"], inputs["revealed_mask"],
    )

    # Flip targets on inapplicable entries
    y2 = inputs["y_true"].clone()
    f2 = inputs["f_true"].clone()
    t2 = inputs["t_true"].clone()
    inapp = ~inputs["applicable_mask"]
    y2[inapp] = 1.0 - y2[inapp]
    f2[inapp] = 1.0 - f2[inapp]
    t2[inapp] = t2[inapp] * 100

    result2 = loss_fn(
        inputs["y_logits"], inputs["f_dist"], inputs["t_dist"],
        y2, f2, t2,
        inputs["applicable_mask"], inputs["revealed_mask"],
    )

    assert torch.allclose(result1["loss"], result2["loss"]), (
        f"Loss changed: {result1['loss'].item():.6f} vs {result2['loss'].item():.6f}"
    )


def test_loss_t_only_on_success() -> None:
    """T loss is zero when all y_true == 0."""
    inputs = _make_loss_inputs()
    loss_fn = PredictionNLLLoss()

    y_all_fail = torch.zeros_like(inputs["y_true"])
    result = loss_fn(
        inputs["y_logits"], inputs["f_dist"], inputs["t_dist"],
        y_all_fail, inputs["f_true"], inputs["t_true"],
        inputs["applicable_mask"], inputs["revealed_mask"],
    )
    assert result["loss_t"].item() == 0.0


def test_loss_gradient_flows() -> None:
    """All head parameters receive non-zero gradients from composite loss."""
    torch.manual_seed(2)
    B_, M_ = 4, 8
    ctx = torch.randn(B_, M_, D_MODEL)
    pad_mask = torch.zeros(B_, M_, dtype=torch.bool)

    y_head = YHead(D_MODEL, dropout=0.0)
    f_head = FHead(D_MODEL, dropout=0.0)
    t_head = THead(D_MODEL, dropout=0.0)
    joint_head = JointYHead(D_MODEL, n_heads=4, rank=8, dropout=0.0)
    loss_fn = PredictionNLLLoss()

    marginal = y_head(ctx, pad_mask)
    y_logits = joint_head(ctx, marginal, pad_mask)
    f_dist = f_head(ctx, pad_mask)
    t_dist = t_head(ctx, pad_mask)

    # Targets with some Y=1 to activate T loss
    y_true = torch.tensor(
        [[1, 0, 1, 0, 1, 0, 1, 0]] * B_, dtype=torch.float32,
    )
    f_true = torch.where(y_true == 1, torch.ones_like(y_true), torch.rand(B_, M_))
    t_true = torch.rand(B_, M_) * 5 + 0.1
    applicable_mask = torch.ones(B_, M_, dtype=torch.bool)
    revealed_mask = torch.ones(B_, M_, dtype=torch.bool)

    result = loss_fn(
        y_logits, f_dist, t_dist,
        y_true, f_true, t_true,
        applicable_mask, revealed_mask,
    )
    result["loss"].backward()

    all_heads = [y_head, f_head, t_head, joint_head]
    for head in all_heads:
        for name, param in head.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert param.grad.abs().max() > 0, f"{name} has zero gradient"


# ===================================================================
# Calibration test
# ===================================================================


def test_yhead_calibration_synthetic() -> None:
    """Train YHead on synthetic data and check decile calibration.

    Generates N instances with M candidates, each candidate j having a fixed
    true P(Y_j=1). After training, predicted probabilities binned into deciles
    must match empirical frequencies within 0.05.
    """
    torch.manual_seed(42)

    N = 2000  # instances
    M_ = 20  # candidates per instance
    d = 32
    n_steps = 1000
    batch_size = 64

    # --- Synthetic data ---
    # True success probability per candidate (shared across instances)
    true_p = torch.rand(M_) * 0.9 + 0.05  # [0.05, 0.95]

    # Sample binary outcomes: (N, M)
    y_true = torch.bernoulli(true_p.unsqueeze(0).expand(N, -1))

    # Embeddings that encode true_p: base per candidate + per-instance noise
    # Use a fixed linear mapping so the info is learnable
    base_embed = torch.randn(M_, d)
    # Add signal: scale one direction by true_p
    signal_dir = torch.randn(d)
    signal_dir = signal_dir / signal_dir.norm()
    for j in range(M_):
        base_embed[j] += true_p[j] * signal_dir * 3.0

    # (N, M, d): shared base + small noise per instance
    ctx = base_embed.unsqueeze(0).expand(N, -1, -1).clone()
    ctx += torch.randn(N, M_, d) * 0.3

    pad_mask = torch.zeros(N, M_, dtype=torch.bool)

    # --- Train YHead ---
    head = YHead(d, hidden_dim=64, dropout=0.0)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    head.train()
    for _ in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        logits = head(ctx[idx], pad_mask[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_true[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Evaluate calibration ---
    head.eval()
    with torch.no_grad():
        all_logits = head(ctx, pad_mask)
        pred_probs = torch.sigmoid(all_logits)  # (N, M)

    # Flatten for binning
    pred_flat = pred_probs.reshape(-1)
    y_flat = y_true.reshape(-1)

    # Bin into 10 deciles
    bin_edges = torch.linspace(0, 1, 11)
    bucket_idx = torch.bucketize(pred_flat, bin_edges) - 1
    bucket_idx = bucket_idx.clamp(0, 9)

    max_error = 0.0
    bins_checked = 0
    for b in range(10):
        in_bin = bucket_idx == b
        count = in_bin.sum().item()
        if count < 20:
            continue
        mean_pred = pred_flat[in_bin].mean().item()
        empirical_freq = y_flat[in_bin].mean().item()
        error = abs(mean_pred - empirical_freq)
        max_error = max(max_error, error)
        bins_checked += 1
        assert error < 0.05, (
            f"Decile {b}: predicted={mean_pred:.3f}, "
            f"empirical={empirical_freq:.3f}, error={error:.3f} >= 0.05"
        )

    assert bins_checked >= 5, (
        f"Only {bins_checked} decile bins had >= 20 samples; "
        f"need at least 5 populated bins for meaningful calibration check"
    )
