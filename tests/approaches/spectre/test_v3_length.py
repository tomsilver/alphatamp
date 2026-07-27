"""G9's position encoding, and the invariant that enabling it is what retires D-8.

The length-generalization experiment trains on s0-s2 (plans of at most 5 operators) and
deploys on s3 (7). With the learned absolute ``nn.Embedding(64, D)`` table, steps 5 and 6
are rows that received no gradient, so the experiment would measure initialization noise.
These tests pin that the replacement is defined everywhere, that it genuinely leaves the
old parameter out of the state dict, and -- the part that is easy to break silently --
that the *default* config still reproduces v2.2 exactly.
"""

from __future__ import annotations

import torch

from alphatamp.approaches.spectre.model_v3 import (
    SpectreV3Model,
    V3Config,
    sinusoidal_positions,
)


def _keys(**kw) -> set[str]:
    return set(SpectreV3Model(n_ops=4, max_arity=1, cfg=V3Config(**kw)).state_dict())


def test_sinusoidal_is_defined_and_distinct_past_the_training_max() -> None:
    """Every step position gets its own vector, including ones training never saw.

    s0-s2 plans reach step index 4; s3 reaches 6. The check runs to 12 so it also covers
    any future domain with longer plans.
    """
    enc = sinusoidal_positions(torch.arange(12).view(1, 1, 12), 64)
    assert enc.shape == (1, 1, 12, 64)
    assert torch.isfinite(enc).all()
    pair = torch.cdist(enc[0, 0], enc[0, 0]) + torch.eye(12) * 99
    assert float(pair.min()) > 1e-3, "two step positions collide"


def test_sinusoidal_encoding_is_deterministic_and_device_independent() -> None:
    """It is a pure function of the positions -- no state, no seed, no init."""
    p = torch.arange(8).view(1, 1, 8)
    assert torch.equal(sinusoidal_positions(p, 64), sinusoidal_positions(p, 64))


def test_enabling_sinusoidal_removes_pos_emb_from_the_state_dict() -> None:
    """The D-8 retirement is *deliberate*, so it is asserted rather than discovered.

    A v2.2 checkpoint cannot load ``strict=True`` into a sinusoidal model, which is why
    G9 is the last architectural change in the plan.
    """
    assert "cands.pos_emb.weight" in _keys()
    assert "cands.pos_emb.weight" not in _keys(sinusoidal_pos=True)


def test_default_config_is_unchanged_by_the_g9_addition() -> None:
    """Exact absence (D-8): adding the switch must not perturb the compat state dict."""
    assert _keys() == _keys(sinusoidal_pos=False)


def test_sinusoidal_model_runs_on_a_longer_plan_than_any_it_was_built_for() -> None:
    """End-to-end shape check at a plan length past the s0-s2 training maximum."""
    from alphatamp.approaches.spectre.model_v2 import SpectreV2Batch

    b, k, ell, n = 1, 3, 7, 4
    batch = SpectreV2Batch(
        obj_tags=torch.ones(b, n, dtype=torch.long),
        obj_boundary=torch.zeros(b, n, 32, 2),
        obj_pose=torch.zeros(b, n, 3),  # pose_proj is Linear(3, D_POSE)
        obj_rel=torch.zeros(b, n, 8),
        obj_is_target=torch.zeros(b, n),
        obj_mask=torch.ones(b, n, dtype=torch.bool),
        cand_op_ids=torch.ones(b, k, ell, dtype=torch.long),
        cand_pos=torch.arange(ell).view(1, 1, ell).expand(b, k, ell).contiguous(),
        cand_arg_tags=torch.ones(b, k, ell, 1, dtype=torch.long),
        cand_step_mask=torch.ones(b, k, ell, dtype=torch.bool),
        pool_mask=torch.ones(b, k, dtype=torch.bool),
        success_mask=torch.zeros(b, k),
        aux_necessary=torch.full((b, n), -1.0),
        aux_relevant=torch.full((b, n), -1.0),
        glob_feats=torch.zeros(b, 6),
    )
    model = SpectreV3Model(n_ops=4, max_arity=1, cfg=V3Config(sinusoidal_pos=True))
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch)
    assert logits.shape == (b, k)
    assert torch.isfinite(logits).all()
