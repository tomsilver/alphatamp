"""Gate-1 tests: the config-gated 3D scene widening.

A ``point_dim=3``/``pose_dim=4`` model consumes a 3D point-cloud batch and forwards to
``(B, K)`` logits, while the default (2/3) config is byte-identical to the 2D path (the
input Linears keep their old shapes). Pure-tensor, no sim.
"""

from __future__ import annotations

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset import _collate_base
from alphatamp.approaches.spectre.encoders import N_BOUNDARY_POINTS
from alphatamp.approaches.spectre.model import SpectreConfig, SpectreModel


def _example(point_dim: int, pose_dim: int, n_obj: int = 3, k: int = 2):
    import dataclasses

    from alphatamp.approaches.spectre.dataset import SpectreExample

    p = N_BOUNDARY_POINTS
    return SpectreExample(
        obj_tags=np.arange(1, n_obj + 1, dtype=np.int64),
        obj_boundary=np.random.randn(n_obj, p, point_dim).astype(np.float32),
        obj_pose=np.random.randn(n_obj, pose_dim).astype(np.float32),
        obj_rel=np.random.randn(n_obj, 3).astype(np.float32),
        obj_is_goal=np.array([1.0] + [0.0] * (n_obj - 1), dtype=np.float32),
        op_ids=[[1, 2] for _ in range(k)],
        arg_tags=[[[1, 2], [2, 3]] for _ in range(k)],
        success=[True, False][:k],
        aux_necessary=np.full(n_obj, -1.0, np.float32),
        aux_relevant=np.full(n_obj, -1.0, np.float32),
        avail=[True] * k,
        fact_type_ids=[],
        fact_tier_ids=[],
        fact_arg_tags=[],
        prior=[[0.0, 0.0] for _ in range(k)],
        overlap=[[0.0, 0.0] for _ in range(k)],
    )


def _forward(point_dim: int, pose_dim: int) -> torch.Tensor:
    cfg = SpectreConfig(point_dim=point_dim, pose_dim=pose_dim)
    model = SpectreModel(n_ops=5, max_arity=2, cfg=cfg).eval()
    batch = _collate_base(
        [_example(point_dim, pose_dim), _example(point_dim, pose_dim)], max_arity=2
    )
    with torch.no_grad():
        out = model(batch)
    logits = out[0] if isinstance(out, tuple) else out
    return logits


def test_3d_model_forwards_on_point_cloud_batch() -> None:
    logits = _forward(point_dim=3, pose_dim=4)
    assert logits.shape == (2, 2)  # (B, K)
    assert torch.isfinite(logits).all()


def test_3d_encoder_input_widths() -> None:
    model = SpectreModel(
        n_ops=5, max_arity=2, cfg=SpectreConfig(point_dim=3, pose_dim=4)
    )
    assert model.scene.footprint.point_mlp[0].in_features == 3
    assert model.scene.pose_proj.in_features == 4


def test_2d_default_widths_unchanged() -> None:
    model = SpectreModel(n_ops=5, max_arity=2, cfg=SpectreConfig())
    assert model.scene.footprint.point_mlp[0].in_features == 2
    assert model.scene.pose_proj.in_features == 3
    logits = _forward(point_dim=2, pose_dim=3)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
