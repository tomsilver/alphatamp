"""Tests for the PointSetEncoder upgrade (docs/pointset_encoder_upgrade.md).

T1-T7 from the doc's §6, plus config-off byte-additivity and a checkpoint round-trip.
Pure-tensor / analytic-shape, no simulator.

Note on the two confirmed deviations from the doc (see the plan / ADR):
- T7 is **relaxed**: the residual LayerNorm is kept, so with ``out_proj`` zeroed the
  EdgeConv branch reduces to ``LayerNorm(h)`` (the message path is provably inert), not
  to the exact identity ``h``. We assert the former.
- The 3D orientation oracle is **away-from-origin** (convex box centered at the item
  frame), not the doc's unavailable sensor viewpoint.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset import (
    SpectreExample,
    _box_inside,
    _collate_base,
    _shapely_inside,
    compute_point_feats,
    resample_ring,
)
from alphatamp.approaches.spectre.encoders import (
    N_BOUNDARY_POINTS,
    EdgeConv,
    PointSetEncoder,
    point_feat_dim,
)
from alphatamp.approaches.spectre.model import SpectreConfig, SpectreModel
from alphatamp.approaches.spectre.vocab import Vocab

P = N_BOUNDARY_POINTS


# --------------------------------------------------------------------------- #
# analytic shapes
# --------------------------------------------------------------------------- #
def _circle_ring(r: float = 0.3, n: int = P) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1).astype(np.float32)


def _square_ring(h: float = 0.2, n: int = P) -> np.ndarray:
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    return resample_ring(corners, n)


def _horseshoe_ring(
    R: float = 0.4, r: float = 0.2, n_arc: int = 18, gap: float = 0.7
) -> np.ndarray:
    th = np.linspace(gap, 2.0 * np.pi - gap, n_arc)
    outer = np.stack([R * np.cos(th), R * np.sin(th)], axis=1)
    inner = np.stack([r * np.cos(th[::-1]), r * np.sin(th[::-1])], axis=1)
    return np.concatenate([outer, inner], axis=0).astype(np.float32)


def _box_cloud(hx: float = 0.025, hy: float = 0.025, hz: float = 0.025) -> np.ndarray:
    # A cube (the real ``small_half``): well-proportioned, so PCA normals are clean at
    # the deployed 3D k=6. The tall block (0.025²x0.12) is PCA-degenerate at 32 pts at
    # any k -- documented as the 3D-kNN guardrail in decisions/07 2026-08-18, not
    # asserted clean here (its F3 signal is height, carried by coords, not normals).
    from alphatamp.approaches.spectre.envs.restock3d.scene_geometry import (
        object_point_cloud,
    )

    return object_point_cloud((hx, hy, hz))


# --------------------------------------------------------------------------- #
# example / batch harness (mirrors test_scene_3d_widening)
# --------------------------------------------------------------------------- #
def _example(
    *,
    scene_3d: bool = False,
    use_pca: bool = True,
    use_edge: bool = True,
    n_obj: int = 3,
    k: int = 2,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    point_dim = 3 if scene_3d else 2
    pose_dim = 4 if scene_3d else 3
    c_pt = point_feat_dim(use_pca, point_dim)
    kk = 6 if scene_3d else 4
    pf = rng.standard_normal((n_obj, P, c_pt)).astype(np.float32)
    knn = (
        np.tile(np.arange(kk, dtype=np.int64), (n_obj, P, 1)) % P if use_edge else None
    )
    return SpectreExample(
        obj_tags=np.arange(1, n_obj + 1, dtype=np.int64),
        obj_boundary=rng.standard_normal((n_obj, P, point_dim)).astype(np.float32),
        obj_pose=rng.standard_normal((n_obj, pose_dim)).astype(np.float32),
        obj_rel=rng.standard_normal((n_obj, 3)).astype(np.float32),
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
        point_feats=pf,
        knn_idx=knn,
    )


def _v1_example(n_obj: int = 3, k: int = 2, seed: int = 0):
    """A config-off example: no point_feats/knn_idx (v1 FootprintEncoder path)."""
    rng = np.random.default_rng(seed)
    return SpectreExample(
        obj_tags=np.arange(1, n_obj + 1, dtype=np.int64),
        obj_boundary=rng.standard_normal((n_obj, P, 2)).astype(np.float32),
        obj_pose=rng.standard_normal((n_obj, 3)).astype(np.float32),
        obj_rel=rng.standard_normal((n_obj, 3)).astype(np.float32),
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


# --------------------------------------------------------------------------- #
# T1 — config-off equivalence / additivity
# --------------------------------------------------------------------------- #
def test_t1_config_off_selects_footprint() -> None:
    model = SpectreModel(n_ops=5, max_arity=2, cfg=SpectreConfig())
    assert model.scene.use_pointset is False
    assert hasattr(model.scene, "footprint")
    assert not hasattr(model.scene, "pointset")
    # No PointSetEncoder / EdgeConv / MultiSeedPMA keys leak into the default state dict.
    keys = list(model.state_dict().keys())
    assert not any("pointset" in k for k in keys)
    assert any(k.startswith("scene.footprint.") for k in keys)


def test_t1_config_off_batch_has_no_pointset_fields() -> None:
    batch = _collate_base([_v1_example(seed=1), _v1_example(seed=2)], max_arity=2)
    assert batch.point_feats is None
    assert batch.knn_idx is None


def test_t1_config_off_forward_deterministic() -> None:
    torch.manual_seed(0)
    m1 = SpectreModel(n_ops=5, max_arity=2, cfg=SpectreConfig()).eval()
    torch.manual_seed(0)
    m2 = SpectreModel(n_ops=5, max_arity=2, cfg=SpectreConfig()).eval()
    batch = _collate_base([_v1_example(seed=3), _v1_example(seed=4)], max_arity=2)
    with torch.no_grad():
        o1 = m1(batch)[0]
        o2 = m2(batch)[0]
    assert torch.equal(o1, o2)


def test_pointset_on_selects_pointset() -> None:
    cfg = SpectreConfig(use_pca_feats=True, use_edgeconv=True, pma_seeds=4)
    model = SpectreModel(n_ops=5, max_arity=2, cfg=cfg)
    assert model.scene.use_pointset is True
    assert hasattr(model.scene, "pointset")
    assert not hasattr(model.scene, "footprint")


# --------------------------------------------------------------------------- #
# T2 — feature correctness on analytic shapes
# --------------------------------------------------------------------------- #
def test_t2_circle() -> None:
    ring = _circle_ring()
    feats, _ = compute_point_feats(
        ring, _shapely_inside(list(map(tuple, ring))), 4, False, True
    )
    assert feats.shape == (P, 6)  # C_pt = 6 in 2D
    normal = feats[:, 2:4]
    radial = ring / np.linalg.norm(ring, axis=1, keepdims=True)
    dot = (normal * radial).sum(1)
    assert dot.min() > 0.98  # normals radial-outward
    khat = feats[:, 4]
    assert (khat > 0).all()  # convex everywhere
    assert khat.std() < 0.05  # roughly constant
    assert abs(float(khat.mean()) - float(np.tanh(2 * np.pi / 32))) < 0.1
    assert feats[:, 5].max() < 0.05  # flatness f ~ 0 on a smooth curve


def test_t2_square() -> None:
    ring = _square_ring()
    feats, _ = compute_point_feats(
        ring, _shapely_inside(list(map(tuple, ring))), 4, False, True
    )
    khat = feats[:, 4]
    assert khat.max() > 0.3  # positive spikes at corners
    assert khat.min() > -0.05  # convex: no strictly-reflex points
    # mid-edge points (the median-curvature mass) are ~flat
    assert float(np.median(khat)) < 0.05


def test_t2_horseshoe() -> None:
    ring = _horseshoe_ring()
    pts = resample_ring(list(map(tuple, ring)), P)
    feats, _ = compute_point_feats(
        pts, _shapely_inside(list(map(tuple, ring))), 4, False, True
    )
    khat = feats[:, 4]
    rad = np.linalg.norm(pts, axis=1)
    inner = rad < 0.3  # pocket-facing (concave) arc
    outer = rad > 0.35  # outer (convex) arc
    assert float(khat[inner].mean()) < 0.0  # concave pocket: negative curvature
    assert float(khat[outer].mean()) > 0.3  # outer arc: positive curvature


def test_t2_box_3d() -> None:
    cloud = _box_cloud()
    feats, knn = compute_point_feats(cloud, _box_inside(cloud), 6, True, True)
    assert feats.shape == (P, 8)  # C_pt = 8 in 3D
    assert knn.shape == (P, 6)
    normal = feats[:, 3:6]
    # away-from-origin orientation: dot(n, p) >= 0 for a convex box centered at origin.
    assert (normal * cloud).sum(1).min() >= -1e-4
    f = feats[:, 6]
    assert (f >= 0).all() and f.max() <= 1.0 / 3.0 + 1e-4  # Pauly surface variation
    assert (feats[:, 7] == 0.0).all()  # pad column


# --------------------------------------------------------------------------- #
# T3 — PCA vs exact ring cross-check + Gauss-Bonnet (dev-only, ring order allowed)
# --------------------------------------------------------------------------- #
def test_t3_pca_tangent_matches_ring_tangent_circle() -> None:
    ring = _circle_ring()
    feats, _ = compute_point_feats(
        ring, _shapely_inside(list(map(tuple, ring))), 4, False, True
    )
    normal = feats[:, 2:4]
    # PCA tangent (rot90 of the oriented normal) vs exact ring tangent (central diff).
    tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=1)
    nxt = np.roll(ring, -1, axis=0)
    prv = np.roll(ring, 1, axis=0)
    ring_tan = nxt - prv
    ring_tan /= np.linalg.norm(ring_tan, axis=1, keepdims=True)
    align = np.abs((tangent * ring_tan).sum(1))  # |cos| — sign of tangent is free
    assert align.min() > 0.98


def test_t3_gauss_bonnet_turning_angles() -> None:
    for ring in (_circle_ring(), _square_ring(), _horseshoe_ring()):
        nxt = np.roll(ring, -1, axis=0)
        prv = np.roll(ring, 1, axis=0)
        a = nxt - ring
        b = ring - prv
        ang = np.arctan2(
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],  # cross
            (a * b).sum(1),  # dot
        )
        # A simple closed polygon turns by ±2π in total (Hopf Umlaufsatz); sign is the
        # traversal orientation, so compare the magnitude.
        assert abs(abs(float(ang.sum())) - 2.0 * np.pi) < 1e-3


def test_t3_validate_flag_passes_on_clean_shapes() -> None:
    for ring in (_circle_ring(), _square_ring()):
        compute_point_feats(
            ring, _shapely_inside(list(map(tuple, ring))), 4, False, True, validate=True
        )
    cloud = _box_cloud()
    compute_point_feats(cloud, _box_inside(cloud), 6, True, True, validate=True)


# --------------------------------------------------------------------------- #
# T4 — end-to-end permutation invariance (encoder level)
# --------------------------------------------------------------------------- #
def test_t4_permutation_invariance() -> None:
    torch.manual_seed(0)
    c_pt, n_obj, k = 6, 2, 3
    enc = PointSetEncoder(
        c_pt, use_edgeconv=True, use_point_sab=True, pma_seeds=4
    ).eval()
    pf = torch.randn(1, n_obj, P, c_pt)
    knn = torch.randint(0, P, (1, n_obj, P, k))
    knn[..., 0] = torch.arange(P).view(1, 1, P)  # self at column 0
    mask = torch.ones(1, n_obj, dtype=torch.bool)
    perm = torch.randperm(P)
    inv = torch.argsort(perm)
    pf_p = pf[:, :, perm, :]
    knn_p = inv[knn[:, :, perm, :]]  # remap neighbor indices consistently
    with torch.no_grad():
        d0 = enc(pf, knn, mask)
        d1 = enc(pf_p, knn_p, mask)
    assert torch.allclose(d0, d1, atol=1e-4)


# --------------------------------------------------------------------------- #
# T5 — orientation correctness (the observable consequence of sign-robustness)
# --------------------------------------------------------------------------- #
def test_t5_orientation_outward_convex() -> None:
    # The inside test fixes the PCA normal outward regardless of eigh's arbitrary sign.
    ring = _circle_ring()
    feats, _ = compute_point_feats(
        ring, _shapely_inside(list(map(tuple, ring))), 4, False, True
    )
    radial = ring / np.linalg.norm(ring, axis=1, keepdims=True)
    assert ((feats[:, 2:4] * radial).sum(1) > 0).all()

    cloud = _box_cloud()
    bf, _ = compute_point_feats(cloud, _box_inside(cloud), 6, True, True)
    assert ((bf[:, 3:6] * cloud).sum(1) >= -1e-4).all()


def test_t5_orientation_reflects_pocket() -> None:
    # On the concave pocket the outward normal points away from material (toward origin).
    ring = _horseshoe_ring()
    pts = resample_ring(list(map(tuple, ring)), P)
    feats, _ = compute_point_feats(
        pts, _shapely_inside(list(map(tuple, ring))), 4, False, True
    )
    rad = np.linalg.norm(pts, axis=1)
    inner = rad < 0.3
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert float((feats[inner, 2:4] * (-radial[inner])).sum(1).mean()) > 0.0


# --------------------------------------------------------------------------- #
# T6 — object-order invariance with the PointSet model
# --------------------------------------------------------------------------- #
def _perm_objects(batch, perm: torch.Tensor):
    b = dataclasses.replace(batch)  # shallow copy of the dataclass
    for name in (
        "obj_tags",
        "obj_boundary",
        "obj_pose",
        "obj_rel",
        "obj_is_goal",
        "obj_mask",
        "point_feats",
        "knn_idx",
    ):
        t = getattr(b, name)
        if t is not None:
            setattr(b, name, t[:, perm])
    return b


def test_t6_object_order_invariance_pointset() -> None:
    cfg = SpectreConfig(use_pca_feats=True, use_edgeconv=True, pma_seeds=4)
    model = SpectreModel(n_ops=5, max_arity=2, cfg=cfg).eval()
    n_obj = 4
    batch = _collate_base(
        [_example(n_obj=n_obj, seed=7), _example(n_obj=n_obj, seed=8)], max_arity=2
    )
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        base = model(batch)[0]
        permuted = model(_perm_objects(batch, perm))[0]
    # Candidate logits are invariant to object-row order (scene tokens are a set).
    assert torch.allclose(base, permuted, atol=1e-4)


# --------------------------------------------------------------------------- #
# T7 — zero-init nesting (RELAXED: LayerNorm kept => reduces to LayerNorm(h))
# --------------------------------------------------------------------------- #
def test_t7_edgeconv_zero_init_is_layernorm() -> None:
    ec = EdgeConv(64).eval()
    # out_proj is zero-initialized at construction; assert the message branch is inert.
    assert torch.count_nonzero(ec.out_proj.weight) == 0
    assert torch.count_nonzero(ec.out_proj.bias) == 0
    h = torch.randn(2, P, 64)
    knn = torch.arange(4).view(1, 1, 4).expand(2, P, 4) % P
    pmask = torch.ones(2, P, dtype=torch.bool)
    with torch.no_grad():
        got = ec(h, knn, pmask)
        want = ec.ln(h)
    assert torch.allclose(got, want, atol=1e-6)


# --------------------------------------------------------------------------- #
# checkpoint round-trip: TrainConfig -> asdict -> load_checkpoint -> strict load
# --------------------------------------------------------------------------- #
def _mini_vocab() -> Vocab:
    return Vocab(
        config_hash="test",
        operators={"op0": 1, "op1": 2},
        predicates={"p0": {"arity": 1, "idx": 1}},
        types={"t": 1},
        max_operator_arity=2,
        max_predicate_arity=1,
        max_skeleton_length=4,
        max_atoms_per_state=4,
        max_objects_per_state=4,
        max_pool_size=4,
    )


def _roundtrip(tmp_path, train_cfg) -> None:
    from alphatamp.approaches.spectre.inference import load_checkpoint
    from alphatamp.approaches.spectre.model import N_OVERLAP_COV
    from alphatamp.approaches.spectre.model import SpectreConfig as _SC

    vocab = _mini_vocab()
    # Build the saved model with the exact config load_checkpoint reconstructs from the
    # persisted cfg dict, so a strict load must match. Mirrors inference.load_checkpoint.
    n_ov = (
        (N_OVERLAP_COV if train_cfg.coverage_feats else 2)
        if train_cfg.use_overlap
        else 0
    )
    sc = _SC(
        n_overlap_feats=n_ov,
        n_prior_feats=0,
        d_rel=train_cfg.d_rel,
        point_dim=3 if train_cfg.scene_3d else 2,
        pose_dim=4 if train_cfg.scene_3d else 3,
        use_pca_feats=train_cfg.use_pca_feats,
        use_edgeconv=train_cfg.use_edgeconv,
        use_point_sab=train_cfg.use_point_sab,
        pma_seeds=train_cfg.pma_seeds,
        edgeconv_k=train_cfg.edgeconv_k,
        max_tags=train_cfg.max_tags,
        use_records=train_cfg.use_records,
        evidence_attn=train_cfg.evidence_attn,
        coverage_feats=train_cfg.coverage_feats,
        use_state_delta=train_cfg.use_state_delta,
        n_predicates=len(vocab.predicates),
        max_pred_arity=vocab.max_predicate_arity,
    )
    model = SpectreModel(n_ops=len(vocab.operators), max_arity=2, cfg=sc)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": dataclasses.asdict(train_cfg),
            "n_ops": len(vocab.operators),
            "selected": "test",
        },
        ckpt,
    )
    loaded, _deploy = load_checkpoint(ckpt, vocab, "cpu")  # strict=True inside
    want_pointset = (
        train_cfg.use_pca_feats
        or train_cfg.use_edgeconv
        or train_cfg.use_point_sab
        or train_cfg.pma_seeds > 1
    )
    assert loaded.scene.use_pointset is want_pointset
    assert loaded.cfg.use_pca_feats == train_cfg.use_pca_feats
    assert loaded.cfg.pma_seeds == train_cfg.pma_seeds


def test_checkpoint_roundtrip_pointset_2d(tmp_path) -> None:
    from alphatamp.approaches.spectre.train import TrainConfig

    _roundtrip(
        tmp_path,
        TrainConfig(use_pca_feats=True, use_edgeconv=True, pma_seeds=4),
    )


def test_checkpoint_roundtrip_pointset_3d(tmp_path) -> None:
    from alphatamp.approaches.spectre.train import TrainConfig

    _roundtrip(
        tmp_path,
        TrainConfig(scene_3d=True, use_pca_feats=True, use_edgeconv=True, pma_seeds=4),
    )


def test_checkpoint_roundtrip_config_off(tmp_path) -> None:
    from alphatamp.approaches.spectre.train import TrainConfig

    _roundtrip(tmp_path, TrainConfig())  # config-off -> FootprintEncoder path


# --------------------------------------------------------------------------- #
# collate shape sanity for the pointset path
# --------------------------------------------------------------------------- #
def test_collate_pointset_shapes_2d() -> None:
    batch = _collate_base([_example(seed=1), _example(seed=2)], max_arity=2)
    assert batch.point_feats is not None and batch.knn_idx is not None
    assert batch.point_feats.shape == (2, 3, P, 6)
    assert batch.knn_idx.shape == (2, 3, P, 4)
    assert batch.knn_idx.dtype == torch.int64


def test_collate_pointset_shapes_3d() -> None:
    ex = [_example(scene_3d=True, seed=1), _example(scene_3d=True, seed=2)]
    batch = _collate_base(ex, max_arity=2)
    assert batch.point_feats is not None and batch.knn_idx is not None
    assert batch.point_feats.shape == (2, 3, P, 8)
    assert batch.knn_idx.shape == (2, 3, P, 6)
