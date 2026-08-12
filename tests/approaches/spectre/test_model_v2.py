"""Tests for the v2.2 geometry-aware model + tensorizer (Step 8).

Key structural properties: footprint point-set permutation invariance, object-order
permutation invariance of the logits (the scene is a set, candidates reference objects by
tag), and the **anti-collapse** guarantee — two candidates with the same operator sequence
but different argument objects (tags) produce different logits (v1 could not tell them
apart, which forced the length-only ranking).
"""

from __future__ import annotations

import numpy as np
import torch

from alphatamp.approaches.spectre import model_v2 as M


def _batch(B=2, N=6, K=5, L=7, A=1, max_tags=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return M.SpectreV2Batch(
        obj_tags=torch.randint(1, max_tags + 1, (B, N), generator=g),
        obj_boundary=torch.randn(B, N, M.N_BOUNDARY_POINTS, 2, generator=g),
        obj_pose=torch.randn(B, N, 3, generator=g),
        obj_rel=torch.randn(B, N, M.D_REL, generator=g),
        obj_is_goal=torch.zeros(B, N),
        obj_mask=torch.ones(B, N, dtype=torch.bool),
        cand_op_ids=torch.randint(1, 4, (B, K, L), generator=g),
        cand_arg_tags=torch.randint(1, max_tags + 1, (B, K, L, A), generator=g),
        cand_pos=torch.arange(L).view(1, 1, L).expand(B, K, L).contiguous(),
        cand_step_mask=torch.ones(B, K, L, dtype=torch.bool),
        pool_mask=torch.ones(B, K, dtype=torch.bool),
        glob_feats=torch.randn(B, M.D_GLOBAL_IN, generator=g),
        success_mask=torch.zeros(B, K, dtype=torch.bool),
        aux_necessary=torch.zeros(B, N),
        aux_relevant=torch.zeros(B, N),
    )


def test_forward_shapes_and_grad():
    batch = _batch()
    model = M.SpectreV2Model(n_ops=3, max_arity=1, max_tags=16)
    logits, aux = model(batch)
    assert logits.shape == (2, 5) and aux.shape == (2, 6, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_footprint_point_permutation_invariance():
    enc = M.FootprintEncoder().eval()
    ring = torch.randn(1, 1, M.N_BOUNDARY_POINTS, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)
    perm = torch.randperm(M.N_BOUNDARY_POINTS)
    with torch.no_grad():
        d0 = enc(ring, mask)
        d1 = enc(ring[:, :, perm, :], mask)
    assert torch.allclose(d0, d1, atol=1e-5)


def test_object_order_permutation_invariance_of_logits():
    # permuting the object set (tags follow each object) leaves the per-candidate logits
    # unchanged: the scene is unordered memory and candidates bind objects by tag.
    batch = _batch(seed=1)
    model = M.SpectreV2Model(n_ops=3, max_arity=1, max_tags=16).eval()
    with torch.no_grad():
        l0, _ = model(batch)
        perm = torch.randperm(batch.obj_tags.shape[1])
        b2 = M.SpectreV2Batch(
            **{
                **batch.__dict__,
                "obj_tags": batch.obj_tags[:, perm],
                "obj_boundary": batch.obj_boundary[:, perm],
                "obj_pose": batch.obj_pose[:, perm],
                "obj_rel": batch.obj_rel[:, perm],
                "obj_is_goal": batch.obj_is_goal[:, perm],
                "obj_mask": batch.obj_mask[:, perm],
                "aux_necessary": batch.aux_necessary[:, perm],
                "aux_relevant": batch.aux_relevant[:, perm],
            }
        )
        l1, _ = model(b2)
    assert torch.allclose(l0, l1, atol=1e-4)


def test_anti_collapse_same_ops_different_args_differ():
    # two candidates, identical operator ids but different argument tags → different logits.
    batch = _batch(B=1, N=6, K=2, L=3, A=1, seed=2)
    batch.cand_op_ids[0, 1] = batch.cand_op_ids[0, 0]  # same op sequence
    batch.cand_arg_tags[0, 0, :, 0] = torch.tensor([1, 2, 3])
    batch.cand_arg_tags[0, 1, :, 0] = torch.tensor([4, 5, 6])  # different objects
    model = M.SpectreV2Model(n_ops=3, max_arity=1, max_tags=16).eval()
    with torch.no_grad():
        logits, _ = model(batch)
    assert not torch.isclose(logits[0, 0], logits[0, 1], atol=1e-4)


def test_tensorizer_on_synthetic_geometry_episode():
    import dataclasses

    from _fixtures import build_toy_episode

    from alphatamp.approaches.spectre import dataset_v2 as D2
    from alphatamp.approaches.spectre.schema import ObjectGeometry, SceneGeometry
    from alphatamp.approaches.spectre.vocab import extract_vocab

    ep = build_toy_episode()
    ring = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
    objs = tuple(
        ObjectGeometry(
            name=nm,
            pose=(float(i), 0.0, 0.0),
            boundary=ring,
            family="test",
            area=1.0,
            concave=False,
            is_target=(nm == "block_0"),
        )
        for i, nm in enumerate(sorted(ep.object_registry))
    )
    ep = dataclasses.replace(
        ep,
        scene_geometry=SceneGeometry(
            objects=objs, containers=(), frame={"drawer_w": 10.0}
        ),
    )

    class _V:  # minimal vocab stub for op ids
        operators = {"Pick": 1, "Place": 2}
        max_operator_arity = 2

    ex = D2.build_v2_example(ep, _V(), rng=np.random.default_rng(0), max_tags=16)
    batch = D2.collate_v2([ex], max_arity=2)
    assert batch.obj_boundary.shape[-2:] == (M.N_BOUNDARY_POINTS, 2)
    model = M.SpectreV2Model(n_ops=2, max_arity=2, max_tags=16)
    logits, _ = model(batch)
    assert logits.shape[0] == 1 and torch.isfinite(logits[batch.pool_mask]).all()
