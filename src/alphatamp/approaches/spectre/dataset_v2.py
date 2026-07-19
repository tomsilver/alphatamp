"""v2.2 tensorizer: geometry-carrying ``EpisodeRecord`` → ``SpectreV2Batch`` (Step 8).

Additive to v1's ``dataset.py``. Consumes ``episode.scene_geometry`` (Step 3) and the
skeleton pool, binds every object mention to its episode-local **tag** (Step 7), resamples
each object's boundary ring to a fixed point set, and computes relation-to-target scalars.
Marginal-labeled negatives can be excluded from the success mask (a belt-and-suspenders
label-hygiene flag; with the §8.4 certificate applied most negatives are proven).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.model_v2 import (
    D_GLOBAL_IN,
    D_REL,
    N_BOUNDARY_POINTS,
    SpectreV2Batch,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.tags import assign_tags
from alphatamp.approaches.spectre.vocab import Vocab


def resample_ring(
    points: list[tuple[float, float]], p: int = N_BOUNDARY_POINTS
) -> np.ndarray:
    """Arc-length-uniform resample of a closed boundary ring to ``p`` points (P, 2)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros((p, 2), dtype=np.float32)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        return np.tile(pts[0], (p, 1)).astype(np.float32)
    targets = np.linspace(0.0, total, p, endpoint=False)
    out = np.empty((p, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg) and cum[j + 1] < t:
            j += 1
        span = seg[j] if seg[j] > 1e-12 else 1.0
        frac = (t - cum[j]) / span
        out[i] = closed[j] * (1 - frac) + closed[j + 1] * frac
    return out.astype(np.float32)


@dataclass
class _V2Example:
    """Per-episode numpy/py arrays before collation."""

    obj_tags: np.ndarray
    obj_boundary: np.ndarray
    obj_pose: np.ndarray
    obj_rel: np.ndarray
    obj_is_target: np.ndarray
    op_ids: list  # list[list[int]] per candidate
    arg_tags: list  # list[list[list[int]]]
    success: list
    aux_necessary: np.ndarray
    aux_relevant: np.ndarray


def _target_name(episode: EpisodeRecord) -> Optional[str]:
    if episode.scene_geometry is None:
        return None
    for o in episode.scene_geometry.objects:
        if o.is_target:
            return o.name
    return None


def build_v2_example(
    episode: EpisodeRecord,
    vocab: Vocab,
    rng: Optional[np.random.Generator] = None,
    max_tags: int = 32,
    exclude_marginal: bool = False,
) -> _V2Example:
    """Tensorize one geometry-carrying episode. ``rng`` set ⇒ tag permutation (training)."""
    if episode.scene_geometry is None:
        raise ValueError("build_v2_example requires scene_geometry (Step 3)")
    canon = canonicalize_episode(episode, rng=None)  # structure/names; tags added below
    geo = canon.scene_geometry
    assert geo is not None
    obj_names = [o.name for o in geo.objects]
    tags = assign_tags(obj_names, rng=rng, max_tags=max_tags)

    # scale for pose normalization: drawer frame extent (fallback to buffer bounds).
    frame = geo.frame or {}
    scale = max(
        float(frame.get("drawer_w", 0.0)), float(frame.get("drawer_d", 0.0)), 1.0
    )
    tgt = next((o for o in geo.objects if o.is_target), None)
    tx, ty = (tgt.pose[0], tgt.pose[1]) if tgt else (0.0, 0.0)

    n = len(geo.objects)
    obj_tags = np.array([tags[o.name] for o in geo.objects], dtype=np.int64)
    obj_boundary = np.stack([resample_ring(list(o.boundary)) for o in geo.objects])
    obj_pose = np.array(
        [[o.pose[0] / scale, o.pose[1] / scale, o.pose[2]] for o in geo.objects],
        dtype=np.float32,
    )
    obj_is_target = np.array(
        [1.0 if o.is_target else 0.0 for o in geo.objects], np.float32
    )
    rel = np.zeros((n, D_REL), dtype=np.float32)
    for i, o in enumerate(geo.objects):
        dx, dy = (o.pose[0] - tx) / scale, (o.pose[1] - ty) / scale
        dist = math.hypot(dx, dy)
        rel[i, :6] = [dx, dy, dist, o.area, math.sin(o.pose[2]), math.cos(o.pose[2])]
        rel[i, 6] = 1.0 if o.concave else 0.0
        rel[i, 7] = float(o.area) / (tgt.area if tgt else 1.0)

    # candidates: operator-schema ids + arg tags (by canonical object name).
    op_ids, arg_tags, success = [], [], []
    for skel, out in zip(canon.skeleton_pool, canon.outcomes):
        ops, ats = [], []
        for op in skel.operator_seq:
            ops.append(int(vocab.operators.get(op.name, 0)))  # name -> idx (0 = OOV)
            ats.append([tags.get(a.name, 0) for a in op.parameters])
        op_ids.append(ops)
        arg_tags.append(ats)
        s = out.outcome == "success"
        if exclude_marginal and out.outcome == "fail":
            meta = out.refiner_metadata or {}
            if meta.get("status") == "marginal":
                s = None  # excluded from the loss
        success.append(s)

    aux = canon.aux_labels
    necessary = np.array(
        [
            1.0 if aux and o.name in aux.necessary else (-1.0 if aux is None else 0.0)
            for o in geo.objects
        ],
        dtype=np.float32,
    )
    relevant = np.array(
        [
            1.0 if aux and o.name in aux.relevant else (-1.0 if aux is None else 0.0)
            for o in geo.objects
        ],
        dtype=np.float32,
    )
    return _V2Example(
        obj_tags,
        obj_boundary,
        obj_pose,
        rel,
        obj_is_target,
        op_ids,
        arg_tags,
        success,
        necessary,
        relevant,
    )


def _glob_feats(ex: _V2Example) -> np.ndarray:
    n_obj = len(ex.obj_tags)
    k = len(ex.op_ids)
    mean_len = float(np.mean([len(o) for o in ex.op_ids])) if ex.op_ids else 0.0
    return np.array(
        [float(n_obj), float(k), mean_len, 0.0, 0.0, 0.0], dtype=np.float32
    )[:D_GLOBAL_IN]


def collate_v2(examples: list[_V2Example], max_arity: int) -> SpectreV2Batch:
    """Pad + stack per-episode examples into a ``SpectreV2Batch``."""
    b = len(examples)
    n = max(len(e.obj_tags) for e in examples)
    k = max(len(e.op_ids) for e in examples)
    ell = max((len(o) for e in examples for o in e.op_ids), default=1)
    p = N_BOUNDARY_POINTS

    obj_tags = np.zeros((b, n), np.int64)
    obj_boundary = np.zeros((b, n, p, 2), np.float32)
    obj_pose = np.zeros((b, n, 3), np.float32)
    obj_rel = np.zeros((b, n, D_REL), np.float32)
    obj_is_target = np.zeros((b, n), np.float32)
    obj_mask = np.zeros((b, n), bool)
    cand_op_ids = np.zeros((b, k, ell), np.int64)
    cand_arg_tags = np.zeros((b, k, ell, max_arity), np.int64)
    cand_pos = np.zeros((b, k, ell), np.int64)
    cand_step_mask = np.zeros((b, k, ell), bool)
    pool_mask = np.zeros((b, k), bool)
    success = np.zeros((b, k), bool)
    aux_nec = np.full((b, n), -1.0, np.float32)
    aux_rel = np.full((b, n), -1.0, np.float32)
    glob = np.zeros((b, D_GLOBAL_IN), np.float32)

    for bi, e in enumerate(examples):
        no = len(e.obj_tags)
        obj_tags[bi, :no] = e.obj_tags
        obj_boundary[bi, :no] = e.obj_boundary
        obj_pose[bi, :no] = e.obj_pose
        obj_rel[bi, :no] = e.obj_rel
        obj_is_target[bi, :no] = e.obj_is_target
        obj_mask[bi, :no] = True
        aux_nec[bi, :no] = e.aux_necessary
        aux_rel[bi, :no] = e.aux_relevant
        glob[bi] = _glob_feats(e)
        for ki, (ops, ats, s) in enumerate(zip(e.op_ids, e.arg_tags, e.success)):
            pool_mask[bi, ki] = True
            success[bi, ki] = bool(s) if s is not None else False
            for li, (oid, at) in enumerate(zip(ops, ats)):
                cand_op_ids[bi, ki, li] = oid
                cand_pos[bi, ki, li] = li
                cand_step_mask[bi, ki, li] = True
                cand_arg_tags[bi, ki, li, : len(at)] = at[:max_arity]

    t = torch.as_tensor
    return SpectreV2Batch(
        obj_tags=t(obj_tags),
        obj_boundary=t(obj_boundary),
        obj_pose=t(obj_pose),
        obj_rel=t(obj_rel),
        obj_is_target=t(obj_is_target),
        obj_mask=t(obj_mask),
        cand_op_ids=t(cand_op_ids),
        cand_arg_tags=t(cand_arg_tags),
        cand_pos=t(cand_pos),
        cand_step_mask=t(cand_step_mask),
        pool_mask=t(pool_mask),
        glob_feats=t(glob),
        success_mask=t(success),
        aux_necessary=t(aux_nec),
        aux_relevant=t(aux_rel),
    )
