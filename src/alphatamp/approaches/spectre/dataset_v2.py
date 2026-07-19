"""v2.2 tensorizer: geometry-carrying ``EpisodeRecord`` → ``SpectreV2Batch`` (Step 8).

Additive to v1's ``dataset.py``. Consumes ``episode.scene_geometry`` (Step 3) and the
skeleton pool, binds every object mention to its episode-local **tag** (Step 7),
resamples each object's boundary ring to a fixed point set, and computes relation-to-
target scalars. Marginal-labeled negatives can be excluded from the success mask (a
belt-and-suspenders label-hygiene flag; with the §8.4 certificate applied most negatives
are proven).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.facts import gather_context_facts
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v2 import (
    D_GLOBAL_IN,
    D_REL,
    MAX_FACT_ARGS,
    N_BOUNDARY_POINTS,
    SpectreV2Batch,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.tags import assign_tags
from alphatamp.approaches.spectre.vocab import Vocab


def sample_context(
    fail_idx: list[int],
    rng: np.random.Generator,
    p_empty: float = 0.35,
    p_drop_facts: float = 0.3,
    max_f: int = 8,
) -> tuple[frozenset[int], bool]:
    """Sample a failed-context ``F`` for evidence training + an evidence-dropout flag.

    Heavy mass at ``|F|=0`` (the ``t=0`` deployment state the static pathway must own,
    P-D), otherwise a small uniform size. ``hide_facts`` (evidence dropout) trains the
    ranker to stand on geometry alone even when failures exist. Returns ``(F,
    hide_facts)``.
    """
    if not fail_idx or rng.random() < p_empty:
        return frozenset(), False
    size = int(rng.integers(1, min(max_f, len(fail_idx)) + 1))
    chosen = rng.choice(np.asarray(fail_idx), size=size, replace=False)
    return frozenset(int(i) for i in chosen), bool(rng.random() < p_drop_facts)


def _fact_arrays(
    episode: EpisodeRecord, context_f: frozenset[int], tags: dict[str, int]
) -> tuple[list[int], list[int], list[list[int]]]:
    """Tensorize the **hint**-tier facts of the failed skeletons in ``context_f`` to
    tag-bound ids.

    Proof-tier facts (blocked-at-contents / pack-impossible) are routed to the sound
    ``overlap`` demotion feature instead of the learned tokens, so the ranker consumes
    them as the precise proof-demotion rule rather than the crude "prefer longer" a
    token invites.
    """
    from alphatamp.approaches.spectre.facts import TIER_IDS

    hint = TIER_IDS["hint"]
    type_ids: list[int] = []
    tier_ids: list[int] = []
    arg_tags: list[list[int]] = []
    for fr in gather_context_facts(episode, sorted(context_f)):
        if fr.tier_id != hint:
            continue  # proofs handled structurally (overlap), not as tokens
        ats = [tags[a] for a in fr.args if a in tags][:MAX_FACT_ARGS]
        type_ids.append(fr.type_id)
        tier_ids.append(fr.tier_id)
        arg_tags.append(ats)
    return type_ids, tier_ids, arg_tags


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
    # Step-11 typed evidence (empty in the static path).
    avail: list  # bool per candidate: not in the failed context F
    fact_type_ids: list  # int per fact
    fact_tier_ids: list  # int per fact
    fact_arg_tags: list  # list[list[int]] per fact (object tags, capped)
    prior: list  # [−index/K, −len/max_len] per candidate (a-priori default-order prior)
    overlap: (
        list  # [subset⊆blocked (sound demotion), jaccard-with-failed] per candidate
    )


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
    evidence: bool = False,
    context_f: Optional[frozenset[int]] = None,
    hide_facts: bool = False,
    augment_tags: bool = True,
) -> _V2Example:
    """Tensorize one geometry-carrying episode.

    ``rng`` set ⇒ tag permutation (training).     Evidence pathway (Step 11): with
    ``evidence=True`` and ``context_f`` unset, a failed     context ``F`` is sampled
    (``rng`` required); with ``context_f`` given (eval rollout) it     is used as-is.
    Candidates in ``F`` are marked unavailable, and the typed facts of ``F``     become
    fact tokens unless ``hide_facts`` (evidence dropout) suppresses them.
    """
    if episode.scene_geometry is None:
        raise ValueError("build_v2_example requires scene_geometry (Step 3)")
    canon = canonicalize_episode(episode, rng=None)  # structure/names; tags added below
    geo = canon.scene_geometry
    assert geo is not None
    obj_names = [o.name for o in geo.objects]
    tags = assign_tags(
        obj_names, rng=(rng if augment_tags else None), max_tags=max_tags
    )

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
    op_ids, arg_tags = [], []
    success: list[Optional[bool]] = []
    removals: list[int] = []  # staged (place-buffer) count per candidate = plan length
    cand_subsets: list[frozenset] = []  # staged item names per candidate
    for skel, out in zip(canon.skeleton_pool, canon.outcomes):
        ops, ats = [], []
        for op in skel.operator_seq:
            ops.append(int(vocab.operators.get(op.name, 0)))  # name -> idx (0 = OOV)
            ats.append([tags.get(a.name, 0) for a in op.parameters])
        op_ids.append(ops)
        arg_tags.append(ats)
        removals.append(sum(1 for op in skel.operator_seq if op.name == "place-buffer"))
        cand_subsets.append(
            frozenset(
                op.parameters[0].name
                for op in skel.operator_seq
                if op.name == "place-buffer"
            )
        )
        s: Optional[bool] = out.outcome == "success"
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

    # typed-evidence context F: candidates in F are "already tried" (unavailable); their
    # facts become tokens unless evidence dropout hides them.
    fail_idx = [i for i, out in enumerate(canon.outcomes) if out.outcome == "fail"]
    hide = hide_facts
    if context_f is not None:
        ctx: frozenset[int] = frozenset(int(i) for i in context_f)
    elif evidence and rng is not None:
        ctx, hide = sample_context(fail_idx, rng)
    else:
        ctx = frozenset()
    avail = [i not in ctx for i in range(len(op_ids))]
    ftype: list[int]
    ftier: list[int]
    farg: list[list[int]]
    if hide or not ctx:
        ftype, ftier, farg = [], [], []
    else:
        ftype, ftier, farg = _fact_arrays(canon, ctx, tags)

    # a-priori default-order / short-first prior per candidate (higher = tried sooner).
    k = len(op_ids)
    max_rm = max(removals) if removals else 1
    prior = [[-(i / max(k - 1, 1)), -(removals[i] / max(max_rm, 1))] for i in range(k)]

    # structural evidence features: relate each candidate's action-set to the OBSERVED failed
    # sets in F (zeroed under evidence dropout / empty context). blocked = failed subsets with
    # a blocked-at-contents fact; a candidate ⊆ a blocked set is provably also-blocked (sound
    # proof-demotion). Jaccard-with-failed is a mild similarity hint.
    overlap: list[list[float]] = [[0.0, 0.0] for _ in range(k)]
    if ctx and not hide:
        blocked = [
            cand_subsets[f]
            for f in ctx
            if (canon.outcomes[f].post_mortem is not None)
            and any(
                fc.fact_type == "blocked-at-contents"
                for fc in canon.outcomes[f].post_mortem.facts  # type: ignore[union-attr]
            )
        ]
        failed = [cand_subsets[f] for f in ctx]
        for i, ci in enumerate(cand_subsets):
            dead = 1.0 if any(ci <= bf for bf in blocked) else 0.0
            jac = max(
                (len(ci & ff) / max(len(ci | ff), 1) for ff in failed), default=0.0
            )
            overlap[i] = [dead, float(jac)]

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
        avail,
        ftype,
        ftier,
        farg,
        prior,
        overlap,
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

    fmax = max((len(e.fact_type_ids) for e in examples), default=0)

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
    avail = np.zeros((b, k), bool)
    cand_prior = np.zeros((b, k, 2), np.float32)
    cand_overlap = np.zeros((b, k, 2), np.float32)
    success = np.zeros((b, k), bool)
    aux_nec = np.full((b, n), -1.0, np.float32)
    aux_rel = np.full((b, n), -1.0, np.float32)
    glob = np.zeros((b, D_GLOBAL_IN), np.float32)
    fa = max(MAX_FACT_ARGS, 1)
    fact_type = np.zeros((b, fmax), np.int64)
    fact_tier = np.zeros((b, fmax), np.int64)
    fact_arg = np.zeros((b, fmax, fa), np.int64)
    fact_mask = np.zeros((b, fmax), bool)

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
            avail[bi, ki] = bool(e.avail[ki]) if ki < len(e.avail) else True
            cand_prior[bi, ki] = e.prior[ki]
            cand_overlap[bi, ki] = e.overlap[ki]
            success[bi, ki] = bool(s) if s is not None else False
            for li, (oid, at) in enumerate(zip(ops, ats)):
                cand_op_ids[bi, ki, li] = oid
                cand_pos[bi, ki, li] = li
                cand_step_mask[bi, ki, li] = True
                cand_arg_tags[bi, ki, li, : len(at)] = at[:max_arity]
        for fi, (ty, ti, ar) in enumerate(
            zip(e.fact_type_ids, e.fact_tier_ids, e.fact_arg_tags)
        ):
            fact_type[bi, fi] = ty
            fact_tier[bi, fi] = ti
            fact_mask[bi, fi] = True
            fact_arg[bi, fi, : len(ar)] = ar[:fa]

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
        avail_mask=t(avail),
        cand_prior=t(cand_prior),
        cand_overlap=t(cand_overlap),
        fact_type_ids=t(fact_type) if fmax else None,
        fact_tier_ids=t(fact_tier) if fmax else None,
        fact_arg_tags=t(fact_arg) if fmax else None,
        fact_mask=t(fact_mask) if fmax else None,
    )


class SpectreV2Dataset(Dataset):
    """Torch dataset over geometry-carrying episodes for the v2 model.

    Filters to trainable episodes (>= 1 success, >= 2 skeletons). ``augment`` permutes
    the object tags per epoch (seeded from ``(seed, episode_idx, epoch)``); eval uses
    ``rng=None`` (deterministic). One episode == one training example (its whole
    candidate pool).
    """

    def __init__(
        self,
        split_dir: Path,
        vocab: Vocab,
        max_tags: int = 32,
        augment: bool = True,
        seed: int = 0,
        exclude_marginal: bool = False,
        evidence: bool = False,
    ) -> None:
        self.vocab = vocab
        self.max_tags = max_tags
        self.augment = augment
        self.seed = seed
        self.exclude_marginal = exclude_marginal
        self.evidence = evidence
        self.epoch = 0
        self._paths = []
        for p in list_episodes(split_dir):
            ep = load_episode(p)
            if (
                ep.scene_geometry is not None
                and ep.summary.num_success >= 1
                and len(ep.skeleton_pool) >= 2
            ):
                self._paths.append(p)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> "_V2Example":
        ep = load_episode(self._paths[idx])
        rng = None
        if self.augment or self.evidence:
            rng = np.random.default_rng((self.seed, idx, self.epoch))
        return build_v2_example(
            ep,
            self.vocab,
            rng=rng,
            max_tags=self.max_tags,
            exclude_marginal=self.exclude_marginal,
            evidence=self.evidence,
            augment_tags=self.augment,
        )


def make_collate(max_arity: int):
    """A picklable-ish collate closure for a DataLoader (returns a
    ``SpectreV2Batch``)."""

    def _collate(examples: list) -> SpectreV2Batch:
        return collate_v2(examples, max_arity=max_arity)

    return _collate
