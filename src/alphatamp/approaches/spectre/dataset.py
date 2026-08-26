"""V3 tensorizer: ``EpisodeRecord`` -> ``SpectreBatch``, via the domain contract.

Structurally this is v2.2's tensorizer with the DD2D literals removed. Two things it no
longer knows:

- **which objects a candidate manipulates** -- was ``op.name == "place-buffer"``, now
  :func:`domain.DomainSpec.manipulated`, verified identical on 120000/120000 skeletons;
- **which failures license demotion** -- was ``failure_action.startswith("retrieve")``,
  now the per-query axiom declaration plus the observation's own budget flag.

Three v2.2 features are deliberately *not* carried over:

- ``exclude_marginal`` was inert twice over (the DD2D refiner only writes ``status`` in
  ``{feasible, infeasible}``, and even when it fired ``collate`` folded the resulting
  ``None`` back to ``False`` because the batch has no per-candidate ignore mask). v3
  declines to carry a flag that silently does nothing; reinstating the behaviour needs a
  real label mask, not a flag.
- ``demotion_source="computed"`` -- the geometry-reconstruction path -- is dropped (R2).
  It was worth a measured ~14%, but it is the last per-environment geometry routine in
  the deployment story, and the observed signal is what generalizes.
- The **short-first prior** as a scorer feature (R1). The plan-length column survives
  only as the within-length loss's bucket key, which is now :func:`domain.length_key`;
  the model sees no prior (``SpectreConfig.n_prior_feats == 0``). It was a per-dataset
  hand switch that diverged training on the easier collection, and note it was never a
  clean feature ablation anyway: enabling it also zero-inits the scorer head.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Sequence

import numpy as np
import torch

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.encoders import (
    D_GLOBAL_IN,
    D_REL,
    MAX_FACT_ARGS,
    N_BOUNDARY_POINTS,
)
from alphatamp.approaches.spectre.facts import TIER_IDS, gather_context_facts
from alphatamp.approaches.spectre.failure_record import records_for_candidate
from alphatamp.approaches.spectre.model import (
    MAX_ATTEMPTS,
    MAX_DELTA_ATOMS,
    MAX_RECORD_ARGS,
    MAX_RECORD_CULPRITS,
    N_OVERLAP_COV,
    N_RECORD_SCALARS,
    N_STEP_SCALARS,
    SpectreBatch,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.tags import assign_tags
from alphatamp.approaches.spectre.unified_evidence import blame as _unified_blame
from alphatamp.approaches.spectre.unified_evidence import (
    coverage_and_waste,
    records_from_failure_records,
    scene_filters,
)
from alphatamp.approaches.spectre.vocab import Vocab

__all__ = [
    "build_example",
    "collate",
    "sample_context",
    "build_record_arrays",
    "compute_point_feats",
    "pointset_emission",
    "atom_emission",
]


def pointset_emission(cfg, scene_3d: bool) -> tuple[bool, bool, int]:
    """``(pointset_feats, use_pca_feats, edgeconv_k)`` for :func:`build_example`.

    Reads the five PointSetEncoder switches off any cfg-like object (a ``TrainConfig``
    or a ``SpectreConfig`` -- the field names match), so training, validation-selection
    and deployment all derive the tensorizer's emission from one place and can never
    desync. ``edgeconv_k=0`` resolves to 4 (2D) / 6 (3D).
    """
    pointset = bool(
        cfg.use_pca_feats or cfg.use_edgeconv or cfg.use_point_sab or cfg.pma_seeds > 1
    )
    k = cfg.edgeconv_k or (6 if scene_3d else 4)
    return pointset, bool(cfg.use_pca_feats), int(k)


def atom_emission(cfg) -> tuple[bool, bool]:
    """``(emit_init_atoms, emit_goal_atoms)`` for :func:`build_example`.

    Reads the atom-input switches off any cfg-like object (a ``TrainConfig`` or a
    ``SpectreConfig`` -- the field names match), so training, validation-selection and
    deployment all derive the tensorizer's emission from one place and can never desync
    (the same discipline as :func:`pointset_emission`). ``atom_mode == "off"`` (the
    default) emits neither, so the config-off tensorizer is byte-unchanged.
    """
    on = getattr(cfg, "atom_mode", "off") == "profiles"
    return (
        on and bool(getattr(cfg, "use_init_atoms", True)),
        on and bool(getattr(cfg, "use_goal_atoms", True)),
    )


#: One atom of a state delta, as ``(predicate id, argument tags)``.
DeltaAtomArray = tuple[int, list[int]]
#: One record's ``s_j - s_0``, as ``(added atoms, deleted atoms)``.
DeltaArrays = tuple[list[DeltaAtomArray], list[DeltaAtomArray]]
# : One record token: ``(schema id, arg tags, culprit tags, scalars)``, optionally
# followed : by the state delta. The trailing element is present iff the delta was
# requested -- the : model reads it by position, and a 4-tuple stream is byte-for-byte
# the pre-delta one.
RecordArray = tuple  # (int, list[int], list[int], list[float][, DeltaArrays])


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


def sample_point_cloud(
    points: Sequence[tuple[float, float, float]], p: int = N_BOUNDARY_POINTS
) -> np.ndarray:
    """Fixed-size ``(p, 3)`` point cloud from a stored 3D point set (pad/truncate to p).

    The 3D analogue of :func:`resample_ring`: the point-set encoder pools symmetrically,
    so order does not matter and no arc-length parameterisation applies -- we only
    guarantee a fixed count. Restock3D stores exactly ``p`` points, so this is a no-op
    there.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) point cloud, got shape {pts.shape}")
    if len(pts) == p:
        return pts
    if len(pts) > p:
        return pts[:p]
    return pts[np.arange(p) % len(pts)]


def _rot90_ccw(n: np.ndarray) -> np.ndarray:
    """Rotate each 2D vector 90° CCW: ``(x, y) -> (-y, x)`` (shape ``(P, 2)``)."""
    return np.stack([-n[:, 1], n[:, 0]], axis=1)


def _shapely_inside(boundary: Sequence[tuple[float, float]]):
    """Inside-test closure over the *source* polygon (2D, tensorizer-time only).

    Rebuilds the ``shapely`` polygon from the stored exterior ring -- the polygon never
    enters the model input, so this is a tensorizer-only oracle (doc §2.3). shapely is
    imported lazily so the config-off 2D path takes on no new import.
    """
    from shapely.geometry import Point, Polygon  # lazy: off path never imports

    poly = Polygon(boundary)
    if not poly.is_valid:
        poly = poly.buffer(0)  # heal any self-touching concave ring

    def _inside(q: np.ndarray) -> bool:
        return bool(poly.contains(Point(float(q[0]), float(q[1]))))

    return _inside


def _box_inside(pts: np.ndarray):
    """Inside-test closure for an axis-aligned box centered at the item-frame origin
    (3D).

    Restock3D objects are convex boxes centered at the origin, so the half-extents are
    ``|p|_max`` over the cloud and an exact analytic inside test needs nothing beyond
    the stored point set (the doc's sensor-viewpoint oracle is unavailable -- no camera
    is recorded). See docs/decisions 2026-08-18.
    """
    half = np.abs(np.asarray(pts, dtype=np.float64)).max(axis=0)

    def _inside(q: np.ndarray) -> bool:
        return bool(np.all(np.abs(np.asarray(q, dtype=np.float64)) <= half))

    return _inside


def compute_point_feats(
    pts: np.ndarray,
    inside_fn,
    k: int,
    scene_3d: bool,
    use_pca_feats: bool,
    *,
    validate: bool = False,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point differential features + Euclidean kNN indices for one object's points.

    ``pts`` is ``(P, d)`` with ``d in {2, 3}``, item frame, centroid ~ origin. Returns
    ``(point_feats (P, C_pt), knn_idx (P, k) int64)``. Everything is order-free
    (Euclidean kNN, PCA, the inside test, the quadric fit), so the deployed path never
    uses ring order -- see doc §2. With ``use_pca_feats=False`` the features are just the
    coordinates (``C_pt = d``) and ``knn_idx`` is still computed so an EdgeConv graph
    exists.

    - 2D (``C_pt=6``): ``[x, y, n_x, n_y, κ̂, f]`` -- oriented normal, signed curvature,
      PCA flatness ``λ2/(λ1+λ2)``.
    - 3D (``C_pt=8``): ``[x, y, z, n_x, n_y, n_z, f, 0]`` -- oriented normal, Pauly
      surface variation ``λ3/Σλ``; no signed curvature (a later column, doc §2.4).
    """
    pts = np.asarray(pts, dtype=np.float64)
    p_count, d = pts.shape
    k = int(min(max(k, 1), p_count))

    # --- kNN (§2.1): self included (distance 0 sorts first), so the k-set doubles as
    # the PCA neighborhood and the EdgeConv graph. Stable sort => deterministic on input.
    diff = pts[:, None, :] - pts[None, :, :]  # (P, P, d)
    d2 = (diff * diff).sum(-1)  # (P, P)
    knn_idx = np.argsort(d2, axis=1, kind="stable")[:, :k]  # (P, k)
    nn_d2 = np.take_along_axis(d2, knn_idx, axis=1)  # (P, k)
    if k > 1:
        hbar = np.sqrt(np.maximum(nn_d2[:, 1:], 0.0)).mean(axis=1)  # mean non-self dist
    else:
        hbar = np.zeros(p_count)
    hbar = np.maximum(hbar, eps)

    if not use_pca_feats:
        return pts.astype(np.float32), knn_idx.astype(np.int64)

    # --- local PCA frame (§2.2): batched eigendecomposition of the neighbor covariance.
    nbr = pts[knn_idx]  # (P, k, d)
    cen = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("pki,pkj->pij", cen, cen) / max(k, 1)  # (P, d, d)
    evals, evecs = np.linalg.eigh(
        cov
    )  # ascending eigenvalues; evecs[:, :, i] <-> evals[i]
    normal = evecs[:, :, 0].copy()  # smallest-variance direction
    if scene_3d:
        f = evals[:, 0] / (
            evals.sum(axis=1) + eps
        )  # Pauly surface variation ∈ [0, 1/3]
    else:
        f = evals[:, 0] / (evals[:, 0] + evals[:, 1] + eps)  # flatness ∈ [0, 0.5]

    # duplicate-point / degenerate fallback (§2.2): tiny largest eigenvalue.
    valid = evals[:, -1] > eps
    if validate:
        assert bool(valid.all()), "degenerate PCA neighborhood on clean sim data"
    if not valid.all():
        bad = ~valid
        f = np.where(bad, 0.0, f)
        default_n = np.zeros(d)
        default_n[0] = 1.0
        normal[bad] = default_n

    # --- orientation (§2.3): flip the normal outward via the inside test on p + ε·n.
    if inside_fn is not None:
        q = pts + (0.5 * hbar)[:, None] * normal
        flip = np.array([inside_fn(q[i]) for i in range(p_count)], dtype=bool)
        normal[flip] *= -1.0

    if not scene_3d:
        # Consistent tangent from the oriented normal (§2.3), not ring winding.
        tangent = _rot90_ccw(normal)  # (P, 2)
        # --- signed curvature (§2.4): quadric v ≈ b·u + a·u², κ = -2a, κ̂ = tanh(κ·h̄).
        rel = nbr - pts[:, None, :]  # (P, k, 2)
        u = np.einsum("pki,pi->pk", rel, tangent)
        v = np.einsum("pki,pi->pk", rel, normal)
        design = np.stack([u, u * u], axis=-1)  # (P, k, 2)
        ata = np.einsum("pki,pkj->pij", design, design) + eps * np.eye(2)[None]
        atv = np.einsum("pki,pk->pi", design, v)
        coef = np.linalg.solve(ata, atv)  # (P, 2) = [b, a]
        kappa = -2.0 * coef[:, 1]
        khat = np.where(valid, np.tanh(kappa * hbar), 0.0)
        feats = np.concatenate(
            [pts, normal, khat[:, None], f[:, None]], axis=1
        )  # (P, 6)
    else:
        feats = np.concatenate(
            [pts, normal, f[:, None], np.zeros((p_count, 1))], axis=1
        )  # (P, 8)

    return feats.astype(np.float32), knn_idx.astype(np.int64)


@dataclasses.dataclass
class SpectreExample:
    """Per-episode numpy/py arrays before collation."""

    obj_tags: np.ndarray
    obj_boundary: np.ndarray
    obj_pose: np.ndarray
    obj_rel: np.ndarray
    obj_is_goal: np.ndarray
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
    # PointSetEncoder inputs (doc pointset_encoder_upgrade.md); None on the config-off
    # path so the byte-unchanged v1 tensorizer emits exactly what it did before.
    point_feats: Optional[np.ndarray] = None  # (N, P, C_pt) per-point features
    knn_idx: Optional[np.ndarray] = None  # (N, P, k) int64 Euclidean-kNN indices
    # init/goal atom profiles (doc spectre_atom_input_guide.md); None on the config-off
    # path (atom_mode="off"). ``pred`` is the vocab id +1 (0 = pad); ``arg_tags`` are
    # object tags in the scene tag namespace (0 = PAD_TAG). Init and goal kept separate.
    init_atom_pred: Optional[np.ndarray] = None  # (A_i,) int64
    init_atom_arg_tags: Optional[np.ndarray] = None  # (A_i, M) int64
    goal_atom_pred: Optional[np.ndarray] = None  # (A_g,) int64
    goal_atom_arg_tags: Optional[np.ndarray] = None  # (A_g, M) int64
    # Rung-1 evidence-step stream (docs/failed_records_fix.md F-A); None unless
    # record_mode="steps". A list of StepArray tuples (see build_evidence_steps).
    rec_steps: Optional[list] = None


def _glob_feats(ex: SpectreExample) -> np.ndarray:
    n_obj = len(ex.obj_tags)
    k = len(ex.op_ids)
    mean_len = float(np.mean([len(o) for o in ex.op_ids])) if ex.op_ids else 0.0
    return np.array(
        [float(n_obj), float(k), mean_len, 0.0, 0.0, 0.0], dtype=np.float32
    )[:D_GLOBAL_IN]


def _collate_base(examples: list[SpectreExample], max_arity: int) -> SpectreBatch:
    """Pad + stack per-episode examples into a ``SpectreBatch`` (record fields
    unset)."""
    b = len(examples)
    n = max(len(e.obj_tags) for e in examples)
    k = max(len(e.op_ids) for e in examples)
    ell = max((len(o) for e in examples for o in e.op_ids), default=1)
    p = N_BOUNDARY_POINTS

    fmax = max((len(e.fact_type_ids) for e in examples), default=0)

    # ``obj_rel`` width comes from the examples, not the ``D_REL`` constant: the
    # target-anchored scene emits 8, the anchor-free deployed scene emits 3, and one
    # collator serves both. All examples in a batch share a builder and therefore a
    # width; assert it rather than silently truncating a mismatched one to
    # ``examples[0]``.
    d_rel = examples[0].obj_rel.shape[-1] if examples else 3
    assert all(
        e.obj_rel.shape[-1] == d_rel for e in examples
    ), "obj_rel width differs within a batch"
    # Point/pose widths likewise come from the examples, not the literal 2/3: the 2D
    # boundary-ring path emits (P, 2) + (3,), the Restock3D point cloud (P, 3) + (4,).
    # One collator serves both; a batch shares a builder and therefore a width.
    point_dim = examples[0].obj_boundary.shape[-1] if examples else 2
    pose_dim = examples[0].obj_pose.shape[-1] if examples else 3
    assert all(
        e.obj_boundary.shape[-1] == point_dim for e in examples
    ), "obj_boundary point_dim differs within a batch"
    assert all(
        e.obj_pose.shape[-1] == pose_dim for e in examples
    ), "obj_pose pose_dim differs within a batch"
    # PointSet per-point features / kNN are trailing-nullable, emitted only when the
    # encoder upgrade is on. Widths (C_pt, k) come from the examples, same discipline as
    # point_dim/pose_dim above; config-off leaves both None and the batch is unchanged.
    _pf0 = examples[0].point_feats if examples else None
    point_feats: Optional[np.ndarray] = None
    knn_idx: Optional[np.ndarray] = None
    if _pf0 is not None:
        c_pt = _pf0.shape[-1]
        assert all(
            e.point_feats is not None and e.point_feats.shape[-1] == c_pt
            for e in examples
        ), "point_feats C_pt differs within a batch"
        point_feats = np.zeros((b, n, p, c_pt), np.float32)
        _kn0 = examples[0].knn_idx
        if _kn0 is not None:
            kdim = _kn0.shape[-1]
            assert all(
                e.knn_idx is not None and e.knn_idx.shape[-1] == kdim for e in examples
            ), "knn_idx k differs within a batch"
            knn_idx = np.zeros((b, n, p, kdim), np.int64)
    # init/goal atom profiles (doc spectre_atom_input_guide.md): trailing-nullable,
    # emitted only when atom_mode="profiles". Init and goal are gated independently, so
    # each pair may be present or None on its own. Widths (A, M) come from the examples,
    # same discipline as point_feats; config-off leaves all four None.
    _ai0 = examples[0].init_atom_pred if examples else None
    _aia0 = examples[0].init_atom_arg_tags if examples else None
    _ag0 = examples[0].goal_atom_pred if examples else None
    _aga0 = examples[0].goal_atom_arg_tags if examples else None
    init_atom_pred: Optional[np.ndarray] = None
    init_atom_arg_tags: Optional[np.ndarray] = None
    goal_atom_pred: Optional[np.ndarray] = None
    goal_atom_arg_tags: Optional[np.ndarray] = None
    if _ai0 is not None and _aia0 is not None:
        a_i = max(
            (len(e.init_atom_pred) for e in examples if e.init_atom_pred is not None),
            default=0,
        )
        m_i = _aia0.shape[-1]
        init_atom_pred = np.zeros((b, a_i), np.int64)
        init_atom_arg_tags = np.zeros((b, a_i, m_i), np.int64)
    if _ag0 is not None and _aga0 is not None:
        a_g = max(
            (len(e.goal_atom_pred) for e in examples if e.goal_atom_pred is not None),
            default=0,
        )
        m_g = _aga0.shape[-1]
        goal_atom_pred = np.zeros((b, a_g), np.int64)
        goal_atom_arg_tags = np.zeros((b, a_g, m_g), np.int64)
    obj_tags = np.zeros((b, n), np.int64)
    obj_boundary = np.zeros((b, n, p, point_dim), np.float32)
    obj_pose = np.zeros((b, n, pose_dim), np.float32)
    obj_rel = np.zeros((b, n, d_rel), np.float32)
    obj_is_goal = np.zeros((b, n), np.float32)
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
        obj_is_goal[bi, :no] = e.obj_is_goal
        obj_mask[bi, :no] = True
        if point_feats is not None and e.point_feats is not None:
            point_feats[bi, :no] = e.point_feats
            if knn_idx is not None and e.knn_idx is not None:
                knn_idx[bi, :no] = e.knn_idx
        if (
            init_atom_pred is not None
            and init_atom_arg_tags is not None
            and e.init_atom_pred is not None
            and e.init_atom_arg_tags is not None
        ):
            init_atom_pred[bi, : len(e.init_atom_pred)] = e.init_atom_pred
            init_atom_arg_tags[bi, : len(e.init_atom_pred)] = e.init_atom_arg_tags
        if (
            goal_atom_pred is not None
            and goal_atom_arg_tags is not None
            and e.goal_atom_pred is not None
            and e.goal_atom_arg_tags is not None
        ):
            goal_atom_pred[bi, : len(e.goal_atom_pred)] = e.goal_atom_pred
            goal_atom_arg_tags[bi, : len(e.goal_atom_pred)] = e.goal_atom_arg_tags
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
    return SpectreBatch(
        obj_tags=t(obj_tags),
        obj_boundary=t(obj_boundary),
        obj_pose=t(obj_pose),
        obj_rel=t(obj_rel),
        obj_is_goal=t(obj_is_goal),
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
        point_feats=t(point_feats) if point_feats is not None else None,
        knn_idx=t(knn_idx) if knn_idx is not None else None,
        init_atom_pred=t(init_atom_pred) if init_atom_pred is not None else None,
        init_atom_arg_tags=(
            t(init_atom_arg_tags) if init_atom_arg_tags is not None else None
        ),
        goal_atom_pred=t(goal_atom_pred) if goal_atom_pred is not None else None,
        goal_atom_arg_tags=(
            t(goal_atom_arg_tags) if goal_atom_arg_tags is not None else None
        ),
    )


def sample_context(
    fail_idx: list[int],
    rng: np.random.Generator,
    p_empty: float = 0.35,
    p_drop_facts: float = 0.3,
    max_f: int = 8,
    phi: Optional[int] = None,
) -> tuple[frozenset[int], bool]:
    """Sample a failure context ``F`` plus an evidence-dropout flag.

    Mass is heavy at ``|F| = 0`` because that is the deployment start: the static
    pathway has to stand on its own before any failure has been observed. ``hide_facts``
    drops the evidence for an example so the ranker cannot become dependent on it.

    ``phi`` (F-C2 rollout-aligned curriculum, ``decisions.md`` 2026-08-23) reshapes the
    *size* draw to the deployment visit distribution of **this** episode. A deployed
    rollout that ends at attempt ``phi + 1`` queries the ranker exactly once at each of
    ``|F| = 0, 1, ..., phi``, so ``Uniform{0..phi}`` *is* that rollout's visit
    distribution -- unlike the fixed ``[1, max_f]`` uniform, which both caps at 8 (the
    hard strata visit |F| well past that) and over-weights the tail. ``phi`` is a
    reference policy's deployed FP on the episode (built by
    ``experiments/spectre/fc2_build_phi.py``). The ``p_empty`` branch is kept **on top**
    so the static ranker still gets a fixed floor of |F|=0 training mass (30% in the
    deployed F-C2 arm, per the 2026-08-22 user directive) even on hard episodes whose
    rollout mass at |F|=0 is otherwise ~1/(phi+1). Capping at ``len(fail_idx)`` and never
    past ``phi`` respects the 2026-08-22 rollout-alignment guardrail: never oversample
    |F| a good ranker never reaches. ``None`` keeps the historical ``[1, max_f]`` draw.
    """
    if not fail_idx or rng.random() < p_empty:
        return frozenset(), False
    if phi is not None:
        # Uniform over {0..phi}, truncated to what the pool can supply. size==0 is a
        # legitimate |F|=0 draw (this episode's rollout does visit the empty context).
        size = int(rng.integers(0, min(int(phi), len(fail_idx)) + 1))
    else:
        size = int(rng.integers(1, min(max_f, len(fail_idx)) + 1))
    if size <= 0:
        return frozenset(), False
    chosen = rng.choice(np.asarray(fail_idx), size=size, replace=False)
    return frozenset(int(i) for i in chosen), bool(rng.random() < p_drop_facts)


def _hint_fact_arrays(
    episode: EpisodeRecord, context_f: frozenset[int], tags: dict[str, int]
) -> tuple[list[int], list[int], list[list[int]]]:
    """Hint-tier facts of the failed candidates, bound to episode-local tags.

    Proof-tier facts are excluded on purpose. Their sound consequence is applied outside
    the network as demotion; handing them to the scorer as *tokens* invites it to learn
    the crude correlate instead ("blocked sets are large, so prefer longer plans"),
    which measurably harmed the easy strata in v2.2 until the tiers were split.
    """
    hint = TIER_IDS["hint"]
    type_ids: list[int] = []
    tier_ids: list[int] = []
    arg_tags: list[list[int]] = []
    for fact in gather_context_facts(episode, sorted(context_f)):
        if fact.tier_id != hint:
            continue
        type_ids.append(fact.type_id)
        tier_ids.append(fact.tier_id)
        arg_tags.append([tags[a] for a in fact.args if a in tags][:MAX_FACT_ARGS])
    return type_ids, tier_ids, arg_tags


def _aggregate_per_query(records: list) -> list:
    """Collapse a candidate's failures to one record per distinct ``(schema, args)``.

    §6.1 defines a record as *the failing query and its arguments* -- one per query, not
    one per failed sample. The instrumented refiner emits one per sample, so a candidate
    whose `place-buffer(o)` was retried across many buffer poses contributes hundreds of
    near-identical tokens (measured: mean 2.2 per candidate but **max 290**, so a single
    s1 context at |F|=30 reached ~720 tokens against v2.2's ~40 facts). That is not
    extra information -- the samples are 99.3% distinct only in *which pose* failed,
    which the token does not even encode -- but it does dilute the scorer's attention
    and let one unlucky candidate dominate the evidence memory.

    Aggregation keeps the deepest occurrence (the furthest the plan got), sums effort,
    and takes the union of culprits, so nothing the token *encodes* is lost.
    """
    best: dict[tuple, list] = {}
    for rec in records:
        key = (rec.schema, rec.args)
        cur = best.get(key)
        if cur is None:
            best[key] = [rec, rec.n_step, set(rec.culprits)]
            continue
        cur[1] += rec.n_step
        cur[2].update(rec.culprits)
        if rec.step_index > cur[0].step_index:
            cur[0] = rec
    out = []
    for rec, effort, culprits in best.values():
        out.append(
            dataclasses.replace(rec, n_step=effort, culprits=tuple(sorted(culprits)))
        )
    return out


def _delta_arrays(delta, tags: dict[str, int], vocab: Vocab, arity: int) -> DeltaArrays:
    """One record's state delta as ``(added, deleted)`` lists of ``(pred id, arg
    tags)``.

    Predicate ids are shifted by **+1**: the vocab reserves index 0 for ``<OOV>`` while
    the embedding reserves it for padding, so without the shift an unknown predicate
    would be indistinguishable from an empty slot. ``Vocab.pred_idx`` raises on OOV,
    hence the ``.get`` -- the same idiom the schema lookup below uses.
    """

    def _role(atoms) -> list[DeltaAtomArray]:
        out = []
        for pred, args in atoms[:MAX_DELTA_ATOMS]:
            pid = int(vocab.predicates.get(pred, {"idx": 0})["idx"]) + 1
            out.append((pid, [tags[a] for a in args if a in tags][:arity]))
        return out

    if delta is None:
        return ([], [])
    return (_role(delta.added), _role(delta.deleted))


def _atom_profile_arrays(
    atoms, tags: dict[str, int], vocab: Vocab, arity: int
) -> tuple[np.ndarray, np.ndarray]:
    """One atom set as ``(pred ids (A,), arg tags (A, arity))`` for the
    AtomProfileEncoder.

    Mirrors :func:`_delta_arrays`: the predicate id carries the same **+1 shift** (vocab
    reserves 0 for ``<OOV>`` while the embedding reserves 0 for padding, so an unknown
    predicate stays distinct from an empty slot), and object arguments bind to the
    episode-local ``tags`` namespace with the same guarded lookup used everywhere the
    tag join could meet a non-object arg. Atoms are ``sorted`` so the emitted arrays are
    deterministic across runs (the atom set is a ``frozenset``); the encoder's scatter-
    sum is order-invariant, so ordering never changes a logit. A 0-ary atom (e.g.
    handempty) stays in-array with a real predicate id and an all-PAD arg row -- the
    encoder routes it to the global term by detecting that empty row.
    """
    rows = sorted(
        atoms, key=lambda a: (a.predicate.name, tuple(e.name for e in a.entities))
    )
    pred = np.zeros(len(rows), dtype=np.int64)
    argt = np.zeros((len(rows), arity), dtype=np.int64)  # 0 = PAD_TAG
    for i, atom in enumerate(rows):
        pred[i] = int(vocab.predicates.get(atom.predicate.name, {"idx": 0})["idx"]) + 1
        names = [tags[e.name] for e in atom.entities if e.name in tags][:arity]
        argt[i, : len(names)] = names
    return pred, argt


def build_record_arrays(
    episode: EpisodeRecord,
    context_f: frozenset[int],
    tags: dict[str, int],
    vocab: Vocab,
    spec: DomainSpec,
    aggregate: bool = False,
    state_delta: bool = False,
    record_holdout: bool = True,
) -> list[RecordArray]:
    """Failure records of the tried candidates, as ``(schema_id, args, culprits,
    scalars)``.

    **Proof-tier records are excluded**, exactly as v2.2 excluded proof-tier facts.
    Their sound consequence is applied outside the net as demotion; feeding them in as
    tokens invites the scorer to learn the crude correlate instead ("blocked sets are
    large, so prefer longer plans"), which measurably wrecked the easy strata in v2.2
    until the tiers were split. What reaches the net is evidence the deduction could
    *not* use.

    ``record_holdout`` (default ``True`` = current behavior) gates that exclusion. The
    learned-pathway workstream (``docs/failed_records_fix.md`` P-1) sets it ``False`` to
    feed the proof-tier ∧ provable records — on DD2D the certificate-grade ``retrieve``
    failures, whose signal the ``dead``/``coverage`` scalars already carry via the
    separate ``unified_evidence`` path — into the token stream for the first time, so a
    tokens-only arm is measured against the records it withheld, not a handicapped
    baseline. The deployed scalars-on config leaves it ``True`` (there the holdout is
    near-harmless de-duplication).

    Scalars are ``[j/L, log1p(effort)/10, exhausted, effort_is_total]``. The last is not
    decoration: backfilled records report whole-attempt effort while instrumented ones
    report per-step, so the flag tells the net which quantity it is reading instead of
    letting a re-collection silently redefine the column.

    ``state_delta`` appends §6.1's ``s_j`` as a fifth element, the delta from ``s_0``.
    It is computed here, on ``episode`` -- which ``build_example`` has already
    canonicalized -- so its object names land in the same ``tags`` namespace as the args
    and culprits. Computing it *after* aggregation is what makes it the *furthest
    reached* state per query: ``_aggregate_per_query`` keeps the deepest record, and the
    delta rides along on it. Note the delta's **size** is ~fully determined by the
    ``j/L`` scalar already present (measured corr 0.940), so what it adds is object
    *identity*; no count feature is derived from it, which would just be another length
    proxy.
    """
    arity = max(vocab.max_predicate_arity, 1)
    out: list[RecordArray] = []
    for idx in sorted(context_f):
        skeleton = episode.skeleton_pool[idx]
        plan_len = max(len(skeleton.operator_seq), 1)
        cand_records = records_for_candidate(
            episode, idx, spec, with_state_delta=state_delta
        )
        if aggregate:
            cand_records = _aggregate_per_query(cand_records)
        for rec in cand_records:
            if (
                record_holdout
                and spec.axioms_for(rec.schema).proof_tier()
                and rec.proves_failure()
            ):
                continue  # handled structurally by demotion, not learned
            row: RecordArray = (
                int(vocab.operators.get(rec.schema, 0)),
                [tags[a] for a in rec.args if a in tags][:MAX_RECORD_ARGS],
                # `dev_blame` is the fallback for environments with no class-1 channel:
                # objects named by the collateral deviation rather than by a check.
                # Never both -- `culprits` is empty exactly where `dev_blame` is
                # populated -- so the slot always carries one provenance, and on DD2D
                # `dev_blame` is absent and this reduces to the original expression.
                [tags[c] for c in (rec.culprits or rec.dev_blame) if c in tags][
                    :MAX_RECORD_CULPRITS
                ],
                [
                    rec.step_index / plan_len,
                    math.log1p(max(rec.n_step, 0)) / 10.0,
                    1.0 if rec.exhausted else 0.0,
                    1.0 if rec.effort_is_total else 0.0,
                ],
            )
            if state_delta:
                row = row + (_delta_arrays(rec.state_delta, tags, vocab, arity),)
            out.append(row)
    return out


# ── Rung-1 evidence-step stream (docs/failed_records_fix.md F-A) ──────────────────────
# One STEP token instead of one summary token per record: the failed step of an attempt
# plus each culprit's establishing step, all encoded in the CANDIDATE namespace so the
# shared CandidateEncoder makes a failed `place_short(b)` and the current candidate's
# `place_short(b)` identical vectors (the load-bearing "shared encoder" of F-A). A
# `StepArray` is (op_id, arg_tags, pos, status, attempt, culprit_tags, culprit_counts,
# scalars).
StepArray = tuple
STEP_STATUS_FAILED = 1  # the failed step of an attempt
STEP_STATUS_ESTABLISH = 2  # a successful prior step that seated a culprit
# N_STEP_SCALARS, MAX_ATTEMPTS are imported from `model` (single source of truth).


def build_evidence_steps(
    episode: EpisodeRecord,
    context_f: frozenset[int],
    tags: dict[str, int],
    vocab: Vocab,
    spec: DomainSpec,
    record_holdout: bool = True,
) -> list[StepArray]:
    """Per failed attempt: its failed step + each culprit's establishing step.

    Rung-1 enrichment (F-A). Steps are emitted in the candidate namespace (op id +
    position + arg tags) so the shared ``CandidateEncoder.encode_steps`` embeds them with
    the current candidate's weights. Per-culprit sample counts (F-A2) are recovered from
    the **raw** (pre-aggregation) records — ``_aggregate_per_query`` would union them away.
    A blameless failure emits exactly one token (the rung-0 budget); a culprit-bearing one
    emits ``1 + |culprits|``. The ``record_holdout`` gate matches ``build_record_arrays``.

    ``(schema, args)`` is the step identity (valid only because no skeleton repeats a
    ground action — no un-store op); the establishing step is the *last* prefix step whose
    args name the culprit, exactly the seating-chart scan the ``regroup`` feature uses.
    """
    arity = max(vocab.max_operator_arity, 1)
    out: list[StepArray] = []
    for attempt_id, idx in enumerate(sorted(context_f)):
        skeleton = episode.skeleton_pool[idx]
        seq = [
            (op.name, tuple(p.name for p in op.parameters))
            for op in skeleton.operator_seq
        ]
        raw = records_for_candidate(episode, idx, spec, with_state_delta=False)
        # per-(schema, args) per-culprit raw sample counts, before aggregation unions them.
        counts: dict = {}
        for r in raw:
            cc = counts.setdefault((r.schema, r.args), {})
            for c in r.culprits or r.dev_blame:
                cc[c] = cc.get(c, 0) + 1
        for rec in _aggregate_per_query(raw):
            if (
                record_holdout
                and spec.axioms_for(rec.schema).proof_tier()
                and rec.proves_failure()
            ):
                continue
            t = min(max(int(rec.step_index), 0), max(len(seq) - 1, 0))
            culprits = [c for c in (rec.culprits or rec.dev_blame) if c in tags]
            cc = counts.get((rec.schema, rec.args), {})
            cul_tags = [tags[c] for c in culprits][:MAX_RECORD_CULPRITS]
            cul_counts = [math.log1p(cc.get(c, 1)) for c in culprits][
                :MAX_RECORD_CULPRITS
            ]
            scalars = [
                1.0 if rec.exhausted else 0.0,
                math.log1p(max(rec.n_step, 0)) / 10.0,
                1.0 if rec.effort_is_total else 0.0,
            ]
            out.append(
                (
                    int(vocab.operators.get(rec.schema, 0)),
                    [tags[a] for a in rec.args if a in tags][:arity],
                    t,
                    STEP_STATUS_FAILED,
                    attempt_id,
                    cul_tags,
                    cul_counts,
                    scalars,
                )
            )
            for (
                c
            ) in culprits:  # establishing step = last prefix step naming the culprit
                for u in range(t - 1, -1, -1):
                    if c in seq[u][1]:
                        out.append(
                            (
                                int(vocab.operators.get(seq[u][0], 0)),
                                [tags[a] for a in seq[u][1] if a in tags][:arity],
                                u,
                                STEP_STATUS_ESTABLISH,
                                attempt_id,
                                [],
                                [],
                                [0.0] * N_STEP_SCALARS,
                            )
                        )
                        break
    return out


def build_example(
    episode: EpisodeRecord,
    vocab: Vocab,
    rng: Optional[np.random.Generator] = None,
    max_tags: int = 32,
    evidence: bool = False,
    context_f: Optional[frozenset[int]] = None,
    hide_facts: bool = False,
    augment_tags: bool = True,
    spec: Optional[DomainSpec] = None,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
    coverage_feats: bool = False,
    coverage_mode: str = "both",
    repeat_feats: bool = False,
    regroup_feats: bool = False,
    state_delta: bool = False,
    record_holdout: bool = True,
    record_mode: str = "summary",
    scene_3d: bool = False,
    pointset_feats: bool = False,
    use_pca_feats: bool = False,
    edgeconv_k: int = 0,
    emit_init_atoms: bool = False,
    emit_goal_atoms: bool = False,
    evidence_context: Optional[frozenset[int]] = None,
) -> tuple[SpectreExample, list[RecordArray]]:
    """Tensorize one geometry-carrying episode for the v3 model.

    Returns ``(example, record_arrays)``. The records come back from *here* rather than
    from a separate call because they must use the same canonicalization and the same
    tag assignment as the example -- computing them separately would both double the
    canonicalization cost and risk binding record tags to a different permutation than
    the scene tags, which is exactly the identity bug that made record tokens carry no
    object information at all. A caller whose model has no record encoder simply ignores
    the second element; it costs nothing to produce when the context is empty.

    ``spec`` defaults to the contract registered for the episode's own ``env_variant``,
    so a caller cannot accidentally tensorize DD2D under another domain's axioms.
    """
    if episode.scene_geometry is None:
        raise ValueError("build_example requires scene_geometry")
    spec = spec or spec_for(episode.provenance.env_variant)

    canon = canonicalize_episode(episode, rng=None)
    geo = canon.scene_geometry
    assert geo is not None
    tags = assign_tags(
        [o.name for o in geo.objects],
        rng=(rng if augment_tags else None),
        max_tags=max_tags,
    )

    # --- scene tokens -------------------------------------------------------
    # Normalisation frame. `drawer_w`/`drawer_d` are DD2D's spelling and are kept as
    # accepted aliases so every stored DD2D episode tensorizes byte-identically; a
    # second environment writes the generic `frame_w`/`frame_d` instead. An absent frame
    # still falls back to `scale = 1.0` -- unnormalised, which is what the older
    # RT2D/kinder records get and what SB2D would silently get if it wrote neither
    # spelling.
    frame = geo.frame or {}
    # `drawer_w`/`drawer_d` are DD2D's spelling; `frame_w`/`frame_d` the generic one a
    # second environment writes. Require at least one: an absent frame used to fall back
    # to scale=1.0 *silently*, leaving obj_pose unnormalized and mixing units across
    # environments (cm on DD2D, m on SB2D). Fail loudly and name the fix instead.
    _fw = frame.get("drawer_w", frame.get("frame_w"))
    _fd = frame.get("drawer_d", frame.get("frame_d"))
    if _fw is None and _fd is None:
        raise ValueError(
            "scene_geometry.frame lacks a normalization extent: write frame_w/frame_d "
            "(DD2D uses drawer_w/drawer_d). Without it obj_pose is unnormalized "
            "(scale=1)."
        )
    scale = max(float(_fw or 0.0), float(_fd or 0.0), 1.0)
    # The goal channel is `is_goal` (any object named by the goal atoms), not
    # `is_target` (the one object a DD2D JSON flagged). `is_target` presupposes a single
    # distinguished target and is silently all-zero on an env whose goal names several
    # objects (SB2D); `is_goal` is well-defined for any goal, including N>1 targets,
    # and is byte-identical to `is_target` on every DD2D episode (proven 720/720). The
    # target-anchored `obj_rel` columns (dx, dy, dist to the target, area ratio to the
    # target) and the privileged `concave` flag are cut for the same reason -- only the
    # three anchor-free per-object scalars `[area, sinθ, cosθ]` remain. Absolute
    # position is unaffected (it lives in `obj_pose`). See docs/decisions 2026-08-08.
    goal_objs = spec.goal_objects(canon)

    n_obj = len(geo.objects)
    obj_tags = np.array([tags[o.name] for o in geo.objects], dtype=np.int64)
    if scene_3d:
        # 3D path: the analytic point cloud + an (x, y, z, yaw) pose. Restock3D populates
        # ``point_cloud``/``pose_z`` on every object; a 2D episode reaching here under
        # ``scene_3d`` is a config/data mismatch, so fail loudly rather than lift a flat
        # footprint to z=0 silently.
        clouds = []
        for o in geo.objects:
            if o.point_cloud is None:
                raise ValueError(
                    f"scene_3d=True but object {o.name!r} has no point_cloud "
                    "(train a 2D env without --scene-3d)."
                )
            clouds.append(sample_point_cloud(o.point_cloud))
        obj_boundary = np.stack(clouds)
        obj_pose = np.array(
            [
                [
                    o.pose[0] / scale,
                    o.pose[1] / scale,
                    (o.pose_z or 0.0) / scale,
                    o.pose[2],
                ]
                for o in geo.objects
            ],
            dtype=np.float32,
        )
    else:
        obj_boundary = np.stack([resample_ring(list(o.boundary)) for o in geo.objects])
        obj_pose = np.array(
            [[o.pose[0] / scale, o.pose[1] / scale, o.pose[2]] for o in geo.objects],
            dtype=np.float32,
        )
    obj_is_goal = np.array(
        [1.0 if o.name in goal_objs else 0.0 for o in geo.objects], dtype=np.float32
    )
    rel = np.zeros((n_obj, D_REL), dtype=np.float32)
    for i, o in enumerate(geo.objects):
        rel[i] = [o.area, *_sin_cos(o.pose[2])]

    # --- PointSetEncoder per-point features (doc pointset_encoder_upgrade.md) ------
    # Computed from the same point set that feeds ``obj_boundary`` plus an inside-test
    # oracle (2D: the source polygon; 3D: the origin-centered box). Gated so the
    # config-off path skips it and ``obj_boundary``/``obj_pose`` above stay unchanged.
    point_feats: Optional[np.ndarray] = None
    knn_idx: Optional[np.ndarray] = None
    if pointset_feats:
        _k = edgeconv_k if edgeconv_k > 0 else (6 if scene_3d else 4)
        pf_list: list[np.ndarray] = []
        kn_list: list[np.ndarray] = []
        for i, o in enumerate(geo.objects):
            pts_i = obj_boundary[i]
            if use_pca_feats:
                inside = _box_inside(pts_i) if scene_3d else _shapely_inside(o.boundary)
            else:
                inside = None
            pf_i, kn_i = compute_point_feats(pts_i, inside, _k, scene_3d, use_pca_feats)
            pf_list.append(pf_i)
            kn_list.append(kn_i)
        point_feats = np.stack(pf_list)
        knn_idx = np.stack(kn_list)

    # --- init/goal atom profiles (doc spectre_atom_input_guide.md) ----------
    # Emitted as (pred id, arg tags) per atom, mirroring the state-delta idiom; the model
    # scatter-sums each atom onto the object tokens it mentions. Read off the canonical
    # atoms (``canon``), so atom object names share the ``tags`` namespace with the scene
    # and candidate tokens. Off ⇒ None, leaving the config-off tensorizer byte-unchanged.
    _atom_arity = max(vocab.max_predicate_arity, 1)
    init_atom_pred: Optional[np.ndarray] = None
    init_atom_arg_tags: Optional[np.ndarray] = None
    goal_atom_pred: Optional[np.ndarray] = None
    goal_atom_arg_tags: Optional[np.ndarray] = None
    if emit_init_atoms:
        init_atom_pred, init_atom_arg_tags = _atom_profile_arrays(
            canon.initial_abstract_state.atoms, tags, vocab, _atom_arity
        )
    if emit_goal_atoms:
        goal_atom_pred, goal_atom_arg_tags = _atom_profile_arrays(
            canon.goal_atoms, tags, vocab, _atom_arity
        )

    # --- candidate tokens ---------------------------------------------------
    op_ids: list[list[int]] = []
    arg_tags: list[list[list[int]]] = []
    success: list[Optional[bool]] = []
    subsets: list[frozenset[str]] = []
    lengths: list[int] = []
    for skel, out in zip(canon.skeleton_pool, canon.outcomes):
        op_ids.append(
            [int(vocab.operators.get(op.name, 0)) for op in skel.operator_seq]
        )
        arg_tags.append(
            [[tags.get(a.name, 0) for a in op.parameters] for op in skel.operator_seq]
        )
        subsets.append(spec.manipulated(skel, goal_objs))
        lengths.append(spec.length_key(skel))
        success.append(out.outcome == "success")

    # Aux targets stay -1 (= ignore) until the necessity labeller lands: no collection
    # has ever populated `aux_labels`, so v2.2's aux head was masked out entirely and
    # never received a gradient. Pretending otherwise here would silently train it.
    aux = np.full(n_obj, -1.0, dtype=np.float32)

    # --- failure context ----------------------------------------------------
    k = len(op_ids)
    fail_idx = [i for i, out in enumerate(canon.outcomes) if out.outcome == "fail"]
    hide = hide_facts
    if context_f is not None:
        ctx: frozenset[int] = frozenset(int(i) for i in context_f)
    elif evidence and rng is not None:
        ctx, hide = sample_context(fail_idx, rng)
    else:
        ctx = frozenset()
    avail = [i not in ctx for i in range(k)]

    # W2 evidence-composition probe (docs/failed_records_fix_part2.md §2): cap the failure
    # EVIDENCE (record + hint-fact tokens) to a subset of the context while leaving `avail`
    # (re-try mask) and the |F| gate on the full `ctx`. `None` ⇒ ev_ctx = ctx, byte-identical.
    # The caller (deployed_rollout_traced, evidence_cap_k) passes the k most-recently-tried
    # failures, so the model conditions on fewer records without changing which candidates
    # remain available or what |F| the residual's gate reads.
    ev_ctx = ctx if evidence_context is None else (frozenset(evidence_context) & ctx)

    if hide or not ev_ctx:
        ftype: list[int] = []
        ftier: list[int] = []
        farg: list[list[int]] = []
    else:
        ftype, ftier, farg = _hint_fact_arrays(canon, ev_ctx, tags)

    # --- planner signals + structural evidence ------------------------------
    # Column 0 (enumeration order) is inert: the v3 scorer takes no prior features.
    # Column 1 carries the normalized plan length, kept only because the within-length
    # PL loss buckets on it; it is `domain.length_key` rescaled, so the partition is the
    # domain's.
    max_len = max(lengths) if lengths else 1
    prior = [[-(i / max(k - 1, 1)), -(lengths[i] / max(max_len, 1))] for i in range(k)]

    # `overlap_mode` zeroes a column rather than narrowing the tensor, so the state dict
    # shape is untouched and the D-8 exact-absence oracle keeps loading. A zeroed column
    # *is* the feature's absence: its weight receives no gradient signal from it.
    #
    # Dropping `dead` is a C5 argument, not a tuning knob. `dead` is the proof rule fed
    # to the net as a feature, and it is strongly anti-correlated with subset size
    # (corr −0.284; mean |S| 1.38 dead vs 2.39 alive on dd2d_v4 train), so the net can
    # fit it as "short ⇒ bad". That is sound only where the rule actually fired and is
    # L4's failure mode everywhere else -- which is why s1, the stratum on which short
    # *is* correct, regressed. The sound consequence still applies outside the net as
    # the demotion offset, where a wrong weight cannot override it.
    want_dead = overlap_mode in ("both", "dead")
    want_jac = overlap_mode in ("both", "jaccard")
    want_cov = coverage_feats
    # `coverage_mode` splits the pair the same way, and for the same reason: they answer
    # different questions. `coverage` asks "does this candidate discharge the culprit
    # before re-entering the situation that named it"; `waste` asks "of the steps its
    # own causal chain cannot justify, do any answer to no named culprit". (Unified
    # definitions; see `unified_evidence.py`.) They have only ever been measured
    # together, so which one carries the effect is unknown; zeroing one column isolates
    # it without changing any shape.
    want_coverage = want_cov and coverage_mode in ("both", "coverage")
    want_waste = want_cov and coverage_mode in ("both", "waste")
    # `repeat` (F3 exact-step certificate) and `regroup` (F2 seating-chart) append two
    # more columns after the coverage pair, gated by their own flags and zeroed the same
    # way. Trailing-additive: the width grows only when a flag is on, so an older
    # checkpoint (flags absent) reconstructs at its original width and loads. Both are
    # the learned-feature analogue of the P2 oracle certificates (docs/adaptivity_probe_
    # plan_restock3d_v3.md): `repeat` fires on a blameless, exhausted failure of a
    # `step_certificate` schema (restock3d F3); `regroup` on a culprit-bearing record's
    # seating chart (restock3d F2).
    want_repeat = repeat_feats
    want_regroup = regroup_feats
    want_rr = want_repeat or want_regroup
    n_ov = 2 + (2 if want_cov else 0) + (2 if want_rr else 0)
    overlap = [[0.0] * n_ov for _ in range(k)]
    _uni_records: list = []
    _uni_pool: frozenset = frozenset()
    _uni_universal: frozenset = frozenset()
    _rr_repeat_steps: set = set()  # exact (schema, canon-args) certificates (repeat)
    _rr_charts: list = []  # seating charts: list[frozenset[(schema, args)]]
    _cand_step_sets: list = []  # per-candidate set of (schema, canon-args) steps
    if ctx and not hide:
        blocked = [subsets[f] for f in ctx if spec.licenses_demotion(canon.outcomes[f])]
        failed = [subsets[f] for f in ctx]
        if want_cov:
            # Lifted operators come from the pool's own `GroundOperator.parent`, so the
            # filters stay env-agnostic -- nothing here needs to know the domain.
            lifted = frozenset(
                op.parent for skel in canon.skeleton_pool for op in skel.operator_seq
            )
            _uni_universal, actionable = scene_filters(
                lifted, frozenset(canon.initial_abstract_state.objects)
            )
            _uni_records = records_from_failure_records(canon, ctx, spec)
            _uni_pool = frozenset(
                n
                for r in _uni_records
                for n in _unified_blame(r)
                if n in actionable and n not in _uni_universal
            )
        if want_rr:
            _cand_step_sets = [
                {
                    (op.name, tuple(p.name for p in op.parameters))
                    for op in skel.operator_seq
                }
                for skel in canon.skeleton_pool
            ]
            for f in ctx:
                seq_f = [
                    (op.name, tuple(p.name for p in op.parameters))
                    for op in canon.skeleton_pool[f].operator_seq
                ]
                for r in records_for_candidate(canon, f, spec):
                    t = min(max(int(r.step_index), 0), len(seq_f) - 1)
                    failed_step = seq_f[t]
                    # `blame == empty` = neither a class-1 culprit nor a class-2 deviation
                    # witness: an intrinsic dead step, not a means-failure and not F2.
                    blame_empty = not r.culprits and not r.dev_blame
                    if (
                        want_repeat
                        and spec.axioms_for(r.schema).step_certificate
                        and r.proves_failure()
                        and blame_empty
                    ):
                        _rr_repeat_steps.add(failed_step)
                    if (
                        want_regroup
                        and r.culprits
                        and spec.axioms_for(r.schema).grouping_certificate
                    ):
                        # Seating chart = the failed step + each culprit's establishing
                        # step (the last prefix step naming it). Sound in v3: no un-store
                        # op, so co-occurrence implies final co-residency.
                        chart = {failed_step}
                        for c in r.culprits:
                            for u in range(t - 1, -1, -1):
                                if c in seq_f[u][1]:
                                    chart.add(seq_f[u])
                                    break
                        _rr_charts.append(frozenset(chart))
        for i, si in enumerate(subsets):
            dead = 1.0 if any(si <= b for b in blocked) else 0.0
            jaccard = max(
                (len(si & f) / max(len(si | f), 1) for f in failed), default=0.0
            )
            row = [
                dead if want_dead else 0.0,
                float(jaccard) if want_jac else 0.0,
            ]
            if want_cov:
                # The unified definitions (`unified_evidence.py`). Deployed since
                # 2026-07-31. Computes discretionary work from the candidate's own
                # causal structure -- "does this candidate discharge the culprit before
                # re-entering the situation that named it" -- rather than from a
                # goal-object subtraction.
                _cov, _wst = coverage_and_waste(
                    list(canon.skeleton_pool[i].operator_seq),
                    _uni_records,
                    _uni_pool,
                    canon.initial_abstract_state.atoms,
                    canon.goal_atoms,
                    _uni_universal,
                )
                row += [
                    _cov if want_coverage else 0.0,
                    _wst if want_waste else 0.0,
                ]
            if want_rr:
                steps_i = _cand_step_sets[i]
                row += [
                    (
                        (1.0 if (_rr_repeat_steps & steps_i) else 0.0)
                        if want_repeat
                        else 0.0
                    ),
                    (
                        (1.0 if any(ch <= steps_i for ch in _rr_charts) else 0.0)
                        if want_regroup
                        else 0.0
                    ),
                ]
            overlap[i] = row

    records = (
        build_record_arrays(
            canon,
            ev_ctx,
            tags,
            vocab,
            spec,
            aggregate_records,
            state_delta,
            record_holdout,
        )
        if (ev_ctx and not hide)
        else []
    )
    rec_steps = (
        build_evidence_steps(canon, ev_ctx, tags, vocab, spec, record_holdout)
        if (record_mode == "steps" and ev_ctx and not hide)
        else None
    )

    return (
        SpectreExample(
            obj_tags,
            obj_boundary,
            obj_pose,
            rel,
            obj_is_goal,
            op_ids,
            arg_tags,
            success,
            aux,
            aux.copy(),
            avail,
            ftype,
            ftier,
            farg,
            prior,
            overlap,
            point_feats,
            knn_idx,
            init_atom_pred,
            init_atom_arg_tags,
            goal_atom_pred,
            goal_atom_arg_tags,
            rec_steps=rec_steps,
        ),
        records,
    )


def _sin_cos(theta: float) -> tuple[float, float]:
    return math.sin(theta), math.cos(theta)


def collate(
    examples: list[SpectreExample],
    max_arity: int,
    records: Optional[list[list[RecordArray]]] = None,
    max_pred_arity: int = 1,
) -> SpectreBatch:
    """Pad + stack examples into a batch the v3 model consumes.

    ``records`` is per-example and optional; without it the result is exactly a v2.2
    batch, which is what keeps the compat path (and the equivalence oracle) intact.
    """
    # `_collate_base` hard-codes a width-2 `cand_overlap` and D-7 freezes it, so a wider
    # example is stacked here instead. Narrow *copies* go to `_collate_base` -- mutating
    # the caller's examples in place and restoring them afterwards would work today and
    # break the moment anything holds a reference.
    wide = max((len(e.overlap[0]) if e.overlap else 2) for e in examples)
    narrow = (
        [
            dataclasses.replace(e, overlap=[row[:2] for row in e.overlap])
            for e in examples
        ]
        if wide > 2
        else examples
    )
    base = _collate_base(narrow, max_arity=max_arity)
    batch = SpectreBatch(**base.__dict__)
    if wide > 2:
        b_, k_ = batch.pool_mask.shape
        ov_arr = np.zeros((b_, k_, wide), np.float32)
        for bi, e in enumerate(examples):
            for ki, row in enumerate(e.overlap[:k_]):
                ov_arr[bi, ki] = row
        batch.cand_overlap = torch.as_tensor(ov_arr)
    # Rung-1 evidence steps (docs/failed_records_fix.md F-A). Independent of the summary
    # `records` stream, and carried on the example itself (built with the same
    # canonicalization), so it is padded here rather than threaded through the signature.
    _steps = [e.rec_steps or [] for e in examples]
    if any(_steps):
        b = len(examples)
        s = max(len(st) for st in _steps)
        st_op = np.zeros((b, s), np.int64)
        st_arg = np.zeros((b, s, max_arity), np.int64)
        st_pos = np.zeros((b, s), np.int64)
        st_status = np.zeros((b, s), np.int64)
        st_attempt = np.zeros((b, s), np.int64)
        st_cul = np.zeros((b, s, MAX_RECORD_CULPRITS), np.int64)
        st_cnt = np.zeros((b, s, MAX_RECORD_CULPRITS), np.float32)
        st_scal = np.zeros((b, s, N_STEP_SCALARS), np.float32)
        st_mask = np.zeros((b, s), bool)
        for bi, st in enumerate(_steps):
            for si, (op, ar, pos, status, attempt, cul, cnt, scal) in enumerate(st):
                st_op[bi, si] = op
                st_arg[bi, si, : len(ar)] = ar
                st_pos[bi, si] = pos
                st_status[bi, si] = status
                st_attempt[bi, si] = min(int(attempt), MAX_ATTEMPTS - 1)
                st_cul[bi, si, : len(cul)] = cul
                st_cnt[bi, si, : len(cnt)] = cnt
                st_scal[bi, si] = scal
                st_mask[bi, si] = True
        t = torch.as_tensor
        batch.rec_step_op_ids = t(st_op)
        batch.rec_step_arg_tags = t(st_arg)
        batch.rec_step_pos = t(st_pos)
        batch.rec_step_status = t(st_status)
        batch.rec_step_attempt = t(st_attempt)
        batch.rec_step_culprit_tags = t(st_cul)
        batch.rec_step_culprit_counts = t(st_cnt)
        batch.rec_step_scalars = t(st_scal)
        batch.rec_step_mask = t(st_mask)
    if not records or not any(records):
        return batch

    b = len(examples)
    r = max(len(rs) for rs in records)
    schema = np.zeros((b, r), np.int64)
    args = np.zeros((b, r, MAX_RECORD_ARGS), np.int64)
    culprits = np.zeros((b, r, MAX_RECORD_CULPRITS), np.int64)
    scalars = np.zeros((b, r, N_RECORD_SCALARS), np.float32)
    mask = np.zeros((b, r), bool)
    # Emitted on the *presence of the delta element*, never on whether any delta is
    # non-empty. Gating on non-emptiness would encode a j=0 record one way beside a
    # batch-mate that has a delta and another way alone -- and deploy collates a single
    # example per step, so both cases are routine (~48% of aggregated tokens are empty).
    wants_delta = any(len(row) > 4 for rs in records for row in rs)
    a = max(max_pred_arity, 1)
    d_pred = np.zeros((b, r, 2, MAX_DELTA_ATOMS), np.int64)
    d_args = np.zeros((b, r, 2, MAX_DELTA_ATOMS, a), np.int64)
    for bi, rs in enumerate(records):
        for ri, row in enumerate(rs):
            sid, ar, cu, sc = row[:4]
            schema[bi, ri] = sid
            args[bi, ri, : len(ar)] = ar
            culprits[bi, ri, : len(cu)] = cu
            scalars[bi, ri] = sc
            mask[bi, ri] = True
            if not wants_delta or len(row) <= 4:
                continue
            for role, atoms in enumerate(row[4]):
                for ai, (pid, atags) in enumerate(atoms[:MAX_DELTA_ATOMS]):
                    d_pred[bi, ri, role, ai] = pid
                    d_args[bi, ri, role, ai, : len(atags)] = atags[:a]

    t = torch.as_tensor
    batch.rec_schema_ids = t(schema)
    batch.rec_arg_tags = t(args)
    batch.rec_culprit_tags = t(culprits)
    batch.rec_scalars = t(scalars)
    batch.rec_mask = t(mask)
    if wants_delta:
        batch.rec_delta_pred_ids = t(d_pred)
        batch.rec_delta_arg_tags = t(d_args)
    return batch
