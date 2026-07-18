"""DD2D ground-truth labeling -- the Day-1 fallback labeler (spec Section 8.4 fallback).

Every candidate is labeled ``feasible`` / ``infeasible`` / ``marginal`` under an explicit
compute budget. This is the honest Day-1 milestone: a **real positive certificate** (an
accessible delta-clearance packing of the subset into the buffer) plus the **sound H1 area
bound** for negatives. Any negative NOT proven by the area bound (or by drawer-side
extraction infeasibility) is left **provisional -> marginal**, never a hard ``infeasible``:

* ``feasible``  <=> an extraction order exists (Section 7b) AND an accessible delta-packing is found;
* ``infeasible`` <=> no extraction order, OR the area bound proves no packing can exist;
* ``marginal``   <=> neither (reason ``inaccessible`` = packs but no grasp order clears;
                    reason ``budget`` = no packing found within the restart budget --
                    provisional, because the arrangement-complete negative certificate
                    (spec Section 8.4) is deferred).

Accessible packing (spec Section 8.2): a packing PLUS an insertion order PLUS, for each
item, a grasp whose fingers clear the already-placed items. Because the compaction sampler
places items bottom-left-first and fingers may overhang the (wall-less) buffer edge, the
incremental pack order is itself an accessible order when each placement is graspable
against the already-staged set -- so we fold accessibility into the packing loop.

Filters (spec Section 9.4), the ONLY generation-time filters:
* **F1** target blocked, **F2** >= 2 distinct minimal clearing subsets, **F3** >= 1
  confidently-feasible candidate.
"""

from __future__ import annotations

import random

from shapely import Polygon

from .enumerate import (
    Candidate,
    _blocker_sets,
    _clears_target,
    _footprints,
    _minimal_sets,
    target_open_grasp,
)
from .grasps import has_grasp
from .world import DrawerScene, place_polygon, sample_buffer_pose

RESTARTS = 3  # positive-certificate packing restarts (spec Section 8.3)
PLACE_TRIES = 8  # per-item accessible-pose attempts within a restart


# --------------------------------------------------------------------------- #
# positive accessible-packing certificate
# --------------------------------------------------------------------------- #
def _pack(
    scene: DrawerScene, subset, rng: random.Random, inflate: float, require_access: bool
):
    """Incrementally pack ``subset`` into the buffer (largest first) with
    delta/2-clearance.

    If ``require_access`` each placement must also have a grasp clearing the already-staged
    items (an accessible order by construction). Returns a witness dict or ``None``.
    """
    order = sorted(subset, key=lambda n: -scene.items[n].shape.area)
    staged_raw: list[Polygon] = []
    staged_inf: list[Polygon] = []
    poses: dict[str, tuple[float, float, float]] = {}
    for name in order:
        shape = scene.items[name].shape
        placed = False
        for _ in range(PLACE_TRIES):
            pose = sample_buffer_pose(
                shape, scene.buffer, staged_inf, rng, inflate=inflate
            )
            if pose is None:
                continue
            if require_access and has_grasp(shape, pose, staged_raw) is None:
                continue  # not graspable clearing the already-staged items
            fp = place_polygon(shape.polygon, pose)
            staged_raw.append(fp)
            staged_inf.append(fp.buffer(inflate) if inflate > 0 else fp)
            poses[name] = pose
            placed = True
            break
        if not placed:
            return None
    return {"order": order, "poses": poses}


def _area_bound_infeasible(scene: DrawerScene, subset) -> bool:
    """Sound H1 pruning (spec Section 8.4): sum of delta/2-deflated areas > buffer area
    => no packing can exist.

    One-directional (infeasibility only).
    """
    half = scene.margin / 2.0
    total = 0.0
    for n in subset:
        d = scene.items[n].shape.polygon.buffer(-half)
        total += float(d.area) if not d.is_empty else 0.0
    return total > scene.buffer.area


# --------------------------------------------------------------------------- #
# candidate label
# --------------------------------------------------------------------------- #
def label_candidate(scene: DrawerScene, cand: Candidate, seed: int = 0) -> Candidate:
    """Label one candidate in place (fills ``cand.meta``); returns it for chaining."""
    subset = cand.subset
    inflate = scene.margin / 2.0

    if not cand.extractable:
        cand.meta.update(label="infeasible", reason="extraction", witness=None)
        return cand
    if _area_bound_infeasible(scene, subset):
        cand.meta.update(label="infeasible", reason="packing", witness=None)
        return cand

    packed_any = False
    for r in range(RESTARTS):
        rng = random.Random((seed * 7919 + r) & 0xFFFFFFFF)
        witness = _pack(scene, subset, rng, inflate, require_access=True)
        if witness is not None:
            cand.meta.update(
                label="feasible",
                reason="",
                witness={
                    "order": witness["order"],
                    "poses": {k: list(v) for k, v in witness["poses"].items()},
                },
            )
            return cand
    # not accessibly packable within budget: distinguish inaccessible vs provisional budget
    for r in range(RESTARTS):
        rng = random.Random((seed * 7919 + 100 + r) & 0xFFFFFFFF)
        if _pack(scene, subset, rng, inflate, require_access=False) is not None:
            packed_any = True
            break
    reason = "inaccessible" if packed_any else "budget"
    cand.meta.update(label="marginal", reason=reason, witness=None)
    return cand


def label_all(
    scene: DrawerScene, candidates: list[Candidate], seed: int = 0
) -> list[Candidate]:
    for i, c in enumerate(candidates):
        label_candidate(scene, c, seed=seed + 31 * i)
    return candidates


def min_feasible_subset_size(candidates: list[Candidate]) -> int | None:
    """The size of the smallest confidently-feasible clearing subset (``None`` if none).

    ``>= 2`` means no single-object removal is a feasible clearing plan -- the problem
    genuinely requires identifying a blocking SUBSET (docs/dd2d.md).
    """
    sizes = [c.size for c in candidates if c.meta.get("label") == "feasible"]
    return min(sizes) if sizes else None


# --------------------------------------------------------------------------- #
# decision-relevance filters (spec Section 9.4)
# --------------------------------------------------------------------------- #
def decision_filters(scene: DrawerScene, candidates: list[Candidate]) -> dict:
    """Evaluate F1/F2/F3 AFTER the full labeling pass (spec Section 9.3)."""
    fps = _footprints(scene)
    f1 = target_open_grasp(scene, fps) is None
    minimal = _minimal_sets(_blocker_sets(scene, fps))
    clearing_minimal = [m for m in minimal if _clears_target(scene, fps, m)]
    f2 = len(clearing_minimal) >= 2
    f3 = any(c.meta.get("label") == "feasible" for c in candidates)
    return {"F1": f1, "F2": f2, "F3": f3, "n_clearing_minimal": len(clearing_minimal)}
