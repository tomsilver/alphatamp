"""DD2D geometric candidate enumeration (spec Section 7).

Dataset infrastructure, **not** a planner: it computes the set of clearing candidates
using information the symbolic layer deliberately lacks (which items block the target's
grasp fingers), so it is disclosed and never presented as a baseline. For each target
grasp cell it finds the blocking item set; the minimal sets under inclusion are the core
clearing subsets, grown by seeded supersets (adjacent item, <2 cm) up to a cap. Two
re-checks close the first-order-optimistic blocker computation:

* **clearing** (a): with the subset removed, the target has >= 1 collision-free grasp;
* **extraction order** (b): there is an order in which each member, at its turn, has a
  grasp whose fingers clear all items not yet removed and the wall band. Candidates with
  no extraction order are RETAINED but pre-flagged infeasible(extraction) -- planners
  face them, so the dataset keeps them (spec Section 7 / M2).

Published order: ascending |S|, ties by a seeded permutation (spec Section 7 / M5 -- itself
a weak packing heuristic, which is why the Tier-2 planner also offers random/slack orders).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from shapely import Polygon

from .grasps import Grasp, finger_rects, grasp_cells, has_grasp
from .world import DrawerScene

ADJACENCY = (
    2.0  # superset growth: an item within 2 cm of a minimal set (spec Section 7)
)
MAX_CANDIDATES = 40  # spec Section 7


@dataclass
class Candidate:
    """A clearing subset = a staging skeleton (pick/place-buffer per member,
    retrieve)."""

    subset: frozenset[str]
    members: list[str]  # a valid drawer-removal order if one exists, else sorted names
    extractable: bool  # (b): an extraction order over the drawer exists
    extraction_reason: str | None = None  # "extraction" if not extractable
    meta: dict = field(
        default_factory=dict
    )  # filled by the labeler (label, reason, ...)

    @property
    def size(self) -> int:
        return len(self.subset)


# --------------------------------------------------------------------------- #
# footprint / obstacle helpers
# --------------------------------------------------------------------------- #
def _footprints(scene: DrawerScene) -> dict[str, Polygon]:
    return {n: st.footprint() for n, st in scene.items.items()}


def _obstacles(
    fps: dict[str, Polygon], present: set[str], ignore: str, wall: Polygon
) -> list[Polygon]:
    return [fps[n] for n in present if n != ignore] + [wall]


# --------------------------------------------------------------------------- #
# target blocker sets per grasp cell
# --------------------------------------------------------------------------- #
def target_open_grasp(
    scene: DrawerScene, fps: dict[str, Polygon] | None = None
) -> Grasp | None:
    """A collision-free grasp of the target in the *initial* scene, or None if blocked
    (the F1 test).

    Fingers must also clear the wall band.
    """
    fps = fps or _footprints(scene)
    tstate = scene.target_state()
    present = set(scene.item_names()) - {scene.target}
    obs = _obstacles(fps, present, scene.target, scene.wall_band)
    return has_grasp(tstate.shape, tstate.pose, obs)


def _blocker_sets(scene: DrawerScene, fps: dict[str, Polygon]) -> list[frozenset[str]]:
    """Per usable target grasp cell, the set of items whose footprints hit its fingers.

    Cells whose fingers hit the wall band are discarded (that grasp is impossible).
    """
    tstate = scene.target_state()
    blockers = [n for n in scene.item_names() if n != scene.target]
    sets: list[frozenset[str]] = []
    for g in grasp_cells(tstate.shape):
        lf, rf = finger_rects(g, tstate.pose)
        if (
            lf.intersection(scene.wall_band).area > 1e-9
            or rf.intersection(scene.wall_band).area > 1e-9
        ):
            continue  # this grasp direction is walled off regardless of item removal
        hit = frozenset(
            n
            for n in blockers
            if fps[n].intersection(lf).area > 1e-9
            or fps[n].intersection(rf).area > 1e-9
        )
        sets.append(
            hit
        )  # empty set => target already graspable this way (F1 handles it)
    return sets


def _minimal_sets(sets: list[frozenset[str]]) -> list[frozenset[str]]:
    nonempty = {s for s in sets if s}
    minimal: list[frozenset[str]] = []
    for s in sorted(nonempty, key=len):
        if not any(m <= s for m in minimal):
            minimal.append(s)
    return minimal


# --------------------------------------------------------------------------- #
# re-checks
# --------------------------------------------------------------------------- #
def _clears_target(
    scene: DrawerScene, fps: dict[str, Polygon], subset: frozenset[str]
) -> bool:
    tstate = scene.target_state()
    present = set(scene.item_names()) - subset - {scene.target}
    obs = _obstacles(fps, present, scene.target, scene.wall_band)
    return has_grasp(tstate.shape, tstate.pose, obs) is not None


def _extraction_order(
    scene: DrawerScene, fps: dict[str, Polygon], subset: frozenset[str]
):
    """Find an order to remove every member from the drawer, each having a clear grasp
    against all items not yet removed + the wall band.

    Memoised over the removed subset.
    """
    all_names = set(scene.item_names())
    members = tuple(sorted(subset))
    memo: dict[frozenset[str], list[str] | None] = {}

    def solve(removed: frozenset[str]) -> list[str] | None:
        if removed == subset:
            return []
        if removed in memo:
            return memo[removed]
        present = all_names - removed
        for x in members:
            if x in removed:
                continue
            obs = _obstacles(fps, present, x, scene.wall_band)
            if has_grasp(scene.items[x].shape, scene.items[x].pose, obs) is not None:
                rest = solve(removed | {x})
                if rest is not None:
                    memo[removed] = [x, *rest]
                    return memo[removed]
        memo[removed] = None
        return None

    return solve(frozenset())


# --------------------------------------------------------------------------- #
# public: full enumeration
# --------------------------------------------------------------------------- #
def enumerate_candidates(scene: DrawerScene, seed: int = 0) -> list[Candidate]:
    """The clearing candidates of ``scene`` in published order (spec Section 7)."""
    rng = random.Random((seed * 1_000_003 + 17) & 0xFFFFFFFF)
    fps = _footprints(scene)
    minimal = _minimal_sets(_blocker_sets(scene, fps))

    # supersets: each minimal set U one adjacent item, seeded order, until the cap
    candidates_sets: list[frozenset[str]] = list(minimal)
    seen = set(minimal)
    blockers = [n for n in scene.item_names() if n != scene.target]
    for base in minimal:
        neighbours = _adjacent(fps, base, blockers)
        rng.shuffle(neighbours)
        for nb in neighbours:
            if len(candidates_sets) >= MAX_CANDIDATES:
                break
            sup = base | {nb}
            if sup not in seen:
                seen.add(sup)
                candidates_sets.append(sup)
        if len(candidates_sets) >= MAX_CANDIDATES:
            break

    out: list[Candidate] = []
    for s in candidates_sets:
        if not _clears_target(scene, fps, s):
            continue  # (a): does not actually clear the target -> drop
        order = _extraction_order(scene, fps, s)
        out.append(
            Candidate(
                subset=s,
                members=order if order is not None else sorted(s),
                extractable=order is not None,
                extraction_reason=None if order is not None else "extraction",
            )
        )
    # published order: ascending |S|, ties by a seeded permutation
    tie = {c.subset: rng.random() for c in out}
    out.sort(key=lambda c: (c.size, tie[c.subset]))
    return out


def _adjacent(
    fps: dict[str, Polygon], base: frozenset[str], blockers: list[str]
) -> list[str]:
    base_geom = None
    for n in base:
        base_geom = fps[n] if base_geom is None else base_geom.union(fps[n])
    out = []
    for n in blockers:
        if n in base:
            continue
        if base_geom is not None and base_geom.distance(fps[n]) < ADJACENCY:
            out.append(n)
    return out
