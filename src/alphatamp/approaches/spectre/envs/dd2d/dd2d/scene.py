"""DD2D forward scene generator (spec Section 9.1).

Naturalistic, *not* engineered: sample drawer/buffer/fill, drop a target near the
centre, then fill with settled clutter (items lean into each other and the walls). No
step steers blockers toward the target beyond density itself -- the decision-relevance
filters (target actually blocked, a real choice of clearing subsets, a solvable
certificate) live in :mod:`blocks_tamp.dd2d.problem`, applied only after labeling.

Units: centimetres.
"""

from __future__ import annotations

import random

from shapely import box as shp_box

from .shapes import sample_shape
from .world import DrawerScene, ItemState, collar_pose, place_polygon, settle_pose

# small, round, well-packing family used for the DEFAULT "collar" crowding prior: a tight
# ring of small cans hugs the target with few angular gaps, and they stay graspable +
# packable so the forced clearing pair is feasible (larger/elongated shapes leave big gaps;
# bowl is now the large circle, so only can is used here). generate_scene(diverse_crowd=True)
# overrides this and draws collar items from every family instead.
_COLLAR_FAMILIES = ("can",)
# when crowding, the target itself is a compact round shape so a small collar can fully
# ring it (an elongated target keeps graspable slots at its tips -> single removal suffices).
_COMPACT_TARGET_FAMILIES = ("can",)

# spec P1/P3/P5/P6
DRAWER_W = (35.0, 50.0)
DRAWER_D = (28.0, 40.0)
BUFFER_L = (25.0, 45.0)
BUFFER_D = (12.0, 20.0)
WALL_BAND = 1.5  # spec P2
BUFFER_GAP = 6.0  # buffer sits 6 cm to the drawer's right (spec Section 5.2)
FILL_RANGE = (0.35, 0.55)
N_RANGE = (9, 14)  # items including the target (spec P5)
BOX_MAX_FRAC = 0.45  # a large box rejected if bbox area > 45% of drawer short-side^2 (spec Section 4)


def _drawer_geometry(rng: random.Random, lam: float):
    W = rng.uniform(*DRAWER_W)
    D = rng.uniform(*DRAWER_D)
    drawer = shp_box(0.0, 0.0, W, D)
    outer = shp_box(-WALL_BAND, -WALL_BAND, W + WALL_BAND, D + WALL_BAND)
    wall_band = outer.difference(drawer)
    L = rng.uniform(*BUFFER_L) * lam
    d = rng.uniform(*BUFFER_D) * lam
    bx0 = W + BUFFER_GAP
    buffer = shp_box(bx0, 0.0, bx0 + L, d)
    dims = {"W": W, "D": D, "buffer_L": L, "buffer_d": d, "lambda": lam}
    return drawer, wall_band, buffer, dims


def _acceptable(shape, W: float, D: float) -> bool:
    """Reject a large box that is too big for the drawer (spec Section 4)."""
    if shape.family != "box":
        return True
    short = min(W, D)
    return shape.size[0] * shape.size[1] <= BOX_MAX_FRAC * short * short


def generate_scene(
    seed: int,
    lam: float = 1.0,
    split: str = "train",
    fill: float | None = None,
    n_items: int | None = None,
    crowd: int = 0,
    diverse_crowd: bool = False,
    fill_max: float | None = None,
    extra_families: dict[str, float] | None = None,
    require_families: tuple[str, ...] = (),
) -> DrawerScene:
    """Sample one drawer scene (target + settled clutter).

    ``crowd`` (default 0 = naturalistic baseline) is a disclosed **difficulty prior**: it
    places ``crowd`` "collar" items hugging the target from evenly-spaced bearings so that
    opposing items bracket the target's grasp corridors — clearing then requires removing a
    diametric **pair** (a 2+ subset), not a single object. Without it, `settle_pose` drops
    each neighbour from a random bearing so a lone item usually sits alone in one corridor
    and a single removal suffices (see notebook.md 2026-07-05 tuning; docs/dd2d.md).

    ``diverse_crowd`` (default False) draws collar items from the **full weighted family
    distribution** instead of only the round ``_COLLAR_FAMILIES`` — so concave shapes can
    participate in the pincer rather than only landing in the outer clutter. Expect a looser
    ring (non-round items leave larger angular gaps / fail placement more often), so the
    measured subset-required rate typically drops; ``require_subset`` restores it by
    resampling.

    The last three arguments serve the held-out generalization sets (docs/decisions
    2026-08-01) and are inert at their defaults:

    - ``fill_max`` raises the upper bound of the sampled coverage cap (default ``FILL_RANGE``
      max), so denser scenes with more items actually place instead of stopping at the cap.
    - ``extra_families`` augments the clutter/collar sampling pool with additional families
      (e.g. the held-out ``tee``/``cross``); the target pool is left unchanged.
    - ``require_families`` forces >= 1 item of each named family into the scene (best-effort
      placement; the problem generator rejects any scene that still lacks one).
    """
    rng = random.Random((seed * 2_654_435_761 + 0x9E37) & 0xFFFFFFFF)
    drawer, wall_band, buffer, dims = _drawer_geometry(rng, lam)
    W, D = dims["W"], dims["D"]
    _fill_hi = FILL_RANGE[1] if fill_max is None else fill_max
    fill = rng.uniform(FILL_RANGE[0], _fill_hi) if fill is None else fill
    n_items = rng.randint(*N_RANGE) if n_items is None else n_items
    drawer_area = drawer.area

    items: dict[str, ItemState] = {}

    # target: any rotation, pose uniform over the central 50% x 50% (spec P17). When
    # crowding, bias it to a compact round shape so the collar can fully ring it.
    target_fam = rng.choice(_COMPACT_TARGET_FAMILIES) if crowd > 0 else None
    for _ in range(200):
        shp = sample_shape(rng, family=target_fam, split=split)
        if not _acceptable(shp, W, D):
            continue
        theta = rng.uniform(0, 6.283185307179586)
        cx = rng.uniform(0.25 * W, 0.75 * W)
        cy = rng.uniform(0.25 * D, 0.75 * D)
        fp = place_polygon(shp.polygon, (cx, cy, theta))
        if drawer.buffer(1e-7).covers(fp):
            items["target"] = ItemState(
                "target", shp, (cx, cy, theta), "drawer", is_target=True
            )
            break
    if "target" not in items:  # pragma: no cover - defensive
        raise RuntimeError(f"could not place a target for seed {seed}")

    coverage = items["target"].footprint().area / drawer_area
    idx = 0

    # collar: pincer the target from evenly-spaced bearings (a disclosed difficulty prior).
    # Round _COLLAR_FAMILIES give the tightest ring; diverse_crowd instead draws from every
    # family (family=None) so concave shapes participate in the pincer, not just the clutter.
    if crowd > 0:
        tcx, tcy = items["target"].footprint().centroid.coords[0]
        base = rng.uniform(0, 6.283185307179586)
        for i in range(crowd):
            bearing = (
                base + 6.283185307179586 * i / crowd + rng.uniform(-0.12, 0.12) / crowd
            )
            collar_fam = None if diverse_crowd else rng.choice(_COLLAR_FAMILIES)
            shp = sample_shape(
                rng, family=collar_fam, split=split, extra_weights=extra_families
            )
            if not _acceptable(
                shp, W, D
            ):  # no-op for round cans; guards a giant box collar item
                continue
            obstacles = [st.footprint() for st in items.values()] + [wall_band]
            pose = collar_pose(
                shp, drawer, obstacles, (tcx, tcy), bearing, rng, backoff=0.06
            )
            if pose is None:
                continue
            fp = place_polygon(shp.polygon, pose)
            if not drawer.buffer(1e-7).covers(fp) or any(
                fp.intersection(o).area > 1e-9 for o in obstacles
            ):
                continue
            name = f"o{idx}"
            idx += 1
            items[name] = ItemState(name, shp, pose, "drawer", is_target=False)
            coverage += fp.area / drawer_area

    # forced held-out families: guarantee >= 1 item of each named family, placed like
    # clutter but not gated on the fill cap (they are required). Best-effort: if none of the
    # placement tries lands, the scene simply lacks the family and generate_dd2d_problem
    # rejects and resamples it -- the same pattern require_subset uses.
    for req_fam in require_families:
        for _ in range(200):
            shp = sample_shape(rng, family=req_fam, split=split)
            if not _acceptable(shp, W, D):
                continue
            obstacles = [st.footprint() for st in items.values()] + [wall_band]
            pose = settle_pose(shp, drawer, obstacles, rng)
            if pose is None:
                continue
            fp = place_polygon(shp.polygon, pose)
            if not drawer.buffer(1e-7).covers(fp) or any(
                fp.intersection(o).area > 1e-9 for o in obstacles
            ):
                continue
            name = f"o{idx}"
            idx += 1
            items[name] = ItemState(name, shp, pose, "drawer", is_target=False)
            coverage += fp.area / drawer_area
            break

    tries = 0
    while len(items) < n_items and coverage < fill and tries < 400:
        tries += 1
        shp = sample_shape(rng, split=split, extra_weights=extra_families)
        if not _acceptable(shp, W, D):
            continue
        obstacles = [st.footprint() for st in items.values()] + [wall_band]
        pose = settle_pose(shp, drawer, obstacles, rng)
        if pose is None:
            continue
        fp = place_polygon(shp.polygon, pose)
        # settle_pose backs off after contact; re-verify it is legal (contained + free)
        if not drawer.buffer(1e-7).covers(fp) or any(
            fp.intersection(o).area > 1e-9 for o in obstacles
        ):
            continue
        name = f"o{idx}"
        idx += 1
        items[name] = ItemState(name, shp, pose, "drawer", is_target=False)
        coverage += fp.area / drawer_area

    margin = 1.0  # label margin delta (spec P12), overridable by the problem generator
    return DrawerScene(
        drawer=drawer,
        wall_band=wall_band,
        buffer=buffer,
        items=items,
        target="target",
        margin=margin,
        dims=dims,
    )
