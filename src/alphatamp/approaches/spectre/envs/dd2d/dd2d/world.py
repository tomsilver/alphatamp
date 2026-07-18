"""DD2D world model -- drawer / wall-band / buffer geometry + mutable occupancy.

The resting-pose state space is 2D (spec Section 5.1's height-stratified SE(2)xlift
abstraction: transfer is collision-free by the height argument, so all continuous
difficulty lives in grasps and buffer placements). :class:`DrawerScene` is the static
scene; :class:`DrawerWorld` is the mutable occupancy a staging skeleton is replayed
against (pick from the drawer -> hand -> place on the buffer).

The buffer strip has **no walls** (spec Section 5.2): only already-staged items obstruct
placements and finger approaches there; fingers may overhang the painted boundary. The
drawer's 1.5 cm **wall band** is a first-class obstacle for grasp fingers (encoded here
as an ordinary polygon, matching the spec's ``wall`` pseudo-object, Section 6.1 / M6).

:func:`sample_buffer_pose` is the compaction-biased buffer placement sampler (spec
Section 6.3): deliberately incomplete and cheap, so on an infeasible subset it can only
fail-to-find, consuming budget -- that expensive failure is the cost structure under
study, not a bug. It is shared by the refiner (real staging) and the labeler (the
positive accessible-packing certificate), with an ``inflate`` margin for the latter.

Units: centimetres.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from shapely import Polygon
from shapely.affinity import rotate, translate

from .shapes import Shape

_EPS = 1e-9
_CONTAIN_TOL = 1e-7


# --------------------------------------------------------------------------- #
# pose transform + scene containers
# --------------------------------------------------------------------------- #
def place_polygon(poly: Polygon, pose: tuple[float, float, float]) -> Polygon:
    """Place an item-frame polygon at world ``pose = (x, y, theta)`` (rotate about the
    origin/centroid, then translate)."""
    x, y, theta = pose
    return translate(rotate(poly, theta, origin=(0, 0), use_radians=True), x, y)


@dataclass
class ItemState:
    """One item's identity + current placement (pose + region)."""

    name: str
    shape: Shape
    pose: tuple[float, float, float]  # (x, y, theta)
    region: str  # "drawer" | "buffer" | "hand" | "removed"
    is_target: bool = False

    def footprint(self) -> Polygon:
        return place_polygon(self.shape.polygon, self.pose)


@dataclass
class DrawerScene:
    """Static top-down drawer scene (drawer interior, wall band, buffer strip,
    items)."""

    drawer: Polygon
    wall_band: Polygon
    buffer: Polygon
    items: dict[str, ItemState]  # initial states (item -> pose/region)
    target: str
    margin: float  # label margin delta (spec P12)
    dims: dict = field(default_factory=dict)  # W, D, buffer L/d, lambda (provenance)

    def item_names(self) -> list[str]:
        return list(self.items)

    def blockers(self) -> list[str]:
        return [n for n in self.items if n != self.target]

    def target_state(self) -> ItemState:
        return self.items[self.target]


@dataclass
class StreamCounter:
    """Lightweight stream-call / EGE accounting (spec Section 2); used for demo
    stats."""

    calls: int = 0
    eges: int = 0

    def sample(self, n_eges: int = 1) -> None:
        self.calls += 1
        self.eges += n_eges

    def test(self, n_eges: int = 1) -> None:
        self.calls += 1
        self.eges += n_eges


# --------------------------------------------------------------------------- #
# collision helpers (module-level so the labeler can reuse them without a world)
# --------------------------------------------------------------------------- #
def collision_free(poly: Polygon, obstacles: list[Polygon]) -> bool:
    """Is ``poly`` free of every obstacle (boundary touching allowed, area 0)?"""
    for obs in obstacles:
        if poly.intersection(obs).area > _EPS:
            return False
    return True


def contained(poly: Polygon, region: Polygon) -> bool:
    return bool(region.buffer(_CONTAIN_TOL).covers(poly))


# --------------------------------------------------------------------------- #
# compaction-biased buffer-pose sampler (spec Section 6.3)
# --------------------------------------------------------------------------- #
def _gumbel(rng: random.Random, beta: float) -> float:
    u = min(max(rng.random(), 1e-12), 1 - 1e-12)
    return -beta * math.log(-math.log(u))


def sample_buffer_pose(
    shape: Shape,
    buffer_poly: Polygon,
    staged_polys: list[Polygon],
    rng: random.Random,
    m_p: int = 15,
    step: float = 0.15,
    inflate: float = 0.0,
    beta: float = 0.3,
) -> tuple[float, float, float] | None:
    """Sample a compaction-biased buffer pose for ``shape`` among ``staged_polys``.

    Draws ``m_p`` candidates (orientation on a 15 deg grid + jitter), pushes each toward
    the bottom-left contact (the compaction bias), keeps the collision-free ones
    (footprint contained in the buffer and clear of staged items, both grown by
    ``inflate``), and returns the lowest-``x + 0.01*y`` candidate with Gumbel tie-noise.
    ``inflate = delta/2`` yields a sound >= delta-clearance packing for the labeler;
    ``inflate = 0`` is the refiner's real placement. Returns ``None`` if none fit.
    """
    bx0, by0, bx1, by1 = buffer_poly.bounds
    staged_bounds = [s.bounds for s in staged_polys]

    def valid(pose: tuple[float, float, float]) -> bool:
        fp = place_polygon(shape.polygon, pose)
        if inflate > 0:
            fp = fp.buffer(inflate)
        return contained(fp, buffer_poly) and collision_free(fp, staged_polys)

    best: tuple[float, tuple[float, float, float]] | None = None
    for _ in range(m_p):
        grid = math.radians(15.0) * rng.randrange(0, 24)
        theta = grid + math.radians(rng.uniform(-7.5, 7.5))
        fx0, fy0, fx1, fy1 = place_polygon(shape.polygon, (0.0, 0.0, theta)).bounds
        hx, hy = (fx1 - fx0) / 2.0, (fy1 - fy0) / 2.0
        pad = inflate
        xlo, xhi = bx0 - fx0 + pad, bx1 - fx1 - pad
        ylo, yhi = by0 - fy0 + pad, by1 - fy1 - pad
        if xlo > xhi or ylo > yhi:
            continue  # item cannot fit the empty buffer at this orientation

        # contact proposal (0.7): abut a staged item or a buffer edge, then compact;
        # slide proposal (0.3): a uniform free pose. (spec Section 6.3)
        if staged_bounds and rng.random() < 0.7:
            rb = staged_bounds[rng.randrange(len(staged_bounds))]
            if rng.random() < 0.5:  # to the right of the reference, bottoms aligned
                cx, cy = rb[2] + hx + step, rb[1] - fy0
            else:  # above the reference, left edges aligned
                cx, cy = rb[0] - fx0, rb[3] + hy + step
        else:
            cx, cy = xlo, rng.uniform(ylo, yhi)  # seed a column against the left wall
        cx = min(max(cx, xlo), xhi)
        cy = min(max(cy, ylo), yhi)
        if not valid((cx, cy, theta)):
            cx, cy = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)  # uniform fallback
            if not valid((cx, cy, theta)):
                continue
        cx, cy = _push_bottom_left(valid, cx, cy, theta, step, xlo, ylo)
        score = cx + 0.01 * cy + _gumbel(rng, beta)
        if best is None or score < best[0]:
            best = (score, (cx, cy, theta))
    return best[1] if best is not None else None


def _push_bottom_left(
    valid, cx: float, cy: float, theta: float, step: float, xlo: float, ylo: float
):
    """Greedily slide toward -x then -y while still valid (the compaction, spec 6.3)."""
    for _ in range(400):
        moved = False
        for dx, dy in ((-step, 0.0), (0.0, -step)):
            nx, ny = cx + dx, cy + dy
            if nx >= xlo - step and ny >= ylo - step and valid((nx, ny, theta)):
                cx, cy, moved = nx, ny, True
        if not moved:
            break
    return cx, cy


def settle_pose(
    shape: Shape,
    region: Polygon,
    obstacles: list[Polygon],
    rng: random.Random,
    max_tries: int = 30,
    backoff: float = 0.2,
) -> tuple[float, float, float] | None:
    """Settled-clutter placement (spec Section 9.1): uniform proposal -> translate
    toward the nearest contact along a random direction -> back off ``backoff`` ->
    reject on overlap.

    Returns a pose or ``None`` after ``max_tries``.
    """
    bx0, by0, bx1, by1 = region.bounds
    for _ in range(max_tries):
        theta = rng.uniform(0, 2 * math.pi)
        fp0 = place_polygon(shape.polygon, (0.0, 0.0, theta))
        fx0, fy0, fx1, fy1 = fp0.bounds
        xlo, xhi = bx0 - fx0, bx1 - fx1
        ylo, yhi = by0 - fy0, by1 - fy1
        if xlo > xhi or ylo > yhi:
            continue
        cx, cy = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
        pose = (cx, cy, theta)
        fp = place_polygon(shape.polygon, pose)
        if not contained(fp, region) or not collision_free(fp, obstacles):
            continue
        # translate toward the nearest contact along a random direction, back off
        ang = rng.uniform(0, 2 * math.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        cx, cy = _slide_to_contact(
            shape, region, obstacles, cx, cy, theta, dx, dy, backoff
        )
        return (cx, cy, theta)
    return None


def collar_pose(
    shape: Shape,
    region: Polygon,
    obstacles: list[Polygon],
    target_center: tuple[float, float],
    bearing: float,
    rng: random.Random,
    backoff: float = 0.15,
    max_tries: int = 8,
) -> tuple[float, float, float] | None:
    """Place a "collar" item hugging the target from direction ``bearing`` (a difficulty
    prior; see docs/dd2d.md "Requiring a blocking subset").

    Unlike :func:`settle_pose`
    (which slides toward a *random* contact, so a lone neighbour ends up alone in a grasp
    corridor), this proposes a start centre *outward* from ``target_center`` along
    ``bearing`` and slides **inward** (toward the target) until contact, so opposing
    collar items bracket the target's grasp corridors → clearing needs a diametric pair.
    Returns a pose or ``None`` if it cannot be placed clear of walls/other items.
    """
    tx, ty = target_center
    ux, uy = math.cos(bearing), math.sin(bearing)
    rmax = shape.r_max
    for _ in range(max_tries):
        theta = rng.uniform(0, 2 * math.pi)
        # start far enough out along the bearing to clear the target, then slide inward
        reach = rng.uniform(0.6, 1.0) * max(
            region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1]
        )
        cx, cy = tx + ux * (reach), ty + uy * (reach)
        # pull the start point back inside the region envelope so sliding has room
        cx = min(max(cx, region.bounds[0] + rmax), region.bounds[2] - rmax)
        cy = min(max(cy, region.bounds[1] + rmax), region.bounds[3] - rmax)
        cx, cy = _slide_to_contact(
            shape, region, obstacles, cx, cy, theta, -ux, -uy, backoff
        )
        fp = place_polygon(shape.polygon, (cx, cy, theta))
        if contained(fp, region) and collision_free(fp, obstacles):
            return (cx, cy, theta)
    return None


def _slide_to_contact(
    shape, region, obstacles, cx, cy, theta, dx, dy, backoff, step=0.15
):
    last = (cx, cy)
    for _ in range(200):
        nx, ny = cx + dx * step, cy + dy * step
        fp = place_polygon(shape.polygon, (nx, ny, theta))
        if contained(fp, region) and collision_free(fp, obstacles):
            last = (cx, cy) = (nx, ny)
        else:
            break
    # back off along the travel direction
    return (last[0] - dx * backoff, last[1] - dy * backoff)


# --------------------------------------------------------------------------- #
# mutable occupancy replayed against by the refiner
# --------------------------------------------------------------------------- #
class DrawerWorld:
    """Mutable per-item occupancy for replaying a staging skeleton.

    ``pick`` moves a drawer item into the hand; ``place_buffer`` drops the held item on
    the buffer at a bound pose; ``extract`` removes the target. Collision queries are
    region-local: the drawer counts remaining drawer items + the wall band; the buffer
    counts only staged buffer items (no walls, spec Section 5.2).
    """

    def __init__(
        self, scene: DrawerScene, counter: StreamCounter | None = None
    ) -> None:
        self.scene = scene
        self.wall_band = scene.wall_band
        self.buffer_poly = scene.buffer
        self.states: dict[str, ItemState] = {
            n: ItemState(s.name, s.shape, s.pose, s.region, s.is_target)
            for n, s in scene.items.items()
        }
        self.held: str | None = None
        self.counter = counter or StreamCounter()

    # -- queries -------------------------------------------------------------
    def footprint(self, name: str) -> Polygon:
        return self.states[name].footprint()

    def region_items(self, region: str, ignore: str | None = None) -> list[str]:
        return [n for n, s in self.states.items() if s.region == region and n != ignore]

    def drawer_obstacles(self, ignore: str | None = None) -> list[Polygon]:
        obs = [self.footprint(n) for n in self.region_items("drawer", ignore)]
        obs.append(self.wall_band)
        return obs

    def buffer_obstacles(self, ignore: str | None = None) -> list[Polygon]:
        return [self.footprint(n) for n in self.region_items("buffer", ignore)]

    # -- mutations -----------------------------------------------------------
    def pick(self, name: str) -> bool:
        st = self.states.get(name)
        if self.held is not None or st is None or st.region != "drawer":
            return False
        st.region = "hand"
        self.held = name
        return True

    def place_buffer(self, name: str, pose: tuple[float, float, float]) -> bool:
        if self.held != name:
            return False
        st = self.states[name]
        st.pose = pose
        st.region = "buffer"
        self.held = None
        return True

    def extract(self, name: str) -> bool:
        st = self.states.get(name)
        if self.held is not None or st is None or st.region != "drawer":
            return False
        st.region = "removed"
        return True

    # -- snapshot / restore (for backjumping) --------------------------------
    def snapshot(self) -> dict:
        return {
            "states": {n: (s.pose, s.region) for n, s in self.states.items()},
            "held": self.held,
        }

    def restore(self, snap: dict) -> None:
        for n, (pose, region) in snap["states"].items():
            self.states[n].pose = pose
            self.states[n].region = region
        self.held = snap["held"]
