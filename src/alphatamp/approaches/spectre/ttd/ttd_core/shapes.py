"""Procedural concave shape library (TTD spec §4.2).

Footprints are "polar stars": vertex angles sampled with a minimum angular gap, radii
drawn uniformly, then 1–3 vertices pushed inward to force reflex vertices. Star-shaped
about the local origin by construction (so the origin is a valid convex-decomposition
kernel, and the origin is the placement reference point of :mod:`.geometry`). Each shape
is scaled to a target area and rejected unless it is simple, has >= 1 reflex vertex,
edges >= 1.0 cm, and >= 1 admissible antipodal grasp pair (§4.3.1).

Out-of-family real-footprint shapes (§4.2.1) and the val/test/out-of-generator splits
are deferred to later chunks; chunk 1 ships the procedural train + disjoint held-out
library only (§8.8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from . import geometry
from .geometry import GeometryError, Vertices
from .params import HELD_OUT_OFFSET, OP_A, OperatingPoint, TTDParams, default_params


@dataclass(frozen=True)
class ShapeGenParams:
    """Tunable knobs for :func:`generate_shape` (spec §4.2)."""

    n_range: tuple[int, int] = (8, 14)
    min_angular_gap_rad: float = 0.15
    radius_range: tuple[float, float] = (0.55, 1.0)
    n_push_range: tuple[int, int] = (1, 3)
    push_factor_range: tuple[float, float] = (0.35, 0.6)
    min_edge_cm: float = 1.0
    antipodal_tol_deg: float = 10.0
    aperture_range_cm: tuple[float, float] = (0.5, 14.0)
    angle_max_tries: int = 200


@dataclass(frozen=True)
class Shape:
    """A generated footprint with cached descriptors (spec §4.2)."""

    vertices: Vertices
    seed: int
    area_cm2: float
    perimeter_cm: float
    convexity_defect: float
    aspect_ratio: float

    def polygon(self) -> "geometry.Polygon":
        """Rebuild the shapely polygon (CCW, reference point at the local origin)."""
        return geometry.to_polygon(self.vertices)


@dataclass(frozen=True)
class ShapeLibrary:
    """A procedural shape library with a seed-disjoint held-out split (spec §8.8)."""

    train: list[Shape]
    held_out: list[Shape] = field(default_factory=list)


def _sample_angles(
    rng: np.random.Generator, n: int, min_gap: float, max_tries: int
) -> np.ndarray | None:
    """Sort ``n`` angles in [0, 2π) with min cyclic gap >= ``min_gap`` (or None)."""
    two_pi = 2.0 * np.pi
    for _ in range(max_tries):
        angles = np.sort(rng.uniform(0.0, two_pi, size=n))
        gaps = np.diff(angles)
        wrap = two_pi - (angles[-1] - angles[0])
        if gaps.size > 0 and float(gaps.min()) >= min_gap and wrap >= min_gap:
            return angles
    return None


def generate_shape(
    seed: int,
    target_area_cm2: float,
    gen: ShapeGenParams = ShapeGenParams(),
) -> Shape | None:
    """One seeded generation attempt; returns None if the shape is rejected (spec
    §4.2)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(gen.n_range[0], gen.n_range[1] + 1))
    angles = _sample_angles(rng, n, gen.min_angular_gap_rad, gen.angle_max_tries)
    if angles is None:
        return None
    radii = rng.uniform(gen.radius_range[0], gen.radius_range[1], size=n)
    n_push = int(rng.integers(gen.n_push_range[0], gen.n_push_range[1] + 1))
    push_idx = rng.choice(n, size=n_push, replace=False)
    radii[push_idx] *= rng.uniform(
        gen.push_factor_range[0], gen.push_factor_range[1], size=n_push
    )
    raw = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    try:
        poly = geometry.to_polygon(raw)
    except GeometryError:
        return None
    scale = float(np.sqrt(target_area_cm2 / poly.area))
    verts = geometry.normalize_ccw(raw * scale)
    if geometry.min_edge_length(verts) < gen.min_edge_cm:
        return None
    if geometry.count_reflex_vertices(verts) < 1:
        return None
    if not geometry.has_admissible_antipodal_pair(
        verts, tol_deg=gen.antipodal_tol_deg, d_range=gen.aperture_range_cm
    ):
        return None
    poly = geometry.to_polygon(verts)
    return Shape(
        vertices=verts,
        seed=seed,
        area_cm2=float(poly.area),
        perimeter_cm=float(poly.length),
        convexity_defect=geometry.convexity_defect(poly),
        aspect_ratio=geometry.aspect_ratio(poly),
    )


def generate_shape_retry(
    seed: int,
    target_area_cm2: float,
    *,
    gen: ShapeGenParams = ShapeGenParams(),
    max_tries: int = 64,
) -> Shape:
    """Retry generation with derived sub-seeds until a shape is accepted (spec §4.2).

    The returned shape records the identity ``seed`` (deterministically reproducible),
    not the accepted sub-seed. Raises :class:`RuntimeError` if all tries are rejected.
    """
    for child in np.random.SeedSequence(seed).spawn(max_tries):
        sub_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        shape = generate_shape(sub_seed, target_area_cm2, gen)
        if shape is not None:
            return replace(shape, seed=seed)
    raise RuntimeError(
        f"generate_shape_retry: no valid shape in {max_tries} tries (seed={seed})"
    )


def sample_p5_area(rng: np.random.Generator, params: TTDParams) -> float:
    """Draw a library-wide target area from the P5 band (spec §3)."""
    return float(rng.uniform(params.lib_area_lo_cm2, params.lib_area_hi_cm2))


def sample_p5b_area(rng: np.random.Generator, op: OperatingPoint) -> float:
    """Draw a candidate-member target area from the P5b band (spec §3)."""
    return float(rng.uniform(op.member_area_lo_cm2, op.member_area_hi_cm2))


def _generate_at_lib_seed(
    lib_seed: int,
    band: Literal["P5", "P5b"],
    op: OperatingPoint,
    params: TTDParams,
    gen: ShapeGenParams,
    max_tries: int,
) -> Shape:
    """Generate one library shape whose identity is ``lib_seed`` (area + shape from
    it)."""
    rng = np.random.default_rng(lib_seed)
    target = sample_p5_area(rng, params) if band == "P5" else sample_p5b_area(rng, op)
    return generate_shape_retry(lib_seed, target, gen=gen, max_tries=max_tries)


def build_library(
    *,
    n_train: int = 500,
    n_held_out: int = 100,
    base_seed: int = 0,
    band: Literal["P5", "P5b"] = "P5",
    op: OperatingPoint = OP_A,
    params: TTDParams | None = None,
    gen: ShapeGenParams = ShapeGenParams(),
    max_tries: int = 64,
) -> ShapeLibrary:
    """Build train + disjoint held-out shape sets (spec §4.2, §8.8).

    Held-out seeds come from a block offset by :data:`HELD_OUT_OFFSET`, guaranteeing the
    two seed sets — and hence the shapes — are disjoint.
    """
    p = params if params is not None else default_params(op)
    train = [
        _generate_at_lib_seed(base_seed + i, band, op, p, gen, max_tries)
        for i in range(n_train)
    ]
    held_out = [
        _generate_at_lib_seed(
            base_seed + HELD_OUT_OFFSET + i, band, op, p, gen, max_tries
        )
        for i in range(n_held_out)
    ]
    return ShapeLibrary(train=train, held_out=held_out)


def shape_to_json(shape: Shape) -> dict[str, object]:
    """Serialize a shape to vertex list + generating seed (spec §8.8)."""
    return {
        "vertices": [[float(x), float(y)] for x, y in shape.vertices],
        "seed": int(shape.seed),
    }


def shape_from_json(data: dict[str, object]) -> Shape:
    """Rebuild a shape from JSON, recomputing descriptors from the vertices."""
    verts = geometry.normalize_ccw(np.asarray(data["vertices"], dtype=np.float64))
    poly = geometry.to_polygon(verts)
    return Shape(
        vertices=verts,
        seed=int(data["seed"]),  # type: ignore[call-overload]
        area_cm2=float(poly.area),
        perimeter_cm=float(poly.length),
        convexity_defect=geometry.convexity_defect(poly),
        aspect_ratio=geometry.aspect_ratio(poly),
    )


def library_to_json(lib: ShapeLibrary) -> str:
    """Serialize a whole library to a JSON string (spec §4.2 published library)."""
    return json.dumps(
        {
            "train": [shape_to_json(s) for s in lib.train],
            "held_out": [shape_to_json(s) for s in lib.held_out],
        }
    )


def library_from_json(text: str) -> ShapeLibrary:
    """Rebuild a library from a JSON string produced by :func:`library_to_json`."""
    data = json.loads(text)
    return ShapeLibrary(
        train=[shape_from_json(s) for s in data["train"]],
        held_out=[shape_from_json(s) for s in data.get("held_out", [])],
    )
