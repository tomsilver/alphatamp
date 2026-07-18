"""Op-level cost calibration (TTD spec §5.3).

A stream call is not a cost unit (costs span ~272x across kinds); costs are instead the
dot product of the :class:`~...ttd_core.counters.OpCounter` C-op counts with a per-op
microsecond table measured here on pinned library versions and reference hardware. The
§10.0 pre-pilot accumulates op counts while measuring the η landscape, then multiplies
by this table to report the calibrated geometric cost and the labeling-cost estimate.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Callable

from shapely.geometry import box
from shapely.ops import unary_union

from ..ttd_core import geometry, shapes
from ..ttd_core.counters import CKind, OpCounter

_CKINDS: tuple[CKind, ...] = (
    "poly_construct",
    "buffer",
    "nfp",
    "ifp",
    "union",
    "minkowski",
)


@dataclass(frozen=True)
class CalibrationTable:
    """Per-op-kind microsecond costs measured on reference hardware (spec §5.3)."""

    us_per_op: dict[str, float]
    hardware: str
    n_iters: int

    def cost_us(self, counter: OpCounter) -> float:
        """Calibrated geometric cost (µs) of an accumulated op counter."""
        return sum(
            self.us_per_op.get(kind, 0.0) * n for kind, n in counter.c_ops.items()
        )


def _time_us(fn: Callable[[], object], n_iters: int) -> float:
    """Mean microseconds per call of ``fn`` over ``n_iters`` (after warmup)."""
    for _ in range(3):
        fn()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - start) / n_iters * 1e6


def calibrate(
    *, n_iters: int = 100, seed: int = 0, hardware: str | None = None
) -> CalibrationTable:
    """Measure µs/op for each geometry primitive (spec §5.3).

    Uses representative member shapes so the NFP/IFP costs reflect the concave family
    the benchmark actually nests; timings are hardware- and library-version-specific, so
    pin both when publishing.
    """
    hw = hardware if hardware is not None else platform.platform()
    poly_a = shapes.generate_shape_retry(seed, 40.0).polygon()
    poly_b = shapes.generate_shape_retry(seed + 1, 40.0).polygon()
    verts_a = geometry.to_vertices(poly_a)
    parts = geometry.convex_decompose(poly_a)
    part_a = parts[0]
    part_b = parts[1] if len(parts) > 1 else parts[0]
    container = box(0.0, 0.0, 26.0, 18.0)
    shifted_b = geometry.snap(poly_b)  # cheap no-op to keep a distinct object

    us: dict[str, float] = {
        "poly_construct": _time_us(lambda: geometry.to_polygon(verts_a), n_iters),
        "buffer": _time_us(lambda: geometry.inflate(poly_a, 0.9), n_iters),
        "minkowski": _time_us(
            lambda: geometry.minkowski_sum_convex(part_a, part_b), n_iters
        ),
        "ifp": _time_us(lambda: geometry.ifp(poly_a, container), n_iters),
        "nfp": _time_us(lambda: geometry.nfp(poly_a, poly_b), n_iters),
        "union": _time_us(lambda: unary_union([poly_a, shifted_b]), n_iters),
    }
    return CalibrationTable(us_per_op=us, hardware=hw, n_iters=n_iters)
