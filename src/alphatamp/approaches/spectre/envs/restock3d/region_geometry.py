"""Region geometry for the **kinematic** Restock3D (single multi-section shelf).

A *region* is a single-object placement target on one **section** of the one shelf: section 0 is
the **tall** section (bottom, large clearance — a tall block fits) and section 1 is the **short**
section (top, small clearance — a tall block collides with the board capping it). Regions are NOT
PyBullet bodies and NOT objects in the low-level state — they are pure metadata (world centre,
footprint, the section's placement surface z, and the section's *cell clearance*, i.e. the vertical
gap to the board above). The abstractor (``InRegion``) reads footprint + surface z; the
region-parameterised place controller reads centre + surface z; feasibility itself is decided by
real PyBullet collision (a tall block collides with the short-section ceiling), not by this
metadata.

The shelf is a single body whose boards are placed at cumulative per-section gaps (:func:`section_surfaces`
and :func:`board_center_zs` are the shared geometry, consumed by :mod:`kinematic_env` when it builds
the boards and here when regions are laid out). Everything is **deterministic in the stratum** (no
per-seed jitter) so the models factory and the collection env agree without threading the problem id.
Names ``region_0_{i}`` (tall section) / ``region_1_{i}`` (short section) match the prior build so
``compare``/downstream code is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

# STRATA[stratum] = (n_small, n_tall, n_tall_regions, n_short_regions)
from .generator import STRATA

_TALL_SECTION, _SHORT_SECTION = 0, 1


@dataclass(frozen=True)
class RegionInfo:
    """A shelf region: world centre, footprint, the section it sits on, and its cell
    clearance."""

    name: str
    shelf: int  # section index: 0 = tall (bottom), 1 = short (top)
    center_xy: tuple[
        float, float
    ]  # world (x, y) of the region centre (object resting xy)
    half_xy: tuple[float, float]  # world-aligned footprint half-extents (x, y)
    cell_clearance: (
        float  # vertical gap above the section surface (F3 height reference)
    )
    surface_z: float  # world z of the region's shelf placement surface


def section_surfaces(config) -> list[tuple[float, float]]:
    """``[(surface_z, clearance), ...]`` for section 0 (tall, bottom) then 1 (short, top).

    Section 0's surface is ``bottom_surface_z``; section 1 sits one board-thickness above the tall
    section's ceiling board. Both are read from the env ``config`` (``bottom_surface_z``,
    ``section_clearances``, ``shelf_height``).
    """
    t = config.shelf_height
    surfaces: list[tuple[float, float]] = []
    surf = config.bottom_surface_z
    for clearance in config.section_clearances:
        surfaces.append((float(surf), float(clearance)))
        surf = surf + clearance + t  # next section's surface = this ceiling board's top
    return surfaces


def board_center_zs(config) -> list[float]:
    """World z of each board's centre: one board below every section surface, plus a top board
    capping the last (short) section — ``len(section_clearances) + 1`` boards total."""
    t = config.shelf_height
    surfaces = section_surfaces(config)
    centers = [surf - t / 2 for surf, _ in surfaces]
    last_surf, last_clear = surfaces[-1]
    centers.append(last_surf + last_clear + t / 2)  # ceiling board of the top section
    return [float(z) for z in centers]


def _row_xs(n: int, center_x: float, pitch: float) -> list[float]:
    """``n`` region centres evenly pitched about ``center_x`` (along the shelf width)."""
    span = (n - 1) * pitch
    x0 = center_x - span / 2
    return [round(x0 + i * pitch, 4) for i in range(n)]


def compute_region_infos(config, stratum: int) -> dict[str, RegionInfo]:
    """World geometry of every region for ``stratum``, from the env ``config``.

    Region centres are laid out along the shelf width (world x) at the shelf's front strip
    (world y), on each section's placement surface (world z). Section 0 (tall) and section 1
    (short) share the single shelf's (x, y); they differ in surface z and clearance.
    """
    _, _, n_tall_reg, n_short_reg = STRATA[stratum]
    surfaces = section_surfaces(config)
    sx, sy = config.shelf_pose.position[0], config.shelf_pose.position[1]
    front_y = sy - config.region_front_offset  # front strip faces -y (robot side)

    infos: dict[str, RegionInfo] = {}
    for section, n_reg in ((_TALL_SECTION, n_tall_reg), (_SHORT_SECTION, n_short_reg)):
        surf_z, clearance = surfaces[section]
        xs = _row_xs(n_reg, sx, config.region_pitch)
        for i, rx in enumerate(xs, start=1):
            infos[f"region_{section}_{i}"] = RegionInfo(
                name=f"region_{section}_{i}",
                shelf=section,
                center_xy=(float(rx), float(front_y)),
                half_xy=(config.region_half_x, config.region_half_y),
                cell_clearance=float(clearance),
                surface_z=float(surf_z),
            )
    return infos
