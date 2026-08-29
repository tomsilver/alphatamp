"""Section-band geometry for **Restock3D v2** (continuous packing).

Where v1 lays out several *single-object* discrete regions per shelf section
(:mod:`region_geometry`), v2 treats each shelf **section as one wide continuous
placement band**: placement x is sampled uniformly across the band, y is the section's
front strip (a constant) plus a tiny jitter, and z is the section surface. There is no
per-object region and no object->region assignment; capacity/crowding is an *emergent*
continuous-packing constraint decided by real PyBullet collision (two objects with
overlapping x collide; a full section cannot fit another object).

To keep the low-level env (:mod:`kinematic_env`) untouched, a section is represented
with the existing :class:`region_geometry.RegionInfo` structure — a **2-entry** dict
``{section_0, section_1}`` whose ``half_xy[0]`` is the WIDE band half-width. The env's
``goal_reached`` / ``_name_in_any_region`` and the place controller's ``surface_z`` /
``center_xy`` reads then work verbatim on the two bands.

Names ``section_0`` (tall/bottom) / ``section_1`` (short/top) mirror the
``region_{section}_*`` section index. Everything is deterministic in the env config (no
per-seed jitter), matching v1.
"""

from __future__ import annotations

from .region_geometry import RegionInfo, section_surfaces

_TALL_SECTION, _SHORT_SECTION = 0, 1

#: Per-side buffer (m) from the shelf board's x-edge to the outermost object CENTER. Object half_x is
#: 0.025 and the front grasp needs a little lateral clearance for the fingers straddling the +/-x faces,
#: so ~0.04 keeps the object fully on the board with a small margin. The band is intentionally wide
#: (the user wants as much lateral width as possible); uniform sampling + real-collision resampling
#: self-corrects an over-generous end. Physical maximum (no overhang) would be object half_x = 0.025.
_X_BAND_END_MARGIN = 0.04

#: y jitter (m) added to the section's front strip. Tiny on purpose: placing DEEPER (larger y) into the
#: shelf is harder and we already know the front strip places reliably (it is v1's region y), so v2
#: reuses that y and only jitters it slightly for genuine backtracking retries.
Y_JITTER = 0.01


def band_half_x(config) -> float:
    """Half-width of a section's continuous x-band about the shelf centre (world x)."""
    return config.shelf_width / 2 - _X_BAND_END_MARGIN


def section_x_band(config) -> tuple[float, float]:
    """``(x_lo, x_hi)`` legal object-CENTER x-range on a section (world x)."""
    sx = config.shelf_pose.position[0]
    half = band_half_x(config)
    return (sx - half, sx + half)


def compute_section_infos(config) -> dict[str, RegionInfo]:
    """The two section bands (tall bottom, short top) as wide :class:`RegionInfo`s.

    ``center_xy`` is the shelf centre x at the front strip y; ``half_xy[0]`` is the WIDE
    band half-width (the continuous-x sampling range); ``surface_z`` /
    ``cell_clearance`` come from :func:`region_geometry.section_surfaces`.
    """
    surfaces = section_surfaces(config)
    sx = config.shelf_pose.position[0]
    front_y = config.shelf_pose.position[1] - config.region_front_offset
    half_xy = (band_half_x(config), config.region_half_y)
    infos: dict[str, RegionInfo] = {}
    for section in (_TALL_SECTION, _SHORT_SECTION):
        surf_z, clearance = surfaces[section]
        infos[f"section_{section}"] = RegionInfo(
            name=f"section_{section}",
            shelf=section,
            center_xy=(float(sx), float(front_y)),
            half_xy=half_xy,
            cell_clearance=float(clearance),
            surface_z=float(surf_z),
        )
    return infos
