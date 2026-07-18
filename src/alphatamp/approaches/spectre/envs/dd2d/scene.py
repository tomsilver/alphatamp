"""Backend-agnostic geometric scene description for a sorting problem.

This is the continuous-world twin of the symbolic problem: it holds object poses
and sizes so a geometry backend (PyBullet or the numpy-2D fallback) can attempt
refinement and rendering. Object sizes mirror
``policy-guided-lazy-tamp/.../make_problem.py`` (the LAZY sorting generator):
4.5 cm cube blocks and taller (10 cm) blockers.

LAYOUT NOTE: the LAZY paper places the four colored tables on a 1.2 m cross. A
single *fixed* 7-DoF Panda cannot do clean top-down grasps across a cross that
wide (especially the table directly behind it), so we use a **robot-friendly
front-facing 2x2 layout** within the arm's reliable top-down workspace
(calibration: 36/36 poses reachable, tilt < 20 deg). The sorting *semantics* are
unchanged: 4 color tables, two start tables (near row) + two destinations (far
row), blocks sorted to matching colors, taller blockers obstruct. See
decisions.md ("front-facing 2x2 layout").
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- world constants --------------------------------------------------------
ARM_POS: tuple[float, float] = (0.0, 0.0)  # Panda base at world origin, facing +x
ARM_REACH: float = 0.7  # metres; used only by the analytic refiner
BLOCK_SIZE: tuple[float, float, float] = (0.045, 0.045, 0.045)
BLOCKER_SIZE: tuple[float, float, float] = (0.045, 0.045, 0.10)  # taller -> obstructs
TABLE_HALF_EXTENT: float = 0.12  # half-width of a (square) table top

# table name -> centre (x, y). Front-facing 2x2: starts on the near row (x=0.36),
# destinations on the far row (x=0.60); columns at y=+/-0.24. All within the
# Panda's clean top-down workspace (see calibration / LAYOUT NOTE above).
TABLE_CENTERS: dict[str, tuple[float, float]] = {
    "blue_table": (0.36, 0.24),  # start (near)
    "purple_table": (0.36, -0.24),  # start (near)
    "red_table": (0.60, 0.24),  # destination (far)
    "green_table": (0.60, -0.24),  # destination (far)
}
START_TABLES: tuple[str, str] = ("purple_table", "blue_table")
# Tables stacking builds its goal towers on (the far row). Sorting uses these as
# its colour destinations; stacking reuses them purely as clear build surfaces so
# towers rise away from the cluttered near row the blocks start on.
DEST_TABLES: tuple[str, str] = ("red_table", "green_table")


@dataclass
class SceneObject:
    """A movable block or blocker with a current 3-D pose."""

    name: str
    color: str  # "red" | "green" | "blocker"
    is_blocker: bool
    pose: tuple[float, float, float]  # (x, y, z) world position of the object centre
    size: tuple[float, float, float]  # full extents (x, y, z)

    @property
    def footprint_radius(self) -> float:
        return 0.5 * max(self.size[0], self.size[1])


@dataclass
class Table:
    name: str
    center: tuple[float, float]
    half_extent: float = TABLE_HALF_EXTENT


@dataclass
class GeometricScene:
    """All movable objects + tables.

    Consumed by the geometry backends.
    """

    objects: list[SceneObject] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    arm_pos: tuple[float, float] = ARM_POS
    arm_reach: float = ARM_REACH

    def by_name(self, name: str) -> SceneObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def table_in_reach(self, name: str) -> bool:
        cx, cy = self.table(name).center
        ax, ay = self.arm_pos
        return ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 <= self.arm_reach + self.table(
            name
        ).half_extent
