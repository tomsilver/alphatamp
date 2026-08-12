"""Sorting problem generation.

Mirrors ``make_sorting_problem`` in
``policy-guided-lazy-tamp/experiments/blocks_world/data_generation/make_problem.py``:
half the blocks are red and half green, all blocks + blockers start scattered on
the two *start* tables (purple, blue), and the goal sends red blocks to the red
table, green blocks to the green table, and each blocker back to the start table
it began on. Blockers are taller and may obstruct grasps/placements (a geometric
fact resolved in ``refine.py``, not in the symbolic plan).

A :class:`SortingProblem` carries both the symbolic problem (objects, init/goal
literals, PDDL text) and the :class:`GeometricScene` used for refinement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .scene import (
    BLOCK_SIZE,
    BLOCKER_SIZE,
    START_TABLES,
    TABLE_CENTERS,
    GeometricScene,
    SceneObject,
    Table,
)

Literal = tuple  # e.g. ("on-table", "red_block0", "red_table")

# table name -> the colour of block that belongs there
COLOR_TABLE = {"red": "red_table", "green": "green_table"}


@dataclass
class ObjectInfo:
    """Per-object metadata recorded in the PIGINet example."""

    name: str
    category: str  # "block" | "blocker"
    color: str  # "red" | "green" | "blocker"
    size: tuple[float, float, float]
    is_blocker: bool
    start_table: str


@dataclass
class BlocksWorldProblem:
    """A blocks-world TAMP instance (sorting, stacking, ...).

    Carries both the symbolic problem (objects, init/goal literals, PDDL text) and
    the :class:`GeometricScene` used for refinement. ``problem_type`` records which
    generator produced it; the PDDL ``(:domain blocksworld)`` is shared by all types.
    """

    problem_id: str
    objects: list[ObjectInfo]
    tables: list[str]
    init_facts: list[Literal]
    goal_facts: list[Literal]
    scene: GeometricScene
    seed: int
    num_blocks: int
    num_blockers: int
    problem_type: str = "sorting"

    # -- PDDL emission -------------------------------------------------------
    def to_pddl_problem(self, name: str | None = None) -> str:
        name = name or self.problem_id
        block_names = [o.name for o in self.objects]
        objs = "    " + " ".join(block_names) + " - block\n"
        objs += "    " + " ".join(self.tables) + " - table"

        def fact(lit: Literal) -> str:
            return f"({' '.join(lit)})"

        init = "\n    ".join(fact(f) for f in self.init_facts)
        goal_parts = " ".join(fact(f) for f in self.goal_facts)
        return (
            f"(define (problem {name})\n"
            f"  (:domain blocksworld)\n"
            f"  (:objects\n{objs})\n"
            f"  (:init\n    {init})\n"
            f"  (:goal (and {goal_parts})))\n"
        )


# Back-compat alias: the package began as a sorting-only project.
SortingProblem = BlocksWorldProblem


# Min centre-to-centre spacing when sampling start poses. Kept comfortably above
# the refinement grasp-clearance (0.05 m) so a *typical* block is graspable, while
# random crowding still produces some obstructions -> a feasible/infeasible mix.
MIN_SPACING = 0.10


# Floor spacing for crowded tables: just above the 0.045 m block width so
# objects don't overlap but may pack tightly (heavy crowding -> obstruction).
SPACING_FLOOR = 0.052


def _sample_on_table(
    table_name: str,
    n: int,
    footprint: float,
    rng: random.Random,
    z: float,
    table_half_extent: float,
) -> list[tuple[float, float, float]]:
    """Rejection-sample n non-overlapping (x, y, z) positions on a table top.

    Uses ``MIN_SPACING`` when ``n`` fits; on dense tables it geometrically shrinks
    the spacing toward ``SPACING_FLOOR`` so generation never fails across the
    sweep grid (the resulting crowding is itself a source of difficulty). A smaller
    ``table_half_extent`` packs the same object count tighter -> more obstruction.
    """
    cx, cy = TABLE_CENTERS[table_name]
    half = table_half_extent - footprint / 2  # keep object edges inside the table top
    spacing = max(MIN_SPACING, 2.2 * footprint)

    # Sparse case: random rejection sampling for a natural layout.
    placed: list[tuple[float, float, float]] = []
    attempts = 0
    while len(placed) < n and attempts < 6000:
        attempts += 1
        x = cx + rng.uniform(-half, half)
        y = cy + rng.uniform(-half, half)
        if all(math.hypot(x - px, y - py) >= spacing for px, py, _ in placed):
            placed.append((x, y, z))
    if len(placed) >= n:
        return placed

    # Dense case: jittered grid guarantees placement up to the table's capacity.
    return _grid_on_table(cx, cy, half, n, footprint, rng, z)


def _grid_on_table(cx, cy, half, n, footprint, rng, z):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    xs = _cell_centers(cx, half, cols)
    ys = _cell_centers(cy, half, rows)
    # jitter within the slack between cell size and object width (full footprint)
    cell = 2 * half / max(cols, rows)
    jit = max(0.0, (cell - footprint) / 2)
    out: list[tuple[float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            if len(out) >= n:
                break
            x = xs[c] + rng.uniform(-jit, jit)
            y = ys[r] + rng.uniform(-jit, jit)
            out.append((x, y, z))
    return out


def _cell_centers(center, half, k):
    if k == 1:
        return [center]
    step = 2 * half / k
    return [center - half + step * (i + 0.5) for i in range(k)]


def generate_sorting_problem(
    num_blocks: int = 4,
    num_blockers: int | None = None,
    seed: int = 0,
    table_half_extent: float | None = None,
    blocker_height: float | None = None,
) -> BlocksWorldProblem:
    """Generate one sorting instance. ``num_blockers`` defaults to ``num_blocks``.

    ``table_half_extent`` / ``blocker_height`` override the scene defaults to make the
    workspace denser (a tighter cluttered bench, reachability preserved) -- the
    principled difficulty knob when crowding alone is too easy. ``None`` keeps the
    calibrated defaults (``TABLE_HALF_EXTENT`` / ``BLOCKER_SIZE``).
    """
    from .scene import TABLE_HALF_EXTENT

    if num_blockers is None:
        num_blockers = max(1, num_blocks // 2)
    half = TABLE_HALF_EXTENT if table_half_extent is None else table_half_extent
    blocker_size = (
        BLOCKER_SIZE
        if blocker_height is None
        else (BLOCKER_SIZE[0], BLOCKER_SIZE[1], blocker_height)
    )
    rng = random.Random(seed)

    num_red = num_blocks // 2
    num_green = num_blocks - num_red

    objects: list[ObjectInfo] = []
    for i in range(num_red):
        objects.append(
            ObjectInfo(f"red_block{i}", "block", "red", BLOCK_SIZE, False, "")
        )
    for i in range(num_green):
        objects.append(
            ObjectInfo(f"green_block{i}", "block", "green", BLOCK_SIZE, False, "")
        )
    for i in range(num_blockers):
        objects.append(
            ObjectInfo(f"blocker{i}", "blocker", "blocker", blocker_size, True, "")
        )

    # assign each object a start table (round-robin over the two start tables)
    for idx, o in enumerate(objects):
        o.start_table = START_TABLES[idx % len(START_TABLES)]

    # sample geometric poses, grouped per start table to avoid overlaps
    scene_objs: list[SceneObject] = []
    for table_name in START_TABLES:
        here = [o for o in objects if o.start_table == table_name]
        if not here:
            continue
        footprint = max(o.size[0] for o in here)
        zs = [o.size[2] / 2.0 for o in here]
        # sample with the largest footprint so spacing is safe for all
        poses = _sample_on_table(
            table_name, len(here), footprint, rng, z=0.0, table_half_extent=half
        )
        for o, (x, y, _), z in zip(here, poses, zs):
            scene_objs.append(
                SceneObject(o.name, o.color, o.is_blocker, (x, y, z), o.size)
            )

    tables = list(TABLE_CENTERS.keys())
    scene = GeometricScene(
        objects=scene_objs,
        tables=[
            Table(name, center, half_extent=half)
            for name, center in TABLE_CENTERS.items()
        ],
    )

    # symbolic init: every object on its start table, clear, hand empty
    init_facts: list[Literal] = [("handempty",)]
    for o in objects:
        init_facts.append(("on-table", o.name, o.start_table))
        init_facts.append(("clear", o.name))

    # symbolic goal: colour-matched tables for blocks; blockers back home
    goal_facts: list[Literal] = []
    for o in objects:
        if o.is_blocker:
            goal_facts.append(("on-table", o.name, o.start_table))
        else:
            goal_facts.append(("on-table", o.name, COLOR_TABLE[o.color]))

    return BlocksWorldProblem(
        problem_id=f"sorting_b{num_blocks}_k{num_blockers}_s{seed}",
        objects=objects,
        tables=tables,
        init_facts=init_facts,
        goal_facts=goal_facts,
        scene=scene,
        seed=seed,
        num_blocks=num_blocks,
        num_blockers=num_blockers,
        problem_type="sorting",
    )


# --------------------------------------------------------------------------- #
# problem-type registry
# --------------------------------------------------------------------------- #
def _generate_stacking_problem(*args, **kwargs) -> BlocksWorldProblem:
    # imported lazily so problem.py has no import cycle with stacking.py
    from .stacking import generate_stacking_problem

    return generate_stacking_problem(*args, **kwargs)


def _generate_clutter_problem(*args, **kwargs) -> BlocksWorldProblem:
    # imported lazily so problem.py has no import cycle with clutter.py
    from .clutter import generate_clutter_problem

    return generate_clutter_problem(*args, **kwargs)


def _generate_capacitated_loading_problem(*args, **kwargs):
    # E1 -- a different world (2D bin-packing), not a BlocksWorldProblem. Lazily
    # imported so ``import blocks_tamp.problem`` never requires shapely/matplotlib.
    # Takes E1's own kwargs (num_items, tightness, margin, ...), not num_blocks.
    from .e1.problem import generate_e1_problem

    return generate_e1_problem(*args, **kwargs)


def _generate_dd2d_problem(*args, **kwargs):
    # DD2D -- Drawer Decluttering in 2D (a different world: Shapely rotated polygons +
    # a grasp model + target-retrieval). Lazily imported so ``import blocks_tamp.problem``
    # never requires shapely/matplotlib. Takes DD2D's own kwargs (lam, margin, seed, ...).
    from .drawer.problem import generate_dd2d_problem

    return generate_dd2d_problem(*args, **kwargs)


# name -> generator. ``make_problem`` dispatches over this so every entry point
# (demo, future sweep/collect) is problem-type agnostic. Note ``capacitated_loading``
# (E1) and ``dd2d`` return their own problem types with a different world model +
# refiner than the blocks-world types; use ``blocks_tamp.e1.demo`` /
# ``blocks_tamp.dd2d.demo`` to run them end to end.
PROBLEM_GENERATORS = {
    "sorting": generate_sorting_problem,
    "stacking": _generate_stacking_problem,
    "clutter": _generate_clutter_problem,
    "capacitated_loading": _generate_capacitated_loading_problem,
    "dd2d": _generate_dd2d_problem,
}


def make_problem(problem_type: str, **kwargs) -> BlocksWorldProblem:
    """Generate one instance of ``problem_type`` ("sorting" | "stacking" | ...)."""
    try:
        gen = PROBLEM_GENERATORS[problem_type]
    except KeyError:
        raise ValueError(
            f"unknown problem_type {problem_type!r}; "
            f"known: {sorted(PROBLEM_GENERATORS)}"
        ) from None
    return gen(**kwargs)
