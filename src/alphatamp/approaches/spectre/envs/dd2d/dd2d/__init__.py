"""DD2D -- Drawer Decluttering in 2D (docs/dd2d_spec.md).

A top-down 2D kitchen-drawer TAMP environment: a *target* item starts ungraspable
(neighbours block every two-finger grasp); the robot stages a **subset of blocker
items** onto an adjacent **buffer** to clear the target, then retrieves it. The hard
decision is *which subset to stage* -- the staged set must jointly pack into the
limited buffer AND be stageable/extractable by the actual gripper. Feasibility hinges
on a global continuous packing statistic a low-order classifier should struggle with
(the plan-feasibility signal PIGINet/LAZY study).

This is a self-contained subpackage backed by **Shapely**; it reuses only the
domain-agnostic layer of ``blocks_tamp`` (``skeleton``, ``record``, ``RefineResult``/
``BoundStep``, ``RenderResult``/``GeometryBackend``, ``ObjectInfo``, the planner's
``domain_pddl_path`` seam). Nothing here imports the PyBullet/Panda stack.

``render`` is intentionally NOT imported here so ``import blocks_tamp.dd2d`` stays
matplotlib-free.
"""

from __future__ import annotations

from .grasps import (
    Grasp,
    finger_rects,
    grasp_cells,
    grasp_cfree,
    has_grasp,
    isolation_graspable,
)
from .planning import DD2DPlanner, make_dd2d_planner
from .problem import DD2DProblem, DrawerScene, generate_dd2d_problem, make_dd2d_problem
from .refine import DD2DRefiner
from .shapes import Shape, sample_shape
from .world import DrawerWorld, ItemState, place_polygon, sample_buffer_pose

__all__ = [
    "Shape",
    "sample_shape",
    "Grasp",
    "grasp_cells",
    "finger_rects",
    "grasp_cfree",
    "has_grasp",
    "isolation_graspable",
    "ItemState",
    "DrawerWorld",
    "place_polygon",
    "sample_buffer_pose",
    "DrawerScene",
    "DD2DProblem",
    "generate_dd2d_problem",
    "make_dd2d_problem",
    "DD2DRefiner",
    "DD2DPlanner",
    "make_dd2d_planner",
]
