"""Restock3D — a kinematic-PyBullet restock environment for SPECTRE.

A floor staging area holds goal objects (small cubes + tall blocks) that must be stored into
single-object regions on ONE shelf whose boards make a **tall section (bottom)** and a **short
section (top)**. Region capacity and cell height are geometry, invisible above the abstraction line
(``Place`` has no ``Clear`` precondition) — so a height-/capacity-blind task planner produces many
goal-reaching skeletons that fail refinement, which an oracle avoids.

Feasibility is decided by **real PyBullet collision**, not a hand-written gate: the kinematic env
reverts colliding moves and the pick/place controllers raise ``TrajectorySamplingFailure`` when
motion planning finds no collision-free solution. A tall block, kept upright by the front-grasp
translate-only place, genuinely collides the board capping the short section (F3); a placement that
over-assigns a region collides its resident (F2); adjacent floor clutter blocks a grasp (F1). This
is the kinematic rebuild of the earlier MuJoCo/dynamic3d attempt, whose soft collisions let the
robot shove blockers aside (the ShelfObstruct3D inertness failure).

See ``docs/restock3d_proposal.md`` and the ADRs in ``docs/decisions/07-stickbutton2d.md``.
"""
