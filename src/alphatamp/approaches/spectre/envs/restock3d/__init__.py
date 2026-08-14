"""Restock3D — a MuJoCo TidyBot restock environment for SPECTRE.

A floor staging area holds goal objects (small cubes + tall blocks) that must be stored into
single-object shelf regions split across a short cell and a tall cell. Region capacity and cell
height are geometry, invisible above the abstraction line — so a height-blind / capacity-blind
task planner produces many goal-reaching skeletons that fail refinement (an oracle avoids them).

Difficulty comes from a *sampler-level geometric feasibility gate*, not physics: a placement that
over-assigns a region (F2, self-inflicted culprits) or puts a tall object under a short cell (F3,
culprit-free exhaustion) is rejected, so infeasible candidates genuinely fail. Physics only
transitions accepted samples and renders demos. This is the DD2D lesson applied in MuJoCo — it is
what avoids the ShelfObstruct3D failure where physics squeezed past obstructions (FP=0).

See ``docs/restock3d_proposal.md`` and the ADRs in ``docs/decisions/07-stickbutton2d.md``.
"""
