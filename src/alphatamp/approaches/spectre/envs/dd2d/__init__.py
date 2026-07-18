"""DD2D slice of the blocks_tamp pipeline: the Drawer-Decluttering-in-2D TAMP
environment mapped onto the PIGINet plan-feasibility record format.

Migrated out of the ``envsearch`` research repo (see ``MIGRATION_DD2D.md``). This
package holds DD2D + the shared domain-agnostic layer (+ optionally the PIGINet
baseline); the sorting / clutter / stacking / E1 environments were intentionally
excluded, so this ``__init__`` does NOT eagerly import them.

Pipeline: generate a problem -> enumerate diverse task plans (skeletons) with a
FORBID-SEARCH / top-k planner -> refine each skeleton geometrically -> emit a
PIGINet training-example record. See ``docs/dd2d.md`` and
``docs/piginet_record_schema.md``.
"""

from __future__ import annotations

from .problem import (
    BlocksWorldProblem,
    ObjectInfo,
    SortingProblem,
    make_problem,
)
from .skeleton import Action, Skeleton

__all__ = [
    "Action",
    "Skeleton",
    "BlocksWorldProblem",
    "ObjectInfo",
    "SortingProblem",
    "make_problem",
]
