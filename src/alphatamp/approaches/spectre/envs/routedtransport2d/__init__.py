"""RoutedTransport2D v1 — multi-axis-latent, tag-augmented K_3,3 substrate.

See ``docs/archive/ROUTED_TRANSPORT2D_SPEC.md`` (spectre package docs) for the
design rationale and ``docs/archive/SPECTRE_METHOD_SPEC.md`` for the
method-side contracts this env satisfies.

Public surface:

- :func:`make_problem` — sample a deterministic ``ProblemInstance`` from a seed.
- :class:`ProblemInstance` — bundle of (latent, tags, abstract state, goal, objects).
- :class:`ClosedFormSkeletonGenerator` — duck-compatible with bilevel_planning's
  ``RelationalHeuristicSearchAbstractPlanGenerator`` interface.
- :class:`ThreeGateRefiner` — duck-compatible with ``BacktrackingRefiner``.
- :func:`create_routedtransport_models` — builds a ``SesameModels`` for collect.py.
- :class:`RoutedTransport2DEnv` — stub gym env that wraps a ProblemInstance per reset.
- :func:`routed_transport_variants` — env_registry helper to build ``ExtraVariant`` rows.
"""

from alphatamp.approaches.spectre.envs.routedtransport2d.env_models import (
    create_routedtransport_models,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.gym_env import (
    RoutedTransport2DEnv,
    routed_transport_variants,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    ProblemInstance,
    make_problem,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.refiner import (
    RefineOutcome,
    ThreeGateRefiner,
)

__all__ = [
    "ClosedFormSkeletonGenerator",
    "ProblemInstance",
    "RefineOutcome",
    "RoutedTransport2DEnv",
    "ThreeGateRefiner",
    "create_routedtransport_models",
    "make_problem",
    "routed_transport_variants",
]
