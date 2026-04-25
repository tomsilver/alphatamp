"""Spec §8.3 #7 — no universally-safe skeleton.

For every skeleton in the pool, there must exist at least one (blocked_color,
blocked_grasp) mode under which a gate would fire (color or grasp). This is the
structural guarantee that SPECTRE's adaptivity mechanism activates on RT2D.
"""

from __future__ import annotations

import itertools

from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
    _skeleton_family,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    make_problem,
)


def test_no_skeleton_safe_in_all_six_modes() -> None:
    p = make_problem(seed=0, variant="n3-v1")
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    pool = list(
        itertools.islice(gen(None, p.initial_abstract_state, p.goal, 1.0, None), 30)
    )
    modes = [(c, g) for c in ("A", "B", "C") for g in ("top", "side")]

    for _state_plan, action_plan in pool:
        loaded_colors, grasp = _skeleton_family(action_plan)
        # The skeleton has a fired gate in mode (c, g) iff c ∈ loaded_colors
        # OR g == grasp. The skeleton is "safe" only in the unique mode where
        # neither holds.
        safe_modes = [(c, g) for c, g in modes if c not in loaded_colors and g != grasp]
        assert len(safe_modes) == 1, (
            f"skeleton with family {loaded_colors,grasp} should have exactly one"
            f" safe mode; got {safe_modes}"
        )
