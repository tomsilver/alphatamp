"""Spec §8.3 #10 — tractability across many sampled problems.

Across 100 generated problems (subsample of the 500-problem training set),
≥99% must have at least one feasible skeleton in their candidate pool — i.e.,
``ThreeGateRefiner`` must produce ≥1 success. We sample 100 not 500 so this
test stays under a second.
"""

from __future__ import annotations

import itertools

from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    make_problem,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.refiner import (
    ThreeGateRefiner,
)


def test_at_least_one_success_per_problem_majority() -> None:
    n_problems = 100
    succ_counts = []
    for seed in range(n_problems):
        p = make_problem(seed=seed, variant="n3-v1")
        gen = ClosedFormSkeletonGenerator(p, k_cap=30)
        succs = 0
        for idx, (state_plan, action_plan) in enumerate(
            itertools.islice(gen(None, p.initial_abstract_state, p.goal, 1.0, None), 30)
        ):
            r = ThreeGateRefiner(p, seed=idx, base_op_fail_rate=0.0)
            if r(None, state_plan, action_plan, 1.0, None) is not None:
                succs += 1
        succ_counts.append(succs)

    # With base_op_fail_rate=0, a problem has ≥1 success iff its right-family
    # is tag-feasible. The rejection sampling at make_problem time guarantees
    # this for every accepted problem. So we expect 100% feasibility.
    feasible = sum(1 for c in succ_counts if c >= 1)
    assert feasible / n_problems >= 0.99, f"only {feasible}/{n_problems} feasible"


def test_pool_size_constant_across_seeds() -> None:
    """The pool size for n3-v1 must be exactly 30 regardless of layout/tags/latent."""
    for seed in range(20):
        p = make_problem(seed=seed, variant="n3-v1")
        gen = ClosedFormSkeletonGenerator(p, k_cap=30)
        pool = list(
            itertools.islice(
                gen(None, p.initial_abstract_state, p.goal, 1.0, None), 200
            )
        )
        assert len(pool) == 30, f"seed={seed} pool={len(pool)}"
