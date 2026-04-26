"""Tests for ``_make_plan_generator``'s three-way dispatch on ``cfg.plan_generator``.

Verifies that routedtransport2d collections can opt into the same heuristic-search
generator the kinder envs already use, while keeping the deterministic closed-form
enumerator as the default. Does *not* run a full collection — only checks the
constructed generator object's type.
"""

from __future__ import annotations

from types import SimpleNamespace

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)

from alphatamp.approaches.spectre.collect import _make_plan_generator
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.routedtransport2d.operators import (
    ALL_OPERATORS,
    ALL_PREDICATES,
    ALL_TYPES,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    make_problem,
)


def _rt2d_cfg(plan_generator: str) -> CollectionConfig:
    return CollectionConfig(
        env_id="routedtransport2d/RoutedTransport2D-n3-v1",
        env_variant="routedtransport2d_n3_v1",
        model_name="routedtransport2d",
        model_kwargs={"variant": "n3-v1"},
        split="train",
        num_problems=1,
        problem_seed_start=0,
        problem_seed_end=1,
        K_max=30,
        plan_generator=plan_generator,  # type: ignore[arg-type]
    )


def test_dispatch_closed_form_default() -> None:
    """Default ``plan_generator`` is ``"closed_form"`` and routes to enumerator."""
    cfg = _rt2d_cfg("closed_form")
    assert cfg.plan_generator == "closed_form"
    problem = make_problem(seed=0, variant="n3-v1")
    obs = {"_problem": problem}
    env_models = SimpleNamespace(
        types=ALL_TYPES,
        predicates=ALL_PREDICATES,
        operators=ALL_OPERATORS,
    )
    gen = _make_plan_generator(cfg, env_models, obs, problem_id=0)  # type: ignore[arg-type]
    assert isinstance(gen, ClosedFormSkeletonGenerator)


def test_dispatch_heuristic_search_for_rt2d() -> None:
    """Setting ``plan_generator="heuristic_search"`` for RT2D bypasses closed-form."""
    cfg = _rt2d_cfg("heuristic_search")
    # ``obs`` is irrelevant on this path (the ``del obs`` in collect.py); pass an
    # empty dict to confirm the dispatcher does not assert on its contents.
    env_models = SimpleNamespace(
        types=ALL_TYPES,
        predicates=ALL_PREDICATES,
        operators=ALL_OPERATORS,
    )
    gen = _make_plan_generator(cfg, env_models, {}, problem_id=0)  # type: ignore[arg-type]
    assert isinstance(gen, RelationalHeuristicSearchAbstractPlanGenerator)


def test_config_hash_distinguishes_plan_generator() -> None:
    """Switching the field changes ``config_hash`` so collections live in distinct
    dirs."""
    a = _rt2d_cfg("closed_form").config_hash
    b = _rt2d_cfg("heuristic_search").config_hash
    assert a != b
