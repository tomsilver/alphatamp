"""F-C2 rollout-aligned |F| curriculum: sampler invariants + config contract.

The curriculum lives entirely in `sample_context`'s size draw (a training-time context
sampler). It touches neither the model nor `build_example`'s emitted arrays nor the
deploy path, so there is no state_dict / checkpoint surface to test -- only that the
``phi`` draw stays inside the deployment visit range and that ``phi=None`` reproduces the
historical draw exactly. See decisions.md 2026-08-23.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from alphatamp.approaches.spectre.dataset import sample_context
from alphatamp.approaches.spectre.train import TrainConfig


def test_phi_draw_never_exceeds_phi_or_fail_count() -> None:
    """|F| in {0..min(phi, |fail|)} across many draws; p_empty=0 isolates the size law."""
    fail_idx = list(range(50))
    rng = np.random.default_rng(0)
    sizes = set()
    for _ in range(2000):
        ctx, _ = sample_context(fail_idx, rng, p_empty=0.0, phi=12)
        assert ctx.issubset(set(fail_idx))
        assert len(ctx) <= 12  # never past phi (the rollout never visits |F|>phi)
        sizes.add(len(ctx))
    # Uniform{0..12} -> every size in [0, 12] is reachable (0 is a legitimate draw).
    assert sizes == set(range(13))


def test_phi_truncated_to_available_failures() -> None:
    """phi larger than |fail| truncates to |fail| -- the pool cannot supply more."""
    fail_idx = [3, 7, 9]  # only 3 failures available
    rng = np.random.default_rng(1)
    for _ in range(500):
        ctx, _ = sample_context(fail_idx, rng, p_empty=0.0, phi=40)
        assert len(ctx) <= 3


def test_p_empty_floor_still_applies_under_phi() -> None:
    """The static |F|=0 floor is kept on top of the rollout draw (user directive 30%)."""
    fail_idx = list(range(30))
    rng = np.random.default_rng(2)
    empt = sum(
        1
        for _ in range(4000)
        if not sample_context(fail_idx, rng, p_empty=0.30, phi=20)[0]
    )
    # 0.30 forced-empty + 0.70 * Uniform{0..20}'s own 1/21 mass at 0 -> ~0.333.
    assert 0.28 < empt / 4000 < 0.40


def test_phi_none_reproduces_historical_uniform_draw() -> None:
    """phi=None is the legacy [1, min(max_f, |fail|)] draw, bit-for-bit per RNG stream."""
    fail_idx = list(range(20))
    a = sample_context(fail_idx, np.random.default_rng(5), p_empty=0.0, max_f=8)
    b = sample_context(
        fail_idx, np.random.default_rng(5), p_empty=0.0, max_f=8, phi=None
    )
    assert a == b
    # legacy law never draws |F|=0 outside the p_empty branch, and caps at max_f.
    rng = np.random.default_rng(6)
    for _ in range(1000):
        ctx, _ = sample_context(fail_idx, rng, p_empty=0.0, max_f=8)
        assert 1 <= len(ctx) <= 8


def test_config_defaults_and_persistence() -> None:
    """Default is the uniform mode; the two fields persist into the checkpoint dict."""
    assert TrainConfig().context_mode == "uniform"
    assert TrainConfig().phi_path == ""
    d = asdict(TrainConfig(context_mode="rollout", phi_path="x.json"))
    assert d["context_mode"] == "rollout" and d["phi_path"] == "x.json"
