"""The per-candidate refinement-abandonment cap (deployed wall-clock configuration).

Two code paths carry the cap, tested separately:

- **Fixed-order accounting** (``precompute_dd2d_cache._fp_and_refine_capped``) for the
  score-ordered methods (astar / PIGINet / SPECTRE-static): pure, so it is pinned fast
  on toy episodes with hand-checked ``(fp_capped, refine_s_capped)``.
- **The rollout** (``inference_v3.deployed_rollout_v3_traced(refine_cap_s=…)``): a
  slow-feasible candidate over the cap must stop being a *stopping* success -- it is
  abandoned into the failure context and the loop continues -- so the adaptive order can
  diverge. Verified on real dd2d_v4 episodes (slow, skipped without the gitignored
  data), including that ``refine_cap_s=None`` reproduces the uncapped rollout exactly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from _fixtures import build_toy_episode

# Test idioms: deferred heavy imports inside functions, and reaching a module-private
# cap helper on the (experiments/) precompute module under test.
# pylint: disable=import-outside-toplevel,protected-access


_ROOT = Path(__file__).resolve().parents[3]


def _load_precompute() -> ModuleType:
    """Import the (experiments/) precompute module for its pure cap helper.

    It lives outside the package, so load it by path; skip cleanly if that fails rather
    than fail the suite on an environment where the experiments tree is absent.
    """
    path = _ROOT / "experiments" / "spectre" / "precompute_dd2d_cache.py"
    if not path.is_file():
        pytest.skip("precompute_dd2d_cache.py not found")
    spec = importlib.util.spec_from_file_location("precompute_dd2d_cache", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - import env issue
        pytest.skip(f"cannot import precompute_dd2d_cache: {exc!r}")
    return module


# --------------------------------------------------------------------------- #
# Fixed-order accounting -- fast, pure, hand-checked
# --------------------------------------------------------------------------- #
# build_toy_episode sets refinement_wall_clock_s = 0.1 * (i + 1), so candidate i has
# time 0.1*(i+1): idx0=0.1, idx1=0.2, idx2=0.3, ...


def test_fixed_order_cap_stops_at_first_sub_cap_success() -> None:
    """A cap above every time reproduces the uncapped stop and refine sum."""
    m = _load_precompute()
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))  # times .1 .2 .3
    # Cap above every time: identical to uncapped -- stop at idx2, pay all three.
    fp, refine = m._fp_and_refine_capped(ep, [0, 1, 2], cap=1.0)
    assert fp == 2.0
    assert refine == pytest.approx(0.6)  # .1 + .2 + .3
    # Order that hits the success first: no failures, pay only its time.
    fp, refine = m._fp_and_refine_capped(ep, [2, 0, 1], cap=1.0)
    assert fp == 0.0
    assert refine == pytest.approx(0.3)


def test_fixed_order_cap_abandons_slow_feasible_and_censors() -> None:
    """When the only feasible candidate is over the cap, the order is censored."""
    m = _load_precompute()
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))  # times .1 .2 .3
    # Cap below the only feasible candidate's time (0.3): it is abandoned, nothing stops
    # the order -> censored (fp == pool size), every attempt charged the cap.
    fp, refine = m._fp_and_refine_capped(ep, [0, 1, 2], cap=0.25)
    assert fp == 3.0
    assert refine == pytest.approx(0.1 + 0.2 + 0.25)  # min(t, cap) each


def test_fixed_order_cap_skips_over_slow_feasible_to_fast_one() -> None:
    """A slow feasible candidate is abandoned; the order stops at a later fast one."""
    m = _load_precompute()
    # Two feasible candidates: idx0 (fast, .1) and idx2 (slow, .3).
    ep = build_toy_episode(outcomes=("success", "fail", "success"))  # times .1 .2 .3
    # Order reaches the slow feasible (idx2, .3) before the fast one (idx0, .1); a 0.25
    # cap abandons idx2 and stops at idx0.
    fp, refine = m._fp_and_refine_capped(ep, [1, 2, 0], cap=0.25)
    assert fp == 2.0  # idx1 (fail) and idx2 (abandoned) both count as failures
    assert refine == pytest.approx(0.2 + 0.25 + 0.1)


def test_deployed_cap_constant_is_positive() -> None:
    """The deployed cap constant is a positive number of seconds."""
    m = _load_precompute()
    assert m.REFINE_CAP_S > 0


# --------------------------------------------------------------------------- #
# Rollout cap -- real dd2d_v4 data, slow
# --------------------------------------------------------------------------- #
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
# The deployed DD2D checkpoint. Repointed 2026-08-08 from `checkpoints_v3_v3final_s0`:
# the cap regimes (sub-cap stop vs slow-feasible stop) need a trained model's realistic
# picks, and this is the canonical deployed dir, now the width-3 narrowed model.
_CKPT = (
    _ROOT
    / "data"
    / "spectre"
    / "checkpoints_v3_unified"
    / "dd2d_v4"
    / "seed_0"
    / "best.pt"
)

_needs_v4 = pytest.mark.skipif(
    not (_V4.is_dir() and _VOCAB.is_file() and _CKPT.is_file()),
    reason="dd2d_v4 collection / vocab / v3 checkpoint absent (gitignored)",
)


def _strided_episodes(n: int):
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    paths = list_episodes(_V4)
    stride = max(1, len(paths) // n)
    out = []
    for path in paths[::stride][:n]:
        ep = load_episode(path)
        if ep.scene_geometry is not None:
            out.append(ep)
    return out


def _refine_along(ep, order, cap) -> float:
    total = 0.0
    for idx in order:
        t = float(ep.outcomes[idx].refinement_wall_clock_s or 0.0)
        total += min(t, cap) if cap is not None else t
        if ep.outcomes[idx].outcome == "success" and (cap is None or t <= cap):
            break
    return total


@pytest.mark.slow
@_needs_v4
def test_rollout_cap_none_is_identical_and_diverges_when_capped() -> None:
    """refine_cap_s=None matches the default; a cap diverges on a slow-feasible stop."""
    from alphatamp.approaches.spectre.inference_v3 import (
        deployed_rollout_v3_traced,
        load_v3_checkpoint,
    )
    from alphatamp.approaches.spectre.vocab import Vocab

    vocab = Vocab.from_json(_VOCAB)
    model, deploy = load_v3_checkpoint(_CKPT, vocab, "cpu")
    cap = 2.0

    diverged = False
    checked_fast = 0
    for ep in _strided_episodes(40):
        a0, t0 = deployed_rollout_v3_traced(model, ep, vocab, "cpu", **deploy)
        # refine_cap_s=None must reproduce the default call bit-for-bit.
        a_none, t_none = deployed_rollout_v3_traced(
            model, ep, vocab, "cpu", refine_cap_s=None, **deploy
        )
        assert (a_none, t_none.order) == (a0, t0.order)
        # Uncapped refine along the order matches the trace's accumulation.
        assert t0.refine_capped_seconds == pytest.approx(
            _refine_along(ep, t0.order, None), abs=1e-6
        )

        a_cap, t_cap = deployed_rollout_v3_traced(
            model, ep, vocab, "cpu", refine_cap_s=cap, **deploy
        )
        # The trace's capped refine equals min(t, cap) summed along the capped order.
        assert t_cap.refine_capped_seconds == pytest.approx(
            _refine_along(ep, t_cap.order, cap), abs=1e-6
        )
        first_succ_t = float(ep.outcomes[t0.order[-1]].refinement_wall_clock_s or 0.0)
        if ep.outcomes[t0.order[-1]].outcome == "success" and first_succ_t <= cap:
            # The uncapped stop is already sub-cap: capping changes nothing here.
            assert t_cap.order == t0.order
            assert a_cap == a0
            checked_fast += 1
        elif ep.outcomes[t0.order[-1]].outcome == "success" and first_succ_t > cap:
            # The uncapped stop is a slow-feasible candidate: the cap abandons it, so
            # the order must extend past it and the attempt count cannot fall.
            diverged = True
            assert t_cap.order != t0.order
            assert a_cap >= a0

    assert checked_fast > 0, "no sub-cap-stop problem exercised the no-op branch"
    assert (
        diverged
    ), "no slow-feasible stop in the sample -- widen it to hit the cap path"
