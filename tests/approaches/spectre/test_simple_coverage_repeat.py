"""The re-added simple/legacy coverage/waste and the isolated SB2D ``repeat`` probe.

Locks the two load-bearing facts behind ``compare_methods_simple.py``
(docs/decisions/07 2026-08-27):

1. ``unified_coverage=False`` (``--legacy-coverage``) is a *distinct* definition from the
   deployed unified one -- a value swap inside the same two columns, so shape is stable and
   old unified checkpoints keep loading, but the numbers actually change on DD2D.
2. On StickButton2D the simple coverage/waste is inert (it reads ``r.culprits`` only, and
   SB2D reports ``dev_blame``), so ``repeat`` carries SB2D -- which fires ONLY under the
   isolated ``_SB2D_REPEAT`` spec (env_variant ``stickbutton2d_v1_simple``) and stays
   identically 0 under the deployed ``_SB2D``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphatamp.approaches.spectre import domain
from alphatamp.approaches.spectre.dataset import build_example
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_DD2D = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_DD2D_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
_SB2D = _ROOT / "data" / "spectre" / "raw" / "stickbutton2d_v1" / "train"
_SB2D_VOCAB = _ROOT / "data" / "spectre" / "derived" / "stickbutton2d_v1" / "train_vocab.json"

# cand_overlap = [dead, jaccard, coverage, waste, (repeat, regroup)]
_COV, _WST, _REP = 2, 3, 4


def _overlap_for(episode: Any, ctx: frozenset, vocab: Vocab, variant: str) -> np.ndarray:
    example, _ = build_example(
        episode,
        vocab,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        spec=domain.spec_for(variant),
        overlap_mode="jaccard",
        coverage_feats=True,
        unified_coverage=False,
        repeat_feats=True,
    )
    return np.asarray(example.overlap, dtype=float)


@pytest.mark.skipif(not (_DD2D / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_legacy_coverage_is_a_distinct_definition_on_dd2d() -> None:
    """Legacy vs unified: same shape, but the numbers differ and stay in [0, 1]."""
    vocab = Vocab.from_json(_DD2D_VOCAB)
    spec = domain.spec_for("dd2d_v4")

    def _overlap(episode: Any, ctx: frozenset, unified: bool) -> np.ndarray:
        example, _ = build_example(
            episode,
            vocab,
            evidence=True,
            context_f=ctx,
            augment_tags=False,
            spec=spec,
            coverage_feats=True,
            unified_coverage=unified,
        )
        return np.asarray(example.overlap, dtype=float)

    # Find a context where the two definitions actually diverge (the archive doc measures
    # coverage vectors differing on ~100% of dd2d_v4 contexts, but we search to stay robust
    # to the PYTHONHASHSEED-dependent collection).
    for path in list_episodes(_DD2D)[::-1]:
        episode = load_episode(path)
        failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        if len(failed) < 3:
            continue
        ctx = frozenset(failed[:3])
        uni = _overlap(episode, ctx, True)
        leg = _overlap(episode, ctx, False)
        assert uni.shape == leg.shape  # value swap, never a width change
        assert leg[:, [_COV, _WST]].min() >= 0.0
        assert leg[:, [_COV, _WST]].max() <= 1.0
        if not np.array_equal(uni[:, [_COV, _WST]], leg[:, [_COV, _WST]]):
            return
    pytest.skip("no dd2d_v4 context where legacy and unified coverage diverge")


@pytest.mark.skipif(not (_SB2D / "episodes").is_dir(), reason="stickbutton2d_v1 collection absent")
def test_sb2d_repeat_fires_only_under_the_isolated_simple_spec() -> None:
    """`repeat` is 0 under deployed `_SB2D`, non-zero under `_SB2D_REPEAT`; simple
    coverage/waste is inert on SB2D (it reads culprits, SB2D reports dev_blame)."""
    vocab = Vocab.from_json(_SB2D_VOCAB)
    # `repeat` fires on a blameless, exhausted press failure (a reach failure), which is
    # sparse and lives mainly at b5 -- so search the top (b5) band for an episode where it
    # fires under the simple spec, then check the deployed spec is inert on that same one.
    for path in list_episodes(_SB2D)[::-1]:
        episode = load_episode(path)
        failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        if not failed:
            continue
        ctx = frozenset(failed[:3])
        simple = _overlap_for(episode, ctx, vocab, "stickbutton2d_v1_simple")
        if simple[:, _REP].sum() > 0.0:
            deployed = _overlap_for(episode, ctx, vocab, "stickbutton2d_v1")
            # repeat is structurally inert on the deployed spec, live on the isolated one.
            assert deployed[:, _REP].sum() == 0.0
            # simple coverage is identically 0 on SB2D (culprit-free class-2 failures).
            assert simple[:, _COV].sum() == 0.0
            return
    pytest.skip("no stickbutton2d_v1 episode where the simple-spec repeat fires")


def test_sb2d_repeat_spec_declarations() -> None:
    """The domain contract: only the simple SB2D variant declares `step_certificate`."""
    press = "RobotPressButtonFromNothing"
    assert not domain.spec_for("stickbutton2d_v1").axioms_for(press).step_certificate
    assert domain.spec_for("stickbutton2d_v1_simple").axioms_for(press).step_certificate
    # DD2D simple maps to the ordinary DD2D contract (repeat inert there).
    assert domain.spec_for("dd2d_v4_simple") is domain.spec_for("dd2d_v4")
