"""``--coverage-mode``: isolating ``coverage`` from ``waste``, and the loader that
carries the choice from training to deployment.

The two columns were only ever switched on together, so which one carries v3's effect was
unmeasured. Splitting them has to satisfy three things, and each is a separate way to get
a silently-wrong ablation:

1. Zeroing a column must not change any tensor shape -- otherwise the arms are not
   architecturally comparable and the D-8 exact-absence oracle stops loading.
2. The *other* column must be untouched, or "coverage only" is really "coverage plus a
   perturbed waste".
3. The choice must round-trip through the checkpoint into the deploy kwargs. It changes
   what ``build_example`` *emits*, not what the model *contains*, so a mismatch is
   invisible to ``load_state_dict`` and would fail silently at deploy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphatamp.approaches.spectre.dataset import build_example
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"

_COVERAGE_COL, _WASTE_COL = 2, 3  # cand_overlap = [dead, jaccard, coverage, waste]


def _overlap(episode, vocab, ctx, mode: str) -> np.ndarray:
    example, _ = build_example(
        episode,
        vocab,
        rng=None,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        coverage_feats=True,
        coverage_mode=mode,
    )
    return np.asarray(example.overlap, dtype=float)


def _episode_with_evidence() -> Any:
    """First test episode that has failures to build a non-empty context from.

    Strided rather than truncated: episodes are stored in seed order and the collector
    fills strata in seed bands, so ``[:n]`` yields only stratum 0 -- where the target is
    already graspable, so no candidate has culprits and every coverage column is 0.
    """
    vocab = Vocab.from_json(_VOCAB)
    paths = list_episodes(_V4)
    for path in paths[::7]:
        episode = load_episode(path)
        if episode.scene_geometry is None:
            continue
        failed = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        if len(failed) < 3:
            continue
        ctx = frozenset(failed[:3])
        both = _overlap(episode, vocab, ctx, "both")
        if both[:, _COVERAGE_COL].any() and both[:, _WASTE_COL].any():
            return episode, vocab, ctx
    pytest.skip("no dd2d_v4 episode produced non-zero coverage and waste")


@pytest.mark.skipif(not (_V4 / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_coverage_mode_zeroes_one_column_and_leaves_the_other_exact() -> None:
    """Each mode blanks exactly its own column; the survivor is bit-identical."""
    episode, vocab, ctx = _episode_with_evidence()
    both = _overlap(episode, vocab, ctx, "both")
    cov = _overlap(episode, vocab, ctx, "coverage")
    waste = _overlap(episode, vocab, ctx, "waste")

    assert both.shape == cov.shape == waste.shape  # shape is never narrowed

    # coverage-only: waste blanked, coverage untouched
    assert not cov[:, _WASTE_COL].any()
    np.testing.assert_array_equal(cov[:, _COVERAGE_COL], both[:, _COVERAGE_COL])
    # waste-only: the mirror image
    assert not waste[:, _COVERAGE_COL].any()
    np.testing.assert_array_equal(waste[:, _WASTE_COL], both[:, _WASTE_COL])
    # the columns the mode does not name are identical in every arm
    for col in (0, 1):
        np.testing.assert_array_equal(cov[:, col], both[:, col])
        np.testing.assert_array_equal(waste[:, col], both[:, col])


@pytest.mark.skipif(not (_V4 / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_coverage_mode_is_inert_at_empty_context() -> None:
    """Both columns are 0 at |F|=0 regardless of mode, so the first attempt is static.

    This is what makes the coverage features a purely *adaptive* signal: nothing has
    been observed yet, so there is nothing for them to say.
    """
    episode, vocab, _ = _episode_with_evidence()
    for mode in ("both", "coverage", "waste"):
        rows = _overlap(episode, vocab, frozenset(), mode)
        assert not rows[:, _COVERAGE_COL].any()
        assert not rows[:, _WASTE_COL].any()


@pytest.mark.skipif(not (_V4 / "episodes").is_dir(), reason="dd2d_v4 collection absent")
def test_coverage_mode_default_is_exact_absence() -> None:
    """Omitting the flag reproduces the pre-flag behaviour byte for byte.

    Every v3 feature is config-gated so that *off* is exactly the older model (D-8); a
    new knob that perturbed the default would retire the equivalence oracle by accident.
    """
    episode, vocab, ctx = _episode_with_evidence()
    explicit = _overlap(episode, vocab, ctx, "both")
    example, _ = build_example(
        episode,
        vocab,
        rng=None,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        coverage_feats=True,  # no coverage_mode -> default
    )
    np.testing.assert_array_equal(np.asarray(example.overlap, float), explicit)


@pytest.mark.skipif(not _VOCAB.is_file(), reason="dd2d_v4 vocab absent")
def test_load_checkpoint_round_trips_the_deploy_kwargs(tmp_path) -> None:
    """The switches that change what the tensorizer emits come back off the checkpoint.

    ``load_state_dict(strict=True)`` catches a wrong *architecture*, but ``overlap_mode``
    and ``coverage_mode`` are invisible to it -- deploying under the wrong one feeds the
    model a column it never saw populated. So they are read back, never passed in.

    Self-contained: it saves a checkpoint through the same ``asdict(TrainConfig)`` path
    ``run_training`` uses, rather than reading a disk artifact -- a real deployed
    checkpoint is the width-3 narrowed model now, and a pre-narrowing artifact no
    longer loads by design.
    """
    from dataclasses import asdict

    import torch

    from alphatamp.approaches.spectre.inference import load_checkpoint
    from alphatamp.approaches.spectre.model import (
        N_OVERLAP_COV,
        SpectreConfig,
        SpectreModel,
    )
    from alphatamp.approaches.spectre.train import TrainConfig

    vocab = Vocab.from_json(_VOCAB)
    # the deployed arm: jaccard overlap, coverage on both columns, records aggregated,
    # state-delta off (the v3final flags) -- built and saved as run_training does.
    cfg = TrainConfig(
        overlap_mode="jaccard",
        use_overlap=True,
        coverage_feats=True,
        aggregate_records=True,
        use_records=True,
        evidence_attn=True,
        use_state_delta=False,
    )
    model = SpectreModel(
        n_ops=len(vocab.operators),
        max_arity=vocab.max_operator_arity,
        cfg=SpectreConfig(
            n_overlap_feats=N_OVERLAP_COV,
            n_prior_feats=0,
            d_rel=cfg.d_rel,
            use_records=True,
            evidence_attn=True,
            coverage_feats=True,
            n_predicates=len(vocab.predicates),
            max_pred_arity=vocab.max_predicate_arity,
        ),
    )
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "cfg": asdict(cfg),
            "n_ops": len(vocab.operators),
            "state_dict": model.state_dict(),
        },
        ckpt,
    )

    _model, deploy = load_checkpoint(ckpt, vocab, "cpu")
    assert set(deploy) == {
        "overlap_mode",
        "aggregate_records",
        "coverage_feats",
        "coverage_mode",
        "repeat_feats",
        "regroup_feats",
        "state_delta",
        "record_holdout",
    }
    # the deployed arm: jaccard overlap, coverage on, both columns
    assert deploy["overlap_mode"] == "jaccard"
    assert deploy["coverage_feats"] is True
    assert deploy["aggregate_records"] is True
    assert deploy["coverage_mode"] == "both"
    assert deploy["state_delta"] is False
    # emission-only, defaults to the historical holdout (P-1 flips it off explicitly)
    assert deploy["record_holdout"] is True
    # repeat/regroup default off for a checkpoint trained without them (this arm)
    assert deploy["repeat_feats"] is False
    assert deploy["regroup_feats"] is False
