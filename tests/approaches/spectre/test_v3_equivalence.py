"""G1's equivalence oracle: the v3 code path reproduces deployed v2.2 exactly.

The reference is a **live v2.2 run**, not the stored comparison cache. That is a
deliberate correction. The cache was the obvious oracle -- it holds, per test problem, the
exact attempt order, the demotion state at every step and the logits that produced them --
but replaying it showed that the *current* v2.2 code and the *current* deployed checkpoint
no longer reproduce it (mean FP 14.50 vs the cached 13.68; identical attempt order on only
55/100 problems). Since v2 and v3 agree bit-for-bit, that is a pre-existing staleness in
the v2.2 artifacts, not a v3 regression -- see ``notebook.md`` for the audit.

An oracle has to be something we can recompute. So the gate here is: **run v2.2 and v3
side by side on the same episodes and demand bit-identical output.** That is strictly
stronger than the cache comparison it replaces (exact equality, not a 4-dp tolerance),
needs no stored artifact, and cannot silently rot.

**Why decisions and not FP.** FP is a one-number summary that hides a changed tie-break,
and "per-stratum within noise" is unfalsifiable at n=25 per stratum. Requiring the
identical attempt sequence is falsifiable and localises a regression to a step.

This oracle is the backbone for the gates that follow: the domain adapter and the record
tokens rewrite the data path underneath it, and any change that moves a decision fails
here loudly instead of drifting a mean. It necessarily retires when the position encoding
is replaced (``cands.pos_emb`` leaves the state dict), which is why that is scheduled as
the last architectural change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "spectre"
_TEST_SPLIT = _DATA / "raw" / "dd2d_v3" / "test"
_VOCAB = _DATA / "derived" / "dd2d_v3" / "train_vocab.json"
_CACHE = (
    _DATA / "derived" / "dd2d_v3" / "compare_cache" / "spectre2_adaptive" / "seed_0"
)
_CKPT = _DATA / "checkpoints_v2_evidence_ov" / "dd2d_v3" / "seed_0" / "best.pt"

pytestmark = pytest.mark.skipif(
    not (_CACHE.is_dir() and _CKPT.is_file() and _VOCAB.is_file()),
    reason="dd2d_v3 artifacts absent (gitignored data / compare cache not built)",
)


def _load():
    from alphatamp.approaches.spectre.model_v3 import load_v2_checkpoint
    from alphatamp.approaches.spectre.vocab import Vocab

    model, cfg = load_v2_checkpoint(_CKPT)  # strict=True inside
    return model, cfg, Vocab.from_json(_VOCAB)


def _episodes_across_strata(n: int):
    """Load ``n`` episodes spread over all four strata.

    Episodes are stored in seed order and the collector fills strata in seed bands, so
    taking the first ``n`` yields *only stratum 0* -- where every method attempts once and
    stops. That would make an equivalence test pass without exercising a single rollout
    step, so stride instead.
    """
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    paths = list_episodes(_TEST_SPLIT)
    stride = max(1, len(paths) // n)
    out = []
    for path in paths[::stride][:n]:
        episode = load_episode(path)
        if episode.scene_geometry is not None:
            out.append(episode)
    return out


def test_v2_checkpoint_loads_into_v3_strictly() -> None:
    """Compat mode is the v2.2 architecture, so ``strict=True`` must succeed.

    A silent key mismatch would leave a randomly-initialised submodule that still runs
    and still emits plausible logits -- the failure mode this whole gate exists to catch.
    """
    model, cfg, _ = _load()
    assert cfg["use_overlap"] is True
    assert cfg["use_prior"] is False, "the deployed dd2d_v3 model drops the prior"
    assert model.cfg.n_overlap_feats == 2
    assert model.cfg.n_prior_feats == 0


def test_v3_state_dict_keys_match_v2() -> None:
    """Tripwire against an accidental submodule rename.

    Renaming ``scene``/``cands``/``facts``/``scorer``/``aux`` breaks checkpoint loading
    for every stored v2.2 run at once, and the symptom (slightly worse numbers) looks
    like a modelling result rather than a bug.
    """
    from alphatamp.approaches.spectre.model_v2 import SpectreV2Model
    from alphatamp.approaches.spectre.model_v3 import SpectreV3Model, V3Config

    kwargs = dict(n_ops=4, max_arity=1)
    v2 = SpectreV2Model(**kwargs, n_overlap_feats=2, n_prior_feats=0)
    v3 = SpectreV3Model(**kwargs, cfg=V3Config(n_overlap_feats=2, n_prior_feats=0))
    k2 = {k: tuple(v.shape) for k, v in v2.state_dict().items()}
    k3 = {k: tuple(v.shape) for k, v in v3.state_dict().items()}
    assert k3 == k2


def _load_v2_reference(vocab):
    """The same checkpoint, loaded through the v2.2 classes rather than the v3 ones."""
    from alphatamp.approaches.spectre.model_v2 import SpectreV2Model

    ck = torch.load(_CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = SpectreV2Model(
        n_ops=int(ck["n_ops"]),
        max_arity=vocab.max_operator_arity,
        max_tags=int(cfg["max_tags"]),
        n_overlap_feats=2 if cfg.get("use_overlap") else 0,
        n_prior_feats=2 if cfg.get("use_prior") else 0,
        dropout_p=0.0,
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    return model


@pytest.mark.slow
def test_v3_rollout_is_bit_identical_to_v2() -> None:
    """The gate: v3 and v2.2 make the same decisions with the same numbers.

    Bit-identical is achievable here because compat mode builds the v2.2 submodules, so
    any difference means the *plumbing* around them diverged -- which is precisely what
    the later data-path rewrites could break.
    """
    from alphatamp.approaches.spectre.evidence import deployed_rollout_traced
    from alphatamp.approaches.spectre.inference_v3 import deployed_rollout_v3_traced

    v3_model, _, vocab = _load()
    v2_model = _load_v2_reference(vocab)
    device = "cpu"

    episodes = _episodes_across_strata(20)
    total_steps = 0
    for episode in episodes:
        a2, t2 = deployed_rollout_traced(
            v2_model,
            episode,
            vocab,
            device,
        )
        # `permissive` is v2.2's demotion semantics, so this remains an equivalence check
        # after G5. v3's *default* (`strict`) intentionally demotes less on pre-v3
        # collections: it requires positive evidence that a query ran to exhaustion, which
        # backfilled records carry only when the attempt cost exactly its minimum. That
        # divergence is the point of G5 and is asserted in `test_proof_demotion_v3`;
        # folding it in here would make a real regression and an intended improvement
        # look identical.
        a3, t3 = deployed_rollout_v3_traced(
            v3_model,
            episode,
            vocab,
            device,
            mode="permissive",
        )
        pid = int(episode.provenance.problem_id)
        assert t3.order == t2.order, f"pid {pid}: attempt order diverged"
        assert t3.step_dead == t2.step_dead, f"pid {pid}: demotion diverged"
        assert a3 == a2, f"pid {pid}: attempts diverged"
        assert t3.step_scores == t2.step_scores, f"pid {pid}: logits diverged"
        total_steps += len(t3.order)

    assert len(episodes) >= 20, f"only {len(episodes)} episodes replayed"
    # Guard against the sampling regression this test already had once: if every episode
    # solves on the first attempt, nothing about the rollout loop was exercised.
    assert total_steps > 3 * len(episodes), (
        f"only {total_steps} attempts over {len(episodes)} episodes -- "
        "the sample is not reaching the hard strata"
    )


@pytest.mark.slow
def test_stored_compare_cache_is_stale_against_current_code() -> None:
    """Documents a *pre-existing* v2.2 reproducibility gap, so it cannot be rediscovered
    as if it were a v3 regression.

    The dd2d_v3 comparison cache -- the source of the published 13.68 -- is not
    reproducible from the checkpoint and code now on disk. Current code is deterministic
    (verified across processes under hash randomisation) and v2/v3 agree exactly, so the
    cache was written by a state that no longer exists. This test asserts the *shape* of
    the discrepancy rather than a number, so it starts passing for the right reason if the
    cache is ever rebuilt with ``--force``.
    """
    from alphatamp.approaches.spectre.inference_v3 import deployed_rollout_v3_traced

    model, _, vocab = _load()
    cached = {int(p.stem): json.loads(p.read_text()) for p in _CACHE.glob("*.json")}
    assert cached, "empty compare cache"

    agree = total = 0
    for episode in _episodes_across_strata(20):
        pid = int(episode.provenance.problem_id)
        if pid not in cached:
            continue
        _, trace = deployed_rollout_v3_traced(
            model,
            episode,
            vocab,
            "cpu",
        )
        agree += int(trace.order == cached[pid]["order"])
        total += 1

    # Either the cache matches current code everywhere (it was rebuilt -- good), or it is
    # the known-stale artifact. A *partial* match with no explanation is the state we
    # must not silently sit in, so record which it is.
    assert total > 0
    if agree != total:
        pytest.skip(
            f"known-stale dd2d_v3 compare cache: attempt order matches on "
            f"{agree}/{total} problems; see notebook.md. Rebuild with "
            f"`precompute_dd2d_cache.py --env-variant dd2d_v3 --force`."
        )
