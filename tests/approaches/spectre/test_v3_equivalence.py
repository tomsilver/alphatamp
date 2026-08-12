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

This oracle was the backbone for the gates that follow: the domain adapter and the record
tokens rewrite the data path underneath it, and any change that moves a decision fails
here loudly instead of drifting a mean.

**Rollout bit-identity retired 2026-08-08.** Deployed v3 now narrows the scene relation to
the anchor-free ``[area, sinθ, cosθ]`` triple (``V3Config.d_rel = 3``) where v2.2 reads a
width-8 target-anchored vector, so a *deployed* v3 rollout is no longer bit-identical to
v2.2 by design -- ``build_v3_example`` emits a narrower scene than ``build_v2_example``.
What survives, because it still can: v2.2 checkpoints load into a **compat-mode**
``SpectreV3Model`` (``d_rel = 8``, the width v2.2 was trained on), the shared submodule
structure still matches at that width, and a forward pass over the *same* width-8 batch is
still bit-identical between the v2.2 classes and compat-v3. That is the plumbing guard the
data-path rewrites need; only the deployed-width rollout comparison is gone. See
docs/decisions 2026-08-08.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "spectre"
_TEST_SPLIT = _DATA / "raw" / "dd2d_v3" / "test"
_VOCAB = _DATA / "derived" / "dd2d_v3" / "train_vocab.json"
_CKPT = _DATA / "checkpoints_v2_evidence_ov" / "dd2d_v3" / "seed_0" / "best.pt"

pytestmark = pytest.mark.skipif(
    not (_CKPT.is_file() and _VOCAB.is_file() and _TEST_SPLIT.is_dir()),
    reason="dd2d_v3 artifacts absent (gitignored data not present)",
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


def test_compat_v3_state_dict_keys_match_v2() -> None:
    """Tripwire against an accidental submodule rename, at the compat scene width.

    Renaming ``scene``/``cands``/``facts``/``scorer``/``aux`` breaks checkpoint loading
    for every stored v2.2 run at once, and the symptom (slightly worse numbers) looks
    like a modelling result rather than a bug. The v3 side is built at ``d_rel=8`` -- the
    width v2.2 was trained on -- because that is the config ``load_v2_checkpoint`` uses to
    reload a frozen v2.2 checkpoint; the *deployed* v3 (``d_rel=3``) deliberately differs
    and is checked by :func:`test_deployed_v3_narrows_the_scene`.
    """
    from alphatamp.approaches.spectre.model_v2 import D_REL, SpectreV2Model
    from alphatamp.approaches.spectre.model_v3 import SpectreV3Model, V3Config

    kwargs = dict(n_ops=4, max_arity=1)
    v2 = SpectreV2Model(**kwargs, n_overlap_feats=2, n_prior_feats=0)
    v3 = SpectreV3Model(
        **kwargs, cfg=V3Config(n_overlap_feats=2, n_prior_feats=0, d_rel=D_REL)
    )
    k2 = {k: tuple(v.shape) for k, v in v2.state_dict().items()}
    k3 = {k: tuple(v.shape) for k, v in v3.state_dict().items()}
    assert k3 == k2


def test_deployed_v3_narrows_the_scene() -> None:
    """The deployed default is the narrowed scene, and it is really narrower than v2.2.

    The narrowing must be the *default* -- v3 should not have to opt in to dropping the
    target-anchored columns. A regression that reverted ``V3Config.d_rel`` to 8 would
    silently restore the DD2D-target assumption; this fails if it does.
    """
    from alphatamp.approaches.spectre.model_v2 import D_REL_V3
    from alphatamp.approaches.spectre.model_v3 import SpectreV3Model, V3Config

    assert V3Config().d_rel == D_REL_V3 == 3
    v3 = SpectreV3Model(n_ops=4, max_arity=1, cfg=V3Config(n_overlap_feats=2))
    assert v3.scene.d_rel == 3
    assert v3.scene.rel_proj.in_features == 3


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
def test_compat_v3_forward_is_bit_identical_to_v2() -> None:
    """The surviving plumbing gate: the shared submodules compute the same logits.

    The old rollout-equivalence test drove ``deployed_rollout_v3_traced``, which now
    tensorizes the *narrowed* (width-3) scene and so can no longer reproduce v2.2. This
    replaces it at the forward-pass level and at the compat width: load the same v2.2
    checkpoint through the v2.2 classes and through a compat-mode ``SpectreV3Model``
    (``d_rel=8``), feed both the *same* width-8 ``build_v2_example`` batch, and demand
    bit-identical logits. Any divergence means the plumbing around the shared submodules
    drifted -- exactly what the data-path rewrites could break -- without depending on the
    deployed tensorizer that intentionally diverged.
    """
    from alphatamp.approaches.spectre.dataset_v2 import build_v2_example, collate_v2

    v3_model, _, vocab = _load()  # compat-mode v3, d_rel=8
    v2_model = _load_v2_reference(vocab)

    episodes = _episodes_across_strata(20)
    assert len(episodes) >= 20, f"only {len(episodes)} episodes replayed"
    for episode in episodes:
        ex = build_v2_example(episode, vocab, rng=None, evidence=False)
        batch = collate_v2([ex], max_arity=vocab.max_operator_arity)
        with torch.no_grad():
            l2, _ = v2_model(batch)
            l3, _ = v3_model(batch)
        pid = int(episode.provenance.problem_id)
        assert torch.equal(l2, l3), f"pid {pid}: compat-v3 logits diverged from v2.2"
