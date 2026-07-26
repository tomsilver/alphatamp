"""v3 tensorizer: ``EpisodeRecord`` -> ``SpectreV3Batch``, via the domain contract.

Structurally this is v2.2's tensorizer with the DD2D literals removed. Two things it no
longer knows:

- **which objects a candidate manipulates** -- was ``op.name == "place-buffer"``, now
  :func:`domain.DomainSpec.manipulated`, verified identical on 120000/120000 skeletons;
- **which failures license demotion** -- was ``failure_action.startswith("retrieve")``,
  now the per-query axiom declaration plus the observation's own budget flag.

Three v2.2 features are deliberately *not* carried over:

- ``exclude_marginal`` was inert twice over (the DD2D refiner only writes ``status`` in
  ``{feasible, infeasible}``, and even when it fired ``collate`` folded the resulting
  ``None`` back to ``False`` because the batch has no per-candidate ignore mask). v3
  declines to carry a flag that silently does nothing; reinstating the behaviour needs a
  real label mask, not a flag.
- ``demotion_source="computed"`` -- the geometry-reconstruction path -- is dropped (R2).
  It was worth a measured ~14%, but it is the last per-environment geometry routine in
  the deployment story, and the observed signal is what generalizes.
- The **short-first prior** as a scorer feature (R1). The plan-length column survives only
  as the within-length loss's bucket key, which is now :func:`domain.length_key`; the
  model sees no prior (``V3Config.n_prior_feats == 0``). It was a per-dataset hand switch
  that diverged training on the easier collection, and note it was never a clean feature
  ablation anyway: enabling it also zero-inits the scorer head.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.dataset_v2 import (
    _V2Example,
    collate_v2,
    resample_ring,
)
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.facts import TIER_IDS, gather_context_facts
from alphatamp.approaches.spectre.model_v2 import D_REL, MAX_FACT_ARGS
from alphatamp.approaches.spectre.model_v3 import SpectreV3Batch
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.tags import assign_tags
from alphatamp.approaches.spectre.vocab import Vocab

__all__ = ["build_v3_example", "collate_v3", "sample_context"]


def sample_context(
    fail_idx: list[int],
    rng: np.random.Generator,
    p_empty: float = 0.35,
    p_drop_facts: float = 0.3,
    max_f: int = 8,
) -> tuple[frozenset[int], bool]:
    """Sample a failure context ``F`` plus an evidence-dropout flag.

    Mass is heavy at ``|F| = 0`` because that is the deployment start: the static pathway
    has to stand on its own before any failure has been observed. ``hide_facts`` drops the
    evidence for an example so the ranker cannot become dependent on it.
    """
    if not fail_idx or rng.random() < p_empty:
        return frozenset(), False
    size = int(rng.integers(1, min(max_f, len(fail_idx)) + 1))
    chosen = rng.choice(np.asarray(fail_idx), size=size, replace=False)
    return frozenset(int(i) for i in chosen), bool(rng.random() < p_drop_facts)


def _hint_fact_arrays(
    episode: EpisodeRecord, context_f: frozenset[int], tags: dict[str, int]
) -> tuple[list[int], list[int], list[list[int]]]:
    """Hint-tier facts of the failed candidates, bound to episode-local tags.

    Proof-tier facts are excluded on purpose. Their sound consequence is applied outside
    the network as demotion; handing them to the scorer as *tokens* invites it to learn
    the crude correlate instead ("blocked sets are large, so prefer longer plans"), which
    measurably harmed the easy strata in v2.2 until the tiers were split.
    """
    hint = TIER_IDS["hint"]
    type_ids: list[int] = []
    tier_ids: list[int] = []
    arg_tags: list[list[int]] = []
    for fact in gather_context_facts(episode, sorted(context_f)):
        if fact.tier_id != hint:
            continue
        type_ids.append(fact.type_id)
        tier_ids.append(fact.tier_id)
        arg_tags.append([tags[a] for a in fact.args if a in tags][:MAX_FACT_ARGS])
    return type_ids, tier_ids, arg_tags


def build_v3_example(
    episode: EpisodeRecord,
    vocab: Vocab,
    rng: Optional[np.random.Generator] = None,
    max_tags: int = 32,
    evidence: bool = False,
    context_f: Optional[frozenset[int]] = None,
    hide_facts: bool = False,
    augment_tags: bool = True,
    spec: Optional[DomainSpec] = None,
) -> _V2Example:
    """Tensorize one geometry-carrying episode for the v3 model.

    ``spec`` defaults to the contract registered for the episode's own ``env_variant``,
    so a caller cannot accidentally tensorize DD2D under another domain's axioms.
    """
    if episode.scene_geometry is None:
        raise ValueError("build_v3_example requires scene_geometry")
    spec = spec or spec_for(episode.provenance.env_variant)

    canon = canonicalize_episode(episode, rng=None)
    geo = canon.scene_geometry
    assert geo is not None
    tags = assign_tags(
        [o.name for o in geo.objects],
        rng=(rng if augment_tags else None),
        max_tags=max_tags,
    )

    # --- scene tokens -------------------------------------------------------
    frame = geo.frame or {}
    scale = max(
        float(frame.get("drawer_w", 0.0)), float(frame.get("drawer_d", 0.0)), 1.0
    )
    target = next((o for o in geo.objects if o.is_target), None)
    tx, ty = (target.pose[0], target.pose[1]) if target else (0.0, 0.0)

    n_obj = len(geo.objects)
    obj_tags = np.array([tags[o.name] for o in geo.objects], dtype=np.int64)
    obj_boundary = np.stack([resample_ring(list(o.boundary)) for o in geo.objects])
    obj_pose = np.array(
        [[o.pose[0] / scale, o.pose[1] / scale, o.pose[2]] for o in geo.objects],
        dtype=np.float32,
    )
    obj_is_target = np.array(
        [1.0 if o.is_target else 0.0 for o in geo.objects], dtype=np.float32
    )
    rel = np.zeros((n_obj, D_REL), dtype=np.float32)
    for i, o in enumerate(geo.objects):
        dx, dy = (o.pose[0] - tx) / scale, (o.pose[1] - ty) / scale
        rel[i, :6] = [dx, dy, math.hypot(dx, dy), o.area, *_sin_cos(o.pose[2])]
        rel[i, 6] = 1.0 if o.concave else 0.0
        rel[i, 7] = float(o.area) / (target.area if target else 1.0)

    # --- candidate tokens ---------------------------------------------------
    goal_objs = spec.goal_objects(canon)
    op_ids: list[list[int]] = []
    arg_tags: list[list[list[int]]] = []
    success: list[Optional[bool]] = []
    subsets: list[frozenset[str]] = []
    lengths: list[int] = []
    for skel, out in zip(canon.skeleton_pool, canon.outcomes):
        op_ids.append(
            [int(vocab.operators.get(op.name, 0)) for op in skel.operator_seq]
        )
        arg_tags.append(
            [[tags.get(a.name, 0) for a in op.parameters] for op in skel.operator_seq]
        )
        subsets.append(spec.manipulated(skel, goal_objs))
        lengths.append(spec.length_key(skel))
        success.append(out.outcome == "success")

    # Aux targets stay -1 (= ignore) until the necessity labeller lands: no collection
    # has ever populated `aux_labels`, so v2.2's aux head was masked out entirely and
    # never received a gradient. Pretending otherwise here would silently train it.
    aux = np.full(n_obj, -1.0, dtype=np.float32)

    # --- failure context ----------------------------------------------------
    k = len(op_ids)
    fail_idx = [i for i, out in enumerate(canon.outcomes) if out.outcome == "fail"]
    hide = hide_facts
    if context_f is not None:
        ctx: frozenset[int] = frozenset(int(i) for i in context_f)
    elif evidence and rng is not None:
        ctx, hide = sample_context(fail_idx, rng)
    else:
        ctx = frozenset()
    avail = [i not in ctx for i in range(k)]

    if hide or not ctx:
        ftype: list[int] = []
        ftier: list[int] = []
        farg: list[list[int]] = []
    else:
        ftype, ftier, farg = _hint_fact_arrays(canon, ctx, tags)

    # --- planner signals + structural evidence ------------------------------
    # Column 0 (enumeration order) is inert: the v3 scorer takes no prior features. Column
    # 1 carries the normalized plan length, kept only because the within-length PL loss
    # buckets on it; it is `domain.length_key` rescaled, so the partition is the domain's.
    max_len = max(lengths) if lengths else 1
    prior = [[-(i / max(k - 1, 1)), -(lengths[i] / max(max_len, 1))] for i in range(k)]

    overlap = [[0.0, 0.0] for _ in range(k)]
    if ctx and not hide:
        blocked = [subsets[f] for f in ctx if spec.licenses_demotion(canon.outcomes[f])]
        failed = [subsets[f] for f in ctx]
        for i, si in enumerate(subsets):
            dead = 1.0 if any(si <= b for b in blocked) else 0.0
            jaccard = max(
                (len(si & f) / max(len(si | f), 1) for f in failed), default=0.0
            )
            overlap[i] = [dead, float(jaccard)]

    return _V2Example(
        obj_tags,
        obj_boundary,
        obj_pose,
        rel,
        obj_is_target,
        op_ids,
        arg_tags,
        success,
        aux,
        aux.copy(),
        avail,
        ftype,
        ftier,
        farg,
        prior,
        overlap,
    )


def _sin_cos(theta: float) -> tuple[float, float]:
    return math.sin(theta), math.cos(theta)


def collate_v3(examples: list[_V2Example], max_arity: int) -> SpectreV3Batch:
    """Pad + stack examples into a batch the v3 model consumes."""
    return collate_v2(examples, max_arity=max_arity)
