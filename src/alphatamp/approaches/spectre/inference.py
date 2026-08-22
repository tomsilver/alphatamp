"""The SPECTRE deployed ranker: a purely learned listwise re-ranker over the pool.

The loop is the deployment story in five lines: score the pool with the failures
observed so far, mask the already-tried candidates, try the argmax, observe the outcome,
stop on the first success. Nothing outside the network touches the ordering — proof-
tier demotion was cut from the method on 2026-07-30 (``decisions.md``); v3 is a purely
learned ranker.

The per-step trace exists because the comparison cache stores it: persisting the raw
logits lets the analysis notebook show what the ranker thought at every step without
ever running inference at load time.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from alphatamp.approaches.spectre.dataset import (
    atom_emission,
    build_example,
    collate,
    pointset_emission,
)
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.encoders import D_REL
from alphatamp.approaches.spectre.model import (
    N_OVERLAP_COV,
    SpectreConfig,
    SpectreModel,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.vocab import Vocab

# Sentinel applied to already-attempted candidates before the argmax, so a candidate is
# never re-tried within a rollout.
_TRIED = -1e9


def _zero_scene_columns(batch, cols: frozenset[str]):
    """Zero a whole scene channel in place -- a deploy-time diagnostic.

    Mirrors ``suppress_records``: it feeds a trained model a *null* version of an input
    it was trained on, to price how much the deployed model leans on that channel.
    ``"is_goal"`` blanks the goal-membership boolean; ``"rel"`` blanks the anchor-free
    ``obj_rel`` triple ``[area, sinθ, cosθ]``. Not a deployment mode: it only measures
    reliance. Batch tensors are rebuilt every step, so mutating them never leaks across
    steps. (An earlier form of this hook, tied to the pre-narrowing width-8 ``obj_rel``,
    priced the removal of the target-anchored columns before they were cut -- see the
    Step-0 measurement in docs/notebook 2026-08-08.)
    """
    if "is_goal" in cols:
        batch.obj_is_goal.zero_()
    if "rel" in cols:
        batch.obj_rel.zero_()
    return batch


def load_checkpoint(
    ckpt: Path | str, vocab: Vocab, device: str = "cpu"
) -> tuple[SpectreModel, dict]:
    """Rebuild a trained v3 model, with dropout off, plus its **deploy kwargs**.

    The second return value is the set of feature switches that change what
    :func:`build_example` *emits* rather than what the model *contains*, so they are
    invisible to ``load_state_dict`` and a mismatch fails silently instead of loudly:
    deploying under a different ``overlap_mode`` (or ``coverage_mode``) than a model
    trained under feeds it a column it has never seen populated, or blanks one it relies
    on. Reading them back off the checkpoint — never accepting them from the caller — is
    what makes that unrepresentable. Splat the dict into
    :func:`deployed_rollout_traced` / :func:`build_example`.

    Switches that *do* change the architecture (``use_records``, ``evidence_attn``, and
    ``coverage_feats`` via the ``cand_overlap`` width) are rebuilt into ``SpectreConfig``
    here, where ``strict=True`` catches any error. Older checkpoints predate several of
    these keys, hence ``.get``.
    """
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = SpectreModel(
        n_ops=int(ck["n_ops"]),
        max_arity=vocab.max_operator_arity,
        cfg=SpectreConfig(
            n_overlap_feats=(
                (
                    2
                    + (2 if cfg.get("coverage_feats") else 0)
                    + (
                        2
                        if (cfg.get("repeat_feats") or cfg.get("regroup_feats"))
                        else 0
                    )
                )
                if cfg.get("use_overlap")
                else 0
            ),
            n_prior_feats=0,
            # Scene-relation width is bound to the checkpoint: deployed v3 is 3, and a
            # checkpoint predating the narrowing has no key and was 8-wide. `strict=True`
            # below is the backstop -- a wrong width fails to load rather than silently
            # scoring the un-narrowed model.
            d_rel=int(cfg.get("d_rel", D_REL)),
            # Scene point/pose widths are bound to the checkpoint via its ``scene_3d``
            # flag (3/4 for a Restock3D point cloud, else the 2D 2/3). ``strict=True``
            # below is the backstop: a wrong width fails to load rather than silently
            # feeding a 3D-trained encoder 2D tensors. Older checkpoints: no key -> 2D.
            point_dim=3 if cfg.get("scene_3d") else 2,
            pose_dim=4 if cfg.get("scene_3d") else 3,
            # PointSetEncoder switches (doc pointset_encoder_upgrade.md). They select the
            # PointSetEncoder submodule over v1's FootprintEncoder, so `strict=True` is
            # the backstop. Older checkpoints have no keys -> all off / seeds=1 -> the v1
            # path loads byte-identically. ``use_pca_feats``/``edgeconv_k`` also change
            # what ``build_example`` emits; they are threaded into the tensorizer from
            # ``model.cfg`` in ``deployed_rollout_traced`` via ``pointset_emission``.
            use_pca_feats=bool(cfg.get("use_pca_feats")),
            use_edgeconv=bool(cfg.get("use_edgeconv")),
            use_point_sab=bool(cfg.get("use_point_sab")),
            pma_seeds=int(cfg.get("pma_seeds", 1)),
            edgeconv_k=int(cfg.get("edgeconv_k", 0)),
            max_tags=int(cfg.get("max_tags", 32)),
            dropout_p=0.0,
            use_records=bool(cfg.get("use_records")),
            evidence_attn=bool(cfg.get("evidence_attn")),
            coverage_feats=bool(cfg.get("coverage_feats")),
            use_state_delta=bool(cfg.get("use_state_delta")),
            n_predicates=len(vocab.predicates),
            max_pred_arity=vocab.max_predicate_arity,
            # Atom-input switches (doc spectre_atom_input_guide.md). They select the
            # AtomProfileEncoder submodule, so `strict=True` is the backstop. Older
            # checkpoints have no key -> "off" -> nothing built -> byte-identical load.
            # Emission is threaded into the tensorizer from `model.cfg` in
            # `deployed_rollout_traced` via `atom_emission`, like the pointset switches.
            atom_mode=str(cfg.get("atom_mode", "off")),
            use_init_atoms=bool(cfg.get("use_init_atoms", True)),
            use_goal_atoms=bool(cfg.get("use_goal_atoms", True)),
        ),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval().to(device), {
        "overlap_mode": str(cfg.get("overlap_mode", "both")),
        "aggregate_records": bool(cfg.get("aggregate_records")),
        "coverage_feats": bool(cfg.get("coverage_feats")),
        "coverage_mode": str(cfg.get("coverage_mode", "both")),
        # Emitted-only (no architectural submodule): the width is the single point the
        # scorer Linear cares about, and it is recomputed above from the same flags.
        "repeat_feats": bool(cfg.get("repeat_feats")),
        "regroup_feats": bool(cfg.get("regroup_feats")),
        # Emitted-only: whether the tensorizer drops proof-tier ∧ provable records from
        # the token stream. `.get(..., True)` keeps every pre-key checkpoint on the
        # historical holdout, so only a model trained with `--no-record-holdout` deploys
        # without it (docs/failed_records_fix.md P-1).
        "record_holdout": bool(cfg.get("record_holdout", True)),
        # Architectural *and* emitted: the encoder needs the submodules and the
        # tensorizer
        # needs to produce the arrays, so it appears in both places -- exactly as
        # `coverage_feats` does -- with the checkpoint as the single source of truth.
        "state_delta": bool(cfg.get("use_state_delta")),
    }


@dataclasses.dataclass(frozen=True)
class Trace:
    """Step-aligned record of one rollout; one entry per attempt made.

    ``step_scores`` are the **raw** model logits, before the tried-mask. Raw on purpose:
    the tried sentinels would swamp a rendered score column, and the effective row is
    exactly reconstructible from ``order``. Entries for candidates already in the
    failure context come back ``-inf`` from the model's own availability mask, so at
    step ``t`` the non-finite entries are exactly ``order[:t]``; a JSON serialiser must
    map them to ``null``.

    ``step_dead`` is retained as an always-empty ``[]`` per step so the stored cache
    JSON schema is unchanged; proof-tier demotion was cut from the method (2026-07-30),
    so no candidate is ever demoted.
    """

    order: list[int]
    step_scores: list[list[float]]
    step_dead: list[list[int]]
    infer_seconds: float = 0.0
    """Wall-clock spent on inference across the rollout: per-step tensorization
    (``build_example`` + ``collate``) + the model forward, summed.

    Defaulted so callers that ignore timing are unaffected; the timing bracket includes
    the device sync, so on cuda it is a true end-to-end measure. Warm the model up once
    before a timed pass so one-time CUDA init does not land in the first step.
    """

    refine_capped_seconds: float = 0.0
    """Refinement wall-clock along the realized order, each candidate's stored
    ``refinement_wall_clock_s`` clamped to ``refine_cap_s`` (uncapped sum when no cap).

    Reuses the per-candidate refiner times stored on the episode; the rollout must
    accumulate it here (rather than a caller summing ``_refine_seconds`` over ``order``)
    because a capped rollout's order can contain a *slow-feasible* candidate that did
    not stop the loop -- a plain "sum to first success" would break there and
    undercount. 0.0 when the episode carries no per-candidate times.
    """


@torch.no_grad()
def deployed_rollout_traced(
    model: SpectreModel,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    spec: Optional[DomainSpec] = None,
    max_tags: int = 32,
    max_attempts: Optional[int] = None,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
    coverage_feats: bool = False,
    coverage_mode: str = "both",
    repeat_feats: bool = False,
    regroup_feats: bool = False,
    state_delta: bool = False,
    record_holdout: bool = True,
    suppress_records: bool = False,
    zero_scene_cols: frozenset[str] = frozenset(),
    refine_cap_s: Optional[float] = None,
) -> tuple[int, Trace]:
    """Run the deployed ranker; return ``(attempts_to_first_success, trace)``.

    ``attempts`` is 1-indexed (the rollout FP reported downstream is ``attempts - 1``).
    ``spec`` defaults to the contract registered for the episode's own ``env_variant``.

    ``max_attempts`` censors the rollout at a fixed budget. Reporting always runs
    uncensored -- the budget equals the pool cap, so it never binds -- and censoring
    exists only for *checkpoint selection*, where the metric is recomputed every epoch
    and the full loop otherwise costs several times the training step it is selecting
    over. This mirrors the split the project already runs: selection under a budget,
    reporting without one.

    The ordering is purely the model's: proof-tier demotion was cut from the deployed
    method on 2026-07-30 (``decisions.md``), so nothing outside the network reorders the
    pool. ``Trace.step_dead`` is emitted as an always-empty list per step to keep the
    stored cache schema unchanged.

    ``suppress_records=True`` is a **diagnostic**, not a deployment mode: it runs a
    records-trained model with its evidence memory emptied at every step. Deliberately a
    train/deploy mismatch, and useful precisely because of that -- it separates "training
    with records damaged the weights" (still bad with records suppressed) from "the
    evidence input misleads at deploy" (good with them suppressed). Never report a number
    produced with it as a method result.

    ``zero_scene_cols`` is the geometry analogue of ``suppress_records`` and is likewise
    a **diagnostic**: it blanks a scene channel at deploy to price how much the model
    leans on it. ``"is_goal"`` blanks the goal-membership boolean; ``"rel"`` blanks the
    anchor-free ``obj_rel`` triple. A small FP delta means the channel is close to inert
    for ranking; a large one means it is load-bearing. Never a method number.

    ``refine_cap_s`` models a **per-candidate refinement-abandonment cap**: a deployment
    that bounds each skeleton's refinement at ``refine_cap_s`` seconds before moving on.
    A feasible candidate whose stored ``refinement_wall_clock_s`` exceeds the cap is
    then *not* a stopping success -- it is abandoned and observed like any other
    failure (so it enters the failure context and re-ranks the pool), and the loop
    continues. This only reorders the *ranking*, never removes a plan (P-E holds): the
    pool is still exhausted in order, so a problem is lost only if every feasible
    candidate exceeds the cap. ``Trace.refine_capped_seconds`` accumulates the
    wall-clock the capped deployment pays. ``None`` (default) is the uncapped rollout.
    """
    model.eval()
    spec = spec or spec_for(episode.provenance.env_variant)
    # The scene representation is a property of the checkpoint: a model with a 3D scene
    # encoder (``point_dim == 3``) must be fed 3D examples. Derived from the model's own
    # persisted config so inference cannot desync from what the weights expect.
    scene_3d = getattr(model.cfg, "point_dim", 2) == 3
    # PointSetEncoder emission (doc pointset_encoder_upgrade.md): derived from the
    # model's own config so the tensorizer produces exactly the per-point features/kNN
    # the weights were trained on. Older checkpoints -> (False, False, k): nothing sent.
    _ps_feats, _ps_pca, _ps_k = pointset_emission(model.cfg, scene_3d)
    # Atom-input emission (doc spectre_atom_input_guide.md): derived from the model's own
    # config so the tensorizer emits exactly the atoms the weights were trained on. Older
    # checkpoints -> (False, False): nothing sent.
    _emit_init, _emit_goal = atom_emission(model.cfg)
    n_candidates = len(episode.skeleton_pool)

    def _stops(o) -> bool:
        # A candidate ends the rollout only if it refines *and* does so within the cap;
        # a slow-feasible candidate over the cap is abandoned and treated as a failure.
        if o.outcome != "success":
            return False
        if refine_cap_s is None:
            return True
        return float(o.refinement_wall_clock_s or 0.0) <= refine_cap_s

    success = {i for i, o in enumerate(episode.outcomes) if _stops(o)}
    tried: list[int] = []
    step_scores: list[list[float]] = []
    step_dead: list[list[int]] = []
    infer_seconds = 0.0
    refine_capped_seconds = 0.0

    budget = n_candidates if max_attempts is None else min(max_attempts, n_candidates)
    while len(tried) < budget:
        _t_infer = time.perf_counter()
        example, records = build_example(
            episode,
            vocab,
            rng=None,
            max_tags=max_tags,
            evidence=True,
            context_f=frozenset(tried),
            augment_tags=False,
            spec=spec,
            overlap_mode=overlap_mode,
            aggregate_records=aggregate_records,
            coverage_feats=coverage_feats,
            coverage_mode=coverage_mode,
            repeat_feats=repeat_feats,
            regroup_feats=regroup_feats,
            state_delta=state_delta,
            record_holdout=record_holdout,
            scene_3d=scene_3d,
            pointset_feats=_ps_feats,
            use_pca_feats=_ps_pca,
            edgeconv_k=_ps_k,
            emit_init_atoms=_emit_init,
            emit_goal_atoms=_emit_goal,
        )
        # Records are passed at deployment too, not just in training. Omitting them here
        # would deploy a records-trained model blind to its own evidence -- the train/
        # deploy input mismatch the proposal warns about, and one that degrades silently.
        batch = collate(
            [example],
            max_arity=vocab.max_operator_arity,
            records=[[] if suppress_records else records],
            max_pred_arity=vocab.max_predicate_arity,
        ).to(device)
        if zero_scene_cols:
            batch = _zero_scene_columns(batch, zero_scene_cols)
        logits, _ = model(batch)
        raw = logits[0].detach().cpu().numpy().astype(float)
        # end-to-end inference time: tensorize + collate + forward. The .cpu() above
        # already forces a device sync; synchronize() is a defensive no-op making the
        # bracket a true wall-clock even if that copy is ever removed.
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        infer_seconds += time.perf_counter() - _t_infer
        step_scores.append([float(x) for x in raw])
        step_dead.append([])

        row = raw.copy()
        if tried:
            row[tried] = _TRIED
        pick = int(np.argmax(row))
        tried.append(pick)
        _t_pick = float(episode.outcomes[pick].refinement_wall_clock_s or 0.0)
        refine_capped_seconds += (
            _t_pick if refine_cap_s is None else min(_t_pick, refine_cap_s)
        )
        if pick in success:
            break

    return len(tried), Trace(
        order=list(tried),
        step_scores=step_scores,
        step_dead=step_dead,
        infer_seconds=infer_seconds,
        refine_capped_seconds=round(refine_capped_seconds, 6),
    )


def deployed_rollout(
    model: SpectreModel,
    episode: EpisodeRecord,
    vocab: Vocab,
    device: str,
    spec: Optional[DomainSpec] = None,
    max_tags: int = 32,
) -> int:
    """Attempts to first success.

    See :func:`deployed_rollout_traced` for the trace.
    """
    attempts, _ = deployed_rollout_traced(model, episode, vocab, device, spec, max_tags)
    return attempts
