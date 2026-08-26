"""Training loop for SPECTRE v3.

Same objective as v2.2 -- listwise Plackett-Luce over the pool, plus the same loss
restricted to within plan-length buckets so length cannot be used as a shortcut. What
changes is the evidence pathway (v3 failure-record tokens instead of five bespoke fact
types) and the checkpoint selector.

**Selection is deployed-val-FP, not `relrank`.** v2.2 selected on a
difficulty-normalized rank statistic that turned out to be miscalibrated on dd2d_v3
(never below 1, i.e. never better than random) and could pick an underfit epoch. v3
selects on the quantity actually reported: mean failed attempts before the first
success, from the real deployed rollout on val -- model scores plus sound demotion.
Three guards, each from a specific failure:

- the **rule used for selection is frozen** at ``permissive`` regardless of the mode a
  gate deploys with, so a change to the demotion rule cannot silently move the selector
  underneath a comparison;
- selection uses a **3-epoch moving average**, because a single val pass is noisy and
  ``argmin`` over 30 epochs is a maximization-biased estimator;
- selection is **uncensored and over the whole val split**, because a budget is a
  ceiling on the statistic and the models differ in the tail above it. G6 shipped with
  the selector censored at 30 attempts over a 50-episode subsample: it scored v2.2 at
  11.12 and v3 at 11.40 -- indistinguishable -- while the same two models were 4+ FP
  apart uncensored on test, because s2/s3 episodes routinely need 30-40+ attempts and
  every one was clipped to the same number. A selector blind to the region where models
  differ ranks epochs by noise. Cheaper recoveries if this ever costs too much: run it
  every K epochs, or uncensored on a stride -- never censored below the separating tail.

The aux head is *not* trained: no collection populates ``aux_labels``, so v2.2's masked
BCE contributed exactly zero, and necessity conditioning was cut from v3
(``decisions.md`` 2026-07-26). Pretending to train it would be theatre.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alphatamp.approaches.spectre.dataset import (
    atom_emission,
    build_example,
    collate,
    pointset_emission,
    sample_context,
)
from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.encoders import D_REL
from alphatamp.approaches.spectre.inference import deployed_rollout_traced
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.loss import plackett_luce_loss, within_length_pl_loss
from alphatamp.approaches.spectre.model import (
    N_OVERLAP_COV,
    SpectreConfig,
    SpectreModel,
)
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`run_training`, persisted into every checkpoint."""

    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 8
    max_tags: int = 32
    dropout_p: float = 0.1
    augment: bool = True
    seed: int = 0
    warmup_epochs: int = 2
    within_length_weight: float = 1.0
    use_overlap: bool = True
    use_records: bool = True
    select_window: int = 3
    # Uncensored, whole-split selection. `select_budget=None` means "run to the pool
    # cap", the same convention reporting uses (`decisions.md` 2026-06-07). See module
    # docstring for why censoring here was silently fatal.
    val_episodes: int = 100
    select_budget: Optional[int] = None
    num_workers: int = 4
    # "both" | "jaccard" | "dead" | "none" -- which cand_overlap columns the net sees.
    # Dropping `dead` is C5 hygiene (the sound rule stays outside the net as demotion);
    # see the note in `dataset.build_example`.
    overlap_mode: str = "both"
    # Collapse a candidate's failures to one record per (schema, args). The refiner
    # emits one per failed *sample*, which lets one unlucky candidate contribute
    # hundreds of tokens; §6.1 defines a record per failing *query*.
    aggregate_records: bool = False
    # Emit proof-tier ∧ provable records into the token stream (default True = the
    # historical holdout at `dataset.build_record_arrays`). The learned-pathway
    # workstream (docs/failed_records_fix.md P-1) sets this False so a tokens-only arm
    # sees the certificate-grade records (DD2D `retrieve`) the `dead`/`coverage` scalars
    # already read via `unified_evidence`. Emission-only (no submodule); the deployed
    # scalars-on recipe leaves it True, where the holdout is near-harmless de-duplication.
    record_holdout: bool = True
    # Rung-1 learned pathway (docs/failed_records_fix.md F-A/F-B2). `record_mode="steps"`
    # emits + consumes the evidence-STEP stream (failed step + culprit establishing steps,
    # shared-encoder embedded) instead of the summary record tokens; `use_step_join` adds
    # the pre-pooling StepJoin so candidate step tokens can join against that memory. Both
    # default off/"summary" (byte-identical). Changes both what `build_example` emits and
    # the architecture, so both are persisted and rebuilt at load time.
    record_mode: str = "summary"
    use_step_join: bool = False
    # F-B1: exact match-primitive edge biases on the step-join's attention (candidate-step ×
    # record-token equality indicators, zero-init learned gates). Requires use_step_join;
    # reads no compiled scalar. Architectural (adds gate params), so persisted + rebuilt at
    # load time; off = byte-identical.
    step_join_match_bias: bool = False
    # Separate cross-attention channel for evidence (EvidenceCrossAttentionScorer).
    evidence_attn: bool = False
    # Observed coverage/waste on cand_overlap; the s3 signal `dead` was proxying for.
    coverage_feats: bool = False
    # Which of the pair the net sees, by zeroing the other column: both | coverage |
    # waste. They have only ever been measured together, so this isolates them.
    coverage_mode: str = "both"
    # `repeat` overlap column: the F3 exact-step certificate veto (fires on a candidate
    # containing the exact failed step of a blameless, exhausted failure of a
    # `step_certificate` schema). The deployed restock3d_v3 feature (2026-08-21): it
    # revived an otherwise-inert adaptivity, adaptive 3.13 vs the coverage-only 12.18,
    # ~97% of the P2 oracle ceiling. Additive + trailing (off reproduces the prior state
    # dict); inert where no schema is `step_certificate` (DD2D/SB2D). See
    # docs/adaptivity_probe_plan_restock3d_v3.md.
    repeat_feats: bool = False
    # DEPRECATED, off by default, do NOT enable -- to be removed in a later refactor.
    # `regroup` = F2 seating-chart co-occurrence; the P2 decomposition priced it at ~1% of
    # the headroom and the retrain confirmed it adds nothing over `repeat` on v3
    # (+repeat+regroup 3.19 ≈ +repeat 3.13). Gated by `grouping_certificate` so it stays
    # inert (not wrong-polarity) on DD2D/SB2D, but it is not part of any deployed recipe.
    regroup_feats: bool = False
    # §6.1's `s_j`: each record token also carries the abstract state at its failing
    # step,
    # as the delta from s_0 (which atoms the prefix added, which it deleted).
    use_state_delta: bool = False
    # Scene-relation width, persisted so ``load_checkpoint`` reloads the right shape.
    # Fixed at the deployed 3 (the anchor-free triple); a field only so it round-trips
    # through the saved cfg and a checkpoint predating the narrowing (no key -> 8-wide)
    # fails to load rather than scoring the un-narrowed model. See docs/decisions
    # 2026-08-08.
    d_rel: int = D_REL
    # 3D scene representation (Restock3D): emit an analytic point cloud + (x,y,z,yaw)
    # pose instead of the 2D boundary ring + (x,y,theta). Drives both what
    # ``build_example`` emits (``scene_3d``) and the model's ``point_dim``/``pose_dim``
    # (3/4 vs 2/3), so data and encoder widths agree. Default OFF keeps DD2D/SB2D
    # byte-identical. See docs/decisions 2026-08-18.
    scene_3d: bool = False
    # PointSetEncoder upgrade switches (doc pointset_encoder_upgrade.md). All default off
    # / seeds=1 so the default recipe stays byte-identical (the v1 FootprintEncoder
    # path). ``use_pca_feats``/``edgeconv_k`` also change what ``build_example`` emits
    # (C_pt / knn), so all five are persisted via ``asdict(cfg)`` and mapped into
    # ``SpectreConfig`` at both train and load time. See docs/decisions 2026-08-18.
    use_pca_feats: bool = False
    use_edgeconv: bool = False
    use_point_sab: bool = False
    pma_seeds: int = 1
    edgeconv_k: int = 0  # 0 -> resolve by dim (4 in 2D, 6 in 3D)
    # Atom-input switches (doc spectre_atom_input_guide.md). ``atom_mode="profiles"``
    # feeds the init-state + goal atoms to the model (per-object profiles). Changes both
    # what ``build_example`` emits and the architecture, so all three are persisted via
    # ``asdict(cfg)`` and mapped into ``SpectreConfig`` at train and load time. Default
    # "off" keeps every env byte-identical.
    atom_mode: str = "off"
    use_init_atoms: bool = True
    use_goal_atoms: bool = True
    # Failure-context mass. v2.2's defaults put ~35% of examples at |F|=0 and dropped
    # evidence from 30% of the rest, so >half of training carries no evidence -- while a
    # deployed rollout sees |F|=0 exactly ONCE per episode and |F|>0 for every attempt
    # after it. Over-weighting the static case is the same rollout-alignment error as
    # the |F| cap, on the other axis.
    p_empty: float = 0.35
    p_drop_facts: float = 0.3
    # G9: restrict the *training* split to these strata (empty = all). This is
    # experiment design, not a model input -- C2 bans stratum as an input or a
    # test-time gate, and this is neither: it decides which episodes exist during
    # training, exactly as the proposal's "train s0-s2, deploy s3" protocol (§7.4 A4)
    # requires.
    train_strata: tuple[int, ...] = ()
    # The env_variant being trained. Only needed so the ``train_strata`` filter decodes a
    # problem id with the RIGHT banding: ``restock3d_v2`` bands five strata per split, not
    # the shared four, so ``_keep`` must route its ``stratum_of`` through the v2 decoder.
    # Persisted as metadata; default matches ``--env``.
    env: str = "dd2d_v4"
    # Weight averaging for lower-variance deployment. "none" | "ema". This is a training
    # *process* lever, not an input or architecture switch: it changes which weights are
    # saved, never what the model contains or what `build_example` emits. OFF ("none")
    # never constructs the EMA shadow and takes the current code path byte-for-byte (the
    # D-8 exact-absence discipline), so `weight_avg="none"` runs are bit-identical to
    # pre-change training. Added 2026-08-08 to recover the domain-agnostic
    # (narrowed-input) model's across-seed variance without touching inputs/architecture
    # -- the removed scene columns were inference-inert (probe Δ0.00) and the best
    # narrowed seed matches the baseline, so the gap is optimization variance, which EMA
    # targets directly.
    weight_avg: str = "none"
    # Per-optimizer-step EMA decay; effective averaging window ~ 1/(1-decay) steps.
    # 0.999 over the post-warmup tail is a fine-grained local average in the single
    # basin the cosine-to-zero LR settles into. On a tiny dataset (few steps/epoch, e.g.
    # SB2D) drop toward 0.99 if the EMA barely separates from the raw model.
    ema_decay: float = 0.999
    # Epoch at which the EMA shadow is (re-)seeded from the live weights and updates
    # begin. Default = warmup_epochs so the shadow never averages in the random init or
    # the high-LR warmup iterates.
    ema_start_epoch: int = 2
    # F-C2 rollout-aligned |F| curriculum (`decisions.md` 2026-08-23). "uniform"
    # (default) is the historical `sample_context` draw -- byte-for-byte unchanged.
    # "rollout" reshapes the training context size to each episode's deployment visit
    # distribution, Uniform{0..phi_e}, where phi_e is a reference policy's deployed FP on
    # that episode, read from `phi_path` keyed by problem_id. Purely a *training-time*
    # context sampler: it changes neither the model nor what `build_example` emits nor
    # the deploy/eval path, so it adds no `state_dict` keys and old checkpoints are
    # unaffected. A missing problem_id in the map falls back to the uniform draw.
    context_mode: str = "uniform"
    phi_path: str = ""
    # X2 (docs/failed_records_fix_part2.md §3): train the record/evidence channel as a
    # zero-init |F|-gated residual on a frozen, warm-started pure-static trunk.
    # `residual_adaptive` selects `ResidualEvidenceScorer` (architecture; persisted in cfg and
    # rebuilt at load). `init_static_from` is a pure-static checkpoint whose matching submodules
    # are copied in at construction; `freeze_static` then sets requires_grad=False on exactly
    # those loaded params so only the residual trains. The last two are training-process levers
    # (not architecture): they change which weights are initialized/updated, never the state
    # dict shape, so they need no load-time round-trip.
    residual_adaptive: bool = False
    init_static_from: str = ""
    freeze_static: bool = False


class SpectreDataset(Dataset):
    """Episodes -> ``(SpectreExample, record arrays)``.

    Loads **raw** and lets ``build_example`` canonicalize once. That is not
    incidental: ``canonicalize_episode`` is not idempotent, and feeding it an
    already-canonical episode silently changes the object->tag binding (the bug that
    skewed every cached comparison number until 2026-07-26).
    """

    def __init__(
        self,
        split_dir: Path,
        vocab: Vocab,
        cfg: TrainConfig,
        spec: Optional[DomainSpec] = None,
    ) -> None:
        self.vocab = vocab
        self.cfg = cfg
        self.spec = spec
        self.epoch = 0
        self._paths = [
            p
            for p in list_episodes(split_dir)
            if _keep(load_episode(p), cfg.train_strata, cfg.env)
        ]
        # F-C2: per-episode reference-policy FP (phi_e), keyed by problem_id, used only
        # when cfg.context_mode == "rollout". Loaded once; a problem_id absent from the
        # map -> phi=None -> the uniform draw (graceful fallback).
        self._phi: dict[int, int] = {}
        if cfg.context_mode == "rollout" and cfg.phi_path:
            with open(cfg.phi_path, "r", encoding="utf-8") as f:
                self._phi = {int(k): int(v) for k, v in json.load(f).items()}

    @property
    def paths(self) -> list[Path]:
        """Episode paths, in split order.

        Public because the selector needs to *stride* this list rather than take a
        prefix: the collector fills strata in seed bands, so a prefix is the easy half.
        """
        return self._paths

    def set_epoch(self, epoch: int) -> None:
        """Reseed the per-epoch F-subset sampling, so each epoch draws new contexts."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        episode = load_episode(self._paths[idx])
        spec = self.spec or spec_for(episode.provenance.env_variant)
        rng = np.random.default_rng((self.cfg.seed, idx, self.epoch))
        fail_idx = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        phi = (
            self._phi.get(int(episode.provenance.problem_id))
            if self.cfg.context_mode == "rollout"
            else None
        )
        ctx, hide = sample_context(
            fail_idx,
            rng,
            p_empty=self.cfg.p_empty,
            p_drop_facts=self.cfg.p_drop_facts,
            phi=phi,
        )
        _ps, _pca, _k = pointset_emission(self.cfg, self.cfg.scene_3d)
        _ai, _ag = atom_emission(self.cfg)
        example, records = build_example(
            episode,
            self.vocab,
            rng=rng,
            max_tags=self.cfg.max_tags,
            evidence=True,
            context_f=ctx,
            hide_facts=hide,
            augment_tags=self.cfg.augment,
            spec=spec,
            overlap_mode=self.cfg.overlap_mode,
            aggregate_records=self.cfg.aggregate_records,
            coverage_feats=self.cfg.coverage_feats,
            coverage_mode=self.cfg.coverage_mode,
            repeat_feats=self.cfg.repeat_feats,
            regroup_feats=self.cfg.regroup_feats,
            state_delta=self.cfg.use_state_delta,
            record_holdout=self.cfg.record_holdout,
            record_mode=self.cfg.record_mode,
            scene_3d=self.cfg.scene_3d,
            pointset_feats=_ps,
            use_pca_feats=_pca,
            edgeconv_k=_k,
            emit_init_atoms=_ai,
            emit_goal_atoms=_ag,
        )
        if not self.cfg.use_records:
            records = []
        return example, records


def _trainable(ep) -> bool:
    return (
        ep.scene_geometry is not None
        and ep.summary.num_success >= 1
        and len(ep.skeleton_pool) >= 2
    )


def _claim_out_dir(out_dir: Path) -> None:
    """Refuse to start if another live run already owns this checkpoint directory.

    Two runs of the same arm silently interleave their writes to ``best.pt``, so the
    file ends up from whichever finished last and the checkpoint's provenance is
    unrecoverable.
    That happened during the 2026-07-27 push: a relaunch after a crash left two
    processes on one path, and the same config scored 8.57 then 8.39 as the second
    overwrote the first. The conclusion survived because the config was identical; it
    would not have if the arms had differed.

    A stale marker (owner no longer alive) is reclaimed rather than fatal, so a killed
    run does not block the directory forever.
    """
    marker = out_dir / ".owner"
    if marker.is_file():
        try:
            owner = int(marker.read_text().strip())
            os.kill(owner, 0)  # signal 0 = liveness probe, sends nothing
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale or unreadable -> reclaim
        else:
            raise RuntimeError(
                f"{out_dir} is already being written by pid {owner}. Two runs sharing "
                f"a checkpoint dir produce a best.pt of unrecoverable provenance. "
                f"Use a different --out-suffix, or stop that run first."
            )
    marker.write_text(str(os.getpid()))
    atexit.register(lambda: marker.unlink(missing_ok=True))


def _keep(ep, strata: tuple[int, ...], env_variant: str | None = None) -> bool:
    """``_trainable`` plus the optional G9 stratum restriction on the training split."""
    if not _trainable(ep):
        return False
    if not strata:
        return True
    from alphatamp.approaches.spectre.compare import stratum_of

    return stratum_of(int(ep.provenance.problem_id), env_variant) in strata


def _make_collate(max_arity: int, max_pred_arity: int = 1):
    def _collate(items):
        examples = [e for e, _ in items]
        records = [r for _, r in items]
        return collate(
            examples,
            max_arity=max_arity,
            records=records,
            max_pred_arity=max_pred_arity,
        )

    return _collate


@torch.no_grad()
def deployed_val_fp(
    model: SpectreModel,
    episodes: list,
    vocab: Vocab,
    device: str,
    spec: DomainSpec,
    max_tags: int,
    budget: Optional[int] = None,
    overlap_mode: str = "both",
    aggregate_records: bool = False,
    coverage_feats: bool = False,
    coverage_mode: str = "both",
    repeat_feats: bool = False,
    regroup_feats: bool = False,
    state_delta: bool = False,
    record_holdout: bool = True,
    record_mode: str = "summary",
) -> float:
    """Mean failed attempts before first success, on the real deployed loop.

    The selector measures exactly what is deployed: a purely learned ranker, with no
    proof-demotion (cut from the method on 2026-07-30). ``budget=None`` runs to the pool
    cap, i.e. uncensored -- the same convention reporting uses, and the only setting
    under which this statistic can see the s2/s3 tail where models actually differ.
    """
    model.eval()
    fps = []
    for ep in episodes:
        attempts, _ = deployed_rollout_traced(
            model,
            ep,
            vocab,
            device,
            spec=spec,
            max_tags=max_tags,
            max_attempts=budget,
            overlap_mode=overlap_mode,
            aggregate_records=aggregate_records,
            coverage_feats=coverage_feats,
            coverage_mode=coverage_mode,
            repeat_feats=repeat_feats,
            regroup_feats=regroup_feats,
            state_delta=state_delta,
            record_holdout=record_holdout,
            record_mode=record_mode,
        )
        fps.append(float(attempts) - 1.0)
    return float(np.mean(fps)) if fps else float("inf")


def _ema_update(ema, model, decay: float) -> None:
    """In-place EMA of ``model``'s weights into ``ema``.

    Float tensors decay toward the live weights; the rare non-float tensor (none exist
    in this LayerNorm-only model today) is copied verbatim so the shadow stays a valid,
    loadable state dict. Called only when EMA is enabled; the ``None`` guard in
    :func:`_run_epoch` keeps the OFF path bit-identical to pre-change training.
    """
    with torch.no_grad():
        for e, p in zip(ema.state_dict().values(), model.state_dict().values()):
            if e.is_floating_point():
                e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
            else:
                e.copy_(p)


def _run_epoch(
    model, loader, device, wl_weight: float, opt=None, ema=None, ema_decay: float = 0.0
) -> float:
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logits, _ = model(batch)
            loss = plackett_luce_loss(logits, batch.success_mask, batch.pool_mask)
            if train and wl_weight and batch.cand_prior is not None:
                loss = loss + wl_weight * within_length_pl_loss(
                    logits,
                    batch.success_mask,
                    batch.pool_mask,
                    batch.cand_prior[:, :, 1],
                )
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                _ema_update(ema, model, ema_decay)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def run_training(
    cfg: TrainConfig,
    train_dir: Path,
    val_dir: Path,
    vocab: Vocab,
    out_dir: Path,
    device: Optional[str] = None,
) -> dict:
    """Train one ranker, writing ``best.pt`` and ``log.jsonl`` under ``out_dir``.

    Returns the run summary (best selection score, epochs, training-set size).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    _claim_out_dir(out_dir)

    probe = load_episode(list_episodes(train_dir)[0])
    spec = spec_for(probe.provenance.env_variant)

    train_ds = SpectreDataset(train_dir, vocab, cfg, spec)
    val_ds = SpectreDataset(val_dir, vocab, cfg, spec)
    collate = _make_collate(vocab.max_operator_arity, vocab.max_predicate_arity)
    # Tensorization is ~79% of a training step (measured) and the model is tiny, so the
    # loader is the bottleneck, not the GPU. `persistent_workers` stays off on purpose:
    # workers are re-forked each epoch and so pick up `set_epoch`, which drives both the
    # tag permutation and the failure-context sampling.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
    )
    # Stride, never truncate. The collector fills strata in seed bands and episodes are
    # stored in seed order, so `[:50]` would hand the selector only strata 0-1 -- the
    # easy half -- and it would happily pick a checkpoint hopeless on s2/s3.
    _val_paths = val_ds.paths
    _stride = max(1, len(_val_paths) // max(cfg.val_episodes, 1))
    val_episodes = [load_episode(p) for p in _val_paths[::_stride]][: cfg.val_episodes]

    model = SpectreModel(
        n_ops=len(vocab.operators),
        max_arity=vocab.max_operator_arity,
        cfg=SpectreConfig(
            n_overlap_feats=(
                (
                    2
                    + (2 if cfg.coverage_feats else 0)
                    + (2 if (cfg.repeat_feats or cfg.regroup_feats) else 0)
                )
                if cfg.use_overlap
                else 0
            ),
            n_prior_feats=0,
            d_rel=cfg.d_rel,
            point_dim=3 if cfg.scene_3d else 2,
            pose_dim=4 if cfg.scene_3d else 3,
            use_pca_feats=cfg.use_pca_feats,
            use_edgeconv=cfg.use_edgeconv,
            use_point_sab=cfg.use_point_sab,
            pma_seeds=cfg.pma_seeds,
            # Raw (0 = auto): the model reads k from the runtime knn tensor, and
            # ``pointset_emission`` resolves 0 -> dim-default the same at train & deploy.
            edgeconv_k=cfg.edgeconv_k,
            max_tags=cfg.max_tags,
            dropout_p=cfg.dropout_p,
            use_records=cfg.use_records,
            evidence_attn=cfg.evidence_attn,
            coverage_feats=cfg.coverage_feats,
            use_state_delta=cfg.use_state_delta,
            n_predicates=len(vocab.predicates),
            max_pred_arity=vocab.max_predicate_arity,
            atom_mode=cfg.atom_mode,
            use_init_atoms=cfg.use_init_atoms,
            use_goal_atoms=cfg.use_goal_atoms,
            record_mode=cfg.record_mode,
            use_step_join=cfg.use_step_join,
            step_join_match_bias=cfg.step_join_match_bias,
            residual_adaptive=cfg.residual_adaptive,
        ),
    ).to(device)
    # X2 warm-start + freeze (docs/failed_records_fix_part2.md §3): load a pure-static
    # checkpoint into the matching (static) submodules, then optionally freeze exactly those
    # params so only the zero-init record/evidence residual trains. The static keys
    # (scene/cands/facts/atoms/aux/scorer.{attn,glob_proj,head}) key-match; the residual keys
    # (records.*/scorer.{evid_attn,adaptive_head,gate}.*) are `missing` and keep their init.
    frozen_keys: set[str] = set()
    if cfg.init_static_from:
        static_sd = torch.load(cfg.init_static_from, map_location=device)["state_dict"]
        own = model.state_dict()
        loadable = {k: v for k, v in static_sd.items() if k in own}
        if len(loadable) != len(static_sd):
            missing_in_model = set(static_sd) - set(own)
            raise ValueError(
                f"warm-start checkpoint has {len(missing_in_model)} keys absent from the "
                f"model (e.g. {sorted(missing_in_model)[:3]}); the static backbone must match"
            )
        incompat = model.load_state_dict(loadable, strict=False)
        frozen_keys = set(loadable)
        print(
            f"[warm-start] {cfg.init_static_from}: loaded {len(loadable)} static keys; "
            f"{len(incompat.missing_keys)} residual keys left at init",
            flush=True,
        )
        if cfg.freeze_static:
            for name, p in model.named_parameters():
                if name in frozen_keys:
                    p.requires_grad_(False)
            n_frozen = sum(1 for n, _ in model.named_parameters() if n in frozen_keys)
            n_train = sum(1 for p in model.parameters() if p.requires_grad)
            print(
                f"[freeze-static] froze {n_frozen} static params; "
                f"{n_train} trainable (residual) params remain",
                flush=True,
            )
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ema_note = (
        f"(decay={cfg.ema_decay},start={cfg.ema_start_epoch})"
        if cfg.weight_avg == "ema"
        else ""
    )
    print(
        f"[train] seed={cfg.seed} device={device} n_train={len(train_ds)} "
        f"n_val={len(val_ds)} epochs={cfg.epochs} records={cfg.use_records} "
        f"overlap={cfg.overlap_mode if cfg.use_overlap else 'off'} "
        f"state_delta={cfg.use_state_delta} "
        f"weight_avg={cfg.weight_avg}"
        f"{ema_note} "
        f"selection=deployed-val-FP(ma{cfg.select_window}, "
        f"n={len(val_episodes)}, "
        f"budget={cfg.select_budget if cfg.select_budget else 'uncensored'})",
        flush=True,
    )

    log: list[dict] = []
    best = float("inf")
    t0 = time.time()
    # EMA shadow: built lazily at `ema_start_epoch` from the post-warmup weights, so it
    # only ever averages in-basin iterates (never the random init / warmup). `None`
    # everywhere when weight_avg != "ema", which keeps the OFF path byte-identical.
    ema_model: Optional[SpectreModel] = None
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = _lr_at(epoch, cfg)
        if cfg.weight_avg == "ema" and epoch == cfg.ema_start_epoch:
            ema_model = copy.deepcopy(model).eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)
        tr = _run_epoch(
            model,
            train_loader,
            device,
            cfg.within_length_weight,
            opt,
            ema=ema_model,
            ema_decay=cfg.ema_decay,
        )
        va = _run_epoch(model, val_loader, device, 0.0, None)

        # Keyword, not positional: the selector must see exactly the inputs training
        # feeds,
        # and a parameter inserted into this list would otherwise shift every switch
        # after
        # it by one -- silently selecting under a different configuration than it
        # trained.
        def _val_fp(m: SpectreModel) -> float:
            # Closes over the epoch's selector config so the raw and EMA passes are
            # scored identically (select what you deploy). Kwargs inlined, not splatted,
            # so mypy checks each against `deployed_val_fp`'s typed signature.
            return deployed_val_fp(
                m,
                val_episodes,
                vocab,
                device,
                spec,
                max_tags=cfg.max_tags,
                budget=cfg.select_budget,
                overlap_mode=cfg.overlap_mode,
                aggregate_records=cfg.aggregate_records,
                coverage_feats=cfg.coverage_feats,
                coverage_mode=cfg.coverage_mode,
                repeat_feats=cfg.repeat_feats,
                regroup_feats=cfg.regroup_feats,
                state_delta=cfg.use_state_delta,
                record_holdout=cfg.record_holdout,
                record_mode=cfg.record_mode,
            )

        fp = _val_fp(model)
        # Select what you deploy: when EMA is on, the EMA weights are the ones that
        # would be shipped, so the selector must score *them* -- not only the raw model.
        # `None` until the shadow exists (epoch < ema_start_epoch).
        fp_ema = _val_fp(ema_model) if ema_model is not None else None
        log.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "val_loss": va,
                "val_fp": fp,
                "val_fp_ema": fp_ema,
            }
        )
        # moving average: a single 100-episode val pass is noisy, and argmin over 30
        # epochs would systematically pick the luckiest one rather than the best model.
        # Keep-the-better: smooth the raw and (when present) the EMA series separately
        # and save whichever weights produced the lower smoothed val_fp. Because both
        # are scored on the same metric, turning EMA on can never select a *worse*
        # checkpoint than off -- it can only help or be inert (the arm's safety
        # property).
        window = [r["val_fp"] for r in log[-cfg.select_window :]]
        smoothed = float(np.mean(window))
        candidates: list[tuple[float, str, SpectreModel]] = [(smoothed, "raw", model)]
        ema_window = [
            r["val_fp_ema"]
            for r in log[-cfg.select_window :]
            if r.get("val_fp_ema") is not None
        ]
        smoothed_ema = float(np.mean(ema_window)) if ema_window else None
        if smoothed_ema is not None and ema_model is not None:
            candidates.append((smoothed_ema, "ema", ema_model))
        cand_val, which, winner = min(candidates, key=lambda c: c[0])
        improved = cand_val < best
        if improved:
            best = cand_val
            torch.save(
                {
                    "state_dict": winner.state_dict(),
                    "cfg": asdict(cfg),
                    "n_ops": len(vocab.operators),
                    "selected": which,
                },
                out_dir / "best.pt",
            )
        if epoch == 0 or epoch == cfg.epochs - 1 or (epoch + 1) % 5 == 0:
            per = (time.time() - t0) / (epoch + 1)
            ema_str = f" ema={fp_ema:.2f}" if fp_ema is not None else ""
            print(
                f"[train] seed={cfg.seed} epoch {epoch + 1}/{cfg.epochs} "
                f"train={tr:.4f} val={va:.4f} val_fp={fp:.2f}{ema_str} "
                f"ma={smoothed:.2f} best={best:.2f}"
                f"{f' *{which}' if improved else ''} | "
                f"{per:.1f}s/ep ETA {per * (cfg.epochs - epoch - 1) / 60:.1f}m",
                flush=True,
            )
    (out_dir / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    return {"best_val_fp": best, "epochs": len(log), "n_train": len(train_ds)}


def _lr_at(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(argv=None) -> int:
    """CLI entry point; see the module docstring for the selection protocol."""
    ap = argparse.ArgumentParser(description="Train the SPECTRE v3 ranker")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--wl-weight", type=float, default=1.0)
    ap.add_argument("--no-records", action="store_true", help="ablate record tokens")
    ap.add_argument("--no-overlap", action="store_true", help="ablate [dead, jaccard]")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--val-episodes",
        type=int,
        default=TrainConfig.val_episodes,
        help="val episodes used by the selector (strided, never truncated)",
    )
    ap.add_argument(
        "--select-budget",
        type=int,
        default=None,
        help="censor the selector's rollout at N attempts; omit for uncensored",
    )
    ap.add_argument(
        "--overlap-mode",
        default="both",
        choices=["both", "jaccard", "dead", "none"],
        help="which cand_overlap columns the net sees; the sound rule is applied "
        "outside the net as demotion regardless",
    )
    ap.add_argument(
        "--p-empty",
        type=float,
        default=TrainConfig.p_empty,
        help="fraction of training examples with an empty failure context",
    )
    ap.add_argument(
        "--p-drop-facts",
        type=float,
        default=TrainConfig.p_drop_facts,
        help="evidence dropout rate on the remaining examples",
    )
    ap.add_argument(
        "--coverage-feats",
        action="store_true",
        help="append observed coverage/waste to cand_overlap (the §5.1 necessity "
        "features, grounded in reported culprits instead of a predicted head)",
    )
    ap.add_argument(
        "--coverage-mode",
        default=TrainConfig.coverage_mode,
        choices=["both", "coverage", "waste"],
        help="which of the coverage/waste pair the net sees; the other column is "
        "zeroed (shape unchanged). Only meaningful with --coverage-feats",
    )
    ap.add_argument(
        "--repeat-feats",
        action="store_true",
        help="append the `repeat` overlap column: a per-candidate veto firing on a "
        "blameless, exhausted failure of a `step_certificate` schema (restock3d F3)",
    )
    ap.add_argument(
        "--regroup-feats",
        action="store_true",
        help="DEPRECATED (do not use; to be removed): the `regroup` F2 seating-chart "
        "column. Adds nothing over --repeat-feats on v3 (~1%% of headroom); off by default",
    )
    ap.add_argument(
        "--evidence-attn",
        action="store_true",
        help="give evidence its own cross-attention channel instead of making it "
        "compete with the scene inside one softmax",
    )
    ap.add_argument(
        "--aggregate-records",
        action="store_true",
        help="one record token per (schema, args) instead of per failed sample",
    )
    ap.add_argument(
        "--no-record-holdout",
        action="store_true",
        help="feed proof-tier AND provable records into the token stream instead of "
        "dropping them (docs/failed_records_fix.md P-1: the corrected tokens-only "
        "baseline). Default keeps the historical holdout.",
    )
    ap.add_argument(
        "--record-mode",
        choices=["summary", "steps"],
        default="summary",
        help="'steps' = rung-1 evidence-step stream (failed step + culprit establishing "
        "steps, shared-encoder embedded) instead of one summary token per record "
        "(docs/failed_records_fix.md F-A).",
    )
    ap.add_argument(
        "--step-join",
        action="store_true",
        help="add the pre-pooling StepJoin (F-B2): candidate step tokens cross-attend "
        "over the evidence memory before pooling, so a step-level join is representable.",
    )
    ap.add_argument(
        "--step-join-match-bias",
        action="store_true",
        help="F-B1: add exact match-primitive edge biases (candidate-step × record-token "
        "equality indicators, zero-init learned gates) to the step-join. Needs --step-join.",
    )
    ap.add_argument(
        "--state-delta",
        action="store_true",
        help="each record token also carries s_j as the delta from s_0 (§6.1): which "
        "atoms the failing prefix added and which it deleted",
    )
    ap.add_argument(
        "--train-strata",
        type=int,
        nargs="*",
        default=[],
        help="restrict the TRAINING split to these strata, e.g. --train-strata 0 1 2",
    )
    ap.add_argument(
        "--weight-avg",
        default=TrainConfig.weight_avg,
        choices=["none", "ema"],
        help="EMA weight averaging for lower-variance deployment; 'none' is the "
        "byte-identical current path",
    )
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=TrainConfig.ema_decay,
        help="per-step EMA decay (only with --weight-avg ema)",
    )
    ap.add_argument(
        "--ema-start-epoch",
        type=int,
        default=TrainConfig.ema_start_epoch,
        help="epoch at which the EMA shadow is seeded (default = warmup_epochs)",
    )
    ap.add_argument(
        "--select-window",
        type=int,
        default=TrainConfig.select_window,
        help=(
            "moving-average window for the val-FP selector; "
            "widen for a jitterier model"
        ),
    )
    ap.add_argument(
        "--context-mode",
        choices=("uniform", "rollout"),
        default=TrainConfig.context_mode,
        help=(
            "F-C2 |F| curriculum: 'uniform' (default, historical [1,max_f] draw) or "
            "'rollout' (per-episode Uniform{0..phi_e} from --phi-path)"
        ),
    )
    ap.add_argument(
        "--phi-path",
        type=str,
        default=TrainConfig.phi_path,
        help=(
            "JSON {problem_id: reference-policy FP} for --context-mode rollout; "
            "build with experiments/spectre/fc2_build_phi.py"
        ),
    )
    ap.add_argument(
        "--residual-adaptive",
        action="store_true",
        help="X2: record/evidence channel as a zero-init |F|-gated residual over a static base "
        "(ResidualEvidenceScorer); pair with --init-static-from/--freeze-static",
    )
    ap.add_argument(
        "--init-static-from",
        type=str,
        default=TrainConfig.init_static_from,
        help="pure-static checkpoint (best.pt) to warm-start the static trunk from",
    )
    ap.add_argument(
        "--freeze-static",
        action="store_true",
        help="freeze the warm-started static params (requires_grad=False); only the residual trains",
    )
    ap.add_argument(
        "--scene-3d",
        action="store_true",
        help="3D point-cloud scene representation (Restock3D); default 2D footprint",
    )
    ap.add_argument(
        "--use-pca-feats",
        action="store_true",
        help="PointSetEncoder: per-point normal/curvature/flatness features",
    )
    ap.add_argument(
        "--use-edgeconv",
        action="store_true",
        help="PointSetEncoder: one DGCNN EdgeConv interaction layer",
    )
    ap.add_argument(
        "--use-point-sab",
        action="store_true",
        help="PointSetEncoder: optional global SAB over the point set (off by default)",
    )
    ap.add_argument(
        "--pma-seeds",
        type=int,
        default=TrainConfig.pma_seeds,
        help="PointSetEncoder: MultiSeedPMA seeds (1 == single-seed pool)",
    )
    ap.add_argument(
        "--edgeconv-k",
        type=int,
        default=TrainConfig.edgeconv_k,
        help="PointSetEncoder: kNN degree (0 = auto: 4 in 2D, 6 in 3D)",
    )
    ap.add_argument(
        "--atom-mode",
        choices=["off", "profiles"],
        default=TrainConfig.atom_mode,
        help="atom input: 'profiles' feeds init-state + goal atoms to the model",
    )
    ap.add_argument(
        "--no-init-atoms",
        action="store_true",
        help="with --atom-mode profiles: drop the initial-abstract-state atoms",
    )
    ap.add_argument(
        "--no-goal-atoms",
        action="store_true",
        help="with --atom-mode profiles: drop the goal atoms",
    )
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args(argv)

    root = Path(a.data_root)
    vocab = Vocab.from_json(root / "derived" / a.env / "train_vocab.json")
    cfg = TrainConfig(
        epochs=a.epochs,
        seed=a.seed,
        lr=a.lr,
        within_length_weight=a.wl_weight,
        use_records=not a.no_records,
        use_overlap=not a.no_overlap,
        num_workers=a.num_workers,
        val_episodes=a.val_episodes,
        select_budget=a.select_budget,
        overlap_mode=a.overlap_mode,
        aggregate_records=a.aggregate_records,
        record_holdout=not a.no_record_holdout,
        record_mode=a.record_mode,
        use_step_join=a.step_join,
        step_join_match_bias=a.step_join_match_bias,
        evidence_attn=a.evidence_attn,
        coverage_feats=a.coverage_feats,
        coverage_mode=a.coverage_mode,
        repeat_feats=a.repeat_feats,
        regroup_feats=a.regroup_feats,
        use_state_delta=a.state_delta,
        p_empty=a.p_empty,
        p_drop_facts=a.p_drop_facts,
        train_strata=tuple(a.train_strata),
        env=a.env,
        weight_avg=a.weight_avg,
        ema_decay=a.ema_decay,
        ema_start_epoch=a.ema_start_epoch,
        select_window=a.select_window,
        context_mode=a.context_mode,
        phi_path=a.phi_path,
        residual_adaptive=a.residual_adaptive,
        # `{seed}` in the static-init path is substituted with this run's seed, so a
        # multi-seed sweep warm-starts each residual seed from the matching static seed.
        init_static_from=a.init_static_from.replace("{seed}", str(a.seed)),
        freeze_static=a.freeze_static,
        scene_3d=a.scene_3d,
        use_pca_feats=a.use_pca_feats,
        use_edgeconv=a.use_edgeconv,
        use_point_sab=a.use_point_sab,
        pma_seeds=a.pma_seeds,
        edgeconv_k=a.edgeconv_k,
        atom_mode=a.atom_mode,
        use_init_atoms=not a.no_init_atoms,
        use_goal_atoms=not a.no_goal_atoms,
    )
    sub = "checkpoints_spectre"
    if a.no_records:
        sub += "_norec"
    if a.no_overlap:
        sub += "_noov"
    if a.atom_mode != "off":
        sub += "_atoms"
    sub += a.out_suffix
    out = root / sub / a.env / f"seed_{a.seed}"
    res = run_training(
        cfg, root / "raw" / a.env / "train", root / "raw" / a.env / "val", vocab, out
    )
    print(f"[train] done: {res} -> {out}/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
