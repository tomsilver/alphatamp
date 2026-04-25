"""Test-time inference helper per ``SPECTRE_RT2D_METHOD_SPEC.md`` §10.5.

Usage:

    state = init_inference_state(model, episode, vocab, prior)
    while pool remaining:
        idx = select_next_skeleton(state, model)
        outcome = ...        # consult the episode's pre-recorded outcome
        if outcome.success:
            break
        record_failure(state, idx)

The episode-start cost is one batched Φ forward over the K candidates;
subsequent steps run Ψ over a set of size ``len(fail_indices)`` plus a
broadcasted σ. For ``K ≤ 30`` and ``t ≤ 30`` both are trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.dataset import (
    SpectreTrainingExample,
    collate_spectre_batch,
)
from alphatamp.approaches.spectre.model import SpectreModel
from alphatamp.approaches.spectre.priors import BasePrior, ZeroPrior
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class InferenceState:
    """Per-episode state — encoded pool, priors, pool mask, fail history."""

    e_S: Tensor  # (K, D) — episode-start Φ embeddings
    priors: Tensor  # (K,)
    pool_mask: Tensor  # (K,) bool — True for slots still in R
    fail_indices: list[int] = field(default_factory=list)


def init_inference_state(
    model: SpectreModel,
    episode: EpisodeRecord,
    vocab: Vocab,
    prior: BasePrior | None = None,
    device: torch.device | str = "cpu",
) -> InferenceState:
    """Encode every skeleton in the episode pool once.

    Uses **deterministic canonicalization** (no augmentation) — this is the
    test-time contract per spec §4.5.

    Skeletons whose outcome is ``"error"`` are excluded from the inference
    pool (their slots in ``pool_mask`` start False).
    """
    if prior is None:
        prior = ZeroPrior()

    # Canonicalize once with rng=None so local ids match the training-time
    # eval-mode ordering (alphabetical within each type).
    ep_view = canonicalize_episode(episode, rng=None, type_aug_policy=None)

    # Build a single SpectreTrainingExample-shaped object whose ``r_skeletons``
    # is the full pool. ``f_skeletons`` is empty — we only need Φ here, so
    # the F-side tensors will collate to width 1 and never be consumed.
    pool = ep_view.skeleton_pool
    error_indices = {o.skeleton_idx for o in episode.outcomes if o.outcome == "error"}
    priors_list: list[float] = []
    for j, skel in enumerate(pool):
        priors_list.append(
            float(prior.score(episode.provenance.problem_id, j, skel, episode))
        )

    example = SpectreTrainingExample(
        problem_id=episode.provenance.problem_id,
        initial_abstract_state=ep_view.initial_abstract_state,
        goal_atoms=ep_view.goal_atoms,
        object_registry=ep_view.object_registry,
        r_skeletons=pool,
        r_priors=tuple(priors_list),
        r_success_mask=tuple(False for _ in pool),  # not consumed at inference
        f_skeletons=(),
    )
    batch = collate_spectre_batch([example], vocab)

    # Move tensors to device.
    def _to(t: Tensor) -> Tensor:
        return t.to(device)

    model.eval()
    with torch.no_grad():
        e_R = model.encode_pool(
            _to(batch.r_op_ids),
            _to(batch.r_op_arg_type_ids),
            _to(batch.r_op_arg_local_ids),
            _to(batch.r_op_mask),
            _to(batch.s0_pred_ids),
            _to(batch.s0_arg_type_ids),
            _to(batch.s0_arg_local_ids),
            _to(batch.s0_atom_mask),
            _to(batch.s0_type_histogram),
            _to(batch.r_sL_pred_ids),
            _to(batch.r_sL_arg_type_ids),
            _to(batch.r_sL_arg_local_ids),
            _to(batch.r_sL_atom_mask),
        )  # (1, K, D)
    e_S = e_R[0].detach()
    priors_t = torch.tensor(priors_list, dtype=torch.float32, device=e_S.device)
    pool_mask = torch.ones(len(pool), dtype=torch.bool, device=e_S.device)
    for idx in error_indices:
        pool_mask[idx] = False
    return InferenceState(
        e_S=e_S, priors=priors_t, pool_mask=pool_mask, fail_indices=[]
    )


def select_next_skeleton(state: InferenceState, model: SpectreModel) -> int:
    """Return the argmax-index over the remaining pool (spec §10.5)."""
    device = state.e_S.device
    if state.fail_indices:
        f_emb = state.e_S[state.fail_indices].unsqueeze(0)  # (1, |F|, D)
        f_mask = torch.ones(1, len(state.fail_indices), dtype=torch.bool, device=device)
    else:
        # Send a synthetic 1-token "empty" set; the context encoder routes
        # to ``c_0`` via the all-False mask check.
        f_emb = torch.zeros(
            1, 1, state.e_S.size(-1), device=device, dtype=state.e_S.dtype
        )
        f_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        c = model.encode_context(f_emb, f_mask)  # (1, D)
        e_R = state.e_S.unsqueeze(0)  # (1, K, D)
        priors = state.priors.unsqueeze(0)  # (1, K)
        logits = model.score(e_R, c, priors, prior_dropout=False)  # (1, K)
        neg_inf = torch.tensor(-float("inf"), dtype=logits.dtype, device=device)
        logits = torch.where(state.pool_mask.unsqueeze(0), logits, neg_inf)
        idx = int(logits.argmax(dim=-1).item())
    return idx


def record_failure(state: InferenceState, skeleton_idx: int) -> None:
    """Move ``skeleton_idx`` from R into F (in place)."""
    if not state.pool_mask[skeleton_idx].item():
        raise ValueError(
            f"skeleton {skeleton_idx} is not in the remaining pool"
            f" (already attempted or excluded as error)"
        )
    state.fail_indices = list(state.fail_indices) + [int(skeleton_idx)]
    new_mask = state.pool_mask.clone()
    new_mask[skeleton_idx] = False
    state.pool_mask = new_mask


# Re-export for convenience; downstream code can ``from inference import *``.
__all__ = [
    "InferenceState",
    "init_inference_state",
    "select_next_skeleton",
    "record_failure",
]
