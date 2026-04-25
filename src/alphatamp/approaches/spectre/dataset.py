"""PyTorch Dataset + collate for SPECTRE.

Online F-subset sampling (pipeline spec §11): ``__getitem__`` loads an episode,
samples ``F ⊆ FAIL_e``, and returns a structured training example. Tensorization
happens in ``collate_spectre_batch`` so the Dataset output stays inspectable.

Per ``SPECTRE_RT2D_METHOD_SPEC.md`` this module now exposes:

- :class:`FSamplingConfig` — four sampling modes per §8.2 (fix #4). Default
  ``rollout_aligned_mix`` weights ``(0.25, 0.25, 0.5)`` on
  ``(uniform_subsets, uniform_size, log_normal)``.
- ``num_f_samples_per_epoch`` — F-subsample multiplier per §8.1 (fix #5).
  ``__len__`` becomes ``num_episodes * num_f_samples_per_epoch``; each
  ``__getitem__`` decomposes the linear index into ``(episode_idx,
  f_sample_idx)`` and seeds its RNG from ``(seed, episode_idx,
  f_sample_idx, epoch)`` so that successive epochs see different ``(F, aug)``
  pairs for the same episode.
- ``type_aug_policy`` — per-type augmentation policy per §4.6 (fix #2),
  threaded through to :func:`canonicalize_episode`.
- Per-skeleton ``s_L`` atom tensors — the spec §4.1 ``SkeletonInput`` carries
  both ``s_0`` and ``s_L`` atom token sequences so Φ encodes the start- and
  end-state of the skeleton.

**Critical invariant** (pipeline spec §11.6; violated in prior Attempt 2): F
must contain only failed skeletons, never successes. We assert this inside
``__getitem__`` so a regression fails loudly rather than silently corrupting
training.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundAtom
from torch.utils.data import Dataset

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.priors import BasePrior
from alphatamp.approaches.spectre.schema import EpisodeRecord, SkeletonRecord
from alphatamp.approaches.spectre.vocab import Vocab

_F_SAMPLING_MODES = (
    "uniform_subsets",
    "uniform_size",
    "log_normal",
    "rollout_aligned_mix",
)


@dataclass(frozen=True)
class FSamplingConfig:
    """Configuration for F-subset sampling modes (spec §8.2)."""

    mode: str = "rollout_aligned_mix"
    # Mixture weights for ``rollout_aligned_mix``: (uniform_subsets,
    # uniform_size, log_normal). Default puts log_normal in the lead because
    # it is the only component matched to the test-time visit shape.
    mix_weights: tuple[float, float, float] = (0.25, 0.25, 0.5)
    log_normal_mu: float = 0.0
    log_normal_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in _F_SAMPLING_MODES:
            raise ValueError(
                f"FSamplingConfig.mode={self.mode!r} not in {_F_SAMPLING_MODES}"
            )
        weight_sum = sum(self.mix_weights)
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(
                f"FSamplingConfig.mix_weights must sum to 1.0; got {weight_sum}"
            )
        if any(w < 0 for w in self.mix_weights):
            raise ValueError(
                "FSamplingConfig.mix_weights must be non-negative;"
                f" got {self.mix_weights}"
            )


FSampling = Union[FSamplingConfig, str]


def _sample_f_subset(
    fail_indices: list[int],
    rng: np.random.Generator,
    config: FSamplingConfig,
) -> frozenset[int]:
    """Draw ``F ⊆ FAIL_e`` per ``config`` (spec §8.2)."""
    n = len(fail_indices)
    if n == 0:
        return frozenset()
    if config.mode == "rollout_aligned_mix":
        sub_mode = str(
            rng.choice(
                ["uniform_subsets", "uniform_size", "log_normal"],
                p=list(config.mix_weights),
            )
        )
    else:
        sub_mode = config.mode
    if sub_mode == "uniform_subsets":
        keep = rng.random(size=n) < 0.5
        return frozenset(int(idx) for idx, k in zip(fail_indices, keep) if k)
    if sub_mode == "uniform_size":
        size = int(rng.integers(0, n + 1))
    elif sub_mode == "log_normal":
        raw = float(
            rng.lognormal(mean=config.log_normal_mu, sigma=config.log_normal_sigma)
        )
        size = max(0, min(int(round(raw)), n))
    else:  # pragma: no cover — exhaustiveness guard
        raise NotImplementedError(sub_mode)
    chosen = rng.choice(
        np.asarray(fail_indices, dtype=np.int64), size=size, replace=False
    )
    return frozenset(int(c) for c in chosen.tolist())


@dataclass
class SpectreTrainingExample:
    """One ``(R, SUCC ∩ R, F)`` triple plus everything the collate needs.

    Kept as a plain Python dataclass, not tensors, so ``__getitem__`` output is
    easy to pretty-print and unit-test.
    """

    problem_id: int
    initial_abstract_state: RelationalAbstractState  # post-canonicalization
    goal_atoms: frozenset[GroundAtom]
    object_registry: dict[str, str]
    r_skeletons: tuple[SkeletonRecord, ...]
    r_priors: tuple[float, ...]
    r_success_mask: tuple[bool, ...]  # True where the skeleton actually succeeded
    f_skeletons: tuple[SkeletonRecord, ...]


class SpectreDataset(Dataset[SpectreTrainingExample]):
    """Torch Dataset over a split's raw episode files.

    Filters out non-trainable episodes at init per pipeline spec §11.5:
    - ``num_skeletons < 2`` (nothing to rank)
    - ``num_success == 0`` (PL loss undefined)

    Error-outcome skeletons are excluded from both ``R`` and ``F`` at sample
    time (pipeline spec §5.6).

    Each episode contributes ``num_f_samples_per_epoch`` distinct training
    examples per epoch (spec §8.1 fix #5). The trainer should call
    :meth:`set_epoch` once per epoch so the per-call RNG produces fresh
    ``(F, augmentation)`` pairs.
    """

    def __init__(
        self,
        split_dir: Path,
        prior: BasePrior,
        seed: int,
        f_sampling: FSampling = "rollout_aligned_mix",
        augment: bool = True,
        type_aug_policy: dict[str, bool] | None = None,
        num_f_samples_per_epoch: int = 1,
        episode_cache_size: int = 64,
    ) -> None:
        self._split_dir = split_dir
        self._prior = prior
        self._seed = seed
        if isinstance(f_sampling, str):
            f_sampling = FSamplingConfig(mode=f_sampling)
        self._f_sampling = f_sampling
        self._augment = augment
        self._type_aug_policy = dict(type_aug_policy) if type_aug_policy else {}
        if num_f_samples_per_epoch < 1:
            raise ValueError("num_f_samples_per_epoch must be >= 1")
        self._num_f_samples_per_epoch = num_f_samples_per_epoch
        self._epoch = 0

        all_paths = list_episodes(split_dir)
        self._episode_paths: list[Path] = []
        self._filtered: list[tuple[int, str]] = []

        for p in all_paths:
            ep = load_episode(p)
            summary = ep.summary
            if summary.num_skeletons < 2:
                self._filtered.append((ep.provenance.problem_id, "num_skeletons<2"))
                continue
            if summary.num_success == 0:
                self._filtered.append((ep.provenance.problem_id, "num_success==0"))
                continue
            self._episode_paths.append(p)

        # LRU-cache the episode loader so repeated __getitem__ calls on the
        # same episode (across epochs / workers) skip gzip+pickle work.
        @lru_cache(maxsize=episode_cache_size)
        def _cached(path_str: str) -> EpisodeRecord:
            return load_episode(Path(path_str))

        self._load_cached = _cached

    def __len__(self) -> int:
        return len(self._episode_paths) * self._num_f_samples_per_epoch

    @property
    def num_episodes(self) -> int:
        """Number of distinct episodes (before the F-sample multiplier)."""
        return len(self._episode_paths)

    @property
    def num_f_samples_per_epoch(self) -> int:
        """Distinct ``(R, F)`` examples drawn per episode per epoch (spec §8.1)."""
        return self._num_f_samples_per_epoch

    @property
    def filtered_problem_ids(self) -> list[tuple[int, str]]:
        """``(problem_id, reason)`` tuples for episodes excluded at init."""
        return list(self._filtered)

    def set_epoch(self, epoch: int) -> None:
        """Advance the per-call RNG seed so each epoch sees fresh ``(F, aug)``."""
        self._epoch = int(epoch)

    def _decompose_index(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return divmod(index, self._num_f_samples_per_epoch)

    def _rng_for(self, episode_idx: int, f_sample_idx: int) -> np.random.Generator:
        """Deterministic per-(seed, episode, f-sample, epoch) RNG.

        The epoch term ensures successive epochs draw fresh F-subsets and
        augmentation permutations, even though the linear ``__getitem__``
        index is identical (the DataLoader's shuffling does not change the
        seed itself, only the order of indices visited).
        """
        return np.random.default_rng(
            (self._seed, episode_idx, f_sample_idx, self._epoch)
        )

    def __getitem__(self, index: int) -> SpectreTrainingExample:
        episode_idx, f_sample_idx = self._decompose_index(index)
        ep = self._load_cached(str(self._episode_paths[episode_idx]))
        rng = self._rng_for(episode_idx, f_sample_idx)

        succ = set(ep.success_indices())
        fail = set(ep.fail_indices())
        errs = set(ep.error_indices())

        f_indices = _sample_f_subset(sorted(fail), rng, self._f_sampling)
        r_indices = succ | (fail - f_indices)

        # Invariants (I8–I11, pipeline spec §11.6).
        assert f_indices.issubset(
            fail
        ), "I8 violated: F must contain only failed skeletons"
        assert r_indices.issuperset(succ), "I9 violated: R must include all successes"
        assert r_indices.isdisjoint(f_indices), "I10 violated: R and F must be disjoint"
        assert len(r_indices) + len(f_indices) + len(errs) == len(
            ep.skeleton_pool
        ), "I11 violated: R ∪ F ∪ ERRS must partition the full pool"

        # Canonicalize; augmentation applies a random within-type permutation
        # gated by ``type_aug_policy`` (RT2D pins width/size/zone/passage).
        ep_view = canonicalize_episode(
            ep,
            rng=rng if self._augment else None,
            type_aug_policy=self._type_aug_policy if self._augment else None,
        )

        r_sorted = sorted(r_indices)
        f_sorted = sorted(f_indices)

        r_skeletons = tuple(ep_view.skeleton_pool[i] for i in r_sorted)
        r_priors = tuple(
            float(self._prior.score(ep.provenance.problem_id, i, r_skeletons[j], ep))
            for j, i in enumerate(r_sorted)
        )
        r_success_mask = tuple(i in succ for i in r_sorted)
        f_skeletons = tuple(ep_view.skeleton_pool[i] for i in f_sorted)

        return SpectreTrainingExample(
            problem_id=ep.provenance.problem_id,
            initial_abstract_state=ep_view.initial_abstract_state,
            goal_atoms=ep_view.goal_atoms,
            object_registry=ep_view.object_registry,
            r_skeletons=r_skeletons,
            r_priors=r_priors,
            r_success_mask=r_success_mask,
            f_skeletons=f_skeletons,
        )


# ---------------------------------------------------------------------------
# Collate: structured examples → padded tensors
# ---------------------------------------------------------------------------


@dataclass
class SpectreBatch:
    """One collated training batch.

    Shapes follow ``SPECTRE_RT2D_METHOD_SPEC.md`` §10.2. All integer id tensors
    use 0 = ``<OOV>`` / padding; the mask tensors distinguish real tokens
    from pads. Per spec §10.2 footnote and AS-BUILT §3.7, ``s_0`` is stored
    once per example and replicated per-skeleton at model-input time;
    ``s_L`` is stored per-skeleton because it varies across the pool.
    """

    # R-pool operator tokens
    r_op_ids: torch.Tensor  # (B, R, L)              long
    r_op_arg_type_ids: torch.Tensor  # (B, R, L, A)           long
    r_op_arg_local_ids: torch.Tensor  # (B, R, L, A)           long
    r_op_mask: torch.Tensor  # (B, R, L)              bool
    r_mask: torch.Tensor  # (B, R)                 bool
    r_priors: torch.Tensor  # (B, R)                 float
    r_success_mask: torch.Tensor  # (B, R)                 bool

    # R-pool s_L atom tokens (per-skeleton)
    r_sL_pred_ids: torch.Tensor  # (B, R, ML)             long
    r_sL_arg_type_ids: torch.Tensor  # (B, R, ML, P)          long
    r_sL_arg_local_ids: torch.Tensor  # (B, R, ML, P)          long
    r_sL_atom_mask: torch.Tensor  # (B, R, ML)             bool

    # F-pool operator tokens (same schema as R, minus priors/success)
    f_op_ids: torch.Tensor  # (B, F, L)              long
    f_op_arg_type_ids: torch.Tensor  # (B, F, L, A)           long
    f_op_arg_local_ids: torch.Tensor  # (B, F, L, A)           long
    f_op_mask: torch.Tensor  # (B, F, L)              bool
    f_mask: torch.Tensor  # (B, F)                 bool

    # F-pool s_L atom tokens (per-skeleton)
    f_sL_pred_ids: torch.Tensor  # (B, F, ML)             long
    f_sL_arg_type_ids: torch.Tensor  # (B, F, ML, P)          long
    f_sL_arg_local_ids: torch.Tensor  # (B, F, ML, P)          long
    f_sL_atom_mask: torch.Tensor  # (B, F, ML)             bool

    # s_0: per-example, replicated per-skeleton at model-input time
    s0_pred_ids: torch.Tensor  # (B, M0)                long
    s0_arg_type_ids: torch.Tensor  # (B, M0, P)             long
    s0_arg_local_ids: torch.Tensor  # (B, M0, P)             long
    s0_atom_mask: torch.Tensor  # (B, M0)                bool
    s0_type_histogram: torch.Tensor  # (B, T)                 long

    problem_ids: torch.Tensor  # (B,)                   long

    metadata: dict = field(default_factory=dict)


def _local_id(obj_name: str) -> int:
    """Parse ``"{type}_{idx}"`` back into its within-type index."""
    return int(obj_name.rsplit("_", 1)[1])


def _encode_operator(
    op, vocab: Vocab, max_arity: int
) -> tuple[int, list[int], list[int]]:
    """Return ``(op_idx, arg_type_ids, arg_local_ids)``, padded to ``max_arity``."""
    op_idx = vocab.op_idx(op.name)
    type_ids = [vocab.type_idx(p.type.name) for p in op.parameters]
    local_ids = [_local_id(p.name) + 1 for p in op.parameters]  # +1 so 0 = pad
    pad = max_arity - len(op.parameters)
    if pad > 0:
        type_ids = type_ids + [0] * pad
        local_ids = local_ids + [0] * pad
    return op_idx, type_ids, local_ids


def _encode_atom(
    atom, vocab: Vocab, max_arity: int
) -> tuple[int, list[int], list[int]]:
    pred_idx = vocab.pred_idx(atom.predicate.name)
    type_ids = [vocab.type_idx(e.type.name) for e in atom.entities]
    local_ids = [_local_id(e.name) + 1 for e in atom.entities]
    pad = max_arity - len(atom.entities)
    if pad > 0:
        type_ids = type_ids + [0] * pad
        local_ids = local_ids + [0] * pad
    return pred_idx, type_ids, local_ids


def _encode_skeleton(
    skel: SkeletonRecord,
    vocab: Vocab,
    max_skel_len: int,
    max_op_arity: int,
) -> tuple[list[int], list[list[int]], list[list[int]], list[bool]]:
    op_ids: list[int] = []
    arg_types: list[list[int]] = []
    arg_locals: list[list[int]] = []
    mask: list[bool] = []
    for op in skel.operator_seq:
        oi, ti, li = _encode_operator(op, vocab, max_op_arity)
        op_ids.append(oi)
        arg_types.append(ti)
        arg_locals.append(li)
        mask.append(True)
    while len(op_ids) < max_skel_len:
        op_ids.append(0)
        arg_types.append([0] * max_op_arity)
        arg_locals.append([0] * max_op_arity)
        mask.append(False)
    return op_ids, arg_types, arg_locals, mask


def _encode_atom_set(
    atoms: Iterable[GroundAtom],
    vocab: Vocab,
    max_atoms: int,
    max_pred_arity: int,
) -> tuple[list[int], list[list[int]], list[list[int]], list[bool]]:
    pred_ids: list[int] = []
    type_ids: list[list[int]] = []
    local_ids: list[list[int]] = []
    mask: list[bool] = []
    for atom in atoms:
        pi, ti, li = _encode_atom(atom, vocab, max_pred_arity)
        pred_ids.append(pi)
        type_ids.append(ti)
        local_ids.append(li)
        mask.append(True)
    while len(pred_ids) < max_atoms:
        pred_ids.append(0)
        type_ids.append([0] * max_pred_arity)
        local_ids.append([0] * max_pred_arity)
        mask.append(False)
    return pred_ids, type_ids, local_ids, mask


def collate_spectre_batch(
    batch: list[SpectreTrainingExample],
    vocab: Vocab,
) -> SpectreBatch:
    """Pad a list of examples to tensors keyed to the Φ input spec."""
    b = len(batch)
    if b == 0:
        raise ValueError("collate_spectre_batch requires a non-empty batch")

    max_r = max(len(ex.r_skeletons) for ex in batch)
    max_f = max(len(ex.f_skeletons) for ex in batch) or 1  # Ψ wants at least width 1
    max_op_arity = max(vocab.max_operator_arity, 1)
    max_pred_arity = max(vocab.max_predicate_arity, 1)
    max_skel_len = max(
        (
            len(skel.operator_seq)
            for ex in batch
            for skel in (*ex.r_skeletons, *ex.f_skeletons)
        ),
        default=1,
    )
    num_types = len(vocab.types)

    # Sum up s0 atom max across the batch.
    max_s0_atoms = max(len(ex.initial_abstract_state.atoms) for ex in batch) or 1
    # s_L atom max across all R and F skeletons in the batch.
    max_sL_atoms = (
        max(
            (
                len(skel.final_abstract_state.atoms)
                for ex in batch
                for skel in (*ex.r_skeletons, *ex.f_skeletons)
            ),
            default=1,
        )
        or 1
    )

    def _blank_op_tokens(
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((b, w, max_skel_len), dtype=torch.long),
            torch.zeros((b, w, max_skel_len, max_op_arity), dtype=torch.long),
            torch.zeros((b, w, max_skel_len, max_op_arity), dtype=torch.long),
            torch.zeros((b, w, max_skel_len), dtype=torch.bool),
        )

    def _blank_sL_tokens(
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((b, w, max_sL_atoms), dtype=torch.long),
            torch.zeros((b, w, max_sL_atoms, max_pred_arity), dtype=torch.long),
            torch.zeros((b, w, max_sL_atoms, max_pred_arity), dtype=torch.long),
            torch.zeros((b, w, max_sL_atoms), dtype=torch.bool),
        )

    r_op_ids, r_arg_types, r_arg_locals, r_op_mask = _blank_op_tokens(max_r)
    f_op_ids, f_arg_types, f_arg_locals, f_op_mask = _blank_op_tokens(max_f)
    r_sL_pred, r_sL_at, r_sL_al, r_sL_mask = _blank_sL_tokens(max_r)
    f_sL_pred, f_sL_at, f_sL_al, f_sL_mask = _blank_sL_tokens(max_f)

    r_mask = torch.zeros((b, max_r), dtype=torch.bool)
    f_mask = torch.zeros((b, max_f), dtype=torch.bool)
    r_priors = torch.zeros((b, max_r), dtype=torch.float32)
    r_success_mask = torch.zeros((b, max_r), dtype=torch.bool)

    s0_pred_ids = torch.zeros((b, max_s0_atoms), dtype=torch.long)
    s0_arg_type_ids = torch.zeros((b, max_s0_atoms, max_pred_arity), dtype=torch.long)
    s0_arg_local_ids = torch.zeros((b, max_s0_atoms, max_pred_arity), dtype=torch.long)
    s0_atom_mask = torch.zeros((b, max_s0_atoms), dtype=torch.bool)
    s0_type_hist = torch.zeros((b, num_types), dtype=torch.long)

    problem_ids = torch.zeros(b, dtype=torch.long)

    def _fill_skels(
        skels: tuple[SkeletonRecord, ...],
        op_ids_t: torch.Tensor,
        arg_types_t: torch.Tensor,
        arg_locals_t: torch.Tensor,
        op_mask_t: torch.Tensor,
        sL_pred_t: torch.Tensor,
        sL_at_t: torch.Tensor,
        sL_al_t: torch.Tensor,
        sL_mask_t: torch.Tensor,
        example_i: int,
    ) -> None:
        for j, skel in enumerate(skels):
            oi, ti, li, msk = _encode_skeleton(skel, vocab, max_skel_len, max_op_arity)
            op_ids_t[example_i, j] = torch.tensor(oi, dtype=torch.long)
            arg_types_t[example_i, j] = torch.tensor(ti, dtype=torch.long)
            arg_locals_t[example_i, j] = torch.tensor(li, dtype=torch.long)
            op_mask_t[example_i, j] = torch.tensor(msk, dtype=torch.bool)

            sL_atoms = list(skel.final_abstract_state.atoms)
            sp, st, sl, sm = _encode_atom_set(
                sL_atoms, vocab, max_sL_atoms, max_pred_arity
            )
            sL_pred_t[example_i, j] = torch.tensor(sp, dtype=torch.long)
            sL_at_t[example_i, j] = torch.tensor(st, dtype=torch.long)
            sL_al_t[example_i, j] = torch.tensor(sl, dtype=torch.long)
            sL_mask_t[example_i, j] = torch.tensor(sm, dtype=torch.bool)

    for i, ex in enumerate(batch):
        problem_ids[i] = ex.problem_id

        _fill_skels(
            ex.r_skeletons,
            r_op_ids,
            r_arg_types,
            r_arg_locals,
            r_op_mask,
            r_sL_pred,
            r_sL_at,
            r_sL_al,
            r_sL_mask,
            i,
        )
        _fill_skels(
            ex.f_skeletons,
            f_op_ids,
            f_arg_types,
            f_arg_locals,
            f_op_mask,
            f_sL_pred,
            f_sL_at,
            f_sL_al,
            f_sL_mask,
            i,
        )

        r_mask[i, : len(ex.r_skeletons)] = True
        f_mask[i, : len(ex.f_skeletons)] = True
        for j, p in enumerate(ex.r_priors):
            r_priors[i, j] = p
        for j, s in enumerate(ex.r_success_mask):
            r_success_mask[i, j] = s

        atoms = list(ex.initial_abstract_state.atoms)
        sp, st, sl, sm = _encode_atom_set(atoms, vocab, max_s0_atoms, max_pred_arity)
        s0_pred_ids[i] = torch.tensor(sp, dtype=torch.long)
        s0_arg_type_ids[i] = torch.tensor(st, dtype=torch.long)
        s0_arg_local_ids[i] = torch.tensor(sl, dtype=torch.long)
        s0_atom_mask[i] = torch.tensor(sm, dtype=torch.bool)

        for obj_name, type_name in ex.object_registry.items():
            del obj_name
            if type_name in vocab.types:
                s0_type_hist[i, vocab.type_idx(type_name)] += 1

    return SpectreBatch(
        r_op_ids=r_op_ids,
        r_op_arg_type_ids=r_arg_types,
        r_op_arg_local_ids=r_arg_locals,
        r_op_mask=r_op_mask,
        r_mask=r_mask,
        r_priors=r_priors,
        r_success_mask=r_success_mask,
        r_sL_pred_ids=r_sL_pred,
        r_sL_arg_type_ids=r_sL_at,
        r_sL_arg_local_ids=r_sL_al,
        r_sL_atom_mask=r_sL_mask,
        f_op_ids=f_op_ids,
        f_op_arg_type_ids=f_arg_types,
        f_op_arg_local_ids=f_arg_locals,
        f_op_mask=f_op_mask,
        f_mask=f_mask,
        f_sL_pred_ids=f_sL_pred,
        f_sL_arg_type_ids=f_sL_at,
        f_sL_arg_local_ids=f_sL_al,
        f_sL_atom_mask=f_sL_mask,
        s0_pred_ids=s0_pred_ids,
        s0_arg_type_ids=s0_arg_type_ids,
        s0_arg_local_ids=s0_arg_local_ids,
        s0_atom_mask=s0_atom_mask,
        s0_type_histogram=s0_type_hist,
        problem_ids=problem_ids,
    )
