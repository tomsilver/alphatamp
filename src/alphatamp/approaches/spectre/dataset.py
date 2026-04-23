"""PyTorch Dataset + collate for SPECTRE.

Online F-subset sampling (pipeline spec §11): ``__getitem__`` loads an episode,
samples ``F ⊆ FAIL_e``, and returns a structured training example. Tensorization
happens in ``collate_spectre_batch`` so the Dataset output stays inspectable.

**Critical invariant** (pipeline spec §11.6; violated in prior Attempt 2): F
must contain only failed skeletons, never successes. We assert this inside
``__getitem__`` so a regression fails loudly rather than silently corrupting
training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

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

FSampling = Literal["uniform_subsets"]


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
    """

    def __init__(
        self,
        split_dir: Path,
        prior: BasePrior,
        seed: int,
        f_sampling: FSampling = "uniform_subsets",
        augment: bool = True,
        episode_cache_size: int = 64,
    ) -> None:
        self._split_dir = split_dir
        self._prior = prior
        self._seed = seed
        self._f_sampling = f_sampling
        self._augment = augment

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
        return len(self._episode_paths)

    @property
    def filtered_problem_ids(self) -> list[tuple[int, str]]:
        """``(problem_id, reason)`` tuples for episodes excluded at init."""
        return list(self._filtered)

    def _rng_for(self, index: int) -> np.random.Generator:
        """Deterministic per-(seed, index) RNG.

        Does not depend on epoch counter: each ``__getitem__`` call uses the
        same seed (for reproducibility of a single call). Multi-epoch variety
        comes from torch's DataLoader shuffling + from the inherent
        randomness encoded by re-seeding on every call with the instance
        seed plus the index.
        """
        return np.random.default_rng((self._seed, index))

    def __getitem__(self, index: int) -> SpectreTrainingExample:
        ep = self._load_cached(str(self._episode_paths[index]))
        rng = self._rng_for(index)

        succ = set(ep.success_indices())
        fail = set(ep.fail_indices())
        errs = set(ep.error_indices())

        # Sample F ⊆ FAIL uniformly over the power set.
        if self._f_sampling == "uniform_subsets":
            mask = rng.random(size=len(fail)) < 0.5
            f_indices = {idx for idx, keep in zip(sorted(fail), mask) if keep}
        else:  # pragma: no cover - exhaustiveness guard
            raise NotImplementedError(self._f_sampling)

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

        # Canonicalize; augmentation applies a random within-type permutation.
        ep_view = canonicalize_episode(ep, rng if self._augment else None)

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

    Shapes follow ``SPECTRE_METHOD_SPEC.md`` §4.1.3. All integer id tensors
    use 0 = ``<OOV>`` / padding; the mask tensors distinguish real tokens
    from pads.
    """

    # R tokens
    r_op_ids: torch.Tensor  # (B, R, L)                        long
    r_op_arg_type_ids: torch.Tensor  # (B, R, L, A)                     long
    r_op_arg_local_ids: torch.Tensor  # (B, R, L, A)                     long
    r_op_mask: torch.Tensor  # (B, R, L)                        bool
    r_mask: torch.Tensor  # (B, R)                           bool
    r_priors: torch.Tensor  # (B, R)                           float
    r_success_mask: torch.Tensor  # (B, R)                           bool

    # F tokens (same schema as R, minus priors/success)
    f_op_ids: torch.Tensor  # (B, F, L)                        long
    f_op_arg_type_ids: torch.Tensor  # (B, F, L, A)                     long
    f_op_arg_local_ids: torch.Tensor  # (B, F, L, A)                     long
    f_op_mask: torch.Tensor  # (B, F, L)                        bool
    f_mask: torch.Tensor  # (B, F)                           bool

    # s0: per-example, replicated per-skeleton at model-input time
    s0_pred_ids: torch.Tensor  # (B, P)                           long
    s0_arg_type_ids: torch.Tensor  # (B, P, AP)                       long
    s0_arg_local_ids: torch.Tensor  # (B, P, AP)                       long
    s0_atom_mask: torch.Tensor  # (B, P)                           bool
    s0_type_histogram: torch.Tensor  # (B, T)                           long

    problem_ids: torch.Tensor  # (B,)                             long

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

    def _blank_tokens(
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((b, w, max_skel_len), dtype=torch.long),
            torch.zeros((b, w, max_skel_len, max_op_arity), dtype=torch.long),
            torch.zeros((b, w, max_skel_len, max_op_arity), dtype=torch.long),
            torch.zeros((b, w, max_skel_len), dtype=torch.bool),
        )

    r_op_ids, r_arg_types, r_arg_locals, r_op_mask = _blank_tokens(max_r)
    f_op_ids, f_arg_types, f_arg_locals, f_op_mask = _blank_tokens(max_f)

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
        skels, op_ids_t, arg_types_t, arg_locals_t, op_mask_t, example_i
    ) -> None:
        for j, skel in enumerate(skels):
            oi, ti, li, msk = _encode_skeleton(skel, vocab, max_skel_len, max_op_arity)
            op_ids_t[example_i, j] = torch.tensor(oi, dtype=torch.long)
            arg_types_t[example_i, j] = torch.tensor(ti, dtype=torch.long)
            arg_locals_t[example_i, j] = torch.tensor(li, dtype=torch.long)
            op_mask_t[example_i, j] = torch.tensor(msk, dtype=torch.bool)

    for i, ex in enumerate(batch):
        problem_ids[i] = ex.problem_id

        _fill_skels(ex.r_skeletons, r_op_ids, r_arg_types, r_arg_locals, r_op_mask, i)
        _fill_skels(ex.f_skeletons, f_op_ids, f_arg_types, f_arg_locals, f_op_mask, i)

        r_mask[i, : len(ex.r_skeletons)] = True
        f_mask[i, : len(ex.f_skeletons)] = True
        for j, p in enumerate(ex.r_priors):
            r_priors[i, j] = p
        for j, s in enumerate(ex.r_success_mask):
            r_success_mask[i, j] = s

        atoms = list(ex.initial_abstract_state.atoms)
        for j, atom in enumerate(atoms):
            pi, ti, li = _encode_atom(atom, vocab, max_pred_arity)
            s0_pred_ids[i, j] = pi
            s0_arg_type_ids[i, j] = torch.tensor(ti, dtype=torch.long)
            s0_arg_local_ids[i, j] = torch.tensor(li, dtype=torch.long)
            s0_atom_mask[i, j] = True

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
        f_op_ids=f_op_ids,
        f_op_arg_type_ids=f_arg_types,
        f_op_arg_local_ids=f_arg_locals,
        f_op_mask=f_op_mask,
        f_mask=f_mask,
        s0_pred_ids=s0_pred_ids,
        s0_arg_type_ids=s0_arg_type_ids,
        s0_arg_local_ids=s0_arg_local_ids,
        s0_atom_mask=s0_atom_mask,
        s0_type_histogram=s0_type_hist,
        problem_ids=problem_ids,
    )
