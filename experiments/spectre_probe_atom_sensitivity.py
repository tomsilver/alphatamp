"""Atom-sensitivity + frozen-Φ linear-probe diagnostic for SPECTRE.

Run after ``spectre_train.py``. Two read-only probes:

D.1 — Atom mutation sensitivity. Mutate the static-tag atoms in s₀
(``PassageWidth``, ``ItemSize``) to flip narrow↔wide / small↔large, re-run
Φ on every R-skeleton, and report ``‖e'(s) − e(s)‖ / ‖e(s)‖``. A trained
Φ that genuinely reads s₀ should show a substantial Δ; a Φ that ignores
those atoms will show Δ ≈ 0. Compares against an "atom-row shuffle"
control (set-permutation invariance ⇒ Δ = 0 by construction) and a
"randomize all s₀ predicates" saturation point.

D.2 — Frozen-Φ linear separability. Encode every (skeleton, success)
pair on val into ``(e(s), label)``, fit logistic regression with k-fold
CV, report mean held-out AUROC. If linear-probe AUROC ≫ live AUROC(0),
σ has converged to mis-using a representation that *does* carry signal
(training-dynamics failure). If linear-probe AUROC ≈ 0.5, the bottleneck
is in Φ's representation itself (architectural failure).

Usage::

    python experiments/spectre_probe_atom_sensitivity.py \
        env=routedtransport2d_n3_v1 seed=0
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from alphatamp.approaches.spectre.dataset import (
    SpectreBatch,
    SpectreDataset,
    collate_spectre_batch,
)
from alphatamp.approaches.spectre import inference
from alphatamp.approaches.spectre.env_registry import (
    get_static_tag_predicates,
    get_type_aug_policy,
)
from alphatamp.approaches.spectre.model import SpectreModel
from alphatamp.approaches.spectre.priors import ZeroPrior
from alphatamp.approaches.spectre.vocab import Vocab


@dataclass
class ProbeRow:
    """One row of the D.1 sensitivity table."""

    label: str
    rel_delta_mean: float
    rel_delta_p10: float
    rel_delta_p90: float
    n_examples: int


def _load_checkpoint(
    ckpt_path: Path,
    vocab: Vocab,
    device: torch.device,
    fallback_static_tag_predicates: list[str] | None = None,
) -> SpectreModel:
    """Forward to the shared loader so probe + notebook + future tools agree."""
    return inference.load_checkpoint(
        ckpt_path,
        vocab,
        device=device,
        fallback_static_tag_predicates=fallback_static_tag_predicates,
    )


def _encode_pool(model: SpectreModel, batch: SpectreBatch) -> torch.Tensor:
    """Run Φ over the R-pool and return e_R: (B, R, D)."""
    return model.encode_pool(
        batch.r_op_ids,
        batch.r_op_arg_type_ids,
        batch.r_op_arg_local_ids,
        batch.r_op_mask,
        batch.s0_pred_ids,
        batch.s0_arg_type_ids,
        batch.s0_arg_local_ids,
        batch.s0_atom_mask,
        batch.s0_type_histogram,
        batch.r_sL_pred_ids,
        batch.r_sL_arg_type_ids,
        batch.r_sL_arg_local_ids,
        batch.r_sL_atom_mask,
    )


def _clone_batch(batch: SpectreBatch) -> SpectreBatch:
    """Shallow copy with fresh tensor clones for the s_0 fields we mutate.

    Other fields are aliased — D.1 only mutates s_0 atom tensors.
    """
    new = copy.copy(batch)
    new.s0_pred_ids = batch.s0_pred_ids.clone()
    new.s0_arg_type_ids = batch.s0_arg_type_ids.clone()
    new.s0_arg_local_ids = batch.s0_arg_local_ids.clone()
    new.s0_atom_mask = batch.s0_atom_mask.clone()
    new.s0_type_histogram = batch.s0_type_histogram.clone()
    return new


def _swap_local_id_for_predicate(
    batch: SpectreBatch,
    pred_idx: int,
    arg_slot: int,
    swap_a: int,
    swap_b: int,
) -> SpectreBatch:
    """Swap encoded local-id ``swap_a`` ↔ ``swap_b`` in a specified arg slot
    of every real atom matching ``pred_idx``.
    """
    new = _clone_batch(batch)
    pred_match = (batch.s0_pred_ids == pred_idx) & batch.s0_atom_mask  # (B, M0)
    arg_col = new.s0_arg_local_ids[..., arg_slot]  # view of (B, M0)
    is_a = pred_match & (arg_col == swap_a)
    is_b = pred_match & (arg_col == swap_b)
    arg_col[is_a] = swap_b
    arg_col[is_b] = swap_a
    return new


def _shuffle_atom_rows(batch: SpectreBatch, rng: np.random.Generator) -> SpectreBatch:
    """Permutation-invariance control: shuffle the atom row order for each example.

    The atom-pool inside Φ is a Set Transformer; shuffling the row order of
    the input atom tokens must leave ``e(s)`` unchanged up to numerical
    noise. Δ here is the floating-point noise floor.
    """
    new = _clone_batch(batch)
    bsz, m0 = batch.s0_pred_ids.shape
    for b in range(bsz):
        perm = rng.permutation(m0)
        new.s0_pred_ids[b] = batch.s0_pred_ids[b][perm]
        new.s0_arg_type_ids[b] = batch.s0_arg_type_ids[b][perm]
        new.s0_arg_local_ids[b] = batch.s0_arg_local_ids[b][perm]
        new.s0_atom_mask[b] = batch.s0_atom_mask[b][perm]
    return new


def _scramble_predicates(
    batch: SpectreBatch, n_predicates: int, rng: np.random.Generator
) -> SpectreBatch:
    """Saturation control: replace every real atom's predicate with a random one.

    Reports the magnitude of Δe(s) when s₀ is destroyed, giving a "max
    sensitivity" reference. PassageWidth-flip Δ as a fraction of this Δ
    indicates how much of the model's static-atom-reading capacity is
    targeted at width/size.
    """
    new = _clone_batch(batch)
    bsz, m0 = batch.s0_pred_ids.shape
    if n_predicates <= 1:
        return new
    # Random predicate ids in [1, n_predicates) (0 is OOV / pad).
    rand = torch.from_numpy(rng.integers(low=1, high=n_predicates, size=(bsz, m0))).to(
        dtype=batch.s0_pred_ids.dtype
    )
    rand = torch.where(batch.s0_atom_mask, rand, batch.s0_pred_ids)
    new.s0_pred_ids = rand
    return new


def _rel_delta(
    e_orig: torch.Tensor, e_mut: torch.Tensor, r_mask: torch.Tensor
) -> list[float]:
    """Per-skeleton ``‖e' − e‖ / ‖e‖`` over R-valid slots, returned as a flat list."""
    diff = (e_mut - e_orig).norm(dim=-1)  # (B, R)
    base = e_orig.norm(dim=-1).clamp(min=1e-9)  # (B, R)
    rel = (diff / base).cpu().numpy()
    mask = r_mask.cpu().numpy()
    return [
        float(rel[b, j])
        for b in range(rel.shape[0])
        for j in range(rel.shape[1])
        if mask[b, j]
    ]


def _summarize(label: str, deltas: list[float]) -> ProbeRow:
    arr = np.asarray(deltas, dtype=np.float64) if deltas else np.zeros(0)
    if arr.size == 0:
        return ProbeRow(
            label=label,
            rel_delta_mean=float("nan"),
            rel_delta_p10=float("nan"),
            rel_delta_p90=float("nan"),
            n_examples=0,
        )
    return ProbeRow(
        label=label,
        rel_delta_mean=float(arr.mean()),
        rel_delta_p10=float(np.percentile(arr, 10)),
        rel_delta_p90=float(np.percentile(arr, 90)),
        n_examples=int(arr.size),
    )


def _move_batch(batch: SpectreBatch, device: torch.device) -> SpectreBatch:
    fields: dict = {}
    for name, val in batch.__dict__.items():
        fields[name] = val.to(device) if isinstance(val, torch.Tensor) else val
    return SpectreBatch(**fields)


def _probe_d1_atom_sensitivity(
    model: SpectreModel,
    val_dataset: SpectreDataset,
    vocab: Vocab,
    device: torch.device,
    seed: int,
    batch_size: int = 4,
) -> list[ProbeRow]:
    """D.1: mutate s₀ static-tag atoms and report ``‖Δe(s)‖/‖e(s)‖``."""
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_spectre_batch(b, vocab),
    )
    pw_idx = vocab.pred_idx("PassageWidth")  # arg slot 1 = width_level
    is_idx = vocab.pred_idx("ItemSize")  # arg slot 1 = size_level
    n_predicates = len(vocab.predicates)
    rng = np.random.default_rng(seed)
    # Encoded local-ids start at 1 (0 = pad). RT2D has 3 width / size levels,
    # encoded ids ∈ {1, 2, 3}; swap 1 ↔ 3 = "narrow ↔ wide" / "small ↔ large".
    deltas: dict[str, list[float]] = {
        "shuffle_atom_rows (control: must be ~0)": [],
        "PassageWidth: width_level 1↔3": [],
        "ItemSize: size_level 1↔3": [],
        "scramble_predicates (saturation)": [],
    }

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            e_orig = _encode_pool(model, batch)

            # Each mutation: build a mutated batch, encode, compute relative Δ.
            mutated = _shuffle_atom_rows(batch, rng)
            e_mut = _encode_pool(model, mutated)
            deltas["shuffle_atom_rows (control: must be ~0)"].extend(
                _rel_delta(e_orig, e_mut, batch.r_mask)
            )

            mutated = _swap_local_id_for_predicate(
                batch, pw_idx, arg_slot=1, swap_a=1, swap_b=3
            )
            e_mut = _encode_pool(model, mutated)
            deltas["PassageWidth: width_level 1↔3"].extend(
                _rel_delta(e_orig, e_mut, batch.r_mask)
            )

            mutated = _swap_local_id_for_predicate(
                batch, is_idx, arg_slot=1, swap_a=1, swap_b=3
            )
            e_mut = _encode_pool(model, mutated)
            deltas["ItemSize: size_level 1↔3"].extend(
                _rel_delta(e_orig, e_mut, batch.r_mask)
            )

            mutated = _scramble_predicates(batch, n_predicates, rng)
            e_mut = _encode_pool(model, mutated)
            deltas["scramble_predicates (saturation)"].extend(
                _rel_delta(e_orig, e_mut, batch.r_mask)
            )

    return [_summarize(label, vals) for label, vals in deltas.items()]


def _probe_d2_linear_separability(
    model: SpectreModel,
    val_dataset: SpectreDataset,
    vocab: Vocab,
    device: torch.device,
    seed: int,
    batch_size: int = 4,
    n_splits: int = 5,
) -> tuple[float, int, int]:
    """D.2: frozen-Φ → logistic regression → AUROC under k-fold CV.

    Returns ``(mean_auroc, n_pos, n_neg)``.
    """
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_spectre_batch(b, vocab),
    )
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            e_R = _encode_pool(model, batch).cpu().numpy()  # (B, R, D)
            r_mask = batch.r_mask.cpu().numpy()
            r_succ = batch.r_success_mask.cpu().numpy()
            for b in range(e_R.shape[0]):
                for j in range(e_R.shape[1]):
                    if not r_mask[b, j]:
                        continue
                    embeddings.append(e_R[b, j])
                    labels.append(1 if r_succ[b, j] else 0)
    if not embeddings:
        return float("nan"), 0, 0
    X = np.stack(embeddings, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        scores = clf.decision_function(X[test_idx])
        aurocs.append(float(roc_auc_score(y[test_idx], scores)))
    return float(np.mean(aurocs)), n_pos, n_neg


def _passage_type_ids(vocab: Vocab) -> set[int]:
    """All vocab type ids whose name describes a passage object."""
    out: set[int] = set()
    for type_name in vocab.types:
        if type_name == "passage" or type_name.startswith("passage_color_"):
            out.add(vocab.type_idx(type_name))
    return out


def _used_passage_args(
    op_arg_type_ids: torch.Tensor,  # (L, A)
    op_arg_local_ids: torch.Tensor,  # (L, A)
    op_mask: torch.Tensor,  # (L,)
    passage_type_ids: set[int],
) -> set[tuple[int, int]]:
    """Return ``{(type_id, local_id)}`` for passages used as op args in this skeleton."""
    used: set[tuple[int, int]] = set()
    L, A = op_arg_type_ids.shape
    for l_idx in range(L):
        if not bool(op_mask[l_idx].item()):
            continue
        for a_idx in range(A):
            t = int(op_arg_type_ids[l_idx, a_idx].item())
            if t in passage_type_ids:
                used.add((t, int(op_arg_local_ids[l_idx, a_idx].item())))
    return used


def _mutate_passage_width_subset(
    batch: SpectreBatch,
    b: int,
    pw_idx: int,
    target_passages: set[tuple[int, int]],
    swap_a: int,
    swap_b: int,
) -> SpectreBatch:
    """Clone ``batch``; swap width_level on PassageWidth atoms whose passage
    arg ``(type_id, local_id)`` is in ``target_passages``. Touches only
    example ``b``; other examples in the batch are aliased.
    """
    new = _clone_batch(batch)
    pred_match = (new.s0_pred_ids[b] == pw_idx) & new.s0_atom_mask[b]
    if not pred_match.any():
        return new
    m0 = new.s0_pred_ids.shape[1]
    for m in range(m0):
        if not bool(pred_match[m].item()):
            continue
        p_type = int(new.s0_arg_type_ids[b, m, 0].item())
        p_local = int(new.s0_arg_local_ids[b, m, 0].item())
        if (p_type, p_local) not in target_passages:
            continue
        wl = int(new.s0_arg_local_ids[b, m, 1].item())
        if wl == swap_a:
            new.s0_arg_local_ids[b, m, 1] = swap_b
        elif wl == swap_b:
            new.s0_arg_local_ids[b, m, 1] = swap_a
    return new


def _encode_single_skeleton(
    model: SpectreModel, batch: SpectreBatch, b: int, j: int
) -> torch.Tensor:
    """Encode one skeleton (example ``b``, slot ``j``) using ``batch``'s s_0.

    Returns shape ``(D,)``. Uses B=1 K=1 slices so a per-skeleton-mutated
    s_0 can be passed in via the batch object without altering other
    examples' encodings.
    """
    e = model.encode_pool(
        batch.r_op_ids[b : b + 1, j : j + 1],
        batch.r_op_arg_type_ids[b : b + 1, j : j + 1],
        batch.r_op_arg_local_ids[b : b + 1, j : j + 1],
        batch.r_op_mask[b : b + 1, j : j + 1],
        batch.s0_pred_ids[b : b + 1],
        batch.s0_arg_type_ids[b : b + 1],
        batch.s0_arg_local_ids[b : b + 1],
        batch.s0_atom_mask[b : b + 1],
        batch.s0_type_histogram[b : b + 1],
        batch.r_sL_pred_ids[b : b + 1, j : j + 1],
        batch.r_sL_arg_type_ids[b : b + 1, j : j + 1],
        batch.r_sL_arg_local_ids[b : b + 1, j : j + 1],
        batch.r_sL_atom_mask[b : b + 1, j : j + 1],
    )
    return e[0, 0]


@dataclass
class D3Summary:
    """Summary statistics for D.3 used-vs-unused binding specificity.

    The ratio of means is the robust aggregate; mean of per-skeleton ratios
    is biased upward by skeletons with very small Δ_unused. Median and
    geometric mean of per-skeleton ratios both summarize the typical
    skeleton without the small-denominator inflation.
    """

    ratio_of_means: float
    median_of_ratios: float
    geomean_of_ratios: float
    mean_of_ratios: float  # kept for transparency; do not interpret naively
    n_ratios: int
    n_skipped: int


def _probe_d3_binding_specificity(
    model: SpectreModel,
    val_dataset: SpectreDataset,
    vocab: Vocab,
    device: torch.device,
    batch_size: int = 4,
    swap_a: int = 1,
    swap_b: int = 3,
) -> tuple[ProbeRow, ProbeRow, D3Summary]:
    """D.3: per-skeleton, mutate PassageWidth on USED vs UNUSED passages.

    For each skeleton in the val pool, identify the passages it uses as
    operator args, partition the s_0 PassageWidth atoms into a "used" set
    and "unused" set, flip ``width_level`` on each set independently, and
    compute ``‖e' − e‖ / ‖e‖`` per skeleton. Returns the per-skeleton
    distributions of Δ_used and Δ_unused as ``ProbeRow``s plus a
    :class:`D3Summary` with multiple aggregates of the per-skeleton ratio
    so the caller can reason about typical-vs-tail behavior.
    """
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_spectre_batch(b, vocab),
    )
    pw_idx = vocab.pred_idx("PassageWidth")
    passage_type_ids = _passage_type_ids(vocab)
    if not passage_type_ids:
        empty = D3Summary(
            ratio_of_means=float("nan"),
            median_of_ratios=float("nan"),
            geomean_of_ratios=float("nan"),
            mean_of_ratios=float("nan"),
            n_ratios=0,
            n_skipped=0,
        )
        return (
            _summarize("D.3 used", []),
            _summarize("D.3 unused", []),
            empty,
        )

    used_deltas: list[float] = []
    unused_deltas: list[float] = []
    ratios: list[float] = []
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            bsz = batch.r_op_ids.shape[0]
            kpool = batch.r_op_ids.shape[1]
            m0 = batch.s0_pred_ids.shape[1]
            for b in range(bsz):
                pred_mask_row = (batch.s0_pred_ids[b] == pw_idx) & batch.s0_atom_mask[b]
                if not pred_mask_row.any():
                    continue
                # All passage args mentioned in this example's PassageWidth atoms.
                all_passages: set[tuple[int, int]] = set()
                for m in range(m0):
                    if not bool(pred_mask_row[m].item()):
                        continue
                    all_passages.add(
                        (
                            int(batch.s0_arg_type_ids[b, m, 0].item()),
                            int(batch.s0_arg_local_ids[b, m, 0].item()),
                        )
                    )
                for j in range(kpool):
                    if not bool(batch.r_mask[b, j].item()):
                        continue
                    used = (
                        _used_passage_args(
                            batch.r_op_arg_type_ids[b, j],
                            batch.r_op_arg_local_ids[b, j],
                            batch.r_op_mask[b, j],
                            passage_type_ids,
                        )
                        & all_passages
                    )
                    unused = all_passages - used
                    if not used or not unused:
                        # Ratio is undefined if either partition is empty;
                        # skip but track for the report.
                        n_skipped += 1
                        continue

                    e_orig = _encode_single_skeleton(model, batch, b, j)
                    used_batch = _mutate_passage_width_subset(
                        batch, b, pw_idx, used, swap_a, swap_b
                    )
                    e_used = _encode_single_skeleton(model, used_batch, b, j)
                    unused_batch = _mutate_passage_width_subset(
                        batch, b, pw_idx, unused, swap_a, swap_b
                    )
                    e_unused = _encode_single_skeleton(model, unused_batch, b, j)

                    base = float(e_orig.norm().clamp(min=1e-9).item())
                    d_used = float((e_used - e_orig).norm().item()) / base
                    d_unused = float((e_unused - e_orig).norm().item()) / base
                    used_deltas.append(d_used)
                    unused_deltas.append(d_unused)
                    if d_unused > 1e-6:
                        ratios.append(d_used / d_unused)

    used_row = _summarize("PassageWidth: USED passages flipped", used_deltas)
    unused_row = _summarize("PassageWidth: UNUSED passages flipped", unused_deltas)

    # Robust aggregate: ratio of population means (no small-denom inflation).
    if used_deltas and unused_deltas:
        mean_used = float(np.mean(used_deltas))
        mean_unused = float(np.mean(unused_deltas))
        ratio_of_means = mean_used / mean_unused if mean_unused > 0 else float("nan")
    else:
        ratio_of_means = float("nan")

    if ratios:
        ratios_arr = np.asarray(ratios, dtype=np.float64)
        # Geomean is well-defined for strictly-positive ratios; the mask
        # above already filters d_unused, but a zero d_used yields a zero
        # ratio that breaks log. Drop those for the geomean only.
        positive = ratios_arr[ratios_arr > 0]
        geomean = (
            float(np.exp(np.mean(np.log(positive)))) if positive.size else float("nan")
        )
        summary = D3Summary(
            ratio_of_means=ratio_of_means,
            median_of_ratios=float(np.median(ratios_arr)),
            geomean_of_ratios=geomean,
            mean_of_ratios=float(np.mean(ratios_arr)),
            n_ratios=int(ratios_arr.size),
            n_skipped=n_skipped,
        )
    else:
        summary = D3Summary(
            ratio_of_means=ratio_of_means,
            median_of_ratios=float("nan"),
            geomean_of_ratios=float("nan"),
            mean_of_ratios=float("nan"),
            n_ratios=0,
            n_skipped=n_skipped,
        )
    return used_row, unused_row, summary


def _print_d1_table(rows: list[ProbeRow]) -> None:
    print(f"\n{'mutation':<48s} {'mean':>8s} {'p10':>8s} {'p90':>8s} {'n':>6s}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r.label:<48s}"
            f" {r.rel_delta_mean:>8.4f} {r.rel_delta_p10:>8.4f}"
            f" {r.rel_delta_p90:>8.4f} {r.n_examples:>6d}"
        )


@hydra.main(
    config_path="conf",
    config_name="spectre_train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Probe entrypoint — re-uses spectre_train.yaml for paths."""
    data_root = Path(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    seed = int(cfg.seed)
    val_dir = data_root / "raw" / env_variant / "val"
    vocab_path = data_root / "derived" / env_variant / "train_vocab.json"
    ckpt_path = Path(cfg.out_dir) / env_variant / f"seed_{seed}" / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}; run spectre_train first."
        )
    if not vocab_path.exists():
        raise FileNotFoundError(f"No vocab at {vocab_path}.")
    if not (val_dir / "episodes").exists():
        raise FileNotFoundError(f"No val episodes at {val_dir}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.from_json(vocab_path)
    type_aug_policy = get_type_aug_policy(env_variant) or dict(vocab.type_aug_policy)

    print(f"env_variant={env_variant} seed={seed}")
    print(f"  ckpt={ckpt_path}")
    print(f"  device={device}")

    val_dataset = SpectreDataset(
        split_dir=val_dir,
        prior=ZeroPrior(),
        seed=seed + 10_000,
        f_sampling="rollout_aligned_mix",
        augment=False,
        type_aug_policy=type_aug_policy,
        num_f_samples_per_epoch=1,
    )
    print(f"  val episodes: {len(val_dataset)}")

    model = _load_checkpoint(
        ckpt_path,
        vocab,
        device,
        fallback_static_tag_predicates=get_static_tag_predicates(env_variant),
    )

    # ---------------- D.1 ----------------
    print("\n[D.1] Atom-sensitivity probe")
    rows = _probe_d1_atom_sensitivity(model, val_dataset, vocab, device, seed)
    _print_d1_table(rows)
    print("\nInterpretation:")
    print("  control row (shuffle_atom_rows) should be ~0 (numerical floor).")
    print("  saturation row gives the 'maximum possible' Δ from a destroyed s_0.")
    print("  PassageWidth/ItemSize rows: a Φ that reads static atoms shows")
    print("  Δ that is a non-trivial fraction of saturation (≳ 0.10 absolute,")
    print("  ≳ 1/3 of saturation). Δ ≈ control ⇒ Φ ignores those atoms.")

    # ---------------- D.2 ----------------
    print("\n[D.2] Frozen-Φ linear-probe AUROC (k-fold CV on val)")
    auroc, n_pos, n_neg = _probe_d2_linear_separability(
        model, val_dataset, vocab, device, seed
    )
    print(f"  linear-probe AUROC: {auroc:.4f}")
    print(f"  n_pos={n_pos}  n_neg={n_neg}  total={n_pos + n_neg}")
    print("\nInterpretation:")
    print("  If linear-probe AUROC >> live AUROC(0): e(s) carries linearly-")
    print("  separable success signal but σ has converged to mis-using it.")
    print("  If linear-probe AUROC ≈ 0.5: Φ's representation is the bottleneck.")

    # ---------------- D.3 ----------------
    print("\n[D.3] Binding-specificity probe (per-skeleton USED vs UNUSED passages)")
    used_row, unused_row, d3 = _probe_d3_binding_specificity(
        model, val_dataset, vocab, device
    )
    _print_d1_table([used_row, unused_row])
    print(
        f"  ratio of means        Δ_used.mean / Δ_unused.mean = {d3.ratio_of_means:.3f}"
    )
    print(
        f"  median of ratios      median_i (Δ_used / Δ_unused) = {d3.median_of_ratios:.3f}"
    )
    print(
        f"  geomean of ratios     exp(mean log(ratios))        = {d3.geomean_of_ratios:.3f}"
    )
    print(
        f"  mean of ratios (biased; do not interpret naively)    = {d3.mean_of_ratios:.3f}"
    )
    print(f"  n_ratios={d3.n_ratios}  skipped (empty used/unused set)={d3.n_skipped}")
    print(
        "\nInterpretation (use ratio_of_means and median_of_ratios; mean_of_ratios is biased):"
    )
    print("  ratio >> 1 (≥ 3): Φ binds passage-width to specific operator args.")
    print("    → Bottleneck is in σ; try Step F3-A (Φ-dropout / larger σ).")
    print("  ratio ≈ 1–2: Φ reads PassageWidth globally with weak binding.")
    print("    → Bottleneck is in the SkeletonEncoder transformer; try Step F3-B.")
    print("  ratio ≈ 0 or both Δs ≈ control: Φ_s collapsed; re-investigate F1.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
