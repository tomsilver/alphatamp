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
from alphatamp.approaches.spectre.env_registry import get_type_aug_policy
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
    ckpt_path: Path, vocab: Vocab, device: torch.device
) -> SpectreModel:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("config", {}) or {}
    use_atom_sab2 = bool(cfg_dict.get("use_atom_sab2", True))
    prior_dropout_p = float(cfg_dict.get("prior_dropout_p", 0.2))
    model = SpectreModel(
        vocab,
        prior_dropout_p=prior_dropout_p,
        use_atom_sab2=use_atom_sab2,
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


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

    model = _load_checkpoint(ckpt_path, vocab, device)

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


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
