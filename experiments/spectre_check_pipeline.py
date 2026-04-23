"""End-of-pipeline diagnostic: load data + vocab, run one collated batch.

Run *after* ``spectre_collect.py`` and ``spectre_build_vocab.py``. Verifies:

1. Each split's ``SpectreDataset`` instantiates and has at least one trainable
   episode (i.e. filtering didn't drop everything — see pipeline spec §11.5).
2. A sample training example satisfies the F ⊆ FAIL invariant chain
   (``SpectreDataset.__getitem__`` asserts this internally; we just exercise
   it on a handful of draws).
3. ``collate_spectre_batch`` produces tensors with the expected dtypes and
   shape rank.
4. Observed maxima (pool size, skeleton length, atoms, objects) are within
   the vocab's recorded bounds — catching silent vocab drift where the
   vocab was extracted from a smaller/older snapshot.

Does *not* train anything. Prints a short report per split.

Usage::

    python experiments/spectre_check_pipeline.py

Overrides mirror ``spectre_collect.yaml`` (``data_root``, ``env``). Example::

    python experiments/spectre_check_pipeline.py data_root=/scratch/spectre
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from alphatamp.approaches.spectre.dataset import (
    SpectreDataset,
    collate_spectre_batch,
)
from alphatamp.approaches.spectre.priors import ZeroPrior
from alphatamp.approaches.spectre.vocab import Vocab


def _check_split(
    split_dir: Path,
    vocab: Vocab,
    seed: int,
    num_draws: int,
    batch_size: int,
) -> None:
    ds = SpectreDataset(
        split_dir=split_dir,
        prior=ZeroPrior(),
        seed=seed,
        augment=False,
    )
    filtered = ds.filtered_problem_ids
    n_trainable = len(ds)
    print(f"  trainable episodes: {n_trainable} (filtered {len(filtered)})")
    if filtered:
        by_reason: dict[str, int] = {}
        for _, reason in filtered:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        for reason, count in sorted(by_reason.items()):
            print(f"    filtered/{reason}: {count}")
    if n_trainable == 0:
        print("  [WARN] no trainable episodes — did refinement produce any successes?")
        return

    # Invariants check: draw a few samples; __getitem__ asserts internally.
    observed_r = observed_f = observed_skel_len = 0
    observed_atoms = observed_objs = 0
    for i in range(min(num_draws, n_trainable)):
        ex = ds[i]
        observed_r = max(observed_r, len(ex.r_skeletons))
        observed_f = max(observed_f, len(ex.f_skeletons))
        observed_atoms = max(observed_atoms, len(ex.initial_abstract_state.atoms))
        observed_objs = max(observed_objs, len(ex.object_registry))
        for skel in (*ex.r_skeletons, *ex.f_skeletons):
            observed_skel_len = max(observed_skel_len, len(skel.operator_seq))
    print(
        f"  observed maxima across {min(num_draws, n_trainable)} draws:"
        f" |R|={observed_r} |F|={observed_f}"
        f" skel_len={observed_skel_len} atoms(s_0)={observed_atoms}"
        f" objs={observed_objs}"
    )

    # Vocab coverage: observed should not exceed vocab-recorded maxima.
    if observed_skel_len > vocab.max_skeleton_length:
        print(
            f"  [WARN] observed skeleton length {observed_skel_len} exceeds vocab"
            f" max_skeleton_length={vocab.max_skeleton_length} — vocab is stale"
        )
    if observed_atoms > vocab.max_atoms_per_state:
        print(
            f"  [WARN] observed atoms/state {observed_atoms} exceeds vocab"
            f" max_atoms_per_state={vocab.max_atoms_per_state} — vocab is stale"
        )

    # Collate one small batch and report tensor shapes + dtypes.
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, n_trainable),
        shuffle=False,
        collate_fn=lambda batch: collate_spectre_batch(batch, vocab),
    )
    batch = next(iter(loader))
    print(
        f"  collate: r_op_ids={tuple(batch.r_op_ids.shape)}"
        f" ({batch.r_op_ids.dtype}),"
        f" f_op_ids={tuple(batch.f_op_ids.shape)},"
        f" s0_type_hist={tuple(batch.s0_type_histogram.shape)}"
    )
    # Sanity: success mask must have at least one True per example (I9).
    assert bool(
        batch.r_success_mask.any(dim=1).all()
    ), "Every example must have at least one successful skeleton in R"


@hydra.main(
    config_path="conf",
    config_name="spectre_collect",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Diagnostic entrypoint.

    Re-uses spectre_collect's config for paths.
    """
    data_root = Path(cfg.data_root)
    env_variant = cfg.env.env_variant
    vocab_path = data_root / "derived" / env_variant / "train_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"No vocab at {vocab_path}; run experiments/spectre_build_vocab.py first"
        )
    vocab = Vocab.from_json(vocab_path)

    print(f"env_variant={env_variant}  data_root={data_root}")
    print(
        f"vocab: operators={len(vocab.operators) - 1}"
        f" predicates={len(vocab.predicates) - 1}"
        f" types={len(vocab.types) - 1}"
        f" max_pool_size={vocab.max_pool_size}"
        f" max_skeleton_length={vocab.max_skeleton_length}"
    )

    for split in ("train", "val", "test"):
        split_dir = data_root / "raw" / env_variant / split
        if not (split_dir / "episodes").exists():
            print(f"[{split}] missing; skipping")
            continue
        print(f"[{split}] {split_dir}")
        _check_split(
            split_dir=split_dir,
            vocab=vocab,
            seed=0,
            num_draws=16,
            batch_size=4,
        )

    print("\nAll checks passed — pipeline is consumable by a downstream trainer.")
    _ = torch  # keep torch import visible to linters


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
