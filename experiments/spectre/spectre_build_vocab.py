"""Extract the SPECTRE vocab from a train split, write ``train_vocab.json``.

Run once after train collection completes::

    python experiments/spectre/spectre_build_vocab.py

OOV-checks val/test if their episodes are present, printing (not raising)
warnings on unknown operator / predicate / type names. Per
``src/alphatamp/approaches/spectre/docs/archive/SPECTRE_METHOD_SPEC.md`` §7.2,
v0.1 assumes no OOV at test time.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import hydra
from omegaconf import DictConfig

from alphatamp.approaches.spectre.env_registry import get_type_aug_policy
from alphatamp.approaches.spectre.io import load_episode
from alphatamp.approaches.spectre.vocab import (
    extract_vocab,
    validate_vocab,
)


@hydra.main(
    config_path="conf",
    config_name="spectre_build_vocab",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for SPECTRE vocab construction."""
    data_root = Path(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    train_dir = data_root / "raw" / env_variant / "train"
    if not (train_dir / "episodes").exists():
        raise FileNotFoundError(
            f"No train episodes under {train_dir}; run spectre_collect first."
        )

    # Vocab records the train split's config_hash for later sanity checks.
    first = next((train_dir / "episodes").glob("ep_*.pkl.gz"), None)
    assert first is not None
    train_hash = load_episode(first).provenance.config_hash

    vocab = extract_vocab(train_dir, config_hash=train_hash)

    # Inject the per-type augmentation policy from the env registry per
    # src/alphatamp/approaches/spectre/docs/archive/SPECTRE_RT2D_METHOD_SPEC.md
    # §10.1. Empty dict for kinder envs (which treat every type as
    # augmentable=True).
    type_aug_policy = get_type_aug_policy(env_variant)
    if type_aug_policy:
        vocab = replace(vocab, type_aug_policy=type_aug_policy)

    out = data_root / "derived" / env_variant / "train_vocab.json"
    vocab.to_json(out)
    print(f"Wrote vocab to {out}")
    print(
        f"  operators={len(vocab.operators) - 1} (excluding <OOV>);"
        f" predicates={len(vocab.predicates) - 1}; types={len(vocab.types) - 1}"
    )
    print(
        f"  max_skeleton_length={vocab.max_skeleton_length};"
        f" max_pool_size={vocab.max_pool_size}"
    )

    for split_name in cfg.validate_splits:
        split_dir = data_root / "raw" / env_variant / str(split_name)
        if not (split_dir / "episodes").exists():
            print(f"  [skip] {split_name}: no episodes")
            continue
        findings = validate_vocab(vocab, split_dir)
        if findings:
            print(f"  [{split_name}] OOV findings ({len(findings)}):")
            for f in findings[:10]:
                print(f"    - {f}")
            if len(findings) > 10:
                print(f"    ... ({len(findings) - 10} more)")
        else:
            print(f"  [{split_name}] clean")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
