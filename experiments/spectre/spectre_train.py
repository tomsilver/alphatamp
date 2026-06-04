"""Train the SPECTRE model on a SPECTRE-collected dataset.

Usage::

    python experiments/spectre/spectre_train.py env=routedtransport2d_n3_v1 seed=0
    python experiments/spectre/spectre_train.py -m env=routedtransport2d_n3_v1 seed=0,1,2

Reads ``data/spectre/raw/<env_variant>/{train,val}/episodes/`` (collected by
``experiments/spectre/spectre_collect.py``) and ``data/spectre/derived/<env_variant>/
train_vocab.json``. Writes checkpoints + a JSONL training log under
``data/spectre/checkpoints/<env_variant>/seed_<seed>/``.

See ``src/alphatamp/approaches/spectre/docs/archive/SPECTRE_RT2D_METHOD_SPEC.md``
§8 for the full training contract.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from alphatamp.approaches.spectre.env_registry import (
    get_static_tag_predicates,
    get_type_aug_policy,
)
from alphatamp.approaches.spectre.train import TrainingConfig, train
from alphatamp.approaches.spectre.vocab import Vocab


def _build_training_config(cfg: DictConfig) -> TrainingConfig:
    raw = cast(dict[str, Any], OmegaConf.to_container(cfg.train, resolve=True))
    assert isinstance(raw, dict)
    # OmegaConf returns lists for tuple fields; convert back.
    if "f_sampling_mix_weights" in raw and isinstance(
        raw["f_sampling_mix_weights"], list
    ):
        raw["f_sampling_mix_weights"] = tuple(raw["f_sampling_mix_weights"])
    # The slurm wrapper passes ``seed=$SEED`` at the top-level Hydra
    # namespace (which also drives the out_dir). Top-level always wins
    # over ``train.seed`` so ``--array=1-3`` actually trains seeds 1/2/3
    # — not three copies of seed 0.
    raw["seed"] = int(cfg.seed)
    return TrainingConfig(**raw)


@hydra.main(
    config_path="conf",
    config_name="spectre_train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for SPECTRE model training."""
    data_root = Path(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    train_dir = data_root / "raw" / env_variant / "train"
    val_dir = data_root / "raw" / env_variant / "val"
    vocab_path = data_root / "derived" / env_variant / "train_vocab.json"
    if not (train_dir / "episodes").exists():
        raise FileNotFoundError(
            f"No train episodes under {train_dir}; run spectre_collect first."
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"No train_vocab.json at {vocab_path}; run spectre_build_vocab first."
        )

    vocab = Vocab.from_json(vocab_path)
    type_aug_policy = get_type_aug_policy(env_variant)
    if not type_aug_policy and vocab.type_aug_policy:
        # Fallback: use the policy already serialized in the vocab JSON.
        type_aug_policy = dict(vocab.type_aug_policy)
    static_tag_predicates = get_static_tag_predicates(env_variant)

    out_dir = Path(cfg.out_dir) / env_variant / f"seed_{int(cfg.seed)}"
    training_cfg = _build_training_config(cfg)

    print(f"Training SPECTRE on {env_variant}, seed={training_cfg.seed}")
    print(f"  train_dir={train_dir}")
    print(f"  val_dir={val_dir}")
    print(f"  vocab={vocab_path} (config_hash={vocab.config_hash})")
    print(f"  out_dir={out_dir}")
    print(f"  config={asdict(training_cfg)}")
    if training_cfg.use_static_tag_pool:
        print(f"  static_tag_predicates={static_tag_predicates}")

    best_path = train(
        cfg=training_cfg,
        train_dir=train_dir,
        val_dir=val_dir,
        vocab=vocab,
        type_aug_policy=type_aug_policy,
        out_dir=out_dir,
        static_tag_predicates=static_tag_predicates,
    )
    print(f"Wrote best checkpoint to {best_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
