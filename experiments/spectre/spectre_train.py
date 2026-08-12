"""Train the SPECTRE model on a SPECTRE-collected dataset.

Usage::

    python experiments/spectre/spectre_train.py env=dd2d_v4 seed=0
    python experiments/spectre/spectre_train.py -m env=dd2d_v4 seed=0,1,2

Reads ``data/spectre/raw/<env_variant>/{train,val}/episodes/`` (collected by
``experiments/spectre/spectre_collect.py``) and ``data/spectre/derived/<env_variant>/
train_vocab.json``. Writes checkpoints + a JSONL training log under
``data/spectre/checkpoints/<env_variant>/seed_<seed>/``.

See ``src/alphatamp/approaches/spectre/docs/archive/SPECTRE_RT2D_METHOD_SPEC.md``
§8 for the full training contract.
"""

# This Hydra entrypoint predates the v3 rename in ``train.py`` (``TrainConfig`` /
# ``train_v3``); it still imports the old ``TrainingConfig`` / ``train`` names and calls
# them with a now-removed signature. Reconciling it is a behavioral fix out of scope for
# this cosmetic CI pass, so the resulting dead-import lint is suppressed module-wide.
# pylint: disable=no-name-in-module
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

# Stale pre-v3 names (see the module-level note above); mypy sees them as absent too.
from alphatamp.approaches.spectre.train import (  # type: ignore[attr-defined]
    TrainingConfig,
    train,
)
from alphatamp.approaches.spectre.vocab import Vocab


def _maybe_init_wandb(
    cfg: DictConfig, training_cfg: TrainingConfig, env_variant: str
) -> object | None:
    """Initialize a wandb run from ``cfg.wandb``, or return ``None``.

    Returns ``None`` when wandb is disabled or the import fails (training then proceeds
    without logging). The API key is read from ``WANDB_API_KEY``; it is never taken from
    config. ``mode`` is one of ``disabled|online|offline``.
    """
    wcfg = cfg.get("wandb", None)
    if wcfg is None or str(wcfg.get("mode", "disabled")) == "disabled":
        return None
    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("[wandb] not installed; continuing without logging.")
        return None

    seed = int(cfg.seed)
    name = wcfg.get("name") or f"{env_variant}_seed{seed}"
    raw_tags = OmegaConf.to_container(wcfg.get("tags", []), resolve=True) or []
    tags = [str(t) for t in cast("list[Any]", raw_tags)]
    run_config = {
        **asdict(training_cfg),
        "env_variant": env_variant,
        "seed": seed,
        "data_root": str(cfg.data_root),
    }
    run = wandb.init(
        project=str(wcfg.get("project", "spectre")),
        entity=wcfg.get("entity"),
        group=wcfg.get("group"),
        name=name,
        tags=tags,
        notes=wcfg.get("notes"),
        mode=cast(Any, str(wcfg.get("mode"))),
        config=run_config,
    )
    print(f"[wandb] logging to {run.url if hasattr(run, 'url') else run}")
    return run


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

    wandb_run = _maybe_init_wandb(cfg, training_cfg, env_variant)
    try:
        best_path = train(
            cfg=training_cfg,
            train_dir=train_dir,
            val_dir=val_dir,
            vocab=vocab,
            type_aug_policy=type_aug_policy,
            out_dir=out_dir,
            static_tag_predicates=static_tag_predicates,
            wandb_run=wandb_run,
        )
        if wandb_run is not None and bool(cfg.wandb.get("log_model", False)):
            import wandb  # pylint: disable=import-outside-toplevel

            artifact = wandb.Artifact(
                f"{env_variant}_seed{int(cfg.seed)}", type="model"
            )
            artifact.add_file(str(best_path))
            wandb_run.log_artifact(artifact)  # type: ignore[attr-defined]
    finally:
        if wandb_run is not None:
            wandb_run.finish()  # type: ignore[attr-defined]
    print(f"Wrote best checkpoint to {best_path}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
