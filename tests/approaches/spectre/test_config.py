"""Tests for ``spectre.config`` hashing and YAML round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphatamp.approaches.spectre.config import CollectionConfig


def _base_kwargs() -> dict:
    return {
        "env_id": "kinder/ClutteredStorage2D-b5-v0",
        "env_variant": "clutteredstorage2d_b5",
        "model_name": "clutteredstorage2d",
        "model_kwargs": {"num_blocks": 5},
        "split": "train",
        "num_problems": 5,
        "problem_seed_start": 0,
        "problem_seed_end": 5,
    }


def test_config_hash_is_stable_and_created_at_insensitive() -> None:
    """Two configs differing only in ``created_at`` must share a hash."""
    cfg1 = CollectionConfig(**_base_kwargs())
    cfg2 = CollectionConfig(**_base_kwargs())
    # Force divergent timestamps; hash must not care.
    object.__setattr__(cfg2, "created_at", "2099-01-01T00:00:00+00:00")
    assert cfg1.config_hash == cfg2.config_hash


def test_config_hash_changes_when_budget_changes() -> None:
    """Bumping a budget field produces a fresh hash — triggers re-collection."""
    cfg1 = CollectionConfig(**_base_kwargs())
    cfg2 = CollectionConfig(**{**_base_kwargs(), "K_max": 100})
    assert cfg1.config_hash != cfg2.config_hash


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """``to_yaml`` + ``from_yaml`` preserves the config hash."""
    cfg = CollectionConfig(**_base_kwargs())
    path = tmp_path / f"collection_{cfg.config_hash}.yaml"
    cfg.to_yaml(path)
    loaded = CollectionConfig.from_yaml(path)
    assert loaded.config_hash == cfg.config_hash


def test_collect_instrumentation_true_is_rejected() -> None:
    """v0.1 does not support the instrumentation flag."""
    with pytest.raises(NotImplementedError):
        CollectionConfig(**{**_base_kwargs(), "collect_instrumentation": True})


def test_invalid_seed_range_rejected() -> None:
    """Empty or inverted seed ranges are rejected."""
    with pytest.raises(ValueError):
        CollectionConfig(
            **{**_base_kwargs(), "problem_seed_start": 5, "problem_seed_end": 5}
        )
