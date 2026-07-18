"""Tests for ``spectre.env_registry``: bootstrapping non-default kinder ids."""

from __future__ import annotations

import gymnasium

from alphatamp.approaches.spectre.env_registry import (
    cluttered_storage_variants,
    register_extra_envs,
)


def test_b5_id_registers_after_bootstrap() -> None:
    """``ClutteredStorage2D-b5-v0`` is not default-registered; bootstrap exposes it."""
    register_extra_envs(cluttered_storage_variants([5]))
    assert "kinder/ClutteredStorage2D-b5-v0" in gymnasium.registry


def test_idempotent() -> None:
    """Calling register_extra_envs twice is a no-op."""
    register_extra_envs(cluttered_storage_variants([5]))
    before = len(gymnasium.registry)
    register_extra_envs(cluttered_storage_variants([5]))
    assert len(gymnasium.registry) == before


def test_does_not_clobber_default_ids() -> None:
    """Pre-registered defaults (``-b7``, ``-b15``) are untouched."""
    register_extra_envs(cluttered_storage_variants([5]))
    spec = gymnasium.registry["kinder/ClutteredStorage2D-b7-v0"]
    assert spec.kwargs["num_blocks"] == 7
