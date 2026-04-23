"""Tests for ``spectre.canonicalize``: renaming + permutation semantics."""

from __future__ import annotations

import numpy as np
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre.canonicalize import canonicalize_episode


def _atom_strs(ep) -> set[str]:
    return {str(a) for a in ep.initial_abstract_state.atoms}


def test_canonical_names_are_typed_local_ids() -> None:
    """After canonicalization, every object name is ``"{type}_{idx}"``."""
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    canon = canonicalize_episode(ep, rng=None)
    for obj_name, type_name in canon.object_registry.items():
        assert obj_name.startswith(type_name + "_"), (obj_name, type_name)
        idx_str = obj_name.rsplit("_", 1)[1]
        assert idx_str.isdigit()


def test_deterministic_without_rng() -> None:
    """Two ``rng=None`` canonicalizations produce identical records."""
    ep = build_toy_episode()
    a = canonicalize_episode(ep, rng=None)
    b = canonicalize_episode(ep, rng=None)
    assert _atom_strs(a) == _atom_strs(b)
    assert a.object_registry == b.object_registry


def test_permutation_stays_within_type() -> None:
    """Random permutations never mix object names across types."""
    ep = build_toy_episode(num_blocks=4, outcomes=("fail", "fail", "fail", "success"))
    canon = canonicalize_episode(ep, rng=np.random.default_rng(0))
    for obj_name, type_name in canon.object_registry.items():
        assert obj_name.startswith(type_name + "_"), (obj_name, type_name)
    blocks = [n for n, t in canon.object_registry.items() if t == "block"]
    assert sorted(blocks) == [f"block_{i}" for i in range(4)]


def test_canonical_skeleton_references_canonical_objects() -> None:
    """Every operator's arg names match the renumbered registry."""
    ep = build_toy_episode(outcomes=("fail", "success"))
    canon = canonicalize_episode(ep, rng=None)
    registry_names = set(canon.object_registry)
    for skel in canon.skeleton_pool:
        for op in skel.operator_seq:
            for arg in op.parameters:
                assert arg.name in registry_names


def test_preserves_outcomes_and_summary() -> None:
    """Canonicalization does not touch outcomes, provenance, or summary."""
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    canon = canonicalize_episode(ep, rng=np.random.default_rng(0))
    assert canon.outcomes == ep.outcomes
    assert canon.summary == ep.summary
    assert canon.provenance == ep.provenance
