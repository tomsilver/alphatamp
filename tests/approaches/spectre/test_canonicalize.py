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


def test_class2_deviation_object_names_are_renamed() -> None:
    """Object names *inside* ``dev_added``/``dev_deleted`` follow the renaming.

    This is the nested case of the trap ``_remap_refiner_metadata`` exists for: if the
    arguments of a serialized deviation atom keep their raw names, they stop resolving
    against the canonical scene tags and every class-2 record degenerates to "some
    failure of some schema" -- silently, because nothing raises on an unresolved tag.
    """
    ep = build_toy_episode(num_blocks=4, outcomes=("fail", "fail", "fail", "success"))
    raw_name = next(n for n, t in ep.object_registry.items() if t == "block")
    ep.outcomes[0].refiner_metadata["failures"] = [
        {
            "step_index": 0,
            "schema": "pick",
            "args": [raw_name],
            "culprits": [],
            "dev_added": [["Pressed", [raw_name]], ["AboveNoButton", []]],
            "dev_deleted": [["Holding", [raw_name, raw_name]]],
            "dev_blame": [raw_name],
        }
    ]

    # A permutation makes the mapping non-trivial, so a pass-through would be visible.
    canon = canonicalize_episode(ep, rng=np.random.default_rng(3))
    entry = canon.outcomes[0].refiner_metadata["failures"][0]  # type: ignore[index]
    canonical = entry["args"][0]
    assert canonical in canon.object_registry
    # The deviation must follow the *same* mapping as the flat `args` list; any other
    # name here is a tag that will not resolve against the scene tokens.
    assert entry["dev_added"] == [["Pressed", [canonical]], ["AboveNoButton", []]]
    assert entry["dev_deleted"] == [["Holding", [canonical, canonical]]]
    assert entry["dev_blame"] == [canonical]


def test_metadata_without_deviation_keys_is_untouched() -> None:
    """Exact absence: a DD2D-shaped entry gains no keys it did not have."""
    ep = build_toy_episode(outcomes=("fail", "success"))
    ep.outcomes[0].refiner_metadata["failures"] = [
        {"step_index": 0, "schema": "retrieve", "args": [], "culprits": ["__wall__"]}
    ]
    entry = (
        canonicalize_episode(ep, rng=None)
        .outcomes[0]
        .refiner_metadata["failures"][0]  # type: ignore[index]
    )
    assert "dev_added" not in entry and "dev_blame" not in entry
    # A non-object sentinel stays visible as itself rather than being dropped.
    assert entry["culprits"] == ["__wall__"]
