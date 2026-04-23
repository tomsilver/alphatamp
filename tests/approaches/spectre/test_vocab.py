"""Tests for ``spectre.vocab``: extract + validate + JSON round-trip."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest
from _fixtures import write_toy_split

from alphatamp.approaches.spectre.vocab import (
    OOV_TOKEN,
    Vocab,
    extract_vocab,
    validate_vocab,
)


def test_extract_vocab_indexes_oov_at_zero(tmp_path: Path) -> None:
    """Index 0 is reserved for ``<OOV>`` in every table."""
    write_toy_split(
        tmp_path / "train",
        outcomes_per_problem=[("fail", "success"), ("success", "fail")],
    )
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    assert vocab.operators[OOV_TOKEN] == 0
    assert vocab.types[OOV_TOKEN] == 0
    assert vocab.predicates[OOV_TOKEN]["idx"] == 0


def test_extract_vocab_captures_toy_domain(tmp_path: Path) -> None:
    """Expected operators / predicates / types from the toy domain."""
    write_toy_split(
        tmp_path / "train",
        outcomes_per_problem=[("fail", "success"), ("fail", "success", "fail")],
    )
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    assert "Pick" in vocab.operators
    assert "Place" in vocab.operators
    assert "OnTable" in vocab.predicates
    assert "Holding" in vocab.predicates
    assert "Clear" in vocab.predicates
    assert "block" in vocab.types
    assert "robot" in vocab.types
    assert vocab.max_skeleton_length == 2


def test_json_roundtrip(tmp_path: Path) -> None:
    """``Vocab.to_json`` + ``from_json`` round-trip cleanly."""
    write_toy_split(tmp_path / "train", [("fail", "success")])
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    out = tmp_path / "train_vocab.json"
    vocab.to_json(out)
    loaded = Vocab.from_json(out)
    assert loaded.operators == vocab.operators
    assert loaded.predicates == vocab.predicates
    assert loaded.types == vocab.types


def test_validate_vocab_clean(tmp_path: Path) -> None:
    """Val split drawn from the same domain reports no OOV findings."""
    write_toy_split(tmp_path / "train", [("fail", "success")])
    write_toy_split(tmp_path / "val", [("success", "fail")])
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    assert not validate_vocab(vocab, tmp_path / "val")


def test_vocab_raises_on_oov_lookup(tmp_path: Path) -> None:
    """``vocab.op_idx`` hard-fails on unknown names per SPEC §7.2."""
    write_toy_split(tmp_path / "train", [("fail", "success")])
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    with pytest.raises(KeyError, match="Unknown lifted operator"):
        vocab.op_idx("NeverSeen")


def test_extract_vocab_finds_intermediate_state_predicates(tmp_path: Path) -> None:
    """Regression: ``Holding`` appears only between Pick and Place.

    Because Substage A stores only ``s_0`` and ``s_L``, a naive atom-scan
    misses predicates that are added-then-deleted within a skeleton. The
    fixture's cycle Pick → Place restores ``s_0`` in ``s_L``, so
    ``HOLDING`` never appears in any stored atom set — only in the
    reconstructed intermediate state. ``extract_vocab`` must pick it up
    via STRIPS progression.
    """
    write_toy_split(tmp_path / "train", [("success",)])
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    stored_predicate_names: set[str] = set()
    for path in (tmp_path / "train" / "episodes").glob("ep_*.pkl.gz"):
        # Sanity: confirm the fixture's STORED atoms genuinely lack Holding.
        # If they do, extract_vocab's previous behavior would have missed it.
        with gzip.open(path, "rb") as f:
            ep = pickle.load(f)
        for atom in ep.initial_abstract_state.atoms:
            stored_predicate_names.add(atom.predicate.name)
        for skel in ep.skeleton_pool:
            for atom in skel.final_abstract_state.atoms:
                stored_predicate_names.add(atom.predicate.name)
        for atom in ep.goal_atoms:
            stored_predicate_names.add(atom.predicate.name)
    # Sanity on fixture: Holding genuinely absent from stored states/goal.
    assert "Holding" not in stored_predicate_names, (
        "Fixture drifted: Holding appears in stored atoms so the regression"
        " is no longer exercising the intermediate-state path."
    )
    # The actual fix: extract_vocab still sees it via trajectory reconstruction.
    assert "Holding" in vocab.predicates


def test_validate_vocab_flags_missing_intermediate_predicate(
    tmp_path: Path,
) -> None:
    """``validate_vocab`` must also walk trajectories — not just stored atoms."""
    write_toy_split(tmp_path / "train", [("success",)])
    vocab = extract_vocab(tmp_path / "train", config_hash="abc")
    # Deliberately corrupt the vocab: drop ``Holding``. The val-side check
    # should rediscover that the intermediate trajectory references an
    # unknown predicate.
    dropped_predicates = {k: v for k, v in vocab.predicates.items() if k != "Holding"}
    broken_vocab = Vocab(
        config_hash=vocab.config_hash,
        operators=vocab.operators,
        predicates=dropped_predicates,
        types=vocab.types,
        max_operator_arity=vocab.max_operator_arity,
        max_predicate_arity=vocab.max_predicate_arity,
        max_skeleton_length=vocab.max_skeleton_length,
        max_atoms_per_state=vocab.max_atoms_per_state,
        max_objects_per_state=vocab.max_objects_per_state,
        max_pool_size=vocab.max_pool_size,
        max_objects_per_type=vocab.max_objects_per_type,
    )
    findings = validate_vocab(broken_vocab, tmp_path / "train")
    assert any(
        "Holding" in f for f in findings
    ), f"Expected a Holding finding from intermediate-state walk; got {findings}"
