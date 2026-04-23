"""Tests for ``spectre.vocab``: extract + validate + JSON round-trip."""

from __future__ import annotations

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
