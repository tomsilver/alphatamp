"""Tests for the DD2D -> SPECTRE ``EpisodeRecord`` converter.

Exercises the real migrated raw_v2 dataset when present; skips cleanly when the
(gitignored) data is absent so CI without it still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import alphatamp.approaches.spectre as spectre_pkg
from alphatamp.approaches.spectre.envs.dd2d.spectre_convert import (
    CONVERTER_VERSION,
    config_hash,
    convert_problem_dir,
)
from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import (
    ALL_PREDICATES,
    OPERATOR_BY_NAME,
)
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory
from alphatamp.approaches.spectre.vocab import OOV_TOKEN, extract_vocab

_RAW_V2 = Path(spectre_pkg.__file__).resolve().parents[4] / "data" / "dd2d" / "raw_v2"


def _first_train_problem() -> Path:
    train = _RAW_V2 / "train"
    if not train.exists():
        pytest.skip(f"DD2D raw_v2 dataset not present at {_RAW_V2}")
    dirs = sorted(p for p in train.glob("dd2d_*") if p.is_dir())
    if not dirs:
        pytest.skip("DD2D raw_v2/train has no problem directories")
    return dirs[0]


def test_convert_problem_dir_shape_and_invariants() -> None:
    """A converted problem dir yields a valid EpisodeRecord with matching counts."""
    problem_dir = _first_train_problem()
    records = sorted(problem_dir.glob("[0-9]*.json"))

    ep = convert_problem_dir(problem_dir, split="train")
    ep.validate()  # I1-I4; also runs in __post_init__

    # One skeleton + one outcome per JSON record.
    assert ep.summary.num_skeletons == len(records)
    assert len(ep.skeleton_pool) == len(records)
    assert len(ep.outcomes) == len(records)

    # DD2D labels are bool -> success/fail only; no errors.
    assert ep.summary.num_error == 0
    assert ep.summary.num_success + ep.summary.num_fail == ep.summary.num_skeletons

    # Counts equal the raw label tallies.
    labels = [bool(json.loads(p.read_text(encoding="utf-8"))["label"]) for p in records]
    assert ep.summary.num_success == sum(labels)
    assert ep.summary.num_fail == len(labels) - sum(labels)
    assert ep.summary.num_success >= 1  # persisted DD2D problems are solvable


def test_initial_state_drops_geometry() -> None:
    """s_0 carries only STRIPS drawer atoms; at-pose geometry is dropped."""
    ep = convert_problem_dir(_first_train_problem(), split="train")
    domain_preds = {p.name for p in ALL_PREDICATES}
    for atom in ep.initial_abstract_state.atoms:
        assert atom.predicate.name in domain_preds
    assert not any(
        a.predicate.name == "at-pose" for a in ep.initial_abstract_state.atoms
    )
    # Every object is an item; goal is extracted(target).
    assert set(ep.object_registry.values()) == {"item"}
    assert sorted(str(a) for a in ep.goal_atoms) == ["(extracted target)"]


def test_all_skeletons_reconstruct() -> None:
    """Every stored skeleton progresses under STRIPS with preconditions satisfied."""
    ep = convert_problem_dir(_first_train_problem(), split="train")
    for skel in ep.skeleton_pool:
        traj = reconstruct_trajectory(
            ep.initial_abstract_state, skel.operator_seq, verify_preconditions=True
        )
        assert traj[-1] == skel.final_abstract_state


def test_operator_names_are_drawer_domain() -> None:
    """Grounded operators use exactly the drawer domain names."""
    ep = convert_problem_dir(_first_train_problem(), split="train")
    seen = {op.name for skel in ep.skeleton_pool for op in skel.operator_seq}
    assert seen <= set(OPERATOR_BY_NAME)
    assert seen <= {"pick", "place-buffer", "retrieve"}


def test_vocab_over_converted_episode(tmp_path: Path) -> None:
    """extract_vocab on a converted episode recovers the full drawer vocabulary."""
    ep = convert_problem_dir(_first_train_problem(), split="train")
    split_dir = tmp_path / "train"
    atomic_write_pickle_gz(ep, split_dir / "episodes" / "ep_00000.pkl.gz")

    vocab = extract_vocab(split_dir, config_hash=config_hash("dd2d_v2"))
    ops = set(vocab.operators) - {OOV_TOKEN}
    preds = set(vocab.predicates) - {OOV_TOKEN}
    types = set(vocab.types) - {OOV_TOKEN}

    assert ops == {"pick", "place-buffer", "retrieve"}
    # All six predicates appear once intermediate states are reconstructed
    # (on-buffer / holding / extracted live only mid-trajectory).
    assert preds == {p.name for p in ALL_PREDICATES}
    assert types == {"item"}


def test_config_hash_is_deterministic() -> None:
    """The stamped config hash is stable and version-keyed."""
    assert config_hash("dd2d_v2") == config_hash("dd2d_v2")
    assert config_hash("dd2d_v2") != config_hash("other")
    # v2 added SceneGeometry; v3 adds the refiner's typed failure observations,
    # per-candidate wall-clock and generation params. The pin is deliberate: the hash is
    # keyed on this string, so a schema change to the converter must be a conscious edit.
    assert CONVERTER_VERSION == "dd2d_convert_v3"
