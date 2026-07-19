"""v2.2.1 schema geometry/evidence layer: round-trip, guarded invariants, and the
legacy-pickle migration shim (RT2D/kinder records written before these fields existed
must still load)."""

from __future__ import annotations

import dataclasses
import gzip
import pickle

import pytest
from _fixtures import build_toy_episode

from alphatamp.approaches.spectre.io import atomic_write_pickle_gz, load_episode
from alphatamp.approaches.spectre.schema import (
    AuxLabels,
    Fact,
    ObjectGeometry,
    PostMortemRecord,
    SceneGeometry,
)

_UNIT_RING = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))


def _geometry_for(ep) -> SceneGeometry:
    """A SceneGeometry covering every registered object (satisfies I5)."""
    objs = tuple(
        ObjectGeometry(
            name=name,
            pose=(float(i), 0.0, 0.0),
            boundary=_UNIT_RING,
            family="test",
            area=1.0,
            concave=False,
            is_target=(i == 0),
        )
        for i, name in enumerate(sorted(ep.object_registry))
    )
    return SceneGeometry(
        objects=objs, containers=(), units="cm", frame={"drawer_w": 10.0}
    )


def test_geometry_roundtrip(tmp_path):
    ep = build_toy_episode()
    ep = dataclasses.replace(
        ep,
        scene_geometry=_geometry_for(ep),
        aux_labels=AuxLabels(
            necessary=frozenset({"block_0"}), relevant=frozenset({"block_0", "block_1"})
        ),
    )
    path = tmp_path / "episodes" / "ep_0.pkl.gz"
    atomic_write_pickle_gz(ep, path)
    loaded = load_episode(path)
    assert loaded == ep
    assert loaded.scene_geometry is not None
    assert {o.name for o in loaded.scene_geometry.objects} == set(ep.object_registry)
    assert loaded.aux_labels.necessary == frozenset({"block_0"})


def test_post_mortem_roundtrip(tmp_path):
    ep = build_toy_episode(outcomes=("fail", "success"))
    fail_out = dataclasses.replace(
        ep.outcomes[0],
        post_mortem=PostMortemRecord(
            skeleton_idx=0,
            refinement_seed=1000,
            failed_step_index=1,
            failed_schema="place-buffer",
            facts=(Fact("blocked-at-contents", ("block_1",), "proof"),),
        ),
    )
    ep = dataclasses.replace(ep, outcomes=(fail_out, ep.outcomes[1]))
    path = tmp_path / "episodes" / "ep_1.pkl.gz"
    atomic_write_pickle_gz(ep, path)
    loaded = load_episode(path)
    assert loaded.outcomes[0].post_mortem.facts[0].tier == "proof"
    assert loaded.outcomes[1].post_mortem is None


def test_i5_missing_geometry_raises():
    ep = build_toy_episode()
    partial = SceneGeometry(objects=(), containers=())  # covers no registered object
    with pytest.raises(AssertionError, match="I5"):
        dataclasses.replace(ep, scene_geometry=partial)


def test_i6_post_mortem_on_success_raises():
    ep = build_toy_episode(outcomes=("fail", "success"))
    bad = dataclasses.replace(
        ep.outcomes[1],  # a "success" outcome
        post_mortem=PostMortemRecord(skeleton_idx=1, refinement_seed=1),
    )
    with pytest.raises(AssertionError, match="I6"):
        dataclasses.replace(ep, outcomes=(ep.outcomes[0], bad))


def test_canonicalize_renames_geometry_consistently():
    """canonicalize must rename scene_geometry/aux_labels to the canonical ids so the
    canonicalized episode still satisfies I5 (regression: it previously left geometry with
    pre-canonical names → I5 violation in the v1 dataset path)."""
    import numpy as np

    from alphatamp.approaches.spectre.canonicalize import canonicalize_episode

    ep = build_toy_episode()
    ep = dataclasses.replace(
        ep,
        scene_geometry=_geometry_for(ep),
        aux_labels=AuxLabels(necessary=frozenset({"block_0"}), relevant=frozenset()),
    )
    # with augmentation (rng set) the names are permuted; geometry must follow.
    can = canonicalize_episode(ep, rng=np.random.default_rng(0))
    geo_names = {o.name for o in can.scene_geometry.objects}
    assert set(can.object_registry) <= geo_names  # I5 holds post-canonicalization
    assert can.aux_labels.necessary <= set(can.object_registry)
    # deterministic canonicalization (rng=None) also keeps them aligned.
    can2 = canonicalize_episode(ep)
    assert {o.name for o in can2.scene_geometry.objects} == set(can2.object_registry)


def test_no_geometry_still_valid():
    # RT2D/kinder-style record: no geometry, no post-mortems → validates & round-trips.
    ep = build_toy_episode()
    assert ep.scene_geometry is None and ep.aux_labels is None


def test_legacy_pickle_migrates(tmp_path):
    """A pickle written before the v2.2.1 fields existed (attrs absent from __dict__)
    loads via the shim with the new fields defaulted to None — no AttributeError."""
    ep = build_toy_episode()
    # Simulate an old on-disk record: strip the new attrs (bypassing frozen __delattr__).
    object.__delattr__(ep, "scene_geometry")
    object.__delattr__(ep, "aux_labels")
    object.__delattr__(ep.outcomes[0], "post_mortem")
    path = tmp_path / "episodes" / "ep_legacy.pkl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(ep, f, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_episode(path)
    assert loaded.scene_geometry is None
    assert loaded.aux_labels is None
    assert (
        loaded.outcomes[0].post_mortem is None
    )  # would AttributeError without the shim
