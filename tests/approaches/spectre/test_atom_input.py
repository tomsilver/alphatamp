"""Tests for the atom-input feature (Rung A) — ``docs/spectre_atom_input_guide.md``.

SPECTRE gains the initial abstract state atoms and the goal atoms as input: each
object's scene token gets per-object profiles of the (init / goal) atoms mentioning it,
and 0-ary atoms route to the scorer's global token. The pathway is additive, zero-
initialized and config-gated, so ``atom_mode="off"`` is byte-identical to the pre-atom
model.

The guide's §5 tests, plus a checkpoint round-trip and collate-shape sanity. Mostly
pure-tensor / synthetic; the augmentation-consistency test needs the dd2d_v4 collection.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alphatamp.approaches.spectre.dataset import (
    SpectreExample,
    _atom_profile_arrays,
    _collate_base,
    atom_emission,
    build_example,
)
from alphatamp.approaches.spectre.encoders import AtomProfileEncoder
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model import SpectreConfig, SpectreModel
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"
_needs_data = pytest.mark.skipif(not _V4.is_dir(), reason="dd2d_v4 collection absent")

# Enough predicate rows that the synthetic atom ids (1..3) are always valid; arity 2 so
# the slot embedding has two rows (binary predicates exist, e.g. SB2D RobotAboveButton).
_N_PRED = 8
_ARITY = 2


# --------------------------------------------------------------------------- #
# synthetic example / vocab harness (mirrors test_pointset_encoder)
# --------------------------------------------------------------------------- #
def _atom_example(
    *, n_obj: int = 3, k: int = 2, seed: int = 0, with_atoms: bool = True
):
    """A v1-style example (FootprintEncoder path) with optional init/goal atom arrays.

    obj_tags are ``1..n_obj``. Init = {p(o1), p(o2), <0-ary>}; goal = {on(o1, o3)} — a
    binary atom, so the slot path is exercised.
    """
    rng = np.random.default_rng(seed)
    kwargs = dict(
        obj_tags=np.arange(1, n_obj + 1, dtype=np.int64),
        obj_boundary=rng.standard_normal((n_obj, 32, 2)).astype(np.float32),
        obj_pose=rng.standard_normal((n_obj, 3)).astype(np.float32),
        obj_rel=rng.standard_normal((n_obj, 3)).astype(np.float32),
        obj_is_goal=np.array([1.0] + [0.0] * (n_obj - 1), dtype=np.float32),
        op_ids=[[1, 2] for _ in range(k)],
        arg_tags=[[[1, 2], [2, 3]] for _ in range(k)],
        success=[True, False][:k],
        aux_necessary=np.full(n_obj, -1.0, np.float32),
        aux_relevant=np.full(n_obj, -1.0, np.float32),
        avail=[True] * k,
        fact_type_ids=[],
        fact_tier_ids=[],
        fact_arg_tags=[],
        prior=[[0.0, 0.0] for _ in range(k)],
        overlap=[[0.0, 0.0] for _ in range(k)],
    )
    if with_atoms:
        init_pred = np.array([2, 2, 1], dtype=np.int64)  # p, p, and a 0-ary atom
        init_arg = np.zeros((3, _ARITY), dtype=np.int64)
        init_arg[0, 0] = 1  # p(o1)
        init_arg[1, 0] = 2  # p(o2)
        goal_pred = np.array([3], dtype=np.int64)  # binary
        goal_arg = np.zeros((1, _ARITY), dtype=np.int64)
        goal_arg[0] = [1, 3]  # on(o1, o3)
        kwargs.update(
            init_atom_pred=init_pred,
            init_atom_arg_tags=init_arg,
            goal_atom_pred=goal_pred,
            goal_atom_arg_tags=goal_arg,
        )
    return SpectreExample(**kwargs)  # type: ignore[arg-type]


def _model(
    *, atom_mode: str, evidence_attn: bool = False, seed: int = 0
) -> SpectreModel:
    torch.manual_seed(seed)
    return SpectreModel(
        n_ops=5,
        max_arity=2,
        cfg=SpectreConfig(
            n_overlap_feats=0,
            dropout_p=0.0,
            evidence_attn=evidence_attn,
            atom_mode=atom_mode,
            n_predicates=_N_PRED,
            max_pred_arity=_ARITY,
        ),
    ).eval()


def _enc(
    seed: int = 0, use_init: bool = True, use_goal: bool = True
) -> AtomProfileEncoder:
    torch.manual_seed(seed)
    return AtomProfileEncoder(
        n_predicates=_N_PRED,
        max_pred_arity=_ARITY,
        max_tags=8,
        use_init=use_init,
        use_goal=use_goal,
    ).eval()


# --------------------------------------------------------------------------- #
# T1 — config-off equivalence / additivity
# --------------------------------------------------------------------------- #
def test_config_off_builds_no_atom_module() -> None:
    model = _model(atom_mode="off")
    assert model.atoms is None
    keys = list(model.state_dict().keys())
    assert not any(k.startswith("atoms.") for k in keys)


def test_config_off_batch_has_no_atom_fields() -> None:
    batch = _collate_base([_atom_example(with_atoms=False, seed=1)], max_arity=2)
    assert batch.init_atom_pred is None
    assert batch.init_atom_arg_tags is None
    assert batch.goal_atom_pred is None
    assert batch.goal_atom_arg_tags is None


def test_config_off_ignores_atoms_in_batch() -> None:
    """An off model produces identical logits whether or not the batch carries atoms."""
    off = _model(atom_mode="off")
    with_atoms = _collate_base([_atom_example(with_atoms=True, seed=3)], max_arity=2)
    without = _collate_base([_atom_example(with_atoms=False, seed=3)], max_arity=2)
    with torch.no_grad():
        assert torch.equal(off(with_atoms)[0], off(without)[0])


def test_profiles_on_selects_atom_module() -> None:
    model = _model(atom_mode="profiles")
    assert model.atoms is not None
    assert any(k.startswith("atoms.") for k in model.state_dict().keys())


def test_atom_mode_tokens_is_reserved_not_built() -> None:
    with pytest.raises(NotImplementedError):
        _model(atom_mode="tokens")


# --------------------------------------------------------------------------- #
# T2 — zero-init equivalence: profiles at init == off, on a batch carrying atoms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("evidence_attn", [False, True])
def test_zero_init_profiles_is_a_no_op_at_init(evidence_attn: bool) -> None:
    off = _model(atom_mode="off", evidence_attn=evidence_attn, seed=0)
    on = _model(atom_mode="profiles", evidence_attn=evidence_attn, seed=0)
    # atoms is built last, so every pre-existing parameter keeps its init draw.
    shared = off.state_dict()
    on_sd = on.state_dict()
    assert all(torch.equal(shared[k], on_sd[k]) for k in shared)
    assert on.atoms is not None
    assert torch.count_nonzero(on.atoms.obj_proj.weight) == 0
    assert torch.count_nonzero(on.atoms.glob_proj_atom.weight) == 0

    batch = _collate_base([_atom_example(with_atoms=True, seed=5)], max_arity=2)
    assert (
        batch.init_atom_pred is not None and int((batch.init_atom_pred != 0).sum()) > 0
    )
    with torch.no_grad():
        assert torch.equal(off(batch)[0], on(batch)[0])


def test_learned_profiles_change_the_logits() -> None:
    """After perturbing the (zero-init) projections, atoms actually move the ranking."""
    on = _model(atom_mode="profiles", seed=0)
    assert on.atoms is not None
    with torch.no_grad():
        on.atoms.obj_proj.weight.normal_()
        on.atoms.glob_proj_atom.weight.normal_()
    with_atoms = _collate_base([_atom_example(with_atoms=True, seed=6)], max_arity=2)
    without = _collate_base([_atom_example(with_atoms=False, seed=6)], max_arity=2)
    with torch.no_grad():
        assert not torch.allclose(on(with_atoms)[0], on(without)[0])


# --------------------------------------------------------------------------- #
# T3 — atom-set permutation invariance (scatter-sum is order-free)
# --------------------------------------------------------------------------- #
def test_atom_set_permutation_invariance() -> None:
    enc = _enc()
    obj_tags = torch.tensor([[1, 2, 3]])
    pred = torch.tensor([[2, 2, 1]])  # two unary + a 0-ary
    arg = torch.zeros(1, 3, _ARITY, dtype=torch.long)
    arg[0, 0, 0] = 1
    arg[0, 1, 0] = 2
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        o0, g0 = enc._profiles(pred, arg, obj_tags)
        o1, g1 = enc._profiles(pred[:, perm], arg[:, perm], obj_tags)
    assert torch.allclose(o0, o1, atol=1e-6)
    assert torch.allclose(g0, g1, atol=1e-6)


# --------------------------------------------------------------------------- #
# T4 — argument-slot sensitivity and predicate binding
# --------------------------------------------------------------------------- #
def test_arg_slot_is_positional_on_a_b_vs_b_a() -> None:
    """``on(a, b)`` and ``on(b, a)`` produce different profiles (slot is encoded)."""
    enc = _enc()
    obj_tags = torch.tensor([[1, 2]])
    ab = torch.tensor([[[1, 2]]])  # on(o1, o2)
    ba = torch.tensor([[[2, 1]]])  # on(o2, o1)
    pred = torch.tensor([[3]])
    with torch.no_grad():
        prof_ab, _ = enc._profiles(pred, ab, obj_tags)
        prof_ba, _ = enc._profiles(pred, ba, obj_tags)
    assert not torch.allclose(prof_ab, prof_ba)


def test_predicate_binds_to_its_own_arguments() -> None:
    """``{q(a), r(b)}`` and ``{q(b), r(a)}`` differ (per-atom projection before
    pool)."""
    enc = _enc()
    obj_tags = torch.tensor([[1, 2]])
    pred = torch.tensor([[2, 3]])  # q, r
    one = torch.zeros(1, 2, _ARITY, dtype=torch.long)
    one[0, 0, 0] = 1  # q(o1)
    one[0, 1, 0] = 2  # r(o2)
    two = torch.zeros(1, 2, _ARITY, dtype=torch.long)
    two[0, 0, 0] = 2  # q(o2)
    two[0, 1, 0] = 1  # r(o1)
    with torch.no_grad():
        prof_one, _ = enc._profiles(pred, one, obj_tags)
        prof_two, _ = enc._profiles(pred, two, obj_tags)
    assert not torch.allclose(prof_one, prof_two)


def test_zero_ary_atom_routes_to_the_global_term_only() -> None:
    """A 0-ary atom contributes to the global term and to no per-object profile."""
    enc = _enc()
    obj_tags = torch.tensor([[1, 2, 3]])
    pred = torch.tensor([[1]])  # a single 0-ary atom (all-PAD args)
    arg = torch.zeros(1, 1, _ARITY, dtype=torch.long)
    with torch.no_grad():
        obj, glob = enc._profiles(pred, arg, obj_tags)
    assert torch.count_nonzero(obj) == 0  # no object is named
    assert torch.count_nonzero(glob) > 0  # but the global term picks it up


# --------------------------------------------------------------------------- #
# T5 — augmentation consistency: atom arg tags share the scene tag namespace
# --------------------------------------------------------------------------- #
def _dd2d_episode_with_failures() -> tuple:
    paths = list_episodes(_V4)
    for path in paths[:: max(1, len(paths) // 8)]:
        episode = load_episode(path)
        fails = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        if fails:
            return episode, frozenset(fails[:30])
    pytest.skip("no test episode with a failure")


@_needs_data
def test_atom_tags_follow_the_scene_tag_permutation() -> None:
    vocab = Vocab.from_json(_VOCAB)
    episode, ctx = _dd2d_episode_with_failures()
    example, _ = build_example(
        episode,
        vocab,
        rng=np.random.default_rng(0),
        evidence=True,
        context_f=ctx,
        augment_tags=True,
        emit_init_atoms=True,
        emit_goal_atoms=True,
    )
    scene_tags = {int(t) for t in example.obj_tags}
    assert example.init_atom_arg_tags is not None
    assert example.goal_atom_arg_tags is not None
    atom_tags = {
        int(t)
        for arr in (example.init_atom_arg_tags, example.goal_atom_arg_tags)
        for t in arr.reshape(-1)
        if t != 0
    }
    assert atom_tags  # the sample really carries atom arguments
    assert atom_tags <= scene_tags


# --------------------------------------------------------------------------- #
# T6 — OOV predicate maps to the guarded id without error
# --------------------------------------------------------------------------- #
def _mini_vocab() -> Vocab:
    return Vocab(
        config_hash="test",
        operators={"op0": 1, "op1": 2},
        predicates={
            "<OOV>": {"arity": 0, "idx": 0},
            "p0": {"arity": 1, "idx": 1},
            "on": {"arity": 2, "idx": 2},
        },
        types={"t": 1},
        max_operator_arity=2,
        max_predicate_arity=2,
        max_skeleton_length=4,
        max_atoms_per_state=4,
        max_objects_per_state=4,
        max_pool_size=4,
    )


def test_oov_predicate_maps_to_guarded_id() -> None:
    vocab = _mini_vocab()
    atom = SimpleNamespace(
        predicate=SimpleNamespace(name="never_seen"),
        entities=[SimpleNamespace(name="o1")],
    )
    pred, argt = _atom_profile_arrays([atom], {"o1": 1}, vocab, 2)
    assert pred.tolist() == [1]  # vocab OOV idx 0, +1 shift -> 1 (distinct from pad 0)
    assert argt.tolist() == [[1, 0]]


def test_non_object_arg_is_dropped_not_crashing() -> None:
    """An arg not in the tag table (e.g. a sentinel) is skipped, never a KeyError."""
    vocab = _mini_vocab()
    atom = SimpleNamespace(
        predicate=SimpleNamespace(name="on"),
        entities=[SimpleNamespace(name="o1"), SimpleNamespace(name="__wall__")],
    )
    pred, argt = _atom_profile_arrays([atom], {"o1": 1}, vocab, 2)
    assert pred.tolist() == [3]  # on idx 2, +1
    assert argt.tolist() == [[1, 0]]  # only o1 resolves; the sentinel is dropped


# --------------------------------------------------------------------------- #
# T7 — object-order invariance with atoms on
# --------------------------------------------------------------------------- #
def _perm_objects(batch, perm: torch.Tensor):
    b = dataclasses.replace(batch)
    for name in (
        "obj_tags",
        "obj_boundary",
        "obj_pose",
        "obj_rel",
        "obj_is_goal",
        "obj_mask",
    ):
        t = getattr(b, name)
        if t is not None:
            setattr(b, name, t[:, perm])
    return b


def test_object_order_invariance_with_atoms() -> None:
    """Permuting object rows (and the tags atoms bind to) leaves the logits unchanged.

    The atom profiles bind objects by *tag*, not by row, so a relabel-consistent object
    permutation must not change a candidate logit.
    """
    on = _model(atom_mode="profiles", seed=0)
    assert on.atoms is not None
    with torch.no_grad():  # perturb so the (zero-init) atom path is actually exercised
        on.atoms.obj_proj.weight.normal_()
        on.atoms.glob_proj_atom.weight.normal_()
    # A single 3-object example whose object rows carry tags 1,2,3 in order, so permuting
    # rows also permutes tags consistently (row i holds tag i+1).
    batch = _collate_base(
        [_atom_example(n_obj=3, with_atoms=True, seed=7)], max_arity=2
    )
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        base = on(batch)[0]
        permuted = on(_perm_objects(batch, perm))[0]
    assert torch.allclose(base, permuted, atol=1e-4)


# --------------------------------------------------------------------------- #
# atom_emission helper
# --------------------------------------------------------------------------- #
def test_atom_emission_resolves_switches() -> None:
    assert atom_emission(SimpleNamespace(atom_mode="off")) == (False, False)
    assert atom_emission(
        SimpleNamespace(atom_mode="profiles", use_init_atoms=True, use_goal_atoms=True)
    ) == (True, True)
    assert atom_emission(
        SimpleNamespace(atom_mode="profiles", use_init_atoms=False, use_goal_atoms=True)
    ) == (False, True)


# --------------------------------------------------------------------------- #
# collate shape sanity
# --------------------------------------------------------------------------- #
def test_collate_atom_shapes() -> None:
    batch = _collate_base(
        [
            _atom_example(with_atoms=True, seed=1),
            _atom_example(with_atoms=True, seed=2),
        ],
        max_arity=2,
    )
    assert batch.init_atom_pred is not None and batch.goal_atom_pred is not None
    assert batch.init_atom_arg_tags is not None and batch.goal_atom_arg_tags is not None
    assert batch.init_atom_pred.shape == (2, 3)  # 3 init atoms
    assert batch.init_atom_arg_tags.shape == (2, 3, _ARITY)
    assert batch.goal_atom_pred.shape == (2, 1)  # 1 goal atom
    assert batch.goal_atom_arg_tags.shape == (2, 1, _ARITY)
    assert batch.init_atom_pred.dtype == torch.int64


# --------------------------------------------------------------------------- #
# checkpoint round-trip: TrainConfig -> asdict -> load_checkpoint -> strict load
# --------------------------------------------------------------------------- #
def _roundtrip(tmp_path, train_cfg) -> None:
    from alphatamp.approaches.spectre.inference import load_checkpoint
    from alphatamp.approaches.spectre.model import N_OVERLAP_COV

    vocab = _mini_vocab()
    on = train_cfg.atom_mode == "profiles"
    # Build the saved model with the exact config load_checkpoint reconstructs from the
    # persisted TrainConfig dict, so a strict load must match (mirrors load_checkpoint).
    n_ov = (
        (N_OVERLAP_COV if train_cfg.coverage_feats else 2)
        if train_cfg.use_overlap
        else 0
    )
    sc = SpectreConfig(
        n_overlap_feats=n_ov,
        d_rel=train_cfg.d_rel,
        use_records=train_cfg.use_records,
        evidence_attn=train_cfg.evidence_attn,
        coverage_feats=train_cfg.coverage_feats,
        use_state_delta=train_cfg.use_state_delta,
        atom_mode=train_cfg.atom_mode,
        use_init_atoms=train_cfg.use_init_atoms,
        use_goal_atoms=train_cfg.use_goal_atoms,
        n_predicates=len(vocab.predicates),
        max_pred_arity=vocab.max_predicate_arity,
    )
    model = SpectreModel(n_ops=len(vocab.operators), max_arity=2, cfg=sc)
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": dataclasses.asdict(train_cfg),
            "n_ops": len(vocab.operators),
            "selected": "test",
        },
        ckpt,
    )
    loaded, _deploy = load_checkpoint(ckpt, vocab, "cpu")  # strict=True inside
    assert (loaded.atoms is not None) is on
    assert loaded.cfg.atom_mode == train_cfg.atom_mode


def test_checkpoint_roundtrip_atoms(tmp_path) -> None:
    from alphatamp.approaches.spectre.train import TrainConfig

    _roundtrip(tmp_path, TrainConfig(atom_mode="profiles"))


def test_checkpoint_roundtrip_config_off(tmp_path) -> None:
    from alphatamp.approaches.spectre.train import TrainConfig

    _roundtrip(tmp_path, TrainConfig())
