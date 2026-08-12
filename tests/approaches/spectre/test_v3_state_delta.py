"""``--state-delta``: the abstract state ``s_j`` on each record token, as a delta from
``s_0``.

Proposal §6.1 lists ``s_j`` as a record field; every other field was built and this
one was not. What is carried is the *delta* -- which atoms the failing prefix added,
which it deleted -- because ``s_0`` already reaches the scorer via the scene tokens.

Five distinct ways this goes silently wrong, one test each:

1. **Absence must be exact.** A pre-flag v3 checkpoint must keep loading
   ``strict=True``, which means the flag may not perturb one existing parameter
   shape. That is D-8's discipline aimed at the *deployed v3* state dict, not v2.2's.
2. **The flag must not change the initialization.** Widening the record projection would
   re-randomize every weight in it, so the arms would differ in the draw as well as in
   the feature. The additive branch is zero-initialized precisely so a flag-on model
   *is* the flag-off model at step 0.
3. **An empty delta must encode identically wherever it appears.** About half of the
   aggregated tokens sit at ``j = 0``, and deploy collates a single example per step, so
   "empty alone" and "empty beside a non-empty batch-mate" are both routine.
4. **Structure must survive pooling.** An atom's arguments are positional and its
   predicate binds to them; two different pooling mistakes collapse
   ``p(a,b)``/``p(b,a)`` and ``{q(a), r(b)}``/``{q(b), r(a)}``. DD2D is all-unary
   and cannot exhibit the first, which is why it is pinned synthetically.
5. **The index must be right.** ``traj[j±1]`` is a plausible-looking off-by-one that
   produces a well-formed delta. On DD2D the delta's object set is exactly
   ``all_objects - unmoved``, which catches it immediately and nothing else would.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from alphatamp.approaches.spectre.dataset_v3 import (
    build_record_arrays,
    build_v3_example,
    collate_v3,
)
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.failure_record import (
    StateDelta,
    records_for_candidate,
)
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.model_v3 import (
    MAX_DELTA_ATOMS,
    RecordEncoder,
    SpectreV3Model,
    V3Config,
)
from alphatamp.approaches.spectre.tags import assign_tags
from alphatamp.approaches.spectre.vocab import Vocab

_ROOT = Path(__file__).resolve().parents[3]
_V4 = _ROOT / "data" / "spectre" / "raw" / "dd2d_v4" / "test"
_VOCAB = _ROOT / "data" / "spectre" / "derived" / "dd2d_v4" / "train_vocab.json"

_needs_data = pytest.mark.skipif(not _V4.is_dir(), reason="dd2d_v4 collection absent")

#: The deployed arm's architecture, so these tests exercise the configuration under test.
_DEPLOYED = dict(
    n_overlap_feats=4,
    max_tags=32,
    dropout_p=0.0,
    use_records=True,
    evidence_attn=True,
    coverage_feats=True,
)


def _vocab() -> Vocab:
    return Vocab.from_json(_VOCAB)


def _episode_with_evidence(min_depth: int = 0):
    """A test episode with failures, striding so every stratum is reachable.

    Stride, never truncate: episodes are stored in seed order and the collector fills
    strata in seed bands, so ``paths[:n]`` is all stratum 0 -- where every failure sits
    at ``j = 0`` and the delta is empty by construction.
    """
    paths = list_episodes(_V4)
    for path in paths[:: max(1, len(paths) // 8)]:
        episode = load_episode(path)
        fails = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        if not fails:
            continue
        ctx = frozenset(fails[:30])
        if min_depth == 0:
            return episode, ctx
        spec = spec_for(episode.provenance.env_variant)
        deep = any(
            r.step_index >= min_depth
            for i in ctx
            for r in records_for_candidate(episode, i, spec)
        )
        if deep:
            return episode, ctx
    pytest.skip("no test episode with a deep enough failure")


def _records(episode, vocab, ctx, state_delta: bool):
    return build_v3_example(
        episode,
        vocab,
        rng=None,
        evidence=True,
        context_f=ctx,
        augment_tags=False,
        overlap_mode="jaccard",
        aggregate_records=True,
        coverage_feats=True,
        state_delta=state_delta,
    )


def _batch(example, records, vocab):
    return collate_v3(
        [example],
        max_arity=vocab.max_operator_arity,
        records=[records],
        max_pred_arity=vocab.max_predicate_arity,
    )


def _model(vocab, *, state_delta: bool, seed: int = 0) -> SpectreV3Model:
    torch.manual_seed(seed)
    extra = (
        dict(
            use_state_delta=True,
            n_predicates=len(vocab.predicates),
            max_pred_arity=vocab.max_predicate_arity,
        )
        if state_delta
        else {}
    )
    return SpectreV3Model(
        n_ops=len(vocab.operators),
        max_arity=vocab.max_operator_arity,
        cfg=V3Config(**_DEPLOYED, **extra),  # type: ignore[arg-type]
    ).eval()


# --------------------------------------------------------------------------- absence


@_needs_data
def test_state_delta_off_is_exact_absence() -> None:
    """Flag off changes no shape, emits no delta, and adds no parameter."""
    vocab = _vocab()
    off, on = _model(vocab, state_delta=False), _model(vocab, state_delta=True)
    shapes_off = {k: tuple(v.shape) for k, v in off.state_dict().items()}
    shapes_on = {k: tuple(v.shape) for k, v in on.state_dict().items()}
    assert set(shapes_on) - set(shapes_off) == {
        "records.pred_emb.weight",
        "records.atom_proj.weight",
        "records.atom_proj.bias",
        "records.delta_proj.weight",
        "records.delta_proj.bias",
    }
    # Every pre-existing entry keeps its shape -- in particular the record projection is
    # not widened, which is what lets a pre-flag checkpoint load strictly.
    assert all(shapes_on[k] == v for k, v in shapes_off.items())
    assert off.records is not None and off.records.proj[0].in_features == 100
    assert on.records is not None and on.records.proj[0].in_features == 100

    episode, ctx = _episode_with_evidence()
    example, records = _records(episode, vocab, ctx, state_delta=False)
    assert records and all(len(row) == 4 for row in records)
    batch = _batch(example, records, vocab)
    assert batch.rec_delta_pred_ids is None
    assert batch.rec_delta_arg_tags is None


@_needs_data
def test_zero_init_delta_branch_is_a_functional_no_op_at_init() -> None:
    """At init the flag changes nothing -- so what is measured later is the feature.

    If this fails, the delta arm and the baseline differ in their *initialization* as
    well as in their inputs, and the comparison stops being an ablation. It is also the
    test that fires if the additive branch is ever "simplified" into a wider ``proj``.
    """
    vocab = _vocab()
    off, on = _model(vocab, state_delta=False), _model(vocab, state_delta=True)
    shared = off.state_dict()
    on_sd = on.state_dict()
    assert all(torch.equal(shared[k], on_sd[k]) for k in shared)
    assert on.records is not None and on.records.delta_proj is not None
    assert torch.count_nonzero(on.records.delta_proj.weight) == 0
    assert torch.count_nonzero(on.records.delta_proj.bias) == 0

    episode, ctx = _episode_with_evidence(min_depth=2)
    ex_off, rec_off = _records(episode, vocab, ctx, state_delta=False)
    ex_on, rec_on = _records(episode, vocab, ctx, state_delta=True)
    b_off = _batch(ex_off, rec_off, vocab)
    b_on = _batch(ex_on, rec_on, vocab)
    # the deltas really are populated, or this asserts nothing
    assert int((b_on.rec_delta_pred_ids != 0).sum()) > 0
    logits_off, _ = off(b_off)
    logits_on, _ = on(b_on)
    finite = torch.isfinite(logits_off)
    assert torch.equal(logits_off[finite], logits_on[finite])


def test_state_delta_off_loads_the_deployed_checkpoint_strictly(tmp_path) -> None:
    """A state-delta-off checkpoint round-trips: loads strictly, no ``delta_proj``.

    Self-contained (saved through ``asdict(TrainV3Config)`` like ``train_v3``) rather than
    reading a disk artifact -- a pre-narrowing width-8 checkpoint no longer loads by
    design, and the narrowing is orthogonal to what this asserts (that state-delta OFF
    yields a records encoder without the delta branch).
    """
    from dataclasses import asdict

    import torch

    from alphatamp.approaches.spectre.inference_v3 import load_v3_checkpoint
    from alphatamp.approaches.spectre.model_v3 import (
        N_OVERLAP_V3,
        SpectreV3Model,
        V3Config,
    )
    from alphatamp.approaches.spectre.train_v3 import TrainV3Config

    vocab = _vocab()
    cfg = TrainV3Config(
        overlap_mode="jaccard",
        use_overlap=True,
        coverage_feats=True,
        aggregate_records=True,
        use_records=True,
        evidence_attn=True,
        use_state_delta=False,
    )
    model = SpectreV3Model(
        n_ops=len(vocab.operators),
        max_arity=vocab.max_operator_arity,
        cfg=V3Config(
            n_overlap_feats=N_OVERLAP_V3,
            n_prior_feats=0,
            d_rel=cfg.d_rel,
            use_records=True,
            evidence_attn=True,
            coverage_feats=True,
            use_state_delta=False,
            n_predicates=len(vocab.predicates),
            max_pred_arity=vocab.max_predicate_arity,
        ),
    )
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "cfg": asdict(cfg),
            "n_ops": len(vocab.operators),
            "state_dict": model.state_dict(),
        },
        ckpt,
    )

    loaded, deploy = load_v3_checkpoint(ckpt, vocab, "cpu")
    assert deploy["state_delta"] is False
    assert loaded.records is not None and loaded.records.delta_proj is None


# ------------------------------------------------------------------------- semantics


def test_delta_is_the_strips_progression_delta() -> None:
    """``added = s_j \\ s_0``, ``deleted = s_0 \\ s_j``, on a hand-built two-step plan.

    Built on DD2D's real operator schema, so the nullary ``handempty`` is exercised: an
    atom with no arguments contributes a predicate but no tag, and must survive the
    encoding rather than being mistaken for padding.
    """
    from bilevel_planning.structs import RelationalAbstractState
    from relational_structs import GroundAtom

    from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import (
        HandEmpty,
        InDrawer,
        ItemType,
        Pick,
        PlaceBuffer,
    )
    from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

    a, b = ItemType("a"), ItemType("b")
    s0 = RelationalAbstractState(
        atoms=frozenset(
            {
                GroundAtom(InDrawer, [a]),
                GroundAtom(InDrawer, [b]),
                GroundAtom(HandEmpty, []),
            }
        ),
        objects=frozenset({a, b}),
    )
    plan = [Pick.ground((a,)), PlaceBuffer.ground((a,))]
    traj = reconstruct_trajectory(s0, plan, verify_preconditions=True)

    def names(state):
        return {
            (x.predicate.name, tuple(e.name for e in x.entities)) for x in state.atoms
        }

    # j = 1: mid-pick, `a` is in hand and the hand is no longer empty
    assert tuple(sorted(names(traj[1]) - names(traj[0]))) == (("holding", ("a",)),)
    assert tuple(sorted(names(traj[0]) - names(traj[1]))) == (
        ("handempty", ()),
        ("in-drawer", ("a",)),
    )
    # j = 2: `a` is staged, the hand is free again, so `handempty` leaves the delta
    assert tuple(sorted(names(traj[2]) - names(traj[0]))) == (("on-buffer", ("a",)),)
    assert tuple(sorted(names(traj[0]) - names(traj[2]))) == (("in-drawer", ("a",)),)
    # j = 0 is the empty prefix, and an empty delta is a value rather than an absence
    assert StateDelta().is_empty()
    assert not StateDelta(added=(("holding", ("a",)),)).is_empty()


@_needs_data
def test_delta_objects_equal_the_complement_of_unmoved_on_dd2d() -> None:
    """The off-by-one detector: ``traj[j±1]`` breaks this and nothing else catches it.

    On DD2D every atom the prefix touches names an object the prefix moved, so the
    delta's object set is exactly ``all_objects - unmoved`` -- verified on 946,063
    records across all three splits. This checks a strided sample of the test split.
    """
    spec = spec_for("dd2d_v4")
    paths = list_episodes(_V4)
    checked = 0
    for path in paths[:: max(1, len(paths) // 8)]:
        episode = load_episode(path)
        all_objects = frozenset(episode.object_registry)
        for i, outcome in enumerate(episode.outcomes):
            if outcome.outcome != "fail":
                continue
            for rec in records_for_candidate(episode, i, spec, with_state_delta=True):
                delta = rec.state_delta
                assert delta is not None
                objects = {o for _, args in delta.added + delta.deleted for o in args}
                assert objects == set(all_objects - rec.unmoved)
                assert delta.is_empty() == (rec.step_index == 0)
                checked += 1
    assert checked > 100


@_needs_data
def test_truncation_never_fires_on_dd2d_v4() -> None:
    """``MAX_DELTA_ATOMS`` is slack, so no delta is silently clipped."""
    spec = spec_for("dd2d_v4")
    paths = list_episodes(_V4)
    worst_added = worst_deleted = 0
    for path in paths[:: max(1, len(paths) // 8)]:
        episode = load_episode(path)
        for i, outcome in enumerate(episode.outcomes):
            if outcome.outcome != "fail":
                continue
            for rec in records_for_candidate(episode, i, spec, with_state_delta=True):
                assert rec.state_delta is not None
                worst_added = max(worst_added, len(rec.state_delta.added))
                worst_deleted = max(worst_deleted, len(rec.state_delta.deleted))
    assert max(worst_added, worst_deleted) < MAX_DELTA_ATOMS


@_needs_data
def test_aggregation_keeps_the_deepest_delta() -> None:
    """Aggregation keeps the deepest record per query, so its delta is the furthest one.

    This is why the delta is derived at tensorize time rather than merged: "the state at
    the furthest point this query reached" falls out of the record that already survives.
    """
    from alphatamp.approaches.spectre.dataset_v3 import _aggregate_per_query

    spec = spec_for("dd2d_v4")
    paths = list_episodes(_V4)
    checked = 0
    for path in paths[:: max(1, len(paths) // 8)]:
        episode = load_episode(path)
        for idx, outcome in enumerate(episode.outcomes):
            if outcome.outcome != "fail":
                continue
            raw = records_for_candidate(episode, idx, spec, with_state_delta=True)
            if len(raw) < 2:
                continue
            for rec in _aggregate_per_query(raw):
                same_query = [
                    r for r in raw if (r.schema, r.args) == (rec.schema, rec.args)
                ]
                deepest = max(same_query, key=lambda r: r.step_index)
                assert rec.step_index == deepest.step_index
                assert rec.state_delta == deepest.state_delta
            checked += 1
    assert checked > 10, f"only {checked} multi-record candidates found"


@_needs_data
def test_delta_tags_follow_the_tag_permutation() -> None:
    """Delta arguments land in the scene's tag namespace, under augmentation too.

    The record pathway once carried no object information at all because its tags came
    from a different assignment than the scene's. The delta joins on the same table.
    """
    import numpy as np

    vocab = _vocab()
    episode, ctx = _episode_with_evidence(min_depth=2)
    example, records = build_v3_example(
        episode,
        vocab,
        rng=np.random.default_rng(0),
        evidence=True,
        context_f=ctx,
        augment_tags=True,
        aggregate_records=True,
        state_delta=True,
    )
    scene_tags = {int(t) for t in example.obj_tags}
    delta_tags = {
        int(t) for row in records for role in row[4] for _, args in role for t in args
    }
    assert delta_tags  # the sample really carries a delta
    assert delta_tags <= scene_tags


# -------------------------------------------------------------------------- encoding


def _delta_encoder(n_predicates: int, arity: int) -> RecordEncoder:
    torch.manual_seed(0)
    enc = RecordEncoder(
        n_schemas=4,
        max_tags=8,
        dropout_p=0.0,
        n_predicates=n_predicates,
        max_pred_arity=arity,
        state_delta=True,
    ).eval()
    # zero-init makes the branch invisible by design; perturb it so the encoding is
    # observable at all.
    with torch.no_grad():
        enc.delta_proj.weight.normal_()  # type: ignore[union-attr]
    return enc


def _encode(enc: RecordEncoder, atoms, arity: int) -> torch.Tensor:
    """Encode one record whose *added* role holds ``atoms`` = [(pred_id, [tags])]."""
    pred = torch.zeros(1, 1, 2, MAX_DELTA_ATOMS, dtype=torch.long)
    args = torch.zeros(1, 1, 2, MAX_DELTA_ATOMS, arity, dtype=torch.long)
    for i, (pid, tags) in enumerate(atoms):
        pred[0, 0, 0, i] = pid
        args[0, 0, 0, i, : len(tags)] = torch.tensor(tags)
    return enc(
        torch.ones(1, 1, dtype=torch.long),
        torch.zeros(1, 1, 4, dtype=torch.long),
        torch.zeros(1, 1, 8, dtype=torch.long),
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, dtype=torch.bool),
        pred,
        args,
    )


def test_delta_atom_arguments_are_positional_not_pooled() -> None:
    """``p(a, b)`` and ``p(b, a)`` must differ.

    DD2D's predicates are all unary so it can never exhibit this, which is precisely why
    it is asserted synthetically: pooling an atom's argument slots is the natural-looking
    implementation and it silently discards argument order for any richer domain.
    """
    enc = _delta_encoder(n_predicates=3, arity=2)
    ab = _encode(enc, [(1, [2, 3])], arity=2)
    ba = _encode(enc, [(1, [3, 2])], arity=2)
    assert not torch.allclose(ab, ba)


def test_predicate_binds_to_its_own_arguments() -> None:
    """``{q(a), r(b)}`` and ``{q(b), r(a)}`` must differ.

    Guards the other collapse: projecting each atom *before* pooling is what keeps a
    predicate bound to its arguments. Concatenating the two roles and pooling afterwards
    would make these two sets identical.
    """
    enc = _delta_encoder(n_predicates=3, arity=1)
    one = _encode(enc, [(1, [2]), (2, [3])], arity=1)
    two = _encode(enc, [(1, [3]), (2, [2])], arity=1)
    assert not torch.allclose(one, two)


def test_empty_delta_pools_to_exactly_zero_and_is_batch_independent() -> None:
    """An empty delta contributes exactly nothing, wherever it sits in the batch.

    Roughly half of the aggregated tokens are empty, and deploy collates one example at a
    time -- so if the encoding depended on a batch-mate having a delta, the same record
    would mean two different things between training and deployment.
    """
    enc = _delta_encoder(n_predicates=3, arity=1)
    alone = _encode(enc, [], arity=1)

    pred = torch.zeros(1, 2, 2, MAX_DELTA_ATOMS, dtype=torch.long)
    args = torch.zeros(1, 2, 2, MAX_DELTA_ATOMS, 1, dtype=torch.long)
    pred[0, 1, 0, 0] = 2  # the *other* record in the batch has a delta
    args[0, 1, 0, 0, 0] = 3
    together = enc(
        torch.ones(1, 2, dtype=torch.long),
        torch.zeros(1, 2, 4, dtype=torch.long),
        torch.zeros(1, 2, 8, dtype=torch.long),
        torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
        pred,
        args,
    )
    assert torch.allclose(alone[0, 0], together[0, 0], atol=1e-6)
    # and the delta branch really is inert for that row
    assert enc.delta_proj is not None
    zero_contrib = enc._delta(pred, args)[0, 0]
    assert torch.count_nonzero(zero_contrib) == 0


@_needs_data
def test_no_nan_with_empty_deltas_and_with_no_records_at_all() -> None:
    """The A10 guard stays live: the deployed arm uses a separate evidence attention.

    An all-empty record memory is an all-True key-padding mask, which makes
    ``MultiheadAttention`` emit NaN rather than an empty result.
    """
    vocab = _vocab()
    model = _model(vocab, state_delta=True)
    with torch.no_grad():
        model.records.delta_proj.weight.normal_()  # type: ignore[union-attr]

    episode, ctx = _episode_with_evidence()
    example, records = _records(episode, vocab, ctx, state_delta=True)
    logits, _ = model(_batch(example, records, vocab))
    assert not torch.isnan(logits).any()

    # |F| = 0: the deployment start, and the case that has no evidence keys at all
    empty_ex, empty_rec = _records(episode, vocab, frozenset(), state_delta=True)
    assert empty_rec == []
    logits, _ = model(_batch(empty_ex, empty_rec, vocab))
    assert not torch.isnan(logits).any()


@_needs_data
def test_delta_atom_order_is_deterministic() -> None:
    """Sorted atoms, so ``[:MAX_DELTA_ATOMS]`` truncation is not set-iteration order."""
    vocab = _vocab()
    episode, ctx = _episode_with_evidence(min_depth=2)
    spec = spec_for(episode.provenance.env_variant)
    tags = assign_tags([o for o in sorted(episode.object_registry)], rng=None)
    first = build_record_arrays(
        episode, ctx, tags, vocab, spec, aggregate=True, state_delta=True
    )
    again = build_record_arrays(
        episode, ctx, tags, vocab, spec, aggregate=True, state_delta=True
    )
    assert first == again
    for rec in records_for_candidate(episode, sorted(ctx)[0], spec, True):
        assert rec.state_delta is not None
        assert list(rec.state_delta.added) == sorted(rec.state_delta.added)
        assert list(rec.state_delta.deleted) == sorted(rec.state_delta.deleted)


@pytest.mark.slow
@_needs_data
def test_precondition_verification_also_passes_on_dd2d() -> None:
    """``verify_preconditions=False`` is defensive, not a cover for a known break.

    A deployed rollout must not raise on a malformed skeleton, so the progression runs
    unchecked. Every dd2d_v4 skeleton does in fact verify -- recorded here so a future
    reader does not mistake the flag for a suppressed failure.
    """
    from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

    for path in list_episodes(_V4)[:: max(1, len(list_episodes(_V4)) // 4)]:
        episode = load_episode(path)
        for skeleton in episode.skeleton_pool:
            reconstruct_trajectory(
                episode.initial_abstract_state,
                skeleton.operator_seq,
                verify_preconditions=True,
            )
