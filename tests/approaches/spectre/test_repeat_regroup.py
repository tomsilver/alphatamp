"""``repeat`` / ``regroup`` overlap features
(docs/adaptivity_probe_plan_restock3d_v3.md).

``repeat`` is the F3 exact-step veto: it fires on a candidate that contains the exact
failed step of a *blameless, exhausted* failure of a ``step_certificate`` schema.
``regroup`` is the F2 seating-chart: it fires on a candidate that re-assembles a
culprit-bearing failure's chart (failed step + each culprit's establishing step).

Self-contained: a toy episode + injected ``refiner_metadata`` + a custom ``DomainSpec``,
so these run without the gitignored restock3d_v3 collection. Both are the learned-feature
analogue of the P2 oracle certificates (0 soundness violations); the two gates tested
here -- ``step_certificate`` and ``blame == empty`` -- are what the P2b diagnostic proved
load-bearing (exact-vetoing a culprit-bearing step killed 263 real successes).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from _fixtures import build_toy_episode, write_toy_split

from alphatamp.approaches.spectre.dataset import build_example
from alphatamp.approaches.spectre.domain import (
    DomainSpec,
    QueryAxioms,
    spec_for,
)
from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.schema import ObjectGeometry, SceneGeometry
from alphatamp.approaches.spectre.vocab import extract_vocab

# cand_overlap = [dead, jaccard, coverage, waste, repeat, regroup]
_REPEAT, _REGROUP = 4, 5

_UNIT_RING = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))


def _geometry_for(ep) -> SceneGeometry:
    """A SceneGeometry covering every registered object (build_example requires one)."""
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


# A domain that declares the toy ``Place`` schema both certificates (v3's ``place_*``).
_PLACE_CERT = DomainSpec(
    axioms={"Place": QueryAxioms(step_certificate=True, grouping_certificate=True)}
)
# Same, but only the step (F3/repeat) certificate -- regroup must stay inert.
_PLACE_STEP_ONLY = DomainSpec(axioms={"Place": QueryAxioms(step_certificate=True)})


def _fail_meta(schema, args, step_index, culprits=()):
    """A v3-shaped failure dict: class-1 (culprits) leaves dev None; blameless leaves
    []."""
    return {
        "schema": schema,
        "args": list(args),
        "step_index": step_index,
        "culprits": list(culprits),
        "exhausted": True,
        "budget_exhausted": False,
        "dev_added": None if culprits else [],
        "dev_deleted": None if culprits else [],
    }


def _inject(ep, idx, meta):
    outs = list(ep.outcomes)
    outs[idx] = dataclasses.replace(
        outs[idx], outcome="fail", refiner_metadata={"failures": [meta]}
    )
    return dataclasses.replace(ep, outcomes=tuple(outs))


@pytest.fixture
def toy(tmp_path):
    """A 3-candidate toy episode (skeleton i = Pick/Place of block_i) + its vocab."""
    d = tmp_path / "train"
    write_toy_split(d, [("fail", "fail", "success")])
    vocab = extract_vocab(d, "testhash")
    ep = load_episode(list_episodes(d)[0])
    ep = dataclasses.replace(ep, scene_geometry=_geometry_for(ep))
    return ep, vocab


def _overlap(ep, vocab, ctx, spec, **kw):
    ex, _ = build_example(
        ep,
        vocab,
        rng=None,
        context_f=frozenset(ctx),
        augment_tags=False,
        spec=spec,
        coverage_feats=True,
        repeat_feats=True,
        regroup_feats=True,
        **kw,
    )
    return np.asarray(ex.overlap, dtype=float)


# --------------------------------------------------------------------------- #
# domain declaration
# --------------------------------------------------------------------------- #
def test_v3_spec_declares_place_step_certificate_but_not_proof_tier() -> None:
    """V3 declares ``place_*`` a step certificate, with ``proof_tier`` left False so the
    ``dead`` column / demotion / token-holdout stay byte-unchanged from EMPTY_SPEC."""
    spec = spec_for("restock3d_v3")
    for schema in ("place_tall", "place_short"):
        assert spec.axioms_for(schema).step_certificate is True
        assert spec.axioms_for(schema).grouping_certificate is True
        assert spec.axioms_for(schema).proof_tier() is False
    # reach-over pick is culprit-bearing but neither certificate holds
    assert spec.axioms_for("pick").step_certificate is False
    assert spec.axioms_for("pick").grouping_certificate is False
    # DD2D (`place-buffer`) and SB2D (button-press) declare NO `step_certificate` (graceful
    # degradation, 2026-08-25): the `repeat` transfer probe leaked (44.6% DD2D / 10.9% SB2D)
    # because these are context/order-dependent means-failures, so it was retired. With the
    # declaration gone the deployed `--repeat-feats` recipe leaves `repeat` identically 0 on
    # both envs. proof_tier() stays False regardless, so dead/demotion/token-holdout are
    # byte-unchanged; only restock3d_v3 keeps a genuine F3 step_certificate.
    dd2d = spec_for("dd2d_v4")
    assert dd2d.axioms_for("place-buffer").step_certificate is False
    assert dd2d.axioms_for("place-buffer").proof_tier() is False
    assert dd2d.axioms_for("pick").step_certificate is False
    sb2d = spec_for("stickbutton2d_v1_kinder")
    assert sb2d.axioms_for("StickPressButtonFromNothing").step_certificate is False
    assert sb2d.axioms_for("StickPressButtonFromNothing").proof_tier() is False
    assert sb2d.axioms_for("PlaceStick").step_certificate is False
    # a still-undeclared env -> repeat is inert there (graceful degradation)
    assert spec_for("restock3d_v2").axioms_for("place_tall").step_certificate is False


# --------------------------------------------------------------------------- #
# leakage invariant + width
# --------------------------------------------------------------------------- #
def test_repeat_regroup_are_zero_at_empty_context(toy) -> None:
    """Both columns are exactly 0 at |F|=0, so the first attempt stays static."""
    ep, vocab = toy
    ov = _overlap(ep, vocab, frozenset(), _PLACE_CERT)
    assert not ov[:, _REPEAT].any()
    assert not ov[:, _REGROUP].any()


def test_flags_widen_overlap_by_two_and_off_is_unchanged(toy) -> None:
    """Trailing-additive: the pair appears only when a flag is on."""
    ep, vocab = toy

    def width(**kw):
        ex, _ = build_example(
            ep,
            vocab,
            rng=None,
            context_f=frozenset({0}),
            augment_tags=False,
            spec=_PLACE_CERT,
            coverage_feats=True,
            **kw,
        )
        return np.asarray(ex.overlap).shape[1]

    assert width() == 4  # [dead, jaccard, coverage, waste]
    assert width(repeat_feats=True) == 6
    assert width(regroup_feats=True) == 6
    assert width(repeat_feats=True, regroup_feats=True) == 6


# --------------------------------------------------------------------------- #
# repeat: fires on a blameless step-certificate failure; two gates
# --------------------------------------------------------------------------- #
def test_repeat_fires_on_blameless_step_certificate_failure(toy) -> None:
    ep, vocab = toy
    ep = _inject(ep, 0, _fail_meta("Place", ["robot_0", "block_0"], 1))
    ov = _overlap(ep, vocab, {0}, _PLACE_CERT)
    # only candidate 0 contains Place(robot_0, block_0)
    assert ov[0, _REPEAT] == 1.0
    assert ov[1, _REPEAT] == 0.0 and ov[2, _REPEAT] == 0.0


def test_repeat_requires_a_step_certificate_schema(toy) -> None:
    ep, vocab = toy
    ep = _inject(ep, 0, _fail_meta("Place", ["robot_0", "block_0"], 1))
    ov = _overlap(ep, vocab, {0}, spec_for("dd2d_v4"))  # Place undeclared here
    assert not ov[:, _REPEAT].any()


def test_repeat_excludes_a_culprit_bearing_failure(toy) -> None:
    """The blame == empty gate: exact-vetoing a culprit-bearing step is unsound
    (P2b)."""
    ep, vocab = toy
    ep = _inject(
        ep, 0, _fail_meta("Place", ["robot_0", "block_0"], 1, culprits=("block_0",))
    )
    ov = _overlap(
        ep, vocab, {0}, _PLACE_CERT
    )  # step_certificate holds, but blame != empty
    assert not ov[:, _REPEAT].any()


# --------------------------------------------------------------------------- #
# regroup: fires when a candidate re-assembles the seating chart
# --------------------------------------------------------------------------- #
def test_regroup_fires_on_a_reassembled_chart(toy) -> None:
    """Candidate 0 = Pick(b0)->Place(b0); a failure at Place(b0) blaming b0 has chart
    {Place(b0), establishing(b0)=Pick(b0)} = candidate 0's whole plan. Only candidate 0
    contains both, so regroup fires there and nowhere else."""
    ep, vocab = toy
    ep = _inject(
        ep, 0, _fail_meta("Place", ["robot_0", "block_0"], 1, culprits=("block_0",))
    )
    ov = _overlap(ep, vocab, {0}, _PLACE_CERT)
    assert ov[0, _REGROUP] == 1.0
    assert ov[1, _REGROUP] == 0.0 and ov[2, _REGROUP] == 0.0


def test_regroup_requires_a_grouping_certificate_schema(toy) -> None:
    """Without the domain declaring the schema a grouping certificate, regroup is inert
    -- graceful degradation, since a culprit-bearing failure elsewhere (DD2D's blocker
    staging) is wrong-polarity for the re-assembly signal (§5.2)."""
    ep, vocab = toy
    ep = _inject(
        ep, 0, _fail_meta("Place", ["robot_0", "block_0"], 1, culprits=("block_0",))
    )
    ov = _overlap(ep, vocab, {0}, _PLACE_STEP_ONLY)  # step cert, but NOT grouping cert
    assert not ov[:, _REGROUP].any()


def test_regroup_is_zero_for_a_blameless_failure(toy) -> None:
    """Regroup keys off culprits; a blameless (F3) failure produces no chart."""
    ep, vocab = toy
    ep = _inject(ep, 0, _fail_meta("Place", ["robot_0", "block_0"], 1))
    ov = _overlap(ep, vocab, {0}, _PLACE_CERT)
    assert not ov[:, _REGROUP].any()
