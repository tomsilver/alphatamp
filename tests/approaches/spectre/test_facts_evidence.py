"""Tests for the Step-11 typed-evidence pathway: fact gathering, the F-context
tensorizer, and the live scramble gauge.

Synthetic episodes only (no data dependency).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alphatamp.approaches.spectre import dataset_v2 as D2
from alphatamp.approaches.spectre import evidence as EV
from alphatamp.approaches.spectre import model_v2 as M
from alphatamp.approaches.spectre.facts import (
    FACT_TYPE_IDS,
    gather_context_facts,
)
from alphatamp.approaches.spectre.schema import (
    Fact,
    ObjectGeometry,
    PostMortemRecord,
    SceneGeometry,
)


class _V:
    operators = {"Pick": 1, "Place": 2}
    max_operator_arity = 2


def _toy_geo_episode_with_pm():
    from _fixtures import build_toy_episode  # type: ignore

    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    ring = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
    names = sorted(ep.object_registry)
    objs = tuple(
        ObjectGeometry(
            name=nm,
            pose=(float(i), 0.0, 0.0),
            boundary=ring,
            family="test",
            area=1.0,
            concave=False,
            is_target=(nm == names[0]),
        )
        for i, nm in enumerate(names)
    )
    # attach a typed fact to fail #0 (extraction-failed on a real object).
    pm = PostMortemRecord(
        skeleton_idx=0,
        refinement_seed=0,
        facts=(Fact("extraction-failed", (names[1],), "hint"),),
    )
    outs = list(ep.outcomes)
    outs[0] = dataclasses.replace(outs[0], post_mortem=pm)
    return dataclasses.replace(
        ep,
        outcomes=tuple(outs),
        scene_geometry=SceneGeometry(
            objects=objs, containers=(), frame={"drawer_w": 10.0}
        ),
    )


def test_gather_context_facts_only_from_F():
    ep = _toy_geo_episode_with_pm()
    assert gather_context_facts(ep, [1, 2]) == []  # no post_mortem on 1/2
    got = gather_context_facts(ep, [0])
    assert len(got) == 1
    assert got[0].type_id == FACT_TYPE_IDS["extraction-failed"]
    assert got[0].source_idx == 0


def test_evidence_example_marks_avail_and_facts():
    ep = _toy_geo_episode_with_pm()
    ex = D2.build_v2_example(
        ep, _V(), rng=None, max_tags=16, evidence=True, context_f=frozenset({0})
    )
    assert ex.avail[0] is False and ex.avail[2] is True  # F candidate unavailable
    assert len(ex.fact_type_ids) == 1  # the extraction-failed fact
    # evidence dropout hides facts but keeps the context removed.
    ex0 = D2.build_v2_example(
        ep,
        _V(),
        rng=None,
        max_tags=16,
        evidence=True,
        context_f=frozenset({0}),
        hide_facts=True,
    )
    assert ex0.avail[0] is False and ex0.fact_type_ids == []


def test_scramble_changes_args_not_structure():
    ep = _toy_geo_episode_with_pm()
    ex = D2.build_v2_example(
        ep, _V(), rng=None, max_tags=16, evidence=True, context_f=frozenset({0})
    )
    batch = D2.collate_v2([ex], max_arity=2)
    scr = EV.scramble_fact_identities(batch, np.random.default_rng(0))
    # type/tier/mask preserved; only arg tags may change.
    assert (scr.fact_type_ids == batch.fact_type_ids).all()
    assert (scr.fact_tier_ids == batch.fact_tier_ids).all()
    assert (scr.fact_mask == batch.fact_mask).all()


def test_scramble_gauge_nonneg_and_zero_without_facts():
    ep = _toy_geo_episode_with_pm()
    model = M.SpectreV2Model(n_ops=2, max_arity=2, max_tags=16)
    ex = D2.build_v2_example(
        ep, _V(), rng=None, max_tags=16, evidence=True, context_f=frozenset({0})
    )
    batch = D2.collate_v2([ex], max_arity=2)
    g = EV.scramble_gauge(model, batch, "cpu", np.random.default_rng(0))
    assert g >= 0.0
    # a static (no-fact) batch has a zero gauge by construction.
    ex_static = D2.build_v2_example(ep, _V(), rng=None, max_tags=16)
    static_batch = D2.collate_v2([ex_static], max_arity=2)
    assert (
        EV.scramble_gauge(model, static_batch, "cpu", np.random.default_rng(0)) == 0.0
    )


def test_evidence_rollout_returns_fp():
    ep = _toy_geo_episode_with_pm()
    model = M.SpectreV2Model(n_ops=2, max_arity=2, max_tags=16)
    fp_on = EV.evidence_rollout(model, ep, _V(), "cpu", use_facts=True, max_tags=16)
    fp_off = EV.evidence_rollout(model, ep, _V(), "cpu", use_facts=False, max_tags=16)
    assert isinstance(fp_on, int) and 0 <= fp_on < len(ep.skeleton_pool)
    assert isinstance(fp_off, int)


def test_deployed_rollout_traced_order():
    """``deployed_rollout_traced`` yields the same count plus the realized order."""
    ep = _toy_geo_episode_with_pm()
    model = M.SpectreV2Model(n_ops=2, max_arity=2, max_tags=16)
    att = EV.deployed_rollout(model, ep, _V(), "cpu", max_tags=16)
    assert isinstance(att, int) and 1 <= att <= len(ep.skeleton_pool)

    att2, trace = EV.deployed_rollout_traced(model, ep, _V(), "cpu", max_tags=16)
    assert att2 == att  # deterministic model -> identical path
    order = trace.order
    assert isinstance(order, list) and len(order) == att
    assert len(set(order)) == len(order)  # each pool index tried at most once
    assert all(0 <= i < len(ep.skeleton_pool) for i in order)
    # the rollout ends at the first success, so the last attempt is a success.
    succ = {i for i, o in enumerate(ep.outcomes) if o.outcome == "success"}
    assert order[-1] in succ


def test_deployed_rollout_traced_step_scores_and_dead():
    """The trace is step-aligned: one raw score row + one dead set per attempt."""
    ep = _toy_geo_episode_with_pm()
    k = len(ep.skeleton_pool)
    model = M.SpectreV2Model(n_ops=2, max_arity=2, max_tags=16)
    att, trace = EV.deployed_rollout_traced(model, ep, _V(), "cpu", max_tags=16)

    assert len(trace.step_scores) == att
    assert len(trace.step_dead) == att
    assert all(len(row) == k for row in trace.step_scores)
    # Rows are raw: neither this loop's -1e9 tried-mask nor the -1e6 demotion offset
    # is baked in. The *model* still masks its own failure context, so at step t the
    # non-finite entries are exactly the candidates attempted before step t.
    for t, row in enumerate(trace.step_scores):
        nonfinite = {i for i, x in enumerate(row) if not math.isfinite(x)}
        assert nonfinite == set(trace.order[:t])
        assert all(abs(x) < 1e5 for x in row if math.isfinite(x))
    # Proof-demotion only ever accumulates, so the dead sets are nested.
    for prev, cur in zip(trace.step_dead, trace.step_dead[1:]):
        assert set(prev) <= set(cur)
    # Nothing is demoted before the first failure has been observed.
    assert trace.step_dead[0] == []


def test_deployed_rollout_traced_demotion_is_sound():
    """A demoted candidate is never one that actually succeeds (proof-demotion)."""
    ep = _toy_geo_episode_with_pm()
    model = M.SpectreV2Model(n_ops=2, max_arity=2, max_tags=16)
    _, trace = EV.deployed_rollout_traced(model, ep, _V(), "cpu", max_tags=16)
    succ = {i for i, o in enumerate(ep.outcomes) if o.outcome == "success"}
    assert set(trace.step_dead[-1]).isdisjoint(succ)
