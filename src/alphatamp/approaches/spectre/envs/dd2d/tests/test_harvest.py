"""Tests for the DD2D post-mortem harvest (proposal §6.2/§6.4).

The mandated check (§6.2) is the **harvest-state replay**: replaying the stored bound
prefix into a fresh world must reproduce the recorded state hash. We also check fact
tiers (DD2D's registry makes deducible facts proofs; an empty registry makes them hints)
and the constructive/blocked facts on a real blocked-target instance.
"""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.envs.dd2d.drawer import harvest as H
from alphatamp.approaches.spectre.envs.dd2d.drawer.problem import generate_dd2d_problem
from alphatamp.approaches.spectre.envs.dd2d.drawer.refine import DD2DRefiner
from alphatamp.approaches.spectre.envs.dd2d.soundness import (
    DD2D_REGISTRY,
    EMPTY_REGISTRY,
    SoundnessRegistry,
)


def _blocked_instance():
    prob = generate_dd2d_problem(
        lam=0.5,
        seed=42,
        margin=1.0,
        split="train",
        n_items=11,
        crowd=5,
        require_subset=True,
        min_subset=2,
        certify=True,
        time_budget=4.0,
    )
    ref = DD2DRefiner(budget=None, time_budget=4.0)
    rr = ref.refine(prob.retrieve_only_skeleton(), prob.scene, seed=1)
    return prob.scene, rr


@pytest.mark.slow
def test_harvest_replay_hash_matches():
    scene, rr = _blocked_instance()
    pm = H.harvest_facts(scene, rr, frozenset(), skeleton_idx=0, refinement_seed=1)
    world = H.replay_prefix(scene, pm.harvest_prefix)
    assert H.harvest_state_hash(world) == pm.harvest_state_hash


@pytest.mark.slow
def test_harvest_blocked_at_contents_is_proof():
    scene, rr = _blocked_instance()
    assert not rr.feasible
    pm = H.harvest_facts(scene, rr, frozenset(), skeleton_idx=0, refinement_seed=1)
    blocked = [f for f in pm.facts if f.fact_type == "blocked-at-contents"]
    assert blocked, "a blocked target must yield a blocked-at-contents fact"
    assert blocked[0].tier == "proof"  # DD2D registry
    # every drawer blocker is in the witness contents (removal-monotone args)
    assert len(blocked[0].args) >= 1


@pytest.mark.slow
def test_empty_registry_makes_everything_hints():
    scene, rr = _blocked_instance()
    pm = H.harvest_facts(
        scene,
        rr,
        frozenset(),
        skeleton_idx=0,
        refinement_seed=1,
        registry=EMPTY_REGISTRY,
    )
    assert all(f.tier == "hint" for f in pm.facts)


def test_soundness_registry_tiers():
    assert DD2D_REGISTRY.tier("blocked-at-contents") == "proof"
    assert DD2D_REGISTRY.tier("grasp-witness") == "hint"  # not proof-eligible
    assert DD2D_REGISTRY.tier("pack-impossible") == "proof"
    assert EMPTY_REGISTRY.tier("blocked-at-contents") == "hint"
    # a partial registry does not grant proofs
    partial = SoundnessRegistry(model_fidelity=True, exactness=True)
    assert partial.tier("blocked-at-contents") == "hint"


def test_prefix_reprs_roundtrip_pick_place():
    # a synthetic bound plan of pick+place must serialize to replayable reprs.
    class _Step:
        def __init__(self, params):
            self.params = params

    plan = [
        _Step({"phase": "pick", "item": "o1"}),
        _Step({"phase": "place", "item": "o1", "pose": [1.2345, 2.0, 0.5]}),
    ]
    reprs = H.prefix_reprs(plan)
    assert reprs[0] == "pick|o1"
    assert reprs[1].startswith("place|o1|1.2345|")
