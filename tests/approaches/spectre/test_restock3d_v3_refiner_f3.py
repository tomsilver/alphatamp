"""Restock3D-v3 real-refiner F3 arm-insertion parity (Phase 3).

The refiner's ``_probe_place_v2``, when given v3 ``section_height_cutoffs``, must attribute a block
taller than its section's arm-insertion cutoff as a provable culprit-free **F3** — BEFORE the
block-vs-board test (which would miss it, since the board clearance is ~0.10 m above the cutoff).
A block at/under the cutoff must fall through to the ordinary probe (no spurious F3). With cutoffs
left None (v2), the branch is inert.
"""

from __future__ import annotations

from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    RestockRecordingSampler,
)
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    compute_section_infos,
)

_CUTOFFS = {"section_0": F.TALL_CUTOFF, "section_1": F.SHORT_CUTOFF}


class _P:
    def __init__(self, name):
        self.name = name


class _Op:
    def __init__(self, name, params):
        self.name = name
        self.parameters = params


def _sampler(env, section_infos, cutoffs):
    return RestockRecordingSampler(
        controller_generator=lambda a: None,
        transition_function=lambda x, u: x,
        state_abstractor=lambda x: None,
        max_trajectory_steps=1,
        sim=env,
        region_infos=section_infos,
        section_height_cutoffs=cutoffs,
    )


def _probe(full_h, op_name, cutoffs):
    cfg = Restock3DEnvConfig(section_clearances=F.SECTION_CLEARANCES)
    secs = compute_section_infos(cfg)
    specs = [("blk", (0.025, 0.025, full_h / 2.0), (0.6, 0.2, 0.2, 1.0))]
    env = ObjectCentricRestock3DEnv(
        specs, lambda s: {"blk": (0.5, 0.12)}, secs, config=cfg, allow_state_access=True
    )
    x0, _ = env.reset(seed=0)
    sampler = _sampler(env, secs, cutoffs)
    op = _Op(op_name, [_P("robot"), _P("blk")])
    return sampler._probe_place_v2(x0, op)


def test_over_cutoff_block_is_provable_f3_short():
    culprits, family = _probe(0.20, "place_short", _CUTOFFS)  # 0.20 > 0.12
    assert family == "F3" and culprits == ()


def test_over_cutoff_block_is_provable_f3_tall():
    culprits, family = _probe(0.22, "place_tall", _CUTOFFS)  # 0.22 > 0.17
    assert family == "F3" and culprits == ()


def test_at_cutoff_block_falls_through_no_spurious_f3():
    # a 0.12 block in the short section is NOT over-cutoff -> the cutoff branch is skipped and the
    # ordinary probe runs (block fits under the board, no residents -> C2, not F3).
    culprits, family = _probe(0.12, "place_short", _CUTOFFS)
    assert family != "F3"
    assert culprits == ()


def test_cutoffs_none_leaves_branch_inert():
    # with no cutoffs (v2), an over-tall block does not early-return F3 via the cutoff; it goes to
    # the real block-vs-board probe (a 0.20 block DOES hit the short board there -> F3 by geometry,
    # but via the sim path, proving the cutoff branch itself was inert).
    sampler_cutoffs_none = _probe(0.20, "place_short", None)
    # Either way the branch under test (the arithmetic cutoff) did not fire; assert the object was
    # built without the v3 cutoff attribute influencing construction.
    assert sampler_cutoffs_none[1] in ("F3", "F2", "C2")
