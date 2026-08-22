"""Restock3D-v3 generator + strata tests: determinism, acceptance bands, dimension ranges,
the hard-strata no-universal-rule property, problem-id encoding, and env reproducibility.
"""

from __future__ import annotations

from alphatamp.approaches.spectre import compare
from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d import generator_v3 as G
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S


def test_determinism():
    for seed in (0, 5, 17):
        for st in S.STRATA:
            assert G.build_spec_v3(seed, st) == G.build_spec_v3(seed, st)


def test_acceptance_and_bands():
    for st in S.STRATA:
        p = S.params(st)
        for seed in range(20):
            spec = G.build_spec_v3(seed, st)  # must not raise
            blocks = spec.blocks()
            nf, _tot, rho = F.feasible_ratio(blocks)
            assert nf >= 1
            assert p.rho_band[0] <= rho <= p.rho_band[1]
            fill = F.min_fill_over_feasible(blocks)
            assert fill is not None and S.FILL_BAND[0] <= fill <= S.FILL_BAND[1]


def test_dims_in_range_and_composition():
    for st in S.STRATA:
        spec = G.build_spec_v3(3, st)
        assert (
            len(spec.names)
            == len(spec.widths)
            == len(spec.heights)
            == len(spec.floor)
            == spec.n
        )
        for w in spec.widths:
            assert F.WIDTH_MIN - 1e-9 <= w <= F.WIDTH_MAX + 1e-9
        for h in spec.heights:
            assert 0.05 - 1e-9 <= h <= F.TALL_CUTOFF + 1e-9
        # at least one forced (tall-only) block is present per problem
        assert any(h > F.SHORT_CUTOFF for h in spec.heights)


def test_hard_strata_crack_both_greedy():
    for st in (2, 3):
        for seed in range(15):
            blocks = G.build_spec_v3(seed, st).blocks()
            for rule in F.HAND_RULES.values():
                assert not F.split_is_feasible(rule(blocks), blocks)


def test_problem_id_roundtrip_and_stratum_of():
    for st in S.STRATA:
        pid = S.problem_id("val", st, 7)
        assert S.decode(pid) == ("val", st, 7)
        assert S.stratum_of(pid) == st
        # v3 rides the shared 4-band, so compare.stratum_of decodes it with no routing edit
        assert compare.stratum_of(pid, "restock3d_v3") == st


def test_env_build_matches_spec():
    from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
        ObjectCentricRestock3DEnvV3,
    )

    spec_fn, pose_fn, secs, cfg = G.stratum_env_args_v3(1)
    env = ObjectCentricRestock3DEnvV3(
        spec_fn, pose_fn, secs, config=cfg, allow_state_access=True
    )
    pid = S.problem_id("train", 1, 2)
    x0, _ = env.reset(seed=pid)
    spec = G.build_spec_v3(pid, 1)
    assert len([o for o in x0 if o.name.startswith("obj_goal")]) == spec.n
    for i, name in enumerate(spec.names):
        o = x0.get_object_from_name(name)
        assert abs(x0.get(o, "half_extent_x") - spec.widths[i] / 2.0) < 1e-6
        assert abs(x0.get(o, "half_extent_z") - spec.heights[i] / 2.0) < 1e-6
