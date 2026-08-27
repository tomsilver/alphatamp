"""Restock3D-v3 SYNTHETIC (analytic-refiner) collection path in
``collect.collect_episode``.

``CollectionConfig(refiner_mode="analytic")`` labels each skeleton with the pure-
geometry ``feasibility_v3.classify_skeleton`` (no motion planning) and SYNTHESIZES the
wall-clock: a fail costs the full ``r_cap`` (= ``refinement_timeout_s``); a success
costs ``U[0.6,0.8]*r_cap``. The resulting ``EpisodeRecord`` must be byte-identical in
shape to a real-refiner run -- populated pool, scene geometry, per-candidate outcomes,
and ``refiner_metadata["failures"]`` dicts in the same shape the real refiner emits --
so every downstream stage consumes it unchanged.
"""

from __future__ import annotations

from alphatamp.approaches.spectre.collect import collect_episode
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S

_FM_KEYS = {
    "step_index",
    "schema",
    "args",
    "culprits",
    "n_step",
    "exhausted",
    "budget_exhausted",
    "dev_added",
    "dev_deleted",
}


def _collect(stratum: int, index: int = 0):
    k_max, r_cap = S.budget(stratum)
    pid = S.problem_id("train", stratum, index)
    cfg = CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant="restock3d_v3",
        model_name="restock3d_v3",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=pid,
        problem_seed_end=pid + 1,
        K_max=k_max,
        abstract_plan_timeout_s=120.0,
        refinement_timeout_s=r_cap,
        num_sampling_attempts_per_step=18,
        max_trajectory_steps=500,
        plan_generator="closed_form",
        refiner_mode="analytic",
    )
    return collect_episode(cfg, pid), r_cap


def test_analytic_episode_is_valid_and_synthetic_wall_clock():
    ep, r_cap = _collect(stratum=0)

    # Standard EpisodeRecord shape.
    assert ep.summary.num_skeletons == len(ep.skeleton_pool) == len(ep.outcomes)
    assert ep.summary.num_skeletons > 0
    assert ep.scene_geometry is not None and ep.scene_geometry.objects
    assert (ep.provenance.gen_params or {}).get("stratum") == 0
    assert ep.summary.num_error == 0  # analytic path never errors

    # No motion planning happened, but every skeleton is labelled + timed.
    assert ep.summary.num_success + ep.summary.num_fail == ep.summary.num_skeletons

    lo, hi = 0.6 * r_cap, 0.8 * r_cap
    for o in ep.outcomes:
        wc = o.refinement_wall_clock_s
        if o.outcome == "fail":
            assert wc == r_cap, f"fail wall-clock {wc} != r_cap {r_cap}"
            # A failed candidate carries a first-violation dict in failure_metadata shape.
            fails = (o.refiner_metadata or {}).get("failures")
            assert fails and set(fails[0]) == _FM_KEYS
            assert o.stuck_step_index == int(fails[0]["step_index"])
        else:
            assert (
                lo - 1e-9 <= wc <= hi + 1e-9
            ), f"success wall-clock {wc} not in [{lo},{hi}]"


def test_analytic_failure_families_have_correct_culprit_shape():
    # A crowded stratum surfaces F2 (residents culprits) + F3 (culprit-free height). F4 reach-over
    # is now culprit-free too (restock3d-v3 tracks ONLY F2 culprits), so F2 is the sole
    # culprit-bearing family.
    ep, _ = _collect(stratum=3)
    saw_culprit_bearing = saw_f3_free = False
    for o in ep.outcomes:
        for f in (o.refiner_metadata or {}).get("failures", []) or []:
            assert set(f) == _FM_KEYS
            if f["schema"] in ("place_tall", "place_short") and not f["culprits"]:
                # height F3: culprit-free, empty (not None) deviation -> a class-2 record.
                saw_f3_free = True
                assert f["dev_added"] == [] and f["dev_deleted"] == []
            if f["culprits"]:
                # F2 residents: class-1, deviation is None (the only culprit-bearing family now).
                saw_culprit_bearing = True
                assert f["dev_added"] is None and f["dev_deleted"] is None
    assert (
        saw_f3_free
    ), "expected at least one culprit-free F3 failure on the crowded stratum"
    assert saw_culprit_bearing, "expected at least one culprit-bearing F2 failure"


def test_real_mode_default_untouched():
    # The default refiner_mode is "real"; analytic is strictly opt-in (v2/other envs unchanged).
    cfg = CollectionConfig(
        env_id=S.env_id(0),
        env_variant="restock3d_v3",
        model_name="restock3d_v3",
        model_kwargs={"stratum": 0},
        split="train",
        num_problems=1,
        problem_seed_start=0,
        problem_seed_end=1,
    )
    assert cfg.refiner_mode == "real"


def _base_cfg(mode: str) -> CollectionConfig:
    return CollectionConfig(
        env_id=S.env_id(0),
        env_variant="restock3d_v3_real",
        model_name="restock3d_v3",
        model_kwargs={"stratum": 0},
        split="train",
        num_problems=1,
        problem_seed_start=0,
        problem_seed_end=1,
        K_max=8,
        plan_generator="closed_form",
        refiner_mode=mode,  # type: ignore[arg-type]
    )


def test_hybrid_mode_is_valid_and_pins_distinct_hash():
    # hybrid_prune is an accepted refiner_mode and pins a config_hash distinct from analytic/real,
    # so hybrid-collected episodes self-describe in provenance.
    hashes = {m: _base_cfg(m).config_hash for m in ("analytic", "real", "hybrid_prune")}
    assert len(set(hashes.values())) == 3


def test_outcome_record_label_source_optional_and_replace_preserved():
    # label_source is a trailing optional field (default None) so legacy/other-env collections are
    # unaffected, and canonicalize's dataclasses.replace preserves it.
    import dataclasses

    from alphatamp.approaches.spectre.schema import OutcomeRecord

    o = OutcomeRecord(
        skeleton_idx=0,
        outcome="fail",
        refinement_wall_clock_s=1.0,
        refinement_seed=7,
    )
    assert o.label_source is None  # default: single-mode / legacy collections
    o2 = dataclasses.replace(o, label_source="real")
    assert o2.label_source == "real"
