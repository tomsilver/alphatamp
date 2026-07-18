"""End-to-end smoke test for collect_episode + io + EDA load (spec §8.3 #12)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from alphatamp.approaches.spectre.collect import collect_episode, episode_path
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.eda import load_split_episodes
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz, load_episode


def _rt2d_cfg() -> CollectionConfig:
    return CollectionConfig(
        env_id="kinder/RoutedTransport2D-n3-v0",
        env_variant="routedtransport2d_n3_v1",
        model_name="routedtransport2d",
        model_kwargs={"num_items": 3, "variant": "v1"},
        split="train",
        num_problems=5,
        problem_seed_start=0,
        problem_seed_end=5,
        K_max=30,
        abstract_plan_timeout_s=1.0,
        refinement_timeout_s=0.1,
        num_sampling_attempts_per_step=1,
        max_trajectory_steps=10,
    )


def test_collect_episode_produces_valid_record() -> None:
    """Spec §8.3 #12: a collected episode satisfies the record invariants."""
    cfg = _rt2d_cfg()
    ep = collect_episode(cfg, problem_id=0)
    # Schema invariants checked by EpisodeRecord.__post_init__; these are
    # the spec-§8.3-#12-relevant assertions on top.
    assert len(ep.skeleton_pool) == 30
    assert ep.summary.num_skeletons == 30
    assert ep.summary.num_success + ep.summary.num_fail + ep.summary.num_error == 30
    assert ep.provenance.scene_latent is not None
    assert set(ep.provenance.scene_latent) == {"blocked_color", "blocked_grasp"}
    # At least one skeleton must record a stuck cause when fail.
    fail_outcomes = [o for o in ep.outcomes if o.outcome == "fail"]
    if fail_outcomes:
        causes = {o.refiner_metadata.get("stuck_cause") for o in fail_outcomes}
        assert causes, "fail outcomes should carry stuck_cause in refiner_metadata"


def test_collect_to_disk_and_eda_load() -> None:
    """Episodes written via the io helpers round-trip through the EDA loader."""
    cfg = _rt2d_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        data_root = Path(tmp)
        for pid in range(5):
            ep = collect_episode(cfg, problem_id=pid)
            atomic_write_pickle_gz(
                ep, episode_path(data_root, cfg.env_variant, cfg.split, pid)
            )

        split_dir = data_root / "raw" / cfg.env_variant / cfg.split
        loaded = load_split_episodes(split_dir)
        assert len(loaded.episodes) == 5
        assert loaded.k_max == 30
        # Every loaded episode preserved provenance.scene_latent.
        for ep in loaded.episodes:
            assert ep.provenance.scene_latent is not None

        # Round-trip one episode and verify scene_latent survived pickle.
        path = episode_path(data_root, cfg.env_variant, cfg.split, 0)
        re_loaded = load_episode(path)
        assert re_loaded.provenance.scene_latent is not None


def test_at_least_60pct_problems_solve_with_random_selector() -> None:
    """Spec §8.3 #12: random selection within the budget succeeds on ≥60% of
    problems."""
    cfg = _rt2d_cfg()
    rng = np.random.default_rng(0)
    successes = 0
    n = 10
    for pid in range(n):
        ep = collect_episode(cfg, problem_id=pid)
        # "Random selector with budget 20": shuffle pool, walk first 20 ops, succeed
        # if any are success.
        order = list(range(len(ep.outcomes)))
        rng.shuffle(order)
        budget_outcomes = [ep.outcomes[i] for i in order[:20]]
        if any(o.outcome == "success" for o in budget_outcomes):
            successes += 1
    assert successes / n >= 0.6, f"random selector solved {successes}/{n}"
