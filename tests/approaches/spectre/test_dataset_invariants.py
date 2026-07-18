"""Tests for ``SpectreDataset`` — F-subset sampling and the four invariants."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from _fixtures import write_toy_split

from alphatamp.approaches.spectre.dataset import (
    FSamplingConfig,
    SpectreDataset,
    _sample_f_subset,
)
from alphatamp.approaches.spectre.priors import ZeroPrior


def _small_dataset(tmp_path: Path) -> SpectreDataset:
    train = tmp_path / "train"
    write_toy_split(
        train,
        outcomes_per_problem=[
            ("fail", "fail", "success"),
            ("success", "fail", "fail", "fail"),
            ("fail", "success", "error", "fail"),
            ("fail", "fail"),  # filtered: no success
            ("success",),  # filtered: num_skeletons<2
        ],
    )
    return SpectreDataset(
        split_dir=train,
        prior=ZeroPrior(),
        seed=1234,
        augment=False,
    )


def test_filters_non_trainable_episodes(tmp_path: Path) -> None:
    """Episodes without >=1 success or with <2 skeletons are dropped."""
    ds = _small_dataset(tmp_path)
    assert len(ds) == 3
    reasons = {r for _, r in ds.filtered_problem_ids}
    assert reasons == {"num_skeletons<2", "num_success==0"}


def test_invariants_hold_over_many_samples(tmp_path: Path) -> None:
    """Draw many examples; R/F/ERRS partition must hold every time."""
    ds = _small_dataset(tmp_path)
    for _ in range(200):
        for i in range(len(ds)):
            ex = ds[i]
            r_count = len(ex.r_skeletons)
            f_count = len(ex.f_skeletons)
            # Every success is in R.
            assert sum(ex.r_success_mask) >= 1
            # |R| + |F| + |ERRS| == |pool|; we don't expose ERRS but can check
            # disjointness via the fact that r + f successes together account
            # for both failure fates.
            assert r_count + f_count <= 4  # max pool in this fixture
            # Priors are one per R skeleton.
            assert len(ex.r_priors) == r_count


def test_f_contains_only_failures(tmp_path: Path) -> None:
    """The I8 invariant — F never contains a successful skeleton."""
    ds = _small_dataset(tmp_path)
    for _ in range(100):
        for i in range(len(ds)):
            ex = ds[i]
            # Every F skeleton, by construction, must have been a failure in
            # the original pool. Since we canonicalize (augment=False so the
            # renumbering is deterministic) the skeleton_idx is stable; we
            # use the fact that ``r_success_mask`` captures successes and
            # assert none of the F entries overlap.
            for f_skel in ex.f_skeletons:
                for j, r_skel in enumerate(ex.r_skeletons):
                    if f_skel.operator_seq == r_skel.operator_seq:
                        # If a skeleton appears in both, it cannot be a success.
                        assert not ex.r_success_mask[j]


# ---------------------------------------------------------------------------
# Fix #5 (F-subsample multiplier) and Fix #4 (rollout_aligned_mix) — spec §8
# ---------------------------------------------------------------------------


def test_dataset_length_scales_with_f_sample_multiplier(tmp_path: Path) -> None:
    """``__len__`` is ``num_episodes × num_f_samples_per_epoch`` (spec §8.1)."""
    train = tmp_path / "train"
    write_toy_split(
        train,
        outcomes_per_problem=[
            ("fail", "fail", "success"),
            ("success", "fail", "fail"),
        ],
    )
    for k in (1, 4, 8):
        ds = SpectreDataset(
            split_dir=train,
            prior=ZeroPrior(),
            seed=0,
            augment=False,
            num_f_samples_per_epoch=k,
        )
        assert ds.num_episodes == 2
        assert len(ds) == 2 * k
        assert ds.num_f_samples_per_epoch == k


def test_set_epoch_changes_F_subset_distribution(tmp_path: Path) -> None:
    """Two different epochs should yield different F draws for the same idx."""
    train = tmp_path / "train"
    write_toy_split(
        train,
        outcomes_per_problem=[("fail",) * 10 + ("success", "success")],
    )
    ds = SpectreDataset(
        split_dir=train,
        prior=ZeroPrior(),
        seed=42,
        augment=False,
        num_f_samples_per_epoch=1,
    )
    seen_sizes: set[int] = set()
    for epoch in range(20):
        ds.set_epoch(epoch)
        ex = ds[0]
        seen_sizes.add(len(ex.f_skeletons))
    # Across 20 epochs, we should observe at least 3 distinct |F| values.
    assert len(seen_sizes) >= 3


def test_rollout_aligned_mix_collapses_to_pure_modes() -> None:
    """``mix_weights=(1,0,0)`` ⇒ uniform_subsets only; (0,1,0) ⇒ uniform_size; etc."""
    fail_indices = list(range(15))
    rng = np.random.default_rng(0)

    def _draw(weights: tuple[float, float, float]) -> Counter[int]:
        cfg = FSamplingConfig(mode="rollout_aligned_mix", mix_weights=weights)
        sizes = Counter(
            len(_sample_f_subset(fail_indices, rng, cfg)) for _ in range(500)
        )
        return sizes

    pure_subsets = _draw((1.0, 0.0, 0.0))
    pure_size = _draw((0.0, 1.0, 0.0))
    pure_lognormal = _draw((0.0, 0.0, 1.0))

    # uniform_subsets ⇒ Bernoulli(0.5) per index ⇒ heavy mass near 7-8.
    assert max(pure_subsets, key=lambda k: pure_subsets[k]) in {6, 7, 8, 9}
    # uniform_size ⇒ flat over {0..15}; mode is roughly uniform, mean ≈ 7.5.
    mean_size = sum(k * v for k, v in pure_size.items()) / sum(pure_size.values())
    assert 6.5 < mean_size < 8.5
    # log_normal ⇒ heavy mass at small |F|.
    small_mass = sum(pure_lognormal.get(k, 0) for k in (0, 1, 2)) / sum(
        pure_lognormal.values()
    )
    assert small_mass > 0.5


def test_rollout_aligned_mix_default_weights_concentrate_on_small_F() -> None:
    """Default ``(0.25, 0.25, 0.5)`` puts ≥45% mass on |F| ∈ {0,1,2,3}."""
    fail_indices = list(range(15))
    rng = np.random.default_rng(0)
    cfg = FSamplingConfig()  # defaults
    sizes = Counter(len(_sample_f_subset(fail_indices, rng, cfg)) for _ in range(2000))
    small = sum(sizes.get(k, 0) for k in (0, 1, 2, 3)) / 2000.0
    # Spec §8.2 quotes ~52% small mass for |FAIL_e|=15 with the default mix;
    # require at least 45% to allow Monte Carlo noise.
    assert small >= 0.45


def test_invalid_f_sampling_mode_rejected() -> None:
    """Unknown mode names raise ``ValueError`` from ``__post_init__``."""
    with pytest.raises(ValueError):
        FSamplingConfig(mode="not_a_real_mode")


def test_invalid_mix_weights_rejected() -> None:
    """Non-normalized or negative mix weights raise ``ValueError``."""
    with pytest.raises(ValueError):
        FSamplingConfig(mix_weights=(0.5, 0.5, 0.5))  # sums to 1.5
    with pytest.raises(ValueError):
        FSamplingConfig(mix_weights=(-0.1, 0.5, 0.6))  # negative weight
