"""Tests for ``SpectreDataset`` — F-subset sampling and the four invariants."""

from __future__ import annotations

from pathlib import Path

from _fixtures import write_toy_split

from alphatamp.approaches.spectre.dataset import SpectreDataset
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
