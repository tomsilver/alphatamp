"""Tests for the LAZY untyped-adaptive baseline and the seed-checksum guard (Step 6)."""

from __future__ import annotations

import numpy as np
import pytest
from _fixtures import write_toy_split

from alphatamp.approaches.spectre import eda


def _split(tmp_path, name, outcomes):
    d = tmp_path / name
    write_toy_split(d, outcomes)
    return eda.load_split_episodes(d)


def test_lazy_runs_and_reaches_success(tmp_path):
    outcomes = [("fail", "fail", "success"), ("fail", "success", "fail")]
    split = _split(tmp_path, "s", outcomes)
    res = eda.lazy_baseline(split, split, attempt_budget=10)
    assert res.attempts.shape == (2,)
    assert not res.censored.any()  # both episodes have a success within budget
    assert res.name.startswith("B_LAZY(beta=")


def test_lazy_beta_zero_equals_default_order(tmp_path):
    outcomes = [("fail", "fail", "success"), ("fail", "success", "fail")]
    split = _split(tmp_path, "s", outcomes)
    lazy0 = eda.lazy_baseline(split, split, attempt_budget=10, betas=(0.0,))
    default = eda.default_order_baseline(split, attempt_budget=10)
    # beta=0 -> score is -index -> pure default order.
    assert np.array_equal(lazy0.attempts, default.attempts)


def test_lazy_beta_is_tuned_on_train(tmp_path):
    split = _split(tmp_path, "s", [("fail", "fail", "success")])
    res = eda.lazy_baseline(split, split, attempt_budget=10, betas=(0.0, 1.0, 4.0))
    assert "beta=" in res.name  # a beta was selected and recorded


def test_seed_checksum_distinct(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"seed_{i}.pt"
        p.write_bytes(f"weights-{i}".encode())
        paths.append(p)
    sums = eda.assert_distinct_seed_checkpoints(paths)
    assert len(set(sums.values())) == 3


def test_seed_checksum_rejects_duplicates(tmp_path):
    p0 = tmp_path / "a.pt"
    p0.write_bytes(b"same")
    p1 = tmp_path / "b.pt"
    p1.write_bytes(b"same")  # duplicate content = duplicated seed
    p2 = tmp_path / "c.pt"
    p2.write_bytes(b"different")
    with pytest.raises(AssertionError, match="duplicate"):
        eda.assert_distinct_seed_checkpoints([p0, p1, p2])


def test_seed_checksum_requires_three(tmp_path):
    p = tmp_path / "only.pt"
    p.write_bytes(b"x")
    with pytest.raises(AssertionError, match=">= 3"):
        eda.assert_distinct_seed_checkpoints([p])
