"""Step-6 gate (docs/piginet_dd2d_plan.md): PIGINet tokenizer.

Sequence length = |π|+|G|+|I|; causal-plan mask (plan block lower-triangular, rest all-
ones); variable |π| / |I| run; init-dropout fires past n_max (plan+goal never dropped).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

from alphatamp.approaches.spectre.envs.dd2d.dd2d.collect import (
    DD2DCollectConfig,
    collect_problem,
)
from alphatamp.approaches.spectre.piginet.encoders import Encoders
from alphatamp.approaches.spectre.piginet.tokenize import PIGINetTokenizer
from alphatamp.approaches.spectre.envs.dd2d.record import PIGINetExample


@pytest.fixture(scope="module")
def tok():
    return PIGINetTokenizer(Encoders(device="cpu"), n_max=64)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    # stratum-2 -> a 5-action plan + several objects (richer sequence)
    res = collect_problem(
        seed=1,
        stratum=2,
        config=DD2DCollectConfig(crowd=5, time_budget=8.0),
        split_dir=str(d / "train"),
    )
    assert res.kept
    pdir = str(d / "train" / res.problem_id)
    recs = sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))
    return [PIGINetExample.load(r) for r in recs], pdir


def test_sequence_length_and_mask(tok, sample):
    exs, pdir = sample
    ex = exs[-1]  # the positive (longest plan)
    out = tok.tokenize(ex, pdir)
    n_plan, n_goal, n_init = out["n_plan"], out["n_goal"], out["n_init"]
    n = n_plan + n_goal + n_init
    assert out["X"].shape == (n, tok.d)
    assert n_plan == len(ex.task_plan) and n_goal == len(ex.goal_literals)
    assert n_init == len(ex.init_literals)  # no dropout (n_max=64 >> seq)

    mask = out["attn_mask"]
    assert mask.shape == (n, n)
    # plan block lower-triangular
    pblk = mask[:n_plan, :n_plan]
    assert torch.equal(pblk, torch.tril(torch.ones(n_plan, n_plan, dtype=torch.bool)))
    # every non-plan-plan block is all-ones (goal/init attend everywhere; plan attends fwd)
    assert mask[n_plan:, :].all() and mask[:, n_plan:].all()
    # a future plan token is masked from an earlier one
    if n_plan >= 2:
        assert not mask[0, 1] and mask[1, 0]


def test_variable_plan_and_init_run(tok, sample):
    exs, pdir = sample
    # the negatives have different plan lengths (different staged subsets) -> all tokenize
    lens = set()
    for ex in exs:
        out = tok.tokenize(ex, pdir)
        lens.add(out["n_plan"])
        assert out["X"].shape[0] == out["n_plan"] + out["n_goal"] + out["n_init"]
    assert len(lens) >= 1  # at least ran; typically multiple plan lengths present


def test_init_dropout(sample):
    exs, pdir = sample
    ex = exs[-1]
    small = PIGINetTokenizer(Encoders(device="cpu"), n_max=10)
    out = small.tokenize(ex, pdir, rng=np.random.default_rng(0))
    n = out["n_plan"] + out["n_goal"] + out["n_init"]
    assert n == 10  # dropped down to n_max
    assert out["n_plan"] == len(ex.task_plan)  # plan never dropped
    assert out["n_goal"] == len(ex.goal_literals)  # goal never dropped
    assert out["n_init"] == 10 - out["n_plan"] - out["n_goal"]
