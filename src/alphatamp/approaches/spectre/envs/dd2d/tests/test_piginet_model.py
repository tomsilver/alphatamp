"""Step-7 gate: PIGINet model forward, batched==per-sample, mask structure, grads."""

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
from alphatamp.approaches.spectre.envs.dd2d.record import PIGINetExample
from alphatamp.approaches.spectre.piginet.dataset import record_tokens
from alphatamp.approaches.spectre.piginet.encoders import Encoders
from alphatamp.approaches.spectre.piginet.model import PIGINet
from alphatamp.approaches.spectre.piginet.tokenize import PIGINetTokenizer


@pytest.fixture(scope="module")
def enc():
    return Encoders(device="cpu")


@pytest.fixture(scope="module")
def group(enc, tmp_path_factory):
    """A real problem-group (shared object bank + a couple of records) + its clip512."""
    d = tmp_path_factory.mktemp("ds")
    res = collect_problem(
        seed=1,
        stratum=2,
        config=DD2DCollectConfig(crowd=5, time_budget=8.0),
        split_dir=str(d / "train"),
    )
    assert res.kept
    pdir = str(d / "train" / res.problem_id)
    recs = [
        PIGINetExample.load(r)
        for r in sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))
    ]
    ex0 = recs[0]
    obj_names = [o["name"] for o in ex0.objects]
    import imageio.v2 as imageio
    from PIL import Image

    path_by_obj = {im["object"]: im["path"] for im in ex0.images}
    clip512 = np.stack(
        [
            enc.clip_image(
                Image.fromarray(imageio.imread(os.path.join(pdir, path_by_obj[n])))
            )
            .detach()
            .numpy()
            for n in obj_names
        ]
    ).astype(np.float32)
    pose = np.array([o["pose"] for o in ex0.objects], dtype=np.float32)
    shape = np.array(
        [
            [
                o["shape"]["w"],
                o["shape"]["h"],
                o["shape"]["area"],
                float(o["shape"]["concave"]),
            ]
            for o in ex0.objects
        ],
        dtype=np.float32,
    )
    dwh = ex0.provenance["drawer_wh"]
    toks = [record_tokens(ex, obj_names, len(obj_names), 64) for ex in recs[:3]]
    g = {
        "pid": res.problem_id,
        "obj_names": obj_names,
        "pose": pose,
        "shape": shape,
        "clip512": clip512,
        "drawer_wh": dwh,
        "records": toks,
    }
    return (
        g,
        recs[:3],
        pdir,
        {n: torch.from_numpy(clip512[i]) for i, n in enumerate(obj_names)},
    )


def test_forward_shape_and_finite(enc, group):
    g, recs, _, _ = group
    model = PIGINet(enc)
    logits, gids, labels = model([g])
    assert logits.shape == (len(recs),) and torch.isfinite(logits).all()
    assert gids.shape == (len(recs),) and labels.shape == (len(recs),)


def test_batched_matches_persample(enc, group):
    g, recs, pdir, clip_by_obj = group
    model = PIGINet(enc)
    tok = PIGINetTokenizer(enc, n_max=64)
    # copy the model's learned PEs into the reference tokenizer so both use identical params
    tok.pe_goal, tok.pe_init = model.pe_goal, model.pe_init
    X, _, _, _ = model.embed_batch([g])
    for i, ex in enumerate(recs):
        ref = tok.tokenize(ex, pdir, clip512_by_obj=clip_by_obj)["X"]
        n = ref.shape[0]
        assert torch.allclose(
            X[i, :n], ref, atol=1e-4
        ), f"record {i} batched != per-sample"


def test_causal_plan_mask_structure(enc, group):
    g, recs, _, _ = group
    model = PIGINet(enc, heads=8)
    _, src_mask, key_pad, _ = model.embed_batch([g])
    npl = recs[0].task_plan.__len__() if False else g["records"][0]["n_plan"]
    m0 = src_mask[0]  # first head of record 0; True = blocked
    # plan block strictly upper-triangular blocked; lower+diagonal allowed
    assert bool(m0[0, npl]) is False  # plan attends to a goal/init token (full)
    if npl >= 2:
        assert (
            bool(m0[0, 1]) is True and bool(m0[1, 0]) is False
        )  # 0 can't see 1; 1 can see 0
    # nothing blocked outside the plan block
    assert not m0[npl:, :].any() and not m0[:, npl:].any()


def test_padding_invariance(enc, group):
    """A record's logit is unchanged whether batched alone or padded next to a longer
    one."""
    g, _, _, _ = group
    # split into a 1-record group (short plan) and keep the full group (longer records)
    short = dict(g)
    short["records"] = [g["records"][0]]
    model = PIGINet(enc).eval()
    with torch.no_grad():
        alone, _, _ = model([short])
        together, gids, _ = model([short, g])
    assert torch.allclose(alone[0], together[0], atol=1e-5)


def test_clip_frozen_grads_flow_to_mlps(enc, group):
    g, _, _, _ = group
    model = PIGINet(enc)
    logits, _, _ = model([g])
    logits.sum().backward()
    assert model.head.weight.grad is not None
    assert model.enc.mlp_obj[0].weight.grad is not None
    assert all(p.grad is None for p in model.enc.clip.parameters())  # CLIP frozen
