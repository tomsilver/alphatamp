"""Step-6 gate (docs/piginet_dd2d_plan.md): PIGINet element encoders.

Frozen CLIP + trainable MLPs -> g_text / g_val / g_img / g_obj. A module-scoped CPU
``Encoders`` fixture amortizes the one-time CLIP load; a tiny real problem provides objects +
crops so the identity-binding check runs on real data.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

from alphatamp.approaches.spectre.envs.dd2d.dd2d.collect import (
    DD2DCollectConfig,
    collect_problem,
)
from alphatamp.approaches.spectre.piginet.encoders import D, Encoders


@pytest.fixture(scope="module")
def enc():
    return Encoders(device="cpu")


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    res = collect_problem(
        seed=0,
        stratum=1,
        config=DD2DCollectConfig(crowd=0, time_budget=5.0),
        split_dir=str(d / "train"),
    )
    assert res.kept
    pdir = str(d / "train" / res.problem_id)
    rec = sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))[0]
    from alphatamp.approaches.spectre.envs.dd2d.record import PIGINetExample

    return PIGINetExample.load(rec), pdir


def test_text_feat_shape_and_cached(enc):
    a = enc.text_feat("pick")
    b = enc.text_feat("pick")
    assert a.shape == (D,) and torch.isfinite(a).all()
    assert torch.equal(a, b)  # deterministic (cached CLIP-text, same MLP)
    assert not torch.equal(enc.text_feat("pick"), enc.text_feat("retrieve"))


def test_value_feat_shape_and_pose_sensitivity(enc):
    p1 = enc.value_feat("pose", [10.0, 5.0, 1.0], drawer_wh=[40.0, 30.0])
    p2 = enc.value_feat("pose", [30.0, 25.0, 4.0], drawer_wh=[40.0, 30.0])
    s = enc.value_feat("shape", [5.0, 5.0, 25.0, 0.0])
    assert p1.shape == (D,) and s.shape == (D,)
    assert torch.isfinite(p1).all() and torch.isfinite(s).all()
    assert not torch.equal(p1, p2)  # different poses -> different value embeddings


def test_object_feat_identity_binding(enc, sample):
    ex, pdir = sample
    dwh = ex.provenance["drawer_wh"]
    # two DISTINCT objects (different pose, possibly same family) -> distinct g_obj
    o0, o1 = ex.objects[0], ex.objects[1]
    import imageio.v2 as imageio
    from PIL import Image

    def crop(o):
        p = next(im["path"] for im in ex.images if im["object"] == o["name"])
        return Image.fromarray(imageio.imread(os.path.join(pdir, p)))

    f0 = enc.object_feat(o0, crop(o0), dwh)
    f1 = enc.object_feat(o1, crop(o1), dwh)
    assert f0.shape == (D,) and torch.isfinite(f0).all()
    assert (
        f0 - f1
    ).norm() > 1e-4  # objects at different poses must not collapse (§5.6)


def test_obj_channels_ablation_image_only(sample):
    ex, pdir = sample
    enc_img = Encoders(device="cpu", obj_channels=("img",))
    o = ex.objects[0]
    import imageio.v2 as imageio
    from PIL import Image

    p = next(im["path"] for im in ex.images if im["object"] == o["name"])
    f = enc_img.object_feat(
        o,
        Image.fromarray(imageio.imread(os.path.join(pdir, p))),
        ex.provenance["drawer_wh"],
    )
    assert f.shape == (D,) and torch.isfinite(f).all()
