"""PIGINet dataset + CLIP cache (Step 7 of docs/piginet_dd2d_plan.md).

* :func:`precompute_clip_cache` — frozen CLIP-512 per (problem, object) crop (crops are
  shared across a problem's records), written once to ``<cache>/<split>/<pid>.npz``.
* :class:`PIGINetDataset` — **grouped by problem** (the ranking loss + val rollout-FP proxy
  need each problem's plans together). Per problem it keeps the single positive + a
  subsample of ≤K negatives (the nearest-to-success ones + a random spread) to tame the
  50:1 / max-197-plans-per-problem imbalance. ``__getitem__`` returns a *group*: the shared
  object bank (poses/shapes/clip512) + one token-index structure per kept record.

The token-index structures reference a per-group stacked feature bank
``[text(|VOCAB|) ; objects(n_obj) ; pose-values(n_obj)]``; the model (``model.py``) builds
that bank with the trainable MLPs each forward and assembles tokens by masked-mean gather —
so the frozen CLIP work is cached and only the tiny MLPs run per step.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..record import PIGINetExample
from .glosses import VOCAB

_WORD_IDX = {w: i for i, w in enumerate(VOCAB)}
_V = len(VOCAB)


# --------------------------------------------------------------------------- #
# CLIP cache (frozen, one-time)
# --------------------------------------------------------------------------- #
def precompute_clip_cache(data_root, split, encoders, cache_dir, progress=True) -> str:
    import imageio.v2 as imageio
    from PIL import Image

    out = os.path.join(cache_dir, split)
    os.makedirs(out, exist_ok=True)
    pdirs = sorted(glob.glob(os.path.join(data_root, split, "dd2d_*")))
    for i, pdir in enumerate(pdirs):
        pid = os.path.basename(pdir)
        npz = os.path.join(out, f"{pid}.npz")
        if os.path.exists(npz):
            continue
        rec = sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))[0]
        ex = PIGINetExample.load(rec)
        feats = {}
        for img in ex.images:
            p = os.path.join(pdir, img["path"]) if img.get("path") else None
            if p and os.path.exists(p):
                v = encoders.clip_image(Image.fromarray(imageio.imread(p)))
                feats[img["object"]] = v.detach().cpu().numpy().astype(np.float32)
            else:
                feats[img["object"]] = np.zeros(512, dtype=np.float32)
        np.savez(npz, **feats)
        if progress and (i + 1) % 50 == 0:
            print(f"  [clip-cache {split}] {i+1}/{len(pdirs)}")
    return out


# --------------------------------------------------------------------------- #
# record -> token-index structure
# --------------------------------------------------------------------------- #
def _obj_index(obj_names):
    return {n: i for i, n in enumerate(obj_names)}


def record_tokens(
    ex: PIGINetExample, obj_names, n_obj: int, n_max: int, rng=None
) -> dict:
    """Element-index token structure for one record.

    Indices reference the per-group stacked bank ``[text(|VOCAB|) ; obj(n_obj) ; pose-
    val(n_obj)]`` (pose values == object poses).
    """
    oidx = _obj_index(obj_names)
    obj_off, val_off = _V, _V + n_obj
    toks: list[list[int]] = []
    kind: list[int] = []  # 0 plan, 1 goal, 2 init
    planpos: list[int] = []

    for i, step in enumerate(ex.task_plan):  # plan
        el = [_WORD_IDX[step[0]]] + [obj_off + oidx[a] for a in step[1:]]
        toks.append(el)
        kind.append(0)
        planpos.append(i)
    for lit in ex.goal_literals:  # goal
        el = [_WORD_IDX[lit[0]]] + [obj_off + oidx[a] for a in lit[1:]]
        toks.append(el)
        kind.append(1)
        planpos.append(-1)
    init_tok, init_kind = [], []
    for lit in ex.init_literals:  # init
        if lit[0] == "at-pose":  # [at-pose, name, [x,y,theta]]
            name = lit[1]
            el = [_WORD_IDX["at-pose"], obj_off + oidx[name], val_off + oidx[name]]
        else:
            el = [_WORD_IDX[lit[0]]] + [obj_off + oidx[a] for a in lit[1:]]
        init_tok.append(el)

    # init-dropout to n_max (never plan/goal); no-op in practice (max seq ~36 << 64)
    budget = n_max - len(toks) - len(init_tok)
    if budget < 0:
        keep = max(n_max - len(toks), 0)
        rng = rng or np.random.default_rng()
        sel = (
            sorted(rng.choice(len(init_tok), size=keep, replace=False)) if keep else []
        )
        init_tok = [init_tok[j] for j in sel]
    for el in init_tok:
        toks.append(el)
        kind.append(2)
        planpos.append(-1)

    max_e = max((len(t) for t in toks), default=1)
    idx = np.zeros((len(toks), max_e), dtype=np.int64)
    mask = np.zeros((len(toks), max_e), dtype=np.float32)
    for r, el in enumerate(toks):
        idx[r, : len(el)] = el
        mask[r, : len(el)] = 1.0
    n_plan = kind.count(0)
    return {
        "elem_idx": idx,
        "elem_mask": mask,
        "kind": np.asarray(kind, dtype=np.int64),
        "planpos": np.asarray(planpos, dtype=np.int64),
        "n_plan": n_plan,
        "n_goal": kind.count(1),
        "n_init": kind.count(2),
        "label": float(ex.label),
        # eval metadata (rollout-FP orderings + per-stratum breakdown)
        "plan_idx": int(ex.provenance.get("plan_idx", 0)),
        "length": len(ex.task_plan),
        "stratum": ex.provenance.get("stratum"),
    }


# --------------------------------------------------------------------------- #
# subsampling (imbalance handling)
# --------------------------------------------------------------------------- #
def _subsample(recs, k: int, rng):
    """Keep the positive (last plan_idx) + <=k negatives: the nearest-to-success ones
    (highest plan_idx) + a random spread of the rest."""
    pos = [r for r in recs if r[1].label]
    neg = sorted(
        [r for r in recs if not r[1].label], key=lambda r: r[0]
    )  # by plan_idx asc
    if k <= 0 or len(neg) <= k:
        return pos + neg
    n_near = k // 2
    near = neg[-n_near:] if n_near else []
    pool = neg[:-n_near] if n_near else neg
    n_rand = k - len(near)
    rand_idx = rng.choice(len(pool), size=min(n_rand, len(pool)), replace=False)
    rand = [pool[j] for j in sorted(rand_idx)]
    return pos + rand + near


# --------------------------------------------------------------------------- #
# dataset (grouped by problem)
# --------------------------------------------------------------------------- #
class PIGINetDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        cache_dir,
        n_max=64,
        subsample_k=16,
        problem_ids=None,
        seed=0,
    ):
        self.n_max = n_max
        self.k = subsample_k
        self.cache = os.path.join(cache_dir, split)
        self.rng = np.random.default_rng(seed)
        pdirs = sorted(glob.glob(os.path.join(data_root, split, "dd2d_*")))
        if problem_ids is not None:
            pids = set(problem_ids)
            pdirs = [p for p in pdirs if os.path.basename(p) in pids]
        self.groups = [self._build_group(p) for p in pdirs]
        self.groups = [g for g in self.groups if g is not None]

    def _build_group(self, pdir):
        pid = os.path.basename(pdir)
        recs = []
        for r in sorted(glob.glob(os.path.join(pdir, "[0-9]*.json"))):
            ex = PIGINetExample.load(r)
            recs.append((ex.provenance.get("plan_idx", 0), ex))
        if not recs:
            return None
        recs = _subsample(recs, self.k, self.rng)
        ex0 = recs[0][1]
        obj_names = [o["name"] for o in ex0.objects]
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
        clip = np.load(os.path.join(self.cache, f"{pid}.npz"))
        clip512 = np.stack([clip[n] for n in obj_names]).astype(np.float32)
        drawer_wh = ex0.provenance.get("drawer_wh", [50.0, 40.0])
        toks = [
            record_tokens(ex, obj_names, len(obj_names), self.n_max, self.rng)
            for _, ex in recs
        ]
        return {
            "pid": pid,
            "obj_names": obj_names,
            "pose": pose,
            "shape": shape,
            "clip512": clip512,
            "drawer_wh": drawer_wh,
            "records": toks,
        }

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.groups[i]


def collate(groups):
    """A batch is just the list of problem-groups; the model runs the trainable MLPs."""
    return list(groups)
