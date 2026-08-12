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

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .dd2d_adapter import DD2DDomain
from .record import PIGINetExample

#: Predicate whose argument list carries a continuous pose rather than only objects.
#: A shared convention across domains, not a DD2D name: `record_tokens` binds its
#: second element to the object bank and its third to the pose-value bank, which is
#: how the concrete initial state reaches a *low-level* predictor at all. An adapter
#: that emits no `at-pose` literal hands PIGINet object identities with no positions.
POSE_PREDICATE = "at-pose"


def _default_word_idx() -> dict[str, int]:
    """DD2D's word index -- the default so pre-lift call sites are unchanged."""
    return {w: i for i, w in enumerate(DD2DDomain().vocab)}


# --------------------------------------------------------------------------- #
# CLIP cache (frozen, one-time)
# --------------------------------------------------------------------------- #
def precompute_clip_cache(
    data_root, split, encoders, cache_dir, progress=True, domain=None
) -> str:
    """Frozen CLIP-512 per (problem, object), written once to
    ``<cache>/<split>/<pid>.npz``.

    The crops come from the domain, so an environment that stores rendered PNGs and one
    that rasterises from stored geometry both land in the same cache format. An object
    the domain has no crop for gets a zero vector, which is what the DD2D path did for a
    missing file.
    """
    dom = domain if domain is not None else DD2DDomain(data_root)
    out = os.path.join(cache_dir, split)
    os.makedirs(out, exist_ok=True)
    ids = [pid for pid, _ in dom.problems(split)]
    for i, pid in enumerate(ids):
        npz = os.path.join(out, f"{pid}.npz")
        if os.path.exists(npz):
            continue
        crops = dom.crops(split, pid)
        feats = {}
        for name in dom.object_names(split, pid):
            crop = crops.get(name)
            if crop is None:
                feats[name] = np.zeros(512, dtype=np.float32)
            else:
                v = encoders.clip_image(crop)
                feats[name] = v.detach().cpu().numpy().astype(np.float32)
        np.savez(npz, **feats)
        if progress and (i + 1) % 50 == 0:
            print(f"  [clip-cache {split}] {i+1}/{len(ids)}")
    return out


# --------------------------------------------------------------------------- #
# record -> token-index structure
# --------------------------------------------------------------------------- #
def _obj_index(obj_names):
    return {n: i for i, n in enumerate(obj_names)}


def record_tokens(
    ex: PIGINetExample, obj_names, n_obj: int, n_max: int, rng=None, word_idx=None
) -> dict:
    """Element-index token structure for one record.

    Indices reference the per-group stacked bank ``[text(|VOCAB|) ; obj(n_obj) ; pose-
    val(n_obj)]`` (pose values == object poses).
    """
    widx = word_idx if word_idx is not None else _default_word_idx()
    oidx = _obj_index(obj_names)
    obj_off, val_off = len(widx), len(widx) + n_obj
    toks: list[list[int]] = []
    kind: list[int] = []  # 0 plan, 1 goal, 2 init
    planpos: list[int] = []

    for i, step in enumerate(ex.task_plan):  # plan
        el = [widx[step[0]]] + [obj_off + oidx[a] for a in step[1:]]
        toks.append(el)
        kind.append(0)
        planpos.append(i)
    for lit in ex.goal_literals:  # goal
        el = [widx[lit[0]]] + [obj_off + oidx[a] for a in lit[1:]]
        toks.append(el)
        kind.append(1)
        planpos.append(-1)
    init_tok, init_kind = [], []
    for lit in ex.init_literals:  # init
        if lit[0] == POSE_PREDICATE:  # [pred, name, [x, y, theta]]
            name = lit[1]
            el = [widx[POSE_PREDICATE], obj_off + oidx[name], val_off + oidx[name]]
        else:
            el = [widx[lit[0]]] + [obj_off + oidx[a] for a in lit[1:]]
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
def _pid_stratum(pid) -> int:
    """Stratum index (0-3) recovered from a problem id, matching ``compare.stratum_of``.

    Both DD2D (``dd2d_..._s<seed>``) and StickButton2D (``sb2d_s<pid>``) ids end in
    ``_s<int>``. StickButton2D examples carry no ``stratum`` in provenance (only
    ``num_buttons``), so this pid-derived value is the one uniform source the held-out
    training filter and the eval per-stratum breakdown both agree on.
    """
    # Local import: ``compare`` imports ``piginet.eval``, so a top-level import is circular.
    from alphatamp.approaches.spectre.compare import stratum_of  # noqa: PLC0415

    return stratum_of(int(str(pid).rsplit("_s", 1)[-1]))


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
        domain=None,
        keep_strata=None,
    ):
        self.n_max = n_max
        self.k = subsample_k
        self.cache = os.path.join(cache_dir, split)
        self.rng = np.random.default_rng(seed)
        self.domain = domain if domain is not None else DD2DDomain(data_root)
        self.word_idx = {w: i for i, w in enumerate(self.domain.vocab)}
        keep = None if problem_ids is None else set(problem_ids)
        # Held-out-stratum training: keep only problems whose pid-derived stratum index is
        # in ``keep_strata`` (0-3). ``None`` keeps every stratum, so the default path is
        # unchanged. Stratum comes from the pid (via ``_pid_stratum``), not provenance,
        # because SB2D examples carry no ``stratum`` field -- deriving it uniformly is what
        # makes the filter correct on both environments.
        keep_s = None if keep_strata is None else set(keep_strata)
        self.groups = [
            g
            for g in (
                self._build_group(pid, recs)
                for pid, recs in self.domain.problems(split)
                if (keep is None or pid in keep)
                and (keep_s is None or _pid_stratum(pid) in keep_s)
            )
            if g is not None
        ]

    def _build_group(self, pid, examples):
        recs = [(ex.provenance.get("plan_idx", 0), ex) for ex in examples]
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
        # `drawer_wh` is DD2D's spelling and stays the wire key; the fallback is now the
        # domain's frame rather than a hardcoded 50x40, so a metre-scale environment does
        # not silently normalise its poses against a centimetre drawer.
        drawer_wh = ex0.provenance.get("drawer_wh", list(self.domain.frame_extent))
        toks = [
            record_tokens(
                ex, obj_names, len(obj_names), self.n_max, self.rng, self.word_idx
            )
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
