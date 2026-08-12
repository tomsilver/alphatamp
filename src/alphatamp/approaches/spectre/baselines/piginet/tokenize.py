"""PIGINet tokenizer (paper §IV-B): record -> input token sequence + attention mask.

Turns one :class:`~blocks_tamp.record.PIGINetExample` into the sequence a transformer
encoder consumes:

    x = [ h(π_1) … h(π_{n1}),  h(G_1) … h(G_{n2}),  h(I_1) … h(I_{n3}) ]

where each token ``h(z) = mean(element embeddings) + positional_encoding`` (averaging, not
concatenation, handles the variable element count per literal — paper §IV-B). Plan tokens
get a **sinusoidal** PE indexed by action position; goal and init tokens get a single
**learned** PE each (their internal order is irrelevant). The attention mask is all-ones
except the **plan×plan block, which is lower-triangular** (the causal-plan bias: an action's
feasibility does not depend on later actions). Init tokens are dropped uniformly at random
down to ``n_max`` when the sequence is too long (plan/goal are never dropped).

Per-object ``g_obj`` embeddings are computed once per record (each object recurs across many
literals). Crops are loaded from ``<record_dir>/images/<obj>__topdown.png`` unless a
precomputed ``clip512`` map is supplied (the Step-7 cached path).
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn

from .encoders import D, Encoders


def _sinusoidal_pe(n: int, d: int, device) -> torch.Tensor:
    """Standard sinusoidal positional encodings, shape (n, d)."""
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d)
    )
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PIGINetTokenizer(nn.Module):
    def __init__(
        self,
        encoders: Encoders | None = None,
        d: int = D,
        n_max: int = 64,
        device: str = "cpu",
        **enc_kwargs,
    ) -> None:
        super().__init__()
        self.enc = encoders or Encoders(d=d, device=device, **enc_kwargs)
        self.d = self.enc.d
        self.n_max = n_max
        self.device = self.enc.device
        # learned positional encodings (order within G / I is irrelevant -> shared)
        self.pe_goal = nn.Parameter(torch.zeros(self.d))
        self.pe_init = nn.Parameter(torch.zeros(self.d))
        nn.init.normal_(self.pe_goal, std=0.02)
        nn.init.normal_(self.pe_init, std=0.02)
        self.to(self.device)

    # -- per-object embeddings ----------------------------------------------
    def _object_feats(self, ex, record_dir, clip512_by_obj) -> dict[str, torch.Tensor]:
        drawer_wh = ex.provenance.get("drawer_wh")
        path_by_obj = {img["object"]: img.get("path") for img in ex.images}
        feats: dict[str, torch.Tensor] = {}
        need_img = "img" in self.enc.obj_channels
        for o in ex.objects:
            name = o["name"]
            clip512 = None
            if need_img:
                if clip512_by_obj is not None and name in clip512_by_obj:
                    clip512 = clip512_by_obj[name]
                else:
                    p = path_by_obj.get(name)
                    full = os.path.join(record_dir, p) if p else None
                    if full and os.path.exists(full):
                        import imageio.v2 as imageio
                        from PIL import Image

                        clip512 = self.enc.clip_image(
                            Image.fromarray(imageio.imread(full))
                        )
                    else:  # no crop -> zero CLIP vector (occluded/missing, per paper)
                        clip512 = torch.zeros(512, device=self.device)
            feats[name] = self.enc.object_feat(o, clip512, drawer_wh)
        return feats

    # -- one literal / action -> one token ----------------------------------
    def _token(self, elems: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(elems, dim=0).mean(dim=0)

    def _plan_token(self, step, obj_feats) -> torch.Tensor:
        op, args = step[0], step[1:]
        return self._token([self.enc.text_feat(op)] + [obj_feats[a] for a in args])

    def _literal_token(self, lit, obj_feats, drawer_wh) -> torch.Tensor:
        pred, rest = lit[0], lit[1:]
        elems = [self.enc.text_feat(pred)]
        if pred == "at-pose":  # [at-pose, name, [x,y,theta]]
            name, pose = rest[0], rest[1]
            elems += [obj_feats[name], self.enc.value_feat("pose", pose, drawer_wh)]
        else:
            elems += [
                obj_feats[a] for a in rest
            ]  # object args (may be empty, e.g. handempty)
        return self._token(elems)

    # -- full sequence -------------------------------------------------------
    def tokenize(self, ex, record_dir: str, clip512_by_obj=None, rng=None) -> dict:
        drawer_wh = ex.provenance.get("drawer_wh")
        obj_feats = self._object_feats(ex, record_dir, clip512_by_obj)

        # plan tokens (+ sinusoidal PE by action index)
        plan = [self._plan_token(s, obj_feats) for s in ex.task_plan]
        if plan:
            pe = _sinusoidal_pe(len(plan), self.d, self.device)
            plan = [t + pe[i] for i, t in enumerate(plan)]

        goal = [
            self._literal_token(l, obj_feats, drawer_wh) + self.pe_goal
            for l in ex.goal_literals
        ]
        init = [
            self._literal_token(l, obj_feats, drawer_wh) + self.pe_init
            for l in ex.init_literals
        ]

        # init-dropout: keep plan + goal, drop init uniformly down to n_max
        budget = self.n_max - len(plan) - len(goal)
        n_init = len(init)
        if n_init > max(budget, 0):
            keep = max(budget, 0)
            rng = rng or np.random.default_rng()
            idx = sorted(rng.choice(n_init, size=keep, replace=False)) if keep else []
            init = [init[i] for i in idx]

        tokens = plan + goal + init
        X = (
            torch.stack(tokens, dim=0)
            if tokens
            else torch.zeros(0, self.d, device=self.device)
        )
        mask = self._attn_mask(len(plan), len(goal), len(init))
        return {
            "X": X,
            "attn_mask": mask,
            "n_plan": len(plan),
            "n_goal": len(goal),
            "n_init": len(init),
        }

    def _attn_mask(self, n_plan, n_goal, n_init) -> torch.Tensor:
        """(n,n) bool: True = allowed to attend.

        All-ones except plan×plan lower-triangular.
        """
        n = n_plan + n_goal + n_init
        m = torch.ones(n, n, dtype=torch.bool, device=self.device)
        if n_plan:
            m[:n_plan, :n_plan] = torch.tril(
                torch.ones(n_plan, n_plan, dtype=torch.bool, device=self.device)
            )
        return m
