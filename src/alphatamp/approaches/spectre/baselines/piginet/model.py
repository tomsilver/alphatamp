"""PIGINet model (Step 7 of docs/piginet_dd2d_plan.md; paper §IV-B).

Transformer encoder over the token sequence built by the Step-6 encoders, with the
causal-plan attention mask, reading position 0 (first plan token) → a feasibility logit.

The forward runs a **batched** tokenization for speed: the frozen CLIP features are cached
(``dataset``), so per forward only the trainable MLPs run — once in bulk per problem-group —
and tokens are assembled by masked-mean gather from a per-group stacked feature bank
``[text(|VOCAB|) ; objects(n_obj) ; pose-values(n_obj)]``. A test cross-checks this against
the per-sample :meth:`PIGINetTokenizer.tokenize` to de-risk the index bookkeeping.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoders import D, Encoders


def _sinusoidal(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d)
    )
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PIGINet(nn.Module):
    def __init__(
        self,
        encoders: Encoders | None = None,
        d: int = D,
        layers: int = 3,
        heads: int = 8,
        n_max: int = 64,
        device: str = "cpu",
        dropout: float = 0.2,
        feat_noise: float = 0.0,
        **enc_kwargs,
    ) -> None:
        super().__init__()
        self.enc = encoders or Encoders(d=d, device=device, **enc_kwargs)
        self.d = self.enc.d
        self.heads = heads
        self.n_max = n_max
        self.device = self.enc.device
        self.feat_noise = feat_noise  # train-time Gaussian noise on cached CLIP feats (image-aug proxy)
        self.pe_goal = nn.Parameter(torch.zeros(self.d))
        self.pe_init = nn.Parameter(torch.zeros(self.d))
        nn.init.normal_(self.pe_goal, std=0.02)
        nn.init.normal_(self.pe_init, std=0.02)
        self.register_buffer(
            "_sin_pe", _sinusoidal(n_max, self.d, self.device), persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            self.d,
            heads,
            dim_feedforward=4 * self.d,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # enable_nested_tensor=False: the nested-tensor fast path lacks an MPS kernel and
        # silently falls back to CPU (pathologically slow); the plain padded path is fine.
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.d, 1)
        self.to(self.device)

    # -- per-record token embeddings from a group's stacked bank -------------
    def _record_tokens(self, rec, stacked) -> torch.Tensor:
        idx = torch.as_tensor(rec["elem_idx"], device=self.device)  # (n_tok, max_e)
        mask = torch.as_tensor(rec["elem_mask"], device=self.device)  # (n_tok, max_e)
        feats = stacked[idx]  # (n_tok, max_e, d)
        tok = (feats * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(
            1.0
        )
        kind = torch.as_tensor(rec["kind"], device=self.device)
        planpos = torch.as_tensor(rec["planpos"], device=self.device)
        pe = torch.zeros_like(tok)
        pe[kind == 0] = self._sin_pe[planpos[kind == 0].clamp_min(0)]
        pe[kind == 1] = self.pe_goal
        pe[kind == 2] = self.pe_init
        return tok + pe

    def _group_stacked(self, g, text_bank) -> torch.Tensor:
        clip = torch.as_tensor(g["clip512"], device=self.device)
        if self.training and self.feat_noise > 0:  # feature-space image augmentation
            clip = clip + self.feat_noise * torch.randn_like(clip)
        pose, shape = g["pose"], g["shape"]
        obj = self.enc.object_feat_batch(
            clip, pose, shape, g["drawer_wh"]
        )  # (n_obj, d)
        val = self.enc.value_feat_batch("pose", pose, g["drawer_wh"])  # (n_obj, d)
        return torch.cat([text_bank, obj, val], dim=0)

    def embed_batch(self, batch):
        """-> (X (B,S,d), src_mask (B*heads,S,S) bool, key_pad (B,S) bool, meta)."""
        text_bank = self.enc.text_bank()
        seqs, n_plans, group_ids, labels = [], [], [], []
        for gi, g in enumerate(batch):
            stacked = self._group_stacked(g, text_bank)
            for rec in g["records"]:
                seqs.append(self._record_tokens(rec, stacked))
                n_plans.append(rec["n_plan"])
                group_ids.append(gi)
                labels.append(rec["label"])
        B = len(seqs)
        S = max(s.shape[0] for s in seqs)
        X = torch.zeros(B, S, self.d, device=self.device)
        key_pad = torch.ones(B, S, dtype=torch.bool, device=self.device)
        block = torch.zeros(B, S, S, dtype=torch.bool, device=self.device)
        for b, (s, npl) in enumerate(zip(seqs, n_plans)):
            n = s.shape[0]
            X[b, :n] = s
            key_pad[b, :n] = False
            if npl > 0:  # plan block: causal (block strict upper triangle)
                tri = torch.triu(
                    torch.ones(npl, npl, dtype=torch.bool, device=self.device),
                    diagonal=1,
                )
                block[b, :npl, :npl] = tri
        src_mask = block.repeat_interleave(self.heads, dim=0)  # (B*heads, S, S)
        meta = {
            "group_ids": torch.tensor(group_ids, device=self.device),
            "labels": torch.tensor(labels, device=self.device, dtype=torch.float32),
        }
        return X, src_mask, key_pad, meta

    def forward(self, batch):
        X, src_mask, key_pad, meta = self.embed_batch(batch)
        y = self.encoder(X, mask=src_mask, src_key_padding_mask=key_pad)  # (B,S,d)
        logits = self.head(self.drop(y[:, 0])).squeeze(-1)  # position 0
        return logits, meta["group_ids"], meta["labels"]
