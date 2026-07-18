"""PIGINet element encoders (paper §IV-A): g_text, g_val, g_img, g_obj.

An :class:`Encoders` module wraps a **frozen** CLIP ViT-B/32 (shared for text + image, the
Step-0 open_clip stack) plus small **trainable** MLPs that map each modality to a common
width ``d``. It is the shared front end for the tokenizer (Step 6) and is trained end-to-end
in Step 7 (only the MLPs + downstream transformer train; CLIP stays frozen).

* ``text_feat(word)``  — MLP over a cached CLIP-text embedding of the word's NL gloss.
* ``value_feat(type, values)`` — typed-padded continuous values (pose / shape) → MLP.
* ``image_feat(pil_or_512)`` — MLP over a CLIP-image embedding of an object crop.
* ``object_feat(obj, clip512, drawer_wh)`` = **g_obj** — fuse the enabled object channels
  (image ⊕ pose ⊕ shape by default) → MLP. ``obj_channels`` is the image-only / geom-only
  ablation switch (docs/piginet_dd2d_plan.md).

CLIP-text is a fixed finite vocabulary, so its embeddings are precomputed once at init.
CLIP-image is deterministic per crop; the Step-7 dataset precomputes it per (problem, object)
and passes the 512-d vector in, but ``clip_image`` can also encode a crop live (used in tests).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .glosses import VOCAB, gloss

CLIP_DIM = 512
D = 256

# value types (paper §IV-A): T = [pose, shape], L = [3, 4]
VALUE_TYPES = ("pose", "shape")
VALUE_LENS = {"pose": 3, "shape": 4}
_SUM_L = sum(VALUE_LENS.values())  # 7
_VAL_IN = len(VALUE_TYPES) + _SUM_L  # one-hot(2) + padded(7) = 9

# fixed domain normalisers (cm; DD2D shapes.py / scene.py ranges) so shape values land ~[-1,1]
_SHAPE_MAX = np.array([25.0, 25.0, 150.0, 1.0], dtype=np.float32)  # w, h, area, concave
_TWO_PI = 2.0 * np.pi


def _mlp(d_in: int, d_out: int) -> nn.Sequential:
    """Linear + ReLU (paper: 'a linear and a ReLU layer'); MLP_img is 3-layer per
    §IV-A."""
    return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU())


class Encoders(nn.Module):
    def __init__(
        self,
        d: int = D,
        device: str = "cpu",
        obj_channels: tuple[str, ...] = ("img", "pose", "shape"),
        clip_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        super().__init__()
        import open_clip

        self.d = d
        self.device = torch.device(device)
        assert all(c in ("img", "pose", "shape") for c in obj_channels) and obj_channels
        self.obj_channels = tuple(obj_channels)

        # frozen CLIP (shared text + image)
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.clip.eval().to(self.device)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # trainable MLPs
        self.mlp_text = _mlp(CLIP_DIM, d)
        self.mlp_val = _mlp(_VAL_IN, d)
        self.mlp_img = nn.Sequential(  # 3-layer per paper §IV-A
            nn.Linear(CLIP_DIM, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        # object_feat concatenates the per-channel *d-dim* features (image_feat/value_feat
        # each return d), so mlp_obj maps d * (#channels) -> d.
        self.mlp_obj = _mlp(d * len(self.obj_channels), d)

        # precompute CLIP-text of every gloss once (fixed vocab)
        self.register_buffer("_text_cache", self._build_text_cache(), persistent=False)
        self._word_index = {w: i for i, w in enumerate(VOCAB)}
        self.to(self.device)

    # -- CLIP-text cache -----------------------------------------------------
    @torch.no_grad()
    def _build_text_cache(self) -> torch.Tensor:
        toks = self.tokenizer([gloss(w) for w in VOCAB]).to(self.device)
        feats = self.clip.encode_text(toks).float()  # (|VOCAB|, 512)
        return feats

    # -- g_text --------------------------------------------------------------
    def text_feat(self, word: str) -> torch.Tensor:
        idx = self._word_index[word]
        return self.mlp_text(self._text_cache[idx])

    # -- g_val ---------------------------------------------------------------
    def _norm_values(self, vtype: str, values, drawer_wh=None) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        if vtype == "pose":
            W, D_ = drawer_wh or (50.0, 40.0)
            x, y, th = v
            return np.array(
                [2.0 * x / W - 1.0, 2.0 * y / D_ - 1.0, (th % _TWO_PI) / np.pi - 1.0],
                dtype=np.float32,
            )
        if vtype == "shape":
            return (v / _SHAPE_MAX).astype(np.float32)
        raise ValueError(vtype)  # pragma: no cover

    def _typed_padded(self, vtype: str, norm: np.ndarray) -> torch.Tensor:
        idx = VALUE_TYPES.index(vtype)
        vec = np.zeros(_VAL_IN, dtype=np.float32)
        vec[idx] = 1.0  # one-hot(type)
        offset = len(VALUE_TYPES) + sum(VALUE_LENS[t] for t in VALUE_TYPES[:idx])
        vec[offset : offset + VALUE_LENS[vtype]] = norm
        return torch.from_numpy(vec).to(self.device)

    def value_feat(self, vtype: str, values, drawer_wh=None) -> torch.Tensor:
        norm = self._norm_values(vtype, values, drawer_wh)
        return self.mlp_val(self._typed_padded(vtype, norm))

    # -- g_img ---------------------------------------------------------------
    @torch.no_grad()
    def clip_image(self, pil) -> torch.Tensor:
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        return self.clip.encode_image(x).float().squeeze(0)  # (512,)

    def image_feat(self, pil_or_512) -> torch.Tensor:
        clip512 = (
            pil_or_512 if torch.is_tensor(pil_or_512) else self.clip_image(pil_or_512)
        )
        return self.mlp_img(clip512.to(self.device))

    # -- batched variants (Step-7 model forward runs these in bulk) ----------
    def text_bank(self) -> torch.Tensor:
        """All vocab word features, (|VOCAB|, d) — computed once per forward, then
        indexed."""
        return self.mlp_text(self._text_cache)

    def _typed_padded_batch(self, vtype: str, norm: np.ndarray) -> torch.Tensor:
        n = norm.shape[0]
        idx = VALUE_TYPES.index(vtype)
        vec = np.zeros((n, _VAL_IN), dtype=np.float32)
        vec[:, idx] = 1.0
        offset = len(VALUE_TYPES) + sum(VALUE_LENS[t] for t in VALUE_TYPES[:idx])
        vec[:, offset : offset + VALUE_LENS[vtype]] = norm
        return torch.from_numpy(vec).to(self.device)

    def value_feat_batch(self, vtype: str, values, drawer_wh=None) -> torch.Tensor:
        v = np.asarray(values, dtype=np.float32).reshape(-1, VALUE_LENS[vtype])
        if vtype == "pose":
            W, D_ = drawer_wh or (50.0, 40.0)
            norm = np.stack(
                [
                    2.0 * v[:, 0] / W - 1.0,
                    2.0 * v[:, 1] / D_ - 1.0,
                    (v[:, 2] % _TWO_PI) / np.pi - 1.0,
                ],
                axis=1,
            ).astype(np.float32)
        else:  # shape
            norm = (v / _SHAPE_MAX).astype(np.float32)
        return self.mlp_val(self._typed_padded_batch(vtype, norm))  # (n, d)

    def object_feat_batch(self, clip512, pose, shape, drawer_wh=None) -> torch.Tensor:
        """g_obj for a batch of objects, (n, d).

        ``clip512`` is (n, 512) precomputed CLIP.
        """
        parts: list[torch.Tensor] = []
        if "img" in self.obj_channels:
            parts.append(self.image_feat(clip512.to(self.device)))
        if "pose" in self.obj_channels:
            parts.append(self.value_feat_batch("pose", pose, drawer_wh))
        if "shape" in self.obj_channels:
            parts.append(self.value_feat_batch("shape", shape))
        return self.mlp_obj(torch.cat(parts, dim=-1))

    # -- g_obj (single) ------------------------------------------------------
    def object_feat(self, obj: dict, clip512, drawer_wh=None) -> torch.Tensor:
        """Fuse the enabled object channels -> a single d-dim object embedding.

        ``clip512`` is a precomputed CLIP-image vector (or a PIL crop, encoded live). Needed
        only when ``"img"`` is in ``obj_channels``.
        """
        parts: list[torch.Tensor] = []
        if "img" in self.obj_channels:
            parts.append(self.image_feat(clip512))
        if "pose" in self.obj_channels:
            parts.append(self.value_feat("pose", obj["pose"], drawer_wh))
        if "shape" in self.obj_channels:
            s = obj["shape"]
            parts.append(
                self.value_feat(
                    "shape", [s["w"], s["h"], s["area"], float(s["concave"])], drawer_wh
                )
            )
        return self.mlp_obj(torch.cat(parts, dim=-1))
