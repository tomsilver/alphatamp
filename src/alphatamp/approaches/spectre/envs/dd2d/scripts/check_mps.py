"""Step 0 gate (docs/piginet_dd2d_plan.md): verify the local ML stack + MPS.

Confirms the M3 laptop can run the PIGINet model stack:
  1. torch imports and reports an MPS device;
  2. a tiny tensor op runs on `mps`;
  3. frozen CLIP ViT-B/32 (open_clip) loads, moves to `mps`, and embeds a dummy
     224x224 image + a text string to FINITE 512-d vectors.

The whole PIGINet model channel is frozen-CLIP encoders + a tiny transformer, so
if this passes, training/eval on MPS is viable (no Della needed).

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/check_mps.py
(The script sets the fallback env var itself if unset, so a plain
 `.venv/bin/python scripts/check_mps.py` also works.)
"""

from __future__ import annotations

import os
import sys

# Any op open_clip uses that lacks an MPS kernel silently falls back to CPU
# instead of crashing. CLIP is a one-time cached pass at train time, so a
# fallback here is harmless. Set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[check_mps] FAIL: {msg}")
    raise SystemExit(1)


def main() -> int:
    print(f"[check_mps] torch {torch.__version__}")
    print(f"[check_mps] mps.is_available() = {torch.backends.mps.is_available()}")
    print(f"[check_mps] mps.is_built()     = {torch.backends.mps.is_built()}")

    # --- Gate 1: MPS device is available -----------------------------------
    if not torch.backends.mps.is_available():
        _fail(
            "torch.backends.mps.is_available() is False -- MPS not usable on this machine"
        )
    device = torch.device("mps")

    # --- Gate 2: a tiny op runs on MPS -------------------------------------
    x = torch.randn(1024, 1024, device=device)
    y = (x @ x).sum()
    if not torch.isfinite(y):
        _fail("tiny MPS matmul produced a non-finite result")
    print(f"[check_mps] tiny MPS op OK  (sum={y.item():.3e})")

    # --- Gate 3: frozen CLIP ViT-B/32 text + image embed on MPS ------------
    import open_clip

    print(
        "[check_mps] loading CLIP ViT-B/32 (open_clip, pretrained=laion2b_s34b_b79k)..."
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval().to(device)
    for p in model.parameters():  # frozen, exactly as used downstream
        p.requires_grad_(False)

    dummy_image = torch.rand(1, 3, 224, 224, device=device)  # a fake preprocessed crop
    text_tokens = tokenizer(["a small round can in a drawer"]).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(dummy_image)
        txt_feat = model.encode_text(text_tokens)

    for name, feat in (("image", img_feat), ("text", txt_feat)):
        if feat.shape[-1] != 512:
            _fail(f"CLIP {name} embedding dim {feat.shape[-1]} != 512")
        if not torch.isfinite(feat).all():
            _fail(f"CLIP {name} embedding has non-finite values")
        print(
            f"[check_mps] CLIP {name} embed OK  shape={tuple(feat.shape)} "
            f"norm={feat.float().norm().item():.3f} device={feat.device}"
        )

    print(
        "[check_mps] PASS: MPS available, tiny op + frozen CLIP text/image embeds all finite (512-d)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
