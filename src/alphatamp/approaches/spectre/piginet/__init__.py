"""Simplified PIGINet (Yang et al. 2023) for DD2D — model, training, eval.

Steps 6–8 of docs/piginet_dd2d_plan.md. Step 6 (this): the multimodal front end —
``glosses`` (word→NL phrase for CLIP-text), ``encoders.Encoders`` (frozen CLIP ViT-B/32 +
trainable MLPs producing g_text/g_val/g_img/g_obj), and ``tokenize.PIGINetTokenizer``
(token averaging + positional encodings + causal-plan mask + init-dropout). The transformer
encoder, losses, training, and eval land in Steps 7–8.
"""
