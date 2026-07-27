"""VLMPlan — a zero-shot VLM planning baseline, in the KinDER convention.

The comparison grid this baseline completes has two axes, *training data* and
*perception*: PIGINet is the trained low-level (pixels + literals) predictor, SPECTRE
the trained abstract-first predictor, and VLMPlan the **zero-training-data,
generic-perception endpoint**. It is a corner of the grid, not a defeated rival — and it
is the reviewer-obvious "did you try just asking a VLM?" question, answered on record.

Where the trained methods *rank* a planner-supplied pool of candidate skeletons, VLMPlan
must *produce* the ordered list itself, over several generation rounds, seeing only its
own prior proposals between them (never refinement outcomes — see `loop.py`). It is
scored on the same metric as everything else: failed refinement attempts before the first
success.

Layout — env-agnostic core plus exactly one env-aware module:

- `template.py`  verbatim KinDER prompt + the appended deviation blocks
- `parsing.py`   vendored KinDER line parser, per-block error semantics
- `adapter.py`   the `EnvAdapter` contract a new environment implements
- `models.py`    factory over `prpl_llm_utils` (local server or frontier API)
- `loop.py`      generation rounds, dedup, stall/exhaustion
- `score.py`     realized attempt sequence -> FP + diagnostics
- `dd2d_adapter.py`  **the only DD2D-aware module**
"""
