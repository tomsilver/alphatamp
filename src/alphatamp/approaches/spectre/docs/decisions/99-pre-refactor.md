# SPECTRE Decisions — pre-refactor

Decisions imported from the pre-refactor spec stack, kept as the bullet list they were written as.
Index: [README.md](README.md).

<!--trailer-->
## Pre-refactor decisions worth remembering (imported from specs/history)

- **2026-04 — Listwise PL loss over pointwise BCE.** Attempt 2 failed because
  BCE is not rollout-aligned; PL `−log P(argmax ∈ SUCC)` is. Load-bearing.
- **2026-04 — F contains failures only.** Test-time F can never contain a
  success; training F ⊆ FAIL_e strictly. Violation was an Attempt-2 root cause.
- **2026-04 — RT2D over kinder kinematic2d.** Lookup-table baseline (B3) is
  near-oracle on kinder envs → no research gap; RT2D engineered so beating B4
  requires relational tag binding (see `archive/SYNTHETIC_ENVIRONMENT.md`).
  *(⚠️ revisited 2026-06-25 → see the 2026-06-25 pivot entry: RT2D was
  effectively partially observable to the policy, and the evaluation now prefers
  pre-existing envs meeting the representation-advantage property wishlist.)*
- **2026-04 — Layer 2 (parquet) collapsed in the data pipeline.** At
  500/100/100-episode scale, globbing + loading raw episodes is fast enough;
  EDA operates in memory (`archive/SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` §3.1
  has the migration-back checklist).
- **2026-04 — Live frozen-dataclass schema instead of plain dicts.** Pickle
  stability insurance hasn't been needed; live objects let downstream code
  call substrate APIs directly (as-built §3.2).
- **2026-04 — Set-Transformer atom pooling, per-type augmentation policy,
  vocab-driven arity sizing, rollout-aligned F-mix, F-sample multiplier** —
  RT2D fixes 1–5 (`archive/SPECTRE_RT2D_METHOD_SPEC.md` §9).
- **2026-04 — AUROC(3) is the offline diagnostic that tracks test attempts;
  atom-sensitivity probes (D.1/D.2) are red herrings.** Never optimize for the
  probes. *Superseded for model selection (2026-04-25): checkpointing and
  early stopping use rollout-based `val_rollout_attempts` (see the
  overfitting-response entry below); AUROC(3) remains a secondary diagnostic.*
- **2026-04/05 — Overfitting response sequence:** diagnose → extra dropout →
  rollout-based validation/checkpoint selection (`checkpoint_metric =
  "val_rollout_attempts"` in `train.py`, used for both checkpointing and early
  stopping — aligned with the rollout-based test-time objective) → heuristic
  (FF z-score) prior as warm start (`train.prior_type`). Evaluation of prior
  choice pending.
