# SPECTRE — Decision Log (ADR style)

One entry per consequential decision. Newest first. Format: context → decision
→ consequences. Refactor-era entries record what moved, what deliberately did
not, and why.

---

## 2026-06-07 — Report uncensored evaluation results (attempt budget = pool cap)

**Context.** The RT2D-n3 headline table and figures
(`experiments/spectre/analyze_spectre.ipynb`) were generated with
`ATTEMPT_BUDGET = 30` — equal to the candidate-pool cap — while the living
docs and the writeup described the evaluation attempt budget as 20. At budget
20, ~2–4% of episodes hit the cap and are censored to 21 (budget + 1); at
budget 30 the budget never binds (pool ≤ 30), so every episode runs to its
true first-success attempt and nothing is censored. The frozen-context
ablation (notebook 2026-06-06) surfaced the discrepancy: the full-variant
mean only reproduced the headline at budget 30, not 20.

**Decision.** Headline / reported evaluation metrics use the **uncensored**
budget — attempt budget = candidate-pool cap (30 for RT2D-n3) — so reported
attempt counts are the true time-to-first-success with no censoring. An
uncensored distribution is more informative than a censored one, especially
in the tail: it shows where any method (SPECTRE included) does badly rather
than collapsing those episodes to a single censored value. This is the
standard for SPECTRE evaluation tables going forward; if a future env's pool
cap differs, the eval budget tracks that cap.

**Scope — what this does NOT change.** Model selection and early stopping stay
on `val_rollout_attempts` at its own rollout budget (20) — a separate knob
from evaluation reporting (selection picks the checkpoint; this decision
governs how the chosen checkpoint is reported). The rollout-metric discipline
(proposal.md §5) is untouched.

**Consequences.** Writeup §Training and `archive/README.md` corrected 20 → 30
(the writeup's headline numbers were always budget-30; only the stated budget
was wrong). Pending reconciliation in the same commit: the "attempt budget 20"
phrasing in `proposal.md` §1 and the spectre `CLAUDE.md` headline line, which
refer to the evaluation/reporting budget and should read 30 (uncensored) — the
`val_rollout_attempts` mentions (budget 20, model selection) stay as-is.

---

## 2026-06-06 — Documentation discipline codified in CLAUDE.md

**Context.** The instruction to keep the living docs updated was a single
passive bullet ("Record run outcomes in notebook.md; lasting decisions in
decisions.md; method changes in proposal.md") — where but never when. It
demonstrably failed: `notebook.md` stayed empty for ~2 months of training
runs, every pre-refactor ADR below was reconstructed retroactively, and the
stale AUROC-as-key-metric claims survived ~6 weeks after the rollout-metric
change.

**Decision.** The spectre `CLAUDE.md` gains a "Documentation discipline"
section: a change-type → doc → format routing table (run numbers — including
negative results — → `notebook.md`; lasting choices → `decisions.md` ADR;
method/pipeline/protocol changes → `proposal.md` in place + §6 reconcile), a
before-commit rule (the doc entry ships in the same commit as the change),
a materiality threshold (mechanical refactors/formatting/typos exempt), and
a litmus test ("in 3 months, will we know this happened and why?").

**Alternative rejected.** Mechanical enforcement via a Claude Code hook:
project-level `.claude/settings.json` is committed and would fire for every
user of the shared monorepo, not just spectre development. A personal
`settings.local.json` hook remains an option if instructions alone prove
insufficient.

**Consequences.** Doc updates are part of the definition of done for any
non-trivial spectre commit. This entry is the first written under the rule.

---

## 2026-06-06 — Dated writeup snapshots in `docs/archive/`

**Context.** A high-quality paper-style writeup of the full project state was
deposited as `archive/SPECTRE_WRITEUP_APR_2026.md` (dated 2026-04-27 — two
days after the move to rollout-based model selection, whose checkpoints its
results use). It is a valuable reference but will go stale; the living docs
must not defer to it.

**Decision.** Writeups are dated, frozen, narrative exports named
`SPECTRE_WRITEUP_<MON>_<YYYY>.md` under `docs/archive/`, catalogued in the
"Snapshots" section of `archive/README.md`. The living docs (`proposal.md` /
`decisions.md` / `notebook.md`) remain the source of truth and win on
disagreement. At deposit time: reconcile any divergence into the living docs
first (headline results → a dated `notebook.md` entry; new limitations /
future-work items → `proposal.md` §6), then freeze. After freezing, snapshots
are not edited — staleness annotations go in `archive/README.md`. (One
documented exception: the 2026-06-06 fix of the writeup's pool-cap-30 /
attempt-budget-20 conflation.)

**Consequences.** `notebook.md` seeded with the 2026-04-27 results entry;
writeup-only limitations (data efficiency, Ψ fixed-size summary,
compositional generalization, x₀-conditioned prior) merged into
`proposal.md` §6. Next snapshot due when multi-seed RT2D results land.

---

## 2026-06-04 — Silo refactor: scope and placement

**Context.** Spectre files were scattered across a shared monorepo (root spec
docs, mixed `experiments/`, spectre edits to shared configs). Refactor executed
on branch `spectre-refactor`; safety/reversibility prioritized over tidiness.

**Decisions.**

1. **Docs home = `src/alphatamp/approaches/spectre/docs/`.** Original specs
   moved byte-unchanged to `docs/archive/` (historical notes live in
   `archive/README.md`, not in the files, to keep them unchanged);
   consolidated living proposal in `docs/proposal.md`; this log; `notebook.md`
   for running EDA notes; `RESEARCH_LIT.md` → `docs/research_lit.md`.
2. **Hydra configs live in `experiments/spectre/conf/`, not
   `src/.../spectre/conf/` + `pkg://`.** All five spectre Hydra entry points
   are scripts under `experiments/`; moving scripts and configs *together*
   keeps every `@hydra.main(config_path="conf")` byte-identical, requires no
   `__init__.py` in config dirs, and no package-data additions to the shared
   `pyproject.toml`. The `pkg://` route works under the editable install but
   is strictly more moving parts for zero extra siloing.
3. **All spectre experiment files moved into `experiments/spectre/`**: the 5
   `.py` entry points, 2 `.slurm`, 3 submit/collect `.sh`, the analysis
   notebook and its output artifacts. Shared `experiments/conf/` now contains
   only other-project configs.
4. **The 3 env configs (`clutteredstorage2d_b5`, `routedtransport2d_n3_v1`,
   `stickbutton2d_b5`) moved with spectre's conf tree, not deleted.** They
   were believed unused/historical but are *live*: the first two are composed
   as `defaults` by `spectre_collect`/`spectre_build_vocab`/`spectre_train`;
   the third is selected via CLI override in `submit_spectre_stickbutton2d_b5.sh`.
   Grep-proven that no other project references them.
5. **Shared `experiments/conf/hydra/launcher/slurm.yaml` restored to `main`'s
   values (4 cpus / 16 GB).** Spectre work had bumped it to 8/32 in place — a
   contamination of a shared config also referenced by `collect_data.py`.
   Spectre keeps its tuning in its own copy at
   `experiments/spectre/conf/hydra/launcher/slurm.yaml`, which resolves via
   spectre's config_path.
6. **`.gitignore` `archive/` rule anchored to `/archive/`.** The unanchored
   rule (meant for the root archive of old experiment results) silently
   ignored the new `docs/archive/`. Verified only two `archive` dirs exist in
   the repo, so anchoring is behavior-preserving for everything else.
7. **Deliberately left in place:** `src/alphatamp/approaches/spectre/`
   (IS the importable package), `tests/approaches/spectre/` (import paths),
   `data/spectre/` (the `data_root: "data/spectre"` convention in configs and
   shell scripts is unchanged), `experiments/slurm_outputs/` (shared scratch,
   gitignored — spectre keeps writing there rather than adding new ignore
   rules for a private dir), `experiments/__init__.py` (shared), all
   other-project files (bandit/BOX, sim-free param policy, LLM
   cluttered-storage), `tests/datasets/*.pkl` (other-project fixtures; 1-byte
   pickle churn was `git restore`d, not committed).
8. **Pre-refactor cleanup commits:** `eda.py` (+3-line `set_name` helper)
   committed; `.gitignore` merge-conflict markers fixed and scratch/data
   ignores added (`.data/`, `.sandbox-*`, `data/spectre/{raw,checkpoints,configs,derived}/`,
   `*.ipynb`).

**Consequences / follow-ups.**

- Anyone with muscle-memory paths (`python experiments/spectre_train.py …`)
  must add the `spectre/` segment.
- `main`'s `.gitignore` may still carry the merge-conflict markers — fix worth
  upstreaming separately.
- The slimmed root `CLAUDE.md` and the launcher revert only exist on this
  branch until merged.

---

## Pre-refactor decisions worth remembering (imported from specs/history)

- **2026-04 — Listwise PL loss over pointwise BCE.** Attempt 2 failed because
  BCE is not rollout-aligned; PL `−log P(argmax ∈ SUCC)` is. Load-bearing.
- **2026-04 — F contains failures only.** Test-time F can never contain a
  success; training F ⊆ FAIL_e strictly. Violation was an Attempt-2 root cause.
- **2026-04 — RT2D over kinder kinematic2d.** Lookup-table baseline (B3) is
  near-oracle on kinder envs → no research gap; RT2D engineered so beating B4
  requires relational tag binding (see `archive/SYNTHETIC_ENVIRONMENT.md`).
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
