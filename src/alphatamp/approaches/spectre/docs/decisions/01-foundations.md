# SPECTRE Decisions — Foundations

7 entries, 2000-01-01 .. 2026-06-24 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-06-11-b6-higher-horizons-incremental-scoring"></a>
## 2026-06-11 — B6 higher horizons: incremental scoring, top-m pruning, no capping

<!--strip-->
> **id** `2026-06-11-b6-higher-horizons-incremental-scoring` · **status** active ·
> **tracks** baselines, tooling
<!--/strip-->

**Context.** B6's per-decision cost is `O(K^{h+1})` (the `O(K^{h−1})` backup tree
× the `O(K²)` re-conditioning leaf), and every RT2D-n3 pool is exactly K=30, so
the exact search was ~12 min at `h=3` and intractable at `h≥4`. The goal was to
reach higher horizons (does the lookahead premium keep growing, or saturate?)
without distorting the evaluated problem.

**Decisions.**

1. **Reject pool capping (train or eval) as the tractability lever.** Capping
   the candidate pool to the first `K_cap` planner-ordered skeletons was
   considered and rejected. *Eval-capping* changes the evaluated problem: the new
   `solvability_at_cap` diagnostic shows RT2D-n3 successes sit at **every**
   planner depth (test solvable@15 ≈ 0.46, @20 ≈ 0.60, reaching ~1.0 only at
   k=30), so capping below 30 censors real successes. *Train-capping* (a briefly
   adopted "symmetric cap" idea) deletes estimator observations for no benefit —
   the per-key/pairwise NB estimands are properties of the data distribution, not
   of a skeleton's pool position, so the rising OOV it induced was self-inflicted.
   **The q-model is always fit on the full train pools, and eval always uses the
   full K=30** (uncensored-eval discipline, 2026-06-07). A `candidate_cap` knob
   was *not* added.

2. **Incremental Naive-Bayes scoring is the real lever.** The leaf was secretly
   `O(K³)`: each re-conditioning rollout step appends a failure, so every
   `(candidate, F)` score was a fresh `Σ_{k'∈F}` recompute. The search now threads
   a scoring context (`dp_on_counts._Ctx`) that extends per-candidate
   `S_succ`/`S_fail` by **one pairwise term per failure edge** (`O(K)`), turning
   the leaf into `O(K²)`. Measured on RT2D-n3 (exact, unpruned): h=2 **93 s → 9 s**,
   h=3 **740 s → 86 s**, h=4 from intractable to **~minutes** — all reproducing the
   exact attempt counts bitwise. Scores use `np.log` so the incremental `S_succ`
   equals `_adaptive_score` bitwise, preserving the `h=1 ≡ B4` identity. Synthetic
   test models without the primitives fall back to the recompute closures; an
   equivalence test pins the two paths together.

3. **Top-m pruning kept as an optional knob, off by default.** With incremental
   scoring making the *exact* search tractable through h=4, `dp_on_counts_baseline`
   now **defaults to `m=None` (exact)** — a deviation from the in-flight plan's
   `m=12`, justified because exact is now affordable and pruning is lossy (m=12
   cost ~0.09 attempts at h=3: 9.24 vs 9.15, for only ~2×). Pruning remains
   available (`m=12`) to push `h≥5`: it restricts the `min` at each **internal**
   lookahead node to the top-m candidates by greedy index; the **root argmin and
   the leaf walk are never pruned**, so which skeleton may actually be attempted
   is unrestricted and `h=1` is untouched. Guarded by an `m≥K`-equals-unpruned
   exactness test.

**Consequences.** New `eda.solvability_at_cap` (+ notebook figure gating any
capping). `dp_on_counts.py` rewritten around `_Ctx` (incremental + closure
backends) with a `m` pruning width; `eda._build_dp_model` supplies the
incremental primitives (`log_succ`/`log_fail`/`delta` with a shared delta cache).
Exact h-sweep numbers and paired stats: see [`notebook.md` 2026-06-11](../notebook/01-foundations.md#2026-06-11-b6-exact-h-sweep). Belief-MDP
-over-z and any planner/refiner/abstraction change remain out of scope.

---

<a id="2026-06-08-dp-on-counts-b6-baseline"></a>
## 2026-06-08 — DP-on-counts (B6): lookahead skeleton-selection baseline

<!--strip-->
> **id** `2026-06-08-dp-on-counts-b6-baseline` · **status** active · **tracks**
> baselines
<!--/strip-->

**Context.** B4 (Adaptive Historical) is the headline non-learned adaptive
baseline, but it is *myopic*: at each step it picks the single skeleton with the
highest Naive-Bayes success score and never reasons about what the resulting
failure set leaves for later. We wanted a fair, count-only baseline that shares
B4's estimator but adds multi-step lookahead — to bracket how much of any
SPECTRE-vs-B4 gap is "B4 is myopic" vs "B4's count model is weak". The method is
a receding-horizon expectimax over the cost-to-first-success recursion
`V(F) = min_{σ∈R}[c(σ) + q(σ|F)·V(F∪{σ})]`, solved online to depth `h`. It is a
**baseline** (B6), not SPECTRE, and touches no planner/abstraction/refiner.
Several modelling choices were load-bearing and non-obvious.

**Decisions.**

1. **Base-policy depth indexing (h=1 ≡ B4 exactly).** `h=1` is the no-lookahead
   base greedy policy `argmin index(σ|F)`; `h≥2` is
   `argmin_σ[c(σ) + q(σ|F)·W_{h−2}(F∪{σ})]`. The literal one-step backup
   `argmin[c + q·V̂_0]` is a *policy-improvement* of B4, not B4 (counterexample:
   a candidate with marginally lower fail-prob but a much worse continuation gets
   displaced) — so the only way to honour "h=1 reproduces B4 exactly" is to make
   `h=1` the base policy and have `h` count improvement steps above it. Default
   `h=2` = one real lookahead level.

2. **Calibrated two-class NB posterior for `q` (reject `1−clip(exp(score))`).**
   B4's score `S_succ = log p̂(k) + Σ_{k'∈F} log[p̂(k|k')/p̂(k)]` exponentiates to
   an *unnormalized* NB score that exceeds 1 for `|F|≥2`; `1−clip(exp(S_succ))`
   would force `q=0` (a guaranteed success) precisely when conditioning is most
   informative, confounding the h≥2 regime the baseline exists to probe. Instead
   `q(σ|F) = σ(S_fail − S_succ)` with the complementary
   `S_fail = log(1−p̂(k)) + Σ log[(1−p̂(k|k'))/(1−p̂(k))]` — a proper posterior in
   `(0,1)`, no clip. B4's *ranking* still uses the raw `S_succ`, so this does not
   change B4 or the B6 `h=1` selection.

3. **Re-conditioning greedy leaf `W_0 = V^base` (reject the frozen `Σ c·Π q`).**
   The leaf is the true re-conditioning value of the base policy — a
   stationary-greedy rollout that re-selects `σ*` and re-evaluates `q` at each
   step. A frozen leaf (`q` pinned at `F`) is not the value of any policy under
   the re-conditioning dynamics, so the modeled-value monotonicity
   `W_0 ≥ W_1 ≥ W_2` is *not* guaranteed and in fact breaks under positive
   co-failure correlation — which RT2D is engineered to have. With `W_0 = V^base`,
   `W_{ℓ+1} = TW_ℓ ≤ W_ℓ` by policy improvement (`TV^π ≤ V^π` for any stationary
   `π`); since the bound is index-agnostic, the leaf is ordered by the *same*
   index `h=1` uses, keeping leaf-base ≡ `h=1`-base exactly (no nesting wrinkle).
   Cost: the leaf is an `O(K²)` rollout, mitigated by an episode-independent
   `q`/`S_succ` cache plus a call-scoped `W` memo; per-decision is `O(K^{h−1}·K²)`,
   tractable for `h≤3, K≤30`.

4. **`time` objective cost = train per-key mean refine time.** `attempts` uses
   `c≡1`; `time` uses the mean `refinement_wall_clock_s` per canonical key fit on
   train (OOV keys → global mean). Per-skeleton times are logged on
   `OutcomeRecord` but never pre-aggregated, so `_fit_refine_costs` aggregates
   them (mirrors `_fit_marginals`).

5. **Default eval budget = 30 (uncensored).** `dp_on_counts_baseline` defaults
   `attempt_budget=30` (= RT2D-n3 pool cap, the uncensored standard, this log
   2026-06-07) rather than the B1–B5 legacy default of 20, so a direct caller
   does not silently reintroduce censoring. Model selection's
   `val_rollout_attempts` budget (20) is unaffected.

**Monotonicity is a property of the modeled value, not realized rollouts.** The
unit test asserts `W_0 ≥ W_1 ≥ W_2` on the *modeled* value (including a
positive-correlation instance). Realized held-out attempt means need **not** be
monotone in `h` — finite-count `q` and sparser conditioning at large `|F|` make
deeper lookahead optimize against a less reliable model — so the realized
h-curve is reported as a result, never used as a pass/fail gate.

**Out of scope (future work).** A belief-MDP-over-latent-`z` variant: it needs
the scene latent `z` logged per training problem and is an oracle-structured
reference, not a fair count baseline. No refiner/planner/abstraction changes.

**Consequences.** New module `dp_on_counts.py` (env-free search) +
`eda.dp_on_counts_baseline` (B6); B4's NB scorer extracted to the shared
`_adaptive_score` (B4 output unchanged, guarded by existing tests). B6 registered
in `analyze_spectre.py` (h∈{1,2,3} sweep, comparison table/figures, extreme-q
diagnostic). The baseline roster is now B1–B6 (B6 = DP-on-counts; B4 is its
`h=1` special case). Run numbers: see [`notebook.md` 2026-06-08](../notebook/README.md).

---

<a id="2026-06-07-analysis-notebook-converted-marimo"></a>
## 2026-06-07 — Analysis notebook converted to marimo (`.py`)

<!--strip-->
> **id** `2026-06-07-analysis-notebook-converted-marimo` · **status** active ·
> **tracks** tooling
<!--/strip-->

**Context.** The analysis notebook was a Jupyter `.ipynb`
(`experiments/spectre/analyze_spectre.ipynb`, gitignored as scratch). Jupyter's
JSON-on-disk format is opaque to Claude Code — cell outputs are elided and edits
are clumsy — which made iterating on the EDA/comparison notebook with CC
painful.

**Decision.** Convert the notebook to a **marimo** notebook
(`experiments/spectre/analyze_spectre.py`): a pure-Python, text-first format
(cells are `@app.cell` functions) that CC can read and edit directly. The new
`.py` is the canonical analysis notebook and is **tracked**. Both files are kept
for now — the `.ipynb` stays gitignored alongside it — but forward-looking
"how to run the analysis" references point to the `.py`. Behaviour is preserved:
the marimo notebook reproduces every number and artifact (verified — pool-cap
1.000, overlap 0.973/OVERLAPPING, SPECTRE mean-attempts ≈ 5.67, §6 verdict
FAIL on 3.4, plus the SVG/PDF exports). Data root now resolves relative to the
notebook file (`mo.notebook_dir()`), so it runs from any launch directory rather
than only from `experiments/`.

**Keeping it out of CI.** A marimo `.py` does not satisfy `mypy`/`pylint`, so it
is excluded: `[tool.mypy] exclude` regex and `.pylintrc ignore-patterns` both
skip `analyze_spectre.py`; `black`/`isort`/`docformatter` still format it
(marimo files are black-compatible). `marimo` added to the `develop` extra so
the tracked notebook is runnable from a dev install. `run_ci_checks.sh` is
unaffected (verified: mypy reports no issues, pylint skips the file).

**Consequences.** marimo's single-definition dataflow rule forced a few local
renames during conversion (shared plotting temporaries `_`-prefixed; the
`_color_tag` helper promoted to a shared `color_tag`; the styled-table cell split
so its Styler renders, replacing Jupyter's `display(...)`). Historical
references to the `.ipynb` in past records (this log's 2026-06-07 uncensored-eval
entry; `archive/README.md`) are left as-is — they accurately describe figures
generated by the `.ipynb` before this conversion.

---

<a id="2026-06-07-uncensored-evaluation-at-pool-cap"></a>
## 2026-06-07 — Report uncensored evaluation results (attempt budget = pool cap)

<!--strip-->
> **id** `2026-06-07-uncensored-evaluation-at-pool-cap` · **status** active ·
> **tracks** evaluation
<!--/strip-->

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

<a id="2026-06-06-documentation-discipline-codified"></a>
## 2026-06-06 — Documentation discipline codified in CLAUDE.md

<!--strip-->
> **id** `2026-06-06-documentation-discipline-codified` · **status** active ·
> **tracks** process
<!--/strip-->

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

<a id="2026-06-06-dated-writeup-snapshots"></a>
## 2026-06-06 — Dated writeup snapshots in `docs/archive/`

<!--strip-->
> **id** `2026-06-06-dated-writeup-snapshots` · **status** active · **tracks** process
<!--/strip-->

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

<a id="2026-06-04-silo-refactor-scope-placement"></a>
## 2026-06-04 — Silo refactor: scope and placement

<!--strip-->
> **id** `2026-06-04-silo-refactor-scope-placement` · **status** active · **tracks**
> process, infra
<!--/strip-->

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

