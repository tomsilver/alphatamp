# Adaptivity on Restock3D-v3 — probe plan, fixes, and cross-env non-regression

**Status: EXECUTED 2026-08-21.** Probes P0–P4 ran; the coverage canonicalize bug was found + fixed;
`repeat` (F3 exact-step certificate, `step_certificate` flag) was built + deployed and **revived the
inert adaptivity — adaptive 12.18 → 3.13, ~97% of the P2 oracle ceiling**; `regroup` was priced at
~1%, gated by `grouping_certificate` (inert cross-env), and is **deprecated/off**. Results:
[`notebook` 2026-08-21](notebook/07-stickbutton2d.md#2026-08-21-restock3d-v3-adaptivity-revived-repeat-f3)
/ [`decisions` 2026-08-21](decisions/07-stickbutton2d.md#2026-08-21-restock3d-v3-adaptivity-revived-coverage-canonicalize);
fix-only pre-registration in [`adaptivity_fix_only_prereg.md`](adaptivity_fix_only_prereg.md). The
DD2D/SB2D non-regression retrains (§5.3) remain deferred (pre-checks §5.2 passed).

**Original status:** probe plan + feature spec, 2026-08-21. Nothing here is built yet except where marked.
**Epistemic tags:** `[M]` measured · `[D]` by-construction (follows from code/definitions) · `[P]` registered prediction · `[U]` unverified.
**Authoritative sources:** `as_built.md` (§5 evidence, §10 tables), `unified_evidence.py` (blame :307, coverage :479, covered :425, waste :555, superfluous :506, _justified :529), `dataset.py` (culprit pool :988/:319, overlap vector :999), `feasibility_v3.py` (classify_skeleton :248).
**Metric throughout:** FP = failed refinements before first success, lower better. Probes run on the collected analytic-labeled v3 dataset; paper eval stays on the real refiner — so every ceiling below is an *analytic-model* ceiling (the G1/audit-slice agreement is what licenses reading it as approximately real).

---

## 1. The result being explained

v3 (400/100/100, n = 6–9, 3 seeds) `[M]`:

| method | ALL | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|
| astar-dist | 38.41 | 5.48 | 14.72 | 49.64 | 83.80 |
| PIGINet | 38.11 ± 1.01 | 6.05 | 16.15 | 56.04 | 74.20 |
| SPECTRE-adaptive | 11.11 ± 0.98 | 1.81 | 3.53 | 13.05 | 26.04 |
| SPECTRE-static | 11.05 ± 0.88 | 1.77 | 3.43 | 12.69 | 26.29 |
| LAZY-adaptive | 11.79 ± 0.08 | 2.64 | 6.25 | 18.15 | 20.11 |

Two facts to explain: (a) adaptive − static = +0.06, inside seed noise — adaptivity is inert; (b) LAZY **beats** SPECTRE at n=9 (20.11 vs 26.04) — crude untyped failure conditioning is harvesting value the typed pathway is not. The representation win (11 vs 38) is real and not in question.

## 2. Working diagnosis and hypotheses

**Core diagnosis** `[D, needs measurement]`: v3's decision is *which blocks share a level* (grouping/assignment); coverage and waste speak *ordering* and *idle work*. Specifically:

- **F3 (height), ~85% of failures** (stratum-0 census: ~1114 F3 vs ~197 culprit-bearing F2 `[M]`), is blameless: `blame(F3) = ∅`, so F3 never populates the culprit pool `K`. After a height-only failure history, coverage = waste = 0 for every candidate — yet "b cannot go on the short level" is the highest-value sentence the environment utters (it tightens the bottom budget for every remaining candidate).
- **F2 (crowding)** names residents as culprits, but `covered()` asks an ordering question ("discharge the culprit before re-entering the failed situation") of an assignment decision. Worse, matching is by abstract *effects*, and `place_tall(b)` / `place_short(b)` have identical effects — the level bit, the only bit that matters, is erased by construction. Hence near-constant coverage within pools.
- **Waste** is vacuous: every step is goal-necessary, the superfluous set is empty, the convention returns a constant.
- **F4 (reach-over)**, the one family coverage genuinely understands, barely occurs (geometry prior orders picks correctly).

**Hypotheses** (compatible; probes assign weight):

- **H1 — evidence-blank:** the scalar columns are near-constant within pools under realistic failure contexts. `[P: true]`
- **H2 — harvestable headroom:** observed failures logically prune enough candidates that a perfect re-ranker sits well below static. `[P: true, large at n≥8]`
- **H3 — no headroom / pool-limited:** pools lack assignment diversity, or feasible splits rank so deep that no evidence-based re-ranking helps. `[P: false, but P2 decides]`
- **H4 — model-side:** evidence reaches the model non-constant (esp. via record tokens + evidence-attn) but the trained model ignores it. `[U]`

## 3. Probes (≈ one day total, no training)

**P0 — incidence census** (minutes). Per-stratum histogram of failure families, culprit counts, and provable-marker (`proves_failure()`) status across the collected dataset. *Learn:* how much evidence mass is blameless; verify the analytic classifier marks F3 provable (implementation check — if it doesn't, fix before repeat lands).

**P1 — scalar variance audit** (≈1 h). For each episode, sample failure contexts F with the *same sampler training uses*; compute within-pool std of the coverage and waste columns; report the distribution and the fraction of contexts with std < 0.01, per stratum and per context composition (F3-only vs mixed).
*Registered prediction:* ≥70% of contexts have coverage std < 0.01; waste ≡ constant.
*Learn:* H1 directly, and separately per column. If coverage variance is substantial, the diagnosis above is wrong in an interesting way — stop and re-read the contexts before building anything.

**P2 — oracle re-ranker ceiling** (afternoon; **the decisive probe**). Replay each test rollout as bookkeeping over the stored pool + records, no learning:

- *oracle-strict:* maintain a constraint set from observed failures. A height record on `place_X(b)` kills every candidate containing that exact step. A crowding record (failed step on target t, culprits/residents R, level L) kills every candidate whose L-group ⊇ {t} ∪ R — equivalently: contains the failed step and all establishing steps (§4.2). Re-rank survivors by the frozen static scores; roll out; count FP.
- *oracle-graded:* additionally demote (not kill) partial rebuilds, weighted by the re-assembled chart fraction.
- Also log pool diagnostics: distinct level-splits per pool; depth of the first feasible candidate under static order.

*Registered prediction:* FP_static − FP_strict ≥ 30% of FP_static at n=8/9.
*Learn:* headroom (H2 vs H3). **Abort criterion:** if headroom < 10% of static at every stratum, feature work is pointless — the bottleneck is the pool; jump to §4.6 pool fix. The strict-vs-graded gap prices `regroup_frac` (§4.3) before any retrain.

**P3 — LAZY decomposition** (≈1 h). `eda.lazy_baseline` with β=0 vs tuned β at n=8/9. *Learn:* how much FP the untyped overlap term alone buys; calibrates the crude-signal floor that typed features must beat.

**P4 — channel ablation at eval** (≈1 h, existing checkpoints). SPECTRE-adaptive with (a) scalar overlap columns zeroed, (b) record tokens masked, (c) both, (d) scrambled contexts (records from other episodes). *Registered prediction:* ΔFP ≈ 0 in all four cells (the model has learned to ignore blank inputs). Any materially nonzero cell revises H4 and redirects part of the fix to training-side (§4.6).

**Decision point A.** Expected outcome H1 ∧ H2 (blank inputs, real headroom) → proceed to §4. H3 → pool workstream. H4-nonzero → add the training-side isolating ablation before/alongside features.

## 4. Fixes

### 4.1 `repeat` (binary, certificate-scoped)

For each observed record that is an **intrinsic step certificate**: does the candidate contain the exact failed step (schema + tag-canonicalized args)? `repeat = max` over such records; 0 when none.

**Scope [M, corrected 2026-08-21] — `proof_tier(schema) ∧ provable ∧ blame == ∅`, NOT "provable ∧ culprit-free".** The blame-structure census (P5) refuted the simpler scope: "provable" alone is uninformative here (the analytic classifier makes **100 %** of v3 records provable), and "provable ∧ culprit-free" is **not env-safe** — DD2D's blameless-provable records are **92 %** *means-failures* (an exhausted pick/place sample, context-dependent), which `repeat` would wrongly veto; SB2D shows empty-`culprits` ≠ blameless (36 % blame via the deviation channel), so the predicate must be `blame == ∅`. The discriminator between an intrinsic certificate (v3 F3, dead in any context) and a means-failure is `proof_tier` (monotone ∧ local ∧ exact). The `blame == ∅` term then excludes v3's culprit-bearing F2 within the same `place` schema.

- *v3 dependency [M]:* v3 is currently `EMPTY_SPEC`, so **`repeat` requires a v3 `DomainSpec` declaring `place_tall`/`place_short` proof-tier** (their blameless failure = the F3 height proof). Care: do not let that declaration spuriously fire the `dead`/demotion path on F2 — gate that path with `blame == ∅` too, or give `repeat` a dedicated per-schema axiom flag.
- *Target [M]:* the P2 F3-only decomposition prices `repeat` at **74 % of the 8.24-FP headroom** (11.05 → 2.86), roughly flat across strata — the workhorse.
- *Cross-env [M]:* DD2D → only blameless `retrieve` (~1 %, already proof-tier) qualifies → `repeat` ≈ inert (the "improves DD2D s3" hypothesis is downgraded to *possible-but-small*, gated on §5.2). SB2D `EMPTY_SPEC` → never fires → inert.
- *`dead` audit [M]:* `dead` (`dataset.py:995`) is object-set *subsumption* over proof-demotable failures, not exact-step membership — `repeat` is a genuinely new column, not a scoping change.
- *Soundness [M]:* exact-vetoing a *culprit-bearing* (F2) step kills 263 real successes in the oracle — the `blame == ∅` gate is load-bearing.

### 4.2 `regroup` (binary, grouping certificate)

For each culprit-bearing record: recover the **seating chart** = the failed step + each culprit's **establishing step** (the last step before the failure index in the *failed candidate's* skeleton whose args mention that culprit; culprits with no establishing step contribute no condition). `regroup_one = 1` iff the candidate contains the failed step and every chart step (schema + canonicalized args). `regroup = max` over culprit-bearing records; 0 when none.

```python
def establishing(failed_skel, t_fail, culprit):
    for u in reversed(range(t_fail)):
        if culprit in args(failed_skel[u]):
            return (schema(failed_skel[u]), args(failed_skel[u]))
    return None

def regroup_one(cand, rec, pool):
    if (rec.schema, rec.args) not in steps(cand):
        return 0
    for k in rec.culprits:
        e = establishing(pool[rec.candidate_idx], rec.step_index, k)
        if e is not None and e not in steps(cand):
            return 0
    return 1
```

- *Why step identity, not effects:* the level bit lives only in the operator name (`place_tall` vs `place_short` — identical abstract effects by design); name-matching is what lets the feature see levels without new predicates or geometry.
- *Variants:* **unordered** (co-occurrence anywhere in the candidate) is primary — sound in v3 because there are no un-store/rearrange operators, so co-occurrence implies final co-residency and capacity is a set property. **Ordered** (chart steps required before the re-attempt) is the assumption-free conservative variant, behind the same flag. Record the soundness condition in `decisions.md` so the argument travels with the feature.
- *Expected cross-env behavior* `[D]`: DD2D/SB2D culprits are initial-state objects with no establishing steps → the chart reduces to the failed step, which is goal-necessary → present in ~every candidate → `regroup` ≈ constant → inert-harmless. Verify in §5.2, don't assume.

### 4.3 `regroup_frac` (graded, optional, own flag)

`max` over culprit-bearing records of the fraction of that record's chart the candidate re-assembles. Heuristic by construction (partial rebuilds are *suspect*, not certified). **Gate on two numbers before wiring in:** P2's strict-vs-graded gap, and a within-pool variance pre-check on collected data (a graded column that barely varies inside pools is decoration — the coverage lesson).

### 4.4 Plumbing conventions

Overlap vector extends `[dead, jaccard, coverage, waste]` → append `[repeat, regroup(, regroup_frac)]`. Trailing-additive; older pickles/checkpoints load with the new columns absent → zeroed (shim); flags `--repeat-feats`, `--regroup-feats`, `--regroup-frac` recorded in checkpoint config. Coverage and waste are **not modified** — they own the ordering and idle-work channels DD2D/SB2D exercise.

### 4.5 Leakage invariant

All new columns are exactly 0 at |F| = 0 (first attempt purely static). Add to the existing invariant test.

### 4.6 Conditional fixes, by probe outcome

- **P2 headroom ≈ 0 (H3):** read the pool diagnostics; if pools are split-poor, the fix is pool-side (split-diversity quota or annealed assignment costs in the prior) — separate workstream, spec only if triggered.
- **P4 nonzero (H4):** training-side isolating ablation (failure-context sampling breadth, evidence-attn capacity) — one variable at a time.
- **Rejected for now:** assigning F3 the block itself as culprit. It reroutes height evidence through coverage's ordering semantics and muddies channel attribution. Revisit only in isolation, and only if `repeat` fails to capture oracle-strict's height component.

## 5. Cross-env non-regression (DD2D / SB2D)

**5.1 Principles.** Additive + independently switchable; per-env training means an inert feature costs ≈0 in expectation but can add small-sample variance; certificates scoped to provable records avoid the known anti-signal pocket by construction rather than by hope.

**5.2 Pre-checks** (before any retrain; ~1 h per env on existing datasets). Per env, within-pool variance and label-sign of `repeat` / `regroup` (/ `frac`):

- DD2D expectation `[P]`: `repeat` live only on provable-record contexts and *positively* aligned with infeasibility (soundness); `regroup` ≈ constant.
- SB2D expectation `[P]`: both ≈ constant.
- Any live cell *anti-correlated* with labels → stop and investigate before any joint retrain.

**5.3 Retrain protocol.** Matched settings, 3 checksum-distinct seeds per arm per env. Arms: baseline recipe / +`repeat` / +`repeat`+`regroup` (/ +`frac` if gated in). One rung per feature; no cross-quoting across mismatched settings. Comparisons are **arm-vs-arm on identical data** — not vs the stale §10 caches (which predate the PointSetEncoder/atom-profile retrain).

**5.4 Gates.**

- DD2D/SB2D: no stratum-level regression (paired bootstrap; CI must not show worsening at any stratum) and ALL-FP within CI of the baseline arm. Registered optional improvement hypothesis: DD2D s3 improves via the proof-demotion mechanism (§4.1).
- v3: adaptive − static CI-clean positive, and report **harvested fraction** = (FP_static − FP_adaptive) / (FP_static − FP_oracle-strict) per stratum — the number that says how much of the available headroom the learned pathway actually captured.

## 6. Order of operations

1. P0–P4 (one day, no training) → Decision point A.
2. `dead` audit; implement `repeat` + `regroup` + invariant tests behind flags.
3. §5.2 pre-checks on DD2D/SB2D/v3 collected data.
4. v3 retrain arms; read against the P2 ceiling.
5. DD2D/SB2D retrain arms; gates.
6. `decisions.md` entry (feature definitions, soundness conditions, probe numbers); promote the §7 matrix tags `[P]`→`[M]`.

## 7. Appendix — feature-activation matrix (current beliefs)

| | DD2D | SB2D | Restock3D-v3 |
|---|---|---|---|
| decision type | removal subset + order | ordering w/ stick state | capacity assignment (groupings) |
| coverage | ● `[M]` | ● `[M]` (post-U2) | ◐ `[M]` near-constant |
| waste | ● `[M]` strongest | ◐ `[U]` | ○ `[M]` vacuous |
| repeat | ○ `[M]` inert (no `step_certificate` schema) | ○ `[M]` inert | ● `[M]` **workhorse — 97% of the ceiling, adaptive 12.18→3.13** |
| regroup | ○ `[M]` inert (gated off by `grouping_certificate`) | ○ `[M]` inert | ◐ `[M]` **~1%, DEPRECATED/off** |

Measured reading (2026-08-21): the environment-agnostic evidence vocabulary activates by each
domain's failure structure — DD2D exercises coverage/waste, **v3 exercises `repeat` (F3 certificate)**,
SB2D leans on coverage. `repeat` is the sole v3 lever (`regroup` deprecated). The gates
(`step_certificate`/`grouping_certificate`, `blame==∅`) keep each feature sound-where-it-fires and
inert elsewhere; the cross-env pre-check (§5.2) caught ungated `regroup` firing wrong-polarity on DD2D
before it could ship. No single environment validates the vocabulary; the suite is the argument.
