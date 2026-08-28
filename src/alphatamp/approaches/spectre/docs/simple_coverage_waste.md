# Simple (legacy) coverage / waste

A reference for the **simple** coverage/waste evidence features — what they are, why they
exist, their exact mathematical definition, and how they are implemented. This is a
*selectable option* (`--legacy-coverage`); the **default** is the richer
[unified](archive/unified_culprits_coverage_waste.md) definition. The two are a
value-swap inside the same two `cand_overlap` columns, so a model trained with either
loads and scores under the same architecture.

- **Deployed default:** unified (`unified_evidence.py`), since 2026-07-31.
- **This option:** simple/legacy, re-added 2026-08-27 (ADR
  [`decisions/07` 2026-08-27](decisions/07-stickbutton2d.md#2026-08-28-simple-coverage-waste-re-added-option-compare)),
  used by `experiments/spectre/compare_methods_simple.py`.
- **Results** under this option:
  [`notebook/07` 2026-08-28](notebook/07-stickbutton2d.md#2026-08-28-compare-methods-simple-simple-coverage-waste).

---

## 1. What they are

SPECTRE ranks a pool of candidate skeletons and re-ranks it *during* refinement, after
each failure. Coverage and waste are two per-candidate scalars, appended to the
`cand_overlap` feature block, that turn the **observed failures so far** into a signal
about which remaining candidate to try next. They are computed only from information the
refiner already produced — no extra probes, no predicted head — so they are purely
*observational* (they respect the observation-only invariant).

Each refinement failure of a candidate reports, as a side effect, the set of objects that
**caused a collision** (`FailureRecord.culprits` — class-1 evidence). As attempts roll in
during an episode, the union of those culprits is the running answer to *"which objects
have been seen blocking, in this problem?"* Coverage and waste score each surviving
candidate against that running culprit set.

---

## 2. Reasoning

The intuition is **precision / recall over the objects blamed for failures.**

- **Coverage ≈ recall.** Of the objects observed blocking so far, what fraction does this
  candidate manipulate? A candidate that stages exactly the blockers scores high.
- **Waste ≈ 1 − precision.** Of the objects this candidate moves, what fraction has never
  been seen blocking anything? A candidate that busies itself with irrelevant objects
  scores high (bad).

**Why waste is often the more useful of the two.** Coverage rewards *covering* the culprit
set, but the culprit set accumulates noise: an object can be blamed by a failure for
reasons that do not recur, so as junk enters the culprit pool, coverage rewards covering
junk. Waste is anchored to the candidate's own footprint (its denominator is per-candidate),
so it stays discriminating even when the culprit pool is imperfect. Empirically, on DD2D
waste carries the signal and coverage adds little; the reverse holds on other domains — see
§6 and the [unified doc](archive/unified_culprits_coverage_waste.md) §10.

**Worked intuition.** Target `T`, real blockers `{A, B}`, irrelevant objects `{C, D}`.
Attempt 0 is static and picks a plan staging `{C}`; `C` stages fine but `retrieve(T)` fails
because every grasp collides with `A` or `B`, so the culprit pool becomes `{A, B}`. Across
the surviving pool:

| candidate stages | coverage `|S∩K|/|K|` | waste `|S\K|/|S|` | reading |
|---|---|---|---|
| `{A, B}` | 2/2 = **1.0** | 0/2 = **0.0** | responds to exactly the evidence |
| `{A}` | 1/2 = **0.5** | 0/1 = **0.0** | partial cover, no waste |
| `{C, D}` | 0/2 = **0.0** | 2/2 = **1.0** | ignores the culprits, all waste |
| `{A, B, D}` | 2/2 = **1.0** | 1/3 = **0.33** | covers, plus one wasted object |

The learned ranker consumes these as features (not as a hand-coded rule), so it decides how
to weight them; a failed *non-learned* re-ranking probe on either feature is not grounds to
drop it (it tests monotone usability, not information content).

**Why keep it as an option at all.** The unified definition is stronger but harder to
explain — it derives "discretionary work" from each candidate's own causal structure. The
simple definition is a one-line formula over an observed set. It is the honest, legible
baseline for *"how much of the win is the feature engineering?"* On DD2D it costs ~1.8 FP
versus unified but still dominates every non-SPECTRE method (§6).

---

## 3. Mathematical definition

Fix an episode with candidate pool `{c₁, …, c_K}`, goal-role object set `G_obj`, and a
**failure context** `F` = the set of candidate indices whose refinement has been observed
to fail so far in the rollout.

**Manipulated set** of a candidate `c` (its footprint), excluding goal objects because they
appear in every candidate:

$$ S(c) \;=\; \Big(\bigcup_{\text{op} \in c}\ \text{args(op)}\Big)\ \setminus\ G_\text{obj} $$

**Culprit pool** — the union of class-1 collision culprits over every failed candidate in
the context:

$$ K \;=\; \bigcup_{f \in F}\ \bigcup_{r \,\in\, \text{records}(f)} \text{culprits}(r) $$

where `records(f)` are the failure records of candidate `f` and `culprits(r)` are the
objects a validity check (collision, bounds) named when it rejected the sample.

**The two features:**

$$ \text{coverage}(c) \;=\; \frac{|S(c) \cap K|}{|K|}, \qquad
   \text{waste}(c) \;=\; \frac{|S(c) \setminus K|}{|S(c)|} $$

**Conventions (both load-bearing):**

- `0/0 := 0`. Empty `K` ⇒ coverage 0; empty `S(c)` ⇒ waste 0.
- **Leakage invariant:** when `F = ∅` (no failure observed yet), both features are defined
  as exactly `0`. The first attempt is therefore purely static, and the signal only accrues
  as the rollout observes failures. A leakage audit over `|F| = 0` returns zero non-zero
  values.
- `K` uses **class-1 culprits only** (`r.culprits`), never class-2 deviation blame
  (`r.dev_blame`). This is deliberate — see §5.

---

## 4. Per-environment behaviour

Because `K` is built from class-1 culprits only, the feature's usefulness depends on whether
the environment *produces* class-1 culprits:

- **DD2D (drawer packing):** collision-rejected grasps name the blocking objects, so `K` is
  populated and both coverage and waste are **live, non-constant** signals (measured:
  coverage ∈ {0, 0.2, 0.4, 0.6}, waste ∈ {0, ⅓, ½, ⅔, 1} on a real s3 pool). This is the
  historical DD2D signal (the pre-2026-07-31 deployed feature gave DD2D 7.44 FP).

- **StickButton2D:** kinder's collision check returns a bool, so **every** SB2D failure is a
  class-2 *deviation* (`dev_blame`), and there are **no class-1 culprits at all**. Hence
  `K ≡ ∅`, so on SB2D:
  - `coverage(c) ≡ 0` for every candidate, and
  - `waste(c) ≡ 1` for every candidate with a non-empty footprint.
  Both are **constant across the pool**, therefore **ranking-inert** (a constant column
  cannot reorder candidates). This is exactly the blindness/anti-signal that motivated the
  unified definition — and it is left as-is here on purpose: rather than re-engineer the
  simple formula for SB2D, SB2D's adaptive signal is carried by **`repeat`** (§5).

---

## 5. Carrying SB2D with `repeat` (isolated)

Since the simple coverage/waste is inert on SB2D, `compare_methods_simple.py` turns on the
**`repeat`** feature to carry SB2D. `repeat` is the F3 exact-step certificate: a candidate
that contains the exact failed step of a *blameless, exhausted* failure of a
`step_certificate`-declared schema is vetoed (as a learned column). It is load-bearing on
Restock3D-v3 and **inert on DD2D/SB2D by default**, because those domains declare no
`step_certificate`.

To make it fire on SB2D, `compare_methods_simple.py` uses a dedicated domain spec
**`_SB2D_REPEAT`** (`domain.py`) — a clone of `_SB2D` that declares `step_certificate=True`
on the four press schemas — routed **only** to the env_variant `stickbutton2d_v1_simple`.
The deployed `_SB2D` / `stickbutton2d_v1` is untouched and stays byte-reproducible.

> ⚠️ **Soundness caveat.** This re-declaration resurrects a probe that was *retired* on
> 2026-08-26 as **unsound**: an SB2D press failure is context-dependent ("this button
> cannot be pressed *from here*"), which the bare `(schema, args)` step key does not encode,
> so the veto over-fires (~10.9% of *feasible* SB2D candidates flagged). It works only
> because `repeat` is a *learned* column the model can down-weight, not a hard prune. A
> 1-seed probe measured SB2D `+repeat` at −0.79 FP; `compare_methods_simple.py` reproduces
> −0.72 at 3 seeds (§6). Read it as "a learned hint that helps," not a certificate.

`proof_tier()` stays False on `_SB2D_REPEAT` (no `monotone`/`local`/`exact`), so
`dead`/demotion/token-holdout are byte-unchanged.

---

## 6. Results (summary)

`compare_methods_simple.py`, 3 seeds, test n=100 (full tables + caveats:
[`notebook/07` 2026-08-28](notebook/07-stickbutton2d.md#2026-08-28-compare-methods-simple-simple-coverage-waste)):

- **DD2D** — SPECTRE-adaptive **8.23 ± 1.65**, first by far (PIGINet 17.27, LAZY 23.26,
  astar 34.52). ~1.8 FP behind the deployed unified 6.42, but still dominant. §4 ablation:
  the scalars carry the bulk (Δ −9.78 vs static); records a modest positive residual.
- **SB2D** — SPECTRE-adaptive **1.78 ± 0.29**, best/tied (LAZY 1.85, PIGINet 2.02). The
  `+scalars` arm *is* the repeat arm (simple coverage/waste inert), and gives Δ **−0.72** vs
  static — `repeat` carries SB2D. **Records *hurt* SB2D** (full 1.78 > scalars 1.50, W3
  interference), so on SB2D the deployed `full` arm is not the best.

Read together: the abstract-representation advantage is **not** an artifact of the unified
feature-engineering — the simple, legible feature keeps SPECTRE ahead of every baseline.

---

## 7. Implementation

The switch is a value-swap inside the existing two coverage columns — same tensor shape
(`cand_overlap = [dead, jaccard, coverage, waste (, repeat, regroup)]`, base coverage width
`N_OVERLAP_COV = 4`), so unified and legacy checkpoints load interchangeably (`strict=True`).

**Flag.** `TrainConfig.unified_coverage: bool = True` (`train.py`); CLI `--legacy-coverage`
sets it `False` (`action="store_false"`). Persisted into the checkpoint via `asdict(cfg)`.

**Data path** (`dataset.py::build_example`, param `unified_coverage: bool = True`):

```python
# setup (once per example, when a failure context exists):
if want_cov and unified_coverage:
    ...                                   # unified: scene_filters + records_from_failure_records
elif want_cov:
    culprits = frozenset(                 # simple: the class-1 culprit union K
        o
        for f in ctx
        for r in records_for_candidate(canon, f, spec)
        for o in r.culprits
    )

# per-candidate row (si == S(c) == subsets[i] == spec.manipulated(skel, goal_objs)):
if want_cov and unified_coverage:
    _cov, _wst = coverage_and_waste(...)  # unified
    ...
elif want_cov:
    row += [
        len(si & culprits) / max(len(culprits), 1) if want_coverage else 0.0,  # coverage
        len(si - culprits) / max(len(si), 1)       if want_waste    else 0.0,  # waste
    ]
```

`S(c)` is `spec.manipulated(skeleton, goal_objs)` = `args(c) \ goal_objects`
(`domain.py::manipulated`); `culprits` reads `FailureRecord.culprits` via
`records_for_candidate` on the **canonicalized** episode (so object names live in the same
tag namespace as `S(c)`). `coverage_mode` (`both`/`coverage`/`waste`) zeroes the unwanted
column for both definitions.

**Deploy path** (`inference.py`):

- `load_checkpoint` emits `"unified_coverage": bool(cfg.get("unified_coverage", True))` in
  the deploy-kwargs. **The default is `True`** — every checkpoint trained before the flag
  existed is unified and carries no key, so a `False` default would silently re-score all of
  them under the simple formula. *(Standing invariant.)*
- `deployed_rollout_traced` threads `unified_coverage` into `build_example`; callers splat
  the deploy-kwargs, so nothing else changes.

**Threading sites** (all pass `unified_coverage` through): `train.py`
(`SpectreDataset.__getitem__`, `deployed_val_fp`, the `train_v3` deployed eval, argparse,
`main()` config build); `inference.py` (`load_checkpoint`, `deployed_rollout_traced`);
`dataset.py` (`build_example`).

**SB2D repeat isolation** (`domain.py`): `_SB2D_REPEAT` = `_SB2D` +
`QueryAxioms(step_certificate=True)` on the four press schemas, mapped only from
`stickbutton2d_v1_simple`; `dd2d_v4_simple → _DD2D` (repeat inert). The `repeat` gate itself
lives in `dataset.py` (`spec.axioms_for(r.schema).step_certificate ∧ r.proves_failure() ∧
blame_empty`).

**Comparison wiring.** `precompute_dd2d_cache.py` onboards `dd2d_v4_simple` /
`stickbutton2d_v1_simple` via `_SPECTRE_ONLY_VARIANTS` (SPECTRE-only, no native
v2/PIGINet row); `_V3_ARM_OVERRIDES → checkpoints_spectre_atoms_simple_full`, iso arm →
`checkpoints_spectre_norec_atoms_simple_scalars`. `compare_envs.py` `DD2D_SIMPLE` /
`SB2D_SIMPLE` graft every baseline + the definition-invariant static/+records/+recjac
ablation arms from the parent cache via `legacy_only`; only the coverage-bearing
`+scalars`/`full` SPECTRE arms are native. Training driver:
`experiments/spectre/refresh_dd2d_sb2d_train_simple.sh` (retrains only those two arms per
env, warm-started from the deployed static trunks).

**Tests.** `tests/approaches/spectre/test_coverage_mode.py` (the `unified_coverage`
deploy-kwarg round-trip) and `test_simple_coverage_repeat.py` (the legacy definition is
distinct on DD2D; the SB2D repeat gating + simple-coverage inertness).

---

## 8. Relationship to the unified definition

The unified definition ([`archive/unified_culprits_coverage_waste.md`](archive/unified_culprits_coverage_waste.md))
generalizes both features from the operator schema alone: coverage becomes recall over each
failure's "story" (discharge the culprit *before re-entering* the situation that named it,
with a state-entailment test for class-2 hazards), and waste becomes precision over
*causally unexplained* work (steps the abstraction's own backward-relevance pass cannot
justify) rather than "touches a non-goal object." That richer denominator is what makes
unified work on tool-using domains like SB2D where the simple `S(c) = args \ goal_objects`
denominator is degenerate. The simple definition here is the pre-unification formula,
retained as a legible, weaker option.
