# SPECTRE Notebook — v3 migration

8 entries, 2026-07-26 .. 2026-07-26 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-07-26-g0-g1-instrumentation-13-68-unreproducible"></a>
## 2026-07-26 — v3 G0/G1: instrumentation is observation-only; **the published dd2d_v3 13.68 does not reproduce from on-disk artifacts**

<!--strip-->
> **id** `2026-07-26-g0-g1-instrumentation-13-68-unreproducible` · **status** amended
> · **tracks** method, env-dd2d, data
>
> ⚠️ **AMENDED** — the cause was double canonicalization in the cache builder, not
> code staleness, and the corrected figure is **14.50**. **Do not quote 13.68.**
<!--/strip-->

- **What:** SPECTRE v3 migration gates G0 (instrument the DD2D refiner, collect `dd2d_v4`)
  and G1 (v3 scaffold + equivalence oracle). Full audit numbers below are on the complete
  corpora unless stated.

- **G0 — instrumentation is observation-only [verified].** Four emission sites in
  `dd2d/refine.py` now record `FailureObservation(step_index, schema, args, culprits,
  unmoved, n_step, exhausted, budget_exhausted)`. `grasp_cfree` was refactored to
  `grasp_blocker(...) < 0` so the culprit falls out of the collision loop that already ran
  rather than costing a new query. Differential gate (replay stored dd2d_v3 candidates
  through the instrumented refiner at their stored seeds): **`label` / `steps_bound` /
  `plan_length` / `failure_action` identical on 290/290**; full dd2d env suite 123 passed.
- **The 20 s `time_budget` is binding, and `n_attempts` is contaminated by it.** Measured
  p99 elapsed 20.0046 s, max 20.0208 s. Split by budget-boundness: fast (<19 s, n=286)
  reproduces `n_attempts` **286/286**; budget-bound (n=4, 1.4%) reproduces **0/4**, and
  systematically (1912→3904, 3116→3934, 3101→3685, 1898→3935) — this box is faster than
  the one that collected v3, so more stream calls fit in the same 20 s. **`n_attempts` on
  a budget-bound candidate measures host CPU speed, not the problem.** Consequences: the
  identity gate asserts `n_attempts` only off-budget; the v3 record names it `n_total` and
  masks it where `budget_exhausted`; the `3s+1` exactness witness is unaffected (those
  candidates are nowhere near the floor, so they were already excluded from proof tier).
- **D6 answered with a mechanism.** `refine()` loops `while idx < n and not exhausted()`
  then sets `failure_action = str(plan[best_reached])` — the deepest step *reached*, not
  necessarily one tested. On a budget exit it still reports `retrieve(target)` though the
  retrieve never ran. That is the confirmed cause of all **12/18694** dd2d_v2 demotion
  violations (all one candidate, `n_attempts=2406`, i.e. 240× the straight-through floor);
  dd2d_v3 has **0/19547**. The leaking axiom is **exactness**, not locality.

- **G1 — v3 == v2.2 bit-for-bit [verified].** `model_v3` compat mode builds the v2.2
  submodules, so the deployed checkpoint loads `strict=True` (91 keys, head (256,130), no
  `prior_gate`). v3 and v2 rollouts agree on `order`, `step_dead`, attempts **and logits
  exactly**, over 20 episodes strided across all four strata.
- **The stored dd2d_v3 comparison cache — the source of the published 13.68 — does not
  reproduce from the checkpoint and code now on disk.** Recomputing all 100 test
  problems with the current code and `checkpoints_v2_evidence_ov/dd2d_v3/seed_0`:

  | | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | cached (published) | **13.68** | 0.00 | 4.60 | 26.20 | 23.92 |
  | recomputed now | **14.50** | 0.00 | 4.64 | 25.68 | 27.68 |

  Per-problem FP identical on **61/100**; identical attempt order on **55/100**. At t=0 the
  static logits correlate 0.996 with the cache but reorder the top-20 (17 inversions,
  max |Δ| 0.387). Ruled out: it is not v3 (v2 and v3 agree bit-for-bit), not dropout
  (identical in eval at `dropout_p` 0.0 vs 0.1), not device (CPU and CUDA agree), not
  nondeterminism (identical across processes under `PYTHONHASHSEED=random`), and not the
  other v3 checkpoint (`_prior_ov` is further away, max |Δ| 3.49). `model_v2.py` and
  `dataset_v2.py` are clean at HEAD and the episode pickles predate the cache.
  **Cause identified later the same day: double canonicalization in the cache builder —
  see the entry below. Not code staleness.**

<a id="2026-07-26-g7-p-v3-3-falsified"></a>
## 2026-07-26 — G7: P-v3-3 falsified — `cand_overlap` is load-bearing, and the net has already internalised the demotion rule

<!--strip-->
> **id** `2026-07-26-g7-p-v3-3-falsified` · **status** active · **tracks** method ·
> **resolves** P-v3-3
<!--/strip-->

- **What:** the S4/G7 2×2, {overlap on/off} × {demotion on/off}, on dd2d_v4 test [1-seed
  dev]. The two *training* arms already existed — G6b's record arms have exactly G7's
  configs (records ON, overlap ON/OFF, same seed/epochs/lr, verified against the stored
  cfgs) — so this gate needed **no training at all**, only the eval-time demotion axis.
  That axis is new: `deployed_rollout_v3_traced(..., apply_demotion=False)`, deliberately
  not a third `DemotionMode` (the modes say what licenses a *sound deduction*; this says
  whether to *act* on one). The proof state still advances either way, pinned by test.
- **Result — uncensored deployed FP, dd2d_v4 test, n=100:**

  | overlap | demotion | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|---|
  | ON | ON | **16.17** | 0.00 | 8.56 | 22.00 | 34.12 |
  | ON | OFF | 16.30 | 0.00 | 8.56 | 22.28 | 34.36 |
  | OFF | ON | 21.24 | 0.00 | 16.60 | 25.60 | 42.76 |
  | OFF | OFF | 23.06 | 0.00 | 16.60 | 28.56 | 47.08 |

- **P-v3-3 is FALSIFIED.** The prediction was that removing `cand_overlap` is
  performance-neutral because tag attention learns soft set-overlap. It is not:
  **−5.07 FP, CI [−8.56, −1.78]** with demotion on, **−6.76, CI [−10.48, −3.28]** with it
  off. Per R7's own escape clause, overlap is reinstated as honest features and reported.
- **Demotion is nearly free when overlap is on (0.13 FP) and worth 1.82 when it is off.**
  Read together with the row above: the net's learned `dead` column has already
  internalised the proof rule, so the outside-the-net offset finds little left to correct.
  The same holds for v2.2 — scoring the yardstick with demotion off gives **14.70 vs
  14.66**, i.e. its 26.44 at s3 is pure model quality, not the proof rule.
- **Keep both anyway, and the reason is soundness not FP.** The learned `dead` column is a
  *correlate*; the offset is a *proof*. C5 exists so that a wrong weight can never override
  a sound deduction. Paying 0.13 FP for that guarantee is the right trade, and it is now a
  measured trade rather than an assumed one.
- **Takeaway / next:** the interesting consequence is for `dead` as a **feature**, not as a
  rule — see the G8 entry.

<a id="2026-07-26-g6b-uncensoring-the-selector"></a>
## 2026-07-26 — G6b: uncensoring the selector closes the v2.2 gap; the record increment survives, but records *alone* do not

<!--strip-->
> **id** `2026-07-26-g6b-uncensoring-the-selector` · **status** active · **tracks**
> evaluation · **defines** G6b
<!--/strip-->

- **What:** G6 re-run with exactly one change — the checkpoint selector is uncensored
  (`select_budget` 30 → `None`, i.e. run to the pool cap) and reads the **whole** val split
  (`val_episodes` 50 → 100). Same data, same recipe, same 30 epochs, same 1 seed, three
  arms trained in parallel (~50 min). Scored by `spectre_score_v3.py`, which now also
  loads the v2.2 yardstick in D-8 compat mode (`--v2-arm`), so every row below comes from
  **one instrument on one set of episodes** and the comparisons are genuinely paired.
- **Result — uncensored deployed FP, dd2d_v4 test, n=100:**

  | arm | ALL | s0 | s1 | s2 | s3 | sel. epoch |
  |---|---|---|---|---|---|---|
  | G6b records + overlap | **16.17** | 0.00 | 8.56 | **22.00** | 34.12 | 12 |
  | G6b records only | 21.24 | 0.00 | 16.60 | 25.60 | 42.76 | 13 |
  | G6b no records (the bar) | 19.54 | 0.00 | **3.64** | 33.80 | 40.72 | 23 |
  | *v2.2 yardstick (reference)* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* | — |

- **The gate PASSES, more strongly than under censoring.** records+overlap vs the bar:
  **−3.37 FP, 95% CI [−6.16, −0.64]**, excludes 0 (G6 measured −2.36).
- **The v2.2 gap was the selector, and it is now closed.** records+overlap vs the yardstick
  is **+1.51 FP, CI [−2.29, +5.72] — includes 0**, i.e. indistinguishable. The *same*
  comparison for the censored-selector checkpoint is **+3.93, CI [+0.37, +7.95]**, which
  excludes 0. So under G6 v3 was genuinely worse than v2.2; under G6b it is not. The
  2026-07-26 G6 open issue is resolved, and 18.59 should not be quoted as v3's level.
- **Mechanism, measured:** the selector's dynamic range across epochs. Censored@30/n=50 the
  per-epoch `val_fp` spans ≈ [11.1, 17.5]; uncensored/n=100 it spans ≈ [17.1, 32.0] — about
  **2.5× the signal**. Censoring did not add noise, it removed range, which is why the G6
  curves looked stable and still selected badly.
- **⚠ Unexpected 1 — records *alone* no longer help.** records-only vs the bar is
  **+1.70, CI [−2.25, +5.81]** (includes 0); under the censored selector it looked like
  −1.80. Only records **with** `cand_overlap` beat the bar. So the increment attributed to
  "record tokens" in G6 is really a records×overlap interaction. This lands directly on
  **G7**, whose 2×2 was designed on the premise that `dead` is redundant with the demotion
  applied outside the net — that premise now needs testing rather than assuming.
- **⚠ Unexpected 2 — evidence helps s2/s3 and *hurts* s1.** The no-records bar is the best
  arm at s1 (**3.64** vs 8.56) while being far worse at s2 (33.80 vs 22.00) and s3 (40.72
  vs 34.12). This is the same shape as the v2.2-era "evidence harms s1" problem that
  2026-07-19 fixed by routing proof-tier facts out of the learned pathway; it has
  reappeared in v3's record-token pathway. At s1 the first attempt often succeeds, so the
  failure set is mostly noise — plausible, but not yet demonstrated.
- **Takeaway / next:** G6b unblocks G7. Carry two questions into it: does `jaccard` (not
  `dead`) explain the interaction above, and can the s1 regression be removed by gating
  record consumption on |F| rather than by dropping features. v3 currently *matches* v2.2
  rather than beating it — acceptable for a consolidation gate whose claim is "same
  performance on less bespoke machinery", but it is not yet a win.

<a id="2026-07-26-g6-record-tokens"></a>
## 2026-07-26 — G6: record tokens beat the fact pathway, but the censored val selector cost more than it saved

<!--strip-->
> **id** `2026-07-26-g6-record-tokens` · **status** retracted · **tracks** method,
> evaluation · **defines** G6 · **superseded by**
> 2026-07-26-g6b-uncensoring-the-selector
>
> ⚠️ **RETRACTED** — these arm levels (18.59 / 19.15 / 20.95) came from a selector
> censored at 30 attempts, below the tail that separates the models. Corrected to
> 16.17 / 21.24 / 19.54 by G6b. The "−3.37 record increment" was `cand_overlap`, not
> records. **Do not quote either figure.**
<!--/strip-->

> ⚠️ **Superseded by the G6b entry above (same day).** The arm *levels* here (18.59 / 19.15 /
> 20.95) were produced by the censored selector diagnosed in this very entry and **must not be
> quoted** — G6b's 16.17 / 21.24 / 19.54 replace them. Two conclusions here also do not
> survive: the v2.2 gap is *not* real (it was the selector), and "records-only is −1.80" flips
> to +1.70 (n.s.). What stands: the record increment passes, and the diagnosis of the selector.

- **What:** three v3 arms trained on dd2d_v4 [1-seed dev], scored **uncensored** on the
  test split with paired-bootstrap CIs over problems (`spectre_score_v3.py`). All three
  hold `cand_overlap` out where stated so the record increment is not measured against a
  bar carrying the same set-overlap signal. Trained **in parallel** (16.9 min for three).
- **Result — uncensored deployed FP, dd2d_v4 test, n=100:**

  | arm | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | v3 records + overlap | 18.59 | 0.00 | 4.52 | 30.96 | 38.88 |
  | v3 records only | 19.15 | 0.00 | 13.12 | 25.52 | 37.96 |
  | v3 no records (the bar) | 20.95 | 0.00 | 4.80 | 35.48 | 43.52 |
  | *v2.2 yardstick (reference)* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

- **The record increment PASSES.** records+overlap vs the no-records bar:
  **−2.36 FP, 95% CI [−4.42, −0.48], excludes 0.** Records-only is −1.80 with CI
  [−5.38, +1.79] (includes 0), i.e. the overlap features and the record tokens are worth
  having together. So one canonical record + role-separated tokens is at least as good as
  the five bespoke fact types it replaced, on strictly less bespoke machinery.
- **⚠ But every v3 arm underperforms the v2.2 yardstick (18.59 vs 14.66), and the cause is
  the selector I introduced, not the representation.** The no-records bar (20.95) is
  already worse than v2.2, so the regression lives in the shared v3 *training* path.
  Diagnosis: scoring the v2.2 checkpoint on the **same censored val metric v3 selects on**
  gives **11.12**, against v3's selected 11.40 / 11.79 / 13.01 — indistinguishable. Yet on
  uncensored test those same models differ by 4+ FP. **Censoring the selector at 30
  attempts truncates exactly the s2/s3 tail (FP 30–40+) where the models actually differ**,
  so the selection signal is saturated. Training curves are stable and plateau by ~epoch 12
  with sensible epoch choice, which rules out selector *noise*; the problem is selector
  *blindness*.
- **Takeaway / next:** the censoring was a speed trade (91 s/epoch uncensored vs 35 s) made
  before parallel training existed. With `spectre_sweep.py` running arms concurrently that
  trade is no longer worth taking — an uncensored 100-episode selector at ~91 s/epoch is
  ~45 min for a whole 3-arm sweep. **Re-run G6 with the budget removed before treating
  18.59 as v3's level**; the increment result stands either way, since both arms shared the
  handicap. Cheap alternatives if cost returns: run the selector every K epochs, or keep it
  uncensored but on a strided subsample.

<a id="2026-07-26-g5-one-failurerecord"></a>
## 2026-07-26 — G5: one FailureRecord + a declarative certificate rule replaces the five fact types, with a measurable soundness win

<!--strip-->
> **id** `2026-07-26-g5-one-failurerecord` · **status** active · **tracks** method ·
> **defines** G5
<!--/strip-->

- **What:** `failure_record.py` (one canonical record; instrumented on dd2d_v4, backfilled
  on older collections) + `proof_demotion_v3.py` (the generic rule: demote σ′ if it issues
  the **same query on the same args** at some j′ with `U(σ′,j′) ⊇ U(σ,j)`, and the domain
  declares that query monotone + local + exact). Two modes: `permissive` = v2.2 semantics
  (for the equivalence check), `strict` = requires positive evidence the query ran.
- **Reduction [verified]:** permissive mode reproduces v2.2's demotions
  **candidate-for-candidate** on dd2d_v3, checked at four points along the growing failure
  set for 12 episodes. The generic rule is only allowed to differ *after* it agrees.
- **Deployed cost: none.** Re-running the deployed v2.2 checkpoint on dd2d_v4 under the v3
  rule gives **identical per-problem FP on 100/100** — ALL 14.66, s1 6.20, s2 26.00,
  s3 26.44, exactly the yardstick. The generic rule is a drop-in for the DD2D-specific one.
- **Soundness win, and the return on instrumentation:**

  | collection | mode | demoted | demoted-but-feasible |
  |---|---|---|---|
  | dd2d_v2 | permissive (= v2.2) | 4183 | **12** / 3289 |
  | dd2d_v2 | strict | 3918 | **0** |
  | dd2d_v3 | strict | 3784 | 0 |
  | **dd2d_v4** (instrumented) | permissive | 4029 | 0 |
  | **dd2d_v4** (instrumented) | strict | **4029** | 0 |

  Strict mode eliminates all 12 dd2d_v2 unsound demotions — the ones v2.2 made by trusting
  a `retrieve` failure that had actually stopped on the wall-clock budget. On *backfilled*
  data that soundness costs ~6% of demotions, because exactness has to be derived from a
  conservative witness (an attempt whose sampler calls equal the minimum possible cannot
  have re-sampled, so its reported query really ran — declared as
  `DomainSpec.min_calls_per_schema`, one integer per operator). **On instrumented data it
  costs nothing**: strict and permissive demote identically, because the refiner reports
  `exhausted` directly. That is the concrete payoff of the dd2d_v4 re-collection.
- **Takeaway / next:** the DD2D-specific demotion is gone with no performance cost and a
  strictly sounder rule. The G1 oracle now runs in `permissive` mode — v2.2's semantics —
  so an intended improvement can't be mistaken for a regression. Next: G6, replacing the
  five fact types in the *learned* pathway with role-separated record tokens.

<a id="2026-07-26-d2-advantage-is-length-calibration"></a>
## 2026-07-26 — D2: SPECTRE v2.2's entire advantage is cross-length calibration; within a length it is at best planner-order

<!--strip-->
> **id** `2026-07-26-d2-advantage-is-length-calibration` · **status** active ·
> **tracks** method, evaluation · **defines** D2
<!--/strip-->

- **What:** the pre-registered s2 fork (`experiments/spectre/spectre_d2_s2.py`), on dd2d_v4
  test with the deployed v2.2 checkpoint [1-seed dev]. Every row is a rollout over the same
  pool restriction, so nothing mixes static-vs-adaptive with restricted-vs-full. The
  **length oracle** restricts the attemptable pool to minimum-length candidates — it uses
  the stratum, so it is a diagnostic bound and never a model input.
- **Result — mean rollout FP:**

  | stratum | min-len candidates | v2.2 full | v2.2 length-oracle | astar full | astar length-oracle |
  |---|---|---|---|---|---|
  | s1 | 10 / 200 (5%) | 6.20 | 5.20 | 2.24 | **1.24** |
  | s2 | 97 / 200 (48%) | 26.00 | **33.92** | 17.08 | **5.80** |
  | s3 | 92 / 200 (46%) | 26.44 | 30.76 | 118.76 | **30.36** |

- **The fork answers "within-length" at every stratum, not just s2.** Handing the model the
  correct plan length makes it *worse* at s2 (26.00 → 33.92) and s3 (26.44 → 30.76), while
  the same restriction makes plain planner order dramatically better (s2 17.08 → 5.80).
- **Where v2.2's win actually comes from.** At s3 the model beats astar 4.5× on the full
  pool (26.44 vs 118.76) but *ties* it under the length oracle (30.76 vs 30.36). So the
  entire s3 advantage is **length calibration** — knowing to prefer long plans — and none
  of it is within-length discrimination. At s2 the model is materially *worse* than index
  order within the correct length (33.92 vs 5.80). Roughly, 33.92 failures among 97
  candidates is what random ordering gives with ~2 feasible: **the ranker is effectively random
  within a length.** This sits uneasily with the v2.2 claim that the within-length PL loss
  "forces the geometry signal at every stratum"; its within-length AUROC (0.585–0.673) does
  not translate into ordering that beats plain enumeration order.
- **Consequences for the plan.** (i) **G8's premise is weakened as pre-registered**:
  necessity conditioning mainly improves *length* calibration, which is the one thing v2.2
  already does well, so P-v3-1 (s2 ≤ 17.08) will only be reachable through the *within*-length
  component of the necessity features — `coverage`/`waste` differing between same-size
  subsets — and that in turn depends on the necessity head predicting p_i accurately from
  geometry. Worth testing the head's per-object AUROC *before* building the conditioning.
  (ii) **A new and cheaper lead:** the planner's enumeration order is a strong within-length
  signal (5.80 at s2) that the deployed model **cannot see at all** — R1 removed the prior,
  and the deployed dd2d_v3/v4 config already had `use_prior=False`. The prior was dropped
  wholesale because its *short-first* column collapsed s3, but that is column 1; column 0
  (`−index/K`, enumeration order) is a different signal and was never separately ablated.
- **Takeaway / next:** ping the user — the fork's answer changes what G8 is for. Cheap
  follow-ups, in order: (a) necessity-head accuracy alone, (b) an **index-only** prior
  (column 0 without short-first), which the diagnostic directly motivates.

<a id="2026-07-26-canonicalize-not-idempotent"></a>
## 2026-07-26 — `canonicalize_episode` is not idempotent, and the comparison cache applied it twice

<!--strip-->
> **id** `2026-07-26-canonicalize-not-idempotent` · **status** active · **tracks**
> data, tooling
<!--/strip-->

- **What:** chasing why a *freshly built* dd2d_v4 cache disagreed with a direct rollout on
  the same checkpoint (s2 23.92 vs 26.00), after ruling out staleness, dropout, device and
  cross-process nondeterminism (all verified identical, on CPU and CUDA).
- **Result:** `canonicalize_episode(canonicalize_episode(ep)) != canonicalize_episode(ep)`.
  A second pass permutes object names differently (`item_10` → `item_2`, …). Scene poses
  are untouched, but the object→**tag** binding changes — and tags are the join key the
  entire v2.2/v3 representation runs on. `precompute_dd2d_cache` sourced episodes from
  `eda.load_split_episodes`, which canonicalizes on load, and `build_v2_example`
  canonicalizes again, so **every cached SPECTRE number was computed on doubly-
  canonicalized episodes** while training loads raw and canonicalizes once.
- **Impact (dd2d_v4 test, deployed v2.2, n=100):**

  | | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | single-canon (matches training) | **14.66** | 0.00 | 6.20 | **26.00** | **26.44** |
  | double-canon (what the cache did) | 14.85 | 0.00 | 6.16 | 23.92 | 29.32 |

  Per-problem FP differs on **35/100**. The aggregate barely moves, but per-stratum swings
  2–3 FP in *both* directions.
- **Takeaway / next:** fixed in `precompute_dd2d_cache` (a `_RawSplit` view feeds raw
  episodes to the tensorizers, so evaluation matches training); the dd2d_v4 yardstick was
  regenerated and now agrees with a direct rollout. Two things worth keeping: (i) **every
  published SPECTRE comparison number — dd2d_v2's 17.09, dd2d_v3's 13.68 and its
  per-stratum row — was produced under double canonicalization** and would move by ~0.2
  overall / 2–3 per stratum if regenerated; (ii) a pure *relabeling* moving s2/s3 by 2–3 FP
  is itself a measurement of how tag-permutation-invariant the deployed model actually is.
  Per-epoch tag permutation (P-A) was supposed to buy that invariance; it is evidently only
  approximate, which is a legitimate robustness finding rather than only a bug.
  `eda.load_split_episodes` still canonicalizes for the EDA baselines that key on canonical
  skeletons — only the model-cache paths changed.
- **`dd2d_v4` collected: 400/100/100, exactly 100 (train) / 25 (val,test) per stratum**,
  125 min wall-clock (76.1 + 21.8 + 27.4, 14 workers). Records carry culprits, per-step
  effort, exhausted-vs-budget, backjump count, `elapsed` and the generator arguments;
  600 episodes converted (`dd2d_convert_v3`), vocab OOV-clean on val/test.
- **⚠ The v4-vs-v3 identity gate FAILS, and the cause is DD2D's generator, not the
  instrumentation.** `generate_dd2d_problem` is deterministic within a process but
  **`PYTHONHASHSEED`-dependent across processes**: with everything else fixed, seed 500039
  gives target pose (23.696, 9.206) under `PYTHONHASHSEED=0` and (14.981, 21.960) under
  `=1`, each reproducibly. Same seed can even yield a different `n_items` (9 vs 10 for
  750039). So every DD2D collection — v2, v3, v4 — is a valid sample but not a reproducible
  one. Divergence over the 597 matched problems: **99.2% identical scenes, 94.6% identical
  labels, 90.3% identical plans, 86.9% fully identical**; 98/119400 candidate labels differ
  (0.08%), plus 3 boundary problems where a different seed was kept. Decision (user):
  accept v4 as collected and document, rather than fix + re-collect. Likely leak site:
  `enumerate.py`'s `present = set(scene.item_names()) - {scene.target}`.
- **G3 — dd2d_v4 yardstick [1-seed dev].** Deployed v2.2 (`--evidence --use-overlap`,
  no prior) retrained on dd2d_v4; compare cache built (astar + spectre2). Mean rollout FP:

  | method | ALL | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | astar-dist | 34.52 | 0.00 | 2.24 | **17.08** | 118.76 |
  | SPECTREv2-adaptive | **14.66** | 0.00 | 6.20 | 26.00 | 26.44 |
  | SPECTREv2-static | 20.08 | 0.00 | 8.60 | 30.36 | 41.36 |

  Cross-checks: v4's 14.66 sits beside dd2d_v3's *recomputed* 14.50, and astar is 34.52
  vs v3's 34.65 — so the two collections agree at method level, as the 0.08% label
  divergence predicts. **The s2 gap survives on v4** (26.00 vs astar 17.08), so G8's
  target is real here and not an artifact of v3. PIGINet has no v4 row yet (it trains on
  the native JSON with its own CLIP cache); the table warns rather than silently omitting.
  A second arm, `--evidence` **without** overlap (relrank 1.428 vs 1.374), is trained and
  held for G6's evidence-increment bar, which must be measured with `cand_overlap` out of
  both arms.
- **Seed policy → 1-seed for development** (user directive), 3 seeds reserved for the
  final paper evaluation. Consequence for the gates: "no stratum regresses beyond seed
  noise" is unmeasurable at 1 seed, so the working acceptance rule becomes a **paired
  bootstrap over problems** (`eda.bootstrap_mean_difference`, the same instrument the
  P1/P4/P5 gates used) — comparing two methods on the *same* problems, which is both
  more powerful than a seed spread and available now.
- **D4 — necessity labeller built; `d_hat == stratum` exactly, but s3 pool coverage is far
  worse than assumed.** `necessity.py`: p_i = fraction of **minimum-size feasible
  manipulated sets** containing object i, deduped by subset, any-ordering-feasible, goal
  objects excluded. On dd2d_v3 train, `max |d_hat - stratum| = 0.0` over 400 episodes.
  Mean distinct minimal solutions rises with stratum (1.00 / 1.19 / 1.34 / 1.51 at
  s0-s3) — i.e. soft-vs-binary matters most exactly where s2/s3 live.
  **Correction to the pre-registered risk:** subset-lattice coverage by size on dd2d_v4
  train is 1.000 / 1.000 / 1.000 for |S| = 0,1,2 but only **0.171 mean, 0.045 min** for
  |S| = 3 (the plan's register guessed 66-77%). With k=200 candidates and C(12,3)=220 that
  is expected, but it means the *set* of minimal solutions seen at s3 is a ~17% sample
  drawn in planner-preference order, so p_i at s3 is biased toward planner-preferred
  solutions. `min_size` itself is unbiased (the pool contains a true-minimal subset on
  400/400, matching the collector's own `min_feasible_subset`), so `d_hat` is safe; it is
  the per-object split that is noisy. Report this beside any G8 number.
- **G2 — the domain contract replaces 11 DD2D literals, oracle still exact [verified].**
  `domain.py` derives a candidate's manipulated set, its plan length, and whether its failure
  licenses demotion from the operator schema + a three-line axiom declaration. Licensed by
  whole-corpus identity, not spot checks: `args(σ) \ goal_objects` == the `place-buffer`
  filter on **120000/120000** skeletons, and `len(operator_seq) == 2·|staged|+1` on the same
  120000. `dataset_v3` was then rewritten off the contract (no delegation to v2), and the
  **G1 oracle stayed bit-identical** — which is the whole point of having gated it first.
  Spectre suite 385 green.
- **Takeaway / next:** the v3 equivalence oracle was re-pointed from the stale cache to a
  **live v2.2 run** — strictly stronger (exact equality, no 4-dp tolerance) and it cannot
  rot. The staleness is recorded as a skipping regression test rather than left to be
  rediscovered as a v3 regression. Nothing downstream depends on 13.68 (the v3 yardstick is
  a fresh 3-seed v2.2 run on dd2d_v4), but **13.68 should not be quoted again until the
  cache is rebuilt with `--force`**, which will settle whether the corrected figure is
  14.50. Open for the user: whether to rebuild and restate, or to treat 13.68 as
  provisional pending the dd2d_v4 numbers.

<a id="2026-07-26-vlmplan-scale-comparison"></a>
## 2026-07-26 — VLMPlan scale comparison (Qwen3-VL 8B vs 32B, same family): scale buys geometry, and makes the always-act bias *worse*

<!--strip-->
> **id** `2026-07-26-vlmplan-scale-comparison` · **status** active · **tracks**
> baselines
<!--/strip-->

- **What:** Second VLMPlan arm — `qwen3-vl-32b-instruct` (Q4_K_M, local, 32768 ctx), dd2d_v3
  test, all 100 problems, **loop constants and prompt identical to the 8B arm**, so the pair
  is a clean scale comparison (same family, both Instruct/non-reasoning). ~8.5 h generation
  (307 s/problem, 60 tok/s vs the 8B's 205) + ~2 h scoring. Gate: label agreement **0.983**.
  Cache `compare_cache/vlmplan_qwen32b/`.
- **Result — mean rollout FP, dd2d_v3 test (n=100), lower better:**

  | method | s0 | s1 | s2 | s3 | ALL |
  |---|---|---|---|---|---|
  | astar-dist | **0.00** | 2.24 | 17.08 | 119.28 | 34.65 |
  | PIGINet | 0.04 | 4.92 | 18.60 | 51.12 | 18.67 |
  | SPECTRE-adaptive | 0.00 | 9.20 | 29.52 | 53.00 | 22.93 |
  | SPECTRE-static | 0.00 | 27.44 | 27.20 | 46.36 | 25.25 |
  | SPECTREv2-adaptive | 0.00 | 4.60 | 26.20 | **23.92** | **13.68** |
  | SPECTREv2-static | 0.00 | 4.44 | 32.64 | 39.40 | 19.12 |
  | VLMPlan-8B | 4.24 | **2.88** | 16.04 | 96.28 | 29.86 |
  | **VLMPlan-32B** | 6.76 | 5.04 | **13.16** | 69.24 | 23.55 |

- **Scale helps overall (−21%, 29.86 → 23.55) but the effect is opposite at the two ends.**
  Where the task is geometric, 32B is much better: **s3 96.28 → 69.24** (−28%) and **s2
  16.04 → 13.16**, the latter making it the **best method in the table at s2** (vs astar
  17.08, SPECTREv2-adaptive 26.20). Where the right answer is restraint, it is *worse*:
  **s0 4.24 → 6.76** and s1 2.88 → 5.04.
- **The always-act bias is not a capacity limit — it scales the wrong way.** s0 means the
  target is already graspable, so `retrieve` alone (pool index 0) is correct and every other
  method scores 0.00. The bigger model still never proposes it first, and costs *more*
  because it proposes more and stages more items per attempt. Better format compliance
  (parse 62% → **85%**), 2.4× the plan yield (19.8 → **47.2** plans/problem) and zero
  truncation bought nothing at s0. This is the failure
  `vlmplan_dd2d_implementation_plan.md` §8 predicted; the descoped probe would have caught it.
- **Generation quality, 8B → 32B:** plans/problem 19.8 → 47.2; parse 62% → 85%; duplicates
  39% → 45%; truncated rounds 15 → **0**; first success found by the model itself
  **68/100 → 87/100**; censored 1 → 2; off-pool attempts 2.4 → 3.1.
- **Duplicates remain the ceiling, not the budget.** Both arms end by stalling, and the
  larger model's duplicate rate is *higher* (45%). Extra scale produces more plans, not more
  distinct hypotheses — so raising `max_rounds` is not the lever.
- **Takeaway / next:** VLMPlan-32B beats the non-learned planner order (23.55 vs 34.65) and
  SPECTRE v1 (22.93/25.25), and stays behind PIGINet (18.67) and SPECTREv2-adaptive (13.68).
  The headline for the paper is the *shape*: a zero-shot VLM is competitive-to-best in the
  middle strata and fails at both extremes, and scaling moves those two failures in opposite
  directions. **1-seed dev, single run per arm.** Cheapest next win is the s0 do-nothing case.

