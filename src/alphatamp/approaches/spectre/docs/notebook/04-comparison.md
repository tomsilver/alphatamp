# SPECTRE Notebook — Method comparison and VLMPlan

10 entries, 2026-07-23 .. 2026-07-25 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-07-25-vlmplan-8b-test-split"></a>
## 2026-07-25 — VLMPlan-8B on the dd2d_v3 **test** split: best-in-table at s2, catastrophic at s0 (always-act bias)

<!--strip-->
> **id** `2026-07-25-vlmplan-8b-test-split` · **status** active · **tracks** baselines
<!--/strip-->

- **What:** First VLMPlan run on a real held-out split — `qwen3-vl-8b-instruct` (local, LM
  Studio, 32768 ctx), dd2d_v3 test, all **100 problems** (25/stratum), loop constants frozen
  from the pilot (τ=0.2, R=2, 10 plans/round, max 12 rounds, budget 200). 89 min generation
  (55 s/problem) + scoring. Cache `compare_cache/vlmplan_qwen8b/`.
- **Gate:** live-vs-stored label agreement **0.983** (n=60, 1 disagreement) — above the 0.95
  bar, so in-pool and off-pool labels are on the same function and the FPs are trustworthy.
- **Result — mean rollout FP, dd2d_v3 test (n=100), lower better:**

  | method | s0 | s1 | s2 | s3 | ALL |
  |---|---|---|---|---|---|
  | astar-dist | **0.00** | **2.24** | 17.08 | 119.28 | 34.65 |
  | PIGINet | 0.04 | 4.92 | 18.60 | 51.12 | 18.67 |
  | SPECTRE-adaptive | 0.00 | 9.20 | 29.52 | 53.00 | 22.93 |
  | SPECTRE-static | 0.00 | 27.44 | 27.20 | 46.36 | 25.25 |
  | SPECTREv2-adaptive | 0.00 | 4.60 | 26.20 | **23.92** | **13.68** |
  | SPECTREv2-static | 0.00 | 4.44 | 32.64 | 39.40 | 19.12 |
  | **VLMPlan-8B** | 4.24 | 2.88 | **16.04** | 96.28 | 29.86 |

  Overall it **beats the non-learned planner order** (29.86 vs 34.65) and loses to every
  trained method — but the aggregate hides two opposite regimes.
- **It is genuinely good in the middle.** At **s2 it is the best method in the table**
  (16.04, ahead of astar 17.08 and SPECTREv2-adaptive 26.20), and at s1 it beats every
  *learned* method (2.88 vs 4.60/4.92), losing only to astar. It found the feasible plan
  itself on 25/25 s1 and 14/25 s2 problems.
- **s0 is an always-act bias, and it is total.** Stratum 0 means the target is already
  graspable, so the answer is `retrieve` alone — pool index 0, which is why every other
  method scores 0.00. **0/25** of the model's first proposals staged nothing; it always moves
  something first (worst cases staged 2–3 items and cost 16–18 FP). This is precisely the
  failure `vlmplan_dd2d_implementation_plan.md` §8 predicted — "include a stratum-0 case
  (correct answer = stage nothing) to catch an always-stage-something bias before it silently
  costs stratum 0". That probe was descoped; the bias surfaced in the headline instead.
- **s3 unchanged in character:** 96.28, 17/25 fall through to the published-order fill, 1
  censored. It still does not propose feasible 3-subsets. Mimicry ρ climbs with stratum
  (−0.06 / 0.05 / 0.21 / 0.52); the s3 value is high because the fill *is* published order.
- **Generation quality** (5417 plan blocks over 496 rounds): parsed **62%**, of those only
  **2% symbolically invalid**, **39% duplicates**; 19.8 accepted plans/problem; **97/100
  problems ended by stalling**, not by the round cap — duplicates, not the budget, are the
  ceiling.
- **Truncation fixed and now measured: 15/496 rounds = 3.0%**, down from **16/104 = 15%** on
  the 2026-07-24 smoke run, where it had been silent. Raising `max_tokens` 4096→8192 (served
  at 32768 ctx) also raised yield materially — one dry-run problem went from ~23 plans to
  **48 plans in 12 rounds**, and a 7361-token response would previously have been cut at
  4096. The 2026-07-24 smoke numbers were produced under truncation and are superseded.
- **Takeaway / next:** VLMPlan is not uniformly weak — it is *regime-dependent*: strong where
  a single blocker is visually identifiable, useless at both ends (do-nothing and 3-subset).
  The s0 result is the cheapest available improvement and a real finding about zero-shot VLM
  planners. **1-model, 1-seed dev numbers.** Next: Qwen3-VL-**32B**-Instruct for a
  same-family scale comparison.

<a id="2026-07-25-rejected-reasoning-model-arm"></a>
## 2026-07-25 — `gemma-4-31b-qat` rejected as the large VLMPlan arm: it is a reasoning model

<!--strip-->
> **id** `2026-07-25-rejected-reasoning-model-arm` · **status** active · **tracks**
> baselines
<!--/strip-->

- **What:** With the Qwen3-VL-32B download stalled at 35% (6.95 of ~19 GB), tried
  `google/gemma-4-31b-qat` as the large arm — it was complete on disk and ships its own
  `mmproj`, so genuinely multimodal rather than a text-only substitute.
- **Result:** Vision works — correct answer to "which item is red?" in 3.3 s. But it is a
  **reasoning model**: that answer cost **229 completion tokens of which 222 were
  `reasoning_tokens`** (~95%). Reasoning tokens count against `max_tokens` but are stripped
  before the text the parser sees, so the plan budget is consumed by invisible thinking. On
  the real 10-plan prompt it exceeded 2 minutes without finishing 3 rounds (vs ~10 s/round
  for the 8B); an earlier 64-token probe returned an **empty string**, 61/64 tokens spent
  reasoning.
- **Takeaway / next:** Rejected — not a size problem, a model-class problem. The arm axis
  that matters is *instruct-tuned, non-reasoning, same family*, so the large arm is
  **Qwen3-VL-32B-Instruct** (already 35% downloaded, `mmproj` complete); avoid Qwen3-VL
  **Thinking** variants for the same reason. Holding the family fixed also upgrades the pair
  from a confounded size+family contrast to a clean **scale** comparison.
  `SEQUENCE_METHODS` second arm is now `VLMPlan-32B` → `vlmplan_qwen32b`.

<a id="2026-07-25-v3-reversal-was-short-first-prior"></a>
## 2026-07-25 — The dd2d_v3 "reversal" was an artifact: the short-first PRIOR collapsed v2 at s3; dropping it restores SPECTREv2-adaptive to best (1-seed dev)

<!--strip-->
> **id** `2026-07-25-v3-reversal-was-short-first-prior` · **status** active ·
> **tracks** method, baselines · **supersedes** 2026-07-24-retrained-all-three-on-v3
<!--/strip-->

- **What:** Investigated the 2026-07-24 dd2d_v3 result (v2-adaptive "collapsed" to 24.96 with
  s3=85.52; PIGINet "won" at 18.67). The user flagged it as implausible — the grasp fix *adds*
  feasibility, so methods should improve, not reverse; and v2 has strictly more information than v1
  yet did *worse* at s3. Read-only diagnosis: **(a)** pipeline is faithful — recomputing from the
  *surviving* original dd2d_v2 v2 checkpoint (`checkpoints_v2_evidence_prior_ov/dd2d_v2`, I earlier
  wrongly thought it was gone) reproduces the published **17.09** exactly; **(b)** recipe + training
  code are byte-identical to that checkpoint (git archaeology); **(c)** v3 is *easier* than v2 at
  every stratum (feasibility s0 .23→.33 … s3 .018→.023), so the reversal can't be "harder data";
  **(d)** v2 training *diverged* into a short-first length bias (val_relrank 0.99@e4 → 2.1 while
  val_loss kept dropping), and the noisy relrank selector grabbed the underfit epoch-4 fluke.
  Ablation found the cause: the **short-first `--use-prior`** (`[−index,−length]`) over-biases
  cross-length ordering on the easier v3 data — it buries the (mostly length-7) s3 feasibles.
  **Dropping the prior** (`--evidence --use-overlap`, keeping within-length PL + proof-demotion)
  fixes s3 *and* restores training convergence (val deployed-FP stable ~13–16 across epochs 12–29).
  Val-justified: no-prior val ALL **16.9** vs with-prior **29.9**. Retrained v2 no-prior, rebuilt the
  spectre2 cache (v1/PIGINet/astar/lenctx untouched).
- **Result** — corrected mean rollout **FP (test, n=100, 1-seed):**

  | method | FP(ALL) | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | astar-dist | 34.65 | 0.00 | 2.24 | 17.08 | 119.28 |
  | PIGINet (BCE) | 18.67 | 0.04 | 4.92 | 18.60 | 51.12 |
  | SPECTRE-adaptive (v1) | 22.93 | 0.00 | 9.20 | 29.52 | 53.00 |
  | SPECTRE-static (v1) | 25.25 | 0.00 | 27.44 | 27.20 | 46.36 |
  | **SPECTREv2-adaptive** | **13.68** | 0.00 | 4.60 | 26.20 | **23.92** |
  | SPECTREv2-static | 19.12 | 0.00 | 4.44 | 32.64 | 39.40 |

  Selection cross-check (robust to it): relrank-default `best.pt` = 13.68; deployed-val-FP epoch 14
  = 15.88 — same converged band. T0: dropping the prior flips v2-static from short-first
  (pearson −0.66) to **long-first (+0.58)**, matching the long s3 feasibles.
- **Takeaway / next:** The 2026-07-24 "PIGINet wins / negative-control confirmed" reading was an
  **artifact of the short-first prior on v3 — retracted.** Corrected picture: **SPECTREv2-adaptive is
  best overall (13.68)**, beating PIGINet (18.67) and even the v2-data 17.09 (v3 is easier), and it
  **dominates s3 (23.92** vs 46–119) via proof-demotion + a now-appropriate long-first base — the
  same qualitative shape as the v2-data result (v2 best, strong s3, weaker s2: v2 s2=26.20 still
  trails PIGINet 18.60). The **prior is now a data-dependent knob** (helped v2/RT2D, hurts the easier
  v3) and is dropped for v3 ([`decisions.md` 2026-07-25](../decisions/README.md)). Selection lesson: relrank is *miscalibrated*
  on v3 (never <1) but picks a converged epoch once the destabilizing prior is gone. Open: 3-seed
  reproduction (the prior's v2-helps/v3-hurts dependency + the s2 gap both want ≥3 seeds).

<a id="2026-07-24-retrained-all-three-on-v3"></a>
## 2026-07-24 — Retrained v1/v2/PIGINet on grasp-fixed dd2d_v3 + rebuilt the comparison: the headline FLIPS — PIGINet (low-level) wins, abstract-first behaves like the packing negative control (1-seed dev)

<!--strip-->
> **id** `2026-07-24-retrained-all-three-on-v3` · **status** superseded · **tracks**
> baselines, evaluation · **superseded by**
> 2026-07-25-v3-reversal-was-short-first-prior
>
> ⚠️ **SUPERSEDED** — the PIGINet-wins table was a training artifact of the
> short-first prior. Corrected: SPECTREv2-adaptive 13.68 → 14.50 is best and s3 is
> fixed. **Do not quote this table.**
<!--/strip-->

> ⚠️ **CORRECTED 2026-07-25 — do not cite this entry's result/takeaway.** The v2-adaptive s3
> collapse and the "PIGINet wins / negative control" reading below were a **training artifact** (the
> short-first `--use-prior` over-biasing cross-length ordering on the easier v3 data), *not* a real
> effect. After dropping the prior, SPECTREv2-adaptive is best (13.68) and s3 is fixed (23.92). See
> the 2026-07-25 entry above. The pipeline/method were faithful; only the trained v2 checkpoint was
> bad.

- **What:** Re-ran the full DD2D pipeline on **dd2d_v3** — the exact-count, grasp-fixed re-collection
  (100/100/100/100 train, 100 val, 100 test; λ=0.8, crowd=5 diverse, k=200, tb=20). Converted
  (`dd2d_convert.py … raw_root=data/dd2d/raw_v3 overwrite=true`), **harvested** post-mortems
  (`spectre_harvest.py --env dd2d_v3`: 67k/17k/17k facts, extraction-failed dominant), rebuilt vocab
  (3 ops / 6 preds / 1 type, OOV-clean), retrained all three: **v1** (`spectre_train env=dd2d_v3`,
  early-stop @13, val_rollout 9.49), **v2.2 observed** (`train_v2 --evidence --use-prior
  --use-overlap`, 30 ep), **PIGINet-BCE** (`piginet.train --arm weighted_bce --select auprc` on the
  native raw_v3 JSON). Rebuilt the 6-method compare cache (`precompute_dd2d_cache.py` now
  `--env-variant`-parameterized) + lenctx; notebook repointed to dd2d_v3 (n=100). VLMPlan pilot
  (train-band, 16 problems) set aside so the test table is clean. **1-seed dev.**
- **Result** — mean rollout **FP (test, n=100, 1-seed; lower better):**

  | method | FP(ALL) | s0 | s1 | s2 | s3 |
  |---|---|---|---|---|---|
  | astar-dist | 34.65 | 0.00 | 2.24 | 17.08 | 119.28 |
  | **PIGINet (BCE)** | **18.67** | 0.04 | 4.92 | 18.60 | 51.12 |
  | SPECTRE-adaptive (v1) | 22.93 | 0.00 | 9.20 | 29.52 | 53.00 |
  | SPECTRE-static (v1) | 25.25 | 0.00 | 27.44 | 27.20 | 46.36 |
  | SPECTREv2-adaptive | 24.96 | 2.04 | 3.72 | **8.56** | 85.52 |
  | SPECTREv2-static | 28.56 | 2.04 | 3.96 | 10.08 | 98.16 |

  Per-stratum winner: s0 astar/v1 (0.00); s1 astar (2.24); **s2 SPECTREv2-adaptive (8.56)**;
  **s3 PIGINet (51.12)**. PIGINet val **AUPRC 0.429 / AUROC 0.745** (up sharply from the dd2d_v2
  PIGINet's 0.256 / 0.658 — the grasp-fixed data is cleaner for the low-level predictor).
  **T0 length fit (mean per-episode):** astar R²=0.710 / pear=−0.842 / η²=0.745; PIGINet R²=0.188 /
  pear=−0.150 / η²=0.230; **SPECTRE-static (v1) R²=0.388 / pear=+0.620 / η²=1.000** (still a pure
  length lookup); **SPECTREv2-static R²=0.442 / pear=−0.663 / η²=0.460** (much less length-locked
  than v1, now short-biased).
- **Takeaway / next:** The comparison **reverses** vs dd2d_v2. There, SPECTREv2-adaptive was best
  (17.09) and PIGINet worst (29.70); on grasp-fixed dd2d_v3 **PIGINet is best overall (18.67)** and
  SPECTREv2 drops to 24.96 with an **s3 collapse (85.52 — the worst method at s3)**. Reading:
  SPECTREv2's geometry+evidence rep still helps at the **relational mid-stratum s2** (8.56, a clean
  win) and s1, but s3 (3-blocker) feasibility now hinges on **true packing geometry the abstract
  representation can't see**, so the **low-level predictor regains its edge** — exactly the
  **packing negative-control / crossover** direction the pivot predicts (`proposal.md` §0/§6). Two
  corroborations: (a) v2's **val_relrank ≈ random** (noisy, mostly ≥1 across all 30 epochs; the
  selected 0.987 is a single epoch-4 dip) — no stable within-length signal from the abstract rep on
  this data; (b) even s0 shows v2 FP 2.04 (occasionally front-loads a non-packing staging), the same
  packing-blindness. v1 (abstract-only) edges v2 overall (22.93 vs 24.96) because v2's s3 loss
  outweighs its s2 win. **All 1-seed dev — the noisy v2 selection makes 3-seed reproduction the
  blocking next step before any DD2D SPECTRE claim; the negative-control *direction* (low-level wins
  on grasp-fixed packing), though, is the expected result.** [`decisions.md` 2026-07-24](../decisions/README.md) has the
  retarget/parameterization decision.

<a id="2026-07-24-vlmplan-baseline-smoke-tested"></a>
## 2026-07-24 — VLMPlan baseline built and smoke-tested (Qwen3-VL-8B, local): it works, and it loses to astar-dist

<!--strip-->
> **id** `2026-07-24-vlmplan-baseline-smoke-tested` · **status** partially-superseded
> · **tracks** baselines · **superseded by** 2026-07-25-vlmplan-8b-test-split
>
> ⚠️ **PARTIALLY SUPERSEDED** — produced under a binding `max_tokens` cap.
<!--/strip-->

- **What:** Built the zero-shot VLM planning baseline (`vlmplan/`, KinDER convention) and ran
  it end-to-end on **dd2d_v3** (the fresh collection) — 5-problem pilot to set the loop
  constants, then a **16-problem smoke** (4 per stratum, train seeds 0/250007/500000/750000
  bands). Local `qwen3-vl-8b-instruct` via LM Studio, images on, temperature 1.0,
  `plans_per_round=10`, τ=0.2, R=2, `max_rounds=12`, attempt budget 200. Protocol in
  [`decisions.md` 2026-07-24](../decisions/README.md).
- **Gate — live-vs-stored label agreement (the prerequisite for mixing stored in-pool with
  live off-pool labels):** **1.000** (n=40) on v3. On the **stale v2** collection the same
  check gives **0.917** (n=168) in *both* directions — the fingerprint of that day's two
  grasp changes. Also measured: the refiner is deterministic at v2 settings (live-vs-live
  60/60), and the 2026-07-19 reconstruction invariant still holds (0/1624 stored-feasible
  subsets read as blocked). So the live refiner tracks current env code; v2's gap is
  staleness, now quantified rather than assumed.
- **Generation quality** (16 problems, 880 plan blocks emitted): parsed **535/880 = 0.61**;
  of those parsed, **0/535 symbolically invalid** and 164 (31%) duplicates; **23.2 accepted
  plans/problem** (min 10, max 43) over ~5 rounds before stalling. Two prompt/parser fixes
  were load-bearing to get here, both disclosed in `vlmplan/prompts/PROVENANCE.md`:
  stating each skill's preconditions/effects (without it the model ended **28/28** valid
  plans with `pick(target)` instead of `retrieve(target)` — 100% invalid), and accepting
  `pick(item_2)` for `pick(item_2:item)[]` (without it **31/31** blocks in a round were
  rejected on format alone).
- **Result — FP vs astar-dist on the *same 16 problems*** (VLMPlan's FP counts off-pool
  attempts; astar = published order):

  | stratum | astar-dist | VLMPlan | Δ |
  |---|---|---|---|
  | s0 | 0.0 | 2.0 | +2.0 |
  | s1 | 2.0 | 21.0 | +19.0 |
  | s2 | 15.8 | 19.5 | +3.8 |
  | s3 | 123.2 | 123.2 | +0.0 |
  | **ALL** | **35.2** | **41.4** | **+6.2** |

  Wins 3/16, ties 7, loses 6. First success found by the **model itself on 8/16**, by the
  published-order fill on 8/16 (**all four s3 problems** fell through to the fill). 0
  censored, mean 4.6 off-pool attempts/problem, 73 live refines total. Trivial-mimicry null
  `spearman(realized, published)` = **0.46** mean (n=12 with ≥2 in-pool attempts) — so it is
  *not* merely reproducing the planner's size-ascending order.
- **The s3 exact tie is an identity, not a coincidence.** When every proposal is in-pool and
  none is feasible, the fill replays the remaining pool in published order, so the same
  infeasible plans precede the first success either way and FP is *provably* unchanged.
  VLMPlan can therefore only win by proposing a feasible plan earlier (s2: 15→10, 10→0) and
  only lose by proposing failures ahead of where astar would already have succeeded — either
  off-pool (pid 250008: 34 off-pool misses, FP 1→35) or in-pool (pid 2: 8 infeasible pool
  plans before the feasible one astar hits at rank 0).
- **Takeaway / next:** The harness works end-to-end and the baseline is *functional* — it
  proposes valid, mostly-novel plans and finds the feasible one unaided on half the problems.
  On this model it is **worse than the non-learned planner order**, and the failure is
  concentrated exactly where the benchmark is designed to bite: it never proposes a feasible
  3-subset at s3. **These are 1-model, 16-problem, train-split plumbing numbers** — not
  reportable. Next: the full test split, `Qwen3-VL-32B`, and the GPT-5.x arm, so a weak result
  can be attributed to the task rather than to an 8B quantized model.

<a id="2026-07-24-post-grasp-sanity-astar"></a>
## 2026-07-24 — Post-grasp-change sanity: astar (pyperplan) planner + backjumping refiner + env work across s0–s3

<!--strip-->
> **id** `2026-07-24-post-grasp-sanity-astar` · **status** active · **tracks**
> baselines, env-dd2d
<!--/strip-->

- **What:** Verified the whole DD2D planner/refiner/env pipeline still works after the two grasp
  changes, by running the end-to-end demo (`envs/dd2d/dd2d/demo.py`, imported from the envsearch
  repo) with the **astar baseline** (`--planner pyperplan --pyperplan-search astar
  --pyperplan-heuristic dist` = the notebook's "astar-dist"). Extended `demo.py` with `--stratum
  {0,1,2,3}` (exact min-feasible-subset: 0 = `unblocked_target`, 2/3 = `min_subset` floor + exact
  filter, 1/None = naturalistic) and `--min-blockers/--max-blockers` (per-problem blocker-count
  range), since the CLI could not express exact strata or a blocker range. Run config: λ=0.8,
  margin=1, k=200, crowd=5 diverse, blockers 8–10, 20 problems/stratum.
- **Result (full 20/stratum run, k=200, astar-dist, blockers 8–10):** all four strata generate
  **exactly** (Strata-seen: s0/s1/s2/s3 = 20 each), and the astar baseline's mean first-feasible
  rank rises cleanly with stratum — **s0 1.0** (20/20), **s1 3.4** (20/20), **s2 15.2** (20/20),
  **s3 119.4** (18/20). The s3 mean matches the ~120-ish the full-dataset collection saw; 2/20 s3
  problems have their feasible 3-subset beyond rank 200 (the expected geometry-blind tail). Refiner +
  per-plan videos + render confirmation all OK across all strata. s0 under crowd=5 generates fine —
  the new internal grasps give the target more grasp options, so unblocked-target scenes are easy to
  find.
- **Correction (recorded):** an earlier k=40 s3 smoke showed 0 feasible and I wrongly extrapolated
  "not found ≤ k=200". At k=40 the ascending-length enumeration hasn't reached 3-object stagings yet
  (retrieve + ~9 one-object + ~72 two-object come first; 3-object stagings start ~rank 82), so k=40
  cannot reach s3 and is uninformative about k=200. s3 IS solvable within k=200 (mean rank ~119).
- **Takeaway / next:** The grasp changes did not break the planner/refiner/env; the demo is a
  working per-stratum harness for the astar baseline. Also added `demo.py --workers N` — a
  `ProcessPoolExecutor` over problem slots (each slot draws from a disjoint seed space, so results
  are **worker-count-invariant**: serial and 16-worker runs give byte-identical aggregates,
  verified). Full 20/stratum × 4 videos run to `out_dd2d/astar_demo/s{0,1,2,3}/` (numbers there are
  illustrative on the **stale** pre-recollection labels — the grasp model changed twice today, so
  this is a plumbing check, not reportable data).

<a id="2026-07-24-grasp-reaches-concavities"></a>
## 2026-07-24 — Grasp model reaches into concavities: grip the dumbbell bar / into the horseshoe opening (internal antipodal grasps)

<!--strip-->
> **id** `2026-07-24-grasp-reaches-concavities` · **status** active · **tracks**
> env-dd2d
<!--/strip-->

- **What:** Follow-up to the same-day contact-run fix. Reviewing the demo videos showed the model
  still only ever gripped the **outer envelope** — no grasp reached into a concave region (e.g. it
  could not hold the middle bar of a dumbbell). Root cause: `direction_admissible` only emits the
  **global x-extreme** supporting lines. Added `_internal_grasps` to `grasps.py`: a scan-line
  antipodal enumerator that, per direction, grips any **strictly-internal flat feature** where the
  fingers fit — validated by (1) **finger-fit** (finger rects clear the item's own material — "the
  grippers fit" in the concavity) and (2) **full-face flat contact** (≥ 0.9·`FINGER_WIDTH` of each
  finger face on the boundary, which also excludes curved-shape sliver pinches). `grasp_cells` =
  global grasps **+** internal grasps; everything else (`finger_rects`/`grasp_cfree`/`has_grasp`,
  the `Grasp` dataclass, which already carried arbitrary `xmin`/`xmax`) unchanged.
- **Result** (20 samples/family; `finger gap` and `finger∩item overlap` both **0** everywhere):

  | family | mean cells | internal cells | example internal grasp |
  |---|---|---|---|
  | can / bowl / box / pillcase | 2–7.5 | **0** | — (convex: no flat internal feature; circles keep only their tangent grasp) |
  | dumbbell | 9.6 | ~5/9 | **the bar** — α=90°, sep ≈ 1.76 cm, fingers beside the waist |
  | horseshoe | 17.1 | ~7/16 | **the spine** — α=0°, sep ≈ 2.31 cm, right finger **inside the C-opening**; also both prongs |
  | shoe | 15.8 | ~10/15 | **an arm** — a finger in the concave L-corner |

  All internal grasps are **full-face** (demo reads "2.50 of 2.5 cm finger on material").
- **Deliverables (vision-inspected):** regenerated `out_dd2d/grasp_demos/*.mp4`; the sweep now tags
  and always shows an **"INTERNAL GRASP — a finger reaches into the concave region"** cell. Confirmed
  visually: dumbbell gripped at the **middle bar**, horseshoe finger **in the opening**, shoe finger
  **in the L-corner**. Tests added: `test_internal_grasp_on_dumbbell_waist`,
  `test_internal_grasp_on_horseshoe_spine`, `test_fingers_fit_in_isolation` (all families),
  `test_convex_families_have_no_internal_grasp`; existing invariants stay green.
- **Takeaway / next:** The gripper now grasps concave regions where it physically fits — realistic,
  and the paper's grasp claims are defensible. Direction-of-difficulty note: this is
  **monotone-easier** (adds grasps → feasible-candidate sets can only grow, `min_feasible_subset` can
  only drop), **partially offsetting** the earlier-today no-air-grasp change (which was
  monotone-harder). Both are realism-driven. Re-collection (→ vocab → retrain → recompare) is still
  required and deferred — the change again shifts DD2D labels (extraction only; packing/certificate
  unaffected).

<a id="2026-07-24-grasp-fixed-contacts-material"></a>
## 2026-07-24 — Grasp model fixed to contact material (no more air-grasps) + `banana`→blocky `horseshoe`; demos show full-face concave grasps

<!--strip-->
> **id** `2026-07-24-grasp-fixed-contacts-material` · **status** active · **tracks**
> env-dd2d
<!--/strip-->

- **What:** Two coupled changes to make the DD2D gripper defensible for the paper (prompted by
  the 2026-07-23 air-grasp finding, ahead of adding a VLM-planning baseline):
  1. **Grasp model** (`grasps.py`): `direction_admissible` now draws each slide from the
     **intersection of the two supporting lines' *actual* contact runs** (new
     `_contact_runs_on_line` + `_intersect_runs`), not the y-**hull**. So every emitted grasp
     cell has **both fingers on material** (gap = 0); a finger can no longer close across a
     C-opening / waist. Circles keep their valid single-tangent-point grasp; the fatal
     "require full 2.5 cm finger face" variant was rejected (it kills 100% of can/bowl).
  2. **Shape** (`shapes.py`): replaced the curved `banana` with a blocky, right-angled
     **`horseshoe`** — a vertical spine + two **equal-length** prongs, opening +x, **symmetric
     about y=0**, one 8-vertex rectilinear polygon. Prong thickness ≥ `FINGER_WIDTH` (2.5 cm)
     so a flat finger makes **full-face** contact where the old banana only ever got a tangent
     point. Renamed everywhere (`_CONCAVE_FAMILIES`, weights, glosses, certificate comment,
     demo, tests).
- **Result:**
  - Grasp-cell contact, 40 samples/family: **max finger gap 0.0000 across ALL 7 families**
    (was up to 2.6 cm on concave). Cells/shape: can/bowl 2.0, box 7.6, pillcase 6.2, shoe 6.5
    (unchanged); **dumbbell 5.6→4.9** (waist air-cells removed); old curved banana collapsed to
    1.0 (why it was replaced). `horseshoe`: **9–10 cells/shape, 0 floating, 40/40 ungraspable-free,
    40/40 have a ≥2.5 cm full-face grasp**; area mean 42.7 cm² (35–50), bbox ~6.2×8.8 cm —
    banana-comparable. Symmetric about the horizontal axis on all 40 (symdiff < 1e-6).
  - **Deliverables (vision-inspected):** `out_dd2d/shape_families.png` (all 7 families ×3 — clean
    symmetric horseshoe row, others unchanged); `out_dd2d/grasp_demos/{horseshoe,shoe,dumbbell}_s{0,1,2}.mp4`
    — isolation sweep shows **full-face green contact** (α=90° reads "2.50 of 2.5 cm finger on
    material"), the in-clutter clip ends in a collision-free grip + elevated-carry lift; contact
    table reads **0/N floating** for every clip. Convex `can` regression clip unchanged
    (point contact, valid).
  - Tests: `test_every_grasp_cell_makes_contact` (all families, gap ≤ tol) +
    `test_horseshoe_grasp_is_full_face` added; the old `test_..._close_onto_a_gap` (which asserted
    the *bug*) inverted to `test_no_grasp_cell_floats_on_concave_families`. Full dd2d + spectre
    suites **444 pass** — no label-count assertion shifted.
- **Takeaway / next:** The gripper now grasps concave items correctly and the demos prove it
  visually — the realism objection is closed. Per the 2026-07-23 follow-up this is
  **monotone-harder** (removes grasp cells → feasible sets shrink, strata can only rise), so it
  **invalidates the v2.2 collection/checkpoints/comparison** — DD2D must be **re-collected → vocab
  → retrain → recompare** before any DD2D number is trusted again (deferred; user runs it when the
  VLM-baseline timing allows). Existing collected JSON is stale on both axes (family name **and**
  old-model grasp labels).

<a id="2026-07-23-adaptive-traces-carry-step-scores"></a>
## 2026-07-23 — Adaptive traces now carry per-step scores; planner inspector rebuilt (FP table unchanged)

<!--strip-->
> **id** `2026-07-23-adaptive-traces-carry-step-scores` · **status** active ·
> **tracks** tooling
<!--/strip-->

- **What:** Made the DD2D comparison cache record *what the adaptive rankers thought at
  every step*, so the notebook's §7 planner inspector can show promotions/demotions
  without ever running inference at load. `deployed_rollout_traced` (v2) and
  `ChoiceStep.scores` (v1, via a new `inference.score_pool` / `argmax_in_pool` split)
  emit the raw per-step logits + the provably-dead set; `precompute_dd2d_cache.py`
  persists them; rebuilt `spectre_adaptive` + `spectre2_adaptive` (`--force`).
- **Result:**
  - **Every published FP number is unchanged** — all 851 `(method, problem)` rows are
    identical pre/post rebuild (astar 33.11 / PIGINet 29.70 / SPECTRE-adaptive 23.39 /
    SPECTRE-static 22.63 / SPECTREv2-adaptive 17.09 / SPECTREv2-static 18.89). Only
    new keys were added.
  - Rebuild cost **2569 steps (v2) + 3463 (v1)** at ~26 ms/step ⇒ ~1 min each;
    **+11 MB** cache at 4-dp rounding.
  - The stored v2 `step_dead` matches an **independent offline `ProofState` replay
    exactly** (0 mismatches / 142 problems) and is sound (**0** demoted-but-feasible).
    v1's is empty everywhere, as it must be — v1 has no proof-demotion.
  - Cheap-replay measurement that motivated storing rather than recomputing: the
    offline replay is **0.031 s for all 142 problems** (0.2 ms each), mean 3.5 demoted.
  - **Serialisation gotcha:** the v2 model masks its own failure context, so a step's
    raw row carries `-inf` for every already-attempted candidate — not strict JSON.
    Stored as `null`, read back as `NaN`. Verified the non-finite set is *exactly*
    `order[:t]` at every step. v1's `score_pool` is unmasked, so its rows are fully
    finite (0 NaN in 692 600 entries).
  - `ad.score` = **score at the step the candidate was picked**, not the final-step
    score: at the final step every attempted candidate is masked, which left the whole
    attempted prefix blank. Score-at-pick is available for **6032/6032** attempted
    candidates across both families.
- **Takeaway / next:** The inspector now reads promotions/demotions directly. Worked
  example, pid 1750021 (s3, FP 125): proof-demotion at attempt 34 buried
  `stage {1}` from static rank **1** to position **192** (Δrank −191) — the sound
  filter visibly overriding a confident wrong static score. Still **1-seed dev**;
  a 3-seed reproduction remains the open item from 2026-07-20.

---

<a id="2026-07-23-concave-grasp-sanity-demo"></a>
## 2026-07-23 — Concave-grasp sanity demo: the gripper closes onto a concavity in 11/41 cells; the banana only ever gets a tangent point

<!--strip-->
> **id** `2026-07-23-concave-grasp-sanity-demo` · **status** active · **tracks**
> env-dd2d
<!--/strip-->

- **What:** Built a visual sanity check of the DD2D two-rectangle gripper on the three
  **concave** families (`envs/dd2d/dd2d/demo_grasp_concave.py`, + `tests/test_demo_grasp_concave.py`).
  Nine mp4 clips (3 samples × banana/shoe/dumbbell) → `out_dd2d/grasp_demos/`; each clip is an
  **isolation sweep** (the gripper stepping through the item's admissible `(alpha, s)` cells,
  fingers closing flush, with the supporting lines, the *actual* contact runs, and per-finger
  contact status drawn) followed by an **in-clutter** animation of what `has_grasp` really does
  (cells in order, red where the fingers penetrate a neighbour/wall, until the first free one,
  then lift). Prompted by the question of whether flat rectangular fingers behave sensibly on a
  *curved* banana.
- **Result** — over all 41 admissible cells of the 9 sampled items. "floating" = a finger closes
  onto the concavity instead of material; "min touch" = weakest both-fingers-touching cell, cm of
  the 2.5 cm finger face actually on material:

  | family | cells | floating | max gap (cm) | min touch (cm) |
  |---|---|---|---|---|
  | banana (3 samples) | 11 | 10/11 | 2.62 | **0.00** (every contacting cell is a tangent point) |
  | shoe (3 samples) | 15 | 0/15 | 0.00 | 1.58 |
  | dumbbell (3 samples) | 15 | 3/15 | 2.28 | 0.44 |

  Convex families (`box`/`can`/`pillcase`/`bowl`) never float a finger — pinned by test.
- **Why:** `direction_admissible` keeps only the **y-hull** of each supporting line's contact
  set. On a convex footprint that set is one connected run, so the hull is exact; on a C-opening
  / waist it is disconnected, and a slide drawn from the middle of the hull lands opposite a gap.
  The dumbbell's mid-waist cell floats *both* fingers by 2.28 cm; the banana's only cells either
  float or pinch the outer arc at a single tangent point.
- **Takeaway / next:** The gripper is **not broken** — it is doing exactly what the
  supporting-line abstraction specifies, and `grasps.py` already disclaims force closure
  (spec §5.3 / m7). But "graspable" on a concave item can mean a gap closure, and
  `has_grasp` feeds `label.py` → the DD2D feasibility labels, so a no-contact grasp counts
  as a successful pick. Costed in the follow-up below.

### Follow-up (same day) — costing the fix: contact-required is cheap, contact-*length* is fatal

- **What:** Swept stricter admissibility rules over 40 sampled shapes × all 7 families
  (cells surviving, shapes left ungraspable in isolation), then measured the label blast
  radius on 40 generated scenes (430 items, λ=0.8, crowd=5 diverse).
- **Result (a) — mean grasp cells per shape, current → rule:**

  | family | current | fingers must touch | ≥0.25 cm contact | ≥1.0 cm |
  |---|---|---|---|---|
  | can / bowl | 2.0 | 2.0 (0 dead) | **0.0 (40/40 dead)** | 0.0 |
  | box | 7.6 | 7.6 | 7.6 | 7.6 |
  | pillcase | 6.2 | 6.2 | 6.2 | 5.0 |
  | shoe | 6.5 | 6.5 | 6.5 | 6.5 |
  | dumbbell | 5.6 | 4.6 | 4.6 | 3.8 |
  | banana | 2.9 | **0.8 (8/40 dead)** | 0.0 (40/40) | 0.0 |

- **Result (b) — label blast radius:** items whose "graspable" verdict rests *only* on a
  floating grasp: **2/430 = 0.47%** (1 dumbbell, 1 banana). In-situ graspable rate
  0.495 → 0.491; F1 target-blocked rate **0.825 → 0.825 (unchanged)**.
- **Takeaway:** Two things. (1) **A minimum contact *length* is a non-starter** — a circle
  meets its supporting line at a single tangent point exactly like the banana does, so any
  positive threshold kills 100% of `can` and `bowl`, the two most-weighted families. Point
  contact is the generic case for *any* smooth convex boundary, **not** a concave
  pathology; the genuine anomaly is only the **positive gap**. This corrects the
  over-reading in the entry above. (2) Requiring the fingers to **touch** (gap ≤ 0) is
  free for 5 of 7 families, costs the dumbbell ~18% of its cells, and hits only the
  banana hard (−72% of cells; 20% of sampled bananas become ungraspable in isolation and
  would be resampled, shifting the banana sub-distribution toward wider openings).
  The change is **strictly monotone — problems can only get harder**: removing grasp cells
  can never make an infeasible candidate feasible, so feasible-candidate sets shrink and
  `min_feasible_subset` can only rise (strata shift up). Not acted on: 0.47% of verdicts
  do not justify invalidating the v2.2 collection + checkpoints + comparison numbers
  trained on it. Revisit at the next collection boundary, where it is nearly free.

