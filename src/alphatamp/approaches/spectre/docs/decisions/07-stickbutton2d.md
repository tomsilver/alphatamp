# SPECTRE Decisions — StickButton2D as a second environment

3 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---
<a id="2026-08-10-held-out-stratum-anomalies-resolved-matched-controls-per-stratum"></a>
## 2026-08-10 — Held-out-stratum anomalies resolved: matched controls, per-stratum paired bootstrap, correct-size b5

<!--strip-->
> **id** `2026-08-10-held-out-stratum-anomalies-resolved-matched-controls-per-stratum`
> · **status** active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**Context.** The held-out-stratum experiment
([2026-08-09](#2026-08-09-held-out-stratum-generalization-train-s0-s2-b1-b3-evaluate)) produced two
results that read as incoherent against "a training superset should help": DD2D SPECTRE-adaptive
trained on the *subset* s0–s2 had a lower ALL FP (5.35) than the deployed *full* number (5.78), and
on SB2D the SPECTRE-vs-PIGINet ranking *flipped* (full: SPECTRE 1.84 < PIGINet 2.28; held-out:
PIGINet 1.68 < SPECTRE 2.10), with PIGINet appearing to *improve* when b5 was held out. An audit
found three confounds: the held-out models use the *current* code (domain-agnostic inputs +
`--select-window 5`), matching the **5.92 / 1.84** deployed models, **not** the frozen **5.78 / 1.69**
target-anchored yardsticks the anomaly compared against; ALL averages the held-out stratum with the
trained strata; and SB2D's b5 **train split was only 17 episodes ≈ 6 % of training** (vs DD2D's
s3 = 25 %), a near-null perturbation swamped by the documented ~1 FP run-to-run noise floor.

**Decision.** Resolve the anomalies with (a) **matched full-strata controls** — current-code,
same-recipe models on the *same* test problems, not the frozen yardstick — compared (b) **per
stratum with a paired bootstrap over problems** (`experiments/spectre/holdout_vs_full.py`, reusing
`eda.bootstrap_mean_difference`), headlining the held-out stratum, not the pooled ALL; and (c) for
SB2D, **fix the root cause by collecting b5 train to the correct 100** so the full model can
actually learn b5. The b5 re-collection goes into a **new variant `stickbutton2d_v2`** (b1/b2/b3 and
val/test reused from v1 by copy/symlink; b5 train collected 17→100 via the new
`sb2d_collect.py --env-variant` flag), leaving `stickbutton2d_v1` and its published numbers frozen.
Only the *full* model is retrained (the subset = b1/b2/b3 is identical to the held-out arm and is
reused). **DD2D full control = the deployed `dd2d_v4` cache** (the correct all-strata full at
100/stratum, current code) — a fresh seed-matched DD2D arm was attempted but trained pathologically
slowly (~700 s/epoch vs ~6, cause undiagnosed) and was abandoned rather than block for hours; the
deployed cache differs only in the training draw, which the paired-over-problems bootstrap absorbs.
Judgment calls made autonomously and recorded here: the b5 collection was **stopped at train=100 /
val=22** rather than spend 20–60 min more on b5 val's low keep-rate (val=22 ≈ the deployed val
size, adequate for selection); and the full-vs-subset comparison is surfaced via the standalone
diagnostic script rather than a `compare_methods.py` cell, to avoid colliding with the concurrent
LAZY work in that file.

**Consequences.** With matched controls + correct-size b5 + per-stratum paired CIs, **no
"holding-out-data-helps" effect survives, and the aggregate anomalies are noise**:
- **DD2D.** SPECTRE-adaptive ALL Δ(subset−full) = −0.57 [−1.51, +0.29] — **not significant** (the
  5.35 < 5.92 "anomaly" is within noise). The held-out **s3 is directionally coherent** (subset
  9.97 vs full 8.79, Δ +1.19 [−0.77, +2.89], ns at n=25). PIGINet is decisively coherent: held-out
  **s3 Δ +40.69 [+26.80, +54.77]** (full 45.20 ≪ subset 85.89). The one significant sub-headline
  effect is **s1 specialization**: subset 1.88 vs full 4.84, Δ −2.96 [−4.93, −1.31] — holding out
  the hard s3 makes the model *better* on the trained stratum s1, which is what dragged ALL down.
- **SB2D (b5 now 100).** The flip is **gone**. SPECTRE-adaptive ALL Δ −0.06 [−0.60, +0.47] and
  PIGINet ALL Δ −0.21 [−0.75, +0.25] — both **within noise**. On held-out **b5, SPECTRE is
  directionally coherent** (subset 6.87 vs full 6.13, Δ +0.73, ns) — the coherent direction that
  the 17-episode full model could not produce; **PIGINet shows no effect** (subset 5.36 vs full
  5.79, Δ −0.43 [−2.44, +1.27], ns). The significant effect is again **trained-strata
  specialization** (b3: subset 1.20 vs full 2.12, Δ −0.92 [−1.72, −0.28]).
- **Reading, in one line:** *"superset helps" holds on the held-out stratum in direction (3/4
  cases), is significant only where the effect is large (DD2D PIGINet), and is otherwise inside the
  ~1 FP noise / 25-test-problem resolution; the ALL "subset wins" was confound + noise.* The
  robust cross-environment finding is that holding out the hardest stratum *specializes* the model
  on an easy trained stratum (DD2D s1, SB2D b3).
- **b5 correct-size is a real fix, not cosmetic:** it removed the 17-episode artifact (the full
  model now trains on b5) and turned SB2D b5 from "no measurable full-vs-subset difference" into a
  coherent-direction (if still ns) one for SPECTRE, and confirmed PIGINet's SB2D-b5 non-separation
  is a *powered-on-training* negative, not a training-size artifact. `stickbutton2d_v2` +
  `_v2_kinder` are registered variants; v1 is byte-unchanged (267 train).
- **Traps:** the fresh DD2D control's ~100× slowdown (undiagnosed — concurrent with the b5
  collection tail) is a reminder to not co-schedule heavy SPECTRE training with a 30-worker
  collection; and `kill`/`pkill` are sandbox-blocked, so runaway training is stopped via `TaskStop`
  on the launching job, not a shell kill.

---

<a id="2026-08-09-lazy-policy-guided-adaptive-baseline-added-dd2d"></a>
## 2026-08-09 — LAZY policy-guided adaptive baseline added (DD2D + SB2D)

<!--strip-->
> **id** `2026-08-09-lazy-policy-guided-adaptive-baseline-added-dd2d` · **status**
> active · **tracks** baselines, method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**Context.** The comparison had two learned methods — SPECTRE (adaptive) and PIGINet (a
*static* ranker) — plus astar and VLMPlan. The paper claims SPECTRE beats other *adaptive*
methods, but there was no learned adaptive competitor in the figures. LAZY (Khodeir et al,
*Policy-Guided Lazy Search with Feedback for TAMP*) is the canonical one: a learned GAT
policy guiding refinement order, updated online by feasibility statistics ϕ. Its reference
code (`baselines/drake-tamp/`, `lifted_merged`) is ~80% blocks-world/PDDLStream-specific and
does integrated incremental search over a computation graph SPECTRE has no analog for, so it
was **re-implemented** over the fixed candidate-pool substrate every method here uses. New
package `baselines/lazy/` (the folder PIGINet will move into too); deviations enumerated in
`baselines/lazy/PROVENANCE.md`.

**Decision.** Realize LAZY at **maximum fidelity** on the pool-reranking substrate:
- **Prefix-tree policy.** Build a trie over the pool's canonicalized operator sequences; at
  each node the GAT policy scores the candidate next-operators, π(op|node)=softmax, and
  π(skeleton)=∏ per-action π along its path. Object encoder = the literal
  `torch_geometric.nn.GATv2Conv` (2 layers). BC = cross-entropy over candidate next-ops with
  the demonstrated next-op (toward a feasible leaf) as target.
- **Feasibility ϕ = (succ+1)/(att+1)** keyed on the per-operator canonical key (the
  `utils.anonymise` analog), fit from train outcomes with per-operator failure attribution
  (`records_for_candidate`; DD2D `culprits`, SB2D `dev_blame` + suffix-blame fallback), and
  updated **online** as skeletons fail. Combined as π̄=π·ϕ/Σ with a LevinTS 1/path_prob
  priority.
- Registered as its own **`LAZY_FAMILIES`** in `compare.py` (adaptive-only, one row) — not
  `SPECTRE_FAMILIES` (forces a static twin) and not `SEQUENCE_METHODS` (off-pool semantics).
  Cached by a new `cache_lazy` in `precompute_dd2d_cache.py` (adaptive record shape +
  §2b timing). SB2D is trained + cached on `stickbutton2d_v1` (LAZY is image-free) and
  **grafted** into the kinder display via `legacy_only`, matching SPECTRE's wiring.
- Selection metric is **val rollout-FP** (the project arbiter), not BC cross-entropy (which
  has a label-conflict floor at the high-fanout root). 3 seeds, matching PIGINet/SPECTRE.

Two design calls made autonomously (user asleep; recorded here per the standing directive):
(1) the **action-selection head** is an attention-pool + MLP over
`[op-embed ‖ arg-node embeddings ‖ pooled graph context]`, not the paper's third
cross-attention `GATv2Conv`, for batching robustness — the GAT policy proper is still the
literal `GATv2Conv` (`PROVENANCE.md` deviation 4); (2) plain `(succ+1)/(att+1)` without the
reference attempt-adaptive multiplier (`PROVENANCE.md` deviation 5).

**Consequences.** LAZY is a legitimate adaptive baseline, and the result is a clean
cross-environment split (3 seeds; paired bootstrap over the 100 test problems):

| | DD2D (dd2d_v4) | SB2D (kinder) |
|---|---|---|
| **LAZY-adaptive** | **23.26 ± 0.50** | **1.85 ± 0.02** |
| SPECTRE-adaptive | 5.92 ± 0.29 | 1.84 ± 0.25 |
| PIGINet | 17.27 ± 0.19 | 2.28 ± 0.29 |
| SPECTRE-static | 21.65 ± 1.13 | 1.98 ± 0.28 |
| astar-dist | 34.52 | 16.29 |

- **DD2D (packing negative control): both learned rankers beat LAZY decisively.**
  SPECTRE−LAZY = −17.34, CI [−24.0, −11.4]; PIGINet−LAZY = −5.99, CI [−9.96, −2.28] (both
  exclude 0). LAZY still beats the naive order (astar 34.52) and VLMPlan (35.23), carried
  entirely by s3 (LAZY 58.65 vs astar's pathological ~119 on val), and ≈ SPECTRE-static.
- **SB2D (relational): LAZY ties everything.** SPECTRE-adaptive−LAZY = −0.01, CI
  [−0.72, +0.72]; LAZY−PIGINet = −0.44, CI [−1.18, +0.29] — neither separates. This is the
  same non-separation the standing SB2D finding reports for SPECTRE vs PIGINet, now extended
  to a third adaptive method: on SB2D no method separates; on DD2D the learned rankers win.
- A **diagnostic** isolating the policy (astar / ϕ-only / LAZY) confirms the GAT policy is
  load-bearing, not inert: on DD2D val LAZY 28.70 < astar 35.66 while ϕ-only is *worse*
  (49.03) — the policy carries it; on SB2D test ϕ-only already reaches 2.40 (feasibility is
  highly discriminative there) and the policy sharpens it to 1.86.
- **Caveats.** LAZY's SB2D b5 uses the small (17-episode) b5 train split, so 4.56 at b5 is
  substantially a generalization number (a b5 expansion was in progress). LAZY's seed sd is
  tiny (±0.02 SB2D) because seeds share the deterministic canonicalization and the fitted ϕ;
  only model init varies. `torch_geometric` 2.8.0 was added to `pyproject.toml` (core only;
  runs on Blackwell sm_120, G0 verified).

---

<a id="2026-08-09-held-out-stratum-generalization-train-s0-s2-b1-b3-evaluate"></a>
## 2026-08-09 — Held-out-stratum generalization: train s0-s2 / b1-b3, evaluate the never-trained s3 / b5

<!--strip-->
> **id** `2026-08-09-held-out-stratum-generalization-train-s0-s2-b1-b3-evaluate` ·
> **status** active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**Context.** The method comparison has only ever trained and tested the learned rankers
(SPECTRE, PIGINet) on the *same* strata — DD2D s0–s3, SB2D b1/b2/b3/b5. We had never held a
stratum out of training entirely, so the comparison could not say whether the learned
representations generalize to an *unseen* stratum (a new min-feasible-subset size on DD2D, a
new button count on SB2D). This is a distinct axis from the 2026-08-01 unseen-count /
unseen-shape tests, which held out geometry, not a stratum. The training split already carries
100 problems per stratum (400 total), so holding out s3 / b5 is a train-time exclusion, **not a
re-collection**.

**Decision.** Train SPECTRE and PIGINet on s0–s2 (DD2D) / b1/b2/b3 (SB2D), evaluate on all four
strata of the standard test split, and report the held-out stratum (s3 / b5) as the headline.
Least-invasive mechanics:
- **`--train-strata`.** SPECTRE already had `TrainV3Config.train_strata`; PIGINet gained a
  matching `--train-strata` (a stratum filter in `PIGINetDataset`, reusing the `problem_ids`
  hook). SB2D examples carry no `stratum` in provenance (only `num_buttons`), so the stratum is
  recovered from the pid via `compare.stratum_of` (`_pid_stratum`), uniform across both envs.
  Both filter train **and** val, so the checkpoint selector never sees the held-out stratum.
- **Held-out variants are raw-symlinks** to their backing collection
  (`dd2d_v4_holdout_s3 → dd2d_v4`, `stickbutton2d_v1_holdout_b5 → stickbutton2d_v1`,
  `stickbutton2d_v1_kinder_holdout_b5 → stickbutton2d_v1_kinder`) with a copied vocab — only the
  trained checkpoint differs. Registered in `_PIGINET_PATHS` / `_REFINE_CAP_S` /
  `sb2d_adapter._SB2D_CROP_SOURCE`; SPECTRE scored via
  `--v3-arm spectre3:checkpoints_v3_holdout_s{seed}`.
- **astar and VLMPlan are training-free**, so astar is re-scored natively and the frontier
  VLMPlan-GPT5.6 is reused verbatim by symlinking its cache subdir (identical test problems).
  SB2D keeps the deployed two-cache split (SPECTRE on the instrumented v1 refiner, PIGINet on
  kinder crops, tied by `legacy_variant`).
- Two new notebook entries (`dd2d_holdout_s3`, `sb2d_holdout_b5`) — `EnvSpec`s only, no notebook
  edit; FP **and** §2b wall-clock, full per-stratum breakdown.

**Consequences.**
- **DD2D: SPECTRE-adaptive generalizes to the unseen stratum; PIGINet collapses.** Held-out s3
  (3 seeds): SPECTRE-adaptive **9.97 ± 1.59** vs PIGINet **85.89 ± 9.25**, astar 118.76, VLMPlan
  59.30 — a ~9× margin over the low-level predictor. ALL: SPECTRE-adaptive 5.35 (≈ its in-dist
  5.78) vs PIGINet 27.88 (≫ its in-dist 17.27). SPECTRE's held-out s3 (9.97) is within noise of
  its in-*distribution* s3 (9.19): the abstract ranker barely notices s3 was withheld.
  **Static → adaptive does the lifting** (s3 static 44.27 → adaptive 9.97), exactly the shape of
  the 2026-08-04 shape-generalization finding — the representation alone is not shift-invariant,
  the failure-conditioned re-ranking recovers the win.
- **SB2D: the representation advantage does not reproduce here either.** Held-out b5: PIGINet
  **5.36 ± 0.66** ≈ SPECTRE-adaptive **6.87 ± 1.38** (PIGINet marginally ahead, within seed
  spread); ALL PIGINet 1.68 ≈ SPECTRE-adaptive 2.10. Consistent with the standing in-distribution
  SB2D finding. The adaptive increment still helps (b5 adaptive 6.87 < static 7.37).
- **Honest cross-environment statement:** held-out-stratum generalization reproduces the
  in-distribution and shape-generalization pattern — **the abstract representation wins decisively
  on DD2D and ties (loses marginally) on SB2D**; the adaptive re-ranking is positive on both and
  is what carries DD2D's OOD win.
- **Sanity anchors:** the training-free rows match the deployed numbers exactly — astar
  (DD2D 34.52 ALL / SB2D b5 61.56) and VLMPlan-GPT5.6 (DD2D 35.23 ALL / SB2D b5 22.40) — proving
  the reuse-by-symlink is correct. Read the **headline stratum** (s3 / b5), not the pooled ALL,
  which mixes the held-out with the in-distribution strata.
- **Traps this exercised:** an outer `>` log redirect is evaluated in the caller's cwd (the
  package dir, not the repo root), so a repo-relative redirect silently fails before the script
  runs; two sweeps sharing an arm name interleave their per-arm logs (checkpoints stay separate —
  env-variant is a path component); and uncapped PIGINet CPU processes each grab all cores, so 6
  at once thrash the box (load 94 → cap with `OMP_NUM_THREADS`).

---

<a id="2026-08-09-narrowed-input-variance-selector-noise-fixed-wider"></a>
## 2026-08-09 — Narrowed-input variance is selector noise, fixed by a wider selection window

<!--strip-->
> **id** `2026-08-09-narrowed-input-variance-selector-noise-fixed-wider` · **status**
> active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**Context.** The domain-agnostic scene narrowing
([2026-08-08](#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)) is the right input
surface and is kept, but the retrain regressed on the 3-seed *mean*: DD2D 6.63 ± 0.68 vs baseline
5.78, the whole gap at s1 (7.41 ± 2.94 vs 3.44); SB2D 2.10 ± 0.43 vs 1.69. The probe proved the
removed columns are inference-inert, so "inference-inert yet training-useful" would be incoherent
as an explanation. Two facts identified the real cause as **across-seed optimization variance**,
not information loss: the *best* narrowed seed matched or beat the baseline on both envs (DD2D 6.03,
SB2D 1.62), and the across-seed std jumped ~7× (DD2D 0.10 → 0.68), concentrated at the known
high-variance strata (DD2D s1, SB2D b5). So the narrower input made *optimization* noisier; the fix
had to be a training-process lever that does not touch inputs or architecture.

**Decision.** Two levers were built and tried, both config-gated and off-by-default (existing
training byte-unchanged): **(1) a wider val-selection moving-average window** (`--select-window`,
ma3 → ma5), and **(2) EMA weight averaging** (`--weight-avg ema`, per-step decay 0.999, seeded
post-warmup, with keep-the-better selection so it can never select worse than raw on the val
metric). **The deployed fix is `--select-window 5`.** The `TrainV3Config` default stays 3 so the
frozen baseline's provenance is untouched; the deployed recipe (`spectre_sweep` `v3final` preset,
`sb2d_finalize.sh`) opts in. EMA is kept, tested and one flag away, but is **not** the deployed
lever on either environment.

**Consequences.**
- **`select-window-5` recovers parity, and confirms the diagnosis was selector noise.** DD2D:
  ALL 6.63 → **5.92 ± 0.29** (paired vs baseline Δ+0.14, CI [−0.37, +0.68] — *tied*; vs narrowed
  Δ−0.71, CI [−1.52, −0.05] — *significantly better*); **s1 7.41 → 4.84**, its std 2.94 → 1.03;
  best seed 5.69, *below* baseline. SB2D: 2.10 → **1.84** (vs frozen Δ+0.15, CI includes 0). The
  jittery high-variance model plus the too-short 3-epoch window was locking the selector onto
  unlucky epochs; widening the window de-noises the selection. One training flag, **no deploy-time
  machinery** — the deployed checkpoint is just a better-selected narrowed model.
- **EMA is inert on these two environments.** DD2D EMA arm 6.51 ± 0.60 ≈ narrowed 6.63 (Δ−0.12,
  CI [−0.70, +0.40]); its keep-the-better selector chose *raw* on 2/3 seeds because the EMA weights
  scored *worse* than raw on the val metric (decay 0.999 too slow for even DD2D's ~1500 steps; on
  SB2D's short training likewise). The keep-the-better property held — EMA never selected a worse
  checkpoint — so the arm cost nothing but did not help. Kept in-code (validated by unit + e2e
  tests) because it is sound and a domain with genuine trajectory oscillation may want it; it is
  simply not what this variance needed.
- **Cross-arm comparison is confounded by run-to-run GPU nondeterminism** (accepted; no
  deterministic-algorithm flag). The narrowed/sw5/ema arms are separate 3-seed draws, so single
  cross-arm mean deltas mix the lever with run-to-run noise. The load-bearing signals are therefore
  the *within-run* EMA-vs-raw val comparison (from the keep-better logs) and the *variance* shrink,
  not any one cross-arm mean — which is why the s1 std collapse (2.94 → 1.03) and the CI-includes-0
  vs baseline are what the decision rests on.
- **Deployed numbers move:** DD2D 5.78 → **5.92**, SB2D 1.69 → **1.84** (both tie the frozen
  target-anchored baseline). The frozen baselines (5.78 / 1.69) are the *old target-anchored*
  model and remain the yardstick; the deployed model is now the domain-agnostic narrowed model at
  ma5. `spectre3` compare caches rebuilt with `--force`.

---

<a id="2026-08-08-vlmplan-headline-swapped-gpt-5-6-terra-gripper-geometry"></a>
## 2026-08-08 — VLMPlan headline swapped to gpt-5.6-terra + gripper-geometry disclosure

<!--strip-->
> **id** `2026-08-08-vlmplan-headline-swapped-gpt-5-6-terra-gripper-geometry` ·
> **status** active · **tracks** baselines, evaluation, method, env-dd2d,
> env-stickbutton2d
<!--/strip-->

**Context.** The headline VLMPlan row was **gpt-5.6-luna** and scored poorly — DD2D 62.98
(the *worst* method, worse than the naive planner order) and SB2D 11.85. Two reviewer-obvious
criticisms threatened it as a fair "zero-data corner": (1) luna is the *weaker* GPT-5.6 tier,
so "you only tried a weak model" applies even to the frontier arm; and (2) an input audit of
what the VLM is actually fed found that on **DD2D the prompt never disclosed the gripper's
dimensions** and never drew it — only the qualitative phrase "two-finger gripper" — even though
DD2D feasibility is decided precisely by whether a 2.5×2.0 cm, 0.5–12 cm-aperture gripper can
close on the target past its neighbours. Every *other* method effectively knows this fixed
domain constant (the trained rankers absorb it from feasibility labels); a zero-shot VLM has no
such training, so withholding it is a handicap unique to the baseline.

**Decision.** (1) **Replace luna with `gpt-5.6-terra`** (the stronger tier) as the single
headline `VLMPlan-GPT5.6` row, over the same `openai_responses` backend; luna is dropped (its
cache stays on disk, unreferenced). (2) **Disclose the gripper geometry in the text prompt** —
DD2D states finger size / thickness / aperture / approach-angle count, imported from
`envs/dd2d/dd2d/grasps.py` so they cannot drift; SB2D adds arm-extension and gripper-jaw widths
(its reach limit already carried the operative consequence). Recorded as `PROVENANCE.md`
deviation 9, same class as the operator-semantics disclosures 4/7/8 (removes a handicap, states
nothing about *which* items to stage). (3) **Keep `reasoning.effort: low`** — a 4-problem
low-vs-medium pilot could not discriminate (it drew only easy-mode DD2D problems, all solved at
FP 0), so a **full-scale medium-effort DD2D arm** was run: 33.5 vs low 35.23, paired 95% CI
[−18.6, +15.1] — a wash, with medium slightly *more* censored. Low also matches luna, keeping
the swap a clean model+prompt change.

**Consequences.** terra + the disclosure roughly **halve** FP on both environments:
**DD2D 62.98 → 35.23** (now ~tied with astar 34.52; was the worst method) and
**SB2D 11.85 → 6.42** (self-solves 39/40, 0 censored, and now beats the naive order across all
strata — notably b3 0.90 < astar 2.96, where luna had over-thought). Label-agreement 0.983
(DD2D) / 1.000 (SB2D). The qualitative conclusions are unchanged and now more defensible: DD2D
remains a **negative control** (VLM ~parity with the naive order, still far behind the learned
rankers 5.78–17.27; behaviour is bimodal — 14/40 trivially-graspable targets solved on the
first attempt, staging problems flood 100–200 failing off-pool proposals), and SB2D VLMPlan is
a **genuine planner** that still trails the learned rankers ~3–4×. VLMPlan is a single
generation run → bare mean (like deterministic astar), no across-seed ±; use the notebook's
across-problem bootstrap for a spread. Traps recorded: **a 1-per-stratum DD2D pilot is
unrepresentative** (draws only easy-mode problems, under-estimating FP 0 vs ~27 — pilot on a
representative sample or the whole stratum); and the Responses API takes no fixed seed, so
re-runs genuinely vary. Cost: token-metered (the `/v1/costs` endpoint needs an admin key), on
the order of a few dollars for both envs at low effort. Wiring: `compare.py`
`SEQUENCE_METHODS`/`TIMED_METHODS` repoint `VLMPlan-GPT5.6 → vlmplan_terra`; new
`vlmplan_{dd2d,sb2d}_terra.yaml`; new test `test_dd2d_geometry_discloses_gripper_dimensions`;
`test_frontier_arm_is_registered` updated to `vlmplan_terra`.

---

<a id="2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target"></a>
## 2026-08-08 — Domain-agnostic scene inputs: is_goal replaces is_target, obj_rel narrowed to the anchor-free triple

<!--strip-->
> **id** `2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target` · **status**
> active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**Context.** A cross-environment audit of the v3 input surface found that the *architecture*
is domain-agnostic — `model_v3.py` has no environment name, `dataset_v3.py` has no
`env_variant` comparison, and the DD2D and SB2D deployed flag sets are byte-identical — but
that several **generic-looking scene columns silently carry DD2D semantics and degrade on
SB2D**, with the right shape and no error. `obj_is_target` (set from a DD2D JSON `category ==
"target"` flag) is `≡ 0` on SB2D, where `scene_geometry` leaves it False; and with no target,
`obj_rel`'s target-anchored columns silently change meaning — `[dx, dy, dist]` from
target-relative offsets to absolute world coordinates, `area/target.area` from a ratio to a
raw area. The governing rule (user directive this session): **a feature that is semantically
inapplicable in some environment should not be used at all; a feature that is applicable but
degenerate may stay.** `is_target` and the target-anchored geometry presuppose a *single
distinguished target* — meaningless on a two-target problem or on SB2D's all-buttons goal — so
they go; `cand_overlap` is well-defined everywhere and merely near-constant on SB2D, so it
stays. `concave` goes for a second, stronger reason: it is a deterministic function of the
boundary ring the model already receives, so it is redundant as well as privileged.

Two facts made the change safe to reason about. **(1)** `is_target` was already
`is_goal`, mislabeled: on every DD2D episode `goal_objects(ep) == {o : o.is_target}` (proven
720/720 across all four dd2d_v4 variants and all splits — Gate 1/Gate 2, `notebook/07`
2026-08-08), because the DD2D goal `(extracted target)` names exactly the target. **(2)** There
is no goal tensor and no s₀ atom tensor in the batch; `obj_is_target` was the *only* explicit
goal channel, so the boolean is **replaced**, not deleted.

**Decision.** The deployed v3 scene relation narrows from the width-8 target-anchored vector to
the **width-3 anchor-free triple `[area, sinθ, cosθ]`** (`D_REL_V3 = 3`), and the goal boolean
becomes `obj_is_goal` — 1.0 for any object named by the goal atoms, computed by
`spec.goal_objects`, correct for any number of targets. `ObjectGeometry.is_target` stays in the
schema and in stored data; the v3 tensorizer simply stops reading it, so no re-collection.

Mechanics: `D_REL_V3` lives beside the frozen `D_REL = 8` in `model_v2.py`; `SceneEncoder`
takes `d_rel` per instance, so v2.2 instances stay 8-wide and v3 instances are 3-wide from the
same class. `d_rel` is a `V3Config`/`TrainV3Config` field (**default 3 — narrowing is the
default, not opt-in**) and is persisted, so `load_v3_checkpoint` reloads the right width and a
pre-narrowing checkpoint fails `strict=True` rather than silently scoring the un-narrowed model.
Compat mode (`from_v2_checkpoint_cfg`) keeps `d_rel = 8`, so the frozen v2.2 baseline still
loads and forward-scores exactly. `collate_v2` now allocates `obj_rel` width from the example,
serving both. The shared batch field `obj_is_target` is renamed `obj_is_goal` throughout;
`build_v2_example` keeps computing it from the DD2D-equivalent target flag (byte-identical on
DD2D, its only domain) so the baseline's inputs are untouched.

This deliberately departs from the standing invariant *"a new v3 feature is an additive
zero-initialized branch, never a widened `Linear`"* — that invariant governs **additions**, and
a removal cannot be expressed additively. It is accepted here rather than worked around.

**Consequences.**
- **The removal is free on the deployed model.** A Step-0 inference-time probe zeroed the
  target boolean, the anchored `obj_rel` columns, and both, on the existing deployed
  checkpoints (3 seeds each): **Δ = +0.00 FP, CI [+0.00, +0.00]** on *both* environments and
  *every* stratum (DD2D 5.78 ± 0.10 unchanged; SB2D 1.69 ± 0.26 unchanged) — the deployed
  ranker is completely inert to these columns, consistent with the 2026-08-06 geometry-
  intervention finding. Full numbers in `notebook/07` 2026-08-08.
- **The retrain regressed on the 3-seed *mean* — but as variance, not information loss, and
  it was recovered.** Although the probe shows the removed columns are inference-inert, the
  narrower input made *optimization* noisier: the retrain came in DD2D 6.63 ± 0.68 (vs baseline
  5.78, all the gap at s1: 7.41 ± 2.94) and SB2D 2.10 ± 0.43 (vs 1.69). The best seed matched the
  baseline on both, and the across-seed std jumped ~7× — a training-consistency problem. It was
  fixed by a training-side lever (widening the val-selection window ma3→ma5), which recovers
  parity (DD2D 5.92, CI vs baseline includes 0; SB2D 1.84) and collapses the variance. Full arc
  and the EMA investigation in the follow-up ADR
  [2026-08-09](#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider)
  and `notebook/07` 2026-08-08/09. **On SB2D the boolean also flips from all-zero to a live goal
  channel** (a b5 scene marks all 5 buttons where `is_target` marked none) — an addition the
  removal-only probe cannot price, so SB2D's number is a genuine measurement.
- **The v2.2 rollout-equivalence oracle retires.** A deployed v3 rollout is no longer
  bit-identical to v2.2 by design — `build_v3_example` emits a narrower scene than
  `build_v2_example`. `test_v3_equivalence.py` keeps the guards that still hold: v2.2 loads into
  a compat-mode (`d_rel=8`) `SpectreV3Model`, the shared submodule structure matches at that
  width, and a **forward pass over the same width-8 batch is still bit-identical** — that is the
  plumbing guard the data-path rewrites need. Only the deployed-width rollout comparison is gone.
- **Every pre-narrowing checkpoint (v2.2 and old v3) is unloadable as a deployed v3** (width-3
  vs width-8), by design; they are retrained. `--v2-arm` in `spectre_score_v3` (v2.2 scored
  *through* the narrowed `build_v3`) is incompatible and unsupported after this; the SPECTREv2
  *baseline* is unaffected — it is built entirely through its own v2 path (`SpectreV2Model` +
  `build_v2_example` + `evidence.deployed_rollout_traced`), and its cached 17.27 stands.
- Bundled hygiene, inert for the deployed path (all in `notebook/07`): the `_object_evidence`
  culprit columns gained the `dev_blame` class-2 fallback (else identically zero on any class-2
  env), its depth normaliser dropped the hard-coded DD2D `/8.0` for the episode's own max plan
  length, and the frame-extent lookup now raises instead of silently falling back to `scale=1`.
- **Deferred, recorded not skipped:** scale-invariance of `area`/`obj_boundary` (`/scale²`,
  `/scale`) — the Step-0 probe *kept* `area` and `boundary`, so rescaling them is unmeasured and
  would confound the clean retrain; and loud OOV/truncation on the v3 path.

---

<a id="2026-08-06-shape-generalization-s2-deficit-collection-variance-shape"></a>
## 2026-08-06 — Shape-generalization s2 deficit is collection variance, not shape size or geometry representation

<!--strip-->
> **id** `2026-08-06-shape-generalization-s2-deficit-collection-variance-shape` ·
> **status** active · **tracks** method, evaluation, env-dd2d
<!--/strip-->

**Context.** The shape-only set
([2026-08-04](#2026-08-04-shape-only-dd2d-gen-variant-precompute---test-variant)) showed
SPECTRE-adaptive **s2 = 17.27** vs in-dist 10.49, and its §2b wall-clock was the worst of any
method. A hypothesis (from planner-inspector viewing) was that the new tee/cross figures are
**bigger** — harder to fit the buffer — so they degrade s2. Two facts had to be established
before that could be tested: whether the effect is physical (packing) or representational (how
SPECTRE encodes the shapes), and **whether v3 even reads geometry**. It does — `build_v3_example`
*requires* `scene_geometry` and the deployed `SceneEncoder` consumes, per object, a footprint
point-set encoding of the boundary ring, pose, raw `o.area`, area-relative-to-target and a
concave flag; those scene tokens are cross-attended by every candidate. So "image-free" is not
"geometry-blind", and the size hypothesis was a live, testable representational claim.

**Decision.** Test it three ways, cheapest first, and let the data pick the framing
([notebook 2026-08-06](../notebook/07-stickbutton2d.md#2026-08-06-dd2d-shape-size-sweep-geometry-interventions-size)):

1. **Stored-data gate** (no compute): by convex-hull footprint (the packing-relevant measure —
   nothing packs into a concavity) tee/cross rank 5th–6th of 9; place-buffer *volume* failures
   are 5.3% of all failures; buffer hull-occupancy is ~40% even for feasible candidates and
   *lower* for infeasible ones. Physical packing is not the binding constraint.
2. **Inference-time input interventions** (new `spectre_intervene_geometry.py`): rewrite ONLY
   the tee/cross model-input geometry of the shape-only episodes — area→hull (`hullarea`),
   boundary→convex-hull (`hullshape`), or shrink ×0.7 (`scale07`) — holding skeletons + outcomes
   (true feasibility) fixed, and re-score the *same* dd2d_v4 checkpoint. All three are **inert to
   the digit** (adaptive FP 6.77 / s2 17.27, paired bootstrap +0.00 on every problem; astar
   control byte-identical). SPECTRE's ranking does not use the new shapes' geometry input.
3. **Physical shrink + variance control**: a `--family-size-scale` collector lever (new) collects
   `dd2d_v4gen_shapeonly_sz07` (tee/cross ×0.7). It appears to help hugely (s2 3.17), but a
   **fresh un-shrunk control** (`dd2d_v4gen_shapeonly_fresh`, band 7) reads s2 5.63 — *below*
   the in-dist 10.49 — while **astar s2 is stable at 14–15 across all three collections**. So the
   apparent improvement is collection variance, not size: the gap between the two *un-shrunk*
   collections (17.27 vs 5.63) dwarfs the shrink's residual (5.63 vs 3.17).

Conclusion: **the s2 shape-generalization deficit is a pool-composition / collection-variance
artifact, not shape size and not SPECTRE's geometry representation.** The `17.27` that motivated
the investigation is a high-variance draw of an s2 stratum with ~1.5 unique feasible solutions
(the [2026-08-02](#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact)
finding), on which SPECTRE's learned order — but not the geometry-free astar order — is sensitive
to which solutions land in the k=200 pool.

**Consequences.**
- **Invariant added — read a shape/size-generalization number against a fresh un-shrunk control,
  not a single collection.** s2 is variance-dominated (three draws: 17.27 / 5.63 / 3.17 at fixed
  astar ~14); a single-collection s2 point estimate is not a model signal. This is the
  [2026-08-02](#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact)
  caveat, now shown to bite the shape-only set too.
- **Inference-time geometry interventions are the right tool for "does the model use feature X?"**
  — they hold the problem + labels fixed and rewrite one input, so any FP change is purely the
  model's response. The astar arm (reads only labels) is the built-in null control. Reusable via
  `spectre_intervene_geometry.py` (`hullarea|hullshape|scaleNN`) + `precompute_dd2d_cache.py
  --test-variant`.
- **v3's geometry channel is weakly weighted for ranking.** Consistent with the 2026-07 finding
  that coverage/waste + records carry v3; the footprint/pose/area features are present but the
  deployed ranking is inert to perturbing them for the new shapes (a tee is the most OOD to the
  footprint encoder — kNN-to-train 0.105 — yet that does not move the ranking).
- **New code, all off-by-default / additive:** `shapes.sample_shape(size_scale=)` threaded through
  `scene`/`problem`/`collect` + a `--family-size-scale` CLI flag; `spectre_intervene_geometry.py`;
  `spectre_probe_shape_geometry.py`; a `dd2d_gen_shapeonly_sz07` compare_methods EnvSpec whose
  caveats state size is not the driver; four `DOMAINS` + `_PIGINET_PATHS` variant registrations
  (all resolve to `_DD2D`; a rescale/hull-rewrite is geometry metadata, not a new schema, so the
  dd2d_v4 vocab is reused with no OOV).
- **Does not touch the deployed method or any published number.** Purely diagnostic; the v3
  checkpoint, loss, and headline FP are unchanged.
- **Convention change (2026-08-06 addendum): tee/cross DEFAULT to 0.7x linear**
  (`shapes._FAMILY_DEFAULT_SCALE`; explicit `size_scale` *overrides*, does not stack, so
  `--family-size-scale tee=1.0` restores nominal). This is a **design** choice, not a
  size-drives-FP claim (the interventions above show the ranking is inert to the shapes'
  geometry): at 0.7x the tee/cross hull footprint (~29/33) grasps and packs cleanly in the
  shallow buffer while staying unseen at test time. The `compare_methods.py` object-gen env
  (`dd2d_gen_shapeonly`) now reads a 0.7x collection (the `_sz07` draw) and the separate
  `dd2d_gen_shapeonly_sz07` dropdown was folded into it. tee/cross are held out of training, so
  no retraining. **Read the object-gen numbers as the ALL win over PIGINet (robust across
  draws), not the single-draw s2** — s2 stays variance-dominated at any size.

---

<a id="2026-08-04-shape-only-dd2d-gen-variant-precompute---test-variant"></a>
## 2026-08-04 — Shape-only DD2D gen variant + precompute --test-variant

<!--strip-->
> **id** `2026-08-04-shape-only-dd2d-gen-variant-precompute---test-variant` ·
> **status** active · **tracks** env-dd2d, evaluation, method, tooling
<!--/strip-->

**Context.** The DD2D generalization test
([2026-08-01](#2026-08-01-dd2d-generalization-test-unseen-count-unseen)) used
`dd2d_v4gen_shape`, which changed **two** variables — an unseen 13–15 blocker count *and* the
new tee/cross figures. Its s2 FP degraded sharply; that was later characterized as a
**count-driven pool-composition artifact**
([2026-08-02](#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact)),
not the shapes — but the confound meant the shape effect itself was never measured cleanly. A
clean attribution needs the shapes isolated from count. Two things blocked rendering it in the
comparison notebook: there was no such collection, and `precompute_dd2d_cache.py` bakes one
`env_variant` into both the checkpoint paths and the episode/output paths and rejects unknown
variants, so it could not write a `compare_cache` for a train-old / test-new set the way
`spectre_score_v3.py --test-variant` already scores one.

**Decision.** Three parts.

1. **A shape-only held-out variant, `dd2d_v4gen_shapeonly`.** The tee/cross figures forced
   into every scene (`--shape-set augmented --require-families tee,cross`) at the **trained
   9–12 blocker count** — achieved by *omitting* `--n-items-*` so the collector's default
   count mechanism (10–13 items, no realized floor) is byte-identical to dd2d_v4's. 40 test
   problems stratified s0–s3, seed band [5M,6M) disjoint from train/val/test and the count/shape
   gensets. Reproducer script `collect_dd2d_shapeonly.sh`; conf `dd2d_v4gen_shapeonly.yaml`;
   `DOMAINS["dd2d_v4gen_shapeonly"] = _DD2D` (a shape family is geometry metadata, so no new
   schema and the dd2d_v4 vocab is reused — no OOV).
2. **`precompute_dd2d_cache.py --test-variant`**, mirroring `spectre_score_v3.py`'s convention:
   `--env-variant` is the TRAIN/checkpoint collection, `--test-variant` the TEST/episode one.
   `_configure_paths` gained a `ckpt_variant` argument and a `CKPT_VARIANT` global; the vocab,
   SPECTRE/PIGINet checkpoints, `_v3_ckpt`'s env component and `spec_for` now key off the train
   variant, while the test split, `compare_cache`, `N_PROBLEMS`, refine cap and PIGINet
   data/CLIP-cache key off the episode variant. A `_PIGINET_PATHS["dd2d_v4gen_shapeonly"]` entry
   points its data + fresh CLIP cache at the gen collection (checkpoint from the train entry).
   The same-collection path (no `--test-variant`) is byte-identical to before.
3. **A native `EnvSpec` (`dd2d_gen_shapeonly`)** in `compare_envs.py` (reusing DD2D's scene
   renderer, plan formatter and stratum meaning), so the notebook renders the full FP + §2b
   wall-clock layout for the gen set with no notebook-cell edits.

**Consequences.**

- **The shape effect is now isolated and small.** SPECTRE-adaptive ALL 6.77 ± 0.81 vs in-dist
  5.78 (~1.17×), against the count+shape set's 11.26 (~1.9×); s3 *improves* (9.19→6.03) and s2
  degrades only moderately (10.49→17.27) rather than to ~32. This **confirms** the 2026-08-02
  attribution: the severe s2 OOD degradation was primarily count/pool-composition, not the new
  figures. Numbers and the wall-clock inversion are in
  [notebook/07 2026-08-04](../notebook/07-stickbutton2d.md#2026-08-04-dd2d-shape-only-generalization-shapes-isolated-count).
- **The two scoring instruments agree exactly** (compare cache vs `spectre_score_v3.py`: 6.77 ±
  0.81, s0–s3 identical, paired bootstrap vs astar −24.68 CI [−41.95, −8.88]), which is the
  cross-check that the `--test-variant` split loads the right checkpoints against the right
  episodes.
- **The representation-vs-adaptivity attribution is not shift-invariant.** In-dist the static
  abstract representation carries ~73% of the margin; under the unseen-shape shift SPECTRE-static
  (22.55) falls *behind* PIGINet (15.27) and only the adaptive re-ranking recovers the win — so a
  cross-environment/OOD claim must say *which* component it credits.
- **Onboarding another train-old/test-new gen set is now one `_PIGINET_PATHS` entry + one
  `DOMAINS` line + one `EnvSpec`**, no per-run plumbing. `--test-variant` is the reusable lever.
- Collection remains `PYTHONHASHSEED`-dependent, so the raw `data/dd2d/raw_v4gen_shapeonly` dir
  is the authoritative record (archive it); a re-collection draws a fresh sample.

---

<a id="2026-08-03-sb2d-2b-wall-clock-breakdown-parity-dd2d"></a>
## 2026-08-03 — SB2D §2b wall-clock breakdown at parity with DD2D (per-env 10 s refine cap)

<!--strip-->
> **id** `2026-08-03-sb2d-2b-wall-clock-breakdown-parity-dd2d` · **status** active ·
> **tracks** method, evaluation, env-stickbutton2d, tooling
<!--/strip-->

**Context.** The §2b wall-clock-to-first-success breakdown (plan-gen + inference + refinement, per
method × stratum, capped + uncapped) was complete on DD2D and a stub on SB2D, gated behind
`EnvSpec.has_timing`. It was deferred *by choice, not by data*: SB2D episodes already carry a real
per-candidate `refinement_wall_clock_s` (the shared collector in `collect.py` times every refine),
the precompute (`precompute_dd2d_cache.py`) is env-parameterized, and the `compare.py` loaders are
env-agnostic. Three concrete gaps blocked it: the refine cap `REFINE_CAP_S = 2.0` is a DD2D
constant that censors SB2D's seconds-long feasible refines; SB2D plan-gen was unmeasured
(`_measure_plan_gen` returned `{}` for non-DD2D); and the live SB2D env (`SB2D_KINDER`) grafts
SPECTRE from the `stickbutton2d_v1` legacy cache, but `load_time_records_per_seed` reads only the
primary cache.

**Decision.** Enable §2b on `SB2D_KINDER` with these changes:

- **Per-env refine cap.** `REFINE_CAP_S` becomes a per-variant lookup rebound in
  `_configure_paths` — `dd2d_v4: 2.0`, `stickbutton2d_v1{,_kinder}: 10.0`, default 2.0. The two
  SB2D variants **must share one value**, because the kinder §2b grafts SPECTRE timing from the v1
  cache and the capped fields have to be computed under the same cap.
- **SB2D cap = 10 s, set empirically.** DD2D's 2 s sits above the *whole* feasible distribution
  (p95 0.44 s). SB2D's feasible refines run to seconds (per-candidate p95 10.6 s), so no
  cap-above-the-distribution fits under the 20 s budget. 10 s instead clears the worst *per-problem
  fastest-feasible* (max 8.84 s) with margin — `_feasibility_at_risk(10) = 0`, no problem censored
  — while cutting the many budget-exhausting failures (33 % of all per-candidate refines exceed
  it). So SB2D's cap sits *inside* the feasible distribution, a different regime from DD2D's.
- **SB2D plan-gen measured.** `_measure_plan_gen` dispatches to an SB2D branch that, per stratum,
  rebuilds the kinder env and times the acyclic pool draw via a new env-agnostic helper
  `collect.time_pool_generation` (mirrors `collect_episode`'s setup up to — and times only — the
  `islice` pool draw). Config mirrors `sb2d_collect._config` so the timed pool is the collected
  pool.
- **Timing graft.** New `compare.merge_time_records` (the timing analog of `merge_collections`)
  grafts the `legacy_only` methods' timing from the legacy cache; the notebook loads timing from
  both primary and legacy and merges. A no-op on DD2D (its `legacy_only` VLMPlan-32B is not in
  `TIMED_METHODS`).

Both caches rebuilt under the cap: `--env-variant stickbutton2d_v1 --methods spectre3
--no-ablations --force` and `--env-variant stickbutton2d_v1_kinder --methods astar piginet
--force`.

**Consequences.**

- §2b renders on SB2D for all methods × {b1,b2,b3,b5,ALL}, capped + uncapped, with a non-zero
  plan-gen. FP headline byte-unchanged after the `--force` rebuild (adaptive 1.69 / static 1.98 /
  PIGINet 2.28). DD2D §2b is unchanged (its 2 s cap and numbers preserved; the graft is a no-op).
  Numbers and the finding are in
  [notebook/07 2026-08-03](../notebook/07-stickbutton2d.md#2026-08-03-sb2d-2b-wall-clock-spectre-adaptive-fastest-per-env).
- **The DD2D cap narrative does not transfer.** On SB2D all failures are uniformly expensive (run
  to the 20 s budget), so FP and wall-clock are aligned: SPECTRE-adaptive is fastest capped (11.2 s)
  *and* uncapped (14.0 s), the cap does not flip the ranking, and it helps the **highest-FP** method
  (astar, −48 s) most — the reverse of DD2D, where the cap most helped the low-FP learned ranker.
- **The SB2D cap is a real trade, not near-free accounting.** Because it sits inside the feasible
  distribution, it abandons slow non-fastest feasibles and costs the learned methods **+0.3 FP**
  (adaptive 1.69 → 2.03) — an order of magnitude more than DD2D's +0.05. The safety guard
  (`_feasibility_at_risk`) is what keeps that trade from ever turning a solved problem censored.
- `collect.time_pool_generation` and `compare.merge_time_records` are env-agnostic, so a third
  environment gets §2b by adding its variant to the `_REFINE_CAP_S` map and running the precompute
  — no new timing code.

---

<a id="2026-08-03-frontier-vlm-vlmplan-arm-gpt-5-6-luna-kinder-labeled"></a>
## 2026-08-03 — Frontier-VLM VLMPlan arm (gpt-5.6-luna): kinder-labeled SB2D image, wall-clock, inspector

<!--strip-->
> **id** `2026-08-03-frontier-vlm-vlmplan-arm-gpt-5-6-luna-kinder-labeled` ·
> **status** active · **tracks** baselines, evaluation, method, env-stickbutton2d,
> env-dd2d, tooling
<!--/strip-->

**Context.** VLMPlan — the zero-data corner of the representation grid — was a **local Qwen3-VL**
arm (8B/32B) served through an OpenAI-compatible endpoint. A reviewer can dismiss a weak local
model, so the headline VLMPlan row should be a **frontier** VLM. Separately, VLMPlan lacked parity
with the SPECTRE/PIGINet/astar rows in three ways: it was absent from the §2b wall-clock section
and the §5 planner inspector, and the inspector's plan formatter was DD2D-hardcoded so every SB2D
row printed `retrieve ?`. Finally the SB2D pixel question: the representation contrast is measured
on `stickbutton2d_v1_kinder` (kinder's own pixels, PIGINet-parity), but kinder draws every
unpressed button as an **identical unlabeled red disc**, so those pixels are unusable by a VLM as-is.

**Decision.**

- **Model = `gpt-5.6-luna` over the OpenAI Responses API** (`backend: openai_responses`,
  `reasoning.effort: low`, `max_output_tokens 16384`). GPT-5 reasoning models reject
  `max_tokens`/`temperature` over chat completions; the Responses backend remaps the cap and drops
  `temperature`/`seed`, so round diversity comes from the growing repeat-suppression block, not a
  seed. `luna` (the economy tier) is deliberate — these problems are easy for a human and it keeps
  spend ~$1–2 for the whole study. It is the **headline** VLMPlan row (`VLMPlan-GPT5.6`,
  `cache_subdir vlmplan_luna`); the Qwen arms are kept for a local-vs-frontier contrast.
- **SB2D image = `image_source: kinder_labeled`** — kinder's real env render with Set-of-Mark
  object labels overlaid in data coordinates (via kinder's `ax_callback`, so labels sit exactly on
  the objects). This is deviation 3 (`prompts/PROVENANCE.md`) applied to the second environment:
  labels are load-bearing, not an advantage, because the names appear nowhere in an unlabeled
  render. No reach line is drawn (the table band + text prompt convey it). DD2D keeps its schematic
  labeled renderer. **Both envs now persist the exact scene image sent, to
  `…/vlmplan/<run>/images/<pid>.png`.**
- **Scope = stratified 40/env (10/stratum)**, selected by *striding within each stratum band*
  (`runio._stratified`), never `n_problems=40` (which is one stratum — the stride-never-truncate
  trap). DD2D runs **native on `dd2d_v4`** (aligned with the pool methods), replacing the v3 graft
  for the headline row; SB2D runs native on `stickbutton2d_v1_kinder`.
- **Wall-clock for VLMPlan** = `infer_s` (VLM generation to first success, summed per-round
  `api_s`) + `refine_s` (the collection's stored per-candidate time for an in-pool attempt; the
  run-captured live-refine time for an off-pool one — captured once in the first-success stop
  check, never re-refined at score time), with a capped variant under `REFINE_CAP_S = 2 s`.
  VLMPlan joins `TIMED_METHODS`; `build_time_table` **zeroes the pool `plan_gen_s`** for a sequence
  method, because the VLM's generation *is* its plan-gen (`total = infer_s + refine_s`).
- **Inspector** = VLMPlan is selectable and renders its **own** ordered attempts from the cache
  record (off-pool by design); the DD2D-hardcoded `_plan_label` is replaced by an
  `EnvSpec.plan_label` hook (DD2D `stage {…} → retrieve N`; SB2D the press order), which is what
  fixes `retrieve ?`. The full step sequence is now stored per VLMPlan attempt so the plan renders.

**Consequences.** A clean cross-environment story, consistent with the pilots and the direction the
Qwen arms already showed (see [notebook/07 2026-08-03](../notebook/07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong)
for the per-stratum numbers): the frontier VLM is a **genuine planner on SB2D** (self-solves the
large majority of problems, low FP, correctly grounds buttons from the labeled kinder image) but
**struggles on DD2D**, the proposal's declared *negative control* for continuous packing — it
systematically over-stages and its proposals fail the geometric refinement, exactly as the Qwen
arms did (qwen32b s3 ≈ 69). Two method-level findings this exercised: **VLMPlan wall-clock is
generation-dominated** (the Responses round-trip + reasoning is seconds–minutes, dwarfing the
sub-second refinements), which is the honest cost of a zero-shot frontier planner; and the
label-agreement gate read **1.0 on both envs**, so the numbers are defensible. Reproduce with
`--config-name vlmplan_{dd2d,sb2d}_luna`; the response cache makes a re-score free.

---

<a id="2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration"></a>
## 2026-08-02 — Per-candidate refinement cap is the deployed wall-clock configuration

<!--strip-->
> **id** `2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration` ·
> **status** active · **tracks** method, evaluation, env-dd2d
<!--/strip-->

**Context.** The §2b DD2D wall-clock table showed SPECTREv3-adaptive *slower* overall than
the naive planner order (5.89 vs 4.94 s ALL to first success), with the entire gap at s1
(11.99 ± 7.81 vs astar 0.26 s). The diagnosis
([notebook/07 2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate))
is real, not a measurement bug: v3's s1 FP (3.44) is modestly *worse* than astar's (2.24) —
the planner-cost order already ranks s1's short/cheap feasible plans well — and that ~1.2-
attempt FP gap becomes a ~46× wall-clock gap because of *which* candidates each method fails
on. Feasible refinements finish in <0.5 s (p95 0.44 s); the waste is entirely near-feasible
infeasible candidates that burn the full **20 s** refinement budget. astar's s1 failures are
cheap dead-ends (~0.06 s); v3's few extra failures are the 20 s traps. This is "FP flatters
the learned ranker" running against v3 — FP alone hides it.

**Decision.** The **deployed wall-clock configuration is a per-candidate refinement-
abandonment cap** `REFINE_CAP_S = 2 s`: each skeleton is refined for at most C seconds before
the deployment moves to the next in the ranked order; a candidate not refined within C is
abandoned and treated as a failure. Load-bearing choices:

- **Per-candidate, never per-problem.** A per-problem total budget can starve a solvable
  problem (spend it on traps, never reach the feasible skeleton); a per-candidate cap only
  skips the slow *skeleton*. A problem is lost only if *every* feasible candidate exceeds C —
  measured **0/100** on dd2d_v4 (min-feasible refine time per problem: mean 0.103 s, **max
  0.243 s**). Precompute logs this at-risk count (`_feasibility_at_risk`) so a future
  collection where it is non-zero is caught, not silently censored. Provably lossless with an
  iterative-deepening fallback (exhaust the pool at C; if nothing refines, retry uncapped) for
  any domain with slow-feasible plans; it never fires on dd2d_v4.
- **C = 2 s** ≈ 4.5× the feasible p95, so only genuine near-feasible outliers are cut.
- **The cap faithfully shifts FP, so it is re-run, not accounted.** A slow-feasible candidate
  ranked first is abandoned (FP + 1), and for the *adaptive* rollout it enters the failure
  context and re-ranks the rest — so the order diverges (0/100 astar, 6/300 piginet, 4/300
  static, 6/300 adaptive at C=2 s). `deployed_rollout_v3_traced(refine_cap_s=…)` redefines the
  stopping-success set as `outcome==success and time ≤ C` and re-runs; the fixed-order methods
  derive capped FP/refine on their score-order (`_fp_and_refine_capped`). `min(t, C)` on the
  uncapped stored sums would be silently optimistic on exactly those cells.
- **The published FP headline (§1/§2) stays uncapped** at the pool-cap budget — the metric of
  ranking quality is unchanged. The cap is a *wall-clock* deployment configuration; §2b owns
  it and prints the capped-FP delta beside the table.
- The cap applies to **all four pool-ranking methods** (astar-dist, PIGINet, SPECTREv3-static,
  SPECTREv3-adaptive) — a shared-refiner policy, so fairness requires it. It is a test-time
  accounting change: **no retraining**, checkpoints reused as-is.

**Consequences.** Under the 2 s cap, SPECTREv3-adaptive is the **fastest** method — 1.79 ± 0.44 s
ALL vs astar 2.96, PIGINet 3.14, v3-static 2.53 — its s1 collapses 11.99 → 2.40 and it wins s2
(1.88) and s3 (2.45) decisively. The cap's **FP cost is tiny**: adaptive +0.05 (5.78 → 5.83),
astar +0.00 (failures already sub-cap), PIGINet +0.23, static +0.26 — while cutting adaptive's
wall-clock 3.3×. This is the honest resolution of "FP flatters the learned ranker": the ranker's
value (try few candidates) shows in wall-clock only once each failed try is bounded, because the
cap targets exactly the expensive failures the ranker still makes. **DD2D-only** (SB2D's kinder
`BacktrackingRefiner` records no per-candidate times; `EnvSpec.has_timing` gates the section).
Reproduce with `precompute_dd2d_cache.py --env-variant dd2d_v4 --force` (writes
`refine_s_capped`/`fp_capped` per record + `refine_cap_s` in `meta.json`) then read §2b. New
code: `REFINE_CAP_S` + `_fp_and_refine_capped` + `_feasibility_at_risk` in
`precompute_dd2d_cache.py`; `refine_cap_s` + `V3Trace.refine_capped_seconds` in `inference_v3.py`;
`load_refine_cap_s` + `build_time_table(use_capped=…)` in `compare.py`; `test_refine_cap.py`.
The **residual s1 gap** (v3 2.40 vs astar 0.26) is the modest s1 FP deficit and is a candidate
for the model-side R1 cost/enumeration-index feature.

---

<a id="2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new"></a>
## 2026-08-02 — Kinder-rendered PIGINet crops for StickButton2D via a new env_variant

<!--strip-->
> **id** `2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new` · **status**
> active · **tracks** baselines, evaluation, env-stickbutton2d, data, tooling
<!--/strip-->

**Context.** For the representation contrast to be fair, the pixel input a *model* consumes
should come from the environment's own renderer, not an approximation. On SB2D the only
model reading pixels is **PIGINet** (SPECTRE is image-free: its `SceneEncoder` consumes
vector `scene_geometry` — boundary polygons + poses — read from kinder's
`object_to_multibody2d`, so it is already kinder-native). PIGINet's SB2D crops, though, were
produced by a **schematic** rasteriser (`SB2DDomain.crops`): each object drawn as a lone
polygon on a blank background, with no scene context. DD2D is unaffected — it is not a
kinder env and already renders PIGINet crops from its own env renderer. This is SB2D-only.

**Decision.** Route PIGINet's SB2D pixels through **kinder's built-in renderer**, delivered
as a new env_variant **`stickbutton2d_v1_kinder`** built by *converting*
`stickbutton2d_v1`, not re-collecting it. Five choices are load-bearing:
- **Reconstruct, never regenerate — with the sanctioned exception.** The converter
  (`experiments/spectre/sb2d_render_convert.py`) copies every record **verbatim** (plans,
  timings, outcomes, geometry) and only re-renders the pixels, by resetting the env from the
  stored seed (`env.reset(seed=problem_id)`). That reset is the one sanctioned exception to
  the rule (it is deterministic on SB2D; the same reset backs `vlmplan/sb2d_label.py`). Only
  `provenance.env_variant` changes in the record.
- **Per-object crops from the true scene, not a whole-scene embedding.** Each crop is a
  native `render_2dstate` window (world side = the adapter's `_CROP_WORLD`) centred on the
  stored object pose, so it keeps PIGINet's per-object CLIP channel *and* now carries real
  local context (neighbours, stick, table band, wall) that the schematic discarded. A full
  `scene.png` is materialised alongside for possible future use (no consumer wired).
- **No schema change.** Crops live at `raw/<variant>/<split>/images/<pid>/<obj>.png`, a path
  the reader reconstructs from the pid — so `EpisodeRecord` gains no image field and needs no
  migration shim. The reader is a thin `SB2DKinderDomain(SB2DDomain)` overriding only
  `crops()`; `make_sb2d_domain(data_root, variant)` dispatches on variant, keeping the
  schematic as the documented secondary.
- **SPECTRE is grafted, not retrained.** Because the records are byte-identical and SPECTRE
  is image-free, its numbers cannot differ; the comparison notebook grafts SPECTRE (and
  VLMPlan) from `stickbutton2d_v1` via `EnvSpec.legacy_only`, and only PIGINet (+ the cheap
  deterministic astar) is native to the kinder cache. Retraining would add training noise,
  not signal.
- **Kinder does not manufacture signal it cannot have.** Two unpressed buttons are identical
  red discs in the real env too, so the image channel stays partly degenerate; the win, if
  any, is the positional context the crop now carries, not disc appearance.

**Consequences.** The seam turned out to be one function — `domain.crops` — so model, loss,
tokenizer and CLIP cache are untouched; the change is a converter + a subclass + a variant
row. **The re-run reinforced the standing finding rather than overturning it.** PIGINet
retrained on kinder crops (3 seeds, same weighted-bce/40-epoch recipe) reads **2.28 ± 0.29 FP
ALL** — *slightly worse* than the schematic's 2.02, the whole drop at b5 (7.55 vs 6.39). The
paired bootstrap still does not separate: v3-static − PIGINet = −0.31, CI [−0.95, +0.36];
v3-adaptive − PIGINet = −0.60, CI [−1.24, +0.08]; the adaptive increment holds (−0.29, CI
[−0.51, −0.08]). So "the representation advantage does not reproduce on SB2D" survives the
validity fix, and the pre-registered caveat held — the crop's added context is positional and,
since unpressed buttons are identical discs in the real env, net-neutral-to-mild-distractor.
Full numbers in [notebook/07 2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s).
The schematic `stickbutton2d_v1` stays as the secondary/baseline, so the two are never
silently mixed. One kinder-internal coupling was accepted
(`env.unwrapped._object_centric_env._current_state`, mirroring `base_env.render()`), with a
public fallback and a determinism test guarding it.

---

<a id="2026-08-02-wall-clock-to-first-success-added-compare-methods-reuses-stored"></a>
## 2026-08-02 — Wall-clock-to-first-success added to compare_methods; reuses stored refine times

<!--strip-->
> **id** `2026-08-02-wall-clock-to-first-success-added-compare-methods-reuses-stored`
> · **status** active · **tracks** evaluation, tooling, env-dd2d
<!--/strip-->

**Context.** `compare_methods.py` reported only FP (failed attempts before first success). FP
treats every failed attempt as equal cost, but a DD2D failed refinement ranges ~15 ms (a dead-end)
to ~20 s (budget-exhausted), so FP cannot say whether a method's inference cost is *worth it* in
wall-clock. We added a wall-clock-to-first-success metric = abstract-plan-generation + inference +
refinement.

**Decision.** A new **complementary** metric (FP stays the headline), computed so the cross-method
comparison is fair and the result is durable:
- **Refinement time is reused, not re-run.** The dd2d_v3/v4 refiner stores per-candidate
  `refinement_wall_clock_s`; each method's refine-to-first-success is that summed along its own
  attempt order (adaptive = the cached `order`; static = `argsort(-scores)`). Every method sums the
  *same* per-candidate times over its own ordered subset, so the comparison is fair even though the
  absolute seconds are a within-collection relative measure (collector 8-way parallelism, 20 s
  budget).
- **Inference is measured on GPU** (the deployment-realistic path; `~22 ms/step`, CPU-tensorize +
  GPU-forward, tensorization-dominated), via an `infer_seconds` field on `V3Trace`.
- **Plan-gen is a per-stratum shared constant** (identical pool for all four pool-ranking methods),
  measured by regenerating a few problems per stratum and timing the astar top-k enumeration.
- **All three are persisted** in the compare cache (`refine_s`/`infer_s` per record; per-stratum
  `plan_gen_s` in `meta.json`) — measured once at `--force` cache build, reused at render, never
  recomputed. Scope: the four pool-ranking methods (astar-dist, PIGINet, SPECTREv3-static/adaptive)
  on DD2D; gated by `EnvSpec.has_timing` (SB2D's refiner stores no per-candidate times). The FP
  table is byte-identical after the rebuild (timing fields are additive; scores/FP deterministic).

**Consequences.** The headline finding is that **FP flatters the learned ranker**: SPECTREv3-adaptive
has 6× lower FP than astar (5.8 vs 34.5) but is not faster in wall-clock (5.90 vs 4.94 s ALL),
because astar's many failures are cheap dead-ends (~0.14 s) while SPECTRE's few failures are the
expensive *near-feasible* candidates it correctly ranks high (~0.89 s) — a better ranking surfaces
the costlier failures. Inference is the small term (0.03–0.51 s); the learned ranker's wall-clock
advantage is concentrated at s3 (astar's failure *volume*) and is net-negative at s1/s2. Numbers +
per-stratum breakdown + noise caveats in [notebook/07
2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters).
**Standing implication:** an FP margin on DD2D should not be read as a proportional wall-clock win;
quote the wall-clock section alongside it.

---

<a id="2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact"></a>
## 2026-08-02 — s2 generalization degradation characterized as pool-composition artifact; regen for pair-diversity rejected

<!--strip-->
> **id**
> `2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact` ·
> **status** active · **tracks** env-dd2d, evaluation, method, data
<!--/strip-->

**Context.** The [2026-08-01 generalization test](#2026-08-01-dd2d-generalization-test-unseen-count-unseen)
reported v3's s2 FP degrading 10.49 → 30.23 under the unseen-count shift, framed in that entry's
consequences as v3's "already characterized in-distribution s2 weakness." An objection — s2 (clear
2) cannot be intrinsically harder than s3 (clear 3) — prompted a read-only diagnosis
([notebook/07 2026-08-02](../notebook/07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model)).
The objection is correct: intrinsic/execution difficulty is monotone (astar-dist FP s3 167 ≫ s2
28; generation keep-rate s3 20% ≪ s2 91%; s2 labels 100% sound). Only the *model's* FP inverts,
and it does so for a reason that is neither a model-generalization failure nor a generator bug.

**Decision.** **Root cause = a pool-composition artifact sitting on top of low s2 solution
diversity; characterize it, do not re-engineer.**
- s2 problems have only **~1.5 unique feasible solutions** (feasible pairs). 99% of feasible
  triples are redundant supersets of those pairs (genuine-3 ≈ 0). The circular target admits 18
  diametric grasp axes; an axis opens only when its antipodal blocker pair is cleared, and
  `crowd=5` (odd) yields no antipodal pair.
- In-distribution, the k=200 pool pads those ~1.5 solutions with ~23 redundant feasible triples
  (92 triples enumerated) → 26 feasible → the ranker finds one in ~3 tries. At 14 blockers,
  C(14,2)=91 pairs flood the short-first cap (→172 pair candidates) and crowd the triples out
  (→18 enumerated, 1.1 feasible) → ~2.9 feasible → FP ~30. So the OOD number exposes the true
  low-diversity difficulty that pool padding hid in-distribution (model FP corr(feasible count)
  = −0.82).
- **A generator redesign for substantive feasible-pair diversity was explored and rejected as
  geometrically blocked.** The obvious lever — even collar count so antipodal pairs each open an
  axis — does not work empirically (generator sweep: crowd 5/6/8/10 → ~1.5 feasible pairs) and
  pushes problems to mfs=3: keeping mfs≥2 requires blocking the circular target from all 18 axes,
  which is exactly the coverage that prevents a single removed pair from cleanly opening one axis.
  Any real regen would also imply re-collecting train/val/test + retraining, re-baselining every
  existing SPECTRE result — a large cost against an uncertain geometric payoff.

**Consequences.** The s2 column of the generalization table — and the ALL mean it dominates — is
**confounded by pool composition, not a clean model-generalization signal**, and is recorded as
such (this entry, the notebook entry, the `CLAUDE.md` DD2D-generalization section, and
`proposal.md` §6). The **s3 column is the clean signal**: s3 was already feasible-scarce in
training, so OOD s3 is in-regime and v3 improves there (9.19 → 4.87) while astar stays pathological
— i.e. v3's advantage over the planner order does generalize where the feasible regime is stable.
This entry **refines** the s2 interpretation in the
[2026-08-01 generalization ADR](#2026-08-01-dd2d-generalization-test-unseen-count-unseen) (which
attributed s2 to model weakness); the numbers there are unchanged, the attribution is corrected
here. No code or data changed.

---

<a id="2026-08-01-dd2d-generalization-test-unseen-count-unseen"></a>
## 2026-08-01 — DD2D generalization test — unseen count and unseen shapes

<!--strip-->
> **id** `2026-08-01-dd2d-generalization-test-unseen-count-unseen` · **status** active
> · **tracks** env-dd2d, method, evaluation, data
<!--/strip-->

**Context.** The dd2d_v4-trained SPECTRE v3 checkpoint had only ever been evaluated
in-distribution (9–12 blockers, the base 7 shape families). The proposal's §6 object-count /
compositional-generalization question and §0 wishlist property #4 were *asserted, never
tested*. We wanted a direct OOD test on DD2D along two axes the model never saw: **more
blockers** and **novel shape figures**, scored train-old / test-new against the existing
checkpoint (no retraining).

**Decision.** Three sub-decisions, each load-bearing.

1. **New shapes ride the geometry-general grasp model — no per-shape code.** `dd2d/grasps.py`
   derives both the global-envelope grasp and the internal/concave grasp purely from
   `shape.polygon` (supporting-line contact runs + a scan-line antipodal search), with no
   branch on family anywhere. So a `tee` (bar+stem) and a `cross` (symmetric plus), both
   **concave**, were added to `dd2d/shapes.py` alone (`_build` + `_CONCAVE_FAMILIES`; kept OUT
   of `_FAMILY_WEIGHTS` so the base sampler never draws them, and sized to the finger/aperture
   constants like `horseshoe`). Verified: 0 floating grasps over 30 seeds each, and the real
   refiner certifies scenes containing them at collection — the grasp model carries over to
   the new shapes and their concave regions, exactly as hypothesised.

2. **Held-out collection = fresh band + unseen count with a *realized-count floor* + forced
   families.** Two test-only sets, 40 problems each, stratified s0–s3 (10 each):
   `dd2d_v4gen_count` (14–16 items = 13–15 blockers, old shapes; isolates count) and
   `dd2d_v4gen_shape` (same count + tee/cross in the pool with **≥1 of each forced** per
   scene). New collector flags (all default-preserving): `--seed-band-base` (base 3 = `[3M,4M)`
   for count, base 4 = `[4M,5M)` for shape — disjoint from train/val/test, `--band=1_000_000`
   kept so `compare.stratum_of` stays valid), `--n-items-min/max`, `--shape-set augmented`,
   `--require-families`, `--fill-max`. The **realized-count floor** was the non-obvious
   necessity: a fill-cap sweep showed 12–22% of scenes truncate below 14 items even at
   `fill_max=0.85` (a small sampled drawer can't fit 15), and such a scene falls *back into the
   seen range* — silently defeating the test. Cranking `fill_max` never closes the tail, so the
   generator now rejects and resamples any scene realizing fewer than `min_items`, which
   *guarantees* every kept problem is genuinely unseen-count. `fill_max=0.72` keeps the
   resample rate low.

3. **Score train-old / test-new via `--test-variant`, reusing the dd2d_v4 vocab.**
   `spectre_score_v3.py`'s new `--test-variant` overrides only the episode dir; vocab, model
   config and checkpoints stay from `--env-variant`. Valid with **no OOV and no retraining**
   because the DD2D vocab / `config_hash` are over the fixed operator/predicate/type sets only
   — a shape family is geometry metadata, not a vocab token, and more blockers only add generic
   objects handled by positional local-ids. The domain spec is shared across `dd2d_*` variants
   (registered in `domain.DOMAINS`) and stratum recovery is pid arithmetic. `--astar-baseline`
   computes astar-dist (default order, score = −plan_idx) off each episode's stored outcomes via
   the shared `rollout_fp`, so v3-vs-astar is one instrument, uncensored, paired bootstrap.

**Consequences.** The scoring ran clean (no OOV, no position-index error on the longer
skeletons from denser scenes) — confirming the count/shape invariance and that the position
encoding tolerates the longer plans. In-distribution v3 reproduced **5.78 ± 0.10** exactly,
validating the instrument. Result (v3 ALL FP, 3 seeds; paired vs astar-dist):

| set | v3 ALL | vs astar | s2 | s3 |
|---|---|---|---|---|
| in-dist `dd2d_v4` (n=100) | 5.78 ± 0.10 | −28.74 [−39.6,−18.8] | 10.49 | 9.19 |
| unseen count (n=40) | 9.40 ± 2.62 | −39.95 [−64.0,−18.1] | 30.23 | 4.87 |
| unseen count+shape (n=40) | 11.26 ± 3.44 | −21.89 [−42.6,−3.8] | 31.97 | 10.67 |

**v3 still wins overall on both held-out sets (CI excludes 0), so its advantage over the naive
planner order survives OOD** — but absolute FP degrades ~1.6–1.9× (5.78 → 9.40 → 11.26), and
the honest stratum reading is that **the win is carried by s3** (astar's default order is
pathological there, 108–167 FP), while **at s2 v3's advantage collapses under the count shift**
(30.23 vs astar 28.30; 31.97 vs 22.00 — within the ±9 seed spread), amplifying v3's already
characterized in-distribution s2 weakness. *(⚠️ s2 root cause refined 2026-08-02: this collapse is
dominantly a **pool-composition artifact** — the k=200 pool crowds out the redundant feasible
triples that padded s2 in-distribution — not model weakness; see
[2026-08-02](#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact).)*
The shape set is harder than count-only, as expected. Numbers and caveats live in [notebook 07](../notebook/07-stickbutton2d.md)
2026-08-01. The held-out raw dirs are archived and authoritative (DD2D generation is
PYTHONHASHSEED-dependent, so a re-run yields a fresh sample).

---

<a id="2026-08-01-vlmplan-stops-generating-first-feasible-plan"></a>
## 2026-08-01 — VLMPlan stops generating at the first feasible plan

<!--strip-->
> **id** `2026-08-01-vlmplan-stops-generating-first-feasible-plan` · **status** active
> · **tracks** baselines, evaluation
<!--/strip-->

**Context.** VLMPlan's generation loop ran until it stalled or hit its round cap, then
scoring walked the proposals to the first success. The 200-plan budget was read as a
target to approach; it is a **hard ceiling for the case where proposals keep failing**,
which is a different thing.

The two stages are deliberately split — only generation needs a model, so a re-score is
cheap ([2026-07-24](04-comparison.md#2026-07-24-vlmplan-baseline-protocol)) — and the
side effect was that generation had no labels and therefore no way to know it was done.
It kept proposing after the answer had already been found.

The cost became visible once the b5 grounding bug was fixed
([2026-08-01](07-stickbutton2d.md#2026-08-01-off-pool-proposals-grounded-against-domain-filtered)):
b5 problems went from stalling out at 0 plans to running all 10 rounds for 27 plans at
~884 s each, pushing the 100-problem run from ~9 h to ~14 h — to generate proposals the
scorer would never reach.

**Decision.** `generate_sequence` takes a `stop_check`, called after each round, and
stops at the first proposal known to refine. The runner supplies it; `max_plans` remains
the ceiling for the all-failing case.

**FP is unchanged, and that is the whole argument.** The metric is failures before the
first success, so the rollout never looks past that success — proposals after it are
wall-clock and nothing else. Pinned by
`test_vlmplan.py::test_stop_at_first_success_preserves_fp`.

Labels come from `label_step_sequence`, newly extracted as the **single** definition of
how a proposal is labelled (stored outcome if it matches a pooled candidate, live refine
otherwise) and now called by both the scorer and the stop check. Two copies would drift,
and the symptom would be a run stopping on a "success" the scorer then calls a failure.
They share the on-disk memo, so the refinement work is *moved earlier*, not duplicated,
and `vlmplan_score.py` still runs standalone.

**Consequences.**

- **`n_proposed` changes meaning, and §6 of the comparison notebook reports it.** With a
  stop check the count is censored at the first success — "plans needed", not "plans the
  model can produce". **The DD2D rows were generated without it, so that column is not
  comparable across the two environments.** FP is. `stop_at_first_success: false`
  reproduces the old behaviour, and `stopped_on_success` is recorded per problem so a
  short proposal list is never mistaken for a model that ran out of ideas.
- Wall-clock on the SB2D test run drops by roughly the margin above; the exact saving
  depends on how early the first success lands, which is itself the thing being measured.
- The stop check is *conservative by construction*: it can only fire on a plan the
  scorer would also label a success, because it is the same function.

---

<a id="2026-08-01-off-pool-proposals-grounded-against-domain-filtered"></a>
## 2026-08-01 — Off-pool proposals are grounded against the domain, not the filtered pool

<!--strip-->
> **id** `2026-08-01-off-pool-proposals-grounded-against-domain-filtered` · **status**
> active · **tracks** baselines, method, env-stickbutton2d
<!--/strip-->

**Context.** VLMPlan's protocol says a proposal is held to *exactly* the standard a
planner-emitted skeleton meets — no more, no less
([2026-07-24](04-comparison.md#2026-07-24-vlmplan-baseline-protocol)). The SB2D adapter
implemented "exactly" by recovering lifted operators from the episode's own
`skeleton_pool`, which needs no environment and keeps operator identity aligned with
`pool_index`.

That is stricter than intended, and the gap is not hypothetical. The acyclic pool filter
([2026-08-01](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1))
drops every skeleton containing a `PickStick`/`PlaceStick` cycle, so on b5 **no pooled
plan mentions `PlaceStick`** — while the domain has it and the prompt advertises it. Any
proposal using it died on a `KeyError` and was recorded as inapplicable.

Compounding it, the chaining rule the prompt stated was **false for mixed plans**. It
said "the first press is `...FromNothing`, every later press is `...FromButton`", which
holds only within one uninterrupted run of presses by the same effector. `PlaceStick` and
`PickStickFromButton` both re-add `(AboveNoButton)`; arm presses track `RobotAboveButton`
while stick presses track `StickAboveButton`, so the two never chain into each other.

Together these made the *entire* stick-then-arm strategy unrepresentable — the strategy
the model writes down unprompted ("we must place stick first to use bare arm"). Both b5
pilots returned **0 usable plans**; b5 problem 750000 round 0 was 21 blocks, 19 parsed,
**19 inapplicable**.

**Decision.** Ground against the **domain**, not the filtered pool: `_lifted_by_name`
takes the pool's operators first (env-free, identity-preserving for `pool_index`) and
fills any missing ones from kinder's own `create_bilevel_planning_models`. And correct
the prompt's chaining rule to state effector separation and the two reset actions.

The general rule, which is the part worth carrying to the next environment: **a
pool-generation heuristic is not a legality constraint.** The acyclic filter exists to
stop the pool filling with padding; an off-pool proposal is refined for real and must be
judged against what the domain permits.

**Consequences.**

- Pinned by `test_vlmplan_sb2d.py`: a `PlaceStick` plan grounds, the mixed
  stick→place→arm plan grounds, and — the guard that keeps the other two honest —
  `...FromButton` immediately after `PlaceStick` is still **rejected**, so the tests
  cannot pass on an adapter that simply stopped checking preconditions.
- **The full 100-problem test run was stopped ~8 problems in and restarted.** Its b5
  column would have been near-entirely published-order fallback, and b5 is one of the two
  strata that carry the SB2D result — the other 92 problems were not worth the ~9 hours
  to produce a column known in advance to be an artifact.
- **A wrong disclosure is worse than none.** Deviation 7/8's whole justification is that
  stating a precondition *removes a handicap* every other method gets from the domain for
  free. That argument only holds if the statement is true; the model obeys it either way.
  The corrected note is in `prompts/PROVENANCE.md` deviation 8, with the old text and why
  it was wrong.
- **An unset LLM endpoint is now a hard error.** During the re-pilot the
  `OPENAI_BASE_URL` export was missing and the OpenAI SDK silently fell back to
  `api.openai.com`; 5 requests went to the public API and were rejected 401, and nothing
  was processed only because no valid key was present. A machine with one would have
  completed the run off-box and billed for it. `make_model` now refuses an unconfigured
  endpoint and names the fix, `SPECTRE_VLMPLAN_ALLOW_REMOTE=1` is the deliberate opt-in,
  and `vlmplan_sb2d_32b.yaml` states `base_url` rather than relying on an export.

---

<a id="2026-08-01-comparison-notebook-parameterised-env-registry"></a>
## 2026-08-01 — Comparison notebook parameterised by an env registry

<!--strip-->
> **id** `2026-08-01-comparison-notebook-parameterised-env-registry` · **status**
> active · **tracks** tooling, evaluation
<!--/strip-->

**Context.** `compare_dd2d_methods.py` is where every method comparison is read. Standing
StickButton2D up alongside DD2D was done the obvious way first — copy the file to
`compare_sb2d_methods.py` and edit the constants — which produces two 1400-line notebooks
that share six sections of analysis and drift apart on the first fix applied to one of
them. The project already has a rule against exactly this shape of duplication for
environments (`domain.DomainSpec`, `piginet.PIGINetDomain`, `vlmplan.EnvAdapter`); the
notebook was the last place still forking.

The DD2D assumptions were not concentrated anywhere. They were a hardcoded `env_variant`,
a `primary_name="dd2d_v4"` string tagging every loaded row, strata whose labels mean
min-feasible-subset size, a method list carrying two SPECTRE-v1 rows, `dd2d_*.csv` export
names, a scene renderer imported from `envs/dd2d`, an `n=100` in a chart title, and an
`f"s{k}"` stratum label formatter — a dozen small places, each individually too minor to
notice.

**Decision.** Three files replace the fork:

- `spectre/compare.py` — `dd2d_compare.py` renamed (139 references across 14 files).
- `spectre/compare_envs.py` — the registry: one `EnvSpec` per environment carrying
  variant, legacy graft, stratum labels, axis label, which sections apply, an optional
  scene renderer, and its caveats. **A third environment is one entry.**
- `experiments/spectre/compare_methods.py` — the single notebook, environment chosen by
  an `mo.ui.dropdown`.

`stratum_labels` is the important field. SB2D's button count is recovered by DD2D's
seed-band arithmetic *only because the problem ids were chosen to make that true* — a
coincidence that was implicit in a formula named for DD2D seeds and is now a declaration.

**Caveats live in the registry, beside the number.** `EnvSpec.caveats` renders under §1's
summary table. A reader quoting a figure sees what bounds it in the same view, rather than
in a document they would have to know to open.

**Consequences.**

- **Verified by rendering both environments, not by inspection.** The notebook takes
  `SPECTRE_COMPARE_ENV` for its initial selection specifically so marimo's script mode can
  execute it headlessly for *every* registry entry — otherwise only the entry that sorts
  first is ever smoke-tested. DD2D re-renders unchanged after the rename (7.44 / 17.27 /
  17.27 / 20.66 / 20.86 / 23.55 / 29.86 / 34.52), which is the check that mattered: 139
  mechanical edits is exactly where a silent mis-edit hides.
- Three bugs the fork would have kept: the `collection` column labelled SB2D rows
  `dd2d_v4`; the CSV export wrote `dd2d_method_*.csv` for both environments, so rendering
  the second silently overwrote the first; and §4.3 crashed on an empty frame instead of
  reporting that demotion arms are inapplicable.
- **§4.3 is inapplicable rather than missing on SB2D**, and now says so. Proof-tier
  demotion needs provable query axioms; SB2D resolves to `EMPTY_SPEC`, so the demotion-on
  and demotion-off caches would be bit-identical. Rendering that as an ablation with a
  0.00 Δ would be the worst outcome — a measurement of nothing that looks like a
  measurement.
- Deleted: `compare_dd2d_methods.py`, `compare_sb2d_methods.py`.

---

<a id="2026-08-01-vlmplan-made-env-agnostic-via-labeler-protocol"></a>
## 2026-08-01 — VLMPlan made env-agnostic via a Labeler protocol

<!--strip-->
> **id** `2026-08-01-vlmplan-made-env-agnostic-via-labeler-protocol` · **status**
> active · **tracks** baselines, tooling, env-stickbutton2d
<!--/strip-->

**Context.** VLMPlan is the **zero-training-data corner** of the data × perception grid
([`proposal.md`](../proposal.md) §0), so a second environment without it is a grid missing
a column, not merely a missing row.

`vlmplan/score.py` was already env-agnostic in the parts that matter — budget accounting,
the published-order fill, `label_agreement` — but it reached past its own abstraction in
one place: it imported `DD2DRefiner`, `staging_skeleton` and `reconstruct_scene` directly,
and `REFINER_PRESETS` was keyed by DD2D variant. That import is what makes an *off-pool*
proposal refinable at all, and it is precisely the thing that differs per environment.

The setting of that refiner is not a detail. VLMPlan's score mixes labels from two
sources: stored labels for proposals that match a pool candidate, and live refinement for
the ones that do not. If the live refiner runs at different settings than the collection
did, the two halves of the same row are drawn from different distributions — off-pool
proposals get systematically easier or harder labels than in-pool ones, and the arm's
number moves for a reason that has nothing to do with the model.

**Decision.** Introduce a **`Labeler`** ABC (`vlmplan/adapter.py`) — *given an episode and
a proposed step sequence, return feasible/infeasible* — with `n_refines` and `flush()`.
`score_sequence` and `label_agreement` take one as a parameter. DD2D's implementation
wraps `DD2DRefiner`; SB2D's (`vlmplan/sb2d_label.py`) wraps kinder's `BacktrackingRefiner`
**at the collection's own settings** — `num_sampling_attempts_per_step=5`,
`refinement_timeout_s=20`, `max_trajectory_steps=200` — using the collection's
per-candidate seed rule. `vlmplan/registry.py` dispatches both adapter and labeler on
`env_variant`.

Memoization moved up into a shared `MemoizingLabeler` base keyed on the canonical step
tuple, so both environments get it and neither implements it.

**Consequences.**

- The **label-agreement gate is now the acceptance test for a new environment's labeler,
  not a diagnostic printed after the fact.** SB2D reads **1.000** (35 samples), against
  DD2D's 0.982. It earned that status by catching three real bugs during bring-up, all of
  which presented identically — as stored-success → live-fail, i.e. exactly like env
  drift — at an agreement of 0.571:
  1. the off-pool derived seed was used for plans that *were* in the pool (fixed by
     matching against `pool_index` first);
  2. canonical episode names (`circle_0`) were handed to an env that knows `button0`;
  3. operators were grounded over env objects but the trajectory was progressed from the
     *canonical* initial state.

  None would have been visible in the resulting number. All three were visible in the
  gate.
- Off-pool seeds derive via `hashlib.blake2b`, not `hash()`. Python's `hash()` is
  `PYTHONHASHSEED`-salted, so a re-score in a different process would have drawn different
  labels for the same proposal — the same class of irreproducibility already recorded for
  the DD2D generator ([2026-07-26](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent)).
- **Deviation 8** added to `vlmplan/prompts/PROVENANCE.md`: `_CONTROLLER_NOTE` states the
  chaining rule (`…FromNothing` vs `…FromButton` depends on where the robot already is).
  Without it the 32B model used `RobotPressButtonFromNothing` for every press and produced
  **11/11 precondition violations**; with it, 5/5 valid. This is the same failure the DD2D
  run hit 28/28 times on a different near-synonymous skill pair, so the mitigation is now
  a documented part of the template rather than a per-environment rediscovery.

---

<a id="2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters"></a>
## 2026-08-01 — PIGINet lifted to an env-agnostic package with per-env adapters

<!--strip-->
> **id** `2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters` ·
> **status** active · **tracks** baselines, tooling, env-stickbutton2d
<!--/strip-->

**Context.** The DD2D comparison notebook's headline is SPECTRE v3 against **PIGINet** —
the low-level predictor over concrete state. That row is the whole representation
question: "what should a feasibility predictor represent skeletons and problems over?"
StickButton2D had SPECTRE v3 and the B1–B5 bracket but no PIGINet, so the second
environment could not answer the question the project exists to ask.

PIGINet lived at `envs/dd2d/piginet/` and was DD2D-specific in five places: a gloss table
imported at module scope, `_SHAPE_MAX` in centimetres, a `drawer_wh` key read out of
`provenance`, a `dd2d_*` directory glob, and its paths in the cache driver. Individually
reasonable; together they make a second environment a rewrite.

**Decision.** Lift the package to `spectre/piginet/` behind a `PIGINetDomain` protocol,
with one adapter per environment — the shape `vlmplan/` already established here, and the
same move `domain.DomainSpec` made for SPECTRE v3 itself.

- **The normalisers become domain state, not module constants.** This is the reason the
  abstraction is a class rather than two more imports. PIGINet divides poses by a frame
  extent and shapes by per-field maxima so both land in `[-1, 1]`. DD2D's are centimetres
  over a ~50×40 drawer; StickButton2D is metres over 3.5×2.5 with objects two orders of
  magnitude smaller. Measured: SB2D shape features read `|mean| 0.372` against their own
  divisors and **`|mean| 0.0061`, max 0.05** against DD2D's — a channel 60× flatter, i.e.
  effectively dead. The conclusion "the low-level predictor loses on StickButton2D" was
  available as a *unit bug* wearing a result's clothes, and nothing would have raised.
- `PIGINetExample` / `ImageRef` move to `piginet/record.py`; DD2D's `record.py` keeps its
  builders and re-exports them, so every existing import resolves.
- `SB2DDomain` builds examples from the **same `EpisodeRecord` pickles SPECTRE trains on**
  — so the two methods' labels are identical by construction, not by agreement — and
  rasterises crops from stored `scene_geometry` (*reconstruct, never regenerate*).
- The cache driver's `--env-variant` choices came from `_V2_CKPT_SUBDIR`, i.e. "collections
  with a SPECTRE v2.2 checkpoint". StickButton2D deliberately has none, so it was rejected
  at the CLI despite having PIGINet and v3 rows. Now the union of the method maps, with a
  missing method failing on its own rather than blocking the driver.

**Consequences.**

- **DD2D is unmoved, verified on the metric rather than on bytes.** Re-running the dd2d_v4
  PIGINet cache gives rollout FP **17.0500 before and after**, per-problem identical on all
  100 problems, with labels and rank order identical. Scores drift by ≤2.3e-4 — CUDA float
  nondeterminism in CLIP inference. The plan's stated bar was "byte-identical", and that
  bar was **wrong for a GPU inference path**: it cannot be met by any re-run, refactor or
  not. The right criterion for this class of change is identical labels, identical rank
  order and an identical derived metric.
- **`at-pose` literals are synthesised for StickButton2D.** Its abstract initial state is
  two atoms and names no positions, so a faithful port had to add one pose literal per
  object, exactly as DD2D's records carry natively. Without it PIGINet receives object
  identities with no coordinates — it would stop being a *low-level* predictor, which is
  the only reason it is in the comparison. This is our construction, not stored data.
- **The image channel is degenerate on StickButton2D and stays in anyway.** Every unpressed
  button is the same red disc, so CLIP separates only {button, stick, robot} — which the
  type literals already give. Crops share one fixed world window so relative scale at least
  survives (the stick renders as a bar, a button as a dot). Reported as a bound on what
  this environment's PIGINet row can be claimed to show, not silently absorbed.
- The lifted package keeps its mypy exclusion. It was covered by the vendored-DD2D
  exclusion for its whole life; moving a file is not the moment to impose strict typing on
  it. `domain.py`, `record.py` and the adapters are ours and stay checked.

---

<a id="2026-08-01-both-evidence-classes-stay-wired-stickbutton2d"></a>
## 2026-08-01 — Both evidence classes stay wired; StickButton2D has only class 2

<!--strip-->
> **id** `2026-08-01-both-evidence-classes-stay-wired-stickbutton2d` · **status**
> active · **tracks** method, data, env-stickbutton2d
<!--/strip-->

**Context.** The unified coverage/waste definitions (2026-07-31) are computed over
*records*, and `records_from_failure_records` built them from one field: `culprits`, the
objects the refiner's own validity check named. That is §2's **class 1**, and it is all
DD2D produces.

StickButton2D produces **none of it**. kinder's motion model rejects a colliding
transition by silently declining to move, and its collision predicate returns a bool
without naming anything, so there is no object-naming check to instrument. Every SB2D
failure is §2's **class 2**: the sample executes and the trace check finds observed ≠
predicted. Nothing serialized that. The failure mode was not an error — it was
`coverage ≡ 0`, `waste ≡ 0`, and v3 silently degrading to a static ranker while reporting
a clean run. The same shape as the `S(c) = args \ goal_objects` problem the unified
definitions were introduced to fix, one level down.

A second, smaller thing surfaced with it: `records_from_failure_records` *dropped* any
record with no culprits. On SB2D that would have been every record.

**Decision.** One path, both classes, always wired; emptiness is data, not a branch.

- **Class 2 is serialized** into `refiner_metadata["failures"]` as `dev_added` /
  `dev_deleted` — `(predicate, [arg, ...])` **name pairs**, not `GroundAtom`s, because
  they have to survive `canonicalize_episode`'s renaming. `unified_evidence` rebuilds real
  ground atoms from a per-episode predicate table at read time, since every consumer
  compares them by identity against operator effects.
- **The class-1 slot is emitted anyway, empty**, and vice versa on DD2D. No consumer
  branches on the environment.
- **Blameless records are kept** rather than filtered. A failure that names nobody is
  still an observation that this step failed, and the record-token stream reads it.
- **`waste` abstains on an empty culprit pool** (returns 0.0). This is the one place
  keeping blameless records was *not* already inert: with `K = ∅` nothing justifies any
  idle step, so the ratio would return a maximally confident 1.0 derived from zero
  evidence — and only on contexts that named nobody, i.e. as noise correlated with having
  no information.
- **Deviation-derived blame is stored separately**, as `dev_blame`, and feeds the record
  token's culprit tag slot only where `culprits` is empty. A culprit was named by the
  environment; this was inferred by us from the trace. Collapsing them would let a model
  trained where the signal is observed be deployed where it is inferred with nothing
  recording the difference.

**Consequences.**

- Inertness of the empty channel is a **proof, not a measurement**: a blameless record
  contributes nothing to `K`, `covered` skips it for every object, `_justified` never
  consults it, and `waste` now abstains. Pinned by
  `test_blameless_records_do_not_change_coverage_or_waste`. DD2D re-scores at
  **5.78 ± 0.10** — identical to the pre-change figure, per stratum as well as overall —
  which is what discharges the standing "re-score the frozen baseline under new code
  before training anything" rule.
- Two traps this exposed, both of which produce no symptom:
  - **Nested names must be remapped.** `_remap_refiner_metadata` renamed `args` /
    `culprits` / `unmoved`; the object names *inside* `dev_added` / `dev_deleted` are one
    level deeper. Missing them makes every record's tags fail to resolve and the whole
    stream degenerate to "some failure of some schema".
  - **Positional pairing must filter both sides.** `records_for_candidate` silently drops
    entries missing `schema`/`step_index`; pairing its output against the *unfiltered*
    metadata list shifts every later deviation onto the wrong record, with both sides
    still well-formed.
- SB2D collection runs through `RecordingSampler`, which **re-implements** upstream's
  sampler loop rather than subclassing a hook — upstream computes the achieved abstract
  state to decide accept-or-reject and then discards it behind a payload-free
  `TrajectorySamplingFailure`. That is the one place this port does not simply wrap
  kinder. It is a same-seed differential measurement, not a claim:
  `test_stickbutton2d_observational.py` refines the same pools through both samplers and
  requires identical labels (b2 and b3, 3 problems × 8 candidates each). A prior docstring
  asserted such a test existed; it did not, and writing it is what makes this safe.

---

<a id="2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1"></a>
## 2026-08-01 — Acyclic pool filter and the pooled stickbutton2d_v1 variant

<!--strip-->
> **id** `2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1` · **status** active
> · **tracks** method, data, env-stickbutton2d
<!--/strip-->

**Context.** Standing up StickButton2D as SPECTRE's second environment needed a pool, and
the pool the substrate produces is not usable as-is.
`HeuristicSearchAbstractPlanGenerator` deliberately allows revisiting abstract states —
"that's important because we need to generate multiple abstract plans"
(`heuristic_search_plan_generator.py`) — which on this domain licenses padding any plan
with `PickStickFromNothing` / `PlaceStick` pairs. Those return to `s_0` *exactly*, so A*
enumerates them in `f` order and they fill the pool.

Measured acyclic fraction of a 200-candidate draw, over 6 seeds per variant:

| | b1 | b2 | b3 | b5 |
|---|---|---|---|---|
| acyclic / 200 raw draws | **1–2** | 6–34 | 73–101 | 193–200 |
| acyclic, raw budget 5000 | 1–2 | 6–34 | **200** (≈640 raw) | 200 (200 raw) |

At b1 all 200 candidates are the same plan with 0–199 pickup/putdown cycles prepended,
running to 400 operators. A ranker asked to order that is being asked a question about
padding, not about feasibility.

Separately, the four button counts had to become one dataset. They differ by two orders of
magnitude in pool size, which is a difficulty axis rather than four separate problems.

**Decision.** Two things, both env-agnostic.

1. **Filter cyclic skeletons out of the pool** (`AcyclicPlanGenerator`): reject a skeleton
   if `s_i == s_j` for any `i < j`, identity being the atom set. Applied uniformly to
   every variant, with a `raw_cap` of 5000 draws as the stop rule for variants whose
   acyclic set is genuinely finite. It reads only the abstract state sequence, so it would
   apply unchanged to any environment whose generator revisits states.
2. **Pool b1/b2/b3/b5 into one `env_variant`, `stickbutton2d_v1`**, with button count as
   the stratum, encoded arithmetically into the problem id
   (`envs/stickbutton2d/strata.py`): `pid = split_band·10⁶ + slot·250000 + index`, chosen
   so the existing `dd2d_compare.stratum_of` returns the slot exactly. b10 is dropped —
   0/20 problems solvable within the budget, and the cause is pool prefix homogeneity that
   needs diverse plan *generation*, not a better heuristic
   (`autonomous_stickbutton_session.md` D5).

**Consequences.**

- The filter is near-inert exactly where the ranking problem is real (b5: removes 0–7 of
  200) and removes the degeneracy where it is not. b3 gains: 200 *real* candidates instead
  of ~90 real + ~110 padded, which also makes b3 roughly twice as expensive to collect as
  the pre-filter measurement implied.
- **This is a benchmark-definition choice, not a free simplification, and the caveat is
  real**: a padded plan can be *genuinely* more refinable than its acyclic core, because
  `PlaceStick` puts the stick down somewhere new and re-picking it changes the geometry.
  What is claimed is that a pool of near-duplicates is the wrong ranking problem — not
  that the dropped plans are infeasible. A domain where tool re-placement is the point
  would want this off.
- Strata 0 and 1 are anchors, not contests. With pools of ≈2 and 6–34, b1 reads 0.07 mean
  failed attempts under the *static* order and every method ties it — the same shape as
  DD2D's `s0 = 0.00`. About half of b1's episodes have pool size 1 and are dropped by
  `train_v3._trainable` (`len(skeleton_pool) >= 2`). b3 and b5 carry the result, and a
  pooled "ALL" mean over unbalanced strata should not be read as a method comparison.
- The pid encoding is arithmetic and therefore silently breakable, so it is pinned by a
  unit test against `stratum_of` and each episode independently records
  `provenance.gen_params["stratum"]` as an audit trail. Strata occupy contiguous pid
  bands, which makes **stride, never truncate** load-bearing here: `paths[:N]` returns b1
  only.

---

