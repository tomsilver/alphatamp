# SPECTRE Notebook — StickButton2D as a second environment

3 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---
<a id="2026-08-10-held-out-vs-matched-full-controls-anomaly-confound"></a>
## 2026-08-10 — Held-out vs matched-full controls: the anomaly is confound+noise; b5 expanded to 100

<!--strip-->
> **id** `2026-08-10-held-out-vs-matched-full-controls-anomaly-confound` · **status**
> active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**What.** The [held-out-stratum experiment](#2026-08-09-held-out-stratum-comparison-spectre-generalizes-dd2d-s3)
looked incoherent (subset-trained beat full on DD2D ALL; SB2D ranking flipped, PIGINet *improving*
when b5 held out). Re-ran it properly: **matched full-strata controls** (current-code, same recipe,
same test problems — not the frozen 5.78/1.69 yardstick), **per-stratum paired bootstrap over
problems** (`holdout_vs_full.py`), and — the root-cause fix for SB2D — **b5 train collected to the
correct 100** in a new `stickbutton2d_v2` variant (v1 frozen; only the full model retrained, subset
reused). DD2D full = deployed `dd2d_v4` cache (fresh arm trained pathologically slowly, abandoned).
Protocol/decisions: [`decisions/07` 2026-08-10](../decisions/07-stickbutton2d.md#2026-08-10-held-out-stratum-anomalies-resolved-matched-controls-per-stratum).

**Result — Δ = subset − full, paired bootstrap (positive ⇒ subset worse / full better; `*` = CI excludes 0).**

| env · method | ALL | trained strata | **held-out (s3 / b5)** |
|---|---|---|---|
| DD2D SPECTRE-adaptive | −0.57 [−1.51, +0.29] | **s1 −2.96 [−4.93, −1.31] \*** | +1.19 [−0.77, +2.89] |
| DD2D PIGINet | +10.61 \* | s1 +1.04 (ns) | **+40.69 [+26.80, +54.77] \*** |
| SB2D SPECTRE-adaptive (b5=100) | −0.06 [−0.60, +0.47] | **b3 −0.92 [−1.72, −0.28] \*** | +0.73 [−1.23, +2.63] |
| SB2D PIGINet (b5=100) | −0.21 [−0.75, +0.25] | b3 −0.31 (ns) | −0.43 [−2.44, +1.27] |

Absolute FP (subset / full), held-out column bold:

| | ALL | s0/b1 | s1/b2 | s2/b3 | **s3/b5** |
|---|---|---|---|---|---|
| DD2D SPECTRE sub / full | 5.35 / 5.92 | 0.00 / 0.00 | 1.88 / 4.84 | 9.55 / 10.05 | **9.97 / 8.79** |
| DD2D PIGINet sub / full | 27.88 / 17.27 | 0.04 / 0.05 | 6.08 / 5.04 | 19.51 / 18.77 | **85.89 / 45.20** |
| SB2D SPECTRE sub / full | 2.10 / 2.17 | 0.08 / 0.08 | 0.27 / 0.33 | 1.20 / 2.12 | **6.87 / 6.13** |
| SB2D PIGINet sub / full | 1.68 / 1.89 | 0.07 / 0.08 | 0.32 / 0.41 | 0.99 / 1.29 | **5.36 / 5.79** |

**The anomaly was confound + noise; nothing "helps by holding out data."**
- **DD2D ALL "subset beats full" is not significant** (SPECTRE Δ −0.57, ns) — it was the frozen-vs-current
  baseline confound plus reading ALL. The held-out **s3 is coherent** (full better) — decisively for
  PIGINet (45.20 ≪ 85.89), directionally for SPECTRE. The one significant sub-effect is **s1
  specialization**: holding out hard s3 makes SPECTRE *better* on trained s1 (1.88 vs 4.84), which
  is what pulled ALL down.
- **SB2D flip is gone once b5 is properly sized.** With b5=100 the ALL deltas collapse to noise
  (SPECTRE −0.06, PIGINet −0.21), held-out **b5 SPECTRE is directionally coherent** (6.13 < 6.87 —
  a direction the 17-episode full model could not show), **PIGINet shows no effect** (5.79 vs 5.36,
  ns). Significant effect: **b3 specialization** (1.20 vs 2.12).

**Takeaway-next.** "Superset helps" holds *on the held-out stratum in direction* (3/4 cases),
reaches significance only where the effect is large (DD2D PIGINet s3), and is otherwise inside the
~1 FP noise / 25-test-problem resolution. Read the **held-out stratum with a paired CI**, never the
pooled ALL. The robust cross-environment finding is **trained-strata specialization** (DD2D s1,
SB2D b3): the hard stratum's training examples measurably *cost* an easy stratum. b5-correct-size
(`stickbutton2d_v2`, b5 17→100) is the real fix — it removed the 17-episode artifact and made SB2D
a powered-on-training test; v1 is byte-unchanged. Reproduce: `python experiments/spectre/holdout_vs_full.py`.

---

<a id="2026-08-09-lazy-baseline-results-dd2d-sb2d"></a>
## 2026-08-09 — LAZY baseline results — DD2D + SB2D

<!--strip-->
> **id** `2026-08-09-lazy-baseline-results-dd2d-sb2d` · **status** active · **tracks**
> baselines, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**What.** First results for the newly re-implemented LAZY adaptive baseline (`baselines/lazy/`,
GAT policy π + online feasibility ϕ, π̄=π·ϕ/Σ; faithful-max prefix-tree realization). 3 seeds
each, uncensored test n=100. Design + full protocol:
[decisions/07 2026-08-09](../decisions/07-stickbutton2d.md#2026-08-09-lazy-policy-guided-adaptive-baseline-added-dd2d).

**Result.** Mean failed attempts before first success (± across-seed sd):

| method | DD2D (dd2d_v4) | SB2D (kinder) |
|---|---|---|
| LAZY-adaptive | 23.26 ± 0.50 | 1.85 ± 0.02 |
| SPECTRE-adaptive | 5.92 ± 0.29 | 1.84 ± 0.25 |
| PIGINet | 17.27 ± 0.19 | 2.28 ± 0.29 |
| SPECTRE-static | 21.65 ± 1.13 | 1.98 ± 0.28 |
| astar-dist | 34.52 | 16.29 |
| VLMPlan-GPT5.6 | 35.23 | 6.42 |

- **DD2D:** LAZY beats the naive order (34.52) and VLMPlan (35.23) and ≈ SPECTRE-static, but
  both learned rankers beat it — paired vs LAZY: SPECTRE −17.34 CI [−24.0,−11.4], PIGINet
  −5.99 CI [−9.96,−2.28] (both exclude 0). LAZY's ALL is carried by s3 (58.65 vs astar ~119),
  worse than astar at s1/s2 — the same s3-carried shape SPECTRE shows.
- **SB2D:** LAZY ties everything — SPECTRE−LAZY −0.01 CI [−0.72,+0.72]; LAZY−PIGINet −0.44 CI
  [−1.18,+0.29]. Per stratum LAZY b1 0.08 / b2 0.36 / b3 2.44 / b5 4.56.
- **Policy is load-bearing (diagnostic astar / ϕ-only / LAZY):** DD2D val LAZY 28.70 < astar
  35.66 while ϕ-only is *worse* (49.03) → the GAT policy carries DD2D; SB2D test ϕ-only 2.40
  (feasibility very discriminative there — PickStick/PlaceStick padding), policy sharpens to
  1.86.

**Takeaway-next.** The adaptive-method bar is now on the record: **on DD2D the learned rankers
beat the LAZY adaptive method decisively; on SB2D no method separates** (LAZY joins SPECTRE and
PIGINet in the tie), consistent with the standing SB2D non-separation finding. Caveats: SB2D b5
rests on the small 17-episode b5 train split (a b5 expansion was in progress — re-measure);
LAZY seed sd is tiny because seeds share the deterministic canonicalization + fitted ϕ (only
init varies). BC cross-entropy plateaus at a label-conflict floor (feasible plans diverge at the
root), so selection is on val rollout-FP, not CE.

---

<a id="2026-08-09-held-out-stratum-comparison-spectre-generalizes-dd2d-s3"></a>
## 2026-08-09 — Held-out-stratum comparison — SPECTRE generalizes on DD2D s3, PIGINet ties on SB2D b5

<!--strip-->
> **id** `2026-08-09-held-out-stratum-comparison-spectre-generalizes-dd2d-s3` ·
> **status** active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**What.** Held out a whole stratum from *training* for the first time: SPECTRE + PIGINet
trained on s0–s2 (DD2D) / b1/b2/b3 (SB2D) via `--train-strata 0 1 2`, evaluated on all four
strata of the standard test split, held-out s3 / b5 the headline. astar + the frontier
VLMPlan-GPT5.6 are training-free and reused verbatim. New notebook entries `dd2d_holdout_s3`
and `sb2d_holdout_b5`; FP + §2b wall-clock. Protocol ADR:
[`decisions/07`](../decisions/07-stickbutton2d.md#2026-08-09-held-out-stratum-generalization-train-s0-s2-b1-b3-evaluate).
3 seeds, uncensored, test n=100 (VLMPlan stratified n=40).

**Result — DD2D (train s0/s1/s2, headline = held-out s3).**

| method | ALL | s0 | s1 | s2 | **s3** |
|---|---|---|---|---|---|
| SPECTRE-adaptive | 5.35 ± 0.49 | 0.00 | 1.88 | 9.55 | **9.97 ± 1.59** |
| SPECTRE-static | 20.01 ± 1.87 | 0.00 | 4.97 | 30.81 | **44.27 ± 4.45** |
| PIGINet | 27.88 ± 2.51 | 0.04 | 6.08 | 19.51 | **85.89 ± 9.25** |
| astar-dist | 34.52 | 0.00 | 2.24 | 17.08 | **118.76** |
| VLMPlan-GPT5.6 | 35.23 | 26.90 | 26.70 | 28.00 | **59.30** |

**SPECTRE-adaptive generalizes to the unseen stratum; PIGINet collapses.** On held-out s3 it
beats PIGINet ~9× (9.97 vs 85.89) and beats astar/VLMPlan by more. Its ALL (5.35) is ≈ its
in-distribution 5.78, and its held-out s3 (9.97) is within noise of the in-*distribution* s3
(9.19) — the abstract ranker barely notices s3 was withheld, while PIGINet's ALL blows out
17.27 → 27.88. **The representation alone is not enough**: SPECTRE-static s3 = 44.27, adaptive
= 9.97, the same static-falls-behind / adaptivity-recovers shape as the 2026-08-04 shape
generalization.

**Result — SB2D (train b1/b2/b3, headline = held-out b5).**

| method | ALL | b1 | b2 | b3 | **b5** |
|---|---|---|---|---|---|
| PIGINet | 1.68 ± 0.20 | 0.07 | 0.32 | 0.99 | **5.36 ± 0.66** |
| SPECTRE-adaptive | 2.10 ± 0.42 | 0.08 | 0.27 | 1.20 | **6.87 ± 1.38** |
| SPECTRE-static | 2.31 ± 0.31 | 0.08 | 0.32 | 1.45 | **7.37 ± 1.25** |
| VLMPlan-GPT5.6 | 6.42 | 0.00 | 2.40 | 0.90 | **22.40** |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | **61.56** |

**The representation advantage does not reproduce on SB2D held-out b5** — PIGINet (5.36) ≈
SPECTRE-adaptive (6.87), PIGINet marginally ahead but inside the seed spread; ALL PIGINet 1.68 ≈
adaptive 2.10. Same non-separation as the in-distribution SB2D finding. The adaptive increment is
still positive (b5 adaptive 6.87 < static 7.37).

**Sanity anchors passed:** astar (DD2D 34.52 ALL, SB2D 61.56 b5) and VLMPlan-GPT5.6 (DD2D 35.23
ALL, SB2D 22.40 b5) reproduce the deployed numbers exactly — the training-free reuse-by-symlink
is byte-correct. §2b wall-clock renders for both (caps 2 s / 10 s, all five methods timed).

**Takeaway-next.** Held-out-stratum generalization tells the same cross-environment story as
in-distribution and shape shift: **abstract representation wins decisively on DD2D, ties/loses
marginally on SB2D; the failure-conditioned re-ranking is what carries DD2D's OOD win** (static
alone loses to PIGINet at s3 too). Read the headline stratum (s3 / b5), not the pooled ALL, which
averages held-out with in-distribution strata. The comparison is honest: the same held-out test
problems, the only change is which strata the learned rankers saw in training.

---

<a id="2026-08-09-narrowed-input-variance-recovered-select-window-5-fixes-ema"></a>
## 2026-08-09 — Narrowed-input variance recovered: select-window-5 fixes it, EMA inert

<!--strip-->
> **id** `2026-08-09-narrowed-input-variance-recovered-select-window-5-fixes-ema` ·
> **status** active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**What.** The domain-agnostic narrowing regressed the 3-seed mean (DD2D 6.63 vs 5.78, all at s1;
SB2D 2.10 vs 1.69) as *variance*, not information loss (best seed ≈ baseline; probe Δ0.00; std ~7×
up). Built two config-gated, off-by-default training-process levers — `--select-window` (ma3→ma5)
and `--weight-avg ema` (decay 0.999, post-warmup, keep-the-better selection) — and tested them.
ADR: [decisions/07 2026-08-09](../decisions/07-stickbutton2d.md#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider).

**Result — SB2D triage (cheap env first, 3 seeds).** Both levers reduce variance and reach
not-significantly-worse than the frozen baseline: sw5 1.84 ± 0.25, ema 1.84 ± 0.20 (narrowed
2.10 ± 0.43; frozen 1.69). Paired vs frozen: sw5 Δ+0.15 CI [−0.06, +0.38], ema Δ+0.15 CI
[−0.12, +0.45] — both include 0. But the keep-better logs show **EMA's val_fp is *worse* than
raw** (decay 0.999 too slow for SB2D's short training; raw picked on 2/3 seeds), and sw5 matched
it for free — pointing at selector noise.

**Result — DD2D (headline, 3 seeds, uncensored).**

| arm | ALL | s1 | s2 | s3 | best | vs frozen 5.78 | vs narrowed |
|---|---|---|---|---|---|---|---|
| narrowed | 6.63 ± 0.68 | 7.41 ± 2.94 | 10.33 | 8.76 | 6.03 | +0.85 | — |
| **sw5 (ma5)** | **5.92 ± 0.29** | **4.84 ± 1.03** | 10.05 | 8.79 | **5.69** | **+0.14, CI [−0.37, +0.68]** | **−0.71, CI [−1.52, −0.05]** |
| ema | 6.51 ± 0.60 | 6.65 ± 2.98 | 10.16 | 9.21 | 6.03 | +0.73, CI [−0.03, +1.58] | −0.12, CI [−0.70, +0.40] |

**sw5 ties the frozen baseline and significantly beats the narrowed model**; s1 recovers 7.41 →
4.84 and its std collapses 2.94 → 1.03. **EMA is inert** — 6.51 ≈ narrowed 6.63, keep-better chose
raw on 2/3 seeds (EMA val worse), s1 still 6.65 ± 2.98. Promoted **sw5** to the deployed dirs.

**Takeaway-next.** The regression was **selector noise**: the higher-variance narrowed model needs
a wider selection window to pick reliably; ma5 is the appropriate selection for it, and it costs
nothing at deploy (a better-selected checkpoint, no EMA machinery). EMA didn't engage because the
model's endpoint isn't oscillating in a way a slow average captures — kept in-code (tested) for a
domain that does. **Caveat carried forward:** cross-arm means are confounded by run-to-run GPU
nondeterminism, so the decision rests on the *within-run* EMA-vs-raw val signal and the variance
collapse, not any single cross-arm mean delta. Deployed: DD2D 5.92, SB2D 1.84 (both tie the frozen
baseline).

---

<a id="2026-08-08-terra-gripper-disclosure-halves-vlmplan-fp"></a>
## 2026-08-08 — terra + gripper disclosure halves VLMPlan FP on DD2D and SB2D

<!--strip-->
> **id** `2026-08-08-terra-gripper-disclosure-halves-vlmplan-fp` · **status** active ·
> **tracks** baselines, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**What.** Hardened the headline VLMPlan row against two reviewer criticisms (ADR
[2026-08-08](../decisions/07-stickbutton2d.md#2026-08-08-vlmplan-headline-swapped-gpt-5-6-terra-gripper-geometry)):
swapped the model **gpt-5.6-luna → gpt-5.6-terra** (the stronger tier) and **disclosed the
gripper's real dimensions in the text prompt** (DD2D finger 2.5×2.0 cm / aperture 0.5–12 cm /
18 approach angles, imported from `grasps.py`; SB2D arm/gripper widths — `PROVENANCE.md`
deviation 9), since the audit found the VLM was never told the geometry DD2D feasibility hinges
on. Ran the two-stage protocol native on `dd2d_v4` / `stickbutton2d_v1_kinder`, stratified 40
(10/stratum), `effort: low`.

**Result — FP roughly halves on both envs (single generation seed).**

| env | luna (old) | **terra + disclosure** | per-stratum (terra) | context |
|---|---|---|---|---|
| DD2D | 62.98 | **35.23** (agree 0.983, 1 censored, 35/40 self-solve) | s0 26.9 · s1 26.7 · s2 28.0 · s3 59.3 | astar 34.52 · PIGINet 17.27 · SPECTRE 5.78 |
| SB2D | 11.85 | **6.42** (agree 1.000, 0 censored, 39/40 self-solve) | b1 0.0 · b2 2.4 · b3 0.9 · b5 22.4 | astar 16.29 · 32B 13.18 · PIGINet 2.02 · SPECTRE 1.69 |

**Result — effort is a wash, settled at full scale.** A 4-problem low-vs-medium pilot could not
discriminate: `stratified_per_stratum=1` drew only easy-mode DD2D problems (all solved at FP 0
under both efforts), which also **badly under-estimated FP** (pilot 0 vs the true ~27). So a
full-scale **medium-effort DD2D arm** was run: 33.5 vs low 35.23, paired 95% CI [−18.6, +15.1]
(medium better on 16 problems, low on 12, tie 12), medium slightly *more* censored (2 vs 1).
Low chosen — it also matches luna, keeping the swap a clean model+prompt change. §2b wall-clock
is generation-dominated: DD2D 78 s (infer 76.3 + refine 1.9), SB2D 40.8 s (infer 32.0 + refine
8.9).

**Result — DD2D is bimodal.** 14/40 targets are trivially graspable and solved on the first VLM
attempt (FP 0); the rest flail, flooding 50–200 off-pool proposals that all fail geometric
refinement (per-problem FP up to the 200 censor). The mean is a mixture, not a typical case.

**Takeaway-next.** The stronger model + full geometry disclosure does **not** overturn the
qualitative story — it makes it more defensible: DD2D stays a **negative control** (VLM ~parity
with the naive planner order, far behind the learned rankers; reasoning effort doesn't rescue
the packing weakness), and SB2D VLMPlan is a **genuine planner** that now beats the naive order
across the board (b3 0.9 < astar 2.96) yet still trails the learned rankers ~3–4×, so the
representation ordering is unchanged. Two traps banked: **a 1-per-stratum pilot is
unrepresentative on a bimodal env** (pilot on the whole stratum before trusting an effort/model
call), and the Responses API takes no fixed seed so re-runs vary (bare mean like astar; use the
across-problem bootstrap for a spread).

---

<a id="2026-08-08-narrowing-v3-scene-inputs-domain-agnostic-columns"></a>
## 2026-08-08 — Narrowing the v3 scene inputs to domain-agnostic columns: probe, gates, retrain

<!--strip-->
> **id** `2026-08-08-narrowing-v3-scene-inputs-domain-agnostic-columns` · **status**
> active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d
<!--/strip-->

**What.** Made the v3 scene inputs domain-agnostic (ADR
[2026-08-08](../decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)):
`obj_is_target` → `obj_is_goal` (goal-atom derived), `obj_rel` narrowed 8 → 3 to the
anchor-free `[area, sinθ, cosθ]` (target-anchored offsets + area ratio + the privileged
`concave` flag cut). Gated the change, retrained both deployed models, re-derived numbers.

**Result — gates (no training).**
- **Gate 1:** `goal_objects(ep) == {o : o.is_target}` on **720/720** episodes, all four dd2d_v4
  variants × all splits, 0 mismatches, 0 goal-orphans. The boolean swap is a set-level no-op on
  DD2D.
- **Gate 2:** built `obj_is_goal` (from goal atoms) vs `obj_is_target` (from the flag) **under
  canonicalization** — array-equal on **720/720** episodes. Byte-level no-op, the D-8-style
  exact-absence check for the boolean.

**Result — Step-0 pricing probe (existing deployed checkpoints, 3 seeds each).** An
inference-time hook (`zero_scene_cols`, the geometry analogue of `suppress_records`) blanked the
columns on the *un-narrowed* deployed models and measured FP. Verified the hook actually mutates
the batch (`is_target` 1.0→0, `rel_anchor` 23.0→0, `area`/`sinθ`/`cosθ` preserved) so an
identical result is genuine inertness, not an inert switch.

| arm | DD2D ALL | SB2D ALL |
|---|---|---|
| baseline | 5.78 ± 0.10 | 1.69 ± 0.26 |
| zero `is_target` | 5.78 ± 0.10 | 1.69 ± 0.26 |
| zero anchored `obj_rel` cols | 5.78 ± 0.10 | 1.69 ± 0.26 |
| zero both | 5.78 ± 0.10 | 1.69 ± 0.26 |

Δ (any arm − baseline) = **+0.00, CI [+0.00, +0.00]** on every stratum, both envs. The deployed
ranker is completely inert to these columns for ranking — pre-registered gate ("< 1 FP on DD2D")
passed with room to spare, consistent with the 2026-08-06 geometry-intervention finding.

**Result — asymmetry to quote.** On DD2D the change is provably inert (`is_goal ≡ is_target`,
probe Δ 0.00), so the retrain *reproduces*. On SB2D the boolean flips from **all-zero to a live
goal channel** — a b5 scene now marks all 5 buttons where `is_target` marked none — an *addition*
the removal-only probe cannot price, so SB2D's retrained number is a genuine measurement.

**Result — retrain (3 seeds).** The retrain regressed *on the mean*: DD2D **6.63 ± 0.68** vs
baseline 5.78 (paired +0.85, CI [−0.05, +1.97] — the whole gap at **s1 7.41 ± 2.94** vs 3.44;
s2/s3 tied), SB2D **2.10 ± 0.43** vs 1.69 (paired +0.41, CI [+0.11, +0.80], excludes 0). But *not*
information loss: the **best seed matched/beat the baseline on both** (DD2D 6.03, SB2D 1.62), and
the across-seed std jumped ~7× (0.10 → 0.68). The removed columns are inference-inert (probe
Δ0.00), so this is optimization *variance*, diagnosed and fixed the next day — see the
2026-08-09 entry. Deployed numbers after the fix: DD2D **5.92**, SB2D **1.84** (both tie the
frozen baseline).

**Takeaway-next.** The scene channel was carrying DD2D-specific semantics for free; removing it
costs nothing on the deployed models and makes the input surface honest for N-target problems and
future environments. Deferred, characterized: scale-invariance of `area`/`boundary` (probe kept
them, so unmeasured), and a proper goal-atom token stream (would subsume `obj_is_goal` and handle
nullary/relational goals). The next real degeneracy is `manipulated = args \ goal_objects`, which
makes `jaccard ≈ 1.0` on 98.3% of SB2D candidates — applicable everywhere, so it stays under the
rule, but it is where the SB2D scene features collapse.

---

<a id="2026-08-06-dd2d-shape-size-sweep-geometry-interventions-size"></a>
## 2026-08-06 — DD2D shape-size sweep + geometry interventions: size is not the s2 driver

<!--strip-->
> **id** `2026-08-06-dd2d-shape-size-sweep-geometry-interventions-size` · **status**
> active · **tracks** method, evaluation, env-dd2d
<!--/strip-->

**What.** Chased the shape-only s2 number (adaptive **17.27** vs in-dist 10.49;
[2026-08-04](#2026-08-04-dd2d-shape-only-generalization-shapes-isolated-count)) to a cause. The
inspector suggested SPECTRE fails when it must pick up the new tee/cross figures; the guess was
they are *bigger* (harder to fit the buffer). First established that **v3 is image-free but
geometry-AWARE** — `build_v3_example` requires `scene_geometry`; the `SceneEncoder` feeds every
candidate a footprint point-set encoding of each object's boundary, its pose, raw `o.area`,
area/target-area, and a concave flag — so the size hypothesis is a testable *representational*
claim, not just a physical one. Then tested it three ways (gate → input interventions → physical
shrink + variance control). New tooling: `spectre_intervene_geometry.py` (rewrite tee/cross model
input on fixed problems), a `--family-size-scale` collector lever, `spectre_probe_shape_geometry.py`.

**Result — size is not the driver, at either level.**

*Gate (stored data, no compute).* By convex-**hull** footprint (packing-relevant — nothing packs
into a concavity) cross 46.9→67.9, tee 43.2→60.1, still rank 5th–6th of 9. Failures: pick 66% /
retrieve 28% / **place-buffer-volume 5.3%**. Buffer hull-occupancy ~40–48% even for *feasible*
candidates and **lower** for infeasible ones (0.33–0.37) — capacity never binds; the constraint is
grasp accessibility. New shapes are the failing object 4.8% of the time (< their ~20% share).

*Input interventions (paired; same problems + labels; only tee/cross model-input geometry rewritten;
score the same dd2d_v4 checkpoint).* All three are **inert to the digit**:

| adaptive FP | ALL | s2 | paired Δ vs baseline (ALL / s2) | max\|Δscore\| |
|---|---|---|---|---|
| baseline | 6.77 | 17.27 | — | — |
| hullarea (area 45→hull 64) | 6.77 | 17.27 | +0.00 [+0.00,+0.00] / +0.00 | 0.0003 |
| hullshape (boundary→convex hull) | 6.77 | 17.27 | +0.00 [+0.00,+0.00] / +0.00 | ~0 |
| scale07 (input ×0.7) | 6.77 | 17.27 | +0.00 [+0.00,+0.00] / +0.00 | 0.0009 |

The astar arm (reads only labels) is byte-identical — the built-in null. So **SPECTRE's ranking
does not use the new shapes' geometry input**; correcting area, convexifying, and shrinking all do
nothing. (A footprint-OOD probe: tee is the *most* novel shape to the encoder, kNN-to-train 0.105 >
box 0.070; cross mid 0.039 — yet even that does not move the ranking.)

*Physical shrink + variance control.* The `--family-size-scale` lever collected
`dd2d_v4gen_shapeonly_sz07` (tee/cross ×0.7, hull 60/68→29/33). It *appears* to help hugely — but a
**fresh un-shrunk control** (band 7) shows the improvement is collection variance:

| SPECTRE-adaptive s2 | astar-dist s2 |
|---|---|
| baseline (band 5, unshrunk) **17.27** | 14.20 |
| fresh (band 7, unshrunk, 3 seeds) **5.63** | 14.60 |
| sz07 (band 6, shrunk ×0.7) **3.17** | 13.70 |

**astar s2 is stable at 14–15 across all three** while SPECTRE's swings 17→5.6→3.2, and a fresh
*un-shrunk* draw already reads **5.63 — below the in-dist 10.49**. So the shape-only s2=17.27 does
not reproduce; it was a high-variance draw, and the gap between the two *un-shrunk* collections
(17.27 vs 5.63) dwarfs the shrink's residual (5.63 vs 3.17). (Full all-methods: sz07 adaptive ALL
2.79 / static 15.00 / PIGINet 22.68 *worse* / astar 34.73 — no uniform difficulty change.)

**Takeaway/next.** A size sweep does not resolve the s2 number because **there is no size-driven
deficit** — physical packing is 5% of failures, SPECTRE is provably blind to the new shapes'
geometry input, and s2 is dominated by collection/pool-composition variance (~1.5 unique feasible
solutions; SPECTRE's learned order is sensitive to which land in the k=200 pool — the
[2026-08-02](#2026-08-02-s2-ood-degradation-pool-composition-artifact-model) finding, now shown to
bite the shape-only set too). Read shape/size-generalization at s3 or against a fresh un-shrunk
control, never a single-collection s2 point. ADR + the new invariant:
[decisions/07 2026-08-06](../decisions/07-stickbutton2d.md#2026-08-06-shape-generalization-s2-deficit-collection-variance-shape).
(All numbers 3 seeds, uncensored, n=40, dd2d_v4 checkpoints train-old / test-new.)

**Addendum — tee/cross now DEFAULT to 0.7x, and the object-gen test uses that default.** Given
the size interventions are inert and the deficit is variance (above), we make the two new
concave figures **default to 0.7x linear** (`shapes._FAMILY_DEFAULT_SCALE`; an explicit
`--family-size-scale tee=1.0` restores nominal). The rationale is design, not a size-drives-FP
claim: at 0.7x the tee/cross hull footprint (~29/33) grasps and packs cleanly in the shallow
buffer while still being unseen at test time. `compare_methods.py`'s `dd2d_gen_shapeonly` now
reads a 0.7x collection (the `_sz07` draw, band 6, PIGINet scored on it), so the live object-gen
comparison is at the default size. There: SPECTRE-adaptive **ALL 2.79** (≤ in-dist 5.78 — shape
generalization free) beats **PIGINet 22.68** decisively (paired **−19.88, CI [−31.04, −10.07]**)
and astar 34.72. **Honest caveat:** 2.79 is one 0.7x draw and s2 is variance-dominated (draws of
this test read adaptive s2 = 17.27 / 5.63 / 3.17 at fixed astar ~14); the *robust* statement is
the ALL win over PIGINet, not the exact 2.79. The full-size draws (band 5 `dd2d_v4gen_shapeonly`,
band 7 `_fresh` — the latter scored all-4 at adaptive 5.03 / PIGINet 18.30, paired −13.27) and
their 2026-08-04 entry are retained on disk as the historical / size-sweep record. tee/cross are
held out of training, so the default-size change needs no retraining.

---

<a id="2026-08-04-dd2d-shape-only-generalization-shapes-isolated-count"></a>
## 2026-08-04 — DD2D shape-only generalization: shapes isolated from count

<!--strip-->
> **id** `2026-08-04-dd2d-shape-only-generalization-shapes-isolated-count` ·
> **status** active · **tracks** env-dd2d, evaluation, method
<!--/strip-->

**What.** The `dd2d_v4gen_shape` set moved two variables at once (unseen 13–15 blocker
count *and* the new tee/cross figures), and its s2 FP blew up. To attribute that we built a
**shape-only** held-out set, `dd2d_v4gen_shapeonly`: the two concave figures forced into
every scene (≥1 tee + ≥1 cross) at the **trained 9–12 blocker count** (the collector's
default count mechanism — no `--n-items-*`), 40 problems stratified s0–s3, seed band [5M,6M).
Grasp correctness first re-verified with `demo_grasp_concave --families tee cross` (0 floating
cells; 5–10 internal/concave-region grasps per shape; all clutter scenes grasped). Scored the
dd2d_v4-trained SPECTRE + PIGINet train-old / test-new via a new
`precompute_dd2d_cache.py --test-variant` (protocol ADR
[decisions/07 2026-08-04](../decisions/07-stickbutton2d.md#2026-08-04-shape-only-dd2d-gen-variant-precompute---test-variant)),
and added it as a `compare_methods.py` dropdown env (`dd2d_gen_shapeonly`) with FP + §2b
wall-clock. Two instruments (the compare cache and `spectre_score_v3.py`) agree to the digit.

**Result (mean FP, 3 seeds, uncensored, n=40; in-dist dd2d_v4 headline in brackets):**

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| **SPECTRE-adaptive** | **6.77 ± 0.81** | 0.00 | 3.80 | 17.27 | 6.03 |
| PIGINet | 15.27 ± 1.57 | 0.37 | 14.17 | 7.73 | 38.80 |
| SPECTRE-static | 22.55 ± 1.46 | 0.00 | 18.93 | 35.07 | 36.20 |
| astar-dist | 31.45 | 0.00 | 2.20 | 14.20 | 109.40 |
| *(in-dist adaptive)* | *5.78* | *0.00* | *3.44* | *10.49* | *9.19* |

- **The shape shift is mild and s3 IMPROVES.** Adaptive ALL 6.77 vs in-dist 5.78 (~1.17×),
  far under the count+shape set's 11.26 (~1.9×). s0/s1 unchanged, **s3 9.19→6.03**, s2
  10.49→17.27 — moderate, and nowhere near the count-confounded set's s2 (~32). So the severe
  s2 OOD degradation reported 2026-08-02 was **primarily count-driven pool composition, not the
  shapes**; the residual shape-only s2 lift is real but small. Read the generalization at s3
  (in-regime), where the shapes if anything *help*.
- **Representation win survives OOD; adaptivity carries it here.** Adaptive beats PIGINet
  (6.77 vs 15.27, ~2.3×) and astar (31.45); paired bootstrap vs astar −24.68, CI
  [−41.95, −8.88]. But **SPECTRE-static (22.55) is *worse* than PIGINet** — under the shape
  shift the static t=0 order degrades badly and the failure-conditioned re-ranking
  (static→adaptive 22.55→6.77) recovers it. In-dist the static representation carried ~73% of
  the margin; that split **inverts** OOD.
- **§2b wall-clock (2 s cap): adaptive is FP-best but the SLOWEST capped** — astar 3.22 <
  PIGINet 3.51 < static 4.41 < adaptive 6.68 — inverting the in-dist headline where adaptive
  was fastest. It is all s2 (adaptive 19.65 s), where the rollout's many failed attempts each
  incur a full pool re-scoring (infer_s ≈3.6 s ALL) on top of refinement. The cap costs
  adaptive +0.67 FP (6.77→7.44) vs DD2D's near-free +0.05, because it bites exactly those
  expensive s2 near-feasible candidates. 0 problems censored by the cap.

**Takeaway/next.** Isolating the variable pays off: the s2 alarm from `dd2d_v4gen_shape` was
mostly the count regime, not the new shapes — shape-only s2 sits roughly midway and s3
improves. The learned ranker's advantage over both PIGINet and the planner order is robust to
an unseen-shape shift, but the *source* of the advantage moves: adaptivity, not the static
representation, is what holds up OOD here. The wall-clock inversion (FP-best ≠ time-best when
a stratum's failures are both frequent and re-scoring-heavy) is the same lesson §2b taught on
SB2D, now visible under distribution shift.

---

<a id="2026-08-03-sb2d-2b-wall-clock-spectre-adaptive-fastest-per-env"></a>
## 2026-08-03 — SB2D §2b wall-clock — SPECTRE-adaptive fastest; per-env 10 s cap; astar benefits most

<!--strip-->
> **id** `2026-08-03-sb2d-2b-wall-clock-spectre-adaptive-fastest-per-env` · **status**
> active · **tracks** evaluation, env-stickbutton2d, method
<!--/strip-->

**What.** Brought the §2b wall-clock-to-first-success breakdown to parity on SB2D (it was a stub;
DD2D-only). The raw material already existed — every SB2D `OutcomeRecord` carries a measured
`refinement_wall_clock_s` — so this was wiring, not re-collection: made `REFINE_CAP_S` per-env,
added an SB2D branch to `_measure_plan_gen` (times the acyclic pool draw via the new
`collect.time_pool_generation`), grafted SPECTRE's timing from the `stickbutton2d_v1` legacy cache
(new `compare.merge_time_records`, mirroring the FP graft), and flipped `SB2D_KINDER.has_timing`.
Rebuilt the caches: `precompute_dd2d_cache.py --env-variant stickbutton2d_v1 --methods spectre3
--no-ablations --force` (SPECTRE timing into the legacy cache) and `--env-variant
stickbutton2d_v1_kinder --methods astar piginet --force` (astar/PIGINet timing into the primary
cache, both under the new cap). FP headline unchanged after the `--force` rebuild (adaptive 1.69,
static 1.98, PIGINet 2.28) — timing fields added, FP identical. Protocol ADR:
[decisions/07 2026-08-03](../decisions/07-stickbutton2d.md#2026-08-03-sb2d-2b-wall-clock-breakdown-parity-dd2d).

**Cap = 10 s (vs DD2D's 2 s).** SB2D feasible refines run to seconds (per-candidate p95 10.6 s),
too slow for a DD2D-style cap-above-the-whole-distribution to fit under the 20 s budget. 10 s
instead clears each problem's *fastest*-feasible (per-problem min-feasible maxes at 8.84 s) with
margin, so `_feasibility_at_risk(10) = 0` (no problem censored), while still biting hard: **33 % of
all per-candidate refines exceed 10 s** because SB2D's *failures* run to the 20 s budget (p75 of all
per-candidate refines is 20.0 s; only ~4 % of candidates are feasible).

**Result (ALL, test n=100 for pool methods / n=40 VLMPlan, under the 10 s cap):**

| method | total s | plan-gen | infer | refine | uncapped s | FP (uncap→cap) |
|---|---|---|---|---|---|---|
| **SPECTRE-adaptive** | **11.17** | 0.59 | 0.06 | 10.53 | 13.98 | 1.69 → 2.03 |
| SPECTRE-static | 12.64 | 0.59 | 0.01 | 12.04 | 16.56 | 1.98 → 2.32 |
| PIGINet | 15.15 | 0.59 | 0.04 | 14.52 | 24.22 | 2.28 → 2.41 |
| VLMPlan-GPT5.6 | 53.61 | 0.00 | 37.18 | 16.43 | 105.28 | 11.85 → 11.97 |
| astar-dist | 97.40 | 0.59 | 0.00 | 96.81 | 145.64 | 16.29 → 16.49 |

Plan-gen per stratum (kinder, 3 problems each): **b1 2.09 s** (the padded-plan `_RAW_CAP` grind —
b1's 200-slot pool is mostly stick-cycle duplicates the acyclic filter discards), b2 0.16, b3 0.034,
b5 0.072; the ALL 0.59 s is the balanced-stratum mean. Refinement dominates the total everywhere;
inference is a sliver (VLMPlan excepted — its generation *is* its inference, 37 s).

**Takeaway — the DD2D cap narrative does not transfer, and the SB2D cap has a real FP cost.**
On DD2D the cap *flips* v3-adaptive from ~equal-uncapped to fastest, because there astar's failures
are cheap dead-ends and v3's few failures are the expensive near-feasible traps. On SB2D **all
failures are uniformly expensive** (they run to the 20 s budget), so FP and wall-clock are aligned:
SPECTRE-adaptive is fastest **both** capped (11.17 s) and uncapped (13.98 s), and the cap mainly
compresses astar's huge deficit (−48 s, 145.6 → 97.4). Two things to quote with the table: (1) the
cap helps the **highest-FP** method most in absolute seconds (astar), the reverse of DD2D; (2) the
cap costs the learned methods a **real +0.3 FP** (adaptive 1.69 → 2.03), an order of magnitude more
than DD2D's +0.05, because SB2D's cap sits *inside* the feasible distribution (it abandons slow
non-fastest feasibles) rather than above it. So on SB2D the cap is a genuine wall-clock/FP trade,
not the near-free accounting change it is on DD2D.

---

<a id="2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong"></a>
## 2026-08-03 — VLMPlan on a frontier VLM (gpt-5.6-luna): strong on SB2D, weak on DD2D

<!--strip-->
> **id** `2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong` · **status** active ·
> **tracks** baselines, evaluation, env-stickbutton2d, env-dd2d
<!--/strip-->

**What.** Ran the VLMPlan baseline with a **frontier** VLM — `gpt-5.6-luna` over the OpenAI
Responses API (`reasoning.effort: low`, `max_output_tokens 16384`) — replacing the local Qwen
arms, to answer "did you just try a frontier VLM?" on the record. Stratified 40 test problems/env
(10/stratum), native on `dd2d_v4` and `stickbutton2d_v1_kinder`; SB2D uses the new
`kinder_labeled` image (kinder's real pixels + Set-of-Mark labels). Protocol ADR:
[decisions/07 2026-08-03](../decisions/07-stickbutton2d.md#2026-08-03-frontier-vlm-vlmplan-arm-gpt-5-6-luna-kinder-labeled).
Spend ≈ $1.2 total. **Label-agreement 0.983 (DD2D) / 1.000 (SB2D)** — numbers are defensible.

**Result (uncensored FP, VLMPlan n=40 / others n=100, per-stratum means comparable):**

DD2D (`dd2d_v4`) — s0..s3:

| method | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **5.78** | 0.0 | 3.4 | 10.5 | 9.2 |
| PIGINet | 17.27 | 0.1 | 5.0 | 18.8 | 45.2 |
| VLMPlan-32B (qwen) | 23.55 | 6.8 | 5.0 | 13.2 | 69.2 |
| astar-dist | 34.52 | 0.0 | 2.2 | 17.1 | 118.8 |
| **VLMPlan-GPT5.6** | **62.98** | 43.2 | 35.9 | 46.8 | 126.0 |

StickButton2D (kinder) — b1..b5:

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **1.69** | 0.1 | 0.2 | 1.1 | 5.3 |
| PIGINet | 2.28 | 0.1 | 0.3 | 1.2 | 7.5 |
| **VLMPlan-GPT5.6** | **11.85** | 0.0 | 1.0 | 11.4 | 35.0 |
| VLMPlan-32B (qwen) | 13.18 | 0.7 | 1.3 | 6.2 | 44.5 |
| astar-dist | 16.29 | 0.1 | 0.6 | 3.0 | 61.6 |

Wall-clock to first success (DD2D §2b, deployed 2 s cap): VLMPlan-GPT5.6 **63.7 s**
(plan-gen 0 + infer 57.8 + refine 5.8; uncapped 78.5), vs v3-adaptive 1.8 s / astar 3.0 s —
**generation-dominated and ~20–35× slower** than a learned ranker. On SB2D `first_success_from_vlm
= 35/40, 0 censored`; on DD2D 24 vlm / 11 fill / **5 censored** (all s3).

**Takeaway / next.**
- **A frontier VLM does not change the conclusion.** On both environments VLMPlan is decisively
  beaten by the learned rankers (DD2D ~11×, SB2D ~7×), and the frontier upgrade over the local
  32B is roughly a wash (SB2D 11.85 vs 13.18; DD2D *worse*, 62.98 vs 23.55). The "just ask a VLM"
  objection is now answered with the strongest available model.
- **DD2D is a genuine negative control.** `gpt-5.6-luna` is the **worst** method on DD2D — worse
  than the naive planner order and worse than the local Qwen — because it **over-stages
  confidently and never stalls**: on trivial s0 problems it proposes only 3-blocker stagings that
  all fail the packing refinement (s0 43.2 vs astar 0.0), accruing many diverse failed attempts.
  Continuous packing is exactly where an abstract/generated plan is expected to lose.
- **SB2D: a real planner, still ~6× short.** It self-solves 35/40 and beats astar overall, and
  the **kinder-labeled image works** — luna grounds `circle_N` correctly (0 fill on the easy
  strata). But it *over-thinks b3* (11.4 vs learned ~1.2, and worse than astar's 3.0), and only
  wins at b5 where astar's default order is pathological. So a frontier VLM's edge over the naive
  order is narrow and stratum-specific.
- **Pilot lesson (re-logged): a 1/stratum pilot mis-estimates badly.** The SB2D pilot drew the
  *first* pid in each band (the easiest) and read 2.5 ALL; the stratified 40 reads 11.85. Same
  family as *stride-never-truncate* — pilot on a stratified sample, not the head.
- Not run: `reasoning.effort` sweep (would over-thinking *more* help or hurt DD2D?) and a
  full-100 SB2D. Neither changes the qualitative picture.

---

<a id="2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate"></a>
## 2026-08-02 — DD2D s1 wall-clock blow-up diagnosed; per-candidate refinement cap

<!--strip-->
> **id** `2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate` · **status**
> active · **tracks** method, evaluation, env-dd2d
<!--/strip-->

**What.** Investigated why §2b's DD2D wall-clock showed SPECTREv3-adaptive *slower* than the
naive planner order overall (5.89 vs 4.94 s ALL), with the whole gap at s1 (11.99 ± 7.81 vs
0.26 s) — suspicious because v3 wins every other stratum. Then added a per-candidate
refinement cap and re-measured. Protocol/decision in
[decisions/07 2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration).

**Result — the s1 blow-up is real, not a bug, but the number is noisy.**
- Timing math verified; the table reproduces from the cache.
- v3 is genuinely (modestly) worse than astar at **s1 on FP** — 3.44 vs 2.24 (astar's *worst*
  s1 problem is only FP=5). The planner-cost order already ranks s1's short/cheap feasible
  plans well.
- The ~1.2-attempt FP gap becomes a ~46× wall-clock gap because of *which* candidates fail.
  Across all test episodes feasible refinements are uniformly cheap (s1 mean 0.32 s, p95
  0.44 s) while near-feasible infeasible candidates burn the full 20 s budget. astar s1 refine
  = 0.13 s (cheap dead-ends); v3 = 11.2 s (expensive traps). **Worked case pid 1250023**: pool
  200 with **29 feasible @ 0.17 s each**, the model ranked **15 of the 20 s traps ahead of all
  29** → 240 s, FP=15 where random ≈ 6 (*worse than random* on that problem).
- The 12.00 ± 7.80 is a heavy-tailed 3-seed mean dominated by ~4–5 recurring hard s1 problems
  (pids 1250011/1250015/1250023…) whose FP swings 2→15 across seeds.

**Result — a per-candidate cap fixes it, cheaply and safely.**
- Safety: per-candidate (not per-problem), so a problem is lost only if *every* feasible
  candidate exceeds the cap. Min-feasible refine time per problem is mean 0.103 s, **max
  0.243 s** → **0/100** problems censored at any cap ≥ 1 s; the median problem keeps ~20 sub-2 s
  feasible candidates.
- Under a **2 s** cap (deployed), ALL wall-clock to first success (3 seeds):

  | method | ALL | s0 | s1 | s2 | s3 | uncapped ALL |
  |---|---|---|---|---|---|---|
  | **SPECTREv3-adaptive** | **1.79 ± 0.44** | 0.43 | 2.40 | 1.88 | 2.45 | 5.89 |
  | SPECTREv3-static | 2.53 ± 0.71 | 0.44 | 2.04 | 3.14 | 4.50 | 7.99 |
  | astar-dist | 2.96 | 0.40 | 0.26 | 1.35 | 9.81 | 4.94 |
  | PIGINet | 3.14 ± 0.39 | 0.71 | 2.01 | 3.88 | 5.98 | 8.35 |

- v3-adaptive becomes the **fastest** method; its s1 collapses 11.99 → 2.40. **FP cost of the
  cap** (ALL): adaptive +0.05 (5.78 → 5.83), astar +0.00 (failures already sub-cap), PIGINet
  +0.23, static +0.26 — a faithful re-run (the adaptive order diverges on 6/300 cells), not a
  `min(t, cap)` accounting.

**Takeaway — next.** The uncapped wall-clock over-punishes the learned ranker: its few failures
are the *expensive* near-feasible candidates a good ranker still tries, so bounding per-skeleton
refinement (which the cap does) is what lets the "try few candidates" advantage show in seconds.
Do not read an uncapped wall-clock as v3's deployed cost. The **residual** is s1, where v3 still
trails astar (2.40 vs 0.26) — the modest s1 FP deficit — a candidate for the model-side R1
cost/enumeration-index feature (give the ranker the planner-cost order it currently cannot see).

---

<a id="2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s"></a>
## 2026-08-02 — StickButton2D PIGINet crops re-sourced from kinder's renderer (stickbutton2d_v1_kinder)

<!--strip-->
> **id** `2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s` · **status**
> active · **tracks** baselines, env-stickbutton2d, data
<!--/strip-->

**What.** Re-sourced the PIGINet baseline's SB2D image crops from **kinder's own renderer**
instead of the schematic rasteriser (`SB2DDomain.crops`, which drew each object as a lone
polygon on a blank background). Delivered as a new env_variant `stickbutton2d_v1_kinder`,
built by a converter (`experiments/spectre/sb2d_render_convert.py`) that copies every record
verbatim and only re-renders the pixels by resetting the env from the stored seed. Per
problem it materialises per-object crops (`render_2dstate` windows, world side 1.4 m, 300 dpi
→ 420²) plus a full `scene.png`. The reader is a thin `SB2DKinderDomain(SB2DDomain)` selected
by `make_sb2d_domain`. Rationale and the five load-bearing choices are in
[decisions/07 2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new).

**Result — conversion + validation.**
- The kinder crops render correctly and carry **real scene context** — a button crop shows
  the table band, the wall, and the stick tip, not a lone disc. On a multi-button (b5) scene
  the per-button crops are **not** pixel-identical (they differ by position/context), the
  direct contrast to the schematic where every unpressed button is the same red disc.
- Records are copied **byte-identical** (geometry, skeleton pool, outcomes, object registry,
  goal all `==` the v1 source; only `provenance.env_variant` differs), which is what licenses
  grafting SPECTRE from v1. Vocab is identical to v1's; `spectre_check_pipeline` passes.
- `env.reset(seed=pid)` + `render_2dstate` is **deterministic** (re-render reproduces
  identical pixels). All seven unit tests pass in ~1 s.

**Result — the comparison (ALL FP, test n=100, 3 seeds; PIGINet retrained on kinder crops).**

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **1.69 ± 0.26** | 0.08 | 0.24 | 1.13 | 5.29 |
| SPECTREv3-static | 1.98 ± 0.28 | 0.08 | 0.32 | 1.52 | 5.99 |
| **PIGINet (kinder)** | **2.28 ± 0.29** | 0.07 | 0.35 | 1.17 | 7.55 |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |

Paired bootstrap over the 100 problems (negative = v3 better): v3-static − PIGINet = **−0.31,
CI [−0.95, +0.36]**; v3-adaptive − PIGINet = **−0.60, CI [−1.24, +0.08]** — **neither
separates**. The adaptive increment does: adaptive − static = **−0.29, CI [−0.51, −0.08]**.

**Takeaway — the valid pixels did not overturn the finding; they reinforced it.** With real
kinder crops PIGINet is if anything *slightly worse* than with the schematic (2.28 vs the
prior 2.02, the drop entirely at b5: 7.55 vs 6.39), so the representation advantage **still
does not separate on SB2D**. The honest cross-environment statement is unchanged: the abstract
representation wins on DD2D, ties on SB2D; the adaptive increment is positive on both (−0.29
here, matching the schematic's −0.29). The pre-registered caveat held — the crop's added
context is positional, and since two unpressed buttons are identical discs in the real env,
that context is net-neutral-to-mild-distractor, not new signal. **Validity was the point, not
a better number: PIGINet now reads the environment's own pixels, and the tie survives it.**

---

<a id="2026-08-02-dd2d-wall-clock-first-success-fp-flatters"></a>
## 2026-08-02 — DD2D wall-clock to first success: FP flatters the learned ranker (its failures are the expensive ones)

<!--strip-->
> **id** `2026-08-02-dd2d-wall-clock-first-success-fp-flatters` · **status** active ·
> **tracks** evaluation, env-dd2d, tooling
<!--/strip-->

**What.** Added a **wall-clock-to-first-success** section to `compare_methods.py` (DD2D):
per method, seconds to the first successful refinement = abstract-plan-generation + inference +
refinement, summed over the candidates each tries until the first feasible. FP counts failed
attempts; this weighs each by its real cost (a failed refinement runs ~15 ms to ~20 s) and adds
inference — to answer whether the learned ranker's inference is worth it in practice. Refinement
reuses the stored per-candidate `refinement_wall_clock_s` (every method sums the *same* times over
its own order); inference measured on GPU (~22 ms/step, tensorization-dominated); plan-gen a
per-stratum shared constant. All cached in the compare cache. FP table byte-identical after the
`--force` rebuild (timing fields are additive).

**Result (dd2d_v4 test, n=100, 3 seeds). Breakdown of ALL, seconds:**

| method | plan-gen | inference | refinement | **total** | (FP) |
|---|---|---|---|---|---|
| astar-dist | 0.22 | 0.00 | 4.72 | **4.94** | 34.5 |
| SPECTREv3-adaptive | 0.22 | 0.51 | 5.17 | **5.90** | 5.8 |
| SPECTREv3-static | 0.22 | 0.03 | 7.72 | **7.97** | 21.1 |
| PIGINet | 0.22 | 0.27 | 7.86 | **8.35** | 17.3 |

Per-stratum total (s): astar 0.40/0.26/1.35/**17.77**; v3-adaptive 0.44/**12.00**/2.92/**8.25**;
v3-static 0.42/9.01/7.94/14.49; PIGINet 0.71/8.47/9.65/14.57.

**Takeaway — FP flatters the learned ranker.** SPECTREv3-adaptive has **6× lower FP** than astar
(5.8 vs 34.5) yet is **not faster in wall-clock** (5.90 vs 4.94 s). The reason is the whole point
of measuring time: astar's many failures are **cheap dead-ends** (~0.14 s each — 34.5 × 0.14 ≈
4.7 s), while SPECTRE's few failures are the **expensive near-feasible** candidates it correctly
ranks high, which the refiner burns time trying to refute (~0.89 s each). So a better ranking
surfaces the costlier failures, and the FP win does not carry to wall-clock. Robust sub-findings:

- **Inference is small** — v3-adaptive 0.51 s (per-step × steps), v3-static 0.03 s (one pass),
  PIGINet 0.27 s (BCE head; CLIP features cached, so this undercounts a from-scratch encode).
  Refinement dominates every method; plan-gen ~0.22 s is a shared constant.
- **The win is concentrated at s3**, where astar's *volume* of failures wins out: v3-adaptive
  8.25 s vs astar 17.77 s. At s1/s2 the learned ranker is slower (expensive failures + inference):
  s2 2.92 vs 1.35, s1 12.00 vs 0.26.
- **The ALL "adaptive slightly slower" is s1-sensitive and noisy** — s1 reads 12.00 ± 7.80, a few
  problems where the ranker picked a candidate that refined to the ~20 s budget before failing.
  Read the headline as *"no clear wall-clock win overall despite 6× fewer attempts,"* not a precise
  loss. What is robust across strata is the per-failure cost gap (astar cheap, SPECTRE expensive)
  and that inference is the small term.

**Caveats.** The refine times are a within-collection *relative* measure (8-way worker
parallelism, `time_budget=20 s` per candidate) — fair across methods since each sums the same
per-candidate times, but not an isolated single-core benchmark. Plan-gen is a regenerated
per-stratum proxy (PYTHONHASHSEED-dependent). ADR: [decisions/07
2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-wall-clock-to-first-success-added-compare-methods-reuses-stored).

---

<a id="2026-08-02-s2-ood-degradation-pool-composition-artifact-model"></a>
## 2026-08-02 — s2 OOD degradation is a pool-composition artifact, not model or generator failure

<!--strip-->
> **id** `2026-08-02-s2-ood-degradation-pool-composition-artifact-model` · **status**
> active · **tracks** env-dd2d, evaluation, method
<!--/strip-->

**What.** Root-caused the s2 column of the [2026-08-01 DD2D generalization
result](#2026-08-01-dd2d-generalization-v3-vs-astar-unseen), where v3's FP jumped 10.49 → 30.23
under the unseen-count shift (and dominates the ALL mean). Prompted by the objection that s2
(clear 2) cannot be harder than s3 (clear 3) by construction. All read-only probes on the
collected episodes + the seed-0 checkpoint; no new collection/training/scoring.

**Result.** The intrinsic difficulty ladder is intact and the generator is sound — what shifts is
the *pool's feasible composition*.

- **Not a generator bug.** s2 labels are 100% correct (every s2 problem has a real feasible
  2-subset, none shorter — pool-implied mfs matches the label 10/10 OOD, 25/25 in-dist). Execution
  difficulty is monotone as expected: astar-dist FP **s3 167 ≫ s2 28**; generation keep-rate
  **s3 20% ≪ s2 91%**. s3 is genuinely harder to execute; only the *model's* FP inverts.
- **s2 is genuinely clear-2 but has ~1.5 unique solutions.** 99% of feasible triples are redundant
  supersets of a feasible pair (in-dist 567/575; OOD 8/8); genuine-3 solutions (no feasible pair
  inside) ≈ 0. The circular target admits 18 diametric grasp axes and an axis opens only when its
  antipodal blocker pair is cleared; `crowd=5` is odd → no antipodal pair → ~1.5 feasible pairs.
- **The degradation is dominantly a pool-composition artifact.** Per-length feasibility in the
  k=200 pool:

  | s2 pool, per length | in-dist `dd2d_v4` | unseen count |
  |---|---|---|
  | 2-subset (len 5): candidates / feasible | 96.6 / 2.84 | 172.2 / 1.80 |
  | 3-subset (len 7): candidates / feasible | 92.2 / **23.0** | **18.4 / 1.14** |
  | total feasible | 25.8 | 2.9 |

  In-distribution the feasibility mass is ~23 (redundant) triples; the feasible-**pair** count is
  ~stable OOD (2.84 → 1.80). What collapses is the triples, because at 14 blockers C(14,2)=91
  pairs flood the short-first k=200 cap (→172 pair candidates) and crowd the triples out (92 → 18
  enumerated). The pool covers ~100% of possible pairs but almost none of the triples. So the
  in-dist FP=3 was *flattered* by redundant-triple padding; the shift strips it, exposing the
  problems' true ~1.5-solution difficulty. Model s2 FP corr(feasible count) = **−0.82**, median
  FP 3 → 44 (systematic, not outliers).

**Takeaway-next.** The s2 OOD number (and the ALL mean it dominates) is **confounded by pool
composition, not a clean model-generalization signal.** Read the generalization claim at **s3**
(unaffected — s3 was already feasible-scarce in training, so OOD s3 is in-regime; v3 s3 improves
9.19 → 4.87 while astar s3 stays pathological) plus the s2 caveat, not the s2 point estimate.

A generator redesign to give s2 *substantive* feasible-pair diversity was explored and **rejected
as geometrically blocked**: even collar count (the obvious lever) does not raise diversity
(generator sweep: crowd 5/6/8/10 → ~1.5 feasible pairs) and just pushes problems to mfs=3, because
blocking a circular target from all 18 diametric axes (to keep mfs≥2) fights clean single-pair
openings. Decision: **characterize, do not regen** (regen would also imply re-collecting
train/val/test + retraining, re-baselining every SPECTRE result). ADR:
[decisions/07 2026-08-02](../decisions/07-stickbutton2d.md#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact).

---

<a id="2026-08-01-dd2d-generalization-v3-vs-astar-unseen"></a>
## 2026-08-01 — DD2D generalization: v3 vs astar on unseen count and unseen shapes

<!--strip-->
> **id** `2026-08-01-dd2d-generalization-v3-vs-astar-unseen` · **status** active ·
> **tracks** env-dd2d, evaluation, method
<!--/strip-->

**What.** First OOD generalization test of the dd2d_v4-trained SPECTRE v3 checkpoint on DD2D
itself — train-old / test-new, no retraining. Two held-out sets, 40 problems each, stratified
s0–s3 (10 each): `dd2d_v4gen_count` (14–16 items = 13–15 blockers vs the trained 9–12, old
shapes) and `dd2d_v4gen_shape` (same unseen count + a new `tee` and `cross` concave family,
≥1 of each forced per scene). Scored v3 vs astar-dist, uncensored deployed FP, 3 seeds, paired
bootstrap (`spectre_score_v3.py --test-variant … --astar-baseline`). Protocol ADR:
[decisions/07 2026-08-01](../decisions/07-stickbutton2d.md#2026-08-01-dd2d-generalization-test-unseen-count-unseen).

**Result.** In-distribution v3 reproduced 5.78 ± 0.10 exactly (instrument check). Scoring ran
clean — **no OOV and no position-index error** on the longer skeletons from denser scenes,
confirming the vocab/config are count- and shape-invariant.

| set | v3 ALL | v3 vs astar (paired) | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|---|
| in-dist `dd2d_v4` (n=100) | 5.78 ± 0.10 | −28.74 [−39.6,−18.8]* | 0.00 | 3.44 | 10.49 | 9.19 |
| unseen count (n=40) | 9.40 ± 2.62 | −39.95 [−64.0,−18.1]* | 0.00 | 2.50 | 30.23 | 4.87 |
| unseen count+shape (n=40) | 11.26 ± 3.44 | −21.89 [−42.6,−3.8]* | 0.00 | 2.40 | 31.97 | 10.67 |

astar-dist ALL: 34.52 / 49.35 / 33.15; astar s3 is pathological: 118.76 / 166.80 / 108.60.
(* CI excludes 0.)

**Takeaway-next.** v3's advantage over the naive planner order **survives OOD — it still wins
overall on both sets (CI excludes 0)** to unseen counts and unseen shapes. But three caveats,
to quote together:
- **Absolute FP degrades ~1.6–1.9×** (5.78 → 9.40 → 11.26); generalization is not free, and the
  shape set is harder than count-only.
- **The ALL-level win is carried by s3**, where astar's default order is pathological
  (108–167 FP) and v3 stays 5–11. Balanced strata, but the s3 astar catastrophe dominates the
  mean — do not read ALL as a uniform advantage.
- **At s2 v3's advantage collapses under the shift**: from clearly beating astar in-distribution
  (10.49 vs 17.08) to tying/slightly trailing OOD (30.23 vs 28.30 count; 31.97 vs 22.00 shape,
  both within the ±9 seed spread). This amplifies v3's already-characterized in-distribution s2
  deficit; s2 seed variance (±9–10) is high at n=10/stratum, so read it as "advantage lost,"
  not a precise loss. The count-set s3 improving to 4.87 (below in-dist 9.19, low variance) is
  the mirror image — more blockers give more feasible 3-subsets, which the ranker exploits.

Consistent with §0 wishlist property #4 (object-count / identity generalization): the abstract
representation transfers across counts and novel geometries well enough to keep beating the
planner order, while degrading where the harder within-length s2 discrimination already bit it.

---

<a id="2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage"></a>
## 2026-08-01 — DD2D compare cache rebuilt to the unified coverage/waste definition (7.44 to 5.78)

<!--strip-->
> **id** `2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage` · **status** active
> · **tracks** evaluation, env-dd2d
<!--/strip-->

**What.** The DD2D method-comparison notebook was still reporting SPECTREv3-adaptive at
**7.44** — the pre-unification coverage/waste definition — even though the deployed
checkpoint has been `checkpoints_v3_unified` (unified coverage/waste) since 2026-07-31 and
`spectre_score_v3` already reported **5.78 ± 0.10** for it. The gap was a stale cache, not
a disagreement: `_V3_ARMS["spectre3"]` was repointed to the unified dir on 2026-07-31, but
`precompute_dd2d_cache._dir_complete` skips any full directory, so the pre-unified
`spectre3_{static,adaptive}` compare-cache rows were never overwritten.

**Result.** Rebuilt with
`precompute_dd2d_cache.py --env-variant dd2d_v4 --methods spectre3 --no-ablations --force`
(CPU, LM Studio holding the GPU). The notebook headline is now:

| | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **5.78 ± 0.10** | 0.00 | 3.44 | 10.49 | 9.19 |
| SPECTREv3-static | 21.10 ± 2.11 | 0.00 | 14.21 | 27.76 | 42.44 |

matching the `spectre_score_v3` figures exactly (adaptive was 7.44, static 20.66 under the
old definition). The 3× win over PIGINet (17.27) and the ~11.5 FP margin over v2.2 that the
score instrument already showed are now what the notebook renders too.

**Takeaway.** No new science — this only propagates the 2026-07-31 definition
([`decisions/06`](../decisions/06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition))
to the cache the notebook reads, so a future reader does not see two different v3 numbers
depending on whether they ran the score tool or opened the notebook. **The §4 ablation
arms were deliberately *not* rebuilt** (`--no-ablations`): they predate the unification and
score under the old definition by design, as a matched-settings seed-0 study. That makes
§4's `deployed` row (now unified, ~5.78) not directly comparable to its matched
`cov+waste, tokens` arm (~7.90, old) — the note in §4 and the `DD2D` registry caveat both
now say so. The standing lesson stands: **re-cache with `--force` whenever an arm is
repointed**, because `_dir_complete` keeps a stale full directory.

---

<a id="2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar"></a>
## 2026-08-01 — SB2D VLMPlan-32B: capable but tail-limited; beats astar, loses to learned methods

<!--strip-->
> **id** `2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar` · **status**
> active · **tracks** baselines, evaluation, env-stickbutton2d
<!--/strip-->

**What.** Add the VLMPlan-32B row (the zero-training-data corner) to the SB2D four-method
comparison. Scored on a **stratified 40-problem subset** (10/stratum, test split) rather
than the full 100 — a compute choice, since b3/b5 problems VLMPlan cannot self-solve run
to the ~10-round stall cap. `qwen3-vl-32b-instruct`, corrected prompt (domain grounding +
effector-chaining rule), stop-at-first-success on. Label-agreement gate **1.000** (36
samples).

**Result — the four-method table.** Mean rollout FP, test; VLMPlan n=40 (10/stratum, so
its stratum-weighted ALL is comparable to the n=100 rows), others n=100, 3 seeds.

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| SPECTREv3-adaptive | **1.69** | 0.08 | 0.24 | 1.13 | 5.29 |
| SPECTREv3-static | 1.98 | 0.08 | 0.32 | 1.52 | 5.99 |
| PIGINet | 2.02 | 0.08 | 0.32 | 1.31 | 6.39 |
| **VLMPlan-32B** | **13.18** | 0.70 | 1.30 | 6.20 | 44.50 |
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |

Generation: **35/40 solved from VLMPlan's own proposals** (5 fell back to published
order), **0 censored**. Per-problem FP is heavily right-skewed — 22/40 problems are
**FP=0** (the first proposal refines):

| stratum | self-solve | per-problem FP (sorted) |
|---|---|---|
| b1 | 10/10 | 0,0,0,0,0,0,0,0,0,7 |
| b2 | 10/10 | 0,0,0,0,0,0,0,2,4,7 |
| b3 | 10/10 | 0,0,0,0,1,5,5,15,15,21 |
| b5 | 5/10 | 0,0,8,13,23,32,62,66,91,150 |

**Takeaway — VLMPlan-32B is a genuine planner here, sitting between astar and the learned
methods, and the pilot badly mis-estimated it.** Three points, the first a correction.

1. **The 2-problem pilot was wrong, and this is why 10/stratum was the right call.** The
   pilot drew train problems 500000 (b3) and 750000 (b5), both in the hard tail: 0
   self-solves, FP 34 and a censored 200. From those two I told the summary VLMPlan
   "loses to astar on b3/b5, censored on b5." The stratified test sample overturns it:
   0 censored anywhere, VLMPlan self-solves the *median* problem in one proposal, and it
   **beats astar-dist overall** (13.18 vs 16.29). An earlier registry caveat asserting
   the pilot reading was corrected in the same commit. Two hard problems are not a row.

2. **It beats the naive planner order but only via b5, and loses to it everywhere else.**
   VLMPlan is worse than astar on b1/b2/b3 (0.70 vs 0.08, 1.30 vs 0.56, 6.20 vs 2.96):
   its off-pool proposals are refined for real and charged as attempts, so its
   charged-but-failed guesses cost it exactly where the pool order is already near-
   optimal. It wins on b5 (44.5 vs 61.6) only because astar's *default* order is
   pathological there (61.56). The overall win is a b5 artefact of a weak baseline, not
   broad superiority.

3. **The representation gap is the headline, and it is wide.** VLMPlan-32B (13.18) trails
   SPECTREv3 (1.69) and PIGINet (2.02) by ~7×. The zero-data corner is a real, competent
   point — 35/40 self-solved — that the trained abstract-first and low-level predictors
   both dominate. That is exactly the framing [`proposal.md`](../proposal.md) §0 wants:
   VLMPlan answers "did you try just asking a VLM?" on the record, as a corner of the
   data axis, not a defeated straw man.

**Next.** The row is n=40; the full 100 would tighten b3/b5 (their tails are what the mean
rides on) but cannot move the ~7× representation gap or the qualitative ordering. Left as
a deliberate stopping point unless the paper needs n=100 parity on this row.

---

<a id="2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed"></a>
## 2026-08-01 — SB2D v3 ablation arms: training is not reproducible from the seed alone

<!--strip-->
> **id** `2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed` ·
> **status** active · **tracks** evaluation, baselines, env-stickbutton2d
<!--/strip-->

**What.** Train the six v3 component arms on StickButton2D so §4 of the comparison
notebook — the coverage × record-tokens 2×2 plus the single-column split — has something
to render on the second environment. Same flags as DD2D's arms, `spectre_sweep.py --preset
sb2dabl`, one seed each (the project's 1-seed dev convention), cached via
`precompute_dd2d_cache.py --env-variant stickbutton2d_v1`.

The demotion pair was deliberately omitted: SB2D resolves to `EMPTY_SPEC`, so
`licenses_demotion` is always false and the two caches would be bit-identical. Vacuous
here, not overlooked.

**Result — the arms, seed 0, mean rollout FP on the 100-problem test split.**

| arm | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| deployed (`spectre3`) | **1.76** | 0.08 | 0.32 | 1.00 | 5.64 |
| cov+waste, no tokens | 1.77 | 0.08 | 0.36 | 1.20 | 5.44 |
| waste column only | 1.92 | 0.08 | 0.32 | 1.32 | 5.96 |
| coverage column only | 2.13 | 0.08 | 0.40 | 1.28 | 6.76 |
| no cov/waste, tokens | 2.53 | 0.08 | 0.48 | 1.60 | 7.96 |
| **cov+waste, tokens** (`abl_cov_rec`) | **2.78** | 0.08 | 0.24 | 2.92 | 7.88 |
| neither (no cols, no tokens) | 2.89 | 0.08 | 0.40 | 1.40 | 9.68 |

**Result — the finding, which is about the instrument and not the arms.** `deployed` and
`abl_cov_rec` are **the same flags at the same seeds**. They were trained twice by
accident — the deployed arm from the sweep, the ablation arm from the `sb2dabl` preset —
and they read **1.76 vs 2.78** at seed 0, a gap of **1.02 FP**. Over three seeds the pair
reads 1.69 ± 0.26 vs 2.00 ± 0.28.

Every ablation gap in the table above is smaller than 1.02.

**Takeaway — SB2D's §4 does not separate, and the table must be read against run-to-run
noise rather than against the seed sd.** Three things follow.

1. **No arm ordering in the SB2D 2×2 should be quoted.** The accidental duplicate is a
   free null-effect control: it measures what the pipeline reports for a contrast that is
   *known* to be zero, and it reports 1.02 FP. The largest real contrast here (`neither` −
   `deployed` = 1.13) barely clears its own noise floor.
2. **The seed sd understates the uncertainty.** ±0.26 across three seeds is the spread of
   *one* training run per seed; it does not contain the between-run variance at fixed seed,
   which is roughly four times larger. This is a sharper version of the standing rule that
   a load-bearing per-stratum margin is compared to the seed sd
   ([`decisions/06`](../decisions/06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd)):
   on this environment even the seed sd is the wrong yardstick.
3. **Training is not reproducible from the seed alone.** Not diagnosed further — likely
   CUDA nondeterminism in the tensorization/backward path, which the project has already
   seen at ~2e-4 on inference scores. What is established is the *magnitude of its
   consequence* on a low-FP environment: where DD2D's means sit near 6–17 FP, SB2D's sit
   near 2, so the same absolute jitter is an order of magnitude more of the signal.

The finding is recorded in `compare_envs.SB2D.caveats`, so it renders under §1 of the
notebook rather than living only here.

**Next.** The DD2D §4 numbers are *not* retroactively suspect — DD2D's contrasts run
1–5 FP against means near 15, and its own arms were never duplicated so no null control
exists there. Establishing whether the same jitter applies at DD2D's scale would need one
deliberate duplicate run, which is ~17 min and is the cheapest thing that would firm up
every 1-seed ablation the project has published.

---

<a id="2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity"></a>
## 2026-08-01 — SB2D comparison: v3 and PIGINet are indistinguishable; adaptivity is the only separation

<!--strip-->
> **id** `2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity` ·
> **status** active · **tracks** baselines, evaluation, method, env-stickbutton2d
<!--/strip-->

**What.** Stand PIGINet up on StickButton2D and reproduce the DD2D comparison notebook
there — the representation contrast (low-level predictor vs abstract-first re-ranker) on
the second environment. Three seeds each, BCE arm, AUPRC-selected, same 267/90 train/val
and same 100-problem test split as SPECTRE v3, same labels.

**Result — the comparison table.** Mean rollout FP on the test split (n = 100), uncensored;
`sd` is across the three seeds.

| method | ALL | sd | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|---|
| astar-dist (planner order) | 16.29 | — | 0.08 | 0.56 | 2.96 | 61.56 |
| PIGINet | 2.02 | 0.19 | 0.08 | 0.32 | **1.31** | 6.39 |
| SPECTREv3-static | 1.98 | 0.28 | 0.08 | 0.32 | 1.52 | 5.99 |
| SPECTREv3-adaptive | **1.69** | 0.26 | 0.08 | 0.24 | 1.13 | **5.29** |

Paired bootstrap over the 100 test problems (seed-averaged per problem, 10 000 resamples):

| comparison | Δ | 95% CI | separates |
|---|---|---|---|
| v3-adaptive − PIGINet | −0.337 | [−0.723, +0.053] | **no** |
| v3-static − PIGINet | −0.047 | [−0.437, +0.353] | **no** |
| v3-adaptive − v3-static | −0.290 | [−0.517, −0.073] | **yes** |
| PIGINet − astar-dist | −14.267 | [−21.383, −7.970] | **yes** |

**Takeaway — on StickButton2D the representation advantage does not reproduce; adaptivity
is the only thing that separates.** Two readings, and the second is the load-bearing one.

1. Both learned methods crush the planner order (−14.3 FP for PIGINet alone). The
   feasibility-prediction problem is real here and learning solves a lot of it.
2. **SPECTRE v3 and PIGINet are statistically indistinguishable**, in *both* deployment
   modes. The abstract-first representation buys nothing measurable over the low-level one
   on this environment (v3-static − PIGINet = −0.05, CI spanning zero). What *is*
   significant is the adaptive increment within SPECTRE: −0.29 FP, CI excluding zero.

That inverts DD2D's attribution. There the static representation carried ~73% of the margin
and adaptivity ~27% (`notebook/01` 2026-06-06). Here the static representation carries
**none** of it and adaptivity carries all of it. The pivot's framing — "abstract-first is
the leading candidate, adaptivity is a secondary composable increment" — survives DD2D and
does not survive this environment unchanged.

**Three caveats, all of which cut in PIGINet's favour and none of which rescue the claim.**

- **PIGINet's image channel is degenerate here by construction.** Every unpressed button is
  the same red disc, so CLIP separates only {button, stick, robot} — information the type
  literals already carry. PIGINet matches v3 *despite* getting nothing from pixels; its
  pose/shape channels are doing the work. An environment with informative perception would,
  if anything, favour it more.
- **`at-pose` literals are synthesised by our adapter**, not stored. SB2D's abstract initial
  state names no positions, so a low-level predictor would otherwise receive none. This is
  a deliberate construction to make PIGINet a fair comparator rather than a strawman; it is
  also the single largest discretionary choice in the port.
- **b5's train split is 17 episodes** for both methods. Its column is substantially a
  generalisation number. Both share the split, so neither is advantaged — but the b5 gap
  (5.29 vs 6.39) is the least trustworthy cell in the table.

**Sample size, stated rather than glossed.** v3-adaptive − PIGINet is −0.337 with CI
[−0.723, +0.053] — *nearly* separating. This is "indistinguishable at n = 100 and 3 seeds",
not "equal". A larger test split or more seeds could resolve it either way, and that is the
cheapest experiment that would sharpen this row.

**Takeaway/next.** The honest cross-environment statement today is: *the abstract
representation wins on DD2D and ties on StickButton2D, while the adaptive increment is
positive on both.* Before that goes in a paper: (1) finish the b3/b5 train splits so b5 is
not a 17-episode extrapolation; (2) decide whether the near-miss CI warrants more seeds;
(3) note that DD2D's own PIGINet row and this one now come from the same code, verified
unchanged at FP 17.0500.

---

<a id="2026-08-01-sb2d-collection-b1-b5-bracket-v3-1"></a>
## 2026-08-01 — SB2D collection, B1-B5 bracket, and v3 at 1.69 FP

<!--strip-->
> **id** `2026-08-01-sb2d-collection-b1-b5-bracket-v3-1` · **status** active ·
> **tracks** method, evaluation, baselines, env-stickbutton2d
<!--/strip-->

**What.** The `stickbutton2d_v1` collection, its B1–B5 baseline bracket, the coverage
re-ranking gate on the full test split, and v3 trained on it.

**Result — the collection, as collected.** Rejection is DD2D's `reason="unsolved"`
convention (drop a problem with no feasible skeleton, redraw).

| | train | val | test | mean pool | rejected (train/val/test) | CPU-h |
|---|---|---|---|---|---|---|
| b1 | 100 | 25 | 25 | 1.5 | 5 / 3 / 2 | 0.3 |
| b2 | 100 | 25 | 25 | 18.0 | 11 / 1 / 2 | 6.6 |
| b3 | 50 | 20 | 25 | 200 | 7 / 3 / 2 | 36.6 |
| b5 | 17 | 20 | 25 | 200 | — (job stopped at cutoff) | ~40 |

**Test is complete at 25 per stratum**; train and val for b3/b5 are short of the planned
100/25. Measured throughput was 18.6 (b3) and 11 (b5) keepers/h on 12 workers, so the
original targets were an 8 h and a 13.6 h job. Targets were re-budgeted per split at 00:31
with test held at full size, since test sizes the headline. Rejection rates are low
everywhere (≈10% at b2/b3), so the solvable-scene bias this introduces is small.

**Result — B1–B5 on the test split, uncensored.** Mean failed attempts before first
success:

| | B1 random | B2 default | B3 static-hist | B4 adaptive-hist | B5 oracle |
|---|---|---|---|---|---|
| ALL | 21.04 | 16.29 | **6.41** | 22.56 | 0.00 |
| b1 | 0.24 | 0.08 | 0.08 | 0.08 | 0.00 |
| b2 | 5.22 | 0.56 | 0.36 | 0.24 | 0.00 |
| b3 | 47.79 | **2.96** | 9.84 | 26.88 | 0.00 |
| b5 | 30.90 | 61.56 | **15.36** | 63.04 | 0.00 |

Two results here are worth more than the method comparison they were collected for.

**B4 is worse than random on this environment** (22.56 vs 21.04 overall; 63.04 vs 30.90 at
b5). The Naive-Bayes adaptive baseline is SPECTRE's headline comparison on RT2D and DD2D
because it is the strongest non-learned adaptive ranker there. It is *actively harmful*
here. **The strongest baseline on SB2D is B3, static-historical, at 6.41** — so the bar a
learned method has to clear is a *static* one, and the "adaptivity premium" framing that
motivated the original RT2D design does not transfer.

**At b5 the planner's own enumeration order is worse than shuffling** (B2 61.56 vs B1
30.90). Measuring where the feasible plans sit in the pool explains it: at b3 the first
success is at **1.5%** of the pool and all successes average **12.9%** — A* order is
genuinely informative. At b5 all successes average **49.9%** (i.e. uniform) while the
first arrives only at **30.9%**. So the order is not merely uninformative at b5, it is
anti-correlated, and a random permutation finds a feasible plan sooner.

**Result — Gate A (does coverage rank?): PASS at b5, marginal at b3.** Non-learned
re-ranking of the remaining pool as failures accrue, 100 test episodes:

| | n | static | coverage+waste | coverage only | waste only | oracle |
|---|---|---|---|---|---|---|
| b1 | 25 | 0.08 | 0.08 | 0.08 | 0.08 | 0.00 |
| b2 | 25 | 0.56 | 0.56 | 0.56 | 0.56 | 0.00 |
| b3 | 25 | 2.96 | 4.44 | **2.88** | 5.52 | 0.00 |
| b5 | 25 | 61.56 | **25.56** | **25.56** | 61.56 | 0.00 |

Coverage alone cuts b5 by **58%** (61.56 → 25.56), better on 20/25 problems paired, worse
on 1. At b3 it is a wash on the mean (2.96 → 2.88) while winning 12/25 and losing 1 — the
mean is dragged by a few large regressions.

**`waste` is not neutral on SB2D, it is inert or harmful.** At b5 `waste_only` reproduces
`static` to the last digit (61.56 both) — completely inert, because every b5 plan has the
same length and therefore the same superfluous-step set. At b3 it *hurts*: 5.52 alone
against static's 2.96, and adding it to coverage as a tie-break degrades coverage from 2.88
to 4.44. This sharpens the registered cross-env dominance flip from "waste carries less
here" to "waste carries nothing here, and can carry negative".

Note the ceiling this sets up: the plain coverage re-ranker at b5 (25.56) is **worse than
B3 static-historical (15.36)**. Coverage is a strong *adaptive* signal and a weak static
one; a method that beats B3 has to combine both, which is what v3 is.

**Takeaway/next.** `waste` earning its place is now an open question on this environment
rather than an assumed yes, so v3 was trained in two arms — the deployed `coverage+waste`
and `--coverage-mode coverage`. Reported below.

**Result — v3 on SB2D, 3 seeds, uncensored test split.**

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| B1 random | 21.04 | 0.24 | 5.22 | 47.79 | 30.90 |
| B2 default (A*) | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| B3 static-historical | 6.41 | 0.08 | 0.36 | 9.84 | 15.36 |
| B4 adaptive-historical | 22.56 | 0.08 | 0.24 | 26.88 | 63.04 |
| coverage re-rank (not learned) | — | 0.08 | 0.56 | 2.88 | 25.56 |
| **SPECTRE v3 (deployed)** | **1.69 ± 0.26** | 0.08 ± 0.00 | 0.24 ± 0.08 | **1.13 ± 0.12** | **5.29 ± 1.04** |
| v3, waste column zeroed | 2.04 ± 0.52 | 0.08 ± 0.00 | 0.29 ± 0.12 | 1.85 ± 1.10 | 5.95 ± 1.54 |
| B5 oracle | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

v3 beats the strongest baseline by **3.8x** overall (6.41 → 1.69) and by **2.9x** at b5
(15.36 → 5.29). It is also the only method that is good on *both* hard strata: B2 is the
best baseline at b3 and the worst at b5; B3 is the reverse.

**The waste ablation reverses Gate A's verdict, and that is the interesting part.** As a
hand-coded tie-break waste is harmful at b3 (4.44 against 2.88 for coverage alone). As a
*learned* feature it is worth **+0.36 FP, CI [+0.08, +0.67]**, concentrated at b3 (1.13 vs
1.85). The same column, on the same episodes, helps a model and hurts a rule. The lesson
generalises past this feature: a non-learned probe measures whether a signal is
*monotonically* usable in the direction we guessed, not whether it carries information —
so a failed re-ranking probe is not grounds for dropping a column, and the pass/fail
framing of Gate A was too strong on that point.

**Caveat on b5.** Only **17** b5 episodes are in the training split (the collection was
cut at a wall-clock budget), so b5's 5.29 is largely a *generalisation* result — a model
trained on b1/b2/b3 pools transferring to 5-button pools it barely saw. That is a stronger
claim than the number alone suggests, and also a less stable one; it should be re-measured
on a full b5 train split before it is quoted as a like-for-like stratum result.

**Takeaway/next.** Three things worth doing before this becomes a paper row: (1) finish
the b3/b5 train splits so b5 is not a 17-episode extrapolation; (2) re-run the waste
ablation there, since its value showed up at b3 where data is plentiful; (3) B4 being worse
than random means the "adaptivity premium over B4" framing that motivated RT2D needs
restating for a cross-environment claim — on SB2D the bar is B3, a static ranker.

---

<a id="2026-08-01-stickbutton2d-stood-up-pool-shape-evidence"></a>
## 2026-08-01 — StickButton2D stood up: pool shape, evidence classes, and the two gates

<!--strip-->
> **id** `2026-08-01-stickbutton2d-stood-up-pool-shape-evidence` · **status** active ·
> **tracks** method, data, env-stickbutton2d, evaluation
<!--/strip-->

**What.** Stand up StickButton2D as SPECTRE's second environment end to end: pool filter,
`scene_geometry`, class-2 evidence, pooled `stickbutton2d_v1` variant, and the
400/100/100 collection. Two gates before trusting anything: does the pipeline actually
produce a checkpoint (B), and does coverage still rank on the filtered pools (A).

**Result — pool shape.** Acyclic fraction of a 200-candidate draw, 6 seeds per variant:

| | b1 | b2 | b3 | b5 |
|---|---|---|---|---|
| acyclic / 200 raw | 1–2 | 6–34 | 73–101 | 193–200 |
| acyclic, raw budget 5000 | 1–2 | 6–34 | 200 (≈640 raw) | 200 (200 raw) |
| deployed pool size | ≈2 | 6–34 | 200 | 200 |

Raising the raw budget from 20000 to 5000 changed no pool and cut b1's pool-draw time from
20–61 s to 1–4 s; the 20000 draws were spent enumerating ever-longer padded plans.

**Result — b5's feature degeneracy is structural.** Every b5 acyclic plan has length 6
(1160/1188 skeletons; the rest 7), i.e. 5 presses plus one stick pickup. So
`manipulated = args \ goal_objects = {robot, stick}` for *every* candidate, which pins
`jaccard` and `dead` constant across the pool and collapses the within-length PL loss to a
single bucket. At b5 the only features that can discriminate are `coverage`/`waste` and
the operator/argument token structure. This is the sharpest possible statement of why the
unified definitions were a prerequisite rather than an improvement: on the deployed
`S(c)` formula there would have been *no* usable candidate feature at b5 at all.

**Result — the evidence features do partition a 200-deep pool.** Tensorizing collected
b3/b5 test episodes at `|F| = 3` through the real `build_v3_example`:

| | pool | successes | coverage > 0 | distinct coverage | distinct jaccard |
|---|---|---|---|---|---|
| b3 | 200 | 3–15 | 65–98 | 2 | 1–2 |
| b5 | 200 | 2–9 | 40–69 | 2–4 | **1** |

Three things worth keeping. **Positives are genuinely sparse** — 1–7% of candidates
refine, which is the regime ranking is for. **`jaccard` is constant across the b5 pool**,
confirming the degeneracy above from the data rather than from the argument. And
**coverage is coarse**: with `|K|` of 1–3 it takes 2–4 distinct values, i.e. it is close to
a binary "does this candidate discharge the culprit" rather than a graded score.
Tensorization is not a bottleneck (<0.05 s per 200-candidate episode), so the selector's
per-epoch cost is forward passes, not feature construction.

One episode in six had **all three records blameless** (pure means-failure, no collateral),
giving coverage 0 across the whole pool. That is the case the blameless-record decision
exists to make harmless, and it is not rare.

**Result — Gate B (pipeline produces a checkpoint): PASS.** On 80 train / 20 val b1+b2
episodes, `train_v3` reports `n_train=43 n_val=13` and writes `best.pt`, val_fp improving
1.77 → 1.23 over two epochs. The gap between 80 and 43 is `_trainable`: about half of b1's
episodes have pool size 1. Tensorizing collected episodes through the real
`build_v3_example` gives non-zero, *varying* coverage — 11/15 candidates covered on one
b2 episode, max 1.0 — and record tokens carrying object tags via `dev_blame`. Contexts
whose records are pure means-failure correctly produce coverage 0.

**Result — DD2D is untouched.** `checkpoints_v3_unified` re-scores at **5.78 ± 0.10**
(s0 0.00, s1 3.44 ± 1.36, s2 10.49 ± 0.77, s3 9.19 ± 0.76), identical to the pre-change
figure per stratum as well as overall, after edits to `dataset_v3`, `canonicalize`,
`unified_evidence` and `failure_record`.

**Result — instrumentation is observation-only.** Same-seed differential over b2 and b3,
3 problems × 8 candidates each: `RecordingSampler` and upstream's
`ParameterizedControllerTrajectorySampler` return identical labels, and the recorder
demonstrably captures records (guard against a vacuous pass).

**Takeaway/next.** The porting contract in `porting_guide.md` was incomplete in two ways
that only a real transfer would have found: it assumed a refiner can *name* what it failed
on, and it listed `scene_geometry` as required without saying that its absence produces a
successful-looking training run with no checkpoint. Both are now written up (§2b, §4).

---

