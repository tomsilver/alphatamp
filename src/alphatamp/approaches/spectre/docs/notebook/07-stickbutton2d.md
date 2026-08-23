# SPECTRE Notebook — StickButton2D as a second environment

6 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

---
<a id="2026-08-22-rung-1-result-step-join-over-record-tokens"></a>
## 2026-08-22 — Rung-1 result — step-join over record tokens is the lever, content enrichment inert (C2 confirmed, fixable)

<!--strip-->
> **id** `2026-08-22-rung-1-result-step-join-over-record-tokens` · **status** active ·
> **tracks** method, evaluation, env-dd2d
<!--/strip-->

**What.** First learned-pathway result (docs/failed_records_fix.md F-A/F-B2). Built the rung-1
evidence-step pathway (per failed attempt: a failed-step token + each culprit's establishing step,
embedded by the **shared** CandidateEncoder so a failed `place_short(b)` ≡ the current candidate's
`place_short(b)`) and the **StepJoin** (candidate per-step tokens cross-attend over the evidence
memory *before* the PMA pool — the C2 fix for the pooled evidence query P-0 found). All additive /
zero-init / flag-gated (off byte-identical; 505 fast tests + `test_rung1_steps.py`). Trained a 4-arm
sweep on dd2d_v4, **1 seed (dev)**, on the `abl_only_records` backbone (jaccard + pointset + atoms +
ma5, no compiled coverage/waste/repeat), isolating the two increments: `--record-mode steps`
(content enrichment) and `--step-join` (architecture). ADR:
[`decisions 2026-08-22`](../decisions/07-stickbutton2d.md#2026-08-22-step-join-lever-content-enrichment-inert). Anchors
(same test set, seed 0): floor `abl_floor` **15.77**, ceiling `abl_all` **6.68** (gap 9.09).

**Result.** Test n=100, uncensored deployed FP; paired bootstrap over problems vs `fr_summary`
(the rung-0 tokens-only baseline, retrained as the matched control), * = 95% CI excludes 0.
gap-closure = (15.77 − FP) / 9.09.

| arm | increment | ALL | s1 | s2 | s3 | Δ vs summary | gap-closure |
|---|---|---|---|---|---|---|---|
| `fr_summary` | rung-0 (records, pooled query) | 13.64 | 11.76 | 18.56 | 24.24 | — | 23% |
| `fr_steps` | + enriched evidence steps | 13.60 | 13.72 | 17.16 | 23.52 | −0.04 [−2.30, +2.60] | 24% |
| **`fr_join`** | + step-join (over summary tokens) | **11.84** | 9.40 | 16.52 | 21.44 | **−1.80 [−3.52, −0.16]\*** | **43%** |
| `fr_steps_join` | + enriched steps **and** join | 14.37 | 6.72 | 19.40 | 31.36 | +0.73 [−2.28, +3.48] | 15% |

- `fr_summary` 13.64 ≈ the cached `abl_only_records` 13.70 → the retrained control reproduces (GPU noise).
- **Content enrichment is inert.** `fr_steps` = −0.04 (CI includes 0): richer tokens under the pooled
  query change nothing — the model cannot reach the extra content. Direct confirmation of **C2** (and
  consistent with P-2: the content was already recoverable, so *adding* it does nothing).
- **The step-join is the lever.** `fr_join` = −1.80 (CI excludes 0), the only significant arm. It
  nearly **doubles** the raw-evidence gap-closure (23% → 43%) with **no compiled coverage/waste/repeat**
  — a pre-pooling per-step candidate×evidence attention extracts ~43% of what the hand-compiled
  programs capture, purely from raw record tokens. Improves every non-trivial stratum (s1 11.76→9.40,
  s2 18.56→16.52, s3 24.24→21.44).
- **Enriched steps + join is worse than join alone** (`fr_steps_join` +0.73, ns; s3 blows to 31.36
  while s1 drops to 6.72). Most likely the larger evidence-step memory (failed + establishing steps)
  dilutes the step-join's attention (the rung-2 SNR risk, one rung early), or 1-seed optimization noise.

**Takeaway.** The failure-record inertness is an **architecture** problem, not a content one: the
information is present (P-2) and enriching it is inert (`fr_steps`), while a one-line reordering —
letting candidate *steps* attend over the record tokens before pooling — is what unlocks it
(`fr_join`, −1.80, 23%→43% gap-closure). This clears the doc's **≥25% proceed gate** but not the **50%
headline gate**, on the summary-token step-join alone. So the deployed method's win need not be
attributed to the compiled scalars as hand-engineering: ~half of it is recoverable by a generic
attention join over raw evidence. ⚠️ **1 seed** (fr_join's −1.80 rests on a paired-over-problems CI,
not cross-seed). **Decision (2026-08-22): C1 (content enrichment) is CUT** — inert alone, harmful
combined (dilution), and its one unique value is `regroup`, which is off in practice; the machinery
stays flag-gated off per the build-then-disable convention but is not pursued (the `fr_steps`/
`fr_steps_join` arms are dropped from the sweep). The deployed direction is the **StepJoin over the
summary tokens** alone. Next: (1) 3-seed confirmation of `fr_join`; (2) P-4 teachability — how much of
the remaining floor→ceiling gap is C3 (learnability) vs C2 (needs a sharper join); (3) the combined
`step-join + scalars-on` rung — does the join *add* to the deployed ceiling, or is it substitutive?

---

<a id="2026-08-22-p-2-sufficiency-audit-rung-0-compiled"></a>
## 2026-08-22 — P-2 sufficiency audit at rung 0 — compiled scalars recoverable from tokens (C1 largely ruled out)

<!--strip-->
> **id** `2026-08-22-p-2-sufficiency-audit-rung-0-compiled` · **status** active ·
> **tracks** method, evaluation, env-dd2d, env-restock3d
<!--/strip-->

**What.** `docs/failed_records_fix.md` P-2: is each compiled `cand_overlap` scalar (coverage,
waste, repeat, regroup) a *function of the record tokens* a model reads, or does its computation
consume inputs the tokens dropped (hypothesis C1, content gap)? Built
`experiments/spectre/failed_records_sufficiency.py` (read-only): hold one episode fixed (scene,
init/goal atoms, every candidate skeleton constant), vary only the failure context `F`; for a fixed
candidate `c`, its token input is `token_bag(F)` and its scalar is a function of `(records(F), c,
scene)`. Enumerate every singleton context `{i}` (`i` a failed candidate), group by the aggregated
`token_bag` (schema + role tag-sets + rounded scalars — exactly what a `RecordEncoder` token
carries), and within a same-token group flag any candidate whose scalar disagrees = **a collision =
proof of insufficiency**. Report per-scalar collisions alongside the value distribution (nonzero %,
distinct values) so a 0-collision verdict is read against whether the scalar even varies. 20 train
episodes each, dd2d_v4 + restock3d_v3.

**Result.** **No collisions on any scalar that varies** → *consistent-with-sufficient* everywhere;
the doc's registered "regroup insufficient at rung 0" was **not reproduced empirically** — but see
the ⚠️ below: the audit is badly underpowered for `regroup`, and the doc's *structural* claim that
`regroup` is not computable from the current stream is in fact **correct** (a real content gap the
sparse empirical test missed).

| variant | scalar | checked | collisions | nonzero% | distinct | verdict |
|---|---|---|---|---|---|---|
| dd2d_v4 | coverage | 25400 | 0 | 24.6% | 5 | sufficient |
| dd2d_v4 | waste | 25400 | 0 | 73.8% | 5 | sufficient |
| dd2d_v4 | repeat | 25400 | 0 | 5.4% | 2 | sufficient |
| dd2d_v4 | regroup | 25400 | 0 | 0.0% | 1 | CONSTANT (vacuous — DD2D declares no `grouping_certificate`) |
| restock3d_v3 | coverage | 2160 | 0 | 14.0% | 5 | sufficient |
| restock3d_v3 | waste | 2160 | 0 | 0.0% | 1 | CONSTANT (F3 blameless → empty culprit pool → waste abstains) |
| restock3d_v3 | repeat | 2160 | 0 | 45.0% | 2 | sufficient |
| restock3d_v3 | **regroup** | 2160 | **0** | 0.84% | 2 | **sufficient** (varies, no collision) |

**Takeaway.** **C1 (content gap) is ruled out for the FP-relevant scalars (coverage / waste /
repeat)** — the information each needs *is* recoverable from the record tokens (no collision, and they
carry most of the signal). It is **not** ruled out for `regroup`: `regroup`'s "seating chart" is
`{failed step} ∪ {establishing step of each culprit}`, and the establishing step is `place_?(culprit)`
**with its schema** (`place_tall` vs `place_short`) taken from the *failed* candidate's plan — the
culprit tag is tokenized but that establishing-step **schema is not**, so the doc's "provably not
computable" is structurally **correct**. The audit missed it because `regroup` is vanishingly sparse
(≡0 on DD2D, which declares no `grouping_certificate`; 0.84% nonzero on v3), so "0 collisions in 20
episodes" is underpowered, not a refutation. But `regroup` is a ~0–1% feature on these envs, so its
genuine content gap does not move FP. Combined with P-0 (the evidence query is the *pooled* candidate,
so step-level joins are not representable), the diagnosis for the scalars that matter is **C2
(architecture — the join is present in the inputs but attention can't compute it)** and **C3
(learnability from ~500 episodes)**. Consequence for Phase 2: rung-1's value is twofold — (a) for
coverage/waste/repeat, re-representing the *present* content in an attention-joinable form; (b) for
`regroup`, it is the one thing that *closes* the real content gap (the establishing-step schema), so it
is kept flag-gated for domains where grouping matters, even though it is FP-irrelevant here. The
minimal step-join is what makes the join representable at all; P-4 then separates C2 from C3.
⚠️ "consistent-with-sufficient" is absence-of-collision over a 20-episode sample, not a proof;
`regroup`/`repeat` are sparse (0.8–45% nonzero) so the hunt is far better-powered on coverage/waste.
Next: build rung-1 + step-join, re-run this audit, then P-4.

---

---

<a id="2026-08-22-failure-record-token-holdout-inert-all-collections"></a>
## 2026-08-22 — Failure-record token holdout inert on all collections — P-1 corrected baseline is a no-op (C4a ruled out)

<!--strip-->
> **id** `2026-08-22-failure-record-token-holdout-inert-all-collections` · **status**
> active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d, env-restock3d
<!--/strip-->

**What.** First step of the learned-pathway workstream (`docs/failed_records_fix.md`, P-1): the
failure-record **tokens** are near-inert, and hypothesis **C4a** blamed the *certificate-record
token holdout* — the filter in `dataset.build_record_arrays` that drops every record satisfying
`proof_tier(schema) ∧ proves_failure()` from the token stream (`proof_tier = monotone ∧ local ∧
exact`). The doc predicted a *large* held-out-token delta on DD2D (its C4a box cited a "~92 %
blameless-provable" census), so a tokens-only arm would have been scored against records it never
saw. I put the holdout behind a flag — `record_holdout` (default `True` = current behavior;
`--no-record-holdout`), threaded through `build_example` / `SpectreDataset` / `deployed_val_fp` /
`deployed_rollout_traced` and **round-tripped in `inference.load_checkpoint`** so a model deploys
under exactly what it trained with — and wrote a read-only census
(`tests/approaches/spectre/test_record_holdout.py`): count tokens with the filter on vs off per
env-variant, plus a positive control (a *synthetic* `DomainSpec` declaring `place-buffer`
proof-tier). ADR:
[`decisions 2026-08-22`](../decisions/07-stickbutton2d.md#2026-08-22-record-holdout-flag-p1-holdout-inert).

**Result.** **The holdout is empirically inert on every current collection** — token delta (off −
on) is **0** on dd2d_v4, stickbutton2d_v1 and restock3d_v3. The positive control drops 600/692
tokens (the flag works; the *data* has no qualifying records). Direct census of the record
population (40 dd2d_v4 train episodes, 105 821 records): every record is `pick` or `place-buffer`,
**zero `retrieve`** — and `proof_tier` is empty (`pick`/`place-buffer` are not proof-tier;
`retrieve` **is** the only proof-tier DD2D schema but **never appears as a failing step**). dd2d_v3
is identical (no `retrieve` records). So the doc's C4a premise was wrong: it conflated `proof_tier`
(the holdout predicate) with the `step_certificate` / `blame==∅` ~92 % census (which drives
`repeat`, not the holdout). SB2D / restock3d_v3 declare only `step_certificate`, so `proof_tier()`
is False and the holdout was never active there either. Rung-0 anchors (dd2d_v4, seed 0, adaptive,
from the compare caches): floor `abl_floor` **15.77** → tokens-only `abl_only_records` **13.70** →
scalars-on ceiling `abl_all` **6.68** → deployed 6.04. Tokens-only closes **(15.77−13.70)/(15.77−6.68)
= ~23 %** of the floor→ceiling gap (the −2.07 already in the 2026-08-22 ablation entry).

**Takeaway.** **C4a is ruled out** — the tokens-only −2 is *not* a records-withheld artifact, so the
"corrected P-1 baseline" is a **no-op on dd2d_v4** (the existing `abl_only_records` arm already *is*
it) and the retrain P-1 registered is unnecessary. The token inertness is real and the diagnosis
sharpens to **C1 (content gap)** + **C2 (pooled evidence query, no step-level join)** — exactly what
Phase 2's rung-1 enrichment + minimal step-join target. Records are not *fully* inert (they capture
~2 FP / 23 % standalone), so the bar the enrichment must clear is the **50 %** headline gate, not
zero. The `record_holdout` flag is kept (correct, checkpoint-safe, `strict=True`) as the honest
control and CI guard; it becomes load-bearing only on a future collection whose failing steps *are*
proof-tier (e.g. DD2D `retrieve` failures) — the census test asserts `delta==0` and will flag that.
⚠️ 1 seed (dev); deployed §1 DD2D rows are pre-point-set-upgrade. Next: P-2 sufficiency audit at
rung 0, then rung-1 enrichment.

---

<a id="2026-08-22-single-feature-isolation-ablation-dd2d-sb2d-restock3d"></a>
## 2026-08-22 — Single-feature isolation ablation (DD2D/SB2D/restock3d_v3) + repeat transfer

<!--strip-->
> **id** `2026-08-22-single-feature-isolation-ablation-dd2d-sb2d-restock3d` ·
> **status** active · **tracks** method, evaluation, env-dd2d, env-stickbutton2d,
> env-restock3d
<!--/strip-->

**What.** To attribute SPECTRE's adaptive advantage per environment, ran a single-feature
isolation ablation: 6 arms — `floor` (jaccard-backbone only), `+coverage`, `+waste`, `+repeat`,
`+records`, `all` — × 3 envs (dd2d_v4, stickbutton2d_v1 → shown on sb2d_kinder, restock3d_v3) ×
**1 seed**, each **trained from scratch** with exactly one failure-conditioned feature added on
the shared jaccard backbone (`spectre_sweep.py --preset ablation_{dd2d,sb2d,restock}`), cached
(`precompute_dd2d_cache.py … --seeds 0`), tabled by `ablation_report.py` / `compare_methods.py`
§4.3. `repeat` was **activated on DD2D/SB2D retroactively** (no re-rollout — it reads stored
`FailureRecord` fields) via `domain.py` `step_certificate` declarations (DD2D `place-buffer`,
SB2D the 4 button-press schemas), chosen by a firing/leakage census
(`ablation_repeat_census.py`). ADR:
[`decisions 2026-08-22`](../decisions/07-stickbutton2d.md#2026-08-22-adaptive-feature-isolation-ablation-repeat-activated-dd2d).

**Result.** Δ vs `floor` (mean rollout FP), paired bootstrap over the 100 test problems, seed 0,
* = 95 % CI excludes 0:

| arm | DD2D | SB2D | restock3d_v3 |
|---|---|---|---|
| `floor` (ALL FP) | 15.77 | 2.38 | 12.60 |
| `+coverage` | **−6.87\*** | +0.00 | −0.21 |
| `+waste` | **−9.13\*** | −0.25 | +1.25 |
| `+repeat` | −4.42\* | **−0.79\*** | **−9.59\*** |
| `+records` | −2.07 | −0.30 | +0.49 |
| `all` | −9.09\* | **−1.07\*** | −9.33\* |
| `all` (ALL FP) | 6.68 | 1.31 | 3.27 |
| deployed adaptive (ALL FP) | 6.04 | 1.59 | 3.13 |

Firing/leakage census (train+test): **DD2D** `place-buffer` fires on 34 % of candidates but
**leaks 44.6 %** of feasible candidates (`retrieve`/`pick` are vacuous — never `blame==∅`);
**SB2D** press schemas fire 55 % / **10.9 %** leakage; restock3d's F3 ~0 % (genuinely sound). `all`
reproduces the deployed adaptive within noise on every env (esp. restock 3.27 vs 3.13) — the fresh
1-seed arms are a faithful current-architecture rebuild.

**Takeaway.** The mechanism is **different per environment**: DD2D is carried by **coverage +
waste** (the necessity/packing scalars), while SB2D and restock3d_v3 are carried by **`repeat`**;
**`records` is inert everywhere** as a standalone marginal over the jaccard backbone. The headline
for the transfer question: **retroactively-activated `repeat` HELPS on both DD2D (−4.42) and SB2D
(−0.79) despite high leakage** — the leakage census bounds a *hard veto*, not a *learned feature*,
so an "unsound certificate" can still be a useful learned column. ⚠️ 1 seed (± deferred);
restock3d_v3 is analytic-synthetic (magnitudes upper-bound); DD2D/SB2D §1 deployed rows are stale
(pre-point-set-upgrade). Next: 2 more seeds per arm, real-refiner audit on restock.

---

<a id="2026-08-21-restock3d-v3-adaptivity-revived-repeat-f3"></a>
## 2026-08-21 — restock3d_v3 adaptivity revived: repeat F3 certificate captures 97pct of the oracle ceiling

<!--strip-->
> **id** `2026-08-21-restock3d-v3-adaptivity-revived-repeat-f3` · **status** active ·
> **tracks** method, evaluation, env-restock3d
<!--/strip-->

**⚠️ Analytic-synthetic dataset throughout** (labels = `feasibility_v3.classify_skeleton`, no MP;
[2026-08-20](#2026-08-20-restock3d-v3-synthetic-dataset-collection-spectre)). Read magnitudes as an
upper bound; the ordering (which feature revives adaptivity) is the finding.

**What.** v3's adaptive increment was **inert** (SPECTRE-adaptive ≈ static; `notebook` 2026-08-20).
Probed why, fixed it, retrained. Five probes (P0–P4, no training) + a coverage bug + one new feature.

**Probes.** P0 census: **F3 (height) = 75% of failures and is blameless** (`blame(F3)=∅`), F2
(crowding) 25%, F4 (reach-over) ~0%. P1 variance audit: coverage/waste **identically 0** in every
context — traced to a bug: `canonicalize._remap_refiner_metadata` coerced the F2/F4 records'
`dev_added=None`→`[]`, re-typing them from class-1 (culprits) to class-2-with-empty-deviation, so
`blame()` read the empty collateral deviation and dropped the culprits → `K` empty → coverage 0
(v3-only; DD2D omits the key, SB2D stores a real list). **P2 oracle re-ranker ceiling** (replay static
order + certificate pruning, 0 soundness violations): FP_static **11.05 → 2.81** (a **75% headroom**).
**P2b decomposition:** F3-only pruning (`repeat`) captures **74%** alone at every stratum; F2-only
(`regroup`) **1%**; F2-as-exact-step kills 263 real successes (proving the certificate needs a
`blame==∅` gate). Blame-structure census: "provable ∧ culprit-free" is *not* env-safe (DD2D's 92%
blameless-provable records are means-failures), so the env-agnostic scope is `proof_tier`-class
`step_certificate ∧ provable ∧ blame==∅`.

**Result.** (test n=100, 3 seeds, adaptive FP; static ≈ 12 for every arm — the win is purely
adaptive, leakage-clean.)

| arm | adaptive | adaptive − static (paired) |
|---|---|---|
| evidence-knockout (coverage≡0, the bug) | 11.11 ± 0.98 | +0.06 [−0.11, +0.25] inert |
| fix-only (coverage revived) | 12.18 ± 0.33 | −0.09 [−0.32, +0.12] **still inert** |
| **+repeat (F3 exact-step certificate)** | **3.13 ± 0.09** | **−8.89 [−11.10, −6.80]** ✔ |
| +repeat+regroup | 3.19 ± 0.21 | −9.11 [−11.26, −7.01] ✔ |

- **The coverage bug fix alone recovers nothing** (−0.09, inert; +1.1 FP *worse* vs the knockout,
  CI-clean) — exactly the pre-registration (`docs/adaptivity_fix_only_prereg.md`): coverage speaks
  the F2/*ordering* channel, worth ~1%; the 74% headroom is F3, which is blameless so coverage can
  never see it.
- **`repeat` revives adaptivity decisively**: +repeat − fix-only = **−9.06 [−11.28, −6.95]**, and it
  captures **~97% of the P2 oracle ceiling** (per-stratum 95–107%). Biggest at the crowded strata
  (n=8 17.3→3.0, n=9 26→7.1).
- **`regroup` adds nothing** (+0.06 over +repeat) — the ~1% P2 prediction. **Deprecated, off by
  default, to be removed.**
- **Cross-env pre-check (§5.2)** caught ungated `regroup` firing 42% *wrong-polarity* on DD2D (its
  culprits are blockers you *want* to stage); gating it with `grouping_certificate` → inert on
  DD2D/SB2D. `repeat` is inert there too (no `step_certificate` schema) — graceful degradation.
- **Comparison impact:** old SPECTRE 11.11 ≈ LAZY 11.79 ≪ PIGINet 38.11; **deployed SPECTRE+repeat is
  now 3.13**, decisively ahead of every comparator on v3.

**Takeaway-next.** Deployed the `--repeat-feats` arm (`checkpoints_spectre_atoms_repeat`,
`compare_cache` repointed). The adaptivity that read "inert" was an evidence-*language* mismatch, not
an adaptivity ceiling: v3's decision is grouping/assignment, and the load-bearing signal is the
blameless F3 certificate a `dead`-style veto surfaces, not the ordering coverage/waste speak. Open:
a real-refiner audit slice to price the synthetic magnitude; whether `repeat` helps DD2D's proof-tier
`retrieve` (expected small); removing `regroup`. ADR:
[`decisions/07` 2026-08-21](../decisions/07-stickbutton2d.md#2026-08-21-restock3d-v3-adaptivity-revived-coverage-canonicalize).

---

<a id="2026-08-20-restock3d-v3-synthetic-dataset-collection-spectre"></a>
## 2026-08-20 — restock3d_v3 synthetic dataset: collection + SPECTRE/PIGINet/LAZY comparison

<!--strip-->
> **id** `2026-08-20-restock3d-v3-synthetic-dataset-collection-spectre` · **status**
> active · **tracks** method, env-restock3d, data, evaluation, baselines
<!--/strip-->

**What.** Collected the first **restock3d_v3 dataset** and ran the SPECTRE/PIGINet/LAZY comparison
on it — **fully synthetic**: pools from the geometry prior, labels from the analytic refiner
(`feasibility_v3.classify_skeleton`, no motion planning), per-candidate wall-clock synthesized (fail
= r_cap; success = U[0.6,0.8]·r_cap). 4 strata n=6/7/8/9 at 100/25/25 = **400/100/100**, K_max
40/60/150/200, r_cap 50/70/90/110 s. Then vocab → SPECTRE (deployed `--scene-3d --atom-mode profiles`
recipe, 3 seeds) + PIGINet (3 seeds) + LAZY (3 seeds) → `compare_methods.py` (new `RESTOCK3D_V3`
EnvSpec). ADR: [`decisions/07` 2026-08-20](../decisions/07-stickbutton2d.md#2026-08-20-restock3d-v3-synthetic-dataset-analytic-refiner-collection).

**Collection stats.** Yield (kept iff ≥1 analytically-feasible candidate in the pool) is high on the
light strata and drops with n as the feasible-split set tightens; a per-stratum sequential collector
with reject-resample top-up hit exactly 100/25/25. The geometry-prior first-success index tracks the
earlier plan-attempts finding (n=9 mean ~64 within the K=200 pool). **Failure taxonomy is
F3-dominant:** on stratum 0, `place_short` height-F3 (culprit-free) ≈ 1114 vs F2 crowding-culprit
(place_tall/short residents) ≈ 197, and reach-over F4 is rare (~1/367 episodes — the geometry prior
mostly avoids bad pick orders). So v3 difficulty is **capacity/height/selection**, not reach-over.

**coverage/waste verified live + correct on the analytic path** (not assumed): the culprit pool is
non-empty (`obj_goal*` are `actionable`, not `universal`), coverage varies 0→1 with the right
polarity (a candidate that fails to discharge a culprit before re-entering the situation scores 0),
and abstains at |F|=0. The analytic culprits match the real refiner by construction (shared
`_blocks_reach` for reach-over, mirrored F2 residents, culprit-free F3). *Caveat:* because F3
dominates (feeds the record-token channel, not coverage) and F4 is rare, coverage/waste are correct
but carry less signal than in v2 (whose difficulty was reach-over-driven).

**Bug found + fixed — restock3D PIGINet crops were silently empty (v2 AND v3).** `render.object_crops`
crashed on the mobile robot (no `pose_x`), and the PIGINet adapter swallowed the exception → an
all-zero image channel, i.e. PIGINet's designed height-via-image signal was dead. `object_crops` now
skips un-poseable objects; v3 PIGINet sees real per-block oblique silhouettes. v2's *published*
numbers stand (cached, not rebuilt) but were on the dead channel.

**Result (3 seeds, ± across-seed, test n=100 = 25/stratum, uncensored at the pool cap).**

| method | ALL | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|
| astar-dist | 38.41 | 5.48 | 14.72 | 49.64 | 83.80 |
| **PIGINet** (low-level) | **38.11 ± 1.01** | 6.05 | 16.15 | 56.04 | 74.20 |
| SPECTRE-adaptive | **11.11 ± 0.98** | 1.81 | 3.53 | 13.05 | 26.04 |
| SPECTRE-static | 11.05 ± 0.88 | 1.77 | 3.43 | 12.69 | 26.29 |
| LAZY-adaptive | 11.79 ± 0.08 | 2.64 | 6.25 | 18.15 | 20.11 |

Paired bootstrap (seed-averaged per problem, 95% CI): **SPECTRE−PIGINet −27.00 [−32.97, −21.41]** —
CI excludes 0 and the margin **grows with crowding** (s0 −4.24, s1 −12.61, s2 −42.99, s3 −48.16, each
CI excludes 0); LAZY−PIGINet −26.32 [−33.09, −20.10]; SPECTRE−LAZY −0.68 [−3.25, +1.84] (tied — SPECTRE
edges s1/s2, LAZY edges s3 +5.93); SPECTRE-adaptive−static +0.06 [−0.11, +0.25] (inert).

**The §0 representation crossover appears decisively on v3.** The low-level image predictor **PIGINet
≈ the naive planner order** (38.11 ≈ astar 38.41), while both abstract rankers beat them **~3.4×**
(SPECTRE 11.11, LAZY 11.79). This is far stronger than v2 (where PIGINet 1.96 ≈ SPECTRE 1.44), because
v3's difficulty is **capacity/height/selection** — relational structure the abstraction (+ 3D point
cloud + atoms) encodes directly but oblique silhouettes do not. **PIGINet's crops are real here** (the
robot-skip fix), so this is genuine, not the dead-channel artifact: the low-level predictor learns
*something* (beats astar at n=9, loses at n=8) but nets out at the naive order. **SPECTRE ≈ LAZY** (both
abstract; the two adaptive-vs-adaptive tie), and **adaptivity is inert** (SPECTRE-adaptive ≈ static) —
consistent with v2/DD2D-s0. §2b (synthetic) mirrors FP: learned ~1016 s vs PIGINet/astar ~3345 s.

**⚠️ Read as an upper bound on the representation gap, not a real-refiner result.** The labels are the
*exact analytic* capacity/height function (no MP noise), which favours the representation that encodes
geometry directly (SPECTRE's point cloud) over one that must read it from pixels (PIGINet). A
real-refiner audit would price how much of the −27 FP survives MP stochasticity. The direction (abstract
≫ low-level, growing with crowding) is robust; the magnitude is synthetic-amplified.

**Takeaway-next.** Read every number here as a **synthetic-dataset probe**, not a real-refiner result:
FP reflects the geometry classifier and §2b is r_cap-derived, not measured (the real refiner stays the
future paper-eval instrument, reachable via `refiner_mode='real'`). Next: whether the §0 representation
edge (SPECTRE > PIGINet) appears on the tighter n=8/9 strata now that block *selection* matters and
PIGINet's height channel is live; and a real-refiner audit slice to price the synthetic labels/times.

---

<a id="2026-08-20-restock3d-v3-built-through-gates"></a>
## 2026-08-20 — restock3D v3 BUILT through the gates: per-object-dims env + analytic-collection classifier

<!--strip-->
> **id** `2026-08-20-restock3d-v3-built-through-gates` · **status** active ·
> **tracks** method, env-restock3d, evaluation, data, tooling
<!--/strip-->

**What.** Built restock3D-**v3** — the additive, per-object-dimensions successor to the too-easy
`restock3d_v2` (v2 stays frozen as the negative control) — through the three pre-collection gates. v3
makes block **selection** matter: per-object **widths** ~U[0.02, 0.08] and **heights sampled near the
short/tall cutoff**, on the re-balanced (0.27, 0.22) partition. Collection will use an **analytic
refinability classifier** (pure geometry, no motion planning); the **real refiner** stays the paper-eval
instrument, and their agreement is *measured* (Gate G1). Gated increments, all tested:
- **`feasibility_v3.py`** — single source of truth: capacity formula `Σw + 0.06(n−1) + 2·0.04 ≤ 0.50`,
  cutoffs (short ≤ 0.12, tall ≤ 0.17), split enumeration, the two greedy hand-rules, and
  `classify_skeleton` (emits `refiner_metadata["failures"]` dicts byte-compatible with the real
  `failure_metadata`: height-F3 culprit-free, crowding-F2 = residents, reach-over-F4 via the *shared*
  `_blocks_reach`).
- **env** — `ObjectCentricRestock3DEnvV3` rebuilds movable bodies per seed (object *set* fixed per
  stratum → constant-object Box unaffected); `place_controller_v3` promotes the harness L2R packer to
  production (state-reading slots, consistent-by-construction with `level_fits` — fast parity test);
  `models_v3` reuses v2's operators/abstractor. Real-refiner **F3 parity**: `_probe_place_v2` gained an
  optional arm-insertion cutoff (v3 only; v2 byte-identical when None) so a block in (cutoff, clearance]
  is a provable F3 matching the classifier.
- **generator_v3 + strata_v3** — role-banded heights + widths, enumerate every split, accept on
  (≥1 feasible split, fill band, ρ band, hard strata: both greedy rules fail). 4 strata n=6/7/8/9 on the
  **shared 4-band** (so `compare.stratum_of` needs no edit). Registration: gym ids, `collect.py`
  dispatch (models, geometry-guided pool, real refiner **with cutoffs**, analytic emitter, scene
  geometry), `env_registry`/`domain`; verified end-to-end via `collect_episode` (I5 passes once
  `scene_geometry` learned the `obj_goal` prefix).

**Result.**
- **Calibration** (1500 raw draws/stratum, `restock3d_v3_calibrate_generator.py`): **build-skip 0%**,
  spawn-fail 0%, clean ρ gradient med **0.172 / 0.109 / 0.012 / 0.006**, fill med 0.70→0.96 (≈ f(n)),
  near-threshold blocks med 3–4/problem. Hard strata cost ~18–30 reseeds but always resolve.
  `data/spectre/derived/restock3d_v3/generator_calibration.md`.
- **G3 (difficulty)** — hard strata defeat **both** greedy hand-rules **100%** of the time (by
  construction) with culprits **spread across 8–9 distinct objects** (top-object concentration 0.12–0.15).
- **G2 (static ceiling)** — on a *balanced* (scene, split)→fits set, a probe hits **1.00 on clear cases
  but ~0.88 near-threshold** (perception-degraded ≈ same) — the static representation is **not
  saturated**, the §0 near-threshold headroom. (A naïve random-split G2 was degenerate: at ρ≈0.006 a
  random split is ~99% infeasible; balancing fixed it.)
- **G1 (analytic↔real)** — a first pass with a flat **10 s/candidate** cap read 88% agreement but
  **TP=0**: nothing refined in time. That cap was ~4× too short — a feasible 6-object v3 plan needs
  **~40 s** of real MP (the L2R packer seats every block; **2/2** analytic-feasible candidates succeed in
  **41 s**). Re-scored with a **label-aware budget** (feasible 90 s, infeasible 12 s — an infeasible plan
  fails regardless of time): a clean pilot reads **32/32 = 100.0% agreement** (TP 1, TN 31, 0 FP, 0
  disagreements); with the first pass's infeasible side and the generous-budget feasible check that is
  **feasible ⟹ real-success 3/3** and **infeasible ⟹ real-fail 84/84** — the analytic classifier is a
  **valid proxy**. The 7 "FP" of the first pass were 10 s-cap timeouts, not true disagreements.
- **Plan-generation difficulty — hff vs the geometry prior** (`restock3d_v3_plan_attempts.py`, analytic
  refiner, 200 problems/stratum, K=150 pool cap = the deployment budget). **solve%** = fraction whose
  first analytically-feasible skeleton appears within K; **FP** = failures before it (over solved only).
  The **geometry prior wins solve%, and the gap widens with n**: geom/hff **100/96** (n=6) → **97/91**
  (n=7) → **83/60** (n=8) → **61 / ~0** (n=9). hff on n=9 is not run — **censored on ~every accepted
  problem** (pilot FP 102–421, all > K) and prohibitively slow to enumerate — so it is reported
  assumed-fail. **FP-among-solved is tied** (geom ≈ hff, even slightly higher for geom — a selection
  effect: geom solves the extra hard problems hff censors, and those carry higher FP). **The surprise vs
  v2:** v2's geometry prior beat hff **~200×**, but on v3 it is only a modest edge — because the prior
  orders by **pick distance (reach-over)**, which *was* v2's whole difficulty, whereas **v3's difficulty
  is the width-packing tall/short split** the pick-order prior does not touch. So it lifts coverage but
  leaves **~17–40% of n=8/9 unsolved within 150 candidates** → motivates a **v3-specific capacity-aware
  plan-gen prior** (the deferred eager section-capacity heuristic). Measurement is **memory-bound** — the
  n=9 hff search graph is ~5 GB/worker, so 24 workers swap-thrashed a 59 GB box; capped at ~10 workers.
  Table appended to `data/spectre/derived/restock3d_v3/generator_calibration.md`.

**Takeaway-next.** The analytic-collection premise **holds** — analytic labels match the real refiner
given an adequate real budget, and the analytic labeler makes collection ~40 s/candidate cheaper (no MP).
**Load-bearing:** the real-refiner *eval* budget must be **≥~60 s/candidate** (the strata_v3 r_cap
placeholders 45–80 s are in range; the 10 s-cap trap is the same one as DD2D's "selection metric never
censored below the tail"). Gates cleared → the deferred **collection pass** (analytic labels + 5%
real-refiner audit, budgets from G1) + training + the comparison wiring (compare_envs EnvSpec, precompute
cache, PIGINet v3 scene reconstruction — all deferred here). ADR: `decisions/07` 2026-08-20.

---
<a id="2026-08-20-restock3d-v3-crowded-feasibility-confirmed"></a>
## 2026-08-20 — restock3D v3 crowded feasibility CONFIRMED: re-balanced partition + left-to-right refiner

<!--strip-->
> **id** `2026-08-20-restock3d-v3-crowded-feasibility-confirmed` · **status** active ·
> **tracks** method, evaluation, env-restock3d
<!--/strip-->

**What.** Confirmation experiment that the *most crowded* uniform-footprint v3 scene is physically
solvable, using the fixes the calibration study pointed at. Standalone harness
`experiments/spectre/restock3d_v3_crowded_demo.py` (real controllers/env, no prod edits):
(1) **re-balanced partition** — divider moved DOWN 0.07 m, `section_clearances=(0.27, 0.22)` (was
(0.34, 0.15)), same total shelf height, so each section leaves ~0.10 m gripper headroom for its
max block; (2) **left-to-right analytic refiner** `LeftToRightSectionPlaceController` — computes
each block's EXACT slot x (leftmost centre + i·(w+gap), gap **0.06 m**) with ±0.01 m jitter and
only **5 place-samples**, replacing uniform x sampling; (3) **10 blocks, uniform 0.05×0.05
footprint** — 5 short (**0.12 m**) front row + 5 tall (**0.17 m**) back row (per-section height
limits from the calibration); (4) **oracle** — pick closest-first (right→left), place left-to-right
into the short (top) section, then the same for tall into the tall (bottom) section; (5) execute +
render mp4.

**Result. All 10 blocks placed — the crowded env is feasible.**
- **Single-block gate PASSED** (empty shelf): 0.12 m → short ✓, 0.17 m → tall ✓, and 0.17 m →
  short correctly **F3-fails** (the re-balanced sections still discriminate height).
- **Full 10-block oracle SUCCEEDED**: 5 short into slots x∈{0.164…0.604} (≤**3** place-samples
  each), then 5 tall into the same slots (**1** place-sample each). Video (2571 frames) at
  `envs/restock3d/demos/v3_crowded/crowded_10block_oracle.mp4`.
- **`--place-samples 5` is enough** — max used was 3. The analytic left-to-right refiner needs a
  handful of samples where uniform sampling needed ~18 (calibration), because each sample is at the
  correct x ± small jitter.
- **The flagged risk did NOT bite.** Filling the SHORT (top) section first, then reaching into the
  TALL (bottom) section *underneath the full top shelf*, was the easiest step (1 sample/block) —
  the 45° front-grasp reach into the lower section clears the resident top blocks. The ~0.10 m
  headroom (both sections at the marginal value) held under crowding.

**Takeaway-next.** The **short-section-cube-only blocker from the
[calibration entry](#2026-08-20-restock3d-v3-calibration-pick-place-envelope) is resolved by the
re-balance**: `section_clearances=(0.27, 0.22)` gives tall blocks up to 0.17 m AND short blocks up
to 0.12 m, both packable 5-across at a 0.06 m gap. So a real v3 can be built on: divider at 0.27 m
above the tall floor, block heights ~{0.12 tall-set-in-short, 0.17 tall-set-in-tall} (or a sampled
range up to those), width ≤ ~0.08 m, analytic left-to-right packing (gap 0.06, ≤5 samples). Next:
turn the hand-written oracle + fixed heights into the v3 generator (sampled widths/heights) and a
production left-to-right refiner, then the SPECTRE/PIGINet/LAZY collection. No ADR yet
(confirmation experiment); the v3 design ADR lands when v3 is built.

---

<a id="2026-08-20-restock3d-v3-calibration-pick-place-envelope"></a>
## 2026-08-20 — restock3D v3 calibration: pick/place envelope for varied block widths & heights

<!--strip-->
> **id** `2026-08-20-restock3d-v3-calibration-pick-place-envelope` · **status** active
> · **tracks** method, evaluation, env-restock3d
<!--/strip-->

**What.** restock3D **v2 proved too easy** for the baselines (LAZY near-oracle; preliminary
results, other session), so it can't demonstrate SPECTRE's representation advantage. Pivot to
**v3**, which makes block *selection* matter: (1) **varied block x-widths** (lateral) so choosing
the right subset per level is non-trivial rather than blocks being interchangeable, and (2)
**varied block heights sampled near the short/tall fit cutoff** so "tall vs short" is no longer
trivially separable. Before building v3, ran a **calibration/mapping study** of the *current*
kinematic env's front-grasp pick/place physical envelope (so v3's generator only samples feasible
(width, height) and its refiner packs with the right padding). New standalone harness
`experiments/spectre/restock3d_v3_calibrate.py` (imports the real controllers/env unchanged, **no
production edits**), 3 sweeps, process-isolated PyBullet workers, 12 pick+place retries (18 for
padding). Findings doc `docs/restock3d_v3_calibration.md`. **Measurement only — no v3 built yet.**

**Result.**

- **Heights (goal 1).** Feasible full-height (production *current* grasp): **tall section
  0.05–0.23 m**, **short section 0.05 m ONLY** (0.07 m already fails). ⚠️ **The short section is
  effectively CUBE-ONLY** — the front-grasp gripper needs ~0.10 m of vertical room ABOVE the block
  to place it, so the usable short height ≈ 0.15 clearance − 0.10 ≈ 0.05 m, **far below the 0.15 m
  geometric block-vs-ceiling clearance**. This is **too tight for v3's goal of sampling a height
  *range* near a short/tall cutoff in the short section**: v3 will need a **taller short-section
  clearance** (env change) or an **adjusted short-section place approach**. As-is, v3 height
  variation is a **tall-section** story (0.05–0.23 m); the short/tall decision degenerates to 'cube
  vs taller'. Tall-section max is 0.23 m (0.25 m fails — gripper vs the 0.34 m ceiling); min ≈ 0.05
  m both sections (0.03 m fails — fingers hit the floor).
- **Grasp scheme (goal 1, controller).** The user's proposal to **grasp at the block CENTER** for
  variable heights was tested and **REJECTED**: `center` and `capped-center` schemes FAIL placement
  more than the production `current` scheme (they fail the short section *entirely* and fail 0.05 m
  in tall) — a lower grasp point worsens the diagonal place reach-in. **The controller is already
  height-adaptive (`front_grasp_transform(half_z)`) and handles 0.05–0.23 m fine; keep it, do not
  center-grasp.**
- **Widths (goal 2).** Graspable-face ceiling = the finger aperture **≈ 92 mm open** (inner-pad
  separation; nominal 2F-85 stroke 85 mm). **The kinematic sim is width-PERMISSIVE** — it picks
  *and places* a 0.19 m block (widest tested) at attempt 0, because the grasp attaches
  kinematically and the target is collision-excluded during the reach-in. So **v3 must cap block
  width analytically in the generator** (the sim won't reject an over-wide block); recommend a safe
  max face width ≈ **0.08 m** (≈0.9× aperture). Width's real geometric effect is at **placement**:
  the usable centre-band shrinks by (half_x − 0.025) per side (0.522 m at w≤0.05 → 0.382 m at
  w=0.19).
- **Padding (goal 3), 5 methods vs a tall (0.24 m) neighbour, left-to-right.** Empirical min
  edge-to-edge gap between adjacent blocks (**M4**, real placement, binary-searched):
  **50 mm @ w=0.05, 33 mm @ w=0.07** (centre-to-centre ~0.10 m). **M3** gripper finger-overhang
  lower bound 53 / 44 mm corroborates. Both are **~5–8× the naive finger-pad-thickness estimate
  (~6 mm, M1/M2)** — the whole finger+knuckle assembly plus the diagonal reach-in swept volume
  bind, not the pad. **M5** n-in-a-row seated **5/5** at both widths (edge gap 55 / 38 mm). Capacity
  (M1): the 0.522 m band holds ~5 blocks at these gaps (matches the observed "5 cubes are hard to
  fit"). **Recommend v3 pack left-to-right with an analytic edge gap ≥ 60 mm** (measured max +
  ~10 mm safety), NOT uniform x-band sampling.

**Traps this exercised (each cost real debugging time).**
- **numpy-bool identity trap.** `x is True` / `x is not True` on a `numpy.bool_` (produced by
  comparing numpy-float positions) is always False/True — the two-object binary search silently
  returned `None`. Use truthiness or `bool()`, never `is True`.
- **PyBullet cross-sim interference.** Building >1 `ObjectCentricRestock3DEnv` sequentially in one
  process corrupts the *later* sim's motion planning (placement silently fails, no exception). Each
  measurement must run in its OWN process — the sweeps use `ProcessPoolExecutor`, one sim per worker.
- **Deterministic-x placement is BiRRT-stochastic**, independent of the controller's `default_rng`
  seed, so it needs ~18 retries to seat reliably at a specified x (vs the sampler's ~1) — hence
  `--pad-tries 18`.
- **Sim grasp ≠ real grasp aperture.** Grasping is a kinematic attach when one object overlaps the
  EE marker; the fingers close but the target is collision-excluded, so simulated picks never
  enforce the 85 mm stroke — width MUST be capped in the generator.

**Takeaway-next.** Build **restock3D-v3** on these numbers: **cap block width ≤ ~0.08 m** in the
generator, keep the production `current` grasp, and pack left-to-right with an **analytic edge gap
≥ 60 mm** (replacing uniform x-band sampling). **Height variation lives in the tall section
(0.05–0.23 m)**; the short section is cube-only, so before v3 relies on a short/tall height cutoff
it needs a **taller short-section clearance or a different short-section place strategy** — decide
that first. No ADR yet (this is measurement); the v3 design ADR lands when v3 is built.

---

<a id="2026-08-20-restock3d-4x3-stratum-added-3-strata"></a>
## 2026-08-20 — restock3D 4x3 stratum added: 3 strata x 3 seeds — SPECTRE edges PIGINet at 4x3, LAZY dominates

<!--strip-->
> **id** `2026-08-20-restock3d-4x3-stratum-added-3-strata` · **status** active ·
> **tracks** method, evaluation, env-restock3d, baselines
<!--/strip-->

**What.** The **4×3** section (banding stratum 3) finished collecting, so the restock3D
comparison went from 2 strata to **3** — {2×2, 3×3, 4×3} = banding strata {0, 1, 3} (the
remaining crowded strata 3×4 = 2 and 4×4 = 4 are still collecting). Rebuilt vocab, **retrained
all three learned methods × 3 seeds** on `--train-strata 0 1 3` / `--keep-strata 0,1,3` (125
train / 40 val / 40 test), and rebuilt the comparison cache from scratch (the {0,1} models are
stale). Config change: the §2b per-candidate refinement cap rose **50 → 55 s** — 4×3's slower
feasibles push the per-problem fastest-feasible max to 53.9 s (vs 45.1 s on {0,1}), so 50 s
would censor 9 problems; 55 s censors 0 while still cutting ~76 % of candidate refines. Same
deployed SPECTRE recipe (3D point-set + init/goal atoms fully ON).

**Result (mean failed attempts; 3 seeds, ± across-seed std, uncensored; s0=2×2 n=15, s1=3×3
n=15, s3=4×3 n=10; the stratum axis skips s2=3×4, not collected).**

| method | ALL | 2×2 | 3×3 | **4×3** |
|---|---|---|---|---|
| astar-dist (naive order) | 8.78 | 2.33 | 10.13 | 16.40 |
| PIGINet (low-level) | 1.96 ± 0.26 | 0.02 | 1.20 ± 0.12 | 6.00 ± 0.85 |
| SPECTRE-static | 1.44 ± 0.34 | 0.07 | 1.07 ± 0.13 | 4.07 ± 1.21 |
| SPECTRE-adaptive | 1.47 ± 0.37 | 0.07 | 1.07 ± 0.13 | 4.20 ± 1.30 |
| LAZY-adaptive | **0.19 ± 0.01** | 0.09 | 0.00 | **0.63 ± 0.06** |

Paired bootstrap over problems (seed-averaged, 10k resamples):

- **LAZY dominates, significantly.** vs SPECTRE-adaptive: ALL Δ −1.28 CI [−2.09, −0.64];
  4×3 Δ −3.57 CI [−5.93, −1.63] — both exclude 0. LAZY's GAT policy is far ahead of both the
  abstract and the low-level ranker on this env.
- **The representation advantage *starts to appear* at 4×3 but is not yet significant.**
  SPECTRE-adaptive − PIGINet: 4×3 Δ −1.80 CI [−3.87, **+0.10**] (SPECTRE ahead 4.2 vs 6.0, but
  the CI grazes 0); ALL Δ −0.48 CI [−1.10, +0.03]; 3×3 Δ −0.13 CI [−0.51, +0.22] (tied). So on
  the crowded stratum the abstract representation is directionally ahead of the low-level
  predictor — the first sign of the §0 crossover — but n=10 and a wide seed spread keep it shy
  of significance. **Do not claim the representation win yet;** it needs 3×4/4×4 + more seeds.
- **Adaptivity still gives SPECTRE no lift** (static 1.44 ≈ adaptive 1.47; paired Δ +0.03
  CI [0.00, +0.09], i.e. adaptive a hair *worse*). Consistent across all strata — the pools are
  feasible-dense enough that failure-conditioning has little to exploit.
- **All learned methods crush the naive planner order** (astar 8.78; astar's 4×3 is 16.40).
- **§2b wall-clock (55 s cap, ALL):** LAZY 53 s < SPECTRE 141 s < PIGINet 220 s ≪ astar 561 s
  — refinement-dominated (feasible candidates tens of seconds; plan-gen not measured, inference
  sub-second); the cap saves the highest-FP method most (astar 704 → 561 s).

**Takeaway-next.** The story sharpened: **LAZY is the clear winner, SPECTRE is pulling ahead of
PIGINet at the crowded 4×3 stratum (Δ −1.8, but CI includes 0), and adaptivity is inert.**
Whether the SPECTRE > PIGINet edge becomes significant is the thing to watch as **3×4 and 4×4**
land (they extend the crowding axis and add the asymmetric 3×4) and as seeds grow. Then the
VLMPlan arm. Supersedes the two-stratum snapshot in the
[2026-08-19 entry](#2026-08-19-restock3d-onboarded-comparison-spectre-piginet-lazy) (whose
pooled ALL was over {0,1} only).

---

<a id="2026-08-19-restock3d-onboarded-comparison-spectre-piginet-lazy"></a>
## 2026-08-19 — restock3D onboarded to comparison: SPECTRE/PIGINet/LAZY vs planner on 2x2+3x3 (3 seeds)

<!--strip-->
> **id** `2026-08-19-restock3d-onboarded-comparison-spectre-piginet-lazy` · **status**
> active · **tracks** method, evaluation, env-restock3d, baselines
<!--/strip-->

**What.** Trained the three learned methods on the restock3d_v2 **2×2 + 3×3** sections
(banding strata 0/1, 100 train / 30 val / 30 test after the strata-{0,1} filter), cached
their eval on the held-out test split, and added a **restock3D** section to
`compare_methods.py`. SPECTRE is the deployed recipe with the **new 3D additions fully ON** —
`--scene-3d` (analytic point cloud) + the full PointSetEncoder (`--use-pca-feats
--use-edgeconv --use-point-sab --pma-seeds 4`) + `--atom-mode profiles` (init abstract state +
goal atoms) — on top of `jaccard/coverage/aggregate-records/evidence-attn/state-delta/
select-window 5`. **3 seeds (0,1,2)** for all three learned methods (PIGINet gained a real
seed axis, `piginet_s{seed}` + `{seed}`-templated cache path; the CLIP cache is
checkpoint-independent and shared); astar is deterministic. No §4 ablation, VLMPlan deferred
(both user-chosen). The crowded 4×3/3×4/4×4 strata are still collecting in a separate session
and are out of scope here; a few stragglers on disk are excluded by the strata-{0,1} filter
(train + eval).

**Result (mean failed attempts before first success; 3 seeds, ± across-seed std, uncensored,
n=30 test).**

| method | ALL | 2×2 (s0) | 3×3 (s1) |
|---|---|---|---|
| astar-dist (naive planner order) | 6.23 | 2.33 | 10.13 |
| PIGINet (low-level) | 0.51 ± 0.04 | 0.02 ± 0.04 | 1.00 ± 0.12 |
| SPECTRE-static | 0.51 ± 0.08 | 0.07 ± 0.00 | 0.96 ± 0.15 |
| SPECTRE-adaptive | 0.50 ± 0.09 | 0.07 ± 0.00 | 0.93 ± 0.18 |
| LAZY-adaptive | **0.09 ± 0.04** | 0.09 ± 0.04 | 0.09 ± 0.04 |

- **Every learned method crushes the naive planner order** (6.23 → 0.09–0.51, a ~12–70× FP
  reduction), and the separation is entirely at **3×3**: 2×2 is near-trivial (pools tiny,
  feasibility easy — all learned methods ≈ 0, the "anchor" stratum this env's s0 already is),
  so read 3×3, not the pooled ALL.
- **LAZY is clearly the best (0.09 ALL / 3×3),** well outside seed spread of the rest.
  **SPECTRE ties PIGINet** (0.50 vs 0.51 ALL; 3×3 0.93 ± 0.18 vs 1.00 ± 0.12 — CIs overlap): on
  these easy symmetric strata the abstract representation and the low-level predictor do not
  separate. The 1-seed draw had SPECTRE 0.47 ahead of PIGINet 0.57; **with 3 seeds that gap
  dissolves into noise** — a caution against reading a 1-seed lead.
- **Adaptivity gives SPECTRE no lift here** (static 0.51 ≈ adaptive 0.50) — the small pools +
  high feasible density leave little for failure-conditioning to exploit.
- **§2b wall-clock (per-candidate cap 50 s, ALL):** LAZY 44.3 s < SPECTRE 70.3 s < PIGINet
  78.3 s ≪ astar 344.7 s. **Refinement-dominated** (real PyBullet MP, ~26–45 s / feasible
  candidate); plan-gen is not measured (0, a per-stratum constant dwarfed by refinement) and
  GPU inference is sub-second (PIGINet 0.04 s, SPECTRE/LAZY <0.02 s). The cap (50 s, above every
  problem's fastest-feasible → 0 censored) saves the highest-FP method most: astar 457 → 345 s.

**Takeaway-next.** 3 seeds, but still only the **two easy symmetric strata**, so no
representation/adaptivity claim: SPECTRE = PIGINet and static = adaptive here. The story to
watch is the crowded 4×3/3×4/4×4 strata (still collecting) — that is where reach-over + F3
crowding should force a real ordering and where adaptivity might finally earn its keep. Then
add the VLMPlan arm. The pooled ALL remains a two-stratum average; quote 3×3.

> **Update 2026-08-20 — superseded by the 3-strata result.** 4×3 (banding stratum 3) landed;
> the current comparison is 3 strata × 3 seeds — see
> [2026-08-20](#2026-08-20-restock3d-4x3-stratum-added-3-strata). Headline shift: SPECTRE now
> **edges PIGINet at 4×3** (4.2 vs 6.0, paired Δ −1.8 but CI includes 0), LAZY still dominates,
> adaptivity still inert. The two-stratum numbers below are the onboarding snapshot; quote the
> 3-strata entry.

---

<a id="2026-08-19-restock3d-v2-sequential-per-stratum-collection-redesign"></a>
## 2026-08-19 — restock3d v2 sequential per-stratum collection redesign

<!--strip-->
> **id** `2026-08-19-restock3d-v2-sequential-per-stratum-collection-redesign` ·
> **status** active · **tracks** env-restock3d, data, evaluation, tooling
<!--/strip-->

**What.** Redesigned the restock3d_v2 collection from one mixed 5-stratum job to **sequential,
gated, single-stratum jobs** + halved heavy strata, after the 11-worker mixed run — though safe
(freeRAM bottomed ~19 GB, no watchdog trip) — was memory-bound at ~3.6–3.9 days with an
unpredictable mixed-block-count per-worker peak. Also validated (mid-run, before stopping) that
the extrapolated heavy budgets hold: feasible solves land under r_cap (3×4 worst 71 s/80, 4×3
60 s/80, 4×4 77 s/90), and the 11-worker peak was safe.

**Result.**
- **Sequential per-stratum** (`SEQUENTIAL_ORDER=(0,1,3,2,4)`, `restock3d_v2_run_all.sh`): one
  process per block count → uniform predictable per-worker peak + full RAM reclamation between
  jobs. **Per-stratum worker sizing** `min(0.85·CPU, 0.85·freeRAM/PER_WORKER_GB[s])`, floor-
  guarded — verified across free-RAM levels {12,19,40,55} GB that `free_at_peak` stays ≥ the 6 GB
  watchdog floor for every stratum (2×2→27, 3×3→15, heavy→10 at 55 GB; degrades gracefully at low
  RAM). New `wRSSmax` heartbeat validated live on the 2×2 smoke: workers 1.2→1.6 GB (vs the 1.7 GB
  estimate).
- **Heavy strata halved to 25/10/10** (`strata_v2.SIZES`; light stay 50/15/15) → **295** episodes
  (175/60/60, was 400). Downstream is count-agnostic (`list_episodes`), verified across
  vocab/train/dataset/LAZY/PIGINet — uneven sizes are safe.
- **Resume pre-scan** seeds cells from on-disk episodes; read-only check found the **24 retained
  episodes** across all 5 strata correctly bucketed (train s0-4: 3/3/1/2/1; val 3/2/1/1/1; test
  2/2/1/1/0), all under target so none trimmed. Collection-path code (budgets, generator,
  `_config`, schema) unchanged, so the 24 stay consistent and resume.
- 9/9 strata unit tests green (sizes/per-worker/order/floor-guard); orchestrator `bash -n` clean.

**Takeaway / next.** Est. **~1.5 days** (was ~3.6–3.9), RAM predictable by construction, no
recollection (episodes retained). Watch `wRSSmax` on the first heavy stratum (4×3) vs the 4.5 GB
estimate; watchdog backstops. ADR:
[`decisions/07` 2026-08-19](../decisions/07-stickbutton2d.md#2026-08-19-restock3d-v2-collection-sequential-per-stratum-jobs).

**Follow-up (same day, live): 3×3 `PER_WORKER_GB` under-called → re-sized mid-run.** The 3×3
estimate (3.0 GB, interpolated) proved low — live `wRSSmax` reached **~3.8 GB** (bpg accumulation
over K_max=40 × r_cap=70 non-short-circuit refinement; completions arrive in ~40-min *waves* that
synchronize the workers). At the auto-sized **15 workers × 3.8 GB = 57 GB** a synchronized wave
drove freeRAM to the 6 GB floor (15 × 4 = 60 > 59 = OOM risk). Killed + `PER_WORKER_GB[1]` 3.0→3.8
+ relaunched: 2×2 skipped by pre-scan (0 m), 3×3 resumed 59 kept at the corrected **12 workers**
(45.6 GB peak, ~10 GB free). Confirms the redesign's own thesis — per-stratum RAM *is* predictable,
but only once the per-stratum estimate is right; the `wRSSmax` telemetry + kill/resume made the
correction cheap (no data lost). The **heavy estimate (4.5) is measured-backed (~4.0), not
interpolated**, so 4×3/3×4/4×4 should not need this.

**Heavy estimate validated + tuned for throughput (same day).** The first heavy stratum 4×3 ran
at 10 workers and its `wRSSmax` peaked at **3.7 GB** (candidate ~75/75) — under the conservative
4.5 estimate, confirming the heavy sizing was safe (freeRAM stayed ~26 GB) but left the CPU ~69%
idle (RAM-bound at 10 of 27 possible workers). Per user request, lowered `PER_WORKER_GB[2,3,4]`
**4.5→4.1** so the *fresh* 3×4/4×4 jobs auto-size to **11 workers** (~+10% throughput / ~2–4 h off
the run), leaving the running 4×3 untouched (kill+relaunch would waste ~12 core-h for 2 workers).
Chose 11 not 12 because the peak (3.7) sat at the 11/12 boundary and 11 stays floor-safe even on a
+0.5 GB hot batch (11 × 4.2 = 46 GB → ~9 GB free); 12 would lean on the watchdog. The `wRSSmax`
telemetry is what makes this tuning safe and evidence-based.

**Reverted — the 3.7 GB was an undersample.** Over the next ~2 h of 4×3, `wRSSmax` kept climbing
wave-over-wave: 3.7 → 4.0 → **4.3 GB** (the true crowded-strata peak; batch-dependent, and the
early first-wave sample undershot). At 4.3, 11 workers would leave only ~8 GB free (below the 9 GB
reserve; the running 4×3 at 10 workers stayed safe at ~16 GB throughout). So `PER_WORKER_GB[2,3,4]`
was set **back to 4.5** → 3×4/4×4 auto-size to **10 workers** (10 × 4.3 = 43 GB → ~12–16 GB free),
which is exactly what the original conservative 4.5 estimate targeted — the estimate was *right*;
only the "underutilized RAM" premise was wrong, because the real peak is ~4.3, not 3.7. Net lesson:
**measure the heavy per-worker peak across several waves, not one** — a single wave undersamples.
The revert was free (done before 3×4 launched; no restart). **Final correction (4x3 full run):** the peak kept climbing to **wRSSmax 5.5 GB** with **min freeRAM 4.4 GB and 1 watchdog pause at 10 workers** -- 10 grazed the floor. Raised `PER_WORKER_GB[2,3,4]` to **5.1 => 9 workers** for 3x4/4x4 (typical ~4.9 GB/worker -> ~14 GB free; rare 5.5 GB wave watchdog-handled). Lesson reinforced: **a heavy stratum's per-worker peak must be read from its FULL run (min freeRAM + max wRSSmax over all waves), not any single wave** -- the estimate was chased 3.7->4.0->4.3->4.6->5.5 before settling.

---

<a id="2026-08-19-spectre-atom-input-rung-built-smoke-verified"></a>
## 2026-08-19 — SPECTRE atom-input (Rung A) built + smoke-verified

<!--strip-->
> **id** `2026-08-19-spectre-atom-input-rung-built-smoke-verified` · **status** active
> · **tracks** method, env-dd2d, env-stickbutton2d, env-restock3d, evaluation
<!--/strip-->

**What.** Built the atom-input feature (Rung A) so SPECTRE sees the initial abstract state
atoms and the goal literals directly, as additive per-object profiles + a 0-ary global term.
New `AtomProfileEncoder` (`encoders.py`), `atom_mode` switch on `SpectreConfig`/`TrainConfig`
(+ `--atom-mode`/`--no-init-atoms`/`--no-goal-atoms`), tensorizer `_atom_profile_arrays` +
`atom_emission(cfg)` (mirroring the state-delta and `pointset_emission` idioms), threaded
through train / val-selection / `load_checkpoint` / deploy. Zero-init, config-gated,
module-selection so config-off is byte-identical (D-8). Scope = code + tests + smoke-verify;
2-arm FP evaluation deferred. ADR: [`decisions/07`
2026-08-19](../decisions/07-stickbutton2d.md#2026-08-19-spectre-atom-input-rung-initial-abstract-state).

**Result.** Verified end-to-end, no re-collection needed (all three envs already persist
`initial_abstract_state`/`goal_atoms`; the predicate vocab already walks both).
- **Tests** `tests/approaches/spectre/test_atom_input.py` (20) green: off-equivalence
  (no `atoms.*` keys, logits bit-identical), zero-init no-op at init on a batch that really
  carries atoms, atom-set permutation invariance, arg-slot sensitivity `On(a,b)≠On(b,a)` +
  predicate-binding `{q(a),r(b)}≠{q(b),r(a)}`, 0-ary→global-term-only, augmentation
  consistency (atom arg tags ⊆ scene tag namespace under permutation), OOV predicate → guarded
  id, object-order invariance with atoms on, checkpoint round-trip, collate shapes. **Full
  suite 453 passed** (the equivalence / state-delta / pointset tests unregressed).
- **Smoke train** `--atom-mode profiles`, 1 epoch each: **DD2D** (`n_pred=7`, unary+0-ary),
  **SB2D** (`n_pred=7`, exercises 2-ary init `RobotAboveButton`/`StickAboveButton` + 0-ary
  `AboveNoButton` global route), **restock3d_v2** (`--scene-3d`, `point_dim=3`, `n_pred=5`,
  strictly unary). Each wrote `best.pt`, round-tripped through `load_checkpoint`
  (`model.atoms` present, `atom_mode="profiles"`) and ran a real `deployed_rollout_traced`
  step — proving emission derived from `model.cfg` matches the trained architecture.
- **mypy** clean, **pylint** clean, **format** clean.

**Takeaway-next.** The capability is in and byte-safe; the *value* is a separate, deferred
question. Guide P1/P2 register DD2D/SB2D as **nulls** ($s_0$ derives from geometry the model
already ingests; the net historically under-uses symbolic tokens beside a richer signal), so
the honest next step is the 2-arm `baseline-static` vs `+atoms-static` FP comparison (3 seeds,
uncensored val FP + length-fit / profile-weight-norm probes) — not expected to separate on the
current envs. **Rung B** (per-atom-token cross-attention, reserved as `atom_mode="tokens"`)
becomes necessary only for a domain with binding-critical ≥2-ary init/goal atoms (guide P3); the
tensor surface already carries full per-atom binding, so it is an encoder swap, not a schema
change.

---

<a id="2026-08-19-restock3d-v2-collection-oom-post-mortem-ram-sized"></a>
## 2026-08-19 — restock3d v2 collection OOM post-mortem + RAM-sized concurrency fix

<!--strip-->
> **id** `2026-08-19-restock3d-v2-collection-oom-post-mortem-ram-sized` · **status**
> active · **tracks** env-restock3d, data, evaluation, tooling
<!--/strip-->

**What.** Launched the full 5-stratum collection (250/75/75, budgets from the 2026-08-18
re-check) overnight at `--workers 28` (0.9×CPU) with `--strata [2,3,0,1,4]` (heaviest first, to
surface the extrapolated 3×4/4×3 budgets early). Monitored for the OOM outcome the next morning.

**Result — OOM-killed the whole user session at 76 min, 0/400 kept.** `systemd-oomd` + the kernel
OOM killer fired at 23:55 (memory pressure 85% of the **59 GB** RAM — the box is 59 GiB usable,
not the 64 I'd budgeted against), killing the collection's Python workers **and gnome-shell + the
terminal**. No refinement bug (`err=0`, no tracebacks) — a **resource-sizing** bug, two factors:
- **28 heavy PyBullet workers × ~2 GB > 59 GB.** The OOM log caught one worker at 5.8 GB virt /
  2.09 GB anon-RSS.
- **Heaviest strata scheduled first** (`[2,3,0,1,4]` → 3×4/4×3 lead, ~100 min/problem). All 28
  workers were on the most memory-hungry problems simultaneously and **none had completed** when
  the OOM hit — hence `0/400 kept, 0 done` across 76 min of heartbeats.
- **Single-problem memory probes** (this session): 2×2 (K_max=20, r_cap=40) peak **1.36 GB**,
  20 skel/5 succ, 789 s; 3×4 (K_max=75, r_cap=15) peak **1.74 GB**, 75 skel/**0 succ** (r_cap=15
  is far below the ~45–49 s the pilot needs to pack 3×4, so 0 feasible — confirms r_cap=80 is
  genuinely required, not slack), 1545 s. Full-r_cap heavy peak ≈ 2 GB anon-RSS (the OOM figure).

**Fix (in `restock3d_v2_collect.py`; budgets/spec UNCHANGED).**
- **RAM-sized `--workers`** default `min(0.9·CPU, (avail−13 GB)/2.8 GB)` ≈ **16** on this box
  (was 28). Concurrency is **RAM-bound on the all-heavy tail**, not CPU-bound.
- **Round-robin submission** (was cell-by-cell): the ProcessPoolExecutor is FIFO, so interleaving
  cells makes the running set a **mix of strata** — memory ramps smoothly and a crowded-stratum
  problem still starts in the first workers, without the all-heavy spike. It does **not** lower the
  worst-case peak (the tail is unavoidably all-heavy — only crowded strata remain, and that tail is
  ~88% of the core-hours), so 16 is still sized for `workers × heavy_peak`.
- **Per-stratum ETA** in the heartbeat (a global mean badly misprices a 13 min vs 100 min mix),
  falling back to a `K_max·r_cap·1.05` prior; prints `eta~Xh freeRAM=XGB`.
- **Memory watchdog**: free RAM < 6 GB → pause new submissions (inflight drains, frees RAM);
  < 3 GB → stop. The collection is **resumable** (each kept episode is written immediately;
  relaunch skips existing files), so an abort loses only in-flight work.

**Takeaway / next.** The 16-worker relaunch (2.8 GB/worker) **was re-sized to 11 mid-run.** Live
monitoring at t+90 min caught workers at mean 2.99 / **max 3.62 GB and still climbing** (~candidate
67 of 75) → real crowded-strata peak ~**4 GB** (the `bpg` scratchpad accumulates all sampled states
across 75 candidates; the r_cap=15 probe was ~5× under). Worse, the **FIFO pool grows heavy-
concurrency toward `workers`** (freed light workers pull queued heavy tasks), so the all-heavy tail
would run 16 × 4 GB = 64 GB > 59 GB = **guaranteed OOM**. Caught at freeRAM 10 GB (watchdog hadn't
fired), killed both pids, relaunched `--workers 11` (11 × 4 = 44 GB → ~15 GB free all-heavy),
resumed the 10 kept from disk. **Cost: ~3.6–3.9 days at 11 workers, not ~2.** `_PER_WORKER_GB`
2.8 → 4.0. Side note: the mis-targeted-then-killed first finalize (waited on a stale transient pid)
inadvertently **proved the readiness gate** — vocab + all 3 smoke-trains ran to checkpoints on the
new schema (artifacts deleted; correct finalize rebuilds from the full data).
ADR: [`decisions/07` 2026-08-19](../decisions/07-stickbutton2d.md#2026-08-19-restock3d-v2-collection-ram-sized-concurrency-round-robin).

---

<a id="2026-08-18-restock3d-v2-collection-calibration-re-check-collection-path"></a>
## 2026-08-18 — Restock3D v2 collection calibration re-check (collection-path vs oracle refiner) + budgets set

<!--strip-->
> **id** `2026-08-18-restock3d-v2-collection-calibration-re-check-collection-path` ·
> **status** active · **tracks** method, env-restock3d, data, evaluation
<!--/strip-->

**What.** Before launching the full 5-stratum collection (2×2/3×3/3×4/4×3/4×4, 250/75/75), re-checked
the proposed per-stratum budgets (K_max 20/40/100/100/100, r_cap 40/55/65/75) against (a) the
Aug-18 geometry sweep (`heuristic_sweep_results.md`) and (b) the **pilot's own collected records**
— because the collection path and the sweep use *different* refiners.

**Result.**
- **The sweep timed the wrong refiner for r_cap.** Table B is `refine_skeleton_v2` (oracle certifier,
  short-circuiting, no backtracking); collection uses kinder's `BacktrackingRefiner`. Traced the
  substrate: `refinement_timeout_s` is cooperative (checked only at step-recursion entry), and on a
  failed place step the refiner **backtracks and re-descends**, so infeasible candidates do **not**
  fail fast — each F3/F2 negative burns ≈ r_cap. Per-problem cost ≈ K_max × r_cap.
- **Real collection-path feasible-solve wall (pilot records, per candidate):** 2×2 med 27.5 / max
  31.8 s; 3×3 med 46.5 / **max 60.1** s; 4×4 med 68.2 / max ~74 s (n=2). So r_cap 55/65/75 sit at or
  below the feasible tails → they would relabel the slowest feasible skeletons as negatives. Adjusted
  r_cap → **40/70/80/90** (3×4/4×3 = 80, extrapolated from 3×3/4×4). Pilot failure attribution is
  clean: **101 F3-proof negatives** (culprit-free + exhausted, e.g. `place_short(block_goal1)`) vs 2
  F1/F2 across the train split.
- **K_max from Table A raw per-seed oracle-index capture rates:** 2×2 max 14, 3×3 max 34 (20/40 fine);
  **3×4 = 50≡75≡100** (9/10 seeds ≤12, a lone 178 outlier uncapturable at any K); **4×4 needs 75**
  (three seeds at 66/69/71 that 50 misses; 100 adds nothing over 71); **4×3 is the only config where
  100>75** (50→7/10, 75→8/10, 100→9/10). Chose **K_max=75** across the three crowded strata (~25%
  cheaper than 100, one extra 4×3 resample).
- **Two smoke collects at short r_cap (25 s) gave 3×4 succ=0/15** — confirmed to be the r_cap artifact,
  not a generator bug: an isolated no-MP check shows the geometry generator emits clean all-`place_tall`
  skeletons for both asymmetric configs (3×4 at pool idx 0–17, 4×3 at 18–48, matching Table A's 4×3
  tail).

**Takeaway / next.** Budgets frozen in `strata_v2.BUDGETS`; run is ~20–30 h at ~29 workers, 4×4 the
dominant cost (each rejected draw burns a full pool). Fixed the 5-stratum banding collision
(`V2_STRATUM_BAND = SPLIT_BAND//5` + a v2 `stratum_of`), added recipe keys 14/15, rewrote the
collector (per-stratum budgets, `--test`, dynamic top-up, per-stratum heartbeat, census trim), and
widened the LAZY graph 8→9 (height feature). Collection outcome (per-stratum yields + wall-clock) +
the 3-method smoke-train confirmation to be logged when the run completes.

---

<a id="2026-08-18-pointsetencoder-upgrade-built-smoke-verified-dd2d-sb2d"></a>
## 2026-08-18 — PointSetEncoder upgrade built + smoke-verified (DD2D/SB2D 2D, restock3d_v2 3D)

<!--strip-->
> **id** `2026-08-18-pointsetencoder-upgrade-built-smoke-verified-dd2d-sb2d` ·
> **status** active · **tracks** method, env-dd2d, env-stickbutton2d, env-restock3d
<!--/strip-->

**What.** Implemented the PointSetEncoder upgrade (design doc
`docs/pointset_encoder_upgrade.md`; ADR
[2026-08-18](../decisions/07-stickbutton2d.md#2026-08-18-pointsetencoder-upgrade-per-point-differential-features-edgeconv))
as gated increments: (G1) tensorizer `dataset.compute_point_feats` — per-object Euclidean
kNN + local-PCA oriented normal + 2D signed curvature (`κ̂=tanh(κ·h̄)`) / 3D Pauly surface
variation, emitted as trailing-nullable `SpectreBatch.point_feats`/`knn_idx`; (G2)
`PointSetEncoder` (`lift→MultiSeedPMA→Linear(64·seeds→32)`) + `MultiSeedPMA` in `layers.py`,
selected by `SceneEncoder` only when a switch is on; (G3) `EdgeConv` (zero-init residual);
(G4) optional point SAB; (G5) five checkpoint-persisted `TrainConfig`/`SpectreConfig` flags +
CLI + `pointset_emission` threaded through `train`/`inference` so deploy emits what training
trained on. New `test_pointset_encoder.py` = T1–T7 (T7 relaxed to the LayerNorm form) +
checkpoint round-trip + config-off additivity.

**Result.** 21/21 new tests green; full spectre suite 428 passed / 1 slow-skipped (config-off
byte-identity intact). Analytic-shape feature checks: circle normals radial-outward
(dot≥0.98), κ̂ positive & ~constant; square corners spike, mid-edge ~0; horseshoe pocket κ̂
mean < 0, outer arc > 0; 3D box outward normals (dot(n,p)≥0). Smoke train (2 epochs, upgrade
on `--use-pca-feats --use-edgeconv --pma-seeds 4`) end-to-end on all three envs, each
producing `best.pt` that loads `strict=True` and rolls out: **DD2D** (2D), **SB2D** (2D,
val_fp 0.75), **restock3d_v2_pilot** (`--scene-3d`, 3D `C_pt=8`/`k=16`, val_fp 0.00 on the
tiny/easy pilot). Checkpoint round-trip confirmed `use_pointset=True`, `point_dim`,
`use_pca_feats`/`use_edgeconv`/`pma_seeds` restored.

**Takeaway-next.** No re-collection was needed — all three pipelines are compatible as-is
(features are tensorizer-time). One observed risk to watch in the retrain: on restock3d's
thin/tall analytic boxes `k=16` of 32 points mixes opposite z-faces, so 3D surface-variation
`f` is near-uniform (harmless to the smoke run, but a candidate to fix via a smaller 3D `k`
or a same-face neighbor filter). Deferred (doc §7): the 3-seed retrain + paired-CI FP
guardrail comparison, and the `[area, sinθ, cosθ]` scalar removal (a separate follow-up after
the retrain lands).

---

<a id="2026-08-18-restock3d-v2-3d-spectre-pipeline-pilot"></a>
## 2026-08-18 — restock3d v2 3D SPECTRE pipeline: pilot collection + train verified

<!--strip-->
> **id** `2026-08-18-restock3d-v2-3d-spectre-pipeline-pilot` · **status** active ·
> **tracks** method, env-restock3d, evaluation, data
<!--/strip-->

**What.** Built and verified end-to-end the data-collection + training pipeline that makes
Restock3D **v2 (continuous packing)** collectable and SPECTRE-trainable in **full 3D** (the
representation decision: cubes and tall blocks share a 2D footprint and differ only in
height, so a 2D-footprint scene would be blind to the F3 axis). Six gated increments:
(0) a 3D `SceneGeometry` producer `envs/restock3d/scene_geometry.py` — per object an
**analytic axis-aligned-box point cloud** (32 pts) from ground-truth half-extents + a 3D
pose (`schema.ObjectGeometry` gained optional `point_cloud`/`pose_z`/`height`, `None` for 2D
envs so DD2D/SB2D are byte-unchanged); (1) a **config-gated 3D encoder widening** —
`SpectreConfig.point_dim`/`pose_dim` (default 2/3), `FootprintEncoder`'s input `Linear(2→3)`
and `SceneEncoder.pose_proj` `Linear(3→4)`, a `sample_point_cloud` beside `resample_ring`,
and `--scene-3d` on `train` (inference derives it from `model.cfg.point_dim`); (2) the v2
**instrumented refiner** — `_probe_place_v2` dispatches on `place_tall`/`place_short`, F3 =
ceiling collision (culprit-free), **F2 = section-resident enumeration** (continuous packing
spreads residents across the wide band, so the v1 centre-point probe misses them); (3) v2
**collection registration** — `model_name="restock3d_v2"` branches in `collect.py`, a
`Restock3DV2Env` gym wrapper, `strata_v2.py` (banding stratum 0–3 → committed
`generator.STRATA` recipe keys 10–13 = 1×1..4×4; recipe rides in `model_kwargs` so
`config_hash` + `git_sha` pin it — no runtime injection), a domain-spec + aug-policy entry,
`conf/env/restock3d_v2_pilot.yaml`, and a geometry-guided pool generator wired as the v2
default; (4) vocab; (5) the pilot collector `restock3d_v2_collect.py`.

**Result.** All gates pass with committed unit tests
(`test_restock3d_v2_scene_geometry.py`, `test_scene_3d_widening.py`,
`test_restock3d_v2_refiner.py`; 407 fast tests green, 2D path unchanged). A one-problem
`collect_episode` yields a valid `EpisodeRecord` — pool with feasible + `place_short`-on-tall
**F3** negatives, 3D `scene_geometry`, real culprits, `validate()` OK. **Gate 5 (the
headline):** SPECTRE trained on the 1×1 pilot (`--scene-3d`, `n_train=6` — geometry-carrying
episodes NOT dropped, the silent-no-checkpoint trap avoided) → `best.pt`; the 3D checkpoint
**loads (`point_dim=3`/`pose_dim=4`, `strict=True`) and rolls out** (FP 0 on val). So the full
3D chain — producer → point-cloud encoder → train → checkpoint → load → deploy — works.

**Collection cost finding.** Full-pool refinement is expensive on this real-MP env: the
geometry-guided pool is F3-heavy (`place_short`-on-tall variants), and each infeasible
skeleton burns its full sampling budget in motion planning before failing (~40 s at
18 retries). A **10-retry** budget even rejected feasible 2×2+ skeletons (the sweep used 18);
the tractable recipe is **18 retries + a small `K_max`** (the geometry generator front-loads
a feasible skeleton into the top ~8–12) — larger `K_max` blows up wall-clock without adding
label diversity that matters for the pilot. 4×4 stays the slow/low-yield corner (genuine
section overflow, per the sweep's 4/10 oracle solve-rate).

**Baselines (all built 2026-08-18).** The shared oblique renderer `envs/restock3d/render.py`
(height-visible; world→pixel projection for Set-of-Mark labels + per-object crops) is verified.
**PIGINet** (`baselines/piginet/restock_adapter.py`, `--domain restock3d`) trains on the pilot
→ `ckpt.pt`, its CLIP cache built from the height-aware oblique crops (height reaches it through
the *image*, so its 2D shape scalars + encoders are unchanged). **LAZY** (`make_lazy_domain`
restock branch) trains → `ckpt.pt` (one residual 3D touch-up: graph `geom_dim` 8→9 for a height
node feature). **VLMPlan** (`baselines/vlmplan/restock_adapter.py` + `RestockOffPoolLabeler`,
registered) is built and smoke-tested — grounding validates plans, geometry disclosure flags
TALL vs cube, the labeled oblique snapshot renders; a full VLM *run* needs a backend (billed).

**Pilot collection.** All four configs collected (train s0=8/s1=7/s2=4/s3=2; the 4×4 corner is
overflow-limited, thin by construction). All three trainable methods (SPECTRE-3D, LAZY, PIGINet)
train to checkpoints on the 4-config pilot.

**Takeaway-next.** Full 500/100/100 collection needs **per-config refinement timeouts** (35s
suffices for 1×1/2×2 but 3×3/4×4 need ~75s+ for their feasible refines — averaged ~9-10 min per
3×3/4×4 episode); the LAZY graph height feature; the VLMPlan VLM run; and a `compare_envs`
EnvSpec. The v2 3D scene representation is the load-bearing new decision; see the ADR.

---

<a id="2026-08-18-restock3d-v2-n-tall-x-n"></a>
## 2026-08-18 — Restock3D v2 (n_tall x n_short) sweep: geometry plan-gen attempts + oracle-refinement solve-rate/wall-clock

<!--strip-->
> **id** `2026-08-18-restock3d-v2-n-tall-x-n` · **status** active · **tracks** method,
> env-restock3d, evaluation
<!--/strip-->

**What.** Swept the two v2 metrics over a **(n_tall × n_short) = 1..4 × 1..4 grid** (16 configs, ≤8
objects; the 5-object columns are dropped — 5 objects overflow a section). New parallel harness
`experiments/spectre/restock3d_v2_heuristic_sweep.py` (16×10 = 160 tasks, `ProcessPoolExecutor` fork,
**25 workers = 0.8×32**, single-thread BLAS; a temporary `STRATA`/`CLUTTER` injection builds arbitrary
counts) records per (config, problem): (A) geometry plan-generation **attempts** to emit the oracle plan
(K_MAX=2000; the `plan_generator_v2` heuristic) and (B) **oracle-plan refinement** solve + wall-clock
(`refine_skeleton_v2`, `attempts_per_step=18`, **90 s cap** via a new optional `max_seconds`). Results →
CLI + `data/spectre/derived/restock3d_v2/heuristic_sweep_results.md` (+ JSON). Whole sweep ran in
**270 s (~4.5 min)**.

**Result (16 cells, N=10 each).**
- **Table A — geometry plan-gen: success 10/10 in every cell** (geometry always generates the oracle
  plan). Mean attempts **scale ~2× per tall block** (averaged over n_short: tall 1/2/3/4 → ~1.6 / 6.6 /
  12.3 / 26), weakly increasing in n_short, and **heavy-tailed** at large configs (std ≈ mean, e.g.
  (3,4) 23.1±54.6, (4,3) 41.2±43.1). The tall count drives it because the tall/short section lottery is
  exactly what the pick-cost does *not* control.
- **Table B — refinement: solve-rate is a clean capacity signal**, 144/160 solved overall. 10/10 up to
  ~5 objects, then degrades as objects-per-section rise: (2,4) 8/10, (3,3) 9/10, (3,4) 7/10, (4,2) 8/10,
  (4,3) 8/10, **(4,4) 4/10**. **All 16 failures are genuine section overflow** (`place failed → section`,
  10 short-cube / 6 tall-block) — **zero cap timeouts** (max wall 67.3 s < 90 s), so the degradation is
  real continuous-packing crowding, not a cap artifact, and it confirms **both** sections overflow near
  5 (5 cubes *and* 5 blocks), the reason the grid stops at 4×4. Wall-clock rises monotonically with
  n_total, **14.7 s (2 obj) → 59.2 s (8 obj)**.
- **r3 cell (n_tall=2, n_short=4) reproduces the regime:** plan-gen 8.7±15.1 (CI [2.0, 18.9]) — a lower
  but overlapping draw of the same heavy-tailed distribution as the r3 headline ~15–26 (different seeds
  + `PYTHONHASHSEED`); refine 43.7 s ≈ the 37.9 s solo calibration + ~15% from 25-way parallelism.

**Takeaway / next.** Two orthogonal difficulty axes, cleanly separated: **plan generation** is
tall-count-limited (the geometry heuristic front-loads the oracle order but pays a section-lottery cost
that ~doubles per tall, heavy-tailed), while **refinement feasibility** is objects-per-section-limited
(continuous-packing crowding, ~4 per section is the practical ceiling, 4+4 solves only 40%). The 90 s
cap never bound, so it is a pure safety ceiling here. Wall-clock is reported under 25-way parallelism
(absolute times ~15% above solo); solve-rate is seed-deterministic and contention-free. Per-cell plan-gen
means are `PYTHONHASHSEED`-dependent (quote aggregates). Full tables in
`heuristic_sweep_results.md`.

---

<a id="2026-08-18-restock3d-v2-geometry-informed-pick-cost-vs-geometry-blind"></a>
## 2026-08-18 — Restock3D v2 geometry-informed pick-cost vs geometry-blind hff (oracle-plan generation)

<!--strip-->
> **id** `2026-08-18-restock3d-v2-geometry-informed-pick-cost-vs-geometry-blind` ·
> **status** active · **tracks** method, env-restock3d, evaluation
<!--/strip-->

**What.** A geometry-informed A* pick-cost plan generator for v2 (ADR
[`decisions/07` 2026-08-18](../decisions/07-stickbutton2d.md#2026-08-18-restock3d-v2-geometry-informed-pick-cost-heuristic-nearest-first))
head-to-head vs the stock geometry-blind hff generator on **oracle-plan generation**. New
`plan_generator_v2.py` (`GeometryGuidedRestockPlanGenerator`: pick cost `1 + λ·(#nearer OnFloor)`,
`d(o)=object y`, λ=1 — total plan penalty = Kendall-tau vs nearest-first) + eval
`restock3d_v2_heuristic_eval.py`. 10 r3 problems, enumerate the pool only (**no refinement**, so K_max
is cheap), match = **oracle south-to-north pick order AND both talls via `place_tall`** (F3-feasible;
the 4 cubes' section is free). Per-planner success-rate + mean/std/bootstrap-95%-CI of attempts.

**Result** (K=10000, r3 n=10; **three replicates** R1/R2/R3 — per-problem indices are
`PYTHONHASHSEED`-dependent, aggregates are stable):
- **Geometry: 10/10 success all three; mean attempts 17.9 / 14.8 / 25.5, 95% CI [6.6, 30.9] /
  [7.4, 23.0] / [6.0, 48.8].** Pick-order-only match = **attempt 1, 10/10 all three** (mean 1.0, sd 0) —
  the oracle pick order is literally the first plan yielded; the ~15–26 is the tall-section lottery
  *within* the leading 0-inversion band (16 of 64 section variants are talls-feasible).
- **hff: 8/10 / 6/10 / 5/10 success; mean attempts 3929 / 4070 / 3989** (strikingly stable ~4000), 95% CI
  ~[1700, 6800]. Pick-order-only 9/10 / 9/10 / 7/10, mean ~4100–4700. So hff needs **~150–275× more
  attempts** (~200× central) than geometry and still misses 2–5/10 at K=10000. At smaller budgets hff
  collapses: **1/10 at K=2000**, 0/2 on a K=1000 smoke — it needs ~K=10000 to reach 50–80%.
- hff is **worse than uniform-random** would predict (uniform ≈94% pick-order hit in 2000; hff got
  2/10): its A* enumeration is *clustered* by shared prefixes, not a uniform sample, so a specific
  ordering is reached slowly. Verified hff yields **2000/2000 distinct** plans — no duplicate inflation.
- **`d(o)=y` is forced, not a free choice:** sort-by-y reproduces the oracle south-to-north order
  **10/10**; Euclidean distance from the single robot start pose (0,0) reproduces it **0/10** (object
  band laterally offset at x≈−0.5, x-spread ≈ y-spread) and would optimize a different, non-reach-
  feasible order. The "park pose" is the per-object grasp station directly south; the operative distance
  is the northward reach = y.
- **Tests:** fast pure-Python `_edge_cost` (1 + λ·#nearer OnFloor; picked-away objects drop out; a place
  op is unit; λ scales) + slow enumeration (geometry's first plan == oracle pick order on r2). Both pass.

**Takeaway / next.** A deliberately weak, one-line geometric prior — *rank skeletons by Kendall-tau vs
nearest-first pick order* — is a large plan-generation win on the r3 hard tail: geometry-blind hff
essentially cannot generate the oracle plan at a practical budget (1/10 at K=2000) while the prior
front-loads it (10/10, oracle pick order at attempt 1). The pick-order win is by construction (honest
caveat); the *magnitude* (~200×, range 150–275× over 3 replicates) and hff's sub-uniform clustering are
the measured findings. This is a
**plan-generation prior**, separate from the deferred Phase-2 eager section-capacity heuristic — it does
not touch refinement, collection, or the learned ranker; levers are λ and `d(o)`. Quote **aggregates**
(per-problem indices are `PYTHONHASHSEED`-dependent).

---

<a id="2026-08-17-restock3d-v2-milestone-continuous-packing-certifies"></a>
## 2026-08-17 — Restock3D v2 milestone: continuous packing certifies r0-r3, Stage-0 gate passes

<!--strip-->
> **id** `2026-08-17-restock3d-v2-milestone-continuous-packing-certifies` · **status**
> active · **tracks** method, env-restock3d, evaluation, data
<!--/strip-->

**What.** Milestone build of **restock3d_v2**, the continuous-packing variant (ADR
[`decisions/07` 2026-08-17](../decisions/07-stickbutton2d.md#2026-08-17-restock3d-v2-continuous-packing-variant-two-place)):
two place operators `place_tall`/`place_short` (identical abstract effects; section validated by real
collision), **uniform x-band** sampling instead of discrete regions, geometric `Stored` (drop
`InRegion`). Additive — v1 byte-for-byte; the env is fed two wide `RegionInfo` "section bands" so
`kinematic_env.py` is unchanged. New modules `section_geometry.py`/`models_v2.py`/
`place_controller_v2.py`/`oracle_v2.py`; scripts `restock3d_v2_{stage0,oracle,demos}.py`. Verify
Stage-0 + oracle certification r0–r3.

**Result.**
- **x-band (analytic, no sweep).** Board x-extent `[0.099, 0.701]` (only the 3 boards collide — side
  walls are cosmetic) minus a 0.04 m per-side end margin → object-center band **`[0.139, 0.661]`**
  (`band_half_x = shelf_width/2 − 0.04 ≈ 0.261`), a hair wider than v1's reachable `[0.16, 0.64]`, inside
  the physical max `[0.124, 0.676]`. Two section bands: tall surface 0.29 (clearance 0.34), short 0.6427
  (0.15). y = front strip 1.35 + ±0.01 jitter.
- **Stage-0 4/4** (`restock3d_v2_stage0.py`): `place_tall(cube)`, `place_short(cube)`, `place_tall(block)`
  all place upright on the correct surface; **`place_short(block)` → F3** — the place raises across all 6
  attempts and the ceiling-slide confirms `overlap=True` (the upright 0.24 m block jams under the short
  section's ceiling board). Cubes land upright via the translate-only section place (euler [0,0,0]).
- **Oracle certification r0–r3 = 12/12 (3/3 each)** (`restock3d_v2_oracle.py`), plan_len **3/5/4/6**,
  59/100/80/121 s. Continuous packing certifies on every stratum: r1 (5 cubes, load-balanced 3-short /
  2-tall) and r3 (2 talls tall-section + 4 cubes balanced) pack via per-step resampling — a place whose
  sampled x collides a resident resamples to a free x.
- **Retry budget is load-bearing.** At 6 attempts/step r0 read **2/3** (one last-cube place flakiness);
  at **18** (matching v1's documented ~1/6 placement reliability — each attempt resamples x across the
  band + the pick standoff/rot) every stratum is 3/3. MP is otherwise deterministic (internal seed 0), so
  the diversity comes from the resampled placement/pick params, not MP randomness.
- **Tests:** fast 7/7 (`test_restock3d_v2_{geometry,models}`: x-band bounds, section surfaces, F3
  invariant; predicate set has no `InRegion`; both place ops add `Stored`, 2 params, no region);
  slow (`test_restock3d_v2_oracle`) certifies r0/r2 and confirms the F3 negative — `place_short(block)`
  never stores while `place_tall(block)` does.

**Takeaway / next.** v2 is a working 3D **continuous-packing** testbed: the tall/short choice is a
*symbolic operator token* (`place_tall`/`place_short`, identical abstract effects) validated by real
geometry, and crowding is emergent (overlap→collision→resample), not a discrete slot. The milestone is
complete (env + oracle + Stage-0 + demos + tests + docs). **Phase 2 (deferred):** the v2 instrumented
refiner (F3 + reach-over transfer directly; **F2 → continuous section-capacity attribution**), the eager
section-capacity heuristic, K_max/cap_r recalibration, and collection registration (`strata_v2`,
`ENV_VARIANT="restock3d_v2"`), before any SPECTRE training / learned baselines / `compare_envs` EnvSpec.

---

<a id="2026-08-17-restock3d-reach-blockers-reach-over-eager-relation"></a>
## 2026-08-17 — restock3d reach_blockers (reach-over eager relation) + K_max re-calibration

<!--strip-->
> **id** `2026-08-17-restock3d-reach-blockers-reach-over-eager-relation` · **status**
> active · **tracks** method, env-restock3d, evaluation
<!--/strip-->

**What.** Gate F of the fully-lateral rebuild
([`decisions/07` 2026-08-17](../decisions/07-stickbutton2d.md#2026-08-17-restock3d-fully-lateral-layout-front-grasp-only-strict-collision)):
give the eager heuristic a geometric **`reach_blockers`** relation so the pool/K_max see the reach-over
difficulty (which the oracle already handles via south-to-north), and re-calibrate K_max.

**Result.**
- **Reach-corridor calibration (MP sweeps).** A single south neighbour rarely hard-blocks a front-pick
  (cube-over-cube always clears; a lone south cube even clears a *tall* target at dy=0.30). What blocks
  is **cumulative + involves a tall block**: reconstructing the dense grid-r3, `pick(tall_block)` with
  both same-column cubes present fails **MMM** (3/3 MP-fail), clears to **GGG** once they're removed;
  other-column cubes (dx=0.30) don't block. A tall block directly in-line (dx=0) blocks a *cube* target.
- **Model.** `reach_blockers[B] = {A : A south of B by ≥0.03 m, |A.x−B.x| < 0.12 m, and (A or B is a
  tall block)}` — a **conservative** proxy (over-forbids a lone-blocker case, but safe: south-to-north
  always satisfies it; captures the real multi-object blocking). On grid-r3 it flags exactly each tall
  block's two same-column cubes; `is_feasible_skeleton` then marks **talls-first False, south-to-north
  True**. Wired into `eager_tables` (`build_tables` geometric compute, `is_feasible_skeleton` +
  `make_penalty` union it with the F1 `blockers`).
- **K_max (8 problems/stratum, plain hff vs eager first-feasible).** plain_ff_max / K_max_r:
  **r0 3/4, r1 69/83, r2 173/208** (1/8 censored), **r3 95/114 (7/8 censored past 200)**. The
  reach-over-aware **eager surfaces the feasible at index 0 on r0–r2** (eager_ff_all_zero=True);
  **r3 is the hard tail** — eager_ff is 0 when found but censored past K=50 on the hard problems (the
  oracle certifies r3, but F2+F3+reach-over combine so the pool rarely front-loads a feasible), the same
  unenumerability v1 saw on r3, now with the reach-over constraint added.

**Coverage revived (follow-on).** The reach-over also **revives coverage** — the failure mode F1
retirement was thought to kill. A reach-over pick failure is now attributed by `reach_over_culprits`
(`instrumented_refiner`; the shared `_blocks_reach` geometry, family **F4**) to the un-cleared south
blockers — class-1, actionable culprits — so coverage is live with the **correct** polarity (opposite of
F2, which inverts): south-to-north candidate coverage **1.00** vs talls-first **0.00**
(`restock3d_coverage_probe.py`, rewritten from the F1 probe). **Waste stays degenerate** (the fix is a
reorder of goal-necessary picks, no discretionary relocation; reviving it needs a non-goal
approach-corridor clutter, one flag away). `reach_over_culprits` and the eager `reach_blockers` share
one geometry source now (`_blocks_reach` in `instrumented_refiner`).

**Takeaway / next.** `reach_blockers` + K_max + coverage done; the eager works for r0–r2 and coverage
is live (reach-over). Open: r3 is unenumerable within the pool cap (needs a larger K or a
staged/relocation-aware generator — deferred with collection); the conservative corridor model
over-states enumerability difficulty (a precise cumulative/depth model is a refinement); waste needs
approach-clutter. cap_r stands at r0 56 / r1 65 / r2 57 / r3 60 s (2–3× v1, strict collision +
front-grasp). Fast tests 29 passed.

---

<a id="2026-08-17-restock3d-fully-lateral-rebuild-oracle-certifies-front-grasp-only"></a>
## 2026-08-17 — restock3d fully-lateral rebuild: oracle certifies, front-grasp-only, reach-over ordering

<!--strip-->
> **id**
> `2026-08-17-restock3d-fully-lateral-rebuild-oracle-certifies-front-grasp-only` ·
> **status** active · **tracks** method, env-restock3d, evaluation
<!--/strip-->

**What.** Rebuilt restock3d for true collision-free realism (base was phasing through floor blockers):
fully-lateral disjoint x-bands (buffer | objects | shelf), front-grasp for all picks, strict base
collision + no fallback, region rejection sampling. ADR:
[`decisions/07` 2026-08-17](../decisions/07-stickbutton2d.md#2026-08-17-restock3d-fully-lateral-layout-front-grasp-only-strict-collision).

**Result.**
- **Gate A — front-grasp for cubes+blocks:** Stage-0 **4/4** (cube→tall, cube→short, block→tall,
  block→short still fails F3, ceiling overlap=True); relocate→store **PASS**. The existing
  `front_grasp_transform` grips a cube fine (no calibration change). Only fix: `BufferPlaceController`
  base standoff 0.52→0.72 (the top-down envelope folds the arm into its own base; documented `d≥0.70`).
- **±x blocker calibration (the pivot):** swept a clutter cube AND a full-height block across the whole
  neighbourhood of a target (dx,dy ∈ ±0.14 m) — `grasp_blockers` returns **empty at every offset**.
  A floor neighbour never contacts the arm at the front-grasp config; front-grasp obstruction is only an
  approach-path reach-over. ⇒ ±x sample-and-verify blockers can't work; user chose **reach-over ordering
  only** (no clutter, no buffer).
- **Gate B/C — layout + strict collision:** first oracle run r0/r1/r2 **3/3** but **r3 0/3** (timed out
  at 200 s). Root cause: the grid stacked tall blocks in the back row with cubes in front of them, and
  the front grasp reaches north *over* nearer objects → the naive talls-first order is reach-over-blocked.
  Fixed by ordering the oracle store phase **south-to-north (nearest-first)** → **r3 3/3**. Full re-run,
  all strata **3/3**: cap_r (max×1.2) r0 24.0 / r1 63.9 / r2 42.9 / r3 60.1 s.
- **Gate D — region sampler:** oracle certifies **randomly-sampled** scenes **4/4 on every stratum**
  (cap_r r0 56 / r1 65 / r2 57 / r3 60 s). Exclusion radius (0.12 m) respected, in-band, random object
  types, deterministic.
- **Tests:** fast restock3d suite 29 passed; slow suite 5 passed + 4 skipped (retired F1-clutter tests),
  0 failed.

**Takeaway / next.** The core (fully-lateral, collision-free, front-grasp-only, region-sampled) is done
and oracle-certified; base phase-through resolved for real (strict collision, no fallback). cap_r is
~2–3× v1 (strict collision + front-grasp refinement) — a re-calibration note. The reach-over is the
new difficulty (naive order fails, south-to-north succeeds); **next**: a geometric `reach_blockers`
relation in the eager so the pool/K_max see it, then K_max re-calibration. Buffer/relocation machinery
kept inert (CLUTTER=0), F1 retired; taxonomy now F2 + F3 + reach-over.

---

<a id="2026-08-15-restock3d-f1-clutter-build-mechanism-calibration"></a>
## 2026-08-15 — Restock3D F1 clutter build: mechanism, calibration, pool-generation limits

<!--strip-->
> **id** `2026-08-15-restock3d-f1-clutter-build-mechanism-calibration` · **status**
> active · **tracks** method, env-restock3d, evaluation, data
<!--/strip-->

**What.** Re-added F1 grasp-obstruction clutter + relocation to Restock3D (deferred in v1), fixed the
base-drives-through-blocks bug first, re-ran calibration, and wired coverage/waste. Gated autonomous
run (0 base → 1 blocking → 2 mechanism → 3 sweep → {4 cap, 5 K_max, 6 coverage} → 7). ADR:
[`decisions/07` 2026-08-15](../decisions/07-stickbutton2d.md#2026-08-15-restock3d-f1-clutter-re-added-relocation-buffer).

**Result (by gate).**
- **G0 base collision.** Base footprint **0.55×0.51 m** (AABB) vs floor spacing **~0.30 m**.
  `check_base_collisions=True` + floor movables in base-nav → oracle **r0 50% / r1 0% / r2 0% / r3 0%**
  (wide base boxed). Best-effort (planner avoids floor movables + shelf-only fallback, flag **off**) →
  **r0–r3 100%**. Base-nav demo: pre-fix shelf-only plan drives through **6–7/N** waypoints; primary
  avoidance **refuses** on the short pick hops (no lateral room) so in dense scenes it falls back.
- **G1 blocking.** Cube (top-down grasp): clutter **+y, gap 0.05–0.10 m** → clean F1 (named culprit,
  clutter pickable, no cycle); +x/−x never block; gap 0.12 too far. Block (front grasp): **no** clean
  F1 (side clutter doesn't obstruct; close clutter is itself blocked → cycle). → F1 targets cubes.
- **G2 mechanism.** Oracle **certifies cluttered r1/r3 100%** (~1 call). Relocate→store demo PASS.
  Two substrate bugs found+fixed: (1) SE2 base-plan smoothing trips `SE2Pose`'s ±π assertion → fall
  back to raw waypoints; (2) **the floor was not a registered placement surface**, so a buffer place
  never released the cube (`grasped` stuck, `finger 0.29`) — fixed by counting the floor in
  `_get_surfaces_supporting_object`. 5 fast + 3 slow unit tests pass.
- **G3 sweep** (r1/r3 × k=0..3 × 12, plain K=200 + eager K=50). r1 k=0 plain first-feasible mean
  **34.3**; **every cluttered plain pool is censored** (0 feasible in top-200 — relocate-first plans are
  off the hff gradient). Eager surfaces the feasible **only on r1** (first-feasible **0**, pool=200);
  **r3 eager times out with 0 candidates** (F1+F3 unenumerable). Recipe = **r1=1, r3=0**. k≥2 also
  OOMs the plain K=200 enumeration.
- **G4 cap_r** (8/stratum, 100%): **12.4 / 18.3 / 21.3 / 28.2 s** (r0–r3) — clutter doesn't blow it up.
- **G5 K_max**: r0 **3**, r2 **64** (plain, no regression from adding `PlaceBuffer`); r1 plain censored,
  eager 0.
- **G6 coverage/waste** (P4 probe): culprit-pool K={clutter}; **coverage** relocate-culprit **1.00** vs
  direct **0.00** (RP-3); **waste** relocate-unblamed **1.00** vs relocate-culprit **0.50** (RP-4).
  Non-degenerate, no new compute code.

**Takeaway / next.** **F1 composes with F2 (r1) but not F3 (r3)** at the *pool-generation* level — the
oracle certifies r3 clutter but no planner enumerates a feasible relocate-first plan within budget, so
r3 stays F2+F3 and the deployed r1 pool must be **eager**. Deferred: the full relocation-aware
collection + training, r3 F1 (relocation-aware generator), enforceable base collision (navigable floor).

---

<a id="2026-08-15-restock3d-eager-heuristic-oracle-calibration-timeout"></a>
## 2026-08-15 — Restock3D eager heuristic + oracle calibration: timeout & K_max estimates (autonomous)

<!--strip-->
> **id** `2026-08-15-restock3d-eager-heuristic-oracle-calibration-timeout` ·
> **status** active · **tracks** method, evaluation, env-restock3d
<!--/strip-->

**What.** Autonomous overnight build + run (no human in the loop; ADR
[2026-08-15](../decisions/07-stickbutton2d.md#2026-08-15-restock3d-eager-validity-heuristic-oracle-solver-budget),
session log `docs/autonomous_restock3d_calibration_session.md`) to make Restock3D collectible: an
**eager-validity heuristic** (`astar_eager`), an **oracle solver**, and worker-parallel runs that
estimate the per-candidate **timeout** and **K_max** per stratum. Scope no-clutter v1 (F2+F3);
`samples_per_step=10`; parallel across problems (spawn, 8–24 workers on the 32-core box).

**Result.**
- **Strata reconciliation (step 1).** r0–r3 are correctly implemented for the no-clutter v1 design:
  `STRATA={0:(3,0,2,5),1:(5,0,1,4),2:(3,1,2,4),3:(4,2,3,5)}` = `(n_small,n_tall,n_tall_reg,n_short_reg)`,
  clutter hard-0 for every stratum. "No clutter in r1" is the **expected F1 deferral**, not a bug — r1's
  difficulty is short-cell over-assignment (σ_short=0). r2/r3 both compute d=(σt,σs)=(1,2); they differ
  only in raw counts.
- **Eager heuristic (V1–V3).** V1: on slack r0, eager≈plain (first-feasible 0 vs 1–2), penalties ≈0.
  V2: **eager first-feasible index = 0 on every r0–r3 problem** (the informed order front-loads a
  feasible skeleton). V3: F3 (tall→short) candidates are present in the **plain** top-K pool (~128 r2,
  ~174 r3) but **absent from the eager pool** (0), because λ_h=50 buries them past K=200 — expected,
  and the reason the training pool uses the plain order (ADR DC1).
- **Oracle + timeout (8 problems/stratum, 300 s budget).** 100% certified every stratum, ~1 refiner
  call each — feasible refinement is **fast** (single call), not the ~120 s feared. t_oracle max
  r0 11.4 / r1 19.8 / r2 19.0 / r3 23.7 s → **cap_r (max×1.2) = 13.7 / 23.8 / 22.8 / 28.4 s**
  (`data/spectre/derived/restock3d_v1/oracle_calibration.json`).
- **K_max (20 problems/stratum, no refinement).** Plain-order first-feasible → **K_max_r = 8 / 113 /
  48 / 179** (`ceil(max×1.2)`); r3 has 6/20 censored beyond K=200 (~1/200 density; eager finds them at
  0). Eager first-feasible index = 0 everywhere (collection short-circuit depth ~1)
  (`data/spectre/derived/restock3d_v1/kmax_estimate.json`).

| stratum | cap_r (s) | K_max_r (plain) | plain 1st-feas max | eager 1st-feas | F3 in plain pool |
|---|---|---|---|---|---|
| r0 | 13.7 | 8 | 6 | 0 | 0 (no talls) |
| r1 | 23.8 | 113 | 94 | 0 | 0 (no talls) |
| r2 | 22.8 | 48 | 40 | 0 | ~128 |
| r3 | 28.4 | 179 | 149 (6/20 censored) | 0 | ~174 |

**Takeaway / next.** All four deliverables are in hand: oracle (100% certify), eager heuristic
(first-feasible 0), per-stratum cap_r, per-stratum K_max_r. Two settled findings guide the deferred
collection design: (1) feasible-refinement is cheap and single-call, so the cap can be small (~14–28 s)
and infeasible candidates die at it; (2) the eager order finds the feasible instantly but strips F3, so
pool membership must stay plain (or hybrid) — sizing K_max to the eager index would give an FP-poor
pool. The no-refinement K_max is trusted because `is_feasible_skeleton` is a sound feasibility oracle
here (F2/F3 are real collisions; table-feasible certifies 100%), so the refinement-pilot fallback was
not needed. Deferred: F1/clutter/coverage-waste, full collection, learned baselines, `compare_envs`
EnvSpec. Traps this run added: eager enumeration to K=200 is memory-heavy (24 workers OOM-broke the
pool on r2/r3 → default lowered to 12 + a resubmit-at-4 self-heal); the oracle refine (lighter) is fine
at 24.

---

<a id="2026-08-14-restock3d-kinematic-stage-0-gate-collection-smoke"></a>
## 2026-08-14 — Restock3D kinematic Stage-0 gate + collection smoke

<!--strip-->
> **id** `2026-08-14-restock3d-kinematic-stage-0-gate-collection-smoke` · **status**
> active · **tracks** method, env-restock3d, evaluation
<!--/strip-->

**What.** Rebuilt Restock3D on kinematic PyBullet (ADR
[2026-08-14](../decisions/07-stickbutton2d.md#2026-08-14-restock3d-rebuilt-kinematic-pybullet-real-collision-gating))
and drove it through: (a) a **Stage-0 gate** — the four core cases (cube→tall, cube→short,
block→tall, block→short) via the real front-grasp / top-down controllers, rendered to mp4; (b) an
end-to-end **collection smoke** (`collect_episode` on r2/r3 through the wired pipeline); (c)
**solvability diagnostics** driving sequential multi-object stores.

**Result.**
- **Stage-0 gate PASS on all four, user-approved** after three motion-artifact fixes (arm folding
  through the base → front-pick standoff 0.40–0.50 → 0.70–0.75 + base in the arm collision set; a
  teleporting base → `remap_se2_pose_plan_to_constant_distance` + BiRRT `smooth_amt` 50 → 300;
  barebones render → `realistic_bg=True` room + cupboard walls). F3 made unmistakable by tightening
  the short-cell clearance 0.19 → 0.15 (block overhangs the cupboard top by ~0.09). Verified F3 is a
  clean reach-in collision: `navigated=True`, pre-place MP fails, geometric ceiling-overlap `True`.
- **Grasp-height fix (front pick):** the 0.24 m block is graspable only if the front grasp targets a
  **fixed EE height ~0.13 m** (grip lower on the block), not near its top — the arm's 45° reach tops
  out ~0.16 m, so a near-top grasp of a 0.24 m block puts the EE at ~0.23 m and IK/MP fail.
- **Collection wiring works end-to-end** (no errors): pool-gen → per-skeleton refine → F2/F3
  harvest. **The correct failure families appear:** r2 K_max=15 at a generous 160 s per-candidate
  timeout gave **10 F2 (over-assignment) + 5 F3 (tall→short), first_success=None**. F2 requires a
  *resident* — a prior *successful* place within the same skeleton — so **individual cube/block
  placements DO refine**; the mechanism is real. But **no *full* 8-step r2 skeleton refined in the
  first 15**, for two compounding reasons that are collection-tuning, not correctness: (i) the
  capacity-blind hff interleaves over-assigning / wrong-section skeletons ahead of the feasible ones
  (the intended high baseline-FP shape — but it means a feasible skeleton may sit beyond K_max=15),
  and (ii) a fully-refinable skeleton needs all 8 steps to pass, so even a correct assignment fails if
  one step's motion planning is flaky (≈`p_step^8`). A productive collection therefore needs a larger
  K_max **and** more `num_sampling_attempts_per_step` (per-step retries) — both increase cost.
- **Robustness fixes made along the way** (each keeps Stage-0 green): a well-spaced floor grid
  (`generator._floor_spots`) so a tall block never spawns close enough to block a cube's grasp; base
  nav ignores floor movables (consistent with `check_base_collisions=False`, so the wide base is not
  boxed in by staging); cosmetic shelf walls made visual-only (they were spuriously blocking
  off-centre placements); the *place* reach-in / lift excludes floor movables
  (`_place_reach_collision_ids`) but keeps boards (F3) + shelf residents (F2).
- **Standalone solvability diagnostics are unreliable** — a scratch rollout that did
  `set_state(cur); sim.step(controller.step())` lets the controller's internal `set_state` overwrite,
  stepping from the wrong state; the correct order (used by the refiner and the Stage-0 script) is
  `u = step()` then `set_state(cur); sim.step(u)`. A second scratch bug: a *fresh* rng per attempt with
  the controllers' fixed MP seed (0) gives no param variety and thus no feasible variation — the
  shared *advancing* rng the Stage-0 gate uses is what varies feasibility. Trust the refiner /
  Stage-0 harness, not ad-hoc scratch loops.
- **r0/r1 ARE solvable, confirmed two ways** (2026-08-14, `restock3d_demos.py` + `restock3d_probe.py`).
  Demos storing every object into a feasible region with 18 retries/store: **r0 4/5 seeds fully solved
  (3/3 cubes)**, **r1 3/5 seeds fully solved (5/5 cubes)** — the partials are one store exhausting its
  retries (the ~1/6 placement flakiness). The FP probe (short-circuit at the first refinable skeleton,
  15 retries/step): **r0 baseline FP = 2, r1 = 6** (oracle FP = 0) — the naive hff order tries that
  many over-assigning F2 skeletons before the first feasible one, and FP grows with the stratum. All
  failures F2 (r0/r1 have no blocks, so no F3). So the earlier `first_success=None` was purely the
  under-retried (`num_sampling=3`) collection config, not a mechanism gap.

**Takeaway / next.** The env, controllers, refiner, and pipeline wiring are complete; the Stage-0
mechanism gate is approved and stays green through all the robustness fixes; and the intended F2/F3
evidence is produced. The remaining work to a *productive* collection (one with `first_success`
values, i.e. a measurable baseline↔oracle FP gap) is **collection tuning + throughput**, not a
mechanism gap: sweep K_max up until a feasible skeleton lands in-pool, raise
`num_sampling_attempts_per_step` so an 8-step skeleton's per-step flakiness doesn't sink it, and run
worker-parallel — each feasible candidate is ~120 s of real 3D MP. This is **deferred** (as the plan
scoped full-scale collection) and is best paired with the training run. F1/coverage-waste, training,
and learned baselines remain deferred.

---

<a id="2026-08-14-restock3d-env-built-baseline-oracle-fp-gap"></a>
## 2026-08-14 — Restock3D env built; baseline-oracle FP gap validated across strata r0-r3

<!--strip-->
> **id** `2026-08-14-restock3d-env-built-baseline-oracle-fp-gap` · **status** active ·
> **tracks** method, evaluation, env-restock3d
<!--/strip-->

**What.** Built the Restock3D MuJoCo env (design + rationale in the ADR
[2026-08-14](../decisions/07-stickbutton2d.md#2026-08-14-restock3d-third-environment-mujoco-direct-env-geometric))
and measured the baseline-planner difficulty (`experiments/spectre/restock3d_difficulty.py`): for
each stratum, enumerate the astar/hff skeleton pool (cap 200), classify every candidate with the
geometric feasibility gate (`refine.evaluate_skeleton`), and report the baseline FP = position of
the first feasible candidate in the default order (oracle FP = 0). 6 seeds/stratum, uncensored at
the pool cap. Strata are `(n_small, n_tall, n_tall_regions, n_short_regions)`.

**Result.** The gap grows with stratum and the F2/F3 mix matches the recipe (8 seeds/stratum;
mean ± sd of the baseline FP; oracle FP = 0):

| stratum | recipe | σt | σs | solve | mean FP ± sd | feasible/200 | F2 / F3 |
|---|---|---|---|---|---|---|---|
| r0 | (3,0,2,5) | 2 | 4 | 8/8 | 0.2 ± 0.5 | ~126 | 74 / 0 |
| r1 | (5,0,1,4) | 1 | 0 | 8/8 | 14.9 ± 19.5 | ~8 | 192 / 0 |
| r2 | (3,1,2,4) | 1 | 2 | 8/8 | 12.2 ± 20.9 | ~21 | 65 / 114 |
| r3 | (4,2,3,5) | 1 | 2 | 5/8 | 123.0 ± 69.5 | ~1 | 82 / 116 |

r1 is pure F2 (short-cell over-assignment), r2/r3 add F3 (tall-in-short height). Oracle FP=0 every
stratum, so the astar↔oracle gap is the whole mean-FP column: **r0 (~0) ≪ r1/r2 (~12–15) ≪ r3
(~123)** — r2 clears the ~10 "earns its slot" bar. A smoke collection (`restock3d_collect.py`,
2/stratum) then produced valid
`EpisodeRecord`s; they load and `FailureRecord`s parse through the env-agnostic path — F2 records
name culprits (e.g. `cube_goal3`, `block_goal1`) with `proves_failure=True`, F3 records are
culprit-free + `exhausted` + `proves_failure=True`; vocab builds. The MuJoCo demo (physics pick +
geometric place) reaches the goal.

**Takeaway / next.** Env works and earns its slot on F2+F3. Two things to quote with care: (1) r3
is a **hard tail** — both cells tight → low feasible density (~1/200), so only ~4/6 raw-solvable;
the collector's reject-resample keeps only solvable problems, so this costs collection effort, not
validity. (2) The FP is **very noisy per-problem** — sd ≈ or > the mean (r1 14.9 ± 19.5, r2 12.2 ± 20.9),
because the height-/capacity-blind default order mixes feasible/infeasible near-uniformly (all same
length) so a few high-FP draws dominate. The robust claim is the **gap tier** (r0 ≈0 ≪ r1/r2 ≈12–15
≪ r3 ≈123, oracle 0), not a clean per-stratum ordering — **r1 vs r2 do not separate at 8 seeds**;
stable per-stratum means need the full ~100-problem-per-stratum collection. Deferred (not this pass): F1 grasp
obstruction + coverage/waste discretionary steps, learned baselines, the `compare_envs.py`
`EnvSpec`, multi-slot region capacity.

---

<a id="2026-08-13-shelfobstruct3d-fp-0-shelf-fully-reachable"></a>
## 2026-08-13 — ShelfObstruct3D has no FP>0: shelf fully reachable + obstruction inert; not a re-ranking testbed

<!--strip-->
> **id** `2026-08-13-shelfobstruct3d-fp-0-shelf-fully-reachable` · **status** active ·
> **tracks** method, env-shelf3d, evaluation
<!--/strip-->

**What.** With class-1 obstruction shown physically inert
([earlier today](#2026-08-13-m2-certifying-generator-built-shelf-cube-width)), the fallback FP
source for a class-2 use of ShelfObstruct3D was **reachability** — placements outside the arm's
range fail as class-2 deviations. Mapped the reachable window by refining `pick_blocker →
place(blocker, free_region)` with the free region swept in lateral y and in depth (ly → world-x).

**Result.**
- **Lateral:** place-relocate REACHABLE at y = 0.10 / 0.18 / 0.24; FAILS at 0.30 — but 0.30 is
  *off the shelf* (usable half-width ≈ 0.27), not an arm limit. The whole shelf width is reachable.
- **Depth:** REACHABLE at ly = 0.10 / 0.06 / 0.04 / 0.02, i.e. world-x = 1.40 / 1.44 / 1.46 / 1.48
  — the full usable depth. (The arm reaches deeper than the ~1.42 I'd inferred from the grasp; the
  place base repositions closer.) No unreachable interior region exists.
- **Grasp-obstruction** (an adjacent blocker fouling the lateral finger path) needs two blockers
  within ~0.10 m to obstruct yet ≥ ~0.10 m apart to spawn stably — a ~0.003 m window, unusable.

**Takeaway.** **ShelfObstruct3D has no robust FP>0 mechanism** — the small shelf is *fully
reachable* (lateral + depth), the obstruction is *inert*, and grasp-obstruction is *unhittable*.
So refinement always succeeds on the first candidate (FP=0): the env induces **no re-ranking
difficulty at all**, for class-1 *or* class-2. This supersedes the "leans class-2" implication of
the [class-1 ADR](../decisions/07-stickbutton2d.md#2026-08-13-shelfobstruct3d-class-1-culprits-physically-infeasible-certifying):
the class-2 route is not viable here either, because there is nothing to re-rank. It is the same
FP≡0 that made *vanilla* Shelf3D o1/o2 unusable — the obstruction redesign did not fix it, because
the substrate's small fully-reachable shelf and squeeze-past placement physics defeat every
obstruction/reachability knob. **Recommendation: ShelfObstruct3D (this design) is a dead end for a
SPECTRE re-ranking gap; route the effort to a 2D DD2D-like obstruction env (robust top-down
culprits), or a fundamentally different 3D substrate (a much larger/deeper shelf that is not fully
reachable, or a rigid non-squeeze obstruction).** Kept, all CI-clean & reusable: the M1 obstruction
env + custom shelf grasp (a genuine new skill), the instrumented refiner (both channels), the
certifying generator, and unit tests. Single-scene diagnostics, not results.

---

<a id="2026-08-13-m2-certifying-generator-built-shelf-cube-width"></a>
## 2026-08-13 — M2 certifying generator built; shelf cube-width limit makes obstruction inert

<!--strip-->
> **id** `2026-08-13-m2-certifying-generator-built-shelf-cube-width` · **status**
> active · **tracks** method, env-shelf3d, evaluation
<!--/strip-->

**What.** Built the **M2 certifying generator** (`envs/shelf3d/generator.py`) to reliably land
class-1 obstructions in the culprit band (the direction chosen after the delicacy finding). It
lays a parametric row of target + free regions on shelf 2, places obstructor cubes in the band of
chosen free regions, and a fast **geometric certification** (via the instrumented refiner's
`_place_culprits`) accepts a seed only if an obstructed free region reads `Clear` yet is flagged,
a clear free region exists, and every blocker sits At its target. Then verified an obstructed
candidate by refinement.

**Result.**
- **Generator + geometric certification work.** A level-1 seed certified: `free_region_1` Clear +
  culprit `cube_obstructor1`, `free_region_2` clear, blocker At target → `CERTIFIED=True`. The cert
  is robust to spawn variance — an obstructor that fell to the shelf below was correctly *not*
  flagged (a new **Z-proximity guard** in `_place_culprits` rejects cross-shelf overlaps).
- **But the obstruction is physically inert** — the certified obstructed candidate refined to
  **SUCCESS**. Two measured facts explain it:
  - **Shelf holds cubes ≤ 0.07 m wide only.** Single cube, deep, no overhang: half-extent
    0.045/0.055 (0.09/0.11 m) rests at z **0.328/0.338** (fell to shelf below) vs 0.585 for 0.035.
  - **So Clear-but-blocking overlap ≤ ~0.03 m** (collision 0.07 − At-radius 0.05), and the
    placement squeezes past it rather than failing. A larger obstructor would block but can't stay
    on the shelf.
- Also hit a **lateral grasp-range limit**: a 4-region row (span 0.54) put the blocker at y=−0.259
  where the shelf grasp fails (o1's worked at −0.14); tightened the pitch to 0.15 and cut to 3
  regions to keep the row in ~[−0.18, 0.20].

**Takeaway / next.** Conclusive: **ShelfObstruct3D cannot robustly produce class-1 collision
culprits** — a fundamental geometric/physical limit (thin shelf + bulky gripper + At-radius vs
collision distance), unlike DD2D's 2D top-down obstruction. So it **leans class-2 like SB2D**, and
the `coverage`/`waste` payoff (the reason for the harder env) is not attainable here. ADR:
[decisions/07 2026-08-13](../decisions/07-stickbutton2d.md#2026-08-13-shelfobstruct3d-class-1-culprits-physically-infeasible-certifying).
Kept, all CI-clean: M1 Gate-0 env, instrumented refiner (both channels), certifying generator
(correct; obstruction inert). M3 sweep / M4 coverage/waste are **not worth running on
ShelfObstruct3D as-is** for the class-1 story — decision point for the user: class-2 use of
ShelfObstruct3D, or route class-1 to a 2D-obstruction env. Single-scene diagnostics, not results.

---

<a id="2026-08-13-shelfobstruct3d-instrumented-refiner-class-1-culprit-geometry"></a>
## 2026-08-13 — ShelfObstruct3D instrumented refiner + class-1 culprit geometry is delicate

<!--strip-->
> **id** `2026-08-13-shelfobstruct3d-instrumented-refiner-class-1-culprit-geometry` ·
> **status** active · **tracks** method, env-shelf3d, evaluation
<!--/strip-->

**What.** Built the ShelfObstruct3D **instrumented refiner** (`envs/shelf3d/instrumented_refiner.py`)
— the class-1 evidence that feeds `coverage`/`waste` — and tried to exercise it on a scene that
produces refinement *failures* (o1 has FP=0). The refiner is a `RecordingObstructionSampler` over
the stock `ParameterizedControllerTrajectorySampler`: it keeps every rejection (class-2 deviation
between predicted/achieved abstract state) and, for a failed `place(cube, region)`, runs a
geometric footprint-overlap check naming any other cube blocking the destination (class-1 culprit,
`deviation=None`) — DD2D's `grasp_blocker` idea. `failure_metadata` emits the
`refiner_metadata["failures"]` payload `unified_evidence` consumes.

**Result.**
- **Both channels verified.** Unit test: moving `cube_target1` onto `free_region_1`'s centre →
  `_place_culprits` returns `('cube_target1',)`; on the ground → `()`. Infra: refining o1 cand 0
  through the recording sampler SUCCEEDS and captured a real class-2 deviation during backtracking
  (a transient `pick_target` where the target ended `OnGround` instead of `Holding` — `dev_added
  [OnGround(cube_target1)]`, `dev_deleted [Holding(...)]`). CI-clean (black/isort/mypy green).
- **The key geometric finding.** A clean class-1 obstruction failure needs a cube *farther than the
  abstractor's At-radius from a region centre yet still overlapping a placed cube's footprint*. With
  the At-radius above the cube collision distance (the original `_AT_XY_TOL=0.12` > 2·half-width
  ~0.07) that band is **empty** — any cube close enough to block a placement is assigned `At` the
  region, so the region reads not-`Clear` and the planner never attempts the obstructed place.
  Lowering `_AT_XY_TOL` to **0.05** (o1 still refines end-to-end; placements land within ~0.005)
  opens the band, but it is **narrow (~0.05–0.085 m) and spawn-variance-sensitive**: a clutter cube
  aimed into it landed `At free_region_1` instead (abstraction correctly saw the region occupied →
  planner used the other free region → FP=0). Grasp-obstruction (adjacent blocker fouling the
  lateral finger path) is similarly narrow — the window where two blockers both spawn stably *and*
  obstruct each other is ~0.08–0.10 m, and a 2-blocker attempt had the second blocker fall to the
  shelf below on spawn.

**Takeaway / next.** ShelfObstruct3D's 3D geometry (reachable front-band only, bulky gripper,
At-radius vs cube size, thin-shelf spawn stability) makes **robust class-1 collision culprits hard**,
unlike DD2D's top-down obstruction — so as built the env **leans class-2 like SB2D**, where the
representation/adaptivity advantage did not reproduce. FP>0 is still reachable via reachability /
ordering, but those failures carry no blame (means-failures), so they don't feed `coverage`/`waste`.
The instrumented refiner is done and correct; **the open work is a scene *generator* (M2) that
reliably lands obstructions in the band** (or a redesign that widens it — e.g. larger cubes + a
smaller At-radius + certified per-seed placement), before the M3 sweep and M4 coverage/waste
verification are worth running. This is a decision point on whether ShelfObstruct3D is the right
class-1 testbed or whether the effort should route to a 2D-obstruction env with DD2D-like geometry.
Numbers here are single-scene diagnostics, not results.

---

<a id="2026-08-12-shelfobstruct3d-o1-gate-0-clearing-plan-refines"></a>
## 2026-08-12 — ShelfObstruct3D-o1 Gate 0: clearing plan refines end-to-end

<!--strip-->
> **id** `2026-08-12-shelfobstruct3d-o1-gate-0-clearing-plan-refines` · **status**
> active · **tracks** method, env-shelf3d, evaluation
<!--/strip-->

**What.** Built the ShelfObstruct3D obstruction variant (spectre-local `envs/shelf3d/`) and
ran the M1 Gate-0 de-risk on the hand-authored `ShelfObstruct3D-o1` scene (1 blocker on shelf 2
obstructing `target_region_1`, 1 target on the ground, 2 free regions). Pieces: custom
`PickFromShelfController` (grasp off shelf) + `PlaceToShelfRegionController` (place at a region
centre), `region_geometry.py` (symbolic region → world centre), `models.py` (At/Clear abstractor
+ clear-then-place operators + skills). Design decisions in
[decisions/07 2026-08-12](../decisions/07-stickbutton2d.md#2026-08-12-shelfobstruct3d-obstruction-env-custom-shelf-grasp).

**Result.**
- **Custom shelf grasp is reliable:** `reset_ok 8/8, lifted 8/8` across seeds (gym env). Pick+place
  relocation demonstrated end-to-end (blocker moved target_region_1 → free_region_1).
- **Abstractor correct:** initial atoms `At(cube_blocker1,target_region_1)`,
  `Clear(free_region_1)`, `Clear(free_region_2)`, `HandEmpty`, `OnGround(cube_target1)`; goal
  `At(cube_target1,target_region_1)`. Symbolic pool = 20 candidates; cand 0 = pick_blocker →
  place(free_region_1) → pick_target → place(target_region_1).
- **Full clearing plan refines end-to-end:** all four steps give `abstract match=True`
  (manual step-through), and the real `BacktrackingRefiner` returns **success on cands 0 and 1**
  (6.4 s / 15.4 s).
- **Video:** `envs/shelf3d/demo_o1_clearing.mp4` (live closed-loop execution). Region markers
  (target yellow, free cyan) render as translucent boxes.
- **Two substrate traps found & fixed:** (1) the `ObjectCentricTidyBot3DEnv` `set_state`-per-step
  rollout drops the shelf cube (~80% nondet) — refinement now rolls out on the gym `TidyBot3DEnv`
  with continuous stepping; (2) open-loop action replay diverges (grasp weak, target not placed) —
  videos must be rendered by live closed-loop execution, not recorded-action replay.

**Takeaway / next.** The obstruction **mechanism works** — clear-then-place refines and the
abstract states match, so the hard de-risk (custom shelf grasp + relocation + planning) is done.
o1 FP is 0 (two reachable free regions; first candidate succeeds), so the FP magnitude and the
class-1 **culprit** signal (which feeds `coverage`/`waste`, the whole point for SPECTRE) are the
next milestone: wire the instrumented refiner to capture the collided object on a failed sample,
then the per-seed generator + strata (M2) and the difficulty sweep to hit ~50–100 FP at ≥80%
solve on 3 targets (M3), then coverage/waste verification (M4). Numbers here are 1 seed / single
scene — not paper numbers.

---

<a id="2026-08-12-shelf3d-difficulty-under-baseline-planner-5-seed"></a>
## 2026-08-12 — Shelf3D difficulty under the baseline planner; the 5-seed pilot under-samples the refinement tail

<!--strip-->
> **id** `2026-08-12-shelf3d-difficulty-under-baseline-planner-5-seed` · **status**
> active · **tracks** evaluation, method, env-shelf3d
<!--/strip-->

**What.** Built a standalone Shelf3D difficulty harness — `experiments/spectre/shelf3d_collect.py`
(collector) + `experiments/spectre/shelf3d_difficulty.py` (marimo) — to quantify how hard vanilla
**dynamic3d / MuJoCo TidyBot** `kinder/Shelf3D-o{1,2,8}` is for the SPECTRE baseline astar planner,
before the env config is modified into harder variants. Per problem (= variant × seed) it enumerates
the astar skeleton pool and refines candidates one at a time (`BacktrackingRefiner`, `samples/step=20`,
`horizon=500`) under a per-plan-attempt wall-clock budget, recording each attempt's success/fail +
time. Two phases: a **5-seed × 15-plan pilot** at 30 s/attempt to pick the budget (offline cap
re-derivation — the refiner is monotone in the timeout), then a **20-seed full** collection
(short-circuit) with faithful `set_state`-rendered videos of the successful plans. Env-integration
workarounds (self-registration, egl, ikfast BLAS-symlink) are in
[`decisions/07` 2026-08-12](../decisions/07-stickbutton2d.md#2026-08-12-shelf3d-difficulty-harness-standalone-collector-per-attempt-budget).

**Result.** Full run (20 seeds/variant; o1/o2 at the deployed **20 s/attempt**, o8 at 10 s — the
budget is irrelevant for o8, see below):

| variant | pool | solve | FP | wall-to-first-success (s) | successful-refine range (s) |
|---|---|---|---|---|---|
| **o1** | 1 | **100 % (20/20)** | 0.00 ± 0.00 | 1.96 ± 2.47 | 1.2 – 12.4 |
| **o2** | 2 | **85 % (17/20)** | 0.00 ± 0.00 | 5.09 ± 3.75 | 3.2 – 19.4 |
| **o8** | 15–100 | **0 % (0/20)** | — | — | — |

- **o1/o2 are trivially solvable by the first pooled skeleton** (FP ≡ 0). **o8 is unsolvable**: every
  candidate raises `ValueError: No valid parameters found` from `PlaceShelfController.sample_parameters`
  (`kinder_models/dynamic3d/shelf/parameterized_skills.py`), which rejects any shelf placement within a
  2-D-overhead collision threshold of *any other cube* and exhausts `MAX_SAMPLER_ATTEMPTS` — the 8 floor
  cubes flood that plane. It fails in ~3 s, well under any budget, so o8 is 0 % at **every** budget.
- **o2's three misses are `AssertionError: Motion planning failed`** (the pybullet arm motion planner),
  ~3–4 s each — *not* budget-truncated — and they are **nondeterministic run-to-run**: the 10 s run
  missed seed 11; the 20 s run missed seeds 1/2/3. The exact o2 solve count carries ≈ ±2/20 of variance.

**Takeaway.** (1) **o8 already saturates the vanilla place skill** at 0 % on 8-cube clutter — it is the
natural hard end; harder-variant work should build from o1/o2. (2) **The 5-seed pilot badly
under-sampled the refinement-time tail.** Pilot o2 max was 5.32 s → it read "≥ 8 s sufficient, 30 s
overkill"; but the 20-seed full run's tail reaches **12.4 s** (o1 seed 8 — genuinely budget-truncated at
10 s, solves at 20 s) and **19.45 s** (o2), so **≈ 20–30 s is actually needed and the original 30 s
eyeball was *not* overkill.** The standing "a small-seed report is least trustworthy in the tail"
lesson, concretely: size a budget against the full-sample tail, never a 5-seed max. (3) **Shelf3D
refinement is nondeterministic run-to-run** (MuJoCo/pybullet, not the sampler seed) — like the DD2D
`PYTHONHASHSEED` caveat, solve counts carry run-to-run variance on marginal instances.

---

<a id="2026-08-12-publication-de-versioning-refactor"></a>
## 2026-08-12 — Publication de-versioning refactor

<!--strip-->
> **id** `2026-08-12-publication-de-versioning-refactor` · **status** active ·
> **tracks** process, method, evaluation
<!--/strip-->

**What.** The publication de-versioning refactor (branch `spectre-refactor`): collapsed
v1/v2/v2.2/v3 into one unversioned SPECTRE (`model.py`/`dataset.py`/`train.py`/`inference.py`
+ shared `layers.py`/`encoders.py`), moved the baselines under
`baselines/{vlmplan,piginet,lazy,drake-tamp}/`, flattened `envs/dd2d/dd2d/` →
`envs/dd2d/drawer/`, archived RT2D + TTD, and removed the built-then-disabled features
(proof-demotion, legacy-coverage, obj-evidence, sinusoidal positions, `tail_max_f`,
necessity — all OFF in the deployed recipe, so removal is behaviour-preserving), keeping EMA
and the ablation flags one flag away. Docs were cleaned in the same pass (superseded specs →
`docs/archive/`, `as_built_v3.md` → `as_built.md`). Full decision + judgment calls:
[`decisions/07` 2026-08-12](../decisions/07-stickbutton2d.md#2026-08-12-publication-de-versioning-one-unified-spectre).

**Result.** The test suite shrank as archived-only, with **zero failures at every gate**:
**558 (pre) → 490** (v1/v2 tests archived) **→ 371** (RT2D/TTD archived) **→ 362**
(removed-feature tests dropped) — the drops are archived tests, not regressions.
Deployed-path equivalence held throughout: `checkpoints_v3_unified` loads `strict=True` at
**324311 params** and `deployed_rollout_traced` reproduces the cached `spectre3_adaptive` FP
on sampled episodes. Retrain-verification of the headline numbers (refactored code, 3 seeds/env,
deployed recipe): **DD2D SPECTRE-adaptive 6.29 ± 0.31** vs frozen 5.92 ± 0.29 (val-selection
`best_val_fp` 7.78/7.93/7.80 vs the original's ~7.76); **SB2D 1.75 ± 0.19** vs ~1.84 (SPECTRE ≈
LAZY 1.85 non-separation holds). Ordering preserved (DD2D ≪ PIGINet 17.27 ≪ LAZY 23.26); PIGINet/
LAZY not retrained (only relocated). Both within seed variance — the retrain delta is fresh-run
GPU non-determinism, not a refactor artifact.

**Takeaway.** A behaviour-preserving structural cleanup for external readers; the "v3" naming
that persists across the older docs and append-only logs is now historical and denotes the
current SPECTRE. Retrain-verification landed (above; reproduces within seed variance). Remaining
are cosmetic follow-ups only: the internal `train_v3()` / `SpectreV3Dataset` names and `AuxHead`
(constructed unconditionally → in the deployed state_dict, so it cannot be dropped without a
retrain).

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

