# PIGINet on DD2D — Staged Implementation Plan

**Goal.** Re-implement a simplified PIGINet (Yang et al. 2023, plan-feasibility Transformer) and run its
headline experiment on **DD2D** (drawer decluttering 2D) — an environment PIGINet was never evaluated on.

**Research question.** Does PIGINet cut the number of infeasible refinements DD2D planning wastes before the
first success, and **how much headroom remains between PIGINet and a clairvoyant Oracle** — i.e. is PIGINet
already "good enough" on DD2D, or is there room for a better predictor?

**Headline metric = rollout FP count (PIGINet Fig. 7b).** The single most important number is, per problem,
the **number of failed refinements before the first successful plan**, compared across Baseline / PIGI /
Oracle. This is a *rollout* metric (refine plans in the ranker's order, stop at first success, count the
infeasible ones along the way), not a static classification score. AUPRC/AUROC/accuracy are **secondary**
diagnostics that explain the rollout result; refinement *time* is a secondary rollout metric.

The original PIGINet model code (`fastamp`) is unavailable, so we re-implement it. A detailed prior spec for
the sorting env lives at `archive/.ignore_old_piginet.md`; its model/training/eval architecture is ~90%
reusable and is referenced throughout. This document adapts it to DD2D and stages the work so **each step is
independently verifiable**.

---

## How this plan is executed — step-by-step, you orchestrate

Steps **0–8**, each with **Deliverables** and a **✅ Verification gate** (unit tests, a smoke run, or a
measured go/no-go). Project invariant:

> **Do not start step N+1 until step N's gate is green.** Every step adds tests or a checkable artifact;
> errors must surface at their own step, not bleed downstream.

**Orchestration model:** this is **not** run in one shot. Each step is planned + implemented on its own (its
own plan-mode pass) against already-verified predecessors, then control returns to you to inspect and
intervene before the next step is released.

---

## Locked decisions

| Decision | Choice |
|---|---|
| Object modality | **Full multimodal**: CLIP image crops + pose/shape value tokens + text + init |
| Compute | **Local for everything**: multiprocessing collection + **MPS** training on the M3 (36 GB) |
| Subset mix | **Balanced strata** across `min_feasible_subset ∈ {1,2,3}` (equal *problem* counts) |
| Splits | **400 train / 100 test / 100 val**, disjoint seed bands |
| Object holdout | **None now** — always `split="train"` = the **full 7-family library at nominal sizes** (verified `shapes.py:sample_shape`; train applies no swap and `shift=1.0`, so all objects are seen). Unseen-object holdout deferred |
| Primary metric | **Rollout FP count** (Fig 7b), per stratum, Baseline/PIGI/Oracle. AUPRC = secondary |

**Locked collection config.** blockers 9–12 (⇒ `n_items ∈ {10,11,12,13}`, sampled per problem);
`lambda=0.8`; `margin=1.0`; up to **200 plans** per problem refined in order **stopping at first success**;
`crowd=5`; `diverse_crowd=True`; per-problem seed; **astar planner + distance heuristic**
(`prefer="pyperplan", search="astar", heuristic="dist"`); **uncapped** stream-call budget; `retry_cap=10`;
`samples_per_step=15`; `time_budget=20.0`s; parallel workers. **Do not** use `demo.py --split`.

> **Two gotchas in every step touching the refiner/generator:**
> 1. `DD2DRefiner(budget=None)` silently reverts to 300 unless `time_budget` is set → always pass
>    `budget=None, time_budget=20.0` together (uncapped stream calls, 20 s wall).
> 2. `generate_dd2d_problem(split=...)` is the **shape-family** split, *not* the dataset split → always
>    `split="train"`; the dataset train/test/val split is the collector's seed bands.

---

## Class imbalance — the central modeling challenge

Stop-at-first-success keeps **1 positive + (rank−1) negatives** per problem. With astar+dist the median
attempts-to-first-refinement is **4.5 / 19 / 128** for min-subset **1 / 2 / 3** ⇒ ≈ **3.5 / 18 / 127
negatives per positive** per stratum. With balanced *problem* counts (~133 each) the train set is ~**19.7k
negatives vs ~400 positives → ≈49:1 overall**, and negatives are dominated by s=3 (one s=3 problem yields
~36× the negatives of an s=1 problem). Two skews: **class imbalance** (neg≫pos) and **example-source
imbalance** (a few hard problems swamp the loss). Handling this is a first-class objective (Step 7), and the
headline rollout FP metric (Step 8) is deliberately chosen because it is **per-problem and imbalance-robust**.

---

## Baseline facts (verified in the codebase)

- `blocks_tamp/collect.py` is **sorting-only** → DD2D needs its own collector; do not retrofit it.
- DD2D records carry **no geometry**: `init_literals` ≈ constant, `objects[].size` is bbox `(w,h,6.0)`, image `path=null`. **Signal must come from a geometric channel we add** (Step 1).
- No PIGINet model/training/eval code exists (greenfield).
- `blocks_tamp/dd2d/heuristic_experiment.py` provides reusable infra: `_stable_seed` (deterministic per-skeleton refiner seed → bit-for-bit label replay), a **lazy per-problem refine memo** (one refine per distinct plan), and a `ProcessPoolExecutor` coordinator. Reuse in Steps 3, 4, 8.
- Eval re-ranking seam: `DD2DPlanner._reorder`'s `order="oracle"` sorts by ground-truth `meta["label"]`; a learned scorer is a new ordering.
- `generate_dd2d_problem(...) -> DD2DProblem`; `problem.min_feasible_subset` = stratum; `problem.num_blockers = n_items-1`.
- `make_dd2d_planner(prefer="pyperplan", search="astar", heuristic="dist").plan(problem, k)` = astar+dist pool; `DD2DRefiner(...).refine(sk, scene, seed) -> RefineResult` (`.feasible`, `.n_attempts`, `.failure_action`, `.elapsed`).
- `record.build_example(...)` is generic; `objects` is free-form `list[dict]`, so geometry adds without a schema change.

---

## Dependencies & code layout

`blocks_tamp/requirements-piginet.txt` (into `.venv`, separate so the collector stays torch-free): `torch`
(MPS), `open_clip_torch` (frozen CLIP ViT-B/32), `scikit-learn` (AUPRC/AUROC), `pandas`.

```
blocks_tamp/
  dd2d/
    record_ext.py       # Step 1 — geometry sidecar: pose/shape into objects[], crop PNGs, at-pose init facts
    collect.py          # Steps 2–3 — DD2D stop-at-first-success collector, balanced strata, parallel
  piginet/              # Steps 6–8 — model + training + eval subpackage
    glosses.py  encoders.py  tokenize.py  model.py  dataset.py  losses.py  train.py  eval.py
blocks_tamp/tests/
    test_dd2d_record_ext.py  test_dd2d_collect.py
    test_piginet_encoders.py  test_piginet_tokenize.py  test_piginet_model.py
    test_piginet_losses.py    test_piginet_eval.py
```

Run tests: `.venv/bin/python -m pytest blocks_tamp/tests/<file> -q`.

---

## Step 0 — Environment & MPS bring-up

**Deliverables.** `requirements-piginet.txt`; install into `.venv`; `scripts/check_mps.py` importing torch +
open_clip, reporting device, running a tiny MPS op + a frozen CLIP ViT-B/32 text+image embed.

**✅ Gate.** `torch.backends.mps.is_available()` → `True`; the script embeds a dummy image + text through
frozen CLIP on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`) → finite 512-d vectors, no crash.

---

## Step 1 — Geometry-extended DD2D records (`record_ext.py`)

**Deliverables.**
- `build_dd2d_example(problem, skeleton, refine_result, images_dir, label_source, extra_provenance)` — wraps
  `record.build_example`; augments each `objects[]` dict with `pose:[x,y,theta]` + `shape:{family,w,h,area,concave}`
  from `problem.scene.items[name]`; appends `["at-pose", name, x, y, theta]` facts to `init_literals`.
- `write_crops(problem, images_dir, views=("topdown",))` — `render_scene(problem.scene)` once per problem,
  crop each object by seg `bbox`, write `<images_dir>/<object>__<view>.png`, return `ImageRef`s with `path`.

**✅ Gate — `test_dd2d_record_ext.py`.** Every object dict has 3-float `pose` + `shape`; #`at-pose` facts ==
#objects; one non-empty PNG per (object,view) with dims == bbox extent; `from_json(to_json())` round-trips.

---

## Step 2 — Collector core: one problem (`collect.py::collect_problem`)

**Deliverables.** `collect_problem(seed, stratum, planner=None, refine_fn=None) -> ProblemResult` (injection
seams for testing, à la `collect.py`):
1. `n_items` from `seed` → `{10,11,12,13}`.
2. `generate_dd2d_problem(lam=0.8, seed, margin=1.0, split="train", n_items, crowd=5, diverse_crowd=True,
   require_subset=(stratum>=2), min_subset=stratum, certify=True, budget=None, retry_cap=10,
   samples_per_step=15, time_budget=20.0)`; **accept only if `min_feasible_subset == stratum`**.
3. `planner.plan(problem, 200)` (astar+dist).
4. Refine **in order, stop at first feasible**, seed via `_stable_seed(sk.key())`.
5. Solved ⇒ **1 positive + only the preceding negatives** (rest dropped, unrefined) via `build_dd2d_example`;
   unsolved-in-200 ⇒ **drop**. Return counts + `first_feasible_rank` + `reason`.

**✅ Gate — `test_dd2d_collect.py` (injected fakes).** Stop-at-first (plan#3 feasible ⇒ 3 refined, #4 never
touched, labels `[F,F,T]`); drop-unsolvable ⇒ no examples; exact-stratum rejection; determinism (same
`(seed,stratum)` ⇒ identical serialized examples).

---

## Step 3 — Coordinator: parallel, balanced strata, splits (`collect.py::collect_split` + CLI)

**Deliverables.** Three splits, **disjoint seed bands** (width 1e6): train=[0,1e6) target 400, test=[1e6,2e6)
100, val=[2e6,3e6) 100; per-split **per-stratum sub-targets** (train ≈134/133/133; test/val ≈34/33/33).
`ProcessPoolExecutor` hands out `(seed,stratum)` round-robin across still-open strata, stops a stratum at its
sub-target, drains in flight. Output `<out_root>/<split>/<problem_id>/{NNN.json}` + `images/`, atomic. Per
split `manifest.json`: kept/attempted per stratum, **neg:pos ratio (per stratum + overall)**, seeds, drop
reasons, budget. CLI `python -m blocks_tamp.dd2d.collect --workers 8 --target-train 400 --target-test 100
--target-val 100 --out-root data/dd2d/raw` (+ `--calibrate`).

**✅ Gate — `test_dd2d_collect.py` (coordinator, monkeypatched task).** Stops each stratum at sub-target;
writes only kept dirs; manifest counts match on-disk records; seed bands disjoint. **Real micro-smoke**
(`--target-train 3 --target-test 2 --target-val 2 --workers 2`) writes valid JSON + crops; manifest
self-consistent.

---

## Step 4 — EDA / calibration gate ✅ SKIPPED (satisfied by existing `heuristic_experiment`)

**Status (2026-07-09).** Not implemented as a separate `--calibrate` run — it is **redundant** with
`blocks_tamp/dd2d/heuristic_experiment.py`, whose `astar-dist` arm uses the identical planner + `DD2DRefiner`
+ `_stable_seed` + lazy first-feasible-rank as the collector's stop-at-first (so its per-stratum solve-rate +
first-feasible-rank *are* the Step-4 measurement; neg:pos = rank − 1). See `decisions.md` 2026-07-09.

**Go/no-go = GO**, from `out_dd2d/heuristic_experiment/results_minsubset*.csv` (astar-dist, exact
`min_feasible_subset`):

| stratum | solve% within k=200 | mean first-feasible rank | ⇒ neg:pos |
|---|---|---|---|
| 1 | 100% | 3.2 | ~2 |
| 2 | 100% | 20.1 | ~19 |
| 3 | 71% | 147 | ~146 |

≈**56:1** overall (balanced). Signal strongly present (all mean ranks ≥ 2), strata fillable (s3 71% ⇒ ~1.4×
attempts), imbalance quantified → feeds Step 7. Those runs used `time_budget=10, budget=500, n_items=11`; the
locked config (`time_budget=20`, uncapped, `n_items` 10–13) is marginally **easier**, so these are
conservative. **Exact** per-stratum neg:pos is recorded for free by Step-5 manifests. Carried-forward flag:
**s=3 is the expensive long pole** (median ~155 refines/problem) — a Step-5 budgeting note, not a blocker.

---

## Step 5 — Full dataset collection

**Deliverables.** Overnight, 8+ workers → `data/dd2d/raw/{train,test,val}/…` + manifests, balanced strata.
**Persist all negatives** (faithful to Yang; the refinement cost to reach the first success was already
paid). Imbalance is handled at train time (Step 7), not by discarding data.

**✅ Gate.** Targets met, per-stratum counts ~on target, manifests written; non-degenerate train neg:pos
recorded; **replay-determinism spot-check** (re-collect ~5 seeds → byte-identical records); every non-null
crop `path` exists.

---

## Step 6 — Encoders + tokenizer (`piginet/glosses.py, encoders.py, tokenize.py`)

**Deliverables** (prior spec §5; DD2D specifics).
- `glosses.py` — DD2D vocab → NL glosses (`handempty/in-drawer/target/extracted/at-pose`, `pick/place-buffer/
  retrieve`, `target/item`, `tomato/slateblue(concave)/silver`, `drawer/buffer`).
- `g_text` — frozen CLIP text + `MLP(512→d)+ReLU`, cached per word.
- `g_val` — types `T={pose, shape}`; pose `(x,y,θ)` normalized by drawer bbox / `[-π,π]`; shape
  `(w,h,area,concave)` normalized; `one-hot(type) ⊕ padded value → MLP(→d)+ReLU`.
- `g_img` — frozen CLIP image over per-object crops → `MLP(C·512→d)`, cached per (problem,object,view).
- `g_obj = MLP([g_img(o); g_val(pose(o))])` behind an interface (image-only/geom-only ablations = config flip).
- `tokenize.py::h(z)=mean(encoders)+PE`: plan tokens `[op_text,*args]` sinusoidal PE + **causal plan mask**;
  goal/init learned `pe_G`/`pe_I`; `at-pose` → `[pred_text,g_obj,g_val(pose)]`; **init-dropout** to `n_max`
  (default 64; later sized to train p95).

**✅ Gate — `test_piginet_encoders.py`, `test_piginet_tokenize.py`.** Encoders return `d`-dim; caches return
identical repeats (computed once); **same-type objects at different poses → distinct `g_obj`** (identity
binding, §5.6); seq length `=|π|+|G|+|I|`; plan block lower-triangular, rest ones; variable `|π|`/`|I|` run;
**init-dropout fires** past `n_max` and never drops plan/goal tokens.

---

## Step 7 — Model + training + imbalance handling (`piginet/model.py, losses.py, dataset.py, train.py`)

**Model.** 3 residual attention layers, `d=256`, 4–8 heads, causal-plan mask, position-0 → `Linear→logit`
(sigmoid at inference; logits at train).

**Handling class imbalance** (weighted-BCE = paper baseline; the rest are the committed modifications — final
selection among them is decided in this step's plan-mode pass, judged by the Step-7 gate):
- **Data (persist all, subsample at train time):** per problem cap plans to `positive + ≤K negatives`
  (default K≈16), sampling the few **nearest the success** (most confusable) + a random spread. Equalizes
  per-problem contribution, caps s=3 domination; reversible train-time hyperparameter. Log capped vs raw ratio.
- **Sampling:** problem-uniform batches with bounded plans/problem (or `WeightedRandomSampler` to a target
  batch pos:neg), so a few hard problems can't dominate a batch.
- **Loss (`losses.py`):** **focal loss** (γ≈2, α tuned) as primary — down-weights the many *easy* negatives
  (buried-blocker / obvious-overflow plans) — compared against weighted-BCE `pos_weight=N_neg/N_pos`. Plus an
  optional **per-problem listwise ranking loss** (feasible plan ranked above its problem's negatives), which
  matches deployment (PIGINet *ranks* within a problem) and is inherently imbalance-robust; usable as an
  auxiliary term or a standalone objective.
- **Calibration + threshold:** post-hoc **temperature scaling** on val; **tune the discard threshold on val
  to preserve positive recall (~0.98)** rather than hardcoding 0.5 — ranking drives the speedup, the
  threshold only avoids dropping feasible plans (the Step-8 parity requirement).
- **Metrics:** **AUPRC** (headline classification metric under imbalance) + AUROC + balanced accuracy +
  per-stratum breakdown; PR curve; ECE before/after calibration. **Not** raw accuracy.

**Training.** `dataset.py` reads records + crops + geometry; **precompute & cache** CLIP text/image
embeddings once; tokenize on the fly; collate = pad to batch-max + key-padding mask; image augmentation
before CLIP (sample among a few cached augmented embeddings). AdamW lr 1e-4 wd 1e-2, cosine + ~5% warmup,
batch 32–64, `device="mps"`. **Model selection aligned to the headline:** early-stop on val **AUPRC** (cheap
per-epoch proxy), then pick the final checkpoint among the top-k by **val rollout FP** (mean first-feasible-
rank via a small val replay — the deployment metric). Save checkpoint + `train_metrics.json`.

**✅ Gate.**
- `test_piginet_model.py`: forward → `(B,)` finite logits; masking wired (perturbing a masked token can't
  change position-0 output). `test_piginet_losses.py`: focal/weighted-BCE/ranking losses match hand-computed
  values on tiny inputs; ranking loss is minimized when the positive scores highest.
- **Tiny-overfit gate:** ~20 problems → train loss ≈ 0. If not, tokenizer/render/gloss bug — **fix before any
  full run**.
- **Imbalance gate (decisive):** the chosen strategy must beat the weighted-BCE baseline on **val rollout FP
  (mean first-feasible-rank)** *and* on val AUPRC, **and** achieve positive recall ≥ target at the tuned
  threshold (so PIGI won't drop solves in Step 8). A strategy that improves AUPRC but not rollout FP is
  rejected — the rollout metric is the arbiter.

---

## Step 8 — Evaluation: Baseline / PIGI / Oracle (`piginet/eval.py`) — the headline

**Protocol.** Per **test** problem: generate the k=200 astar+dist pool once, then **replay three orderings**
against a **lazy per-(problem,plan) refine memo** with `_stable_seed` (each plan refined at most once; label
+ cost identical across orderings — only order differs; reuse `heuristic_experiment.py`'s memo):
- **Baseline** — astar+dist order (= collection order), refine to first feasible.
- **PIGI** — sort 200 plans by model score desc, discard `< tuned threshold`, refine to first feasible.
- **Oracle** — feasible-first (memo ground truth) = headroom bracket.

**Metrics, reported PER STRATUM (s=1/2/3) and overall** — mean±std + scatter:
1. **PRIMARY: rollout FP count** = infeasible plans refined before first success (= first-feasible-rank − 1),
   per variant. Also report the **reduction ratio** PIGI-FP / Baseline-FP and the PIGI-vs-Oracle gap. Per
   stratum matters: Baseline FP ≈ 3.5 / 18 / 127, so s=3 is where the biggest savings *and* the biggest
   headroom live; aggregating across strata would mask this.
2. **Refinement cost** to first success (Σ `n_attempts` stream calls + wall time) along each ordering.
3. PIGINet inference time for scoring 200 plans (small; 0 for Baseline).
4. **Solve-rate parity** (§9.3): PIGI's threshold must not drop solves vs Baseline; if it does, refine
   sub-threshold plans at a lower rate (completeness fix) — but the val-tuned high-recall threshold (Step 7)
   should prevent this.
5. Secondary classification diagnostics: AUPRC / AUROC / balanced acc / confusion @ tuned threshold.
6. **Headroom statement**: PIGI-vs-Oracle rollout-FP gap per stratum → the direct answer to "is there room to
   beat PIGINet on DD2D?".

Write per-problem CSV + Fig-7b scatter (rollout FP, per stratum) + Fig-7a-style refine-cost bars to
`out_dd2d/piginet_eval/`.

**Honesty caveat (writeup + `notebook.md`).** DD2D Day-1 negatives are provisional (`marginal`); the
arrangement-complete negative certificate is deferred. All labels come from the **refiner** (binary
feasible-within-budget) — the paper's own noisy scheme — applied **identically at collect and eval**. The
Baseline/PIGI/Oracle *relative* rollout comparison is internally valid; **absolute feasibility numbers
inherit the labeler caveat**. Frame as "planning-time improvement under the refiner's noisy oracle."

**✅ Gate.** `test_piginet_eval.py`: on a scripted pool (fixed labels/costs/scores) the replay computes the
correct rollout FP count + refine-cost for each variant; the memo refines each distinct plan at most once.
**Real dry run** on a small test subset → monotone sane numbers (Oracle ≤ PIGI ≤ Baseline FP in expectation),
CSV/plots render. **Headline run** on the full 100-problem test set → per-stratum Baseline/PIGI/Oracle
rollout-FP table + plots + headroom statement, logged to `notebook.md`.

---

## Risks & mitigations

- **s=3 stratum starves at k=200** (geometry-blind planner may not reach a 3-object feasible staging) → Step
  4 measures; raise k / lower sub-target, log the cap.
- **Degenerate neg:pos / low Baseline FP** (no signal) → Step 4 go/no-go.
- **Extreme imbalance (≈49:1, s=3-dominated)** → Step 7's subsampling + focal/ranking loss + AUPRC +
  high-recall threshold; the rollout FP gate is the arbiter.
- **CLIP weak on 2D polygons** → `g_obj` fuses pose+shape; Step 6 identity-binding test guards degeneracy.
- **MPS op gaps** → `PYTORCH_ENABLE_MPS_FALLBACK=1`; CLIP is a one-time cached pass.

## Deferred (out of scope now)

Unseen-object / unseen-blocker-count generalization — for the object variant, evaluate a model trained on
`split="train"` against test problems generated with `split="holdout"` (the wired-but-unused diagnostic in
`shapes.py`: holds out the `bowl` family via `{"bowl":"can"}` and enlarges every shape ~15% via `shift=1.15`);
image-vs-geometry input ablation (reachable via the `g_obj` interface); Della offload.

## Living docs (per CLAUDE.md)

- `notebook.md` — Step 4 EDA + imbalance numbers, Step 5 dataset stats, Step 7 convergence + imbalance-gate
  result, Step 8 per-stratum rollout-FP headline + headroom.
- `decisions.md` — ADRs: full-multimodal object channel; local-MPS training; balanced-strata 400/100/100;
  DD2D geometry-record extension; **rollout FP as primary metric + the imbalance-handling strategy**; noisy
  refiner label used consistently at collect+eval.
- `proposal.md` — create only if a result changes the method (e.g. large Oracle headroom motivates a better
  predictor).
