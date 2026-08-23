# SPECTRE — Living Proposal

**2026-06-25 — Direction pivot.** The contribution is reframed from *adaptive
test-time reordering* to a **representation question for plan-feasibility
prediction** in fully-observable, deterministic bilevel TAMP. §0 below is the
current framing; the prior adaptivity-first proposal (the original §1–§6) is
retained byte-unchanged under **"Superseded framing (April 2026)"**. Rationale:
[`decisions.md` 2026-06-25](decisions/02-pivot.md#2026-06-25-direction-pivot-representation-question).

## Status & reading order

- **Current** (source of truth): **§0** below, [`decisions.md`](decisions/README.md)
  (newest entry 2026-06-25), [`notebook.md`](notebook/README.md) (newest entry
  2026-06-25).
- **Frozen / historical**: the original §1–§6 of this file (under "Superseded
  framing (April 2026)"); the April writeup
  [`archive/SPECTRE_WRITEUP_APR_2026.md`](archive/SPECTRE_WRITEUP_APR_2026.md)
  (frozen 2026-06-25). Where the current framing disagrees with these, §0 and
  `decisions.md` win.

## §0 — Current direction (representation-first)

> **Epistemic status.** The representation advantage stated here is a
> **hypothesis with a falsifiable prediction**, *not* an established result. The
> ~27% adaptivity finding ([`notebook.md` 2026-06-06](notebook/README.md)), the B6 lookahead-saturation
> sweep ([`notebook.md` 2026-06-11](notebook/01-foundations.md#2026-06-11-b6-exact-h-sweep)), and RT2D's construction are established;
> the crossover below is conjecture to be tested.

**The contribution.** A *representation question* for plan-feasibility prediction
in bilevel TAMP: **what should a feasibility predictor represent skeletons and
problems over?** The standard low-level predictor — PIGINet-style: predict from
the concrete low-level initial state (multi-camera images + relational literals),
scored once — may **not** be the most effective substrate. The hypothesis is that
much of what determines refinement feasibility is relational structure that a
**richer-than-pixels, cheaper-than-full-state** representation can capture more
sample-efficiently and with weaker perception. Another representation of interest
is the target representation; currently, most targets are binary labels (1 for
successful refinement, 0 for failed refinement) but we hypothesize that a richer
representation could deliver more bits of information per training label.

**Design space (abstract-first is one point, not the thesis).** The current
**leading candidate** is *abstract-first* — predict primarily from the abstract
state + skeleton structure — but it is one point in a space that also includes
learned latents, object-centric / graph features, intermediate
symbolic-plus-coarse-geometric states, and invented predicates. Abstract-first
may prove **too lossy**; we follow the experiments.

**Falsifiable prediction (the crossover).** In the **low-data or weak-perception**
regime, a well-chosen representation should **match or beat** a low-level
(PIGINet-style) predictor on downstream planning efficiency; given **abundant
data and strong perception**, the low-level predictor should **regain its edge**.

**Efficiency, not access.** This is a sample-efficiency / perception-lightness
claim, **not** an information-access claim: under full observability and
determinism, *no* representation can beat an ideal low-level predictor on
information grounds. The advantage sought exists only under realistic data /
perception constraints.

**Negative control.** Where feasibility hinges on fine continuous fit (e.g.,
dense packing), any compressed representation is expected to **lose**; such
domains bound the claim.

**x₀ stance is open.** We are **not committed** to dropping or keeping the
low-level state x₀. PIGINet's own ablation shows x₀ carries signal *in their
kitchen problems*, which weakens the *universal* data-efficiency rationale for
dropping it — but does not establish that x₀ must always be included. Whether
dropping low-level state is a helpful abstraction is **domain-dependent**;
experiment-driven (cf. the §6 "x₀-conditioned prior" item, now central rather
than future-work).

**Adaptivity is secondary.** Within-episode refinement failures carry free
instance-specific signal, but our own analysis attributes only a **minority
(~27%)** of the method's margin to this component; the static representation does
the bulk ([`notebook.md` 2026-06-06](notebook/README.md)). The SPECTRE re-ranker is therefore a
**secondary, composable** increment — orthogonal to, and combinable with,
whichever representation wins.

**Evaluation plan.** *Prefer* **pre-existing environments** — but only those that
exhibit the properties we hypothesize make a relational/abstract representation
stronger than a low-level predictor; **bespoke, hand-crafted environments remain
in scope** where they more realistically expose that advantage. Candidate
properties (an open, evolving list):

1. Feasibility is governed by relational structure the abstraction captures.
2. The low-level state is high-dimensional and distracting, or otherwise hard to
   extract relational structure from (which the abstraction hands over for free).
3. Perception is genuinely limited or costly in the domain.
4. **Object-count / identity generalization** — feasibility structure recurs
   across object identities and counts, so an identity-agnostic, compositional
   relational abstraction generalizes from few problems where pixel features
   don't.
5. **Long horizon / large diverse pool** — long plans and many goal-reaching
   skeletons make feasibility prediction non-trivial in the first place, and the
   abstraction's compression advantage over raw perception grows with scene
   complexity.

Concretely, the planned sweep uses pre-existing homes — PIGINet's kitchens with
progressively **degraded perception**, and **clutter/distractor** domains (e.g.,
Khodeir et al.) augmented with a **low-level baseline** — over the grid of
**perception-degradation × training-set size**. Primary metric:
**time-to-first-success** in refinement attempts; secondary: **time-to-k**.
See [`research_lit.md`](research_lit.md) for the candidate environments and the
low-level (PIGINet-class) comparison.

**The zero-data corner: VLMPlan.** The data axis needs an endpoint at *zero* training
problems, so a zero-shot VLM planner (the KinDER LLMPlan/VLMPlan convention) is a
comparison method alongside PIGINet (trained low-level) and SPECTRE (trained
abstract-first) — `vlmplan/`, wired into both the DD2D and StickButton2D comparisons; see
`decisions.md` 2026-07-24 for the protocol. **The headline arm is a frontier VLM,
`gpt-5.6-terra` over the OpenAI Responses API** (2026-08-08, replacing the weaker
`gpt-5.6-luna`), so the corner is not dismissible as "you only tried a weak local model";
the earlier local Qwen arms are kept for a local-vs-frontier contrast. It is framed as **a
corner of the grid, not a defeated rival**, and it answers the reviewer-obvious "did you try
just asking a VLM?" on the record. Three properties are load-bearing for it to be a fair
point rather than a straw man: it never sees refinement outcomes (static, so it is not
competing with SPECTRE's adaptivity); it is given the operator semantics and the
object-centric geometry that every other method reads from the domain and the state —
including, since 2026-08-08, the **gripper's own dimensions** (finger size, aperture; the
fixed domain constant the trained methods absorb from labels — `PROVENANCE.md` deviation 9),
and on StickButton2D kinder's own scene render with Set-of-Mark object labels overlaid,
because kinder draws every unpressed button as an identical unlabeled disc; and because it
*generates* plans instead of reordering the candidate pool, its off-pool proposals are
refined for real and charged as attempts. Its wall-clock is now reported too
(generation-dominated — the Responses round-trip + reasoning dwarfs the sub-second
refinements). Empirically, terra + the gripper disclosure roughly **halve** the earlier luna
FP on both environments (**SB2D 11.85→6.42, DD2D 62.98→35.23**): it is a **genuine planner on
the relational StickButton2D** (self-solves 39/40, beats the naive order across the board, but
still ~3–4× behind the learned rankers) yet only reaches **parity with the naive planner order
on the DD2D packing negative control** (35.23 vs astar 34.52, bimodal — trivially-graspable
targets solved instantly, staging problems flailed) and stays far behind the learned rankers
there. A full-scale medium-effort DD2D arm confirmed reasoning effort does not rescue it (33.5,
tied with low). So the stronger model does not overturn the negative control; it makes the
bound more defensible. See [`notebook/07` 2026-08-08](notebook/07-stickbutton2d.md). Deviations
from the KinDER template are enumerated in `vlmplan/prompts/PROVENANCE.md` for the appendix.

**The learned-adaptive competitor: LAZY.** PIGINet is a *static* learned ranker; to show
SPECTRE beats other *adaptive* methods the comparison needs a learned adaptive competitor, so
**LAZY** (Khodeir et al, *Policy-Guided Lazy Search with Feedback*) is re-implemented over the
fixed candidate pool as a comparison method — a GAT policy π guiding refinement order, updated
online by feasibility statistics ϕ (`baselines/lazy/`, deviations in
`baselines/lazy/PROVENANCE.md`; protocol
[`decisions/07` 2026-08-09](decisions/07-stickbutton2d.md#2026-08-09-lazy-policy-guided-adaptive-baseline-added-dd2d)).
It is wired into both the DD2D and StickButton2D figures as `LAZY-adaptive`, 3 seeds.
Empirically it draws the **adaptive-method bar** the same way the representation story splits:
**on DD2D (packing) both learned rankers beat it decisively** — LAZY 23.26 vs SPECTRE 5.92 /
PIGINet 17.27 (paired CIs exclude 0), though it still beats the naive order (34.52) and VLMPlan
(35.23), carried by s3 — **while on StickButton2D no method separates** (LAZY 1.85 ≈
SPECTRE 1.84 ≈ PIGINet 2.28; paired CIs include 0), extending the standing SB2D non-separation
finding to a third adaptive method. A policy-isolation diagnostic confirms the GAT policy is
load-bearing (not just the ϕ feedback). See
[`notebook/07` 2026-08-09](notebook/07-stickbutton2d.md#2026-08-09-lazy-baseline-results-dd2d-sb2d).

---

## Superseded framing (April 2026)

*Retained byte-unchanged as the record of the adaptivity-first direction. See §0
above and [`decisions.md` 2026-06-25](decisions/02-pivot.md#2026-06-25-direction-pivot-representation-question) for what superseded it; nothing below has
been rephrased.*

**S**keleton-**P**ool **E**mbedding with **C**ontextual **T**ransformer for
**RE**ordering: a learned adaptive re-ranker that reorders a TAMP skeleton pool
*during* the refinement loop, conditioning on the skeletons that have already
failed.

This is the single living document for the project. It consolidates the original
spec stack (see [`archive/README.md`](archive/README.md)); where it disagrees
with an archived spec, this document wins. Full architectural detail lives in
`archive/SPECTRE_RT2D_METHOD_SPEC.md`; full data-pipeline detail in
`archive/SPECTRE_TRAINING_PIPELINE_AS_BUILT.md`. Related work:
[`research_lit.md`](research_lit.md). A paper-style narrative snapshot of the
method and results as of 2026-04-27 — including the formal problem statement
(π, ℱₜ, the Attempts/T objectives) — is frozen at
[`archive/SPECTRE_WRITEUP_APR_2026.md`](archive/SPECTRE_WRITEUP_APR_2026.md);
known-stale points are catalogued in [`archive/README.md`](archive/README.md).

---

## 1. Problem

Bilevel TAMP planners enumerate a pool `S = {s₁ … s_K}` of abstract skeletons,
then refine them one at a time until one succeeds. The order of refinement
attempts dominates wall-clock on hard problems. Static rankers (historical
success rates, PIGINet-style feasibility scorers) fix the order before the
first attempt. SPECTRE's claim: **the identity of skeletons that already failed
carries exploitable information about which remaining skeletons to try next**,
and a learned, failure-conditioned re-ranker can beat any static order.

Metric: **mean time-to-first-success** (attempts) on a held-out test split,
mean ± std over ≥ 3 seeds, evaluated uncensored at attempt budget 30 (= the
candidate-pool cap, so the budget never binds — see `decisions.md`
2026-06-07). Headline comparison: the
**adaptivity premium** over B4 (the strongest non-learned adaptive baseline).
SPECTRE is the candidate method; B1–B5 are the baselines (never the reverse):

| | Baseline | Role |
|---|---|---|
| B1 | random floor | bottom anchor |
| B2 | default planner order | deployment baseline |
| B3 | static-historical (Laplace-smoothed success rates on canonical keys) | the static ranker to strictly beat |
| B4 | adaptive-historical (Naive-Bayes log-odds over pairwise failure conditionals) | **headline comparison** — empirical lower bound on the adaptivity premium |
| B5 | oracle | top anchor / headroom |
| B6 | DP-on-counts (receding-horizon expectimax over B4's calibrated counts, depth `h`) | tests whether *lookahead* over the same count model beats B4's myopic greedy; `h=1` ≡ B4 ([`decisions.md` 2026-06-08](decisions/01-foundations.md#2026-06-08-dp-on-counts-b6-baseline)) |

## 2. Why RoutedTransport2D

On the kinder kinematic2d envs (ClutteredStorage2D etc.), B3 — a lookup table —
is near-oracle: train/test skeleton pools overlap heavily and one skeleton
family dominates (`archive/SYNTHETIC_ENVIRONMENT.md`). No research gap there.
RoutedTransport2D (RT2D, `archive/ROUTED_TRANSPORT2D_SPEC.md`) was built so
that beating B4 requires *relational* reasoning: a hidden scene latent
(blocked color / blocked grasp) plus per-problem static tags
(`PassageWidth`, `ItemSize`, `TagOn`) determine which skeleton *family*
survives, and the only way to identify the family early is to bind s₀'s static
atoms to operator arguments and to read evidence out of observed failures.
Historical baselines on canonical keys structurally cannot do the per-problem
tag binding (spec §3.5).

The environment lives in-package: `alphatamp/approaches/spectre/envs/routedtransport2d/`
(gym env, closed-form plan generator, three-gate refiner, problem generator),
registered via `env_registry.py`. Earlier collections on ClutteredStorage2D-b5/b7
and StickButton2D-b5 are historical (the configs and data remain usable).

## 3. Method (current, post-RT2D-spec)

Three modules, trained jointly end-to-end (~185k params, d=64 throughout —
see `archive/SPECTRE_RT2D_METHOD_SPEC.md` §3–§7 for exact shapes):

- **Skeleton encoder Φ** — per-skeleton transformer over interleaved
  `[STATE_0, OP_1 … OP_L, STATE_L]` tokens (Substage A: intermediate states not
  encoded; recoverable on demand via STRIPS progression in `trajectory.py`).
  Atom pooling inside the state tokens is a **Set Transformer (SAB + PMA)**,
  not Deep Sets — required for the relational join between `PassageWidth` and
  `ItemSize` atoms (RT2D fix 1). Object identities are canonicalized to typed
  local ids (`canonicalize.py`); within-type permutation augmentation applies
  **only to augmentable types** — ordered/semantic types (`WidthLevel`,
  `SizeLevel`, `GraspMode`, `Zone`, `Passage*`) are frozen (RT2D fix 2).
  Embedding/MLP widths are sized from the vocab's max arities at init (fix 3).
- **Context encoder Ψ** — permutation-invariant Set Transformer (SAB×2 +
  PMA_k=1) over the set of failed-skeleton embeddings; learned `c₀` for the
  empty history.
- **Scorer σ** — MLP over `[e(s); c_t; π_proj(π(s))]` → scalar logit, with
  prior-dropout 0.2 and init-toward-prior so an untrained σ ≈ α·π.
- **Plug-in prior π** — not trained jointly. Originally `ZeroPrior`; a
  `HeuristicPrior` (per-episode z-score of negated pyperplan-FF trajectory
  cost) was added post-spec (`train.prior_type: zero|heuristic`) to give σ a
  warm start that mirrors what B2/FF can see.

**Loss is load-bearing:** listwise Plackett-Luce
`ℒ = logsumexp(logits over R) − logsumexp(logits over SUCC∩R)` — i.e.
`−log P(argmax picks a success)`, rollout-aligned with time-to-first-success.
Attempt 2 failed precisely because pointwise BCE is not rollout-aligned.

**F-subset discipline:** each training example is `(R, SUCC∩R, F)` with
`F ⊆ FAIL_e` only — **F must never contain successes** (test-time
distribution; the other Attempt-2 root cause). `|F|` is sampled from the
`rollout_aligned_mix` (0.25 uniform-subsets / 0.25 uniform-size /
0.5 log-normal) so training mass matches the test-time visit distribution,
which is heavy at `|F| ∈ {0,1,2,3}` (RT2D fix 4). Each episode contributes
`num_f_samples_per_epoch = 8` examples per epoch (fix 5).

## 4. Pipeline

Four stages; every stage is a Hydra entry point under `experiments/spectre/`
with configs in `experiments/spectre/conf/` (see the home `CLAUDE.md` for
exact commands):

1. **Data collection** — `spectre_collect.py`: for each problem seed, enumerate
   the skeleton pool (K_max cap), refine *every* skeleton non-short-circuiting
   with a stable per-skeleton seed, and persist one gzip-pickled
   `EpisodeRecord` per problem under
   `data/spectre/raw/<env_variant>/<split>/episodes/`. Budgets: 500 train /
   100 val / 100 test per env. Atomic writes; worker-parallel under SLURM.
2. **Vocab build** — `spectre_build_vocab.py`: extract operator/predicate/type
   vocab from **train only** (`<OOV>` reserved at id 0), walking full STRIPS
   reconstructions so intermediate-only predicates (e.g. `Holding`) are not
   missed; OOV-validate val/test. Output: `data/spectre/derived/<env_variant>/train_vocab.json`.
3. **Training** — `spectre_train.py`: `SpectreDataset` samples `(R, F)`
   examples online (no materialized Layer 2 — the parquet layer from the
   original pipeline spec was deliberately collapsed, see
   `archive/SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` §3.1). AdamW + cosine
   schedule, checkpoints to `data/spectre/checkpoints/<run>/<env_variant>/seed_<i>/`.
   Post-spec changes: validation selection is **rollout-based** (simulated
   sparse rollout on val episodes) rather than val-PL-loss; extra dropout
   added against overfitting; static environment predicates separated out
   (and `Connects` dropped from them while topology is constant across
   problems).
4. **Experiments / analysis** — `experiments/spectre/analyze_spectre.py`
   (a marimo notebook) drives `eda.py`: EDA gates, B1–B5 baseline brackets, rollout simulation of
   trained checkpoints, method-comparison table + attempt-distribution plots.
   `spectre_check_pipeline.py` sanity-checks a collection; multi-seed training
   via `spectre_train.slurm`.

## 5. De-risking gates

Run in order; each gate must pass before spending compute on the next stage.

1. **EDA pass bar** (per env, on the 500-episode train collection —
   `archive/SPECTRE_EDA_SPEC.md` §6): pool cap saturated ≥ 95%; canonical
   skeleton set not trivially small (rarefaction not saturated, or U ≥ 4× cap);
   episode success rate ≥ 50%; default-order budget exhaustion ≥ 10%; and the
   **adaptive premium Δ = B3 − B4 > 0 with 95% CI excluding zero** — Δ ≈ 0 is
   *blocking* (Ψ would have nothing to learn).
2. **Pre-training smoke tests** (RT2D spec §11.1): forward shapes, empty-F
   path returns c₀, augmentation invariance/equivariance, vocab-arity sizing,
   PL-loss limit behavior. Covered by `tests/approaches/spectre/`.
3. **During-training bar:** AUROC(0) > 0.55 after 1 epoch (else Φ broken);
   AUROC(3) − AUROC(0) ≥ 0.05 by epoch 5 (else Ψ collapsed to the prior).
4. **End-to-end bar** (test split, ≥ 3 seeds): beat B3 by ≥ 1 attempt; beat B4
   by ≥ 0.3 attempts (headline); step-1 success rate no worse than B2.

**Metric discipline (hard-won):** model selection and early stopping are
**rollout-based** — `val_rollout_attempts` (simulated sparse rollout on the
val split, attempt budget 20; `checkpoint_metric` in `train.py`) — because the
test-time objective is itself rollout-based. Validation AUROC(3) is a
*secondary* offline diagnostic (it drives the during-training gates in §5),
never the selection criterion. The atom-sensitivity probes (D.1/D.2,
`experiments/spectre/spectre_probe_atom_sensitivity.py`) do *not* predict
rollout performance — they are diagnostics only; never optimize for them.

## 6. Open questions / current frontier

- **Overfitting on RT2D-n3.** The recent commit train (`diagnose overfitting` →
  dropout → rollout-based validation → heuristic prior) is working through a
  train/val generalization gap. Open: is the gap data-bounded (500 episodes) or
  capacity/regularization-bounded?
- **Prior choice.** Does `HeuristicPrior` (FF-cost z-score) improve time-to-
  first-success over `ZeroPrior`, or does it just speed early epochs? (Multi-
  seed comparison pending; checkpoint dirs `heuristic_prior/` vs `c1_baseline*/`.)
- **Static-predicate handling.** Static env predicates are separated and may be
  underattended by Φ; `Connects` is currently dropped because topology is
  constant. If problem topologies ever vary, this must be revisited.
- **Φ_s pooling depth / Ψ depth** — one more SAB layer each if the acceptance
  bar is missed (RT2D spec §12).
- **Ψ's fixed-size summary.** Ψ pools the whole failure set into one d=64
  vector; distinct failure patterns compete for capacity at large |F|. If
  long episodes suffer, expose the per-failure embeddings to the scorer
  directly instead of (or alongside) the pooled c_t (writeup, Limitations).
- **Data efficiency.** Collection refines every pooled skeleton per problem —
  the dominant collection cost — and scaling of test attempts with |D| is
  uncharacterized; a 1–2 order-of-magnitude sweep over training-set size is
  among the most informative experiments not yet run (writeup, Limitations).
- **Compositional generalization.** Train on one (N, zones, passages)
  configuration, test on another. The architecture factors across object
  counts by design (typed local ids, set pooling, variable-length sequences).
  On RT2D this is still untested; **on DD2D it now is** (2026-08-01): the
  dd2d_v4-trained v3 checkpoint was scored train-old / test-new on held-out sets
  with **unseen blocker counts** (13–15 vs the trained 9–12) and **unseen shape
  figures** (a T and a cross, added with no per-shape grasp code). v3's advantage
  over the naive planner order **survives** OOD (still wins overall, CI excludes
  zero, on both sets), while absolute FP degrades ~1.6–1.9× and its s2 advantage
  collapses under the shift — carried at the ALL level by s3, where the planner
  order is pathological. Scoring hit **no OOV** (the vocab is over the fixed
  operator/predicate/type set, so a new shape family is geometry metadata, not a
  token). *Caveat: the s2 collapse is a pool-composition artifact — s2 problems have
  only ~1.5 unique feasible solutions, padded in-distribution by redundant feasible
  triples the k=200 pool crowds out at high count — not a clean model signal; read the
  generalization at s3 (diagnosed 2026-08-02).* **Confirmed by a shape-only isolation
  (2026-08-04):** a held-out set that forces the tee/cross figures at the *trained* 9–12
  blocker count (`dd2d_v4gen_shapeonly`) degrades only ~1.17× overall (adaptive 6.77 vs
  5.78), s3 *improves* (9.19→6.03) and s2 lifts only moderately (10.49→17.27, vs the
  count-confounded ~32) — so the severe s2 OOD degradation was primarily count-driven, not
  the new shapes. One twist: under the shape shift **SPECTRE-static falls behind PIGINet**
  (22.55 vs 15.27) and only the adaptive re-ranking recovers the win, so the
  representation-vs-adaptivity attribution is not shift-invariant. See
  [`decisions/07` 2026-08-01](decisions/07-stickbutton2d.md#2026-08-01-dd2d-generalization-test-unseen-count-unseen)
  and [`notebook/07` 2026-08-02](notebook/07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model),
  and the shape-only follow-up
  [`decisions/07` 2026-08-04](decisions/07-stickbutton2d.md#2026-08-04-shape-only-dd2d-gen-variant-precompute---test-variant)
  / [`notebook/07` 2026-08-04](notebook/07-stickbutton2d.md#2026-08-04-dd2d-shape-only-generalization-shapes-isolated-count).
  *⚠️ Even the residual shape-only s2 lift is not real (2026-08-06): it is collection variance,
  not the new shapes' size. Three probes settle it — the new shapes are mid-sized even by
  convex-hull footprint and buffer packing is 5% of failures (physical gate); **v3 is image-free
  but geometry-AWARE** (it encodes each object's boundary/pose/area), yet paired inference-time
  interventions that rewrite the tee/cross model-input geometry — correcting area to hull,
  convexifying the boundary, shrinking ×0.7 — leave FP identical to the digit, so the ranking is
  inert to the new shapes' geometry; and a fresh un-shrunk control reads s2 5.63 (below the
  in-dist 10.49) while astar s2 is stable at 14–15 across collections. So the shape-only s2=17.27
  was a high-variance draw of a ~1.5-solution stratum, not a shape effect. This also refines the
  §0 framing: "abstract-first / image-free" does not mean geometry-free — v3 already ingests
  object-centric geometry, it is simply weakly weighted for ranking. See
  [`decisions/07` 2026-08-06](decisions/07-stickbutton2d.md#2026-08-06-shape-generalization-s2-deficit-collection-variance-shape)
  / [`notebook/07` 2026-08-06](notebook/07-stickbutton2d.md#2026-08-06-dd2d-shape-size-sweep-geometry-interventions-size).*
  *(2026-08-08: the scene inputs were then **narrowed to domain-agnostic columns** acting on
  exactly this inertness — the target-anchored `obj_rel` offsets and the privileged `concave`
  flag were cut, `obj_is_target` became goal-derived `obj_is_goal`, and an inference probe priced
  the removal at Δ 0.00 FP on both deployed models. v3 stays geometry-AWARE — it still ingests
  each object's boundary, pose and area — but only the **anchor-free** subset, so "image-free"
  now also means target-agnostic. See
  [`decisions/07` 2026-08-08](decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target).)*
  **A third generalization axis — held-out *stratum* — was added 2026-08-09.** Distinct from
  the unseen-count/shape tests (which hold out geometry), this holds out a whole stratum from
  *training*: SPECTRE + PIGINet trained on s0–s2 (DD2D) / b1/b2/b3 (SB2D) and evaluated on the
  never-trained s3 / b5 (same test problems; astar + VLMPlan training-free and reused). It
  reproduces the cross-environment story of the other two axes: **on DD2D the abstract ranker
  generalizes and the low-level predictor collapses** — held-out s3 SPECTRE-adaptive 9.97 vs
  PIGINet 85.89 (~9×), SPECTRE's ALL 5.35 ≈ its in-dist 5.78 while PIGINet blows out
  17.27 → 27.88 — **and again the static representation alone is not shift-invariant** (s3 static
  44.27 → adaptive 9.97; the failure-conditioned re-ranking recovers the win). **On SB2D the
  advantage does not reproduce**: held-out b5 PIGINet 5.36 ≈ SPECTRE-adaptive 6.87 (PIGINet
  marginally ahead, within seed spread), the same non-separation as in-distribution.
  **Coherence follow-up (2026-08-10).** Read the *other* axis — subset-vs-full, "does a training
  superset help?" — the first pass looked incoherent (subset ALL below the deployed full; SB2D
  ranking flipped). With **matched full-strata controls** (current-code, not the frozen yardstick)
  and a **per-stratum paired bootstrap**, that dissolves: the aggregate deltas are within noise
  (DD2D SPECTRE ALL Δ −0.57 ns; SB2D ALL Δ ≈ 0), the held-out stratum is *directionally coherent*
  (full ≥ subset) — significant only where the effect is large (DD2D PIGINet s3 45.20 ≪ 85.89) —
  and the one robust effect is **trained-strata specialization** (holding out the hard stratum
  measurably *improves* an easy one: DD2D s1, SB2D b3). SB2D's b5 train was expanded **17→100**
  (new frozen-preserving variant `stickbutton2d_v2`) to remove the 6 %-perturbation artifact, which
  killed the "flip." See
  [`decisions/07` 2026-08-09](decisions/07-stickbutton2d.md#2026-08-09-held-out-stratum-generalization-train-s0-s2-b1-b3-evaluate)
  / [`notebook/07` 2026-08-09](notebook/07-stickbutton2d.md#2026-08-09-held-out-stratum-comparison-spectre-generalizes-dd2d-s3),
  resolved in
  [`decisions/07` 2026-08-10](decisions/07-stickbutton2d.md#2026-08-10-held-out-stratum-anomalies-resolved-matched-controls-per-stratum)
  / [`notebook/07` 2026-08-10](notebook/07-stickbutton2d.md#2026-08-10-held-out-vs-matched-full-controls-anomaly-confound).
- **x₀-conditioned prior.** A PIGINet-style feasibility predictor over the
  concrete initial state as an additional scorer input — a strict
  generalization of the current deliberately x₀-free setup (writeup, Future
  work). *(⚠️ 2026-06-25: now central, not future-work — the low-level vs.
  abstract substrate question is the contribution; see §0.)*
- **DD2D as the packing / negative-control testbed.** DD2D (Drawer Decluttering
  2D) is wired in as env_variant `dd2d_v2` via a JSON→EpisodeRecord converter
  (`envs/dd2d/spectre_convert.py`; [`decisions.md` 2026-07-12](decisions/03-dd2d-v2.2.md#2026-07-12-dd2d-integration-converter-not-native-env)), abstract-only for
  now. Feasibility is a continuous packing problem, so abstract-first is expected
  to *lose* — DD2D is the negative control that bounds the representation claim
  (§0). The source JSON retains per-object poses/shapes/sizes, so DD2D is also the
  natural home for the x₀-conditioned comparator above: the crossover prediction
  is testable here once a low-level baseline is stood up. Label caveat: DD2D's
  Day-1 labeler marks non-area-proven negatives as *marginal*, so no
  label-dependent number until its negative certificate lands.
- **Restock3D — the third (bespoke, 3D) comparison environment (2026-08-19).**
  Restock3D (`restock3d_v2`, kinematic PyBullet) stores tall blocks + short cubes onto a
  shelf; feasibility is decided by **real PyBullet collision** and hinges on relational
  structure the abstraction captures — **reach-over** (a nearer object blocks the front-grasp
  of a farther one, so the store order must go far-first) and **F3** (a tall block in the
  short section hits the ceiling) — plus a height axis (cube vs block share a 2D footprint,
  so SPECTRE's scene input is a **full 3D point cloud** and it now also ingests the initial
  abstract state + goal atoms). It is now wired into the DD2D/SB2D comparison
  (`compare_methods.py`), scoped to the collected sections — as of 2026-08-20 **2×2 + 3×3 +
  4×3 at 3 seeds** (3×4 + 4×4 still collecting). **All three learned rankers crush the naive
  planner order** (astar 8.78 ALL FP vs LAZY 0.19, SPECTRE 1.44, PIGINet 1.96). Two signals
  now that the crowded **4×3** stratum is in: (i) **LAZY dominates** (4×3 paired Δ −3.57 vs
  SPECTRE, CI excludes 0); (ii) **the §0 representation advantage begins to appear** — SPECTRE
  edges PIGINet at 4×3 (4.2 vs 6.0, paired Δ −1.8) — **but the CI still includes 0** [−3.87,
  +0.10] (n=10, wide seed spread), so it is suggestive, not established; the easy 2×2/3×3 are
  tied. **Adaptivity stays inert** (SPECTRE-static ties SPECTRE-adaptive on every stratum).
  Whether the SPECTRE > PIGINet edge becomes significant awaits 3×4/4×4 + more seeds. See
  [`decisions/07` 2026-08-19](decisions/07-stickbutton2d.md#2026-08-19-restock3d-added-third-comparison-environment-v2)
  / [`notebook/07` 2026-08-20](notebook/07-stickbutton2d.md#2026-08-20-restock3d-4x3-stratum-added-3-strata).
  *(⚠️ 2026-08-20: **restock3d_v2 is being retired as too easy** — LAZY sits near-oracle, so the
  env can't separate the methods. Direction pivots to **restock3D-v3**, which makes block
  *selection* matter by varying block **x-widths** and **heights near the short/tall cutoff**. A
  pre-build **calibration study** mapped the current env's pick/place envelope — tall-section
  height 0.05–0.23 m, **short section cube-only** (gripper-limited, flagged for a v3 clearance
  change), width capped by the ~92 mm finger aperture, packing edge gap ≥60 mm, and the production
  height-adaptive grasp kept. Findings: [`docs/restock3d_v3_calibration.md`](restock3d_v3_calibration.md)
  and [`notebook/07` 2026-08-20](notebook/07-stickbutton2d.md#2026-08-20-restock3d-v3-calibration-pick-place-envelope).
  **v3 is now BUILT through the three pre-collection gates (2026-08-20):** additive per-object-dims env
  (`feasibility_v3`/`generator_v3`/`strata_v3`/`place_controller_v3`/`models_v3`, env_variant
  `restock3d_v3`, re-balanced (0.27, 0.22)), collected via an **analytic refinability classifier** (pure
  geometry, no MP) whose labels are byte-compatible with the real refiner. Gates cleared: G3 hard strata
  defeat both greedy hand-rules 100% (culprits spread across 8–9 objects); G2 static ceiling 1.00 clear /
  ~0.88 near-threshold (not saturated); **G1 analytic↔real agreement ~100% under a label-aware budget**
  (infeasible 53/53, feasible confirmed) — after correcting a flat-10 s-cap artifact (real v3 refinement
  needs ~40 s/candidate, so the eval budget must be ≥~60 s). v2 stays frozen as the negative control.
  Deferred to the collection pass: the real collection, training, and the comparison wiring. ADR:
  [`decisions/07` 2026-08-20](decisions/07-stickbutton2d.md#2026-08-20-restock3d-v3-per-object-dims-analytic-collection).)*
  **v3 SYNTHETIC comparison — the §0 crossover appears decisively (2026-08-21).** A fully synthetic
  dataset (analytic labels + synthesized wall-clock; `refiner_mode="analytic"`) of 400/100/100 over
  n=6/7/8/9 was collected and SPECTRE/PIGINet/LAZY trained at 3 seeds. **The low-level predictor
  PIGINet ≈ the naive planner order** (38.11 ± 1.01 ≈ astar 38.41 ALL FP), while both abstract rankers
  beat them **~3.4×** (SPECTRE 11.11 ± 0.98, LAZY 11.79 ± 0.08). Paired SPECTRE−PIGINet **−27.00
  [−32.97, −21.41]**, *growing with crowding* (s2 −43, s3 −48). This is far stronger than v2 (PIGINet
  1.96 ≈ SPECTRE 1.44) because v3 difficulty is capacity/height/**selection** — relational structure
  the abstraction (+3D point cloud +atoms) encodes but oblique silhouettes do not; PIGINet's crops are
  *real* here (a robot-in-`object_crops` bug that had silently zeroed restock PIGINet's image channel
  for v2 *and* v3 was fixed). **SPECTRE ≈ LAZY** (both abstract; tied). **⚠️ Read as an upper bound**,
  not a real-refiner result: the analytic labels are the exact geometric feasibility function (no MP
  noise), which favours the geometry-encoding representation; a real-refiner audit slice would price
  how much of the −27 survives. ADR/notebook
  [`decisions/07`](decisions/07-stickbutton2d.md#2026-08-20-restock3d-v3-synthetic-dataset-analytic-refiner-collection)
  / [`notebook/07` 2026-08-21](notebook/07-stickbutton2d.md#2026-08-20-restock3d-v3-synthetic-dataset-collection-spectre).)*
  *(⚠️ 2026-08-21: the "**adaptivity is inert**" reading above was **overturned** — it was an evidence-
  language mismatch masked by a bug. A canonicalize bug had zeroed coverage/waste on v3; fixing it
  recovered nothing (coverage speaks *ordering*, worth ~1%), but the **F3 exact-step certificate
  `repeat`** — the blameless height mass coverage can never see, 74% of the oracle headroom — dropped
  **SPECTRE-adaptive 12.18 → 3.13** (−9.06 paired, ~97% of the P2 ceiling), now decisively ahead of
  LAZY 11.79 / PIGINet 38.11. Purely adaptive (static unchanged); `regroup` (F2, ~1%) deprecated.
  `repeat` is a `step_certificate`-gated overlap column, inert on DD2D/SB2D (graceful degradation). See
  [`decisions/07` 2026-08-21](decisions/07-stickbutton2d.md#2026-08-21-restock3d-v3-adaptivity-revived-coverage-canonicalize).)*
- **PIGINet's SB2D pixel source is now kinder-native.** The low-level comparator
  on StickButton2D previously read a *schematic* crop (each object drawn as a lone
  polygon on a blank background); for the representation contrast to be fair, PIGINet
  should see the environment's own pixels. As of 2026-08-02 the crops come from
  kinder's built-in renderer, via a converted env_variant `stickbutton2d_v1_kinder`
  that copies every v1 record verbatim and only re-images it (reconstruct-from-seed;
  SPECTRE is image-free and unaffected). The SB2D representation contrast is therefore
  being **re-measured on valid pixels**, and the standing SB2D finding
  ("the advantage does not reproduce; PIGINet ties v3 despite a degenerate image
  channel") is pending that re-run — noting the ceiling is positional context, since
  unpressed buttons are identical discs in the env too. See
  [`decisions/07` 2026-08-02](decisions/07-stickbutton2d.md#2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new)
  and [`notebook/07` 2026-08-02](notebook/07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s).
- **DAgger round** — only if test-time gap appears that offline AUROC(t) does
  not predict.
- **Deferred from the original spec:** cost-weighted PL (wall-clock metric),
  OOV graceful fallback, refiner-instrumentation features for Ψ (deliberately
  excluded so the SPECTRE-vs-B4 gap is attributable to skeleton structure, not
  refiner introspection). *(⚠️ 2026-08-02: a simpler realization of the wall-clock
  objective landed — a **per-candidate refinement cap** (deployment knob, not a loss
  change) bounds each skeleton's refinement, so on DD2D v3-adaptive goes from ~equal
  uncapped wall-clock to **fastest** at a +0.05 FP cost; see
  [`decisions/07` 2026-08-02](decisions/07-stickbutton2d.md#2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration).
  The cap is now **per-env** and §2b runs on **SB2D too** (2026-08-03), where the finding
  **inverts**: SB2D's failures are uniformly expensive, so FP and wall-clock align
  (v3-adaptive fastest capped *and* uncapped), the cap helps the highest-FP method most, and
  because SB2D's 10 s cap sits *inside* the feasible distribution it costs a real +0.3 FP — see
  [`decisions/07` 2026-08-03](decisions/07-stickbutton2d.md#2026-08-03-sb2d-2b-wall-clock-breakdown-parity-dd2d).
  Cost-weighted PL — training the ranker to minimize expected wall-clock — remains the
  loss-level version, still deferred.)*
- **Learned pathway from raw failure evidence (2026-08-22).** A workstream asking whether the
  adaptive signal the *compiled* scalars (coverage/waste/repeat) carry can instead be **learned**
  from the raw failure-record tokens, so the typed features become training-time scaffolding rather
  than a need-to-have (the "wins are hand-engineering" objection). Four probes narrowed the cause of
  the tokens' near-inertness: **P-1** ruled out the certificate-record token holdout (empirically
  inert on every collection — C4a out); **P-2** found the FP-relevant scalars *recoverable* from the
  tokens (C1 content gap out for coverage/waste/repeat; `regroup` is a genuine but ~0–1%, FP-irrelevant
  exception — its establishing-step schema is dropped); **P-0** found the evidence cross-attention query
  is the **pooled** candidate, so a step-level join is not representable (C2). The fix
  (`--record-mode steps` / `--step-join`, additive + flag-gated, off byte-identical) shows the lever is
  **architecture, not content**: enriching the token content is inert (`fr_steps` −0.04), while a
  pre-pooling per-step candidate×evidence join over the raw record tokens (`fr_join`) is the only arm
  that moves FP. **Magnitude is modest and not established at 3 seeds:** paired `fr_join` − `fr_summary`
  = −1.80 (seed 0) → −2.38 [−4.34, −0.67] (seeds 0+1) → **−1.47 [−2.97, +0.11] (3 seeds, grazes 0)**,
  **robust at s2/s3** but s1-noisy (the seed-0 "43% gap-closure" was an optimistic draw). So *some* of
  the deployed win is recoverable by a generic attention join over raw evidence — it is not purely
  hand-engineering — but the aggregate effect is directional, not yet CI-clean. The deployed scalars-on
  method is **unchanged** (this is a parallel probe). **C1 (content enrichment) was cut** — inert alone,
  harmful combined (dilution), and its one unique value (`regroup`) is off in practice; the machinery
  stays flag-gated off per the build-then-disable convention but is not pursued. Next: more seeds +
  s1-variance fix to settle the −1.47, P-4 teachability (C3 vs C2 for the residual gap), step-join +
  scalars-on (additive?). See
  [`decisions/07` 2026-08-22](decisions/07-stickbutton2d.md#2026-08-22-step-join-lever-content-enrichment-inert)
  and [`notebook/07` 2026-08-22](notebook/07-stickbutton2d.md#2026-08-22-rung-1-result-step-join-over-record-tokens);
  full plan in `docs/failed_records_fix.md`.
