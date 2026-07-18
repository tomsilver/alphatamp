# SPECTRE — Living Proposal

**2026-06-25 — Direction pivot.** The contribution is reframed from *adaptive
test-time reordering* to a **representation question for plan-feasibility
prediction** in fully-observable, deterministic bilevel TAMP. §0 below is the
current framing; the prior adaptivity-first proposal (the original §1–§6) is
retained byte-unchanged under **"Superseded framing (April 2026)"**. Rationale:
[`decisions.md`](decisions.md) 2026-06-25.

## Status & reading order

- **Current** (source of truth): **§0** below, [`decisions.md`](decisions.md)
  (newest entry 2026-06-25), [`notebook.md`](notebook.md) (newest entry
  2026-06-25).
- **Frozen / historical**: the original §1–§6 of this file (under "Superseded
  framing (April 2026)"); the April writeup
  [`archive/SPECTRE_WRITEUP_APR_2026.md`](archive/SPECTRE_WRITEUP_APR_2026.md)
  (frozen 2026-06-25). Where the current framing disagrees with these, §0 and
  `decisions.md` win.

## §0 — Current direction (representation-first)

> **Epistemic status.** The representation advantage stated here is a
> **hypothesis with a falsifiable prediction**, *not* an established result. The
> ~27% adaptivity finding (`notebook.md` 2026-06-06), the B6 lookahead-saturation
> sweep (`notebook.md` 2026-06-11), and RT2D's construction are established;
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
the bulk (`notebook.md` 2026-06-06). The SPECTRE re-ranker is therefore a
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

---

## Superseded framing (April 2026)

*Retained byte-unchanged as the record of the adaptivity-first direction. See §0
above and `decisions.md` 2026-06-25 for what superseded it; nothing below has
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
| B6 | DP-on-counts (receding-horizon expectimax over B4's calibrated counts, depth `h`) | tests whether *lookahead* over the same count model beats B4's myopic greedy; `h=1` ≡ B4 (`decisions.md` 2026-06-08) |

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
  counts by design (typed local ids, set pooling, variable-length sequences)
  but this has never been tested (writeup, Future work).
- **x₀-conditioned prior.** A PIGINet-style feasibility predictor over the
  concrete initial state as an additional scorer input — a strict
  generalization of the current deliberately x₀-free setup (writeup, Future
  work). *(⚠️ 2026-06-25: now central, not future-work — the low-level vs.
  abstract substrate question is the contribution; see §0.)*
- **DD2D as the packing / negative-control testbed.** DD2D (Drawer Decluttering
  2D) is wired in as env_variant `dd2d_v2` via a JSON→EpisodeRecord converter
  (`envs/dd2d/spectre_convert.py`; `decisions.md` 2026-07-12), abstract-only for
  now. Feasibility is a continuous packing problem, so abstract-first is expected
  to *lose* — DD2D is the negative control that bounds the representation claim
  (§0). The source JSON retains per-object poses/shapes/sizes, so DD2D is also the
  natural home for the x₀-conditioned comparator above: the crossover prediction
  is testable here once a low-level baseline is stood up. Label caveat: DD2D's
  Day-1 labeler marks non-area-proven negatives as *marginal*, so no
  label-dependent number until its negative certificate lands.
- **DAgger round** — only if test-time gap appears that offline AUROC(t) does
  not predict.
- **Deferred from the original spec:** cost-weighted PL (wall-clock metric),
  OOV graceful fallback, refiner-instrumentation features for Ψ (deliberately
  excluded so the SPECTRE-vs-B4 gap is attributable to skeleton structure, not
  refiner introspection).
