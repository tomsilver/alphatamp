# SPECTRE v2.2 — As-Built State (2026-07-25)

A single reference describing the SPECTRE v2.2 system **as it actually stands on
disk**, and reconciling it against the original proposal
[`SPECTRE_v2.2.md`](SPECTRE_v2.2.md) (v2.2.1, 2026-07-18). The living docs each
hold a slice — the *current method* (`SPECTRE_v2.2.md`), the *per-decision ADRs*
(`decisions.md`), the *dated run numbers* (`notebook.md`), and the *session
learnings* (`consolidation_2026-07-19.md`); this document ties them together into
"what is built, what works, and where it diverged from the spec and why."

**Standalone.** The model is documented in full in **§2.4** (inputs → embeddings →
layers → heads → prediction target → loss → deployment, with the Set-Transformer
blocks explained inline), and **§7** records what works, what doesn't, and what we
learned iterating. You do not need to open `model.py`, `SPECTRE_v2.2.md`, or any
other file to understand the architecture; the other docs are provenance.

> **What changed since the 2026-07-20 snapshot (this update).** The DD2D grasp model
> was made physically realistic (2026-07-24 — fingers slide on true material contact
> runs, the curved `banana` became a blocky right-angled `horseshoe`, and grasps now
> reach into concave regions), which shifted the feasibility labels and **invalidated the
> dd2d_v2 collection/checkpoints**. We added an exact-count collector and **re-collected
> `dd2d_v3`** (grasp-fixed, exactly 100 problems/stratum = 400/100/100, λ=0.8, tb=20),
> **retrained v1/v2/PIGINet on it, and rebuilt the comparison** (§3.7). An initial run read
> as "PIGINet wins / SPECTREv2 collapses at s3" — this was a **training artifact of the
> short-first prior on the easier data**; dropping the prior restores **SPECTREv2-adaptive
> to best overall (13.68, s3 fixed 23.92)**. PIGINet is now retrained (BCE/AUPRC) — P2 is
> no longer deferred — and a zero-shot **VLMPlan** baseline was added. Details:
> `decisions.md`/[`notebook.md` 2026-07-24](notebook/README.md)/25. The v2.2 *system architecture* (model_v2 /
> dataset_v2 / proof-demotion / losses) is unchanged; what changed is the dataset, the
> deployed config (prior off for v3), and the comparison landscape.

> **Seed discipline.** Numbers are labelled **[3-seed gate]** (checksum-distinct
> seeds, the reportable acceptance-gate numbers) or **[1-seed dev]** (fast
> iteration per user directive; *not* reportable until reproduced over ≥3 seeds).
> Do not quote a [1-seed dev] number in a writeup without re-running it.

---

## 1. Orientation

**What v2.2 is.** A learned re-ranker for bilevel TAMP that, given a pool of
candidate skeletons for a problem, orders them so refinement tries the feasible
one first. It conditions on three things v1 did not: **object-centric scene
geometry** (per-object footprint + pose), **typed post-mortem evidence** read
from the refiner's own failures, and **sound deductions** consumed by a symbolic
proof/hint split *outside* the network. Metrics: rollout **FP** (failed attempts
before first success) and mean **attempts-to-first-success**, uncensored (budget
= pool cap).

**Testbed — DD2D** (Drawer Decluttering 2D): a top-down drawer of 9–14 rigid
items; the target starts un-graspable; the robot stages a *subset* of blockers
onto a size-limited buffer to open a two-finger grasp corridor. The hard question
is *which subset* — it must both clear the target and jointly fit the buffer.

**Why v2.2 exists.** v1 was diagnosed as geometry- and object-blind: its ranking
was a pure function of plan *length* (η²(length) = 1.00, architecture-forced —
anonymous object ids + no geometry made same-length skeletons staging *different*
subsets identical inputs), and its failure context carried only length. An
image-conditioned predictor (PIGINet) reached ~79% within-length discrimination,
so the geometric signal exists and is learnable. v2.2 rebuilds the representation
to capture it.

**Build status.** The v2.2.1 plan's **Steps 1–11 are built and gated**
(certificate, geometry schema, G0, harvest, comparison harness, tags,
model_v2/dataset_v2, ladder+P1, proof-demotion+P4, typed-evidence+P5). A large
**unplanned arc** — "make v2 good and generalizable" (the length-bias diagnosis
and its generalizable fixes) — was inserted after Step 11. **Since then
(2026-07-24/25):** the grasp model was made realistic, `dd2d_v3` was re-collected,
v1/v2/PIGINet were **retrained on the grasp-fixed data**, and the comparison was
rebuilt (§3.7) — this exposed and fixed a prior-induced s3 collapse (the deployed
v3 model drops the prior). **PIGINet is now retrained** (BCE/AUPRC on both dd2d_v2
and dd2d_v3), and a zero-shot **VLMPlan** baseline was added. **Still deferred:**
the information-parity P2 2×2 (v2.2-crops / PIGINet-polygons), Step 12 (shape-family
shift / P3), Step 13 (second environment), the DAgger re-collection round.

---

## 2. The system as built (module map)

All paths relative to `src/alphatamp/approaches/spectre/`. v1 modules
(`model.py`, `dataset.py`, `inference.py`, `eda.py`) are untouched; v2 landed as
parallel modules selected by flags.

### 2.1 Data layer

| Module | What it provides |
|---|---|
| `schema.py` | v2.2.1 geometry/evidence dataclasses `SceneGeometry` / `ObjectGeometry` (pose, boundary ring, family, area, concave, `is_target`) / `ContainerGeometry` / `Fact` (type, args, tier, scalars) / `PostMortemRecord` / `AuxLabels`, all **trailing-nullable** on `EpisodeRecord.{scene_geometry, aux_labels}` and `OutcomeRecord.post_mortem`. Guarded invariants **I5** (every registered object has geometry) / **I6** (a post-mortem indexes its own `fail` outcome), firing only when the field is present. |
| `envs/dd2d/spectre_convert.py` | JSON→`EpisodeRecord` converter, `CONVERTER_VERSION = "dd2d_convert_v2"`. `_parse_scene_geometry` builds `SceneGeometry` from the per-object `boundary`/`shape`/`pose` + `provenance` buffer/drawer dims; returns `None` (abstract-only) if any object lacks a boundary ring. Abstract STRIPS state stays **x0-free** (`at-pose` literals dropped). |
| `envs/dd2d/dd2d/record_ext.py` | The PIGINet-side sidecar that writes each object's item-frame `boundary` ring (exterior polygon, rounded to `_ROUND = 4`), consumed by the converter. |
| `io.load_episode` | Migration shim filling the new trailing attrs on pre-v2.2.1 pickles via `object.__setattr__` (frozen-dataclass unpickle skips `__init__`), so RT2D/kinder corpora load unchanged. |

### 2.2 Geometry & sound deductions

| Module | What it provides |
|---|---|
| `envs/dd2d/spectre_geometry.py` | **Reconstruct-don't-regenerate.** Every post-hoc grasp query is a pure function of stored `SceneGeometry`: `reconstruct_scene`/`reconstruct_wall_band` rebuild a live `DrawerScene` from stored poses; `target_blocked_after_removing(scene_geometry, subset)` is the `blocked-at-contents` proof condition (uses the env's own `has_grasp`); `grasp_witness_after_removing` is the hint. Reconstructing from the *same poses the labeler used* means a proof can never contradict a label. |
| `envs/dd2d/dd2d/certificate.py` | Arrangement-complete negative packing certificate (§8.4). `certify_infeasible_by_packing(...) -> Optional[bool]` is **three-valued**: `True` = proven infeasible, `False` = a packing was found, `None` = undecided (→ stays `marginal`). Built from scratch on Shapely (`constrained_delaunay_triangles` exact convex decomposition for the 3 concave families; exact NFP/IFP; Lipschitz rotation grid `Δθ = δ/(4·r_max)`; **all placement orders** up to `MAX_ORDER_ITEMS = 5`; H1 area bound on **exact deflated areas**). Budgets `DEFAULT_EGE_BUDGET = 100_000`, `DEFAULT_TIME_BUDGET_S = 5.0`; INFEASIBLE only on full exhaustion, timeout ⇒ `None`. **Sound:** 0 false-infeasible over ~730 constructed-feasible packings. |
| `envs/dd2d/soundness.py` | `SoundnessRegistry` (model-fidelity / exactness / removal-monotone / locality → `tier(fact_type)`); `DD2D_REGISTRY` declares all four (deducible facts are proofs); `EMPTY_REGISTRY` ⇒ every fact a hint ("learning is the floor"). |

### 2.3 Harvest (typed post-mortem evidence)

| Module | What it provides |
|---|---|
| `facts.py` | Fact vocabulary. `FACT_TYPE_IDS = {blocked-at-contents:1, extraction-failed:2, grasp-witness:3, pack-exhausted:4, pack-impossible:5}`; `TIER_IDS = {proof:1, hint:2}`. `FactRecord` + `gather_context_facts(episode, failed_indices)` flattens post-mortem facts of the failed skeletons. |
| `envs/dd2d/spectre_harvest.py` (+ `envs/dd2d/dd2d/harvest.py`) | Offline harvest: `harvest_episode` returns a copy with `post_mortem` populated on every `fail`. **As built** (see §4), facts are *reconstructed* from stored geometry (`blocked-at-contents` proof, `grasp-witness`/`pack-impossible`) + read from stored `refiner_metadata` (`extraction-failed`/`pack-exhausted` hints) — no re-refinement. Certificate default is **on** in the library API but **off** in the experiment runner (`_RUN_CERT = False`; opt-in `--run-certificate`) since it proves 0 pack-impossibles at λ=0.8. |

### 2.4 The model, end-to-end (`model_v2.py`)

*This subsection is self-contained: it describes the whole network — inputs, embeddings,
layers, heads, prediction target, loss, and deployment — with no need to read v1 or the
proposal. The two attention primitives (SAB/PMA) live in v1's `model.py` but are explained
in full below.*

**What it computes.** SPECTRE v2.2 (`SpectreV2Model`, ≈ 277k trainable params, `D_MODEL = 64`)
is a **listwise re-ranker**. One forward pass consumes an entire problem — the scene (its
objects' geometry) and the pool of `K` candidate skeletons — and emits **one feasibility logit
per candidate**, shape `(B, K)`, so `argsort(−logits)` is the order refinement should try them
in. A secondary per-object auxiliary head emits `(B, M, 2)` = (necessary, relevant) logits that
inject geometry gradient in low data. There is **no per-candidate probability / BCE**: the model
is trained by a listwise ranking loss (below) and a logit is meaningful only *relative to the
other candidates in the same pool*.

**Data flow.**
```
 objects ──► SceneEncoder ─────────────────┐  (B,M,64) scene tokens
 (geometry) tag+footprint+pose+rel,         │
            2× self-attention (SAB)         ├─► CrossAttentionScorer ─► logits (B,K)
 skeletons ─► CandidateEncoder ─────────────┤   candidate = query, cross-attends
 (programs) op+pos+arg-tag, pool (PMA)       │   over memory [scene ; global ; facts];
            (B,K,64) candidate queries       │   head([cand;attended;overlap;prior])
 facts ────► FactEncoder ───────────────────┘   + additive prior_gate anchor
 (evidence) type+tier+arg-tag  (B,F,64)          ▲
                                                 └─ cand_overlap=[dead,jaccard], cand_prior=[-idx,-len]
 scene tokens ──► AuxHead ──► (B,M,2) necessary/relevant
```

Everything binds through **episode-local object tags** (`tags.assign_tags`): each object gets a
tag id `1..max_tags` (0 = pad), assigned per episode and **re-permuted every epoch** so no id
accrues global meaning — the network must read the *content* a tag points at, not memorize the
id. The **same** tag id is emitted into the object's scene token, into every candidate operator's
argument slot that mentions it, and into fact arguments about it; that shared id is the **join
key** that lets the scorer connect "this operator stages object 7" → "object 7's geometry" →
"object 7 was in a failed set." This is what removed the v1 collapse (§1): with anonymous ids and
no geometry, two same-length skeletons staging *different* subsets were literally identical
inputs.

#### Inputs — the `SpectreV2Batch` (built by `dataset_v2.build_v2_example` + `collate_v2`)

Dims: `B` episodes, `M` objects, `K` candidates, `L` operators/skeleton, `A` = max operator arity
(DD2D: `A = 1`), `P = 32` boundary points, `F` hint-facts (0 at t = 0). "0 = pad" throughout.

| tensor | shape | what it is |
|---|---|---|
| `obj_tags` | (B,M) int64 | episode-local object tag — **the join key** |
| `obj_boundary` | (B,M,P,2) f32 | exterior boundary ring in the item frame, arc-length-resampled to 32 (x,y) points |
| `obj_pose` | (B,M,3) f32 | `(x/scale, y/scale, θ)`, `scale = max(drawer_w, drawer_d)` |
| `obj_rel` | (B,M,8) f32 | relation-to-target: `[dx, dy, dist, area, sinθ, cosθ, concave, area/target_area]` |
| `obj_is_target` | (B,M) f32 | 1 for the retrieval target, else 0 |
| `obj_mask` | (B,M) bool | real-object mask |
| `cand_op_ids` | (B,K,L) int64 | operator-schema vocab id per step |
| `cand_arg_tags` | (B,K,L,A) int64 | each operator's argument-slot object tags |
| `cand_pos` | (B,K,L) int64 | operator position 0…L−1 |
| `cand_step_mask` | (B,K,L) bool | real-step mask |
| `pool_mask` | (B,K) bool | real-candidate mask |
| `glob_feats` | (B,6) f32 | `[n_objects, K, mean_plan_len, 0, 0, 0]` — **only the first 3 are populated; the last 3 (buffer dims) are hardcoded 0** and reach the model only via object geometry |
| `cand_prior` | (B,K,2) f32 | a-priori planner signals `[−index/K (default order), −removals/max (short-first)]` |
| `cand_overlap` | (B,K,2) f32 | failure-context features `[dead, jaccard]` (below); `[0,0]` when F = ∅ / under evidence-dropout |
| `avail_mask` | (B,K) bool | candidate not yet tried (∉ F); the model −inf-masks the rest |
| `success_mask` | (B,K) bool | **training target** — candidate refined successfully |
| `aux_necessary/relevant` | (B,M) f32 | aux targets, `1 / 0 / −1 (ignore)` |
| `fact_*` (`type_ids`, `tier_ids`, `arg_tags (B,F,12)`, `mask`) | (B,F,…) | typed hint-evidence tokens; **all `None` at t = 0** |

`build_v2_example` canonicalizes the episode, assigns tags, and builds the tokens from
`SceneGeometry`. Given a failure context `F` (the already-tried candidates), it computes the two
overlap features: `dead = 1` iff the candidate's staged subset `⊆` any **observed-blocked** failed
subset (a *sound*, removal-monotone deduction — "if removing a superset didn't unblock the target,
removing a subset can't either"), and `jaccard` = max Jaccard of its subset with any failed subset
(a soft hint). `demotion_source` decides how "blocked" is read: **observed** (default — the refiner
reported `failure_action = retrieve`, i.e. all removals ran and the target was still un-graspable;
no geometry) or **computed** (the harvested `blocked-at-contents` geometry fact).

#### The two attention primitives (self-contained; shared with v1's `model.py`)

Both are from the **Set Transformer** family, operating on masked variable-size sets at
`D_MODEL = 64`, `N_HEADS = 4` (head dim 16), a `64→256→64` GELU feed-forward, post-norm residuals,
dropout 0.1. No positional encodings inside them — order-invariance is the point.

- **SAB (Set Attention Block)** — multi-head **self**-attention over a set:
  `h = LN(x + MHA(x, x, x))`, then `h = LN(h + FFN(h))`; output shape = input. Lets set elements
  attend to each other.
- **PMA (Pooling by Multi-head Attention)** — reduces a set to **one** vector via a single learned
  *seed* query: `h = LN(seed + MHA(seed, x, x))`, then `+FFN`, squeeze → `(…,64)`. Used to pool a
  boundary ring into a descriptor and a skeleton's steps into a candidate vector.

#### Embeddings & per-token encoders

- **FootprintEncoder** (`obj_boundary → 32-d shape descriptor`): a shared per-point MLP `2→16→64`
  lifts each of the 32 boundary points, **PMA** pools them, `Linear(64→32)` outputs the descriptor
  — order- and start-vertex-invariant, and concave-safe (a point *set*, not a rasterization).
- **SceneEncoder** (`→ scene tokens (B,M,64)`): per object, concat `[tag(32); descriptor(32);
  pose→Linear(3→8); rel→Linear(8→8); is_target(1)] = 81`, project `Linear(81→64)+LN`, then **2×
  SAB** so objects attend to each other — the *relational join* v1 lacked.
- **CandidateEncoder** (`→ candidate vectors (B,K,64)`): a skeleton is a *program over the scene*.
  Per operator step, fuse `op_emb(n_ops+1→64) + pos_emb(64) + arg_proj(A·tag32 → 64)`, LayerNorm,
  then **PMA** over the L steps → one 64-d vector per candidate. Argument slots carry object
  **tags**, so a candidate vector knows *which* objects it stages.
- **FactEncoder** (`→ fact tokens (B,F,64)`, only when evidence is present): concat
  `[type_emb(6→64); tier_emb(3→8); mean-pooled arg-tag(32)] = 104 → Linear(→64)`. Facts carry
  object identity through their argument tags (same tag space), so a fact about object 7 lands on
  object 7.

#### The scorer and heads (`CrossAttentionScorer`, `AuxHead`)

Each **candidate vector is a query** that cross-attends over a memory built from
`[scene tokens (M); a global token = Linear(6→64)(glob_feats) (1); fact tokens (F, if any)]`,
masked so pads are ignored. The attended context `(B,K,64)` is concatenated with the raw candidate
and the two planner-feature blocks and scored by an MLP:

```
logit = head([ cand(64) ; attended(64) ; cand_overlap(2) ; cand_prior(2) ])   # Linear(→256) → GELU → Dropout → Linear(→1)
        + prior_gate(cand_prior)                                              # additive residual anchor
```

Head-input width is **config-dependent**: with prior + overlap it is `128 + 2 + 2 = 132` (the
**dd2d_v2** deployed model); the **dd2d_v3** deployed model **drops the prior** (`n_prior = 0`, no
`prior_gate`), so its head input is `128 + 2 = 130` (see §3.7). The **`prior_gate`**
(`Linear(n_prior→1)`) is initialized so the final head is zeroed and `prior_gate` carries a fixed
weight `3.0` on the `−index/K` (default-order) column — so an **untrained model ranks ≈ default
planner order**, and everything geometric is a learned *residual correction* on that anchor.
Finally the logits are `−inf`-masked at `~avail_mask` (pads, and during a rollout the already-tried
candidates). The **AuxHead** is a single `Linear(64→2)` on each scene token → per-object
(necessary, relevant).

**Proof vs hint at the network boundary.** The net never branches on a fact's tier (a proof-tier
fact would only appear as an 8-d tier embedding). The *sound* consequences of proofs are enforced
**outside** the net — through the structural `dead` overlap feature and deployment-time
proof-demotion (below) — so a wrong network weight can never override a proof, and a wrong proof
can only reorder (never delete) a candidate. Hint-tier facts flow through the fact tokens + the
`jaccard` column.

#### What it predicts, and the training objective (`loss.py`)

The prediction is the **ranking**, supervised by `success_mask`. The ranker loss is **listwise
Plackett–Luce only** (no candidate-level BCE):

```
PL = − ( logsumexp(logits over successful candidates) − logsumexp(logits over the whole pool) )
```

= −log P(an argmax pick lands on a feasible candidate) — the training analog of
time-to-first-success. A **within-length PL** term applies the *same* formula within each
plan-length bucket (keyed by the `−len/max_len` prior column), removing plan length as an
exploitable shortcut so geometry must decide feasibility *within* a stratum. The auxiliary
`necessary/relevant` head is trained with a small (weight 0.2) masked **BCE** — so "no BCE" is
precise about the *ranker*, not the aux head. Total train loss =
`PL(pool) + 0.2·BCE_aux + wl·within_length_PL` (`wl = 1.0` default, `--wl-weight`). Training samples
a failure context `F` per example (heavy mass at `|F| = 0`, the deployment start), applies
**evidence dropout** so the static pathway stands alone, and watches a **scramble gauge** (mean
|Δlogit| when the object identities inside the facts are permuted) as the "the ranker actually
reads fact identity" detector.

#### Deployment (`evidence.deployed_rollout` + `proof_demotion.py`)

At test time the trained model *is* the ranker, wrapped in a rollout: score with `F = tried`, take
the `argmax` of the (availability-masked, proof-demoted) logits, attempt it, stop on the first
success. A `ProofState` accumulates provably-dead candidates (subset-of-an-observed-blocked-set)
and `demote_scores` subtracts a **finite** `1e6` so they sort last **without ever leaving the
pool** — if everything is proven dead they are still attempted, in order (completeness invariant
**P-E**). This is the *adaptive* deployment mode; the *static* mode is simply the `F = ∅` scores.

#### Constants & parameter count

`D_MODEL = 64`, `N_HEADS = 4` (head dim 16), `FFN_DIM = 256`, dropout `0.1`; embedding dims
`D_TAG = 32`, `D_DESCRIPTOR = 32`, `D_POSE = 8`, `D_REL = 8`, tier `8`; `N_BOUNDARY_POINTS = 32`,
`MAX_FACT_ARGS = 12`, `N_FACT_TYPES = 5`, `N_PRIOR = 2`, `N_OVERLAP = 2`, `D_GLOBAL_IN = 6`,
`MAX_TAGS_DEFAULT = 32`. **Measured** parameter count (DD2D vocab: `n_ops = 4`, `max_arity = 1`):
**≈ 276.9k** for the deployed dd2d_v3 config (no prior) / **277.4k** with prior + overlap — *not*
the "~268k" an earlier draft of this doc stated (that figure was never asserted in `model_v2.py`;
~185k is a v1-only number).

### 2.5 Tensorizer / dataset

*(How these feed the network is explained end-to-end in §2.4; the tables in §2.5–§2.6
are the code index — where each piece lives on disk.)*

| Module | What it provides |
|---|---|
| `tags.py` | `assign_tags(names, rng, max_tags)` — episode-local object→tag bijection (`PAD_TAG = 0`); deterministic at `rng=None` (eval), re-permuted per epoch in training. The join key that discharges **P-A** and removes the v1 collapse. |
| `dataset_v2.py` | `build_v2_example(episode, vocab, *, demotion_source="observed", evidence, context_f, hide_facts, augment_tags, ...)` → `_V2Example`; `collate_v2` → `SpectreV2Batch`; `SpectreV2Dataset`. Overlap features `[dead, jaccard]` computed only when a context F is present: `dead = subset ⊆ any observed-blocked set` (sound removal-monotone demotion), `jaccard = max overlap with a failed set` (hint). `_fact_arrays` filters to **hint-tier only**. The `blocked` set is derived from **observed** `refiner_metadata.failure_action.startswith("retrieve")` (default) or **computed** `blocked-at-contents` fact (opt-in). |
| `loss.py` | `plackett_luce_loss` (global top-1 listwise PL) + `within_length_pl_loss` (top-1 PL **within each plan-length bucket**; key = the `−len/max_len` prior column). No pointwise BCE. |

### 2.6 Training, deployment, and harness

| Module | What it provides |
|---|---|
| `train_v2.py` | `TrainV2Config`: `use_prior`, `use_overlap`, `within_length_weight` (default 1.0), `demotion_source ∈ {observed, computed}`, `evidence`, aux_weight 0.2, AdamW + cosine; CLI `--lr` (peak LR) + an env-gated per-epoch checkpoint dump (`SPECTRE_SAVE_ALL_EPOCHS`, diagnostic). **Checkpoint selection is rollout-based** (`_val_relative_rank`: mean first-feasible-rank / random-baseline-rank at t=0, difficulty-normalized), not val PL loss — but see §3.7: it is *miscalibrated on dd2d_v3* (never <1), safe only once the destabilizing prior is dropped. Checkpoint dir suffixes `_prior` / `_ov` / `_comp`; **`use_prior` is now a data-dependent knob** — default-on for dd2d_v2/RT2D, **off for the deployed dd2d_v3 model** (§3.7, §4.2#3). |
| `evidence.py` | `scramble_gauge` (identity-scramble logit sensitivity — the "facts are used" detector), `evidence_rollout` (facts-on/off increment), **`deployed_rollout`** (the deployed ranker = model scores + sound proof-demotion; `demotion_source`-parameterized). |
| `proof_demotion.py` | `ProofState` (removal-monotone `blocked-at-contents` and superset `pack-impossible` bookkeeping), `demote`/`demote_scores` (finite offset — pushes provably-dead candidates to the back, **never drops the pool**, so a wrong proof only reorders — **P-E**). |
| `eda.py` | `lazy_baseline` (LAZY untyped-adaptive: prior − β·overlap-with-failed), `assert_distinct_seed_checkpoints` (SHA-256 seed-distinctness guard). |
| `ladder.py` | `variance_ladder` (nested-R² length → +slack → +proximity → residual) + `beats_slack_paired` (the operational P1 gate). |
| `g0.py` | `buffer_slack`, GBDT probe, `within_length_auroc` (size-conditional concordance), `choose_lambda_star(..., operating_range=(0.7, 0.95))` — λ* constrained to DD2D's designed range, maximizing the oracle − within-length-GBDT gap. |

**Experiment scripts** (`experiments/spectre/`, all present): `spectre_g0`,
`spectre_harvest`, `spectre_handrule_p4`, `spectre_eval_p5`, `spectre_eval_v2`
(ladder / P1 / P2), `spectre_main_table`, `spectre_collect`, `dd2d_convert`,
`spectre_train`, `precompute_dd2d_cache` (now `--env-variant`-parameterized — see
below), `compare_dd2d_methods`, and the zero-shot VLM baseline `vlmplan_run` /
`vlmplan_score` (`vlmplan/` package).

**dd2d_v3 pipeline additions (2026-07-24/25).** The DD2D env grasp model was made
realistic (`envs/dd2d/dd2d/grasps.py` — fingers slide on true material contact runs +
internal concave grasps; `shapes.py` — `banana`→blocky `horseshoe`), and the collector
(`envs/dd2d/dd2d/collect.py`) now guarantees **exact per-stratum counts** (in-flight cap +
truncation). Together these produced the grasp-fixed `dd2d_v3` collection, whose feasibility
labels differ from dd2d_v2. `precompute_dd2d_cache.py` gained `--env-variant` (repoints
test/vocab/checkpoints/cache per collection and derives `N_PROBLEMS` from the split) and a
per-variant v2-checkpoint map `_V2_CKPT_SUBDIR` (dd2d_v2 → `checkpoints_v2_evidence_prior_ov`;
dd2d_v3 → the no-prior `checkpoints_v2_evidence_ov`).

---

## 3. Results & gates as built

| Gate | Outcome | Seeds | Key numbers |
|---|---|---|---|
| **G0** (benchmark tests its thesis) | **PASS**, λ* = 0.8 | sweep | within-length GBDT AUROC ≈ 0.539 (≈chance) while oracle solves (1.0), feas 31.4%. Cheap stats capture length/area, not subset identity — "area is the new length," size-control mandatory. |
| **P1** (beats the cheapest statistic) | **PASS** | 3-seed gate | η²(length) = 0.227 (v1 was 1.00). Ladder: length 0.20 → +slack 0.20 → +proximity 0.45 → **residual 0.535**. Beats slack on strata≥2 by **Δ = 54.7 FP**, CI excludes 0 all 3 seeds. |
| **P2** (≥ PIGINet) | **DEFERRED** | — | needs PIGINet retrained on the λ=0.8 data. v2-static already beats default (70) and slack (105.5) on strata≥2. |
| **P4** (hand-rule proof-demotion cuts FP) | **PASS** | uncensored | ALL: 33.11 → 22.03, ΔFP **+11.08**, CI (7.77, 14.73). strata≥2: 69.98 → 46.15, ΔFP **+23.83**, CI (17.80, 30.06). Soundness telemetry 0 by construction. |
| **P5** (typed evidence beats untyped) | **PASS** | 3-seed gate | scramble gauge **0.091 ± 0.100** (>0 ⇒ fact identity used); evidence increment **+6.22**, CI (4.15, 8.43); vs LAZY **+31.57**, CI (14.15, 48.74). Decomposition **LAZY 71.1 → static 45.6 → evidence 39.5** (the representation does the bulk, typed evidence a secondary composable +6). |

**In-distribution main table** — **dd2d_v2 (pre-grasp-fix), historical** (λ=0.8 test,
142 eps; learned rows 3-seed; mean attempts, uncensored). Strata counts
s0/s1/s2/s3 = 32/44/35/31. *(The grasp-fixed dd2d_v3 refresh is §3.7.)*

| method | all | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| random | 25.73 | 12.10 | 12.82 | 31.07 | 52.08 |
| default-order | 34.11 | 1.00 | 2.86 | 18.43 | **130.32** |
| slack-order | 54.69 | 1.00 | 14.89 | 72.89 | 146.06 |
| LAZY(β=1.0) | 34.16 | 1.00 | 2.86 | 18.60 | 130.39 |
| hand-rule(proofs) | **23.03** | 1.00 | 2.86 | 17.03 | 81.16 |
| v2-static | 26.65 | 1.00 | 7.58 | 56.03 | 47.01 |
| v2-evidence | 27.32 | 1.00 | 26.64 | 44.24 | **36.37** |

This table surfaced the problem the post-Step-11 arc fixed: **the learned models
win s3 but lose the easy strata**, and un-split typed evidence *harmed* s1
(+23.8 FP isolated). Diagnosis: the static ranker used **plan length as a
feasibility proxy** (corr(logit, length) = +0.42; feasibility-AUROC ≈ chance on
s1/s2, 0.78 on s3), because the hard s3 episodes dominate the PL gradient.

**Generalizable fixes → deployed ranker** — **dd2d_v2 (pre-grasp-fix)** [1-seed dev].
Default-order prior + within-length PL loss + rollout-based difficulty-normalized
selection + proof/hint split (proofs → `dead` demotion, hints → learned tokens):

| method | all | s1 | s2 | s3 |
|---|---|---|---|---|
| default-order | 34.1 | 2.86 | 18.4 | 130.3 |
| hand-rule (proofs) | 23.0 | 2.86 | 17.0 | 81.2 |
| **deployed, observed demotion** (default) | **18.2** | 4.61 | 38.8 | 32.1 |
| **deployed, computed demotion** (opt-in) | **15.99** | 4.57 | 33.9 | 27.5 |

The deployed ranker (`deployed_rollout`, model + proof-demotion) is **best
overall**, ties s1, wins s3 handily. **s2 is the one regime it trails** — an
intrinsically hard needle-hunt (~2% of size-2 plans feasible, extraction/
blocking-limited, ~17 ceiling for *all* methods). The geometry predicate's value
is **counterfactual demotion, ~14%** (computed catches ~697 blocked signals,
observed ~245; the gap = plans that died at extraction before the grasp was
attempted).

### 3.7 — Grasp-fixed dd2d_v3 comparison (2026-07-25, [1-seed dev])

After the grasp-model fix the DD2D labels changed, so v1/v2/PIGINet were retrained on the
re-collected `dd2d_v3` (100/stratum, n=100 test) and the comparison rebuilt
(`compare_dd2d_methods.py`, `--env-variant dd2d_v3`). Corrected mean rollout **FP**
(lower is better):

| method | all | s0 | s1 | s2 | **s3** |
|---|---|---|---|---|---|
| astar-dist | 34.65 | 0.00 | 2.24 | 17.08 | 119.28 |
| PIGINet (BCE) | 18.67 | 0.04 | 4.92 | 18.60 | 51.12 |
| SPECTRE-adaptive (v1) | 22.93 | 0.00 | 9.20 | 29.52 | 53.00 |
| SPECTRE-static (v1) | 25.25 | 0.00 | 27.44 | 27.20 | 46.36 |
| **SPECTREv2-adaptive** | **13.68** | 0.00 | 4.60 | 26.20 | **23.92** |
| SPECTREv2-static | 19.12 | 0.00 | 4.44 | 32.64 | 39.40 |

**SPECTREv2-adaptive is best overall (13.68)** — beats PIGINet (18.67) and even the dd2d_v2
deployed 17.09 (v3 is *easier*: feasibility rose at every stratum) — and **dominates s3
(23.92** vs everyone else's 46–119) via proof-demotion + a now-appropriate long-first base.
Same qualitative shape as dd2d_v2 (v2 best, strong s3, weaker s2 — v2 s2 = 26.20 still trails
PIGINet 18.60).

**The prior artifact (why the first v3 run "reversed").** The 2026-07-24 rebuild showed
SPECTREv2-adaptive collapsing to 24.96 (s3 **85.52**, *worse than v1*) and PIGINet "winning" —
impossible on merit, since v2 has strictly more information than v1. Diagnosis
(`decisions.md` / [`notebook.md` 2026-07-25](notebook/README.md)): the pipeline was faithful (rescoring the *surviving*
original dd2d_v2 checkpoint reproduces 17.09 exactly) and the recipe/code byte-identical; the
failure was **training divergence into the short-first length shortcut** — the `--use-prior`
residual `[−index, −len]` over-biases cross-length ordering on the easier v3 data, buries the
(long) s3 feasibles, and the miscalibrated relrank selector then grabs an underfit epoch.
**Dropping the prior** (`--evidence --use-overlap`, val-justified: 16.9 vs 29.9) fixes s3 *and*
restores training convergence. Selection cross-check (robust to it): relrank-default best.pt =
13.68, deployed-val-FP epoch 14 = 15.88. PIGINet legitimately improved on the cleaner data
(v3 val AUPRC 0.43 vs dd2d_v2's 0.26).

---

## 4. Proposal (v2.2.1) vs as-built

### 4.1 Built as specified

The core of the proposal was executed faithfully:

- **Negative certificate** (§6.5, Task 0) — sound arrangement-complete certificate
  built from scratch on Shapely. *Nuance:* the scrapped `ttd_core` was
  deliberately **not** reused (user call); a Brunn–Minkowski `(√A − r√π)²` area
  term was added then **removed** (it upper-bounds eroded area → fabricated
  infeasibilities; caught by a tight-case battery the loose-only one missed).
- **Schema geometry/evidence layer** (§6.1) — trailing-nullable dataclasses +
  I5/I6 + migration shim, exactly as designed.
- **Object tags** (§7, P-A) — `assign_tags`, per-episode bijection, per-epoch
  permutation; the anti-collapse regression is green.
- **model_v2 token families + cross-attention scorer + aux head** (§7) — scene /
  candidate / fact / global tokens; per-candidate cross-attention; necessary/
  relevant aux head, small weight, ablatable.
- **Listwise PL loss** (§8) — v1's hardest-won decision, kept.
- **Proof/hint split + outside-the-net proof-demotion filter** (§5, §7) — proofs
  compile to demotion on the ranking, never the pool (P-E).
- **Live scramble gauge** (§8), **LAZY baseline** + **seed-checksum guard** (§9),
  and the **elimination ladder** (§10.3) — all present and driving their gates.

### 4.2 Changed / reworked (and why)

| # | Spec said | Built as | Why |
|---|---|---|---|
| 1 | **Harvest** replays the refiner's deepest bound prefix (`bound_plan`) into a fresh `DrawerWorld`; harvests refiner-trace facts incl. `extracted-ok`/`packed-ok`; writes `harvest_prefix` + a replayable `state_hash` (§6.2). | Harvest **reconstructs** facts from *stored geometry* (`blocked-at-contents`, `grasp-witness`, `pack-impossible`) + reads hints off *stored `refiner_metadata`* (`extraction-failed`, `pack-exhausted`). **No re-refinement.** Refiner-trace-only facts and `harvest_prefix`/`state_hash` are **deferred/empty**. | The definitive collection deliberately dropped `bound_plan` (decoupled the expensive certificate/harvest from the multi-hour collection). Re-refinement would require regenerating the scene — the exact bug the *reconstruct-don't-regenerate* rule forbids. The metadata hints (`failure_action` ≈ 93% `pick` at λ=0.8) are what keep the hint tier non-empty. |
| 2 | Step 4 picked **λ* = 0.5** (max oracle−GBDT gap). | **λ* = 0.8.** `choose_lambda_star` gained an `operating_range=(0.7, 0.95)` constraint. | 0.5 is off-design (tighter than DD2D was built for); at 0.5 stratum-3 is nearly ungenerable (~18 h for a balanced 125). 0.8 is the design default where s3 generates, feasibility is highest, and within-length degradation still holds. |
| 3 | Domain-flavored static scalars stay **harness-side** as null models the encoder must beat; schema-generic scalars *may* be model inputs (§7). | A **default-order prior** `[−index/K, −len/max_len]` is folded into the model as an **additive residual** (init-toward-default-order). **Data-dependent (2026-07-25): dropped for the deployed dd2d_v3 model.** | index/length are schema-generic planner signals (not DD2D geometry), so this is within the spec's allowance — and it realizes the P2 caveat ("fold the distance/default prior in as a feature") *early*. It was load-bearing for the dd2d_v2 length-bias fix, **but the prior is not universal**: its short-first bias over-biases cross-length ordering on the easier grasp-fixed dd2d_v3 (buries the long s3 feasibles → s3 collapse + training divergence), so the deployed v3 model **drops it**, chosen on val (§3.7; [`decisions.md` 2026-07-25](decisions/README.md)). |
| 4 | Computed overlap features: witness-overlap counts (max/mean), coverage flags, proven-dead flag, proven-prefix credit (§7). | Reduced to `[dead = subset ⊆ blocked, max-Jaccard-with-failed]`. | The `dead` flag (sound proof-demotion) + a mild Jaccard hint were sufficient to fix the evidence-harm; prefix-credit facts depend on the deferred trace harvest (#1). |
| 5 | — (not in the proposal) | **within-length PL loss**, **rollout-based difficulty-normalized selection**, and the **`demotion_source` observed/computed flag** were added. | The post-Step-11 length-bias arc. The within-length loss removes length as a shortcut cue; difficulty-normalized selection stops hard episodes dominating checkpoint choice; the observed default makes proof-demotion hard-coding-free (the computed predicate becomes an opt-in worth a measured ~14%). |
| 6 | **Wall-clock is the primary metric** (§10.1), FP secondary. | On DD2D, **FP is primary in practice**; wall-clock falls back to an EGE/`n_attempts` proxy. | The DD2D converter sets `refinement_wall_clock_s = 0` — per-skeleton refine time isn't in the raw JSON. Full timing would need a collector change. Disclosed; wall-clock remains primary on native-collection envs (RT2D). |
| 7 | `D_GLOBAL_IN = 6` reserves buffer dims + pool stats as a global token. | `_glob_feats` emits `[n_obj, k, mean_len, 0, 0, 0]` — the last 3 slots (buffer dims / mean-subset-size) are **zeroed**. | Minor incompleteness; the buffer geometry currently reaches the scorer only through the container token, not the global feature vector. |

### 4.3 Spec'd but deferred (+ now-completed / additions)

- **P2 / PIGINet low-level comparator** — **no longer deferred** (2026-07-20/25):
  PIGINet was retrained with BCE (paper loss), AUPRC-selected, on the λ=0.8 `dd2d_v2`
  data and again on grasp-fixed `dd2d_v3`, and is a live row in the comparison (§3.7;
  v3 val AUPRC 0.43 / AUROC 0.75). **Still pending:** the information-parity 2×2
  (v2.2-with-crops / PIGINet-with-polygons) — the same-input head-to-head.
- **VLMPlan zero-shot baseline** — **added** (2026-07-24/25, separate work): a
  zero-training-data VLM planner (`vlmplan/`, KinDER convention) occupying the data-axis
  endpoint, run on the dd2d_v3 test split (two model arms). Protocol: `decisions.md`
  2026-07-24; v3 test run: [`decisions.md` 2026-07-25](decisions/README.md). Not a v2.2 module — noted for the
  comparison landscape.
- **Step 12 — shape-family shift / P3** (the registered "larger evidence
  increment under shift" test): no experiment/test; only render-family fragments
  exist. User paused shift work.
- **Step 13 — second environment**: none of the v2.2 generalization-contract
  interfaces are ported to a not-designed-by-us domain yet. (RoutedTransport2D is
  in-repo but is the v2.1-era abstract substrate — no geometry layer — not the
  Task-6b second env.)
- **DAgger re-collection round** (§6.6) — budgeted in the proposal, not built.

### 4.4 Framing shifts already resolved *in the proposal* (not build changes)

For completeness: the **pre-mortem probe** was removed v2.1→v2.2 (provably a
post-mortem of the direct plan on DD2D), and the **percept/σ two-copy layer** was
removed v2.2→v2.2.1 (reduced to the §2 scope note). These are spec-level
decisions the build simply inherited.

---

## 5. Load-bearing constraints that emerged

Two constraints, implicit in the proposal, became explicit and load-bearing
during the build. They should be checked against every future design choice; the
full statement is [`consolidation_2026-07-19.md`](consolidation_2026-07-19.md)
§1. In brief:

- **Generalizability** — no hand-crafted per-environment predicate may be
  *load-bearing*. A domain computation is allowed only as an *opt-in increment*
  (hence the `demotion_source` flag, and the rejection of a `clears` predicate
  that "unlocked" performance but hid the learning question). The framework —
  proof/hint split, structural set-relation features, default-order prior,
  within-length loss, rollout selection — uses signals present in *any* TAMP
  problem. With an empty registry, everything is a hint and the ranker still
  learns.
- **Realism / a-priori-ness** — only two legitimate information sources:
  *a-priori* (plan length, enumeration order, ground-truth geometry) and
  *observed* (what the refiner reports as it attempts plans). The feasibility
  labels / oracle / stratum are **the answer** and are off-limits as inputs or
  test-time gates. (Stratum = minimal feasible plan length is a *property of the
  solution* — this is why stratum-gating was rejected.)
- **Per-dataset validation of optional components** *(surfaced 2026-07-25)* — a
  component or hyperparameter tuned on one distribution can *hurt* another. The
  default-order prior helped on dd2d_v2/RT2D but collapsed s3 on the easier grasp-fixed
  dd2d_v3; the fix was to **re-select it on validation per dataset**, not carry it over.
  Optional increments (the prior, `demotion_source`, `wl-weight`) are knobs to validate,
  not constants — and checkpoint selection must be validated too (relrank is miscalibrated
  on dd2d_v3).

Supporting rules: **reconstruct, don't regenerate** (post-hoc geometry from
stored poses, never from the seed); **the net weights sound relational features,
it does not learn them** (set-containment is a universal-AND attention
approximates poorly, and soundness needs the exact test); **listwise PL only**
(global + within-length buckets, never BCE); **1-seed to iterate, ≥3 to report.**

---

## 6. Current state & open threads

**Deployed ranker** = `v2-evidence + within-length + overlap + proof/hint split`,
scored by the model then filtered by proof-demotion (`evidence.deployed_rollout`). Best
method overall on both collections [1-seed dev]; **s2** is the open weak stratum
(intrinsically hard). Everything in the deployed stack is domain-agnostic; the geometry
predicate is an opt-in flag worth ~14%. **The default-order prior is dataset-dependent:**
*on* for **dd2d_v2/RT2D** (`_v2_evidence_prior_ov`, 17.09 on dd2d_v2), *off* for the
easier grasp-fixed **dd2d_v3** (`_v2_evidence_ov`, **13.68**, s3 fixed — §3.7).

**Checkpoints** (gitignored). dd2d_v2: `checkpoints_v2` (static), `_v2_prior`,
`_v2_evidence`, `_v2_evidence_prior`, `_v2_evidence_prior_ov` (observed demotion —
**deployed for dd2d_v2**), `_v2_evidence_prior_ov_comp` (computed demotion). dd2d_v3:
`checkpoints/dd2d_v3` (v1), **`checkpoints_v2_evidence_ov/dd2d_v3` (no-prior — deployed
for dd2d_v3)**, and `checkpoints_v2_evidence_prior_ov/dd2d_v3` (the artifactual with-prior
run, kept for reference). PIGINet: `envs/dd2d/out_dd2d/piginet_bce{,_v3}/ckpt.pt`.

**Selection caveat.** `_val_relative_rank` (relrank) is *miscalibrated on dd2d_v3* — it
never drops below 1 (≈random) even for the good no-prior model — but is safe once the
destabilizing prior is dropped (the model then converges, so relrank lands on a converged
epoch); a deployed-val-FP cross-check confirms (§3.7).

**Open threads** (priority order):
1. **3-seed validation** of the deployed dd2d_v3 numbers (and the prior's
   v2-helps/v3-hurts data-dependence) — the cheapest high-value consolidation; a
   precondition for any writeup number.
2. **Second environment** — the real generalization test (needs the §11
   generalization-contract interfaces).
3. **s2 / scene-conditional regime detection** — let the ranker gate its
   length-preference on a *learned*, a-priori-legal difficulty estimate (this would also
   subsume the manual prior-on/off dataset choice).
4. **Wire deployed-val-FP checkpoint selection into `train_v2`** so relrank's
   miscalibration cannot mis-select (currently done offline).
5. **P2 information-parity 2×2** (v2.2-crops / PIGINet-polygons) and **shift tests (P3)**
   — still deferred (PIGINet itself is now run, §4.3).
6. Wire `deployed_rollout` into the main-table/eval scripts so the deployed
   numbers become the canonical reported ones.

---

## 7. What works, what doesn't, what we learned

A consolidated log of the iteration on the SPECTRE model. Detailed run numbers are in
`notebook.md`; the per-decision rationale is in `decisions.md`.

### 7.1 The arc

**v1** was a plan-*length* lookup (η²(length) = 1.00), object- and geometry-blind — anonymous ids
and no geometry made two same-length skeletons staging *different* subsets identical inputs. **v2.2**
rebuilt the representation: episode-local tags (the join key), object-centric geometry, a
Set-Transformer relational scene encoder, typed post-mortem evidence, and sound proof-demotion
*outside* the net. A post-Step-11 **"make it good and generalizable"** arc then fixed a length bias
that made the learned model win s3 but lose the easy strata. Finally the DD2D grasp model was made
realistic and the data re-collected (**dd2d_v3**); an apparent "PIGINet wins / v2 collapses at s3"
reversal turned out to be a **training artifact of the short-first prior on the easier data** — fixed
by dropping the prior (§3.7).

### 7.2 What works

- **Episode-local tags** dissolved the v1 collapse — η²(length) 1.00 → 0.23; same-length skeletons
  staging different subsets are now distinguishable (P-A).
- **Object-centric geometry + the Set-Transformer scene encoder** (footprint descriptors + 2× SAB
  relational join) — v2-static beats v1-static and carries genuine *within-length* signal (AUROC
  0.585–0.673 on dd2d_v3, vs v1-static's ≈ chance).
- **Listwise Plackett–Luce loss** — rollout-aligned (−log P(argmax feasible)); v1's hardest-won
  decision, kept.
- **Within-length PL loss** — removes plan length as an exploitable shortcut so geometry must decide
  feasibility within a stratum.
- **Proof/hint split + sound proof-demotion** — proofs compile to a finite-offset demotion on the
  *ranking* (never drops the pool; a wrong proof only reorders — P-E); structural, not learned. This
  is what lets the deployed adaptive ranker beat its own static mode and win s3.
- **Observed demotion (default)** — the "blocked" signal is read from the refiner's own failure
  report (no geometry predicate) → hard-coding-free and generalizable; effectively sound (1/6376
  edge case).
- **The deployed adaptive ranker is best overall on both collections** — dd2d_v2 17.09 (with prior),
  dd2d_v3 13.68 (without) — beating astar, PIGINet, and v1.
- **The pipeline-faithfulness debugging move** — rescoring the *surviving* original checkpoint
  reproduced its published number exactly, isolating the dd2d_v3 problem to *training* (not
  scoring/code). A reusable diagnostic.

### 7.3 What doesn't work / weak spots

- **Pointwise BCE** — killed Attempt 2; mis-aligned with time-to-first-success. The ranker is
  PL-only (the aux head's weight-0.2 BCE is the only BCE anywhere).
- **A per-environment `clears` predicate** — rejected: it "unlocked" performance but is hand-crafted
  geometry that hides the learning question (the generalizability constraint, §5).
- **Un-split typed evidence** — consumed as a crude "prefer longer" cue, it *harmed* s1 (+23.8 FP)
  until the proof/hint split routed proofs structurally.
- **The short-first prior is data-dependent** — load-bearing on dd2d_v2/RT2D, but on the easier
  grasp-fixed dd2d_v3 it over-biased cross-length ordering, buried the long s3 feasibles, and caused
  a training divergence + s3 collapse. Dropped for v3 (§3.7): a component that helps one distribution
  can hurt another.
- **`relrank` checkpoint selection is miscalibrated on dd2d_v3** — it never drops below 1 (≈ random)
  even for the good model; it is safe only once the destabilizing prior is removed (the model then
  converges). A deployed-val-FP selection is the robust fix (still done offline).
- **s2 is the persistent weak stratum** — an intrinsically hard needle-hunt (~2% of size-2 plans
  feasible, extraction/packing-limited); a ~17-FP ceiling for *all* methods, learned or not.
- **Known code incompletenesses** (surfaced auditing the model for this doc): `glob_feats` leaves its
  buffer-dimension slots hardcoded `0` (buffer geometry reaches the scorer only via object tokens);
  `exclude_marginal` is effectively inert (a marginal fail's label becomes `False`, not dropped from
  the PL denominator); the `pack-impossible` superset-demotion path exists but is never triggered at
  deployment (`observe_failure` is always called with `pack_impossible=False`); `Fact.scalars`
  (depth/samples) are not consumed by the tensorizer.
- **Everything is 1-seed dev** — the deployed numbers, the prior's data-dependence, and the s2 gap
  all await a ≥ 3-seed reproduction before they are writeup-reportable.

### 7.4 Hard-won principles (full statements in §5)

- **Generalizability** — no hand-crafted per-env predicate may be *load-bearing*; domain
  computations are opt-in increments only.
- **Realism / a-priori-ness** — only *a-priori* (length, order, ground-truth geometry) and *observed*
  (what the refiner reports) are legitimate inputs; feasibility labels / oracle / stratum are the
  answer and are off-limits.
- **Reconstruct, don't regenerate** — post-hoc geometry is computed from stored poses, never
  re-derived from the seed, so a proof can never contradict a label.
- **The net weights sound relational features; it does not learn them** — set-containment is a
  universal-AND attention approximates poorly, and soundness needs the exact test.
- **Validate optional components and checkpoint selection per-dataset, on val** — the prior and
  `relrank` both looked universal until dd2d_v3 proved otherwise.
- **Listwise PL only; 1-seed to iterate, ≥ 3 to report.**
