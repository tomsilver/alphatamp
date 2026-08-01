# SPECTRE Decisions — StickButton2D as a second environment

3 entries, 2026-08-01 .. (OPEN — new entries go here). Newest first.
Index and cross-reference tables: [README.md](README.md).

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

