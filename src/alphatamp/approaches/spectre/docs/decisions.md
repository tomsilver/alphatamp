# SPECTRE — Decision Log (ADR style)

One entry per consequential decision. Newest first. Format: context → decision
→ consequences. Refactor-era entries record what moved, what deliberately did
not, and why.

---

## 2026-07-18 — v2.2.1 schema geometry/evidence layer: additive + migration shim

**Context.** v2.2.1 needs ground-truth object geometry (for the geometry-aware model) and
typed post-mortem evidence carried on the episode records, without breaking the many
existing RT2D/kinder `EpisodeRecord` pickles or the abstract-first pipeline.

**Decisions.** (a) **Additive, trailing, nullable fields** — `EpisodeRecord.{scene_geometry,
aux_labels}`, `OutcomeRecord.post_mortem`, plus new frozen dataclasses (`SceneGeometry`/
`ObjectGeometry`/`ContainerGeometry`/`Fact`/`PostMortemRecord`/`AuxLabels`) — all default
`None`/empty, so every existing construction site and RT2D/kinder record round-trips
unchanged. New invariants I5/I6 are **guarded** (fire only when the field is present).
(b) **Load-time migration shim over global regeneration.** Frozen-dataclass pickles restore
via `__dict__` and skip `__init__`/`__post_init__`, so pre-v2.2.1 pickles lack the new
attrs (→ `AttributeError`). `io.load_episode` fills the defaults via `object.__setattr__`.
Chosen over bumping a schema version + regenerating all corpora: RT2D data lives only on
other machines and need not be re-collected; DD2D is re-collected regardless (for
post-mortems). (c) **Geometry is a converter/`record_ext` change, not a raw re-collection**
for the *geometry* part — the DD2D JSON already had pose/shape; `record_ext` now also writes
the item-frame `boundary` ring and `spectre_convert` reads it (`CONVERTER_VERSION`
`v1→v2`). A dir predating `boundary` yields `scene_geometry=None` (abstract-only). The
**abstract STRIPS state stays x0-free**; geometry rides on `scene_geometry`, never in the
atoms.

**Consequences.** On-disk format grew (optional) fields; `test_schema_v2_geometry.py`
covers round-trip + I5/I6 + legacy-pickle migration; the `dd2d_convert` version-pin test
updated to v2. No change to the model/loss/training yet — this is the data-layer
foundation Steps 5 (post-mortem population) and 8 (geometry model) build on. `notebook.md`
2026-07-18.

---

## 2026-07-18 — DD2D arrangement-complete negative packing certificate (v2.2.1 Task 0)

**Context.** v2.2.1 makes completing the arrangement-complete negative certificate the
**blocking Task 0**: until packing-infeasibility can be *proven* (not left provisional
`marginal(budget)`), no label-dependent DD2D number is trustworthy. The spec is
`dd2d_spec.md` §8.4 + P16/P19. The repo already contained an NFP/IFP/nesting packing
substrate (`ttd/ttd_core/`), but the user directed that **ttd is scrapped** (still in the
tree only because it hasn't been removed) and must not be reused.

**Decisions.**

(a) **Build the certificate from scratch on Shapely** (`envs/dd2d/dd2d/certificate.py`),
not on `ttd_core`. Rejected reusing `ttd_core.nesting`: user call, and independently its
`INFEASIBLE` is discretization-relative (fixed 1°/5° grid, not the Lipschitz grid) and its
`packs()` folds `TIMEOUT` into `False` — unsound as a proof.

(b) **Exact convex decomposition via `shapely.constrained_delaunay_triangles`** (Shapely
2.1+). Each triangle ⊆ the shape and the triangles exactly cover it → exact NFP for the 3
**concave** families (`banana`/`shoe`/`dumbbell`). Rejected the plain-Delaunay-of-vertices
fallback (what sank ttd's concave path): it need not respect a reflex boundary → NFP wrong
→ possible false infeasible. `convex_parts` verifies exact cover and refuses (→ marginal)
otherwise.

(c) **All placement orders, not a fixed order.** A single fixed-order sequential DFS is
**not** sound: the first item's free region is the whole IFP, whose only vertices are the
container corners, so interior-only packings are unreachable → false infeasible.
Soundness argument used instead: bottom-left-compact any packing; the most-bottom-left
item is pinned into a container corner (an IFP vertex) and inductively each item in BL
order lands on a free-region vertex — so the BL order (∈ all orders) reaches it. We only
attempt the full all-orders exhaustion for `|S| ≤ MAX_ORDER_ITEMS = 5`; larger subsets no
area bound settles fall to `marginal`, never a partial-search `infeasible`.

(d) **Remove the Brunn–Minkowski area term; keep H1 on exact deflated areas.** A
`Σ(√Aᵢ − (δ/2)√π)²` bound was added then removed: for fixed original area the disk
*maximises* eroded area (isoperimetric), so that expression is an *upper* bound on the
deflated area → it overestimates packed area and **fabricates infeasibilities on tight
buffers**. Since the DFS already computes the exact δ/2-deflated polygons, H1 on their
exact areas is the tightest sound area bound. Process lesson recorded: the
zero-false-infeasible battery **must include tight/near-threshold cases** — a loose-only
battery hid this bug.

(e) **INFEASIBLE only on full exhaustion; timeout ⇒ `None` (marginal, reason=budget),
never infeasible.** Budget = P19 (5 s / 1e5 EGEs). A degenerate δ/2-deflation (thin shape
vanishes) also ⇒ `None`. The verdict is three-valued: `True` (proven infeasible), `False`
(a packing was found ⇒ not infeasible), `None` (undecided → stays marginal).

(f) **Integrate behind a `use_certificate` flag, default off.** `label.py`'s
`label_candidate`/`label_all` gain the flag; it is off inside `generate_dd2d_problem`'s
rejection-sampling loop (where the certificate is called hundreds of times and only the
feasible labels — unaffected by it — drive strata/F3) and on only for authoritative
once-per-candidate labeling. On-by-default hung the DD2D suite. On a `True` verdict the
`marginal(budget)` becomes proven `infeasible(packing)`; on `False` (a packing exists) it
is reclassified `marginal(inaccessible)`; `None` stays `marginal(budget)`.

**Consequences.** Sound: 0 false-infeasible over ~730 constructed-feasible packings (loose
+ tight, concave + circles, |S|=2–4); 16 new `test_certificate.py` tests + 49 DD2D + 259
spectre tests green. At λ=0.8 the certificate proves 0 packing-infeasibles (infeasibility
is extraction-dominated at loose λ) and reclassifies all budget-marginals it saw to
`inaccessible` (they pack). The real-scene *tight-λ* proven-infeasible characterization is
deferred to **Step 4**'s λ-sweep (generation at tight λ is slow). No change to the SPECTRE
model/loss/pipeline; the certificate is a labeler-side soundness upgrade. Applying it to
stamp the SPECTRE training labels (refiner-`fail` outcomes) at collection time is wired in
**Step 5**. `notebook.md` 2026-07-18 has the numbers.

---

## 2026-07-18 — Modernize + pin the substrate deps so a fresh machine resolves

**Context.** Development moved from a MacBook M3 Pro (CPU/MPS) to a new Ubuntu
26.04 workstation (RTX 5090, Ryzen 9 9950X, 64 GB) for GPU training. A fresh
`uv pip install -e ".[develop,ttd]"` on the new box **failed to resolve** — the
root `pyproject.toml` pinned `kindergarden[kinematic2d]==0.0.8` but left the
`kinder-baselines` and `bilevel-planning` git sources **unpinned** (no `rev=`).
Both pins were introduced together on 2026-03-22 (`62d3784`) when compatible, but
upstream then drifted: kinder-baselines bumped to `kindergarden>=0.1.0`
(2026-04-29) and later `bilevel-planning>=0.1.4`, and **dropped the
`kinematic2d` extra** entirely (kindergarden 0.2.0 has no such extra). With no
lockfile, a fresh resolve pulls the drifted HEAD and conflicts; the MacBook only
still works because its venv was resolved months ago and cached — never
re-resolved. User chose "modernize + pin" over reproducing the MacBook's exact
(unpinnable, un-frozen) set or pinning to the ~4-month-old compatible commit.

**Decisions.**

(a) **Bump the whole prpl-mono substrate to one coherent commit `e215d1fc`**
(was `df145d5c` for `relational_structs`/`prpl_utils`/`prpl_llm_utils`/
`tomsgeoms2d`; `bilevel-planning` was previously an *unpinned* prpl-mono source).
prpl-mono is one monorepo — mixing commits across its subpackages is what causes
API breakage — so all five move together. `e215d1fc` provides
`bilevel-planning==0.1.4` (satisfies kinder-baselines HEAD). Added an explicit
`rev=` to the `bilevel-planning` source so it can no longer drift against the
pinned `relational_structs`.

(b) **Bump `kindergarden` 0.0.8 → 0.2.0 and drop the `[kinematic2d]` extra.**
The extra no longer exists (kinder-baselines PR #77 "drop dead kindergarden
extras"); kinder packages now depend on bare `kindergarden>=0.1.0`. `pymunk` (the
kinematic2d substrate) is already a direct alphatamp dep, so dropping the extra
loses nothing.

(c) **Pin both kinder-baselines sources to HEAD `4c731dc8`** (was unpinned) —
`kinder-bilevel-planning` and `kinder-models`, for reproducibility.

**Consequences.** Fresh resolve succeeds; **all spectre tests pass** (254 incl.
slow), spectre mypy clean, spectre pylint 10.00/10 — i.e. the substrate bump did
not break spectre. torch is the cu130 build (`2.13.0+cu130`, see the spectre
`CLAUDE.md` compute-resources note), GPU-verified on the RTX 5090 (sm_120). The
`pyproject.toml` `torch` requirement is left **unpinned** on purpose (the cu130
index is applied at install time, not baked in, so SLURM/other machines are
unaffected). **Reproducibility caveat:** prior spectre results/checkpoints were
produced on the older MacBook substrate (kindergarden 0.0.8 / prpl-mono
`df145d5c`); numbers regenerated on this box use the newer substrate, so
re-verify before comparing across the boundary.

**Follow-up (2026-07-18) — restore `run_ci_checks.sh`.** Two further repo-wide
fixes (not spectre-specific) so the CI script runs on a fresh machine:
(1) **capped `pytest>=7.2.2,<8`** in the `develop` extra — the fresh resolve pulls
`pytest 9.1.1`, but the latest `pytest-pylint` (0.21.0) uses the `path` collect
hook removed in pytest 8.0, INTERNALERRORing `pytest . --pylint`; `<8` (→ 7.4.4)
is the working bound (the upstream kinder-baselines `<9.1` cap is *not* enough).
(2) **excluded the untracked `kb/` sibling checkout** (a local knowledge-base clone
of kinder-baselines & friends, with its own `.git`) from git and every CI tool —
`.gitignore /kb/`, `run_autoformat.sh` docformatter `--exclude`, `[tool.isort]
skip_glob`, `[tool.mypy] exclude ^kb/`, `[tool.pytest] norecursedirs`, `.pylintrc
ignore` — mirroring how `.venv/`/`archive/`/vendored-dd2d are handled; otherwise
`black .`/`isort .`/`mypy .`/`pytest . --pylint` descend into it. After these,
autoformat + pylint (217 pass) are clean and spectre stays fully green.

**Sibling-project failures surfaced by the modern toolchain/CUDA (NOT spectre) —
resolved least-invasively to get full CI green (user call: skip, don't deep-fix
other projects' internals):** (1) the **pre-existing** `mypy` error in
`experiments/collect_data.py:67` — `render()`'s return type widened under newer
gymnasium/mypy — fixed by annotating the local `frame: Any` (matches the method's
own `-> Any`). (2) 4 `simfree_param_policy` tests raise a `cuda:0`-vs-`cpu`
mismatch on a GPU box (pass CPU-only) — marked `skipif(torch.cuda.is_available())`
via a shared `_SKIP_ON_CUDA` marker; the real fix is to thread a device through
that approach. (3) `practice_makes_perfect` fails **device-independently** (CPU
too) with `AbstractPlanGenerationError` under the new substrate — marked
`xfail(strict=False)` with that reason; needs a genuine sibling-project fix, not a
skip. **Autoformat churn:** the newer `docformatter` rewraps ~8 tracked docstrings
(committed so the tree stays autoformat-clean; re-verified idempotent). After all
this, `./run_ci_checks.sh` is green end-to-end (mypy 0 / pylint 217 pass / pytest
269 pass, 11 skipped, 1 xfailed) and spectre itself is untouched by these
sibling-only changes.

---

## 2026-07-12 — DD2D integration: JSON→EpisodeRecord converter, not a native env

**Context.** DD2D (Drawer Decluttering 2D) was migrated in-package under
`envs/dd2d/` with an already-collected PIGINet-style dataset
(`data/dd2d/raw_v2/{train,val,test}`, 425/120/124 problems) and its own
generation pipeline (`envs/dd2d/dd2d/collect.py`). Goal: make DD2D usable as a
SPECTRE problem/dataset, keep the ability to generate more, and start training
SPECTRE on it. Key structural fact: SPECTRE's training path
(`dataset.py`→`vocab.py`→`train.py`) consumes *only* serialized `EpisodeRecord`
pickles — the `SesameModels`/gym/refiner machinery exists solely so `collect.py`
can *generate* episodes from a live sim. And each DD2D problem directory (200
`NNN.json` candidate skeletons over a shared objects/init/goal, each with a
feasibility `label`) already *is* a SPECTRE episode.

**Decisions.**

(a) **Converter, not a native SPECTRE env.** Wire DD2D by converting its JSON to
`EpisodeRecord` (`envs/dd2d/spectre_convert.py`), reusing the entire downstream
pipeline unchanged. Rejected building `create_dd2d_models`/gym stub/closed-form
generator/refiner adapter + `collect.py` dispatch branches: DD2D's refiner is a
geometric packing solver that does not fit the controller-sampler contract, and
fresh generation is already served by `envs/dd2d/dd2d/collect.py`. New data =
run that collector → re-run the converter. Far less code, no re-derivation of
DD2D geometry into the substrate.

(b) **Abstract-only for v1 (x₀-free).** The converter keeps only the six drawer
STRIPS predicates and drops the DD2D `at-pose`/geometry literals — SPECTRE is
deliberately x₀-free. Continuous poses/shapes/sizes remain in the source JSON for
a future x₀-conditioned comparator (proposal §6), not wired now. Consequence:
DD2D is expected to be a **negative control** — abstract-first drops exactly the
packing signal feasibility depends on. Confirmed at epoch 0: AUROC(3) < AUROC(0)
(`notebook.md` 2026-07-12).

(c) **One variant `dd2d_v2` spanning all item counts.** DD2D problems mix
n∈{10..13} within a split; the architecture factors across object counts (typed
local ids, set pooling), so a single variant is natural. The single object type
`item` is fully augmentable (target marked by the `target` predicate, not
identity): `env_registry._TYPE_AUG_POLICIES["dd2d_v2"] = {"item": True}`, no
static-tag stream.

(d) **Label caveat (blocking for research numbers, not for training).** DD2D's
Day-1 labeler marks non-area-proven negatives as *marginal*, not
proven-infeasible (`MIGRATION_DD2D.md` §4); the converter maps `label==false`→
`"fail"` for training, but no label-dependent SPECTRE number is reportable until
the arrangement-complete negative certificate lands.

**Consequences.** New: `envs/dd2d/spectre_operators.py`,
`envs/dd2d/spectre_convert.py`, `experiments/spectre/dd2d_convert.py`,
`conf/dd2d_convert.yaml`, `conf/env/dd2d_v2.yaml`,
`tests/approaches/spectre/test_dd2d_convert.py`; one `env_registry.py` entry.
No change to the model, loss, F-subset discipline, or rollout-based selection —
DD2D flows through the exact same `EpisodeRecord` schema as RT2D/kinder. Verified
end-to-end: 669 episodes converted (0 failures), vocab (3 ops/6 preds/1 type,
OOV-clean), pipeline check + 1-epoch train run pass.

The **vendored DD2D env code** (everything under `envs/dd2d/` except the
`spectre_*` adapter files) is excluded from strict `mypy` (`pyproject.toml`
`[tool.mypy] exclude`) and `pylint` (`.pylintrc` `ignore-paths`) via a
`(?!spectre_)` negative lookahead — it arrived from `envsearch` with 100+
pre-existing type errors and is treated like `lib/` vendoring, while SPECTRE's
own adapter stays fully checked. Open follow-up (not decided here): whether to
let `run_ci_checks.sh`'s repo-wide `black .`/`isort .` normalize the vendored
tree once, or exclude it from formatting too.

---

## 2026-06-25 — Direction pivot: from adaptive reordering to a representation question

**Context.** The project's headline had been *adaptive test-time reordering* (the
SPECTRE re-ranker), evaluated on the bespoke RT2D env against the adaptive
baseline B4. Two results undercut that as the lead: the Ψ-ablation (`notebook.md`
2026-06-06) attributes only **~27%** of SPECTRE's margin over B4 to
failure-conditioning — the **static** Φ+σ representation carries ~73%; and the B6
DP-on-counts sweep (`notebook.md` 2026-06-11) showed lookahead over the count
model is **small, fragile, and saturated**, i.e. not the missing ingredient. A
structural reading of RT2D explains why, and motivates a reframe. Each decision
below is recorded with rationale.

**Decisions.**

(a) **Reframe adaptivity-primary → representation-primary.** The contribution is
now a *representation question for plan-feasibility prediction in
fully-observable (FO), deterministic bilevel TAMP*: what should a feasibility
predictor represent skeletons/problems over? Rationale: the empirics put the
margin in the static representation, not the failure-conditioning. (See
`proposal.md` §0.)

(b) **Demote SPECTRE/reordering to a secondary, composable increment.**
Within-episode failures carry free instance-specific signal, but it is a minority
of the margin; treat the re-ranker as orthogonal to — and combinable with —
whichever representation wins, not as the headline.

(c) **RT2D was effectively partially observable *to the policy* — a mislabeling.**
RT2D was described as FO+deterministic, but the policy π was denied x₀ and the toy
three-gate refiner had **privileged access to the latent z**. To the policy the
problem was therefore effectively partially observable, and the discrete gating
latent had to be **manufactured** — which is why RT2D felt contrived. Rationale:
record so we do not re-derive the bespoke env as if it were a faithful FO TAMP
instance.

(d) **The no-x₀ design was a handicap *in RT2D* — but the nuance matters.**
PIGINet's own ablation shows x₀ carries real signal **in their kitchen
problems**, and PIGINet already works at **150–600 problems**, so the
*data-efficiency* rationale for dropping x₀ does not hold *universally*. This does
**not** establish that x₀ must always be included: whether dropping low-level
state is a helpful abstraction is **domain-dependent**, and there may be problems
where it helps. We are **not committed** either way — the x₀ stance is
experiment-driven. Rationale: avoid over-correcting from "drop x₀ always" to "keep
x₀ always"; both are empirical questions.

(e) **The FO information-ceiling bounds the adaptive component's value.** In
FO+deterministic TAMP, the within-episode refinement failures add **no
information beyond x₀** at the predictor's ceiling (the outcome of every skeleton
is a deterministic function of x₀). This is the structural reason the adaptive
signal is small here, and the structural reason for the pivot. Rationale: it
makes the ~27% finding expected, not a defect of Ψ.

(f) **Reinterpret the 27% finding as "the static representation does the work."**
The ablation is now read as positive evidence for the representation thesis: most
achievable gain is captured by the static ranking, with online updating a small
add-on — consistent with (e).

(g) **Adopt the efficiency/representation framing + crossover prediction +
negative control.** The claim is **efficiency / perception-lightness, not
information access**: under FO+determinism no representation beats an ideal
low-level predictor on information grounds. **Falsifiable prediction (a
hypothesis, not a result):** a *crossover* — in the low-data / weak-perception
regime a well-chosen (richer-than-pixels, cheaper-than-full-state) representation
matches or beats a low-level PIGINet-style predictor on downstream planning
efficiency, while the low-level predictor regains its edge with abundant data +
strong perception. **Negative control:** dense-packing / fine-continuous-fit
domains, where any compressed representation is expected to lose, bound the
claim. *Abstract-first* is the current leading candidate but only one point in a
design space (learned latents, object-centric/graph features, intermediate
symbolic+coarse-geometric states, invented predicates), and may prove too lossy.

(h) **Prefer pre-existing environments that meet a hypothesized-advantage
property wishlist; keep bespoke in scope.** We prefer pre-existing envs *only if*
they exhibit properties we expect to favor a relational/abstract representation,
and keep **bespoke, hand-crafted** envs in scope where they better expose the
advantage. The (open, evolving) property list: (1) feasibility governed by
relational structure the abstraction captures; (2) low-level state
high-dimensional/distracting or hard to extract relational structure from; (3)
perception genuinely limited or costly; (4) object-count/identity generalization;
(5) long horizon / large diverse pool. Planned homes: PIGINet kitchens with
degraded perception, and Khodeir clutter/distractor domains augmented with a
low-level baseline, swept over perception-degradation × training-set size.
Primary metric time-to-first-success; secondary time-to-k. Rationale: the
property combination needed for the *adaptive* claim (shared, refinement-decidable,
instance-specific gating) is rare in pre-existing benchmarks, but the
*representation* claim has real pre-existing homes.

(i) **Freeze the April writeup.** `archive/SPECTRE_WRITEUP_APR_2026.md` is frozen
with a banner (2026-06-25); it reflects the adaptive-reordering framing and is
retained as historical record. Rationale: the living docs (`proposal.md` §0,
this log, `notebook.md`) are the source of truth and must not defer to the frozen
snapshot.

**Consequences.** `proposal.md` now leads with §0 (representation-first), with the
original §1–§6 retained byte-unchanged under "Superseded framing (April 2026)";
`research_lit.md` reframes PIGINet as the low-level static predictor we compare
against and adds a representation lens; `notebook.md` 2026-06-25 records the
reinterpretation and the forward sweep. The RT2D env, the SPECTRE model, and the
B1–B6 baselines/code are **unchanged** — this is a framing pivot, not a code
change. No planner/refiner/abstraction change. What survives intact: the
rollout-based model-selection discipline, the PL loss, the F-subset discipline.

---

## 2026-06-11 — B6 higher horizons: incremental scoring, top-m pruning, no capping

**Context.** B6's per-decision cost is `O(K^{h+1})` (the `O(K^{h−1})` backup tree
× the `O(K²)` re-conditioning leaf), and every RT2D-n3 pool is exactly K=30, so
the exact search was ~12 min at `h=3` and intractable at `h≥4`. The goal was to
reach higher horizons (does the lookahead premium keep growing, or saturate?)
without distorting the evaluated problem.

**Decisions.**

1. **Reject pool capping (train or eval) as the tractability lever.** Capping
   the candidate pool to the first `K_cap` planner-ordered skeletons was
   considered and rejected. *Eval-capping* changes the evaluated problem: the new
   `solvability_at_cap` diagnostic shows RT2D-n3 successes sit at **every**
   planner depth (test solvable@15 ≈ 0.46, @20 ≈ 0.60, reaching ~1.0 only at
   k=30), so capping below 30 censors real successes. *Train-capping* (a briefly
   adopted "symmetric cap" idea) deletes estimator observations for no benefit —
   the per-key/pairwise NB estimands are properties of the data distribution, not
   of a skeleton's pool position, so the rising OOV it induced was self-inflicted.
   **The q-model is always fit on the full train pools, and eval always uses the
   full K=30** (uncensored-eval discipline, 2026-06-07). A `candidate_cap` knob
   was *not* added.

2. **Incremental Naive-Bayes scoring is the real lever.** The leaf was secretly
   `O(K³)`: each re-conditioning rollout step appends a failure, so every
   `(candidate, F)` score was a fresh `Σ_{k'∈F}` recompute. The search now threads
   a scoring context (`dp_on_counts._Ctx`) that extends per-candidate
   `S_succ`/`S_fail` by **one pairwise term per failure edge** (`O(K)`), turning
   the leaf into `O(K²)`. Measured on RT2D-n3 (exact, unpruned): h=2 **93 s → 9 s**,
   h=3 **740 s → 86 s**, h=4 from intractable to **~minutes** — all reproducing the
   exact attempt counts bitwise. Scores use `np.log` so the incremental `S_succ`
   equals `_adaptive_score` bitwise, preserving the `h=1 ≡ B4` identity. Synthetic
   test models without the primitives fall back to the recompute closures; an
   equivalence test pins the two paths together.

3. **Top-m pruning kept as an optional knob, off by default.** With incremental
   scoring making the *exact* search tractable through h=4, `dp_on_counts_baseline`
   now **defaults to `m=None` (exact)** — a deviation from the in-flight plan's
   `m=12`, justified because exact is now affordable and pruning is lossy (m=12
   cost ~0.09 attempts at h=3: 9.24 vs 9.15, for only ~2×). Pruning remains
   available (`m=12`) to push `h≥5`: it restricts the `min` at each **internal**
   lookahead node to the top-m candidates by greedy index; the **root argmin and
   the leaf walk are never pruned**, so which skeleton may actually be attempted
   is unrestricted and `h=1` is untouched. Guarded by an `m≥K`-equals-unpruned
   exactness test.

**Consequences.** New `eda.solvability_at_cap` (+ notebook figure gating any
capping). `dp_on_counts.py` rewritten around `_Ctx` (incremental + closure
backends) with a `m` pruning width; `eda._build_dp_model` supplies the
incremental primitives (`log_succ`/`log_fail`/`delta` with a shared delta cache).
Exact h-sweep numbers and paired stats: see `notebook.md` 2026-06-11. Belief-MDP
-over-z and any planner/refiner/abstraction change remain out of scope.

---

## 2026-06-08 — DP-on-counts (B6): lookahead skeleton-selection baseline

**Context.** B4 (Adaptive Historical) is the headline non-learned adaptive
baseline, but it is *myopic*: at each step it picks the single skeleton with the
highest Naive-Bayes success score and never reasons about what the resulting
failure set leaves for later. We wanted a fair, count-only baseline that shares
B4's estimator but adds multi-step lookahead — to bracket how much of any
SPECTRE-vs-B4 gap is "B4 is myopic" vs "B4's count model is weak". The method is
a receding-horizon expectimax over the cost-to-first-success recursion
`V(F) = min_{σ∈R}[c(σ) + q(σ|F)·V(F∪{σ})]`, solved online to depth `h`. It is a
**baseline** (B6), not SPECTRE, and touches no planner/abstraction/refiner.
Several modelling choices were load-bearing and non-obvious.

**Decisions.**

1. **Base-policy depth indexing (h=1 ≡ B4 exactly).** `h=1` is the no-lookahead
   base greedy policy `argmin index(σ|F)`; `h≥2` is
   `argmin_σ[c(σ) + q(σ|F)·W_{h−2}(F∪{σ})]`. The literal one-step backup
   `argmin[c + q·V̂_0]` is a *policy-improvement* of B4, not B4 (counterexample:
   a candidate with marginally lower fail-prob but a much worse continuation gets
   displaced) — so the only way to honour "h=1 reproduces B4 exactly" is to make
   `h=1` the base policy and have `h` count improvement steps above it. Default
   `h=2` = one real lookahead level.

2. **Calibrated two-class NB posterior for `q` (reject `1−clip(exp(score))`).**
   B4's score `S_succ = log p̂(k) + Σ_{k'∈F} log[p̂(k|k')/p̂(k)]` exponentiates to
   an *unnormalized* NB score that exceeds 1 for `|F|≥2`; `1−clip(exp(S_succ))`
   would force `q=0` (a guaranteed success) precisely when conditioning is most
   informative, confounding the h≥2 regime the baseline exists to probe. Instead
   `q(σ|F) = σ(S_fail − S_succ)` with the complementary
   `S_fail = log(1−p̂(k)) + Σ log[(1−p̂(k|k'))/(1−p̂(k))]` — a proper posterior in
   `(0,1)`, no clip. B4's *ranking* still uses the raw `S_succ`, so this does not
   change B4 or the B6 `h=1` selection.

3. **Re-conditioning greedy leaf `W_0 = V^base` (reject the frozen `Σ c·Π q`).**
   The leaf is the true re-conditioning value of the base policy — a
   stationary-greedy rollout that re-selects `σ*` and re-evaluates `q` at each
   step. A frozen leaf (`q` pinned at `F`) is not the value of any policy under
   the re-conditioning dynamics, so the modeled-value monotonicity
   `W_0 ≥ W_1 ≥ W_2` is *not* guaranteed and in fact breaks under positive
   co-failure correlation — which RT2D is engineered to have. With `W_0 = V^base`,
   `W_{ℓ+1} = TW_ℓ ≤ W_ℓ` by policy improvement (`TV^π ≤ V^π` for any stationary
   `π`); since the bound is index-agnostic, the leaf is ordered by the *same*
   index `h=1` uses, keeping leaf-base ≡ `h=1`-base exactly (no nesting wrinkle).
   Cost: the leaf is an `O(K²)` rollout, mitigated by an episode-independent
   `q`/`S_succ` cache plus a call-scoped `W` memo; per-decision is `O(K^{h−1}·K²)`,
   tractable for `h≤3, K≤30`.

4. **`time` objective cost = train per-key mean refine time.** `attempts` uses
   `c≡1`; `time` uses the mean `refinement_wall_clock_s` per canonical key fit on
   train (OOV keys → global mean). Per-skeleton times are logged on
   `OutcomeRecord` but never pre-aggregated, so `_fit_refine_costs` aggregates
   them (mirrors `_fit_marginals`).

5. **Default eval budget = 30 (uncensored).** `dp_on_counts_baseline` defaults
   `attempt_budget=30` (= RT2D-n3 pool cap, the uncensored standard, this log
   2026-06-07) rather than the B1–B5 legacy default of 20, so a direct caller
   does not silently reintroduce censoring. Model selection's
   `val_rollout_attempts` budget (20) is unaffected.

**Monotonicity is a property of the modeled value, not realized rollouts.** The
unit test asserts `W_0 ≥ W_1 ≥ W_2` on the *modeled* value (including a
positive-correlation instance). Realized held-out attempt means need **not** be
monotone in `h` — finite-count `q` and sparser conditioning at large `|F|` make
deeper lookahead optimize against a less reliable model — so the realized
h-curve is reported as a result, never used as a pass/fail gate.

**Out of scope (future work).** A belief-MDP-over-latent-`z` variant: it needs
the scene latent `z` logged per training problem and is an oracle-structured
reference, not a fair count baseline. No refiner/planner/abstraction changes.

**Consequences.** New module `dp_on_counts.py` (env-free search) +
`eda.dp_on_counts_baseline` (B6); B4's NB scorer extracted to the shared
`_adaptive_score` (B4 output unchanged, guarded by existing tests). B6 registered
in `analyze_spectre.py` (h∈{1,2,3} sweep, comparison table/figures, extreme-q
diagnostic). The baseline roster is now B1–B6 (B6 = DP-on-counts; B4 is its
`h=1` special case). Run numbers: see `notebook.md` 2026-06-08.

---

## 2026-06-07 — Analysis notebook converted to marimo (`.py`)

**Context.** The analysis notebook was a Jupyter `.ipynb`
(`experiments/spectre/analyze_spectre.ipynb`, gitignored as scratch). Jupyter's
JSON-on-disk format is opaque to Claude Code — cell outputs are elided and edits
are clumsy — which made iterating on the EDA/comparison notebook with CC
painful.

**Decision.** Convert the notebook to a **marimo** notebook
(`experiments/spectre/analyze_spectre.py`): a pure-Python, text-first format
(cells are `@app.cell` functions) that CC can read and edit directly. The new
`.py` is the canonical analysis notebook and is **tracked**. Both files are kept
for now — the `.ipynb` stays gitignored alongside it — but forward-looking
"how to run the analysis" references point to the `.py`. Behaviour is preserved:
the marimo notebook reproduces every number and artifact (verified — pool-cap
1.000, overlap 0.973/OVERLAPPING, SPECTRE mean-attempts ≈ 5.67, §6 verdict
FAIL on 3.4, plus the SVG/PDF exports). Data root now resolves relative to the
notebook file (`mo.notebook_dir()`), so it runs from any launch directory rather
than only from `experiments/`.

**Keeping it out of CI.** A marimo `.py` does not satisfy `mypy`/`pylint`, so it
is excluded: `[tool.mypy] exclude` regex and `.pylintrc ignore-patterns` both
skip `analyze_spectre.py`; `black`/`isort`/`docformatter` still format it
(marimo files are black-compatible). `marimo` added to the `develop` extra so
the tracked notebook is runnable from a dev install. `run_ci_checks.sh` is
unaffected (verified: mypy reports no issues, pylint skips the file).

**Consequences.** marimo's single-definition dataflow rule forced a few local
renames during conversion (shared plotting temporaries `_`-prefixed; the
`_color_tag` helper promoted to a shared `color_tag`; the styled-table cell split
so its Styler renders, replacing Jupyter's `display(...)`). Historical
references to the `.ipynb` in past records (this log's 2026-06-07 uncensored-eval
entry; `archive/README.md`) are left as-is — they accurately describe figures
generated by the `.ipynb` before this conversion.

---

## 2026-06-07 — Report uncensored evaluation results (attempt budget = pool cap)

**Context.** The RT2D-n3 headline table and figures
(`experiments/spectre/analyze_spectre.ipynb`) were generated with
`ATTEMPT_BUDGET = 30` — equal to the candidate-pool cap — while the living
docs and the writeup described the evaluation attempt budget as 20. At budget
20, ~2–4% of episodes hit the cap and are censored to 21 (budget + 1); at
budget 30 the budget never binds (pool ≤ 30), so every episode runs to its
true first-success attempt and nothing is censored. The frozen-context
ablation (notebook 2026-06-06) surfaced the discrepancy: the full-variant
mean only reproduced the headline at budget 30, not 20.

**Decision.** Headline / reported evaluation metrics use the **uncensored**
budget — attempt budget = candidate-pool cap (30 for RT2D-n3) — so reported
attempt counts are the true time-to-first-success with no censoring. An
uncensored distribution is more informative than a censored one, especially
in the tail: it shows where any method (SPECTRE included) does badly rather
than collapsing those episodes to a single censored value. This is the
standard for SPECTRE evaluation tables going forward; if a future env's pool
cap differs, the eval budget tracks that cap.

**Scope — what this does NOT change.** Model selection and early stopping stay
on `val_rollout_attempts` at its own rollout budget (20) — a separate knob
from evaluation reporting (selection picks the checkpoint; this decision
governs how the chosen checkpoint is reported). The rollout-metric discipline
(proposal.md §5) is untouched.

**Consequences.** Writeup §Training and `archive/README.md` corrected 20 → 30
(the writeup's headline numbers were always budget-30; only the stated budget
was wrong). Pending reconciliation in the same commit: the "attempt budget 20"
phrasing in `proposal.md` §1 and the spectre `CLAUDE.md` headline line, which
refer to the evaluation/reporting budget and should read 30 (uncensored) — the
`val_rollout_attempts` mentions (budget 20, model selection) stay as-is.

---

## 2026-06-06 — Documentation discipline codified in CLAUDE.md

**Context.** The instruction to keep the living docs updated was a single
passive bullet ("Record run outcomes in notebook.md; lasting decisions in
decisions.md; method changes in proposal.md") — where but never when. It
demonstrably failed: `notebook.md` stayed empty for ~2 months of training
runs, every pre-refactor ADR below was reconstructed retroactively, and the
stale AUROC-as-key-metric claims survived ~6 weeks after the rollout-metric
change.

**Decision.** The spectre `CLAUDE.md` gains a "Documentation discipline"
section: a change-type → doc → format routing table (run numbers — including
negative results — → `notebook.md`; lasting choices → `decisions.md` ADR;
method/pipeline/protocol changes → `proposal.md` in place + §6 reconcile), a
before-commit rule (the doc entry ships in the same commit as the change),
a materiality threshold (mechanical refactors/formatting/typos exempt), and
a litmus test ("in 3 months, will we know this happened and why?").

**Alternative rejected.** Mechanical enforcement via a Claude Code hook:
project-level `.claude/settings.json` is committed and would fire for every
user of the shared monorepo, not just spectre development. A personal
`settings.local.json` hook remains an option if instructions alone prove
insufficient.

**Consequences.** Doc updates are part of the definition of done for any
non-trivial spectre commit. This entry is the first written under the rule.

---

## 2026-06-06 — Dated writeup snapshots in `docs/archive/`

**Context.** A high-quality paper-style writeup of the full project state was
deposited as `archive/SPECTRE_WRITEUP_APR_2026.md` (dated 2026-04-27 — two
days after the move to rollout-based model selection, whose checkpoints its
results use). It is a valuable reference but will go stale; the living docs
must not defer to it.

**Decision.** Writeups are dated, frozen, narrative exports named
`SPECTRE_WRITEUP_<MON>_<YYYY>.md` under `docs/archive/`, catalogued in the
"Snapshots" section of `archive/README.md`. The living docs (`proposal.md` /
`decisions.md` / `notebook.md`) remain the source of truth and win on
disagreement. At deposit time: reconcile any divergence into the living docs
first (headline results → a dated `notebook.md` entry; new limitations /
future-work items → `proposal.md` §6), then freeze. After freezing, snapshots
are not edited — staleness annotations go in `archive/README.md`. (One
documented exception: the 2026-06-06 fix of the writeup's pool-cap-30 /
attempt-budget-20 conflation.)

**Consequences.** `notebook.md` seeded with the 2026-04-27 results entry;
writeup-only limitations (data efficiency, Ψ fixed-size summary,
compositional generalization, x₀-conditioned prior) merged into
`proposal.md` §6. Next snapshot due when multi-seed RT2D results land.

---

## 2026-06-04 — Silo refactor: scope and placement

**Context.** Spectre files were scattered across a shared monorepo (root spec
docs, mixed `experiments/`, spectre edits to shared configs). Refactor executed
on branch `spectre-refactor`; safety/reversibility prioritized over tidiness.

**Decisions.**

1. **Docs home = `src/alphatamp/approaches/spectre/docs/`.** Original specs
   moved byte-unchanged to `docs/archive/` (historical notes live in
   `archive/README.md`, not in the files, to keep them unchanged);
   consolidated living proposal in `docs/proposal.md`; this log; `notebook.md`
   for running EDA notes; `RESEARCH_LIT.md` → `docs/research_lit.md`.
2. **Hydra configs live in `experiments/spectre/conf/`, not
   `src/.../spectre/conf/` + `pkg://`.** All five spectre Hydra entry points
   are scripts under `experiments/`; moving scripts and configs *together*
   keeps every `@hydra.main(config_path="conf")` byte-identical, requires no
   `__init__.py` in config dirs, and no package-data additions to the shared
   `pyproject.toml`. The `pkg://` route works under the editable install but
   is strictly more moving parts for zero extra siloing.
3. **All spectre experiment files moved into `experiments/spectre/`**: the 5
   `.py` entry points, 2 `.slurm`, 3 submit/collect `.sh`, the analysis
   notebook and its output artifacts. Shared `experiments/conf/` now contains
   only other-project configs.
4. **The 3 env configs (`clutteredstorage2d_b5`, `routedtransport2d_n3_v1`,
   `stickbutton2d_b5`) moved with spectre's conf tree, not deleted.** They
   were believed unused/historical but are *live*: the first two are composed
   as `defaults` by `spectre_collect`/`spectre_build_vocab`/`spectre_train`;
   the third is selected via CLI override in `submit_spectre_stickbutton2d_b5.sh`.
   Grep-proven that no other project references them.
5. **Shared `experiments/conf/hydra/launcher/slurm.yaml` restored to `main`'s
   values (4 cpus / 16 GB).** Spectre work had bumped it to 8/32 in place — a
   contamination of a shared config also referenced by `collect_data.py`.
   Spectre keeps its tuning in its own copy at
   `experiments/spectre/conf/hydra/launcher/slurm.yaml`, which resolves via
   spectre's config_path.
6. **`.gitignore` `archive/` rule anchored to `/archive/`.** The unanchored
   rule (meant for the root archive of old experiment results) silently
   ignored the new `docs/archive/`. Verified only two `archive` dirs exist in
   the repo, so anchoring is behavior-preserving for everything else.
7. **Deliberately left in place:** `src/alphatamp/approaches/spectre/`
   (IS the importable package), `tests/approaches/spectre/` (import paths),
   `data/spectre/` (the `data_root: "data/spectre"` convention in configs and
   shell scripts is unchanged), `experiments/slurm_outputs/` (shared scratch,
   gitignored — spectre keeps writing there rather than adding new ignore
   rules for a private dir), `experiments/__init__.py` (shared), all
   other-project files (bandit/BOX, sim-free param policy, LLM
   cluttered-storage), `tests/datasets/*.pkl` (other-project fixtures; 1-byte
   pickle churn was `git restore`d, not committed).
8. **Pre-refactor cleanup commits:** `eda.py` (+3-line `set_name` helper)
   committed; `.gitignore` merge-conflict markers fixed and scratch/data
   ignores added (`.data/`, `.sandbox-*`, `data/spectre/{raw,checkpoints,configs,derived}/`,
   `*.ipynb`).

**Consequences / follow-ups.**

- Anyone with muscle-memory paths (`python experiments/spectre_train.py …`)
  must add the `spectre/` segment.
- `main`'s `.gitignore` may still carry the merge-conflict markers — fix worth
  upstreaming separately.
- The slimmed root `CLAUDE.md` and the launcher revert only exist on this
  branch until merged.

---

## Pre-refactor decisions worth remembering (imported from specs/history)

- **2026-04 — Listwise PL loss over pointwise BCE.** Attempt 2 failed because
  BCE is not rollout-aligned; PL `−log P(argmax ∈ SUCC)` is. Load-bearing.
- **2026-04 — F contains failures only.** Test-time F can never contain a
  success; training F ⊆ FAIL_e strictly. Violation was an Attempt-2 root cause.
- **2026-04 — RT2D over kinder kinematic2d.** Lookup-table baseline (B3) is
  near-oracle on kinder envs → no research gap; RT2D engineered so beating B4
  requires relational tag binding (see `archive/SYNTHETIC_ENVIRONMENT.md`).
  *(⚠️ revisited 2026-06-25 → see the 2026-06-25 pivot entry: RT2D was
  effectively partially observable to the policy, and the evaluation now prefers
  pre-existing envs meeting the representation-advantage property wishlist.)*
- **2026-04 — Layer 2 (parquet) collapsed in the data pipeline.** At
  500/100/100-episode scale, globbing + loading raw episodes is fast enough;
  EDA operates in memory (`archive/SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` §3.1
  has the migration-back checklist).
- **2026-04 — Live frozen-dataclass schema instead of plain dicts.** Pickle
  stability insurance hasn't been needed; live objects let downstream code
  call substrate APIs directly (as-built §3.2).
- **2026-04 — Set-Transformer atom pooling, per-type augmentation policy,
  vocab-driven arity sizing, rollout-aligned F-mix, F-sample multiplier** —
  RT2D fixes 1–5 (`archive/SPECTRE_RT2D_METHOD_SPEC.md` §9).
- **2026-04 — AUROC(3) is the offline diagnostic that tracks test attempts;
  atom-sensitivity probes (D.1/D.2) are red herrings.** Never optimize for the
  probes. *Superseded for model selection (2026-04-25): checkpointing and
  early stopping use rollout-based `val_rollout_attempts` (see the
  overfitting-response entry below); AUROC(3) remains a secondary diagnostic.*
- **2026-04/05 — Overfitting response sequence:** diagnose → extra dropout →
  rollout-based validation/checkpoint selection (`checkpoint_metric =
  "val_rollout_attempts"` in `train.py`, used for both checkpointing and early
  stopping — aligned with the rollout-based test-time objective) → heuristic
  (FF z-score) prior as warm start (`train.prior_type`). Evaluation of prior
  choice pending.
