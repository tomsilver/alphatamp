# SPECTRE — Implementation & DD2D-Run Audit (2026-07-14)

**Scope.** An authoritative audit of the current SPECTRE implementation and the most
recent DD2D (Drawer Decluttering 2D) training/evaluation run, in which SPECTRE was
compared against **SPECTRE-static**, **PIGINet_v3**, and **astar-dist** (pure planning)
on the held-out DD2D test split (**124 problems, pool = 200 skeletons each**).

Every claim is grounded in a `file:line` citation or a direct data read (reproduction
commands in the [Verification](#verification) section). This is a frozen, dated audit
snapshot; the living docs (`proposal.md` / `decisions.md` / `notebook.md`) remain the
source of truth and win on any disagreement.

**Question map.**

| # | Question | Verdict |
|---|---|---|
| 1 | Canonicalization scope: per-episode or per-skeleton? Is augmentation consistent across (F, R)? Is canonical order correlated with the planner's? | **Per episode**; augmentation ON and consistent; order **uncorrelated**. Settles **H6**; **H2 preconditions met**. |
| 2 | Abstract-plan representation: option A `[s0, op…]` or option B (full interleaving)? | **Neither** — option A **+ terminal state**: `[s0, op1…opL, sL]`. |
| 3 | Censoring convention for unsolved-within-budget episodes? | Metric is **rollout-FP**; censoring/exclusion paths are **inert** on this run (all pools solvable, budget = pool cap). |
| 4 | Do all 4 methods use the same candidate pool per problem? | **Yes** — identical pool; methods differ **only** in ordering/scoring. |

---

## Finding 1 — Canonicalization scope: PER EPISODE (settles H6; H2 preconditions met)

### Scope = per episode, shared across the pool R and the failed set F

Local ids are assigned **per episode**, not per skeleton. `canonicalize_episode` builds
**one** `old_name → new Object` mapping (`canonicalize.py:162`, via `_renumber_mapping`)
and applies that *single* mapping to `initial_abstract_state`, `goal_atoms`,
`object_registry`, and **every** skeleton in `skeleton_pool` (`canonicalize.py:162-186`):

```python
# canonicalize.py:162-178
mapping = _renumber_mapping(episode, rng, type_aug_policy)
new_s0 = _remap_state(episode.initial_abstract_state, mapping)
new_goal = frozenset(_remap_atom(a, mapping) for a in episode.goal_atoms)
new_registry = {obj.name: obj.type.name for obj in mapping.values()}
new_skeletons: list[SkeletonRecord] = []
for skel in episode.skeleton_pool:
    new_ops = tuple(_remap_operator(op, mapping) for op in skel.operator_seq)
    ...
```

In `dataset.py.__getitem__`, one `canonicalize_episode` call produces `ep_view`, and
**both** R and F skeletons are sliced from the *same* `ep_view.skeleton_pool`
(`dataset.py:270-285`): `r_skeletons = tuple(ep_view.skeleton_pool[i] for i in r_sorted)`
and `f_skeletons = tuple(ep_view.skeleton_pool[i] for i in f_sorted)`.

**Consequence.** Object identity is trackable across the candidate pool **and** the
failed set within one training example. Per-skeleton renumbering — which would make
identity reasoning impossible by construction — is **not** what happens. This is the
precondition H2's hoped-for mechanism ("failed set = {these objects}, so promote their
supersets") needs in order to be *architecturally possible*.

### Renumbering augmentation: present, ON in this run, and consistent across (F, R)

- The random within-type permutation is the augmentation (`canonicalize.py:99-105`):

  ```python
  for type_name in sorted(by_type):
      names = sorted(by_type[type_name])
      augmentable = policy.get(type_name, True)
      if rng is None or not augmentable:
          permutation = list(range(len(names)))     # deterministic (eval/inference)
      else:
          permutation = list(rng.permutation(len(names)))   # augmentation (train)
  ```

- **The DD2D run had it ON.** `experiments/spectre/conf/spectre_train.yaml`
  `train.augment: true`, passed through as `rng=rng if self._augment else None`
  (`dataset.py:271`); and DD2D's single object type is augmentable
  (`env_registry.py:66`, `_DD2D_TYPE_AUG_POLICY = {"item": True}`).
- **Consistent across the whole (F, R) example.** Because a single `mapping` is applied
  to the entire episode, F and R necessarily share the *same* permutation — never
  independently drawn. The RNG is deterministic per `(seed, episode_idx, f_sample_idx,
  epoch)` (`dataset.py:234-244`, `_rng_for`) and the *same* rng object drives both the
  F-subset draw and the augmentation permutation (`dataset.py:249-271`), so a given
  `(F, augmentation)` pair is reproducible and jointly seeded.

At encode time the within-type index is parsed back out and offset by +1 (id 0 = pad):
`dataset.py:356-358` (`_local_id`), used at `dataset.py:367,380`.

### Canonical ordering is NOT correlated with the planner's / geometry

The within-type index is derived purely from the **alphabetical sort of the original
object name string** (`canonicalize.py:99-100`), then optionally randomly permuted. It
is **not** order-of-appearance in the plan and **not** geometry-sorted. Two supporting
facts:

- **DD2D raw object names** are `target, o0, o1, …, o8` (verified from a sample
  `test/…/000.json`). `target` is distinguished by the `target`/`extracted` **predicate**
  (goal `[['extracted','target']]`), not by its id; canonicalization renames it to
  `item_<idx>` like any other.
- `object_registry` is a **name-keyed dict** (`spectre_convert.py:141-144`,
  `object_registry = {name: ItemType.name for name in objects}`; RT2D/kinder equivalent
  in `collect.py`), so it cannot encode plan order in the first place. Even if upstream
  names carried an ordering, canonicalization discards it for alphabetical-name order
  (deterministic mode) or a random permutation (train augmentation).

### H6 verdict — no identity-ordering leakage

> **H6 (conjecture):** local ids smuggle in a distilled geometric prior if the
> within-type renumbering augmentation was off *and* the canonical ordering tracks
> geometry (e.g. `obs:0` = "most proximity-suspicious item").

Both leakage conditions fail here:

1. **Augmentation was ON**, so id *values* were randomized across training examples. The
   model saw every permutation and therefore *cannot* have baked in "`item_0` = most
   suspicious." The id→geometry association channel is destroyed at the across-example
   level.
2. **The canonical (eval) order is alphabetical-by-name**, not the planner's ordering
   and not geometry.

Crucially, the per-episode-consistent mapping **preserves within-example identity
linkage** (H2's precondition) while **destroying across-example id→geometry association**
(the H6 channel) — exactly the desired separation.

**Implication for H2.** The canonicalization *preconditions* for H2's mechanism are met:
identity is trackable across the pool and the failed set within an example, so if H2
fails it is **not** because identity reasoning was impossible by construction. H2's
failure (if any) rests on the separate information argument — in fully-observable,
deterministic TAMP, within-episode refinement failures add no information beyond x₀
(every skeleton's outcome is a deterministic function of x₀), so a failure close to the
feasible set and one far from it both present identically as `FAIL`
([`decisions.md` 2026-06-25](decisions/02-pivot.md#2026-06-25-direction-pivot-representation-question)(e)). This audit settles H6 and the preconditions; it does
**not** by itself resolve H2's learnability, which is an information question, not a
canonicalization one.

---

## Finding 2 — Abstract-plan representation: option A + terminal state (not full interleaving)

**Answer: neither pure option A nor pure option B.** The per-skeleton sequence fed to
the skeleton encoder Φ is

```
[STATE_0, OP_1, OP_2, …, OP_L, STATE_L]
```

— the initial abstract state, the L operators, and the **final** abstract state.
Intermediate abstract states s₁…s_{L-1} are **not** encoded ("Substage A"). It is
option A augmented with a single terminal state token; it is not the full interleaving
of option B.

Definitive sequence assembly — `model.py:516-533`, `SkeletonEncoder.forward`:

```python
# ------- Stitch sequence: [STATE_0, OP_1, ..., OP_L, STATE_L] -------
seq_len = l_max + 2
seq = torch.zeros(bsz, k, seq_len, D_MODEL, ...)
seq[:, :, 0, :] = s0_tok
seq[:, :, 1 : 1 + l_max, :] = op_tok
seq[:, :, 1 + l_max, :] = sL_tok
```

`seq_len = l_max + 2` (one s₀ + L ops + one s_L). Three token types only —
`TOKEN_TYPE_S0=0`, `TOKEN_TYPE_OP=1`, `TOKEN_TYPE_SL=2` (`model.py:46-48`) — with only
position 0 and the last position being state tokens.

Corroborating facts:

- **On-disk schema stores only the endpoints.** `SkeletonRecord` persists `operator_seq`
  and `final_abstract_state` only (`schema.py:66-73`, docstring "Substage A"); s₀ is
  stored once at the episode level. Intermediate states are never persisted, so Φ could
  not interleave them without recomputation. They are recoverable via STRIPS progression
  (`trajectory.py`), but that path is **not** wired into Φ.
- **No config flag** switches to full option-B interleaving — the design is fixed.
- **Minor caveat:** s_L's type-histogram is set equal to s₀'s (`model.py:707-711`),
  valid here because DD2D/RT2D operators never add or delete objects.

---

## Finding 3 — Censoring convention: metric is rollout-FP, and censoring is inert on this run

### The reported metric is not "mean attempts with a budget" — it is rollout-FP

The DD2D 4-method comparison reports **rollout false-positives (FP) = failed refinement
attempts before the first success** (`dd2d_compare.py`,
`experiments/spectre/compare_dd2d_methods.py`). FP relates to attempts as:

```
FP = attempts_to_first_success − 1
```

There are two metric families in the codebase; the 4-method DD2D comparison you asked
about is the **rollout-FP** family (pool cap 200, 124 test problems), **not** the RT2D
"mean-attempts / `val_rollout_attempts`" family (pool cap 30, model-selection budget 20)
used by `eda.py` / `analyze_spectre.py` / `train.py`.

### Two code paths, by method

- **Static methods** (astar-dist, PIGINet_v3, SPECTRE-static) — a pure ranking
  statistic, **no sequential budget**. FP via `dd2d_compare.rollout_fp` (`:54-68`):

  ```python
  def rollout_fp(scores, labels) -> float | None:
      pos = [s for s, lbl in zip(scores, labels) if lbl > 0.5]
      if not pos:
          return None                      # no feasible skeleton → EXCLUDED
      top = max(pos)
      strict = sum(1 for s, lbl in zip(scores, labels) if lbl < 0.5 and s > top)
      ties  = sum(1 for s, lbl in zip(scores, labels) if lbl < 0.5 and s == top)
      return float(strict) + 0.5 * float(ties)
  ```

  = number of infeasible skeletons ranked strictly above the best-scoring feasible one
  (+0.5 per exact score tie). Returns `None` — problem **excluded entirely** — when the
  pool has no feasible skeleton (`dd2d_compare.py:93-94,117-118`).

- **SPECTRE-adaptive** — an actual sequential rollout, `attempts − 1`:
  `eda.spectre_evaluate(attempt_budget=200, freeze_context=False)`
  (`precompute_dd2d_cache.py:154-176`; `fp = float(att) - 1.0`). The rollout's censoring
  branch (`eda.py:1379-1401`) sets `attempts = attempt_budget + 1 = 201` (→ FP = 200) on
  budget exhaustion or pool exhaustion without success — i.e. the "budget+1" convention.

### The two exclusion filters select the identical problem set

Static methods drop no-feasible pools via `rollout_fp → None`; SPECTRE-adaptive drops
them via the trainable filter `_trainable_episodes` (`eda.py:385`):
`[i for i, ep in … if ep.summary.num_success >= 1]`. `num_success >= 1` ⟺ `pos`
non-empty in `rollout_fp`, so all four methods score **exactly the same 124 problems**.

### On this dataset, neither censoring path activates (verified by direct data read)

- All **124** DD2D test problems have a pool of **exactly 200** skeletons, each with
  **≥ 1 feasible** skeleton (min successes per episode = 1; 0 episodes with no success).
  → `rollout_fp` never returns `None`; **0 problems excluded** — all four methods report
  **n = 124** (confirmed from `dd2d_method_comparison.csv`).
- Budget 200 = pool cap → **uncensored** (mirrors the RT2D "budget = pool cap" discipline,
  [`decisions.md` 2026-06-07](decisions/README.md)). Max observed FP by method: astar **199**, PIGINet **129**,
  SPECTRE-adaptive **145.67**, SPECTRE-static **159.67** — all `< 200`, so the
  `attempt_budget + 1` censoring **never fires**. (astar's 199 is a genuine uncensored
  FP — the lone feasible ranked last in a 200-pool — not a censor cap.)

### Consistency verdict

Because **no episode contributes a censored or excluded value**, the convention **cannot
manufacture or hide a gap** on this run — the reported means are true uncensored FP. Mean
FP by method (all strata, n = 124): **astar 33.01 / PIGINet 20.39 / SPECTRE-adaptive
19.23 / SPECTRE-static 23.75** (matches the notebook's hardcoded ballpark check,
`compare_dd2d_methods.py:145`).

The **"122.76 ± 39"** figure is **astar-dist on stratum 3** (hardest min-feasible-subset
band): reproduced mean = **122.8** from the CSV — a genuine mean of real per-problem FPs
(feasible skeletons ranked deep in the 200-pool), **not** a censoring artifact. Per-stratum
astar means: s0 ≈ 0.0, s1 ≈ 1.8, s2 ≈ 16.1, **s3 ≈ 122.8**.

### Latent risks to record (for future datasets/budgets — not this run)

1. **Asymmetric censoring.** The static methods have **no sequential budget** (they
   report true FP up to 199), while SPECTRE-adaptive imposes a 200-step budget and would
   **right-censor at FP = 200** (`att = budget + 1`) if a feasible were never reached.
   On a dataset whose pools exceed the hardcoded `attempt_budget=200`
   (`precompute_dd2d_cache.py:159`), or with a budget < pool size, this asymmetry *could*
   bias the adaptive-vs-static comparison. Harmless here only because every DD2D pool is
   solvable and pool size = adaptive budget = 200.
2. **Tie-handling definitional wrinkle (non-censoring).** Static FP awards **half-credit
   for exact score ties** (`0.5 * ties`, `dd2d_compare.py:67`), whereas the adaptive
   rollout breaks ties deterministically by `argmax`. So the static-FP and adaptive-FP
   definitions are not strictly identical even before censoring — a minor accounting
   difference, worth stating so the adaptive-vs-static delta isn't over-read at the
   sub-attempt level.

---

## Finding 4 — Candidate pool: identical across all 4 methods; only the order differs

**Yes.** The pool is the DD2D collector's frozen **k = 200 astar-dist skeleton pool** per
problem — one `NNN.json` per candidate, all sharing the same objects/init/goal, each
pre-refined and labeled — generated **once upstream** and shared. astar + SPECTRE read
the converted `EpisodeRecord` pickles; PIGINet reads the same raw JSON directly. The two
representations are the same set of skeletons with the same labels, aligned by
`plan_idx == skeleton_idx` (`dd2d_compare.py:17-19`; converter enumeration at
`spectre_convert.py:118-193`). **No method prunes, caps, or regenerates the pool.** The
rollout-FP metric is itself pool-order-invariant (it only counts infeasibles ranked above
the best feasible), so equivalence rests solely on the identical `{skeleton → label}`
set, which holds.

Methods differ **only** in the scoring/ordering function:

| Method | Pool source | Scoring / ordering site | Order type |
|---|---|---|---|
| astar-dist | converted `EpisodeRecord` | `precompute_dd2d_cache.py:85` — `score = -plan_idx` | fixed, non-learned planner order |
| PIGINet_v3 | raw DD2D JSON (`PIGINetDataset`) | `envs/dd2d/piginet/eval.py:55` — learned CLIP+transformer logits | fixed, learned one-shot |
| SPECTRE-static | converted `EpisodeRecord` | `precompute_dd2d_cache.py:186-196` — empty-context (c₀) logits, scored once | fixed, learned at c₀ |
| SPECTRE-adaptive | converted `EpisodeRecord` | `inference.py:206-224`, driven by `eda.py:1379-1398` | **re-ranked after each failure** via the failed-set context Ψ |

The static-vs-adaptive difference is exactly the `freeze_context` flag
(`inference.py:206-224`): adaptive re-encodes the context `c` from the embeddings of the
already-failed skeletons and re-scores the remaining pool after every failed attempt;
static/frozen pins the context to `c₀` at every step ("exactly a static ranking by the
initial logits", `inference.py:198-203`). SPECTRE-static is thus the strict same-policy
comparator to PIGINet (both fixed one-shot rankings); SPECTRE-adaptive is the only method
that re-orders the remaining pool using the observed failed set.

---

## Verification

Reproduce the two data facts the findings rest on (run from repo root after
`source .venv/bin/activate`):

```bash
# 124 test episodes, pool size 200, every pool solvable (min successes = 1)
python - <<'PY'
from pathlib import Path
from alphatamp.approaches.spectre import eda
test = eda.load_split_episodes(Path("data/spectre/raw/dd2d_v2/test"))
sizes = [len(ep.skeleton_pool) for ep in test.episodes]
nsucc = [sum(1 for o in ep.outcomes if o.outcome == "success") for ep in test.episodes]
print("episodes", len(test.episodes), "pool", min(sizes), max(sizes),
      "min_succ", min(nsucc), "zero_succ", sum(1 for s in nsucc if s == 0))
PY

# Per-method mean FP (all strata) + astar stratum-3 mean = 122.8
python - <<'PY'
import csv, statistics
from collections import defaultdict
rows = defaultdict(list); strat = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("experiments/spectre/dd2d_method_comparison.csv")):
    rows[r["method"]].append(float(r["fp"]))
    strat[r["method"]][r["stratum"]].append(float(r["fp"]))
for m, v in rows.items():
    print(f"{m:18s} n={len(v)} mean={statistics.mean(v):.2f} max={max(v):.2f}")
print("astar s3 mean =", round(statistics.mean(strat["astar-dist"]["3"]), 1))
PY
```

Cited `file:line` anchors to spot-check: `canonicalize.py:99-105,162-186`;
`dataset.py:234-244,249-285,356-380`; `model.py:46-48,516-533,707-711`; `schema.py:66-73`;
`env_registry.py:66`; `spectre_convert.py:118-193`; `dd2d_compare.py:54-68,93-118`;
`precompute_dd2d_cache.py:85,154-196`; `inference.py:198-224`; `eda.py:385,1379-1401`;
`experiments/spectre/conf/spectre_train.yaml` (`train.augment`, `rollout_attempt_budget`).
