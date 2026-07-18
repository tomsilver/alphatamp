# DD2D demo arguments — explained

Answers to your questions about `blocks_tamp/dd2d/demo.py` (and the generator it drives,
`blocks_tamp/dd2d/{problem,scene,shapes,world,label}.py`). Written in plain language; file:line
references are given so you can verify.

A quick mental model first. Every DD2D problem is a **top-down household drawer** holding one
**target** item plus several **blocker** items. The target starts ungraspable (neighbors block
every two-finger grasp). To solve it, the robot moves some blockers out of the drawer into a
small side **buffer** (the staging area to the right), then grabs the target. The interesting
question is *which* blockers must move, and whether they *fit* in the buffer.

---

## 1. `--num-items` — exact count, or sampled?

**Short answer:** When you set it, it is an **exact upper target**, not a random range. It is *not*
"sample between 9 and NUM_ITEMS." But two things can make the *actual* count come out lower or
higher than the number you pass, so treat it as "aim for this many" rather than a hard guarantee.

**Details** (`scene.py:82`, `scene.py:126-143`):
- Default (`None`) → the scene draws a count uniformly from **9–14** (`N_RANGE`, `scene.py:36`).
- Fixed (e.g. `--num-items 11`) → the generator fills the drawer until it reaches **exactly** that
  many items *or* the drawer hits its area-fill cap (`coverage < fill`, `fill` ≈ 0.35–0.55) *or* it
  runs out of placement attempts. So the final count is **≤ your number**; for reasonable counts it
  usually lands exactly on it, but a small drawer + a big number can stop early.
- The count **includes the target** (so `--num-items 11` = 1 target + up to 10 blockers).
- **Caveat with `--crowd`:** the "collar" items are **counted inside** the `--num-items` budget, not
  added on top. Order in `scene.py`: place the target, then the collar loop places up to `crowd`
  cans, then the fill loop tops up with clutter **only until `len(items) == num_items`**
  (`scene.py:127`). So the total is normally just `num_items`, of which up to `crowd` are the collar
  and the rest is generic clutter. The total exceeds `num_items` **only** in the rare case where the
  collar loop *alone* places more than `num_items - 1` items (needs `crowd > num_items - 1` **and**
  every collar placement to succeed — in practice the tight ring saturates and later placements fail).
  This is why a `--crowd 10` run still shows ~9–14 items, not ~19–24.

**Note:** "blockers" is not a separate knob — the number of blockers is just `num_items - 1`
(`problem.py:151`). See also `docs/dd2d.md` and the EDA notes for the *decision-relevant* count,
`min_feasible_subset` (how many blockers actually *must* move), which is a different quantity.

**For your train→test robustness experiment:** Yes — this is a clean and supported generalization
axis. Fix `--num-items` at training (e.g. 11), then raise it at test (14, 16, 18…) to show
generalization to more crowded, novel drawers. Two things to keep in mind:
- The **buffer does not grow with `--num-items`** (its size is set by `--lambda`, see Q2). So more
  items means more potential blockers competing for the *same* staging space — difficulty rises
  naturally, which is usually what you want for a robustness curve.
- The geometry-informed `candidates` planner handles large counts fine; the geometry-blind
  `pyperplan`/`symk` baselines get expensive as items grow (SymK has a hard wall past ~14 total
  objects — `docs/dd2d.md`). For test-time scaling, prefer the `candidates` planner.

**Holding out item *categories* (e.g. no bananas/shoes at train, introduce at test):**
**Not currently supported as a general capability.** The shape library has 7 families
(`can, bowl, box, pillcase, dumbbell, shoe, banana`; `shapes.py:33-44`), but there is no
argument to *exclude* a chosen family at train and *add* it at test. The only built-in
distribution-shift hook is `--split holdout` (Q4), which does a *fixed* small shift, not an
arbitrary category holdout. Adding true category holdout is a small change: give `sample_shape`
an `allowed_families` / `exclude_families` argument (`shapes.py:174`) and thread it through
`generate_scene` → `generate_dd2d_problem` → the demo. (The concave-vs-convex flag *is* tracked per
item and stratified in analysis — `shapes.py:8-9` — but that is a reporting split, not a train/test
holdout.)

---

## 2. `--lambda` — what is it?

**Short answer:** It is the **size of the staging buffer** (the side area where evicted blockers
are placed), as a scale factor. It is **not** the spacing between items. Smaller λ = smaller buffer
= harder to fit the blockers you remove.

**Details** (`scene.py:46-49`):
- The buffer is a rectangle to the right of the drawer. Its length and depth are drawn from
  `BUFFER_L` ≈ 25–45 cm and `BUFFER_D` ≈ 12–20 cm, then **both multiplied by λ**.
- So `--lambda 0.8` gives a buffer 80% of the nominal size; `--lambda 0.6` shrinks it further.
- Effect: with a **tight** buffer, the set of blockers you evict may not physically **pack** into
  the staging area — this creates "buffer-overflow" infeasibility, which is the packing-feasibility
  signal DD2D is built around. With a **loose** buffer, almost any subset fits, and the only
  remaining difficulty is drawer-side (which blockers are themselves buried).
- This is why λ is called the **difficulty dial** (`demo.py:14-15`); the interesting regime is
  roughly **[0.75, 0.9]**.

Think of λ as "how big is the shelf I'm allowed to stage things on." It changes total capacity, not
how items are spaced.

---

## 3. `--margin` — how is it different from λ?

**Short answer:** `margin` (δ, default **1.0 cm**) is the **minimum clearance required between items
(and between items and the buffer walls) when packing them into the buffer**. λ sets *how big* the
buffer is; margin sets *how tightly you're allowed to pack* inside whatever buffer you have. They
are different levers on the same "does it fit?" question.

**The margin is between:** any two staged items, and each item and the buffer boundary — i.e. it is
a safety gap around every packed footprint.

**Details** (`label.py:74`, `label.py:88`, `label.py:62`):
- When the labeler checks whether a chosen blocker subset can be packed, it **inflates each item's
  footprint by δ/2** before testing for overlap (`inflate = scene.margin / 2`, `label.py:88`,
  `label.py:62`). Two items each grown by δ/2 must therefore sit at least **δ apart** to count as a
  valid packing. So δ is a required inter-item gap.
- It also feeds the "provably can't pack" shortcut: each item's area is **deflated by δ/2** and
  summed; if that still exceeds the buffer area, packing is impossible (`_area_bound_infeasible`,
  `label.py:71-79`).
- Bigger δ ⇒ items must be spaced further apart ⇒ packing is harder / easier to rule out. Smaller δ
  ⇒ more permissive packing. `docs/dd2d.md` calls δ the **refiner-slack / isolation knob**.
- Why it exists: it makes the feasibility label robust to sampling noise and to the fact that the
  gripper fingers have real thickness (the fingers are ~2 cm, wider than δ=1 cm, which is why DD2D
  certifies *accessible* packings, not bare ones — `docs/dd2d.md`).

**λ vs margin in one line:** λ = *size of the shelf*; margin = *required breathing room around each
object you place on it*.

---

## 4. `--split {train, holdout}` — what's the difference?

**Short answer:** `holdout` is a **mild distribution shift of the item shapes**, meant for a
generalization test. It changes the *shapes you sample*, not the drawer size, item count, or task.
It is **wired but deferred** — the hook exists and works, but it is not yet used in a formal
train/test experiment.

**What actually changes on `holdout`** (`shapes.py:167-171`, `shapes.py:190`, `shapes.py:129-131`):
- **All item dimensions scale by +15%** (`shift = 1.15`) — every family is sampled a bit larger.
- **One family is swapped:** `bowl → can` (`_family_swap`, `shapes.py:169`). So large-circle bowls
  never appear on the holdout split; they are replaced by cans.
- Everything else (drawer sizes, buffer, fill, λ, margin, item *count*, the task itself) is the
  **same** as train. It is purely a shape-library shift.

**What does *not* change:** it is not just a data label — it genuinely alters the sampled geometry.
But it is a *fixed, small* shift (‑/+15% size + one swap), **not** a configurable "these categories
are unseen at train" holdout. For arbitrary category holdout, see the note at the end of Q1.

---

## 5. `--crowd` — what does it do?

**Short answer:** It **adds `crowd` new "collar" items** (small cans) in a ring hugging the
target, and biases the **target itself to a compact round shape**. It does **not** pick existing
blockers and slide them inward — it *introduces* new items positioned to pincer the target.

**Details** (`scene.py:107-124`, `scene.py:88-89`):
- For `i` in `range(crowd)`, it places a small can at an **evenly-spaced bearing** around the
  target, slid *inward* toward it (`collar_pose`, `world.py`). Evenly-spaced bearings mean opposite
  items straddle the target's grasp corridors in **pairs**.
- Because a two-finger grasp is only free if **both** opposing finger corridors are clear, a
  pincering pair means you must remove a **2+ blocker subset** (not a single object) to expose the
  target. That is the whole point: `--crowd` raises the fraction of problems that genuinely require
  identifying a *subset*.
- It also biases the **target** to a round `can` so a small ring can fully surround it (an
  elongated target would keep graspable slots at its tips, `scene.py:24-26`).
- Measured effect (`notebook.md`, `docs/dd2d.md`): `--crowd 0` ≈ naturalistic (only ~5–10% of
  problems need a 2+ subset); `--crowd 10` ≈ **~50%** need a 2+ subset. Default is **10**.
- **`--diverse-crowd`** draws collar items from **all** families (not just round cans), so concave
  shapes (dumbbell/shoe/banana) join the pincer instead of only landing in the outer clutter as
  distractors. Because non-round items leave larger angular gaps and fail `collar_pose` placement
  more often, the ring is looser: measured effect at `--crowd 10 --lambda 0.6` is the 2+-subset rate
  dropping from **~50% → ~10%**. Pair it with `--require-subset` to restore a high rate by resampling.
  The problem-id gains a `dc` marker (`..._c10dc_s0`) so diverse datasets don't collide with plain ones.
  **Caveat:** a high floor like `--min-subset 3` combined with `--diverse-crowd` is often too rare and
  can exhaust `max_resamples` (a loud `RuntimeError`, not a silent skip); raise `--crowd`, lower the
  floor, or drop `--diverse-crowd` if you need deep subsets.
- **Interaction with `--num-items` (see Q1):** collar items are **counted within** the `--num-items`
  budget, not added on top — the collar fills some of the blocker slots and generic clutter fills the
  rest, so the total is normally just `num_items`. It exceeds `num_items` only in the rare
  collar-saturation edge case. Use `--crowd 0` for a clutter-only baseline.

**Related knobs:** `--min-subset N` keeps only problems whose smallest feasible clearing subset is
≥ N, and **implies `--require-subset`**; `--require-subset` on its own is the same as `--min-subset 2`.
These enforce a **hard floor by rejection** — a strict version of what `--crowd` does probabilistically
(so they need enough `--crowd` to actually produce that mass, else generation thrashes). `--crowd`
shapes the distribution; `--min-subset` / `--require-subset` enforce the floor.

---

## Cheat sheet

| Argument | Controls | Bigger value → | Default |
|---|---|---|---|
| `--num-items` | target count of items incl. target (upper target, not random) | more blockers, denser drawer | sampled 9–14 |
| `--lambda` | **size** of the staging buffer | looser packing, easier | 0.8 |
| `--margin` | required **gap** between packed items/walls (δ) | tighter packing rule, harder | 1.0 cm |
| `--split` | item-shape distribution (`holdout` = +15% size, bowl→can) | — (train vs shifted test) | train |
| `--crowd` | # of collar items ringing the target (forces 2+ subsets) | more subset-required problems | 10 |
| `--diverse-crowd` | collar drawn from **all** families, not just round cans | concave shapes join the pincer; looser ring → lower subset rate | off |

**Not currently supported (would be small additions):** excluding chosen item categories at train
and revealing them at test (only the fixed `--split holdout` shift exists today).
