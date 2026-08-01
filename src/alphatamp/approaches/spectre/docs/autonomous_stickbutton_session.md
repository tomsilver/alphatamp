# Autonomous session — making StickButton2D b5/b10 collectable

**Date:** 2026-07-28/29. **Run unsupervised** at the user's direction, with a standing brief
and hard constraints. This document records every decision taken without supervision, so they
can be reviewed or reversed as a batch.

## The brief, as given

> Continue autonomously trying to get b5/b10 working (**at least 50% success rate on any given
> problem**, given a budget of **200 attempts per problem and 20 seconds per attempt**). Don't
> mess with the kinder refiners/samplers. Instead, focus your efforts on developing a
> reasonable A* heuristic — the abstract planning level, using our existing A* baseline
> planner. The heuristic just gets a concrete (and optionally abstract) state.

Plus a constraint added mid-session:

> Whatever working heuristic you develop must be **simple to interpret**. If it's extremely
> complex or hard to interpret, it's too hacky. It should be something I can easily tell a
> story about.

**Success criterion adopted:** ≥50% of problems have ≥1 refinable skeleton within a
200-candidate pool at 20 s/attempt, measured with the **stock** kinder sampler and refiner.

## Constraints I held myself to

1. **No changes to kinder's refiner or trajectory sampler.** In particular this rules out the
   `acceptance="superset"` relaxation explored earlier in the day
   (`sampler.py`) — it stays in the tree, off by default, unused by anything on this path.
   Everything below happens in the abstract plan generator.
2. **Interpretability is a hard requirement, not a preference.** No learned weights, no
   per-variant tuning tables, no sweep-detection geometry. The final heuristic is four terms,
   each with a one-line justification.
3. **Measure, don't infer.** Earlier in the day I asserted a cause (obstacle avoidance) from a
   property of the controllers rather than from failure data, and it was wrong. Every claim
   below is tied to a measurement.

## The problem, stated precisely

With the stock sampler, a step is accepted only if the achieved abstract state **equals** the
planned one (`parameterized_controller_sampler.py:89`). The env presses *any* button the robot
or stick drives over. So a skeleton refines only if no leg of its trajectory sweeps a
still-unpressed button — otherwise the world runs *ahead* of the plan and the check rejects a
trajectory whose final state would have been correct.

Measured: of the extra atoms at failure, **41/41 were buttons the remaining plan still
intended to press, 0 were strays**. Nothing is going wrong physically; the work is being done
in the wrong *order*.

And critically — **every press ordering has the same plan length**. So a count-based heuristic
rates all of them identically: 120 orderings at b5, ~3.6M at b10, enumerated arbitrarily, of
which only the sweep-free ones refine. That is the entire b5/b10 problem, and it is squarely a
heuristic problem, which is why the brief's framing is the right one.

## Decisions

### D1 — The heuristic gains a "distance to the nearest unpressed button" term

```
h(s) = |unpressed|
     + 1  if a stick pickup is unavoidable   (some unpressed button is out of arm's reach,
     + 1  if a stick putdown is unavoidable    hand empty / holding, respectively)
     + distance(robot, nearest unpressed button) / world_diagonal
```

**The story.** Count the work left — one press per remaining button, plus the stick trip you
can already see coming. Then, among the many orderings that cost the same number of actions,
prefer to be *close to the next button*.

**Why that last term is the whole ballgame.** The robot presses whatever it drives over, so a
plan breaks when it crosses a button it hasn't gotten to yet. But if you always go to the
**nearest** remaining button, nothing unpressed can be on the way — anything on that segment
would have been nearer, and would have been your target instead. So "always walk to the
nearest one" and "never press a button out of order" are *the same preference*. The heuristic
is not modelling the failure; it is expressing the ordering that avoids it.

The distance is normalised by the world diagonal so it stays in `[0, 1)` and can never
outweigh a whole action.

### D2 — Distance-to-nearest, **not** remaining-tour-length

Both are natural; only one works, and the reason is worth recording because it is not obvious.

Each press adds 1 to `g` and removes 1 from `|unpressed|`, so **the counting part is constant
along every path** and the distance term is the sole discriminator. Ranking by the *remaining*
tour then inverts the preference: clearing a far outlier early shrinks what is left, so A*
rates far-first plans best — the exact opposite of nearest-first. Distance-to-the-next-button
has no such inversion, because going far leaves you far from everything else.

Measured on b5/seed5 at a 200-candidate budget: remaining-tour reached its first success at
candidate **145**; distance-to-nearest at **78**.

### D3 — Reach is enforced by **grounding**, not by a heuristic surcharge

`RobotPressButton*(robot, b)` is not grounded at all when `b` is past the robot's reach
(`geometry.robot_reach_max_y`, 1.405, derived from the env config).

The `+1` surcharge alone cannot fix this: it is a *constant*, and pressing an out-of-reach
button still lowers `|unpressed|` by one, so A* keeps rating those plans optimal. Reach is an
applicability fact, so it belongs in applicability. Measured effect at b5: 67 ground operators
→ 49–61.

### D4 — The counting term is weighted above 1, because `g` carries no information here

`h` uses `count_weight · |unpressed|` with `count_weight > 1`.

With weight exactly 1, `g + h` is **depth-invariant** (each press adds 1 to `g` and removes 1
from the count), so the search has no pressure to go deeper — it plateaus across shallow
states. At b5 that is survivable; at b10 the branching factor makes it fatal. Measured, pool
generation with a 30 s budget:

| count_weight | b5 pool | b10 pool |
|---|---|---|
| 1.0 | 200 (0.1 s) | **0 — timed out** |
| 1.5 | 200 (0.0 s) | 200 (0.2 s) |
| 2.0 | 200 (0.0 s) | 200 (0.2 s) |

**The story stays simple:** every plan here is the same length, so `g` tells us nothing about
plan quality; weighting the remaining-work term makes the search greedy on *how much is left*
rather than breadth-first over ties, with distance as the tie-break. The exact weight is
chosen by measurement (below), not tuned per variant.

### D5 — b10's real obstacle is pool *prefix* diversity, and a heuristic cannot fix it

This is the finding that decides how far the brief can be met, so it is recorded even
though it is negative.

Refinement failures happen at **step 0–1** (measured: 102/120 skeletons fail at step 0). So
a pool whose 200 members share an opening move fails *as a block* — the budget of 200
attempts is spent on 200 variations of the same bad start. Distinct press orders over 200
candidates, as (distinct 1st press / distinct first three):

| weight | b5 | b10 |
|---|---|---|
| 1.0 | 5 / 32 | *empty pool* |
| **1.05** | **5 / 32** | **1 / 1** |
| 1.5 | 2 / 7 | 1 / 1 |
| 2.0 | 1 / 2 | 1 / 1 |

At b10 **every candidate shares the same first three presses at every workable weight.**
Note the pool is not short of variety in absolute terms — all 200 orderings are distinct —
it is varied only in its *tail*, which is precisely where it does not matter.

This is structural, not a tuning failure. A single A* run yields goals in `f` order, so
alternative openings surface only after their whole subtree is exhausted; at 10 buttons
that never happens within any sane budget. Getting prefix diversity needs a *diverse
planning* mechanism (forbid-loop / top-k over openings, as DD2D's enumerator does), which
is a change to the plan generator, not to `h`.

**Tried and rejected: quantising the distance term.** The idea was that rounding distances
would make near-equal openings tie exactly, letting the generator's own RNG tie-breaker
diversify the prefix. It does nothing — b10 stays at 1 distinct opening at every rounding
level tested (0.1, 0.25, 0.5, 1.0 world units), and b5 is unchanged at 5. The shared prefix
is not caused by distance ties; A* dives into whichever opening's subtree has the lowest
`f` and finds 200 goals there before ever reconsidering. Recorded so it is not retried.

### D6 — how good the ceiling is, independent of search

Refining the single explicit nearest-first plan (built directly, not searched for) gives
**b3 55%, b5 25%, b10 5%** of problems. So the nearest-first *argument* is only
approximately right: the robot has a body (base radius 0.1) and the stick is 1.25 long, so
even the nearest hop can sweep a button sitting beside the corridor. Nearest-first is a good
prior, not a guarantee — which is exactly why it needs 200 attempts rather than 1, and why
b10 (5% per attempt, one shared opening) is the hard case.

## Results

Deployed heuristic, **stock** kinder sampler and refiner, 200 attempts per problem, 20 s per
attempt, 20 problems per variant, every candidate refined (non-short-circuiting):

| variant | problems with ≥1 success | mean #successes / 200 |
|---|---|---|
| b3 | **20/20 (100%)** | 15.4 |
| b5 | **15/20 (75%)** | 4.0 |
| b10 | **0/20 (0%)** | 0.0 |

**b5 meets the brief (75% ≥ 50%). b10 does not, and I believe it cannot without a plan
generator change — see D5.**

For contrast, the same variants before this session: b5 **0/8**, b10 **0/4**. And b3 improves
too — the first refinable candidate now arrives at index 2–10, against 14–16 before.

The b5 failures are seeds 1, 2, 9, 12, 13. Nothing distinguishes them structurally that I
found; they are consistent with the per-attempt success rate simply being low enough that
200 draws from a pool with 5 distinct openings sometimes miss.

### Honest caveats

- **One seed per problem, 20 problems per variant.** Enough to separate 75% from 0%, not
  enough to quote 75% to the percentage point.
- **b10 occasionally fails to generate a pool at all** (seed 9 returned 0 candidates within
  the 30 s abstract-planning budget). Raise `abstract_plan_timeout_s` if b10 is revisited.
- **b5 is expensive**: ~900–2700 s per problem at 200 candidates, because every candidate is
  refined. A 400/100/100 collection at b5 is roughly 100–150 CPU-hours; budget ~5 h at
  30-way parallelism.

### End-to-end verification

`spectre_collect.py env=stickbutton2d_b5 K_max=200` → `spectre_build_vocab.py` →
`spectre_check_pipeline.py`, all green, **0 episodes filtered**. The two collected episodes
report 12 successes (first@0) and 5 successes (first@108) — matching the measurement run's
seeds 4 and 5 exactly, which confirms the collector and the harness drive the same generator.
Vocab: 7 operators / 6 predicates / 3 types, val+test OOV-clean.

CI on the touched files: `black`/`isort`/`docformatter` clean, `mypy` clean over the whole
spectre package (70 files), pylint clean, 449 spectre tests pass. The 35 repo-wide mypy errors
that remain are pre-existing on this branch (v3 test files) and untouched by this session.

## Recommendation

Adopt b1/b2/b3/b5 as the collectable set for the pooled StickButton2D dataset and drop b10.
b5 at 75% with a mean of 4 positives per 200-candidate pool is a *good* SPECTRE problem —
positives are scarce enough that ranking matters, which is exactly the regime the method is
for. b3 at 15.4 positives is comparatively easy.

If b10 is wanted later, the work is **prefix-diverse plan generation** (D5), not a better
heuristic — and that should be scoped as its own change to the generator.

## Time budget

Session began ~00:10 EDT 2026-07-29 with a 10:00 EDT deadline. Ample for the measurements
above. If the b10 diverse-pool question is picked up later, it is a plan-generator change
and should be scoped separately — it is not a heuristic tweak.

## What I did NOT do, and why

- **Did not touch the refiner or sampler.** Per the brief. `acceptance="superset"` remains
  off; it is not part of any recommendation here.
- **Did not add sweep-detection geometry** (checking whether a segment passes near an
  unpressed button). It would likely work better, and it is exactly the "too hacky to tell a
  story about" style the brief rules out. Recorded as an untaken lead.
- **Did not tune per variant.** One heuristic, one weight, all button counts.
