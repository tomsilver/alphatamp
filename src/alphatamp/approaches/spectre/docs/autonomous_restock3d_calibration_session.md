# Autonomous session — Restock3D eager heuristic + oracle + budget/K_max calibration

**2026-08-14/15, unsupervised overnight run.** No human in the loop: the user approved a gated plan,
asked for the run to proceed autonomously to ~09:00, and directed that design choices be made by best
judgment, recorded here + in an ADR, and reported at the end. This narrative is **unratified** (judged
without a human); the ratified protocol is
[`decisions/07` 2026-08-15](decisions/07-stickbutton2d.md#2026-08-15-restock3d-eager-validity-heuristic-oracle-solver-budget),
numbers in [`notebook/07` 2026-08-15](notebook/07-stickbutton2d.md#2026-08-15-restock3d-eager-heuristic-oracle-calibration-timeout).

The build ran as hard gates (A→B→E, A→C→D→E, {all}→F); every gate passed. What follows is the set of
non-trivial judgment calls made along the way, each with its rationale, so they can be reviewed.

## Design choices (best-judgment, no human)

- **DC1 — the eager order is a collection accelerator + a named baseline, NOT the training-pool
  membership order.** Empirically the eager first-feasible index is 0 on every r0–r3 problem, but the
  eager top-100 pool contains **0** tall→short (F3) candidates (λ_h=50 buries them past K=200) vs 57–86
  in the plain pool. Since all goal plans are equal length, a working eager order strictly front-loads
  the ~1–3 feasibles and demotes all F3 — you cannot both find the feasible first and keep F3 in a small
  pool. So pool membership + the reported classical baseline use the **plain** hff order (F3/F2-rich),
  the eager order is the collection short-circuit accelerator + the `astar-eager` arm, V3 (F3 presence)
  is judged on the plain pool, and K_max is sized from the plain first-feasible index. Matches
  governance §8 of the heuristic guide and the plan's flagged deferral.

- **DC2 — V1 (eager≈hff on slack) is read on r0, not r1.** r1 has σ_short=0 (the F2 stratum), so its
  penalties are correctly non-zero and eager diverges from hff (index 0 vs 6–14) — the heuristic
  working, not miswiring. On the genuinely-slack r0, eager≈plain (0 vs 1–2) with ≈0 penalties, so the
  V1 regression holds where it is meant to.

- **DC3 — the no-refinement K_max is trusted; the refinement-pilot fallback was skipped.**
  `is_feasible_skeleton` is a sound feasibility oracle for this env: F2 (region reuse) and F3
  (tall→short) are real PyBullet collisions, so a table-infeasible skeleton cannot refine (no false
  negatives), and table-feasible skeletons certify 100% via the oracle (Gate D). Hence the plain-order
  first-feasible index equals the real baseline FP; a K≤100 refinement pilot would only add censoring on
  r3, not resolve it. r3's 6/20 censoring beyond K=200 is a pool-coverage property (~1/200 feasible
  density), not classifier error. Consequence for the deferred collection design: plain-order r3 needs
  K_max>200 or reject-resample ~30%; eager finds r3 feasibles at index 0 but strips F3 → a hybrid is the
  natural config. cap_r covers the cost either way (feasible refine ~24 s; infeasible die at the cap).

- **DC4 — K_max enumeration workers lowered + self-heal added.** Enumerating K=200 pools is more
  memory-heavy than the oracle refine; 24 workers OOM-broke the pool on r2/r3 (a dead worker fails the
  pending futures). Lowered the kmax default to 12, ran at 8, and added a resubmit-at-4-workers self-heal
  pass (recovered all transient failures → 0 final errors). The oracle run (lighter, one skeleton
  refined per worker) was fine at 24.

## CI scoping (judgment)

Full `./run_ci_checks.sh` was **not** run: the working tree carries the user's uncommitted ShelfObstruct3D
WIP (separate effort), which pollutes a repo-wide `mypy .` / `pytest --pylint` regardless of this work.
Instead the restock3d changes were verified scoped: every **checked** file (the `envs/restock3d/*`
package modules, `collect.py`, `config.py`, `restock3d_collect.py`, and the two new tests) is mypy-clean
and pylint 10.00/10; the one-off restock3d diagnostic/calibration scripts were added to the mypy exclude
and pylint `ignore-patterns` alongside the existing `shelf3d_difficulty.py` / `holdout_vs_full.py`
precedent; the fast test suite (incl. the StickButton2D observational test that exercises the refactored
failure-harvest path) passes. Pre-existing mypy debt in `collect.py`'s shared failure-harvest block was
fixed cleanly (a `_failure_metadata_fn` helper) as a side benefit.

## Deferred (unchanged)

F1 / clutter / coverage-waste, the full train/val/test collection and its split sizing, learned baselines
(PIGINet / VLMPlan / LAZY), the `compare_envs.py` `EnvSpec`, and the physical-robot phase.
