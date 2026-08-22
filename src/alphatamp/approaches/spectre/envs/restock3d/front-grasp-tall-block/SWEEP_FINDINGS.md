# Front-Grasp Calibration Sweep — Short 5 cm Cube

**Question:** front-grasping was calibrated only for the tall block
(`0.05 × 0.05 × 0.127 m`). Can the *same* front-grasp skill be calibrated to
also handle a **short cube**, consistently (many times, not once)?

**Answer: yes.** With one calibration change — grasp the cube's **center**
instead of near its top — the short cube is front-grasped and placed on the
shelf **12/12** across 12 distinct seeds, single attempt, no retries. The grasp
**orientation is unchanged** (the tall block's 45° front grasp), and so is the
base standoff.

---

## Target object

A symmetric **5 cm cube**, half-extents `(0.025, 0.025, 0.025)`.

> Why not the *literal* default Shelf3D block `(0.05, 0.025, 0.025)`? That block
> is **10 cm** along x, so its ±x faces (which the front grasp closes on) are
> 10 cm apart — wider than the Robotiq 2F-85's **85 mm** stroke. A real-world-
> valid front grasp of its side faces is physically impossible; only a re-
> oriented grasp on its narrow faces would work. The symmetric 5 cm cube keeps
> the same 5 cm height as the default block while making the ±x faces graspable.

---

## Why this needed a sweep (the physics)

- The grasp orientation is a pure `Rx(φ)`, and `Rx` leaves the tool **+x** axis
  fixed, so the fingers straddle the **±x faces at every pitch** — the grip
  stays real-world-valid across the whole sweep. Pitch only tilts the *approach*
  axis in the y-z plane.
- At the tall block's **45°-from-vertical** pitch, the wrist grasps near
  z ≈ 0.12. A 5 cm cube's graspable band is at z ≈ 0.025–0.05 — far lower.
- The place is translate-only and rigid, so the **wrist orientation at shelf
  insertion equals the grasp orientation**. Pitch therefore trades off pick
  reachability (wants more vertical) against shelf insertion (wants more
  horizontal). The sweep scored the **full pick→place**, capturing both.
- The floor is **not** a collision body in this env, so "fingers through the
  floor" is not auto-rejected; the sweep added an explicit floor-clearance gate.

Sweep axes: **pitch** β (from vertical), **grip-height** offset (contact z above
cube center), **standoff** distance. `grip_backoff` fixed at 0.02 m.

---

## Stage A — grasp-config feasibility (cheap prune, 5 seeds/cell)

Per cell: IK solves at the grasp pose (base at the standoff), the EE marker box
overlaps the cube, no arm↔base collision, and every gripper link stays above the
floor. **35 / 45 cells passed all 5 seeds.** Structure:

- **β = 15°, 30°:** feasible at every standoff (0.62 / 0.70 / 0.78) — robust.
- **β = 45°:** feasible at d ≥ 0.70 (d = 0.62 folds the arm into the base).
- **β = 0° (top-down):** feasible only at the closer standoffs (fails d = 0.78).
- **β = 60°:** marginal — fingers approach the floor unless grip-height ≥ 0.01.

Takeaways: **d ≥ 0.70** is needed to keep the arm off its own base (same as the
tall block); more-horizontal pitches can't reach a low cube.

## Stage B — full pick→place rollouts (5 seeds/cell)

| pitch β | grip-h | standoff | grasp | goal | per-seed |
|:---:|:---:|:---:|:---:|:---:|:---|
| 30° | 0.01 | 0.70 | 5/5 | 3/5 | G g G g G |
| 45° | 0.01 | 0.72 | 5/5 | 4/5 | G g G G G |
| 30° | 0.01 | 0.78 | 5/5 | 3/5 | G g G g G |
| 15° | 0.01 | 0.70 | 5/5 | **0/5** | g g g g g |
| **45°** | **0.00** | **0.72** | **5/5** | **5/5** | **G G G G G** |

(`G` = full pick→place goal reached; `g` = grasped but place did not reach goal.)

Key insights:
- **Pick is 5/5 in every cell** — front-grasping the short cube is not the hard
  part.
- **β = 15° places 0/5:** a too-vertical wrist grasps the low cube fine but
  can't insert it into the shelf cell — direct confirmation of the pitch↔place
  coupling.
- **β = 45° is best for the place**, i.e. keep the tall block's orientation.
- **Grip-height 0.00 (grasp the cube center) beats 0.01** (5/5 vs 4/5): fingers
  sit squarely on the ±x faces at mid-height, the most stable grip.

## Held-out verification (guard against overfitting)

Winning cell re-run on **7 fresh seeds** (5, 6, 7, 42, 99, 123, 2024):
**7/7 grasp, 7/7 goal.** Combined with Stage B: **12/12 full pick→place**.

---

## The calibration

```python
# same 45° front orientation as the tall block; grasp the cube CENTER; same standoff
SMALL_CUBE_FRONT_GRASP_TRANSFORM = front_grasp_transform(45.0, 0.0, 0.02)
#   = Pose((0.0, -0.01414, 0.01414), Rx(-135°))   # cf. tall block (0,-0.02,0.057)
SMALL_CUBE_PICK_DISTANCE_BOUNDS  = (0.70, 0.75)   # unchanged from the tall block
SMALL_CUBE_HALF_EXTENTS          = (0.025, 0.025, 0.025)
```

Only the grasp *height* changed (grasp the middle of the short cube instead of
near the top of the tall one). Orientation, standoff, place logic: all unchanged.

## Reproduce

```bash
cd kinder-baselines            # monorepo venv has kinder / kinder_models / bilevel_planning
.venv/bin/python  <scratch>/sweep_front_grasp_small_cube.py     # prints Stage A + Stage B, writes sweep_results.csv
cd kinder-bilevel-planning
.venv/bin/python experiments/demo_front_shelf3d_small.py --seeds 0 42 7   # writes the mp4
```

Video: `demo_videos/front_shelf3d_small_cube.mp4` (three seeds concatenated).
