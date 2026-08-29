# Restock3D-v3 real-robot deployment (SPECTRE proof-of-concept)

This folder is a self-contained kit for running the trained SPECTRE re-ranker on a
hand-specified Restock3D-v3 scene and exporting the resulting plan for a real TidyBot. It is a
proof of concept — one plan, planned live and executed once — not a robust deployment. There is
no perception: the simulated scene is exactly the one described in the scene file.

## Folder layout

- `deploy.py` — the live planning script (entry point).
- `deploy_scene.py`, `robot_export.py` — the scene builder and the plan exporter it uses.
- `checkpoint/` — the trained SPECTRE checkpoint (`best.pt`) and vocab (`train_vocab.json`).
- `scenes/` — input scenes, one directory per scene (a `demo6/` example is included).
- `outputs/` — where each run writes its plan (one subdirectory per scene).
- `README.md` (this file) and `ROBOT_EXECUTION.md` (how to run the exported plan on the robot).

## Running

With the repository's virtual environment active:

```bash
cd experiments/spectre/restock3d_deploy
python deploy.py                            # runs the bundled scenes/demo6 example
python deploy.py --scene scenes/myscene --render
```

The checkpoint, vocab, input scene and output directory all default to this folder. A run
prints every attempt (the skeleton tried, its operator sequence, and why a failed refinement
failed) and writes into `outputs/<scene name>/`:

- `plan_level_a.json` — per-operator waypoints (base pose + grasp/place end-effector pose +
  gripper events).
- `plan_level_b.json` / `plan_level_b.npz` — the full absolute base+joint trajectory, directly
  replayable on the TidyBot.
- `plan.mp4` — with `--render`, a simulation video of the plan.

The default pool size (`K_max`) and per-candidate refinement cap come from the
`restock3d_v3_real` collection budgets for the scene's object count; `--k-max` /
`--refinement-timeout` override them.

Runs are deterministic: the script pins the Python hash seed and re-executes, so a given
scene always yields the same candidate pool and plan. A scene is solvable only if that pool
contains a feasible skeleton; if a run reports `NO PLAN FOUND`, `--k-max` enumerates more of
the pool.

See [`ROBOT_EXECUTION.md`](ROBOT_EXECUTION.md) for running the exported plan on the real robot.

## Writing a scene

A scene is a directory under `scenes/` containing a `scene.yaml`:

```yaml
objects:
  - name: obj_goal1      # optional; defaults to obj_goal1..N in file order
    width:  0.05         # full x-extent (m)   — the side the gripper straddles
    height: 0.15         # full z-extent (m)   — the F3-critical dimension
    depth:  0.05         # full y-extent (m)   — optional, default 0.05
    floor:  [-0.70, 0.75]  # (x, y) floor position, robot frame
  # ... one entry per object
# Optional (omit to match the trained shelf):
# shelf:    { x: 0.4, y: 1.4 }
# sections: { clearances: [0.27, 0.22] }   # [tall(bottom), short(top)]
```

Each object is modeled as an axis-aligned box `width × depth × height`. Real objects should be
ones that grasp like a box (cuboids, or cylinders/toys that a box closing on the ±x faces can
bound). The front grasp closes on the object's left/right (±x) faces, so `width` is the
dimension that must fit the gripper.

## Coordinate frame

All coordinates are in the robot's world frame (origin = the robot's start / home pose), in
meters and radians. Object floor positions are measured relative to a repeatable robot home
pose.

- **+x** is the lateral axis — left↔right across the shelf's wide face. The shelf's wide
  dimension (~0.60 m) runs along x, objects pack left-to-right along x, and the staging area
  and the shelf are separated along x. Facing the shelf, +x is to the right.
- **+y** is the forward / depth axis — from the robot toward the shelf. The robot stages in a
  corridor at low y and approaches every object and the shelf from the front (its −y side),
  reaching in +y (the shelf's open front faces −y; its shallow depth ~0.25 m runs along y). The
  shelf is centered near (x ≈ 0.4, y ≈ 1.4).
- **z** is up; objects rest on the floor (z = 0).

Objects to store start on the floor between the robot and the shelf and off to one side of it,
in a band roughly x ∈ (−0.80, −0.20) (lateral) × y ∈ (0.60, 1.20) (forward).

## The shelf

The shelf has two sections, stacked:

- **Tall section** (bottom): objects up to height ≤ 0.17 m.
- **Short section** (top): objects up to height ≤ 0.12 m — the top shelf has less headroom for
  the arm to insert.

An object taller than 0.12 m must go in the tall (bottom) section; taller than 0.17 m fits
nowhere. This is what makes object selection matter, and it is the structure the re-ranker
learns to exploit.

## Constraints

The validator reports a warning for each of these (warnings, not errors — a real object may
legitimately sit outside a soft bound), and reports when a scene is geometrically unsolvable:

- **Width** ∈ [0.02, 0.08] m. The gripper aperture is ~0.09 m; wider objects do not close.
- **Height** ≤ 0.17 m (otherwise the object fits in no section).
- **Depth** defaults to 0.05 m. It may be set per object — collision in simulation is exact —
  but the ranker was trained on ~0.05 m depth, so other depths are mildly out-of-distribution.
- **Floor** within x ∈ (−0.80, −0.20), y ∈ (0.60, 1.20), with objects at least 0.12 m apart
  (center-to-center), so the front grasp of one object is not blocked by a neighbour.
- **At least one feasible packing** must exist: the objects must fit into the two sections
  respecting the per-level capacity Σ(width) + 0.06·(n_in_level − 1) + 0.08 ≤ 0.50 m. If no
  assignment works, the scene is unsolvable and no plan can succeed.
- **Object count** is ideally 6–9 (the range the ranker was trained on). Other counts run — the
  model is count-agnostic — but are out-of-distribution, and the default budget clamps to the
  nearest of strata 6/7/8/9.
