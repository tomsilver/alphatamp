# Executing a SPECTRE plan on the real TidyBot

Each run writes two representations of the refined plan into `outputs/<scene name>/`. Both
describe the same motion; either can be used, depending on the available control stack.

The key fact that makes this straightforward: the simulator's robot **is** a `tidybot-kinova`
— a holonomic (SE2) mobile base plus a Kinova 7-DOF arm — with the world origin at the robot's
home pose, in meters and radians. The plan the refiner produces is therefore already a robot
command stream. In particular, the **custom front grasp** and the **`place_tall`** /
**`place_short`** motions are ordinary arm+base trajectories in the plan; the trajectory is
replayed rather than re-derived with a grasp planner on the real robot.

## Level B — absolute trajectory (`plan_level_b.npz` / `.json`) — the primary artifact

`plan_level_b.npz` contains, one row per timestep `t`:

| array | shape | meaning |
|---|---|---|
| `base` | `(T, 3)` | base pose `(x, y, θ)` in the world/home frame (m, m, rad) |
| `joints` | `(T, 7)` | the 7 Kinova joint angles (rad) |
| `gripper` | `(T,)` | finger opening (m); ~`≤0.01` is closed |
| `ee_pos` | `(T, 3)` | end-effector position (m), world frame |
| `ee_quat` | `(T, 4)` | end-effector orientation quaternion `(x, y, z, w)` |
| `actions` | `(T-1, 11)` | per-step deltas `[base_dx, base_dy, base_dθ, dj1..dj7, gripper]` |
| `obj_goal*` | `(T, 7)` | each object's world pose `(x,y,z, qx,qy,qz,qw)` |

`plan_level_b.json` mirrors `base`/`joints`/`gripper`/`ee_pos`/`ee_quat` per timestep for quick
inspection.

**Joint-space replay (recommended, since the arm and base are the same model):** stream
`joints[t]` to the arm and drive the base to `base[t]` in sequence, toggling the gripper when
`gripper[t]` crosses the open/close threshold (equivalently, when `actions[t][10]` is `< -0.5`
for close / `> 0.5` for open). The steps are already densified to small increments.

**Cartesian replay:** command the end-effector to `ee_pos[t]` / `ee_quat[t]` with a separate IK
solver and drive the base to `base[t]`. This path suits a real arm whose mounting or DH differs
from the simulation URDF, where the raw `joints` stream would not transfer directly.

## Level A — semantic waypoints (`plan_level_a.json`) — the summary

One entry per operator (`pick`, `place_tall`, `place_short`), in execution order:

```json
{
  "kind": "pick",              // or "place_tall" / "place_short"
  "operator": "pick",
  "args": ["robot", "obj_goal3"],
  "object": "obj_goal3",
  "timestep": 812,             // index into Level B
  "base_se2_target": [x, y, theta],
  "ee_pose": { "position": [...], "quaternion": [...] },
  "gripper_event": "close",    // "close" for pick, "open" for place
  "object_pose": { "position": [...], "quaternion": [...] }
}
```

This suits the paper figure, or a controller that only needs the grasp/place end-effector pose
and base target per operator (running its own base motion, IK, and a straight approach). The
`timestep` field links each waypoint back to the dense Level-B trajectory.

For a controller that wants a pre-grasp / pre-place approach pose: the simulation backs the
end-effector off along the tool approach axis by `0.12 m` before a grasp and `0.13 m` before a
place; the same offset applied to `ee_pose` along its local approach direction reproduces it,
as do the dense Level-B states leading into the waypoint's `timestep`.

## Calibration notes and caveats

- The robot's home pose is the world origin; the scene is measured relative to a repeatable
  home pose (see `README.md`).
- Joint replay assumes the same Kinova model and arm-to-base mount as the simulation URDF. When
  these differ, the Cartesian (`ee_pos` / `ee_quat`) stream with a separate IK solver applies
  instead.
- The simulation `gripper` / `finger_state` (open threshold ~`0.01 m`) maps to the physical
  gripper's open/close command; the plan only ever fully opens or fully closes.
- The simulation is kinematic: no dynamics or friction, and collisions are purely geometric, so
  contact-rich placements warrant re-validation on hardware.
- This is a proof of concept — one plan, executed once. An e-stop should remain within reach.
