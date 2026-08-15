# TidyBot Gripper: Dimensions, Mechanics, and Graspable Object Range

Everything about the TidyBot's gripper relevant to grasping cubes — the physical dimensions, how the
fingers open/close, how a grasp is detected in each simulator vs. on hardware, and what range of cube
sizes it can comfortably grasp. Numbers are read directly from the model/asset files (cited at the
bottom); values that are derived rather than literally in a file are flagged.

## TL;DR

- **Gripper:** Robotiq **2F-85** (2-finger, single-motor, 4-bar linkage) mounted on the Kinova Gen3
  7-DOF arm. Same model in MuJoCo, PyBullet, and on the real robot.
- **Max opening (stroke):** **~85 mm** pad-to-pad when fully open → **0 mm** fully closed.
- **Fingertip pads (the gripping faces):** **~22 mm wide × ~37.5 mm tall**, ~6–8 mm thick, rubber-ish
  friction (0.6–0.7).
- **1 actuated DOF.** Sim command = a finger "state" scalar (**0.0 = open, 0.8 = closed**, a joint
  angle in radians); real robot command = normalized position (**0.0 = open, 1.0 = closed**). No
  width-in-meters command anywhere — you command "how closed," not "how many mm."
- **Grasp point:** between the fingertips, **0.12 m** beyond the arm flange (this is the frame arm IK
  targets: `tool_frame` / `pinch_site`).
- **Cube sizes actually used:** **3–5 cm** across the grasp axis (see table) — all comfortably within
  the 85 mm stroke.
- **⚠️ Big caveat:** in the **kinematic (PyBullet) planning envs** (Shelf3D, Table3D, Transport3D, …)
  grasping is a **collision-marker abstraction** — physical pad width is **not** enforced. The 85 mm
  limit only bites in the **MuJoCo physics sim and on the real robot**.

---

## 1. What the gripper is

A **Robotiq 2F-85**: two fingers driven by a **single motor** through a 4-bar linkage, so the two
fingers always mirror each other — you cannot move them independently. "2F-85" = 2 Fingers, **85 mm**
maximum stroke. It sits on the Kinova Gen3 arm's flange.

In the models this shows up as:
- **1 real actuated joint** (`finger_joint`, range **0–0.8 rad**), with the rest of the linkage
  following it:
  - **MuJoCo:** 8 gripper joints reduced to 1 controllable DOF via a tendon + equality constraints; a
    single `fingers_actuator` with `ctrlrange 0–255` (Robotiq's native 0 = open … 255 = closed),
    `forcerange ±5`.
  - **PyBullet:** 1 real joint + **5 mimic joints** coupled to it (`finger_state_to_joints(s) =
    [s, s, s, s, −s, −s]`).
- **Pinocchio URDF** (used only for gravity compensation on the real arm) collapses the gripper to a
  single rigid block with no articulated fingers — it doesn't model finger motion at all.

---

## 2. Dimensions

| Quantity | Value | Source / note |
|---|---|---|
| Max opening (fully open, pad-to-pad) | **~85 mm** | By construction (`stroke="85"` in the 2F-85 xacro). Not a literal gap number in the files; it's the nominal 2F-85 stroke. |
| Min opening (fully closed) | **~0 mm** (pads meet) | Derived from the linkage at `finger_joint = 0.8 rad`. |
| Fingertip pad — width | **22 mm** | MuJoCo `pad_box` half-extents (0.011 in X); PyBullet pad box `0.022` wide. |
| Fingertip pad — height | **37.5 mm** | Two stacked 18.75 mm boxes (MuJoCo); PyBullet pad box `0.0375` tall. |
| Fingertip pad — thickness | **~6.35–8 mm** | 8 mm in MuJoCo, 6.35 mm in the PyBullet URDF (models differ slightly). |
| Pad friction | **0.6–0.7** | MuJoCo `pad_box1`/`pad_box2`. |
| Grasp point offset from flange | **0.12 m** | `tool_frame`/`pinch_site` sits 0.12 m past `end_effector_link`, between the fingertips. This is the IK target (`ee_offset = 0.12`). |

**Space between the fingers** is therefore **0 mm (closed) to ~85 mm (open)**, and the flat gripping
surface that contacts an object is about a **22 mm × 37.5 mm** rectangle on each finger. The exact
open/closed gap in meters is *not* written in any asset file — it emerges from the linkage forward
kinematics; only the joint travel (0 → 0.8 rad ↔ ~85 → 0 mm) is explicit.

---

## 3. How the fingers are commanded (and the open/closed conventions)

There are **two different "closedness" scales** in play — both use 0 = open, but the closed value
differs, so don't mix them up:

| Context | Command | Open | Closed | Where |
|---|---|---|---|---|
| Kinematic sim (`finger_state`) | joint-angle scalar | **0.0** | **0.8** | `KinovaGen3RobotiqGripperPyBulletRobot` (`open_fingers_state=0.0`, `closed_fingers_state=0.8`) |
| Planner action (last dim) | thresholded | `> 0.5` → open | `< −0.5` → close | `Kinematic3DRobotActionSpace`; between = no change |
| Real robot | normalized position | **0.0** | **1.0** | `kinova.py::open_gripper()`→0, `close_gripper()`→1 (Robotiq `GRIPPER_POSITION` mode) |
| MuJoCo actuator | ctrl | **0** | **255** | `fingers_actuator` tendon, Robotiq native units |

On the real robot the low-level loop maps the normalized `[0,1]` command to motor units `[0,100]`; there
is **no width-in-millimeters API** — you always command "fraction closed," never a target gap.

---

## 4. How a grasp actually happens (this determines what's graspable)

**This is the most important section for judging graspability, because the two simulators do it very
differently.**

### Kinematic3D envs (PyBullet) — the Shelf3D/Table3D/Transport3D/etc. planning envs
Grasping here is a **geometric abstraction, not physics**:
- There's an **invisible collision box at the end-effector** (`end_effector_viz`), size
  **20 mm × 20 mm × 70 mm** (half-extents `0.01, 0.01, 0.035`), re-posed to the hand every step.
- On a "close" action, a grasp **succeeds if and only if exactly one movable object overlaps that box**
  (0 objects or ≥2 objects → the grasp fails).
- The fingers then close **cosmetically** — the closing loop stops at `closed_fingers_state = 0.8`
  whether or not the pads ever touch the object. **The grasp holds regardless of the object's width.**
- **Release** succeeds when the held object is within **5 mm** (`min_placement_dist`, Transport3D uses
  10 mm) of a supporting surface.

➡️ **In these envs, physical pad width and the 85 mm stroke impose no limit.** "Graspable" means the
object fits/overlaps that 2 cm × 2 cm × 7 cm marker and is the *only* object doing so. You could even
"grasp" something wider than 85 mm in this sim — but it would **not** work on the real robot. Treat the
85 mm stroke and pad geometry as the *real* constraint that this abstraction hides.

### Dynamic3D (MuJoCo) and the real robot
These are **real physics** — the fingers must physically close onto the object and hold it by friction.
Here the **85 mm stroke, the 22 × 37.5 mm pad faces, and the 0.6–0.7 friction all matter**, exactly as
on hardware.

---

## 5. Cube / object sizes used today

Full edge lengths in meters (2 × half-extent). "Grasp axis" = the dimension that ends up between the
pads (the pick skills rotate the hand ~90°, so for the elongated Shelf3D blocks a *short* 5 cm axis is
what's grasped).

| Env | Object | Full size (m) | Grasp-axis width | Config knob |
|---|---|---|---|---|
| **Shelf3D (kinematic)** | cuboid | **0.10 × 0.05 × 0.05** | **~50 mm** | `Shelf3DEnvConfig.block_half_extents` |
| **PrplLab3D** | cuboid | **0.10 × 0.05 × 0.05** | ~50 mm | `prpl3d.py` |
| Transport3D | cube | **0.05 × 0.05 × 0.05** | ~50 mm | `block_size=0.05` |
| Table3D | cube | **0.05³** | ~50 mm | `block_size=0.05` |
| Obstruction3D | target/obstructions | target ~0.032–0.08; obstructions ~0.02–0.06 (randomized) | ~30–80 mm | per-episode random |
| **Shelf3D (MuJoCo / real TidyBot)** | cube | **0.04 × 0.04 × 0.04** | **~40 mm** | task JSON `"size": 0.02` (half-extent) |
| Real-robot perceiver placeholder | assumed cube | **0.03 × 0.03 × 0.03** | ~30 mm | `bb_*=0.03` in `kinder_ground_perceiver.py` |

So across the whole suite the grasp-axis widths used are **~30–50 mm** (up to ~80 mm for the largest
randomized Obstruction3D blocks).

---

## 6. What range of cube sizes can the gripper *easily* grasp?

**On the real robot / in MuJoCo physics** (bound by the 85 mm stroke and the pad geometry):

- **Hard maximum:** ~**85 mm** across the grasp axis (the stroke). Leave margin, so practically aim for
  **≤ ~70–75 mm** so the fingers don't need to open to their mechanical limit.
- **Practical easy range:** roughly **~15 mm to ~70 mm** across the grasp axis.
  - Below ~10–15 mm, a top grasp risks the fingers closing past the object or the pads/fingertips
    striking the surface underneath before gripping (the sim even reverts grasps where the fingers
    penetrate the table/floor).
  - The **22 mm-wide, 37.5 mm-tall** pad faces give a solid contact patch on objects in this range;
    objects taller than the pad still grasp fine (the pad contacts a partial height).
- **The cubes we actually use (30–50 mm) sit squarely in the easy zone** — well clear of both the small
  and large ends. A 3 cm perceiver-placeholder cube, a 4 cm MuJoCo cube, and a 5 cm kinematic cube are
  all comfortable grasps.

**In the kinematic planning envs** (collision-marker abstraction): there is effectively **no width
limit** — any object that overlaps the ~2 cm × 2 cm × 7 cm end-effector marker (and is the only object
there) is "grasped." This is convenient for planning but means the sim will happily accept grasps that
the real 2F-85 could not perform. When designing new object sizes for anything intended to run on
hardware, size them to the **real** range above (≤ ~70 mm grasp axis), not to whatever the kinematic
marker will accept.

**Rules of thumb for new variants:**
- Keep the **grasp-axis** dimension between **~15 mm and ~70 mm** for reliable real-robot grasps
  (30–50 mm is the proven sweet spot).
- If a block is elongated, remember the pick skill rotates the hand ~90° and grasps a **short** axis —
  size *that* axis into range, not the long one.
- Pads are only ~22 mm wide and ~37.5 mm tall; very large/heavy objects, or objects whose graspable
  face is smaller than the pad, will be less stable even if they technically fit.

---

## 7. Sources (files read)

- **MuJoCo gripper geometry & actuator:** `kindergarden/src/kinder/envs/dynamic3d/models/kinova_gen3/gen3_2f85.xml` (pads, driver joint `0–0.8`, tendon `fingers_actuator` `0–255`, `pinch_site` at −0.181525).
- **PyBullet gripper model:** `pybullet-helpers/.../assets/urdf/kortex_description/grippers/robotiq_2f_85/urdf/robotiq_arg2f_85_model_macro.xacro` (`stroke="85"`), loaded via `.../gen3_7dof.urdf`.
- **PyBullet robot class / finger_state:** `pybullet-helpers/src/pybullet_helpers/robots/kinova.py` (`KinovaGen3RobotiqGripperPyBulletRobot`: `open=0.0`, `closed=0.8`, 6 finger joints, mimic mapping).
- **Grasp/release detection & EE marker:** `kindergarden/src/kinder/envs/kinematic3d/base_env.py` (`end_effector_viz` box `0.02×0.02×0.07`, single-object grasp rule, `min_placement_dist=5e-3`); action semantics in `kinematic3d/utils.py`.
- **Cube sizes:** `kindergarden/src/kinder/envs/kinematic3d/{shelf3d,transport3d,table3d,obstruction3d,prpl3d}.py`; `kindergarden/src/kinder/envs/dynamic3d/tasks/Shelf3D/Shelf3D-o{1,2,8}.json` + `dynamic3d/objects/primitive_objects.py`; placeholder in `prpl-tidybot/.../perceivers/kinder_ground_perceiver.py`.
- **Real gripper command:** `prpl-tidybot/src/prpl_tidybot/kinova.py` (`open_gripper()=0`, `close_gripper()=1`, `GRIPPER_POSITION` mode; normalized `[0,1]`), `arm_server.py`, `arm_controller.py`.
- **Grasp frame / offset:** `prpl-tidybot/src/prpl_tidybot/ik_solver.py` (`ee_offset=0.12` → `pinch_site` at −0.181525) and `tool_frame` in the URDFs.

> Note: the pad-to-pad gap as a function of finger angle is **not** given in closed form in any file —
> only the joint travel (0 → 0.8 rad ↔ ~85 → 0 mm) is explicit. The 85 mm / 0 mm endpoints are the
> nominal 2F-85 stroke, confirmed by the `stroke="85"` model parameter.
