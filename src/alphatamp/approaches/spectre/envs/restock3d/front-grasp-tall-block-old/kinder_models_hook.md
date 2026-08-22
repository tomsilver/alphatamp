# Optional: patching `kinder_models` (Option B)

The shipped `front_grasp_skills.py` uses a **self-contained** `FrontGroundPickController`
(it overrides `step()` fully), so you do **not** need to touch `kinder_models`. This
document is only for teams who **own/fork `kinder_models`** and would rather add a small
reusable hook to the base `GroundPickController` and keep the front pick minimal.

If you apply this patch, you can replace the shipped self-contained
`FrontGroundPickController` with the shorter version at the bottom.

## The patch

File: `kinder_models/src/kinder_models/kinematic3d/shelf3d/parameterized_skills.py`,
class `GroundPickController`.

### 1. Make the grasp transform overridable (in `__init__`)

Add one line at the end of `GroundPickController.__init__`:

```python
        self._last_gripper_state: float = 0.0
        # NEW: object-frame -> EE grasp transform, overridable by subclasses.
        self._grasp_transform = GRASP_TRANSFORM_TO_OBJECT
```

### 2. Use it in `step()`

In the pre-grasp branch of `step()`, change the grasp-pose computation from the module
constant to the instance attribute:

```python
                target_end_effector_pose = multiply_poses(
                    target_grasp_pose_world,
-                   GRASP_TRANSFORM_TO_OBJECT,
+                   self._grasp_transform,
                )
```

### 3. Extract the approach planner (so subclasses can add a standoff + reach-in)

Pull the IK + motion-plan + remap block out of `step()` into a method, and call it:

```python
    def _plan_grasp_approach(self, target_end_effector_pose):
        """Plan the arm approach to the grasp pose (IK + joint-space MP).

        Subclasses may override to add a pre-grasp standoff + straight-line reach.
        """
        try:
            joint_positions = inverse_kinematics(
                self._sim.robot.arm, target_end_effector_pose,
                validate=True, set_joints=False,
            )
        except InverseKinematicsError as e:
            raise TrajectorySamplingFailure(
                f"IK failed for target pose {target_end_effector_pose}") from e
        joint_plan = run_motion_planning(
            self._sim.robot.arm,
            initial_positions=self._sim.robot.arm.get_joint_positions(),
            target_positions=joint_positions,
            collision_bodies=self._sim._get_collision_object_ids(),
            seed=0, physics_client_id=self._sim.physics_client_id,
        )
        if joint_plan is None:
            raise TrajectorySamplingFailure("Motion planning failed")
        return remap_joint_position_plan_to_constant_distance(
            joint_plan, self._sim.robot.arm,
            max_distance=self._sim.config.max_action_mag / 2,
        )
```

and in `step()`'s pre-grasp branch replace the inline IK+MP+remap with:

```python
                joint_plan = self._plan_grasp_approach(target_end_effector_pose)
                self._current_arm_joint_plan = joint_plan[1:]
```

These three edits are **behavior-preserving** for the default top-down pick (all existing
Shelf3D tests still pass).

## Minimal `FrontGroundPickController` once the hook exists

With the patch applied, the front pick only needs to set the transform and override the
approach planner (instead of the whole `step()`):

```python
class FrontGroundPickController(GroundPickController):
    def __init__(self, objects, sim):
        super().__init__(objects, sim)
        self._grasp_transform = FRONT_GRASP_TRANSFORM_TO_OBJECT

    def sample_parameters(self, x, rng):
        distance = rng.uniform(*FRONT_PICK_DISTANCE_BOUNDS)
        rot = rng.uniform(*FRONT_PICK_ROT_BOUNDS)
        return np.array([distance, rot])

    def _plan_grasp_approach(self, target_end_effector_pose):
        # ... the standoff + run_smooth_motion_planning_to_pose +
        #     smoothly_follow_end_effector_path body from the shipped
        #     front_grasp_skills.py::_front_plan_grasp_approach ...
```

**Recommendation:** unless you already fork `kinder_models`, prefer the shipped
self-contained pick (Option A) — it needs no upstream changes and works against a stock,
pip/git-installed `kinder_models`.
