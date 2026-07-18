"""A PyBullet world with a real 7-DoF Franka Panda for geometric refinement.

Builds the sorting scene (ground plane + Panda at the origin + colored table slabs
+ one box per object) in headless ``p.DIRECT`` mode and exposes the geometric
queries a refiner needs: top-down grasp sampling, real inverse kinematics
(`calculateInverseKinematics`), collision checking against the other objects
(`getClosestPoints`), placement sampling, and frame capture for video.

This replaces the analytic feasibility model with real arm geometry: a top-down
grasp is infeasible when the arm/gripper cannot reach a collision-free config —
e.g. a *taller* blocker next to the target intrudes on the gripper. Joint layout
(verified against the shipped URDF): arm joints 0-6, fingers 9 & 10 (0=closed,
0.04=open each), IK end-effector link 11 (`panda_grasptarget`, between fingertips).
"""

from __future__ import annotations

import math

import numpy as np

from ..scene import GeometricScene
from . import _boxes

EE_LINK = 11
ARM_JOINTS = list(range(7))
FINGER_JOINTS = (9, 10)
REST_POSE = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.785]
GRIPPER_OPEN = 0.04  # per finger (metres); 0.0 = closed
IK_POS_TOL = 0.02  # accept an IK solution if FK position error < this
COLLISION_PENETRATION = -0.001  # closest-point distance below this = collision
DESCENT_HEIGHTS = (0.14, 0.09, 0.04)  # grasp-approach column probe offsets above target
APPROACH_DISTS = (
    0.14,
    0.07,
    0.0,
)  # probe distances back along the (possibly tilted) approach axis


def _quat_axis_angle(axis, angle):
    """Quaternion (x, y, z, w) for a rotation of ``angle`` about (unit-normalised)
    ``axis``."""
    ax = np.array(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    ax = ax / n
    s = math.sin(angle / 2.0)
    return [ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle / 2.0)]


class PandaWorld:
    def __init__(self, scene: GeometricScene, gui: bool = False):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.scene = scene
        self._cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        ax, ay = scene.arm_pos
        self.panda = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[ax, ay, 0.0],
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.table_ids = _boxes.add_tables(p, scene)
        self.object_ids = _boxes.add_objects(p, scene, mass=0.0)
        self.name_to_body = {v: k for k, v in self.object_ids.items()}
        self.topdown = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        self.held: str | None = None
        self.reset_arm(REST_POSE)
        self.set_gripper(GRIPPER_OPEN)

    # -- low-level joint control --------------------------------------------
    def reset_arm(self, q):
        for i, j in enumerate(ARM_JOINTS):
            self.p.resetJointState(self.panda, j, q[i])

    def set_gripper(self, width):
        for j in FINGER_JOINTS:
            self.p.resetJointState(self.panda, j, width)

    # -- inverse kinematics --------------------------------------------------
    def ik(self, pos, orn=None):
        """Solve IK for the grasp target link; return the 7 arm angles or None."""
        orn = self.topdown if orn is None else orn
        sol = self.p.calculateInverseKinematics(
            self.panda,
            EE_LINK,
            list(pos),
            orn,
            restPoses=REST_POSE + [0.0, 0.0],
            maxNumIterations=300,
            residualThreshold=1e-4,
        )
        q = list(sol[:7])
        self.reset_arm(q)
        achieved = self.p.getLinkState(self.panda, EE_LINK)[4]
        if np.linalg.norm(np.array(achieved) - np.array(pos)) > IK_POS_TOL:
            return None
        return q

    # -- collision -----------------------------------------------------------
    def _arm_hits_objects(self, ignore=()):
        """True if the arm (at its current config) collides with any object body other
        than those in ``ignore`` (names) or the currently held object."""
        self.p.performCollisionDetection()
        skip = set(ignore) | ({self.held} if self.held else set())
        for body, name in self.object_ids.items():
            if name in skip:
                continue
            pts = self.p.getClosestPoints(self.panda, body, 0.02)
            if any(pt[8] < COLLISION_PENETRATION for pt in pts):
                return True
        return False

    def collision_free(self, q, ignore=()):
        self.reset_arm(q)
        self.set_gripper(GRIPPER_OPEN)
        return not self._arm_hits_objects(ignore=ignore)

    # -- grasp / placement feasibility --------------------------------------
    def resting_z(self, name):
        """Centre z of ``name`` resting *directly on a table* (= half its height)."""
        return self.scene.by_name(name).size[2] / 2.0

    def object_z(self, name):
        """Current centre z of ``name`` in the live world (elevated when it sits in a
        tower).

        Use this to grasp an object where it actually is.
        """
        return self.p.getBasePositionAndOrientation(self.name_to_body[name])[0][2]

    def support_top_z(self, lower):
        """Top-surface z of support block ``lower`` at its current height."""
        return self.object_z(lower) + self.scene.by_name(lower).size[2] / 2.0

    def top_grasp_feasible(self, xy, target_name, z=None):
        """Real top-down grasp/placement test at planar ``xy`` for ``target_name``:

        IK + collision-free at height ``z`` AND down the approach column. ``z`` defaults
        to the object's *current* height (so grasping/unstacking a stacked block reaches
        it); placement callers pass the resting/tower height explicitly.
        Returns (q_pre, q_grasp) on success, else None.
        """
        if z is None:
            z = self.object_z(target_name)
        # descent column: probe a few heights, then the grasp itself
        q_pre = None
        for dz in DESCENT_HEIGHTS:
            q = self.ik((xy[0], xy[1], z + dz))
            if q is None or not self.collision_free(q, ignore=(target_name,)):
                return None
            if q_pre is None:
                q_pre = q
        q_grasp = self.ik((xy[0], xy[1], z))
        if q_grasp is None or not self.collision_free(q_grasp, ignore=(target_name,)):
            return None
        return q_pre, q_grasp

    # -- sampled (non-top-down) grasps: used by the backtracking refiner --------
    def grasp_pose(
        self,
        xy,
        name,
        rng,
        yaw_range=(0.0, math.pi),
        tilt_max=0.3,
        xy_jitter=0.01,
        z=None,
    ):
        """Sample a grasp pose: top-down rotated by a yaw about the approach axis,
        a tilt away from straight-down, and a small off-centre xy offset. ``z`` defaults
        to the object's *current* height (grasping); placement callers pass the target
        placement height. Returns (pos, orn) for the grasp-target link."""
        if z is None:
            z = self.object_z(name)
        pos = (
            xy[0] + rng.uniform(-xy_jitter, xy_jitter),
            xy[1] + rng.uniform(-xy_jitter, xy_jitter),
            z,
        )
        yaw = rng.uniform(*yaw_range)
        tilt = rng.uniform(0.0, tilt_max)
        tilt_dir = rng.uniform(0.0, 2 * math.pi)
        # local-z yaw (gripper spin), then a world-frame tilt off vertical
        _, orn = self.p.multiplyTransforms(
            [0, 0, 0],
            self.topdown,
            [0, 0, 0],
            self.p.getQuaternionFromEuler([0, 0, yaw]),
        )
        q_tilt = _quat_axis_angle((math.cos(tilt_dir), math.sin(tilt_dir), 0.0), tilt)
        _, orn = self.p.multiplyTransforms([0, 0, 0], q_tilt, [0, 0, 0], orn)
        return pos, orn

    def grasp_feasible_at(self, pos, orn, target_name):
        """IK + collision-free at the grasp config and back along its approach axis
        (generalises ``top_grasp_feasible`` to an arbitrary orientation).

        Returns (q_pre, q_grasp) or None.
        """
        R = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        approach = R[:, 2]  # world direction the gripper points (toward the object)
        q_pre = q_grasp = None
        for d in APPROACH_DISTS:  # 0.14 (pre-grasp) ... 0.0 (grasp)
            probe = (
                pos[0] - approach[0] * d,
                pos[1] - approach[1] * d,
                pos[2] - approach[2] * d,
            )
            q = self.ik(probe, orn)
            if q is None or not self.collision_free(q, ignore=(target_name,)):
                return None
            if q_pre is None:
                q_pre = q
            q_grasp = q
        return q_pre, q_grasp

    # -- backtracking support: snapshot / restore world state -------------------
    def snapshot(self):
        """Capture object base poses + held, so a backtracking step can be undone."""
        poses = {
            name: self.p.getBasePositionAndOrientation(body)
            for name, body in self.name_to_body.items()
        }
        return {"poses": poses, "held": self.held}

    def restore(self, snap):
        for name, (pos, ornb) in snap["poses"].items():
            self.p.resetBasePositionAndOrientation(self.name_to_body[name], pos, ornb)
        self.held = snap["held"]

    def sample_placement(self, table_name, obj_name, rng, n_samples=24):
        """Sample a reachable, collision-free, non-overlapping placement on a table.

        Returns (xy, q_place, (q_pre, q_grasp_like)) or None.
        """
        t = self.scene.table(table_name)
        half = t.half_extent - self.scene.by_name(obj_name).footprint_radius
        z = self.resting_z(obj_name)  # object will rest directly on the table
        for _ in range(n_samples):
            xy = (
                t.center[0] + rng.uniform(-half, half),
                t.center[1] + rng.uniform(-half, half),
            )
            if not self._placement_footprint_free(xy, obj_name):
                continue
            grasp = self.top_grasp_feasible(
                xy, obj_name, z=z
            )  # placing == reach + clear column
            if grasp is None:
                continue
            return xy, grasp[1], grasp
        return None

    def _placement_footprint_free(self, xy, obj_name, ignore=()):
        """True if no object's footprint overlaps ``xy``.

        ``ignore`` excludes the
        support block when stacking (the new block sits *on* it at the same xy).
        """
        r = self.scene.by_name(obj_name).footprint_radius
        skip = {obj_name, self.held, *ignore}
        for name, body in self.name_to_body.items():
            if name in skip:
                continue
            ox, oy, _ = self.p.getBasePositionAndOrientation(body)[0]
            other_r = self.scene.by_name(name).footprint_radius
            if math.hypot(ox - xy[0], oy - xy[1]) < r + other_r + 0.01:
                return False
        return True

    # -- world mutation (between actions) -----------------------------------
    def move_object(self, name, xy, z=None):
        body = self.name_to_body[name]
        if z is None:
            z = self.resting_z(name)
        self.p.resetBasePositionAndOrientation(body, [xy[0], xy[1], z], [0, 0, 0, 1])

    def lift_object_away(self, name):
        """Park a grasped object out of the workspace so it stops colliding."""
        body = self.name_to_body[name]
        self.p.resetBasePositionAndOrientation(body, [0.0, 0.0, 2.0], [0, 0, 0, 1])

    def object_xy(self, name):
        return tuple(
            self.p.getBasePositionAndOrientation(self.name_to_body[name])[0][:2]
        )

    def column_members(self, xy, eps=0.01):
        """Names of objects stacked at (approximately) planar ``xy`` -- the tower
        column.

        Used so stacking onto a tower ignores the blocks beneath the support (they share
        the support's xy and are not lateral obstructions).
        """
        out = []
        for name, body in self.name_to_body.items():
            ox, oy, _ = self.p.getBasePositionAndOrientation(body)[0]
            if math.hypot(ox - xy[0], oy - xy[1]) <= eps:
                out.append(name)
        return out

    def reset_objects(self):
        """Restore every object to its initial scene pose (for backtracking).

        Honour the scene pose z so **pre-stacked initial towers keep their height**
        across passes (flat scenes have ``pose[2] == size/2``, so no change there).
        """
        for o in self.scene.objects:
            self.move_object(o.name, (o.pose[0], o.pose[1]), z=o.pose[2])
        self.held = None
        self.reset_arm(REST_POSE)
        self.set_gripper(GRIPPER_OPEN)

    # -- rendering -----------------------------------------------------------
    def capture_frame(self, width=480, height=360, view="oblique"):
        p = self.p
        if view == "topdown":
            eye, target, up = [0.45, 0.0, 1.4], [0.45, 0.0, 0.0], [1, 0, 0]
        else:
            eye, target, up = [1.25, -1.05, 0.95], [0.42, 0.0, 0.05], [0, 0, 1]
        vm = p.computeViewMatrix(eye, target, up)
        pm = p.computeProjectionMatrixFOV(
            fov=55, aspect=width / height, nearVal=0.05, farVal=4.0
        )
        w, h, rgba, _, _ = p.getCameraImage(
            width, height, vm, pm, renderer=p.ER_TINY_RENDERER
        )
        return np.reshape(np.array(rgba, dtype=np.uint8), (h, w, 4))[:, :, :3]

    def close(self):
        try:
            self.p.disconnect(self._cid)
        except Exception:  # pragma: no cover
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
