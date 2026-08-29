"""Export a refined Restock3D-v3 plan into a form the real TidyBot can execute.

The refiner returns a ``bilevel_planning.structs.Plan(states, actions)`` where each state
is a ``Restock3DObjectCentricState`` and each action is the env's 11-D vector
``[base_dx, base_dy, base_drot, dj1..dj7, gripper]``. Because the sim robot **is** a
``tidybot-kinova`` (holonomic SE2 base + Kinova 7-DOF arm, world origin = the robot's
start pose, meters/radians), the state trajectory already contains the full robot command
stream -- including the custom front grasp and the ``place_tall``/``place_short`` moves,
which are just arm trajectories, not something to re-derive on the real arm.

Two levels are written (see ``experiments/spectre/deploy_scenes/ROBOT_EXECUTION.md``):

- **Level B -- absolute trajectory** (``plan_level_b.npz`` + ``.json``): per timestep the
  base SE2, the 7 joint angles, the gripper state, and the end-effector world pose; plus
  the raw 11-D delta actions and every object's world pose. The directly-replayable file.
- **Level A -- semantic waypoints** (``plan_level_a.json``): one entry per operator (pick
  / place_tall / place_short) with the base SE2 target, the grasp/place EE pose, the
  gripper event, and the object pose -- the summary for the figure and for a controller
  that runs its own IK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

_CAVEATS = [
    "World origin = the robot's home pose; +x lateral (along shelf), +y forward, z up.",
    "Units are meters and radians throughout.",
    "The 7-joint stream assumes the SAME Kinova model and arm-to-base mount as the sim.",
    "gripper < -0.5 => close, > 0.5 => open (delta action[10]); finger_state is the "
    "absolute gripper opening (open threshold ~0.01 m) -- calibrate to real hardware.",
    "The sim is kinematic: no dynamics/friction, collisions are purely geometric, so "
    "re-validate contact-rich placements on hardware.",
    "The front grasp and place_tall/place_short are ordinary arm+base motion in the "
    "trajectory; replay the joint stream (Level B) to reproduce them -- nothing custom.",
]


def _pose_of(sim, state) -> tuple[list[float], list[float]]:
    """End-effector world pose (position, quaternion) at ``state`` via the sim's FK."""
    sim.set_state(state)
    ee = sim.robot.arm.get_end_effector_pose()
    return [float(v) for v in ee.position], [float(v) for v in ee.orientation]


def _grasp_release_events(states) -> list[tuple[str, int, str]]:
    """Detect ``("grasp"|"release", timestep, object_name)`` from grasp transitions."""
    events: list[tuple[str, int, str]] = []
    prev: Optional[str] = None
    for t, st in enumerate(states):
        cur = st.grasped_object
        if cur != prev:
            if prev is None and cur is not None:
                events.append(("grasp", t, cur))
            elif prev is not None and cur is None:
                events.append(("release", t, prev))
            prev = cur
    return events


def export_plan(
    plan,
    action_plan,
    sim,
    out_dir: str | Path,
    *,
    meta: Optional[dict] = None,
) -> dict[str, str]:
    """Write Level A + Level B files for ``plan`` into ``out_dir``; return the paths.

    ``action_plan`` is the winning skeleton's operator sequence (labels the Level A
    waypoints); ``sim`` is the models' internal sim (used for forward kinematics).
    ``meta`` is an optional dict folded into both files' headers.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = list(plan.states)
    actions = list(plan.actions)
    n = len(states)

    obj_names = [o.name for o in states[0] if o.name.startswith("obj_goal")]

    # --- Level B: absolute trajectory -------------------------------------------
    base = np.array(
        [[st.base_pose.x, st.base_pose.y, st.base_pose.rot] for st in states]
    )
    joints = np.array([list(st.joint_positions) for st in states], dtype=float)
    gripper = np.array([float(st.finger_state) for st in states])
    ee_pos = np.zeros((n, 3))
    ee_quat = np.zeros((n, 4))
    for t, st in enumerate(states):
        p, q = _pose_of(sim, st)
        ee_pos[t] = p
        ee_quat[t] = q
    act = (
        np.array([np.asarray(a, dtype=float).ravel() for a in actions])
        if actions
        else np.zeros((0, 11))
    )
    obj_pose = {
        name: np.array(
            [
                list(st.get_object_pose(name).position)
                + list(st.get_object_pose(name).orientation)
                for st in states
            ]
        )
        for name in obj_names
    }

    header = {
        "units": "m/rad",
        "frame": "robot_home_pose_SE2_identity",
        "n_timesteps": n,
        "n_objects": len(obj_names),
        "object_names": obj_names,
        "operator_sequence": [
            {"op": op.name, "args": [p.name for p in op.parameters]}
            for op in action_plan
        ],
        "action_layout": ["base_dx", "base_dy", "base_drot"]
        + [f"djoint_{i}" for i in range(1, 8)]
        + ["gripper"],
        "caveats": _CAVEATS,
        **(meta or {}),
    }

    npz_path = out_dir / "plan_level_b.npz"
    np.savez_compressed(
        npz_path,
        base=base,
        joints=joints,
        gripper=gripper,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        actions=act,
        # Object-pose arrays keyed by object name (obj_goal* never collides with the
        # arrays above). ``object_names`` in the JSON header lists them.
        **{name: arr for name, arr in obj_pose.items()},
    )

    level_b_json = {
        "header": header,
        "trajectory": [
            {
                "t": t,
                "base_se2": [round(float(v), 6) for v in base[t]],
                "joints": [round(float(v), 6) for v in joints[t]],
                "gripper": round(float(gripper[t]), 6),
                "ee_pos": [round(float(v), 6) for v in ee_pos[t]],
                "ee_quat": [round(float(v), 6) for v in ee_quat[t]],
            }
            for t in range(n)
        ],
    }
    b_json_path = out_dir / "plan_level_b.json"
    b_json_path.write_text(json.dumps(level_b_json, indent=2))

    # --- Level A: semantic waypoints --------------------------------------------
    events = _grasp_release_events(states)
    grasps = [(t, name) for kind, t, name in events if kind == "grasp"]
    releases = [(t, name) for kind, t, name in events if kind == "release"]
    gi = ri = 0
    waypoints: list[dict] = []
    for op in action_plan:
        args = [p.name for p in op.parameters]
        target = next((a for a in args if a.startswith("obj_goal")), None)
        if op.name == "pick":
            if gi < len(grasps):
                t, name = grasps[gi]
                gi += 1
                waypoints.append(
                    _waypoint(
                        "pick",
                        op.name,
                        args,
                        t,
                        base,
                        ee_pos,
                        ee_quat,
                        gripper,
                        "close",
                        name or target,
                        states,
                    )
                )
        elif op.name.startswith("place"):
            if ri < len(releases):
                t, name = releases[ri]
                ri += 1
                waypoints.append(
                    _waypoint(
                        op.name,
                        op.name,
                        args,
                        t,
                        base,
                        ee_pos,
                        ee_quat,
                        gripper,
                        "open",
                        name or target,
                        states,
                    )
                )

    level_a = {"header": header, "waypoints": waypoints}
    a_path = out_dir / "plan_level_a.json"
    a_path.write_text(json.dumps(level_a, indent=2))

    return {
        "level_a": str(a_path),
        "level_b_json": str(b_json_path),
        "level_b_npz": str(npz_path),
    }


def _waypoint(
    kind: str,
    op_name: str,
    args: list,
    t: int,
    base: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    gripper: np.ndarray,
    grip_event: str,
    obj: Optional[str],
    states: list,
) -> dict:
    """One Level-A semantic waypoint at timestep ``t``."""
    st = states[t]
    obj_pose = st.get_object_pose(obj) if obj else None
    return {
        "kind": kind,
        "operator": op_name,
        "args": args,
        "object": obj,
        "timestep": int(t),  # index into Level B
        "base_se2_target": [round(float(v), 6) for v in base[t]],
        "ee_pose": {
            "position": [round(float(v), 6) for v in ee_pos[t]],
            "quaternion": [round(float(v), 6) for v in ee_quat[t]],
        },
        "gripper_event": grip_event,
        "gripper_state": round(float(gripper[t]), 6),
        "object_pose": (
            {
                "position": [round(float(v), 6) for v in obj_pose.position],
                "quaternion": [round(float(v), 6) for v in obj_pose.orientation],
            }
            if obj_pose is not None
            else None
        ),
    }
