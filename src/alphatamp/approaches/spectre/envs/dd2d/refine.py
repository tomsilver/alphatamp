"""Skeleton refinement -- the inner search that binds continuous parameters.

This mirrors REFINE-PLAN (PIGINet paper) / Algorithm 2 of the LAZY paper
(``policy-guided-lazy-tamp/lifted/sampling.py``): for each action in a skeleton we
sample continuous values (grasp / placement) and check geometric feasibility; on a
dead-end we backtrack to the first action and resample, up to a bounded number of
attempts. Whether the skeleton is fully bound within the budget is the (noisy)
feasibility *label* used for a PIGINet example.

The geometric model is intentionally simple but captures the sorting obstruction
mechanic faithfully: a top-down grasp/placement is blocked when a *taller* object
(a blocker, 10cm vs a 4.5cm block) sits within gripper clearance of the target.
Moving such a blocker aside (an action the diverse planner does enumerate) clears
the obstruction -- exactly the feasibility structure a plan-feasibility predictor
is meant to learn. Feasibility is stochastic across attempts because placement
poses are sampled, matching how REFINE-PLAN searches over sampler outputs.

The model here is render-backend independent; the PyBullet / numpy-2D backends in
``geometry/`` are only for rendering. (Design note recorded in notebook.md.)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .scene import GeometricScene
from .skeleton import Action, Skeleton

# gripper clearance: a taller object whose footprint is within this horizontal
# margin of the target obstructs a top-down grasp/placement.
GRASP_CLEARANCE = 0.05
PLACEMENT_SAMPLES = 24  # poses tried per place/stack action before declaring dead-end


@dataclass
class _ObjState:
    name: str
    xy: tuple[float, float]
    z: float
    height: float
    radius: float  # footprint radius
    table: str | None  # current table (None if held / stacked)
    on_block: str | None = None


@dataclass
class BoundStep:
    action: Action
    params: dict  # bound continuous params, e.g. {"grasp": (...), "place_xy": (...)}


@dataclass
class RefineResult:
    status: str  # "feasible" | "infeasible"
    steps_bound: int  # furthest prefix length bound in any attempt ("how far we got")
    plan_length: int
    n_attempts: int
    failure_action: str | None  # str(action) where the best attempt got stuck
    bound_plan: list[BoundStep] = field(default_factory=list)
    elapsed: float = 0.0  # wall-clock seconds the refinement took (for anytime curves)

    @property
    def feasible(self) -> bool:
        return self.status == "feasible"


# --------------------------------------------------------------------------- #
# geometric helpers
# --------------------------------------------------------------------------- #
def _reachable(scene: GeometricScene, xy: tuple[float, float]) -> bool:
    ax, ay = scene.arm_pos
    return math.hypot(xy[0] - ax, xy[1] - ay) <= scene.arm_reach


def _top_down_clear(target: _ObjState, world: dict[str, _ObjState]) -> bool:
    """True if no *taller* object intrudes on a vertical grasp/placement column."""
    for other in world.values():
        if other.name == target.name or other.table is None:
            continue
        if other.height <= target.height + 1e-6:
            continue  # only taller objects obstruct a top-down approach
        gap = math.hypot(other.xy[0] - target.xy[0], other.xy[1] - target.xy[1])
        gap -= other.radius + target.radius
        if gap < GRASP_CLEARANCE:
            return False
    return True


def _footprint_free(
    xy: tuple[float, float], radius: float, world: dict[str, _ObjState], ignore: str
) -> bool:
    for other in world.values():
        if other.name == ignore or other.table is None:
            continue
        if (
            math.hypot(other.xy[0] - xy[0], other.xy[1] - xy[1])
            < other.radius + radius + 0.01
        ):
            return False
    return True


def _sample_placement(
    scene: GeometricScene,
    table: str,
    obj: _ObjState,
    world: dict[str, _ObjState],
    rng: random.Random,
) -> tuple[float, float] | None:
    """Sample a reachable, collision-free, top-down-clear pose on ``table``."""
    cx, cy = scene.table(table).center
    half = scene.table(table).half_extent - obj.radius
    for _ in range(PLACEMENT_SAMPLES):
        xy = (cx + rng.uniform(-half, half), cy + rng.uniform(-half, half))
        if not _reachable(scene, xy):
            continue
        if not _footprint_free(xy, obj.radius, world, ignore=obj.name):
            continue
        # tentatively move target there and check the descent column is clear
        saved = obj.xy
        obj.xy = xy
        ok = _top_down_clear(obj, world)
        if ok:
            return xy
        obj.xy = saved
    return None


# --------------------------------------------------------------------------- #
# refinement
# --------------------------------------------------------------------------- #
def _initial_world(scene: GeometricScene) -> dict[str, _ObjState]:
    world: dict[str, _ObjState] = {}
    # infer each object's current table from nearest table centre
    for o in scene.objects:
        table = min(
            scene.tables,
            key=lambda t: math.hypot(t.center[0] - o.pose[0], t.center[1] - o.pose[1]),
        ).name
        world[o.name] = _ObjState(
            name=o.name,
            xy=(o.pose[0], o.pose[1]),
            z=o.pose[2],
            height=o.size[2],
            radius=o.footprint_radius,
            table=table,
        )
    return world


def _try_once(
    skeleton: Skeleton, scene: GeometricScene, rng: random.Random
) -> tuple[int, list[BoundStep]]:
    """One refinement pass.

    Returns (#steps bound, bound steps) until first dead-end.
    """
    world = _initial_world(scene)
    held: str | None = None
    bound: list[BoundStep] = []

    for action in skeleton.actions:
        op = action.name
        if op in ("pick", "unstack"):
            target = world[action.args[0]]
            if (
                held is not None
                or not _reachable(scene, target.xy)
                or not _top_down_clear(target, world)
            ):
                break
            target.table = None
            target.on_block = None
            held = target.name
            bound.append(BoundStep(action, {"grasp_xy": target.xy}))
        elif op == "place":
            b, table = action.args
            target = world[b]
            if held != b:
                break
            xy = _sample_placement(scene, table, target, world, rng)
            if xy is None:
                break
            target.xy, target.table, held = xy, table, None
            target.z = target.height / 2.0  # now rests directly on the table
            bound.append(BoundStep(action, {"place_xy": xy, "place_z": target.z}))
        elif op == "stack":
            b, lower = action.args
            target, base = world[b], world[lower]
            if (
                held != b
                or not _reachable(scene, base.xy)
                or not _top_down_clear(base, world)
            ):
                break
            target.xy, target.z = base.xy, base.z + base.height
            target.table, target.on_block, held = None, lower, None
            bound.append(
                BoundStep(
                    action, {"on": lower, "place_xy": target.xy, "place_z": target.z}
                )
            )
        else:  # pragma: no cover - unknown operator
            break
    return len(bound), bound


def refine_skeleton(
    skeleton: Skeleton,
    scene: GeometricScene,
    max_attempts: int = 20,
    seed: int = 0,
) -> RefineResult:
    """Attempt to bind a skeleton's continuous params; backtrack-to-first on dead-
    end."""
    rng = random.Random(seed)
    best_steps = 0
    best_bound: list[BoundStep] = []
    for attempt in range(1, max_attempts + 1):
        n, bound = _try_once(skeleton, scene, rng)
        if n > best_steps:
            best_steps, best_bound = n, bound
        if n == len(skeleton):
            return RefineResult(
                status="feasible",
                steps_bound=n,
                plan_length=len(skeleton),
                n_attempts=attempt,
                failure_action=None,
                bound_plan=bound,
            )
    failure = None
    if best_steps < len(skeleton):
        failure = str(skeleton.actions[best_steps])
    return RefineResult(
        status="infeasible",
        steps_bound=best_steps,
        plan_length=len(skeleton),
        n_attempts=max_attempts,
        failure_action=failure,
        bound_plan=best_bound,
    )
