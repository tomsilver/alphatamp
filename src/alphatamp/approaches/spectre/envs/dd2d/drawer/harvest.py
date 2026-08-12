"""Post-mortem typed-fact harvest for DD2D (proposal §6.2 / §6.4).

Given a *failed* refinement's ``RefineResult`` (which already carries the deepest bound
prefix as ``bound_plan``) and the scene, reconstruct the **harvest state** — the world
after executing that prefix — and read typed facts off exact geometric checks at that
state. Facts are tiered via the domain's soundness registry (``soundness.py``); DD2D's
registry makes the deducible ones proofs.

Facts produced (DD2D):

| type | tier (DD2D) | meaning |
|---|---|---|
| blocked-at-contents | proof | target has no clear grasp with the harvest drawer contents C (removal-monotone ⇒ blocked at every C′⊇C) |
| grasp-witness | hint | drawer items blocking every open target corridor at the harvest state |
| extracted-ok | proof | a prefix ``pick`` proved that item extractable under its contents |
| packed-ok | proof | the prefix's placed set packs (witness placements on the buffer) |
| pack-impossible | proof | the §8.4 certificate proves the full staged subset cannot pack |
| pack-exhausted | hint | staging (a ``place-buffer``) exhausted its budget for the subset |

The harvest state is stored as a **replayable** prefix + a state hash; a unit test replays
the prefix into a fresh world and asserts the hash matches (proposal §6.2).
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from alphatamp.approaches.spectre.schema import Fact, PostMortemRecord

from ..soundness import DD2D_REGISTRY, SoundnessRegistry
from .certificate import certify_infeasible_by_packing
from .enumerate import _blocker_sets, _footprints
from .grasps import has_grasp
from .world import DrawerScene, DrawerWorld

_POSE_ROUND = 4  # decimals for the replayable prefix + state hash


def _pose_str(pose) -> str:
    return "|".join(f"{float(v):.{_POSE_ROUND}f}" for v in pose)


def prefix_reprs(bound_plan) -> tuple[str, ...]:
    """Serialize the bound prefix to replayable ``phase:item[:pose]`` strings."""
    reprs = []
    for step in bound_plan:
        phase = step.params["phase"]
        item = step.params["item"]
        if phase == "place":
            reprs.append(f"place|{item}|{_pose_str(step.params['pose'])}")
        else:  # pick / retrieve carry no bound pose
            reprs.append(f"{phase}|{item}")
    return tuple(reprs)


def replay_prefix(scene: DrawerScene, reprs) -> DrawerWorld:
    """Rebuild the harvest world by replaying serialized prefix steps into a fresh world."""
    world = DrawerWorld(scene)
    for r in reprs:
        parts = r.split("|")
        phase, item = parts[0], parts[1]
        if phase == "pick":
            world.pick(item)
        elif phase == "place":
            pose = (float(parts[2]), float(parts[3]), float(parts[4]))
            world.place_buffer(item, pose)
        elif phase == "retrieve":
            world.extract(item)
    return world


def harvest_state_hash(world: DrawerWorld) -> str:
    """Canonical hash of the harvest occupancy (name, region, rounded pose), sorted."""
    items = sorted(
        (n, s.region, tuple(round(float(v), _POSE_ROUND) for v in s.pose))
        for n, s in world.states.items()
    )
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


def _harvest_scene(scene: DrawerScene, world: DrawerWorld) -> DrawerScene:
    """A scene view over only the DRAWER items (+ target) at the harvest state — buffer /
    removed items must not count as drawer-side corridor blockers (their poses are stale
    after a move)."""
    keep = {
        n: s
        for n, s in world.states.items()
        if s.region == "drawer" or n == scene.target
    }
    return DrawerScene(
        drawer=scene.drawer,
        wall_band=scene.wall_band,
        buffer=scene.buffer,
        items=keep,
        target=scene.target,
        margin=scene.margin,
        dims=scene.dims,
    )


def _grasp_witness(scene: DrawerScene, world: DrawerWorld) -> frozenset[str]:
    """Drawer items blocking *every* still-open target corridor at the harvest state
    (their removal is necessary to open any of those corridors)."""
    hscene = _harvest_scene(scene, world)
    sets = [s for s in _blocker_sets(hscene, _footprints(hscene)) if s]
    if not sets:
        return frozenset()
    witness = set(sets[0])
    for s in sets[1:]:
        witness &= s
    return frozenset(witness)


def _staged_subset(bound_plan) -> tuple[list[str], list[str]]:
    """(items picked, items placed) in the bound prefix, in order."""
    picked = [s.params["item"] for s in bound_plan if s.params["phase"] == "pick"]
    placed = [s.params["item"] for s in bound_plan if s.params["phase"] == "place"]
    return picked, placed


def harvest_facts(
    scene: DrawerScene,
    refine_result,
    skeleton_subset: frozenset[str],
    skeleton_idx: int,
    refinement_seed: int,
    registry: SoundnessRegistry = DD2D_REGISTRY,
    run_certificate: bool = True,
) -> PostMortemRecord:
    """Harvest a ``PostMortemRecord`` from one failed refinement (proposal §6.2/§6.4)."""
    t0 = time.perf_counter()
    bound_plan = list(refine_result.bound_plan)
    reprs = prefix_reprs(bound_plan)
    world = replay_prefix(scene, reprs)
    state_hash = harvest_state_hash(world)

    facts: list[Fact] = []

    def add(fact_type: str, args, scalars=()) -> None:
        facts.append(
            Fact(
                fact_type=fact_type,
                args=tuple(sorted(args)),
                tier=registry.tier(fact_type),
                scalars=tuple(scalars),
            )
        )

    # blocked-at-contents (proof): target has no clear grasp with the harvest contents.
    tstate = world.states[scene.target]
    if (
        has_grasp(
            tstate.shape, tstate.pose, world.drawer_obstacles(ignore=scene.target)
        )
        is None
    ):
        drawer_blockers = [n for n in world.region_items("drawer") if n != scene.target]
        add("blocked-at-contents", drawer_blockers)
        witness = _grasp_witness(scene, world)
        if witness:
            add("grasp-witness", witness, scalars=(("n_witness", float(len(witness))),))

    # constructive proofs from the successful prefix.
    picked, placed = _staged_subset(bound_plan)
    for item in picked:
        add("extracted-ok", (item,))
    if placed:
        add("packed-ok", placed)

    # pack-impossible (proof): the certificate proves the full subset cannot pack.
    if run_certificate and skeleton_subset:
        verdict = certify_infeasible_by_packing(scene, skeleton_subset)
        if verdict is True:
            add("pack-impossible", skeleton_subset)

    # pack-exhausted (hint): staging a place-buffer exhausted its budget.
    fa = refine_result.failure_action or ""
    if "place-buffer" in fa and skeleton_subset:
        add(
            "pack-exhausted",
            skeleton_subset,
            scalars=(("n_attempts", float(refine_result.n_attempts)),),
        )

    failed_idx: Optional[int] = (
        refine_result.steps_bound if refine_result.steps_bound is not None else None
    )
    return PostMortemRecord(
        skeleton_idx=skeleton_idx,
        refinement_seed=refinement_seed,
        failed_step_index=failed_idx,
        failed_schema=(fa.split("(")[0] if fa else None),
        failed_args=(),
        harvest_prefix=reprs,
        harvest_state_hash=state_hash,
        facts=tuple(facts),
        harvest_cost_s=round(time.perf_counter() - t0, 6),
    )
