"""Restock3D coverage polarity probe (P4) — the REACH-OVER mechanism.

Coverage/waste are fully env-agnostic (``unified_evidence``): they read only a candidate's operator
effects, the observed failure ``UnifiedRecord``s, and the goal/initial atoms. Under the fully-lateral
front-grasp env (decisions/07 2026-08-17) **F1 grasp-obstruction clutter is retired** (a floor
neighbour never collides the arm at the front-grasp config), so the coverage carrier is now the depth
**reach-over**: a goal south of another (in the lateral corridor, tall block involved) blocks the
farther goal's front-pick along the *approach path*. That approach-path collision is invisible to the
final-config ``grasp_blockers`` probe, so it is attributed geometrically by ``reach_over_culprits``
(the same corridor rule the eager ``reach_blockers`` table uses) — giving the reach-over pick failure
**class-1 culprits** (the south blockers), which are actionable (each is a goal that gets stored) and
feed coverage with the **correct** polarity (unlike F2, where coverage inverts).

This probe builds a reach-over scene (a tall block behind two same-column cubes), forms the real
reach-over record (culprits from ``reach_over_culprits``), and checks:
  * **culprit pool** ``K`` is non-empty (the blockers are actionable).
  * **RP-3 (coverage)** a **south-to-north** candidate that stores the blockers before re-picking the
    tall block *covers* them (coverage 1.0); a **talls-first** candidate does not (coverage 0.0).
  * **Waste stays degenerate** — the reach-over fix is *reordering goal-necessary picks*, not a
    discretionary relocation, so the superfluous set is empty and waste is 0 on every candidate. To
    revive waste too, a non-goal approach-corridor clutter (relocatable to the buffer) would be
    needed (the set-aside option; see the proposal As-built §"Coverage/waste").

    python experiments/spectre/restock3d_coverage_probe.py
"""

from __future__ import annotations

import os

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)

from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)

from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    reach_over_culprits,
)
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.models import (
    CubeType,
    HandEmpty,
    Holding,
    InRegion,
    OnFloor,
    RobotType,
    Stored,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller import RegionType
from alphatamp.approaches.spectre.envs.restock3d.region_geometry import (
    compute_region_infos,
)
from alphatamp.approaches.spectre.unified_evidence import (
    UnifiedRecord,
    coverage_and_waste,
    culprit_pool,
    universal_objects,
)

# A reach-over scene (the dense grid-r3 pattern): a tall block behind two same-column cubes, plus a
# second column that does NOT block it (dx=0.30). Deterministic so the reach-over is guaranteed.
_POSES = {
    "cube_goal1": (-0.65, 0.60),
    "cube_goal2": (-0.35, 0.60),
    "cube_goal3": (-0.65, 0.90),
    "cube_goal4": (-0.35, 0.90),
    "block_goal1": (-0.65, 1.20),
    "block_goal2": (-0.35, 1.20),
}


def _operators():
    r = Variable("?r", RobotType)
    t = Variable("?t", CubeType)
    g = Variable("?g", RegionType)
    pick = LiftedOperator(
        "pick",
        [r, t],
        {LiftedAtom(HandEmpty, [r]), LiftedAtom(OnFloor, [t])},
        {LiftedAtom(Holding, [r, t])},
        {LiftedAtom(HandEmpty, [r]), LiftedAtom(OnFloor, [t])},
    )
    place = LiftedOperator(
        "place",
        [r, t, g],
        {LiftedAtom(Holding, [r, t])},
        {
            LiftedAtom(HandEmpty, [r]),
            LiftedAtom(InRegion, [t, g]),
            LiftedAtom(Stored, [t]),
        },
        {LiftedAtom(Holding, [r, t])},
    )
    return pick, place


def main() -> None:
    cfg = Restock3DEnvConfig()
    specs = [
        (
            n,
            cfg.tall_half if n.startswith("block") else cfg.small_half,
            (0.5, 0.5, 0.5, 1),
        )
        for n in _POSES
    ]
    sim = ObjectCentricRestock3DEnv(
        specs,
        lambda s: _POSES,
        compute_region_infos(cfg, 3),
        config=cfg,
        allow_state_access=True,
    )
    x0, _ = sim.reset(seed=0)

    # Real reach-over attribution: which floor goals block block_goal1's front-pick reach.
    culprits = reach_over_culprits(sim, "block_goal1", x0)
    print(f"reach_over_culprits(block_goal1) = {culprits}", flush=True)
    assert culprits, "no reach-over culprits -> coverage would be inert"

    pick, place = _operators()
    rob = Object("robot", RobotType)
    objs = {n: Object(n, CubeType) for n in _POSES}
    reg = {
        "region_1_1": Object("region_1_1", RegionType),
        "region_1_2": Object("region_1_2", RegionType),
        "region_0_1": Object("region_0_1", RegionType),
        "region_0_2": Object("region_0_2", RegionType),
    }

    def pk(o):
        return pick.ground((rob, objs[o]))

    def pl(o, rg):
        return place.ground((rob, objs[o], reg[rg]))

    # south-to-north: the two blockers are stored before block_goal1 is picked (covers them).
    s2n = [
        pk("cube_goal1"),
        pl("cube_goal1", "region_1_1"),
        pk("cube_goal3"),
        pl("cube_goal3", "region_1_2"),
        pk("block_goal1"),
        pl("block_goal1", "region_0_1"),
    ]
    # talls-first: block_goal1 picked before its blockers are cleared (reach-over doomed; uncovered).
    talls = [
        pk("block_goal1"),
        pl("block_goal1", "region_0_1"),
        pk("cube_goal1"),
        pl("cube_goal1", "region_1_1"),
        pk("cube_goal3"),
        pl("cube_goal3", "region_1_2"),
    ]

    # The observed reach-over failure: the talls-first candidate's Pick(block_goal1) is rejected,
    # blaming the un-cleared south blockers (class-1; deviation=None).
    record = UnifiedRecord(
        failed_step=pk("block_goal1"), deviation=None, check_blame=tuple(culprits)
    )
    records = [record]
    universal = universal_objects(s2n)
    pool = culprit_pool(records, s2n)
    init = frozenset(
        {GroundAtom(HandEmpty, [rob])}
        | {GroundAtom(OnFloor, [objs[n]]) for n in _POSES}
    )
    goal = frozenset(GroundAtom(Stored, [objs[n]]) for n in _POSES)
    print(f"culprit pool K = {sorted(pool)}  (blockers must be actionable)", flush=True)
    assert set(culprits) <= pool, "reach-over culprit not actionable -> coverage inert"

    cov_s2n, waste_s2n = coverage_and_waste(s2n, records, pool, init, goal, universal)
    cov_talls, waste_talls = coverage_and_waste(
        talls, records, pool, init, goal, universal
    )

    print(
        "\n==== COVERAGE/WASTE (reach-over record blames %s) ====" % list(culprits),
        flush=True,
    )
    print(
        f"  south-to-north : coverage={cov_s2n:.2f} waste={waste_s2n:.2f}  (clears blockers first)",
        flush=True,
    )
    print(
        f"  talls-first    : coverage={cov_talls:.2f} waste={waste_talls:.2f}  (reach-over doomed)",
        flush=True,
    )

    rp3 = cov_s2n > cov_talls
    nondegen = bool(pool) and (cov_s2n > 0 or cov_talls > 0)
    print(
        f"\n  culprit-pool non-empty: {bool(pool)}\n"
        f"  RP-3 coverage(south-to-north) > coverage(talls-first): {rp3}\n"
        f"  coverage non-degenerate + correct polarity on reach-over: {nondegen and rp3}\n"
        f"  waste degenerate (reorder of goal-necessary picks, no discretionary step): "
        f"{waste_s2n == 0.0 and waste_talls == 0.0}",
        flush=True,
    )
    print(
        f"\n==== COVERAGE PROBE: {'PASS' if (nondegen and rp3) else 'FAIL'} ====",
        flush=True,
    )


if __name__ == "__main__":
    main()
