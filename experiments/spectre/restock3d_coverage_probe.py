"""Restock3D Gate-6 coverage/waste polarity probe (P4).

Coverage/waste are fully env-agnostic (``unified_evidence``): they read only a candidate's operator
effects, the observed failure ``UnifiedRecord``s, and the goal/initial atoms. F1 clutter is what makes
them non-degenerate on Restock3D -- an F1 failure names the blocking clutter as a class-1 culprit
(``grasp_blockers``), the clutter is *actionable* (``PlaceBuffer`` adds ``OnBuffer(clutter)``), so it
enters the culprit pool ``K``, and a candidate that relocates it *covers* the culprit.

This probe builds a cluttered r1 scene, forms the real F1 record (from ``grasp_blockers`` -- the same
probe the refiner uses), and prints:
  * **RP culprit pool** -- ``K`` is non-empty (the clutter is actionable), so coverage/waste can act.
  * **RP-3 (coverage, F1)** -- a relocate-first candidate covers the clutter culprit (high coverage);
    a direct candidate that ignores it does not (0 coverage).
  * **RP-4 (waste)** -- relocating the *blamed* clutter is justified (0 waste); relocating a
    *different* clutter that the record does not blame is an unjustified superfluous step (waste > 0).

    python experiments/spectre/restock3d_coverage_probe.py

Wiring note: ``coverage_feats`` is already plumbed through ``TrainConfig``/``dataset``/``model`` (adds
the ``[coverage, waste]`` overlap columns). The training run itself is deferred.
"""

from __future__ import annotations

import os

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)

import kinder
from relational_structs import Object

from alphatamp.approaches.spectre.collect import _make_env_models, _restock_extras
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
    grasp_blockers,
)
from alphatamp.approaches.spectre.envs.restock3d.oracle import (
    build_skeleton,
    solve_assignment,
)
from alphatamp.approaches.spectre.unified_evidence import (
    UnifiedRecord,
    coverage_and_waste,
    culprit_pool,
    universal_objects,
)


def _cfg(stratum: int) -> CollectionConfig:
    import alphatamp.approaches.spectre.envs.restock3d.strata as S

    start = S.problem_id("train", stratum, 0)
    return CollectionConfig(
        env_id=f"spectre/Restock3D-r{stratum}-v0",
        env_variant="restock3d_v1",
        split="train",
        model_name="restock3d",
        model_kwargs={"stratum": stratum},
        num_problems=1,
        problem_seed_start=start,
        problem_seed_end=start + 1,
        K_max=50,
        plan_generator="closed_form",
    )


def _set_clutter(stratum: int, k: int) -> None:
    import alphatamp.approaches.spectre.envs.restock3d.generator as gen
    import alphatamp.approaches.spectre.envs.restock3d.kinematic_env as ke

    gen._CLUTTER_PER_STRATUM[stratum] = k  # pylint: disable=protected-access
    ke.CLUTTER_PER_STRATUM[stratum] = k


def _pick_of(ops, obj_name):
    return next(o for o in ops if o.name == "pick" and o.parameters[1].name == obj_name)


def main() -> None:
    register_extra_envs()
    stratum = 1
    _set_clutter(
        stratum, 2
    )  # 2 clutters (one blames cube_goal1, one blames cube_goal2)
    cfg = _cfg(stratum)
    start = cfg.problem_seed_start
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=start)
        em = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = em.observation_to_state(obs)
        s0 = em.state_abstractor(x0)
        goal = em.goal_deriver(x0)
        region_infos = _restock_extras["region_infos"]
        goal_names = _restock_extras["goal_names"]
        lifted = {op.name: op for op in em.operators}
        assignment = solve_assignment(region_infos, goal_names)

        # Real F1 culprits (same probe the refiner uses).
        blk1, _ = grasp_blockers(em_sim := _restock_extras["sim"], "cube_goal1", x0)
        blk2, _ = grasp_blockers(em_sim, "cube_goal2", x0)
        print(f"grasp_blockers: cube_goal1 <- {blk1}, cube_goal2 <- {blk2}", flush=True)
        assert blk1 and blk2 and blk1 != blk2, "need two distinct F1 culprits"

        # The full relocate-first oracle skeleton (relocates both clutters, then stores all cubes).
        _, reloc_ops = build_skeleton(
            x0,
            s0,
            assignment,
            lifted,
            blockers={"cube_goal1": frozenset(blk1), "cube_goal2": frozenset(blk2)},
        )
        # A direct skeleton (no relocation) -- fails F1 on the blocked cubes.
        _, direct_ops = build_skeleton(x0, s0, assignment, lifted, blockers=None)

        # The observed F1 failure: the DIRECT candidate's Pick(cube_goal1) is rejected, clutter1 named.
        f1_record = UnifiedRecord(
            failed_step=_pick_of(direct_ops, "cube_goal1"),
            deviation=None,  # class-1: a validity check rejected the sample
            check_blame=tuple(blk1),
        )
        records = [f1_record]

        universal = universal_objects(reloc_ops)
        pool = culprit_pool(records, reloc_ops)
        init_atoms, goal_atoms = s0.atoms, goal.atoms
        print(f"culprit pool K = {sorted(pool)}  (must contain {blk1[0]})", flush=True)
        assert blk1[0] in pool, "F1 culprit not actionable -> coverage would be inert"

        # RP-3: the relocate-first candidate covers the clutter1 culprit; the direct one does not.
        cov_reloc, waste_reloc = coverage_and_waste(
            reloc_ops, records, pool, init_atoms, goal_atoms, universal
        )
        cov_direct, waste_direct = coverage_and_waste(
            direct_ops, records, pool, init_atoms, goal_atoms, universal
        )

        # RP-4: a candidate that relocates only the OTHER clutter (clutter2, not blamed by this
        # record) leaves the culprit uncovered and its relocation unjustified -> waste.
        _, only2_ops = build_skeleton(
            x0, s0, assignment, lifted, blockers={"cube_goal2": frozenset(blk2)}
        )
        cov_only2, waste_only2 = coverage_and_waste(
            only2_ops, records, pool, init_atoms, goal_atoms, universal
        )

        print(
            "\n==== GATE-6 COVERAGE/WASTE (F1 record blames %s) ====" % blk1[0],
            flush=True,
        )
        print(
            f"  relocate-both   : coverage={cov_reloc:.2f} waste={waste_reloc:.2f}  (covers culprit)",
            flush=True,
        )
        print(
            f"  direct (no reloc): coverage={cov_direct:.2f} waste={waste_direct:.2f}  (ignores culprit)",
            flush=True,
        )
        print(
            f"  relocate-other  : coverage={cov_only2:.2f} waste={waste_only2:.2f}  (unjustified reloc)",
            flush=True,
        )

        rp3 = cov_reloc > cov_direct
        rp4 = waste_only2 > waste_reloc
        nondegen = bool(pool) and (cov_reloc > 0 or cov_direct > 0 or waste_only2 > 0)
        print(
            f"\n  RP culprit-pool non-empty: {bool(pool)}\n"
            f"  RP-3 coverage(relocate) > coverage(direct): {rp3}\n"
            f"  RP-4 waste(relocate-other) > waste(relocate-culprit): {rp4}\n"
            f"  coverage/waste non-degenerate on restock3d F1+relocation: {nondegen}",
            flush=True,
        )
        print(
            f"\n==== GATE-6: {'PASS' if (nondegen and rp3) else 'FAIL'} ====",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
