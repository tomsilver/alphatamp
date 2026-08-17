"""F1 clutter + relocation tests for the kinematic Restock3D.

Fast pure-logic tests (order-aware ``is_feasible_skeleton``, T5 penalty polarity) use
lightweight stand-in operators/atoms and need no simulator. Slow tests (marked) build
the real r1 clutter scene and exercise ``grasp_blockers``, the ``OnBuffer`` abstraction,
and the oracle's relocation skeleton through PyBullet + kinder (+ IKFast).
"""

from __future__ import annotations

import glob
import os
import pathlib
from types import SimpleNamespace

import pytest

from alphatamp.approaches.spectre.envs.restock3d.eager_tables import (
    EagerTables,
    EagerWeights,
    is_feasible_skeleton,
    make_penalty,
)


def _op(name: str, *argnames: str) -> SimpleNamespace:
    """A stand-in ground operator with the ``.name`` / ``.parameters[i].name`` shape the
    eager table logic reads."""
    return SimpleNamespace(
        name=name, parameters=[SimpleNamespace(name=a) for a in argnames]
    )


def _state(*atoms: tuple[str, str]) -> SimpleNamespace:
    """A stand-in abstract state: each atom is ``(predicate_name, object_name)``
    (unary)."""
    return SimpleNamespace(
        atoms=[
            SimpleNamespace(
                predicate=SimpleNamespace(name=p),
                objects=[SimpleNamespace(name=o)],
            )
            for p, o in atoms
        ]
    )


_TABLES = EagerTables(
    tall_regions=frozenset(),
    short_regions=frozenset({"region_1_1", "region_1_2"}),
    tall_goal=frozenset(),
    blockers={"cube_goal1": frozenset({"clutter1"})},
)


def test_is_feasible_direct_pick_is_infeasible() -> None:
    """A skeleton that picks a blocked goal without relocating its clutter first is
    F1-infeasible."""
    direct = [
        _op("pick", "robot", "cube_goal1"),
        _op("place", "robot", "cube_goal1", "region_1_1"),
    ]
    assert is_feasible_skeleton(direct, _TABLES) is False


def test_is_feasible_relocate_first_is_feasible() -> None:
    """Relocating the clutter to a buffer before the blocked goal's pick clears F1."""
    reloc = [
        _op("pick", "robot", "clutter1"),
        _op("place_buffer", "robot", "clutter1"),
        _op("pick", "robot", "cube_goal1"),
        _op("place", "robot", "cube_goal1", "region_1_1"),
    ]
    assert is_feasible_skeleton(reloc, _TABLES) is True


def test_is_feasible_relocate_after_is_infeasible() -> None:
    """Relocating the clutter AFTER the blocked pick does not help (order matters)."""
    late = [
        _op("pick", "robot", "cube_goal1"),  # clutter still on floor here -> F1
        _op("place", "robot", "cube_goal1", "region_1_1"),
        _op("pick", "robot", "clutter1"),
        _op("place_buffer", "robot", "clutter1"),
    ]
    assert is_feasible_skeleton(late, _TABLES) is False


def test_t5_penalty_polarity() -> None:
    """T5 penalises picking a blocked goal while its clutter is OnFloor; zero once
    relocated."""
    penalty = make_penalty(_TABLES, EagerWeights())
    pick_blocked = _op("pick", "robot", "cube_goal1")
    on_floor = _state(("OnFloor", "clutter1"))
    relocated = _state(("OnBuffer", "clutter1"))
    assert penalty(pick_blocked, on_floor) == EagerWeights().b
    assert penalty(pick_blocked, relocated) == 0.0
    # An unblocked goal is never penalised at pick.
    assert penalty(_op("pick", "robot", "cube_goal2"), on_floor) == 0.0


def test_no_clutter_tables_are_inert() -> None:
    """With empty blockers (r0/r2), the pick penalty is 0 and picks never gate
    feasibility."""
    tables = EagerTables(frozenset(), frozenset({"region_1_1"}), frozenset())
    penalty = make_penalty(tables, EagerWeights())
    assert penalty(_op("pick", "robot", "cube_goal1"), _state()) == 0.0
    assert is_feasible_skeleton(
        [
            _op("pick", "robot", "cube_goal1"),
            _op("place", "robot", "cube_goal1", "region_1_1"),
        ],
        tables,
    )


# --------------------------------------------------------------------------------------------------
# Slow integration tests: real r1 clutter scene through PyBullet + kinder.
#
# RETIRED (decisions/07 2026-08-16): under the fully-lateral layout + unified FRONT grasp, blockers
# were dropped (reach-over ordering is the difficulty) so CLUTTER_PER_STRATUM=0 -- r1 has no
# ``clutter1`` body, and the front grasp is not obstructed by a floor neighbour anyway. The buffer/
# relocation machinery is kept inert (one flag away), so these three tests are skipped rather than
# removed. The fast pure-logic eager tests above (is_feasible / T5 with hand-built blockers) still run.
# --------------------------------------------------------------------------------------------------
_RETIRED_CLUTTER = pytest.mark.skip(
    reason="F1 clutter/relocation retired under the unified front grasp; CLUTTER=0, machinery inert "
    "(decisions/07 2026-08-16)"
)


def _blas_shim() -> None:
    b = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
    os.environ.setdefault("LAPACK_DIR", b)
    os.environ.setdefault("BLAS_DIR", b)
    pathlib.Path(b).mkdir(parents=True, exist_ok=True)
    for a, (sd, pt) in {
        "liblapack.a": ("lapack", "liblapack.so.3*"),
        "libblas.a": ("blas", "libblas.so.3*"),
    }.items():
        lk = pathlib.Path(b) / a
        if not (lk.exists() or lk.is_symlink()):
            cands = sorted(
                glob.glob(f"/usr/lib/x86_64-linux-gnu/{sd}/{pt}")
                + glob.glob(f"/usr/lib/x86_64-linux-gnu/{pt}")
            )
            real = next((c for c in cands if os.path.isfile(c)), None)
            if real:
                lk.symlink_to(real)


def _r1_sim():
    pytest.importorskip("kinder")
    _blas_shim()
    from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
        ObjectCentricRestock3DEnv,
        stratum_env_args,
    )

    specs, pose_fn, region_infos, config = stratum_env_args(1)
    sim = ObjectCentricRestock3DEnv(
        specs, pose_fn, region_infos, config=config, allow_state_access=True
    )
    x0, _ = sim.reset(seed=0)
    return sim, x0, region_infos


@_RETIRED_CLUTTER
@pytest.mark.slow
def test_grasp_blockers_names_clutter_no_cycle() -> None:
    """The clutter obstructs cube_goal1's grasp (named), and is itself pickable (no
    cycle)."""
    from alphatamp.approaches.spectre.envs.restock3d.instrumented_refiner import (
        grasp_blockers,
    )

    sim, x0, _ = _r1_sim()
    gb_goal, reach_goal = grasp_blockers(sim, "cube_goal1", x0)
    assert reach_goal and "clutter1" in gb_goal
    gb_clut, reach_clut = grasp_blockers(sim, "clutter1", x0)
    assert reach_clut and "cube_goal1" not in gb_clut


@_RETIRED_CLUTTER
@pytest.mark.slow
def test_abstractor_emits_onbuffer_not_stored() -> None:
    """A cube relocated into the buffer zone abstracts OnBuffer (not OnFloor, not
    Stored)."""
    from pybullet_helpers.geometry import Pose, set_pose

    from alphatamp.approaches.spectre.envs.restock3d.generator import (
        build_spec,
        goal_object_names,
    )
    from alphatamp.approaches.spectre.envs.restock3d.models import RestockAbstractor
    from alphatamp.approaches.spectre.envs.restock3d.place_controller import (
        BUFFER_SPOTS,
    )

    sim, _x0, region_infos = _r1_sim()
    abst = RestockAbstractor(region_infos, goal_object_names(build_spec(0, 1)))
    cid = sim._object_name_to_pybullet_id("clutter1")
    hz = sim._get_half_extents("clutter1")[2]
    bx, by = BUFFER_SPOTS[0]
    set_pose(cid, Pose((bx, by, hz)), sim.physics_client_id)
    atoms = {str(a) for a in abst.state_abstractor(sim.get_state()).atoms}
    assert "(OnBuffer clutter1)" in atoms
    assert "(OnFloor clutter1)" not in atoms
    assert "(Stored clutter1)" not in atoms


@_RETIRED_CLUTTER
@pytest.mark.slow
def test_oracle_builds_feasible_relocation_skeleton() -> None:
    """build_tables computes the F1 blockers and the oracle skeleton relocates them
    (feasible)."""
    import kinder

    from alphatamp.approaches.spectre.collect import _make_env_models, _restock_extras
    from alphatamp.approaches.spectre.config import CollectionConfig
    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.restock3d.eager_tables import build_tables
    from alphatamp.approaches.spectre.envs.restock3d.oracle import (
        build_skeleton,
        solve_assignment,
    )

    pytest.importorskip("kinder")
    _blas_shim()
    register_extra_envs()
    cfg = CollectionConfig(
        env_id="spectre/Restock3D-r1-v0",
        env_variant="restock3d_v1",
        split="train",
        model_name="restock3d",
        model_kwargs={"stratum": 1},
        num_problems=1,
        problem_seed_start=0,
        problem_seed_end=1,
        K_max=50,
        plan_generator="closed_form",
    )
    env = kinder.make(cfg.env_id)
    try:
        obs, _ = env.reset(seed=0)
        em = _make_env_models(cfg, env.observation_space, env.action_space)
        x0 = em.observation_to_state(obs)
        s0 = em.state_abstractor(x0)
        region_infos = _restock_extras["region_infos"]
        goal_names = _restock_extras["goal_names"]
        tables = build_tables(
            region_infos, goal_names, sim=_restock_extras.get("sim"), state=x0
        )
        assert tables.blockers, "no F1 blockers computed on r1"
        lifted = {op.name: op for op in em.operators}
        assignment = solve_assignment(region_infos, goal_names)
        _, direct = build_skeleton(x0, s0, assignment, lifted, blockers=None)
        assert not is_feasible_skeleton(direct, tables)
        _, reloc = build_skeleton(x0, s0, assignment, lifted, blockers=tables.blockers)
        assert is_feasible_skeleton(reloc, tables)
        assert any(op.name == "place_buffer" for op in reloc)
    finally:
        env.close()
