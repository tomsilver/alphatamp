"""Tests for the **Restock3D v2** geometry-informed A* pick-cost plan generator.

The generator (:mod:`envs.restock3d.plan_generator_v2`) replaces the stock hff generator's unit
operator cost with a nearest-first pick cost ``c(pick(o), s) = 1 + lam * (# nearer OnFloor)``, so the
0-inversion (oracle south-to-north) pick order is the strictly-cheapest band and is yielded first.

* the ``_edge_cost`` test is pure Python (no sim) — it builds a hand-made abstract state;
* the enumeration test builds the v2 sim (like ``test_restock3d_v2_oracle``) and is marked slow.
"""

from __future__ import annotations

import itertools

import pytest

from alphatamp.approaches.spectre.envs.restock3d import models_v2 as M
from alphatamp.approaches.spectre.envs.restock3d.plan_generator_v2 import (
    GeometryGuidedRestockPlanGenerator,
    pick_distance_from_state,
)


def _pick_order(action_plan) -> list[str]:
    return [op.parameters[1].name for op in action_plan if op.name == "pick"]


def test_edge_cost_counts_nearer_onfloor() -> None:
    """Pick(o) pays 1 + lam*(nearer OnFloor); other ops pay 1; picked-away objects don't
    count."""
    from bilevel_planning.structs import RelationalAbstractState
    from relational_structs import (
        GroundAtom,
        LiftedAtom,
        LiftedOperator,
        Object,
        Variable,
    )

    robot = Object("robot", M.RobotType)
    near = Object("cube_goal1", M.CubeType)  # d = 0.6 (closest)
    mid = Object("cube_goal2", M.CubeType)  # d = 0.9
    far = Object("block_goal1", M.CubeType)  # d = 1.2 (farthest)
    pick_distance = {"cube_goal1": 0.6, "cube_goal2": 0.9, "block_goal1": 1.2}

    r = Variable("?robot", M.RobotType)
    t = Variable("?target", M.CubeType)
    pick = LiftedOperator(
        "pick",
        [r, t],
        preconditions={LiftedAtom(M.HandEmpty, [r]), LiftedAtom(M.OnFloor, [t])},
        add_effects={LiftedAtom(M.Holding, [r, t])},
        delete_effects=set(),
    )
    place = LiftedOperator(
        "place_tall",
        [r, t],
        preconditions={LiftedAtom(M.Holding, [r, t])},
        add_effects={LiftedAtom(M.Stored, [t])},
        delete_effects=set(),
    )

    # Bypass domain construction -- _edge_cost only reads _pick_distance / _lam and the (state, op).
    gen = GeometryGuidedRestockPlanGenerator.__new__(GeometryGuidedRestockPlanGenerator)
    gen._pick_distance = pick_distance
    gen._lam = 1.0

    # All three still on the floor.
    all_floor = RelationalAbstractState(
        {
            GroundAtom(M.OnFloor, [near]),
            GroundAtom(M.OnFloor, [mid]),
            GroundAtom(M.OnFloor, [far]),
            GroundAtom(M.HandEmpty, [robot]),
        },
        {robot, near, mid, far},
    )
    assert gen._edge_cost(all_floor, pick.ground((robot, near))) == 1.0  # 0 nearer
    assert gen._edge_cost(all_floor, pick.ground((robot, mid))) == 2.0  # 1 nearer
    assert gen._edge_cost(all_floor, pick.ground((robot, far))) == 3.0  # 2 nearer
    # A place op is always unit cost.
    assert gen._edge_cost(all_floor, place.ground((robot, far))) == 1.0

    # Once `near` is off the floor (picked), it no longer penalizes picking `far`.
    near_gone = RelationalAbstractState(
        {GroundAtom(M.OnFloor, [mid]), GroundAtom(M.OnFloor, [far])},
        {robot, near, mid, far},
    )
    assert (
        gen._edge_cost(near_gone, pick.ground((robot, far))) == 2.0
    )  # only `mid` nearer

    # lam scales the penalty.
    gen._lam = 0.5
    assert gen._edge_cost(all_floor, pick.ground((robot, far))) == 2.0  # 1 + 0.5*2


@pytest.mark.slow
def test_geometry_generator_yields_oracle_pick_order_first() -> None:
    """On an r2 problem the geometry generator's FIRST plan has the oracle south-to-
    north pick order; a stock hff generator yields valid goal plans too."""
    pytest.importorskip("kinder")
    import kinder
    from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
        RelationalHeuristicSearchAbstractPlanGenerator,
    )
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.restock3d.models_v2 import (
        create_restock3d_v2_models,
    )

    register_extra_envs()
    env = kinder.make("spectre/Restock3D-r2-v0")
    try:
        obs, _ = env.reset(seed=0)
        models = create_restock3d_v2_models(
            env.observation_space, env.action_space, stratum=2
        ).models
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        goal_names = [
            o.name for o in x0 if o.name.startswith(("cube_goal", "block_goal"))
        ]
        # Oracle south-to-north order == sort goal objects by y (oracle_v2.build_skeleton_v2 key).
        oracle_order = sorted(
            goal_names, key=lambda n: x0.get_object_pose(n).position[1]
        )

        def _draw_first(gen) -> list:
            bpg: BilevelPlanningGraph = BilevelPlanningGraph()
            bpg.add_abstract_state_node(s0)
            bpg.add_state_node(x0)
            bpg.add_state_abstractor_edge(x0, s0)
            first = next(iter(gen(x0, s0, goal, 60.0, bpg)))
            return first[1]  # action plan

        geo = GeometryGuidedRestockPlanGenerator(
            models.types,
            models.predicates,
            models.operators,
            seed=0,
            pick_distance=pick_distance_from_state(x0, goal_names),
            lam=1.0,
        )
        geo_first = _draw_first(geo)
        assert _pick_order(geo_first) == oracle_order

        hff: RelationalHeuristicSearchAbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                models.types,
                models.predicates,
                models.operators,
                heuristic_name="hff",
                seed=0,
            )
        )
        hff_first = _draw_first(hff)
        # hff yields a valid, full-length goal plan (one pick + one place per goal object).
        assert len(_pick_order(hff_first)) == len(goal_names)
    finally:
        env.close()
