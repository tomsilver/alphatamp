"""Tests for DD2D (Drawer Decluttering in 2D).

See docs/dd2d_spec.md and docs/dd2d.md.
Small, deterministic instances; the DD2D candidate enumerator + backjumping refiner are
exercised end to end. The generic top-k planner is not used here (it produces only
``retrieve(target)`` / goal-moving plans -- see blocks_tamp/dd2d/planning.py).
"""

from __future__ import annotations

import math
import random

import pytest
from shapely import box as shp_box
from shapely.affinity import rotate
from shapely.geometry import Point

from alphatamp.approaches.spectre.envs.dd2d.dd2d import enumerate as EN
from alphatamp.approaches.spectre.envs.dd2d.dd2d import label as L
from alphatamp.approaches.spectre.envs.dd2d.dd2d import scene as SC
from alphatamp.approaches.spectre.envs.dd2d.dd2d.enumerate import Candidate
from alphatamp.approaches.spectre.envs.dd2d.dd2d.grasps import (
    FINGER_WIDTH,
    direction_admissible,
    finger_rects,
    grasp_cells,
    has_grasp,
    isolation_graspable,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.label import min_feasible_subset_size
from alphatamp.approaches.spectre.envs.dd2d.dd2d.planning import (
    DD2DPlanner,
    make_dd2d_planner,
    staging_skeleton,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import generate_dd2d_problem
from alphatamp.approaches.spectre.envs.dd2d.dd2d.refine import DD2DRefiner
from alphatamp.approaches.spectre.envs.dd2d.dd2d.render import (
    render_episode,
    render_scene,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.shapes import (
    FAMILIES,
    Shape,
    sample_shape,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.world import (
    DrawerWorld,
    collar_pose,
    collision_free,
    contained,
    place_polygon,
    sample_buffer_pose,
)
from alphatamp.approaches.spectre.envs.dd2d.planning import (
    ForbidLoopPlanner,
    make_planner,
)
from alphatamp.approaches.spectre.envs.dd2d.record import (
    PIGINetExample,
    build_example,
    build_image_refs,
)
from alphatamp.approaches.spectre.envs.dd2d.skeleton import Action, Skeleton


def _max_place_buffer(skeletons):
    return max(
        (sum(a.name == "place-buffer" for a in s.actions) for s in skeletons), default=0
    )


GEN_KW = dict(lam=0.6, seed=0, margin=1.0, budget=200)


@pytest.fixture(scope="module")
def problem():
    return generate_dd2d_problem(**GEN_KW)


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
def test_every_family_polygonises_and_is_graspable():
    for fam in FAMILIES:
        s = sample_shape(random.Random(hash(fam) & 0xFFFF), family=fam)
        assert s.polygon.is_valid and not s.polygon.is_empty and s.area > 0
        assert isolation_graspable(s)
        assert s.concave == (fam in {"dumbbell", "shoe", "horseshoe"})


def test_shape_sampling_deterministic():
    a = sample_shape(random.Random(5), family="box")
    b = sample_shape(random.Random(5), family="box")
    assert list(a.polygon.exterior.coords) == list(b.polygon.exterior.coords)


# --------------------------------------------------------------------------- #
# grasps
# --------------------------------------------------------------------------- #
def test_horseshoe_has_inadmissible_direction():
    hs = sample_shape(random.Random(3), family="horseshoe")
    adm = [direction_admissible(hs, math.pi * i / 18)[0] for i in range(18)]
    assert not all(
        adm
    )  # the C-opening gives directions with disjoint L/R contacts (spec Section 5.3)


def test_every_grasp_cell_makes_contact():
    """Core invariant of the contact-run grasp model: every emitted cell has BOTH fingers
    on material (gap 0), for every family -- no cell closes onto a concavity gap."""
    tol = 1e-6
    for fam in FAMILIES:
        for seed in range(6):
            shape = sample_shape(random.Random(seed), family=fam)
            fp = place_polygon(shape.polygon, (0.0, 0.0, 0.0))
            for g in grasp_cells(shape):
                left, right = finger_rects(g, (0.0, 0.0, 0.0))
                assert left.distance(fp) <= tol and right.distance(fp) <= tol, (
                    fam,
                    seed,
                    g.alpha,
                )


def test_horseshoe_grasp_is_full_face():
    """The blocky horseshoe admits a grasp whose fingers make >= FINGER_WIDTH of flat
    contact -- full-face, not a curve's tangent point (the reason it replaced the
    banana)."""
    for seed in range(6):
        shape = sample_shape(random.Random(seed), family="horseshoe")
        fp = place_polygon(shape.polygon, (0.0, 0.0, 0.0))
        best = 0.0
        for g in grasp_cells(shape):
            touch = []
            for finger in finger_rects(g, (0.0, 0.0, 0.0)):
                touch.append(finger.intersection(fp.buffer(1e-6)).length)
            best = max(best, min(touch))
        assert best >= FINGER_WIDTH - 0.2, (seed, best)


def _is_internal_grasp(shape, g, tol=1e-3):
    """A grasp that pinches an internal feature: ``[xmin, xmax]`` strictly inside the
    footprint's rotated x-extent (a finger reaches into a concavity)."""
    b = rotate(shape.polygon, -g.alpha, origin=(0, 0), use_radians=True).bounds
    return g.xmin > b[0] + tol or g.xmax < b[2] - tol


def test_internal_grasp_on_dumbbell_waist():
    """The gripper can hold the dumbbell's middle bar -- a narrow internal feature, far
    from the block-to-block envelope width."""
    for seed in range(6):
        shape = sample_shape(random.Random(seed), family="dumbbell")
        internal = [g for g in grasp_cells(shape) if _is_internal_grasp(shape, g)]
        # the bar is much narrower than the envelope; require a clearly-internal narrow grip
        assert any(g.width < 4.0 for g in internal), (seed, [g.width for g in internal])


def test_internal_grasp_on_horseshoe_spine():
    """The gripper can reach a finger into the C-opening to grip a prong/spine, with
    full-face flat contact."""
    for seed in range(6):
        shape = sample_shape(random.Random(seed), family="horseshoe")
        fp = place_polygon(shape.polygon, (0.0, 0.0, 0.0))
        ok = False
        for g in grasp_cells(shape):
            if not _is_internal_grasp(shape, g):
                continue
            faces = [
                finger.intersection(fp.boundary).length
                for finger in finger_rects(g, (0.0, 0.0, 0.0))
            ]
            if min(faces) >= 0.9 * FINGER_WIDTH:
                ok = True
                break
        assert ok, seed


def test_fingers_fit_in_isolation():
    """Every grasp cell's fingers clear the item's *own* material (in isolation) -- an
    internal grasp is only emitted where the grippers physically fit in the concavity.
    """
    for fam in FAMILIES:
        for seed in range(6):
            shape = sample_shape(random.Random(seed), family=fam)
            fp = place_polygon(shape.polygon, (0.0, 0.0, 0.0))
            for g in grasp_cells(shape):
                for finger in finger_rects(g, (0.0, 0.0, 0.0)):
                    assert finger.intersection(fp).area < 1e-3, (fam, seed, g.alpha)


def test_convex_families_have_no_internal_grasp():
    """Convex footprints have no flat internal sub-feature, and the full-face rule
    excludes curved slivers -- so they only ever grip the outer envelope."""
    for fam in ["can", "bowl", "box", "pillcase"]:
        for seed in range(6):
            shape = sample_shape(random.Random(seed), family=fam)
            assert not any(_is_internal_grasp(shape, g) for g in grasp_cells(shape)), (
                fam,
                seed,
            )


def test_oversized_shape_has_no_grasp_cells():
    huge = Shape("can", Point(0, 0).buffer(10.0), False)  # diameter 20 > 12 cm aperture
    assert grasp_cells(huge) == []
    assert not isolation_graspable(huge)


def test_hemmed_item_is_ungraspable():
    box_shape = sample_shape(random.Random(1), family="box")
    ring = shp_box(-20, -20, 20, 20).difference(box_shape.polygon.buffer(0.4))
    assert has_grasp(box_shape, (0, 0, 0.0), [ring]) is None
    assert has_grasp(box_shape, (0, 0, 0.0), []) is not None


# --------------------------------------------------------------------------- #
# world + buffer sampler
# --------------------------------------------------------------------------- #
def test_buffer_sampler_packs_contained_and_free():
    buf = shp_box(0, 0, 20, 12)
    can = sample_shape(random.Random(2), family="can")
    staged = []
    for i in range(8):
        p = sample_buffer_pose(can, buf, staged, random.Random(100 + i))
        if p is None:
            break
        fp = place_polygon(can.polygon, p)
        assert contained(fp, buf)
        assert collision_free(fp, staged)
        staged.append(fp)
    assert len(staged) >= 2


def test_buffer_sampler_deterministic():
    buf = shp_box(0, 0, 20, 12)
    can = sample_shape(random.Random(2), family="can")
    assert sample_buffer_pose(can, buf, [], random.Random(7)) == sample_buffer_pose(
        can, buf, [], random.Random(7)
    )


def test_drawerworld_snapshot_restore(problem):
    w = DrawerWorld(problem.scene)
    o = problem.scene.blockers()[0]
    snap = w.snapshot()
    assert w.pick(o)
    assert w.states[o].region == "hand"
    w.restore(snap)
    assert w.states[o].region == "drawer" and w.held is None


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def test_scene_items_in_drawer_and_non_overlapping():
    sc = SC.generate_scene(0, lam=1.0)
    fps = {n: st.footprint() for n, st in sc.items.items()}
    for n, fp in fps.items():
        assert sc.drawer.buffer(1e-6).covers(fp), f"{n} not contained"
    names = list(fps)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert fps[names[i]].intersection(fps[names[j]]).area <= 1e-6
    tx, ty, _ = sc.items["target"].pose
    assert 0.25 * sc.dims["W"] <= tx <= 0.75 * sc.dims["W"]


def test_scene_deterministic():
    a = SC.generate_scene(3, lam=1.0)
    b = SC.generate_scene(3, lam=1.0)
    assert a.item_names() == b.item_names()
    assert a.items["target"].pose == b.items["target"].pose


def _collar_families(seed, diverse):
    # fill=0.0 disables the clutter top-up loop, so every non-target item is a collar item
    sc = SC.generate_scene(seed, lam=1.0, crowd=12, fill=0.0, diverse_crowd=diverse)
    # scene stays legal (contained + non-overlapping) regardless of collar diversity
    fps = [st.footprint() for st in sc.items.values()]
    for i, fp in enumerate(fps):
        assert sc.drawer.buffer(1e-6).covers(fp)
        for other in fps[i + 1 :]:
            assert fp.intersection(other).area <= 1e-6
    return {st.shape.family for n, st in sc.items.items() if not st.is_target}


def test_diverse_crowd_draws_collar_from_all_families():
    # default: collar is round-only (every collar item is in _COLLAR_FAMILIES)
    for seed in range(6):
        assert _collar_families(seed, diverse=False) <= set(SC._COLLAR_FAMILIES)
    # diverse: collar reaches beyond the round families across seeds
    seen: set[str] = set()
    for seed in range(12):
        seen |= _collar_families(seed, diverse=True)
    assert seen - set(
        SC._COLLAR_FAMILIES
    ), f"diverse collar never left the round families: {seen}"


def test_diverse_crowd_problem_id_and_require_subset_compat():
    # id marker + stored flag distinguish diverse datasets
    p = generate_dd2d_problem(
        lam=0.6, seed=0, crowd=10, diverse_crowd=True, certify=False
    )
    assert p.diverse_crowd and "dc" in p.problem_id
    q = generate_dd2d_problem(lam=0.6, seed=0, crowd=10, certify=False)
    assert not q.diverse_crowd and "dc" not in q.problem_id
    # F4 (require_subset) still honored under a diverse collar (resamples until it holds)
    r = generate_dd2d_problem(
        lam=0.6,
        seed=0,
        crowd=12,
        diverse_crowd=True,
        require_subset=True,
        certify=False,
    )
    assert r.min_feasible_subset is not None and r.min_feasible_subset >= 2


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #
def test_candidates_published_order_and_validity(problem):
    cands = problem.candidates
    assert len(cands) >= 2
    sizes = [c.size for c in cands]
    assert sizes == sorted(sizes)  # ascending |S| (published order)
    assert len(cands) <= EN.MAX_CANDIDATES
    for c in cands:
        assert c.subset and problem.target not in c.subset
        assert (c.extraction_reason == "extraction") == (not c.extractable)


def test_buried_member_is_extraction_infeasible():
    # a hand-built scene: target blocked by a blocker that is itself walled into a corner
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.world import DrawerScene, ItemState

    drawer = shp_box(0, 0, 40, 30)
    outer = shp_box(-1.5, -1.5, 41.5, 31.5)
    wall = outer.difference(drawer)
    buffer = shp_box(46, 0, 76, 16)
    tshape = Shape("can", Point(0, 0).buffer(4.0), False)
    bshape = Shape("can", Point(0, 0).buffer(3.5), False)
    # blocker jammed into the bottom-left corner touching two walls, target just above it
    items = {
        "target": ItemState("target", tshape, (5.0, 9.0, 0.0), "drawer", True),
        "b": ItemState("b", bshape, (3.6, 3.6, 0.0), "drawer", False),
    }
    scene = DrawerScene(drawer, wall, buffer, items, "target", 1.0)
    fps = EN._footprints(scene)
    # 'b' has no grasp clearing the walls -> any candidate containing it is extraction-infeasible
    assert has_grasp(bshape, items["b"].pose, [wall]) is None


# --------------------------------------------------------------------------- #
# labeler + filters
# --------------------------------------------------------------------------- #
def test_labels_are_three_valued_with_valid_reasons(problem):
    labels = {c.meta["label"] for c in problem.candidates}
    assert labels <= {"feasible", "infeasible", "marginal"}
    assert "feasible" in labels  # F3 guarantees at least one
    for c in problem.candidates:
        assert c.meta["reason"] in {
            "",
            "extraction",
            "packing",
            "inaccessible",
            "budget",
        }
        if c.meta["label"] == "feasible":
            assert c.meta["witness"] is not None and "order" in c.meta["witness"]


def test_area_bound_is_a_sound_infeasibility_prune(problem):
    # the whole blocker set cannot fit the (tight) buffer -> H1 proves infeasible
    all_blockers = frozenset(problem.scene.blockers())
    assert L._area_bound_infeasible(problem.scene, all_blockers)


def test_generated_problem_passes_all_filters(problem):
    filt = L.decision_filters(problem.scene, problem.candidates)
    assert filt["F1"] and filt["F2"] and filt["F3"]


def test_generation_deterministic():
    a = generate_dd2d_problem(**GEN_KW)
    b = generate_dd2d_problem(**GEN_KW)
    assert a.problem_id == b.problem_id
    assert a.init_facts == b.init_facts and a.goal_facts == b.goal_facts
    assert a.intended_skeleton().key() == b.intended_skeleton().key()


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
def _dd2d_symbolically_valid(problem, sk) -> bool:
    names = {a.name for a in sk.actions}
    if not names <= {"pick", "place-buffer", "retrieve"}:
        return False
    return (
        sk.actions[-1].name == "retrieve" and sk.actions[-1].args[0] == problem.target
    )


def test_planner_skeletons_end_in_retrieve_and_are_valid(problem):
    sks = DD2DPlanner(order="published").plan(problem, 12)
    assert len(sks) >= 2
    keys = [s.key() for s in sks]
    assert len(keys) == len(set(keys))  # distinct
    for s in sks:
        assert _dd2d_symbolically_valid(problem, s)
    lengths = [s.length for s in sks]
    assert lengths == sorted(lengths)  # ascending (published order)


def test_orderings_differ(problem):
    pub = [s.key() for s in DD2DPlanner("published").plan(problem, 12)]
    oracle = [s.key() for s in DD2DPlanner("oracle").plan(problem, 12)]
    assert set(pub) == set(oracle)  # same set of candidates, possibly different order
    # oracle puts a feasible candidate first
    first = DD2DPlanner("oracle").plan(problem, 1)[0]
    ref = DD2DRefiner(budget=200)
    assert ref.refine(first, problem.scene, seed=0).feasible


# --------------------------------------------------------------------------- #
# refiner
# --------------------------------------------------------------------------- #
def test_intended_refines_and_retrieve_only_does_not(problem):
    ref = DD2DRefiner(budget=200)
    assert ref.refine(problem.intended_skeleton(), problem.scene, seed=0).feasible
    r = ref.refine(problem.retrieve_only_skeleton(), problem.scene, seed=0)
    assert not r.feasible  # target is blocked (F1) -> "just grab it" fails


def test_infeasible_candidate_stuck_within_plan(problem):
    ref = DD2DRefiner(budget=200)
    sks = DD2DPlanner("published").plan(problem, 12)
    infeasible = [ref.refine(s, problem.scene, seed=7) for s in sks]
    bad = [r for r in infeasible if not r.feasible]
    assert bad, "expected at least one infeasible skeleton"
    for r in bad:
        assert 0 <= r.steps_bound < r.plan_length
        assert r.failure_action is not None


def test_refiner_deterministic(problem):
    ref = DD2DRefiner(budget=200)
    sk = problem.intended_skeleton()
    a = ref.refine(sk, problem.scene, seed=11)
    b = ref.refine(sk, problem.scene, seed=11)
    assert (a.status, a.steps_bound, a.n_attempts) == (
        b.status,
        b.steps_bound,
        b.n_attempts,
    )


# --------------------------------------------------------------------------- #
# feasibility signal + end to end
# --------------------------------------------------------------------------- #
def test_feasibility_signal_present(problem):
    ref = DD2DRefiner(budget=200)
    sks = DD2DPlanner("published").plan(problem, 12)
    res = [ref.refine(s, problem.scene, seed=1000 + i) for i, s in enumerate(sks)]
    assert any(r.feasible for r in res)
    assert any(not r.feasible for r in res)  # low base rate: not everything refines


def test_end_to_end_build_example(problem):
    ref = DD2DRefiner(budget=200)
    sk = problem.intended_skeleton()
    res = ref.refine(sk, problem.scene, seed=0)
    render = render_scene(problem.scene, width=320)
    imgs = build_image_refs(problem, render=render, views=("topdown",))
    ex = build_example(
        problem,
        sk,
        res,
        "dd2d-candidates",
        images=imgs,
        label_source=ref.label_source,
        extra_provenance={"refiner": ref.name},
    )
    assert ex.task_plan == sk.to_tokens_as_lists()
    assert ex.label is True
    assert ex.label_source == "refine_buffer_stage"
    assert len(imgs) == len(problem.objects)
    ex2 = PIGINetExample.from_json(ex.to_json())
    assert ex2.problem_id == ex.problem_id and ex2.task_plan == ex.task_plan


def test_render_scene_segments_all_items(problem):
    r = render_scene(problem.scene, width=400)
    assert len(r.segment_ids()) >= problem.num_blocks - 1
    assert set(r.id_to_name.values()) >= {o.name for o in problem.objects}


def test_render_episode_writes_file(problem, tmp_path):
    ref = DD2DRefiner(budget=200)
    sk = problem.intended_skeleton()
    res = ref.refine(sk, problem.scene, seed=0)
    out = render_episode(
        problem.scene,
        res.bound_plan,
        res.feasible,
        res.failure_action,
        str(tmp_path / "ep.mp4"),
        fmt="gif",
    )
    import os

    assert os.path.exists(out) and os.path.getsize(out) > 0


# --------------------------------------------------------------------------- #
# requiring a blocking SUBSET (the crowd knob) -- docs/dd2d.md
# --------------------------------------------------------------------------- #
def _cand(names, label):
    c = Candidate(subset=frozenset(names), members=list(names), extractable=True)
    c.meta["label"] = label
    return c


def test_min_feasible_subset_size_helper():
    assert min_feasible_subset_size([]) is None
    assert (
        min_feasible_subset_size([_cand("ab", "marginal"), _cand("c", "infeasible")])
        is None
    )
    # a size-1 feasible present -> 1; only size-2 feasible -> 2
    assert (
        min_feasible_subset_size([_cand("ab", "feasible"), _cand("c", "feasible")]) == 1
    )
    assert (
        min_feasible_subset_size([_cand("c", "marginal"), _cand("ab", "feasible")]) == 2
    )


def test_collar_pose_places_item_near_target():
    drawer = shp_box(0, 0, 40, 30)
    target = Shape("can", Point(0, 0).buffer(4.0), False)
    tpose = (20.0, 15.0, 0.0)
    tfp = place_polygon(target.polygon, tpose)
    collar = sample_shape(random.Random(1), family="can")
    pose = collar_pose(
        collar, drawer, [tfp], (20.0, 15.0), math.radians(30), random.Random(0)
    )
    assert pose is not None
    fp = place_polygon(collar.polygon, pose)
    assert drawer.buffer(1e-6).covers(fp)
    assert fp.distance(tfp) < 1.0  # hugs the target


def test_crowd_raises_subset_required_rate():
    # over a handful of seeds, crowding makes a meaningful fraction require a 2+ subset
    n = sum(
        generate_dd2d_problem(lam=0.6, seed=s, crowd=10, budget=200).requires_subset
        for s in range(6)
    )
    assert n >= 2, f"expected >=2/6 crowded problems to require a subset, got {n}"


@pytest.fixture(scope="module")
def subset_problem():
    return generate_dd2d_problem(
        lam=0.6, seed=0, crowd=10, require_subset=True, budget=200
    )


def test_require_subset_forces_a_multi_blocker_solution(subset_problem):
    p = subset_problem
    assert p.requires_subset and p.min_feasible_subset >= 2
    sk = p.intended_skeleton()
    n_place = sum(a.name == "place-buffer" for a in sk.actions)
    assert n_place >= 2 and sk.length >= 5  # stages >=2 blockers before retrieve
    assert DD2DRefiner(budget=200).refine(sk, p.scene, seed=0).feasible


def test_single_object_removal_is_insufficient(subset_problem):
    """The crux: on a require-subset problem, staging only the FIRST member of the
    intended subset and retrieving does NOT clear the target (one object isn't enough).
    """
    p = subset_problem
    first = p.intended_skeleton().actions[0].args[0]  # first staged blocker
    one_obj = Skeleton(
        (
            Action("pick", (first,)),
            Action("place-buffer", (first,)),
            Action("retrieve", (p.target,)),
        )
    )
    res = DD2DRefiner(budget=200).refine(one_obj, p.scene, seed=0)
    assert not res.feasible  # removing a single blocker leaves the target ungraspable
    assert (
        res.failure_action == f"retrieve({p.target})"
    )  # staged fine, still can't grasp


def test_crowd_zero_default_is_baseline(problem):
    # the module fixture uses no crowd -> generator default crowd=0 preserved
    assert problem.crowd == 0


# --------------------------------------------------------------------------- #
# pyperplan as a fair, deeper baseline -- docs/dd2d.md "Fair baselines"
# --------------------------------------------------------------------------- #
def test_unbounded_slack_reaches_multi_object_stagings(problem):
    capped = ForbidLoopPlanner(length_slack=2).plan(problem, 80)
    deep = ForbidLoopPlanner(length_slack=None).plan(problem, 80)
    # slack=2 caps at shortest(1)+2=3 -> single-object stagings only
    assert _max_place_buffer(capped) <= 1
    assert all(s.length <= 3 for s in capped)
    # unbounded reaches 2+ object stagings and returns strictly more plans
    assert _max_place_buffer(deep) >= 2
    assert len(deep) > len(capped)


def test_pyperplan_kwargs_thread_through():
    # make_planner previously dropped kwargs on the pyperplan path
    p = make_planner(prefer="pyperplan", length_slack=None)
    assert isinstance(p, ForbidLoopPlanner) and p.length_slack is None
    # a stray kwarg meant for another planner must not crash the fallback
    assert isinstance(make_planner(prefer="pyperplan", timeout=5.0), ForbidLoopPlanner)
    # the DD2D pyperplan path defaults to unbounded (the fair baseline)
    assert make_dd2d_planner(prefer="pyperplan").length_slack is None
    assert make_dd2d_planner(prefer="pyperplan", length_slack=4).length_slack == 4


def test_deep_baseline_reaches_the_feasible_plan(subset_problem):
    """The crux: on a subset-required instance the deep (unbounded) pyperplan baseline
    enumerates the feasible plan at a large k, while the slack=2 cap cannot -- so a standard
    planner CAN now reach it (just deep in the ranking), a fair baseline."""
    p = subset_problem
    assert p.min_feasible_subset == 2  # intended is a length-5 two-object staging
    intended_key = p.intended_skeleton().key()
    capped = ForbidLoopPlanner(length_slack=2).plan(p, 200)
    deep = ForbidLoopPlanner(length_slack=None).plan(p, 200)
    assert intended_key not in {s.key() for s in capped}  # capped can't reach length 5
    assert intended_key in {s.key() for s in deep}  # deep enumerates the feasible pair
    assert (
        DD2DRefiner(budget=200).refine(p.intended_skeleton(), p.scene, seed=0).feasible
    )


# --------------------------------------------------------------------------- #
# tunable refiner budget knobs (samples-per-step / retry-cap / time-budget)
# --------------------------------------------------------------------------- #
def test_samples_per_step_is_threaded_as_m_p(problem, monkeypatch):
    seen: list[int] = []
    import alphatamp.approaches.spectre.envs.dd2d.dd2d.refine as R

    orig = R.sample_buffer_pose
    monkeypatch.setattr(
        R,
        "sample_buffer_pose",
        lambda *a, **kw: (seen.append(kw.get("m_p")), orig(*a, **kw))[1],
    )
    DD2DRefiner(budget=200, samples_per_step=7).refine(
        problem.intended_skeleton(), problem.scene, seed=0
    )
    assert seen and set(seen) == {7}  # every sampler call got m_p=7


def test_time_budget_stops_immediately(problem):
    # uncapped stream calls, ~zero wall-clock -> nothing binds
    r = DD2DRefiner(budget=0, time_budget=1e-9).refine(
        problem.intended_skeleton(), problem.scene, seed=0
    )
    assert not r.feasible and r.steps_bound == 0 and r.n_attempts == 0


def test_time_governed_refine_without_call_cap(problem):
    ref = DD2DRefiner(budget=0, time_budget=2.0)  # governed by time, not stream calls
    assert ref.budget is None
    assert ref.refine(problem.intended_skeleton(), problem.scene, seed=0).feasible


def test_both_budgets_disabled_falls_back(recwarn):
    ref = DD2DRefiner(budget=0, time_budget=None)
    assert (
        ref.budget == 300
    )  # safety fallback so an infeasible subset can't thrash forever
    assert any("unbounded search" in str(w.message) for w in recwarn)


def test_retry_cap_and_defaults_preserved(problem):
    assert (
        DD2DRefiner(budget=200, retry_cap=3)
        .refine(problem.intended_skeleton(), problem.scene, seed=0)
        .feasible
    )
    d = DD2DRefiner()  # spec defaults P13/P14/P15
    assert (d.budget, d.retry_cap, d.samples_per_step, d.time_budget) == (
        300,
        10,
        15,
        None,
    )


# --------------------------------------------------------------------------- #
# best-first heuristic arms (heuristic_experiment) -- gbf/astar over the DD2D domain
# --------------------------------------------------------------------------- #
def test_bfs_arm_unchanged(problem):
    """Search='bfs' must be byte-for-byte the established unbounded baseline
    (regression)."""
    baseline = ForbidLoopPlanner(length_slack=None).plan(problem, 60)
    via_dd2d = make_dd2d_planner(prefer="pyperplan", search="bfs").plan(problem, 60)
    assert make_dd2d_planner(prefer="pyperplan").name == "pyperplan-bfs-diverse"
    assert [s.key() for s in via_dd2d] == [s.key() for s in baseline]


def test_heuristic_arm_planner_names():
    assert (
        make_dd2d_planner(prefer="pyperplan", search="gbf", heuristic="hff").name
        == "pyperplan-gbf-hff"
    )
    assert (
        make_dd2d_planner(prefer="pyperplan", search="astar", heuristic="dist").name
        == "pyperplan-astar-dist"
    )
    with pytest.raises(ValueError):
        make_dd2d_planner(prefer="pyperplan", search="gbf", heuristic="bogus")


@pytest.mark.parametrize("search", ["gbf", "astar"])
@pytest.mark.parametrize("heuristic", ["hff", "dist"])
def test_bestfirst_arms_enumerate_valid_skeletons(subset_problem, search, heuristic):
    pl = make_dd2d_planner(prefer="pyperplan", search=search, heuristic=heuristic)
    sks = pl.plan(subset_problem, 20)
    assert len(sks) >= 2
    assert len({s.key() for s in sks}) == len(sks)  # distinct
    for s in sks:
        assert _dd2d_symbolically_valid(subset_problem, s)


def test_gbf_dist_reorders_off_ascending_length(subset_problem):
    """The whole point: a best-first arm's order is NOT the blind ascending-length order.
    gbf+distance minimises remaining proximity mass, so it front-loads multi-object stagings.
    """
    sks = make_dd2d_planner(prefer="pyperplan", search="gbf", heuristic="dist").plan(
        subset_problem, 20
    )
    lengths = [s.length for s in sks]
    assert lengths != sorted(lengths)  # not the ascending-length baseline
    assert _max_place_buffer(sks[:3]) >= 2  # reaches multi-blocker stagings early


def test_distance_heuristic_sign():
    """H falls when a NEAR (blocking) item is cleared vs when a FAR (distractor) item
    is."""
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.heuristics import (
        distance_heuristic_factory,
    )
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.world import DrawerScene, ItemState

    drawer, wall, buf = (
        shp_box(0, 0, 40, 30),
        shp_box(-1, -1, 41, 31),
        shp_box(46, 0, 76, 16),
    )
    can = lambda: Shape("can", Point(0, 0).buffer(2.0), False)
    items = {
        "target": ItemState("target", can(), (10.0, 10.0, 0.0), "drawer", True),
        "near": ItemState("near", can(), (13.0, 10.0, 0.0), "drawer", False),
        "far": ItemState("far", can(), (30.0, 10.0, 0.0), "drawer", False),
    }
    scene = DrawerScene(drawer, wall, buf, items, "target", 1.0)

    class _P:  # minimal duck-type: the distance factory only reads .scene / .target
        pass

    p = _P()
    p.scene, p.target = scene, "target"
    h = distance_heuristic_factory("inv")(None, p)

    class _N:
        def __init__(self, state):
            self.state = state

    common = {"(handempty)", "(target target)", "(in-drawer target)"}
    rm_near = _N(frozenset(common | {"(on-buffer near)", "(in-drawer far)"}))
    rm_far = _N(frozenset(common | {"(in-drawer near)", "(on-buffer far)"}))
    assert h(rm_near) < h(
        rm_far
    )  # clearing the near/blocking item is the more promising state
