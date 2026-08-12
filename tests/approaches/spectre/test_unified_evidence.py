"""Pin `unified_evidence` against the worked examples of the design document.

Both tables in ``docs/unified_culprits_coverage_waste.md`` §7 (DD2D) and §8 (SB2D)
are reproduced number-for-number. They are the spec's own statement of intent, so a
divergence here means either the implementation or the document is wrong — which is
exactly what these probes exist to catch before a collection is built on the
definitions.

The SB2D case uses the **real** kinder operator schemas rather than a transcription,
since the whole construction is derived from preconditions and effects; a hand-copied
operator would test the transcription instead of the design.
"""

from __future__ import annotations

import pytest
from relational_structs import GroundAtom, Object

from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import (
    OPERATOR_BY_NAME,
    Extracted,
    InDrawer,
    ItemType,
    Target,
)
from alphatamp.approaches.spectre.unified_evidence import (
    Deviation,
    UnifiedRecord,
    actionable_objects,
    anchored,
    blame,
    collateral,
    coverage,
    culprit_pool,
    matched_steps,
    superfluous_steps,
    universal_objects,
    waste,
)

# --------------------------------------------------------------------------- #
# §7 — DD2D
# --------------------------------------------------------------------------- #
_T = Object("t", ItemType)
_O1 = Object("o1", ItemType)
_O2 = Object("o2", ItemType)
_O3 = Object("o3", ItemType)
_DD2D_OBJECTS = [_T, _O1, _O2, _O3]


def _dd2d_ground_ops() -> list:
    return [
        OPERATOR_BY_NAME[name].ground((o,))
        for name in ("pick", "place-buffer", "retrieve")
        for o in _DD2D_OBJECTS
    ]


def _stage(*blockers) -> list:
    """``[pick(o), place-buffer(o) …] ++ retrieve(t)`` — DD2D's staging skeleton."""
    steps = []
    for o in blockers:
        steps.append(OPERATOR_BY_NAME["pick"].ground((o,)))
        steps.append(OPERATOR_BY_NAME["place-buffer"].ground((o,)))
    steps.append(OPERATOR_BY_NAME["retrieve"].ground((_T,)))
    return steps


_DD2D_INIT = frozenset(
    {GroundAtom(Target, [_T])} | {GroundAtom(InDrawer, [o]) for o in _DD2D_OBJECTS}
)
_DD2D_GOAL = frozenset({GroundAtom(Extracted, [_T])})


def test_dd2d_has_no_universal_object() -> None:
    """`Universal = ∅` on DD2D by construction — `pick(o1)` does not mention `o2`.

    This is why excluding universal objects from `K` is provably a no-op on DD2D
    rather than an empirical one, and it is worth a test because the whole
    backward-compatibility argument rests on it.
    """
    assert universal_objects(_dd2d_ground_ops()) == frozenset()
    assert actionable_objects(_dd2d_ground_ops()) == {"t", "o1", "o2", "o3"}


def test_dd2d_worked_example_table() -> None:
    """§7's table: the class-1 collision record on a terminal context."""
    ops = _dd2d_ground_ops()
    universal = universal_objects(ops)
    record = UnifiedRecord(
        failed_step=OPERATOR_BY_NAME["retrieve"].ground((_T,)),
        deviation=None,  # class 1: the grasp check rejected the sample
        check_blame=("o1",),
    )
    records = [record]
    pool = culprit_pool(records, ops)
    assert pool == {"o1"}

    c1, c2, c3 = _stage(_O1), _stage(_O2), _stage(_O1, _O2)
    expected = {
        "c1": (c1, 1.0, 0.0),
        "c3": (c3, 1.0, 0.5),
        "c2": (c2, 0.0, 1.0),
    }
    for name, (cand, exp_cov, exp_waste) in expected.items():
        cov = coverage(cand, records, pool, _DD2D_INIT, universal)
        wst = waste(cand, records, pool, _DD2D_GOAL, universal)
        assert cov == pytest.approx(exp_cov), f"{name} coverage"
        assert wst == pytest.approx(exp_waste), f"{name} waste"


def test_dd2d_second_attempt_escalates() -> None:
    """§7 attempt 2: once `o2` is also blamed, `c3` becomes the fully-covered
    candidate."""
    ops = _dd2d_ground_ops()
    universal = universal_objects(ops)
    retrieve_t = OPERATOR_BY_NAME["retrieve"].ground((_T,))
    records = [
        UnifiedRecord(failed_step=retrieve_t, check_blame=("o1",)),
        UnifiedRecord(failed_step=retrieve_t, check_blame=("o2",)),
    ]
    pool = culprit_pool(records, ops)
    assert pool == {"o1", "o2"}

    c2, c3 = _stage(_O2), _stage(_O1, _O2)
    assert coverage(c3, records, pool, _DD2D_INIT, universal) == pytest.approx(1.0)
    assert waste(c3, records, pool, _DD2D_GOAL, universal) == pytest.approx(0.0)
    assert coverage(c2, records, pool, _DD2D_INIT, universal) == pytest.approx(0.5)
    assert waste(c2, records, pool, _DD2D_GOAL, universal) == pytest.approx(0.0)


def test_dd2d_waste_denominator_is_the_staging_steps() -> None:
    """§5's DD2D compat argument: `handempty` is filtered, so each staging pair
    dead-ends.

    If `handempty` were anchored it would chain every pair into the causal spine and the
    denominator would collapse to zero, silently breaking waste on the deployed
    environment.
    """
    universal = universal_objects(_dd2d_ground_ops())
    assert superfluous_steps(_stage(_O1), _DD2D_GOAL, universal) == {0, 1}
    assert superfluous_steps(_stage(_O1, _O2), _DD2D_GOAL, universal) == {0, 1, 2, 3}


# --------------------------------------------------------------------------- #
# §8 — SB2D
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sb2d():
    """Real SB2D lifted operators, grounded over the §8 example's four buttons."""
    import kinder  # pylint: disable=import-outside-toplevel

    # pylint: disable-next=import-outside-toplevel
    from kinder.envs.kinematic2d.object_types import (
        CircleType,
        CRVRobotType,
        RectangleType,
    )

    # pylint: disable-next=import-outside-toplevel
    from kinder_bilevel_planning.env_models import (
        create_bilevel_planning_models,
    )

    kinder.register_all_environments()
    env = kinder.make("kinder/StickButton2D-b5-v0")
    try:
        models = create_bilevel_planning_models(
            "stickbutton2d",
            env.observation_space,
            env.action_space,
            num_buttons=5,
        )
    finally:
        env.close()

    lifted = {op.name: op for op in models.operators}
    robot = Object("robot", CRVRobotType)
    stick = Object("stick", RectangleType)
    buttons = {i: Object(f"button{i}", CircleType) for i in (1, 2, 3, 4)}

    def rp(target, frm=None):
        if frm is None:
            return lifted["RobotPressButtonFromNothing"].ground(
                (robot, buttons[target])
            )
        return lifted["RobotPressButtonFromButton"].ground(
            (robot, buttons[target], buttons[frm])
        )

    def sp(target, frm=None):
        if frm is None:
            return lifted["StickPressButtonFromNothing"].ground(
                (robot, stick, buttons[target])
            )
        return lifted["StickPressButtonFromButton"].ground(
            (robot, stick, buttons[target], buttons[frm])
        )

    def pick(frm=None):
        if frm is None:
            return lifted["PickStickFromNothing"].ground((robot, stick))
        return lifted["PickStickFromButton"].ground((robot, stick, buttons[frm]))

    place = lifted["PlaceStick"].ground((robot, stick))

    ground_ops = []
    for _name, op in lifted.items():
        types = [p.type.name for p in op.parameters]
        pools = {
            "crv_robot": [robot],
            "rectangle": [stick],
            "circle": list(buttons.values()),
        }
        import itertools  # pylint: disable=import-outside-toplevel

        for combo in itertools.product(*[pools[t] for t in types]):
            ground_ops.append(op.ground(tuple(combo)))

    goal = frozenset(
        GroundAtom(
            next(p for p in models.predicates if p.name == "Pressed"), [buttons[i]]
        )
        for i in (1, 2, 3, 4)
    )
    init = frozenset(
        {
            GroundAtom(
                next(p for p in models.predicates if p.name == "HandEmpty"), [robot]
            ),
            GroundAtom(
                next(p for p in models.predicates if p.name == "AboveNoButton"), []
            ),
        }
    )
    return {
        "rp": rp,
        "sp": sp,
        "pick": pick,
        "place": place,
        "ground_ops": ground_ops,
        "goal": goal,
        "init": init,
        "buttons": buttons,
        "robot": robot,
        "stick": stick,
    }


def test_sb2d_filters(sb2d) -> None:
    """§1: the robot is universal, the stick is not (absent from robot-press
    operators)."""
    universal = universal_objects(sb2d["ground_ops"])
    assert universal == {"robot"}
    # The stick must stay a possible culprit — §4 relies on a knocked-loose `Grasped`.
    assert "stick" in actionable_objects(sb2d["ground_ops"]) - universal


def test_sb2d_anchoring_rejects_bookkeeping_atoms(sb2d) -> None:
    """`HandEmpty(robot)` and `AboveNoButton()` must not anchor anything."""
    universal = universal_objects(sb2d["ground_ops"])
    assert anchored(sb2d["init"], universal) == frozenset()
    # ...while a press effect does anchor, via the button.
    press = sb2d["rp"](2)
    assert anchored(press.add_effects, universal) == frozenset(press.add_effects)


def test_sb2d_out_of_reach_record_yields_no_culprits(sb2d) -> None:
    """§8's means-failure record: `Δ̃_r = (∅, ∅)`, so `K` stays empty.

    This is the case that motivated the collateral restriction. Under the *unrestricted*
    deviation this record would have blamed `b4` and the robot, and its `D_r` half would
    have handed full coverage credit to a candidate retrying the identical doomed press.
    """
    step = sb2d["rp"](4)
    record = UnifiedRecord(
        failed_step=step,
        deviation=Deviation(
            added=frozenset(step.delete_effects),  # AboveNoButton failed to be deleted
            deleted=frozenset(step.add_effects),  # its own adds never materialized
        ),
    )
    assert collateral(record).is_empty()
    assert blame(record) == frozenset()
    assert culprit_pool([record], sb2d["ground_ops"]) == frozenset()


def test_sb2d_worked_example_table(sb2d) -> None:
    """§8's coverage table, including `c_D` denied credit across modality."""
    rp, sp, pick, place = sb2d["rp"], sb2d["sp"], sb2d["pick"], sb2d["place"]
    universal = universal_objects(sb2d["ground_ops"])

    pressed_b1 = next(a for a in rp(1).add_effects if a.predicate.name == "Pressed")
    record = UnifiedRecord(
        failed_step=rp(2),  # the approach to b2 brushed b1
        deviation=Deviation(added=frozenset({pressed_b1}), deleted=frozenset()),
    )
    records = [record]
    pool = culprit_pool(records, sb2d["ground_ops"])
    assert pool == {"button1"}

    c_b = [rp(1), rp(2, 1), rp(3, 2), pick(3), sp(4)]
    c_c = [rp(3), rp(2, 3), rp(1, 2), pick(1), sp(4)]
    c_d = [pick(), sp(4), sp(2, 4), sp(1, 2), sp(3, 1)]
    c_e = [rp(1), rp(2, 1), rp(3, 2), pick(3), place, pick(), sp(4)]

    for name, cand, expected in (
        ("c_B", c_b, 1.0),
        ("c_C", c_c, 0.0),
        ("c_D", c_d, 0.0),
        ("c_E", c_e, 1.0),
    ):
        got = coverage(cand, records, pool, sb2d["init"], universal)
        assert got == pytest.approx(expected), f"{name} coverage"

    # c_D is matched through a *stick* press — effect-based matching across modality.
    assert matched_steps(c_d, record, universal) == {2}


def test_sb2d_waste_dissolves_the_stick_anti_signal(sb2d) -> None:
    """§8: `c_B` is all-live (waste 0), `c_E`'s place-repick cycle is unexplained
    (waste 1).

    The deployed object-level feature gave both candidates `S(c) = {stick}` and
    therefore waste 1.0 — it could not see the difference at all.
    """
    rp, sp, pick, place = sb2d["rp"], sb2d["sp"], sb2d["pick"], sb2d["place"]
    universal = universal_objects(sb2d["ground_ops"])
    pressed_b1 = next(a for a in rp(1).add_effects if a.predicate.name == "Pressed")
    records = [
        UnifiedRecord(
            failed_step=rp(2),
            deviation=Deviation(added=frozenset({pressed_b1}), deleted=frozenset()),
        )
    ]
    pool = culprit_pool(records, sb2d["ground_ops"])

    c_b = [rp(1), rp(2, 1), rp(3, 2), pick(3), sp(4)]
    c_e = [rp(1), rp(2, 1), rp(3, 2), pick(3), place, pick(), sp(4)]

    assert superfluous_steps(c_b, sb2d["goal"], universal) == frozenset()
    assert waste(c_b, records, pool, sb2d["goal"], universal) == pytest.approx(0.0)

    assert superfluous_steps(c_e, sb2d["goal"], universal) == {3, 4}
    assert waste(c_e, records, pool, sb2d["goal"], universal) == pytest.approx(1.0)


def test_leakage_invariant_no_records_means_zero(sb2d) -> None:
    """Both features are exactly 0 before any failure is observed (§0)."""
    rp, sp, pick = sb2d["rp"], sb2d["sp"], sb2d["pick"]
    universal = universal_objects(sb2d["ground_ops"])
    cand = [rp(1), rp(2, 1), rp(3, 2), pick(3), sp(4)]
    assert coverage(cand, [], frozenset(), sb2d["init"], universal) == 0.0
    assert waste(cand, [], frozenset(), sb2d["goal"], universal) == 0.0


def test_memoized_matches_naive_recomputation() -> None:
    """The `_Memo` hoist is a pure speedup: identical output, on real dd2d_v4 data.

    `matched_steps`, `touch`, `blame` and `collateral` were previously recomputed
    inside the innermost loops. Hoisting them cut 283 s/epoch to a fraction of that,
    and the only thing that makes the optimisation safe is that it changes nothing — so
    this compares the memoized path against a from-scratch recomputation, candidate by
    candidate.
    """
    from pathlib import Path

    from relational_structs.utils import all_ground_operators

    from alphatamp.approaches.spectre.domain import spec_for
    from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import (
        ALL_OPERATORS as DD2D_ALL,)
    from alphatamp.approaches.spectre.io import list_episodes, load_episode
    from alphatamp.approaches.spectre.unified_evidence import (
        coverage_and_waste,
        records_from_failure_records,
    )

    split = Path("data/spectre/raw/dd2d_v4/test")
    paths = list_episodes(split)
    if not paths:
        pytest.skip("dd2d_v4 test split not present")

    checked = 0
    for path in paths[:3]:
        episode = load_episode(path)
        spec = spec_for(episode.provenance.env_variant)
        ground = list(
            all_ground_operators(DD2D_ALL, set(episode.initial_abstract_state.objects))
        )
        univ = universal_objects(ground)
        fails = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"][:3]
        recs = records_from_failure_records(episode, frozenset(fails), spec)
        if not recs:
            continue
        pool = culprit_pool(recs, ground)
        if not pool:
            continue
        for skeleton in episode.skeleton_pool[:40]:
            cand = list(skeleton.operator_seq)
            fast_cov, fast_wst = coverage_and_waste(
                cand,
                recs,
                pool,
                episode.initial_abstract_state.atoms,
                episode.goal_atoms,
                univ,
            )
            # `memo=None` forces the original per-call recomputation.
            slow_cov = coverage(
                cand, recs, pool, episode.initial_abstract_state.atoms, univ
            )
            slow_wst = waste(cand, recs, pool, episode.goal_atoms, univ)
            assert fast_cov == slow_cov
            assert fast_wst == slow_wst
            checked += 1
    assert checked > 0, "no candidate exercised the memoized path"


# --------------------------------------------------------------------------- #
# blameless records: kept, and provably inert
# --------------------------------------------------------------------------- #
def test_blameless_records_do_not_change_coverage_or_waste() -> None:
    """Records that name nobody are kept, and must not move either feature.

    `records_from_failure_records` no longer filters them: an environment with no
    class-1 channel produces nothing else, and whether a record blames anyone should be
    data rather than a reason to drop it. That is only safe because every consumer
    already skips a blameless record -- it adds nothing to `K`, `covered` skips it per
    object and `_justified` never consults it -- so this pins the claim rather than
    assuming it.
    """
    ops = _dd2d_ground_ops()
    universal = universal_objects(ops)
    retrieve_t = OPERATOR_BY_NAME["retrieve"].ground((_T,))
    pick_o1 = OPERATOR_BY_NAME["pick"].ground((_O1,))
    real = [UnifiedRecord(failed_step=retrieve_t, check_blame=("o1",))]
    padded = real + [
        UnifiedRecord(failed_step=pick_o1, check_blame=()),
        UnifiedRecord(failed_step=retrieve_t, deviation=Deviation()),
    ]

    assert culprit_pool(padded, ops) == culprit_pool(real, ops) == {"o1"}
    pool = culprit_pool(real, ops)
    for cand in (_stage(_O1), _stage(_O2), _stage(_O1, _O2), _stage()):
        assert coverage(cand, padded, pool, _DD2D_INIT, universal) == coverage(
            cand, real, pool, _DD2D_INIT, universal
        )
        assert waste(cand, padded, pool, _DD2D_GOAL, universal) == waste(
            cand, real, pool, _DD2D_GOAL, universal
        )


def test_waste_abstains_when_no_culprit_is_named() -> None:
    """The one arithmetic edge case keeping blameless records would otherwise expose.

    With an empty `K` nothing can justify any idle step, so the ratio would be a
    maximally confident 1.0 derived from zero evidence -- and, worse, it would appear
    only on contexts whose records happened to blame nobody, i.e. as noise correlated
    with having no information.
    """
    ops = _dd2d_ground_ops()
    universal = universal_objects(ops)
    blameless = [
        UnifiedRecord(
            failed_step=OPERATOR_BY_NAME["pick"].ground((_O1,)), check_blame=()
        )
    ]
    cand = _stage(_O1, _O2)
    assert superfluous_steps(cand, _DD2D_GOAL, universal)  # denominator is non-empty
    assert waste(cand, blameless, frozenset(), _DD2D_GOAL, universal) == 0.0
    assert coverage(cand, blameless, frozenset(), _DD2D_INIT, universal) == 0.0


def test_malformed_failure_entry_does_not_shift_deviation_alignment() -> None:
    """Deviations stay attached to the record they came from.

    `records_from_failure_records` pairs its records with the raw metadata positionally,
    and `records_for_candidate` silently drops entries missing `schema`/`step_index`. If
    only one side filters, every deviation after the malformed entry lands on the wrong
    record — both sides stay well-formed, so nothing raises and the features are simply
    wrong.
    """
    from _fixtures import build_toy_episode  # pylint: disable=import-outside-toplevel

    # pylint: disable-next=import-outside-toplevel
    from alphatamp.approaches.spectre.domain import (
        EMPTY_SPEC,
    )

    # pylint: disable-next=import-outside-toplevel
    from alphatamp.approaches.spectre.unified_evidence import (
        records_from_failure_records,
    )

    ep = build_toy_episode(outcomes=("fail", "success"))
    obj = next(iter(ep.object_registry))
    ep.outcomes[0].refiner_metadata["failures"] = [
        {"malformed": True},  # dropped by records_for_candidate
        {
            "step_index": 0,
            "schema": "pick",
            "args": [obj],
            "culprits": [],
            "dev_added": [["OnTable", [obj]]],
            "dev_deleted": [],
        },
    ]

    recs = records_from_failure_records(ep, frozenset({0}), EMPTY_SPEC)
    assert len(recs) == 1
    # The surviving record must carry the deviation from the *well-formed* entry, not
    # `None` inherited from the malformed one at position 0.
    assert recs[0].deviation is not None
    assert not recs[0].is_class_1
