"""Tests for the pooled StickButton2D collection: pool filter, ids, geometry, evidence.

Each of these guards a failure mode that is silent rather than loud — a degenerate pool
that still has 200 entries, a stratum encoding that mislabels every episode, missing
geometry that makes training exit 0 with no checkpoint, and object names that stop
resolving after canonicalization.
"""

from __future__ import annotations

import pytest

from alphatamp.approaches.spectre.dd2d_compare import stratum_of
from alphatamp.approaches.spectre.envs.stickbutton2d import strata
from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import _is_acyclic


class _FakeAtom:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeAtom) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _FakeState:
    def __init__(self, *names: str) -> None:
        self.atoms = frozenset(_FakeAtom(n) for n in names)


def test_acyclic_rejects_a_returning_plan() -> None:
    """PickStick then PlaceStick returns to s_0 exactly — the padding this filter kills."""
    s0 = _FakeState("HandEmpty", "AboveNoButton")
    held = _FakeState("Grasped", "AboveNoButton")
    assert _is_acyclic([s0, held]) is True
    assert _is_acyclic([s0, held, _FakeState("HandEmpty", "AboveNoButton")]) is False


def test_acyclic_accepts_a_progressing_plan() -> None:
    plan = [
        _FakeState("HandEmpty"),
        _FakeState("Grasped"),
        _FakeState("Grasped", "Pressed(b0)"),
        _FakeState("Grasped", "Pressed(b0)", "Pressed(b1)"),
    ]
    assert _is_acyclic(plan) is True


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("num_buttons", strata.BUTTON_COUNTS)
def test_problem_id_round_trips(split: str, num_buttons: int) -> None:
    for index in (0, 1, 99, 12_345):
        pid = strata.problem_id(split, num_buttons, index)
        assert strata.decode(pid) == (split, num_buttons, index)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_stratum_of_recovers_the_button_count(split: str) -> None:
    """The arithmetic identity the 15 existing ``stratum_of`` call sites depend on.

    If this breaks, every per-stratum table silently reports the wrong variant rather
    than erroring.
    """
    for slot, num_buttons in enumerate(strata.BUTTON_COUNTS):
        for index in (0, 42, 24_999):
            pid = strata.problem_id(split, num_buttons, index)
            assert stratum_of(pid) == slot


def test_splits_never_share_a_problem_id() -> None:
    """Distinct ids across splits are what stop a train scene reappearing in test."""
    ids = {
        strata.problem_id(split, b, i)
        for split in strata.SPLIT_SIZES
        for b in strata.BUTTON_COUNTS
        for i in range(50)
    }
    assert len(ids) == 3 * len(strata.BUTTON_COUNTS) * 50


def test_index_overflow_is_rejected_not_wrapped() -> None:
    """An index past the band would decode as a different button count."""
    with pytest.raises(ValueError, match="overflows the stratum band"):
        strata.problem_id("train", 3, 250_000)


def test_split_sizes_pool_to_the_intended_totals() -> None:
    n = len(strata.BUTTON_COUNTS)
    assert (
        strata.SPLIT_SIZES["train"] * n,
        strata.SPLIT_SIZES["val"] * n,
        strata.SPLIT_SIZES["test"] * n,
    ) == (400, 100, 100)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _sb2d_state(num_buttons: int = 3, seed: int = 0):
    # pylint: disable=import-outside-toplevel
    import kinder
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    from alphatamp.approaches.spectre.env_registry import register_extra_envs

    register_extra_envs()
    env = kinder.make(f"kinder/StickButton2D-b{num_buttons}-v0")
    try:
        obs, _ = env.reset(seed=seed)
        models = create_bilevel_planning_models(
            "stickbutton2d",
            env.observation_space,
            env.action_space,
            num_buttons=num_buttons,
        )
        return models.observation_to_state(obs)
    finally:
        env.close()


def test_scene_geometry_covers_every_abstract_object() -> None:
    """Invariant I5: every ``object_registry`` key needs geometry, or ``validate`` raises."""
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.stickbutton2d.scene_geometry import (
        build_scene_geometry,
    )

    geo = build_scene_geometry(_sb2d_state(3))
    names = {o.name for o in geo.objects}
    assert names == {"robot", "stick", "button0", "button1", "button2"}
    assert {c.kind for c in geo.containers} == {"table", "world"}
    assert geo.frame == {"frame_w": 3.5, "frame_d": 2.5}


def test_stick_pose_is_its_centroid_not_its_corner() -> None:
    """``Rectangle(x, y, w, h, theta)`` takes the lower-left corner.

    Reading it as a centroid would displace the 1.25-long stick by 0.625 world units and
    nothing downstream would complain.
    """
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.stickbutton2d.scene_geometry import (
        build_scene_geometry,
    )

    state = _sb2d_state(3)
    stick_obj = next(o for o in state if o.name == "stick")
    corner_y = float(state.get(stick_obj, "y"))
    height = float(state.get(stick_obj, "height"))

    stick = next(o for o in build_scene_geometry(state).objects if o.name == "stick")
    assert stick.pose[1] == pytest.approx(corner_y + height / 2.0)
    # The ring is centred, so it straddles the origin rather than starting there.
    ys = [p[1] for p in stick.boundary]
    assert min(ys) == pytest.approx(-height / 2.0)
    assert max(ys) == pytest.approx(height / 2.0)
    assert stick.area == pytest.approx(
        height * float(state.get(stick_obj, "width")), rel=1e-6
    )


def test_robot_geometry_is_the_base_disc() -> None:
    """The arm is configuration, not footprint; only the base blocks motion."""
    # pylint: disable=import-outside-toplevel
    import math

    from alphatamp.approaches.spectre.envs.stickbutton2d.scene_geometry import (
        build_scene_geometry,
    )

    state = _sb2d_state(3)
    robot_obj = next(o for o in state if o.name == "robot")
    radius = float(state.get(robot_obj, "base_radius"))
    robot = next(o for o in build_scene_geometry(state).objects if o.name == "robot")
    assert robot.family == "circle"
    assert robot.area == pytest.approx(math.pi * radius * radius)
    assert robot.pose[0] == pytest.approx(float(state.get(robot_obj, "x")))
