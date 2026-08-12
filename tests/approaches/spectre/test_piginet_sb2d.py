"""Tests for the StickButton2D PIGINet adapter.

The failure modes guarded here are all silent: a unit mismatch that collapses a feature
channel toward zero, a gloss table with a hole in it that degrades a word to its raw token,
a crop map missing an object so CLIP reads zeros, and a problem-id that does not survive
the round trip the comparison cache does on it. None of them raise; each would just make
the low-level baseline look worse than it is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alphatamp.approaches.spectre.baselines.piginet.dd2d_adapter import DD2DDomain
from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import GLOSSES, SB2DDomain

_ROOT = Path(__file__).resolve().parents[3]
_SPLIT = _ROOT / "data" / "spectre" / "raw" / "stickbutton2d_v1" / "test"

pytestmark = pytest.mark.skipif(
    not (_SPLIT / "episodes").is_dir(),
    reason="stickbutton2d_v1 collection not present (gitignored data)",
)


@pytest.fixture(scope="module")
def domain():
    return SB2DDomain(_ROOT / "data" / "spectre")


@pytest.fixture(scope="module")
def sample(domain):
    """One problem's (pid, examples), plus the episode it was built from."""
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.io import list_episodes, load_episode

    pid, examples = next(iter(domain.problems("test")))
    episode = load_episode(list_episodes(_SPLIT)[0])
    return pid, examples, episode


# --------------------------------------------------------------------------- #
# the normalisers — the reason `domain` exists at all
# --------------------------------------------------------------------------- #
def test_sb2d_scales_come_from_the_env_config(domain) -> None:
    """Derived, not hardcoded, so a config change cannot silently invalidate them."""
    # pylint: disable=import-outside-toplevel
    from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig

    cfg = StickButton2DEnvConfig()
    assert domain.frame_extent == (
        float(cfg.world_max_x) - float(cfg.world_min_x),
        float(cfg.world_max_y) - float(cfg.world_min_y),
    )
    # h_max is the stick's length; it is the largest extent in the scene.
    assert domain.shape_max[1] == pytest.approx(float(cfg.stick_shape[1]))


def test_sb2d_shape_features_do_not_underflow(domain, sample) -> None:
    """The highest-risk trap in the port, pinned as a test.

    DD2D's divisors are centimetres over a ~50x40 drawer; StickButton2D is metres over
    3.5x2.5 with objects two orders of magnitude smaller. Against the DD2D constants every
    shape feature collapses toward 0 and PIGINet reads as hopeless -- a unit bug that would
    be published as "the low-level predictor loses on this environment".
    """
    _, examples, _ = sample
    shapes = np.array(
        [
            [o["shape"]["w"], o["shape"]["h"], o["shape"]["area"], 0.0]
            for o in examples[0].objects
        ],
        dtype=np.float32,
    )
    own = shapes / domain.shape_max
    wrong = shapes / DD2DDomain().shape_max

    assert own.max() <= 1.0 + 1e-6, "own normaliser must bound the features"
    assert np.abs(own).mean() > 0.05, "own normaliser must not squash them either"
    # The regression this guards: the wrong divisors are ~an order of magnitude flatter.
    assert np.abs(wrong).mean() < np.abs(own).mean() / 5.0


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #
def test_labels_are_the_collections_own(sample) -> None:
    """PIGINet and SPECTRE must be scored against identical ground truth.

    Not "agree with" -- *be*. The examples are built from the same `EpisodeRecord` the
    ranker trains on, so a divergence here would mean the comparison compares two
    different label sets.
    """
    _, examples, episode = sample
    assert len(examples) == len(episode.outcomes)
    for ex, out in zip(examples, episode.outcomes):
        assert ex.label == (out.outcome == "success")


def test_task_plan_mirrors_the_operator_sequence(sample) -> None:
    _, examples, episode = sample
    for ex, skel in zip(examples, episode.skeleton_pool):
        assert ex.task_plan == [
            [op.name] + [p.name for p in op.parameters] for op in skel.operator_seq
        ]


def test_pose_literals_are_synthesised_for_every_object(sample) -> None:
    """Without them PIGINet sees a two-atom abstract state and no positions at all.

    StickButton2D's initial abstract state is `{HandEmpty, AboveNoButton}`. A *low-level*
    predictor that never receives a coordinate is not a low-level predictor, so the adapter
    emits an `at-pose` literal per object exactly as DD2D's records carry one.
    """
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.baselines.piginet.dataset import POSE_PREDICATE

    _, examples, _ = sample
    ex = examples[0]
    posed = {lit[1] for lit in ex.init_literals if lit[0] == POSE_PREDICATE}
    assert posed == {o["name"] for o in ex.objects}


def test_problem_id_round_trips_through_the_cache_convention(sample) -> None:
    """`precompute_dd2d_cache` recovers the integer via `pid.split("_s")[-1]`."""
    pid, _, episode = sample
    assert int(pid.split("_s")[-1]) == episode.provenance.problem_id


# --------------------------------------------------------------------------- #
# vocabulary and crops
# --------------------------------------------------------------------------- #
def test_every_word_in_the_data_is_glossed(domain) -> None:
    """An unglossed word silently falls back to its raw token, losing §IV-A's rephrasing."""
    seen: set[str] = set()
    for n, (_pid, examples) in enumerate(domain.problems("test")):
        for ex in examples[:5]:
            seen.update(step[0] for step in ex.task_plan)
            seen.update(lit[0] for lit in ex.init_literals)
            seen.update(lit[0] for lit in ex.goal_literals)
            seen.update(o["color"] for o in ex.objects)
        if n >= 20:
            break
    missing = sorted(seen - set(GLOSSES))
    assert not missing, f"unglossed domain words: {missing}"


def test_crops_cover_every_object(domain, sample) -> None:
    """A missing crop is cached as a zero vector, not an error."""
    pid, examples, _ = sample
    crops = domain.crops("test", pid)
    assert set(crops) == {o["name"] for o in examples[0].objects}
    assert all(img.size == img.size and min(img.size) > 0 for img in crops.values())


def test_crops_preserve_relative_scale(domain, sample) -> None:
    """The stick must not render like a button.

    Crops share one fixed world window rather than being individually framed, so relative
    size survives. It is the only visual difference this environment affords -- every
    unpressed button is the same red disc -- and rendering each object to fill its own
    frame would erase it.
    """
    pid, _, _ = sample
    crops = domain.crops("test", pid)
    filled = {
        name: int((np.asarray(img).sum(axis=2) > 60).sum())
        for name, img in crops.items()
    }
    buttons = [v for n, v in filled.items() if n.startswith("button")]
    if buttons and "stick" in filled:
        assert filled["stick"] > max(buttons), filled
