"""Tests for the concave-grasp demo's diagnostics.

The demo asserts something about the gripper model on screen -- that a finger can close
onto a concavity rather than material -- so the measurement behind that claim is pinned
here, independently of any rendering.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from alphatamp.approaches.spectre.envs.dd2d.dd2d.demo_grasp_concave import (
    CONCAVE_FAMILIES,
    contact_runs,
    finger_contacts,
    finger_gaps,
    graspable_demo_scene,
    pad_to_common,
    select_cells,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.grasps import (
    FINGER_WIDTH,
    grasp_cells,
    grasp_cfree,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.shapes import sample_shape

_TOL = 1e-6


@pytest.mark.parametrize("family", ["box", "can", "pillcase", "bowl"])
def test_convex_families_never_float_a_finger(family):
    """On a convex footprint the supporting-line construction is exact: the extreme-x
    line touches along one connected run, so every finger lands on material."""
    rng = random.Random(11)
    for _ in range(4):
        shape = sample_shape(rng, family=family)
        for g in grasp_cells(shape):
            left, right = finger_gaps(shape, g)
            assert left <= _TOL and right <= _TOL, (family, g.alpha, left, right)


def test_no_grasp_cell_floats_on_concave_families():
    """The fixed grasp model's guarantee, on the families that used to break it: a finger
    never closes across the C-opening / waist.

    (This inverts the old assertion that such floating cells *exist* -- that was the bug.)
    Scanned over seeds because it is a property of the family, not of one sample.
    """
    for fam in CONCAVE_FAMILIES:
        for seed in range(8):
            shape = sample_shape(random.Random(seed), family=fam)
            for g in grasp_cells(shape):
                assert max(finger_gaps(shape, g)) <= _TOL, (fam, seed, g.alpha)


def test_contact_length_is_bounded():
    for fam in CONCAVE_FAMILIES:
        shape = sample_shape(random.Random(3), family=fam)
        for g in grasp_cells(shape):
            gaps = finger_gaps(shape, g)
            contacts = finger_contacts(shape, g)
            for gap, touched in zip(gaps, contacts):
                assert -_TOL <= touched <= FINGER_WIDTH + _TOL
                # no returned cell floats, so contact is never zero-because-of-a-gap
                assert gap <= _TOL


def test_every_admissible_cell_has_a_contact_run_on_both_lines():
    """``direction_admissible`` only keeps directions whose supporting lines both touch
    the footprint, so a run must exist even where the *finger* misses it."""
    for fam in CONCAVE_FAMILIES:
        shape = sample_shape(random.Random(5), family=fam)
        for g in grasp_cells(shape):
            assert contact_runs(shape, g, g.xmin)
            assert contact_runs(shape, g, g.xmax)


def test_select_cells_caps_count():
    shape = sample_shape(random.Random(0), family="dumbbell")
    cells = grasp_cells(shape)
    if len(cells) < 3:  # pragma: no cover - depends on the sampled dimensions
        pytest.skip("sampled dumbbell admits too few cells to exercise the cap")
    chosen = select_cells(shape, cells, max_cells=2)
    assert len(chosen) == 2
    assert all(g in cells for g in chosen)


def test_pad_to_common_letterboxes_to_one_size():
    frames = [
        np.zeros((10, 8, 3), dtype=np.uint8),
        np.zeros((6, 12, 3), dtype=np.uint8),
    ]
    padded = pad_to_common(frames)
    assert {f.shape for f in padded} == {(10, 12, 3)}


@pytest.mark.parametrize("family", list(CONCAVE_FAMILIES))
def test_demo_scene_ends_up_graspable(family):
    """The density ladder must actually reach a graspable target, or the clutter clip
    never shows a pick."""
    rng = random.Random(2)
    shape = sample_shape(rng, family=family)
    scene = graspable_demo_scene(shape, rng)
    pose = scene.items["target"].pose
    obstacles = [st.footprint() for n, st in scene.items.items() if n != "target"]
    obstacles.append(scene.wall_band)
    assert any(grasp_cfree(g, pose, obstacles) for g in grasp_cells(shape))
