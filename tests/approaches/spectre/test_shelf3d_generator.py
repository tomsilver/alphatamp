"""Unit tests for the ShelfObstruct3D scene generator (pure-Python layout, no env
needed)."""

from alphatamp.approaches.spectre.envs.shelf3d import generator as G


def test_build_spec_layout() -> None:
    """The layout has the requested region counts, evenly-pitched centres, and
    obstructors at exactly the band offset from their free region."""
    spec = G.build_spec(seed=0, n_targets=2, n_free=3, n_obstructed=2)
    assert len(spec.target_region_y) == 2
    assert len(spec.free_region_y) == 3
    assert len(spec.obstructed_free) == 2
    # All region centres (targets then frees) are one contiguous row at the fixed pitch.
    row = spec.target_region_y + spec.free_region_y
    diffs = [round(b - a, 6) for a, b in zip(row, row[1:])]
    assert all(d == round(G._REGION_PITCH, 6) for d in diffs)
    # Each obstructor sits _BAND_OFFSET from its free region's centre.
    for k, i in enumerate(spec.obstructed_free):
        assert (
            abs(spec.obstructor_y[k] - (spec.free_region_y[i] + G._BAND_OFFSET)) < 1e-9
        )


def test_build_spec_deterministic() -> None:
    """Same seed -> identical layout; a different seed genuinely differs."""
    a = G.build_spec(3, 1, 2, 1)
    b = G.build_spec(3, 1, 2, 1)
    assert a == b
    assert G.build_spec(4, 1, 2, 1) != a


def test_build_task_config_structure() -> None:
    """The task config has the fixtures/regions/cubes/goal the model factory expects."""
    spec = G.build_spec(0, 2, 2, 1)
    cfg = G.build_task_config(spec)
    assert cfg["scene"] == "lab2"
    assert "cupboard_1" in cfg["fixtures"]["cupboard"]
    cubes = cfg["objects"]["cube"]
    for i in (1, 2):
        assert cubes[f"cube_blocker{i}"]["size"] == G._BLOCKER_SIZE
        assert cubes[f"cube_target{i}"]["size"] == G._TARGET_SIZE
        assert f"target_region_{i}" in cfg["regions"]
        assert f"blocker{i}_init_region" in cfg["regions"]
        assert ["on", f"cube_target{i}", f"target_region_{i}"] in cfg["goal_state"]
    for j in (1, 2):
        assert f"free_region_{j}" in cfg["regions"]
    # One obstructor cube per obstructed free region, blocker-sized (spawn-stable).
    n_obs = len(spec.obstructor_y)
    for k in range(1, n_obs + 1):
        assert cubes[f"cube_obstructor{k}"]["size"] == G._OBSTRUCTOR_SIZE
        assert f"obstructor{k}_init_region" in cfg["regions"]


def test_obstructor_pitch_isolation() -> None:
    """An obstructor lands in exactly one region's collision range, not its neighbour's
    -- the pitch must exceed the band offset plus the cube collision distance."""
    collision = 2 * G._OBSTRUCTOR_SIZE + 0.01  # two blocker-sized cubes + margin
    # Distance from an obstructor to the *next* region centre.
    to_neighbour = G._REGION_PITCH - G._BAND_OFFSET
    assert to_neighbour > collision
