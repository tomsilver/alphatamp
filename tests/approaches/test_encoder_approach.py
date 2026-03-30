"""Tests for EncoderApproach phase-1 vocabulary behavior."""

import kinder
import pytest
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.encoder_approach import EncoderApproach


def test_encoder_approach_builds_vocabulary() -> None:
    """build_vocab should build a non-empty top-k vocabulary and count table."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5,
        vocabulary_size=3,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    vocab = approach.build_vocab(seed_ids=[101, 102, 103], k=3)

    counts = approach.get_skeleton_counts()

    assert len(vocab) > 0
    assert len(vocab) <= 3
    assert set(vocab).issubset(set(counts))
    assert all(count > 0 for count in counts.values())

    env.close()  # type: ignore[no-untyped-call]


def test_encoder_approach_rebuilds_counts_from_seed_list() -> None:
    """build_vocab should rebuild counts from the provided seed list."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5,
        vocabulary_size=5,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    approach.build_vocab(seed_ids=[101, 102], k=5)
    counts_before = approach.get_skeleton_counts()

    approach.build_vocab(seed_ids=[101], k=5)
    counts_after = approach.get_skeleton_counts()

    assert counts_before
    assert counts_after
    assert counts_before != counts_after

    env.close()  # type: ignore[no-untyped-call]


@pytest.mark.slow
def test_encoder_approach_planning_uses_vocabulary() -> None:
    """Planning should attempt only vocabulary skeletons."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=3,
        training_planning_timeout=3,
        vocabulary_size=2,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    obs, _ = env.reset(seed=101)
    approach.build_vocab(seed_ids=[101, 102, 103], k=2)

    tried: list[tuple[tuple, tuple]] = []

    def _fake_refiner(x0, skel_states, skel_ops, timeout, bpg):
        del x0, timeout, bpg
        tried.append((tuple(skel_states), tuple(skel_ops)))
        return None

    approach._refiner = _fake_refiner  # pylint: disable=protected-access

    with pytest.raises(TimeoutError):
        approach.run_planning(obs, timeout=1)

    vocab_set = set(approach.get_skeleton_vocabulary())
    assert tried
    assert all(skeleton in vocab_set for skeleton in tried)

    env.close()  # type: ignore[no-untyped-call]


def test_encoder_approach_reconstructs_abstract_state_sequence() -> None:
    """Reconstruction returns state sequence for applicable ops and None on mismatch."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5,
        vocabulary_size=5,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    obs, _ = env.reset(seed=101)
    problem = approach._observation_to_planning_problem(
        obs
    )  # pylint: disable=protected-access
    x0 = problem.initial_state
    s0 = env_models.state_abstractor(x0)

    bpg = BilevelPlanningGraph()
    bpg.add_state_node(x0)
    bpg.add_abstract_state_node(s0)
    bpg.add_state_abstractor_edge(x0, s0)

    skeleton = next(
        approach._base_abstract_plan_generator(  # pylint: disable=protected-access
            x0,
            s0,
            problem.goal,
            5.0,
            bpg,
        )
    )

    op_sequence = tuple(skeleton[1])
    reconstructed = approach.reconstruct_abstract_state_sequence(s0, op_sequence)
    assert reconstructed is not None
    assert len(reconstructed) == len(op_sequence) + 1

    # Deliberately make sequence inapplicable by repeating first op immediately.
    mismatched_sequence = (op_sequence[0],) + op_sequence
    mismatched = approach.reconstruct_abstract_state_sequence(s0, mismatched_sequence)
    assert mismatched is None

    env.close()  # type: ignore[no-untyped-call]


def test_encoder_approach_build_dataset_semantics() -> None:
    """Inapplicable entries should skip refinement and get timeout runtime."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5.0,
        vocabulary_size=2,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    vocab = approach.build_vocab(seed_ids=[101, 102], k=1)
    assert vocab
    valid_sequence = vocab[0]
    invalid_sequence = valid_sequence + (valid_sequence[0],)
    approach._op_sequence_vocab = [
        valid_sequence,
        invalid_sequence,
    ]  # pylint: disable=protected-access

    original_reconstruct = approach.reconstruct_abstract_state_sequence

    def _fake_reconstruct(initial_abstract_state, op_sequence):
        if op_sequence == invalid_sequence:
            return None
        return original_reconstruct(initial_abstract_state, op_sequence)

    approach.reconstruct_abstract_state_sequence = _fake_reconstruct  # type: ignore[method-assign]
    approach._refiner = (
        lambda *args, **kwargs: object()
    )  # pylint: disable=protected-access,unnecessary-lambda-assignment

    dataset = approach.build_dataset(seed_ids=[101])
    applicability = dataset["applicability"]
    success = dataset["success"]
    refinement_time = dataset["refinement_time"]

    assert applicability.shape == (1, 2)
    assert success.shape == (1, 2)
    assert refinement_time.shape == (1, 2)

    assert applicability[0, 0] == 1.0
    assert success[0, 0] == 1.0
    assert (
        0.0 <= refinement_time[0, 0] <= approach._training_planning_timeout
    )  # pylint: disable=protected-access

    assert applicability[0, 1] == 0.0
    assert success[0, 1] == 0.0
    # Inapplicable entries are skipped entirely; no time is spent on them.
    assert refinement_time[0, 1] == 0.0  # pylint: disable=protected-access

    env.close()  # type: ignore[no-untyped-call]


def test_encoder_approach_cycle_pruning_defaults_on() -> None:
    """Cycle pruning should drop skeletons with repeated abstract states by default."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5.0,
        vocabulary_size=5,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    cyclic_sequence = ["s0", "s1", "s0"]
    acyclic_sequence = ["s0", "s1", "s2"]

    assert approach._abstract_state_sequence_has_cycle(
        cyclic_sequence
    )  # pylint: disable=protected-access
    assert not approach._abstract_state_sequence_has_cycle(
        acyclic_sequence
    )  # pylint: disable=protected-access

    env.close()  # type: ignore[no-untyped-call]


def test_encoder_approach_build_vocab_accepts_prune_cycles_override() -> None:
    """Method-level prune_cycles override should be accepted and preserved."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5.0,
        vocabulary_size=3,
        env_id="kinder/Obstruction2D-o1-v0",
        prune_cycles=False,
    )

    vocab = approach.build_vocab(seed_ids=[101, 102], k=3, prune_cycles=True)
    assert isinstance(vocab, list)
    assert approach._prune_cycles is False  # pylint: disable=protected-access

    env.close()  # type: ignore[no-untyped-call]


# ---------------------------------------------------------------------------
# Pure-numpy tests for filter_vocab_by_success_rate / apply_vocab_filter_to_dataset
# (No environment or planner required.)
# ---------------------------------------------------------------------------


def _make_filter_dataset(
    applicability: list[list[float]],
    success: list[list[float]],
    vocab: list | None = None,
) -> dict:
    """Build a minimal dataset dict for filtering tests."""
    import numpy as np

    app = np.array(applicability, dtype=np.float32)
    suc = np.array(success, dtype=np.float32)
    n_seeds, n_vocab = app.shape
    if vocab is None:
        # Synthetic hashable vocab entries.
        vocab = [("op_" + str(i),) for i in range(n_vocab)]
    return {
        "seed_ids": list(range(n_seeds)),
        "op_sequence_vocab": vocab,
        "applicability": app,
        "success": suc,
        "refinement_time": np.ones_like(app),
    }


def test_filter_vocab_removes_never_applicable() -> None:
    """Sequences that are never applicable should always be removed
    (success_rate=NaN)."""
    # Col 0: always applicable, always fails.
    # Col 1: never applicable.
    # Col 2: always applicable, always succeeds.
    dataset = _make_filter_dataset(
        applicability=[[1, 0, 1], [1, 0, 1]],
        success=[[0, 0, 1], [0, 0, 1]],
    )
    filtered_vocab, keep_indices, stats = EncoderApproach.filter_vocab_by_success_rate(
        dataset, threshold=0.0
    )
    # Col 1 (never applicable) and Col 0 (always fails) are both removed at threshold=0.0.
    assert 1 not in keep_indices
    assert stats["original_size"] == 3
    assert stats["filtered_size"] == 1
    assert keep_indices == [2]
    assert filtered_vocab == [dataset["op_sequence_vocab"][2]]


def test_filter_vocab_strict_removes_always_fail() -> None:
    """threshold=0.0 should remove exactly the sequences with zero successes."""
    # Col 0: 2/4 applicable, 0 success  → always-fail applicable → removed
    # Col 1: 4/4 applicable, 1 success  → success_rate=0.25     → kept
    # Col 2: 3/4 applicable, 3 success  → success_rate=1.0      → kept
    dataset = _make_filter_dataset(
        applicability=[[1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0]],
        success=[[0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 0, 1]],
    )
    _, keep_indices, stats = EncoderApproach.filter_vocab_by_success_rate(
        dataset, threshold=0.0
    )
    assert 0 not in keep_indices
    assert 1 in keep_indices
    assert 2 in keep_indices
    assert stats["removed_count"] == 1


def test_filter_vocab_threshold_removes_low_success_rate() -> None:
    """threshold=0.5 should additionally remove sequences with success_rate < 0.5."""
    # Col 0: applicable=2, success=0  → rate=0.0   → removed
    # Col 1: applicable=4, success=1  → rate=0.25  → removed at threshold=0.5
    # Col 2: applicable=4, success=3  → rate=0.75  → kept
    dataset = _make_filter_dataset(
        applicability=[[1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]],
        success=[[0, 0, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]],
    )
    _, keep_indices, stats = EncoderApproach.filter_vocab_by_success_rate(
        dataset, threshold=0.5
    )
    assert keep_indices == [2]
    assert stats["filtered_size"] == 1


def test_filter_vocab_ranked_by_success_rate_descending() -> None:
    """Returned vocab should be ordered by descending success_rate."""
    # Col 0: rate=0.25, Col 1: rate=1.0, Col 2: rate=0.5
    dataset = _make_filter_dataset(
        applicability=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        success=[[1, 1, 1], [0, 1, 1], [0, 1, 0], [0, 1, 0]],
    )
    _, keep_indices, _ = EncoderApproach.filter_vocab_by_success_rate(
        dataset, threshold=0.0
    )
    import numpy as np

    app = dataset["applicability"]
    suc = dataset["success"]
    rates = (suc.sum(axis=0) / app.sum(axis=0)).tolist()
    # Verify indices are in descending rate order.
    assert rates[keep_indices[0]] >= rates[keep_indices[1]] >= rates[keep_indices[2]]


def test_filter_vocab_invalid_threshold_raises() -> None:
    """Threshold outside [0, 1] should raise ValueError."""
    dataset = _make_filter_dataset(applicability=[[1]], success=[[1]])
    with pytest.raises(ValueError, match="threshold must be in"):
        EncoderApproach.filter_vocab_by_success_rate(dataset, threshold=1.5)
    with pytest.raises(ValueError, match="threshold must be in"):
        EncoderApproach.filter_vocab_by_success_rate(dataset, threshold=-0.1)


def test_apply_vocab_filter_slices_matrices_correctly() -> None:
    """apply_vocab_filter_to_dataset should return correctly sliced matrices."""
    import numpy as np

    dataset = _make_filter_dataset(
        applicability=[[1, 0, 1], [0, 1, 1]],
        success=[[1, 0, 0], [0, 1, 1]],
    )
    keep_indices = [2, 0]  # reordered — col 2 first, then col 0
    filtered = EncoderApproach.apply_vocab_filter_to_dataset(dataset, keep_indices)

    assert filtered["applicability"].shape == (2, 2)
    assert filtered["success"].shape == (2, 2)
    assert filtered["refinement_time"].shape == (2, 2)

    # Column order must follow keep_indices.
    np.testing.assert_array_equal(
        filtered["applicability"], dataset["applicability"][:, keep_indices]
    )
    np.testing.assert_array_equal(
        filtered["success"], dataset["success"][:, keep_indices]
    )
    assert filtered["op_sequence_vocab"] == [
        dataset["op_sequence_vocab"][i] for i in keep_indices
    ]
    assert filtered["seed_ids"] == dataset["seed_ids"]


def test_apply_vocab_filter_propagates_per_seed_fields() -> None:
    """Per-seed list fields should be copied unchanged."""
    dataset = _make_filter_dataset(
        applicability=[[1, 1], [1, 1]],
        success=[[1, 0], [0, 1]],
    )
    dataset["initial_low_level_states"] = ["state_a", "state_b"]
    dataset["problem_goals"] = ["goal_a", "goal_b"]

    filtered = EncoderApproach.apply_vocab_filter_to_dataset(dataset, [0])
    assert filtered["initial_low_level_states"] == ["state_a", "state_b"]
    assert filtered["problem_goals"] == ["goal_a", "goal_b"]


def test_apply_vocab_filter_empty_keep_indices_raises() -> None:
    """Empty keep_indices should raise ValueError."""
    dataset = _make_filter_dataset(applicability=[[1]], success=[[1]])
    with pytest.raises(ValueError, match="keep_indices must be non-empty"):
        EncoderApproach.apply_vocab_filter_to_dataset(dataset, [])


# ---------------------------------------------------------------------------
# steps_completed_fraction invariants
# ---------------------------------------------------------------------------


def test_build_dataset_steps_completed_fraction_invariants() -> None:
    """steps_completed_fraction == 1.0 iff success == 1.0; all values in [0, 1].

    Uses a mocked refiner that cycles through three behaviours per call so that we
    exercise full success (fraction=1.0), partial failure (fraction=0.5), and complete
    failure (fraction=0.0) without running any real simulation.
    """
    import numpy as np

    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    approach = EncoderApproach(
        env_models,
        seed=123,
        num_training_skeletons_per_problem=5,
        training_planning_timeout=5.0,
        vocabulary_size=1,
        env_id="kinder/Obstruction2D-o1-v0",
    )

    vocab = approach.build_vocab(seed_ids=[101, 102], k=1)
    assert vocab, "Expected at least one vocab entry from build_vocab"

    # Sentinel strings used as fake abstract states — they are hashable and
    # distinct from any real AbstractState object.
    FAKE_S1 = "__fake_s1__"
    FAKE_S2 = "__fake_s2__"

    # Always return an applicable 2-step abstract state sequence.
    # abstract_state_sequence[0] is the initial state (not counted); we use a
    # dummy string there too — the only states that matter for step-counting are
    # abstract_state_sequence[1:] = [FAKE_S1, FAKE_S2].
    approach.reconstruct_abstract_state_sequence = (  # type: ignore[method-assign]
        lambda _s0, _ops: ["__fake_s0__", FAKE_S1, FAKE_S2]
    )

    call_index = [0]

    def fake_refiner(_x0, _abstract_state_seq, _op_seq, _timeout, bpg):
        idx = call_index[0]
        call_index[0] += 1
        if idx % 3 == 0:
            # Full success: reach both steps.
            bpg.add_abstract_state_node(FAKE_S1)
            bpg.add_abstract_state_node(FAKE_S2)
            return object()  # non-None → success
        if idx % 3 == 1:
            # Partial failure: only the first step reached (fraction = 0.5).
            bpg.add_abstract_state_node(FAKE_S1)
            return None
        # Complete failure: no steps reached (fraction = 0.0).
        return None

    approach._refiner = fake_refiner  # pylint: disable=protected-access

    # 3 seeds × 1 vocab entry = exactly 3 refiner calls, one per behaviour.
    dataset = approach.build_dataset(seed_ids=[101, 102, 103], show_progress=False)

    steps = dataset["steps_completed_fraction"]
    success = dataset["success"]
    applicability = dataset["applicability"]

    assert steps.shape == (3, 1)

    # Invariant 1: all values lie within [0, 1].
    assert np.all(steps >= 0.0), f"Negative step fraction found:\n{steps}"
    assert np.all(steps <= 1.0), f"Step fraction > 1 found:\n{steps}"

    # Invariant 2: for applicable entries, steps==1.0 iff success==1.0.
    app_mask = applicability > 0.5
    np.testing.assert_array_equal(
        steps[app_mask] == 1.0,
        success[app_mask] == 1.0,
        err_msg="steps_completed_fraction==1.0 must match success==1.0 for applicable entries",
    )

    # Verify the three specific values produced by the cycling mock.
    assert steps[0, 0] == pytest.approx(
        1.0
    ), f"seed 0: expected 1.0 (success), got {steps[0, 0]}"
    assert steps[1, 0] == pytest.approx(
        0.5
    ), f"seed 1: expected 0.5 (partial), got {steps[1, 0]}"
    assert steps[2, 0] == pytest.approx(
        0.0
    ), f"seed 2: expected 0.0 (full fail), got {steps[2, 0]}"

    env.close()  # type: ignore[no-untyped-call]
