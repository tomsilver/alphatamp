"""Tests for PrefixGenerator and oracle ranking."""

from __future__ import annotations

import torch
import pytest

from alphatamp.training.prefix_generator import (
    PrefixGenerator,
    PrefixStep,
    _compute_oracle_ranking,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def long_teacher_instance():
    """M=5, all applicable, Y=[0,0,0,1,0], T=[4,2,5,3,1].

    Teacher order by ascending T: [4(T=1), 1(T=2), 3(T=3), 0(T=4), 2(T=5)]
    Reveals: 4(fail), 1(fail), 3(success) → 3 steps.
    """
    M = 5
    return dict(
        applicability=torch.ones(M, dtype=torch.float32),
        success=torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32),
        steps_completed_fraction=torch.tensor([0.5, 0.3, 0.2, 1.0, 0.7], dtype=torch.float32),
        refinement_time=torch.tensor([4.0, 2.0, 5.0, 3.0, 1.0], dtype=torch.float32),
        lengths=torch.tensor([3.0, 5.0, 4.0, 6.0, 2.0], dtype=torch.float32),
    )


@pytest.fixture
def mixed_instance():
    """M=6, applicable={0,1,3,4}, inapplicable={2,5}.

    Y=[0,1,0,0,1,0], T=[2,5,0,3,1,0].
    Teacher order: [4(T=1), 0(T=2), 3(T=3), 1(T=5)]
    idx 4 has Y=1 → teacher stops after first reveal → 1 step.
    """
    M = 6
    return dict(
        applicability=torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.float32),
        success=torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.float32),
        steps_completed_fraction=torch.tensor([0.5, 1.0, 0.0, 0.3, 1.0, 0.0], dtype=torch.float32),
        refinement_time=torch.tensor([2.0, 5.0, 0.0, 3.0, 1.0, 0.0], dtype=torch.float32),
        lengths=torch.tensor([3.0, 5.0, 4.0, 6.0, 2.0, 3.0], dtype=torch.float32),
    )


@pytest.fixture
def all_fail_instance():
    """M=4, all applicable, no successes. T=[3,1,4,2].

    Teacher order: [1(T=1), 3(T=2), 0(T=3), 2(T=4)]
    No success → exhaustion → 5 steps (4 reveals + 1 terminal).
    """
    M = 4
    return dict(
        applicability=torch.ones(M, dtype=torch.float32),
        success=torch.zeros(M, dtype=torch.float32),
        steps_completed_fraction=torch.tensor([0.5, 0.3, 0.2, 0.7], dtype=torch.float32),
        refinement_time=torch.tensor([3.0, 1.0, 4.0, 2.0], dtype=torch.float32),
        lengths=torch.tensor([3.0, 5.0, 4.0, 6.0], dtype=torch.float32),
    )


# ===================================================================
# Oracle ranking tests
# ===================================================================


def test_oracle_ranking_successes_before_failures() -> None:
    """All Y=1 candidates get lower ranks than all Y=0 candidates."""
    y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
    t = torch.tensor([5, 3, 2, 1, 4], dtype=torch.float32)
    mask = torch.ones(5, dtype=torch.bool)

    ranking = _compute_oracle_ranking(y, t, mask)

    success_ranks = ranking[y > 0.5]
    failure_ranks = ranking[y < 0.5]
    assert success_ranks.max() < failure_ranks.min(), (
        f"Success ranks {success_ranks.tolist()} not all before "
        f"failure ranks {failure_ranks.tolist()}"
    )


def test_oracle_ranking_ascending_t_within_group() -> None:
    """Within successes and within failures, ranks follow ascending T."""
    y = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
    t = torch.tensor([5, 3, 1, 7, 3], dtype=torch.float32)
    mask = torch.ones(5, dtype=torch.bool)

    ranking = _compute_oracle_ranking(y, t, mask)

    # Successes at indices 0(T=5), 2(T=1), 4(T=3)
    # Ascending T: 2(T=1), 4(T=3), 0(T=5) → ranks 0, 1, 2
    assert ranking[2] < ranking[4] < ranking[0]

    # Failures at indices 1(T=3), 3(T=7)
    # Ascending T: 1(T=3), 3(T=7) → ranks 3, 4
    assert ranking[1] < ranking[3]
    assert ranking[1] > ranking[0]  # failures after successes


# ===================================================================
# First prefix / basic structure tests
# ===================================================================


def test_first_prefix_empty_history(mixed_instance) -> None:
    """First prefix step has |H|=0 — only inapplicable entries revealed."""
    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(**mixed_instance)

    step0 = steps[0]
    assert step0.step_index == 0

    app = mixed_instance["applicability"]
    inapplicable = app < 0.5

    # Only inapplicable positions should be revealed at step 0
    assert step0.revealed_mask[inapplicable].all(), "Inapplicable should be revealed"
    applicable = app > 0.5
    assert not step0.revealed_mask[applicable].any(), (
        "No applicable skeleton should be revealed at step 0"
    )

    # Revealed outcomes should be zero everywhere at step 0
    # (inapplicable have ground-truth 0 anyway)
    assert (step0.revealed_outcomes["y"] == 0).all()
    assert (step0.revealed_outcomes["f"] == 0).all()
    assert (step0.revealed_outcomes["t"] == 0).all()


def test_inapplicable_always_revealed(long_teacher_instance) -> None:
    """Every step has inapplicable entries in revealed_mask."""
    # Add some inapplicable entries
    inst = long_teacher_instance.copy()
    inst["applicability"] = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.float32)
    # Pad other tensors to M=6
    inst["success"] = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32)
    inst["steps_completed_fraction"] = torch.tensor(
        [0.5, 0.3, 0.0, 1.0, 0.7, 0.0], dtype=torch.float32
    )
    inst["refinement_time"] = torch.tensor(
        [4.0, 2.0, 0.0, 3.0, 1.0, 0.0], dtype=torch.float32
    )
    inst["lengths"] = torch.tensor(
        [3.0, 5.0, 4.0, 6.0, 2.0, 3.0], dtype=torch.float32
    )

    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(**inst)

    inapplicable = inst["applicability"] < 0.5
    for step in steps:
        assert step.revealed_mask[inapplicable].all(), (
            f"Step {step.step_index}: inapplicable not in revealed_mask"
        )


def test_all_inapplicable_instance() -> None:
    """All-inapplicable instance produces exactly 1 step with empty C_t."""
    M = 4
    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(
        applicability=torch.zeros(M, dtype=torch.float32),
        success=torch.zeros(M, dtype=torch.float32),
        steps_completed_fraction=torch.zeros(M, dtype=torch.float32),
        refinement_time=torch.zeros(M, dtype=torch.float32),
        lengths=torch.tensor([3.0, 5.0, 4.0, 6.0], dtype=torch.float32),
    )

    assert len(steps) == 1
    assert steps[0].revealed_mask.all(), "All should be revealed"
    assert (steps[0].oracle_ranking == -1).all(), "No candidates → all -1"
    assert steps[0].step_index == 0


# ===================================================================
# Teacher-forced mode tests
# ===================================================================


def test_teacher_forced_ascending_t_order(long_teacher_instance) -> None:
    """Teacher-forced reveals in ascending refinement_time order."""
    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(**long_teacher_instance)

    # Extract reveal order by diffing consecutive revealed_masks
    reveal_order = []
    for i in range(1, len(steps)):
        diff = steps[i].revealed_mask.long() - steps[i - 1].revealed_mask.long()
        newly_revealed = torch.where(diff > 0)[0]
        assert len(newly_revealed) == 1, "Exactly one reveal per step"
        reveal_order.append(newly_revealed[0].item())

    # The last reveal (success at idx 3) happens after the last emitted step
    # but before termination — infer it from the final state
    final_revealed = steps[-1].revealed_mask
    initial_revealed = steps[0].revealed_mask
    all_revealed_applicable = torch.where(final_revealed & ~initial_revealed)[0]

    # For this instance: teacher reveals 4(T=1), 1(T=2), 3(T=3)
    # Steps emitted: 3 (before each reveal). Last reveal (3, success) terminates.
    # reveal_order from diffs captures reveals between step 0→1 and 1→2: [4, 1]
    # The 3rd reveal (idx 3) terminates without emitting another step.
    # Total applicable revealed = all_revealed_applicable
    assert reveal_order == [4, 1], f"Expected [4, 1], got {reveal_order}"

    # Verify ascending T among all reveals
    t = long_teacher_instance["refinement_time"]
    all_reveals = reveal_order + [
        idx.item()
        for idx in all_revealed_applicable
        if idx.item() not in reveal_order
    ]
    for i in range(len(all_reveals) - 1):
        assert t[all_reveals[i]] <= t[all_reveals[i + 1]], (
            f"Not ascending T: {t[all_reveals[i]]} > {t[all_reveals[i + 1]]}"
        )


def test_teacher_forced_terminates_on_success(long_teacher_instance) -> None:
    """Teacher-forced stops after revealing Y=1.

    Instance: reveals 4(fail), 1(fail), 3(success). Expect 3 steps emitted
    (steps 0, 1, 2 — decision points before each reveal).
    """
    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(**long_teacher_instance)

    assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"

    # Step indices should be 0, 1, 2
    assert [s.step_index for s in steps] == [0, 1, 2]


def test_teacher_forced_terminates_on_exhaustion(all_fail_instance) -> None:
    """No successes → N_applicable + 1 steps (including terminal empty-C_t step)."""
    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(**all_fail_instance)

    n_applicable = int((all_fail_instance["applicability"] > 0.5).sum().item())
    expected = n_applicable + 1
    assert len(steps) == expected, f"Expected {expected} steps, got {len(steps)}"

    # Last step should have empty candidate set
    last = steps[-1]
    applicable_mask = all_fail_instance["applicability"] > 0.5
    candidate_mask = applicable_mask & ~last.revealed_mask
    assert not candidate_mask.any(), "Last step should have empty C_t"


# ===================================================================
# Epsilon-random mode tests
# ===================================================================


def test_epsilon_random_deviates_from_teacher(all_fail_instance) -> None:
    """With epsilon=1.0, reveal order differs from teacher at least once.

    Uses no-success instance (M=4) to get a long sequence. With 4 applicable
    skeletons and full random selection, the probability of matching the exact
    teacher order is 1/4! = 1/24, so over 50 trials we will almost surely
    see a deviation.
    """
    gen_teacher = PrefixGenerator(mode="teacher_forced")
    teacher_steps = gen_teacher.generate(**all_fail_instance)

    teacher_reveals = []
    for i in range(1, len(teacher_steps)):
        diff = teacher_steps[i].revealed_mask.long() - teacher_steps[i - 1].revealed_mask.long()
        teacher_reveals.append(torch.where(diff > 0)[0][0].item())

    gen_random = PrefixGenerator(mode="epsilon_random", epsilon=1.0)
    found_deviation = False

    for seed in range(50):
        rng = torch.Generator().manual_seed(seed)
        random_steps = gen_random.generate(**all_fail_instance, rng=rng)

        random_reveals = []
        for i in range(1, len(random_steps)):
            diff = random_steps[i].revealed_mask.long() - random_steps[i - 1].revealed_mask.long()
            random_reveals.append(torch.where(diff > 0)[0][0].item())

        if random_reveals != teacher_reveals:
            found_deviation = True
            break

    assert found_deviation, "epsilon=1.0 should deviate from teacher order"


# ===================================================================
# On-policy mode tests
# ===================================================================


def test_on_policy_random_model_like_epsilon_1(all_fail_instance) -> None:
    """on_policy with random scores behaves like epsilon_random with epsilon=1.

    Both produce valid prefix sequences of the same length (since the instance
    has no successes, both exhaust all applicable → N_app + 1 steps).
    Structural invariants (termination, mask correctness) are identical.
    """
    M = all_fail_instance["applicability"].shape[0]
    n_applicable = int((all_fail_instance["applicability"] > 0.5).sum().item())
    expected_steps = n_applicable + 1

    # on_policy with random scoring
    rng_model = torch.Generator().manual_seed(99)

    def random_score_fn(step: PrefixStep) -> torch.Tensor:
        return torch.rand(M, generator=rng_model)

    gen_on_policy = PrefixGenerator(mode="on_policy")
    steps_on_policy = gen_on_policy.generate(
        **all_fail_instance, score_fn=random_score_fn,
    )

    assert len(steps_on_policy) == expected_steps, (
        f"on_policy: expected {expected_steps} steps, got {len(steps_on_policy)}"
    )

    # epsilon_random with epsilon=1.0
    gen_eps = PrefixGenerator(mode="epsilon_random", epsilon=1.0)
    rng_eps = torch.Generator().manual_seed(42)
    steps_eps = gen_eps.generate(**all_fail_instance, rng=rng_eps)

    assert len(steps_eps) == expected_steps, (
        f"epsilon=1: expected {expected_steps} steps, got {len(steps_eps)}"
    )

    # Both should have identical structural properties
    for steps, label in [(steps_on_policy, "on_policy"), (steps_eps, "eps=1")]:
        # First step: empty history
        assert not (steps[0].revealed_mask & (all_fail_instance["applicability"] > 0.5)).any(), (
            f"{label}: step 0 should have no applicable revealed"
        )
        # Last step: all revealed
        assert steps[-1].revealed_mask.all() or not (
            (all_fail_instance["applicability"] > 0.5) & ~steps[-1].revealed_mask
        ).any(), f"{label}: last step should have all applicable revealed"
        # Step indices sequential
        assert [s.step_index for s in steps] == list(range(expected_steps)), (
            f"{label}: step indices not sequential"
        )


# ===================================================================
# Regression: PrefixStep → TokenBuilder compatibility
# ===================================================================


def test_prefix_step_compatible_with_token_builder() -> None:
    """PrefixStep fields unpack into TokenBuilder.forward without error."""
    from alphatamp.models.token_builder import TokenBuilder

    d_skel, d_out = 16, 8
    tb = TokenBuilder(d_skel=d_skel, d_out=d_out, dropout=0.0)
    tb.eval()

    M = 6
    gen = PrefixGenerator(mode="teacher_forced")
    steps = gen.generate(
        applicability=torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.float32),
        success=torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.float32),
        steps_completed_fraction=torch.tensor([0.5, 1.0, 0.0, 0.3, 1.0, 0.0]),
        refinement_time=torch.tensor([2.0, 5.0, 0.0, 3.0, 1.0, 0.0]),
        lengths=torch.tensor([3.0, 5.0, 4.0, 6.0, 2.0, 3.0]),
    )

    step = steps[0]
    skeleton_embeds = torch.randn(1, M, d_skel)

    with torch.no_grad():
        tokens = tb(
            skeleton_embeds=skeleton_embeds,
            applicability=step.applicability.unsqueeze(0),
            revealed_mask=step.revealed_mask.unsqueeze(0),
            y=step.revealed_outcomes["y"].unsqueeze(0),
            f=step.revealed_outcomes["f"].unsqueeze(0),
            t=step.revealed_outcomes["t"].unsqueeze(0),
            lengths=step.lengths.unsqueeze(0),
        )

    assert tokens.shape == (1, M, d_skel + d_out)
    assert tokens.dtype == torch.float32
