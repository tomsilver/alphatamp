"""The StickButton2D instrumentation changed no label.

``AcceptanceTrajectorySampler`` does not subclass a hook — it **re-implements**
upstream's ``ParameterizedControllerTrajectorySampler.__call__`` loop, because
upstream computes the achieved abstract state to decide accept-or-reject and then
discards it behind a payload-free ``TrajectorySamplingFailure``. Without that state
there is no class-2 evidence and ``coverage``/``waste`` are identically zero on this
environment.

Re-implementing the loop is the risk that buys it, and the whole collection runs through
it, so "faithful" has to be a measurement rather than a claim. This is the differential
check: the same problems, the same candidates, the same per-candidate seeds, refined once
with upstream's sampler and once with ours, asserting **identical labels**.

Both samplers draw from the refiner's own ``rng``, so any extra or reordered draw would
desynchronise the streams and show up as a label divergence — which is exactly what makes
a same-seed differential the right instrument here.
"""

from __future__ import annotations

import itertools

import pytest

pytestmark = pytest.mark.slow

_PROBLEMS = 3
_CANDIDATES = 8
_TIMEOUT_S = 20.0
_SAMPLES_PER_STEP = 5
_MAX_TRAJECTORY_STEPS = 200


def _refine_pool(num_buttons: int, problem_id: int, instrumented: bool) -> list[bool]:
    """Labels for the first ``_CANDIDATES`` skeletons, under one sampler
    implementation."""
    # pylint: disable=import-outside-toplevel
    import kinder
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
    from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
    from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
        ParameterizedControllerTrajectorySampler,
    )
    from bilevel_planning.utils import RelationalControllerGenerator
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (
        make_plan_generator,
    )
    from alphatamp.approaches.spectre.envs.stickbutton2d.instrumented_refiner import (
        RecordingSampler,
    )

    register_extra_envs()
    env = kinder.make(f"kinder/StickButton2D-b{num_buttons}-v0")
    try:
        obs, _ = env.reset(seed=problem_id)
        models = create_bilevel_planning_models(
            "stickbutton2d",
            env.observation_space,
            env.action_space,
            num_buttons=num_buttons,
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        bpg = BilevelPlanningGraph()  # type: ignore[var-annotated]
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        generator = make_plan_generator(models, x0, seed=problem_id)
        pool = list(itertools.islice(generator(x0, s0, goal, 60.0, bpg), _CANDIDATES))

        kwargs = {
            "controller_generator": RelationalControllerGenerator(models.skills),
            "transition_function": models.transition_fn,
            "state_abstractor": models.state_abstractor,
            "max_trajectory_steps": _MAX_TRAJECTORY_STEPS,
        }
        sampler = (
            RecordingSampler(**kwargs)
            if instrumented
            else ParameterizedControllerTrajectorySampler(
                **kwargs  # type: ignore[arg-type]
            )
        )

        labels = []
        for idx, (state_plan, action_plan) in enumerate(pool):
            if instrumented:
                sampler.clear()  # type: ignore[attr-defined]
            refiner = BacktrackingRefiner(
                trajectory_sampler=sampler,
                num_sampling_attempts_per_step=_SAMPLES_PER_STEP,
                # The per-candidate seed is what makes the two runs comparable at all:
                # both samplers consume the same rng, so identical seeds mean identical
                # parameter draws unless one of them draws differently.
                seed=idx,
            )
            try:
                plan = refiner(x0, state_plan, action_plan, _TIMEOUT_S, bpg)
                labels.append(plan is not None)
            except BaseException:  # pylint: disable=broad-exception-caught
                labels.append(False)
        return labels
    finally:
        env.close()


@pytest.mark.parametrize("num_buttons", [2, 3])
def test_recording_sampler_reproduces_upstream_labels(num_buttons: int) -> None:
    """Instrumented and stock refinement agree candidate for candidate."""
    for problem_id in range(_PROBLEMS):
        stock = _refine_pool(num_buttons, problem_id, instrumented=False)
        instrumented = _refine_pool(num_buttons, problem_id, instrumented=True)
        assert instrumented == stock, (
            f"b{num_buttons} problem {problem_id}: instrumentation changed labels\n"
            f"  stock        {stock}\n  instrumented {instrumented}"
        )


def test_recording_sampler_actually_recorded_something() -> None:
    """Guard against the test above passing because nothing was instrumented at all.

    If ``RecordingSampler`` silently stopped capturing rejections, every label would
    still match and the differential above would be vacuous — while the collection
    quietly produced episodes with no evidence in them.
    """
    # pylint: disable=import-outside-toplevel
    import kinder
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
    from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
    from bilevel_planning.utils import RelationalControllerGenerator
    from kinder_bilevel_planning.env_models import create_bilevel_planning_models

    from alphatamp.approaches.spectre.env_registry import register_extra_envs
    from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (
        make_plan_generator,
    )
    from alphatamp.approaches.spectre.envs.stickbutton2d.instrumented_refiner import (
        RecordingSampler,
        failure_metadata,
    )

    register_extra_envs()
    env = kinder.make("kinder/StickButton2D-b3-v0")
    try:
        obs, _ = env.reset(seed=0)
        models = create_bilevel_planning_models(
            "stickbutton2d", env.observation_space, env.action_space, num_buttons=3
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)
        bpg = BilevelPlanningGraph()  # type: ignore[var-annotated]
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        generator = make_plan_generator(models, x0, seed=0)
        pool = list(itertools.islice(generator(x0, s0, goal, 60.0, bpg), _CANDIDATES))

        sampler = RecordingSampler(
            controller_generator=RelationalControllerGenerator(models.skills),
            transition_function=models.transition_fn,
            state_abstractor=models.state_abstractor,
            max_trajectory_steps=_MAX_TRAJECTORY_STEPS,
        )
        harvested = []
        for idx, (state_plan, action_plan) in enumerate(pool):
            sampler.clear()
            refiner = BacktrackingRefiner(
                trajectory_sampler=sampler,
                num_sampling_attempts_per_step=_SAMPLES_PER_STEP,
                seed=idx,
            )
            try:
                ok = refiner(x0, state_plan, action_plan, _TIMEOUT_S, bpg) is not None
            except BaseException:  # pylint: disable=broad-exception-caught
                ok = False
            if not ok:
                harvested += failure_metadata(
                    sampler, action_plan, _SAMPLES_PER_STEP, budget_exhausted=False
                )

        assert harvested, "no failure was captured — instrumentation is inert"
        entry = harvested[0]
        assert entry["schema"] and entry["n_step"] > 0
        # Class 2 is the only channel this environment has; a record with neither a
        # deviation nor a culprit would carry no evidence at all.
        assert "dev_added" in entry and "dev_deleted" in entry
    finally:
        env.close()
