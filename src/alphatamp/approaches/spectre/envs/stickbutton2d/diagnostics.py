"""Per-button achievability probe for StickButton2D.

**Why per-button and not per-skeleton.** StickButton2D's goal is a static conjunction
over *every* button (``kinder_bilevel_planning/.../stickbutton2d.py:150-157``). So a
single button that no controller can press makes *every* skeleton in the pool infeasible,
and episode feasibility falls roughly as the product of the per-button achievable rates.
Measuring at the skeleton level cannot distinguish "the ordering was unlucky" from "this
scene is impossible", and costs ``K_max`` refinements to say so; measuring per button
costs ``2N`` and answers directly.

Each button is probed on the two routes that can press it, from the problem's own initial
state:

- **robot route** — ``RobotPressButtonFromNothing(robot, button)``
- **stick route** — ``PickStickFromNothing(robot, stick)`` then
  ``StickPressButtonFromNothing(robot, stick, button)``

Both are genuine one/two-step abstract plans handed to the real
:class:`BacktrackingRefiner`, so a route "works" here exactly when it would work inside a
full skeleton. The probe is read-only with respect to the environment.

**``predicted_solvable`` is a diagnostic, NOT a bound in either direction.** Measured
against ground truth (`full` mode) it errs both ways: it called b2 55% / b3 35% where
the true rate is 100%, and (under ``acceptance="superset"``) b5 75% where it is 0%.

It under-estimates because probing from ``x0`` tries only one route to each button,
while a real skeleton may reach the same button from a different predecessor state — via
``RobotPressButtonFromButton``, with a different approach path — or press it incidentally
in passing. It over-estimates because a real skeleton must *chain* the presses, and each
extra step is another chance to fail.

So use this to attribute *why* refinement fails (out-of-reach vs a stuck controller vs
extra atoms), which is what it is good at, and use `full` mode to decide whether a
variant is collectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.structs import RelationalAbstractState, SesameModels
from bilevel_planning.utils import RelationalControllerGenerator
from relational_structs import GroundOperator, LiftedOperator, Object

from alphatamp.approaches.spectre.envs.stickbutton2d.geometry import (
    ButtonReach,
    classify_buttons,
)
from alphatamp.approaches.spectre.envs.stickbutton2d.sampler import (
    Acceptance,
    AcceptanceTrajectorySampler,
)

FailureMode = Literal[
    "ok",
    "out_of_reach",
    "effect_not_achieved",
    "extra_atoms",
    "transition_error",
]

_ROBOT_PRESS = "RobotPressButtonFromNothing"
_PICK_STICK = "PickStickFromNothing"
_STICK_PRESS = "StickPressButtonFromNothing"


@dataclass(frozen=True)
class RouteProbe:
    """Outcome of trying one route (robot or stick) to one button."""

    achieved: bool
    mode: FailureMode
    missing: tuple[str, ...]
    extra: tuple[str, ...]


@dataclass(frozen=True)
class ButtonProbe:
    """Per-button result: can *any* route press it, and if not, why."""

    button: str
    needs_stick: bool
    robot: RouteProbe
    stick: RouteProbe

    @property
    def achievable(self) -> bool:
        """True if at least one route presses this button from the initial state."""
        return self.robot.achieved or self.stick.achieved


@dataclass(frozen=True)
class ProblemProbe:
    """Per-problem roll-up.

    ``predicted_solvable`` is a diagnostic, not a bound (see module doc).
    """

    problem_id: int
    num_buttons: int
    buttons: tuple[ButtonProbe, ...]
    reach_max_y: float

    @property
    def num_achievable(self) -> int:
        """How many buttons have at least one working route."""
        return sum(1 for b in self.buttons if b.achievable)

    @property
    def predicted_solvable(self) -> bool:
        """Every button reachable by some route — necessary for any skeleton to work."""
        return self.num_achievable == self.num_buttons

    def blockers(self) -> tuple[ButtonProbe, ...]:
        """The buttons that no route can press."""
        return tuple(b for b in self.buttons if not b.achievable)


def _operators_by_name(env_models: SesameModels) -> dict[str, LiftedOperator]:
    return {op.name: op for op in env_models.operators}


def _progress(
    state: RelationalAbstractState, op: GroundOperator
) -> RelationalAbstractState:
    """STRIPS progression — the same rule ``RelationalAbstractSuccessorGenerator``
    uses."""
    atoms = (set(state.atoms) - set(op.delete_effects)) | set(op.add_effects)
    return RelationalAbstractState(atoms, set(state.objects))


def _named(state: RelationalAbstractState, name: str) -> Object:
    for obj in state.objects:
        if obj.name == name:
            return obj
    raise KeyError(f"object {name!r} not in abstract state")


def _classify(
    sampler: AcceptanceTrajectorySampler, needs_stick: bool, is_robot_route: bool
) -> RouteProbe:
    missing = tuple(sorted(str(a) for a in sampler.last_expected - sampler.last_actual))
    extra = tuple(sorted(str(a) for a in sampler.last_actual - sampler.last_expected))
    if is_robot_route and needs_stick:
        # Expected by construction: the button is past robot_reach_max_y.
        mode: FailureMode = "out_of_reach"
    elif sampler.hit_transition_failure:
        mode = "transition_error"
    elif missing:
        mode = "effect_not_achieved"
    else:
        mode = "extra_atoms"
    return RouteProbe(achieved=False, mode=mode, missing=missing, extra=extra)


_OK = RouteProbe(achieved=True, mode="ok", missing=(), extra=())


def _refine(
    sampler: AcceptanceTrajectorySampler,
    bpg: BilevelPlanningGraph,
    x0: object,
    s0: RelationalAbstractState,
    action_plan: list[GroundOperator],
    seed: int,
    budgets: tuple[int, float],
) -> bool:
    """Refine one short probe plan, deriving its expected abstract-state sequence.

    ``budgets`` is ``(num_sampling_attempts_per_step, timeout_s)``. Every failure mode
    is swallowed into ``False``: the probe only asks whether the route works, and the
    sampler's ``last_expected``/``last_actual`` carry the reason for the caller.
    ``BaseException`` is deliberate — ``TrajectorySamplingFailure`` and
    ``TransitionFailure`` do not derive from ``Exception``.
    """
    attempts, timeout_s = budgets
    state_plan = [s0]
    for op in action_plan:
        state_plan.append(_progress(state_plan[-1], op))
    refiner = BacktrackingRefiner(
        trajectory_sampler=sampler,
        num_sampling_attempts_per_step=attempts,
        seed=seed,
    )
    try:
        return refiner(x0, state_plan, action_plan, timeout_s, bpg) is not None
    except BaseException:  # pylint: disable=broad-exception-caught
        return False


def probe_problem(
    env_models: SesameModels,
    x0: object,
    problem_id: int,
    *,
    num_sampling_attempts_per_step: int = 5,
    max_trajectory_steps: int = 200,
    timeout_s: float = 20.0,
    reach: ButtonReach | None = None,
    acceptance: Acceptance = "exact",
) -> ProblemProbe:
    """Probe every button in one problem on both routes.

    Costs ``2N`` refinements. Each probe gets a fresh :class:`BacktrackingRefiner` (so
    outcomes are independent) but shares one :class:`BilevelPlanningGraph`; the graph is
    scratch here and never read back.

    ``acceptance`` is passed through to :class:`AcceptanceTrajectorySampler`; see its
    docstring for what ``"superset"`` relaxes and why that stays sound.
    """
    s0 = env_models.state_abstractor(x0)
    reach = reach if reach is not None else classify_buttons(x0)
    ops = _operators_by_name(env_models)

    robot = _named(s0, "robot")
    stick = _named(s0, "stick")
    button_names = sorted(
        (o.name for o in s0.objects if o.name.startswith("button")),
        key=lambda n: (len(n), n),
    )

    budgets = (num_sampling_attempts_per_step, timeout_s)
    results: list[ButtonProbe] = []
    for idx, name in enumerate(button_names):
        button = _named(s0, name)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        sampler = AcceptanceTrajectorySampler(
            controller_generator=RelationalControllerGenerator(env_models.skills),
            transition_function=env_models.transition_fn,
            state_abstractor=env_models.state_abstractor,
            max_trajectory_steps=max_trajectory_steps,
            acceptance=acceptance,
        )

        needs_stick = name in reach.needs_stick
        seed = problem_id * 1000 + idx

        robot_plan = [ops[_ROBOT_PRESS].ground((robot, button))]
        if _refine(sampler, bpg, x0, s0, robot_plan, seed, budgets):
            robot_route = _OK
        else:
            robot_route = _classify(sampler, needs_stick, True)

        if robot_route.achieved:
            stick_route = RouteProbe(False, "ok", (), ())  # not attempted
        else:
            stick_plan = [
                ops[_PICK_STICK].ground((robot, stick)),
                ops[_STICK_PRESS].ground((robot, stick, button)),
            ]
            if _refine(sampler, bpg, x0, s0, stick_plan, seed, budgets):
                stick_route = _OK
            else:
                stick_route = _classify(sampler, needs_stick, False)

        results.append(
            ButtonProbe(
                button=name,
                needs_stick=needs_stick,
                robot=robot_route,
                stick=stick_route,
            )
        )

    return ProblemProbe(
        problem_id=problem_id,
        num_buttons=len(button_names),
        buttons=tuple(results),
        reach_max_y=reach.reach_max_y,
    )
