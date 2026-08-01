"""Trajectory sampler with a configurable abstract-state acceptance test.

Upstream's :class:`ParameterizedControllerTrajectorySampler` accepts a sampled skill
execution only when the resulting abstract state equals the planned one *exactly*
(``parameterized_controller_sampler.py:89``), and raises a payload-free
:class:`TrajectorySamplingFailure` otherwise. That gives us neither a knob nor an
explanation, so this module re-implements the loop — faithfully — adding both.

Two acceptance rules:

``"exact"`` (default)
    Byte-for-byte upstream behaviour. Anything measured under this rule is a measurement
    of stock kinder.

``"superset"``
    Accept when ``planned.atoms ⊆ achieved.atoms``.

**Why superset is sound for goal achievement.** Every StickButton2D operator has only
positive preconditions, and ``Pressed`` is never deleted by any operator. So if each step
achieves at least its planned atoms, the final achieved state contains the final planned
state, which contains the goal — the refined trajectory genuinely presses every button.
What the rule tolerates is *extra* atoms, and in this environment those are incidental
button presses (the robot or stick sweeping over a button en route, which the env always
counts as a press) and multi-button overlaps. Both are progress toward an all-``Pressed``
goal, never damage.

**Why it is still a deviation.** The planned abstract-state sequence is symbolic
progression from ``s_0``; once reality carries extra atoms it diverges from that
sequence, so later steps are checked against a plan that no longer describes the world.
This is why the rule is opt-in and reported rather than being made the default silently.
See ``docs/kinder_stickbutton2d_map.md`` §7 for what it does and does not buy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import RelationalAbstractState, TransitionFailure
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from relational_structs import GroundOperator

Acceptance = Literal["exact", "superset"]


class AcceptanceTrajectorySampler(ParameterizedControllerTrajectorySampler):
    """Upstream's sampler plus a configurable acceptance rule and failure bookkeeping.

    ``last_expected`` / ``last_actual`` hold the planned and achieved atom sets from the
    most recent call (successful or not), and ``hit_transition_failure`` records whether
    the rollout was cut short by a :class:`TransitionFailure`. Callers use these to
    attribute *why* a step failed, which upstream discards.
    """

    def __init__(
        self, *args: object, acceptance: Acceptance = "exact", **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.acceptance: Acceptance = acceptance
        self.last_expected: frozenset[object] = frozenset()
        self.last_actual: frozenset[object] = frozenset()
        self.hit_transition_failure = False

    def __call__(  # type: ignore[override]
        self,
        x: object,
        s: RelationalAbstractState,
        a: GroundOperator,
        ns: RelationalAbstractState,
        bpg: BilevelPlanningGraph,
        rng: np.random.Generator,
    ) -> tuple[list[object], list[object]]:
        del s  # upstream ignores it too; the plan supplies the target directly
        controller = self._controller_generator(a)
        x_traj: list[object] = [x]
        u_traj: list[object] = []
        controller.reset(x, controller.sample_parameters(x, rng))
        self.hit_transition_failure = False

        for _ in range(self._max_trajectory_steps):
            if controller.terminated():
                break
            u = controller.step()
            try:
                nx = self._transition_function(x, u)
            except TransitionFailure:
                self.hit_transition_failure = True
                break
            controller.observe(nx)
            x_traj.append(nx)
            u_traj.append(u)
            bpg.add_state_node(nx)
            bpg.add_action_edge(x, u, nx)
            x = nx

        final_state = x_traj[-1]
        final_abstract_state = self._state_abstractor(final_state)
        bpg.add_abstract_state_node(final_abstract_state)
        bpg.add_state_abstractor_edge(final_state, final_abstract_state)

        self.last_expected = frozenset(ns.atoms)
        self.last_actual = frozenset(final_abstract_state.atoms)
        if self.acceptance == "superset":
            accepted = self.last_expected <= self.last_actual
        else:
            accepted = final_abstract_state == ns
        if accepted:
            return x_traj, u_traj
        raise TrajectorySamplingFailure()
