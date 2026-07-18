"""SesameModels factory for RoutedTransport2D.

Mirrors the ``kinder_bilevel_planning.env_models.<env>.create_bilevel_planning_models``
contract: takes ``(observation_space, action_space, **kwargs)`` and returns a
:class:`bilevel_planning.structs.SesameModels`. The dispatcher in
``alphatamp.approaches.spectre.collect`` routes to this factory when
``cfg.model_name == "routedtransport2d"``.

What's special about this factory:

- ``transition_fn`` is a sentinel: RT2D's refiner is closed-form and does not
  need to simulate per-step transitions. The callable raises immediately if
  invoked, surfacing routing bugs early instead of silently returning bogus
  states.
- ``skills`` are no-ops: each lifted skill has a controller whose ``step()``
  raises and whose ``terminated()`` returns True. RT2D bypasses the trajectory
  sampler; these are placeholders required by the SesameModels contract
  (LiftedSkill init asserts operator and controller share parameters).
- ``observation_to_state`` unwraps the ProblemInstance carried in the obs dict.
  ``state_abstractor`` and ``goal_deriver`` then read the pre-built
  ``initial_abstract_state`` and ``goal`` directly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Box, Space
from relational_structs import LiftedOperator, Variable

from alphatamp.approaches.spectre.envs.routedtransport2d import operators as ops
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    ProblemInstance,
)

# ---- Stub state representation --------------------------------------------


class _StubState:
    """Carries the ProblemInstance through the SesameModels callbacks.

    RT2D has no concrete state. The "observation" emitted by the gym env is
    just a dict containing the ProblemInstance; ``observation_to_state`` wraps
    it in a hashable container that ``state_abstractor`` and ``goal_deriver``
    can pull from.
    """

    __slots__ = ("problem",)

    def __init__(self, problem: ProblemInstance) -> None:
        self.problem = problem

    # Hashable on identity is fine — collect.py never compares states by value.
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


# ---- No-op controller -----------------------------------------------------


class _NoOpController(GroundParameterizedController):
    """Placeholder controller that never produces actions.

    RT2D doesn't run a trajectory sampler, but the SesameModels contract still
    requires LiftedSkill objects with paired operator + controller, and the
    LiftedSkill ``__post_init__`` checks parameter equality between them.
    """

    def sample_parameters(self, x: object, rng: np.random.Generator) -> None:
        del x, rng

    def reset(self, x: object, params: object) -> None:
        del x, params

    def terminated(self) -> bool:
        return True

    def step(self) -> object:
        raise RuntimeError(
            "RoutedTransport2D should not step a controller — refinement is closed-form"
        )

    def observe(self, x: object) -> None:
        del x


def _make_skill(op: LiftedOperator) -> LiftedSkill:
    """Pair a lifted operator with a no-op LiftedParameterizedController."""
    variables: Sequence[Variable] = list(op.parameters)
    controller: LiftedParameterizedController = LiftedParameterizedController(
        variables=variables,
        controller_cls=_NoOpController,
        params_space=None,
    )
    return LiftedSkill(operator=op, controller=controller)


# ---- Factory --------------------------------------------------------------


def _transition_fn_unused(x: _StubState, u: object) -> _StubState:
    del x, u
    raise RuntimeError(
        "RoutedTransport2D transition_fn invoked — closed-form refinement should"
        " bypass the trajectory sampler entirely. Likely cause: dispatch in"
        " collect.py is not routing this episode through ThreeGateRefiner."
    )


def _stub_state_space() -> Space:
    return Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)


def create_routedtransport_models(
    observation_space: Space,
    action_space: Space,
    num_items: int = 3,
    variant: str = "v1",
) -> SesameModels:
    """Build a ``SesameModels`` for RoutedTransport2D.

    ``num_items`` and ``variant`` are accepted for parity with the kinder
    factories but only the ``num_items`` field is currently load-bearing; all
    abstract-state and operator content is N-independent (operators are
    lifted) and the variant is informational.
    """
    del num_items, variant  # informational; structure is uniform across variants

    state_space = _stub_state_space()

    def observation_to_state(o: dict[str, Any]) -> _StubState:
        problem = o.get("_problem") if isinstance(o, dict) else None
        if not isinstance(problem, ProblemInstance):
            raise RuntimeError(
                "RoutedTransport2D observation must carry a ProblemInstance under"
                f" key '_problem'; got {type(o).__name__}"
            )
        return _StubState(problem)

    def state_abstractor(x: _StubState) -> RelationalAbstractState:
        return x.problem.initial_abstract_state

    def goal_deriver(x: _StubState) -> RelationalAbstractGoal:
        # Reattach this state_abstractor to the cached goal so ``check_state``
        # works downstream (collect.py only checks ``isinstance(goal,
        # RelationalAbstractGoal)`` and reads ``goal.atoms``, but other consumers
        # may invoke check_state).
        return RelationalAbstractGoal(
            atoms=x.problem.goal.atoms,
            state_abstractor=state_abstractor,
        )

    skills: set[LiftedSkill] = {_make_skill(op) for op in ops.ALL_OPERATORS}

    return SesameModels(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        transition_fn=_transition_fn_unused,
        types=set(ops.ALL_TYPES),
        predicates=set(ops.ALL_PREDICATES),
        observation_to_state=observation_to_state,
        state_abstractor=state_abstractor,
        goal_deriver=goal_deriver,
        skills=skills,
    )
