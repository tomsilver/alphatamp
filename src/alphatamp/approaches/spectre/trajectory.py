"""STRIPS progression over a stored skeleton.

Per ``SPECTRE_METHOD_SPEC.md`` §4.1.5 (and §8.1 on Substage B), intermediate
abstract states ``s_1, …, s_{L-1}`` are deterministic functions of
``(s_0, g_1, …, g_i)`` under STRIPS semantics and are therefore recoverable at
encode / analysis time. We persist only ``s_0`` plus each skeleton's
``final_abstract_state`` (Substage A); whenever downstream code needs the full
trajectory — vocab extraction, Substage B encoding, consistency checks — it
calls :func:`reconstruct_trajectory`.

The progression rule mirrors
``bilevel_planning.utils.RelationalAbstractSuccessorGenerator``::

    next_atoms = (state.atoms - op.delete_effects) | op.add_effects

but operates over a concrete pre-grounded skeleton rather than generating
successors during search.
"""

from __future__ import annotations

from typing import Sequence

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundOperator


def apply_operator(
    state: RelationalAbstractState,
    op: GroundOperator,
) -> RelationalAbstractState:
    """STRIPS forward step: ``s' = (s.atoms \\ del(op)) ∪ add(op)``.

    Does not check preconditions; callers that care should use
    :func:`reconstruct_trajectory` or check explicitly. Object set is inherited
    unchanged — STRIPS operators do not introduce new objects.
    """
    new_atoms = (state.atoms - op.delete_effects) | op.add_effects
    return RelationalAbstractState(atoms=new_atoms, objects=state.objects)


def reconstruct_trajectory(
    initial_state: RelationalAbstractState,
    operator_seq: Sequence[GroundOperator],
    verify_preconditions: bool = True,
) -> list[RelationalAbstractState]:
    """Return ``[s_0, s_1, …, s_L]`` via STRIPS progression through ``operator_seq``.

    The returned list always has length ``len(operator_seq) + 1``. When
    ``verify_preconditions=True``, raises :class:`AssertionError` if any
    operator's preconditions are not satisfied in its preceding state — this
    catches malformed skeletons (e.g. schema corruption or a bug in the
    symbolic planner's output) before they silently poison downstream code.
    """
    trajectory: list[RelationalAbstractState] = [initial_state]
    current = initial_state
    for i, op in enumerate(operator_seq):
        if verify_preconditions and not op.preconditions.issubset(current.atoms):
            missing = op.preconditions - current.atoms
            raise AssertionError(
                f"Step {i} operator {op.name}: preconditions not satisfied in "
                f"preceding state. Missing atoms: {sorted(missing, key=str)}"
            )
        current = apply_operator(current, op)
        trajectory.append(current)
    return trajectory
