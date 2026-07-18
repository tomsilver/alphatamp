"""Symbolic action / skeleton (task-plan) data structures.

A *skeleton* here is what the PIGINet paper calls a "task plan" pi: an ordered
sequence of discrete, grounded high-level actions whose *continuous* arguments
(grasp pose, IK config, placement pose) are left unbound. Refinement (see
``refine.py``) is what searches for those continuous values.

Names deliberately mirror the LAZY codebase
(``policy-guided-lazy-tamp/lifted/utils.py`` ``PropositionalAction``) so a future
port onto the real ``lifted/`` search is mechanical: an action stringifies as
``pick(panda,red_block0,red_table)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Action:
    """One grounded, symbolic action.

    Continuous params are intentionally absent.
    """

    name: str  # operator: pick / place / stack / unstack
    args: tuple[
        str, ...
    ]  # discrete object arguments, e.g. ("panda", "red_block0", "red_table")

    def __str__(self) -> str:
        return f"{self.name}({','.join(self.args)})"

    def as_tuple(self) -> tuple[str, ...]:
        """Flat tuple form used in the PIGINet record (operator + args)."""
        return (self.name, *self.args)


@dataclass(frozen=True)
class Skeleton:
    """An ordered task plan: a sequence of :class:`Action` instances."""

    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            object.__setattr__(self, "actions", tuple(self.actions))

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __str__(self) -> str:
        return " -> ".join(str(a) for a in self.actions)

    @property
    def length(self) -> int:
        return len(self.actions)

    def key(self) -> tuple[tuple[str, ...], ...]:
        """Hashable identity used to dedupe skeletons (order-sensitive)."""
        return tuple(a.as_tuple() for a in self.actions)

    def to_tokens(self) -> list[tuple[str, ...]]:
        """Flat token list for the PIGINet record (paper Table II, task plan)."""
        return [a.as_tuple() for a in self.actions]

    def to_tokens_as_lists(self) -> list[list[str]]:
        """JSON-friendly token list ([operator, *args] per step)."""
        return [list(a.as_tuple()) for a in self.actions]

    @classmethod
    def from_action_tuples(cls, rows: Sequence[Sequence[str]]) -> "Skeleton":
        return cls(tuple(Action(r[0], tuple(r[1:])) for r in rows))
