"""Op-level cost counters (TTD spec §5.3).

A stream call is the control-flow unit but *not* a cost unit — the spec measured a
~272x cost spread across stream kinds. Costs are instead accounted at the level of
geometric primitives: **C-ops** (polygon construction, buffer, NFP/IFP construction,
union, Minkowski sum) and **P-ops** (prepared-predicate evaluations, which begin in the
streams chunk). Every ``ttd_core`` geometry constructor takes an optional
:class:`OpCounter` and bumps it; ``calibrate.py`` (a later chunk) turns raw counts into
a calibrated cost via a per-op µs table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CKind = Literal["poly_construct", "buffer", "nfp", "ifp", "union", "minkowski"]
"""Kinds of construction op tracked by :class:`OpCounter` (spec §5.3)."""


@dataclass
class OpCounter:
    """Accumulates geometric-primitive op counts (spec §5.3).

    ``p_ops`` (prepared-predicate evaluations) stays 0 until the streams chunk; chunk 1
    exercises only the C-ops. Pass an instance into any geometry constructor to record
    its cost; pass ``None`` to skip accounting.
    """

    p_ops: int = 0
    c_ops: dict[str, int] = field(default_factory=dict)

    def bump(self, kind: CKind, n: int = 1) -> None:
        """Increment the C-op counter for ``kind`` by ``n``."""
        self.c_ops[kind] = self.c_ops.get(kind, 0) + n

    def bump_p(self, n: int = 1) -> None:
        """Increment the prepared-predicate (P-op) counter by ``n``."""
        self.p_ops += n

    def total_c(self) -> int:
        """Total C-ops across all kinds."""
        return sum(self.c_ops.values())

    def merge(self, other: "OpCounter") -> None:
        """Fold ``other``'s counts into this counter in place."""
        self.p_ops += other.p_ops
        for kind, n in other.c_ops.items():
            self.c_ops[kind] = self.c_ops.get(kind, 0) + n
