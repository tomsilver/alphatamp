"""Proof-demotion filter — sound deductions consumed *outside* the network (proposal §5/§7).

Proof-tier facts compile to demotion rules on the *ranking*, never the pool: a
provably-dead candidate is pushed to the back, never deleted (P-E completeness). The
network cannot override a proof; a wrong proof only reorders. This is the non-learned
half of the proof/hint split and the core of the zero-parameter **hand-rule stack** (§9).

Removal-monotone deductions on DD2D subsets (each keyed by the *failed* staged subset ``F``):

- ``blocked-at-contents`` — the attempt staged ``F`` and the target was still blocked
  (no clear grasp with the remaining contents). By removal-monotonicity, any candidate
  staging ``S ⊆ F`` leaves at least as much in the drawer, so it is also blocked → dead.
- ``pack-impossible`` — subset ``F`` provably cannot pack. Any candidate staging
  ``S ⊇ F`` packs even more, so it is also impossible → dead.

Soundness telemetry: the fraction of demoted candidates that nonetheless succeed. Under a
correct registry this is 0; a nonzero value is a live model-error alarm (§6.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProofState:
    """Accumulates provably-dead candidates over a rollout (indices into the pool)."""

    subsets: list  # subsets[i] = frozenset staged by candidate i
    dead: set = field(default_factory=set)
    _blocked_supersets: list = field(default_factory=list)  # F with blocked-at-contents
    _impossible_subsets: list = field(default_factory=list)  # F with pack-impossible

    def observe_failure(
        self, failed_idx: int, blocked: bool, pack_impossible: bool
    ) -> None:
        """Record proof facts from one failed attempt (candidate ``failed_idx``)."""
        fset = self.subsets[failed_idx]
        if blocked:
            self._blocked_supersets.append(fset)
        if pack_impossible:
            self._impossible_subsets.append(fset)
        self._recompute()

    def _recompute(self) -> None:
        for i, s in enumerate(self.subsets):
            if i in self.dead:
                continue
            # blocked-at-contents(F): S ⊆ F ⇒ S leaves ⊇ (all−F) ⇒ also blocked.
            if any(s <= f for f in self._blocked_supersets):
                self.dead.add(i)
                continue
            # pack-impossible(F): S ⊇ F ⇒ also impossible.
            if any(f <= s for f in self._impossible_subsets):
                self.dead.add(i)

    def is_dead(self, idx: int) -> bool:
        return idx in self.dead


def demote(order, dead: set) -> list:
    """Stable reorder: keep live candidates in their given order, then the dead ones
    (also in order) — never dropped (P-E). If everything is dead the pool is unchanged in
    order, so completeness holds (all candidates remain attemptable)."""
    order = list(order)
    live = [i for i in order if i not in dead]
    demoted = [i for i in order if i in dead]
    return live + demoted


def demote_scores(scores, dead: set, offset: float = 1e6):
    """Equivalent demotion in score space: subtract a finite offset from dead candidates so
    they rank last but are never −inf (still attemptable if all else fails)."""
    import numpy as np

    out = np.asarray(scores, dtype=float).copy()
    idx = [i for i in dead if 0 <= i < len(out)]
    if idx:
        out[idx] -= offset
    return out
