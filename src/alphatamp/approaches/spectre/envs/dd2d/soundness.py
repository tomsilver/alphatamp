"""Per-domain soundness registry for the typed post-mortem facts (proposal §6.3).

A negative/constructive deduction harvested from the refiner is **proof-tier** only under
explicit, declared assumptions; otherwise it is a **hint**. The registry is a few declared
lines per domain, reported verbatim in the paper. An *empty* registry runs everything as
hints — the proof tier is an optimization, never a requirement (proposal §4).

DD2D declares all four assumptions (§6.3): the refiner's exact checks are against the true
geometry (model fidelity); failing checks enumerate finite sets exactly (exactness);
removing an item leaves other poses unchanged (removal-monotonicity / quasi-statics, true
in DD2D by construction); and actions outside the container do not change collision status
inside it (locality). So DD2D's deducible facts are proofs.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fact types that can be proof-tier — but only under a satisfying registry (§6.4).
PROOF_ELIGIBLE = frozenset(
    {
        "blocked-at-contents",  # removal-monotone drawer-clearing deduction
        "pack-impossible",  # sound area / arrangement-complete certificate
        "extracted-ok",  # constructive: a prefix pick proved the item extractable
        "packed-ok",  # constructive: a prefix place proved the staged set packs
    }
)


@dataclass(frozen=True)
class SoundnessRegistry:
    """Declared per-domain assumptions gating the proof tier (proposal §6.3)."""

    model_fidelity: bool = False
    exactness: bool = False
    removal_monotone: bool = False
    locality: bool = False

    def proofs_allowed(self) -> bool:
        """All four assumptions hold ⇒ deducible facts may be proof-tier."""
        return (
            self.model_fidelity
            and self.exactness
            and self.removal_monotone
            and self.locality
        )

    def tier(self, fact_type: str) -> str:
        """Proof only for a proof-eligible fact type under a fully-declared registry;
        otherwise hint. An empty registry ⇒ everything is a hint."""
        if fact_type in PROOF_ELIGIBLE and self.proofs_allowed():
            return "proof"
        return "hint"


# DD2D declares all four assumptions (§6.3).
DD2D_REGISTRY = SoundnessRegistry(
    model_fidelity=True, exactness=True, removal_monotone=True, locality=True
)

# A domain that declares nothing — every fact is a hint, proof demotion inactive.
EMPTY_REGISTRY = SoundnessRegistry()
