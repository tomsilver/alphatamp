"""Typed-fact vocabulary + context gathering for the Step-11 evidence pathway.

At rollout step ``t`` the evidence context is the union of typed post-mortem facts
harvested from the skeletons that have already **failed** (the set ``F``). This module
fixes the small categorical fact vocabulary and turns ``(episode, F)`` into a flat list
of fact records the tensorizer can embed. Fact *arguments* are object names (bound to
episode-local tags by the dataset), so a fact carries **identity**, not just a count —
the property the live scramble gauge and P5 test.
"""

from __future__ import annotations

from typing import Iterable

from alphatamp.approaches.spectre.schema import EpisodeRecord

# Fixed fact-type ids (0 = pad/none). Kept small + explicit; new types append at the end.
FACT_TYPE_IDS: dict[str, int] = {
    "blocked-at-contents": 1,
    "extraction-failed": 2,
    "grasp-witness": 3,
    "pack-exhausted": 4,
    "pack-impossible": 5,
}
N_FACT_TYPES = len(FACT_TYPE_IDS)

# Tier ids (0 = pad). Proof-tier facts are *also* consumed by proof-demotion outside the
# net; exposing the tier lets the scorer weight them differently if it helps.
TIER_IDS: dict[str, int] = {"proof": 1, "hint": 2}


class FactRecord:
    """One flattened context fact: type id, tier id, argument object names, source
    skeleton."""

    __slots__ = ("type_id", "tier_id", "args", "source_idx")

    def __init__(
        self, type_id: int, tier_id: int, args: tuple[str, ...], source_idx: int
    ) -> None:
        self.type_id = type_id
        self.tier_id = tier_id
        self.args = args
        self.source_idx = source_idx


def gather_context_facts(
    episode: EpisodeRecord, failed_indices: Iterable[int]
) -> list[FactRecord]:
    """Flatten the post-mortem facts of the failed skeletons in ``failed_indices``.

    Unknown fact types are skipped (forward-compatible). Order follows
    ``failed_indices`` then the per-record fact order; the model is permutation-
    invariant over the memory, so order is not load-bearing.
    """
    out: list[FactRecord] = []
    for idx in failed_indices:
        pm = episode.outcomes[idx].post_mortem
        if pm is None:
            continue
        for f in pm.facts:
            tid = FACT_TYPE_IDS.get(f.fact_type)
            if tid is None:
                continue
            out.append(
                FactRecord(
                    type_id=tid,
                    tier_id=TIER_IDS.get(f.tier, 0),
                    args=tuple(f.args),
                    source_idx=idx,
                )
            )
    return out
