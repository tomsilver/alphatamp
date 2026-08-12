"""Feasibility statistics ϕ with online feedback (LAZY III-E).

ϕ is the Laplace-smoothed per-operator success rate ``ϕ(k) = (succ(k)+1)/(att(k)+1)``,
keyed on the renaming-invariant per-operator key ``k=(op_name, typed-local args)`` — the
analog of LAZY's anonymised computation-graph key. It is fit from the train split's
refinement outcomes and then updated online during a test rollout: when a skeleton fails,
the attributed operators' attempt counts are incremented, which down-weights every
remaining pool skeleton that shares them (the "feedback" of LAZY's title).

Attribution (``decisions/07`` 2026-08-09):
- success  → every operator on the plan gets ``succ+=1, att+=1`` (a feasible witness).
- fail     → the operator(s) at the failing ``step_index`` (from
             :func:`failure_record.records_for_candidate`) get ``att+=1`` (bottleneck).
- fail with no usable attribution (SB2D has no class-1 culprit channel) → suffix-blame:
  every operator gets ``att+=1`` (the whole skeleton is charged).
- error    → skipped (as everywhere in the codebase).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from alphatamp.approaches.spectre.baselines.lazy.tree import STOP, OpKey, op_key
from alphatamp.approaches.spectre.failure_record import records_for_candidate
from alphatamp.approaches.spectre.schema import EpisodeRecord

# op key -> [successes, attempts]
PhiStats = dict[OpKey, list[int]]


def _skeleton_op_keys(episode: EpisodeRecord, i: int) -> list[OpKey]:
    return [op_key(op) for op in episode.skeleton_pool[i].operator_seq]


def attributed_keys(episode: EpisodeRecord, i: int) -> list[OpKey]:
    """Operator keys charged an attempt when candidate ``i`` fails refinement.

    ``episode`` must be canonicalized so the keys match the fitted ϕ. Bottleneck when
    the refiner named a failing step; whole-skeleton suffix-blame otherwise.
    """
    keys = _skeleton_op_keys(episode, i)
    recs = records_for_candidate(episode, i)
    blamed = sorted({r.step_index for r in recs if 0 <= r.step_index < len(keys)})
    if blamed:
        return [keys[s] for s in blamed]
    return keys  # unattributed -> charge the whole skeleton


def fit_phi(episodes: Iterable[EpisodeRecord]) -> PhiStats:
    """Fit ϕ counts from canonicalized train episodes."""
    stats: PhiStats = defaultdict(lambda: [0, 0])
    for ep in episodes:
        for i, out in enumerate(ep.outcomes):
            if out.outcome == "success":
                for k in _skeleton_op_keys(ep, i):
                    stats[k][0] += 1
                    stats[k][1] += 1
            elif out.outcome == "fail":
                for k in attributed_keys(ep, i):
                    stats[k][1] += 1
            # "error" -> skipped
    return dict(stats)


def phi_value(stats: PhiStats, key: OpKey) -> float:
    """``(succ+1)/(att+1)``; unseen key (and STOP) → 1.0."""
    if key == STOP:
        return 1.0
    sa = stats.get(key)
    if sa is None:
        return 1.0
    return (sa[0] + 1.0) / (sa[1] + 1.0)


def observe_failure(stats: PhiStats, episode: EpisodeRecord, i: int) -> None:
    """Online feedback: charge an attempt to candidate ``i``'s attributed operators.

    Mutates ``stats`` in place (call on a copy of the fitted prior, per rollout).
    """
    for k in attributed_keys(episode, i):
        if k in stats:
            stats[k][1] += 1
        else:
            stats[k] = [0, 1]  # unseen op now has one failed attempt -> ϕ=0.5


def copy_stats(stats: PhiStats) -> PhiStats:
    """Deep-ish copy of the counts (values are 2-element lists)."""
    return {k: [v[0], v[1]] for k, v in stats.items()}
