"""Necessity labels: how likely is each object to be *required*?

The v2.2 model has a per-object ``necessary``/``relevant`` auxiliary head, but no
collection has ever populated ``EpisodeRecord.aux_labels`` -- the masked BCE sees only
``-1`` (ignore), so the head has never received a gradient. v3's necessity conditioning is
therefore a build-from-scratch, starting with the label this module produces.

**The label.** For each object *i*, ``p_i`` = the fraction of **minimum-size feasible
manipulated sets** that contain *i*, and the scene's difficulty estimate is
``d_hat = sum_i p_i`` -- the expected size of a minimal solution.

Four choices, each load-bearing:

- **Soft, not binary.** The schema's original wording was "in *every* minimal feasible
  subset" (a set intersection). Measured on dd2d_v3 that intersection is **empty in 33.2%
  of episodes** -- those episodes would supply no non-trivial positive at all -- and it
  under-estimates difficulty by an amount that *grows with stratum* (3.37 vs 4 at s3),
  which is the direction that re-creates the s3 collapse the prior caused. The soft
  marginal has zero error: ``d_hat == stratum`` exactly on all 400 train episodes.
- **Deduped by subset.** The pool enumerates *orderings*, so the same manipulated set
  appears ~2x among 2-blocker candidates. Counting orderings would weight a subset by how
  many of its permutations happened to be sampled -- a sampler artifact leaking into a
  supervised label.
- **Any-ordering-feasible.** 4.20% of multi-ordering subsets have order-dependent
  feasibility. A subset that succeeds under some ordering is a real solution, so it counts
  as feasible. (That 4.20% is also a ceiling on what *any* subset-level feature can
  discriminate -- worth remembering before blaming the model for the residue.)
- **Goal objects excluded**, via ``DomainSpec.manipulated``. DD2D's target is retrieved by
  every candidate, so including it would spend a logit on a constant the model is already
  told by ``obj_is_target``.

**Legality.** These are derived from feasibility labels, so they are *training supervision*
-- which is what labels are for -- not an input. Nothing here runs at inference: the head
predicts ``p_i`` from geometry alone. Be plain about what that makes it, though: on DD2D
``d_hat`` equals the stratum exactly, so the head is frankly a learned difficulty
estimator. That is legitimate and it is also the honest description.

**Known bias.** Minimality is over the *observed* pool. Coverage is complete for 1- and
2-subsets but only ~66-77% of 3-subsets, so ``p_i`` is noisiest exactly at s3 -- the
stratum the length-generalization experiment deploys on. :func:`necessity_labels` returns
the coverage alongside the labels so a caller can report it rather than discover it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alphatamp.approaches.spectre.domain import DomainSpec, spec_for
from alphatamp.approaches.spectre.schema import EpisodeRecord

__all__ = ["NecessityLabels", "necessity_labels"]


@dataclass(frozen=True)
class NecessityLabels:
    """Per-episode necessity supervision."""

    p: dict[str, float]
    """Object name -> marginal probability it is required (0 for objects never needed)."""

    d_hat: float
    """``sum(p)`` -- the expected size of a minimal solution."""

    n_minimal: int
    """Distinct minimum-size feasible manipulated sets found. 1 means unambiguous."""

    min_size: int
    """Size of a minimal solution (equals the stratum on DD2D)."""

    def as_targets(self, names: list[str]) -> list[float]:
        """Labels aligned to ``names``, for the tensorizer."""
        return [self.p.get(n, 0.0) for n in names]


def necessity_labels(
    episode: EpisodeRecord, spec: Optional[DomainSpec] = None
) -> Optional[NecessityLabels]:
    """Compute necessity labels for one episode, or ``None`` if nothing is feasible.

    ``None`` rather than all-zeros: an episode with no feasible candidate carries no
    information about what is *required*, and labelling every object 0 would teach the head
    that nothing is ever needed. Such episodes are already excluded from training by the
    ``num_success >= 1`` filter, so this is belt-and-braces.
    """
    spec = spec or spec_for(episode.provenance.env_variant)
    goal_objs = spec.goal_objects(episode)

    # Dedupe orderings: a manipulated set is feasible iff ANY of its orderings succeeded.
    feasible_by_subset: dict[frozenset[str], bool] = {}
    for skeleton, outcome in zip(episode.skeleton_pool, episode.outcomes):
        subset = spec.manipulated(skeleton, goal_objs)
        ok = outcome.outcome == "success"
        feasible_by_subset[subset] = feasible_by_subset.get(subset, False) or ok

    feasible = [s for s, ok in feasible_by_subset.items() if ok]
    if not feasible:
        return None

    min_size = min(len(s) for s in feasible)
    minimal = [s for s in feasible if len(s) == min_size]

    counts: dict[str, int] = {}
    for subset in minimal:
        for name in subset:
            counts[name] = counts.get(name, 0) + 1
    p = {name: c / len(minimal) for name, c in counts.items()}

    return NecessityLabels(
        p=p,
        d_hat=sum(p.values()),
        n_minimal=len(minimal),
        min_size=min_size,
    )


def subset_coverage(episode: EpisodeRecord, spec: Optional[DomainSpec] = None) -> dict:
    """How much of the subset lattice the pool actually enumerates, by size.

    ``{size: (distinct_subsets_present, n_choose_k)}``. The necessity label is only as
    good as this: a minimal solution the pool never enumerated cannot be counted, so a
    size whose coverage is far below 1.0 has a downward-biased ``p_i``.
    """
    from math import comb

    spec = spec or spec_for(episode.provenance.env_variant)
    goal_objs = spec.goal_objects(episode)
    n_manipulable = len(frozenset(episode.object_registry) - goal_objs)

    by_size: dict[int, set[frozenset[str]]] = {}
    for skeleton in episode.skeleton_pool:
        subset = spec.manipulated(skeleton, goal_objs)
        by_size.setdefault(len(subset), set()).add(subset)
    return {
        size: (len(subsets), comb(n_manipulable, size))
        for size, subsets in sorted(by_size.items())
    }
