"""Turn a generated proposal sequence into the shared rollout-FP metric.

**Live-refine.** DD2D's candidate pool holds the 200 shortest planner plans — every
one-blocker staging and most *ordered* two-blocker stagings, but only a few percent of
the three-blocker orderings. A VLM proposing a three-item staging is therefore almost
always off-pool, which is exactly where stratum 3 lives. Silently dropping those
proposals would hand VLMPlan free attempts and flatter it; so an off-pool proposal is
refined for real, on the scene reconstructed from the record's stored geometry, and costs
an attempt like any other.

**In-pool proposals are never re-refined.** Their label is read off the stored
``OutcomeRecord``, so VLMPlan sees byte-identical labels to every other method in the
comparison and no accounting difference can come from relabelling.

**Budget.** The attempt budget is the pool cap (200), matching the uncensored-evaluation
convention (``docs/decisions.md`` 2026-06-07). Attempts spent on wrong off-pool guesses
are real cost — that is the point. When the model's own sequence runs out before the
budget, the tail is filled from the planner's published order, so an exhausted episode
degrades to the astar-dist baseline rather than becoming a missing value; fill slots are
flagged so the degradation is visible in the diagnostics.

**The two label sources must be checked, not assumed.** In-pool labels come off disk;
off-pool labels are computed now. Those agree only if the env code has not moved since
the collection, so :func:`label_agreement` measures it — re-label stored pool plans live
and compare. Measured 2026-07-24, n=168 each:

- ``dd2d_v3`` (collected *after* that day's grasp changes): **0.982**.
- ``dd2d_v2`` (collected before them): **0.917**, and in *both* directions — the
  fingerprint of the two same-day changes, one monotone-harder (contact-run fix) and one
  monotone-easier (internal concave grasps). Exactly the staleness ``docs/decisions.md``
  2026-07-24 flags, now quantified.

So the live refiner tracks the current env code, and v2's gap was the labels moving, not
a bug here: the refiner is deterministic at v2's settings (live-vs-live 60/60) and the
2026-07-19 reconstruction invariant still holds (0/1624 stored-feasible subsets read as
blocked). v3's residual ~1.8% is boundary noise — the collection runs the refiner under a
*wall-clock* budget, so a marginal packing can land either side of the cutoff. Check this
gate before trusting any VLMPlan figure, and prefer the freshest collection.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from alphatamp.approaches.spectre.envs.dd2d.drawer.collect import (
    _stable_seed as _collector_stable_seed,)
from alphatamp.approaches.spectre.envs.dd2d.drawer.planning import staging_skeleton
from alphatamp.approaches.spectre.envs.dd2d.drawer.refine import DD2DRefiner
from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import reconstruct_scene
from alphatamp.approaches.spectre.schema import EpisodeRecord

from .adapter import EnvAdapter, Labeler, Step
from .dd2d_adapter import DD2DAdapter

logger = logging.getLogger(__name__)

ATTEMPT_BUDGET = 200

# A live off-pool refinement must use the SAME refiner settings the collection used, or
# its labels are drawn from a different distribution than the stored in-pool ones. The
# settings live in each collection's ``manifest.json`` ``config`` block, and they differ
# between collections — v2 ran ``time_budget: 4.0``, v3 ran ``20.0`` — so this is a
# per-variant preset, never a single hard-coded default.
REFINER_PRESETS: dict[str, dict[str, Any]] = {
    "dd2d_v2": {
        "budget": None,
        "retry_cap": 10,
        "samples_per_step": 15,
        "time_budget": 4.0,
    },
    "dd2d_v3": {
        "budget": None,
        "retry_cap": 10,
        "samples_per_step": 15,
        "time_budget": 20.0,
    },
    # v4 = v3's refiner settings (lam=0.8, crowd=5, k=200, retry_cap=10,
    # samples_per_step=15, time_budget=20.0) plus observation-only instrumentation, so an
    # off-pool label uses the same distribution as the stored in-pool ones. See
    # conf/env/dd2d_v4.yaml.
    "dd2d_v4": {
        "budget": None,
        "retry_cap": 10,
        "samples_per_step": 15,
        "time_budget": 20.0,
    },
}
DEFAULT_VARIANT = "dd2d_v3"


def refiner_kwargs_for(env_variant: str) -> dict[str, Any]:
    """Refiner settings for a collection, from its manifest-recorded config."""
    if env_variant not in REFINER_PRESETS:
        raise KeyError(
            f"No refiner preset for env_variant {env_variant!r}. Add one from that "
            f"collection's manifest.json 'config' block; known: "
            f"{sorted(REFINER_PRESETS)}"
        )
    return dict(REFINER_PRESETS[env_variant])


def stable_seed(key: object) -> int:
    """Deterministic per-skeleton refiner seed.

    Re-exported from the collector rather than reimplemented, so a live off-pool label
    uses exactly the seed the collection would have used for the same plan.
    """
    return int(_collector_stable_seed(key))


@dataclass
class Attempt:
    """One refinement attempt in the realized sequence."""

    members: list[str]
    in_pool: bool
    pool_idx: int | None
    label: str
    source: str  # "vlm" | "fill"
    round_index: int | None = None
    # Refinement wall-clock for this attempt: the stored per-candidate time for an
    # in-pool plan, the run-captured live-refine time for an off-pool one, ``None`` if
    # unknown.
    refine_s: float | None = None
    # The full canonical step sequence, ``[[name, [args...]], ...]``. Kept so the
    # planner inspector can render VLMPlan's actual ordered plans (its attempts are off
    # the shared pool, so there is no skeleton to index) via the env's plan formatter.
    steps: list = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the cache record's ``attempts`` list."""
        return {
            "members": list(self.members),
            "in_pool": self.in_pool,
            "pool_idx": self.pool_idx,
            "label": self.label,
            "source": self.source,
            "round": self.round_index,
            "refine_s": self.refine_s,
            "steps": [[name, list(args)] for name, args in self.steps],
        }


@dataclass
class ScoreResult:
    """The scored episode — the payload of one compare-cache record."""

    problem_id: int
    stratum: int
    fp: float
    censored: bool
    attempts: list[Attempt] = field(default_factory=list)
    n_offpool: int = 0
    n_fill_used: int = 0
    first_success_source: str | None = None
    n_live_refines: int = 0
    # Wall-clock to first success (the VLMPlan row of the comparison's §2b).
    # ``infer_s`` is the VLM generation cost (summed round api_s; set by the caller from
    # the sequences file). ``refine_s`` sums the per-attempt refinement up to and
    # including first success. The ``_capped`` pair re-walks that order under the
    # deployed per-candidate refinement cap (a slow near-feasible candidate is abandoned
    # at the cap), mirroring the pool methods so the two are comparable.
    infer_s: float = 0.0
    refine_s: float = 0.0
    refine_s_capped: float = 0.0
    fp_capped: float | None = None

    @property
    def order(self) -> list[int]:
        """Pool indices of the realized attempts; ``-1`` marks an off-pool plan."""
        return [a.pool_idx if a.pool_idx is not None else -1 for a in self.attempts]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form of the whole compare-cache record."""
        return {
            "problem_id": self.problem_id,
            "stratum": self.stratum,
            "fp": self.fp,
            "censored": self.censored,
            "order": self.order,
            "attempts": [a.as_dict() for a in self.attempts],
            "n_attempts": len(self.attempts),
            "n_offpool": self.n_offpool,
            "n_fill_used": self.n_fill_used,
            "n_live_refines": self.n_live_refines,
            "first_success_source": self.first_success_source,
            "infer_s": self.infer_s,
            "refine_s": self.refine_s,
            "refine_s_capped": self.refine_s_capped,
            "fp_capped": self.fp_capped,
        }


class MemoizingLabeler(Labeler):
    """A :class:`~.adapter.Labeler` with a persistent memo. Subclasses do the refining.

    The memo is keyed by ``(problem_id, canonical step tuple)`` and written to disk, so
    re-scoring a run — after a code change, or to regenerate the cache — never re-refines
    a plan it has already labelled.
    """

    def __init__(self, memo_path: Path | None = None) -> None:
        self._memo_path = memo_path
        self._memo: dict[str, str] = {}
        if memo_path is not None and memo_path.is_file():
            self._memo = json.loads(memo_path.read_text(encoding="utf-8"))
        # Per-refine wall-clock, keyed identically to the label memo, persisted to a
        # sidecar. The off-pool refine happens once (in the run's first-success stop
        # check); its wall-clock is captured there so the wall-clock section never has to
        # re-refine at score time — a refine can cost up to the collection's time budget
        # (20 s on DD2D), so re-doing it per attempt would be minutes per problem.
        self._times_path = (
            memo_path.with_name(memo_path.stem + "_times.json")
            if memo_path is not None
            else None
        )
        self._times: dict[str, float] = {}
        if self._times_path is not None and self._times_path.is_file():
            self._times = json.loads(self._times_path.read_text(encoding="utf-8"))
        self.n_refines = 0

    @staticmethod
    def _key(problem_id: int, steps: Sequence[Step]) -> str:
        return f"{problem_id}|" + ";".join(
            f"{name}({','.join(args)})" for name, args in steps
        )

    def label(self, episode: object, steps: Sequence[Step]) -> str:
        """Feasibility of one off-pool proposal, memoised."""
        assert isinstance(episode, EpisodeRecord)
        key = self._key(int(episode.provenance.problem_id), steps)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if episode.scene_geometry is None:
            raise ValueError(
                f"episode {episode.provenance.problem_id} has no scene_geometry; "
                "off-pool proposals cannot be labelled without it"
            )
        started = time.perf_counter()
        label = self._refine(episode, steps)
        self._times[key] = time.perf_counter() - started
        self.n_refines += 1
        self._memo[key] = label
        return label

    def refine_seconds(
        self, episode: EpisodeRecord, steps: Sequence[Step]
    ) -> float | None:
        """Recorded wall-clock of the off-pool refine for this plan, or ``None``.

        ``None`` means this labeler never refined it (e.g. an in-pool plan, whose time is
        the stored per-candidate one instead, or a plan labelled by a different run).
        """
        return self._times.get(self._key(int(episode.provenance.problem_id), steps))

    def _refine(self, episode: EpisodeRecord, steps: Sequence[Step]) -> str:
        raise NotImplementedError

    def flush(self) -> None:
        """Persist the memo (and the refine-time sidecar) so a re-score never re-refines
        a labelled plan."""
        if self._memo_path is None:
            return
        self._memo_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._memo_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._memo), encoding="utf-8")
        tmp.replace(self._memo_path)
        if self._times_path is not None:
            tmp_t = self._times_path.with_suffix(".json.tmp")
            tmp_t.write_text(json.dumps(self._times), encoding="utf-8")
            tmp_t.replace(self._times_path)


class OffPoolLabeler(MemoizingLabeler):
    """DD2D: reconstruct the scene from stored geometry and run ``DD2DRefiner``.

    The memo key changed on 2026-08-01 from the staged-member tuple to the full canonical
    step sequence, because the generic signature no longer carries "members". Existing
    memo files therefore miss and are re-refined once -- a cost, not a correctness
    problem, since the refiner is deterministic at fixed settings.
    """

    def __init__(
        self,
        memo_path: Path | None = None,
        env_variant: str = DEFAULT_VARIANT,
        adapter: DD2DAdapter | None = None,
    ) -> None:
        super().__init__(memo_path)
        self._refiner = DD2DRefiner(**refiner_kwargs_for(env_variant))
        self._adapter = adapter or DD2DAdapter()

    def _refine(self, episode: EpisodeRecord, steps: Sequence[Step]) -> str:
        assert episode.scene_geometry is not None  # `label` checks it before calling
        target = self._adapter.target_name(episode)
        members = self._adapter.discretionary_objects(steps)
        scene = reconstruct_scene(episode.scene_geometry)
        skeleton = staging_skeleton(target, list(members))
        result = self._refiner.refine(skeleton, scene, seed=stable_seed(skeleton.key()))
        return "success" if result.feasible else "fail"


def label_step_sequence(
    episode: EpisodeRecord,
    steps: tuple[Step, ...],
    adapter: EnvAdapter,
    labeler: Labeler,
    pool: dict[tuple[object, ...], int] | None = None,
    stored: Sequence[str] | None = None,
) -> tuple[str, int | None]:
    """Label one proposal: ``(label, pool_index or None)``.

    **The single definition of how a VLMPlan proposal is labelled.** A plan matching a
    pooled candidate takes that candidate's *stored* outcome and is never re-refined, so
    VLMPlan reads byte-identical labels to every other method; anything else is refined
    live at the collection's settings.

    Extracted so the generation loop's first-success stop check and the scorer use the
    same rule. Two copies of this would drift, and the symptom would be a run that stops
    generating on a "success" the scorer then labels a failure.
    """
    pool = adapter.pool_index(episode) if pool is None else pool
    stored = [o.outcome for o in episode.outcomes] if stored is None else stored
    pool_idx = pool.get(adapter.canonical_key(steps))
    if pool_idx is not None:
        return stored[pool_idx], pool_idx
    return labeler.label(episode, steps), None


def _stored_refine_seconds(episode: EpisodeRecord, pool_idx: int) -> float | None:
    """The collection's stored per-candidate refine wall-clock for pool skeleton j.

    The same number every pool method replays (DD2D v3/v4 instrumented collections).
    SB2D outcomes carry it too, but it is not the deployed-cap-instrumented figure, so
    the SB2D wall-clock is reported for completeness only.
    """
    if 0 <= pool_idx < len(episode.outcomes):
        value = getattr(episode.outcomes[pool_idx], "refinement_wall_clock_s", None)
        return float(value) if value is not None else None
    return None


def _fp_refine_capped(attempts: Sequence[Attempt], cap: float) -> tuple[float, float]:
    """Re-walk the realized order under a per-candidate refinement cap.

    Charges ``min(t, cap)`` per attempt and stops at the first success reached *within*
    the cap; a feasible-but-slow candidate (``t > cap``) is abandoned and counts against
    FP, exactly the deployed-cap semantics the pool methods use. Because the uncapped
    walk already stopped at the first success, the recorded attempts end there — on DD2D
    the feasible p95 (0.44 s) is far below the 2 s cap, so the success is essentially
    never the abandoned-slow case and this is exact; in the rare case it is,
    ``fp_capped`` is conservatively the censored count.
    """
    total = 0.0
    for i, attempt in enumerate(attempts):
        t = attempt.refine_s if attempt.refine_s is not None else 0.0
        total += min(t, cap)
        if attempt.label == "success" and t <= cap:
            return float(i), total
    return float(len(attempts)), total


def score_sequence(
    episode: EpisodeRecord,
    proposals: Sequence[tuple[tuple[Step, ...], int]],
    adapter: EnvAdapter,
    stratum: int,
    labeler: Labeler | None = None,
    attempt_budget: int = ATTEMPT_BUDGET,
    fill_from_published: bool = True,
    env_variant: str = DEFAULT_VARIANT,
    refine_cap_s: float = 2.0,
) -> ScoreResult:
    """Walk the proposals (then the published-order fill) to the first success.

    ``proposals`` is ``[(steps, round_index), ...]`` in the order the model produced
    them. Returns the FP the comparison table reports, and the per-attempt / total
    refinement wall-clock the §2b wall-clock section reports (``infer_s`` is filled
    by the caller from the sequences file). In-pool attempts take the collection's
    stored per-candidate time; off-pool attempts take the time the run captured when it
    refined them (see ``MemoizingLabeler``).
    """
    pool = adapter.pool_index(episode)
    stored = [o.outcome for o in episode.outcomes]
    labeler = labeler or OffPoolLabeler(env_variant=env_variant)
    refine_seconds = getattr(labeler, "refine_seconds", None)

    sequence: list[tuple[tuple[Step, ...], str, int | None]] = [
        (steps, "vlm", round_index) for steps, round_index in proposals
    ]
    if fill_from_published:
        proposed = {adapter.canonical_key(steps) for steps, _ in proposals}
        sequence += [
            (steps, "fill", None)
            for steps in adapter.published_order(episode)
            if adapter.canonical_key(steps) not in proposed
        ]

    result = ScoreResult(
        problem_id=int(episode.provenance.problem_id),
        stratum=stratum,
        fp=float(attempt_budget),
        censored=True,
    )
    for steps, source, round_index in sequence[:attempt_budget]:
        members = adapter.discretionary_objects(steps)
        label, pool_idx = label_step_sequence(
            episode, steps, adapter, labeler, pool=pool, stored=stored
        )
        if pool_idx is not None:
            refine_s = _stored_refine_seconds(episode, pool_idx)
        elif refine_seconds is not None:
            refine_s = refine_seconds(episode, steps)  # run-captured live-refine time
        else:
            refine_s = None
        if pool_idx is None:
            result.n_offpool += 1
        if source == "fill":
            result.n_fill_used += 1
        result.attempts.append(
            Attempt(
                members=members,
                in_pool=pool_idx is not None,
                pool_idx=pool_idx,
                label=label,
                source=source,
                round_index=round_index,
                refine_s=refine_s,
                steps=list(steps),
            )
        )
        if label == "success":
            result.fp = float(len(result.attempts) - 1)
            result.censored = False
            result.first_success_source = source
            break

    result.n_live_refines = labeler.n_refines
    result.refine_s = sum((a.refine_s or 0.0) for a in result.attempts)
    result.fp_capped, result.refine_s_capped = _fp_refine_capped(
        result.attempts, refine_cap_s
    )
    return result


def published_order_fp(
    episode: EpisodeRecord, attempt_budget: int = ATTEMPT_BUDGET
) -> float:
    """FP of the planner's own order — the reference an exhausted episode falls back to.

    Used as a sanity check that the fill path reproduces astar-dist rather than as a
    reported number (astar-dist has its own cache dir).
    """
    for j, outcome in enumerate(episode.outcomes[:attempt_budget]):
        if outcome.outcome == "success":
            return float(j)
    return float(attempt_budget)


def label_agreement(
    episodes: Sequence[EpisodeRecord],
    adapter: EnvAdapter,
    samples_per_episode: int = 6,
    seed: int = 0,
    env_variant: str = DEFAULT_VARIANT,
    make_labeler: Callable[[], Labeler] | None = None,
) -> dict[str, Any]:
    """Re-label stored pool plans live and report agreement with the stored outcome.

    The consistency gate on mixing the two label sources (see the module docstring).
    ``agreement`` well below 1.0 means the env code has moved since the collection, so a
    VLMPlan number is scored against two different label functions and must not be
    reported until the data is re-collected. Disagreements are returned split by
    direction, because that is what identifies *which* change moved them.
    """
    rng = random.Random(seed)
    agree = 0
    stored_fail_live_success = 0
    stored_success_live_fail = 0
    _new_labeler = make_labeler or (lambda: OffPoolLabeler(env_variant=env_variant))
    for episode in episodes:
        pool = adapter.published_order(episode)
        labeler = _new_labeler()
        indices = rng.sample(range(len(pool)), min(samples_per_episode, len(pool)))
        successes = [
            j for j, o in enumerate(episode.outcomes) if o.outcome == "success"
        ]
        if successes:  # always include a positive; the pool is mostly negatives
            indices.append(successes[0])
        for j in indices:
            live = labeler.label(episode, pool[j])
            stored = episode.outcomes[j].outcome
            if live == stored:
                agree += 1
            elif stored == "success":
                stored_success_live_fail += 1
            else:
                stored_fail_live_success += 1
    total = agree + stored_fail_live_success + stored_success_live_fail
    return {
        "n": total,
        "agreement": agree / total if total else None,
        "stored_fail_live_success": stored_fail_live_success,
        "stored_success_live_fail": stored_success_live_fail,
    }


def spearman_vs_published(order: Sequence[int]) -> float | None:
    """Rank correlation between the realized in-pool order and the published order.

    The pre-registered **trivial-mimicry null**: a value near 1 means the model
    reproduced the planner's ascending-size enumeration, so its number says little about
    geometric reasoning regardless of where it lands. ``None`` when there are fewer than
    two in-pool attempts to correlate.
    """
    in_pool = [idx for idx in order if idx >= 0]
    if len(in_pool) < 2:
        return None
    positions = np.arange(len(in_pool), dtype=float)
    published = np.asarray(in_pool, dtype=float)
    if published.std() == 0.0:
        return None
    ranked = np.argsort(np.argsort(published)).astype(float)
    if ranked.std() == 0.0 or positions.std() == 0.0:
        return None
    return float(np.corrcoef(positions, ranked)[0, 1])


def assert_single_run(out_dir: Path, run: str) -> None:
    """Refuse to mix two runs' records in one comparison-cache directory.

    A cache directory *is* one method row: the reader averages every record in it. Two
    runs writing there (a 5-problem pilot and a 16-problem smoke, say) would silently
    produce a row that is neither — per-problem records from different models, prompts or
    loop constants. Caught here rather than surfacing as an inexplicable mean.
    """
    for path in sorted(out_dir.glob("*.json")):
        existing = json.loads(path.read_text(encoding="utf-8")).get("run")
        if existing is not None and existing != run:
            raise ValueError(
                f"{out_dir} already holds records from run {existing!r}, but this is "
                f"run {run!r}. A cache dir is one method row; mixing runs would average "
                f"two different configurations. Use a different `cache_subdir` for this "
                f"run, or delete {out_dir}."
            )


def write_record(
    out_dir: Path,
    result: ScoreResult,
    extra: dict[str, Any] | None = None,
    force: bool = False,
    writer: Callable[[Path, dict[str, Any]], None] | None = None,
) -> bool:
    """Write one compare-cache record; ``False`` if it already existed."""
    path = out_dir / f"{result.problem_id}.json"
    if path.exists() and not force:
        return False
    payload = result.as_dict()
    payload["spearman_vs_published"] = spearman_vs_published(result.order)
    payload.update(extra or {})
    if writer is not None:
        writer(path, payload)
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return True
