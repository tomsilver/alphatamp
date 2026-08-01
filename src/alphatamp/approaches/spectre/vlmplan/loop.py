"""The batched generation loop: rounds -> parse -> ground -> dedup -> stall or stop.

This module is env-agnostic; everything DD2D-specific arrives through the
:class:`~.adapter.EnvAdapter`.

**The static hard line.** Between rounds the model is shown only its own previously
proposed plans, for de-duplication. It never observes a refinement outcome. Any outcome
feedback would make this an adaptive method — a different row in the comparison table,
and not the zero-shot endpoint of the data axis this baseline exists to occupy.

**Filtered-for-free.** Malformed lines, symbolically inapplicable plans and duplicates
are dropped without consuming attempt budget. The other methods draw from a planner that
guarantees symbolic validity by construction, and symbolic checking is free relative to
refinement, which is the resource the metric counts. All three rates are reported so the
discount is visible rather than assumed.

The published-order fallback fill is deliberately *not* here — see `score.py`. Generation
records what the model proposed; scoring turns that into a realized attempt sequence
under a budget. Keeping them apart is what lets a run be re-scored after a re-collection
without re-querying the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from prpl_llm_utils.models import PretrainedLargeModel

from .adapter import EnvAdapter, Step
from .parsing import ParseStats, parse_response
from .template import PromptConfig, build_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopConfig:
    """Loop constants. Set from the pilot, then frozen before the test split."""

    max_plans: int = 200
    plans_per_round: int = 10
    tau: float = 0.2
    stall_rounds: int = 2
    max_rounds: int = 12
    max_retries: int = 3
    retry_backoff_s: float = 2.0


@dataclass(frozen=True)
class Proposal:
    """One accepted plan, with where it came from."""

    steps: tuple[Step, ...]
    round_index: int
    block_index: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the sequences file."""
        return {
            "steps": [[name, list(args)] for name, args in self.steps],
            "round": self.round_index,
            "block": self.block_index,
        }


@dataclass
class RoundLog:
    """Per-round accounting — the raw material for the yield-vs-depth diagnostic."""

    round_index: int
    n_requested: int = 0
    n_blocks: int = 0
    n_parsed: int = 0
    n_malformed: int = 0
    n_invalid: int = 0
    n_duplicate: int = 0
    n_new: int = 0
    n_decoration_repaired: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    prompt_chars: int = 0
    response_chars: int = 0
    # Server-reported usage. ``truncated`` means the completion hit the output cap, so
    # the last plan block was cut mid-line and lost — silent quality loss unless it is
    # surfaced. The 2026-07-24 smoke run truncated 16/104 responses at max_tokens=4096
    # and nothing said so; hence recorded, not inferred from response length.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False

    @property
    def yield_rate(self) -> float:
        """New unique valid plans per plan requested."""
        return self.n_new / max(1, self.n_requested)


@dataclass
class GenerationResult:
    """Everything one problem's generation produced."""

    problem_id: int
    proposals: list[Proposal] = field(default_factory=list)
    rounds: list[RoundLog] = field(default_factory=list)
    stalled: bool = False
    hit_max_rounds: bool = False
    stopped_on_success: bool = False
    parse_reasons: list[str] = field(default_factory=list)

    @property
    def n_truncated(self) -> int:
        """Rounds whose completion hit the output cap (and so lost their last plan)."""
        return sum(1 for r in self.rounds if r.truncated)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form for the sequences file."""
        return {
            "problem_id": self.problem_id,
            "proposals": [p.as_dict() for p in self.proposals],
            "rounds": [asdict(r) for r in self.rounds],
            "stalled": self.stalled,
            "hit_max_rounds": self.hit_max_rounds,
            "stopped_on_success": self.stopped_on_success,
            "n_truncated": self.n_truncated,
            # Capped: the reason list is a debugging aid, not a dataset, and a
            # pathological model can otherwise emit thousands per problem.
            "parse_reasons": self.parse_reasons[:50],
        }


def _query_with_retry(
    model: PretrainedLargeModel,
    prompt: str,
    images: Sequence[Any],
    hyperparameters: dict[str, Any],
    config: LoopConfig,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Query with backoff; ``(text, error, usage)``, text=None on permanent failure.

    A failed round is *not* silently skipped — it is returned as an error and counted as
    a zero-yield round by the caller, so a flaky backend shows up as stalling rather than
    as a shorter run.

    ``usage`` is the backend's own token accounting (``prompt_tokens`` /
    ``completion_tokens``), which is what makes truncation detectable exactly rather than
    guessed from response length.
    """
    last_error: str | None = None
    for attempt in range(config.max_retries):
        try:
            response = model.query(
                prompt, imgs=list(images) or None, hyperparameters=hyperparameters
            )
            return response.text, None, dict(response.metadata or {})
        except Exception as exc:  # noqa: BLE001 - backend errors are heterogeneous
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("query failed (attempt %d): %s", attempt + 1, last_error)
            if attempt + 1 < config.max_retries:
                time.sleep(config.retry_backoff_s * (attempt + 1))
    return None, last_error, {}


def generate_sequence(
    adapter: EnvAdapter,
    problem: object,
    problem_id: int,
    model: PretrainedLargeModel,
    config: LoopConfig,
    decode: dict[str, Any],
    base_seed: int = 0,
    stop_check: Callable[[Sequence[Proposal]], bool] | None = None,
) -> GenerationResult:
    """Run the multi-round loop for one problem and return its ordered proposals.

    ``stop_check`` is called after each round with the proposals so far and should
    return True once one of them is known to refine. **A feasible plan ends the
    episode**: ``max_plans`` (= the pool cap) is a hard ceiling for the case where every
    proposal keeps failing, not a target to fill. Generating past the first success
    cannot change the reported FP — the rollout would never have reached those
    proposals — so it is pure wall-clock. Measured on StickButton2D b5, where a problem
    ran all 10 rounds to accumulate 27 plans that the scorer then never looked past the
    first few of.

    It also changes what ``n_proposed`` means (§6 of the comparison notebook): with a
    stop check the count is censored at the first success, so it reads as "plans needed"
    rather than "plans the model can produce". Runs generated with and without a stop
    check are therefore not comparable on that column — but they are on FP.
    """
    result = GenerationResult(problem_id=problem_id)
    prompt_config = PromptConfig(plans_per_round=config.plans_per_round)

    skills = adapter.skills(problem)
    objects = adapter.objects(problem)
    images = adapter.images(problem)
    controllers = adapter.controllers_str(problem)
    typed_objects = adapter.typed_objects_str(problem)
    type_hierarchy = adapter.type_hierarchy_str(problem)
    goal_str = adapter.goal_str(problem)
    init_state_str = adapter.init_state_str(problem)

    seen: set[tuple[object, ...]] = set()
    consecutive_stalls = 0

    for round_index in range(config.max_rounds):
        log = RoundLog(round_index=round_index, n_requested=config.plans_per_round)
        started = time.time()

        prompt = build_prompt(
            controllers=controllers,
            typed_objects=typed_objects,
            type_hierarchy=type_hierarchy,
            goal_str=goal_str,
            init_state_str=init_state_str,
            config=prompt_config,
            previous_plans=[adapter.plan_str(p.steps) for p in result.proposals],
        )
        log.prompt_chars = len(prompt)
        # Fixed key set on every call: SQLite3PretrainedLargeModelCache asserts the
        # hyperparameter columns never change for the life of the cache object.
        #
        # ``seed`` varies per round and is a real API parameter, so rounds are genuinely
        # independent draws even when the repeat-suppression block is empty (which is
        # exactly what happens after a round whose plans were all rejected). ``round``
        # is cache-key-only and just keeps the cache legible.
        hyperparameters = {
            **decode,
            "seed": base_seed + round_index,
            "problem_id": problem_id,
            "round": round_index,
        }
        text, error, usage = _query_with_retry(
            model, prompt, images, hyperparameters, config
        )

        if text is None:
            log.error = error
            log.elapsed_s = time.time() - started
            result.rounds.append(log)
            consecutive_stalls += 1
            if consecutive_stalls >= config.stall_rounds:
                result.stalled = True
                break
            continue

        log.response_chars = len(text)
        _record_usage(log, usage, decode)
        plans, stats = parse_response(text, skills, objects, adapter.type_ancestors)
        _record_parse(log, stats)
        result.parse_reasons += stats.reasons

        for raw in plans:
            grounded = adapter.ground(raw, problem)
            if grounded is None:
                log.n_invalid += 1
                continue
            key = adapter.canonical_key(grounded)
            if key in seen:
                log.n_duplicate += 1
                continue
            seen.add(key)
            log.n_new += 1
            result.proposals.append(
                Proposal(
                    steps=grounded,
                    round_index=round_index,
                    block_index=raw.block_index,
                )
            )
            if len(result.proposals) >= config.max_plans:
                break

        log.elapsed_s = time.time() - started
        result.rounds.append(log)

        if len(result.proposals) >= config.max_plans:
            break
        # Checked after the round is logged, so the round that found the success is
        # still recorded with its accounting.
        if stop_check is not None and stop_check(result.proposals):
            result.stopped_on_success = True
            break
        consecutive_stalls = (
            consecutive_stalls + 1 if log.yield_rate < config.tau else 0
        )
        if consecutive_stalls >= config.stall_rounds:
            result.stalled = True
            break
    else:
        result.hit_max_rounds = True

    return result


def _record_parse(log: RoundLog, stats: ParseStats) -> None:
    log.n_blocks = stats.n_blocks
    log.n_parsed = stats.n_ok
    log.n_malformed = stats.n_malformed + stats.n_empty
    log.n_decoration_repaired = stats.n_decoration_repaired


def _record_usage(log: RoundLog, usage: dict[str, Any], decode: dict[str, Any]) -> None:
    """Record the backend's token accounting and flag a truncated completion.

    ``completion_tokens == max_tokens`` is an exact truncation signal: the model was cut
    off mid-generation, so the final plan block is incomplete and the parser drops it. A
    backend that reports no usage leaves ``truncated`` False rather than guessing.
    """
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    log.prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
    log.completion_tokens = (
        int(completion_tokens) if completion_tokens is not None else None
    )
    max_tokens = decode.get("max_tokens")
    log.truncated = bool(
        log.completion_tokens is not None
        and max_tokens is not None
        and log.completion_tokens >= int(max_tokens)
    )
