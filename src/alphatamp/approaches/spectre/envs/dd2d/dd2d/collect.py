"""DD2D dataset collector (Steps 2-3 of docs/piginet_dd2d_plan.md).

The PIGINet collection protocol (Yang et al. §VI-C) for DD2D. The sorting
``blocks_tamp/collect.py`` hard-codes ``generate_sorting_problem`` and is not reusable
here, so this is a DD2D-native collector.

Step 2 (this commit) is the **per-problem** core :func:`collect_problem`:

Generate one instance of a requested min-feasible-subset **stratum**, enumerate up to
``k`` diverse task plans with the **astar + distance-heuristic** planner, refine them
**in order stopping at the first feasible plan**, and persist **one positive + only the
negatives that preceded it** (drop the whole problem if nothing refines within ``k``).

Injection seams (``problem`` / ``planner`` / ``refine_fn``) make the stop-at-first-
success, drop-unsolvable, and exact-stratum logic unit-testable without the heavy
generator/planner/ refiner. Per-problem disk I/O (crop PNGs + record JSON, co-located in
``<split_dir>/<problem_id>/``) happens worker-side because crops need the live scene,
which is expensive to ship back through a process pool.

Step 3 adds the parallel coordinator (balanced strata, disjoint seed bands, manifests).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field

from ..record import PIGINetExample
from ..skeleton import Skeleton
from .record_ext import build_dd2d_example, write_crops

_LABEL_SOURCE = "refine_buffer_stage"  # DD2DRefiner.label_source


# --------------------------------------------------------------------------- #
# config + result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DD2DCollectConfig:
    """Locked collection knobs (docs/piginet_dd2d_plan.md).

    Fields are surfaced so the Step-3 coordinator / Step-4 EDA can vary them (e.g.
    ``crowd`` per stratum) without touching :func:`collect_problem`.
    """

    lam: float = 0.8
    margin: float = 1.0
    crowd: int = 5
    diverse_crowd: bool = True
    k: int = 200  # diverse plans enumerated + refined per problem
    budget: int | None = (
        None  # refiner stream-call cap; None = uncapped (needs time_budget)
    )
    retry_cap: int = 10
    samples_per_step: int = 15
    time_budget: float = 20.0  # wall-clock seconds per plan
    full_pool: bool = (
        True  # refine ALL k plans (multi pos/neg); False = legacy stop-at-first
    )
    # Held-out generalization knobs (docs/decisions 2026-08-01); all inert at their
    # defaults so a standard collection is byte-identical.
    n_items_range: tuple[int, int] | None = (
        None  # None => locked {10..13}; else uniform [lo, hi]
    )
    require_families: tuple[str, ...] = ()  # force >= 1 of each into every scene
    extra_families: dict[str, float] | None = None  # augment the clutter/collar pool
    fill_max: float | None = None  # raise the coverage cap for denser scenes


@dataclass
class ProblemResult:
    """Outcome of collecting one problem.

    ``examples`` is non-empty iff kept (the positive is last).
    """

    problem_id: str
    seed: int
    stratum: int
    n_items: int
    kept: bool
    reason: str  # solved | wrong_stratum | gen_failed | no_plans | unsolved | error:*
    n_skeletons: int = 0
    n_refined: int = 0
    n_pos: int = 0
    n_neg: int = 0
    first_feasible_rank: int | None = None
    min_feasible_subset: int | None = None
    wall_time: float = 0.0
    examples: list = field(default_factory=list)  # list[PIGINetExample]


# --------------------------------------------------------------------------- #
# deterministic seeds
# --------------------------------------------------------------------------- #
def _stable_seed(key) -> int:
    """Deterministic, cross-process refiner seed for a skeleton (mirrors
    ``heuristic_experiment._stable_seed``), so a plan's feasibility label replays bit-
    for-bit regardless of which worker/order first reaches it."""
    return int(hashlib.md5(repr(key).encode()).hexdigest()[:8], 16)


def _sample_n_items(seed: int, n_items_range: tuple[int, int] | None = None) -> int:
    """Per-problem item count, deterministic per seed and decorrelated from the stratum.

    Default {10,11,12,13} => 9-12 blockers (num_blockers = n_items-1). ``n_items_range=(lo,
    hi)`` draws uniformly in [lo, hi] instead -- the held-out unseen-count band
    (docs/decisions 2026-08-01)."""
    if n_items_range is None:
        return 10 + _stable_seed(("nitems", seed)) % 4
    lo, hi = n_items_range
    return lo + _stable_seed(("nitems", seed)) % (hi - lo + 1)


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# one problem (the §VI-C protocol)
# --------------------------------------------------------------------------- #
def collect_problem(
    seed: int,
    stratum: int,
    config: DD2DCollectConfig,
    split_dir: str,
    planner=None,
    refine_fn=None,
    problem=None,
) -> ProblemResult:
    """Collect one problem: exact-stratum filter, stop-at-first-success, drop-if-
    unsolved.

    ``problem`` / ``planner`` / ``refine_fn`` are injection seams for testing; in
    production all three are ``None`` and built from the locked config.
    ``refine_fn(skeleton, scene, seed) -> RefineResult``.
    """
    t0 = time.time()
    n_items = _sample_n_items(seed, config.n_items_range)

    def _drop(reason: str, pid: str | None = None) -> ProblemResult:
        return ProblemResult(
            problem_id=pid or f"dd2d_s{seed}_st{stratum}",
            seed=seed,
            stratum=stratum,
            n_items=n_items,
            kept=False,
            reason=reason,
            wall_time=round(time.time() - t0, 3),
        )

    # 1) generate (or use injected problem)
    if problem is None:
        from .problem import generate_dd2d_problem

        try:
            problem = generate_dd2d_problem(
                lam=config.lam,
                seed=seed,
                margin=config.margin,
                split="train",
                n_items=n_items,
                crowd=(
                    0 if stratum == 0 else config.crowd
                ),  # collar would block a stratum-0 target
                diverse_crowd=config.diverse_crowd,
                require_subset=(stratum >= 2),
                min_subset=max(stratum, 2),
                unblocked_target=(stratum == 0),
                require_families=config.require_families,
                extra_families=config.extra_families,
                fill_max=config.fill_max,
                min_items=(config.n_items_range[0] if config.n_items_range else None),
                certify=True,
                budget=config.budget,
                retry_cap=config.retry_cap,
                samples_per_step=config.samples_per_step,
                time_budget=config.time_budget,
            )
        except RuntimeError:
            return _drop("gen_failed")

    # 2) exact-stratum filter (min_subset floors mfs at `stratum`; reject mfs > stratum)
    if problem.min_feasible_subset != stratum:
        res = _drop("wrong_stratum", pid=problem.problem_id)
        res.min_feasible_subset = problem.min_feasible_subset
        return res

    # 3) plan (astar + distance heuristic)
    if planner is None:
        from .planning import make_dd2d_planner

        planner = make_dd2d_planner(
            prefer="pyperplan", search="astar", heuristic="dist"
        )
    skeletons = planner.plan(problem, config.k)
    if not skeletons:
        res = _drop("no_plans", pid=problem.problem_id)
        res.min_feasible_subset = problem.min_feasible_subset
        return res

    # 4) refine in order, stop at the first feasible plan
    if refine_fn is None:
        from .refine import DD2DRefiner

        refiner = DD2DRefiner(
            budget=config.budget,
            retry_cap=config.retry_cap,
            samples_per_step=config.samples_per_step,
            time_budget=config.time_budget,
        )
        refine_fn = refiner.refine

    tried: list[tuple[Skeleton, object, int]] = (
        []
    )  # (skeleton, RefineResult, refine_seed)
    first_feasible_rank: int | None = None
    for i, sk in enumerate(skeletons):
        rseed = _stable_seed(sk.key())
        res = refine_fn(sk, problem.scene, rseed)
        tried.append((sk, res, rseed))
        if res.feasible and first_feasible_rank is None:
            first_feasible_rank = i + 1
        if res.feasible and not config.full_pool:
            break  # legacy stop-at-first-success

    solved = any(
        r.feasible for _, r, _ in tried
    )  # full-pool: keep if >=1 feasible plan exists
    result = ProblemResult(
        problem_id=problem.problem_id,
        seed=seed,
        stratum=stratum,
        n_items=n_items,
        kept=False,
        reason="",
        n_skeletons=len(skeletons),
        n_refined=len(tried),
        first_feasible_rank=first_feasible_rank,
        min_feasible_subset=problem.min_feasible_subset,
    )
    if not solved:
        result.reason = "unsolved"
        result.wall_time = round(time.time() - t0, 3)
        return result

    # 5) persist: crops once (kept only), one record per tried skeleton (positive last)
    prob_dir = os.path.join(split_dir, problem.problem_id)
    os.makedirs(prob_dir, exist_ok=True)
    refs = write_crops(problem, os.path.join(prob_dir, "images"))

    examples: list[PIGINetExample] = []
    for plan_idx, (sk, res, rseed) in enumerate(tried):
        ex = build_dd2d_example(
            problem,
            sk,
            res,
            planner.name,
            images=refs,
            label_source=_LABEL_SOURCE,
            extra_provenance={
                "stratum": stratum,
                "n_items": n_items,
                "n_items_realized": len(problem.scene.items),
                "plan_idx": plan_idx,
                "split": os.path.basename(os.path.normpath(split_dir)),
                "refine_seed": rseed,
                "planner_search": "astar",
                "planner_heuristic": "dist",
                # v3: the exact arguments this scene was generated and refined under.
                # `decisions.md` 2026-07-19 records that omitting these forced the
                # "reconstruct, don't regenerate" rule, because a post-hoc consumer had
                # to *infer* them and a miss silently produced a different scene with
                # the same object names. Storing them does not retire that rule -- the
                # record's own poses stay authoritative -- but it makes provenance
                # auditable and any future regeneration checkable rather than guessed.
                "gen_params": {
                    "lam": config.lam,
                    "margin": config.margin,
                    "crowd": 0 if stratum == 0 else config.crowd,
                    "diverse_crowd": config.diverse_crowd,
                    "require_subset": stratum >= 2,
                    "min_subset": max(stratum, 2),
                    "unblocked_target": stratum == 0,
                    "n_items": n_items,
                    "require_families": list(config.require_families),
                    "extra_families": config.extra_families,
                    "fill_max": config.fill_max,
                    "min_items": (
                        config.n_items_range[0] if config.n_items_range else None
                    ),
                    "certify": True,
                },
                "refiner_params": {
                    "budget": config.budget,
                    "retry_cap": config.retry_cap,
                    "samples_per_step": config.samples_per_step,
                    "time_budget": config.time_budget,
                },
            },
        )
        _atomic_write(os.path.join(prob_dir, f"{plan_idx:03d}.json"), ex.to_json())
        examples.append(ex)

    result.kept = True
    result.reason = "solved"
    result.examples = examples
    result.n_pos = sum(1 for e in examples if e.label)
    result.n_neg = sum(1 for e in examples if not e.label)
    result.wall_time = round(time.time() - t0, 3)
    return result


# --------------------------------------------------------------------------- #
# split / stratum partition
# --------------------------------------------------------------------------- #
STRATA = (0, 1, 2, 3)  # min_feasible_subset values (0 = target directly graspable)


def _split_bands(band: int = 1_000_000) -> dict[str, tuple[int, int]]:
    """Disjoint seed bands per dataset split (problem_id encodes the seed, so disjoint
    bands => no instance is shared across splits)."""
    return {
        "train": (0 * band, 1 * band),
        "test": (1 * band, 2 * band),
        "val": (2 * band, 3 * band),
    }


def _stratum_targets(total: int, n: int = 3) -> list[int]:
    """Split ``total`` into ``n`` balanced per-stratum sub-targets, remainder front-
    loaded (400 -> [134,133,133]; 100 -> [34,33,33]; 3 -> [1,1,1]; 2 -> [1,1,0])."""
    base, rem = divmod(total, n)
    return [base + 1 if i < rem else base for i in range(n)]


def _stratum_bands(seed_band: tuple[int, int], n: int = 3) -> list[tuple[int, int]]:
    """Divide a split's seed band into ``n`` disjoint sub-bands, one per stratum, so no
    seed is ever used for two strata (problem_id has no stratum field -> would otherwise
    collide)."""
    s0, s1 = seed_band
    w = (s1 - s0) // n
    return [(s0 + i * w, s0 + (i + 1) * w if i < n - 1 else s1) for i in range(n)]


# --------------------------------------------------------------------------- #
# process-pool worker
# --------------------------------------------------------------------------- #
def _collect_task(args) -> ProblemResult:
    """Top-level (picklable) pool worker.

    Records are written worker-side by ``collect_problem``; we clear ``examples`` before
    returning so the process-pool payload stays lean (the coordinator needs only
    counts).
    """
    seed, stratum, config, split_dir = args
    try:
        res = collect_problem(seed, stratum, config, split_dir)
    except Exception as e:  # keep the run alive; record the failure
        return ProblemResult(
            problem_id=f"dd2d_s{seed}_st{stratum}",
            seed=seed,
            stratum=stratum,
            n_items=_sample_n_items(seed, config.n_items_range),
            kept=False,
            reason=f"error:{type(e).__name__}",
        )
    res.examples = []  # drop heavy payload; NNN.json already on disk
    return res


# --------------------------------------------------------------------------- #
# split coordinator (balanced strata)
# --------------------------------------------------------------------------- #
def _seed_from_problem_id(pid: str) -> int | None:
    """Parse the trailing ``…_s{seed}`` from a DD2D problem_id
    (``dd2d_n11_l80_c5dc_s42``)."""
    tail = pid.rsplit("_s", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


def _stratum_of_seed(
    seed: int, sub_bands: list[tuple[int, int]], strata: tuple[int, ...]
) -> int | None:
    for (lo, hi), s in zip(sub_bands, strata):
        if lo <= seed < hi:
            return s
    return None


def _load_resume_state(split_dir, sub_bands, strata):
    """Recover (skip_set, per-stratum kept counts) for a ``--resume`` run:

    kept = existing ``<problem_id>/`` dirs (authoritative); skip = kept seeds ∪ logged
    attempted seeds (``attempted.log``, so previously-dropped seeds aren't re-tried).
    """
    skip: set[int] = set()
    kept_by_stratum: dict[int, int] = {s: 0 for s in strata}
    if os.path.isdir(split_dir):
        for name in os.listdir(split_dir):
            if not os.path.isdir(os.path.join(split_dir, name)):
                continue
            seed = _seed_from_problem_id(name)
            if seed is None:
                continue
            s = _stratum_of_seed(seed, sub_bands, strata)
            if s is not None:
                skip.add(seed)
                kept_by_stratum[s] += 1
    log_path = os.path.join(split_dir, "attempted.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if parts and parts[0].isdigit():
                    skip.add(int(parts[0]))
    return skip, kept_by_stratum


def _fmt_hms(secs: float) -> str:
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def _truncate_to_targets(
    split_dir: str,
    sub_bands: list[tuple[int, int]],
    sub_targets: list[int],
    strata: tuple[int, ...],
) -> dict[int, list[str]]:
    """Delete any overshoot so each stratum has EXACTLY its sub-target on disk.

    Keeps the first ``sub_target`` kept problems per stratum (lowest seed = collected
    first, deterministic/reproducible) and ``rmtree``s the rest. Idempotent and a no-op
    when already exact (the common case, since the in-flight cap prevents overshoot in a
    fresh run); it exists to guarantee exact counts under ``--resume`` over a split a
    prior (pre-cap) run overshot, and as a belt-and-suspenders invariant. Returns
    ``{stratum: [surviving problem_ids sorted by seed]}``.
    """
    target_of = dict(zip(strata, sub_targets))
    by_stratum: dict[int, list[tuple[int, str]]] = {s: [] for s in strata}
    if os.path.isdir(split_dir):
        for name in os.listdir(split_dir):
            if not os.path.isdir(os.path.join(split_dir, name)):
                continue
            seed = _seed_from_problem_id(name)
            if seed is None:
                continue
            s = _stratum_of_seed(seed, sub_bands, strata)
            if s is not None:
                by_stratum[s].append((seed, name))
    survivors: dict[int, list[str]] = {}
    for s in strata:
        entries = sorted(by_stratum[s])  # ascending seed
        keep, drop = entries[: target_of[s]], entries[target_of[s] :]
        for _seed, name in drop:
            shutil.rmtree(os.path.join(split_dir, name), ignore_errors=True)
        survivors[s] = [name for _seed, name in keep]
    return survivors


def collect_split(
    split_name: str,
    seed_band: tuple[int, int],
    target: int,
    config: DD2DCollectConfig,
    workers: int,
    out_root: str,
    strata: tuple[int, ...] = STRATA,
    progress: bool = True,
    resume: bool = False,
    progress_every: float = 20.0,
) -> dict:
    """Collect one split until each stratum hits its balanced sub-target, over disjoint
    per-stratum seed sub-bands.

    Records are written worker-side (crops + NNN.json); this coordinator only tallies,
    logs attempts (for ``--resume``), and writes the manifest.
    """
    split_dir = os.path.join(out_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    sub_targets = _stratum_targets(target, len(strata))
    sub_bands = _stratum_bands(seed_band, len(strata))
    skip: set[int] = set()
    resume_kept = {s: 0 for s in strata}
    if resume:
        skip, resume_kept = _load_resume_state(split_dir, sub_bands, strata)

    st_state = {
        s: {
            "target": sub_targets[i],
            "sub_band": list(sub_bands[i]),
            "seeds": iter(range(*sub_bands[i])),
            "kept": resume_kept[s],
            "attempted": 0,
            "n_pos": 0,
            "n_neg": 0,
            "reasons": Counter(),
            "exhausted": False,
            "in_flight": 0,  # tasks submitted but not yet completed (overshoot guard)
        }
        for i, s in enumerate(strata)
    }
    kept_ids: list[str] = []
    seeds_used: list[int] = []
    t0 = time.time()
    last_print = [t0]
    rr = itertools.cycle(strata)
    log_f = open(os.path.join(split_dir, "attempted.log"), "a")

    if progress:
        ttot = sum(v["target"] for v in st_state.values())
        rk = sum(resume_kept.values())
        print(
            f"# [{split_name}] target {target} "
            f"(strata {dict(zip(strata, sub_targets))}) | workers {workers}"
            f"{f' | RESUMING (+{rk} kept, {len(skip)} seeds skipped)' if resume else ''}",
            flush=True,
        )

    def next_task():
        """Round-robin the next open stratum; return (seed, stratum) or None.

        A stratum is "open" only while ``kept + in_flight < target``: counting in-flight
        tasks caps submission so a stratum can never overshoot its sub-target (in-flight
        keeps that complete after the target is otherwise reachable are never spawned),
        and workers freed by a filled stratum flow to the under-target ones.
        """
        for _ in range(len(strata)):
            s = next(rr)
            st = st_state[s]
            if st["kept"] + st["in_flight"] >= st["target"] or st["exhausted"]:
                continue
            while True:
                try:
                    seed = next(st["seeds"])
                except StopIteration:
                    st["exhausted"] = True
                    break
                if seed in skip:  # already attempted/kept on a prior run
                    continue
                return seed, s
        return None

    def _status_line() -> str:
        elapsed = time.time() - t0
        parts, etas = [], []
        for s in strata:
            st = st_state[s]
            parts.append(f"s{s} {st['kept']}/{st['target']}")
            new_kept = st["kept"] - resume_kept[s]  # progress THIS run (rate basis)
            remaining = st["target"] - st["kept"]
            if remaining <= 0:
                continue
            if new_kept > 0:
                etas.append(remaining / (new_kept / elapsed))
        done = sum(v["kept"] for v in st_state.values())
        tot = sum(v["target"] for v in st_state.values())
        att = sum(v["attempted"] for v in st_state.values())
        drop = att - sum(
            max(0, v["kept"] - resume_kept[k]) for k, v in st_state.items()
        )
        run_kept = done - sum(resume_kept.values())
        rate = run_kept / (elapsed / 60) if elapsed > 0 else 0.0
        eta = f"~{_fmt_hms(max(etas))}" if etas else "estimating…"
        return (
            f"  [{split_name}] {_fmt_hms(elapsed)} | kept {done}/{tot}  "
            f"({', '.join(parts)}) | attempted {att} (drop {drop}) | "
            f"{rate:.1f} kept/min | ETA {eta}"
        )

    def absorb(res: ProblemResult) -> None:
        st = st_state[res.stratum]
        st["attempted"] += 1
        st["reasons"][res.reason] += 1
        log_f.write(f"{res.seed},{res.stratum},{int(res.kept)}\n")
        log_f.flush()
        if res.kept:
            st["kept"] += 1
            st["n_pos"] += res.n_pos
            st["n_neg"] += res.n_neg
            kept_ids.append(res.problem_id)
            seeds_used.append(res.seed)
        if progress and time.time() - last_print[0] >= progress_every:
            print(
                _status_line(), flush=True
            )  # flush: sparse output else block-buffers when piped
            last_print[0] = time.time()

    try:
        if workers <= 1:
            while True:
                task = next_task()
                if task is None:
                    break
                seed, stratum = task
                absorb(_collect_task((seed, stratum, config, split_dir)))
        else:
            # Tail-drain observability + robustness (opaque-hang hardening):
            #  - wake every ``progress_every`` s even if nothing completed, so a
            #    long drain of over-prefetched slow tasks is never silent;
            #  - flag any single task running past a generous soft-timeout (still
            #    bounded by the refiner's time_budget/k — informational, never
            #    cancelled), so "is it stuck?" is answerable from the log;
            #  - survive an abnormally-killed worker (OOM / shapely segfault):
            #    a BrokenProcessPool would otherwise abort the whole (multi-hour)
            #    run instead of degrading to the seeds already on disk.
            soft_timeout_s = max(300.0, 5.0 * (config.time_budget or 20.0))
            with ProcessPoolExecutor(max_workers=workers) as pool:
                inflight: dict = {}
                warned: set = set()  # futures already flagged slow (flag once)

                def submit_next() -> bool:
                    task = next_task()
                    if task is None:
                        return False
                    seed, stratum = task
                    fut = pool.submit(_collect_task, (seed, stratum, config, split_dir))
                    inflight[fut] = (seed, stratum, time.time())
                    st_state[stratum]["in_flight"] += 1
                    return True

                for _ in range(workers * 2):
                    if not submit_next():
                        break
                broken = False
                while inflight and not broken:
                    done, _pending = wait(
                        inflight, return_when=FIRST_COMPLETED, timeout=progress_every
                    )
                    if not done:  # heartbeat: nothing finished this interval
                        now = time.time()
                        for fut, (seed, stratum, t_sub) in list(inflight.items()):
                            age = now - t_sub
                            if age >= soft_timeout_s and fut not in warned:
                                warned.add(fut)
                                print(
                                    f"  [{split_name}] SLOW seed={seed} s{stratum} "
                                    f"running {age:.0f}s (>{soft_timeout_s:.0f}s); still "
                                    f"bounded by refiner time_budget/k, not cancelled",
                                    flush=True,
                                )
                        if progress:
                            oldest = max(now - t for _, _, t in inflight.values())
                            print(
                                f"  [{split_name}] {_fmt_hms(now - t0)} | draining "
                                f"{len(inflight)} in-flight, none completed in "
                                f"{progress_every:.0f}s | oldest {oldest:.0f}s",
                                flush=True,
                            )
                            last_print[0] = now
                        continue
                    for fut in done:
                        seed, stratum, _t_sub = inflight.pop(fut)
                        st_state[stratum][
                            "in_flight"
                        ] -= 1  # free the slot before refill
                        try:
                            res = fut.result()
                        except BrokenProcessPool as e:
                            # Pool is dead: this seed + every other in-flight seed
                            # are lost. Log, record synthetic drops so the manifest
                            # reflects them, and finalize the split (--resume picks
                            # up the rest) rather than crashing the whole run.
                            print(
                                f"  [{split_name}] WORKER DIED seed={seed} s{stratum} "
                                f"({type(e).__name__}); finalizing split with "
                                f"{sum(v['kept'] for v in st_state.values())} kept so "
                                f"far, {len(inflight)} in-flight seeds abandoned "
                                f"(re-run with --resume to continue)",
                                flush=True,
                            )
                            for lost_seed, lost_stratum, _ in [
                                (seed, stratum, _t_sub)
                            ] + list(inflight.values()):
                                absorb(
                                    ProblemResult(
                                        problem_id=f"dd2d_s{lost_seed}_st{lost_stratum}",
                                        seed=lost_seed,
                                        stratum=lost_stratum,
                                        n_items=_sample_n_items(
                                            lost_seed, config.n_items_range
                                        ),
                                        kept=False,
                                        reason=f"error:{type(e).__name__}",
                                    )
                                )
                            inflight.clear()
                            broken = True
                            break
                        absorb(res)
                        submit_next()
    finally:
        log_f.close()

    # Guarantee exactly the sub-target per stratum: drop any overshoot from disk and
    # re-derive the kept tallies/ids from the surviving dirs so the manifest is exact.
    survivors = _truncate_to_targets(split_dir, sub_bands, sub_targets, strata)
    for s in strata:
        st_state[s]["kept"] = len(survivors[s])
    kept_ids = [pid for s in strata for pid in survivors[s]]
    seeds_used = [
        sd for sd in (_seed_from_problem_id(pid) for pid in kept_ids) if sd is not None
    ]

    return _write_manifest(
        split_dir,
        split_name,
        seed_band,
        config,
        st_state,
        kept_ids,
        seeds_used,
        t0,
        progress,
    )


def _ratio(n_neg: int, n_pos: int) -> float:
    return round(n_neg / n_pos, 3) if n_pos else 0.0


def _write_manifest(
    split_dir,
    split_name,
    seed_band,
    config,
    st_state,
    kept_ids,
    seeds_used,
    t0,
    progress,
) -> dict:
    strata_summary = {
        str(s): {
            "sub_target": st["target"],
            "kept": st["kept"],
            "attempted": st["attempted"],
            "n_pos": st["n_pos"],
            "n_neg": st["n_neg"],
            "neg_pos_ratio": _ratio(st["n_neg"], st["n_pos"]),
            "seed_subband": st["sub_band"],
            "exhausted": st["exhausted"],
            "reasons": dict(st["reasons"]),
        }
        for s, st in st_state.items()
    }
    tot_kept = sum(st["kept"] for st in st_state.values())
    tot_att = sum(st["attempted"] for st in st_state.values())
    tot_pos = sum(st["n_pos"] for st in st_state.values())
    tot_neg = sum(st["n_neg"] for st in st_state.values())
    summary = {
        "split": split_name,
        "seed_band": list(seed_band),
        "config": asdict(config),
        "strata": strata_summary,
        "overall": {
            "kept": tot_kept,
            "attempted": tot_att,
            "n_pos": tot_pos,
            "n_neg": tot_neg,
            "neg_pos_ratio": _ratio(tot_neg, tot_pos),
        },
        "wall_min": round((time.time() - t0) / 60, 2),
        "problem_ids": kept_ids,
        "seeds_used": seeds_used,
    }
    _atomic_write(
        os.path.join(split_dir, "manifest.json"), json.dumps(summary, indent=2)
    )
    if progress:
        under = [s for s, st in st_state.items() if st["kept"] < st["target"]]
        warn = f"  UNDER-TARGET strata {under} (band exhausted)" if under else ""
        print(
            f"# [{split_name}] done: kept {tot_kept} "
            f"(attempted {tot_att}), neg:pos={summary['overall']['neg_pos_ratio']} "
            f"-> {split_dir}/{warn}",
            flush=True,
        )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="DD2D PIGINet dataset collector (balanced min-subset strata, parallel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-root", default=os.path.join("data", "dd2d", "raw"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-train", type=int, default=400)
    ap.add_argument("--target-test", type=int, default=100)
    ap.add_argument("--target-val", type=int, default=100)
    ap.add_argument("--splits", default="train,test,val")
    ap.add_argument("--band", type=int, default=1_000_000)
    # locked-config overrides (default to the locked values)
    ap.add_argument("--crowd", type=int, default=5)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.8)
    ap.add_argument("--time-budget", type=float, default=20.0)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run: skip already-kept + logged-attempted seeds",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny targets (1/stratum) for a real-path plumbing check",
    )
    # Held-out generalization set (docs/decisions 2026-08-01). All default to the standard
    # collection; a held-out run sets a fresh band base, an unseen item count, and (for the
    # shape set) the augmented pool + forced families.
    ap.add_argument(
        "--seed-band-base",
        type=int,
        default=None,
        help="override the TEST split's seed band to [N*band, (N+1)*band) for a held-out "
        "set disjoint from train/val/test (use 3, 4, ...; keep --band=1_000_000 so "
        "compare.stratum_of stays valid)",
    )
    ap.add_argument("--n-items-min", type=int, default=None)
    ap.add_argument("--n-items-max", type=int, default=None)
    ap.add_argument(
        "--shape-set",
        choices=("base", "augmented"),
        default="base",
        help="'augmented' adds the held-out tee/cross families to the clutter/collar pool",
    )
    ap.add_argument(
        "--require-families",
        default="",
        help="comma-separated families to force >=1 of into every scene (e.g. tee,cross)",
    )
    ap.add_argument("--fill-max", type=float, default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        args.target_train = args.target_test = args.target_val = 3  # -> 1 per stratum
        args.workers = min(args.workers, 2)

    n_items_range = None
    if args.n_items_min is not None or args.n_items_max is not None:
        if args.n_items_min is None or args.n_items_max is None:
            ap.error("--n-items-min and --n-items-max must be given together")
        n_items_range = (args.n_items_min, args.n_items_max)
    extra_families = None
    if args.shape_set == "augmented":
        from .shapes import (  # lazy: avoid a module-level shapes import
            NEW_SHAPE_WEIGHTS,
        )

        extra_families = dict(NEW_SHAPE_WEIGHTS)
    require_families = tuple(
        f.strip() for f in args.require_families.split(",") if f.strip()
    )

    config = DD2DCollectConfig(
        crowd=args.crowd,
        lam=args.lam,
        time_budget=args.time_budget,
        n_items_range=n_items_range,
        require_families=require_families,
        extra_families=extra_families,
        fill_max=args.fill_max,
    )
    bands = _split_bands(args.band)
    if args.seed_band_base is not None:
        base = args.seed_band_base
        bands["test"] = (base * args.band, (base + 1) * args.band)
    targets = {
        "train": args.target_train,
        "test": args.target_test,
        "val": args.target_val,
    }
    selected = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(
        f"# Collecting DD2D splits {selected} -> {args.out_root} "
        f"(workers={args.workers}, crowd={config.crowd}, lambda={config.lam}, "
        f"time_budget={config.time_budget}s, k={config.k}, full_pool={config.full_pool}, "
        f"strata={STRATA}, resume={args.resume})"
    )
    if (
        config.n_items_range
        or config.require_families
        or config.extra_families
        or config.fill_max is not None
        or args.seed_band_base is not None
    ):
        print(
            f"# HELD-OUT generalization set: seed_band(test)={bands.get('test')}, "
            f"n_items_range={config.n_items_range}, shape_set={args.shape_set}, "
            f"require_families={list(config.require_families)}, fill_max={config.fill_max}",
            flush=True,
        )
    print(
        "# full_pool=True: every problem refines ALL k plans -> many pos+neg per problem "
        "(no length confound). Strata {0,1,2,3}. Live 'ETA ~…' is authoritative.",
        flush=True,
    )
    for split_name in selected:
        collect_split(
            split_name,
            bands[split_name],
            targets[split_name],
            config,
            args.workers,
            args.out_root,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
