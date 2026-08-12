"""End-to-end DD2D demo: drawer-decluttering problem(s) -> diverse staging skeletons ->
real backjumping refinement -> PIGINet records + per-plan execution videos (+ a
render-confirmation frame). Mirrors ``blocks_tamp/demo.py`` with DD2D knobs.

    python -m blocks_tamp.dd2d.demo --num-items 11 --lambda 0.8 --seed 0 --num-problems 2 --max-videos 4
    python -m blocks_tamp.dd2d.demo --lambda 0.6 --seed 3 --order slack   # tighter buffer, slack ordering

It "attempts to declutter a drawer": prints the diverse staging plans (which blocker subset
to evict), refines each with the real grasp+packing refiner -- printing stream calls each
took -- writes a dataset of PIGINetExample JSONs, and renders a video per plan (full
retrieval when feasible, partial up to the failing action when not, with the elevated-carry
convention). ``--num-problems N`` runs N consecutive seeds (distinct problems).

The buffer scale ``--lambda`` is the difficulty dial (spec P4): smaller = tighter buffer =
more buffer-overflow (joint packing) failures; the interesting regime is ~[0.75, 0.9].
"""

from __future__ import annotations

import argparse
import os
import random

from ..record import build_example, build_image_refs
from ..rendering import confirm_rendering
from .planning import make_dd2d_planner
from .problem import generate_dd2d_problem
from .refine import DD2DRefiner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--num-items",
        type=int,
        default=None,
        help="items incl. target (default: sampled 9-14)",
    )
    ap.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.8,
        help="buffer scale (smaller=tighter=harder)",
    )
    ap.add_argument("--margin", type=float, default=1.0, help="label margin delta (cm)")
    ap.add_argument("--split", choices=["train", "holdout"], default="train")
    ap.add_argument(
        "--k", type=int, default=12, help="diverse staging skeletons to enumerate"
    )
    ap.add_argument("--seed", type=int, default=0, help="first problem seed")
    ap.add_argument(
        "--num-problems", type=int, default=1, help="run this many consecutive seeds"
    )
    ap.add_argument(
        "--order",
        choices=["published", "random", "slack", "oracle"],
        default="published",
    )
    ap.add_argument(
        "--planner",
        choices=["candidates", "symk", "pyperplan"],
        default="candidates",
        help="'candidates' (default) is geometry-informed; 'pyperplan'/'symk' are the "
        "geometry-blind standard baselines. pyperplan defaults to unbounded depth "
        "(k-driven) -- a subset-required instance needs a large --k (~n_blockers^subset) "
        "to reach the feasible plan; that gap is the intended baseline comparison",
    )
    ap.add_argument(
        "--pyperplan-slack",
        default="none",
        help="[pyperplan] plan-length budget: 'none' (default) = unbounded/k-driven fair "
        "baseline, or an int = cap plans at shortest+slack",
    )
    ap.add_argument(
        "--pyperplan-search",
        choices=["bfs", "gbf", "astar"],
        default="bfs",
        help="[pyperplan] enumeration frontier: 'bfs' (blind ascending-length baseline) "
        "or 'gbf'/'astar' (best-first ordered by --pyperplan-heuristic). See "
        "blocks_tamp/dd2d/heuristic_experiment.py",
    )
    ap.add_argument(
        "--pyperplan-heuristic",
        choices=["hff", "hadd", "dist", "dist-avg", "dist-radius"],
        default="hff",
        help="[pyperplan, gbf/astar only] heuristic: 'hff'/'hadd' (off-the-shelf, "
        "geometry-blind) or 'dist' (hand-written geometric distance prior)",
    )
    ap.add_argument(
        "--crowd",
        type=int,
        default=10,
        help="collar crowding prior: pincer the target so ~half the problems require a 2+ "
        "blocker SUBSET (0 = naturalistic baseline; see docs/dd2d.md)",
    )
    ap.add_argument(
        "--diverse-crowd",
        dest="diverse_crowd",
        action="store_true",
        help="draw collar items from ALL families, not just round ones, so concave shapes "
        "join the pincer (default: round-only). Tends to lower the subset rate; "
        "pair with --require-subset to keep it high",
    )
    ap.add_argument(
        "--require-subset",
        action="store_true",
        help="only keep problems that need a 2+ blocker clearing subset "
        "(equivalent to --min-subset 2; --min-subset sets a higher floor)",
    )
    ap.add_argument(
        "--min-subset",
        type=int,
        default=None,
        help="only keep problems whose smallest feasible clearing subset is >= this "
        "(IMPLIES --require-subset; --require-subset alone uses a floor of 2)",
    )
    ap.add_argument(
        "--stratum",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="pin the EXACT stratum = min feasible clearing subset size: 0 = target "
        "directly graspable (retrieve-only), 1 = one removal suffices, 2/3 = a 2-/3-blocker "
        "subset is required. Overrides --require-subset/--min-subset. Problems are "
        "resampled (advancing the seed) until num_problems of exactly this stratum are found",
    )
    ap.add_argument(
        "--min-blockers",
        type=int,
        default=None,
        help="sample the blocker count (non-target items) per problem from "
        "[min-blockers, max-blockers] instead of --num-items (total items = blockers + 1)",
    )
    ap.add_argument(
        "--max-blockers",
        type=int,
        default=None,
        help="upper end of the per-problem blocker-count range (see --min-blockers)",
    )
    # refiner budget knobs (spec P13/P14/P15) -- tune the refinement cost model for demos/collection
    ap.add_argument(
        "--max-stream-calls",
        dest="max_stream_calls",
        type=int,
        default=300,
        help="refiner cap on TOTAL stream calls per skeleton (spec's B); <=0 = no call cap "
        "(combine with --time-budget / --retry-cap to govern by time + per-step instead)",
    )
    ap.add_argument(
        "--retry-cap",
        type=int,
        default=10,
        help="sample-buffer-pose CALLS per place-buffer step before backjump (t)",
    )
    ap.add_argument(
        "--samples-per-step",
        type=int,
        default=15,
        help="candidate poses tried INSIDE one sample-buffer-pose call (m_p, sampler strength)",
    )
    ap.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="wall-clock seconds per plan (default: unbounded); stops refinement when reached",
    )
    ap.add_argument(
        "--no-certify",
        action="store_true",
        help="skip generation-time certification (faster)",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=6,
        help="render up to N plan clips per problem, ALWAYS including the first feasible plan",
    )
    ap.add_argument("--video-format", choices=["mp4", "gif"], default="mp4")
    ap.add_argument("--out-dir", default="data/dd2d/out_dd2d")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process problems in parallel across N worker processes (problems are "
        "independent + deterministic per slot, so results are worker-count-invariant). "
        "1 = serial",
    )
    args = ap.parse_args(argv)

    # --min-subset implies --require-subset; --require-subset alone means floor 2.
    require_subset = args.require_subset or args.min_subset is not None
    min_subset = args.min_subset if args.min_subset is not None else 2

    _silence_up_credits()
    os.makedirs(os.path.join(args.out_dir, "dataset"), exist_ok=True)
    print(
        f"# Problem type: dd2d (drawer decluttering)  | lambda={args.lam}  order={args.order}  "
        f"planner={_build_planner(args).name}  crowd={args.crowd}"
        f"{' (diverse)' if args.diverse_crowd else ''}"
        f"{f'  stratum={args.stratum}' if args.stratum is not None else ''}"
        f"{f'  require-subset(>={min_subset})' if require_subset and args.stratum is None else ''}"
        f"  workers={args.workers}"
    )
    _r = _build_refiner(args)
    print(
        f"# Refiner: {_r.name} ({_r.label_source})  | "
        f"budget={'uncapped' if _r.budget is None else _r.budget} stream calls, "
        f"retry_cap={_r.retry_cap}, samples_per_step={_r.samples_per_step}, "
        f"time_budget={'none' if _r.time_budget is None else f'{_r.time_budget}s'}"
    )

    grand_feasible = grand_total = grand_calls = 0
    n_require_subset = 0
    ranks: list[int | None] = []  # first-feasible rank per problem (baseline cost)
    strata_seen: dict[int, int] = {}
    render_note: str | None = None

    # one work item per problem slot; slots use disjoint seed spaces so results do not
    # depend on the worker count (a slot finds the same problem serial or parallel).
    slots = list(range(args.num_problems))
    stats_list: list[dict] = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _process_problem, args, s, require_subset, min_subset, s == 0
                ): s
                for s in slots
            }
            for fut in as_completed(futs):
                st = fut.result()
                if st.get("log"):
                    print(st["log"], flush=True)
                stats_list.append(st)
    else:
        for s in slots:
            st = _process_problem(args, s, require_subset, min_subset, s == 0)
            if st.get("log"):
                print(st["log"], flush=True)
            stats_list.append(st)

    for st in sorted(stats_list, key=lambda d: d["slot"]):
        if not st.get("ok"):
            continue
        ranks.append(st["first_feasible"])
        n_require_subset += st["requires_subset"]
        strata_seen[st["mfs"]] = strata_seen.get(st["mfs"], 0) + 1
        grand_feasible += st["n_feasible"]
        grand_total += st["n_skeletons"]
        grand_calls += st["stream_calls"]
        if st.get("render_note"):
            render_note = st["render_note"]

    ds_dir = os.path.join(args.out_dir, "dataset")
    vid_dir = os.path.join(args.out_dir, "videos")
    n_problems = len(ranks)  # problems actually produced (may be < requested if a
    # stratum could not be hit within the resample budget)
    shortfall = (
        f" (requested {args.num_problems}; the stratum filter/resample budget was the limit)"
        if n_problems < args.num_problems
        else ""
    )
    print(
        f"\n# SUMMARY over {n_problems} problem(s){shortfall}: "
        f"{grand_feasible}/{grand_total} skeletons feasible | {grand_calls} total stream calls "
        f"({grand_calls/max(grand_total,1):.0f} per skeleton)"
    )
    print(
        f"# {n_require_subset}/{n_problems} problems REQUIRED a 2+ blocker subset "
        f"({n_require_subset/max(n_problems,1):.0%}) "
        f"[crowd={args.crowd}{' diverse' if args.diverse_crowd else ''}"
        f"{f', stratum={args.stratum}' if args.stratum is not None else ''}"
        f"{f', require-subset(>={min_subset})' if require_subset and args.stratum is None else ''}]"
    )
    print(
        "# Strata seen (min feasible subset -> count): "
        + ", ".join(f"s{k}:{strata_seen[k]}" for k in sorted(strata_seen))
    )
    solved = [r for r in ranks if r is not None]
    mean_rank = sum(r + 1 for r in solved) / len(solved) if solved else 0.0
    print(
        f"# BASELINE [{_build_planner(args).name}]: solved {len(solved)}/{n_problems} "
        f"within k={args.k}"
        f"{f'; mean first-feasible rank {mean_rank:.1f}' if solved else ''} "
        f"(vs. the geometry-informed 'candidates' planner, which ranks the feasible plan near the top)"
    )
    print(f"# Wrote {grand_total} PIGINetExample records to {ds_dir}/")
    if args.max_videos > 0:
        print(f"# Wrote episode videos to {vid_dir}/")
    if render_note:
        print(render_note)
    return 0


def _sample_n_items(args, rng: random.Random) -> int | None:
    """Total item count (blockers + target) for the next problem: from the
    ``--min/max-blockers`` range if set, else ``--num-items`` (or ``None`` = generator
    default 9-14)."""
    if args.min_blockers is not None or args.max_blockers is not None:
        lo = args.min_blockers if args.min_blockers is not None else args.max_blockers
        hi = args.max_blockers if args.max_blockers is not None else args.min_blockers
        return 1 + rng.randint(min(lo, hi), max(lo, hi))
    return args.num_items


def _generate_one(args, seed, n_items, require_subset, min_subset):
    """One ``generate_dd2d_problem`` call with the stratum knob applied.

    ``--stratum`` maps to the generator's exact-stratum controls: 0 -> ``unblocked_target``
    (retrieve-only feasible), 2/3 -> a ``min_subset`` floor (the caller filters to the
    exact size), 1 / None -> the naturalistic path (require-subset knobs as given).
    """
    kw = dict(
        lam=args.lam,
        seed=seed,
        margin=args.margin,
        split=args.split,
        n_items=n_items,
        crowd=args.crowd,
        diverse_crowd=args.diverse_crowd,
        certify=not args.no_certify,
        budget=args.max_stream_calls,
        retry_cap=args.retry_cap,
        samples_per_step=args.samples_per_step,
        time_budget=args.time_budget,
    )
    if args.stratum == 0:
        kw["unblocked_target"] = True
    elif args.stratum is not None and args.stratum >= 2:
        kw["require_subset"] = True
        kw["min_subset"] = args.stratum
    else:  # stratum == 1 or None: naturalistic (optionally require-subset)
        kw["require_subset"] = require_subset
        kw["min_subset"] = min_subset
    return generate_dd2d_problem(**kw)


# disjoint seed budget per problem slot; a slot never advances into the next slot's
# base, so a slot finds the same problem regardless of how many run concurrently.
_SEED_STRIDE = 10_000


def _build_planner(args):
    """The planner selected by ``--planner`` (+ pyperplan search/heuristic knobs)."""
    planner_kwargs = {}
    if args.planner == "pyperplan":
        planner_kwargs["length_slack"] = (
            None if args.pyperplan_slack == "none" else int(args.pyperplan_slack)
        )
        planner_kwargs["search"] = args.pyperplan_search
        if args.pyperplan_search != "bfs":
            planner_kwargs["heuristic"] = args.pyperplan_heuristic
    return make_dd2d_planner(prefer=args.planner, order=args.order, **planner_kwargs)


def _build_refiner(args):
    return DD2DRefiner(
        budget=args.max_stream_calls,
        retry_cap=args.retry_cap,
        samples_per_step=args.samples_per_step,
        time_budget=args.time_budget,
    )


def _find_problem(args, slot, require_subset, min_subset):
    """The one stratum-matching problem for this ``slot``, from its disjoint seed space.

    Deterministic in ``slot`` (not in completion order), so parallel and serial runs
    produce the same set of problems. Returns ``None`` if the stratum could not be hit
    within the slot's seed budget (reported as a shortfall)."""
    n_rng = random.Random(
        (args.seed * 2_654_435_761 + 0x9E37 + slot * 7919) & 0xFFFFFFFF
    )
    base = args.seed + slot * _SEED_STRIDE
    for off in range(_SEED_STRIDE):
        n_items = _sample_n_items(args, n_rng)
        try:
            problem = _generate_one(
                args, base + off, n_items, require_subset, min_subset
            )
        except RuntimeError:
            continue
        if args.stratum is None or problem.min_feasible_subset == args.stratum:
            return problem
    return None


def _process_problem(args, slot, require_subset, min_subset, do_confirm):
    """Find + fully process one problem slot (generate, plan, refine every skeleton,
    write records + videos). Returns a picklable stats dict; runs in a worker process
    when ``--workers>1``. All heavy objects are built here (never pickled across the
    pool)."""
    _silence_up_credits()
    problem = _find_problem(args, slot, require_subset, min_subset)
    if problem is None:
        return {
            "ok": False,
            "slot": slot,
            "log": f"# slot {slot}: no stratum-{args.stratum} problem within seed budget",
        }
    planner = _build_planner(args)
    refiner = _build_refiner(args)
    render_scene, render_episode, backend_cls = (
        _load_render() if args.max_videos >= 0 else (None, None, None)
    )
    ds_dir = os.path.join(args.out_dir, "dataset")
    vid_dir = os.path.join(args.out_dir, "videos")

    skeletons = planner.plan(problem, args.k)
    mfs = problem.min_feasible_subset
    subset_tag = (
        f"REQUIRES a {mfs}-blocker SUBSET"
        if problem.requires_subset
        else f"solvable by 1 object (min feasible subset = {mfs})"
    )
    lines = [
        f"\n# Problem {problem.problem_id}  ({len(skeletons)} diverse skeletons via "
        f"{planner.name}; {len(problem.feasible_candidates())}/{len(problem.candidates)} "
        f"candidates labeled feasible) -> {subset_tag}"
    ]
    render = render_scene(problem.scene) if render_scene else None
    results = []
    n_feasible = 0
    calls_list = []
    for i, sk in enumerate(skeletons):
        res = refiner.refine(sk, problem.scene, seed=1000 + i)
        results.append(res)
        n_feasible += res.feasible
        calls_list.append(res.n_attempts)
        flag = "FEASIBLE  " if res.feasible else "infeasible"
        detail = (
            ""
            if res.feasible
            else f"  (stuck@ {res.failure_action}, {res.steps_bound}/{res.plan_length})"
        )
        lines.append(
            f"  [{flag}] len {sk.length:>2}  calls {res.n_attempts:>3}  {sk}{detail}"
        )
        imgs = build_image_refs(problem, render=render, views=("topdown",))
        ex = build_example(
            problem,
            sk,
            res,
            planner.name,
            images=imgs,
            label_source=refiner.label_source,
            extra_provenance={"refiner": refiner.name},
        )
        ex.save(os.path.join(ds_dir, f"{problem.problem_id}_plan{i:02d}.json"))

    if render_episode is not None and args.max_videos > 0:
        for i in _select_video_indices(results, args.max_videos):
            res = results[i]
            tag = "success" if res.feasible else "failure"
            path = os.path.join(
                vid_dir, f"{problem.problem_id}_plan{i:02d}_{tag}.{args.video_format}"
            )
            written = render_episode(
                problem.scene,
                res.bound_plan,
                res.feasible,
                res.failure_action,
                path,
                fmt=args.video_format,
            )
            lines.append(f"  video[plan{i:02d} {tag}] -> {written}")

    first_feasible = next((i for i, r in enumerate(results) if r.feasible), None)
    baseline = (
        f"first feasible at rank {first_feasible + 1}/{len(skeletons)} refined"
        if first_feasible is not None
        else f"NO feasible plan in {len(skeletons)} refined"
    )
    tot = sum(calls_list)
    lines.append(
        f"  -> {n_feasible}/{len(skeletons)} feasible | baseline: {baseline} | stream calls: "
        f"total={tot}, mean={tot/max(len(calls_list),1):.0f}, "
        f"min={min(calls_list, default=0)}, max={max(calls_list, default=0)}"
    )

    render_note = None
    if do_confirm and render_scene is not None:
        chk = confirm_rendering(
            problem.scene, backend=backend_cls(), out_dir=args.out_dir
        )
        render_note = (
            f"# Render confirmation [{'OK' if chk.ok else 'FAILED'}]: backend={chk.backend}, "
            f"{chk.n_segments} object segments -> {chk.png_path}"
        )
    return {
        "ok": True,
        "slot": slot,
        "mfs": mfs,
        "requires_subset": problem.requires_subset,
        "n_feasible": n_feasible,
        "n_skeletons": len(skeletons),
        "first_feasible": first_feasible,
        "stream_calls": tot,
        "render_note": render_note,
        "log": "\n".join(lines),
    }


def _select_video_indices(results, max_videos: int) -> list[int]:
    """Always include the first feasible plan (the solution); fill remaining slots with
    the earliest plans (typically the leading failures).

    Returns sorted indices.
    """
    if max_videos <= 0 or not results:
        return []
    chosen: list[int] = []
    first_feasible = next((i for i, r in enumerate(results) if r.feasible), None)
    if first_feasible is not None:
        chosen.append(first_feasible)
    for i in range(len(results)):
        if len(chosen) >= max_videos:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen[:max_videos])


def _load_render():
    from .render import DD2DRenderBackend, render_episode, render_scene

    return render_scene, render_episode, DD2DRenderBackend


def _silence_up_credits() -> None:
    try:
        from unified_planning.shortcuts import get_environment

        get_environment().credits_stream = None
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
