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
    ap.add_argument("--out-dir", default="out_dd2d")
    args = ap.parse_args(argv)

    # --min-subset implies --require-subset; --require-subset alone means floor 2.
    require_subset = args.require_subset or args.min_subset is not None
    min_subset = args.min_subset if args.min_subset is not None else 2

    _silence_up_credits()
    planner_kwargs = {}
    if args.planner == "pyperplan":
        planner_kwargs["length_slack"] = (
            None if args.pyperplan_slack == "none" else int(args.pyperplan_slack)
        )
        planner_kwargs["search"] = args.pyperplan_search
        if args.pyperplan_search != "bfs":
            planner_kwargs["heuristic"] = args.pyperplan_heuristic
    planner = make_dd2d_planner(prefer=args.planner, order=args.order, **planner_kwargs)
    refiner = DD2DRefiner(
        budget=args.max_stream_calls,
        retry_cap=args.retry_cap,
        samples_per_step=args.samples_per_step,
        time_budget=args.time_budget,
    )
    render_scene, render_episode, backend_cls = (
        _load_render() if args.max_videos >= 0 else (None, None, None)
    )
    ds_dir = os.path.join(args.out_dir, "dataset")
    vid_dir = os.path.join(args.out_dir, "videos")
    os.makedirs(ds_dir, exist_ok=True)
    print(
        f"# Problem type: dd2d (drawer decluttering)  | lambda={args.lam}  order={args.order}  "
        f"planner={planner.name}  crowd={args.crowd}{' (diverse)' if args.diverse_crowd else ''}"
        f"{f'  require-subset(>={min_subset})' if require_subset else ''}"
    )
    print(
        f"# Refiner: {refiner.name} ({refiner.label_source})  | "
        f"budget={'uncapped' if refiner.budget is None else refiner.budget} stream calls, "
        f"retry_cap={refiner.retry_cap}, samples_per_step={refiner.samples_per_step}, "
        f"time_budget={'none' if refiner.time_budget is None else f'{refiner.time_budget}s'}"
    )

    grand_feasible = grand_total = grand_calls = 0
    n_require_subset = 0
    ranks: list[int | None] = (
        []
    )  # rank of first feasible plan per problem (baseline cost)
    last_scene = None
    for pi in range(args.num_problems):
        seed = args.seed + pi
        problem = generate_dd2d_problem(
            lam=args.lam,
            seed=seed,
            margin=args.margin,
            split=args.split,
            n_items=args.num_items,
            crowd=args.crowd,
            diverse_crowd=args.diverse_crowd,
            require_subset=require_subset,
            min_subset=min_subset,
            certify=not args.no_certify,
            budget=args.max_stream_calls,
            retry_cap=args.retry_cap,
            samples_per_step=args.samples_per_step,
            time_budget=args.time_budget,
        )
        last_scene = problem.scene
        n_require_subset += problem.requires_subset
        skeletons = planner.plan(problem, args.k)
        mfs = problem.min_feasible_subset
        subset_tag = (
            f"REQUIRES a {mfs}-blocker SUBSET"
            if problem.requires_subset
            else f"solvable by 1 object (min feasible subset = {mfs})"
        )
        print(
            f"\n# Problem {problem.problem_id}  ({len(skeletons)} diverse skeletons via {planner.name}; "
            f"{len(problem.feasible_candidates())}/{len(problem.candidates)} candidates labeled feasible) "
            f"-> {subset_tag}"
        )

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
            print(
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
                    vid_dir,
                    f"{problem.problem_id}_plan{i:02d}_{tag}.{args.video_format}",
                )
                written = render_episode(
                    problem.scene,
                    res.bound_plan,
                    res.feasible,
                    res.failure_action,
                    path,
                    fmt=args.video_format,
                )
                print(f"  video[plan{i:02d} {tag}] -> {written}")

        first_feasible = next((i for i, r in enumerate(results) if r.feasible), None)
        ranks.append(first_feasible)
        baseline = (
            f"first feasible at rank {first_feasible + 1}/{len(skeletons)} refined"
            if first_feasible is not None
            else f"NO feasible plan in {len(skeletons)} refined"
        )
        tot = sum(calls_list)
        print(
            f"  -> {n_feasible}/{len(skeletons)} feasible | baseline: {baseline} | stream calls: total={tot}, "
            f"mean={tot/max(len(calls_list),1):.0f}, min={min(calls_list, default=0)}, max={max(calls_list, default=0)}"
        )
        grand_feasible += n_feasible
        grand_total += len(skeletons)
        grand_calls += tot

    print(
        f"\n# SUMMARY over {args.num_problems} problem(s): "
        f"{grand_feasible}/{grand_total} skeletons feasible | {grand_calls} total stream calls "
        f"({grand_calls/max(grand_total,1):.0f} per skeleton)"
    )
    print(
        f"# {n_require_subset}/{args.num_problems} problems REQUIRED a 2+ blocker subset "
        f"({n_require_subset/max(args.num_problems,1):.0%}) "
        f"[crowd={args.crowd}{' diverse' if args.diverse_crowd else ''}"
        f"{f', require-subset(>={min_subset})' if require_subset else ''}]"
    )
    solved = [r for r in ranks if r is not None]
    mean_rank = sum(r + 1 for r in solved) / len(solved) if solved else 0.0
    print(
        f"# BASELINE [{planner.name}]: solved {len(solved)}/{args.num_problems} within k={args.k}"
        f"{f'; mean first-feasible rank {mean_rank:.1f}' if solved else ''} "
        f"(vs. the geometry-informed 'candidates' planner, which ranks the feasible plan near the top)"
    )
    print(f"# Wrote {grand_total} PIGINetExample records to {ds_dir}/")
    if render_episode is not None and args.max_videos > 0:
        print(f"# Wrote episode videos to {vid_dir}/")

    if last_scene is not None and render_scene is not None:
        chk = confirm_rendering(last_scene, backend=backend_cls(), out_dir=args.out_dir)
        print(
            f"# Render confirmation [{'OK' if chk.ok else 'FAILED'}]: backend={chk.backend}, "
            f"{chk.n_segments} object segments -> {chk.png_path}"
        )
    return 0


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
