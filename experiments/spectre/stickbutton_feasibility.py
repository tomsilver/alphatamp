"""StickButton2D dataset-feasibility harness.

Answers one question before any collection compute is spent: **can StickButton2D yield a
dataset that is not all-negative?** ``dataset.py`` drops episodes with
``num_success == 0``, so a variant whose problems never refine is unusable no matter how
good the model is.

Two modes:

``probe`` (default, cheap — ~2N refinements per problem)
    Runs the per-button achievability diagnostic
    (``envs/stickbutton2d/diagnostics.py``). Good at attributing *why* refinement fails
    (out-of-reach vs a stuck controller vs extra atoms). **It is not a bound on the
    episode success rate in either direction** — measured against `full` mode it
    under-estimated b2/b3 (55%/35% vs a true 100%) and over-estimated b5. Do not size a
    collection from it.

``full`` (expensive — up to ``k_max`` refinements per problem)
    Draws the real pool with the geometry-aware generator and refines it in order,
    reporting the first-success index. This is the ground truth the probe approximates.
    Measured cost anchor: b5 at 200 candidates took ~380 s per problem.

Both modes run problems concurrently — problems are independent, and serial runs leave
both the GPU and ~30 CPU threads idle (spectre ``CLAUDE.md``, "Use the hardware").
``spawn`` is required: pyperplan and bilevel_planning keep module-level caches that do
not survive a concurrent ``fork``.

Examples::

    python experiments/spectre/stickbutton_feasibility.py \
        --variants 1,2,3,5,10 --problems 20
    python experiments/spectre/stickbutton_feasibility.py \
        --variants 3,5 --problems 10 --mode full
"""

from __future__ import annotations

import argparse
import collections
import itertools
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

if TYPE_CHECKING:  # heavy substrate imports, only needed for annotations
    from bilevel_planning.structs import SesameModels

    from alphatamp.approaches.spectre.envs.stickbutton2d.sampler import Acceptance

_HEARTBEAT_S = 30.0


@dataclass(frozen=True)
class ProblemOutcome:
    """One problem's result, in whichever mode was run."""

    num_buttons: int
    problem_id: int
    solvable: bool
    num_achievable: int
    first_success_idx: int | None
    num_success: int
    pool_size: int
    wall_clock_s: float
    blockers: tuple[str, ...]
    failure_modes: tuple[str, ...]


def _build_models(
    num_buttons: int, problem_id: int
) -> tuple[Any, "SesameModels[Any, Any, Any]", Any]:
    """Make the env, its SesameModels and the concrete initial state for one problem."""
    import kinder  # pylint: disable=import-outside-toplevel
    from kinder_bilevel_planning.env_models import (  # pylint: disable=import-outside-toplevel
        create_bilevel_planning_models,
    )

    kinder.register_all_environments()
    env = kinder.make(f"kinder/StickButton2D-b{num_buttons}-v0")
    obs, _ = env.reset(seed=problem_id)
    env_models = create_bilevel_planning_models(
        "stickbutton2d",
        env.observation_space,
        env.action_space,
        num_buttons=num_buttons,
    )
    x0 = env_models.observation_to_state(obs)
    return env, env_models, x0


def _run_probe(args: tuple[int, int, dict[str, float]]) -> ProblemOutcome:
    """Worker: cheap per-button achievability probe."""
    num_buttons, problem_id, budgets = args
    acceptance: Acceptance = "superset" if budgets.get("superset", 0.0) else "exact"
    from alphatamp.approaches.spectre.envs.stickbutton2d.diagnostics import (  # pylint: disable=import-outside-toplevel
        probe_problem,
    )

    start = time.perf_counter()
    env, env_models, x0 = _build_models(num_buttons, problem_id)
    try:
        probe = probe_problem(
            env_models,
            x0,
            problem_id,
            num_sampling_attempts_per_step=int(budgets["samples"]),
            max_trajectory_steps=int(budgets["horizon"]),
            timeout_s=float(budgets["timeout"]),
            acceptance=acceptance,
        )
    finally:
        env.close()

    modes: list[str] = []
    for button in probe.blockers():
        modes.append(f"robot:{button.robot.mode}")
        modes.append(f"stick:{button.stick.mode}")
    return ProblemOutcome(
        num_buttons=num_buttons,
        problem_id=problem_id,
        solvable=probe.predicted_solvable,
        num_achievable=probe.num_achievable,
        first_success_idx=None,
        num_success=0,
        pool_size=0,
        wall_clock_s=time.perf_counter() - start,
        blockers=tuple(b.button for b in probe.blockers()),
        failure_modes=tuple(modes),
    )


def _run_full(args: tuple[int, int, dict[str, float]]) -> ProblemOutcome:
    """Worker: draw the real pool and refine it in order."""
    num_buttons, problem_id, budgets = args
    from bilevel_planning.bilevel_planning_graph import (  # pylint: disable=import-outside-toplevel
        BilevelPlanningGraph,
    )
    from bilevel_planning.refiners.backtracking_refiner import (  # pylint: disable=import-outside-toplevel
        BacktrackingRefiner,
    )
    from bilevel_planning.utils import (  # pylint: disable=import-outside-toplevel
        RelationalControllerGenerator,
    )

    from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (  # pylint: disable=import-outside-toplevel
        make_plan_generator,
    )
    from alphatamp.approaches.spectre.envs.stickbutton2d.sampler import (  # pylint: disable=import-outside-toplevel
        AcceptanceTrajectorySampler,
    )

    acceptance: Acceptance = "superset" if budgets.get("superset", 0.0) else "exact"
    start = time.perf_counter()
    env, env_models, x0 = _build_models(num_buttons, problem_id)
    try:
        s0 = env_models.state_abstractor(x0)
        goal = env_models.goal_deriver(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)

        generator = make_plan_generator(env_models, x0, seed=problem_id)
        pool = list(
            itertools.islice(
                generator(x0, s0, goal, float(budgets["plan_timeout"]), bpg),
                int(budgets["k_max"]),
            )
        )
        sampler = AcceptanceTrajectorySampler(
            controller_generator=RelationalControllerGenerator(env_models.skills),
            transition_function=env_models.transition_fn,
            state_abstractor=env_models.state_abstractor,
            max_trajectory_steps=int(budgets["horizon"]),
            acceptance=acceptance,
        )
        first_idx: int | None = None
        num_success = 0
        for idx, (state_plan, action_plan) in enumerate(pool):
            refiner = BacktrackingRefiner(
                trajectory_sampler=sampler,
                num_sampling_attempts_per_step=int(budgets["samples"]),
                seed=idx,
            )
            try:
                ok = (
                    refiner(x0, state_plan, action_plan, float(budgets["timeout"]), bpg)
                    is not None
                )
            except BaseException:  # pylint: disable=broad-exception-caught
                ok = False
            if ok:
                num_success += 1
                if first_idx is None:
                    first_idx = idx
    finally:
        env.close()

    return ProblemOutcome(
        num_buttons=num_buttons,
        problem_id=problem_id,
        solvable=num_success > 0,
        num_achievable=0,
        first_success_idx=first_idx,
        num_success=num_success,
        pool_size=len(pool),
        wall_clock_s=time.perf_counter() - start,
        blockers=(),
        failure_modes=(),
    )


def _iter_results(
    mode: str, jobs: Sequence[tuple[int, int, dict[str, float]]], workers: int
) -> Iterator[ProblemOutcome]:
    """Run jobs concurrently, emitting a heartbeat with progress and ETA."""
    worker = _run_probe if mode == "probe" else _run_full
    start = time.perf_counter()
    done = 0
    last_beat = start
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for outcome in pool.imap_unordered(worker, jobs):
            done += 1
            yield outcome
            now = time.perf_counter()
            if now - last_beat >= _HEARTBEAT_S or done == len(jobs):
                elapsed = now - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - done) / rate if rate > 0 else float("nan")
                print(
                    f"[heartbeat] {done}/{len(jobs)} problems  "
                    f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m",
                    flush=True,
                )
                last_beat = now


def _summarize(mode: str, results: list[ProblemOutcome]) -> None:
    """Print the per-variant table plus the dominant blocking failure modes."""
    by_variant: dict[int, list[ProblemOutcome]] = collections.defaultdict(list)
    for r in results:
        by_variant[r.num_buttons].append(r)

    label = "predicted-solvable" if mode == "probe" else "has >=1 success"
    print(f"\n=== StickButton2D feasibility ({mode} mode) ===")
    header = f"{'variant':<9}{'n':<5}{label:<20}{'rate':<9}{'mean s/problem':<16}"
    if mode == "full":
        header += f"{'median first-succ idx':<24}{'mean #success':<14}"
    print(header)
    print("-" * len(header))

    for nb in sorted(by_variant):
        rows = by_variant[nb]
        solved = [r for r in rows if r.solvable]
        rate = len(solved) / len(rows)
        mean_t = sum(r.wall_clock_s for r in rows) / len(rows)
        line = f"b{nb:<8}{len(rows):<5}{len(solved):<20}{rate:>6.0%}   {mean_t:<16.1f}"
        if mode == "full":
            idxs = sorted(
                r.first_success_idx for r in solved if r.first_success_idx is not None
            )
            med = idxs[len(idxs) // 2] if idxs else float("nan")
            mean_s = sum(r.num_success for r in rows) / len(rows)
            line += f"{med:<24}{mean_s:<14.1f}"
        print(line)

    if mode == "probe":
        print("\nBlocking failure modes (buttons no route could press):")
        for nb in sorted(by_variant):
            modes = collections.Counter(
                m for r in by_variant[nb] for m in r.failure_modes
            )
            if modes:
                top = ", ".join(f"{k}={v}" for k, v in modes.most_common(4))
                print(f"  b{nb:<3} {top}")
        print(
            "\nNote: predicted-solvable is a DIAGNOSTIC, not a bound in either"
            " direction — it under-estimated b2/b3 and over-estimated b5 against"
            " `--mode full`. Use the failure modes above, not this rate."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, run the sweep, print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="1,2,3,5,10")
    parser.add_argument("--problems", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--mode", choices=("probe", "full"), default="probe")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--plan-timeout", type=float, default=60.0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument(
        "--superset-acceptance",
        action="store_true",
        help=(
            "accept a step when the expected abstract state is a"
            " SUBSET of the achieved one, instead of upstream's exact equality. Sound"
            " for goal achievement; tolerates incidental button presses. A deviation"
            " from stock kinder semantics, so it is opt-in and reported."
        ),
    )
    args = parser.parse_args(argv)

    variants = [int(v) for v in args.variants.split(",") if v.strip()]
    budgets = {
        "k_max": float(args.k_max),
        "timeout": args.timeout,
        "plan_timeout": args.plan_timeout,
        "samples": float(args.samples),
        "horizon": float(args.horizon),
        "superset": 1.0 if args.superset_acceptance else 0.0,
    }
    jobs = [
        (nb, pid, budgets)
        for nb in variants
        for pid in range(args.seed_start, args.seed_start + args.problems)
    ]

    print(
        f"StickButton2D feasibility: mode={args.mode} variants={variants} "
        f"problems={args.problems} -> {len(jobs)} jobs on {args.workers} workers"
    )
    if args.mode == "full":
        print(
            "Expect ~minutes per problem at high button counts "
            "(b5/200 candidates measured at ~380 s single-threaded)."
        )

    results = list(_iter_results(args.mode, jobs, args.workers))
    _summarize(args.mode, results)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    raise SystemExit(main())
