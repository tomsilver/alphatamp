"""Probes for the unified culprits/coverage/waste definitions.

Answers, before a 400/100/100 collection is built around them, whether the definitions in
``docs/unified_culprits_coverage_waste.md`` actually do anything on StickButton2D.

Three probes, in dependency order:

``classes`` (**the gate**)
    Histogram of failure records by class. Culprits exist only when a failure has
    *collateral* deviation; a pure means-failure (out-of-reach) yields ``Δ̃_r = (∅, ∅)`` and
    blames nobody. If the large majority of SB2D failures are means-failure, ``K`` stays
    empty, coverage is identically inert, and ``rerank`` would be measuring noise.

``rerank``
    Adaptive rollout re-ranking the remaining pool by coverage (desc), tie-broken by waste
    (asc), against a static floor, single-feature arms, and an oracle ceiling.

``dd2d``
    Offline disagreement between unified and deployed coverage/waste over stored dd2d_v4
    episodes. Exactly zero means existing DD2D checkpoints and numbers survive the change.

**Why one refinement pass serves every arm.** A candidate's outcome is a deterministic
function of the problem and the candidate's own refinement seed — not of when it was tried.
So each pool is refined exactly once, the ``(label, record)`` pairs are cached, and every
ordering is then simulated offline. This is the same trick ``precompute_dd2d_cache`` uses,
and it is what makes a five-arm comparison cost one arm's compute.

Examples::

    python experiments/spectre/unified_probe.py classes --problems 15
    python experiments/spectre/unified_probe.py rerank  --problems 15
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
from typing import Any, Optional, Sequence

_HEARTBEAT_S = 30.0


@dataclass
class ProblemPool:
    """One problem's pool, refined once: labels plus the record each failure
    explains."""

    num_buttons: int
    problem_id: int
    candidates: list  # list[list[GroundOperator]]
    labels: list[bool]
    records: list[Optional[Any]]  # UnifiedRecord | None, parallel to candidates
    ground_ops: list
    initial_atoms: frozenset
    goal_atoms: frozenset
    wall_clock_s: float


def _refine_pool(args: tuple[int, int, int, float]) -> Optional[ProblemPool]:
    """Worker: draw one problem's pool and refine every candidate exactly once."""
    num_buttons, problem_id, k_max, timeout_s = args
    import kinder  # pylint: disable=import-outside-toplevel
    from bilevel_planning.bilevel_planning_graph import (  # pylint: disable=import-outside-toplevel
        BilevelPlanningGraph,
    )
    from bilevel_planning.refiners.backtracking_refiner import (  # pylint: disable=import-outside-toplevel
        BacktrackingRefiner,
    )
    from bilevel_planning.utils import (  # pylint: disable=import-outside-toplevel
        RelationalControllerGenerator,
    )
    from kinder_bilevel_planning.env_models import (  # pylint: disable=import-outside-toplevel
        create_bilevel_planning_models,
    )
    from relational_structs.utils import (  # pylint: disable=import-outside-toplevel
        all_ground_operators,
    )

    from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (  # pylint: disable=import-outside-toplevel
        make_plan_generator,
    )
    from alphatamp.approaches.spectre.envs.stickbutton2d.instrumented_refiner import (  # pylint: disable=import-outside-toplevel
        RecordingSampler,
        refine_with_record,
    )

    start = time.perf_counter()
    kinder.register_all_environments()
    env = kinder.make(f"kinder/StickButton2D-b{num_buttons}-v0")
    try:
        obs, _ = env.reset(seed=problem_id)
        models = create_bilevel_planning_models(
            "stickbutton2d",
            env.observation_space,
            env.action_space,
            num_buttons=num_buttons,
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        goal = models.goal_deriver(x0)

        bpg = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        generator = make_plan_generator(models, x0, seed=problem_id)
        pool = list(itertools.islice(generator(x0, s0, goal, 30.0, bpg), k_max))
        if not pool:
            return None

        sampler = RecordingSampler(
            controller_generator=RelationalControllerGenerator(models.skills),
            transition_function=models.transition_fn,
            state_abstractor=models.state_abstractor,
            max_trajectory_steps=200,
        )
        labels: list[bool] = []
        records: list[Optional[Any]] = []
        for idx, (state_plan, action_plan) in enumerate(pool):
            refiner = BacktrackingRefiner(
                trajectory_sampler=sampler,
                num_sampling_attempts_per_step=5,
                seed=idx,
            )
            ok, record = refine_with_record(
                refiner, sampler, x0, state_plan, action_plan, timeout_s, bpg
            )
            labels.append(ok)
            records.append(record)

        return ProblemPool(
            num_buttons=num_buttons,
            problem_id=problem_id,
            candidates=[list(ap) for _, ap in pool],
            labels=labels,
            records=records,
            ground_ops=list(all_ground_operators(models.operators, s0.objects)),
            initial_atoms=frozenset(s0.atoms),
            goal_atoms=frozenset(goal.atoms),
            wall_clock_s=time.perf_counter() - start,
        )
    finally:
        env.close()


def _collect(jobs: Sequence[tuple], workers: int) -> list[ProblemPool]:
    """Refine every requested problem concurrently, with progress and ETA."""
    out: list[ProblemPool] = []
    start = time.perf_counter()
    last_beat = start
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_refine_pool, jobs), 1):
            if result is not None:
                out.append(result)
            now = time.perf_counter()
            if now - last_beat >= _HEARTBEAT_S or done == len(jobs):
                elapsed = now - start
                eta = (len(jobs) - done) * elapsed / done
                print(
                    f"[heartbeat] {done}/{len(jobs)} pools  "
                    f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m",
                    flush=True,
                )
                last_beat = now
    return out


# --------------------------------------------------------------------------- #
# P1 — the gate
# --------------------------------------------------------------------------- #
def probe_classes(pools: Sequence[ProblemPool]) -> None:
    """Classify every failure record and report whether ``K`` ever populates."""
    from alphatamp.approaches.spectre.unified_evidence import (  # pylint: disable=import-outside-toplevel
        blame,
        collateral,
        culprit_pool,
    )

    per_variant: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    pool_sizes: dict[int, list[int]] = collections.defaultdict(list)

    for problem in pools:
        counter = per_variant[problem.num_buttons]
        records = [r for r in problem.records if r is not None]
        for record in records:
            dev = collateral(record)
            if dev.is_empty():
                counter["means_failure_only"] += 1
            else:
                if dev.added:
                    counter["collateral_add"] += 1
                if dev.deleted:
                    counter["collateral_delete"] += 1
            counter["total_records"] += 1
        counter["failures_without_record"] += sum(
            1
            for lab, r in zip(problem.labels, problem.records)
            if not lab and r is None
        )
        # How big is K once every observed failure is in the context?
        pool_sizes[problem.num_buttons].append(
            len(culprit_pool(records, problem.ground_ops))
        )
        counter["blamed_objects"] += len({o for r in records for o in blame(r)})

    print("\n=== P1: failure-record classes (the gate) ===")
    header = (
        f"{'variant':<9}{'records':<10}{'means-only':<14}{'collateral-add':<17}"
        f"{'collateral-del':<16}{'mean |K|':<10}"
    )
    print(header)
    print("-" * len(header))
    for nb in sorted(per_variant):
        c = per_variant[nb]
        total = max(c["total_records"], 1)
        ks = pool_sizes[nb]
        print(
            f"b{nb:<8}{c['total_records']:<10}"
            f"{c['means_failure_only']:>5} ({c['means_failure_only'] / total:>4.0%}) "
            f"{c['collateral_add']:>7} ({c['collateral_add'] / total:>4.0%})  "
            f"{c['collateral_delete']:>6} ({c['collateral_delete'] / total:>4.0%}) "
            f"{sum(ks) / max(len(ks), 1):>9.1f}"
        )
    print(
        "\nGATE: coverage can only act through collateral records. If means-only"
        " dominates,\n      K stays empty and the rerank probe would measure noise."
    )


# --------------------------------------------------------------------------- #
# P3 — the re-rank rollout
# --------------------------------------------------------------------------- #
def _rollout(problem: ProblemPool, arm: str) -> Optional[int]:
    """Failed attempts before the first success, under one ordering policy.

    ``None`` when the pool has no feasible candidate at all (the problem is unusable and
    is excluded from every arm alike).
    """
    from alphatamp.approaches.spectre.unified_evidence import (  # pylint: disable=import-outside-toplevel
        coverage,
        culprit_pool,
        universal_objects,
        waste,
    )

    if not any(problem.labels):
        return None
    n = len(problem.candidates)
    if arm == "oracle":
        return 0
    if arm == "static":
        return problem.labels.index(True)

    universal = universal_objects(problem.ground_ops)
    remaining = list(range(n))
    context: list = []
    attempts = 0
    while remaining:
        if context:
            pool = culprit_pool(context, problem.ground_ops)
            scored = []
            for i in remaining:
                cand = problem.candidates[i]
                cov = coverage(cand, context, pool, problem.initial_atoms, universal)
                wst = waste(cand, context, pool, problem.goal_atoms, universal)
                if arm == "coverage_only":
                    key = (-cov, i)
                elif arm == "waste_only":
                    key = (wst, i)
                else:  # coverage desc, waste asc, then pool order
                    key = (-cov, wst, i)
                scored.append((key, i))
            scored.sort()
            remaining = [i for _, i in scored]
        pick = remaining.pop(0)
        if problem.labels[pick]:
            return attempts
        attempts += 1
        record = problem.records[pick]
        if record is not None:
            context.append(record)
    return attempts


def probe_rerank(pools: Sequence[ProblemPool]) -> None:
    """Compare adaptive coverage/waste ordering against static, single-feature and
    oracle."""
    arms = ("static", "coverage_waste", "coverage_only", "waste_only", "oracle")
    per_variant: dict[int, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for problem in pools:
        results = {arm: _rollout(problem, arm) for arm in arms}
        if results["static"] is None:
            continue  # unusable problem, excluded from every arm alike
        for arm in arms:
            per_variant[problem.num_buttons][arm].append(results[arm])

    print(
        "\n=== P3: mean failed attempts before first success (paired per problem) ==="
    )
    header = f"{'variant':<9}{'n':<5}" + "".join(f"{a:<18}" for a in arms)
    print(header)
    print("-" * len(header))
    for nb in sorted(per_variant):
        rows = per_variant[nb]
        n = len(rows["static"])
        line = f"b{nb:<8}{n:<5}"
        for arm in arms:
            line += f"{sum(rows[arm]) / max(n, 1):<18.2f}"
        print(line)

    for nb in sorted(per_variant):
        rows = per_variant[nb]
        paired = [s - c for s, c in zip(rows["static"], rows["coverage_waste"])]
        better = sum(1 for d in paired if d > 0)
        worse = sum(1 for d in paired if d < 0)
        print(
            f"  b{nb}: coverage+waste vs static — better on {better}, worse on {worse},"
            f" tied on {len(paired) - better - worse}"
            f" (mean paired delta {sum(paired) / max(len(paired), 1):+.2f} attempts)"
        )


# --------------------------------------------------------------------------- #
# P2 — DD2D backward compatibility
# --------------------------------------------------------------------------- #
def probe_dd2d(data_root: Path, env_variant: str, max_episodes: int) -> None:
    """Offline: does the unified definition ever disagree with the deployed one on DD2D?

    Deployed coverage is ``|S(c) ∩ K| / |K|``, i.e. per culprit exactly the predicate
    ``k ∈ S(c)``. The unified class-1 test replaces that with index precedence against
    the record's matched steps. §4 predicts they coincide wherever the context is
    terminal or unmatched — which on DD2D is the dominant case, since the blocked query
    is the final ``retrieve``. Exactly zero disagreement means existing DD2D checkpoints
    and published numbers survive the change untouched.
    """
    from relational_structs.utils import (  # pylint: disable=import-outside-toplevel
        all_ground_operators,
    )

    from alphatamp.approaches.spectre.domain import (  # pylint: disable=import-outside-toplevel
        spec_for,
    )
    from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import (  # pylint: disable=import-outside-toplevel
        ALL_OPERATORS,
    )
    from alphatamp.approaches.spectre.failure_record import (  # pylint: disable=import-outside-toplevel
        records_for_candidate,
    )
    from alphatamp.approaches.spectre.io import (  # pylint: disable=import-outside-toplevel
        list_episodes,
        load_episode,
    )
    from alphatamp.approaches.spectre.unified_evidence import (  # pylint: disable=import-outside-toplevel
        UnifiedRecord,
        covered,
        culprit_pool,
        predicted_states,
        universal_objects,
        waste,
    )

    split_dir = data_root / "raw" / env_variant / "test"
    paths = list_episodes(split_dir)
    if not paths:
        raise SystemExit(f"no episodes under {split_dir}")

    total_pairs = 0
    disagreements = 0
    waste_pairs = 0
    waste_disagreements = 0
    examples: list[str] = []
    universal_nonempty = 0

    for path in paths[:max_episodes]:
        episode = load_episode(path)
        spec = spec_for(episode.provenance.env_variant)
        goal_objects = spec.goal_objects(episode)
        subsets = [spec.manipulated(s, goal_objects) for s in episode.skeleton_pool]
        objects = set(episode.initial_abstract_state.objects)
        ground_ops = list(all_ground_operators(ALL_OPERATORS, objects))
        universal = universal_objects(ground_ops)
        if universal:
            universal_nonempty += 1

        fails = [i for i, o in enumerate(episode.outcomes) if o.outcome == "fail"]
        for f in fails[:5]:  # a few singleton contexts per episode
            recs = records_for_candidate(episode, f, spec)
            unified = [
                UnifiedRecord(
                    failed_step=episode.skeleton_pool[f].operator_seq[
                        min(
                            r.step_index, len(episode.skeleton_pool[f].operator_seq) - 1
                        )
                    ],
                    deviation=None,
                    check_blame=tuple(r.culprits),
                )
                for r in recs
                if r.culprits
            ]
            if not unified:
                continue
            pool = culprit_pool(unified, ground_ops)
            if not pool:
                continue
            deployed_culprits = frozenset(o for r in recs for o in r.culprits)

            for i, skeleton in enumerate(episode.skeleton_pool):
                candidate = list(skeleton.operator_seq)
                states = predicted_states(
                    candidate, episode.initial_abstract_state.atoms
                )
                for k in pool:
                    unified_cov = covered(k, candidate, unified, states, universal)
                    deployed_cov = k in subsets[i]
                    total_pairs += 1
                    if unified_cov != deployed_cov:
                        disagreements += 1
                        if len(examples) < 5:
                            examples.append(
                                f"  ep {episode.provenance.problem_id} ctx={f} "
                                f"cand={i} culprit={k}: unified={unified_cov} "
                                f"deployed={deployed_cov}"
                            )
                # waste: object ratio (deployed) vs step ratio (unified)
                dep_waste = len(subsets[i] - deployed_culprits) / max(
                    len(subsets[i]), 1
                )
                uni_waste = waste(
                    candidate, unified, pool, episode.goal_atoms, universal
                )
                waste_pairs += 1
                if abs(dep_waste - uni_waste) > 1e-9:
                    waste_disagreements += 1

    print("\n=== P2: unified vs deployed on DD2D (offline, no training) ===")
    print(f"episodes read           : {min(len(paths), max_episodes)}")
    print(
        f"episodes w/ universal!=0: {universal_nonempty}  (expected 0, by construction)"
    )
    print(
        f"coverage pairs          : {total_pairs}  disagreements: {disagreements}"
        f"  ({disagreements / max(total_pairs, 1):.4%})"
    )
    print(
        f"waste candidates        : {waste_pairs}  disagreements: {waste_disagreements}"
        f"  ({waste_disagreements / max(waste_pairs, 1):.4%})"
    )
    for line in examples:
        print(line)
    if disagreements == 0 and waste_disagreements == 0:
        print(
            "\nEXACTLY ZERO: existing DD2D checkpoints and numbers stand untouched"
            " under the unified definition."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, refine the requested pools once, run the requested probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=("classes", "rerank", "both", "dd2d"))
    parser.add_argument("--variants", default="3,5")
    parser.add_argument("--problems", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--data-root", default="data/spectre")
    parser.add_argument("--env-variant", default="dd2d_v4")
    parser.add_argument("--max-episodes", type=int, default=40)
    args = parser.parse_args(argv)

    if args.probe == "dd2d":
        probe_dd2d(Path(args.data_root), args.env_variant, args.max_episodes)
        return 0

    variants = [int(v) for v in args.variants.split(",") if v.strip()]
    jobs = [
        (nb, pid, args.k_max, args.timeout)
        for nb in variants
        for pid in range(args.seed_start, args.seed_start + args.problems)
    ]
    print(
        f"Refining {len(jobs)} pools (k_max={args.k_max}, timeout={args.timeout}s)"
        f" on {args.workers} workers — one pass serves every arm."
    )
    pools = _collect(jobs, args.workers)
    print(f"{len(pools)} pools usable.")

    if args.probe in ("classes", "both"):
        probe_classes(pools)
    if args.probe in ("rerank", "both"):
        probe_rerank(pools)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    raise SystemExit(main())
