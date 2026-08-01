"""Gate A, offline: does coverage-ranking beat the static order on the collected pools?

The question is the one ``unified_probe.py rerank`` answers — re-rank the remaining pool
by ``coverage`` desc (tie-break ``waste`` asc) as failures accrue, against a static floor
and an oracle ceiling — but computed from **collected episodes** rather than from a fresh
refinement.

That is not a shortcut, it is the same measurement on strictly better data. The probe
refines a pool at the collection budgets and throws the episodes away; the collection
refines the same pools and keeps them, so running the probe first duplicates hours of
refinement to answer a question the collection already contains. Running it offline also
means the gate is measured on the **exact** pools the model will train on, including the
serialization and canonicalization round trip that the in-memory probe path skips — which
is where object identity is most likely to be silently lost.

Arms mirror the probe's exactly so the numbers are comparable to the pre-filter
measurement (b3 static 7.20 → coverage 5.87; b5 24.20 → 17.00):

===================  =========================================================
``static``           the collection's own order — the floor to beat
``coverage_waste``   coverage desc, waste asc — the deployed feature pair
``coverage_only``    coverage desc alone
``waste_only``       waste asc alone
``oracle``           first success immediately — the ceiling
===================  =========================================================

Usage::

    python experiments/spectre/sb2d_rerank_gate.py --split test
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from relational_structs.utils import (  # noqa: E402  pylint: disable=wrong-import-position
    all_ground_operators,
)

from alphatamp.approaches.spectre.domain import (  # noqa: E402  pylint: disable=wrong-import-position
    spec_for,
)
from alphatamp.approaches.spectre.io import (  # noqa: E402  pylint: disable=wrong-import-position
    list_episodes,
    load_episode,
)
from alphatamp.approaches.spectre.unified_evidence import (  # noqa: E402  pylint: disable=wrong-import-position
    coverage_and_waste,
    culprit_pool,
    records_from_failure_records,
    universal_objects,
)

ARMS = ("static", "coverage_waste", "coverage_only", "waste_only", "oracle")


def _rollout(episode, arm: str, ground_ops, universal) -> Optional[int]:
    """Failed attempts before the first success, under one ordering policy.

    Mirrors ``unified_probe._rollout``: the context grows with each failure, the pool is
    re-scored from scratch each step, and ties fall back to pool order so the arms differ
    only in their key.
    """
    labels = [o.outcome == "success" for o in episode.outcomes]
    if not any(labels):
        return None
    if arm == "oracle":
        return 0
    if arm == "static":
        return labels.index(True)

    candidates = [list(s.operator_seq) for s in episode.skeleton_pool]
    spec = spec_for(episode.provenance.env_variant)
    remaining = list(range(len(candidates)))
    tried: list[int] = []
    # The context grows by exactly one candidate per step, so rebuilding it from the whole
    # `tried` set every step is quadratic for no reason. On a 200-deep pool that is the
    # difference between minutes and hours per episode.
    context: list = []
    attempts = 0
    while remaining:
        if tried:
            context.extend(
                records_from_failure_records(episode, frozenset(tried[-1:]), spec)
            )
            pool = culprit_pool(context, ground_ops)
            scored = []
            for i in remaining:
                # One shared `_Memo` for both features -- they depend on the same
                # per-(candidate, record) quantities, and computing them separately
                # doubles the hoisted work.
                cov, wst = coverage_and_waste(
                    candidates[i],
                    context,
                    pool,
                    episode.initial_abstract_state.atoms,
                    episode.goal_atoms,
                    universal,
                )
                if arm == "coverage_only":
                    key = (-cov, 0.0, i)
                elif arm == "waste_only":
                    key = (wst, 0.0, i)
                else:
                    key = (-cov, wst, i)
                scored.append((key, i))
            scored.sort()
            remaining = [i for _, i in scored]
        pick = remaining.pop(0)
        if labels[pick]:
            return attempts
        attempts += 1
        tried.append(pick)
    return attempts


def _episode_row(path_str: str) -> Optional[tuple[int, dict[str, float]]]:
    """Every arm's result for one episode: ``(num_buttons, {arm: failed attempts})``.

    A worker entry point — episodes are independent, and on 200-candidate pools each one
    re-scores the whole remaining pool at every step, so this is where the time goes.
    Returns ``None`` for an episode with no feasible skeleton (nothing to rank).
    """
    episode = load_episode(Path(path_str))
    ground_ops = list(
        all_ground_operators(
            {op.parent for s in episode.skeleton_pool for op in s.operator_seq},
            set(episode.initial_abstract_state.objects),
        )
    )
    universal = universal_objects(ground_ops)
    results = {arm: _rollout(episode, arm, ground_ops, universal) for arm in ARMS}
    if results["static"] is None:
        return None
    nb = int((episode.provenance.gen_params or {}).get("num_buttons", 0))
    return nb, {arm: float(results[arm]) for arm in ARMS}


def _paired(a: Sequence[float], b: Sequence[float]) -> str:
    better = sum(1 for x, y in zip(a, b) if y < x)
    worse = sum(1 for x, y in zip(a, b) if y > x)
    return f"better on {better}, worse on {worse}, tied on {len(a) - better - worse}"


def main(argv: list[str] | None = None) -> int:
    """Report the re-ranking table, per button count."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env-variant", default="stickbutton2d_v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    root = REPO / a.data_root / "raw" / a.env_variant / a.split
    paths = list_episodes(root)
    if not paths:
        print(f"no episodes under {root}")
        return 2
    if a.max_episodes:
        # Stride, never truncate: strata occupy contiguous problem-id bands, so a prefix
        # would be one button count.
        step = max(1, len(paths) // a.max_episodes)
        paths = paths[::step][: a.max_episodes]

    rows: dict[int, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    skipped = 0
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for done, out in enumerate(
            pool.imap_unordered(_episode_row, [str(p) for p in paths]), 1
        ):
            if out is None:
                skipped += 1
            else:
                nb, res = out
                for arm in ARMS:
                    rows[nb][arm].append(res[arm])
            if done % 10 == 0 or done == len(paths):
                elapsed = time.perf_counter() - start
                print(
                    f"[heartbeat] {done}/{len(paths)} episodes"
                    f"  elapsed {elapsed / 60:.1f}m"
                    f"  ETA {(len(paths) - done) * elapsed / done / 60:.1f}m",
                    flush=True,
                )

    print(f"\n# {a.env_variant} {a.split}: mean failed attempts before first success")
    print(f"# {len(paths) - skipped} episodes ({skipped} with no feasible skeleton)\n")
    header = f"{'variant':<9}{'n':<5}" + "".join(f"{arm:<17}" for arm in ARMS)
    print(header)
    print("-" * len(header))
    for nb in sorted(rows):
        cells = ""
        for arm in ARMS:
            vals = rows[nb][arm]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells += f"{statistics.mean(vals):.2f} ± {sd:.2f}".ljust(17)
        print(f"b{nb:<8}{len(rows[nb]['static']):<5}{cells}")

    print()
    for nb in sorted(rows):
        base = rows[nb]["static"]
        for arm in ("coverage_only", "coverage_waste"):
            print(f"  b{nb} {arm:<15} vs static — {_paired(base, rows[nb][arm])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
