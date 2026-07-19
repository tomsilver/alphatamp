"""Offline post-mortem harvest runner (Step 11) — populate ``post_mortem`` on collected
DD2D episodes, in place, from stored geometry + refiner metadata.

The definitive collection was run without the harvest in the hot loop (``decisions.md``
2026-07-19 decoupling). This pass augments each episode's ``fail`` outcomes with the
geometry-grounded typed facts the evidence pathway consumes (blocked-at-contents proof,
extraction-failed / pack-exhausted hints), reconstructing from the record — never
regenerating or re-refining. It is **idempotent**: an outcome that already has a
``post_mortem`` is left untouched, so re-running is safe.

The §8.4 packing certificate is **off by default**: at λ=0.8 it proves 0 pack-
impossibles (extraction-dominated regime, ``decisions.md`` 2026-07-18) and costs ~0.5
s/fail, so it is pure overhead here; pass ``--run-certificate`` to include it at tight
λ.

python experiments/spectre/spectre_harvest.py --split train python
experiments/spectre/spectre_harvest.py --split val --split test
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_RUN_CERT = False  # module-level so worker processes inherit the flag


def _harvest_one(path_str: str):
    """Load, harvest, atomically re-write one episode.

    Returns (n_fail_pm, fact_counter).
    """
    from alphatamp.approaches.spectre.envs.dd2d.spectre_harvest import harvest_episode
    from alphatamp.approaches.spectre.io import atomic_write_pickle_gz, load_episode

    path = Path(path_str)
    ep = load_episode(path)
    if ep.scene_geometry is None:
        return (0, Counter())
    if all(
        o.post_mortem is not None or o.outcome != "fail" for o in ep.outcomes
    ) and any(o.post_mortem is not None for o in ep.outcomes):
        # already harvested — idempotent skip (still count for the report).
        counts: Counter = Counter()
        n = 0
        for o in ep.outcomes:
            if o.post_mortem is not None:
                n += 1
                for f in o.post_mortem.facts:
                    counts[f.fact_type] += 1
        return (n, counts)

    out = harvest_episode(ep, run_certificate=_RUN_CERT)
    atomic_write_pickle_gz(out, path)
    counts = Counter()
    n = 0
    for o in out.outcomes:
        if o.post_mortem is not None:
            n += 1
            for f in o.post_mortem.facts:
                counts[f.fact_type] += 1
    return (n, counts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline DD2D post-mortem harvest")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env", default="dd2d_v2")
    ap.add_argument("--split", action="append", default=None, help="repeatable")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--run-certificate", action="store_true")
    args = ap.parse_args(argv)

    global _RUN_CERT
    _RUN_CERT = args.run_certificate
    splits = args.split or ["train", "val", "test"]

    from alphatamp.approaches.spectre.io import list_episodes

    for split in splits:
        split_dir = Path(args.data_root) / "raw" / args.env / split
        paths = [str(p) for p in list_episodes(split_dir)]
        print(
            f"# harvest {split}: {len(paths)} episodes, workers={args.workers}",
            flush=True,
        )
        total_pm = 0
        facts: Counter = Counter()
        # ProcessPoolExecutor inherits `_RUN_CERT` via fork.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, (n_pm, cnt) in enumerate(pool.map(_harvest_one, paths, chunksize=4)):
                total_pm += n_pm
                facts.update(cnt)
                if (i + 1) % 100 == 0:
                    print(f"  ... {i + 1}/{len(paths)}", flush=True)
        print(
            f"# {split}: {total_pm} post_mortems | facts={dict(facts)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
