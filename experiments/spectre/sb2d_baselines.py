"""B1–B5 baseline bracket for the pooled StickButton2D collection, per stratum.

``eda.py``'s baselines are already environment-agnostic — they consume a
:class:`eda.LoadedSplit` (canonicalized episodes plus canonical skeleton keys) and never
name a DD2D operator — so standing up the bracket on a second environment needs a runner,
not a reimplementation. This is that runner.

Two settings are deliberate and differ from ``eda.py``'s defaults:

**Uncensored.** ``attempt_budget`` is the pool cap (200), not 20. SPECTRE reports
uncensored so the budget never binds (``decisions.md`` 2026-06-07), and on
StickButton2D a 20-attempt censor would clip exactly the b5 tail where the methods
separate.

**Per stratum.** The pooled variant's strata are button counts, and they differ by two
orders of magnitude in pool size (b1 ≈ 2 candidates, b5 = 200). A single pooled mean is
dominated by whichever stratum happens to be hardest, so every number is reported per
button count as well as overall — the same discipline the DD2D tables use.

Usage::

    python experiments/spectre/sb2d_baselines.py --split test
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from alphatamp.approaches.spectre import (  # noqa: E402  pylint: disable=wrong-import-position,line-too-long
    eda,
)
from alphatamp.approaches.spectre.envs.stickbutton2d.strata import (  # noqa: E402  pylint: disable=wrong-import-position,line-too-long
    BUTTON_COUNTS,
    ENV_VARIANT,
)


def _split_by_stratum(loaded: "eda.LoadedSplit") -> dict[int, "eda.LoadedSplit"]:
    """Partition a loaded split into one sub-split per button count.

    Reads ``provenance.gen_params['num_buttons']`` rather than decoding the problem id,
    so a broken id encoding shows up as a disagreement rather than being reproduced.
    """
    out: dict[int, tuple[list, list]] = {b: ([], []) for b in BUTTON_COUNTS}
    for episode, keys in zip(loaded.episodes, loaded.skeleton_keys):
        nb = int((episode.provenance.gen_params or {}).get("num_buttons", 0))
        if nb in out:
            out[nb][0].append(episode)
            out[nb][1].append(keys)
    return {
        nb: eda.LoadedSplit(
            episodes=eps,
            skeleton_keys=keys,
            k_max=max((len(k) for k in keys), default=0),
        )
        for nb, (eps, keys) in out.items()
        if eps
    }


def _run(train: "eda.LoadedSplit", test: "eda.LoadedSplit", budget: int) -> dict:
    """Every baseline that applies, as ``{name: mean attempts}``."""
    results = {
        "B1 random": eda.random_floor_baseline(test, attempt_budget=budget),
        "B2 default": eda.default_order_baseline(test, attempt_budget=budget),
        "B3 static-hist": eda.static_historical_baseline(
            train, test, attempt_budget=budget
        ),
        "B4 adaptive-hist": eda.adaptive_historical_baseline(
            train, test, attempt_budget=budget
        ),
        "B5 oracle": eda.oracle_ceiling(test, attempt_budget=budget),
    }
    # `attempts` is 1-indexed (1 = the first candidate tried succeeded). Reported as
    # **failed attempts before first success** (`attempts - 1`), which is the FP the v3
    # rollout reports, so the baseline column and the method column mean the same thing.
    return {
        name: float((res.attempts - 1).mean()) if len(res.attempts) else float("nan")
        for name, res in results.items()
    }


def main(argv: list[str] | None = None) -> int:
    """Print the bracket, overall and per stratum."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--env-variant", default=ENV_VARIANT)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--budget",
        type=int,
        default=200,
        help="attempt budget; default = the pool cap, i.e. uncensored",
    )
    a = ap.parse_args(argv)

    root = REPO / a.data_root / "raw" / a.env_variant
    train = eda.load_split_episodes(root / "train")
    test = eda.load_split_episodes(root / a.split)
    if not train.episodes or not test.episodes:
        print(
            f"missing episodes under {root} (train={len(train.episodes)},"
            f" {a.split}={len(test.episodes)})"
        )
        return 2

    print(
        f"\n# {a.env_variant} {a.split}: mean FAILED attempts before first success"
        f" (uncensored at budget {a.budget})"
    )
    print(f"# train={len(train.episodes)} {a.split}={len(test.episodes)}\n")

    rows = {"ALL": _run(train, test, a.budget)}
    per_stratum_test = _split_by_stratum(test)
    per_stratum_train = _split_by_stratum(train)
    for nb in sorted(per_stratum_test):
        if nb in per_stratum_train:
            rows[f"b{nb}"] = _run(per_stratum_train[nb], per_stratum_test[nb], a.budget)

    names = list(rows["ALL"])
    header = f"{'stratum':<10}" + "".join(f"{n:<20}" for n in names)
    print(header)
    print("-" * len(header))
    for label, res in rows.items():
        cells = "".join(f"{res[n]:<20.2f}" for n in names)
        print(f"{label:<10}{cells}")

    counts = {nb: len(s.episodes) for nb, s in per_stratum_test.items()}
    print(f"\nepisodes per stratum ({a.split}): {counts}")
    if len(set(counts.values())) > 1:
        print(
            "  NOTE: strata are unbalanced — a pooled ALL mean weights them by count,"
            " so read the per-stratum rows, not ALL, when comparing methods."
        )
    pooled = [len(k) for k in test.skeleton_keys]
    print(
        f"pool sizes ({a.split}): min {min(pooled)} median"
        f" {statistics.median(pooled):.0f} max {max(pooled)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
