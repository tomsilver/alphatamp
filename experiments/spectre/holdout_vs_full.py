"""Paired per-stratum diagnostic: held-out (subset) vs matched full-strata control.

Resolves the "held-out beats full" anomaly by comparing the subset-trained (held-out
stratum) and full-strata models on the SAME test problems, **per stratum with a paired
bootstrap** over problems -- not the pooled ALL, which averages the held-out stratum
with the trained strata and is dominated by their run-to-run variance.

For each (env, method): Δ = subset − full per stratum. **Positive Δ ⇒ subset worse
(full better).** The held-out stratum (DD2D s3 / SB2D b5) is the column where "more
training data helps" is actually testable; a CI that excludes 0 there settles it. The
trained strata show whether holding out the hard stratum *specialized* the model on
the easy ones.

Read-only: consumes the compare caches only. Run after the full-control arms are scored:
    python experiments/spectre/holdout_vs_full.py            # all envs present
    python experiments/spectre/holdout_vs_full.py --env dd2d # one env
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alphatamp.approaches.spectre import compare, eda

REPO = Path(__file__).resolve().parents[2]
DERIVED = REPO / "data" / "spectre" / "derived"

# (env key, stratum labels, held-out stratum index, and per-method (subset, full)
# sources).
# A source is (cache_variant, kind, subdir): kind "named" reads an adaptive `_adaptive`
# subdir via load_named_fp_records (fp field); kind "piginet" reads the static `piginet`
# dir via load_fp_records + rollout_fp. Full SPECTRE = the `spectre3_full_adaptive` arm
# in the held-out cache; full PIGINet = the deployed/v2 cache's `piginet` on the same
# pids.
SPECS = {
    "dd2d": {
        "labels": {0: "s0", 1: "s1", 2: "s2", 3: "s3"},
        "held_out": 3,
        "methods": {
            # DD2D full control = the deployed dd2d_v4 cache: the correct all-strata
            # full (100/stratum, current code, same recipe, same test problems). A fresh
            # seed-matched arm was attempted but trained pathologically slowly (~700
            # s/ep vs ~6 s/ep, cause undiagnosed) and was abandoned rather than block
            # for hours; the deployed cache differs only in the training draw. See the
            # 2026-08-10 ADR.
            "SPECTRE-adaptive": {
                "subset": ("dd2d_v4_holdout_s3", "named", "spectre3_adaptive"),
                "full": ("dd2d_v4", "named", "spectre3_adaptive"),
            },
            "PIGINet": {
                "subset": ("dd2d_v4_holdout_s3", "piginet", None),
                "full": ("dd2d_v4", "piginet", None),
            },
        },
    },
    "sb2d": {
        "labels": {0: "b1", 1: "b2", 2: "b3", 3: "b5"},
        "held_out": 3,
        "methods": {
            "SPECTRE-adaptive": {
                "subset": ("stickbutton2d_v1_holdout_b5", "named", "spectre3_adaptive"),
                "full": ("stickbutton2d_v2", "named", "spectre3_adaptive"),
            },
            "PIGINet": {
                "subset": ("stickbutton2d_v1_kinder_holdout_b5", "piginet", None),
                "full": ("stickbutton2d_v2_kinder", "piginet", None),
            },
        },
    },
}


def _fp_by_pid(source: tuple[str, str, str | None]) -> dict[int, float] | None:
    """Seed-averaged per-problem FP for one source, or None if its cache is absent."""
    variant, kind, subdir = source
    cache = DERIVED / variant / "compare_cache"
    if kind == "named":
        d = cache / f"{subdir.split('_adaptive')[0]}_adaptive"
        if not d.is_dir() and not (d.parent / subdir).is_dir():
            return None
        recs = compare.load_named_fp_records(cache, subdir, subdir)
    else:  # piginet static
        if not (cache / "piginet").exists():
            return None
        recs = [
            r for r in compare.load_fp_records(cache) if r.get("method") == "PIGINet"
        ]
    out = {
        int(r["problem_id"]): float(r["fp"]) for r in recs if r.get("fp") is not None
    }
    return out or None


def _bootstrap(sub: dict[int, float], full: dict[int, float], stratum: int | None):
    """Paired Δ = subset − full over the shared pids in one stratum (None = ALL)."""
    pids = sorted(set(sub) & set(full))
    if stratum is not None:
        pids = [p for p in pids if compare.stratum_of(p) == stratum]
    if not pids:
        return None
    a = np.array([sub[p] for p in pids])
    b = np.array([full[p] for p in pids])
    d = eda.bootstrap_mean_difference(a, b, num_resamples=10_000, seed=0)
    return d.point, d.ci_low, d.ci_high, len(pids)


def _report(env: str) -> None:
    spec = SPECS[env]
    labels = spec["labels"]
    held = spec["held_out"]
    print(
        f"\n=== {env}: Δ = subset − full  (positive ⇒ subset worse / full better) ==="
    )
    print(f"    held-out stratum = {labels[held]}  (the coherence column)")
    for method, srcs in spec["methods"].items():
        sub = _fp_by_pid(srcs["subset"])
        full = _fp_by_pid(srcs["full"])
        if sub is None or full is None:
            miss = "subset" if sub is None else "full"
            print(f"  {method:<18} — {miss} cache not present yet; skipped")
            continue
        cells = []
        for s in [None] + sorted(labels):
            r = _bootstrap(sub, full, s)
            name = "ALL" if s is None else labels[s]
            if r is None:
                cells.append(f"{name}: n/a")
                continue
            pt, lo, hi, n = r
            star = " *" if (lo > 0 or hi < 0) else ""
            flag = "<<" if s == held else ""
            cells.append(f"{name}{flag}: {pt:+.2f} [{lo:+.2f},{hi:+.2f}] n{n}{star}")
        print(f"  {method:<18} " + "  ".join(cells))
    print("  (* = CI excludes 0; << = held-out stratum)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=sorted(SPECS), default=None)
    a = ap.parse_args()
    for env in [a.env] if a.env else sorted(SPECS):
        _report(env)


if __name__ == "__main__":
    main()
