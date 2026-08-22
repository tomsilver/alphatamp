"""Single-feature-isolation ablation report (2026-08-21): mean rollout FP per arm.

Reads the cached `abl_*_adaptive` rows written by `precompute_dd2d_cache.py` for the
three main-results collections and prints, per environment, each ablation arm's ALL /
per-stratum mean FP plus a paired-bootstrap `Δ vs floor` (the jaccard-only backbone).
This is the command-line twin of §4.3 in `compare_methods.py`; both read the same cache.
Seed 0 for now (2 more seeds deferred). See the ablation ADR +
`ablation_repeat_census.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphatamp.approaches.spectre import compare, eda

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "spectre" / "derived"

ARMS = {
    "floor": "abl_floor_adaptive",
    "+cov": "abl_only_cov_adaptive",
    "+waste": "abl_only_waste_adaptive",
    "+repeat": "abl_only_repeat_adaptive",
    "+records": "abl_only_records_adaptive",
    "all": "abl_all_adaptive",
}
#: env label -> (collection holding the cache). SB2D's arms live under `stickbutton2d_v1`.
ENVS = {"DD2D": "dd2d_v4", "SB2D": "stickbutton2d_v1", "RESTOCK3D_V3": "restock3d_v3"}


def report() -> None:
    for name, var in ENVS.items():
        cd = DER / var / "compare_cache"
        dfs = {
            k: pd.DataFrame(compare.load_named_fp_records_per_seed(cd, v, v))
            for k, v in ARMS.items()
        }
        strata = sorted(dfs["floor"]["stratum"].unique())
        floor = dfs["floor"]
        fbypid = dict(zip(floor["problem_id"], floor["fp"]))
        print(f"\n===== {name} ({var})  strata={strata}  (seed 0) =====")
        print(
            f"{'arm':10s} {'ALL':>6s} "
            + " ".join(f"{'s'+str(k):>6s}" for k in strata)
            + "   Δ vs floor [95% CI]"
        )
        for k, sub in dfs.items():
            allm = sub["fp"].mean()
            per = " ".join(f"{sub[sub.stratum == s]['fp'].mean():6.2f}" for s in strata)
            if k == "floor":
                dtxt = "(reference)"
            else:
                common = sorted(set(sub["problem_id"]) & set(fbypid))
                a = sub.set_index("problem_id").loc[common, "fp"].to_numpy()
                b = pd.Series(fbypid).loc[common].to_numpy()
                d = eda.bootstrap_mean_difference(a, b, num_resamples=10_000, seed=0)
                star = "" if d.ci_low <= 0 <= d.ci_high else " *"
                dtxt = f"{d.point:+.2f} [{d.ci_low:+.2f}, {d.ci_high:+.2f}]{star}"
            print(f"{k:10s} {allm:6.2f} {per}   {dtxt}")
        try:
            r = pd.DataFrame(
                compare.load_named_fp_records_per_seed(cd, "spectre3_adaptive", "dep")
            )
            r0 = r[r.seed == 0] if "seed" in r.columns else r
            print(
                f"  [deployed adaptive seed0] ALL={r0['fp'].mean():6.2f}  "
                "(DD2D/SB2D stale, pre-point-set-upgrade)"
            )
        except Exception as e:  # pragma: no cover - context row only
            print(f"  [deployed ref n/a] {e}")


if __name__ == "__main__":
    report()
