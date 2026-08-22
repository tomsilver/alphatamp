"""Restock3D **v2** oracle certification (continuous packing).

Builds the v2 oracle skeleton per sampled problem (tall blocks to the tall section, cubes balanced
across both, stored south-to-north) and certifies it by the manual multi-object rollout with
per-step resampling (:func:`oracle_v2.certify_stratum`). This is the milestone feasibility check —
it does NOT use the collection pipeline / real BacktrackingRefiner (deferred to Phase 2), so the
per-candidate cap / K_max recalibration are not produced here.

Run (from the repo root, venv active)::

    python experiments/spectre/restock3d_v2_oracle.py --strata 0,1,2,3 --problems 6

Prints a per-stratum certified-rate line and an overall PASS/FAIL (PASS = every sampled problem
certified feasible).
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). ----------
import glob
import os
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
pathlib.Path(_B).mkdir(parents=True, exist_ok=True)
for _a, (_sd, _pt) in {
    "liblapack.a": ("lapack", "liblapack.so.3*"),
    "libblas.a": ("blas", "libblas.so.3*"),
}.items():
    _lk = pathlib.Path(_B) / _a
    if not (_lk.exists() or _lk.is_symlink()):
        _cs = sorted(
            glob.glob(f"/usr/lib/x86_64-linux-gnu/{_sd}/{_pt}")
            + glob.glob(f"/usr/lib/x86_64-linux-gnu/{_pt}")
        )
        _r = next((c for c in _cs if os.path.isfile(c)), None)
        if _r:
            _lk.symlink_to(_r)

import argparse
import time

from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import certify_stratum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", type=str, default="0,1,2,3")
    parser.add_argument("--problems", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=18)
    args = parser.parse_args()

    strata = [int(s) for s in args.strata.split(",")]
    all_ok = True
    for stratum in strata:
        t0 = time.perf_counter()
        results = certify_stratum(
            stratum, args.problems, attempts_per_step=args.attempts
        )
        n_ok = sum(1 for r in results if r.certified_feasible)
        dt = time.perf_counter() - t0
        fails = [
            f"pid{r.problem_id}({r.note})" for r in results if not r.certified_feasible
        ]
        all_ok = all_ok and (n_ok == len(results))
        print(
            f"[stratum {stratum}] certified {n_ok}/{len(results)}  "
            f"({dt:.0f}s, plan_len={results[0].plan_len if results else '-'})"
            + (f"  FAILS: {fails}" if fails else ""),
            flush=True,
        )

    print("\n==== ORACLE v2 CERTIFICATION:", "PASS" if all_ok else "FAIL", "====")


if __name__ == "__main__":
    main()
