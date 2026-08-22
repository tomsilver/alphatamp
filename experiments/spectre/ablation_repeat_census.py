"""Firing + soundness census for the `repeat` overlap feature on DD2D / SB2D.

`repeat` (dataset.build_example) vetoes any candidate that re-uses a step
`(schema, args)` for which some FAILED candidate had a record that is

    step_certificate(schema)  AND  proves_failure()  AND  blame_empty

with `proves_failure() = exhausted and not budget_exhausted` and
`blame_empty = not culprits and not dev_blame`. On restock3d_v3 the certified
schema is `place_{tall,short}` (F3 = a too-tall block hitting the ceiling), an
intrinsic dead step. This script asks, for a hypothetical `step_certificate`
declaration on DD2D / SB2D schemas: how often would `repeat` FIRE, and how often
would it be UNSOUND (flag a step that a *feasible* candidate actually uses -- a
means-failure the ordering resolves, not an intrinsic dead step)?

Read-only. Reads stored raw episodes; nothing is re-refined -- exactly the point:
activating `repeat` retroactively needs no re-rollout. See the ablation ADR.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import pickle
from pathlib import Path

from alphatamp.approaches.spectre.failure_record import records_for_candidate

REPO = Path(__file__).resolve().parents[2]


def _load_split(variant: str, split: str):
    for p in sorted(
        glob.glob(str(REPO / f"data/spectre/raw/{variant}/{split}/episodes/*"))
    ):
        with gzip.open(p, "rb") as f:
            yield pickle.load(f)


def _step_set(skel):
    return {(op.name, tuple(p.name for p in op.parameters)) for op in skel.operator_seq}


def census(variant: str, declared: frozenset[str], splits=("train", "test")):
    """Per-schema record stats + firing/leakage if `declared` schemas are certified."""
    rec_by_schema = collections.Counter()
    provable_by_schema = collections.Counter()
    blameless_prov_by_schema = collections.Counter()

    n_episodes = 0
    n_cands = 0
    n_feasible = 0
    n_flagged = 0  # candidates flagged repeat=1 (any declared schema)
    n_flagged_feasible = 0  # LEAKAGE: feasible candidates flagged (unsound veto)
    eps_with_fire = 0

    for split in splits:
        for ep in _load_split(variant, split):
            n_episodes += 1
            fail_idx = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
            # repeat_steps: blameless+provable+exhausted steps of DECLARED schemas,
            # gathered from the episode's failed candidates (the full-context ceiling).
            repeat_steps: set = set()
            for i in fail_idx:
                seq = [
                    (op.name, tuple(p.name for p in op.parameters))
                    for op in ep.skeleton_pool[i].operator_seq
                ]
                for r in records_for_candidate(ep, i):
                    rec_by_schema[r.schema] += 1
                    prov = r.proves_failure()
                    blameless = not r.culprits and not r.dev_blame
                    if prov:
                        provable_by_schema[r.schema] += 1
                    if prov and blameless:
                        blameless_prov_by_schema[r.schema] += 1
                    if r.schema in declared and prov and blameless:
                        t = min(max(int(r.step_index), 0), len(seq) - 1)
                        repeat_steps.add(seq[t])
            if repeat_steps:
                eps_with_fire += 1
            for i, o in enumerate(ep.outcomes):
                n_cands += 1
                feasible = o.outcome == "success"
                n_feasible += feasible
                flagged = bool(repeat_steps & _step_set(ep.skeleton_pool[i]))
                n_flagged += flagged
                if flagged and feasible:
                    n_flagged_feasible += 1

    print(
        f"\n================  {variant}  (declared step_certificate: {sorted(declared) or '—'})"
    )
    print(f"episodes={n_episodes}  candidates={n_cands}  feasible={n_feasible}")
    print("  records by schema:          ", dict(rec_by_schema))
    print("  provable by schema:         ", dict(provable_by_schema))
    print("  blameless+provable by schema:", dict(blameless_prov_by_schema))
    print(f"  episodes where repeat fires: {eps_with_fire}/{n_episodes}")
    print(
        f"  candidates flagged repeat=1: {n_flagged}/{n_cands} "
        f"({100*n_flagged/max(n_cands,1):.1f}%)"
    )
    print(
        f"  LEAKAGE (feasible flagged):  {n_flagged_feasible}/{n_feasible} "
        f"({100*n_flagged_feasible/max(n_feasible,1):.1f}% of feasible) "
        f"<- unsound vetoes; 0 = sound"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    # DD2D: probe each candidate schema. Only `place-buffer` fires (retrieve/pick
    # failures always carry culprits/dev_blame -> never blameless), and it leaks 44.6%
    # of feasible candidates -> the DEPLOYED ablation declaration, an intentional
    # negative-transfer stress test (a packing means-failure is not an intrinsic dead
    # step).
    census("dd2d_v4", frozenset(), a.splits)  # stats only, no declaration
    for d in [{"retrieve"}, {"pick"}, {"place-buffer"}]:
        census("dd2d_v4", frozenset(d), a.splits)
    # SB2D: the 4 press schemas are the terminal-manipulation analogue of restock's
    # place_{tall,short}; ~55% firing, ~10.9% leakage (far more sound than DD2D's
    # packing) -> the DEPLOYED ablation declaration.
    _SB2D_PRESS = {
        "StickPressButtonFromNothing",
        "RobotPressButtonFromNothing",
        "StickPressButtonFromButton",
        "RobotPressButtonFromButton",
    }
    census("stickbutton2d_v1", frozenset(), a.splits)
    census("stickbutton2d_v1", frozenset(_SB2D_PRESS), a.splits)


if __name__ == "__main__":
    main()
