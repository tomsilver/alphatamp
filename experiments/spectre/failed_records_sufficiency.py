"""P-2 sufficiency audit: is each compiled scalar a function of the token stream?

``docs/failed_records_fix.md`` P-2. Each ``cand_overlap`` scalar (coverage, waste, repeat,
regroup) is compiled by us at tensorize time from the failure records + the candidate
skeleton. The learned-pathway question is whether a model reading only the **record
tokens** (plus its fixed per-candidate/scene inputs) *could* reproduce it. If not, no
architecture or training change can — the required input is simply absent from the stream
(hypothesis C1, content gap).

**The test (a rigorous insufficiency detector).** Hold one episode fixed — so the scene,
the initial/goal atoms and every candidate skeleton are constant — and vary only the
failure context ``F``. For a fixed candidate ``c``, its scalar is a function of
``(records(F), c, scene)``; its *token* input is ``token_bag(F)``. If two contexts produce
the **same** ``token_bag`` but a **different** scalar for the same ``c``, then the scalar
depends on something about ``F`` that never reached a token → **proof of insufficiency**.
Zero collisions over a large sample is *consistent with* sufficiency (not a proof).

We enumerate every singleton context ``F = {i}`` (``i`` a failed candidate), group them by
their aggregated ``token_bag``, and within each same-token group check whether any
candidate's scalar disagrees. Singletons are the cleanest collision source: two failed
candidates with token-identical failures but different *establishing steps* (seating
charts) give identical ``token_bag``s while a third candidate's ``regroup`` differs.

Read-only: tensorizes stored raw episodes through the real ``build_example`` path (the same
one training uses) and never trains or re-refines. The four scalars are read straight out of
``example.cand_overlap`` columns ``[dead, jaccard, coverage, waste, repeat, regroup]``.

Usage::

    python experiments/spectre/failed_records_sufficiency.py \
        --variants dd2d_v4 restock3d_v3 --episodes 25 --max-fail 40
"""

from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path

from alphatamp.approaches.spectre.dataset import build_example
from alphatamp.approaches.spectre.domain import spec_for
from alphatamp.approaches.spectre.io import load_episode
from alphatamp.approaches.spectre.vocab import Vocab

REPO = Path(__file__).resolve().parents[2]

#: cand_overlap column index of each scalar when coverage+repeat+regroup are all emitted:
#: [dead, jaccard, coverage, waste, repeat, regroup].
_SCALAR_COL = {"coverage": 2, "waste": 3, "repeat": 4, "regroup": 5}


def _token_bag(records) -> tuple:
    """Order-independent, hashable digest of a context's record tokens.

    One RecordArray is ``(schema_id, arg_tag_ids, culprit_tag_ids, scalars[, delta])``.
    We digest exactly what a `RecordEncoder` token carries — schema, the two role tag sets,
    and the rounded scalars — so two contexts collide here iff the model's evidence input is
    identical. ``state_delta`` is deliberately left out of the stream (below), so it is not
    part of the digest; the four audited scalars never read it.
    """
    items = []
    for r in records:
        schema_id, args, culprits, scalars = r[0], r[1], r[2], r[3]
        items.append(
            (
                int(schema_id),
                tuple(sorted(int(a) for a in args)),
                tuple(sorted(int(c) for c in culprits)),
                tuple(round(float(s), 3) for s in scalars),
            )
        )
    return tuple(sorted(items))


def audit_variant(variant: str, max_episodes: int, max_fail: int) -> dict:
    raw = REPO / "data" / "spectre" / "raw" / variant / "train" / "episodes"
    vocab_path = REPO / "data" / "spectre" / "derived" / variant / "train_vocab.json"
    paths = sorted(glob.glob(str(raw / "*")))
    if not paths or not vocab_path.exists():
        print(f"[{variant}] no collection — skipped")
        return {}
    vocab = Vocab.from_json(vocab_path)
    spec = spec_for(variant)

    # Per scalar: how many (episode, token_bag, candidate) groups were checked, and how many
    # showed >1 distinct value across same-token contexts (= insufficiency witnesses).
    checked = collections.Counter()
    collisions = collections.Counter()
    example_witness: dict[str, tuple] = {}
    # Value distribution per scalar over EVERY (singleton-context, candidate) evaluation,
    # so a 0-collision verdict can be read against how much the scalar even varies (a
    # near-constant scalar makes the collision hunt underpowered, not sufficient).
    nonzero = collections.Counter()
    total_vals = collections.Counter()
    distinct: dict[str, set] = {n: set() for n in _SCALAR_COL}
    n_ep = 0

    for p in paths:
        if n_ep >= max_episodes:
            break
        ep = load_episode(Path(p))
        if ep.scene_geometry is None:
            continue
        fail_idx = [i for i, o in enumerate(ep.outcomes) if o.outcome == "fail"]
        if len(fail_idx) < 2:
            continue
        n_ep += 1
        # For each singleton context, the token_bag and the (K, 6) scalar matrix.
        by_token: dict[tuple, list[tuple[int, list]]] = collections.defaultdict(list)
        for i in fail_idx[:max_fail]:
            ex, recs = build_example(
                ep,
                vocab,
                evidence=True,
                context_f=frozenset({i}),
                augment_tags=False,
                spec=spec,
                overlap_mode="both",
                aggregate_records=True,
                coverage_feats=True,
                coverage_mode="both",
                repeat_feats=True,
                regroup_feats=True,
                state_delta=False,
                record_holdout=False,
            )
            by_token[_token_bag(recs)].append((i, ex.overlap))
            for name, col in _SCALAR_COL.items():
                for row in ex.overlap:
                    v = round(float(row[col]), 4)
                    total_vals[name] += 1
                    distinct[name].add(v)
                    if v != 0.0:
                        nonzero[name] += 1
        # Within each same-token group, look for a candidate whose scalar disagrees.
        for members in by_token.values():
            if len(members) < 2:
                continue
            overlaps = [ov for _i, ov in members]
            k = len(overlaps[0])
            for name, col in _SCALAR_COL.items():
                for c in range(k):
                    vals = {round(float(ov[c][col]), 4) for ov in overlaps}
                    checked[name] += 1
                    if len(vals) > 1 and name not in example_witness:
                        example_witness[name] = (
                            Path(p).name,
                            [m[0] for m in members],
                            c,
                            sorted(vals),
                        )
                    if len(vals) > 1:
                        collisions[name] += 1
    return {
        "variant": variant,
        "n_episodes": n_ep,
        "checked": dict(checked),
        "collisions": dict(collisions),
        "witness": example_witness,
        "nonzero": dict(nonzero),
        "total_vals": dict(total_vals),
        "distinct": {n: len(s) for n, s in distinct.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["dd2d_v4", "restock3d_v3"])
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--max-fail", type=int, default=40)
    a = ap.parse_args()

    for variant in a.variants:
        res = audit_variant(variant, a.episodes, a.max_fail)
        if not res:
            continue
        print(f"\n=== {variant} (episodes={res['n_episodes']}) ===")
        print(
            f"{'scalar':10s} {'checked':>9s} {'collis':>7s} "
            f"{'nonzero%':>9s} {'distinct':>9s}  verdict"
        )
        for name in _SCALAR_COL:
            chk = res["checked"].get(name, 0)
            col = res["collisions"].get(name, 0)
            tv = res["total_vals"].get(name, 0)
            nz = res["nonzero"].get(name, 0)
            nzpct = (100.0 * nz / tv) if tv else 0.0
            dist = res["distinct"].get(name, 0)
            if dist <= 1:
                verdict = "CONSTANT (audit vacuous — scalar never varies)"
            elif chk == 0:
                verdict = "no same-token groups"
            elif col > 0:
                verdict = "INSUFFICIENT (collision proof)"
            else:
                verdict = "consistent-with-sufficient"
            print(
                f"{name:10s} {chk:9d} {col:7d} {nzpct:8.2f}% {dist:9d}  {verdict}"
            )
        for name, w in res["witness"].items():
            print(f"  witness[{name}]: {w[0]} contexts={w[1]} cand={w[2]} values={w[3]}")


if __name__ == "__main__":
    main()
