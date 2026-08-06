"""Read-only shape / packing / failure diagnostics for a DD2D collection.

Consolidates the gate probes used to characterize the shape-generalization deficit
(docs/notebook/07 2026-08-06). Three tables for one env-variant:

1. **Geometry by family** -- raw polygon area, convex-hull area (the packing-relevant
   footprint, since nothing packs into a concavity), bbox, and area/bbox fill.
2. **Failure attribution by stratum** -- of every failed pooled candidate, the deepest
   failing schema: ``pick`` / ``retrieve`` / ``place-buffer`` split into *volume* (empty
   culprits = no packing pose found) vs *access* (a pose found, grasp blocked). Reads the
   stored ``refine.failures`` in the native JSON.
3. **Buffer hull-occupancy** -- staged convex-hull area over buffer area, split by
   feasible vs infeasible and by whether the candidate stages a tee/cross. A capacity
   limit would make *infeasible* candidates the *fuller* ones.

Usage::

    python experiments/spectre/spectre_probe_shape_geometry.py \
        --variant dd2d_v4gen_shapeonly --raw-root data/dd2d/raw_v4gen_shapeonly
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from alphatamp.approaches.spectre.io import list_episodes, load_episode

NEW = frozenset({"tee", "cross"})


def _stratum_of(pid: int, band: int = 1_000_000, stratum_band: int = 250_000) -> int:
    return (pid % band) // stratum_band


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def geometry_table(episodes: list) -> None:
    """Print raw/hull/bbox area and fill per shape family."""
    raw = collections.defaultdict(list)
    hull = collections.defaultdict(list)
    bbox = collections.defaultdict(list)
    for ep in episodes:
        for o in ep.scene_geometry.objects:
            poly = Polygon(o.boundary)
            b = np.array(o.boundary)
            raw[o.family].append(poly.area)
            hull[o.family].append(poly.convex_hull.area)
            bbox[o.family].append(b[:, 0].ptp() * b[:, 1].ptp())
    print("\n=== 1. geometry by family (hull = packing footprint) ===")
    print(
        f"{'family':10s} {'n':>4s} {'raw':>7s} {'hull':>7s} {'bbox':>7s} {'fill':>6s}"
    )
    for k in sorted(hull, key=lambda k: -np.mean(hull[k])):
        r, hu, bb = np.mean(raw[k]), np.mean(hull[k]), np.mean(bbox[k])
        tag = "  <-- NEW" if k in NEW else ""
        fill = 100 * hu / bb
        print(f"{k:10s} {len(raw[k]):4d} {r:7.1f} {hu:7.1f} {bb:7.1f} {fill:6.1f}{tag}")


def failure_table(raw_root: Path, split: str) -> None:
    """Print the deepest-failing-schema breakdown per stratum from native JSON."""
    dirs = sorted(glob.glob(str(raw_root / split / "dd2d_*")))
    by_str: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    tot: collections.Counter = collections.Counter()
    for d in dirs:
        files = sorted(glob.glob(d + "/*.json"))
        pid = int(_read_json(files[0])["provenance"]["seed"])
        s = _stratum_of(pid)
        for f in files:
            r = _read_json(f)
            if r.get("label"):
                by_str[s]["success"] += 1
                tot["success"] += 1
                continue
            rf = r.get("refine", {})
            schema = str(rf.get("failure_action") or "?").split("(", maxsplit=1)[0]
            vol = None
            for fo in rf.get("failures", []):
                if fo.get("schema") == schema:
                    vol = len(fo.get("culprits") or []) == 0
            tag = schema
            if schema == "place-buffer":
                tag = "place-buffer/volume" if vol else "place-buffer/access"
            by_str[s][tag] += 1
            tot[tag] += 1
    print("\n=== 2. failure attribution (deepest failing schema, all candidates) ===")
    for s in sorted(by_str):
        print(f"  s{s}: {dict(by_str[s])}")
    fails = {k: v for k, v in tot.items() if k != "success"}
    nf = sum(fails.values()) or 1
    pbv = fails.get("place-buffer/volume", 0)
    print(f"  TOTAL: {dict(tot)}")
    print(f"  place-buffer/volume share of failures: {pbv}/{nf} = {pbv / nf:.3f}")


def occupancy_table(episodes: list) -> None:
    """Print buffer hull-occupancy split by feasibility and new-shape staging."""
    occ = collections.defaultdict(list)
    for ep in episodes:
        s = _stratum_of(ep.provenance.problem_id)
        hull = {
            o.name: Polygon(o.boundary).convex_hull.area
            for o in ep.scene_geometry.objects
        }
        fam = {o.name: o.family for o in ep.scene_geometry.objects}
        buf = [c for c in ep.scene_geometry.containers if c.kind == "buffer"]
        if not buf:
            continue
        x0, y0, x1, y1 = buf[0].bounds
        barea = (x1 - x0) * (y1 - y0)
        for sk, out in zip(ep.skeleton_pool, ep.outcomes):
            staged = [
                p.name
                for op in sk.operator_seq
                if op.name == "place-buffer"
                for p in op.parameters
            ]
            if not staged:
                continue
            o = sum(hull.get(n, 0) for n in staged) / barea
            feas = out.outcome == "success"
            sn = any(fam.get(n) in NEW for n in staged)
            occ[(s, feas, sn)].append(o)
    print("\n=== 3. buffer hull-occupancy (staged hull / buffer area) ===")
    print(f"{'str':4s} {'feasible':9s} {'stagesNew':10s} {'n':>5s} {'occ':>6s}")
    for s in sorted({k[0] for k in occ}):
        for feas in (True, False):
            for sn in (False, True):
                v = occ.get((s, feas, sn), [])
                if v:
                    m = np.mean(v)
                    print(f"s{s:<3d} {str(feas):9s} {str(sn):10s} {len(v):5d} {m:6.2f}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry: load one variant's episodes and print the three tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, help="episode env-variant (pickles)")
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--raw-root",
        default=None,
        help="native JSON root for the failure table (e.g. data/dd2d/raw_<variant>); "
        "omit to skip table 2",
    )
    args = ap.parse_args(argv)

    split_dir = Path(args.data_root) / "raw" / args.variant / args.split
    episodes = [load_episode(p) for p in list_episodes(split_dir)]
    print(f"variant={args.variant}  episodes={len(episodes)}")
    geometry_table(episodes)
    if args.raw_root:
        failure_table(Path(args.raw_root), args.split)
    occupancy_table(episodes)


if __name__ == "__main__":
    main()
