"""Inference-time geometry interventions for the DD2D shape-generalization probe.

SPECTRE v3 is image-free but **geometry-aware**: ``build_v3_example`` feeds the model,
per object, the shape (boundary ring -> footprint encoder), pose, **raw polygon area**,
area-relative-to-target and a concave flag (``model_v2.py`` ``SceneEncoder``). So the
shape-generalization question -- does the s2 deficit on the new tee/cross figures come
from how the model *represents* their size/shape? -- is testable by rewriting **only the
model-input geometry** of collected episodes and re-scoring the same checkpoint.
Feasibility (the stored outcomes) is physical and already measured, so it is held fixed;
only what the ``SceneEncoder`` sees changes.

This rewrites the ``tee``/``cross`` ``ObjectGeometry`` of a source collection into a new
test-variant, leaving names / registry / skeleton pool / outcomes byte-identical
(asserted). Modes:

* ``hullarea``  -- set ``area = convex_hull.area`` (boundary ring **kept**). Isolates the
  raw-area mis-scaling channel: raw polygon area understates a concave shape's packing
  footprint by ~40%, and the old *convex* families had area == hull in training, so only
  the new shapes are mis-scaled. (``rel[i,3]`` and the area-ratio ``rel[i,7]``.)
* ``hullshape`` -- set ``boundary = convex hull ring`` (recentred), ``area = hull area``,
  ``concave = False``. Makes tee/cross convex, testing the footprint-encoder OOD channel.
* ``scaleNN`` (e.g. ``scale07``) -- shrink the tee/cross ``boundary`` **and** ``area`` by
  the linear factor NN/10 in the *model input only*, keeping the stored (large-shape)
  feasibility labels. Mirrors the physical ``sz07`` collection's direction but on the
  SAME problems with UNCHANGED feasibility, so any FP change is purely SPECTRE's ranking
  responding to the size input.

Score the output with the existing train-old/test-new machinery, e.g.::

    python experiments/spectre/spectre_score_v3.py --env-variant dd2d_v4 \
        --test-variant dd2d_v4gen_shapeonly_hullarea --arm 'v3:checkpoints_v3_unified' \
        --astar-baseline --seeds 0 1 2

The intervened variants are registered in ``domain.DOMAINS`` (same ``_DD2D`` spec) and
reuse the dd2d_v4 vocab. astar is a geometry-free control: its FP is identical across
modes (it reads only the unchanged outcomes), the built-in null.

Protocol: docs/decisions/07 2026-08-06.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import replace
from pathlib import Path

from shapely.geometry import Polygon

from alphatamp.approaches.spectre.io import (
    atomic_write_pickle_gz,
    list_episodes,
    load_episode,
)
from alphatamp.approaches.spectre.schema import EpisodeRecord, ObjectGeometry

NEW_FAMILIES = frozenset({"tee", "cross"})
MODES = ("hullarea", "hullshape")
_SCALE_RE = re.compile(
    r"^scale(\d+)$"
)  # scale07 -> 0.7 (linear); shrinks boundary + area


def _recentered_hull_ring(boundary: tuple) -> tuple[tuple[float, float], ...]:
    """Convex-hull exterior ring of a boundary, recentred on its centroid (the item-frame
    convention: centroid at the origin, so the separate pose stays authoritative)."""
    hull = Polygon(boundary).convex_hull
    cx, cy = hull.centroid.coords[0]
    ring = [(x - cx, y - cy) for x, y in hull.exterior.coords[:-1]]
    return tuple(ring)


def _scale_factor(mode: str) -> float | None:
    m = _SCALE_RE.match(mode)
    return int(m.group(1)) / 10.0 if m else None


def _intervene_object(o: ObjectGeometry, mode: str) -> ObjectGeometry:
    if o.family not in NEW_FAMILIES:
        return o
    scale = _scale_factor(mode)
    if scale is not None:
        # linear shrink of the item-frame boundary (centroid at 0); area ~ scale^2
        ring = tuple((x * scale, y * scale) for x, y in o.boundary)
        return replace(o, boundary=ring, area=float(o.area) * scale * scale)
    hull_area = float(Polygon(o.boundary).convex_hull.area)
    if mode == "hullarea":
        return replace(o, area=hull_area)
    # hullshape: convex boundary + hull area + drop the concave flag
    return replace(
        o,
        boundary=_recentered_hull_ring(o.boundary),
        area=hull_area,
        concave=False,
    )


def _intervene_episode(ep: EpisodeRecord, mode: str, out_variant: str) -> EpisodeRecord:
    sg = ep.scene_geometry
    assert sg is not None, "intervention requires scene_geometry"
    new_objs = tuple(_intervene_object(o, mode) for o in sg.objects)
    new_sg = replace(sg, objects=new_objs)
    new_prov = replace(ep.provenance, env_variant=out_variant)
    return replace(ep, scene_geometry=new_sg, provenance=new_prov)


def _assert_loss_free(src: EpisodeRecord, dst: EpisodeRecord, mode: str) -> None:
    """Only tee/cross ``area``/``boundary``/``concave`` (+ provenance.env_variant) may
    differ; everything the rollout depends on must be identical."""
    assert src.object_registry == dst.object_registry
    assert src.skeleton_pool == dst.skeleton_pool  # frozen dataclasses compare by value
    assert src.outcomes == dst.outcomes
    assert src.goal_atoms == dst.goal_atoms
    assert src.provenance.problem_id == dst.provenance.problem_id
    so, do = src.scene_geometry, dst.scene_geometry
    assert so is not None and do is not None
    for a, b in zip(so.objects, do.objects):
        assert a.name == b.name and a.pose == b.pose and a.is_target == b.is_target
        if a.family in NEW_FAMILIES:
            if _scale_factor(mode) is not None:
                assert b.area <= a.area + 1e-6  # shrink lowers area
            else:
                assert b.area >= a.area - 1e-6  # hull area >= raw area
            if mode == "hullshape":
                assert b.concave is False
        else:
            assert a == b  # untouched families are byte-identical


def main(argv: list[str] | None = None) -> None:
    """CLI entry: rewrite one variant's tee/cross geometry into a new test-variant."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        required=True,
        help=f"one of {MODES} or 'scaleNN' (e.g. scale07 = x0.7 linear shrink)",
    )
    ap.add_argument("--src-variant", default="dd2d_v4gen_shapeonly")
    ap.add_argument(
        "--out-variant",
        default=None,
        help="default: <src-variant>_<mode>",
    )
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument("--vocab-from", default="dd2d_v4")
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)
    if args.mode not in MODES and _scale_factor(args.mode) is None:
        ap.error(f"--mode must be one of {MODES} or 'scaleNN'; got {args.mode!r}")

    out_variant = args.out_variant or f"{args.src_variant}_{args.mode}"
    data_root = Path(args.data_root)
    src_dir = data_root / "raw" / args.src_variant / args.split
    out_dir = data_root / "raw" / out_variant / args.split
    paths = list_episodes(src_dir)
    if not paths:
        raise SystemExit(f"no episodes under {src_dir / 'episodes'}")

    print(
        f"intervene {args.mode}: {args.src_variant} -> {out_variant}  ({len(paths)} ep)"
    )
    d_area: list[tuple[float, float]] = []
    for p in paths:
        ep = load_episode(p)
        new_ep = _intervene_episode(ep, args.mode, out_variant)
        _assert_loss_free(ep, new_ep, args.mode)
        assert ep.scene_geometry is not None and new_ep.scene_geometry is not None
        for a, b in zip(ep.scene_geometry.objects, new_ep.scene_geometry.objects):
            if a.family in NEW_FAMILIES:
                d_area.append((a.area, b.area))
        atomic_write_pickle_gz(new_ep, out_dir / "episodes" / p.name)

    # reuse the train-variant vocab (op/pred/type set is geometry-invariant, so no OOV)
    vocab_src = data_root / "derived" / args.vocab_from / "train_vocab.json"
    vocab_dst = data_root / "derived" / out_variant / "train_vocab.json"
    vocab_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vocab_src, vocab_dst)

    n = len(d_area)
    mean_raw = sum(a for a, _ in d_area) / n
    mean_new = sum(b for _, b in d_area) / n
    print(
        f"  rewrote {n} tee/cross objects: area {mean_raw:.1f} -> {mean_new:.1f} "
        f"(x{mean_new / mean_raw:.2f})"
    )
    print(f"  wrote episodes -> {out_dir / 'episodes'}")
    print(f"  copied vocab   -> {vocab_dst}")


if __name__ == "__main__":
    main()
