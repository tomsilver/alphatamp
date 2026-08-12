"""Visualize a single PIGINet dataset point (Step 5 of docs/piginet_dd2d_plan.md).

Renders one figure that maps 1:1 to what PIGINet ingests for a record (paper §IV / our
Step-6 encoders): the initial state, the task plan π, the goal/init literals, the
continuous-value features (poses + shapes), the per-object image crops (CLIP-image inputs),
the text vocabulary (CLIP-text inputs), and the feasibility label + refine diagnostics.

    python -m blocks_tamp.dd2d.inspect_example data/dd2d/raw/train/<pid>/000.json --out fig.png
    python -m blocks_tamp.dd2d.inspect_example data/dd2d/raw/train/<pid>/       # picks a record
    python -m blocks_tamp.dd2d.inspect_example data/dd2d/raw/train              # picks a problem

Reads only the record JSON + its sibling ``scene.png`` / ``images/`` (written at collection
time) -- no scene regeneration.
"""

from __future__ import annotations

import argparse
import glob
import math
import os

from ..record import PIGINetExample


def _find_record(path: str) -> str:
    """Resolve a record JSON from a file / problem-dir / split-dir / dataset-root
    arg."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct = sorted(glob.glob(os.path.join(path, "[0-9]*.json")))
        if direct:
            return direct[0]
        nested = sorted(
            glob.glob(os.path.join(path, "**", "[0-9]*.json"), recursive=True)
        )
        if nested:
            return nested[0]
    raise FileNotFoundError(f"no record JSON found at {path!r}")


def _text_tokens(ex: PIGINetExample) -> dict[str, list[str]]:
    """The finite vocabulary this datapoint exercises -> what g_text (CLIP-text)
    encodes."""
    ops = sorted({step[0] for step in ex.task_plan})
    preds = sorted({lit[0] for lit in ex.init_literals + ex.goal_literals})
    cats = sorted({o["category"] for o in ex.objects})
    colors = sorted({o["color"] for o in ex.objects})
    return {"operators": ops, "predicates": preds, "categories": cats, "colors": colors}


def visualize_record(record_path: str, out_path: str | None = None) -> str:
    """Render the datapoint at ``record_path`` to a PNG (default: sibling
    ``inspect.png``)."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    record_path = _find_record(record_path)
    record_dir = os.path.dirname(os.path.abspath(record_path))
    ex = PIGINetExample.load(record_path)
    out_path = out_path or os.path.join(record_dir, "inspect.png")

    objs = ex.objects
    n_obj = len(objs)
    ncol = min(6, max(1, n_obj))
    nrow_crops = math.ceil(n_obj / ncol)

    fig = plt.figure(figsize=(16, 9 + 1.6 * nrow_crops))
    gs = fig.add_gridspec(
        3, 4, height_ratios=[1.35, 0.9 * nrow_crops, 1.15], hspace=0.32, wspace=0.28
    )

    lbl = "POSITIVE (feasible)" if ex.label else "NEGATIVE (infeasible)"
    fig.suptitle(
        f"PIGINet datapoint — {ex.problem_id}  |  plan_idx "
        f"{ex.provenance.get('plan_idx', '?')}  |  {lbl}",
        fontsize=15,
        fontweight="bold",
        color=("#1a7f37" if ex.label else "#b3261e"),
    )

    # -- 1) initial state ----------------------------------------------------
    ax_scene = fig.add_subplot(gs[0, 0:2])
    scene_png = os.path.join(record_dir, "scene.png")
    if os.path.exists(scene_png):
        ax_scene.imshow(imageio.imread(scene_png))
    else:
        ax_scene.text(0.5, 0.5, "(scene.png not found)", ha="center", va="center")
    ax_scene.axis("off")
    ax_scene.set_title("Initial state (drawer + buffer; target = red 'T')", fontsize=11)

    # -- 2) plan pi ----------------------------------------------------------
    ax_plan = fig.add_subplot(gs[0, 2])
    ax_plan.axis("off")
    plan_lines = [
        f"{i+1:>2}. {step[0]}({', '.join(step[1:])})"
        for i, step in enumerate(ex.task_plan)
    ]
    ax_plan.set_title(f"Plan π  ({len(ex.task_plan)} actions)", fontsize=11, loc="left")
    ax_plan.text(
        0.0,
        0.98,
        "\n".join(plan_lines) or "(empty)",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax_plan.transAxes,
    )

    # -- 3) goal + symbolic init --------------------------------------------
    ax_gi = fig.add_subplot(gs[0, 3])
    ax_gi.axis("off")
    goal = "\n".join(f"({' '.join(map(str, l))})" for l in ex.goal_literals)
    sym_init = [l for l in ex.init_literals if l and l[0] != "at-pose"]
    init = "\n".join(f"({' '.join(map(str, l))})" for l in sym_init)
    ax_gi.set_title("Goal G  /  symbolic Init I", fontsize=11, loc="left")
    ax_gi.text(
        0.0,
        0.98,
        f"GOAL:\n{goal}\n\nINIT (symbolic):\n{init}",
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        transform=ax_gi.transAxes,
    )

    # -- 5) per-object image crops (g_img) ----------------------------------
    fig.text(
        0.5,
        gs[1, :].get_position(fig).y1 + 0.028,
        "Image features g_img — per-object segmented crops (CLIP-image input)",
        ha="center",
        fontsize=11,
    )
    sub = gs[1, :].subgridspec(nrow_crops, ncol, hspace=0.6, wspace=0.15)
    path_by_obj = {img["object"]: img.get("path") for img in ex.images}
    for i, o in enumerate(objs):
        ax = fig.add_subplot(sub[i // ncol, i % ncol])
        ax.axis("off")
        p = path_by_obj.get(o["name"])
        full = os.path.join(record_dir, p) if p else None
        if full and os.path.exists(full):
            ax.imshow(imageio.imread(full))
        else:
            ax.text(0.5, 0.5, "(no crop)", ha="center", va="center", fontsize=7)
        tag = (
            "T" if o["category"] == "target" else ("▲" if o["shape"]["concave"] else "")
        )
        ax.set_title(f"{o['name']} {tag}\n{o['shape']['family']}", fontsize=7)

    # -- 4) value features (g_val): pose + shape ----------------------------
    ax_val = fig.add_subplot(gs[2, 0:2])
    ax_val.axis("off")
    dwh = ex.provenance.get("drawer_wh")
    header = (
        f"{'object':<9}{'x':>7}{'y':>7}{'θ':>7}   {'w':>5}{'h':>5}{'area':>7} concave"
    )
    rows = [header, "-" * len(header)]
    for o in objs:
        px, py, pt = o["pose"]
        s = o["shape"]
        rows.append(
            f"{o['name']:<9}{px:>7.1f}{py:>7.1f}{pt:>7.2f}   "
            f"{s['w']:>5.1f}{s['h']:>5.1f}{s['area']:>7.1f}  {'yes' if s['concave'] else ''}"
        )
    ax_val.set_title(
        "Value features g_val — pose (x,y,θ) + shape  " f"[norm ref: drawer_wh={dwh}]",
        fontsize=11,
        loc="left",
    )
    ax_val.text(
        0.0,
        0.98,
        "\n".join(rows),
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.5,
        transform=ax_val.transAxes,
    )

    # -- 6) text vocab (g_text) + 7) provenance -----------------------------
    ax_meta = fig.add_subplot(gs[2, 2:4])
    ax_meta.axis("off")
    toks = _text_tokens(ex)
    tok_txt = "\n".join(f"  {k:<11}: {', '.join(v)}" for k, v in toks.items())
    r = ex.refine
    prov = ex.provenance
    meta_txt = (
        f"TEXT features g_text (CLIP-text vocabulary):\n{tok_txt}\n\n"
        f"LABEL: {'POSITIVE' if ex.label else 'NEGATIVE'}   (source: {ex.label_source})\n"
        f"REFINE: status={r.get('status')}  bound={r.get('steps_bound')}/{r.get('plan_length')}"
        f"  n_attempts={r.get('n_attempts')}  stuck@={r.get('failure_action')}\n"
        f"PROVENANCE: stratum(min_subset)={prov.get('stratum')}  n_items={prov.get('num_blocks')}"
        f"  planner={prov.get('planner')} ({prov.get('planner_search')}/{prov.get('planner_heuristic')})"
        f"\n            seed={prov.get('seed')}  split={prov.get('split')}  "
        f"refine_seed={prov.get('refine_seed')}"
    )
    ax_meta.text(
        0.0,
        0.98,
        meta_txt,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        transform=ax_meta.transAxes,
    )

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "record", help="record .json, or a problem/split/dataset dir (picks a record)"
    )
    ap.add_argument(
        "--out", default=None, help="output PNG (default: <record_dir>/inspect.png)"
    )
    args = ap.parse_args(argv)
    out = visualize_record(args.record, args.out)
    print(f"# wrote datapoint visualization -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
