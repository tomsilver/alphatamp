"""Render representative samples of every DD2D shape family to a labelled grid.

Each family in :mod:`blocks_tamp.dd2d.shapes` is a *parametric distribution*, so two
instances of the same family differ in size / proportion / shape noise. This tool samples
``N_SAMPLES`` instances per family and lays them out one family per row, tagging the
concave families, so you can eyeball the shape library at a glance.

    PYTHONPATH=. .venv/bin/python -m blocks_tamp.dd2d.render_families
    # -> out/dd2d/shape_families.png
"""

from __future__ import annotations

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPoly

from alphatamp.approaches.spectre.envs.dd2d.dd2d.shapes import FAMILIES, sample_shape

N_SAMPLES = 3
HALF = (
    12.0  # axis half-extent (cm); shared across panels so relative sizes are comparable
)
_CONVEX_COLOR = "#4a7fb5"
_CONCAVE_COLOR = "#d98a3d"


def render(out_path: str, n_samples: int = N_SAMPLES, seed0: int = 1000) -> str:
    families = list(FAMILIES)
    nrows, ncols = len(families), n_samples
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.4, nrows * 2.4), squeeze=False
    )
    fig.suptitle(
        "DD2D shape families — 3 sampled instances each\n"
        "(each family is a parametric distribution; samples differ in size/proportion/noise)",
        fontsize=13,
        y=0.997,
    )
    seen_concave = False
    for r, fam in enumerate(families):
        rng = random.Random(seed0 + r)  # deterministic per family
        for c in range(ncols):
            ax = axes[r][c]
            shape = sample_shape(rng, family=fam)
            concave = shape.concave
            seen_concave |= concave
            color = _CONCAVE_COLOR if concave else _CONVEX_COLOR
            xs, ys = shape.polygon.exterior.xy
            ax.add_patch(
                MplPoly(
                    list(zip(xs, ys)),
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.2,
                    alpha=0.85,
                )
            )
            ax.set_xlim(-HALF, HALF)
            ax.set_ylim(-HALF, HALF)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            w, h = shape.size
            ax.set_title(f"{w:.1f}×{h:.1f} cm  A={shape.area:.0f}", fontsize=8)
            if c == 0:
                tag = f"{fam}\n(concave)" if concave else fam
                ax.set_ylabel(
                    tag,
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=30,
                )

    handles = [Patch(facecolor=_CONVEX_COLOR, edgecolor="black", label="convex family")]
    if seen_concave:
        handles.append(
            Patch(
                facecolor=_CONCAVE_COLOR,
                edgecolor="black",
                label="concave family (waist / L-corner / C-opening)",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/dd2d/shape_families.png")
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    path = render(args.out, n_samples=args.samples, seed0=args.seed)
    print("wrote", path)


if __name__ == "__main__":
    main()
