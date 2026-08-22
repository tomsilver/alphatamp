"""Shelf3D difficulty tables -- presentation layer over ``shelf3d_collect.py`` output.

Reads the per-problem JSON records written by ``shelf3d_collect.py`` and renders, per
variant (o1/o2/o8), the three headline metrics under the baseline astar planner:

  * **solve rate** -- fraction of problems whose pool yielded a plan that refined within
    the budget;
  * **FP** (failed attempts before first success) -- mean +/- std over *solved* problems;
  * **wall-clock to first success** -- mean +/- std over *solved* problems.

    .venv/bin/marimo edit experiments/spectre/shelf3d_difficulty.py     # interactive
    .venv/bin/marimo run  experiments/spectre/shelf3d_difficulty.py     # read-only app
    SPECTRE_SHELF3D_DIR=data/spectre/shelf3D-kinder/_pilot \\
        .venv/bin/python experiments/spectre/shelf3d_difficulty.py       # headless render

For a **pilot** collection (``--pilot``: every candidate refined at a generous per-attempt
budget), a second section sweeps the per-attempt budget *offline*: because the refiner is
deterministic given its seed and monotone in the timeout, an attempt counts as a success
under budget ``T`` iff it actually succeeded with ``wall_clock_s <= T`` (charging
``min(wall_clock_s, T)`` per attempt) -- so one generous run yields the metrics at every
smaller budget, and the smallest budget where solve-rate/FP/wall plateau is the one to
deploy. This re-derivation is only valid for pilot (non-short-circuit) data; on a full
short-circuit run only the collection budget is meaningful.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
        # Shelf3D difficulty (baseline astar planner)

        How hard is vanilla `kinder/Shelf3D-o{1,2,8}-v0` (dynamic3d MuJoCo TidyBot) for the
        SPECTRE baseline planner? **FP** = failed refinement attempts before the first
        success; reported with **wall-clock to first success** and **solve rate**, per
        variant.
        """)
    return (mo,)


@app.cell
def _(mo):
    import os
    from pathlib import Path

    # Default to an absolute path derived from this notebook's own location, so
    # `marimo run`/`marimo edit` work regardless of the directory they are launched from
    # (a relative "data/..." default only resolves when run from the repo root).
    # SPECTRE_SHELF3D_DIR still overrides. This notebook lives at
    # <repo>/experiments/spectre/, so parents[2] is the repo root.
    _here = globals().get("__file__")
    _default = (
        str(Path(_here).resolve().parents[2] / "data" / "spectre" / "shelf3D-kinder")
        if _here
        else os.path.join("data", "spectre", "shelf3D-kinder")
    )
    dir_path = mo.ui.text(
        value=os.environ.get("SPECTRE_SHELF3D_DIR", _default),
        label="results dir (globs *.json recursively)",
        full_width=True,
    )
    dir_path
    return dir_path, os


@app.cell
def _(dir_path, mo, os):
    import glob
    import json

    def _load(root):
        # A pilot collection lives under a ``_pilot`` subdir of the full out-root, so when
        # pointed at the root we must exclude it (else full + pilot mix into one variant).
        # Pointing directly at ``.../_pilot`` includes it.
        root_is_pilot = "_pilot" in root.split(os.sep)
        by_variant = {}
        for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            if not root_is_pilot and "_pilot" in path.split(os.sep):
                continue
            try:
                rec = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue
            if "attempts" not in rec or "variant" not in rec:
                continue
            by_variant.setdefault(rec["variant"], []).append(rec)
        return by_variant

    by_variant = _load(dir_path.value)
    n_total = sum(len(v) for v in by_variant.values())
    mo.stop(
        not by_variant,
        mo.md(
            f"No Shelf3D result JSONs under `{dir_path.value}` -- run "
            "`python experiments/spectre/shelf3d_collect.py` first."
        ),
    )

    # Variant display order; the collection budget and mode (assumed uniform per dir).
    ORDER = [v for v in ("o1", "o2", "o8") if v in by_variant]
    ORDER += [v for v in by_variant if v not in ORDER]
    _all = [r for rs in by_variant.values() for r in rs]
    modes = sorted({r.get("mode", "full") for r in _all})
    budgets = sorted({float(r["per_attempt_budget_s"]) for r in _all})
    is_pilot = "pilot" in modes
    max_budget = max(budgets)
    mo.md(
        f"Loaded **{n_total}** problems across variants **{ORDER}** from "
        f"`{dir_path.value}` -- mode(s) `{modes}`, collection budget(s) "
        f"`{budgets}` s/attempt."
    )
    return ORDER, budgets, by_variant, is_pilot, max_budget


@app.cell
def _(ORDER, by_variant, max_budget, mo):
    import math

    def metrics_at_cap(records, cap):
        """Solve rate + FP/wall lists over solved problems, at per-attempt budget
        ``cap``.

        An attempt is a success under ``cap`` iff it succeeded with ``wall_clock_s <=
        cap``; each attempt is charged ``min(wall_clock_s, cap)``. Valid for smaller
        ``cap`` only on non-short-circuit (pilot) data.
        """
        fps, walls, n_solved = [], [], 0
        for r in records:
            cum, first_idx = 0.0, None
            for a in r["attempts"]:
                cum += min(a["wall_clock_s"], cap)
                if a["outcome"] == "success" and a["wall_clock_s"] <= cap:
                    first_idx = a["plan_idx"]
                    break
            if first_idx is not None:
                n_solved += 1
                fps.append(first_idx)
                walls.append(cum)
        n = len(records)
        return {
            "n": n,
            "n_solved": n_solved,
            "solve_rate": n_solved / n if n else 0.0,
            "fp": fps,
            "wall": walls,
        }

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def _std(xs):
        if len(xs) < 2:
            return float("nan")
        m = _mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    def _pm(xs, prec=2):
        if not xs:
            return "--"
        s = _std(xs)
        return (
            f"{_mean(xs):.{prec}f} ± {s:.{prec}f}"
            if not math.isnan(s)
            else (f"{_mean(xs):.{prec}f}")
        )

    header = (
        "| variant | n | solve rate | FP (mean ± std) | wall-to-first-success s |\n"
    )
    header += "|---|---|---|---|---|\n"
    body = ""
    for v in ORDER:
        g = metrics_at_cap(by_variant[v], max_budget)
        body += (
            f"| **{v}** | {g['n']} | {g['solve_rate']:.0%} ({g['n_solved']}/{g['n']}) "
            f"| {_pm(g['fp'])} | {_pm(g['wall'])} |\n"
        )
    mo.md(
        f"## Difficulty at {max_budget:g} s/attempt\n\n"
        + header
        + body
        + "\n\n*FP / wall-clock are over **solved** problems; unsolved problems set the "
        "solve rate. o1's pool is a single skeleton, so its FP is structurally 0.*"
    )
    return (metrics_at_cap,)


@app.cell
def _(ORDER, metrics_at_cap, by_variant, is_pilot, max_budget, mo):
    mo.stop(
        not is_pilot,
        mo.md(
            "*(Budget-saturation sweep shown only for `--pilot` data, where every candidate "
            "was refined so smaller budgets are re-derivable.)*"
        ),
    )

    _grid = [
        t for t in (1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60) if t <= max_budget + 1e-9
    ]
    if max_budget not in _grid:
        _grid.append(max_budget)
    _grid = sorted(set(_grid))

    def _sweep_table(metric):
        head = "| budget s | " + " | ".join(ORDER) + " |\n"
        head += "|" + "---|" * (len(ORDER) + 1) + "\n"
        rows = ""
        for cap in _grid:
            cells = []
            for v in ORDER:
                g = metrics_at_cap(by_variant[v], cap)
                if metric == "solve":
                    cells.append(f"{g['solve_rate']:.0%} ({g['n_solved']}/{g['n']})")
                elif metric == "fp":
                    xs = g["fp"]
                    cells.append(f"{sum(xs)/len(xs):.2f}" if xs else "--")
                else:  # wall
                    xs = g["wall"]
                    cells.append(f"{sum(xs)/len(xs):.2f}" if xs else "--")
            rows += f"| {cap:g} | " + " | ".join(cells) + " |\n"
        return head + rows

    mo.md(
        "## Budget saturation (pilot, offline re-derivation)\n\n"
        "Pick the smallest budget where these plateau per variant.\n\n"
        "### Solve rate vs per-attempt budget\n\n"
        + _sweep_table("solve")
        + "\n### Mean FP vs per-attempt budget\n\n"
        + _sweep_table("fp")
        + "\n### Mean wall-clock-to-first-success (s) vs per-attempt budget\n\n"
        + _sweep_table("wall")
    )
    return


if __name__ == "__main__":
    app.run()
