"""DD2D heuristic-arm comparison -- reactive presentation layer over the experiment
output.

Loads ``out_dd2d/heuristic_experiment/results.csv`` + ``run_meta.json`` (written by
``python -m blocks_tamp.dd2d.heuristic_experiment``) and compares the five enumeration arms
(bfs / astar-hff / gbf-hff / astar-dist / gbf-dist) on first-feasible rank:

  * success-rate bar chart with Wilson 95% CI error bars (the headline deliverable),
  * histogram of attempts-until-first-success (first-feasible rank),
  * a "solved within N refinements" CDF (solve-rate-vs-budget ramp),
  * a box plot of the rank distribution, and
  * a paired per-problem scatter (a geometric arm vs the blind baseline).

    .venv/bin/marimo edit blocks_tamp/dd2d/heuristic_notebook.py     # interactive
    .venv/bin/marimo run  blocks_tamp/dd2d/heuristic_notebook.py     # read-only app

The harness owns the numbers; this notebook only reads its CSV, so re-running the harness then
refreshing the notebook updates every figure.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # canonical arm order + colours (uninformed -> off-the-shelf symbolic -> hand-written geometric)
    ARM_ORDER = ["bfs", "astar-hff", "gbf-hff", "astar-dist", "gbf-dist"]
    ARM_COLOR = {
        "bfs": "#72767a",
        "astar-hff": "#4c78a8",
        "gbf-hff": "#7fa9d0",
        "astar-dist": "#e45756",
        "gbf-dist": "#f58518",
    }

    mo.md(
        r"""
        # DD2D — does the *kind* of heuristic reorder toward the feasible plan?

        Five enumeration arms on the same subset-requiring problems, compared on **first-feasible
        rank** (# skeletons refined until the first feasible one):

        | arm | search | heuristic | geometry? | hand-written? |
        |---|---|---|---|---|
        | `bfs` | BFS | none | no | no (current baseline) |
        | `astar-hff` | A* | pyperplan hFF | no | no (off-the-shelf) |
        | `gbf-hff` | GBF | pyperplan hFF | no | no (off-the-shelf) |
        | `astar-dist` | A* | distance prior | **yes (coarse)** | yes (simple) |
        | `gbf-dist` | GBF | distance prior | **yes (coarse)** | yes (simple) |

        If the *symbolic* heuristics (hFF) don't help but the *geometric* prior does, the useful
        signal is geometric — what PIGINet learns.
        """
    )
    return ARM_COLOR, ARM_ORDER, mo


@app.cell
def _(mo):
    import os

    csv_path = mo.ui.text(
        value=os.path.join("out_dd2d", "heuristic_experiment", "results.csv"),
        label="results.csv path",
        full_width=True,
    )
    csv_path
    return csv_path, os


@app.cell
def _(ARM_ORDER, csv_path, mo, os):
    import csv as _csv
    import json

    def _load(path):
        out = []
        with open(path) as f:
            for r in _csv.DictReader(f):
                r["first_feasible_rank"] = (
                    int(r["first_feasible_rank"]) if r["first_feasible_rank"] else None
                )
                r["solved"] = str(r["solved"]).lower() in ("true", "1")
                r["starved"] = str(r["starved"]).lower() in ("true", "1")
                r["num_skeletons"] = (
                    int(r["num_skeletons"]) if r["num_skeletons"] else 0
                )
                r["cum_calls"] = int(r["cum_calls"]) if r["cum_calls"] else 0
                r["min_feasible_subset"] = (
                    int(r["min_feasible_subset"]) if r["min_feasible_subset"] else None
                )
                out.append(r)
        return out

    rows = _load(csv_path.value) if os.path.exists(csv_path.value) else []
    _meta_path = os.path.join(os.path.dirname(csv_path.value), "run_meta.json")
    meta = json.load(open(_meta_path)) if os.path.exists(_meta_path) else {}
    mo.stop(
        not rows,
        mo.md(
            f"⚠️ No rows at `{csv_path.value}` — run "
            f"`python -m blocks_tamp.dd2d.heuristic_experiment` first."
        ),
    )

    arms = [_a for _a in ARM_ORDER if any(r["arm"] == _a for r in rows)]
    K = max((r["num_skeletons"] for r in rows), default=200)
    n_problems = len({r["problem_id"] for r in rows})
    _cfg = meta.get("config", {})
    mo.md(
        f"Loaded **{len(rows)}** rows · **{n_problems}** problems · **{len(arms)}** arms · k≈**{K}** "
        f"· min_subset≥**{_cfg.get('min_subset', '?')}** · crowd **{_cfg.get('crowd', '?')}** "
        f"· λ **{_cfg.get('lam', '?')}** · from `{csv_path.value}`"
    )
    return K, arms, meta, n_problems, rows


@app.cell
def _(arms, mo, rows):
    import math
    import statistics

    def wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0, 0.0)
        p = k / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (p, max(0.0, center - half), min(1.0, center + half))

    def summarize_arm(a):
        rs = [r for r in rows if r["arm"] == a]
        solved = [r for r in rs if r["solved"]]
        p, lo, hi = wilson(len(solved), len(rs))
        ranks = sorted(r["first_feasible_rank"] for r in solved)
        return {
            "n": len(rs),
            "n_solved": len(solved),
            "p": p,
            "lo": lo,
            "hi": hi,
            "mean": round(statistics.mean(ranks), 1) if ranks else None,
            "median": statistics.median(ranks) if ranks else None,
            "starved": sum(1 for r in rs if r["starved"]),
        }

    summ = {a: summarize_arm(a) for a in arms}
    _hdr = "| arm | n | solved | success | 95% CI | mean rank | median | starved |\n"
    _hdr += "|---|---|---|---|---|---|---|---|\n"
    _body = ""
    for _a in arms:
        _s = summ[_a]
        _body += (
            f"| `{_a}` | {_s['n']} | {_s['n_solved']} | **{_s['p']:.1%}** | "
            f"[{_s['lo']:.1%}, {_s['hi']:.1%}] | "
            f"{_s['mean'] if _s['mean'] is not None else '—'} | "
            f"{_s['median'] if _s['median'] is not None else '—'} | {_s['starved']} |\n"
        )
    mo.md("## Summary\n\n" + _hdr + _body)
    return (summ,)


@app.cell
def _(ARM_COLOR, arms, summ):
    import matplotlib.pyplot as plt
    import numpy as np

    fig_bar, ax_bar = plt.subplots(figsize=(7.5, 4.2))
    _x = np.arange(len(arms))
    _probs = [summ[_a]["p"] for _a in arms]
    _lo = [max(0.0, summ[_a]["p"] - summ[_a]["lo"]) for _a in arms]
    _hi = [max(0.0, summ[_a]["hi"] - summ[_a]["p"]) for _a in arms]
    ax_bar.bar(
        _x,
        _probs,
        yerr=[_lo, _hi],
        capsize=6,
        color=[ARM_COLOR.get(_a, "gray") for _a in arms],
        alpha=0.9,
    )
    for _xi, _a in zip(_x, arms):
        ax_bar.text(
            _xi,
            min(summ[_a]["p"] + _hi[int(_xi)] + 0.03, 1.02),
            f"{summ[_a]['p']:.0%}\n(n={summ[_a]['n']})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_xticks(_x)
    ax_bar.set_xticklabels(arms, rotation=15)
    ax_bar.set_ylabel("success rate (solved within k)")
    ax_bar.set_title("Success rate by arm (Wilson 95% CI)")
    ax_bar.grid(axis="y", alpha=0.3)
    fig_bar.tight_layout()
    ax_bar
    return np, plt


@app.cell
def _(ARM_COLOR, K, arms, np, plt, rows):
    fig_hist, ax_hist = plt.subplots(figsize=(8.5, 4.4))
    _bins = np.linspace(1, K, 26)
    for _a in arms:
        _solved = [
            r["first_feasible_rank"] for r in rows if r["arm"] == _a and r["solved"]
        ]
        if _solved:
            ax_hist.hist(
                _solved,
                bins=_bins,
                histtype="step",
                linewidth=2,
                color=ARM_COLOR.get(_a, "gray"),
                label=f"{_a} (solved)",
            )
    _cen = K * 1.06
    _w = K * 0.02
    _off = -_w * (len(arms) - 1) / 2
    for _i, _a in enumerate(arms):
        _un = sum(1 for r in rows if r["arm"] == _a and not r["solved"])
        if _un:
            ax_hist.bar(
                _cen + _off + _i * _w,
                _un,
                width=_w,
                color=ARM_COLOR.get(_a, "gray"),
                alpha=0.5,
                hatch="///",
                edgecolor="k",
            )
    ax_hist.axvline(K, color="k", ls=":", lw=1, alpha=0.6)
    ax_hist.set_xlabel(
        "first-feasible rank  (attempts until first success; ≥k = unsolved)"
    )
    ax_hist.set_ylabel("# problems")
    ax_hist.set_title("Attempts-until-first-success distribution by arm")
    ax_hist.legend(fontsize=8, ncol=2)
    fig_hist.tight_layout()
    ax_hist
    return


@app.cell
def _(ARM_COLOR, K, arms, np, plt, rows):
    fig_cdf, ax_cdf = plt.subplots(figsize=(8, 4.4))
    _budgets = np.arange(1, K + 1)
    for _a in arms:
        _rs = [r for r in rows if r["arm"] == _a]
        _n = len(_rs) or 1
        _ranks = [r["first_feasible_rank"] for r in _rs if r["solved"]]
        _frac = [sum(1 for _x in _ranks if _x <= _b) / _n for _b in _budgets]
        ax_cdf.plot(_budgets, _frac, color=ARM_COLOR.get(_a, "gray"), lw=2, label=_a)
    ax_cdf.set_xlabel("refinement budget N (skeletons refined, in arm order)")
    ax_cdf.set_ylabel("fraction of problems solved ≤ N")
    ax_cdf.set_title("Solved-within-N-refinements (solve-rate-vs-budget)")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend(fontsize=9)
    fig_cdf.tight_layout()
    ax_cdf
    return


@app.cell
def _(ARM_COLOR, arms, plt, rows):
    _data = [
        [r["first_feasible_rank"] for r in rows if r["arm"] == _a and r["solved"]]
        for _a in arms
    ]
    _present = [(_a, _d) for _a, _d in zip(arms, _data) if _d]
    fig_box, ax_box = plt.subplots(figsize=(7.5, 4.2))
    if _present:
        _bp = ax_box.boxplot(
            [_d for _a, _d in _present],
            tick_labels=[_a for _a, _d in _present],
            showmeans=True,
            patch_artist=True,
        )
        for _patch, (_a, _d) in zip(_bp["boxes"], _present):
            _patch.set_facecolor(ARM_COLOR.get(_a, "gray"))
            _patch.set_alpha(0.6)
    ax_box.set_ylabel("first-feasible rank (solved only)")
    ax_box.set_title("Rank distribution by arm")
    ax_box.tick_params(axis="x", rotation=15)
    ax_box.grid(axis="y", alpha=0.3)
    fig_box.tight_layout()
    ax_box
    return


@app.cell
def _(arms, mo):
    _geo = [_a for _a in arms if _a.endswith("dist")] or [
        _a for _a in arms if _a != "bfs"
    ]
    baseline_pick = mo.ui.dropdown(
        options=arms,
        value="bfs" if "bfs" in arms else arms[0],
        label="baseline arm (x)",
    )
    geo_pick = mo.ui.dropdown(
        options=arms, value=_geo[0] if _geo else arms[-1], label="comparison arm (y)"
    )
    mo.hstack([baseline_pick, geo_pick])
    return baseline_pick, geo_pick


@app.cell
def _(K, baseline_pick, geo_pick, mo, np, plt, rows):
    _ax_pick, _ay_pick = baseline_pick.value, geo_pick.value

    def _rank_by_problem(a):
        # unsolved -> plotted in the far corner at K*1.1 (censored)
        return {
            r["problem_id"]: (r["first_feasible_rank"] if r["solved"] else K * 1.1)
            for r in rows
            if r["arm"] == a
        }

    _rx, _ry = _rank_by_problem(_ax_pick), _rank_by_problem(_ay_pick)
    _common = sorted(set(_rx) & set(_ry))
    mo.stop(not _common, mo.md("No problems in common for the selected arms."))
    _xs = np.array([_rx[_p] for _p in _common])
    _ys = np.array([_ry[_p] for _p in _common])

    fig_sc, ax_scatter = plt.subplots(figsize=(6, 6))
    _lim = K * 1.18
    ax_scatter.plot([0, _lim], [0, _lim], "k--", alpha=0.5, lw=1, label="y = x")
    ax_scatter.scatter(
        _xs, _ys, alpha=0.6, color="#e45756", edgecolor="k", linewidth=0.3
    )
    ax_scatter.axvline(K, color="gray", ls=":", lw=1)
    ax_scatter.axhline(K, color="gray", ls=":", lw=1)
    _below = int(np.sum(_ys < _xs))
    ax_scatter.set_xlim(0, _lim)
    ax_scatter.set_ylim(0, _lim)
    ax_scatter.set_xlabel(f"{_ax_pick} first-feasible rank")
    ax_scatter.set_ylabel(f"{_ay_pick} first-feasible rank")
    ax_scatter.set_title(
        f"Per-problem rank: {_ay_pick} vs {_ax_pick}\n"
        f"({_below}/{len(_common)} better for {_ay_pick}; corner = unsolved)"
    )
    ax_scatter.legend()
    fig_sc.tight_layout()
    ax_scatter
    return


if __name__ == "__main__":
    app.run()
