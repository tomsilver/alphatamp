"""DD2D difficulty EDA -- reactive presentation layer over the harness output.

Loads ``out/dd2d_eda/episodes.csv`` (written by ``python -m blocks_tamp.dd2d.eda``) and
renders the two headline deliverables: the attempts-until-success (first-feasible-rank)
distribution and the per-stratum + pooled success probability with Wilson 95% CIs.

    .venv/bin/marimo edit blocks_tamp/dd2d/eda_notebook.py     # interactive
    .venv/bin/marimo run  blocks_tamp/dd2d/eda_notebook.py     # read-only app

The harness owns the numbers; this notebook only reads its CSV, so re-running the harness
then refreshing the notebook updates every figure.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
        # DD2D difficulty EDA on Pure Planning Baseline

        How hard is DD2D for a pure planning baseline?
        Stratified by `min_feasible_subset` ∈ {1, 2, 3} — the minimal number of blockers that *must*
        be moved to clear the target.
        """)
    return (mo,)


@app.cell
def _(mo):
    import os

    csv_path = mo.ui.text(
        value=os.path.join("out", "dd2d_eda", "episodes.csv"),
        label="episodes.csv path",
        full_width=True,
    )
    csv_path
    return csv_path, os


@app.cell
def _(csv_path, mo, os):
    import csv as _csv

    def _load(path):
        rows = []
        with open(path) as f:
            for r in _csv.DictReader(f):
                r["min_feasible_subset"] = (
                    int(r["min_feasible_subset"]) if r["min_feasible_subset"] else None
                )
                r["num_skeletons"] = (
                    int(r["num_skeletons"]) if r["num_skeletons"] else 0
                )
                r["first_feasible_rank"] = (
                    int(r["first_feasible_rank"]) if r["first_feasible_rank"] else None
                )
                r["solved"] = str(r["solved"]).lower() in ("true", "1")
                r["stream_calls"] = int(r["stream_calls"]) if r["stream_calls"] else 0
                rows.append(r)
        return rows

    _exists = os.path.exists(csv_path.value)
    rows = _load(csv_path.value) if _exists else []
    K = max((r["num_skeletons"] for r in rows), default=200)
    mo.stop(
        not rows,
        mo.md(
            f"⚠️ No rows at `{csv_path.value}` — run `python -m blocks_tamp.dd2d.eda` first."
        ),
    )
    mo.md(f"Loaded **{len(rows)}** episodes · k≈**{K}** · from `{csv_path.value}`")
    return K, rows


@app.cell
def _(K, mo, rows):
    import math
    import statistics

    # self-contained mirror of blocks_tamp.dd2d.eda.{wilson_ci,_summarize_group}
    def _wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0, 0.0)
        p = k / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (p, max(0.0, center - half), min(1.0, center + half))

    def _quant(xs, q):
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None

    def _group(rs):
        n = len(rs)
        solved = [r for r in rs if r["solved"]]
        p, lo, hi = _wilson(len(solved), n)
        ranks = sorted(int(r["first_feasible_rank"]) for r in solved)
        att = {"n_solved": len(ranks)}
        if ranks:
            att.update(
                mean=round(statistics.mean(ranks), 2),
                median=statistics.median(ranks),
                p25=_quant(ranks, 0.25),
                p75=_quant(ranks, 0.75),
                min=ranks[0],
                max=ranks[-1],
            )
        return {
            "n": n,
            "n_solved": len(solved),
            "success_prob": p,
            "ci95_low": lo,
            "ci95_high": hi,
            "unsolved_frac": (n - len(solved)) / n if n else 0.0,
            "attempts_until_success": att,
            "k": K,
        }

    STRATA = (1, 2, 3)
    groups = {
        s: _group([r for r in rows if r["min_feasible_subset"] == s]) for s in STRATA
    }
    groups = {s: g for s, g in groups.items() if g["n"] > 0}
    overall = _group(rows)

    def _fmt_att(g):
        a = g["attempts_until_success"]
        if not a.get("n_solved"):
            return "—"
        return f"mean {a['mean']} · median {a['median']} · p25 {a['p25']} · p75 {a['p75']} · max {a['max']}"

    header = (
        "| stratum | n | success | 95% CI | unsolved | attempts-to-first-success |\n"
    )
    header += "|---|---|---|---|---|---|\n"
    body = ""
    for _s, _g in groups.items():
        body += (
            f"| subset={_s} | {_g['n']} | **{_g['success_prob']:.1%}** | "
            f"[{_g['ci95_low']:.1%}, {_g['ci95_high']:.1%}] | {_g['unsolved_frac']:.0%} | {_fmt_att(_g)} |\n"
        )
    body += (
        f"| **overall** | {overall['n']} | **{overall['success_prob']:.1%}** | "
        f"[{overall['ci95_low']:.1%}, {overall['ci95_high']:.1%}] | {overall['unsolved_frac']:.0%} | "
        f"{_fmt_att(overall)} |\n"
    )
    mo.md("## Summary\n\n" + header + body)
    return groups, overall


@app.cell
def _(groups, overall):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"subset={_s}" for _s in groups] + ["overall"]
    gs = list(groups.values()) + [overall]
    probs = [_g["success_prob"] for _g in gs]
    # error-bar *lengths* (distance from the bar top) must be >= 0; clamp tiny
    # floating-point negatives that arise when p sits on the CI edge (e.g. p=1.0,
    # where the Wilson upper bound rounds to 0.99999999 -> hi = -1e-8).
    lo = [max(0.0, _g["success_prob"] - _g["ci95_low"]) for _g in gs]
    hi = [max(0.0, _g["ci95_high"] - _g["success_prob"]) for _g in gs]
    ns = [_g["n"] for _g in gs]
    colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b"][: len(gs) - 1] + ["#72767a"]

    fig_prob, ax_prob = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax_prob.bar(x, probs, yerr=[lo, hi], capsize=6, color=colors, alpha=0.9)
    for _xi, _p, _n in zip(x, probs, ns):
        ax_prob.text(
            _xi,
            min(_p + max(hi) + 0.03, 1.02),
            f"{_p:.0%}\n(n={_n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_prob.set_ylim(0, 1.15)
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(labels)
    ax_prob.set_ylabel("success probability (solved within k)")
    ax_prob.set_title("DD2D pyperplan success probability (Wilson 95% CI)")
    ax_prob.grid(axis="y", alpha=0.3)
    fig_prob.tight_layout()
    ax_prob
    return np, plt


@app.cell
def _(K, groups, np, plt, rows):
    # Attempts-until-success (first-feasible rank) distribution, by stratum,
    # with a censored "≥k (unsolved)" bar at the far right.
    strata = list(groups)
    palette = {1: "#4c78a8", 2: "#f58518", 3: "#e45756"}
    solved_ranks = {
        _s: [
            r["first_feasible_rank"]
            for r in rows
            if r["min_feasible_subset"] == _s and r["solved"]
        ]
        for _s in strata
    }
    unsolved_n = {
        _s: sum(1 for r in rows if r["min_feasible_subset"] == _s and not r["solved"])
        for _s in strata
    }

    fig_hist, ax_hist = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(1, K, 26)
    for _s in strata:
        if solved_ranks[_s]:
            ax_hist.hist(
                solved_ranks[_s],
                bins=bins,
                histtype="stepfilled",
                alpha=0.5,
                color=palette.get(_s, "gray"),
                label=f"subset={_s} (solved)",
            )
    # censored bar(s): unsolved counts, hatched, placed just past k
    cen_x = K * 1.06
    width = K * 0.03
    offset = -width * (len(strata) - 1) / 2
    for _i, _s in enumerate(strata):
        if unsolved_n[_s]:
            ax_hist.bar(
                cen_x + offset + _i * width,
                unsolved_n[_s],
                width=width,
                color=palette.get(_s, "gray"),
                alpha=0.5,
                hatch="///",
                edgecolor="k",
                label=f"subset={_s} (≥k unsolved)",
            )
    ax_hist.axvline(K, color="k", ls=":", lw=1, alpha=0.6)
    ax_hist.set_xlabel("first-feasible rank  (attempts until success)")
    ax_hist.set_ylabel("# episodes")
    ax_hist.set_title(
        "Attempts-until-success distribution (pyperplan; ≥k = unsolved within budget)"
    )
    ax_hist.legend(fontsize=8, ncol=2)
    fig_hist.tight_layout()
    ax_hist
    return


if __name__ == "__main__":
    app.run()
