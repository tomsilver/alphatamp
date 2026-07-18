import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # DD2D — SPECTRE vs PIGINet vs Pure Planning (astar-dist) (held-out test rollout FPs)

    Compares three plan-feasibility methods on the **held-out DD2D test split**
    by **rollout false-positives (FP)** —
    the number of failed refinement attempts before the first success.

    **Methods.**

    - **astar-dist** —  the non-learned baseline.
    - **PIGINet** — low-level predictor (CLIP + transformer over
      object image features + literals). Static one-shot ranking.
    - **SPECTRE-adaptive** — the trained SPECTRE re-ranker in its deployment mode:
      re-ranks the pool after every failure.
    - **SPECTRE-static** — SPECTRE ranked once at the empty failure context; the
      strict same-policy comparator to PIGINet.
    """
    )
    return


@app.cell
def _(mo):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots  # noqa: F401  (registers the 'science' style)
    import seaborn as sns

    from alphatamp.approaches.spectre import dd2d_compare

    sns.set_theme(context="notebook", style="whitegrid")
    plt.style.use(["science", "no-latex", "nature"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    _nb = mo.notebook_dir()
    REPO = (_nb / ".." / "..").resolve() if _nb is not None else Path("..").resolve()
    CACHE_DIR = REPO / "data" / "spectre" / "derived" / "dd2d_v2" / "compare_cache"

    METHODS = dd2d_compare.METHOD_ORDER
    COLORS = {
        "astar-dist": "#7f7f7f",
        "PIGINet_v3": "#ff7f0e",
        "SPECTRE-adaptive": "#1f77b4",
        "SPECTRE-static": "#7fb8de",
    }
    STRATA = [0, 1, 2, 3]
    print(f"cache: {CACHE_DIR}")
    return (
        CACHE_DIR,
        COLORS,
        METHODS,
        STRATA,
        dd2d_compare,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1 · Load cached per-problem FP

    Reads the precomputed raw scores and derives one rollout-FP per (method,
    problem) from
    `data/spectre/derived/dd2d_v2/compare_cache/`.
    """
    )
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, pd):
    df = pd.DataFrame(dd2d_compare.load_fp_records(CACHE_DIR))
    print(df.groupby(["method", "stratum"]).size().unstack())
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2 · Summary table 

    Mean ± std (across problems) FP per method per stratum.
    """
    )
    return


@app.cell
def _(METHODS, STRATA, df, np, pd):
    def _cell(sub):
        return f"{sub['fp'].mean():.2f} ± {sub['fp'].std():.2f}" if len(sub) else "—"

    _rows = []
    for _method in METHODS:
        _row = {"method": _method}
        for _k in STRATA:
            _row[f"s{_k}"] = _cell(df[(df.method == _method) & (df.stratum == _k)])
        _row["ALL"] = _cell(df[df.method == _method])
        _rows.append(_row)
    summary_df = pd.DataFrame(_rows).set_index("method")

    _astar_all = df[df.method == "astar-dist"]["fp"].mean()
    _pig_all = df[df.method == "PIGINet_v3"]["fp"].mean()
    _ok = np.isclose(_astar_all, 33.01, atol=0.5) and np.isclose(
        _pig_all, 20.39, atol=0.5
    )
    print(
        f"ballpark check  astar ALL={_astar_all:.2f} (exp 33.01)  "
        f"PIGINet ALL={_pig_all:.2f} (exp 20.39)  -> "
        f"{'OK' if _ok else 'DEVIATION!'}"
    )
    summary_df
    return (summary_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3 · Mean FP per stratum (± std-dev error bars)

    Lower is better. Error bars are the across-problem std-dev, clipped at 0.
    """
    )
    return


@app.cell
def _(COLORS, METHODS, STRATA, df, np, plt):
    _groups = [str(k) for k in STRATA] + ["ALL"]
    _x = np.arange(len(_groups))
    _w = 0.2
    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    for _i, _m in enumerate(METHODS):
        _means, _stds = [], []
        for _k in STRATA:
            _s = df[(df.method == _m) & (df.stratum == _k)]["fp"]
            _means.append(_s.mean())
            _stds.append(_s.std())
        _all = df[df.method == _m]["fp"]
        _means.append(_all.mean())
        _stds.append(_all.std())
        _means = np.array(_means)
        _stds = np.array(_stds)
        # FP >= 0: clip the lower whisker so the bar bottom never dips below zero.
        _lower = np.minimum(_stds, _means)
        _ax.bar(
            _x + (_i - 1.5) * _w,
            _means,
            _w,
            yerr=[_lower, _stds],
            capsize=2,
            label=_m,
            color=COLORS[_m],
            error_kw={"elinewidth": 0.8},
        )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_groups)
    _ax.set_ylim(bottom=0)
    _ax.set_xlabel("min-feasible-subset stratum")
    _ax.set_ylabel("rollout FP (fails before first success)")
    _ax.set_title("Mean rollout FP by stratum (DD2D test, n=124)")
    _ax.legend(ncol=2)
    plt.tight_layout()
    plt.gca()
    return


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(
#         r"""
#     ## 4 · FP distributions (violin plots)
#     """
#     )
#     return


# @app.cell
# def _(COLORS, METHODS, STRATA, df, plt, sns):
#     _panels = STRATA + ["ALL"]
#     _fig, _axes = plt.subplots(1, len(_panels), figsize=(15, 3.6), sharey=False)
#     for _ax, _k in zip(_axes, _panels):
#         _sub = df if _k == "ALL" else df[df.stratum == _k]
#         sns.violinplot(
#             data=_sub,
#             x="method",
#             y="fp",
#             order=METHODS,
#             hue="method",
#             palette=COLORS,
#             legend=False,
#             cut=0,
#             inner="quartile",
#             density_norm="width",
#             ax=_ax,
#         )
#         _ax.set_ylim(bottom=0)
#         _ax.set_title(f"stratum {_k}" if _k != "ALL" else "ALL strata")
#         _ax.set_xlabel("")
#         _ax.set_ylabel("rollout FP" if _k == 0 else "")
#         _ax.set_xticklabels(
#             [m.replace("SPECTRE-", "SP-") for m in METHODS],
#             rotation=45,
#             ha="right",
#         )
#     plt.tight_layout()
#     plt.gca()
#     return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4 · Survival curves

    Fraction of problems solved within ≤ k failed attempts (higher & further-left
    is better).
    """
    )
    return


@app.cell
def _(COLORS, METHODS, STRATA, df, np, plt):
    def _cdf(vals, ks):
        vals = np.asarray(vals)
        return [np.mean(vals <= k) for k in ks]

    _ks = np.arange(0, 201)
    _panels = ["ALL"] + STRATA
    _fig, _axes = plt.subplots(1, len(_panels), figsize=(15, 3.4), sharey=True)
    for _ax, _k in zip(_axes, _panels):
        _sub = df if _k == "ALL" else df[df.stratum == _k]
        for _m in METHODS:
            _v = _sub[_sub.method == _m]["fp"].values
            _ax.plot(_ks, _cdf(_v, _ks), label=_m, color=COLORS[_m])
        _ax.set_title("ALL strata" if _k == "ALL" else f"stratum {_k}")
        _ax.set_xlabel("failed attempts k")
        _ax.set_ylim(0, 1.02)
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("P(FP ≤ k)")
    _axes[0].legend(loc="lower right")
    plt.tight_layout()
    plt.gca()
    return


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(
#         r"""
#     ## 5 · Paired per-problem comparison (SPECTRE-adaptive vs the others)

#     Points below the y=x line are
#     problems where SPECTRE-adaptive has fewer FPs (wins).
#     """
#     )
#     return


# @app.cell
# def _(COLORS, STRATA, df, plt):
#     _piv = df.pivot_table(index="problem_id", columns="method", values="fp")
#     _strat = _piv.index.to_series().map(
#         lambda s: min(3, (int(s) - 1_000_000) // 250_000)
#     )
#     _cmap = {0: "#4c72b0", 1: "#55a868", 2: "#c44e52", 3: "#8172b3"}

#     def _scatter(ax, xcol):
#         for _k in STRATA:
#             _mask = _strat == _k
#             ax.scatter(
#                 _piv.loc[_mask, xcol],
#                 _piv.loc[_mask, "SPECTRE-adaptive"],
#                 s=18,
#                 alpha=0.7,
#                 color=_cmap[_k],
#                 label=f"stratum {_k}",
#                 edgecolors="none",
#             )
#         _lim = max(_piv[xcol].max(), _piv["SPECTRE-adaptive"].max()) * 1.1
#         ax.plot([0, _lim], [0, _lim], "k--", lw=0.8, alpha=0.6)
#         _wins = int((_piv["SPECTRE-adaptive"] < _piv[xcol]).sum())
#         _losses = int((_piv["SPECTRE-adaptive"] > _piv[xcol]).sum())
#         _ties = int((_piv["SPECTRE-adaptive"] == _piv[xcol]).sum())
#         ax.set_xscale("symlog", linthresh=1)
#         ax.set_yscale("symlog", linthresh=1)
#         ax.set_xlabel(f"{xcol} FP")
#         ax.set_ylabel("SPECTRE-adaptive FP")
#         ax.set_title(f"vs {xcol}\nSPECTRE wins {_wins} / tie {_ties} / lose {_losses}")

#     _ = COLORS  # keep palette dep explicit for marimo
#     _fig, _axes = plt.subplots(1, 2, figsize=(11, 5))
#     _scatter(_axes[0], "PIGINet_v3")
#     _scatter(_axes[1], "astar-dist")
#     _axes[0].legend(loc="upper left", fontsize=7)
#     plt.tight_layout()
#     plt.gca()
#     return


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"## 7 · Export per-problem FPs")
#     return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5 · T0 — Length-R² (does plan length alone explain each ranking?)

    Test **T0** from `docs/spectre_piginet_hypotheses_and_tests_v2.md`, run for **all
    four** methods. A DD2D plan's *length* is its operator count
    (`2·(blockers staged)+1`). For each method's one-shot per-skeleton scores we
    measure, per episode:

    - **η²(length)** — fraction of score variance explained by length group
      (categorical R²). η² ≈ 1 ⇒ the score is *a function of length* — the ranking
      cannot tell same-length (same-size-subset) plans apart.
    - **within-length frac** = 1 − η² — the share of score variance that *is*
      same-length discrimination (geometry / subset signal).
    - **Spearman(score, length)** — sign = long-plan preference (`+` prefers longer).

    H1 predicts SPECTRE-static ≈ a pure length ranking (η² ≈ 1). The **target-flag**
    feature of the elimination argument is degenerate here (every plan ends in
    `retrieve(target)`; none picks it), so length is the only non-trivial
    cross-candidate feature.

    SPECTRE-adaptive has no static per-skeleton scores; its **t=0** ranking is
    *provably identical* to SPECTRE-static (same checkpoint, empty failure context →
    `c₀`), so its one-shot row mirrors static. Its deployed behaviour is shown
    separately as a **realized-order length ladder** (does it climb to longer plans
    as it fails?).
    """
    )
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, pd):
    from alphatamp.approaches.spectre import eda as _eda

    # Plan lengths per problem, aligned to the cached score index (skeleton_idx).
    _spectre_test = CACHE_DIR.parents[2] / "raw" / "dd2d_v2" / "test"
    _episodes = _eda.load_split_episodes(_spectre_test).episodes
    lengths_by_pid = {
        int(ep.provenance.problem_id): [len(s.operator_seq) for s in ep.skeleton_pool]
        for ep in _episodes
    }
    fit_df = pd.DataFrame(
        dd2d_compare.load_length_fit_records(CACHE_DIR, lengths_by_pid)
    )
    pos_df = pd.DataFrame(
        dd2d_compare.load_position_by_length_records(CACHE_DIR, lengths_by_pid)
    )
    ladder_df = pd.DataFrame(
        dd2d_compare.load_adaptive_ladder_records(CACHE_DIR, lengths_by_pid)
    )
    print(
        f"lengths for {len(lengths_by_pid)} problems; "
        f"fit rows={len(fit_df)}  ladder rows={len(ladder_df)}"
    )
    return fit_df, ladder_df, pos_df


@app.cell
def _(STRATA, fit_df, np, pd):
    def _m(sub, col):
        return f"{sub[col].mean():.2f}" if len(sub) else "—"

    _rows = []
    for _method in ["astar-dist", "PIGINet_v3", "SPECTRE-static"]:
        _sub = fit_df[fit_df.method == _method]
        _row = {
            "method": _method,
            "η²(len)": _m(_sub, "eta2"),
            "within-len": _m(_sub, "within_frac"),
            "spearman": _m(_sub, "spearman"),
        }
        for _k in STRATA:
            _row[f"η² s{_k}"] = _m(_sub[_sub.stratum == _k], "eta2")
        _rows.append(_row)
    # SPECTRE-adaptive one-shot == SPECTRE-static (c₀ identity).
    _stat = fit_df[fit_df.method == "SPECTRE-static"]
    _arow = {
        "method": "SPECTRE-adaptive (t=0 ≡ static)",
        "η²(len)": _m(_stat, "eta2"),
        "within-len": _m(_stat, "within_frac"),
        "spearman": _m(_stat, "spearman"),
    }
    for _k in STRATA:
        _arow[f"η² s{_k}"] = _m(_stat[_stat.stratum == _k], "eta2")
    _rows.append(_arow)
    t0_table = pd.DataFrame(_rows).set_index("method")

    _s_eta = fit_df[fit_df.method == "SPECTRE-static"]["eta2"].mean()
    _a_sp = fit_df[fit_df.method == "astar-dist"]["spearman"].mean()
    _ok = np.isclose(_s_eta, 1.00, atol=0.01) and np.isclose(_a_sp, -0.86, atol=0.03)
    print(
        f"T0 self-check  static η²={_s_eta:.3f} (exp 1.00)  "
        f"astar spearman={_a_sp:.3f} (exp -0.86)  -> {'OK' if _ok else 'DEVIATION!'}"
    )
    t0_table
    return (t0_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Learned length curve

    Mean within-episode attempt-position (0 = tried first … 1 = tried last) of each
    plan-length tier. Reveals *which* length policy each method learned: astar is
    monotone short-first; PIGINet is ~flat (it discriminates *within* length, not by
    it); SPECTRE-static is a non-monotone length lookup (front-loads 1- and
    3-blocker plans, buries 2-blocker plans).
    """
    )
    return


@app.cell
def _(COLORS, pos_df, plt):
    _fig, _ax = plt.subplots(figsize=(6.2, 4.0))
    for _m in ["astar-dist", "PIGINet_v3", "SPECTRE-static"]:
        _sub = pos_df[pos_df.method == _m].sort_values("length")
        _ax.plot(
            _sub["length"],
            _sub["mean_pos"],
            marker="o",
            label=_m,
            color=COLORS[_m],
        )
    _ax.set_xticks([1, 3, 5, 7])
    _ax.set_xticklabels(["1 (m0)", "3 (m1)", "5 (m2)", "7 (m3)"])
    _ax.set_xlabel("plan length (blockers staged)")
    _ax.set_ylabel("mean attempt-position (0 = tried first)")
    _ax.set_ylim(0, 1)
    _ax.set_title("Learned length curve (DD2D test)")
    _ax.legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(COLORS, fit_df, np, plt):
    # η² vs within-length: how much of the ranking is length vs same-length signal.
    _methods = ["astar-dist", "PIGINet_v3", "SPECTRE-static"]
    _eta = [fit_df[fit_df.method == m]["eta2"].mean() for m in _methods]
    _within = [fit_df[fit_df.method == m]["within_frac"].mean() for m in _methods]
    _x = np.arange(len(_methods))
    _fig, _ax = plt.subplots(figsize=(6.2, 4.0))
    _ax.bar(_x - 0.2, _eta, 0.4, label="η² (length)", color="#4c72b0")
    _ax.bar(
        _x + 0.2,
        _within,
        0.4,
        label="within-length (subset signal)",
        color="#c44e52",
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels([m.replace("SPECTRE-", "SP-") for m in _methods])
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("fraction of score variance")
    _ax.set_title("Length vs same-length signal per method")
    _ax.legend()
    _ = COLORS  # keep palette dep explicit for marimo
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### SPECTRE-adaptive — realized-order length ladder

    Over the sequence it *actually attempts* (until first success), does adaptive
    climb to longer plans as it fails? Positive `spearman(position, length)` ⇒ a
    length-escalation ladder. (Reads the cached `order` trace; averaged over seeds.)
    """
    )
    return


@app.cell
def _(STRATA, ladder_df, pd):
    def _c(sub, col):
        return f"{sub[col].mean():.2f}" if len(sub) else "—"

    _rows = []
    for _label, _col in [
        ("spearman(position, length)", "spearman"),
        ("mean first-attempt length", "first_len"),
        ("mean last-attempt length", "last_len"),
        ("mean # attempts", "n_steps"),
    ]:
        _row = {"metric": _label}
        for _k in STRATA:
            _row[f"s{_k}"] = _c(ladder_df[ladder_df.stratum == _k], _col)
        _row["ALL"] = _c(ladder_df, _col)
        _rows.append(_row)
    adaptive_ladder_table = pd.DataFrame(_rows).set_index("metric")
    adaptive_ladder_table
    return (adaptive_ladder_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaway (T0)

    - **SPECTRE-static is, mechanically, a learned plan-length lookup** (η² ≈ 1.00,
      within-length ≈ 0): it ranks by *how many* blockers a plan stages and is blind
      to *which* same-size subset is correct. Its s3 win over astar-dist is a length
      re-ordering (3-blocker plans above 2-blocker plans), not subset knowledge — and
      by the elimination argument, length was the only signal it *could* use.
    - The learned length curve is **non-monotone** (front-load m1/m3, bury m2), not a
      simple "prefer longer" — a mild surprise worth a follow-up, but still 100%
      length.
    - **PIGINet is the structural opposite** (~79% within-length variance): it uses
      geometry to rank same-length plans. So SPECTRE's headline win means "a length
      prior beat PIGINet's geometry on this stratum mix," not "SPECTRE discriminates
      subsets better" (it cannot at all).
    - **Adaptive t=0 ≡ static**; its only lever is the realized-order re-ranking above.

    This motivates the v2.1 typed-evidence design: breaking the same-length symmetry
    needs diagnostic signal the current abstract `[s0, ops, sL]` representation does
    not carry.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6 · T1 — length-only-context intervention (does c_t use identity?)

    Test **T1** (length-only variant) from
    `docs/spectre_piginet_hypotheses_and_tests_v2.md`. Rerun the SPECTRE-adaptive
    rollout, but replace every failed skeleton in the failure context Ψ sees with a
    **random *other* pool skeleton of the same plan length** — correct length,
    random object identity, consistent `s_L` (it is a real pooled skeleton). The
    selection still masks the *real* attempted skeletons; only the context is
    scrambled (`eda.spectre_evaluate_length_only_context`, averaged over 3 seeds × 3
    surrogate draws in `spectre_lenctx/`).

    **If mean rollout-FP is unchanged ⇒ Ψ ignores which objects failed and uses only
    size/length (H2).** A drop ⇒ identity carries signal → escalate to the full T1
    suite. (Regression guard: `scramble=False` reproduces SPECTRE-adaptive bitwise.)
    """
    )
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, df, pd):
    _lc = pd.DataFrame(
        dd2d_compare.load_named_fp_records(
            CACHE_DIR, "spectre_lenctx", "SPECTRE-adaptive-lenctx"
        )
    )
    _ad = df[df.method == "SPECTRE-adaptive"][["problem_id", "stratum", "fp"]].rename(
        columns={"fp": "fp_adaptive"}
    )
    t1_df = _ad.merge(
        _lc[["problem_id", "fp"]].rename(columns={"fp": "fp_lenctx"}),
        on="problem_id",
    )
    print(f"T1 paired problems: {len(t1_df)}")
    return (t1_df,)


@app.cell
def _(STRATA, pd, t1_df):
    from alphatamp.approaches.spectre import eda as _eda

    def _row(sub, label):
        a = sub["fp_adaptive"].values
        lc = sub["fp_lenctx"].values
        delta = _eda.bootstrap_mean_difference(lc, a, num_resamples=10_000, seed=0)
        return {
            "stratum": label,
            "n": len(sub),
            "adaptive": f"{a.mean():.2f}",
            "len-only ctx": f"{lc.mean():.2f}",
            "Δ (lc−ad)": f"{delta.point:+.2f}",
            "95% CI": f"[{delta.ci_low:+.2f}, {delta.ci_high:+.2f}]",
        }

    _rows = [_row(t1_df[t1_df.stratum == _k], f"s{_k}") for _k in STRATA]
    _rows.append(_row(t1_df, "ALL"))
    t1_table = pd.DataFrame(_rows).set_index("stratum")
    t1_table
    return (t1_table,)


@app.cell
def _(plt, t1_df):
    _cmap = {0: "#4c72b0", 1: "#55a868", 2: "#c44e52", 3: "#8172b3"}
    _fig, _ax = plt.subplots(figsize=(5.0, 5.0))
    for _k in sorted(t1_df.stratum.unique()):
        _s = t1_df[t1_df.stratum == _k]
        _ax.scatter(
            _s["fp_adaptive"],
            _s["fp_lenctx"],
            s=18,
            alpha=0.7,
            color=_cmap[_k],
            label=f"s{_k}",
            edgecolors="none",
        )
    _lim = max(t1_df["fp_adaptive"].max(), t1_df["fp_lenctx"].max()) * 1.05
    _ax.plot([0, _lim], [0, _lim], "k--", lw=0.8, alpha=0.6)
    _ax.set_xlabel("SPECTRE-adaptive FP (real context)")
    _ax.set_ylabel("length-only-context FP")
    _ax.set_title(
        "T1: identity-scrambled vs real context\n(points on y=x ⇒ identity unused)"
    )
    _ax.legend(title="stratum")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(t1_df):
    from alphatamp.approaches.spectre import eda as _eda

    _a = t1_df["fp_adaptive"].values
    _lc = t1_df["fp_lenctx"].values
    _d = _eda.bootstrap_mean_difference(_lc, _a, num_resamples=10_000, seed=0)
    _rel = abs(_d.point) / max(1e-9, _a.mean())
    _unused = (_d.ci_low <= 0 <= _d.ci_high) and _rel < 0.02
    print(
        f"T1 verdict  ALL Δ(lc−ad)={_d.point:+.3f} FP  "
        f"95% CI [{_d.ci_low:+.3f}, {_d.ci_high:+.3f}]  |Δ|/adaptive={_rel:.1%}"
    )
    print(
        "=> "
        + (
            "IDENTITY UNUSED — H2 confirmed (Ψ uses length/size only)"
            if _unused
            else "identity matters — escalate to full T1 suite"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Takeaway (T1)

    Scrambling the *identity* of every failed skeleton to a random same-length plan
    leaves SPECTRE-adaptive's rollout-FP **unchanged** (Δ ≈ 0, CI includes 0 in every
    stratum). The context embedding `c_t` does shift a hair under scrambling
    (‖Δc‖ ~ 5e-3 on the d=64 vector), but never enough to change a selection — so the
    context module Ψ carries only a *whisper* of identity and is, functionally,
    **size/length-only**. This confirms **H2**: the adaptive gains (concentrated on
    s0/s1) are length-regime escalation, not failed-subset identification.

    Combined with T0 (SPECTRE-static ≈ a pure length ranking, η²≈1.0), SPECTRE on
    DD2D uses **length and nothing identity-specific** — neither in the static ranker
    nor in the adaptive context. Breaking the same-length symmetry (telling *which*
    same-size subset is right) needs diagnostic evidence the abstract representation
    does not carry — the v2.1 typed-evidence motivation.
    """
    )
    return


@app.cell
def _(df, mo):
    _out = mo.notebook_dir() / "dd2d_method_comparison.csv"
    df.to_csv(_out, index=False)
    print(f"wrote {_out}  ({len(df)} rows)")
    return


if __name__ == "__main__":
    app.run()
