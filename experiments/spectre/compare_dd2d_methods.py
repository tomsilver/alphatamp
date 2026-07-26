import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# DD2D — SPECTRE v1 / v2.2 vs PIGINet vs Pure Planning (astar-dist)

          Compares six plan-feasibility methods on the **held-out DD2D test split**
          (n = 100) by **rollout false-positives (FP)** — the number of failed
          refinement attempts before the first success.

          **Methods.**

          - **astar-dist** — the non-learned planner-order baseline (score = −plan_idx).
          - **PIGINet** — low-level predictor (CLIP + transformer over object image
            features + literals), **trained with BCE** (the original-paper baseline
            loss), AUPRC-selected. Static one-shot ranking.
          - **SPECTRE-adaptive / -static** (v1) — the abstract-only re-ranker in its two
            deployment modes: adaptive re-ranks the pool after every failure; static
            ranks once at the empty failure context (c₀).
          - **SPECTREv2-adaptive / -static** — the v2.2 geometry- + typed-evidence
            re-ranker, **observed** proof-demotion. Adaptive = `deployed_rollout`
            (model scores + sound proof-demotion); static = empty-context logits.
          - **VLMPlan-8B / -32B** — the zero-shot VLM baseline (KinDER convention): the
            zero-training-data, generic-perception corner of the data × perception grid.
            Two arms of the same family, `Qwen3-VL-{8B,32B}-Instruct`, so the pair is a
            **scale** comparison. Optional — each row appears only when its cache exists.

          > **Collection.** This reads **dd2d_v3**, the re-collection made after the
          > 2026-07-24 grasp-model changes. Every learned method here was retrained on v3
          > and re-scored on its test split, so the table is internally consistent — the
          > older v2 numbers are not comparable to these.

          > **1-seed dev.** v1 and v2 each ship a single checkpoint seed on disk, so
          > every learned row here is a **1-seed** figure — for iteration, not a
          > writeup-reportable mean±std.

          > **Why both arms are Qwen3-VL Instruct.** A `gemma-4-31b-qat` arm was tried
          > and rejected: it is a *reasoning* model, spending ~95% of its output budget on
          > hidden thinking tokens that count against `max_tokens` but never reach the
          > parser. Holding the family fixed keeps this a scale comparison and keeps the
          > prompt/parse behaviour identical across arms. Avoid Qwen3-VL **Thinking**
          > variants for the same reason.

          > **VLMPlan is scored differently, on purpose.** Every other method *reorders*
          > the shared 200-candidate pool, so its FP counts only in-pool attempts.
          > VLMPlan *generates* its own plans, and most multi-item stagings are not in the
          > pool at all, so its FP **counts off-pool attempts too** (labelled by a live
          > refiner on the reconstructed scene). It can therefore reach plans the pool
          > does not contain — an edge at s3 — while paying for every wrong guess. That
          > asymmetry is the honest accounting, but it is an asymmetry. For the same
          > reason VLMPlan is absent from the T0 length sections below, which need a score
          > for each of the 200 pool candidates.
          """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
          > ⚠️ **The §5 (T0) and §6 (T1) prose below was written against dd2d_v2.** The
          > tables and figures now render **v3** data, but the surrounding takeaways still
          > quote v2 numbers (η² ≈ 1.0, Δ ≈ 0 per stratum, n = 142). Read those cells as
          > *hypotheses to re-check against the v3 figures beside them*, not as v3
          > conclusions. Re-deriving them on v3 is a separate task.
          """)
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
    # Which DD2D collection this notebook reads. dd2d_v3 is the re-collection made after
    # the 2026-07-24 grasp-model changes (contact-run fix + blocky horseshoe + internal
    # concave grasps); every learned checkpoint here was retrained and re-scored on it.
    # dd2d_v2 is the prior collection (stale labels) — switching back is a one-line edit,
    # but a variant's caches only exist for methods actually re-run against it.
    ENV_VARIANT = "dd2d_v3"
    CACHE_DIR = REPO / "data" / "spectre" / "derived" / ENV_VARIANT / "compare_cache"

    METHODS = dd2d_compare.METHOD_ORDER
    COLORS = {
        "astar-dist": "#7f7f7f",
        "PIGINet": "#ff7f0e",
        "SPECTRE-adaptive": "#1f77b4",
        "SPECTRE-static": "#7fb8de",
        "SPECTREv2-adaptive": "#2ca02c",
        "SPECTREv2-static": "#98df8a",
        "VLMPlan-8B": "#d62728",
        "VLMPlan-32B": "#9467bd",
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
    mo.md(r"""## 1 · Load cached per-problem FP

          Reads the precomputed raw scores and derives one rollout-FP per (method,
          problem) from `data/spectre/derived/dd2d_v3/compare_cache/`.
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
    mo.md(\
          r"""## 2 · Summary table

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

    # Sanity readout: per-method mean FP over ALL test problems (fresh 1-seed cache,
    # BCE-trained PIGINet — no stale hard-coded expectations to check against).
    _ = np  # keep numpy dep explicit for marimo
    print("mean FP (ALL):")
    for _m in METHODS:
        _v = df[df.method == _m]["fp"]
        print(f"  {_m:20s} {_v.mean():6.2f}  (n={len(_v)})")
    summary_df
    return (summary_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 3 · Mean FP per stratum (± std-dev error bars)

          Lower is better. Error bars are the across-problem std-dev, clipped at 0.
          """
             )
    return


@app.cell
def _(COLORS, METHODS, STRATA, df, np, plt):
    _groups = [str(k) for k in STRATA] + ["ALL"]
    _x = np.arange(len(_groups))
    # Derive the bar width and offset from the method count: the previous constants were
    # sized for 4 methods and left the group visibly off-centre once it grew.
    _w = 0.8 / len(METHODS)
    _off = (len(METHODS) - 1) / 2
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
            _x + (_i - _off) * _w,
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
    _ax.set_title("Mean rollout FP by stratum (DD2D test, n=100)")
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
    mo.md(\
          r"""## 4 · Survival curves

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


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""
#     ## 5 · T0 — how much does plan length explain each ranking?

#     Test **T0**, for every method with static per-skeleton scores. A DD2D plan's
#     *length* is its operator count (`2·(blockers staged)+1`). For each method we
#     measure, **per episode** (so the statistic is already per-problem-normalized —
#     a per-episode correlation is unchanged by z-scoring each problem's logits):

#     - **R²(len)** *(headline)* — the **linear** length-R² = `pearson(score, length)²`:
#       the fraction of score variance a straight-line fit on length explains.
#     - **pearson** — signed linear correlation (`+` ⇒ higher-scored plans are longer).
#     - **η²(len)** *(secondary)* — fraction of score variance explained by length
#       *group* (categorical R²); credits **any** length structure, monotone or not.
#     - **spearman** — rank correlation between score and length.

#     **Reading R² vs η².** R²(len) ≪ η²(len) means the length preference is
#     *non-monotone* — length drives the ranking, but not in a straight line, so the
#     linear R² understates it. That is exactly v1-static's regime (it front-loads
#     1- and 3-blocker plans, buries 2-blocker plans), so **read R² alongside η² and
#     the learned length curve below**, not on its own.

#     Each `*-adaptive` method has no static per-skeleton scores; its **t=0** ranking
#     is *provably identical* to its `*-static` twin (same checkpoint, empty failure
#     context → `c₀`), so its one-shot row mirrors static. Its deployed behaviour is
#     the **realized-order length ladder** further down.
#     """)
#     return


@app.cell
def _(CACHE_DIR, dd2d_compare, pd):
    from alphatamp.approaches.spectre import eda as eda_mod

    # Load test episodes once; keyed by problem_id. Reused by the T0 length fits and
    # the §7 planner inspector / §8 length-bias explorer (scene geometry + plan text).
    _spectre_test = CACHE_DIR.parents[2] / "raw" / CACHE_DIR.parent.name / "test"
    ep_by_pid = {
        int(ep.provenance.problem_id): ep
        for ep in eda_mod.load_split_episodes(_spectre_test).episodes
    }
    # Plan lengths per problem, aligned to the cached score index (skeleton_idx).
    lengths_by_pid = {
        pid: [len(s.operator_seq) for s in ep.skeleton_pool]
        for pid, ep in ep_by_pid.items()
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
    return ep_by_pid, fit_df, ladder_df, lengths_by_pid, pos_df


# @app.cell
# def _(STRATA, fit_df, pd):
#     def _m(sub, col):
#         return f"{sub[col].mean():.2f}" if len(sub) else "—"

#     def _row(label, sub):
#         r = {
#             "method": label,
#             "R²(len)": _m(sub, "r2"),
#             "pearson": _m(sub, "pearson"),
#             "η²(len)": _m(sub, "eta2"),
#             "spearman": _m(sub, "spearman"),
#         }
#         for _k in STRATA:
#             r[f"R² s{_k}"] = _m(sub[sub.stratum == _k], "r2")
#         return r

#     _static = ["astar-dist", "PIGINet", "SPECTRE-static", "SPECTREv2-static"]
#     _rows = [_row(_meth, fit_df[fit_df.method == _meth]) for _meth in _static]
#     # Each *-adaptive one-shot == its *-static twin (c₀ identity).
#     for _lab, _twin in [
#         ("SPECTRE-adaptive (t=0 ≡ static)", "SPECTRE-static"),
#         ("SPECTREv2-adaptive (t=0 ≡ static)", "SPECTREv2-static"),
#     ]:
#         _rows.append(_row(_lab, fit_df[fit_df.method == _twin]))
#     t0_table = pd.DataFrame(_rows).set_index("method")
#     print(
#         "T0: R²(len) = linear (monotone) length-R² = pearson²; "
#         "η²(len) = categorical (any-shape) length-R². "
#         "R² ≪ η² ⇒ a non-monotone length preference (see the length curve below)."
#     )
#     t0_table
#     return (t0_table,)


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""
#     ### Learned length curve

#     Mean within-episode attempt-position (0 = tried first … 1 = tried last) of each
#     plan-length tier. Reveals *which* length policy each method learned, and is what
#     a scalar R² cannot show: astar is monotone short-first; PIGINet is ~flat (it
#     discriminates *within* length, not by it); SPECTRE-static (v1) is a non-monotone
#     length lookup (front-loads 1- and 3-blocker plans, buries 2-blocker plans) — the
#     shape behind its high η² but low linear R². Compare where SPECTREv2-static lands.
#     """)
#     return


# @app.cell
# def _(COLORS, pos_df, plt):
#     _fig, _ax = plt.subplots(figsize=(6.2, 4.0))
#     for _m in ["astar-dist", "PIGINet", "SPECTRE-static", "SPECTREv2-static"]:
#         _sub = pos_df[pos_df.method == _m].sort_values("length")
#         _ax.plot(
#             _sub["length"],
#             _sub["mean_pos"],
#             marker="o",
#             label=_m,
#             color=COLORS[_m],
#         )
#     _ax.set_xticks([1, 3, 5, 7])
#     _ax.set_xticklabels(["1 (m0)", "3 (m1)", "5 (m2)", "7 (m3)"])
#     _ax.set_xlabel("plan length (blockers staged)")
#     _ax.set_ylabel("mean attempt-position (0 = tried first)")
#     _ax.set_ylim(0, 1)
#     _ax.set_title("Learned length curve (DD2D test)")
#     _ax.legend()
#     plt.tight_layout()
#     plt.gca()
#     return


@app.cell
def _(COLORS, fit_df, np, plt):
    # Linear (R²) vs categorical (η²) length-dependence: their gap = non-monotonicity.
    _methods = ["astar-dist", "PIGINet", "SPECTRE-static", "SPECTREv2-static"]
    _r2 = [fit_df[fit_df.method == m]["r2"].mean() for m in _methods]
    _eta = [fit_df[fit_df.method == m]["eta2"].mean() for m in _methods]
    _x = np.arange(len(_methods))
    _fig, _ax = plt.subplots(figsize=(6.6, 4.0))
    _ax.bar(_x - 0.2, _r2, 0.4, label="R²(len) — linear/monotone", color="#4c72b0")
    _ax.bar(
        _x + 0.2, _eta, 0.4, label="η²(len) — categorical/any-shape", color="#c44e52"
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(
        [m.replace("SPECTRE", "SP").replace("-", "-\n") for m in _methods]
    )
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("fraction of score variance from length")
    _ax.set_title(
        "Linear vs categorical length-R² per method\n(gap ⇒ non-monotone length preference)"
    )
    _ax.legend()
    _ = COLORS  # keep palette dep explicit for marimo
    plt.tight_layout()
    plt.gca()
    return


# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""### Realized-order length ladder (v1 vs v2 adaptive)

#           Over the sequence each adaptive method *actually attempts* (until first
#           success), does it climb to longer plans as it fails? Positive
#           `spearman(position, length)` ⇒ a length-escalation ladder. (Reads the cached
#           `order` trace.)
#           """)
#     return


@app.cell
def _(ladder_df, pd):
    _adaptive = ["SPECTRE-adaptive", "SPECTREv2-adaptive"]

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
        for _m in _adaptive:
            _row[_m] = _c(ladder_df[ladder_df.method == _m], _col)
        _rows.append(_row)
    adaptive_ladder_table = pd.DataFrame(_rows).set_index("metric")
    adaptive_ladder_table
    return (adaptive_ladder_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""### Takeaway (T0)

          - **Read R² *with* η².** A high η² but low linear R² (v1-static) means the
            ranking *is* a length lookup, just a **non-monotone** one — the linear R²
            alone would hide it. The learned-length-curve above is the honest picture.
          - **SPECTRE v1-static is, mechanically, a plan-length lookup** (η² near 1): it
            ranks by *how many* blockers a plan stages, blind to *which* same-size subset
            is correct. Its s3 win over astar is a length re-ordering, not subset knowledge.
          - **SPECTREv2-static should carry more within-length signal** (lower η², a
            flatter length curve) — the whole point of the geometry + tag representation.
            Check its R²/η² gap and its curve against v1's.
          - **PIGINet is the low-level comparator**: it ranks *within* a length tier using
            image geometry (high within-length variance), the structural opposite of a
            pure length prior.
          - **Each `*-adaptive` t=0 ≡ its `*-static` twin**; the adaptive lever is the
            realized-order re-ranking (length ladder below).
          """
             )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 6 · T1 — length-only-context intervention (does c_t use identity?)

          Test **T1** (length-only variant) from
          `docs/spectre_piginet_hypotheses_and_tests_v2.md`. Rerun the SPECTRE-adaptive
          rollout, but replace every failed skeleton in the failure context Ψ sees with
          a **random *other* pool skeleton of the same plan length** — correct length,
          random object identity, consistent `s_L` (it is a real pooled skeleton). The
          selection still masks the *real* attempted skeletons; only the context is
          scrambled (`eda.spectre_evaluate_length_only_context`, averaged over 3 seeds ×
          3 surrogate draws in `spectre_lenctx/`).

          **If mean rollout-FP is unchanged ⇒ Ψ ignores which objects failed and uses
          only size/length (H2).** A drop ⇒ identity carries signal → escalate to the
          full T1 suite. (Regression guard: `scramble=False` reproduces SPECTRE-adaptive
          bitwise.)
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


# @app.cell
# def _(plt, t1_df):
#     _cmap = {0: "#4c72b0", 1: "#55a868", 2: "#c44e52", 3: "#8172b3"}
#     _fig, _ax = plt.subplots(figsize=(5.0, 5.0))
#     for _k in sorted(t1_df.stratum.unique()):
#         _s = t1_df[t1_df.stratum == _k]
#         _ax.scatter(
#             _s["fp_adaptive"],
#             _s["fp_lenctx"],
#             s=18,
#             alpha=0.7,
#             color=_cmap[_k],
#             label=f"s{_k}",
#             edgecolors="none",
#         )
#     _lim = max(t1_df["fp_adaptive"].max(), t1_df["fp_lenctx"].max()) * 1.05
#     _ax.plot([0, _lim], [0, _lim], "k--", lw=0.8, alpha=0.6)
#     _ax.set_xlabel("SPECTRE-adaptive FP (real context)")
#     _ax.set_ylabel("length-only-context FP")
#     _ax.set_title(
#         "T1: identity-scrambled vs real context\n(points on y=x ⇒ identity unused)"
#     )
#     _ax.legend(title="stratum")
#     plt.tight_layout()
#     plt.gca()
#     return


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
    mo.md(\
          r"""### Takeaway (T1)

          Scrambling the *identity* of every failed skeleton to a random same-length
          plan leaves SPECTRE-adaptive's rollout-FP **unchanged** (Δ ≈ 0, CI includes 0
          in every stratum). The context embedding `c_t` does shift a hair under
          scrambling (‖Δc‖ ~ 5e-3 on the d=64 vector), but never enough to change a
          selection — so the context module Ψ carries only a *whisper* of identity and
          is, functionally, **size/length-only**. This confirms **H2**: the adaptive
          gains (concentrated on s0/s1) are length-regime escalation, not failed-subset
          identification.

          Combined with T0 (SPECTRE-static ≈ a pure length ranking, η²≈1.0), SPECTRE on
          DD2D uses **length and nothing identity-specific** — neither in the static
          ranker nor in the adaptive context. Breaking the same-length symmetry (telling
          *which* same-size subset is right) needs diagnostic evidence the abstract
          representation does not carry — the v2.1 typed-evidence motivation.
          """
             )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 7 · Planner inspector — scene + ordered plans

          Step through test problems with **◀ / ▶** (or the dropdown). Three panels:

          - **Scene** — the initial DD2D drawer, drawn from the episode's stored geometry.
            The **red** item is the retrieval target; blue items are concave; the dark frame
            is the wall band; the dashed box is the buffer. Labels are the item index
            (`item_5` → `5`), so they match the `stage {…}` sets in the plan table.
          - **Every method on this problem** — rollout FP and first-feasible rank for all six
            methods at once, so they can be compared without toggling. **Independent of the
            method dropdown.**
          - **Ordered plans** for the *selected* method — top-ranked → bottom-ranked, 10 per
            page, sortable by any column.

          For a `*-adaptive` method the toggle switches between its **realized attempt order**
          and its **t=0 score order**; static methods always rank by score. In realized order
          the table also carries the *static twin's* rank and score, so `Δrank` shows exactly
          which plans adaptivity promoted (`+`) or demoted (`−`), and `demoted@t` names the
          failure whose proof killed a candidate outright.

          > An adaptive ranker re-scores the pool after **every** failure, so no candidate has
          > a single score. The cache stores the whole per-step matrix; `ad.score` reports the
          > step each candidate was *picked* on — the opinion the rollout acted on.
          """
             )
    return


@app.cell
def _(df, mo):
    # The selected problem lives in mo.state, not in the dropdown, so the ◀/▶ buttons
    # and the dropdown stay in sync: whichever control fires, every consumer reads
    # get_pid(). Split across cells on purpose — marimo does not re-run the cell that
    # *set* the state, so the buttons and the dropdown must not share one.
    INSPECT_PIDS = sorted(int(p) for p in df["problem_id"].unique())
    get_pid, set_pid = mo.state(INSPECT_PIDS[0])
    return INSPECT_PIDS, get_pid, set_pid


@app.cell
def _(INSPECT_PIDS, mo, set_pid):
    def _step_pid(delta: int):
        # Functional setter: reads the current value at click time, so this cell never
        # needs to re-run and can never close over a stale problem id.
        def _move(pid):
            i = INSPECT_PIDS.index(pid) if pid in INSPECT_PIDS else 0
            return INSPECT_PIDS[min(max(i + delta, 0), len(INSPECT_PIDS) - 1)]

        return _move

    inspect_prev = mo.ui.button(
        label="◀ prev",
        on_change=lambda _: set_pid(_step_pid(-1)),
        keyboard_shortcut="[",
    )
    inspect_next = mo.ui.button(
        label="next ▶",
        on_change=lambda _: set_pid(_step_pid(+1)),
        keyboard_shortcut="]",
    )
    return inspect_prev, inspect_next


@app.cell
def _(INSPECT_PIDS, get_pid, mo, set_pid):
    inspect_pid = mo.ui.dropdown(
        options={str(p): p for p in INSPECT_PIDS},
        value=str(get_pid()),
        on_change=set_pid,
        label="test problem",
    )
    return (inspect_pid,)


@app.cell
def _(METHODS, mo):
    # These two drive the plan table only — the scene and the per-method overview below
    # are deliberately method-independent.
    inspect_model = mo.ui.dropdown(options=METHODS, value=METHODS[4], label="method")
    inspect_realized = mo.ui.switch(
        value=True, label="adaptive: realized attempt order (off = t=0 score order)"
    )
    return inspect_model, inspect_realized


@app.cell
def _(inspect_model, inspect_next, inspect_pid, inspect_prev, inspect_realized, mo):
    mo.hstack(
        [inspect_pid, inspect_prev, inspect_next, inspect_model, inspect_realized],
        justify="start",
        gap=1.0,
        align="center",
    )
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, np):
    # method -> (kind, static-scores dir, adaptive dir | None). Both modes of a SPECTRE
    # family share one checkpoint, so an adaptive method's "static twin" scores are its
    # own t=0 (c₀) logits.
    INSPECT_SPEC = {
        "astar-dist": ("static", "astar", None),
        "PIGINet": ("static", "piginet", None),
        "SPECTRE-static": ("static", "spectre_static/seed_0", None),
        "SPECTRE-adaptive": ("adaptive", "spectre_static/seed_0", "spectre_adaptive"),
        "SPECTREv2-static": ("static", "spectre2_static/seed_0", None),
        "SPECTREv2-adaptive": (
            "adaptive",
            "spectre2_static/seed_0",
            "spectre2_adaptive",
        ),
        # "sequence": generates its own attempt order rather than ranking the pool, so it
        # has no per-pool score row. Its table is the realized attempt list, including the
        # off-pool attempts that have no pool index at all.
        **{
            name: ("sequence", None, subdir)
            for name, subdir in dd2d_compare.SEQUENCE_METHODS.items()
        },
    }

    def insp_load(method, pid):
        """``(static scores | None, AdaptiveTrace | None)`` for one method+problem."""
        _kind, sdir, adir = INSPECT_SPEC[method]
        rec = dd2d_compare.load_static_scores(CACHE_DIR, sdir, pid) if sdir else None
        scores = np.asarray(rec["scores"], float) if rec else None
        trace = dd2d_compare.load_adaptive_trace(CACHE_DIR, adir, pid) if adir else None
        return scores, trace

    def insp_sequence(method, pid):
        """The raw attempt list of a sequence method (VLMPlan), or ``None``.

        Read straight from the record rather than via ``AdaptiveTrace``: an off-pool
        attempt has no pool index, so it cannot be represented in a pool-indexed order.
        """
        _kind, _sdir, adir = INSPECT_SPEC[method]
        if _kind != "sequence" or adir is None:
            return None
        path = CACHE_DIR / adir / "seed_0" / f"{int(pid)}.json"
        if not path.is_file():
            return None
        import json as _json

        return _json.loads(path.read_text())

    def insp_effective(trace, step):
        """The row the step-``step`` pick was actually made from: raw logits, with
        unavailable (already-attempted) entries at ``-inf`` and provably-dead ones
        pushed back by the demotion offset."""
        row = np.asarray(trace.step_scores[step], float)
        row = np.where(np.isnan(row), -np.inf, row)
        dead = (trace.step_dead or [[]] * len(trace.step_scores))[step]
        if dead:
            row[list(dead)] -= 1e6
        return row

    def insp_order(method, pid, realized):
        """Display order of pool indices for a method, plus its trace when one applies.

        Realized adaptive view = the sequence actually attempted, then the never-tried
        tail ranked by the **final-step** row (the model's most-informed opinion, after
        every observed failure). Otherwise a plain score ranking.

        A **sequence** method returns only the in-pool part of its realized order — its
        off-pool attempts have no pool index, so they cannot appear in a pool-indexed list
        and are shown by the plan table instead (which reads the raw record).
        """
        if INSPECT_SPEC[method][0] == "sequence":
            rec = insp_sequence(method, pid)
            if rec is None:
                return [], None
            return [int(i) for i in rec["order"] if int(i) >= 0], None
        scores, trace = insp_load(method, pid)
        use_trace = trace is not None and realized
        if use_trace:
            seen = set(trace.order)
            if trace.step_scores:
                tail_row = insp_effective(trace, len(trace.step_scores) - 1)
            elif scores is not None:
                tail_row = scores  # legacy cache without per-step scores
            else:
                return list(trace.order), trace
            tail = [
                int(i)
                for i in np.argsort(-tail_row, kind="stable")
                if int(i) not in seen
            ]
            return list(trace.order) + tail, trace
        if scores is None:
            return [], None
        return [int(i) for i in np.argsort(-scores, kind="stable")], None

    return INSPECT_SPEC, insp_effective, insp_load, insp_order, insp_sequence


@app.cell
def _(
    INSPECT_PIDS,
    METHODS,
    dd2d_compare,
    df,
    ep_by_pid,
    get_pid,
    insp_order,
    inspect_model,
    inspect_realized,
    mo,
    pd,
    plt,
):
    from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (
        reconstruct_scene as _reconstruct_scene,)
    from alphatamp.approaches.spectre.envs.dd2d.spectre_render import (
        scene_figure as _scene_fig,)

    _ = plt  # keep the pyplot dep explicit for marimo

    _pid = get_pid()
    _ep = ep_by_pid[_pid]
    _feas = [o.outcome == "success" for o in _ep.outcomes]
    _k = len(_ep.skeleton_pool)
    _fmin = min(
        (len(s.operator_seq) for s, ok in zip(_ep.skeleton_pool, _feas) if ok),
        default=None,
    )

    # Every method's outcome on THIS problem — the method dropdown does not touch it.
    _fp_here = {
        r["method"]: r["fp"] for r in df[df.problem_id == _pid].to_dict("records")
    }
    _rows = []
    for _m in METHODS:
        _order, _tr = insp_order(_m, _pid, inspect_realized.value)
        _ff = next((r for r, i in enumerate(_order) if _feas[i]), None)
        _rows.append(
            {
                "": "▶" if _m == inspect_model.value else "",
                "method": _m,
                "FP": _fp_here.get(_m),
                "1st-feasible rank": _ff,
                "attempts": len(_tr.order) if _tr is not None else None,
            }
        )
    inspect_overview = mo.ui.table(
        pd.DataFrame(_rows),
        selection=None,
        pagination=False,
        show_column_summaries=False,
    )

    try:
        _scene = mo.as_html(_scene_fig(_reconstruct_scene(_ep.scene_geometry)))
    except Exception as _e:  # noqa: BLE001 — geometry render is best-effort in the UI
        _scene = mo.md(f"*(scene render unavailable: {_e})*")

    mo.vstack(
        [
            mo.md(
                f"### problem **{_pid}**"
                f" &nbsp;·&nbsp; stratum **{dd2d_compare.stratum_of(_pid)}**"
                f" &nbsp;·&nbsp; pool **{_k}** &nbsp;·&nbsp; feasible **{sum(_feas)}/{_k}**"
                f" &nbsp;·&nbsp; shortest feasible plan **{_fmin}** ops"
                f" &nbsp;·&nbsp; [{INSPECT_PIDS.index(_pid) + 1}/{len(INSPECT_PIDS)}]"
            ),
            mo.hstack(
                [_scene, inspect_overview], widths=[1.35, 1], gap=1, align="start"
            ),
            mo.md(
                "<sub>`FP` is the cached headline metric; for a static method it differs "
                "from `1st-feasible rank` only by `rollout_fp`'s half-credit on exact "
                "score ties. `attempts` is blank for methods that never run a "
                "rollout.</sub>"
            ),
        ]
    )
    return (inspect_overview,)


@app.cell
def _(
    INSPECT_SPEC,
    ep_by_pid,
    get_pid,
    insp_load,
    insp_order,
    insp_sequence,
    inspect_model,
    inspect_realized,
    lengths_by_pid,
    mo,
    np,
    pd,
):
    import math as _math

    def _plan_label(skel):
        staged = [
            op.parameters[0].name.split("_")[-1]
            for op in skel.operator_seq
            if op.name == "place-buffer"
        ]
        tgt = next(
            (
                op.parameters[0].name.split("_")[-1]
                for op in skel.operator_seq
                if op.name == "retrieve"
            ),
            "?",
        )
        head = "stage {" + ", ".join(staged) + "} → " if staged else ""
        return f"{head}retrieve {tgt}"

    _method = inspect_model.value
    _pid = get_pid()
    _ep = ep_by_pid[_pid]
    _lens = lengths_by_pid[_pid]
    _feas = [o.outcome == "success" for o in _ep.outcomes]
    _scores, _trace = insp_load(_method, _pid)
    _order, _tr = insp_order(_method, _pid, inspect_realized.value)

    # "Adaptive view" needs a realized trace *with* per-step scores; a legacy cache
    # (order only) falls back to the plain score table.
    _adaptive = _tr is not None and bool(_tr.step_scores)
    _st_rank = (
        {int(i): r for r, i in enumerate(np.argsort(-_scores, kind="stable"))}
        if _scores is not None
        else {}
    )
    # An adaptive method re-scores the whole pool after every failure, so a candidate
    # has no single score — only a per-step one. The honest per-row number is the score
    # at the step it was *picked* (that is the opinion it acted on); never-attempted
    # candidates get the final step, the most-informed context. Reading the final row
    # for an attempted candidate would be blank by construction: the model masks its own
    # failure context, so those entries come back NaN.
    _first_dead: dict[int, int] = {}
    if _adaptive and _tr.step_dead:
        for _t, _dead in enumerate(_tr.step_dead):
            for _i in _dead:
                _first_dead.setdefault(int(_i), _t)

    # A sequence method (VLMPlan) has no pool-indexed ranking at all: it generated its own
    # attempts, and the off-pool ones have no pool index. So its table is the raw realized
    # attempt list off the record, with the label the scorer actually used.
    _seq = insp_sequence(_method, _pid)
    if _seq is not None:
        _rows = []
        for _rank, _att in enumerate(_seq["attempts"]):
            _members = ", ".join(str(m).split("_")[-1] for m in _att["members"])
            _rows.append(
                {
                    "attempt": _rank + 1,
                    "plan": ("stage {" + _members + "} → " if _members else "")
                    + f"retrieve {str(_seq.get('target', 'target')).split('_')[-1]}",
                    "len": 2 * len(_att["members"]) + 1,
                    "feasible": "✓" if _att["label"] == "success" else "",
                    "source": _att["source"],
                    "in pool": "✓" if _att["in_pool"] else "off-pool",
                    "pool idx": _att["pool_idx"],
                    "round": _att["round"],
                }
            )
        _plan_df = pd.DataFrame(_rows)
        _ff = next(
            (r for r, a in enumerate(_seq["attempts"]) if a["label"] == "success"), None
        )
    else:
        _rows = []
        for _rank, _i in enumerate(_order):
            _row = {"rank": _rank}
            if _adaptive:
                _row["attempt"] = _rank + 1 if _rank < len(_tr.order) else None
            _row["plan"] = _plan_label(_ep.skeleton_pool[_i])
            _row["len"] = int(_lens[_i])
            _row["feasible"] = "✓" if _feas[_i] else ""
            if _adaptive:
                _at = _rank if _rank < len(_tr.order) else len(_tr.step_scores) - 1
                _v = _tr.step_scores[_at][_i]
                _row["ad.score"] = None if _math.isnan(_v) else round(float(_v), 3)
                _row["st.rank"] = _st_rank.get(_i)
                _row["st.score"] = (
                    None if _scores is None else round(float(_scores[_i]), 3)
                )
                _row["Δrank"] = (
                    None if _i not in _st_rank else int(_st_rank[_i]) - _rank
                )
                _row["demoted@"] = _first_dead.get(_i)
            elif _scores is not None:
                _row["score"] = round(float(_scores[_i]), 3)
            _rows.append(_row)
        _plan_df = pd.DataFrame(_rows)
        _ff = next((r for r, i in enumerate(_order) if _feas[i]), None)

    def _style_cell(row_id, name, value):
        style = {}
        if row_id == _ff:
            style["background-color"] = "#e6f4ea"  # the FP boundary: first feasible
        if name == "Δrank" and isinstance(value, (int, float)):
            style["color"] = (
                "#2e7d32" if value > 0 else ("#c62828" if value < 0 else "#888")
            )
        return style

    if not len(_plan_df):
        inspect_plan_table = mo.md(
            f"*(no cached scores for **{_method}** on problem {_pid})*"
        )
    else:
        inspect_plan_table = mo.ui.table(
            _plan_df,
            pagination=True,
            page_size=10,
            selection=None,
            show_column_summaries=False,
            freeze_columns_left=["rank"],
            style_cell=_style_cell,
        )

    _kind = INSPECT_SPEC[_method][0]
    if _seq is not None:
        _legend = (
            f"**{_method}** · the sequence it actually **generated**, in order. Unlike "
            "every other method this is not a re-ranking of the 200-candidate pool: "
            "`source=vlm` rows are the model's own proposals, `source=fill` rows are the "
            "published-order fallback used after it ran dry, and `off-pool` rows are plans "
            "the pool does not contain (labelled by a live refiner). Off-pool attempts "
            "**do** count toward FP — see the note at the top."
        )
    elif _adaptive:
        _legend = (
            f"**{_method}** · realized attempt order over {len(_tr.step_scores)} steps. "
            "`attempt` = the step the rollout actually ran it — it stops at the first "
            f"success, so those are exactly ranks 0…{_ff}; blank rows were **never "
            "tried** and are ordered by the final-step opinion. "
            "`ad.score` = the score at the step the candidate was **picked** (an adaptive "
            "ranker re-scores after every failure, so there is no single score); "
            "never-attempted rows use the final, most-informed step. "
            "`st.*` is the same checkpoint's **t=0** (`c₀`) ranking, so "
            "`Δrank = st.rank − rank` reads directly as **+ promoted** by adaptivity, "
            "**− demoted**. `demoted@t` = proof-demotion proved it dead from the "
            "attempt-`t` failure (v2 only — v1 has no proof-demotion)."
        )
    elif _kind == "adaptive":
        _legend = (
            f"**{_method}** · **t=0 score order** — identical to its `*-static` twin "
            "(same checkpoint, empty failure context `c₀`). Flip the toggle for the "
            "realized rollout."
        )
    else:
        _legend = (
            f"**{_method}** · one-shot score ranking. Static methods never run a "
            "rollout, so there is no attempt order to show."
        )
    mo.vstack([mo.md(_legend), inspect_plan_table])
    return (inspect_plan_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 8 · Length-bias explorer — logit vs plan length

        Pick a logit-bearing method. We pool every test candidate as
        (**within-problem z-scored logit**, plan length) — z-scoring per problem removes
        the cross-problem logit-scale confound so the correlation reflects the genuine
        *within-problem* length preference. The scatter shows the OLS best-fit line, the
        **pooled Pearson r** (and R² = r²), and the **mean per-episode r** (the T0 stat).
        astar is excluded (its "scores" are enumeration index, not logits); each
        `*-adaptive` method's t=0 logits are identical to its `*-static` twin.
        """)
    return


@app.cell
def _(mo):
    lb_model = mo.ui.dropdown(
        options=["PIGINet", "SPECTRE-static", "SPECTREv2-static"],
        value="SPECTREv2-static",
        label="method (logit-bearing)",
    )
    lb_model
    return (lb_model,)


@app.cell
def _(CACHE_DIR, COLORS, fit_df, lb_model, lengths_by_pid, np, plt):
    import json as _json

    _DIR = {
        "PIGINet": "piginet",
        "SPECTRE-static": "spectre_static/seed_0",
        "SPECTREv2-static": "spectre2_static/seed_0",
    }
    _m = lb_model.value
    _L, _Z = [], []
    for _p in sorted((CACHE_DIR / _DIR[_m]).glob("*.json")):
        _rec = _json.loads(_p.read_text())
        _s = np.asarray(_rec["scores"], float)
        if _s.std() == 0.0:
            continue
        _Z.append((_s - _s.mean()) / _s.std())
        _L.append(np.asarray(lengths_by_pid[int(_rec["problem_id"])], float))
    _L = np.concatenate(_L)
    _Z = np.concatenate(_Z)
    _r = float(np.corrcoef(_L, _Z)[0, 1])
    _mean_ep_r = float(fit_df[fit_df.method == _m]["pearson"].mean())
    _slope, _icpt = np.polyfit(_L, _Z, 1)
    _xs = np.array([_L.min(), _L.max()])

    _jit = (np.random.default_rng(0).random(_L.size) - 0.5) * 0.4
    _fig, _ax = plt.subplots(figsize=(6.6, 4.3))
    _ax.scatter(_L + _jit, _Z, s=8, alpha=0.22, color=COLORS[_m], edgecolors="none")
    _ax.plot(
        _xs, _slope * _xs + _icpt, "k-", lw=1.6, label=f"OLS (slope {_slope:+.3f})"
    )
    _ax.axhline(0, color="0.7", lw=0.6)
    _ax.set_xticks(sorted(np.unique(_L).astype(int).tolist()))
    _ax.set_xlabel("plan length (operator count)")
    _ax.set_ylabel("within-problem z-scored logit")
    _ax.set_title(
        f"{_m}: logit vs plan length (DD2D test)\n"
        f"pooled Pearson r = {_r:+.3f}  (R² = {_r ** 2:.3f})   "
        f"mean per-episode r = {_mean_ep_r:+.3f}"
    )
    _ax.legend(loc="best")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 9 · VLMPlan diagnostics — did it work, and is the number informative?

          Only rendered when the VLMPlan cache exists. Two questions the headline FP alone
          cannot answer:

          - **Did the model find the plan, or did the fallback?** `first_success_source`
            is `vlm` when the model's own proposal succeeded and `fill` when the
            published-order fallback found it after the model ran dry. A row whose FP looks
            respectable but whose success came from `fill` is really an astar-dist number.
          - **Is it just mimicking the planner?** The pre-registered **trivial-mimicry
            null**: `spearman_vs_published` near 1 means the model reproduced the planner's
            ascending-size enumeration, in which case its FP tells us about size ordering,
            not geometric reasoning — regardless of where it lands.

          Plus the generation-quality rates (`off-pool`, rounds used) that say where
          zero-shot proposal capacity ran out.
          """
             )
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, mo, pd):
    # One block per model arm. An arm with no cache is skipped rather than erroring, so
    # the notebook loads whether zero, one or both arms have been run.
    _frames, _blocks = {}, []
    for _arm, _subdir in dd2d_compare.SEQUENCE_METHODS.items():
        _rows = dd2d_compare.load_vlmplan_diagnostics(CACHE_DIR, _subdir)
        if not _rows:
            _blocks.append(
                mo.md(
                    f"*(**{_arm}** — no cache under `compare_cache/{_subdir}/seed_0/`)*"
                )
            )
            continue
        _v = pd.DataFrame(_rows)
        # Records written before the generation-stats fields existed carry None; treat
        # those as 0 so the aggregation works on a mixed cache.
        for _col in ("n_truncated", "n_proposed", "n_offpool", "n_rounds_used"):
            _v[_col] = pd.to_numeric(_v.get(_col), errors="coerce").fillna(0)
        _frames[_arm] = _v
        _n = len(_v)
        _model = next(
            (m.get("model_name") for m in _v.get("model", []) if isinstance(m, dict)),
            "unknown model",
        )
        _trunc = int(_v["n_truncated"].sum())
        _by_stratum = (
            _v.groupby("stratum")
            .agg(
                n=("fp", "size"),
                mean_fp=("fp", "mean"),
                mean_offpool=("n_offpool", "mean"),
                mean_rounds=("n_rounds_used", "mean"),
                proposed=("n_proposed", "mean"),
                found_by_vlm=("first_success_source", lambda s: (s == "vlm").sum()),
                found_by_fill=("first_success_source", lambda s: (s == "fill").sum()),
                censored=("censored", "sum"),
                truncated=("n_truncated", "sum"),
                mimicry=("spearman_vs_published", "mean"),
            )
            .reset_index()
        )
        # Truncation is a config fault, not a model result: a truncated round loses its
        # last plan block, so a nonzero count means the arm under-reports the model.
        _warn = (
            ""
            if _trunc == 0
            else (
                f" · ⚠️ **{_trunc} truncated rounds** — completions hit the output cap, "
                "so those rounds lost their last plan. Raise `model.decode.max_tokens` "
                "and the served context, then re-run this arm."
            )
        )
        _blocks += [
            mo.md(
                f"#### {_arm}  <sub>`{_model}`</sub>\n\n"
                f"**{_n} problems** · found by the model itself "
                f"**{int((_v.first_success_source == 'vlm').sum())}/{_n}**, "
                f"by the published-order fill "
                f"**{int((_v.first_success_source == 'fill').sum())}/{_n}**, "
                f"censored **{int(_v.censored.sum())}** · "
                f"mean proposed **{_v.n_proposed.mean():.0f}** · "
                f"mean off-pool attempts **{_v.n_offpool.mean():.1f}** · "
                f"mimicry ρ **{_v.spearman_vs_published.mean():.2f}**{_warn}"
            ),
            mo.ui.table(
                _by_stratum,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
        ]
    vlm_frames = _frames
    mo.vstack(_blocks)
    return (vlm_frames,)


@app.cell
def _(df, mo):
    _out = mo.notebook_dir() / "dd2d_method_comparison.csv"
    df.to_csv(_out, index=False)
    print(f"wrote {_out}  ({len(df)} rows)")
    return


if __name__ == "__main__":
    app.run()
