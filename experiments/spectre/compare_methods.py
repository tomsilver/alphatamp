import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Method comparison — SPECTRE vs the low-level and zero-shot baselines

          Plan-feasibility methods on the held-out **test split**, by **rollout FP**
          (failed refinement attempts before the first success; lower is better). **Pick
          the environment below** — strata, method list, caveats and the §5 scene all come
          from its `compare_envs.py` entry.
          """)
    return


@app.cell
def _(mo):
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots  # noqa: F401  (registers the 'science' style)
    import seaborn as sns

    from alphatamp.approaches.spectre import compare, compare_envs

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

    # The environment picker. Everything below keys off it; changing it re-renders the
    # whole notebook against another collection.
    #
    # `SPECTRE_COMPARE_ENV` sets the initial selection. In the browser it is just a
    # default, but in marimo's script mode a UI element reads back its default, so this
    # is the only way to exercise a non-first environment headlessly -- which is how the
    # notebook gets smoke-tested for every registry entry rather than only the one that
    # happens to sort first.
    _default_env = os.environ.get("SPECTRE_COMPARE_ENV", next(iter(compare_envs.ENVS)))
    env_picker = mo.ui.dropdown(
        options=list(compare_envs.ENVS),
        value=_default_env,
        label="environment",
    )
    return REPO, compare, compare_envs, env_picker


@app.cell(hide_code=True)
def _(env_picker, mo):
    mo.md(f"### Environment\n\n{env_picker}")
    return


@app.cell
def _(REPO, compare_envs, env_picker, mo, np, pd, plt, sns, compare):
    ENV = compare_envs.get(env_picker.value)
    ENV_VARIANT = ENV.env_variant
    LEGACY_VARIANT = ENV.legacy_variant or ENV.env_variant
    LEGACY_ONLY = list(ENV.legacy_only)
    DERIVED = REPO / "data" / "spectre" / "derived"
    CACHE_DIR = DERIVED / ENV_VARIANT / "compare_cache"
    LEGACY_CACHE = DERIVED / LEGACY_VARIANT / "compare_cache"

    # A method's records live in the primary cache, except grafted (`legacy_only`) methods,
    # which are in the legacy cache. Used by the inspector (§5) and the VLMPlan diag (§6).
    def cache_for(method):
        return LEGACY_CACHE if method in LEGACY_ONLY else CACHE_DIR

    COLORS = {
        "astar-dist": "#7f7f7f",
        "PIGINet": "#ff7f0e",
        "SPECTRE-adaptive": "#d62728",
        "SPECTRE-static": "#ff9896",
        "VLMPlan-32B": "#9467bd",
        "VLMPlan-GPT5.6": "#8c564b",
    }
    STRATA = sorted(ENV.stratum_labels)
    SLAB = ENV.stratum_labels
    # Read off the cache rather than hardcoded: the DD2D notebook said "n=100" in a title
    # even when a subset was loaded, which is the sort of caption that quietly becomes
    # wrong.
    N_PROBLEMS = len({p.stem for p in (CACHE_DIR / "astar").glob("*.json")}) or 0
    print(f"env={ENV.key}\nprimary: {CACHE_DIR}\nlegacy:  {LEGACY_CACHE}")
    return (
        CACHE_DIR,
        COLORS,
        ENV,
        ENV_VARIANT,
        LEGACY_CACHE,
        LEGACY_ONLY,
        LEGACY_VARIANT,
        N_PROBLEMS,
        SLAB,
        STRATA,
        cache_for,
        compare,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load

          Reads precomputed per-problem FP from the primary + legacy caches (grafting the
          methods without a native row). Two frames: **`df_seeds`** (per method/seed/problem
          — §1/§2's `±` is across seeds) and **`df`** (seed-collapsed; §3/§5/the CSV).
          """)
    return


@app.cell
def _(CACHE_DIR, ENV_VARIANT, LEGACY_CACHE, LEGACY_ONLY, LEGACY_VARIANT, compare, pd):
    _primary = compare.load_fp_records_per_seed(CACHE_DIR)
    _legacy = compare.load_fp_records_per_seed(LEGACY_CACHE)
    merged = compare.merge_collections(
        _primary,
        _legacy,
        LEGACY_ONLY,
        primary_name=ENV_VARIANT,
        legacy_name=LEGACY_VARIANT,
    )
    df_seeds = pd.DataFrame(merged)
    # The seed-collapsed view, for every per-problem consumer. Kept as a separate frame
    # rather than by re-loading: one read, one merge, and the two frames provably agree.
    # `select_seed` is deliberately NOT called any more -- it kept one seed per method,
    # which is what made every `±` in this notebook an across-*problem* number.
    df = (
        df_seeds.groupby(
            ["method", "collection", "problem_id", "stratum"], as_index=False
        )["fp"]
        .mean()
        .astype({"problem_id": int, "stratum": int})
    )
    COLLECTION = dict(zip(df["method"], df["collection"]))
    METHODS = [m for m in compare.METHOD_ORDER if m in set(df["method"])]

    print("method                seeds  collection   n")
    for _m in METHODS:
        _s = df_seeds[df_seeds.method == _m]
        _n_seeds = _s["seed"].nunique(dropna=True)
        print(
            f"  {_m:<20s} {('-' if _n_seeds == 0 else _n_seeds):>4}   "
            f"{COLLECTION[_m]:<10s}  {len(df[df.method == _m]):>3d}"
        )
    # A method with fewer seeds than the rest is reported, never left to be inferred from
    # a missing `±`: it changes what its row means, not just how precisely it is known.
    _counts = {
        m: df_seeds[df_seeds.method == m]["seed"].nunique(dropna=True) for m in METHODS
    }
    _learned = {m: c for m, c in _counts.items() if c}
    if _learned and len(set(_learned.values())) > 1:
        print(f"\n!! uneven seed counts across learned methods: {_learned}")
    return COLLECTION, METHODS, df, df_seeds, merged


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1 · Summary table — mean FP per stratum

          `±` = across-seed spread of the per-stratum mean; `seeds` = how many went into it
          (`-` = a single deterministic run). A one-seed row shows a bare mean, never
          `± 0.00`. Stratum meaning + per-env caveats are printed under the table.
          """)
    return


@app.cell
def _(COLLECTION, METHODS, compare, df, df_seeds, merged, mo, pd):
    # `build_table` is the shared implementation (also behind `spectre_v3_table.py`), so
    # this table and the CLI reporter cannot drift apart. It takes the PER-SEED records
    # -- feeding it the collapsed frame would silently give the across-problem spread of
    # a seed-mean, which is the bug this section previously had.
    summary_header, summary_rows, summary_tidy = compare.build_table(merged)
    summary_df = pd.DataFrame(summary_tidy)

    print("mean FP (ALL), lower is better:")
    for _m in sorted(METHODS, key=lambda m: df[df.method == m]["fp"].mean()):
        _e = next(
            t for t in summary_tidy if t["method"] == _m and t["stratum"] == "ALL"
        )
        _sd = (
            ""
            if pd.isna(_e["std_fp_across_seeds"])
            else f" ± {_e['std_fp_across_seeds']:.2f}"
        )
        # Match the table's convention: a deterministic run reads `-`, not "1 seed",
        # because "1 seed" implies a sample of a distribution that was never sampled.
        _seeds = df_seeds[df_seeds.method == _m]["seed"].nunique(dropna=True)
        print(
            f"  {_m:<20s} {_e['mean_fp']:6.2f}{_sd:<9s} "
            f"[{COLLECTION[_m]}, {_seeds if _seeds else '-'} seed(s)]"
        )
    mo.md(
        compare.render_markdown(summary_header, summary_rows)
        + "\n\n"
        + ENV.stratum_meaning
        # + (
        #     "\n\n**Read before quoting a number:**\n"
        #     + "\n".join(f"- {c}" for c in ENV.caveats)
        #     if ENV.caveats
        #     else ""
        # )
    )
    return summary_df, summary_tidy


# --- commented out 2026-07-27 (notebook trim); uncomment both cells to restore the
# --- across-seed spread table. It reads `--` at one cached seed and fills in once
# --- `precompute_dd2d_cache.py --seeds 0 1 2 ...` has run.
# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""### Across-seed spread
#
#           The same table computed over **every cached seed**, where `±` is the
#           spread of the per-stratum mean *across seeds* rather than across problems.
#           With one seed cached it reads `--`; it fills in automatically once
#           `precompute_dd2d_cache.py
#           --seeds 0 1 2 ...` has run. Kept visible so the distinction between the two
#           spreads is never implicit.
#           """)
#     return


# @app.cell
# def _(CACHE_DIR, compare, mo):
#     _all_seeds = compare.load_fp_records_per_seed(CACHE_DIR)
#     _header, _rows, _tidy = compare.build_table(_all_seeds)
#     mo.md(
#         f"*{CACHE_DIR.parent.name} only (native rows); "
#         f"± = across seeds of the per-stratum mean.*\n\n"
#         + compare.render_markdown(_header, _rows)
#     )
#     return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2 · Mean FP per stratum (± across-seed std)

          §1 as a bar chart. Error bars = across-seed std; a bar with **no cap** is a
          single deterministic run (astar, VLMPlan). Hatched = grafted from the legacy
          collection.
          """)
    return


@app.cell
def _(
    COLLECTION,
    COLORS,
    ENV,
    ENV_VARIANT,
    METHODS,
    N_PROBLEMS,
    SLAB,
    STRATA,
    np,
    plt,
    summary_tidy,
):
    # One source of truth with §1: read the means and across-seed stds straight out of
    # `build_table`'s tidy output rather than recomputing from a frame, which is how a
    # chart and the table above it end up disagreeing.
    _by = {(t["method"], t["stratum"]): t for t in summary_tidy}
    _groups = [SLAB.get(k, str(k)) for k in STRATA] + ["ALL"]
    _x = np.arange(len(_groups))
    # Bar width and offset derive from the method count so the group stays centred as
    # methods are added or a cache is absent.
    _w = 0.8 / len(METHODS)
    _off = (len(METHODS) - 1) / 2
    _fig, _ax = plt.subplots(figsize=(10, 4.4))
    for _i, _m in enumerate(METHODS):
        _keys = list(STRATA) + ["ALL"]
        _means = np.array([_by[(_m, _k)]["mean_fp"] for _k in _keys])
        # nan std = a single run, not a zero spread. Drawn as a bar with no cap; the
        # legend says so, because a capless bar and a zero-length cap look alike.
        _stds = np.nan_to_num(
            np.array([_by[(_m, _k)]["std_fp_across_seeds"] for _k in _keys]), nan=0.0
        )
        # FP >= 0: clip the lower whisker so the bar bottom never dips below zero.
        _lower = np.minimum(_stds, _means)
        _ax.bar(
            _x + (_i - _off) * _w,
            _means,
            _w,
            yerr=[_lower, _stds],
            capsize=2,
            label=_m + ("" if COLLECTION[_m] == ENV_VARIANT else " †"),
            color=COLORS[_m],
            hatch=None if COLLECTION[_m] == ENV_VARIANT else "//",
            edgecolor="white",
            linewidth=0.4,
            error_kw={"elinewidth": 0.8},
        )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_groups)
    _ax.set_ylim(bottom=0)
    _ax.set_xlabel(ENV.stratum_axis_label)
    _ax.set_ylabel("rollout FP (fails before first success)")
    _ax.set_title(
        f"Mean rollout FP by stratum ({ENV_VARIANT} test, n={N_PROBLEMS})\n"
        "error bars = ± across-seed std · no cap = single run · † = grafted from legacy"
    )
    _ax.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(CACHE_DIR, ENV, LEGACY_CACHE, LEGACY_ONLY, compare):
    if not ENV.has_timing:
        time_records = []
        plan_gen_s = {}
        refine_cap_s = None
    else:
        # SPECTRE's §2b timing on the kinder SB2D variant is grafted from the legacy cache,
        # exactly as the FP path grafts its FP -- `load_time_records_per_seed` reads a single
        # cache dir, so merge the `legacy_only` methods' timing in from `LEGACY_CACHE`. A
        # no-op on DD2D (its `legacy_only` VLMPlan-32B carries no timing). `plan_gen_s` and
        # `refine_cap_s` still come from the PRIMARY cache's meta.json.
        _prim_t = compare.load_time_records_per_seed(CACHE_DIR)
        _leg_t = (
            compare.load_time_records_per_seed(LEGACY_CACHE)
            if LEGACY_CACHE != CACHE_DIR
            else []
        )
        time_records = compare.merge_time_records(_prim_t, _leg_t, LEGACY_ONLY)
        plan_gen_s = compare.load_plan_gen_s(CACHE_DIR)
        refine_cap_s = compare.load_refine_cap_s(CACHE_DIR)
    return plan_gen_s, refine_cap_s, time_records


@app.cell(hide_code=True)
def _(ENV, mo):
    mo.md(
        r"""## 2b · Wall-clock to first success — is the inference worth it?

        Mean **seconds to the first successful refinement** = plan-gen + inference +
        refinement, summed over the candidates each method tries. Weighs FP by real cost
        (a failed refine runs ~15 ms–20 s). **Under the deployed per-candidate refinement
        cap** (`refine_cap_s`): a slow near-feasible *trap* is abandoned at the cap, so an
        *uncapped* total over-punishes the learned ranker (its few failures are the
        expensive ones). The uncapped total + the cap's tiny FP cost are printed below.
        VLMPlan's plan-gen is 0 (its generation *is* the inference `infer_s`).
        """
        if ENV.has_timing
        else r"""## 2b · Wall-clock to first success

             _Deferred for this environment: its episodes carry real per-candidate refinement
             times, but filling §2b needs a per-env refinement cap (the DD2D 2 s cap would
             censor SB2D's ~10 s feasible refines) plus a precompute run. VLMPlan's own
             wall-clock is already cached._
             """
    )
    return


@app.cell
def _(ENV, compare, mo, plan_gen_s, refine_cap_s, summary_tidy, time_records):
    if not (ENV.has_timing and time_records):
        time_tidy: list = []
        _out = mo.md("")
    else:
        _h, _r, time_tidy = compare.build_time_table(
            time_records, plan_gen_s, use_capped=True
        )
        _, _, _unc = compare.build_time_table(
            time_records, plan_gen_s, use_capped=False
        )
        _all = {t["method"]: t for t in time_tidy if t["stratum"] == "ALL"}
        _all_unc = {t["method"]: t for t in _unc if t["stratum"] == "ALL"}
        _fp_unc = {
            t["method"]: t["mean_fp"] for t in summary_tidy if t["stratum"] == "ALL"
        }
        _capnote = (
            f"a {refine_cap_s:g}s per-candidate refinement cap"
            if refine_cap_s is not None
            else "no refinement cap"
        )
        print(f"wall-clock to first success (ALL), seconds — under {_capnote}:")
        for _m in sorted(_all, key=lambda m: _all[m]["mean_seconds"]):
            _e = _all[_m]
            _u = _all_unc[_m]["mean_seconds"]
            _fc, _fu = _e.get("fp_capped"), _fp_unc.get(_m)
            _fpn = (
                f"  | FP {_fu:.2f}->{_fc:.2f}"
                if (_fc is not None and _fu is not None)
                else ""
            )
            print(
                f"  {_m:<20s} {_e['mean_seconds']:8.3f}s  = plan-gen "
                f"{_e['plan_gen_s']:.3f} + infer {_e['infer_s']:.4f} + refine "
                f"{_e['refine_s']:.3f}   (uncapped {_u:6.3f}s){_fpn}"
            )
        _out = mo.md(
            compare.render_markdown(_h, _r)
            + f"\n\n_Total = plan-gen + inference + refinement seconds under **{_capnote}** "
            "— each skeleton refined for at most that long before the next, the deployed "
            "wall-clock configuration; ± is across seeds. The cap targets the expensive "
            "near-feasible failures a good ranker still tries, so it helps the learned "
            "ranker most and costs only a tiny FP increase (uncapped total and FP cost in "
            "the printout). Refinement reuses stored per-candidate times — a "
            "within-collection *relative* measure (collector parallelism, 20 s refine "
            "budget), fair across methods since each sums the same times. Inference is GPU "
            "wall-clock (CPU-tensorize + GPU-forward); plan-gen is a regenerated per-stratum "
            "proxy; PIGINet's inference is BCE-head only (CLIP features are cached)._"
        )
    _out
    return (time_tidy,)


@app.cell
def _(ENV, ENV_VARIANT, METHODS, N_PROBLEMS, np, plt, refine_cap_s, time_tidy):
    if ENV.has_timing and time_tidy:
        _all = {t["method"]: t for t in time_tidy if t["stratum"] == "ALL"}
        _ms = [m for m in METHODS if m in _all]
        _x = np.arange(len(_ms))
        _pg = np.array([_all[m]["plan_gen_s"] for m in _ms])
        _inf = np.array([_all[m]["infer_s"] for m in _ms])
        _ref = np.array([_all[m]["refine_s"] for m in _ms])
        _fig, _ax = plt.subplots(figsize=(9, 4.4))
        _ax.bar(_x, _pg, label="abstract plan-gen", color="#9aa4ad")
        _ax.bar(_x, _inf, bottom=_pg, label="inference", color="#e8a33c")
        _cap = "" if refine_cap_s is None else f", {refine_cap_s:g}s cap"
        _ax.bar(_x, _ref, bottom=_pg + _inf, label=f"refinement", color="#4a7fb5")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_ms, rotation=20, ha="right")
        _ax.set_ylabel("wall-clock to first success (s)")
        _ax.set_title(
            f"Wall-clock breakdown, ALL ({ENV_VARIANT} test, n={N_PROBLEMS}{_cap})\n"
            "refinement dominates; inference is the small orange sliver — the cost question"
        )
        _ax.legend()
        plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ENV, METHODS, SLAB, STRATA, compare, mo, time_tidy):
    # Exact per-component numbers behind the stacked bar above: one table per component
    # (inference, refinement), rows = method, cols = strata + ALL, each cell mean ± the
    # across-seed std. Plan-gen is a per-stratum constant shared by the pool methods, so
    # it gets one shared row rather than a column that repeats.
    if not (ENV.has_timing and time_tidy):
        _out = mo.md("")
    else:
        _by = {(t["method"], t["stratum"]): t for t in time_tidy}
        _methods = [m for m in METHODS if any(t["method"] == m for t in time_tidy)]
        _cols = list(STRATA) + ["ALL"]
        _labels = [SLAB.get(s, str(s)) for s in STRATA] + ["ALL"]

        def _cell(mean, std):
            # bare mean when the std is NaN: astar (deterministic) / VLMPlan (1 seed).
            if mean is None or (isinstance(mean, float) and mean != mean):
                return "—"
            if std is None or (isinstance(std, float) and std != std):
                return f"{mean:.3f}"
            return f"{mean:.3f} ± {std:.3f}"

        def _component_table(mean_key, std_key):
            _rows = []
            for _m in _methods:
                _r = [_m]
                for _s in _cols:
                    _e = _by.get((_m, _s), {})
                    _r.append(_cell(_e.get(mean_key), _e.get(std_key)))
                _rows.append(_r)
            return compare.render_markdown(["method"] + _labels, _rows)

        # plan-gen is identical across the pool methods (and 0 for VLMPlan), so show it once.
        _pool = next((m for m in _methods if m not in compare.SEQUENCE_METHODS), None)
        if _pool is not None:
            _pg = [
                f"{_by.get((_pool, _s), {}).get('plan_gen_s', 0.0):.3f}" for _s in _cols
            ]
            _pg_md = compare.render_markdown(
                ["component"] + _labels, [["plan-gen (s)"] + _pg]
            )
        else:
            _pg_md = ""

        _out = mo.md(
            "### §2b breakdown — exact per-component seconds (mean ± across-seed std)\n\n"
            "**Inference** — GPU forward for the learned methods, the VLM generation call "
            "for VLMPlan, `0` for astar.\n\n"
            + _component_table("infer_s", "infer_std")
            + "\n\n**Refinement** — per-candidate refine time summed to first success, "
            "under the deployed cap.\n\n"
            + _component_table("refine_s", "refine_std")
            + "\n\n**Plan-gen** — a per-stratum constant shared across the pool methods; "
            "VLMPlan's is 0 (its generation *is* the inference above).\n\n"
            + _pg_md
            + "\n\n_`±` is across the training seeds (3 for the learned methods); a bare "
            "number is a single deterministic / 1-seed run (astar, VLMPlan-GPT5.6). These "
            "reconcile with the §2b total table and the stacked bars._"
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3 · Survival curves

          Fraction of problems solved within ≤ k failed attempts (higher & further-left is
          better). `ALL` = whole split, the rest by stratum. Each curve is the mean of the
          per-seed curves.
          """)
    return


@app.cell
def _(COLORS, METHODS, SLAB, STRATA, df_seeds, np, plt):
    def _cdf(vals, ks):
        vals = np.asarray(vals)
        return [np.mean(vals <= k) for k in ks]

    def _mean_cdf(sub, ks):
        """Mean over seeds of each seed's own survival curve.

        A ``seed=None`` method is one deterministic run, so it contributes exactly one
        curve -- `groupby` would drop those rows entirely, hence the explicit branch.
        """
        if sub["seed"].notna().any():
            curves = [_cdf(g["fp"].values, ks) for _, g in sub.groupby("seed")]
        else:
            curves = [_cdf(sub["fp"].values, ks)]
        return np.mean(np.asarray(curves), axis=0)

    _ks = np.arange(0, 201)
    _panels = ["ALL"] + STRATA
    _fig, _axes = plt.subplots(1, len(_panels), figsize=(15, 3.4), sharey=True)
    for _ax, _k in zip(_axes, _panels):
        _sub = df_seeds if _k == "ALL" else df_seeds[df_seeds.stratum == _k]
        for _m in METHODS:
            _v = _sub[_sub.method == _m]
            _ax.plot(_ks, _mean_cdf(_v, _ks), label=_m, color=COLORS[_m])
        _ax.set_title("ALL strata" if _k == "ALL" else SLAB.get(_k, f"stratum {_k}"))
        _ax.set_xlabel("failed attempts k")
        _ax.set_ylim(0, 1.02)
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("P(FP ≤ k)")
    _axes[0].legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(ENV, mo):
    (mo.md(r"""## 4 · Ablation — what makes SPECTRE adaptive?

          Two adaptive components (both exactly zero at `F=∅`, accruing as the rollout
          observes failures): the **`coverage`/`waste`** columns and **record tokens** (one
          token per failing query). Both are switched on/off below at a matched setting.

          - **`coverage`** = recall over the failures' named-culprit pool `K` — the fraction
            of `K` a candidate discharges.
          - **`waste`** = precision over unexplained work — of the steps the abstraction
            says were unneeded, the fraction answering to nothing the evidence named.

          <!--
          Unified definition (deployed 2026-07-31). ⚠️ The arms below were **scored under
          the pre-unification definition**, so read §4 internally, not against §1. Arms are
          **seed 0** (only the deployed config was multi-seed), accepted by paired bootstrap
          over problems; `Δ vs floor` is measured against the no-columns/no-tokens arm.
          -->
          """) if ENV.has_ablations else mo.md(""))
    return


@app.cell
def _(CACHE_DIR, compare, pd):
    # Ablation arms live in their own cache dirs, not in SPECTRE_FAMILIES: an ablation is
    # one method's components switched off, not a method in the comparison.
    # Every arm stays loaded even when a table below does not show it -- `cov+waste,
    # tokens` is §4.2's "both" row, and the rest are one edit away from being restored.
    # The row builders skip arms absent from `abl_df`, so deleting an entry here would
    # silently drop a row rather than error.
    ABL_ARMS = {
        # "cov+waste" = --coverage-feats, which adds BOTH columns. The single-column arms
        # below are the --coverage-mode split; naming them apart matters because the pair
        # and the coverage column alone do not behave the same.
        "cov+waste, tokens": "abl_cov_rec_adaptive",
        "cov+waste, no tokens": "abl_cov_norec_adaptive",
        "no cov/waste, tokens": "abl_nocov_rec_adaptive",
        "neither (no cols, no tokens)": "abl_nocov_norec_adaptive",
        "coverage column only": "abl_cov_only_adaptive",
        "waste column only": "abl_waste_only_adaptive",
        "deployed (cov+waste, tokens)": "spectre3_adaptive",
        "deployed, records suppressed": "abl_suppress_records_adaptive",
    }

    _rows, _missing = [], []
    for _label, _subdir in ABL_ARMS.items():
        if not (CACHE_DIR / _subdir).is_dir():
            _missing.append(_label)
            continue
        _rows += [
            {**r, "arm": _label}
            for r in compare.load_named_fp_records_per_seed(CACHE_DIR, _subdir, _label)
        ]
    # Give an empty frame its expected columns, so an env with **no ablation arms cached**
    # (the kinder variant -- its ablations are SPECTRE-internal and read on the `sb2d` entry)
    # renders §4 as empty tables rather than raising on `abl_df["arm"]`/`abl_df.seed`.
    abl_df = pd.DataFrame(
        _rows or [], columns=["problem_id", "stratum", "fp", "seed", "arm"]
    )
    # PINNED TO SEED 0, explicitly. Only the deployed arm has more than one trained seed,
    # and this is not cosmetic: `_abl_row` pairs on `problem_id`, and `.loc[common]` over
    # a multi-seed frame returns one row per (seed, problem), so the paired bootstrap
    # below would silently receive arrays of different lengths.
    abl_df = abl_df[abl_df.seed == 0]
    # A missing arm is *reported*, never silently dropped -- a 2x2 quietly rendering as a
    # 2x1 is exactly how an ablation gets over-read.
    if _missing:
        print(f"!! not cached, omitted from the tables below: {_missing}")
    print(f"ablation arms loaded (seed 0): {sorted(abl_df['arm'].unique())}")
    return ABL_ARMS, abl_df


@app.cell(hide_code=True)
def _(ENV, mo):
    (
        mo.md(
            r"""### 4.1 · coverage × record tokens

          One component at a time. `Δ vs floor` is a paired bootstrap over problems against
          the no-columns/no-tokens arm (negative = better); a CI excluding 0 is starred."""
        )
        if ENV.has_ablations
        else mo.md("")
    )
    return


@app.cell
def _(ENV, STRATA, abl_df, mo, pd):
    from alphatamp.approaches.spectre import eda as _eda

    # The Δ baseline is the both-off floor arm (no columns AND no tokens), per-pid.
    _floor = abl_df[abl_df.arm == "neither (no cols, no tokens)"]
    FLOOR_BY_PID = dict(zip(_floor["problem_id"], _floor["fp"]))

    def _abl_row(label, sub, base_by_pid=FLOOR_BY_PID, cells=STRATA):
        row = {"arm": label, "n": len(sub)}
        row["ALL"] = f"{sub['fp'].mean():.2f}" if len(sub) else "—"
        for k in cells:
            s = sub[sub.stratum == k]["fp"]
            row[f"s{k}"] = f"{s.mean():.2f}" if len(s) else "—"
        # Pair on problem_id: both arms saw identical problems, so pairing removes the
        # between-problem variance that dominates between-arm variance here.
        common = sorted(set(sub["problem_id"]) & set(base_by_pid))
        if common and len(sub):
            a = sub.set_index("problem_id").loc[common, "fp"].to_numpy()
            b = pd.Series(base_by_pid).loc[common].to_numpy()
            d = _eda.bootstrap_mean_difference(a, b, num_resamples=10_000, seed=0)
            star = "" if d.ci_low <= 0 <= d.ci_high else " *"
            row["Δ vs floor"] = (
                f"{d.point:+.2f} [{d.ci_low:+.2f}, {d.ci_high:+.2f}]{star}"
            )
        else:
            row["Δ vs floor"] = "—"
        return row

    _order = [
        "no cov/waste, tokens",
        "cov+waste, no tokens",
        "deployed (cov+waste, tokens)",
    ]
    _rows = [
        _abl_row(_a, abl_df[abl_df.arm == _a])
        for _a in _order
        if _a in set(abl_df["arm"])
    ]
    ablation_2x2 = pd.DataFrame(_rows).set_index("arm") if _rows else pd.DataFrame()
    (ablation_2x2 if ENV.has_ablations else mo.md(""))
    return FLOOR_BY_PID, ablation_2x2


# --- commented out 2026-07-27 (notebook trim): the same four numbers as 4.1, arranged
# --- as a grid. Uncomment together with the two commented rows in 4.1's `_order`,
# --- otherwise it renders cells the table above no longer shows.
# @app.cell
# def _(abl_df, np, pd, plt):
#     # The 2x2 as a grid: rows = coverage on/off, cols = tokens on/off.
#     _grid = {
#         ("cov+waste", "tokens"): "cov+waste, tokens",
#         ("cov+waste", "no tokens"): "cov+waste, no tokens",
#         ("no cov/waste", "tokens"): "no cov/waste, tokens",
#         ("no cov/waste", "no tokens"): "neither (no cols, no tokens)",
#     }
#     _have = set(abl_df["arm"])
#     _tbl = pd.DataFrame(
#         [
#             [
#                 (
#                     f"{abl_df[abl_df.arm == _grid[(r, c)]]['fp'].mean():.2f}"
#                     if _grid[(r, c)] in _have
#                     else "not cached"
#                 )
#                 for c in ["no tokens", "tokens"]
#             ]
#             for r in ["no cov/waste", "cov+waste"]
#         ],
#         index=["no cov/waste", "cov+waste"],
#         columns=["no tokens", "tokens"],
#     )
#     _ = (np, plt)
#     print("mean FP (ALL) — lower is better\n")
#     _tbl
#     return


@app.cell(hide_code=True)
def _(ENV, mo):
    (mo.md(r"""### 4.2 · `coverage` vs `waste`, separated

          Each column on alone (record tokens stay on for all three). `coverage` = recall
          over the named-culprit pool; `waste` = precision over unexplained work. `neither`
          keeps tokens but no columns.""") if ENV.has_ablations else mo.md(""))
    return


@app.cell
def _(ENV, FLOOR_BY_PID, STRATA, abl_df, mo, pd):
    from alphatamp.approaches.spectre import eda as _eda2

    def _row2(label, sub):
        r = {"arm": label, "ALL": f"{sub['fp'].mean():.2f}" if len(sub) else "—"}
        for k in STRATA:
            s = sub[sub.stratum == k]["fp"]
            r[f"s{k}"] = f"{s.mean():.2f}" if len(s) else "—"
        common = sorted(set(sub["problem_id"]) & set(FLOOR_BY_PID))
        if common and len(sub):
            a = sub.set_index("problem_id").loc[common, "fp"].to_numpy()
            b = pd.Series(FLOOR_BY_PID).loc[common].to_numpy()
            d = _eda2.bootstrap_mean_difference(a, b, num_resamples=10_000, seed=0)
            star = "" if d.ci_low <= 0 <= d.ci_high else " *"
            r["Δ vs floor"] = (
                f"{d.point:+.2f} [{d.ci_low:+.2f}, {d.ci_high:+.2f}]{star}"
            )
        else:
            r["Δ vs floor"] = "—"
        return r

    _order = ["no cov/waste, tokens", "waste column only", "coverage column only"]
    _labels = {
        "no cov/waste, tokens": "neither",
        "waste column only": "waste only",
        "coverage column only": "coverage only",
    }
    _cov_rows = [
        _row2(_labels[_a], abl_df[abl_df.arm == _a])
        for _a in _order
        if _a in set(abl_df["arm"])
    ]
    coverage_split = (
        pd.DataFrame(_cov_rows).set_index("arm") if _cov_rows else pd.DataFrame()
    )
    (coverage_split if ENV.has_ablations else mo.md(""))
    return (coverage_split,)


# --- commented out 2026-07-27 (notebook trim): §4.4, the deploy-time suppress-records
# --- diagnostic. The arm is still cached (`abl_suppress_records`), so uncommenting both
# --- cells restores it. Measured 2026-07-27: 7.33 suppressed vs 7.50 as-trained, i.e.
# --- the deployed model does not read its record tokens at inference -- recorded in
# --- notebook.md, which is why the section is redundant here.
# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""### 4.4 · Does the deployed model actually read its record tokens?
#
#           A **deploy-time** diagnostic on the deployed checkpoint: run it with the
#           evidence memory emptied at every step. Deliberately a train/deploy mismatch,
#           and useful precisely because of that — it separates *"training with records
#           shaped the weights"* from *"the model reads the tokens at inference"*. A
#           small
#           gap means the tokens are largely inert at deploy even though training on them
#           mattered.
#
#           **Never quote the suppressed row as a method result.**
#           """)
#     return


# @app.cell
# def _(STRATA, abl_df, pd):
#     _pair = ["deployed (cov+waste, tokens)", "deployed, records suppressed"]
#     _have = set(abl_df["arm"])
#     if all(_a in _have for _a in _pair):
#         _rows = []
#         for _a in _pair:
#             _s = abl_df[abl_df.arm == _a]
#             _r = {"arm": _a, "ALL": f"{_s['fp'].mean():.2f}"}
#             for _k in STRATA:
#                 _r[f"s{_k}"] = f"{_s[_s.stratum == _k]['fp'].mean():.2f}"
#             _rows.append(_r)
#         suppress_table = pd.DataFrame(_rows).set_index("arm")
#     else:
#         suppress_table = pd.DataFrame(
#             {"note": ["suppress-records arm not cached"]}
#         ).set_index("note")
#     suppress_table
#     return (suppress_table,)


# --- commented out 2026-07-27 (notebook trim): grouped bars over the §4.1 arms. The
# --- per-stratum numbers are already in the 4.1 table. Uncomment together with the two
# --- commented rows in 4.1's `_order` if the full 2x2 is restored.
# @app.cell
# def _(COLORS, STRATA, abl_df, np, plt):
#     # Grouped bars over the arms that are cached, per stratum.
#     _order = [
#         a
#         for a in [
#             "neither (no cols, no tokens)",
#             "no cov/waste, tokens",
#             "cov+waste, no tokens",
#             "cov+waste, tokens",
#             "deployed (cov+waste, tokens)",
#         ]
#         if a in set(abl_df["arm"])
#     ]
#     _groups = [f"s{k}" for k in STRATA] + ["ALL"]
#     _x = np.arange(len(_groups))
#     _w = 0.8 / max(len(_order), 1)
#     _off = (len(_order) - 1) / 2
#     _pal = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(len(_order), 1)))
#     _fig, _ax = plt.subplots(figsize=(9, 4.0))
#     for _i, _a in enumerate(_order):
#         _s = abl_df[abl_df.arm == _a]
#         _means = [_s[_s.stratum == _k]["fp"].mean() for _k in STRATA]
#         _means.append(_s["fp"].mean())
#         _ax.bar(_x + (_i - _off) * _w, _means, _w, label=_a, color=_pal[_i])
#     _ax.set_xticks(_x)
#     _ax.set_xticklabels(_groups)
#     _ax.set_ylim(bottom=0)
#     _ax.set_ylabel("mean rollout FP")
#     _ax.set_xlabel(ENV.stratum_axis_label)
#     _ax.set_title("v3 ablation: coverage × record tokens (matched settings, 1 seed)")
#     _ax.legend(fontsize=7)
#     _ = COLORS
#     plt.tight_layout()
#     plt.gca()
#     return


@app.cell(hide_code=True)
def _(ENV, mo):
    mo.md(f"""## 5 · Planner inspector — scene + ordered plans

          Step through test problems with **◀ / ▶** (or the dropdown). Three panels:

          - **Scene** — {ENV.scene_legend}
          - **Every method on this problem** — rollout FP + first-feasible rank
            (independent of the method dropdown).
          - **Ordered plans** for the *selected* method, top-ranked → bottom; VLMPlan
            shows its own generated attempts (off the shared pool).

          For a `*-adaptive` method the toggle switches its **realized attempt order** vs
          its **t=0 score order**; in realized order `Δrank` shows which plans adaptivity
          promoted (`+`) / demoted (`−`) vs the static twin. The overview `FP` is the
          3-seed mean; the plan list is seed 0, so `1st-feasible rank` need not equal it.
          """)
    return


@app.cell
def _(REPO, cache_for, compare, np):
    # method -> (kind, static-scores dir, adaptive dir | None). The two SPECTRE modes share
    # one checkpoint, so an adaptive method's "static twin" scores are its own t=0 (c₀)
    # logits. Each method is read from its own cache via `cache_for` (primary, or the legacy
    # cache for a grafted method), so on the kinder variant PIGINet (native) and SPECTRE
    # (grafted) both render.
    INSPECT_SPEC = {
        "astar-dist": ("static", "astar", None),
        "PIGINet": ("static", "piginet/seed_0", None),
        "SPECTRE-static": ("static", "spectre3_static/seed_0", None),
        "SPECTRE-adaptive": (
            "adaptive",
            "spectre3_static/seed_0",
            "spectre3_adaptive",
        ),
    }
    # Sequence methods (VLMPlan) produce their own ordered attempt list off the shared
    # pool, so there is no skeleton to index; they're rendered from the cached `attempts`.
    INSPECT_SEQ = {
        m: d
        for m, d in compare.SEQUENCE_METHODS.items()
        if (cache_for(m) / d / "seed_0").is_dir()
    }
    INSPECT_METHODS = [
        m for m in INSPECT_SPEC if (cache_for(m) / INSPECT_SPEC[m][1]).is_dir()
    ] + list(INSPECT_SEQ)

    def insp_load(method, pid):
        """``(static scores | None, AdaptiveTrace | None)`` for one method+problem,
        read from the method's own cache (primary or legacy)."""
        _kind, sdir, adir = INSPECT_SPEC[method]
        _cache = cache_for(method)
        rec = compare.load_static_scores(_cache, sdir, pid) if sdir else None
        scores = np.asarray(rec["scores"], float) if rec else None
        trace = compare.load_adaptive_trace(_cache, adir, pid) if adir else None
        return scores, trace

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
        """
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

    _ = REPO
    return (
        INSPECT_METHODS,
        INSPECT_SPEC,
        INSPECT_SEQ,
        insp_effective,
        insp_load,
        insp_order,
    )


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
    return inspect_next, inspect_prev


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
def _(INSPECT_METHODS, mo):
    # These two drive the plan table only — the scene and the per-method overview below
    # are deliberately method-independent.
    _default = (
        "SPECTRE-adaptive"
        if "SPECTRE-adaptive" in INSPECT_METHODS
        else INSPECT_METHODS[0]
    )
    inspect_model = mo.ui.dropdown(
        options=INSPECT_METHODS, value=_default, label="method"
    )
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
def _(
    ENV_VARIANT,
    INSPECT_METHODS,
    INSPECT_PIDS,
    INSPECT_SEQ,
    REPO,
    cache_for,
    compare,
    df,
    get_pid,
    insp_order,
    inspect_model,
    inspect_realized,
    mo,
    pd,
    plt,
):
    from alphatamp.approaches.spectre import eda as eda_mod

    _ = plt  # keep dep explicit for marimo

    # Episodes come from the PRIMARY collection: the inspector renders v4 scenes and v4
    # pools, which is why it only offers v4-native methods.
    _test_dir = REPO / "data" / "spectre" / "raw" / ENV_VARIANT / "test"
    ep_by_pid = {
        int(ep.provenance.problem_id): ep
        for ep in eda_mod.load_split_episodes(_test_dir).episodes
    }

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
    for _m in INSPECT_METHODS:
        if _m in INSPECT_SEQ:
            # VLMPlan: its own attempt sequence, not a pool ranking. "1st-feasible rank"
            # is its own first-success index (== FP); a pid outside its stratified subset
            # has no record, shown blank.
            _vrec = compare.load_vlmplan_attempts(cache_for(_m), INSPECT_SEQ[_m], _pid)
            _rows.append(
                {
                    "": "▶" if _m == inspect_model.value else "",
                    "method": _m,
                    "FP": _fp_here.get(_m),
                    "1st-feasible rank": (
                        None
                        if _vrec is None or _vrec.get("censored")
                        else int(_vrec["fp"])
                    ),
                    "attempts": None if _vrec is None else len(_vrec["attempts"]),
                }
            )
            continue
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

    # The renderer comes from the environment registry, so the inspector shows the same
    # picture the VLMPlan prompt attached -- a reviewer comparing them is comparing like
    # with like. An environment with no renderer degrades to a note rather than an error.
    try:
        if ENV.render_scene is None:
            _scene = mo.md(f"*(no scene renderer registered for {ENV.key})*")
        else:
            _scene = mo.image(ENV.render_scene(_ep))
    except Exception as _e:  # noqa: BLE001 — geometry render is best-effort in the UI
        _scene = mo.md(f"*(scene render unavailable: {_e})*")

    mo.vstack(
        [
            mo.md(
                f"### problem **{_pid}**"
                f" &nbsp;·&nbsp; stratum **{compare.stratum_of(_pid)}**"
                f" &nbsp;·&nbsp; pool **{_k}** &nbsp;·&nbsp; feasible "
                f"**{sum(_feas)}/{_k}**"
                f" &nbsp;·&nbsp; shortest feasible plan **{_fmin}** ops"
                f" &nbsp;·&nbsp; [{INSPECT_PIDS.index(_pid) + 1}/{len(INSPECT_PIDS)}]"
            ),
            mo.hstack(
                [_scene, inspect_overview], widths=[1.35, 1], gap=1, align="start"
            ),
            mo.md(
                "<sub>`FP` is the cached headline metric; for a static method it "
                "differs from `1st-feasible rank` only by `rollout_fp`'s half-credit "
                "on exact "
                "score ties. `attempts` is blank for methods that never run a "
                "rollout.</sub>"
            ),
        ]
    )
    return ep_by_pid, inspect_overview


@app.cell
def _(
    ENV,
    INSPECT_SEQ,
    INSPECT_SPEC,
    cache_for,
    compare,
    ep_by_pid,
    get_pid,
    insp_load,
    insp_order,
    inspect_model,
    inspect_realized,
    mo,
    np,
    pd,
):
    import math as _math

    def _steps_of_skel(skel):
        """A pooled skeleton's operators as ``[(name, [args...]), ...]`` for the env
        formatter."""
        return [(op.name, [p.name for p in op.parameters]) for op in skel.operator_seq]

    def _plan_label_steps(steps):
        # The env-specific formatter (compare_envs). Replaces the DD2D-hardcoded local that
        # printed "retrieve ?" for every StickButton2D row.
        if ENV.plan_label is not None:
            return ENV.plan_label(steps)
        return " → ".join(name for name, _ in steps)

    _method = inspect_model.value
    _pid = get_pid()
    _ep = ep_by_pid[_pid]

    if _method in INSPECT_SEQ:
        # VLMPlan: render the model's OWN ordered proposals from the cached record; there
        # is no shared skeleton pool to index (its plans are off-pool by design).
        _vrec = compare.load_vlmplan_attempts(
            cache_for(_method), INSPECT_SEQ[_method], _pid
        )
        if _vrec is None:
            inspect_plan_table = mo.md(
                f"*(**{_method}** has no record for problem {_pid} — its stratified "
                "subset is 10/stratum, not the full test set.)*"
            )
            _legend = (
                f"**{_method}** generates its own plans, so it is shown off its own "
                "attempt list rather than the shared pool."
            )
        else:
            _att = _vrec["attempts"]
            _ff = int(_vrec["fp"]) if not _vrec.get("censored") else None
            _rows = []
            for _rank, _a in enumerate(_att):
                _steps = [(n, list(g)) for n, g in _a.get("steps", [])]
                _rows.append(
                    {
                        "attempt": _rank + 1,
                        "plan": (
                            _plan_label_steps(_steps)
                            if _steps
                            else ", ".join(_a.get("members", []))
                        ),
                        "len": len(_steps),
                        "feasible": "✓" if _a.get("label") == "success" else "",
                        "source": _a.get("source"),
                        "in pool": "✓" if _a.get("in_pool") else "",
                        "round": _a.get("round"),
                    }
                )

            def _style_seq(row_id, name, value):
                return {"background-color": "#e6f4ea"} if row_id == _ff else {}

            inspect_plan_table = mo.ui.table(
                pd.DataFrame(_rows),
                pagination=True,
                page_size=15,
                selection=None,
                show_column_summaries=False,
                style_cell=_style_seq,
            )
            _src = _vrec.get("first_success_source")
            _legend = (
                f"**{_method}** · the plans the model itself proposed, in order, each "
                f"refined for real. **FP = {_vrec['fp']}** (green = first feasible). "
                "`source` is `vlm` for a model proposal, `fill` for the published-order "
                "fallback used after the model ran dry; `in pool` marks a proposal that "
                f"coincides with a pooled candidate. First success from **{_src}**. "
                "Off-pool plans are expected here and are refined and charged like any "
                "other attempt."
            )
    else:
        # ---- pool-ranking methods (astar / SPECTRE): rank the shared candidate pool ----
        _lens = [len(s.operator_seq) for s in _ep.skeleton_pool]
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
        # has no single score. The honest per-row number is the score at the step it was
        # *picked* (that is the opinion it acted on); never-attempted candidates get the
        # final step, the most-informed context. Reading the final row for an attempted
        # candidate would be blank by construction: the model masks its own failure
        # context, so those entries come back NaN.
        _first_dead: dict[int, int] = {}
        if _adaptive and _tr.step_dead:
            for _t, _dead in enumerate(_tr.step_dead):
                for _i in _dead:
                    _first_dead.setdefault(int(_i), _t)

        _rows = []
        for _rank, _i in enumerate(_order):
            _row = {"rank": _rank}
            if _adaptive:
                _row["attempt"] = _rank + 1 if _rank < len(_tr.order) else None
            _row["plan"] = _plan_label_steps(_steps_of_skel(_ep.skeleton_pool[_i]))
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
                style["background-color"] = "#e6f4ea"  # FP boundary: first feasible
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
        if _adaptive:
            _legend = (
                f"**{_method}** · realized attempt order over {len(_tr.step_scores)} "
                "steps. `attempt` = the step the rollout actually ran it — it stops at "
                f"the first success, so those are exactly ranks 0…{_ff}; blank rows were "
                "**never tried** and are ordered by the final-step opinion. "
                "`ad.score` = the score at the step the candidate was **picked** (an "
                "adaptive ranker re-scores after every failure, so there is no single "
                "score); never-attempted rows use the final, most-informed step. "
                "`st.*` is the same checkpoint's **t=0** (`c₀`) ranking, so "
                "`Δrank = st.rank − rank` reads directly as **+ promoted** by adaptivity, "
                "**− demoted**. `demoted@t` = proof-demotion proved it dead from the "
                "attempt-`t` failure."
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
    mo.md(r"""## 6 · VLMPlan — usable plans generated per problem

          `n_proposed` = unique, valid, parseable plans each VLM arm generated itself
          (budget 200) — a capacity measure, not an attempt count. Error bars = ± std
          across problems within a stratum.
          """)
    return


@app.cell
def _(COLORS, ENV, SLAB, STRATA, cache_for, compare, np, pd, plt):
    _arms, _frames = [], {}
    for _arm, _subdir in compare.SEQUENCE_METHODS.items():
        # Each arm from its own cache: legacy for grafted (VLMPlan-32B), primary for
        # native (VLMPlan-GPT5.6). Reading LEGACY_CACHE for all dropped the native arm.
        _rows = compare.load_vlmplan_diagnostics(cache_for(_arm), _subdir)
        if not _rows:
            continue
        _v = pd.DataFrame(_rows)
        # Records written before the generation-stats fields existed carry None.
        _v["n_proposed"] = pd.to_numeric(_v.get("n_proposed"), errors="coerce").fillna(
            0
        )
        _frames[_arm] = _v
        _arms.append(_arm)

    _groups = [SLAB.get(k, str(k)) for k in STRATA] + ["ALL"]
    _x = np.arange(len(_groups))
    _w = 0.8 / max(len(_arms), 1)
    _off = (len(_arms) - 1) / 2
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    for _i, _arm in enumerate(_arms):
        _v = _frames[_arm]
        _sel = [_v[_v.stratum == _k]["n_proposed"] for _k in STRATA]
        _sel.append(_v["n_proposed"])
        _means = np.array([s.mean() for s in _sel])
        _stds = np.array([s.std() for s in _sel])
        # A count is >= 0, so clip the lower whisker rather than let it cross zero.
        _ax.bar(
            _x + (_i - _off) * _w,
            _means,
            _w,
            yerr=[np.minimum(_stds, _means), _stds],
            capsize=3,
            label=_arm,
            color=COLORS[_arm],
            error_kw={"elinewidth": 0.9},
        )
        print(
            f"  {_arm:<12s} "
            + "  ".join(
                f"{_g} {m:.1f}±{s:.1f}" for _g, m, s in zip(_groups, _means, _stds)
            )
        )
    if not _arms:
        _ax.text(
            0.5,
            0.5,
            f"no VLMPlan cache for {ENV.key}",
            ha="center",
            transform=_ax.transAxes,
        )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_groups)
    _ax.set_ylim(bottom=0)
    _ax.set_xlabel(ENV.stratum_axis_label)
    _ax.set_ylabel("usable plans generated (n_proposed)")
    _ax.set_title(
        "VLMPlan: unique valid plans produced per problem\n"
        "(plan budget 200; higher = less reliance on the fallback)"
    )
    # upper-left is the one corner no bar reaches; "best" lands it on top of s3/ALL.
    if _arms:
        _ax.legend(loc="upper left")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ENV_VARIANT, df_seeds, mo, summary_df):
    # Two files, because they answer different questions: the per-(method, seed, problem)
    # rows for anything that wants to re-aggregate, and the summary table exactly as §1
    # rendered it (mean + across-seed std + seed count per stratum).
    _dir = mo.notebook_dir()
    # Named for the collection: these two files used to be `dd2d_*` unconditionally, so
    # rendering the notebook for a second environment silently overwrote the first one's
    # export with rows that still said `dd2d` in the filename.
    _comparison = _dir / f"{ENV_VARIANT}_method_comparison.csv"
    _summary = _dir / f"{ENV_VARIANT}_method_summary.csv"
    df_seeds.to_csv(_comparison, index=False)
    summary_df.to_csv(_summary, index=False)
    print(
        f"wrote {_comparison}  ({len(df_seeds)} rows)\n"
        f"wrote {_summary}     ({len(summary_df)} rows)"
    )
    return


if __name__ == "__main__":
    app.run()
