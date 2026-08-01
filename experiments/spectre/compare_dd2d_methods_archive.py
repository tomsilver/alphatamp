import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# DD2D comparison — archived analyses (dd2d_v3)

          Sections retired from `compare_dd2d_methods.py` on 2026-07-27, when that
          notebook was retargeted to **dd2d_v4** to lead with SPECTRE v3. They are kept
          runnable rather than deleted: each answers a question that was live at the time
          and may become live again.

          > **This notebook reads `dd2d_v3`, deliberately.** Every analysis below depends
          > on artifacts that exist only for that collection — the `spectre_lenctx`
          > intervention cache, PIGINet's per-candidate static scores, and the two VLMPlan
          > runs. Nothing here has been re-derived on dd2d_v4.

          | § | what it asks | status |
          |---|---|---|
          | 1 | **T0** — how much of each ranking is explained by plan length? | prose written against **dd2d_v2**; figures render v3 |
          | 2 | **T1** — does the failure context use object *identity*, or only length? | concluded H2 (length-only) on v1 |
          | 3 | Length-bias explorer — logit vs plan length, per method | diagnostic |
          | 4 | VLMPlan diagnostics — did the model find the plan, or the fallback? | diagnostic |

          > ⚠️ **The §1 and §2 prose was written against dd2d_v2.** The tables and figures
          > render **v3** data, but the surrounding takeaways still quote v2 numbers
          > (η² ≈ 1.0, Δ ≈ 0 per stratum, n = 142). Read those cells as *hypotheses to
          > re-check against the figures beside them*, not as conclusions. This was already
          > true before the move; re-deriving them was out of scope then and remains so.

          > ⚠️ **T1 (§2) concerns SPECTRE v1 only.** v3 replaced the pooled context vector
          > with per-failure record tokens *and* observed coverage/waste, so "the context
          > is length-only" is a statement about v1's Ψ and says nothing about v3. The v3
          > analogue is the `suppress_records` diagnostic in the main notebook's ablation.
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

    from alphatamp.approaches.spectre import compare

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
    # dd2d_v3 on purpose — see the header. The lenctx / PIGINet / VLMPlan caches these
    # sections read do not exist for dd2d_v4.
    ENV_VARIANT = "dd2d_v3"
    CACHE_DIR = REPO / "data" / "spectre" / "derived" / ENV_VARIANT / "compare_cache"

    METHODS = [m for m in compare.METHOD_ORDER if not m.startswith("SPECTREv3")]
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
    return CACHE_DIR, COLORS, METHODS, STRATA, dd2d_compare, np, pd, plt, sns


@app.cell
def _(CACHE_DIR, dd2d_compare, pd):
    df = pd.DataFrame(compare.load_fp_records(CACHE_DIR))
    print(df.groupby(["method", "stratum"]).size().unstack())
    return (df,)


@app.cell
def _(CACHE_DIR, dd2d_compare, pd):
    from alphatamp.approaches.spectre import eda as eda_mod

    # Test episodes keyed by problem_id; reused by the T0 fits and the length-bias
    # explorer for plan text and per-candidate lengths.
    _spectre_test = CACHE_DIR.parents[2] / "raw" / CACHE_DIR.parent.name / "test"
    ep_by_pid = {
        int(ep.provenance.problem_id): ep
        for ep in eda_mod.load_split_episodes(_spectre_test).episodes
    }
    lengths_by_pid = {
        pid: [len(s.operator_seq) for s in ep.skeleton_pool]
        for pid, ep in ep_by_pid.items()
    }
    fit_df = pd.DataFrame(compare.load_length_fit_records(CACHE_DIR, lengths_by_pid))
    pos_df = pd.DataFrame(
        compare.load_position_by_length_records(CACHE_DIR, lengths_by_pid)
    )
    ladder_df = pd.DataFrame(
        compare.load_adaptive_ladder_records(CACHE_DIR, lengths_by_pid)
    )
    print(
        f"lengths for {len(lengths_by_pid)} problems; "
        f"fit rows={len(fit_df)}  ladder rows={len(ladder_df)}"
    )
    return ep_by_pid, fit_df, ladder_df, lengths_by_pid, pos_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1 · T0 — how much does plan length explain each ranking?

          Test **T0**, for every method with static per-skeleton scores. A DD2D plan's
          *length* is its operator count (`2·(blockers staged)+1`). For each method we
          measure, **per episode** (so the statistic is already per-problem-normalized —
          a per-episode correlation is unchanged by z-scoring each problem's logits):

          - **R²(len)** *(headline)* — the **linear** length-R² = `pearson(score, length)²`:
            the fraction of score variance a straight-line fit on length explains.
          - **pearson** — signed linear correlation (`+` ⇒ higher-scored plans are longer).
          - **η²(len)** *(secondary)* — fraction of score variance explained by length
            *group* (categorical R²); credits **any** length structure, monotone or not.
          - **spearman** — rank correlation between score and length.

          **Reading R² vs η².** R²(len) ≪ η²(len) means the length preference is
          *non-monotone* — length drives the ranking, but not in a straight line, so the
          linear R² understates it. That is exactly v1-static's regime (it front-loads
          1- and 3-blocker plans, buries 2-blocker plans), so **read R² alongside η² and
          the learned length curve below**, not on its own.

          Each `*-adaptive` method has no static per-skeleton scores; its **t=0** ranking
          is *provably identical* to its `*-static` twin (same checkpoint, empty failure
          context → `c₀`), so its one-shot row mirrors static.
          """)
    return


@app.cell
def _(STRATA, fit_df, pd):
    def _m(sub, col):
        return f"{sub[col].mean():.2f}" if len(sub) else "—"

    def _row(label, sub):
        r = {
            "method": label,
            "R²(len)": _m(sub, "r2"),
            "pearson": _m(sub, "pearson"),
            "η²(len)": _m(sub, "eta2"),
            "spearman": _m(sub, "spearman"),
        }
        for _k in STRATA:
            r[f"R² s{_k}"] = _m(sub[sub.stratum == _k], "r2")
        return r

    _static = ["astar-dist", "PIGINet", "SPECTRE-static", "SPECTREv2-static"]
    _rows = [_row(_meth, fit_df[fit_df.method == _meth]) for _meth in _static]
    # Each *-adaptive one-shot == its *-static twin (c₀ identity).
    for _lab, _twin in [
        ("SPECTRE-adaptive (t=0 ≡ static)", "SPECTRE-static"),
        ("SPECTREv2-adaptive (t=0 ≡ static)", "SPECTREv2-static"),
    ]:
        _rows.append(_row(_lab, fit_df[fit_df.method == _twin]))
    t0_table = pd.DataFrame(_rows).set_index("method")
    print(
        "T0: R²(len) = linear (monotone) length-R² = pearson²; "
        "η²(len) = categorical (any-shape) length-R². "
        "R² ≪ η² ⇒ a non-monotone length preference (see the length curve below)."
    )
    t0_table
    return (t0_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Learned length curve

          Mean within-episode attempt-position (0 = tried first … 1 = tried last) of each
          plan-length tier. Reveals *which* length policy each method learned, and is what
          a scalar R² cannot show: astar is monotone short-first; PIGINet is ~flat (it
          discriminates *within* length, not by it); SPECTRE-static (v1) is a non-monotone
          length lookup (front-loads 1- and 3-blocker plans, buries 2-blocker plans) — the
          shape behind its high η² but low linear R².
          """)
    return


@app.cell
def _(COLORS, pos_df, plt):
    _fig, _ax = plt.subplots(figsize=(6.2, 4.0))
    for _m in ["astar-dist", "PIGINet", "SPECTRE-static", "SPECTREv2-static"]:
        _sub = pos_df[pos_df.method == _m].sort_values("length")
        _ax.plot(
            _sub["length"], _sub["mean_pos"], marker="o", label=_m, color=COLORS[_m]
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
        "Linear vs categorical length-R² per method\n"
        "(gap ⇒ non-monotone length preference)"
    )
    _ax.legend()
    _ = COLORS  # keep palette dep explicit for marimo
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Realized-order length ladder (v1 vs v2 adaptive)

          Over the sequence each adaptive method *actually attempts* (until first
          success), does it climb to longer plans as it fails? Positive
          `spearman(position, length)` ⇒ a length-escalation ladder. (Reads the cached
          `order` trace.)
          """
             )
    return


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
          - **PIGINet is the low-level comparator**: it ranks *within* a length tier using
            image geometry (high within-length variance), the structural opposite of a
            pure length prior.
          - **Each `*-adaptive` t=0 ≡ its `*-static` twin**; the adaptive lever is the
            realized-order re-ranking (length ladder above).
          """
             )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 2 · T1 — length-only-context intervention (does c_t use identity?)

          Test **T1** (length-only variant). Rerun the **SPECTRE v1** adaptive rollout,
          but replace every failed skeleton in the failure context Ψ sees with a
          **random *other* pool skeleton of the same plan length** — correct length,
          random object identity, consistent `s_L` (it is a real pooled skeleton). The
          selection still masks the *real* attempted skeletons; only the context is
          scrambled (`eda.spectre_evaluate_length_only_context`, averaged over 3 seeds ×
          3 surrogate draws in `spectre_lenctx/`).

          **If mean rollout-FP is unchanged ⇒ Ψ ignores which objects failed and uses
          only size/length (H2).** A drop ⇒ identity carries signal. (Regression guard:
          `scramble=False` reproduces SPECTRE-adaptive bitwise.)
          """
             )
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, df, pd):
    _lc = pd.DataFrame(
        compare.load_named_fp_records(
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
          plan leaves **SPECTRE v1**-adaptive's rollout-FP **unchanged** (Δ ≈ 0, CI
          includes 0 in every stratum). The context embedding `c_t` does shift a hair
          under scrambling (‖Δc‖ ~ 5e-3 on the d=64 vector), but never enough to change
          a selection — so v1's context module Ψ carries only a *whisper* of identity
          and is, functionally, **size/length-only**. This confirms **H2**: v1's
          adaptive gains (concentrated on s0/s1) are length-regime escalation, not
          failed-subset identification.

          Combined with T0 (v1-static ≈ a pure length ranking, η²≈1.0), SPECTRE **v1**
          on DD2D uses **length and nothing identity-specific** — neither in the static
          ranker nor in the adaptive context. Breaking the same-length symmetry (telling
          *which* same-size subset is right) needs diagnostic evidence the abstract
          representation does not carry — the v2.1 typed-evidence motivation, and
          ultimately v3's observed `coverage`/`waste`, which *do* read object identity
          out of the refiner's reported culprits.
          """
             )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 3 · Length-bias explorer — logit vs plan length

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
          r"""## 4 · VLMPlan diagnostics — did it work, and is the number informative?

          Two questions the headline FP alone cannot answer:

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
    # One block per model arm. An arm with no cache is skipped rather than erroring.
    _frames, _blocks = {}, []
    for _arm, _subdir in compare.SEQUENCE_METHODS.items():
        _rows = compare.load_vlmplan_diagnostics(CACHE_DIR, _subdir)
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


if __name__ == "__main__":
    app.run()
