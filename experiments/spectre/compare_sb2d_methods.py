import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# StickButton2D — SPECTRE v3 vs PIGINet and pure planning

          Compares plan-feasibility methods on the **held-out StickButton2D test split**
          (n = 100) by **rollout false-positives (FP)** — failed refinement attempts before
          the first success. Lower is better. The DD2D sibling of this notebook is
          `compare_dd2d_methods.py`; the two are deliberately diffable.

          > **Three seeds per learned method.** Both methods train on the *same* collection
          > and are scored against the *same* labels — PIGINet's examples are built from
          > the very `EpisodeRecord` pickles SPECTRE ranks, so the two cannot disagree
          > about which plans are feasible.

          **Methods.**

          - **astar-dist** — the non-learned planner-order baseline (score = −plan_idx).
            On this environment it is *worse than random* at b5; see the caveats.
          - **PIGINet** — the low-level predictor (CLIP + transformer over object image
            features + literals), **trained with BCE** (the original-paper baseline loss),
            AUPRC-selected. Static one-shot ranking.
          - **SPECTREv3-adaptive / -static** — the abstract-first re-ranker: one
            `FailureRecord` per failure, observed `coverage`/`waste`, and the record's
            abstract state delta. Purely learned.

          ---

          ### Read these three caveats before quoting a number

          **1. Strata are button counts, not subset sizes.** `s0…s3` = **b1, b2, b3, b5**.
          b1 and b2 are anchors with pools of ~2 and 6–34 that every method ties on
          (b1 static FP = 0.08); **b3 and b5 carry the result**. A pooled "ALL" mean over
          strata this unbalanced is not a method comparison.

          **2. PIGINet's image channel is degenerate here, by construction.** Every
          unpressed button renders as the same red disc of the same radius, so its CLIP
          crop is pixel-identical to every other button's — the image channel separates
          only {button, stick, robot}, which the type literals already give. `pose` and
          `shape` work exactly as on DD2D and are where this environment's signal lives.
          This is a fact about the perception StickButton2D affords, **not** evidence about
          low-level prediction in general, and it bounds what the representation contrast
          on this environment can be claimed to show.

          **3. b5's training split is 17 episodes.** The collection was cut at a wall-clock
          budget, so *both* methods' b5 column is substantially a generalisation result
          rather than a like-for-like stratum. Neither is advantaged — they share the split
          — but neither number should be quoted as a trained-on-b5 result.
          """)


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
    # StickButton2D, the second evaluation environment. There is no legacy collection to
    # graft from -- every row here is native to `stickbutton2d_v1` -- so LEGACY_* point at
    # the same variant and LEGACY_ONLY is empty. Keeping the machinery rather than tearing
    # it out means this notebook stays diffable against its DD2D sibling.
    ENV_VARIANT = "stickbutton2d_v1"
    LEGACY_VARIANT = "stickbutton2d_v1"
    DERIVED = REPO / "data" / "spectre" / "derived"
    CACHE_DIR = DERIVED / ENV_VARIANT / "compare_cache"
    LEGACY_CACHE = DERIVED / LEGACY_VARIANT / "compare_cache"
    # Methods with no dd2d_v4 row, grafted from dd2d_v3. Only VLMPlan is left: PIGINet
    # was retrained natively on v4 at three seeds on 2026-07-28, and regenerating VLMPlan
    # is two model arms x 100 problems (~10.5 h) to move a row that cannot plausibly shift
    # on a 0.08% label change.
    # Dropping a name here removes it from §1, §2 and §3 at once -- METHODS derives from
    # the loaded frame -- so the two SPECTRE-v1 entries are the single point of restore.
    # Nothing is grafted: VLMPlan and SPECTRE v2.2 were scoped out of the StickButton2D
    # comparison (v3 is the headline; the VLM row is a separate follow-on), so they are
    # absent rather than carried over from another collection.
    LEGACY_ONLY = []

    COLORS = {
        "astar-dist": "#7f7f7f",
        "PIGINet": "#ff7f0e",
        "SPECTRE-adaptive": "#1f77b4",
        "SPECTRE-static": "#7fb8de",
        "SPECTREv2-adaptive": "#2ca02c",
        "SPECTREv2-static": "#98df8a",
        "SPECTREv3-adaptive": "#d62728",
        "SPECTREv3-static": "#ff9896",
        "VLMPlan-8B": "#9467bd",
        "VLMPlan-32B": "#c5b0d5",
    }
    STRATA = [0, 1, 2, 3]
    # On DD2D a stratum is the min-feasible-subset size; here it is the button count, and
    # the two are not comparable. b1/b2 are anchors every method ties on (pools of ~2 and
    # 6-34); b3/b5 are the contest.
    STRATUM_LABEL = {0: "b1", 1: "b2", 2: "b3", 3: "b5"}
    print(f"primary: {CACHE_DIR}\nlegacy:  {LEGACY_CACHE}")
    return (
        CACHE_DIR,
        COLORS,
        ENV_VARIANT,
        LEGACY_CACHE,
        LEGACY_ONLY,
        LEGACY_VARIANT,
        REPO,
        STRATA,
        STRATUM_LABEL,
        dd2d_compare,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load

          Reads precomputed per-problem scores from both caches and grafts the methods
          that have no dd2d_v4 row. Nothing here runs inference — build the caches with:

          ```
          # spectre3 needs --force: its seed_0 dir predates the state-delta model
          python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 \
              --methods spectre3 --seeds 0 1 2 --force --no-ablations
          python experiments/spectre/precompute_dd2d_cache.py --env-variant dd2d_v4 \
              --methods astar piginet spectre2 --seeds 0 1 2
          ```

          Two frames come out of this, and the split is deliberate:

          - **`df_seeds`** — one row per *(method, seed, problem)*. §1 and §2 use it, so
            their `±` is the spread **across seeds**.
          - **`df`** — the same data collapsed to the per-*(method, problem)* mean over
            seeds. Every per-problem view (§3, §5, the CSV) uses it, so a method with
            three seeds still contributes one curve and one row per problem.
          """
             )
    return


@app.cell
def _(CACHE_DIR, LEGACY_CACHE, LEGACY_ONLY, LEGACY_VARIANT, dd2d_compare, pd):
    _primary = dd2d_compare.load_fp_records_per_seed(CACHE_DIR)
    _legacy = dd2d_compare.load_fp_records_per_seed(LEGACY_CACHE)
    merged = dd2d_compare.merge_collections(
        _primary,
        _legacy,
        LEGACY_ONLY,
        primary_name="dd2d_v4",
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
    METHODS = [m for m in dd2d_compare.METHOD_ORDER if m in set(df["method"])]

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
    mo.md(\
          r"""## 1 · Summary table — mean FP per stratum, ± across seeds

          **Stratum** = minimum feasible subset size (`s0` needs no blocker moved, `s3`
          needs three); s2/s3 are where methods separate.

          `±` is the spread **across seeds** of the per-stratum mean — the quantity a
          gate is judged on, and the one that says whether a margin is real. The `seeds`
          column is how many went into it: `3` for the learned methods, `-` for a single
          deterministic run (astar, and VLMPlan, which has one). **A row with one seed
          shows a bare mean, never `± 0.00`** — zero would claim a stability nobody
          measured.
          """)
    return


@app.cell
def _(COLLECTION, METHODS, dd2d_compare, df, df_seeds, merged, mo, pd):
    # `build_table` is the shared implementation (also behind `spectre_v3_table.py`), so
    # this table and the CLI reporter cannot drift apart. It takes the PER-SEED records
    # -- feeding it the collapsed frame would silently give the across-problem spread of
    # a seed-mean, which is the bug this section previously had.
    summary_header, summary_rows, summary_tidy = dd2d_compare.build_table(merged)
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
    mo.md(dd2d_compare.render_markdown(summary_header, summary_rows))
    return summary_df, summary_tidy


# --- commented out 2026-07-27 (notebook trim); uncomment both cells to restore the
# --- across-seed spread table. It reads `--` at one cached seed and fills in once
# --- `precompute_dd2d_cache.py --seeds 0 1 2 ...` has run.
# @app.cell(hide_code=True)
# def _(mo):
#     mo.md(r"""### Across-seed spread
#
#           The same table computed over **every cached seed**, where `±` is the spread of
#           the per-stratum mean *across seeds* rather than across problems. With one seed
#           cached it reads `--`; it fills in automatically once `precompute_dd2d_cache.py
#           --seeds 0 1 2 ...` has run. Kept visible so the distinction between the two
#           spreads is never implicit.
#           """)
#     return


# @app.cell
# def _(CACHE_DIR, dd2d_compare, mo):
#     _all_seeds = dd2d_compare.load_fp_records_per_seed(CACHE_DIR)
#     _header, _rows, _tidy = dd2d_compare.build_table(_all_seeds)
#     mo.md(
#         f"*{CACHE_DIR.parent.name} only (native rows); "
#         f"± = across seeds of the per-stratum mean.*\n\n"
#         + dd2d_compare.render_markdown(_header, _rows)
#     )
#     return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 2 · Mean FP per stratum (± across-seed std)

          Lower is better. Error bars are the **across-seed** std-dev of the per-stratum
          mean — the same numbers as §1, read from the same table, clipped at 0 so a bar
          never dips below zero (FP ≥ 0).

          A bar with **no cap** has one run and therefore no measurable spread (astar,
          VLMPlan); that is different from a spread of zero. Hatched bars are grafted
          from the older collection.
          """
             )
    return


@app.cell
def _(COLLECTION, COLORS, ENV_VARIANT, METHODS, STRATA, np, plt, summary_tidy):
    # One source of truth with §1: read the means and across-seed stds straight out of
    # `build_table`'s tidy output rather than recomputing from a frame, which is how a
    # chart and the table above it end up disagreeing.
    _by = {(t["method"], t["stratum"]): t for t in summary_tidy}
    _groups = [str(k) for k in STRATA] + ["ALL"]
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
    _ax.set_xlabel("min-feasible-subset stratum")
    _ax.set_ylabel("rollout FP (fails before first success)")
    _ax.set_title(
        f"Mean rollout FP by stratum ({ENV_VARIANT} test, n=100)\n"
        "error bars = ± across-seed std · no cap = single run · † = grafted from dd2d_v3"
    )
    _ax.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 3 · Survival curves

          Fraction of problems solved within ≤ k failed attempts (higher & further-left
          is better). The `ALL` panel is the whole split; the rest split by stratum.

          Each curve is the **mean of the per-seed curves**, matching how §1 averages
          per-seed statistics. Pooling every (seed, problem) attempt into one curve
          would fold seed spread into what reads as a distribution over problems, and
          would make a 3-seed method look smoother than a 1-seed one for no real reason.
          """
             )
    return


@app.cell
def _(COLORS, METHODS, STRATA, df_seeds, np, plt):
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
        _ax.set_title("ALL strata" if _k == "ALL" else f"stratum {_k}")
        _ax.set_xlabel("failed attempts k")
        _ax.set_ylim(0, 1.02)
        _ax.grid(True, alpha=0.3)
    _axes[0].set_ylabel("P(FP ≤ k)")
    _axes[0].legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 4 · Ablation — what makes v3 adaptive?

          <!--
          Two components carry v3, and both are **adaptive**: they are exactly zero at
          `F=∅` and accrue only as the rollout observes failures.
          -->

          **`coverage` / `waste`** — two columns on each candidate, computed from the
          objects the refiner *reported* as blocking (`FailureRecord.culprits`) while
          failing the candidates already tried:

          ```
          coverage = |S(c) ∩ culprits| / |culprits|     waste = |S(c) \ culprits| / |S(c)|
          ```

          <!--
          These are §5.1's necessity features with per-object necessity **observed rather
          than predicted** — no head, no second loss, no geometry routine. They replaced
          `dead`, which was a *length* proxy (corr(dead,|S|) = −0.284): right at s3 where
          long plans are needed, wrong at s1 where short ones are, so tuning it only traded
          strata.

          
          **Record tokens** — one token per failing query, carrying the schema, arguments
          and observed culprits of each failure, attended over by a dedicated evidence
          channel.

          All arms below are held at the **same matched setting** — `--overlap-mode
          jaccard`, no record aggregation, no evidence-attention — so each contrast varies
          only what it names. The *deployed* row also carries record aggregation and
          evidence-attention — two smaller implementation switches, not part of the
          contrast under test.

          > **1 seed per arm — unlike §1–§3, which are 3-seed.** Only the deployed
          > configuration was ever trained at more than one seed; these component arms
          > are a frozen seed-0 study. They are accepted by **paired bootstrap over
          > problems** (the project's stated 1-seed convention): pairing removes the
          > between-problem variance that otherwise dominates. Everything here, including
          > the v2.2 baseline the Δ column is measured against, is seed 0.

          > **The `deployed` row post-dates the component arms.** It now carries the
          > state delta (`decisions.md` 2026-07-28); the six component arms predate it
          > and were not re-run. That makes `deployed` context for the contrast, not a
          > cell in it — which is what it always was, since it also carries record
          > aggregation and evidence-attention that the matched arms do not.

          > ### What is left when **both** components are off
          >
          > (The arm with no coverage/waste *and* no tokens — cached as
          > `abl_nocov_norec`, not shown in the tables below. Not to be confused with
          > §4.2's `neither`, which means neither *column* but keeps tokens.)
          >
          > Switching both off does **not** leave a static ranker. Three things still
          > respond to the failure set, and only the first two are model inputs:
          >
          > 1. **`avail_mask`** — already-tried candidates are forced to `-inf`. This is the
          >    "just the previously tried skeletons" channel, but it only *removes* them
          >    from the argmax; it cannot re-rank the survivors.
          > 2. **`jaccard`** — `cand_overlap[:, 1]`, the max Jaccard overlap between a
          >    candidate's manipulated set and any already-failed candidate's set. This is
          >    the one *learned* adaptive feature left. (Column 0, `dead`, is zeroed by
          >    `--overlap-mode jaccard`.)
          > 3. ~~**Proof-demotion**~~ — **cut from the method on 2026-07-30**, so on the
          >    deployed model there is no longer a third channel: `avail_mask` and
          >    `jaccard` are all that remain when both components are off. §4.3 prices the
          >    cut. On this floor arm the offset was worth **1.09 FP** (15.47 → 16.56) —
          >    roughly *all* of the arm's remaining adaptivity — against only **0.23** on
          >    the deployed model (7.20 → 7.44, 3 seeds), which is what made it affordable
          >    to remove.
          >
          > On `dd2d_v4` there is no fourth channel: the model falls back to v2.2's
          > hint-tier fact tokens when records are off, but **that collection has no
          > harvested `post_mortem` facts at all** (0 fact tokens on every example
          > measured), so the fallback is inert here.
          -->
          """)
    return


@app.cell
def _(CACHE_DIR, dd2d_compare, pd):
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
        # §4.3's pair; loaded here too so a missing dir is reported once, in one place.
        # These are the demotion-ON arms: demotion was cut from the method on 2026-07-30,
        # so switching it back on is now the ablation.
        "deployed + demotion": "abl_with_demotion_adaptive",
        "floor + demotion": "abl_floor_with_demotion_adaptive",
    }

    _rows, _missing = [], []
    for _label, _subdir in ABL_ARMS.items():
        if not (CACHE_DIR / _subdir).is_dir():
            _missing.append(_label)
            continue
        _rows += [
            {**r, "arm": _label}
            for r in dd2d_compare.load_named_fp_records_per_seed(
                CACHE_DIR, _subdir, _label
            )
        ]
    abl_df = pd.DataFrame(_rows)
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
def _(mo):
    # Restored 2026-07-28: this decorator had been left orphaned when the markdown below
    # it was commented out in the notebook trim, so it decorated the *next* cell object
    # and marimo refused to load the file at all.
    mo.md(\
          r"""### 4.1 · coverage × record tokens

          One component at a time, matched settings. `Δ vs v2.2` is a paired bootstrap
          over problems against the v2.2 yardstick (negative = better); a CI excluding 0
          is starred. Everything in this table is **seed 0**, including the baseline.
          """)
    return


@app.cell
def _(STRATA, abl_df, df_seeds, pd):
    from alphatamp.approaches.spectre import eda as _eda

    def _abl_row(label, sub, base_by_pid, cells=STRATA):
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
            row["Δ vs v2.2"] = (
                f"{d.point:+.2f} [{d.ci_low:+.2f}, {d.ci_high:+.2f}]{star}"
            )
        else:
            row["Δ vs v2.2"] = "—"
        return row

    # Seed 0 of v2.2, to match the seed-0 ablation arms. Taken from `df_seeds` rather
    # than the collapsed `df`, whose v2.2 row is now a 3-seed mean -- pairing a seed-0
    # arm against a 3-seed mean would make every Δ below a different comparison than the
    # one it claims to be.
    _v2 = df_seeds[(df_seeds.method == "SPECTREv2-adaptive") & (df_seeds.seed == 0)]
    V2_BY_PID = dict(zip(_v2["problem_id"], _v2["fp"]))

    _order = [
        # commented out 2026-07-27 (notebook trim); both arms are still cached and
        # loaded -- uncomment to restore the full 2x2. `cov+waste, tokens` is the same
        # cell as `deployed` minus record aggregation and evidence-attention.
        # "neither (no cols, no tokens)",
        # "cov+waste, tokens",
        "no cov/waste, tokens",
        "cov+waste, no tokens",
        "deployed (cov+waste, tokens)",
    ]
    _rows = [
        _abl_row(_a, abl_df[abl_df.arm == _a], V2_BY_PID)
        for _a in _order
        if _a in set(abl_df["arm"])
    ]
    # _rows.append(_abl_row("v2.2 yardstick", _v2, V2_BY_PID))
    ablation_2x2 = pd.DataFrame(_rows).set_index("arm")
    ablation_2x2
    return V2_BY_PID, ablation_2x2


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
def _(mo):
    mo.md(\
          r"""### 4.2 · `coverage` vs `waste`, separated

          The two columns have only ever been switched on together. `--coverage-mode`
          zeroes one of them (rather than narrowing the tensor, so the state-dict shape and
          the exact-absence equivalence oracle are untouched), which isolates each.

          They ask different questions: **`coverage`** is "does this candidate remove the
          objects the refiner reported as blocking", **`waste`** is "does it also remove
          objects that were never implicated". `no cov/waste, tokens` is the floor with
          neither column.
          """
             )
    return


@app.cell
def _(STRATA, V2_BY_PID, abl_df, pd):
    from alphatamp.approaches.spectre import eda as _eda2

    def _row2(label, sub):
        r = {"arm": label, "ALL": f"{sub['fp'].mean():.2f}" if len(sub) else "—"}
        for k in STRATA:
            s = sub[sub.stratum == k]["fp"]
            r[f"s{k}"] = f"{s.mean():.2f}" if len(s) else "—"
        common = sorted(set(sub["problem_id"]) & set(V2_BY_PID))
        if common and len(sub):
            a = sub.set_index("problem_id").loc[common, "fp"].to_numpy()
            b = pd.Series(V2_BY_PID).loc[common].to_numpy()
            d = _eda2.bootstrap_mean_difference(a, b, num_resamples=10_000, seed=0)
            star = "" if d.ci_low <= 0 <= d.ci_high else " *"
            r["Δ vs v2.2"] = f"{d.point:+.2f} [{d.ci_low:+.2f}, {d.ci_high:+.2f}]{star}"
        else:
            r["Δ vs v2.2"] = "—"
        return r

    _order = [
        "no cov/waste, tokens",
        "waste column only",
        "coverage column only",
        # "cov+waste, tokens",
    ]
    # "neither" here means neither *column* -- record tokens are still on in all four.
    _labels = {
        "no cov/waste, tokens": "neither",
        "waste column only": "waste only",
        "coverage column only": "coverage only",
        # "cov+waste, tokens": "both",
    }
    coverage_split = pd.DataFrame(
        [
            _row2(_labels[_a], abl_df[abl_df.arm == _a])
            for _a in _order
            if _a in set(abl_df["arm"])
        ]
    ).set_index("arm")
    coverage_split
    return (coverage_split,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""### 4.3 · What the demotion cut costs

          **Proof-tier demotion was cut from the method on 2026-07-30**
          (`decisions.md`). SPECTREv3 as reported everywhere above is a **purely learned
          ranker**: nothing outside the network touches its ordering. This section is the
          price of that choice.

          The offset it gives up: when an observed failure *proves* a candidate dead, its
          score was pushed back by a finite amount — never to `-inf`, so a wrong proof
          could only reorder, never lose the feasible plan (C5 / P-E). Sound, and now
          off. The machinery is kept and one flag away (`apply_demotion=True`), which is
          what the `+ demotion` rows below run.

          Both arms are **exactly paired** — same weights, same seeds, same episodes, and
          the proof state is advanced either way, so they differ only in the offset. It is
          a deploy-time switch; no retraining is involved.

          Read the two rows against each other: the gap between them is how much of the
          sound rule's value the *learned* components had already absorbed, and it is why
          the cut is affordable.

          > **The learned signal is a correlate, not a proof**, so this is a real trade and
          > not a free simplification — on a domain whose proofs fire more often than
          > DD2D's 6% it would go the other way. A Δ of exactly 0.00 would instead mean the
          > switch never took effect, which is why the two cache dirs are asserted to differ.
          """
             )
    return


@app.cell
def _(CACHE_DIR, STRATA, dd2d_compare, np, pd):
    from alphatamp.approaches.spectre import eda as _eda3

    # Own frame, deliberately NOT `abl_df`: that one is pinned to seed 0 so §4.1/§4.2's
    # bootstrap gets equal-length arrays. Here the pairs are matched per seed and the
    # collapse happens before the bootstrap, which is what lets the deployed row use all
    # three of its seeds.
    # (label, WITH-demotion dir, WITHOUT-demotion dir). Inverted 2026-07-30: demotion was
    # cut from the method, so the deployed arm is the *without* column and switching it
    # back on is the ablation. `Δ` below is (deployed − with demotion), i.e. what the
    # deployed configuration gives up by being purely learned.
    _PAIRS = [
        ("deployed v3", "abl_with_demotion_adaptive", "spectre3_adaptive"),
        (
            "floor: jaccard only",
            "abl_floor_with_demotion_adaptive",
            "abl_nocov_norec_adaptive",
        ),
    ]

    def _by_seed(subdir):
        """``{seed: {pid: fp}}`` for one cached arm, or None when it is absent."""
        if not (CACHE_DIR / subdir).is_dir():
            return None
        out = {}
        for r in dd2d_compare.load_named_fp_records_per_seed(CACHE_DIR, subdir, "x"):
            out.setdefault(r["seed"], {})[r["problem_id"]] = r["fp"]
        return out

    def _mean_over(seed_map, pids, stratum=None):
        sel = [
            p for p in pids if stratum is None or dd2d_compare.stratum_of(p) == stratum
        ]
        if not sel:
            return float("nan")
        per_seed = [np.mean([m[p] for p in sel]) for m in seed_map.values()]
        return float(np.mean(per_seed))

    _rows, _missing = [], []
    for _label, _on_dir, _off_dir in _PAIRS:
        _on, _off = _by_seed(_on_dir), _by_seed(_off_dir)
        if _on is None or _off is None:
            _missing.append(_label)
            continue
        # Only seeds cached for BOTH arms, so the pair is never half-matched.
        _seeds = sorted(set(_on) & set(_off))
        _on = {s: _on[s] for s in _seeds}
        _off = {s: _off[s] for s in _seeds}
        _pids = sorted(set.intersection(*[set(m) for m in _on.values()]))
        _variants = (
            ("deployed — no demotion", _off),
            ("+ demotion (ablation)", _on),
        )
        for _tag, _m in _variants:
            _r = {
                "arm": f"{_label} · {_tag}",
                "seeds": len(_seeds),
                "ALL": f"{_mean_over(_m, _pids):.2f}",
            }
            for _k in STRATA:
                _r[f"s{_k}"] = f"{_mean_over(_m, _pids, _k):.2f}"
            _r["Δ cost of the cut"] = ""
            _rows.append(_r)
        # Paired bootstrap on the seed-mean per problem -- the same collapse
        # `spectre_score_v3.py` does before pairing.
        _a = np.array([np.mean([_off[s][p] for s in _seeds]) for p in _pids])
        _b = np.array([np.mean([_on[s][p] for s in _seeds]) for p in _pids])
        _d = _eda3.bootstrap_mean_difference(_a, _b, num_resamples=10_000, seed=0)
        _star = "" if _d.ci_low <= 0 <= _d.ci_high else " *"
        _rows[-1][
            "Δ cost of the cut"
        ] = f"{_d.point:+.2f} [{_d.ci_low:+.2f}, {_d.ci_high:+.2f}]{_star}"
        # The failure mode this whole section is exposed to: if the two arms are
        # identical the switch never took effect and the ablation reads 0.00 with
        # nothing looking wrong. Say so loudly rather than rendering it as a result.
        if not np.any(_a != _b):
            print(
                f"!! {_label}: demotion-ON and demotion-OFF caches are IDENTICAL — the "
                "switch did not take effect; do not read the Δ as a measurement"
            )
    if _missing:
        print(f"!! not cached, omitted from §4.3: {_missing}")
    demotion_ablation = pd.DataFrame(_rows).set_index("arm")
    demotion_ablation
    return (demotion_ablation,)


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
#           shaped the weights"* from *"the model reads the tokens at inference"*. A small
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
#     _ax.set_xlabel("min-feasible-subset stratum")
#     _ax.set_title("v3 ablation: coverage × record tokens (matched settings, 1 seed)")
#     _ax.legend(fontsize=7)
#     _ = COLORS
#     plt.tight_layout()
#     plt.gca()
#     return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 5 · Planner inspector — scene + ordered plans

          Step through test problems with **◀ / ▶** (or the dropdown). Three panels:

          - **Scene** — the initial DD2D drawer, drawn from the episode's stored geometry.
            The **red** item is the retrieval target; blue items are concave; the dark
            frame is the wall band; the dashed box is the buffer. Labels are the item index
            (`item_5` → `5`), so they match the `stage {…}` sets in the plan table.
          - **Every method on this problem** — rollout FP and first-feasible rank, so they
            can be compared without toggling. **Independent of the method dropdown.**
          - **Ordered plans** for the *selected* method — top-ranked → bottom-ranked.

          For a `*-adaptive` method the toggle switches between its **realized attempt
          order** and its **t=0 score order**. In realized order the table also carries the
          *static twin's* rank and score, so `Δrank` shows exactly which plans adaptivity
          promoted (`+`) or demoted (`−`), and `demoted@t` names the failure whose proof
          killed a candidate outright.

          > **Only `dd2d_v4`-native methods appear here.** A grafted method's cached scores
          > index the **dd2d_v3** candidate pool, while the scene and plan list rendered
          > below come from the **dd2d_v4** episode. On the ~5% of problems whose pools
          > differ, its rank column would be quietly wrong — so PIGINet, VLMPlan and
          > SPECTRE v1 are excluded from this section rather than shown with a subtly
          > incorrect ordering. Their FP appears in §1–§3.

          > An adaptive ranker re-scores the pool after **every** failure, so no candidate
          > has a single score. The cache stores the whole per-step matrix; `ad.score`
          > reports the step each candidate was *picked* on — the opinion the rollout acted
          > on.

          > **`demoted@t` no longer changes anything for v3.** Proof-demotion was cut on
          > 2026-07-30, so for SPECTREv3 that column reads as *"a proof would have killed
          > this candidate at attempt t"* — the deduction is still computed and recorded,
          > it just no longer moves the ordering. For SPECTREv2, which keeps its demotion,
          > it is still causal.

          > **This section shows seed 0**, while §1–§3 report 3-seed means. It renders one
          > checkpoint's per-step score matrix, which has no multi-seed analogue — an
          > averaged attempt order is not an order any model ran. The `FP` column in the
          > overview table *is* the 3-seed mean, so it will not generally equal the
          > `1st-feasible rank` of the seed-0 ordering beside it.
          """
             )
    return


@app.cell
def _(CACHE_DIR, REPO, dd2d_compare, np):
    # method -> (kind, static-scores dir, adaptive dir | None). Both modes of a SPECTRE
    # family share one checkpoint, so an adaptive method's "static twin" scores are its
    # own t=0 (c₀) logits. Restricted to dd2d_v4-native methods -- see the section note.
    INSPECT_SPEC = {
        "astar-dist": ("static", "astar", None),
        "SPECTREv2-static": ("static", "spectre2_static/seed_0", None),
        "SPECTREv2-adaptive": (
            "adaptive",
            "spectre2_static/seed_0",
            "spectre2_adaptive",
        ),
        "SPECTREv3-static": ("static", "spectre3_static/seed_0", None),
        "SPECTREv3-adaptive": (
            "adaptive",
            "spectre3_static/seed_0",
            "spectre3_adaptive",
        ),
    }
    INSPECT_METHODS = [
        m for m in INSPECT_SPEC if (CACHE_DIR / INSPECT_SPEC[m][1]).is_dir()
    ]

    def insp_load(method, pid):
        """``(static scores | None, AdaptiveTrace | None)`` for one method+problem."""
        _kind, sdir, adir = INSPECT_SPEC[method]
        rec = dd2d_compare.load_static_scores(CACHE_DIR, sdir, pid) if sdir else None
        scores = np.asarray(rec["scores"], float) if rec else None
        trace = dd2d_compare.load_adaptive_trace(CACHE_DIR, adir, pid) if adir else None
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
    return INSPECT_METHODS, INSPECT_SPEC, insp_effective, insp_load, insp_order


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
        "SPECTREv3-adaptive"
        if "SPECTREv3-adaptive" in INSPECT_METHODS
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
    CACHE_DIR,
    ENV_VARIANT,
    INSPECT_METHODS,
    INSPECT_PIDS,
    REPO,
    dd2d_compare,
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
    from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (
        reconstruct_scene as _reconstruct_scene,)
    from alphatamp.approaches.spectre.envs.dd2d.spectre_render import (
        scene_figure as _scene_fig,)

    _ = (plt, CACHE_DIR)  # keep deps explicit for marimo

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
                f" &nbsp;·&nbsp; pool **{_k}** &nbsp;·&nbsp; feasible "
                f"**{sum(_feas)}/{_k}**"
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
    return ep_by_pid, inspect_overview


@app.cell
def _(
    INSPECT_SPEC,
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
        _row["plan"] = _plan_label(_ep.skeleton_pool[_i])
        _row["len"] = int(_lens[_i])
        _row["feasible"] = "✓" if _feas[_i] else ""
        if _adaptive:
            _at = _rank if _rank < len(_tr.order) else len(_tr.step_scores) - 1
            _v = _tr.step_scores[_at][_i]
            _row["ad.score"] = None if _math.isnan(_v) else round(float(_v), 3)
            _row["st.rank"] = _st_rank.get(_i)
            _row["st.score"] = None if _scores is None else round(float(_scores[_i]), 3)
            _row["Δrank"] = None if _i not in _st_rank else int(_st_rank[_i]) - _rank
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
    if _adaptive:
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
    mo.md(\
          r"""## 6 · VLMPlan — usable plans generated per problem

          How much of each VLM arm's attempt list it produced *itself*, before the
          published-order fallback takes over. `n_proposed` counts the **unique, valid,
          parseable** plans a run generated (the plan budget is 200, the same size as
          the candidate pool), so it is a capacity measure, not an attempt count.

          Error bars are ± std **across problems** within a stratum.
          """
             )
    return


@app.cell
def _(COLORS, LEGACY_CACHE, STRATA, dd2d_compare, np, pd, plt):
    _arms, _frames = [], {}
    for _arm, _subdir in dd2d_compare.SEQUENCE_METHODS.items():
        _rows = dd2d_compare.load_vlmplan_diagnostics(LEGACY_CACHE, _subdir)
        if not _rows:
            continue
        _v = pd.DataFrame(_rows)
        # Records written before the generation-stats fields existed carry None.
        _v["n_proposed"] = pd.to_numeric(_v.get("n_proposed"), errors="coerce").fillna(
            0
        )
        _frames[_arm] = _v
        _arms.append(_arm)

    _groups = [f"s{k}" for k in STRATA] + ["ALL"]
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
        _ax.text(0.5, 0.5, "no VLMPlan cache", ha="center", transform=_ax.transAxes)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_groups)
    _ax.set_ylim(bottom=0)
    _ax.set_xlabel("min-feasible-subset stratum")
    _ax.set_ylabel("usable plans generated (n_proposed)")
    _ax.set_title(
        "VLMPlan: unique valid plans produced per problem\n"
        "(plan budget 200; higher = less reliance on the fallback)"
    )
    # upper-left is the one corner no bar reaches; "best" lands it on top of s3/ALL.
    _ax.legend(loc="upper left")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df_seeds, mo, summary_df):
    # Two files, because they answer different questions: the per-(method, seed, problem)
    # rows for anything that wants to re-aggregate, and the summary table exactly as §1
    # rendered it (mean + across-seed std + seed count per stratum).
    _dir = mo.notebook_dir()
    df_seeds.to_csv(_dir / "dd2d_method_comparison.csv", index=False)
    summary_df.to_csv(_dir / "dd2d_method_summary.csv", index=False)
    print(
        f"wrote {_dir / 'dd2d_method_comparison.csv'}  ({len(df_seeds)} rows)\n"
        f"wrote {_dir / 'dd2d_method_summary.csv'}     ({len(summary_df)} rows)"
    )
    return


if __name__ == "__main__":
    app.run()
