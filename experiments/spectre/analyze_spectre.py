import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# SPECTRE EDA

          Go/no-go diagnostic battery per `SPECTRE_EDA_SPEC.md`. Runs Group 1 sanity,
          train↔test key-overlap diagnostic, Group 2 five baselines, Group 3 scalars
          with bootstrap CIs, and the §6 pass bar.

          All heavy lifting lives in `alphatamp.approaches.spectre.eda`; this notebook
          is a thin presentation layer.

          **Eval-split convention.** Baselines B1–B5 evaluate on the **test** split for
          honest comparison. B3/B4 additionally fit `»` on the **train** split.
          Validation is intentionally untouched — reserved for SPECTRE's own
          hyperparameter selection later.
          """
             )
    return


@app.cell
def _(mo):
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scienceplots
    import seaborn as sns
    from scipy.stats import wilcoxon

    from alphatamp.approaches.spectre import eda

    sns.set_theme(context="notebook", style="whitegrid")

    # Resolve the data root relative to this notebook file so the notebook runs
    # from any launch directory. Data lives at the repo root
    # (<repo>/data/spectre); this notebook is at <repo>/experiments/spectre, so
    # the root is two directories up. Fall back to the legacy cwd-relative path
    # if marimo cannot determine the notebook directory.
    _nb_dir = mo.notebook_dir()
    DATA_ROOT = (
        (_nb_dir / ".." / ".." / "data" / "spectre").resolve()
        if _nb_dir is not None
        else Path("../data/spectre")
    )
    ENV_VARIANT = "dd2d_v4"
    ATTEMPT_BUDGET = 30

    train_dir = DATA_ROOT / "raw" / ENV_VARIANT / "train"
    test_dir = DATA_ROOT / "raw" / ENV_VARIANT / "test"
    print(f"train_dir: {train_dir}")
    print(f"test_dir:  {test_dir}")

    # SPECTRE candidate-method evaluation. Swap this path to compare different
    # trained checkpoints (e.g. c1_baseline / e_augoff / f3b_static_pool).
    SPECTRE_CHECKPOINT_PATH = (
        DATA_ROOT / "checkpoints" / "r3_visit_rate" / ENV_VARIANT / "seed_0" / "best.pt"
    )
    # SPECTRE_NAME = f"SPECTRE_{SPECTRE_CHECKPOINT_PATH.parents[2].name}"  # e.g. "SPECTRE_c2_revert"

    SPECTRE_NAME = "Learned Adaptive Reordering"
    print(f"SPECTRE checkpoint: {SPECTRE_CHECKPOINT_PATH}")
    print(f"SPECTRE label:      {SPECTRE_NAME}")

    plt.style.use(["science", "no-latex", "nature"])
    plt.rcParams.update(
        {
            # Times New Roman throughout, matching ICLR body font
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            # ICLR uses 10 pt body; size labels so they read clearly at column width
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            # Clean spines (science style already drops top/right; reinforce here)
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Slightly heavier lines so they survive downscaling to column width
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # High-res output
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return (
        ATTEMPT_BUDGET,
        DATA_ROOT,
        ENV_VARIANT,
        SPECTRE_CHECKPOINT_PATH,
        SPECTRE_NAME,
        eda,
        np,
        pd,
        plt,
        sns,
        test_dir,
        train_dir,
        wilcoxon,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 1.

          Load splits (train + test)
          """
             )
    return


@app.cell
def _(eda, test_dir, train_dir):
    train = eda.load_split_episodes(train_dir)
    test = eda.load_split_episodes(test_dir)

    print(f"train: {len(train.episodes)} episodes, k_max={train.k_max}")
    print(f"test:  {len(test.episodes)} episodes, k_max={test.k_max}")

    train_n_success = sum(1 for ep in train.episodes if ep.summary.num_success >= 1)
    test_n_success = sum(1 for ep in test.episodes if ep.summary.num_success >= 1)
    print(
        f"train trainable (n_succ>=1): {train_n_success}"
        f" | test trainable (n_succ>=1): {test_n_success}"
    )
    return test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## Group 1 — episode sanity (on train)

          Per spec §1, Group 1 describes the training collection itself.
          """
             )
    return


@app.cell
def _(eda, train):
    # 3.1 Pool cap confirmation
    cap_frac = eda.pool_cap_fraction(train)
    print(f"3.1 pool_cap_fraction = {cap_frac:.3f}  (expect ~1.0)")
    return (cap_frac,)


@app.cell
def _(eda, np, plt, train):
    # 3.2 Cross-problem skeleton diversity
    U, N_slots = eda.count_unique_canonical_keys(train)
    ratio = U / N_slots if N_slots else 0.0
    print(f"3.2 U = {U}  N_slots = {N_slots}  U/N_slots = {ratio:.3f}")
    curve = eda.rarefaction_curve(train, num_shuffles=10, seed=0)
    jaccards = eda.jaccard_pair_sample(train, num_pairs=10000, seed=0)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].plot(np.arange(1, len(curve) + 1), curve)
    _axes[0].set_xlabel("episodes processed")
    _axes[0].set_ylabel("cumulative unique canonical keys")
    _axes[0].set_title("Rarefaction curve (avg over 10 shuffles)")
    _axes[1].hist(jaccards, bins=30, range=(0, 1))
    _axes[1].set_xlabel("Jaccard similarity between episode pool key-sets")
    _axes[1].set_ylabel("count")
    _axes[1].set_title(f"Pool Jaccard histogram (n={len(jaccards)} pairs)")
    plt.tight_layout()
    plt.show()
    return (U,)


@app.cell
def _(eda, np, plt, train):
    # 3.3 Episode success rate
    frac_with_success, n_succ_over_k = eda.success_rate_distribution(train)
    print(f"3.3 fraction_with_success = {frac_with_success:.3f}")
    print(
        f"    mean n_succ/K = {n_succ_over_k.mean():.3f}  median = {np.median(n_succ_over_k):.3f}"
    )
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(n_succ_over_k, bins=20, range=(0, 1))
    _ax.set_xlabel("n_succ / K (per episode)")
    _ax.set_ylabel("count")
    _ax.set_title("Episode success-fraction distribution (train)")
    plt.tight_layout()
    plt.show()
    return (frac_with_success,)


@app.cell
def _(ATTEMPT_BUDGET, eda, train):
    # 3.4 Default-order budget exhaustion
    exhaust_frac = eda.default_order_budget_exhaustion(
        train, attempt_budget=ATTEMPT_BUDGET
    )
    print(
        f"3.4 default_order_budget_exhaustion = {exhaust_frac:.3f}"
        f"  (fraction with T_default > {ATTEMPT_BUDGET})"
    )
    return (exhaust_frac,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## 3.5 Train↔test key-overlap diagnostic

          Interpretation context for Group 2 / Group 3:

          - `test_keys_seen_fraction ≥ 0.8`: overlapping-pool regime, B3/B4 carry signal.
          - `≤ 0.1`: disjoint-pool regime, B3/B4 degenerate to default order,
            Δ≈0 is mechanical. SPECTRE's Φ/Ψ may still exploit structure the discrete
            baselines cannot (spec §5.1 caveat).
          """
             )
    return


@app.cell
def _(eda, test, train):
    overlap = eda.train_eval_key_overlap(train, test)
    print(f"unique train keys: {overlap.num_unique_train_keys}")
    print(f"unique test  keys: {overlap.num_unique_test_keys}")
    print(
        f"test keys seen in train: {overlap.test_keys_seen_in_train}"
        f"  fraction: {overlap.test_keys_seen_fraction:.3f}"
    )
    print(
        f"median per-episode seen fraction:"
        f" {overlap.median_per_episode_seen_fraction:.3f}"
    )
    print(
        f"pairwise co-occurrence density:"
        f" {overlap.pairwise_cooccurrence_density:.3f}"
    )
    print(f"regime: {overlap.regime().upper()}")
    return (overlap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## Group 2 — baselines (all five on test)

          B1 random floor, B2 heuristic-aware order, B3 static-historical (fit on
          train), B4 adaptive-historical (fit on train), B5 oracle. All five evaluate on
          the test split. Trainable subset (`n_succ >= 1`) is applied before
          aggregating.

          **B2 note.** B2 ranks each problem's stored skeleton pool by the FF
          heuristic's cumulative score along each skeleton's STRIPS trajectory — the
          same heuristic an A*-based abstract-plan generator would consult. The closed-
          form lex order produced by the routedtransport2d enumerator is reported as
          `B2_default_order_lex` alongside it as a sanity row, but is not the baseline
          we compare against (it ignores the problem instance).
          """
             )
    return


@app.cell
def _(ATTEMPT_BUDGET, eda, pd, test, train):
    b1 = eda.random_floor_baseline(
        test, attempt_budget=ATTEMPT_BUDGET, mc_permutations=100, seed=0
    )
    b2_lex = eda.default_order_baseline(test, attempt_budget=ATTEMPT_BUDGET)
    b3 = eda.static_historical_baseline(train, test, attempt_budget=ATTEMPT_BUDGET)
    b4 = eda.adaptive_historical_baseline(train, test, attempt_budget=ATTEMPT_BUDGET)
    b5 = eda.oracle_ceiling(test, attempt_budget=ATTEMPT_BUDGET)
    summary = pd.DataFrame(
        [
            {
                "baseline": _r.name,
                "mean_attempts": float(_r.attempts.mean()),
                "sd_attempts": float(_r.attempts.std()),
                "mean_wall_clock_s": float(_r.wall_clock.mean()),
                "censoring_rate": float(_r.censored.mean()),
                "n_episodes": len(_r.attempts),
            }
            for _r in (b1, b2_lex, b3, b4, b5)
        ]
    ).set_index("baseline")
    # B2_default_order (lex) is the deployment baseline order.
    summary
    return b2_lex, b3, b4, b5, summary


@app.cell
def _(eda, np, plt, test, train):
    # Solvability-at-cap: fraction of problems solvable within the first k
    # planner-ordered candidates. Gates any eval-side pool capping — successes
    # sit at every depth (test reaches ~1.0 only at k=30), so capping the
    # candidate pool would censor real successes. This is why B6 runs at the full
    # K=30 with no capping (decisions.md 2026-06-11).
    _sol_train = eda.solvability_at_cap(train, k_max=train.k_max)
    _sol_test = eda.solvability_at_cap(test, k_max=test.k_max)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(np.arange(1, len(_sol_train) + 1), _sol_train, "-o", ms=3, label="train")
    _ax.plot(np.arange(1, len(_sol_test) + 1), _sol_test, "-s", ms=3, label="test")
    _ax.set_xlabel("candidate cap k (first k planner-ordered skeletons)")
    _ax.set_ylabel("fraction of problems solvable within first k")
    _ax.set_title("Solvability vs candidate cap (RT2D-n3)")
    _ax.set_ylim(0, 1.02)
    _ax.legend()
    plt.tight_layout()
    plt.show()
    print(
        f"test solvable@15 = {_sol_test[14]:.2f}, @20 = {_sol_test[19]:.2f}, "
        f"@30 = {_sol_test[-1]:.2f}  (capping below 30 censors successes)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## B6 — DP-on-counts (receding-horizon lookahead)

          A new evaluation **baseline** (not SPECTRE): it reuses B4's calibrated count
          model as its `q`-model and looks `h−1` steps ahead over the
          cost-to-first-success Bellman recursion. By construction `h=1` reproduces B4
          exactly; `h≥2` adds expectimax lookahead with a re-conditioning greedy leaf.

          The search is **exact** (`m=None`): incremental Naive-Bayes scoring makes the
          `O(K²)`-leaf expectimax tractable through `h=4` (h=2 ≈ 9s, h=3 ≈ 86s, h=4 ≈
          minutes) at the full pool **K=30**, with no candidate capping (see the
          solvability-at-cap figure above: successes sit at every planner depth, so
          capping would censor real successes — `decisions.md` 2026-06-11). Set the
          optional `m=12` only to push `h≥5`. Swept over `DP_HORIZONS` on the uncensored
          budget (= pool cap 30).

          **Read the paired stats, not the marginal means** (printed below): the
          lookahead premium over B4 (= the `h=1` row) is small and saturating, against a
          much larger gap to SPECTRE — i.e. lookahead on the count model is not the
          missing ingredient. Extend `DP_HORIZONS` to `(1,2,3,4)` for the (slower) h=4
          row.
          """
             )
    return


@app.cell
def _(ATTEMPT_BUDGET, eda, np, pd, test, train, wilcoxon):
    DP_HORIZONS = (1, 2, 3)  # add 4 for the ~minutes-long exact h=4 row
    dp_sweep = {
        _h: eda.dp_on_counts_baseline(
            train,
            test,
            attempt_budget=ATTEMPT_BUDGET,
            depth=_h,
            objective="attempts",
        )
        for _h in DP_HORIZONS
    }
    b6_h1 = dp_sweep[1]
    b6_h2 = dp_sweep[2]

    dp_summary = pd.DataFrame(
        [
            {
                "method": dp_sweep[_h].name,
                "mean_attempts": float(dp_sweep[_h].attempts.mean()),
                "sd_attempts": float(dp_sweep[_h].attempts.std()),
                "mean_wall_clock_s": float(dp_sweep[_h].wall_clock.mean()),
                "censoring_rate": float(dp_sweep[_h].censored.mean()),
                "n_episodes": len(dp_sweep[_h].attempts),
            }
            for _h in DP_HORIZONS
        ]
    ).set_index("method")

    # Paired per-problem stats: the h-sweep is over the SAME test problems, so
    # report paired differences (Wilcoxon signed-rank + win/tie/loss), not just
    # marginal means. Positive Δ ⇒ the deeper horizon used fewer attempts.
    print("Paired per-problem lookahead premium (Δ = shallower − deeper attempts):")
    for _lo, _hi in zip(DP_HORIZONS[:-1], DP_HORIZONS[1:]):
        _a = dp_sweep[_lo].attempts
        _b = dp_sweep[_hi].attempts
        _d = _a - _b
        _w = int((_d > 0).sum())
        _t = int((_d == 0).sum())
        _l = int((_d < 0).sum())
        _p = wilcoxon(_a, _b).pvalue if np.any(_d != 0) else float("nan")
        print(
            f"  h{_lo}->h{_hi}: Δmean={_d.mean():+.3f}  "
            f"win/tie/loss={_w}/{_t}/{_l}  wilcoxon p={_p:.3g}"
        )

    # Calibration diagnostic: extreme-q frequency along each test episode's B4
    # rollout (|F| grows). The softmax posterior has no clip; this confirms q does
    # not saturate to 0/1 in the conditioning regime B6 exploits.
    _stats = eda._fit_adaptive(train)
    _eps = 1e-3
    _q_vals = []
    for _ep_idx in eda._trainable_episodes(test):
        _keys = test.skeleton_keys[_ep_idx]
        _remaining = set(range(len(_keys)))
        _failed: list = []
        while _remaining:
            for _idx in _remaining:
                _q_vals.append(eda._adaptive_q(_stats, _keys[_idx], _failed))
            _best = max(
                _remaining,
                key=lambda i: (eda._adaptive_score(_stats, _keys[i], _failed), -i),
            )
            _failed.append(_keys[_best])
            _remaining.discard(_best)
    _q_arr = np.array(_q_vals)
    _extreme = float(((_q_arr < _eps) | (_q_arr > 1 - _eps)).mean())
    print(
        f"DP-on-counts q calibration: extreme-q (<{_eps} or >{1 - _eps}) "
        f"fraction = {_extreme:.4f} over {_q_arr.size} (candidate, F) states"
    )
    dp_summary
    return (b6_h2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## SPECTRE — candidate method

          SPECTRE is the trained ranker we want to compare *against* B1–B5 (it is
          **not** a baseline). Loads `SPECTRE_CHECKPOINT_PATH`, runs the same per-
          episode attempt loop on test, and appends a row to the summary table above.
          Reuses `BaselineResult` only because its per-episode (attempts, wall_clock,
          censored, problem_ids) schema is generic.
          """
             )
    return


@app.cell
def _(
    ATTEMPT_BUDGET,
    DATA_ROOT,
    ENV_VARIANT,
    SPECTRE_CHECKPOINT_PATH,
    SPECTRE_NAME,
    eda,
    pd,
    summary,
    test,
):
    import torch

    from alphatamp.approaches.spectre import inference
    from alphatamp.approaches.spectre.env_registry import get_static_tag_predicates
    from alphatamp.approaches.spectre.vocab import Vocab

    vocab = Vocab.from_json(DATA_ROOT / "derived" / ENV_VARIANT / "train_vocab.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spectre_model = inference.load_checkpoint(
        SPECTRE_CHECKPOINT_PATH,
        vocab,
        device=device,
        fallback_static_tag_predicates=get_static_tag_predicates(ENV_VARIANT),
    )
    spectre_prior = inference.load_prior_for_checkpoint(SPECTRE_CHECKPOINT_PATH)
    print(f"SPECTRE prior: {type(spectre_prior).__name__}")
    spectre_result = eda.spectre_evaluate(
        test,
        spectre_model,
        vocab,
        attempt_budget=ATTEMPT_BUDGET,
        prior=spectre_prior,
        device=device,
        name=SPECTRE_NAME,
    )
    summary_1 = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "baseline": spectre_result.name,
                        "mean_attempts": float(spectre_result.attempts.mean()),
                        "sd_attempts": float(spectre_result.attempts.std()),
                        "mean_wall_clock_s": float(spectre_result.wall_clock.mean()),
                        "censoring_rate": float(spectre_result.censored.mean()),
                        "n_episodes": len(spectre_result.attempts),
                    }
                ]
            ).set_index("baseline"),
        ]
    )
    # Reconstruct the prior the checkpoint was trained against — eval with a
    # mismatched prior would silently produce bad numbers.
    # Append SPECTRE row to the summary table above. The 'baseline' column name
    # is kept for backwards compatibility; row label distinguishes SPECTRE from B1-B5.
    summary_1
    return device, spectre_model, spectre_prior, spectre_result, vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## Frozen-context (Ψ) ablation — is the context encoder load-bearing?

          De-risking ablation: does SPECTRE's strength come from the skeleton encoder
          Φ, or from the failure-conditioning context encoder Ψ? We compare the full
          pipeline against a **frozen-context** variant that, at every rollout step,
          pins the scorer's context vector to the learned empty-set vector `c₀`
          regardless of the actual failure set — removing the adaptive element. With a
          fixed context the per-skeleton scores never change, so the frozen variant is
          exactly a **learned static ranker**; the full-vs-frozen gap is SPECTRE's own
          analogue of the B3−B4 adaptive premium.

          Both variants share the checkpoint and the deterministic rollout, so the
          comparison is perfectly paired per episode. At attempt 1 the failure set is
          empty and the full variant *also* uses `c₀`, so the two always agree on the
          first pick — divergence can only begin at attempt 2.

          Inference-time freeze only (no retraining). Implementation +
          method/decision notes: `docs/notebook.md` (2026-06-06 entry),
          `docs/decisions.md`, and the `spectre_ablate_context.py` runner (whose
          numbers these cells reproduce).
          """
             )
    return


@app.cell
def _(
    ATTEMPT_BUDGET,
    device,
    eda,
    np,
    spectre_model,
    spectre_prior,
    spectre_result,
    test,
    vocab,
):
    # Traced rollouts for both variants (same checkpoint, deterministic). The
    # full run reproduces the headline ``spectre_result`` exactly; we keep its
    # trace and reuse ``spectre_result`` for the BaselineResult-based metrics.
    abl_full_res, full_traces = eda.spectre_evaluate_traced(
        test,
        spectre_model,
        vocab,
        attempt_budget=ATTEMPT_BUDGET,
        prior=spectre_prior,
        device=device,
        name="SPECTRE",
        freeze_context=False,
    )
    frozen_result, frozen_traces = eda.spectre_evaluate_traced(
        test,
        spectre_model,
        vocab,
        attempt_budget=ATTEMPT_BUDGET,
        prior=spectre_prior,
        device=device,
        name="SPECTRE-frozen-context",
        freeze_context=True,
    )
    # The ablation's full variant must match the headline SPECTRE row.
    assert np.allclose(abl_full_res.attempts, spectre_result.attempts)
    print(
        f"full mean attempts:   {spectre_result.attempts.mean():.3f}"
        f"  (censoring {spectre_result.censored.mean():.3f})"
    )
    print(
        f"frozen mean attempts: {frozen_result.attempts.mean():.3f}"
        f"  (censoring {frozen_result.censored.mean():.3f})"
    )
    return frozen_result, frozen_traces, full_traces


@app.cell
def _(SPECTRE_NAME):
    rename = {
        "B1_random_floor": "Random Ordering",
        "B2_heuristic_search": "Abstract Plan Generator With Heuristic",
        "B2_default_order": "Abstract Plan Generator",
        "B3_static_historical": "Static Historical Baseline",
        "B4_adaptive_historical": "Adaptive Historical Baseline",
        "B6_dp_h1_attempts": "DP-on-counts (h=1 ≡ B4)",
        "B6_dp_h2_attempts": "DP-on-counts (h=2)",
        "B6_dp_h3_attempts": "DP-on-counts (h=3)",
        "SPECTRE-frozen-context": "Frozen-Context Ablation (Static Ranker)",
        SPECTRE_NAME: SPECTRE_NAME,
    }
    return (rename,)


@app.cell
def _(
    ATTEMPT_BUDGET,
    b2_lex,
    b3,
    b4,
    b6_h2,
    frozen_result,
    np,
    plt,
    rename,
    spectre_result,
):
    # B2-B4 baselines, then DP-on-counts (B6, h=2 representative), then the
    # frozen-context ablation, then full SPECTRE. The frozen variant is an
    # ablation of SPECTRE, NOT a baseline (baselines are B1-B6); it gets its own
    # color so it never reads as full SPECTRE's purple.
    baselines = [b2_lex, b3, b4, b6_h2, frozen_result, spectre_result]
    colors = {
        "B1": "tab:gray",
        "B2": "tab:blue",
        "B3": "tab:orange",
        "B4": "tab:green",
        "B5": "tab:red",
        "B6": "tab:olive",
        "SPECTRE": "tab:purple",
        "SPECTRE_FROZEN": "tab:brown",
    }

    def color_tag(name: str) -> str:
        if name == "SPECTRE-frozen-context":
            return "SPECTRE_FROZEN"
        return name.split("_")[0] if name.startswith("B") else "SPECTRE"

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4.5))
    bins_a = np.arange(1, ATTEMPT_BUDGET + 3, 2) - 0.5
    for _r in baselines:
        _axes[0].hist(
            _r.attempts,
            bins=bins_a,
            histtype="step",
            linewidth=2,
            label=rename.get(_r.name),
            color=colors.get(color_tag(_r.name)),
        )
    _axes[0].set_xlabel("attempts to first success (T)")
    _axes[0].set_ylabel("count (test episodes)")
    _axes[0].set_title(
        "Attempts Distribution: Baselines vs Learned Adaptive Reordering"
    )
    _axes[0].legend()
    _axes[0].set_xticks(range(1, ATTEMPT_BUDGET + 1))
    for _r in baselines:
        _axes[1].hist(
            _r.wall_clock,
            bins=10,
            histtype="step",
            linewidth=2,
            label=_r.name,
            color=colors.get(color_tag(_r.name)),
        )
    _axes[1].set_xlabel("Cumulative Refinement Wall-clock (s)")
    _axes[1].set_ylabel("Count (Test Episodes)")
    _axes[1].set_title(
        "Wall-clock Distribution: Baselines vs Learned Adaptive Reordering"
    )
    _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return baselines, color_tag, colors


@app.cell
def _(baselines, color_tag, colors, np, plt, rename):
    from collections import Counter

    _fig, _axes = plt.subplots(1, 1, figsize=(10, 8))
    for _r in baselines:
        pass_rate = Counter(_r.attempts)
        cumulative_success_rate = np.array([pass_rate.get(i, 0) for i in range(30)])
        cumulative_success_rate = cumulative_success_rate.cumsum() / 100
        _axes.plot(
            np.arange(1, 30 + 1),
            cumulative_success_rate,
            label=rename.get(_r.name),
            color=colors.get(color_tag(_r.name)),
        )
    _axes.set_xlabel("Number of Abstract Plan Attempts")
    _axes.set_ylabel("Cumulative Success Rate")
    _axes.set_title("Cumulative Success Rate: Baselines vs SPECTRE")
    _axes.legend()
    _fig.savefig("cumulative_success_rate.svg", format="svg")
    return


@app.cell
def _(baselines, color_tag, colors, rename):
    method_order = [rename.get(_r.name, _r.name) for _r in baselines]
    palette = {
        rename.get(_r.name, _r.name): colors.get(color_tag(_r.name)) for _r in baselines
    }
    return method_order, palette


@app.cell
def _(baselines, method_order, palette, pd, plt, rename, sns):
    violin_wall_data = pd.DataFrame(
        [
            {"method": rename.get(_r.name, _r.name), "wall_clock_s": float(w)}
            for _r in baselines
            for w in _r.wall_clock
        ]
    )
    _fig, _ax = plt.subplots(figsize=(16, 8))
    sns.violinplot(
        data=violin_wall_data,
        x="method",
        y="wall_clock_s",
        order=method_order,
        palette=palette,
        inner="box",
        cut=0,
        ax=_ax,
    )
    _ax.set_xlabel("")
    _ax.set_ylabel("Cumulative Refinement Time to First Success (s)")
    _ax.set_title("Distribution of Refinement Time to First Success")
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.show()
    _fig.savefig("refinement_time_dist.svg", format="svg")
    return


@app.cell
def _(baselines, color_tag, colors, pd, plt, rename, sns):
    violin_data = pd.DataFrame(
        [
            {
                "method": rename.get(_r.name, _r.name),
                "attempts": int(a),
                "color": colors.get(color_tag(_r.name)),
            }
            for _r in baselines
            for a in _r.attempts
        ]
    )
    method_order_1 = [rename.get(_r.name, _r.name) for _r in baselines]
    palette_1 = {
        rename.get(_r.name, _r.name): colors.get(color_tag(_r.name)) for _r in baselines
    }
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=violin_data,
        x="method",
        y="attempts",
        order=method_order_1,
        palette=palette_1,
        inner="box",
        cut=0,
        ax=_ax,
    )
    _ax.set_xlabel("")
    _ax.set_ylabel("Attempts to First Success")
    _ax.set_title("Distribution of Attempts to First Success")
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(SPECTRE_NAME, baselines, pd, rename):
    # ── Research-paper comparison table: baselines vs SPECTRE ────────────────────
    spectre = next((_r for _r in baselines if _r.name == SPECTRE_NAME))
    comparison = [_r for _r in baselines if _r.name != SPECTRE_NAME]
    sp_mean_att = float(spectre.attempts.mean())
    sp_std_att = float(spectre.attempts.std())
    sp_mean_t = float(spectre.wall_clock.mean())
    sp_std_t = float(spectre.wall_clock.std())
    rows = []
    for _r in comparison:
        m_att = float(_r.attempts.mean())
        s_att = float(_r.attempts.std())
        m_t = float(_r.wall_clock.mean())
        s_t = float(_r.wall_clock.std())
        d_att = m_att - sp_mean_att
        d_t = m_t - sp_mean_t
        pct_t = d_t / sp_mean_t * 100
        rows.append(
            {
                "Method": rename.get(_r.name, _r.name),
                "Attempts (mean±std)": f"{m_att:.2f} ± {s_att:.2f}",
                "Time / s (mean±std)": f"{m_t:.2f} ± {s_t:.2f}",
                "Δ Attempts": f"+{d_att:.2f}" if d_att >= 0 else f"{d_att:.2f}",
                "Δ Time / s": f"+{d_t:.2f}" if d_t >= 0 else f"{d_t:.2f}",
                "Δ Time (%)": f"+{pct_t:.1f}%" if pct_t >= 0 else f"{pct_t:.1f}%",
            }
        )
    # SPECTRE row (reference, Δ = baseline)
    rows.append(
        {
            "Method": rename.get(spectre.name, spectre.name),
            "Attempts (mean±std)": f"{sp_mean_att:.2f} ± {sp_std_att:.2f}",
            "Time / s (mean±std)": f"{sp_mean_t:.2f} ± {sp_std_t:.2f}",
            "Δ Attempts": "—",
            "Δ Time / s": "—",
            "Δ Time (%)": "—",
        }
    )
    df_cmp = pd.DataFrame(rows)
    _spectre_label = rename.get(spectre.name, spectre.name)
    # The frozen-context row is an ablation of SPECTRE, not a baseline — give it
    # its own light-purple fill, distinct from the green SPECTRE row and the
    # plain baseline rows.
    _frozen_label = rename.get("SPECTRE-frozen-context", "SPECTRE-frozen-context")

    def _row_style(row):
        # if row["Method"] == _spectre_label:
        #     return ["background-color: #d5e8d4; font-weight: bold"] * len(row)
        # if row["Method"] == _frozen_label:
        #     return ["background-color: #e8d5e8; font-style: italic"] * len(row)
        return [""] * len(row)

    # Styler as the cell's last expression → marimo renders it inline.
    df_cmp.style.apply(_row_style, axis=1).set_caption(
        "Baselines + frozen-context ablation vs. SPECTRE — mean ± std over 100 "
        "test episodes; Δ = row − SPECTRE (positive = SPECTRE is better)"
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#2c3e50"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "6px 10px"),
                ],
            },
            {
                "selector": "td",
                "props": [("text-align", "center"), ("padding", "5px 10px")],
            },
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "12px"),
                    ("font-style", "italic"),
                    ("padding-bottom", "6px"),
                ],
            },
        ]
    ).hide(
        axis="index"
    )
    return (df_cmp,)


@app.cell
def _(df_cmp, plt):
    # ── Export the comparison table as a PDF for Overleaf ─────────────────────────
    _col_labels = list(df_cmp.columns)
    _cell_text = df_cmp.values.tolist()
    _n_rows = len(_cell_text)
    _n_cols = len(_col_labels)
    # Oversized figure — height trimmed to table content after rendering.
    _fig, _ax = plt.subplots(figsize=(13, 0.55 * (_n_rows + 1) + 0.5))
    _ax.set_position([0, 0, 1, 1])  # axes fills the whole canvas
    _ax.axis("off")
    _tbl = _ax.table(
        cellText=_cell_text,
        colLabels=_col_labels,
        loc="upper left",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    _tbl.auto_set_font_size(False)
    _tbl.set_fontsize(10)
    _tbl.auto_set_column_width(list(range(_n_cols)))
    for j in range(_n_cols):
        _tbl[0, j].set_facecolor("#2c3e50")
        _tbl[0, j].set_text_props(color="white", fontweight="bold")
    _spectre_row = _n_rows  # last row; row 0 is the header
    for j in range(_n_cols):
        _tbl[_spectre_row, j].set_facecolor("#d5e8d4")
        _tbl[_spectre_row, j].set_text_props(fontweight="bold")
    # The frozen-context ablation row (not a baseline) gets a distinct fill.
    _frozen_rows = [
        i
        for i, _m in enumerate(df_cmp["Method"], start=1)
        if "Frozen-Context" in str(_m)
    ]
    for i in _frozen_rows:
        for j in range(_n_cols):
            _tbl[i, j].set_facecolor("#e8d5e8")
    _highlight = {_spectre_row, *_frozen_rows}
    for i in range(1, _n_rows):
        if i % 2 == 0 and i not in _highlight:
            for j in range(_n_cols):
                _tbl[i, j].set_facecolor("#f5f5f5")
    # Resize figure height to match exact row count so there is no dead space.
    _row_height_in = 0.35
    _fig.set_size_inches(13, _row_height_in * (_n_rows + 1))
    _fig.savefig("method_comparison_table.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(_fig)
    print("Saved: method_comparison_table.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## SPECTRE adaptive premium and oracle headroom

          Paired bootstrap of SPECTRE against B3 (static-historical) and B5 (oracle),
          using the same `_assert_aligned` machinery as the existing Group 3 analysis.
          """
             )
    return


@app.cell
def _(b3, b5, eda, pd, spectre_result):
    sp_vs_b3 = eda.adaptive_premium(
        b3, spectre_result, metric="attempts", num_resamples=10_000, seed=0
    )
    sp_vs_b5 = eda.headroom(
        spectre_result, b5, metric="attempts", num_resamples=10_000, seed=0
    )
    sp_walls_vs_b3 = eda.adaptive_premium(
        b3, spectre_result, metric="wall_clock", num_resamples=10_000, seed=0
    )

    pd.DataFrame(
        [
            {
                "metric": "SPECTRE − B3 (attempts saved)",
                "point": sp_vs_b3.point,
                "ci_low": sp_vs_b3.ci_low,
                "ci_high": sp_vs_b3.ci_high,
            },
            {
                "metric": "SPECTRE − B3 (wall_clock saved)",
                "point": sp_walls_vs_b3.point,
                "ci_low": sp_walls_vs_b3.ci_low,
                "ci_high": sp_walls_vs_b3.ci_high,
            },
            {
                "metric": "SPECTRE → oracle gap (attempts)",
                "point": sp_vs_b5.point,
                "ci_low": sp_vs_b5.ci_low,
                "ci_high": sp_vs_b5.ci_high,
            },
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## Frozen-context ablation — paired diagnostics

          Full vs frozen on the same checkpoint and the same 100 test episodes
          (perfectly paired). Primary result: mean ± std attempts and the paired
          bootstrap Δ; then how *similar* the two rollouts are (per-index agreement,
          first-divergence) and where the frozen static ranker lands relative to B3 / B4
          (success@K).
          """
             )
    return


@app.cell
def _(eda, frozen_result, pd, spectre_result):
    abl_delta = eda.bootstrap_mean_difference(
        frozen_result.attempts, spectre_result.attempts
    )
    abl_wins, abl_ties, abl_losses = eda.win_tie_loss(spectre_result, frozen_result)
    print(
        f"paired Δ attempts (frozen − full): {abl_delta.point:+.3f}"
        f"  [95% CI {abl_delta.ci_low:+.3f}, {abl_delta.ci_high:+.3f}]"
        "  (positive = the context encoder Ψ saves attempts)"
    )
    print(
        f"per-episode full vs frozen: {abl_wins} wins / {abl_ties} ties /"
        f" {abl_losses} losses"
    )
    _abl_primary = pd.DataFrame(
        [
            {
                "variant": _label,
                "mean_attempts": float(_r.attempts.mean()),
                "std_attempts": float(_r.attempts.std()),
                # "censoring_rate": float(_r.censored.mean()),
                "mean_wall_clock_s": float(_r.wall_clock.mean()),
            }
            for _label, _r in (
                ("SPECTRE (full)", spectre_result),
                ("SPECTRE-frozen-context", frozen_result),
            )
        ]
    ).set_index("variant")
    _abl_primary
    return


@app.cell
def _(ATTEMPT_BUDGET, eda, frozen_traces, full_traces, plt):
    _agreement = eda.per_index_agreement(
        full_traces, frozen_traces, max_index=ATTEMPT_BUDGET
    )
    _rows = [(t, rate, n_co) for (t, rate, n_co) in _agreement if n_co > 0]
    _ts = [t for t, _, _ in _rows]
    _rates = [rate for _, rate, _ in _rows]
    _nco = [n_co for _, _, n_co in _rows]

    _divergence = eda.first_divergence_distribution(full_traces, frozen_traces)
    _int_keys = sorted(k for k in _divergence if isinstance(k, int))
    _has_never = "never" in _divergence
    _labels = [str(k) for k in _int_keys] + (["never"] if _has_never else [])
    _counts = [_divergence[k] for k in _int_keys] + (
        [_divergence["never"]] if _has_never else []
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _ax0 = _axes[0]
    _ax0.plot(_ts, _rates, marker="o", color="tab:purple")
    _ax0.set_xlabel("attempt index t")
    _ax0.set_ylabel("P(full and frozen pick the same skeleton)")
    _ax0.set_ylim(0, 1.05)
    _ax0.set_title("Per-index choice agreement (t=1 ≡ 1.0 by construction)")
    _ax0b = _ax0.twinx()
    _ax0b.bar(_ts, _nco, alpha=0.15, color="tab:gray")
    _ax0b.set_ylabel("# co-running episodes (bars)")
    _ax0b.grid(False)

    _axes[1].bar(_labels, _counts, color="tab:brown")
    _axes[1].set_xlabel("first attempt index where the variants diverge")
    _axes[1].set_ylabel("# test episodes")
    _axes[1].set_title("First-divergence distribution (min possible = 2)")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    ATTEMPT_BUDGET,
    b3,
    b4,
    eda,
    frozen_result,
    np,
    plt,
    rename,
    spectre_result,
):
    _ks = np.arange(1, ATTEMPT_BUDGET + 1)
    _curves = [
        ("B3_static_historical", b3, "tab:orange", "-"),
        ("B4_adaptive_historical", b4, "tab:green", "-"),
        ("SPECTRE-frozen-context", frozen_result, "tab:brown", "--"),
        (spectre_result.name, spectre_result, "tab:purple", "-"),
    ]
    _fig, _ax = plt.subplots(figsize=(9, 5.5))
    for _name, _r, _color, _ls in _curves:
        _sk = eda.success_at_k(_r, k_max=ATTEMPT_BUDGET)
        _ax.plot(
            _ks, _sk, _ls, color=_color, linewidth=2, label=rename.get(_name, _name)
        )
    _ax.set_xlabel("attempt budget K")
    _ax.set_ylabel("fraction of test episodes solved within ≤ K")
    _ax.set_title("Success@K: frozen-context ablation vs B3 / B4 / SPECTRE")
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## Group 3 — adaptive premium Δ and headroom H

          Paired bootstrap (10,000 resamples) over test episodes. Because all five
          baselines share the same `problem_ids` ordering (enforced by
          `adaptive_premium` / `headroom` via `_assert_aligned`), the resampled indices
          apply identically to both arms.
          """
             )
    return


@app.cell
def _(b2_lex, b3, b4, b5, eda, pd):
    delta_attempts = eda.adaptive_premium(
        b3, b4, metric="attempts", num_resamples=10_000, seed=0
    )
    delta_walls = eda.adaptive_premium(
        b3, b4, metric="wall_clock", num_resamples=10_000, seed=0
    )
    h_attempts = eda.headroom(
        b2_lex, b5, metric="attempts", num_resamples=10_000, seed=0
    )
    h_walls = eda.headroom(
        b2_lex, b5, metric="wall_clock", num_resamples=10_000, seed=0
    )

    table = pd.DataFrame(
        [
            {
                "metric": "adaptive_premium (attempts)",
                "point": delta_attempts.point,
                "ci_low": delta_attempts.ci_low,
                "ci_high": delta_attempts.ci_high,
            },
            {
                "metric": "adaptive_premium (wall_clock)",
                "point": delta_walls.point,
                "ci_low": delta_walls.ci_low,
                "ci_high": delta_walls.ci_high,
            },
            {
                "metric": "headroom (attempts)",
                "point": h_attempts.point,
                "ci_low": h_attempts.ci_low,
                "ci_high": h_attempts.ci_high,
            },
            {
                "metric": "headroom (wall_clock)",
                "point": h_walls.point,
                "ci_low": h_walls.ci_low,
                "ci_high": h_walls.ci_high,
            },
        ]
    )
    table
    return delta_attempts, h_attempts


@app.cell(hide_code=True)
def _(mo):
    mo.md(\
          r"""## §6 Pass bar

          Primary conditions 1–5 must all hold. Condition 6 (headroom ≥ 2) is flagged
          but non-blocking. An interpretive caveat is printed when the disjoint-pool
          regime is detected (spec §5.1).
          """
             )
    return


@app.cell
def _(
    ENV_VARIANT,
    U,
    cap_frac,
    delta_attempts,
    eda,
    exhaust_frac,
    frac_with_success,
    h_attempts,
    overlap,
    train,
):
    verdict = eda.evaluate_pass_bar(
        pool_cap_fraction_value=cap_frac,
        diversity_U=U,
        k_max=train.k_max,
        success_fraction=frac_with_success,
        budget_exhaustion_fraction=exhaust_frac,
        adaptive_premium_ci=delta_attempts,
        headroom_ci=h_attempts,
        key_overlap=overlap,
    )

    print(f"3.1 pool_cap_saturated       : {verdict.pool_cap_saturated}")
    print(f"3.2 diversity_nontrivial     : {verdict.diversity_nontrivial}")
    print(f"3.3 success_rate_adequate    : {verdict.success_rate_adequate}")
    print(f"3.4 default_budget_exhaustion: {verdict.default_budget_exhaustion}")
    print(f"5.1 adaptive_premium_positive: {verdict.adaptive_premium_positive}")
    print(
        f"5.2 headroom_meaningful      : {verdict.headroom_meaningful} (non-blocking)"
    )
    print(f"   disjoint_pools_flag       : {verdict.disjoint_pools_flag}")

    note = verdict.interpretive_note()
    if note:
        print("\nINTERPRETIVE NOTE:")
        print(note)

    print()
    if verdict.primary_pass:
        print(f"VERDICT: PROCEED to SPECTRE training on {ENV_VARIANT}.")
    else:
        failing = [
            name
            for name, passed in [
                ("3.1 pool_cap", verdict.pool_cap_saturated),
                ("3.2 diversity", verdict.diversity_nontrivial),
                ("3.3 success_rate", verdict.success_rate_adequate),
                ("3.4 budget_exhaustion", verdict.default_budget_exhaustion),
                ("5.1 adaptive_premium", verdict.adaptive_premium_positive),
            ]
            if not passed
        ]
        print(f"VERDICT: FAIL (failing: {failing}).")
        print("Document the decision (proceed, reconfigure, drop) before continuing.")
    return


@app.cell
def _(eda, train):
    top10 = eda.top_successful_skeleton_keys(train, n=10, rank_by="successes")
    for rank, s in enumerate(top10, start=1):
        print(
            f"#{rank:<2}  successes={s.successes:<4}  appearances={s.appearances:<4}"
            f"  p_hat={s.p_hat:.3f}  length={len(s.key)} ops"
        )
        print(eda.format_skeleton_key(s.key))
        print()
    return


if __name__ == "__main__":
    app.run()
