"""Frozen-context ablation for SPECTRE (Ψ de-risking).

Compares the full SPECTRE pipeline against a frozen-context variant on the
test split: at every rollout step the ablated variant feeds the scorer the
learned empty-F vector ``c_0`` instead of Ψ's encoding of the actual failure
set, removing the adaptive element. Because the score of each skeleton is
then fixed, the frozen variant is exactly a *learned static ranker* — the
full-vs-frozen gap is SPECTRE's own analogue of the B3−B4 adaptive premium.

Both variants share the checkpoint and the deterministic rollout, so the
comparison is perfectly paired per episode. At attempt 1 the failure set is
empty and the full variant also uses ``c_0`` — the two variants always agree
on the first pick, and divergence can only begin at attempt 2.

Reported per checkpoint:

- mean ± std attempts to first success, censoring rate, mean wall-clock;
- paired bootstrap CI on the frozen−full mean-attempts difference and
  per-episode win/tie/loss counts;
- same-choice agreement rate per attempt index (with co-running counts)
  and the first-divergence-index histogram;
- success-at-K curves for both variants.

Artifacts (JSON + CSV) land under ``<data_root>/derived/ablation_context/``
(gitignored); headline numbers go into ``docs/notebook.md`` by hand. The
frozen-SPECTRE row for the B1–B5 comparison table can be rebuilt in
``analyze_spectre.py`` from the dumped per-episode arrays.

Usage::

    python experiments/spectre/spectre_ablate_context.py \
        env=routedtransport2d_n3_v1

    # explicit checkpoint list (e.g. a future multi-seed rerun):
    python experiments/spectre/spectre_ablate_context.py \
        env=routedtransport2d_n3_v1 \
        '+ckpts=[data/spectre/checkpoints/<run>/<env>/seed_0/best.pt]'
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from alphatamp.approaches.spectre import eda, inference
from alphatamp.approaches.spectre.env_registry import get_static_tag_predicates
from alphatamp.approaches.spectre.vocab import Vocab

# Default evaluation attempt budget — the documented protocol (proposal.md,
# ``attempt_budget + 1 == 21`` censoring). Override with ``+attempt_budget=30``
# to match ``analyze_spectre.py``'s headline table, which was generated
# with ATTEMPT_BUDGET = 30 (≥ the pool cap, so it never binds: censoring 0).
DEFAULT_ATTEMPT_BUDGET = 20

# The checkpoint behind the headline notebook results. Override with
# ``+ckpts=[...]`` for multi-checkpoint / multi-seed runs.
DEFAULT_CKPTS = [
    "data/spectre/checkpoints/r3_visit_rate/routedtransport2d_n3_v1/seed_0/best.pt",
]


def _ckpt_label(ckpt_path: Path) -> str:
    """Compact run label, e.g. ``r3_visit_rate_seed_0``."""
    # .../checkpoints/<run>/<env_variant>/<seed_i>/best.pt
    parts = ckpt_path.parts
    try:
        anchor = parts.index("checkpoints")
        run, _env, seed = parts[anchor + 1 : anchor + 4]
        return f"{run}_{seed}"
    except (ValueError, IndexError):
        return ckpt_path.parent.name


def _summary_row(result: eda.BaselineResult) -> dict[str, float]:
    return {
        "mean_attempts": float(result.attempts.mean()),
        "std_attempts": float(result.attempts.std()),
        "censoring_rate": float(result.censored.mean()),
        "mean_wall_clock_s": float(result.wall_clock.mean()),
        "n_episodes": float(len(result.attempts)),
    }


def _evaluate_checkpoint(
    ckpt_path: Path,
    test: eda.LoadedSplit,
    vocab: Vocab,
    env_variant: str,
    device: torch.device,
    attempt_budget: int,
) -> dict[str, Any]:
    """Run full + frozen variants on one checkpoint; return all metrics."""
    model = inference.load_checkpoint(
        ckpt_path,
        vocab,
        device=device,
        fallback_static_tag_predicates=get_static_tag_predicates(env_variant),
    )
    prior = inference.load_prior_for_checkpoint(ckpt_path)

    full_res, full_traces = eda.spectre_evaluate_traced(
        test,
        model,
        vocab,
        attempt_budget=attempt_budget,
        prior=prior,
        device=device,
        name="SPECTRE",
        freeze_context=False,
    )
    frozen_res, frozen_traces = eda.spectre_evaluate_traced(
        test,
        model,
        vocab,
        attempt_budget=attempt_budget,
        prior=prior,
        device=device,
        name="SPECTRE-frozen-context",
        freeze_context=True,
    )

    # Paired stats: frozen − full, so a positive Δ = attempts the context
    # encoder saves.
    delta = eda.bootstrap_mean_difference(frozen_res.attempts, full_res.attempts)
    wins, ties, losses = eda.win_tie_loss(full_res, frozen_res)
    agreement = eda.per_index_agreement(
        full_traces, frozen_traces, max_index=attempt_budget
    )
    divergence = eda.first_divergence_distribution(full_traces, frozen_traces)

    return {
        "ckpt": str(ckpt_path),
        "label": _ckpt_label(ckpt_path),
        "attempt_budget": attempt_budget,
        "full": _summary_row(full_res),
        "frozen": _summary_row(frozen_res),
        "paired_delta_frozen_minus_full": {
            "point": delta.point,
            "ci_low": delta.ci_low,
            "ci_high": delta.ci_high,
        },
        "win_tie_loss_full_vs_frozen": {"wins": wins, "ties": ties, "losses": losses},
        "agreement_by_index": [
            {"t": t, "agreement": rate, "n_co_running": n_co}
            for t, rate, n_co in agreement
        ],
        "first_divergence_hist": {
            str(k): v
            for k, v in sorted(
                divergence.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])
            )
        },
        "success_at_k": {
            "full": eda.success_at_k(full_res, k_max=attempt_budget).tolist(),
            "frozen": eda.success_at_k(frozen_res, k_max=attempt_budget).tolist(),
        },
        # Per-episode arrays so the notebook can drop the frozen variant
        # into the B1–B5 comparison table without re-running rollouts.
        "per_episode": {
            "problem_ids": full_res.problem_ids.tolist(),
            "full_attempts": full_res.attempts.tolist(),
            "frozen_attempts": frozen_res.attempts.tolist(),
            "full_wall_clock_s": full_res.wall_clock.tolist(),
            "frozen_wall_clock_s": frozen_res.wall_clock.tolist(),
            "full_censored": full_res.censored.tolist(),
            "frozen_censored": frozen_res.censored.tolist(),
        },
    }


def _print_report(rep: dict[str, Any]) -> None:
    full, frozen = rep["full"], rep["frozen"]
    delta = rep["paired_delta_frozen_minus_full"]
    wtl = rep["win_tie_loss_full_vs_frozen"]
    print(f"\n=== {rep['label']} ({rep['ckpt']}) ===")
    print(f"{'variant':<24s} {'mean':>7s} {'std':>7s} {'cens.':>7s} {'wall_s':>8s}")
    for label, row in (("SPECTRE", full), ("SPECTRE-frozen-context", frozen)):
        print(
            f"{label:<24s} {row['mean_attempts']:>7.3f} {row['std_attempts']:>7.3f}"
            f" {row['censoring_rate']:>7.3f} {row['mean_wall_clock_s']:>8.2f}"
        )
    print(
        f"paired Δ attempts (frozen − full): {delta['point']:+.3f}"
        f"  [95% CI {delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}]"
        f"  (positive = context encoder saves attempts)"
    )
    print(
        f"per-episode full-vs-frozen: {wtl['wins']} wins"
        f" / {wtl['ties']} ties / {wtl['losses']} losses"
    )
    print("agreement by attempt index (t=1 is 1.0 by construction — both use c_0):")
    print(f"{'t':>4s} {'agree':>7s} {'n_co':>5s}")
    for row in rep["agreement_by_index"]:
        if row["n_co_running"] == 0:
            continue
        print(f"{row['t']:>4d} {row['agreement']:>7.3f} {row['n_co_running']:>5d}")
    print(f"first-divergence histogram: {rep['first_divergence_hist']}")
    budget = int(rep["attempt_budget"])
    k_marks = [1, 3, 5, 9, 15, budget]
    s_full = rep["success_at_k"]["full"]
    s_frozen = rep["success_at_k"]["frozen"]
    print(f"{'success@K':<12s}" + "".join(f" K={k:<3d}" for k in k_marks))
    print(f"{'  full':<12s}" + "".join(f" {s_full[k - 1]:.2f} " for k in k_marks))
    print(f"{'  frozen':<12s}" + "".join(f" {s_frozen[k - 1]:.2f} " for k in k_marks))


def _dump_artifacts(reports: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rep in reports:
        json_path = out_dir / f"{rep['label']}_budget{rep['attempt_budget']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"wrote {json_path}")
    budget = reports[0]["attempt_budget"]
    csv_path = out_dir / f"summary_budget{budget}.csv"
    fields = [
        "label",
        "attempt_budget",
        "variant",
        "mean_attempts",
        "std_attempts",
        "censoring_rate",
        "mean_wall_clock_s",
        "n_episodes",
        "paired_delta_point",
        "paired_delta_ci_low",
        "paired_delta_ci_high",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rep in reports:
            delta = rep["paired_delta_frozen_minus_full"]
            for variant in ("full", "frozen"):
                writer.writerow(
                    {
                        "label": rep["label"],
                        "attempt_budget": rep["attempt_budget"],
                        "variant": variant,
                        **rep[variant],
                        "paired_delta_point": delta["point"],
                        "paired_delta_ci_low": delta["ci_low"],
                        "paired_delta_ci_high": delta["ci_high"],
                    }
                )
    print(f"wrote {csv_path}")


@hydra.main(
    config_path="conf",
    config_name="spectre_train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Ablation entrypoint — re-uses spectre_train.yaml for paths."""
    data_root = Path(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    test_dir = data_root / "raw" / env_variant / "test"
    vocab_path = data_root / "derived" / env_variant / "train_vocab.json"
    ckpts = [Path(p) for p in cfg.get("ckpts", DEFAULT_CKPTS)]
    attempt_budget = int(cfg.get("attempt_budget", DEFAULT_ATTEMPT_BUDGET))

    for ckpt in ckpts:
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}.")
    if not vocab_path.exists():
        raise FileNotFoundError(f"No vocab at {vocab_path}.")
    if not (test_dir / "episodes").exists():
        raise FileNotFoundError(f"No test episodes at {test_dir}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.from_json(vocab_path)
    test = eda.load_split_episodes(test_dir)

    print(f"env_variant={env_variant} device={device}")
    print(f"  test episodes: {len(test.episodes)}  attempt_budget={attempt_budget}")
    print(f"  checkpoints: {[str(c) for c in ckpts]}")

    reports = []
    for ckpt in ckpts:
        rep = _evaluate_checkpoint(
            ckpt, test, vocab, env_variant, device, attempt_budget
        )
        _print_report(rep)
        reports.append(rep)

    if len(reports) > 1:
        print("\n=== aggregate across checkpoints (mean ± std of per-ckpt means) ===")
        for variant in ("full", "frozen"):
            means = np.array([r[variant]["mean_attempts"] for r in reports])
            print(
                f"{variant:<8s} mean attempts: {means.mean():.3f} ± {means.std():.3f}"
                f"  (n={len(means)} checkpoints)"
            )

    _dump_artifacts(reports, data_root / "derived" / "ablation_context")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
