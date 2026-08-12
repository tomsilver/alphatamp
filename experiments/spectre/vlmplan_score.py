"""Score VLMPlan proposal sequences into a comparison-cache row (stage 2 of 2).

Reads what ``vlmplan_run.py`` generated and turns each proposal sequence into the shared
rollout-FP metric, writing one record per problem in the shape
``compare_methods.py`` reads. No model is involved, so this is cheap to re-run —
which is the point of splitting it out::

    python experiments/spectre/vlmplan_score.py env=dd2d_v3 split=train

In-pool proposals take their label from the stored ``OutcomeRecord``; off-pool proposals
(most three-item stagings — the pool holds only a few percent of those orderings) are
refined live on the scene reconstructed from stored geometry, and cost an attempt like
any other. See ``vlmplan/score.py`` for why that is the honest accounting.

``check_label_agreement`` runs the consistency gate first: re-label stored pool plans
live and compare. Well below 1.0 means the env code has moved since the collection, so
in-pool and off-pool attempts are on different label functions and the numbers must not
be reported.

Output::

    <data_root>/derived/<env_variant>/compare_cache/<cache_subdir>/seed_<seed>/<pid>.json
    <data_root>/derived/<env_variant>/vlmplan/<run>/offpool_labels.json   (memo)
    <data_root>/derived/<env_variant>/vlmplan/<run>/score_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from alphatamp.approaches.spectre.baselines.vlmplan import runio
from alphatamp.approaches.spectre.baselines.vlmplan import score as score_mod
from alphatamp.approaches.spectre.baselines.vlmplan.registry import (
    make_adapter,
    make_labeler_factory,
)
from alphatamp.approaches.spectre.compare import stratum_of

REPO = Path(__file__).resolve().parents[2]

# Below this, stored and live labels disagree enough that mixing them is not defensible.
AGREEMENT_WARN_BELOW = 0.95


@hydra.main(version_base=None, config_path="conf", config_name="vlmplan")
def main(cfg: DictConfig) -> None:
    """Score every generated sequence into the comparison cache."""
    data_root = REPO / str(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    out_root = runio.run_dir(data_root, env_variant, str(cfg.run))
    seq_dir = out_root / "sequences"
    if not seq_dir.is_dir():
        raise FileNotFoundError(
            f"No generated sequences at {seq_dir}. Run vlmplan_run.py first."
        )
    out_dir = runio.compare_cache_dir(
        data_root, env_variant, str(cfg.cache_subdir), int(cfg.seed)
    )
    # One cache dir == one method row, so it must hold exactly one run's records.
    score_mod.assert_single_run(out_dir, str(cfg.run))

    episodes = runio.select_episodes(
        runio.split_dir(data_root, env_variant, str(cfg.split)),
        n_problems=int(cfg.n_problems),
        problem_ids=[int(p) for p in (cfg.problem_ids or [])],
        stratified_per_stratum=int(cfg.get("stratified_per_stratum", 0)),
        stratum_of=stratum_of,
    )
    adapter = make_adapter(env_variant, with_images=False)  # no render needed

    summary: dict[str, Any] = {
        "env_variant": env_variant,
        "split": str(cfg.split),
        "run": str(cfg.run),
        "seed": int(cfg.seed),
        "attempt_budget": int(cfg.attempt_budget),
    }

    if bool(cfg.check_label_agreement):
        gate = score_mod.label_agreement(
            episodes[: int(cfg.label_agreement_episodes)],
            adapter,
            samples_per_episode=int(cfg.label_agreement_samples),
            seed=int(cfg.seed),
            env_variant=env_variant,
            # A fresh labeler per episode, and deliberately no memo path: a gate that
            # read its answers out of the run's memo would be checking the memo, not
            # the environment.
            make_labeler=make_labeler_factory(env_variant),
        )
        summary["label_agreement"] = gate
        print(f"label-agreement gate ({env_variant}): {gate}")
        agreement = gate["agreement"]
        if agreement is not None and agreement < AGREEMENT_WARN_BELOW:
            print(
                "  [WARN] stored and live labels disagree by more than "
                f"{(1 - AGREEMENT_WARN_BELOW) * 100:.0f}%. The env code has probably "
                "moved since this collection, so in-pool and off-pool attempts are on "
                "different label functions. Treat results as plumbing only."
            )

    labeler = make_labeler_factory(
        env_variant, memo_path=out_root / "offpool_labels.json"
    )()

    run_config_path = out_root / "run_config.json"
    run_config: dict[str, Any] = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file()
        else {}
    )

    rows: list[dict[str, Any]] = []
    n_written = n_missing = 0
    for episode in episodes:
        pid = int(episode.provenance.problem_id)
        seq_path = seq_dir / f"{pid}.json"
        if not seq_path.is_file():
            n_missing += 1
            continue
        result = score_mod.score_sequence(
            episode,
            runio.load_proposals(seq_path),
            adapter,
            stratum=stratum_of(pid),
            labeler=labeler,
            attempt_budget=int(cfg.attempt_budget),
            fill_from_published=bool(cfg.fill_from_published),
            env_variant=env_variant,
            refine_cap_s=float(cfg.get("refine_cap_s", 2.0)),
        )
        # VLM generation wall-clock (the §2b "inference" component) comes from the
        # sequences file's per-round api_s, not from anything the scorer re-times.
        result.infer_s = runio.load_infer_seconds(seq_path)
        score_mod.write_record(
            out_dir,
            result,
            extra={
                "run": str(cfg.run),
                "split": str(cfg.split),
                "model": run_config.get("model", {}),
                "with_images": run_config.get("with_images"),
                "loop": run_config.get("loop", {}),
                # Generation-side quality, copied onto the row so the notebook reports FP
                # and how the plans were produced from one place.
                **runio.load_generation_stats(seq_path),
            },
            force=bool(cfg.overwrite),
        )
        n_written += 1
        rows.append(result.as_dict())
        print(
            f"  pid {pid} s{stratum_of(pid)}: fp={result.fp:g} "
            f"attempts={len(result.attempts)} offpool={result.n_offpool} "
            f"fill={result.n_fill_used} first={result.first_success_source}"
            f"{'  [censored]' if result.censored else ''}"
        )

    labeler.flush()

    if rows:
        n = len(rows)
        by_stratum: dict[str, list[float]] = {}
        for row in rows:
            by_stratum.setdefault(f"s{row['stratum']}", []).append(float(row["fp"]))
        summary.update(
            {
                "n_problems": n,
                "mean_fp": sum(r["fp"] for r in rows) / n,
                "mean_offpool": sum(r["n_offpool"] for r in rows) / n,
                "n_censored": sum(1 for r in rows if r["censored"]),
                # The headline smoke question: did the MODEL find the feasible plan, or
                # did the published-order fill find it after the model ran dry?
                "first_success_from_vlm": sum(
                    1 for r in rows if r["first_success_source"] == "vlm"
                ),
                "first_success_from_fill": sum(
                    1 for r in rows if r["first_success_source"] == "fill"
                ),
                "total_live_refines": labeler.n_refines,
                "mean_infer_s": sum(r.get("infer_s", 0.0) for r in rows) / n,
                "mean_refine_s": sum(r.get("refine_s", 0.0) for r in rows) / n,
                "mean_refine_s_capped": sum(r.get("refine_s_capped", 0.0) for r in rows)
                / n,
                "mean_fp_by_stratum": {
                    k: sum(v) / len(v) for k, v in sorted(by_stratum.items())
                },
            }
        )
        runio.write_json(out_root / "score_summary.json", summary)
        print(json.dumps(summary, indent=2))

    tail = f"  (missing {n_missing})" if n_missing else ""
    print(f"wrote {n_written} records -> {out_dir}{tail}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
