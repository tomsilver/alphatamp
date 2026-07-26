"""Generate VLMPlan proposal sequences for DD2D problems (stage 1 of 2).

This is the only stage that touches a model. It writes, per problem, the ordered list of
plans the model proposed plus a per-round transcript; scoring is a separate entry point
(``vlmplan_score.py``) so that relabelling after a re-collection — or any change to the
metric — never re-queries the model.

Local dev arm (LM Studio or vLLM, both OpenAI-compatible)::

    lms server start
    export OPENAI_BASE_URL=http://localhost:1234/v1 OPENAI_API_KEY=lm-studio
    python experiments/spectre/vlmplan_run.py env=dd2d_v3 split=train n_problems=5

Switching to a frontier API arm is a ``model`` override, not a code change::

    python experiments/spectre/vlmplan_run.py model.model_name=gpt-5 \\
        model.base_url=null

Output (per run)::

    <data_root>/derived/<env_variant>/vlmplan/<run>/sequences/<problem_id>.json
    <data_root>/derived/<env_variant>/vlmplan/<run>/transcripts/<problem_id>.jsonl
    <data_root>/derived/<env_variant>/vlmplan/<run>/run_config.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from alphatamp.approaches.spectre.vlmplan import runio
from alphatamp.approaches.spectre.vlmplan.dd2d_adapter import DD2DAdapter
from alphatamp.approaches.spectre.vlmplan.loop import LoopConfig, generate_sequence
from alphatamp.approaches.spectre.vlmplan.models import ModelConfig, make_model
from alphatamp.approaches.spectre.vlmplan.template import PromptConfig, build_prompt

REPO = Path(__file__).resolve().parents[2]


def model_config(cfg: DictConfig) -> ModelConfig:
    """``cfg.model`` as a :class:`ModelConfig`, with the cache path made absolute.

    Hydra runs with ``chdir: false``, but the cache path is written relative to the repo
    root in the config, so resolve it here rather than depending on the launch
    directory.
    """
    raw = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(raw, dict)
    kwargs: dict[str, Any] = dict(raw)  # type: ignore[arg-type]
    cache_path = kwargs.get("cache_path")
    if cache_path and not Path(str(cache_path)).is_absolute():
        kwargs["cache_path"] = str(REPO / str(cache_path))
    return ModelConfig(**kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="vlmplan")
def main(cfg: DictConfig) -> None:
    """Generate and persist one VLMPlan proposal sequence per selected problem."""
    data_root = REPO / str(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    out_root = runio.run_dir(data_root, env_variant, str(cfg.run))
    seq_dir = out_root / "sequences"
    transcript_dir = out_root / "transcripts"
    seq_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    mc = model_config(cfg)
    model = make_model(mc)
    adapter = DD2DAdapter(
        with_images=bool(cfg.with_images), image_width_px=int(cfg.image_width_px)
    )
    _loop = OmegaConf.to_container(cfg.loop, resolve=True)
    loop_cfg = LoopConfig(**_loop)  # type: ignore[arg-type]
    decode = dict(mc.decode)
    prompt_cfg = PromptConfig(plans_per_round=loop_cfg.plans_per_round)

    runio.write_json(
        out_root / "run_config.json",
        {
            "env_variant": env_variant,
            "split": str(cfg.split),
            "seed": int(cfg.seed),
            "with_images": bool(cfg.with_images),
            "image_width_px": int(cfg.image_width_px),
            "loop": OmegaConf.to_container(cfg.loop, resolve=True),
            "model": mc.describe(),
        },
    )

    episodes = runio.select_episodes(
        runio.split_dir(data_root, env_variant, str(cfg.split)),
        n_problems=int(cfg.n_problems),
        problem_ids=[int(p) for p in (cfg.problem_ids or [])],
    )
    print(
        f"VLMPlan generate: {len(episodes)} problems from {env_variant}/{cfg.split}  "
        f"model={mc.model_name}  images={bool(cfg.with_images)}  run={cfg.run}"
    )

    n_written = n_skipped = 0
    for episode in episodes:
        pid = int(episode.provenance.problem_id)
        out_path = seq_dir / f"{pid}.json"
        if out_path.exists() and not bool(cfg.overwrite):
            n_skipped += 1
            continue

        started = time.time()
        result = generate_sequence(
            adapter,
            episode,
            pid,
            model,
            loop_cfg,
            decode,
            base_seed=int(cfg.seed) * 1000,
        )
        elapsed = time.time() - started

        payload = result.as_dict()
        payload["plans"] = [adapter.plan_str(p.steps) for p in result.proposals]
        payload["elapsed_s"] = elapsed
        runio.write_json(out_path, payload)

        # The transcript is the released reproducibility artifact: each round's exact
        # prompt alongside its accounting. Completions live in the prpl_llm_utils
        # response cache, keyed by (model, prompt, images, hyperparameters).
        with open(transcript_dir / f"{pid}.jsonl", "w", encoding="utf-8") as handle:
            for round_log in result.rounds:
                prompt = build_prompt(
                    controllers=adapter.controllers_str(episode),
                    typed_objects=adapter.typed_objects_str(episode),
                    type_hierarchy=adapter.type_hierarchy_str(episode),
                    goal_str=adapter.goal_str(episode),
                    init_state_str=adapter.init_state_str(episode),
                    config=prompt_cfg,
                    previous_plans=[
                        adapter.plan_str(p.steps)
                        for p in result.proposals
                        if p.round_index < round_log.round_index
                    ],
                )
                handle.write(
                    json.dumps(
                        {
                            "problem_id": pid,
                            "round": round_log.round_index,
                            "prompt": prompt,
                            "n_new": round_log.n_new,
                            "n_malformed": round_log.n_malformed,
                            "n_invalid": round_log.n_invalid,
                            "n_duplicate": round_log.n_duplicate,
                            "error": round_log.error,
                        }
                    )
                    + "\n"
                )

        n_written += 1
        print(
            f"  pid {pid}: {len(result.proposals)} plans in {len(result.rounds)} rounds "
            f"({elapsed:.1f}s){'  [stalled]' if result.stalled else ''}"
        )

    print(f"wrote {n_written}, skipped {n_skipped} -> {seq_dir}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
