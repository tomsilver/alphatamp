"""Collect training datasets for SimFreeParamPolicyApproach over many seeds.

Uses Hydra for configuration. Each invocation collects data for a single seed.
Use Hydra multirun to sweep over seeds:

    python experiments/collect_data.py seed=0,1,2,3,4 -m
    python experiments/collect_data.py 'seed=range(0,100)' -m
    python experiments/collect_data.py 'seed=range(0,100)' hydra/launcher=joblib -m
    python experiments/collect_data.py 'seed=range(0,100)' hydra/launcher=slurm -m
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import kinder
import numpy as np
from gymnasium.wrappers import RecordVideo
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.filter_feasibility_classifier import (
    FilterFeasibilityClassifier,
)
from alphatamp.approaches.scorers.classifier_parameter_scorer import (
    ClassifierParameterScorer,
)
from alphatamp.approaches.scorers.regressor_abstract_action_scorer import (
    AbstractActionScorer,
)
from alphatamp.approaches.simfree_param_policy_approach import (
    SimFreeParamPolicyApproach,
)
from alphatamp.approaches.simulator_free_base_approach import sesame_models_to_sim_free
from alphatamp.approaches.utils.approach_step_error import ApproachStepError


class AbstractOverlayWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Gymnasium wrapper that overlays the current abstract action and plan on rendered
    frames."""

    def __init__(self, env: gym.Env) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        self._current_action_label: str = ""
        self._current_plan_label: str = ""
        self._current_param_label: str = ""

    def set_action_label(self, label: str) -> None:
        """Update the abstract action label drawn on subsequent video frames."""
        self._current_action_label = "Action: " + label

    def set_plan_label(self, label: str) -> None:
        """Update the abstract plan label drawn on subsequent video frames."""
        self._current_plan_label = "Plan: " + label

    def set_param_label(self, label: str) -> None:
        """Update the parameter label drawn on subsequent video frames."""
        self._current_param_label = "Params: " + label

    def render(self) -> Any:
        frame: list | None = self.env.render()
        if frame is None:
            return frame
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=16)
        lines = [
            (self._current_action_label, (255, 255, 0)),
            (self._current_plan_label, (200, 200, 200)),
            (self._current_param_label, (100, 220, 255)),
        ]
        y = 8.0
        for text, color in lines:
            if not text:
                continue
            bbox = draw.textbbox((8, y), text, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((8, y), text, fill=color, font=font)
            y = bbox[3] + 6.0
        return np.array(img)


@hydra.main(config_path="conf", config_name="collect_data_config", version_base=None)
def main(cfg: DictConfig):
    """Collect training data for a single seed."""

    seed = int(cfg.seed)
    num_steps = int(cfg.num_steps)
    max_resamples = int(cfg.max_resamples)
    reset_every = int(cfg.reset_every)

    # Build env.
    kinder.register_all_environments()
    env = kinder.make(cfg.env.id, render_mode="rgb_array" if cfg.record_video else None)

    overlay_wrapper: AbstractOverlayWrapper | None = None
    if cfg.record_video:
        overlay_wrapper = AbstractOverlayWrapper(env)
        env = overlay_wrapper
        video_dir = Path(cfg.output_dir) / f"seed_{seed}" / "videos"
        env = RecordVideo(env, str(video_dir), name_prefix=f"seed_{seed}")

    obs, _ = env.reset(seed=seed)

    # Build env models and convert to simulator-free.
    env_models = create_bilevel_planning_models(
        cfg.env.model_name,
        env.observation_space,
        env.action_space,
        **cfg.env.model_kwargs,
    )
    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Feasibility classifier.
    filter_classifier = FilterFeasibilityClassifier()
    feasibility_classifier_learner = StaticFeasibilityClassifierLearner(
        filter_classifier
    )

    # Explorer.
    train_explorer = ExploitExplorer(
        sim_free_env_models, feasibility_classifier_learner, seed
    )

    # Scorer configs from Hydra.
    parameter_configs = {
        "hidden_layer_sizes": tuple(cfg.parameter_scorer.hidden_layer_sizes)
    }
    abstract_action_configs = {
        "hidden_dim": int(cfg.abstract_action_scorer.hidden_dim),
        "num_layers": int(cfg.abstract_action_scorer.num_layers),
        "num_epochs": int(cfg.abstract_action_scorer.num_epochs),
    }
    q_network_configs = {
        "hidden_dim": int(cfg.q_network.hidden_dim),
        "num_layers": int(cfg.q_network.num_layers),
        "num_epochs": int(cfg.q_network.num_epochs),
        "num_ensemble_nets": int(cfg.q_network.num_ensemble_nets),
    }

    # Build approach.
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=feasibility_classifier_learner,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs={"configs": parameter_configs},
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
        q_network_configs=q_network_configs,
        max_resamples=max_resamples,
        train_every=int(cfg.train_every),
        seed=seed,
    )

    # Train: step through the environment and collect data.
    approach.train()
    approach.reset(obs, {})

    task_completed = False
    reset_count = 0

    def _env_reset(episode: int) -> Any:
        new_obs, _ = env.reset(seed=seed + episode)
        return new_obs

    for step in range(num_steps):
        try:
            action = approach.step()
        except ApproachStepError:
            # Stuck in a terminal state — reset the environment but keep
            # all learned models and datasets so the approach can continue improving.
            reset_count += 1
            print(f"Step {step}: ApproachStepError, resetting env (reset #{reset_count})")
            obs = _env_reset(reset_count)
            approach.reset_episode(obs)
            continue

        if overlay_wrapper is not None:
            overlay_wrapper.set_action_label(
                approach.get_most_recent_abstract_action_str() or ""
            )
            params = approach.get_most_recent_parameter()
            if params is not None:
                param_arr = np.asarray(params).ravel()
                overlay_wrapper.set_param_label(
                    "[" + ", ".join(f"{v:.3f}" for v in param_arr) + "]"
                )
            else:
                overlay_wrapper.set_param_label("")
            plan = approach.get_abstract_plan()
            if plan is not None:
                plan_step = approach.get_current_abstract_plan_step()
                parts = [
                    f"[{a.short_str}]" if i == plan_step else a.short_str
                    for i, a in enumerate(plan[1])
                ]
                overlay_wrapper.set_plan_label(" \u2192 ".join(parts))
            else:
                overlay_wrapper.set_plan_label("")

        obs, reward, done, _, _ = env.step(action)
        approach.update(obs, float(reward), done, {})

        print(f"Executing step: {step}")
        if done:
            task_completed = True
            reset_count += 1
            print(f"Step {step}: task completed, resetting env (reset #{reset_count})")
            obs = _env_reset(reset_count)
            approach.reset_episode(obs)
        elif (step + 1) % reset_every == 0:
            reset_count += 1
            print(f"Step {step}: periodic reset (reset #{reset_count})")
            obs = _env_reset(reset_count)
            approach.reset_episode(obs)

    # Save datasets to a per-seed directory.
    output_dir = Path(cfg.output_dir) / f"seed_{seed}"
    approach.save_datasets(output_dir)

    # Report.
    parameter_dataset = approach.get_parameter_dataset()
    abstract_plan_dataset = approach.get_abstract_plan_dataset()
    abstract_action_dataset = approach.get_abstract_action_dataset()

    print(f"Seed {seed}: completed={task_completed}")
    print(f"# Parameters: {sum(len(v) for v in parameter_dataset.values())}")
    print(f"# Abstract plan: {len(abstract_plan_dataset)}")
    print(
        f"# Abstract Actions: {sum(len(v) for v in abstract_action_dataset.values())}"
    )
    print(f"  Saved to: {output_dir}")

    env.close()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
