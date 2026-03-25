"""Kinder environment test for SimFreeParamPolicyApproach.

Runs the approach on a configurable kinder environment and logs
the same metrics as bandit_test.py: rolling success rate, overall success
rate, total successes, parameter resamples, and resample exhaustions.

Usage::

    python experiments/kinder_env_test.py
    python experiments/kinder_env_test.py --env obstruction2d --complexity 5 --num-steps 4000
    python experiments/kinder_env_test.py --plot
        --save experiments/slurm_outputs/kinder_env_test.png
"""

from __future__ import annotations

import argparse
import logging
from collections import deque

import kinder
import matplotlib.pyplot as plt
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.filter_feasibility_classifier import (
    FilterFeasibilityClassifier,
)
from alphatamp.approaches.scorers.abstract_action_scorers.regressor_abstract_action_scorer import (  # pylint:disable=line-too-long
    AbstractActionScorer,
)
from alphatamp.approaches.scorers.parameter_scorers.classifier_parameter_scorer import (
    ClassifierParameterScorer,
)
from alphatamp.approaches.simfree_param_policy_approach import (
    SimFreeParamPolicyApproach,
)
from alphatamp.approaches.simulator_free_base_approach import (
    sesame_models_to_sim_free,
)
from alphatamp.approaches.utils.approach_step_error import ApproachStepError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Registry of supported kinder environments.
# Each entry maps a short name → (env_id_template, model_name, complexity_kwarg).
# The env_id_template uses {n} as a placeholder for the complexity integer.
_ENV_REGISTRY: dict[str, tuple[str, str, str]] = {
    "clutteredretrieval2d": (
        "kinder/ClutteredRetrieval2D-o{n}-v0",
        "clutteredretrieval2d",
        "num_obstructions",
    ),
    "obstruction2d": (
        "kinder/Obstruction2D-o{n}-v0",
        "obstruction2d",
        "num_obstructions",
    ),
    "dynobstruction2d": (
        "kinder/DynObstruction2D-o{n}-v0",
        "dynobstruction2d",
        "num_obstructions",
    ),
    "clutteredstorage2d": (
        "kinder/ClutteredStorage2D-b{n}-v0",
        "clutteredstorage2d",
        "num_boxes",
    ),
}


def _get_param_scorer_loss_curves(approach: SimFreeParamPolicyApproach) -> list[float]:
    """Extract per-fit loss curve from any MLPClassifier-backed parameter scorers."""
    curves: list[float] = []
    for (
        scorer_fn
    ) in (
        approach._abstract_action_to_scoring_function.values()  # pylint: disable=protected-access
    ):
        clf = getattr(scorer_fn, "_classifier", None)
        if clf is not None and hasattr(clf, "loss_curve_"):
            curves.extend(clf.loss_curve_)
    return curves


def _get_q_network_loss_curves(approach: SimFreeParamPolicyApproach) -> list[float]:
    """Extract per-epoch Q-network training losses accumulated so far."""
    return list(approach.get_q_network_loss_metrics())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    num_steps: int = 4000,
    max_resamples: int = 20,
    reset_every: int = 30,
    log_every: int = 100,
    seed: int = 0,
    env: str = "clutteredretrieval2d",
    complexity: int = 1,
    use_abstract_plan_scorer: bool = True,
    use_parameter_scorer: bool = True,
) -> dict:
    """Run an experiment on a kinder environment and return collected metrics.

    Args:
        env: Short environment name from _ENV_REGISTRY (e.g. "clutteredretrieval2d").
        complexity: The complexity integer passed to the env (e.g. num_obstructions).

    Returns a dict with keys:
        steps                 — list of step indices at each log point
        success_rates         — rolling success rate at each log point
        overall_success_rates — total_successes / total_episodes at each log point
        total_successes       — cumulative successes at each log point
        episode_counts        — cumulative episode count at each log point
        exhaustion_counts     — cumulative resample exhaustion count at each log point
        param_loss            — per-fit sklearn loss values
        q_loss                — per-epoch Q-network loss values
    """

    env_id_template, model_name, complexity_kwarg = _ENV_REGISTRY[env]
    print(f"env={env}  complexity={complexity}  ({complexity_kwarg}={complexity})")

    # Build env
    kinder.register_all_environments()
    env_id = env_id_template.format(n=complexity)
    gym_env = kinder.make(env_id, render_mode="rgb_array")
    obs, _ = gym_env.reset(seed=seed)

    # Build models
    env_models = create_bilevel_planning_models(
        model_name,
        gym_env.observation_space,
        gym_env.action_space,
        **{complexity_kwarg: complexity},
    )
    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Feasibility: allow all plans
    filter_clf = FilterFeasibilityClassifier()
    feasibility_learner = StaticFeasibilityClassifierLearner(filter_clf)

    # Explorer
    train_explorer = ExploitExplorer(sim_free_env_models, feasibility_learner, seed)

    parameter_scorer_configs: dict = {"configs": {"hidden_layer_sizes": (32, 32)}}
    abstract_action_configs = {"hidden_dim": 32, "num_layers": 2, "num_epochs": 50}
    q_network_configs = {
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 500,
        "num_ensemble_nets": 3,
    }

    # Approach
    approach = SimFreeParamPolicyApproach(
        env_models=sim_free_env_models,
        feasibility_classifier_learner=feasibility_learner,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs=parameter_scorer_configs,
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
        q_network_configs=q_network_configs,
        max_resamples=max_resamples,
        train_every=1,
        param_sample_count=100,
        seed=seed,
        use_abstract_plan_scorer=use_abstract_plan_scorer,
        use_parameter_scorer=use_parameter_scorer,
    )

    approach.train()
    approach.reset(obs, {})

    # Tracking
    recent: deque[int] = deque(maxlen=20)
    total_successes = 0
    total_episodes = 0
    reset_count = 0
    episode_success = False

    log_steps: list[int] = []
    success_rates: list[float] = []
    overall_success_rates: list[float] = []
    episode_counts: list[int] = []
    cumulative_successes: list[int] = []
    exhaustion_counts: list[int] = []
    param_loss: list[float] = []
    q_loss: list[float] = []

    header = (
        f"{'Step':>6}  {'Rolling success':>15}  {'Overall success':>15}  "
        f"{'Total successes':>16}  {'Resamples':>10}  {'Exhaustions':>12}"
    )
    print(header)
    print("-" * 82)

    for step in range(num_steps):
        logging.info("CLUTTERED RETRIEVAL STEP: %d", step)

        try:
            action = approach.step()
        except ApproachStepError:
            recent.append(0)
            episode_success = False
            total_episodes += 1
            reset_count += 1
            obs, _ = gym_env.reset(seed=seed + reset_count)
            approach.reset_episode(obs)
            continue

        obs, reward, done, _, _ = gym_env.step(action)
        approach.update(obs, float(reward), done, {})

        if done:
            episode_success = True

        if done or (step + 1) % reset_every == 0:
            recent.append(int(episode_success))
            total_successes += int(episode_success)
            total_episodes += 1
            episode_success = False
            reset_count += 1
            obs, _ = gym_env.reset(seed=seed + reset_count)
            approach.reset_episode(obs)

        if (step + 1) % log_every == 0:
            rolling_rate = sum(recent) / len(recent) if recent else 0.0
            overall_rate = total_successes / total_episodes if total_episodes else 0.0
            param_ds = approach.get_parameter_dataset()
            total_data = sum(len(v) for v in param_ds.values())
            exhaustion_count = approach.get_resample_exhaustion_count()
            print(
                f"{step+1:>6}  {rolling_rate:>15.2%}  {overall_rate:>15.2%}  "
                f"{total_successes:>16}  {total_data:>10}  {exhaustion_count:>12}"
            )
            log_steps.append(step + 1)
            success_rates.append(rolling_rate)
            overall_success_rates.append(overall_rate)
            episode_counts.append(total_episodes)
            cumulative_successes.append(total_successes)
            exhaustion_counts.append(exhaustion_count)
            param_loss.extend(_get_param_scorer_loss_curves(approach))
            q_loss = _get_q_network_loss_curves(approach)

    print("\nDone.")
    param_ds = approach.get_parameter_dataset()
    print(f"  Total parameter samples: {sum(len(v) for v in param_ds.values())}")
    print(f"  Total successes: {total_successes} / {total_episodes} episodes")
    gym_env.close()

    return {
        "steps": log_steps,
        "success_rates": success_rates,
        "overall_success_rates": overall_success_rates,
        "episode_counts": episode_counts,
        "total_successes": cumulative_successes,
        "exhaustion_counts": exhaustion_counts,
        "param_loss": param_loss,
        "q_loss": q_loss,
    }


def plot_results(
    num_steps: int = 4000,
    seed: int = 0,
    env: str = "clutteredretrieval2d",
    complexity: int = 1,
    save_path: str | None = None,
    use_abstract_plan_scorer: bool = True,
    use_parameter_scorer: bool = True,
    **kwargs,
) -> None:
    """Run main() and plot success rate and episode count."""
    results = main(
        num_steps=num_steps,
        seed=seed,
        env=env,
        complexity=complexity,
        use_abstract_plan_scorer=use_abstract_plan_scorer,
        use_parameter_scorer=use_parameter_scorer,
        **kwargs,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Overall success rate ---
    ax = axes[0]
    ax.plot(
        results["steps"],
        [v * 100 for v in results["overall_success_rates"]],
        marker="s",
        linestyle="--",
        label="Overall",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Overall success rate (successes / episodes)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Episodes over steps ---
    ax = axes[1]
    ax.plot(
        results["steps"],
        results["episode_counts"],
        marker="^",
        color="tab:green",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative episodes")
    ax.set_title("Episodes completed over steps")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Kinder environment test for SimFreeParamPolicyApproach"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run and produce result plots",
    )
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--max-resamples", type=int, default=20)
    parser.add_argument(
        "--reset-every",
        type=int,
        default=300,
        help="Force-reset episode after this many steps (default: 300)",
    )
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env",
        default="clutteredretrieval2d",
        choices=list(_ENV_REGISTRY),
        help="Short environment name (default: clutteredretrieval2d)",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=1,
        metavar="N",
        help="Complexity integer for the environment, e.g. num_obstructions (default: 1)",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save figure to PATH instead of displaying it",
    )
    parser.add_argument(
        "--no-abstract-plan-scorer",
        action="store_true",
        help="Ablation: always use the first candidate plan, skip BALD scoring",
    )
    parser.add_argument(
        "--no-parameter-scorer",
        action="store_true",
        help="Ablation: always use the first parameter sample, skip scorer",
    )
    args = parser.parse_args()

    if args.plot:
        plot_results(
            num_steps=args.num_steps,
            max_resamples=args.max_resamples,
            reset_every=args.reset_every,
            log_every=args.log_every,
            seed=args.seed,
            env=args.env,
            complexity=args.complexity,
            save_path=args.save,
            use_abstract_plan_scorer=not args.no_abstract_plan_scorer,
            use_parameter_scorer=not args.no_parameter_scorer,
        )
    else:
        main(
            num_steps=args.num_steps,
            max_resamples=args.max_resamples,
            reset_every=args.reset_every,
            log_every=args.log_every,
            seed=args.seed,
            env=args.env,
            complexity=args.complexity,
            use_abstract_plan_scorer=not args.no_abstract_plan_scorer,
            use_parameter_scorer=not args.no_parameter_scorer,
        )
