"""Kinder environment test for SimFreeParamPolicyApproach.

Runs the approach on a configurable kinder environment and logs
the same metrics as bandit_test.py: rolling success rate, overall success
rate, total successes, parameter resamples, and resample exhaustions.

Training uses a fresh seed on every episode reset so the agent never
revisits the same environment configuration.  At every log checkpoint a
separate eval loop runs ``num_eval_seeds`` episodes on a fixed set of
held-out seeds to measure generalisation consistently across checkpoints.

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


def _run_eval_loop(
    approach: SimFreeParamPolicyApproach,
    gym_env,
    eval_seeds: list[int],
    reset_every: int,
) -> tuple[float, int, int]:
    """Run held-out eval episodes and return (success_rate, successes, episodes).

    Switches the approach to eval mode, runs one episode per seed in
    ``eval_seeds``, then switches back to train mode.  The same fixed seed
    set is used at every checkpoint so results are directly comparable.
    No learning occurs during eval.
    """
    if not eval_seeds:
        return 0.0, 0, 0

    approach.eval()
    eval_successes = 0
    last_obs = None

    for eval_seed in eval_seeds:
        last_obs, _ = gym_env.reset(seed=eval_seed)
        approach.reset_episode(last_obs)

        episode_success = False
        for _ in range(reset_every):
            try:
                action = approach.step()
            except ApproachStepError:
                break
            last_obs, reward, done, _, _ = gym_env.step(action)
            approach.update(last_obs, float(reward), done, {})
            if done:
                episode_success = True
                break

        eval_successes += int(episode_success)

    # Clear lingering episode state so the subsequent train reset_episode
    # does not record a spurious failure (the failure guard checks
    # _most_recent_parameter, which reset_episode sets to None).
    if last_obs is not None:
        approach.reset_episode(last_obs)

    approach.train()
    num_eval = len(eval_seeds)
    return eval_successes / num_eval, eval_successes, num_eval


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
    num_eval_seeds: int = 10,
    param_temperature: float = 1.0,
) -> dict:
    """Run an experiment on a kinder environment and return collected metrics.

    Args:
        env: Short environment name from _ENV_REGISTRY (e.g. "clutteredretrieval2d").
        complexity: The complexity integer passed to the env (e.g. num_obstructions).
        num_eval_seeds: Number of fixed held-out eval seeds.  The same seeds
            are evaluated at every log checkpoint for consistent comparison.

    Returns a dict with keys:
        steps                      — list of step indices at each log point
        success_rates              — train rolling success rate at each log point
        overall_success_rates      — train total_successes / total_episodes
        total_successes            — train cumulative successes
        episode_counts             — train cumulative episode count
        exhaustion_counts          — cumulative resample exhaustion count
        eval_success_rates         — eval success rate at each log point
        eval_cumulative_successes  — eval cumulative successes across checkpoints
        param_loss                 — per-fit sklearn loss values
        q_loss                     — per-epoch Q-network loss values
    """

    env_id_template, model_name, complexity_kwarg = _ENV_REGISTRY[env]
    print(
        f"env={env}  complexity={complexity}  ({complexity_kwarg}={complexity})  "
        f"num_eval_seeds={num_eval_seeds}"
    )

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
        "num_epochs": 200,
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
        train_every=5,
        param_sample_count=100,
        seed=seed,
        use_abstract_plan_scorer=use_abstract_plan_scorer,
        use_parameter_scorer=use_parameter_scorer,
        param_temperature=param_temperature,
    )

    approach.train()
    approach.reset(obs, {})

    # Reserve a fixed set of eval seeds that won't overlap with training.
    # Training seeds start at ``seed`` and increment; eval seeds live in a
    # separate high range so there is never a collision.
    eval_seeds = [seed + 1_000_000 + i for i in range(num_eval_seeds)]

    # Tracking
    recent: deque[int] = deque(maxlen=5)
    total_successes = 0
    total_episodes = 0
    train_seed_counter = 0  # monotonically increasing — fresh seed each reset
    episode_success = False

    log_steps: list[int] = []
    success_rates: list[float] = []
    overall_success_rates: list[float] = []
    episode_counts: list[int] = []
    cumulative_successes: list[int] = []
    exhaustion_counts: list[int] = []
    eval_success_rates: list[float] = []
    eval_cumulative_successes: list[int] = []
    total_eval_successes = 0
    param_loss: list[float] = []
    q_loss: list[float] = []

    header = (
        f"{'Step':>6}  "
        f"{'[Train]':>7}  {'Roll%':>7}  {'Overall%':>8}  {'Succ':>6}  {'Exhaus':>6}  "
        f"{'[Eval]':>6}  {'Rate%':>7}  {'Succ/N':>6}"
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
            train_seed_counter += 1
            obs, _ = gym_env.reset(seed=seed + train_seed_counter)
            approach.reset_episode(obs, truncated=False)
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
            train_seed_counter += 1
            obs, _ = gym_env.reset(seed=seed + train_seed_counter)
            approach.reset_episode(obs)

        if (step + 1) % log_every == 0:
            rolling_rate = sum(recent) / len(recent) if recent else 0.0
            overall_rate = total_successes / total_episodes if total_episodes else 0.0
            exhaustion_count = approach.get_resample_exhaustion_count()

            # --- Eval loop on fixed held-out seeds ---
            eval_rate, eval_succ, _ = _run_eval_loop(
                approach, gym_env, eval_seeds, reset_every
            )
            total_eval_successes += eval_succ

            print(
                f"{step+1:>6}  "
                f"{'':>7}  {rolling_rate:>7.2%}  {overall_rate:>8.2%}  "
                f"{total_successes:>6}  {exhaustion_count:>6}  "
                f"{'':>6}  {eval_rate:>7.2%}  {eval_succ:>3}/{num_eval_seeds:<3}"
            )

            log_steps.append(step + 1)
            success_rates.append(rolling_rate)
            overall_success_rates.append(overall_rate)
            episode_counts.append(total_episodes)
            cumulative_successes.append(total_successes)
            exhaustion_counts.append(exhaustion_count)
            eval_success_rates.append(eval_rate)
            eval_cumulative_successes.append(total_eval_successes)
            param_loss.extend(_get_param_scorer_loss_curves(approach))
            q_loss = _get_q_network_loss_curves(approach)

            # Restore training state — start a fresh episode on the next train seed.
            train_seed_counter += 1
            obs, _ = gym_env.reset(seed=seed + train_seed_counter)
            approach.reset_episode(obs)
            episode_success = False

    print("\nDone.")
    param_ds = approach.get_parameter_dataset()
    print(f"  Total parameter samples: {sum(len(v) for v in param_ds.values())}")
    print(f"  Train successes: {total_successes} / {total_episodes} episodes")
    print(
        f"  Eval  successes: {total_eval_successes} / "
        f"{len(log_steps) * num_eval_seeds} episodes"
    )
    gym_env.close()

    return {
        "steps": log_steps,
        "success_rates": success_rates,
        "overall_success_rates": overall_success_rates,
        "episode_counts": episode_counts,
        "total_successes": cumulative_successes,
        "exhaustion_counts": exhaustion_counts,
        "eval_success_rates": eval_success_rates,
        "eval_cumulative_successes": eval_cumulative_successes,
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

    # --- Overall success rate: train vs eval ---
    ax = axes[0]
    ax.plot(
        results["steps"],
        [v * 100 for v in results["overall_success_rates"]],
        marker="s",
        linestyle="--",
        label="Train (overall)",
    )
    ax.plot(
        results["steps"],
        [v * 100 for v in results["eval_success_rates"]],
        marker="o",
        linestyle="-",
        label="Eval (held-out)",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success rate: train vs eval")
    ax.set_ylim(0, 105)
    ax.legend()
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
    ax.set_ylabel("Cumulative train episodes")
    ax.set_title("Train episodes completed over steps")
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
    parser.add_argument("--num-steps", type=int, default=50000)
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
    parser.add_argument(
        "--num-eval-seeds",
        type=int,
        default=10,
        metavar="N",
        help="Number of fixed held-out eval seeds (default: 10)",
    )
    parser.add_argument(
        "--param-temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Boltzmann temperature for parameter policy sampling (default: 1.0)",
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
            num_eval_seeds=args.num_eval_seeds,
            param_temperature=args.param_temperature,
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
            num_eval_seeds=args.num_eval_seeds,
            param_temperature=args.param_temperature,
        )
