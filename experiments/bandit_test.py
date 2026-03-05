"""Minimal 1D bandit test for SimFreeParamPolicyApproach.

Two abstract actions: 'Reach' (learn a placement position) and 'Widen'
(increase the success threshold — no symbolic effects).  The agent must
choose a placement position close to a random target in [0, 1].

Demonstrates that SimFreeParamPolicyApproach can discover that using the
Widen action before Reach leads to fewer parameter resamples.  Initially
the approach uses [Reach] alone (the shortest plan).  After repeated
resample failures, BALD selects [Widen, Reach] due to high epistemic
uncertainty, discovers the widened threshold makes Reach easier, and
gradually shifts toward preferring that plan.

Widen has no add/delete effects in the symbolic model, so the planner
cannot reason about its benefit — the approach must discover it empirically.
The heuristic search generator still produces [Widen, Reach] as an
alternative plan because Widen has empty preconditions (always applicable)
and self-loops on the same abstract state.

Usage::

    python experiments/bandit_test.py
    python experiments/bandit_test.py --plot
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from relational_structs.object_centric_state import ObjectCentricState

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
from alphatamp.approaches.simulator_free_base_approach import SimulatorFreeSesameModels
from alphatamp.approaches.utils.approach_step_error import ApproachStepError

# ---------------------------------------------------------------------------
# Relational world constants
# ---------------------------------------------------------------------------

ROBOT_TYPE = Type("robot")
ROBOT_OBJ = Object("robot", ROBOT_TYPE)
AT_GOAL_PRED = Predicate("AtGoal", [ROBOT_TYPE])
ROBOT_VAR = Variable("?r", ROBOT_TYPE)

# Feature layout for ObjectCentricState: [target_pos, is_solved, is_widened]
TYPE_FEATURES = {ROBOT_TYPE: ["target_pos", "is_solved", "is_widened"]}

SUCCESS_THRESHOLD = 0.05       # |action - target| must be below this normally
WIDE_SUCCESS_THRESHOLD = 0.15  # threshold after Widen is applied


# ---------------------------------------------------------------------------
# Minimal gymnasium environment
# ---------------------------------------------------------------------------


class OneDimBanditEnv(gym.Env):  # type: ignore[type-arg]
    """A 1-D bandit: the agent must place at the (random) target position.

    Observation: [target_pos, is_solved, is_widened]  — shape (3,)
    Action:      [placement_pos, widen_flag]           — shape (2,)

    If widen_flag > 0.5 the step widens the success threshold (no placement).
    Otherwise the agent attempts a placement at placement_pos, perturbed by
    Gaussian noise with std ``execution_noise_std``.  The noise makes Reach
    non-trivially hard: even a perfectly learned parameter fails with positive
    probability, so the advantage of Widen (wider threshold) persists after
    training.

    Reward 1.0 and done=True when |noisy_placement - target| < threshold.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        fixed_target: float | None = None,
        execution_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.observation_space = Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )
        self.action_space = Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
        )
        self._fixed_target = fixed_target
        self._execution_noise_std = execution_noise_std
        self._target: float = 0.5
        self._is_widened: bool = False
        self._rng = np.random.default_rng(0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._fixed_target is not None:
            self._target = self._fixed_target
        else:
            self._target = float(self._rng.uniform(0.0, 1.0))
        self._is_widened = False
        obs = np.array([self._target, 0.0, 0.0], dtype=np.float32)
        return obs, {"target": self._target}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if float(action[1]) > 0.5:
            # Widen step: increase success threshold, no placement attempted.
            self._is_widened = True
            obs = np.array([self._target, 0.0, 1.0], dtype=np.float32)
            return obs, 0.0, False, False, {"target": self._target}

        # Reach step: attempt placement (with optional execution noise).
        noise = self._rng.normal(0.0, self._execution_noise_std) if self._execution_noise_std > 0.0 else 0.0
        placement = float(np.clip(action[0] + noise, 0.0, 1.0))
        threshold = WIDE_SUCCESS_THRESHOLD if self._is_widened else SUCCESS_THRESHOLD
        success = abs(placement - self._target) < threshold
        obs = np.array(
            [self._target, 1.0 if success else 0.0, float(self._is_widened)],
            dtype=np.float32,
        )
        return obs, 1.0 if success else 0.0, success, False, {"target": self._target}

    def render(self) -> None:  # type: ignore[override]
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class ReachController(GroundParameterizedController):
    """Places the robot at a sampled position; fails (raises) if prev action failed."""

    def __init__(self, objects: list[Object]) -> None:
        super().__init__(objects)
        self._robot = objects[0]
        self._params: np.ndarray | None = None
        self._prev_failed: bool = False

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=1).astype(np.float32)

    def reset(self, x: ObjectCentricState, params: np.ndarray) -> None:
        self._params = params
        self._prev_failed = False

    def observe(self, x: ObjectCentricState) -> None:
        """Mark as failed if the environment did not reach the goal."""
        solved = x.get(self._robot, "is_solved")
        self._prev_failed = float(solved) < 0.5

    def terminated(self) -> bool:
        return False

    def step(self) -> np.ndarray:
        if self._prev_failed:
            raise TrajectorySamplingFailure("Previous placement missed target.")
        assert self._params is not None
        # Action is [placement_pos, widen_flag=0]
        return np.array([self._params[0], 0.0], dtype=np.float32)


class WideController(GroundParameterizedController):
    """One-shot controller that widens the environment's success threshold.

    Has no preconditions or symbolic effects — the planner cannot reason about
    its benefit.  The approach must discover empirically that executing Widen
    before Reach reduces the number of parameter resamples required.
    """

    def __init__(self, objects: list[Object]) -> None:
        super().__init__(objects)
        self._done: bool = False

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> np.ndarray:
        # No meaningful parameters; return a dummy value.
        return np.zeros(1, dtype=np.float32)

    def reset(self, x: ObjectCentricState, params: np.ndarray) -> None:
        self._done = False

    def observe(self, x: ObjectCentricState) -> None:
        pass  # Widen always succeeds; nothing to track.

    def terminated(self) -> bool:
        return self._done

    def step(self) -> np.ndarray:
        # Signal to the environment to widen the threshold.
        self._done = True
        return np.array([0.0, 1.0], dtype=np.float32)  # widen_flag = 1


# ---------------------------------------------------------------------------
# Helper: build SimulatorFreeSesameModels
# ---------------------------------------------------------------------------


def _obs_to_state(obs: np.ndarray) -> ObjectCentricState:
    """Wrap a raw gymnasium observation in an ObjectCentricState."""
    return ObjectCentricState(
        data={ROBOT_OBJ: obs.copy()},
        type_features=TYPE_FEATURES,
    )


def _state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
    """Return {AtGoal(robot)} if solved, else {}."""
    is_solved = float(x.get(ROBOT_OBJ, "is_solved")) > 0.5
    atoms: set[GroundAtom] = {AT_GOAL_PRED([ROBOT_OBJ])} if is_solved else set()
    return RelationalAbstractState(atoms=atoms, objects={ROBOT_OBJ})


def _goal_deriver(_: ObjectCentricState) -> RelationalAbstractGoal:
    return RelationalAbstractGoal(
        atoms={AT_GOAL_PRED([ROBOT_OBJ])},
        state_abstractor=_state_abstractor,
    )


def build_bandit_env_models(env: OneDimBanditEnv) -> SimulatorFreeSesameModels:
    """Build minimal SimulatorFreeSesameModels for the bandit environment."""

    # Reach(?r) — pre={}, add={AtGoal(?r)}, del={}
    reach_op = LiftedOperator(
        name="Reach",
        parameters=[ROBOT_VAR],
        preconditions=set(),
        add_effects={LiftedAtom(AT_GOAL_PRED, [ROBOT_VAR])},
        delete_effects=set(),
    )
    reach_ctrl: LiftedParameterizedController = LiftedParameterizedController(
        variables=[ROBOT_VAR],
        controller_cls=ReachController,
        params_space=Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    )
    reach_skill = LiftedSkill(operator=reach_op, controller=reach_ctrl)

    # Widen(?r) — pre={}, add={}, del={} (no symbolic effects)
    # The planner generates [Reach] and [Widen, Reach] as alternative plans
    # because Widen has empty preconditions and self-loops on the same abstract
    # state.  The approach must discover via BALD that Widen reduces resamples.
    widen_op = LiftedOperator(
        name="Widen",
        parameters=[ROBOT_VAR],
        preconditions=set(),
        add_effects=set(),
        delete_effects=set(),
    )
    widen_ctrl: LiftedParameterizedController = LiftedParameterizedController(
        variables=[ROBOT_VAR],
        controller_cls=WideController,
        params_space=Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    )
    widen_skill = LiftedSkill(operator=widen_op, controller=widen_ctrl)

    return SimulatorFreeSesameModels(
        observation_space=env.observation_space,
        state_space=env.observation_space,
        action_space=env.action_space,
        types={ROBOT_TYPE},
        predicates={AT_GOAL_PRED},
        observation_to_state=_obs_to_state,
        state_abstractor=_state_abstractor,
        goal_deriver=_goal_deriver,
        skills={reach_skill, widen_skill},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_CLASSIFIER_SCORER_CONFIGS: dict = {"configs": {"hidden_layer_sizes": (16, 16)}}


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


def _plan_uses_widen(approach: SimFreeParamPolicyApproach) -> bool:
    """Return True if the current abstract plan contains a Widen action."""
    plan = approach.get_abstract_plan()
    if plan is None:
        return False
    _, actions = plan
    return any("Widen" in a.short_str for a in actions)


def main(
    num_steps: int = 2000,
    max_resamples: int = 20,
    reset_every: int = 30,
    log_every: int = 100,
    seed: int = 0,
    fixed_target: float | None = None,
    noise_std: float = 0.0,
) -> dict:
    """Run the bandit experiment and return collected metrics.

    Returns a dict with keys:
        steps           — list of step indices at each log point
        success_rates   — rolling success rate at each log point
        total_successes — cumulative successes at each log point
        widen_rates     — rolling fraction of steps with a Widen plan at each log point
        param_loss      — per-fit sklearn loss values
    """

    # Build env
    env = OneDimBanditEnv(fixed_target=fixed_target, execution_noise_std=noise_std)
    obs, _ = env.reset(seed=seed)

    # Build models
    env_models = build_bandit_env_models(env)

    # Feasibility: allow all plans (filter classifier with nothing filtered)
    filter_clf = FilterFeasibilityClassifier()
    feasibility_learner = StaticFeasibilityClassifierLearner(filter_clf)

    # Explorer
    train_explorer = ExploitExplorer(env_models, feasibility_learner, seed)

    abstract_action_configs = {"hidden_dim": 32, "num_layers": 2, "num_epochs": 20}
    q_network_configs = {
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 20,
        "num_ensemble_nets": 3,
    }

    # Approach
    approach = SimFreeParamPolicyApproach(
        env_models=env_models,
        feasibility_classifier_learner=feasibility_learner,
        train_explorer=train_explorer,
        parameter_scorer_class=ClassifierParameterScorer,
        parameter_scorer_configs=_CLASSIFIER_SCORER_CONFIGS,
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
        q_network_configs=q_network_configs,
        max_resamples=max_resamples,
        train_every=1,
        param_sample_count=100,
        seed=seed,
    )

    approach.train()
    approach.reset(obs, {})

    # Tracking
    recent: deque[int] = deque(maxlen=log_every)
    recent_widen: deque[int] = deque(maxlen=log_every)
    total_successes = 0
    reset_count = 0

    log_steps: list[int] = []
    success_rates: list[float] = []
    widen_rates: list[float] = []
    cumulative_successes: list[int] = []
    param_loss: list[float] = []

    header = (
        f"{'Step':>6}  {'Rolling success':>15}  "
        f"{'Widen rate':>10}  {'Total successes':>16}  {'Resamples':>10}"
    )
    print(header)
    print("-" * 65)

    for step in range(num_steps):
        try:
            action = approach.step()
        except ApproachStepError:
            reset_count += 1
            obs, _ = env.reset(seed=seed + reset_count)
            approach.reset_episode(obs)
            continue

        # Track whether the current plan includes Widen before taking the step.
        recent_widen.append(int(_plan_uses_widen(approach)))

        obs, reward, done, _, _ = env.step(action)
        approach.update(obs, float(reward), done, {})

        success = int(done)
        recent.append(success)
        total_successes += success

        if done or (step + 1) % reset_every == 0:
            reset_count += 1
            obs, _ = env.reset(seed=seed + reset_count)
            approach.reset_episode(obs)

        if (step + 1) % log_every == 0:
            rolling_rate = sum(recent) / len(recent) if recent else 0.0
            widen_rate = sum(recent_widen) / len(recent_widen) if recent_widen else 0.0
            param_ds = approach.get_parameter_dataset()
            total_data = sum(len(v) for v in param_ds.values())
            print(
                f"{step+1:>6}  {rolling_rate:>15.2%}  {widen_rate:>10.2%}  "
                f"{total_successes:>16}  {total_data:>10}"
            )
            log_steps.append(step + 1)
            success_rates.append(rolling_rate)
            widen_rates.append(widen_rate)
            cumulative_successes.append(total_successes)
            param_loss.extend(_get_param_scorer_loss_curves(approach))

    print("\nDone.")
    param_ds = approach.get_parameter_dataset()
    print(f"  Total parameter samples: {sum(len(v) for v in param_ds.values())}")
    print(f"  Total successes: {total_successes} / {num_steps}")
    env.close()

    return {
        "steps": log_steps,
        "success_rates": success_rates,
        "widen_rates": widen_rates,
        "total_successes": cumulative_successes,
        "param_loss": param_loss,
    }


def plot_results(
    num_steps: int = 2000,
    seed: int = 0,
    save_path: str | None = None,
    noise_std: float = 0.0,
    **kwargs: Any,
) -> None:
    """Run main() and plot success rate, Widen plan fraction, and loss curves."""
    results = main(num_steps=num_steps, seed=seed, noise_std=noise_std, **kwargs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Rolling success rate ---
    ax = axes[0]
    ax.plot(results["steps"], [v * 100 for v in results["success_rates"]], marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling success rate (%)")
    ax.set_title("Rolling success rate")
    ax.grid(True, alpha=0.3)

    # --- Widen plan fraction ---
    ax = axes[1]
    ax.plot(
        results["steps"],
        [v * 100 for v in results["widen_rates"]],
        marker="s",
        color="tab:orange",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps with Widen plan (%)")
    ax.set_title("Fraction of steps using [Widen, …, Reach] plan")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Parameter scorer loss ---
    ax = axes[2]
    loss = results["param_loss"]
    if loss:
        ax.plot(loss)
        ax.set_xlabel("Cumulative sklearn fit iteration")
        ax.set_ylabel("Training loss")
        ax.set_title("Parameter scorer training loss")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No loss data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Parameter scorer training loss")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1-D bandit test for SimFreeParamPolicyApproach"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run and produce result plots",
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--max-resamples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        metavar="STD",
        help="Gaussian execution noise std added to Reach placements (default: 0.0)",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save figure to PATH instead of displaying it",
    )
    args = parser.parse_args()

    if args.plot:
        plot_results(
            num_steps=args.num_steps,
            max_resamples=args.max_resamples,
            seed=args.seed,
            save_path=args.save,
            noise_std=args.noise_std,
        )
    else:
        main(num_steps=args.num_steps, max_resamples=args.max_resamples, seed=args.seed, noise_std=args.noise_std)
