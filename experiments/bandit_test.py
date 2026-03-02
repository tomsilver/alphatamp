"""Minimal 1D bandit test for SimFreeParamPolicyApproach.

One abstract action ('Reach'), one continuous parameter to learn.  The agent
must choose a placement position close to a random target in [0, 1].

Demonstrates that SimFreeParamPolicyApproach can learn — the ClassifierParameterScorer
gradually assigns higher probability to parameters near the target, so the
resample count falls and the success rate rises over time.

Usage::

    python experiments/bandit_test.py

The script prints rolling success rate every ``log_every`` steps so you can see
the approach learning.
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Any, Callable

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
from alphatamp.approaches.scorers.abstract_action_scorers.regressor_abstract_action_scorer import (
    AbstractActionScorer,
)
from alphatamp.approaches.scorers.parameter_scorers.classifier_parameter_scorer import (
    ClassifierParameterScorer,
)
from alphatamp.approaches.scorers.parameter_scorers.naive_parameter_scorer import (
    NaiveParameterScorer,
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

# Feature layout for ObjectCentricState: [target_pos, is_solved]
TYPE_FEATURES = {ROBOT_TYPE: ["target_pos", "is_solved"]}

SUCCESS_THRESHOLD = 0.05  # |action - target| must be below this to succeed


# ---------------------------------------------------------------------------
# Minimal gymnasium environment
# ---------------------------------------------------------------------------


class OneDimBanditEnv(gym.Env):  # type: ignore[type-arg]
    """A 1-D bandit: the agent must place at the (random) target position.

    Observation: [target_pos, is_solved]  — shape (2,)
    Action:      [placement_pos]          — shape (1,)

    Reward 1.0 and done=True when |placement - target| < SUCCESS_THRESHOLD.
    Otherwise reward 0.0 and done=False (the episode continues until the
    approach resets it externally).
    """

    metadata = {"render_modes": []}

    def __init__(self, fixed_target: float | None = None) -> None:
        super().__init__()
        self.observation_space = Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self.action_space = Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        self._fixed_target = fixed_target
        self._target: float = 0.5
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
        obs = np.array([self._target, 0.0], dtype=np.float32)
        return obs, {"target": self._target}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        placement = float(np.clip(action[0], 0.0, 1.0))
        success = abs(placement - self._target) < SUCCESS_THRESHOLD
        obs = np.array([self._target, 1.0 if success else 0.0], dtype=np.float32)
        reward = 1.0 if success else 0.0
        return obs, reward, success, False, {"target": self._target}

    def render(self) -> None:  # type: ignore[override]
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Controller: one-shot placement
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
        return self._params.copy()


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


def _goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
    return RelationalAbstractGoal(
        atoms={AT_GOAL_PRED([ROBOT_OBJ])},
        state_abstractor=_state_abstractor,
    )


def build_bandit_env_models(env: OneDimBanditEnv) -> SimulatorFreeSesameModels:
    """Build minimal SimulatorFreeSesameModels for the bandit environment."""

    # PDDL operator: Reach(?r) — pre={}, add={AtGoal(?r)}, del={}
    op = LiftedOperator(
        name="Reach",
        parameters=[ROBOT_VAR],
        preconditions=set(),
        add_effects={LiftedAtom(AT_GOAL_PRED, [ROBOT_VAR])},
        delete_effects=set(),
    )

    ctrl = LiftedParameterizedController(
        variables=[ROBOT_VAR],
        controller_cls=ReachController,
        params_space=Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    )

    skill = LiftedSkill(operator=op, controller=ctrl)

    return SimulatorFreeSesameModels(
        observation_space=env.observation_space,
        state_space=env.observation_space,  # state == obs here
        action_space=env.action_space,
        types={ROBOT_TYPE},
        predicates={AT_GOAL_PRED},
        observation_to_state=_obs_to_state,
        state_abstractor=_state_abstractor,
        goal_deriver=_goal_deriver,
        skills={skill},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_SCORER_CLASSES = {
    "classifier": ClassifierParameterScorer,
    "naive": NaiveParameterScorer,
}

_SCORER_CONFIGS: dict[str, dict] = {
    "classifier": {"configs": {"hidden_layer_sizes": (16, 16)}},
    "naive": {"configs": {}},
}


def _get_param_scorer_loss_curves(approach: SimFreeParamPolicyApproach) -> list[float]:
    """Extract per-fit loss curve from any MLPClassifier-backed parameter scorers."""
    curves: list[float] = []
    for scorer_fn in approach._abstract_action_to_scoring_function.values():
        clf = getattr(scorer_fn, "_classifier", None)
        if clf is not None and hasattr(clf, "loss_curve_"):
            curves.extend(clf.loss_curve_)
    return curves


def main(
    num_steps: int = 2000,
    max_resamples: int = 20,
    reset_every: int = 30,
    log_every: int = 100,
    seed: int = 0,
    fixed_target: float | None = None,
    scorer: str = "classifier",
) -> dict:
    """Run the bandit experiment and return collected metrics.

    Returns a dict with keys:
        steps           — list of step indices at each log point
        success_rates   — rolling success rate at each log point
        total_successes — cumulative successes at each log point
        param_loss      — per-fit sklearn loss values (classifier scorer only)
    """
    if scorer not in _SCORER_CLASSES:
        raise ValueError(
            f"Unknown scorer '{scorer}'. Choose from: {list(_SCORER_CLASSES)}"
        )

    print(f"\n=== Scorer: {scorer} ===")

    # Build env
    env = OneDimBanditEnv(fixed_target=fixed_target)
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
        parameter_scorer_class=_SCORER_CLASSES[scorer],
        parameter_scorer_configs=_SCORER_CONFIGS[scorer],
        abstract_action_scorer_class=AbstractActionScorer,
        abstract_action_scorer_configs={"configs": abstract_action_configs},
        q_network_configs=q_network_configs,
        max_resamples=max_resamples,
        train_every=1,
        seed=seed,
    )

    approach.train()
    approach.reset(obs, {})

    # Tracking
    recent: deque[int] = deque(maxlen=log_every)
    total_successes = 0
    reset_count = 0

    log_steps: list[int] = []
    success_rates: list[float] = []
    cumulative_successes: list[int] = []
    param_loss: list[float] = []

    print(
        f"{'Step':>6}  {'Rolling success':>15}  {'Total successes':>16}  {'Resamples':>10}"
    )
    print("-" * 55)

    for step in range(num_steps):
        try:
            action = approach.step()
        except ApproachStepError:
            reset_count += 1
            obs, _ = env.reset(seed=seed + reset_count)
            approach.reset_episode(obs)
            continue

        obs, reward, done, _, info = env.step(action)
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
            param_ds = approach.get_parameter_dataset()
            total_data = sum(len(v) for v in param_ds.values())
            print(
                f"{step+1:>6}  {rolling_rate:>15.2%}  {total_successes:>16}  "
                f"{total_data:>10}"
            )
            log_steps.append(step + 1)
            success_rates.append(rolling_rate)
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
        "total_successes": cumulative_successes,
        "param_loss": param_loss,
    }


def compare_scorers(
    scorers: list[str],
    num_steps: int = 2000,
    seed: int = 0,
    save_path: str | None = None,
    **kwargs: Any,
) -> None:
    """Run main() for each scorer and plot success rate + loss curves."""
    results: dict[str, dict] = {}
    for s in scorers:
        results[s] = main(num_steps=num_steps, seed=seed, scorer=s, **kwargs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Rolling success rate ---
    ax = axes[0]
    for s, r in results.items():
        ax.plot(r["steps"], [v * 100 for v in r["success_rates"]], marker="o", label=s)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling success rate (%)")
    ax.set_title("Rolling success rate by scorer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Parameter scorer loss (classifier only) ---
    ax = axes[1]
    any_loss = False
    for s, r in results.items():
        loss = r["param_loss"]
        if loss:
            ax.plot(loss, label=s)
            any_loss = True
    if any_loss:
        ax.set_xlabel("Cumulative sklearn fit iteration")
        ax.set_ylabel("Training loss")
        ax.set_title("Parameter scorer training loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No loss data\n(naive scorer has none)",
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
        "--scorer",
        choices=list(_SCORER_CLASSES),
        default="classifier",
        help="Which parameter scorer to use (default: classifier)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all scorers and produce comparison plots",
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save comparison figure to PATH instead of displaying it",
    )
    args = parser.parse_args()

    if args.compare:
        compare_scorers(
            list(_SCORER_CLASSES),
            num_steps=args.num_steps,
            seed=args.seed,
            save_path=args.save,
        )
    else:
        main(num_steps=args.num_steps, seed=args.seed, scorer=args.scorer)
