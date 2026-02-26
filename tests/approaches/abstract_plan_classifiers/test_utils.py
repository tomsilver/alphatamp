"""Tests for the approaches/abstract_plan_classifiers/utils.py."""

from collections import defaultdict

import kinder
import numpy as np
import pytest
import torch
from bilevel_planning.utils import (
    cached_all_ground_operators,
    get_all_ground_atoms_for_predicate,
)
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from torch import nn

from alphatamp.approaches.abstract_explorers.exploit_explorer import ExploitExplorer
from alphatamp.approaches.abstract_plan_classifiers.q_network import (
    PerActionQNetwork,
    create_abstract_plan_sequence,
)
from alphatamp.approaches.abstract_plan_classifiers.utils import train_q_network
from alphatamp.approaches.feasibility_classifier_learners.static_feasibility_classifier_learner import (  # pylint:disable=line-too-long
    StaticFeasibilityClassifierLearner,
)
from alphatamp.approaches.feasibility_classifiers.filter_feasibility_classifier import (
    FilterFeasibilityClassifier,
)
from alphatamp.approaches.scorers.regressor_abstract_action_scorer import (
    AbstractActionScorer,
)
from alphatamp.approaches.simulator_free_base_approach import (
    sesame_models_to_sim_free,
)


@pytest.mark.slow
def test_train_q_network():
    """Train the PerActionQNetwork using ground atoms/operators from a real environment
    and synthetically generated training data."""

    # Set up the kinder environment.
    kinder.register_all_environments()
    env = kinder.make("kinder/ClutteredRetrieval2D-o10-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="q-function")

    env_models = create_bilevel_planning_models(
        "clutteredretrieval2d",
        env.observation_space,
        env.action_space,
        num_obstructions=10,
    )
    sim_free_env_models = sesame_models_to_sim_free(env_models)

    # Ground atoms and operators from the environment.
    obs, _ = env.reset(seed=123)
    x0 = sim_free_env_models.observation_to_state(obs)
    s0 = sim_free_env_models.state_abstractor(x0)

    all_ground_operators = tuple(
        sorted(cached_all_ground_operators(sim_free_env_models.operators, s0.objects))
    )

    all_ground_atoms_set = set()
    for predicate in sim_free_env_models.predicates:
        all_ground_atoms_set.update(
            get_all_ground_atoms_for_predicate(predicate, s0.objects)
        )
    all_ground_atoms = tuple(sorted(all_ground_atoms_set))

    # Generate abstract plans from the explorer across a few seeds.
    filter_classifier = FilterFeasibilityClassifier()
    filter_learner = StaticFeasibilityClassifierLearner(filter_classifier)
    explorer = ExploitExplorer(sim_free_env_models, filter_learner, 123)

    abstract_plans = []
    for seed in range(10):
        obs_i, _ = env.reset(seed=seed)
        try:
            plan = explorer.generate_abstract_plan(obs_i)
            abstract_plans.append(plan)
        except Exception:
            continue

    assert len(abstract_plans) > 0, "Should generate at least one abstract plan"

    # Build synthetic training data for the AbstractActionScorers.
    # For each plan, encode it and assign a random resample count per action,
    # keyed by the action's short_str descriptor (same format as the saved dataset).
    rng = np.random.default_rng(42)
    synthetic_dataset: dict[str, list] = defaultdict(list)

    for plan in abstract_plans:
        states, actions = plan
        for i, action in enumerate(actions):
            # Build the prefix skeleton up to (but not including) this action.
            prefix_states = states[: i + 1]
            prefix_actions = actions[:i]
            prefix_plan = (prefix_states, prefix_actions)

            sequence, seq_len = create_abstract_plan_sequence(
                all_ground_atoms, all_ground_operators, prefix_plan
            )
            resample_count = int(rng.integers(0, 6))
            synthetic_dataset[action.short_str].append(
                (sequence, seq_len, resample_count)
            )

    # Build per-action AbstractActionScorers and train on synthetic data.
    abstract_action_configs = {
        "hidden_dim": 32,
        "num_layers": 2,
        "num_epochs": 5,
    }
    trained_scorers = {}
    for op in all_ground_operators:
        scorer = AbstractActionScorer(
            all_ground_atoms, all_ground_operators, configs=abstract_action_configs
        )
        descriptor = op.short_str
        if descriptor in synthetic_dataset:
            entries = synthetic_dataset[descriptor]
            features = [torch.FloatTensor(e[0]) for e in entries]
            lengths = torch.tensor([e[1] for e in entries], dtype=torch.long)
            targets = torch.FloatTensor([e[2] for e in entries]).unsqueeze(-1)
            scorer.train(features, targets, lengths, nn.MSELoss())
        trained_scorers[op] = scorer

    # Build and train the PerActionQNetwork.
    q_net = PerActionQNetwork(
        all_ground_atoms, all_ground_operators, hidden_dim=32, num_layers=2
    )

    num_epochs = 20
    losses = train_q_network(
        q_net,
        abstract_plans,
        all_ground_atoms,
        all_ground_operators,
        trained_scorers,
        batch_size=4,
        num_epochs=num_epochs,
        verbose=True,
    )

    # One loss per epoch, all finite and non-negative.
    assert len(losses) == num_epochs
    for loss in losses:
        assert isinstance(loss, float)
        assert loss >= 0
        assert np.isfinite(loss)

    # Loss should not diverge: final average should not be much worse than initial.
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) * 1.5

    # Predictions should have the correct shape for each plan.
    for plan in abstract_plans:
        preds = q_net.predict(plan)
        num_actions = len(plan[1])
        assert preds.shape == (num_actions,)

    env.close()
