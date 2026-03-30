"""Util functions for Q Networks."""

import logging

import numpy as np
import torch
from relational_structs.pddl import GroundAtom
from scipy.stats import entropy
from torch import nn

from alphatamp.approaches.abstract_plan_classifiers.q_network import (
    PerActionQNetwork,
    create_abstract_plan_sequence,
)
from alphatamp.approaches.scorers.abstract_action_scorers.regressor_abstract_action_scorer import (  # pylint:disable=line-too-long
    AbstractActionScorer,
)
from alphatamp.structs import GroundOperator, RelationalAbstractState, Skeleton


def conditional_abstract_action_q_value(
    history_states: list[RelationalAbstractState],
    history_actions: list[GroundOperator],
    current_action: GroundOperator,
    trained_abstract_action_scorers: dict[GroundOperator, AbstractActionScorer],
) -> float:
    """Compute Q_abstract((s_{n-1}, a_{n-1}) | (s_0, a_1, s_1, a_2, ..., a_{n-2})).

    This is the conditional Q-value for the current state-action pair given the history.
    This is computed by using the trained abstract action scorer.

    Args:
        history_states: List of abstract states (s_0, s_1, ..., s_{n-2})
        history_actions: List of abstract actions (a_1, a_2, ..., a_{n-2})
        current_action: The current abstract action a_{n-1}
        trained_abstract_action_scorers: Dictionary mapping abstract actions
                                        to trained abstract action scorers

    Returns:
        The conditional Q_abstract value for (s_{n-1}, a_{n-1}) given the history
    """
    # Compute Q_abstract((s_{n-1}, a_{n-1}) | history)
    # using the trained abstract action scorer.
    # This approximates sum_phi Q_param(x_{n-1}, a_{n-1}, phi)

    abstract_action_scorer = trained_abstract_action_scorers[current_action]

    conditional_abstract_plan = (history_states, history_actions)

    conditional_abstract_plan_score = abstract_action_scorer.score(
        conditional_abstract_plan
    )
    return conditional_abstract_plan_score


def abstract_plan_q_value(
    abstract_plan: Skeleton,
    q_value_cache: dict,
    gamma: float,
    trained_abstract_action_scorers: dict[GroundOperator, AbstractActionScorer],
) -> float:
    """Recursively compute Q_abstract((s_0, a_1, s_1, a_2, ..., a_{n-1})).

    Implements the recursive update rule:
    Q_abstract((s_0, a_1, ..., a_{n-1}))
    <- gamma * Q_abstract((s_0, a_1, ..., a_{n-1}))
    + (1-gamma) * Q̂_abstract((s_0, a_1, ..., a_{n-1}))

    where:
    Q̂_abstract((s_0, a_1, ..., a_{n-1})) =

    Q_abstract((s_0, a_1, ..., a_{n-2}))
    + Q_abstract((s_{n-1}, a_{n-1}) | (s_0, a_1, ..., a_{n-2}))

    Args:
        abstract_plan: The abstract plan (Skeleton) containing states and actions
        q_value_cache: Dictionary to cache Q-values for sequences
        gamma: Discount factor for the update rule
        trained_skill_scorers:
            Dictionary mapping abstract actions to trained parameter scorers

    Returns:
        The Q_abstract value for the sequence
    """
    # Create a frozen representation of the sequence for caching
    # Use tuple of (state hashes, action strings) as key for hashability
    states, actions = abstract_plan
    state_hashes = tuple(hash(state) for state in states)
    action_strings = tuple(action.short_str for action in actions)
    sequence_key = (state_hashes, action_strings)

    # Base case: empty sequence (just initial state, no actions)
    if len(actions) == 0:
        return 0.0

    # Get current Q-value from cache (default to 0.0 if not present)
    current_q_value = q_value_cache.get(sequence_key, 0.0)

    # Compute Q_hat_abstract recursively
    if len(actions) == 1:
        # Base case: single action, Q_abstract for empty prefix is
        q_prefix = 0.0
    else:
        # Recursive case: compute Q_abstract for shorter sequence
        prefix_states = states[:-1]  # s_0, s_1, ..., s_{n-2}
        prefix_actions = actions[:-1]  # a_1, a_2, ..., a_{n-2}
        prefix_abstract_plan = (prefix_states, prefix_actions)
        q_prefix = abstract_plan_q_value(
            prefix_abstract_plan,
            q_value_cache,
            gamma,
            trained_abstract_action_scorers,
        )

    # Compute conditional Q-value: Q_abstract((s_{n-1}, a_{n-1}) | history)
    current_action = actions[-1]  # a_{n-1}
    history_states = states[:-1]  # s_0, s_1, ..., s_{n-2}
    history_actions = actions[:-1]  # a_1, a_2, ..., a_{n-2}

    conditional_q_value = conditional_abstract_action_q_value(
        history_states,
        history_actions,
        current_action,
        trained_abstract_action_scorers,
    )

    # Compute Q_hat_abstract
    q_hat = q_prefix + conditional_q_value

    # Update Q-value using the update rule
    updated_q_value = gamma * current_q_value + (1 - gamma) * q_hat

    # Cache the updated value
    q_value_cache[sequence_key] = updated_q_value

    return updated_q_value


def estimated_abstract_plan_q_value(
    abstract_plan: Skeleton, q_network: PerActionQNetwork
) -> np.ndarray:
    """Estimate the per-action Q-values for an abstract plan using the trained
    Q-network.

    Args:
        abstract_plan: The abstract plan (Skeleton) to estimate Q-values for
        q_network: The trained PerActionQNetwork

    Returns:
        Array of shape (num_actions,) where each element is the predicted
        failure rate for that action conditioned on prior actions.
    """
    return q_network.predict(abstract_plan)


def compute_per_action_targets(
    abstract_plan: Skeleton,
    trained_abstract_action_scorers: dict[GroundOperator, AbstractActionScorer],
) -> list[float]:
    """Compute target failure rates for each action in the plan.

    For each action a_i in the plan, computes the predicted failure rate for
    that action, conditioned on the history (s_0, a_1, ..., a_{i-1}).

    This is the "ground truth" target that the Q-network will learn to predict.
    Each action's target is computed independently using the corresponding
    abstract action scorer.

    Args:
        abstract_plan: The abstract plan (Skeleton) containing states and actions
        trained_abstract_action_scorers: Dictionary mapping abstract actions
                                        to trained abstract action scorers

    Returns:
        List of length n (number of actions) where element i is the predicted
        failure rate for action a_i conditioned on (s_0, a_1, ..., a_{i-1})
    """
    states, actions = abstract_plan
    targets = []

    for i, action in enumerate(actions):
        # Build the history prefix for this action:
        # - history_states: all states up to the state after action i
        # - history_actions: all actions before action i
        #
        # For action a_i (0-indexed), we condition on:
        #   states: s_0, s_1, ..., s_{i} (i states including current one)
        #   actions: a_0, a_1, ..., a_{i-1} (all previous actions)
        history_states = list(states[: i + 1])
        history_actions = list(actions[:i])

        # Get the conditional failure rate for this action given the history prefix
        q_i = conditional_abstract_action_q_value(
            history_states,
            history_actions,
            action,
            trained_abstract_action_scorers,
        )
        targets.append(q_i)

    return targets


def train_q_network(
    q_network: PerActionQNetwork,
    abstract_plans: list[Skeleton],
    all_ground_atoms: tuple[GroundAtom, ...],
    all_ground_operators: tuple[GroundOperator, ...],
    trained_abstract_action_scorers: dict[GroundOperator, AbstractActionScorer],
    batch_size: int = 32,
    num_epochs: int = 10,
    verbose: bool = True,
) -> list[float]:
    """Train the PerActionQNetwork to predict per-action failure rates.

    The network learns to predict, for each action a_i in a plan, the failure
    rate for that action conditioned on the history (s_0, a_1, ..., a_{i-1}).

    Args:
        q_network: The PerActionQNetwork to train
        abstract_plans: List of abstract plans to use for training
        all_ground_atoms: All possible ground atoms in the environment
        all_ground_operators: All possible ground operators in the environment
        trained_abstract_action_scorers:
            Dictionary mapping abstract actions to trained abstract action scorers
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        verbose: Whether to log training progress

    Returns:
        List of average losses per epoch
    """
    # Compute per-action target Q-values for all abstract plans
    # Each plan produces a list of targets, one per action
    if verbose:
        logging.info("Computing per-action target Q-values...")

    per_action_targets = []  # List of lists, one per plan
    valid_plans = []

    for abstract_plan in abstract_plans:
        try:
            # Compute targets for each action in this plan
            targets = compute_per_action_targets(
                abstract_plan,
                trained_abstract_action_scorers,
            )
            per_action_targets.append(targets)
            valid_plans.append(abstract_plan)
        except Exception as e:
            if verbose:
                logging.info(f"Warning: Skipping plan due to error: {e}")
            continue

    if len(valid_plans) == 0:
        raise ValueError("No valid abstract plans for training")

    # Convert abstract plans to sequence embeddings
    if verbose:
        logging.info("Converting abstract plans to sequences...")

    sequences = []
    sequence_lengths = []
    for plan in valid_plans:
        sequence, seq_len = create_abstract_plan_sequence(
            all_ground_atoms, all_ground_operators, plan
        )
        sequences.append(torch.FloatTensor(sequence))
        sequence_lengths.append(seq_len)

    # Convert per-action targets to tensors
    # Each element is a tensor of shape (seq_len,) for that plan
    target_tensors = [torch.FloatTensor(t) for t in per_action_targets]

    # Training loop
    epoch_losses = []
    # Use reduction='none' so we can apply masking for variable-length sequences
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Create indices for shuffling
    num_samples = len(sequences)
    indices = np.arange(num_samples)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data
        np.random.shuffle(indices)

        # Train in batches
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]

            # Get batch sequences, targets, and lengths
            batch_sequences = [sequences[idx] for idx in batch_indices]
            batch_lengths = torch.tensor(
                [sequence_lengths[idx] for idx in batch_indices], dtype=torch.long
            )
            # batch_targets is a list of tensors with varying lengths
            batch_targets = [target_tensors[idx] for idx in batch_indices]

            loss = q_network.train_step(
                batch_sequences, batch_targets, batch_lengths, loss_fn
            )
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(avg_loss)

        if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}"
            )

    if verbose:
        logging.info("Training completed!")

    return epoch_losses


def convert_q_value_to_probability(
    failure_rate_per_action: list[float], num_retries: int
) -> float:
    """Given predicted per-action failure rates, return the probability that the
    abstract plan succeeds within num_retries total retries."""

    # Clamp failure rates to [0, 1].
    clamped_failure_rates = [
        min(max(failure_rate, 0.0), 1.0) for failure_rate in failure_rate_per_action
    ]

    # Per-action success probability = 1 - failure_rate.
    action_probs = [1.0 - failure_rate for failure_rate in clamped_failure_rates]

    K = num_retries
    # dp[k] = probability of being successful up to current action
    # using exactly k retries so far.
    dp = np.zeros(K + 1)

    # Base case: Before action 0, we have used 0 retries with prob 1.0
    dp[0] = 1.0

    for p in action_probs:
        new_dp = np.zeros(K + 1)
        # For each possible number of retries already spent (k_spent)
        for k_spent in range(K + 1):
            if dp[k_spent] == 0:
                continue

            # How many retries can we afford to spend on THIS action?
            k_rem = K - k_spent

            # Probability that this action succeeds within k_rem attempts
            # (Given we have k_rem + 1 total tries available for it)

            for k_this_action in range(k_rem + 1):
                # Prob of failing exactly k_this_action times then succeeding
                # P(F = j) = (1-p)^j * p
                prob_term = (1 - p) ** k_this_action * p

                new_dp[k_spent + k_this_action] += dp[k_spent] * prob_term
        dp = new_dp

    # The total success probability is the sum of probabilities of
    # finishing the last action within the total budget K.
    return float(np.sum(dp))


def calculate_bald_objective(ensemble_probabilities: list[float]) -> float:
    """Given an ensemble of probabilities of success for an abstract plan, return the
    BALD objective (epistemic uncertainty) for that plan."""

    eps = 1e-12

    def _binary_entropy(p: float) -> float:
        """Compute the binary entropy H(p) in bits, with numerical clipping."""
        p_clipped = float(np.clip(p, eps, 1.0 - eps))
        return float(entropy([p_clipped, 1.0 - p_clipped], base=2))

    avg_prob = float(np.average(ensemble_probabilities))
    overall_uncertainty = _binary_entropy(avg_prob)

    aleatoric_uncertainty = 0.0
    for prob in ensemble_probabilities:
        aleatoric_uncertainty += _binary_entropy(prob)
    aleatoric_uncertainty /= len(ensemble_probabilities)

    return overall_uncertainty - aleatoric_uncertainty
