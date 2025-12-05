import numpy as np
from collections import defaultdict
from alphatamp.structs import GroundOperator, Skeleton, RelationalAbstractState
from alphatamp.approaches.parameter_scorers.base_parameter_scorer import ParameterScorer
import torch.nn as nn

class QNetwork:
    def __init__(self):
        self._model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._model(x)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._optimizer.zero_grad()
        loss = self._model(features, labels)
        loss.backward()

def create_marginal_q_caches() -> tuple[dict[str, float], dict[str, int]]:
    """
    Create empty caches for marginal Q-values and dataset sizes.
    
    Returns:
        A tuple of (marginal_q_cache, dataset_size_cache) dictionaries
    """
    return ({}, {})


def generate_training_data(
        features_and_labels: list
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Reformat training data into numpy arrays."""

        features_list = []

        # Generate a row in the training dataset.
        for datapoint in features_and_labels:
            observation, parameter, _ = datapoint
            observation_arr = np.array(observation)
            parameter_arr = np.array(parameter)

            # The features are the state observation and the parameter.
            feature_arr = (observation_arr, parameter_arr)

            features_list.append(feature_arr)

        return features_list


def conditional_abstract_action_q_value(
    history_states: list[RelationalAbstractState],
    history_actions: list[GroundOperator],
    current_action: GroundOperator,
    trained_abstract_action_scorers: dict[GroundOperator, ParameterScorer],
) -> float:
    """
    Compute Q_abstract((s_{n-1}, a_{n-1}) | (s_0, a_1, s_1, a_2, ..., a_{n-2})).
    
    This is the conditional Q-value for the current state-action pair given the history.
    According to the approximation, this is computed by using the trained abstract action scorer.
    
    Args:
        history_states: List of abstract states (s_0, s_1, ..., s_{n-2})
        history_actions: List of abstract actions (a_1, a_2, ..., a_{n-2})
        current_action: The current abstract action a_{n-1}
        trained_abstract_action_scorers: Dictionary mapping abstract actions to trained abstract action scorers
    
    Returns:
        The conditional Q_abstract value for (s_{n-1}, a_{n-1}) given the history
    """
    # Compute Q_abstract((s_{n-1}, a_{n-1}) | history) using the trained abstract action scorer.
    # This approximates sum_phi Q_param(x_{n-1}, a_{n-1}, phi)
    
    abstract_action_scorer = trained_abstract_action_scorers[current_action]

    conditional_abstract_plan = (history_states, history_actions)

    conditional_abstract_plan_score = abstract_action_scorer.score(conditional_abstract_plan)
    return conditional_abstract_plan_score


def abstract_sequence_q_value(
    abstract_plan: Skeleton,
    q_value_cache: dict,
    gamma: float,
    parameter_dataset: defaultdict[str, list],
    trained_skill_scorers: dict[GroundOperator, ParameterScorer],
    marginal_q_cache: dict | None = None,
    dataset_size_cache: dict[str, int] | None = None
) -> float:
    """
    Recursively compute Q_abstract((s_0, a_1, s_1, a_2, ..., a_{n-1})).
    
    Implements the recursive update rule:
    Q_abstract((s_0, a_1, ..., a_{n-1})) <- γ * Q_abstract((s_0, a_1, ..., a_{n-1}))
                                         + (1-γ) * Q̂_abstract((s_0, a_1, ..., a_{n-1}))
    
    where:
    Q̂_abstract((s_0, a_1, ..., a_{n-1})) = Q_abstract((s_0, a_1, ..., a_{n-2}))
                                          + Q_abstract((s_{n-1}, a_{n-1}) | (s_0, a_1, ..., a_{n-2}))
    
    Args:
        abstract_plan: The abstract plan (Skeleton) containing states and actions
        q_value_cache: Dictionary to cache Q-values for sequences (keyed by tuple representation)
        gamma: Discount factor γ for the update rule
        parameter_dataset: Dataset mapping abstract action descriptors to parameter data
        trained_skill_scorers: Dictionary mapping abstract actions to trained parameter scorers
        marginal_q_cache: Optional cache dictionary for marginal Q-values (keyed by action descriptor)
        dataset_size_cache: Optional cache dictionary tracking dataset sizes (keyed by action descriptor)
    
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
    
    # Compute Q̂_abstract recursively
    if len(actions) == 1:
        # Base case: single action, Q_abstract for empty prefix is 0
        q_prefix = 0.0
    else:
        # Recursive case: compute Q_abstract for shorter sequence
        prefix_states = states[:-1]  # s_0, s_1, ..., s_{n-2}
        prefix_actions = actions[:-1]  # a_1, a_2, ..., a_{n-2}
        prefix_abstract_plan = (prefix_states, prefix_actions)
        q_prefix = abstract_sequence_q_value(
            prefix_abstract_plan, q_value_cache, gamma,
            parameter_dataset, trained_skill_scorers,
            marginal_q_cache, dataset_size_cache
        )
    
    # Compute conditional Q-value: Q_abstract((s_{n-1}, a_{n-1}) | history)
    current_state = states[-1]  # s_{n-1}
    current_action = actions[-1]  # a_{n-1}
    history_states = states[:-1]  # s_0, s_1, ..., s_{n-2}
    history_actions = actions[:-1]  # a_1, a_2, ..., a_{n-2}
    
    conditional_q_value = conditional_abstract_action_q_value(
        history_states, history_actions, current_state, current_action,
        parameter_dataset, trained_skill_scorers,
        marginal_q_cache, dataset_size_cache
    )
    
    # Compute Q̂_abstract
    q_hat = q_prefix + conditional_q_value
    
    # Update Q-value using the update rule
    updated_q_value = gamma * current_q_value + (1 - gamma) * q_hat
    
    # Cache the updated value
    q_value_cache[sequence_key] = updated_q_value
    
    return updated_q_value


def estimated_abstract_plan_q_value(abstract_plan: Skeleton, q_function: QNetwork):
    """Estimate the Q-value for an abstract plan."""
    pass
    