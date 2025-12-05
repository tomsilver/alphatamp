from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from alphatamp.approaches.scorers.regressor_abstract_action_scorer import AbstractActionScorer
from alphatamp.structs import GroundOperator, RelationalAbstractState, Skeleton


def create_abstract_plan_sequence(abstract_plan: Skeleton, feature_dim: int = 2) -> tuple[np.ndarray, int]:
    """Create a sequence embedding for an abstract plan.
    
    Each timestep represents a state-action pair. The sequence has shape (seq_len, feature_dim)
    where each row is [hash(state), hash(action)].
    
    Args:
        abstract_plan: The abstract plan (Skeleton) containing states and actions
        feature_dim: Dimension of features per timestep (default: 2 for state+action)
    
    Returns:
        A tuple of (sequence array of shape (seq_len, feature_dim), sequence length)
    """
    states, actions = abstract_plan
    # Each timestep is a state-action pair
    # We need at least one state (initial state), and actions correspond to transitions
    seq_len = len(actions)  # Number of actions = number of timesteps
    if seq_len == 0:
        # Empty plan - return a single timestep with initial state
        if len(states) > 0:
            return np.array([[hash(states[0]), 0.0]], dtype=np.float32), 1
        return np.array([[0.0, 0.0]], dtype=np.float32), 1
    
    sequence = []
    for i in range(seq_len):
        # For each action, use the state before the action and the action itself
        state_idx = min(i, len(states) - 1)
        state_hash = hash(states[state_idx])
        action_hash = hash(actions[i])
        sequence.append([float(state_hash), float(action_hash)])
    
    return np.array(sequence, dtype=np.float32), seq_len


class QNetwork:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2, lr: float = 0.001):
        """Initialize the Q-network with LSTM.
        
        Args:
            input_dim: Dimension of input features per timestep (default: 2 for state+action)
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            lr: Learning rate for optimizer
        """
        self._lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_dim, 1)
        self._optimizer = torch.optim.Adam(
            list(self._lstm.parameters()) + list(self._fc.parameters()), 
            lr=lr
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lstm.to(self._device)
        self._fc.to(self._device)
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Padded input tensor of shape (batch_size, max_seq_len, input_dim)
            lengths: Tensor of actual sequence lengths, shape (batch_size,)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Pack padded sequences
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self._lstm(packed)
        
        # Use the last hidden state from the last layer
        # h_n shape: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]  # Take the last layer, shape: (batch_size, hidden_dim)
        
        # Pass through fully connected layer
        output = self._fc(last_hidden)  # Shape: (batch_size, 1)
        
        return output

    def predict(self, abstract_plan: Skeleton) -> float:
        """Predict Q-value for a single abstract plan.
        
        Args:
            abstract_plan: The abstract plan to predict Q-value for
        
        Returns:
            Predicted Q-value as a float
        """
        self._lstm.eval()
        self._fc.eval()
        with torch.no_grad():
            sequence, seq_len = create_abstract_plan_sequence(abstract_plan, self._input_dim)
            
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self._device)  # (1, seq_len, input_dim)
            lengths = torch.tensor([seq_len], dtype=torch.long)
            
            prediction = self.forward(x, lengths)
            return prediction.cpu().item()
    
    def train_step(
        self, 
        features: list[torch.Tensor], 
        targets: torch.Tensor,
        lengths: torch.Tensor,
        loss_fn: nn.Module
    ) -> float:
        """Perform a single training step.
        
        Args:
            features: List of sequence tensors, each of shape (seq_len, input_dim)
            targets: Target Q-values tensor of shape (batch_size, 1)
            lengths: Tensor of actual sequence lengths, shape (batch_size,)
            loss_fn: Loss function
        
        Returns:
            The loss value
        """
        assert loss_fn is not None
        self._lstm.train()
        self._fc.train()
        self._optimizer.zero_grad()
        
        # Pad sequences to the same length
        padded_features = pad_sequence(features, batch_first=True)  # (batch_size, max_len, input_dim)
        padded_features = padded_features.to(self._device)
        
        predictions = self.forward(padded_features, lengths)
        loss = loss_fn(predictions, targets)
        loss.backward()
        self._optimizer.step()
        
        return loss.item()


def create_marginal_q_caches() -> tuple[dict[str, float], dict[str, int]]:
    """Create empty caches for marginal Q-values and dataset sizes.

    Returns:
        A tuple of (marginal_q_cache, dataset_size_cache) dictionaries
    """
    return ({}, {})


def generate_training_data(
    features_and_labels: list,
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
    trained_abstract_action_scorers: dict[GroundOperator, AbstractActionScorer],
) -> float:
    """Compute Q_abstract((s_{n-1}, a_{n-1}) | (s_0, a_1, s_1, a_2, ..., a_{n-2})).

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
    Q_abstract((s_0, a_1, ..., a_{n-1})) <- γ * Q_abstract((s_0, a_1, ..., a_{n-1}))
                                         + (1-γ) * Q̂_abstract((s_0, a_1, ..., a_{n-1}))

    where:
    Q̂_abstract((s_0, a_1, ..., a_{n-1})) = Q_abstract((s_0, a_1, ..., a_{n-2}))
                                          + Q_abstract((s_{n-1}, a_{n-1}) | (s_0, a_1, ..., a_{n-2}))

    Args:
        abstract_plan: The abstract plan (Skeleton) containing states and actions
        q_value_cache: Dictionary to cache Q-values for sequences (keyed by tuple representation)
        gamma: Discount factor γ for the update rule
        trained_skill_scorers: Dictionary mapping abstract actions to trained parameter scorers

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


def estimated_abstract_plan_q_value(abstract_plan: Skeleton, q_network: QNetwork) -> float:
    """Estimate the Q-value for an abstract plan using the trained Q-network.
    
    Args:
        abstract_plan: The abstract plan (Skeleton) to estimate Q-value for
        q_network: The trained Q-network
    
    Returns:
        The estimated Q-value
    """
    return q_network.predict(abstract_plan)


def train_q_network(
    q_network: QNetwork,
    abstract_plans: list[Skeleton],
    q_value_cache: dict,
    gamma: float,
    trained_abstract_action_scorers: dict[GroundOperator, AbstractActionScorer],
    batch_size: int = 32,
    num_epochs: int = 10,
    verbose: bool = True,
) -> list[float]:
    """Train the Q-network to regress onto abstract_plan_q_value.
    
    Args:
        q_network: The Q-network to train
        abstract_plans: List of abstract plans to use for training
        q_value_cache: Dictionary to cache Q-values for sequences
        gamma: Discount factor for the update rule
        trained_abstract_action_scorers: Dictionary mapping abstract actions to trained abstract action scorers
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        verbose: Whether to print training progress
    
    Returns:
        List of average losses per epoch
    """
    # Compute target Q-values for all abstract plans
    if verbose:
        print("Computing target Q-values...")
    
    target_q_values = []
    valid_plans = []
    
    for abstract_plan in abstract_plans:
        try:
            target_q = abstract_plan_q_value(
                abstract_plan,
                q_value_cache,
                gamma,
                trained_abstract_action_scorers,
            )
            target_q_values.append(target_q)
            valid_plans.append(abstract_plan)
        except Exception as e:
            if verbose:
                print(f"Warning: Skipping plan due to error: {e}")
            continue
    
    if len(valid_plans) == 0:
        raise ValueError("No valid abstract plans for training")
    
    # Convert abstract plans to sequence embeddings
    if verbose:
        print("Converting abstract plans to sequences...")
    
    sequences = []
    sequence_lengths = []
    for plan in valid_plans:
        sequence, seq_len = create_abstract_plan_sequence(plan, q_network._input_dim)
        sequences.append(torch.FloatTensor(sequence))
        sequence_lengths.append(seq_len)
    
    # Convert targets to tensor
    targets_tensor = torch.FloatTensor(np.array(target_q_values)).unsqueeze(1).to(q_network._device)
    
    # Training loop
    if verbose:
        print(f"Training Q-network on {len(valid_plans)} plans for {num_epochs} epochs...")
    
    epoch_losses = []
    loss_fn = nn.MSELoss()
    
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
            batch_indices = indices[i:i + batch_size]
            
            # Get batch sequences and lengths
            batch_sequences = [sequences[idx] for idx in batch_indices]
            batch_lengths = torch.tensor([sequence_lengths[idx] for idx in batch_indices], dtype=torch.long)
            batch_targets = targets_tensor[batch_indices]
            
            loss = q_network.train_step(batch_sequences, batch_targets, batch_lengths, loss_fn)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(avg_loss)
        
        if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
    
    if verbose:
        print("Training completed!")
    
    return epoch_losses
