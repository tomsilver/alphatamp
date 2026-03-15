"""Q network that returns how feasible an abstract plan might be."""

import numpy as np
import torch
from relational_structs.pddl import GroundAtom, GroundOperator
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from alphatamp.structs import FrozenSkeleton, RelationalAbstractState, Skeleton


def create_abstract_state_embedding(
    all_ground_atoms: tuple[GroundAtom, ...], abstract_state: RelationalAbstractState
) -> np.ndarray:
    """Create an abstract state embedding that contains information about the present
    ground atoms."""

    abstract_state_embedding = np.zeros(len(all_ground_atoms))

    for active_ground_atom in abstract_state.atoms:
        active_ground_atom_index = all_ground_atoms.index(active_ground_atom)
        abstract_state_embedding[active_ground_atom_index] = 1

    return abstract_state_embedding


def create_abstract_actions_embedding(
    all_ground_operators: tuple[GroundOperator, ...], abstract_action: GroundOperator
):
    """Create an abstract action embedding that contains information about which ground
    operator is present."""

    abstract_action_embedding = np.zeros(len(all_ground_operators))

    active_ground_operator_index = all_ground_operators.index(abstract_action)
    abstract_action_embedding[active_ground_operator_index] = 1

    return abstract_action_embedding


def create_abstract_plan_sequence(
    all_ground_atoms: tuple[GroundAtom, ...],
    all_ground_operators: tuple[GroundOperator, ...],
    abstract_plan: Skeleton | FrozenSkeleton,
) -> tuple[np.ndarray, int]:
    """Create a sequence embedding for an abstract plan.

    Each timestep represents a state-action pair.
    The sequence has shape (seq_len, 2)
    where each row is [hash(state), hash(action)].

    Args:
        abstract_plan: The abstract plan (Skeleton) containing states and actions

    Returns:
        A tuple of (sequence array of shape (seq_len, 2), sequence length)
    """
    states, actions = abstract_plan

    if isinstance(states, tuple):
        states = list(states)
    if isinstance(actions, tuple):
        actions = list(actions)

    # Each timestep is a state-action pair
    # We need at least one state (initial state), and actions correspond to transitions
    seq_len = len(actions)  # Number of actions = number of timesteps
    if seq_len == 0:
        # Empty plan - return a single timestep with initial state
        if len(states) > 0:
            state_action_embedding = (
                np.concatenate(
                    [
                        create_abstract_state_embedding(all_ground_atoms, states[0]),
                        np.zeros(len(all_ground_operators)),
                    ]
                )
                .reshape(1, -1)
                .astype(np.float32)
            )
            return (state_action_embedding, 1)
        return (
            np.zeros(
                (1, len(all_ground_atoms) + len(all_ground_operators)), dtype=np.float32
            ),
            1,
        )

    sequence = []
    for i in range(seq_len):
        # For each action, use the state before the action and the action itself
        state_idx = min(i, len(states) - 1)
        state_embedding = create_abstract_state_embedding(
            all_ground_atoms, states[state_idx]
        )
        action_embedding = create_abstract_actions_embedding(
            all_ground_operators, actions[i]
        )

        state_action_embedding = np.concatenate([state_embedding, action_embedding])
        sequence.append(state_action_embedding)

    return np.array(sequence, dtype=np.float32), seq_len


class QNetwork:
    """Q network that outputs a single scalar failure rate in (0, 1) for an abstract plan.

    Used by AbstractActionScorer to predict the failure rate for the final action in a
    plan, conditioned on the full history. The network outputs raw logits; sigmoid is
    applied at prediction time to obtain a probability.
    """

    def __init__(
        self,
        all_ground_atoms: tuple[GroundAtom, ...],
        all_ground_operators: tuple[GroundOperator, ...],
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
    ):
        """Initialize the Q-network with LSTM.

        Args:
            all_ground_atoms: Tuple of all possible ground atoms in env
            all_ground_operators: Tuple of all possible ground operators in env
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            lr: Learning rate for optimizer
        """
        input_dim = len(all_ground_atoms) + len(all_ground_operators)
        self._lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_dim, 1)
        self._optimizer = torch.optim.Adam(
            list(self._lstm.parameters()) + list(self._fc.parameters()), lr=lr
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lstm.to(self.device)
        self._fc.to(self.device)
        self._input_dim = input_dim

        # Abstract plan embeddings
        self._all_ground_atoms = all_ground_atoms
        self._all_ground_operators = all_ground_operators

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Padded input tensor of shape (batch_size, max_seq_len, input_dim)
            lengths: Tensor of actual sequence lengths, shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, 1) - a single scalar per sequence
        """
        # Pack padded sequences
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass - only need the final hidden state
        _, (h_n, _) = self._lstm(packed)

        # Use the last hidden state from the last layer
        # h_n shape: (num_layers, batch_size, hidden_dim)
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Pass through fully connected layer
        output = self._fc(last_hidden)  # (batch_size, 1)

        return output

    def predict(self, abstract_plan: Skeleton) -> float:
        """Predict failure rate for a single abstract plan.

        Args:
            abstract_plan: The abstract plan to predict for

        Returns:
            Predicted failure rate as a float in (0, 1)
        """
        self._lstm.eval()
        self._fc.eval()
        with torch.no_grad():
            sequence, seq_len = create_abstract_plan_sequence(
                self._all_ground_atoms, self._all_ground_operators, abstract_plan
            )

            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            lengths = torch.tensor([seq_len], dtype=torch.long)

            prediction = self.forward(x, lengths)
            return torch.sigmoid(prediction).cpu().item()

    def train_step(
        self,
        features: list[torch.Tensor],
        targets: torch.Tensor,
        lengths: torch.Tensor,
        loss_fn: nn.Module,
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

        # Ensure every feature tensor is at least 2D (seq_len, input_dim)
        features_3d = [f.unsqueeze(0) if f.ndim == 1 else f for f in features]

        # Pad sequences to the same length
        padded_features = pad_sequence(features_3d, batch_first=True).to(self.device)

        predictions = self.forward(padded_features, lengths)
        loss = loss_fn(predictions, targets)
        loss.backward()
        self._optimizer.step()

        return loss.item()


class PerActionQNetwork:
    """Q network that outputs a vector of failure rates in (0, 1), one per action in the plan.

    Each element i in the output represents the predicted failure rate for action a_i,
    conditioned on the prior actions (s_0, a_1, ..., a_{i-1}). The network outputs raw
    logits; sigmoid is applied at prediction time to obtain probabilities.
    """

    def __init__(
        self,
        all_ground_atoms: tuple[GroundAtom, ...],
        all_ground_operators: tuple[GroundOperator, ...],
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 0.001,
    ):
        """Initialize the per-action Q-network with LSTM.

        Args:
            all_ground_atoms: Tuple of all possible ground atoms in env
            all_ground_operators: Tuple of all possible ground operators in env
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            lr: Learning rate for optimizer
        """
        input_dim = len(all_ground_atoms) + len(all_ground_operators)
        self._lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_dim, 1)
        self._optimizer = torch.optim.Adam(
            list(self._lstm.parameters()) + list(self._fc.parameters()), lr=lr
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lstm.to(self.device)
        self._fc.to(self.device)
        self._input_dim = input_dim

        # Abstract plan embeddings
        self._all_ground_atoms = all_ground_atoms
        self._all_ground_operators = all_ground_operators

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Padded input tensor of shape (batch_size, max_seq_len, input_dim)
            lengths: Tensor of actual sequence lengths, shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, max_seq_len, 1)
            Each position i gives the predicted resamples for action a_i
            conditioned on the history (s_0, a_1, ..., a_{i-1}).
        """
        # Pack padded sequences for efficient LSTM processing
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass - get outputs at ALL timesteps, not just final
        # lstm_out contains the hidden state for each timestep
        # At timestep i, the hidden state has processed (s_0, a_1, ..., a_i)
        lstm_out, _ = self._lstm(packed)

        # Unpack to get padded tensor of all hidden states
        # lstm_out_padded: (batch_size, max_seq_len, hidden_dim)
        lstm_out_padded, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )

        # Apply FC layer to EACH timestep's hidden state independently
        # This gives us a prediction for each action in the sequence
        # output: (batch_size, max_seq_len, 1)
        output = self._fc(lstm_out_padded)

        return output

    def predict(self, abstract_plan: Skeleton) -> np.ndarray:
        """Predict per-action failure rates for an abstract plan.

        Args:
            abstract_plan: The abstract plan to predict for

        Returns:
            Array of shape (seq_len,) where each element is the predicted failure rate
            in (0, 1) for that action conditioned on prior actions.
            Element i = predicted failure rate for action a_i given (s_0, a_1, ..., a_{i-1})
        """
        self._lstm.eval()
        self._fc.eval()
        with torch.no_grad():
            sequence, seq_len = create_abstract_plan_sequence(
                self._all_ground_atoms, self._all_ground_operators, abstract_plan
            )

            # Convert to tensor and add batch dimension
            # x: (1, seq_len, input_dim)
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            lengths = torch.tensor([seq_len], dtype=torch.long)

            # output: (1, seq_len, 1)
            output = self.forward(x, lengths)

            # Apply sigmoid to convert logits to failure rates, then flatten to (seq_len,)
            return torch.sigmoid(output).squeeze(0).squeeze(-1).cpu().numpy()

    def train_step(
        self,
        features: list[torch.Tensor],
        targets: list[torch.Tensor],
        lengths: torch.Tensor,
        loss_fn: nn.Module,
    ) -> float:
        """Perform a single training step with per-action targets.

        Args:
            features: List of sequence tensors, each of shape (seq_len, input_dim)
            targets: List of per-action target tensors, each of shape (seq_len,)
                     where each element is the target failure rate in [0, 1] for that action
            lengths: Tensor of actual sequence lengths, shape (batch_size,)
            loss_fn: Loss function (should be nn.BCEWithLogitsLoss with reduction='none')

        Returns:
            The loss value
        """
        assert loss_fn is not None
        self._lstm.train()
        self._fc.train()
        self._optimizer.zero_grad()

        # Ensure every feature tensor is at least 2D (seq_len, input_dim)
        # This handles the edge case where a sequence of length 1 might be passed as 1D
        features_3d = [f.unsqueeze(0) if f.ndim == 1 else f for f in features]

        # Pad sequences to the same length
        # padded_features: (batch_size, max_seq_len, input_dim)
        padded_features = pad_sequence(features_3d, batch_first=True).to(self.device)

        # predictions: (batch_size, max_seq_len, 1)
        predictions = self.forward(padded_features, lengths)

        # Pad target sequences to match predictions shape
        # Each target is (seq_len,), we need to add a dimension and pad
        # targets_with_dim: list of (seq_len, 1) tensors
        targets_with_dim = [t.unsqueeze(-1) for t in targets]
        # padded_targets: (batch_size, max_seq_len, 1)
        padded_targets = pad_sequence(targets_with_dim, batch_first=True).to(
            self.device
        )

        # Create a mask to ignore padded positions in the loss
        # We only want to compute loss on valid (non-padded) positions
        # mask: (batch_size, max_seq_len, 1)
        max_seq_len = padded_features.shape[1]
        mask = torch.zeros(len(features), max_seq_len, 1, device=self.device)
        for i, length in enumerate(lengths):
            # Convert tensor lengths to Python ints so
            # they can be used as slice indices
            if isinstance(length, torch.Tensor):
                length = int(length.item())
            else:
                length = int(length)

            mask[i, :length, :] = 1.0

        # Compute element-wise loss and apply mask
        # This ensures padded positions don't contribute to the loss
        elementwise_loss = loss_fn(predictions, padded_targets)
        masked_loss = (elementwise_loss * mask).sum() / mask.sum()

        masked_loss.backward()
        self._optimizer.step()

        return masked_loss.item()
