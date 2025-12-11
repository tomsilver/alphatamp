"""Q network that returns how feasible an abstract plan might be."""

import numpy as np
import torch
from relational_structs.objects import Object
from relational_structs.pddl import GroundAtom, GroundOperator
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from alphatamp.approaches.simulator_free_base_approach import SimulatorFreeSesameModels
from alphatamp.structs import FrozenSkeleton, RelationalAbstractState, Skeleton


def create_abstract_state_embedding(
    all_ground_atoms: tuple[GroundAtom, ...], abstract_state: RelationalAbstractState
) -> np.ndarray:
    """Create an abstract state embedding that contains information about the present
    ground atoms."""

    abstract_state_embedding = np.zeros(len(all_ground_atoms))

    for index, ground_atom in enumerate(all_ground_atoms):
        if ground_atom in abstract_state.atoms:
            abstract_state_embedding[index] = 1

    return abstract_state_embedding


def create_abstract_actions_embedding(
    all_ground_operators: tuple[GroundOperator, ...], abstract_action: GroundOperator
):
    """Create an abstract action embedding that contains information about which ground
    operator is present."""

    abstract_action_embedding = np.zeros(len(all_ground_operators))

    for index, ground_operator in enumerate(all_ground_operators):
        if ground_operator == abstract_action:
            abstract_action_embedding[index] = 1
            break

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
            state_action_embedding = np.concatenate(
                [
                    create_abstract_state_embedding(all_ground_atoms, states[0]),
                    np.zeros(len(all_ground_operators)),
                ]
            )
            return (state_action_embedding, 1)
        return np.array([[0.0, 0.0]], dtype=np.float32), 1

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
    """Q network that returns how feasible an abstract plan might be."""

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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lstm.to(self._device)
        self._fc.to(self._device)
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
            Output tensor of shape (batch_size, 1)
        """
        # Pack padded sequences
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        _, (h_n, _) = self._lstm(packed)

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
            sequence, seq_len = create_abstract_plan_sequence(
                self._all_ground_atoms, self._all_ground_operators, abstract_plan
            )

            # Convert to tensor and add batch dimension
            x = (
                torch.FloatTensor(sequence).unsqueeze(0).to(self._device)
            )  # (1, seq_len, input_dim)
            lengths = torch.tensor([seq_len], dtype=torch.long)

            prediction = self.forward(x, lengths)
            return prediction.cpu().item()

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

        # Pad sequences to the same length
        padded_features = pad_sequence(
            features, batch_first=True
        )  # (batch_size, max_len, input_dim)
        padded_features = padded_features.to(self._device)

        predictions = self.forward(padded_features, lengths)
        loss = loss_fn(predictions, targets)
        loss.backward()
        self._optimizer.step()

        return loss.item()
