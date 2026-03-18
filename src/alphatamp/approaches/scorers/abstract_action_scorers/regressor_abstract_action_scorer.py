"""An abstract action scorer that uses a LSTM for scoring."""

from torch import Tensor, nn

from alphatamp.approaches.abstract_plan_classifiers.q_network import QNetwork
from alphatamp.structs import Skeleton


class AbstractActionScorer:
    """A abstract action scorer that uses a LSTM for scoring."""

    def __init__(self, all_ground_atoms, all_ground_operators, configs: dict):
        self._regressor = QNetwork(
            all_ground_atoms,
            all_ground_operators,
            hidden_dim=configs["hidden_dim"],
            num_layers=configs["num_layers"],
        )

        self._num_epochs = configs["num_epochs"]

    def train(
        self,
        features: list,
        targets: Tensor,
        lengths: Tensor,
        loss_fn: nn.Module,
    ) -> list[float]:
        """Given training data, update scorer."""
        losses = []
        for _ in range(self._num_epochs):
            losses.append(
                self._regressor.train_step(features, targets, lengths, loss_fn)
            )
        return losses

    def score(self, previous_abstract_plan: Skeleton) -> float:
        """Score the action given the previous abstract plan."""

        # Score is a predicted failure rate in [0, 1]
        return min(max(self._regressor.predict(previous_abstract_plan), 0.0), 1.0)
