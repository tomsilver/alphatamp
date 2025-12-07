"""An abstract action scorer that uses a LSTM for scoring."""

from torch import Tensor, nn

from alphatamp.approaches.abstract_plan_classifiers.q_network import QNetwork
from alphatamp.structs import Skeleton


class AbstractActionScorer:
    """A abstract action scorer that uses a MLP for scoring."""

    def __init__(self, configs: dict):
        self._regressor = QNetwork(
            input_dim=configs["input_dim"],
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
    ):
        """Given training data, update scorer."""
        for _ in range(self._num_epochs):
            self._regressor.train_step(features, targets, lengths, loss_fn)

    def score(self, previous_abstract_plan: Skeleton) -> float:
        """Score the action given the previous abstract plan."""

        # Score should be a positive number (number of predicted resamples)
        return max(0, self._regressor.predict(previous_abstract_plan))
