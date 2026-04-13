"""A bootstrapped ensemble parameter scorer trained with pairwise ranking loss.

Each head is an independent MLP trained with `nn.MarginRankingLoss` on the same
pool of (positive, negative) pairs, but with bootstrap-resampled pairs and a
different random initialization per head. The ensemble supports Thompson
sampling: callers invoke `sample_head(rng)` once per decision, then subsequent
`score()` calls route through the selected head. `exploit()` picks head 0
deterministically for evaluation.

The `score()` output is an unbounded real number; only the relative ordering
under one sampled head matters, which `ParameterPolicy.sample_parameters`
consumes via argmax.
"""

from typing import Any, TypeVar

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from alphatamp.approaches.scorers.base_scorer import BaseScorer

_O = TypeVar("_O")  # observation


class RankingParameterScorer(BaseScorer):
    """Bootstrapped ensemble of siamese MLPs trained with margin ranking."""

    def __init__(self, configs: dict):
        self._hidden_layer_sizes: tuple[int, ...] = tuple(
            configs["hidden_layer_sizes"]
        )
        self._epochs: int = int(configs.get("epochs", 300))
        self._lr: float = float(configs.get("lr", 1e-3))
        self._margin: float = float(configs.get("margin", 1.0))
        self._weight_decay: float = float(configs.get("weight_decay", 1e-4))
        self._batch_cap: int = int(configs.get("max_pairs_per_fit", 20000))
        self._num_heads: int = int(configs.get("num_heads", 5))
        self._bootstrap: bool = bool(configs.get("bootstrap", True))

        self._nets: list[nn.Module] = []
        self._scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False
        self._active_head: int = 0
        self._rng_np: np.random.Generator = np.random.default_rng(
            configs.get("seed")
        )
        # Accumulated per-epoch losses across training fits, exposed for
        # logging (mirrors sklearn MLPClassifier.loss_curve_). Interleaves
        # all K heads' losses per fit in head order.
        self.loss_curve_: list[float] = []

    def _make_net(self, in_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in self._hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _build_ensemble(self, in_dim: int) -> None:
        # Different torch seed per head → different init → disagreement.
        self._nets = []
        for _ in range(self._num_heads):
            torch.manual_seed(int(self._rng_np.integers(0, 2**31 - 1)))
            self._nets.append(self._make_net(in_dim))

    def train(  # type: ignore[override]
        self,
        pos_features: np.ndarray | None,
        neg_features: np.ndarray | None,
    ) -> None:
        """Fit every ensemble head on bootstrap resamples of the same pairs.

        Row `i` of `pos_features` should score higher than row `i` of
        `neg_features`. Passing `None` or empty arrays is a no-op — the
        ensemble keeps whatever state it had.
        """
        if (
            pos_features is None
            or neg_features is None
            or len(pos_features) == 0
            or len(neg_features) == 0
        ):
            return
        assert len(pos_features) == len(neg_features)

        # Cap before bootstrap so the training cost is bounded.
        if len(pos_features) > self._batch_cap:
            idx = self._rng_np.choice(
                len(pos_features), size=self._batch_cap, replace=False
            )
            pos_features = pos_features[idx]
            neg_features = neg_features[idx]

        all_feats = np.vstack([pos_features, neg_features])
        self._scaler.fit(all_feats)
        pos_all = self._scaler.transform(pos_features)
        neg_all = self._scaler.transform(neg_features)

        if not self._nets:
            self._build_ensemble(pos_all.shape[1])

        loss_fn = nn.MarginRankingLoss(margin=self._margin)
        n_pairs = len(pos_all)

        for net in self._nets:
            if self._bootstrap:
                boot = self._rng_np.integers(0, n_pairs, size=n_pairs)
            else:
                boot = np.arange(n_pairs)
            pos_t = torch.as_tensor(pos_all[boot], dtype=torch.float32)
            neg_t = torch.as_tensor(neg_all[boot], dtype=torch.float32)
            target = torch.ones(pos_t.shape[0], dtype=torch.float32)
            opt = torch.optim.Adam(
                net.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )
            net.train()
            for _ in range(self._epochs):
                opt.zero_grad()
                s_pos = net(pos_t).squeeze(-1)
                s_neg = net(neg_t).squeeze(-1)
                loss = loss_fn(s_pos, s_neg, target)
                loss.backward()
                opt.step()
                self.loss_curve_.append(float(loss.detach().item()))

        self._fitted = True

    def sample_head(self, rng: np.random.Generator) -> None:
        """Thompson-draw: pick one ensemble head uniformly at random.

        No-op if the ensemble hasn't been trained yet (in which case `score()`
        returns a constant and argmax is arbitrary — acceptable cold-start
        behaviour).
        """
        if not self._fitted or not self._nets:
            return
        self._active_head = int(rng.integers(0, len(self._nets)))

    def exploit(self) -> None:
        """Pick a deterministic head for evaluation / exploitation.

        Using head 0 is arbitrary but consistent; what matters is that all
        decisions within one eval run go through the same head so their
        scores are directly comparable.
        """
        self._active_head = 0

    def score(self, obs: _O, parameter: Any) -> float:
        if not self._fitted or not self._nets:
            return 0.0
        state_arr = np.array(obs, dtype=np.float64)
        parameter_arr = np.array(parameter, dtype=np.float64)
        feat = np.append(state_arr, parameter_arr).reshape(1, -1)
        feat = self._scaler.transform(feat)
        net = self._nets[self._active_head]
        net.eval()
        with torch.no_grad():
            out = net(torch.as_tensor(feat, dtype=torch.float32))
        return float(out.item())
