"""SPECTRE: Skeleton-Pool Embedding with Contextual Transformer for REordering.

A learned listwise re-ranker for bilevel-TAMP skeleton pools. See ``docs/proposal.md``
for the current method and evaluation direction.
"""

from alphatamp.approaches.spectre.inference import (
    Trace,
    deployed_rollout,
    deployed_rollout_traced,
    load_checkpoint,
)
from alphatamp.approaches.spectre.loss import (
    plackett_luce_loss,
    within_length_pl_loss,
)
from alphatamp.approaches.spectre.model import (
    SpectreBatch,
    SpectreConfig,
    SpectreModel,
)

__all__ = [
    "SpectreBatch",
    "SpectreConfig",
    "SpectreModel",
    "Trace",
    "deployed_rollout",
    "deployed_rollout_traced",
    "load_checkpoint",
    "plackett_luce_loss",
    "within_length_pl_loss",
]
