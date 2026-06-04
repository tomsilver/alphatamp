"""SPECTRE: Skeleton-Pool Embedding with Contextual Transformer for REordering.

Data pipeline + model subpackage. See ``docs/archive/SPECTRE_RT2D_METHOD_SPEC.md``
for the RT2D-specific method and training spec;
``docs/archive/SPECTRE_METHOD_SPEC.md`` for the original (kinder-env) method
spec; and ``docs/archive/SPECTRE_TRAINING_PIPELINE_SPEC.md`` for the
data-collection pipeline motivation.
"""

from alphatamp.approaches.spectre.inference import (
    InferenceState,
    init_inference_state,
    record_failure,
    select_next_skeleton,
)
from alphatamp.approaches.spectre.loss import plackett_luce_loss
from alphatamp.approaches.spectre.model import SpectreModel

__all__ = [
    "InferenceState",
    "SpectreModel",
    "init_inference_state",
    "plackett_luce_loss",
    "record_failure",
    "select_next_skeleton",
]
