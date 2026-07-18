"""Per-problem static tags: passage widths and item sizes (spec §3.3).

Width and size are sampled per-problem from the default distributions in
spec §3.3 and exposed in s_0 as static atoms ``PassageWidth(p, w)`` and
``ItemSize(i, s)``. They never appear in operator preconditions or effects, so
the canonical skeleton key (B4's signature) is independent of tag values —
the mechanism by which v1 confounds B4 (spec §3.5).
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np

WidthLevel = Literal["narrow", "medium", "wide"]
SizeLevel = Literal["small", "medium", "large"]

WIDTH_LEVELS: Final[tuple[WidthLevel, ...]] = ("narrow", "medium", "wide")
SIZE_LEVELS: Final[tuple[SizeLevel, ...]] = ("small", "medium", "large")

# Total order for compatibility: size <= width iff
# SIZE_ORDER[size] <= WIDTH_ORDER[width] (spec §3.3 compatibility table).
SIZE_ORDER: Final[dict[str, int]] = {"small": 0, "medium": 1, "large": 2}
WIDTH_ORDER: Final[dict[str, int]] = {"narrow": 0, "medium": 1, "wide": 2}

# Default distributions (spec §3.3).
DEFAULT_WIDTH_PROBS: Final[tuple[float, ...]] = (0.20, 0.40, 0.40)
DEFAULT_SIZE_PROBS: Final[tuple[float, ...]] = (0.30, 0.40, 0.30)


def is_compatible(size: str, width: str) -> bool:
    """Return True iff a size-``size`` item fits through a width-``width`` passage."""
    return SIZE_ORDER[size] <= WIDTH_ORDER[width]


def sample_width(rng: np.random.Generator) -> WidthLevel:
    """Draw a width level from the default distribution."""
    idx = int(rng.choice(3, p=DEFAULT_WIDTH_PROBS))
    return WIDTH_LEVELS[idx]


def sample_size(rng: np.random.Generator) -> SizeLevel:
    """Draw a size level from the default distribution."""
    idx = int(rng.choice(3, p=DEFAULT_SIZE_PROBS))
    return SIZE_LEVELS[idx]
