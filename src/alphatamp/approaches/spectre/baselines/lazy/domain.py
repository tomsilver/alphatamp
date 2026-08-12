"""Per-environment configuration for the LAZY baseline (mirrors ``piginet``'s adapters).

An environment differs only in its geometry normalisers and which collection it reads;
the tree/graph/model/feasibility/rollout code is env-agnostic. ``make_lazy_domain``
dispatches on ``env_variant`` prefix, exactly like
``piginet.sb2d_adapter.make_sb2d_domain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# domain.py is 6 dirs below the repo root (.../baselines/lazy/domain.py).
REPO = Path(__file__).resolve().parents[6]
DEFAULT_DATA_ROOT = REPO / "data" / "spectre"


@dataclass(frozen=True)
class LazyDomain:
    """Geometry scales + data locations for one env-variant."""

    name: str
    env_variant: str
    frame_extent: tuple[float, float]
    shape_max: np.ndarray
    data_root: Path = DEFAULT_DATA_ROOT

    def split_dir(self, split: str) -> Path:
        """Raw episode split directory for this variant."""
        return self.data_root / "raw" / self.env_variant / split

    @property
    def vocab_path(self) -> Path:
        """Frozen train vocab JSON for this variant."""
        return self.data_root / "derived" / self.env_variant / "train_vocab.json"


def make_lazy_domain(
    env_variant: str, data_root: Path | str = DEFAULT_DATA_ROOT
) -> LazyDomain:
    """Build the domain for ``env_variant`` (DD2D cm-scale vs SB2D config-derived)."""
    data_root = Path(data_root)
    if env_variant.startswith("dd2d"):
        return LazyDomain(
            name="dd2d",
            env_variant=env_variant,
            frame_extent=(50.0, 40.0),
            shape_max=np.array([25.0, 25.0, 150.0, 1.0], dtype=np.float32),
            data_root=data_root,
        )
    if env_variant.startswith("stickbutton2d"):
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.baselines.piginet.sb2d_adapter import (
            _config_scales,
        )

        frame, shape_max = _config_scales()
        return LazyDomain(
            name="stickbutton2d",
            env_variant=env_variant,
            frame_extent=frame,
            shape_max=shape_max,
            data_root=data_root,
        )
    raise ValueError(f"no LAZY domain for env_variant {env_variant!r}")
