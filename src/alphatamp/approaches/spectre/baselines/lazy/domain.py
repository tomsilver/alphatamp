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
    #: Graph node-geometry width: 8 (default) or 9 (Restock3D v2 = +height, the F3 axis).
    #: When 9, ``shape_max`` must carry a 5th entry (the height normalizer).
    geom_dim: int = 8

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
    if env_variant.startswith("restock3d"):
        # Restock3D (continuous packing). ``frame_extent`` matches the scene-geometry
        # producer's normalization frame (shelf_width, shelf_y + shelf_depth), unchanged
        # across versions. ``shape_max`` normalizes (w, h, area, concave, height) -- a
        # **9-dim** graph geometry: the 5th entry normalizes the appended height feature so
        # the GAT sees the F3 axis. This is the Gate-6 widening; DD2D/SB2D stay at geom_dim=8.
        #
        # v3 varies per-object width + height (v2 kept them constant), so its divisors come
        # from the v3 envelope (width <= WIDTH_MAX=0.08, depth fixed 0.05, height <= tall
        # cutoff 0.17); v2 keeps its constant-footprint 0.05/0.24 values. Measured maxima from
        # the collected data: v3 width 0.080, depth 0.050, area 0.004, height 0.170.
        if env_variant.startswith("restock3d_v3"):
            shape_max = np.array([0.08, 0.05, 0.004, 1.0, 0.17], dtype=np.float32)
        else:
            shape_max = np.array([0.05, 0.05, 0.0025, 1.0, 0.24], dtype=np.float32)
        return LazyDomain(
            name="restock3d",
            env_variant=env_variant,
            frame_extent=(0.60198, 1.654),
            shape_max=shape_max,
            data_root=data_root,
            geom_dim=9,
        )
    raise ValueError(f"no LAZY domain for env_variant {env_variant!r}")
