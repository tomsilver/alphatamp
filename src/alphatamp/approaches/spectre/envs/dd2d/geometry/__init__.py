"""Geometry/render backends.

In this mapping, *refinement feasibility* is computed analytically in
``refine.py`` (render-backend independent). These backends exist to **confirm that
rendering PIGINet-style segmented object images is doable** -- the original
kitchen-worlds/LAZY stacks render none. Two interchangeable implementations:

* :class:`~blocks_tamp.geometry.pybullet_backend.PyBulletBackend` -- builds the
  real 3-D scene and renders RGB + a true segmentation mask headlessly via
  PyBullet's TinyRenderer (the same simulator PIGINet used).
* :class:`~blocks_tamp.geometry.numpy2d_backend.Numpy2DBackend` -- a dependency-light
  top-down rasteriser (numpy only) used when PyBullet is unavailable.

A backend renders to a :class:`RenderResult` carrying an RGB image, an integer
segmentation buffer, and an id->object-name map -- everything a downstream image
encoder (CLIP, etc.) would need to crop per-object views.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..scene import GeometricScene


@dataclass
class RenderResult:
    rgb: np.ndarray  # (H, W, 3) uint8
    seg: np.ndarray  # (H, W) int32; pixel value = object id (>=0), -1 background
    id_to_name: dict[int, str]
    view: str

    def segment_ids(self) -> set[int]:
        return {int(v) for v in np.unique(self.seg) if v >= 0}


class GeometryBackend(ABC):
    name: str = "abstract"

    @classmethod
    @abstractmethod
    def available(cls) -> bool:  # pragma: no cover - trivial
        ...

    @abstractmethod
    def render_segmented(
        self,
        scene: GeometricScene,
        view: str = "topdown",
        width: int = 256,
        height: int = 256,
    ) -> RenderResult: ...


def get_backend(prefer: str = "pybullet") -> GeometryBackend:
    """Return the best available render backend (``prefer`` in {pybullet, numpy2d})."""
    from .numpy2d_backend import Numpy2DBackend
    from .pybullet_backend import PyBulletBackend

    if prefer == "pybullet" and PyBulletBackend.available():
        return PyBulletBackend()
    return Numpy2DBackend()


__all__ = ["GeometryBackend", "RenderResult", "get_backend"]
