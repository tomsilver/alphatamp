"""DD2D's :class:`~.domain.PIGINetDomain` — the behaviour this package had before it moved.

Every constant here was previously a module-level literal inside the package
(``glosses.GLOSSES``, ``encoders._SHAPE_MAX``, the ``drawer_wh or (50, 40)`` fallback, the
``dd2d_*`` glob in ``dataset``). They are reproduced **exactly**, because DD2D's trained
checkpoints and its published comparison numbers depend on them: the vocab order indexes
the frozen CLIP-text cache, and the normalisers scale every value feature. The lift is
verified by re-running the DD2D PIGINet cache and diffing it byte-for-byte
(``decisions/07``).

Data lives as a tree of per-record JSON with rendered PNG crops, which is why
:meth:`problems` globs and :meth:`crops` reads files — StickButton2D's adapter answers the
same two questions from ``EpisodeRecord`` pickles and a rasteriser instead.
"""

from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np

from alphatamp.approaches.spectre.piginet.record import PIGINetExample

#: DD2D domain vocabulary → colloquial NL glosses for the CLIP-text encoder (paper §IV-A).
#: PIGINet feeds each domain word through a frozen language model as a short English phrase
#: rather than its raw token ("rephrasing helps the network deal with out-of-distribution
#: names like ``isjointto``").
#:
#: Words appearing as *object arguments* (``target``, ``o0`` …) are NOT glossed — they are
#: encoded by the image+geometry object channel, not by text.
GLOSSES: dict[str, str] = {
    # operators (task-plan actions)
    "pick": "grasp and lift an object out of the drawer",
    "place-buffer": "place the carried object onto the staging buffer area beside the drawer",
    "retrieve": "grasp and remove the target object from the drawer",
    # predicates (init / goal literals)
    "handempty": "the robot gripper is empty and holding nothing",
    "in-drawer": "an object is resting inside the drawer",
    "target": "the target object that must be extracted",
    "extracted": "the target object has been removed from the drawer",
    "at-pose": "an object is located at this position and orientation",
    # object categories
    "item": "an ordinary household item cluttering the drawer",
    # object colors (also encode the concave/convex distinction in our render)
    "tomato": "a red target object",
    "slateblue": "a blue concave-shaped item",
    "silver": "a grey convex item",
    # shape families
    "can": "a small round can",
    "bowl": "a large round bowl",
    "box": "a rectangular box",
    "pillcase": "a long rounded capsule-shaped case",
    "dumbbell": "a dumbbell with two ends and a narrow waist",
    "shoe": "an L-shaped shoe with a concave corner",
    "horseshoe": "a blocky C-shaped horseshoe with two prongs",
    # regions
    "drawer": "the drawer interior holding the clutter",
    "buffer": "the staging buffer area beside the drawer",
}

VOCAB: list[str] = sorted(GLOSSES)


class DD2DDomain:
    """DD2D: centimetres, a ~50x40 drawer, JSON records with rendered PNG crops."""

    name = "dd2d"

    def __init__(self, data_root: str | None = None) -> None:
        self.data_root = data_root

    @property
    def vocab(self):
        """Glossable words, sorted — the order that indexes the CLIP-text cache."""
        return VOCAB

    def gloss(self, word: str) -> str:
        """NL phrase for a domain word (falls back to the word itself if unglossed)."""
        return GLOSSES.get(word, word)

    @property
    def frame_extent(self) -> tuple[float, float]:
        """Drawer ``(width, depth)`` in cm — the historical ``drawer_wh`` fallback."""
        return (50.0, 40.0)

    @property
    def shape_max(self) -> np.ndarray:
        """``[w, h, area, concave]`` divisors in cm (DD2D shapes.py / scene.py ranges)."""
        return np.array([25.0, 25.0, 150.0, 1.0], dtype=np.float32)

    # -- data ----------------------------------------------------------------
    def _problem_dirs(self, split: str) -> list[str]:
        assert self.data_root is not None, "DD2DDomain needs data_root to read a split"
        return sorted(glob.glob(os.path.join(self.data_root, split, "dd2d_*")))

    def problems(self, split: str) -> Iterator[tuple[str, list[PIGINetExample]]]:
        """Each problem directory's records, in ``plan_idx`` order."""
        for pdir in self._problem_dirs(split):
            recs = [
                PIGINetExample.load(r)
                for r in sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))
            ]
            if recs:
                yield os.path.basename(pdir), recs

    def crops(self, split: str, problem_id: str) -> dict:
        """Per-object crops read from the record's own rendered PNGs.

        Crops are shared across a problem's records, so the first record names them all.
        A missing file yields no entry; the CLIP cache writes zeros for those, which is
        the behaviour this had before the lift.
        """
        # pylint: disable=import-outside-toplevel
        import imageio.v2 as imageio
        from PIL import Image

        assert self.data_root is not None
        pdir = os.path.join(self.data_root, split, problem_id)
        rec = sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))[0]
        ex = PIGINetExample.load(rec)
        out = {}
        for img in ex.images:
            path = os.path.join(pdir, img["path"]) if img.get("path") else None
            if path and os.path.exists(path):
                out[img["object"]] = Image.fromarray(imageio.imread(path))
        return out

    def object_names(self, split: str, problem_id: str) -> list[str]:
        """Every object the CLIP cache must carry for this problem."""
        # pylint: disable=import-outside-toplevel
        assert self.data_root is not None
        pdir = os.path.join(self.data_root, split, problem_id)
        rec = sorted(glob.glob(os.path.join(pdir, "[0-9]*.json")))[0]
        return [img["object"] for img in PIGINetExample.load(rec).images]
