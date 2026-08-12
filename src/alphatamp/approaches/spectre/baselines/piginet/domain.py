"""Everything PIGINet needs to know about an environment, in one object.

Before this existed the DD2D specifics were scattered as module constants and glob
patterns: a gloss table imported at module scope, ``_SHAPE_MAX`` in centimetres, a
``drawer_wh`` key read out of ``provenance``, and a ``dd2d_*`` directory glob. Each is
individually reasonable and together they mean a second environment is a rewrite rather
than a declaration -- the same shape of problem ``domain.DomainSpec`` solved for SPECTRE
v3, and that ``vlmplan``'s env-agnostic core + ``*_adapter.py`` split solved for the VLM
baseline. This follows those.

**The normalisers are the reason this is a class and not a pair of imports.** PIGINet
divides poses by a frame extent and shapes by a per-field maximum so both land in
``[-1, 1]``. DD2D's are in centimetres over a ~50x40 drawer; StickButton2D's are in
metres over a 3.5x2.5 world with objects two orders of magnitude smaller. Leave them as
module constants and every StickButton2D feature underflows to ~0 -- the model then
reads as hopeless, and the conclusion "the low-level predictor loses on this
environment" is a unit bug wearing a result's clothes. Making them domain state is what
makes that unrepresentable.

An adapter supplies three things: the vocabulary (glosses), the numeric scales, and how
to enumerate a split -- because the two environments store their data differently (DD2D
a tree of per-record JSON plus rendered PNGs, StickButton2D ``EpisodeRecord`` pickles
with geometry to render from). What it does *not* supply is any part of the model, the
tokenizer's structure, or the losses.
"""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence

import numpy as np

from alphatamp.approaches.spectre.baselines.piginet.record import PIGINetExample


class PIGINetDomain(Protocol):
    """The per-environment contract.

    See the module docstring for why each part is here.
    """

    name: str

    # -- vocabulary ----------------------------------------------------------
    @property
    def vocab(self) -> Sequence[str]:
        """Every glossable domain word, in a **stable order**.

        Order is load-bearing: it indexes the frozen CLIP-text cache, so changing it
        invalidates every trained checkpoint of this domain without changing any shape.
        """

    def gloss(self, word: str) -> str:
        """The colloquial English phrase for a domain word (paper §IV-A).

        PIGINet feeds domain words through a frozen language model as phrases rather
        than raw tokens, because "rephrasing helps the network deal with
        out-of-distribution names". Unglossed words fall back to themselves.
        """

    # -- numeric scales ------------------------------------------------------
    @property
    def frame_extent(self) -> tuple[float, float]:
        """``(width, depth)`` of the world in the same units as object poses."""

    @property
    def shape_max(self) -> np.ndarray:
        """Per-field divisor for ``[w, h, area, concave]`` shape values."""

    # -- data ----------------------------------------------------------------
    def problems(self, split: str) -> Iterator[tuple[str, list[PIGINetExample]]]:
        """``(problem_id, examples)`` for one split, grouped by problem.

        Grouped because the ranking loss and the val rollout-FP proxy both need a
        problem's candidate plans together.
        """

    def crops(self, split: str, problem_id: str) -> dict:
        """``{object_name: PIL.Image}`` for one problem's per-object segmented crops.

        Called once per problem when building the CLIP cache, never during training.
        An environment with no images may return ``{}``; :class:`Encoders` then needs
        ``obj_channels`` without ``"img"``, or it will read zeros.
        """
