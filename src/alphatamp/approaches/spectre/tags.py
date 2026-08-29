"""Episode-local object tags — the P-A binding (proposal §7).

Each object in an episode gets a **tag** (an integer id); the *same* tag is used
wherever that object appears — scene tokens, skeleton argument slots, fact arguments
— so the model can look up an object's content (geometry, role) from its tag. Tags
are:

- **episode-local**: assigned per episode, independent across episodes;
- **deterministic at eval** (``rng=None``): sorted-name order, so inference is
  reproducible;
- **permuted per epoch in training** (``rng`` seeded from
  ``(seed, episode_idx, epoch)``):
  a random injection into ``[1, max_tags]`` so no tag id accumulates global meaning
  — the network must use the *content* a tag points at, never the id.

Tag ``0`` is reserved for pad / OOV (mirroring the vocab's local-id-0 convention). This
supersedes v1's typed-local-ids *inside the v2 tensorizer* (``canonicalize`` still runs
first for structure/type names); binding args to tags — and thus to per-object
geometry in the scene tokens — is what provably removes v1's length-only collapse
(two same-length skeletons over different objects now differ, because their tags point
at different geometry).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

PAD_TAG = 0


def assign_tags(
    object_names: Iterable[str],
    rng: Optional[np.random.Generator] = None,
    max_tags: Optional[int] = None,
) -> dict[str, int]:
    """Map each object name to a distinct tag id in ``[1, max_tags]`` (``0`` = pad/OOV).

    ``rng=None`` → deterministic (sorted-name order → tags ``1..n``),
    for eval/inference.
    ``rng`` set → a random injection of the ``n`` objects into ``[1, max_tags]``,
    redrawn per epoch at training time so no id is stable. ``max_tags`` defaults
    to ``n``.
    """
    names = sorted(object_names)
    n = len(names)
    if max_tags is None:
        max_tags = n
    if n > max_tags:
        raise ValueError(f"{n} objects exceed max_tags={max_tags}")
    if rng is None:
        chosen = list(range(1, n + 1))  # deterministic 1..n
    else:
        # a random injection: distinct tags drawn from [1, max_tags].
        chosen = [int(t) + 1 for t in rng.permutation(max_tags)[:n]]
    return {name: chosen[i] for i, name in enumerate(names)}


def tag_seed(seed: int, episode_idx: int, epoch: int) -> np.random.Generator:
    """Reproducible per-(seed, episode, epoch) generator for training-time tag
    permutation (mirrors the F-sampling seeding discipline in ``dataset.py``)."""
    return np.random.default_rng((int(seed), int(episode_idx), int(epoch)))
