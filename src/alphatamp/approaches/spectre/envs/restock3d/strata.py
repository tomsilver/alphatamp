"""Problem-id encoding for the ``restock3d_v1`` collection.

Restock3D's four difficulty strata (r0-r3, by ``d = (sigma_tall, sigma_short)``) are pooled into
one env_variant so a single model trains across them, with the stratum index playing the role
min-feasible-subset size plays on DD2D. The encoding is arithmetic on purpose::

    problem_id = split_band * SPLIT_BAND + stratum * STRATUM_BAND + index

with ``SPLIT_BAND = 1_000_000`` and ``STRATUM_BAND = 250_000`` from ``compare``, so
``compare.stratum_of(pid)`` — ``min(3, (pid % SPLIT_BAND) // STRATUM_BAND)`` — returns the stratum
exactly. That is what lets the existing per-stratum call sites (train._keep, spectre_score's
per-stratum table, the compare cache) work on this collection with no per-environment branch.

Two invariants it buys, both load-bearing (mirroring ``envs/stickbutton2d/strata.py``):

- **Splits never share a seed.** ``env.reset(seed=problem_id)`` is the problem, so distinct split
  bands make a train/test scene collision unrepresentable rather than merely avoided.
- **A stratum is a contiguous pid band**, so the project's *stride, never truncate* rule applies:
  ``paths[:N]`` returns only r0.

Because a silently wrong identity would mislabel every stratum without erroring, :func:`problem_id`
is pinned against ``stratum_of`` by a unit test.
"""

from __future__ import annotations

from typing import Literal

from alphatamp.approaches.spectre.compare import SPLIT_BAND, STRATUM_BAND

Split = Literal["train", "val", "test"]

#: The four strata (see ``generator.STRATA``).
STRATA: tuple[int, ...] = (0, 1, 2, 3)

SPLIT_BANDS: dict[str, int] = {"train": 0, "val": 1, "test": 2}

#: Keepers per split, split evenly across the four strata (pools to 400 / 100 / 100).
SPLIT_SIZES: dict[str, int] = {"train": 400, "val": 100, "test": 100}

ENV_VARIANT = "restock3d_v1"


def problem_id(split: str, stratum: int, index: int) -> int:
    """Encode ``(split, stratum, index)`` into a collection-wide problem id."""
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum}")
    if index >= STRATUM_BAND:
        raise ValueError(
            f"index {index} overflows the stratum band ({STRATUM_BAND}); it would be read"
            f" back as a different stratum"
        )
    return SPLIT_BANDS[split] * SPLIT_BAND + stratum * STRATUM_BAND + index


def decode(pid: int) -> tuple[str, int, int]:
    """Inverse of :func:`problem_id`: ``(split, stratum, index)``."""
    band, rest = divmod(int(pid), SPLIT_BAND)
    stratum, index = divmod(rest, STRATUM_BAND)
    split = next(s for s, b in SPLIT_BANDS.items() if b == band)
    return split, stratum, index
