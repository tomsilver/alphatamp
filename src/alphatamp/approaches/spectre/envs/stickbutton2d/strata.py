"""Problem-id encoding for the pooled ``stickbutton2d_v1`` collection.

The four button counts are pooled into **one** env variant so a single model trains
across them, and button count becomes the difficulty stratum — the role
min-feasible-subset size plays on DD2D. Measured pool sizes make that ordering real:
b1 ≈ 2 candidates, b2 6–34, b3 200, b5 200.

The encoding is arithmetic on purpose::

    problem_id = split_band * SPLIT_BAND + slot * STRATUM_BAND + index

with ``SPLIT_BAND = 1_000_000`` and ``STRATUM_BAND = 250_000`` taken from
``dd2d_compare``, so ``compare.stratum_of(pid)`` — ``min(3, (pid % SPLIT_BAND) //
STRATUM_BAND)`` — returns the slot **exactly**. That is what lets fifteen existing call
sites (``train._keep``, ``spectre_score``'s per-stratum table, the compare cache)
work on this collection with no change, instead of growing a per-environment branch each.

Two things it also buys:

- **Splits never share an env seed.** ``env.reset(seed=problem_id)`` is the problem, so
  reusing indices across splits would put the same scene in train and test. Distinct
  bands make that unrepresentable rather than merely avoided.
- **A stratum is a contiguous pid band**, which is exactly the shape that makes
  ``paths[:N]`` return only b1. The project's *stride, never truncate* rule
  (``decisions.md`` 2026-07-27) applies with full force to this collection.

Because a silently wrong arithmetic identity would mislabel every stratum without
erroring, :func:`problem_id` is checked against ``stratum_of`` by a unit test, and each
episode independently records ``provenance.gen_params["stratum"]`` as an audit trail.
"""

from __future__ import annotations

from typing import Literal

from alphatamp.approaches.spectre.compare import SPLIT_BAND, STRATUM_BAND

Split = Literal["train", "val", "test"]

#: Button counts by stratum index. b10 is absent deliberately: a single A* run cannot
#: produce prefix-diverse pools at 10 buttons, so 0/20 problems were solvable within the
#: 200-attempt budget (``docs/autonomous_stickbutton_session.md`` D5). Recovering it
# needs : diverse plan *generation*, not a better heuristic.
BUTTON_COUNTS: tuple[int, ...] = (1, 2, 3, 5)

SPLIT_BANDS: dict[str, int] = {"train": 0, "val": 1, "test": 2}

#: Keepers per (variant, split). Pools to 400 / 100 / 100 across the four counts.
SPLIT_SIZES: dict[str, int] = {"train": 100, "val": 25, "test": 25}

ENV_VARIANT = "stickbutton2d_v1"


def slot_of(num_buttons: int) -> int:
    """Stratum index for a button count."""
    return BUTTON_COUNTS.index(num_buttons)


def problem_id(split: str, num_buttons: int, index: int) -> int:
    """Encode ``(split, button count, index)`` into a collection-wide problem id."""
    if index >= STRATUM_BAND:
        raise ValueError(
            f"index {index} overflows the stratum band ({STRATUM_BAND}); it would be"
            f" read back as a different button count"
        )
    return SPLIT_BANDS[split] * SPLIT_BAND + slot_of(num_buttons) * STRATUM_BAND + index


def decode(pid: int) -> tuple[str, int, int]:
    """Inverse of :func:`problem_id`: ``(split, num_buttons, index)``."""
    band, rest = divmod(int(pid), SPLIT_BAND)
    slot, index = divmod(rest, STRATUM_BAND)
    split = next(s for s, b in SPLIT_BANDS.items() if b == band)
    return split, BUTTON_COUNTS[slot], index


def env_id(num_buttons: int) -> str:
    """kinder env id for a button count."""
    return f"kinder/StickButton2D-b{num_buttons}-v0"
