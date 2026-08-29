"""Problem-id encoding + per-stratum budgets for the ``restock3d_v2`` full collection.

The v2 continuous-packing SPECTRE dataset collects **five** configs -- two symmetric
(2x2, 3x3 tall-block x short-cube) at 50/15/15 and three crowded (3x4, 4x3, 4x4) at
**25/10/10** -- for **175 train / 60 val / 60 test = 295** (``SIZES``). They are collected
**sequentially, one banding stratum per process** (``SEQUENTIAL_ORDER``): a single-stratum
job has one block count, so its per-worker RAM peak (``PER_WORKER_GB``) is uniform and
predictable, which lets each job be sized to its own safe concurrency and fully reclaims
memory between strata. (This replaced a single mixed job whose per-worker peak was
unpredictable -- see decisions/07 2026-08-19.) Two stratum notions, deliberately separated:

- the **banding stratum** ``0..4`` -- the difficulty index -- used in ``problem_id`` so a
  v2-aware ``stratum_of(pid)`` returns it (contiguous pid bands => *stride, never
  truncate*);
- the **recipe key** -- a committed ``generator.STRATA`` entry (NOT runtime-injected)
  selecting the object counts. The env is built with the recipe key (carried in
  ``model_kwargs`` and the gym id ``spectre/Restock3Dv2-r{key}-v0``), so ``config_hash`` +
  ``git_sha`` pin the composition.

**Banding note (why v2 owns its band).** ``compare.STRATUM_BAND = SPLIT_BAND // 4`` only
fits four strata per split; a fifth would make ``problem_id("train",4,0)`` collide with
``problem_id("val",0,0)`` (duplicate env seeds + overwritten records). Those constants are
shared by DD2D/SB2D/restock3d_v1, so v2 must **not** edit them -- it defines its own
``V2_STRATUM_BAND = SPLIT_BAND // 5`` locally and provides its own ``stratum_of`` (returns
0..4, no ``min(3, ...)`` clamp). Routing the shared analysis/training consumers
(``compare.stratum_of``, ``train.py`` filter, the baseline datasets) to this decoder is a
deferred follow-up -- vocab-build does not stratify.

``env.reset(seed=problem_id)`` is the problem, so distinct split bands make a train/val
scene collision unrepresentable rather than merely avoided.
"""

from __future__ import annotations

from typing import Literal

from alphatamp.approaches.spectre.compare import SPLIT_BAND

Split = Literal["train", "val", "test"]

#: v2-local stratum sub-band: five strata per ``SPLIT_BAND``-wide split band (vs compare's
#: four). 50/15/15 indices (+ resample cushion) fit trivially under 200k.
V2_STRATUM_BAND = SPLIT_BAND // 5

#: Banding strata (difficulty index).
STRATA: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Banding stratum -> committed ``generator.STRATA`` recipe key (see ``STRATA_V2_PILOT``).
#: Keys 11/12/13 are the pre-existing symmetric 2x2/3x3/4x4; 14/15 are the new asymmetric
#: 3x4/4x3.
RECIPE_KEYS: dict[int, int] = {0: 11, 1: 12, 2: 14, 3: 15, 4: 13}

#: (n_tall, n_short) per banding stratum -- documentation; the authority is
#: ``generator.STRATA_V2_PILOT``.
CONFIGS: dict[int, tuple[int, int]] = {
    0: (2, 2),
    1: (3, 3),
    2: (3, 4),
    3: (4, 3),
    4: (4, 4),
}

#: Per-stratum ``(K_max, r_cap_s)`` collection budgets. K_max from the geometry generator's
#: oracle-index capture rates (Table A); r_cap from the *collection-path* feasible-solve tail
#: measured on the pilot records (BacktrackingRefiner, not the oracle certifier). See the
#: 5-stratum full-collection ADR (decisions/07 2026-08-18).
BUDGETS: dict[int, tuple[int, float]] = {
    0: (20, 40.0),
    1: (40, 70.0),
    2: (75, 80.0),
    3: (75, 80.0),
    4: (75, 90.0),
}

SPLIT_BANDS: dict[str, int] = {"train": 0, "val": 1, "test": 2}

#: **Per-stratum** keepers per split. Light strata (2x2, 3x3) at 50/15/15; the three crowded
#: strata (3x4, 4x3, 4x4) halved to 25/10/10 (they dominate collection cost). Totals:
#: 175 train / 60 val / 60 test = 295. Superseded ``PER_CONFIG`` (uniform 250/75/75).
SIZES: dict[int, dict[str, int]] = {
    0: {"train": 50, "val": 15, "test": 15},  # 2x2
    1: {"train": 50, "val": 15, "test": 15},  # 3x3
    2: {"train": 25, "val": 10, "test": 10},  # 3x4
    3: {"train": 25, "val": 10, "test": 10},  # 4x3
    4: {"train": 25, "val": 10, "test": 10},  # 4x4
}

#: Kept for documentation only (the uniform pre-2026-08-19 sizing); no longer read by the
#: collector, which now uses per-stratum :data:`SIZES`.
PER_CONFIG: dict[str, int] = {"train": 50, "val": 15, "test": 15}

#: Conservative upper-bound per-worker peak RSS (GB) for a single problem of each stratum --
#: the ``bpg`` scratchpad accumulates every sampled state across all K_max candidates, so the
#: peak scales with K_max x r_cap x samples and differs sharply by block count. 2x2 measured
#: 1.36 GB; 3x3 live-measured ~3.8 GB (corrected up from an interpolated 3.0). The crowded strata
#: kept climbing wave-over-wave -- 4x3's full run showed **wRSSmax up to 5.5 GB** (typical heavy
#: worker ~4.9 GB) and **min freeRAM 4.4 GB with ONE watchdog pause** at 10 workers. So 10 was
#: borderline; raised to **5.1 => 9 workers** on the crowded strata (9 x 4.9 = 44 GB -> ~15 GB
#: free typical; a rare synchronized 5.5 GB wave is watchdog-handled). The estimate was chased
#: up 3.7 -> 4.0 -> 4.3 -> 4.6 -> 5.5 across waves -- **a single/early wave undersamples the heavy
#: peak badly**. See decisions/07 2026-08-19/20. The ``wRSSmax`` heartbeat validates this live.
PER_WORKER_GB: dict[int, float] = {0: 1.7, 1: 3.8, 2: 5.1, 3: 5.1, 4: 5.1}

#: Order the strata are collected in, one sequential (gated) single-stratum job each:
#: 2x2, 3x3, 4x3, 3x4, 4x4. Light strata first (cheap, high-concurrency); the crowded strata
#: last, each capped to its own safe worker count. See decisions/07 2026-08-19.
SEQUENTIAL_ORDER: tuple[int, ...] = (0, 1, 3, 2, 4)

ENV_VARIANT = "restock3d_v2"


def recipe_key(stratum: int) -> int:
    """The committed ``generator.STRATA`` key for a banding stratum."""
    if stratum not in RECIPE_KEYS:
        raise ValueError(f"unknown stratum {stratum}")
    return RECIPE_KEYS[stratum]


def budget(stratum: int) -> tuple[int, float]:
    """``(K_max, r_cap_s)`` collection budget for a banding stratum."""
    if stratum not in BUDGETS:
        raise ValueError(f"unknown stratum {stratum}")
    return BUDGETS[stratum]


def sizes(stratum: int) -> dict[str, int]:
    """Per-split keeper targets (``{"train":.., "val":.., "test":..}``) for a stratum."""
    if stratum not in SIZES:
        raise ValueError(f"unknown stratum {stratum}")
    return SIZES[stratum]


def per_worker_gb(stratum: int) -> float:
    """Conservative upper-bound per-worker peak RSS (GB) for a stratum's problems."""
    if stratum not in PER_WORKER_GB:
        raise ValueError(f"unknown stratum {stratum}")
    return PER_WORKER_GB[stratum]


def env_id(stratum: int) -> str:
    """The registered v2 gym id for a banding stratum."""
    return f"spectre/Restock3Dv2-r{recipe_key(stratum)}-v0"


def problem_id(split: str, stratum: int, index: int) -> int:
    """Encode ``(split, banding-stratum, index)`` into a collection-wide problem id."""
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum}")
    if index >= V2_STRATUM_BAND:
        raise ValueError(
            f"index {index} overflows the stratum band ({V2_STRATUM_BAND}); it would be"
            f" read back as a different stratum"
        )
    return SPLIT_BANDS[split] * SPLIT_BAND + stratum * V2_STRATUM_BAND + index


def decode(pid: int) -> tuple[str, int, int]:
    """Inverse of :func:`problem_id`: ``(split, banding-stratum, index)``."""
    band, rest = divmod(int(pid), SPLIT_BAND)
    stratum, index = divmod(rest, V2_STRATUM_BAND)
    split = next(s for s, b in SPLIT_BANDS.items() if b == band)
    return split, stratum, index


def stratum_of(pid: int) -> int:
    """V2-aware banding stratum (0..4) from a problem id, any split.

    The v2 analogue of ``compare.stratum_of`` -- divides by ``V2_STRATUM_BAND`` and does
    **not** clamp to 3, so it can return 4. Shared consumers still call
    ``compare.stratum_of`` (4-stratum); route them here before any per-stratum
    restock3d_v2 analysis.
    """
    return (int(pid) % SPLIT_BAND) // V2_STRATUM_BAND
