"""Restock3D-**v3** analytic feasibility core — the single source of truth for the pinned
geometry constants, the capacity formula, height cutoffs, split enumeration, the named
greedy hand-rules, and the pure-geometry refinability *classifier*.

v3 makes block **selection** matter: blocks carry **per-object widths** and **heights sampled
near the short/tall fit cutoff** (unlike v2's type-keyed constant dims), so which subset goes on
which level is a real reasoning problem. This module is imported by:

* the **generator** (``generator_v3.py``) — split enumeration + acceptance filter;
* the **analytic classifier** used at collection time — ``classify_skeleton`` walks a candidate
  skeleton and, at the first violation, emits a ``refiner_metadata["failures"]`` dict in the exact
  shape ``instrumented_refiner.failure_metadata`` produces (so the SPECTRE downstream consumes it
  unchanged and the analytic labels are the *same kind of evidence* the real refiner emits);
* the **hand-rule baselines** and the **gates** (G1/G2/G3).

Nothing here runs motion planning — feasibility is pure arithmetic (capacity + height cutoffs) plus
the **shared** reach-over geometry (``instrumented_refiner._blocks_reach``, the same rule the real
refiner uses for reach-over culprit attribution, so the two agree by construction — measured at
Gate G1). The pinned constants were established by the v3 calibration study
(``docs/restock3d_v3_calibration.md``) and the crowded-feasibility experiment
(``docs/notebook/07-stickbutton2d.md`` 2026-08-20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# The reach-over corridor rule + its constants are shared verbatim with the real refiner so the
# analytic reach-over F4 and the refiner's ``reach_over_culprits`` agree by construction.
from .instrumented_refiner import _blocks_reach
from .place_controller import _FRONT_GRASP_MIN_HALF_Z

# ---------------------------------------------------------------------------------------------
# Pinned constants (calibrated) — DO NOT re-derive elsewhere; import from here.
# ---------------------------------------------------------------------------------------------
#: Re-balanced shelf partition (tall, short), same total shelf height as v2's (0.34, 0.15).
SECTION_CLEARANCES: tuple[float, float] = (0.27, 0.22)
#: Arm-insertion height cutoffs per section (~0.10 m gripper headroom under each board).
TALL_CUTOFF: float = 0.17
SHORT_CUTOFF: float = 0.12
CUTOFF: dict[str, float] = {"tall": TALL_CUTOFF, "short": SHORT_CUTOFF}
#: Graspable full-width range (<= ~0.9x the ~92 mm finger aperture; the sim is width-permissive).
WIDTH_MIN: float = 0.02
WIDTH_MAX: float = 0.08
#: Required face-to-face gap between adjacent blocks (>= calibration's 33-50 mm empirical min).
GAP: float = 0.06
#: Face-to-face lateral packing budget per level (conservative inside the 0.522 m physical band).
USABLE: float = 0.50
#: Clearance reserved from each end wall (matches v2 ``_X_BAND_END_MARGIN``; walls are visual-only).
END_MARGIN: float = 0.04

SECTIONS: tuple[str, str] = ("tall", "short")
_EPS = 1e-9


@dataclass(frozen=True)
class Block:
    """A v3 block: full x-width, full z-height, and (optional) floor xy for reach-over."""

    name: str
    width: float
    height: float
    x: float = 0.0
    y: float = 0.0


# ---------------------------------------------------------------------------------------------
# Height eligibility + reach-tall
# ---------------------------------------------------------------------------------------------
def height_eligible(height: float, section: str) -> bool:
    """A block fits a section (arm can insert it) iff its full height is within the section's
    calibrated cutoff. This is the F3 rule — arm-insertion limited, ~0.10 m below the board.
    """
    return height <= CUTOFF[section] + _EPS


def is_reach_tall(height: float) -> bool:
    """Whether a block counts as "tall" for the reach-over corridor (the front grasp reaches
    over it). Matches the refiner's ``half_extent_z >= _FRONT_GRASP_MIN_HALF_Z`` test.
    """
    return (height / 2.0) >= _FRONT_GRASP_MIN_HALF_Z


def section_of_op(op_name: str) -> str:
    """``place_tall`` -> ``tall``; ``place_short`` -> ``short``."""
    return "tall" if op_name == "place_tall" else "short"


# ---------------------------------------------------------------------------------------------
# Capacity formula (single source of truth)
# ---------------------------------------------------------------------------------------------
def level_used(widths: Sequence[float]) -> float:
    """Lateral space a set of block widths consumes on one level:
    ``sum(widths) + GAP*(n-1) + 2*END_MARGIN`` (0 for an empty level)."""
    n = len(widths)
    if n == 0:
        return 0.0
    return float(sum(widths)) + GAP * (n - 1) + 2 * END_MARGIN


def level_fits(widths: Sequence[float]) -> bool:
    """The capacity formula: a set of widths fits one level iff ``level_used <= USABLE``."""
    return level_used(widths) <= USABLE + _EPS


# ---------------------------------------------------------------------------------------------
# Splits (assignments of blocks to {tall, short})
# ---------------------------------------------------------------------------------------------
def _level_widths(
    assignment: dict[str, str], blocks: Sequence[Block]
) -> tuple[list[float], list[float]]:
    tall_w = [b.width for b in blocks if assignment[b.name] == "tall"]
    short_w = [b.width for b in blocks if assignment[b.name] == "short"]
    return tall_w, short_w


def split_is_feasible(assignment: dict[str, str], blocks: Sequence[Block]) -> bool:
    """A split is feasible iff every block is height-eligible for its assigned section AND both
    levels pass the capacity formula."""
    for b in blocks:
        if not height_eligible(b.height, assignment[b.name]):
            return False
    tall_w, short_w = _level_widths(assignment, blocks)
    return level_fits(tall_w) and level_fits(short_w)


def enumerate_feasible_splits(blocks: Sequence[Block]) -> list[dict[str, str]]:
    """Every feasible assignment of the blocks across the two levels (<= 2^n, n <= ~12)."""
    n = len(blocks)
    out: list[dict[str, str]] = []
    for mask in range(1 << n):
        assignment = {
            b.name: ("short" if (mask >> i) & 1 else "tall")
            for i, b in enumerate(blocks)
        }
        if split_is_feasible(assignment, blocks):
            out.append(assignment)
    return out


def feasible_ratio(blocks: Sequence[Block]) -> tuple[int, int, float]:
    """``(n_feasible, n_total_splits, rho)`` where ``rho = n_feasible / 2^n`` — how many of all
    splits solve the instance (the difficulty knob; hard strata want a small non-zero rho).
    """
    n = len(blocks)
    total = 1 << n
    n_feasible = len(enumerate_feasible_splits(blocks))
    return n_feasible, total, (n_feasible / total if total else 0.0)


def fill_fraction(assignment: dict[str, str], blocks: Sequence[Block]) -> float:
    """Fraction of the two-level capacity a split consumes: ``(used_tall+used_short)/(2*USABLE)``."""
    tall_w, short_w = _level_widths(assignment, blocks)
    return (level_used(tall_w) + level_used(short_w)) / (2 * USABLE)


def min_fill_over_feasible(blocks: Sequence[Block]) -> Optional[float]:
    """The loosest feasible packing's fill fraction (None if no feasible split). A high value
    means even the roomiest solution is tight — the "tight but not degenerate" acceptance knob.
    """
    splits = enumerate_feasible_splits(blocks)
    if not splits:
        return None
    return min(fill_fraction(a, blocks) for a in splits)


# ---------------------------------------------------------------------------------------------
# Named greedy hand-rules (shared by the generator acceptance filter and Gate G3).
# Each returns a deterministic assignment that may be feasible or not — hard strata are
# constructed so BOTH rules pick an infeasible split (the "no universal rule" property).
# ---------------------------------------------------------------------------------------------
def _slack_after(widths: Sequence[float], w: float) -> float:
    return USABLE - level_used(list(widths) + [w])


def greedy_widest_best_fit(blocks: Sequence[Block]) -> dict[str, str]:
    """Widest-first best-fit: place blocks widest-first into the height-eligible level where they
    fit most snugly (least leftover slack); fall back to an eligible level (infeasible) if none
    fits, or ``tall`` if height-impossible."""
    assignment: dict[str, str] = {}
    widths: dict[str, list[float]] = {"tall": [], "short": []}
    for b in sorted(blocks, key=lambda z: -z.width):
        eligible = [s for s in SECTIONS if height_eligible(b.height, s)]
        fitting = [s for s in eligible if level_fits(widths[s] + [b.width])]
        if fitting:
            chosen = min(fitting, key=lambda s: _slack_after(widths[s], b.width))
        elif eligible:
            chosen = eligible[0]
        else:
            chosen = "tall"
        assignment[b.name] = chosen
        widths[chosen].append(b.width)
    return assignment


def greedy_send_shortest_up(blocks: Sequence[Block]) -> dict[str, str]:
    """Send-shortest-up: shortest-first, fill the short section while it fits, rest to tall."""
    assignment: dict[str, str] = {}
    widths: dict[str, list[float]] = {"tall": [], "short": []}
    for b in sorted(blocks, key=lambda z: z.height):
        if height_eligible(b.height, "short") and level_fits(
            widths["short"] + [b.width]
        ):
            assignment[b.name] = "short"
            widths["short"].append(b.width)
        else:
            assignment[b.name] = "tall"
            widths["tall"].append(b.width)
    return assignment


HAND_RULES = {
    "widest_best_fit": greedy_widest_best_fit,
    "send_shortest_up": greedy_send_shortest_up,
}


# ---------------------------------------------------------------------------------------------
# Analytic refinability classifier — emits a failure dict in ``failure_metadata`` shape.
# ---------------------------------------------------------------------------------------------
def _failure_dict(
    step_index: int,
    op_name: str,
    args: Sequence[str],
    culprits: Sequence[str],
    num_attempts: int,
) -> dict:
    """One ``refiner_metadata["failures"]`` entry, byte-compatible with
    ``instrumented_refiner.failure_metadata`` so the SPECTRE downstream reads it unchanged. A
    class-1 record (F2 residents / F4 reach-over blockers) carries ``culprits`` and no deviation;
    a culprit-free height (F3) record carries an empty deviation and ``proves_failure()``.
    """
    is_class_1 = bool(culprits)
    return {
        "step_index": int(step_index),
        "schema": str(op_name),
        "args": list(args),
        "culprits": list(culprits),
        "n_step": int(num_attempts),
        "exhausted": True,
        "budget_exhausted": False,
        "dev_added": None if is_class_1 else [],
        "dev_deleted": None if is_class_1 else [],
    }


def classify_skeleton(
    plan_steps: Iterable[tuple[str, Sequence[str]]],
    block_dims: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    num_attempts: int = 18,
) -> Optional[dict]:
    """Classify a candidate skeleton **without refining** and **in order**, returning ``None`` if
    it is feasible, else the first-violation failure dict.

    ``plan_steps`` is the ordered plan as ``(op_name, arg_names)`` with ``op_name`` in
    ``{pick, place_tall, place_short}`` and ``arg_names`` the operator's parameters (``target =
    arg_names[1]``). ``block_dims[name] = (width, height)``; ``positions[name] = (x, y)`` the
    floor spot. Failure families (distinguished downstream only by schema + culprits + deviation,
    exactly as the real refiner serializes them):

    * **height F3** — a place of a block taller than its section's cutoff -> culprit-free.
    * **crowding F2** — a place that overflows the level's capacity formula -> culprits = the
      residents already stored on that level.
    * **reach-over F4** — a pick whose south reach corridor still holds uncleared floor blockers
      -> **culprit-free** (F4 is dead: parity with the real ``_probe_pick``, which returns no pick
      culprits). Still a class-2 failure, so the feasibility label is unchanged; only culprit
      attribution is dropped. Restock3D-v3 tracks ONLY F2 (crowding) culprits.
    """
    cleared: set[str] = set()  # objects already picked (off the floor)
    placed_names: dict[str, list[str]] = {"place_tall": [], "place_short": []}
    placed_widths: dict[str, list[float]] = {"place_tall": [], "place_short": []}

    for step_index, (op_name, args) in enumerate(plan_steps):
        args = tuple(args)
        if op_name == "pick":
            target = args[1]
            bx, by = positions[target]
            b_tall = is_reach_tall(block_dims[target][1])
            culprits = sorted(
                a
                for a in positions
                if a != target
                and a not in cleared
                and _blocks_reach(
                    (positions[a][0], positions[a][1], 0.0),
                    is_reach_tall(block_dims[a][1]),
                    (bx, by, 0.0),
                    b_tall,
                )
            )
            if culprits:
                # F4 (reach-over) is DEAD: track NO pick culprits, for parity with the real
                # refiner's ``_probe_pick`` (culprit-free, returns "C2"). The reach-over is still
                # an infeasibility (a class-2 failure), so the feasibility label is unchanged --
                # only the culprit attribution is dropped. Restock3D-v3 tracks ONLY F2 (crowding)
                # culprits. Inert to repeat/regroup (pick has no step/grouping certificate).
                return _failure_dict(step_index, op_name, args, (), num_attempts)
            cleared.add(target)
        elif op_name in ("place_tall", "place_short"):
            target = args[1]
            width, height = block_dims[target]
            section = section_of_op(op_name)
            # F3 (height) is tested first — matches the real probe (shelf_hit before residents).
            if not height_eligible(height, section):
                return _failure_dict(step_index, op_name, args, (), num_attempts)
            # F2 (crowding): the residents already on this level are the culprits.
            residents = list(placed_names[op_name])
            if not level_fits(placed_widths[op_name] + [width]):
                return _failure_dict(
                    step_index, op_name, args, tuple(sorted(residents)), num_attempts
                )
            placed_names[op_name].append(target)
            placed_widths[op_name].append(width)
    return None
