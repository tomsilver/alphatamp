"""K_3,3 zone topology with the matching-based 3-edge-coloring (spec §4.1).

Six zones partitioned L = {L1, L2, L3}, R = {R1, R2, R3}; nine passages, three
per color, each color being a perfect matching (spec lines 308-318):

- Color A: L1-R1, L2-R2, L3-R3      (identity matching)
- Color B: L1-R2, L2-R3, L3-R1      (left-shift)
- Color C: L1-R3, L2-R1, L3-R2      (right-shift)

Properties guaranteed by inspection (spec lines 320-326):
S1. Removing any one color leaves a connected 6-cycle.
S2. Every single-color subgraph is a disconnected matching (no same-color
    same-side path).
S3. Every same-side zone pair has exactly 3 length-2 paths, one per color-pair.
"""

from __future__ import annotations

from typing import Final

# Zone names by side.
L_ZONES: Final[tuple[str, ...]] = ("L1", "L2", "L3")
R_ZONES: Final[tuple[str, ...]] = ("R1", "R2", "R3")
ALL_ZONES: Final[tuple[str, ...]] = L_ZONES + R_ZONES

COLORS: Final[tuple[str, ...]] = ("A", "B", "C")

# Per-color edge lists; each edge is (l_zone, r_zone). Order is canonical
# (lex by l_zone) so ``passage_id`` enumeration is stable.
COLOR_EDGES: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "A": (("L1", "R1"), ("L2", "R2"), ("L3", "R3")),
    "B": (("L1", "R2"), ("L2", "R3"), ("L3", "R1")),
    "C": (("L1", "R3"), ("L2", "R1"), ("L3", "R2")),
}

# Stable passage ids: ``passage_<color>_<idx>`` for idx in 0..2 within each
# color. Names are deterministic and used both as Object names and as keys in
# ``passage_widths`` dicts.
PASSAGE_NAMES: Final[dict[str, tuple[str, str, str]]] = {
    color: (
        f"passage_{color.lower()}_0",
        f"passage_{color.lower()}_1",
        f"passage_{color.lower()}_2",
    )
    for color in COLORS
}


def all_passage_names() -> tuple[str, ...]:
    """Return all 9 passage names in canonical (color, idx) order."""
    return tuple(name for color in COLORS for name in PASSAGE_NAMES[color])


def color_of_passage(passage_name: str) -> str:
    """Recover the color letter ('A'/'B'/'C') from a passage name."""
    # ``passage_a_0`` → 'A'.
    return passage_name.split("_")[1].upper()


def passage_endpoints(passage_name: str) -> tuple[str, str]:
    """Return ``(l_zone, r_zone)`` for a passage by name."""
    color = color_of_passage(passage_name)
    idx = int(passage_name.split("_")[2])
    return COLOR_EDGES[color][idx]


def passages_between(zone_a: str, zone_b: str) -> tuple[str, ...]:
    """Passage names connecting ``zone_a`` and ``zone_b`` (both directions)."""
    pair = {zone_a, zone_b}
    out = []
    for color in COLORS:
        for i, edge in enumerate(COLOR_EDGES[color]):
            if set(edge) == pair:
                out.append(PASSAGE_NAMES[color][i])
    return tuple(out)


def color_pairs() -> tuple[frozenset[str], ...]:
    """The 3 unordered color pairs over {A, B, C} in canonical order."""
    return (frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"B", "C"}))


# ---- BFS within a color-pair subgraph (the canonical detour mechanism) ----


def _color_pair_neighbors(
    zone: str, color_pair: tuple[str, ...]
) -> list[tuple[str, str]]:
    """Return ``[(passage_name, neighbor_zone), ...]`` reachable from ``zone`` via
    passages of color in ``color_pair``, sorted by passage name."""
    out: list[tuple[str, str]] = []
    for color in color_pair:
        for i, (l, r) in enumerate(COLOR_EDGES[color]):
            p_name = PASSAGE_NAMES[color][i]
            if l == zone:
                out.append((p_name, r))
            elif r == zone:
                out.append((p_name, l))
    out.sort()
    return out


def bfs_color_pair_path(
    src: str, dst: str, color_pair: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """Canonical shortest-path ``src → dst`` through the ``color_pair`` subgraph.

    Returns a list of ``(passage_name, src_zone, dst_zone)`` hops. Empty list
    if ``src == dst``. Tie-breaking: at each BFS frontier, neighbors are
    visited in passage-name lex order, so path reconstruction picks the
    lex-canonical predecessor. This determinism is what guarantees a stable
    canonical-skeleton-key for typed-local-id renumbering downstream.
    """
    if src == dst:
        return []
    visited: dict[str, tuple[str, str]] = {src: ("", "")}
    frontier = [src]
    while frontier and dst not in visited:
        next_frontier: list[str] = []
        for z in frontier:
            for p_name, nbr in _color_pair_neighbors(z, color_pair):
                if nbr in visited:
                    continue
                visited[nbr] = (z, p_name)
                next_frontier.append(nbr)
        frontier = next_frontier
    if dst not in visited:
        raise AssertionError(
            f"BFS failed: no path from {src} to {dst} via color_pair={color_pair}"
        )
    hops: list[tuple[str, str, str]] = []
    cur = dst
    while cur != src:
        prev, passage = visited[cur]
        hops.append((passage, prev, cur))
        cur = prev
    hops.reverse()
    return hops
