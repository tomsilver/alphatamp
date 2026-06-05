"""K_3,3 topology + 3-edge-coloring tests (spec §8.3 #1-2)."""

from __future__ import annotations

from collections import Counter

import pytest

from alphatamp.approaches.spectre.envs.routedtransport2d import topology as topo


def test_zone_partition_bipartite() -> None:
    """6 zones split 3/3 between L and R."""
    assert len(topo.L_ZONES) == 3
    assert len(topo.R_ZONES) == 3
    assert len(set(topo.L_ZONES) & set(topo.R_ZONES)) == 0
    assert len(topo.ALL_ZONES) == 6


def test_passages_and_colors() -> None:
    """3 colors × 3 edges per color = 9 passages, all cross the bipartition."""
    assert len(topo.COLORS) == 3
    assert len(topo.all_passage_names()) == 9
    for color in topo.COLORS:
        assert len(topo.PASSAGE_NAMES[color]) == 3
        edges = topo.COLOR_EDGES[color]
        assert len(edges) == 3
        # Every edge crosses the L/R bipartition.
        for l_zone, r_zone in edges:
            assert l_zone in topo.L_ZONES
            assert r_zone in topo.R_ZONES


def test_each_color_is_perfect_matching() -> None:
    """Each color class is a 3-edge perfect matching of L → R."""
    for color in topo.COLORS:
        edges = topo.COLOR_EDGES[color]
        l_seen = Counter(l for l, _ in edges)
        r_seen = Counter(r for _, r in edges)
        assert all(c == 1 for c in l_seen.values()), color
        assert all(c == 1 for c in r_seen.values()), color
        assert set(l_seen) == set(topo.L_ZONES)
        assert set(r_seen) == set(topo.R_ZONES)


def test_removing_one_color_leaves_connected_six_cycle() -> None:
    """Topology property S1 — every 2-color subgraph reaches all 6 zones."""
    for color_pair in topo.color_pairs():
        # BFS from any zone should reach all 6.
        start = topo.ALL_ZONES[0]
        seen = {start}
        frontier = [start]
        cp_tuple = tuple(sorted(color_pair))
        while frontier:
            nxt: list[str] = []
            for z in frontier:
                neighbors = (
                    topo._color_pair_neighbors(  # pylint: disable=protected-access
                        z, cp_tuple
                    )
                )
                for _p, nbr in neighbors:
                    if nbr not in seen:
                        seen.add(nbr)
                        nxt.append(nbr)
            frontier = nxt
        assert seen == set(topo.ALL_ZONES), color_pair


@pytest.mark.parametrize("color", list(topo.COLORS))
def test_color_of_passage_roundtrip(color: str) -> None:
    """color_of_passage inverts PASSAGE_NAMES for every passage."""
    for p_name in topo.PASSAGE_NAMES[color]:
        assert topo.color_of_passage(p_name) == color


def test_bfs_path_empty_when_src_eq_dst() -> None:
    """A path from a zone to itself is the empty hop list."""
    assert not topo.bfs_color_pair_path("L1", "L1", ("A", "B"))


def test_bfs_path_same_side_two_hops_via_color_pair_subgraph() -> None:
    """Same-side travel takes two hops through the color-pair subgraph.

    The earlier bug: R2 → R1 via {B, C} requires the (C, B) detour ordering
    because (B, C) starting from R2 ends at R3, not R1.
    """
    hops = topo.bfs_color_pair_path("R2", "R1", ("B", "C"))
    assert len(hops) == 2
    src_to_mid, mid_to_dst = hops  # pylint: disable=unbalanced-tuple-unpacking
    assert src_to_mid[1] == "R2"
    assert mid_to_dst[2] == "R1"
    # Intermediate zone is on the L side (we crossed the bipartition).
    assert src_to_mid[2] in topo.L_ZONES
    # Every passage used is in color_pair.
    assert topo.color_of_passage(src_to_mid[0]) in ("B", "C")
    assert topo.color_of_passage(mid_to_dst[0]) in ("B", "C")


def test_bfs_path_opposite_side_direct_in_pair() -> None:
    """Opposite-side travel is one hop when the direct color is in the pair."""
    hops = topo.bfs_color_pair_path("L1", "R1", ("A", "B"))
    assert len(hops) == 1  # Direct passage: L1-R1 is color A, in pair.
    assert topo.color_of_passage(hops[0][0]) == "A"


def test_bfs_path_opposite_side_direct_not_in_pair_uses_three_hops() -> None:
    """L1-R1 is color A; with pair {B, C} the 6-cycle forces three hops."""
    hops = topo.bfs_color_pair_path("L1", "R1", ("B", "C"))
    assert len(hops) == 3
    for p_name, _src, _dst in hops:
        assert topo.color_of_passage(p_name) in ("B", "C")


def test_color_pairs_are_three_unordered_pairs() -> None:
    """color_pairs() is exactly the three unordered 2-subsets of {A,B,C}."""
    pairs = topo.color_pairs()
    assert len(pairs) == 3
    assert all(len(p) == 2 for p in pairs)
    assert set(pairs) == {
        frozenset({"A", "B"}),
        frozenset({"A", "C"}),
        frozenset({"B", "C"}),
    }
