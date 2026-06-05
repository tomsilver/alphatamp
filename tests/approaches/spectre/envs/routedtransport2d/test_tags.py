"""Tag atoms in s_0 + persistence (spec §8.3 #8)."""

from __future__ import annotations

import itertools

from alphatamp.approaches.spectre.envs.routedtransport2d import operators as ops
from alphatamp.approaches.spectre.envs.routedtransport2d import topology as topo
from alphatamp.approaches.spectre.envs.routedtransport2d.plan_generator import (
    ClosedFormSkeletonGenerator,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    make_problem,
)
from alphatamp.approaches.spectre.envs.routedtransport2d.tags import (
    SIZE_LEVELS,
    WIDTH_LEVELS,
    is_compatible,
)


def test_initial_state_has_nine_passage_widths() -> None:
    """S0 carries exactly one PassageWidth atom per passage (9 total)."""
    p = make_problem(seed=0, variant="n3-v1")
    width_atoms = [
        a for a in p.initial_abstract_state.atoms if a.predicate is ops.PassageWidth
    ]
    assert len(width_atoms) == 9
    seen_passages = {a.entities[0].name for a in width_atoms}
    assert seen_passages == set(topo.all_passage_names())


def test_initial_state_has_one_itemsize_per_item() -> None:
    """S0 carries exactly one ItemSize atom per item."""
    p = make_problem(seed=0, variant="n3-v1")
    size_atoms = [
        a for a in p.initial_abstract_state.atoms if a.predicate is ops.ItemSize
    ]
    assert len(size_atoms) == p.num_items
    seen_items = {a.entities[0].name for a in size_atoms}
    assert seen_items == {f"item_{i}" for i in range(p.num_items)}


def test_tag_atoms_persist_along_skeleton() -> None:
    """Spec §8.3 #8: PassageWidth/ItemSize atoms appear unchanged in every state."""
    p = make_problem(seed=0, variant="n3-v1")
    static_atoms = {
        a
        for a in p.initial_abstract_state.atoms
        if a.predicate in (ops.PassageWidth, ops.ItemSize)
    }
    gen = ClosedFormSkeletonGenerator(p, k_cap=30)
    for state_plan, _action_plan in itertools.islice(
        gen(None, p.initial_abstract_state, p.goal, 1.0, None), 5
    ):
        for s in state_plan:
            assert static_atoms.issubset(
                s.atoms
            ), "tag atoms must persist along the entire skeleton"


def test_compatibility_table() -> None:
    """Spec §3.3 compatibility table: small fits all, large only fits wide."""
    expected = {
        ("small", "narrow"): True,
        ("small", "medium"): True,
        ("small", "wide"): True,
        ("medium", "narrow"): False,
        ("medium", "medium"): True,
        ("medium", "wide"): True,
        ("large", "narrow"): False,
        ("large", "medium"): False,
        ("large", "wide"): True,
    }
    for size in SIZE_LEVELS:
        for width in WIDTH_LEVELS:
            assert is_compatible(size, width) == expected[(size, width)]
