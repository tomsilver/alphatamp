"""Tests for ``spectre.trajectory``: STRIPS progression over a skeleton."""

from __future__ import annotations

import pytest
from _fixtures import (
    CLEAR,
    ON_TABLE,
    PICK,
    PLACE,
    build_toy_episode,
)
from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundOperator, Object

from alphatamp.approaches.spectre.trajectory import (
    apply_operator,
    reconstruct_trajectory,
)

# Shared across tests: one robot, one block, and the canonical s_0 where the
# block is on the table and clear.
_ROBOT = Object("robot_0", PICK.parameters[0].type)
_BLOCK = Object("block_0", PICK.parameters[1].type)
_S0_ATOMS = {ON_TABLE([_BLOCK]), CLEAR([_BLOCK])}
_S0 = RelationalAbstractState(atoms=set(_S0_ATOMS), objects={_ROBOT, _BLOCK})


def _pick_op() -> GroundOperator:
    return PICK.ground((_ROBOT, _BLOCK))


def _place_op() -> GroundOperator:
    return PLACE.ground((_ROBOT, _BLOCK))


# ---------------------------------------------------------------------------
# apply_operator
# ---------------------------------------------------------------------------


def test_apply_operator_pick_adds_holding_removes_ontable() -> None:
    """After ``Pick``, HOLDING is added and OnTable is removed."""
    s1 = apply_operator(_S0, _pick_op())
    atom_strs = {str(a) for a in s1.atoms}
    assert any("Holding" in s for s in atom_strs)
    assert not any("OnTable" in s for s in atom_strs)
    # Objects set is inherited unchanged (STRIPS operators do not introduce
    # new objects).
    assert s1.objects == _S0.objects


def test_apply_operator_does_not_verify_preconditions() -> None:
    """``apply_operator`` trusts its caller.

    Passing ``Place`` to an ``s_0`` where nothing is held still returns the
    STRIPS successor (add-effects added, delete-effects removed); the state
    is merely meaningless.
    """
    s_out = apply_operator(_S0, _place_op())
    assert s_out.objects == _S0.objects  # no crash


# ---------------------------------------------------------------------------
# reconstruct_trajectory
# ---------------------------------------------------------------------------


def test_reconstruct_trajectory_length() -> None:
    """Length == len(op_seq) + 1, and the first element is ``s_0``."""
    traj = reconstruct_trajectory(_S0, (_pick_op(), _place_op()))
    assert len(traj) == 3
    assert traj[0] is _S0


def test_reconstruct_trajectory_pick_place_cycles_back_to_s0() -> None:
    """Pick → Place on the same block returns to ``s_0`` (net-null STRIPS cycle)."""
    traj = reconstruct_trajectory(_S0, (_pick_op(), _place_op()))
    # s_2 restores OnTable and Clear, drops Holding → equals s_0 in atoms.
    assert traj[2].atoms == _S0.atoms


def test_reconstruct_trajectory_exposes_holding_in_middle() -> None:
    """``Holding`` only appears between Pick and Place; trajectory surfaces it."""
    traj = reconstruct_trajectory(_S0, (_pick_op(), _place_op()))
    mid_names = {a.predicate.name for a in traj[1].atoms}
    assert "Holding" in mid_names
    # And it's absent in s_0 and s_L.
    assert "Holding" not in {a.predicate.name for a in traj[0].atoms}
    assert "Holding" not in {a.predicate.name for a in traj[2].atoms}


def test_reconstruct_trajectory_empty_op_seq_returns_only_s0() -> None:
    """With no operators, the trajectory is exactly ``[s_0]``."""
    traj = reconstruct_trajectory(_S0, ())
    assert traj == [_S0]


def test_reconstruct_trajectory_raises_on_precondition_violation() -> None:
    """Placing from ``s_0`` (nothing held) violates ``Place``'s preconditions."""
    with pytest.raises(AssertionError, match="Place.*preconditions not satisfied"):
        reconstruct_trajectory(_S0, (_place_op(),))


def test_reconstruct_trajectory_can_skip_precondition_check() -> None:
    """``verify_preconditions=False`` returns the STRIPS-successor verbatim."""
    traj = reconstruct_trajectory(_S0, (_place_op(),), verify_preconditions=False)
    assert len(traj) == 2


# ---------------------------------------------------------------------------
# Consistency with stored final_abstract_state on the toy fixture.
# ---------------------------------------------------------------------------


def test_reconstructed_final_matches_stored_final_on_toy_fixture() -> None:
    """The fixture is STRIPS-consistent: trajectory[-1] == stored final."""
    ep = build_toy_episode(outcomes=("fail", "fail", "success"))
    for skel in ep.skeleton_pool:
        traj = reconstruct_trajectory(ep.initial_abstract_state, skel.operator_seq)
        assert traj[-1].atoms == skel.final_abstract_state.atoms, (
            f"skeleton {skel.skeleton_idx}: STRIPS trajectory final diverges"
            f" from stored final"
        )
