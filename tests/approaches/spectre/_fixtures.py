"""Synthetic ``EpisodeRecord`` builder for tests that don't need a live env.

Gives us a tiny but real relational-structs toy domain so the schema, canonicalization,
vocab, priors, dataset, and collate layers can be exercised without calling kinder or
the bilevel planner.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import (
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)

from alphatamp.approaches.spectre.io import atomic_write_pickle_gz
from alphatamp.approaches.spectre.schema import (
    EpisodeRecord,
    OutcomeRecord,
    ProvenanceBlock,
    SkeletonRecord,
    SummaryBlock,
)

# ---- Toy domain ----------------------------------------------------------
BLOCK = Type("block")
ROBOT = Type("robot")

ON_TABLE = Predicate("OnTable", [BLOCK])
HOLDING = Predicate("Holding", [ROBOT, BLOCK])
CLEAR = Predicate("Clear", [BLOCK])

_r = Variable("?r", ROBOT)
_b = Variable("?b", BLOCK)

PICK = LiftedOperator(
    name="Pick",
    parameters=[_r, _b],
    preconditions={ON_TABLE([_b]), CLEAR([_b])},
    add_effects={HOLDING([_r, _b])},
    delete_effects={ON_TABLE([_b])},
)
PLACE = LiftedOperator(
    name="Place",
    parameters=[_r, _b],
    preconditions={HOLDING([_r, _b])},
    add_effects={ON_TABLE([_b]), CLEAR([_b])},
    delete_effects={HOLDING([_r, _b])},
)


def _state(atoms: set, objs: set) -> RelationalAbstractState:
    return RelationalAbstractState(atoms=atoms, objects=objs)


def _make_provenance(
    problem_id: int, config_hash: str = "deadbeef0000"
) -> ProvenanceBlock:
    return ProvenanceBlock(
        problem_id=problem_id,
        env_id="test/Toy-v0",
        env_variant="toy",
        split="train",
        config_hash=config_hash,
        problem_seed=problem_id,
        git_sha="test",
        collection_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package_versions={"bilevel_planning": "test"},
    )


def build_toy_episode(
    problem_id: int = 0,
    num_blocks: int | None = None,
    outcomes: tuple[str, ...] = ("fail", "fail", "success"),
) -> EpisodeRecord:
    """Build an ``EpisodeRecord`` whose pool has one skeleton per block.

    Skeleton ``i`` is ``Pick(robot_0, block_i) → Place(robot_0, block_i)``.
    ``outcomes`` is parallel to the pool; its length equals ``num_blocks``.
    If ``num_blocks`` is ``None``, it defaults to ``len(outcomes)``.
    """
    if num_blocks is None:
        num_blocks = len(outcomes)
    assert len(outcomes) == num_blocks
    robot = Object("robot_0", ROBOT)
    blocks = [Object(f"block_{i}", BLOCK) for i in range(num_blocks)]

    s0_atoms: set = {ON_TABLE([b]) for b in blocks} | {CLEAR([b]) for b in blocks}
    s0 = _state(s0_atoms, {robot, *blocks})
    # Goal uses a persistent-state predicate (``Clear``), not the transient
    # ``Holding``. This mirrors the real cluttered-storage goal (``OnShelf``,
    # not ``Holding``) — so ``Holding`` lives purely in intermediate states
    # and exercises the trajectory-reconstruction path in vocab extraction.
    goal_atoms = frozenset({CLEAR([blocks[0]])})

    skels: list[SkeletonRecord] = []
    outs: list[OutcomeRecord] = []
    for i, outcome in enumerate(outcomes):
        pick = PICK.ground((robot, blocks[i]))
        place = PLACE.ground((robot, blocks[i]))
        # Pick → Place on the same block is a STRIPS-null cycle: after Pick
        # we have HOLDING(robot, b_i) and lose OnTable(b_i); after Place we
        # restore OnTable(b_i), Clear(b_i) and drop HOLDING. Net effect on
        # atoms is zero. We store the true progression so the fixture mirrors
        # real planner output (where HOLDING never appears in s_L either).
        final = _state(set(s0_atoms), {robot, *blocks})
        skels.append(
            SkeletonRecord(
                skeleton_idx=i,
                operator_seq=(pick, place),
                final_abstract_state=final,
            )
        )
        outs.append(
            OutcomeRecord(
                skeleton_idx=i,
                outcome=outcome,  # type: ignore[arg-type]
                refinement_wall_clock_s=0.1 * (i + 1),
                refinement_seed=1000 + i,
            )
        )
    first_succ = next((j for j, o in enumerate(outs) if o.outcome == "success"), None)
    summary = SummaryBlock(
        num_skeletons=num_blocks,
        num_success=sum(1 for o in outs if o.outcome == "success"),
        num_fail=sum(1 for o in outs if o.outcome == "fail"),
        num_error=sum(1 for o in outs if o.outcome == "error"),
        first_success_idx=first_succ,
        total_wall_clock_s=sum(o.refinement_wall_clock_s for o in outs),
        pool_truncated=False,
    )
    registry = {obj.name: obj.type.name for obj in {robot, *blocks}}
    return EpisodeRecord(
        provenance=_make_provenance(problem_id),
        initial_abstract_state=s0,
        goal_atoms=goal_atoms,
        object_registry=registry,
        skeleton_pool=tuple(skels),
        outcomes=tuple(outs),
        summary=summary,
    )


def write_toy_split(
    split_dir: Path,
    outcomes_per_problem: list[tuple[str, ...]],
) -> list[Path]:
    """Write one toy episode per entry in ``outcomes_per_problem``."""
    paths: list[Path] = []
    for pid, outcomes in enumerate(outcomes_per_problem):
        ep = build_toy_episode(
            problem_id=pid, num_blocks=len(outcomes), outcomes=outcomes
        )
        path = split_dir / "episodes" / f"ep_{pid:05d}.pkl.gz"
        atomic_write_pickle_gz(ep, path)
        paths.append(path)
    return paths
