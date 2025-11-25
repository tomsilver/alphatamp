"""Generates batches of abstract plans and then ranks them using a score function, where
higher scores are considered better."""

from itertools import islice
from typing import Callable, Iterator, TypeAlias, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Goal,
    RelationalAbstractState,
)
from relational_structs import GroundOperator

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]


class BatchRankingAbstractPlanGenerator(
    AbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """Generates batches of abstract plans and then ranks them using a score function,
    where higher scores are considered better."""

    def __init__(
        self,
        base_generator: AbstractPlanGenerator[
            _X, RelationalAbstractState, GroundOperator
        ],
        score_fn: Callable[[Skeleton, list[Skeleton]], float],
        batch_size: int,
        seed: int,
    ) -> None:
        self._base_generator = base_generator
        self._score_fn = score_fn
        self._batch_size = batch_size
        # In the future, make this public or find another workaround.
        abstract_successor_fn = (
            self._base_generator._abstract_successor_function  # pylint: disable=protected-access
        )
        super().__init__(abstract_successor_fn, seed)

    def __call__(
        self,
        x0: _X,
        s0: RelationalAbstractState,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, RelationalAbstractState, GroundOperator],
    ) -> Iterator[Skeleton]:

        # This should be refactored soon.
        iterator = self._base_generator(x0, s0, goal, timeout, bpg)
        prev: list[Skeleton] = []
        while batch := list(islice(iterator, self._batch_size)):
            # NOTE: we need to reorder after every failed attempt because of prev.
            while batch:
                # Optimization: Compute scores once per iteration
                # We zip with index to track original position
                scored_candidates = []
                scores = []
                for i, skel in enumerate(batch):
                    score = self._score_fn(skel, prev)
                    scores.append(score)
                    scored_candidates.append((i, skel, score))

                score_range = max(scores) - min(scores) if scores else 0.0
                DISCRIMINATORY_THRESHOLD = 0.01

                # Only reorder if scoring function is discriminatory
                if len(batch) > 1 and score_range > DISCRIMINATORY_THRESHOLD:
                    # Find the best skeleton without sorting the whole list in-place.
                    # We want to maximize: (score, -len(plan), random)
                    # This matches the logic of batch.sort() + batch.pop()

                    def priority_fn(item):
                        _, skel, score = item
                        return (score, -len(skel[1]), self._rng.uniform())

                    # Find index of the best candidate
                    best_idx, _, _ = max(scored_candidates, key=priority_fn)

                    # Remove and yield the best one
                    skeleton = batch.pop(best_idx)
                else:
                    # Use base generator's order (first in batch)
                    skeleton = batch.pop(0)

                # Uncomment to debug.
                # print("YIELDING")
                # for a in skeleton[1]:
                #     print(a.short_str)
                # print()

                yield skeleton

                # NOTE: assuming that every previous skeleton failed.
                # This works with SesamePlanner, but may not be true in general.
                # SesamePlanner generally stops after the first success.
                prev.append(skeleton)
