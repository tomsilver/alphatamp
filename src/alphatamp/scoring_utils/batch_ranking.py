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

    def _scores_discriminate(self, batch: list[Skeleton], prev: list[Skeleton]) -> bool:
        """Check if scoring function provides discriminatory information.

        Returns False if all scores are nearly identical (within epsilon).
        """
        if len(batch) <= 1:
            return False  # No point in reordering a single skeleton

        scores = [self._score_fn(skeleton, prev) for skeleton in batch]
        score_range = max(scores) - min(scores)

        # If scores vary by less than this threshold, don't reorder
        DISCRIMINATORY_THRESHOLD = 0.01

        return score_range > DISCRIMINATORY_THRESHOLD

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
                # Only reorder if scoring function is discriminatory
                # Otherwise, preserve base generator's order
                if self._scores_discriminate(batch, prev):
                    tiebreaking_score_fn = lambda x: (
                        self._score_fn(x, prev),
                        -len(x[1]),
                        self._rng.uniform(),
                    )
                    batch.sort(key=tiebreaking_score_fn)
                    skeleton = batch.pop()
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
