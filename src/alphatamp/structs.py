"""Common data structures."""

from typing import Callable, TypeAlias, Union

import numpy as np
from bilevel_planning.structs import (
    RelationalAbstractState,
)
from numpy.typing import NDArray
from relational_structs.pddl import GroundOperator

# We use the term Skeleton to refer specifically to abstract plans over relational
# abstract states and ground operator actions.
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]


MaxTrainIters = Union[int, Callable[[int], int]]
Array = NDArray[np.float32]
