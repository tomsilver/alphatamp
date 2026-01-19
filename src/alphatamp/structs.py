"""Common data structures."""

from typing import TypeAlias, Union, Callable
import numpy as np
from numpy.typing import NDArray

from bilevel_planning.structs import (
    RelationalAbstractState,
)
from relational_structs.pddl import GroundOperator, GroundAtom
from relational_structs.objects import Object
from relational_structs.object_centric_state import ObjectCentricState

# We use the term Skeleton to refer specifically to abstract plans over relational
# abstract states and ground operator actions.
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]


MaxTrainIters = Union[int, Callable[[int], int]]
Array = NDArray[np.float32]
