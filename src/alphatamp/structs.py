"""Common data structures."""

from typing import TypeAlias

from bilevel_planning.structs import (
    RelationalAbstractState,
)
from relational_structs import GroundOperator

# We use the term Skeleton to refer specifically to abstract plans over relational
# abstract states and ground operator actions.
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]
