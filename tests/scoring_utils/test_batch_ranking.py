"""Tests for batch_ranking.py."""

from typing import TypeAlias

import prbench
from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.structs import (
    RelationalAbstractState,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalControllerGenerator,
)
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from relational_structs import GroundOperator

from alphatamp.scoring_utils.batch_ranking import BatchRankingAbstractPlanGenerator

Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]


def _score_skeleton(skeleton: Skeleton, _: list[Skeleton]) -> float:
    """Naive scoring function: longer length is better"""

    return len(skeleton[0])


def test_batch_ranking() -> None:
    """Tests for BatchRankingAbstractPlanGenerator()."""

    # Test in a PRBench environment where the first skeleton won't work.
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o1-v0")
    env_models = create_bilevel_planning_models(
        "obstruction2d", env.observation_space, env.action_space, num_obstructions=1
    )

    # Set parameter values
    max_skill_horizon = 100
    heuristic_name = "hff"
    skeleton_batch_size = 100
    seed = 123

    ## Create the planning components.

    # Create the sampler.
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(env_models.skills),
        transition_function=env_models.transition_fn,
        state_abstractor=env_models.state_abstractor,
        max_trajectory_steps=max_skill_horizon,
    )

    assert isinstance(trajectory_sampler, ParameterizedControllerTrajectorySampler)

    # Create the abstract plan generator.
    base_abstract_plan_generator: AbstractPlanGenerator = (
        RelationalHeuristicSearchAbstractPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            heuristic_name,
            seed=seed,
        )
    )
    batched_abstract_plan_generator: AbstractPlanGenerator = (
        BatchRankingAbstractPlanGenerator(
            base_abstract_plan_generator,
            score_fn=_score_skeleton,
            batch_size=skeleton_batch_size,
            seed=seed,
        )
    )

    assert isinstance(batched_abstract_plan_generator, AbstractPlanGenerator)

    env.close()  # type: ignore[no-untyped-call]
